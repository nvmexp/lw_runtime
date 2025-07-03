// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <FrontEnd/Canonical/CanonicalProgram.h>

#include <Context/Context.h>
#include <Context/LLVMManager.h>
#include <Context/ProgramManager.h>
#include <Context/UpdateManager.h>
#include <Device/LWDADevice.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/exceptions/ValidationError.h>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <iomanip>

using namespace optix;
using namespace prodlib;

static std::string hexString( size_t u, unsigned int width = 8 )
{
    std::stringstream ss;
    ss << "0x" << std::setw( width ) << std::setfill( '0' ) << std::hex << u;
    return ss.str();
}

/**
 * Constructors/destructors
 */

CanonicalProgram::CanonicalProgram( Context* context, size_t ptxHash )
    : m_context( context )
    , m_ptxHash( ptxHash )
{
}

CanonicalProgram::CanonicalProgram( const std::string&      inputFunctionName,
                                    lwca::ComputeCapability targetMin,
                                    lwca::ComputeCapability targetMax,
                                    size_t                  ptxHash,
                                    Context*                context )
    : m_context( context )
    , m_targetMin( targetMin )
    , m_targetMax( targetMax )
    , m_ptxHash( ptxHash )
{
    RT_ASSERT_MSG( targetMin < targetMax, "Invalid compute target range" );
    if( context )
        m_id = context->getProgramManager()->createCanonicalProgramId( this );
    else
        // Use a fake ID in the testing environment only
        m_id.reset( new CanonicalProgramID( 0 ) );
    m_inputFunctionName       = inputFunctionName;
    m_universallyUniqueName   = inputFunctionName + "_ptx" + hexString( m_ptxHash, 16 );
    m_universallyUniqueNumber = std::hash<std::string>()( inputFunctionName ) ^ m_ptxHash;
}

CanonicalProgram::~CanonicalProgram() NOEXCEPT_FALSE
{
    for( const VariableReference* reference : m_attributeReferences )
        delete reference;
    for( const VariableReference* reference : m_variableReferences )
        delete reference;
    for( const CallSiteIdentifier* csID : m_ownedCallSites )
        delete csID;

    // The canonical program owns the entire module(s)
    if( m_function )
        delete m_function->getParent();
    if( m_intersectionFunction )
        delete m_intersectionFunction->getParent();
    if( m_attributeDecoder )
        delete m_attributeDecoder->getParent();

    RT_ASSERT_MSG( m_usedAsSemanticType.empty(), "Semantic type uses are not empty at destroy" );
    RT_ASSERT_MSG( m_usedByRayType.empty(), "Used by ray types are not empty at destroy" );
    RT_ASSERT_MSG( m_usedOnDevice.empty(), "Device utilization set is not empty at destroy" );
}

void CanonicalProgram::finalize( llvm::Function* function )
{
    assert( function->getName().str() == m_universallyUniqueName );
    m_function = function;
}

/**
 * Metadata
 */
const std::string& CanonicalProgram::getInputFunctionName() const
{
    return m_inputFunctionName;
}

const std::string& CanonicalProgram::getUniversallyUniqueName() const
{
    return m_universallyUniqueName;
}

unsigned CanonicalProgram::getFunctionSignature() const
{
    return m_signatureId;
}

size_t CanonicalProgram::getPTXHash() const
{
    return m_ptxHash;
}

int CanonicalProgram::get32bitAttributeKind() const
{
    return ( m_universallyUniqueNumber >> 32 ) ^ (unsigned int)( m_universallyUniqueNumber );
}


/**
 * Target architecture
 */
lwca::ComputeCapability CanonicalProgram::getTargetMin() const
{
    return m_targetMin;
}

lwca::ComputeCapability CanonicalProgram::getTargetMax() const
{
    return m_targetMax;
}

bool CanonicalProgram::isValidForDevice( const Device* device ) const
{
    lwca::ComputeCapability smversion( 999 );  // CPU device is equivalent to max SM version
    if( const LWDADevice* cdevice = dynamic_cast<const LWDADevice*>( device ) )
        smversion = cdevice->computeCapability();
    return m_targetMin <= smversion && smversion <= m_targetMax;
}

/**
 * Module information
 */
static llvm::Function* rematerializeFunction( llvm::LLVMContext& llvmContext, std::vector<char>& bitcode, const std::string& name )
{
    std::unique_ptr<llvm::MemoryBuffer> buffer =
        llvm::MemoryBuffer::getMemBuffer( llvm::StringRef( bitcode.data(), bitcode.size() ), "cachedcp", false );
    llvm::Expected<std::unique_ptr<llvm::Module>> moduleOrError = llvm::parseBitcodeFile( buffer->getMemBufferRef(), llvmContext );
    if( llvm::Error error = moduleOrError.takeError() )
    {
        std::string errorMessage;
        llvm::raw_string_ostream errorStream( errorMessage );
        errorStream << error;
        throw CompileError( RT_EXCEPTION_INFO, "Error parsing cached bitcode with error: " + errorStream.str() );
    }

    llvm::Module* module = moduleOrError.get().release();
    llvm::Function* function = module->getFunction( name );
    if( !function )
        throw CompileError( RT_EXCEPTION_INFO, "Could not find function: '" + name + "' in cached module" );

    // Free the vector
    std::vector<char>().swap( bitcode );
    return function;
}

const llvm::Function* CanonicalProgram::llvmFunction() const
{
    if( !m_function )
    {
        RT_ASSERT( !m_lazyLoadBitcode.empty() );
        m_function = rematerializeFunction( m_context->getLLVMManager()->llvmContext(), m_lazyLoadBitcode,
                                            getUniversallyUniqueName() );
    }
    return m_function;
}

const llvm::Function* CanonicalProgram::llvmIntersectionFunction() const
{
    if( !m_intersectionFunction )
    {
        RT_ASSERT( !m_lazyLoadIntersectionBitcode.empty() );
        m_intersectionFunction = rematerializeFunction( m_context->getLLVMManager()->llvmContext(),
                                                        m_lazyLoadIntersectionBitcode, getUniversallyUniqueName() );
    }
    return m_intersectionFunction;
}

const llvm::Function* CanonicalProgram::llvmAttributeDecoder() const
{
    if( !m_attributeDecoder )
    {
        RT_ASSERT( !m_lazyLoadAttributeBitcode.empty() );
        m_attributeDecoder = rematerializeFunction( m_context->getLLVMManager()->llvmContext(), m_lazyLoadAttributeBitcode,
                                                    "__decode_attributes." + getUniversallyUniqueName() );
    }
    return m_attributeDecoder;
}


CanonicalProgramID CanonicalProgram::getID() const
{
    RT_ASSERT( m_id != nullptr );
    return *m_id;
}

Context* CanonicalProgram::getContext() const
{
    return m_context;
}

/**
 * Graph properties
 */

void CanonicalProgram::receivePropertyDidChange_UsedAsSemanticType( SemanticType stype, bool added ) const
{
    bool changed = m_usedAsSemanticType.addOrRemoveProperty( stype, added );
    if( changed && m_context )
        m_context->getUpdateManager()->eventCanonicalProgramSemanticTypeDidChange( this, stype, added );
    // validation of the SemanticType should happen before this function is called.
}

void CanonicalProgram::receivePropertyDidChange_InheritedSemanticType( SemanticType stype, bool added ) const
{
    bool changed = m_inheritedSemanticType.addOrRemoveProperty( stype, added );
    if( changed && m_context )
    {
        // Only bound callable programs have inherited semantic types. During context destruction
        // the m_usedAsSemanticType property for ST_BOUND_CALLABLE_PROGRAM may already have been cleared
        // so we cannot assert that here.
        m_context->getUpdateManager()->eventCanonicalProgramInheritedSemanticTypeDidChange( this, ST_BOUND_CALLABLE_PROGRAM,
                                                                                            stype, added );
    }
}

void CanonicalProgram::receivePropertyDidChange_UsedByRayType( unsigned int rayType, bool added ) const
{
    bool changed = m_usedByRayType.addOrRemoveProperty( rayType, added );
    if( changed && m_context )
        m_context->getUpdateManager()->eventCanonicalProgramUsedByRayTypeDidChange( this, rayType, added );
}

void CanonicalProgram::receivePropertyDidChange_UsedOnDevice( const Device* device, bool added ) const
{
    m_usedOnDevice.addOrRemoveProperty( device->allDeviceListIndex(), added );
    // We are consciously not generating an event for this.  The only time a plan would need
    // to know about this is if you chose one CanonicalProgram over another, and this only
    // oclwrs for our internal programs.  Since we will not be changing these Programs over
    // the life of the Context, the situation where we would miss adding a program is almost
    // impossible.  I'm going to forgo adding all the code to support this unlikely case in
    // favor of writing this long comment.
}

void CanonicalProgram::receivePropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    bool changed = m_directCaller.addOrRemoveProperty( cpid, added );
    if( changed && m_context )
        m_context->getUpdateManager()->eventCanonicalProgramDirectCallerDidChange( this, cpid, added );
}

void CanonicalProgram::receivePropertyDidChange_calledFromCallsite( CallSiteIdentifier* csId, bool added ) const
{
    m_calledFromCallsites.addOrRemoveProperty( csId, added );
}

const std::vector<const CallSiteIdentifier*>& CanonicalProgram::getCallSites() const
{
    return m_ownedCallSites;
}

bool CanonicalProgram::isUsedAsSemanticType( SemanticType stype ) const
{
    return m_usedAsSemanticType.contains( stype );
}

bool CanonicalProgram::isUsedAsSemanticTypes( const std::vector<SemanticType>& stypes ) const
{
    for( auto stype : stypes )
        if( isUsedAsSemanticType( stype ) )
            return true;
    return false;
}

bool CanonicalProgram::isUsedAsInheritedSemanticType( SemanticType stype ) const
{
    return m_inheritedSemanticType.contains( stype );
}

bool CanonicalProgram::isUsedAsInheritedSemanticTypes( const std::vector<SemanticType>& stypes ) const
{
    for( auto stype : stypes )
        if( isUsedAsInheritedSemanticType( stype ) )
            return true;
    return false;
}

bool CanonicalProgram::isUsedAsSingleSemanticType() const
{
    return m_usedAsSemanticType.size() == 1;
}

SemanticType CanonicalProgram::getSingleSemanticType() const
{
    RT_ASSERT_MSG( isUsedAsSingleSemanticType(), "Multiple semantic types are used in CanonicalProgram" );
    return *m_usedAsSemanticType.begin();
}

void CanonicalProgram::getAllUsedSemanticTypes( std::vector<SemanticType>& used_stypes ) const
{
    for( auto stype : m_usedAsSemanticType )
        used_stypes.push_back( stype );
}

bool CanonicalProgram::isUsedByRayTypes( const GraphProperty<unsigned int, false>& rayTypes ) const
{
    // RayType ~0 indiciates that it can produce any raytype
    if( rayTypes.contains( ~0 ) )
        return true;
    else
        return m_usedByRayType.intersects( rayTypes );
}

bool CanonicalProgram::tracesUnknownRayType() const
{
    return m_producesRayTypes.contains( ~0 );
}

void CanonicalProgram::markTracesUnknownRayType()
{
    // RayType ~0 indiciates that it can produce any raytype
    if( !m_producesRayTypes.contains( ~0 ) )
        m_producesRayTypes.addOrRemoveProperty( ~0, true );
}

const GraphProperty<unsigned int, false>& CanonicalProgram::producesRayTypes() const
{
    return m_producesRayTypes;
}

bool CanonicalProgram::hasDirectCaller( const std::set<const CanonicalProgram*, IDCompare>& callers ) const
{
    for( auto candidate : callers )
        if( m_directCaller.contains( candidate->getID() ) )
            return true;

    return false;
}

bool CanonicalProgram::isUsedOnDevice( const Device* device ) const
{
    return m_usedOnDevice.contains( device->allDeviceListIndex() );
}

void CanonicalProgram::validateSemanticType( SemanticType stype ) const
{
    if( m_context->useRtxDataModel() )
    {
        if( stype == optix::ST_BOUND_CALLABLE_PROGRAM )
        {
            // Some calls are always illegal in bound callable programs
            // when using RTX.
            if( m_callsTerminateRay )
                throw ValidationError( RT_EXCEPTION_INFO, "rtTerminateRay is not allowed from bound callable program "
                                                              + getInputFunctionName() );
            if( m_hasAttributeStores )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "Writing attributes is not allowed from bound callable program " + getInputFunctionName() );
            if( m_hasAttributeLoads )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "Reading attributes is not allowed from bound callable program " + getInputFunctionName() );
            if( m_hasLwrrentRayAccess )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "Using the semantic variable rtLwrrentRay is not allowed from bound callable "
                                       "program "
                                           + getInputFunctionName() );
            if( m_callsPotentialIntersection )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "rtPotentialIntersection is not allowed from bound callable program " + getInputFunctionName() );
            if( m_callsReportIntersection )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "rtReportIntersection is not allowed from bound callable program " + getInputFunctionName() );
            if( m_callsIgnoreIntersection )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "rtIgnoreIntersection is not allowed from bound callable program " + getInputFunctionName() );
            if( m_callsIntersectChild )
                throw ValidationError( RT_EXCEPTION_INFO, "rtIntersectChild is not allowed from bound callable program "
                                                              + getInputFunctionName() );
            if( m_callsGetPrimitiveIndex )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "rtGetPrimitiveIndex is not allowed from bound callable program " + getInputFunctionName() );
            if( m_accessesHitKind )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "rtIsTriangleHit, rtIsTriangleHitBackFace and rtIsTriangleHitFrontFace are not "
                                       "allowed from bound callable program "
                                           + getInputFunctionName() );
            if( m_callsGetInstanceFlags )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "rtGetInstanceFlags is not allowed from bound callable program " + getInputFunctionName() );
            if( m_callsGetRayFlags )
                throw ValidationError( RT_EXCEPTION_INFO, "rtGetRayFlags is not allowed from bound callable program "
                                                              + getInputFunctionName() );
            if( m_callsGetRayMask )
                throw ValidationError( RT_EXCEPTION_INFO, "rtGetRayMask is not allowed from bound callable program "
                                                              + getInputFunctionName() );
            if( m_callsGetLowestGroupChildIndex )
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "m_callsGetLowestGroupChildIndex is not allowed from bound callable program "
                                           + getInputFunctionName() );

            // The inherited semantic types of bound callable programs will narrow the
            // set of other legal calls.
            return;
        }
    }
    else if( stype == optix::ST_BOUND_CALLABLE_PROGRAM )
    {
        // The inherited semantic types of bound callable programs will narrow the
        // set of legal calls.
        return;
    }

    if( m_callsTransform && !isTransformCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtTransform* call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsGetPrimitiveIndex && !isGetPrimitiveIndexLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtGetPrimitiveIndex call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsGetInstanceFlags && !isGetInstanceFlagsLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtGetInstanceFlags call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsGetRayFlags && !isLwrrentRayAccessLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtGetRayFlags call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsGetRayMask && !isLwrrentRayAccessLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtGetRayMask call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsGetLowestGroupChildIndex && !isGetLowestGroupChildIndexCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtGetLowestGroupChildIndexCallLegal call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_accessesHitKind && !isGetHitKindLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO,
                               "found illegal call to rtIsTriangleHit, rtIsTriangleHitBackFace or "
                               "rtIsTriangleHitFrontFace call in "
                                   + getInputFunctionName() + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsTrace && !isTraceCallLegal( stype, m_context->useRtxDataModel() ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtTrace call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsThrow && !isThrowCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtThrow call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsTerminateRay && !isTerminateRayCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtTerminateRay call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsIgnoreIntersection && !isIgnoreIntersectionCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtIgnoreIntersection call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsIntersectChild && !isIntersectChildCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtIntersectChild call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsPotentialIntersection && !isPotentialIntersectionCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtPotentialIntersection call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsReportIntersection && !isReportIntersectionCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtReportIntersection call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsExceptionCode && !isExceptionCodeCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "rtExceptionCode call is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsBindlessCallableProgram && !isBindlessCallableCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "Call of bindless callable program is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );
    if( m_callsBoundCallableProgram && !isBoundCallableCallLegal( stype ) )
        throw ValidationError( RT_EXCEPTION_INFO, "Call of bound callable program is not allowed in " + getInputFunctionName()
                                                      + " function with semantic type " + semanticTypeToString( stype ) );

    if( ( m_hasAttributeStores || m_hasAttributeLoads ) && !isAttributeAccessLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO,
                               getInputFunctionName() + " function with semantic type " + semanticTypeToString( stype )
                                 + " reads or writes attributes. Attribute accesses are only allowed in any hit, closest hit and intersection programs. ");
    }

    if( m_hasAttributeStores && !isAttributeWriteLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO,
                               getInputFunctionName() + " function with semantic type " + semanticTypeToString( stype )
                                   + " writes attributes. Attributes can be written only by intersection programs." );
    }

    if( m_hasPayloadAccesses && !isPayloadAccessLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, getInputFunctionName() + " function with semantic type "
                                                      + semanticTypeToString( stype )
                                                      + " accesses the rtPayload semantic variable." );
    }
    if( m_hasPayloadStores && !isPayloadStoreLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, getInputFunctionName() + " function with semantic type "
                                                      + semanticTypeToString( stype )
                                                      + " stores a value in the rtPayload semantic variable." );
    }
    if( m_hasLwrrentRayAccess && !isLwrrentRayAccessLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, getInputFunctionName() + " function with semantic type "
                                                      + semanticTypeToString( stype )
                                                      + " accesses the rtLwrrentRay semantic variable." );
    }
    if( m_hasLwrrentTimeAccess && !isLwrrentTimeAccessLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, getInputFunctionName() + " function with semantic type "
                                                      + semanticTypeToString( stype )
                                                      + " accesses the rtLwrrentTime semantic variable." );
    }
    if( m_accessesIntersectionDistance && !isIntersectionDistanceAccessLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, getInputFunctionName() + " function with semantic type "
                                                      + semanticTypeToString( stype )
                                                      + " accesses the rtIntersectionDistance semantic variable." );
    }

    if( m_hasBufferStores && !isBufferStoreLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, getInputFunctionName() + " function with semantic type "
                                                      + semanticTypeToString( stype )
                                                      + " attempts to store a value in a buffer." );
    }

    if( ( m_globalPointerMayEscape || m_globalConstPointerMayEscape || m_payloadPointerMayEscape || m_bindlessBufferPointerMayEscape )
        && !isPointerEscapeLegal( stype ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, getInputFunctionName() + " function with semantic type "
                                                      + semanticTypeToString( stype )
                                                      + " has a potential pointer escape." );
    }
}


/**
 * Representation of variable and attribute references.
 */

const CanonicalProgram::VariableReferenceListType& CanonicalProgram::getAttributeReferences() const
{
    return m_attributeReferences;
}

const CanonicalProgram::VariableReferenceListType& CanonicalProgram::getVariableReferences() const
{
    return m_variableReferences;
}


/**
 * SLOW search for variable of a given name - used only for testing
 */
const VariableReference* CanonicalProgram::findAttributeReference( const std::string& name ) const
{
    for( const VariableReference* reference : m_attributeReferences )
    {
        if( reference->getInputName() == name )
            return reference;
    }
    return nullptr;
}

const VariableReference* CanonicalProgram::findVariableReference( const std::string& name ) const
{
    for( const VariableReference* reference : m_variableReferences )
    {
        if( reference->getInputName() == name )
            return reference;
    }
    return nullptr;
}


const VariableReference* CanonicalProgram::findVariableReference( VariableReferenceID id ) const
{
    for( const VariableReference* reference : m_variableReferences )
    {
        if( reference->getReferenceID() == id )
            return reference;
    }
    return nullptr;
}

std::vector<SemanticType> CanonicalProgram::getInheritedSemanticTypes() const
{
    std::vector<SemanticType> result;
    result.reserve( m_inheritedSemanticType.size() );
    for( SemanticType stype : m_inheritedSemanticType )
        result.push_back( stype );
    return result;
}

static void readOrWriteBitcode( PersistentStream* stream, const llvm::Function* function, std::vector<char>& bitcode )
{
    if( stream->reading() )
    {
        // Read bitcode. Note from SGP: bitcode will not always be needed in
        // the rtx code path. If reading it here becomes a bottleneck, it
        // will be possible to put the bitcode in a different cache entry.
        int size = -1;
        readOrWrite( stream, &size, "bitcodeSize" );
        if( stream->error() || size < 0 )
            return;
        bitcode.resize( size );
        stream->readOrWrite( bitcode.data(), size, "lazyLoadBitcode", PersistentStream::Opaque );
    }
    else if( stream->writing() )
    {
        if( function )
        {
            // Generate and then write the bitcode.
            llvm::SmallVector<char, 65536> buffer;
            llvm::raw_svector_ostream ostream( buffer );
            llvm::WriteBitcodeToFile( *function->getParent(), ostream );
            ostream.str();  // Flush stream to buffer
            RT_ASSERT_MSG( buffer.size() >= 4, "error writing bitcode to cache" );

            // Write bitcode to cache
            int size = (int)buffer.size();
            readOrWrite( stream, &size, "bitcodeSize" );
            stream->readOrWrite( &buffer[0], size, "lazyLoadBitcode", PersistentStream::Opaque );
        }
        else
        {
            // We only have the bitcode. Write it directly.
            int size = (int)bitcode.size();
            readOrWrite( stream, &size, "bitCodeSize" );
            stream->readOrWrite( bitcode.data(), size, "lazyLoadBitcode", PersistentStream::Opaque );
        }
    }
    else
    {
        // When hashing, do not consider the bitcode.
    }
}

void optix::readOrWrite( PersistentStream* stream, CanonicalProgram* cp, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "CanonicalProgram" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    if( stream->error() )
        return;

    // Read/write scalar member data
    readOrWrite( stream, &cp->m_inputFunctionName, "inputFunctionName" );
    readOrWrite( stream, &cp->m_universallyUniqueName, "universallyUniqueName" );
    // Note: ID is handled external to this function
    readOrWrite( stream, &cp->m_signatureId, "signatureId" );
    readOrWrite( stream, &cp->m_targetMin, "targetMin" );
    readOrWrite( stream, &cp->m_targetMax, "targetMax" );
    // Note: ptxHash is handled externally
    readOrWrite( stream, &cp->m_universallyUniqueNumber, "universallyUniqueNumber" );
    // Note: dynamic properties are not stored.
    readOrWrite( stream, &cp->m_producesUnknownRayType, "producesUnknownRayType" );
    readOrWrite( stream, &cp->m_producesRayTypes, "producesRayTypes" );

    readOrWrite( stream, &cp->m_callsTrace, "callsTrace" );                            // 1
    readOrWrite( stream, &cp->m_callsGetPrimitiveIndex, "callsGetPrimitiveIndex" );    // 2
    readOrWrite( stream, &cp->m_callsGetPrimitiveIndex, "callsGetInstanceFlags" );     // 3
    readOrWrite( stream, &cp->m_callsGetPrimitiveIndex, "callsGetRayFlags" );          // 4
    readOrWrite( stream, &cp->m_callsGetPrimitiveIndex, "accessesHitKind" );           // 5
    readOrWrite( stream, &cp->m_traceHasTime, "traceHasTime" );                        // 6
    readOrWrite( stream, &cp->m_callsThrow, "callsThrow" );                            // 7
    readOrWrite( stream, &cp->m_callsTerminateRay, "callsTerminateRay" );              // 8
    readOrWrite( stream, &cp->m_callsIgnoreIntersection, "callsIgnoreIntersection" );  // 9
    readOrWrite( stream, &cp->m_callsIntersectChild, "callsIntersectChild" );          // 10

    readOrWrite( stream, &cp->m_callsPotentialIntersection, "callsPotentialIntersection" );      // 11
    readOrWrite( stream, &cp->m_callsReportIntersection, "callsReportIntersection" );            // 12
    readOrWrite( stream, &cp->m_isBuiltInIntersection, "isBuiltInIntersection" );                // 13
    readOrWrite( stream, &cp->m_callsTransform, "callsTransform" );                              // 14
    readOrWrite( stream, &cp->m_callsExceptionCode, "callsExceptionCode" );                      // 15
    readOrWrite( stream, &cp->m_callsBoundCallableProgram, "callsBoundCallableProgram" );        // 16
    readOrWrite( stream, &cp->m_callsBindlessCallableProgram, "callsBindlessCallableProgram" );  // 17
    readOrWrite( stream, &cp->m_hasLwrrentRayAccess, "hasLwrrentRayAccess" );                    // 18
    readOrWrite( stream, &cp->m_hasLwrrentTimeAccess, "hasLwrrentTimeAccess" );                  // 19
    readOrWrite( stream, &cp->m_accessesIntersectionDistance, "accessesIntersectionDistance" );  // 20


    readOrWrite( stream, &cp->m_maxPayloadSize, "maxPayloadSize" );                                  // 21
    readOrWrite( stream, &cp->m_maxPayloadRegisterCount, "maxPayloadRegisterCount" );                // 22
    readOrWrite( stream, &cp->m_maxAttributeData32bitValues, "maxAttributeData32bitValues" );        // 23
    readOrWrite( stream, &cp->m_bindlessBufferPointerMayEscape, "bindlessBufferPointerMayEscape" );  // 24
    readOrWrite( stream, &cp->m_payloadPointerMayEscape, "payloadPointerMayEscape" );                // 25
    readOrWrite( stream, &cp->m_globalPointerMayEscape, "globalPointerMayEscape" );                  // 26
    readOrWrite( stream, &cp->m_globalConstPointerMayEscape, "globalConstPointerMayEscape" );        // 27
    readOrWrite( stream, &cp->m_hasPayloadStores, "hasPayloadStores" );                              // 28
    readOrWrite( stream, &cp->m_hasPayloadAccesses, "hasPayloadAccesses" );                          // 29
    readOrWrite( stream, &cp->m_hasDynamicPayloadAccesses, "hasDynamicPayloadAccesses" );            // 30

    readOrWrite( stream, &cp->m_hasBufferStores, "hasBufferStores" );        // 31
    readOrWrite( stream, &cp->m_hasAttributeStores, "hasAttributeStores" );  // 32
    readOrWrite( stream, &cp->m_hasAttributeLoads, "hasAttributeLoads" );    // 33
    readOrWrite( stream, &cp->m_hasGlobalStores, "hasGlobalStores" );        // 34
    readOrWrite( stream, &cp->m_hasMotionIndexArg, "hasMotionIndexArg" );    // 35

    readOrWrite( stream, &cp->m_callsGetLowestGroupChildIndex, "callsGetLowestGroupChildIndex" );  // 36
    readOrWrite( stream, &cp->m_callsGetRayMask, "callsGetRayMask" );                              // 37

    readOrWrite( stream, &cp->m_accessesLaunchIndex, "accessesLaunchIndex" );  // 38

    readOrWrite( stream, &cp->m_attributeReferences,
                 [cp]() -> VariableReference* { return new VariableReference( cp ); }, "attributeReferences" );
    readOrWrite( stream, &cp->m_variableReferences,
                 [cp]() -> VariableReference* { return new VariableReference( cp ); }, "variableReferences" );

    readOrWrite( stream, &cp->m_ownedCallSites, [cp]() -> CallSiteIdentifier* { return new CallSiteIdentifier( cp ); },
                 "callSiteIdentifiers" );

    // Up to three different kinds of bitcode
    readOrWriteBitcode( stream, cp->m_function, cp->m_lazyLoadBitcode );
    readOrWriteBitcode( stream, cp->m_intersectionFunction, cp->m_lazyLoadIntersectionBitcode );
    readOrWriteBitcode( stream, cp->m_attributeDecoder, cp->m_lazyLoadAttributeBitcode );

    // Check version again since this is a big / complex object.
    stream->readOrWriteObjectVersion( version );
}
