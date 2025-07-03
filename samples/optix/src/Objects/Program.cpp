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

#include <Objects/Program.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/SBTManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Context/ValidationManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Exceptions/FileNotFound.h>
#include <Exceptions/TypeMismatch.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/Mangle.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/Geometry.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GlobalScope.h>
#include <Objects/GraphNode.h>
#include <Objects/Material.h>
#include <Objects/Variable.h>
#include <Util/LinkedPtrHelpers.h>
#include <Util/Misc.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidSource.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <exp/context/EncryptionManager.h>

#include <llvm/IR/Function.h>

#include <fstream>
#include <sstream>

using namespace optix;
using namespace prodlib;
using namespace corelib;

static inline void computeReferenceResolutionLogic( bool bIN, bool bV, bool& bR, bool& bOUT );

//------------------------------------------------------------------------
// CTOR/DTOR
//------------------------------------------------------------------------

Program::Program( Context* context )
    : LexicalScope( context, RT_OBJECT_PROGRAM )
{
    m_progId = m_context->getObjectManager()->registerObject( this );

    reallocateRecord();
}

Program::~Program()
{
    // All objects must have linked pointers detached before the destructor is called.
    RT_ASSERT_MSG( m_linkedPointers.empty(), "Program destroyed while references remain" );

    // All virtual children must have been removed
    RT_ASSERT_MSG( m_virtualParents.empty(), "Program destroyed while virtual parents remain" );

    // All annotations must have been removed
    RT_ASSERT_MSG( m_rootAnnotations.empty(), "Program destroyed while root annotations remain" );

    reallocateSbtIndex( false );

    deleteVariables();

    // Mark the canonical programs as no longer used on the device
    for( Device* device : m_context->getDeviceManager()->visibleDevices() )
    {
        if( const CanonicalProgram* cp = getCanonicalProgramNoThrow( device ) )
            cp->receivePropertyDidChange_UsedOnDevice( device, false );
    }

    const ProgramManager* pm = m_context->getProgramManager();
    for( const auto& element : m_callSitePotentialCallees )
    {
        const std::string&  callSiteUniqueName = element.first;
        CallSiteIdentifier* csId               = pm->getCallSiteByUniqueName( callSiteUniqueName );
        if( csId )
        {
            csId->addOrRemovePotentialCallees( element.second, false );
        }
    }

    for( auto cp : m_canonicalPrograms )
    {
        // Remove the buffer escape property if applicable
        if( cp->canBindlessBufferPointerEscape() )
            getContext()->getProgramManager()->addOrRemoveRawBindlessBufferAccesses( false );

        // The program will be released later when the ProgramManager is destroyed
    }

    // Clear header
    m_context->getTableManager()->clearProgramHeader( *m_progId );
}


//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void Program::createFromFile( const std::string& fileName, const std::string& functionName, optix::lwca::ComputeCapability targetMax )
{
    std::ifstream input_file( fileName.c_str() );

    if( !input_file )
        throw FileNotFound( RT_EXCEPTION_INFO, fileName );

    std::stringstream str;
    str << input_file.rdbuf();

    if( input_file.fail() )
        throw UnknownError( RT_EXCEPTION_INFO, "Error reading input file (" + fileName + ")" );

    createFromString( str.str().c_str(), fileName, functionName, targetMax );
}

void Program::createFromFiles( const std::vector<std::string>& fileNames, const std::string& functionName, optix::lwca::ComputeCapability targetMax )
{
    std::vector<std::string> ptxStrings;
    ptxStrings.reserve( fileNames.size() );

    for( size_t i = 0; i < fileNames.size(); ++i )
    {
        std::ifstream input_file( fileNames[i].c_str() );

        if( !input_file )
            throw FileNotFound( RT_EXCEPTION_INFO, fileNames[i] );

        std::stringstream str;
        str << input_file.rdbuf();

        if( input_file.fail() )
            throw UnknownError( RT_EXCEPTION_INFO, "Error reading input file (" + fileNames[i] + ")" );

        ptxStrings.push_back( str.str() );
    }

    // Create the string views after loading all sources to ensure that no
    // pointers are moved.
    std::vector<prodlib::StringView> ptxStringViews;
    ptxStringViews.reserve( ptxStrings.size() );
    for( const std::string& str : ptxStrings )
    {
        ptxStringViews.push_back( {str.c_str(), str.size()} );
    }

    createFromStrings( ptxStringViews, fileNames, functionName, targetMax );
}

void Program::createFromString( const char* str, const std::string& fileName, const std::string& functionName, optix::lwca::ComputeCapability targetMax )
{
    // TODO use vectorized strlen
    prodlib::StringView strView( toStringView( str ) );
    // If encryption is enabled then every user program must be encrypted
    if( m_context->getEncryptionManager()->isEncryptionEnabled() && !m_context->getEncryptionManager()->hasEncryptionPrefix( strView ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Invalid encoding: detected unencrypted input PTX after encryption was enabled." );

    addPTX( strView, fileName, functionName, targetMax );
    finishAddingCanonicalPrograms();
}

void Program::createFromStrings( const std::vector<prodlib::StringView>& strs,
                                 const std::vector<std::string>&         fileNames,
                                 const std::string&                      functionName,
                                 optix::lwca::ComputeCapability          targetMax )
{
    // If encryption is enabled then every user program must be encrypted
    if( !strs.empty() && m_context->getEncryptionManager()->isEncryptionEnabled()
        && !m_context->getEncryptionManager()->hasEncryptionPrefix( strs[0] ) )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                     "Invalid encoding: detected unencrypted input PTX after encryption was enabled." );

    addPTX( strs, fileNames, functionName, targetMax );
    finishAddingCanonicalPrograms();
}

void Program::createFromProgram( Program* program )
{
    if( program->m_context != m_context )
        throw IlwalidValue( RT_EXCEPTION_INFO,
                            "program_in does not belong to the same context as the context used to create the new "
                            "program." );

    for( const CanonicalProgram* canonicalProgram : program->m_canonicalPrograms )
    {
        addCanonicalProgram( canonicalProgram );
    }
    finishAddingCanonicalPrograms();
}

int Program::getAPIId()
{
    markAsBindless();
    return getId();
}


void Program::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    LexicalScope::validate();

    // Will throw an exception if a program was not found for an active device
    for( Device* device : m_context->getDeviceManager()->uniqueActiveDevices() )
        getCanonicalProgram( device );

    if( !m_usedAsBoundCallableProgram.empty() )
        validateSemanticType( ST_BOUND_CALLABLE_PROGRAM );

    for( auto stype : m_inheritedSemanticType )
        validateSemanticType( stype );

    for( auto cp : m_canonicalPrograms )
    {
        if( cp->isUsedAsSemanticType( ST_ATTRIBUTE ) && !cp->isUsedAsSingleSemanticType() )
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "program cannot be used as both ST_ATTRIBUTE and another semantic type" );
    }
}

void Program::setCallsitePotentialCallees( const std::string& callSiteName, const std::vector<int>& calleeIds )
{
    ProgramManager* pm = m_context->getProgramManager();
    ObjectManager*  om = m_context->getObjectManager();
    // The callee Ids are programIds not CanonicalProgramIds. So, we need to translate...
    std::vector<CanonicalProgramID> calleeCpIds;
    for( int progId : calleeIds )
    {
        Program* program = om->getProgramById( progId );
        RT_ASSERT( program );
        std::vector<CanonicalProgramID> cpIds = program->getCanonicalProgramIDs();
        calleeCpIds.insert( calleeCpIds.end(), cpIds.begin(), cpIds.end() );
    }
    for( const CanonicalProgram* cp : m_canonicalPrograms )
    {
        std::string         callSiteUniqueName = cp->getUniversallyUniqueName() + "." + callSiteName;
        CallSiteIdentifier* csId               = pm->getCallSiteByUniqueName( callSiteUniqueName );
        if( !csId )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Invalid call site name: " + callSiteName + " in program " + cp->getInputFunctionName() );
        auto lwrCalleeIt = m_callSitePotentialCallees.find( callSiteUniqueName );
        if( lwrCalleeIt != m_callSitePotentialCallees.end() )
        {
            csId->addOrRemovePotentialCallees( lwrCalleeIt->second, false );
        }
        csId->addOrRemovePotentialCallees( calleeCpIds, true );
        m_callSitePotentialCallees[callSiteUniqueName] = calleeCpIds;
    }
}

//------------------------------------------------------------------------
// Canonical program management
//------------------------------------------------------------------------

const CanonicalProgram* Program::getCanonicalProgram( const Device* device ) const
{
    const CanonicalProgram* bestCP = getCanonicalProgramNoThrow( device );
    if( bestCP )
        return bestCP;

    // Compute the minimum device in the list for the error string.
    lwca::ComputeCapability minSM( 9999 );
    for( const CanonicalProgram* cp : m_canonicalPrograms )
    {
        if( minSM > cp->getTargetMin() )
            minSM = cp->getTargetMin();
    }
    throw IlwalidSource( RT_EXCEPTION_INFO, "The supplied PTX is SM" + minSM.toString()
                                                + " and not compatible with the device " + device->deviceName() );
}

void Program::addPTX( const prodlib::StringView&     ptx,
                      const std::string&             file,
                      const std::string&             functionName,
                      optix::lwca::ComputeCapability targetMax,
                      bool                           useDiskCache )
{
    llog( 20 ) << "addPTX " << file << " " << functionName << '\n';

    preAddCodeCheck( "addPTX" );

    const std::vector<prodlib::StringView> ptxStrings{ptx};
    const std::vector<std::string>         files{file};
    const CanonicalProgram*                cp =
        getContext()->getProgramManager()->canonicalizePTX( ptxStrings, files, functionName, targetMax, useDiskCache );
    addCanonicalProgram( cp );
}

void Program::addPTX( const std::vector<prodlib::StringView>& ptxStrings,
                      const std::vector<std::string>&         files,
                      const std::string&                      functionName,
                      optix::lwca::ComputeCapability          targetMax,
                      bool                                    useDiskCache )
{
    RT_ASSERT_MSG( ptxStrings.size() == files.size(), "addPTX called with mismatched number of strings and files" );

    llog( 20 ) << "addPTX (multiple strings): " << functionName << '\n';
    for( size_t i = 0; i < ptxStrings.size(); ++i )
    {
        llog( 20 ) << "  " << i << ": " << files[i] << '\n';
    }

    preAddCodeCheck( "addPTX (multiple strings)" );

    const CanonicalProgram* cp =
        getContext()->getProgramManager()->canonicalizePTX( ptxStrings, files, functionName, targetMax, useDiskCache );
    addCanonicalProgram( cp );
}

void Program::addLLVM( llvm::Function* func, CanonicalizationType type, optix::lwca::ComputeCapability targetMin, optix::lwca::ComputeCapability targetMax )
{
    llog( 20 ) << "addLLVM " << func->getName().data() << '\n';

    preAddCodeCheck( "addLLVM" );

    size_t                  ptxHash = corelib::hashString( func->getName() );
    const CanonicalProgram* cp =
        getContext()->getProgramManager()->canonicalizeFunction( func, type, targetMin, targetMax, ptxHash );
    addCanonicalProgram( cp );
}

void Program::finishAddingCanonicalPrograms()
{
    RT_ASSERT_MSG( !m_finishedAddingCanonicalPrograms, "finishAddingCanonicalPrograms called twice" );

    // Find the best matching CanonicalProgram for each device and add
    // usage propreties to CanonicalProgram.  Produce an error if no
    // variant for any device is found.  This catches the most common
    // case right at Program creation as a colwenience for the user.
    bool someVariantFound = false;
    for( Device* device : m_context->getDeviceManager()->visibleDevices() )
    {
        if( const CanonicalProgram* cp = getCanonicalProgramNoThrow( device ) )
        {
            someVariantFound = true;
            cp->receivePropertyDidChange_UsedOnDevice( device, true );
        }
    }

    if( !someVariantFound )
    {
        // Compute the minimum device in the list for the error string.
        lwca::ComputeCapability minSM( 9999 );
        for( const CanonicalProgram* cp : m_canonicalPrograms )
        {
            if( minSM > cp->getTargetMin() )
                minSM = cp->getTargetMin();
        }
        throw IlwalidSource( RT_EXCEPTION_INFO, "The supplied PTX requires at least one device with SM "
                                                    + minSM.toString() + " or greater." );
    }

    // Notify the TableManager that there are new canonical programs
    m_context->getTableManager()->notifyCanonicalProgramAddedToProgram( this );

    m_finishedAddingCanonicalPrograms = true;
}

void Program::postSetActiveDevices()
{
    // We need to make sure there is a program variant for the current set of active
    // devices.
    subscribeForValidation();
}

void Program::preAddCodeCheck( const char* functionName ) const
{
    // For proper variable reference resolution we assume that all
    // CanonicalPrograms are added before variables are declared, before
    // the program is attached to a scope and before it is unbound.
    // Assert these conditions.
    RT_ASSERT_MSG( getVariableCount() == 0,
                   std::string( functionName ) + " called with Variables attached to the scope" );
    RT_ASSERT_MSG( m_linkedPointers.empty(),
                   std::string( functionName ) + " called on Program after it was attached to the graph" );
    RT_ASSERT_MSG( !isBindless(), std::string( functionName ) + " called on Program after it was declared bindless" );
}

void Program::addCanonicalProgram( const CanonicalProgram* cp )
{
    // Ensure that there are no overlapping variants already in the list
    for( const CanonicalProgram* canonicalProgram : m_canonicalPrograms )
    {
        if( cp->getTargetMin() < canonicalProgram->getTargetMax() && cp->getTargetMax() > canonicalProgram->getTargetMin() )
        {
            RT_ASSERT_FAIL_MSG( "Overlapping PTX variants are not allowed" );
        }
    }

    m_canonicalPrograms.push_back( cp );

    // Populate the reference sets.  Since we are guaranteed that no
    // objects have been attached as a parent until all PTX is added and
    // no variables exist, we do not need to propagate the set changes.
    for( auto varref : cp->getVariableReferences() )
        m_unresolvedSet.addOrRemoveProperty( varref->getReferenceID(), true );

    // And populate attributes
    for( auto varref : cp->getAttributeReferences() )
        m_unresolvedAttributeSet.addOrRemoveProperty( varref->getReferenceID(), true );

    // NOTE: A potential optimization here is to only track raw bindless buffer accesses when the Program becomes attached/detached.
    //       Until this proves to be a performance problem, we will just assume that the existence of the Program makes the buffers raw access.
    if( cp->canBindlessBufferPointerEscape() )
        getContext()->getProgramManager()->addOrRemoveRawBindlessBufferAccesses( true );

    subscribeForValidation();
}

const CanonicalProgram* Program::getCanonicalProgramNoThrow( const Device* device ) const
{
    // Find the appropriate variant for this device
    const CanonicalProgram* bestCP = nullptr;
    for( const CanonicalProgram* cp : m_canonicalPrograms )
    {
        // Ignore CPs that are invalid for the given device.
        if( !cp->isValidForDevice( device ) )
            continue;

        // The best CP is the one with the highes minimum target SM version.
        if( bestCP && cp->getTargetMin() <= bestCP->getTargetMin() )
            continue;

        bestCP = cp;
    }
    return bestCP;
}


//------------------------------------------------------------------------
// Internal API
//------------------------------------------------------------------------

unsigned Program::getFunctionSignature() const
{
    // Return the signature of the first variant
    RT_ASSERT( !m_canonicalPrograms.empty() );
    return m_canonicalPrograms[0]->getFunctionSignature();
}

std::string Program::getInputFunctionName() const
{
    // Return the name of the first variant
    RT_ASSERT( !m_canonicalPrograms.empty() );
    return m_canonicalPrograms[0]->getInputFunctionName();
}

int Program::getId() const
{
    RT_ASSERT( m_progId != nullptr );
    return *m_progId;
}

bool Program::isBindless() const
{
    return m_isBindless;
}

bool Program::isUsedAsBoundingBoxProgram() const
{
    return !m_usedAsBoundingBoxProgram.empty();
}

bool Program::hasMotionIndexArg() const
{
    // Return this from the first variant
    RT_ASSERT( !m_canonicalPrograms.empty() );
    return m_canonicalPrograms[0]->hasMotionIndexArg();
}

std::vector<CanonicalProgramID> Program::getCanonicalProgramIDs()
{
    std::vector<CanonicalProgramID> ids;
    for( auto& cp : m_canonicalPrograms )
    {
        ids.push_back( cp->getID() );
    }
    return ids;
}

unsigned int Program::get32bitAttributeKind() const
{
    RT_ASSERT_MSG( m_canonicalPrograms.size() == 1,
                   "Only single CanonicalProgram per Program supported for getting attribute kind.  Lwrrently "
                       + std::to_string( m_canonicalPrograms.size() ) + " are attached." );
    return m_canonicalPrograms[0]->get32bitAttributeKind();
}


//------------------------------------------------------------------------
// LinkedPtr relationship mangement
//------------------------------------------------------------------------

void Program::detachFromParents()
{
    // Detach linked pointers - being careful to not ilwalidate the iterator
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        if( LexicalScope* parent = getLinkToProgramFrom<GlobalScope, Geometry, Material, GraphNode>( parentLink ) )
            parent->detachLinkedChild( parentLink );
        else if( Variable* variable = getLinkToProgramFrom<Variable>( parentLink ) )
            variable->detachLinkedChild( parentLink );
        else if( Buffer* buffer = getLinkToProgramFrom<Buffer>( parentLink ) )
            buffer->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( std::string( "Unexpected linked pointer type in Program: " ) + typeid( *parentLink ).name() );

        iter = m_linkedPointers.begin();
    }

    // Remove references from global scope, attachment and semantic type
    // for bindless programs
    if( isBindless() )
    {
        GlobalScope* globalScope = m_context->getGlobalScope();
        ProgramRoot  root( globalScope->getScopeID(), ST_BINDLESS_CALLABLE_PROGRAM, 0 );

        receivePropertyDidChange_UsedAsSemanticType( ST_BINDLESS_CALLABLE_PROGRAM, false );
        receivePropertyDidChange_Attachment( false );

        attachOrDetachProperty_UnresolvedReference( m_context->getGlobalScope(), root, false );

        reallocateSbtIndex( false );
    }
}

void Program::detachLinkedChild( const LinkedPtr_Link* link )
{
    RT_ASSERT_FAIL_MSG( "Program object does not have LinkedPtr children" );
}


//------------------------------------------------------------------------
// Object record access and management
//------------------------------------------------------------------------

size_t Program::getRecordBaseSize() const
{
    return sizeof( cort::ProgramRecord );
}

void Program::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::ProgramRecord* p = getObjectRecord<cort::ProgramRecord>();
    p->programID           = getId();
    LexicalScope::writeRecord();
    writeHeader();
}

void Program::notifyParents_offsetDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( LexicalScope* parent = getLinkToProgramFrom<GlobalScope, Geometry, Material, GraphNode>( parentLink ) )
            parent->childOffsetDidChange( parentLink );
        else if( getLinkToProgramFrom<Variable>( parentLink ) )
            // Variable does not need to be notified.  The Program offset is
            // not tracked in the variable - only the ID.
            ;
        else if( getLinkToProgramFrom<Buffer>( parentLink ) )
            // No need to notify the buffer either.
            ;
        else
            RT_ASSERT_FAIL_MSG( std::string( "Unexpected linked pointer type in Program: " ) + typeid( *parentLink ).name() );
    }
}

void Program::offsetDidChange() const
{
    notifyParents_offsetDidChange();
    if( ( isBindless() && isAttached() ) || m_usedAsBoundingBoxProgram.count() || m_usedAsBoundCallableProgram.count() )
        m_context->getSBTManager()->callableProgramOffsetDidChange( this );
}

void Program::writeHeader() const
{
    if( !m_progId )
        return;  // This can happen during construction of the object
    if( m_SBTIndex )
        m_context->getTableManager()->writeProgramHeader( *m_progId, *m_SBTIndex );
    else
        m_context->getTableManager()->writeProgramHeader( *m_progId, getRecordOffset() );
    // canonicalProgramID filled by notifyCanonicalProgramAdded and by the TableManager for each device
}

//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

static inline void computeReferenceResolutionLogic( bool bIN, bool bV, bool& bR, bool& bOUT )
{
    /*
   * IN = union(all CanonicalPrograms)                 // No dynamic attach of canonical programs
   * R = intersect(IN, V);                             // Resolution change
   * OUT = IN - V                                      // Notify parents
   */

    bool bI = bIN;
    bR      = bI && bV;
    bOUT    = bI && !bV;
}

void Program::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    if( callsBoundCallableProgram() )
    {
        // As the linked pointers are processed, we must remap the program
        // IDs. However, there is no good way to determine which linked
        // pointer belongs to which scope root - and it does not really
        // matter. The first time we encounter a particular parent, push
        // the reference to all scope roots corresponding to that
        // reference. Note that this loop is not structured to handle
        // large numbers of root annotations but we do not expect many. An
        // auxiliary index could be added if this is a performance issue.
        std::set<LexicalScope*> visited;
        for( auto parentLink : m_linkedPointers )
        {
            if( LexicalScope* parent = getLinkToProgramFrom<GlobalScope, Geometry, Material, GraphNode>( parentLink ) )
            {
                if( visited.insert( parent ).second )
                {
                    for( const auto& ann_parent : m_rootAnnotations )
                    {
                        if( parent->getScopeID() == ann_parent.first.scopeid )
                        {
                            VariableReferenceID remap_refid = ann_parent.second.mapReference( refid );
                            parent->receivePropertyDidChange_UnresolvedReference( this, remap_refid, added );
                        }
                    }
                }
            }
            else
            {
                RT_ASSERT_MSG( getLinkToProgramFrom<Variable>( parentLink ) != nullptr
                                   || getLinkToProgramFrom<Buffer>( parentLink ) != nullptr,
                               std::string( "Unexpected linked pointer type in Program: " ) + typeid( *parentLink ).name() );
            }
        }

        if( isBindless() && isAttached() )
        {
            ProgramRoot bindlessRoot( m_context->getGlobalScope()->getScopeID(), ST_BINDLESS_CALLABLE_PROGRAM, 0 );
            const auto& ann_parent          = m_rootAnnotations.at( bindlessRoot );
            VariableReferenceID remap_refid = ann_parent.mapReference( refid );
            m_context->getGlobalScope()->receivePropertyDidChange_UnresolvedReference( this, remap_refid, added );
        }
    }
    else
    {
        // Normal path
        // Parents are only: GlobalScope, Geometry, Material, GraphNode
        for( auto parentLink : m_linkedPointers )
        {
            if( LexicalScope* parent = getLinkToProgramFrom<GlobalScope, Geometry, Material, GraphNode>( parentLink ) )
                parent->receivePropertyDidChange_UnresolvedReference( this, refid, added );
            else
                RT_ASSERT_MSG( getLinkToProgramFrom<Variable>( parentLink ) != nullptr
                                   || getLinkToProgramFrom<Buffer>( parentLink ) != nullptr,
                               std::string( "Unexpected linked pointer type in Program: " ) + typeid( *parentLink ).name() );
        }

        // Bindless programs are implicitly attached to global scope
        if( isBindless() && isAttached() )
            m_context->getGlobalScope()->receivePropertyDidChange_UnresolvedReference( this, refid, added );
    }

    // In addition, send references to the virtual parents - again
    // taking care to remap references if appropriate.
    for( auto root : m_virtualParents )
    {
        LexicalScope*       parent      = m_context->getObjectManager()->getLexicalScopeById( root.scopeid );
        VariableReferenceID remap_refid = refid;
        if( callsBoundCallableProgram() )
        {
            RT_ASSERT_MSG( m_rootAnnotations.count( root ) > 0, "Globalscope root not found for bindless program" );
            remap_refid = m_rootAnnotations.at( root ).mapReference( refid );
        }
        parent->receivePropertyDidChange_UnresolvedReference( this, remap_refid, added );
    }
}

void Program::attachOrDetachProperty_UnresolvedReference( LexicalScope* parent, const ProgramRoot& root, bool attached, bool virtualParent )
{
    scopeTrace( "begin attachOrDetachProperty_UnresolvedReference", ~0, attached, parent );

    Annotation* pi = nullptr;

    // Bound callable programs remap program references to an ID that is
    // unique to this scope root so that it can be handled specially
    // when the references are bound.
    if( callsBoundCallableProgram() )
    {
        ProgramManager* pm = m_context->getProgramManager();
        pi                 = &m_rootAnnotations[root];

        if( attached )
        {
            pi->count++;

            // Populate the virtual references for bound callable programs.
            for( auto refid : m_unresolvedSet )
            {
                const VariableReference* vref = m_context->getProgramManager()->getVariableReferenceById( refid );
                if( !vref->getType().isProgram() )
                    continue;

                if( pi->programReferences.count( refid ) != 0 )
                    continue;  // Already have mapping

                // Register a virtual reference
                const VariableReference* virtual_vref = pm->registerVirtualVariableReference( root, vref );
                pi->programReferences.insert( std::make_pair( refid, virtual_vref->getReferenceID() ) );
            }
        }

        // Also add a virtual parent to the programs that were resolved at
        // this scope (if any)
        for( auto refid : getResolvedReferenceSet() )
        {
            const VariableReference* vref = m_context->getProgramManager()->getVariableReferenceById( refid );
            if( !vref->getType().isProgram() )
                continue;

            Variable* variable = getVariableByToken( vref->getVariableToken() );
            if( !variable || !variable->isProgram() )
                continue;

            if( Program* child = variable->getProgram() )
                child->addOrRemove_VirtualParent( root, attached );
        }
    }

    // Callwlate output set and propagate with the remapped IDs
    for( auto refid : m_unresolvedSet )
    {
        bool bIN = true;
        bool bV  = haveVariableForReference( refid );
        // Callwlate derived set
        bool bR, bOUT;
        computeReferenceResolutionLogic( bIN, bV, bR, bOUT );

        // Propagate the reference. Omit the annotation for non-program references.
        if( bOUT )
        {
            VariableReferenceID remapped_refid = pi ? pi->mapReference( refid ) : refid;
            parent->receivePropertyDidChange_UnresolvedReference( this, remapped_refid, attached );
        }
    }

    // If we are detaching the root scope, clean up the parent annotations
    if( !attached && pi )
    {
        RT_ASSERT_MSG( pi->count > 0, "Detaching root with count of zero" );
        if( --pi->count == 0 )
        {
            // Remove the virtual parent record
            ProgramManager* pm = m_context->getProgramManager();
            for( auto refs : pi->programReferences )
            {
                const VariableReference* vref = m_context->getProgramManager()->getVariableReferenceById( refs.second );
                pm->removeVirtualVariableReference( vref );
            }

            if( !virtualParent )
            {
                // See comment in BindingManager.cpp
                m_context->getBindingManager()->forceDetach( root );
            }

            m_rootAnnotations.erase( root );
        }
    }


    // Flush the queue
    if( !virtualParent )
        m_context->getBindingManager()->processVirtualParentQueue();

    scopeTrace( "end attachOrDetachProperty_UnresolvedReference", ~0, attached, parent );
}

void Program::receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added )
{
    // Note that references cannot be added after a program is
    // created. Furthermore, we are certain that no variables will be
    // bound when the program is created. References from bound callable
    // children will be added elsewhere.
    RT_ASSERT_MSG( getVariableCount() == 0, "References added to Program after initial construction" );
}

//------------------------------------------------------------------------
// Unresolved attribute property
//------------------------------------------------------------------------

void Program::attachOrDetachProperty_UnresolvedAttributeReference( Geometry* geometry, bool attached ) const
{
    // Attributes are passed directly through without resolution at the
    // program scope.
    for( auto refid : m_unresolvedAttributeSet )
        geometry->receivePropertyDidChange_UnresolvedAttributeReference( this, refid, attached );
}

void Program::attachOrDetachProperty_UnresolvedAttributeReference( Material* material, bool attached ) const
{
    // Attributes are passed directly through without resolution at the
    // program scope.
    for( auto refid : m_unresolvedAttributeSet )
        material->receivePropertyDidChange_UnresolvedAttributeReference( this, refid, attached );
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void Program::sendPropertyDidChange_Attachment( bool added ) const
{
    // No further propagation of attachment
}


//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void Program::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    for( auto cp : m_canonicalPrograms )
        cp->receivePropertyDidChange_DirectCaller( cpid, added );
}

void Program::attachOrDetachProperty_DirectCaller( LexicalScope* atScope, bool added ) const
{
    for( auto cp : m_canonicalPrograms )
        atScope->receivePropertyDidChange_DirectCaller( cp->getID(), added );
}

//------------------------------------------------------------------------
// Semantic type property and used by raytype property
//------------------------------------------------------------------------

void Program::receivePropertyDidChange_UsedAsSemanticType( SemanticType stype, bool added )
{
    // Property not stored on program. Pass it down to CanonicalProgram
    for( auto cp : m_canonicalPrograms )
        cp->receivePropertyDidChange_UsedAsSemanticType( stype, added );

    if( stype == ST_BOUNDING_BOX )
    {
        bool changed = m_usedAsBoundingBoxProgram.addOrRemoveProperty( added );
        if( changed )
            reallocateSbtIndex( added );
    }
    else if( stype == ST_BOUND_CALLABLE_PROGRAM )
    {
        bool changed = m_usedAsBoundCallableProgram.addOrRemoveProperty( added );
        if( changed )
            reallocateSbtIndex( added );
    }
}

void Program::receivePropertyDidChange_UsedByRayType( unsigned int rayType, bool added )
{
    // Property not stored on program. Pass it down to CanonicalProgram
    for( auto cp : m_canonicalPrograms )
        cp->receivePropertyDidChange_UsedByRayType( rayType, added );
}

void Program::receivePropertyDidChange_InheritedSemanticType( SemanticType stype, bool added )
{
    bool changed = m_inheritedSemanticType.addOrRemoveProperty( stype, added );

    if( changed )
    {
        for( auto cp : m_canonicalPrograms )
            cp->receivePropertyDidChange_InheritedSemanticType( stype, added );

        subscribeForValidation();
        if( added && m_context->useRtxDataModel() )
        {
            // A new inherited SemanticType was added. Update the corresponding
            // SBT entry. No need to change the SBT allocation since it was
            // already allocated big enough when this Program was marked as being
            // a bound callable program.
            RT_ASSERT( m_SBTIndex );
            m_context->getSBTManager()->updateBoundCallableProgramEntry( this, stype );
        }
    }
}

void Program::validateSemanticType( SemanticType stype ) const
{
    for( auto cp : m_canonicalPrograms )
        cp->validateSemanticType( stype );
}


//------------------------------------------------------------------------
// Handling of bound callable programs
//------------------------------------------------------------------------

void Program::connectOrDisconnectProperties_VirtualParent( const ProgramRoot& root, bool connecting )
{
    // A new scope has attached itself as a virtual parent of this
    // program. Establish the appropriate properties on the connection.

    // Add (remove) the inherited semantic type
    receivePropertyDidChange_InheritedSemanticType( root.stype, connecting );

    // For closest hit, any hit and miss programs - also add (remove) the ray type
    if( root.stype == ST_CLOSEST_HIT || root.stype == ST_ANY_HIT || root.stype == ST_MISS )
        receivePropertyDidChange_UsedByRayType( root.index, connecting );

    // Note: attachment is not propagated here. It is pushed through the program variable
    LexicalScope* parent = m_context->getObjectManager()->getLexicalScopeById( root.scopeid );

    // For closest hit, any hit and intersect programs - also add (remove) attributes
    if( root.stype == ST_CLOSEST_HIT || root.stype == ST_ANY_HIT )
        attachOrDetachProperty_UnresolvedAttributeReference( static_cast<Material*>( parent ), connecting );
    else if( root.stype == ST_INTERSECTION )
        attachOrDetachProperty_UnresolvedAttributeReference( static_cast<Geometry*>( parent ), connecting );

    // Add (remove) unresolved references - with the same annotation
    attachOrDetachProperty_UnresolvedReference( parent, root, connecting, true );
}

void Program::addOrRemove_VirtualParent( const ProgramRoot& root, bool added )
{
    bool changed = m_virtualParents.addOrRemoveProperty( root, added );
    if( changed )
        m_context->getBindingManager()->enqueueVirtualParentConnectOrDisconnect( this, root, added );
}

void Program::programBindingDidChange( LexicalScope* atScope, VariableReferenceID refid, bool added )
{
    const VariableReference* vref = m_context->getProgramManager()->getVariableReferenceById( refid );
    if( atScope->getClass() == RT_OBJECT_PROGRAM )
    {
        // Connect to the bindee's real parents and virtual parents.
        Program* other = managedObjectCast<Program>( atScope );
        RT_ASSERT( other != nullptr );
        for( const auto& parent : other->m_virtualParents )
            addOrRemove_VirtualParent( parent, added );

        if( added )
            for( const auto& parent : other->m_rootAnnotations )
                addOrRemove_VirtualParent( parent.first, added );
    }
    else
    {
        // Look up the virtual variable reference to determine the virtual
        // parent associated with this binding.
        RT_ASSERT_MSG( vref->getLinkedReference() != nullptr, "Binding program to non-virtual reference" );
        addOrRemove_VirtualParent( vref->getBoundProgramRoot(), added );
    }
}


bool Program::hasVirtualParent( const ProgramRoot& root ) const
{
    // See BindingManager for hacky details of forced cleanup of
    // cirlwlar bound callable program references.
    return m_virtualParents.contains( root );
}

void Program::dropVirtualParents( const ProgramRoot& root )
{
    // See BindingManager for hacky details of forced cleanup of
    // cirlwlar bound callable program references.
    int count = m_virtualParents.count( root );
    for( int i = 0; i < count; ++i )
        m_virtualParents.addOrRemoveProperty( root, false );
}


//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------

void Program::markAsBindless()
{
    if( m_isBindless )
        return;

    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation
    this->validateSemanticType( ST_BINDLESS_CALLABLE_PROGRAM );

    m_isBindless = true;

    // Push unresolved references to global scope and mark this as
    // attached
    GlobalScope* globalScope = m_context->getGlobalScope();
    ProgramRoot  root( globalScope->getScopeID(), ST_BINDLESS_CALLABLE_PROGRAM, 0 );
    attachOrDetachProperty_UnresolvedReference( globalScope, root, true );

    // Add attachment and bindless semantic types
    receivePropertyDidChange_Attachment( true );
    receivePropertyDidChange_UsedAsSemanticType( ST_BINDLESS_CALLABLE_PROGRAM, true );

    reallocateSbtIndex( true );

    subscribeForValidation();
}

bool Program::callsBoundCallableProgram() const
{
    for( auto cp : m_canonicalPrograms )
        if( cp->callsBoundCallableProgram() )
            return true;
    return false;
}

VariableReferenceID Program::Annotation::mapReference( VariableReferenceID refid ) const
{
    auto iter = programReferences.find( refid );
    if( iter != programReferences.end() )
        return iter->second;
    else
        return refid;
}

size_t Program::getSBTRecordIndex() const
{
    if( !m_SBTIndex )
        return static_cast<size_t>( -1 );  // 0 could be a valid SBT index
    return *m_SBTIndex;
}

std::vector<SemanticType> Program::getInheritedSemanticTypes() const
{
    std::vector<SemanticType> types;
    types.reserve( m_inheritedSemanticType.size() );
    for( SemanticType stype : m_inheritedSemanticType )
        types.push_back( stype );
    return types;
}

void Program::reallocateSbtIndex( bool added )
{
    if( !m_context->useRtxDataModel() )
        return;

    const bool         haveOldSBTIndex = static_cast<bool>( m_SBTIndex );
    const unsigned int oldSBTIndex     = haveOldSBTIndex ? *m_SBTIndex : 0U;

    m_SBTIndex.reset();
    if( added )
    {
        m_SBTIndex = m_context->getSBTManager()->callableProgramDidChange( this );
        if( m_SBTIndex && ( !haveOldSBTIndex || oldSBTIndex != *m_SBTIndex ) )
        {
            writeHeader();
        }
    }
}
