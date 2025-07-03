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

#include <Context/ProgramManager.h>

#include <AS/Traversers.h>
#include <LWCA/ComputeCapability.h>
#include <Context/Context.h>
#include <Context/LLVMManager.h>
#include <Context/ObjectManager.h>
#include <Context/UpdateManager.h>
#include <FrontEnd/Canonical/CallSiteIdentifier.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <FrontEnd/PTX/Canonical/C14n.h>
#include <FrontEnd/PTX/PTXHeader.h>
#include <FrontEnd/PTX/PTXNamespaceMangle.h>
#include <FrontEnd/PTX/PTXtoLLVM.h>
#include <Objects/Buffer.h>
#include <Objects/Program.h>
#include <Objects/VariableType.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/MemoryStream.h>
#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>

#include <exp/context/DeviceContext.h>
#include <exp/context/DiskCache.h>
#include <exp/context/EncryptionManager.h>
#include <exp/context/ErrorHandling.h>

#include <corelib/misc/String.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidSource.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/Encryption.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Thread.h>

#include <llvm/ADT/StringExtras.h>

#include <llvm/IR/Verifier.h>
#include <llvm/Support/DJB.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <memory>
#include <sstream>

using namespace optix;
using namespace optix::lwca;
using namespace prodlib;
using namespace corelib;

namespace {
Knob<bool> k_parseLineNumbers( RT_DSTRING( "context.preserveLWLineInfo" ),
                               true,
                               RT_DSTRING( "Preserve line info in PTX frontend" ) );
Knob<bool> k_cacheEnabled( RT_DSTRING( "diskcache.canonical.enabled" ),
                           true,
                           RT_DSTRING( "Enable or disable the disk cache for canonical programs." ) );
}  // namespace

const uint32_t ProgramManager::m_teaKey[4] = {0x9ef2a195, 0x900943a8, 0xb2256e4a, 0xf4d1dc61};

ProgramManager::ProgramManager( Context* context )
    : m_context( context )
{
}

ProgramManager::~ProgramManager()
{
    for( VariableReferenceIDListType* reference : m_ilwerseReferences )
        delete reference;

    // Delete all canonical programs when the program manager is destroyed
    for( auto cp : m_canonicalPrograms )
        delete cp;
}

Thread::Mutex g_compileMutex;

CanonicalProgram* ProgramManager::canonicalizeFunction( llvm::Function*         fn,
                                                        CanonicalizationType    type,
                                                        lwca::ComputeCapability targetMin,
                                                        lwca::ComputeCapability targetMax,
                                                        size_t                  ptxHash )
{
    CanonicalProgram* cp = nullptr;

    switch( type )
    {
        case CanonicalizationType::CT_TRAVERSER:
        case CanonicalizationType::CT_PTX:
        {
            TIMEVIZ_SCOPE( "canonicalize" );
            C14n canonicalization( fn, type, targetMin, targetMax, ptxHash, m_context, m_context->getLLVMManager(),
                                   m_context->getProgramManager(), m_context->getObjectManager() );
            cp = canonicalization.run();
            break;
        }
            // Default case omitted intentionally.
    }

    // Take ownership
    m_canonicalPrograms.push_back( cp );

    // Send create event
    m_context->getUpdateManager()->eventCanonicalProgramCreate( cp );

    return cp;
}


// Look for a line in the ptx like ".target sm_35" and extract the "35"
// If not found, returns a default version
static int parseSmVersion( const char* ptx )
{
    int smVersion = 10;

    std::istringstream iss( ptx );
    std::string        line;
    // for each line in ptx
    while( std::getline( iss, line ) )
    {
        // skip initial whitespace
        const size_t pos = line.find_first_not_of( " \t" );
        if( pos != std::string::npos )
        {
            // match ".target "
            if( line.compare( pos, 8, ".target " ) == 0 )
            {
                // skip extra whitespace after ".target "
                const size_t sm_pos = line.find_first_not_of( " \t", pos + 8 );
                if( sm_pos != std::string::npos )
                {
                    // match "sm_" and read integer
                    if( line.compare( sm_pos, 3, "sm_" ) == 0 )
                    {
                        std::istringstream m( line.substr( sm_pos + 3 ) );
                        m >> smVersion;
                        break;
                    }
                }
            }
        }
    }
    return smVersion;
}

static size_t hashPTX( const Context* context, const std::vector<prodlib::StringView>& ptxStrings )
{
    // Compute hash
    // SGP: Should this be MD5?

    // Account for knobs and Context attributes
    std::string knobs = C14n::getCanonicalizationOptions( context );
    size_t      seed  = corelib::hashString( knobs );

    // Combine string hashes, based on boost::hash_combine
    for( const prodlib::StringView& s : ptxStrings )
    {
        size_t h = corelib::hashString( s.data(), s.size() );
        seed ^= h + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 );
    }
    return seed;
}

const CanonicalProgram* ProgramManager::canonicalizePTX( const std::vector<prodlib::StringView>& inputPtxStrings,
                                                         const std::vector<std::string>&         filenames,
                                                         const std::string&                      functionName,
                                                         lwca::ComputeCapability                 targetMax,
                                                         bool                                    useDiskCache )
{
    TIMEVIZ_FUNC;
    Thread::Lock lock( g_compileMutex );

    const size_t hashOnEncrypted = hashPTX( m_context, inputPtxStrings );

    // Step 1: Look in the in-memory cache.
    PTXModuleMap::iterator iter = m_ptxModules.find( hashOnEncrypted );
    if( iter != m_ptxModules.end() )
    {
        PTXModule* ptxModule = iter->second.get();
        auto       cpiter    = ptxModule->m_canonicalPrograms.find( functionName );
        if( cpiter != ptxModule->m_canonicalPrograms.end() )
            return cpiter->second;
    }

    // Optionally decrypt
    std::vector<std::vector<char>>   decryptedPtxDatas;
    std::vector<prodlib::StringView> decryptedPtxStringViews;
    size_t                           hashOnPlaintext = hashOnEncrypted;
    if( !inputPtxStrings.empty() && m_context->getEncryptionManager()->isEncryptionEnabled() )
    {
        decryptedPtxDatas.reserve( inputPtxStrings.size() );

        for( const prodlib::StringView& s : inputPtxStrings )
        {
            // The check that all user PTX is encrypted after encryption has been enabled is already done in
            // Program::createFromString/s(), since this method here is also used for internal default/dummy programs
            // which are never encrypted.
            if( !m_context->getEncryptionManager()->hasEncryptionPrefix( s ) )
            {
                decryptedPtxStringViews.push_back( s );
            }
            else
            {
                std::vector<char>       decryptedPtxData;
                optix_exp::ErrorDetails errDetails;
                if( OptixResult result = m_context->getEncryptionManager()->decrypt( s, decryptedPtxData, errDetails ) )
                    throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, errDetails.m_description );

                decryptedPtxDatas.push_back( std::move( decryptedPtxData ) );
                const std::vector<char>& ptx = decryptedPtxDatas.back();
                decryptedPtxStringViews.push_back( {ptx.data(), ptx.size()} );
            }
        }

        // Encryption keys may vary per Context, so also compute a stable hash on plaintext
        hashOnPlaintext = hashPTX( m_context, decryptedPtxStringViews );

        // Check the in-memory cache again, in case we have a plaintext version already
        // from before encryption was enabled.
        if( iter == m_ptxModules.end() )
        {
            iter = m_ptxModules.find( hashOnPlaintext );
            if( iter != m_ptxModules.end() )
            {
                PTXModule* ptxModule = iter->second.get();
                auto       cpiter    = ptxModule->m_canonicalPrograms.find( functionName );
                if( cpiter != ptxModule->m_canonicalPrograms.end() )
                    return cpiter->second;
            }
        }
    }

    const std::vector<prodlib::StringView>& ptxStrings = decryptedPtxStringViews.empty() ? inputPtxStrings : decryptedPtxStringViews;

    if( iter == m_ptxModules.end() )
    {
        // Create a placeholder module after parsing out just the SM version.
        const int smVersion = parseSmVersion( ptxStrings[0].data() );
        for( size_t i = 1; i < ptxStrings.size(); ++i )
        {
            if( parseSmVersion( ptxStrings[i].data() ) != smVersion )
                throw IlwalidSource( RT_EXCEPTION_INFO, std::string( "Function '" ) + functionName
                                                            + "': multiple ptx inputs must have matching sm versions" );
        }

        ComputeCapability targetMin( smVersion );
        iter = m_ptxModules
                   .emplace( hashOnEncrypted, std::make_shared<PTXModule>( hashOnPlaintext, targetMin, targetMax ) )
                   .first;
    }
    PTXModule* ptxModule = iter->second.get();

    //
    // Step 2: Check the disk cache.
    //

    std::string diskCacheKey;
    if( useDiskCache && m_context->getDiskCache()->isActive() && k_cacheEnabled.get() )
    {
        diskCacheKey = constructDiskCacheKey( hashOnPlaintext, functionName );
        if( CanonicalProgram* cached_cp = loadCanonicalProgramFromDiskCache( diskCacheKey, hashOnPlaintext ) )
        {
            // Place the CanonicalProgram in the memory cache
            ptxModule->m_canonicalPrograms.emplace( functionName, cached_cp );
            ureport2( m_context->getUsageReport(), "INFO" ) << "Program cache HIT  : " << functionName << std::endl;
            return cached_cp;
        }
        ureport2( m_context->getUsageReport(), "INFO" ) << "Program cache MISS : " << functionName << std::endl;
    }

    //
    // Step 3: If we do not even have the PTX, then parse it.
    //
    if( !ptxModule->m_rawModule )
    {
        TIMEVIZ_SCOPE( "PTXtoLLVM" );
        // LLVM module not present - colwert PTX and add it

        // Build header string.  Because of rti_comment this must be done
        // for each ptx input.
        // TODO: rti_comment is ignored now (OP-269).
        std::string headers = createPTXHeaderString( ptxStrings );

        // Colwert to LLVM. LLVM Bitness will always match the CPU bitness,
        bool      parseLineNumbers = k_parseLineNumbers.get();
        PTXtoLLVM ptx2llvm( m_context->getLLVMManager()->llvmContext(), &m_context->getLLVMManager()->llvmDataLayout() );

        std::string moduleName = filenames[0];
        if( filenames.size() > 1 )
            moduleName += " (multiple input)";

        ptxModule->m_rawModule = ptx2llvm.translate( moduleName, headers, ptxStrings, parseLineNumbers, functionName );
    }


    // Step 4: Canonicalize the function
    llvm::Module*   rawModule = ptxModule->m_rawModule;
    llvm::Function* fn        = findFunction( rawModule, functionName );

    // SGP: really? What happens if a user gives us a program called traverse?
    CanonicalizationType type =
        fn->getName().startswith( "traverse" ) ? CanonicalizationType::CT_TRAVERSER : CanonicalizationType::CT_PTX;

    // setting this module flag prevents the BitcodeWriter from stripping metadata (which caused programs read from cache to lose lineinfo)
    if( !rawModule->getModuleFlag( "Debug Info Version" ) )
    {
        rawModule->addModuleFlag( llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION );
    }

    CanonicalProgram* cp = canonicalizeFunction( fn, type, ptxModule->m_targetMin, ptxModule->m_targetMax, ptxModule->m_hash );
    ptxModule->m_canonicalPrograms.emplace( functionName, cp );

    // Step 5: Save the canonicalized version to the cache
    if( useDiskCache && m_context->getDiskCache()->isActive() && k_cacheEnabled.get() )
    {
        saveCanonicalProgramToDiskCache( diskCacheKey, cp );
    }

    return cp;
}

llvm::Function* ProgramManager::findFunction( llvm::Module* module, const std::string& functionName ) const
{
    // Usually we can find the function name directly
    if( llvm::Function* fn = module->getFunction( functionName ) )
        return fn;


    // Otherwise, try a mangled name, but we need to ignore the call signature, so only check the prefix.
    std::string mangled = PTXNamespaceMangle( functionName, true, true, "" );
    for( llvm::Module::iterator I = module->begin(), IE = module->end(); I != IE; ++I )
    {
        // Compare the mangled prefix name to the function name
        if( I->getName().startswith( mangled ) )
            return &*I;
    }

    // We cannot find the function.
    lerr << "Function '" << functionName << "' not found in PTX\n";
    lerr << "list of functions: \n";
    for( const llvm::Function& I : *module )
    {
        std::string f = I.getName();
        lerr << "\t" << f << "\n";
    }
    throw IlwalidSource( RT_EXCEPTION_INFO, std::string( "Function '" ) + functionName + "' not found in PTX" );
}


std::string ProgramManager::constructDiskCacheKey( size_t hash, const std::string& functionName ) const
{
    std::ostringstream cachedname;
    cachedname << "cp-" << (void*)hash << "-" << functionName << "-v2";
    llog( 13 ) << "DiskCache: Canonical program cache tag: " << cachedname.str() << '\n';
    return cachedname.str();
}

CanonicalProgram* ProgramManager::loadCanonicalProgramFromDiskCache( const std::string& cachekey, size_t hash )
{
    // Look for a disk cache entry
    std::unique_ptr<optix::PersistentStream> stream;
    optix_exp::ErrorDetails                  errDetails;
    if( m_context->getDiskCache()->find( cachekey, stream, m_context->getDeviceContextLogger(), errDetails ) )
    {
        llog( 2 ) << "DiskCache: " << errDetails.m_description;
        return nullptr;
    }
    if( !stream )
        return nullptr;
    RT_ASSERT( stream->reading() );

    TIMEVIZ_SCOPE( "CP cache read" );

    // Read the encrypted canonical program as BLOB from the cache into a memory buffer
    MemoryReader buffer;
    readOrWrite( stream.get(), &buffer, "encryptedCp" );

    // Decrypt the canonical program in the memory buffer
    tea_decrypt( reinterpret_cast<unsigned char*>( buffer.getBuffer() ), buffer.getBufferSize(), m_teaKey );

    // Deserialize the decrypted canonical program from the memory buffer
    std::unique_ptr<CanonicalProgram> cp( new CanonicalProgram( m_context, hash ) );
    readOrWrite( &buffer, cp.get(), "cp" );
    if( buffer.error() )
    {
        llog( 13 ) << "DiskCache: Canonical program cache miss due to failed deserialization\n";
        return nullptr;
    }

    ObjectManager*                        om = m_context->getObjectManager();
    std::vector<const VariableReference*> varrefs;
    for( const VariableReference* varref : cp->getAttributeReferences() )
    {
        VariableReference* mutableVarRef = const_cast<VariableReference*>( varref );
        mutableVarRef->m_variableToken   = om->registerVariableName( varref->getInputName() );
        registerVariableReference( mutableVarRef );
    }
    for( const VariableReference* varref : cp->getVariableReferences() )
    {
        VariableReference* mutableVarRef = const_cast<VariableReference*>( varref );
        mutableVarRef->m_variableToken   = om->registerVariableName( varref->getInputName() );
        registerVariableReference( mutableVarRef );
    }
    for( const CallSiteIdentifier* csId : cp->getCallSites() )
    {
        CallSiteIdentifier* mutableCsId = const_cast<CallSiteIdentifier*>( csId );
        registerCallSite( mutableCsId );
    }

    // Object is valid - take ownership, create the ID and send the
    // create event.
    CanonicalProgram* c = cp.release();
    m_canonicalPrograms.push_back( c );
    c->m_id = createCanonicalProgramId( c );
    m_context->getUpdateManager()->eventCanonicalProgramCreate( c );

    llog( 13 ) << "DiskCache: Canonical program cache hit cp: " << c->getID() << '\n';
    return c;
}

void ProgramManager::saveCanonicalProgramToDiskCache( const std::string& cacheKey, CanonicalProgram* cp )
{
    std::unique_ptr<PersistentStream> stream;
    optix_exp::ErrorDetails           errDetails;
    if( m_context->getDiskCache()->insert( cacheKey, stream, errDetails ) )
    {
        llog( 2 ) << "DiskCache: " << errDetails.m_description;
        return;
    }
    if( !stream )  // possible if cache is inactive
        return;

    // Save to disk cache
    RT_ASSERT( !stream->reading() );
    TIMEVIZ_SCOPE( "CP cache write" );
    llog( 13 ) << "DiskCache: Canonical program cache write\n";

    // Serialize the canonical program into a memory buffer
    MemoryWriter buffer;
    readOrWrite( &buffer, cp, "cp" );

    // Encrypt the serialized canonical program in the memory buffer
    size_t bufferSize = buffer.getBufferSize();
    tea_encrypt( reinterpret_cast<unsigned char*>( buffer.getBuffer() ), bufferSize, m_teaKey );

    // Write the encrypted canonical program as BLOB from the memory buffer into the cache
    readOrWrite( stream.get(), &buffer, "encryptedCp" );
    stream->flush( m_context->getDeviceContextLogger() );
}

void ProgramManager::resetVariableReferences( unsigned int beginReferenceID )
{
    for( unsigned int refid = beginReferenceID; refid < m_variableReferences.size(); ++refid )
    {
        const VariableReference* varref_toremove = m_variableReferences[refid];
        unsigned short           token           = varref_toremove->getVariableToken();
        auto&                    vec             = *m_ilwerseReferences[token];
        auto                     it = std::remove( vec.begin(), vec.end(), varref_toremove->getReferenceID() );
        m_variableReferencesByUniqueName.erase( varref_toremove->getUniversallyUniqueName() );
        vec.erase( it, vec.end() );
    }
    m_variableReferences.resize( beginReferenceID );
}

const CanonicalProgram* ProgramManager::getCanonicalProgramById( CanonicalProgramID id ) const
{
    return m_idMap.get( id );
}

const VariableReference* ProgramManager::getVariableReferenceById( VariableReferenceID id ) const
{
    RT_ASSERT( id < m_variableReferences.size() );
    return m_variableReferences[id];
}

const VariableReference* optix::ProgramManager::getVariableReferenceByUniversallyUniqueName( const std::string& uuname ) const
{
    auto it = m_variableReferencesByUniqueName.find( uuname );
    RT_ASSERT_MSG( it != m_variableReferencesByUniqueName.end(),
                   "Unknown variable reference from universally unique name: " + uuname );
    return it->second;
}

const ProgramManager::VariableReferenceIDListType& ProgramManager::getReferencesForVariable( unsigned short token ) const
{
    if( token >= m_ilwerseReferences.size() || !m_ilwerseReferences[token] )
        return m_emptyList;
    else
        return *m_ilwerseReferences[token];
}

int ProgramManager::numberOfAssignedReferences()
{
    return m_variableReferences.size();
}

ReusableID ProgramManager::createCanonicalProgramId( CanonicalProgram* program )
{
    ReusableID program_id = m_idMap.insert( program );
    return program_id;
}

const VariableReference* ProgramManager::registerVirtualVariableReference( const ProgramRoot& root, const VariableReference* linked_vref )
{
    RT_ASSERT_MSG( linked_vref->m_linkedReference == nullptr, "Virtual reference ceated for virtual reference" );
    // WARNING: this will leak reference IDs. Revisit if bound callable
    // programs are common. To solve it, make m_variableReference a
    // ReusableIDMap.
    VariableReference* new_vref = new VariableReference( *linked_vref );
    new_vref->m_linkedReference = linked_vref;
    registerVariableReference( new_vref );

    new_vref->m_root = root;

    // Note: virtual references never appear in LLVM so they do not
    // have a UUName -> varref mapping.

    return new_vref;
}

void ProgramManager::removeVirtualVariableReference( const VariableReference* vref )
{
    RT_ASSERT_MSG( vref->m_linkedReference != nullptr, "Attempting to destroy non-virtual reference" );

    m_variableReferences[vref->m_refid] = nullptr;

    // This is inefficient. We expect program references to be rare.
    unsigned short token = vref->getVariableToken();
    auto&          array = *m_ilwerseReferences[token];
    auto           iter  = algorithm::find( array, vref->m_refid );
    RT_ASSERT( iter != array.end() );
    array.erase( iter );

    delete vref;
}

void ProgramManager::registerVariableReference( VariableReference* varref )
{
    varref->m_refid = m_variableReferences.size();
    m_variableReferences.push_back( varref );

    const std::string& uniqueName = varref->getUniversallyUniqueName();
    auto               inserted   = m_variableReferencesByUniqueName.emplace( uniqueName, varref );
    if( !inserted.second )
    {
        RT_ASSERT( varref->m_linkedReference );
    }

    unsigned short token = varref->getVariableToken();

    if( token >= m_ilwerseReferences.size() )
        m_ilwerseReferences.resize( token + 1, nullptr );
    if( !m_ilwerseReferences[token] )
        m_ilwerseReferences[token] = new std::vector<VariableReferenceID>();
    m_ilwerseReferences[token]->push_back( varref->getReferenceID() );
}

void ProgramManager::registerCallSite( CallSiteIdentifier* csId )
{
    const std::string& uniqueName = csId->getUniversallyUniqueName();
    auto               inserted   = m_callSiteByUniqueName.emplace( uniqueName, csId );
    RT_ASSERT( inserted.second );
}

CallSiteIdentifier* ProgramManager::getCallSiteByUniqueName( const std::string& csName ) const
{
    auto it = m_callSiteByUniqueName.find( csName );
    if( it == m_callSiteByUniqueName.end() )
        return nullptr;  // To be handled at call site
    return it->second;
}

unsigned ProgramManager::registerFunctionType( llvm::FunctionType* ftype )
{
    // Print function signature into string
    std::string              str;
    llvm::raw_string_ostream rso( str );
    ftype->print( rso );
    rso.flush();

    // Hash and truncate to 24 bits to fit into VariableType
    unsigned token = llvm::djbHash( str ) & 0x00FFFFFF;

    m_functionSignatures.insert( std::make_pair( token, str ) );

    // nobody needs a notification of a new FunctionType token
    return token;
}

std::string ProgramManager::getFunctionSignatureForToken( unsigned token ) const
{
    auto it = m_functionSignatures.find( token );
    if( it != m_functionSignatures.end() )
        return it->second;

    return std::string();
}

ProgramManager::PTXModule::PTXModule( size_t hash, ComputeCapability targetMin, ComputeCapability targetMax )
    : m_hash( hash )
    , m_targetMin( targetMin )
    , m_targetMax( targetMax )
{
}

static void updateBindlessBufferPolicies( Context* context )
{
    // Update ALL bindless buffers' policies
    for( const auto& buf : context->getObjectManager()->getBuffers() )
    {
        if( buf->isBindless() )
            buf->hasRawAccessDidChange();
    }
}

bool ProgramManager::hasRawBindlessBufferAccesses() const
{
    return m_rawBindlessBufferAccesses != 0;
}

void ProgramManager::addOrRemoveRawBindlessBufferAccesses( bool added )
{
    bool changed = false;
    if( added )
    {
        m_rawBindlessBufferAccesses++;
        if( m_rawBindlessBufferAccesses == 1 )
            changed = true;
    }
    else
    {
        RT_ASSERT( m_rawBindlessBufferAccesses != 0 );
        m_rawBindlessBufferAccesses--;
        if( m_rawBindlessBufferAccesses == 0 )
            changed = true;
    }

    if( changed )
        updateBindlessBufferPolicies( m_context );
}
