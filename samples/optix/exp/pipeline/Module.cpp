/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <exp/builtinIS/BuiltinISCompileTimeConstants.h>
#include <exp/builtinIS/LwrvePtx.h>
#include <exp/builtinIS/SpherePtx.h>
#include <exp/context/DeviceContext.h>
#include <exp/context/DiskCache.h>
#include <exp/context/EncryptionManager.h>
#include <exp/context/ErrorHandling.h>
#include <exp/context/OptixResultOneShot.h>
#include <exp/functionTable/compileOptionsTranslate.h>
#include <exp/pipeline/Module.h>
#include <exp/pipeline/NopIS_ptx_bin.h>
#include <exp/pipeline/ReadLWVMContainer.h>

#include <prodlib/misc/LWTXProfiler.h>

#include <rtcore/interface/types.h>

#include <FrontEnd/PTX/DataLayout.h>
#include <FrontEnd/PTX/PTXHeader.h>
#include <FrontEnd/PTX/PTXtoLLVM.h>

// Needed to get the driver version
#include <Util/LWML.h>
#include <Util/optixUuid.h>
#include <Util/ProgressiveHash.h>
#include <Util/PersistentStream.h>

#include <corelib/compiler/LLVMUtil.h>
#include <corelib/math/MathUtil.h>
#include <corelib/misc/String.h>
#include <corelib/system/Timer.h>
#include <prodlib/exceptions/Exception.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/exceptions/IlwalidSource.h>
#include <prodlib/exceptions/RTCoreError.h>
#include <prodlib/misc/HostStopwatch.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/System.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <queue>
#include <sstream>

static const unsigned int* s_optixUUID = optix::getOptixUUID();
std::atomic_int optix_exp::Module::s_serializedModuleId( 1 );

extern "C" {
OptixResult optixTaskExelwte( OptixTask task, OptixTask* additionalTasks, unsigned int maxNumAdditionalTasks, unsigned int* numAdditionalTasksCreated );
}

namespace {
// clang-format off
PublicKnob<bool>   k_cacheEnabled( RT_PUBLIC_DSTRING( "diskcache.enabled" ), true, RT_PUBLIC_DSTRING( "Enable / disable the on-disk compilation cache" ) );
Knob<std::string>  k_disableNoinlineFunc( RT_DSTRING( "o7.compile.disableNoinlineFunc" ), "", RT_DSTRING( "Comma separated list of LWCA functions (substring matching) for which not to enable calling OptiX API functions. Only in effect when Noinline compilation is enabled." ) );
HiddenPublicKnob<bool>         k_allowRawLLVMBitcodeInput( RT_PUBLIC_DSTRING( "compile.allowLLVMInput" ), false, RT_PUBLIC_DSTRING( "Allow input to be raw LLVM bitcode outside of LWVMIR containers." ) );
Knob<bool>         k_allowZeroEntryFunctions( RT_DSTRING( "o7.compile.allowZeroEntryFunctions" ), false, RT_DSTRING( "Allow for zero entry point functions in the input" ) );
Knob<unsigned int> k_splitModuleMaxNumBins( RT_DSTRING( "o7.splitModuleMaxNumBins" ), 0, RT_DSTRING( "Maximum number of SubModules to split the module into (0 means number of CPUs)" ) );
Knob<bool>         k_serializeModuleId( RT_DSTRING( "o7.compile.serializeModuleId" ), false, RT_DSTRING( "When creating mangled names, use a serial number instead of a hash of the input. Useful for diffing runs." ) );
Knob<bool>         k_sphereLowMem( RT_DSTRING( "o7.sphereLowMem" ), true, RT_DSTRING( "Low memory spheres." ) );
// clang-format on
}  // namespace

extern "C" OptixResult optixModuleDestroy( OptixModule moduleAPI );

namespace optix_exp {

void readOrWrite( optix::PersistentStream* stream, SubModule::EntryFunctionInfo* entryFunctionInfo, const char* label )
{
    auto tmp = stream->pushObject( label, "entryFunctionInfo" );
    readOrWrite( stream, &entryFunctionInfo->m_traceCallCount, "traceCallCount" );
    readOrWrite( stream, &entryFunctionInfo->m_continuationCallableCallCount, "continuationCallableCallCount" );
    readOrWrite( stream, &entryFunctionInfo->m_directCallableCallCount, "directCallableCallCount" );
    readOrWrite( stream, &entryFunctionInfo->m_basicBlockCount, "basicBlockCount" );
    readOrWrite( stream, &entryFunctionInfo->m_instructionCount, "instructionCount" );
}

void readOrWrite( optix::PersistentStream* stream, SubModule::NonEntryFunctionInfo* nonEntryFunctionInfo, const char* label )
{
    auto tmp = stream->pushObject( label, "entryFunctionInfo" );
    readOrWrite( stream, &nonEntryFunctionInfo->m_count, "directCallableCallCount" );
    readOrWrite( stream, &nonEntryFunctionInfo->m_basicBlockCount, "basicBlockCount" );
    readOrWrite( stream, &nonEntryFunctionInfo->m_instructionCount, "instructionCount" );
}

void readOrWrite( optix::PersistentStream* stream, EntryFunctionSemantics* entryFunctionSemantics, const char* label )
{
    readOrWrite( stream, &entryFunctionSemantics->m_payloadTypeMask, "payloadTypeMask" );
}

std::string SubModule::getSymbolTypeString( ModuleSymbolType type )
{
    switch( type )
    {
        case ModuleSymbolType::DATA:
            return "variable";
        case ModuleSymbolType::FUNCTION:
            return "function";
    }
    return "invalid symbol type";
}

OptixResult SubModule::destroy( ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    if( m_rtcModule )
        result += m_rtcModule->destroy( m_parentModule->getDeviceContext(), errDetails );
    return result;
}

void SubModule::registerEntryFunction( const std::string&            unmangledName,
                                       EntryFunctionInfo&&           entryFunctionInfo,
                                       const EntryFunctionSemantics& entryFunctionSemantics,
                                       SemanticType                  stype )
{
    if( entryFunctionSemantics.m_payloadTypeMask == 0 )
    {
        const std::string& mangledName   = m_parentModule->getMangledName( unmangledName, OPTIX_PAYLOAD_TYPE_DEFAULT, stype );
        m_entryFunctionInfo[mangledName] = entryFunctionInfo;
    }
    else
    {
        // register the function for every supported payload type
        unsigned int       payloadTypeMask = entryFunctionSemantics.m_payloadTypeMask;
        OptixPayloadTypeID lwrrentID       = OPTIX_PAYLOAD_TYPE_ID_0;
        while( payloadTypeMask )
        {
            if( payloadTypeMask & 0x1 )
            {
                const std::string& mangledName   = m_parentModule->getMangledName( unmangledName, lwrrentID, stype );
                m_entryFunctionInfo[mangledName] = entryFunctionInfo;
            }
            lwrrentID = ( OptixPayloadTypeID )( (unsigned int)lwrrentID << 1 );
            payloadTypeMask >>= 1;
        };
    }

    m_entryFunctionSemantics[unmangledName] = entryFunctionSemantics;
}

void SubModule::setNonEntryFunctionInfo( NonEntryFunctionInfo&& nonEntryFunctionInfo )
{
    m_nonEntryFunctionInfo = std::move( nonEntryFunctionInfo );
}

void SubModule::addExportedDataSymbol( const std::string& symbolName, size_t symbolSize )
{
    m_exportedSymbols.insert( std::make_pair( symbolName, ModuleSymbol{ symbolSize, ModuleSymbolType::DATA } ) );
}

void SubModule::addImportedDataSymbol( const std::string& symbolName, size_t symbolSize )
{
    m_importedSymbols.insert( std::make_pair( symbolName, ModuleSymbol{ symbolSize, ModuleSymbolType::DATA } ) );
}

void SubModule::addExportedFunctionSymbol( const std::string& symbolName, size_t symbolSize )
{
    m_exportedSymbols.insert( std::make_pair( symbolName, ModuleSymbol{ symbolSize, ModuleSymbolType::FUNCTION } ) );
}

void SubModule::addImportedFunctionSymbol( const std::string& symbolName, size_t symbolSize )
{
    m_importedSymbols.insert( std::make_pair( symbolName, ModuleSymbol{ symbolSize, ModuleSymbolType::FUNCTION } ) );
}

OptixResult SubModule::saveToStream( optix::PersistentStream* stream, const char* label, ErrorDetails& errDetails )
{
    auto tmp = stream->pushObject( label, "subModule" );
    readOrWrite( stream, &m_nonEntryFunctionInfo, "nonEntryFunctionInfo" );
    readOrWrite( stream, &m_entryFunctionInfo, "entryFunctionInfo" );
    readOrWrite( stream, &m_exportedSymbols, "exportedSymbols" );
    readOrWrite( stream, &m_importedSymbols, "importedSymbols" );
    readOrWrite( stream, &m_usesTextureIntrinsic, "usesTextureIntrinsic" );
    readOrWrite( stream, &m_entryFunctionSemantics, "entryFunctionSemantics" );
    readOrWrite( stream, &m_mangledEntryFunctionToProgramIndex, "mangledEntryFunctionToProgramIndex" );

    // Get module from rtcore
    DeviceContext* context = m_parentModule->getDeviceContext();
    Rtlw64 blobSize = 0;
    if( context->getRtcore().compiledModuleGetCachedBlob( m_rtcModule->m_rtcModule, 0, nullptr, &blobSize ) )
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA,
                                      "Couldn't determine module cache blob "
                                      "size" );
    std::vector<char> blob( blobSize );
    Rtlw64            checkSize = 0;
    if( context->getRtcore().compiledModuleGetCachedBlob( m_rtcModule->m_rtcModule, blobSize, blob.data(), &checkSize ) )
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA, "Couldn't fetch module cache blob" );
    if( blobSize != checkSize )
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA, "Cached module size mismatch" );
    optix::readOrWrite( stream, &blob, "blob" );

    return OPTIX_SUCCESS;
}

OptixResult SubModule::readFromStream( optix::PersistentStream* stream, const char* label, const std::string& cacheKey, ErrorDetails& errDetails )
{
    auto tmp = stream->pushObject( label, "subModule" );
    readOrWrite( stream, &m_nonEntryFunctionInfo, "nonEntryFunctionInfo" );
    readOrWrite( stream, &m_entryFunctionInfo, "entryFunctionInfo" );
    readOrWrite( stream, &m_exportedSymbols, "exportedSymbols" );
    readOrWrite( stream, &m_importedSymbols, "importedSymbols" );
    readOrWrite( stream, &m_usesTextureIntrinsic, "usesTextureIntrinsic" );
    readOrWrite( stream, &m_entryFunctionSemantics, "entryFunctionSemantics" );
    readOrWrite( stream, &m_mangledEntryFunctionToProgramIndex, "mangledEntryFunctionToProgramIndex" );

    std::vector<char> blob;
    optix::readOrWrite( stream, &blob, "blob" );

    DeviceContext* context = m_parentModule->getDeviceContext();
    LwdaContextPushPop lwCtx( context );
    if( OptixResult result = lwCtx.init( errDetails ) )
        return result;

    RtcCompiledModule compiledModule;
    if( const RtcResult res = context->getRtcore().compiledModuleFromCachedBlob( context->getRtcDeviceContext(), blob.data(),
                                                                                 blob.size(), &compiledModule ) )
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA,
                                      "Failed to compile module from cache data for key " + cacheKey );

    if( OptixResult result = lwCtx.destroy( errDetails ) )
        return result;

    m_rtcModule.reset( new RtcoreModule( context, compiledModule ) );

    return OPTIX_SUCCESS;
}

Module::Module( DeviceContext* context, std::string&& ptxHash )
    : OpaqueApiObject( OpaqueApiObject::ApiType::Module )
    , m_context( context )
    , m_ptxHash( std::move( ptxHash ) )
    , m_moduleId( 0 ) // id not needed in the disk cache, only for the key.
    , m_hasDebugInformation( false )
{
}

Module::Module( DeviceContext*              context,
                InternalCompileParameters&& compileParams,
                bool                        decryptInput,
                const char*                 input,
                size_t                      inputSize,
                char*                       logString,
                size_t*                     logStringSize )
    : OpaqueApiObject( OpaqueApiObject::ApiType::Module )
    , m_context( context )
    , m_moduleId( s_serializedModuleId++ )
    , m_compileState( new std::atomic<OptixModuleCompileState>( OPTIX_MODULE_COMPILE_STATE_NOT_STARTED ) )
    , m_initialTask( new InitialCompileTask( this, std::move( compileParams ), decryptInput, input, inputSize ) )
    , m_numSubModulesLeft( new std::atomic<int>( 0 ) )
    , m_numSubModulesLwrrentlyActive( new std::atomic<int>( 0 ) )
    , m_taskLogLock( new std::mutex() )
    , m_logString( logString )
    , m_logStringSize( logStringSize )
    , m_logStringMemSize( logStringSize ? *logStringSize : 0 )
    , m_hasDebugInformation( false )
{
}

OptixResult Module::moveAssignFromCache( std::unique_ptr<Module>& otherModule, ErrorDetails& errDetails )
{
    m_ptxHash = std::move( otherModule->m_ptxHash );
    // m_moduleId - keep ID from original Module
    m_pipelineParamsSize = std::move( otherModule->m_pipelineParamsSize );
    m_compileParameters = std::move( otherModule->m_compileParameters );
    // m_compileState
    // m_initialTask
    // m_taskLogLock
    // m_taskLogsWithErrors
    // m_taskLogsWithoutErrors
    // m_logString
    // m_logStringSize
    // m_logStringMemSize
    m_mangledEntryFunctionNameToSubModule = std::move( otherModule->m_mangledEntryFunctionNameToSubModule );
    m_entryFunctionSemantics = std::move( otherModule->m_entryFunctionSemantics );
    m_subModules = std::move( otherModule->m_subModules );
    for( SubModule* subModule : m_subModules )
        subModule->m_parentModule = this;
    m_nonEntryFunctionModule = std::move( otherModule->m_nonEntryFunctionModule );
    if( m_nonEntryFunctionModule )
        m_nonEntryFunctionModule->m_parentModule = this;
    // m_deviceContextIndex
    m_moduleIdentifier = std::move( otherModule->m_moduleIdentifier );
    m_hasDebugInformation = otherModule->hasDebugInformation();
    return OPTIX_SUCCESS;
}

OptixResult Module::destroy( bool doUnregisterModule, ErrorDetails& errDetails )
{
    OptixModuleCompileState state = m_compileState->load();
    OptixResultOneShot result;
    if( doUnregisterModule )
        result += m_context->unregisterModule( this, errDetails );
    if( m_initialTask )
        result += m_initialTask->destroy( errDetails );
    for( OptixTask taskAPI : m_subModuleCompileTasks )
    {
        Task* task = reinterpret_cast<Task*>( taskAPI );
        if( task )
        {
            result += task->destroy( errDetails );
            delete task;
        }
    }
    for( SubModule* subModule : m_subModules )
    {
        result += subModule->destroy( errDetails );
        delete subModule;
    }
    m_subModules.clear();
    if( m_nonEntryFunctionModule )
    {
        result += m_nonEntryFunctionModule->destroy( errDetails );
        delete m_nonEntryFunctionModule;
    }

    return result;
}

const SubModule* Module::getSubModule( const char* mangledName ) const
{
    const auto& it = m_mangledEntryFunctionNameToSubModule.find( std::string( mangledName ) );
    assert( it != m_mangledEntryFunctionNameToSubModule.end() );
    return m_subModules[it->second];
}

std::vector<const SubModule*> Module::getSubModuleAndDependencies( const char* mangledName ) const
{
    // For now return all the SubModule objects, since we don't partition them in any
    // particular way.
    std::vector<const SubModule*> subModules( m_subModules.begin(), m_subModules.end() );
    return subModules;
}

void Module::addSubModule( SubModule* subModule )
{
    subModule->m_moduleIndex = static_cast<int>( m_subModules.size() );
    m_subModules.push_back( subModule );
}

void Module::addSubModuleCompilationTask( Task* task )
{
    m_subModuleCompileTasks.push_back( optix_exp::apiCast( task ) );
    (*m_numSubModulesLeft)++;
}

bool Module::hasEntryFunction( const std::string& unmangledName ) const
{
    return m_entryFunctionSemantics.count( unmangledName ) > 0;
}

EntryFunctionSemantics Module::getEntryFunctionSemantics( const std::string& unmangledName ) const
{
    const auto& it = m_entryFunctionSemantics.find( unmangledName );
    return it != m_entryFunctionSemantics.end() ? it->second : EntryFunctionSemantics();
}

unsigned int Module::getCompatiblePayloadTypeId( const CompilePayloadType& type, unsigned int optixPayloadTypeMask ) const
{
    for( int i = 0; i < m_compileParameters.payloadTypes.size(); ++i )
    {
        OptixPayloadTypeID payloadTypeId = (OptixPayloadTypeID)(1 << i);
        if( optixPayloadTypeMask & payloadTypeId )
        {
            if( m_compileParameters.payloadTypes[i] == type )
                return payloadTypeId;
        }
    }
    return OPTIX_PAYLOAD_TYPE_DEFAULT;
}

const CompilePayloadType* Module::getPayloadTypeFromId( unsigned int typeId ) const
{
    int i = 0;
    for( ; typeId != 0 && (typeId & 1) == 0 ; typeId >>= 1, i++ );

    if( typeId == 0x1 )
        return &m_compileParameters.payloadTypes[i];

    return nullptr;
}

std::string Module::getMangledName( const llvm::StringRef& name, unsigned int optixPayloadTypeID, SemanticType stype ) const
{
    if( stype == ST_ILWALID )
        stype = optix_exp::getSemanticTypeForFunctionName( name, m_context->isNoInlineEnabled(), k_disableNoinlineFunc.get() );
    if( stype == ST_NOINLINE )
        return name;

    std::string mangledName = name.str();

    if( optixPayloadTypeID != 0 )
    {
        const CompilePayloadType* payloadType = getPayloadTypeFromId( optixPayloadTypeID );
        if( payloadType )
            mangledName = mangledName + "_ptID" + payloadType->mangledName;
    }

    if( k_serializeModuleId.get() )
        mangledName = mangledName + "_ID" + std::to_string( m_moduleId );
    else if( !m_ptxHash.empty() )
        mangledName = mangledName + "_" + m_ptxHash;

    return mangledName;
}

OptixResult Module::getRtcCompiledModuleAndProgramIndex( const std::string& mangledName,
                                                         RtcCompiledModule& rtcModule,
                                                         Rtlw32&            programIndex,
                                                         ErrorDetails&      errDetails ) const
{
    OptixResultOneShot result;

    const SubModule* subModule = getSubModule( mangledName.c_str() );
    auto             iter      = subModule->m_mangledEntryFunctionToProgramIndex.find( mangledName );
    if( iter == subModule->m_mangledEntryFunctionToProgramIndex.end() )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Program name was not found in module" );
    rtcModule    = subModule->getRtcCompiledModule();
    programIndex = iter->second;

    return result;
}

void Module::setModuleCompletionData( bool decryptInput, std::string&& encryptedCacheKey, std::string&& cacheKey )
{
    m_completionData.m_decryptInput      = decryptInput;
    m_completionData.m_encryptedCacheKey = std::move( encryptedCacheKey );
    m_completionData.m_cacheKey          = std::move( cacheKey );
}

void Module::aggregateSubModuleData()
{
    for( size_t i = 0; i < m_subModules.size(); ++i )
    {
        // assert( this == subModule->m_parent );
        for( const auto& nameInfoPair : m_subModules[i]->m_entryFunctionInfo )
        {
            const std::string& mangledName = nameInfoPair.first;
            m_mangledEntryFunctionNameToSubModule.emplace( mangledName, i );
        }

        const auto& entryFunctionSemantics = m_subModules[i]->m_entryFunctionSemantics;
        m_entryFunctionSemantics.insert( entryFunctionSemantics.begin(), entryFunctionSemantics.end() );
    }
}

void Module::setCompileParameters( InternalCompileParameters&& compileParams )
{
    m_compileParameters = std::move( compileParams );
}

OptixResult Module::loadModuleFromDiskCache( DeviceContext*           context,
                                             const std::string&       cacheKey,
                                             std::unique_ptr<Module>& module,
                                             ErrorDetails&            errDetails )
{
    std::unique_ptr<optix::PersistentStream> stream;
    if( OptixResult result = context->getDiskCache()->find( cacheKey, stream, context->getLogger(), errDetails ) )
        return result;

    if( stream )
    {
        stream->readOrWriteObjectVersion( s_optixUUID );
        if( stream->error() )
        {
            module.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA, "Invalid OptiX version for key " + cacheKey );
        }

        stream->readOrWriteObjectVersion( context->getRtcoreUUID() );
        if( stream->error() )
        {
            module.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA, "Invalid RTX version for key " + cacheKey );
        }

        std::string ptxHash;
        optix::readOrWrite( stream.get(), &ptxHash, "ptxHash" );
        if( stream->error() )
        {
            module.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA, "Invalid PTX hash for key " + cacheKey );
        }

        size_t pipelineParamsSize = Module::s_ilwalidPipelineParamsSize;
        optix::readOrWrite( stream.get(), &pipelineParamsSize, "pipelineParamSize" );
        if( stream->error() )
        {
            module.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA,
                                          "Invalid pipeline parameter size data for key " + cacheKey );
        }

        module.reset( new Module( context, std::move( ptxHash ) ) );

        module->m_pipelineParamsSize = pipelineParamsSize;

        optix::readOrWrite( stream.get(), &module->m_moduleIdentifier, "moduleIdentifier" );
        if( stream->error() )
        {
            module->destroy( errDetails );
            module.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA,
                "Invalid module identifier data for key " + cacheKey );
        }

        bool hasDebugInformation{};
        optix::readOrWrite( stream.get(), &hasDebugInformation, "hasDebugInformation" );
        if( stream->error() )
        {
            module->destroy( errDetails );
            module.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA,
                "Invalid debug information flag for key " + cacheKey );
        }
        module->m_hasDebugInformation = hasDebugInformation;

        {
            size_t size = ~0ULL;
            optix::readOrWrite( stream.get(), &size, "size" );
            if( size == ~0ULL || size > 1000000ULL )
                stream->setError();
            if( stream->error() )
            {
                module->destroy( errDetails );
                module.reset();
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA,
                                              "Invalid number of sub modules for key " + cacheKey );
            }
            module->m_subModules.resize( size );

            OptixResultOneShot result;
            for( size_t i = 0; i < size; ++i )
            {
                SubModule* subModule = module->m_subModules[i] = new SubModule();
                subModule->m_parentModule = module.get();

                std::string label( "subModule[" + std::to_string( i ) + "]" );
                result += subModule->readFromStream( stream.get(), label.c_str(), cacheKey, errDetails );
            }
            if( result || stream->error() )
            {
                module->destroy( errDetails );
                module.reset();
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA, "Invalid subModule data for key " + cacheKey );
            }
        }

        bool hasNonEntryFunctionModule = false;
        optix::readOrWrite( stream.get(), &hasNonEntryFunctionModule, "hasNonEntryFunctionModule" );
        if( hasNonEntryFunctionModule )
        {
            module->m_nonEntryFunctionModule = new SubModule();
            module->m_nonEntryFunctionModule->m_parentModule = module.get();
            OptixResult result = module->m_nonEntryFunctionModule->readFromStream( stream.get(), "nonEntryFunctionModule", cacheKey, errDetails );
            if( result || stream->error() )
            {
                module->destroy( errDetails );
                module.reset();
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA,
                                              "Invalid non-entry function data data for key " + cacheKey );
            }

        }

        // Rebuild the aggregate data structures from the new subModules
        module->aggregateSubModuleData();
    }
    return OPTIX_SUCCESS;
}

OptixResult Module::saveModuleToDiskCache( DeviceContext*                            context,
                                           const std::string&                        cacheKey,
                                           const optix_exp::Module* module,
                                           ErrorDetails&                             errDetails )
{
    std::unique_ptr<optix::PersistentStream> stream;
    if( OptixResult result = context->getDiskCache()->insert( cacheKey, stream, errDetails ) )
        return result;

    stream->readOrWriteObjectVersion( s_optixUUID );
    stream->readOrWriteObjectVersion( context->getRtcoreUUID() );
    optix::readOrWrite( stream.get(), &module->m_ptxHash, "ptxHash" );
    optix::readOrWrite( stream.get(), &module->m_pipelineParamsSize, "pipelineParamSize" );
    optix::readOrWrite( stream.get(), &module->m_moduleIdentifier, "moduleIdentifier" );
    bool hasDebugInformation = module->hasDebugInformation();
    optix::readOrWrite( stream.get(), &hasDebugInformation, "hasDebugInformation" );

    {
        auto tmp = stream->pushObject( "subModules", "vector" );
        size_t size = module->m_subModules.size();
        optix::readOrWrite( stream.get(), &size, "size" );

        for( size_t i = 0; i < module->m_subModules.size(); ++i )
            if( OptixResult result = module->m_subModules[i]->saveToStream( stream.get(), ( "subModule[" + std::to_string( i ) + "]" ).c_str(), errDetails ) )
                return result;
    }

    bool hasNonEntryFunctionModule = module->m_nonEntryFunctionModule != nullptr;
    optix::readOrWrite( stream.get(), &hasNonEntryFunctionModule, "hasNonEntryFunctionModule" );
    if( hasNonEntryFunctionModule )
        if( OptixResult result = module->m_nonEntryFunctionModule->saveToStream( stream.get(), "nonEntryFunctionModule", errDetails ) )
            return result;

    stream->flush( context->getLogger() );
    // add logging "added module with functions ... to cache using key ..."
    return OPTIX_SUCCESS;
}

static size_t hashIlwalue( size_t hashValue, size_t seed )
{
    // Combine seed and hash, based on boost::hash_combine
    seed ^= hashValue + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 );
    return seed;
}

static std::string getFinalHashValue( size_t hashValue, size_t seed )
{
    size_t newHash = hashIlwalue( hashValue, seed );
    std::ostringstream s;
    s << "0x" << std::hex << newHash;
    return s.str();
}

// Hash the encrypted PTX streamwise and return result via OUT parameter hashValue.
static OptixResult hashEncryptedPtx( EncryptionManager& encryptionManager, const char* ptx, size_t ptxLen, size_t& hashValue, ErrorDetails& errDetails )
{
    optix::ProgressiveHash h;

    const char* encrypted    = ptx + encryptionManager.getEncryptionPrefix().size();
    size_t      encryptedLen = ptxLen - encryptionManager.getEncryptionPrefix().size();
    while( encryptedLen > 0 )
    {
        // while this is lwrrently aligned with the window size setting inside the PTX parser, ie
        //   #define OPTIX_DECRYPTED_BUFFER_SIZE 256
        // it is not required to be equal. But since XXH3 buffers internally with the same setting
        //   #define XXH3_INTERNALBUFFER_SIZE 256
        // it wouldn't make sense to set it to something smaller.
        static const unsigned int SIZE = 256;
        char                      decrypted[SIZE];
        size_t                    decryptedLen;
        size_t                    numConsumed;

        size_t windowSize{ SIZE };
        if( OptixResult result = encryptionManager.decryptString( { encrypted, encryptedLen }, decrypted, decryptedLen,
                                                                  windowSize, numConsumed, errDetails ) )
            return result;
        h.update( decrypted, decryptedLen );
        encrypted += numConsumed;
        encryptedLen -= numConsumed;
    }

    hashValue = h.digest();
    return OPTIX_SUCCESS;
}

OptixResult hashInternalCompileParameters( DeviceContext*                   context,
                                           optix::PersistentStream*         stream,
                                           const InternalCompileParameters& compileParams,
                                           const char*                      label,
                                           ErrorDetails&                    errDetails )
{
    if( !stream->hashing() )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                      "Cache key generation failed for InternalCompileParameters. Stream is not in "
                                      "hashing mode." );
    stream->pushObject( label, "InternalCompileParameters" );
    optix::readOrWrite( stream, &compileParams.maxSmVersion, "maxSmVersion" );
    optix::readOrWrite( stream, (int*)&compileParams.abiVersion, "abiVersion" );
    optix::readOrWrite( stream, &compileParams.sbtHeaderSize, "sbtHeaderSize" );
    optix::readOrWrite( stream, &compileParams.noinlineEnabled, "noinlineEnabled" );
    optix::readOrWrite( stream, &compileParams.validationModeDebugExceptions, "validationModeDebugExceptions" );

    optix::readOrWrite( stream, &compileParams.maxRegisterCount, "maxRegisterCount" );
    optix::readOrWrite( stream, &compileParams.optLevel, "optLevel" );
    optix::readOrWrite( stream, &compileParams.debugLevel, "debugLevel" );
    {
        auto tmp = stream->pushObject( "specializedLaunchParam", "CompileBoundValueEntry" );
        for( const CompileBoundValueEntry& entry : compileParams.specializedLaunchParam )
        {
            optix::readOrWrite( stream, &entry.offset, "offset" );
            optix::readOrWrite( stream, const_cast<std::vector<char>*>( &entry.value ), "value" );
            optix::readOrWrite( stream, &entry.annotation, "annotation" );
        }
    }

    optix::readOrWrite( stream, &compileParams.usesMotionBlur, "usesMotionBlur" );
    optix::readOrWrite( stream, &compileParams.traversableGraphFlags, "traversableGraphFlags" );
    optix::readOrWrite( stream, &compileParams.numAttributeValues, "numAttributeValues" );
    optix::readOrWrite( stream, &compileParams.exceptionFlags, "exceptionFlags" );

    // Field was introduced with ABI 27.
    if( context->getAbiVersion() >= OptixABI::ABI_27 )
        optix::readOrWrite( stream, &compileParams.usesPrimitiveTypeFlags, "usesPrimitiveTypeFlags" );

    optix::readOrWrite( stream, &compileParams.pipelineLaunchParamsVariableName, "pipelineLaunchParamsVariableName" );

    auto tmp = stream->pushObject( "payloadTypes", "CompilePayloadType" );
    for( const auto& type : compileParams.payloadTypes )
    {
        auto tmp = stream->pushObject( "semantics", "OptixPayloadType" );
        for( const unsigned int semantics : type.semantics )
        {
            optix::readOrWrite( stream, &semantics, "OptixPayloadSemantics" );
        }
    }
    optix::readOrWrite( stream, &compileParams.abiVariant, "abiVariant" );

    optix::readOrWrite( stream, &compileParams.callableParamRegCount, "callableParamRegCount" );
    optix::readOrWrite( stream, &compileParams.inlineCallLimitHigh, "inlineCallLimitHigh" );
    optix::readOrWrite( stream, &compileParams.inlineCallLimitLow, "inlineCallLimitLow" );
    optix::readOrWrite( stream, &compileParams.inlineInstructionLimit, "inlineInstructionLimit" );
    optix::readOrWrite( stream, &compileParams.removeUnusedNoinlineFunctions, "removeUnusedNoinlineFunctions" );
    optix::readOrWrite( stream, &compileParams.forceInlineSet, "forceInlineSet" );
    optix::readOrWrite( stream, &compileParams.disableNoinlineFunc, "disableNoinlineFunc" );
    optix::readOrWrite( stream, &compileParams.allowIndirectFunctionCalls, "allowIndirectFunctionCalls" );
    optix::readOrWrite( stream, &compileParams.disableActiveMaskCheck, "disableActiveMaskCheck" );

    optix::readOrWrite( stream, &compileParams.enableLwstomABIProcessing, "enableLwstomABIProcessing" );
    optix::readOrWrite( stream, &compileParams.numAdditionalABIScratchRegs, "numAdditionalABIScratchRegs" );
    optix::readOrWrite( stream, &compileParams.enableCoroutines, "enableCoroutines" );
    optix::readOrWrite( stream, &compileParams.enableProfiling, "enableProfiling" );
    optix::readOrWrite( stream, &compileParams.useSoftwareTextureFootprint, "useSoftwareTextureFootprint" );
    // Need to consider whether this should be part of the cache since it doesn't affect
    // the validity of a cached module (any compiled module can be interchanged with others
    // that have different values of this parameter).
    optix::readOrWrite( stream, &compileParams.splitModuleMinBinSize, "splitModuleMinBinSize" );

    optix::readOrWrite( stream, &compileParams.useD2IR, "useD2IR" );
    optix::readOrWrite( stream, &compileParams.enableCallableParamCheck, "enableCallableParamCheck" );
    optix::readOrWrite( stream, &compileParams.paramCheckExceptionRegisters, "paramCheckExceptionRegisters" );
    optix::readOrWrite( stream, &compileParams.addBuiltinPrimitiveCheck, "addBuiltinPrimitiveCheck" );
    optix::readOrWrite( stream, &compileParams.isBuiltinModule, "isBuiltinModule" );
    optix::readOrWrite( stream, &compileParams.elideUserThrow, "elideUserThrow" );
    optix::readOrWrite( stream, &compileParams.hideModule, RT_DSTRING( "hideModule" ) );
    optix::readOrWrite( stream, const_cast<std::vector<unsigned int>*>( &compileParams.privateCompileTimeConstants ),
                        "privateCompileTimeConstants" );
    // This is relevant because it changes name mangling.
    optix::readOrWrite( stream, &compileParams.serializeModuleId, "serializeModuleId" );

    return OPTIX_SUCCESS;
}

// The module cache key contains of:
// "ptx-<ptx size>-key<MD5Hasher result of: ptx hash + module compile options hash + pipeline compile options hash + internal params hash + rtcOptions hash>
//  -sm_<sm version>-rtc<TTU available>-drv<driver version>"
static OptixResult generateCacheKey( DeviceContext*                   context,
                                     const char*                      ptx,
                                     size_t                           ptxLen,
                                     const InternalCompileParameters& internalParameters,
                                     bool                             decryptInput,
                                     std::string&                     cacheKey,
                                     std::string&                     ptxHash,
                                     ErrorDetails&                    errDetails )
{
    size_t hashValue;
    if ( !decryptInput )
    {
        hashValue = corelib::hashString( ptx, ptxLen );
    }
    else
    {
        EncryptionManager& encryptionManager = context->getEncryptionManager();
        if ( OptixResult result = hashEncryptedPtx( encryptionManager, ptx, ptxLen, hashValue, errDetails ) )
            return result;
    }
    // The values in privateCompileTimeConstants can change the code in significant ways,
    // so we need to hash this into the ptxHash to get unique function names when
    // different options are specified.
    for( size_t i = 0; i < internalParameters.privateCompileTimeConstants.size(); ++i )
        hashValue = hashIlwalue( hashValue, i << 32 | internalParameters.privateCompileTimeConstants[i] );
    ptxHash = getFinalHashValue( hashValue, ptxLen );

    std::unique_ptr<optix::PersistentStream> hasher = context->getDiskCache()->createHasher();
    optix::readOrWrite( hasher.get(), &ptxHash, "ptxHash" );
    hasher->readOrWriteObjectVersion( s_optixUUID );
    hasher->readOrWriteObjectVersion( context->getRtcoreUUID() );

    if( OptixResult result = hashInternalCompileParameters( context, hasher.get(), internalParameters, "internalParams", errDetails ) )
        return result;

    std::string digest = hasher->getDigestString();

    int                smVersion = context->getComputeCapability();
    std::ostringstream diskCacheKey;
    diskCacheKey << "ptx-" << ptxLen << "-key" << digest << "-sm_" << smVersion << "-rtc" << context->hasTTU() << "-drv"
                 << optix::LWML::driverVersion();
    cacheKey = diskCacheKey.str();
    return OPTIX_SUCCESS;
}

static OptixResult getModuleFromDiskCache( DeviceContext*                     context,
                                           InternalCompileParameters&&        internalParameters,
                                           const std::string&                 cacheKey,
                                           std::unique_ptr<Module>&           cachedModule,
                                           ErrorDetails&                      errDetails )
{

    if( !k_cacheEnabled.get() || !context->isDiskCacheActive() )
    {
        cachedModule.reset();
        return OPTIX_SUCCESS;
    }

    if( OptixResult result = Module::loadModuleFromDiskCache( context, cacheKey, cachedModule, errDetails ) )
        return result;

    if( !cachedModule )
    {
        context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Print, "DISKCACHE",
                                       corelib::stringf( "Cache miss for key: %s", cacheKey.c_str() ).c_str() );
        return OPTIX_SUCCESS;
    }

    // TODO: Use higher log level for these for debugging only? Do users need that much information? Should we introduce a LOG_LEVEL::Verbose for users?
    context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Print, "DISKCACHE",
                                   corelib::stringf( "Cache hit for key: %s", cacheKey.c_str() ).c_str() );

    //if( OptixResult result = context->registerModule( cachedModule.get(), errDetails ) )
    //{
    //    cachedModule->destroy( errDetails );
    //    return result;
    //}

    // Call this only after it successfully created the module, otherwise you could "move" it and the caller would loose the data.
    cachedModule->setCompileParameters( std::move( internalParameters ) );

    return OPTIX_SUCCESS;
}

static void putModuleIntoDiskCache( DeviceContext* context, const std::string& cacheKey, const Module* module, ErrorDetails& errDetails )
{
    if( !k_cacheEnabled.get() || !context->isDiskCacheActive() )
        return;

    OptixResult r = Module::saveModuleToDiskCache( context, cacheKey, module, errDetails );
    switch( r )
    {
        case OPTIX_SUCCESS:
            context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Print, "DISKCACHE",
                                           corelib::stringf( "Inserted module in cache with key: %s", cacheKey.c_str() ).c_str() );
            break;
        case OPTIX_ERROR_DISK_CACHE_ILWALID_DATA:
        case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR:
            context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Print, "DISKCACHE", errDetails.m_description.c_str() );
            break;
        default:
            context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Print, "DISKCACHE",
                                           corelib::stringf( "Unexpected error when inserting module to cache: [%s] %s",
                                                             APIError::getErrorName( r ), errDetails.m_description.c_str() )
                                               .c_str() );
            break;
    }
}

static OptixResult createSubModuleCompilationTasks( DeviceContext*                       context,
                                                    InternalCompileParameters&&          compileParams,
                                                    optix_exp::Module*                   optixModule,
                                                    std::unique_ptr<llvm::LLVMContext>&& llvmContext,
                                                    llvm::Module*                        llvmModule,
                                                    std::string&&                        ptxHash,
                                                    unsigned int                         maxNumAdditionalTasks,
                                                    ErrorDetails&                        errDetails )
{
    // Keep all non-zero initializations of rtcOptions members in setRtcCompileOptions.
    RtcCompileOptions rtcOptions = {};
    // Run this now to catch errors before starting compilation
    if( OptixResult result = optix_exp::setRtcCompileOptions( rtcOptions, compileParams, errDetails ) )
        return result;

    // We can move this into the Module now, since we either create it or we error out. No
    // second chances.
    optixModule->setCompileParameters( std::move( compileParams ) );

    optixModule->setPtxHash( std::move( ptxHash ) );

    if( OptixResult result = optix_exp::createSubModules( optixModule, std::move( llvmContext ), llvmModule, maxNumAdditionalTasks, errDetails ) )
    {
        // Note: destroy not really needed at this point in time (module not registered
        //       yet, no rtcModule existing yet) adding it anyways to guard for future
        //       changes.
        return result;
    }

    // Create all the SubModule compilation tasks.

    for( SubModule* subModule : optixModule->getSubModules() )
        optixModule->addSubModuleCompilationTask( new Module::SubModuleCompileTask( subModule, rtcOptions ) );

    return OPTIX_SUCCESS;
}

Module::SubModuleCompileTask::SubModuleCompileTask( SubModule* subModule, const RtcCompileOptions& rtcOptions )
    : Task( subModule->m_parentModule->getDeviceContext() )
    , m_subModule( subModule )
    , m_rtcOptions( rtcOptions )
{
}

void Module::SubModuleCompileTask::logErrorDetails( OptixResult result, ErrorDetails&& errDetails )
{
    m_subModule->m_parentModule->logTaskErrorDetails( result, std::move( errDetails ) );
}

OptixResult Module::SubModuleCompileTask::exelwteImpl( OptixTask*    additionalTasksAPI,
                                                       unsigned int  maxNumAdditionalTasks,
                                                       unsigned int* numAdditionalTasksCreated,
                                                       ErrorDetails& errDetails )
{
    bool usedLWPTXFallback = false;
    if( OptixResult result = optix_exp::compileSubModule( m_subModule, usedLWPTXFallback, errDetails ) )
    {
        m_subModule->m_llvmContext.reset();
        m_subModule->m_llvmModule = nullptr;
        return result;
    }

    // Give each SubModule a chance to compile to generate errors. This emulates the old
    // behavior where as many errors as possible are generated before returning. We should
    // return OPTIX_SUCCESS here, since this SubModule was OK so far. The failed SubModule
    // will prevent this incomplete SubModule from doing anything harmful.
    OptixModuleCompileState state = m_subModule->m_parentModule->m_compileState->load();
    if( state == OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE )
    {
        m_subModule->m_llvmContext.reset();
        m_subModule->m_llvmModule = nullptr;
        return OPTIX_SUCCESS;
    }

    if( usedLWPTXFallback )
        m_rtcOptions.useLWPTX = true;

    // Compile with rtcore
    std::string serializedModuleBuffer = corelib::serializeModule( m_subModule->m_llvmModule );

    m_subModule->m_llvmContext.reset();
    m_subModule->m_llvmModule = nullptr;

    RtcCompiledModule rtcModule;

    DeviceContext* context = m_subModule->m_parentModule->getDeviceContext();
    LwdaContextPushPop lwCtx( context );
    if( OptixResult result = lwCtx.init( errDetails ) )
        return result;

    if( const RtcResult rtcResult = context->getRtcore().compileModule( context->getRtcDeviceContext(), &m_rtcOptions,
                                                                        serializedModuleBuffer.c_str(),
                                                                        serializedModuleBuffer.size(), &rtcModule ) )
    {
        OptixResult result = errDetails.logDetails( rtcResult, "Module compilation failed" );
        lwCtx.destroy( errDetails );
        return result;
    }
    // We need to create a map of mangled names to program index before we create the
    // RtcoreModule which will potentially deduplicate the module and we lose the ability
    // to generate the mapping.
    for( const auto& nameInfoPair : m_subModule->m_entryFunctionInfo )
    {
        const std::string& mangledName  = nameInfoPair.first;
        Rtlw32             programIndex = ~0;
        if( const RtcResult rtcResult =
                context->getRtcore().compiledModuleGetEntryFunctionIndex( rtcModule, mangledName.c_str(), &programIndex ) )
        {
            OptixResult result = errDetails.logDetails( rtcResult, "Failed to get entry index" );
            lwCtx.destroy( errDetails );
            if( const RtcResult rtcResult = context->getRtcore().compiledModuleDestroy( rtcModule ) )
                errDetails.logDetails( rtcResult, "Error while destroying module" );
            return result;
        }
        m_subModule->m_mangledEntryFunctionToProgramIndex.emplace( mangledName, programIndex );
    }

    m_subModule->m_rtcModule.reset( new RtcoreModule( context, rtcModule ) );

    if( OptixResult result = lwCtx.destroy( errDetails ) )
        return result;

    return OPTIX_SUCCESS;
}

OptixResult Module::SubModuleCompileTask::execute( OptixTask*    additionalTasksAPI,
                                                   unsigned int  maxNumAdditionalTasks,
                                                   unsigned int* numAdditionalTasksCreated,
                                                   ErrorDetails& errDetails )
{
    (*m_subModule->m_parentModule->m_numSubModulesLwrrentlyActive)++;

    OptixResultOneShot result;
    result += exelwteImpl( additionalTasksAPI, maxNumAdditionalTasks, numAdditionalTasksCreated, errDetails );
    if( result )
        m_subModule->m_parentModule->m_compileState->store( OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE );

    // needs to be last, since it may change the state of the module
    result += m_subModule->m_parentModule->subModuleFinished( m_subModule, errDetails );
    return result;
}

// THIS FUNCTION MUST BE THREAD SAFE FOR REENTRY
OptixResult Module::subModuleFinished( SubModule* subModule, ErrorDetails& errDetails )
{
    // These are atomic, so thread safe. Control flow is dictated by numSubModulesLeft, so
    // make sure that m_numSubModulesLeft is updated last.
    int numSubModulesLwrrentlyActive = --( *m_numSubModulesLwrrentlyActive );
    int numSubModulesLeft            = --( *m_numSubModulesLeft );

    if( numSubModulesLeft > 0 )
        return OPTIX_SUCCESS;
    if( numSubModulesLeft < 0 )
    {
        // This case should never happen, but if we find ourselves here, better to know
        // about it than to pretend everything is OK.
        //
        // Now there could be a race condition where the finishing thread could set this
        // value, but it will only set it to COMPLETED if the previous state is
        // STARTED. All other cases will be FAILED which is the same as here.
        m_compileState->store( OPTIX_MODULE_COMPILE_STATE_FAILED );
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Underflow in the number of SubModule objects left" );
    }

    // SINGLE THREADED exelwtion follows

    // Again, this case should never happen, but make sure there aren't any unexpected
    // tasks pending before we start to read state that could be set by those tasks.
    if( numSubModulesLwrrentlyActive != 0 )
    {
        m_compileState->store( OPTIX_MODULE_COMPILE_STATE_FAILED );
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                      "There are still active SubModule tasks on this Module" );
    }

    // Delete these atomics we no longer need, so they aren't held until the Module is
    // deleted for no good reason.
    m_numSubModulesLeft.reset();
    m_numSubModulesLwrrentlyActive.reset();

    // Do all the finish work, if we are in a successful state.
    OptixModuleCompileState state = m_compileState->load();

    if( state != OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE && state != OPTIX_MODULE_COMPILE_STATE_FAILED )
    {
        aggregateSubModuleData();

        // Double check if we have at least one entry point function now that we have
        // compiled all the SubModule objects.
        bool zeroEntryAllowed = getCompileParameters().isBuiltinModule || k_allowZeroEntryFunctions.get();
        if( !zeroEntryAllowed && m_mangledEntryFunctionNameToSubModule.empty() )
        {
            m_compileState->store( OPTIX_MODULE_COMPILE_STATE_FAILED );
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, "No functions with semantic types found" );
        }

        // Add it to the cache
        if( m_completionData.m_decryptInput && m_context->getEncryptionManager().isWeakVariant() )
            putModuleIntoDiskCache( m_context, m_completionData.m_encryptedCacheKey, this, errDetails );
        putModuleIntoDiskCache( m_context, m_completionData.m_cacheKey, this, errDetails );
    }

    // Only transition the state to COMPLETED if the previous state was STARTED, otherwise set
    // the state to FAILED.
    OptixModuleCompileState expected  = OPTIX_MODULE_COMPILE_STATE_STARTED;
    OptixModuleCompileState desired   = OPTIX_MODULE_COMPILE_STATE_COMPLETED;
    bool                    exchanged = m_compileState->compare_exchange_strong( expected, desired );
    if( !exchanged )
        m_compileState->store( OPTIX_MODULE_COMPILE_STATE_FAILED );

    return OPTIX_SUCCESS;
}


OptixResult createModuleWithTasks( DeviceContext*                     context,
                                   const OptixModuleCompileOptions*   moduleCompileOptions,
                                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                                   const char*                        input,
                                   size_t                             inputLen,
                                   OptixModule*                       moduleAPI,
                                   OptixTask*                         taskAPI,
                                   bool                               allowUnencryptedIfEncryptionIsEnabled,
                                   bool                               isBuiltinModule,
                                   bool                               enableLwstomPrimitiveVA,
                                   bool                               useD2IR,
                                   const std::vector<unsigned int>&   privateCompileTimeConstants,
                                   char*                              logString,
                                   size_t*                            logStringSize,
                                   ErrorDetails&                      errDetails )
{
    InternalCompileParameters compileParams;
    if( OptixResult result = setInternalCompileOptions( compileParams, moduleCompileOptions, pipelineCompileOptions,
                                                        context, isBuiltinModule, enableLwstomPrimitiveVA, useD2IR,
                                                        privateCompileTimeConstants, errDetails ) )
        return result;

    // Do we need to decrypt the data? We don't for now, since encryption can
    // only be set once per DeviceContext. We should check to see if encryption has been
    // enabled, so if it gets enabled afterward we could continue to use the unencrypted
    // path.
    EncryptionManager& encryptionManager = context->getEncryptionManager();
    bool               decryptInput      = encryptionManager.isEncryptionEnabled();

    if( decryptInput )
    {
        if( !encryptionManager.hasEncryptionPrefix( {input, inputLen} ) )
        {
            if( !allowUnencryptedIfEncryptionIsEnabled )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "Unencrypted PTX after PTX encryption has been enabled, or encrypted PTX "
                                              "has wrong prefix" );
            decryptInput = false;
        }
    }

    // Create the module.

    std::unique_ptr<Module> optixModule( new Module( context, std::move( compileParams ), decryptInput, input, inputLen, logString, logStringSize ) );
    if( OptixResult result = context->registerModule( optixModule.get(), errDetails ) )
    {
        optixModule->destroyWithoutUnregistration( errDetails );
        return result;
    }

    // Grab the initial task from the Module.
    Task* task = optixModule->getInitialTask();

    *moduleAPI = optix_exp::apiCast( optixModule.release() );
    *taskAPI   = optix_exp::apiCast( task );
    return OPTIX_SUCCESS;
}

OptixResult createModule( DeviceContext*                     context,
                          const OptixModuleCompileOptions*   moduleCompileOptions,
                          const OptixPipelineCompileOptions* pipelineCompileOptions,
                          const char*                        input,
                          size_t                             inputLen,
                          OptixModule*                       moduleAPI,
                          bool                               allowUnencryptedIfEncryptionIsEnabled,
                          bool                               isBuiltinModule,
                          bool                               enableLwstomPrimitiveVA,
                          bool                               useD2IR,
                          const std::vector<unsigned int>&   privateCompileTimeConstants,
                          char*                              logString,
                          size_t*                            logStringSize,
                          ErrorDetails&                      errDetails )
{
    OptixTask taskAPI;
    if( OptixResult result = createModuleWithTasks( context, moduleCompileOptions, pipelineCompileOptions, input,
                                                    inputLen, moduleAPI, &taskAPI, allowUnencryptedIfEncryptionIsEnabled,
                                                    isBuiltinModule, enableLwstomPrimitiveVA, useD2IR,
                                                    privateCompileTimeConstants, logString, logStringSize, errDetails ) )
        return result;

    unsigned int numTasks = k_splitModuleMaxNumBins.isSet() ? k_splitModuleMaxNumBins.get() : 1;
    if( numTasks <= 0 )
        numTasks = prodlib::getNumberOfCPUCores();
    std::vector<OptixTask> newTasks( numTasks );
    OptixResultOneShot     result;
    std::queue<OptixTask>  tasks;
    tasks.push( taskAPI );
    while( !tasks.empty() )
    {
        OptixTask task = tasks.front();
        tasks.pop();
        unsigned int numTasksCreated;
        result += optixTaskExelwte( task, newTasks.data(), newTasks.size(), &numTasksCreated );
        for( unsigned int i = 0; i < numTasksCreated; ++i )
            tasks.push( newTasks[i] );
    }

    if( result )
    {
        // Ignore return result here
        optixModuleDestroy( *moduleAPI );
        return result;
    }

    return result;
}

Module::InitialCompileTask::InitialCompileTask( Module*                     parentModule,
                                                InternalCompileParameters&& compileParams,
                                                bool                        decryptInput,
                                                const char*                 input,
                                                size_t                      inputSize )
    : Task( parentModule->getDeviceContext() )
    , m_parentModule( parentModule )
    , m_compileParams( std::move( compileParams ) )
    , m_decryptInput( decryptInput )
    , m_input( input )
    , m_inputSize( inputSize )
{
}

OptixResult Module::InitialCompileTask::exelwteImpl( OptixTask*    additionalTasksAPI,
                                                     unsigned int  maxNumAdditionalTasks,
                                                     unsigned int* numAdditionalTasksCreated,
                                                     ErrorDetails& errDetails )
{
    DeviceContext*          context           = getDeviceContext();
    EncryptionManager&      encryptionManager = context->getEncryptionManager();
    std::string             encryptedCacheKey;
    std::string             cacheKey;
    std::string             inputHash;
    std::unique_ptr<Module> cachedModule;

    if( m_decryptInput )
    {
        // In the weak variant (deterministic session key), look up module first by hash
        // over the encrypted INPUT.
        if( encryptionManager.isWeakVariant() )
        {
            std::string encryptedInputHash;
            if( OptixResult result = generateCacheKey( context, m_input, m_inputSize, m_compileParams, false,
                                                       encryptedCacheKey, encryptedInputHash, errDetails ) )
                return result;

            if( OptixResult result = getModuleFromDiskCache( context, std::move( m_compileParams ), encryptedCacheKey,
                                                             cachedModule, errDetails ) )
                return result;

            if( cachedModule )
            {
                if( OptixResult result = m_parentModule->moveAssignFromCache( cachedModule, errDetails ) )
                    return result;
                m_parentModule->m_compileState->store( OPTIX_MODULE_COMPILE_STATE_COMPLETED );
                return OPTIX_SUCCESS;
            }
        }
    }

    if( OptixResult result = generateCacheKey( context, m_input, m_inputSize, m_compileParams, m_decryptInput, cacheKey,
                                               inputHash, errDetails ) )
        return result;

    if( OptixResult result = getModuleFromDiskCache( context, std::move( m_compileParams ), cacheKey, cachedModule,
                                                     errDetails ) )
        return result;

    if( cachedModule )
    {
        if( OptixResult result = m_parentModule->moveAssignFromCache( cachedModule, errDetails ) )
            return result;
        m_parentModule->m_compileState->store( OPTIX_MODULE_COMPILE_STATE_COMPLETED );
        return OPTIX_SUCCESS;
    }

    /* input can be encrypted or not. Unencrypted input can be:
       1. lwvmc (with llvm bitcode)
       2. llvm bitcode (only for internal modules or with knob set)
       3. ptx
    */

    std::vector<char> decryptedInputBuf;
    llvm::StringRef   bitcode{m_input, m_inputSize};
    if( m_decryptInput )
    {
        size_t decryptedLen = 0;
        size_t numConsumed  = 0;
        size_t windowSize{ 16 };

        // Make room for 16 decrypted characters and a null terminator.
        decryptedInputBuf.resize( windowSize + 1 );

        // We're using decryptStringWithPrefix() here with a window size of 16 so that we get 16 *decrypted*
        // characters.  Before we were passing a StringView with only 16 *encrypted* characters to
        // decrypt(), which would fail when the last character was the first character of a
        // two-character "escape sequence", e.g. "\1\1" is used to encode '\0'.  decryptStringWithPrefix() is
        // able to look ahead in the encrypted string, so it doesn't have this problem.
        if( OptixResult result = encryptionManager.decryptStringWithPrefix( { m_input, m_inputSize }, decryptedInputBuf.data(),
                                                                            decryptedLen, windowSize, numConsumed, errDetails ) )
        {
            return result;
        }
        decryptedInputBuf.resize( decryptedLen + 1 );
        decryptedInputBuf[decryptedLen] = '\0';
        llvm::StringRef header{ decryptedInputBuf.data(), decryptedInputBuf.size() };
        if( isLWVMContainer( header )
            || llvm::isBitcode( reinterpret_cast<const unsigned char*>( header.data() ),
                                reinterpret_cast<const unsigned char*>( header.data() ) + header.size() ) )
        {
            // Decrypt the entire buffer if we are using LLVM bitcode, since we can't
            // incrementally decrypt it (unlike PTX).
            if( OptixResult result = encryptionManager.decrypt( {m_input, m_inputSize}, decryptedInputBuf, errDetails ) )
                return result;
            // decrypt adds a 0 terminator which we do not want for bitcode input (causes "bad bitcode signature").
            decryptedInputBuf.pop_back();
            bitcode = llvm::StringRef{ decryptedInputBuf.data(), decryptedInputBuf.size() };
        }
    }

    std::unique_ptr<llvm::LLVMContext> llvmContext;
    llvm::Module*                      llvmModule = nullptr;

    std::unique_ptr<llvm::MemoryBuffer> irBuffer( llvm::MemoryBuffer::getMemBuffer( bitcode, "", false /*null-terminated*/ ) );
    bool                                isLWVMContainerInput = isLWVMContainer( bitcode );
    if( isLWVMContainerInput )
    {
        if( OptixResult result = getModuleFromLWVMContainer( irBuffer, llvmModule, errDetails ) )
            return result;
        llvmContext.reset( &llvmModule->getContext() );
    }
    else
    {
        llvmContext.reset( new llvm::LLVMContext );
    }

    // Only allow bitcode parsing if the input was a LWVM container or for internal modules (or the knob is set).
    bool allowBitcode = k_allowRawLLVMBitcodeInput.get() || m_compileParams.isBuiltinModule;
    if( isLWVMContainerInput
        || ( allowBitcode
             && llvm::isBitcode( reinterpret_cast<const unsigned char*>( irBuffer->getBufferStart() ),
                                 reinterpret_cast<const unsigned char*>( irBuffer->getBufferEnd() ) ) ) )
    {
        llvm::DataLayout DL( optix::createDataLayoutForLwrrentProcess() );
        if( !llvmModule )
        {
            llvm::Expected<std::unique_ptr<llvm::Module>> ModUPtr =
                llvm::parseBitcodeFile( irBuffer->getMemBufferRef(), *llvmContext );
            if( llvm::Error E = ModUPtr.takeError() )
            {
                std::string              errMsg;
                llvm::raw_string_ostream errorStream( errMsg );
                errorStream << E;
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, "Failed to load OptiX IR: " + errMsg );
            }
            llvmModule = ModUPtr.get().release();
        }
        // Ensure that the data layout matches the one we need.
        llvmModule->setDataLayout( DL );

        optix::PTXtoLLVM        ptx2llvm( *llvmContext, &DL );
        static std::atomic<int> counter( 0 );
        std::string             moduleName = corelib::stringf( "llvmInput-module-%03d", ++counter );
        std::string             headers    = optix::retrieveOptixPTXDeclarations();
        try
        {
            // Process input module to make it work with OptiX (e.g. add the PTX headers)
            llvmModule = ptx2llvm.translate( moduleName, headers, llvmModule,
                                             m_compileParams.optLevel == OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 );
        }
        catch( const prodlib::IlwalidSource& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, std::string( "Invalid OptiX IR input: " ) + e.what() );
        }
        catch( const prodlib::CompileError& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, std::string( "Error during compilation: " ) + e.what() );
        }
        catch( const prodlib::Exception& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, std::string( "Unknown OptiX internal exception: " ) + e.what() );
        }
        catch( const std::exception& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, std::string( "Unknown std::exception: " ) + e.what() );
        }
        catch( ... )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, "Unknown error while processing OptiX IR input" );
        }
    }
    else
    {
        // PTX
        try
        {
            // PTX to LLVM
            llvm::DataLayout DL( optix::createDataLayoutForLwrrentProcess() );
            bool             parseLineNumbers = m_compileParams.debugLevel != OPTIX_COMPILE_DEBUG_LEVEL_NONE;

            optix::PTXtoLLVM        ptx2llvm( *llvmContext, &DL );
            static std::atomic<int> counter( 0 );
            std::string             dumpName   = corelib::stringf( "ptx2llvm-module-%03d", ++counter );
            std::string             moduleName = dumpName;
            // this retrieves the stored/hard-coded list of PTX declarations only, w/o any further parsing
            std::string headers = optix::retrieveOptixPTXDeclarations();

            std::vector<prodlib::StringView> inputStrings;
            if( !m_decryptInput )
            {
                inputStrings.push_back( {irBuffer->getBufferStart(), irBuffer->getBufferSize()} );
                llvmModule = ptx2llvm.translate( moduleName, headers, inputStrings, parseLineNumbers, dumpName );
            }
            else
            {
                size_t prefixSize = encryptionManager.getEncryptionPrefix().size();
                if( m_inputSize && m_inputSize <= prefixSize )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX,
                                                  std::string( "Invalid encrypted PTX input" ) );
                inputStrings.push_back( {m_input + prefixSize, m_inputSize - prefixSize} );
                // decryptString is a function pointer, ie the callback to decrypt the encrypted PTX string
                llvmModule = ptx2llvm.translate( moduleName, headers, inputStrings, parseLineNumbers, dumpName, &encryptionManager, decryptString );
            }
        }
        catch( const prodlib::IlwalidSource& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, std::string( "Invalid PTX input: " ) + e.what() );
        }
        catch( const prodlib::CompileError& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, std::string( "Error during compilation: " ) + e.what() );
        }
        catch( const prodlib::Exception& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, std::string( "Unknown OptiX internal exception: " ) + e.what() );
        }
        catch( const std::exception& e )
        {
            return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, std::string( "Unknown std::exception: " ) + e.what() );
        }
        catch( ... )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, "Unknown error while processing PTX input" );
        }

        // store presence of debug information inside the module. While the retrieval from parent module is not strictly neccessary
        // it might improve performance considerably as reads from atomics are, in comparison to writes, "free".
        if( isPtxDebugEnabled( llvmModule ) && !m_parentModule->hasDebugInformation() )
            m_parentModule->setHasDebugInformation();
    }

    // Check for available debug info in the module.
    llvm::DebugInfoFinder dif;
    dif.processModule( *llvmModule );
    if( dif.compile_unit_count() > 1 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX,
                                      "Invalid debug info in module. Only one compile unit is supported.\n" );
    if( dif.compile_unit_count() == 1 )
    {
        llvm::DICompileUnit*                   lw  = *dif.compile_units().begin();
        llvm::DICompileUnit::DebugEmissionKind dek = lw->getEmissionKind();
        if( dek == llvm::DICompileUnit::FullDebug )
        {
            // The backend will produce an error when passing in a module with full debug info
            // without specifying full debug mode. Catch this early and also avoid having the backend
            // error swallowed in release.
            if( m_compileParams.debugLevel != OPTIX_COMPILE_DEBUG_LEVEL_FULL )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX,
                                              "Optimized debugging is not supported. Module is built with full debug "
                                              "info, but requested debug level is not "
                                              "\"OPTIX_COMPILE_DEBUG_LEVEL_FULL\".\n" );
            // store presence of debug information inside the module
            m_parentModule->setHasDebugInformation();
        }
        else if( m_compileParams.debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_FULL )
            errDetails.m_compilerFeedback
                << "Warning: Requested debug level \"OPTIX_COMPILE_DEBUG_LEVEL_FULL\", but input module does not "
                   "include full debug information.\n";
        else if( m_compileParams.debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_MODERATE && dek == llvm::DICompileUnit::NoDebug )
            errDetails.m_compilerFeedback << "Warning: Requested debug level "
                                             "\"OPTIX_COMPILE_DEBUG_LEVEL_MODERATE\", but input module does not "
                                             "include any debug information.\n";
        if( m_compileParams.debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL && dek == llvm::DICompileUnit::NoDebug )
            errDetails.m_compilerFeedback << "Warning: Requested debug level "
                                             "\"OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL\", but input module does not "
                                             "include any debug information.\n";
    }
    else if( m_compileParams.debugLevel != OPTIX_COMPILE_DEBUG_LEVEL_NONE )
    {
        errDetails.m_compilerFeedback << "Warning: Requested debug level not equal to "
                                         "\"OPTIX_COMPILE_DEBUG_LEVEL_NONE\", but input module does not "
                                         "include any debug information.\n";
    }

    // If the knob is set we need to adjust the number of bins within our allotment
    if( k_splitModuleMaxNumBins.isSet() )
    {
        unsigned int numTasks = k_splitModuleMaxNumBins.get();
        if( numTasks <= 0 )
            numTasks = prodlib::getNumberOfCPUCores();
        if( numTasks < maxNumAdditionalTasks )
            maxNumAdditionalTasks = numTasks;
    }

    if( OptixResult result = createSubModuleCompilationTasks( context, std::move( m_compileParams ), m_parentModule,
                                                              std::move( llvmContext ), llvmModule,
                                                              std::move( inputHash ), maxNumAdditionalTasks, errDetails ) )
        return result;

    m_parentModule->setModuleCompletionData( m_decryptInput, std::move( encryptedCacheKey ), std::move( cacheKey ) );

    unsigned int numTasks = static_cast<unsigned int>( m_parentModule->m_subModuleCompileTasks.size() );
    if( numTasks > maxNumAdditionalTasks )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                      corelib::stringf( "Unexpectedly got more new tasks (%u) than were allowed (%u)",
                                                        numTasks, maxNumAdditionalTasks ) );
    *numAdditionalTasksCreated = numTasks;
    for( unsigned int i       = 0; i < m_parentModule->m_subModuleCompileTasks.size(); ++i )
        additionalTasksAPI[i] = m_parentModule->m_subModuleCompileTasks[i];

    return OPTIX_SUCCESS;
}

OptixResult Module::InitialCompileTask::execute( OptixTask*    additionalTasksAPI,
                                                 unsigned int  maxNumAdditionalTasks,
                                                 unsigned int* numAdditionalTasksCreated,
                                                 ErrorDetails& errDetails )
{
    m_parentModule->m_compileState->store( OPTIX_MODULE_COMPILE_STATE_STARTED );
    if( OptixResult result = exelwteImpl( additionalTasksAPI, maxNumAdditionalTasks, numAdditionalTasksCreated, errDetails ) )
    {
        m_parentModule->m_compileState->store( OPTIX_MODULE_COMPILE_STATE_FAILED );
        return result;
    }
    return OPTIX_SUCCESS;
}

void Module::InitialCompileTask::logErrorDetails( OptixResult result, ErrorDetails&& errDetails )
{
    m_parentModule->logTaskErrorDetails( result, std::move( errDetails ) );
}

void Module::logTaskErrorDetails( OptixResult result, ErrorDetails&& errDetails )
{
    // Send to the logger
    optix_exp::DeviceContextLogger& clog = getDeviceContext()->getLogger();
    if( result )
    {
        std::ostringstream compileFeedback;
        // putting error first in the logString to avoid it falling off if the buffer is too small
        compileFeedback << "COMPILE ERROR: " << errDetails.m_description << "\n";
        compileFeedback << errDetails.m_compilerFeedback.str();
        clog.callback( DeviceContextLogger::LOG_LEVEL::Error, "COMPILE FEEDBACK", compileFeedback.str().c_str() );
    }
    else
    {
        clog.callback( DeviceContextLogger::LOG_LEVEL::Print, "COMPILE FEEDBACK", errDetails.m_compilerFeedback.str().c_str() );
    }

    if( m_logString == nullptr || ( m_logStringSize && *m_logStringSize == 0 ) )
        return;

    std::lock_guard<std::mutex> lock( *m_taskLogLock.get() );

    auto& errorList = result ? m_taskLogsWithErrors : m_taskLogsWithoutErrors;
    errorList.push_back( std::move( errDetails ) );

    // Now build up the output string. Errors first. Then feedback. We rebuild the string
    // at the end of ever task, because we don't know when the user will decide to stop
    // exelwting tasks and read it.
    std::ostringstream compileFeedbackCombined;
    for( const ErrorDetails& errDetails : m_taskLogsWithErrors )
        compileFeedbackCombined << "COMPILE ERROR: " << errDetails.m_description << "\n";
    for( const ErrorDetails& errDetails : m_taskLogsWithErrors )
        compileFeedbackCombined << errDetails.m_compilerFeedback.str();
    for( const ErrorDetails& errDetails : m_taskLogsWithoutErrors )
        compileFeedbackCombined << errDetails.m_compilerFeedback.str();

    size_t newSize = m_logStringMemSize;
    optix_exp::copyCompileDetails( compileFeedbackCombined, m_logString, &newSize );
    *m_logStringSize = newSize;
}

static OptixResult getLwrveBuiltinPtx( DeviceContext*                   context,
                                       const char**                     builtinPtx,
                                       size_t&                          builtinPtxLen,
                                       unsigned int                     builtinISModuleType,
                                       unsigned int                     usesPrimitiveTypeFlags,
                                       const std::vector<unsigned int>& privateCompileTimeConstants,
                                       ErrorDetails&                    errDetails )
{
    if( builtinISModuleType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR )
    {
        if( ( usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR ) == 0 )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "Cannot create builtin module with builtinISModuleType "
                                          "OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR."
                                          "The pipeline must support OptixLwrveType "
                                          "OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR (see "
                                          "OptixPipelineCompileOptions::usesPrimitiveTypeFlags)." );
        }

        if( privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_LOWMEM] )
        {
            *builtinPtx   = optix::data::getLinearLwrveLowMemIntersectorSources()[1];
            builtinPtxLen = optix::data::getLinearLwrveLowMemIntersectorSourceSizes()[0];
        }
        else
        {
            *builtinPtx   = optix::data::getLinearLwrveIntersectorSources()[1];
            builtinPtxLen = optix::data::getLinearLwrveIntersectorSourceSizes()[0];
        }
    }
    else if( builtinISModuleType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE )
    {
        if( ( usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE ) == 0 )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "Cannot create builtin module with builtinISModuleType "
                                          "OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE."
                                          "The pipeline must support OptixLwrveType "
                                          "OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE (see "
                                          "OptixPipelineCompileOptions::usesPrimitiveTypeFlags)." );
        }


        if( privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_LOWMEM] )
        {
            *builtinPtx   = optix::data::getQuadraticLwrvePhantomLowMemIntersectorSources()[1];
            builtinPtxLen = optix::data::getQuadraticLwrvePhantomLowMemIntersectorSourceSizes()[0];
        }
        else
        {
            *builtinPtx   = optix::data::getQuadraticLwrvePhantomIntersectorSources()[1];
            builtinPtxLen = optix::data::getQuadraticLwrvePhantomIntersectorSourceSizes()[0];
        }
    }
    else if( builtinISModuleType == OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE )
    {
        if( ( usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LWBIC_BSPLINE ) == 0 )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "Cannot create builtin module with builtinISModuleType "
                                          "OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE."
                                          "The pipeline must support OptixPrimitiveTypeFlags "
                                          "OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LWBIC_BSPLINE "
                                          "(see OptixPipelineCompileOptions::usesPrimitiveTypeFlags)." );
        }

        if( privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_LOWMEM] )
        {
            *builtinPtx   = optix::data::getLwbicLwrvePhantomLowMemIntersectorSources()[1];
            builtinPtxLen = optix::data::getLwbicLwrvePhantomLowMemIntersectorSourceSizes()[0];
        }
        else
        {
            *builtinPtx   = optix::data::getLwbicLwrvePhantomIntersectorSources()[1];
            builtinPtxLen = optix::data::getLwbicLwrvePhantomIntersectorSourceSizes()[0];
        }
    }
    else if( builtinISModuleType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM )
    {
        if( ( usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM ) == 0 )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "Cannot create builtin module with builtinISModuleType "
                                          "OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM."
                                          "The pipeline must support OptixPrimitiveTypeFlags "
                                          "OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM "
                                          "(see OptixPipelineCompileOptions::usesPrimitiveTypeFlags)." );
        }

        if( privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_LOWMEM] )
        {
            *builtinPtx   = optix::data::getCatmullRomPhantomLowMemIntersectorSources()[1];
            builtinPtxLen = optix::data::getCatmullRomPhantomLowMemIntersectorSourceSizes()[0];
        }
        else
        {
            *builtinPtx   = optix::data::getCatmullRomPhantomIntersectorSources()[1];
            builtinPtxLen = optix::data::getCatmullRomPhantomIntersectorSourceSizes()[0];
        }
    }
    return OPTIX_SUCCESS;
}

static OptixResult getSphereBuiltinPtx( DeviceContext* context,
                                        const char**   builtinPtx,
                                        size_t&        builtinPtxLen,
                                        unsigned int   builtinISModuleType,
                                        unsigned int   usesPrimitiveTypeFlags,
                                        ErrorDetails&  errDetails )
{
    if( ( usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE ) == 0 )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Cannot create builtin module with builtinISModuleType "
                                      "OPTIX_PRIMITIVE_TYPE_SPHERE."
                                      "The pipeline must support OptixPrimitiveTypeFlags "
                                      "OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE "
                                      "(see OptixPipelineCompileOptions::usesPrimitiveTypeFlags)." );
    }

    *builtinPtx   = optix::data::getSphereIntersectorSources()[1];
    builtinPtxLen = optix::data::getSphereIntersectorSourceSizes()[0];

    return OPTIX_SUCCESS;
}

OptixResult builtinISModuleGet( DeviceContext*                     context,
                                const OptixModuleCompileOptions*   moduleCompileOptions,
                                const OptixPipelineCompileOptions* pipelineCompileOptions,
                                const OptixBuiltinISOptions*       builtinISOptions,
                                OptixModule*                       builtinModuleAPI,
                                bool                               allowUnencryptedIfEncryptionIsEnabled,
                                ErrorDetails&                      errDetails )
{
    unsigned int usesPrimitiveTypeFlags = 0;
    unsigned int builtinISModuleType    = (unsigned int)builtinISOptions->builtinISModuleType;

    bool enableExceptions = ( pipelineCompileOptions->exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG );

    if( context->getAbiVersion() >= OptixABI::ABI_27 )
    {
        usesPrimitiveTypeFlags = pipelineCompileOptions->usesPrimitiveTypeFlags;

        // By default support triangles and custom primitives.
        if( usesPrimitiveTypeFlags == 0 )
            usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }
    else  // ABI 26
    {
        switch( builtinISModuleType )
        {
            case 0x2521:
                builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
                break;
            case 0x2522:
                builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
                break;
            case 0x2523:
                builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE;
                break;
            default:
                builtinISModuleType = 0;
                break;
        }

        usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE
                                 | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE
                                 | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LWBIC_BSPLINE;
    }

    const char* builtinPtx    = nullptr;
    size_t      builtinPtxLen = 0;

    EncryptionManager& internalEncryptionManager = context->getInternalEncryptionManager();
    std::vector<char>  decryptedBuiltinPtxData;

    bool usesMotionBlur = builtinISOptions->usesMotionBlur && pipelineCompileOptions->usesMotionBlur;
    if( builtinISOptions->usesMotionBlur && !pipelineCompileOptions->usesMotionBlur )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Inconsistent motion blur settings for pipeline and builtin module. "
                                      "(See OptixPipelineCompileOptions::usesMotionBlur and "
                                      "OptixBuiltinISOptions::usesMotionBlur)." );
    }
    // If the build flag OPTIX_BUILD_FLAG_PREFER_FAST_BUILD is set and ABI version >= 51, a separate low memory version
    // of the intersector will be used for lwrves.
    bool builtinISLowMem = ( builtinISModuleType == OPTIX_PRIMITIVE_TYPE_SPHERE ) ?
                               k_sphereLowMem.get() :
                               ( context->getAbiVersion() >= OptixABI::ABI_51 )
                                   && ( builtinISOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_BUILD );
    bool lwrveCapsOff = ( context->getAbiVersion() >= OptixABI::ABI_54 )
                        && ( ( builtinISOptions->lwrveEndcapFlags == OPTIX_LWRVE_ENDCAP_DEFAULT )
                             && ( builtinISModuleType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR ) );

    // Lwrrently there are builtin intersectors for hair of degree 1, 2, or 3 with cirlwlar cross-section and spheres,
    // with or without motion blur.
    std::vector<unsigned int> privateCompileTimeConstants( BUILTIN_IS_COMPILE_TIME_CONSTANT_NUM );
    privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_MOTION]        = usesMotionBlur;
    privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_LOWMEM]        = builtinISLowMem;
    privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_EXCEPTION]     = enableExceptions;
    privateCompileTimeConstants[BUILTIN_IS_COMPILE_TIME_CONSTANT_LWRVE_CAPSOFF] = lwrveCapsOff;
    switch( builtinISModuleType )
    {
        case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
        case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
        case OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE:
        case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:

            if( OptixResult result = getLwrveBuiltinPtx( context, &builtinPtx, builtinPtxLen, builtinISModuleType,
                                                         usesPrimitiveTypeFlags, privateCompileTimeConstants, errDetails ) )
                return result;

            if( OptixResult result =
                    internalEncryptionManager.decrypt( {builtinPtx, builtinPtxLen}, decryptedBuiltinPtxData, errDetails ) )
                return result;

            builtinPtx    = decryptedBuiltinPtxData.data();
            builtinPtxLen = decryptedBuiltinPtxData.size();

            break;
        case OPTIX_PRIMITIVE_TYPE_SPHERE:
            if( OptixResult result = getSphereBuiltinPtx( context, &builtinPtx, builtinPtxLen, builtinISModuleType,
                                                          usesPrimitiveTypeFlags, errDetails ) )
                return result;

            if( OptixResult result =
                    internalEncryptionManager.decrypt( {builtinPtx, builtinPtxLen}, decryptedBuiltinPtxData, errDetails ) )
                return result;

            builtinPtx    = decryptedBuiltinPtxData.data();
            builtinPtxLen = decryptedBuiltinPtxData.size();

            break;
        case OPTIX_PRIMITIVE_TYPE_TRIANGLE:
            if( ( usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ) == 0 )
            {
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "Cannot create builtin module with builtinISModuleType "
                                              "OPTIX_PRIMITIVE_TYPE_TRIANGLE."
                                              "The pipeline must support OptixPrimitiveTypeFlags "
                                              "OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE (see "
                                              "OptixPipelineCompileOptions::usesPrimitiveTypeFlags)." );
            }

            builtinPtx    = optix::data::getNopISSources()[1];
            builtinPtxLen = optix::data::getNopISSourceSizes()[0];

            break;
        case OPTIX_PRIMITIVE_TYPE_LWSTOM:
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "builtinISModuleType OPTIX_PRIMITIVE_TYPE_LWSTOM is not "
                                          "a builtin primitive type." );
        default:
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Unknown builtinISModuleType." );
    }

    OptixResult result = createModule( context, moduleCompileOptions, pipelineCompileOptions, builtinPtx, builtinPtxLen,
                                       builtinModuleAPI, allowUnencryptedIfEncryptionIsEnabled,
                                       /*isBuiltinModule=*/true, /*enableLwstomPrimitiveVA=*/!builtinISLowMem,
                                       context->isD2IREnabled(), privateCompileTimeConstants,
                                       /*logString*/ nullptr, /*logStringSize*/ nullptr, errDetails );
    return result;
}

}  // end namespace optix_exp

extern "C" OptixResult optixModuleCreateFromPTX( OptixDeviceContext                 contextAPI,
                                                 const OptixModuleCompileOptions*   moduleCompileOptions,
                                                 const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                 const char*                        PTX,
                                                 size_t                             PTXsize,
                                                 char*                              logString,
                                                 size_t*                            logStringSize,
                                                 OptixModule*                       moduleAPI )
{
    prodlib::HostStopwatch stopWatch;

    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT_W_LOG_STRING();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::MODULE_CREATE_FROM_PTX );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( moduleCompileOptions );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( pipelineCompileOptions );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( PTX );
    OPTIX_CHECK_ZERO_ARGUMENT_W_LOG_STRING( PTXsize );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( moduleAPI );
    *moduleAPI = nullptr;

    try
    {
        optix_exp::ErrorDetails  errDetails;
        OptixModuleCompileOptions translatedModuleCompileOptions;
        OptixResult result = optix_exp::translateABI_OptixModuleCompileOptions( moduleCompileOptions, pipelineCompileOptions, context->getAbiVersion(),
                                                                                &translatedModuleCompileOptions, errDetails );
        if( result )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
            return result;
        }
        OptixPipelineCompileOptions translatedPipelineCompileOptions;
        result = optix_exp::translateABI_PipelineCompileOptions( pipelineCompileOptions, context->getAbiVersion(),
                                                                 &translatedPipelineCompileOptions, errDetails );
        if( result )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
            return result;
        }
        // Once the API has been ilwoked to compile a module, users can no longer change
        // the no-inline setting.
        context->makeNoInlineImmutable();

        result = createModule( context, &translatedModuleCompileOptions, &translatedPipelineCompileOptions, PTX, PTXsize,
                               moduleAPI, /*allowUnencryptedIfEncryptionIsEnabled*/ false, /*isBuiltinModule=*/false,
                               /*enableLwstomPrimitiveVA=*/false, context->isD2IREnabled(),
                               /*privateCompileTimeConstants*/ {}, logString, logStringSize, errDetails );

        if( optix_exp::Metrics* metrics = context->getMetrics() )
        {
            double duration = stopWatch.getElapsed();
            metrics->logFloat( "module_compile_from_ptx_time_ms", duration, errDetails );
        }

        if( result && !errDetails.m_description.empty() )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
        }

        return result;
    }
    OPTIX_API_EXCEPTION_CHECK_W_LOG_STRING;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixModuleCreateFromPTXWithTasks( OptixDeviceContext                 contextAPI,
                                                          const OptixModuleCompileOptions*   moduleCompileOptions,
                                                          const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                          const char*                        PTX,
                                                          size_t                             PTXsize,
                                                          char*                              logString,
                                                          size_t*                            logStringSize,
                                                          OptixModule*                       moduleAPI,
                                                          OptixTask*                         firstTaskAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT_W_LOG_STRING();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::MODULE_CREATE_FROM_PTX_WITH_TASKS );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( moduleCompileOptions );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( pipelineCompileOptions );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( PTX );
    OPTIX_CHECK_ZERO_ARGUMENT_W_LOG_STRING( PTXsize );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( moduleAPI );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( firstTaskAPI );
    *moduleAPI    = nullptr;
    *firstTaskAPI = nullptr;

    try
    {
        optix_exp::ErrorDetails   errDetails;
        OptixModuleCompileOptions translatedModuleCompileOptions;
        OptixResult result = optix_exp::translateABI_OptixModuleCompileOptions( moduleCompileOptions, pipelineCompileOptions, context->getAbiVersion(),
                                                                                &translatedModuleCompileOptions, errDetails );
        if( result )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
            return result;
        }
        OptixPipelineCompileOptions translatedPipelineCompileOptions;
        result = optix_exp::translateABI_PipelineCompileOptions( pipelineCompileOptions, context->getAbiVersion(),
                                                                 &translatedPipelineCompileOptions, errDetails );
        if( result )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
            return result;
        }
        // Once the API has been ilwoked to compile a module, users can no longer change
        // the no-inline setting.
        context->makeNoInlineImmutable();

        result = createModuleWithTasks( context, &translatedModuleCompileOptions, &translatedPipelineCompileOptions, PTX,
                                        PTXsize, moduleAPI, firstTaskAPI, /*allowUnencryptedIfEncryptionIsEnabled*/ false,
                                        /*isBuiltinModule=*/false, /*enableLwstomPrimitiveVA=*/false, context->isD2IREnabled(),
                                        /*privateCompileTimeConstants*/ {}, logString, logStringSize, errDetails );
        if( result && !errDetails.m_description.empty() )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
        }

        return result;
    }
    OPTIX_API_EXCEPTION_CHECK_W_LOG_STRING;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixModuleGetCompilationState( OptixModule moduleAPI, OptixModuleCompileState* state )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Module, module, "OptixModule" );
    SCOPED_LWTX_RANGE( module->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::MODULE_GET_COMPILATION_STATE );
    optix_exp::DeviceContextLogger& clog = module->getDeviceContext()->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( state );

    try
    {
        *state = module->getCompileState();
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixModuleDestroy( OptixModule moduleAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Module, module, "OptixModule" );
    SCOPED_LWTX_RANGE( module->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::MODULE_DESTROY );
    optix_exp::DeviceContextLogger& clog = module->getDeviceContext()->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = module->destroy( errDetails );
        if( result )
            clog.sendError( errDetails );
        if( result != OPTIX_ERROR_ILLEGAL_DURING_TASK_EXELWTE )
            delete module;
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixBuiltinISModuleGet( OptixDeviceContext                 contextAPI,
                                                const OptixModuleCompileOptions*   moduleCompileOptions,
                                                const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                const OptixBuiltinISOptions*       builtinISOptions,
                                                OptixModule*                       builtinModuleAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::BUILTIN_IS_MODULE_GET );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( moduleCompileOptions );
    OPTIX_CHECK_NULL_ARGUMENT( pipelineCompileOptions );
    OPTIX_CHECK_NULL_ARGUMENT( builtinISOptions );
    OPTIX_CHECK_NULL_ARGUMENT( builtinModuleAPI );
    *builtinModuleAPI = nullptr;

    try
    {
        optix_exp::ErrorDetails   errDetails;
        OptixModuleCompileOptions translatedModuleCompileOptions;
        OptixResult result = optix_exp::translateABI_OptixModuleCompileOptions( moduleCompileOptions, pipelineCompileOptions, context->getAbiVersion(),
                                                                                &translatedModuleCompileOptions, errDetails );
        if( result )
        {
            clog.sendError( errDetails );
            return result;
        }
        OptixPipelineCompileOptions translatedPipelineCompileOptions;
        result = optix_exp::translateABI_PipelineCompileOptions( pipelineCompileOptions, context->getAbiVersion(),
                                                                 &translatedPipelineCompileOptions, errDetails );
        if( result )
        {
            clog.sendError( errDetails );
            return result;
        }

        result = builtinISModuleGet( context, &translatedModuleCompileOptions, &translatedPipelineCompileOptions, builtinISOptions,
                                     builtinModuleAPI, /*allowUnencryptedIfEncryptionIsEnabled*/ true, errDetails );

        optix_exp::DeviceContextLogger::LOG_LEVEL level = optix_exp::DeviceContextLogger::LOG_LEVEL::Print;
        if( result )
        {
            level = optix_exp::DeviceContextLogger::LOG_LEVEL::Error;
            std::ostringstream compileFeedback2;
            // putting error first in the logString to avoid it falling off if the buffer is too small
            compileFeedback2 << "COMPILE ERROR: " << errDetails.m_description << "\n";
            compileFeedback2 << errDetails.m_compilerFeedback.str();
            std::swap( compileFeedback2, errDetails.m_compilerFeedback );
        }
        if( errDetails.m_compilerFeedback.str().length() > 0 )
            clog.callback( level, "COMPILE FEEDBACK", errDetails.m_compilerFeedback.str().c_str() );

        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

void optix::readOrWrite( PersistentStream* stream, optix_exp::SubModule::ModuleSymbol* symbol, const char* label )
{
    auto tmp = stream->pushObject( label, "moduleSymbol" );
    readOrWrite( stream, &symbol->size, "size" );
    readOrWrite( stream, (int*)&symbol->type, "type" );
}
