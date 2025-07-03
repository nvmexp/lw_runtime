// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Context/RTCore.h>

#include <exp/context/ForceDeprecatedCompiler.h>

#include <prodlib/exceptions/RTCoreError.h>
#include <prodlib/exceptions/UnknownError.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <corelib/system/ExelwtableModule.h>
#include <corelib/system/System.h>

#include <rtcore/interface/rtcore.h>

#include <lwselwreloadlibrary/SelwreExelwtableModule.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>

// The name of the rtcore library
#ifdef __linux__
#include <common/inc/lwUnixVersion.h>
#define RTCORE_DRIVER_LIB_NAME "liblwidia-rtcore.so." LW_VERSION_STRING
#define RTCORE_BC_DRIVER_LIB_NAME "liblwidia-rtcore-bc.so." LW_VERSION_STRING
#define RTCORE_SDK_LIB_NAME "liblwidia-rtcore-sdk.so." LW_VERSION_STRING
#elif defined( __APPLE__ )
#define RTCORE_DRIVER_LIB_NAME "liblwidia-rtcore.dylib"
#define RTCORE_SDK_LIB_NAME "liblwidia-rtcore-sdk.dylib"
#else
#define RTCORE_DRIVER_LIB_NAME "lwrtum64.dll"
#define RTCORE_BC_DRIVER_LIB_NAME "lwrtum64-bc.dll"
#define RTCORE_SDK_LIB_NAME "lwrtum64-sdk.dll"
#endif

using namespace prodlib;

namespace {
// clang-format off
Knob<std::string>     k_rtcoreKnobs(             RT_DSTRING("rtcore.knobs"),                    "",     RT_DSTRING( "Knob string passed to RTcore (see rtcore.h for syntax)" ) );
HiddenPublicKnob<int> k_rtxExtraDirectStackSize( RT_PUBLIC_DSTRING("rtx.extraDirectStackSize"), -1,     RT_PUBLIC_DSTRING( "Increase the direct stack size by the given amount. Default is -1, which means don't change it." ) );
// clang-format on

// TODO: This is a local definition of the pointer to the interface exported by rtcore. As far as I can see
// there is no definition of this in rtcore.h
typedef RtcResult ( *rtcGetExportTablePtr )( const void** ppExportTable, const Rtlwuid* pExportTableId );

class RTCoreInitializer
{
  public:
    void initialize( optix::RTCoreAPI& rtcore, RtcResult* returnResult );

  private:
    std::mutex m_initializationLock;
    bool       m_initialized = false;

    static void rtcoreLogCallback( RtcLogMsgType type, const char* file, int line, const char* function, int level, const char* msg );

    std::string getKnobs();
};

void RTCoreInitializer::rtcoreLogCallback( RtcLogMsgType type, const char* file, int line, const char* function, int level, const char* msg )
{
#define MSG_WITH_POS file << ":" << line << " (in " << function << "): " << msg

    if( type == RTC_LOG_MSG_TYPE_ASSERT )
        lfatal << MSG_WITH_POS;
    else if( type == RTC_LOG_MSG_TYPE_ERROR )
        lerr << MSG_WITH_POS;
    else if( type == RTC_LOG_MSG_TYPE_WARNING )
        lwarn << MSG_WITH_POS;
    else
        llog( level ) << msg;  // nothing serious, leave out the location info

#undef MSG_WITH_POS
}

std::string RTCoreInitializer::getKnobs()
{
    // Set the rtcore knobs if the corresponding Optix knob is set
    std::string rtcoreKnobs( k_rtcoreKnobs.get() );
    if( !k_rtxExtraDirectStackSize.isDefault() )
    {
        // If the extra direct stack size is set, add the corresponding rtcore knob in addition
        if( !rtcoreKnobs.empty() )
            rtcoreKnobs += ",";
        rtcoreKnobs += "pipeline.extraDirectStackSize:" + std::to_string( k_rtxExtraDirectStackSize.get() );
    }
    return rtcoreKnobs;
}

void RTCoreInitializer::initialize( optix::RTCoreAPI& rtcore, RtcResult* returnResult )
{
    std::lock_guard<std::mutex> lock( m_initializationLock );
    if( m_initialized )
        return;

    int             major = 0, minor = 0;
    const RtcResult getVersionResult = rtcore.getVersion( &major, &minor, nullptr );
    if( getVersionResult != RTC_SUCCESS )
    {
        if( returnResult )
            *returnResult = getVersionResult;
        return;
    }
    llog( 20 ) << "Initializing RTcore " << major << "." << minor << '\n';


    int            debugLogLevel = prodlib::log::level();
    PFNRTCDEBUGLOG debugLogCb    = rtcoreLogCallback;

    RtcResult initResult = rtcore.init( debugLogLevel, debugLogCb, getKnobs().c_str() );

    // rtcInit can return RTC_ERROR_ALREADY_INITIALIZED in any mode (debug, develop, and release)
    // when initialized from multiple APIs.  Fow now, swallow the error and move on.
    if( initResult == RTC_ERROR_ALREADY_INITIALIZED )
    {
#if defined( OPTIX_ENABLE_LOGGING )
        lwarn << "RTcore already initialized and ignoring knob and log setting\n";
#endif
        initResult = RTC_SUCCESS;
    }

    // Copy the error out if we need to.
    if( returnResult )
        *returnResult = initResult;

    if( initResult )
        return;

    m_initialized = true;
}

inline void checkRTCoreResult( RtcResult result, const char* call, RtcResult* returnResult )
{
    if( returnResult )
        *returnResult = result;
    else if( result != RTC_SUCCESS )
        throw prodlib::RTCoreError( RT_EXCEPTION_INFO, call, result );
}

#define CHECK( call ) checkRTCoreResult( call, #call, returnResult )

}  // namespace

namespace optix {

// Static pointer to the rtcore library. Needs g_rtcoreLibraryMutex.
static std::unique_ptr<corelib::ExelwtableModule> g_rtcoreLibrary;

// Mutex for g_rtcoreLibrary.
static std::mutex g_rtcoreLibraryMutex;

static RTCoreInitializer g_rtcoreInitializer;

bool RTCoreAPI::m_useLibraryFromSdk = false;

void RTCoreAPI::setRtcoreLibraryVariant( bool useLibraryFromSdk )
{
    m_useLibraryFromSdk = useLibraryFromSdk;
}

static bool useLwvm34()
{
    std::string value;
    return corelib::getelw( OPTIX_FORCE_DEPRECATED_COMPILER_STR, value )
           && std::stoi( value ) == optix_exp::OptixForceDeprecatedCompilerValues::LWVM34_LWPTX;
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::finishConstruction()
{
    const std::string libName =
        m_useLibraryFromSdk ? RTCORE_SDK_LIB_NAME : ( useLwvm34() ? RTCORE_BC_DRIVER_LIB_NAME : RTCORE_DRIVER_LIB_NAME );

    std::lock_guard<std::mutex> lock( g_rtcoreLibraryMutex );
    if( !g_rtcoreLibrary )
    {
        g_rtcoreLibrary.reset( m_useLibraryFromSdk ? new corelib::ExelwtableModule( libName.c_str() ) :
                                                     new SelwreExelwtableModule( libName.c_str() ) );
        if( !g_rtcoreLibrary->init() )
        {
#if defined( OPTIX_ENABLE_LOGGING )
            lprint << "Failed to load " << libName << "\n";
#endif
            return RTC_ERROR_UNKNOWN;
        }

#if defined( OPTIX_ENABLE_LOGGING )
        const std::string filePath = g_rtcoreLibrary->getPath( "rtcGetExportTable" );
        if( filePath.length() == 0 )
            lprint << "Loaded " << libName << " from an unknown location\n";
        else
            lprint << "Loaded " << libName << " from \"" << filePath.c_str() << "\"\n";
#endif
    }

    const rtcGetExportTablePtr rtcGetExportTable =
        reinterpret_cast<rtcGetExportTablePtr>( g_rtcoreLibrary->getFunction( "rtcGetExportTable" ) );
    if( !rtcGetExportTable )
    {
#if defined( OPTIX_ENABLE_LOGGING )
        lprint << "Unable to get export table function from library " << libName << ".\n";
#endif
        return RTC_ERROR_UNKNOWN;
    }

    if( RtcResult res = rtcGetExportTable( (const void**)&m_exports, &RTC_ETID_RTCore ) )
    {
#if defined( OPTIX_ENABLE_LOGGING )
        lprint << "Unable to initialize interface to rtcore: " << res << "\n";
#endif
        return res;
    }

    return RTC_SUCCESS;
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::initializeRTCoreLibraryWithKnobs()
{
    RtcResult result = finishConstruction();
    if( result != RTC_SUCCESS )
        return result;
    g_rtcoreInitializer.initialize( *this, &result );
    return result;
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::getVersion( int* major, int* minor, int* build )
{
    return m_exports->rtcGetVersion( major, minor, build );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::rtcGetBuildUUID( Rtlw32 uuid[4] )
{
    return m_exports->rtcGetBuildUUID( uuid );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::init( int debugLogLevel, PFNRTCDEBUGLOG debugLogCb, const char* debugKnobs )
{
    return m_exports->rtcInit( debugLogLevel, debugLogCb, debugKnobs );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::deviceContextCreateForLWDA( LWcontext                  context,
                                                                     const RtcDeviceProperties* properties,
                                                                     RtcDeviceContext*          devctx )
{
    return m_exports->rtcDeviceContextCreateForLWDA( context, properties, devctx );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::deviceContextDestroy( RtcDeviceContext devctx )
{
    return m_exports->rtcDeviceContextDestroy( devctx );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::deviceContextGetLimit( RtcDeviceContext devctx, RtcLimit limit, Rtlw64* value )
{
    return m_exports->rtcDeviceContextGetLimit( devctx, limit, value );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::deviceContextGetCompatibilityIdentifier( RtcDeviceContext devctx,
                                                                                  RtcDeviceContextCompatibilityType type,
                                                                                  Rtlwuid* identifier )
{
    return m_exports->rtcDeviceContextGetCompatibilityIdentifier( devctx, type, identifier );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::deviceContextCheckCompatibility( RtcDeviceContext                  devctx,
                                                                          RtcDeviceContextCompatibilityType type,
                                                                          const Rtlwuid*                    identifier )
{
    return m_exports->rtcDeviceContextCheckCompatibility( devctx, type, identifier );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::commandListCreateForLWDA( RtcDeviceContext devctx, LWstream stream, RtcCommandList* cmdlist )
{
    return m_exports->rtcCommandListCreateForLWDA( devctx, stream, cmdlist );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::commandListDestroy( RtcCommandList cmdlist )
{
    return m_exports->rtcCommandListDestroy( cmdlist );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compileModule( RtcDeviceContext         context,
                                                        const RtcCompileOptions* options,
                                                        const char*              inputSerializedModuleBuffer,
                                                        Rtlw64                   bufferSize,
                                                        RtcCompiledModule*       compiledModule )
{
    return m_exports->rtcCompileModule( context, options, inputSerializedModuleBuffer, bufferSize, nullptr, 0,
#if RTCORE_API_VERSION >= 13
        nullptr /*baseModule*/,
#endif
        compiledModule );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compileNamedConstant( RtcDeviceContext context, const char* symbolName, int nbytes, RtcCompiledModule* compiledModule )
{
    return m_exports->rtcCompileNamedConstant( context, symbolName, nbytes, compiledModule );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleGetCachedBlob( RtcCompiledModule compiledModule,
                                                                      Rtlw64            bufferSize,
                                                                      void*             blob,
                                                                      Rtlw64*           blobSize )
{
    return m_exports->rtcCompiledModuleGetCachedBlob( compiledModule, bufferSize, blob, blobSize );
}

#if RTCORE_API_VERSION >= 25
CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleGetStackSize( RtcCompiledModule compiledModule,
                                                                     Rtlw32            entryIndex,
                                                                     Rtlw32*           directStackFrameSize,
                                                                     Rtlw32*           continuationStackFrameSize )
{
    return m_exports->rtcCompiledModuleGetStackSize( compiledModule, entryIndex, directStackFrameSize, continuationStackFrameSize );
}
#else
CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleGetStackSize( RtcCompiledModule compiledModule,
                                                                     const char*       symbolName,
                                                                     Rtlw32*           directStackFrameSize,
                                                                     Rtlw32*           continuationStackFrameSize )
{
    return m_exports->rtcCompiledModuleGetStackSize( compiledModule, symbolName, directStackFrameSize, continuationStackFrameSize );
}
#endif

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleFromCachedBlob( RtcDeviceContext   context,
                                                                       const void*        blob,
                                                                       Rtlw64             blobSize,
                                                                       RtcCompiledModule* compiledModule )
{
    return m_exports->rtcCompiledModuleFromCachedBlob( context, blob, blobSize, compiledModule );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleDestroy( RtcCompiledModule module )
{
    return m_exports->rtcCompiledModuleDestroy( module );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::accelComputeMemoryUsage( RtcDeviceContext                     context,
                                                                  const RtcAccelOptions*               accelOptions,
                                                                  unsigned                             numItemArrays,
                                                                  const RtcBuildInput*                 buildInputs,
                                                                  const RtcBuildInputOverrides* const* overrides,
                                                                  RtcAccelBufferSizes*                 bufferSizes )
{
    return m_exports->rtcAccelComputeMemoryUsage( context, accelOptions, numItemArrays, buildInputs, overrides, bufferSizes );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::accelBuild( RtcCommandList                       commandList,
                                                     const RtcAccelOptions*               accelOptions,
                                                     unsigned                             numItemArrays,
                                                     const RtcBuildInput*                 buildInputs,
                                                     const RtcBuildInputOverrides* const* overrides,
                                                     const RtcAccelBuffers*               buffers,
                                                     unsigned                             numEmittedProperties,
                                                     const RtcAccelEmitDesc*              emittedProperties )
{
    return m_exports->rtcAccelBuild( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr, accelOptions, numItemArrays,
                                     buildInputs, overrides, buffers, numEmittedProperties, emittedProperties );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::accelEmitProperties( RtcCommandList       commandList,
                                                              RtcGpuVA*            sourceAccels,
                                                              unsigned             numSourceAccels,
                                                              RtcAccelPropertyType type,
                                                              RtcGpuVA             resultBuffer,
                                                              Rtlw64               resultBufferSize )
{
    return m_exports->rtcAccelEmitProperties( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr, sourceAccels,
                                              numSourceAccels, type, resultBuffer, resultBufferSize );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::accelCopy( RtcCommandList commandList,
                                                    RtcGpuVA       sourceBuffer,
                                                    RtcCopyMode    mode,
                                                    RtcGpuVA       resultBuffer,
                                                    Rtlw64         resultBufferSize )
{
    return m_exports->rtcAccelCopy( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr, sourceBuffer, mode,
                                    resultBuffer, resultBufferSize );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::accelRelocate( RtcCommandList commandList,
                                                        RtcGpuVA       traversableVAs,
                                                        Rtlw32         numTraversableVAs,
                                                        RtcGpuVA       accelBuffer,
                                                        Rtlw64         accelBufferSize )
{
    return m_exports->rtcAccelRelocate( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr, traversableVAs,
                                        numTraversableVAs, accelBuffer, accelBufferSize );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::colwertPointerToTraversableHandle( RtcDeviceContext      context,
                                                                            RtcGpuVA              pointer,
                                                                            RtcTraversableType    traversableType,
                                                                            RtcAccelType          accelType,
                                                                            RtcTraversableHandle* traversableHandle )
{
    return m_exports->rtcColwertPointerToTraversableHandle( context, pointer, traversableType, accelType, traversableHandle );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::colwertTraversableHandleToPointer( RtcDeviceContext     context,
                                                                            RtcTraversableHandle traversableHandle,
                                                                            RtcGpuVA*            pointer,
                                                                            RtcTraversableType*  traversableType,
                                                                            RtcAccelType*        accelType )
{
    return m_exports->rtcColwertTraversableHandleToPointer( context, traversableHandle, pointer, traversableType, accelType );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::getSbtRecordHeaderSize( Rtlw64* nbytes )
{
    return m_exports->rtcGetSbtRecordHeaderSize( nbytes );
}

#if RTCORE_API_VERSION >= 25
CHECK_RTCORE_RESULT RtcResult RTCoreAPI::packSbtRecordHeader( RtcDeviceContext        context,
                                                              const RtcCompiledModule moduleGlobalOrCH,
                                                              Rtlw32                  entryFunctionIndexGlobalOrCH,
                                                              const RtcCompiledModule moduleAH,
                                                              Rtlw32                  entryFunctionIndexAH,
                                                              const RtcCompiledModule moduleIS,
                                                              Rtlw32                  entryFunctionIndexIS,
                                                              void*                   sbtHeaderHostPointer )
{
    return m_exports->rtcPackSbtRecordHeader( context, moduleGlobalOrCH, entryFunctionIndexGlobalOrCH, moduleAH,
                                              entryFunctionIndexAH, moduleIS, entryFunctionIndexIS, sbtHeaderHostPointer );
}
#else
CHECK_RTCORE_RESULT RtcResult RTCoreAPI::packSbtRecordHeader( RtcDeviceContext        context,
                                                              const RtcCompiledModule moduleGlobalOrCH,
                                                              const char*             entryFunctionNameGlobalOrCH,
                                                              const RtcCompiledModule moduleAH,
                                                              const char*             entryFunctionNameAH,
                                                              const RtcCompiledModule moduleIS,
                                                              const char*             entryFunctionNameIS,
                                                              void*                   sbtHeaderHostPointer )
{
    return m_exports->rtcPackSbtRecordHeader( context, moduleGlobalOrCH, entryFunctionNameGlobalOrCH, moduleAH,
                                              entryFunctionNameAH, moduleIS, entryFunctionNameIS, sbtHeaderHostPointer );
}
#endif

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineCreate( RtcDeviceContext          context,
                                                         const RtcPipelineOptions* pipelineOptions,
                                                         const RtcCompileOptions*  compileOptions,
                                                         const RtcCompiledModule*  modules,
                                                         int                       moduleCount,
                                                         RtcPipeline*              pipeline )
{
    return m_exports->rtcPipelineCreate( context, pipelineOptions, compileOptions, modules, moduleCount, pipeline );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineDestroy( RtcPipeline pipeline )
{
    return m_exports->rtcPipelineDestroy( pipeline );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineGetInfo( RtcPipeline pipeline, RtcPipelineInfoType type, Rtlw64 dataSize, void* data )
{
    return m_exports->rtcPipelineGetInfo( pipeline, type, dataSize, data );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineGetLaunchBufferInfo( RtcPipeline pipeline, Rtlw64* nbytes, Rtlw64* align )
{
    return m_exports->rtcPipelineGetLaunchBufferInfo( pipeline, nbytes, align );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineGetNamedConstantInfo( RtcPipeline pipeline, const char* symbolName, Rtlw64* offset, Rtlw64* nbytes )
{
    return m_exports->rtcPipelineGetNamedConstantInfo( pipeline, symbolName, offset, nbytes );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineGetScratchBufferInfo3D( RtcPipeline pipeline,
                                                                         RtcS64      width,
                                                                         RtcS64      height,
                                                                         RtcS64      depth,
                                                                         Rtlw64*     nbytesMin,
                                                                         Rtlw64*     nbytes,
                                                                         Rtlw64*     align )
{
    return m_exports->rtcPipelineGetScratchBufferInfo3D( pipeline, width, height, depth, nbytesMin, nbytes, align );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineGetStackSize( RtcPipeline pipeline,
                                                               Rtlw32*     directCallableStackSizeFromTraversal,
                                                               Rtlw32*     directCallableStackSizeFromState,
                                                               Rtlw32*     continuationStackSize,
                                                               Rtlw32*     maxTraversableGraphDepth )
{
    return m_exports->rtcPipelineGetStackSize( pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                                               continuationStackSize, maxTraversableGraphDepth );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::pipelineSetStackSize( RtcPipeline pipeline,
                                                               Rtlw32      directCallableStackSizeFromTraversal,
                                                               Rtlw32      directCallableStackSizeFromState,
                                                               Rtlw32      continuationStackSize,
                                                               Rtlw32      maxTraversableGraphDepth )
{
    return m_exports->rtcPipelineSetStackSize( pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                                               continuationStackSize, maxTraversableGraphDepth );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::launch3D( RtcCommandList cmdlist,
                                                   RtcPipeline    pipeline,
                                                   RtcGpuVA       launchBufferVA,
                                                   RtcGpuVA       scratchBufferVA,
                                                   RtcGpuVA       raygenSbtRecordVA,
                                                   RtcGpuVA       exceptionSbtRecordVA,
                                                   RtcGpuVA       firstMissSbtRecordVA,
                                                   Rtlw32         missSbtRecordSize,
                                                   Rtlw32         missSbtRecordCount,
                                                   RtcGpuVA       firstInstanceSbtRecordVA,
                                                   Rtlw32         instanceSbtRecordSize,
                                                   Rtlw32         instanceSbtRecordCount,
                                                   RtcGpuVA       firstCallableSbtRecordVA,
                                                   Rtlw32         callableSbtRecordSize,
                                                   Rtlw32         callableSbtRecordCount,
                                                   RtcGpuVA       toolsOutputVA,
                                                   Rtlw64         toolsOutputSize,
                                                   Rtlw64         scratchBufferSizeInBytes,
                                                   RtcS64         width,
                                                   RtcS64         height,
                                                   RtcS64         depth )
{
    return m_exports->rtcLaunch3D( cmdlist,
                                   /*launchPriority*/ 0, /*qmdDesc*/ nullptr, pipeline,
#if RTCORE_API_VERSION >= 13
                                   0 /*specializationHandle*/,
#endif
                                   launchBufferVA, scratchBufferVA,
                                   raygenSbtRecordVA, exceptionSbtRecordVA, firstMissSbtRecordVA, missSbtRecordSize,
                                   missSbtRecordCount, firstInstanceSbtRecordVA, instanceSbtRecordSize, instanceSbtRecordCount,
                                   firstCallableSbtRecordVA, callableSbtRecordSize, callableSbtRecordCount,
                                   toolsOutputVA, toolsOutputSize, scratchBufferSizeInBytes, width, height, depth );
}

#if RTCORE_API_VERSION >= 25
CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleGetHash( RtcCompiledModule module, Rtlw64* hash )
{
    return m_exports->rtcCompiledModuleGetHash( module, hash );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleGetEntryFunctionIndex( RtcCompiledModule module,
                                                                              const char*       entryFunctionName,
                                                                              Rtlw32*           entryFunctionIndex )
{
    return m_exports->rtcCompiledModuleGetEntryFunctionIndex( module, entryFunctionName, entryFunctionIndex );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::compiledModuleGetEntryFunctionName( RtcCompiledModule module,
                                                                             Rtlw32            entryFunctionIndex,
                                                                             Rtlw32            nameBufferSize,
                                                                             char*             nameBuffer,
                                                                             Rtlw32*           entryFunctionNameSize )
{
    return m_exports->rtcCompiledModuleGetEntryFunctionName( module, entryFunctionIndex, nameBufferSize, nameBuffer, entryFunctionNameSize );
}
#endif

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#if RTCORE_API_VERSION >= 31
CHECK_RTCORE_RESULT RtcResult RTCoreAPI::visibilityMapArrayComputeMemoryUsage( RtcDeviceContext context,
                                                                               const RtcVisibilityMapArrayBuildInput* buildInput,
                                                                               RtcMicromeshBufferSizes* bufferSizes )
{
    return m_exports->rtcVisibilityMapArrayComputeMemoryUsage( context, buildInput, bufferSizes );
}


CHECK_RTCORE_RESULT RtcResult RTCoreAPI::visibilityMapArrayBuild( RtcCommandList                         commandList,
                                                                  const RtcVisibilityMapArrayBuildInput* buildInput,
                                                                  const RtcMicromeshBuffers*             buffers,
                                                                  unsigned int                numEmittedProperties,
                                                                  const RtcMicromeshEmitDesc* emittedProperties )
{
    return m_exports->rtcVisibilityMapArrayBuild( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr, buildInput,
                                                  buffers, numEmittedProperties, emittedProperties );
}


CHECK_RTCORE_RESULT RtcResult RTCoreAPI::displacedMicromeshArrayComputeMemoryUsage( RtcDeviceContext context,
                                                                                    const RtcDisplacedMicromeshArrayBuildInput* buildInput,
                                                                                    RtcMicromeshBufferSizes* bufferSizes )
{
    return m_exports->rtcDisplacedMicromeshArrayComputeMemoryUsage( context, buildInput, bufferSizes );
}


CHECK_RTCORE_RESULT RtcResult RTCoreAPI::displacedMicromeshArrayBuild( RtcCommandList commandList,
                                                                       const RtcDisplacedMicromeshArrayBuildInput* buildInput,
                                                                       const RtcMicromeshBuffers*  buffers,
                                                                       unsigned int                numEmittedProperties,
                                                                       const RtcMicromeshEmitDesc* emittedProperties )
{
    return m_exports->rtcDisplacedMicromeshArrayBuild( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr,
                                                       buildInput, buffers, numEmittedProperties, emittedProperties );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::micromeshEmitProperties( RtcCommandList            commandList,
                                                                  const RtcGpuVA*           sourceMmArrays,
                                                                  unsigned int              numSourceMmArrays,
                                                                  RtcMicromeshPropertyType  type,
                                                                  RtcGpuVA                  resultBuffer,
                                                                  Rtlw64                    resultBufferSize )
{
    return m_exports->rtcMicromeshEmitProperties( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr,
                                                  sourceMmArrays, numSourceMmArrays, type, resultBuffer, resultBufferSize );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::micromeshCopy( RtcCommandList       commandList,
                                                        RtcGpuVA             sourceBuffer,
                                                        RtcMicromeshCopyMode mode,
                                                        RtcGpuVA             resultBuffer,
                                                        Rtlw64               resultBufferSize )
{
    return m_exports->rtcMicromeshCopy( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr,
                                        sourceBuffer, mode, resultBuffer, resultBufferSize );
}

CHECK_RTCORE_RESULT RtcResult RTCoreAPI::micromeshRelocate( RtcCommandList    commandList,
                                                            RtcGpuVA          mmArrayBuffer,
                                                            Rtlw64            mmArrayBufferSize )
{
    return m_exports->rtcMicromeshRelocate( commandList, /*launchPriority*/ 0, /*qmdDesc*/ nullptr,
                                            mmArrayBuffer, mmArrayBufferSize );
}
#endif  // RTCORE_API_VERSION >= 31
#endif  // LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )


RTCore::RTCore()
{
    RtcResult* returnResult = nullptr;
    CHECK( m_api.finishConstruction() );
}

void RTCore::setRtcoreLibraryVariant( bool useLibraryFromSdk )
{
    RTCoreAPI::setRtcoreLibraryVariant( useLibraryFromSdk );
}

void RTCore::initializeRTCoreLibraryWithKnobs( RtcResult* returnResult )
{
    CHECK( m_api.initializeRTCoreLibraryWithKnobs() );
}

void RTCore::getVersion( int*       major, /* [out] pointer to output (optional) */
                         int*       minor, /* [out] pointer to output (optional) */
                         int*       build, /* [out] pointer to build CL (optional) */
                         RtcResult* returnResult )
{
    CHECK( m_api.getVersion( major, minor, build ) );
}

void RTCore::rtcGetBuildUUID( Rtlw32     uuid[4], /* [out] pointer to output */
                              RtcResult* returnResult )
{
    CHECK( m_api.rtcGetBuildUUID( uuid ) );
}

void RTCore::init( int            debugLogLevel, /* [in] active log level in [0,100] */
                   PFNRTCDEBUGLOG debugLogCb,    /* [in] log function callback (optional) */
                   const char*    debugKnobs,    /* [in] debug knob overrides (optional) */
                   RtcResult*     returnResult )
{
    CHECK( m_api.init( debugLogLevel, debugLogCb, debugKnobs ) );
}

void RTCore::deviceContextCreateForLWDA( LWcontext context, /* [in] the LWCA context associated with the device context */
                                         const RtcDeviceProperties* properties, /* [in] device properties supplied by the product */
                                         RtcDeviceContext* devctx, /* [out] the device context to be created */
                                         RtcResult*        returnResult )
{
    CHECK( m_api.deviceContextCreateForLWDA( context, properties, devctx ) );
}

void RTCore::deviceContextDestroy( RtcDeviceContext devctx, /* [in] the device context to be destroyed */
                                   RtcResult*       returnResult )
{
    CHECK( m_api.deviceContextDestroy( devctx ) );
}

void RTCore::deviceContextGetLimit( RtcDeviceContext devctx, /* [in] the device context to query the limit for */
                                    RtcLimit         limit,  /* [in] the limit to query */
                                    Rtlw64*          value,  /* [out] pointer to the retured value */
                                    RtcResult*       returnResult )
{
    CHECK( m_api.deviceContextGetLimit( devctx, limit, value ) );
}

void RTCore::deviceContextGetCompatibilityIdentifier( RtcDeviceContext devctx, /* [in] the device context to query the identifier for */
                                                      RtcDeviceContextCompatibilityType type, /* [in] the type of compatibility queried */
                                                      Rtlwuid* identifier, /* [out] the device and driver identifier for the selected type */
                                                      RtcResult* returnResult )
{
    CHECK( m_api.deviceContextGetCompatibilityIdentifier( devctx, type, identifier ) );
}

void RTCore::deviceContextCheckCompatibility( RtcDeviceContext devctx, /* [in] the device context to match the identifier against */
                                              RtcDeviceContextCompatibilityType type, /* [in] type of compatibility check */
                                              const Rtlwuid* identifier, /* [in] the device and driver identifier for the selected type */
                                              RtcResult* returnResult )
{
    CHECK( m_api.deviceContextCheckCompatibility( devctx, type, identifier ) );
}

void RTCore::commandListCreateForLWDA( RtcDeviceContext devctx, /* [in] the device context associated with the command list */
                                       LWstream         stream, /* [in] the parent LWCA stream */
                                       RtcCommandList* cmdlist, /* [out] the RTcore command list to be created */
                                       RtcResult*      returnResult )
{
    CHECK( m_api.commandListCreateForLWDA( devctx, stream, cmdlist ) );
}

void RTCore::commandListDestroy( RtcCommandList cmdlist, /* [in] the command list to be destroyed */
                                 RtcResult*     returnResult )
{
    CHECK( m_api.commandListDestroy( cmdlist ) );
}

void RTCore::compileModule( RtcDeviceContext         context, /* [in] the device context the module is for */
                            const RtcCompileOptions* options, /* [in] options */
                            const char* inputSerializedModuleBuffer, /* [in] the input serialized module buffer according to the LWVM-RT spec */
                            Rtlw64             bufferSize,     /* [in] size of serialized buffer */
                            RtcCompiledModule* compiledModule, /* [out] the result module */
                            RtcResult*         returnResult )
{
    CHECK( m_api.compileModule( context, options, inputSerializedModuleBuffer, bufferSize, compiledModule ) );
}

void RTCore::compileNamedConstant( RtcDeviceContext   context,        /* [in] the device context the module is for */
                                   const char*        symbolName,     /* [in] name of the constant */
                                   int                nbytes,         /* [in] size in bytes of the constant */
                                   RtcCompiledModule* compiledModule, /* [out] the result module */
                                   RtcResult*         returnResult )
{
    CHECK( m_api.compileNamedConstant( context, symbolName, nbytes, compiledModule ) );
}

void RTCore::compiledModuleGetCachedBlob( RtcCompiledModule compiledModule, /* [in] the module to serialize */
                                          Rtlw64 bufferSize, /* [in] size in bytes of the buffer pointer to by 'blob' (0 if 'blob' is null). */
                                          void* blob, /* [out] pointer to a destination buffer receiving the blob data (optional) */
                                          Rtlw64* blobSize, /* [out] amount of storage in bytes required to hold the blob data (optional) */
                                          RtcResult* returnResult )
{
    CHECK( m_api.compiledModuleGetCachedBlob( compiledModule, bufferSize, blob, blobSize ) );
}

#if RTCORE_API_VERSION >= 25
void RTCore::compiledModuleGetStackSize( RtcCompiledModule compiledModule, /* [in] the module */
                                         Rtlw32  symbolIndex, /* [in] the index of the function in the module */
                                         Rtlw32* directStackFrameSize, /* [out] amount of storage in bytes required for the direct (ABI low level) stack */
                                         Rtlw32* continuationStackFrameSize, /* [out] amount of storage in bytes required for the continuation (rtcore SW level) stack */
                                         RtcResult* returnResult )
{
    CHECK( m_api.compiledModuleGetStackSize( compiledModule, symbolIndex, directStackFrameSize, continuationStackFrameSize ) );
}
#else
void RTCore::compiledModuleGetStackSize( RtcCompiledModule compiledModule, /* [in] the module */
                                         const char*       symbolName, /* [in] the name of the function in the module */
                                         Rtlw32* directStackFrameSize, /* [out] amount of storage in bytes required for the direct (ABI low level) stack */
                                         Rtlw32* continuationStackFrameSize, /* [out] amount of storage in bytes required for the continuation (rtcore SW level) stack */
                                         RtcResult* returnResult )
{
    CHECK( m_api.compiledModuleGetStackSize( compiledModule, symbolName, directStackFrameSize, continuationStackFrameSize ) );
}
#endif

void RTCore::compiledModuleFromCachedBlob( RtcDeviceContext context, /* [in] the device context the module is for */
                                           const void*      blob,    /* [in] the blob data to deserialize */
                                           Rtlw64 blobSize, /* [in] the size in bytes of the buffer pointed to by 'blob' */
                                           RtcCompiledModule* compiledModule, /* [out] the result module */
                                           RtcResult*         returnResult )
{
    CHECK( m_api.compiledModuleFromCachedBlob( context, blob, blobSize, compiledModule ) );
}

void RTCore::compiledModuleDestroy( RtcCompiledModule module, /* [in] the module to be destroyed */
                                    RtcResult*        returnResult )
{
    CHECK( m_api.compiledModuleDestroy( module ) );
}

void RTCore::accelComputeMemoryUsage( RtcDeviceContext       context,       /* [in] device context of the pipeline */
                                      const RtcAccelOptions* accelOptions,  /* [in] accel options */
                                      unsigned int           numItemArrays, /* [in] number of elements in buildInputs */
                                      const RtcBuildInput*   buildInputs,   /* [in] an array of RtcBuildInput objects */
                                      RtcBuildInputOverrides const* const* overrides, /* [in] an array of RtcBuildInputOverride objects, May be NULL, entries may be NULL */
                                      RtcAccelBufferSizes* bufferSizes, /* [out] fills in buffer sizes */
                                      RtcResult*           returnResult )
{
    CHECK( m_api.accelComputeMemoryUsage( context, accelOptions, numItemArrays, buildInputs, overrides, bufferSizes ) );
}

void RTCore::accelBuild( RtcCommandList         commandList,   /* [in] command list in which to enqueue build kernels */
                         const RtcAccelOptions* accelOptions,  /* [in] accel options */
                         unsigned int           numItemArrays, /* [in] number of elements in buildInputs */
                         const RtcBuildInput*   buildInputs,   /* [in] an array of RtcBuildInput objects */
                         RtcBuildInputOverrides const* const* overrides, /* [in] an array of RtcBuildInputOverride objects, May be NULL, entries may be NULL */
                         const RtcAccelBuffers* buffers, /* [in] the buffers used for build */
                         unsigned int numEmittedProperties, /* [in] number of post-build properties to populate (may be zero) */
                         const RtcAccelEmitDesc* emittedProperties, /* [in/out] types of requested properties and output buffers */
                         RtcResult*              returnResult )
{
    CHECK( m_api.accelBuild( commandList, accelOptions, numItemArrays, buildInputs, overrides, buffers,
                             numEmittedProperties, emittedProperties ) );
}

void RTCore::accelEmitProperties( RtcCommandList       commandList,      /* [in] command list */
                                  RtcGpuVA*            sourceAccels,     /* [in] input accels */
                                  unsigned int         numSourceAccels,  /* [in] number of elements */
                                  RtcAccelPropertyType type,             /* [in] type of information requested */
                                  RtcGpuVA             resultBuffer,     /* [out] output buffer for the properties */
                                  Rtlw64               resultBufferSize, /* [in] size of output buffer */
                                  RtcResult*           returnResult )
{
    CHECK( m_api.accelEmitProperties( commandList, sourceAccels, numSourceAccels, type, resultBuffer, resultBufferSize ) );
}

void RTCore::accelCopy( RtcCommandList commandList,      /* [in] command list */
                        RtcGpuVA       sourceBuffer,     /* [in] input accel */
                        RtcCopyMode    mode,             /* [in] specify the output format of the copied accel */
                        RtcGpuVA       resultBuffer,     /* [out] copied accel */
                        Rtlw64         resultBufferSize, /* [in] size of cloned accel */
                        RtcResult*     returnResult )
{
    CHECK( m_api.accelCopy( commandList, sourceBuffer, mode, resultBuffer, resultBufferSize ) );
}

void RTCore::accelRelocate( RtcCommandList commandList,       /* [in] command list */
                            RtcGpuVA       traversableVAs,    /* [in] List of updated top->bottom level references for the relocated accel.
                                                                      Used for top-level accels only.
                                                                      Order and number of traversables must match the original build. */
                            Rtlw32         numTraversableVAs, /* [in] number of traversable VAs */
                            RtcGpuVA       accelBuffer,       /* [in/out] input accel */
                            Rtlw64 accelBufferSize, /* [in] Optional result buffer size. Must be ~0ULL if the size is unknown. Used for validation only. */
                            RtcResult* returnResult )
{
    CHECK( m_api.accelRelocate( commandList, traversableVAs, numTraversableVAs, accelBuffer, accelBufferSize ) );
}

void RTCore::colwertPointerToTraversableHandle( RtcDeviceContext context, /* [in] device context */
                                                RtcGpuVA pointer, /* [in] pointer to traversalbe allocated in RtcDeviceContext */
                                                RtcTraversableType traversableType, /* [in] Type of RtcTraversableHandle to create */
                                                RtcAccelType accelType, /* [in] Type of accel if traversableType is RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL, ignored otherwise */
                                                RtcTraversableHandle* traversableHandle, /* [out] traversable handle. traversableHandle must be in host memory */
                                                RtcResult* returnResult )
{
    CHECK( m_api.colwertPointerToTraversableHandle( context, pointer, traversableType, accelType, traversableHandle ) );
}

void RTCore::colwertTraversableHandleToPointer( RtcDeviceContext     context,           /* [in] device context */
                                                RtcTraversableHandle traversableHandle, /* [ouint] traversable handle. */
                                                RtcGpuVA* pointer, /* [out] pointer to traversalbe allocated in RtcDeviceContext */
                                                RtcTraversableType* traversableType, /* [out] Type of RtcTraversableHandle to create */
                                                RtcAccelType* accelType, /* [ioutn] Type of accel if traversableType is RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL, ignored otherwise */
                                                RtcResult* returnResult )
{
    CHECK( m_api.colwertTraversableHandleToPointer( context, traversableHandle, pointer, traversableType, accelType ) );
}

void RTCore::getSbtRecordHeaderSize( Rtlw64*    nbytes, /* [out] size in bytes of the sbt record header */
                                     RtcResult* returnResult )
{
    CHECK( m_api.getSbtRecordHeaderSize( nbytes ) );
}

#if RTCORE_API_VERSION >= 25
void RTCore::packSbtRecordHeader( RtcDeviceContext        context, /* [in] the device context the module is for */
                                  const RtcCompiledModule moduleGlobalOrCH, /* [in] the module containing the 1st entry function (any global type, or CH if packing a hit record) */
                                  Rtlw32 entryFunctionIndexGlobalOrCH, /* [in] the index of the 1st entry function */
                                  const RtcCompiledModule moduleAH, /* [in] the module containing the any hit entry function (only if packing a hit record) */
                                  Rtlw32 entryFunctionIndexAH, /* [in] the index of the any hit entry function */
                                  const RtcCompiledModule moduleIS, /* [in] the module containing the intersection entry function (only if packing a hit record) */
                                  Rtlw32 entryFunctionIndexIS, /* [in] the index of the intersection entry function */
                                  void*       sbtHeaderHostPointer, /* [out] the result sbt record header */
                                  RtcResult*  returnResult )
{
    CHECK( m_api.packSbtRecordHeader( context, moduleGlobalOrCH, entryFunctionIndexGlobalOrCH, moduleAH,
                                      entryFunctionIndexAH, moduleIS, entryFunctionIndexIS, sbtHeaderHostPointer ) );
}
#else
void RTCore::packSbtRecordHeader( RtcDeviceContext        context, /* [in] the device context the module is for */
                                  const RtcCompiledModule moduleGlobalOrCH, /* [in] the module containing the 1st entry function (any global type, or CH if packing a hit record) */
                                  const char* entryFunctionNameGlobalOrCH, /* [in] the name of the 1st entry function */
                                  const RtcCompiledModule moduleAH, /* [in] the module containing the any hit entry function (only if packing a hit record) */
                                  const char* entryFunctionNameAH, /* [in] the name of the any hit entry function */
                                  const RtcCompiledModule moduleIS, /* [in] the module containing the intersection entry function (only if packing a hit record) */
                                  const char* entryFunctionNameIS, /* [in] the name of the intersection entry function */
                                  void*       sbtHeaderHostPointer, /* [out] the result sbt record header */
                                  RtcResult*  returnResult )
{
    CHECK( m_api.packSbtRecordHeader( context, moduleGlobalOrCH, entryFunctionNameGlobalOrCH, moduleAH,
                                      entryFunctionNameAH, moduleIS, entryFunctionNameIS, sbtHeaderHostPointer ) );
}
#endif

void RTCore::pipelineCreate( RtcDeviceContext          context, /* [in] the device context the pipeline is for */
                             const RtcPipelineOptions* pipelineOptions, /* [in] pipeline options */
                             const RtcCompileOptions*  compileOptions,  /* [in] compile options */
                             const RtcCompiledModule*  modules, /* [in] the list of modules to create a pipeline from */
                             int                       moduleCount, /* [in] number of modules */
                             RtcPipeline*              pipeline,    /* [out] the result pipeline */
                             RtcResult*                returnResult )
{
    CHECK( m_api.pipelineCreate( context, pipelineOptions, compileOptions, modules, moduleCount, pipeline ) );
}

void RTCore::pipelineDestroy( RtcPipeline pipeline, /* [in] the pipeline to be destroyed */
                              RtcResult*  returnResult )
{
    CHECK( m_api.pipelineDestroy( pipeline ) );
}

void RTCore::pipelineGetInfo( RtcPipeline         pipeline, /* [in] pipeline to query information for */
                              RtcPipelineInfoType type, /* [in] what type of information to query about the pipeline */
                              Rtlw64     dataSize,      /* [in] size of the data structure receiving the information */
                              void*      data, /* [out] type-specific struct containing the queried information */
                              RtcResult* returnResult )
{
    CHECK( m_api.pipelineGetInfo( pipeline, type, dataSize, data ) );
}

void RTCore::pipelineGetLaunchBufferInfo( RtcPipeline pipeline, /* [in] pipeline to query launch buffer information for */
                                          Rtlw64*     nbytes,   /* [out] launch buffer size requirement */
                                          Rtlw64*     align,    /* [out] launch buffer alignment requirement */
                                          RtcResult* returnResult )
{
    CHECK( m_api.pipelineGetLaunchBufferInfo( pipeline, nbytes, align ) );
}

void RTCore::pipelineGetNamedConstantInfo( RtcPipeline pipeline,   /* [in] the pipeline to query the info for */
                                           const char* symbolName, /* [in] name of the constant */
                                           Rtlw64*     offset,     /* [out] offset relative to launch buffer start */
                                           Rtlw64*     nbytes,     /* [out] size of the constant */
                                           RtcResult*  returnResult )
{
    CHECK( m_api.pipelineGetNamedConstantInfo( pipeline, symbolName, offset, nbytes ) );
}

void RTCore::pipelineGetScratchBufferInfo3D( RtcPipeline pipeline, /* [in] pipeline to query launch buffer information for */
                                             RtcS64 width, /* [in] number of elements to compute, must match rtcLaunch parameter */
                                             RtcS64  height,    /* [in] number of elements to compute */
                                             RtcS64  depth,     /* [in] number of elements to compute */
                                             Rtlw64* nbytesMin, /* [out] minimum scratch buffer size requirement */
                                             Rtlw64* nbytes, /* [out] requested scratch buffer size for efficient exelwtion */
                                             Rtlw64*    align, /* [out] scratch buffer alignment requirement */
                                             RtcResult* returnResult )
{
    CHECK( m_api.pipelineGetScratchBufferInfo3D( pipeline, width, height, depth, nbytesMin, nbytes, align ) );
}

void RTCore::pipelineGetStackSize( RtcPipeline pipeline, /* [in] pipeline to query stack size information  */
                                   Rtlw32*     directCallableStackSizeFromTraversal, /* [out] size in bytes of direct (ABI level) stack required for callables from IS/AH shaders */
                                   Rtlw32*     directCallableStackSizeFromState, /* [out] size in bytes of direct (ABI level) stack required for callables from RG/CH/MS shaders */
                                   Rtlw32* continuationStackSize, /* [out] size in bytes of continuation (rtcore SW level) stack required for rtcLaunch */
                                   Rtlw32* maxTraversableGraphDepth, /* [out] Maximum depth of a traversable graph passed to trace. 0 means the default of 2 */
                                   RtcResult* returnResult )
{
    CHECK( m_api.pipelineGetStackSize( pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                                       continuationStackSize, maxTraversableGraphDepth ) );
}

void RTCore::pipelineSetStackSize( RtcPipeline pipeline, /* [in] pipeline to set the stack size */
                                   Rtlw32      directCallableStackSizeFromTraversal, /* [in] size in bytes of direct (ABI level) stack required for callables from IS/AH shaders */
                                   Rtlw32      directCallableStackSizeFromState, /* [in] size in bytes of direct (ABI level) stack required for callables from RG/CH/MS shaders */
                                   Rtlw32 continuationStackSize, /* [in} size in bytes of continuation (rtcore SW level) stack required for rtcLaunch */
                                   Rtlw32 maxTraversableGraphDepth, /* [in] Maximum depth of a traversable graph passed to trace. 0 means the default of 2 */
                                   RtcResult* returnResult )
{
    CHECK( m_api.pipelineSetStackSize( pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                                       continuationStackSize, maxTraversableGraphDepth ) );
}

void RTCore::launch3D( RtcCommandList cmdlist,         /* [in] the command list to enqueue the launch into */
                       RtcPipeline    pipeline,        /* [in] the pipeline to be launched */
                       RtcGpuVA       launchBufferVA,  /* [in] device address of the launch buffer for this launch */
                       RtcGpuVA       scratchBufferVA, /* [in] device address of the scratch buffer for this launch */
                       RtcGpuVA raygenSbtRecordVA, /* [in] device address of the SBT record of the ray gen program to start at */
                       RtcGpuVA exceptionSbtRecordVA, /* [in] device address of the SBT record of the exception shader */
                       RtcGpuVA firstMissSbtRecordVA, /* [in] device address of the SBT record of the first miss program */
                       Rtlw32 missSbtRecordSize,      /* [in] size of a single SBT record in bytes */
                       Rtlw32 missSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
                       RtcGpuVA firstInstanceSbtRecordVA, /* [in] device address of the SBT record of the first hit program */
                       Rtlw32 instanceSbtRecordSize, /* [in] size of a single SBT record in bytes */
                       Rtlw32 instanceSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
                       RtcGpuVA firstCallableSbtRecordVA, /* [in] device address of the SBT record of the first callable program */
                       Rtlw32 callableSbtRecordSize, /* [in] size of a single SBT record in bytes */
                       Rtlw32 callableSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
                       RtcGpuVA toolsOutputVA, /* [in] device address where exception and profiling information shoud be written */
                       Rtlw64     toolsOutputSize,          /* [in] size of the tools output buffer */
                       Rtlw64     scratchBufferSizeInBytes, /* [in] size of the scratch buffer in bytes */
                       RtcS64     width,                    /* [in] number of elements to compute */
                       RtcS64     height,                   /* [in] number of elements to compute */
                       RtcS64     depth,                    /* [in] number of elements to compute */
                       RtcResult* returnResult )
{
    CHECK( m_api.launch3D( cmdlist, pipeline, launchBufferVA, scratchBufferVA, raygenSbtRecordVA, exceptionSbtRecordVA,
                           firstMissSbtRecordVA, missSbtRecordSize, missSbtRecordCount, firstInstanceSbtRecordVA, instanceSbtRecordSize,
                           instanceSbtRecordCount, firstCallableSbtRecordVA, callableSbtRecordSize, callableSbtRecordCount,
                           toolsOutputVA, toolsOutputSize, scratchBufferSizeInBytes, width, height, depth ) );
}

#if RTCORE_API_VERSION >= 25
void RTCore::compiledModuleGetHash( RtcCompiledModule module, /* [in] the  module */
                                    Rtlw64*           hash,   /* [out] the hash value from the module SASS binary */
                                    RtcResult*        returnResult )
{
    CHECK( m_api.compiledModuleGetHash( module, hash ) );
}

void RTCore::compiledModuleGetEntryFunctionIndex( RtcCompiledModule module,      /* [in] the  module */
                                                  const char* entryFunctionName, /* [in] the entry function name in the module */
                                                  Rtlw32* entryFunctionIndex,    /* [out] the index of the function in the module */
                                                  RtcResult* returnResult )
{
    CHECK( m_api.compiledModuleGetEntryFunctionIndex( module, entryFunctionName, entryFunctionIndex ) );
}

void RTCore::compiledModuleGetEntryFunctionName( RtcCompiledModule module, /* [in] the  module */
                                                 Rtlw32 entryFunctionIndex, /* [in] the index of the function in the module to get name (optional) */
                                                 Rtlw32 nameBufferSize, /* [in] size in bytes of the buffer pointer to by 'nameBuffer' (0 if 'nameBuffer' is null)(optional) */
                                                 char* nameBuffer, /* [out] the entry function name in the module at the index (optional) */
                                                 Rtlw32* entryFunctionNameSize, /* [out] the size in bytes of entry function name in the module at the index (optional) */
                                                 RtcResult* returnResult )
{
    CHECK( m_api.compiledModuleGetEntryFunctionName( module, entryFunctionIndex, nameBufferSize, nameBuffer, entryFunctionNameSize ) );
}
#endif

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#if RTCORE_API_VERSION >= 31
void RTCore::visibilityMapArrayComputeMemoryUsage( RtcDeviceContext                        context,
                                                   const RtcVisibilityMapArrayBuildInput* buildInput,
                                                   RtcMicromeshBufferSizes*                bufferSizes,
                                                   RtcResult*                              returnResult )
{
    CHECK( m_api.visibilityMapArrayComputeMemoryUsage( context, buildInput, bufferSizes ) );
}

void RTCore::visibilityMapArrayBuild( RtcCommandList                          commandList,
                                      const RtcVisibilityMapArrayBuildInput* buildInput,
                                      const RtcMicromeshBuffers*              buffers,
                                      unsigned int                            numEmittedProperties,
                                      const RtcMicromeshEmitDesc*            emittedProperties,
                                      RtcResult*                              returnResult )
{
    CHECK( m_api.visibilityMapArrayBuild( commandList, buildInput, buffers, numEmittedProperties, emittedProperties ) );
}

void RTCore::displacedMicromeshArrayComputeMemoryUsage( RtcDeviceContext                             context,
                                                        const RtcDisplacedMicromeshArrayBuildInput* buildInput,
                                                        RtcMicromeshBufferSizes*                     bufferSizes,
                                                        RtcResult*                                   returnResult )
{
    CHECK( m_api.displacedMicromeshArrayComputeMemoryUsage( context, buildInput, bufferSizes ) );
}

void RTCore::displacedMicromeshArrayBuild( RtcCommandList                              commandList,
                                           const RtcDisplacedMicromeshArrayBuildInput* buildInput,
                                           const RtcMicromeshBuffers*                  buffers,
                                           unsigned int                                numEmittedProperties,
                                           const RtcMicromeshEmitDesc*                 emittedProperties,
                                           RtcResult*                                  returnResult )
{
    CHECK( m_api.displacedMicromeshArrayBuild( commandList, buildInput, buffers, numEmittedProperties, emittedProperties ) );
}

void RTCore::micromeshEmitProperties( RtcCommandList           commandList,
                                      const RtcGpuVA*          sourceMmArrays,
                                      unsigned int             numSourceMmArrays,
                                      RtcMicromeshPropertyType type,
                                      RtcGpuVA resultBuffer,
                                      Rtlw64 resultBufferSize,
                                      RtcResult* returnResult )
{
    CHECK( m_api.micromeshEmitProperties( commandList, sourceMmArrays, numSourceMmArrays, type, resultBuffer, resultBufferSize ) );
}

void RTCore::micromeshCopy( RtcCommandList       commandList,
                            RtcGpuVA             sourceBuffer,
                            RtcMicromeshCopyMode mode,
                            RtcGpuVA             resultBuffer,
                            Rtlw64               resultBufferSize,
                            RtcResult*           returnResult )
{
    CHECK( m_api.micromeshCopy( commandList, sourceBuffer, mode, resultBuffer, resultBufferSize ) );
}

void RTCore::micromeshRelocate( RtcCommandList commandList,
                                RtcGpuVA       mmArrayBuffer,
                                Rtlw64         mmArrayBufferSize,
                                RtcResult*     returnResult )
{
    CHECK( m_api.micromeshRelocate( commandList, mmArrayBuffer, mmArrayBufferSize ) );
}

#endif  // RTCORE_API_VERSION >= 31
#endif  // LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )

}  // namespace optix
