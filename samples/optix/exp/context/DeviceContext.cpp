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
#define OPTIX_DEFINE_ABI_VERSION_ONLY
#include <optix_function_table.h>
#undef OPTIX_DEFINE_ABI_VERSION_ONLY

#include <exp/context/DeviceContext.h>
#include <exp/context/DiskCache.h>
#include <exp/context/EncryptionManager.h>
#include <exp/context/ErrorHandling.h>
#include <exp/context/ForceDeprecatedCompiler.h>
#include <exp/context/GpuWarmup.h>
#include <exp/context/OptixResultOneShot.h>
#include <exp/context/WatchdogTimer.h>
#include <exp/tools/ptxEncryptionBuildTool/secrets.h>

#include <rtcore/interface/types.h>

#include <Context/RTCore.h>
#include <Util/BinaryData.h>
#include <Util/optixUuid.h>
#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/ExelwtableModule.h>
#include <corelib/system/Preprocessor.h>
#include <corelib/system/System.h>
#include <prodlib/exceptions/RTCoreError.h>
#include <prodlib/exceptions/UnknownError.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/System.h>

#include <llvm/Support/Threading.h>

// this is needed to properly decode Chip IDs, lives in /dev/gpu_drv/bugfix_main/sdk/lwpu/inc/
#include <Lwcm.h>

#include <string.h>

#include <cmath>
#include <map>
#include <vector>

static const char* CACHE_DB_FILE_NAME = "optix7cache.db";

namespace {
// clang-format off
Knob<bool>         k_disableTTU( RT_DSTRING( "o7.disableTTU" ), false, RT_DSTRING( "Disables use of the TTU." ) );
Knob<bool>         k_disableMTTU( RT_DSTRING( "o7.disableMTTU" ), false, RT_DSTRING( "Disables use of the MTTU." ) );
#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
Knob<bool>         k_disableDisplacedMicromeshTTU( RT_DSTRING( "o7.disableDisplacedMicromeshTTU" ), false, RT_DSTRING( "Disables use of the DisplacedMicromeshTTU." ) );
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
Knob<bool>         k_forceTTUenabled( RT_DSTRING( "lwca.forceTTUenabled" ), false, RT_DSTRING( "Forces use of TTU/MTTU, regardless of what LWCA reports.  o7.disableTTU takes precendence." ) );
Knob<size_t>       k_launchResourceBufferSize( RT_DSTRING( "o7.launchResources.bufferSize" ), 8388608, RT_DSTRING( "Size in bytes of the launch resources ring buffer." ) );
Knob<unsigned int> k_launchResourceNumEvents( RT_DSTRING( "o7.launchResources.numEvents" ), 1000, RT_DSTRING( "Number of elements of the event ring buffer." ) );
Knob<size_t>       k_lowWaterMark( RT_DSTRING( "diskcache.lowWaterMark" ), 1u << 30, RT_DSTRING( "The cache size will be reduced to the low-water mark if the high-water mark is exceeded. The value 0 disables garbage collection. The default value is 1GB." ) );
Knob<size_t>       k_highWaterMark( RT_DSTRING( "diskcache.highWaterMark" ), 1u << 31, RT_DSTRING( "The cache size will be reduced to the low-water mark if the high-water mark is exceeded. The value 0 disables garbage collection. The default value is 2GB." ) );
Knob<std::string>  k_enableNoInlineSemanticType( RT_DSTRING( "o7.enableNoInline" ), "", RT_DSTRING( "Enables regular LWCA functions to call a set of OptiX API functions instead of inlining them. Valid values are 0 and 1. Default is empty, which means 0" ) );
Knob<bool>         k_enableValidationMode( RT_DSTRING( "o7.enableValidationMode" ), false, RT_DSTRING( "Enables validation mode." ) );
Knob<std::string>  k_gatherLwptiMetrics( RT_DSTRING( "launch.gatherMetrics" ), "", RT_DSTRING( "Comma separated list of LWPTI metrics to gather." ) );
Knob<bool>         k_gpuWarmupEnabled( RT_DSTRING( "rtx.gpuWarmupEnabled" ), true, RT_DSTRING( "Enable the GPU warm-up kernel." ) );
Knob<int>          k_logLevel( RT_DSTRING( "log.level" ), 5, RT_DSTRING( "Log level (0 is off; 100 is maximum verbosity)" ) );
Knob<size_t>       k_maximumDeviceMemory( RT_DSTRING( "lwca.maximumDeviceMemory" ), 0, RT_DSTRING( "Set a limit on the visible device memory. Default is 0, which means use what the driver reports." ) );

PublicKnob<bool>   k_enableD2IR( RT_PUBLIC_DSTRING( "compile.enableFeature.newBackend" ), true, RT_PUBLIC_DSTRING( "Enable new compiler backend." ) );
PublicKnob<bool>   k_enablePTXFallback( RT_PUBLIC_DSTRING( "compile.enableBackendFallback" ), false, RT_PUBLIC_DSTRING( "Enable fallback to old compiler backend." ) );
Knob<unsigned int> k_splitModuleMinBinSize( RT_DSTRING( "o7.splitModuleMinBinSize" ), 1000, RT_DSTRING( "Minimum number of basic blocks to assign in each SubModule" ) );

// clang-format on
}  // namespace

namespace optix_exp {

class DefaultLogger
{
  public:
    static void callback( unsigned int level, const char* tag, const char* message, void* cbdata )
    {
        DefaultLogger* self = static_cast<DefaultLogger*>( cbdata );
        self->callback( level, tag, message );
    }

    void callback( unsigned int level, const char* tag, const char* message )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        llog( level ) << tag << ": " << message << "\n";
    }

  private:
    std::mutex m_mutex;
};

static DefaultLogger g_defaultLogger;

//////////////////////////////////////////////////////////////////

DeviceContextLogger::DeviceContextLogger()
    : m_callback( &optix_exp::DefaultLogger::callback )
    , m_callbackData( &optix_exp::g_defaultLogger )
{
}

OptixResult DeviceContextLogger::setCallback( OptixDeviceContextOptions* options, ErrorDetails& errDetails )
{
    if( !options )
        return OPTIX_SUCCESS;
    return setCallback( options->logCallbackFunction, options->logCallbackData, options->logCallbackLevel, errDetails );
}

OptixResult DeviceContextLogger::setCallback( OptixLogCallback callbackFunction, void* callbackData, unsigned int callbackLevel, ErrorDetails& errDetails )
{
    // Check errors first before setting data
    if( callbackLevel >= LOG_LEVEL::Invalid )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, std::string( "log level " ) + std::to_string( callbackLevel )
                                                                     + " is out of bounds [0,4]" );

    return setCallbackNoCheck( callbackFunction, callbackData, callbackLevel, errDetails );
}

OptixResult DeviceContextLogger::setCallbackNoCheck( OptixLogCallback callbackFunction,
                                                     void*            callbackData,
                                                     unsigned int     callbackLevel,
                                                     ErrorDetails&    errDetails )
{
    // Note that updates of all three values happens atomically, although the documentation guarantees that only for
    // callbackFunction and callbackData, not callbackLevel. The reason for that is that the callback() method itself
    // reads m_level first without acquiring the lock.
    std::lock_guard<std::mutex> lock( m_mutex );
    if( callbackFunction )
    {
        m_callback     = callbackFunction;
        m_callbackData = callbackData;
    }
    else
    {
        m_callback     = &optix_exp::DefaultLogger::callback;
        m_callbackData = &optix_exp::g_defaultLogger;
    }
    m_level = !k_logLevel.isSet() ? callbackLevel : static_cast<unsigned int>( k_logLevel.get() );
    return OPTIX_SUCCESS;
}

void DeviceContextLogger::callback( LOG_LEVEL level, const char* tag, const char* message )
{
    // This check is not thread-safe and tools might flag this. Should not be a big problem and avoids a lock if the
    // level is set to Disabled.
    if( m_level == LOG_LEVEL::Disabled )
        return;

    OptixLogCallback callbackCopy;
    void*            callbackDataCopy;
    unsigned int     levelCopy;

    {
        std::lock_guard<std::mutex> lock( m_mutex );
        callbackCopy     = m_callback;
        callbackDataCopy = m_callbackData;
        levelCopy        = m_level;
    }

    if( level <= levelCopy && levelCopy != LOG_LEVEL::Disabled )
        callbackCopy( static_cast<unsigned int>( level ), tag, message, callbackDataCopy );
}

void DeviceContextLogger::callback( unsigned int level, const char* tag, const char* message )
{
    if( level <= m_level && m_level != LOG_LEVEL::Disabled )
        m_callback( level, tag, message, m_callbackData );
}

void DeviceContextLogger::sendError( const std::string& error )
{
    callback( LOG_LEVEL::Error, "ERROR", error.c_str() );
}

void DeviceContextLogger::sendError( const ErrorDetails& errDetails )
{
    sendError( errDetails.m_description );
}

//////////////////////////////////////////////////////////////////

LlvmStart& LlvmStart::get()
{
    // Using C++11 "magic static" for thread safety
    static LlvmStart initializer;
    return initializer;
}

LlvmStart::LlvmStart()
{
    m_started = true;
}

bool LlvmStart::started()
{
    return m_started && llvm::llvm_is_multithreaded();
}

//////////////////////////////////////////////////////////////////

OptixResult validateContextAPI( OptixDeviceContext contextAPI, DeviceContext*& context )
{
    DeviceContextLogger defaultLogger;
    if( contextAPI == nullptr )
    {
        defaultLogger.sendError( "context argument is null" );
        return OPTIX_ERROR_ILWALID_DEVICE_CONTEXT;
    }
    if( const OptixResult implCastResult = implCast( contextAPI, context ) )
    {
        defaultLogger.sendError( "context argument is not an OptixDeviceContext" );
        return implCastResult;
    }
    return OPTIX_SUCCESS;
}
//////////////////////////////////////////////////////////////////

OptixResult validateContextAPI( OptixDeviceContext contextAPI, DeviceContext*& context, char* logString, size_t* logStringSize )
{
    DeviceContextLogger defaultLogger;
    if( contextAPI == nullptr )
    {
        std::string errMsg = "context argument is null";
        defaultLogger.sendError( errMsg );
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );
        return OPTIX_ERROR_ILWALID_DEVICE_CONTEXT;
    }
    if( const OptixResult implCastResult = implCast( contextAPI, context ) )
    {
        std::string errMsg = "context argument is not an OptixDeviceContext";
        defaultLogger.sendError( errMsg );
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );
        return implCastResult;
    }
    return OPTIX_SUCCESS;
}

//////////////////////////////////////////////////////////////////

ScopedCommandList::ScopedCommandList( DeviceContext* context )
    : m_context( context )
{
}

ScopedCommandList::~ScopedCommandList()
{
    if( m_commandList )
    {
        lerr << "ScopedCommandList destructor called before destroy()\n";
    }
}

OptixResult ScopedCommandList::init( LWstream stream, ErrorDetails& errDetails )
{
    if( const RtcResult rtcResult =
            m_context->getRtcore().commandListCreateForLWDA( m_context->getRtcDeviceContext(), stream, &m_commandList ) )
        return errDetails.logDetails( rtcResult, "failed to create command list" );

    return OPTIX_SUCCESS;
}

OptixResult ScopedCommandList::destroy( ErrorDetails& errDetails )
{
    if( m_commandList == nullptr )
        return OPTIX_SUCCESS;

    const RtcResult rtcResult = m_context->getRtcore().commandListDestroy( m_commandList );
    m_commandList             = nullptr;

    if( rtcResult )
        return errDetails.logDetails( rtcResult, "failed to destroy command list" );

    return OPTIX_SUCCESS;
}


//////////////////////////////////////////////////////////////////

LwdaContextPushPop::LwdaContextPushPop( DeviceContext* context, bool allowInternalContextChange )
    : m_deviceContext( context )
    , m_tolerateCtxChange( allowInternalContextChange )
{
}

LwdaContextPushPop::~LwdaContextPushPop()
{
    if( m_previousCtx )
    {
        lerr << "LwdaContextPushPop destructor called before destroy()\n";
    }
}

OptixResult LwdaContextPushPop::init( ErrorDetails& errDetails )
{
    if( m_previousCtx )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                      "Error in LWCA context handling: Context already pushed" );

    if( LWresult result = corelib::lwdaDriver().LwCtxGetLwrrent( &m_previousCtx ) )
        return errDetails.logDetails( result, "Failed to query current LWCA context." );
    if( m_previousCtx != m_deviceContext->getLwdaContext() )
    {
        if( LWresult result = corelib::lwdaDriver().LwCtxSetLwrrent( m_deviceContext->getLwdaContext() ) )
        {
            return errDetails.logDetails( result, "Failed to push LWCA context" );
        }
    }
    return OPTIX_SUCCESS;
}

OptixResult LwdaContextPushPop::destroy( ErrorDetails& errDetails )
{
    LWcontext ctx;
    if( LWresult result = corelib::lwdaDriver().LwCtxGetLwrrent( &ctx ) )
    {
        m_previousCtx = nullptr;
        return errDetails.logDetails( result, "Failed to get the current LWCA context" );
    }
    if( !m_tolerateCtxChange && ctx != m_deviceContext->getLwdaContext() )
    {
        m_previousCtx = nullptr;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "LWCA context has changed unexpectedly" );
    }
    if( m_previousCtx != ctx )
    {
        if( LWresult result = corelib::lwdaDriver().LwCtxSetLwrrent( m_previousCtx ) )
        {
            m_previousCtx = nullptr;
            return errDetails.logDetails( result, "Failed to reset LWCA context" );
        }
    }
    m_previousCtx = nullptr;
    return OPTIX_SUCCESS;
}

//////////////////////////////////////////////////////////////////

std::mutex DeviceContext::s_mutex;

DeviceContext::DeviceContext( OptixABI abiVersion, LWcontext lwdaContext, OptixDeviceContextOptions* options )
    : OpaqueApiObject( OpaqueApiObject::ApiType::DeviceContext )
    , m_diskCacheLocation( "<USE DEFAULT>" )
    , m_diskCacheLowWaterMark( k_lowWaterMark.get() )
    , m_diskCacheHighWaterMark( k_highWaterMark.get() )
    , m_encryptionManager( new optix_exp::EncryptionManager() )
    , m_internalEncryptionManager( new optix_exp::EncryptionManager() )
    , m_ttuWatchdog( new optix_exp::WatchdogTimer() )
    , m_gpuWarmup( new optix_exp::GpuWarmup() )
    , m_lwdaContext( lwdaContext )
    , m_rtcoreUUID{}
    , m_abiVersion( abiVersion )
    // set limits based on archived ABI versions
    // TODO: Query from rtcore, Types.hpp, bug: 2628943
    // archived ABIs 20, 22 only support 1<<24 (pre fat instances)
    , m_maxInstancesPerIAS( 1 << ( abiVersion <= OptixABI::ABI_22 ? 24 : 28 ) )
    , m_maxSbtRecordsPerGAS( 1 << ( abiVersion <= OptixABI::ABI_22 ? 24 : 28 ) )
    , m_maxInstanceId( ( 1 << ( abiVersion <= OptixABI::ABI_22 ? 24 : 28 ) ) - 1 )
    , m_maxSbtOffset( ( 1 << ( abiVersion <= OptixABI::ABI_22 ? 24 : 28 ) ) - 1 )
    , m_enableNoInline( k_enableNoInlineSemanticType.get() == "1" )
    , m_enableLWPTXFallback( k_enablePTXFallback.get() )
    , m_splitModuleMinBinSize( k_splitModuleMinBinSize.get() )
    , m_validationMode( ( abiVersion < OptixABI::ABI_37 ) ?
                            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF :
                            ( options ? options->validationMode : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF ) )
{
    optix::data::acquireLoader();
    if( k_enableValidationMode.get() )
        m_validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
}

DeviceContext::~DeviceContext()
{
    optix::data::releaseLoader();
}

std::atomic<unsigned int> DeviceContext::s_serialNumber = {0};

OptixResult DeviceContext::getDeviceName( std::string& outDeviceName, ErrorDetails& errDetails )
{
    LWresult lwResult;

    LWdevice lwDevice;
    lwResult = corelib::lwdaDriver().LwCtxGetDevice( &lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails( lwResult, "Could not get the current LWCA device" );

    char deviceName[256];
    lwResult = corelib::lwdaDriver().LwDeviceGetName( deviceName, sizeof( deviceName ), lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails( lwResult, "Could not get LWCA device name" );

    outDeviceName = deviceName;

    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::getPCIBusID( std::string& outPCIBusID, ErrorDetails& errDetails )
{
    LWresult lwResult;

    LWdevice lwDevice;
    lwResult = corelib::lwdaDriver().LwCtxGetDevice( &lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails( lwResult, "Could not get the current LWCA device" );

    char pciBusID[256];
    lwResult = corelib::lwdaDriver().LwDeviceGetPCIBusId( pciBusID, sizeof( pciBusID ), lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails( lwResult, "Could not get LWCA device PCI bus ID" );

    outPCIBusID = pciBusID;

    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::init( ErrorDetails& errDetails )
{
    std::string deviceName;
    if( OptixResult result = getDeviceName( deviceName, errDetails ) )
        return result;

    // The LWTX row name starts with "OptiX API/", which Nsight Systems treats
    // as an expandable parent row that contains all rows associated with OptiX
    // contexts. After that, each OptiX row consists of the address of that
    // context (to guarantee uniqueness when multiple contexts are used on a
    // single device) and the name of the context's associated LWCA device.
    std::string lwtxRowName = corelib::stringf( "OptiX API/[%p] %s", this, deviceName.c_str() );

    // Initialize LWTX
    m_lwtxProfiler.reset( new LWTXProfiler( lwtxRowName.c_str() ) );

    if( OptixResult result = m_launchResources.init( k_launchResourceBufferSize.get(), k_launchResourceNumEvents.get(), errDetails ) )
        return result;

    LlvmStart& initLlvm = LlvmStart::get();
    if( !initLlvm.started() )
    {
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                      "Could not set LLVM to multi-threaded mode or it was not built with "
                                      "support" );
    }

    if( initializeDiskCache( errDetails ) )
    {
        // Issue a warning to the log. User needs to call optixDeviceContextSetCacheEnabled
        // to get error code.
        m_logger.callback( DeviceContextLogger::LOG_LEVEL::Warning, "WARNING", errDetails.m_description.c_str() );
        // disable the disk cache if initialization fails.
        m_diskCache.reset();
    }

    if( OptixResult result = m_internalEncryptionManager->setOptixSalt( secrets::optixSalt, SALT_LENGTH, errDetails ) )
        return result;
    if( OptixResult result = m_internalEncryptionManager->setVendorSalt( secrets::vendorSalt, SALT_LENGTH, errDetails ) )
        return result;
    if( OptixResult result = m_internalEncryptionManager->setPublicVendorKey( secrets::vendorPublicKey, PUBLIC_KEY_LENGTH, errDetails ) )
        return result;

    if( OptixResult result = initD2IREnabled( errDetails ) )
        return result;

    if( !k_gatherLwptiMetrics.get().empty() )
    {
        if( const OptixResult result = m_lwptiProfiler.initialize( m_lwdaContext, k_gatherLwptiMetrics.get(), errDetails ) )
            return result;

        std::string pciBusId;
        if( OptixResult result = getPCIBusID( pciBusId, errDetails ) )
            return result;

        // Name each LWPTI file file with the devices's PCI bus ID followed by its name.
        std::string lwptiFileName = corelib::stringf( "%s-%s.json", pciBusId.c_str(), deviceName.c_str() );
        // Replace all spaces and colons in the file name with dashes.
        for( char& c : lwptiFileName )
        {
            if( c == ' ' || c == ':' )
                c = '-';
        }

        const bool openWasSuccessful = m_lwptiProfiler.openOutputFile( lwptiFileName );
        if( !openWasSuccessful )
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                          "Could not open " + lwptiFileName + " for LWPTI profiling output." );
    }

    if( k_gpuWarmupEnabled.get() && getComputeCapability() == 75 && hasTTU() )
    {
        // Load the warm-up kernel from an embedded fatbin.
        if( OptixResult result = getGpuWarmup().init( errDetails ) )
            return result;
    }

#if defined( __linux__ ) || defined( _WIN32 )
    // Log a warning when GPU memory exceeds host memory
    size_t       gpuTotal  = 0;
    size_t       hostTotal = prodlib::getTotalSystemMemoryInBytes();
    const size_t memLimit  = k_maximumDeviceMemory.get();
    corelib::lwdaDriver().LwMemGetInfo( nullptr, &gpuTotal );
    gpuTotal = ( memLimit > 0 && memLimit < gpuTotal ) ? memLimit : gpuTotal;
    if( gpuTotal > hostTotal )
    {
        const double gpuGiB  = static_cast<double>( gpuTotal ) / 1073741824.0;
        const double hostGiB = static_cast<double>( hostTotal ) / 1073741824.0;
        m_logger.callback(
            DeviceContextLogger::LOG_LEVEL::Print, "MEMORY",
            corelib::stringf( "GPU memory capacity exceeds host memory (GPU: %0.2f GiB, host: %0.2f GiB). "
                              "Please ensure that host allocations do not exceed the amount of "
                              "physical memory available.",
                              gpuGiB, hostGiB )
                .c_str() );
    }
#endif
    std::string magic;
    if( corelib::getelw( "OPTIX_O7_ENABLE_METRICS", magic ) && magic == "8675309" )
    {
        m_metrics.reset( new Metrics( this ) );

        std::string pciBusId;
        if( OptixResult result = getPCIBusID( pciBusId, errDetails ) )
            return result;

        const size_t begin = pciBusId.find_last_of( ':' );
        const size_t end   = pciBusId.find_first_of( '.' );
        if( begin == std::string::npos || end == std::string::npos || end < begin )
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Malformed LWCA device PCI bus ID string" );

        std::string deviceId  = pciBusId.substr( begin + 1, ( end - 1 ) - begin );
        std::string contextId = corelib::stringf( "%p%03d", this, this->getSerialNumber() );

        if( OptixResult result = m_metrics->init( deviceName, deviceId, contextId, errDetails ) )
            return result;
    }

    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::destroy( ErrorDetails& errDetails )
{
    llog( 6 ) << "launch resources pressure count, ring (" << m_launchResources.m_ringPressureCount << "), event ("
              << m_launchResources.m_eventPressureCount << ")\n";

    OptixResultOneShot result;
    result += destroyPipelines( errDetails );
    result += destroyProgramGroups( errDetails );
    result += destroyModules( errDetails );
    result += destroyDenoisers( errDetails );

    LwdaContextPushPop lwCtx( this );
    result += lwCtx.init( errDetails );

    if( isLwptiProfilingEnabled() )
        result += m_lwptiProfiler.deinitialize( errDetails );

    result += m_launchResources.destroy( errDetails );

    if( getRtcDeviceContext() )
    {
        if( const RtcResult rtcResult = m_rtcore.deviceContextDestroy( getRtcDeviceContext() ) )
        {
            result += errDetails.logDetails( rtcResult, "Error destroying RTX device context" );
        }
    }

    if( m_metrics )
    {
        result += m_metrics->destroy( errDetails );
        m_metrics.reset();
    }

    if( m_diskCache )
    {
        result += m_diskCache->destroy( m_logger );
        m_diskCache.reset();
    }

    if( m_gpuWarmup )
    {
        result += m_gpuWarmup->destroy( errDetails );
        m_gpuWarmup.reset();
    }

    result += lwCtx.destroy( errDetails );

    return result;
}

int DeviceContext::getCallableParamRegCount() const
{
    return m_callableParamRegCount;
}

OptixResult DeviceContext::registerModule( Module* module, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_modulesMutex );
    m_modules.addItem( module );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::unregisterModule( Module* module, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_modulesMutex );
    m_modules.removeItem( module );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::destroyModules( ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_modulesMutex );

    OptixResultOneShot result;
    for( Module* module : m_modules )
        result += module->destroyWithoutUnregistration( errDetails );

    m_modules.clear();

    return result;
}

OptixResult DeviceContext::registerPipeline( Pipeline* pipeline, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_pipelinesMutex );
    m_pipelines.addItem( pipeline );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::unregisterPipeline( Pipeline* pipeline, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_pipelinesMutex );
    m_pipelines.removeItem( pipeline );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::destroyPipelines( ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_pipelinesMutex );

    OptixResultOneShot result;
    for( Pipeline* pipeline : m_pipelines )
        result += pipeline->destroyWithoutUnregistration( errDetails );

    m_pipelines.clear();

    return result;
}

OptixResult DeviceContext::registerDenoiser( Denoiser* denoiser, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_denoisersMutex );
    m_denoisers.addItem( denoiser );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::unregisterDenoiser( Denoiser* denoiser, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_denoisersMutex );
    m_denoisers.removeItem( denoiser );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::destroyDenoisers( ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_denoisersMutex );

    OptixResultOneShot result;
    for( Denoiser* denoiser : m_denoisers )
        result += denoiser->destroyWithoutUnregistration( errDetails );

    m_denoisers.clear();

    return result;
}

OptixResult DeviceContext::registerProgramGroup( ProgramGroup* programGroup, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_programGroupsMutex );
    m_programGroups.addItem( programGroup );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::unregisterProgramGroup( ProgramGroup* programGroup, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_programGroupsMutex );
    m_programGroups.removeItem( programGroup );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::destroyProgramGroups( ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_programGroupsMutex );

    OptixResultOneShot result;
    for( ProgramGroup* programGroup : m_programGroups )
        result += programGroup->destroyWithoutUnregistration( errDetails );

    m_programGroups.clear();

    return result;
}

int DeviceContext::getComputeCapability() const
{
    return m_lwComputeCapabilityMajor * 10 + m_lwComputeCapabilityMinor;
}

int DeviceContext::getMaxThreadsPerMultiProcessor() const
{
    return m_lwMaxThreadsPerMultiProcessor;
}

int DeviceContext::getMaxThreadsPerBlock() const
{
    return m_lwMaxThreadsPerBlock;
}

int DeviceContext::getMultiProcessorCount() const
{
    return m_lwMultiProcessorCount;
}

OptixResult DeviceContext::setLwdaDeviceProperties( ErrorDetails& errDetails )
{
    LWresult lwResult;

    // TODO Is it correct not to push/pop the context here (this was done
    // in the original version with direct driver API calls, but fails with
    // the wrapper)?
    LWdevice lwDevice;
    lwResult = corelib::lwdaDriver().LwCtxGetDevice( &lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails( lwResult, "Could not get the current LWCA device" );

    lwResult = corelib::lwdaDriver().LwDeviceGetAttribute( &m_lwComputeCapabilityMajor,
                                                           LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails(
            lwResult, "Could not get the LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR LWCA device attribute" );

    lwResult = corelib::lwdaDriver().LwDeviceGetAttribute( &m_lwComputeCapabilityMinor,
                                                           LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails(
            lwResult, "Could not get the LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR LWCA device attribute" );

    lwResult = corelib::lwdaDriver().LwDeviceGetAttribute( &m_lwMaxThreadsPerMultiProcessor,
                                                           LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails(
            lwResult, "Could not get the LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR LWCA device attribute" );

    lwResult = corelib::lwdaDriver().LwDeviceGetAttribute( &m_lwMaxThreadsPerBlock,
                                                           LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails(
            lwResult, "Could not get the LW_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK LWCA device attribute" );

    lwResult = corelib::lwdaDriver().LwDeviceGetAttribute( &m_lwMultiProcessorCount,
                                                           LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, lwDevice );
    if( lwResult != LWDA_SUCCESS )
        return errDetails.logDetails(
            lwResult, "Could not get the LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT LWCA device attribute" );

    unsigned int arch, impl;
    lwResult = corelib::lwdaDriver().LwDeviceGetArchImpl( lwDevice, &arch, &impl );
    if( lwResult == LWDA_SUCCESS )
    {
        m_architecture               = arch;
        m_architectureImplementation = impl;
    }

    unsigned char supportsTTU = 0;
    lwResult                  = corelib::lwdaDriver().LwDeviceQueryTTU( &supportsTTU, lwDevice );
    if( lwResult == LWDA_SUCCESS )
    {
        m_hasTTU = supportsTTU != 0;
    }

    if( k_forceTTUenabled.get() )
    {
        if( !m_hasTTU )
        {
            getLogger().callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "KNOB_OVERRIDE",
                                  RT_DSTRING( "TTU has been force-enabled" ) );
        }
        m_hasTTU = true;
    }

    if( k_disableTTU.get() )
        m_hasTTU = false;

    m_hasMotionTTU = m_hasTTU && ( m_architecture > LW_CFG_ARCHITECTURE_TU100 );

    if( k_disableMTTU.get() )
        m_hasMotionTTU = false;

#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
    m_hasDisplacedMicromeshTTU = m_hasTTU && ( m_architecture > 0x170 ); // 0x170 is Ampere, but LW_CFG_ARCHITECTURE_GA100 is not defined (see Lwcm.h)

    if( k_disableDisplacedMicromeshTTU.get() )
        m_hasDisplacedMicromeshTTU = false;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::getRtcDeviceProperties( ErrorDetails& errDetails )
{
    Rtlw64 rtcoreValue = 0;
    if( const RtcResult result =
            m_rtcore.deviceContextGetLimit( getRtcDeviceContext(), RTC_LIMIT_MAX_MAX_TRACE_RELWRSION_DEPTH, &rtcoreValue ) )
        return errDetails.logDetails( result, "Unable to retrieve OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH" );
    m_rtcMaxTraceRelwrsionDepth = static_cast<unsigned int>( rtcoreValue );

    if( const RtcResult result = m_rtcore.rtcGetBuildUUID( m_rtcoreUUID ) )
        return errDetails.logDetails( result, "Unable to retrieve RTX build UUID." );

    Rtlw64 maxParamRegisters = 0;
    if( const RtcResult rtcResult =
            m_rtcore.deviceContextGetLimit( getRtcDeviceContext(), RTC_LIMIT_MAX_CALLABLE_PARAM_REGISTERS, &maxParamRegisters ) )
        return errDetails.logDetails( rtcResult, "Unable to retrieve max parameter register count" );
    m_callableParamRegCount = static_cast<int>( maxParamRegisters );

    Rtlw64 maxTraversalGraphDepth = 0;
    if( const RtcResult rtcResult =
            m_rtcore.deviceContextGetLimit( getRtcDeviceContext(), RTC_LIMIT_MAX_MAX_TRAVERSAL_GRAPH_DEPTH, &maxTraversalGraphDepth ) )
        return errDetails.logDetails( rtcResult, "Unable to retrieve max traversable graph depth" );
    m_maxTraversalGraphDepth = static_cast<int>( maxTraversalGraphDepth );

    return OPTIX_SUCCESS;
}

bool DeviceContext::isDiskCacheActive() const
{
    std::lock_guard<std::mutex> lock( m_diskCacheConfigMutex );
    return m_diskCache && m_diskCache->isActive();
}

OptixResult DeviceContext::initializeDiskCache( ErrorDetails& errDetails )
{
    // This is either called on context creation or from setDiskCacheEnabled()
    // or setDiskCacheLocation() and both of those have m_diskCacheConfigMutex locked.
    if( m_diskCache )
    {
        // Explicitly destroy the DiskCache to force the database to close.
        m_diskCache->destroy( m_logger );
    }

    m_diskCache.reset( new optix_exp::DiskCache() );
    OptixResult result = m_diskCache->init( m_logger, errDetails, CACHE_DB_FILE_NAME, m_diskCacheLocation,
                                            m_diskCacheLowWaterMark, m_diskCacheHighWaterMark );

    // Update the memory limits, in case they have been overridden by OPTIX_CACHE_MAXSIZE
    m_diskCache->getMemoryLimits( m_diskCacheLowWaterMark, m_diskCacheHighWaterMark );

    // Store resolved disk cache path, so the user can find out where it was
    // supposed to be located in the case of error (instead of potentially
    // getting "<USE DEFAULT>" in that case).
    m_diskCacheLocation = m_diskCache->getPath( CACHE_DB_FILE_NAME );
    if( result )
    {
        // disable the disk cache if initialization fails.
        m_diskCache.reset();
        return result;
    }

    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::setDiskCacheEnabled( bool enabled, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_diskCacheConfigMutex );
    if( m_diskCache && !m_diskCache->isDisabledByElwironment() )
    {
        m_diskCache->setIsActive( enabled );
    }
    else if( enabled )
    {
        return initializeDiskCache( errDetails );
    }
    // We do not need to store enabled == false if no disk cache is initialized;
    // the only way to create disk cache in that case is to call setCacheEnabled again
    // with enabled == true.
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::setDiskCacheLocation( const std::string& location, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_diskCacheConfigMutex );
    m_diskCacheLocation = location;
    if( !m_diskCache )
    {
        // Last initialization attempt failed. We will use the location
        // we just stored to initialize the disk cache the next time
        // the user calls optixDeviceContextSetCacheEnabled
        return OPTIX_SUCCESS;
    }
    else if( m_diskCache->isDisabledByElwironment() )
    {
        // Don't attempt to set the path if the disk cache has been disabled by
        // the environment, and clear the location.
        m_diskCacheLocation = "";

        getLogger().callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Print, "DISK CACHE",
                              "The disk cache has been disabled by setting the environment variable "
                              "OPTIX_CACHE_MAXSIZE=0.  The API call will be ignored." );
    }
    else if( OptixResult result = m_diskCache->setPath( m_diskCacheLocation, CACHE_DB_FILE_NAME, m_logger, errDetails ) )
    {
        // Reset the disk cache, to make sure that the next call
        // to enable will initialize the disk cache again (with potential failure).
        m_diskCache.reset();
        return result;
    }
    return OPTIX_SUCCESS;
}

std::string DeviceContext::getDiskCacheLocation() const
{
    std::lock_guard<std::mutex> lock( m_diskCacheConfigMutex );
    if( m_diskCache )
        return m_diskCache->getPath( CACHE_DB_FILE_NAME );

    return m_diskCacheLocation;
}

OptixResult DeviceContext::getLibraryPath( std::string& libPath, ErrorDetails& errDetails ) const
{
#ifdef _WIN32
    const char* const soname = "lwoptix.dll";
#else
    const char* const soname = "liblwoptix.so.1";
#endif
    corelib::ExelwtableModule em( soname );
    if( !em.init() )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Cannot get library path" );
    libPath = em.getPath( "optixQueryFunctionTable" );

    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::setDiskCacheMemoryLimits( size_t lowWaterMark, size_t highWaterMark, ErrorDetails& errDetails )
{
    if( lowWaterMark > highWaterMark && lowWaterMark > 0 && highWaterMark > 0 )
        return errDetails.logDetails(
            OPTIX_ERROR_ILWALID_VALUE,
            corelib::stringf( "highWaterMark (given: %zu) cannot be smaller than lowWaterMark (given: %zu)", highWaterMark, lowWaterMark )
                .c_str() );

    std::lock_guard<std::mutex> lock( m_diskCacheConfigMutex );

    // Don't try to set the limits if the size has been set by the environment variable
    if( m_diskCache && m_diskCache->isSizeSetByElwironment() )
    {
        getLogger().callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Print, "DISK CACHE",
                              "The disk cache memory limits have been set by the OPTIX_CACHE_MAXSIZE "
                              "environment variable.  The API call will be ignored." );
        return OPTIX_SUCCESS;
    }

    m_diskCacheLowWaterMark  = lowWaterMark;
    m_diskCacheHighWaterMark = highWaterMark;
    if( !m_diskCache )
        return OPTIX_SUCCESS;
    return m_diskCache->setMemoryLimits( lowWaterMark, highWaterMark, m_logger, errDetails );
}

void DeviceContext::getDiskCacheMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const
{
    std::lock_guard<std::mutex> lock( m_diskCacheConfigMutex );
    lowWaterMark  = m_diskCacheLowWaterMark;
    highWaterMark = m_diskCacheHighWaterMark;
}

const Rtlw32* DeviceContext::getRtcoreUUID() const
{
    return m_rtcoreUUID;
}

OptixResult DeviceContext::setNoInlineEnabled( bool enabled, ErrorDetails& errDetails )
{
    if( k_enableNoInlineSemanticType.get().empty() )
    {
        std::lock_guard<std::mutex> lock( m_enableNoInlineMutex );
        if( m_enableNoInlineImmutable && m_enableNoInline != enabled )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( R"msg(Cannot modify "CompileNoInline" property on device context after creating a module, lwrrently enabled: %s)msg",
                                                            m_enableNoInline ? "true" : "false" ) );
        m_enableNoInline = enabled;
    }
    else
    {
        lwarn << "Knob " << k_enableNoInlineSemanticType.getName() << " has been explicitly set to value "
              << k_enableNoInlineSemanticType.get()
              << ", which overrides all other attempts to enable/disable compilation of noinline functions." << std::endl;
    }
    return OPTIX_SUCCESS;
}

bool DeviceContext::isNoInlineEnabled() const
{
    return m_enableNoInline;
}

void DeviceContext::makeNoInlineImmutable()
{
    std::lock_guard<std::mutex> lock( m_enableNoInlineMutex );
    m_enableNoInlineImmutable = true;
}

OptixResult DeviceContext::initD2IREnabled( ErrorDetails& errDetails )
{
    // Priority
    // 1. Set knob
    // 2. OPTIX_FORCE_DEPRECATED_COMPILER
    // 3. User set value
    // 4. Default knob
    if( k_enableD2IR.isSet() )
    {
        m_enableD2IR.store( k_enableD2IR.get() );
        m_canSetD2IRViaAPI = false;
        return OPTIX_SUCCESS;
    }
    std::string elwVarValue;
    if( corelib::getelw( OPTIX_FORCE_DEPRECATED_COMPILER_STR, elwVarValue ) )
    {
        unsigned int value = -1;
        sscanf( elwVarValue.c_str(), "%u", &value );

        switch( value )
        {
            case OptixForceDeprecatedCompilerValues::LWVM7_D2IR:
                m_enableD2IR.store( true );
                break;
            case OptixForceDeprecatedCompilerValues::LWVM7_LWPTX:
            case OptixForceDeprecatedCompilerValues::LWVM34_LWPTX:
                m_enableD2IR.store( false );
                break;
            default:
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, std::string( "Unknown value for " ) + OPTIX_FORCE_DEPRECATED_COMPILER_STR
                                                                             + ": " + elwVarValue );
        }
        m_canSetD2IRViaAPI = false;
        return OPTIX_SUCCESS;
    }
    m_enableD2IR.store( k_enableD2IR.get() );
    return OPTIX_SUCCESS;
}

OptixResult DeviceContext::setD2IREnabled( bool enabled, ErrorDetails& errDetails )
{
    if( m_canSetD2IRViaAPI )
    {
        m_enableD2IR.store( enabled );
    }
    else
    {
        std::string elwVarValue;
        if( k_enableD2IR.isSet() )
        {
            getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "KNOB_OVERRIDE",
                                  corelib::stringf( "Knob %s has been explicitly set to value %d, which overrides "
                                                    "all other attempts to enable/disable the new backend.",
                                                    k_enableD2IR.getName().c_str(), k_enableD2IR.get() )
                                      .c_str() );
        }
        else if( corelib::getelw( OPTIX_FORCE_DEPRECATED_COMPILER_STR, elwVarValue ) )
        {
            getLogger().callback(
                DeviceContextLogger::LOG_LEVEL::Warning, "ELW_OVERRIDE",
                corelib::stringf( "Environment variable %s has been explicitly set to value %s, which overrides all "
                                  "other attempts to enable/disable the new backend.",
                                  OPTIX_FORCE_DEPRECATED_COMPILER_STR, elwVarValue.c_str() )
                    .c_str() );
        }
    }
    return OPTIX_SUCCESS;
}

bool DeviceContext::isD2IREnabled() const
{
    return m_enableD2IR.load();
}

OptixResult DeviceContext::setLWPTXFallbackEnabled( bool enabled, ErrorDetails& errDetails )
{
    if( !k_enablePTXFallback.isSet() )
        m_enableLWPTXFallback.store( enabled );
    else
    {
        getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "KNOB_OVERRIDE",
                              corelib::stringf( "Knob %s has been explicitly set to value %d, which overrides all "
                                                "other attempts to enable/disable the new backend.",
                                                k_enablePTXFallback.getName().c_str(), k_enablePTXFallback.get() )
                                  .c_str() );
    }
    return OPTIX_SUCCESS;
}

bool DeviceContext::isLWPTXFallbackEnabled() const
{
    return m_enableLWPTXFallback.load();
}

OptixResult DeviceContext::setSplitModuleMinBinSize( unsigned int minBinSize, ErrorDetails& errDetails )
{
    if( !k_splitModuleMinBinSize.isSet() )
        m_splitModuleMinBinSize.store( minBinSize ? minBinSize : k_splitModuleMinBinSize.get() );
    else
    {
        getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "KNOB_OVERRIDE",
                              corelib::stringf( "Knob %s has been explicitly set to value %u, which overrides all "
                                                "other attempts to set value.",
                                                k_splitModuleMinBinSize.getName().c_str(), k_splitModuleMinBinSize.get() )
                                  .c_str() );
    }
    return OPTIX_SUCCESS;
}

unsigned int DeviceContext::getSplitModuleMinBinSize() const
{
    return m_splitModuleMinBinSize.load();
}

OptixDeviceContextValidationMode DeviceContext::getValidationMode() const
{
    return m_validationMode;
}

unsigned int DeviceContext::registerTraversable( LWdeviceptr buffer )
{
    unsigned int version = 0;
    {
        std::lock_guard<std::mutex> lock( m_registeredTraversableMutex );

        version = m_registeredTraversableVersion;

        auto it = m_registeredTraversableMap.insert( std::make_pair( buffer, version ) );

        if( it.second == false )
        {
            it.first->second = ++m_registeredTraversableVersion;
        }
    }

    return version;
}

void DeviceContext::startAsyncTimer( const std::string& name, lwdaStream_t stream )
{
    if( m_metrics )
        m_metrics->startAsyncTimer( name, stream );
}

void DeviceContext::stopAsyncTimerAndRecordMetric( const std::string& name, lwdaStream_t stream )
{
    if( m_metrics )
        m_metrics->stopAsyncTimerAndRecordMetric( name, stream );
}

bool DeviceContext::hasValidationModeDebugExceptions() const
{
    return m_validationMode == OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
}

bool DeviceContext::hasValidationModeSpecializationConsistency() const
{
    return m_validationMode == OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
}

bool DeviceContext::hasValidationModeCheckStreamState() const
{
    return m_validationMode == OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
}

bool DeviceContext::hasValidationModeCheckLwrrentLwdaContext() const
{
    return m_validationMode == OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
}

static OptixResult createDeviceContext( OptixABI                   abiVersion,
                                        LWcontext                  fromContext,
                                        OptixDeviceContextOptions* options,
                                        OptixDeviceContext*&       contextAPI,
                                        optix_exp::ErrorDetails&   errDetails )
{
    // Finalize knobs. Error string is sent to the logger once we have a logger further down.
    std::string knobsErrorString;
    knobRegistry().finalizeKnobs( knobsErrorString );

    if( fromContext == nullptr )
    {
        LWcontext lwCtx;
        LWresult  lwRes = corelib::lwdaDriver().LwCtxGetLwrrent( &lwCtx );
        if( lwRes != LWDA_SUCCESS )
            return errDetails.logDetails(
                lwRes, "Specified 0 as the LWCA context, but the current LWCA context could not be queried." );
        if( lwCtx == nullptr )
            return errDetails.logDetails( OPTIX_ERROR_LWDA_NOT_INITIALIZED,
                                          "Specified 0 as the LWCA context, but the current LWCA context could not be "
                                          "queried." );
        fromContext = lwCtx;
    }

    std::unique_ptr<optix_exp::DeviceContext> context( new optix_exp::DeviceContext( abiVersion, fromContext, options ) );

    if( OptixResult result = context->getLogger().setCallback( options, errDetails ) )
        return result;

    // Log knob errors as warnings. Error messages that were generated during knob processing are not considered serious
    // enough to report them as errors here, only as warnings.
    if( !knobsErrorString.empty() )
        context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "KNOBS", knobsErrorString.c_str() );

    // Log non-default knobs.
    std::ostringstream nonDefaultKnobs;
    knobRegistry().printNonDefaultKnobs( nonDefaultKnobs );
    context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Print, "KNOBS", nonDefaultKnobs.str().c_str() );

    LwdaContextPushPop lwCtx( context.get() );
    if( OptixResult result = lwCtx.init( errDetails ) )
    {
        return result;
    }

    if( OptixResult result = context->setLwdaDeviceProperties( errDetails ) )
    {
        lwCtx.destroy( errDetails );
        // Destroy the context.  This could generate more errors, but we will just log
        // them and move on.
        context->destroy( errDetails );
        return result;
    }

    {
        if( const RtcResult result = context->getRtcore().initializeRTCoreLibraryWithKnobs() )
        {
            lwCtx.destroy( errDetails );
            return errDetails.logDetails( result, "Error initializing RTX library" );
        }

        Rtlw64 headerSize;
        if( const RtcResult result = context->getRtcore().getSbtRecordHeaderSize( &headerSize ) )
        {
            lwCtx.destroy( errDetails );
            return errDetails.logDetails( result, "Error retrieving RTX header size" );
        }
        context->setSbtHeaderSize( static_cast<unsigned int>( headerSize ) );
        if( OPTIX_SBT_RECORD_HEADER_SIZE < context->getSbtHeaderSize() )
        {
            lwCtx.destroy( errDetails );
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                          "Error: OptiX SBT Header size ( " + std::to_string( OPTIX_SBT_RECORD_HEADER_SIZE )
                                              + " ) is too small.  Needs to be at least " + std::to_string( headerSize ) );
        }

        RtcDeviceProperties deviceProperties = {};
        // Increment version number to force rtcore to generate a new identifier,
        // and prevent backward compatibility (even without having to rebuild rtcore.)
        deviceProperties.productIdentifier  = {'O', 'X', '7', /* version number: */ 0};
        deviceProperties.chipArchitecture   = context->getArchitecture();
        deviceProperties.chipImplementation = context->getArchitectureImplementation();
        deviceProperties.hasTTU             = context->hasTTU() ? 1 : 0;
        static const unsigned int* version  = optix::getOptixUUID();
        memcpy( &deviceProperties.productUuid, version, sizeof( unsigned int ) * 4 );

        RtcDeviceContext rtcDeviceContext;
        if( const RtcResult result = context->getRtcore().deviceContextCreateForLWDA( fromContext, &deviceProperties, &rtcDeviceContext ) )
        {
            lwCtx.destroy( errDetails );
            return errDetails.logDetails( result, "Error creating RTX context" );
        }
        context->setRtcDeviceContext( rtcDeviceContext );
    }

    if( OptixResult result = context->getRtcDeviceProperties( errDetails ) )
    {
        lwCtx.destroy( errDetails );
        // Destroy the context.  This could generate more errors, but we will just log
        // them and move on.
        context->destroy( errDetails );
        return result;
    }

    if( OptixResult result = context->init( errDetails ) )
    {
        lwCtx.destroy( errDetails );
        context->destroy( errDetails );
        return result;
    }
    if( OptixResult result = lwCtx.destroy( errDetails ) )
    {
        context->destroy( errDetails );
        return result;
    }

    *contextAPI = optix_exp::apiCast( context.release() );
    return OPTIX_SUCCESS;
}

static bool checkABIVersion( int abiVersion )
{
    switch( (OptixABI)abiVersion )
    {
        case OptixABI::ABI_18:
        case OptixABI::ABI_19:
        case OptixABI::ABI_20:
        case OptixABI::ABI_21:
        case OptixABI::ABI_22:
        case OptixABI::ABI_23:
        case OptixABI::ABI_24:
        case OptixABI::ABI_25:
        case OptixABI::ABI_26:
        case OptixABI::ABI_27:
        case OptixABI::ABI_28:
        case OptixABI::ABI_29:
        case OptixABI::ABI_30:
        case OptixABI::ABI_31:
        case OptixABI::ABI_32:
        case OptixABI::ABI_33:
        case OptixABI::ABI_34:
        case OptixABI::ABI_35:
        case OptixABI::ABI_36:
        case OptixABI::ABI_37:
        case OptixABI::ABI_38:
        case OptixABI::ABI_39:
        case OptixABI::ABI_40:
        case OptixABI::ABI_41:
        case OptixABI::ABI_42:
        case OptixABI::ABI_43:
        case OptixABI::ABI_44:
        case OptixABI::ABI_45:
        case OptixABI::ABI_46:
        case OptixABI::ABI_47:
        case OptixABI::ABI_48:
        case OptixABI::ABI_49:
        case OptixABI::ABI_50:
        case OptixABI::ABI_51:
        case OptixABI::ABI_52:
        case OptixABI::ABI_53:
        case OptixABI::ABI_54:
        case OptixABI::ABI_55:
        case OptixABI::ABI_56:
        case OptixABI::ABI_57:
        case OptixABI::ABI_58:
        case OptixABI::ABI_59:
        case OptixABI::ABI_60:
            return true;
    }
    return false;
}

static OptixResult optixDeviceContextCreateImpl( LWcontext                  fromContext,
                                                 OptixDeviceContextOptions* options,
                                                 OptixDeviceContext*        contextAPI,
                                                 int                        abiVersion )
{
    optix_exp::DeviceContextLogger clog;
    optix_exp::ErrorDetails        errDetails;
    OptixResult                    result = clog.setCallback( options, errDetails );
    if( result )
        return result;
    OPTIX_CHECK_NULL_ARGUMENT( contextAPI );
    if( !checkABIVersion( abiVersion ) )
    {
        clog.sendError( "Invalid ABI version: " + std::to_string( abiVersion ) );
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    *contextAPI = nullptr;
    try
    {
        result = createDeviceContext( (OptixABI)abiVersion, fromContext, options, contextAPI, errDetails );
        if( result )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;
    return OPTIX_SUCCESS;
}

// For internal use.
OptixResult optixDeviceContextCreate_lwrrent( LWcontext fromContext, OptixDeviceContextOptions* options, OptixDeviceContext* contextAPI )
{
    return optix_exp::optixDeviceContextCreateImpl( fromContext, options, contextAPI, OPTIX_ABI_VERSION );
}

}  // end namespace optix_exp

#define OPTIX_CREATE_CREATE_FUNCTION_NAME( x ) optixDeviceContextCreate_##x

// Use this macro to define a new version optixDeviceContextCreate on OPTIX_ABI_VERSION change.
// See exp/functionTable/functionTable.cpp for details.
// clang-format off
#define OPTIX_DEVICE_CONTEXT_CREATE_IMPL( ABI_VERSION )                                                     \
extern "C" OptixResult OPTIX_CREATE_CREATE_FUNCTION_NAME( ABI_VERSION )( LWcontext fromContext, OptixDeviceContextOptions* options, OptixDeviceContext* contextAPI ) \
{                                                                                                           \
        return optix_exp::optixDeviceContextCreateImpl( fromContext, options, contextAPI, ABI_VERSION );    \
}
// clang-format on

// Define optixDeviceContextCreate for current ABI version.

// functionTble_20
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 18 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 19 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 20 );
// functionTable_22
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 21 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 22 );
// functionTable_25
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 23 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 24 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 25 );
// functionTable_38
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 26 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 27 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 28 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 29 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 30 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 31 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 32 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 33 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 34 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 35 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 36 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 37 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 38 );
// functionTable_52
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 39 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 40 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 41 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 42 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 43 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 44 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 45 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 46 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 47 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 48 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 49 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 50 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 51 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 52 );
// functionTable_lwrrent
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 53 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 54 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 55 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 56 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 57 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 58 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 59 );
OPTIX_DEVICE_CONTEXT_CREATE_IMPL( 60 );


extern "C" OptixResult optixDeviceContextDestroy( OptixDeviceContext contextAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = context->destroy( errDetails );
        if( result )
            clog.sendError( errDetails );
        delete( context );
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;
}

namespace optix_exp {
static OptixResult getDeviceProperty( DeviceContext* context, OptixDeviceProperty property, void* value, size_t size, ErrorDetails& errDetails )
{
    switch( property )
    {
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getRtcMaxTraceRelwrsionDepth();
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getMaxSceneGraphDepth();
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getMaxPrimsPerGAS();
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getMaxSbtRecordsPerGAS();
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getMaxInstancesPerIAS();
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getMaxInstanceId();
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getNumBitsInstanceVisibilityMask();
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_RTCORE_VERSION:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_RTCORE_VERSION "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->hasMotionTTU() ? 20 : ( context->hasTTU() ? 10 : 0 );
            return OPTIX_SUCCESS;
        }
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET:
        {
            if( size != sizeof( unsigned int ) )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "size of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET "
                                              "needs to be sizeof( unsigned int ) or "
                                                  + std::to_string( sizeof( unsigned int ) ) );
            *reinterpret_cast<unsigned int*>( value ) = context->getMaxSbtOffset();
            return OPTIX_SUCCESS;
        }
    }
    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                  corelib::stringf( "Requested invalid device property: 0x%X", static_cast<int>( property ) ) );
}
}  // end namespace optix_exp

extern "C" OptixResult optixDeviceContextGetProperty( OptixDeviceContext contextAPI, OptixDeviceProperty property, void* value, size_t size )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_GET_PROPERTY );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( value );
    OPTIX_CHECK_ZERO_ARGUMENT( size );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result = getDeviceProperty( context, property, value, size, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDeviceContextSetLogCallback( OptixDeviceContext contextAPI,
                                                         OptixLogCallback   callbackFunction,
                                                         void*              callbackData,
                                                         unsigned int       callbackLevel )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_SET_LOG_CALLBACK );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult result = context->getLogger().setCallback( callbackFunction, callbackData, callbackLevel, errDetails );
        if( result )
        {
            context->getLogger().sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDeviceContextSetCacheEnabled( OptixDeviceContext contextAPI, int enabled )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_SET_CACHE_ENABLED );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    optix_exp::ErrorDetails         errDetails;
    if( enabled < 0 || enabled > 1 )
    {
        clog.sendError( "\"enabled\" must be 0 or 1" );
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    if( OptixResult result = context->setDiskCacheEnabled( static_cast<bool>( enabled ), errDetails ) )
    {
        clog.sendError( errDetails );
        return result;
    }
    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDeviceContextSetCacheLocation( OptixDeviceContext contextAPI, const char* location )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_GET_CACHE_LOCATION );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( location );
    try
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result = context->setDiskCacheLocation( location, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDeviceContextSetCacheDatabaseSizes( OptixDeviceContext contextAPI, size_t lowWaterMark, size_t highWaterMark )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_SET_CACHE_DATABASE_SIZES );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = context->setDiskCacheMemoryLimits( lowWaterMark, highWaterMark, errDetails );

        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDeviceContextGetCacheEnabled( OptixDeviceContext contextAPI, int* enabled )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_GET_CACHE_ENABLED );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( enabled );
    *enabled = context->isDiskCacheActive() ? 1 : 0;
    return OPTIX_SUCCESS;
}

namespace optix_exp {
static OptixResult getCacheLocation( DeviceContext* context, char* location, size_t locationSize, ErrorDetails& errDetails )
{
    std::string path = context->getDiskCacheLocation();
    if( path.size() >= locationSize )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Given character buffer is too small to receive current cache location. Size "
                                      "needed: "
                                          + std::to_string( path.size() + 1 ) );
    }
    std::strncpy( location, path.c_str(), locationSize );
    location[path.size()] = 0;
    return OPTIX_SUCCESS;
}
}  // namespace optix_exp

extern "C" OptixResult optixDeviceContextGetCacheLocation( OptixDeviceContext contextAPI, char* location, size_t locationSize )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_GET_CACHE_LOCATION );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( location );
    OPTIX_CHECK_ZERO_ARGUMENT( locationSize );
    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = optix_exp::getCacheLocation( context, location, locationSize, errDetails );
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDeviceContextGetCacheDatabaseSizes( OptixDeviceContext contextAPI, size_t* lowWaterMark, size_t* highWaterMark )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DEVICE_CONTEXT_GET_CACHE_DATABASE_SIZES );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( lowWaterMark );
    OPTIX_CHECK_NULL_ARGUMENT( highWaterMark );
    context->getDiskCacheMemoryLimits( *lowWaterMark, *highWaterMark );
    return OPTIX_SUCCESS;
}
