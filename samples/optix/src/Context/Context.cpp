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

#define OPTIX_OPTIONAL_FEATURE_DEPRECATED_ATTRIBUTES
#define OPTIX_OPTIONAL_FEATURE_INTERNAL_ATTRIBUTES

#include <Context/Context.h>

#include <AS/ASManager.h>
#include <Context/BindingManager.h>
#include <Context/LLVMManager.h>
#include <Context/ObjectManager.h>
#include <Context/PlanManager.h>
#include <Context/ProfileManager.h>
#include <Context/ProgramManager.h>
#include <Context/RTCore.h>
#include <Context/SBTManager.h>
#include <Context/SharedProgramManager.h>
#include <Context/TableManager.h>
#include <Context/TelemetryManager.h>
#include <Context/UpdateManager.h>
#include <Context/ValidationManager.h>
#include <Context/WatchdogManager.h>
#include <Control/ErrorManager.h>
#include <Control/PrintManager.h>
#include <Device/APIDeviceAttributes.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Device/DeviceSet.h>
#include <Device/MaxDevices.h>
#include <Exceptions/ExceptionHelpers.h>
#include <Exceptions/TimeoutException.h>
#include <ExelwtionStrategy/ExelwtionStrategy.h>
#include <ExelwtionStrategy/NullES.h>
#include <ExelwtionStrategy/Plan.h>
#include <ExelwtionStrategy/RTX/RTXES.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MemoryManager.h>
#include <Objects/Geometry.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Objects/Material.h>
#include <Objects/StreamBuffer.h>
#include <Objects/Variable.h>
#include <ThreadPool/ThreadPool.h>
#include <Util/ApiTime.h>
#include <Util/CodeRange.h>
#include <Util/Metrics.h>
#include <Util/LWML.h>
#include <Util/NodegraphPrinter.h>
#include <Util/RangeVector.h>

#include <exp/context/DeviceContext.h>
#include <exp/context/DiskCache.h>
#include <exp/context/EncryptionManager.h>
#include <exp/context/ErrorHandling.h>

#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
#include <optix_7_types.h>
#undef __OPTIX_INCLUDE_INTERNAL_HEADERS__

#include <private/optix_declarations_private.h>
#include <private/optix_version_string.h>

#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>

#include <corelib/system/System.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <algorithm>

using namespace prodlib;
using namespace corelib;


// Default stack size as it was in ancient versions of Optix.  We now scale
// this as best we can to fit the needs of the current version.
// Doesn't have any effect for RTX exelwtion mode.
static const size_t STACK_SIZE_DEFAULT = 1024;

// Default for maximum call depth for callable programs.
// If the default value changes please update the documentation in optix_host.h.
static const unsigned int MAX_CALLABLE_PROGRAM_DEPTH_DEFAULT = 5;

// Default for maximum trace depth.
// If the default value changes please update the documentation in optix_host.h.
static const unsigned int MAX_TRACE_DEPTH_DEFAULT = 5;

// Decrease FirstExceptionCode when adding new exceptions. The last in the list
// is always RT_EXCEPTION_USER, since the space above that is reserved for
// user-exception codes.
static const int FIRST_EXCEPTION_CODE = RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS;
static const int LAST_EXCEPTION_CODE  = RT_EXCEPTION_USER;
static const int NUM_EXCEPTIONS       = 1 + LAST_EXCEPTION_CODE - FIRST_EXCEPTION_CODE;

static const char* CACHE_DB_FILE_NAME = "optixcache.db";

namespace optix {

namespace {

// clang-format off
PublicKnob<std::string> k_forceExelwtionStrategy(     RT_PUBLIC_DSTRING( "context.forceExelwtionStrategy" ),   "",    RT_PUBLIC_DSTRING( "Exelwtion strategy to use (default is empty, meaning rtx)" ) );
Knob<bool>              k_printKnobs(                 RT_DSTRING( "context.printKnobs" ),               false,        RT_DSTRING( "Print all knobs" ) );
Knob<bool>              k_skipLaunch(                 RT_DSTRING( "context.skipLaunch" ),               false,        RT_DSTRING( "Perform compilation and all data transfer, but skip lwca launch" ) );
Knob<float>             k_stackSizeMultiplier(        RT_DSTRING( "context.stackSizeMultiplier" ),      2.5,          RT_DSTRING( "Multiplier for rtContextSetStackSize" ) );
Knob<int>               k_stackSizeAdditional(        RT_DSTRING( "context.stackSizeAdditional" ),      128,          RT_DSTRING( "Additional stack to allocate in rtContextSetStackSize" ) );
Knob<int>               k_stackSizeAlignment(         RT_DSTRING( "context.stackSizeAlignment" ),       16,           RT_DSTRING( "Alignment for the stack size. Used both as allocation and size alignment." ) );
Knob<bool>              k_killAfterCompile(           RT_DSTRING( "context.killAfterCompile" ),         false,        RT_DSTRING( "Exit application after the plan has been compiled" ) );
Knob<bool>              k_killAtFirstTrace(           RT_DSTRING( "context.killAtFirstTrace" ),         false,        RT_DSTRING( "Exit application at first trace kernel launch" ) );
Knob<std::string>       k_saveNodegraph(              RT_DSTRING( "context.saveNodegraph" ),            "",           RT_DSTRING( "Save node graph to a dot file" ) );
Knob<bool>              k_saveNodegraphRefs(          RT_DSTRING( "context.saveNodegraph.refs" ),       true,         RT_DSTRING( "Save reference info in the node graph" ) );
Knob<RangeVector>       k_metricCaptureRanges(        RT_DSTRING( "metrics.launchRanges" ),             {},           RT_DSTRING( "Restricts metrics capture to specified ranges of launch call indices, separated by commas (e.g. 1-10,54,84-)." ));
Knob<int>               k_enableUsageReport(          RT_DSTRING( "log.enableUsageReport" ),            0,            RT_DSTRING( "Set usage report level and pipe to optix log ." ) );
Knob<int>               k_preferFastRecompiles(       RT_DSTRING( "context.preferFastRecompiles" ),     -1,           RT_DSTRING( "Override for RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES (default of -1 means use the default attribute value)" ) );
Knob<int>               k_forceInlineUserFunctions(   RT_DSTRING( "context.forceInlineUserFunctions" ), -1,           RT_DSTRING( "Force inlining of user functions. On" ) );
HiddenPublicKnob<std::string> k_traversalOverride(    RT_PUBLIC_DSTRING( "rtx.traversalOverride" ),     "",           RT_PUBLIC_DSTRING( "Override traversal to the given traversal" ) );
Knob<int>               k_hasMotionOverride(          RT_DSTRING( "rtx.hasMotionOverride" ),            -1,           RT_DSTRING( "Override motion blur" ) );
Knob<size_t>            k_lowWaterMark(               RT_DSTRING( "diskcache.lowWaterMark" ),           1u << 30,     RT_DSTRING( "The cache size will be reduced to the low-water mark if the high-water mark is exceeded. The value 0 disables garbage collection. The default value is 1GB." ));
Knob<size_t>            k_highWaterMark(              RT_DSTRING( "diskcache.highWaterMark" ),          1u << 31,     RT_DSTRING( "The cache size will be reduced to the low-water mark if the high-water mark is exceeded. The value 0 disables garbage collection. The default value is 2GB." ));
Knob<int>               k_streamPoolSize(             RT_DSTRING( "lwca.streamPoolSize" ),              4,            RT_DSTRING( "The number of lwca streams to create per device." ) );
Knob<bool>              k_ilwokeCallbacksPerMipLevel( RT_DSTRING( "context.ilwokeCallbacksPerMipLevel" ), false,      RT_DSTRING( "Force demand load textures to ilwoke callbacks per mip-level, not per tile" ) );
Knob<unsigned int>      k_numVirtualPages(            RT_DSTRING( "rtx.demandLoadNumPages" ),           1U << 26,     RT_DSTRING( "Number of virtual pages available for demand load resources" ) );
Knob<int>               k_abiCompatibility(           RT_DSTRING( "context.abiCompatibility" ), Context::ABI_18_USE_DEMAND_LOAD_CALLBACK_PER_TILE, RT_DSTRING( "ABI compatibility to use for demand-load paging" ) );
Knob<bool>              k_stampMemoryBlocks(          RT_DSTRING( "rtx.demandLoad.stampMemoryBlocks" ), false,        RT_DSTRING( "Stamp allocation id into memory blocks after ilwoking the fill callback" ) );
// clang-format on

}  // namespace

std::string Context::m_defaultExelwtionStrategy;

// Returns true if the launchIndex is found in a range and sets
// outRange and outRangeIndex if they are not nullptr.
static bool getLaunchRange( size_t launchIndex, Range& outRange, size_t& outRangeIndex )
{
    const RangeVector& ranges = k_metricCaptureRanges.get();
    if( ranges.empty() )
    {
        // implicit range of all launches
        outRange      = {0, SIZE_MAX};
        outRangeIndex = 0;
        return true;
    }

    for( size_t i = 0; i < ranges.size(); ++i )
        if( ranges[i].begin <= launchIndex && launchIndex < ranges[i].end )
        {
            outRange      = ranges[i];
            outRangeIndex = i;
            return true;
        }

    return false;
}

static void initKnobs( bool print = true )
{
    // validate knobs
    {
        std::string errstr;
        knobRegistry().finalizeKnobs( errstr );
        if( !errstr.empty() )
        {
            // initKnobs() is called from the Context constructor,
            // meaning that the error string in the exception won't make it
            // to the user (because we won't have an optix context).
            lerr << errstr << std::endl;
            throw IlwalidValue( RT_EXCEPTION_INFO, errstr );
        }
    }

    // print all knobs for a quick overview
    if( k_printKnobs.get() )
        knobRegistry().printKnobs( lprint_stream );

    // log all non-default knobs
    if( print )
        knobRegistry().printNonDefaultKnobs( lprint_stream );
}

static void logDeviceAttributes( const DeviceArray& devices, UsageReport& ur, TelemetryManager* telemetryManager, const int numVisibleDevices )
{
    MetricsScope  scope( "set_devices", Metrics::ARRAY );
    std::ostream& urs = ur.getPreambleStream();
    for( const optix::Device* device : devices )
    {
        MetricsScope scope;

        const APIDeviceAttributes& attributes = DeviceManager::getAPIDeviceAttributes( device->visibleDeviceListIndex() );

        Metrics::logInt( "lwda_device", attributes.lwdaDeviceOrdinal );
        urs << "LWCA device: " << attributes.lwdaDeviceOrdinal << std::endl;
        telemetryManager->setContextCreateData( "lwda_device", attributes.lwdaDeviceOrdinal );

        urs << "    " << attributes.pciBusId << std::endl;

        Metrics::logString( "gpu_name", attributes.name.c_str() );
        urs << "    " << attributes.name << std::endl;
        telemetryManager->setContextCreateData( "gpu_name", attributes.name.c_str() );

        Metrics::logInt( "sm_count", attributes.multiprocessorCount );
        urs << "    "
            << "SM count: " << attributes.multiprocessorCount << std::endl;
        telemetryManager->setContextCreateData( "sm_count", attributes.multiprocessorCount );

        int computeCapability = attributes.computeCapability.x * 10 + attributes.computeCapability.y;
        Metrics::logInt( "sm_arch", computeCapability );
        urs << "    "
            << "SM arch: " << computeCapability << std::endl;
        telemetryManager->setContextCreateData( "sm_arc", computeCapability );

        int clockRate = attributes.clockRate / 1000;
        Metrics::logInt( "sm_KHz", clockRate );
        urs << "    "
            << "SM clock: " << clockRate << " KHz" << std::endl;
        telemetryManager->setContextCreateData( "sm_clock", clockRate );

        RTsize totalMem = attributes.totalMemory / ( 1 << 20 );
        Metrics::logInt( "gpu_total_MB", totalMem );
        urs << "    "
            << "GPU memory: " << totalMem << " MB" << std::endl;
        telemetryManager->setContextCreateData( "gpu_memory", totalMem );

        Metrics::logInt( "tcc", attributes.tccDriver );
        urs << "    "
            << "TCC driver: " << attributes.tccDriver << std::endl;
        telemetryManager->setContextCreateData( "tcc_driver", attributes.tccDriver );

        std::string compatibleOrdinals;
        for( int index = 0; index < attributes.compatibleDevices[0]; ++index )
        {
            if( !compatibleOrdinals.empty() )
                compatibleOrdinals += ", ";
            compatibleOrdinals += std::to_string( attributes.compatibleDevices[index + 1] );
        }
        if( compatibleOrdinals.empty() )
            compatibleOrdinals = "<none>";
        Metrics::logString( "compatible_devices", compatibleOrdinals.c_str() );
        urs << "    "
            << "Compatible devices: " << compatibleOrdinals << std::endl;
        telemetryManager->setContextCreateData( "compatible_devices", compatibleOrdinals.c_str() );

        Metrics::logInt( "rtcore_version", attributes.rtcoreVersion );
        urs << "    "
            << "RT core version: " << attributes.rtcoreVersion << std::endl;
        telemetryManager->setContextCreateData( "rtcore_version", attributes.rtcoreVersion );
    }
}

static void logObjectUsage( const ObjectManager& objectMgr, UsageReport& usageReport )
{
    if( !usageReport.isActive( 2 ) )
        return;

    const ReusableIDMap<LexicalScope*>&   lscopes     = objectMgr.getLexicalScopes();
    const ReusableIDMap<Program*>&        programs    = objectMgr.getPrograms();
    const ReusableIDMap<Buffer*>&         buffers     = objectMgr.getBuffers();
    const ReusableIDMap<TextureSampler*>& texSamplers = objectMgr.getTextureSamplers();

    ureport2( usageReport, "SCENE STAT" ) << "Node graph object summary:" << std::endl;
    UsageReport::IndentFrame urif( usageReport );
    int                      numAccelerations     = 0;
    int                      numGroups            = 0;
    int                      numGeometryGroups    = 0;
    int                      numTransforms        = 0;
    int                      numSelectors         = 0;
    int                      numGeometryInstances = 0;
    int                      numGeometries        = 0;
    int                      numMaterials         = 0;
    int                      numPrims             = 0;

    for( const auto& scope : lscopes )
    {
        if( scope->getClass() == RT_OBJECT_ACCELERATION )
        {
            ++numAccelerations;
        }
        else if( scope->getClass() == RT_OBJECT_GROUP )
        {
            ++numGroups;
        }
        else if( scope->getClass() == RT_OBJECT_GEOMETRY_GROUP )
        {
            ++numGeometryGroups;
        }
        else if( scope->getClass() == RT_OBJECT_TRANSFORM )
        {
            ++numTransforms;
        }
        else if( scope->getClass() == RT_OBJECT_SELECTOR )
        {
            ++numSelectors;
        }
        else if( scope->getClass() == RT_OBJECT_GEOMETRY_INSTANCE )
        {
            ++numGeometryInstances;
        }
        else if( scope->getClass() == RT_OBJECT_GEOMETRY )
        {
            ++numGeometries;
            const Geometry* geometry = static_cast<const Geometry*>( scope );
            numPrims += geometry->getPrimitiveCount();
        }
        else if( scope->getClass() == RT_OBJECT_MATERIAL )
        {
            ++numMaterials;
        }
    }

    ureport2( usageReport, "SCENE STAT" ) << "RTprogram         : " << programs.size() << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTbuffer          : " << buffers.size() << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTtexturesampler  : " << texSamplers.size() << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTacceleration    : " << numAccelerations << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTgroup           : " << numGroups << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTgeometrygroup   : " << numGeometryGroups << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTtransform       : " << numTransforms << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTselector        : " << numSelectors << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTgeometryinstance: " << numGeometryInstances << std::endl;
    ureport2( usageReport, "SCENE STAT" ) << "RTgeometry        : " << numGeometries << std::endl;
    {
        UsageReport::IndentFrame urif( usageReport );
        ureport2( usageReport, "SCENE STAT" ) << "Total prim: " << numPrims << std::endl;
    }
    ureport2( usageReport, "SCENE STAT" ) << "RTmaterial        : " << numMaterials << std::endl;
}

static void outputToLog( int lvl, const char* tag, const char* msg, void* )
{
    llog( 5 ) << '[' << lvl << "][" << std::left << std::setw( 12 ) << tag << "] " << msg;
}

static void deviceContextLoggerCallback( unsigned int level, const char* tag, const char* message, void* cbdata )
{
    Context* self = static_cast<Context*>( cbdata );
    self->deviceLoggerCallback( level, tag, message );
}

Context::Context( AbiCompatibility abiCompatibility )
    : m_cancel_progressive_flag( false )
    , m_asyncLaunchesInProgress( false )
    , m_demandLoadStampMemoryBlocks( k_stampMemoryBlocks.get() )
    , m_abiCompatibility( k_abiCompatibility.isSet() ? static_cast<AbiCompatibility>( k_abiCompatibility.get() ) : abiCompatibility )
{
    m_startTime = getTimerTick();

    // Init singletons
    initKnobs();
    corelib::lwdaDriver();
    Metrics::init();

    // Get the export table for RTX exelwtion.  When we don't statically link against rtcore
    // we can selectively do this.
    m_rtcore.reset( new RTCore() );

    // Do this before initializing the devices, so DeviceManager can create the appropriate
    // set of default devices.
    if( !k_forceExelwtionStrategy.isDefault() )
        m_exelwtionStrategy = k_forceExelwtionStrategy.get();
    else
        m_exelwtionStrategy = m_defaultExelwtionStrategy;
    if( m_exelwtionStrategy.empty() )
        // Neither the knob nor the default is set, so choose rtx. This allows the
        // knob to allows override the value when the knob is set. That's why the default
        // of the knob is set to "". Otherwise it would not be possible to determine if
        // the knob is set or not.
        m_exelwtionStrategy = "rtx";

    if( !( m_exelwtionStrategy == "rtx" || m_exelwtionStrategy == "null" ) )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO,
                            std::string( "Exelwtion strategy \"" ) + m_exelwtionStrategy + "\" is not recognized" );
    }

    // TODO: As per lwbugs/3364337, it is an error to enable the megakernel ES.  When the time
    //       comes to remove megakernel entirely, we should probably take the null ES with it.

    if( m_exelwtionStrategy == "rtx" )
        m_dataModel = DataModel::RTX;
    else
        m_dataModel = DataModel::Canonical;

    if( !k_preferFastRecompiles.isDefault() )
    {
        m_preferFastRecompiles = static_cast<bool>( k_preferFastRecompiles.get() );
    }

    if( !k_forceInlineUserFunctions.isDefault() )
    {
        m_forceInlineUserFunctions = static_cast<bool>( k_forceInlineUserFunctions.get() );
    }

    // Initialize the DeviceContextLogger before the disk cache because that's
    // what the logger is needed for.
    m_deviceContextLogger.reset( new optix_exp::DeviceContextLogger() );
    optix_exp::ErrorDetails errDetails;
    if( m_deviceContextLogger->setCallbackNoCheck( deviceContextLoggerCallback, this, prodlib::log::level(), errDetails ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, errDetails.m_description );

    initializeDiskCache();

    // Create manager instances.
    m_deviceManager.reset( new DeviceManager( this, m_dataModel == DataModel::RTX ) );

    // Determine number of threads that can be launched conlwrrently.
    const int numThreads = getDeviceManager()->cpuDevice()->getNumThreads();

    m_memoryManager.reset( new MemoryManager( this ) );
    m_ASManager.reset( new ASManager( this ) );
    m_bindingManager.reset( new BindingManager( this ) );
    m_encryptionManager.reset( new optix_exp::EncryptionManager() );
    m_errorManager.reset( new ErrorManager() );
    m_llvmManager.reset( new LLVMManager( this ) );
    m_updateManager.reset( new UpdateManager( this ) );
    m_objectManager.reset( new ObjectManager( this ) );
    m_planManager.reset( new PlanManager( this ) );
    m_printManager.reset( new PrintManager( this ) );
    m_profileManager.reset( new ProfileManager( this ) );
    m_programManager.reset( new ProgramManager( this ) );
    m_sbtManager.reset( new SBTManager( this ) );
    m_sharedProgramManager.reset( new SharedProgramManager( this ) );
    m_tableManager.reset( new TableManager( this ) );
    m_telemetryManager.reset( new TelemetryManager( this ) );
    m_validationManager.reset( new ValidationManager( this ) );
    m_watchdogManager.reset( new WatchdogManager( this ) );

    bool ilwokeCallbacksPerTileNotPerMipLevel = m_abiCompatibility >= ABI_18_USE_DEMAND_LOAD_CALLBACK_PER_TILE;
    if( !k_ilwokeCallbacksPerMipLevel.isDefault() )
        ilwokeCallbacksPerTileNotPerMipLevel = !k_ilwokeCallbacksPerMipLevel.get();
    m_pagingManager = createPagingService( this, ilwokeCallbacksPerTileNotPerMipLevel, getDemandLoadNumVirtualPages() );

    m_threadPool.reset( new ThreadPool( static_cast<float>( numThreads ), 1.0f, numThreads ) );

    // This must be called after the ThreadPool has been set because PagingManager will query
    // the context for the thread pool when multithreaded callbacks are enabled.
    if( m_abiCompatibility == ABI_16_USE_MULTITHREADED_DEMAND_LOAD_CALLBACKS_BY_DEFAULT )
        m_pagingManager->setMultiThreadedCallbacksEnabled( true );

    lprint << OPTIX_BUILD_DESCRIPTION << "\n";
    lprint << "Display driver: " << LWML::driverVersion() << "\n";
    m_deviceManager->printAvailableDevices();

    m_usageReport.getPreambleStream() << "\n" << OPTIX_BUILD_DESCRIPTION << std::endl;
    m_usageReport.getPreambleStream() << "Display driver: " << LWML::driverVersion() << std::endl;
    m_usageReport.getPreambleStream() << "Devices available:" << std::endl;

    logDeviceAttributes( m_deviceManager->activeDevices(), m_usageReport, m_telemetryManager.get(),
                         getDeviceManager()->visibleDevices().size() );

    if( k_enableUsageReport.get() != 0 )
    {
        m_usageReport.setUserCallback( outputToLog, k_enableUsageReport.get(), nullptr );
    }

    m_telemetryManager->setContextCreateData( "optix_build", OPTIX_BUILD_DESCRIPTION );
    m_telemetryManager->setContextCreateData( "display_driver", LWML::driverVersion().c_str() );

    // Ensure that the embedded version of the build description makes it into
    // the DLL. This print will not occur in release build nor under
    // normal cirlwmstances with a developer build.
    llog( 0x7FFFFFFF ) << OPTIX_BUILD_DESCRIPTION_EMBEDDED;

    // Make sure we create the null program first, so that it can get CanonicalProgramID = 0.
    m_sharedProgramManager->getNullProgram();

    // Set up profile manager
    m_profileManager->postSetActiveDevices();

    m_globalScope = new GlobalScope( this );

    // Create exelwtion strategies
    m_nullES.reset( new NullES( this ) );
    m_rtxES.reset( new RTXES( this ) );

    // Initialize the stack size
    setStackSize( STACK_SIZE_DEFAULT );

    // Initialize the maximum callable program call depth
    setMaxCallableProgramDepth( MAX_CALLABLE_PROGRAM_DEPTH_DEFAULT );

    // Initialize the maximum trace depth
    setMaxTraceDepth( MAX_TRACE_DEPTH_DEFAULT );

    // Initialize exceptions (disable all except for stack overflow).
    setExceptionEnabled( RT_EXCEPTION_STACK_OVERFLOW, true );

    // Default for printing is off. Calling this function explicitly gives the
    // override knob a chance to kick in.
    setPrintEnabled( false );

    m_telemetryManager->uploadContextCreateData();

    // Init rtx motion blur
    m_rtxHasMotionBlur = k_hasMotionOverride.isDefault() ? false : k_hasMotionOverride.get();

    // Init rtx traversal mode
    m_rtxNeedsUniversalTraversal = rtxTraversablesNeedUniversalTraversal();

    // Initialize max number of async conlwrrent launches
    m_maxConlwrrentLaunches = k_streamPoolSize.get();
}

unsigned int Context::s_numVirtualPages = k_numVirtualPages.get();

Context::~Context()
{
    // Error manager is reset outside of tearDown because we may need it to report
    // tearDown's errors
    m_errorManager.reset();
}

static LWresult filterIlwalidOrDestroyedContext( LWresult status )
{
    return status == LWDA_ERROR_ILWALID_CONTEXT || status == LWDA_ERROR_CONTEXT_IS_DESTROYED ? LWDA_SUCCESS : status;
}

void Context::tearDown()
{
    Metrics::logInt( "num_compiles", m_numCompiles );

    finishFrame( false );

    m_telemetryManager->setContextTeardownData( "context_lifetime", getDeltaSeconds( m_startTime ) );
    m_telemetryManager->uploadContextTeardownData();

    m_shuttingDown = true;

    // First, perform some cleanup that the user could have done himself: destroy
    // outstanding API objects and internally generated helper objects.
    m_objectManager->destroyAllObjects();
    m_globalScope = nullptr;
    // m_deviceManager->setActiveDevices( {} );  // TODO: we want to do this eventually, but TableManager isn't ready for
    // zero active devices. Steve is going to revisit TM

    // destroy internal context resources before device teardown below
    m_planManager.reset();
    m_printManager.reset();

    m_ASManager.reset();
    m_profileManager.reset();

    m_bindingManager.reset();
    m_encryptionManager.reset();
    m_objectManager.reset();

    // Destroy the DiskCache to ensure that the usage report is updated
    m_diskCache->destroy( getDeviceContextLogger() );
    m_diskCache.reset();
    m_deviceContextLogger.reset();

    // Delete the program manager and update after managed objects have been torn down
    m_sbtManager.reset();
    m_sharedProgramManager.reset();
    m_tableManager.reset();
    m_programManager.reset();
    m_pagingManager->tearDown();
    m_pagingManager.reset();
    m_validationManager.reset();
    m_updateManager.reset();
    m_telemetryManager.reset();
    m_watchdogManager.reset();

    // Destroy exelwtion strategies
    m_nullES.reset();
    m_rtxES.reset();

    // rtxES contains a cache of llvm::Modules.
    // For this reason we need to delete the LLVMContext after rtxES.
    m_llvmManager.reset();

    // Delete memory manager almost last
    m_memoryManager->shutdown();
    m_memoryManager.reset();

    // Now we need to try and flush whatever profiling data is being collected before exit.
    // As of LWCA 5.0 it isn't an error to call lwProfilerStop without calling
    // lwProfilerInitialize, and according to the online programming guide, calling
    // lwProfilerStop should flush the buffer.
    // Note: Ignore illegal LWCA context errors, but throw on all others
    CALL_LWDA_DRIVER_THROW( filterIlwalidOrDestroyedContext( corelib::lwdaDriver().LwProfilerStop() ) );

    // Delete devices once managed objects have been deleted since
    // deleting a device means destroying the lwca context.  Resource
    // cleanup will fail if lwca has been shut down before.
    m_deviceManager.reset();
}

void Context::launchFromAPI( unsigned int entry, int dim, RTsize width, RTsize height, RTsize depth )
{
    TIMEVIZ_FUNC;
    timerTick    t0          = getTimerTick();
    const size_t launchIndex = m_launchCount - 1;

    // Issue interop functions
    m_memoryManager->enterFromAPI();

    // Perform the launch
    const DeviceSet launchDevices = m_deviceManager->activeDevices();
    launchFromCommandList( entry, dim, width, height, depth, launchDevices, lwca::Stream() );

    // Rework interop
    m_memoryManager->exitToAPI();

    Metrics::logFloat( "launch_msec", getDeltaMilliseconds( t0 ) );
    ureport1( m_usageReport, "TIMING" ) << "Total launch time: " << getDeltaMilliseconds( t0 ) << " ms" << std::endl;

    if( launchIndex == 0 )
    {
        Metrics::logFloat( "msec_to_first_frame", getDeltaMilliseconds( m_startTime ) );
        ureport1( m_usageReport, "TIMING" )
            << "Time from RTcontext creation to first frame: " << getDeltaMilliseconds( m_startTime ) << " ms" << std::endl;
    }
}

bool Context::rtxTraversablesHaveMotionBlur() const
{
    if( !k_hasMotionOverride.isDefault() )
        return k_hasMotionOverride.get();

    // Check if rtx requires motion blur for any of the traversal entry points
    for( auto node : getObjectManager()->getTraversables() )
    {
        // rtx needs motion blur and hierarchies
        if( node->hasMotionAabbs() )
            return true;
    }

    return false;
}

bool Context::rtxTraversablesNeedUniversalTraversal() const
{
    if( k_traversalOverride.get() == "Utrav" || k_traversalOverride.get() == "MTTU" )
        return true;
    // pre-Turing GPUs and those w/o TTU should use Utrav as well unless explicitly opted out
    if( !getDeviceManager()->activeDevicesSupportTTU() && getPreferWatertightTraversal() )
        return true;

    // Check if rtx requires universal traversal for any of the traversal entry points
    for( auto node : getObjectManager()->getTraversables() )
    {
        // rtx needs Universal traversal for motion blur and multi-level hierarchies
        if( node->rtxTraversableNeedUniversalTraversal() )
            return true;
    }

    return false;
}

void Context::rtxHasMotionBlurDidChange()
{
    // Inform plans that universal traversal has changed.
    getUpdateManager()->eventContextHasMotionBlurChanged( m_rtxHasMotionBlur );
}

void Context::rtxUniversalTraversalDidChange()
{
    // Inform plans that universal traversal has changed.
    getUpdateManager()->eventContextNeedsUniversalTraversalChanged( m_rtxNeedsUniversalTraversal );

    // Notify all traversables to set the RtxUniversalTraversal propery.
    // This relwrsively switches all attached nodes to RtxUniversalTraversal.
    for( auto node : getObjectManager()->getTraversables() )
    {
        node->receivePropertyDidChange_RtxUniversalTraversal();
    }
}

void Context::validateLaunchParams( unsigned int entry, int dim, RTsize width, RTsize height, RTsize depth )
{
    const size_t launchIndex = m_launchCount - 1;
    Range        range;
    size_t       rangeIndex = 0;
    const bool   rangeFound = getLaunchRange( launchIndex, range, rangeIndex );
    Metrics::setEnabled( rangeFound );
    MetricsScope launchScope;
    if( rangeFound )
    {
        Metrics::logInt( "range_index", rangeIndex );
        if( range.begin == launchIndex )
            Metrics::logFloat( stringf( "msec_to_launch_range<%zu>", rangeIndex ).c_str(), getDeltaMilliseconds( m_startTime ) );
    }

    ureport2( m_usageReport, "INFO" ) << "Launch index " << launchIndex << "." << std::endl;
    if( m_usageReport.isActive( 2 ) )
        m_usageReport.pushIndent();

    logObjectUsage( *m_objectManager, m_usageReport );
    if( launchIndex == 0 )
    {
        ureport1( m_usageReport, "TIMING" ) << "Time to first launch: " << getDeltaMilliseconds( m_startTime ) << " ms"
                                            << std::endl;
    }
    Metrics::logInt( "launch_index", launchIndex );
    if( launchIndex == 0 )
        Metrics::logFloat( "msec_to_first_launch", getDeltaMilliseconds( m_startTime ) );
    Metrics::logInt( "entry_index", entry );


    // Error checking.
    RT_ASSERT( dim >= 1 && dim <= 3 );
    if( dim == 3 )
    {
        if( ( width * depth ) >= UINT_MAX )
        {
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO,
                                         "Invalid launch dimensions.  For 3D launches, the product of 'width' and "
                                         "'depth' "
                                         "must be smaller than 429496729 (2^32)." );
        }
    }

    if( entry >= m_numEntryPoints )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid entry point index: ", entry );
}

void Context::launchFromCommandList( unsigned int               entry,
                                     int                        dim,
                                     RTsize                     width,
                                     RTsize                     height,
                                     RTsize                     depth,
                                     const DeviceSet&           launchDevices,
                                     const optix::lwca::Stream& syncStream )
{
    TIMEVIZ_FUNC;

    ++m_launchCount;

    validateLaunchParams( entry, dim, width, height, depth );

    // Enable the devices, creating any contexts necessary.
    getDeviceManager()->enableActiveDevices();

    // Validate if necessary.
    getValidationManager()->run();

    if( useRtxDataModel() )
    {
        // Check if rtx requires universal traversal for any of the traversal entry points
        const bool rtxNeedsUniversalTraversal = rtxTraversablesNeedUniversalTraversal();

        if( rtxNeedsUniversalTraversal != m_rtxNeedsUniversalTraversal )
        {
            if( rtxNeedsUniversalTraversal )
                llog( 20 ) << "Enabling universal traversal\n";
            else
                llog( 20 ) << "Disabling universal traversal\n";

            m_rtxNeedsUniversalTraversal = rtxNeedsUniversalTraversal;

            rtxUniversalTraversalDidChange();
        }

        // Check if rtx requires universal traversal for any of the traversal entry points
        const bool rtxHasMotionBlur = rtxTraversablesHaveMotionBlur();

        if( rtxHasMotionBlur != m_rtxHasMotionBlur )
        {
            if( rtxHasMotionBlur )
                llog( 20 ) << "Enabling motion blur\n";
            else
                llog( 20 ) << "Disabling motion blur\n";

            m_rtxHasMotionBlur = rtxHasMotionBlur;

            rtxHasMotionBlurDidChange();
        }
    }

    // Setup the AABB programs if they have not already been created.  This was
    // formerly done in the Context constructor, but has been deferred to allow
    // lazy initialization of the DiskCache.
    //
    // This is not strictly required, since the programs will be created during
    // the call to buildAccelerationStructures below.  But we are leaving the
    // explicit setup to keep the compile time and the AS build time separate.
    m_ASManager->setupInitialPrograms();

    // Update all required acceleration structures.  Note that this may
    // compile and launch additional kernels and modify the object
    // records before the actual kernel launch.
    {
        timerTick taccel = getTimerTick();
        m_ASManager->buildAccelerationStructures();
        ureport2( m_usageReport, "TIMING" ) << "Acceleration update time: " << getDeltaMilliseconds( taccel ) << " ms" << std::endl;
    }

    // Low-level launch sequence
    // If requested, skip launch for trace calls.
    if( !k_skipLaunch.get() )
    {
        launchPrepare( entry, dim, launchDevices, launchDevices.count() );

        if( syncStream.get() != nullptr )
        {
            launchFrameAsync( width, height, depth, launchDevices, cort::AabbRequest(), syncStream );
        }
        else
        {
            launchFrame( width, height, depth, launchDevices, cort::AabbRequest() );
            launchComplete();
        }
    }

    // Print host profiling result.
    m_profileManager->finalizeApiLaunch();

    if( m_usageReport.isActive( 2 ) )
        m_usageReport.popIndent();
}

void Context::launchPrepare( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices )
{
    TIMEVIZ_FUNC;

    RT_ASSERT( !m_lwrrentPlan );

    // Optionally save a dot file containing the nodegraph.
    saveNodeGraph( "launchKernels" );

    MetricsScope kernelLaunchScope;
    Metrics::logInt( "kernel_launch_index", m_kernelLaunchCounter );
    Metrics::logInt( "entry_index", entry );

    // Give PagingManager a chance to allocate per-device storage before we lock down MemoryManager.
    m_pagingManager->launchPrepare( devices );

    // Lock down memory manager and tables
    lockMemoryManager();

    try
    {
        preparePlan( entry, dimensionality, devices, numLaunchDevices );

        if( m_launchCount == 1 )
        {
            if( m_kernelLaunchCounter == 1 )
                Metrics::logFloat( "msec_to_first_kernel", getDeltaMilliseconds( m_startTime ) );

            if( entry != getAabbEntry() )
            {
                Metrics::logFloat( "msec_to_first_trace", getDeltaMilliseconds( m_startTime ) );
                Metrics::logFloat( "api_msec_to_first_trace", getApiTime() * 1e3 );
                Metrics::logFloat( "compile_msec_to_first_trace", m_compileMsecToFirstTrace );
                if( k_killAtFirstTrace.get() )
                {
                    llog( 20 ) << "EXITING due to knob: " << k_killAtFirstTrace.getName() << "\n";
                    // in contrast to k_killAfterCompile we return 0 - otherwise the Jenkins tests fail
                    exit( 0 );
                }
            }
        }

        // Initialize memory for device-side profiler.  This needs to happen after compile.
        m_profileManager->beginKernelLaunch();
    }
    catch( const Exception& )
    {
        unlockMemoryManager();
        throw;
    }
}

void Context::launchFrame( RTsize width, RTsize height, RTsize depth, const DeviceSet& devices, const cort::AabbRequest& aabbRequest )
{
    RT_ASSERT( m_lwrrentPlan );
    if( aabbRequest.aabbOutputPointer )
        RT_ASSERT_MSG( devices.count() == 1, "Can only have one device in the launch device set when computing AABBs" );
    TIMEVIZ_FUNC;

    ++m_kernelLaunchCounter;

    MetricsScope kernelLaunchScope;
    Metrics::logInt( "kernel_launch_index", m_kernelLaunchCounter );
    Metrics::logInt( "entry_index", m_lwrrentEntry );

    m_lwrrentPlan->setKernelLaunchCounter( m_kernelLaunchCounter );

    // Acquire launch resources and wait handle.
    std::shared_ptr<LaunchResources> launchResource =
        this->determineExelwtionStrategy( m_lwrrentEntry )
            ->acquireLaunchResources( devices, m_lwrrentPlan->getTask(), optix::lwca::Stream(), width, height, depth );
    std::shared_ptr<WaitHandle> waiter = m_lwrrentPlan->getTask()->acquireWaitHandle( launchResource );

    // Execute the task and block on the wait handle.
    m_lwrrentPlan->getTask()->launch( waiter, m_lwrrentEntry, m_lwrrentDimensionality, width, height, depth,
                                      m_subframe_index, aabbRequest );
    waiter->block();
    waiter->checkFrameStatus();
    m_lastKernelTime = waiter->getElapsedMilliseconds();

    RTsize elems = width * height * depth;
    Metrics::logString( "kernel_type", m_lwrrentEntry == getAabbEntry() ? "aabb" : "trace" );
    Metrics::logInt( "kernel_elems", elems );
    Metrics::logFloat( "kernel_msec", m_lastKernelTime );
}

std::shared_ptr<WaitHandle> Context::launchFrameAsync( RTsize                     width,
                                                       RTsize                     height,
                                                       RTsize                     depth,
                                                       const DeviceSet&           devices,
                                                       const cort::AabbRequest&   aabbRequest,
                                                       const optix::lwca::Stream& syncStream )
{
    RT_ASSERT( m_lwrrentPlan );
    if( aabbRequest.aabbOutputPointer )
        RT_ASSERT_MSG( devices.count() == 1, "Can only have one device in the launch device set when computing AABBs" );
    TIMEVIZ_FUNC;

    ++m_kernelLaunchCounter;

    MetricsScope kernelLaunchScope;
    Metrics::logInt( "kernel_launch_index", m_kernelLaunchCounter );
    Metrics::logInt( "entry_index", m_lwrrentEntry );

    m_lwrrentPlan->setKernelLaunchCounter( m_kernelLaunchCounter );

    // Acquire launch resources
    std::shared_ptr<LaunchResources> launchResource =
        this->determineExelwtionStrategy( m_lwrrentEntry )
            ->acquireLaunchResources( devices, m_lwrrentPlan->getTask(), syncStream, width, height, depth );
    std::shared_ptr<WaitHandle> waiter = m_lwrrentPlan->getTask()->acquireWaitHandle( launchResource );

    // Store the wait handle in the list of pending wait handles, but not for AABB launches.
    // AABB launches are only semi-async, they will launch a bunch of launches in parallel but
    // will wait for them all to complete before returning.
    if( !aabbRequest.aabbOutputPointer )
    {
        std::lock_guard<std::mutex> lock( m_waitHandles_mutex );
        m_waitHandles.addItem( waiter );
    }

    // Note that storing the wait handle must be done before doing the launch to avoid the
    // possibility that the LWCA callback that cleans up the waiter is done before the waiter
    // is put in the list.
    m_lwrrentPlan->getTask()->launch( waiter, m_lwrrentEntry, m_lwrrentDimensionality, width, height, depth,
                                      m_subframe_index, aabbRequest );

    return waiter;
}


void Context::launchComplete()
{
    TIMEVIZ_FUNC;

    RT_ASSERT( m_lwrrentPlan );

    unlockMemoryManager();

    m_pagingManager->launchComplete();

    // Stop profiling.  Do this after the memory manager is unlocked, so we see the dirty bits on the device memory
    m_profileManager->finalizeKernelLaunch( m_lwrrentPlan->getTask()->getProfileMapping().get() );

    // Wait to reset this after we finalize the profiler
    m_lwrrentPlan = nullptr;
}

void Context::lockMemoryManager()
{
    TIMEVIZ_FUNC;

    // Finalize allocations required by the table manager
    m_tableManager->allocateTables();

    // Map all allocations to the device(s) before initiating the
    // planning process since allocations may ilwalidate plan
    // assumptions.
    m_memoryManager->syncAllMemoryBeforeLaunch();

    // Begin synchronization of tables now that all pointers have been allocated.
    m_tableManager->syncTablesForLaunch();
}

void Context::unlockMemoryManager()
{
    TIMEVIZ_FUNC;

    // Release tables and map them back to the host.  This needs to happen before
    // MemoryManager::releaseMemoryAfterLaunch, since that function can change the MAccess
    // objects of buffers.
    m_tableManager->launchCompleted();

    // Complete memory manager operations after launch, including
    // prefetching of some allocations.
    m_memoryManager->releaseMemoryAfterLaunch();

    if( m_sbtManager )
        m_sbtManager->launchCompleted();
}

bool Context::isRunning() const
{
    throw prodlib::AssertionFailure( RT_EXCEPTION_INFO,
                                     "Unimplemented: this function is not implemented yet and it serves as a "
                                     "placeholder for a future implementation" );
}

void Context::finishFrame( bool syncBuffers )
{
    // no-op for now, no async launches yet
}

void Context::launchProgressive( unsigned int max_subframes, unsigned int entryPointIndex, int dim, RTsize width, RTsize height, RTsize depth )
{
    // Handle errors caused by previous async launches here. In the future, we
    // may want to allow any API function to return errors generated
    // asynchronously, but for now it's just progressive launches.
    if( m_async_exception )
    {
        rethrowException( m_async_exception );
    }

    if( m_progressive_launch_in_progress )
    {
        return;
    }

    // we log it only when it is actually exelwting a launch.
    TIMEVIZ_SCOPE( "progressive launch" );

    m_progressive_launch_in_progress = true;
    m_cancel_progressive_flag        = false;

    for( const auto& stream : getObjectManager()->getStreamBuffers() )
    {
        Buffer* source = stream->getSource();
        if( source )
        {
            if( source->getWidth() != stream->getWidth() || source->getHeight() != stream->getHeight()
                || source->getDimensionality() != 2 || stream->getDimensionality() != 2 )
            {
                throw IlwalidValue( RT_EXCEPTION_INFO,
                                    "Size mismatch between stream buffer and source, or invalid dimensionality" );
            }
            if( source->getFormat() != RT_FORMAT_FLOAT3 && source->getFormat() != RT_FORMAT_FLOAT4 )
            {
                throw IlwalidValue( RT_EXCEPTION_INFO,
                                    "Invalid buffer format for stream source buffer (must be RT_FORMAT_FLOAT3 or "
                                    "RT_FORMAT_FLOAT4, RT_FORMAT_FLOAT4 is recommended for best performance)" );
            }
            if( stream->getFormat() != RT_FORMAT_UNSIGNED_BYTE4 )
            {
                throw IlwalidValue( RT_EXCEPTION_INFO,
                                    "Invalid stream buffer format (must be RT_FORMAT_UNSIGNED_BYTE4)" );
            }
        }
    }

    m_progressive_worker_handle = std::async( std::launch::async, &Context::progressiveLaunchWorker, this,
                                              max_subframes, entryPointIndex, dim, width, height, depth );
}

void Context::stopProgressiveLaunch( bool hard_stop )
{
    if( !m_progressive_launch_in_progress )
        return;

    m_cancel_progressive_flag = true;
    m_progressive_worker_handle.wait();
    m_progressive_launch_in_progress = false;

    // The API guarantees that after a hard stop, no more frames arrive at the client.
    // So we make sure stream buffers don't report a ready frame (could happen if there was
    // an update that's never been read before the client called stop)
    if( hard_stop )
    {
        for( const auto& stream : getObjectManager()->getStreamBuffers() )
            stream->markNotReady();
    }
}

void optix::Context::preparePlan( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices )
{
    // Determine if a recent plan can still be used.  Otherwise, create a new plan.
    Plan* plan = m_planManager->findValidPlan( entry, dimensionality, devices, numLaunchDevices );
    if( plan )
    {
        llog( 30 ) << "Using existing plan: " << plan->summaryString() << '\n';
    }
    else
    {
        const ExelwtionStrategy* es      = determineExelwtionStrategy( entry );
        std::unique_ptr<Plan>    newPlan = es->createPlan( entry, dimensionality, devices, numLaunchDevices );
        RT_ASSERT_MSG( !newPlan->getTask(), "Plan should not compile task in planning stage" );

        // Find an already-compiled plan from the cache if possible. If it
        // is new, insert this plan into the cache.
        plan = m_planManager->findOrCachePlan( std::move( newPlan ) );
    }

    // Compile a task (set of kernels) if it has not been compiled.
    if( !plan->hasBeenCompiled() )
    {
        llog( 20 ) << "Compiling plan:    " << plan->summaryString() << '\n';
        ureport1( m_usageReport, "INFO" ) << "Compilation triggered " << std::endl;
        timerTick t0 = getTimerTick();
        plan->compile();
        m_numCompiles++;
        double duration = getDeltaMilliseconds( t0 );
        Metrics::logFloat( "compile_msec", duration );
        if( m_launchCount == 1 )
            m_compileMsecToFirstTrace += duration;
        UsageReport::IndentFrame urif( m_usageReport );
        ureport1( m_usageReport, "TIMING" ) << "Compilation time: " << getDeltaMilliseconds( t0 ) << " ms" << std::endl;
    }

    if( k_killAfterCompile.get() )
    {
        lwarn << "EXITING due to knob: " << k_killAfterCompile.getName() << "\n";
        exit( 1 );
    }

    RT_ASSERT_MSG( plan->hasBeenCompiled(), "Plan did not compile task" );

    // Make sure the plan's frame task is active
    m_planManager->activate( plan );
    m_lwrrentPlan           = plan;
    m_lwrrentEntry          = entry;
    m_lwrrentDimensionality = dimensionality;
}

void optix::Context::launchAsync( unsigned int               entry,
                                  int                        dim,
                                  RTsize                     width,
                                  RTsize                     height,
                                  RTsize                     depth,
                                  const DeviceSet&           launchDevices,
                                  const optix::lwca::Stream& syncStream )
{
    TIMEVIZ_FUNC;

    // Do one-time initialization for the first async launch, otherwise just do the launch.
    // Assumption here is that nothing can change between conlwrrent async launches, anything
    // that would require a full launch will have triggered a hard sync (call to finishAsyncLaunches)
    // first.
    if( !m_asyncLaunchesInProgress )
    {
        m_asyncLaunchesInProgress = true;

        // Pre-launch interop.
        m_memoryManager->enterFromAPI();

        launchFromCommandList( entry, dim, width, height, depth, launchDevices, syncStream );
    }
    else
    {
        // If the entry has changed between calls (allowed without forcing a sync point), then
        // make sure that the current plan is the correct one.
        if( entry != m_lwrrentEntry )
            preparePlan( entry, dim, launchDevices, launchDevices.count() );

        launchFrameAsync( width, height, depth, launchDevices, cort::AabbRequest(), syncStream );
    }
}

optix_exp::DeviceContextLogger& Context::getDeviceContextLogger() const
{
    return *m_deviceContextLogger.get();
}

void Context::deviceLoggerCallback( unsigned int level, const char* tag, const char* message )
{
    if( strncmp( tag, "DISK CACHE", strlen( tag ) ) == 0 )
    {
        ureport2( m_usageReport, "DISK CACHE" ) << message << "\n";
    }
    else
    {
        llog( level ) << tag << ":" << message << "\n";
    }
}

void LWDA_CB Context::launchFinished( std::shared_ptr<WaitHandle>& waitHandle )
{
    std::lock_guard<std::mutex> lock( m_waitHandles_mutex );

    // If the waitHandle isn't in the list then nothing to do. Can happen if finishAsyncLaunches
    // is called before the lwca callback has been made.
    if( !m_waitHandles.itemIsInList( waitHandle ) )
        return;

    // Clear the handle resources. Must be done while holding the wait handle lock to make sure
    // that the API thread is not accessing the wait handle while doing this.
    waitHandle->releaseResources();

    // Remove it from the list of pending wait handles.
    m_waitHandles.removeItem( waitHandle );
}

void Context::finishAsyncLaunches()
{
    TIMEVIZ_FUNC;

    // Nothing to do if no asynchronous launch is pending.
    if( !m_asyncLaunchesInProgress )
        return;

    std::vector<std::shared_ptr<WaitHandle>> waitHandles;
    {
        std::lock_guard<std::mutex> lock( m_waitHandles_mutex );
        waitHandles = m_waitHandles.clearAndMove();
    }

    // Wait for all current async launches to complete. Do this without holding the lock since
    // the stream callback might do a call to clean up a handle and this pauses the stream which
    // could cause a deadlock.
    for( auto& waiter : waitHandles )
    {
        waiter->block();
        waiter->checkFrameStatus();
    }

    launchComplete();

    // Rework interop
    m_memoryManager->exitToAPI();

    m_asyncLaunchesInProgress = false;
}

void Context::setDevices( size_t numDevices, const int* ordinals )
{
    // Map device ordinals to devices
    DeviceArray devices( numDevices );
    const int   numActive = static_cast<int>( getDeviceManager()->visibleDevices().size() );
    for( size_t i = 0; i < numDevices; ++i )
    {
        const int ordinal = ordinals[i];
        if( ordinal < 0 || ordinal >= numActive )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid device number", ordinal );
        devices[i] = getDeviceManager()->visibleDevices()[ordinal];
    }

    // Device manager will check for duplicates and ilwoke all other necessary steps.
    getDeviceManager()->setActiveDevices( devices );

    m_usageReport.getPreambleStream() << "Devices selected:\n";
    logDeviceAttributes( getDeviceManager()->activeDevices(), m_usageReport, m_telemetryManager.get(), numActive );
}

void Context::getDevices( std::vector<int>& devices ) const
{
    // Loop over the active devices, but return the "visible" device number
    const DeviceArray& active     = getDeviceManager()->activeDevices();
    size_t             numDevices = active.size();
    devices.reserve( numDevices );
    for( size_t i = 0; i < numDevices; ++i )
        devices.push_back( active[i]->visibleDeviceListIndex() );
}

unsigned int Context::getMaxConlwrrentLaunches()
{
    return m_maxConlwrrentLaunches;
}

void Context::validateAttributeSize( RTsize actualSize, size_t expectedSize, const char* attribute ) const
{
    if( static_cast<size_t>( actualSize ) != expectedSize )
        throw IlwalidValue( RT_EXCEPTION_INFO, attributeCopyErrorString( expectedSize, actualSize, attribute ) );
}

#define VALIDATE_ATTRIBUTE_SIZE( expectedSize_, attr_ ) validateAttributeSize( size, expectedSize_, #attr_ )

void Context::setAttribute( RTcontextattribute attrib, RTsize size, const void* p )
{
    switch( attrib )
    {
        case RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT:
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Cannot set read only context attribute: RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT." );
            break;

        case RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS );
            int numThreads = *static_cast<const int*>( p );
            if( numThreads < 1 || numThreads > 1024 )
            {
                throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid value for RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS" );
            }
            getDeviceManager()->cpuDevice()->setNumThreads( numThreads );
            getThreadPool()->setNumWorkerThreads( numThreads );
            getThreadPool()->setCpuLoadLimit( static_cast<float>( numThreads ) );
            break;
        }

        case RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY:
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Cannot set read only context attribute: RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY." );
            break;

        case RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE:
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Cannot set read only context attribute: RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE." );
            break;

        case RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF );
            // NOP on Goldenrod, since we don't have paging
            break;

        case RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED );
            getDiskCache()->setIsActive( *static_cast<const int*>( p ) );
            break;

        case RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( char* ), RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION );
            std::string path( static_cast<const char*>( p ) );
            setDiskCacheLocation( path );
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( RTsize ) * 2, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS );
            const RTsize lowWaterMark  = static_cast<const RTsize*>( p )[0];
            const RTsize highWaterMark = static_cast<const RTsize*>( p )[1];

            // This will throw if setting the limits fails
            setDiskCacheMemoryLimits( lowWaterMark, highWaterMark );

            break;
        }
        case RT_CONTEXT_ATTRIBUTE_MAX_CONLWRRENT_LAUNCHES:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_MAX_CONLWRRENT_LAUNCHES );
            // Knob has precedence. Allow setting the value only if knob is on default.
            if( k_streamPoolSize.isDefault() )
            {
                const int value = *static_cast<const int*>( p );
                if( value <= 0 )
                {
                    throw IlwalidValue( RT_EXCEPTION_INFO,
                                        "RT_CONTEXT_ATTRIBUTE_MAX_CONLWRRENT_LAUNCHES must be bigger than zero." );
                }
                m_maxConlwrrentLaunches = value;
                getDeviceManager()->setStreamPoolSize( value );
                getPlanManager()->setMaxCachedPlanCount( 2 * value );
            }
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES:
        {
            // Ignore user value if already set by knob override
            if( !k_preferFastRecompiles.isDefault() )
                return;

            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES );
            const bool value = *static_cast<const int*>( p ) != 0;
            if( m_preferFastRecompiles != value )
            {
                getUpdateManager()->eventContextSetPreferFastRecompiles( m_preferFastRecompiles, value );
                m_preferFastRecompiles = value;
            }
        }
        case RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS:
        {
            // Ignore user value if already set by knob override
            if( !k_forceInlineUserFunctions.isDefault() )
                return;

            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS );
            m_forceInlineUserFunctions = *static_cast<const int*>( p ) != 0;
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_OPTIX_SALT:
        {
            VALIDATE_ATTRIBUTE_SIZE( 32, RT_CONTEXT_ATTRIBUTE_OPTIX_SALT );
            optix_exp::ErrorDetails errDetails;
            if( OptixResult result = m_encryptionManager->setOptixSalt( p, size, errDetails ) )
                throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, errDetails.m_description );
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_VENDOR_SALT:
        {
            VALIDATE_ATTRIBUTE_SIZE( 32, RT_CONTEXT_ATTRIBUTE_VENDOR_SALT );
            optix_exp::ErrorDetails errDetails;
            if( OptixResult result = m_encryptionManager->setVendorSalt( p, size, errDetails ) )
                throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, errDetails.m_description );
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_PUBLIC_VENDOR_KEY:
        {
            if( size == 0 )
                throw IlwalidValue( RT_EXCEPTION_INFO,
                                    "Zero-length size for attribute RT_CONTEXT_ATTRIBUTE_PUBLIC_VENDOR_KEY" );
            optix_exp::ErrorDetails errDetails;
            if( OptixResult result = m_encryptionManager->setPublicVendorKey( p, size, errDetails ) )
                throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, errDetails.m_description );
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_PREFER_WATERTIGHT_TRAVERSAL:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_PREFER_WATERTIGHT_TRAVERSAL );
            m_preferWatertightTraversal = *static_cast<const int*>( p ) != 0;
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_NUM_THREADS:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_NUM_THREADS );
            const int numThreads = *static_cast<const int*>( p );
            m_pagingManager->setMultiThreadedCallbacksEnabled( numThreads != 0 );
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_MULTITHREADED_CALLBACKS_ENABLED:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_ENABLE_DEMAND_LOAD_MULTITHREADED_CALLBACKS );
            const bool enabled = *static_cast<const int*>( p ) != 0;
            m_pagingManager->setMultiThreadedCallbacksEnabled( enabled );
            break;
        }
        case RT_CONTEXT_ATTRIBUTE_USE_HARDWARE_SPARSE_TEXTURES:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_USE_HARDWARE_SPARSE_TEXTURES );
            const bool enabled = *static_cast<const int*>( p ) != 0;
            m_pagingManager->setLwdaSparseTexturesEnabled( enabled );
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_SUBFRAME_INDEX:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( unsigned int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_SUBFRAME_INDEX );
            m_subframe_index = *static_cast<const unsigned int*>( p );
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_PROGRAM_ID_HINT:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_PROGRAM_ID_HINT );
            const int id = *static_cast<const int*>( p );
            llog( 30 ) << "Received program ID hint: " << id << std::endl;
            m_objectManager->setNextProgramIdHint( id );
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_BUFFER_ID_HINT:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE );
            const int id = *static_cast<const int*>( p );
            llog( 30 ) << "Received buffer ID hint: " << id << std::endl;
            m_objectManager->setNextBufferIdHint( id );
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_TEXTURE_SAMPLER_ID_HINT:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_TEXTURE_SAMPLER_ID_HINT );
            const int id = *static_cast<const int*>( p );
            llog( 30 ) << "Received texture sampler ID hint: " << id << std::endl;
            m_objectManager->setNextTextureSamplerIdHint( id );
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_STACK_SIZE:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( size_t ) * 3, RT_CONTEXT_INTERNAL_ATTRIBUTE_STACK_SIZE );
            const size_t continuationStackSize                = static_cast<const size_t*>( p )[0];
            const size_t directCallableStackSizeFromTraversal = static_cast<const size_t*>( p )[1];
            const size_t directCallableStackSizeFromState     = static_cast<const size_t*>( p )[2];

            if( continuationStackSize != m_continuationStackSize || directCallableStackSizeFromTraversal != m_directCallableStackSizeFromTraversal
                || directCallableStackSizeFromState != m_directCallableStackSizeFromState )
            {
                getUpdateManager()->eventContextSetAttributeStackSize(
                    m_continuationStackSize, m_directCallableStackSizeFromTraversal, m_directCallableStackSizeFromState,
                    continuationStackSize, directCallableStackSizeFromTraversal, directCallableStackSizeFromState );
                setAttributeStackSize( continuationStackSize, directCallableStackSizeFromTraversal, directCallableStackSizeFromState );
            }
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_FORCE_SYNCHRONOUS:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_FORCE_SYNCHRONOUS );
            const bool forced = *static_cast<const int*>( p ) != 0;
            if( forced != m_demandLoadForceSynchronous )
            {
                m_demandLoadForceSynchronous = forced;
                m_pagingManager->forceSynchronousRequestsChanged();
            }
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_USE_ASYNC_COPIES:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_USE_ASYNC_COPIES );
            const bool useAsyncCopies = *static_cast<const int*>( p ) != 0;
            if( useAsyncCopies != m_demandLoadUseAsyncCopies )
            {
                m_demandLoadUseAsyncCopies = useAsyncCopies;
                m_pagingManager->useAsynchronousCopiesChanged();
            }
            break;
        }
        case RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_STAMP_MEMORY_BLOCKS:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_USE_ASYNC_COPIES );
            const bool stampMemoryBlocks = *static_cast<const int*>( p ) != 0;
            if( stampMemoryBlocks != m_demandLoadStampMemoryBlocks )
            {
                m_demandLoadStampMemoryBlocks = stampMemoryBlocks;
                m_pagingManager->stampMemoryBlocksChanged();
            }
            break;
        }

        default:
            // devices are addressed by RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY+ordinal
            if( attrib >= RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY
                && attrib < RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY + OPTIX_MAX_DEVICES )
            {
                throw IlwalidValue( RT_EXCEPTION_INFO,
                                    "Cannot set read only context attribute: "
                                    "RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY." );
            }
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid context attribute" );
    }
}

void Context::getAttribute( RTcontextattribute attrib, RTsize size, void* p ) const
{
    switch( attrib )
    {
        case RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT );
            *static_cast<int*>( p ) = 0x7fffffff;
            break;

        case RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS );
            *static_cast<int*>( p ) = getDeviceManager()->cpuDevice()->getNumThreads();
            break;

        case RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( RTsize ), RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY );
            *static_cast<RTsize*>( p ) = getMemoryManager()->getUsedHostMemory();
            break;

        case RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE );
            *static_cast<int*>( p ) = 0;
            break;

        case RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF );
            *static_cast<int*>( p ) = 0;
            break;

        case RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED );
            *static_cast<int*>( p ) = getDiskCache()->isActive();
            break;

        case RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( char** ), RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION );
            *static_cast<const char**>( p ) = getPublicString( getDiskCacheLocation() );
            break;

        case RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( RTsize ) * 2, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS );
            size_t lowWaterMark, highWaterMark;
            getDiskCacheMemoryLimits( lowWaterMark, highWaterMark );
            static_cast<RTsize*>( p )[0] = lowWaterMark;
            static_cast<RTsize*>( p )[1] = highWaterMark;
            break;
        }

        case RT_CONTEXT_ATTRIBUTE_MAX_CONLWRRENT_LAUNCHES:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_MAX_CONLWRRENT_LAUNCHES );
            *static_cast<int*>( p ) = m_maxConlwrrentLaunches;
            break;
        }

        case RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES );
            *static_cast<int*>( p ) = static_cast<int>( m_preferFastRecompiles );
            break;

        case RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS );
            *static_cast<int*>( p ) = m_forceInlineUserFunctions;
            break;

        case RT_CONTEXT_ATTRIBUTE_OPTIX_SALT:
        {
            VALIDATE_ATTRIBUTE_SIZE( 32, RT_CONTEXT_ATTRIBUTE_OPTIX_SALT );
            optix_exp::ErrorDetails errDetails;
            if( const OptixResult result = m_encryptionManager->getOptixSalt( p, size, errDetails ) )
                throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, errDetails.m_description );
            break;
        }

        case RT_CONTEXT_ATTRIBUTE_PREFER_WATERTIGHT_TRAVERSAL:
        {
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_PREFER_WATERTIGHT_TRAVERSAL );
            *static_cast<int*>( p ) = static_cast<int>( m_preferWatertightTraversal );
            break;
        }

        case RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_NUM_THREADS:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_NUM_THREADS );
            *static_cast<int*>( p ) =
                m_pagingManager->getMultiThreadedCallbacksEnabled() ? getDeviceManager()->cpuDevice()->getNumThreads() : 0;
            break;

        case RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_MULTITHREADED_CALLBACKS_ENABLED:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_ATTRIBUTE_ENABLE_DEMAND_LOAD_MULTITHREADED_CALLBACKS );
            *static_cast<int*>( p ) = m_pagingManager->getMultiThreadedCallbacksEnabled() ? 1 : 0;
            break;

        case RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_FORCE_SYNCHRONOUS:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_FORCE_SYNCHRONOUS );
            *static_cast<int*>( p ) = m_demandLoadForceSynchronous ? 1 : 0;
            break;

        case RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_USE_ASYNC_COPIES:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_USE_ASYNC_COPIES );
            *static_cast<int*>( p ) = m_demandLoadUseAsyncCopies ? 1 : 0;
            break;

        case RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_STAMP_MEMORY_BLOCKS:
            VALIDATE_ATTRIBUTE_SIZE( sizeof( int ), RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_USE_ASYNC_COPIES );
            *static_cast<int*>( p ) = m_demandLoadStampMemoryBlocks ? 1 : 0;
            break;

        default:
            // devices are addressed by RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY+ordinal
            if( attrib >= RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY
                && attrib < RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY + OPTIX_MAX_DEVICES )
            {
                const int          ordinal = attrib - RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY;
                const DeviceArray& devices = getDeviceManager()->visibleDevices();
                if( ordinal > static_cast<int>( devices.size() ) || ordinal < 0 )
                    throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid device ordinal" );
                else
                {
                    VALIDATE_ATTRIBUTE_SIZE( sizeof( RTsize ), RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY );
                    *static_cast<RTsize*>( p ) = devices[ordinal]->getAvailableMemory();
                }
            }
            else
            {
                throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid context attribute" );
            }
            break;
    }
}

#undef VALIDATE_ATTRIBUTE_SIZE

void Context::setEntryPointCount( unsigned int numEntryPoints )
{
    if( numEntryPoints == m_numEntryPoints )
        return;

    getUpdateManager()->eventContextSetEntryPointCount( m_numEntryPoints, numEntryPoints );
    m_numEntryPoints = numEntryPoints;
    getGlobalScope()->setEntryPointCount( numEntryPoints );
}

unsigned int Context::getEntryPointCount() const
{
    return m_numEntryPoints;
}

void Context::setRayTypeCount( unsigned int numRayTypes )
{
    if( numRayTypes == m_numRayTypes )
        return;

    getUpdateManager()->eventContextSetRayTypeCount( m_numRayTypes, numRayTypes );
    m_numRayTypes = numRayTypes;

    getGlobalScope()->setRayTypeCount( numRayTypes );

    for( const auto& scope : getObjectManager()->getLexicalScopes() )
    {
        if( AbstractGroup* g = managedObjectCast<AbstractGroup>( scope ) )
            g->rayTypeCountDidChange();
        else if( Material* matl = managedObjectCast<Material>( scope ) )
            matl->setRayTypeCount( numRayTypes );
    }
}

unsigned int Context::getRayTypeCount() const
{
    return m_numRayTypes;
}

void Context::setStackSize( size_t stackSize )
{
    // megakernel only, no effect on rtx exelwtion mode
    const size_t alignedStackSize =
        align( size_t( stackSize * k_stackSizeMultiplier.get() + k_stackSizeAdditional.get() ), k_stackSizeAlignment.get() );

    if( alignedStackSize != m_stackSize )
    {
        getUpdateManager()->eventContextSetStackSize( m_stackSize, alignedStackSize );
        llog( 15 ) << "Context::setStackSize changing stack size from " << m_stackSize << " to " << alignedStackSize << "\n";
        m_stackSize = alignedStackSize;
    }
}

size_t Context::getStackSize() const
{
    return m_stackSize;
}

void Context::setAttributeStackSize( size_t continuationStackSize, size_t directCallableStackSizeFromTraversal, size_t directCallableStackSizeFromState )
{
    // rtx only, no effect on megakernel exelwtion mode
    // If all stack size values are 0, the attribute will be ignored for stack size computation.
    m_continuationStackSize                = continuationStackSize;
    m_directCallableStackSizeFromTraversal = directCallableStackSizeFromTraversal;
    m_directCallableStackSizeFromState     = directCallableStackSizeFromState;
}

void Context::getAttributeStackSize( AttributeStackSize& attrStackSize ) const
{
    attrStackSize.continuation          = m_continuationStackSize;
    attrStackSize.direct_from_traversal = m_directCallableStackSizeFromTraversal;
    attrStackSize.direct_from_state     = m_directCallableStackSizeFromState;
}

void Context::setAPIMaxCallableProgramDepth( unsigned int maxCallableProgramDepth )
{
    m_setByAPIMaxCallableProgramDepth = true;
    setMaxCallableProgramDepth( maxCallableProgramDepth );
}

void Context::setMaxCallableProgramDepth( unsigned int maxCallableProgramDepth )
{
    if( maxCallableProgramDepth != m_maxCallableProgramDepth )
    {
        getUpdateManager()->eventContextSetMaxCallableProgramDepth( m_maxCallableProgramDepth, maxCallableProgramDepth );
        llog( 15 ) << "Context::setMaxCallableProgramDepth changing maximum callable program call depth from "
                   << m_maxCallableProgramDepth << " to " << maxCallableProgramDepth << "\n";
        m_maxCallableProgramDepth = maxCallableProgramDepth;
    }
}

unsigned int Context::getMaxCallableProgramDepth() const
{
    return m_maxCallableProgramDepth;
}

unsigned int Context::getMaxCallableProgramDepthDefault() const
{
    return MAX_CALLABLE_PROGRAM_DEPTH_DEFAULT;
}

void Context::setAPIMaxTraceDepth( unsigned int maxTraceDepth )
{
    m_setByAPIMaxTraceDepth = true;
    setMaxTraceDepth( maxTraceDepth );
}

void Context::setMaxTraceDepth( unsigned int maxTraceDepth )
{
    if( maxTraceDepth != m_maxTraceDepth )
    {
        getUpdateManager()->eventContextSetMaxTraceDepth( m_maxTraceDepth, maxTraceDepth );
        llog( 15 ) << "Context::setMaxTraceDepth changing maximum trace depth from " << m_maxTraceDepth << " to "
                   << maxTraceDepth << "\n";
        m_maxTraceDepth = maxTraceDepth;
    }
}

unsigned int Context::getMaxTraceDepth() const
{
    return m_maxTraceDepth;
}

unsigned int Context::getMaxTraceDepthDefault() const
{
    return MAX_TRACE_DEPTH_DEFAULT;
}

void Context::setPrintEnabled( bool enabled )
{
    m_printManager->setPrintEnabled( enabled );
}

bool Context::getPrintEnabled() const
{
    return m_printManager->getPrintEnabled();
}

void Context::setPrintBufferSize( size_t bufsize )
{
    m_printManager->setPrintBufferSize( bufsize );
}

size_t Context::getPrintBufferSize() const
{
    return m_printManager->getPrintBufferSize();
}

void Context::setPrintLaunchIndex( int x, int y, int z )
{
    m_printManager->setPrintLaunchIndex( x, y, z );
}

int3 Context::getPrintLaunchIndex() const
{
    return m_printManager->getPrintLaunchIndex();
}

void Context::setExceptionEnabled( uint64_t& flags, const RTexception exception, const bool enabled )
{
    if( exception == RT_EXCEPTION_ALL )
    {
        // Set all available exceptions.
        if( enabled )
            flags = (uint64_t)-1;
        else
            flags = 0;
    }
    else
    {
        const int idx = getExceptionFlagIndex( exception );
        RT_ASSERT( idx < NUM_EXCEPTIONS );
        setBit( flags, idx, enabled );
    }
}

bool Context::getExceptionEnabled( const uint64_t flags, const RTexception exception )
{
    if( exception == RT_EXCEPTION_ALL )
    {
        // Check if all exceptions are enabled.
        for( int i = 0; i < NUM_EXCEPTIONS; ++i )
            if( !isBitSet( flags, i ) )
                return false;
        return true;
    }
    else
    {
        const int idx = getExceptionFlagIndex( exception );
        RT_ASSERT( idx < NUM_EXCEPTIONS );
        return isBitSet( flags, idx );
    }
}

bool Context::hasAnyExceptionEnabled() const
{
    return hasAnyExceptionEnabled( m_exceptionFlags );
}

bool Context::hasOnlyStackOverflowEnabled() const
{
    return hasOnlyStackOverflowEnabled( m_exceptionFlags );
}

UsageReport& Context::getUsageReport()
{
    return m_usageReport;
}

uint64_t Context::getExceptionFlags() const
{
    return m_exceptionFlags;
}

const char* Context::getPublicString( const std::string& string ) const
{
    m_publicString = string;

    return m_publicString.c_str();
}

void Context::setUsageReportCallback( RTusagereportcallback callback, int verbosity, void* cbdata )
{
    if( verbosity < 0 || verbosity > 3 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Usage report verbosity must be in [0, 3] range.  Given: ", verbosity );
    if( callback == nullptr && verbosity != 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "If callback function pointer is NULL, verbosity must be zero. Given: ", verbosity );

    m_usageReport.setUserCallback( callback, verbosity, cbdata );
}

bool Context::useRtxDataModel() const
{
    RT_ASSERT_MSG( m_dataModel != DataModel::Unset, "Illegal use of data model before being set" );
    return m_dataModel == DataModel::RTX;
}

RTCore* Context::getRTCore() const
{
    return m_rtcore.get();
}

bool Context::RtxUniversalTraversalEnabled() const
{
    return m_rtxNeedsUniversalTraversal;
}

bool Context::RtxMotionBlurEnabled() const
{
    return m_rtxHasMotionBlur;
}

RTXES* Context::getRTXExelwtionStrategy() const
{
    return m_rtxES.get();
}

unsigned int Context::getAabbEntry() const
{
    // m_numEntryPoints is always the entrypoint for the bbox program
    return m_numEntryPoints;
}

GlobalScope* Context::getGlobalScope() const
{
    return m_globalScope;
}


size_t Context::getLaunchCount() const
{
    return m_launchCount;
}

size_t Context::getKernelLaunchCount() const
{
    return m_kernelLaunchCounter;
}

size_t Context::getDenoiserLaunchCount() const
{
    return m_denoiserLaunchCounter;
}

size_t Context::getSSIMPredictorLaunchCount() const
{
    return m_ssimPredictorLaunchCounter;
}

void Context::incrDenoiserLaunchCount()
{
    ++m_denoiserLaunchCounter;
}

void Context::incrSSIMPredictorLaunchCount()
{
    ++m_ssimPredictorLaunchCounter;
}

double Context::getDenoiserTimeSpent() const
{
    return m_denoiserTimeSpent;
}

double Context::getSSIMPredictorTimeSpent() const
{
    return m_ssimPredictorTimeSpent;
}

void Context::addDenoiserTimeSpent( double timeSpent )
{
    m_denoiserTimeSpent += timeSpent;
}

void Context::addSSIMPredictorTimeSpent( double timeSpent )
{
    m_ssimPredictorTimeSpent += timeSpent;
}

ASManager* Context::getASManager() const
{
    return m_ASManager.get();
}

BindingManager* Context::getBindingManager() const
{
    return m_bindingManager.get();
}

DeviceManager* Context::getDeviceManager() const
{
    return m_deviceManager.get();
}

optix_exp::DiskCache* Context::getDiskCache() const
{
    return m_diskCache.get();
}

optix_exp::EncryptionManager* Context::getEncryptionManager() const
{
    return m_encryptionManager.get();
}

ErrorManager* Context::getErrorManager() const
{
    return m_errorManager.get();
}

LLVMManager* Context::getLLVMManager() const
{
    return m_llvmManager.get();
}

MemoryManager* Context::getMemoryManager() const
{
    return m_memoryManager.get();
}

ObjectManager* Context::getObjectManager() const
{
    return m_objectManager.get();
}

PagingService* Context::getPagingManager() const
{
    return m_pagingManager.get();
}

PlanManager* Context::getPlanManager() const
{
    return m_planManager.get();
}

ProfileManager* Context::getProfileManager() const
{
    return m_profileManager.get();
}

ProgramManager* Context::getProgramManager() const
{
    return m_programManager.get();
}

SBTManager* Context::getSBTManager() const
{
    return m_sbtManager.get();
}

SharedProgramManager* Context::getSharedProgramManager() const
{
    return m_sharedProgramManager.get();
}

TableManager* Context::getTableManager() const
{
    return m_tableManager.get();
}

UpdateManager* Context::getUpdateManager() const
{
    return m_updateManager.get();
}

ValidationManager* Context::getValidationManager() const
{
    return m_validationManager.get();
}

WatchdogManager* Context::getWatchdogManager() const
{
    return m_watchdogManager.get();
}

optix::ThreadPool* Context::getThreadPool() const
{
    return m_threadPool.get();
}

int Context::getExceptionFlagIndex( const RTexception exception )
{
    // Make sure the exception index is valid.
    if( ( exception < FIRST_EXCEPTION_CODE || exception > LAST_EXCEPTION_CODE ) && exception != RT_EXCEPTION_ALL )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid exception index: ", exception );
    }

    return exception - FIRST_EXCEPTION_CODE;
}


void Context::setExceptionEnabled( const RTexception exception, const bool enabled )
{
    // Make sure the exception index is valid.
    if( ( exception < FIRST_EXCEPTION_CODE || exception > LAST_EXCEPTION_CODE ) && exception != RT_EXCEPTION_ALL )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid exception index: ", exception );
    }

    const uint64_t oldFlags = m_exceptionFlags;
    setExceptionEnabled( m_exceptionFlags, exception, enabled );

    // Notify of changes if necessary
    if( oldFlags != m_exceptionFlags )
        getUpdateManager()->eventContextSetExceptionFlags( oldFlags, m_exceptionFlags );
}

bool Context::getExceptionEnabled( const RTexception exception ) const
{
    // Make sure the exception index is valid.
    if( ( exception < FIRST_EXCEPTION_CODE || exception > LAST_EXCEPTION_CODE ) && exception != RT_EXCEPTION_ALL )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid exception index: ", exception );
    }

    return getExceptionEnabled( m_exceptionFlags, exception );
}

bool Context::hasAnyExceptionEnabled( const uint64_t flags )
{
    return flags != 0;
}

bool Context::hasOnlyStackOverflowEnabled( const uint64_t flags )
{
    const uint64_t soFlag = uint64_t( 1 ) << getExceptionFlagIndex( RT_EXCEPTION_STACK_OVERFLOW );
    return flags == soFlag;
}

bool Context::hasProductSpecificExceptionsEnabled( uint64_t flags )
{
    const uint64_t soFlag  = uint64_t( 1 ) << getExceptionFlagIndex( RT_EXCEPTION_STACK_OVERFLOW );
    const uint64_t tdeFlag = uint64_t( 1 ) << getExceptionFlagIndex( RT_EXCEPTION_TRACE_DEPTH_EXCEEDED );
    return ( flags & ~( soFlag | tdeFlag ) ) != 0;
}

void Context::setDefaultExelwtionStrategy( const std::string& defaultExelwtionStrategy )
{
    m_defaultExelwtionStrategy = defaultExelwtionStrategy;
}

std::string Context::getDefaultExelwtionStrategy()
{
    return m_defaultExelwtionStrategy;
}

void Context::setForceDemandLoadWholeMipLevel( bool forced )
{
    if( !k_ilwokeCallbacksPerMipLevel.isSet() )
        k_ilwokeCallbacksPerMipLevel.set( forced );
}

bool Context::getForceDemandLoadWholeMipLevel()
{
    return k_ilwokeCallbacksPerMipLevel.get();
}

void Context::setDemandLoadNumVirtualPages( unsigned int numPages )
{
    s_numVirtualPages = numPages;
}

unsigned int Context::getDemandLoadNumVirtualPages()
{
    return s_numVirtualPages;
}

bool Context::getDemandLoadForceSynchronous() const
{
    return m_demandLoadForceSynchronous;
}

bool Context::getDemandLoadUseAsyncCopies() const
{
    return m_demandLoadUseAsyncCopies;
}

bool Context::getDemandLoadStampMemoryBlocks() const
{
    return m_demandLoadStampMemoryBlocks;
}

void Context::progressiveLaunchWorker( unsigned int max_subframes, unsigned int entryPointIndex, int dim, RTsize width, RTsize height, RTsize depth )
{
    // This function is the entry point of a thread to handle progressive launches. Because the LWCA Contexts need to be
    // properly bound to work here, we iterate over the devices of this context and bind them to this thread in
    // preparation for the thread. This is possibly not the right thing to do, but close enough. The keen eye will
    // see that the object is destroyed immediately, effectively doing a bind(), unbind() per device. Keeping the
    // binding to the whole function scope gives a segfault, so no good either.
    //
    // TODO: there is possibly a LWDAContext()->bind() missing from inside the launch setup, now that there is a
    // launch from a separate thread. Given the multiple device operations and different scopes ilwolved, it gets
    // quite tricky to pinpoint the right place(s).

    m_subframe_index = 0;
    while( m_cancel_progressive_flag == false && ( max_subframes == 0 || m_subframe_index < max_subframes ) )
    {
        try
        {
            // Attempt a regular launch and remember any errors so we can report them later
            // TODO: Can this use launchInternal so that interop is not handled each time?
            launchFromAPI( entryPointIndex, dim, width, height, depth );
        }
        catch( Exception& e )
        {
            m_async_exception = e.clone();
            break;
        }
        catch( const std::exception& e )
        {
            m_async_exception = new UnknownError( RT_EXCEPTION_INFO, e.what() );
            break;
        }
        catch( ... )
        {
            m_async_exception = new UnknownError( RT_EXCEPTION_INFO, "Caught unknown exception in progressive launch" );
            break;
        }

        for( const auto& stream : getObjectManager()->getStreamBuffers() )
        {
            // If this is the first frame of the progressive launch, reset the aclwmulation buffer. We do this as late
            // as possible, because the client app may map the stream while the trace kernel is already running and
            // still expect updates from the previous progressive launch.
            if( m_subframe_index == 0 )
            {
                stream->resetAclwm();
            }

            stream->fillFromSource( max_subframes );
        }

        m_subframe_index++;
    }
}

ExelwtionStrategy* Context::determineExelwtionStrategy( unsigned int /*entry*/ ) const
{
    // This function is designed to facilitate dynamic switching of
    // exelwtion strategies, including CPU strategies when applicable.
    std::string es = m_exelwtionStrategy;
    if( es == "null" )
    {
        return m_nullES.get();
    }
    else if( es == "rtx" )
    {
        if( deviceCast<LWDADevice>( m_deviceManager->primaryDevice() ) )
        {
            return m_rtxES.get();
        }
        else
        {
            RT_ASSERT_FAIL_MSG( "RTX not defined for primary device" );
        }
    }
    else
    {
        RT_ASSERT_FAIL_MSG( "Invalid exelwtion strategy" );
    }
}

void Context::saveNodeGraph( const char* where, bool forceWrite ) const
{
#if defined( DEBUG ) || defined( DEVELOP )
    std::string saveNodegraph( forceWrite ? "graph.dot" : k_saveNodegraph.get() );
    if( saveNodegraph.empty() )
        return;

    NodegraphPrinter printer( getObjectManager(), getProgramManager(), getBindingManager() );
    printer.setPrintReferences( k_saveNodegraphRefs.get() );
    printer.run();

    std::string filename = insertBeforeExtension( saveNodegraph, stringf( "%02zu", m_nodegraphSaveNumber ) );
    m_nodegraphSaveNumber++;
    lprint << "Saving node graph" << ( where ? " at " + std::string( where ) : "" ) << " to file: " << filename << '\n';
    if( !saveStringToFile( printer.str(), filename ) )
        throw UnknownError( RT_EXCEPTION_INFO, "Error saving node graph" + ( where ? " at " + std::string( where ) : "" )
                                                   + " to file: " + filename + "\n" );
#endif
}

bool Context::shuttingDown() const
{
    return m_shuttingDown;
}

std::string Context::attributeCopyErrorString( size_t source_size, RTsize dest_size, const char* attribute ) const
{
    std::ostringstream out;
    out << "destination memory size (" << dest_size << ") for " << attribute << " is not equal to " << source_size;
    return out.str();
}

float Context::getLastKernelTime()
{
    return m_lastKernelTime;
}

bool Context::initializeDiskCache()
{
    // If DiskCache initialization fails, the cache is disabled and a warning is
    // added to the usage report.  The application can attempt to set a new
    // location via the API, but failure in that case will produce an error.
    m_diskCache.reset( new optix_exp::DiskCache() );
    optix_exp::ErrorDetails errDetails;
    if( OptixResult result = m_diskCache->init( getDeviceContextLogger(), errDetails, CACHE_DB_FILE_NAME ) )
    {
        m_diskCache.reset( new optix_exp::DiskCache() );
        getDeviceContextLogger().callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "[DISK CACHE]",
                                           errDetails.m_description.c_str() );
        return false;
    }
    return true;
}

void Context::setDiskCacheLocation( const std::string& path )
{
    // Cache the old memory limits so they can be restored after resetting.
    size_t prevLow, prevHigh;
    getDiskCacheMemoryLimits( prevLow, prevHigh );

    // Explicitly destroy the DiskCache to force the database to close.
    m_diskCache->destroy( getDeviceContextLogger() );
    m_diskCache.reset();

    // Construct a new DiskCache with the new path.
    m_diskCache.reset( new optix_exp::DiskCache() );
    optix_exp::ErrorDetails errDetails;
    if( OptixResult result = m_diskCache->init( getDeviceContextLogger(), errDetails, CACHE_DB_FILE_NAME, path ) )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, errDetails.m_description );
    }

    // Restore the previous limits.
    setDiskCacheMemoryLimits( prevLow, prevHigh );
}

std::string Context::getDiskCacheLocation() const
{
    return m_diskCache->getPath( CACHE_DB_FILE_NAME );
}

void Context::setDiskCacheMemoryLimits( size_t lowWaterMark, size_t highWaterMark )
{
    lowWaterMark  = k_lowWaterMark.isDefault() ? lowWaterMark : k_lowWaterMark.get();
    highWaterMark = k_highWaterMark.isDefault() ? highWaterMark : k_highWaterMark.get();

    optix_exp::ErrorDetails errDetails;
    if( m_diskCache->setMemoryLimits( lowWaterMark, highWaterMark, getDeviceContextLogger(), errDetails ) != OPTIX_SUCCESS )
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid OptiX cache memory limits, low: " + std::to_string( lowWaterMark )
                                                   + ", high: " + std::to_string( highWaterMark ) );
    }
}

void Context::getDiskCacheMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const
{
    m_diskCache->getMemoryLimits( lowWaterMark, highWaterMark );
}

}  // namespace optix
