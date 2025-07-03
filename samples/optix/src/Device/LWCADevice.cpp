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

#include <Device/LWDADevice.h>

#include <LWCA/Memory.h>
#include <LWCA/Module.h>
#include <LWCA/TexRef.h>
#include <Context/Context.h>
#include <Context/RTCore.h>
#include <Control/ManagedLWDAModule.h>
#include <Device/APIDeviceAttributes.h>
#include <Device/DeviceManager.h>
#include <Util/BufferUtil.h>
#include <Util/CodeRange.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/Metrics.h>
#include <Util/LWML.h>
#include <Util/UsageReport.h>
#include <Util/optixUuid.h>

#include <corelib/misc/Cast.h>
#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/PtxCompilerUtils.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidContext.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/RTFormatUtil.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Thread.h>

#include <rtcore/interface/rtcore.h>

// this is needed to properly decode Chip IDs, lives in /dev/gpu_drv/bugfix_main/sdk/lwpu/inc/
#include <Lwcm.h>

using namespace optix;
using namespace optix::lwca;
using namespace prodlib;
using namespace corelib;


namespace {
// clang-format off
Knob<bool>         k_forceTTUenabled(         RT_DSTRING("lwca.forceTTUenabled"),         false, RT_DSTRING( "Forces use of TTU/MTTU, regardless of what LWCA reports.  o7.disableTTU takes precendence." ) );
Knob<int>          k_moduleCacheSize(         RT_DSTRING("lwca.moduleCacheSize"),             8, RT_DSTRING( "Maximum number of entries in the module cache, per device" ) );
Knob<size_t>       k_lwdaPrintfBufferSize (   RT_DSTRING("lwca.printfBufferSize"),            0, RT_DSTRING( "Change size of LWCA's printf buffer size. Default is 0, which means don't change it." ) );
PublicKnob<size_t> k_lwdaStackSize(           RT_PUBLIC_DSTRING("lwca.stackSize"),            0, RT_PUBLIC_DSTRING( "Change the size of the local memory stack for LWCA applications that make use of the ABI.  Default is 0, which means don't change it." ) );
Knob<bool>         k_p2pAllowed(              RT_DSTRING("mem.p2pAllowed"),                true, RT_DSTRING( "Allow use of LWCA peer to peer memory access" ) );
Knob<bool>         k_allowBindless(           RT_DSTRING("lwca.allowBindlessTextures"),    true, RT_DSTRING( "Allow use of bindless textures" ) );
Knob<int>          k_maximumBoundTextures(    RT_DSTRING("lwca.maximumBoundTextures"),       -1, RT_DSTRING( "Number of bound hardware textures" ) );
Knob<int>          k_maximumBindlessTextures( RT_DSTRING("lwca.maximumBindlessTextures"),    -1, RT_DSTRING( "Number of bindless hardware textures" ) );
Knob<std::string>  k_gatherLwptiMetrics(      RT_DSTRING( "launch.gatherMetrics" ),          "", RT_DSTRING( "Comma separated list of LWPTI metrics to gather." ) );

// Knobs for ptxas
Knob<bool>        k_useCompileFast(          RT_DSTRING("lwca.useOCGCompileFast"),        true, RT_DSTRING( "Use -cloning=no --fast-compile" ) );
Knob<int>         k_maxRegCount(             RT_DSTRING("lwca.maxRegCount"),                 0, RT_DSTRING( "Set ptxas max reg count. 0 = let OptiX figure out a good value based on GPU architecture. -1 = don't limit registers." ) );
Knob<int>         k_forceSmVersion(          RT_DSTRING("lwca.forceSmVersion"),              0, RT_DSTRING( "Force a specific SM target (20,30,35,etc.) when using external ptxas/lwlink binaries" ) );
Knob<int>         k_sassOptimization(        RT_DSTRING("lwca.sassOptimization"),           -1, RT_DSTRING( "Sass optimization flag value. -1 = don't specify optimization explicitly" ) );
Knob<std::string> k_offlinePtxas(            RT_DSTRING("lwca.offlinePtxas"),               "", RT_DSTRING( "Specify ptxas location for offline compilation" ) );
Knob<std::string> k_offlineIOPath(           RT_DSTRING("lwca.offlineIOPath"),              "", RT_DSTRING( "Where to read and write files for offline ptxas/lwlink processing" ) );
Knob<std::string> k_ptxasArgs(               RT_DSTRING("lwca.ptxasArgs"),                  "", RT_DSTRING( "String with command-line arguments to ptxas" ) );
Knob<bool>        k_useOriBackendOnKepler(   RT_DSTRING("lwca.useOriBackendOnKepler"),    true, RT_DSTRING( "Use ORI compiler backend on Kepler GPUs" ) );
Knob<int>         k_cacheModeL1(             RT_DSTRING("lwca.cacheModeL1"),                 1, RT_DSTRING( "0 = disable L1 (dlcm=cg), 1 = enable L1 (dlcm=ca), -1 = don't specify dlcm" ) );
Knob<bool>        k_ptxasVerbose(            RT_DSTRING("lwca.verbosePtxas"),            false, RT_DSTRING( "Enable verbose output for ptxas" ) );
Knob<bool>        k_addLineInfo(             RT_DSTRING("lwca.addLineInfo"),             false, RT_DSTRING( "Generate line-number information for device code." ) );
// clang-format on
}

LWDADevice::LWDADevice( optix::Context* context, int lwdaOrdinal, lwca::ComputeCapability overrideSMVersion )
    : Device( context )
    , m_lwdaOrdinal( lwdaOrdinal )
    , m_overrideSMVersion( overrideSMVersion )
{

    if( lwdaOrdinal >= 0 )
        m_device = lwca::Device::get( lwdaOrdinal );
    m_SMVersion  = m_device.computeCapability();

    // Query Chip architecture and implementation. Use values in Lwcm.h to interpret them.
    unsigned int arch, impl;
    LWresult     res = lwdaDriver().LwDeviceGetArchImpl( m_device.get(), &arch, &impl );
    if( res == LWDA_SUCCESS )
    {
        m_architecture               = arch;
        m_architectureImplementation = impl;
    }
    unsigned char supportsTTU = 0;
    res                       = corelib::lwdaDriver().LwDeviceQueryTTU( &supportsTTU, m_device.get() );
    if( res == LWDA_SUCCESS )
    {
        m_supportsTTU = supportsTTU != 0;
    }

    if( k_forceTTUenabled.get() )
    {
        if( !m_supportsTTU )
        {
            lwarn << RT_DSTRING( "TTU has been force-enabled by knob.\n" );
        }
        m_supportsTTU = true;
    }
}

LWDADevice::~LWDADevice()
{
    disable();
}

ComputeCapability LWDADevice::computeCapability() const
{
    if( m_overrideSMVersion != SM_NONE() )
        return m_overrideSMVersion;
    return m_SMVersion;
}

void LWDADevice::enable()
{
    if( isEnabled() )
        return;

    const LWML::MemoryInfo meminfo0 = LWML::getMemoryInfo( m_device.getPCIBusId() );

    static Thread::Mutex g_lwdaDeviceEnableMutex;
    Thread::Lock         lock( g_lwdaDeviceEnableMutex );


    //
    // NOTE:  I don't believe the comment below is accurate any more. Both runtime and
    // driver API contexts should work fine. I haven't tested this though. --martin
    //
    /* In order for LWCA and OptiX to interoperate, it is necessary that there be only
     a single LWCA context per device. Furthermore, LWCA must be initialized
     using the LWCA runtime API. After initialization the application may call the
     LWCA driver API, but only using the LWCA contexts created by the LWCA
     runtime API per device. There are two possible LWCA interop scenarios:
     1. The application has created a LWCA context (by performing some LWCA
        runtime operations) prior to OptiX initialization. OptiX will latch on to the
        existing LWCA context owned by the runtime API instead of creating its own.
     2. OptiX creates its own LWCA context using the LWCA runtime API (either
        upon a call to rtContextLaunch or
        rtBufferGetDevicePointer) prior to the application creating any
        LWCA contexts. Any subsequent LWCA calls made by the application will
        use OptiX's already created contexts.
  */
    // init runtime LWCA context if it was not initialized outside OptiX

    unsigned int primaryCtxflags;
    int          primaryCtxActive;
    lwca::Context::devicePrimaryCtxGetState( m_device, &primaryCtxflags, &primaryCtxActive );

    // note that LW_CTX_MAP_HOST is implicit for runtime LWCA context,
    // so no LW_CTX_MAP_HOST in devicePrimaryCtxSetFlags and devicePrimaryCtxGetState
    if( primaryCtxActive )
    {
        if( !( primaryCtxflags & LW_CTX_LMEM_RESIZE_TO_MAX ) )
        {
            lwarn << "Device flags have already been set. OptiX may not work unless primary LWCA context has been "
                     "created "
                     "with lwdaSetDeviceFlags(lwdaDeviceMapHost | lwdaDeviceLmemResizeToMax ) called before it.\n";
        }
    }
    else
    {
        lwca::Context::devicePrimaryCtxSetFlags( m_device, LW_CTX_LMEM_RESIZE_TO_MAX );
    }

    m_lwdaContext = lwca::Context::devicePrimaryCtxRetain( m_device );

    if( m_context->useRtxDataModel() )
    {
        m_lwdaContext.setLwrrent();

        m_context->getRTCore()->initializeRTCoreLibraryWithKnobs();

        RtcDeviceProperties deviceProperties = {};
        // Increment version number to force rtcore to generate a new identifier,
        // and prevent backward compatibility (even without having to rebuild rtcore.)
        deviceProperties.productIdentifier  = {'O', 'X', '6', /* version number: */ 0};
        deviceProperties.chipArchitecture   = m_architecture;
        deviceProperties.chipImplementation = m_architectureImplementation;
        deviceProperties.hasTTU             = m_supportsTTU;
        memcpy( &deviceProperties.productUuid, getOptixUUID(), sizeof( unsigned int ) * 4 );

        m_context->getRTCore()->deviceContextCreateForLWDA( m_lwdaContext.get(), &deviceProperties, &m_rtcDeviceContext );
    }

    size_t userRequestedPrintBufferSize = m_context->getPrintBufferSize();
    // A value of 0 means that the user did not specify any limit.
    // We change the default buffer size if the user has specified a request (via setPrintBufferSize)
    // or if the knob has been set.
    if( !k_lwdaPrintfBufferSize.isDefault() || userRequestedPrintBufferSize != 0 )
    {
        m_lwdaContext.setLwrrent();
        // WARNING, this limit can be only be applied to LWCA contexts for which no launch has ever taken place.
        // If this condition is not met a LWCA invalid value error is triggered.
        setPrintBufferSize( userRequestedPrintBufferSize );
    }

    if( !k_lwdaStackSize.isDefault() )
    {
        m_lwdaContext.setLwrrent();
        size_t old_size = m_lwdaContext.getLimit( LW_LIMIT_STACK_SIZE );
        size_t new_size = k_lwdaStackSize.get();
        m_lwdaContext.setLimit( LW_LIMIT_STACK_SIZE, new_size );
        lprint << "Changing LWCA stack size from " << old_size << " to " << new_size << "\n";
    }

    if( ( m_isCompileFastAvailable = isJITOptionAvailable( JIT_OPTION_COMPILE_FAST ) ) )
    {
        llog( 20 ) << "Compile fast available for " << m_device.getName() << std::endl;
    }
    else
    {
        llog( 20 ) << "Compile fast NOT available for " << m_device.getName() << std::endl;
    }

    if( ( m_isOriBackendOnKeplerAvailable = isJITOptionAvailable( JIT_OPTION_ORI_ON_KEPLER, 3 ) ) )
    {
        llog( 20 ) << "Kepler ORI backend available for " << m_device.getName() << std::endl;
    }
    else
    {
        llog( 20 ) << "Kepler ORI backend NOT available for " << m_device.getName() << std::endl;
    }

    // Create streams on the current LWCA context
    m_lwdaContext.setLwrrent();

    setStreamPoolSize( m_context->getMaxConlwrrentLaunches() );

    if( !m_event.get() )
        m_event = Event::create();

    // If LWPTI profiling is enabled, make sure this device's profiler is initialized.
    if( !k_gatherLwptiMetrics.get().empty() && !m_lwptiProfiler.isInitialized() )
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result = m_lwptiProfiler.initialize( lwdaContext().get(), k_gatherLwptiMetrics.get(), errDetails ) )
            throw UnknownError( RT_EXCEPTION_INFO, errDetails.m_description );

        std::string outputFileName = lwdaDevice().getName() + "-" + lwdaDevice().getPCIBusId();
        for( char& c : outputFileName )
        {
            if( c == ' ' || c == ':' )
                c = '-';
        }
        outputFileName += ".json";

        bool openWasSuccessful = m_lwptiProfiler.openOutputFile( outputFileName );
        if( !openWasSuccessful )
            throw UnknownError( RT_EXCEPTION_INFO,
                                corelib::stringf( "Error while opening LWPTI output file: %s", outputFileName.c_str() ) );
    }

    m_enabled = true;

    // Log memory used by LWCA context to usage report
    const LWML::MemoryInfo meminfo1 = LWML::getMemoryInfo( m_device.getPCIBusId() );
    if( meminfo0.total > 0 && meminfo1.total > 0 )  // check that LWML queries worked
    {
        const unsigned long long delta = meminfo1.used - meminfo0.used;
        ureport2( m_context->getUsageReport(), "MEM USAGE" )
            << "LWCA context memory (LWCA device: " << lwdaOrdinal() << "): " << std::fixed << std::setprecision( 1 )
            << static_cast<double>( delta ) / ( 1024.0 * 1024.0 ) << " MBytes" << std::endl;
    }
}

void LWDADevice::disable()
{
    if( !isEnabled() )
        return;

    m_lwdaContext.setLwrrent();

    if( m_lwptiProfiler.isInitialized() )
    {
        optix_exp::ErrorDetails errDetails;
        m_lwptiProfiler.deinitialize( errDetails );
    }

    // This helps stuff like Nsight get our data.  We don't want to call it after every
    // frame, because the overhead of this call is too high for that.
    lwca::Context::synchronize();

    // Ensure nobody is keeping a handle to a module when this function is called. LWCA modules
    // are associated with a LWCA context, so they cannot survive the context destroy.
    for( const auto& entry : m_cachedModules )
    {
        RT_ASSERT( entry.second.module.use_count() == 1 );
    }
    m_cachedModules.clear();

    if( m_context->useRtxDataModel() )
    {
        // Destroy rtcore context and cmdlist
        m_context->getRTCore()->deviceContextDestroy( m_rtcDeviceContext );
        for( auto list : m_rtcCommandLists )
            m_context->getRTCore()->commandListDestroy( list );
        m_rtcDeviceContext = nullptr;
        m_rtcCommandLists.clear();
    }

    // Destroy the streams in the stream pool
    for( auto stream : m_streams )
        stream.destroy();
    m_streams.clear();

    m_event.destroy();

    // Release context (replaces detroy)
    lwca::Context::devicePrimaryCtxRelease( m_device );
    m_lwdaContext = lwca::Context();

    // With the destruction of this device's LWCA context, this device no longer
    // has access to its peers.
    m_peerAccessEnabled.clear();
    m_lwlinkAccessible.clear();

    m_enabled = false;
}

void LWDADevice::makeLwrrent()
{
    RT_ASSERT_MSG( isEnabled(), "LWCA device not enabled. Make sure DeviceManager::enableActiveDevices() was called" );
    RT_ASSERT_MSG( m_lwdaContext.isValid(), "LWCA device context is not valid." );
    m_lwdaContext.setLwrrent();
}

const lwca::Device& LWDADevice::lwdaDevice() const
{
    return m_device;
}

const lwca::Context& LWDADevice::lwdaContext() const
{
    return m_lwdaContext;
}

RtcDeviceContext LWDADevice::rtcContext() const
{
    return m_rtcDeviceContext;
}

RtcCommandList LWDADevice::primaryRtcCommandList() const
{
    return m_rtcCommandLists[0];
}

int LWDADevice::lwdaOrdinal() const
{
    return m_lwdaOrdinal;
}

lwca::Stream LWDADevice::primaryStream() const
{
    return m_streams[0];
}

lwca::Stream optix::LWDADevice::getLwdaStream( unsigned int index ) const
{
    return m_streams[index % m_streams.size()];
}

RtcCommandList LWDADevice::getRtcCommandList( unsigned int index ) const
{
    return m_rtcCommandLists[index % m_rtcCommandLists.size()];
}


void LWDADevice::syncToDevice( LWDADevice* otherDevice )
{
    otherDevice->makeLwrrent();
    otherDevice->m_event.record( otherDevice->m_streams[0] );
    makeLwrrent();
    m_streams[0].waitEvent( otherDevice->m_event, 0 );
}

void LWDADevice::syncStreamToPrimaryStream( lwca::Stream stream )
{
    makeLwrrent();
    m_event.record( primaryStream() );
    stream.waitEvent( m_event, 0 );
}

std::string LWDADevice::deviceName() const
{
    std::ostringstream out;
    out << "LWDADevice " << allDeviceListIndex() << " (" << m_device.getName() << ")";
    return out.str();
}

void LWDADevice::dump( std::ostream& out ) const
{
    Device::dump( out, "LWCA" );
    out << "  LWCA ordinal         : " << m_lwdaOrdinal << std::endl;
    out << "  LWdevice             : " << m_device.get() << std::endl;
    out << "  Name                 : " << m_device.getName() << std::endl;
    out << "  Compute capability   : " << m_device.computeCapability().toString( true ) << std::endl;
    out << "  Total Memory         : " << m_device.totalMem() / 1024 / 1024 << " MB\n";
    out << "  PCI Bus ID           : " << m_device.getPCIBusId() << std::endl;
    out << "  Max threads/CTA      : " << m_device.MAX_THREADS_PER_BLOCK() << std::endl;
    out << "  Max CTA dim          : " << m_device.MAX_BLOCK_DIM_X() << "x" << m_device.MAX_BLOCK_DIM_Y() << "x"
        << m_device.MAX_BLOCK_DIM_Z() << std::endl;
    out << "  Max grid dim         : " << m_device.MAX_GRID_DIM_X() << "x" << m_device.MAX_GRID_DIM_Y() << "x"
        << m_device.MAX_GRID_DIM_Z() << std::endl;
    out << "  Max smem/CTA         : " << m_device.MAX_SHARED_MEMORY_PER_BLOCK() / 1024 << " KB\n";
    out << "  Constant mem         : " << m_device.TOTAL_CONSTANT_MEMORY() / 1024 << " KB\n";
    out << "  Warp size            : " << m_device.WARP_SIZE() << std::endl;
    out << "  Max pitch            : " << m_device.MAX_PITCH() << std::endl;
    out << "  Max regs/CTA         : " << m_device.MAX_REGISTERS_PER_BLOCK() << std::endl;
    out << "  Hot clock            : " << m_device.CLOCK_RATE() / 1000 << " Mhz\n";
    out << "  Tex alignment        : " << m_device.TEXTURE_ALIGNMENT() << std::endl;
    out << "  Num SMs              : " << m_device.MULTIPROCESSOR_COUNT() << std::endl;
    out << "  Exelwtion timeout    : " << m_device.KERNEL_EXEC_TIMEOUT() << std::endl;
    out << "  Integrated           : " << m_device.INTEGRATED() << std::endl;
    out << "  Can map host memory  : " << m_device.CAN_MAP_HOST_MEMORY() << std::endl;
    out << "  Compute mode         : " << m_device.COMPUTE_MODE() << std::endl;
    out << "  Tex1d width          : " << m_device.MAXIMUM_TEXTURE1D_WIDTH() << std::endl;
    out << "  Tex2d size           : " << m_device.MAXIMUM_TEXTURE2D_WIDTH() << "x"
        << m_device.MAXIMUM_TEXTURE2D_HEIGHT() << std::endl;
    out << "  Tex3d size           : " << m_device.MAXIMUM_TEXTURE3D_WIDTH() << "x"
        << m_device.MAXIMUM_TEXTURE3D_HEIGHT() << "x" << m_device.MAXIMUM_TEXTURE3D_DEPTH() << std::endl;
    out << "  Tex2d layered size   : " << m_device.MAXIMUM_TEXTURE2D_LAYERED_WIDTH() << "x"
        << m_device.MAXIMUM_TEXTURE2D_LAYERED_HEIGHT() << "[" << m_device.MAXIMUM_TEXTURE2D_LAYERED_LAYERS() << "]"
        << std::endl;
    out << "  Surface alignment    : " << m_device.SURFACE_ALIGNMENT() << std::endl;
    out << "  Conlwrrent kernels   : " << m_device.CONLWRRENT_KERNELS() << std::endl;
    out << "  ECC enabled          : " << m_device.ECC_ENABLED() << std::endl;
    out << "  PCI bus ID           : " << m_device.PCI_BUS_ID() << std::endl;
    out << "  PCI device ID        : " << m_device.PCI_DEVICE_ID() << std::endl;
    out << "  TCC driver           : " << m_device.TCC_DRIVER() << std::endl;
    out << "  Memory clock         : " << m_device.MEMORY_CLOCK_RATE() / 1000 << " Mhz" << std::endl;
    out << "  Glbl mem bus width   : " << m_device.GLOBAL_MEMORY_BUS_WIDTH() << std::endl;
    out << "  L2 cache size        : " << m_device.L2_CACHE_SIZE() / 1024 << " KB" << std::endl;
    out << "  Max threads / SM     : " << m_device.MAX_THREADS_PER_MULTIPROCESSOR() << std::endl;
    out << "  Async engine count   : " << m_device.ASYNC_ENGINE_COUNT() << std::endl;
    out << "  Unified addressing   : " << m_device.UNIFIED_ADDRESSING() << std::endl;
    out << "  Tex1d layered size   : " << m_device.MAXIMUM_TEXTURE1D_LAYERED_WIDTH() << "["
        << m_device.MAXIMUM_TEXTURE1D_LAYERED_LAYERS() << "]" << std::endl;
    out << "  Tex1d gather size    : " << m_device.MAXIMUM_TEXTURE2D_GATHER_WIDTH() << "x"
        << m_device.MAXIMUM_TEXTURE2D_GATHER_HEIGHT() << std::endl;
    out << "  Tex3d alternate size : " << m_device.MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE() << "x"
        << m_device.MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE() << "x" << m_device.MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE() << std::endl;
    out << "  PCI domain ID        : " << m_device.PCI_DOMAIN_ID() << std::endl;
    out << "  Tex pitch alignment  : " << m_device.TEXTURE_PITCH_ALIGNMENT() << std::endl;
    out << "  Lwbemap size         : " << m_device.MAXIMUM_TEXTURELWBEMAP_WIDTH() << "x"
        << m_device.MAXIMUM_TEXTURELWBEMAP_LAYERED_WIDTH() << "[" << m_device.MAXIMUM_TEXTURELWBEMAP_LAYERED_LAYERS()
        << "]" << std::endl;
    out << "  Surf1d width         : " << m_device.MAXIMUM_SURFACE1D_WIDTH() << std::endl;
    out << "  Surf2d size          : " << m_device.MAXIMUM_SURFACE2D_WIDTH() << "x"
        << m_device.MAXIMUM_SURFACE2D_HEIGHT() << std::endl;
    out << "  Surf3d size          : " << m_device.MAXIMUM_SURFACE3D_WIDTH() << "x"
        << m_device.MAXIMUM_SURFACE3D_HEIGHT() << "x" << m_device.MAXIMUM_SURFACE3D_DEPTH() << std::endl;
    out << "  Surf1d layered size  : " << m_device.MAXIMUM_SURFACE1D_LAYERED_WIDTH() << "["
        << m_device.MAXIMUM_SURFACE1D_LAYERED_LAYERS() << "]" << std::endl;
    out << "  Surf2d layered size  : " << m_device.MAXIMUM_SURFACE2D_LAYERED_WIDTH() << "x"
        << m_device.MAXIMUM_SURFACE2D_LAYERED_HEIGHT() << "[" << m_device.MAXIMUM_SURFACE2D_LAYERED_LAYERS() << "]"
        << std::endl;
    out << "  Surf lwbemap size    : " << m_device.MAXIMUM_SURFACELWBEMAP_WIDTH() << "x"
        << m_device.MAXIMUM_SURFACELWBEMAP_LAYERED_WIDTH() << "[" << m_device.MAXIMUM_SURFACELWBEMAP_LAYERED_LAYERS()
        << "]" << std::endl;
    out << "  Tex1d linear width   : " << m_device.MAXIMUM_TEXTURE1D_LINEAR_WIDTH() << std::endl;
    out << "  Tex2d linear size    : " << m_device.MAXIMUM_TEXTURE2D_LINEAR_WIDTH() << "x"
        << m_device.MAXIMUM_TEXTURE2D_LINEAR_HEIGHT() << std::endl;
    out << "  Tex2d linear pitch   : " << m_device.MAXIMUM_TEXTURE2D_LINEAR_PITCH() << std::endl;
    out << "  Tex2d mipmapped size : " << m_device.MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH() << "x"
        << m_device.MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT() << std::endl;
    out << "  Tex1d mipmapped width: " << m_device.MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH() << std::endl;
    out << "  Stream priorities    : " << m_device.STREAM_PRIORITIES_SUPPORTED() << std::endl;
}

void LWDADevice::dumpShort( std::ostream& out ) const
{
    out << "Device: " << m_device.get() << ", " << m_device.getName() << ", pciBusID: " << m_device.PCI_BUS_ID()
        << ", pciDeviceID: " << m_device.PCI_DEVICE_ID() << ", pciDomainID: " << m_device.PCI_DOMAIN_ID()
        << ", TCC: " << m_device.TCC_DRIVER() << std::endl;
}

/*
 * Module management
 */

#if 0
// TODO: migrate these functions
  Module::Handle Module::createFromMemoryLwbin(const char *lwbin)
  {
#if 0
    LWmodule module;
    CALL_LWDA_DRIVER_THROW(lwdaDriver().lwModuleLoadData(&module, lwbin));
    return Handle( new Module(module) );
#endif
  }

  // A fallback that allows OCG JIT to parse some command line options that aren't passable as enums
  Module::Handle Module::createFromMemoryPtxLoadFat( const char *ptx, int maxRegCount, int optLevel, int target /*lwca::ComputeCapability target */ )
  {
#if 0

    std::ostringstream p;
    p << "compute_" << valueFromComputeCapability(target);
    const std::string s = p.str();
    const char *pstr = s.c_str();

    __lwdaFatPtxEntry ptxE[] = {{const_cast<char *>(pstr), const_cast<char *>(ptx)}, {0, 0}};
    __lwdaFatLwbinEntry lwbin[] = {{0, 0}};
    __lwdaFatDebugEntry debug[] = {{0, 0}};

    __lwdaFatLwdaBinary fat;
    memset(&fat, 0, sizeof(fat));

    fat.magic = __lwdaFatMAGIC;
    fat.version = __lwdaFatVERSION;
    fat.key = const_cast<char*>( "Key" );
    fat.ident = const_cast<char*>( "Ident" );

    std::ostringstream usageMode;
    usageMode << k_ptxasArgs.get();

    if( maxRegCount != -1 )
      usageMode << "-maxrregcount=" << maxRegCount;
    std::string strUsageMode = usageMode.str();
    fat.usageMode = const_cast<char*>(strUsageMode.c_str());
    fat.ptx = ptxE;
    fat.lwbin = lwbin;
    fat.debug = debug;
    fat.debugInfo = 0;
    fat.flags = __lwdaFatDontSearchFlag;
#if !defined( ENABLE_LWBIN_GETTING )
    fat.flags |= __lwdaFatDontCacheFlag;
#endif

#if 0
    LWmodule module;
    CALL_LWDA_DRIVER_THROW( lwdaDriver().lwModuleLoadFatBinary( &module, &fat ) );
    return Handle( new lwca::Module(module) );
#endif
#endif
  }
#endif

// RK: I would like to move this to PtxCompilerUtils, but the code depends on src/LWCA includes.
lwca::Module LWDADevice::createFromMemoryPtxLoadData( const std::string& ptx, int maxRegCount )
{
    makeLwrrent();
    LWjit_option     options[16];
    void*            optiolwalues[16];
    static const int LOG_BUF_SIZE = 4096;
    char             log[LOG_BUF_SIZE]{};
    char             errors[LOG_BUF_SIZE]{};
    int              n = 0;

#define ADD_OPTION( key, value )                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        options[n]        = key;                                                                                       \
        optiolwalues[n++] = (void*)( value );                                                                          \
    } while( 0 )

    if( maxRegCount != -1 )
        ADD_OPTION( LW_JIT_MAX_REGISTERS, (size_t)maxRegCount );

    const int optLevel = k_sassOptimization.get();
    if( optLevel != -1 )
        ADD_OPTION( LW_JIT_OPTIMIZATION_LEVEL, (size_t)optLevel );

    const int cacheModeL1 = k_cacheModeL1.get();
    if( cacheModeL1 != -1 )
        ADD_OPTION( LW_JIT_CACHE_MODE, cacheModeL1 == 0 ? LW_JIT_CACHE_OPTION_CG : LW_JIT_CACHE_OPTION_CA );

    if( m_isOriBackendOnKeplerAvailable && k_useOriBackendOnKepler.get() )
        ADD_OPTION( JIT_OPTION_ORI_ON_KEPLER, 1 );

    if( m_isCompileFastAvailable && k_useCompileFast.get() )
        ADD_OPTION( JIT_OPTION_COMPILE_FAST, 1 );

    ADD_OPTION( LW_JIT_FALLBACK_STRATEGY, LW_PREFER_PTX );
    ADD_OPTION( LW_JIT_INFO_LOG_BUFFER, log );
    ADD_OPTION( LW_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (size_t)LOG_BUF_SIZE );
    ADD_OPTION( LW_JIT_ERROR_LOG_BUFFER, errors );
    ADD_OPTION( LW_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (size_t)LOG_BUF_SIZE );
    ADD_OPTION( LW_JIT_WALL_TIME, 0 );  // Must be the last option since that's where we look for the clock value.

    LWresult     err;
    lwca::Module module      = lwca::Module::loadDataEx( ptx.c_str(), n, options, optiolwalues, &err );
    log[LOG_BUF_SIZE - 1]    = '\0';
    errors[LOG_BUF_SIZE - 1] = '\0';

    const float compiletime = binary_cast<float>( (int)(size_t)optiolwalues[n - 1] );
    llog( 20 ) << "lwModuleLoadDataEx OCG compile time: " << compiletime << " ms\n";
    llog( 20 ) << "lwModuleLoadDataEx errors: " << errors << std::endl;
    llog( 20 ) << "lwModuleLoadDataEx log: " << log << std::endl;

    if( err != LWDA_SUCCESS )
        throw LwdaError( RT_EXCEPTION_INFO, errors, err );

    return module;
}

int LWDADevice::getMaxRegisters() const
{
    const int smVersionMajor = computeCapability().major();
    const int smVersionMinor = computeCapability().minor();

    // TODO: These numbers may need to be updated.
    // They were decided based only on average launch times for vmd, arnold, and pixar traces.
    switch( smVersionMajor )
    {
        // clang-format off
  case 3:  return 64;
  case 5:  return 72;
  case 6:  return smVersionMinor == 0 ? 168 : 80;
  case 7:  return 168;
  default: return 128;
            // clang-format on
    }
}

lwca::Module LWDADevice::compileModuleFromPTX( const std::string& ptx, int launchCounter )
{
    TIMEVIZ_FUNC;
    CodeRange range( "PTX->SASS" );
    timerTick t0 = getTimerTick();

    // Determine the OptiX-specific options for ptxas.
    const int  smVersion   = k_forceSmVersion.isDefault() ? (int)computeCapability().version() : k_forceSmVersion.get();
    const bool compileOnly = false;
    const bool useCompileFast         = k_useCompileFast.get();
    const int  maxRegCount            = k_maxRegCount.get() != 0 ? k_maxRegCount.get() : getMaxRegisters();
    const bool useOriBackendOnKepler  = k_useOriBackendOnKepler.get();
    const int  cacheModeL1            = k_cacheModeL1.get();
    const bool ptxasVerbose           = k_ptxasVerbose.get();
    const bool addLineInfo            = k_addLineInfo.get();
    const std::string& additionalArgs = k_ptxasArgs.get();

    // Get the ptxas options as a vector of strings.
    std::vector<std::string> ptxasOptions =
        getPtxasOptions( smVersion, compileOnly, useCompileFast, maxRegCount, k_sassOptimization.get(),
                         addLineInfo ? 1 : 0, useOriBackendOnKepler, cacheModeL1, ptxasVerbose, additionalArgs );

    // If requested via knob, use an external ptxas, otherwise use the JIT.
    lwca::Module module;
    if( !k_offlinePtxas.get().empty() )
    {
        // Get the path to the ptxas binary.
        const std::string& ptxasFile    = k_offlinePtxas.get();
        const std::string& ioFileFolder = k_offlineIOPath.get();
        RT_ASSERT_MSG( !ptxasFile.empty(), "compile.offlinePtxas knob needs to be set." );

        std::string        fileNameSuffix = stringf( "%03d", launchCounter );
        const std::string& lwbinFile =
            corelib::createFromMemoryPtxOffline( ptx, ptxasFile, ptxasOptions, ioFileFolder, fileNameSuffix );
        if( lwbinFile.empty() )
            throw AssertionFailure( RT_EXCEPTION_INFO, "Offline ptxas compilation failed" );
        module = Module::load( lwbinFile.c_str() );
    }
    else
    {
        module = createFromMemoryPtxLoadData( ptx, maxRegCount );
    }

    Metrics::logFloat( "ocg_msec", getDeltaMilliseconds( t0 ) );

    return module;
}

ManagedLWDAModule LWDADevice::moduleFromPTX( const std::string& ptx, int launchCounter )
{
    // Compute the hash
    const size_t hash = corelib::hashString( ptx );

    // See if the PTX is in the cache for this device. If not, compile
    // the module and put it in the cache.
    CachedModulesMap::iterator iter = m_cachedModules.find( hash );
    if( iter == m_cachedModules.end() )
    {
        // We're about to insert a new entry into the cache. Prune the cache
        // first in case it's getting too large.
        pruneModuleCache();

        lwca::Module module = compileModuleFromPTX( ptx, launchCounter );

        ModuleCacheEntry entry;
        entry.hash            = hash;
        entry.module          = ManagedLWDAModule( new lwca::Module( module ) );
        m_cachedModules[hash] = entry;

        return entry.module;
    }

    return iter->second.module;
}

void LWDADevice::pruneModuleCache()
{
    // Attempt to remove modules if we're caching too many of them.
    // We'll only remove modules of which we know that the cache is
    // the only place still holding a reference, i.e. the module is
    // no longer used by any plan / frame task. It'd be functionally
    // correct to just remove whatever we like, because we gave out
    // shared pointers to the modules, but we want to keep the ones
    // that are still in use for tracking / reporting reasons.

    while( (int)m_cachedModules.size() >= k_moduleCacheSize.get() )
    {
        auto it = algorithm::find_if( m_cachedModules, []( const CachedModulesMap::value_type& elem ) {
            return elem.second.module.use_count() == 1;
        } );

        if( it == m_cachedModules.end() )
        {
            break;
        }

        m_cachedModules.erase( it );
    }
}

/*
 * Memory management.  Simply forward to lwmalloc/lwfree for now, but
 * this mechanism can be extended to more carefully manage memory on
 * the device.
 */

LWDADevice::AllocationHandle* LWDADevice::allocate( size_t size )
{
    RT_ASSERT( isEnabled() );
    std::unique_ptr<AllocationHandle> ah( new AllocationHandle );
    makeLwrrent();
    ah->deviceptr = lwca::memAlloc( size );
    ah->size      = size;
    return ah.release();
}


void LWDADevice::detachAllocationHandle( LWDADevice::AllocationHandle* ah )
{
    if( !isEnabled() )
        return;

    makeLwrrent();
    lwca::memFree( ah->deviceptr );
    delete ah;
}

bool LWDADevice::supportsHWBindlessTexture() const
{
    if( k_allowBindless.get() )
        return computeCapability() >= SM( 30 );
    else
        return false;
}

bool LWDADevice::supportsTTU() const
{
    return m_supportsTTU;
}

bool LWDADevice::supportsMotionTTU() const
{
    return ( m_architecture > LW_CFG_ARCHITECTURE_TU100 ) && supportsTTU();
}

bool LWDADevice::supportsLwdaSparseTextures() const
{
    int supportsSparseTextures = 0;
    CALL_LWDA_DRIVER_THROW( lwdaDriver().LwDeviceGetAttribute( &supportsSparseTextures, LW_DEVICE_ATTRIBUTE_SPARSE_LWDA_ARRAY_SUPPORTED,
                                                               m_device.get() ) );
    return static_cast<bool>( supportsSparseTextures );
}

bool LWDADevice::supportsTextureFootprint() const
{
    return computeCapability() >= SM( 75 );
}

bool LWDADevice::canAccessPeer( const Device& peerDev ) const
{
    if( k_p2pAllowed.get() )
    {
        if( const LWDADevice* peerLWDADev = deviceCast<const LWDADevice>( &peerDev ) )
        {
            // Windows requires that the devices be accessible over LWlink and they are
            // capable to do peer to peer over LWCA. This is only possible at this time in TCC mode, not in WDDM mode.
            // Checking with the driver adds this constraint.
            // See OP-2195
            int driver_canAccess = 0;
            CALL_LWDA_DRIVER_THROW(
                lwdaDriver().LwDeviceCanAccessPeer( &driver_canAccess, m_device.get(), peerLWDADev->m_device.get() ) );
            return driver_canAccess == 1;
        }
    }
    return false;
}

bool LWDADevice::canAccessPeerViaLwlink( const Device& peerDev ) const
{
    if( k_p2pAllowed.get() )
    {
        if( const LWDADevice* peerLWDADev = deviceCast<const LWDADevice>( &peerDev ) )
        {
            return LWML::canAccessPeerViaLwLink( m_device.getPCIBusId(), peerLWDADev->m_device.getPCIBusId() );
        }
    }
    return false;
}

bool LWDADevice::enablePeerAccess( const Device& peerDev )
{
    if( const LWDADevice* peerLWDADev = deviceCast<const LWDADevice>( &peerDev ) )
    {
        makeLwrrent();
        try
        {
            m_lwdaContext.enablePeerAccess( peerLWDADev->m_lwdaContext, 0 );
        }
        catch( const LwdaError& e )
        {
            if( e.getErrorCode() == LWDA_ERROR_TOO_MANY_PEERS )
                return false;
            else if( e.getErrorCode() != LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED )
                throw;
        }
        m_peerAccessEnabled.insert( &peerDev );
        if( canAccessPeerViaLwlink( peerDev ) )
            m_lwlinkAccessible.insert( &peerDev );

        return true;
    }
    else
    {
        RT_ASSERT_FAIL_MSG( "enable peer access not on LWCA device." );
    }
}

DeviceSet LWDADevice::peerEnabled() const
{
    return m_peerAccessEnabled;
}

DeviceSet LWDADevice::lwlinkAccessible() const
{
    return m_lwlinkAccessible;
}

void LWDADevice::ilwalidatePeerAccess( const Device& peerDev )
{
    m_peerAccessEnabled.remove( &peerDev );
    m_lwlinkAccessible.remove( &peerDev );
}

void LWDADevice::disablePeerAccess( const Device& peerDev )
{
    if( const LWDADevice* peerLWDADev = deviceCast<const LWDADevice>( &peerDev ) )
    {
        m_lwdaContext.disablePeerAccess( peerLWDADev->m_lwdaContext );
        m_peerAccessEnabled.remove( &peerDev );
    }
    else
    {
        RT_ASSERT_FAIL_MSG( "disable peer access not on LWCA device." );
    }
}

bool LWDADevice::isPeerAccessEnabled( const Device& peerDev ) const
{
    if( deviceCast<const LWDADevice>( &peerDev ) )
    {
        return m_peerAccessEnabled.isSet( &peerDev );
    }
    else
    {
        RT_ASSERT_FAIL_MSG( "disable peer access not on LWCA device." );
    }
}

unsigned int LWDADevice::maximumBoundTextures() const
{
    // There does not appear to be a way to query this
    if( k_maximumBoundTextures.get() != -1 )
        return k_maximumBoundTextures.get();
    else if( computeCapability() >= SM( 30 ) )
        return 256;
    else
        return 128;
}

unsigned int LWDADevice::maximumBindlessTextures() const
{
    // There does not appear to be a way to query this
    if( k_maximumBindlessTextures.get() != -1 )
        return k_maximumBindlessTextures.get();
    else
        return 1024 * 1024;
}

namespace {

// A temporary low level lwContext usable for queries. Using this avoids interference with
// the LWCA runtime and its device initialization flags.
// TODO: This is completely unnecessary.  The code below should just be rewritten to detect errors instead of relying on expeptions.
//
class TemporaryLWcontext
{
  public:
    explicit TemporaryLWcontext( lwca::Device lwdaDevice )
    {
        // For LWCA interop, save current user context.
        m_existing_LWcontext = lwca::Context::getLwrrent();
        m_this_LWcontext     = lwca::Context::create( 0, lwdaDevice );
    }

    ~TemporaryLWcontext()
    {
        if( m_this_LWcontext.isValid() )
        {
            // no throw in destructors
            m_this_LWcontext.destroy();
        }

        if( m_existing_LWcontext.isValid() )
        {
            lwca::Context ctx = lwca::Context::getLwrrent();
            if( ctx.get() != m_existing_LWcontext.get() )
            {
                lerr << "TemporaryLWcontext restore failed." << std::endl;
                m_existing_LWcontext.setLwrrent();
            }
        }
    }

  private:
    lwca::Context m_existing_LWcontext;
    lwca::Context m_this_LWcontext;
};

}  //local namespace

size_t LWDADevice::getAvailableMemory() const
{
    // If the internal lwca context has not been initialized yet,
    // create a temporary context to query available memory then
    // destroy it to avoid mucking with interop
    if( !m_lwdaContext.get() )
    {
        TemporaryLWcontext tempLWctx( lwca::Device( m_device.get() ) );
        size_t             free, total;
        CALL_LWDA_DRIVER_THROW( lwdaDriver().LwMemGetInfo( &free, &total ) );
        return free;
    }
    else
    {
        LWcontext activeLWcontext = nullptr;
        CALL_LWDA_DRIVER_THROW( lwdaDriver().LwCtxGetLwrrent( &activeLWcontext ) );

        if( m_lwdaContext.get() != activeLWcontext )
            CALL_LWDA_DRIVER_THROW( lwdaDriver().LwCtxSetLwrrent( m_lwdaContext.get() ) );

        size_t free, total;
        CALL_LWDA_DRIVER_THROW( lwdaDriver().LwMemGetInfo( &free, &total ) );

        if( m_lwdaContext.get() != activeLWcontext )
            CALL_LWDA_DRIVER_THROW( lwdaDriver().LwCtxSetLwrrent( activeLWcontext ) );
        return free;
    }
}

size_t LWDADevice::getTotalMemorySize() const
{
    return m_device.totalMem();
}

void LWDADevice::getAPIDeviceAttributes( APIDeviceAttributes& attributes ) const
{
    attributes.maxThreadsPerBlock      = m_device.MAX_THREADS_PER_BLOCK();
    attributes.clockRate               = m_device.CLOCK_RATE();
    attributes.multiprocessorCount     = m_device.MULTIPROCESSOR_COUNT();
    attributes.exelwtionTimeoutEnabled = m_device.KERNEL_EXEC_TIMEOUT();
    attributes.maxHardwareTextureCount = supportsHWBindlessTexture() ? maximumBindlessTextures() : maximumBoundTextures();
    attributes.name                    = m_device.getName( nullptr );
    attributes.computeCapability = make_int2( m_device.COMPUTE_CAPABILITY_MAJOR(), m_device.COMPUTE_CAPABILITY_MINOR() );
    attributes.totalMemory       = getTotalMemorySize();
    attributes.tccDriver         = m_device.TCC_DRIVER();
    attributes.lwdaDeviceOrdinal = m_device.get();
    attributes.pciBusId          = m_device.getPCIBusId();
    attributes.compatibleDevices = getCompatibleOrdinals();
    attributes.rtcoreVersion     = supportsMotionTTU() ? 20 : ( supportsTTU() ? 10 : 0 );
}

bool LWDADevice::isCompatibleWith( const Device* otherDevice ) const
{
    // LWCA devices are considered compatible if they have the same SM version
    const LWDADevice* otherLwda = deviceCast<const LWDADevice>( otherDevice );
    if( !otherLwda )
        return false;
    return computeCapability() == otherLwda->computeCapability();
}

void LWDADevice::setPrintBufferSize( size_t bufferSize )
{
    // The value of the knob has priority over the value set using the API.
    if( !k_lwdaPrintfBufferSize.isDefault() )
        bufferSize = k_lwdaPrintfBufferSize.get();
    LWresult errorCode;
    m_lwdaContext.setLimit( LW_LIMIT_PRINTF_FIFO_SIZE, bufferSize, &errorCode );
    if( errorCode == LWDA_ERROR_ILWALID_VALUE )
        throw IlwalidValue( RT_EXCEPTION_INFO,
                            "Print buffer size could not be set. This is likely because a launch has "
                            "already been exelwted for the current context." );
    else if( errorCode != LWDA_SUCCESS )
        throw LwdaError( RT_EXCEPTION_INFO, "setLwdaLimit, LW_LIMIT_PRINTF_FIFO_SIZE", errorCode );
    else
        llog( 10 ) << "LWCA printf buffer size has been changed to " << bufferSize << " Bytes \n";
}

void LWDADevice::setStreamPoolSize( const int newPoolSize )
{
    if( m_streams.size() == 0 || static_cast<int>( m_streams.size() ) != newPoolSize )
    {
        // Create streams on the current LWCA context
        m_lwdaContext.setLwrrent();

        if( static_cast<int>( m_streams.size() ) > newPoolSize )
        {
            // we need to destroy streams
            for( size_t i = static_cast<size_t>( newPoolSize ); i < m_streams.size(); ++i )
            {
                lwca::Stream stream;
                std::swap( stream, m_streams.back() );
                m_streams.pop_back();
                stream.destroy();

                if( m_context->useRtxDataModel() )
                {
                    RtcCommandList cmdlist;
                    std::swap( cmdlist, m_rtcCommandLists.back() );
                    m_rtcCommandLists.pop_back();
                    m_context->getRTCore()->commandListDestroy( cmdlist );
                }
            }
        }
        else
        {
            // we need to add streams
            const size_t oldPoolSize = m_streams.size();

            m_streams.resize( newPoolSize );
            m_rtcCommandLists.resize( newPoolSize );

            for( size_t i = oldPoolSize; i < m_streams.size(); ++i )
            {
                lwca::Stream stream = Stream::create();
                m_streams[i]        = stream;
                if( m_context->useRtxDataModel() )
                    m_context->getRTCore()->commandListCreateForLWDA( m_rtcDeviceContext, stream.get(), &m_rtcCommandLists[i] );
            }
        }
    }
}

// -----------------------------------------------------------------------------
bool LWDADevice::isJITOptionAvailable( LWjit_option option )
{
    // Compile the following dummy kernel with the given option, if no errors are generated then it meas that the option is avaible on the target device.

    std::string dummyPtx =
        ".version 4.1\n"
        ".target sm_30\n"
        ".address_size 64\n"
        ".visible.entry _Z8Kernel_Av()\n"
        "{\n"
        "  ret;\n"
        "}\n";

    LWjit_option options[] = {option};

    void* optiolwalues[] = {
        (void*)1,
    };

    LWresult     err = LWDA_ERROR_UNKNOWN;
    lwca::Module module =
        lwca::Module::loadDataEx( dummyPtx.c_str(), sizeof( options ) / sizeof( LWjit_option ), options, optiolwalues, &err );
    bool isAvailable = err == LWDA_SUCCESS;
    if( isAvailable )
        module.unload();
    return isAvailable;
}

// -----------------------------------------------------------------------------
bool LWDADevice::isJITOptionAvailable( LWjit_option option, int computeCapabilityMajor )
{
    if( static_cast<int>( computeCapability().major() ) == computeCapabilityMajor )
    {
        return isJITOptionAvailable( option );
    }
    return false;
}

// -----------------------------------------------------------------------------
bool LWDADevice::smVersionSupportsLDG( int smVersion )
{
    return smVersion >= 35;
}
