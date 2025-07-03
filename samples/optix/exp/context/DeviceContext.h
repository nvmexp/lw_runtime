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

#pragma once

#include <optix_types.h>

#include <Context/RTCore.h>
#include <rtcore/interface/types.h>

#include <exp/context/CompiledCache.h>
#include <exp/context/LaunchResourceManager.h>
#include <exp/context/Metrics.h>
#include <exp/context/OpaqueApiObject.h>
#include <exp/context/OptixABI.h>
#include <exp/denoise/Denoise.h>
#include <exp/pipeline/Module.h>
#include <exp/pipeline/Pipeline.h>
#include <exp/pipeline/ProgramGroup.h>

#include <prodlib/misc/lwpti/LwptiProfiler.h>
#include <prodlib/misc/LWTXProfiler.h>

#include <Util/IndexedVector.h>

#include <lwca.h>
#include <atomic>
#include <mutex>
#include <map>

namespace optix_exp {

class DiskCache;
class EncryptionManager;
class ErrorDetails;
class GpuWarmup;
class WatchdogTimer;

class DeviceContextLogger
{
  public:
    DeviceContextLogger();

    // Prevent copies of DeviceContextLogger since the copy would not react to state changes of the source.
    DeviceContextLogger( const DeviceContextLogger& ) = delete;
    DeviceContextLogger& operator=( const DeviceContextLogger& ) = delete;

    OptixResult setCallback( OptixDeviceContextOptions* options, ErrorDetails& errDetails );
    OptixResult setCallback( OptixLogCallback callbackFunction, void* callbackData, unsigned int callbackLevel, ErrorDetails& errDetails );
    OptixResult setCallbackNoCheck( OptixLogCallback callbackFunction, void* callbackData, unsigned int callbackLevel, ErrorDetails& errDetails );

    enum LOG_LEVEL
    {
        Disabled = 0,
        Fatal    = 1,
        Error    = 2,
        Warning  = 3,
        Print    = 4,
        Invalid  = 5
    };

    void callback( LOG_LEVEL level, const char* tag, const char* message );
    void callback( unsigned int level, const char* tag, const char* message );
    void sendError( const std::string& message );
    void sendError( const ErrorDetails& errDetails );

  private:
    OptixLogCallback m_callback;  // Never nullptr (after initialization).
    void*            m_callbackData;
    unsigned int     m_level = LOG_LEVEL::Print;

    std::mutex m_mutex;
};

class LlvmStart
{
  public:
    static LlvmStart& get();  // initializer
    bool              started();

  private:
    LlvmStart();
    bool m_started;
};

class ScopedCommandList
{
  public:
    ScopedCommandList( DeviceContext* context );
    ~ScopedCommandList();

    OptixResult init( LWstream stream, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    RtcCommandList get() { return m_commandList; }
  private:
    DeviceContext* m_context     = nullptr;
    RtcCommandList m_commandList = nullptr;
};

// For some operations it must be ensured that the correct LWCA context is
// active. This class helps keeping track of that by a) checking that the
// context that was passed into optixDeviceContextCreate is the current one
// before and after such an operation or b) setting that LWCA
// context if a is not the case and resetting it to the previously active
// context.
// In general it is expected for the context to stay the same during any
// OptiX function and an error is produced. In some cases this cannot be
// guaranteed (the denoiser is using the LWCA runtime API), so it is possible
// to disable that error checking by passing true for ctxMayChange.
class LwdaContextPushPop
{
  public:
    LwdaContextPushPop( DeviceContext* context, bool allowInternalContextChange = false );
    ~LwdaContextPushPop();
    OptixResult init( ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

  private:
    DeviceContext* m_deviceContext     = nullptr;
    LWcontext      m_previousCtx       = 0;
    bool           m_tolerateCtxChange = false;
};

class DeviceContext : public OpaqueApiObject
{
  public:
    DeviceContext( OptixABI abiVersion, LWcontext lwdaContext, OptixDeviceContextOptions* options );
    ~DeviceContext();


    OptixResult init( ErrorDetails& errDetails );

    OptixResult destroy( ErrorDetails& errDetails );

    // Returns the ABI version.
    //
    // Note that this is the ABI version of the archived function table, which is not necessarily the same ABI version
    // that the user requested. E.g. user requests for ABI versions 18 and 19 and mapped to the function table for ABI
    // ABI version 20.
    OptixABI getAbiVersion() const { return m_abiVersion; }

    // Returns the SM capability for this device (60, 75, etc.).
    int getComputeCapability() const;

    // Returns true iff this device supports TTU ray tracing.
    bool hasTTU() const { return m_hasTTU; };

    // Returns true iff this device supports TTU motion ray tracing.
    bool hasMotionTTU() const { return m_hasMotionTTU; };

#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
    // Returns true iff this device supports TTU displaced micro meshes.
    bool hasDisplacedMicromeshTTU() const { return m_hasDisplacedMicromeshTTU; };
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    // Returns the maximum number of resident threads per multiprocessor
    int getMaxThreadsPerMultiProcessor() const;

    // Returns the maximum number of threads per block
    int getMaxThreadsPerBlock() const;

    // Returns the number of multiprocessors
    int getMultiProcessorCount() const;

    // Returns the chip architecture, see sdk/lwpu/inc/Lwcm.h.
    unsigned int getArchitecture() const { return m_architecture; }

    // Returns the chip architecture implementation, see sdk/lwpu/inc/Lwcm.h.
    unsigned int getArchitectureImplementation() const { return m_architectureImplementation; }

    // Returns the limit for OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH.
    unsigned int getRtcMaxTraceRelwrsionDepth() const { return m_rtcMaxTraceRelwrsionDepth; }

    // Returns the SBT header size.
    unsigned int getSbtHeaderSize() const { return m_sbtHeaderSize; }

    // Returns the device property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH.
    unsigned int getMaxSceneGraphDepth() const { return m_maxTraversalGraphDepth; }

    // Returns the device property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS.
    unsigned int getMaxPrimsPerGAS() const { return s_maxPrimsPerGAS; }

    // Returns the device property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS.
    unsigned int getMaxSbtRecordsPerGAS() const { return m_maxSbtRecordsPerGAS; }

    // Returns the device property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS
    // fix this number for ABI 20 (pre fat instances)
    unsigned int getMaxInstancesPerIAS() const { return m_maxInstancesPerIAS; }

    // Returns the device property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
    unsigned int getMaxInstanceId() const { return m_maxInstanceId; }

    // Returns the device property OPTIX_DEVICE_PROPERTY_LIMIT_SBT_OFFSET.
    unsigned int getMaxSbtOffset() const { return m_maxSbtOffset; }

    // Returns the device property OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK.
    unsigned int getNumBitsInstanceVisibilityMask() const { return s_numBitsInstanceVisibilityMask; }

    // Returns the RTCore device property RTC_LIMIT_MAX_CALLABLE_PARAM_REGISTERS.
    int getCallableParamRegCount() const;

    OptixResult registerModule( Module* module, ErrorDetails& errDetails );
    OptixResult unregisterModule( Module* module, ErrorDetails& errDetails );
    OptixResult destroyModules( ErrorDetails& errDetails );

    OptixResult registerPipeline( Pipeline* pipeline, ErrorDetails& errDetails );
    OptixResult unregisterPipeline( Pipeline* pipeline, ErrorDetails& errDetails );
    OptixResult destroyPipelines( ErrorDetails& errDetails );

    OptixResult registerDenoiser( Denoiser* denoiser, ErrorDetails& errDetails );
    OptixResult unregisterDenoiser( Denoiser* denoiser, ErrorDetails& errDetails );
    OptixResult destroyDenoisers( ErrorDetails& errDetails );

    OptixResult registerProgramGroup( ProgramGroup* programGroup, ErrorDetails& errDetails );
    OptixResult unregisterProgramGroup( ProgramGroup* programGroup, ErrorDetails& errDetails );
    OptixResult destroyProgramGroups( ErrorDetails& errDetails );

    DeviceContextLogger& getLogger() { return m_logger; }

    LaunchResourceManager& getLaunchResources() { return m_launchResources; }

    EncryptionManager& getEncryptionManager() { return *m_encryptionManager.get(); }

    EncryptionManager& getInternalEncryptionManager() { return *m_internalEncryptionManager.get(); }

    WatchdogTimer& getTTUWatchdog() { return *m_ttuWatchdog.get(); }

    GpuWarmup& getGpuWarmup() { return *m_gpuWarmup.get(); }

    optix::RTCoreAPI& getRtcore() { return m_rtcore; }

    RtcDeviceContext getRtcDeviceContext() const { return m_rtcDeviceContext; }

    LWcontext getLwdaContext() const { return m_lwdaContext; }

    CompiledCache& getRtcCompiledModuleCache() { return m_compiledCache; }

    // TODO Refactor device context creation to get rid of these setters.
    //@{

    // Sets the SBT header size.
    void setSbtHeaderSize( unsigned int sbtHeaderSize ) { m_sbtHeaderSize = sbtHeaderSize; }

    // Sets the rtcore device context.
    void setRtcDeviceContext( RtcDeviceContext rtcDeviceContext ) { m_rtcDeviceContext = rtcDeviceContext; }

    // Sets the lwca device properties.
    OptixResult setLwdaDeviceProperties( ErrorDetails& errDetails );

    // Gets the rtcore device properties.
    OptixResult getRtcDeviceProperties( ErrorDetails& errDetails );

    //@}

    std::unique_ptr<DiskCache>& getDiskCache() { return m_diskCache; }
    OptixResult initializeDiskCache( ErrorDetails& errDetails );
    OptixResult setDiskCacheLocation( const std::string& location, ErrorDetails& errDetails );
    OptixResult setDiskCacheMemoryLimits( size_t lowWaterMark, size_t highWaterMark, ErrorDetails& errDetails );
    OptixResult setDiskCacheEnabled( bool enabled, ErrorDetails& errDetails );
    bool        isDiskCacheActive() const;
    std::string getDiskCacheLocation() const;
    void getDiskCacheMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const;

    const Rtlw32* getRtcoreUUID() const;

    OptixResult setNoInlineEnabled( bool enabled, ErrorDetails& errDetails );
    bool isNoInlineEnabled() const;
    void makeNoInlineImmutable();

    OptixResult setD2IREnabled( bool enabled, ErrorDetails& errDetails );
    bool isD2IREnabled() const;

    OptixResult setLWPTXFallbackEnabled( bool enabled, ErrorDetails& errDetails );
    bool isLWPTXFallbackEnabled() const;

    OptixResult setSplitModuleMinBinSize( unsigned int minBinSize, ErrorDetails& errDetails );
    unsigned int getSplitModuleMinBinSize() const;

    OptixDeviceContextValidationMode getValidationMode() const;
    bool                             hasValidationModeDebugExceptions() const;
    bool                             hasValidationModeSpecializationConsistency() const;
    bool                             hasValidationModeCheckStreamState() const;
    bool                             hasValidationModeCheckLwrrentLwdaContext() const;

    prodlib::LwptiProfiler& getLwptiProfiler() { return m_lwptiProfiler; }
    bool                    isLwptiProfilingEnabled() const { return m_lwptiProfiler.isInitialized(); }

    unsigned int registerTraversable( LWdeviceptr buffer );

    void startAsyncTimer( const std::string& name, lwdaStream_t stream = 0 );

    void stopAsyncTimerAndRecordMetric( const std::string& name, lwdaStream_t stream = 0 );

    OptixResult getLibraryPath( std::string& libPath, ErrorDetails& errDetails ) const;

  private:
	static std::atomic<unsigned int> s_serialNumber;
    static unsigned int getSerialNumber() { return s_serialNumber++; }

    DeviceContextLogger m_logger;

    LaunchResourceManager m_launchResources;

    std::unique_ptr<DiskCache> m_diskCache;
    // If disk cache initialization at context creation time
    // fails with the default values, the user can reconfigure it
    // and activate it again. To do that, the context needs to
    // hold on to the disk cache configuration
    std::string        m_diskCacheLocation;
    size_t             m_diskCacheLowWaterMark;
    size_t             m_diskCacheHighWaterMark;
    mutable std::mutex m_diskCacheConfigMutex;

    std::unique_ptr<EncryptionManager> m_encryptionManager;
    std::unique_ptr<EncryptionManager> m_internalEncryptionManager;

    std::unique_ptr<WatchdogTimer> m_ttuWatchdog;
    std::unique_ptr<GpuWarmup>     m_gpuWarmup;

    optix::RTCoreAPI m_rtcore;
    RtcDeviceContext m_rtcDeviceContext = 0;
    LWcontext        m_lwdaContext      = 0;

    prodlib::LwptiProfiler m_lwptiProfiler;

    Rtlw32 m_rtcoreUUID[4];

    OptixABI m_abiVersion = OptixABI::ABI_MIN;

    // device attributes
    int m_lwComputeCapabilityMajor      = 0;
    int m_lwComputeCapabilityMinor      = 0;
    int m_lwMaxThreadsPerMultiProcessor = 0;
    int m_lwMaxThreadsPerBlock          = 0;
    int m_lwMultiProcessorCount         = 0;
    int m_callableParamRegCount         = 0;
    int m_maxTraversalGraphDepth        = 0;

    unsigned int m_architecture               = 0;
    unsigned int m_architectureImplementation = 0;

    bool m_hasTTU       = false;
    bool m_hasMotionTTU = false;
#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
    bool m_hasDisplacedMicromeshTTU = false;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    unsigned int m_rtcMaxTraceRelwrsionDepth = 0;
    unsigned int m_sbtHeaderSize             = 0;

    // TODO: Query from rtcore and verify, Types.hpp, bug: 2628943
    static constexpr unsigned int s_maxPrimsPerGAS                = 1 << 29;
    static constexpr unsigned int s_numBitsInstanceVisibilityMask = 8;
    unsigned int                  m_maxSbtRecordsPerGAS;
    unsigned int                  m_maxInstancesPerIAS;
    unsigned int                  m_maxInstanceId;
    unsigned int                  m_maxSbtOffset;

    bool m_enableNoInline;
    // once locked, m_enableNoInline is immutable
    // set this as soon as the first module gets compiled or pipeline linked
    std::mutex m_enableNoInlineMutex;
    bool       m_enableNoInlineImmutable = false;

    std::atomic<bool> m_enableD2IR;
    bool m_canSetD2IRViaAPI = true; // Setting m_enableD2IR from the API is ignored if set via knob or environment var.
    OptixResult initD2IREnabled( ErrorDetails& errDetails );

    std::atomic<bool> m_enableLWPTXFallback;

    std::atomic<unsigned int> m_splitModuleMinBinSize;

    using ModuleListType = optix::IndexedVector<Module*, Module::DeviceContextIndex_fn>;
    ModuleListType m_modules;
    std::mutex     m_modulesMutex;

    using PipelineListType = optix::IndexedVector<Pipeline*, Pipeline::DeviceContextIndex_fn>;
    PipelineListType m_pipelines;
    std::mutex       m_pipelinesMutex;

    using DenoiserListType = optix::IndexedVector<Denoiser*, Denoiser::DeviceContextIndex_fn>;
    DenoiserListType m_denoisers;
    std::mutex       m_denoisersMutex;

    using ProgramGroupListType = optix::IndexedVector<ProgramGroup*, ProgramGroup::DeviceContextIndex_fn>;
    ProgramGroupListType m_programGroups;
    std::mutex           m_programGroupsMutex;

    CompiledCache m_compiledCache;

    // Lock that protects llvm initialization.
    static std::mutex s_mutex;

    // Get the human-friendly name of this context's LWCA device.
    OptixResult getDeviceName( std::string& outDeviceName, ErrorDetails& errDetails );

    // Get this PCI bus ID of this context's LWCA device.
    OptixResult getPCIBusID( std::string& outPCIBusID, ErrorDetails& errDetails );

    // Profiler used to emit LWTX ranges.
    std::unique_ptr<LWTXProfiler> m_lwtxProfiler;

    OptixDeviceContextValidationMode m_validationMode;

    // traversable register
    size_t                       m_registeredTraversableVersion = 0;
    std::map<LWdeviceptr,size_t> m_registeredTraversableMap;
    std::mutex                   m_registeredTraversableMutex;
    std::unique_ptr<Metrics>     m_metrics;

  public:
    LWTXProfiler* getLWTXProfiler() { return m_lwtxProfiler.get(); }
    Metrics* getMetrics() { return m_metrics.get(); }
};

inline OptixResult implCast( OptixDeviceContext deviceContextAPI, DeviceContext*& deviceContext )
{
    deviceContext = reinterpret_cast<DeviceContext*>( deviceContextAPI );
    // It's OK for deviceContextAPI to be nullptr
    if( deviceContext && deviceContext->m_apiType != OpaqueApiObject::ApiType::DeviceContext )
    {
        return OPTIX_ERROR_ILWALID_DEVICE_CONTEXT;
    }
    return OPTIX_SUCCESS;
}

inline OptixDeviceContext apiCast( DeviceContext* deviceContext )
{
    return reinterpret_cast<OptixDeviceContext>( deviceContext );
}

OptixResult optixDeviceContextCreate_lwrrent( LWcontext fromContext, OptixDeviceContextOptions* options, OptixDeviceContext* contextAPI );

}  // end namespace optix_exp
