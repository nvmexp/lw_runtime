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

#pragma once

#include <LWCA/ComputeCapability.h>
#include <LWCA/Stream.h>
#include <ExelwtionStrategy/WaitHandle.h>
#include <Util/IndexedVector.h>
#include <Util/UsageReport.h>

#include <corelib/system/Timer.h>

#include <o6/optix.h>
#include <vector_types.h>

#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <stdint.h>  // uint64_t

namespace prodlib {
class Exception;
}

namespace cort {
struct AabbRequest;
}

namespace optix_exp {
class DiskCache;
class DeviceContextLogger;
class EncryptionManager;
}

namespace optix {
class ASManager;
class BindingManager;
class Buffer;
class LWPTIProfiler;
class DeviceManager;
class DeviceSet;
class ErrorManager;
class ExelwtionStrategy;
class FrameTask;
class GlobalScope;
class LLVMManager;
class MemoryManager;
class NullES;
class ObjectManager;
class PagingService;
class Plan;
class PlanManager;
class PrintManager;
class ProfileManager;
class Program;
class ProgramManager;
class RTCore;
class RTXES;
class SBTManager;
class SharedProgramManager;
class StreamBuffer;
class TableManager;
class TelemetryManager;
class ThreadPool;
class UpdateManager;
class ValidationManager;
class Variable;
class WatchdogManager;

class Context
{
  public:
    enum AbiCompatibility
    {
        ABI_16_USE_MULTITHREADED_DEMAND_LOAD_CALLBACKS_BY_DEFAULT = 0,
        ABI_17_USE_MAIN_THREAD_DEMAND_LOAD_CALLBACKS_BY_DEFAULT,
        ABI_18_USE_DEMAND_LOAD_CALLBACK_PER_TILE
    };

    explicit Context( AbiCompatibility abiCompatibility = ABI_18_USE_DEMAND_LOAD_CALLBACK_PER_TILE );
    virtual ~Context();

    // Explicitly do most of the cleanup work. This is a separate function
    // rather than part of the d'tor because it might throw.
    void tearDown();

    // Launching. LaunchInternal is used from command lists. Prepare/FrameTask/Cleanup are used in the ASManager.
    void launchFromAPI( unsigned int entryPointIndex, int dim, RTsize width, RTsize height, RTsize depth );
    void launchFromCommandList( unsigned int        entryPointIndex,
                                int                 dim,
                                RTsize              width,
                                RTsize              height,
                                RTsize              depth,
                                const DeviceSet&    launchDevices,
                                const lwca::Stream& syncStream );
    void preparePlan( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices );
    void launchPrepare( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices );
    void launchFrame( RTsize width, RTsize height, RTsize depth, const DeviceSet& devices, const cort::AabbRequest& aabbRequest );
    std::shared_ptr<WaitHandle> launchFrameAsync( RTsize                   width,
                                                  RTsize                   height,
                                                  RTsize                   depth,
                                                  const DeviceSet&         devices,
                                                  const cort::AabbRequest& aabbRequest,
                                                  const lwca::Stream&      syncStream );
    void launchComplete();

    // Makes an asynchronous launch. Multiple asynchronous launches can overlap with the assumption
    // that nothing changes between them. Any API call that require memory synchronization, etc, must
    // call finishAsyncLaunces to force a sync point.
    void launchAsync( unsigned int entry, int dim, RTsize width, RTsize height, RTsize depth, const DeviceSet& launchDevices, const lwca::Stream& syncStream );

    // Called by a LWCA callback when the launch corresponding to the wait handle has finished.
    // NOTE: no lwca calls must be made by this callback.
    void LWDA_CB launchFinished( std::shared_ptr<WaitHandle>& waitHandle );

    // Finishes all ongoing asynchronous launches.
    void finishAsyncLaunches();

    // Conlwrrent Launches
    unsigned int getMaxConlwrrentLaunches();

    // ASManager may also require memory manager operations even if there is not a full launch
    void lockMemoryManager();
    void unlockMemoryManager();

    bool isRunning() const;
    void finishFrame( bool syncBuffers );
    void launchProgressive( unsigned int max_subframes, unsigned int entryPointIndex, int dim, RTsize width, RTsize height = 1, RTsize depth = 1 );
    void stopProgressiveLaunch( bool hard_stop );

    // Active devices
    void setDevices( size_t num_devices, const int* ordinals );
    void getDevices( std::vector<int>& devices ) const;

    // Context attributes
    void setAttribute( RTcontextattribute attribute, RTsize size, const void* p );
    void getAttribute( RTcontextattribute attribute, RTsize size, void* p ) const;

    // Entry points
    void setEntryPointCount( unsigned int numEntryPoints );
    unsigned int getEntryPointCount() const;

    // Ray types
    void setRayTypeCount( unsigned int numRayTypes );
    unsigned int getRayTypeCount() const;

    // Stack size
    void   setStackSize( size_t );
    size_t getStackSize() const;

    // Maximum callable program call depth
    void         setAPIMaxCallableProgramDepth( unsigned int );
    void         setMaxCallableProgramDepth( unsigned int );
    unsigned int getMaxCallableProgramDepth() const;
    unsigned int getMaxCallableProgramDepthDefault() const;

    // Maximum trace depth
    void         setAPIMaxTraceDepth( unsigned int );
    void         setMaxTraceDepth( unsigned int );
    unsigned int getMaxTraceDepth() const;
    unsigned int getMaxTraceDepthDefault() const;

    // Printing
    void setPrintEnabled( bool enabled );
    bool getPrintEnabled() const;
    void setPrintBufferSize( size_t bufsize );
    size_t getPrintBufferSize() const;
    void setPrintLaunchIndex( int x, int y, int z );
    int3 getPrintLaunchIndex() const;

    // Exceptions
    void setExceptionEnabled( RTexception exception, bool enabled );
    bool getExceptionEnabled( RTexception exception ) const;
    bool     hasAnyExceptionEnabled() const;
    bool     hasOnlyStackOverflowEnabled() const;
    uint64_t getExceptionFlags() const;

    // DiskCache
    bool initializeDiskCache();
    void setDiskCacheLocation( const std::string& path );
    std::string getDiskCacheLocation() const;
    void setDiskCacheMemoryLimits( size_t lowWaterMark, size_t highWaterMark );
    void getDiskCacheMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const;

    // The disk cache uses the Optix 7 logging callback. In order to be able to
    // reroute some logging into the usage report, the Context implements a custom
    // callback.
    void deviceLoggerCallback( unsigned int level, const char* tag, const char* message );
    optix_exp::DeviceContextLogger& getDeviceContextLogger() const;

    float getLastKernelTime();

    // Set the given string as public error string and return a pointer to its public storage
    const char* getPublicString( const std::string& string ) const;

    // Usage report callback
    void setUsageReportCallback( RTusagereportcallback callback, int verbosity, void* cbdata );

    // rtcore
    bool    useRtxDataModel() const;
    RTCore* getRTCore() const;  // get the rtcore API export table wrapper
    bool    RtxUniversalTraversalEnabled() const;
    bool    RtxMotionBlurEnabled() const;
    RTXES*  getRTXExelwtionStrategy() const;

    // Return the entry point index of the AABB kernel.
    unsigned int getAabbEntry() const;

    // Return the global scope.
    GlobalScope* getGlobalScope() const;

    // Returns the number of calls to Context::launch();
    size_t getLaunchCount() const;

    // Return the number of kernels that have been launched. Calls to launch() may
    // result in multiple kernel launches, e.g. the computeAABBs kernel.
    size_t getKernelLaunchCount() const;

    // Return the number of times the Denoiser has been launched.
    size_t getDenoiserLaunchCount() const;
    size_t getSSIMPredictorLaunchCount() const;

    // Increment the Denoiser launch counter.
    void incrDenoiserLaunchCount();
    void incrSSIMPredictorLaunchCount();

    // Return the total amount of time spent in the Denoiser.
    double getDenoiserTimeSpent() const;
    double getSSIMPredictorTimeSpent() const;

    // Add to the time spent in the Denoiser
    void addDenoiserTimeSpent( double timeSpent );
    void addSSIMPredictorTimeSpent( double timeSpent );

    // Save the current nodegraph if the knob is set.
    // where: non-null value augments the log to help identify where this was called from
    // forceWrite: when true, will save the node graph regardless of whether the knob is set (useful from a debugger).
    void saveNodeGraph( const char* where = nullptr, bool forceWrite = false ) const;

    // Lwrrently shutting down the context
    bool shuttingDown() const;

    // Manager getters
    ASManager*                    getASManager() const;
    BindingManager*               getBindingManager() const;
    DeviceManager*                getDeviceManager() const;
    optix_exp::DiskCache*         getDiskCache() const;
    optix_exp::EncryptionManager* getEncryptionManager() const;
    ErrorManager*                 getErrorManager() const;
    LLVMManager*                  getLLVMManager() const;
    MemoryManager*                getMemoryManager() const;
    ObjectManager*                getObjectManager() const;
    PagingService*                getPagingManager() const;
    PlanManager*                  getPlanManager() const;
    ProfileManager*               getProfileManager() const;
    ProgramManager*               getProgramManager() const;
    SBTManager*                   getSBTManager() const;
    SharedProgramManager*         getSharedProgramManager() const;
    TableManager*                 getTableManager() const;
    UpdateManager*                getUpdateManager() const;
    ValidationManager*            getValidationManager() const;
    WatchdogManager*              getWatchdogManager() const;

    ThreadPool* getThreadPool() const;

    // TODO: Move these somewhere else (also used by MegakernelPlan).
    static int getExceptionFlagIndex( RTexception exception );
    static void setExceptionEnabled( uint64_t& flags, RTexception exception, bool enabled );
    static bool getExceptionEnabled( uint64_t flags, RTexception exception );
    static bool hasAnyExceptionEnabled( uint64_t flags );
    static bool hasOnlyStackOverflowEnabled( uint64_t flags );
    // Returns true if any product-specific exception, i.e., any exception besides stack overflow
    // and trace depth, is enabled.
    static bool hasProductSpecificExceptionsEnabled( uint64_t flags );

    // Static functions which allow to set and get the default for the exelwtion strategy
    static void setDefaultExelwtionStrategy( const std::string& defaultExelwtionStrategy );
    static std::string getDefaultExelwtionStrategy();

    // Demand Load Paging
    // State controlled by RT_GLOBAL_INTERNAL_ATTRIBUTE_FORCE_DEMAND_LOAD_WHOLE_MIP_LEVEL, which
    // needs to be set before the Context is created.
    static void setForceDemandLoadWholeMipLevel( bool forced );
    static bool getForceDemandLoadWholeMipLevel();

    // State controlled by RT_GLOBAL_ATTRIBUTE_DEMAND_LOAD_NUM_VIRTUAL_PAGES, which needs to be
    // set before the Context is created.
    static void setDemandLoadNumVirtualPages( unsigned int numPages );
    static unsigned int getDemandLoadNumVirtualPages();

    // State controlled by RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_FORCE_SYNCHRONOUS
    bool getDemandLoadForceSynchronous() const;

    // State controlled by RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_USE_ASYNC_COPIES
    bool getDemandLoadUseAsyncCopies() const;

    // State controlled by RT_CONTEXT_INTERNAL_ATTRIBUTE_DEMAND_LOAD_STAMP_MEMORY_BLOCKS
    bool getDemandLoadStampMemoryBlocks() const;

    bool getPreferFastRecompiles() const { return m_preferFastRecompiles; }
    bool getForceInlineUserFunctions() const { return m_forceInlineUserFunctions; }
    bool getPreferWatertightTraversal() const { return m_preferWatertightTraversal; }

    UsageReport& getUsageReport();

    bool isSetByAPIMaxCallableProgramDepth() const { return m_setByAPIMaxCallableProgramDepth; }
    bool isSetByAPIMaxTraceDepth() const { return m_setByAPIMaxTraceDepth; }

    struct AttributeStackSize
    {
        size_t continuation;
        size_t direct_from_traversal;
        size_t direct_from_state;
    };
    void getAttributeStackSize( AttributeStackSize& attrStackSize ) const;

  private:
    void validateLaunchParams( unsigned int entry, int dim, RTsize width, RTsize height, RTsize depth );

    // Worker for progressive launches, running in its own thread.
    void progressiveLaunchWorker( unsigned int max_subframes, unsigned int entryPointIndex, int dim, RTsize width, RTsize height, RTsize depth );

    // Returns one of the exelwtion strategy members.
    //
    // The entry point number is ununsed right now, but might be needed again
    // when we implement the RTX fallback to MK.
    ExelwtionStrategy* determineExelwtionStrategy( unsigned int entry ) const;

    // Helpers
    std::string attributeCopyErrorString( size_t source_size, RTsize dest_size, const char* attribute ) const;
    void validateAttributeSize( RTsize actualSize, size_t expectedSize, const char* attribute ) const;

    // Returns true if any of the attached rtx traversables requires universal traversal
    bool rtxTraversablesNeedUniversalTraversal() const;

    // Returns true if any of the attached rtx traversables has motion blur
    bool rtxTraversablesHaveMotionBlur() const;

    // Update the context after universal traversal is switched on/off
    void rtxUniversalTraversalDidChange();

    // Update the context after motion blur is switched on/off
    void rtxHasMotionBlurDidChange();

    // Stack sizes for RTX exelwtion mode.
    void setAttributeStackSize( size_t continuationStackSize, size_t directCallableStackSizeFromTraversal, size_t directCallableStackSizeFromState );

    // Managers
    std::unique_ptr<ASManager>                    m_ASManager;
    std::unique_ptr<BindingManager>               m_bindingManager;
    std::unique_ptr<DeviceManager>                m_deviceManager;
    std::unique_ptr<optix_exp::DiskCache>         m_diskCache;
    std::unique_ptr<optix_exp::EncryptionManager> m_encryptionManager;
    std::unique_ptr<ErrorManager>                 m_errorManager;
    std::unique_ptr<LLVMManager>                  m_llvmManager;
    std::unique_ptr<MemoryManager>                m_memoryManager;
    std::unique_ptr<ObjectManager>                m_objectManager;
    std::unique_ptr<PagingService>                m_pagingManager;
    std::unique_ptr<PlanManager>                  m_planManager;
    std::unique_ptr<PrintManager>                 m_printManager;
    std::unique_ptr<ProfileManager>               m_profileManager;
    std::unique_ptr<ProgramManager>               m_programManager;
    std::unique_ptr<SBTManager>                   m_sbtManager;
    std::unique_ptr<SharedProgramManager>         m_sharedProgramManager;
    std::unique_ptr<TableManager>                 m_tableManager;
    std::unique_ptr<TelemetryManager>             m_telemetryManager;
    std::unique_ptr<UpdateManager>                m_updateManager;
    std::unique_ptr<ValidationManager>            m_validationManager;
    std::unique_ptr<WatchdogManager>              m_watchdogManager;
    std::unique_ptr<RTCore>                       m_rtcore;

    std::unique_ptr<optix_exp::DeviceContextLogger> m_deviceContextLogger;

    std::unique_ptr<ThreadPool> m_threadPool;

    // Context state
    GlobalScope*        m_globalScope                          = nullptr;  // Freed in destroyAllObjects
    unsigned int        m_numRayTypes                          = 0;
    unsigned int        m_numEntryPoints                       = 0;
    size_t              m_stackSize                            = 0;  // megakernel only
    size_t              m_continuationStackSize                = 0;  // rtx only
    size_t              m_directCallableStackSizeFromTraversal = 0;  // rtx only
    size_t              m_directCallableStackSizeFromState     = 0;  // rtx only
    unsigned int        m_maxCallableProgramDepth              = 0;
    unsigned int        m_maxTraceDepth                        = 0;
    uint64_t            m_exceptionFlags                       = 0;  // enabled exceptions, stored as bit vector
    size_t              m_launchCount                          = 0;  // count the number of calls to Context::launch()
    size_t              m_kernelLaunchCounter                  = 0;  // count kernel launches
    mutable size_t      m_nodegraphSaveNumber                  = 0;  // Sequence number for nodegraph saves
    mutable std::string m_publicString;                              // last string returned through the public API
    bool                m_shuttingDown = false;                      // Lwrrently shutting down the context
    std::string         m_exelwtionStrategy;
    bool                m_preferFastRecompiles       = false;
    bool                m_forceInlineUserFunctions   = true;
    bool                m_rtxNeedsUniversalTraversal = false;
    bool                m_rtxHasMotionBlur           = false;
    bool                m_preferWatertightTraversal  = true;

    // Telemetry
    size_t m_denoiserLaunchCounter      = 0;
    double m_denoiserTimeSpent          = 0;
    size_t m_ssimPredictorLaunchCounter = 0;
    double m_ssimPredictorTimeSpent     = 0;

    // Valid during launch - from launchPrepare to launchComplete
    Plan*        m_lwrrentPlan           = nullptr;  // Lwrrently active plan
    unsigned int m_lwrrentEntry          = ~0U;      // Current active entry
    int          m_lwrrentDimensionality = 0;        // Current active dimensionality


    // Progressive launch state
    bool                m_progressive_launch_in_progress = false;
    std::future<void>   m_progressive_worker_handle;
    std::atomic<bool>   m_cancel_progressive_flag;
    prodlib::Exception* m_async_exception = nullptr;
    unsigned int        m_subframe_index  = 0;

    // Async state
    typedef IndexedVector<std::shared_ptr<WaitHandle>, WaitHandle::contextIndex_fn> WaitHandleListType;
    std::atomic<bool>  m_asyncLaunchesInProgress;
    WaitHandleListType m_waitHandles;
    std::mutex         m_waitHandles_mutex;
    unsigned int       m_maxConlwrrentLaunches = 4;  // Value from knob will override during construction.

    // Exelwtion strategies
    std::unique_ptr<NullES> m_nullES;
    std::unique_ptr<RTXES>  m_rtxES;

    // Stats
    corelib::timerTick m_startTime               = 0;
    float              m_lastKernelTime          = -1.0f;
    int                m_numCompiles             = 0;
    double             m_compileMsecToFirstTrace = 0.0;

    // Usage report
    UsageReport m_usageReport;

    enum class DataModel
    {
        Unset,
        RTX,
        Canonical
    };
    DataModel m_dataModel = DataModel::Unset;

    static std::string m_defaultExelwtionStrategy;

    bool m_setByAPIMaxCallableProgramDepth = false;
    bool m_setByAPIMaxTraceDepth           = false;
    bool m_demandLoadForceSynchronous      = false;
    bool m_demandLoadUseAsyncCopies        = false;
    bool m_demandLoadStampMemoryBlocks;

    // Demand Load buffers/textures
    AbiCompatibility    m_abiCompatibility;
    static unsigned int s_numVirtualPages;
};

}  // namespace optix
