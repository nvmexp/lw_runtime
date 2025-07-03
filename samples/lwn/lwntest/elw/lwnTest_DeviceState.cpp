/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_c.h"
#include "lwnTest/lwnTest_DeviceState.h"
#include "lwnExt/lwnExt_Internal.h"

#include "cmdline.h"
#include "lwn_utils.h"

namespace lwnTest {

#define LWN_UTIL_DEBUG_OUTPUT     0
static void log_output(const char *fmt, ...)
{
#if LWN_UTIL_DEBUG_OUTPUT
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif
}

// Initialize the caps for a given LWN device.
void DeviceCaps::init(LWNdevice *device)
{
    memset(this, 0, sizeof(DeviceCaps));
    if (lwogCheckLWNAPIVersion(26, 3)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_MIN_MAX_FILTERING, &supportsMinMaxReduction);
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_STENCIL8_FORMAT, &supportsStencil8);
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_ASTC_FORMATS, &supportsASTC);
    }
    if (lwogCheckLWNAPIVersion(38, 5)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_CONSERVATIVE_RASTER, &supportsConservativeRaster);
    }
    if (lwogCheckLWNAPIVersion(40, 2)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_ZERO_FROM_UNMAPPED_VIRTUAL_POOL_PAGES,
                            &supportsZeroFromUndefinedMappedPoolPages);
    }
    if (lwogCheckLWNAPIVersion(40, 8)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_PASSTHROUGH_GEOMETRY_SHADERS,
                            &supportsPassthroughGeometryShaders);
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_VIEWPORT_SWIZZLE,
                            &supportsViewportSwizzle);
    }
    if (lwogCheckLWNAPIVersion(40, 10)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_SPARSE_TILED_PACKAGED_TEXTURES,
                            &supportsMaxwellSparsePackagedTextures);
    }
    if (lwogCheckLWNAPIVersion(40, 15)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_ADVANCED_BLEND_MODES,
                            &supportsAdvancedBlendModes);
    }
    if (lwogCheckLWNAPIVersion(40, 18)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_DRAW_TEXTURE,
                            &supportsDrawTexture);
    }
    if (lwogCheckLWNAPIVersion(41, 4)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_TARGET_INDEPENDENT_RASTERIZATION,
                            &supportsTargetIndependentRasterization);
    }
    if (lwogCheckLWNAPIVersion(41, 4)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_FRAGMENT_COVERAGE_TO_COLOR,
                            &supportsFragmentCoverageToColor);
    }
    if (lwogCheckLWNAPIVersion(41, 4)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_POST_DEPTH_COVERAGE,
                            &supportsPostDepthCoverage);
    }
    if (lwogCheckLWNAPIVersion(48, 2)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_IMAGES_USING_TEXTURE_HANDLES,
                            &supportsImagesUsingTextureHandles);
    }
    if (lwogCheckLWNAPIVersion(49, 0)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_SAMPLE_LOCATIONS,
                            &supportsSampleLocations);
    }
    if (lwogCheckLWNAPIVersion(52, 13)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_FRAGMENT_SHADER_INTERLOCK,
                            &supportsFragmentShaderInterlock);
    }
    if (lwogCheckLWNAPIVersion(52, 26)) {
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SUPPORTS_DEBUG_LAYER,
                            &supportsDebugLayer);
    }
    if (lwogCheckLWNAPIVersion(52, 313)) {
        LWNint subgroupSize = 0;
        lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SHADER_SUBGROUP_SIZE, &subgroupSize);

        if (subgroupSize > 0) {
            supportsShaderSubgroup = 1;
        }
    }

    // We have a generic supportsMaxwell2Features cap that can be used for tests but is not directly 
    // exposed in the LWN API. Use fragment shader interlock as a proxy for detecting Maxwell2 or better.
    supportsMaxwell2Features = supportsFragmentShaderInterlock;

    // Query additional properties of the implementation through an API
    // extension (if available).
    supportsShadingRateImage = 0;
    supportsScissorExclusive = 0;
}

void DeviceState::init(LWNdeviceFlagBits   deviceFlags       /* = LWNdeviceFlagBits(0) */,
                       LWNwindowOriginMode windowOriginMode  /* = LWN_WINDOW_ORIGIN_MODE_LOWER_LEFT */,
                       LWNdepthMode        depthMode         /* = LWN_DEPTH_MODE_NEAR_IS_MINUS_W */,
                       LWNqueueFlags       queueFlags)
{
    // Complain if a DeviceState object is created while object tracking
    // is enabled.  Using such a device isn't safe, because unfreed tracked
    // objects created with a temporary device will only be freed at the
    // end of the test where the temporary device has already been freed.
    assert(!IsLWNObjectTrackingEnabled());

    m_initBits = 0;
    m_queueMemory = NULL;
    m_texIDPool = NULL;
    m_completionTracker = NULL;
    m_glslcHelper = NULL;

    // Check the API version.  A major version mismatch between the driver and
    // the test indicates that some API changed in a backwards-incompatible
    // manner.  All tests should be disabled in this case, since they could
    // easily crash otherwise.  Allow device creation to succeed if the driver
    // reports a lower minor version; tests can run successfully as long as
    // isSupported() is properly coded to ensure that nothing needing an
    // unsupported minor version is used.
    lwnDeviceGetInteger(NULL, LWN_DEVICE_INFO_API_MAJOR_VERSION, &m_lwnMajorVersion);
    lwnDeviceGetInteger(NULL, LWN_DEVICE_INFO_API_MINOR_VERSION, &m_lwnMinorVersion);
    if (m_lwnMajorVersion != LWN_API_MAJOR_VERSION) {
        printf("LWN API version mismatch:\n"
            "Reported version:  %d.%d\n"
            "Expected version:  %d.%d\n",
            m_lwnMajorVersion, m_lwnMinorVersion,
            LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION);
        return;
    }

    // Store the API version into the globals, since the API version check
    // function immediately below uses them.
    g_lwnMajorVersion = m_lwnMajorVersion;
    g_lwnMinorVersion = m_lwnMinorVersion;

    // If "-lwnDebug" is specified in the command line, turn on the debug bits
    // for all contexts.  Don't update <m_deviceFlags> so we can tell if this
    // context was "supposed" to be a debug context or not.
    m_requestedDebug = (0 != (deviceFlags & (LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_0_BIT |
                                             LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_1_BIT |
                                             LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT |
                                             LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_3_BIT |
                                             LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT)));

    // Starting with 52.26 the presence of the debug layer needs to be queried. For prior
    // versions the debug layer is present by default.
    int supportsDebugLayer = 1;

    if (lwogCheckLWNAPIVersion(52, 26)) {
        lwnDeviceGetInteger(NULL, LWN_DEVICE_INFO_SUPPORTS_DEBUG_LAYER, &supportsDebugLayer);
    }

    if (!supportsDebugLayer) {
        if (lwnDebugEnabled) {
            printf("WARNING: LWN debug layer forced off as driver does not support it.\n");
        }
        m_requestedDebug = false;
        lwnDebugEnabled = 0;
    }
    if (lwnDebugEnabled) {
        deviceFlags = LWNdeviceFlagBits(deviceFlags | LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT);
        switch (lwnDebugLevel) {
        default:
        case 0:
            deviceFlags = LWNdeviceFlagBits(deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_0_BIT);
            break;
        case 1:
            deviceFlags = LWNdeviceFlagBits(deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_1_BIT);
            break;
        case 2:
            deviceFlags = LWNdeviceFlagBits(deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT);
            break;
        case 3:
            deviceFlags = LWNdeviceFlagBits(deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_3_BIT);
            break;
        case 4:
            deviceFlags = LWNdeviceFlagBits(deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT);
            break;
        }

    }

    // Try to create and initialize a device object.
    LWNdeviceBuilder deviceBuilder;
    lwnDeviceBuilderSetDefaults(&deviceBuilder);
    lwnDeviceBuilderSetFlags(&deviceBuilder, deviceFlags);
    m_deviceFlags = deviceFlags;
    if (!lwnDeviceInitialize(&m_device, &deviceBuilder)) {
        printf("Failed to initialize LWN device structure.\n");
        return;
    }
    m_initBits |= INITIALIZED_DEVICE;

    // Reset the function pointer interface to match the type of the device we
    // just created.  If we are enabling the debug layer, install our debug
    // callback and verify that the debug layer is working properly.
    ReloadLWNEntryPoints(&m_device, m_requestedDebug);
    if (lwnDebugEnabled) {
        lwnDeviceInstallDebugCallback(&m_device, LWNUtilsDebugCallback, NULL, LWN_TRUE);
        m_initBits |= INITIALIZED_DEBUG_CALLBACK;

        // Disable global debug callback since lwntest was started with -lwndebug
        // and will install a regular device debug callbacks.
        lwnInstallGlobalDebugCallback(NULL, NULL);

        // Sanity check the LWN debug layer to be sure it is properly
        // installed.
        if (!SanityCheckLWNDebug(&m_device)) {
            printf("LWN debug layer support for '-lwnDebug' not working properly.\n");
            return;
        }
    }

    // Set the window origin and depth modes for the device based on
    // parameters provided to the constructor.  We need to do this before
    // creating queues or command buffers to ensure they inherit the proper
    // state.
    lwnDeviceSetWindowOriginMode(&m_device, windowOriginMode);
    lwnDeviceSetDepthMode(&m_device, depthMode);

    // Check the version matches up with the GLSLC interface
    // We first check the GLSLC header compiled into lwntest matches the API version reported by GLSLC.
    // Then we check to make sure the GPU code section version the GLSLC library can produce falls within the
    // range of supported versions reported by the LWN driver.
    if (g_glslcLibraryHelper->IsLoaded()) {
        if (lwogCheckLWNAPIVersion(38, 1)) {
            int versionCheck = 1;

            GLSLCversion dllVersion = g_glslcLibraryHelper->GetVersion();

            // Use temporaries to avoid compiler complaining about unsigned comparisons such as "dllVersion.apiMinor < 0"
            LWNuint requiredVersionMinor = GLSLC_API_VERSION_MINOR;
            LWNuint requiredVersionMajor = GLSLC_API_VERSION_MAJOR;
            if (!lwnUtil::GLSLCLibraryHelper::GLSLCCheckAPIVersion(
                        requiredVersionMajor, requiredVersionMinor, dllVersion.apiMajor, dllVersion.apiMinor)) {
                printf("GLSLC DLL reported API version (%d.%d) which is not compatible with the version compiled into lwntest (%d.%d)\n",
                dllVersion.apiMajor, dllVersion.apiMinor, GLSLC_API_VERSION_MAJOR, GLSLC_API_VERSION_MINOR);
                versionCheck = 0;
            }

            if (versionCheck) {
                // Need to query LWN's device for the GLSLC gpu code section versions it supports, then
                // match up against the version reported by the GLSLC DLL
                LWNint glslcMaxBilwersionMajor = 0;
                LWNint glslcMinBilwersionMajor = 0;
                LWNint glslcMaxBilwersionMinor = 0;
                LWNint glslcMinBilwersionMinor = 0;
                lwnDeviceGetInteger(&m_device, LWN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MAJOR_VERSION, &glslcMaxBilwersionMajor);
                lwnDeviceGetInteger(&m_device, LWN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MAJOR_VERSION, &glslcMinBilwersionMajor);
                lwnDeviceGetInteger(&m_device, LWN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MINOR_VERSION, &glslcMaxBilwersionMinor);
                lwnDeviceGetInteger(&m_device, LWN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MINOR_VERSION, &glslcMinBilwersionMinor);

                if (!lwnUtil::GLSLCLibraryHelper::GLSLCCheckGPUCodeVersion(glslcMaxBilwersionMajor, glslcMinBilwersionMajor,
                                                                        glslcMaxBilwersionMinor, glslcMinBilwersionMinor,
                                                                        dllVersion.gpuCodeVersionMajor, dllVersion.gpuCodeVersionMinor)) {
                    printf("GLSLC binary version not compatible with this version of LWN:\n"
                            "LWN reports being able to use GLSLC binary version %d.%d through %d.%d.\n"
                            "GLSLC reported being able to produce a binary version %d.%d.\n",
                            glslcMinBilwersionMajor, glslcMinBilwersionMinor,
                            glslcMaxBilwersionMajor, glslcMaxBilwersionMinor,
                            dllVersion.gpuCodeVersionMajor, dllVersion.gpuCodeVersionMinor);
                    versionCheck = 0;
                }
            }

            if (!versionCheck) {
                printf("Failed GLSLC version check.  Aborting.\n");
                return;
            }
        } else {
            printf("LWN version too old to use with this version of GLSLC.\n");
            return;
        }
    }

    // Query other device caps, as long as the implemented API version supports them.
    m_caps.init(&m_device);

    // Create a primary queue for our tests to use.  If that fails, we're
    // also out of luck.
    LWNqueueBuilder qb;
    lwnQueueBuilderSetDevice(&qb, &m_device);
    lwnQueueBuilderSetDefaults(&qb);

    // If the queue builder supports application-specified memory sizes,
    // program the queue builder to use them.
    if (lwogCheckLWNAPIVersion(52, 11)) {
        lwnQueueBuilderSetCommandMemorySize(&qb, queueCommandMemKB * 1024);
        lwnQueueBuilderSetComputeMemorySize(&qb, queueComputeMemKB * 1024);
        size_t memorySize = lwnQueueBuilderGetQueueMemorySize(&qb);
        m_queueMemory = AlignedStorageAlloc(memorySize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);
        if (!m_queueMemory) {
            printf("Failed to allocate per-queue memory.\n");
            return;
        }
        lwnQueueBuilderSetQueueMemory(&qb, m_queueMemory, memorySize);
    }

    // If the queue builder supports application-specified flush thresholds,
    // program the queue builder to use them.
    if (lwogCheckLWNAPIVersion(52, 12) && queueFlushThresholdKB) {
        int flushThreshold;
        lwnDeviceGetInteger(&m_device, LWN_DEVICE_INFO_QUEUE_COMMAND_MEMORY_MIN_FLUSH_THRESHOLD, &flushThreshold);
        flushThreshold = std::max(flushThreshold, queueFlushThresholdKB * 1024);
        lwnQueueBuilderSetCommandFlushThreshold(&qb, flushThreshold);
    }

    uint32_t qflags = uint32_t(queueFlags);
    if (noZlwll) {
        qflags |= uint32_t(LWN_QUEUE_FLAGS_NO_ZLWLL_BIT);
    }
    lwnQueueBuilderSetFlags(&qb, LWNqueueFlags(qflags));


    if (!lwnQueueInitialize(&m_queue, &qb)) {
        printf("Failed to initialize LWN queue.\n");
        return;
    }
    m_initBits |= INITIALIZED_QUEUE;

#if defined(LW_HOS)
    // Increase timeout so all subtests have anough time to finish
    PFNLWNQUEUESETTIMEOUTLWX lwnQueueSetTimeoutLWX =
        (PFNLWNQUEUESETTIMEOUTLWX)lwnDeviceGetProcAddress(&m_device, "lwnQueueSetTimeoutLWX");
    if (!lwnQueueSetTimeoutLWX) {
        printf("Failed to query function pointer for lwnQueueSetTimeoutLWX");
        return;
    }
    lwnQueueSetTimeoutLWX(&m_queue, 5000000000ULL);
#endif

    // Initialize a queue completion tracker that can track up to 31
    // outstanding fences.
    m_completionTracker = new lwnUtil::CompletionTracker(&m_device, 32);
    if (!m_completionTracker) {
        printf("Failed to allocate LWN completion tracker.\n");
        return;
    }
    m_initBits |= INITIALIZED_COMPLETION_TRACKER;

    // Initialize a QueueCommandBuffer object to submit commands to the queue.
    if (!m_queueCB.init(&m_device, &m_queue, m_completionTracker)) {
        printf("Failed to initialize LWN queue command buffer.\n");
        return;
    }
    m_initBits |= INITIALIZED_QUEUECB;

    // Initialize command buffer memory manager to provide memory for API
    // command buffer usage.
    if (!m_cmdMemManager.init(&m_device, m_completionTracker)) {
        printf("Failed to allocate lwntest command memory manager.\n");
        return;
    }
    m_initBits |= INITIALIZED_CMDBUF_MANAGER;

    // Initialize our texture ID pool utility class and bind the texture and
    // sampler pool memory for rendering.
    m_texIDPool = new lwnUtil::TexIDPool(&m_device);
    if (!m_texIDPool) {
        printf("Failed to allocate the LWN texture ID pool.\n");
        return;
    }
    m_texIDPool->Bind(&m_queueCB);
    m_initBits |= INITIALIZED_TEXIDPOOL;

    m_glslcHelper = new lwnTest::GLSLCHelper(&m_device, DeviceProgramPoolSize, g_glslcLibraryHelper,
        g_glslcHelperCache, g_dxcLibraryHelper);
    if (!m_glslcHelper) {
        printf("Failed to initialize GLSLC compilation helper structure.\n");
        return;
    }
    m_initBits |= INITIALIZED_GLSLC_HELPER;

    LWNmemoryPoolBuilder poolBuilder;
    lwnMemoryPoolBuilderSetDefaults(&poolBuilder);
    lwnMemoryPoolBuilderSetDevice(&poolBuilder, &m_device);
    lwnMemoryPoolBuilderSetFlags(&poolBuilder, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    m_shaderScratchPoolMemory = PoolStorageAlloc(DEFAULT_SHADER_SCRATCH_MEMORY_SIZE);
    lwnMemoryPoolBuilderSetStorage(&poolBuilder, m_shaderScratchPoolMemory, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE);
    if (!lwnMemoryPoolInitialize(&m_shaderScratchPool, &poolBuilder)) {
        printf("Failed to initialize the default global shader scratch memory pool.\n");
        return;
    }
    m_initBits |= INITIALIZED_SCRATCH_POOL;

    if (!m_glslcHelper->SetShaderScratchMemory(&m_shaderScratchPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, &m_queueCB)) {
        printf("Failed to use the default global shader scratch memory pool.\n");
        return;
    }

    lwnDeviceSetIntermediateShaderCache(&m_device, lwnGlasmCacheNumEntries);

    m_initBits |= INITIALIZED_ALL;
}

DeviceState::~DeviceState(void)
{
    // This device should be active when tearing down to make sure debug hooks
    // are still installed if and only if this is a debug device.
    assert(this == DeviceState::GetActive());

    // Make sure all previous rendering commands are done before we start
    // tearing down resources.
    if (isValid()) {
        lwnQueueFinish(&m_queue);
    }

    // Unregister our "-lwndebug" callback if installed.
    if (m_initBits & INITIALIZED_DEBUG_CALLBACK) {
        lwnDeviceInstallDebugCallback(&m_device, LWNUtilsDebugCallback, NULL, LWN_FALSE);
    }

    if (m_initBits & INITIALIZED_CMDBUF_MANAGER) {
        m_cmdMemManager.destroy();
    }

    if (m_initBits & INITIALIZED_QUEUECB) {
        m_queueCB.destroy();
    }

    if (m_initBits & INITIALIZED_COMPLETION_TRACKER) {
        delete m_completionTracker;
    }

    if (m_initBits & INITIALIZED_QUEUE) {
        lwnQueueFinalize(&m_queue);
    }
    AlignedStorageFree(m_queueMemory);

    if (m_initBits & INITIALIZED_TEXIDPOOL) {
        delete m_texIDPool;
    }

    if (m_initBits & INITIALIZED_GLSLC_HELPER) {
        delete m_glslcHelper;
    }

    if (m_initBits & INITIALIZED_SCRATCH_POOL) {
        lwnMemoryPoolFinalize(&m_shaderScratchPool);
        PoolStorageFree(m_shaderScratchPoolMemory);
    }

    if (m_initBits & INITIALIZED_DEVICE) {
        lwnDeviceFinalize(&m_device);
    }
}

DeviceState * DeviceState::g_defaultDevice = NULL;

void DeviceState::SetDefault()
{
    g_defaultDevice = this;
}

DeviceState * DeviceState::GetDefault()
{
    return g_defaultDevice;
}

DeviceState * DeviceState::g_activeDevice = NULL;

void DeviceState::SetActive()
{
    // Update the active DeviceState object.
    g_activeDevice = this;

    // Update various lwntest globals to indicate that all "sub-resources"
    // belonging to the device state are current.  We should phase out the use
    // of all of these globals.
    g_lwnDevice = &m_device;
    g_lwnQueue = &m_queue;
    g_lwnTexIDPool = m_texIDPool;
    g_lwnTracker = m_completionTracker;
    g_glslcHelper = m_glslcHelper;
    g_lwnScratchMemPool = &m_shaderScratchPool;
    g_lwnQueueCB = &m_queueCB;
    g_lwnCommandMem = m_cmdMemManager;

    // Update the global device caps and version to match the device state.
    g_lwnDeviceCaps = m_caps;
    g_lwnMajorVersion = m_lwnMajorVersion;
    g_lwnMinorVersion = m_lwnMinorVersion;

    // Reload the function pointer interface to be appropriate for the active
    // device.
    ReloadLWNEntryPoints(&m_device, m_requestedDebug);
}

void DeviceState::SetDefaultActive()
{
    GetDefault()->SetActive();
}

DeviceState * DeviceState::GetActive()
{
    return g_activeDevice;
}

void DeviceState::destroyGLSLCHelper()
{
    // Our default ExitGraphics() function wants to tear down (and later
    // rebuild) the GLSLC helper after each test without tearing down the
    // device itself.
    if (m_glslcHelper) {
        delete m_glslcHelper;
        m_glslcHelper = NULL;
    }
    if (this == DeviceState::GetActive()) {
        g_glslcHelper = NULL;
    }
}

void DeviceState::rebuildGLSLCHelper()
{
    // Tear down the old GLSLC helper, if any.
    destroyGLSLCHelper();

    // Build a new GLSLC helper for the device state.
    m_glslcHelper = new lwnTest::GLSLCHelper(&m_device, DeviceProgramPoolSize, g_glslcLibraryHelper,
        g_glslcHelperCache, g_dxcLibraryHelper);
    if (!m_glslcHelper) {
        log_output("Error: Could not allocate a GLSLCHelper object\n");
    }

    // Set a default scratch memory requirement, with a global pool.
    m_glslcHelper->SetShaderScratchMemory(&m_shaderScratchPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, &m_queueCB);

    // Update the global GLSLC helper variable if this DeviceState is active.
    if (this == DeviceState::GetActive()) {
        g_glslcHelper = m_glslcHelper;
    }
}

} // namespace lwnTest
