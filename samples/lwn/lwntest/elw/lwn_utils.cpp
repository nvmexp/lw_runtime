/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_c.h"
#include "contexts.h"
#include "lwn_utils.h"
#include "float_util.h"
#include "cmdline.h"

#include "lwnUtil/lwnUtil_AlignedStorageImpl.h"
#include "lwnUtil/lwnUtil_PoolAllocatorImpl.h"
#include "lwnUtil/lwnUtil_GlslcHelperImpl.h"
#include "lwnUtil/lwnUtil_CommandMemImpl.h"
#include "lwnUtil/lwnUtil_QueueCmdBufImpl.h"
#include "lwnUtil/lwnUtil_TexIDPoolImpl.h"

#if defined(LW_LINUX)
#include <dlfcn.h> // For ETC1 support
#endif

//////////////////////////////////////////////////////////////////////////

// Global device and queue objects, created during lwogtest initialization.
LWNdevice *g_lwnDevice = NULL;
LWNqueue *g_lwnQueue = NULL;

// Global GLSLCHelper used to interface with the GLSLC library.
lwnTest::GLSLCHelper *g_glslcHelper = NULL;

// Global GLSLC helper cache, used in conjunction with the command line options -lwnGlslcInput
// and -lwnGlslcOutputFile in order to process precompiled binaries for use with LWNtest.
lwnUtil::GLSLCHelperCache *g_glslcHelperCache = NULL;

lwnUtil::QueueCommandBufferBase *g_lwnQueueCB = NULL;
lwnUtil::CommandBufferMemoryManager g_lwnCommandMem;
lwnUtil::CompletionTracker *g_lwnTracker = NULL;
LWNmemoryPool *g_lwnScratchMemPool = NULL;

// Global pool for texture and sampler IDs associated wih g_lwnDevice, created
// during lwogtest initialization.
lwnUtil::TexIDPool* g_lwnTexIDPool = NULL;

// Global state objects used to initialize device state before each test
static LWNblendState g_lwnBlendState;
static LWNchannelMaskState g_lwnChannelMaskState;
static LWNcolorState g_lwnColorState;
static LWNmultisampleState g_lwnMultisampleState;
static LWNpolygonState g_lwnPolygonState;
static LWNdepthStencilState g_lwnDepthStencilState;

// Global "window" framebuffer, used as a default framebuffer for all LWN tests.
lwnTest::WindowFramebuffer g_lwnWindowFramebuffer;

// Global LWN GetProcAddress entry point.
static PFNLWNDEVICEGETPROCADDRESSPROC g_lwnDeviceGetProcAddress;

// Global variable tracking whether the LWN debug layer has been enabled by
// tests.  When <false>, the debug layer still might be enabled globally via
// the "-lwndebug" option.
static bool g_lwnDebugEnabled = false;

// Major and minor API versions queried from the driver during initialization.
int g_lwnMajorVersion = -1;
int g_lwnMinorVersion = -1;

// Queried LWN device capabilities.
lwnTest::DeviceCaps g_lwnDeviceCaps = { 0 };

#if defined(LW_LINUX)
// Windows doesn't support ETC1, and HOS will execute liblwn-etc1's static
// initializer just by linking against it. For Linux, use a manual load to
// kick off the static initializer.
static void* s_etcHandle = nullptr;

static void InitializeEtc1Lib()
{
    s_etcHandle = dlopen("liblwn-etc1.so", RTLD_LAZY);
    if (s_etcHandle == nullptr) {
        printf("Could not open liblwn-etc1.so");
    }
}

static void FinalizeEtc1Lib()
{
    if (s_etcHandle != nullptr) {
        dlclose(s_etcHandle);
        s_etcHandle = nullptr;
    }
}
#endif // if defined(LW_LINUX)

class IgnoredDebugWarnings
{
private:
    static const int32_t NOT_SET = -1;
    // Extend this field with a fixed size array if you need to support
    // multiple simultaneous warning ignores.
    int32_t m_ignoredMessageID;
public:
    IgnoredDebugWarnings() : m_ignoredMessageID(NOT_SET) {}
    bool isWarningIgnored(int32_t warningID) const { return m_ignoredMessageID == warningID; }
    void ignoreWarning(int32_t warningID) {
        assert(m_ignoredMessageID == NOT_SET);
        m_ignoredMessageID = warningID;
    }
    void allowWarning(int32_t warningID) {
        m_ignoredMessageID = NOT_SET;
    }
    void allowWarningAll() {
        m_ignoredMessageID = NOT_SET;
    }
};

static IgnoredDebugWarnings g_ignoredWarnings;

// Expose a C API for IgnoredDebugWarnings
void DebugWarningIgnore(int32_t warningID) { g_ignoredWarnings.ignoreWarning(warningID); }
void DebugWarningAllow(int32_t warningID)  { g_ignoredWarnings.allowWarning(warningID); }
void DebugWarningAllowAll()                { g_ignoredWarnings.allowWarningAll(); }

//////////////////////////////////////////////////////////////////////////

// Global debug callback used when the "-lwndebug" option is set.
void LWNAPIENTRY LWNUtilsDebugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                                       LWNdebugCallbackSeverity severity, LWNstring message, void *userParam)
{
    // Don't report callbacks when API tests intentionally enable the debug
    // layer.
    if (g_lwnDebugEnabled) {
        return;
    }

    // Filter out informational messages we don't need to report.
    if (id == 1244 && severity == LWN_DEBUG_CALLBACK_SEVERITY_NOTIFICATION) {
        // lwnQueueSubmitCommands:  flushing - queue command memory usage exceeded flush threshold
        return;
    }

    if (g_ignoredWarnings.isWarningIgnored(id)) {
        return;
    }

    if (source == LWN_DEBUG_CALLBACK_SOURCE_INITIALIZATION) {
        printf("\nLWN Global Debug callback (independent from -lwndebug option):\n");
    } else {
        printf("\nLWN Debug callback (from -lwndebug option):\n");
    }
    printf("  source:       0x%08x\n", source);
    printf("  type:         0x%08x\n", type);
    printf("  id:           0x%08x\n", id);
    printf("  severity:     0x%08x\n", severity);
    printf("  userParam:    0x%08x%08x\n", 
           uint32_t(uint64_t(uintptr_t(userParam)) >> 32), uint32_t(uintptr_t(userParam)));
    printf("  message:\n    %s\n", message);

    fflush(stdout);

    switch (type) {
    case LWN_DEBUG_CALLBACK_TYPE_API_ERROR:
        lwnDebugErrorMessageCount++;
        break;
    case LWN_DEBUG_CALLBACK_TYPE_API_WARNING:
        lwnDebugWarningMessageCount++;
        break;
    case LWN_DEBUG_CALLBACK_TYPE_INITIALIZATION_ERROR:
        lwnDebugInitErrorMessageCount++;
        break;
    default:
        assert(0);
        break;
    }
}

// Debug callback used to verify basic operation of the debug layer when
// "-lwnDebug" is specified.  Writes "1" to the provided <userParam> variable
// to indicate that a callback is received.
static void LWNAPIENTRY SanityCheckLWNDebugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                                                    LWNdebugCallbackSeverity severity, LWNstring message, void *userParam)
{
    *((int *) userParam) = 1;
}

// Basic sanity check on the LWN debug layer when "-lwnDebug" is specified.
// This code temporarily disables the normal debug callback, installs a new
// callback and then triggers an error to determine if the callback was
// performed.  Returns 1 if the callback worked and 0 otherwise.
int SanityCheckLWNDebug(LWNdevice *device)
{
    int gotDebugCallback = 0;
    LWNsamplerBuilder sb;
    lwnDeviceInstallDebugCallback(device, LWNUtilsDebugCallback, NULL, LWN_FALSE);
    lwnDeviceInstallDebugCallback(device, SanityCheckLWNDebugCallback, &gotDebugCallback, LWN_TRUE);
    lwnSamplerBuilderSetMinMagFilter(&sb, (LWNminFilter) 0x3456789A, (LWNmagFilter) 0x01234567);
    lwnDeviceInstallDebugCallback(device, SanityCheckLWNDebugCallback, &gotDebugCallback, LWN_FALSE);
    lwnDeviceInstallDebugCallback(device, LWNUtilsDebugCallback, NULL, LWN_TRUE);

    // If we didn't get a debug callback, count that as a debug error for test
    // result reporting purposes.
    if (!gotDebugCallback) {
        lwnDebugErrorMessageCount++;
    }

    return gotDebugCallback;
}

// Set up the DXC library helper during LWN initialization.
static lwnUtil::DXCLibraryHelper *InitializeDXCLibraryHelper(void)
{
    lwnUtil::DXCLibraryHelper *helper = new lwnUtil::DXCLibraryHelper;
    if (!helper) {
        printf("DXC library helper allocation failed.\n");
        return NULL;
    }
    return helper;
}

// Set up the GLSLC library helper during LWN initialization.
static lwnUtil::GLSLCLibraryHelper *InitializeGLSLCLibraryHelper(void)
{
    lwnUtil::GLSLCLibraryHelper *helper = new lwnUtil::GLSLCLibraryHelper;
    if (!helper) {
        printf("GLSLC library helper allocation failed.\n");
        return NULL;
    }

    // Enable logging errors by default from the GLSLC library helper class.
    helper->GetLogger()->SetLoggerFunction(glslcLoggingFunction);
    helper->GetLogger()->SetEnable(true);

    // Load the GLSLC library from disk, or set the function pointers based on GLSLC functions
    // statically linked in.  On all platforms other than Windows, static linkage is the default.
    // On Windows, the build flag GLSLC_LIB_DYNAMIC_LOADING can be used to indicate demand loading (default)
    // or static linkage (from an export lib file, still requires DLL library during runtime).
#if defined(_WIN32) && defined(GLSLC_LIB_DYNAMIC_LOADING)
    if (lwnGlslcDLL) {
        helper->LoadDLL(lwnGlslcDLL);
    } else {
        printf("Shader compilation path not specified.  Compilation using the GLSLC compiler requires the \"-lwnGlslcDLL\"\n"
               "option or an lwntest build linking with the GLSLC exports library.\n");
        delete helper;
        return NULL;
    }
#else
    // Load the static library functions.
    helper->LoadDLL(NULL);
#endif

    // Ensure GLSLC is loaded, and if not error out.
    if (!helper->IsLoaded()) {
        printf("GLSLC library failed to initialize.\n");
        delete helper;
        return NULL;
    }

    return helper;
}

// Set up the LWN bootstrap loader during LWN initialization.
static int InitializeLWNBootstrap(void)
{
    // Install global debug callback to catch early errors.
    lwnInstallGlobalDebugCallback(LWNUtilsDebugCallback, NULL);

    g_lwnDeviceGetProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC) (lwnBootstrapLoader("lwnDeviceGetProcAddress"));

    if (!g_lwnDeviceGetProcAddress) {
        printf("LWN bootstrap loader failed to initialize.\n");
        return 0;
    }

    ReloadCInterface(NULL, g_lwnDeviceGetProcAddress);
    ReloadCppInterface(NULL, g_lwnDeviceGetProcAddress);

    // Check that functions exist to set up a device builder and device, query
    // the version, and release the device if creation fails. If we're missing
    // any of these, give up.
    if (NULL == pfnc_lwnDeviceBuilderSetDefaults ||
        NULL == pfnc_lwnDeviceBuilderSetFlags ||
        NULL == pfnc_lwnDeviceInitialize ||
        NULL == pfnc_lwnDeviceGetInteger ||
        NULL == pfnc_lwnDeviceFinalize)
    {
        printf("LWN bootstrap loader failed to query device functions.\n");
        return 0;
    }

    // Check the API version.  A major version mismatch between the driver and
    // the test indicates that some API changed in a backwards-incompatible
    // manner.  All tests should be disabled in this case, since they could
    // easily crash otherwise.  Allow device creation to succeed if the driver
    // reports a lower minor version; tests can run successfully as long as
    // isSupported() is properly coded to ensure that nothing needing an
    // unsupported minor version is used.
    int lwnMajorVersion, lwnMinorVersion;
    lwnDeviceGetInteger(NULL, LWN_DEVICE_INFO_API_MAJOR_VERSION, &lwnMajorVersion);
    lwnDeviceGetInteger(NULL, LWN_DEVICE_INFO_API_MINOR_VERSION, &lwnMinorVersion);
    if (lwnMajorVersion != LWN_API_MAJOR_VERSION) {
        printf("LWN bootstrap loader failed to query device functions.\n");
        printf("LWN API version mismatch:\n"
               "Reported version:  %d.%d\n"
               "Expected version:  %d.%d\n",
               lwnMajorVersion, lwnMinorVersion,
               LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION);
        return 0;
    }

    return 1;
}

// Create a device and a default queue that tests can use.
// We also create some default state objects that we can use to reinitialize
// queue state between tests.
int InitializeLWN(void)
{
    // Set up the GLSLC library interface.
    g_glslcLibraryHelper = InitializeGLSLCLibraryHelper();
    if (!g_glslcLibraryHelper) {
        return 0;
    }

    // Set up the DXC library interface.
    g_dxcLibraryHelper = InitializeDXCLibraryHelper();
    if (!g_dxcLibraryHelper) {
        return 0;
    }

#if defined(LW_LINUX)
    InitializeEtc1Lib();
#endif

    // Perform the initial bootstrapping operation to set up LWN.
    if (!InitializeLWNBootstrap()) {
        return 0;
    }

    // Create and activate a new set of device state to use in our tests.
    LWNbitfield deviceFlags = 0;
    DeviceState *defaultDevice = new DeviceState(LWNdeviceFlagBits(deviceFlags));
    if (!defaultDevice || !defaultDevice->isValid()) {
        delete defaultDevice;
        return 0;
    }
    defaultDevice->SetDefault();
    defaultDevice->SetActive();

    // Create state objects used to initialize device state before each test.
    lwnBlendStateSetDefaults(&g_lwnBlendState);
    lwnChannelMaskStateSetDefaults(&g_lwnChannelMaskState);
    lwnColorStateSetDefaults(&g_lwnColorState);
    lwnMultisampleStateSetDefaults(&g_lwnMultisampleState);
    lwnPolygonStateSetDefaults(&g_lwnPolygonState);
    lwnDepthStencilStateSetDefaults(&g_lwnDepthStencilState);

    // Initialize the GLSLC cache.
    g_glslcHelperCache = NULL;

    if (lwnGlslcInputFile || lwnGlslcOutputFile) {
        g_glslcHelperCache = new lwnUtil::GLSLCHelperCache(defaultDevice->getDevice());
        if (!g_glslcHelperCache) {
            printf("Failed to initialize the GLSLC cache helper.\n");
            return 0;
        }

        if (lwnGlslcInputFile) {
            bool cacheInitSuccess = LoadAndSetBinaryCacheFromFile(g_glslcHelperCache, lwnGlslcInputFile);
            if (!cacheInitSuccess) {
                printf("Failed to init the GLSLC binary cache with file %s.  Continuing with empty initial cache.\n", lwnGlslcInputFile);
            }
        }
    }

    return 1;
}

void FinalizeLWN(void)
{
    g_lwnWindowFramebuffer.destroy();

    DeviceState *defaultDevice = DeviceState::GetDefault();
    if (!defaultDevice) {
        return;
    }
    delete defaultDevice;

#if defined(LW_LINUX)
    FinalizeEtc1Lib();
#endif

    delete g_glslcHelperCache;
    g_glslcHelperCache = NULL;

    delete g_glslcLibraryHelper;
    g_glslcLibraryHelper = NULL;

    delete g_dxcLibraryHelper;
    g_dxcLibraryHelper = NULL;
}

int lwogCheckLWNAPIVersion(int32_t neededMajor, int32_t neededMinor)
{
    if (g_lwnMajorVersion > neededMajor) {
        return true;
    }

    if (g_lwnMajorVersion < neededMajor) {
        return false;
    }

    return g_lwnMinorVersion >= neededMinor;
}

int lwogCheckLWNGLSLCGpuVersion(uint32_t neededMajor, uint32_t neededMinor)
{
    GLSLCversion dllVersion = g_glslcLibraryHelper->GetVersion();
    return (GLSLCGpuCodeVersionInfo(dllVersion) >=
            GLSLCGpuCodeVersionInfo(neededMajor, neededMinor));
}

int lwogCheckLWNGLSLCPackageVersion(uint32_t neededVer)
{
    GLSLCversion dllVersion = g_glslcLibraryHelper->GetVersion();
    return (dllVersion.package >= neededVer);
}

// Reload LWN function pointers to get the appropriate version after changing
// devices (e.g., when creating or deleting debug devices).
void ReloadLWNEntryPoints(LWNdevice *device, bool apiDebug)
{
    g_lwnDebugEnabled = apiDebug;   // tell our handler to ignore errors for tests using debug mode
    ReloadCInterface(device, g_lwnDeviceGetProcAddress);
    ReloadCppInterface(device, g_lwnDeviceGetProcAddress);
}

// Clear the LWN window to black at initialization time.
void lwogClearWindow()
{
    lwnUtil::QueueCommandBuffer &queueCB = *g_lwnQueueCB;
    float black[4] = { 0, 0, 0, 0 };
    g_lwnWindowFramebuffer.setSize(lwrrentWindowWidth, lwrrentWindowHeight);
    g_lwnWindowFramebuffer.bind();
    lwnCommandBufferClearColor(queueCB, 0, black, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.submit();
    g_lwnWindowFramebuffer.present();
    lwogSwapBuffers();
}

void LWNTestClear(LWNTestResultColor color)
{
    const float green[] = {0.0f, 1.0f, 0.0f};
    const float red[] = {1.0f, 0.0f, 0.0f};
    const float yellow[] = {1.0f, 1.0f, 0.0f};
    const float blue[] = {0.0f, 0.0f, 1.0f};
    const float *selected;
    switch (color) {
    case LWNTEST_COLOR_PASS: selected = green; break;
    case LWNTEST_COLOR_FAIL: selected = red; break;
    case LWNTEST_COLOR_WNF: selected = yellow; break;
    case LWNTEST_COLOR_UNSUPPORTED: selected = blue; break;
    default: selected = red; break;
    };
    lwnUtil::QueueCommandBuffer &queueCB = *g_lwnQueueCB;
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    lwnCommandBufferClearColor(queueCB, 0, selected, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.submit();
}

void LWNTestFinish()
{
    lwnQueueFinish(g_lwnQueue);
}

//////////////////////////////////////////////////////////////////////////

// Non-inline functions for the LWN datatypes classes.
namespace lwnTool { namespace texpkg { namespace dt {

// Constructor/operator for colwerting between our "float16" class and FP32.
float16::float16(double f)
{
    this->m_value = lwF32toS10E5(f);
}
float16::operator double() const
{
    return lwS10E5toF32(this->m_value);
}

// Use macros to stamp out the implementation of the lwnFormat() methods for
// all of our generic, encoded, and packed vector types.
#define MKFMTS(_tprefix, _esuffix)                          \
   template <> LWNformat traits<_tprefix##vec1>::lwnFormat()   \
       { return LWN_FORMAT_ ## R ## _esuffix; }                    \
   template <> LWNformat traits<_tprefix##vec2>::lwnFormat()   \
       { return LWN_FORMAT_ ## RG ## _esuffix; }                   \
   template <> LWNformat traits<_tprefix##vec3>::lwnFormat()   \
       { return LWN_FORMAT_ ## RGB ## _esuffix; }                  \
   template <> LWNformat traits<_tprefix##vec4>::lwnFormat()   \
       { return LWN_FORMAT_ ## RGBA ## _esuffix; }
MKFMTS(, 32F);
MKFMTS(u, 32UI);
MKFMTS(i, 32I);
MKFMTS(u8n, 8);
MKFMTS(u16n, 16);
MKFMTS(i8n, 8SN);
MKFMTS(i16n, 16SN);
MKFMTS(u8, 8UI);
MKFMTS(u16, 16UI);
MKFMTS(i8, 8I);
MKFMTS(i16, 16I);
MKFMTS(u2f8, 8_UI2F);
MKFMTS(u2f16, 16_UI2F);
MKFMTS(u2f32, 32_UI2F);
MKFMTS(i2f8, 8_I2F);
MKFMTS(i2f16, 16_I2F);
MKFMTS(i2f32, 32_I2F);
MKFMTS(f16, 16F);
#undef MKFMTS
template <> LWNformat traits<vec4_rgba4>::lwnFormat()          { return LWN_FORMAT_RGBA4; }
template <> LWNformat traits<vec3_rgb5>::lwnFormat()           { return LWN_FORMAT_RGB5; }
template <> LWNformat traits<vec4_rgb5a1>::lwnFormat()         { return LWN_FORMAT_RGB5A1; }
template <> LWNformat traits<vec3_rgb565>::lwnFormat()         { return LWN_FORMAT_RGB565; }
template <> LWNformat traits<vec4_rgb10a2>::lwnFormat()        { return LWN_FORMAT_RGB10A2; }
template <> LWNformat traits<vec4_rgb10a2ui>::lwnFormat()      { return LWN_FORMAT_RGB10A2UI; }
template <> LWNformat traits<vec4_rgb10a2sn>::lwnFormat()      { return LWN_FORMAT_RGB10A2SN; }
template <> LWNformat traits<vec4_rgb10a2i>::lwnFormat()       { return LWN_FORMAT_RGB10A2I; }
template <> LWNformat traits<vec4_rgb10a2ui_to_f>::lwnFormat() { return LWN_FORMAT_RGB10A2_UI2F; }
template <> LWNformat traits<vec4_rgb10a2i_to_f>::lwnFormat()  { return LWN_FORMAT_RGB10A2_I2F; }
template <> LWNformat traits<vec3_bgr5>::lwnFormat()           { return LWN_FORMAT_BGR5; }
template <> LWNformat traits<vec4_bgr5a1>::lwnFormat()         { return LWN_FORMAT_BGR5A1; }
template <> LWNformat traits<vec3_bgr565>::lwnFormat()         { return LWN_FORMAT_BGR565; }

} } } // namespace lwnTool::texpkg::dt

//////////////////////////////////////////////////////////////////////////

// Re-initializes graphics state to defaults, using state objects created (and
// implicitly initialized) at device creation time.
static void lwnResetGraphicsDefaultState(LWNcommandBuffer *cb)
{
    // BlendState is unique in that it only defines state for a single output
    // blend target. The other state objects do not require this sort of loop.
    for (int i=0; i<8; i++) {
        lwnBlendStateSetBlendTarget(&g_lwnBlendState, i);
        lwnCommandBufferBindBlendState(cb, &g_lwnBlendState);
    }
    lwnCommandBufferBindChannelMaskState(cb, &g_lwnChannelMaskState);
    lwnCommandBufferBindColorState(cb, &g_lwnColorState);
    lwnCommandBufferBindMultisampleState(cb, &g_lwnMultisampleState);
    lwnCommandBufferBindPolygonState(cb, &g_lwnPolygonState);
    lwnCommandBufferBindDepthStencilState(cb, &g_lwnDepthStencilState);

    // VertexState is skipped because it needs to be lwstomized for the vertex
    // data each time.

    // We cannot bind a null fragment and vertex shader, but we can nullify the
    // rest.
    lwnCommandBufferBindProgram(cb, 0, (LWN_SHADER_STAGE_GEOMETRY_BIT |
                                        LWN_SHADER_STAGE_TESS_CONTROL_BIT |
                                        LWN_SHADER_STAGE_TESS_EVALUATION_BIT));

    lwnCommandBufferSetPointSize(cb, 1.0f);
    lwnCommandBufferSetLineWidth(cb, 1.0f);
    lwnCommandBufferSetSampleMask(cb, ~0);
    lwnCommandBufferSetPrimitiveRestart(cb, LWN_FALSE, 0);
    if (g_lwnDeviceCaps.supportsConservativeRaster) {
        lwnCommandBufferSetConservativeRasterEnable(cb, LWN_FALSE);
        lwnCommandBufferSetConservativeRasterDilate(cb, 0);
        lwnCommandBufferSetSubpixelPrecisionBias(cb, 0, 0);
    }

    // Enable tiled caching if it's enabled via a command line debug flag
    if (enableTiledCache) {
        lwnCommandBufferSetTiledCacheAction(cb, LWN_TILED_CACHE_ACTION_ENABLE);
    } else {
        lwnCommandBufferSetTiledCacheAction(cb, LWN_TILED_CACHE_ACTION_DISABLE);
    }

    lwnCommandBufferSetCopyRowStride(cb, 0);
    lwnCommandBufferSetCopyImageStride(cb, 0);

    // We don't bother to initialize the values for state that's disabled by
    // default (e.g., stencil, polygon offset). Tests are expected to
    // initialize this state separately if it's enabled.
    //
    // The same is true for bindable resources (buffers, textures, samplers.)
    // Tests are expected to bind valid resources if they are used; tests
    // should not rely on default values being returned by unbound resources.
}

// Default initGraphics() method for LWN tests, sizing and binding the window
// framebuffer, and setting the scissor and viewport to cover the whole
// framebuffer. We also reinitialize the device state to clear any state left
// over from the previous test.
void lwnDefaultInitGraphics(void)
{
    DeviceState *defaultDevice = DeviceState::GetDefault();
    assert(defaultDevice == DeviceState::GetActive());
    LWNdevice *device = defaultDevice->getDevice();
    LWNcommandBuffer *cb = defaultDevice->getQueueCB();

    lwnResetGraphicsDefaultState(cb);
    g_lwnWindowFramebuffer.setSize(lwrrentWindowWidth, lwrrentWindowHeight);
    g_lwnWindowFramebuffer.bind();
    lwnCommandBufferSetScissor(cb, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    lwnCommandBufferSetViewport(cb, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    EnableLWNObjectTracking();

    defaultDevice->rebuildGLSLCHelper();

    // If running with the debug layer enabled, sanity check the debug layer
    // before running each test to make sure nothing caused it to get
    // uninstalled.
    if (lwnDebugEnabled) {
        SanityCheckLWNDebug(device);
    }
}

// Default exitGraphics() method for LWN tests, which presents the window
// framebuffer and cleans up shader and API object allocations.
void lwnDefaultExitGraphics(void)
{
    DeviceState *defaultDevice = DeviceState::GetDefault();
    assert(defaultDevice == DeviceState::GetActive());
    defaultDevice->destroyGLSLCHelper();

    g_lwnQueueCB->checkUnflushedCommands();
    g_lwnTracker->insertFence(g_lwnQueue);
    g_lwnWindowFramebuffer.present();
    shaderCleanup();
    lwnTest::allocationCleanup();
    DisableLWNObjectTracking();
    DebugWarningAllowAll();

    // LWN assumes that there is one device; the debug state is reset whenever a device is deleted.
    // In order to "restore" the debug state of a device, we use ReloadLWNEntryPoints.
    ReloadLWNEntryPoints(g_lwnDevice, false);
}
