/*
 * Copyright (c) 2016-2018, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <algorithm>
#include <array>

#include "../../elw/cmdline.h"

#if defined(LW_WINDOWS)
#include "windows.h"
#include "GL/gl.h"
#else
#include "EGL/egl.h"
#include "nn/gll.h"
#endif

static const int TEX_WIDTH = 16;
static const int TEX_HEIGHT = 16;

#define DEBUG_MODE 0

#if DEBUG_MODE
#   define DEBUG_PRINT(x) do { printf x; } while(0)
#else
#   define DEBUG_PRINT(x)
#endif

using namespace lwn;

#if !defined(APIENTRY)
#define APIENTRY
#endif

#if defined(LW_WINDOWS)
#define GETPROCADDRESS wglGetProcAddress
#else
#define GETPROCADDRESS eglGetProcAddress
#endif

#if defined(LW_WINDOWS)
// Windows has no GLES headers, and an ancient GL header, so we need to set up some
// symbols ourselves.
#define GL_COLOR_ATTACHMENT0    0x8CE0
#define GL_FRAMEBUFFER          0x8D40
#define GL_TIMEOUT_IGNORED      0xFFFFFFFFFFFFFFFFull
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117

typedef GLint* GLsync;
typedef uint64_t GLuint64;

typedef void (APIENTRY * PFNGLBINDFRAMEBUFFERPROC)(GLenum target, GLuint framebuffer);
typedef void (APIENTRY * PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei n, const GLuint *framebuffers);
typedef void (APIENTRY * PFNGLDRAWBUFFERSPROC)(GLsizei n, const GLenum *bufs);
typedef void (APIENTRY * PFNGLDRAWTEXTURELWPROC)(GLuint texture, GLuint sampler,
                                             GLfloat x0, GLfloat y0, GLfloat x1, GLfloat y1,
                                             GLfloat z,
                                             GLfloat s0, GLfloat t0, GLfloat s1, GLfloat t1);
typedef void (APIENTRY * PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum target, GLenum attachment,
                                                        GLenum textarget, GLuint texture,
                                                        GLint level);
typedef void (APIENTRY * PFNGLGENFRAMEBUFFERSPROC)(GLsizei n, GLuint *framebuffers);
typedef void (APIENTRY * PFNGLTEXSTORAGE2DPROC)(GLenum target, GLsizei levels,
                                                GLenum internalformat, GLsizei width,
                                                GLsizei height);
typedef void (APIENTRY * PFNGLWAITSYNCPROC)(GLsync sync, GLbitfield flags, GLuint64 timeout);
typedef GLsync (APIENTRY * PFNGLFENCESYNCPROC)(GLenum condition, GLbitfield flags);
typedef void (APIENTRY * PFNGLDELETESYNCPROC)(GLsync sync);
static PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;
static PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;
static PFNGLDRAWBUFFERSPROC glDrawBuffers;
static PFNGLDRAWTEXTURELWPROC glDrawTextureLW;
static PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D;
static PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
static PFNGLTEXSTORAGE2DPROC glTexStorage2D;
static PFNGLWAITSYNCPROC glWaitSync;
static PFNGLFENCESYNCPROC glFenceSync;
static PFNGLDELETESYNCPROC glDeleteSync;
#endif // defined(LW_WINDOWS)

static void InitGLEntryPoints()
{
    //glDrawTextureLW = (PFNGLDRAWTEXTURELWPROC)GETPROCADDRESS("glDrawTextureLW");
#if defined(LW_WINDOWS)
    glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC)GETPROCADDRESS("glBindFramebuffer");
    glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)GETPROCADDRESS("glDeleteFramebuffers");
    glDrawBuffers = (PFNGLDRAWBUFFERSPROC)GETPROCADDRESS("glDrawBuffers");
    glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)GETPROCADDRESS("glFramebufferTexture2D");
    glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)GETPROCADDRESS("glGenFramebuffers");
    glTexStorage2D = (PFNGLTEXSTORAGE2DPROC)GETPROCADDRESS("glTexStorage2D");
    glWaitSync = (PFNGLWAITSYNCPROC)GETPROCADDRESS("glWaitSync");
    glFenceSync = (PFNGLFENCESYNCPROC)GETPROCADDRESS("glFenceSync");
    glDeleteSync = (PFNGLDELETESYNCPROC)GETPROCADDRESS("glDeleteSync");
#endif
}

class InteropTexture
{
public:
    InteropTexture(GLuint glTexture, Device *device);
    ~InteropTexture();
    Texture *texture() { return &mTexture; }
    LWNuint id() const { return mID; }
    uint32_t glTexture() const { return m_glTexture; }
private:
    Texture mTexture;
    LWNuint mID;
    uint32_t m_glTexture;
};

InteropTexture::InteropTexture(GLuint glTexture, Device *device) : mID(0), m_glTexture(0)
{
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults().SetGLTextureName(glTexture);
    // Create a texture that isn't managed by a memory pool.
    if (!mTexture.Initialize(&tb)) {
        return;
    }
    mID = g_lwnTexIDPool->Register(&mTexture);

    m_glTexture = tb.GetGLTextureName();
}

InteropTexture::~InteropTexture()
{
    g_lwnTexIDPool->Deregister(&mTexture);
    mTexture.Finalize();
}

class LwnGLInteropTest
{
public:
    LWNTEST_CppMethods();
};

lwString LwnGLInteropTest::getDescription() const
{
    return "Test for LWN/GL interop.\n"
           "1. Create a simple gradient checkerboard texture in GL, create an LWN texture using "
           "it as a source, then render a full-viewport quad using that texture.\n"
           "2. Exercise render-to-texture in GL, using a stretch blit for the previously created "
           "GL texture. Create an LWN texture from the render target texture, and likewise use it "
           "in a full-viewport textured quad.";
}

int LwnGLInteropTest::isSupported() const
{
    return useGL && lwogCheckLWNAPIVersion(40, 15);
}

template<int WIDTH, int HEIGHT>
static GLuint CreateSimpleGLTexture()
{
    // Simple 2D texture.
    static GLubyte texData[WIDTH * HEIGHT * 4];
    GLubyte *texPtr = texData;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            *texPtr++ = x * 255 / WIDTH;
            *texPtr++ = ((x + y) % 2) ? 128 : 255;
            *texPtr++ = y * 255 / HEIGHT;
            *texPtr++ = 255;
        }
    }
    GLuint glTex;
    glGenTextures(1, &glTex);
    glBindTexture(GL_TEXTURE_2D, glTex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE,
                    texData);
    glFinish();
    return glTex;
}

static GLuint CreateRenderTargetGLTexture(GLuint simpleTexture)
{
    // "Typical" use case. NPOT render target.
    static const int RT_WIDTH = 40;
    static const int RT_HEIGHT = 60;
    GLuint rtTex;
    glGenTextures(1, &rtTex);
    glBindTexture(GL_TEXTURE_2D, rtTex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, RT_WIDTH, RT_HEIGHT);

    // Set up the render target
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtTex, 0);
    GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);

    // Blit simpleTexture into the render target, leaving some blue around the edges of the render
    // target.
    glBindTexture(GL_TEXTURE_2D, simpleTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glViewport(0, 0, RT_WIDTH, RT_HEIGHT);
    glClearColor(0.0, 0.0, 0.5, 1.0);
    glScissor(0, 0, RT_WIDTH, RT_HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawTextureLW(simpleTexture, 0,
                    0.1 * RT_WIDTH, 0.1 * RT_HEIGHT, 0.9 * RT_WIDTH, 0.9 * RT_HEIGHT,
                    0.0,
                    0.0, 0.0, 1.0, 1.0);

    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glFinish();

    return rtTex;
}

static void InitProgram(Program* program)
{
    // Program for drawing a full-viewport textured quad.
    VertexShader vs(440);
    vs << "void main() { }\n";
    GeometryShader gs(440);
    gs <<
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=4) out;\n"
        "out vec2 gST; // Texture coordinates\n"
        "void main() {\n"
        "  for (int t = 0; t <= 1; ++t) {\n"
        "    for (int s = 0; s <= 1; ++s) {\n"
        "      gST = vec2(s, t);\n"
        "      gl_Position = vec4(gST * 2.0 - vec2(1.0), 0.0, 1.0);\n"
        "      EmitVertex();\n"
        "    }\n"
        "  }\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec2 gST;\n"
        "out vec4 fColor;\n"
        "layout (binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fColor = texture(tex, gST);\n"
        "}\n";
    if (!g_glslcHelper->CompileAndSetShaders(program, vs, gs, fs)) {
        DEBUG_PRINT(("Shader compile error.\n======\n%s\n\n", g_glslcHelper->GetInfoLog()));
    }
}

static void DrawTexture(const InteropTexture& texture, Sampler *sampler,
                        Device *device, QueueCommandBuffer& queueCB)
{
    TextureHandle texHandle = device->GetTextureHandle(texture.id(), sampler->GetRegisteredID());
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);

    queueCB.DrawArrays(DrawPrimitive::POINTS, 0, 1);
}

void LwnGLInteropTest::doGraphics() const
{
    InitGLEntryPoints();

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    cellTestInit(2, 1);

    Program *program = device->CreateProgram();
    InitProgram(program);
    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);

    SamplerBuilder samplerBuilder;
    samplerBuilder.SetDevice(device).SetDefaults();
    samplerBuilder.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *sampler = samplerBuilder.CreateSampler();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    SetCellViewportScissorPadded(queueCB, 0, 0, 1);
    GLuint simpleGLTex = CreateSimpleGLTexture<TEX_WIDTH,TEX_HEIGHT>();
    InteropTexture simpleLwnTex(simpleGLTex, device);
    DrawTexture(simpleLwnTex, sampler, device, queueCB);
    // Verify that the texture builder returned the correct gl texture.
    if (simpleLwnTex.glTexture() != simpleGLTex) {
        LWNFailTest();
        return;
    }

    SetCellViewportScissorPadded(queueCB, 1, 0, 1);
    GLuint renderTargetGLTex = CreateRenderTargetGLTexture(simpleGLTex);
    InteropTexture renderTargetLwnTex(renderTargetGLTex, device);
    DrawTexture(renderTargetLwnTex, sampler, device, queueCB);
    // Verify that the texture builder returned the correct gl texture.
    if (renderTargetLwnTex.glTexture() != renderTargetGLTex) {
        LWNFailTest();
        return;
    }

    queueCB.submit();
    queue->Finish();
    GLuint textures[] = { simpleGLTex, renderTargetGLTex };
    glDeleteTextures(2, textures);
}

OGTEST_CppTest(LwnGLInteropTest, lwn_gl_interop, );

//=============

struct LwnGlSyncInteropTest { LWNTEST_CppMethods(); };
OGTEST_CppTest(LwnGlSyncInteropTest, lwn_gl_sync_interop, );

lwString LwnGlSyncInteropTest::getDescription() const
{
    return "Test for GL-LWN fences interop: LWNsync created from GLSync object and vice versa.\n"
           "The test runs 4 subtests that verify proper ordering of events with GL-LWN shared syncs\n"
           "in both direction, and inproper ordering without the syncs.\n"
           "Running this test with -v gives more data should it fail.\n";
}

int LwnGlSyncInteropTest::isSupported() const
{
    return useGL && lwogCheckLWNAPIVersion(53, 301);
}

const static int SYNC_RT_WIDTH = 1024;
const static int SYNC_RT_HEIGHT = 1024;

static void glBindRT(GLuint name, GLuint& fbo)
{
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, name, 0);
    GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);
    glFinish();
}

struct SyncInteropQuarterTest {
    const static uint32_t CTRL_SIZE = 1024;
    const static uint32_t CMD_SIZE = 8 * 1024 * 1024; //!< MAX_GP_LENGTH_BYTES
    const static uint32_t PASS = 3;

    SyncInteropStatus syncStatus;
    int loops;
    bool useSync;
    std::array<bool, PASS> isPassBlue;

    uint64_t ctrlSpace[CTRL_SIZE];
    DeviceState *deviceState;
    Device *device;
    MemoryPool *commandPool;
    InteropTexture *disputedTexture;
    Queue *queue;
    CommandBuffer *cb;
    Sync lwnSync;

    GLuint fbo;
    GLsync glSync;

    void lwnBindRT()
    {
        cb->BeginRecording();
        Texture* rt = disputedTexture->texture();
        cb->SetRenderTargets(1, &rt, NULL, NULL, NULL);
        cb->SetViewportScissor(0, 0, SYNC_RT_WIDTH, SYNC_RT_HEIGHT);
        CommandHandle handle = cb->EndRecording();
        queue->SubmitCommands(1, &handle);
        queue->Flush();
    }

    void glPaintItBlue()
    {
        glViewport(0, 0, SYNC_RT_WIDTH, SYNC_RT_HEIGHT);
        glScissor(0, 0, SYNC_RT_WIDTH, SYNC_RT_HEIGHT);
        glClearColor(0.0, 0.0, 0.5, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void lwnPaintItRed()
    {
        cb->ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }

    void glUnbindRT()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
    }

    virtual void core() = 0;

    SyncInteropQuarterTest(bool sync)
        : loops(2560)
        , useSync(sync)
        , isPassBlue()
        , deviceState(DeviceState::GetActive())
        , device(deviceState->getDevice())
        , commandPool(device->CreateMemoryPool(NULL, CMD_SIZE, MemoryPoolType::CPU_COHERENT))
    {
        GLuint name = CreateSimpleGLTexture<SYNC_RT_WIDTH,SYNC_RT_HEIGHT>();
        glBindRT(name, fbo);
        disputedTexture = new InteropTexture{name, device};
    }

    ~SyncInteropQuarterTest()
    {
        glUnbindRT();
        GLuint glDisputedTexture = disputedTexture->glTexture();
        glDeleteTextures(1, &glDisputedTexture);
        commandPool->Free();
    }

    bool isThatOnePixelBlue()
    {
        static const std::array<uint8_t,4> lwnRed = {{ 255, 0, 0, 255 }};
        static const std::array<uint8_t,4> glBlue = {{ 0, 0, 127, 255 }};
        std::array<uint8_t,4> pixel = {{ 0, 0, 0, 0 }};
        glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &pixel);
        if (pixel == glBlue) {
            return true;
        } else if (pixel == lwnRed) {
            return false;
        } else {
            lwnTest::fail("this pixel should be either blue or red... "
                          "%u %u %u %u\n", pixel[0], pixel[1], pixel[2], pixel[3]);
            return false;
        }
    }

    void checkIfRTBlue()
    {
        for (uint32_t pass = 0; pass < PASS; ++pass) {
            loops *= 2;

            queue = deviceState->getQueue();
            cb = device->CreateCommandBuffer();

            cb->AddControlMemory(ctrlSpace, CTRL_SIZE);
            cb->AddCommandMemory(commandPool, 0, CMD_SIZE);

            lwnBindRT();

            core();

            queue->Finish();
            glFinish();

            isPassBlue[pass] = isThatOnePixelBlue();

            glDeleteSync(glSync);
            lwnSync.Finalize();
            cb->Free();
        }
    }
};

struct lwnToGl : public SyncInteropQuarterTest {
    lwnToGl(bool sync)
        : SyncInteropQuarterTest(sync)
    {
    }

    virtual void core()
    {
        cb->BeginRecording();
        {
            for (int i = 0; i < loops; ++i) {
                lwnPaintItRed();
            }

            lwnSync.Initialize(device);
            cb->FenceSync(&lwnSync, SyncCondition::GRAPHICS_WORLD_SPACE_COMPLETE, 0 /*!FLUSH_FOR_CPU*/);
        }
        CommandHandle handle = cb->EndRecording();
        queue->SubmitCommands(1, &handle);
        queue->Flush();

        syncStatus = lwnSync.CreateGLSync((uint64_t*)&glSync);

        if (useSync) {
            glWaitSync(glSync, 0, GL_TIMEOUT_IGNORED);
        }
        glPaintItBlue();
        glFlush();
    }
};

struct glToLWN : public SyncInteropQuarterTest {
    glToLWN(bool sync)
        : SyncInteropQuarterTest(sync)
    {
    }

    virtual void core()
    {
        for (int i = 0; i < loops; ++i) {
            glPaintItBlue();
        }
        glSync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();

        cb->BeginRecording();
        {
            lwnSync.InitializeFromFencedGLSync(device, glSync, &syncStatus);
            if (useSync) {
                cb->WaitSync(&lwnSync);
            }
            lwnPaintItRed();
        }
        CommandHandle handle = cb->EndRecording();
        queue->SubmitCommands(1, &handle);
        queue->Flush();
    }
};

void LwnGlSyncInteropTest::doGraphics() const
{
    bool success = true;
    InitGLEntryPoints();

#if !defined(LW_WINDOWS)
    // TEST_NO_SYNC are not strictly deterministic, allow some retries.
    // Retry will start with a much higher loop count, no need to retry too much.
    // Not used on Windows.
    int nonSynchronizedRetries = 2;
#endif

#define BLUE  blue
#define RED  !blue
#define ANY(quarterTest,color)                      \
    std::any_of(quarterTest.isPassBlue.cbegin(),    \
                quarterTest.isPassBlue.cend(),      \
                [](bool blue){ return color; })

#if defined(LW_WINDOWS)
// Windows has strong ordering in the driver, that interferes with this.
#define TEST_NO_SYNC(T,expect_color,other_color)
#else
#define TEST_NO_SYNC(T,expect_color,other_color)                        \
    {                                                                   \
        int tries = nonSynchronizedRetries;                             \
        bool quarterSuccess = true;                                     \
        T quarterTest(false);                                           \
        do {                                                            \
            quarterTest.checkIfRTBlue();                                \
            quarterSuccess = ANY(quarterTest, expect_color);            \
        } while (--tries && !quarterSuccess);                           \
        if (!quarterSuccess) {                                          \
            lwnTest::fail("The RT was always " #other_color             \
                          ", but we expected it to be " #expect_color   \
                          " at some iteration since we do not "         \
                          "synchronize GL and LWN.\n");                 \
            success = false;                                            \
        }                                                               \
    }
#endif

#define TEST_SYNC(T,expect_color,other_color,direction)                 \
    {                                                                   \
        T quarterTest(true);                                            \
        quarterTest.checkIfRTBlue();                                    \
        success = !ANY(quarterTest, other_color);                       \
        if (!success) {                                                 \
            lwnTest::fail("The RT was " #other_color                    \
                          ", but we expected it to be " #expect_color   \
                          " since we synchronize " direction ".\n");    \
        }                                                               \
    }

    TEST_NO_SYNC(lwnToGl, RED, BLUE)
    // Synchronization reverses expectations
    TEST_SYNC(lwnToGl, BLUE, RED, "LWN -> GL")

    TEST_NO_SYNC(glToLWN, BLUE, RED)
    TEST_SYNC(glToLWN, RED, BLUE, "GL -> LWN")

    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.SetScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    if (success) { queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0); }
    else         { queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0); }
    queueCB.submit();
    queue->Finish();
}
