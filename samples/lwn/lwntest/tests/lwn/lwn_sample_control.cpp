/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "cmdline.h"

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif


using namespace lwn;
using namespace lwn::dt;

// ----------------------------------- LWNSampleControl ------------------------------------------

enum SampleControlTest {
    SAMPCTRL_TEST_PER_PIXEL,
    SAMPCTRL_TEST_PER_SAMPLE_SAMPLEID,
    SAMPCTRL_TEST_PER_SAMPLE_SAMPLEPOS,
    SAMPCTRL_TEST_PER_SAMPLE_SAMPLEMASK,
    SAMPCTRL_TEST_PER_SAMPLE_INTERPOLANT
};

class LWNSampleControl {
    // The multi-sample texture we are testing, with its memory pool.
    MemoryPool *m_textureMSPool;
    Texture *m_textureMS;
    bool m_textureMSInit;

    MemoryPool *m_texturePool;
    Texture *m_texture;
    bool m_textureInit;

    Program *m_programRTPerPixel;
    Program *m_programRTPerSample1;
    Program *m_programRTPerSample2;
    Program *m_programRTPerSample3;
    Program *m_programRTPerSample4;

    // Rest of the stuff we need to draw the texture.
    MemoryPoolAllocator *m_bufpool;
    Program *m_program;
    Sampler *m_sampler;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    Buffer *m_ubo;
    LWNuint m_vertexDataSize;
    Sync *m_sync;

    struct Vertex {
        vec3 position;
        vec3 uv;
    };

public:
    LWNSampleControl();
    ~LWNSampleControl();

    void init();

    void createMSRenderTarget(int width, int height, Format fmt, int samples);
    void fillMSRenderTarget(int width, int height, Format fmt, int samples, SampleControlTest testMode);

    void drawMS(int width, int height, int samples);
    void draw(int width, int height, int samples);
};

LWNSampleControl::LWNSampleControl()
    : m_textureMSInit(false), m_textureInit(false)
{
}

LWNSampleControl::~LWNSampleControl()
{
    m_program->Free();
    m_programRTPerPixel->Free();
    m_programRTPerSample1->Free();
    m_programRTPerSample2->Free();
    m_programRTPerSample3->Free();
    m_programRTPerSample4->Free();

    m_sampler->Free();
    delete m_bufpool;
}

void LWNSampleControl::init() {
    DEBUG_PRINT(("LWNSampleControl:: Creating test assets...\n"));
    Device *device = DeviceState::GetActive()->getDevice();
    m_bufpool = new MemoryPoolAllocator(device, NULL, 0x20000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Compile shaders.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 uv;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "out vec2 ouv;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ouv = uv.xy;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0, std140) uniform Block {\n"
        "    int samples;\n"
        "    int width;\n"
        "    int height;\n"
        "};\n"
        "layout (binding=0) uniform sampler2DMS tex;"
        "void main() {\n"
        "  fcolor = vec4(0.0);\n"
        "  ivec2 ouv_ = ivec2(ouv.x * float(width), ouv.y * float(height));\n"
        "  if (ouv.y < 0.05) {\n"
        "      // Show a green stepped gradient to mark where each individual sample is.\n"
        "      int sampleID = min(int(ouv.x * float(samples)), samples - 1);\n"
        "      fcolor = vec4(0.0, float(sampleID) / float(samples), 0.0, 1.0);\n"
        "  } else if (ouv.y < 0.4) {\n"
        "      // Show individual sample fetched, where the samples should be aligned with\n"
        "      // the top gradient.\n"
        "      int sampleID = min(int(ouv.x * float(samples)), samples - 1);\n"
        "      fcolor = texelFetch(tex, ouv_, sampleID);\n"
        "  } else if (ouv.y < 0.43) {\n"
        "      fcolor = vec4(1.0f);\n"
        "  } else {\n"
        "      // Show the samples added together.\n"
        "      for (int i = 0; i < samples; i++) {\n"
        "          fcolor += texelFetch(tex, ouv_, i);\n"
        "      }\n"
        "      fcolor /= float(samples);\n"
        "  }\n"
        "}\n";

    VertexShader vsRT(440);
    vsRT <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 uv;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "out vec2 ouv;\n"
        "void main() {\n"
        "  gl_Position = vec4(position.xy * 0.83, position.z, 1.0);\n"
        "  ouv = uv.xy;\n"
        "}\n";
    VertexShader vsInterpolant(440);
    vsInterpolant <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 uv;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "sample out vec2 ouv;\n"
        "void main() {\n"
        "  gl_Position = vec4(position.xy, position.z, 1.0);\n"
        "  ouv = uv.xy;\n"
        "}\n";
    FragmentShader fsRTPerPixel(440);
    fsRTPerPixel <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ouv.x, ouv.y, 0.0, 0.0);\n"
        "}\n";
    FragmentShader fsRTPerSample1(440);
    fsRTPerSample1 <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0, std140) uniform Block {\n"
        "    int samples;\n"
        "    int width;\n"
        "    int height;\n"
        "};\n"
        "void main() {\n"
        "  float v = float(gl_SampleID) / float(gl_NumSamples);\n"
        "  fcolor = vec4(v, v, v, 0.0);\n"
        "  if (samples != gl_NumSamples) {\n"
        "    fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  }"
        "}\n";
    FragmentShader fsRTPerSample2(440);
    fsRTPerSample2 <<
        "sample in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0, std140) uniform Block {\n"
        "    int samples;\n"
        "    int width;\n"
        "    int height;\n"
        "};\n"
        "void main() {\n"
        "    vec2 uv = ouv * vec2(width, height);\n"
        "    uv = fract(uv);\n"
        // Flip y since uv starts at 1 at top, but gl_SamplePosition starts y=0 at bottom.
        "    uv.y = 1.0f - uv.y;\n"
        "    vec2 samplePos = gl_SamplePosition;\n"
        "    if (samples != gl_NumSamples && ouv.y >= 0.9) {\n"
        // Just draw red at the top if gl_NumSamples isn't working correctly.
        "        fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "    } else {\n"
        "      if (ouv.y <= 0.8f) { \n"
        "        fcolor = vec4(samplePos.x, samplePos.y, 0, 1);\n"
        "      } else {\n"
        "        fcolor = vec4(uv.x, uv.y, 0, 1);\n"
        "      }\n"
        "    }\n"
        "}\n";
    FragmentShader fsRTPerSample3(440);
    fsRTPerSample3 <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0, std140) uniform Block {\n"
        "    int samples;\n"
        "    int width;\n"
        "    int height;\n"
        "};\n"
        "void main() {\n"
        "    gl_SampleMask[0] = 0x55555555;\n"
        "    fcolor = vec4(1.0);\n"
        "}\n";
    FragmentShader fsRTPerSample4(440);
    fsRTPerSample4 <<
        "sample in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0, std140) uniform Block {\n"
        "    int samples;\n"
        "    int width;\n"
        "    int height;\n"
        "};\n"
        "void main() {\n"
        "    vec2 uv = ouv * vec2(width, height);\n"
        "    uv = fract(uv);\n"
        "    fcolor = vec4(uv.x, uv.y, 1.0, 1.0);\n"
        "}\n";

    m_program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }

    m_programRTPerPixel = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programRTPerPixel, vsRT, fsRTPerPixel)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }

    m_programRTPerSample1 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programRTPerSample1, vsRT, fsRTPerSample1)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }

    m_programRTPerSample2 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programRTPerSample2, vsInterpolant, fsRTPerSample2)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }

    m_programRTPerSample3 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programRTPerSample3, vsRT, fsRTPerSample3)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }

    m_programRTPerSample4 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programRTPerSample4, vsInterpolant, fsRTPerSample4)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }
    g_glslcHelper->SetSeparable(LWN_FALSE);


    // Set up a dummy sampler to be used to sample that texture.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    m_sampler = sb.CreateSampler();

    // Create vertex data.
    const int vertexCount = 4;
    static const Vertex vertexData[] = {
        { vec3(-1, -1, 0.0), vec3(0.0, 1.0, 0.0) },
        { vec3(+1, -1, 0.0), vec3(1.0, 1.0, 0.0) },
        { vec3(+1, +1, 0.0), vec3(1.0, 0.0, 0.0) },
        { vec3(-1, +1, 0.0), vec3(0.0, 0.0, 0.0) }
    };
    m_vertexDataSize = sizeof(vertexData);
    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, uv);
    m_vertexState = vertexStream.CreateVertexArrayState();
    m_vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, *m_bufpool, vertexData);

    // Create shader UBO buffer.
    BufferBuilder uboBuilder;
    uboBuilder.SetDevice(device).SetDefaults();
    uboBuilder.SetDefaults().SetDevice(device);
    m_ubo = m_bufpool->allocBuffer(&uboBuilder, BUFFER_ALIGN_UNIFORM_BIT, 3 * sizeof(LWNuint));
}

void LWNSampleControl::createMSRenderTarget(int width, int height, Format fmt, int samples)
{
    Device *device = DeviceState::GetActive()->getDevice();
    if (m_textureMSInit) {
        // Re-create the texture.
        m_textureMS->Free();
        m_textureMSPool->Free();
    }
    m_textureMSInit = true;

    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device).SetDefaults()
                .SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE)
                .SetFormat(fmt)
                .SetSize2D(width, height)
                .SetSamples(samples);
    size_t textureSize = textureBuilder.GetStorageSize();
    m_textureMSPool = device->CreateMemoryPool(NULL, textureSize, MemoryPoolType::GPU_ONLY);
    m_textureMS = textureBuilder.CreateTextureFromPool(m_textureMSPool, 0);
}

void LWNSampleControl::fillMSRenderTarget(int width, int height, Format fmt, int samples, SampleControlTest testMode)
{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;
    assert(m_textureMSInit);

    Program* program = NULL;
    switch (testMode) {
        case SAMPCTRL_TEST_PER_PIXEL:
            program = m_programRTPerPixel;
            break;
        case SAMPCTRL_TEST_PER_SAMPLE_SAMPLEID:
            program = m_programRTPerSample1;
            break;
        case SAMPCTRL_TEST_PER_SAMPLE_SAMPLEPOS:
            program = m_programRTPerSample2;
            break;
        case SAMPCTRL_TEST_PER_SAMPLE_SAMPLEMASK:
            program = m_programRTPerSample3;
            break;
        case SAMPCTRL_TEST_PER_SAMPLE_INTERPOLANT:
            program = m_programRTPerSample4;
            break;
    }

    MultisampleState msState;
    msState.SetDefaults();
    msState.SetMultisampleEnable(samples > 1 ? LWN_TRUE : LWN_FALSE);
    msState.SetSamples(samples);
    int* uboPtr = (int*) m_ubo->Map();
    uboPtr[0] = samples;
    uboPtr[1] = width;
    uboPtr[2] = height;

    queueCB.Barrier(LWN_BARRIER_ILWALIDATE_SHADER_BIT);
    queueCB.BindMultisampleState(&msState);
    queueCB.SetRenderTargets(1, &m_textureMS, NULL, NULL, NULL);
    queueCB.SetViewportScissor(0, 0, width, height);
    queueCB.ClearColor(0, 0.0, 0.0, 0.5, 1.0);

    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, m_ubo->GetAddress(), 3 * sizeof(LWNuint));
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

    msState.SetMultisampleEnable(LWN_FALSE);
    msState.SetSamples(0);
    queueCB.BindMultisampleState(&msState);
    queueCB.Barrier(LWN_BARRIER_ILWALIDATE_TEXTURE_BIT | LWN_BARRIER_ORDER_FRAGMENTS_BIT);
}

void LWNSampleControl::drawMS(int width, int height, int samples)
{
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    TextureHandle texHandle = device->GetTextureHandle(m_textureMS->GetRegisteredTextureID(), m_sampler->GetRegisteredID());
    assert(texHandle);

    int* uboPtr = (int*) m_ubo->Map();
    uboPtr[0] = samples;
    uboPtr[1] = width;
    uboPtr[2] = height;

    queueCB.Barrier(LWN_BARRIER_ILWALIDATE_SHADER_BIT);
    queueCB.BindProgram(m_program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, m_ubo->GetAddress(), 3 * sizeof(LWNuint));
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

// --------------------------------- LWNSampleControlTest ----------------------------------------

class LWNSampleControlTest {
    static const int cellSize = 68;
    static const int cellMargin = 3;
    SampleControlTest test;

public:
    LWNTEST_CppMethods();
    LWNSampleControlTest(SampleControlTest testVariation);
};

LWNSampleControlTest::LWNSampleControlTest(SampleControlTest testVariation)
    :  test(testVariation)
{}

lwString LWNSampleControlTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Tests per-sample shading using the LWN api.\n"
        "Successful test result should contain boxes iterating through various parameters of width,\n"
        "height, samples, format each containing a multisampled square on a blue background.\n\n"
        "The boxes are split into two parts horizontally; the top part above the white line.\n"
        "displays the individual samples (sample 0 - (NumSamples - 1) from left to right.\n"
        "The bottom below the white line shows the average result of all samples.\n"
        "The thin green bar at the top shows a stepped gradient showing the 'boundaries'\n"
        "between individual samples.\n\n";

    switch (test) {
        case SAMPCTRL_TEST_PER_PIXEL:
            sb <<
                "This subtest serves as a control; it renders to a multi-sample render target\n"
                "with per-sample shading off, and expect all samples fetched to be of the same value.\n"
                "Each box should have a quad colored with a diagonal red-green gradient.\n";
            break;
        case SAMPCTRL_TEST_PER_SAMPLE_SAMPLEID:
            sb <<
                "This subtest ensures that per-sample control is turned on and working\n"
                "when gl_SampleID is used from a fragment shader.\n"
                "Each box should have a white quad with 0.5 opacity, and each individual sample\n"
                "should be a greyscale value directly corresponding to the sampleID.";
            break;
        case SAMPCTRL_TEST_PER_SAMPLE_SAMPLEPOS:
            sb <<
                "This subtest ensures that per-sample control is turned on and working\n"
                "when gl_samplePosition is used from a fragment shader.\n"
                "Each box will draw the gl_SamplePos to the .x and .y channels.  Additionally,\n"
                "the top half of the individual sample cells are drawn with a [0, 1] interpolated\n"
                "with the sample qualifier, and the other half is draw with gl_SamplePosition.\n"
                "The interpolant (uv.x, 1-uv.y) and gl_SamplePosition.xy should be exactly the same.\n"
                "If gl_NumSamples is not returning the correct results, then a red line will be drawn\n"
                "at the top of the cell.\n";

            break;
        case SAMPCTRL_TEST_PER_SAMPLE_SAMPLEMASK:
            sb <<
                "This subtest ensures that per-sample control is turned on and working\n"
                "when gl_sampleID is used from a fragment shader.\n"
                "Each box should have a white quad with 0.5 opacity, and individual samples\n"
                "should alternate between opaque white and transparent.";
            break;
        case SAMPCTRL_TEST_PER_SAMPLE_INTERPOLANT:
            sb <<
                "This subtest ensures that sample interpolant varyings work.\n"
                "Each box should appear purple-grey. Individual samples should\n"
                "show sample position co-ordinates in red & green channels, with blue = 1.0.\n"
                "Those should show up as various shares of purple / cyan.\n";
            break;
    }

    return sb.str();
}

int LWNSampleControlTest::isSupported() const
{
#if defined(SPIRV_ENABLED)
    // This test is failing when using Spir-V. The root cause is in glslang.
    // This version of glslang does not handle variables with the "sample"
    // qualifier correctly. Texture coordinates using the "sample" qualifier
    // do not produce the expected result and the tests using fsRTPerSample2
    // and fsRTPerSample4 will fail when used with Spir-V.
    // See http://lwbugs/200379828
    return (lwogCheckLWNAPIVersion(40, 11) && !useSpirv);
#else
    return lwogCheckLWNAPIVersion(40, 11);
#endif
}

void LWNSampleControlTest::doGraphics() const
{
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.1, 0.1, 0.1, 1.0);

    LWNSampleControl sampleControlTest;
    sampleControlTest.init();

    int heights[] = {
        32,
        128,
        32,
        512,
        3,
        1024
    };
    int widths[] = {
        32,
        128,
        421,
        512,
        4,
        6
    };
    int numSz = __GL_ARRAYSIZE(heights);
    int samples[] = {
        2, 4, 8
    };
    int numSamples = __GL_ARRAYSIZE(samples);
    Format fmts[] = {
        Format::RGBA8,
        Format::RGBA16,
        Format::RGB10A2
    };
    int numFormats = __GL_ARRAYSIZE(fmts);

    int c = 0;
    Sync *sync = device->CreateSync();
    for (int i = 0; i < numSz; i++) {
        for (int j = 0; j < numSamples; j++) {
            for (int l = 0; l < numFormats; l++) {
                DEBUG_PRINT(("    testing %d x %d fmt %d samples %d testcase %d\n",
                    widths[i], heights[i], l, samples[j], (int) test));

                // Create the multisample texture.
                sampleControlTest.createMSRenderTarget(widths[i], heights[i], fmts[l], samples[j]);

                // Fill the multisample texture.
                sampleControlTest.fillMSRenderTarget(
                    widths[i], heights[i], fmts[l], samples[j], test);

                // Draw the multisample texture.
                g_lwnWindowFramebuffer.bind();
                g_lwnWindowFramebuffer.setViewportScissor();
                SetCellViewportScissorPadded(queueCB, c % cellsX, c / cellsX, cellMargin);
                sampleControlTest.drawMS(widths[i], heights[i], samples[j]);

                // Need to CPU wait for everything to complete before destroying and re-creating the texture.
                queueCB.submit();
                queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
                queue->Flush();
                sync->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);

                c++;
            }
        }
    }

    queueCB.submit();
    queue->Finish();
    sync->Free();
}

OGTEST_CppTest(LWNSampleControlTest, lwn_sample_control_per_pixel, (SAMPCTRL_TEST_PER_PIXEL));
OGTEST_CppTest(LWNSampleControlTest, lwn_sample_control_sampleid, (SAMPCTRL_TEST_PER_SAMPLE_SAMPLEID));
OGTEST_CppTest(LWNSampleControlTest, lwn_sample_control_samplepos, (SAMPCTRL_TEST_PER_SAMPLE_SAMPLEPOS));
OGTEST_CppTest(LWNSampleControlTest, lwn_sample_control_samplemask, (SAMPCTRL_TEST_PER_SAMPLE_SAMPLEMASK));
OGTEST_CppTest(LWNSampleControlTest, lwn_sample_control_interpolant, (SAMPCTRL_TEST_PER_SAMPLE_INTERPOLANT));
