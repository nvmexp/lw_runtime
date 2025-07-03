/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

/**********************************************************************/

using namespace lwn;

// lwogtest doesn't like random spew to standard output.  We just eat any
// output unless LWN_BASIC_DO_PRINTF is set to 1.
#define LWN_BASIC_DO_PRINTF     0

#if LWN_BASIC_DO_PRINTF
#define log_output printf
#else
static void log_output(const char *fmt, ...) {}
#endif

/**********************************************************************/

// Alpha-to-coverage test class
class LwnMultisampleA2CovTest {
    bool m_dither;
public:
    LwnMultisampleA2CovTest(bool dither) : m_dither(dither) {}
    LWNTEST_CppMethods();
};

int LwnMultisampleA2CovTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(48,1);
}

lwString LwnMultisampleA2CovTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Tests Alpha to Coverage for each MSAA mode. Each gradient is drawn as a white "
        "bar with alpha values from 0 on the left to 1 on the right. Each group of three "
        "gradients has 0, 2, and 4 samples, respectively from bottom-to-top. (0 is "
        "a control non-MSAA target.)  This test is drawn in 3 groups of three gradients.  "
        "From bottom to top, the groups use SampleMasks of 0xFFFF, 0xAAAA, and 0x5555.  "
        "Alpha-to-coverage dithering is " << (m_dither ? "en" : "dis") << "abled.";
    return sb.str();
}

void LwnMultisampleA2CovTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Shaders and programs
    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec2 texcoord;\n"
        "out IO { vec2 tc; };\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  tc = texcoord;\n"
        "}\n";
    FragmentShader fsCov(440);
    fsCov <<
        "layout(location = 0) out vec4 color;\n"
        "in IO { vec2 tc; };\n"
        "void main() {\n"
        "  color = vec4(1.0, 1.0, 1.0, tc.x);\n"
        "}\n";
    FragmentShader fsTex(440);
    fsTex <<
        "layout(location = 0) out vec4 color;\n"
        "layout(binding = 0) uniform sampler2D tex;\n"
        "in IO { vec2 tc; };\n"
        "void main() {\n"
        "  color = texture(tex, tc);\n"
        "}\n";
    Program *pgmCov = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgmCov, vs, fsCov)) {
        log_output("Shader compile error.  infoLog = \n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    Program *pgmTex = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgmTex, vs, fsTex)) {
        log_output("Shader compile error.  infoLog = \n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    // Vertex buffer
    struct Vertex {
        dt::vec3 position;
        dt::vec2 texcoord;
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vstate = stream.CreateVertexArrayState();
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec2(0.0, 0.0) },
        { dt::vec3(-1.0,  1.0, 0.0), dt::vec2(0.0, 1.0) },
        { dt::vec3( 1.0,  1.0, 0.0), dt::vec2(1.0, 1.0) },
        { dt::vec3( 1.0, -1.0, 0.0), dt::vec2(1.0, 0.0) },
    };
    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();
    cmd.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    cmd.BindVertexArrayState(vstate);

    // Index buffer
    static const uint8_t indexData[] = { 0, 1, 3, 2 };
    Buffer *ibo = AllocAndFillBuffer(device, queue, cmd, allocator, indexData,
                                     sizeof(indexData), BUFFER_ALIGN_INDEX_BIT, false);
    BufferAddress iboAddress = ibo->GetAddress();

    // Multisample state objects
    MultisampleState msOffState;
    MultisampleState msState;
    msOffState.SetDefaults();
    msState.SetDefaults();
    msState.SetMultisampleEnable(LWN_TRUE);
    msState.SetAlphaToCoverageEnable(LWN_TRUE);
    msState.SetAlphaToCoverageDither(m_dither ? LWN_TRUE : LWN_FALSE);

    // Sampler
    SamplerBuilder samplerBuilder;
    samplerBuilder.SetDevice(device).SetDefaults();
    samplerBuilder.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    samplerBuilder.SetWrapMode(WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER,
                               WrapMode::CLAMP_TO_BORDER);
    Sampler *sampler = samplerBuilder.CreateSampler();
    LWNuint samplerID = sampler->GetRegisteredID();

    // Clear the backbuffer
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    cmd.ClearColor();

    static const int msSize[] = { 0, 2, 4 };
    static const unsigned short msMask[] = { 0xFFFF, 0xAAAA, 0x5555 };
    static const int cellWidth = lwrrentWindowWidth;
    static const int cellHeight = lwrrentWindowHeight / (__GL_ARRAYSIZE(msSize) * __GL_ARRAYSIZE(msMask));

    for (unsigned int s = 0; s < __GL_ARRAYSIZE(msSize); s++)
    {
        // Create framebuffer with msSize[s] number of samples
        Framebuffer gradFb(cellWidth, cellHeight);
        gradFb.setFlags(TextureFlags::COMPRESSIBLE);
        gradFb.setColorFormat(0, Format::RGBA8);
        gradFb.setSamples(msSize[s]);
        gradFb.alloc(device);
        Texture *colorTex = gradFb.getColorTexture(0);
        TextureHandle texHandle = device->GetTextureHandle(colorTex->GetRegisteredTextureID(), samplerID);

        msState.SetSamples(msSize[s]);

        for (unsigned int i = 0; i < __GL_ARRAYSIZE(msMask); i++)
        {
            // Draw alpha-to-coverage image
            gradFb.bind(cmd);
            gradFb.setViewportScissor();
            cmd.ClearColor();
            cmd.BindMultisampleState(&msState);
            cmd.SetSampleMask(msMask[i]);
            cmd.BindProgram(pgmCov, ShaderStageBits::ALL_GRAPHICS_BITS);
            cmd.DrawElements(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_BYTE, __GL_ARRAYSIZE(indexData), iboAddress);
            if (msSize[s]) {
                gradFb.downsample(cmd);
            }
            cmd.submit();
            queue->Finish();

            // Copy to backbuffer
            g_lwnWindowFramebuffer.bind();
            cmd.SetViewportScissor(0, ((i * __GL_ARRAYSIZE(msSize)) + s) * cellHeight,
                                   cellWidth, cellHeight);
            cmd.BindMultisampleState(&msOffState);
            cmd.SetSampleMask(msMask[0]);
            cmd.BindProgram(pgmTex, ShaderStageBits::ALL_GRAPHICS_BITS);
            cmd.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
            cmd.DrawElements(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_BYTE, __GL_ARRAYSIZE(indexData), iboAddress);
            cmd.submit();
            queue->Finish();
        }

        gradFb.destroy();
    }
}

OGTEST_CppTest(LwnMultisampleA2CovTest, lwn_multisample_a2cov, (true));
OGTEST_CppTest(LwnMultisampleA2CovTest, lwn_multisample_a2cov_nd, (false));
