#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "cmdline.h"

//////////////////////////////////////////////////////////////////////////

using namespace lwn;

class LWNfragmentCoverageToColor
{
public:
    
    LWNfragmentCoverageToColor(int samp, bool e, bool mask, int c, bool t, bool a2c) :
        numSamples(samp),
        depthSamples(t ? 8 : samp),
        early(e),
        outputMask(mask),
        outcol2(c),
        tir(t),
        alphaToCoverage(a2c)
    {}

    LWNTEST_CppMethods();

private:
    struct Vertex {
        dt::vec3 position;
    };

private:
    static const int texWidth = 2, texHeight = 16;

private:
    int numSamples;
    int depthSamples;

    bool early;
    bool outputMask;
    int outcol2;
    bool tir;
    bool alphaToCoverage;
};

int LWNfragmentCoverageToColor::isSupported() const
{
    // TIR mode wants N > M
    if (tir && numSamples > 4) {
        return 0;
    }

#if defined(SPIRV_ENABLED)
    // We use a custom #pragma in this test (coverageToColorTarget 2), and this
    // #pragma is not supported in SPIR-V.  Since the pragma is required for
    // functional correctness in LWN, we skip this test if compiling with
    // SPIR-V.
    if (useSpirv) {
        return 0;
    }
#endif

    return g_lwnDeviceCaps.supportsTargetIndependentRasterization && 
           g_lwnDeviceCaps.supportsFragmentCoverageToColor &&
           lwogCheckLWNAPIVersion(41, 4);
}

lwString LWNfragmentCoverageToColor::getDescription() const
{
    return "Test LW_fragment_coverage_to_color. Initialize the depth buffer with N horizontal bands where each band "
           "has a different sample killed. Then render a fullscreen triangle which has its coverage written out to "
           "one of the drawbuffers. Read back the buffer and print its values to the screen. Tests with \"early\" in "
           "the name force early_fragment_tests.";
}

void LWNfragmentCoverageToColor::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int coverageOutputIndex = 2; // the color output for coverage to color

    VertexShader vsDef(440);
    vsDef <<
        "layout(location = 0) in vec3 pos;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(pos, 1.0);\n"
        "}\n";

    FragmentShader fsDef(440);
    fsDef <<
        "precision highp float;"
        "layout(location=0) out vec4 outColor;\n"
        "void main()\n"
        "{\n"
        "    outColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "}\n";

    // shader program
    Program *pgmDefault = device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgmDefault, vsDef, fsDef);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    FragmentShader fs(440);
    // Explicitly specify the coverage to color output index.  This is needed since the
    // fragment shader doesn't explicitly output the C2C color, and LWN needs the coverage
    // color target enabled when compiling the program.
    fs << "#pragma coverageToColorTarget 2\n";
    if (early) {
        fs << "layout(early_fragment_tests) in;\n";
    }
    fs << "layout(location=0) out vec4 col;\n";
    // GM20x coverage to color needs to have an output
    // defined and used if not broadcasting.
    // output mask always when writing to more than one output.
    if (outputMask || outcol2) {
        fs << "layout(location=" << coverageOutputIndex << ") out uint mask; \n";
    }
    if (outcol2) {
        fs << "layout(location=" << outcol2 << ") out vec4 c2;\n";
    }
    fs << "void main() {;\n";
    fs << "    col = vec4(0,0.5,0,0.5);\n";
    // GM20x coverage to color needs to have an output
    // defined and used if not broadcasting.
    // output mask always when writing to more than one output.
    fs << ((outputMask || outcol2) ? "    mask = 0xCLw;\n" : "") <<
        (outcol2 ? "    c2 = vec4(1,0,0,1);\n" : "") <<
        "}\n";

    // shader program
    Program *pgm = device->CreateProgram();
    compiled = g_glslcHelper->CompileAndSetShaders(pgm, vsDef, fs);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    // Copy multisample to (horizontally) bloated non-MS. The single sampled
    // surface is numSamples*width of the MS surface. THe shader looks
    // up samples based on modulo of the fragment x coordinate, so for example if
    // sample rate is 4x and gl_FragCoord.x = 7 the texelFetch will lookup
    // sample 7%4 = 3 at MS taxture x = 7/4 = 1. 
    FragmentShader fsDownsample(440);
    fsDownsample <<
        "layout(location=0) out uvec4 outColor;\n"
        "uniform highp usampler2DMS tex;\n"
        "layout(binding = 0) uniform Block {\n"
        "    int numSamples;\n"
        "};\n"
        "void main() {\n"
        "    int x = int(gl_FragCoord.x) / numSamples;\n"
        "    int samp = int(gl_FragCoord.x) % numSamples;\n"
        "    outColor = texelFetch(tex, ivec2(x, int(gl_FragCoord.y)), samp);\n"
        "}\n";

    // shader program
    Program *downsamplepgm = device->CreateProgram();
    compiled = g_glslcHelper->CompileAndSetShaders(downsamplepgm, vsDef, fsDownsample);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    Framebuffer singleSampleFB, multiSampleFB;

    // Create 1X buffer in singleSampleFB
    singleSampleFB.setSize(texWidth*numSamples, texHeight);
    singleSampleFB.setColorFormat(Format::R32UI);
    singleSampleFB.alloc(device);
    singleSampleFB.bind(queueCB);
    singleSampleFB.setViewportScissor();

    LWNuint ucolor_clear[] = { 0, 0, 0, 0 };
    queueCB.ClearColorui(0, ucolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
    
    multiSampleFB.setSize(texWidth, texHeight);
    multiSampleFB.setSamples(numSamples);
    multiSampleFB.setDepthSamples(tir ? depthSamples : 0);
    multiSampleFB.setColorFormat(0, Format::RGBA8);
    multiSampleFB.setColorFormat(1, Format::RGBA8);
    multiSampleFB.setColorFormat(coverageOutputIndex, Format::R32UI);
    multiSampleFB.setDepthStencilFormat(Format::DEPTH24);
    multiSampleFB.alloc(device);
    multiSampleFB.bind(queueCB);
    multiSampleFB.setViewportScissor();

    // bind TIR enabled multisample state before clearing depth
    MultisampleState multisample;
    multisample.SetDefaults().
        SetMultisampleEnable(LWN_TRUE).
        SetSamples(numSamples).
        SetRasterSamples(tir ? depthSamples : 0);
    queueCB.BindMultisampleState(&multisample);

    LWNfloat fcolor_clear[] = { 0, 0, 0, 0 };
    queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.ClearColor(1, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.ClearColorui(coverageOutputIndex, ucolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
    queueCB.submit();
    queue->Finish();

    DepthStencilState depth;
    depth.SetDefaults().
        SetDepthTestEnable(LWN_TRUE).
        SetDepthFunc(DepthFunc::ALWAYS).
        SetDepthWriteEnable(LWN_TRUE);
    queueCB.BindDepthStencilState(&depth);

    ChannelMaskState mask;
    mask.SetDefaults().
        SetChannelMask(0, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE).
        SetChannelMask(1, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE). // binding 1 has no output in the shader, so we don't need this...
        SetChannelMask(2, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE);
    queueCB.BindChannelMaskState(&mask);

    // default fragment shader just outputs a solid color
    queueCB.BindProgram(pgmDefault, ShaderStageBits::ALL_GRAPHICS_BITS);

    MemoryPoolAllocator coherent_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexStreamSet streamSet(stream);
    VertexArrayState vertex = streamSet.CreateVertexArrayState();
    queueCB.BindVertexArrayState(vertex);

    {
        static const Vertex vertexData[] = {
            { dt::vec3(-1.0, -1.0, 0.9) },
            { dt::vec3( 1.0, -1.0, 0.9) },
            { dt::vec3(-1.0,  1.0, 0.9) },
            { dt::vec3( 1.0,  1.0, 0.9) },
        };

        Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), coherent_allocator, vertexData);
        BufferAddress vboAddr = vbo->GetAddress();

        queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, __GL_ARRAYSIZE(vertexData));
    }

    // For each sample, set a region of the screen to give that sample
    // a near depth.
    for (int i = 0; i < depthSamples; ++i) {
        float minY = 2.0f / depthSamples*i - 1.0f;
        float maxY = 2.0f / depthSamples*(i + 1) - 1.0f;

        queueCB.SetSampleMask(1 << i);

        Vertex vertexData[] = {
            { dt::vec3(-1.0, minY, 0.1) },
            { dt::vec3(1.0,  minY, 0.1) },
            { dt::vec3(-1.0, maxY, 0.1) },
            { dt::vec3(1.0, maxY, 0.1) },
        };

        Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), coherent_allocator, vertexData);
        BufferAddress vboAddr = vbo->GetAddress();

        queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, __GL_ARRAYSIZE(vertexData));
    }
    queueCB.SetSampleMask(~0);

    depth.SetDefaults().
        SetDepthTestEnable(LWN_TRUE).
        SetDepthFunc(DepthFunc::LESS).
        SetDepthWriteEnable(LWN_FALSE);
    queueCB.BindDepthStencilState(&depth);

    mask.SetDefaults();
    queueCB.BindChannelMaskState(&mask);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    multisample.SetDefaults().
        SetMultisampleEnable(LWN_TRUE).
        SetRasterSamples(tir ? depthSamples : 0).
        SetSamples(numSamples).
        SetCoverageToColorEnable(LWN_TRUE).
        SetCoverageToColorOutput(coverageOutputIndex).
        SetAlphaToCoverageEnable(alphaToCoverage);
    queueCB.BindMultisampleState(&multisample);

    // Draw a triangle that should fail the depth test for exactly one sample 
    // in each region of the screen. Draw it double sized so we don't have
    // an edge along the diagonal.
    {
        static const Vertex vertexData[] = {
            { dt::vec3(-1.0, -1.0, 0.5) },
            { dt::vec3(3.0, -1.0, 0.5) },
            { dt::vec3(-1.0, 3.0, 0.5) },
        };

        Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), coherent_allocator, vertexData);
        BufferAddress vboAddr = vbo->GetAddress();

        queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, __GL_ARRAYSIZE(vertexData));

        // make sure this scope wait for idle because the memory pool 
        // will be deconstructed when leaving the scopy
        queueCB.submit();
        queue->Finish();
    }

    // render target is the 1X buffer
    singleSampleFB.bind(queueCB);
    singleSampleFB.setViewportScissor();

    // render state should all be default
    multisample.SetDefaults();
    queueCB.BindMultisampleState(&multisample);

    depth.SetDefaults();
    queueCB.BindDepthStencilState(&depth);

    mask.SetDefaults();
    queueCB.BindChannelMaskState(&mask);

    // downsample program
    queueCB.BindProgram(downsamplepgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    typedef struct {
        LWNint numSamples;
    } UniformBlock;

    UniformBlock uboData;
    uboData.numSamples = numSamples;
    Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &uboData, sizeof(uboData),
                                     BUFFER_ALIGN_UNIFORM_BIT, false);

    // Get a handle to be used for setting the buffer as a uniform buffer
    LWNbufferAddress uboAddr = ubo->GetAddress();

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *smp = sb.CreateSampler();

    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(uboData));

    // bind multisample buffer texture with coverage bits
    Texture *tex = multiSampleFB.getColorTexture(coverageOutputIndex, LWN_TRUE);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID()));

    {
        static const Vertex vertexData[] = {
            { dt::vec3(-1.0, -1.0, 0.0) },
            { dt::vec3( 1.0, -1.0, 0.0) },
            { dt::vec3(-1.0,  1.0, 0.0) },
            { dt::vec3( 1.0,  1.0, 0.0) },
        };

        Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), coherent_allocator, vertexData);
        BufferAddress vboAddr = vbo->GetAddress();

        queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, __GL_ARRAYSIZE(vertexData));
    }

    // make sure we sync render to texture
    queueCB.Barrier(LWN_BARRIER_ILWALIDATE_TEXTURE_BIT | LWN_BARRIER_ORDER_FRAGMENTS_BIT);

    // read R32UI texture into CPU accessible memory
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *buffer = coherent_allocator.allocBuffer(&bb, BufferAlignBits(BUFFER_ALIGN_COPY_WRITE_BIT), texWidth * texHeight * numSamples * sizeof(LWNuint));
    LWNuint* data = (LWNuint*)buffer->Map();

    CopyRegion region = { 0, 0, 0, texWidth * numSamples, texHeight, 1 };
    queueCB.CopyTextureToBuffer(singleSampleFB.getColorTexture(0, false), NULL, &region, buffer->GetAddress(), CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();

    // rebind default framebuffer
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
    int x = 0, y = 0;
    int dx = lwrrentWindowWidth / (texWidth*depthSamples);
    int dy = lwrrentWindowHeight / texHeight;
    for (int i = 0; i < texHeight; ++i) {
        for (int j = 0; j < texWidth; ++j) {
            for (int k = 0; k < numSamples; ++k) {
                int coverage = data[(i*texWidth + j)*numSamples + k];
                int was_covered = coverage != 0;
                for (int l = 0; l < numSamples; l++) {
                    queueCB.SetScissor(x + 1, y + 1, dx / numSamples - 1, dy - 1);
                    if (was_covered) {
                        if (coverage & 1) {
                            LWNfloat color[] = { 0, 0.3, k*0.05f, 1 };
                            queueCB.ClearColor(0, color, LWN_CLEAR_COLOR_MASK_RGBA);
                        }
                        else {
                            LWNfloat color[] = { 0, 0.9, k*0.05f, 1 };
                            queueCB.ClearColor(0, color, LWN_CLEAR_COLOR_MASK_RGBA);
                        }
                    }
                    else {
                        LWNfloat color[] = { 0.8, 0.9, 0.8, 1 };
                        queueCB.ClearColor(0, color, LWN_CLEAR_COLOR_MASK_RGBA);
                    }
                    x += dx / numSamples;
                    coverage >>= 1;
                }
            }
        }
        y += dy;
        x = 0;
    }

    queueCB.submit();
    queue->Finish();

    singleSampleFB.destroy();
    multiSampleFB.destroy();
}

    // LWNfragmentCoverageToColor(int samp, bool early, bool mask, int outcol2, bool tir, bool a2c) :
#define C2CTEST(__aa__)                                                                                                    \
    OGTEST_CppTest(LWNfragmentCoverageToColor, lwn_c2c_ ## __aa__ ## xaa,       (__aa__, false, false, 0, false, false));  \
    OGTEST_CppTest(LWNfragmentCoverageToColor, lwn_c2c_ ## __aa__ ## xaa_early, (__aa__, true,  false, 0, false, false));  \
    OGTEST_CppTest(LWNfragmentCoverageToColor, lwn_c2c_ ## __aa__ ## xaa_tir,   (__aa__, false, false, 0, true, false));   \
    OGTEST_CppTest(LWNfragmentCoverageToColor, lwn_c2c_ ## __aa__ ## xaa_a2c,   (__aa__, false, false, 0, false, true));   \
    OGTEST_CppTest(LWNfragmentCoverageToColor, lwn_c2c_ ## __aa__ ## xaa_c2,    (__aa__, false, false, 1, false, false));  \
    OGTEST_CppTest(LWNfragmentCoverageToColor, lwn_c2c_ ## __aa__ ## xaa_m,     (__aa__, false, true,  0, false, false));

C2CTEST(2);
C2CTEST(4);
C2CTEST(8);
