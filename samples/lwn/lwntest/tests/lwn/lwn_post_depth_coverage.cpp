#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNPostDepthCoverage {
public:
    LWNTEST_CppMethods();

private:
    unsigned int    m_columns;
    unsigned int    m_rows;
    int             m_lwrrentTest;

private:
    void drawQuad();
};

static const char vsPassThru[] =
    "precision highp float;\n"
    "layout(location=0) in vec4 pos;\n"
    "layout(location=1) in vec4 col;\n"
    "layout(binding = 0) uniform Block {\n"
    "    float w, h;\n"
    "    int samples;\n"
    "};\n"
    "out vec4 colInterface;\n"
    "void main(void)\n"
    "{\n"
    "   colInterface = col;\n"
    "   vec3 p = vec3(-1.0, -1.0, 0.0) + (vec3(pos) / vec3(w/2, h/2, 1.));\n"
    "   gl_Position = vec4(p, 1.0);\n"
    "}\n";

static const char fsPostDepthCoverageHead[] =
    "precision highp float;\n"
    "layout(early_fragment_tests) in;\n";

static const char fsPostDepthCoverageTail[] =
    "layout(location=0) out vec4 outColor;\n"
    "in vec4 colInterface;\n"
    "layout(binding = 0) uniform Block {\n"
    "    float w, h;\n"
    "    int samples;\n"
    "};\n"
    "void main(void)\n"
    "{\n"
    "  vec4 color = colInterface;\n"
    "  if (gl_SampleMaskIn[0] == ((1 << samples) - 1)) {\n"
    "    color.xy = vec2(0);\n"
    "  }\n"
    "  outColor = color;\n"
    "}\n";

lwString LWNPostDepthCoverage::getDescription() const
{
    return lwString("post_depth_coverage test for LWN\n"
        "(bottom cells): render w/o post_depth_coverage\n"
        "(top cells): render with post_depth_coverage, triangles intersection will have halo\n"
        "Each columns represents sample rate, i.e 2x, 4x, 8x\n");
}

void LWNPostDepthCoverage::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    cellTestInit(3, 2);

    LWNfloat fcolor_clear[] = { 0.5f, 0.5f, 0.5f, 1.0f };
    queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);

    MemoryPoolAllocator coherent_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    struct Vertex {
        dt::vec4 position;
        dt::vec4 color;
    };
    Vertex const vertexData[] = {
        { dt::vec4(0.5f                         , 0.5f                          ,  0.0f, 1.0f), dt::vec4(1.0f, 1.0f, 1.0f, 1.0f) },
        { dt::vec4(lwrrentWindowWidth - 0.5f    , 0.5f                          ,  0.0f, 1.0f), dt::vec4(1.0f, 1.0f, 1.0f, 1.0f) },
        { dt::vec4(0.5f                         , lwrrentWindowHeight - 0.5f    ,  0.0f, 1.0f), dt::vec4(1.0f, 1.0f, 1.0f, 1.0f) },

        { dt::vec4(lwrrentWindowWidth - 0.5f    , 0.5f                          ,  0.5f, 1.0f), dt::vec4(1.0f, 1.0f, 0.5f, 1.0f) },
        { dt::vec4(lwrrentWindowWidth - 0.5f    , lwrrentWindowHeight - 0.5f    ,  1.0f, 1.0f), dt::vec4(1.0f, 1.0f, 0.5f, 1.0f) },
        { dt::vec4(0.5f                         , 0.5f                          , -1.0f, 1.0f), dt::vec4(1.0f, 1.0f, 0.5f, 1.0f) },
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexStreamSet streamSet(stream);
    VertexArrayState vertex = streamSet.CreateVertexArrayState();

    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertexData), coherent_allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults()
      .SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST)
      .SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    Sampler *smp = sb.CreateSampler();

    VertexShader vs(440);
    vs << vsPassThru;

    // emulate a 4x4 matrix for which we lwrrently have no dt support
    typedef struct {
        LWNfloat w, h;
        LWNint samples;
    } UniformBlock;

    // limit to 8x because above we have no real MSAA
    const LWNuint sampleCounts[] = { 2, 4, 8 };

    int col = 0;
    int row = 0;

    DepthStencilState depth;
    MultisampleState multisample;
    depth.SetDefaults();
    multisample.SetDefaults();

    for (unsigned int s = 0; s < __GL_ARRAYSIZE(sampleCounts); s++) {
        LWNint samples = sampleCounts[s];
        for (int post = 0; post < 2; post++) {
            Framebuffer multiSampleFB;

            multiSampleFB.setSize(lwrrentWindowWidth, lwrrentWindowHeight);
            multiSampleFB.setSamples(samples);
            multiSampleFB.setColorFormat(0, Format::RGBA8);
            multiSampleFB.setDepthStencilFormat(Format::DEPTH24);
            multiSampleFB.alloc(device);
            multiSampleFB.bind(queueCB);

            SetCellViewportScissorPadded(queueCB, col, row, 5);

            FragmentShader fs(440);
            fs.addExtension(lwShaderExtension::EXT_post_depth_coverage);
            fs << fsPostDepthCoverageHead << (post ? "layout(post_depth_coverage) in;\n" : "") << fsPostDepthCoverageTail;
            
            // shader program
            Program *pgm = device->CreateProgram();
            LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
            if (!compiled) {
                printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
                return;
            }
            queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

            // preset the framebuffer depth (and clear color)
            LWNfloat fcolor_clear[] = { 0.3, 0.3, 0.3, 1.0 };
            queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
            queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);

            multisample.SetDefaults().
                SetMultisampleEnable(LWN_TRUE).
                SetSamples(samples);
            queueCB.BindMultisampleState(&multisample);
            
            // depth test on
            depth.SetDefaults().
                  SetDepthTestEnable(LWN_TRUE).
                  SetDepthFunc(DepthFunc::LESS).
                  SetDepthWriteEnable(LWN_TRUE);
            queueCB.BindDepthStencilState(&depth);

            UniformBlock uboData;
            uboData.w = (LWNfloat)lwrrentWindowWidth;
            uboData.h = (LWNfloat)lwrrentWindowHeight;
            uboData.samples = samples;

            Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &uboData, sizeof(uboData),
                                             BUFFER_ALIGN_UNIFORM_BIT, false);

            // Get a handle to be used for setting the buffer as a uniform buffer
            LWNbufferAddress uboAddr = ubo->GetAddress();

            queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(uboData));
            queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(uboData));

            // draw two triangles
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, __GL_ARRAYSIZE(vertexData));

            // MS off
            multisample.SetDefaults();
            queueCB.BindMultisampleState(&multisample);

            // depth test off
            depth.SetDefaults();
            queueCB.BindDepthStencilState(&depth);

            // make sure we sync render to texture
            queueCB.Barrier(LWN_BARRIER_ILWALIDATE_TEXTURE_BIT | LWN_BARRIER_ORDER_FRAGMENTS_BIT);

            // draw texture with cell scissor from MS framebuffer to default FB
            g_lwnWindowFramebuffer.bind();

            multiSampleFB.downsample(queueCB);
            Texture *tex = multiSampleFB.getColorTexture(0, false);
            DrawTextureRegion rect = { 0.0f, 0.0f, (float)lwrrentWindowWidth, (float)lwrrentWindowHeight };
            queueCB.DrawTexture(device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID()), &rect, &rect);

            row = (row + 1) % 2;
            if (!row) {
                col++;
            }

            queueCB.submit();
            queue->Finish();

            multiSampleFB.destroy();
        }
    }

}

int LWNPostDepthCoverage::isSupported() const
{
    return g_lwnDeviceCaps.supportsFragmentCoverageToColor && lwogCheckLWNAPIVersion(41, 4);
}

OGTEST_CppTest(LWNPostDepthCoverage, lwn_post_depth_coverage, /* no test args */);
