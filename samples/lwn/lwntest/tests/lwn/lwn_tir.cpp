#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <time.h>

//////////////////////////////////////////////////////////////////////////

using namespace lwn;

class LWNtargetIndependentRasterization
{
public:
    
    LWNtargetIndependentRasterization(int raster_samples, int samples, bool useTable) 
        : m_rasterSamples(raster_samples), m_samples(samples), m_useTable(useTable)
    {}

    LWNTEST_CppMethods();
private:
    static LWNfloat* genLineSphere(int sphslices, int* bufferSize, float scale);

    int     m_rasterSamples;
    int     m_samples;
    bool    m_useTable;
};

int LWNtargetIndependentRasterization::isSupported() const
{
    return g_lwnDeviceCaps.supportsTargetIndependentRasterization && 
           g_lwnDeviceCaps.supportsDrawTexture &&
           lwogCheckLWNAPIVersion(41, 4);
}

lwString LWNtargetIndependentRasterization::getDescription() const
{
    return "Test Target-Independent Rasterization (TIR).\n"
           "This test implements a two-pass additive blending\n"
           "anti-aliasing technique using TIR, which should give\n"
           "comparable results to MSAA with a reduced memory footprint.\n";
}

static const LWNfloat LWN_PI = 3.14154229f;

LWNfloat* LWNtargetIndependentRasterization::genLineSphere(int sphslices, int* bufferSize, float scale)
{
    LWNfloat rho, drho, theta, dtheta;
    LWNfloat x, y, z;
    LWNfloat s, t, ds, dt;
    GLint i, j;

    int count = 0;
    const int slices = sphslices;
    const int stacks = slices;
    LWNfloat* buffer = new LWNfloat[slices * stacks * 4 * 2];
    if (bufferSize) *bufferSize = slices * stacks * 4 * 2 * sizeof(LWNfloat);
    ds = LWNfloat(1.0 / sphslices);;
    dt = LWNfloat(1.0 / sphslices);;
    t = 1.0;
    drho = LWNfloat(LWN_PI / (LWNfloat)stacks);
    dtheta = LWNfloat(2.0 * LWN_PI / (LWNfloat)slices);

    for (i = 0; i < stacks; i++) {
        rho = i * drho;
        s = 0.0;
        for (j = 0; j < slices; j++) { 
            theta = (j == slices) ? LWNfloat(0.0) : LWNfloat(j * dtheta);
            x = -sin(theta) * sin(rho)*scale;
            z = cos(theta) * sin(rho)*scale;
            y = -cos(rho)*scale;
            buffer[count + 0] = x;
            buffer[count + 1] = y;
            buffer[count + 2] = z;
            buffer[count + 3] = 1.0;
            count += 4;

            x = -sin(theta) * sin(rho + drho)*scale;
            z = cos(theta) * sin(rho + drho)*scale;
            y = -cos(rho + drho)*scale;
            buffer[count + 0] = x;
            buffer[count + 1] = y;
            buffer[count + 2] = z;
            buffer[count + 3] = 1.0;
            count += 4;

            s += ds;
        }
        t -= dt;
    }
    return buffer;
}

void LWNtargetIndependentRasterization::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vsTexture(440);
    vsTexture <<
        "layout(location = 0) in vec4 position;\n"
        "out vec2 texcoord;\n"
        "void main() {\n"
        "  gl_Position = position;\n"
        "  texcoord = (vec2(position) + vec2(1.0, 1.0)) * vec2(0.5, 0.5);\n"
        "}\n";

    FragmentShader fs_solid(440);
    fs_solid << "precision highp float;\n"
        "layout(location = 0) out vec4 color;\n"
        "in vec2 texcoord;\n"
        "void main() {\n"
        "    color = vec4(0.0, 1.0, 0.0 , 1.0);\n"
        "}\n";

    // shader program
    Program *pgm = device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vsTexture, fs_solid);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    int bufferSize;
    LWNfloat* lines = genLineSphere(30, &bufferSize, 0.9);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec4 position;
    };

    MemoryPoolAllocator vertex_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream lineStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(lineStream, Vertex, position);
    VertexStreamSet lineStreamSet(lineStream);
    VertexArrayState line = lineStreamSet.CreateVertexArrayState();

    Buffer *vboLines = lineStream.AllocateVertexBuffer(device, bufferSize / (4 * sizeof(float)), vertex_allocator, lines);
    BufferAddress vboLinesAddr = vboLines->GetAddress();

    delete lines;

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    Sampler *sampler = sb.CreateSampler();

    const int texWidth = lwrrentWindowWidth, texHeight = lwrrentWindowHeight;

    // setup a single sampled color buffer
    TextureBuilder textureBuilderColor;
    textureBuilderColor.SetDevice(device).SetDefaults().
        SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(texWidth, texHeight).
        SetFormat(Format::RGBA8).
        SetFlags(TextureFlags::COMPRESSIBLE);
    const LWNuintptr poolSizeColor = textureBuilderColor.GetStorageSize() + textureBuilderColor.GetStorageAlignment();


    LWNuintptr poolSizeColorMS = 0;
    TextureBuilder textureBuilderColorMS;
    // allocate MS color if needed
    if (m_samples > 1) {
        textureBuilderColorMS.SetDevice(device).SetDefaults().
            SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE).
            SetSamples(m_samples).
            SetSize2D(texWidth, texHeight).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags::COMPRESSIBLE);
        poolSizeColorMS = textureBuilderColorMS.GetStorageSize() + textureBuilderColorMS.GetStorageAlignment();
    }

    // and a multisampled depth buffer
    TextureBuilder textureBuilderDepth;
    textureBuilderDepth.SetDevice(device).SetDefaults().
        SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE).
        SetSize2D(texWidth, texHeight).
        SetFormat(Format::DEPTH24_STENCIL8).
        SetSamples(m_rasterSamples).
        SetFlags(TextureFlags::COMPRESSIBLE);
    const LWNuintptr poolSizeDepth = textureBuilderDepth.GetStorageSize() + textureBuilderDepth.GetStorageAlignment();

    MemoryPoolAllocator gpu_allocator(device, NULL, poolSizeColor + poolSizeDepth + ((m_samples > 1) ? poolSizeColorMS : 0), LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    Texture *tex = gpu_allocator.allocTexture(&textureBuilderColor);
    LWNuint textureID = tex->GetRegisteredTextureID();
    TextureHandle texHandle = device->GetTextureHandle(textureID, sampler->GetRegisteredID());

    Texture *texMS = NULL;
    if (m_samples > 1) {
        texMS = gpu_allocator.allocTexture(&textureBuilderColorMS);
    }

    Texture *texDepth = gpu_allocator.allocTexture(&textureBuilderDepth);

    // render to full texture
    queueCB.SetViewportScissor(0, 0, texWidth, texHeight);

    // bind  texture as render target
    if (m_samples > 1) {
        queueCB.SetRenderTargets(1, &texMS, NULL, texDepth, NULL);
    }
    else {
        queueCB.SetRenderTargets(1, &tex, NULL, texDepth, NULL);
    }

    // MS state with raster samples and samples if als multi-sampling
    MultisampleState multisample;
    multisample.SetDefaults().
        SetRasterSamples(m_rasterSamples).
        SetCoverageModulationMode(CoverageModulationMode::RGBA);
    if (m_samples > 1) {
        multisample.SetSamples(m_samples);
    }

    // check getter for consistency in the API
    if ((multisample.GetCoverageModulationMode() != CoverageModulationMode::RGBA) ||
        (multisample.GetRasterSamples() != m_rasterSamples) ||
        (multisample.GetSamples() != ((m_samples > 1) ? m_samples : 0)) ) {
        return;
    }

    // MS state will be the same for the rest of the test
    queueCB.BindMultisampleState(&multisample);
    
    // bind the modulation table we want
    if (m_useTable) {
        LWNfloat entries[16] = { 0 };
        // set all entries to half what the linear setting would be
        for (int i = 1; i <= m_rasterSamples; i++) {
            entries[((i * __GL_ARRAYSIZE(entries)) / m_rasterSamples) - 1] = (float)i / (float)(m_rasterSamples);
        }
        // but set fully covered to 0.25 to make this dim
        entries[__GL_ARRAYSIZE(entries) - 1] = 0.25;
        queueCB.BindCoverageModulationTable(entries);
    } else {
        // NULL would imply linear.
        queueCB.BindCoverageModulationTable(NULL);
    }

    // clear something 
    LWNfloat fcolor_clear[] = { 0, 0, 0, 0 };
    queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);

    // make it big
    queueCB.SetLineWidth(3.0);

    // program that black/white figure
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    // ###############
    // FIRST PASS

    // disable color writes
    ChannelMaskState mask;
    mask.SetDefaults().
         SetChannelMask(0, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE);
    queueCB.BindChannelMaskState(&mask);

    // enable depth writes
    DepthStencilState depth;
    depth.SetDefaults().
        SetDepthTestEnable(LWN_TRUE).
        SetDepthFunc(DepthFunc::LESS);
    queueCB.BindDepthStencilState(&depth);

    // disable blend
    ColorState cs;
    cs.SetDefaults();
    cs.SetBlendEnable(0, false);
    queueCB.BindColorState(&cs);

    // draw scene pass1
    // writing to depth buffer with raster_samples
    queueCB.BindVertexBuffer(0, vboLinesAddr, bufferSize);
    queueCB.BindVertexArrayState(line);
    queueCB.DrawArrays(DrawPrimitive::LINES, 0, bufferSize / (4 * sizeof(float)));

    // ###############
    // SECOND PASS

    // setup blend state for 2nd pass
    BlendState bs;
    bs.SetDefaults();
    bs.SetBlendTarget(0);
    bs.SetBlendFunc(BlendFunc::ONE, BlendFunc::ONE, BlendFunc::ONE, BlendFunc::ONE);
    bs.SetBlendEquation(BlendEquation::ADD, BlendEquation::ADD);
    queueCB.BindBlendState(&bs);

    // enable color writes
    mask.SetDefaults().
        SetChannelMask(0, LWN_TRUE, LWN_TRUE, LWN_TRUE, LWN_TRUE);
    queueCB.BindChannelMaskState(&mask);

    // disable depth writes
    depth.SetDefaults().
        SetDepthTestEnable(LWN_TRUE).
        SetDepthWriteEnable(LWN_FALSE).
        SetDepthFunc(DepthFunc::LEQUAL);
    queueCB.BindDepthStencilState(&depth);

    // enable blending
    cs.SetBlendEnable(0, true);
    queueCB.BindColorState(&cs);

    // draw scene again
    queueCB.DrawArrays(DrawPrimitive::LINES, 0, bufferSize / (4 * sizeof(float)));

    // downsample if MS > 1 samples
    if (m_samples > 1) {
        queueCB.Downsample(texMS, tex);
    }

    // rebind default framebuffer
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    // draw texture honors blend state, so shut it off.
    cs.SetDefaults();
    cs.SetBlendEnable(0, false);
    queueCB.BindColorState(&cs);

    // Insert a fragment barrier to ensure that prior rendering is complete and
    // the texture cache is ilwalidated.  Failing to do so results in
    // intermittent blocky corruption in some of these tests (bug 200780992).
    queueCB.Barrier(BarrierBits::ILWALIDATE_TEXTURE | BarrierBits::ORDER_FRAGMENTS_TILED);

    DrawTextureRegion rect = { 0.0f, 0.0f, (float)lwrrentWindowWidth, (float)lwrrentWindowHeight };
    queueCB.DrawTexture(texHandle, &rect, &rect);

    queueCB.submit();
    queue->Finish();
}

#define TIRTEST(N, M)                                                                                               \
    OGTEST_CppTest(LWNtargetIndependentRasterization, lwn_tir_blend_rs ## N ## x_aa ## M ## x, (N, M, false));      \
    OGTEST_CppTest(LWNtargetIndependentRasterization, lwn_tir_blend_rs ## N ## x_aa ## M ## x_table, (N, M, true));

TIRTEST(2, 1); 
TIRTEST(4, 1);
TIRTEST(8, 1);
TIRTEST(4, 2);
TIRTEST(8, 2);
TIRTEST(8, 4);
