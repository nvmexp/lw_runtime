#include "lwntest_cpp.h"
#include "lwn_utils.h"

//////////////////////////////////////////////////////////////////////////

using namespace lwn;

class LWNsampleLocation {
public:
    LWNsampleLocation(int samp, void(*f)(MultisampleState *ms), bool d) : numSamples(samp), disableMS(d), sampleLocFunc(f) {}
    LWNTEST_CppMethods();

private:
    static const int w = 64;
    static const int h = 64;

    int numSamples;
    bool disableMS;
    void(*sampleLocFunc)(MultisampleState *ms);
};

static const dt::vec3 white(1, 1, 1);
static const float colors[4][2][3] = {
    { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } },
    { { 0.0, 0.0, 1.0 }, { 1.0, 1.0, 0.0 } },
    { { 1.0, 0.0, 1.0 }, { 0.0, 1.0, 1.0 } },
    { { 1.0, 1.0, 1.0 }, { 0.5, 0.5, 0.5 } }
};

lwString LWNsampleLocation::getDescription() const
{
    return "Exercise LW_sample_locations. Programs a sample pattern (regular 4x4 grid, standard 16xAA sample locations, "
        "or smiley face) and draws a different aamode test pattern in each column (sample finder, horizontal, vertical, sliver). "
        "Bottom row has programmable locations disabled. Middle row has programmable sample locations enabled but not pixel-varying, "
        "Top row has pixel varying programmable locations.";
}

int LWNsampleLocation::isSupported() const
{ 
    return g_lwnDeviceCaps.supportsSampleLocations;
}

#define SAMPLEPOS(x,y,i) locations[2*i+0] = x/16.f; locations[2*i+1] = y/16.f;

static void SampleLocations16xGrid(MultisampleState *ms)
{
    float locations[16 * 2];

    SAMPLEPOS(9, 9, 0)
    SAMPLEPOS(7, 5, 1)
    SAMPLEPOS(5, 10, 2)
    SAMPLEPOS(12, 7, 3)
    SAMPLEPOS(3, 6, 4)
    SAMPLEPOS(10, 13, 5)
    SAMPLEPOS(13, 11, 6)
    SAMPLEPOS(11, 3, 7)
    SAMPLEPOS(6, 14, 8)
    SAMPLEPOS(8, 1, 9)
    SAMPLEPOS(4, 2, 10)
    SAMPLEPOS(2, 12, 11)
    SAMPLEPOS(0, 8, 12)
    SAMPLEPOS(15, 4, 13)
    SAMPLEPOS(14, 15, 14)
    SAMPLEPOS(1, 0, 15)

    ms->SetSampleLocations(0, __GL_ARRAYSIZE(locations) / 2, locations);
}

static void SampleLocations16xReg(MultisampleState *ms)
{
    float locations[16 * 2];

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            locations[2 * (i + 4 * j) + 0] = (2 * i + 1.0f) / 8.0f;
            locations[2 * (i + 4 * j) + 1] = (2 * j + 1.0f) / 8.0f;
        }
    }

    ms->SetSampleLocations(0, __GL_ARRAYSIZE(locations) / 2, locations);
}

static void SampleLocationsFace(MultisampleState *ms)
{
    float locations[16 * 2];

    locations[2 * 0 + 0] = 0.35f; locations[2 * 0 + 1] = 0.75f;
    locations[2 * 1 + 0] = 0.65f; locations[2 * 1 + 1] = 0.75f;
    locations[2 * 2 + 0] = 0.25f; locations[2 * 2 + 1] = 0.35f;
    locations[2 * 3 + 0] = 0.75f; locations[2 * 3 + 1] = 0.35f;

    locations[2 * 4 + 0] = 5.f / 16; locations[2 * 4 + 1] = 0.30f;
    locations[2 * 5 + 0] = 6.f / 16; locations[2 * 5 + 1] = 0.25f;
    locations[2 * 6 + 0] = 7.f / 16; locations[2 * 6 + 1] = 0.25f;
    locations[2 * 7 + 0] = 8.f / 16; locations[2 * 7 + 1] = 0.25f;

    locations[2 * 8 + 0] = 9.f / 16; locations[2 * 8 + 1] = 0.25f;
    locations[2 * 9 + 0] = 10.f / 16; locations[2 * 9 + 1] = 0.25f;
    locations[2 * 10 + 0] = 11.f / 16; locations[2 * 10 + 1] = 0.30f;
    locations[2 * 11 + 0] = 8.f / 16; locations[2 * 11 + 1] = 7.f / 16;

    locations[2 * 12 + 0] = 8.f / 16; locations[2 * 12 + 1] = 8.f / 16;
    locations[2 * 13 + 0] = 8.f / 16; locations[2 * 13 + 1] = 9.f / 16;
    locations[2 * 14 + 0] = 8.f / 16; locations[2 * 14 + 1] = 10.f / 16;
    locations[2 * 15 + 0] = 8.f / 16; locations[2 * 15 + 1] = 11.f / 16;

    ms->SetSampleLocations(0, __GL_ARRAYSIZE(locations) / 2, locations);
}

void LWNsampleLocation::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    LWNfloat mvp[16] = {
        2.f / w, 0, 0, 0,
        0, 2.f / h, 0, 0,
        0, 0, 1, 0,
        -1, -1, 0, 1
    };

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec3 pos;\n"
        "layout(location = 1) in vec3 col;\n"
        "layout(binding = 0) uniform Block {\n"
        "    mat4 uMVP;\n"
        "};\n"
        "out vec4 color;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    gl_Position = uMVP * vec4(pos, 1.0);\n"
        "    color = vec4(col, 1.0);\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "precision highp float;"
        "layout(location=0) out vec4 outColor;\n"
        "in vec4 color;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    outColor = color;\n"
        "}\n";

    // shader program
    Program *pgm = device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    MemoryPoolAllocator coherent_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    Framebuffer multiSampleFB;

    if (numSamples) {
        multiSampleFB.setSize(lwrrentWindowWidth, lwrrentWindowHeight);
        multiSampleFB.setSamples(numSamples);
        multiSampleFB.setColorFormat(0, Format::RGBA8);
        multiSampleFB.alloc(device);
        multiSampleFB.bind(queueCB);
        multiSampleFB.setViewportScissor();
    }

    LWNfloat fcolor_clear[] = { 0, 0, 0, 1 };
    queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);

    typedef struct {
        LWNfloat uMVP[16];
    } UniformBlock;

    UniformBlock uboData;
    memcpy(&uboData.uMVP, &mvp, sizeof(mvp));

    Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &uboData, sizeof(uboData),
                                     BUFFER_ALIGN_UNIFORM_BIT, false);

    // Get a handle to be used for setting the buffer as a uniform buffer
    LWNbufferAddress uboAddr = ubo->GetAddress();

    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(uboData));

    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    queueCB.BindVertexArrayState(vertex);

    MultisampleState multisample;
    multisample.SetDefaults().
        SetMultisampleEnable(disableMS ? LWN_FALSE : LWN_TRUE).
        SetSamples(numSamples);

    LWNint gridx, gridy;
    multisample.GetSampleLocationsGrid(&gridx, &gridy);

    for (int celly = 0; celly < 3; ++celly) {

        // celly 0 -> location diabled, grid disabled
        // celly 1 -> location enabled, grid disabled
        // celly 2 -> location enabled, grid enabled
        multisample.SetSampleLocationsEnable(celly ? LWN_TRUE : LWN_FALSE);
        multisample.SetSampleLocationsGridEnable((celly & 2) ? LWN_TRUE : LWN_FALSE);

        // fill in sample locations table
        sampleLocFunc(&multisample);

        queueCB.BindMultisampleState(&multisample);

        {
            queueCB.SetViewportScissor(0, celly*h, w, h);

            const int numVertices = gridx * gridy * 4 * w * h;

            Buffer *vbo = stream.AllocateVertexBuffer(device, numVertices, coherent_allocator, NULL);
            Vertex *current = (Vertex *)vbo->Map();
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < h; j++) {

                    float x = i + (float)i / w;
                    float y = j + (float)j / h;
                    float eps = 0.5f / w;

                    // Draw the sample-finder triangle in a gx X gy grid of pixels, to find
                    // sample locations in the neighbors. Color based on grid element.
                    for (int gx = 0; gx < gridx; ++gx) {
                        for (int gy = 0; gy < gridy; ++gy) {
                            dt::vec3 col(colors[gy][gx & 1][0], colors[gy][gx & 1][1], colors[gy][gx & 1][2]);
                            if (numSamples) {
                                col = col * dt::vec3(float(numSamples), float(numSamples), float(numSamples));
                            }
                            *current++ = { dt::vec3(x + gx - eps, y + gy - eps, 0.0), col };
                            *current++ = { dt::vec3(x + gx + eps, y + gy - eps, 0.0), col };
                            *current++ = { dt::vec3(x + gx + eps, y + gy + eps, 0.0), col };
                            *current++ = { dt::vec3(x + gx - eps, y + gy + eps, 0.0), col };
                        }
                    }
                }
            }

            BufferAddress vboAddr = vbo->GetAddress();
            queueCB.BindVertexBuffer(0, vboAddr, sizeof(Vertex)*numVertices);
            queueCB.DrawArrays(DrawPrimitive::QUADS, 0, numVertices);
        }

        {
            queueCB.SetViewportScissor(1 * w, celly*h, w, h);

            const int numVertices = h * 3;
            Buffer *vbo = stream.AllocateVertexBuffer(device, numVertices, coherent_allocator, NULL);
            Vertex *current = (Vertex*)vbo->Map();
            for (int i = 0; i < h; i++) {
                *current++ = { dt::vec3(0, i, 0)     , white };
                *current++ = { dt::vec3(w, i, 0)     , white };
                *current++ = { dt::vec3(0, i + 1, 0) , white };
            }
            BufferAddress vboAddr = vbo->GetAddress();
            queueCB.BindVertexBuffer(0, vboAddr, sizeof(Vertex)*numVertices);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, numVertices);
        }

        {
            queueCB.SetViewportScissor(2 * w, celly*h, w, h);

            const int numVertices = w * 3;
            Buffer *vbo = stream.AllocateVertexBuffer(device, numVertices, coherent_allocator, NULL);
            Vertex *current = (Vertex*)vbo->Map();
            for (int i = 0; i < w; i++) {
                *current++ = { dt::vec3(i, 0, 0)     , white };
                *current++ = { dt::vec3(i + 1, 0, 0) , white };
                *current++ = { dt::vec3(i + 1, h, 0) , white };
            }
            BufferAddress vboAddr = vbo->GetAddress();
            queueCB.BindVertexBuffer(0, vboAddr, sizeof(Vertex)*numVertices);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, numVertices);
        }

        {
            queueCB.SetViewportScissor(3 * w, celly*h, w, h);

            float x = 0.0;
            float s = 1. + 1. / (float)w;

            const int numVertices = w * 3;
            Buffer *vbo = stream.AllocateVertexBuffer(device, numVertices, coherent_allocator, NULL);
            Vertex *current = (Vertex*)vbo->Map();
            do {
                *current++ = { dt::vec3(x, 0, 0)          , white };
                *current++ = { dt::vec3(x + s, 0, 0)      , white };
                *current++ = { dt::vec3(x + s / 2., h, 0) , white };
                x += s;
            } while (x < (float)w);

            BufferAddress vboAddr = vbo->GetAddress();
            queueCB.BindVertexBuffer(0, vboAddr, sizeof(Vertex)*numVertices);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, numVertices);
        }
    }

    if (numSamples) {
        // rebind default framebuffer
        g_lwnWindowFramebuffer.bind();
        g_lwnWindowFramebuffer.setViewportScissor();

        multisample.SetDefaults();
        queueCB.BindMultisampleState(&multisample);

        multiSampleFB.downsample(queueCB);
        Texture *tex = multiSampleFB.getColorTexture(0, LWN_FALSE);

        SamplerBuilder sb;
        sb.SetDevice(device).
            SetDefaults().
            SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST).
            SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
        Sampler *smp = sb.CreateSampler();

        DrawTextureRegion rect = { 0.0f, 0.0f, (float)lwrrentWindowWidth, (float)lwrrentWindowHeight };
        queueCB.DrawTexture(device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID()), &rect, &rect);
    }
    queueCB.submit();
    queue->Finish();

    multiSampleFB.destroy();
}

#define SAMPLELOC(num) \
    OGTEST_CppTest(LWNsampleLocation, lwn_sample_location_fbo##num##_pattern16, (num, SampleLocations16xGrid, false)); \
    OGTEST_CppTest(LWNsampleLocation, lwn_sample_location_fbo##num##_regular16, (num, SampleLocations16xReg, false)); \
    OGTEST_CppTest(LWNsampleLocation, lwn_sample_location_fbo##num##_face, (num, SampleLocationsFace, false));

SAMPLELOC(2)
SAMPLELOC(4)
SAMPLELOC(8)

OGTEST_CppTest(LWNsampleLocation, lwn_sample_location_window_pattern16, (0, SampleLocations16xGrid, false));
OGTEST_CppTest(LWNsampleLocation, lwn_sample_location_window_regular16, (0, SampleLocations16xReg, false));
OGTEST_CppTest(LWNsampleLocation, lwn_sample_location_window_face, (0, SampleLocationsFace, false));
OGTEST_CppTest(LWNsampleLocation, lwn_sample_location_fbo8_pattern16_msdisable, (8, SampleLocations16xGrid, true));



class LWNdecompressZ {
public:
    LWNdecompressZ(int samp, bool scis) : numSamples(samp), scissor(scis) {}
    LWNTEST_CppMethods();

private:
    int numSamples;
    bool scissor;
};


lwString LWNdecompressZ::getDescription() const
{
    return "Exercise depth decompress-in-place. Render a slope into the depth buffer, "
        "then render two additional passes blended together: (1) draw red with sample "
        "locations moved such that the depth test should fail if depth values are "
        "preserved, (2) draw green with different geometry such that the depth test "
        "should pass. The resulting image is considered a pass if fully green.";
}

int LWNdecompressZ::isSupported() const
{ 
    return  g_lwnDeviceCaps.supportsSampleLocations;
}

static void SampleLocationsXAxis(MultisampleState *ms, bool adjustRight)
{
    float locations[16 * 2];

    for (int i = 0; i < 16; ++i) {
        locations[2 * i + 0] = adjustRight ? 15 / 16.f : 1 / 16.f;
        locations[2 * i + 1] = 0.5f;
    }

    ms->SetSampleLocations(0, __GL_ARRAYSIZE(locations) / 2, locations);
}

void LWNdecompressZ::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec3 pos;\n"
        "layout(location = 1) in vec3 col;\n"
        "out vec4 color;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(pos, 1.0);\n"
        "    color = vec4(col, 1.0);\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "precision highp float;"
        "layout(location=0) out vec4 outColor;\n"
        "in vec4 color;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    outColor = color;\n"
        "}\n";

    // shader program
    Program *pgm = device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    MemoryPoolAllocator coherent_allocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    Framebuffer multiSampleFB;

    if (numSamples) {
        multiSampleFB.setSize(lwrrentWindowWidth, lwrrentWindowHeight);
        multiSampleFB.setSamples(numSamples);
        multiSampleFB.setColorFormat(0, Format::RGBA8);
        multiSampleFB.setDepthStencilFormat(Format::DEPTH24);
        multiSampleFB.alloc(device);
        multiSampleFB.bind(queueCB);
        multiSampleFB.setViewportScissor();
    }

    LWNfloat fcolor_clear[] = { 0, 0, 0, 1 };
    queueCB.ClearColor(0, fcolor_clear, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);

    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    queueCB.BindVertexArrayState(vertex);

    MultisampleState multisample;
    multisample.SetDefaults().
        SetMultisampleEnable(true).
        SetSamples(numSamples).
        SetSampleLocationsEnable(LWN_TRUE);

    SampleLocationsXAxis(&multisample, false);

    queueCB.BindMultisampleState(&multisample);

    BlendState bs;
    bs.SetDefaults();
    bs.SetBlendTarget(0);
    bs.SetBlendFunc(BlendFunc::ONE, BlendFunc::ONE, BlendFunc::ONE, BlendFunc::ONE);
    bs.SetBlendEquation(BlendEquation::ADD, BlendEquation::ADD);
    queueCB.BindBlendState(&bs);

    ColorState cs;
    cs.SetDefaults();
    cs.SetBlendEnable(0, true);
    queueCB.BindColorState(&cs);

    ChannelMaskState mask;
    mask.SetDefaults().
        SetChannelMask(0, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE);
    queueCB.BindChannelMaskState(&mask);

    DepthStencilState depth;
    depth.SetDefaults().
        SetDepthTestEnable(LWN_TRUE).
        SetDepthFunc(DepthFunc::ALWAYS);
    queueCB.BindDepthStencilState(&depth);

    Vertex quad[8] = {
        // With new locations, the same geometry should fail the depth test,
        // so no red should be drawn.
        { dt::vec3(-1, -1, -0.8), dt::vec3(1, 0, 0) }, // RED
        { dt::vec3(1, -1, -0.2), dt::vec3(1, 0, 0) },
        { dt::vec3(1, 1, -0.2), dt::vec3(1, 0, 0) },
        { dt::vec3(-1, 1, -0.8), dt::vec3(1, 0, 0) },

        // Move the geometry forward, so it will pass the depth test even
        // with the new sample locations.
        { dt::vec3(-1, -1, -0.9), dt::vec3(0, 1, 0) }, // GREEN
        { dt::vec3(1, -1, -0.3), dt::vec3(0, 1, 0) },
        { dt::vec3(1, 1, -0.3), dt::vec3(0, 1, 0) },
        { dt::vec3(-1, 1, -0.9), dt::vec3(0, 1, 0) }
    };
    Buffer *vbo = stream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(quad), coherent_allocator, quad);
    BufferAddress vboAddr = vbo->GetAddress();
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(quad));

    // draw first 4 vertices as quad
    queueCB.DrawArrays(DrawPrimitive::QUADS, 0, 4);

    if (scissor) {
        queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth / 2, lwrrentWindowHeight / 2);
    }

    // Resolve the depth values into the depth buffer
    queueCB.ResolveDepthBuffer();

    if (scissor) {
        if (numSamples) {
            multiSampleFB.setViewportScissor();
        } else {
            g_lwnWindowFramebuffer.setViewportScissor();
        }
    }

    mask.SetDefaults().
        SetChannelMask(0, LWN_TRUE, LWN_TRUE, LWN_TRUE, LWN_TRUE);
    queueCB.BindChannelMaskState(&mask);

    depth.SetDefaults().
        SetDepthTestEnable(LWN_TRUE).
        SetDepthWriteEnable(LWN_TRUE).
        SetDepthFunc(DepthFunc::LEQUAL);
    queueCB.BindDepthStencilState(&depth);

    cs.SetBlendEnable(0, false);
    queueCB.BindColorState(&cs);

    // Move the sample locations to the right
    SampleLocationsXAxis(&multisample, true);

    queueCB.BindMultisampleState(&multisample);

    // draw all 8 vertices as quad
    queueCB.DrawArrays(DrawPrimitive::QUADS, 0, __GL_ARRAYSIZE(quad));

    if (numSamples) {
        // rebind default framebuffer
        g_lwnWindowFramebuffer.bind();
        g_lwnWindowFramebuffer.setViewportScissor();

        multisample.SetDefaults();
        queueCB.BindMultisampleState(&multisample);

        multiSampleFB.downsample(queueCB);
        Texture *tex = multiSampleFB.getColorTexture(0, LWN_FALSE);

        SamplerBuilder sb;
        sb.SetDevice(device).
            SetDefaults().
            SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST).
            SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
        Sampler *smp = sb.CreateSampler();

        DrawTextureRegion rect = { 0.0f, 0.0f, (float)lwrrentWindowWidth, (float)lwrrentWindowHeight };
        queueCB.DrawTexture(device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID()), &rect, &rect);
    }

    queueCB.submit();
    queue->Finish();
    multiSampleFB.destroy();
}

#define DIPTEST(N)  \
    OGTEST_CppTest(LWNdecompressZ, lwn_decompz##N, (N, false));          \
    OGTEST_CppTest(LWNdecompressZ, lwn_decompzscissor##N, (N, true));    \

DIPTEST(0)
DIPTEST(2)
DIPTEST(4)
DIPTEST(8)
