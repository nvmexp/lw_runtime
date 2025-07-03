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

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

#ifndef ROUND_UP
    #define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#endif

// Disabled for now as it evidently seems broken.
//#define TEST_RGB565 

using namespace lwn;
using namespace lwn::dt;

// ----------------------------------- LWNTexturePitch ------------------------------------------

class LWNTexturePitch {

    // The pitch linear texture we are testing, with its memory pool.
    MemoryPool *m_pool;
    Texture *m_texture;
    size_t m_textureSize;
    bool m_textureInit;
    LWNint m_textureStrideAlignment;
    LWNint m_textureStrideAlignmentRT;

    // Rest of the stuff we need to draw the texture.
    MemoryPoolAllocator *m_bufpool;
    Program *m_program;
    Program *m_programRT;
    Sampler *m_sampler;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    LWNuint m_vertexDataSize;
    Sync *m_sync;

    struct Vertex {
        vec3 position;
        vec3 uv;
    };

public:
    bool setTexture(int width, int height, Format fmt, bool fill = true, bool rendertarget = true);
    bool init();
    void draw();
    void renderToTexture(int width, int height, Format fmt);
    LWNTexturePitch();
    ~LWNTexturePitch();
};

LWNTexturePitch::LWNTexturePitch()
{
    m_textureInit = false;
}

LWNTexturePitch::~LWNTexturePitch()
{
    delete m_bufpool;
    m_texture->Free();
    m_pool->Free();
    m_sync->Free();
}

bool LWNTexturePitch::setTexture(int width, int height, Format fmt, bool fill, bool rendertarget)
{
    Device *device = DeviceState::GetActive()->getDevice();

    if (m_textureInit) {
        m_texture->Free();
        m_pool->Free();
    }
    m_textureInit = true;

    LWNint strideAlign = rendertarget ? m_textureStrideAlignmentRT : m_textureStrideAlignment;
    size_t pitch = 0;
    if (fmt == Format::RGBA8) {
        pitch = ROUND_UP(4 * width, strideAlign);
    } else if (fmt == Format::RGBA16) {
        pitch = ROUND_UP(4 * sizeof(unsigned short) * width, strideAlign);
    } else if (fmt == Format::RGBA32F) {
        pitch = ROUND_UP(sizeof(vec4) * width, strideAlign);
    } else if (fmt == Format::RGB565) {
        pitch = ROUND_UP(sizeof(vec3_rgb565) * width, strideAlign);
    } else if (fmt == Format::RGB10A2) {
        pitch = ROUND_UP(sizeof(vec4_rgb10a2) * width, strideAlign);
    }
    assert(pitch);

    // Create pitch-linear texture.
    DEBUG_PRINT(("LWNTexturePitch:: Creating %dx%d pitch linear texture...\n", width, height));
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device);
    textureBuilder.SetDefaults()
                .SetTarget(TextureTarget::TARGET_2D) // TODO: add coverage for TARGET_RECTANGLE.
                .SetFormat(fmt)
                .SetSize2D(width, height)
                .SetStride(LWNuint(pitch))
                .SetFlags(TextureFlags::LINEAR);
    m_textureSize = textureBuilder.GetStorageSize();
    DEBUG_PRINT(("LWNTexturePitch:: Allocating textureSize %zu\n", m_textureSize));
    m_pool  = device->CreateMemoryPool(NULL, m_textureSize, MemoryPoolType::CPU_NON_COHERENT);
    assert((LWNmemoryPool *) m_pool);
    m_texture = textureBuilder.CreateTextureFromPool(m_pool, 0);

    unsigned char *ptr = (unsigned char *) m_pool->Map();
    assert(ptr);

    if (fill) {
        memset(ptr, 0, m_textureSize);

        // Fill the texture with a simple gradient.
        DEBUG_PRINT(("LWNTexturePitch:: Filling texture...\n"));

        if (fmt == Format::RGBA8) {
            unsigned char *_ptr = (unsigned char *) ptr;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x ++) {
                    _ptr[x * 4 + 0] = (int)(255.0f * (float) y / height);
                    _ptr[x * 4 + 1] = (int)(255.0f * (float) x / width);
                    _ptr[x * 4 + 2] = 0;
                    _ptr[x * 4 + 3] = 255;
                }
                _ptr = ((unsigned char*) _ptr) + pitch;
            }
        } else if (fmt == Format::RGBA16) {
            unsigned short *_ptr = (unsigned short *) ptr;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x ++) {
                    _ptr[x * 4 + 0] = (int)(65535.0f * (float) y / height);
                    _ptr[x * 4 + 1] = (int)(65535.0f * (float) x / width);
                    _ptr[x * 4 + 2] = 0;
                    _ptr[x * 4 + 3] = 65535;
                }
                _ptr = (unsigned short *) (((unsigned char*) _ptr) + pitch);
            }
        } else if (fmt == Format::RGBA32F) {
            vec4 *_ptr = (vec4 *) ptr;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x ++) {
                    _ptr[x].setX((float) y / height);
                    _ptr[x].setY((float) x / width);
                    _ptr[x].setZ(0.0f);
                    _ptr[x].setW(1.0f);
                }
                _ptr = (vec4 *) (((unsigned char*) _ptr) + pitch);
            }
        } else if (fmt == Format::RGB565) {
            vec3_rgb565 *_ptr = (vec3_rgb565 *) ptr;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x ++) {
                    _ptr[x].setX((float) y / height);
                    _ptr[x].setY((float) x / width);
                    _ptr[x].setZ(0.0f);
                }
                _ptr = (vec3_rgb565 *) (((unsigned char*) _ptr) + pitch);
            }
        } else if (fmt == Format::RGB10A2) {
            vec4_rgb10a2 *_ptr = (vec4_rgb10a2 *) ptr;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x ++) {
                    _ptr[x].setX((float) y / height);
                    _ptr[x].setY((float) x / width);
                    _ptr[x].setZ(0.0f);
                    _ptr[x].setW(1.0f);
                }
                _ptr = (vec4_rgb10a2 *) (((unsigned char*) _ptr) + pitch);
            }
        } else {
            assert(!"Unknown format.");
        }

        m_pool->FlushMappedRange(0, m_textureSize);
    }

    return true;
}


bool LWNTexturePitch::init() {
    DEBUG_PRINT(("LWNTexturePitch:: Creating other test assets...\n"));

    Device *device = DeviceState::GetActive()->getDevice();

    m_bufpool = new MemoryPoolAllocator(device, NULL, 0x10000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

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
        "layout (binding=0) uniform sampler2D tex;"
        "void main() {\n"
        "  fcolor = textureLod(tex, ouv, 0.0);\n"
        "}\n";
    FragmentShader fs2(440);
    fs2 <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(0.23, ouv.x, ouv.y, 1.0) + step(sin((ouv.x + ouv.y) * 24.0), -0.75);\n"
        "}\n";

    m_program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return false;
    }

    m_programRT = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programRT, vs, fs2)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return false;
    }

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
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults()
      .SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    m_sampler = sb.CreateSampler();
    device->GetInteger(DeviceInfo::LINEAR_TEXTURE_STRIDE_ALIGNMENT, &m_textureStrideAlignment);
    device->GetInteger(DeviceInfo::LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT, &m_textureStrideAlignmentRT);
    m_sync = device->CreateSync();

    return true;
}

void LWNTexturePitch::draw()
{
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    TextureHandle texHandle = device->GetTextureHandle(m_texture->GetRegisteredTextureID(), m_sampler->GetRegisteredID());
    assert(texHandle);

    queueCB.BindProgram(m_program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

void LWNTexturePitch::renderToTexture(int width, int height, Format fmt)
{
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();
    Queue *queue = DeviceState::GetActive()->getQueue();
    assert(m_textureInit);

    queueCB.SetRenderTargets(1, &m_texture, NULL, NULL, NULL);
    queueCB.SetViewportScissor(0, 0, width, height);
    queueCB.BindProgram(m_programRT, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

    queueCB.submit();
    queue->FenceSync(m_sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
    queue->Flush();
    queue->WaitSync(m_sync);

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
}

// --------------------------------- LWNTexturePitchTest ----------------------------------------

class LWNTexturePitchTest {
    static const int cellSize = 70;
    static const int cellMargin = 2;

public:
    LWNTEST_CppMethods();
};

lwString LWNTexturePitchTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple test for pitch-linear textures.\n"
        "Draws pitch-linear textures in various sizes and formats, alternating\n"
        "between texture fetching and render target use cases.\n"
        "A passing image would look like alternating squares of red-green gradient and\n"
        "green-blue gradient with diagonal white squares. The red-green gradient is\n"
        "the texture fetch test and the green-blue gradient is the render target test.\n";
    return sb.str();
}

int LWNTexturePitchTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(39, 1);
}

void LWNTexturePitchTest::doGraphics() const
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
    queueCB.ClearColor(0, 0.45, 0.45, 0.45, 1.0);

    LWNTexturePitch pitchLinearTextureTest;
    pitchLinearTextureTest.init();

    Format fmts[] = {
        Format::RGBA8,
        Format::RGBA16,
        Format::RGBA32F,
#ifdef TEST_RGB565
        Format::RGB565,
#endif
        Format::RGB10A2
    };
    int numFormats = sizeof(fmts) / sizeof(fmts[0]);

    int heights[] = {
        16,
        64,
        128,
        3,
        32,
        256
    };
    int widths[] = {
        16,
        64,
        128,
        4,
        421,
        256
    };
    int numSz = sizeof(heights) / sizeof(heights[0]);
    
    int c = 0;
    Sync *sync = device->CreateSync();
    for (int i = 0; i < numSz; i++) {
        for (int j = 0; j < numFormats; j++) {
            for (int k = 0; k < 2; k++) {
                bool renderTarget = (k == 1);

                // Create the test texture.
                pitchLinearTextureTest.setTexture(widths[i], heights[i], fmts[j], !renderTarget, renderTarget);

                if (renderTarget) {
                    // Every second test is filled by rendering to it.
                    // The renderToTexture will insert a fence to make sure rendering is done before
                    // we render out the texture.
                    pitchLinearTextureTest.renderToTexture(widths[i], heights[i], fmts[j]);
                }

                // Draw the texture onto the screen.
                SetCellViewportScissorPadded(queueCB, c % cellsX, c / cellsX, cellMargin);
                pitchLinearTextureTest.draw();

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

OGTEST_CppTest(LWNTexturePitchTest, lwn_texture_pitch, );

