/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_texture_compress.cpp
//

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

using namespace lwn;
using namespace lwn::dt;

#ifndef ROUND_UP
    #define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#endif

// ------------------------------- LWNTextureRuntimeCompress --------------------------------------

class LWNTextureRuntimeCompress {
    // The compressed texture we are rendering to, with its memory pool.
    MemoryPool *m_pool;
    Texture *m_texture;
    size_t m_textureSize;
    bool m_textureInit;

    int m_reinterpretWidth;
    int m_reinterpretHeight;
    int m_mipmapLevel;
    Format m_reinterpretFormat;
    Program *m_reinterpretCompressProgram;
    Program *m_programCompressDXT1;
    Program *m_programCompressDXT5;

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
    bool initTexture(int width, int height, Format fmt, uint32_t miplvl);
    void runtimeCompressTexture();

    bool init();
    void draw(uint32_t miplvl);

    LWNTextureRuntimeCompress();
    ~LWNTextureRuntimeCompress();
};

LWNTextureRuntimeCompress::LWNTextureRuntimeCompress()
{
    m_textureInit = false;
}

LWNTextureRuntimeCompress::~LWNTextureRuntimeCompress()
{
    delete m_bufpool;
    m_texture->Free();
    m_pool->Free();
    m_sync->Free();
}

bool LWNTextureRuntimeCompress::init()
{
    DEBUG_PRINT(("LWNTextureRuntimeCompress:: Creating test assets...\n"));
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();

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
        "layout(binding=0, std140) uniform Block {\n"
        "    uint miplvl;"
        "};\n"
        "void main() {\n"
        "  fcolor = textureLod(tex, ouv, miplvl);\n"
        "}\n";
    FragmentShader fs_compress_dxt1(440);
    fs_compress_dxt1 <<
        "in vec2 ouv;\n"
        "out uvec4 fcolor;\n"
        "void main() {\n"
            // Create a red gradient in RGB565 format.
        "  fcolor.x = int(31.0 * ouv.x) << 11;\n"
            // Create a green gradient in RGB565 format.
        "  fcolor.y = int(63.0 * ouv.y) << 5;\n"
            // Per-pixel checkerboard pattern between c1 and c0,
            // corresponding to the following 4x4 2-bit lookup table:
            // 00 01 00 01
            // 01 00 01 00
            // 00 01 00 01
            // 01 00 01 00
        "  fcolor.z = 4420;\n"
        "  fcolor.w = 4420;\n"
        "}\n";
    FragmentShader fs_compress_dxt5(440);
    fs_compress_dxt5 <<
        "in vec2 ouv;\n"
        "out uvec4 fcolor;\n"
        "void main() {\n"
            // Two 0xFF alpha values followed by a 4x4 3-bit lookup table of all 0xFF alpha.
            // This works for both DXT2/3 and DXT 4/5 encoding.
        "  fcolor.x = 65535 << 16 | 65535;\n"
        "  fcolor.y = 65535 << 16 | 65535;\n"
            // c0 is a green --> yellow gradient in RGB565 format.
            // c1 is a blue --> cyan gradient in RGB565 format.
        "  int c0 = int(31.0 * ouv.x) << 11 | 63 << 5;\n"
        "  int c1 = int(63.0 * ouv.y) << 5 | 31;\n"
        "  fcolor.z = c0 << 16 | c1;\n"
            // Per-pixel checkerboard pattern between c1 and c0,
            // corresponding to the following 4x4 2-bit lookup table:
            // 00 01 00 01
            // 01 00 01 00
            // 00 01 00 01
            // 01 00 01 00
        "  fcolor.w = 4420 << 16 | 4420;\n"
        "}\n";

    m_program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return false;
    }

    m_programCompressDXT1 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programCompressDXT1, vs, fs_compress_dxt1)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return false;
    }

    m_programCompressDXT5 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programCompressDXT5, vs, fs_compress_dxt5)) {
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
      .SetMinMagFilter(MinFilter::NEAREST_MIPMAP_NEAREST, MagFilter::NEAREST);
    m_sampler = sb.CreateSampler();
    m_sync = device->CreateSync();

    BufferBuilder uboBuilder;
    uboBuilder.SetDefaults();
    uboBuilder.SetDevice(device);
    m_ubo = m_bufpool->allocBuffer(&uboBuilder, BUFFER_ALIGN_UNIFORM_BIT, sizeof(uint32_t));

    // Use the vertex array state for the full test.
    queueCB.BindVertexArrayState(m_vertexState);

    return true;
}

bool LWNTextureRuntimeCompress::initTexture(int width, int height, Format fmt, uint32_t miplvl)
{
    Device *device = DeviceState::GetActive()->getDevice();

    if (m_textureInit) {
        m_texture->Free();
        m_pool->Free();
    }

    // Create a compressed texture.

    DEBUG_PRINT(("LWNTexturePitch:: Creating %dx%d compressed texture...\n", width, height));
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device).SetDefaults()
                .SetTarget(TextureTarget::TARGET_2D)
                .SetFormat(fmt)
                .SetSize2D(width, height);
    textureBuilder.SetLevels(miplvl + 1);
    m_textureSize = textureBuilder.GetStorageSize();
    m_pool  = device->CreateMemoryPool(NULL, m_textureSize, MemoryPoolType::CPU_NON_COHERENT);
    assert((LWNmemoryPool *) m_pool);
    m_texture = textureBuilder.CreateTextureFromPool(m_pool, 0);

    // Re-interpret the compressed texture into a non-compressed version of it with
    // elementSize == blockSize.

    int blockWidth = 1, blockHeight = 1;
    m_reinterpretFormat = Format::NONE;
    switch (fmt) {
        case Format::RGB_DXT1:
        case Format::RGBA_DXT1:
            blockWidth = 4;
            blockHeight = 4;
            m_reinterpretFormat = Format::RGBA16UI;
            m_reinterpretCompressProgram = m_programCompressDXT1;
            break;
        case Format::RGBA_DXT3:
        case Format::RGBA_DXT5:
            blockWidth = 4;
            blockHeight = 4;
            m_reinterpretFormat = Format::RGBA32UI;
            m_reinterpretCompressProgram = m_programCompressDXT5;
            break;
        default:
            assert(!"Don't know how to interpret this compressed format.");
            break;
    }
    m_reinterpretWidth = ROUND_UP((width >> miplvl), blockWidth) / blockWidth;
    m_reinterpretHeight = ROUND_UP((height >> miplvl), blockHeight) / blockHeight;
    m_reinterpretWidth = m_reinterpretWidth < 1 ? 1 : m_reinterpretWidth;
    m_reinterpretHeight = m_reinterpretHeight < 1 ? 1 : m_reinterpretHeight;

    // Create the texture from the same pool at the same offset.

    m_mipmapLevel = miplvl;
    m_textureInit = true;
    return true;
}

void LWNTextureRuntimeCompress::runtimeCompressTexture()
{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    assert(m_textureInit);
    assert(m_reinterpretCompressProgram);

    TextureView reinterpretView;
    TextureView* reinterpretViewPtr[] = { &reinterpretView };
    reinterpretView.SetDefaults().SetFormat(m_reinterpretFormat).SetLevels(m_mipmapLevel, 1);
    queueCB.SetRenderTargets(1, &m_texture, reinterpretViewPtr, NULL, NULL);
    queueCB.SetViewportScissor(0, 0, m_reinterpretWidth, m_reinterpretHeight);
    queueCB.BindProgram(m_reinterpretCompressProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

    queueCB.Barrier(LWN_BARRIER_ILWALIDATE_TEXTURE_BIT | LWN_BARRIER_ORDER_FRAGMENTS_BIT);

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
}

void LWNTextureRuntimeCompress::draw(uint32_t miplvl)
{
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();

    memcpy(m_ubo->Map(), &miplvl, sizeof(uint32_t));
    TextureHandle texHandle = device->GetTextureHandle(m_texture->GetRegisteredTextureID(), m_sampler->GetRegisteredID());
    assert(texHandle);

    queueCB.BindProgram(m_program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, m_ubo->GetAddress(), sizeof(uint32_t));
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

// --------------------------------- LWNTextureRuntimeCompressTest ----------------------------------------

class LWNTextureRuntimeCompressTest {
    static const int cellSize = 60;
    static const int cellMargin = 2;

public:
    LWNTEST_CppMethods();
};


lwString LWNTextureRuntimeCompressTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test re-interpreting a compressed texture as an uncompressed.\n"
        "texture of matching elementSize. The test draws a series of\n"
        "gradient checkerboard patterned squares, cycling through DXT1, DXT3 and DXT5\n"
        "formats and different sizes. The idea is to check that block-linear\n"
        "parameters match up between the original compressed image and the\n"
        "re-interpreted version. The DXT1 checkerboard colors are red and green gradients and the\n"
        "DXT3 and 5 checkerboard colors are green-yellow & blue-cyan gradients.\n";
    return sb.str();
}

int LWNTextureRuntimeCompressTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(39, 1);
}

void LWNTextureRuntimeCompressTest::doGraphics() const
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
    queueCB.ClearColor(0, 0.2, 0.2, 0.2, 1.0);

    LWNTextureRuntimeCompress runtimeTextureCompressTest;
    runtimeTextureCompressTest.init();

    Format fmts[] = {
        Format::RGBA_DXT1,
        Format::RGBA_DXT3,
        Format::RGBA_DXT5,
    };
    int numFormats = sizeof(fmts) / sizeof(fmts[0]);

    int heights[] = {
        1,
        32,
        128,
        3,
        32,
        256,
        512,
        320,
        65,
    };
    int widths[] = {
        1,
        32,
        128,
        4,
        421,
        510,
        512,
        320,
        64
    };
    int numSz = sizeof(heights) / sizeof(heights[0]);

    int cellIndex = 0;
    Sync *sync = device->CreateSync();
    for (int sizeIndex = 0; sizeIndex < numSz; sizeIndex++) {
        for (int formatIndex = 0; formatIndex < numFormats; formatIndex++) {
            for (int miplvl = 0; miplvl < 3; miplvl++) {
                if (widths[sizeIndex] >> miplvl == 0 || heights[sizeIndex] >> miplvl == 0) {
                    continue;
                }
                runtimeTextureCompressTest.initTexture(widths[sizeIndex], heights[sizeIndex], fmts[formatIndex], miplvl);
                runtimeTextureCompressTest.runtimeCompressTexture();

                SetCellViewportScissorPadded(queueCB, cellIndex % cellsX, cellIndex / cellsX, cellMargin);
                runtimeTextureCompressTest.draw(miplvl);

                // Need to CPU wait for everything to complete before destroying and re-creating the texture.
                queueCB.submit();
                queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
                queue->Flush();
                sync->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);

                cellIndex++;
            }
        }
    }

    queueCB.submit();
    queue->Finish();
    sync->Free();
}

OGTEST_CppTest(LWNTextureRuntimeCompressTest, lwn_texture_rt_compress, );
