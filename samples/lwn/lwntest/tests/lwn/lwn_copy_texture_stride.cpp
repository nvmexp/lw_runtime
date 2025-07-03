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

// ------------------------------- LWNCopyTextureStride --------------------------------------

class LWNCopyTextureStride {
    // Memory pool where all the textures come from.
    MemoryPoolAllocator *m_pool;

    // Rest of the stuff we need to draw the texture.
    MemoryPoolAllocator *m_bufpool;
    Program *m_program;
    Program *m_programLwbe;
    Sampler *m_sampler;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    LWNuint m_vertexDataSize;

    struct Vertex {
        vec3 position;
        vec3 uv;
    };

public:
    Texture createTexture(bool lwbe, int w, int h, int rowStrideOffset, int imageStrideOffset);

    bool init();
    void createAndDrawTexture(bool lwbe, int w, int h, int rowStrideOffset, int imageStrideOffset);
    LWNCopyTextureStride();
    ~LWNCopyTextureStride();
};

LWNCopyTextureStride::LWNCopyTextureStride()
{
}

LWNCopyTextureStride::~LWNCopyTextureStride()
{
    delete m_bufpool;
    delete m_pool;
}

bool LWNCopyTextureStride::init()
{
    DEBUG_PRINT(("LWNCopyTextureStride:: Creating test assets...\n"));
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();

    m_bufpool = new MemoryPoolAllocator(device, NULL, 0x40000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    m_pool = new MemoryPoolAllocator(device, NULL, 0x40000, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

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
    FragmentShader fsLwbe(440);
    fsLwbe <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout (binding=0) uniform samplerLwbe tex;"
        "void main() {\n"
        "  fcolor = texture(tex, vec3(ouv, 0.5));\n"
        "}\n";

    m_program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program, vs, fs)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return false;
    }

    m_programLwbe = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_programLwbe, vs, fsLwbe)) {
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
    sb.SetDevice(device)
        .SetDefaults()
        .SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    m_sampler = sb.CreateSampler();

    // Use the vertex array state for the full test.
    queueCB.BindVertexArrayState(m_vertexState);

    return true;
}

static void FillTextureRGBA8(char* data, int w, int h, int layers, int rowStride, int imageStride)
{
    // Fill with checkerboard pattern.
    for (int z = 0; z < layers; z++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                u8vec4 *v = (u8vec4 *)&data[ (x * 4) + rowStride * y + imageStride * z];
                bool checker = !!( ((x  / 5) % 2) ^ ((y  / 5) % 2) );
                if (checker) {
                    v->setX(51);
                    v->setY(14);
                    v->setZ(102);
                } else {
                    v->setX(190);
                    v->setY(212);
                    v->setZ(6);
                }
                v->setW(255);
            }
        }
    }
}

void LWNCopyTextureStride::createAndDrawTexture(bool lwbe, int w, int h, int rowStrideOffset, int imageStrideOffset)
{
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();
    assert( w > 0 && h > 0);

    // Create a texture.
    int nfaces = lwbe ? 6 : 1;
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device).SetDefaults()
                  .SetTarget(lwbe ? TextureTarget::TARGET_LWBEMAP : TextureTarget::TARGET_2D)
                  .SetFormat(Format::RGBA8)
                  .SetSize2D(w, h);
    Texture *texture = m_pool->allocTexture(&textureBuilder);

    // Fill the texture from buffer.
    const int rowStride = (w * 4 /* RGBA8 */) + rowStrideOffset;
    const int imageStride = (rowStride * h) + imageStrideOffset;
    const int texelBufferSize = imageStride * nfaces;
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();
    Buffer *texbuf = m_bufpool->allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, texelBufferSize);
    void* data = texbuf->Map();
    memset(data, 0, texelBufferSize);

    FillTextureRGBA8((char*) data, w, h, nfaces, rowStride, imageStride);
    queueCB.SetCopyRowStride(rowStride);
    queueCB.SetCopyImageStride(imageStride);
    if (rowStride != (int) queueCB.GetCopyRowStride() || imageStride != (int) queueCB.GetCopyImageStride()) {
        DEBUG_PRINT(("Getter functions are broken!\n"));
        queueCB.ClearColor(0, 0.8, 0.2, 0.2, 1.0);
        return;
    }
    CopyRegion copyRegion = { 0, 0, 0, w, h, nfaces };
    queueCB.CopyBufferToTexture(texbuf->GetAddress(), texture, NULL, &copyRegion, CopyFlags::NONE);

    TextureHandle texHandle = device->GetTextureHandle(texture->GetRegisteredTextureID(), m_sampler->GetRegisteredID());
    assert(texHandle);

    queueCB.BindProgram(lwbe ? m_programLwbe : m_program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

// --------------------------------- LWNCopyTextureStrideTest ----------------------------------------

class LWNCopyTextureStrideTest {
    static const int cellSize = 40;
    static const int cellMargin = 2;

public:
    LWNTEST_CppMethods();
};


lwString LWNCopyTextureStrideTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple test for command buffer copy row and image stride parameters.  \n"
        "Passing test shows 6 rows of lwbemap checkerboard on top, and 6 rows \n"
        "of normal 2D checkerboard on bottom. The 6 lwbemap rows have different row and image \n"
        "stride parameters but should turn out exactly the same, similar for the 6 2D texture \n"
        "rows.\n";
    return sb.str();
}

int LWNCopyTextureStrideTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(41, 2);
}

void LWNCopyTextureStrideTest::doGraphics() const
{
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.2, 0.2, 0.2, 1.0);

    LWNCopyTextureStride copyTextureStrideTest;
    copyTextureStrideTest.init();

    const static int numSz = 10;
    int heights[numSz] = { 1, 128, 32, 320, 65, 123,  5, 3,   3, 24 };
    int widths[numSz] =  { 1, 128, 32, 128,  4, 123, 15, 4, 132, 23 };

    const static int numOffsets = 6;
    int rowOffsets[numOffsets] = {
        0,
        4,
        0,
        1,
        511,
        42
    };
    int imageOffsets[numOffsets] = {
        0,
        0,
        15,
        1,
        432,
        2132
    };

    int c = 0;
    for (int lwbe = 0; lwbe < 2; lwbe++) {
        for (int j = 0; j < numOffsets; j++) {
            for (int i = 0; i < numSz; i++) {
                // Draw the texture onto the screen.
                SetCellViewportScissorPadded(queueCB, c % cellsX, c / cellsX, cellMargin);
                copyTextureStrideTest.createAndDrawTexture(
                    lwbe != 0, widths[i],
                    lwbe ? widths[i] : heights[i] /* Lwbe textures must be square. */,
                    rowOffsets[j], imageOffsets[j]);
                c++;
            }
            c = ROUND_UP(c, cellsX);
        }
    }

    queueCB.submit();
    queue->Finish();
    queueCB.SetCopyRowStride(0);
    queueCB.SetCopyImageStride(0);
}

OGTEST_CppTest(LWNCopyTextureStrideTest, lwn_copy_texture_stride, );
