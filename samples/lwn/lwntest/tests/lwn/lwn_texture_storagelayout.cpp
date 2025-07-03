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

#define STORAGELAYOUT_MAXBUFFERSIZE 0x80000

using namespace lwn;
using namespace lwn::dt;

// ----------------------------------- LWNTextureLayout ------------------------------------------

class LWNTextureLayout {
    MemoryPoolAllocator *m_pool;

    // Rest of the stuff we need to draw the texture.
    MemoryPoolAllocator *m_bufpool;
    Program *m_program2D;
    Program *m_program3D;
    Sampler *m_sampler;
    VertexArrayState m_vertexState;
    Buffer *m_vbo;
    LWNuint m_vertexDataSize;
    uint32_t *m_buffer;

    struct Vertex {
        vec3 position;
        vec3 uv;
    };

public:
    bool init(void);
    void draw(Texture* texture);

    Texture *createTexture(TextureTarget target, int w, int h, int d, bool expectSmaller);
    void destroyTexture(Texture* texture);

    LWNTextureLayout();
    ~LWNTextureLayout();
};

LWNTextureLayout::LWNTextureLayout()
        : m_pool(NULL), m_bufpool(NULL), m_program2D(NULL), m_program3D(NULL), m_sampler(NULL),
          m_vbo(NULL), m_vertexDataSize(0)
{
    m_buffer = new uint32_t[STORAGELAYOUT_MAXBUFFERSIZE];
    for (int i = 0; i < STORAGELAYOUT_MAXBUFFERSIZE; i++) {
        m_buffer[i] =  i % 2 ? 0xFF33EE33 : 0xFF883388;
    }
}

LWNTextureLayout::~LWNTextureLayout()
{
    delete[] m_buffer;
    delete m_pool;
    delete m_bufpool;
}

bool LWNTextureLayout::init() {
    DEBUG_PRINT(("LWNTextureLayout:: Creating other test assets...\n"));

    Device *device = DeviceState::GetActive()->getDevice();

    m_bufpool = new MemoryPoolAllocator(device, NULL, 0x10000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    m_pool = new MemoryPoolAllocator(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

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
    FragmentShader fs_2D(440);
    fs_2D <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout (binding=0) uniform sampler2D tex;"
        "void main() {\n"
        "  fcolor = textureLod(tex, ouv, 0.0);\n"
        "}\n";
    FragmentShader fs_3D(440);
    fs_3D <<
        "in vec2 ouv;\n"
        "out vec4 fcolor;\n"
        "layout (binding=0) uniform sampler3D tex;"
        "void main() {\n"
        "  fcolor = textureLod(tex, vec3(ouv, 0.5), 0.0);\n"
        "}\n";

    m_program2D = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program2D, vs, fs_2D)) {
        DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
        return false;
    }

    m_program3D = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(m_program3D, vs, fs_3D)) {
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

    return true;
}

void LWNTextureLayout::draw(Texture* texture)
{
    Device *device = DeviceState::GetActive()->getDevice();
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    if (!texture) {
        queueCB.ClearColor(0, 1.0, 0.2, 0.2, 1.0);
        return;
    }
    TextureHandle texHandle = device->GetTextureHandle(texture->GetRegisteredTextureID(), m_sampler->GetRegisteredID());
    assert(texHandle);

    queueCB.BindProgram(texture->GetTarget() == TextureTarget::TARGET_2D ? m_program2D : m_program3D,
                        ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(m_vertexState);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.BindVertexBuffer(0, m_vbo->GetAddress(), m_vertexDataSize);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
}

Texture *LWNTextureLayout::createTexture(TextureTarget target, int w, int h, int d, bool expectSmaller)
{
    Device *device = DeviceState::GetActive()->getDevice();

    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device).SetDefaults().SetTarget(target)
        .SetFormat(Format::RGBA8).SetSize3D(w, h, d);
    size_t textureSize = textureBuilder.GetStorageSize();
    size_t textureSizeMinimal = textureBuilder.
            SetFlags(TextureFlags::MINIMAL_LAYOUT).GetStorageSize();

    if (expectSmaller && textureSize <= textureSizeMinimal) {
        // This test case that should have been be smaller when the minimal flag is enabled.
        DEBUG_PRINT(("size %dx%dx%d normal size %d >= minimal size %d\n",
                w, h, d,
                (int) textureSize, (int) textureSizeMinimal));
        return NULL;
    }
    if (!expectSmaller) {
        if (textureSize != textureSizeMinimal) {
            // This test case that should have been be equal in size when the minimal flag is enabled.
            DEBUG_PRINT(("size %dx%dx%d normal size %d != minimal size %d\n",
                    w, h, d,
                    (int) textureSize, (int) textureSizeMinimal));
            return NULL;
        }
    }

    Texture *texture = m_pool->allocTexture(&textureBuilder);
    if (!texture) {
        DEBUG_PRINT(("alloc texture failed.\n"));
        return NULL;
    }

    assert(w * h * d <= STORAGELAYOUT_MAXBUFFERSIZE);
    CopyRegion region = { 0, 0, 0, w, h, d };
    texture->WriteTexels(NULL, &region, m_buffer);
    texture->FlushTexels(NULL, &region);
    return texture;
}

void LWNTextureLayout::destroyTexture(Texture* texture)
{
    m_pool->freeTexture(texture);
}


// --------------------------------- LWNTextureLayoutTest ----------------------------------------

class LWNTextureLayoutTest {
    static const int cellSize = 110;
    static const int cellMargin = 2;

public:
    LWNTEST_CppMethods();
};

lwString LWNTextureLayoutTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Minimal texture storage layout test that tests textures created with\n"
        "MINIMAL_LAYOUT flag are smaller than textures created without, yet still work.\n";
    return sb.str();
}

int LWNTextureLayoutTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 20);
}

void LWNTextureLayoutTest::doGraphics() const
{
    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    cellTestInit(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.25, 0.25, 0.25, 1.0);

    LWNTextureLayout test;
    test.init();

    struct TestConfig {
        TextureTarget target;
        int width;
        int height;
        int depth;
        bool expectSmaller;
    };
    TestConfig configs[] = {
        { TextureTarget::TARGET_2D, 256, 140, 1, true},
        { TextureTarget::TARGET_2D, 256, 139, 1, true},
        { TextureTarget::TARGET_2D, 256, 138, 1, true},
        { TextureTarget::TARGET_2D, 256, 137, 1, true},
        { TextureTarget::TARGET_2D, 254, 137, 1, true},
        { TextureTarget::TARGET_2D, 258, 129, 1, true},
        { TextureTarget::TARGET_2D, 558, 228, 1, true},
        { TextureTarget::TARGET_2D, 512, 512, 1, false},
        { TextureTarget::TARGET_3D, 558, 228, 3, true},
        { TextureTarget::TARGET_3D, 158, 127, 9, true},
        { TextureTarget::TARGET_3D, 256, 1, 140, true},
        { TextureTarget::TARGET_3D, 256, 3, 19 , true},
        { TextureTarget::TARGET_3D, 3, 128, 129, true},
    };

    for (int k = 0; k < (int)__GL_ARRAYSIZE(configs); k++) {
        // Create minimal layout texture.
        Texture *texture = test.createTexture(configs[k].target, configs[k].width,
                                              configs[k].height, configs[k].depth,
                                              configs[k].expectSmaller);

        // Draw the texture onto the screen.
        SetCellViewportScissorPadded(queueCB, k % cellsX, k / cellsX, cellMargin);
        test.draw(texture);

        // Need to CPU wait for everything to complete before destroying and re-creating the texture.
        queueCB.submit();
        queue->Finish();
        test.destroyTexture(texture);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNTextureLayoutTest, lwn_texture_storagelayout, );

