/*
* Copyright (c) 2018 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include <array>

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNDisableBoth
{
public:
    LWNTEST_CppMethods();

private:
    static const int    m_texWidth  = 128;
    static const int    m_texHeight = 128;
    static const int    m_cellWidth  = 64;
    static const int    m_cellHeight = 64;

    bool compareTextures(const Texture *tex1, const Texture *tex2) const;
};

static const std::array<Format, 38> formatList {{
    // C32
    Format::R32F,
    Format::R32UI,
    Format::R32I,
    Format::RG16F,
    Format::RG16,
    Format::RG16SN,
    Format::RG16UI,
    Format::RG16I,
    Format::RGBA8,
    Format::RGBA8SN,
    Format::RGBA8UI,
    Format::RGBA8I,
    Format::RGBX8_SRGB,
    Format::RGBA8_SRGB,
    Format::RGB10A2,
    Format::RGB10A2UI,
    Format::R11G11B10F,
    Format::RGBX8,
    Format::RGBX8SN,
    Format::RGBX8UI,
    Format::RGBX8I,
    Format::BGRX8,
    Format::BGRA8,
    Format::BGRX8_SRGB,
    Format::BGRA8_SRGB,
    // C64
    Format::RG32F,
    Format::RG32UI,
    Format::RG32I,
    Format::RGBA16F,
    Format::RGBA16,
    Format::RGBA16SN,
    Format::RGBA16UI,
    Format::RGBA16I,
    Format::RGBX16F,
    Format::RGBX16,
    Format::RGBX16SN,
    Format::RGBX16UI,
    Format::RGBX16I
}};


lwString LWNDisableBoth::getDescription() const
{
    lwStringBuf sb;
    sb << "Test to verify the functionality of the DISABLE_BOTH_COMPRESSSION flag. "
          "The test iterates over all C32 / C64 texture formats and creates two "
          "render targets for each. One that allows Both compression and one that "
          "has Both compression disabled. Then a checkerboard pattern is rendered into "
          "each render target and the content of the two textures is compared. If it "
          "is equal a green quad is drawn, if not a red quad is drawn.\n";

    return sb.str();
}

int LWNDisableBoth::isSupported() const
{
#if defined(LW_TEGRA)
    return lwogCheckLWNAPIVersion(53, 311);
#else
    return 0;
#endif
}

bool LWNDisableBoth::compareTextures(const Texture *tex1, const Texture *tex2) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    assert(tex1);
    assert(tex2);
    assert(tex1->GetStorageSize() == tex2->GetStorageSize());

    const size_t size = tex1->GetStorageSize();
    MemoryPoolAllocator allocator(device, NULL, 2 * size, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferBuilder bb;
    bb.SetDefaults().SetDevice(device);

    Buffer *buf1 = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, size);
    Buffer *buf2 = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, size);

    CopyRegion srcRegion = { 0, 0, 0, m_texWidth, m_texHeight, 1 };
    queueCB.CopyTextureToBuffer(tex1, NULL, &srcRegion, buf1->GetAddress(), CopyFlags::NONE);
    queueCB.CopyTextureToBuffer(tex2, NULL, &srcRegion, buf2->GetAddress(), CopyFlags::NONE);

    queueCB.submit();
    queue->Finish();

    uint8_t *ptr1 = (uint8_t*)buf1->Map();
    uint8_t *ptr2 = (uint8_t*)buf2->Map();

    for (size_t i = 0; i < size; ++i) {
        if (ptr1[i] != ptr2[i]) {
            return false;
        }
    }

    return true;
}

void LWNDisableBoth::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(450);
    vs <<
        "layout(location = 0) in vec3 position;\n"
        "\n"
        "void main() {\n"
        "    gl_Position = vec4(position, 1.0);\n"
        "}\n";

    FragmentShader fs(450);
    fs <<
        "layout(location = 0) in vec3 ocolor;\n"
        "layout(location = 0) out vec4 fcolor;\n"
        "\n"
        "const vec3 color[3] = { vec3(1.0f, 0.0f, 0.0f),\n"
        "                        vec3(0.0f, 1.0f, 0.0f),\n"
        "                        vec3(0.0f, 0.0f, 1.0f) };\n"
        "\n"
        "void main() {\n"
        "    const uint s = 16;\n"
        "    uint x_idx = uint(gl_FragCoord.x) / s;\n"
        "    uint y_idx = uint(gl_FragCoord.y) / s;\n"
        "    uint c_idx = (x_idx & 1) + (y_idx & 1);\n"
        "    fcolor = vec4(color[c_idx], 1.0f);\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
    }

    struct Vertex {
        dt::vec3 position;
    };

    // Define a triangle that covers the entire viepowrt
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3( 1.0, -1.0, 0.0) },
        { dt::vec3(-1.0,  1.0, 0.0) },
        { dt::vec3( 1.0,  1.0, 0.0) },
    };

    const uint32_t numVertices = __GL_ARRAYSIZE(vertexData);
    const uint32_t vboSize = numVertices * sizeof(vertexData);

    MemoryPoolAllocator allocator(device, NULL, vboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, numVertices, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, vboSize);
    queueCB.BindProgram(pgm, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);

    int cellX = 0;
    int cellY = 0;

    for (const auto &fmtItr : formatList)
    {
        TextureBuilder tb;
        tb.SetDefaults().SetDevice(device)
          .SetSize2D(m_texWidth, m_texHeight)
          .SetFormat(fmtItr)
          .SetFlags(TextureFlags::COMPRESSIBLE)
          .SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE)
          .SetSamples(4);

        MemoryPoolAllocator poolAllocator(device, NULL, tb.GetStorageSize() * 4, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

        Texture *msTex = poolAllocator.allocTexture(&tb);

        tb.SetFlags(TextureFlags::COMPRESSIBLE | TextureFlags::DISABLE_BOTH_COMPRESSION);
        Texture *msTexCRA = poolAllocator.allocTexture(&tb);

        // The expected page kinds are: LW_MMU_PTE_KIND_C32_MS4_2CRA = 0xe2 or LW_MMU_PTE_KIND_C64_MS4_2CRA = 0xf0.
        // In addition LWN ORs the page kind with __LWN_STORAGE_CLASS_VALID_BIT = 0x100.
        bool success = ((tb.GetStorageClass() == 0x1e2) || (tb.GetStorageClass() == 0x1f0));

        tb.SetSamples(0)
          .SetTarget(TextureTarget::TARGET_2D)
          .SetFlags(TextureFlags::COMPRESSIBLE);

        Texture *rslvMs  = poolAllocator.allocTexture(&tb);
        Texture *rslvCRA = poolAllocator.allocTexture(&tb);

        MultisampleState mss;
        mss.SetDefaults().SetSamples(4);

        queueCB.BindMultisampleState(&mss);
        queueCB.SetViewportScissor(0, 0, m_texWidth, m_texHeight);

        // Render into 4xAA texture
        queueCB.SetRenderTargets(1, &msTex, NULL, NULL, NULL);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        queueCB.Downsample(msTex, rslvMs);

        // Render into 4xAA texturewith Both compression disabled.
        queueCB.SetRenderTargets(1, &msTexCRA, NULL, NULL, NULL);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        queueCB.Downsample(msTex, rslvCRA);

        queueCB.submit();
        queue->Finish();

        g_lwnWindowFramebuffer.bind();
        queueCB.SetViewportScissor(cellX + 1, cellY + 1, m_cellWidth - 1, m_cellHeight - 1);

        mss.SetDefaults();
        queueCB.BindMultisampleState(&mss);

        success = success && compareTextures(rslvMs, rslvCRA);

        if (success) {
            queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
        } else {
            queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
        }

        queueCB.submit();
        queue->Finish();

        cellX += m_cellWidth;
        if (cellX >= lwrrentWindowWidth) {
            cellX = 0;
            cellY += m_cellHeight;
        }
    }
}

OGTEST_CppTest(LWNDisableBoth, lwn_disable_both_compression, );
