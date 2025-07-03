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


using namespace lwn;

class LWNUnboundIDTest {
public:
    static const int cellSize = 60;
    static const int cellMargin = 1;
    static const int cellsX = 8;
    static const int cellsY = 8;

    static const int vertexCount = 4;
    static const size_t coherentPoolSize = 0x100000UL; // 1MB pool size

    LWNTEST_CppMethods();

    void SetNextCell(QueueCommandBuffer &queueCB, CellIterator2D &cell) const;

    void TestTextureRead(Device *device, Queue *queue, QueueCommandBuffer &queueCB,
                         LWNint texID, LWNint smpID, bool texelFetch, bool bindless,
                         bool lastTextureID, bool lastSamplerID) const;
};


static Texture* createTestTexture(int w, int h, MemoryPoolAllocator& alloc)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    TextureBuilder textureBuilder;

    textureBuilder.SetDevice(device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::RGBA8)
        .SetSize2D(w, h);

    Texture *tex = alloc.allocTexture(&textureBuilder);

    const size_t texSize = textureBuilder.GetStorageSize();

    MemoryPoolAllocator pboAllocator(device, NULL, texSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *pbo = pboAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texSize);
    BufferAddress pboAddr = pbo->GetAddress();

    uint8_t *ptr = (uint8_t *)pbo->Map();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x){
            *ptr++ = 32;
            *ptr++ = 64;
            *ptr++ = 128;
            *ptr++ = 255;
        }
    }

    CopyRegion copyRegion = { 0, 0, 0, w, h, 1 };
    queueCB.CopyBufferToTexture(pboAddr, tex, NULL, &copyRegion, CopyFlags::NONE);

    queueCB.submit();
    queue->Finish();

    return tex;
}

lwString LWNUnboundIDTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "This test verifies that making a texture access from a shader using a texture ID\n"
        "or sampler ID outside of the range allowed by the lwrrently bound texture pool\n"
        "or sampler pool returns the value (0, 0, 0, 0). This test does so by testing\n"
        "all permutations of the following and comparing against an expected result:\n"
        "* texture() vs texelFetch()\n"
        "* bound vs bindless textures\n"
        "* valid texture ID vs out of range texture ID\n"
        "* valid sampler ID vs out of range sampler ID\n";
    return sb.str();
}

int LWNUnboundIDTest::isSupported() const
{
    return g_lwnDeviceCaps.supportsDrawTexture && lwogCheckLWNAPIVersion(52, 16);
}

void LWNUnboundIDTest::SetNextCell(QueueCommandBuffer &queueCB, CellIterator2D &cell) const
{
    queueCB.SetViewportScissor(cell.x() * cellSize + cellMargin,
                               cell.y() * cellSize + cellMargin,
                               cellSize - 2 * cellMargin,
                               cellSize - 2 * cellMargin);
    cell++;
}

void LWNUnboundIDTest::TestTextureRead(Device *device, Queue *queue,
                                       QueueCommandBuffer &queueCB,
                                       LWNint texID, LWNint smpID,
                                       bool texelFetch, bool bindless,
                                       bool lastTextureID, bool lastSamplerID) const
{
    VertexShader vs(440);
    vs.addExtension(lwShaderExtension::LW_gpu_shader5);
    vs <<
        "layout(location=0) in vec3 in_position;\n"
        "layout(location=1) in vec3 in_textype;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = vec4(in_position, 1.0);\n"
        "}\n";

    bool expectZero = lastTextureID || (!texelFetch && lastSamplerID);

    const char *checkValue = expectZero ?  "ivec4(0, 0, 0, 0)" : "ivec4(32, 64, 128, 255)";
    const char *samplerDef = bindless ? "" : "layout(binding = 0) uniform sampler2D boundTex;\n";
    const char *sampler = bindless ? "sampler2D(bindlessTex)" : "boundTex";
    const char *texFunc = texelFetch ? "texelFetch" : "texture";
    const char *coord = texelFetch ? "ivec2(0,0)" : "vec2(0,0)";
    const char *lodArg = texelFetch ? ", 0" : "";
    FragmentShader fs(440);
    fs.addExtension(lwShaderExtension::LW_gpu_shader5);
    fs <<
        samplerDef <<
        "layout(binding = 0) uniform Block {\n"
        "    uint64_t bindlessTex;\n"
        "};\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  const vec4 green = vec4(0,1,0,1);\n"
        "  const vec4 red = vec4(1,0,0,1);\n"
        "  vec4 res = " << texFunc << "(" << sampler << ", " << coord << lodArg << ");\n"
        "  fcolor = (ivec4(res*255.0) == " << checkValue << ") ? green : red;\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    typedef struct {
        TextureHandle bindlessTex;
    } UniformBlock;
    UniformBlock uboData;

    // For our unbound (out-of-range) texture and sampler IDs, use
    // MAX_TEXTURE_POOL_SIZE-1 and MAX_SAMPLER_POOL_SIZE-1 respectively.
    // (Note that this will not work if the lwrrently bound texture or sampler
    // pool is the maximum possible size.)
    int unboundTextureID = 0, unboundSamplerID = 0;
    device->GetInteger(lwn::DeviceInfo::MAX_TEXTURE_POOL_SIZE, &unboundTextureID);
    device->GetInteger(lwn::DeviceInfo::MAX_SAMPLER_POOL_SIZE, &unboundSamplerID);
    unboundTextureID--;
    unboundSamplerID--;

    if (lastTextureID) texID = unboundTextureID;
    if (lastSamplerID) smpID = unboundSamplerID;
    uboData.bindlessTex = device->GetTextureHandle(texID, smpID);

    MemoryPoolAllocator coherent_allocator(device, NULL, coherentPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &uboData, sizeof(uboData),
                                     BUFFER_ALIGN_UNIFORM_BIT, false);
    BufferAddress uboAddr = ubo->GetAddress();


    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(uboData));
    if (!bindless) {
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, uboData.bindlessTex);
    }
    queueCB.BindProgram(pgm, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, vertexCount);

    queueCB.submit();
    queue->Finish();
}

void LWNUnboundIDTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();


    // Create some objects that are used in more than one subtest
    struct Vertex {
        dt::vec3 position;
    };

    const Vertex vertices[] = {
        { dt::vec3(-1.0f, -1.0f, 0.0f) },
        { dt::vec3( 1.0f, -1.0f, 0.0f) },
        { dt::vec3(-1.0f,  1.0f, 0.0f) },
        { dt::vec3( 1.0f,  1.0f, 0.0f) }
    };
    assert(vertexCount == __GL_ARRAYSIZE(vertices));

    MemoryPoolAllocator bufferAllocator(device, NULL, sizeof(vertices), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, __GL_ARRAYSIZE(vertices), bufferAllocator, vertices);
    BufferAddress vboAddr = vbo->GetAddress();
    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertices));

    MemoryPoolAllocator texAllocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    const LWNuint texWidth = 16;
    const LWNuint texHeight = 16;

    Texture *tex = createTestTexture(texWidth, texHeight, texAllocator);
    LWNuint texID = tex->GetRegisteredTextureID();

    SamplerBuilder sb;
    sb.SetDevice(device)
      .SetDefaults()
      .SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);

    Sampler *smp = sb.CreateSampler();
    LWNuint smpID = smp->GetRegisteredID();


    // Clear and start subtests
    queueCB.SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);

    CellIterator2D cell(cellsX, cellsY);

    for (LWNuint texelFetch = 0; texelFetch < 2; texelFetch++)
    {
        for (LWNuint bindless = 0; bindless < 2; bindless++)
        {
            for (LWNuint lastTexID = 0; lastTexID < 2; lastTexID++)
            {
                for (LWNuint lastSmpID = 0; lastSmpID < 2; lastSmpID++)
                {
                    SetNextCell(queueCB, cell);
                    TestTextureRead(device, queue, queueCB, texID, smpID,
                                    !!texelFetch, !!bindless, !!lastTexID, !!lastSmpID);
                }
            }
        }
    }

    queueCB.submit();
    queue->Finish();
}


OGTEST_CppTest(LWNUnboundIDTest, lwn_unbound_id, );

