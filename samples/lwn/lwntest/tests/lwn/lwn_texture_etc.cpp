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

#include "lwn_PrivateFormats.h"

#include "g_etc1Tex.h"

using namespace lwn;


class LWNTextureETCTest {
public:
   LWNTEST_CppMethods();

   static const int texWidth  = 64;
   static const int texHeight = 64;
};


static Texture* createETC1Texture(int w, int h, LWNformat fmt, MemoryPoolAllocator& alloc, unsigned char* data)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    TextureBuilder textureBuilder;

    textureBuilder.SetDevice(device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetSize2D(w, h);

    lwnTextureBuilderSetFormat(reinterpret_cast<LWNtextureBuilder*>(&textureBuilder), fmt);

    Texture *tex = alloc.allocTexture(&textureBuilder);

    if (data)
    {
        const size_t texSize = textureBuilder.GetStorageSize();

        MemoryPoolAllocator pboAllocator(device, NULL, texSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

        BufferBuilder bb;
        bb.SetDevice(device).SetDefaults();
        Buffer *pbo = pboAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texSize);
        BufferAddress pboAddr = pbo->GetAddress();

        char *ptr = (char *)pbo->Map();

        memcpy(ptr, data, texSize);

        CopyRegion copyRegion = { 0, 0, 0, w, h, 1 };
        queueCB.CopyBufferToTexture(pboAddr, tex, NULL, &copyRegion, CopyFlags::NONE);

        queueCB.submit();
        queue->Finish();
    }

    return tex;
}


lwString LWNTextureETCTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test to verify the functionality of liblwn-etc1. ETC1 compressed textures are only supported if "
        "lwn-etc1.nso is linked against the application. The lwn.nso has no ETC1 support instead it would "
        "use DXT to uncompress. This test will create one LWN_FORMAT_PRIVATE_RGB8_ETC1 and one "
        "LWN_FORMAT_PRIVATE_RGBA8_ETC1 texture and render them. The expected result are two quads "
        "showing a color gradient.\n "
        "Linking liblwn-etc1.a statically will not enable ETC1 support. Since the application does not "
        "reference any symbols in the library, liblwn-etc1.a does not get linked in.\n";

    return sb.str();
}

int LWNTextureETCTest::isSupported() const
{
    bool isSupported = false;

#if defined(LW_TEGRA)
    isSupported = true;
#endif

    return (isSupported && lwogCheckLWNAPIVersion(52, 6));
}

void LWNTextureETCTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec2 texCoord;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  tc = texCoord;\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "layout(binding = 0) uniform sampler2D smp;\n"
        "in vec2 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec4 texel = texture(smp, tc);\n"
        "  fcolor = texel;\n"
        "}\n";

    Program *program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());

        LWNFailTest();
        return;
    }

    const int vertexCount = 4;

    struct Vertex {
        dt::vec3 position;
        dt::vec2 texCoord;
    };

    const Vertex vertices[] = {
        { dt::vec3(-0.9f, -0.9f, 0.0f), dt::vec2(0.0f, 0.0f) },
        { dt::vec3( 0.9f, -0.9f, 0.0f), dt::vec2(1.0f, 0.0f) },
        { dt::vec3(-0.9f,  0.9f, 0.0f), dt::vec2(0.0f, 1.0f) },
        { dt::vec3( 0.9f,  0.9f, 0.0f), dt::vec2(1.0f, 1.0f) }
    };

    MemoryPoolAllocator bufferAllocator(device, NULL, vertexCount * sizeof(vertices), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, texCoord);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, bufferAllocator, vertices);
    BufferAddress vboAddr = vbo->GetAddress();

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE)
      .SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);

    Sampler *smp = sb.CreateSampler();

    MemoryPoolAllocator texAllocator(device, NULL, 64 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    const int numTests = 2;
    const int cellWidth = lwrrentWindowWidth / numTests;
    const int cellHeight = lwrrentWindowHeight / 2;

    Texture* tex[numTests];
    TextureHandle hTex[numTests];

    // First create 2 ETC1 textures without overloading lwnTexturePoolRegisterTexture.
    tex[0] = createETC1Texture(texWidth, texHeight, LWN_FORMAT_PRIVATE_RGB_ETC1, texAllocator, lwnRGB_ETC1_Data);
    hTex[0] = device->GetTextureHandle(tex[0]->GetRegisteredTextureID(), smp->GetRegisteredID());

    tex[1] = createETC1Texture(texWidth, texHeight, LWN_FORMAT_PRIVATE_RGBA_ETC1, texAllocator, lwnRGBA_ETC1_Data);
    hTex[1] = device->GetTextureHandle(tex[1]->GetRegisteredTextureID(), smp->GetRegisteredID());

    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboAddr, vertexCount * sizeof(vertices));

    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 0.2f, 0.2f, 0.2f, 1.0f);

    for (int i = 0; i < numTests; ++i) {
        queueCB.SetViewport(cellWidth * i, 0, cellWidth, cellHeight);
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, hTex[i]);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, vertexCount);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNTextureETCTest, lwn_texture_etc, );
