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

using namespace lwn;

class LWNSeamlessLwbeTest
{
    static const int texSize = 4;           // Use a small texture so seams are obvious.
    static const int texSlices = 2;         // Test two layers in a lwbe map array.
public:
    LWNTEST_CppMethods();
};

lwString LWNSeamlessLwbeTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Check for seamless lwbemap filtering on LWN.  We render lwbemaps with "
        "a blue face on the center, green faces on top/bottom, and red faces on "
        "the left and right.  The areas near the joins in the faces should be "
        "visibly blurry since we're filtering across faces of different colors. "
        "If we get a sharp image, seamless filtering is disabled.  On the "
        "bottom half of the screen, we render from a single lwbe map texture. "
        "On the top half, we test a lwbe map array, where the left (layer 0) "
        "image is full brightness and the right image (layer 1) is half brightness.";
    return sb.str();
}

int LWNSeamlessLwbeTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(21, 5);
}

void LWNSeamlessLwbeTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Vertex shader that maps 2D coordinates in a normalized range into
    // either the left or right half, depending on instance ID.  Also construct
    // a texture coordinate using (x,y,+0.5), where the center of the image is
    // on the +Z face, and edges are on neighboring faces.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec2 position;\n"
        "out vec4 tc;\n"
        "void main() {\n"
        "  gl_Position.x = position.x * 0.5 - 0.5 + float(gl_InstanceID);\n"
        "  gl_Position.yzw = vec3(position.y, 0.0, 1.0);\n"
        "  tc  = vec4(position, 0.5, gl_InstanceID);\n"
        "}\n";


    FragmentShader fsLwbe(440);
    fsLwbe <<
        "layout(binding=0) uniform samplerLwbe lwbe;\n"
        "in vec4 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(lwbe, tc.xyz);\n"
        "}\n";
    FragmentShader fsLwbeArray(440);
    fsLwbeArray <<
        "layout(binding=0) uniform samplerLwbeArray lwbe;\n"
        "in vec4 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(lwbe, tc);\n"
        "}\n";

    Program *pgmLwbe = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgmLwbe, vs, fsLwbe)) {
        LWNFailTest();
        return;
    }

    Program *pgmLwbeArray = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgmLwbeArray, vs, fsLwbeArray)) {
        LWNFailTest();
        return;
    }

    // Set up a basic default sampler.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();
    LWNuint smpID = smp->GetRegisteredID();

    // Set up lwbe map array and lwbe map textures.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_LWBEMAP_ARRAY);
    tb.SetSize3D(texSize, texSize, 6 * texSlices);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(1);
    LWNsizeiptr texStorageSize = tb.GetPaddedStorageSize();
    MemoryPoolAllocator texAllocator(device, NULL, 2 * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    Texture *lwbeArrayTex = texAllocator.allocTexture(&tb);
    TextureHandle lwbeArrayHandle = device->GetTextureHandle(lwbeArrayTex->GetRegisteredTextureID(), smpID);

    tb.SetTarget(TextureTarget::TARGET_LWBEMAP);
    tb.SetSize2D(texSize, texSize);
    Texture *lwbeTex = texAllocator.allocTexture(&tb);
    TextureHandle lwbeHandle = device->GetTextureHandle(lwbeTex->GetRegisteredTextureID(), smpID);

    // Set up a buffer with pitch data for the two textures, and copy into the
    // texture storage.
    MemoryPoolAllocator texSrcAllocator(device, NULL, 2 * texStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *texSrcDataBuffer = texSrcAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texStorageSize);
    BufferAddress texSrcDataBufferAddr = texSrcDataBuffer->GetAddress();
    dt::u8lwec4 *texSrcBase = (dt::u8lwec4 *) texSrcDataBuffer->Map();
    dt::u8lwec4 *texSrcData = texSrcBase;
    for (int layer = 0; layer < texSlices; layer++) {
        float scale = 0.5 * (2 - layer);
        for (int face = 0; face < 6; face++) {
            dt::u8lwec4 texelData = dt::u8lwec4((0 == (face / 2)) ? scale : 0.0,
                                                (1 == (face / 2)) ? scale : 0.0,
                                                (2 == (face / 2)) ? scale : 0.0, 1.0);
            for (int texel = 0; texel < texSize * texSize; texel++) {
                *texSrcData++ = texelData;
            }
        }
    }
    CopyRegion copyRegion = { 0, 0, 0, texSize, texSize, 6 * texSlices };
    queueCB.CopyBufferToTexture(texSrcDataBufferAddr, lwbeArrayTex, NULL, &copyRegion, CopyFlags::NONE);
    copyRegion.depth = 6;
    queueCB.CopyBufferToTexture(texSrcDataBufferAddr, lwbeTex, NULL, &copyRegion, CopyFlags::NONE);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec2 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec2(-1.0, -1.0) },
        { dt::vec2(-1.0, +1.0) },
        { dt::vec2(+1.0, -1.0) },
        { dt::vec2(+1.0, +1.0) },
    };

    MemoryPoolAllocator vboAllocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // We will lay out our image as a 2x2 grid of squares (with the grid entry
    // corresponding to layer 1 of a non-array lwbemap empty); use the maximum
    // size that fits into the window.
    int squareSize = lwrrentWindowHeight / 2;
    if (lwrrentWindowWidth < lwrrentWindowHeight) {
        squareSize = lwrrentWindowWidth / 2;
    }

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Draw the single square for non-array lwbes.
    queueCB.SetViewportScissor(lwrrentWindowWidth / 2 - squareSize,
                               lwrrentWindowHeight / 4 - squareSize / 2,
                               2 * squareSize, squareSize);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, lwbeHandle);
    queueCB.BindProgram(pgmLwbe, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    // Draw the two (instanced) squares for non-array lwbes.
    queueCB.SetViewportScissor(lwrrentWindowWidth / 2 - squareSize,
                               3 * lwrrentWindowHeight / 4 - squareSize / 2,
                               2 * squareSize, squareSize);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, lwbeArrayHandle);
    queueCB.BindProgram(pgmLwbeArray, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 0, 2);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNSeamlessLwbeTest, lwn_texture_lwbe_seamless, );


class LWNTextureAddressTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTextureAddressTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Basic test of the lwnTextureGetAddress API.";
    return sb.str();
}

int LWNTextureAddressTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(48, 2);
}

void LWNTextureAddressTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Create a number of different pools -- regular and virtual, compressible
    // and not.
    // Because our spec ensures compressible texture doesn't care
    // whether the virtual pool has COMPRESSIBLE flag or not so check this too at 4.
    // 0: Pool REGULAR,                Tex REGULAR
    // 1: Pool VIRTUAL,                Tex REGULAR
    // 2: Pool REGULAR | COMPRESSIBLE, Tex COMPRESSIBLE
    // 3: Pool VIRTUAL | COMPRESSIBLE, Tex COMPRESSIBLE
    // 4: Pool VIRTUAL               , Tex COMPRESSIBLE
#define NUM_TESTING_POOLS 5
    MemoryPool *pools[NUM_TESTING_POOLS];
    for (int i = 0; i < NUM_TESTING_POOLS; i++) {
        MemoryPoolFlags flags = (MemoryPoolFlags::CPU_NO_ACCESS |
                                 MemoryPoolFlags::GPU_CACHED);
        if (i & 1) flags |= MemoryPoolFlags::VIRTUAL;
        if (i & 2) flags |= MemoryPoolFlags::COMPRESSIBLE;
        if (i == 4) flags |= MemoryPoolFlags::VIRTUAL;
        pools[i] = device->CreateMemoryPoolWithFlags(NULL, 1024 * 1024, flags);
    }

    TextureBuilder tb;
    tb.SetDefaults().SetDevice(device);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(1);

    // Loop over the different pools, setting up textures as we go and making
    // sure we at least get internally consistent addresses.
    bool result = true;
    for (int i = 0; i < NUM_TESTING_POOLS; i++) {
        Texture *tex1, *tex2;
        TextureAddress texAddr1, texAddr2;
        BufferAddress poolAddr = pools[i]->GetBufferAddress();

        // Set up a pair of 2D block linear textures at different offsets in
        // the pool and make sure their addresses are at the same offset
        // relative to each other.
        tb.SetTarget(TextureTarget::TARGET_2D);
        tb.SetSize2D(8, 8);
        tb.SetStorage(pools[i], 0);
        tb.SetFlags((i > 1) ? TextureFlags::COMPRESSIBLE : TextureFlags(0));
        tex1 = tb.CreateTextureFromPool(pools[i], 0);
        tex2 = tb.CreateTextureFromPool(pools[i], 512 * 1024);
        texAddr1 = tex1->GetTextureAddress();
        texAddr2 = tex2->GetTextureAddress();
        if (texAddr2 - texAddr1 != 512 * 1024) {
            result = false;
        }
        tex1->Free();
        tex2->Free();

        // Set up a pair of 2D linear textures and make sure their addresses
        // are internally consistent and consistent with the pool's buffer
        // address.
        tb.SetTarget(TextureTarget::TARGET_2D);
        tb.SetSize2D(8, 8);
        tb.SetStorage(pools[i], 0);
        tb.SetFlags(TextureFlags::LINEAR);
        tb.SetStride(64);
        tex1 = tb.CreateTextureFromPool(pools[i], 0);
        tex2 = tb.CreateTextureFromPool(pools[i], 512 * 1024);
        texAddr1 = tex1->GetTextureAddress();
        texAddr2 = tex2->GetTextureAddress();
        if (texAddr1 != poolAddr || texAddr2 != poolAddr + 512 * 1024) {
            result = false;
        }
        tex1->Free();
        tex2->Free();

        // Set up a pair of buffer textures and make sure their addresses are
        // internally consistent and consistent with the pool's buffer
        // address.
        tb.SetTarget(TextureTarget::TARGET_BUFFER);
        tb.SetSize1D(64);
        tb.SetStorage(pools[i], 0);
        tb.SetFlags(TextureFlags(0));
        tb.SetStride(0);
        tex1 = tb.CreateTextureFromPool(pools[i], 0);
        tex2 = tb.CreateTextureFromPool(pools[i], 512 * 1024);
        texAddr1 = tex1->GetTextureAddress();
        texAddr2 = tex2->GetTextureAddress();
        if (texAddr1 != poolAddr || texAddr2 != poolAddr + 512 * 1024) {
            result = false;
        }
        tex1->Free();
        tex2->Free();
    }

    if (result) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 0.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 0.0);
    }

    queueCB.submit();
    queue->Finish();

    for (int i = 0; i < NUM_TESTING_POOLS; i++) {
        pools[i]->Free();
    }
}

OGTEST_CppTest(LWNTextureAddressTest, lwn_texture_get_address, );
