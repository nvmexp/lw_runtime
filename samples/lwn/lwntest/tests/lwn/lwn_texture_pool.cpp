/*
 * Copyright (c) 2015 - 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#ifndef CEIL
#define CEIL(a,b)        (((a)+(b)-1)/(b))
#endif

#ifndef ROUND_UP
#define ROUND_UP(N, S) (CEIL((N),(S)) * (S))
#endif

using namespace lwn;

class LWNTexturePoolTest
{
    // We pick between four 4x4 textures with 4 samplers with different
    // filters/border colors.
    static const int texSize = 4;
    static const int nTextures = 4;
    static const int nSamplers = 4;

    // Size cells so we have a single column for each texture/sampler pair.
    static const int cellSize = 640 / (nTextures * nSamplers);

    // We maintain three sets of texture and sampler pools, and iterate
    // between them.  We save away the IDs assigned to each texture and
    // sampler in each pool.
    static const int nPools = 3;
    struct TexturePoolInfo {
        const TexturePool *apiPool;
        LWNuint ids[nTextures];
    };
    struct SamplerPoolInfo {
        const SamplerPool *apiPool;
        int poolIndex;
        LWNuint ids[nSamplers];
    };

    // Draw a single row (all textures/samplers) from the selected pool.
    void drawRow(Device *device, QueueCommandBuffer &cb,
                 const TexturePoolInfo *tpi, const SamplerPoolInfo *spi,
                 int row) const;

public:
    LWNTEST_CppMethods();
};

lwString LWNTexturePoolTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test exercising multiple texture and sampler pools with updates via command buffers.\n\n"
        "This test draws several of 4x4 checkerboard textures. Each group of four columns "
        "displays a single texture with red / green / blue / white checkers.  Within a group "
        "of four columns, four samplers are used.  Gray borders on the left; white borders on "
        "the right.  First and third are point sampled; second and fourth are linear filtered.\n\n"
        "The test registers textures and samplers into a variety of pools (with different IDs) "
        "and then binds different combinations of pools.  Each row rendered should look the same, "
        "and match the pattern described above.";
    return sb.str();
}

int LWNTexturePoolTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(34, 0);
}

// Loop over all textures and samplers, and render each combination in a column on the given row.
void LWNTexturePoolTest::drawRow(Device *device, QueueCommandBuffer &cb, const TexturePoolInfo *tpi, const SamplerPoolInfo *spi, int row) const
{
    int col = 0;
    for (int tex = 0; tex < nTextures; tex++) {
        for (int smp = 0; smp < nSamplers; smp++) {
            TextureHandle handle = device->GetTextureHandle(tpi->ids[tex], spi->ids[smp]);
            cb.SetViewport(col * cellSize, row * cellSize, cellSize, cellSize);
            cb.BindTexture(ShaderStage::FRAGMENT, 0, handle);
            cb.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            col++;
        }
    }

    // Make sure we didn't run off the end of the screen.
    assert(col * cellSize <= 640);
}

void LWNTexturePoolTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    MemoryPoolAllocator gpu_allocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    MemoryPoolAllocator coherent_allocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec2 texcoord;\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  otc = texcoord;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "layout(binding = 0) uniform sampler2D tex;\n"
        "in vec3 ocolor;\n"
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc);\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Create a staging buffer to hold texel data.
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *stagingBuffer = coherent_allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, 64 * 1024);
    BufferAddress stagingBufferAddr = stagingBuffer->GetAddress();
    uint8_t *stagingMem = (uint8_t *) stagingBuffer->Map();

    // Create a set of textures with different-colored checkerboards.
    uint8_t colors[] = {
        0xFF, 0x00, 0x00, 0xFF,
        0x00, 0xFF, 0x00, 0xFF,
        0x00, 0x00, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,     // black checker used in all textures
    };
    ct_assert(sizeof(colors) == 4 * (nTextures + 1));

    TextureBuilder tb;
    tb.SetDevice(device);
    tb.SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetFormat(Format::RGBA8);
    tb.SetSize2D(texSize, texSize);
    tb.SetLevels(1);

    Texture *textures[nTextures];
    int stagingOffset = 0;
    int cidx;
    for (int texno = 0; texno < nTextures; texno++) {
        int imageBase = stagingOffset;
        textures[texno] = gpu_allocator.allocTexture(&tb);
        for (int y = 0; y < texSize; y++) {
            for (int x = 0; x < texSize; x++) {
                if ((x + y) & 1) {
                    cidx = 4 * nTextures;   // black
                } else {
                    cidx = texno * 4;       // texture-specific checker
                }
                for (int c = 0; c < 4; c++) {
                    stagingMem[stagingOffset + c] = colors[cidx + c];
                }
                stagingOffset += 4;
            }
        }
        CopyRegion copyRegion = { 0, 0, 0, texSize, texSize, 1 };
        queueCB.CopyBufferToTexture(stagingBufferAddr + imageBase, textures[texno], NULL, &copyRegion, CopyFlags::NONE);
    }

    // Set up a set of sampler objects with different filters and border colors.
    Sampler *samplers[nSamplers];
    SamplerBuilder sb;
    sb.SetDevice(device);
    sb.SetDefaults();
    sb.SetWrapMode(WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER);
    ct_assert(nSamplers == 4);
    for (int i = 0; i < 4; i++) {
        if (i & 1) {
            sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
        } else {
            sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
        }
        if (i & 2) {
            float white[] = { 1.0, 1.0, 1.0, 1.0 };
            sb.SetBorderColor(white);
        } else {
            float gray[] = { 0.5, 0.5, 0.5, 1.0 };
            sb.SetBorderColor(gray);
        }
        samplers[i] = sb.CreateSampler();
    }

    // We set up an array of texture pools, where pool 0 is the default
    // (system) pool owned by lwntest, and subsequent pools are created for
    // this test.
    TexIDPool *idPools[nPools];
    TexturePoolInfo texPools[nPools];
    SamplerPoolInfo smpPools[nPools];

    // Get the pool objects and registered IDs from the default pool.
    idPools[0] = g_lwnTexIDPool;
    texPools[0].apiPool = g_lwnTexIDPool->GetTexturePool();
    smpPools[0].apiPool = g_lwnTexIDPool->GetSamplerPool();
    for (int j = 0; j < nTextures; j++) {
        texPools[0].ids[j] = textures[j]->GetRegisteredTextureID();
    }
    for (int j = 0; j < nSamplers; j++) {
        smpPools[0].ids[j] = samplers[j]->GetRegisteredID();
    }

    // Set up the other non-default pools.  We rotate the order of texture and
    // sampler registration so that each pool uses different IDs for its
    // objects.
    for (int pool = 1; pool < nPools; pool++) {
        idPools[pool] = new TexIDPool(device);
        texPools[pool].apiPool = idPools[pool]->GetTexturePool();
        smpPools[pool].apiPool = idPools[pool]->GetSamplerPool();
        for (int j = 0; j < nTextures; j++) {
            int idx = (pool + j) % nTextures;
            texPools[pool].ids[idx] = idPools[pool]->Register(textures[idx]);
        }
        for (int j = 0; j < nSamplers; j++) {
            int idx = (pool + j) % nSamplers;
            smpPools[pool].ids[idx] = idPools[pool]->Register(samplers[idx]);
        }
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec2 texcoord;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.9, -0.9, 0.0), dt::vec2(-0.25, -0.25) },
        { dt::vec3(-0.9, +0.9, 0.0), dt::vec2(-0.25, +1.25) },
        { dt::vec3(+0.9, -0.9, 0.0), dt::vec2(+1.25, -0.25) },
        { dt::vec3(+0.9, +0.9, 0.0), dt::vec2(+1.25, +1.25) },
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, coherent_allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    vertex.bind(queueCB);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    int row = 0;

    // Draw a row with the default texture and sampler pools.
    drawRow(device, queueCB, &texPools[0], &smpPools[0], row++);

    // Draw rows with the alternate pools selected (still paired), using the
    // Bind() method of the TexIDPool class.
    for (int i = 1; i < nPools; i++) {
        idPools[i]->Bind(queueCB);
        drawRow(device, queueCB, &texPools[i], &smpPools[i], row++);
    }

    // Draw rows with a mix and match of texture and sampler pools.
    for (int i = 0; i < nPools; i++) {
        for (int j = 0; j < nPools; j++) {
            TexturePoolInfo *tp = &texPools[i];
            SamplerPoolInfo *sp = &smpPools[j];
            queueCB.SetTexturePool(tp->apiPool);
            queueCB.SetSamplerPool(sp->apiPool);
            drawRow(device, queueCB, tp, sp, row++);
        }
    }

    // Make sure we didn't run off the end of the screen.
    assert(row * cellSize <= 480);

    // Reset to the system texture pool.
    g_lwnTexIDPool->Bind(queueCB);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    for (int i = 1; i < nPools; i++) {
        delete idPools[i];
    }
}

OGTEST_CppTest(LWNTexturePoolTest, lwn_texture_pool, );

using namespace lwn;

class LWNTexturePoolNoncoherentTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTexturePoolNoncoherentTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Testing noncoherent sampler and texture descriptor pools.\n"
        "LWN implementation should automatically flush caches between the two\n"
        "draw commands. If the automatic flush does not happen, the red texture will\n"
        "remain and the result will be a red screen. Otherwise, the correct image will\n"
        "be a green screen.\n";
    return sb.str();
}

int LWNTexturePoolNoncoherentTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 19);
}

void LWNTexturePoolNoncoherentTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Removing reduced performance debug layer warning for CPU_CACHED pool.
    // CPU_CACHED pools are used for the sampler and texture pools being tested.
    int32_t warningID = 0x00000456;
    DebugWarningIgnore(warningID);

    // Display shader
    Program* pgmDisplay;
    VertexShader vs(430);
    FragmentShader fs(430);
    vs <<
        "out vec2 uv;\n"
        "void main() {\n"
        "  vec2 pos = vec2(0.0);\n"
        "  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);\n"
        "  if (gl_VertexID == 1) pos = vec2(1.0, -1.0);\n"
        "  if (gl_VertexID == 2) pos = vec2(1.0, 1.0);\n"
        "  if (gl_VertexID == 3) pos = vec2(-1.0, 1.0);\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  uv = pos * 0.5 + 0.5;\n"
        "}\n";
    fs <<
        "layout(binding=0) uniform sampler2D tex;\n"
        "in vec2 uv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = textureLod(tex, uv, 0.0);\n"
        "}\n";
    pgmDisplay = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgmDisplay, vs, fs);

    // create noncoherent sampler pool and texture pool
    int texDescSize, smpDescSize;
    device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE, &texDescSize);
    device->GetInteger(DeviceInfo::SAMPLER_DESCRIPTOR_SIZE, &smpDescSize);
    int reservedSmpDescCnt, reservedTexDescCnt;
    device->GetInteger(DeviceInfo::RESERVED_TEXTURE_DESCRIPTORS, &reservedTexDescCnt);
    device->GetInteger(DeviceInfo::RESERVED_SAMPLER_DESCRIPTORS, &reservedSmpDescCnt);
    int smpDescCnt = 2 + reservedSmpDescCnt, texDescCnt = 2 + reservedTexDescCnt;
    size_t smpMemPoolSize = smpDescSize * smpDescCnt;
    smpMemPoolSize = ROUND_UP(smpMemPoolSize, 32);
    size_t texMemPoolSize = texDescSize * texDescCnt;
    texMemPoolSize = ROUND_UP(texMemPoolSize, 32);
    MemoryPool* texDescMemPool = device->CreateMemoryPool(NULL, texMemPoolSize, MemoryPoolType::CPU_NON_COHERENT);
    MemoryPool* smpDescMemPool = device->CreateMemoryPool(NULL, smpMemPoolSize, MemoryPoolType::CPU_NON_COHERENT);
    TexturePool tp;
    SamplerPool sp;
    tp.Initialize(texDescMemPool, 0, texDescCnt);
    sp.Initialize(smpDescMemPool, 0, smpDescCnt);
    queueCB.SetTexturePool(&tp);
    queueCB.SetSamplerPool(&sp);

    // create samplers
    Sampler* smpNearest;
    Sampler* smpLOD;
    LWNuint smpNearestID, smpLODID;
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::NEAREST_MIPMAP_NEAREST, MagFilter::NEAREST);
    smpNearest = sb.CreateSampler();
    sp.RegisterSampler(reservedSmpDescCnt+0, smpNearest);
    smpNearestID = reservedSmpDescCnt+0;
    sb.SetLodBias(1.0);
    smpLOD = sb.CreateSampler();
    sp.RegisterSampler(reservedSmpDescCnt+1, smpLOD);
    smpLODID = reservedSmpDescCnt+1;

    // create and fill texture
    const int texDim = 2;
    Texture* tex;
    TextureHandle texHandle;
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetLevels(2);
    tb.SetSize2D(texDim, texDim);
    tb.SetFormat(Format::RGBA8);
    size_t texSize = tb.GetStorageSize();
    size_t texAlignment = tb.GetStorageAlignment();
    texSize = ROUND_UP(texSize, texAlignment);
    MemoryPool* texGpuMemPool = device->CreateMemoryPool(NULL, texSize, MemoryPoolType::GPU_ONLY);
    MemoryPool* texCpuMemPool = device->CreateMemoryPool(NULL, 2*texSize, MemoryPoolType::CPU_COHERENT);
    tex = tb.CreateTextureFromPool(texGpuMemPool, 0);

    // fill 2x2 (mip 0)
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer* texBuff = bb.CreateBufferFromPool(texCpuMemPool, 0, texSize);
    dt::u8lwec4* outPtr = static_cast<dt::u8lwec4*>(texBuff->Map());
    outPtr[0] = dt::u8lwec4(0.0,1.0,0.0,1.0);
    outPtr[1] = dt::u8lwec4(0.0,1.0,0.0,1.0);
    outPtr[2] = dt::u8lwec4(0.0,1.0,0.0,1.0);
    outPtr[3] = dt::u8lwec4(0.0,1.0,0.0,1.0);
    TextureView tv;
    tv.SetDefaults();
    tv.SetLevels(0,1);
    CopyRegion cr = {0,0,0, texDim,texDim,1};
    queueCB.CopyBufferToTexture(texBuff->GetAddress(), tex, 0, &cr, CopyFlags::NONE);
    {
        Sync* sync = device->CreateSync();
        queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
        queueCB.submit();
        queue->Flush();
        sync->Wait(0);
        sync->Free();
    }
    texBuff->Free();

    // fill 1x1 (mip 1)
    Buffer* texBuff2 = bb.CreateBufferFromPool(texCpuMemPool, texSize, texSize);
    dt::u8lwec4* outPtr2 = static_cast<dt::u8lwec4*>(texBuff2->Map());
    outPtr2[0] = dt::u8lwec4(1.0,1.0,0.0,1.0);
    tv.SetLevels(1,1);
    cr = {0,0,0, texDim>>1,texDim>>1,1};
    queueCB.CopyBufferToTexture(texBuff2->GetAddress(), tex, &tv, &cr, CopyFlags::NONE);
    {
        Sync* sync = device->CreateSync();
        queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
        queueCB.submit();
        queue->Flush();
        sync->Wait(0);
        sync->Free();
    }
    texBuff2->Free();

    // texture view
    tv.SetDefaults();
    tv.SetSwizzle(TextureSwizzle::R, TextureSwizzle::ZERO, TextureSwizzle::B, TextureSwizzle::A);

    // draw
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
    queueCB.BindProgram(pgmDisplay, ShaderStageBits::ALL_GRAPHICS_BITS);

    tp.RegisterTexture(reservedTexDescCnt+0, tex, &tv);
    texHandle = device->GetTextureHandle(reservedTexDescCnt+0, smpLODID);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    queueCB.submit();
    queue->Finish();

    // There should be an automatic CPU flush to change the texture from red to green before the next draw.

    tp.RegisterTexture(reservedTexDescCnt+0, tex, NULL);
    texHandle = device->GetTextureHandle(reservedTexDescCnt+0, smpNearestID);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    queueCB.submit();
    queue->Finish();

    // cleanup
    queueCB.SetSamplerPool(g_lwnTexIDPool->GetSamplerPool());
    queueCB.SetTexturePool(g_lwnTexIDPool->GetTexturePool());
    smpNearest->Free();
    smpLOD->Free();
    tex->Free();
    texGpuMemPool->Free();
    texCpuMemPool->Free();
    sp.Finalize();
    tp.Finalize();
    pgmDisplay->Free();
    DebugWarningAllow(warningID);

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNTexturePoolNoncoherentTest, lwn_texture_pool_noncoherent, );
