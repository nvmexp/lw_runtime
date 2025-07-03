/*
 * Copyright (c) 2018 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#if defined(HAS_DEVTOOLS) && defined(LW_TEGRA)

// Try these to see that everything is necessary in
// reproducing the gold image.
#define DRAW_EXPECTED 0
#define GLITCH_BASE 0
#define GLITCH_CONTENT 0
#define GLITCH_COMPBITS 0

// Print content of the various structures.
#define DEBUG_PRINT 0
#define DEBUG_CONTENT 0


#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "lwndevtools_bootstrap.h"
using namespace lwn;

static const LWNdevtoolsBootstrapFunctions* devtools;
struct MemoryPoolMarshal
{
    using mapping_t = LWNdevtoolsMemoryPoolMapping;
    using pde_t = LWNdevtoolsDecodedPde;
    using pte_t = LWNdevtoolsDecodedPte;
    using page_t = LWNdevtoolsMemoryPoolPageType;

    struct Tables
    {
        std::vector<pde_t> pdes;
        std::vector<pte_t> ptes;
    };

    struct Compbits
    {
        LWNdevtoolsComptagInfo info;
        std::vector<uint8_t> bits;
    };

    static const size_t COUNT = 3;
    static constexpr std::array<mapping_t,COUNT> MAP{{ LWN_DEVTOOLS_MEMORY_POOL_MAPPING_PITCH, LWN_DEVTOOLS_MEMORY_POOL_MAPPING_BLOCK_LINEAR, LWN_DEVTOOLS_MEMORY_POOL_MAPPING_SPECIAL }};
    std::array<Tables,COUNT> tables;
    Compbits compbits;
    uint64_t base;
    uint64_t size;
    int flags;

    template<class P, class S>
    static std::vector<P> Get(S* Sizer, LWNdevice* device, LWNmemoryPool* pool, mapping_t type, page_t page)
    {
        uint64_t poolSize = lwnMemoryPoolGetSize(pool);
        uint32_t num = 0;
        Sizer(device, pool, type, 0, poolSize, &num);

        const uint32_t max = 3; // Get them 3 by 3
        std::vector<P> res;
        uint32_t pass = 0;
        uint32_t skip = 0;
        do {
            const auto pos = res.size();
            res.resize(pos + max * sizeof(P));
            const bool success __attribute__((unused)) = devtools->MemoryPoolGetPages(device, pool, type,
                &res[skip], max * sizeof(P), page, skip, &pass, max);
            assert(success);
            skip += pass;
            res.resize(pos + pass * sizeof(P));
        } while (pass == max);
        assert(skip == num);

        return res;
    }

    MemoryPoolMarshal() = default;
    MemoryPoolMarshal(lwn::Device* cppDevice, lwn::MemoryPool* cppPool)
        : MemoryPoolMarshal()
    {
        const auto d = reinterpret_cast<LWNdevice*>(cppDevice);
        const auto p = reinterpret_cast<LWNmemoryPool*>(cppPool);

        const auto GetPdes = [=](mapping_t t) {
            return Get<pde_t>(devtools->MemoryPoolGetNumPdes, d, p, t, LWNdevtoolsMemoryPoolPageType::Pde);
        };
        const auto GetPtes = [=](mapping_t t) {
            return Get<pte_t>(devtools->MemoryPoolGetNumPtes, d, p, t, LWNdevtoolsMemoryPoolPageType::Pte);
        };

        static_assert(__LWN_DEVTOOLS_MEMORY_POOL_MAPPINGS_COUNT == COUNT, "driver missing some mappings?");
        static_assert(MAP.size() == COUNT, "mismatch, missing some mappings");
        std::transform(MAP.begin(), MAP.end(), tables.begin(), [=](mapping_t t) {
            return Tables{ GetPdes(t), GetPtes(t) };
        });

        devtools->MemoryPoolGetComptagInfo(d, p, &compbits.info);

        compbits.bits.resize(compbits.info.size);
        devtools->MemoryPoolReadComptags(d, p, 0, compbits.info.size, &compbits.bits[0]);

        base = cppPool->GetBufferAddress();
        size = cppPool->GetSize();
        flags = cppPool->GetFlags();
    }

    uint64_t GetIova()
    {
        for (const auto& t: tables)
            for (const auto& pte: t.ptes)
                if (pte.valid)
                    return pte.iovaAddr & ~0x400000000; // LW_GPU_PHYSICAL_ADDRESS_MARKER_BIT
        return 0u;
    }

    bool Reserve(lwn::Device* cppDevice)
    {
        const auto d = reinterpret_cast<LWNdevice*>(cppDevice);

        auto reserve_pitch = devtools->ReserveGpuVa(d, base + 0 * size, size, 0);
        auto reserve_block = devtools->ReserveGpuVa(d, base + 1 * size, size, 0);
        auto reserve_special = devtools->ReserveGpuVa(d, base + 2 * size, size, 0);

        auto reserve_iova = devtools->ReserveIova(d, GetIova(), size, 0);

        auto info = compbits.info;
        auto reserve_tags = devtools->ReserveComptags(d, info.iova, info.pages * info.size, 0);

        return reserve_pitch && reserve_block && reserve_special && reserve_special && reserve_iova && reserve_tags;
    }

    LWNmemoryPool Reconstruct(lwn::Device* cppDevice, void* store)
    {
        LWNmemoryPool pool;

        const auto d = reinterpret_cast<LWNdevice*>(cppDevice);

        const auto firstIova = GetIova();

        LWNmemoryPoolBuilder builder;
        lwnMemoryPoolBuilderSetDefaults(&builder);
        lwnMemoryPoolBuilderSetDevice(&builder, d);
        lwnMemoryPoolBuilderSetFlags(&builder, flags);
        lwnMemoryPoolBuilderSetStorage(&builder, store, size);
        std::array<LWNdevtoolsMemoryPoolReservedMapping,3> reservedMappings;
        for (int i = 0; i < 3; ++i) {
            auto& mapping = reservedMappings[i];
            mapping.gpuva = base + i * size;
            mapping.size = size;
            mapping.iova = firstIova;
            mapping.comptagline = 0;
            mapping.mapping = MAP[i];
        }

        auto info = compbits.info;
        reservedMappings[2].comptagline = info.offset / info.comptagsPerCacheline;

        devtools->MemoryPoolBuilderSetReservedMappings(&builder, 3, &reservedMappings[0]);

        lwnMemoryPoolInitialize(&pool, &builder);
        devtools->MemoryPoolWriteComptags(d, &pool, 0, info.size, &compbits.bits[0]);

        return pool;
    }
};
constexpr std::array<LWNdevtoolsMemoryPoolMapping,MemoryPoolMarshal::COUNT> MemoryPoolMarshal::MAP;

std::ostream& operator<<(std::ostream& o, const std::vector<uint8_t>& v)
{
    o << "v<u8>:" << v.size() << " 0x{" << std::hex;
    for (auto c: v)
        o << " " << uint16_t(c);
    o << "}" << std::dec << std::endl;
    return o;
}

template<class T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v)
{
    for (size_t s = 0; s < v.size(); ++s)
        o << "[" << s << "]=" << v[s] << std::endl;
    return o;
}

std::ostream& operator<<(std::ostream& o, const MemoryPoolMarshal::Tables& t)
{
    o << "{" << std::endl
      << "pdes=" << t.pdes << std::endl
      << "ptes=" << t.ptes << std::endl
      << "}";
    return o;
}

std::ostream& operator<<(std::ostream& o, const LWNdevtoolsDecodedPde& d)
{
    o << "[pdeIndex=" << d.pdeIndex
      << " size=" << d.size
      << " pageSize=" << d.pageSize
      << " numPtes=" << d.numPtes
      << " bigValid=" << d.bigValid
      << std::hex
      << " bigIovaAddr=" << d.bigIovaAddr
      << std::dec
      << " smallValid=" << d.smallValid
      << std::hex
      << " smallIovaAddr=" << d.smallIovaAddr
      << std::dec
      << "]";
    return o;
}

std::ostream& operator<<(std::ostream& o, const LWNdevtoolsDecodedPte& t)
{
    o << "[pdeIndex=" << t.pdeIndex
      << " valid=" << t.valid
      << " sparse=" << t.sparse
      << " iova=" << t.iova
      << " cacheable=" << t.cacheable
      << " readable=" << t.readable
      << " writeable=" << t.writeable
      << std::hex
      << " iovaAddr=" << t.iovaAddr
      << std::dec
      << " kind=" << t.kind
      << " comptagline=" << t.comptagline << "]";
    return o;
}

std::ostream& operator<<(std::ostream& o, const LWNdevtoolsComptagInfo& i)
{
    o << "[ltcCount=" << i.ltcCount
      << " slicesPerLtc=" << i.slicesPerLtc
      << " compCacheLineSizePerLtc=" << i.compCacheLineSizePerLtc
      << " cbcBase=" << i.cbcBase
      << " cbcSize=" << i.cbcSize
      << " cbcBasePostDivide=" << i.cbcBasePostDivide
      << " comptagsPerCacheline=" << i.comptagsPerCacheline
      << " compressionPageSize=" << i.compressionPageSize
      << " comptagLineSize=" << i.comptagLineSize
      << " bitsPerComptagLine=" << i.bitsPerComptagLine
      << " bitsPerRamEntry=" << i.bitsPerRamEntry
      << " ramBankWidth=" << i.ramBankWidth
      << " ramEntiresPerCompCacheLine=" << i.ramEntiresPerCompCacheLine
      << " offsetAllocated=" << i.offsetAllocated
      << " offset=" << i.offset
      << " lines=" << i.lines
      << " pages=" << i.pages
      << " size=" << i.size
      << " iova=" << i.iova << "]";
    return o;
}

std::ostream& operator<<(std::ostream& o, const MemoryPoolMarshal::Compbits& b)
{
    o << "info=" << b.info << std::endl
      << "bits=" << b.bits;
    return o;
}

static const char* testDescr = R"(Generate random (but seeded, hence reproductible)
rectangles with clears, to make a compressible texture.
The render target is then serialized, compbits included,
everything is cleaned up, and we rebuild memory pools
and textures from scratch with the newly introduced
devtools hooks.
)";

struct PTE
{
    lwString getDescription() const { return testDescr; }
    int isSupported(void) const { return lwogCheckLWNAPIVersion(53, 309); }
    void initGraphics(void) const { lwnDefaultInitGraphics(); }
    void exitGraphics(void) const { lwnDefaultExitGraphics(); }

    void capture(void);
    void replay(void);
    void doGraphics(void) { capture(); replay(); }

    std::unique_ptr<MemoryPoolMarshal> structure;
    std::vector<uint8_t> content;
};
OGTEST_CppTest(PTE, lwn_pte, );

static void draw(LWNtextureHandle texHandle)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // rebind default framebuffer
    g_lwnWindowFramebuffer.bind();

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec4 position;\n"
        "out vec2 texcoord;\n"
        "void main() {\n"
        "  gl_Position = position;\n"
        "  texcoord = vec2(position);\n"
        "}\n";

    FragmentShader fs_texture(440);
    fs_texture << "precision highp float;\n"
        "layout(location = 0) out vec4 color;\n"
        "layout (binding=0) uniform sampler2D tex;\n"
        "in vec2 texcoord;\n"
        "void main() {\n"
        "  color = vec4(texture(tex, texcoord));\n"
        "}\n";

    Program *pgmTexture = device->CreateProgram();
    auto compiled = g_glslcHelper->CompileAndSetShaders(pgmTexture, vs, fs_texture);
    if (!compiled) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec4 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec4(-1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(-1.0, +1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, +1.0, 0.0, 1.0) },
    };

    MemoryPoolAllocator vertex_allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexStreamSet streamSet(stream);
    VertexArrayState vertex = streamSet.CreateVertexArrayState();

    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vertex_allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindProgram(pgmTexture, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    queueCB.submit();
    queue->Finish();
    pgmTexture->Free();
}

void PTE::capture()
{
    devtools = lwnDevtoolsBootstrap();

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int texDim = 768;
    TextureBuilder textureBuilder;
    textureBuilder.SetDevice(device).SetDefaults().
        SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(texDim, texDim).
        SetFormat(Format::RGBA8);
    textureBuilder.SetFlags(TextureFlags::COMPRESSIBLE);
    const LWNuintptr poolSize = textureBuilder.GetStorageSize();

    const LWNmemoryPoolFlags flagsCompressible = LWN_MEMORY_POOL_TYPE_GPU_ONLY;
    MemoryPoolAllocator compressible_allocator(device, NULL, poolSize, flagsCompressible);

    Texture *tex = compressible_allocator.allocTexture(&textureBuilder);
    queueCB.SetRenderTargets(1, &tex, NULL, NULL, NULL);

    queueCB.SetViewportScissor(0, 0, texDim, texDim);
    LWNfloat blue[] = { 0.0, 0.0, 1.0, 1.0 };
    queueCB.ClearColor(0, blue, LWN_CLEAR_COLOR_MASK_RGBA);

    std::mt19937 mt{42};
    const int ZONES = 36;
    std::uniform_int_distribution<uint32_t> distPos{0, texDim};
    std::uniform_int_distribution<uint8_t> distColor{0, 255};
    for (int i = 0; i < ZONES; ++i) {
        {
            const uint32_t x = distPos(mt);
            const uint32_t y = distPos(mt);

            std::uniform_int_distribution<uint32_t> distH{0, texDim - x};
            const uint32_t h = distH(mt);

            std::uniform_int_distribution<uint32_t> distW{0, texDim - y};
            const uint32_t w = distW(mt);

            queueCB.SetViewportScissor(x, y, h, w);
        }

        {
            const float color[] = { distColor(mt) / 255.0f, distColor(mt) / 255.0f, distColor(mt) / 255.0f, 1.0f };

            queueCB.ClearColor(0, color, LWN_CLEAR_COLOR_MASK_RGBA);
        }
    }

    queueCB.submit();
    queue->Finish();

    {
        MemoryPool *pool = compressible_allocator.pool(tex);

        BufferBuilder bb;
        bb.SetDevice(device).SetDefaults();
        bb.SetStorage(pool, 0, poolSize);

        Buffer src;
        src.Initialize(&bb);

        content.resize(poolSize);
        MemoryPoolAllocator bufAllocator{device, &content[0], poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT};
        Buffer *dst = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, poolSize);

        queueCB.CopyBufferToBuffer(src.GetAddress(), dst->GetAddress(), poolSize, LWN_COPY_FLAGS_NONE);
        queueCB.submit();
        queue->Finish();

        structure = std::make_unique<MemoryPoolMarshal>(device, pool);
        src.Finalize();
    }

#if DEBUG_PRINT
    std::cout << "pitch" << structure->tables[0] << std::endl;
    std::cout << "block" << structure->tables[1] << std::endl;
    std::cout << "special" << structure->tables[2] << std::endl;
    std::cout << "compbits " << structure->compbits << std::endl;
#endif
#if DEBUG_CONTENT
    std::cout << "content " << content << std::endl;
#endif

#if DRAW_EXPECTED
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *sampler = sb.CreateSampler();

    TextureHandle texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), sampler->GetRegisteredID());
    draw(texHandle);
#endif
}

void PTE::replay()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

#if GLITCH_BASE
    structure->base += 1 << 16; // One Page up.
#endif
    structure->Reserve(device);

#if GLITCH_CONTENT
    auto size = structure->size;
    std::fill_n(content.begin() + size/10, size/5, 0x5a);
#endif
#if GLITCH_COMPBITS
    auto& bits = structure->compbits.bits;
    auto cbSize = bits.size();
    std::fill_n(bits.begin() + cbSize/10, cbSize/5, 0xaa);
#endif
    LWNmemoryPool pool = structure->Reconstruct(device, &content[0]);

    Texture tex;
    {
        const int texDim = 768;
        TextureBuilder textureBuilder;
        textureBuilder.SetDevice(device).SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetSize2D(texDim, texDim).
            SetFormat(Format::RGBA8).
            SetStorage((MemoryPool*)&pool, 0).
            SetFlags(TextureFlags::COMPRESSIBLE);
        tex.Initialize(&textureBuilder);
    }

    Sampler sampler;
    {
        SamplerBuilder sb;
        sb.SetDevice(device).
            SetDefaults().
            SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
        sampler.Initialize(&sb);
    }

    auto samplerPool = g_lwnTexIDPool->GetSamplerPool();
    samplerPool->RegisterSampler(420, &sampler);

    auto texturePool = g_lwnTexIDPool->GetTexturePool();
    texturePool->RegisterTexture(420, &tex, nullptr);

    TextureHandle texHandle = device->GetTextureHandle(420, 420);
    queueCB.SetSamplerPool(samplerPool);
    queueCB.SetTexturePool(texturePool);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);

#if !DRAW_EXPECTED
    draw(texHandle);
#else
    (void)texHandle;
#endif

    sampler.Finalize();
    tex.Finalize();
    lwnMemoryPoolFinalize(&pool);
}
#endif // CHEETAH && DEVTOOLS
