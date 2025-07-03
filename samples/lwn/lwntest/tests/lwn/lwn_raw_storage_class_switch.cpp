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
#include <memory>

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

#define CHECK_IF_VALID(p)                                               \
        if (!p) {                                                       \
            LWNFailTest();                                              \
            DEBUG_PRINT("\nlwn_raw_storage_class failed in line %d",__LINE__); \
            return;                                                     \
        }

class LWNrawStorageClassTest
{
public:
    LWNTEST_CppMethods();

    static const int texWidth   = 128;
    static const int texHeight  = 128;
    static const int cellWidth  = 64;
    static const int cellHeight = 64;
};


namespace
{
    class MemPool
    {
    public:

        MemPool(Device *device, size_t size, MemoryPoolFlags flags) : m_asset(NULL), m_created(LWN_FALSE), m_mapped(LWN_FALSE)
        {
            if (!(flags & MemoryPoolFlags::VIRTUAL)) {
                m_asset = lwnUtil::AlignedStorageAlloc(size, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);

                memset(m_asset, 0, size);
            }

            MemoryPoolBuilder mb;
            mb.SetDefaults().SetDevice(device)
              .SetFlags(flags)
              .SetStorage(m_asset, size);

            m_created = m_pool.Initialize(&mb);
            if (!m_created && m_asset) {
                lwnUtil::AlignedStorageFree(m_asset);
                m_asset = NULL;
            }
        }

        ~MemPool()
        {
            if (m_mapped) {
                unmapVirtual();
            }

            m_pool.Finalize();

            if (m_asset) {
                lwnUtil::AlignedStorageFree(m_asset);
            }
        }

        explicit operator MemoryPool*() { return &m_pool; }
        operator bool() const { return (m_created == LWN_TRUE); }

        bool mapVirtual(MemPool &physicalPool, LWNstorageClass storageClass)
        {
            MappingRequest req = { 0 };

            req.physicalOffset = 0;
            req.physicalPool = &physicalPool.m_pool;
            req.size = m_pool.GetSize();
            req.storageClass = storageClass;
            req.virtualOffset = 0;

            m_mapped = m_pool.MapVirtual(1, &req);

            return (m_mapped == LWN_TRUE);
        }

        bool unmapVirtual()
        {
            MappingRequest req;

            req.physicalPool = NULL;
            req.virtualOffset = 0;
            req.physicalOffset = 0;
            req.size = m_pool.GetSize();
            req.storageClass = 0;

            m_mapped = m_pool.MapVirtual(1, &req);

            return (m_mapped == LWN_TRUE);
        }

    private:
        void*                   m_asset;
        MemoryPool              m_pool;
        LWNboolean              m_created;
        LWNboolean              m_mapped;
    };

    struct TestFormat
    {
        RawStorageClass rawStorageClass;
        Format          texFormat;
        bool            compressed;
    };

    const std::array<TestFormat, 6> formatlist = {{ {RawStorageClass::LINEAR,          Format::RGBA8,              false },
                                                    {RawStorageClass::COLOR_AND_ZF32,  Format::RGBA8,              false },
                                                    {RawStorageClass::COLOR_AND_ZF32,  Format::RGBA8,              true  },
                                                    {RawStorageClass::S8_AND_Z16,      Format::DEPTH16,            true  },
                                                    {RawStorageClass::Z24_AND_Z24S8,   Format::DEPTH24,            true  },
                                                    {RawStorageClass::ZF32S8,          Format::DEPTH32F_STENCIL8 , true  }
                                                 }};


    void fillTexture(const Texture *tex, RawStorageClass rawClass)
    {
        DeviceState *deviceState = DeviceState::GetActive();
        QueueCommandBuffer &queueCB = deviceState->getQueueCB();

        int w = tex->GetWidth();
        int h = tex->GetHeight();

        switch (rawClass) {
        case RawStorageClass::LINEAR:
            queueCB.SetRenderTargets(1, &tex, NULL, NULL, NULL);
            queueCB.SetViewportScissor(0, 0, w, h);
            queueCB.ClearColor(0, 0.0f, 0.0f, 1.0f, 1.0f, ClearColorMask::RGBA);
            break;

        case RawStorageClass::COLOR_AND_ZF32:
            queueCB.SetRenderTargets(1, &tex, NULL, NULL, NULL);
            queueCB.SetViewportScissor(0, 0, w, h);
            queueCB.ClearColor(0, 0.5f, 0.0f, 1.0f, 1.0f, ClearColorMask::RGBA);
            break;

        case RawStorageClass::S8_AND_Z16:
            queueCB.SetRenderTargets(0, NULL, NULL, tex, NULL);
            queueCB.SetViewportScissor(0, 0, w, h);
            queueCB.ClearDepthStencil(0.25f, LWN_TRUE, 0, 0);
            break;

        case RawStorageClass::Z24_AND_Z24S8:
            queueCB.SetRenderTargets(0, NULL, NULL, tex, NULL);
            queueCB.SetViewportScissor(0, 0, w, h);
            queueCB.ClearDepthStencil(0.5f, LWN_TRUE, 0, 0);
            break;

        case RawStorageClass::ZF32S8:
            queueCB.SetRenderTargets(0, NULL, NULL, tex, NULL);
            queueCB.SetViewportScissor(0, 0, w, h);
            queueCB.ClearDepthStencil(0.75f, LWN_TRUE, 0, 0);
            break;

        default:
            assert(!"Unsupported raw storage class!");
        }
    }

    bool checkTexture(const Texture *tex)
    {
        DeviceState *deviceState = DeviceState::GetActive();
        Device *device = deviceState->getDevice();
        QueueCommandBuffer &queueCB = deviceState->getQueueCB();
        Queue *queue = deviceState->getQueue();

        const size_t bufferSize = tex->GetStorageSize();
        MemoryPoolAllocator poolAllocator(device, NULL, bufferSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

        BufferBuilder bb;
        bb.SetDefaults().SetDevice(device);

        Buffer *buffer = poolAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, bufferSize);
        if (!buffer) {
            return false;
        }

        const CopyRegion reg = { 0, 0, 0, tex->GetWidth(), tex->GetHeight(), tex->GetDepth() };
        queueCB.CopyTextureToBuffer(tex, NULL, &reg, buffer->GetAddress(), CopyFlags::NONE);

        queueCB.submit();
        queue->Finish();

        uint32_t *ptr = (uint32_t*)buffer->Map();

        if (!ptr) {
            return false;
        }

        assert(tex->GetFormat() == Format::RGBA8);

        // Check that all pixels of the RGBA8 taexture have the same value.
        uint32_t refValue = *(ptr++);
        for (size_t i = 1; i < (buffer->GetSize() / sizeof(uint32_t)); ++i, ++ptr) {
            if (*ptr != refValue) {
                return false;
            }
        }

        return true;
    }

} // namespace

lwString LWNrawStorageClassTest::getDescription() const
{
    lwStringBuf sb;

    sb << "This test verifies that using the barrier LWN_BARRIER_ILWALIDATE_L2_CACHE "
          "resolves corruptions that are seen when data is written from L2 to memory "
          "using the wrong CheetAh Raw storage class (Bug 1968090).\n"
          "The test creates pairs of texture that share the same physical memory but "
          "use different formats where each texture format belongs to a different CheetAh "
          "raw storage class. In each iteration of the test the first texture is written. "
          "This will fill the L2 cache and each cache line is now associated with the raw"
          "storage class of the first texture. After writing the first texture, the second "
          "texture is written as well. Since both share the same physical memory the "
          "same L2 cache lines are now updated with data of the second texture but the"
          "raw storage class information is not updated. If the LWN_BARRIER_ILWALIDATE_L2_CACHE "
          "is not applied between writing the two textures, corruptions will be seen.\n "
          "If no corruption show up, a green square is drawn for each combination of formats. "
          "If corruptions are detected a red square is drawn.";

    return sb.str();
}

int LWNrawStorageClassTest::isSupported() const
{
    // The use of ILWALIDATE_L2_CACHE is only required on CheetAh
    // since dGPUs do not use RAW storage classes. To make sure the
    // new barrier does not break anything on Windows, the test is
    // supported for Windows as well.
    return lwogCheckLWNAPIVersion(53, 303);
}


void LWNrawStorageClassTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs << "layout(location=0) in vec3 position;\n"
          "layout(location=1) in vec2 texCoord;\n"
          "out vec2 tc;\n"
          "void main() {\n"
          "  gl_Position = vec4(position, 1.0);\n"
          "  tc = texCoord;\n"
          "}\n";

    FragmentShader fs(440);
    fs << "layout(binding = 0) uniform sampler2D smp;\n"
           "in vec2 tc;\n"
           "out vec4 fcolor;\n"
           "void main() {\n"
           "  vec4 texel = texture(smp, tc);\n"
           "  fcolor = texel;\n"
           "}\n";

    Program *program = device->CreateProgram();
    if (!program || !g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        DEBUG_PRINT("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());

        LWNFailTest();
        return;
    }

    const int vertexCount = 4;

    struct Vertex {
        dt::vec3 position;
        dt::vec2 texCoord;
    };

    const Vertex vertices[] = {
        { dt::vec3(-1.0f, -1.0f, 0.0f), dt::vec2(0.0f, 0.0f) },
        { dt::vec3( 1.0f, -1.0f, 0.0f), dt::vec2(1.0f, 0.0f) },
        { dt::vec3(-1.0f,  1.0f, 0.0f), dt::vec2(0.0f, 1.0f) },
        { dt::vec3( 1.0f,  1.0f, 0.0f), dt::vec2(1.0f, 1.0f) }
    };

    MemoryPoolAllocator poolAllocator(device, NULL, (128 << 10), LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, texCoord);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, poolAllocator, vertices);
    CHECK_IF_VALID(vbo);
    BufferAddress vboAddr = vbo->GetAddress();

    SamplerBuilder sb;
    sb.SetDefaults().SetDevice(device)
      .SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR)
      .SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);

    Sampler *sampler = sb.CreateSampler();
    CHECK_IF_VALID(sampler);

    // Create dummy texture used to flush dirty data of the L2 cache to memory.
    TextureBuilder tb;
    tb.SetDefaults().SetDevice(device)
      .SetSize2D(512, 512)
      .SetTarget(TextureTarget::TARGET_2D)
      .SetFormat(Format::RGBA8);

    Texture *flushTex = poolAllocator.allocTexture(&tb);

    // Create render target
    tb.SetSize2D(LWNrawStorageClassTest::texWidth, LWNrawStorageClassTest::texHeight);
    Texture *fb = poolAllocator.allocTexture(&tb);
    CHECK_IF_VALID(fb);

    int linearRtAlignment = 0;
    device->GetInteger(DeviceInfo::LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT, &linearRtAlignment);

    int pageSize = 0;
    device->GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);
    const size_t physicalPoolSize = 2 * pageSize;

    // Create physical pool. The memory of this pool will be shared by the two test textures.
    MemPool physicalPool(device, physicalPoolSize, MemoryPoolFlags::GPU_NO_ACCESS | MemoryPoolFlags::CPU_CACHED |
                                                   MemoryPoolFlags::COMPRESSIBLE  | MemoryPoolFlags::PHYSICAL);

    CHECK_IF_VALID(physicalPool);

    auto initializeTexBuilder = [&device, linearRtAlignment](const TestFormat &fmt, TextureBuilder &tb) {
        tb.SetDefaults()
          .SetDevice(device)
          .SetSize2D(LWNrawStorageClassTest::texWidth, LWNrawStorageClassTest::texHeight)
          .SetTarget(TextureTarget::TARGET_2D)
          .SetFormat(fmt.texFormat);

        if (fmt.rawStorageClass == RawStorageClass::LINEAR) {
            assert(fmt.texFormat == Format::RGBA8);

            const size_t stride = lwnUtil::AlignSize((LWNrawStorageClassTest::texWidth * 4), linearRtAlignment);

            tb.SetFlags(TextureFlags::LINEAR_RENDER_TARGET);
            tb.SetStride(stride);
        } else if (fmt.compressed) {
            tb.SetFlags(TextureFlags::COMPRESSIBLE);
        }
    };

    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboAddr, vertexCount * sizeof(vertices));

    g_lwnWindowFramebuffer.bind();
    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 0.2f, 0.2f, 0.2f, 1.0f, ClearColorMask::RGBA);

    int cellX = 0;
    int cellY = 0;

    const MemoryPoolFlags virtualPoolFlags = MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED |
                                             MemoryPoolFlags::COMPRESSIBLE  | MemoryPoolFlags::VIRTUAL;

    struct TexDeleter {
        void operator()(Texture* t) { t->Free(); }
    } texDeleter;

    typedef std::unique_ptr<Texture, TexDeleter&> ScopedTexture;

    for (auto fmtItr1 : formatlist) {
        MemPool virtualPool1(device, physicalPoolSize, virtualPoolFlags);

        initializeTexBuilder(fmtItr1, tb);

        virtualPool1.mapVirtual(physicalPool,tb.GetStorageClass());

        ScopedTexture tex1(tb.CreateTextureFromPool(static_cast<MemoryPool*>(virtualPool1), 0), texDeleter);
        CHECK_IF_VALID(tex1);

        for (auto fmtItr2 : formatlist) {
            if (fmtItr1.rawStorageClass != fmtItr2.rawStorageClass) {
                MemPool virtualPool2(device, physicalPoolSize, virtualPoolFlags);

                initializeTexBuilder(fmtItr2, tb);

                virtualPool2.mapVirtual(physicalPool,tb.GetStorageClass());

                ScopedTexture tex2(tb.CreateTextureFromPool(static_cast<MemoryPool*>(virtualPool2), 0), texDeleter);
                CHECK_IF_VALID(tex2);
                TextureHandle hTex2 = device->GetTextureHandle(tex2->GetRegisteredTextureID(), sampler->GetRegisteredID());

                // Ilwalidate the L2 cache to make sure that no information on the raw
                // storage class is stored in the cache when writing to the first texture.
                queueCB.Barrier(BarrierBits::ILWALIDATE_L2_CACHE);

                // Write to the first texture. This will fill the L2 cache with data of the
                // first texture and the raw storage class associated with the cache lines is
                // set to the raw storage class of this texture.
                fillTexture(tex1.get(), fmtItr1.rawStorageClass);

                // WIFI and ilwalidate L2 cache. Since writing to the L2 will only update
                // the cache line but not the associated information on the raw storage class,
                // the cache needs to be ilwalidated before writing the second texture that has
                // a different raw storage class.
                queueCB.Barrier(BarrierBits::ILWALIDATE_L2_CACHE);

                // Write the second texture that has a different raw storage class. Since the L2
                // was ilwalidated, this will allocate new cache lines and the raw storage class
                // associated with the cache lines is set to the raw storage class of this texture.
                fillTexture(tex2.get(), fmtItr2.rawStorageClass);

                // Write to a dummy texture. This will flush the dirty cache lines. These lines
                // are now written back to memory. Without having ilwalidated the L2, the wrong
                // raw storage class would be used here and the second texture would show corruptions.
                fillTexture(flushTex, RawStorageClass::COLOR_AND_ZF32);

                queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS);

                queueCB.SetRenderTargets(1, &fb, NULL, NULL, NULL);
                queueCB.SetViewportScissor(0, 0, texWidth, texHeight);
                queueCB.BindTexture(ShaderStage::FRAGMENT, 0, hTex2);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, vertexCount);

                queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS);

                g_lwnWindowFramebuffer.bind();
                queueCB.SetViewportScissor(cellX + 1, cellY + 1, cellWidth - 1, cellHeight - 1);

                bool rawStorageClassMatches = (tex2->GetRawStorageClass() == fmtItr2.rawStorageClass);

                // Check if the fb texture was rendered correctly and that the GetRawStorageClass
                // function returned the expected value.
                if (rawStorageClassMatches && checkTexture(fb)) {
                    queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
                } else {
                    queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
                }

                queueCB.submit();
                queue->Finish();

                if ((cellX + cellWidth) >= lwrrentWindowWidth) {
                    cellX = 0;
                    cellY += cellHeight;
                } else {
                    cellX += cellWidth;
                }
            }
        }
    }
}

OGTEST_CppTest(LWNrawStorageClassTest, lwn_raw_storage_class_switch, );