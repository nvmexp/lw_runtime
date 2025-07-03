/*
* Copyright (c) 2017 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include <vector>
#include <memory>

#include "cmdline.h"
#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include "lwnExt/lwnExt_Internal.h"

using namespace lwn;

#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

static PFNLWNDEVICEBUILDERSETRESERVEDADDRESSRANGESLWXPROC  s_lwnDeviceBuilderSetReservedAddressRangesLWX = NULL;
static PFNLWNDEVICEBUILDERGETRESERVEDADDRESSRANGESLWXPROC  s_lwnDeviceBuilderGetReservedAddressRangesLWX = NULL;
static PFNLWNMEMORYPOOLBUILDERSETPITCHADDRESSLWXPROC       s_lwnMemorypoolBuilderSetPitchAddressLWX = NULL;
static PFNLWNMEMORYPOOLBUILDERSETBLOCKLINEARADDRESSLWXPROC s_lwnMemorypoolBuilderSetBlockLinearAddressLWX = NULL;
static PFNLWNMEMORYPOOLBUILDERSETSPECIALADDRESSLWXPROC     s_lwnMemorypoolBuilderSetSpecialAddressLWX = NULL;
static PFNLWNMEMORYPOOLGETBLOCKLINEARADDRESSLWXPROC        s_lwnMemoryPoolGetBlockLinearAddressLWX = NULL;
static PFNLWNMEMORYPOOLGETSPECIALADDRESSLWXPROC            s_lwnMemoryPoolGetSpecialAddressLWX = NULL;
static PFLWNMEMORYPOOLBUILDERGETRESERVEDADDRESSESLWXPROC   s_lwnMemoryPoolBuilderGetReservedAddressesLWX = NULL;

namespace
{
#if defined(LW_WINDOWS)
    // Helper function to check at runtime if the OS is Windows 10
    typedef LONG (WINAPI* RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);

    bool isWindows10() {
        HMODULE hMod = ::GetModuleHandleW(L"ntdll.dll");
        if (hMod) {
            RtlGetVersionPtr RtlGetVersion = (RtlGetVersionPtr)::GetProcAddress(hMod, "RtlGetVersion");

            if (RtlGetVersion != nullptr) {
                RTL_OSVERSIONINFOW rovi = { 0 };
                rovi.dwOSVersionInfoSize = sizeof(rovi);
                if ( 0 == RtlGetVersion(&rovi) ) {
                    return (rovi.dwMajorVersion == 10);
                }
            }
        }

        return false;
    }
#endif

    class MemPool
    {
    public:

        enum MappingKind{
            Pitch       = 0,
            BlockLinear = 1,
            Special     = 2
        };

#if defined(LW_TEGRA)
        // On HOS predefined addresses are supported for all 3 mapping kinds.
        static const int NUM_MAPPING_TYPES = 3;
#else
        // On Windows predefined addresses are only supported for pitch linear
        // mappings.
        static const int NUM_MAPPING_TYPES = 1;
#endif

        struct Mapping
        {
            LWNbufferAddress addresses[NUM_MAPPING_TYPES];
        };

        MemPool(Device *device, size_t size, MemoryPoolFlags flags) :
            m_size(lwnUtil::AlignSize(size, LWN_MEMORY_POOL_STORAGE_GRANULARITY)),
            m_asset(NULL),
            m_flags(flags),
            m_poolCreated(false),
            m_poolMapped(false)
        {
            createMemPool(device, NULL);
        }

        MemPool(Device *device, size_t size, MemoryPoolFlags flags, const Mapping& gpuVa) :
            m_size(lwnUtil::AlignSize(size, LWN_MEMORY_POOL_STORAGE_GRANULARITY)),
            m_asset(NULL),
            m_flags(flags),
            m_poolCreated(false),
            m_poolMapped(false)
        {
            createMemPool(device, &gpuVa);
        }

        ~MemPool()
        {
            if (m_poolMapped) {
                unmapVirtual();
            }

            m_pool.Finalize();

            if (m_asset) {
                lwnUtil::AlignedStorageFree(m_asset);
            }
        }

        MemoryPool* getPool() { return &m_pool; }
        size_t      getSize() const { return m_size;  }

        // Returns the address range for the mapping kind
        LWNpoolAddressRange getRange(MappingKind kind) const
        {
            LWNpoolAddressRange range = {};

            if (m_poolCreated) {
                // Only return an address range if the pool was successfully initialized
                range.bufferRange.address = getAddress(kind);
                range.bufferRange.size = m_size;

                if (m_flags & MemoryPoolFlags::VIRTUAL) {
                    range.flags |= MemoryPoolFlags::VIRTUAL;
                }
            }

            return range;
        }

        // Tests if any mapping of this pool intersects with range.
        bool intersects(LWNbufferRange const& range) const
        {
            for (int k = 0; k < NUM_MAPPING_TYPES; ++k) {
                BufferAddress my_addr = getAddress(MappingKind(k));

                // range is above the address range of this pool
                if (range.address >= (my_addr + m_size)) {
                    continue;
                }
                // range is below the address range of this pool
                if ((range.address + range.size) <= my_addr) {
                    continue;
                }

                return true;
            }

            return false;
        }

        // Maps the physical pool into this virtual pool.
        bool mapVirtual(MemoryPool* physicalPool) {

            assert(m_flags & MemoryPoolFlags::VIRTUAL);

            MappingRequest req;

            req.physicalOffset = 0;
            req.physicalPool = physicalPool;
            req.size = m_size;
            req.storageClass = LWN_STORAGE_CLASS_BUFFER;
            req.virtualOffset = 0;

            m_poolMapped = (LWN_TRUE == m_pool.MapVirtual(1, &req));

            return m_poolMapped;
        }

        bool unmapVirtual()
        {
            MappingRequest req;

            req.physicalPool = NULL;
            req.virtualOffset = 0;
            req.physicalOffset = 0;
            req.size = m_size;
            req.storageClass = 0;

            m_poolMapped = !(LWN_TRUE == m_pool.MapVirtual(1, &req));

            return !m_poolMapped;
        }

    private:

        void createMemPool(Device *device, const Mapping *gpuVa)
        {
            if (!(m_flags & MemoryPoolFlags::VIRTUAL)) {
                m_asset = lwnUtil::AlignedStorageAlloc(m_size, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
            }

            MemoryPoolBuilder mb;
            mb.SetDevice(device).SetDefaults()
              .SetFlags(m_flags)
              .SetStorage(m_asset, m_size);

            if (gpuVa && !(m_flags & MemoryPoolFlags::PHYSICAL)) {
                // Assign the addresses for the different mappings to the builder. Setting a mapping to 0 will have no effect.
                s_lwnMemorypoolBuilderSetPitchAddressLWX(reinterpret_cast<LWNmemoryPoolBuilder*>(&mb), gpuVa->addresses[Pitch]);
#if defined(LW_TEGRA)
                s_lwnMemorypoolBuilderSetBlockLinearAddressLWX(reinterpret_cast<LWNmemoryPoolBuilder*>(&mb), gpuVa->addresses[BlockLinear]);
                s_lwnMemorypoolBuilderSetSpecialAddressLWX(reinterpret_cast<LWNmemoryPoolBuilder*>(&mb), gpuVa->addresses[Special]);
#endif
            }

            m_poolCreated = (LWN_TRUE == m_pool.Initialize(&mb));

            if (!m_poolCreated && m_asset) {
                lwnUtil::AlignedStorageFree(m_asset);
                m_asset = NULL;
            }
        }

        BufferAddress getAddress(MappingKind kind) const
        {
            switch (kind)
            {
            case Pitch:
                return m_pool.GetBufferAddress();

            case BlockLinear:
                return s_lwnMemoryPoolGetBlockLinearAddressLWX(reinterpret_cast<const LWNmemoryPool*>(&m_pool));

            case Special:
                return s_lwnMemoryPoolGetSpecialAddressLWX(reinterpret_cast<const LWNmemoryPool*>(&m_pool));

            default:
                assert(!"Unknow kind");
            }

            return 0;
        }

        const size_t            m_size;
        void*                   m_asset;
        const MemoryPoolFlags   m_flags;
        MemoryPool              m_pool;
        bool                    m_poolCreated;
        bool                    m_poolMapped;
    };

    typedef std::unique_ptr<MemPool> ScopedMemPoolPtr;
    typedef std::vector<ScopedMemPoolPtr> MemPoolList;

    class LWNmemPoolMappingTest
    {
    public:
        explicit LWNmemPoolMappingTest(size_t numPools) : m_cellX(0), m_cellY(0), m_numPools(numPools)
        {
            m_heapRange = {};
            m_shaderHeapRange = {};
            m_virtualHeapRange = {};
        }

        ~LWNmemPoolMappingTest();

        bool    init();
        void    intersection();
        void    pitchMapping();
        void    mapIntoLargeRange();
        void    mapIntoOverlappedRanges();

    private:
        static const size_t     LARGE_POOL_SIZE = 64 * 1024 * 1024;
        static const size_t     PHYS_POOL_SIZE  = 512 * 1024;
        static const size_t     BUFFER_SIZE = 512;
        static const int        CELL_WIDTH  = 128;
        static const int        CELL_HEIGHT = 128;

        int                     m_cellX;
        int                     m_cellY;

        Device                  m_localDevice;
        Queue                   m_localQueue;

        const size_t            m_numPools;

        LWNpoolAddressRange     m_heapRange;
        LWNpoolAddressRange     m_shaderHeapRange;
        LWNpoolAddressRange     m_virtualHeapRange;

        // List of address ranges that can be used to create memory pools
        std::vector<LWNpoolAddressRange> m_memPoolRangeList;

        void    drawResult(bool success);
        void    drawQuads(const std::vector<Buffer*>& bufferList);
    };

    struct PoolEntry
    {
        size_t          size;
        MemoryPoolFlags flags;
    };

    static const PoolEntry s_poolList[] = {
        {   64 * 1024, (MemoryPoolFlags::CPU_UNCACHED  | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::COMPRESSIBLE) },
        {        1024, (MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED) },
        {        2048, (MemoryPoolFlags::CPU_CACHED    | MemoryPoolFlags::GPU_UNCACHED) },
        {   64 * 1024, (MemoryPoolFlags::CPU_UNCACHED  | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::SHADER_CODE | MemoryPoolFlags::COMPRESSIBLE) },
        {        4096, (MemoryPoolFlags::CPU_CACHED    | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::SHADER_CODE)  },
        {   64 * 1024, (MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::VIRTUAL) },
        {  256 * 1024, (MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::VIRTUAL) }
    };
} // namespace


LWNmemPoolMappingTest::~LWNmemPoolMappingTest()
{
    m_localQueue.Finalize();
    m_localDevice.Finalize();
}

bool LWNmemPoolMappingTest::init()
{
    DeviceBuilder db;
    db.SetDefaults();

    if (!m_localDevice.Initialize(&db)) {
        return false;
    }

    MemPoolList memPoolList(m_numPools);

    // Create memory pools and get their address ranges in order to have some valid
    // ranges that can be used for the tests.
    for (size_t i = 0; i < m_numPools; ++i) {
        memPoolList[i] = ScopedMemPoolPtr(new MemPool(&m_localDevice, s_poolList[i].size, s_poolList[i].flags));

        const LWNpoolAddressRange range = memPoolList[i]->getRange(MemPool::MappingKind::Pitch);

        // Store address range of the memory pool.
        m_memPoolRangeList.push_back(range);
    }

    // Create a large pool inside the shader heap, one outside the shader heap and one virtual to get some valid address ranges
    // that can be used later to test the creation of multiple pools inside the different ranges.
    MemPool *tmp1 = new MemPool(&m_localDevice, LARGE_POOL_SIZE, MemoryPoolFlags::CPU_UNCACHED  | MemoryPoolFlags::GPU_CACHED);
    MemPool *tmp2 = new MemPool(&m_localDevice, LARGE_POOL_SIZE, MemoryPoolFlags::CPU_UNCACHED  | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::SHADER_CODE);
    MemPool *tmp3 = new MemPool(&m_localDevice, LARGE_POOL_SIZE, MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::VIRTUAL);

    m_heapRange        = tmp1->getRange(MemPool::MappingKind::Pitch);
    m_shaderHeapRange  = tmp2->getRange(MemPool::MappingKind::Pitch);
    m_virtualHeapRange = tmp3->getRange(MemPool::MappingKind::Pitch);

    m_memPoolRangeList.push_back(m_heapRange);
    m_memPoolRangeList.push_back(m_shaderHeapRange);
    m_memPoolRangeList.push_back(m_virtualHeapRange);

    delete tmp1;
    delete tmp2;
    delete tmp3;

    memPoolList.clear();

    m_localDevice.Finalize();

    s_lwnDeviceBuilderSetReservedAddressRangesLWX(reinterpret_cast<LWNdeviceBuilder*>(&db), static_cast<int>(m_memPoolRangeList.size()), m_memPoolRangeList.data());

    // Create new device and reserve all address ranges in m_memPoolRangeList
    if (!m_localDevice.Initialize(&db)) {
        return false;
    }

    QueueBuilder qb;
    qb.SetDefaults().SetDevice(&m_localDevice);

    m_localQueue.Initialize(&qb);

    return true;
}

void LWNmemPoolMappingTest::intersection()
{
    // Verify the Get functions to retrieve data of the DeviceBuilder and the MemoryPoolBuilder are working correctly.
    DeviceBuilder db;

    db.SetDefaults();
    s_lwnDeviceBuilderSetReservedAddressRangesLWX(reinterpret_cast<LWNdeviceBuilder*>(&db), static_cast<int>(m_memPoolRangeList.size()), m_memPoolRangeList.data());

    const LWNpoolAddressRange *ranges = NULL;
    int numRanges = s_lwnDeviceBuilderGetReservedAddressRangesLWX(reinterpret_cast<LWNdeviceBuilder*>(&db), &ranges);

    bool success = (numRanges == int(m_memPoolRangeList.size())) && (ranges == m_memPoolRangeList.data());

    MemoryPoolBuilder mpb;
    mpb.SetDevice(&m_localDevice).SetDefaults();

    s_lwnMemorypoolBuilderSetPitchAddressLWX(reinterpret_cast<LWNmemoryPoolBuilder*>(&mpb), m_memPoolRangeList[0].bufferRange.address);
#if defined(LW_TEGRA)
    s_lwnMemorypoolBuilderSetBlockLinearAddressLWX(reinterpret_cast<LWNmemoryPoolBuilder*>(&mpb), m_memPoolRangeList[1].bufferRange.address);
    s_lwnMemorypoolBuilderSetSpecialAddressLWX(reinterpret_cast<LWNmemoryPoolBuilder*>(&mpb), m_memPoolRangeList[2].bufferRange.address);
#endif

    LWNbufferAddress pl = 0xff, bl = 0xff, special = 0xff;
    s_lwnMemoryPoolBuilderGetReservedAddressesLWX(reinterpret_cast<LWNmemoryPoolBuilder*>(&mpb), &pl, &bl, &special);

    success = success && m_memPoolRangeList[0].bufferRange.address == pl;
#if defined(LW_TEGRA)
    success = success && (m_memPoolRangeList[1].bufferRange.address == bl);
    success = success && (m_memPoolRangeList[2].bufferRange.address == special);
#else
    success = success && (0 == bl);
    success = success && (0 == special);
#endif

    MemPoolList memPoolList(m_numPools);

    // Create memory pools which would typically get mapped to the same addresses as the previously
    // created pools. Since the ranges are reserved they are supposed to end up at different locations.
    for (unsigned int i = 0; i < m_numPools; ++i) {
        memPoolList[i] = ScopedMemPoolPtr(new MemPool(&m_localDevice, s_poolList[i].size, s_poolList[i].flags));

        // Make sure this address does not intersect with a range in the list of reserved addresses.
        std::vector<LWNpoolAddressRange>::const_iterator itr;

        for (itr = m_memPoolRangeList.begin(); itr != m_memPoolRangeList.end(); ++itr) {
            if (memPoolList[i]->intersects(itr->bufferRange)) {
                break;
            }
        }
        success = success && (itr == m_memPoolRangeList.end());
    }

    drawResult(success);
}

void LWNmemPoolMappingTest::pitchMapping()
{
    // Create memory pools with fixed pitch mapping address in ilwerse order to make sure they don't
    // get the expected addresses assigned by chance.
    MemPoolList  memPoolList(m_numPools);

    bool success = true;

    for (int i = (static_cast<int>(m_numPools) - 1); i >= 0 && success; --i) {
        // Create memory pool and ask to get the pitch mapping at m_memPoolRangeList[i].bufferRange.address.
        // No address is provided for block linear and special mapping.
        MemPool::Mapping mapping = {};
        mapping.addresses[MemPool::MappingKind::Pitch] = m_memPoolRangeList[i].bufferRange.address;

        memPoolList[i] = ScopedMemPoolPtr(new MemPool(&m_localDevice, s_poolList[i].size, s_poolList[i].flags, mapping));

        LWNpoolAddressRange range = memPoolList[i]->getRange(MemPool::MappingKind::Pitch);
        success = success && (range.bufferRange.address == m_memPoolRangeList[i].bufferRange.address);
        success = success && (range.bufferRange.size    == m_memPoolRangeList[i].bufferRange.size);
    }

    drawResult(success);
}

void LWNmemPoolMappingTest::mapIntoLargeRange()
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    // Create physical pool used to validate virtual pools
    MemPool physicalPool = MemPool(&m_localDevice, PHYS_POOL_SIZE, MemoryPoolFlags::GPU_NO_ACCESS | MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::PHYSICAL);

    // Create memory pools with fixed addresses inside a large reserved address range. The pools are
    // first created with pitch mapping, then with BlockLinear Mapping and finally with Special mapping.
    int pageSize = 0;
    m_localDevice.GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);

    bool success = true;
    for (int i = 0; i < MemPool::NUM_MAPPING_TYPES && success; ++i) {
        uint64_t heapOffset        = m_heapRange.bufferRange.address;
        uint64_t shaderHeapOffset  = m_shaderHeapRange.bufferRange.address;
        uint64_t virtualHeapOffset = m_virtualHeapRange.bufferRange.address;

        MemPoolList memPoolList(m_numPools);
        std::vector<Buffer*> bufferList;

        for (unsigned int j = 0; j < m_numPools && success; ++j) {
            uint64_t *offset = &heapOffset;
            MemPool::MappingKind kind = MemPool::MappingKind(i);

            // Special mapping is only valid for compressible pools.
            if ((kind == MemPool::MappingKind::Special) && !(s_poolList[j].flags & MemoryPoolFlags::COMPRESSIBLE)) {
                continue;
            }
            // The pitch mapping of a pool with the SHADER_CODE flag set must be inside the shader heap.
            if ((kind == MemPool::MappingKind::Pitch) && (s_poolList[j].flags & MemoryPoolFlags::SHADER_CODE) == MemoryPoolFlags::SHADER_CODE) {
                offset = &shaderHeapOffset;
            }
            if ((s_poolList[j].flags & MemoryPoolFlags::VIRTUAL) == MemoryPoolFlags::VIRTUAL) {
                // Virtual pools accept only pitch mapping addresses
                if (kind != MemPool::MappingKind::Pitch) {
                    continue;
                }
                // Virtual pools need to be allocated inside the virtualHeapRange
                offset = &virtualHeapOffset;
            }

            MemPool::Mapping mapping = {};
            mapping.addresses[i] = *offset;

            memPoolList[j] = ScopedMemPoolPtr(new MemPool(&m_localDevice, s_poolList[j].size, s_poolList[j].flags, mapping));

            if (s_poolList[j].flags & MemoryPoolFlags::VIRTUAL) {
                memPoolList[j]->mapVirtual(physicalPool.getPool());
            }

            // Check if the memory pool was successfully initialized and got the requested address.
            LWNpoolAddressRange range = memPoolList[j]->getRange(kind);
            success = success && (range.bufferRange.address == *offset);

            *offset += lwnUtil::AlignSize(s_poolList[j].size, pageSize);

            // Create a buffer inside the pitch mapping of each pool. The buffer
            // contains a color vector (green).
            if (kind == MemPool::MappingKind::Pitch) {
                BufferBuilder bb;

                bb.SetDefaults().SetDevice(&m_localDevice);
                Buffer *buf = bb.CreateBufferFromPool(memPoolList[j]->getPool(), 0, BUFFER_SIZE);

                if (s_poolList[j].flags & MemoryPoolFlags::CPU_NO_ACCESS) {
                    // Instead of using a temporary buffer to copy the data to the GPU only buffer,
                    // just use the buffer that was created in the first memory pool which is CPU
                    // accessible.
                    queueCB.CopyBufferToBuffer(bufferList[0]->GetAddress(), buf->GetAddress(), BUFFER_SIZE, CopyFlags::NONE);
                } else { 
                    void *ptr = buf->Map();

                    if (ptr) {
                        memset(ptr, 0, BUFFER_SIZE);

                        dt::vec4 *colPtr = static_cast<dt::vec4*>(ptr);
                        *colPtr = dt::vec4(0.0f, 1.0f, 0.0f, 1.0f);

                        if (s_poolList[j].flags & MemoryPoolFlags::CPU_CACHED) {
                            buf->FlushMappedRange(0, sizeof(dt::vec4));

                            // Since we use a seperate device for this test, the local queue needs to be
                            // finished explicitely to make sure the copy triggered by FlushMappedRange
                            // is done before any rendering commands are submitted.
                            m_localQueue.Finish();
                        }
                    }
                }

                success = success && (buf->GetAddress() == mapping.addresses[MemPool::MappingKind::Pitch]);

                bufferList.push_back(buf);
            }
        }

        // Draw quads using the color stored in the previously created buffers.
        drawQuads(bufferList);

        for (auto itr : bufferList) {
            itr->Free();
        }
        bufferList.clear();
    }

    drawResult(success);
}

void LWNmemPoolMappingTest::mapIntoOverlappedRanges()
{
    // Create physical pool used to validate mapping of virtual pools
    MemPool physicalPool = MemPool(&m_localDevice, PHYS_POOL_SIZE, MemoryPoolFlags::GPU_NO_ACCESS | MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::PHYSICAL);

    // Create memory pools at fixed addresses inside a reserved address range, delete them and re-create them with an
    // offset of one page so that they would overlap with the initial pools. The pools are created with all valid mappings.
    int pageSize = 0;
    m_localDevice.GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);

    bool success = true;
    for (int j = 0; j < 2 && success; ++j) {
        uint64_t heapOffset        = m_heapRange.bufferRange.address + j * pageSize;
        uint64_t shaderHeapOffset  = m_shaderHeapRange.bufferRange.address + j * pageSize;
        uint64_t virtualHeapOffset = m_virtualHeapRange.bufferRange.address + j * pageSize;

        MemPoolList  memPoolList(m_numPools);

        for (unsigned int i = 0; i < m_numPools && success; ++i) {
            MemPool::Mapping mapping = {};

            if ((s_poolList[i].flags & MemoryPoolFlags::VIRTUAL) == MemoryPoolFlags::VIRTUAL) {
                mapping.addresses[MemPool::MappingKind::Pitch] = virtualHeapOffset;
                virtualHeapOffset += lwnUtil::AlignSize(s_poolList[i].size, pageSize);
            } else {
#if defined(LW_TEGRA)
                mapping.addresses[MemPool::MappingKind::BlockLinear] = heapOffset;
                heapOffset += lwnUtil::AlignSize(s_poolList[i].size, pageSize);

                if ((s_poolList[i].flags & MemoryPoolFlags::COMPRESSIBLE) == MemoryPoolFlags::COMPRESSIBLE) {
                    mapping.addresses[MemPool::MappingKind::Special] = heapOffset;
                    heapOffset += lwnUtil::AlignSize(s_poolList[i].size, pageSize);
                }
#endif
                if ((s_poolList[i].flags & MemoryPoolFlags::SHADER_CODE) == MemoryPoolFlags::SHADER_CODE) {
                    mapping.addresses[MemPool::MappingKind::Pitch] = shaderHeapOffset;
                    shaderHeapOffset += lwnUtil::AlignSize(s_poolList[i].size, pageSize);
                } else {
                    mapping.addresses[MemPool::MappingKind::Pitch] = heapOffset;
                    heapOffset += lwnUtil::AlignSize(s_poolList[i].size, pageSize);
                }
            }

            memPoolList[i] = ScopedMemPoolPtr(new MemPool(&m_localDevice, s_poolList[i].size, s_poolList[i].flags, mapping));

            // Check if we got the requested addresses
            LWNpoolAddressRange range = memPoolList[i]->getRange(MemPool::MappingKind::Pitch);
            success = success && (range.bufferRange.address == mapping.addresses[MemPool::MappingKind::Pitch]);

#if defined (LW_TEGRA)
            range = memPoolList[i]->getRange(MemPool::MappingKind::BlockLinear);
            success = success && (range.bufferRange.address == mapping.addresses[MemPool::MappingKind::BlockLinear]);

            range = memPoolList[i]->getRange(MemPool::MappingKind::Special);
            success = success && (range.bufferRange.address == mapping.addresses[MemPool::MappingKind::Special]);
#endif
            if ((s_poolList[i].flags & MemoryPoolFlags::VIRTUAL) == MemoryPoolFlags::VIRTUAL) {
                success = success && memPoolList[i]->mapVirtual(physicalPool.getPool());
            }
        }
    }

    drawResult(success);
}

void LWNmemPoolMappingTest::drawQuads(const std::vector<Buffer*>& bufferList)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs << "#extension GL_LW_shader_buffer_load : enable\n"
          "\n"
          "layout(location=0) in vec3 position;\n"
          "struct ColorBuffer {\n"
          "   vec4 color;\n"
          "};\n"
          "layout(binding=0) uniform Block {\n"
          "   ColorBuffer *colPtr;\n"
          "};\n"
          "out vec4 color;\n"
          "void main() {\n"
          "  gl_Position = vec4(position, 1.0);\n"
          "  color = colPtr->color;\n"
          "}\n";

    FragmentShader fs(440);
    fs << "in vec4 color;\n"
          "out vec4 fcolor;\n"
          "void main() {\n"
          "  fcolor = color;\n"
          "}\n";
   

    Program *program = device->CreateProgram();
    if (!program || !g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        DEBUG_PRINT("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
        return;
    }

    const int vertexCount = 4;

    struct Vertex {
        dt::vec4    position;
    };

    const Vertex vertices[] = { { dt::vec4(-1.0f, -1.0f, 0.0f, 1.0f) },
                                { dt::vec4( 1.0f, -1.0f, 0.0f, 1.0f) },
                                { dt::vec4(-1.0f,  1.0f, 0.0f, 1.0f) },
                                { dt::vec4( 1.0f,  1.0f, 0.0f, 1.0f) }
                              };

    const size_t vboSize = vertexCount * sizeof(Vertex);
    const size_t uboSize = sizeof(uint64_t);
    MemoryPoolAllocator allocator(device, NULL, vboSize + uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, allocator, vertices);
    BufferAddress vboGpuVa = vbo->GetAddress();

    BufferBuilder uboBuilder;
    uboBuilder.SetDefaults().SetDevice(device);

    Buffer *ubo = allocator.allocBuffer(&uboBuilder, lwnUtil::BufferAlignBits::BUFFER_ALIGN_UNIFORM_BIT, sizeof(uint64_t));
    BufferAddress uboGpuVa = ubo->GetAddress();

    queueCB.BindProgram(program,ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboGpuVa, vboSize);
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboGpuVa, uboSize);

    for (auto itr : bufferList) {
        BufferAddress bufAddr = itr->GetAddress();

        queueCB.UpdateUniformBuffer(uboGpuVa, uboSize, 0, sizeof(BufferAddress), &bufAddr);
        
        queueCB.SetViewportScissor(m_cellX, m_cellY, CELL_WIDTH - 1, CELL_HEIGHT - 1);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

        m_cellX += CELL_WIDTH;
        if (m_cellX >= lwrrentWindowWidth) {
            m_cellX = 0;
            m_cellY += CELL_HEIGHT;
        }
    }

    queueCB.submit();
    queue->Finish();
}

void LWNmemPoolMappingTest::drawResult(bool success)
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    queueCB.SetViewportScissor(m_cellX, m_cellY, CELL_WIDTH - 1, CELL_HEIGHT - 1);

    if (success) {
        queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
    } else {
        queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
    }

    m_cellX += CELL_WIDTH;
    if (m_cellX >= lwrrentWindowWidth) {
        m_cellX = 0;
        m_cellY += CELL_HEIGHT;
    }
}

class LWNMempoolMapping
{
public:
    LWNTEST_CppMethods();
};


lwString LWNMempoolMapping::getDescription() const
{
    lwStringBuf sb;
    sb << "Tests the capabilities of assigning fixed virtual addresses "
          "to memory pools. The test creates a couple of memory pools "
          "and stores their address ranges. After deleting these pools "
          "and the device, a new device is created and the previously "
          "stored address ranges are reserved. Now new pools are created "
          "inside these reserved ranges. If the test succeeds a green quad "
          "is drawn for each sub test.";

    return sb.str();
}

int LWNMempoolMapping::isSupported() const
{
#if defined(LW_TEGRA)
    // GL interop forces LWN to use the GL address space, which prevents the use of fixed
    // virtual addresses.
    return !useGL && lwogCheckLWNAPIVersion(53, 209);
#else
    if (isWindows10()) {
        return lwogCheckLWNAPIVersion(53, 304);
    }

    return false;
#endif
}

void LWNMempoolMapping::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    s_lwnDeviceBuilderSetReservedAddressRangesLWX  = (PFNLWNDEVICEBUILDERSETRESERVEDADDRESSRANGESLWXPROC)device->GetProcAddress("lwnDeviceBuilderSetReservedAddressRangesLWX");
    s_lwnDeviceBuilderGetReservedAddressRangesLWX  = (PFNLWNDEVICEBUILDERGETRESERVEDADDRESSRANGESLWXPROC)device->GetProcAddress("lwnDeviceBuilderGetReservedAddressRangesLWX");
    s_lwnMemorypoolBuilderSetPitchAddressLWX       = (PFNLWNMEMORYPOOLBUILDERSETPITCHADDRESSLWXPROC)device->GetProcAddress("lwnMemoryPoolBuilderSetPitchAddressLWX");
    s_lwnMemoryPoolBuilderGetReservedAddressesLWX  = (PFLWNMEMORYPOOLBUILDERGETRESERVEDADDRESSESLWXPROC)device->GetProcAddress("lwnMemoryPoolBuilderGetReservedAddressesLWX");

    bool extensionLoaded = s_lwnDeviceBuilderSetReservedAddressRangesLWX &&
                           s_lwnDeviceBuilderGetReservedAddressRangesLWX &&
                           s_lwnMemorypoolBuilderSetPitchAddressLWX && 
                           s_lwnMemoryPoolBuilderGetReservedAddressesLWX;

#if defined(LW_TEGRA)
    s_lwnMemorypoolBuilderSetBlockLinearAddressLWX = (PFNLWNMEMORYPOOLBUILDERSETBLOCKLINEARADDRESSLWXPROC)device->GetProcAddress("lwnMemoryPoolBuilderSetBlockLinearAddressLWX");
    s_lwnMemorypoolBuilderSetSpecialAddressLWX     = (PFNLWNMEMORYPOOLBUILDERSETSPECIALADDRESSLWXPROC)device->GetProcAddress("lwnMemoryPoolBuilderSetSpecialAddressLWX");
    s_lwnMemoryPoolGetBlockLinearAddressLWX        = (PFNLWNMEMORYPOOLGETBLOCKLINEARADDRESSLWXPROC)device->GetProcAddress("lwnMemoryPoolGetBlockLinearAddressLWX");
    s_lwnMemoryPoolGetSpecialAddressLWX            = (PFNLWNMEMORYPOOLGETSPECIALADDRESSLWXPROC)device->GetProcAddress("lwnMemoryPoolGetSpecialAddressLWX");

    extensionLoaded = extensionLoaded && 
                      s_lwnMemorypoolBuilderSetBlockLinearAddressLWX &&
                      s_lwnMemorypoolBuilderSetSpecialAddressLWX &&
                      s_lwnMemoryPoolGetBlockLinearAddressLWX &&
                      s_lwnMemoryPoolGetSpecialAddressLWX;
#endif
    
    if (!extensionLoaded) {
        LWNFailTest();
        return;
    }

    LWNmemPoolMappingTest memPoolTest(__GL_ARRAYSIZE(s_poolList));

    if (!memPoolTest.init()) {
        LWNFailTest();
        return;
    }

    g_lwnWindowFramebuffer.bind();
    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 0.2f, 0.2f, 0.2f, 1.0f, ClearColorMask::RGBA);

    memPoolTest.intersection();
    memPoolTest.pitchMapping();
    memPoolTest.mapIntoLargeRange();
    memPoolTest.mapIntoOverlappedRanges();

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNMempoolMapping, lwn_mempool_mapping, );
