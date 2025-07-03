/*
 * Copyright (c) 2017-2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <lwndevtools_bootstrap.h>

#include <functional>
#include <vector>

class CreateReplayQueueValidator {
public:
    ~CreateReplayQueueValidator();

    void Initialize();
    bool Test();

private:

    // Reserve new VAs
    LWNdevtoolsReservedVa NextVAs(MemoryPoolFlags flags, size_t memorySize);
    bool AllocateNextVAsAndReserveThem(MemoryPoolFlags flags, size_t memorySize, LWNdevtoolsReservedVa &reservedVAs);

    // Create memory pool mappings of reservesed VAs
    std::vector<LWNdevtoolsMemoryPoolReservedMapping> CreateReservedMappingsFromReservedVAs(LWNdevtoolsReservedVa &reservedVAs);

    // Update mappings for queue initialization to new ones
    bool UpdateReservedPoolMappingsForInitialization();

    // Callback when ReplayQueueInitialize requires setting reserved pool mappings
    static void LWNAPIENTRY ReservePool(void *param, LWNmemoryPoolBuilder *builder);

    // Create and destroy replay queue function for test.
    // Initialize the target queue, then destroy it if success and return success or not code
    bool CreateAndDestroyReplayQueue(int queueFlags, int controlMemorySize, int computeMemorySize, int commandMemorySize, bool usePoolReserver);

    // We need to have one live queue for some of the Getter things in
    // devtools bootstrap land.
    llgd_lwn::QueueHolder qh;

    // Bootstrap layer interface
    static const LWNdevtoolsBootstrapFunctions* devtools;

    // Memory pool mapping information used by ReplayQueueInitialize
    std::vector<LWNdevtoolsMemoryPoolReservedMapping> m_mappingForQueueMem;
    std::vector<LWNdevtoolsMemoryPoolReservedMapping> m_mappingForCbMem;
    LlgdInternalMemoryPoolReserver m_queuePoolReserver;
    LlgdInternalMemoryPoolReserver m_commandPoolReserver;

    // MemoryPool storage
    // Note that lwnQueue internal memory pools will require to change storage's
    // memory attribute on initialization. In order to work it fine, the storage
    // for lwnQueue internal memory pools should be on heap, not on stack.
    // This is why __attribute__((aligned(4096))) is not used here.
    static const size_t STORAGE_SIZE = 65536;
    static const size_t STORAGE_ALINT = LWN_MEMORY_POOL_STORAGE_ALIGNMENT;
    LlgdUniqueUint8PtrWithLwstomDeleter spStorage;

    //------------------------------------------------
    // Constants
    //------------------------------------------------

    // Constants come from g_lwndevice_info.cpp::__lwnDeviceInfoGetDefaults
    static const size_t CONTROL_MIN_SIZE = 4096;
    static const size_t COMPUTE_MIN_SIZE = 16384;
    static const size_t COMMAND_MIN_SIZE = 65536;

    // Queue internal pool flags (the values initialized below come from lwnqueue.cpp)
    static const MemoryPoolFlags queueMemFlags;
    static const MemoryPoolFlags cbPoolFlags;

    // Possible all queue flags
    static const std::vector<int> allQueueFlags;
};

const LWNdevtoolsBootstrapFunctions* CreateReplayQueueValidator::devtools = nullptr;
const MemoryPoolFlags CreateReplayQueueValidator::queueMemFlags = MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_UNCACHED;
const MemoryPoolFlags CreateReplayQueueValidator::cbPoolFlags   = MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED;
const std::vector<int> CreateReplayQueueValidator::allQueueFlags{ LWN_QUEUE_FLAGS_NONE, LWN_QUEUE_FLAGS_NO_FRAGMENT_INTERLOCK_BIT, LWN_QUEUE_FLAGS_NO_ZLWLL_BIT, LWN_QUEUE_FLAGS_NO_FRAGMENT_INTERLOCK_BIT | LWN_QUEUE_FLAGS_NO_ZLWLL_BIT };

//------------------------------------------------
// NextVAs
//------------------------------------------------
LWNdevtoolsReservedVa CreateReplayQueueValidator::NextVAs(MemoryPoolFlags flags, size_t memorySize)
{
    LWNdevtoolsReservedVa reserved;
    llgd_lwn::MemoryPoolHolder mph;
    MemoryPoolBuilder mpb1;
    mpb1.SetDevice(g_device).SetDefaults()
        .SetFlags(flags)
        .SetStorage(&(*spStorage), memorySize);
    if (!mph.Initialize(&mpb1)) { __builtin_trap(); }
    reserved = devtools->MemoryPoolProbeVas(mph);
    return reserved;
}

//------------------------------------------------
// AllocateNextVAsAndReserveThem
//------------------------------------------------
bool CreateReplayQueueValidator::AllocateNextVAsAndReserveThem(MemoryPoolFlags flags, size_t memorySize, LWNdevtoolsReservedVa &reservedVAs)
{
    reservedVAs = NextVAs(flags, memorySize);
    auto ret = devtools->ReservedVasReserve(reservedVAs);
    return ret;
}

//------------------------------------------------
// CreateReservedMappingsFromReservedVAs
//------------------------------------------------
std::vector<LWNdevtoolsMemoryPoolReservedMapping> CreateReplayQueueValidator::CreateReservedMappingsFromReservedVAs(LWNdevtoolsReservedVa &reservedVAs)
{
    using mapping_t = LWNdevtoolsMemoryPoolMapping;
    static const size_t MAX_MAPPING_COUNT = 3;
    static const std::vector<mapping_t> mappingTypes{ LWN_DEVTOOLS_MEMORY_POOL_MAPPING_PITCH, LWN_DEVTOOLS_MEMORY_POOL_MAPPING_BLOCK_LINEAR, LWN_DEVTOOLS_MEMORY_POOL_MAPPING_SPECIAL };

    // Colwert VAs to mappings
    auto reservedMappings = std::vector<LWNdevtoolsMemoryPoolReservedMapping>(MAX_MAPPING_COUNT);
    size_t realMappingNum = 0;
    for (size_t i = 0; i < MAX_MAPPING_COUNT; ++i, ++realMappingNum) {
        auto& mapping = reservedMappings[i];
        switch (mappingTypes[i])
        {
        case LWN_DEVTOOLS_MEMORY_POOL_MAPPING_PITCH:
            mapping.gpuva = reservedVAs.pitchGpuVa.start;
            mapping.size = reservedVAs.pitchGpuVa.size;
            break;
        case LWN_DEVTOOLS_MEMORY_POOL_MAPPING_BLOCK_LINEAR:
            mapping.gpuva = reservedVAs.blockLinearGpuVa.start;
            mapping.size = reservedVAs.blockLinearGpuVa.size;
            break;
        case LWN_DEVTOOLS_MEMORY_POOL_MAPPING_SPECIAL:
            mapping.gpuva = reservedVAs.specialGpuVa.start;
            mapping.size = reservedVAs.specialGpuVa.size;
            break;
        default:
            break;
        }
        if (mapping.gpuva == 0 && mapping.size == 0)
        {
            // No more mapping.
            break;
        }
        mapping.iova = reservedVAs.iova.start;
        mapping.comptagline = reservedVAs.comptags.start;
        mapping.mapping = mappingTypes[i];
    }
    reservedMappings.resize(realMappingNum);

    return reservedMappings;
}

//------------------------------------------------
// UpdateReservedPoolMappingsForInitialization
//------------------------------------------------
bool CreateReplayQueueValidator::UpdateReservedPoolMappingsForInitialization()
{
    LWNdevtoolsReservedVa vasQueueMem{ 0 }, vasCbMem{ 0 };

    // Allocate and reserve
    if (!AllocateNextVAsAndReserveThem(queueMemFlags, CONTROL_MIN_SIZE, vasQueueMem))
    {
        return false;
    }
    if (!AllocateNextVAsAndReserveThem(cbPoolFlags, COMMAND_MIN_SIZE, vasCbMem))
    {
        return false;
    }

    m_mappingForQueueMem = CreateReservedMappingsFromReservedVAs(vasQueueMem);
    m_mappingForCbMem    = CreateReservedMappingsFromReservedVAs(vasCbMem);

    return true;
}


//------------------------------------------------
// ReservePool
//------------------------------------------------
void LWNAPIENTRY CreateReplayQueueValidator::ReservePool(void *param, LWNmemoryPoolBuilder *builder)
{
    auto &mappings = *reinterpret_cast<std::vector<LWNdevtoolsMemoryPoolReservedMapping>*>(param);

    devtools->MemoryPoolBuilderSetReservedMappings(builder, mappings.size(), mappings.data());
}

//------------------------------------------------
// CreateAndDestroyReplayQueue
//------------------------------------------------
bool CreateReplayQueueValidator::CreateAndDestroyReplayQueue(int queueFlags, int controlMemorySize, int computeMemorySize, int commandMemorySize, bool usePoolReserver)
{
    llgd_lwn::QueueHolder queue;

    QueueBuilder builder;
    builder.SetDefaults().SetDevice(g_device).
        SetFlags(queueFlags).
        SetCommandMemorySize(commandMemorySize).
        SetComputeMemorySize(computeMemorySize).
        SetControlMemorySize(controlMemorySize);
    const auto success = devtools->ReplayQueueInitialize(static_cast<LWNqueue *>(queue),
        reinterpret_cast<LWNqueueBuilder *>(&builder),
        usePoolReserver ? &m_queuePoolReserver : nullptr,
        usePoolReserver ? &m_commandPoolReserver : nullptr,
        true);

    if (success) {
        queue->Finalize();
    }

    return success;
}

//------------------------------------------------
// Initialize
//------------------------------------------------
void CreateReplayQueueValidator::Initialize()
{
    qh.Initialize(g_device);
    devtools = lwnDevtoolsBootstrap();

    // Allocate memory storage which is aligned to STORAGE_ALINT
    spStorage = LlgdAlignedAllocPodType<uint8_t>(STORAGE_SIZE, STORAGE_ALINT);

    // Memory pool reserver: used in ReplayQueueInitialize when any special mappings are required.
    // Note: mappingForQueueMem and mappingForCbMem will be updated in UpdateReservedPoolMappingsForInitialization().
    m_queuePoolReserver = { &m_mappingForQueueMem, ReservePool };
    m_commandPoolReserver = { &m_mappingForCbMem, ReservePool };
}

//------------------------------------------------
// Dtor
//------------------------------------------------
CreateReplayQueueValidator::~CreateReplayQueueValidator()
{
}

//------------------------------------------------
// Test
//------------------------------------------------
bool CreateReplayQueueValidator::Test()
{
    // Success case: No reserved pools
    for (const auto queueFlag : allQueueFlags)
    {
        TEST(CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE, COMPUTE_MIN_SIZE, COMMAND_MIN_SIZE, /*usePoolReservers*/ false));
    }

    // Success case: With reserved pool mappings
    for (const auto queueFlag : allQueueFlags)
    {
        TEST(UpdateReservedPoolMappingsForInitialization());
        TEST(CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE, COMPUTE_MIN_SIZE, COMMAND_MIN_SIZE, /*usePoolReservers*/ true));
    }

    // Failure case: Not aligned memory sizes
    for (const auto queueFlag : allQueueFlags)
    {
        TEST(!CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE + 1, COMPUTE_MIN_SIZE, COMMAND_MIN_SIZE, /*usePoolReservers*/ false));
        TEST(!CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE, COMPUTE_MIN_SIZE + 1, COMMAND_MIN_SIZE, /*usePoolReservers*/ false));
        TEST(!CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE, COMPUTE_MIN_SIZE, COMMAND_MIN_SIZE + 1, /*usePoolReservers*/ false));
    }

    // Failure case: Memory sizes are less than minimum size
    for (const auto queueFlag : allQueueFlags)
    {
        TEST(!CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE - 1, COMPUTE_MIN_SIZE, COMMAND_MIN_SIZE, /*usePoolReservers*/ false));
        TEST(!CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE, COMPUTE_MIN_SIZE - 1, COMMAND_MIN_SIZE, /*usePoolReservers*/ false));
        TEST(!CreateAndDestroyReplayQueue(queueFlag, CONTROL_MIN_SIZE, COMPUTE_MIN_SIZE, COMMAND_MIN_SIZE - 1, /*usePoolReservers*/ false));
    }

    return true;
}

LLGD_DEFINE_TEST(ReplayQueue, UNIT,
LwError Execute()
{
    CreateReplayQueueValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
