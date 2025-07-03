/*
 * Copyright (c) 2015-2019, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#if defined(LW_HOS)
#include <new>
#include <nn/fs.h>
#include <nn/init.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/nn_Assert.h>
#include <nn/nn_Log.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <lw/lw_MemoryManagement.h>
#define printf NN_LOG
#define assert NN_ASSERT
#define abort NN_ABORT
#endif

#include <lwca.h>
#include <lwdaLWN.h>
#include <lwdaNNAllocator.h>

#include <lwn/lwn_Cpp.h>
#include <lwn/lwn_CppFuncPtr.h>
#include <lwn/lwn_CppFuncPtrImpl.h>
#include <lwn/lwn_CppMethods.h>

#include <string.h>

#define ARRAY_SIZE 8192
#define DEBUG_LOG 0
const size_t heapSize = 100 * 1024 * 1024;
const int noOfCommands = 10;
extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);

using namespace lwn;

// 64 MB memory pool.
static const int POOL_SIZE = 64 << 20;
static char s_poolStorage[POOL_SIZE] __attribute__((aligned(LWN_MEMORY_POOL_STORAGE_ALIGNMENT)));
static char s_controlMemory[4096];
static const int COMMAND_MEMORY = 4096;
static Device device;
static Queue queue;
static CommandBuffer cmd;
static uint64_t commandMemoryBegin;
template <class T>
const T align(const T val, const T align)
{
    return (val + align -1) & ~(align - 1);
}

namespace {

    const int FsHeapSize = 512 * 1024;
    const int TlsHeapSize = 1 * 1024 * 1024;

    uint8_t              g_FsHeapBuffer[FsHeapSize];
    nn::lmem::HeapHandle g_FsHeap;
    char                 g_TlsHeapBuffer[TlsHeapSize];
    nn::util::TypedStorage<nn::mem::StandardAllocator, sizeof(nn::mem::StandardAllocator),
                           NN_ALIGNOF(nn::mem::StandardAllocator)> g_TlsAllocator;

    void FsInitHeap()
    {
        g_FsHeap = nn::lmem::CreateExpHeap(g_FsHeapBuffer, FsHeapSize, nn::lmem::CreationOption_DebugFill);
    }

    void* FsAllocate(size_t size)
    {
        return nn::lmem::AllocateFromExpHeap(g_FsHeap, size);
    }

    void FsDeallocate(void* p, size_t size)
    {
        return nn::lmem::FreeToExpHeap(g_FsHeap, p);
    }

    void* TlsAlloc(size_t size, size_t alignment)
    {
        return nn::util::Get(g_TlsAllocator).Allocate(size, alignment);
    }

    void TlsDealloc(void* p, size_t size)
    {
        nn::util::Get(g_TlsAllocator).Free(p);
        NN_UNUSED(size);
    }

}

void debug(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
           DebugCallbackSeverity::Enum severity, const char *message, void* userParam)
{
    printf("LWN DEBUG ERROR: %s\n", message);
}


//-----------------------------------------------------------------------------
// nninitStartup() is ilwoked before calling nnMain().
//
extern "C" void nninitStartup()
{
    const size_t MallocMemorySize = 16 * 1024 * 1024;
    uintptr_t address;
    nn::Result result = nn::os::SetMemoryHeapSize(heapSize);
    assert(result.IsSuccess());
    result = nn::os::AllocateMemoryBlock(&address, MallocMemorySize);
    assert(result.IsSuccess());
    nn::init::InitializeAllocator(reinterpret_cast<void*>(address), MallocMemorySize);

    // Set file system allocator and deallocator
    FsInitHeap();
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
    nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);
}

#define FAIL_IF(x) \
    if ((x)) { \
        printf("Test failed '%s' on line %d\n", #x, __LINE__); \
        printf("\n\n \t\t &&&& lwn_lwda_interop test FAILED\n"); \
        return; \
    }

static int verify(int *hptr, int size, int baseValue, bool isIncrement) 
{
    if (isIncrement) {
        for (int i=0; i<size ; i++) {
            if (hptr[i] == baseValue + i){
                continue;
            }
            else {
                return true;
            }
        }
    }
    else {
        for (int i=0; i<size; i++) {
            if(hptr[i] == baseValue)
                continue;
            else
                return true;
        }
    }
    return false;
}

static unsigned int DeviceGet(DeviceInfo pname)
{
    int v;
    device.GetInteger(pname, &v);
    return v;
}

namespace {

    const int firmwareMemorySize = 8 * 1024 * 1024;
    char g_FirmwareMemory[firmwareMemorySize] __attribute__((aligned(4096)));
    const int lwdaHeapSize = 512 * 1024 * 1024;
    char g_lwdaHeapBuffer[lwdaHeapSize];
    nn::mem::StandardAllocator  g_LwdaAllocator(g_lwdaHeapBuffer, sizeof(g_lwdaHeapBuffer));

    void *lwdaAllocateCallback(size_t size, size_t align, void *userPtr)
    {
        void  *address = NULL;

        address = g_LwdaAllocator.Allocate(size, align);
        if (!address)
        {
            NN_LOG("Failed to allocate memory.\n");
        }

        return (void *)(address);
    }

    void lwdaNNFreeCallback(void *address, void *userPtr)
    {
        g_LwdaAllocator.Free(address);
    }

    void* lwdaNNReallocateCallback(void *addr, size_t newSize, void *userPtr)
    {
        void  *address = NULL;

        address = g_LwdaAllocator.Reallocate(addr, newSize);

        if (!address)
        {
            NN_LOG("Failed to allocate memory\n");
        }

        return (void *)(address);
    }
}

extern "C" void nnMain()
{
    LWresult status;
    LWdevice dev;
    LWcontext ctx;
    LWdeviceptr dptr_a;
    LWevent pEvent;
    int hptr_a[ARRAY_SIZE];
    int bufferSize;
    int i;
    size_t size = 0;
    unsigned long poolOffset = 0;

    status = lwNNSetAllocator(lwdaAllocateCallback, lwdaNNFreeCallback, lwdaNNReallocateCallback, NULL);
    NN_LOG("lwNNSetAllocator() status = %d\n", status);
    if (LWDA_SUCCESS != status) {
        NN_LOG("Failed in lwNNSetAllocator\n");
        return;
    }

    lw::SetGraphicsAllocator(lwdaAllocateCallback, lwdaNNFreeCallback, lwdaNNReallocateCallback ,NULL);
    lw::SetGraphicsDevtoolsAllocator(lwdaAllocateCallback, lwdaNNFreeCallback, lwdaNNReallocateCallback ,NULL);
    lw::InitializeGraphics(g_FirmwareMemory, sizeof(g_FirmwareMemory));

    bufferSize = ARRAY_SIZE * sizeof(int);
    bufferSize = sizeof(unsigned long) *ARRAY_SIZE;
    // reset the array value to zero
    memset(hptr_a, 0, sizeof(hptr_a));

    memset(s_poolStorage, 0, sizeof(char) * POOL_SIZE);
    //--------------------------------------------------------------------------------
    // Initialize LWCA.
    //--------------------------------------------------------------------------------

    status = lwInit(0);
    printf("lwInit status = %d\n", status);
    FAIL_IF(LWDA_SUCCESS != status);

    status = lwDeviceGet(&dev, 0);
    printf("lwDeviceGet status = %d\n", status);
    FAIL_IF(LWDA_SUCCESS != status);

    status = lwCtxCreate(&ctx, 0, dev);
    printf("lwCtxCreate status = %d\n", status);
    FAIL_IF(LWDA_SUCCESS != status);


    //--------------------------------------------------------------------------------
    // Initialize LWN driver interface.
    //--------------------------------------------------------------------------------

    DeviceGetProcAddressFunc getProcAddress = (DeviceGetProcAddressFunc) ((lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
    lwnLoadCPPProcs(NULL, getProcAddress);

    DeviceBuilder deviceBuilder;
    deviceBuilder.SetDefaults();
    deviceBuilder.SetFlags(LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT);
    if (!device.Initialize(&deviceBuilder)) {
        abort("Failed to init device");
    }

    lwnLoadCPPProcs(&device, getProcAddress);
    device.InstallDebugCallback(debug, NULL, LWN_TRUE);

    int majorVersion = DeviceGet(DeviceInfo::API_MAJOR_VERSION);
    int minorVersion = DeviceGet(DeviceInfo::API_MINOR_VERSION);
    if (majorVersion != LWN_API_MAJOR_VERSION || minorVersion < LWN_API_MINOR_VERSION) {
        abort("API version mismatch (application compiled with %d.%d, driver reports %d.%d).\n",
                LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION, majorVersion, minorVersion);
    }
    printf("API version is compatible (application compiled with %d.%d, driver reports %d.%d).\n",
            LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION, majorVersion, minorVersion);


    //--------------------------------------------------------------------------------
    // Initialize a memory pool and a buffer object.
    //--------------------------------------------------------------------------------

    // Create a memory pool.
    MemoryPoolBuilder pb;
    pb.SetDevice(&device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(s_poolStorage, sizeof(s_poolStorage));
    MemoryPool pool;
    pool.Initialize(&pb);

    // Init GPU queue
    QueueBuilder qb;
    qb.SetDevice(&device)
      .SetDefaults();
    queue.Initialize(&qb);
    // Create a command buffer for GPU commands.
    cmd.Initialize(&device);
    cmd.AddControlMemory(s_controlMemory, sizeof(s_controlMemory));
    poolOffset = align(poolOffset, (unsigned long)DeviceGet(DeviceInfo::COMMAND_BUFFER_COMMAND_ALIGNMENT));
    cmd.AddCommandMemory(&pool, poolOffset, COMMAND_MEMORY);
    commandMemoryBegin = poolOffset;
    poolOffset += COMMAND_MEMORY;

    // Begin recording the program init commands that will follow.
    // Initialize 3 buffers
    Buffer buffer;
    BufferBuilder bb;
    bb.SetDevice(&device)
        .SetDefaults()
        .SetStorage(&pool, poolOffset,  bufferSize);
    poolOffset += bufferSize;
    buffer.Initialize(&bb);
    int *lwnBufPtr = (int *)buffer.Map();


    //--------------------------------------------------------------------------------
    // Do the tests.
    //--------------------------------------------------------------------------------

    // Fill the LWN buffer with 1....
    for(i = 0; i < ARRAY_SIZE; i++) {
        lwnBufPtr[i] = i;
    }
    Buffer buffer1;
    BufferBuilder bb1;
    bb1.SetDevice(&device)
        .SetDefaults()
        .SetStorage(&pool, poolOffset, bufferSize);
    buffer1.Initialize(&bb1);

    poolOffset += bufferSize;
    Buffer buffer2;
    BufferBuilder bb2;
    bb2.SetDevice(&device)
        .SetDefaults()
        .SetStorage(&pool, poolOffset, bufferSize);
    buffer2.Initialize(&bb2);

    int *tmp = (int*)buffer2.Map();
    tmp[0]=0xdeadbeef;

    status = lwLWNbufferGetPointer (&dptr_a, (LWNbuffer *)&buffer1, 0, &size);
    printf("lwLWNbufferGetPointer (dptr_a) %lld status = %d\n", dptr_a, status);
    FAIL_IF(LWDA_SUCCESS != status);

    // Initialize Sync and create a command buffer of noOfCommands commands
    // In each command a copy is done from the buffer, buffer1 and
    // in noOfCommands command, copy buffer2 to buffer1
    Sync sync;
    sync.Initialize(&device);
    status = lwEventCreateFromLWNSync(&pEvent,(LWNsync*)&sync ,0);
    FAIL_IF(LWDA_SUCCESS != status);
    CommandHandle initCommands;
    cmd.BeginRecording();
    for (int i=0; i< noOfCommands; i++) {
        if (i == (noOfCommands -1)) {
            cmd.CopyBufferToBuffer(buffer.GetAddress(), buffer1.GetAddress(), bufferSize, 0);
        }
        else {
            cmd.CopyBufferToBuffer(buffer2.GetAddress(), buffer1.GetAddress(), bufferSize, 0);
        }
    }
    initCommands = cmd.EndRecording();
    queue.SubmitCommands(1, &initCommands);
    queue.FenceSync(&sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
    queue.Flush();
    status = lwStreamWaitEvent(NULL, pEvent, 0);
    FAIL_IF(LWDA_SUCCESS != status);

    // Copy from LWN buffer to LWCA host ptr.
    status = lwMemcpyDtoH(hptr_a, dptr_a, ARRAY_SIZE*sizeof(int));
    printf("\nlwMemcpyHtoD(dptr_a, hptr_a) status = %d\n", status);
    FAIL_IF(LWDA_SUCCESS != status);

    FAIL_IF(verify(hptr_a, ARRAY_SIZE, 0, true));
    // Should get 1...2048
#if DEBUG_LOG
    printf("hptr_a= ");
    for (i = 0; i < ARRAY_SIZE; i++)
        printf("%d, ", hptr_a[i]);
#endif
    // Reset to 10 through LWCA.
    status = lwMemsetD32Async(dptr_a, 10, ARRAY_SIZE, 0);
    printf("\nlwMemsetD32Async(dptr_a = %x) status = %d\n", dptr_a, status);
    FAIL_IF(LWDA_SUCCESS != status);

    status = lwEventRecord(pEvent, 0);
    FAIL_IF(LWDA_SUCCESS != status);

    CommandHandle initCommand;
    cmd.BeginRecording();
    cmd.CopyBufferToBuffer(buffer1.GetAddress(), buffer.GetAddress(), bufferSize, 0);
    initCommand = cmd.EndRecording();
    queue.WaitSync(&sync);
    queue.SubmitCommands(1, &initCommand);
    queue.FenceSync(&sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
    queue.Finish();

#if DEBUG_LOG
    // LWN buffer contents should also be all 10's.
    printf("The values should be all set to 10, after buffer map\n");
    printf("hptr_a= ");
    for (i = 0; i < ARRAY_SIZE; i++)
        printf("%d, ", lwnBufPtr[i]);
#endif
    FAIL_IF(verify(lwnBufPtr, ARRAY_SIZE , 10, false));


    //--------------------------------------------------------------------------------
    // Cleanup.
    //--------------------------------------------------------------------------------
    buffer.Finalize();
    buffer1.Finalize();
    buffer2.Finalize();
    pool.Finalize();
    queue.Finalize();
    device.Finalize();

    printf("\n\n\t\t&&&& lwn_lwda_interop test PASSED\n");
}
