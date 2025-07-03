/*
 * Copyright (c) 2018-2019, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/*
 * blobcache
 *
 * Builds upon gltri to show and test basic functionality of EGL_ANDROID_blob_cache
 */

#include "../gltri/gltri.h"

#include <new>

#include <EGL/eglext.h>

#include <nn/fs.h>
#include <nn/init.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <nn/nn_Assert.h>
#include <nn/nn_Log.h>
#include <nn/os.h>
#include <nn/os/os_SdkMemoryAllocatorForThreadLocal.h>
#include <nn/vi.h>
#include <lw/lw_MemoryManagement.h>

#include <unordered_map>
#include <memory>
#include <string>
#include <fstream>

// NOTE: exact key size is not specified, but driver should only use a constant key size
// which is 32 bytes lwrrently and not expected to change.
EGLsizeiANDROID gKeySize = -1;
std::unordered_map<std::string, std::string> gBlobCache;
unsigned gBlobHits;
unsigned gBlobMisses;
unsigned gBlobSets;

static void printBlobLine(const void* key, ssize_t keySize, ssize_t valSize)
{
    const char* keyChars = static_cast<const char*>(key);
    NN_LOG("keySize=%zd valSize=%zd key=", keySize, valSize);
    for (ssize_t i = 0; i < keySize; i++) {
        NN_LOG("%02x", keyChars[i]);
        if (i % 8 == 7 && i + 1 < keySize) {
            NN_LOG("-");
        }
    }
    NN_LOG("\n");
}

static void setBlobImpl(const void* key, EGLsizeiANDROID keySize,
    const void* value, EGLsizeiANDROID valueSize)
{
    NN_LOG("Calling setBlob: ");
    printBlobLine(key, keySize, valueSize);

    NN_ASSERT(keySize > 0 && valueSize > 0);
    NN_ASSERT(gKeySize == -1 || gKeySize == keySize);
    gKeySize = keySize;

    std::string keyBlob(static_cast<const char*>(key), keySize);
    std::string valBlob(static_cast<const char*>(value), valueSize);
    gBlobCache.emplace(std::move(keyBlob), std::move(valBlob));
    gBlobSets++;
}

static EGLsizeiANDROID getBlobImpl(const void* key, EGLsizeiANDROID keySize,
    void* value, EGLsizeiANDROID valueSize)
{
    NN_LOG("Calling getBlob: ");
    printBlobLine(key, keySize, valueSize);

    NN_ASSERT(keySize > 0 && valueSize >= 0);
    NN_ASSERT(gKeySize == -1 || gKeySize == keySize);
    gKeySize = keySize;

    std::string blobKey(static_cast<const char*>(key), keySize);
    auto found = gBlobCache.find(blobKey);
    if (found != gBlobCache.end()) {
        const std::string& blobValue = found->second;
        ssize_t blobValueSize = blobValue.size();
        if (blobValueSize <= valueSize) {
            memcpy(value, blobValue.data(), blobValueSize);
            gBlobHits++;
        } // valueSize < blobValueSize means query for real blobValueSize

        NN_LOG("Found blobValueSize=%zd\n", blobValueSize);
        return blobValueSize;
    }
    gBlobMisses++;
    return 0;
}

#define HOST_MOUNT_NAME "host"
#define HOST_MOUNT_PATH "C:/HosHostFs"
#define BLOB_FILE_PATH "host:/_lwblobs.bin"

struct HostMount
{
    bool hosRootMounted;
    HostMount()
    {
        nn::Result result = nn::fs::MountHostRoot();
        hosRootMounted = false;
        if (nn::fs::ResultMountNameAlreadyExists::Includes(result)) {
            hosRootMounted = true;
        } else if (!result.IsSuccess()) {
            NN_ASSERT(0);
        } else {
            result = nn::fs::MountHost(HOST_MOUNT_NAME, HOST_MOUNT_PATH);
            NN_ASSERT(result.IsSuccess());
        }
    }
    ~HostMount()
    {
        if (!hosRootMounted) {
            nn::fs::Unmount(HOST_MOUNT_NAME);
        }
        nn::fs::UnmountHostRoot();
    }
};

std::ostream& operator<<(std::ostream& out, const std::string& v)
{
    size_t size = v.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    NN_ASSERT(out);
    out.write(v.data(), size);
    NN_ASSERT(out);
    return out;
}

std::istream& operator>>(std::istream& in, std::string& v)
{
    size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    NN_ASSERT(in);
    char* buf = new char[size];
    NN_ASSERT(buf);
    in.read(buf, size);
    NN_ASSERT(in);
    v.assign(buf, size);
    delete[] buf;
    return in;
}

static void LoadBlobs()
{
    HostMount hostMount;
    gBlobCache.clear();
    std::ifstream blobFile(BLOB_FILE_PATH, std::ifstream::in | std::ifstream::binary);
    if (!blobFile) {
        return;
    }
    size_t blobCount = 0;
    blobFile.read(reinterpret_cast<char*>(&blobCount), sizeof(blobCount));
    NN_ASSERT(blobFile);
    for (size_t i = 0; i < blobCount; i++) {
        std::string keyBlob, valBlob;
        blobFile >> keyBlob >> valBlob;
        NN_LOG("Loaded blob: ");
        printBlobLine(keyBlob.data(), keyBlob.size(), valBlob.size());
        NN_ASSERT(gKeySize == -1 || gKeySize == static_cast<EGLsizeiANDROID>(keyBlob.size()));
        gKeySize = keyBlob.size();
        gBlobCache.emplace(std::move(keyBlob), std::move(valBlob));
    }
    NN_LOG("Loaded %zu blobs from '%s'\n", gBlobCache.size(), BLOB_FILE_PATH);
}

static void SaveBlobs()
{
    HostMount hostMount;
    std::ofstream blobFile(BLOB_FILE_PATH, std::ofstream::out | std::ofstream::binary);
    NN_ASSERT(blobFile);
    size_t blobCount = gBlobCache.size();
    blobFile.write(reinterpret_cast<const char*>(&blobCount), sizeof(blobCount));
    NN_ASSERT(blobFile);
    for (auto it = gBlobCache.begin(); it != gBlobCache.end(); it++) {
        blobFile << it->first << it->second;
    }
    NN_LOG("Saved %zu blobs to '%s'\n", gBlobCache.size(), BLOB_FILE_PATH);
}

const int FsHeapSize = 512 * 1024;
const int GraphicsHeapSize = 256 * 1024 * 1024;
const int TlsHeapSize = 1 * 1024 * 1024;
const int GraphicsFirmwareMemorySize = 8 * 1024 * 1024;

char                        g_FsHeapBuffer[FsHeapSize];
nn::lmem::HeapHandle        g_FsHeap;
char                        g_GraphicsHeapBuffer[GraphicsHeapSize];
nn::mem::StandardAllocator  g_GraphicsAllocator(g_GraphicsHeapBuffer, sizeof(g_GraphicsHeapBuffer));
char                        g_GraphicsFirmwareMemory[GraphicsFirmwareMemorySize] __attribute__((aligned(4096)));
char                        g_TlsHeapBuffer[TlsHeapSize];
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
    NN_UNUSED(size);
    return nn::lmem::FreeToExpHeap(g_FsHeap, p);
}

void* GraphicsAllocate(size_t size, size_t alignment, void *userPtr)
{
    return g_GraphicsAllocator.Allocate(size, alignment);
}

void GraphicsFree(void *addr, void *userPtr)
{
    g_GraphicsAllocator.Free(addr);
}

void *GraphicsReallocate(void* addr, size_t newSize, void *userPtr)
{
    return g_GraphicsAllocator.Reallocate(addr, newSize);
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

extern "C" void nninitStartup()
{
    const size_t MallocMemorySize = 64 * 1024 * 1024;
    nn::Result result = nn::os::SetMemoryHeapSize(64 * 1024 * 1024);
    NN_ASSERT(result.IsSuccess());
    uintptr_t address;
    result = nn::os::AllocateMemoryBlock(&address, MallocMemorySize);
    NN_ASSERT(result.IsSuccess());
    nn::init::InitializeAllocator(reinterpret_cast<void*>(address), MallocMemorySize);

    // Set file system allocator and deallocator
    FsInitHeap();
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
    nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);
}

extern "C" void nnMain()
{
    int argc = nn::os::GetHostArgc();
    char** argv = nn::os::GetHostArgv();
    bool loadDisk = true;
    bool saveDisk = true;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-noload")) {
            loadDisk = false;
        } else if (!strcmp(argv[i], "-nosave")) {
            saveDisk = false;
        } else {
            NN_LOG("blobcache [-noload] [-nosave]\n");
            return;
        }
    }

    lw::SetGraphicsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
    lw::InitializeGraphics(g_GraphicsFirmwareMemory, sizeof(g_GraphicsFirmwareMemory));

    WindowMgr windowMgr;
    EglMgr eglMgr(windowMgr.GetNativeWindowHandle());

    // BlobCache setup
    PFNEGLSETBLOBCACHEFUNCSANDROIDPROC eglSetBlobCacheFuncsANDROID =
        reinterpret_cast<PFNEGLSETBLOBCACHEFUNCSANDROIDPROC>(
            eglGetProcAddress("eglSetBlobCacheFuncsANDROID"));
    if(eglSetBlobCacheFuncsANDROID == NULL) {
        NN_ASSERT(!"eglSetBlobCacheFuncsANDROID does not exist");
    } else {
        eglSetBlobCacheFuncsANDROID(eglMgr.mDisplay, setBlobImpl, getBlobImpl);
        EGLint err = eglGetError();
        if (err != EGL_SUCCESS) {
            NN_LOG("eglSetBlobCacheFuncsANDROID resulted in an error: %#x", err);
            NN_ASSERT(0);
        }
    }

    if (loadDisk) {
        LoadBlobs();
    }
    const unsigned blobsLoaded = gBlobCache.size();

    NN_LOG("GlMgr() may take a while to finish...\n");
    int64_t ns0 = nn::os::GetSystemTick().ToTimeSpan().GetNanoSeconds();
    gBlobHits = gBlobMisses = gBlobSets = 0;

    // unroll nested dependent loop to cause long compile time
    static const char *VERTEX_SOURCE =
        "#version 440 core\n"
        "#pragma optionLW(unroll all)\n"
        "precision highp float;\n"
        "attribute vec2 a_position;\n"
        "uniform float u_time;\n"
        "layout (std430, binding=1) buffer ssbo { float ssboBuf[100]; };\n"
        "void main() {\n"
        "    mat2 xform = mat2(cos(u_time), sin(u_time),\n"
        "                      -sin(u_time), cos(u_time));\n"
        "    gl_Position = vec4(xform * a_position, 0.0, 1.0);\n"
        "    if (u_time > -9000)\n"
        "        return;\n"
        "    for (int i = 0; i < 32; i++) {\n"
        "        for (int j = 0; j < 32 - i; j++) {\n"
        "            if (ssboBuf[j] > ssboBuf[j+1]) {\n"
        "                float x=ssboBuf[j]; ssboBuf[j]=ssboBuf[j+1]; ssboBuf[j+1]=x;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n";

    GlMgr glMgr(VERTEX_SOURCE, nullptr);
    glMgr.DrawFrame(0);

    int64_t ns1 = nn::os::GetSystemTick().ToTimeSpan().GetNanoSeconds();
    NN_LOG("GlMgr() finished time=%.3fs loaded=%u sets=%u hits=%u misses=%u\n",
        (ns1 - ns0) / 1e9, blobsLoaded, gBlobSets, gBlobHits, gBlobMisses);

    // NOTE: key from GL driver includes build id and device id,
    // so cache file should be ilwalidated whenever these change,
    // or unused entries will waste space.
    // To ilwalidate cache file for this sample, simply delete the file
    // BLOB_FILE_PATH in HOST_MOUNT_PATH, or run "blobcache -noload" to
    // regenerate the cache file without asserting on cache miss.
    NN_ASSERT(gBlobMisses == 0 || blobsLoaded == 0 || !"need to ilwalidate blobcache file");

    NN_ASSERT(gBlobSets > 0 || (gBlobHits > 0 && gBlobMisses == 0));
    NN_ASSERT(gBlobSets == gBlobMisses);
    NN_ASSERT((ns1 - ns0) < 1e9 || gBlobMisses > 0);

    if (saveDisk && gBlobSets > 0) {
        SaveBlobs();
    }

    for (int i = 0; i < 500; ++i) {
        glMgr.DrawFrame(i * 0.05);
        eglMgr.SwapBuffers();
    }

    NN_LOG("blobcache program finished\n");
}
