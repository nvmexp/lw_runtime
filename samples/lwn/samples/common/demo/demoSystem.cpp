/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/
////===========================================================================
///  demoSystem.cpp
///
///     This is common apis for the DEMO library.
///
////===========================================================================

#include <demo.h>
#if defined __ANDROID__
#include <android/log.h>
#define printf(...) __android_log_print(ANDROID_LOG_DEBUG, "TAG", __VA_ARGS__)
#define vprintf(...) __android_log_vprint(ANDROID_LOG_DEBUG, "TAG", __VA_ARGS__)
#elif defined LW_HOS
#include <nn/nn_Log.h>
#include <nn/os.h>

#ifndef NN_SDK_BUILD_DEVELOP
#define NN_SDK_BUILD_DEVELOP 1
#endif

#include <nn/lmem/lmem_ExpHeap.h>

#define malloc(...) nn::lmem::AllocateFromExpHeap(s_heapHandle, __VA_ARGS__)
#define free(...) nn::lmem::FreeToExpHeap(s_heapHandle, __VA_ARGS__)

static char s_heapBuffer[128 * 1024 * 1024];
nn::lmem::HeapHandle s_heapHandle;
#endif
#include <stdarg.h>

static BOOL s_demoInitFlag = FALSE;
static BOOL s_demoRunningFlag = FALSE;

static DEMODefaultAllocateFunc s_pAllocFunc = NULL;
static DEMODefaultFreeFunc     s_pFreeFunc  = NULL;

void DEMOInit()
{
    DEMOPrintf("DEMO: Build date - %s %s\n", __DATE__, __TIME__);

#ifdef LW_HOS
    s_heapHandle = nn::lmem::CreateExpHeap(s_heapBuffer, sizeof(s_heapBuffer), 0);
#endif

    s_demoRunningFlag = TRUE;
    s_demoInitFlag = TRUE;
}

void DEMOStopRunning()
{
    s_demoRunningFlag = FALSE;
}

void DEMOShutdown()
{
        // The real purpose of this is to avoid the unused warning
    if (!s_demoInitFlag)
        return;
        
    DEMOStopRunning();

#ifdef LW_HOS
    nn::lmem::DestroyExpHeap(s_heapHandle);
#endif

    s_pAllocFunc = NULL;
    s_pFreeFunc  = NULL;

    s_demoInitFlag = FALSE;
    DEMOPrintf("\nEnd of demo\n");
}

BOOL DEMOIsRunning()
{
    return s_demoRunningFlag;
}

void DEMOSetDefaultAllocator(DEMODefaultAllocateFunc pfnAlloc, DEMODefaultFreeFunc pfnFree)
{
    DEMOAssert(!s_demoInitFlag);
    s_pAllocFunc = pfnAlloc;
    s_pFreeFunc  = pfnFree;
}

void DEMOGetDefaultAllocator(DEMODefaultAllocateFunc *ppfnAlloc, DEMODefaultFreeFunc *ppfnFree)
{
    if (ppfnAlloc) *ppfnAlloc = s_pAllocFunc;
    if (ppfnFree)  *ppfnFree  = s_pFreeFunc;
}

void* DEMOAlloc(u32 size)
{
    DEMOAssert(s_demoInitFlag);
    if (!s_pAllocFunc) {
        return malloc(size);
    }
    return s_pAllocFunc(size, DEMO_BUFFER_ALIGN);
}

void* DEMOAllocEx(u32 size, u32 align)
{
    DEMOAssert(s_demoInitFlag);
    if (!s_pAllocFunc) {
         return malloc(size);
    }
    return s_pAllocFunc(size, align);
}

void DEMOFree(void* ptr)
{
    DEMOAssert(s_demoInitFlag);
    if (!s_pFreeFunc) {
        free(ptr);
    } else {
        s_pFreeFunc(ptr);
    }
}

// ----------------------------------------------------------------------------
//  Random
// ----------------------------------------------------------------------------

static u32 holdrand = 0;

void DEMOSRand(u32 seed)
{
    holdrand = seed;
}

f32 DEMOFRand()
{
    return (f32)DEMORand() / DEMO_RAND_MAX;
}

u32 DEMORand()
{
    return (((holdrand = holdrand * 214013L + 2531011L) >> 16) & 0xffff);
}

// ----------------------------------------------------------------------------
//  Print
// ----------------------------------------------------------------------------
#ifndef LW_HOS

void DEMOPrintf(const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    va_end(args);
}

#endif
