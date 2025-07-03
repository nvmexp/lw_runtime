/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#if defined(LW_HOS)
#include <nn/nn_Log.h>
#include <nn/os.h>
#include <time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "lwos.h"
#include "assert.h"
#include "ogtest.h"
#include "lwn/lwn.h"
#include <sys/time.h>

#include <lwnUtil/lwnUtil_AlignedStorage.h>

// Needed to satisfy linker
__attribute__((weak)) void *__dso_handle;

extern "C" {

#if defined(LW_HOS)

// This file contains reimplementations of many standard POSIX entry points on
// top of the lwos library.  These implementations are by no means complete or
// POSIX-compliant.

static void printFileHandle(LwOsFileHandle fh, const char *format, va_list &ap);

// (v)printf, fprintf send stdout and stderr to LwOsDebugPrintf or NN_LOG.
int int_vprintf(const char *format, va_list ap)
{
    printFileHandle(NULL, format, ap);
    return 0;
}

int int_printf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int_vprintf(format, ap);
    va_end(ap);

    return 0;
}

int int_fprintf(FILE *stream, const char *format, ...)
{
    LwOsFileHandle lwos_file;
    if (stream != stdout && stream != stderr) {
        lwos_file = (LwOsFileHandle) stream;
    } else {
        lwos_file = NULL;
    }

    va_list ap;
    va_start(ap, format);
    printFileHandle(lwos_file, format, ap);
    va_end(ap);
    return 0;
}

// HOS doesn't implement these entry points, but it does provide an fwrite
// entry point that conflicts if we try to implement these directly. So for
// now, we implement them with int_ prefix, and use #define to rename them
// elsewhere.
//
// For the purposes of these functions, FILE* is actually a LwOsFileHandle.
FILE * int_fopen(const char *path, const char *mode);
int int_fclose(FILE *file);
size_t int_fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t int_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
int int_feof(FILE *file);
int int_fflush(FILE *stream);
char *int_fgets(char *s, int size, FILE *stream);
int int_fgetc(FILE *stream);

FILE * int_fopen(const char *path, const char *mode)
{
    LwError err;
    LwU32 flags = 0;
    bool append = false;
    bool truncate = false;
    if (!mode) return NULL;
    while (*mode) {
        switch(*mode) {
            case 'r':
                flags |= LWOS_OPEN_READ;
                break;
            case 'a':
                append = true;
                flags |= LWOS_OPEN_WRITE | LWOS_OPEN_CREATE;
                break;
            case 'w':
                truncate = true;
                flags |= LWOS_OPEN_WRITE | LWOS_OPEN_CREATE;
                break;
            case 'b':
            case 't':
                // Ignored
                break;
            case '+':
                // Fall-through: We don't support the '+' modes
            default:
                assert(false);
        }
        mode++;
    }
    if (!(flags & (LWOS_OPEN_READ | LWOS_OPEN_WRITE))) {
        // mode argument needs to do something with the file
        return NULL;
    }
    LwOsFileHandle lwos_file;
    err = LwOsFopen(path, flags, &lwos_file);
    if (err == LwSuccess) {
        if (truncate) {
            LwOsFtruncate(lwos_file, 0);
        }
        if (append) {
            LwOsFseek(lwos_file, 0, LwOsSeek_End);
        }
        return (FILE *)lwos_file;
    } else {
        return NULL;
    }
}

int int_fseek(FILE *file, long int offset, int origin)
{
    LwOsFileHandle lwos_file = (LwOsFileHandle)file;
    LwOsSeekEnum lwosOrigin;
    switch (origin) {
    case SEEK_END:
        lwosOrigin = LwOsSeek_End;
        break;
    case SEEK_LWR:
        lwosOrigin = LwOsSeek_Lwr;
        break;
    case SEEK_SET:
        lwosOrigin = LwOsSeek_Set;
        break;
    default:
        assert(!"Invalid fseek origin parameter!");
        // Return 1 since this wasn't a success.
        return 1;
        break;
    }
    return LwOsFseek(lwos_file, offset, lwosOrigin);
}

long int int_ftell(FILE *file)
{
    LwU64 position;
    LwOsFileHandle lwos_file = (LwOsFileHandle)file;
    LwOsFtell(lwos_file, &position);

    return position;
}

int int_fclose(FILE *file)
{
    LwOsFileHandle lwos_file = (LwOsFileHandle)file;
    LwOsFclose(lwos_file);
    return 0;
}

size_t int_fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    size_t bytes_read;
    LwOsFileHandle lwos_file = (LwOsFileHandle) stream;

    LwOsFread(lwos_file, ptr, size * nmemb, &bytes_read);
    return bytes_read;
}

size_t int_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    LwError err;
    LwOsFileHandle lwos_file = (LwOsFileHandle) stream;

    err = LwOsFwrite(lwos_file, ptr, size * nmemb);
    if (err == LwSuccess) {
        return size;
    } else {
        return 0;
    }
}

int int_feof(FILE *file)
{
    // Not a legit feof, but hopefully close enough.
    // (doesn't support clearerr, etc)
    LwOsFileHandle lwos_file = (LwOsFileHandle)file;
    LwOsStatType stats;
    LwU64 position;
    LwOsFstat(lwos_file, &stats);
    LwOsFtell(lwos_file, &position);
    return (position == stats.size);
}

int int_fflush(FILE *stream)
{
    assert(stream); // we don't handle the null case
    if ((stream == stdout) || (stream == stderr)) return 0;
    LwOsFileHandle lwos_file = (LwOsFileHandle)stream;
    return (LwOsFflush(lwos_file) == LwSuccess) ? 0 : EOF;
}

char *int_fgets(char *s, int size, FILE *stream)
{
    size_t bytes_read = 0;
    size_t i;
    LwOsFileHandle lwos_file = (LwOsFileHandle) stream;

    // Implemented by greedily reading in the size of the destination buffer,
    // and then resetting the file position after the fact.
    LwOsFread(lwos_file, s, size-1, &bytes_read);
    if (bytes_read == 0) {
        return NULL;
    }
    for (i = 0; i < bytes_read; i++) {
        if (s[i] == '\n') {
            s[i+1] = 0;
            LwOsFseek(lwos_file, (i + 1) - (LwS64)bytes_read, LwOsSeek_Lwr);
            return s;
        }
    }
    s[bytes_read] = 0;
    return s;
}

int int_fgetc(FILE *stream)
{
    LwOsFileHandle lwos_file = (LwOsFileHandle) stream;
    size_t bytes_read = 0;
    char c;
    LwOsFread(lwos_file, &c, 1, &bytes_read);
    if (bytes_read == 1)
    {
        return c;
    }
    return EOF;
}

#endif


void lwogHandleWindowEvents()
{
    // Nothing implemented here.
}

// Data structure describing spawned lwntest threads.
struct LWOGthread {
#if defined(LW_HOS)
    nn::os::ThreadType osThread;
    char *stack;
#else
    LwOsThreadHandle osThread;
#endif
};

#if defined(LW_HOS)
LWOGthread *lwogThreadCreateOnCore(void (*threadFunc)(void*), void *args, size_t stackSize, int idealCore)
{
    LWOGthread *thread = new LWOGthread;

    // On HOS, dynamically allocate stack space for the thread.  If no size is
    // provided, use 128KB.  The size and address of the stack must be
    // aligned.
    if (stackSize == 0) {
        stackSize = 0x20000;
    }
    stackSize = lwnUtil::AlignSize(stackSize, nn::os::ThreadStackAlignment);
    thread->stack = (char *) lwnUtil::AlignedStorageAlloc(stackSize, nn::os::ThreadStackAlignment);

    nn::Result res = nn::os::CreateThread(&thread->osThread, threadFunc, args, 
                                          thread->stack, stackSize,
                                          nn::os::DefaultThreadPriority,
                                          idealCore);
    if (!res.IsSuccess()) {
        assert(!"Failed to create worker thread.\n");
        delete thread;
        return NULL;
    }

    nn::os::StartThread(&thread->osThread);

    return thread;
}
#endif

LWOGthread *lwogThreadCreate(void(*threadFunc)(void*), void *args, size_t stackSize)
{
#if defined(LW_HOS)
    // HOS seems to pick core #0 for all worker threads if you don't specify a
    // core.
    return lwogThreadCreateOnCore(threadFunc, args, stackSize, 0);
#else
    LWOGthread *thread = new LWOGthread;
    if (LwOsThreadCreate(threadFunc, args, &thread->osThread) != LwSuccess) {
        LwOsDebugPrintf("Failed to create worker thread.\n");
        delete thread;
        return NULL;
    }
    return thread;
#endif
}

void lwogThreadWait(LWOGthread *thread)
{
    if (!thread) {
        return;
    }
#if defined(LW_HOS)
    nn::os::WaitThread(&thread->osThread);
    nn::os::DestroyThread(&thread->osThread);
    lwnUtil::AlignedStorageFree(thread->stack);
#else
    LwOsThreadJoin(thread->osThread);
#endif
    delete thread;
}

void lwogThreadYield(void)
{
#if defined(LW_HOS)
    nn::os::YieldThread();
#else
    LwOsThreadYield();
#endif
}

#if defined(LW_HOS)
uint64_t lwogThreadGetAvailableCoreMask(void)
{
    return nn::os::GetThreadAvailableCoreMask();
}

int lwogThreadGetLwrrentCoreNumber(void)
{
    return nn::os::GetLwrrentCoreNumber();
}

void lwogThreadSetCoreMask(int idealCore, uint64_t coreMask)
{
    nn::os::SetThreadCoreMask(nn::os::GetLwrrentThread(), idealCore, coreMask);

}

int lwogThreadSelectCoreRoundRobin(int threadID, uint64_t coreMask)
{
    int coreNum = 0;
    int threadsLeft = threadID + 1;

    if (!coreMask) {
        return 0;
    }

    // Find a slot in <coreMask> for thread <threadID> by looping over the
    // core mask until we have found <threadID>+1 one bits.  Use the core
    // corresponding to the last bit we found.
    while (1) {
        uint64_t coreBit = 1ULL << coreNum;
        if (coreBit & coreMask) {
            threadsLeft--;
            if (threadsLeft == 0) {
                break;
            }
        } else if (coreBit > coreMask) {
            // If we've run off the end of the core mask, loop back around the
            // beginning immediately.
            coreNum = 0;
            continue;
        }
        coreNum = (coreNum + 1) % 64;
    }

    return coreNum;
}

#endif // #if defined(LW_HOS)

void lwogRunOnWorkerThread(void (*threadFunc)(void*), void *args, size_t stackSize /* 0 = default */)
{
    LWOGthread *thread = lwogThreadCreate(threadFunc, args, stackSize);
    lwogThreadWait(thread);
}

void lwogSetWindowTitle(const char *title)
{
    // Nothing implemented here.
}

uint64_t lwogGetTimerValue()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

uint64_t lwogGetTimerFrequency()
{
    return 1000000;
}

extern void lwogSetupGLContext(int enable)
{
    // Nothing implemented here.
}

uint32_t lwogGetRefreshRate()
{
    // For HOS, refresh rate is 60 Hz
    return 60;
}

} // extern "C"

#ifdef LW_HOS
static void printFileHandle(LwOsFileHandle fh, const char *format, va_list &ap)
{
    // Print the contents of <format> and <ap> into file <fh>.
    if (fh != NULL) {
        int BUFSIZE = 256;
        char buf[BUFSIZE];
        int ideal_len = vsnprintf(buf, BUFSIZE, format, ap);
        if (ideal_len < BUFSIZE)
        {
            LwOsFwrite(fh, buf, ideal_len);
        } else {
            // BUFSIZE is too small; allocate a temporary buffer
            char *temp = (char*) malloc(ideal_len+1);
            if (temp) {
                ideal_len = vsnprintf(temp, ideal_len+1, format, ap);
                LwOsFwrite(fh, temp, ideal_len);
                free(temp);
            } else {
                NN_LOG("Unable to allocate vsnprintf buffer\n");
            }
        }
    } else {
        NN_VLOG(format, ap);
    }
}
#endif
