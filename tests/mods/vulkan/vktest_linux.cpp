/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "vkmain.h"
#include <stdlib.h>
#include <time.h>

#include "core/include/tasker.h"

INT32 PlatformPrintf
(
    const char * Format,
    ... //       Arguments
)
{
    va_list Arguments;
    va_start(Arguments, Format);

    int num = PlatformVAPrintf(Format, Arguments);

    va_end(Arguments);

    return num;
}

INT32 PlatformVAPrintf
(
    const char * Format,
    va_list      RestOfArgs
)
{
#if defined(ENABLE_UNIT_TESTING) || defined(NDEBUG)
    // Don't print if we are using the mock driver as the test
    // spew will mostly show faked info:
    return 0;
#else
    const size_t MaxSize = 4096;
    char Buf[MaxSize];

    int num = vsnprintf(Buf, MaxSize, Format, RestOfArgs);

    fwrite(Buf, 1, num, stdout);

    return num;
#endif
}

void PlatformOnEntry()
{
    Tasker::Initialize();
};

RC SharedSysmem::Initialize()
{
    return RC::OK;
}

void SharedSysmem::Shutdown()
{
}

void* Xp::AllocOsEvent(UINT32 hClient, UINT32 hDevice)
{
    return nullptr;
}

void Xp::BreakPoint()
{
    printf("\n*** Xp::BreakPoint called -- aborting! ***\n");
    exit(1);
}

void Xp::FreeOsEvent(void* pFd, UINT32 hClient, UINT32 hDevice)
{
}

RC Xp::WaitOsEvents
(
   void**  pOsEvents,
   UINT32  numOsEvents,
   UINT32* pCompletedIndices,
   UINT32  maxCompleted,
   UINT32* pNumCompleted,
   FLOAT64 timeoutMs
)
{
    return RC::OK;
}

void Xp::SetOsEvent(void* pFd)
{
}

UINT64 Xp::GetWallTimeMS()
{
    return GetWallTimeNS() / 1'000'000;
}

UINT64 Xp::GetWallTimeUS()
{
    return GetWallTimeNS() / 1000;
}

UINT64 Xp::GetWallTimeNS()
{
    struct timespec ts;

    UINT64 timeNs = 0;

    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        timeNs =  static_cast<UINT64>(ts.tv_sec) * 1'000'000'000;
        timeNs += static_cast<UINT64>(ts.tv_nsec);
    }

    return timeNs;
}

UINT64 Xp::QueryPerformanceCounter()
{
    return GetWallTimeNS();
}

UINT64 Xp::QueryPerformanceFrequency()
{
    return 1'000'000'000;
}

Xp::OperatingSystem Xp::GetOperatingSystem()
{
    return OS_LINUXRM;
}
