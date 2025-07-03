/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "gdm_logger.h"
#include "mle_protobuf_writer.h"

#include <time.h>

namespace Mle
{
    constexpr UINT64 NoUid = ~0ULL;
    constexpr INT32  NoId  = -1;

    struct Context
    {
        UINT64 uid       = NoUid;
        UINT64 timestamp = 0;
        INT32  threadId  = NoId;
        INT32  devInst   = NoId;
        INT32  testId    = NoId;
    };
}

namespace
{
    UINT64                s_MleUID      = 0ULL;
    UINT64                s_StartTimeNs = 0;
    LwDiagUtils::Priority s_PrevPri     = LwDiagUtils::PriNone;
    Mle::Context          s_PrevMleContext;

    FILE * s_pFile = nullptr;

    constexpr UINT32 MaxPrintSize = 65536;
    char s_Buff[MaxPrintSize + 1];

    UINT64 GetWallTimeNs()
    {
        struct timespec tv;
        if (0 != clock_gettime(CLOCK_REALTIME, &tv))
        {
            return 0;
        }
        const UINT64 sec = static_cast<UINT64>(tv.tv_sec);
        const UINT64 nsec = static_cast<UINT64>(tv.tv_nsec);
        return 1000000000ULL * sec + nsec;
    }
}

LwDiagUtils::EC GdmLogger::Open(const string & filename)
{
    s_StartTimeNs = GetWallTimeNs();

    s_PrevMleContext.uid        = 0;
    s_PrevMleContext.timestamp  = 0;
    s_PrevMleContext.threadId   = 0;
    s_PrevMleContext.devInst    = 0;
    s_PrevMleContext.testId     = 0;

    const char * openMode = "w+b";
    const LwDiagUtils::EC ec = LwDiagUtils::OpenFile(filename.c_str(), &s_pFile, openMode);

    ByteStream bs;
    Mle::MLE(&bs).file_type("MLE");
    fwrite(bs.data(), 1, bs.size(), s_pFile);
    return ec;
}

INT32 GdmLogger::Printf
(
   LwDiagUtils::Priority pri,
   const char *          format,
   ... //       Arguments
)
{
   va_list arguments;
   va_start(arguments, format);

   // Write out the formatted string.
   INT32 charactersWritten = GdmLogger::VAPrintf(pri, 0, 0, format, arguments);

   va_end(arguments);

   return charactersWritten;
}

INT32 GdmLogger::VAPrintf
(
   INT32                     priority,
   UINT32                    moduleCode,
   UINT32                    sps,
   const char              * format,
   va_list                   restOfArgs
)
{
    // Write out the formatted string.
    INT32 charactersWritten = vsnprintf(s_Buff, sizeof(s_Buff) - 1, format, restOfArgs);
    LWDASSERT(charactersWritten < static_cast<INT32>(sizeof(s_Buff)));

    ByteStream bs;
    const size_t len = charactersWritten -
        ((charactersWritten && (s_Buff[charactersWritten - 1] == '\n')) ? 1 : 0);
    Mle::Entry::print(&bs).EmitValue(Mle::ProtobufString(s_Buff, len), Mle::Output::Force);
    PrintMle(&bs, static_cast<LwDiagUtils::Priority>(priority));
    printf("%s", s_Buff);
    return charactersWritten;
}

void GdmLogger::PrintMle(ByteStream* pBytes, LwDiagUtils::Priority pri)
{
    if (s_pFile == nullptr)
    {
        return;
    }

    Mle::Context ctx;
    ctx.uid       = ++s_MleUID;
    ctx.timestamp = GetWallTimeNs() - s_StartTimeNs;

    const auto MakeId = [](INT32 oldId, INT32 newId) -> INT32
    {
        return (oldId == newId) ? 0 :
                   (newId == -1) ? -1 : (newId + 1);
    };

    auto entry = Mle::MLE::entry(pBytes, Mle::DumpPos(0));
    entry.uid_delta(static_cast<INT64>(ctx.uid - s_PrevMleContext.uid - 1));
    entry.timestamp_delta(static_cast<INT64>(ctx.timestamp - s_PrevMleContext.timestamp));
    entry.thread_id(MakeId(s_PrevMleContext.threadId, ctx.threadId));
    entry.test_id(MakeId(s_PrevMleContext.testId, ctx.testId));
    entry.dev_id(MakeId(s_PrevMleContext.devInst, ctx.devInst));
    if (pri != s_PrevPri)
    {
        entry.priority(static_cast<Mle::Entry::Priority>(pri));
    }
    entry.Finish();

    const auto size = pBytes->size();
    fwrite(pBytes->data(), 1, size, s_pFile);

    // The context fields (written to MLE file above) are defined to have the
    // value of 0 if they did not change between log entries.  This allows us
    // to reduce log size, since these fields very often have the same value
    // in conselwtive log entries, and the value of 0 is not emitted into the
    // log.  To facilitate this, we save the previous value of context fields.
    s_PrevMleContext = ctx;
    s_PrevPri = pri;
}

void GdmLogger::Close()
{
    if (s_pFile)
    {
        fflush(s_pFile);
        fclose(s_pFile);
        s_pFile = nullptr;
    }
}

bool GdmLogger::IsMleLimited()
{ 
    // Default
    return false; 
}
