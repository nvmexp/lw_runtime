/*
* LWIDIA_COPYRIGHT_BEGIN
*
* Copyright 2017-2022 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

#include "vkmodssub.h"
#include "core/include/errormap.h"
#include "core/include/imagefil.h"
#include "core/include/jscript.h"
#include "core/include/platform.h"
#include "core/include/script.h"
#undef UNSIGNED_CAST
#include "core/include/utility.h"
#include "vktest_platform.h"

extern "C" {
    GLenum dglThreadAttach() { return 0; }
    GLenum dglThreadDetach() { return 0; }
}

OneTimeInit::OneTimeInit()
{
}
OneTimeInit::~OneTimeInit()
{
}

void RC::Warn() const
{
    Printf(Tee::PriHigh, "Warning: possible lost RC %d.\n", m_rc);
}

void Tee::FlushToDisk()
{
}

INT32 Printf
(
    INT32        Priority,
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
INT32 Printf
(
    INT32        Priority,
    UINT32       ModuleCode,
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

#ifdef MODS_INCLUDE_DEBUG_PRINTS

INT32 Printf
(
   Tee::PriDebugStub,
   const char * format,
   ... //       arguments
)
{
    return 0;
}

INT32 Printf
(
   Tee::PriDebugStub,
   UINT32       moduleCode,
   const char * format,
   ... //       arguments
)
{
    return 0;
}

#endif

extern "C"
{
    INT32 ModsExterlwAPrintf
    (
       INT32       priority,
       UINT32      moduleCode,
       UINT32      sps,
       const char* format,
       va_list     arguments
    )
    {
        return PlatformVAPrintf(format, arguments);
    }
}

namespace Tee
{
    unsigned GetTeeModuleCoreCode()
    {
        return 0;
    }
}

#ifdef NDEBUG
#define ModsAssertFail
#else
static const char *s_ErrorMessages[] =
{
    "OK",
    #undef  DEFINE_RC
    #define DEFINE_RC(errno, code, message) message,
    #define DEFINE_RETURN_CODES
    #include "core/error/errors.h"
    "Limit"
};

void ModsAssertFail(const char *file, int line, const char *function, const char *cond)
{
    PlatformPrintf("%s: MASSERT(%s) failed at %s line %d\n", function, cond, file, line);
}
#endif

ErrorMap::ErrorMap
(
    RC rc
)
: m_rc(rc)
{
}

const char * ErrorMap::Message() const
{
#ifdef NDEBUG
    return "Unknown error";
#else
    const INT32 rcValue = m_rc.Get();
    if ((rcValue < 0) || (rcValue >= static_cast<INT32>(NUMELEMS(s_ErrorMessages))))
        return "Unknown error";

    return s_ErrorMessages[rcValue];
#endif
}

JSBool C_Global_Help
(
    JSContext *    pContext,
    JSObject  *    pObject,
    uintN          NumArguments,
    jsval     *    pArguments,
    jsval     *    pReturlwalue
)
{
    RETURN_RC(OK);
}
//-------------------------------------------------------------------------------------------------
// Generate a timestamp encoded filename while keeping the filepath for a file so we can use the
// -run_on_error -dump_png_on_error -loops commandline args and get a unique file for each
// frame that miscompares.
string Utility::GetTimestampedFilename(const char *pTestName, const char * pExt)
{
    string filename(pTestName);
    filename += ".";
    filename += pExt;
    return filename;
}

RC Utility::InterpretModsUtilErrorCode(LwDiagUtils::EC ec)
{
    switch (ec)
    {
        case LwDiagUtils::OK:                 return OK;
        case LwDiagUtils::FILE_2BIG:          return RC::FILE_2BIG;
        case LwDiagUtils::FILE_ACCES:         return RC::FILE_ACCES;
        case LwDiagUtils::FILE_AGAIN:         return RC::FILE_AGAIN;
        case LwDiagUtils::FILE_BADF:          return RC::FILE_BADF;
        case LwDiagUtils::FILE_BUSY:          return RC::FILE_BUSY;
        case LwDiagUtils::FILE_CHILD:         return RC::FILE_CHILD;
        case LwDiagUtils::FILE_DEADLK:        return RC::FILE_DEADLK;
        case LwDiagUtils::FILE_EXIST:         return RC::FILE_EXIST;
        case LwDiagUtils::FILE_FAULT:         return RC::FILE_FAULT;
        case LwDiagUtils::FILE_FBIG:          return RC::FILE_FBIG;
        case LwDiagUtils::FILE_INTR:          return RC::FILE_INTR;
        case LwDiagUtils::FILE_ILWAL:         return RC::FILE_ILWAL;
        case LwDiagUtils::FILE_IO:            return RC::FILE_IO;
        case LwDiagUtils::FILE_ISDIR:         return RC::FILE_ISDIR;
        case LwDiagUtils::FILE_MFILE:         return RC::FILE_MFILE;
        case LwDiagUtils::FILE_MLINK:         return RC::FILE_MLINK;
        case LwDiagUtils::FILE_NAMETOOLONG:   return RC::FILE_NAMETOOLONG;
        case LwDiagUtils::FILE_NFILE:         return RC::FILE_NFILE;
        case LwDiagUtils::FILE_NODEV:         return RC::FILE_NODEV;
        case LwDiagUtils::FILE_NOENT:         return RC::FILE_DOES_NOT_EXIST;
        case LwDiagUtils::FILE_NOEXEC:        return RC::FILE_NOEXEC;
        case LwDiagUtils::FILE_NOLCK:         return RC::FILE_NOLCK;
        case LwDiagUtils::FILE_NOMEM:         return RC::FILE_NOMEM;
        case LwDiagUtils::FILE_NOSPC:         return RC::FILE_NOSPC;
        case LwDiagUtils::FILE_NOSYS:         return RC::FILE_NOSYS;
        case LwDiagUtils::FILE_NOTDIR:        return RC::FILE_NOTDIR;
        case LwDiagUtils::FILE_NOTEMPTY:      return RC::FILE_NOTEMPTY;
        case LwDiagUtils::FILE_NOTTY:         return RC::FILE_NOTTY;
        case LwDiagUtils::FILE_NXIO:          return RC::FILE_NXIO;
        case LwDiagUtils::FILE_PERM:          return RC::FILE_PERM;
        case LwDiagUtils::FILE_PIPE:          return RC::FILE_PIPE;
        case LwDiagUtils::FILE_ROFS:          return RC::FILE_ROFS;
        case LwDiagUtils::FILE_SPIPE:         return RC::FILE_SPIPE;
        case LwDiagUtils::FILE_SRCH:          return RC::FILE_SRCH;
        case LwDiagUtils::FILE_XDEV:          return RC::FILE_XDEV;
        case LwDiagUtils::FILE_UNKNOWN_ERROR: return RC::FILE_UNKNOWN_ERROR;
        case LwDiagUtils::SOFTWARE_ERROR:     return RC::SOFTWARE_ERROR;
        case LwDiagUtils::ILWALID_FILE_FORMAT:return RC::ILWALID_FILE_FORMAT;
        case LwDiagUtils::CANNOT_ALLOCATE_MEMORY:return RC::CANNOT_ALLOCATE_MEMORY;
        case LwDiagUtils::FILE_DOES_NOT_EXIST:return RC::FILE_DOES_NOT_EXIST;
        case LwDiagUtils::PREPROCESS_ERROR:   return RC::PREPROCESS_ERROR;
        case LwDiagUtils::BAD_COMMAND_LINE_ARGUMENT:return RC::BAD_COMMAND_LINE_ARGUMENT;
        case LwDiagUtils::BAD_BOUND_JS_FILE:  return RC::SOFTWARE_ERROR;
        case LwDiagUtils::NETWORK_NOT_INITIALIZED: return RC::NETWORK_NOT_INITIALIZED;
        case LwDiagUtils::NETWORK_ALREADY_CONNECTED: return RC::NETWORK_ALREADY_CONNECTED;
        case LwDiagUtils::NETWORK_CANNOT_CREATE_SOCKET: return RC::NETWORK_CANNOT_CREATE_SOCKET;
        case LwDiagUtils::NETWORK_CANNOT_CONNECT: return RC::NETWORK_CANNOT_CONNECT;
        case LwDiagUtils::NETWORK_CANNOT_BIND: return RC::NETWORK_CANNOT_BIND;
        case LwDiagUtils::NETWORK_ERROR:       return RC::NETWORK_ERROR;
        case LwDiagUtils::NETWORK_WRITE_ERROR: return RC::NETWORK_WRITE_ERROR;
        case LwDiagUtils::NETWORK_READ_ERROR:  return RC::NETWORK_READ_ERROR;
        case LwDiagUtils::NETWORK_NOT_CONNECTED: return RC::NETWORK_NOT_CONNECTED;
        case LwDiagUtils::NETWORK_CANNOT_DETERMINE_ADDRESS: return RC::NETWORK_CANNOT_DETERMINE_ADDRESS;
        case LwDiagUtils::TIMEOUT_ERROR:        return RC::TIMEOUT_ERROR;
        case LwDiagUtils::UNSUPPORTED_FUNCTION: return RC::UNSUPPORTED_FUNCTION;
        case LwDiagUtils::BAD_PARAMETER:        return RC::BAD_PARAMETER;
        case LwDiagUtils::DLL_LOAD_FAILED:      return RC::DLL_LOAD_FAILED;
        default:
            MASSERT(!"Unknown Error Code from LwDiagUtils");
            return RC::SOFTWARE_ERROR;
    }
}

Platform::SimulationMode Platform::GetSimulationMode()
{
    return Hardware;
}

UINT64 Platform::GetTimeMS()
{
    return Xp::GetWallTimeMS();
}

UINT64 Platform::GetTimeNS()
{
    return Xp::GetWallTimeNS();
}

bool Platform::PollfuncWrap
(
    PollFunc    pollFunc,
    void*       pArgs,
    const char* pollFuncName
)
{
    return pollFunc(pArgs);
}

void Platform::MemCopy(volatile void *Dst, const volatile void *Src, size_t Count)
{
    memcpy(const_cast<void*>(Dst), const_cast<void*>(Src), Count);
}

void Platform::VirtualRd(const volatile void *Addr, void *Data, UINT32 Count)
{
    MemCopy(Data, Addr, Count);
}

void Platform::VirtualWr(volatile void *Addr, const void *Data, UINT32 Count)
{
    MemCopy(Addr, Data, Count);
}

RC Platform::FlushCpuWriteCombineBuffer()
{
    return RC::OK;
}

void Utility::DelayNS
(
    UINT32 Nanoseconds
)
{
    UINT64 End = Xp::QueryPerformanceCounter() +
        (Nanoseconds * Xp::QueryPerformanceFrequency() / 1000000000);

    while (Xp::QueryPerformanceCounter() < End)
    {
    }
}

UINT16 Utility::Float32ToFloat16( FLOAT32 fin )
{
    return 42;
}

Utility::SelwrityUnlockLevel Utility::GetSelwrityUnlockLevel()
{
    return SUL_LWIDIA_NETWORK;
}

Utility::StopWatch::StopWatch(string name, Tee::Priority pri)
: m_Name(move(name))
, m_StartTime(Xp::GetWallTimeNS())
, m_Pri(pri)
{
}

UINT64 Utility::StopWatch::Stop()
{
    UINT64 elapsed = 0;
    if (m_StartTime)
    {
        elapsed = Xp::GetWallTimeNS() - m_StartTime;
        Printf(m_Pri, "%s: %f s\n",
               m_Name.c_str(), elapsed/1000000000.0);
        m_StartTime = 0;
    }
    return elapsed;
}

Utility::TotalStopWatch::TotalStopWatch(string name)
: m_Name(move(name))
, m_TotalTime(0)
{
}

Utility::TotalStopWatch::~TotalStopWatch()
{
    Printf(Tee::PriNormal, "%s: %f s\n", m_Name.c_str(), m_TotalTime/1000000000.0);
}

Utility::PartialStopWatch::PartialStopWatch(TotalStopWatch& timer)
: m_Timer(timer)
, m_StartTime(Xp::GetWallTimeNS())
{
}

Utility::PartialStopWatch::~PartialStopWatch()
{
    m_Timer.m_TotalTime += Xp::GetWallTimeNS() - m_StartTime;
}

JavaScript *JavaScriptPtr::s_pInstance = nullptr;

RC JavaScript::ToJsval
(
    bool    Boolean,
    jsval * pValue
)
{
    return OK;
}
RC JavaScript::ToJsval
(
    double d,
    jsval * pjsv
)
{
    return OK;
}
RC JavaScript::FromJsval
(
    jsval   Value,
    bool  * pBoolean
)
{
    return OK;
}
RC JavaScript::FromJsval
(
    jsval    Value,
    UINT32 * pUinteger
)
{
    return OK;
}
RC JavaScript::FromJsval
(
    jsval     Value,
    JsArray * pArray
)
{
    return OK;
}
RC JavaScript::FromJsval
(
    jsval       Value,
    JSObject ** ppObject
)
{
    return OK;
}
RC JavaScript::SetElementJsval
(
    JSObject * pArrayObject,
    INT32      Index,
    jsval      Value
)
{
    return OK;
}
void JavaScript::ClearZombies()
{
}

RC Goldelwalues::SetSurfaces(GoldenSurfaces *pgs)
{
    m_pGS = pgs;
    return OK;
}

RC Goldelwalues::Run()
{
    if (m_pGS == nullptr)
        return RC::SOFTWARE_ERROR;

    RC rc;
    UINT32 numSurfaces = m_pGS->NumSurfaces();

    for (UINT32 surfaceIdx = 0; surfaceIdx < numSurfaces; surfaceIdx++)
    {
        void *cachedAddress = m_pGS->GetCachedAddress(surfaceIdx, opCpuDma, 0, nullptr);
        if (cachedAddress == nullptr)
        {
            return RC::MODS_VK_ERROR_MEMORY_MAP_FAILED;
        }

        char filename[128];
        snprintf(filename, sizeof(filename),
            "vktest-%d%s.png", m_Loop, m_pGS->Name(surfaceIdx).c_str());

        CHECK_RC(ImageFile::WritePng(filename, m_pGS->Format(surfaceIdx), cachedAddress,
            m_pGS->Width(surfaceIdx), m_pGS->Height(surfaceIdx),
            m_pGS->Pitch(surfaceIdx), false, false));
    }

    m_pGS->Ilwalidate();

    return rc;
}

RC Goldelwalues::ErrorRateTest(JSObject *tstSObj)
{
    return OK;
}

void Goldelwalues::SetLoop(UINT32 Loop)
{
    m_Loop = Loop;
}

void Goldelwalues::SetLoopAndDbIndex(UINT32 loop, UINT32 dbIdx)
{
    SetLoop(loop);
}

Goldelwalues::Action Goldelwalues::GetAction() const
{
    return Store;
}

UINT32 Goldelwalues::GetCodes() const
{
    return m_Codes;
}

void Goldelwalues::SetCodes(UINT32 codes)
{
    m_Codes = codes;
}

void   Goldelwalues::SetSkipCount(UINT32 skipCount)
{
    m_SkipCount = skipCount;
}

UINT32 Goldelwalues::GetSkipCount() const
{
    return m_SkipCount;
}

void Goldelwalues::SetDumpPng(UINT32 dumpPng)
{
    m_DumpPng = dumpPng;
}

UINT32 Goldelwalues::GetDumpPng() const
{
    return m_DumpPng;
}

void Goldelwalues::SetDeferredRcWasRead(bool bWasRead)
{
    m_DeferredRcWasRead = bWasRead;
}


namespace ErrorLogger
{
    UINT32 d_ErrorCount = 0;
    void LogError(const char *errStr, LogType errType) { d_ErrorCount++; }
    UINT32 GetErrorCount() { return d_ErrorCount; }
}

namespace CommandLine
{
    string LogFileName() { return "mods.log"; }
}

Goldelwalues * GpuTest::GetGoldelwalues()
{
    return &m_Goldelwalues;
}

string Gpu::DeviceIdToString(Gpu::LwDeviceId)
{
    return "unknown";
}

Gpu::LwDeviceId GpuSubdevice::DeviceId()
{
    return m_DeviceId;
}

void GpuSubdevice::SetDeviceId(UINT32 vendorId, UINT32 deviceID)
{
    if (vendorId != 0x10de)
    {
        return;
    }

#define DEFINE_NEW_GPU( DIDStart, DIDEnd, ChipId, LwId, ...) \
    if ((deviceID >= (DIDStart)) && (deviceID <= (DIDEnd))) { m_DeviceId = Gpu::LwId; return; };
#define DEFINE_OBS_GPU(...) // don't care
#define DEFINE_DUP_GPU(DIDStart, DIDEnd, ChipId, LwId) \
    if ((deviceID >= (DIDStart)) && (deviceID <= (DIDEnd))) { m_DeviceId = Gpu::LwId; return; };
#include "gpu/include/gpulist.h"
#undef DEFINE_OBS_GPU
#undef DEFINE_DUP_GPU
#undef DEFINE_NEW_GPU
}

FloorsweepImpl* GpuSubdevice::GetFsImpl()
{
    return &m_FsImpl;
}

RC MemError::LogOffsetError(UINT32 width,
                            UINT64 sampleAddr,
                            UINT64 actual,
                            UINT64 expected,
                            UINT32 pteKind,
                            UINT32 pageSizeKB,
                            const string &patternName,
                            UINT32 patternOffset)
{
    return OK;
}

bool GpuTest::IsSupported()
{
    return OK;
}

void GpuTest::PrintJsProperties(Tee::Priority pri)
{
}

RC GpuTest::AddExtraObjects(JSContext* cx, JSObject* obj)
{
    return OK;
}

RC GpuTest::Setup()
{
    return OK;
}

RC GpuTest::Run()
{
    return OK;
}

RC GpuTest::Cleanup()
{
    return OK;
}

RC GpuTest::EndLoopMLE(UINT64 mleProgress) const
{
    return OK;
}

RC GpuTest::PrintProgressInit(const UINT64 maxIterations) const
{
    return OK;
}

RC GpuTest::PrintProgressUpdate(const UINT64 lwrrentIteration) const
{
    return OK;
}

JSObject * GpuTest::GetJSObject()
{
    return nullptr;
}

void GpuTest::SetWindowParams(HINSTANCE hinstance, HWND hWindow)
{
    m_Hinstance = hinstance;
    m_HWindow = hWindow;
}

Tee::Priority GpuTest::GetVerbosePrintPri() const
{
    return Tee::PriSecret;
}
bool GpuTestConfiguration::GetVerbose()
{
    return m_Verbose;
}

void GpuTestConfiguration::SetVerbose(bool bVerbose)
{
    m_Verbose = bVerbose;
}

UINT32 GpuTestConfiguration::GetGpuIndex() const
{
    return m_GpuIndex;
}

void GpuTestConfiguration::SetGpuIndex(UINT32 gpuIndex)
{
    m_GpuIndex = gpuIndex;
}

void GpuTestConfiguration::SetLoops(UINT32 loops)
{
    m_Loops = loops;
}

UINT32 GpuTestConfiguration::Loops() const
{
    return m_Loops;
}

UINT32 GpuTestConfiguration::DisplayWidth() const
{
    return m_Width;
}

UINT32 GpuTestConfiguration::DisplayHeight() const
{
    return m_Height;
}

UINT32 GpuTestConfiguration::SurfaceWidth() const
{
    return m_Width;
}

UINT32 GpuTestConfiguration::SurfaceHeight() const
{
    return m_Height;
}

void GpuTestConfiguration::SetDisplayWidth(UINT32 width)
{
    m_Width = width;
}

void GpuTestConfiguration::SetDisplayHeight(UINT32 height)
{
    m_Height = height;
}

FLOAT64 GpuTestConfiguration::TimeoutMs() const
{
    return 1000.0;
}

UINT32 GpuTestConfiguration::Seed() const
{
    return 0x12345678;
}

GpuTestConfiguration * GpuTest::GetTestConfiguration()
{
    return &m_TestConfig;
}

UINT32 FloorsweepImpl::FbpMask() const
{
    return 0xF;
}

GpuSubdevice * GpuTest::GetBoundGpuSubdevice()
{
    return &m_BoundGpuSubdevice;
}

static uint32_t *s_pEarlyExit = nullptr;
void SetEarlyExit(uint32_t* earlyExit)
{
    s_pEarlyExit = earlyExit;
}

RC Abort::Check()
{
    if (s_pEarlyExit && *s_pEarlyExit)
    {
        return RC::USER_ABORTED_SCRIPT;
    }

    return OK;
}

RC Abort::CheckAndReset()
{
    return OK;
}
