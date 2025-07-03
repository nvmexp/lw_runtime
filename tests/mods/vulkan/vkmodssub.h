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

#pragma once

#include "core/include/rc.h"
#include "core/include/abort.h"
#include "core/include/color.h"
#include "core/include/fpicker.h"
#include "vulkan.h"
#include "g_lwconfig.h"

#ifdef NDEBUG
#define VerbosePrintf
#else
#define VerbosePrintf PlatformPrintf
#endif

typedef unsigned int GLenum;
extern "C" {
    extern GLenum dglThreadAttach();
    extern GLenum dglThreadDetach();
}

class GoldenSurfaces;
struct JSObject;

class Goldelwalues
{
public:
    enum Action
    {
        Check       //!< normal mode -- check for errors.
        ,Store       //!< store these values as the new golden values
        ,Skip        //!< don't do anything.
    };
    enum Code
    {
        CheckSums  = 1 << 0 //!< do a set of component-wise checksums
        ,Crc        = 1 << 1 //!< do a cyclic redundancy check on color and zeta buffers
        ,DacCrc     = 1 << 2 //!< do a DAC CRC
        ,TmdsCrc    = 1 << 3 //!< do a TMDS/LVDS CRC
        // Unsused bit 4
        ,OldCrc     = 1 << 5 //!< use the CRC algorithm that matches the emulation golden values
        ,Extra      = 1 << 6 //!< extra code (test-specific)
        ,TpcCrc     = 1 << 7 //!< per-TPC CRC code (for shader floorsweeping)
        ,ExtCrc     = 1 << 8 //!< external device CRC
        ,END_CODE   = 1 << 9 // no such code, used in for loops inside golden.cpp
        ,SurfaceCodes = CheckSums|Crc|OldCrc|TpcCrc
        ,DisplayCodes = DacCrc|TmdsCrc
        ,OtherCodes = Extra|ExtCrc
    };
    enum When
    {
        Never      =  0  //!< do (dumpTGA/pause/whatever) never
        ,OnStore    =  1  //!< do (...) on Store
        ,OnCheck    =  2  //!< do (...) on Check
        ,OnSkip     =  4  //!< do (...) on Skip
        ,OnError    =  8  //!< do (...) when Check fails due to too many bad pixels
        ,OnBadPixel = 16  //!< do (...) when Check passes, but there was a bad pixel
        ,Always     = 32  //!< do (...) on every loop
    };
    enum BufferFetchHint
    {
        opCpu       = 0
        ,opCpuDma   = 1
    };

    RC SetSurfaces(GoldenSurfaces *pgs);

    RC Run();

    RC ErrorRateTest
    (
        JSObject * tstSObj
    );

    void SetCodes(UINT32 codes);
    void SetDeferredRcWasRead(bool bWasRead);

    void SetLoop(UINT32 Loop);
    void SetLoopAndDbIndex(UINT32 loop, UINT32 dbIdx);
    void SetSkipCount(UINT32 skipCount);
    void SetDumpPng(UINT32 dumpPng);

    Action GetAction() const;
    UINT32 GetCodes() const;
    UINT32 GetSkipCount() const;
    UINT32 GetDumpPng() const;
    bool GetStopOnError() const { return false; }
private:
    GoldenSurfaces *m_pGS = nullptr;
    UINT32 m_Loop = 9999;
    UINT32 m_Codes = 0;
    UINT32 m_SkipCount = 0;
    UINT32 m_DumpPng = 0;
    bool m_DeferredRcWasRead = true;
};

class GoldenSurfaces
{
public:
    virtual ~GoldenSurfaces() {}

    virtual int NumSurfaces() const = 0;
    virtual const string & Name (int surfNum) const = 0;
    virtual RC CheckAndReportDmaErrors(UINT32 subdevNum) = 0;
    virtual void * GetCachedAddress
    (
        int surfNum,
        Goldelwalues::BufferFetchHint bufFetchHint,
        UINT32 subdevNum,
        vector<UINT08> *surfDumpBuffer
    ) = 0;
    virtual void Ilwalidate() = 0;
    virtual INT32 Pitch(int surfNum) const = 0;
    virtual UINT32 Width(int surfNum) const = 0;
    virtual UINT32 Height(int surfNum) const = 0;
    virtual ColorUtils::Format Format(int surfNum) const = 0;
    virtual UINT32 Display(int surfNum) const = 0;
    virtual RC GetPitchAlignRequirement(UINT32 *pitch) = 0;
private:
    UINT32       m_SkipCount;           //!< loops to skip between checks or stores

};

namespace ErrorLogger
{
    enum LogType
    {
        LT_ERROR = 0xFE,
        LT_INFO = 0xFF
    };
    void LogError(const char *errStr, LogType errType);
    UINT32 GetErrorCount();
};

namespace CommandLine
{
    string LogFileName();
};

class GpuTestConfiguration
{
public:
    UINT32              Loops() const;
    void                SetLoops(UINT32 loops);
    UINT32              DisplayWidth() const;
    UINT32              DisplayHeight() const;
    UINT32              SurfaceWidth() const;
    UINT32              SurfaceHeight() const;
    void                SetDisplayWidth(UINT32 width);
    void                SetDisplayHeight(UINT32 height);
    FLOAT64             TimeoutMs() const;
    UINT32              Seed() const;
    bool                GetVerbose();
    void                SetVerbose(bool bVerbose);
    UINT32              GetGpuIndex() const;
    void                SetGpuIndex(UINT32 gpuIndex);
private:
    bool                m_Verbose = false;
    UINT32              m_Loops = 0;
    UINT32              m_Width = 640;
    UINT32              m_Height = 480;
    UINT32              m_GpuIndex = 0;
};

class FloorsweepImpl
{
public:
    virtual ~FloorsweepImpl() {}
    UINT32 FbpMask() const;
};

class Gpu
{
public:
    enum LwDeviceId
    {
        LW0 = 0
#define DEFINE_NEW_GPU( DIDStart, DIDEnd, ChipId, LwId, Constant, ...) \
        ,LwId = Constant
#define DEFINE_OBS_GPU(...) // don't care
#define DEFINE_DUP_GPU(...) // don't care
#include "gpu/include/gpulist.h"
#undef DEFINE_OBS_GPU
#undef DEFINE_DUP_GPU
#undef DEFINE_NEW_GPU
    };
    static string DeviceIdToString(Gpu::LwDeviceId ndi);
};

namespace Device
{
    enum Feature
    {
        GPUSUB_SUPPORTS_VULKAN
    };
}

class GpuSubdevice
{
public:
    Gpu::LwDeviceId DeviceId();
    void SetDeviceId(UINT32 vendorId, UINT32 deviceID);
    FloorsweepImpl* GetFsImpl();
    bool HasFeature(Device::Feature) const { return true; }
    unsigned GetRTCoreCount() const { return 1; }
    unsigned GetGpuInst() const { return 0; }
    RC HwGpcToVirtualGpc(UINT32 HwGpcNum, UINT32 *pVirtualGpcNum)
    {
        *pVirtualGpcNum = HwGpcNum;
        return RC::OK;
    }
    RC HwGpcToVirtualGpcMask(UINT32 HwGpcMask, UINT32 *pVirtualGpcMask)
    {
        *pVirtualGpcMask = HwGpcMask;
        return RC::OK;
    }

private:
    Gpu::LwDeviceId m_DeviceId = Gpu::LW0;
    FloorsweepImpl m_FsImpl = {};
};

class GpuDevice;

class MemError
{
public:
    RC LogOffsetError(UINT32 width,
                      UINT64 sampleAddr,
                      UINT64 actual,
                      UINT64 expected,
                      UINT32 pteKind,
                      UINT32 pageSizeKB,
                      const string &patternName,
                      UINT32 patternOffset);
};

#ifdef __linux__
typedef struct xcb_connection_t* HINSTANCE;
typedef UINT32                   HWND;
static constexpr HWND NO_WINDOW = ~0U;
#elif defined(_WIN32)
static constexpr HWND NO_WINDOW = nullptr;
#endif

class GpuTest
{
public:
    virtual ~GpuTest() = default;

    string GetName() { return m_Name; }

    Goldelwalues *GetGoldelwalues();

    virtual bool IsSupported();
    virtual void PrintJsProperties(Tee::Priority pri);

    virtual RC Setup();
    virtual RC Run();
    virtual RC Cleanup();

    virtual JSObject * GetJSObject();

    virtual void SetWindowParams(HINSTANCE hinstance, HWND hWindow);

    Tee::Priority GetVerbosePrintPri() const;

    GpuTestConfiguration* GetTestConfiguration();

    GpuSubdevice *GetBoundGpuSubdevice();
    GpuDevice* GetBoundGpuDevice() const { return nullptr; }
    unsigned GetDispMgrReqs() const { return 0; }

    RC AllocDisplay() { return OK; }

    FancyPicker::FpContext* GetFpContext() { return &m_FpCtx; }

protected:
    void SetName(const char* name) { m_Name = name; }
    MemError& GetMemError(UINT32) { return *m_MemError; }
    virtual RC AddExtraObjects(JSContext* cx, JSObject* obj);
    virtual RC PrintProgressInit(const UINT64 maxIterations) const;
    virtual RC PrintProgressUpdate(const UINT64 lwrrentIteration) const;
    virtual RC EndLoopMLE(UINT64 mleProgress) const;

    GpuTestConfiguration m_TestConfig;
    HINSTANCE m_Hinstance = nullptr;
    HWND m_HWindow = NO_WINDOW;

private:
    Goldelwalues m_Goldelwalues;
    string m_Name = "Test";
    GpuSubdevice m_BoundGpuSubdevice;
    MemError* m_MemError = nullptr;
    FancyPicker::FpContext m_FpCtx = {};
};

struct mglTestContext
{
    RC SetProperties(GpuTestConfiguration*, bool, GpuDevice*, unsigned, unsigned, bool, bool, unsigned)
    {
        return OK;
    }
    bool IsSupported() const
    {
        return true;
    }
    RC Setup()
    {
        return OK;
    }
    RC Cleanup()
    {
        return OK;
    }
};
