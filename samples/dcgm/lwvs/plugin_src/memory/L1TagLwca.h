#ifndef L1TAGLWDA_H
#define L1TAGLWDA_H

#include <stdint.h>
#include "L1LwdaUtils.h"
#include "memory_plugin.h"
#include <lwca.h>

/*****************************************************************************/
/* Parameters and error reporting structures for the test kernel */

#define L1_LINE_SIZE_BYTES 128
struct L1TagParams
{
    device_ptr data;
    device_ptr errorCountPtr;
    device_ptr errorLogPtr;
    uint64_t   sizeBytes;
    uint64_t   iterations;
    uint32_t   errorLogLen;
    uint32_t   randSeed;
};

enum TestStage
{
    PreLoad = 0,
    RandomLoad = 1
};

struct L1TagError
{
    uint32_t testStage;
    uint16_t decodedOff;
    uint16_t expectedOff;
    uint64_t iteration;
    uint32_t innerLoop;
    uint32_t smid;
    uint32_t warpid;
    uint32_t laneid;
};



/*****************************************************************************/
/* Class to wrap the LWCA implementation portion of the L1tag plugin */

class L1TagLwda
{
public:

    L1TagLwda(Memory *plugin, TestParameters *tp, mem_globals_p memGlobals)
        : m_plugin(plugin)
        , m_testParameters(tp)
        , m_lwvsDevice(memGlobals->lwvsDevice)
        , m_lwmlDevice(memGlobals->lwmlDevice)
        , m_lwDevice(memGlobals->lwDevice)
        , m_lwCtx(memGlobals->lwCtx)
        , m_lwMod(NULL)
        , m_hostErrorLog(NULL)
        , m_l1Data((LWdeviceptr)NULL)
        , m_devMiscompareCount((LWdeviceptr)NULL)
        , m_devErrorLog((LWdeviceptr)NULL)
    {
    }

    lwvsPluginResult_t TestMain(unsigned int lwmlGpuIndex);

private:

    int                 Setup(void);
    void                Cleanup(void);
    lwvsPluginResult_t  RunTest(void);

    int                 AllocDeviceMem ( int size, LWdeviceptr *ptr);
    int                 AllocHostMem ( int size, void **ptr);
    lwvsPluginResult_t  GetMaxL1CacheSizePerSM(uint32_t &l1PerSMBytes);
    int                 GetLwDevice( LWdevice *lwDevice, std::stringstream &error);
    void                LogLwDeviceLookupFail(std::stringstream &error);
    lwvsPluginResult_t  LogLwdaFail(const char *msg, const char *lwdaFuncS, LWresult lwRes );

    unsigned int    m_gpuIndex;
    Memory         *m_plugin;
    TestParameters *m_testParameters;

    LwvsDevice     *m_lwvsDevice;
    lwmlDevice_t    m_lwmlDevice;

    LWdevice        m_lwDevice; // not owned here
    LWcontext       m_lwCtx;    // not owned here
    LWmodule        m_lwMod;

    L1TagError     *m_hostErrorLog;
    LWdeviceptr     m_l1Data;
    LWdeviceptr     m_devMiscompareCount;
    LWdeviceptr     m_devErrorLog;

    // Test parameters
    uint32_t    m_runtimeMs;
    uint64_t    m_testLoops;
    uint64_t    m_innerIterations;
    uint32_t    m_errorLogLen;
    bool        m_dumpMiscompares;

    L1TagParams m_kernelParams;
};

#endif

