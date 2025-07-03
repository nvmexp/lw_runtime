#include "Memory_wrapper.h"
#include "memory_plugin.h"
#include "PluginStrings.h"

/*****************************************************************************/
Memory::Memory()
{
    m_infoStruct.name = "Memory";
    m_infoStruct.shortDescription = "This plugin will test the memory of a given GPU.";
    m_infoStruct.testGroups = "";
    m_infoStruct.selfParallel = false;
    m_infoStruct.logFileTag = MEMORY_PLUGIN_LF_NAME;

    TestParameters *tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp->AddString(MEMORY_STR_IS_ALLOWED, "False");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "False");
    tp->AddDouble(MEMORY_L1TAG_STR_TEST_DURATION, 1.0, 0.0, 10800.0);
    tp->AddDouble(MEMORY_L1TAG_STR_TEST_LOOPS, 0, 0, 1000000);
    tp->AddDouble(MEMORY_L1TAG_STR_INNER_ITERATIONS, 1024, 1024, 16384);
    tp->AddDouble(MEMORY_L1TAG_STR_ERROR_LOG_LEN, 8192, 8192, 32768);
    tp->AddString(MEMORY_L1TAG_STR_DUMP_MISCOMPARES, "True" );
    m_infoStruct.defaultTestParameters = tp;
}


/*****************************************************************************/
void Memory::Go(TestParameters *testParameters, unsigned int gpu)
{
    std::vector<unsigned int> gpuList(1, gpu);
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(MEMORY_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, "Memory");
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }
    main_entry(gpu, this, testParameters);
}

/*****************************************************************************/
extern "C" {
    Plugin *maker() {
        return new Memory;
    }
    class proxy {
    public:
        proxy()
        {
            factory["Memory"] = maker;
        }
    };    
    proxy p;
}                                            

