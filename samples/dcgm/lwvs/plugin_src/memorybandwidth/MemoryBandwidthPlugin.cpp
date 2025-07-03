
#include "MemoryBandwidthPlugin.h"
#include "PluginStrings.h"
#include "MemoryBandwidth.h"

MemoryBandwidthPlugin::MemoryBandwidthPlugin()
{
    TestParameters *tp;

    m_infoStruct.name = MEMBW_PLUGIN_NAME;
    m_infoStruct.shortDescription = "This plugin will test the memory bandwidth of a list of GPUs.";
    m_infoStruct.testGroups = "Perf";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = MEMBW_PLUGIN_LF_NAME;

    /* Populate default test parameters */
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True"); /* This parameter needs to at least be bootstrapped for the framework */
    tp->AddDouble(MEMBW_STR_MINIMUM_BANDWIDTH, 100.0, 1.0, 100000.0);
    tp->AddString(MEMBW_STR_IS_ALLOWED, "False");
    tp->AddDouble(MEMBW_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK, 0.0, DCGM_FP64_BLANK);

    m_infoStruct.defaultTestParameters = tp;
}

void MemoryBandwidthPlugin::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    int st;
    MemoryBandwidth *membw = 0;
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(MEMBW_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, MEMBW_PLUGIN_NAME);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }

    membw = new MemoryBandwidth(testParameters, this);

    st = membw->Run(gpuList);
    if (main_should_stop)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
    }
    else if (st)
    {
        // Fatal error in plugin or test could not be initialized
        SetResult(LWVS_RESULT_FAIL);
    }
}

extern "C" {
    Plugin *maker() {
        return new MemoryBandwidthPlugin;
    }
    class proxy {
    public:
        proxy()
        {
            factory["memory bandwidth"] = maker;
        }
    };
    proxy p;
}
