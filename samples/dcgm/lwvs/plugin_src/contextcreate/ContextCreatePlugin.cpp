#include "ContextCreatePlugin.h"
#include "PluginStrings.h"
#include "ContextCreate.h"

ContextCreatePlugin::ContextCreatePlugin()
{
    TestParameters *tp;
    m_infoStruct.name = CTXCREATE_PLUGIN_NAME;
    m_infoStruct.shortDescription = "This plugin will attempt to create a LWCA context.";
    m_infoStruct.testGroups = "";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = CTXCREATE_PLUGIN_LF_NAME;

    // Populate default test parameters
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    tp->AddString(CTXCREATE_IGNORE_EXCLUSIVE, "False");
    tp->AddString(CTXCREATE_IS_ALLOWED, "True");
    m_infoStruct.defaultTestParameters = tp;
}

void ContextCreatePlugin::Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    int st;
    InitializeForGpuList(gpuList);

    if (!testParameters->GetBoolFromString(CTXCREATE_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, CTXCREATE_PLUGIN_NAME);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
        return;
    }

    ContextCreate cc(testParameters, this);

    st = cc.Run(gpuList);
    if (!st)
    {
        SetResult(LWVS_RESULT_PASS);
    }
    else if (main_should_stop)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
        AddError(d);
        SetResult(LWVS_RESULT_SKIP);
    }
    else if (st == CONTEXT_CREATE_SKIP)
    {
        SetResult(LWVS_RESULT_SKIP);
    }
    else
    {
        SetResult(LWVS_RESULT_FAIL);
    }

    return;
}

extern "C"
{
    Plugin *maker()
    {
        return new ContextCreatePlugin;
    }

    class proxy
    {
    public:
        proxy()
        {
            factory[CTXCREATE_PLUGIN_WL_NAME] = maker;
        }
    };
    proxy p;
}
