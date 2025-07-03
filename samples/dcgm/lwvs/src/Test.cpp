#include "Test.h"
#include "Plugin.h"
#include "LwidiaValidationSuite.h"
#include <iostream>
#include <vector>
#include "TestParameters.h"
#include "LwvsDeviceList.h"
#include "PluginStrings.h"
#include "common.h"
#include "DcgmError.h"

/* Static constants for Test class */
const lwvsPluginGpuResults_t Test::m_emptyGpuResults;
const lwvsPluginGpuMessages_t Test::m_emptyGpuMessages;
const std::vector<string> Test::m_emptyMessages;

/*****************************************************************************/
Test::Test(Plugin *plugin) : m_infoStruct(), m_plugin(plugin), m_argMap(), m_skipTest(false)
{
    if (m_plugin == NULL)
    {
        m_skipTest = true;
        m_infoStruct.selfParallel = true;
        return;
    }
    m_infoStruct = m_plugin->GetInfoStruct();

    if (!m_infoStruct.defaultTestParameters)
    {
        std::cerr << "Plugin \"" << m_infoStruct.name << "\" missing default test parameters" << std::endl;
        m_skipTest = true;
    }
}

/*****************************************************************************/
Test::~Test()
{
    if (m_plugin)
    {
        delete m_plugin;
    }
}

/*****************************************************************************/
void Test::go(TestParameters * testParameters)
{
    if (!m_skipTest)
    {
        try
        {
            m_plugin->Go(testParameters);
        }
        catch (std::exception &e)
        {
            getOut(e.what());
        }
    }
}

/*****************************************************************************/
void Test::go(TestParameters * testParameters, Gpu * gpu)
{
    if (!m_skipTest)
    {
        try
        {
            m_plugin->Go(testParameters, gpu->getDeviceIndex(Gpu::LWVS_GPUENUM_LWML));
        }
        catch (std::exception &e)
        {
            getOut(e.what());
        }
    }
}

/*****************************************************************************/
void Test::go(TestParameters * testParameters, std::vector<Gpu *> gpus)
{
    std::vector<Gpu *>::iterator it;
    std::vector<unsigned int> gpuIndexes;
    int st;

    if (m_skipTest)
    {
        return;
    }

    for (it = gpus.begin(); it != gpus.end(); it++)
    {
        gpuIndexes.push_back((*it)->getDeviceIndex(Gpu::LWVS_GPUENUM_LWML));
    }
    // put this in a try bracket and catch exceptions but all check return codes

    /* Save GPU state for restoring after the plugin runs */
    LwvsDeviceList *lwvsDeviceList = new LwvsDeviceList(0);
    st = lwvsDeviceList->Init(gpuIndexes);
    if (st)
    {
        getOut(std::string("Unable to initialize LWVS device list"));
    }

    /* See if this plugin needs to be skipped if GOM mode is LOW_DP */
    int pluginRunsWithLowDpGom = testParameters->GetBoolFromString(PS_RUN_IF_GOM_ENABLED);
    if (!pluginRunsWithLowDpGom)
    {
        int anyHaveLowDpGom = lwvsDeviceList->DoAnyGpusHaveGomLowDp();
        if (anyHaveLowDpGom)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GPU_OP_MODE, d);
            m_plugin->AddError(d);
            m_plugin->SetResult(LWVS_RESULT_SKIP);
            lwvsDeviceList->RestoreState(0);
            delete(lwvsDeviceList);
            lwvsDeviceList = 0;
            return;
        }
    }

#if 0 /* Don't do a wait for idle. Some lwstomers allow their GPUs to run hot
        as long as they aren't near the slowdown temp */
    
    /* Wait for the GPUs to reach an idle state before testing */
    st = lwvsDeviceList->WaitForIdle(-1.0, -1.0, -1.0);
    if(st == 1)
    {
        getOut("Timed out waiting for all GPUs to return to an idle state.");
    }
    else if(st < 0)
        getOut("Got an error while waiting for all GPUs to return to an idle state.");
    /* st == 0 falls through, which means all GPUs are idle */
#endif

    /* Start the test */
    try
    {
        m_plugin->Go(testParameters, gpuIndexes);
    }
    catch (std::exception &e)
    {
        /* Restore state and throw the exception higher */
        lwvsDeviceList->RestoreState(0);
        delete(lwvsDeviceList);
        lwvsDeviceList = 0;
        getOut(e.what());
    }

    /* Restore state and throw an exception if something wasn't restored by
     * the plugin */
    lwvsDeviceList->RestoreState(1);
    delete(lwvsDeviceList);
    lwvsDeviceList = 0;
}

/*****************************************************************************/
void Test::getOut(std::string error)
{
    // Create error message for the exception
    std::string errMsg = "\"" + m_infoStruct.name + "\" test: " + error;
    lwvsCommon.mainReturnCode = MAIN_RET_ERROR; /* Return error code to console */
    throw std::runtime_error(errMsg);
}

/*****************************************************************************/
std::string Test::getFullLogFileName()
{
    std::string retStr = lwvsCommon.m_statsPath;
    
    /* Make sure path ends in a / */
    if(retStr.size() > 0 && retStr.at(retStr.size()-1) != '/')
        retStr += "/";

    /* Add the base filename */
    retStr += lwvsCommon.logFile;

    std::string logFileTag = getLogFileTag();

    if(logFileTag.size() > 0)
    {
        retStr += "_";
        retStr += logFileTag;
    }

    switch(lwvsCommon.logFileType)
    {
        default: //Deliberate fall-through
        case LWVS_LOGFILE_TYPE_JSON:
            retStr += ".json";
            break;
        case LWVS_LOGFILE_TYPE_TEXT:
            retStr += ".txt";
            break;
        case LWVS_LOGFILE_TYPE_BINARY:
            retStr += ".stats";
            break;
    }

    return retStr;
}

/*****************************************************************************/


