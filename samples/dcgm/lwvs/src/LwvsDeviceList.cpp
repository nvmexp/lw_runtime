
#include "LwvsDeviceList.h"
#include "common.h"
#include "lwml.h"
#include "timelib.h"

/*****************************************************************************/
#ifndef MAX
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#endif

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

/*****************************************************************************/
LwvsDevice::LwvsDevice(Plugin *plugin)
{
    /* Initialize to bad values */
    m_lwmlIndex = 0xFFFFFFFF;
    m_plugin = plugin;
    m_lwmlDevice = 0;

    memset(&m_savedState, 0, sizeof(m_savedState));


}

/*****************************************************************************/
LwvsDevice::~LwvsDevice()
{
}

/*****************************************************************************/
unsigned int LwvsDevice::GetLwmlIndex()
{
    return m_lwmlIndex;
}

/*****************************************************************************/
int LwvsDevice::SaveState(lwvs_device_state_t *savedState)
{
    char buf[256];
    lwmlReturn_t lwmlSt;

    buf[255] = 0;

    if(savedState->populated)
        return -1; /* This object is already populated */

    memset(savedState, 0, sizeof(*savedState));
    savedState->populated = 1;

    /* Save application clocks */
    lwmlSt = lwmlDeviceGetApplicationsClock(m_lwmlDevice, LWML_CLOCK_GRAPHICS,
                                            &savedState->appClockGraphics);
    lwmlSt = lwmlDeviceGetApplicationsClock(m_lwmlDevice, LWML_CLOCK_MEM,
                                            &savedState->appClockMemory);
    if (lwmlSt == LWML_SUCCESS)
    {
        savedState->appClocksSupported = 1;
    }
    else if (lwmlSt == LWML_ERROR_NOT_SUPPORTED)
    {
        /* Application clocks unsupported */
        savedState->appClocksSupported = 0;
        savedState->appClockGraphics = 0;
        savedState->appClockGraphics = 0;
    }
    else
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetApplicationsClock");
        snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
        d.AddDetail(buf);
        RecordWarning(d, 0);
    }

    /* Save compute mode */
    lwmlSt = lwmlDeviceGetComputeMode(m_lwmlDevice, &savedState->computeMode);
    if (lwmlSt == LWML_SUCCESS)
    {
        savedState->computeModeSupported = 1;
    }
    else if (lwmlSt == LWML_ERROR_NOT_SUPPORTED)
    {
        savedState->computeModeSupported = 0;
        savedState->computeMode = LWML_COMPUTEMODE_DEFAULT;
    }
    else
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetComputeMode");
        snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
        d.AddDetail(buf);
        RecordWarning(d, 0);
    }

    /* Auto boost */
    lwmlSt = lwmlDeviceGetAutoBoostedClocksEnabled(m_lwmlDevice,
                                                   &savedState->autoBoostEnabled,
                                                   &savedState->autoBoostDefaultEnabled);
    if (lwmlSt == LWML_SUCCESS)
    {
        savedState->autoBoostSupported = 1;
    }
    else if (lwmlSt == LWML_ERROR_NOT_SUPPORTED)
    {
        savedState->autoBoostEnabled = LWML_FEATURE_DISABLED;
        savedState->autoBoostDefaultEnabled = LWML_FEATURE_DISABLED;
    }
    else
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetAutoBoostedClocksEnabled");
        snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
        d.AddDetail(buf);
        RecordWarning(d, 0);
    }

    lwmlSt = lwmlDeviceGetCpuAffinity(m_lwmlDevice, LWVS_DL_AFFINITY_SIZE,
                                      &savedState->idealCpuAffinity[0]);
    if (lwmlSt == LWML_SUCCESS)
    {
        savedState->cpuAffinitySupported = 1;
    }
    else if (lwmlSt == LWML_ERROR_NOT_SUPPORTED)
    {
        savedState->cpuAffinitySupported = 0;
    }
    else
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetCpuAffinity");
        snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
        d.AddDetail(buf);
        RecordWarning(d, 0);
    }
    savedState->cpuAffinityWasSet = 0;

    return 0;
}

/*****************************************************************************/
lwmlDevice_t LwvsDevice::GetLwmlDevice()
{
    return m_lwmlDevice;
}

/*****************************************************************************/
int LwvsDevice::DisableAutoBoostedClocks()
{
    lwmlReturn_t lwmlSt;
    lwmlEnableState_t autoBoostEnabled, defaultAutoBoostEnabled;
    char buf[256] = {0};

    if (!m_lwmlDevice)
        return -1;

    lwmlSt = lwmlDeviceGetAutoBoostedClocksEnabled(m_lwmlDevice, &autoBoostEnabled,
                                                   &defaultAutoBoostEnabled);
    if (lwmlSt == LWML_SUCCESS && autoBoostEnabled == LWML_FEATURE_ENABLED)
    {
        /* Auto boost is enabled. Disable it */
        lwmlSt = lwmlDeviceSetAutoBoostedClocksEnabled(m_lwmlDevice, LWML_FEATURE_DISABLED);
        if (lwmlSt == LWML_SUCCESS)
        {
            RecordInfo("Auto boosted clocks were detected and successfully disabled");
        }
        else if (!m_plugin)
            return -1; /* Can't log to null plugin */
        else
        {
            if (lwmlSt == LWML_ERROR_NO_PERMISSION)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetAutoBoostedClocksEnabled");
                snprintf(buf, sizeof(buf), " could not diable auto boosted clocks '%s'", lwmlErrorString(lwmlSt));
                d.AddDetail(buf);
                RecordWarning(d, 1);
            }
            else
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetAutoBoostedClocksEnabled");
                snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
                d.AddDetail(buf);
                RecordWarning(d, 1);
            }
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
int LwvsDevice::SetMaxApplicationClocks(unsigned int maxMemClockMhz, unsigned int maxGraphicsClockMhz)
{
    unsigned int i;
    lwmlReturn_t lwmlSt;
    char buf[256] = {0};

    unsigned int memClocks[128] = {0};
    unsigned int memClockCount = 128;
    unsigned int useMemClock = 0;
    unsigned int graphicClocks[128] = {0};
    unsigned int graphicClockCount = 128;
    unsigned int useGraphicClock = 0;
    unsigned int lwrrentGraphicClock = 0;
    unsigned int lwrrentMemClock = 0;

    if(!m_lwmlDevice)
        return -1;

    /* Try to maximize application clocks */
    lwmlSt = lwmlDeviceGetSupportedMemoryClocks(m_lwmlDevice, &memClockCount,
                                                &memClocks[0]);
    if(lwmlSt == LWML_SUCCESS)
    {
        /* Maximize clocks within our limits */
        for(i=0; i<memClockCount; i++)
        {
            if(!maxMemClockMhz || memClocks[i] <= maxMemClockMhz)
                useMemClock = MAX(useMemClock, memClocks[i]);
        }

        if(!useMemClock)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NO_MEMORY_CLOCKS, d, maxMemClockMhz, memClockCount);
            RecordWarning(d, 1);
            return -1;
        }

        lwmlSt = lwmlDeviceGetSupportedGraphicsClocks(m_lwmlDevice, useMemClock,
                                                      &graphicClockCount, &graphicClocks[0]);
        if(lwmlSt != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetSupportedGraphicsClocks");
            snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
            d.AddDetail(buf);
            RecordWarning(d, 1);
            return -1;
        }

        for(i=0; i<graphicClockCount; i++)
        {
            if(!maxGraphicsClockMhz || graphicClocks[i] <= maxGraphicsClockMhz)
                useGraphicClock = MAX(useGraphicClock, graphicClocks[i]);
        }

        if(!useGraphicClock)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NO_GRAPHICS_CLOCKS, d, maxGraphicsClockMhz, graphicClockCount, useMemClock);
            RecordWarning(d, 1);
            return -1;
        }

        /* Get current graphics and memory clocks */
        lwmlSt = lwmlDeviceGetApplicationsClock(m_lwmlDevice, LWML_CLOCK_GRAPHICS,
                                                &lwrrentGraphicClock);
        lwmlSt = lwmlDeviceGetApplicationsClock(m_lwmlDevice, LWML_CLOCK_MEM,
                                                &lwrrentMemClock);
        if(lwrrentGraphicClock == useGraphicClock && lwrrentMemClock == useMemClock)
        {
            snprintf(buf, sizeof(buf)-1, "Skipping setting of application clocks for LWML device %u "
                     "since they're already set to graphics %u, memory %u.",
                     m_lwmlIndex, lwrrentGraphicClock, lwrrentMemClock);
            RecordInfo(buf);
        }
        else
        {
            /* Set app clocks to max value */
            lwmlSt = lwmlDeviceSetApplicationsClocks(m_lwmlDevice, useMemClock,
                                                     useGraphicClock);
            if(lwmlSt == LWML_SUCCESS)
            {
                snprintf(buf, sizeof(buf)-1, "Set clocks for LWML device %u to graphics %u, mem %u",
                         m_lwmlIndex, useGraphicClock, useMemClock);
                RecordInfo(buf);
            }
            else if(lwmlSt == LWML_ERROR_NO_PERMISSION)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetAppliationClocks");
                snprintf(buf, sizeof(buf), " don't have permission: '%s'. Use sudo lwpu-smi -i "
                         "%u -acp 0 to enable changing application clocks as non-root. Persistence "
                         "mode must be enabled on linux for this setting to have an effect. Use "
                         "sudo lwpu-smi -i %u -pm 1 to enable persistence mode.",
                         lwmlErrorString(lwmlSt), m_lwmlIndex, m_lwmlIndex);
                d.AddDetail(buf);
                RecordWarning(d, 1);
                return -1;
            }
            else
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetAppliationClocks");
                snprintf(buf, sizeof(buf), "'%s' when setting mem %u and graphic %u", lwmlErrorString(lwmlSt),
                         useMemClock, useGraphicClock);
                d.AddDetail(buf);
                RecordWarning(d, 1);
                return -1;
            }
        }
    }

    return 0;
}

/*****************************************************************************/
int LwvsDevice::SetCpuAffinity(void)
{
    char buf[256];
    lwmlReturn_t lwmlSt;

    buf[255] = 0;

    /* Try to set the affinity of the device */
    lwmlSt = lwmlDeviceSetCpuAffinity(m_lwmlDevice);
    if(lwmlSt != LWML_SUCCESS)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetCpuAffinity");
        snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
        d.AddDetail(buf);
        RecordWarning(d, 0);
        return -1;
    }
    else
        m_savedState.cpuAffinityWasSet = 1;

    return 0;
}

/*****************************************************************************/
int LwvsDevice::RestoreState(void)
{
    lwmlReturn_t lwmlSt;
    lwvs_device_state_t afterState;
    char buf[256];
    int NstatesRestored = 0; /* How many device states did we restore? */

    buf[255] = 0;

    /* Record the current state of the device for comparison */
    afterState.populated = 0;
    SaveState(&afterState);

    /* Do the easy check to see if anything has changed */
    if(!memcmp(&m_savedState, &afterState, sizeof(m_savedState)))
    {
        /* Nothing to restore */
        return 0;
    }

    /* Possibly restore application clocks. Use fuzzy logic to see if they've changed since rounding
       errors can occur on products like Titan X */
    int memClockDiff = ABS(((int)m_savedState.appClockMemory) - ((int)afterState.appClockMemory));
    int graphicsClockDiff = ABS(((int)m_savedState.appClockGraphics - ((int)afterState.appClockGraphics)));

    if(m_savedState.appClocksSupported &&
       (memClockDiff > 1 || graphicsClockDiff > 1))
    {
        lwmlSt = lwmlDeviceSetApplicationsClocks(m_lwmlDevice, m_savedState.appClockMemory,
                                                     m_savedState.appClockGraphics);
        if(lwmlSt != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetApplicationsClocks");
            snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
            d.AddDetail(buf);
            RecordWarning(d, 0);
        }
        else
            NstatesRestored++;
    }

    /* Auto boost clocks */
    if(m_savedState.autoBoostSupported &&
       m_savedState.autoBoostEnabled != afterState.autoBoostEnabled)
    {
        lwmlSt = lwmlDeviceSetAutoBoostedClocksEnabled(m_lwmlDevice, m_savedState.autoBoostEnabled);
        if(lwmlSt == LWML_ERROR_NO_PERMISSION)
        {
            /* This can happen normally if you've set auto boost for your PID to one direction and try to set it to the other.
             * RM returns this status code to let you know you have conflicting hints */
            PRINT_DEBUG("", "lwmlDeviceSetAutoBoostedClocksEnabled returned LWML_ERROR_NO_PERMISSION.");
        }
        else if(lwmlSt != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetAutoBoostedClocksEnabled");
            snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
            d.AddDetail(buf);
            RecordWarning(d, 0);
        }
        else
            NstatesRestored++;
    }
    if(m_savedState.autoBoostSupported &&
       m_savedState.autoBoostDefaultEnabled!= afterState.autoBoostDefaultEnabled)
    {
        lwmlSt = lwmlDeviceSetDefaultAutoBoostedClocksEnabled(m_lwmlDevice, m_savedState.autoBoostDefaultEnabled, 0);
        if(lwmlSt != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetDefaultAutoBoostedClocksEnabled");
            snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
            d.AddDetail(buf);
            RecordWarning(d, 0);
        }
        else
            NstatesRestored++;
    }

    if(m_savedState.computeModeSupported && m_savedState.computeMode != afterState.computeMode)
    {
        lwmlSt = lwmlDeviceSetComputeMode(m_lwmlDevice, m_savedState.computeMode);
        if(lwmlSt != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceSetComputeMode");
            snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
            d.AddDetail(buf);
            RecordWarning(d, 0);
        }
        else
            NstatesRestored++;
    }

    if(m_savedState.cpuAffinitySupported && m_savedState.cpuAffinityWasSet)
    {
        /* All we can really do is clear whatever was set. Hopefully that puts
         * cpu affinity back to what it was previously
         */
        lwmlSt = lwmlDeviceClearCpuAffinity(m_lwmlDevice);
        if(lwmlSt != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceClearCpuAffinity");
            snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
            d.AddDetail(buf);
            RecordWarning(d, 0);
        }
        else
            NstatesRestored++;
    }

    if(NstatesRestored > 0)
        return 1;
    else
    {
        PRINT_ERROR("", "Unexpected no states restored. Should have been shortlwt");
        return -1;
    }
}

/*****************************************************************************/
int LwvsDevice::Init(int lwmlDeviceIndex)
{
    char errorText[256];
    lwmlReturn_t lwmlSt;
    char buildPciStr[32] = {0};

    m_lwmlIndex = lwmlDeviceIndex;
    errorText[255] = 0;

    lwmlInit();

    lwmlSt = lwmlDeviceGetHandleByIndex(m_lwmlIndex, &m_lwmlDevice);
    if(lwmlSt != LWML_SUCCESS)
    {
        char buf[256];
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetHandleByIndex");
        snprintf(buf, sizeof(buf), "'%s'", lwmlErrorString(lwmlSt));
        d.AddDetail(buf);
        RecordWarning(d, 1);
        return -1;
    }

    /* Save initial state */
    m_savedState.populated = 0;
    SaveState(&m_savedState);

    return 0;
}

/*****************************************************************************/
void LwvsDevice::RecordInfo(char *logText)
{
    if (m_plugin)
    {
        m_plugin->AddInfo(logText);
    }
    else
    {
        PRINT_INFO("%s", "%s", logText);
    }
}

/*****************************************************************************/
void LwvsDevice::RecordWarning(const DcgmError &d, int failPlugin)
{
    if (m_plugin)
    {
        m_plugin->AddErrorForGpu(m_lwmlIndex, d);
        if (failPlugin)
        {
            m_plugin->SetResultForGpu(m_lwmlIndex, LWVS_RESULT_FAIL);
        }
    }
    else
    {
        PRINT_WARNING("%s", "%s", d.GetMessage().c_str());
        if (failPlugin)
        {
            throw std::runtime_error(d.GetMessage().c_str());
        }
    }
}

/*****************************************************************************/
int LwvsDevice::HasGomLowDp(void)
{
    lwmlGpuOperationMode_t current, pending;
    char buf[256];
    buf[255] = 0;

    lwmlReturn_t lwmlSt = lwmlDeviceGetGpuOperationMode(m_lwmlDevice, &current, &pending);
    if(lwmlSt == LWML_ERROR_NOT_SUPPORTED)
        return 0;
    else if(lwmlSt != LWML_SUCCESS)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetGpuOperationMode");
        snprintf(buf, sizeof(buf), "'%s' for GPU %u", lwmlErrorString(lwmlSt), m_lwmlIndex);
        d.AddDetail(buf);
        RecordWarning(d, 0);
        return -1;
    }

    /* Successfully read GOM. Return appropriate state */
    if(current == LWML_GOM_LOW_DP)
        return 1;
    else
        return 0;

}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
LwvsDeviceList::LwvsDeviceList(Plugin *plugin)
{
    m_plugin = plugin;
}

/*****************************************************************************/
LwvsDeviceList::~LwvsDeviceList(void)
{
    std::vector<LwvsDevice *>::iterator it;
    LwvsDevice *device;

    /* Free all of the devices */
    for(it = m_devices.begin(); it != m_devices.end(); it++)
    {
        device = *it;

        delete(device);
    }

    m_devices.clear();
}

/*****************************************************************************/
int LwvsDeviceList::Init(std::vector<unsigned int>lwmlIndexes)
{
    int i, st;
    unsigned int lwmlIndex;
    LwvsDevice *lwvsDevice;
    char buf[256];

    buf[255] = 0;

    if(m_devices.size() > 0)
    {
        PRINT_ERROR("", "m_devices already initialized.");
        return -1;
    }

    for(i = 0; i < (int)lwmlIndexes.size(); i++)
    {
        lwmlIndex = lwmlIndexes[i];

        lwvsDevice = new LwvsDevice(m_plugin);
        st = lwvsDevice->Init(lwmlIndex);
        if(st)
        {
            DcgmError d;
            snprintf(buf, sizeof(buf)-1, "Got error %d while initializing LwvsDevice index %u",
                     st, lwmlIndex);
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, buf);
            RecordWarning(d, 1);
            delete(lwvsDevice);
            lwvsDevice = 0;
        }

        m_devices.push_back(lwvsDevice);
    }

    return 0;
}

/*****************************************************************************/
void LwvsDeviceList::RecordInfo(char *logText)
{
    if(m_plugin)
        m_plugin->AddInfo(logText);
    else
        PRINT_INFO("%s", "%s", logText);
}

/*****************************************************************************/
void LwvsDeviceList::RecordWarning(const DcgmError &d, int failPlugin)
{
    if (m_plugin)
    {
        m_plugin->AddError(d);
        if (failPlugin)
        {
            m_plugin->SetResult(LWVS_RESULT_FAIL);
        }
    }
    else
    {
        PRINT_WARNING("%s", "%s", d.GetMessage().c_str());
        if (failPlugin)
        {
            throw std::runtime_error(d.GetMessage().c_str());
        }
    }
}

/*****************************************************************************/
int LwvsDeviceList::RestoreState(int failOnRestore)
{
    int i, st;
    LwvsDevice *lwvsDevice;
    std::vector<unsigned int>changedLwmlIndexes;
    char buf[256];

    buf[255] = 0;

    for(i=0; i < (int)m_devices.size(); i++)
    {
        lwvsDevice = m_devices[i];
        st = lwvsDevice->RestoreState();
        if(!st)
            continue; /* Nothing changed. Great */
        else if(failOnRestore)
        {
            /* Save for later complaint. Restore rest of devices for now */
            changedLwmlIndexes.push_back(lwvsDevice->GetLwmlIndex());
        }
    }

    if(changedLwmlIndexes.size() > 0 && failOnRestore)
    {
        std::stringstream ss;
        unsigned int lwmlIndex;
        DcgmError d;

        for(i = 0; i < (int)changedLwmlIndexes.size(); i++)
        {
            if(i > 0)
                ss << ",";
            ss << " " << changedLwmlIndexes[i];
        }
        
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HAD_TO_RESTORE_STATE, d, ss.str().c_str());

        RecordWarning(d, 1);
        return 1;
    }

    return 0;
}

/*****************************************************************************/
int LwvsDeviceList::DoAnyGpusHaveGomLowDp(void)
{
    int deviceIndex;

    for (deviceIndex = 0; deviceIndex < (int)m_devices.size(); deviceIndex++)
    {
        LwvsDevice *lwvsDevice = m_devices[deviceIndex];

        /* Does this device have low DP mode? */
        if(lwvsDevice->HasGomLowDp())
            return 1;
    }

    return 0; /* No devices have low DP mode */
}

/*****************************************************************************/

