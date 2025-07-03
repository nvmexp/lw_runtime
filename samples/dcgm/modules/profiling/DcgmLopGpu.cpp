#include "DcgmLopGpu.h"
#include "lwperf_target.h"
#include "lwperf_host.h"
#include "lwperf_dcgm_target_priv.h"
#include "lwperf_dcgm_host_priv.h"
#include "lwperf_dcgm_target_priv.h"
#include "logging.h"

/*****************************************************************************/
DcgmLopGpu::DcgmLopGpu(int pwDeviceIndex)
{

    m_pwDeviceIndex = pwDeviceIndex;
    m_sessionIsActive = false;
    m_chipName = NULL;
    m_startedSampling = false;
}

/*****************************************************************************/
DcgmLopGpu::~DcgmLopGpu()
{
    StopSampling();
    EndSession();
    FreeConfigs();
}

/*****************************************************************************/
void DcgmLopGpu::FreeConfigs(void)
{
    for(size_t i = 0; i < m_configs.size(); i++)
    {
        if(m_configs[i])
        {
            delete(m_configs[i]);
            m_configs[i] = NULL;
        }
    }
    m_configs.clear();
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::Init(void)
{
    const char* pDeviceName = NULL;
    const char* pChipName = NULL;

    LWPA_Status lwpaStatus = LWPA_Device_GetNames(m_pwDeviceIndex, &pDeviceName, &pChipName);
    if(lwpaStatus != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d", "Got status %d from LWPA_Device_GetNames", lwpaStatus);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    
    m_chipName = pChipName;

    m_initialized = true;

    PRINT_DEBUG("%u", "Init was successful for deviceIndex %u.", m_pwDeviceIndex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::SetConfig(int configIndex, bool triggerDiscard)
{
    dcgmReturn_t dcgmReturn;
    
    if(configIndex < 0 || configIndex >= (int)m_configs.size())
    {
        PRINT_ERROR("%d", "configIndex %d is out of range.", configIndex);
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn = m_configs[configIndex]->SetConfig();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;
    
    m_activeConfigIndex = configIndex;

    PRINT_DEBUG("%d", "Changed active config to %d", configIndex);

    /* Clear out our counters after we change configs */
    if(triggerDiscard)
    {
        dcgmReturn = TriggerDiscard();
        if (dcgmReturn != DCGM_ST_OK)
            return dcgmReturn;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::InitializeWithMetrics(std::vector<const char *> metricNames[DLG_MAX_METRIC_GROUPS], 
                                               std::vector<unsigned int> metricIds[DLG_MAX_METRIC_GROUPS],
                                               int numMetricGroups)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if(numMetricGroups < 1 || numMetricGroups > DLG_MAX_METRIC_GROUPS)
        return DCGM_ST_BADPARAM;
    
    /* Free any previous configs */
    FreeConfigs();

    for(int i = 0; i < numMetricGroups; i++)
    {
        DcgmLopConfig *config = new DcgmLopConfig(m_pwDeviceIndex, i, m_chipName);
        if(!config)
        {
            FreeConfigs();
            return DCGM_ST_MEMORY;
        }
    
        dcgmReturn = config->Init();
        if(dcgmReturn != DCGM_ST_OK)
        {
            delete(config);
            FreeConfigs();
            return dcgmReturn;
        }

        dcgmReturn = config->InitializeWithMetrics(metricNames[i], metricIds[i]);
        if(dcgmReturn != DCGM_ST_OK)
        {
            delete(config);
            FreeConfigs();
            return dcgmReturn;
        }

        m_configs.push_back(config);
    }

    PRINT_DEBUG("%d", "Successfully added %d metric groups.", numMetricGroups);
    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmLopGpu::IsMetricNameValid(const char *metricName)
{
    if(!m_initialized)
        return false;

    /* Get attributes for a metric. LWPA error = doesn't exist */

    return true;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::TriggerDiscard()
{
    LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params triggerDiscardParams;
    LWPA_Status status;
    
    memset(&triggerDiscardParams, 0, sizeof(triggerDiscardParams));
    triggerDiscardParams.structSize = LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params_STRUCT_SIZE;
    triggerDiscardParams.deviceIndex = m_pwDeviceIndex;
    status = LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard(&triggerDiscardParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::GetSamples(std::vector<DcgmLopSample_t> &samples)
{
    dcgmReturn_t dcgmReturn;
    
    if(m_activeConfigIndex < 0 || m_activeConfigIndex >= (int)m_configs.size())
    {
        PRINT_ERROR("%d", "m_activeConfigIndex %d is out of range.", m_activeConfigIndex);
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn = m_configs[m_activeConfigIndex]->GetSamples(samples);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;
    
    /* Swap configs if we have > 1 config */
    if(m_configs.size() < 2)
        return DCGM_ST_OK;
    
    /* Round robin through the configs */
    int newActiveConfigIndex = (m_activeConfigIndex + 1) % m_configs.size();
    dcgmReturn = SetConfig(newActiveConfigIndex, true);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::EnableMetrics()
{
    dcgmReturn_t dcgmReturn;
    
    /* Return error if there is no configuration cached (InitializeWithMetrics hasn't been called) */
    if (m_configs.size() < 1)
    {
        PRINT_ERROR("", "m_configs size is 0. Have you called InitializeWithMetrics?");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmReturn = BeginSession();
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = SetConfig(0, false);
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = StartSampling();
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;
    
    /* Per ajayc - trigger a discard after StartSampling() to make sure our first 
       counter read doesn't read garbage from previous samplings */
    dcgmReturn = TriggerDiscard();
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::DisableMetrics()
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = StopSampling();
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = EndSession();
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::BeginSession()
{
    if(!m_initialized)
        return DCGM_ST_UNINITIALIZED;
    if(m_sessionIsActive)
        return DCGM_ST_OK;
    
    LWPW_DCGM_PeriodicSampler_BeginSession_Params beginSessionParams;
    memset(&beginSessionParams, 0, sizeof(beginSessionParams));
    
    beginSessionParams.structSize = LWPW_DCGM_PeriodicSampler_BeginSession_Params_STRUCT_SIZE;
    beginSessionParams.deviceIndex = m_pwDeviceIndex;
    beginSessionParams.triggerSource = LWPW_DCGM_PeriodicSampler_TriggerSource_CPUTrigger;
    beginSessionParams.maxSampleNameLength = DLG_MAX_SAMPLE_NAME_LENGTH; 
    beginSessionParams.maxCPUTriggers = DLG_MAX_METRIC_GROUPS * 2; /* Recommended by ajayc and grsmith on 8/26/2019. 
                                                                      DLG_MAX_METRIC_GROUPS = max number of passes (swapped)
                                                                      *2 = handles overflow of GPU writing a single sample */
    const LWPA_Status status = LWPW_DCGM_PeriodicSampler_BeginSession(&beginSessionParams);
    if (status == LWPA_STATUS_INSUFFICIENT_PRIVILEGE)
    {
        PRINT_ERROR("%u", "Got status INSUFFICIENT_PRIVILEGE from LWPW_DCGM_PeriodicSampler_BeginSession() on deviceIndex %u", 
                    m_pwDeviceIndex);
        return DCGM_ST_REQUIRES_ROOT;
    }
    else if (status == LWPA_STATUS_INSUFFICIENT_DRIVER_VERSION)
    {
        PRINT_ERROR("%u", "Got status INSUFFICIENT_DRIVER_VERSION from LWPW_DCGM_PeriodicSampler_BeginSession() on deviceIndex %u", 
                    m_pwDeviceIndex);
        return DCGM_ST_INSUFFICIENT_DRIVER_VERSION;
    }
    else if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%u %u", "Got status %u from LWPW_DCGM_PeriodicSampler_BeginSession() on deviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    PRINT_DEBUG("%u", "LWPW_DCGM_PeriodicSampler_BeginSession was successful on device %u", m_pwDeviceIndex);
    m_sessionIsActive = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::EndSession()
{
    if(!m_initialized)
        return DCGM_ST_UNINITIALIZED;
    if(!m_sessionIsActive)
        return DCGM_ST_OK;
    
    LWPW_DCGM_PeriodicSampler_EndSession_Params endSessionParams;
    memset(&endSessionParams, 0, sizeof(endSessionParams));
    
    endSessionParams.structSize = LWPW_DCGM_PeriodicSampler_EndSession_Params_STRUCT_SIZE;
    endSessionParams.deviceIndex = m_pwDeviceIndex;
    const LWPA_Status status = LWPW_DCGM_PeriodicSampler_EndSession(&endSessionParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%u %u", "Got status %u from LWPW_DCGM_PeriodicSampler_EndSession() on deviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    
    PRINT_DEBUG("%u", "EndSession was successful for deviceIndex %u.", m_pwDeviceIndex);
    m_sessionIsActive = false;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::StartSampling(void)
{
    LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params startSamplingParams;
    
    memset(&startSamplingParams, 0, sizeof(startSamplingParams));
    startSamplingParams.structSize = LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params_STRUCT_SIZE;

    startSamplingParams.deviceIndex = m_pwDeviceIndex;
    
    const LWPA_Status status = LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling(&startSamplingParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    PRINT_DEBUG("%u", "StartSampling was successful for deviceIndex %u.", m_pwDeviceIndex);
    m_startedSampling = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopGpu::StopSampling(void)
{
    LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params stopSamplingParams;

    if(!m_startedSampling)
    {
        PRINT_DEBUG("%u", "Ignoring StopSampling for pwDeviceIndex %u since we haven't started sampling.", m_pwDeviceIndex);
        return DCGM_ST_OK;
    }

    memset(&stopSamplingParams, 0, sizeof(stopSamplingParams));
    stopSamplingParams.structSize = LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params_STRUCT_SIZE;
    stopSamplingParams.deviceIndex = m_pwDeviceIndex;
    
    const LWPA_Status status = LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling(&stopSamplingParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    PRINT_DEBUG("%u", "StopSampling was successful for deviceIndex %u.", m_pwDeviceIndex);
    m_startedSampling = false;
    return DCGM_ST_OK;
}

/*****************************************************************************/
