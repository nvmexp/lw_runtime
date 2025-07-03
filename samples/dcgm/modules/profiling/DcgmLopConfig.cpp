
#include "DcgmLopConfig.h"
#include "logging.h"
#include "lwperf_dcgm_host_priv.h"
#include <math.h> //isnan()

/*****************************************************************************/
DcgmLopConfig::DcgmLopConfig(int pwDeviceIndex, int configIndex, const char *chipName)
{

    m_pwDeviceIndex = pwDeviceIndex;
    m_configIndex = configIndex;
    m_chipName = chipName;
    m_metricsContext = NULL;
    m_rawMetricsConfig = NULL;
    m_counterDataBuilder = NULL;
    m_alwaysWriteCounterData = false;
    m_counterDataIndex = 0;

    char value[16] = {0};
    if(!lwosGetElw("__DCGM_ALWAYS_WRITE_COUNTER_DATA", value, sizeof(value)) && value[0] == '1')
    {
        PRINT_INFO("", "__DCGM_ALWAYS_WRITE_COUNTER_DATA was set. enabling m_alwaysWriteCounterData");
        m_alwaysWriteCounterData = true;
    }
    else
        PRINT_INFO("", "__DCGM_ALWAYS_WRITE_COUNTER_DATA was not set or 0");   
}

/*****************************************************************************/
DcgmLopConfig::~DcgmLopConfig()
{
    DestroyRawMetricsConfig();
    DestroyMetricsContext();
    DestroyCounterDataBuilder();
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::Init(void)
{
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmLopConfig::MakeCounterDataImageOptions(LWPW_DCGM_PeriodicSampler_CounterDataImageOptions *options)
{
    if(!options)
        return;

    memset(options, 0, sizeof(*options));
    options->structSize = LWPW_DCGM_PeriodicSampler_CounterDataImageOptions_STRUCT_SIZE;
    options->pCounterDataPrefix = &m_counterDataImagePrefix[0];
    options->counterDataPrefixSize = m_counterDataImagePrefix.size();
    options->maxSampleNameLength = DLG_MAX_SAMPLE_NAME_LENGTH;
    options->maxSamples = DLG_MAX_SAMPLE_COUNT_IN_IMAGE;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::InitializeCounterData(void)
{
    LWPW_DCGM_PeriodicSampler_CounterDataImageOptions options;
    
    MakeCounterDataImageOptions(&options);

    LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params initializeParams;
    
    memset(&initializeParams, 0, sizeof(initializeParams));
    initializeParams.structSize = LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params_STRUCT_SIZE;
    initializeParams.pOptions = &options;
    initializeParams.pCounterDataImage = &m_counterDataImage[0];
    initializeParams.counterDataImageSize = m_counterDataImage.size();
    const LWPA_Status status = LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize(&initializeParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d", "Got status %d from LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize", status);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    m_nextCounterDataStartIndex = 0;

    PRINT_DEBUG("%u", "InitializeCounterData was successful for deviceIndex %u.", m_pwDeviceIndex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::InitializeWithMetrics(std::vector<const char *> &metricNames, 
                                                  std::vector<unsigned int> &metricIds)
{
    dcgmReturn_t dcgmReturn;
    LWPA_Status status;
    
    /* Create the metrics contexts that the config will need */
    dcgmReturn = CreateMetricsContext();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = CreateRawMetricsConfig();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Create configuration */
    dcgmReturn = CreateCounterDataBuilder();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Prepare to add counters to our new configuration */
    LWPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams;
    memset(&beginPassGroupParams, 0, sizeof(beginPassGroupParams));
    beginPassGroupParams.structSize = LWPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE;
    beginPassGroupParams.pRawMetricsConfig = m_rawMetricsConfig;
    status = LWPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_BeginPassGroup failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    /* Add counters to configuration */
    for (size_t i = 0; i < metricNames.size(); i++)
    {
        dcgmReturn = AddMetricToConfig(metricNames[i]);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%s %d", "AddMetricToConfig(%s) returned %d", metricNames[i], dcgmReturn);;
            return dcgmReturn;
        }
    }

    /* Tell PW we are done adding counters to this config */
    LWPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParam;
    memset(&endPassGroupParam, 0, sizeof(endPassGroupParam));
    endPassGroupParam.structSize = LWPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE;
    endPassGroupParam.pRawMetricsConfig = m_rawMetricsConfig;
    status = LWPW_RawMetricsConfig_EndPassGroup(&endPassGroupParam);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_EndPassGroup failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    /* Get configuration object as a binary blob that we can enable */
    dcgmReturn = CreateConfigAndPrefixImages();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Get our counter data image that we will read to */
    LWPW_DCGM_PeriodicSampler_CounterDataImageOptions options;
    MakeCounterDataImageOptions(&options);
    
    LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params callwlateSizeParams; 
    
    memset(&callwlateSizeParams, 0, sizeof(callwlateSizeParams));
    callwlateSizeParams.structSize = LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params_STRUCT_SIZE;
    callwlateSizeParams.pOptions = &options;
    

    status = LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize(&callwlateSizeParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_EndPassGroup failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    m_counterDataImage.resize(callwlateSizeParams.counterDataImageSize);

    PRINT_DEBUG("%d", "counterDataImageSize %d", (int)callwlateSizeParams.counterDataImageSize);

    /* Initialize the counter image data, resetting the cursor to the beginning */
    dcgmReturn = InitializeCounterData();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Everything is good. Mark these metrics and metric IDs as active */
    m_metricNames = metricNames;
    m_metricIds = metricIds;
    m_metricValues.resize(metricNames.size());

    PRINT_DEBUG("%u %d", "InitializeWithMetrics was successful for deviceIndex %u. %d metrics", 
                m_pwDeviceIndex, (int)m_metricNames.size());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::CreateConfigAndPrefixImages()
{
    LWPA_Status status;

    LWPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParam;
    memset(&generateConfigImageParam, 0, sizeof(generateConfigImageParam));

    generateConfigImageParam.structSize = LWPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE;
    generateConfigImageParam.pRawMetricsConfig = m_rawMetricsConfig;
    status = LWPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParam);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_GenerateConfigImage failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    /* Make sure the generated config is single-pass */
    LWPW_RawMetricsConfig_GetNumPasses_Params getNumPasses;
    memset(&getNumPasses, 0, sizeof(getNumPasses));
    getNumPasses.structSize = LWPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE;
    getNumPasses.pRawMetricsConfig = m_rawMetricsConfig;
    status = LWPW_RawMetricsConfig_GetNumPasses(&getNumPasses);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_GetNumPasses failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    size_t passesSum = getNumPasses.numPipelinedPasses + getNumPasses.numIsolatedPasses;
    if(passesSum >= 2)
    {
        PRINT_ERROR("%zu", "Our config takes %zu passes, which is more than single pass.", passesSum);
        return DCGM_ST_PROFILING_MULTI_PASS;
    }

    size_t imageSize = 0;
    LWPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParam;
    getConfigImageParam.structSize = LWPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE;
    getConfigImageParam.pBuffer = NULL;
    getConfigImageParam.bytesAllocated = 0;
    getConfigImageParam.pRawMetricsConfig = m_rawMetricsConfig;
    status = LWPW_RawMetricsConfig_GetConfigImage(&getConfigImageParam);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_GetConfigImage failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    imageSize = getConfigImageParam.bytesCopied;
    m_configImage.resize(imageSize);

    memset(&getConfigImageParam, 0, sizeof(getConfigImageParam));
    getConfigImageParam.structSize = LWPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE;
    getConfigImageParam.pRawMetricsConfig = m_rawMetricsConfig;
    getConfigImageParam.bytesAllocated = imageSize;
    getConfigImageParam.pBuffer = &m_configImage[0];
    status = LWPW_RawMetricsConfig_GetConfigImage(&getConfigImageParam);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_GetConfigImage failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    imageSize = 0;
    
    LWPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams;
    memset(&getCounterDataPrefixParams, 0, sizeof(getCounterDataPrefixParams));
    getCounterDataPrefixParams.structSize = LWPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE;
    getCounterDataPrefixParams.bytesAllocated = 0;
    getCounterDataPrefixParams.pBuffer = NULL;
    getCounterDataPrefixParams.pCounterDataBuilder = m_counterDataBuilder;
    status = LWPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_CounterDataBuilder_GetCounterDataPrefix failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    imageSize = getCounterDataPrefixParams.bytesCopied;
    m_counterDataImagePrefix.resize(imageSize);

    getCounterDataPrefixParams.structSize = LWPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE;
    getCounterDataPrefixParams.bytesAllocated = imageSize;
    getCounterDataPrefixParams.pBuffer = &m_counterDataImagePrefix[0];
    getCounterDataPrefixParams.pCounterDataBuilder = m_counterDataBuilder;
    status = LWPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_CounterDataBuilder_GetCounterDataPrefix failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    PRINT_DEBUG("%u %d", "CreateConfigAndPrefixImages was successful for deviceIndex %u. imageSize %d", 
                m_pwDeviceIndex, (int)imageSize);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::GetSamples(std::vector<DcgmLopSample_t> &samples)
{
    LWPA_Status status;
    dcgmReturn_t dcgmReturn;

    /* If we're more than 90% into our counter buffer, reset it */
    if (m_nextCounterDataStartIndex >= (DLG_MAX_SAMPLE_COUNT_IN_IMAGE * 9) / 10)
    {
        dcgmReturn = InitializeCounterData();
        if(dcgmReturn != DCGM_ST_OK)
            return dcgmReturn;
    }
    
    LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params triggerKeepParams;
    
    memset(&triggerKeepParams, 0, sizeof(triggerKeepParams));
    triggerKeepParams.structSize = LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params_STRUCT_SIZE;
    triggerKeepParams.deviceIndex = m_pwDeviceIndex;
    triggerKeepParams.pSampleName = DLG_SAMPLE_NAME; /* This must be a non-empty string */
    triggerKeepParams.sampleNameLength = DLG_SAMPLE_NAME_LENGTH; /* if pSampleName is given, pass strlen of it */
    status = LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep(&triggerKeepParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    LWPW_DCGM_PeriodicSampler_DecodeCounters_Params decodeCountersParams;
    memset(&decodeCountersParams, 0, sizeof(decodeCountersParams));
    decodeCountersParams.structSize = LWPW_DCGM_PeriodicSampler_DecodeCounters_Params_STRUCT_SIZE;
    decodeCountersParams.deviceIndex = m_pwDeviceIndex;
    decodeCountersParams.pCounterDataImage = &m_counterDataImage[0];
    decodeCountersParams.counterDataImageSize = m_counterDataImage.size();

    status = LWPW_DCGM_PeriodicSampler_DecodeCounters(&decodeCountersParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_PeriodicSampler_DecodeCounters failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    size_t startRangeIndex = m_nextCounterDataStartIndex;
    m_nextCounterDataStartIndex += decodeCountersParams.numSamplesDecoded;

    LWPW_MetricsContext_SetCounterData_Params setCounterDataParams;
    memset(&setCounterDataParams, 0, sizeof(setCounterDataParams));
    setCounterDataParams.structSize = LWPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE;
    setCounterDataParams.pMetricsContext = m_metricsContext;
    setCounterDataParams.isolated = true;
    setCounterDataParams.pCounterDataImage = &m_counterDataImage[0];
    setCounterDataParams.rangeIndex = startRangeIndex;

    status = LWPW_MetricsContext_SetCounterData(&setCounterDataParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_PeriodicSampler_DecodeCounters failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    LWPW_MetricsContext_EvaluateToGpuValues_Params evaluateToGPUValueParams;
    evaluateToGPUValueParams.structSize = LWPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE;
    evaluateToGPUValueParams.pMetricsContext = m_metricsContext;
    evaluateToGPUValueParams.ppMetricNames = &m_metricNames[0];
    evaluateToGPUValueParams.numMetrics = m_metricNames.size();
    evaluateToGPUValueParams.pMetricValues = &m_metricValues[0];
    status = LWPW_MetricsContext_EvaluateToGpuValues(&evaluateToGPUValueParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_MetricsContext_EvaluateToGpuValues failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    if(samples.size() != m_metricValues.size())
        samples.resize(m_metricValues.size());
    
    bool writeCounterDataToFile = m_alwaysWriteCounterData;

    for(unsigned int i = 0; i < m_metricIds.size(); i++)
    {
        samples[i].metricId = m_metricIds[i];
        /* LOP nan values = DCGM blank values */
        if(isnan(m_metricValues[i]))
        {
            samples[i].value = DCGM_FP64_BLANK;
            #ifdef DLG_WRITE_COUNTER_IMAGE_TO_FILE_AFTER_NAN
            writeCounterDataToFile = true;
            #endif
        }
        else
            samples[i].value = m_metricValues[i];
    }

    if(writeCounterDataToFile)
        WriteCounterDataImageToFile();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::DestroyMetricsContext(void)
{
    LWPW_MetricsContext_Destroy_Params destroyParams;

    if(!m_metricsContext)
        return DCGM_ST_OK;

    memset(&destroyParams, 0, sizeof(destroyParams));
    destroyParams.structSize = LWPW_MetricsContext_Destroy_Params_STRUCT_SIZE;
    destroyParams.pMetricsContext = m_metricsContext;

    m_metricsContext = NULL; /* Set to null regardless of the outcome */

    LWPA_Status status = LWPW_MetricsContext_Destroy(&destroyParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_MetricsContext_Destroy returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    PRINT_DEBUG("", "DestroyMetricsContext was successful.");
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::CreateMetricsContext(void)
{
    LWPA_Status status;
    
    /* If we already have a metrics context active, destroy it, as we can't
       reuse it */
    if(m_metricsContext != NULL)
    {
        DestroyMetricsContext();
        /* Continuing on purpose. We will alloc a new one below */
    }

    LWPW_DCGM_MetricsContext_Create_Params metricsContextCreateParams = { LWPW_DCGM_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pPriv = NULL;
    metricsContextCreateParams.pChipName = m_chipName;
    status = LWPW_DCGM_MetricsContext_Create(&metricsContextCreateParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_MetricsContext_Create returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    m_metricsContext = metricsContextCreateParams.pMetricsContext;

    PRINT_DEBUG("%u", "CreateMetricsContext was successful for deviceIndex %u.", m_pwDeviceIndex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::DestroyRawMetricsConfig(void)
{
    LWPW_RawMetricsConfig_Destroy_Params destroyParams;

    if(!m_rawMetricsConfig)
        return DCGM_ST_OK;
    
    memset(&destroyParams, 0, sizeof(destroyParams));
    destroyParams.structSize = LWPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE;
    destroyParams.pRawMetricsConfig = m_rawMetricsConfig;

    m_rawMetricsConfig = NULL; /* Set to null regardless of the outcome */

    LWPA_Status status = LWPW_RawMetricsConfig_Destroy(&destroyParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_RawMetricsConfig_Destroy returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::CreateRawMetricsConfig(void)
{
    LWPA_Status status;
    
    if(m_rawMetricsConfig != NULL)
    {
        DestroyRawMetricsConfig();
        /* Continuing on purpose. We will alloc a new one below */
    }
    
    LWPW_DCGM_RawMetricsConfig_Create_Params configParams = {LWPW_DCGM_RawMetricsConfig_Create_Params_STRUCT_SIZE };
    configParams.activityKind = LWPA_ACTIVITY_KIND_PROFILER;
    configParams.pChipName = m_chipName;

    status = LWPW_DCGM_RawMetricsConfig_Create(&configParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_RawMetricsConfig_Create returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    m_rawMetricsConfig = configParams.pRawMetricsConfig;

    PRINT_DEBUG("%u", "CreateRawMetricsConfig was successful for deviceIndex %u.", m_pwDeviceIndex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::CreateCounterDataBuilder(void)
{
    if(m_counterDataBuilder)
        return DCGM_ST_OK;
    
    LWPW_CounterDataBuilder_Create_Params counterDataBuilderParams;
    memset(&counterDataBuilderParams, 0, sizeof(counterDataBuilderParams));
    
    counterDataBuilderParams.structSize = LWPW_CounterDataBuilder_Create_Params_STRUCT_SIZE;
    counterDataBuilderParams.pChipName = m_chipName;
    LWPA_Status status = LWPW_CounterDataBuilder_Create(&counterDataBuilderParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_CounterDataBuilder_Create failed with %d for deviceIndex %u",
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    m_counterDataBuilder = counterDataBuilderParams.pCounterDataBuilder;

    PRINT_DEBUG("%u", "CreateCounterDataBuilder was successful for deviceIndex %u.", m_pwDeviceIndex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::DestroyCounterDataBuilder(void)
{
    if(m_counterDataBuilder == NULL)
        return DCGM_ST_OK;

    LWPA_Status status = LWPA_CounterDataBuilder_Destroy(m_counterDataBuilder);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPA_CounterDataBuilder_Destroy returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    m_counterDataBuilder = NULL;

    PRINT_DEBUG("%u", "DestroyCounterDataBuilder was successful for deviceIndex %u.", m_pwDeviceIndex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::SetConfig(void)
{
    LWPW_DCGM_PeriodicSampler_SetConfig_Params setConfigParams;
    
    memset(&setConfigParams, 0, sizeof(setConfigParams));
    setConfigParams.structSize = LWPW_DCGM_PeriodicSampler_SetConfig_Params_STRUCT_SIZE;
    setConfigParams.deviceIndex = m_pwDeviceIndex;
    setConfigParams.pConfig = &m_configImage[0];
    setConfigParams.configSize = m_configImage.size();

    const LWPA_Status status = LWPW_DCGM_PeriodicSampler_SetConfig(&setConfigParams);
    if (status == LWPA_STATUS_ILWALID_ARGUMENT)
    {
        PRINT_ERROR("%u", "LWPW_DCGM_PeriodicSampler_SetConfig returned ILWALID_ARGUMENT for pw pwDeviceIndex %u. "
                    "This is likely due to the metrics provided not being single-pass compatible.", 
                    m_pwDeviceIndex);
        return DCGM_ST_PROFILING_MULTI_PASS;
    }
    else if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_DCGM_PeriodicSampler_SetConfig returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    PRINT_DEBUG("%u", "SetConfig was successful for deviceIndex %u.", m_pwDeviceIndex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmLopConfig::AddMetricToConfig(const char* metricName)
{
    dcgmReturn_t retVal = DCGM_ST_OK;

    LWPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams;
    memset(&getMetricPropertiesBeginParams, 0, sizeof(getMetricPropertiesBeginParams));
    
    getMetricPropertiesBeginParams.structSize = LWPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE;
    getMetricPropertiesBeginParams.pMetricName = metricName;
    getMetricPropertiesBeginParams.pMetricsContext = m_metricsContext;

    LWPA_Status status = LWPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams);
    if (status != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d %u", "LWPW_MetricsContext_GetMetricProperties_Begin returned %d for pw pwDeviceIndex %u", 
                    status, m_pwDeviceIndex);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    for (size_t rawMetricIdx = 0; getMetricPropertiesBeginParams.ppRawMetricDependencies[rawMetricIdx]; ++rawMetricIdx)
    {
        const char* const pRawMetricName = getMetricPropertiesBeginParams.ppRawMetricDependencies[rawMetricIdx];

        LWPA_RawMetricRequest rawMetricRequest = { LWPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
        rawMetricRequest.pMetricName = pRawMetricName;
        rawMetricRequest.isolated = true;
        rawMetricRequest.keepInstances = true;

        LWPW_CounterDataBuilder_AddMetrics_Params addMetricParams;
        memset(&addMetricParams, 0, sizeof(addMetricParams)); 
        addMetricParams.structSize = LWPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE;
        addMetricParams.numMetricRequests = 1;
        addMetricParams.pCounterDataBuilder = m_counterDataBuilder;
        addMetricParams.pRawMetricRequests = &rawMetricRequest;
        LWPA_Status status = LWPW_CounterDataBuilder_AddMetrics(&addMetricParams);
        if (status != LWPA_STATUS_SUCCESS)
        {
            PRINT_ERROR("%d", "LWPW_CounterDataBuilder_AddMetrics returned %d", status);
            retVal = DCGM_ST_PROFILING_LIBRARY_ERROR;
            break;
        }

        LWPW_RawMetricsConfig_AddMetrics_Params configAddMetricParams;
        memset(&configAddMetricParams, 0, sizeof(configAddMetricParams));
        configAddMetricParams.structSize = LWPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE;
        configAddMetricParams.numMetricRequests = 1;
        configAddMetricParams.pRawMetricRequests = &rawMetricRequest;
        configAddMetricParams.pRawMetricsConfig = m_rawMetricsConfig;
        status = LWPW_RawMetricsConfig_AddMetrics(&configAddMetricParams);
        if (status != LWPA_STATUS_SUCCESS)
        {
            PRINT_ERROR("%d", "LWPW_CounterDataBuilder_AddMetrics returned %d", status);
            retVal = DCGM_ST_PROFILING_LIBRARY_ERROR;
        }
    }

    /* Free memory allocated with LWPW_MetricsContext_GetMetricProperties_Begin() */
    LWPW_MetricsContext_GetMetricProperties_End_Params endParams;
    memset(&endParams, 0, sizeof(endParams));
    endParams.structSize = LWPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE;
    endParams.pMetricsContext = m_metricsContext;
    LWPW_MetricsContext_GetMetricProperties_End(&endParams);

    PRINT_DEBUG("%u %s", "AddMetricToConfig was successful for deviceIndex %u. metricName %s", 
                m_pwDeviceIndex, metricName);
    return retVal;
}

/*****************************************************************************/
void DcgmLopConfig::WriteCounterDataImageToFile(void)
{
    char cwd[1024] = {0};
    char filename[1024] = {0};

    getcwd(cwd, sizeof(cwd));
    snprintf(filename, sizeof(filename), "%s/counterdata_%d_%d_%d.data", 
             cwd, (int)getpid(), m_configIndex, m_counterDataIndex);
    m_counterDataIndex++;

    PRINT_INFO("%d %s", "Writing %d bytes of counter data to %s", (int)m_counterDataImage.size(), filename);

    FILE *fp = fopen(filename, "wb");
    if(!fp)
    {
        int errnoTemp = errno;
        PRINT_ERROR("%s %d", "Unable to open %s. errno %d", filename, errnoTemp);
        return;
    }

    fwrite(&m_counterDataImage[0], m_counterDataImage.size(), 1, fp);
    fclose(fp);
}

/*****************************************************************************/
