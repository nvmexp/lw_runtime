#pragma once

#include "dcgm_structs.h"
#include <vector>
#include "lwperf_host.h"
#include "lwperf_dcgm_target_priv.h"

/* Constants */
#define DLG_MAX_SAMPLE_COUNT_IN_IMAGE 100 /* Maximum number of samples we will store in the counter
                                             image before decoding any. This should be at least 1.
                                             InitializeCounterData() will be called every time we 
                                             get to 90% of this */

#define DLG_SAMPLE_NAME "dcgm" /* This must be a non-empty string */
#define DLG_SAMPLE_NAME_LENGTH 4 /* This should be the string length of DLG_SAMPLE_NAME */
#define DLG_MAX_SAMPLE_NAME_LENGTH 10 /* This must be >= DLG_SAMPLE_NAME_LENGTH */

#define DLG_MAX_METRIC_GROUPS 5 /* Maximum number of metric groups we can multiplex at once */

/* Struct for a sample returned by ReadMetrics() */
typedef struct 
{
    unsigned int metricId;  /* Sample ID that was passed into metricIds */
    double value;           /* Value of the metric */
} DcgmLopSample_t;

/* 
Class to manage a single LOP configuration for a given GPU. A configuration manages a mapping
of LOP metrics to values on the GPU for a single pass group
*/

class DcgmLopConfig
{
public:
    /*************************************************************************/
    /* 
     * Constructor 
     * 
     * Should not throw. Instead, initialize this class from Init()
     */
    DcgmLopConfig(int pwDeviceIndex, int configIndex, const char *chipName);

    /*************************************************************************/
    /*
     * Soft constructor
     */
    dcgmReturn_t Init(void);

    /*************************************************************************/
    /*
     * Destructor
     */
    ~DcgmLopConfig();

    /*************************************************************************/
    /*
     * Prepare this class to SetConfig() by giving it a set of metrics to build
     * a configuration for.
     * 
     * Returns: DCGM_ST_OK on success
     *          Other DCGM_ST_? error code on failure
     */
    dcgmReturn_t InitializeWithMetrics(std::vector<const char *> &metricNames, 
                                       std::vector<unsigned int> &metricIds);
    
    /*************************************************************************/
    /* 
     * Tell perfworks to snapshot and decode metrics, returning them in samples[]
     */
    dcgmReturn_t GetSamples(std::vector<DcgmLopSample_t> &samples);

    /*************************************************************************/
    /* Various perfworks API helpers */
    dcgmReturn_t CreateMetricsContext();
    dcgmReturn_t CreateRawMetricsConfig();
    dcgmReturn_t CreateCounterDataBuilder();
    dcgmReturn_t DestroyCounterDataBuilder();
    dcgmReturn_t AddMetricToConfig(const char* pMetricName);
    dcgmReturn_t CreateConfigAndPrefixImages();
    dcgmReturn_t SetConfig();
    void MakeCounterDataImageOptions(LWPW_DCGM_PeriodicSampler_CounterDataImageOptions *options);
    dcgmReturn_t InitializeCounterData();
    dcgmReturn_t DestroyMetricsContext();
    dcgmReturn_t DestroyRawMetricsConfig();

private:
    /*************************************************************************/    
    /* Write our current counter data image to a file */
    void WriteCounterDataImageToFile(void);

    /*************************************************************************/

private:
    unsigned int m_pwDeviceIndex; /* Perfworks device index this config belongs to */
    const char *m_chipName;       /* Chip name in Perfworks. Several APIs need this parameter */
    int m_configIndex;            /* Index of our config inside of our owning DcgmLopGpu object */
    bool m_alwaysWriteCounterData; /* Should we always dump our counter data to a file after we read it? */
    unsigned int m_counterDataIndex; /* Counter data image filename index 0,1.2.... used to generate
                                        unique filenames */

    LWPA_MetricsContext *m_metricsContext; /* Perfworks metric context needed by several APIs */
    LWPA_RawMetricsConfig *m_rawMetricsConfig; /* Perfworks raw metric config needed by several APIs */
    LWPA_CounterDataBuilder *m_counterDataBuilder; /* Perfworks counter data builder */
    std::vector<uint8_t> m_configImage; /* Config image that was initialized with InitializeWithMetrics() */
    std::vector<uint8_t> m_counterDataImagePrefix; /* Counter prefix initialized with InitializeWithMetrics() */
    std::vector<uint8_t> m_counterDataImage; /* Counter data image buffer */
    size_t m_nextCounterDataStartIndex; /* Where the next counter data will be written in m_counterDataImage. PW doesn't
                                           reuse the same place in the buffer unless you call InitializeCounterData() again.
                                           So you have to keep track of PW's cursor for it and manually reset the cursor when
                                           it gets to the end. */
    
    /* The following arrays are kept separate to optimize LWPW_MetricsContext_EvaluateToGpuValues() */
    std::vector<const char *>m_metricNames; /* Metrics that are lwrrently watched */
    std::vector<unsigned int>m_metricIds;   /* Metric ids that are lwrrently watched. The indicies of these match up with m_metricNames */
    std::vector<double>m_metricValues;      /* Bucket to catch values in. Same size and indicies as previous two vectors */
};
