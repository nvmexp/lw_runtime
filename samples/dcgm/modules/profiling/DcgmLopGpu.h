#pragma once

#include "dcgm_structs.h"
#include <vector>
#include "DcgmLopConfig.h"

/* 
Class to manage a LOP configuration for a given GPU. A configuration manages a mapping
of LOP metrics to values on the GPU 
*/

/* Features */
//#define DLG_WRITE_COUNTER_IMAGE_TO_FILE_AFTER_NAN 1

class DcgmLopGpu
{
public:
    /*************************************************************************/
    /* 
     * Constructor 
     * 
     * Should not throw. Instead, initialize this class from Init()
     */
    DcgmLopGpu(int pwDeviceIndex);

    /*************************************************************************/
    /*
     * Soft constructor
     */
    dcgmReturn_t Init(void);
    
    /*************************************************************************/
    /* 
     * Destructor
     */
    ~DcgmLopGpu();

    /*************************************************************************/
    /*
     * Prepare this class to Enable() by giving it a set of metrics to build
     * a configuration for.
     * 
     * Returns: DCGM_ST_OK on success
     *          Other DCGM_ST_? error code on failure
     */
    dcgmReturn_t InitializeWithMetrics(std::vector<const char *> metricNames[DLG_MAX_METRIC_GROUPS], 
                                       std::vector<unsigned int> metricIds[DLG_MAX_METRIC_GROUPS],
                                       int numMetricGroup);

    /*************************************************************************/
    /*
     * Ask perfworks if a given metric name is valid for a device. 
     * 
     * Returns: True if the metric name is valid.
     *          False if not
     */
    bool IsMetricNameValid(const char *metricName);

    /*************************************************************************/
    /*
     * Ask perfworks to start our session. 
     * 
     * Returns: DCGM_ST_OK on success
     *          Other DCGM_ST_? error code on failure
     */
    dcgmReturn_t EnableMetrics();

    /*************************************************************************/
    /*
     * Ask perfworks to stop our session.
     * 
     * Returns: DCGM_ST_OK on success
     *          Other DCGM_ST_? error code on failure
     */
    dcgmReturn_t DisableMetrics();

    /*************************************************************************/
    /*
     * Activate one of the configs that was passed to InitializeWithMetrics()
     */
    dcgmReturn_t SetConfig(int configIndex, bool triggerDiscard);

    /*************************************************************************/
    /* 
     * Tell perfworks to snapshot and decode metrics, returning them in samples[]
     */
    dcgmReturn_t GetSamples(std::vector<DcgmLopSample_t> &samples);

    /*************************************************************************/

private:
    
    /*************************************************************************/
    /* Helper methods to send various perfworks commands */
    dcgmReturn_t BeginSession();
    dcgmReturn_t EndSession();
    dcgmReturn_t StartSampling();
    dcgmReturn_t StopSampling();
    dcgmReturn_t TriggerDiscard();
    void FreeConfigs();

    /*************************************************************************/

private:
    unsigned int m_pwDeviceIndex; /* Perfworks device index this config belongs to */
    bool m_initialized;           /* Has Init() run successfully? */
    bool m_sessionIsActive;       /* Is a perfworks session active? Use StartSession/EndSession to change this */
    bool m_startedSampling;       /* Have we successfully called StartSampling() on this GPU yet? */
    const char *m_chipName;       /* Chip name in Perfworks. Several APIs need this parameter */
    
    std::vector<DcgmLopConfig *> m_configs; /* Vector of LOP configs. Only one can be active on a GPU at a time */
    int m_activeConfigIndex;      /* Index into m_configs of the lwrrently active config. -1 = None */
};

