#ifndef _LWVS_LWVS_Plugin_H_
#define _LWVS_LWVS_Plugin_H_

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <signal.h>
#include <pthread.h>

#include "TestParameters.h"
#include "common.h"
#include "Output.h"
#include "Gpu.h"
#include "DcgmMutex.h"
#include "DcgmError.h"

// This is a base class for all test plugins
// Once the EUD and Healthmon are colwerted to a plugin,
// this will likely go away and functionality moved to the Test
// class

typedef struct
{
    std::string name;
    std::string shortDescription;
    std::string testGroups;
    void* lwstomEntry;
    bool selfParallel;
    TestParameters *defaultTestParameters;
    std::string logFileTag; /* Name to add onto log files to have different
                               tests generate different log files */
} infoStruct_t;

// observedMetrics: map the metric name to a map of GPU ID -> value
typedef std::map<std::string, std::map<unsigned int, double> > observedMetrics_t;

class Plugin
{
/***************************PUBLIC***********************************/
public:
    Plugin();
    virtual ~Plugin();

    /* Interface methods for running the plugin */
    virtual void Go(TestParameters *testParameters) = 0;
    virtual void Go(TestParameters *testParameters, unsigned int) = 0;
    virtual void Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList) = 0;

    /* Getters and Setters */
    /*************************************************************************/
    infoStruct_t GetInfoStruct() const
    {
        return m_infoStruct;
    }

    /*************************************************************************/
    void SetOutputObject(Output *obj)
    {
        progressOut = obj;
    }

    /* Plugin results */
    /*************************************************************************/
    /*
     * Gets overall result from the specified results.
     *
     * Returns:
     *      - LWVS_RESULT_PASS if all GPUs had result LWVS_RESULT_PASS
     *      - LWVS_RESULT_FAIL if *any* GPU had result LWVS_RESULT_FAIL
     *      - LWVS_RESULT_WARN if *any* GPU had result LWVS_RESULT_WARN
     *      - LWVS_RESULT_SKIP if all GPUs had result LWVS_RESULT_SKIP
     */
    static lwvsPluginResult_t GetOverallResult(const lwvsPluginGpuResults_t &results);

    /*************************************************************************/
    /*
     * Gets overall result for this test.
     *
     * Returns:
     *      - LWVS_RESULT_PASS if all GPUs had result LWVS_RESULT_PASS
     *      - LWVS_RESULT_FAIL if *any* GPU had result LWVS_RESULT_FAIL
     *      - LWVS_RESULT_WARN if *any* GPU had result LWVS_RESULT_WARN
     *      - LWVS_RESULT_SKIP if all GPUs had result LWVS_RESULT_SKIP
     */
    lwvsPluginResult_t GetResult();

    /*************************************************************************/
    /*
     * Get results for all GPUs.
     *
     * Returns:
     *      - lwvsPluginGpuResults_t (map from gpu id to results)
     */
    const lwvsPluginGpuResults_t& GetGpuResults() const
    {
        return m_results;
    }

    /*************************************************************************/
    /*
     * Sets the result for all GPUs to the result given by res.
     *
     */
    void SetResult(lwvsPluginResult_t res);

    /*************************************************************************/
    /*
     * Sets the result for the GPU given by gpuId to the result given by res.
     *
     */
    void SetResultForGpu(unsigned int gpuId, lwvsPluginResult_t res);

    /*************************************************************************/
    const std::vector<std::string>& GetWarnings() const
    {
        return m_warnings;
    }

    /*************************************************************************/
    const std::vector<DcgmError> &GetErrors() const
    {
        return m_errors;
    }

    /*************************************************************************/
    const lwvsPluginGpuErrors_t &GetGpuErrors() const
    {
        return m_errorsPerGPU;
    }

    /*************************************************************************/
    const lwvsPluginGpuMessages_t& GetGpuWarnings() const
    {
        return m_warningsPerGPU;
    }

    /*************************************************************************/
    const std::vector<std::string>& GetVerboseInfo() const
    {
        return m_verboseInfo;
    }

    /*************************************************************************/
    const lwvsPluginGpuMessages_t& GetGpuVerboseInfo() const
    {
        return m_verboseInfoPerGPU;
    }

    /*************************************************************************/
    inline void RecordObservedMetric(unsigned int gpuId, const std::string &valueName, double value)
    {
        m_values[valueName][gpuId] = value;
    }

    /*************************************************************************/
    observedMetrics_t GetObservedMetrics() const
    {
        return m_values;
    }

    /* Methods */
    /*************************************************************************/
    /*
     * Initializes internal result and message structures for use with the gpus given by gpuList.
     * This method **MUST** be called before the plugin logs any messages or sets a result.
     *
     * This method clears any existing warnings, info messages, and results as a side effect.
     * Sets m_gpuList to a copy of the given gpuList.
     *
     */
    void InitializeForGpuList(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Adds an error for this plugin
     *
     * Thread-safe.
     */
    void AddError(const DcgmError &error);

    /*************************************************************************/
    /*
     * Adds an error for the GPU specified by gpuId
     *
     * Thread-safe.
     */
    void AddErrorForGpu(unsigned int gpuId, const DcgmError &error);

    /*************************************************************************/
    /*
     * Logs an info message.
     *
     * Thread-safe.
     */
    void AddInfo(const std::string &info);

    /*************************************************************************/
    /*
     * Adds a non-GPU specific verbose message.
     *
     * Thread-safe.
     */
    void AddInfoVerbose(const std::string &info);

    /*************************************************************************/
    /*
     * Adds a verbose message for the GPU given by gpuId.
     *
     * Thread-safe.
     */
    void AddInfoVerboseForGpu(unsigned int gpuId, const std::string &info);

    /* Variables */
    Output *progressOut; // Output object passed in from the test framework for progress updates

/***************************PRIVATE**********************************/
private:
    /* Methods */
    /*************************************************************************/
    /*
     * Clears all warnings, info messages, and results.
     *
     */
    void ResetResultsAndMessages();

    /* Variables */
    lwvsPluginGpuResults_t      m_results;                  /* Per GPU results: Pass | Fail | Skip | Warn */
    std::vector<std::string>    m_warnings;                 /* List of general warnings from the plugin */
    std::vector<DcgmError>      m_errors;                   /* List of errors from the plugin */
    lwvsPluginGpuErrors_t       m_errorsPerGPU;             // Per GPU list of errors from the plugin
    lwvsPluginGpuMessages_t     m_warningsPerGPU;           /* Per GPU list of warnings from the plugin */
    std::vector<std::string>    m_verboseInfo;              /* List of general verbose output from the plugin */
    lwvsPluginGpuMessages_t     m_verboseInfoPerGPU;        /* Per GPU list of verbose output from the plugin */
    observedMetrics_t           m_values;                   /* Record the values found for pass/fail criteria */
    /* Mutexes */
    DcgmMutex                   m_dataMutex;                /* Mutex for plugin data */

/***************************PROTECTED********************************/
protected:
    /* Variables */
    infoStruct_t                m_infoStruct;
    std::string                 m_logFile;
    DcgmMutex                   m_mutex;  /* mutex for locking the plugin (for use by subclasses). */
    std::vector<unsigned int>   m_gpuList; /* list of GPU ids for this plugin */
};

// typedef for easier referencing for the factory
typedef Plugin *maker_t();
extern "C" {
    extern std::map<std::string, maker_t *, std::less<std::string> > factory;
}
#endif //_LWVS_LWVS_Plugin_H_
