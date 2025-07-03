#ifndef SMPERFPLUGIN_H
#define SMPERFPLUGIN_H

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "Plugin.h"
#include "PluginDevice.h"
#include "PluginCommon.h"
#include "lwca.h"
#include "lwblas_v2.h"
#include "DcgmRecorder.h"
#include "DcgmError.h"

/*****************************************************************************/
/* Test dimension. Used for both M and N
 * See https://wiki.lwpu.com/lwcompute/index.php/LwBLAS for
 * guidelines for picking matrix size
 */
#define SMPERF_TEST_DIMENSION 1024 /* Test single dimension */


#define SMPERF_MAX_DEVICES 32 /* Maximum number of devices to run this on conlwrrently */

/*****************************************************************************/
/* String constants */

/* Stat names in the JSON output */
#define PERF_STAT_NAME "perf_gflops"

/*****************************************************************************/
/* Class for a single sm perf device */
class SmPerfDevice : public PluginDevice
{
public:
    int allocatedLwblasHandle;           /* Have we allocated lwblasHandle yet? */
    lwblasHandle_t lwblasHandle;         /* Handle to lwBlas */

    /* Device pointers */
    void *deviceA;
    void *deviceB;
    void *deviceC;

    /* Arrays for lwblasDgemm. Allocated at MAX_DIMENSION^2 * sizeof(double) */
    void *hostA;
    void *hostB;
    void *hostC;

    SmPerfDevice(unsigned int ndi, Plugin *p) : PluginDevice(ndi, p), allocatedLwblasHandle(0), lwblasHandle(0),
                                                deviceA(0), deviceB(0), deviceC(0), hostA(0), hostB(0), hostC(0)
    {
    }

    ~SmPerfDevice()
    {
        if (allocatedLwblasHandle != 0)
        {
            PRINT_DEBUG("%d %p", "lwblasDestroy lwdaDeviceIdx %d, handle %p", lwdaDeviceIdx, lwblasHandle);
            lwblasDestroy(lwblasHandle);
            lwblasHandle = 0;
            allocatedLwblasHandle = 0;
        }

        if (hostA)
        {
            lwdaFreeHost(hostA);
            hostA = 0;
        }

        if (hostB)
        {
            lwdaFreeHost(hostB);
            hostB = 0;
        }

        if (hostC)
        {
            lwdaFreeHost(hostC);
            hostC = 0;
        }
    }

};

/*****************************************************************************/
class SmPerfPlugin : public Plugin
{
public:
    SmPerfPlugin();
    ~SmPerfPlugin();

    // unimplemented pure virtual funcs
    void Go(TestParameters *testParameters)
    {
        Go(testParameters, 0);
    }
    void Go(TestParameters *testParameters, unsigned int)
    {
        throw std::runtime_error("Not implemented in this test.");
    }

    /*************************************************************************/
    /*
     * Run SM performance tests
     *
     */
    void Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Public so that worker thread can call this method.
     *
     * Checks whether the test has passed for the given device.
     *
     * NOTE: Error information is stored in errorList in case of test failure.
     *
     * Returns: true if the test passed, false otherwise.
     *
     */
    bool CheckPassFailSingleGpu(SmPerfDevice *device, std::vector<DcgmError> &errorList, timelib64_t startTime,
                               timelib64_t earliestStopTime, bool testFinished=true);

    /*************************************************************************/

private:
    /*************************************************************************/
    /*
     * Initialize the parts of lwml needed for this plugin to run
     *
     * Returns: true on success
     *          false on error
     */
    bool LwmlInit(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Initialize the parts of lwca and lwblas needed for this plugin to run
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int LwdaInit(void);

    /*************************************************************************/
    /*
     * Runs the SM Performance test
     *
     * Returns: 
     *      false if there were issues running the test (test failures are not considered issues), 
     *      true otherwise.
     */
    bool RunTest(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Clean up any resources used by this object, freeing all memory and closing
     * all handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Check whether the test has passed for all GPUs and sets the pass/fail result for each GPU. 
     * Called after test is finished.
     *
     * Returns: true if the test passed for all gpus, false otherwise.
     *
     */
    bool CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime);

    /*************************************************************************/
    /*
     * Check various statistics and device properties to determine if the test
     * has passed.
     *
     * Returns: true if the test passed, false otherwise.
     *
     */
    bool CheckGpuPerf(SmPerfDevice *smDevice, std::vector<DcgmError> &errorList, timelib64_t startTime,
                      timelib64_t endTime);
    bool CheckGpuTemperature(SmPerfDevice *smDevice, std::vector<DcgmError> &errorList,
                            timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished);
    bool CheckLwmlEvents(SmPerfDevice *smDevice, std::vector<DcgmError> &errorList,
                        timelib64_t startTime, timelib64_t earliestStopTime);

    /*************************************************************************/
    TestParameters              *m_testParameters;          /* Parameters for this test, passed in from the framework.
                                                               Set when the go() method is called. DO NOT FREE */

    bool                        m_lwmlInitialized;          /* Has lwmlInit been called? */
    DcgmRecorder                m_dcgmRecorder;             /* DCGM stats recording interfact object */
    bool                        m_dcgmRecorderInitialized;  /* Has DcgmRecorder been initialized? */
    std::vector<SmPerfDevice *> m_device;                   /* Per-device data */
    bool                        m_dcgmCommErrorOclwrred;    /* Has there been a communication error with DCGM? */

    /* Cached parameters read from testParameters */
    double                      m_testDuration;             /* Test duration in seconds */
    double                      m_targetPerf;               /* Performance we are trying to target in gigaflops */
    int                         m_useDgemm;                 /* Whether or not to use dgemm (or sgemm) 1=use dgemm */
    double                      m_sbeFailureThreshold;      /* how many SBEs constitutes a failure */
};

#endif // SMPERFPLUGIN_H
