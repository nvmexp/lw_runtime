#ifndef _LWVS_LWVS_ConstantPerf_H_
#define _LWVS_LWVS_ConstantPerf_H_

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

#define CPERF_TEST_DIMENSION 1280               /* Test single dimension */
#define CPERF_MAX_DEVICES 32                    /* Maximum number of devices to run this on conlwrrently */
#define CPERF_MAX_STREAMS_PER_DEVICE 8          /* Maximum number of Lwca streams to use to pipeline
                                                   operations to the card */
#define CPERF_MAX_CONLWRRENT_OPS_PER_STREAM 100 /* Maximum number of conlwrrent ops
                                                   that can be queued per stream per GPU */

/*****************************************************************************/
/* String constants */

/* Stat names */
#define PERF_STAT_NAME "perf_gflops"

/*****************************************************************************/
/* Per-stream context info */
typedef struct cperf_stream_t
{
    lwdaStream_t lwdaStream; /* Lwca stream handle */

    /* Device pointers */
    void *deviceA;
    void *deviceB;
    void *deviceC;

    /* Arrays for lwblasDgemm. Allocated at MAX_DIMENSION^2 * sizeof(double) */
    void *hostA;
    void *hostB;
    void *hostC;

    /* Timing aclwmulators */
    double usecInCopies; /* How long (microseconds) have we spent copying data to and from the GPU */
    double usecInGemm;   /* How long (microseconds) have we spent running gemm */

    /* Counters */
    int blocksQueued; /* Number of times we've successfully queued CPerfGlobal->atATime ops */

    int NeventsInitalized; /* Number of array entries in the following lwdaEvent_t arrays that are
                              actually initialized */

    /* Events for recording the timing of various activities per stream.
       Look at cperf_queue_one for usage */
    lwdaEvent_t beforeCopyH2D[CPERF_MAX_CONLWRRENT_OPS_PER_STREAM];
    lwdaEvent_t beforeGemm[CPERF_MAX_CONLWRRENT_OPS_PER_STREAM];
    lwdaEvent_t beforeCopyD2H[CPERF_MAX_CONLWRRENT_OPS_PER_STREAM];
    lwdaEvent_t afterCopyD2H[CPERF_MAX_CONLWRRENT_OPS_PER_STREAM];

    lwdaEvent_t afterWorkBlock; /* Every CPerfGlobal->atATime, events, we should
                                   signal this event so that the CPU thread knows
                                   to queue CPerfGlobal->atATime work items again */

} cperf_stream_t, *cperf_stream_p;

/*****************************************************************************/
/* Class for a single constant perf device */
class CPerfDevice : public PluginDevice
{
public:
    int Nstreams;                    /* Number of stream[] entries that are valid */
    cperf_stream_t streams[CPERF_MAX_STREAMS_PER_DEVICE];

    int allocatedLwblasHandle;           /* Have we allocated lwblasHandle yet? */
    lwblasHandle_t lwblasHandle;         /* Handle to lwBlas */

    /* Timing aclwmulators */
    double usecInCopies; /* How long (microseconds) have we spent copying data to and from the GPU */
    double usecInGemm;   /* How long (microseconds) have we spent running gemm */

    CPerfDevice(unsigned int ndi, Plugin *p) : PluginDevice(ndi, p), Nstreams(0), allocatedLwblasHandle(0),
                                               lwblasHandle(0)
    {
        memset(streams, 0, sizeof(streams));
    }

    ~CPerfDevice()
    {
        if (allocatedLwblasHandle)
        {
            lwblasDestroy(lwblasHandle);
            lwblasHandle = 0;
            allocatedLwblasHandle = 0;
        }

        for (int i = 0; i < Nstreams; i++)
        {
            cperf_stream_p cpStream = &streams[i];

            lwdaStreamDestroy(cpStream->lwdaStream);

            if (cpStream->hostA)
            {
                lwdaFreeHost(cpStream->hostA);
                cpStream->hostA = 0;
            }
            if (cpStream->hostB)
            {
                lwdaFreeHost(cpStream->hostB);
                cpStream->hostB = 0;
            }
            if (cpStream->hostC)
            {
                lwdaFreeHost(cpStream->hostC);
                cpStream->hostC = 0;
            }

            if (cpStream->deviceA)
            {
                lwdaFree(cpStream->deviceA);
                cpStream->deviceA = 0;
            }
            if (cpStream->deviceB)
            {
                lwdaFree(cpStream->deviceB);
                cpStream->deviceB = 0;
            }
            if (cpStream->deviceC)
            {
                lwdaFree(cpStream->deviceC);
                cpStream->deviceC = 0;
            }

            for(int j = 0; j < cpStream->NeventsInitalized; j++)
            {
                lwdaEventDestroy(cpStream->beforeCopyH2D[j]);
                lwdaEventDestroy(cpStream->beforeGemm[j]);
                lwdaEventDestroy(cpStream->beforeCopyD2H[j]);
                lwdaEventDestroy(cpStream->afterCopyD2H[j]);
            }
        }
        Nstreams = 0;
    }
};

/*****************************************************************************/
/* Constant Perf plugin */
class ConstantPerf : public Plugin
{
public:
    ConstantPerf();
    ~ConstantPerf();

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
     * Run Targeted Stress tests
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
    bool CheckPassFailSingleGpu(CPerfDevice *device, std::vector<DcgmError> &errorList, timelib64_t startTime,
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
    int LwdaInit();

    /*************************************************************************/
    /*
     * Runs the Targeted Stress test
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
    bool CheckGpuPerf(CPerfDevice *cpDevice, std::vector<DcgmError> &errorList, timelib64_t startTime,
                      timelib64_t endTime);
    bool CheckGpuTemperature(CPerfDevice *cpDevice, std::vector<DcgmError> &errorList,
                             timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished);
    bool CheckLwmlEvents(CPerfDevice *cpDevice, std::vector<DcgmError> &errorList,
                         timelib64_t startTime, timelib64_t earliestStopTime);

    /*************************************************************************/
    /* Variables */
    TestParameters              *m_testParameters;          /* Parameters for this test, passed in from the framework.
                                                               DO NOT DELETE */
    bool                        m_lwmlInitialized;          /* Has lwmlInit been called? */
    bool                        m_dcgmCommErrorOclwrred;    /* Has there been a communication error with DCGM? */
    bool                        m_dcgmRecorderInitialized;  /* Has DcgmRecorder been initialized? */
    DcgmRecorder                m_dcgmRecorder;             /* DCGM stats recording interfact object */
    std::vector<CPerfDevice *>  m_device;                   /* Per-device data */

    /* Cached parameters read from testParameters */
    double                      m_testDuration;             /* Test duration in seconds */
    double                      m_targetPerf;               /* Performance we are trying to target in gigaflops */
    int                         m_useDgemm;                 /* Whether or not to use dgemm (or sgemm) 1=use dgemm */
    int                         m_atATime;                  /* Number of ops to queue to the stream at a time */
    double                      m_sbeFailureThreshold;      /* how many SBEs constitutes a failure */

};



#endif // _LWVS_LWVS_ConstantPerf_H_
