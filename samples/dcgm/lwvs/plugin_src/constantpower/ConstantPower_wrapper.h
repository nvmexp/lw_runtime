#ifndef _LWVS_LWVS_ConstantPower_H_
#define _LWVS_LWVS_ConstantPower_H_

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "Plugin.h"
#include "PluginCommon.h"
#include "PluginDevice.h"
#include "lwca.h"
#include "lwblas_v2.h"
#include "DcgmRecorder.h"
#include "DcgmError.h"

#define CP_MAX_DIMENSION 4096 /* Maximum single dimension */
#define CP_MAX_DEVICES 16 /* Maximum number of devices to run this on conlwrrently */
#define CP_MAX_STREAMS_PER_DEVICE 24  /* Maximum number of Lwca streams to use to pipeline
                                         operations to the card */
#define CP_MAX_OUTPUT_MATRICES 16   /* Maximum number of output arrays or "C" matricies.
                                       We use multiple of these to avoid global memory conflicts
                                       when multiplying and adding A *+ B = C. A and B are
                                       constant throughout the test */

/*****************************************************************************/
/* Class for a single constant power device */
class CPDevice : public PluginDevice
{
public:
    /* LWML Power details */
    double maxPowerTarget; /* Maximum power we can target in watts */

    int NlwdaStreams;                    /* Number of lwdaStream[] entries that are valid */
    lwdaStream_t lwdaStream[CP_MAX_STREAMS_PER_DEVICE]; /* Lwca streams */

    int allocatedLwblasHandle;           /* Have we allocated lwblasHandle yet? */
    lwblasHandle_t lwblasHandle;         /* Handle to lwBlas */

    /* Minimum adjusted value for our matrix dimension. 1 <= X <= MAX_DIMENSION */
    int minMatrixDim;

    /* Should we only make small adjustments in matrix size? This is set to 1 after
     * cp_recalc_matrix_dim has gotten close enough that it thinks it's in the right
     * range
     */
    int onlySmallAdjustments;

    /* Device pointers */
    void *deviceA;
    void *deviceB;
    void *deviceC[CP_MAX_OUTPUT_MATRICES];

    int NdeviceC; /* Number of entries in deviceC that are valid */

    bool m_lowPowerLimit;

	CPDevice(unsigned int ndi, Plugin *p) : PluginDevice(ndi, p), maxPowerTarget(0), NlwdaStreams(0),
	                                        allocatedLwblasHandle(0), lwblasHandle(0), minMatrixDim(0),
											onlySmallAdjustments(0), deviceA(0), deviceB(0), NdeviceC(0),
                                            m_lowPowerLimit(false)
	{
		memset(lwdaStream, 0, sizeof(lwdaStream));
		memset(deviceC, 0, sizeof(deviceC));
	}

	~CPDevice()
	{
		if (allocatedLwblasHandle)
		{
			lwblasDestroy(lwblasHandle);
			lwblasHandle = 0;
			allocatedLwblasHandle = 0;
		}

		for (int i = 0; i < NlwdaStreams; i++)
			lwdaStreamDestroy(lwdaStream[i]);
		NlwdaStreams = 0;

		if (deviceA)
		{
			lwdaFree(deviceA);
			deviceA = 0;
		}
		if (deviceB)
		{
			lwdaFree(deviceB);
			deviceB = 0;
		}

		for (int i = 0; i < NdeviceC; i++)
		{
			if (deviceC[i])
			{
				lwdaFree(deviceC[i]);
				deviceC[i] = 0;
			}
		}
	}
};


/*****************************************************************************/
class ConstantPower : public Plugin
{
public:
    ConstantPower();
    ~ConstantPower();

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
     * Run Targeted Power test
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
    bool CheckPassFailSingleGpu(CPDevice *device, std::vector<DcgmError> &errorList, timelib64_t startTime,
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
     * Runs the Targeted Power test
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
    bool CheckGpuPowerUsage(CPDevice *device, std::vector<DcgmError> &errorList,
                            timelib64_t startTime, timelib64_t earliestStopTime);
    bool CheckGpuTemperature(CPDevice *device, std::vector<DcgmError> &errorList,
                             timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished);
    bool CheckLwmlEvents(CPDevice *device, std::vector<DcgmError> &errorList,
                         timelib64_t startTime, timelib64_t earliestStopTime);

    /*
     * Sets the result to skip if the enforced power limit of any GPU is too low to realistically 
     * hit the target power for the test
     */
    bool EnforcedPowerLimitTooLow();

    /*************************************************************************/
    /* Variables */
    TestParameters              *m_testParameters;          /* Parameters for this test, passed in from the framework.
                                                               DO NOT DELETE */
    bool                        m_lwmlInitialized;          /* Has lwmlInit been called? */
    bool                        m_dcgmCommErrorOclwrred;    /* Has there been a communication error with DCGM? */
    bool                        m_dcgmRecorderInitialized;  /* Has DcgmRecorder been initialized? */
    DcgmRecorder                m_dcgmRecorder;             /* DCGM stats recording interfact object */
    std::vector<CPDevice *>     m_device;                   /* Per-device data */

    /* Cached parameters read from testParameters */
    double                      m_testDuration;             /* Test duration in seconds */
    int                         m_useDgemm;                 /* Whether or not to use dgemm (or sgemm) 1=use dgemm */
    double                      m_targetPower;              /* Target power for the test in watts */
    double                      m_sbeFailureThreshold;      /* how many SBEs constitutes a failure */

    /* Arrays for lwblasDgemm. Allocated at MAX_DIMENSION^2 * sizeof(double) */
    void                        *m_hostA;
    void                        *m_hostB;
    void                        *m_hostC;
};



#endif // _LWVS_LWVS_ConstantPower_H_
