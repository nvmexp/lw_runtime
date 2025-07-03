#ifndef GPUBURNPLUGIN_H
#define GPUBURNPLUGIN_H

#include "Plugin.h"
#include "PluginDevice.h"
#include "PluginCommon.h"
#include "PluginStrings.h"
#include "lwca.h"
#include "lwblas_v2.h"
#include "DcgmRecorder.h"
#include "DcgmError.h"

#define PERF_STAT_NAME      "perf_gflops"
#define GPUBURN_MAX_DEVICES 32


/*****************************************************************************/
/* Class for a single gpuburn device */
class GpuBurnDevice : public PluginDevice
{
public:
    LWdevice        lwDevice;
    LWcontext       lwContext;

    GpuBurnDevice(unsigned int ndi, Plugin *p) : PluginDevice(ndi, p)
    {
        lwmlPciInfo_t pciInfo;
        lwmlReturn_t lwmlSt;
        char buf[256] = { 0 };
        const char *errorString = NULL;

        if ((lwmlSt = lwmlDeviceGetPciInfo(lwmlDevice, &pciInfo)) != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetPciInfo", lwmlErrorString(lwmlSt));
            throw d;
        }

        LWresult lwSt = lwInit(0);
        if (lwSt)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwInit");
            lwvsCommon.errorMask |= GPUBURN_ERR_GENERIC;
            lwGetErrorString(lwSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d)", errorString, static_cast<int>(lwSt));
                d.AddDetail(buf);
            }
            throw d;
        }

        lwSt = lwDeviceGetByPCIBusId(&lwDevice, pciInfo.busId);
        if (lwSt)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwDeviceGetByPCIBusId");
            lwGetErrorString(lwSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d) for GPU %u", errorString, static_cast<int>(lwSt),
                         dcgmDeviceIndex);
                d.AddDetail(buf);
            }
            throw d;
        }

        /* Initialize the runtime implicitly so we can grab its context */
        PRINT_DEBUG("%d", "Attaching to lwca device index %d", (int)lwDevice);
        lwdaSetDevice(lwDevice);
        lwdaFree(0);

        /* Grab the runtime's context */
        lwSt = lwCtxGetLwrrent(&lwContext);
        if (lwSt)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwCtxGetLwrrent");
            lwGetErrorString(lwSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d) for GPU %u", errorString, static_cast<int>(lwSt),
                         dcgmDeviceIndex);
                d.AddDetail(buf);
            }
            throw d;
        }
        else if (lwContext == NULL)
        {
            //lwCtxGetLwrrent doesn't return an error if there's not context, so check and attempt to create one
            lwSt = lwCtxCreate(&lwContext, 0, lwDevice);

            if (lwSt != LWDA_SUCCESS)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwCtxCreate");

                lwGetErrorString(lwSt, &errorString);
                if (errorString != NULL)
                {
                    snprintf(buf, sizeof(buf),
                             "No current LWCA context for GPU %u, and cannot create one: '%s' (%d)",
                             dcgmDeviceIndex, errorString, static_cast<int>(lwSt));
                    d.AddDetail(buf);
                }
                else
                {
                    snprintf(buf, sizeof(buf), "No current LWCA context for GPU %u, and cannot create one: (%d)",
                             dcgmDeviceIndex, static_cast<int>(lwSt));
                    d.AddDetail(buf);
                }

                throw d;
            }
        }
    }

    ~GpuBurnDevice()
    {
    }
};

/*****************************************************************************/
/* GpuBurn plugin */
class GpuBurnPlugin : public Plugin
{
public:
    GpuBurnPlugin();
    ~GpuBurnPlugin();

    // Unimplemented
    void Go(TestParameters *tp)
    {
        Go(NULL, 0);
    }
    void Go(TestParameters *tp, unsigned int i)
    {
        throw std::runtime_error("Not implemented for GPU Burn");
    }

    /*************************************************************************/
    /*
     * Run Diagnostic test
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
    bool CheckPassFailSingleGpu(GpuBurnDevice *device, std::vector<DcgmError> &errorList, timelib64_t startTime,
                                timelib64_t earliestStopTime, bool testFinished=true);

    /*************************************************************************/

private:
    /*************************************************************************/
    /*
     * Initialize the parts of lwml needed for this plugin.
     *
     * Returns: true on success
     *          false on error
     */
    bool LwmlInit(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Runs the Diagnostic test
     *
     * Returns:
     *      false if there were issues running the test (test failures are not considered issues),
     *      true otherwise.
     */
    template<class T> bool RunTest(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Clean up any resources allocated by this object, including memory and
     * file handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Check whether the test has passed for all GPUs and sets the pass/fail result for each GPU.
     * Called after test is finished.
     *
     * Returns: true if the test passed for all gpus, false otherwise.
     */
    bool CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime, const std::vector<int> &errorCount);

    /*************************************************************************/
    /*
     * Checks for errors in the gpu temperature
     *
     *
     * Returns: true if the test passed, false otherwise.
     */
    bool CheckGpuTemperature(GpuBurnDevice *device, std::vector<DcgmError> &errorList,
                            timelib64_t startTime, timelib64_t earliestStopTime, bool testFinished);

    /*************************************************************************/
    /*
     * Checks for single / double bit errors as well as XID errors.
     *
     *
     * Returns: true if the test passed, false otherwise.
     */
    bool CheckLwmlEvents(GpuBurnDevice *device, std::vector<DcgmError> &errorList,
                        timelib64_t startTime, timelib64_t earliestStopTime);


    /*************************************************************************/
    TestParameters               *m_testParameters;          /* Parameters for this test, passed in from the framework.
                                                                Set when the go() method is called. DO NOT FREE */
    std::vector<GpuBurnDevice *> m_device;                   /* Per-device data */
    bool                         m_lwmlInitialized;          /* Has lwmlInit been called? */
    DcgmRecorder                 m_dcgmRecorder;
    bool                         m_dcgmRecorderInitialized;  /* Has DcgmRecorder been initialized? */
    bool                         m_dcgmCommErrorOclwrred;    /* Has there been a communication error with DCGM? */

    /* Cached parameters read from testParameters */
    double                       m_testDuration;             /* test length, in seconds */
    double                       m_sbeFailureThreshold;      /* Failure threshold for SBEs. Below this it's a warning */
    bool                         m_useDoubles;               /* true if we should use doubles instead of floats */
};

#endif //GPUBURNPLUGIN_H

