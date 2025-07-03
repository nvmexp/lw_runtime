#ifndef MEMORYBANDWIDTH_H
#define MEMORYBANDWIDTH_H

#include "lwca.h"
#include "lwblas_v2.h"
#include "Plugin.h"
#include "lwml.h"
#include "LwvsDeviceList.h"
#include "DcgmRecorder.h"
#include "timelib.h"
#include "DcgmError.h"

#define MEMBW_MAX_DEVICES 32 /* Maximum number of devices to run this on conlwrrently */
#define REAL int
#define SI_TEST_COUNT 1
#define MAXTIMES 10000

/*****************************************************************************/
/* Struct for a single memory bandwidth device */
typedef struct membw_device_t
{
    //int lwdaDeviceIdx;    /* Which lwca device index to run on. -1 = unspecified */   
    LWdevice lwDevice;    /* Lwca Device identifier */
    LWcontext lwContext;  /* Lwca Context */

    lwmlDevice_t lwmlDevice; /* LWML handle to the device we're using */
    unsigned int dcgmDeviceIndex; /* LWML device index of the device we're using */

    LwvsDevice *lwvsDevice; /* LWVS device object for controlling/querying this device */
    
    /* Lwca handles and allocations */
    LWevent      lwda_timer[MAXTIMES][SI_TEST_COUNT+1]; /* +1 is for the final counter */
	LWstream     streamA, streamB;
    REAL        *d_a, *d_b, *d_c[SI_TEST_COUNT];
    LWmodule lwModule; /* Loaded module from our PTX String */
    LWfunction lwFuncSetArray; /* Pointer to SetArray() lwca function */
    LWfunction lwFuncStreamTriadCleanup; /* Pointer to StreamTriadCleanup() lwca function */
    LWfunction lwFuncStreamTriad[4][8][4][4]; /* Pointer to StreamTriad() lwca functions. 
                                                 Array indexes are: 
                                                 LD0-LD3, STR0-STR7, 1-4 (0-3), BLK0-BLK3 */
    LWfunction lwFuncTriadOptimal; /* Optimal triad function */

    /* Lwca device properties */
    int maxThreadsPerMultiProcessor;
    int multiProcessorCount;
    int sharedMemPerMultiprocessor;
	
    /* Test performance statistics */
    double       avgtime[SI_TEST_COUNT];
	double       maxtime[SI_TEST_COUNT];
	double       mintime[SI_TEST_COUNT];
	double       mbPerSec[SI_TEST_COUNT];

    /* Variables used between different parts of the membw test. These were globals
       in the original arch program */
    double occ_kern[SI_TEST_COUNT];
    dim3 block_kern[SI_TEST_COUNT];
    dim3 grid_kern[SI_TEST_COUNT];
    int extra_kern[SI_TEST_COUNT];
    float ms_kern[SI_TEST_COUNT];
    double perf_kern[SI_TEST_COUNT];
    double opt_perf_kern[SI_TEST_COUNT];
    int i_kern[SI_TEST_COUNT];
    int j_kern[SI_TEST_COUNT];
    int s_kern[SI_TEST_COUNT];
    int best_lp_kern[SI_TEST_COUNT];
    int best_strd_kern[SI_TEST_COUNT];
    int best_lt_kern[SI_TEST_COUNT];
    int byte_kern[SI_TEST_COUNT];
    int shmem_kern[SI_TEST_COUNT];
	
} membw_device_t, *membw_device_p;

/*****************************************************************************/
class MemoryBandwidth
{
public:
    MemoryBandwidth(TestParameters *testParameters, Plugin *plugin);
    ~MemoryBandwidth();

    /*************************************************************************/
    /*
     * Clean up any resources used by this object, freeing all memory and closing
     * all handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Run memory bandwidth tests
     *
     * Returns 0 on success (this does not indicate that the plugin passed, only that the plugin ran without issues)
     *        <0 on plugin / test initialization failure
     *
     */
    int Run(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/

private:
    /*************************************************************************/
    /*
     * Initialize the parts of lwml needed for this plugin to run
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int LwmlInit(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Initialize the parts of lwca needed for this plugin to run
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int LwdaInit(void);

    /*************************************************************************/
    /*
     * Check to see if serious errors oclwrred during the test run
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int CheckLwmlEvents(membw_device_p mbDevice, std::vector<DcgmError> &errorList,
                        timelib64_t startTime, timelib64_t stopTime);

    /*************************************************************************/
    /*
     * Check various statistics and device properties to determine if the plugin
     * has failed for the given GPU/membw_device or not.
     *
     * Returns: true if plugin has NOT failed
     *          false if plugin has failed
     *
     */
    bool CheckPassFailSingleGpu(membw_device_p device, std::vector<DcgmError> &errorList, 
                                timelib64_t startTime, timelib64_t endTime);
    
    /*************************************************************************/
    /*
     * Check various statistics and device properties to determine if the plugin
     * has passed for all GPUs.
     *
     * Returns: true if plugin has passed for all GPUs tested
     *          false if plugin has failed for at least one of the GPUs tested
     *
     */
    bool CheckPassFail(timelib64_t startTime, timelib64_t endTime);

    /*************************************************************************/
    /*
     * Run the bandwidth test on a GPU
     *
     * Returns: 0 if plugin has NOT failed
     *         <0 if plugin has failed 
     *
     */
    int PerformBandwidthTestOnGpu(int init_value, int element_count, int times, 
                                  membw_device_p gpu);
    
    /*************************************************************************/
    /*
     * Internal test function of PerformBandwidthTestOnGpu
     *
     * Returns: 0 if plugin has NOT failed
     *         <0 if plugin has failed 
     *
     */
    int TestKernels(membw_device_p gpu, int kern, int i, int j, int s, int inshmem, 
                    int ldtp , REAL init_value);
    
    /*************************************************************************/
    int InitializeDeviceMemory(membw_device_p gpu, int init_value, int element_count);
    int LoadLwdaModule(membw_device_p gpu);

    /*************************************************************************/

    Plugin *m_plugin; /* Which plugin we're running as part of. This will be
                         a pointer to a SmPerfPlugin instance */
    TestParameters *m_testParameters; /* Parameters for this test, passed in from
                                         the framework. DO NOT FREE */

    /* Cached parameters read from testParameters */
    double m_minBandwidth; /* Bandwidth we are trying to target in megabytes */
    int m_useDgemm;        /* Whether or not to use dgemm (or sgemm) 1=use dgemm */

    int m_lwmlInitialized; /* Has lwmlInit been called? 1=yes. 0=no */
    int m_shouldStop;      /* Global boolean variable to tell all workers to
                              exit early. 1=exit early. 0=keep running */

    DcgmRecorder *m_dcgmRecorder;
    int m_Ndevices; /* Number of device[] entries that are value */
    membw_device_t m_device[MEMBW_MAX_DEVICES]; /* Per-device data */

	int m_init_value;
	int m_times;
	int m_elements;
    double m_sbeFailureThreshold;
};

/*****************************************************************************/

#endif //MEMORYBANDWIDTH_H
