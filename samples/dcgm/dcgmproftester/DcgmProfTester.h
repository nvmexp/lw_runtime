#pragma once

#include "dcgm_structs.h"
#include "lwca.h"
#include <vector>
#include <string>

/*****************************************************************************/
class DcgmProfTester
{
public:
    /*************************************************************************/
    /* ctor/dtor */
    DcgmProfTester();
    ~DcgmProfTester();

    /*************************************************************************/
    

    /*************************************************************************/
    /*
     * Process the command line and initialize lwca to be able to run
     * 
     * Returns 0 on success. !0 on failure.
     */
    dcgmReturn_t Init(int argc, char *argv[]);

    /*************************************************************************/
    /*
     * Run the tests that were specified on the command line
     * 
     * Returns 0 on success. !0 on failure.
     */
    int RunTests(void);

    /*************************************************************************/

private:
    /*************************************************************************/
    /* The following functions return DCGM_ST_OK on success and other DCGM_ST_* enums on failure */
    dcgmReturn_t LoadLwdaModule(void);
    dcgmReturn_t LwdaInit(void);
    dcgmReturn_t DcgmInit(void);
    dcgmReturn_t CreateDcgmGroups(void);
    dcgmReturn_t WatchFields(long long updateIntervalUsec);
    dcgmReturn_t UnwatchFields(void);
    dcgmReturn_t GetLatestDcgmValue(dcgmFieldValue_v1 *value);

    /* Subtest methods */
    dcgmReturn_t BeginSubtest(std::string testTitle, std::string testTag, bool isLinearTest);
    dcgmReturn_t EndSubtest(void);
    dcgmReturn_t AppendSubtestRecord(double generatedValue, double dcgmValue);
    
    /*************************************************************************/
    /*
     * Process the command line from the program. Returns DCGM_ST_OK on success. !0 on failure.
     */
    dcgmReturn_t ParseCommandLine(int argc, char *argv[]);

    /*************************************************************************/
    /* Individual subtests */
    int RunSubtestGrActivity(void);
    int RunSubtestSmActivity(void);
    int RunSubtestSmOclwpancy(void);
    int RunSubtestPcieBandwidth(void);
    int RunSubtestDramUtil(void);
    int RunSubtestGemmUtil(void);
    int RunSubtestLwLinkBandwidth(void);
    int RunSubtestSmOclwpancyTargetMax(void);

    /*************************************************************************/
    /*
     * Method to get the DCGM and Lwca ordinal for our best LwLink peer
     */
    dcgmReturn_t HelperGetBestLwLinkPeer(unsigned int *peerGpuId, LWdevice *peerLwdaOrdinal);

    /*************************************************************************/
    /*
     * Method to run a sleep kernel on a given number of SMs
     */
    int RunSleepKernel(unsigned int numSms, unsigned int threadsPerSm, unsigned int runForUsec);

    /*************************************************************************/

    int m_maxThreadsPerMultiProcessor; /* Maximum number of threads per SM */
    int m_multiProcessorCount; /* Number of SMs on the GPU */
    int m_sharedMemPerMultiprocessor; /* Number of bytes of shared memory available per SM */
    int m_computeCapabilityMajor; /* Major Compute capability of the GPU */
    int m_computeCapabilityMinor; /* Minor Compute capability of the GPU */
    double m_computeCapability; /* Combined compute capability of the GPU like 7.5 */
    int m_maxMemoryClockMhz; /* Maximum memory clock in MHz */
    int m_memoryBusWidth;    /* Memory bus width in bits. This is used to callwlate m_maximumMemBandwidth */
    double m_maximumMemBandwidth; /* Maximum memory bandwidth in bytes per second */

    /* Test parameters */
    unsigned int m_testFieldId; /* Field ID we are testing. This will determine which subtest gets called. */
    double m_duration; /* Test duration in seconds */
    bool m_targetMaxValue; /* Whether (true) or not (false) we should just target the maximum value for
                              m_testFieldId instead of stair stepping from 0 to 100% */
    bool m_startDcgm;      /* Should we start DCGM and validate metrics against it? */
    bool m_dvsOutput;      /* Should we generate DVS stdout text? */

    unsigned int m_gpuId; /* DCGM GPU ID we are testing on */
    dcgmHandle_t m_dcgmHandle; /* Handle to the host engine */
    dcgmDeviceAttributes_t m_dcgmDeviceAttr; /* DCGM device attributes for this GPU */
    bool m_dcgmIsInitialized; /* Have we started DCGM? */
    dcgmGpuGrp_t m_groupId;   /* Group of GPUs we are watching */
    dcgmFieldGrp_t m_fieldGroupId; /* Fields we are watching for m_groupId */

    LWdevice m_lwdaDevice;   /* Lwca ordinal of the device to use */
    LWcontext m_lwdaContext; /* Lwca context */
    LWmodule m_lwdaModule;   /* Loaded .PTX file that belongs to m_lwdaContext */
    LWfunction m_lwFuncWaitNs; /* Pointer to the waitNs() lwca kernel */
    LWfunction m_lwFuncWaitCycles; /* Pointer to the waitCycles() lwca kernel */

    std::vector<dcgmFieldValue_v1> m_dcgmValues; /* Cache of values that have been fetched so far */
    long long m_sinceTimestamp; /* Cursor for fetching field values from DCGM */

    /* Subtest stuff */
    bool m_subtestInProgress; /* Is there lwrrently a subtest running? */
    bool m_subtestIsLinear;   /* Is the subtest linear (true) or static value (false) */
    std::string m_subtestTitle; /* Current subtest's display name */
    std::string m_subtestTag; /* Current subtest's tag - used for keys and the output filenames */
    std::vector<double> m_subtestDcgmValues; /* subtest DCGM values */
    std::vector<double> m_subtestGelwalues; /* subtest generated values */

    /*************************************************************************/
};

