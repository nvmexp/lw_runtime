#pragma once

#include <signal.h>
#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmSwitchInterface.h"
#include "DcgmLocalFabricManager.h"

#define FATAL_ERROR_POLLING_TIMEOUT             10 // seconds
#define STATS_REPORTING_INTERVAL_CNT            3  // every 30 seconds
#define NON_FATAL_ERROR_REPORTING_INTERVAL_CNT  6  // every 60 seconds

/*****************************************************************************/
/*  Local Fabric Manager stats and errors                                    */
/*  Act as an interface to collect stats/error/ report from LWSwitch, RM and */
/*  LWLinkCoreLib driver. This class starts a thread and poll() the driver   */
/*  interface for stats/error and report the same to Global FM.              */
/*  Global FM can on demand request stats/error from Local FM and the        */
/*  corresponding GPB message handler uses this class's methods to retrieve  */
/*  the current stats information.                                           */
/*****************************************************************************/

class DcgmLocalStatsReporter : public LwcmThread
{
public:
    DcgmLocalStatsReporter(DcgmLocalFabricManagerControl *pLfm,
                           DcgmLWSwitchPhyIdToFdInfoMap &switchIdToFdInfoMap,
                           DcgmLFMLWLinkDevRepo *linkDevRepo);
    ~DcgmLocalStatsReporter();

    // virtual function from LwcmThread
    virtual void run();

    void getSwitchErrors(uint32_t physicalId,
                         uint32_t errorMask,
                         std::queue < SwitchError_struct * > *errQ);

    lwswitch::fmMessage* buildSwitchErrorMsg(uint32_t errorMask,
                                             std::queue < SwitchError_struct * > *errQ );

    void getSwitchInternalLatency(uint32_t physicalId,
                                  std::queue < SwitchLatency_struct * > *latencyQ);

    void getSwitchLwlinkCounter(uint32_t physicalId,
                                std::queue < LwlinkCounter_struct * > *counterQ);

    lwswitch::fmMessage* buildSwitchStatsMsg(std::queue < SwitchLatency_struct * > *latencyQ,
                                             std::queue < LwlinkCounter_struct * > *counterQ);

private:
    dcgmReturn_t registerGpuLWLinkErrorWatch(void);
    dcgmReturn_t unRegisterGpuLWLinkErrorWatch(void);
    void checkForGpuLWLinkError(void);
    void processGpuLWLinkError(unsigned int gpuIndex,dcgmcm_sample_t &lwrrentSample);
    void reportGpuLWLinkRecoveryError(unsigned int gpuIndex);
    void reportGpuLWLinkFatalError(unsigned int gpuIndex);
    void reportSwitchLWLinkRecoveryError(uint32_t physicalId,
                                         LWSWITCH_ERROR &switchError);
    void reportFatalErrors(uint32_t physicalId);
    void reportNonFatalErrors(void);
    void reportStats(void);

    DcgmLWSwitchPhyIdToFdInfoMap mSwitchIdToFdInfoMap;
    DcgmLocalFabricManagerControl *mpLfm;
    DcgmLFMLWLinkDevRepo *mLWLinkDevRepo;
    std::vector<unsigned int> mDcgmGpuIds;

    typedef std::map <unsigned int, dcgmcm_sample_t> GpuIDToDcgmSampleMap;
    GpuIDToDcgmSampleMap mGpuLWLinkErrSample;

    int mNonFatalErrorReportCnt;
    int mStatsReportCnt;
};
