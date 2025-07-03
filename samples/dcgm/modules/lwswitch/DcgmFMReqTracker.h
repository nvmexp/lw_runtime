#pragma once

#include <time.h>
#include <map>
#include <set>


/*****************************************************************************/
/* Implements tracking and timing out of outstanding FM requests             */
/*****************************************************************************/

/*
 * Ideally this request tracking could have been part of DCGM communication classes
 * itself. But that requires more fundamental changes to DCGM.
 * 
 * The tracking is implemented as a separate class as this is required for 
 * LWCMClientConnection and LWCMServerConnection as well.
 *
 */

class DcgmFMTimer;
class DcgmFMConnectionBase;

#define FM_REQ_RESP_TIME_INTRVL 20 // seconds, to accommodate worse case when all GPUs
                                   // are attach/detach request response time.
#define FM_REQ_RESP_TIME_THRESHOLD 4 // total number of seconds

class DcgmFMReqTracker
{

public:
    DcgmFMReqTracker(DcgmFMConnectionBase *parent);
    ~DcgmFMReqTracker();

    // upper layer class this when the socket is connected/disconnected
    void startReqTracking(void);
    void stopReqTracking(void);

    // interface to add/remove a request from tracking
    // calling removeRequest indicates that the channel is proceeding and getting responses.
    int addRequest(unsigned int requestId);
    int removeRequest(unsigned int requestId);

private:
    void dumpPending();
    static void responseTimerCB(void *ctx);
    void runResponseCheck(void);

    typedef std::set <unsigned int> RequestMap;
    RequestMap mPendingReqs;
    DcgmFMTimer *mRspTimer;
    LWOSCriticalSection mLock;

    time_t mLastRspRcvd;
    const int mRspTimeIntrvl; // number of misses
    const int mRspTimeThreshold; // total number of counts to miss
    int mMissedRspCnt;

    DcgmFMConnectionBase *mParent;
};
