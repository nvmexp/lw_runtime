/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
 #pragma once

#include <time.h>
#include <map>
#include <set>


/*****************************************************************************/
/* Implements tracking and timing out of outstanding FM requests             */
/*****************************************************************************/

/*
 * Ideally this request tracking could have been part of FM communication classes
 * itself. But that requires more fundamental changes to FM.
 * 
 * The tracking is implemented as a separate class as this is required for 
 * FMClientConnection and FMServerConnection as well.
 *
 */

class FMTimer;
class FMConnectionBase;

class FMReqTracker
{

public:
    FMReqTracker(FMConnectionBase *parent, uint32_t rspTimeIntrvl, uint32_t mRspTimeThreshold);
    ~FMReqTracker();

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
    FMTimer *mRspTimer;
    LWOSCriticalSection mLock;

    time_t mLastRspRcvd;
    uint32_t mRspTimeIntrvl;    // number of misses
    uint32_t mRspTimeThreshold; // total number of counts to miss
    uint32_t mMissedRspCnt;

    FMConnectionBase *mParent;
};
