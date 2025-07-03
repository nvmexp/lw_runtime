
#include <iostream>
#include <stdexcept>
#include <stdlib.h>

#include "logging.h"
#include "DcgmFMAutoLock.h"
#include "DcgmFMReqTracker.h"
#include "DcgmFMConnectionBase.h"

DcgmFMReqTracker::DcgmFMReqTracker(DcgmFMConnectionBase *parent)
    :mRspTimeIntrvl(FM_REQ_RESP_TIME_INTRVL),
     mRspTimeThreshold(FM_REQ_RESP_TIME_THRESHOLD), mMissedRspCnt(0),
     mParent(parent)
{
    // create response checking timer object
    mRspTimer = new DcgmFMTimer( DcgmFMReqTracker::responseTimerCB, this );

    // lock is required as the pending request map will be accessed from
    // timer context (thread1) and from connection context (thread2)
    lwosInitializeCriticalSection( &mLock );

    mLastRspRcvd = time( 0 );
}

DcgmFMReqTracker::~DcgmFMReqTracker()
{
    DcgmFMAutoLock lock(mLock);

    mPendingReqs.clear();
    delete mRspTimer;
    mRspTimer = NULL;
    lwosDeleteCriticalSection( &mLock );
}

void
DcgmFMReqTracker::startReqTracking(void)
{
    DcgmFMAutoLock lock(mLock);

    mRspTimer->start( mRspTimeIntrvl );
    mMissedRspCnt = 0;
    mLastRspRcvd = time( 0 );

    PRINT_DEBUG( "%d", "Start tracking on connectionId %d.",
                 mParent->mpConnection->GetConnectionId() );
}

void
DcgmFMReqTracker::stopReqTracking(void)
{
    DcgmFMAutoLock lock(mLock);

    mPendingReqs.clear();
    mRspTimer->stop( );

    PRINT_DEBUG( "%d", "Stop tracking on connectionId %d.",
                 mParent->mpConnection->GetConnectionId() );
}

int
DcgmFMReqTracker::addRequest(unsigned int requestId)
{
    DcgmFMAutoLock lock(mLock);

    if ( mPendingReqs.count(requestId) ) {
        PRINT_ERROR( "%d %d", "reguestId %d  already exists on connectionId %d.\n",
                     requestId, mParent->mpConnection->GetConnectionId() );
        return -1;
    }

    if ( mPendingReqs.empty() ) {
        // reset last interval to account for pause between requests
        mLastRspRcvd = time( 0 );
    }

    // just insert into the map
    mPendingReqs.insert( requestId );

    return 0;
}

int
DcgmFMReqTracker::removeRequest(unsigned int requestId)
{
    DcgmFMAutoLock lock(mLock);
    RequestMap::iterator it;

    if ( !mPendingReqs.count(requestId) ) {
        // specified request don't exists
        PRINT_DEBUG( "%d %d", "reguestId %d not found on connectionId %d.\n",
                     requestId, mParent->mpConnection->GetConnectionId() );
        return -1;
    }

    it = mPendingReqs.find( requestId );
    if (it != mPendingReqs.end()) {
        mPendingReqs.erase(it);
        PRINT_DEBUG("%d", "Remove requestId %d", requestId);

        // removing request indicates that we are receiving response. so
        // update current time as last response received time.
        if ( mPendingReqs.empty() ) {
            // reset last interval to account for pause between requests
            mLastRspRcvd = time( 0 );
        }
    }

    return 0;    
}
void DcgmFMReqTracker::dumpPending()
{
    RequestMap::iterator it;
    for ( it = mPendingReqs.begin(); it != mPendingReqs.end(); it++) {
        PRINT_DEBUG("%d", "Pending request ID=%d\n", *it);
    }
}


void
DcgmFMReqTracker::responseTimerCB(void *ctx)
{
    DcgmFMReqTracker *pObj = (DcgmFMReqTracker*)ctx;
    pObj->runResponseCheck();
}

void
DcgmFMReqTracker::runResponseCheck(void)
{
    DcgmFMAutoLock lock(mLock);

    if ( mPendingReqs.empty() ) {
        // nothing specific to do, simply reschedule the timer
        mMissedRspCnt = 0;
        mRspTimer->restart();
        return;
    }

    // check for ack timeout
    time_t timeNow = time(0);
    double elapsedSecs = difftime( timeNow, mLastRspRcvd );

    if ( elapsedSecs >= mRspTimeIntrvl ) {
        mMissedRspCnt++;
        PRINT_DEBUG( "%d %d", "Req Tracker(connectionId=%d) : No response from peer Fabric Node. Count: %d \n", 
                        mParent->mpConnection->GetConnectionId(), mMissedRspCnt );

        dumpPending();
    }

    // check for threshold 
    if  ( mMissedRspCnt >= mRspTimeThreshold ) {
        PRINT_ERROR( "%d, %d", 
		     "Req Tracker : No response from peer Fabric Node. Peer may be unreachable/down, mMissedRspCnt %d, mPendingReqs.size %d.",
		     mMissedRspCnt, (int)mPendingReqs.size());

        // signal request timeout to parent
        mPendingReqs.clear();
        mParent->processRequestTimeout();
    } else {
        // restart ack timer
        mRspTimer->restart();
    }
}

