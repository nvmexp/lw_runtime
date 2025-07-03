/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <iostream>
#include <stdexcept>
#include <stdlib.h>

#include "fm_log.h"
#include "FMAutoLock.h"
#include "FMReqTracker.h"
#include "FMConnectionBase.h"

FMReqTracker::FMReqTracker(FMConnectionBase *parent, uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold)
    : mRspTimeIntrvl(rspTimeIntrvl),
      mRspTimeThreshold(rspTimeThreshold),
      mMissedRspCnt(0),
      mParent(parent)
{
    // create response checking timer object
    mRspTimer = new FMTimer( FMReqTracker::responseTimerCB, this );

    // lock is required as the pending request map will be accessed from
    // timer context (thread1) and from connection context (thread2)
    lwosInitializeCriticalSection( &mLock );

    mLastRspRcvd = time( 0 );
}

FMReqTracker::~FMReqTracker()
{
    {
        FMAutoLock lock(mLock);
    
        mPendingReqs.clear();
        delete mRspTimer;
        mRspTimer = NULL;
    }
    lwosDeleteCriticalSection( &mLock );
}

void
FMReqTracker::startReqTracking(void)
{
    FMAutoLock lock(mLock);

    mRspTimer->start( mRspTimeIntrvl );
    mMissedRspCnt = 0;
    mLastRspRcvd = time( 0 );

    FM_LOG_DEBUG( "request tracker: starting request tracking for connectionId %d.",
                 mParent->mpConnection->GetConnectionId() );
}

void
FMReqTracker::stopReqTracking(void)
{
    FMAutoLock lock(mLock);

    mPendingReqs.clear();
    mRspTimer->stop( );

    FM_LOG_DEBUG( "request tracker: stopping request tracking for connectionId %d.",
                 mParent->mpConnection->GetConnectionId() );
}

int
FMReqTracker::addRequest(unsigned int requestId)
{
    FMAutoLock lock(mLock);

    if ( mPendingReqs.count(requestId) ) {
        FM_LOG_ERROR( "request tracker: unable to add request for tracking as specified request id %d for connection id %d already exists",
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
FMReqTracker::removeRequest(unsigned int requestId)
{
    FMAutoLock lock(mLock);
    RequestMap::iterator it;

    if ( !mPendingReqs.count(requestId) ) {
        // specified request don't exists
        FM_LOG_DEBUG( "request tracker: remove request failed as requestId %d not found for connectionId %d.\n",
                     requestId, mParent->mpConnection->GetConnectionId() );
        return -1;
    }

    it = mPendingReqs.find( requestId );
    if (it != mPendingReqs.end()) {
        mPendingReqs.erase(it);
        // removing request indicates that we are receiving response. so
        // update current time as last response received time.
        mLastRspRcvd = time( 0 );
        // we are getting responses. reset our missed response counter as we need to
        // get conselwtive misses, not interleaved.
        mMissedRspCnt = 0;
    }

    return 0;    
}
void FMReqTracker::dumpPending()
{
    RequestMap::iterator it;
    for ( it = mPendingReqs.begin(); it != mPendingReqs.end(); it++) {
        FM_LOG_DEBUG("request tracker: pending request id=%d\n", *it);
    }
}


void
FMReqTracker::responseTimerCB(void *ctx)
{
    FMReqTracker *pObj = (FMReqTracker*)ctx;
    pObj->runResponseCheck();
}

void
FMReqTracker::runResponseCheck(void)
{
    FMAutoLock lock(mLock);

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
        FM_LOG_DEBUG( "request tracker: no response from local fabric manager for connection id %d missed count: %d \n",
                       mParent->mpConnection->GetConnectionId(), mMissedRspCnt );

        dumpPending();
    }

    // check for threshold 
    if  ( mMissedRspCnt >= mRspTimeThreshold ) {
        FM_LOG_ERROR( "request tracker: request tracking timeout elapsed as local fabric manager is not responding, total number of pending requests %d.",
                      (int)mPendingReqs.size() );

        // signal request timeout to parent
        mPendingReqs.clear();
        mParent->processRequestTimeout();
    } else {
        // restart ack timer
        mRspTimer->restart();
    }
}

