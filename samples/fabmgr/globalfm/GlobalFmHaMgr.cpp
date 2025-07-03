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
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <google/protobuf/text_format.h>

#include "FMCommonTypes.h"
#include "FMErrorCodesInternal.h"
#include "fm_log.h"
#include "FMAutoLock.h"
#include "GlobalFabricManager.h"
#include "GlobalFmFabricParser.h"
#include "GFMFabricPartitionMgr.h"
#include "GlobalFmHaMgr.h"

GlobalFmHaMgr::GlobalFmHaMgr(GlobalFabricManager *pGfm, char *stateFilename)
{
    mpGfm = pGfm;
    mInitDone = true;
    mStartTime = 0;
    mRestartTimer = NULL;

    if (stateFilename != NULL)
    {
        mStateFile = stateFilename;
    }
    else
    {
        mStateFile = DEFAULT_STATE_FILE;
    }

    if (mpGfm->isFabricModeRestart())
    {
        mInitDone = false;
        // create the restart timer, to make sure all clients finish restart
        // before HA_MGR_RESTART_TIMEOUT
        mRestartTimer = new FMTimer( GlobalFmHaMgr::restartTimerCB, this );
    }
}

GlobalFmHaMgr::~GlobalFmHaMgr()
{
    if (mRestartTimer) delete mRestartTimer;
}

// gather HA states and construct mStateFile protobuf
bool
GlobalFmHaMgr::saveStates(void)
{
    if (mpGfm == NULL) {
        return false;
    }

    // gather HA states only for Shared LWSwitch and vGPU based multitenancy modes
    if ((mpGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) && (mpGfm->getFabricMode() != FM_MODE_VGPU)) {
        return true;
    }

    const char *platformId = mpGfm->mpParser->getPlatformId();;

    mFmHaState.set_majorversion(HA_MAJOR_VERSION);
    mFmHaState.set_minorversion(HA_MINOR_VERSION);

    if (platformId)
    {
        mFmHaState.set_platformid(platformId);
    }
    else
    {
        // there is no platformId in the topology, use empty str in saved platformId
        // so that it will not match to anything unintentionally.
        mFmHaState.set_platformid("");
    }

    time_t lwrtime = time(NULL);;
    struct tm *loc_time = localtime(&lwrtime);
    mFmHaState.set_timestamp(asctime(loc_time));

    //
    // save a new copy shared fabric partition state information. Lwrrently
    // saved shared fabric partition state info will be freed automatically
    // as part of set_allocated_ call.
    //
    if (mpGfm->mGfmPartitionMgr)
    {
        fabricmanagerHA::sharedFabricPartiontionInfo *pSharedFabricState;
        pSharedFabricState = new fabricmanagerHA::sharedFabricPartiontionInfo;

        // gather HA states for shared lwswitch virtualization mode
        if (!mpGfm->mGfmPartitionMgr->getSharedFabricHaState(*pSharedFabricState)) {
            FM_LOG_ERROR("failed to get fabric manager partition state information");
            delete pSharedFabricState;
            return false;
        }

        mFmHaState.set_allocated_sharedfabricstate(pSharedFabricState);
    }

    if (saveStateFile() == false) {
        FM_LOG_ERROR("failed to save state of fabric manager");
        return false;
    }

    FM_LOG_DEBUG("successfully saved state of fabric manager");
    return true;
}

// validate saved HA states
bool
GlobalFmHaMgr::validateStates(void)
{
    bool rc = true;

    if (mpGfm == NULL) {
        return false;
    }

    // validate HA states only for Shared LWSwitch and vGPU based multitenancy modes
    if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU))
    {
        if (mpGfm->mGfmPartitionMgr && mFmHaState.has_sharedfabricstate())
        {
            rc = mpGfm->mGfmPartitionMgr->validateSharedFabricHaState(mFmHaState.sharedfabricstate());
        }
    }

    FM_LOG_INFO("successfully validated saved state for fabric manager restart");
    return rc;
}

// load saved HA states to runtime data structures
bool
GlobalFmHaMgr::loadStates(void)
{
    bool rc = true;

    if (mpGfm == NULL) {
        return false;
    }

    // load HA states only for Shared LWSwitch and vGPU based multitenancy modes
    if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU))
    {
        if (mpGfm->mGfmPartitionMgr && mFmHaState.has_sharedfabricstate())
        {
            rc = mpGfm->mGfmPartitionMgr->loadSharedFabricHaState(mFmHaState.sharedfabricstate());
        }
    }

    FM_LOG_INFO("successfully loaded saved state for fabric manager restart");
    return rc;
}

// load saved HA states to runtime data structures
bool
GlobalFmHaMgr::validateAndLoadState(void)
{
    if ( loadStateFile() == false ){
        FM_LOG_ERROR("failed to open/parse saved state file %s for fabric manager restart", mStateFile.c_str());
        return false;
    }

    // validate and load saved states after restart
    if ( validateStates() == false ) {
        FM_LOG_ERROR("failed to validate saved state file %s for fabric manager restart", mStateFile.c_str());
        return false;
    }

    if ( loadStates() == false ) {
       
        FM_LOG_INFO("failed to load saved state file %s for fabric manager restart", mStateFile.c_str());
        return false;
    }

    // start the restart time
    mStartTime = timelib_usecSince1970();
    mRestartTimer->start(1);

    return true;
}

// print HA states read from the file in debug mode
void
GlobalFmHaMgr::dumpState( void )
{
#ifdef DEBUG
    std::string haStateText;

    google::protobuf::TextFormat::PrintToString(mFmHaState, &haStateText);
    FM_LOG_DEBUG("%s", haStateText.c_str());
#endif
}

// Load HA states from persistent state file
bool
GlobalFmHaMgr::loadStateFile(void)
{
    // Read the fm saved states in protobuf binary file.
    std::fstream input(mStateFile.c_str(), std::ios::in | std::ios::binary);
    if ( !input )
    {
        FM_LOG_ERROR("failed to open fabric manager saved state file %s.", mStateFile.c_str());
        return false;
    }
    else if ( !mFmHaState.ParseFromIstream(&input) )
    {
        FM_LOG_ERROR("failed to parse fabric manager saved state file %s.", mStateFile.c_str());
        input.close();
        return false;
    }

    input.close();
    dumpState();

    // platform check
    // if the save HA platform is the same as the one loaded from the topology
    const char *platformId = mpGfm->mpParser->getPlatformId();
    if (mFmHaState.has_platformid() && (platformId != NULL) &&
        (mFmHaState.platformid() != platformId))
    {
        std::ostringstream ss;
        ss << "platform id: " << mFmHaState.platformid().c_str() << " read from saved state file is not matching to the platform id: " 
           << platformId << " specified in fabric topology file";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        return false;
    }

    FM_LOG_INFO("parsed saved state file %s successfully. version: %d.%d, platform id: %s, time stamp: %s.",
                mStateFile.c_str(), mFmHaState.majorversion(), mFmHaState.minorversion(),
                mFmHaState.has_platformid() ? mFmHaState.platformid().c_str() : "Not set",
                mFmHaState.has_timestamp() ? mFmHaState.timestamp().c_str() : "Not set");

    return true;
}

// save HA states to persistent state file
bool
GlobalFmHaMgr::saveStateFile(void)
{
    ofstream  outFile;
    outFile.open( mStateFile.c_str(), ios::binary );

    if ( outFile.is_open() == false )
    {
        FM_LOG_ERROR("failed to open file %s to save fabric manager states for restart, error: %s.",
                     mStateFile.c_str(), strerror(errno));
        return false;
    }

    // write the binary HA state file
    int   fileLength = mFmHaState.ByteSize();
    int   bytesWritten;
    char *bufToWrite = new char[fileLength];

    if ( bufToWrite == NULL )
    {
        FM_LOG_ERROR("failed to allocate memory for saving fabric manager state for restart");
        outFile.close();
        return false;
    }

    mFmHaState.SerializeToArray( bufToWrite, fileLength );
    outFile.write( bufToWrite, fileLength );

    delete[] bufToWrite;
    outFile.close();

    return true;
}

/******************************************************************************************
 This method will be called by each HA clients on-demand to update the over-all HA manager 
 initialization state once the corrsponding client's HA state is reloaded.
 ******************************************************************************************/
void
GlobalFmHaMgr::updateHaInitDoneState(void)
{
    // poll our clients for their initialization done state.

    // Note: The HA done timer will be running and will be stopped in next oclwrrence.
    mInitDone = checkRestartDone() ? true : false;
}

void
GlobalFmHaMgr::restartTimerCB(void* ctx)
{
    GlobalFmHaMgr* pObj = (GlobalFmHaMgr*)ctx;
    pObj->onRestartTimerExpiry();
}

void
GlobalFmHaMgr::onRestartTimerExpiry(void)
{
    mInitDone = checkRestartDone() ? true : false;

    // restart has completed
    if (mInitDone)
    {
        mRestartTimer->stop();
        return;
    }

    // check if restart has timed out
    timelib64_t timeNow = timelib_usecSince1970();

    if ((timeNow - mStartTime) > HA_MGR_RESTART_TIMEOUT*1000000)
    {
        // restarted timed out
        std::ostringstream ss;
        ss << "timeout oclwrred while waiting for list of lwrrently activated fabric partitions for fabric manager restart" << endl;
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // restart the timer to check again
    mRestartTimer->restart();
}

// Check if restart done from all HA manager clients
// lwrrently only GFMFabricPartitionMgr
bool
GlobalFmHaMgr::checkRestartDone(void)
{
    return mpGfm->mGfmPartitionMgr->isInitDone();
}
