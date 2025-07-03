#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <google/protobuf/text_format.h>

#include "DcgmFMCommon.h"
#include "DcgmFMError.h"
#include "DcgmLogging.h"
#include "DcgmFMAutoLock.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFabricParser.h"
#include "DcgmGFMFabricPartitionMgr.h"
#include "DcgmGlobalFabricManagerHaMgr.h"

DcgmGlobalFabricManagerHaMgr::DcgmGlobalFabricManagerHaMgr(bool shared_fabric,
                                                           char *stateFilename,
                                                           DcgmGlobalFabricManager *pGfm)
{
    mSharedFabric = shared_fabric;
    mpGfm = pGfm;
    mInitDone = true;
    mStartTime = 0;

    if (stateFilename != NULL)
    {
        mStateFile = stateFilename;
    }
    else
    {
        mStateFile = DEFAULT_STATE_FILE;
    }

    if (mpGfm->isRestart())
    {
        mInitDone = false;
        // create the restart timer, to make sure all clients finish restart
        // before HA_MGR_RESTART_TIMEOUT
        mRestartTimer = new DcgmFMTimer( DcgmGlobalFabricManagerHaMgr::restartTimerCB, this );
    }
    else
    {
        mRestartTimer = NULL;
    }
}

DcgmGlobalFabricManagerHaMgr::~DcgmGlobalFabricManagerHaMgr()
{
    if (mRestartTimer) delete mRestartTimer;
}

// gather HA states and construct mStateFile protobuf
bool
DcgmGlobalFabricManagerHaMgr::saveStates(void)
{
    fabricmanagerHA::sharedFabricPartiontionInfo *pSharedFabricState;
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

    if (mSharedFabric == true)
    {
        pSharedFabricState = new fabricmanagerHA::sharedFabricPartiontionInfo;

        // gather HA states for shared lwswitch virtualization mode
        if (mpGfm && mpGfm->mGfmPartitionMgr &&
            (mpGfm->mGfmPartitionMgr->getSharedFabricHaState(*pSharedFabricState)))
        {
            mFmHaState.set_allocated_sharedfabricstate(pSharedFabricState);
        }
    }
    else
    {
        // HA not supported on baremetal and full passthrough yet
    }

    return saveStateFile();
}

// validate saved HA states
bool
DcgmGlobalFabricManagerHaMgr::validateStates(void)
{
    bool rc = true;

    if (mSharedFabric == true)
    {
        // validate HA states for shared lwswitch virtualization mode
        if (mpGfm && mpGfm->mGfmPartitionMgr &&
            mFmHaState.has_sharedfabricstate())
        {
            rc = mpGfm->mGfmPartitionMgr->validateSharedFabricHaState(mFmHaState.sharedfabricstate());
        }
    }
    else
    {
        // HA not supported for baremetal and full passthrough yet
    }

    PRINT_INFO("", "HA states are validated.");
    return true;
}

// load saved HA states to runtime data structures
bool
DcgmGlobalFabricManagerHaMgr::loadStates(void)
{
    bool rc = true;

    if (mSharedFabric == true)
    {
        // load HA states for shared lwswitch virtualization mode
        if (mpGfm && mpGfm->mGfmPartitionMgr &&
            mFmHaState.has_sharedfabricstate())
        {
            rc = mpGfm->mGfmPartitionMgr->loadSharedFabricHaState(mFmHaState.sharedfabricstate());
        }
    }
    else
    {
        // HA not supported for baremetal and full passthrough yet
    }

    PRINT_INFO("", "HA states are loaded.");
    return rc;
}

// load saved HA states to runtime data structures
bool
DcgmGlobalFabricManagerHaMgr::validateAndLoadState(void)
{
    if ( loadStateFile() == false ){
        PRINT_ERROR("%s", "Failed to load state file %s.", mStateFile.c_str());
        return false;
    }

    // validate and load saved states after restart
    if ( validateStates() == false ) {
        PRINT_ERROR("%s", "%s", "Failed to validate restart states.");
        return false;
    }

    if ( loadStates() == false ) {
        PRINT_ERROR("%s", "%s", "Failed to load restart states.");
        return false;
    }

    // start the restart time
    mStartTime = timelib_usecSince1970();
    mRestartTimer->start(1);

    return true;
}

// print HA states read from the file in debug mode
void
DcgmGlobalFabricManagerHaMgr::dumpState( void )
{
//#ifdef DEBUG
    std::string haStateText;

    google::protobuf::TextFormat::PrintToString(mFmHaState, &haStateText);
    PRINT_DEBUG("%s", "%s", haStateText.c_str());
//#endif
}

// Load HA states from persistent state file
bool
DcgmGlobalFabricManagerHaMgr::loadStateFile(void)
{
    // Read the fm saved states in protobuf binary file.
    std::fstream input(mStateFile.c_str(), std::ios::in | std::ios::binary);
    if ( !input )
    {
        PRINT_ERROR("%s", "Failed to open file %s.", mStateFile.c_str());
        return false;
    }
    else if ( !mFmHaState.ParseFromIstream(&input) )
    {
        PRINT_ERROR("%s", "Failed to parse file %s.", mStateFile.c_str());
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
        PRINT_ERROR("%s %s", "Saved platform %s is different from topology platform %s.",
                    mFmHaState.platformid().c_str(), platformId);
        FM_SYSLOG_ERR("Saved platform %s is different from topology platform %s.",
                      mFmHaState.platformid().c_str(), platformId);

        return false;
    }

    PRINT_INFO("%s %d %d %s %s", "Parsed state file %s successfully. version %d.%d, platformId: %s, timeStamp: %s.",
               mStateFile.c_str(), mFmHaState.majorversion(), mFmHaState.minorversion(),
               mFmHaState.has_platformid() ? mFmHaState.platformid().c_str() : "Not set",
               mFmHaState.has_timestamp() ? mFmHaState.timestamp().c_str() : "Not set");

    return true;
}

// save HA states to persistent state file
bool
DcgmGlobalFabricManagerHaMgr::saveStateFile(void)
{
    ofstream  outFile;
    outFile.open( mStateFile.c_str(), ios::binary );

    if ( outFile.is_open() == false )
    {
        PRINT_ERROR("%s %s", "Failed to open state file %s, error is %s.",
                    mStateFile.c_str(), strerror(errno));
        FM_SYSLOG_ERR("Failed to open state file %s, error is %s.",
                      mStateFile.c_str(), strerror(errno));
        return false;
    }

    // write the binary HA state file
    int   fileLength = mFmHaState.ByteSize();
    int   bytesWritten;
    char *bufToWrite = new char[fileLength];

    if ( bufToWrite == NULL )
    {
        PRINT_ERROR(" ", "Buffer is empty.");
        outFile.close();
        return false;
    }

    mFmHaState.SerializeToArray( bufToWrite, fileLength );
    outFile.write( bufToWrite, fileLength );

    delete[] bufToWrite;
    outFile.close();

    return true;
}

void
DcgmGlobalFabricManagerHaMgr::restartTimerCB(void* ctx)
{
    DcgmGlobalFabricManagerHaMgr* pObj = (DcgmGlobalFabricManagerHaMgr*)ctx;
    pObj->onRestartTimerExpiry();
}

void
DcgmGlobalFabricManagerHaMgr::onRestartTimerExpiry(void)
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
        ss << "Restart has timed out." << endl;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // restart the timer to check again
    mRestartTimer->restart();
}

// Check if restart done from all HA manager clients
// lwrrently only GFMFabricPartitionMgr
bool
DcgmGlobalFabricManagerHaMgr::checkRestartDone(void)
{
    return mpGfm->mGfmPartitionMgr->isInitDone();
}
