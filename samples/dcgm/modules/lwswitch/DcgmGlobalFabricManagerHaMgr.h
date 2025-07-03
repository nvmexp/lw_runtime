#pragma once

/*****************************************************************************/
/*  Implement all the Global Fabric Manager HA (High Availability)           */
/*  related interfaces/methods.                                              */
/*****************************************************************************/

#include "lwos.h"
#include "fabricmanagerHA.pb.h"

#define HA_MAJOR_VERSION  1
#define HA_MINOR_VERSION  0

#define DEFAULT_STATE_FILE      "/tmp/fabricmanager.state"
#define HA_MGR_RESTART_TIMEOUT  60 // 60 seconds

class DcgmGlobalFabricManager;

class DcgmGlobalFabricManagerHaMgr
{
public:
    DcgmGlobalFabricManagerHaMgr(bool shared_fabric, char *stateFilename,
                                 DcgmGlobalFabricManager *pGfm);
    ~DcgmGlobalFabricManagerHaMgr();

    bool saveStates(void);
    bool loadStates(void);
    bool validateStates(void);
    bool validateAndLoadState(void);

    bool saveStateFile(void);
    bool loadStateFile(void);
    bool isInitDone(void) { return mInitDone; };

private:
    bool mSharedFabric;
    std::string mStateFile;
    bool mInitDone;
    timelib64_t mStartTime;

    DcgmGlobalFabricManager *mpGfm;
    fabricmanagerHA::fmHaState mFmHaState;
    DcgmFMTimer *mRestartTimer;

    static void restartTimerCB(void* ctx);
    void onRestartTimerExpiry(void);
    bool checkRestartDone(void);
    void dumpState( void );
};
