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

class GlobalFabricManager;

class GlobalFmHaMgr
{
public:
    GlobalFmHaMgr(GlobalFabricManager *pGfm, char *stateFilename);
    ~GlobalFmHaMgr();

    bool saveStates(void);
    bool loadStates(void);
    bool validateStates(void);
    bool validateAndLoadState(void);

    bool saveStateFile(void);
    bool loadStateFile(void);
    void updateHaInitDoneState(void);    
    bool isInitDone(void) { return mInitDone; };

private:
    std::string mStateFile;
    bool mInitDone;
    timelib64_t mStartTime;

    GlobalFabricManager *mpGfm;
    fabricmanagerHA::fmHaState mFmHaState;
    FMTimer *mRestartTimer;

    static void restartTimerCB(void* ctx);
    void onRestartTimerExpiry(void);
    bool checkRestartDone(void);
    void dumpState( void );
};
