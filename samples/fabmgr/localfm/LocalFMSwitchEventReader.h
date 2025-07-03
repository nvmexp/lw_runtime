/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
 
#pragma once

#include "FMCommonTypes.h"
#include "LocalFMSwitchInterface.h"
#include "LocalFMInbandEventHndlr.h"
#include "LocalFmErrorReporter.h"

extern "C"
{
    #include "lwswitch_user_api.h"
}

/*******************************************************************************/
/* Local Fabric Manager LWSwitch Event Reader                                  */
/* Abstraction class to wait for events from each LWSwitch device and dispatch */
/* dispatch them to respective class/handlers for further processing           */
/* This class starts a thread and poll() the all the LWSwitch device           */
/*******************************************************************************/

class LocalFabricManagerControl;
class LocalFMLWLinkDevRepo;

/* Switch Event polling timeout */
#define SWITCH_EVENT_POLLING_TIMEOUT 10  // in seconds

class LocalFMSwitchEventReader : public FmThread
{

public:
    LocalFMSwitchEventReader(LocalFabricManagerControl *pLfm,
                             LocalFMLWLinkDevRepo *linkDevRepo);

    ~LocalFMSwitchEventReader();

    // virtual function from FmThread
    virtual void run();

private:
    void dispatchLWSwitchEvents();

    LocalFMErrorReporter *mErrorReporter;
    LocalFMInbandEventHndlr *mInbandEventHndlr;
    static uint32 mSubscribedEvtList[];
    LocalFabricManagerControl *mpLfm;
    lwswitch_event *mSwitchEvents[LWSWITCH_MAX_DEVICES];
};

