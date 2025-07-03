/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */


#ifndef _UTILOS_H_
#define _UTILOS_H_

#include "lwtypes.h"
#include "lwstatus.h"
#include "gtest/gtest.h"
#include "gtest/GtestUtil.h"
#include <string.h>

#define BUFFER_SIZE                                         300

using namespace std;

enum LWSWITCH_DEVICE_STATE_UPDATE
{
    LWSWITCH_DEVICE_STATE_UPDATE_ENABLE_DEVICE              = 0x1,
    LWSWITCH_DEVICE_STATE_UPDATE_DISABLE_DEVICE             = 0x2,
    LWSWITCH_DEVICE_STATE_UPDATE_RESTART_DEVICE             = 0x3
};

struct Application
{
    string      name;
    string      command;
    string      parameters;
    string      directory;

    Application() : name(""), command(""), parameters(""), directory("")
    {
    }

    Application(string      Name,
                string      AppCmd = "",
                string      AppParams = "",
                string      AppDir = "") :
                    name(Name), command(AppCmd),
                    parameters(AppParams),
                    directory(AppDir)
    {
    }

};

struct ApplicationParams
{
    Application app;
    // TODO: Add parameters for launch delay configuration.

    ApplicationParams() : app()
    {
    }
};

void osSleep(LwU32 ms);
bool osIsUserAdmin();
void osDropAdminPrivileges();
void osRestoreAdminPrivileges();

// Process handlers
LwU32 osStartProcess(ApplicationParams *pLaunchAppParams, LwU64 *pProcessHandle);
void osStopProcess(string processName, LwU64 hProcess);

// LWSwitch device state update support.
LwU32 osUpdateLWSwitchDeviceState(LWSWITCH_DEVICE_STATE_UPDATE devState, LwU32 deviceId, string dbdf);

void osAddLWSwitchI2CAdapter(LwU32 *adapters, LwU32 length, string dbdf, LwU32 *numI2cAdapters);

#endif // _UTILOS_H_
