/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
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

void osAddLWSwitchI2CAdapter(LwU32 *adapters, LwU32 *adapterPortNum, LwU32 length, string dbdf, LwU32 *numI2cAdapters);

#endif // _UTILOS_H_
