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

#include "UtilOS.h"
#include <lwtypes.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "gtest/gtest.h"

/**
 * Sleep for set amount of miliseconds
 */
void osSleep(LwU32 ms)
{
    usleep(ms*1000);
}

/**
 * Returns true if current process is root/elevated
 */
bool osIsUserAdmin()
{
    uid_t uid = 0;
    uid = geteuid();
    return uid == 0;
}

void osInit()
{

}

void osDestroy()
{

}

/**
 * Removes root/elevated privileges from the process
 */
void osDropAdminPrivileges()
{
    uid_t uid = 0;

    ASSERT_TRUE(osIsUserAdmin())
        << "Must be admin/root to execute this code";

    uid = getuid();
    if (uid == 0) uid = 0x1234;

    ASSERT_EQ(0, seteuid(uid))
        << "Decrease privileges failed";
}

/**
 * Returns root/elevated privileges to the process
 */
void osRestoreAdminPrivileges()
{
    ASSERT_EQ(0, seteuid(0))
        << "Restore privileges failed";
}
/**
 * API to start process
 */
LwU32 osStartProcess(ApplicationParams *pLaunchAppParams, LwU64 *pProcessHandle)
{
    EXPECT_TRUE(0)
        << "osStartProcess() is NOT implemented";

    return -1;
}

/**
 * API to stop process
 */
void osStopProcess(string processName, LwU64 hProcess)
{
    FAIL()
        << "osStopProcess() is NOT implemented";

    return;
}

/**
 * API to change LWSwitch device state (Enable / Disable / Restart)
 */
LwU32 osUpdateLWSwitchDeviceState(LWSWITCH_DEVICE_STATE_UPDATE devState, LwU32 deviceId, string dbdf)
{
    FAIL()
        << "osUpdateLWSwitchDeviceState() is NOT implemented";

    return -1;
}

void osAddLWSwitchI2CAdapter(LwU32 *adapters, LwU32 length, string dbdf, LwU32 *numI2cAdapters)
{
    return;
}

