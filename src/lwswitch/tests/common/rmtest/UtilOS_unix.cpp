/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "UtilOS.h"
#include <lwtypes.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <poll.h>
#include <sstream>
#include <fstream>
#include <string>
#include <dirent.h>

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

    uid = getuid();

    ASSERT_TRUE(0 == uid)
        << "Must be admin/root to execute this code";

    uid = 0x1234;

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

    switch (devState)
    {
        case LWSWITCH_DEVICE_STATE_UPDATE_ENABLE_DEVICE:
        {
            ofstream bind("/sys/bus/pci/drivers/lwpu-lwswitch/bind");
            bind << dbdf.c_str();
            bind.close();
            break;
        }
        case LWSWITCH_DEVICE_STATE_UPDATE_DISABLE_DEVICE:
        {
            ofstream unbind("/sys/bus/pci/drivers/lwpu-lwswitch/unbind");
            unbind << dbdf.c_str();
            unbind.close();
            break;
        }
        case LWSWITCH_DEVICE_STATE_UPDATE_RESTART_DEVICE:
        {
            ofstream unbindForRestart("/sys/bus/pci/drivers/lwpu-lwswitch/unbind");
            ofstream bindForRestart("/sys/bus/pci/drivers/lwpu-lwswitch/bind");

            unbindForRestart << dbdf.c_str();
            unbindForRestart.close();

            bindForRestart << dbdf.c_str();
            bindForRestart.close();
            break;
        }
        default:
        {
            // Return as Invalid argument.
            return -1;
        }
    }

    return 0;
}

#define LWSWITCH_I2C_DEV_DIR_FMT           "/sys/bus/i2c/devices/"
#define LWSWITCH_I2C_DEV_DIR_INFO_NAME_FMT "/name"
#define LWSWITCH_I2C_DELIMITER             "LWPU LWSwitch"
#define LWSWITCH_I2C_INFO_FMT              "LWPU LWSwitch i2c adapter %d at %02x:%02x.%x"

void osAddLWSwitchI2CAdapter(LwU32 *adapters, LwU32 *adapterPortNum, LwU32 arrLength, string dbdf, LwU32 *numI2cAdapters)
{
    DIR *dir_ptr;
    string bdf = dbdf.substr(5);
    struct dirent *dir_entry;
    LwU32 num_adapters = 0;
    char *bdf_str = const_cast<char*>(bdf.c_str());

    dir_ptr = opendir(LWSWITCH_I2C_DEV_DIR_FMT);
    if (!dir_ptr)
    {
        FAIL() << "Error opening " << LWSWITCH_I2C_DEV_DIR_FMT;
        return;
    }

    /*
     * Skip any leading '0' in the bdf string. The device bdf can
     * potentially have a leading 0, but the bdf entry for the I2C
     * adapter does not.
     */
    while ((*bdf_str == '0') &&
           (*bdf_str != '\0'))
    {
       bdf_str = bdf_str + 1;
    }

    while ((dir_entry = readdir(dir_ptr)) != NULL)
    {
        FILE *fp;
        char dir_path[256];
        ssize_t read;
        char *line;
        size_t len = 0;

        strcpy(dir_path, LWSWITCH_I2C_DEV_DIR_FMT);
        strcat(dir_path, dir_entry->d_name);
        strcat(dir_path, LWSWITCH_I2C_DEV_DIR_INFO_NAME_FMT);

        fp = fopen(dir_path, "r");
        if (fp == NULL)
        {
            continue;
        }

        while ((read = getline(&line, &len, fp)) != -1)
        {
            if (!strstr(line, LWSWITCH_I2C_DELIMITER))
            {
                continue;
            }

            if (strstr(line, bdf_str))
            {
                LwU32 dev_i2c_num;
                LwU32 port_num;
                LwU32 bus;
                LwU32 device;
                LwU32 func;

                if (sscanf(dir_entry->d_name, "i2c-%d", &dev_i2c_num) < 0)
                {
                    FAIL() << "Failed to read I2C id.";
                    fclose(fp);
                    goto cleanup;
                }

                if (sscanf(line, LWSWITCH_I2C_INFO_FMT,
                           &port_num, &bus, &device, &func) < 0)
                {
                    FAIL() << "Failed to read I2C info.";
                    fclose(fp);
                    goto cleanup;
                }

                adapters[num_adapters] = dev_i2c_num;
                adapterPortNum[num_adapters] = port_num;
                num_adapters++;

                if (num_adapters >= arrLength)
                {
                    FAIL() << "Number of adapters greater than expected!";
                    fclose(fp);
                    goto cleanup;
                }
            }
        }

        fclose(fp);
    }

    *numI2cAdapters = num_adapters;

cleanup:
    closedir(dir_ptr);
}

