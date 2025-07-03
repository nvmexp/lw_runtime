/*******************************************************************************
    Copyright (c) 2017-2019 LWpu Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef _IOCTL_LWSWITCH_H_
#define _IOCTL_LWSWITCH_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "ioctl_common_lwswitch.h"
#include "lwCpuUuid.h"

/* 4 chars for "SWX-" prefix + 36 chars for UUID string + 1 char for '\0' */
#define LWSWITCH_UUID_STRING_LENGTH 41

#define LWSWITCH_NIBBLE_TO_CHAR(nibble) \
    (((nibble) > 9) ? (((nibble) - 10) + 'A') : ((nibble) + '0'))

static LW_INLINE
LwU32 lwswitch_uuid_to_string(LwUuid *uuid, char *str, LwU32 strLen)
{
    int i;
    int j = 0;

    if ((uuid == NULL) || (str == NULL) || (strLen < LWSWITCH_UUID_STRING_LENGTH))
    {
        return 0;
    }

    str[j++] = 'S';
    str[j++] = 'W';
    str[j++] = 'X';
    str[j++] = '-';

    for (i = 0; i < LW_UUID_LEN; i++)
    {
        if ((i == 4) || (i == 6) || (i == 8) || (i == 10))
        {
            str[j++] = '-';
        }

        str[j++] = LWSWITCH_NIBBLE_TO_CHAR((uuid->uuid[i] & 0xF0) >> 4);
        str[j++] = LWSWITCH_NIBBLE_TO_CHAR(uuid->uuid[i] & 0x0F);
    }

    str[j++] = '\0';

    return j;
}

/*
 * This file defines IOCTL calls that work with lwpu-lwswitchctl
 * (device agnostic) node.
 */

#define LWSWITCH_VERSION_STRING_LENGTH  64

/*
 * Version string
 */
typedef struct
{
    char version[LWSWITCH_VERSION_STRING_LENGTH];
} LWSWITCH_VERSION;

/*
 * LWSWITCH_CTL_CHECK_VERSION
 *
 * The interface will check if the client's version is supported by the driver.
 *
 * Parameters:
 * user[in]
 *    Version of the interface that the client is compiled with.
 * kernel[out]
 *    Version of the interface that the kernel driver is compiled with.
 * is_compatible[out]
 *    Set to true, if user and kernel version are compatible.
 */
typedef struct
{
    LWSWITCH_VERSION user;
    LWSWITCH_VERSION kernel;
    LwBool is_compatible;
} LWSWITCH_CHECK_VERSION_PARAMS;

/*
 * Max devices supported by the driver
 *
 * See ctrl_dev_lwswitch.h for preprocessor definition modification guidelines.
 */
#define LWSWITCH_MAX_DEVICES 64

/*
 * LWSWITCH_CTL_GET_DEVICES
 *
 * This control call will be removed soon. Use LWSWITCH_CTL_GET_DEVICES_V2 instead.
 *
 * Provides information about registered LwSwitch devices.
 *
 * Parameters:
 * deviceInstance[out]
 *    Device instance of the device. This is same as the device minor number
 *    for Linux platforms.
 */
typedef struct
{
    LwU32 deviceInstance;
    LwU32 pciDomain;
    LwU32 pciBus;
    LwU32 pciDevice;
    LwU32 pciFunction;
    /* See ctrl_dev_lwswitch.h for struct definition modification guidelines */
} LWSWITCH_DEVICE_INSTANCE_INFO;

typedef struct
{
    LwU32 deviceCount;
    LWSWITCH_DEVICE_INSTANCE_INFO info[LWSWITCH_MAX_DEVICES];
    /* See ctrl_dev_lwswitch.h for struct definition modification guidelines */
} LWSWITCH_GET_DEVICES_PARAMS;

/*
 * LWSWITCH_CTL_GET_DEVICES_V2
 *
 * Provides information about registered LwSwitch devices.
 * V2 adds a UUID field to the device instance info struct
 *
 * Parameters:
 * deviceInstance[out]
 *    Device instance of the device. This is same as the device minor number
 *    for Linux platforms.
 */
typedef struct
{
    LwU32 deviceInstance;
    LwUuid uuid;
    LwU32 pciDomain;
    LwU32 pciBus;
    LwU32 pciDevice;
    LwU32 pciFunction;
    LWSWITCH_DRIVER_FABRIC_STATE driverState;
    LWSWITCH_DEVICE_FABRIC_STATE deviceState;
    LWSWITCH_DEVICE_BLACKLIST_REASON deviceReason;
    LwU32 physId;

    /* See ctrl_dev_lwswitch.h for struct definition modification guidelines */
} LWSWITCH_DEVICE_INSTANCE_INFO_V2;

#define LWSWITCH_ILWALID_PHYS_ID        LW_U32_MAX

typedef struct
{
    LwU32 deviceCount;
    LWSWITCH_DEVICE_INSTANCE_INFO_V2 info[LWSWITCH_MAX_DEVICES];
    /* See ctrl_dev_lwswitch.h for struct definition modification guidelines */
} LWSWITCH_GET_DEVICES_V2_PARAMS;

#define CTRL_LWSWITCH_GET_DEVICES         0x01
#define CTRL_LWSWITCH_CHECK_VERSION       0x02
#define CTRL_LWSWITCH_GET_DEVICES_V2      0x03

/*
 * Lwswitchctl (device agnostic) IOCTLs
 */

#define IOCTL_LWSWITCH_GET_DEVICES \
    LWSWITCH_IOCTL_CODE(LWSWITCH_CTL_IO_TYPE, CTRL_LWSWITCH_GET_DEVICES, LWSWITCH_GET_DEVICES_PARAMS, \
                        LWSWITCH_IO_READ_ONLY)
#define IOCTL_LWSWITCH_CHECK_VERSION \
    LWSWITCH_IOCTL_CODE(LWSWITCH_CTL_IO_TYPE, CTRL_LWSWITCH_CHECK_VERSION, LWSWITCH_CHECK_VERSION_PARAMS, \
                        LWSWITCH_IO_WRITE_READ)
#define IOCTL_LWSWITCH_GET_DEVICES_V2 \
    LWSWITCH_IOCTL_CODE(LWSWITCH_CTL_IO_TYPE, CTRL_LWSWITCH_GET_DEVICES_V2, LWSWITCH_GET_DEVICES_V2_PARAMS, \
                        LWSWITCH_IO_READ_ONLY)

#ifdef __cplusplus
}
#endif

#endif //_IOCTL_LWSWITCH_H_
