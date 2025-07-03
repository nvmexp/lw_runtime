/*******************************************************************************
    Copyright (c) 2017-2018 LWpu Corporation

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

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "lwVer.h"
#include "lwswitch_user_linux.h"
#include "lwpu-modprobe-utils.h"
#include "lw_open_cloexec.h"
#include "ioctl_lwswitch.h"
#include "sys/ioctl.h"

static pthread_mutex_t lwswitchapi_mutex = PTHREAD_MUTEX_INITIALIZER;

static LwBool is_lwswitchapi_initialized = LW_FALSE;

//
// LwSwitch driver is lwrrently a part of lwpu.ko. Hence, depend on a client
// do RMAPI init to load lwpu.ko through RMAPI initialization.
//
// Whenever lwswitch driver gets decoupled from lwpu.ko, use this stub to load
// lwswitch.ko on API init.
//
static LW_STATUS _load_kernel_module(void)
{
    return LW_OK;
}

static LW_STATUS _create_char_device(LwU32 minor)
{
    if (lwidia_lwswitch_mknod(minor) == 1)
    {
        return LW_OK;
    }

    //
    // TODO: add lwpu modprobe support in case lwidia_lwswitch_mknod fails.
    //

    return LW_ERR_MODULE_LOAD_FAILED;
}

// Initialize api state
static LW_STATUS _api_state_init(void)
{
    LW_STATUS status;
    int fd = -1, ret;
    LWSWITCH_CHECK_VERSION_PARAMS params;

    if (is_lwswitchapi_initialized)
    {
        return LW_OK;
    }

    // Attempt to load kernel module
    status = _load_kernel_module();
    if (status != LW_OK)
    {
        return status;
    }

    // Create lwpu-lwswitchctl
    status = _create_char_device(LW_LWSWITCH_CTL_MINOR);
    if (status != LW_OK)
    {
        return status;
    }

    fd = lw_open_cloexec(LW_LWSWITCH_CTL_NAME, O_RDWR, 0);
    if (fd < 0)
    {
        return LW_ERR_OPERATING_SYSTEM;
    }

    memset(&params, 0x0, sizeof(params));

    strncpy(params.user.version, LW_VERSION_STRING, strlen(LW_VERSION_STRING));
    params.user.version[LWSWITCH_VERSION_STRING_LENGTH - 1] = '\0';

    ret = ioctl(fd, IOCTL_LWSWITCH_CHECK_VERSION, &params);
    if (ret || !params.is_compatible)
    {
        status = LW_ERR_LIB_RM_VERSION_MISMATCH;
        goto done;
    }

    is_lwswitchapi_initialized = LW_TRUE;

done:
    close(fd);

    return status;
}

static LW_STATUS _create_device_nodes(LWSWITCH_GET_DEVICES_PARAMS *params)
{
    LwU32 i;
    LW_STATUS status;

    for (i = 0; i < params->deviceCount; i++)
    {
        status = _create_char_device(params->info[i].deviceInstance);
        if (status != LW_OK)
        {
            return status;
        }
    }

    return LW_OK;
}

LW_STATUS lwswitch_api_get_devices(LWSWITCH_GET_DEVICES_PARAMS *params)
{
    LW_STATUS status = LW_OK;
    int fd = -1, ret;

    memset(params, 0, sizeof(*params));

    pthread_mutex_lock(&lwswitchapi_mutex);

    status = _api_state_init();
    if (status != LW_OK)
    {
        goto done;
    }

    fd = lw_open_cloexec(LW_LWSWITCH_CTL_NAME, O_RDWR, 0);
    if (fd < 0)
    {
        status = LW_ERR_MODULE_LOAD_FAILED;
        goto done;
    }

    ret = ioctl(fd, IOCTL_LWSWITCH_GET_DEVICES, params);
    if (ret)
    {
        status = LW_ERR_GENERIC;
    }
    else
    {
        status = _create_device_nodes(params);
    }

    close(fd);

done:
    pthread_mutex_unlock(&lwswitchapi_mutex);

    return status;
}

int lwswitch_api_open_device(LwU32 device_instance)
{
    char name[LW_MAX_CHARACTER_DEVICE_FILE_STRLEN];

    if (snprintf(name, LW_MAX_CHARACTER_DEVICE_FILE_STRLEN,
                 LW_LWSWITCH_DEVICE_NAME, device_instance) <= 0)
    {
        return -1;
    }

    return lw_open_cloexec(name, O_RDWR, 0);
}

int lwswitch_api_close_device(int fd)
{
    return (close(fd));
}
