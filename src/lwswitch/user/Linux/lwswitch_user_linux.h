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

#ifndef _LWSWITCH_USER_LINUX_H_
#define _LWSWITCH_USER_LINUX_H_

#include "lwtypes.h"
#include "lwstatus.h"
#include "ioctl_lwswitch.h"

//
// The API provides device information about all the registered devices.
// See ioctl_lwswitch.h for parameter details.
//
// It also creates device nodes lwpu-lwswitchX for the registered devices
// where X stands for a device instance.
//
LW_STATUS lwswitch_api_get_devices(LWSWITCH_GET_DEVICES_PARAMS *params);

//
// The API opens a file handle using a valid device instance.
//
// On success, returns a file handle.
//
int lwswitch_api_open_device(LwU32 device_instance);

//
// The API closes a valid file handle.
//
// On success, returns zero.
//
int lwswitch_api_close_device(int fd);

#endif // _LWSWITCH_USER_LINUX_H_
