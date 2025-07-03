/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LWSWITCH_USER_API_H_
#define _LWSWITCH_USER_API_H_

#ifdef __linux__
#include <stdio.h>
#include <sys/ioctl.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#include <WinIoCtl.h>
#endif

#include "lwtypes.h"
#include "lwstatus.h"
#include "ioctl_lwswitch.h"
#include "lwCpuUuid.h"
#include "ctrl_dev_lwswitch.h"

#define LWSWITCH_EVENT_WAIT_INDEFINITE ((LwU32) -1)

typedef struct lwswitch_device lwswitch_device;

typedef struct lwswitch_event lwswitch_event;

typedef struct
{
    LwUuid uuid;
    LwU32  num_events;
} lwswitch_event_info;

//
// The API provides device information about all the registered devices.
// See ioctl_lwswitch.h for parameter details.
//
// It also performs API initialization if necessary.
//
LW_STATUS lwswitch_api_get_devices(LWSWITCH_GET_DEVICES_V2_PARAMS *params);

//
// The API opens the LWSwitch device with the specified UUID.
//
// On success, an lwswitch_device is allocated and returned to the user.
//
LW_STATUS lwswitch_api_create_device(const LwUuid *uuid, lwswitch_device **device);

//
// The API closes the LWSwitch device handle and frees the lwswitch_device.
//
void lwswitch_api_free_device(lwswitch_device **device);

//
// The API opens an OS-specific event handle and associates it with
// each event ID in event_ids.
//
// On success, an lwswitch_event is allocated and returned to the user.
//
LW_STATUS lwswitch_api_create_event(lwswitch_device *device, LwU32 *event_ids,
                                    LwU32 num_events, lwswitch_event **event);

//
// The API closes the event handle and frees the lwswitch_event.
// This call blocks if the lwswitch_event is lwrrently being used
// by lwswitch_api_control.
//
void lwswitch_api_free_event(lwswitch_event **lwswitch_event);

//
// The API performs an IOCTL on the device
//
LW_STATUS lwswitch_api_control(lwswitch_device *device, LwU32 command, void *params, LwU32 params_size);

//
// The API waits on a list of lwswitch_event, blocking until at least one event oclwrs or the timeout is reached.
//
// On return, users may call lwswitch_api_get_event_info and lwswitch_api_get_signaled_events to access the signaled events.
//
LW_STATUS lwswitch_api_wait_events(lwswitch_event **events, LwU32 num_events, LwU32 timeout_ms);

//
// Sets the fields of the input lwswitch_event_info struct.
//
// On success, info->uuid and info->num_events will be set.
//
LW_STATUS lwswitch_api_get_event_info(lwswitch_event *event, lwswitch_event_info *info);

//
// Gets the signaled events from the lwswitch_event. Must be called after lwswitch_api_wait_events().
//
// event_ids is a user allocated array that is large enough to accomodate lwswitch_event_info->num_events.
//
// On success, event_ids is filled with all signaled events and count is set to the number of valid elements in event_ids.
//
LW_STATUS lwswitch_api_get_signaled_events(lwswitch_event *event, LwU32 *event_ids, LwU32 *count);

//
// Queries the API for the LWSwitch kernel driver version. Users may use this in conjunction with the make variable
// LWSWITCH_CLIENT_PROVIDES_VERSION_COMPAT := 1 to enforce their own compatibility checks.
//
LW_STATUS lwswitch_api_get_kernel_driver_version(LWSWITCH_VERSION *version);

//
// The API acquire the lwswitch capability.
//
// On success, acquire requested the lwswitch capability and return to the user.
//
LW_STATUS lwswitch_api_acquire_capability(lwswitch_device *device, LwU32 capability);

#endif // _LWSWITCH_USER_API_H_
