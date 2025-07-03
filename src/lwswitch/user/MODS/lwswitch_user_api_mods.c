/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <string.h>

#include "lwlink.h"
#include "modsdrv.h"
#include "mods_lwswitch.h"

#include "lwVer.h"
#include "lwswitch_user_api.h"
#include "sys/ioctl.h"
#include "ctrl_dev_lwswitch.h"

struct lwswitch_device
{
    LwU32 instance;
};

struct lwswitch_event
{
    LwU32 dev_inst;
    LwU32 num_fds;
    LwU32 usage_count;

    // this array is arbitrarily sized based on the number of event IDs
    // associated with this lwswitch_event.
    struct
    {
        int fd;
        LwBool signaled;
        LwU32 event_id;
    } data[1];
    // PLACE NOTHING AFTER data
};

static void *lwswitchapi_mutex = NULL;

static LwBool is_lwswitchapi_initialized = LW_FALSE;

static struct
{
    struct
    {
        LwUuid uuid;
        LwU32 dev_inst;
    } dev_info[LWSWITCH_MAX_DEVICES];
    LwU32 num_devices;
} device_map;

LwU32
lwswitch_get_physid
(
    LwU32 instance
)
{
    LWSWITCH_GET_INFO get_info;
    LwlStatus lwswitch_status;

    get_info.count = 1;
    get_info.index[0] = LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID;

    lwswitch_status = lwswitch_mods_ctrl(instance, CTRL_LWSWITCH_GET_INFO, &get_info, sizeof(get_info));
    if (lwswitch_status != LWL_SUCCESS)
    {
        return 0;
    }

    return get_info.info[0];
}

static void _create_device_map
(
    LWSWITCH_GET_DEVICES_V2_PARAMS *params
)
{
    LwU32 i;

    for (i = 0; i < params->deviceCount; i++)
    {
        memcpy(&device_map.dev_info[i].uuid, &params->info[i].uuid, sizeof(params->info[i].uuid));
        device_map.dev_info[i].dev_inst = params->info[i].deviceInstance;
    }

    device_map.num_devices = params->deviceCount;
}

LW_STATUS lwswitch_api_get_devices
(
    LWSWITCH_GET_DEVICES_V2_PARAMS *params
)
{
    LW_STATUS status = LW_OK;
    LwU32 instance = 0;
    struct lwlink_pci_info pciInfo;
    LwU64 linkMask;
    LwlStatus lwswitch_status;

    if (lwswitchapi_mutex == NULL)
    {
        lwswitchapi_mutex = ModsDrvAllocMutex();
        if (lwswitchapi_mutex == NULL)
        {
            return LW_ERR_MODULE_LOAD_FAILED;
        }
    }

    ModsDrvMemSet(params, 0, sizeof(*params));

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    while (1)
    {
        lwswitch_status = lwswitch_mods_get_device_info(instance, (LwU32 *) &linkMask, &pciInfo);
        if (lwswitch_status == LW_OK)
        {
            params->info[instance].deviceInstance = instance;

            // TODO: Fill in UUID
            params->info[instance].uuid.uuid[0] = instance;

            params->info[instance].pciDomain = pciInfo.domain;
            params->info[instance].pciBus = pciInfo.bus;
            params->info[instance].pciDevice = pciInfo.device;
            params->info[instance].pciFunction = pciInfo.function;

            params->info[instance].driverState = LWSWITCH_DRIVER_FABRIC_STATE_STANDBY;
            params->info[instance].deviceState = LWSWITCH_DEVICE_FABRIC_STATE_STANDBY;
            params->info[instance].deviceReason = LWSWITCH_DEVICE_BLACKLIST_REASON_NONE;

            // Get correct physical ID
            params->info[instance].physId = lwswitch_get_physid(instance);

            instance++;
            params->deviceCount = instance;
        }
        else
        {
            break;
        }
    }

    if (params->deviceCount == 0)
    {
        status = LW_WARN_NOTHING_TO_DO;
        goto done;
    }

    _create_device_map(params);

    is_lwswitchapi_initialized  = LW_TRUE;

done:
    ModsDrvReleaseMutex(lwswitchapi_mutex);

    return status;
}

static LW_STATUS _get_dev_inst_from_uuid
(
    const LwUuid *uuid,
    LwU32        *dev_inst
)
{
    LwU32 i;

    if (uuid == NULL)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }

    for (i = 0; i < device_map.num_devices; i++)
    {
        if (memcmp(uuid, &device_map.dev_info[i].uuid, sizeof(*uuid)) == 0)
        {
            *dev_inst = device_map.dev_info[i].dev_inst;
            return LW_OK;
        }
    }

    return LW_ERR_ILWALID_ARGUMENT;
}

LW_STATUS lwswitch_api_create_device
(
    const LwUuid *uuid,
    lwswitch_device **device
)
{
    LW_STATUS status = LW_OK;

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if (*device != NULL)
    {
        ModsDrvReleaseMutex(lwswitchapi_mutex);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (!is_lwswitchapi_initialized)
    {
        status = LW_ERR_ILWALID_STATE;
        goto done;
    }

    // Create dummy session structure. MODS compiles RM, so can call functions directly
    *device = ModsDrvAlloc(sizeof(lwswitch_device));
    if (*device == NULL)
    {
        status = LW_ERR_NO_MEMORY;
        goto done;
    }

    status = _get_dev_inst_from_uuid(uuid, &((*device)->instance));

done:
    if ((status != LW_OK) && (*device != NULL))
    {
        ModsDrvFree(*device);
        *device = NULL;
    }

    ModsDrvReleaseMutex(lwswitchapi_mutex);

    return status;
}

void lwswitch_api_free_device
(
    lwswitch_device **device
)
{
    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if (*device == NULL)
    {
        goto done;
    }

    ModsDrvFree(*device);

    *device = NULL;

done:
    ModsDrvReleaseMutex(lwswitchapi_mutex);
}

LW_STATUS lwswitch_api_control
(
    lwswitch_device *device,
    LwU32           command,
    void           *params,
    LwU32           params_size
)
{
    LW_STATUS status = LW_OK;
    LwlStatus lwswitch_status = LWL_SUCCESS;

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if (device == NULL)
    {
        status = LW_ERR_ILWALID_ARGUMENT;
        goto done;
    }

    lwswitch_status = lwswitch_mods_ctrl(device->instance, _IOC_NR(command), params, params_size);

    // TODO: Translate lwswitch_status/LwlStatus to status/LW_STATUS
    status = lwswitch_status;

done:
    ModsDrvReleaseMutex(lwswitchapi_mutex);
    return status;
}

LW_STATUS lwswitch_api_get_kernel_driver_version(LWSWITCH_VERSION *version)
{
    LW_STATUS status = LW_OK;

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if (!is_lwswitchapi_initialized)
    {
        status = LW_ERR_ILWALID_STATE;
        goto done;
    }

    memcpy(&version->version, LW_VERSION_STRING, strlen(LW_VERSION_STRING));
    version->version[strlen(LW_VERSION_STRING)] = '\0';

done:
    ModsDrvReleaseMutex(lwswitchapi_mutex);
    return status;
}

LW_STATUS lwswitch_api_acquire_capability
(
    lwswitch_device *device,
    LwU32 capability
)
{
    return LW_OK;
}

static void _destroy_lwswitch_event
(
    lwswitch_event **event
)
{
    if (*event != NULL)
    {
        ModsDrvFree(*event);
        *event = NULL;
    }
}

LW_STATUS lwswitch_api_create_event
(
    lwswitch_device *device,
    LwU32           *event_ids,
    LwU32            num_events,
    lwswitch_event **event
)
{
    LW_STATUS status;
    LwU32 i;

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if ((device == NULL) || (*event != NULL))
    {
        ModsDrvReleaseMutex(lwswitchapi_mutex);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    *event = (lwswitch_event *) ModsDrvAlloc(sizeof(lwswitch_event) +
                                       (sizeof((*event)->data) * num_events));
    if (*event == NULL)
    {
        status = LW_ERR_NO_MEMORY;
        goto fail;
    }

    (*event)->num_fds = 0;
    (*event)->dev_inst = device->instance;

    for (i = 0; i < num_events; i++)
    {
        (*event)->data[i].fd       = device->instance;
        (*event)->data[i].event_id = (event_ids[i] == LWSWITCH_DEVICE_EVENT_FATAL) ? 0 /*POLLPRI*/ : 1 /*POLLIN*/;
        (*event)->data[i].signaled = LW_FALSE;
        (*event)->num_fds++;
    }

    (*event)->usage_count = 0;

    ModsDrvReleaseMutex(lwswitchapi_mutex);
    return LW_OK;

fail:
    _destroy_lwswitch_event(event);

    ModsDrvReleaseMutex(lwswitchapi_mutex);
    return status;
}

void lwswitch_api_free_event
(
    lwswitch_event **event
)
{
    ModsDrvAcquireMutex(lwswitchapi_mutex);

    _destroy_lwswitch_event(event);

    ModsDrvReleaseMutex(lwswitchapi_mutex);
}

LW_STATUS lwswitch_api_wait_events
(
    lwswitch_event **events,
    LwU32            num_events,
    LwU32            timeout_ms
)
{
#if 1
    return LW_OK;
#else   //1
    struct pollfd *pfd;
    LwU32 i;
    LwU32 j = 0;
    int lwr_pfd = 0;
    int rc;
    LwU32 num_fds = 0;
    LW_STATUS status = LW_OK;

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if (((int) timeout_ms) < 0)
    {
        timeout_ms = LWSWITCH_EVENT_WAIT_INDEFINITE;
    }

    for (i = 0; i < num_events; i++)
    {
        if (events[i] == NULL)
        {
            ModsDrvReleaseMutex(lwswitchapi_mutex);
            return LW_ERR_ILWALID_ARGUMENT;
        }
        num_fds += events[i]->num_fds;
    }

    pfd = (struct pollfd *) ModsDrvAlloc(num_fds * sizeof(struct pollfd));
    if (pfd == NULL)
    {
        ModsDrvReleaseMutex(lwswitchapi_mutex);
        return LW_ERR_NO_MEMORY;
    }

    // map the fds owned by each lwswitch_event into a pfd structure
    lwr_pfd = 0;
    for (i = 0; i < num_events; i++)
    {
        events[i]->usage_count++;
        for (j = 0; j < events[i]->num_fds; j++)
        {
            pfd[lwr_pfd].fd = events[i]->data[j].fd;
            pfd[lwr_pfd].events = events[i]->data[j].event_id;
            lwr_pfd++;
        }
    }

    ModsDrvReleaseMutex(lwswitchapi_mutex);

    rc = poll(pfd, num_fds, (int) timeout_ms);

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if (rc == 0)
    {
        status = LW_ERR_TIMEOUT;
        goto done;
    }
    else if (rc < 0)
    {
        status = LW_ERR_NOT_SUPPORTED /*_errno_to_lwstatus(errno)*/;
        goto done;
    }

    // map revents returned by poll back into the fds owned by each lwswitch_event
    lwr_pfd = 0;
    for (i = 0; i < num_events; i++)
    {
        for (j = 0; j < events[i]->num_fds; j++)
        {
            if (pfd[lwr_pfd].revents == POLLHUP)
            {
                status = LW_ERR_ILWALID_OBJECT;
                goto done;
            }

            events[i]->data[j].signaled =
                !!(pfd[lwr_pfd++].revents & events[i]->data[j].event_id);
        }
    }

done:
    for (i = 0; i < num_events; i++)
    {
        events[i]->usage_count--;
    }

    ModsDrvReleaseMutex(lwswitchapi_mutex);

    ModsDrvFree(pfd);

    return status;
#endif  //0
}

LW_STATUS lwswitch_api_get_event_info
(
    lwswitch_event      *event,
    lwswitch_event_info *info
)
{
    LW_STATUS status = LW_OK;
    LwU32 i;

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if ((event == NULL) || (info == NULL))
    {
        status = LW_ERR_ILWALID_ARGUMENT;
        goto done;
    }

    for (i = 0; i < device_map.num_devices; i++)
    {
        if (event->dev_inst == device_map.dev_info[i].dev_inst)
        {
            ModsDrvMemCopy(&info->uuid, &device_map.dev_info[i].uuid,
                    sizeof(device_map.dev_info[i].uuid));
        }
    }

    info->num_events = event->num_fds;

done:
    ModsDrvReleaseMutex(lwswitchapi_mutex);
    return status;
}

LW_STATUS lwswitch_api_get_signaled_events
(
    lwswitch_event *event,
    LwU32          *event_ids,
    LwU32          *count
)
{
    LwU32 i;
    LwU32 j = 0;

    ModsDrvAcquireMutex(lwswitchapi_mutex);

    if (event == NULL || event_ids == NULL || count == NULL)
    {
        ModsDrvReleaseMutex(lwswitchapi_mutex);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    for (i = 0; i < event->num_fds; i++)
    {
        if (event->data[i].signaled)
        {
            // TODO: Port event_ids from Linux to MODS
            event_ids[j++] = (event->data[i].event_id == 0 /*POLLPRI*/) ?
                LWSWITCH_DEVICE_EVENT_FATAL : LWSWITCH_DEVICE_EVENT_NONFATAL;
        }
    }

    *count = j;

    ModsDrvReleaseMutex(lwswitchapi_mutex);
    return LW_OK;
}


