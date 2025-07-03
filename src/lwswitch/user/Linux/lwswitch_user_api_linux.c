/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <poll.h>

#include "lwVer.h"
#include "lwswitch_user_api.h"
#include "lwpu-modprobe-utils.h"
#include "lwpu-modprobe-client-utils.h"
#include "lw_open_cloexec.h"
#include "ioctl_lwswitch.h"
#include "sys/ioctl.h"
#include "ioctl_dev_lwswitch.h"
#include "ctrl_dev_lwswitch.h"
#include "lwlink_user_api.h"

struct lwswitch_device
{
    int fd;
    LwU32 dev_inst;
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

static LWSWITCH_VERSION driver_version = {{0}};

static pthread_mutex_t lwswitchapi_mutex = PTHREAD_MUTEX_INITIALIZER;

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

static LW_STATUS _errno_to_lwstatus
(
    int errno_code
)
{
    if (errno_code < 0)
        errno_code = -errno_code;

    switch (errno_code)
    {
        case 0:
            return LW_OK;

        case ENODEV:
        case EMFILE:
        case ENFILE:
            return LW_ERR_OPERATING_SYSTEM;

        case EFAULT:
        case EILWAL:
            return LW_ERR_ILWALID_ARGUMENT;

        case EOPNOTSUPP:
            return LW_ERR_NOT_SUPPORTED;

        case EEXIST:
        case EISDIR:
        case ENAMETOOLONG:
        case ENOENT:
        case ENOTDIR:
            return LW_ERR_ILWALID_PATH;

        case ENXIO:
            return LW_ERR_MODULE_LOAD_FAILED;

        case EPERM:
            return LW_ERR_INSUFFICIENT_PERMISSIONS;

#ifdef EBADFD
        // EBADFD is Linux-specific
        case EBADFD:
            return LW_ERR_ILLEGAL_ACTION;
#endif

        case EAGAIN:
            return LW_ERR_MORE_PROCESSING_REQUIRED;

        case ENOTTY:
        case EBADF:
        case EFBIG:
        case ENOSPC:
        case EOVERFLOW:
        case EROFS:
        case ETXTBSY:
        default:
            return LW_ERR_GENERIC;
    };

    return LW_ERR_GENERIC;
}

static LW_STATUS _get_dev_inst_from_uuid
(
    const LwUuid *uuid,
    LwU32        *dev_inst
)
{
    unsigned int i;

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

static LW_STATUS _create_char_device
(
    LwU32 minor
)
{
    int state;
    char arg[32];

    if ((lwidia_modprobe(0) == 1) && (lwidia_lwswitch_mknod(minor) == 1))
    {
        return LW_OK;
    }

    snprintf(arg, sizeof(arg), "-c=%u", minor);
    arg[sizeof(arg) - 1] = '\0';

    run_lwidia_modprobe("-s", arg, 0);

    state = lwidia_lwswitch_get_file_state(minor);
    if (lwidia_test_file_state(state, LwDeviceFileStateFileExists) &&
        lwidia_test_file_state(state, LwDeviceFileStateChrDevOk))
    {
        return LW_OK;
    }

    return LW_ERR_MODULE_LOAD_FAILED;
}

// Initialize api state
static LW_STATUS _api_state_init
(
    void
)
{
    LW_STATUS status;
    int fd = -1, ret;
    LWSWITCH_CHECK_VERSION_PARAMS params;

    if (is_lwswitchapi_initialized)
    {
        return LW_OK;
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
    if (ret)
    {
        status = LW_ERR_LIB_RM_VERSION_MISMATCH;
        goto done;
    }

#if !defined(LWSWITCH_CLIENT_PROVIDES_VERSION_COMPAT)
    if (!params.is_compatible)
    {
        status = LW_ERR_LIB_RM_VERSION_MISMATCH;
        goto done;
    }
#endif

    memcpy(&driver_version, &params.kernel, sizeof(params.kernel));

    device_map.num_devices = 0;

    is_lwswitchapi_initialized = LW_TRUE;

done:
    close(fd);

    return status;
}

static LW_STATUS _create_device_nodes
(
    LWSWITCH_GET_DEVICES_V2_PARAMS *params
)
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

static void _destroy_lwswitch_event
(
    lwswitch_event **event
)
{
    unsigned int i;

    if (*event != NULL)
    {
        for (i = 0; i < (*event)->num_fds; i++)
        {
            close((*event)->data[i].fd);
        }
        free(*event);
        *event = NULL;
    }
}

LW_STATUS lwswitch_api_get_devices
(
    LWSWITCH_GET_DEVICES_V2_PARAMS *params
)
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

    ret = ioctl(fd, IOCTL_LWSWITCH_GET_DEVICES_V2, params);
    if (ret)
    {
        status = LW_ERR_GENERIC;
        goto done;
    }

    if (params->deviceCount == 0)
    {
        status = LW_WARN_NOTHING_TO_DO;
        goto done;
    }

    status = _create_device_nodes(params);
    if (status != LW_OK)
    {
        goto done;
    }

    _create_device_map(params);

done:
    if (fd >= 0)
    {
        close(fd);
    }

    pthread_mutex_unlock(&lwswitchapi_mutex);

    return status;
}

LW_STATUS lwswitch_api_create_device
(
    const LwUuid     *uuid,
    lwswitch_device **device
)
{
    LW_STATUS status;
    int fd;
    char name[LW_MAX_CHARACTER_DEVICE_FILE_STRLEN];
    LwU32 device_instance;
    int ret;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if (*device != NULL)
    {
        pthread_mutex_unlock(&lwswitchapi_mutex);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (!is_lwswitchapi_initialized)
    {
        status = LW_ERR_ILWALID_STATE;
        goto done;
    }

    //
    // TODO We lwrrently assume that device_map is always valid.
    // UUID -> devInst mappings may change if the device minor numbers
    // are reassigned. To solve this, we will need an ioctl
    // to verify that the UUID of a given device is correct.
    //
    status = _get_dev_inst_from_uuid(uuid, &device_instance);
    if (status != LW_OK)
    {
        goto done;
    }

    *device = (lwswitch_device*) malloc(sizeof(lwswitch_device));
    if (*device == NULL)
    {
        status = LW_ERR_NO_MEMORY;
        goto done;
    }

    ret = snprintf(name, LW_MAX_CHARACTER_DEVICE_FILE_STRLEN,
                   LW_LWSWITCH_DEVICE_NAME, device_instance);

    if ((ret < 0) || (ret >= LW_MAX_CHARACTER_DEVICE_FILE_STRLEN))
    {
        status = LW_ERR_ILWALID_ARGUMENT;
        goto done;
    }

    fd = lw_open_cloexec(name, O_RDWR, 0);
    if (fd < 0)
    {
        status = LW_ERR_OPERATING_SYSTEM;
        goto done;
    }

    (*device)->fd       = fd;
    (*device)->dev_inst = device_instance;

    status = LW_OK;

done:
    if ((status != LW_OK) && (*device != NULL))
    {
        free(*device);
        *device = NULL;
    }

    pthread_mutex_unlock(&lwswitchapi_mutex);

    return status;
}

void lwswitch_api_free_device
(
    lwswitch_device **device
)
{
    pthread_mutex_lock(&lwswitchapi_mutex);

    if (*device == NULL)
    {
        goto done;
    }

    close((*device)->fd);
    free(*device);

    *device = NULL;

done:
    pthread_mutex_unlock(&lwswitchapi_mutex);
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
    unsigned int i;
    int ret;
    int fd = -1;
    char name[LW_MAX_CHARACTER_DEVICE_FILE_STRLEN];
    LWSWITCH_REGISTER_EVENTS_PARAMS p;
    p.numEvents = 1;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if ((device == NULL) || (*event != NULL))
    {
        pthread_mutex_unlock(&lwswitchapi_mutex);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    *event = (lwswitch_event *) malloc(sizeof(lwswitch_event) +
                                       (sizeof((*event)->data) * num_events));
    if (*event == NULL)
    {
        status = LW_ERR_NO_MEMORY;
        goto fail;
    }

    (*event)->num_fds = 0;
    (*event)->dev_inst = device->dev_inst;

    ret = snprintf(name, LW_MAX_CHARACTER_DEVICE_FILE_STRLEN,
                   LW_LWSWITCH_DEVICE_NAME, device->dev_inst);

    if ((ret < 0) || (ret >= LW_MAX_CHARACTER_DEVICE_FILE_STRLEN))
    {
        status = LW_ERR_GENERIC;
        goto fail;
    }

    for (i = 0; i < num_events; i++)
    {
        fd = lw_open_cloexec(name, O_RDWR, 0);
        if (fd < 0)
        {
            status = _errno_to_lwstatus(errno);
            goto fail;
        }

        p.eventIds[0] = event_ids[i];

        if (ioctl(fd, IOCTL_LWSWITCH_REGISTER_EVENTS, &p) == -1)
        {
            status = _errno_to_lwstatus(errno);
            goto fail;
        }

        (*event)->data[i].fd       = fd;
        (*event)->data[i].event_id = event_ids[i];
        (*event)->data[i].signaled = LW_FALSE;
        (*event)->num_fds++;
    }

    (*event)->usage_count = 0;

    pthread_mutex_unlock(&lwswitchapi_mutex);
    return LW_OK;

fail:
    if (fd >= 0)
    {
        close(fd);
    }

    _destroy_lwswitch_event(event);

    pthread_mutex_unlock(&lwswitchapi_mutex);
    return status;
}

void lwswitch_api_free_event
(
    lwswitch_event **event
)
{
    pthread_mutex_lock(&lwswitchapi_mutex);

    while((*event != NULL) && ((*event)->usage_count > 0))
    {
        pthread_mutex_unlock(&lwswitchapi_mutex);
        usleep(100);
        pthread_mutex_lock(&lwswitchapi_mutex);
    }

    _destroy_lwswitch_event(event);

    pthread_mutex_unlock(&lwswitchapi_mutex);
}

LW_STATUS lwswitch_api_control
(
    lwswitch_device *device,
    LwU32            command,
    void            *params,
    LwU32            params_size
)
{
    LW_STATUS status = LW_OK;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if (device == NULL)
    {
        status = LW_ERR_ILWALID_ARGUMENT;
        goto done;
    }

    if (ioctl(device->fd, command, params) == -1)
    {
        status = _errno_to_lwstatus(errno);
        goto done;
    }

done:
    pthread_mutex_unlock(&lwswitchapi_mutex);
    return status;
}

LW_STATUS lwswitch_api_wait_events
(
    lwswitch_event **events,
    LwU32            num_events,
    LwU32            timeout_ms
)
{
    struct pollfd *pfd;
    unsigned int i;
    unsigned int j = 0;
    int lwr_pfd = 0;
    int rc;
    LwU32 num_fds = 0;
    LW_STATUS status = LW_OK;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if (((int) timeout_ms) < 0)
    {
        timeout_ms = LWSWITCH_EVENT_WAIT_INDEFINITE;
    }

    for (i = 0; i < num_events; i++)
    {
        if (events[i] == NULL)
        {
            pthread_mutex_unlock(&lwswitchapi_mutex);
            return LW_ERR_ILWALID_ARGUMENT;
        }
        num_fds += events[i]->num_fds;
    }

    pfd = (struct pollfd *) malloc(num_fds * sizeof(struct pollfd));
    if (pfd == NULL)
    {
        pthread_mutex_unlock(&lwswitchapi_mutex);
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
            pfd[lwr_pfd].events = POLLPRI | POLLIN;
            lwr_pfd++;
        }
    }

    pthread_mutex_unlock(&lwswitchapi_mutex);

    rc = poll(pfd, num_fds, (int) timeout_ms);

    pthread_mutex_lock(&lwswitchapi_mutex);

    if (rc == 0)
    {
        status = LW_ERR_TIMEOUT;
        goto done;
    }
    else if (rc < 0)
    {
        status = _errno_to_lwstatus(errno);
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
                !!(pfd[lwr_pfd++].revents & (POLLIN | POLLPRI));
        }
    }

done:
    for (i = 0; i < num_events; i++)
    {
        events[i]->usage_count--;
    }

    pthread_mutex_unlock(&lwswitchapi_mutex);

    free(pfd);

    return status;
}

LW_STATUS lwswitch_api_get_event_info
(
    lwswitch_event      *event,
    lwswitch_event_info *info
)
{
    LW_STATUS status = LW_OK;
    unsigned int i;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if ((event == NULL) || (info == NULL))
    {
        status = LW_ERR_ILWALID_ARGUMENT;
        goto done;
    }

    for (i = 0; i < device_map.num_devices; i++)
    {
        if (event->dev_inst == device_map.dev_info[i].dev_inst)
        {
            memcpy(&info->uuid, &device_map.dev_info[i].uuid,
                    sizeof(device_map.dev_info[i].uuid));
        }
    }

    info->num_events = event->num_fds;

done:
    pthread_mutex_unlock(&lwswitchapi_mutex);
    return status;
}

LW_STATUS lwswitch_api_get_signaled_events
(
    lwswitch_event *event,
    LwU32          *event_ids,
    LwU32          *count
)
{
    unsigned int i;
    unsigned int j = 0;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if (event == NULL || event_ids == NULL || count == NULL)
    {
        pthread_mutex_unlock(&lwswitchapi_mutex);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    for (i = 0; i < event->num_fds; i++)
    {
        if (event->data[i].signaled)
        {
            event_ids[j++] = event->data[i].event_id;
        }
    }

    *count = j;

    pthread_mutex_unlock(&lwswitchapi_mutex);
    return LW_OK;
}

LW_STATUS lwswitch_api_get_kernel_driver_version(LWSWITCH_VERSION *version)
{
    LW_STATUS status = LW_OK;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if (!is_lwswitchapi_initialized)
    {
        status = LW_ERR_ILWALID_STATE;
        goto done;
    }

    memcpy(version, &driver_version, sizeof(driver_version));

done:
    pthread_mutex_unlock(&lwswitchapi_mutex);
    return status;
}

static LW_STATUS _create_cap_device_file
(
    char *cap_proc_path,
    int  *minor
)
{
    int state;
    char arg[256];

    if ((lwidia_cap_mknod(cap_proc_path, minor) == 1))
    {
        return LW_OK;
    }

    snprintf(arg, sizeof(arg), "-f=%s", cap_proc_path);
    arg[sizeof(arg) - 1] = '\0';

    run_lwidia_modprobe("-s", arg, 0);

    state = lwidia_cap_get_file_state(cap_proc_path);
    if (lwidia_test_file_state(state, LwDeviceFileStateFileExists) &&
        lwidia_test_file_state(state, LwDeviceFileStateChrDevOk))
    {
        return LW_OK;
    }

    return LW_ERR_ILWALID_STATE;
}

LW_STATUS lwswitch_api_acquire_capability
(
    lwswitch_device *device,
    LwU32            capability
)
{
    LW_STATUS status = LW_OK;
    int cap_fd = -1;
    int minor;
    LWSWITCH_ACQUIRE_CAPABILITY_PARAMS params = {0};
    char path[256];
    const char *fmt = NULL;

    pthread_mutex_lock(&lwswitchapi_mutex);

    if (device == NULL)
    {
        status = LW_ERR_ILWALID_ARGUMENT;
        goto done;
    }

    switch (capability)
    {
        case LWSWITCH_CAP_FABRIC_MANAGEMENT:
        {
            fmt = LWLINK_CAP_FABRIC_MGMT_PATH;
            strncpy(path, fmt, sizeof(path));
            break;
        }
        default:
        {
            status = LW_ERR_ILWALID_ARGUMENT;
            goto done;
        }
    }

    path[sizeof(path) - 1] = '\0';

    status = _create_cap_device_file(path, &minor);
    if (status != LW_OK)
    {
        goto done;
    }

    /* Construct the /dev file path */
    snprintf(path, sizeof(path), LW_CAP_DEVICE_NAME, minor);
    path[sizeof(path) - 1] = '\0';

    if (access(path, R_OK) != 0)
    {
        status = LW_ERR_INSUFFICIENT_PERMISSIONS;
        goto done;
    }

    cap_fd = lw_open_cloexec(path, O_RDONLY, 0);
    if (cap_fd < 0)
    {
        status = LW_ERR_OPERATING_SYSTEM;
        goto done;
    }

    params.capDescriptor = cap_fd;
    params.cap = capability;
    if (ioctl(device->fd, IOCTL_LWSWITCH_ACQUIRE_CAPABILITY, &params) == -1)
    {
        status = _errno_to_lwstatus(errno);
    }

    /*
    * Upon success, IOCTL_LWSWITCH_ACQUIRE_CAPABILITY
    * will have duped the capability fd; we can close it now.
    */
    close(cap_fd);

done:
    pthread_mutex_unlock(&lwswitchapi_mutex);
    return status;
}
