/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <search.h>

#include <lwsciipc_common.h>
#include <lwsciipc_os_error.h>
#include <lwsciipc_ivc.h>
#include <lwscierror.h>
#include <lwsciipc_ioctl.h>

#define COMMENT_CHAR 0x23U /* '#' */

#define MAX_LINUX_ERRNO	600	/* FIXME */
#define COMMENT_CHAR 0x23U /* '#' */

#define LWSCIIPC_DEV_NAME	"/dev/lwsciipc"

#define SOC_ID_FILE_PATH        "/proc/device-tree/soc_id"

#define SOCID_BUF_SIZE 4

/* Secure buffer transfer APIs are suported only on CheetAh Linux */
#if !defined(__x86_64__)
#define LWSCIIPC_GET_VUID
#endif

static struct LwSciIpcConfigEntry **s_epDB;
static uint32_t s_noEntries;

#ifdef LWSCIIPC_GET_VUID
static int sciipc_fd;
#endif

int32_t lwsciipc_os_mutex_init(void *mutex, void *attr)
{
#ifdef LWSCIIPC_USE_MUTEX
    pthread_mutex_t *m = mutex;
    pthread_mutexattr_t *a = attr;

    return pthread_mutex_init(m, a);
#else
	return 0;
#endif
}

int32_t lwsciipc_os_mutex_lock(void *mutex)
{
#ifdef LWSCIIPC_USE_MUTEX
    pthread_mutex_t *m = mutex;

    return pthread_mutex_lock(m);
#else
	return 0;
#endif
}

int32_t lwsciipc_os_mutex_unlock(void *mutex)
{
#ifdef LWSCIIPC_USE_MUTEX
    pthread_mutex_t *m = mutex;

    return pthread_mutex_unlock(m);
#else
	return 0;
#endif
}

int32_t lwsciipc_os_mutex_destroy(void *mutex)
{
#ifdef LWSCIIPC_USE_MUTEX
   pthread_mutex_t *m = mutex;

   return pthread_mutex_destroy(m);
#else
	return 0;
#endif
}

static void lwsciipc_os_get_backend_type(const char *str, uint32_t slen,
	uint32_t *fmt_type, uint32_t *backend)
{
        if (!strncmp(str, BACKEND_ITC_NAME, slen)) {
                *backend = LWSCIIPC_BACKEND_ITC;
                *fmt_type = DT_FMT_TYPE1;
        }
        else if (!strncmp(str, BACKEND_IPC_NAME, slen)) {
                *backend = LWSCIIPC_BACKEND_IPC;
                *fmt_type = DT_FMT_TYPE1;
        }
        else if (!strncmp(str, BACKEND_IVC_NAME, slen)) {
                *backend = LWSCIIPC_BACKEND_IVC;
                *fmt_type = DT_FMT_TYPE2;
        }
        else if (!strncmp(str, BACKEND_C2C_NAME, slen)) {
                *backend = LWSCIIPC_BACKEND_C2C;
                *fmt_type = DT_FMT_TYPE2;
        }
        else {
                *backend = LWSCIIPC_BACKEND_MAX;
                *fmt_type = 0;
        }

        return;
}

int32_t lwsciipc_os_get_endpoint_entry_num(uint32_t *entryNum)
{
	FILE *fp;
	char line[MAXBUF];
	char backend_name[MAX_BACKEND_NAME];
	uint32_t fmt_type = 0;
	uint32_t backend_type = 0;
	uint32_t count = 0;

        lwsciipc_dbg("opening file %s", CFG_FILE);
        fp = fopen(CFG_FILE, "r");
        if (fp == NULL) {
                lwsciipc_err("Failed to open %s", CFG_FILE);
                return ErrnoToLwSciErr(errno);
        }

        while (fgets(line, (int32_t)MAXBUF, fp) != NULL) {
                /* skip line starting with comment(#) */
                if ((uint8_t)line[0] == (uint8_t)COMMENT_CHAR) {
                        continue;
                }
		(void)sscanf(line, "%s", backend_name);
		lwsciipc_dbg("%s : backend = *%s*\n", __func__, backend_name);

		lwsciipc_os_get_backend_type(backend_name, strlen(backend_name),
					     &fmt_type, &backend_type);

		if (fmt_type == DT_FMT_TYPE1)
			count += 2;
		else if (fmt_type == DT_FMT_TYPE2)
			count += 1;
        }
	*entryNum = count;

        rewind(fp);
	(void)fclose(fp);

	return 0;
}

int32_t lwsciipc_os_populate_endpoint_db(struct LwSciIpcConfigEntry **epDB)
{
	FILE *fp;
	uint32_t info[2];
	char line[MAXBUF];
	char endpoint_name[LWSCIIPC_MAX_ENDPOINT_NAME];
	char endpoint2_name[LWSCIIPC_MAX_ENDPOINT_NAME];
	char backend_name[MAX_BACKEND_NAME];
	ENTRY item;
	uint32_t fmt_type = 0;
	uint32_t backend_type = 0;
	uint32_t count = 0;
	uint32_t i = 0;

        lwsciipc_dbg("opening file %s", CFG_FILE);
        fp = fopen(CFG_FILE, "r");
        if (fp == NULL) {
                lwsciipc_err("Failed to open %s", CFG_FILE);
                return ErrnoToLwSciErr(errno);
        }

	while (fgets(line, (int32_t)MAXBUF, fp) != NULL)
	{
		/* skip line starting with comment(#) */
		if ((uint8_t)line[0] == (uint8_t)COMMENT_CHAR) {
			continue;
		}

		(void)sscanf(line, "%s", backend_name);
		lwsciipc_dbg("%s : backend = *%s*\n", __func__, backend_name);

		lwsciipc_os_get_backend_type(backend_name, strlen(backend_name),
					     &fmt_type, &backend_type);

		if (fmt_type == DT_FMT_TYPE1) {
			(void)sscanf(line, "%s %s %s %d %d", backend_name,
				     endpoint_name, endpoint2_name, &info[0],
				     &info[1]);

			(void)memcpy(epDB[i]->epName, endpoint_name,
				     sizeof(endpoint_name));
			(void)memcpy(epDB[i]->devName, endpoint_name,
				     sizeof(endpoint_name));
			epDB[i]->backend = backend_type;
			epDB[i]->nFrames = info[0];
			epDB[i]->frameSize = info[1];
			epDB[i]->id = 0;

			(void)memcpy(epDB[i+1]->epName, endpoint2_name,
				     sizeof(endpoint2_name));
			(void)memcpy(epDB[i+1]->devName, endpoint_name,
				     sizeof(endpoint_name));
			epDB[i+1]->backend = backend_type;
			epDB[i+1]->nFrames = info[0];
			epDB[i+1]->frameSize = info[1];
			epDB[i+1]->id = 1;
		} else if (fmt_type == DT_FMT_TYPE2) {
			(void)sscanf(line, "%s %s %d", backend_name,
				     endpoint_name, &info[0]);
			(void)memcpy(epDB[i]->epName, endpoint_name,
				     sizeof(endpoint_name));
			(void)memcpy(epDB[i]->devName, endpoint_name,
				     sizeof(endpoint_name));
			epDB[i]->backend = backend_type;
			epDB[i]->id = info[0];
		}

		if (fmt_type == DT_FMT_TYPE1) {
			i += 2;
		} else if (fmt_type == DT_FMT_TYPE2) {
			i += 1;
		}

	}
	count = i;

        /* enter hash for each endpoint */
        for (i=0; i<count; i++) {
                lwsciipc_dbg("[%d] *%s* *%s* %d %d %d %d", i,
                        epDB[i]->epName, epDB[i]->devName,
                        epDB[i]->backend,
                        epDB[i]->nFrames,
                        epDB[i]->frameSize,
                        epDB[i]->id);

                item.key = epDB[i]->epName;
                item.data = (void *)epDB[i];
                (void) hsearch(item, ENTER);
        }

	(void)fclose(fp);

	return 0;
}

int32_t lwsciipc_os_ioctl(int32_t fd, uint32_t request, void *os_args)
{
	struct lwsciipc_ivc_handle *ivch = (struct lwsciipc_ivc_handle *)os_args;
	struct lwsciipc_ivc_info *ivci;
	int32_t ret = 0;

        lwsciipc_dbg("%s: enter\n", __func__);

	switch (request) {
	case LW_SCI_IPC_IVC_IOCTL_GET_INFO:
		/* get map size info */
		ivci = &ivch->ivc_info;
		ret = ioctl(fd, request, ivci);
		if (ret < 0) {
			lwsciipc_err("%s: get_info ioctl failed\n", __func__);
			return ret;
		}
		lwsciipc_dbg("nframes = %d, frame_size = %d\n", ivci->nframes, ivci->frame_size);
		lwsciipc_dbg("qoffset = 0x%x, qsize = 0x%x\n", ivci->queue_offset, ivci->queue_size);
		lwsciipc_dbg("rx_first = %d area_size = 0x%x\n", ivci->rx_first, ivci->area_size);
		break;
	case LW_SCI_IPC_IVC_IOCTL_NOTIFY_REMOTE:
		ret = ioctl(fd, request);
		if (ret < 0) {
			lwsciipc_err("%s: get_info ioctl failed\n", __func__);
			return ret;
		}
		break;
	default:
		lwsciipc_err("wrong ioctl\n");
		return -1;
	}

	return 0;
}

void *lwsciipc_os_mmap(void *addr, size_t length, int32_t prot, int32_t flags,
			int32_t fd, off_t offset, void *os_args)
{
	return mmap(addr, length, prot, flags, fd, offset);
}

int32_t lwsciipc_os_munmap(void *addr, size_t length)
{
	return munmap(addr, length);
}

LwSciError lwsciipc_os_check_pulse_param(int32_t coid, int16_t priority,
	int16_t code)
{
	return LwSciError_Success;
}

LwSciError lwsciipc_os_open_config(void)
{
    int32_t ret;
    uint32_t i = 0;

    lwsciipc_dbg("enter");

    ret = lwsciipc_os_get_endpoint_entry_num(&s_noEntries);
    if (ret != 0) {
        goto fail;
    }

    (void) hcreate(s_noEntries);

    lwsciipc_dbg("allocating %d entries in endpoint db", s_noEntries);
    s_epDB = (struct LwSciIpcConfigEntry **)
            malloc(s_noEntries * sizeof(struct LwSciIpcConfigEntry *));
    if (s_epDB == NULL) {
        goto fail;
    }

    memset(s_epDB, 0, s_noEntries * sizeof(struct LwSciIpcConfigEntry *));

    for (i = 0; i < s_noEntries; i++) {
        s_epDB[i] = (struct LwSciIpcConfigEntry *)
            malloc(sizeof(struct LwSciIpcConfigEntry));
        if (s_epDB == NULL) {
            goto fail;
        }
    }

    i = 0U;
    lwsciipc_dbg("populating endpoint db");

    ret = lwsciipc_os_populate_endpoint_db(s_epDB);
    if (ret != 0) {
        goto fail;
    }

#ifdef LWSCIIPC_GET_VUID

    /* don't return error as this feature might be enabled for few platforms */
    sciipc_fd = open(LWSCIIPC_DEV_NAME, O_RDWR);
    if (sciipc_fd < 0) {
        lwsciipc_dbg("%s dev node not present\n", LWSCIIPC_DEV_NAME);
    }

#endif

    lwsciipc_dbg("exit");

    return LwSciError_Success;

fail:
    lwsciipc_os_close_config();
    s_noEntries = 0;

    return LwSciError_NotPermitted;
}

LwSciError lwsciipc_os_get_vmid(uint32_t *vmid)
{
    LwSciError ret = LwSciError_NotInitialized;

    /* TODO: LINUX or x86 need to get this info from HV or other config */
    *vmid = 0;
    ret = LwSciError_Success;

    return ret;
}

LwSciError lwsciipc_os_get_socid(uint32_t *socid)
{
    LwSciError ret = LwSciError_NotInitialized;
#if !defined(__x86_64__)
    FILE *fp;
    char soc_id[SOCID_BUF_SIZE];
#endif

#if !defined(__x86_64__)
    /* Use socid 1 by default for CheetAh */
    *socid = 1;
    lwsciipc_dbg("opening socid file : %s\n", SOC_ID_FILE_PATH);
    fp = fopen(SOC_ID_FILE_PATH, "rb");
    if (fp == NULL) {
            lwsciipc_err("Failed to open soc id file: %s", SOC_ID_FILE_PATH);
            ret = ErrnoToLwSciErr(errno);
            goto done;
    }
    if(fread(soc_id, 1, SOCID_BUF_SIZE, fp) == SOCID_BUF_SIZE) {
        *socid =
            (soc_id[0] << 24) | (soc_id[1] << 16) | (soc_id[2] << 8) |
            soc_id[3];
    } else {
            lwsciipc_err("Failed to read soc id file: %s", SOC_ID_FILE_PATH);
            ret = ErrnoToLwSciErr(errno);
            fclose(fp);
            goto done;
    }
    fclose(fp);
#else
        /* X86 uses socid 0 by default */
    *socid = 0;
#endif
    lwsciipc_dbg("socid = %u\n", *socid);
    ret = LwSciError_Success;

#if !defined(__x86_64__)
done:
#endif
    return ret;
}

LwSciError lwsciipc_os_get_config_entry(const char *endpoint,
    struct LwSciIpcConfigEntry **entry)
{
    ENTRY item;
    const ENTRY *found;
    char endpoint_name[LWSCIIPC_MAX_ENDPOINT_NAME];

    lwsciipc_dbg("searching for endpoint *%s*", endpoint);

    (void)memcpy(endpoint_name, endpoint, sizeof(endpoint_name));

    item.key = endpoint_name;
    item.data = (void *)NULL;
    found = hsearch(item, FIND);

    if (found != NULL) {
        /* item is in the table */
        *entry = found->data;
        lwsciipc_dbg("found the endpoint");
        return LwSciError_Success;
    }
    lwsciipc_dbg("endpoint not found");

    return LwSciError_NoSuchEntry;
}

void lwsciipc_os_close_config(void)
{
    uint32_t i;

#ifdef LWSCIIPC_GET_VUID
    close(sciipc_fd);
#endif

    for (i = 0; i < s_noEntries; i++) {
        if (s_epDB[i] != NULL)
            free(s_epDB[i]);
    }

    if (s_epDB != NULL) {
        free(s_epDB);
        s_epDB = NULL;
    }

    s_noEntries = 0;

    hdestroy();
}

LwSciError lwsciipc_os_get_vuid(char *ep_name, uint64_t *vuid)
{
#ifdef LWSCIIPC_GET_VUID
	struct lwsciipc_get_vuid get_vuid;
	int32_t ret = 0;

	strcpy(get_vuid.ep_name, ep_name);
	ret = ioctl(sciipc_fd, LWSCIIPC_IOCTL_GET_VUID, &get_vuid);
	if (ret < 0) {
		lwsciipc_err("%s: get_info ioctl failed\n", __func__);
		return LwSciError_IlwalidState;
	}

	*vuid = get_vuid.vuid;

	return LwSciError_Success;
#else
	return LwSciError_NotSupported;
#endif
}

/* QNX OS stub functions */

LwSciError lwsciipc_os_get_endpoint_mutex(
    struct LwSciIpcConfigEntry *entry,
    int32_t *mutexfd)
{
	return LwSciError_Success;
}

void lwsciipc_os_put_endpoint_mutex(int32_t *fd)
{
}

LwSciError lwsciipc_os_get_endpoint_access_info(const char *endpoint,
    LwSciIpcEndpointAccessInfo *info)
{
	return LwSciError_NotSupported;
}

/*====================================================================
 * Event polling API
 *====================================================================
 */
LwSciError lwsciipc_os_poll_event(void *os_args)
{
	lwsciipc_event_param_t *param = (lwsciipc_event_param_t *)os_args;
	LwSciError ret;

	if (param == NULL) {
        lwsciipc_err("Invalid parameter");
		ret = LwSciError_BadParameter;
		goto fail;
	}

	/* FIXME: select, epoll */
	ret = LwSciError_Success;

fail:
	return ret;
}


/*====================================================================
 * Timer APIs
 *====================================================================
 */

LwSciError lwsciipc_os_init_timer(void *os_args)
{
	lwsciipc_event_param_t *param = (lwsciipc_event_param_t *)os_args;
	LwSciError ret;

	if (param == NULL) {
        lwsciipc_err("Invalid parameter");
		ret = LwSciError_BadParameter;
		goto fail;
	}

	/* FIXME: timer_create */
	ret = LwSciError_Success;

fail:
	return ret;
}

LwSciError lwsciipc_os_start_timer(timer_t timer_id, uint64_t usecTimeout)
{
	LwSciError ret;

	if (timer_id == 0) {
		ret = LwSciError_BadParameter;
		goto fail;
	}

	/* FIXME: timer_settime */
	ret = LwSciError_Success;

fail:
	return ret;
}

LwSciError lwsciipc_os_stop_timer(timer_t timer_id)
{
	LwSciError ret;

	if (timer_id == 0) {
		ret = LwSciError_BadParameter;
		goto fail;
	}

	/* FIXME: timer_settime */
	ret = LwSciError_Success;

fail:
	return ret;
}

LwSciError lwsciipc_os_deinit_timer(timer_t timer_id)
{
	LwSciError ret;

	if (timer_id == 0) {
		ret = LwSciError_BadParameter;
		goto fail;
	}

	/* FIXME: timer_delete */
	ret = LwSciError_Success;

fail:
	return ret;
}

/*
 * in 64bit mode, unsigned long int is same with uint64_t
 */
char *lwsciipc_os_ultoa(uint64_t value, char *buffer, size_t size, int radix)
{
	(void)radix;

	snprintf(buffer, size, "%lx", value);

	return buffer;
}

/*
 * debug log msg with two strings and one value
 */
void lwsciipc_os_debug_2strs(const char *str1, const char *str2,
    int32_t ret)
{
    LWSCIIPC_DBG_STR(str1);
    LWSCIIPC_DBG_STRINT(str2, ret);
}

/*
 * error log msg with two strings and one value
 */
void lwsciipc_os_error_2strs(const char *str1, const char *str2,
    int32_t ret)
{
    LWSCIIPC_ERR_STR(str1);
    LWSCIIPC_ERR_STRINT(str2, ret);
}

