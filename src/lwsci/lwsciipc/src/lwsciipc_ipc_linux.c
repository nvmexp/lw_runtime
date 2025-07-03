/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
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
#include <semaphore.h>
#include <sys/syscall.h>
#include <signal.h>

#include <sivc.h>

#include "lwsciipc_common.h"
#include "lwsciipc_os_error.h"
#include "lwsciipc_ivc.h"
#include "lwsciipc_ipc.h"
#include "lwsciipc_log.h"

/* internal function definitions */
static LwSciError lwsciipc_ipc_update_shm_header(struct lwsciipc_ipc_handle *ipch);
static void lwsciipc_ipc_get_qd_info(struct lwsciipc_ipc_handle *ipch);
static LwSciError lwsciipc_ipc_setup_signal(struct lwsciipc_ipc_handle *ipch);
static void lwsciipc_ipc_notify(struct sivc_queue *sivc);
static LwSciError lwsciipc_ipc_init_resources(struct lwsciipc_ipc_handle *ipch,
	struct LwSciIpcConfigEntry *entry);
static struct lwsciipc_ipc_handle * ivc_to_lwsciipc_ipc_handle(
	struct sivc_queue *sivc);
static pid_t gettid(void);

static pid_t gettid(void)
{
	return syscall(SYS_gettid);
}

static struct lwsciipc_ipc_handle * ivc_to_lwsciipc_ipc_handle(struct sivc_queue *sivc)
{
    return (struct lwsciipc_ipc_handle *)((char *)(sivc) - (char *)&((struct lwsciipc_ipc_handle *)0)->sivc);
}


static LwSciError lwsciipc_ipc_check_end(struct lwsciipc_ipc_handle *ipch)
{
	struct lwsciipc_ipc_shm_header *head = ipch->shm_header;
	char dirpath[LWSCIIPC_MAX_ENDPOINT_NAME] = {0,};
        char *path = &dirpath[0];
        struct stat sts;
	uint32_t id;

	if (ipch->shm_header == NULL) {
		lwsciipc_err("Shm header ptr was not initialized");
		return LwSciError_BadParameter;
	}

	id = ipch->id;

        /* check if current endpoint is oclwpied by other process */
	if (ipch->backend == LWSCIIPC_BACKEND_IPC) {
		if (head->pid[id] != 0) {
			sprintf(path, "/proc/%d", head->pid[id]);
			if (stat(path, &sts) == -1 && errno == ENOENT) {
				lwsciipc_err("pid not 0, but process doesn't exist, (pid:%d)",
						head->pid[id]);
				return LwSciError_IlwalidState;
			} else {
				/* other process is using this endpoint. we can't open endpoint */
				lwsciipc_err("endpoint is already oclwpied by (pid:%d)",
						head->pid[id]);
				return LwSciError_Busy;
			}
		}
	}

	/* check if current endpoint is oclwpied by other thread */
	if (ipch->backend == LWSCIIPC_BACKEND_ITC) {
		if (head->pid[id] != 0 && head->tid[id] != 0) {

			sprintf(path, "/proc/%d/task/%d", head->pid[id], head->tid[id]);
			if (stat(path, &sts) == -1 && errno == ENOENT) {
				lwsciipc_err("tid not 0, but thread doesn't exist, (pid:%d, tid:%d)",
						head->pid[id], head->tid[id]);
				return LwSciError_IlwalidState;
			} else {
				/* other process or thread is using this endpoint. we can't open endpoint */
				lwsciipc_err("endpoint is already oclwpied by (pid:%d, tid:%d)",
						head->pid[id], head->tid[id]);
				return LwSciError_Busy;

			}
		}
	}

	return LwSciError_Success;
}

/**
 * Update Shared memory header
 *
 * pid, tid for both endpoints
 * queue data area info (nframes, frame_size)
 * To decide endpoint id based on current pid/tid info on shared memory
 */
static LwSciError lwsciipc_ipc_update_shm_header(struct lwsciipc_ipc_handle *ipch)
{
	struct lwsciipc_ipc_shm_header *head = ipch->shm_header;
	struct lwsciipc_ivc_info *ivci = &ipch->ivc_info;
	int32_t ret = -1;

        /* check if current endpoint is already oclwpied */
	ret = lwsciipc_ipc_check_end(ipch);
	if ((ret != LwSciError_Success) && (ret != LwSciError_IlwalidState)) {
		return ret;
	}

	/* increment refcnt only if endpoint was not oclwpied by any process */
	if (ret == LwSciError_Success) {
		head->refcnt++;
	}

	head->pid[ipch->id] = getpid();

	if (ipch->backend == LWSCIIPC_BACKEND_ITC)
		head->tid[ipch->id] = gettid();

	head->nframes = ivci->nframes;
	head->frame_size = ivci->frame_size;

	lwsciipc_dbg(" own id : %d, pid: %d, tid: %d refcnt: %d",
		ipch->id, head->pid[ipch->id], head->tid[ipch->id], head->refcnt);
	lwsciipc_dbg("peer id : %d, pid: %d, tid: %d",
		ipch->peer_id, head->pid[ipch->peer_id], head->tid[ipch->peer_id]);

	return LwSciError_Success;
}

/**
 * Get queue data area info
 *
 * rx_base, tx_base
 *
 * if id=0, rx_base is the first qd region
 */
static void lwsciipc_ipc_get_qd_info(struct lwsciipc_ipc_handle *ipch)
{
	struct lwsciipc_ipc_shm_header *head = ipch->shm_header;
	struct lwsciipc_ivc_info *ivci = &ipch->ivc_info;

	if (ipch->id) {
		ipch->rx_base = (uintptr_t)&head[1] + ivci->queue_size;
		ipch->tx_base = (uintptr_t)&head[1];
	}
	else {
		ipch->rx_base = (uintptr_t)&head[1];
		ipch->tx_base = (uintptr_t)&head[1] + ivci->queue_size;
	}

	lwsciipc_dbg("id : %d, rx_base: %tx, tx_base: %tx",
		ipch->id, ipch->rx_base, ipch->tx_base);
}

/**
 * Set up notification signals for both endpoints
 *
 * use message queue based on channel entry name
 */
static LwSciError lwsciipc_ipc_setup_signal(struct lwsciipc_ipc_handle *ipch)
{
	mqd_t mq;
	struct mq_attr attr = {0};
	char *mq_name;

	attr.mq_flags = O_RDWR;
	//attr.mq_maxmsg = 2;
	attr.mq_maxmsg = ipch->ivc_info.nframes;
	attr.mq_msgsize = LWSCIIPC_MQ_SIZE;

	sprintf(ipch->mq_name[0], "/%s0", ipch->dev_name);
	sprintf(ipch->mq_name[1], "/%s1", ipch->dev_name);

	if (ipch->id)
		mq_name = ipch->mq_name[1];
	else
		mq_name = ipch->mq_name[0];

	lwsciipc_dbg("opening mq %s\n", mq_name);

	mq = mq_open(mq_name, O_CREAT | O_NONBLOCK | O_RDWR, IPC_MODE, &attr);
	if (mq < 0) {
		lwsciipc_err("mq_open failed: %s", strerror(errno));
		return ErrnoToLwSciErr(errno);
	}
	ipch->own_mq = mq;

	if (ipch->id)
		mq_name = ipch->mq_name[0];
	else
		mq_name = ipch->mq_name[1];

	lwsciipc_dbg("opening mq %s\n", mq_name);

	mq = mq_open(mq_name, O_CREAT | O_NONBLOCK | O_RDWR, IPC_MODE, &attr);
	if (mq < 0) {
		mq_close(ipch->own_mq);
		ipch->own_mq = 0;
		lwsciipc_err("mq_open failed: %s", strerror(errno));
		return ErrnoToLwSciErr(errno);
	}
	ipch->peer_mq = mq;

	lwsciipc_dbg("own_mq:%d, peer_mq:%d", ipch->own_mq, ipch->peer_mq);

	return LwSciError_Success;
}

#if 0
static void lwsciipc_ipc_dump_mq_info(mqd_t mq)
{
	struct mq_attr attr = {0};
	int ret;

	ret = mq_getattr(mq, &attr);
	if (ret < 0)
		lwsciipc_err("mq_getattr on %d failed\n", mq);

	lwsciipc_dbg("maxmsg = %ld, maxmsgsize =%ld\n", attr.mq_maxmsg, attr.mq_msgsize);
	lwsciipc_dbg("flags  = 0x%lx, lwrmsgs =%ld\n", attr.mq_flags, attr.mq_lwrmsgs);
}
#endif

static void lwsciipc_ipc_notify(struct sivc_queue *sivc)
{
	struct lwsciipc_ipc_handle *ipch = ivc_to_lwsciipc_ipc_handle(sivc);
	int ret;

	if ((ipch == NULL) || (sivc == NULL)) {
		lwsciipc_err("IPC handle is NULL");
		return;
	}

	lwsciipc_dbg("%d : ivc notify - mq_send on peer_mq\n", gettid());
	//lwsciipc_ipc_dump_mq_info(ipch->peer_mq);
	/* send signal to peer */
	ret = mq_send(ipch->peer_mq, ipch->mq_data, LWSCIIPC_MQ_SIZE, 0);
	if ( (ret == -1) && (errno != EAGAIN) ) {
		lwsciipc_err("mq_send failed: %s", strerror(errno));
	}
}

static LwSciError lwsciipc_ipc_init_resources(struct lwsciipc_ipc_handle *ipch,
	struct LwSciIpcConfigEntry *entry)
{
	struct lwsciipc_ivc_info *ivci = &ipch->ivc_info;
	int fd;
	void *addr;
	int ret;

	ipch->id = entry->id;
	ipch->peer_id = entry->id^1;

	ivci->nframes = entry->nFrames;
	ivci->frame_size = entry->frameSize;
	/* get 64B aligned queue size */
	ivci->queue_size = sivc_fifo_size(
		ivci->nframes, ivci->frame_size);

	/* double queue data and shm header */
	ipch->shm_size = ivci->queue_size * 2 +
		sizeof(struct lwsciipc_ipc_shm_header) + SIVC_ALIGN_MASK;

	lwsciipc_dbg("nframes:%d, frame_sz:%d, queue_sz:0x%x, shm_size:0x%x",
		ivci->nframes, ivci->frame_size, ivci->queue_size, (unsigned int)ipch->shm_size);

	lwsciipc_dbg("creating shm area\n");
	/* /dev/shmem/{entry} */
	fd = shm_open(ipch->dev_name, O_CREAT|O_EXCL|O_RDWR, IPC_MODE);
	if (fd == -1) {
		if (errno == EEXIST) {
			lwsciipc_dbg("shm area exist, so open it\n");
			/* memory object exists */
			fd = shm_open(ipch->dev_name, O_RDWR, IPC_MODE);
			if (fd == -1) {
				lwsciipc_err("shm_open is failed: %s", strerror(errno));
				return ErrnoToLwSciErr(errno);
			}
		}
		else {
			/* generic error */
			lwsciipc_err("shm_open is failed: %s", strerror(errno));
			return ErrnoToLwSciErr(errno);
		}
	}
	else {
		lwsciipc_dbg("shm area created, truncate it\n");
		/* memory object is created */
		/* resize shared memory */
		if (ftruncate(fd, ipch->shm_size) == -1) {
			lwsciipc_err("ftruncate is failed: %s\n", strerror(errno));
			return ErrnoToLwSciErr(errno);
		}
	}
	ipch->shm_fd = fd;

	/* Map the share memory area */
	addr = mmap(0, ipch->shm_size, PROT_READ | PROT_WRITE,
		MAP_SHARED, ipch->shm_fd, 0);
	if (addr == MAP_FAILED) {
		lwsciipc_err("mmap is failed: %s", strerror(errno));
		return ErrnoToLwSciErr(errno);
	}
	else
		ipch->shm = (uintptr_t)addr;

	/* get 64B aligned buffer to use ivc common lib */
	ipch->shm_aligned = ((uintptr_t)addr + SIVC_ALIGN_MASK) & ~SIVC_ALIGN_MASK;
	ipch->shm_header = (struct lwsciipc_ipc_shm_header *)ipch->shm_aligned;

	lwsciipc_dbg("[%d] shm_fd: %d, mmap: addr:0x%tx, aligned:0x%tx, header:0x%tx",
		ipch->id, ipch->shm_fd, (uintptr_t)addr,
		(uintptr_t)ipch->shm_aligned, (uintptr_t)ipch->shm_header);

	/* define endpoint id */
	ret = lwsciipc_ipc_update_shm_header(ipch);
	if (ret != LwSciError_Success) {
		return ret;
	}

	/* setup semaphore for data xchange signalling */
	ret = lwsciipc_ipc_setup_signal(ipch);
	if (ret != LwSciError_Success) {
		return ret;
	}

	/* setup rx/tx_base for ipc queue data area */
	lwsciipc_ipc_get_qd_info(ipch);

	return LwSciError_Success;
}

LwSciError lwsciipc_ipc_get_endpoint_info(const struct lwsciipc_ipc_handle *ipch,
	LwSciIpcEndpointInfo *info)
{
	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not initialized");
		return LwSciError_NotInitialized;
	}

	info->nframes = ipch->sivc.nframes;
	info->frame_size = ipch->sivc.frame_size;

	return LwSciError_Success;
}

LwSciError lwsciipc_ipc_get_eventnotifier(
	struct lwsciipc_ipc_handle *ipch,
	LwSciEventNotifier **eventNotifier)
{
	LwSciError ret;

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		ret = LwSciError_BadParameter;
		goto fail;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not initialized");
		ret = LwSciError_NotInitialized;
		goto fail;
	}

	ipch->nativeEvent.fd = ipch->own_mq;
	ret = ipch->eventService->CreateNativeEventNotifier(ipch->eventService,
		&ipch->nativeEvent, &ipch->eventNotifier);
	if (ret != LwSciError_Success) {
		goto fail;
	}

	*eventNotifier = ipch->eventNotifier;

fail:
	return ret;
}

LwSciError lwsciipc_ipc_get_eventfd(const struct lwsciipc_ipc_handle *ipch,
	int32_t *fd)
{
	if (ipch == NULL || fd == NULL) {
		lwsciipc_err("IPC handle or FD is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not initialized");
		return LwSciError_NotInitialized;
	}

	*fd = ipch->own_mq;

	return LwSciError_Success;
}

LwSciError lwsciipc_ipc_read(struct lwsciipc_ipc_handle *ipch, void *buf,
	uint32_t size, uint32_t *bytes)
{
	int32_t err;
	LwSciError ret;

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return LwSciError_NotInitialized;
	}

	err = sivc_read(&ipch->sivc, buf, size);

	if (err < 0) {
		*bytes = 0U;
		update_sivc_err(err);
	} else {
		*bytes = size;
		ret = LwSciError_Success;
	}

	return ret;
}

const volatile void *lwsciipc_ipc_read_get_next_frame(
    struct lwsciipc_ipc_handle *ipch)
{
	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return NULL;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return NULL;
	}

	return sivc_get_read_frame(&ipch->sivc);
}

LwSciError lwsciipc_ipc_read_advance(struct lwsciipc_ipc_handle *ipch)
{
	int32_t err;
	LwSciError ret;

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return LwSciError_NotInitialized;
	}

	err = sivc_read_advance(&ipch->sivc);

	if (err < 0) {
		update_sivc_err(err);
	} else {
		ret = LwSciError_Success;
	}

	return ret;
}

LwSciError lwsciipc_ipc_write(struct lwsciipc_ipc_handle *ipch, const void *buf,
	uint32_t size, uint32_t *bytes)
{
	int32_t err;
	LwSciError ret;

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return LwSciError_NotInitialized;
	}

	err = sivc_write(&ipch->sivc, buf, size);

	if (err < 0) {
		*bytes = 0U;
		update_sivc_err(err);
	} else {
		*bytes = size;
		ret = LwSciError_Success;
	}

	return ret;
}

volatile void *lwsciipc_ipc_write_get_next_frame(
    struct lwsciipc_ipc_handle *ipch)
{
	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return NULL;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return NULL;
	}

	return sivc_get_write_frame(&ipch->sivc);
}

LwSciError lwsciipc_ipc_write_advance(struct lwsciipc_ipc_handle *ipch)
{
	int32_t err;
	LwSciError ret;

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return LwSciError_NotInitialized;
	}

	err = sivc_write_advance(&ipch->sivc);

	if (err < 0) {
		update_sivc_err(err);
	} else {
		ret = LwSciError_Success;
	}

	return ret;
}

/* peek in the next rx buffer at offset off, the count bytes */
LwSciError lwsciipc_ipc_read_peek(struct lwsciipc_ipc_handle *ipch, void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes)
{
	int32_t err;
	LwSciError ret;

	err = sivc_read_peek(&ipch->sivc, buf, offset, count);

	if (err < 0) {
		*bytes = 0U;
		update_sivc_err(err);
	} else {
		*bytes = count;
		ret = LwSciError_Success;
	}

	return ret;
}

/* poke in the next tx buffer at offset off, the count bytes */
LwSciError lwsciipc_ipc_write_poke(struct lwsciipc_ipc_handle *ipch, const void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes)
{
	int32_t err;
	LwSciError ret;

	err = sivc_write_poke(&ipch->sivc, buf, offset, count);

	if (err < 0) {
		*bytes = 0U;
		update_sivc_err(err);
	} else {
		*bytes = count;
	ret = LwSciError_Success;
	}

	return ret;
}

LwSciError lwsciipc_ipc_open_endpoint(struct lwsciipc_ipc_handle **ipcp,
	struct LwSciIpcConfigEntry *entry)
{
	struct lwsciipc_ipc_handle *ipch;
	sem_t *sem = NULL;
	int ret = 0;
	int32_t status = -1;
	struct lwsciipc_ivc_info *ivci;
	uint32_t slen;

	ipch = malloc(sizeof(struct lwsciipc_ipc_handle));
	if (ipch == NULL) {
		lwsciipc_err("Failed to malloc");
		status = LwSciError_InsufficientMemory;
		goto fail;
	}
	lwsciipc_dbg("ipch: %tx", (uintptr_t)ipch);

/* clear handle structure */
	memset(ipch, 0, sizeof(struct lwsciipc_ipc_handle));
	ipch->backend = entry->backend;
	ipch->entry = entry;

	ivci = &ipch->ivc_info;

	slen = strlen(entry->devName);
	if (slen == 0) {
		status = LwSciError_BadParameter;
		goto fail;
	}
	if (slen > LWSCIIPC_MAX_ENDPOINT_NAME) {
		slen = LWSCIIPC_MAX_ENDPOINT_NAME;
	}
	(void)strncpy(ipch->dev_name, entry->devName, sizeof(ipch->dev_name));
	ipch->dev_name[slen] = '\0';

	/* /dev/sem/{entry} */
	sem = sem_open(ipch->dev_name, O_CREAT, IPC_MODE, 1);
	if (sem == SEM_FAILED) {
		lwsciipc_err("Failed to open semaphore %s: %s",
			ipch->dev_name, strerror(errno));
		lwsciipc_ipc_close_endpoint(ipch);
		ipch = NULL; /* ipch is freed by close_endpoint() */
		status = LwSciError_IlwalidState;
		goto fail;
	}
	ipch->sem = sem;
	lwsciipc_dbg("sem:%s, %tx", ipch->dev_name, (uintptr_t)sem);

	ret = sem_wait(sem);
	lwsciipc_dbg("sem_wait done");
	if (ret == -1) {
		lwsciipc_err("sem_wait failed: %s", strerror(errno));
		lwsciipc_ipc_close_endpoint(ipch);
		ipch = NULL; /* ipch is freed by close_endpoint() */
		status = LwSciError_IlwalidState;
		goto fail;
	}

	/* critical section to init resources with named semaphore */
	ret = lwsciipc_ipc_init_resources(ipch, entry);
	if (ret != LwSciError_Success) {
                lwsciipc_err("init resources is failed: %d", ret);
                (void)sem_post(sem);
		lwsciipc_ipc_close_endpoint(ipch);
		ipch = NULL; /* ipch is freed by close_endpoint() */
		status = ret;
		goto fail;
	}

	ret = sem_post(sem);
	lwsciipc_dbg("sem_post done");
	if (ret == -1) {
		lwsciipc_err("sem_post failed: %s", strerror(errno));
		lwsciipc_ipc_close_endpoint(ipch);
		ipch = NULL; /* ipch is freed by close_endpoint() */
		status = LwSciError_IlwalidState;
		goto fail;
	}

	ret = sivc_init(&ipch->sivc, ipch->rx_base, ipch->tx_base,
		ivci->nframes, ivci->frame_size, lwsciipc_ipc_notify, NULL, NULL);

	if (ret != 0) {
		lwsciipc_err("ivc_init failed: %d", ret);
		lwsciipc_ipc_close_endpoint(ipch);
		ipch = NULL; /* ipch is freed by close_endpoint() */
		status = LwSciError_IlwalidState;
		goto fail;
	}

	*ipcp = ipch;
	ipch->is_open = true;
	memcpy(ipch->mq_data, LWSCIIPC_MQ_DATA, sizeof(LWSCIIPC_MQ_DATA));

	lwsciipc_dbg("done\n");

	status = LwSciError_Success;

fail:
        if (status != LwSciError_Success) {
                if (ipch != NULL) {
                        /*
                         * free ipch if it was allocated and hasn't been
                         * freed by lwsciipc_ipc_close_endpoint().
                         */
                        free(ipch);
                }
                ivci = NULL;
                ipch = NULL;
                *ipcp = NULL;
        }
        return status;
}

void lwsciipc_ipc_bind_eventservice(struct lwsciipc_ipc_handle *ipch,
    LwSciEventService *eventService)
{
    ipch->eventService = eventService;
}

LwSciError lwsciipc_ipc_open_endpoint_with_eventservice(
	struct lwsciipc_ipc_handle **ipcp,
	struct LwSciIpcConfigEntry *entry,
	LwSciEventService *eventService)
{
	struct lwsciipc_ipc_handle *ipch = NULL;
	LwSciError ret;

	ret =  lwsciipc_ipc_open_endpoint(ipcp, entry);
	if (ret == LwSciError_Success) {
		ipch = *ipcp;
        lwsciipc_ipc_bind_eventservice(ipch, eventService);
	}
	else {
		goto fail;
	}

fail:
	return ret;
}


void lwsciipc_ipc_close_endpoint(struct lwsciipc_ipc_handle *ipch)
{
	struct lwsciipc_ipc_shm_header *head;
	int ret = 0;
	uint32_t refcnt, isowner = 0;

	lwsciipc_dbg("closing ipc channel");

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return;
	}

	head = ipch->shm_header;

	if (ipch->sem != 0) {
		ret = sem_wait(ipch->sem);
		lwsciipc_dbg("sem_wait done");
		if (ret == -1) {
			lwsciipc_err("sem_wait failed: %s", strerror(errno));
		}
	}
	else {
		return;
	}

	/* check to make sure that the same process/thread that opened the
	 * endpoint is closing the endpoint. Do not increment refcnt if
	 * some other process/thread is trying to close te endpoint
	 */
	if (head->pid[ipch->id] == getpid()) {
		if (ipch->backend == LWSCIIPC_BACKEND_IPC) {
			isowner = 1;
			head->refcnt--;
		} else if (ipch->backend == LWSCIIPC_BACKEND_ITC) {
			if (head->tid[ipch->id] == gettid()) {
				isowner = 1;
				head->refcnt--;
			}
		}
	}

	refcnt = head->refcnt;

	if (ipch->shm) {
		lwsciipc_dbg("unmap the shm area\n");
		if (isowner == 1) {
			if (refcnt > 0) {
				/* clear only the current endpoint specific data in shm */
				head->pid[ipch->id] = 0; /* clear process id */
				head->chid[ipch->id] = 0; /* clear channel id */
				head->tid[ipch->id] = 0; /* clear thread id */
			} else {
				/* clear entire shared memory header */
				memset(ipch->shm_header, 0, sizeof(struct lwsciipc_ipc_shm_header));
			}
		}
		ret = munmap((void *)ipch->shm, ipch->shm_size);
		if (ret == -1) {
			lwsciipc_err("Failed to unmap: (0x%tx, 0x%lx)",
				ipch->shm, ipch->shm_size);
		}
		ipch->shm = 0;
	}

	if (ipch->shm_fd) {
		lwsciipc_dbg("close the shm area\n");
		close(ipch->shm_fd);
		ipch->shm_fd = 0;
	}

	if (ipch->own_mq) {
		lwsciipc_dbg("close mq %d\n", ipch->own_mq);
		ret = mq_close(ipch->own_mq);
		if (ret == -1) {
			lwsciipc_err("mq_close is failed: %s", strerror(errno));
		}
		ipch->own_mq = 0;
	}

	if (ipch->peer_mq) {
		lwsciipc_dbg("close mq %d\n", ipch->peer_mq);
		ret = mq_close(ipch->peer_mq);
		if (ret == -1) {
			lwsciipc_err("mq_close is failed: %s", strerror(errno));
		}
		ipch->peer_mq = 0;
	}

	if (refcnt != 0) {
		lwsciipc_dbg("endpoint is in use\n");
		sem_post(ipch->sem);

		lwsciipc_dbg("close sync sem\n");
		ret = sem_close(ipch->sem);
		if (ret == -1) {
			lwsciipc_err("sem_close is failed: %s", strerror(errno));
		}

		/* clear handle structure */
		memset(ipch, 0, sizeof(struct lwsciipc_ipc_handle));
		free(ipch);

		return;
	}

	lwsciipc_dbg("unlink mq %s\n", ipch->mq_name[ipch->id]);
	ret = mq_unlink(ipch->mq_name[ipch->id]);
	if (ret == -1) {
		lwsciipc_dbg("mq_unlink(%s) is failed: %s",
			ipch->mq_name[ipch->id], strerror(errno));
	}

	lwsciipc_dbg("unlink mq %s\n", ipch->mq_name[ipch->peer_id]);
	ret = mq_unlink(ipch->mq_name[ipch->peer_id]);
	if (ret == -1) {
		lwsciipc_dbg("mq_unlink(%s) is failed: %s",
			ipch->mq_name[ipch->peer_id], strerror(errno));
	}

	if (strlen(ipch->dev_name) <= LWSCIIPC_MAX_ENDPOINT_NAME) {
		/* remove memory object */
		lwsciipc_dbg("unlink shm %s\n", ipch->dev_name);
		ret = shm_unlink(ipch->dev_name);
		if (ret == -1) {
			lwsciipc_dbg("shm_unlink(%s) is failed: %s",
				ipch->dev_name, strerror(errno));
		}
	}

	if (ipch->sem) {
		sem_post(ipch->sem);
		lwsciipc_dbg("close sync sem\n");
		ret = sem_close(ipch->sem);
		if (ret == -1) {
			lwsciipc_err("sem_close is failed: %s", strerror(errno));
		}
		ipch->sem = NULL;
		lwsciipc_dbg("unlink sync sem\n");
		ret = sem_unlink(ipch->dev_name);
		if (ret == -1) {
			lwsciipc_dbg("sem_unlink(%s) is failed: %s",
				ipch->dev_name, strerror(errno));
		}
	}

	/* clear handle structure */
	memset(ipch, 0, sizeof(struct lwsciipc_ipc_handle));
	free(ipch);

	lwsciipc_dbg("done");
}

void lwsciipc_ipc_reset_endpoint(struct lwsciipc_ipc_handle *ipch)
{
	lwsciipc_dbg("resetting ivc channel");

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return;
	}

	sivc_reset(&ipch->sivc);

	lwsciipc_dbg("done");
}

LwSciError lwsciipc_ipc_check_read(struct lwsciipc_ipc_handle *ipch)
{
	LwSciError ret = LwSciError_Success;

	if (ipch == NULL) {
		lwsciipc_err("IVC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return LwSciError_NotInitialized;
	}

	if (ipch->prev_conn == LW_SCI_IPC_EVENT_CONN_RESET) {
		ret = LwSciError_ConnectionReset;
	}
	else if (sivc_can_read(&ipch->sivc) == false) {
		ret = LwSciError_InsufficientMemory;
	}

	return ret;
}

LwSciError lwsciipc_ipc_check_write(struct lwsciipc_ipc_handle *ipch)
{
	LwSciError ret = LwSciError_Success;

	if (ipch == NULL) {
		lwsciipc_err("IVC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return LwSciError_NotInitialized;
	}

	if (ipch->prev_conn == LW_SCI_IPC_EVENT_CONN_RESET) {
		ret = LwSciError_ConnectionReset;
	}
	else if (sivc_can_write(&ipch->sivc) == false) {
		ret = LwSciError_InsufficientMemory;
	}

	return ret;
}

LwSciError lwsciipc_ipc_get_event(struct lwsciipc_ipc_handle *ipch,
	uint32_t *events, struct lwsciipc_internal_handle *inth)
{
	int32_t ret;
	uint32_t conn = 0U;
	uint32_t value = 0U;

	if (ipch == NULL) {
		lwsciipc_err("IPC handle is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not opened yet");
		return LwSciError_NotInitialized;
	}

	lwsciipc_dbg("%d : receive ivc notification -  mq_receive on own_mq\n", gettid());
	//lwsciipc_ipc_dump_mq_info(ipch->own_mq);

mq_read:
	ret = mq_receive(ipch->own_mq, ipch->mq_data, LWSCIIPC_MQ_SIZE, NULL);
	if (ret >= 0) {
		lwsciipc_dbg("read all mq messages\n");
		goto mq_read;
	}
	if ( (ret == -1) && (errno != EAGAIN) ) {
		lwsciipc_err("mq_receive failed: %d %s", errno, strerror(errno));
	}

	if (sivc_need_notify(&ipch->sivc) == false) {
		conn = LW_SCI_IPC_EVENT_CONN_EST;
		if (sivc_can_write(&ipch->sivc)) {
			value |= LW_SCI_IPC_EVENT_WRITE;
		}
		if (sivc_can_read(&ipch->sivc)) {
			value |= LW_SCI_IPC_EVENT_READ;
		}
	}
	else {
		(void)lwsciipc_os_mutex_lock(&inth->wrMutex); /* wr pos/refcnt */
		(void)lwsciipc_os_mutex_lock(&inth->rdMutex); /* rd pos/refcnt */
		ret = sivc_notified(&ipch->sivc);
		(void)lwsciipc_os_mutex_unlock(&inth->rdMutex);
		(void)lwsciipc_os_mutex_unlock(&inth->wrMutex);
		if (ret == 0) {
			conn = LW_SCI_IPC_EVENT_CONN_EST;
			/* check buffer status again after establishment */
			if (sivc_can_write(&ipch->sivc)) {
				value |= LW_SCI_IPC_EVENT_WRITE;
			}
			if (sivc_can_read(&ipch->sivc)) {
				value |= LW_SCI_IPC_EVENT_READ;
			}
		}
		else {
			conn = LW_SCI_IPC_EVENT_CONN_RESET;
		}
	}

	if ((conn & LW_SCI_IPC_EVENT_CONN_MASK) != ipch->prev_conn) {
		*events = (value | conn);
		lwsciipc_dbg("prev_conn = %u, conn = %u, events = %u\n", ipch->prev_conn, conn,
			*events);
		ipch->prev_conn = conn;
	}
	else {
		*events = value;
	}

	return LwSciError_Success;
}

bool lwsciipc_ipc_can_read(struct lwsciipc_ipc_handle *ipch)
{
	return sivc_can_read(&ipch->sivc);
}

bool lwsciipc_ipc_can_write(struct lwsciipc_ipc_handle *ipch)
{
	return sivc_can_write(&ipch->sivc);
}

LwSciError lwsciipc_ipc_set_qnx_pulse_param(
	struct lwsciipc_ipc_handle *ipch, int32_t coid,
	int16_t priority, int16_t code, void *value)
{
	return LwSciError_NotSupported;
}

LwSciError lwsciipc_ipc_endpoint_get_auth_token(
    struct lwsciipc_ipc_handle *ipch, LwSciIpcEndpointAuthToken *authToken)
{
	if (ipch == NULL || authToken == NULL) {
		lwsciipc_err("IPC handle or authToken is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not initialized");
		return LwSciError_NotInitialized;
	}

	*authToken = ipch->own_mq;

	return LwSciError_Success;
}

LwSciError lwsciipc_ipc_endpoint_get_vuid(
    struct lwsciipc_ipc_handle *ipch, LwSciIpcEndpointVuid *vuid)
{
	LwSciError ret;
	uint64_t local_vuid;

	if (ipch == NULL || vuid == NULL) {
		lwsciipc_err("IPC handle or authToken is NULL");
		return LwSciError_BadParameter;
	}

	if (ipch->is_open != true) {
		lwsciipc_err("Endpoint is not initialized");
		return LwSciError_NotInitialized;
	}

	ret = lwsciipc_os_get_vuid(ipch->entry->epName, &local_vuid);
	if (ret != LwSciError_Success) {
		lwsciipc_err("get_vuid not supported\n");
		return ret;
	}
	*vuid = local_vuid;

	return LwSciError_Success;
}

