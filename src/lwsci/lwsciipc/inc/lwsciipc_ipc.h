/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_IPC_H
#define INCLUDED_LWSCIIPC_IPC_H

#include <pthread.h>
#include <semaphore.h>

#ifdef __QNX__
#include <lwscievent_qnx.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif
#endif /* __QNX__ */

#ifdef LINUX
#include <mqueue.h>
#include <lwscievent_linux.h>

#define LWSCIIPC_MQ_SIZE   16U
#define LWSCIIPC_MQ_DATA   "1234567891234\0"

/* 0660 (S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP) */
#define IPC_MODE    0x1B0U

/* for Inter-Thread/Process */
#define IVC_CHHDR_FIELDS 16

struct lwsciipc_ipc_shm_header {
    pid_t pid[2]; /* process id for each endpoints */
    pid_t chid[2]; /* channel id for each endpoints */
    pid_t tid[2]; /* thread id for each endpoints */
    uint32_t nframes;
    uint32_t frame_size;
    uint32_t refcnt;
    uint32_t rsvd[IVC_CHHDR_FIELDS - 9];
};
#endif /* LINUX */

/* 512 Bytes */
struct lwsciipc_ipc_handle
{
    /*
    sivc-instance.h
    struct sivc_queue {
        volatile struct sivc_fifo_header* recv_fifo;
        volatile struct sivc_fifo_header* send_fifo;
        uint32_t w_pos;
        uint32_t r_pos;
        uint32_t nframes;
        uint32_t frame_size;
        sivc_notify_function notify;
        sivc_cache_ilwalidate_function cache_ilwalidate;
        sivc_cache_flush_function cache_flush;
    };
    lwsciipc_ivc_handle extends sivc_queue.
    sivc must be the first data of this structure.
    */
    struct sivc_queue sivc;
    /*
    struct ivc_shared_info {
        uint32_t nframes;
        uint32_t frame_size;
        uint32_t queue_offset;
        uint32_t queue_size;
        uint32_t area_size;
        uint64_t area_pa;
        bool     rx_first;
        uint16_t irq;
    };
    */
    struct LwSciIpcConfigEntry *entry;
    struct lwsciipc_ivc_info ivc_info;

    /* use channel entry name for named device(sem/shmem) */
    /* used to create/open shm/sem dev */
    char dev_name[LWSCIIPC_MAX_ENDPOINT_NAME + 1];
    uintptr_t shm; /* mmapped shared memory address */
    size_t shm_size; /* total shm size */
    int32_t shm_fd;

    uint32_t id;      /* current endpoint id : 0 or 1 */
    uint32_t peer_id; /* peer endpoint id    : 0 or 1 */

    uintptr_t rx_base;    /* rx_base addr of qd */
    uintptr_t tx_base;    /* tx_base addr of qd */

    uint32_t prev_conn; /* previous connection status */

    int32_t mutexfd; /* io-lwsciipc mutex fd */

    LwSciEventService *eventService;
    LwSciEventNotifier *eventNotifier;

#ifdef LINUX
    LwSciNativeEvent nativeEvent;
    uintptr_t shm_aligned; /* 64B aligned address for ivclib */
    sem_t *sem; /* semaphore to initialize resources */
    struct lwsciipc_ipc_shm_header *shm_header;
    char mq_name[2][LWSCIIPC_MAX_ENDPOINT_NAME + 3];
    mqd_t own_mq;
    mqd_t peer_mq;
    char mq_data[LWSCIIPC_MQ_SIZE];
    uint32_t backend;
#endif /* LINUX */

#ifdef __QNX__
    LwSciQnxNativeEvent qnxEvent;
    int32_t pulsefd; /* io-lwsciipc pulse fd */
    int32_t rmState; /* RM_STATE_XXX pulse RM state (local) : pulseState */
    uint32_t abilityId; /* LwSciIpcEndpoint custom abiltyId */

    uint32_t pulseResetFlag; /* pulse reset flag */
    /* setResetFlagEvent by SetPulseParam */
    struct sigevent setResetFlagEvent;
    /* registered setResetFlagEvent for pulse connection resmgr */
    struct sigevent regSetResetFlagEvent;
    /* userEvent by SetPulseParam from application */
    struct sigevent userEvent;
    /* registered userEvent for pulse connection resmgr */
    struct sigevent regProgressEvent;
    /* registered userEvent for peer endpoint */
    struct sigevent regPeerEvent;
    int32_t serverChid; /* local chid to be sent to peer */
    int32_t pid; /* current local pid */
    int32_t clientCoid; /* client coid which is attached to peer(remote) chid/pid */
    int32_t remoteChid; /* peer chid */
    int32_t remotePid; /* peer pid */
    int32_t remoteScoid; /* remote server's coid which is returned by ConnectServerInfo w/ clientCoid */
    int32_t serverScoid; /* local server's coid which is connected to remote coid of peer endpoint */
    struct sigevent remoteEvent; /* event which is created by SetPulseParam of peer endpoint */
    int32_t coid;    /* connection id for pulse event */
#endif /* __QNX__ */

    uint32_t is_open; /* channel is initialized ? (LwBoolTrue/LwBoolFalse) */
} __attribute__((aligned(8)));


/**
 * lwsciipc_ipc.c : inter-thread/process functions
 *
 */
LwSciError lwsciipc_ipc_open_endpoint(struct lwsciipc_ipc_handle **ipcp,
    struct LwSciIpcConfigEntry *entry);
void lwsciipc_ipc_bind_eventservice(struct lwsciipc_ipc_handle *ipch,
    LwSciEventService *eventService);
LwSciError lwsciipc_ipc_open_endpoint_with_eventservice(
    struct lwsciipc_ipc_handle **ipcp,
    struct LwSciIpcConfigEntry *entry,
    LwSciEventService *eventService);
void lwsciipc_ipc_close_endpoint(struct lwsciipc_ipc_handle *ipch);
void lwsciipc_ipc_reset_endpoint(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_read(struct lwsciipc_ipc_handle *ipch, void *buf,
    uint32_t size, uint32_t *bytes);
const volatile void *lwsciipc_ipc_read_get_next_frame(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_read_advance(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_write(struct lwsciipc_ipc_handle *ipch,
    const void *buf, uint32_t size, uint32_t *bytes);
volatile void *lwsciipc_ipc_write_get_next_frame(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_write_advance(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_read_peek(struct lwsciipc_ipc_handle *ipch, void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes);
LwSciError lwsciipc_ipc_write_poke(struct lwsciipc_ipc_handle *ipch, const void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes);

LwSciError lwsciipc_ipc_get_endpoint_info(const struct lwsciipc_ipc_handle *ipch,
    LwSciIpcEndpointInfo *info);
LwSciError lwsciipc_ipc_get_eventnotifier(
    struct lwsciipc_ipc_handle *ipch,
    LwSciEventNotifier **eventNotifier);
#ifdef LINUX
LwSciError lwsciipc_ipc_get_eventfd(const struct lwsciipc_ipc_handle *ipch,
    int32_t *fd);
#endif /* LINUX */
LwSciError lwsciipc_ipc_check_read(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_check_write(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_get_event(struct lwsciipc_ipc_handle *ipch,
    uint32_t *event, struct lwsciipc_internal_handle *inth);
bool lwsciipc_ipc_can_read(struct lwsciipc_ipc_handle *ipch);
bool lwsciipc_ipc_can_write(struct lwsciipc_ipc_handle *ipch);
LwSciError lwsciipc_ipc_set_qnx_pulse_param(
    struct lwsciipc_ipc_handle *ipch, int32_t coid,
    int16_t priority, int16_t code, void *value);
LwSciError lwsciipc_ipc_configure_pulse_param(
    LwSciNativeEvent *thisNativeEvent,
    int32_t coid, int16_t priority, int16_t code, void *value);
void lwsciipc_ipc_unconfigure_pulse_param(
    LwSciNativeEvent *thisNativeEvent);

LwSciError lwsciipc_ipc_endpoint_get_auth_token(
    struct lwsciipc_ipc_handle *ipch, LwSciIpcEndpointAuthToken *authToken);
LwSciError lwsciipc_ipc_endpoint_get_vuid(
    struct lwsciipc_ipc_handle *ipch, LwSciIpcEndpointVuid *vuid);

#endif /* INCLUDED_LWSCIIPC_IPC_H */

