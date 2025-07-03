/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_IVC_H
#define INCLUDED_LWSCIIPC_IVC_H

#ifdef __QNX__
#include <lwscievent_qnx.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif
#endif /* __QNX__ */

#ifdef LINUX
#include <lwscievent_linux.h>
#endif /* LINUX */

/* 168 Bytes */
struct lwsciipc_ivc_handle
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
        bool     rx_first;
        uint64_t area_pa; <--
        uint16_t irq; <--
    };
    */
    struct LwSciIpcConfigEntry *entry;
    struct lwsciipc_ivc_info ivc_info;

    uint32_t *noti_va; /* virtual address of TRAP/MSI notification */
    uint64_t noti_vsz; /* size of of TRAP/MSI IPA virtual address region */

    uintptr_t shm;
    int32_t fd;    /* devv-lwivc devctl fd in QNX */
    int32_t pid;

    uint32_t prev_conn;    /* previous connection status */

    int32_t mutexfd; /* io-lwsciipc mutex fd */

    LwSciEventService *eventService;
#ifdef LINUX
    LwSciNativeEvent nativeEvent;
#endif

#ifdef __QNX__
    LwSciQnxNativeEvent qnxEvent;
    int32_t chid; /* channel id for pulse event */
    int32_t coid; /* connection id for pulse event */
    int32_t iid;  /* interrupt id */
#endif /* __QNX__ */

    uint32_t is_open; /* channel is initialized ? (LwBoolTrue/LwBoolFalse) */
} __attribute__((aligned(8)));


/**
 * lwsciipc_ivc.c : inter-VM functions
 *
 */
LwSciError lwsciipc_ivc_open_endpoint(struct lwsciipc_ivc_handle **ivcp,
    struct LwSciIpcConfigEntry *entry);
void lwsciipc_ivc_bind_eventservice(struct lwsciipc_ivc_handle *ivch,
    LwSciEventService *eventService);
LwSciError lwsciipc_ivc_open_endpoint_with_eventservice(
    struct lwsciipc_ivc_handle **ivcp,
    struct LwSciIpcConfigEntry *entry,
    LwSciEventService *eventService);
void lwsciipc_ivc_close_endpoint(struct lwsciipc_ivc_handle *ivch);
void lwsciipc_ivc_reset_endpoint(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_read(struct lwsciipc_ivc_handle *ivch, void *buf,
    uint32_t size, uint32_t *bytes);
const volatile void *lwsciipc_ivc_read_get_next_frame(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_read_advance(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_write(struct lwsciipc_ivc_handle *ivch,
    const void *buf, uint32_t size, uint32_t *bytes);
volatile void *lwsciipc_ivc_write_get_next_frame(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_write_advance(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_read_peek(struct lwsciipc_ivc_handle *ivch, void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes);
LwSciError lwsciipc_ivc_write_poke(struct lwsciipc_ivc_handle *ivch, const void *buf,
    uint32_t offset, uint32_t count, uint32_t *bytes);

LwSciError lwsciipc_ivc_get_endpoint_info(const struct lwsciipc_ivc_handle *ivch,
    LwSciIpcEndpointInfo *info);
LwSciError lwsciipc_ivc_get_endpoint_info_internal(const struct lwsciipc_ivc_handle *ivch,
    LwSciIpcEndpointInfoInternal *info);
LwSciError lwsciipc_ivc_get_eventnotifier(
    struct lwsciipc_ivc_handle *ivch,
    LwSciEventNotifier **eventNotifier);
#ifdef LINUX
LwSciError lwsciipc_ivc_get_eventfd(const struct lwsciipc_ivc_handle *ivch,
    int32_t *fd);
#endif /* LINUX */
LwSciError lwsciipc_ivc_check_read(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_check_write(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_get_event(struct lwsciipc_ivc_handle *ivch,
    uint32_t *events, struct lwsciipc_internal_handle *inth);
bool lwsciipc_ivc_can_read(struct lwsciipc_ivc_handle *ivch);
bool lwsciipc_ivc_can_write(struct lwsciipc_ivc_handle *ivch);
LwSciError lwsciipc_ivc_set_qnx_pulse_param(
    struct lwsciipc_ivc_handle *ivch, int32_t coid,
    int16_t priority, int16_t code, void *value);
LwSciError lwsciipc_ivc_configure_pulse_param(
    LwSciNativeEvent *thisNativeEvent,
    int32_t coid, int16_t priority, int16_t code, void *value);
void lwsciipc_ivc_unconfigure_pulse_param(
    LwSciNativeEvent *thisNativeEvent);
LwSciError lwsciipc_ivc_unmask_interrupt(LwSciNativeEvent *thisNativeEvent);

LwSciError lwsciipc_ivc_endpoint_get_auth_token(
    struct lwsciipc_ivc_handle *ivch, LwSciIpcEndpointAuthToken *authToken);
LwSciError lwsciipc_ivc_endpoint_get_vuid(
    struct lwsciipc_ivc_handle *ivch, LwSciIpcEndpointVuid *vuid);

#endif /* INCLUDED_LWSCIIPC_IVC_H */

