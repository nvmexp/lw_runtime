/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_C2C_H
#define INCLUDED_LWSCIIPC_C2C_H

#if defined(LINUX) || (LW_IS_SAFETY == 0)
#include <lwscic2c_pcie_ipc.h>
#include <lwscic2c_pcie_stream.h>
#ifdef LINUX
#include <lwscievent_linux.h>
#else /* QNX */
#include <lwscievent_qnx.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif
#endif

#define C2CCTRL_LIB_NAME "liblwscic2c_pcie_ipc.so"
#define C2CCOPY_LIB_NAME "liblwscic2c_pcie_stream.so"
/*
TBD
#define C2CNPMCTRL_LIB_NAME "liblwscic2c_npm_ipc.so"
#define C2CNPMBULK_LIB_NAME "liblwscic2c_npm_stream.so"
*/

struct lwsciipc_c2c_handle
{
    LwSciC2cPcieHandle h;
    struct LwSciIpcConfigEntry *entry;
    int32_t mutexfd; /* io-lwsciipc mutex fd */
} __attribute__((aligned(8)));

typedef struct _lwsciipc_c2c_io_funcs {
    /* C2C cpu(control) APIs */
    int32_t (*open_endpoint)(LwSciC2cPcieEndpointConfig *info,
        LwSciC2cPcieHandle *handle);
    int32_t (*close_endpoint)(LwSciC2cPcieHandle handle);
    int32_t (*reset_endpoint)(LwSciC2cPcieHandle handle);
    int32_t (*read)(LwSciC2cPcieHandle handle, void *buf, size_t size);
    int32_t (*write)(LwSciC2cPcieHandle handle, const void *buf, size_t size);
    int32_t (*get_endpoint_info)(LwSciC2cPcieHandle handle,
        LwSciC2cPcieEndpointInfo *info );
    int32_t (*get_event)(LwSciC2cPcieHandle handle, uint32_t *events);
    int32_t (*set_qnx_pulse_param)(LwSciC2cPcieHandle handle,
        int32_t coid, int16_t priority, int16_t code, void *value);
    int32_t (*get_eventfd)(LwSciC2cPcieHandle handle, int32_t *fd);
    int32_t (*open_endpoint_with_event_service)(
        LwSciC2cPcieEndpointConfig *info, LwSciC2cPcieHandle *handle);
    int32_t (*get_eventnotifier)(LwSciC2cPcieHandle handle,
        LwSciEventNotifier **event_notifier);
    int32_t (*get_endpoint_info_internal)(LwSciC2cPcieHandle c2ch,
        LwSciC2cPcieEndpointInernalInfo *c2c_info);
    int32_t (*bind_eventservice)(
        LwSciC2cPcieHandle handle, LwSciEventService *eventService);
    int32_t (*get_cookie)(
       LwSciC2cPcieHandle handle, LwSciIpcC2cCookie *cookie);
    int32_t (*set_cookie)(
       LwSciC2cPcieHandle handle, LwSciIpcC2cCookie cookie);

    /* C2C stream(copy) APIs */
    int32_t (*get_c2ccopy_funcset)(LwSciC2cPcieCopyFuncs *fn);
    int32_t (*validate_c2ccopy_funcset)(const LwSciC2cPcieCopyFuncs *fn);
} lwsciipc_c2c_io_funcs_t;

/* liblwscic2c_pcie_ipc API symbol name */
#define C2C_OPEN_ENDPOINT              "LwSciC2cPcieOpenEndpoint"
#define C2C_CLOSE_ENDPOINT             "LwSciC2cPcieCloseEndpoint"
#define C2C_RESET_ENDPOINT             "LwSciC2cPcieResetEndpoint"
#define C2C_READ                       "LwSciC2cPcieRead"
#define C2C_WRITE                      "LwSciC2cPcieWrite"
#define C2C_GET_ENDPOINT_INFO          "LwSciC2cPcieGetEndpointInfo"
#define C2C_GET_EVENT                  "LwSciC2cPcieGetEvent"
#define C2C_SET_QNX_PULSE_PARAM        "LwSciC2cPcieSetQnxPulseParam"
#define C2C_GET_LINUX_EVENT_FD         "LwSciC2cPcieGetLinuxEventFd"
#define C2C_OPEN_WITH_EVENT_SERVICE \
    "LwSciC2cPcieOpenEndpointWithEventService"
#define C2C_GET_EVENT_NOTIFIER         "LwSciC2cPcieGetEventNotifier"
#define C2C_GET_ENDPOINT_INFO_INTERNAL "LwSciC2cPcieGetEndpointInfoInternal"
#define C2C_BIND_EVENTSERVICE          "LwSciC2cPcieBindEventService"
#define C2C_GET_COOKIE                 "LwSciC2cPcieGetCookie"
#define C2C_SET_COOKIE                 "LwSciC2cPcieSetCookie"

/* TODO: C2C team can change API name */
#define C2C_GET_C2CCOPY_APISET         "LwSciC2cPcieGetCopyFuncSet"
#define C2C_VALIDATE_C2CCOPY_APISET    "LwSciC2cPcieValidateCopyFuncSet"

/**
 * lwsciipc_c2c.c : inter-chip functions
 *                  CPU transfer (control)
 */
LwSciError lwsciipc_c2c_open_endpoint(struct lwsciipc_c2c_handle **c2cp,
    struct LwSciIpcConfigEntry *entry);
LwSciError lwsciipc_c2c_open_endpoint_with_event_service(
    struct lwsciipc_c2c_handle **c2cp,
    struct LwSciIpcConfigEntry *entry,
    LwSciEventService *eventService);
LwSciError lwsciipc_c2c_bind_eventservice(
    struct lwsciipc_c2c_handle *c2ch,
    LwSciEventService *eventService);
void lwsciipc_c2c_close_endpoint(struct lwsciipc_c2c_handle *c2ch);
void lwsciipc_c2c_reset_endpoint(struct lwsciipc_c2c_handle *c2ch);
LwSciError lwsciipc_c2c_read(struct lwsciipc_c2c_handle *c2ch, void *buf,
    uint32_t size, uint32_t *bytes);
LwSciError lwsciipc_c2c_write(struct lwsciipc_c2c_handle *c2ch,
    const void *buf, uint32_t size, uint32_t *bytes);

LwSciError lwsciipc_c2c_get_endpoint_info(
    const struct lwsciipc_c2c_handle *c2ch,
    LwSciC2cPcieEndpointInfo *info);
LwSciError lwsciipc_c2c_get_endpoint_info_internal(
    const struct lwsciipc_c2c_handle *c2ch,
    LwSciC2cPcieEndpointInernalInfo *c2c_info);
LwSciError lwsciipc_c2c_get_eventnotifier(
    struct lwsciipc_c2c_handle *c2ch,
    LwSciEventNotifier **eventNotifier);

LwSciError lwsciipc_c2c_get_event(struct lwsciipc_c2c_handle *c2ch,
    uint32_t *events, struct lwsciipc_internal_handle *inth);
LwSciError lwsciipc_c2c_set_qnx_pulse_param(
    struct lwsciipc_c2c_handle *c2ch,
    int32_t coid, int16_t priority, int16_t code, void *value);
LwSciError lwsciipc_c2c_get_eventfd(const struct lwsciipc_c2c_handle *c2ch,
    int32_t *fd);
LwSciError lwsciipc_c2c_get_cookie(const struct lwsciipc_c2c_handle *c2ch,
    LwSciIpcC2cCookie *cookie);
LwSciError lwsciipc_c2c_set_cookie(const struct lwsciipc_c2c_handle *c2ch,
    LwSciIpcC2cCookie cookie);

void lwsciipc_c2c_close_library(void);

/**
 * lwsciipc_c2c.c : inter-chip functions
 *                  Bulk tranfer (stream/c2c copy)
 */
LwSciError lwsciipc_c2c_get_c2ccopy_funcset(uint32_t backend, void *fn);
LwSciError lwsciipc_c2c_validate_c2ccopy_funcset(uint32_t backend,
    const void *fn);


#endif /* LINUX || (LW_IS_SAFETY == 0) */
#endif /* INCLUDED_LWSCIIPC_C2C_H */

