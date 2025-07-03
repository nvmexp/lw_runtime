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
#include <dlfcn.h>

#include <lwos_static_analysis.h>

#include "lwsciipc_common.h"
#include "lwsciipc_os_error.h"
#include "lwsciipc_c2c.h"

static void *s_hC2c = NULL;
static void *s_hC2cCopy = NULL;
static LwBoolVar s_C2Csymload = LwBoolFalse;
static lwsciipc_c2c_io_funcs_t s_c2c_funcs;

static LwSciError lwsciipc_c2c_load_symbols_1of2(void);
static LwSciError lwsciipc_c2c_load_symbols_2of2(void);

void lwsciipc_c2c_close_library(void)
{
    if (s_hC2c != NULL) {
        dlclose(s_hC2c);
        s_hC2c = NULL;
    }
    if (s_hC2cCopy != NULL) {
        dlclose(s_hC2cCopy);
        s_hC2cCopy = NULL;
    }
    s_C2Csymload = LwBoolFalse;
}

static LwSciError lwsciipc_c2c_load_symbols_1of2(void)
{
    LwSciError ret = LwSciError_NoSuchEntry;

    s_c2c_funcs.open_endpoint = dlsym(s_hC2c, C2C_OPEN_ENDPOINT);
    if (s_c2c_funcs.open_endpoint == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: "
            "LwSciC2cPcieOpenEndpoint");
        goto error;
    }
    s_c2c_funcs.close_endpoint = dlsym(s_hC2c, C2C_CLOSE_ENDPOINT);
    if (s_c2c_funcs.close_endpoint == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: "
            "LwSciC2cPcieCloseEndpoint");
        goto error;
    }
    s_c2c_funcs.reset_endpoint = dlsym(s_hC2c, C2C_RESET_ENDPOINT);
    if (s_c2c_funcs.reset_endpoint == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: "
            "LwSciC2cPcieResetEndpoint");
        goto error;
    }
    s_c2c_funcs.read = dlsym(s_hC2c, C2C_READ);
    if (s_c2c_funcs.read == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: LwSciC2cPcieRead");
        goto error;
    }
    s_c2c_funcs.write = dlsym(s_hC2c, C2C_WRITE);
    if (s_c2c_funcs.write == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: LwSciC2cPcieWrite");
        goto error;
    }

    ret = LwSciError_Success;

error:
    return ret;
}

static LwSciError lwsciipc_c2c_load_symbols_2of2(void)
{
    LwSciError ret = LwSciError_NoSuchEntry;

    s_c2c_funcs.get_endpoint_info = dlsym(s_hC2c, C2C_GET_ENDPOINT_INFO);
    if (s_c2c_funcs.get_endpoint_info == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: LwSciC2cPcieGetEndpointInfo");
        goto error;
    }
    s_c2c_funcs.get_event = dlsym(s_hC2c, C2C_GET_EVENT);
    if (s_c2c_funcs.get_event == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: LwSciC2cPcieGetEvent");
        goto error;
    }
#ifndef LINUX
    s_c2c_funcs.set_qnx_pulse_param =
        dlsym(s_hC2c, C2C_SET_QNX_PULSE_PARAM);
    if (s_c2c_funcs.set_qnx_pulse_param == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: LwSciC2cPcieSetQnxPulseParam");
        goto error;
    }
#endif
    s_c2c_funcs.get_eventfd = dlsym(s_hC2c, C2C_GET_LINUX_EVENT_FD);
    if (s_c2c_funcs.get_eventfd == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: LwSciC2cPcieGetLinuxEventFd");
        goto error;
    }

    s_c2c_funcs.open_endpoint_with_event_service =
        dlsym(s_hC2c, C2C_OPEN_WITH_EVENT_SERVICE);
    if (s_c2c_funcs.open_endpoint_with_event_service == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: "
            "LwSciC2cPcieOpenEndpointWithEventService");
        goto error;
    }

    s_c2c_funcs.get_eventnotifier =
        dlsym(s_hC2c, C2C_GET_EVENT_NOTIFIER);
    if (s_c2c_funcs.get_eventnotifier == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: LwSciC2cPcieGetEventNotifier");
        goto error;
    }

    s_c2c_funcs.get_endpoint_info_internal =
        dlsym(s_hC2c, C2C_GET_ENDPOINT_INFO_INTERNAL);
    if (s_c2c_funcs.get_endpoint_info_internal == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: "
            "LwSciC2cPcieGetEndpointInfoInternal");
        goto error;
    }

    s_c2c_funcs.bind_eventservice =
        dlsym(s_hC2c, C2C_BIND_EVENTSERVICE);
    if (s_c2c_funcs.bind_eventservice == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: "
            "LwSciC2cPcieBindEventService");
        goto error;
    }

    s_c2c_funcs.get_cookie =
        dlsym(s_hC2c, C2C_GET_COOKIE);
    if (s_c2c_funcs.get_cookie == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: "
            "LwSciC2cPcieGetCookie");
        goto error;
    }

    s_c2c_funcs.set_cookie =
        dlsym(s_hC2c, C2C_SET_COOKIE);
    if (s_c2c_funcs.set_cookie == NULL) {
        LWSCIIPC_ERR_STR(
            "error: could not find symbol: "
            "LwSciC2cPcieSetCookie");
        goto error;
    }

    ret = LwSciError_Success;

error:
    return ret;
}

static LwSciError lwsciipc_c2c_load_c2ccopy_symbols(void)
{
    LwSciError ret = LwSciError_NoSuchEntry;

    s_c2c_funcs.get_c2ccopy_funcset = dlsym(s_hC2cCopy,
        C2C_GET_C2CCOPY_APISET);
    if (s_c2c_funcs.get_c2ccopy_funcset == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: "
            "LwSciC2cPcieGetCopyFuncSet");
        goto error;
    }
    s_c2c_funcs.validate_c2ccopy_funcset = dlsym(s_hC2cCopy,
        C2C_VALIDATE_C2CCOPY_APISET);
    if (s_c2c_funcs.validate_c2ccopy_funcset == NULL) {
        LWSCIIPC_ERR_STR("error: could not find symbol: "
            "LwSciC2cPcieValidateCopyFuncSet");
        goto error;
    }

    ret = LwSciError_Success;

error:
    return ret;
}

static LwSciError lwsciipc_c2c_open_library(void)
{
    LwSciError ret;

    if ((s_hC2c != NULL) && (s_hC2cCopy != NULL) &&
        (s_C2Csymload == LwBoolTrue)) {
        ret = LwSciError_Success;
        goto done;
    }

    /* vm server and update server doesn't need C2C function.
     * hence dlopen is used instead of dynamic linking.
     * but if boot latency with C2C libraries are insignificant, we can use
     * dynamic linking.
     */
    s_hC2c = dlopen(C2CCTRL_LIB_NAME, RTLD_GLOBAL | RTLD_NOW);
    if (s_hC2c == NULL) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_open_library: "
            "liblwscic2c_pcie_ipc.so is not available");
        ret = LwSciError_NoSuchEntry;
        goto error;
    }
    else {
        LWSCIIPC_DBG_STR("lwsciipc_c2c_open_library: "
            "liblwscic2c_pcie_ipc.so is opened successfully");
    }

    s_hC2cCopy = dlopen(C2CCOPY_LIB_NAME, RTLD_GLOBAL | RTLD_NOW);
    if (s_hC2c == NULL) {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_open_library: "
            "liblwscic2c_pcie_stream.so is not available");
        ret = LwSciError_NoSuchEntry;
        goto error;
    }
    else {
        LWSCIIPC_DBG_STR("lwsciipc_c2c_open_library: "
            "liblwscic2c_pcie_stream.so is opened successfully");
    }

    ret = lwsciipc_c2c_load_symbols_1of2();
    if (ret != LwSciError_Success) {
        goto error;
    }
    ret = lwsciipc_c2c_load_symbols_2of2();
    if (ret != LwSciError_Success) {
        goto error;
    }
    ret = lwsciipc_c2c_load_c2ccopy_symbols();
    if (ret != LwSciError_Success) {
        goto error;
    }

    s_C2Csymload = LwBoolTrue;
    LWSCIIPC_DBG_STR(
        "lwsciipc_c2c_open_library: all symbols are loaded successfully");

    ret = LwSciError_Success;

error:
    if (ret != LwSciError_Success) {
        lwsciipc_c2c_close_library();
    }

done:
    return ret;
}

LwSciError lwsciipc_c2c_get_endpoint_info(
    const struct lwsciipc_c2c_handle *c2ch,
    LwSciC2cPcieEndpointInfo *info)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.get_endpoint_info(c2ch->h, info);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_get_endpoint_info", err);
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_get_endpoint_info: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

LwSciError lwsciipc_c2c_get_cookie(
    const struct lwsciipc_c2c_handle *c2ch,
    LwSciIpcC2cCookie *cookie)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.get_cookie(c2ch->h, cookie);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_get_cookie", err);
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_get_cookie: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

LwSciError lwsciipc_c2c_set_cookie(
    const struct lwsciipc_c2c_handle *c2ch,
    LwSciIpcC2cCookie cookie)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.set_cookie(c2ch->h, cookie);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_set_cookie", err);
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_set_cookie: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

LwSciError lwsciipc_c2c_read(struct lwsciipc_c2c_handle *c2ch, void *buf,
    uint32_t size, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.read(c2ch->h, buf, size);
        if (err == 0) {
            *bytes = size;
        }
        else {
            *bytes = 0;
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_read: C2C backend is not supported");
        err = ENOTSUP;
        *bytes = 0;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

LwSciError lwsciipc_c2c_write(struct lwsciipc_c2c_handle *c2ch,
    const void *buf, uint32_t size, uint32_t *bytes)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.write(c2ch->h, buf, size);
        if (err == 0) {
            *bytes = size;
        }
        else {
            *bytes = 0;
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_write: C2C backend is not supported");
        err = ENOTSUP;
        *bytes = 0;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

LwSciError lwsciipc_c2c_open_endpoint(struct lwsciipc_c2c_handle **c2cp,
    struct LwSciIpcConfigEntry *entry)
{
    struct lwsciipc_c2c_handle *c2ch = NULL;
    LwSciC2cPcieEndpointConfig info = {0};
    int32_t err;
    LwSciError ret;

    ret = lwsciipc_c2c_open_library();
    if (ret != LwSciError_Success) {
        goto fail;
    }

    if (s_C2Csymload == LwBoolTrue) {
        c2ch = (struct lwsciipc_c2c_handle *)calloc(1,
            sizeof(struct lwsciipc_c2c_handle));
        if (c2ch == NULL) {
            LWSCIIPC_ERR_STR("error: lwsciipc_c2c_open_endpoint: calloc");
            ret = LwSciError_InsufficientMemory;
            goto fail;
        }

        /* reserve endpoint mutex and authenticate client */
        ret = lwsciipc_os_get_endpoint_mutex(entry, &c2ch->mutexfd);
        if (ret != LwSciError_Success) {
            goto fail;
        }

        c2ch->entry = entry;

        info.epName = entry->epName;
        info.devId = entry->id;

        err = s_c2c_funcs.open_endpoint(&info, &c2ch->h);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_open_endpoint", err);
        }
        else {
            *c2cp = c2ch;
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_open_endpoint: C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

fail:
    if ((ret != LwSciError_Success) && (c2ch != NULL)) {
        free(c2ch);
    }

    return ret;
}

void lwsciipc_c2c_close_endpoint(struct lwsciipc_c2c_handle *c2ch)
{
    if (s_C2Csymload == LwBoolTrue) {
        (void)s_c2c_funcs.close_endpoint(c2ch->h);
        /* release endpoint mutex */
        lwsciipc_os_put_endpoint_mutex(&c2ch->mutexfd);
        /* release internal handle */
        free(c2ch);
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_close_endpoint: C2C backend is not supported");
    }
}

void lwsciipc_c2c_reset_endpoint(struct lwsciipc_c2c_handle *c2ch)
{
    if (s_C2Csymload == LwBoolTrue) {
        (void)s_c2c_funcs.reset_endpoint(c2ch->h);
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_reset_endpoint: C2C backend is not supported");
    }
}

LwSciError lwsciipc_c2c_get_event(struct lwsciipc_c2c_handle *c2ch,
    uint32_t *events, struct lwsciipc_internal_handle *inth)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        (void)lwsciipc_os_mutex_lock(&inth->wrMutex); /* wr pos/refcnt */
        (void)lwsciipc_os_mutex_lock(&inth->rdMutex); /* rd pos/refcnt */
        err = s_c2c_funcs.get_event(c2ch->h, events);
        (void)lwsciipc_os_mutex_unlock(&inth->rdMutex);
        (void)lwsciipc_os_mutex_unlock(&inth->wrMutex);
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_get_event: C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

#ifdef __QNX__
LwSciError lwsciipc_c2c_set_qnx_pulse_param(struct lwsciipc_c2c_handle *c2ch,
    int32_t coid, int16_t priority, int16_t code, void *value)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.set_qnx_pulse_param(c2ch->h,
                coid, priority, code, value);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_set_qnx_pulse_param", err);
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_set_qnx_pulse_param: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}
#endif /* __QNX__ */

#if defined(LINUX)
LwSciError lwsciipc_c2c_get_eventfd(const struct lwsciipc_c2c_handle *c2ch,
    int32_t *fd)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.get_eventfd(c2ch->h, fd);
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_get_eventfd: C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}
#endif /* LINUX */

LwSciError lwsciipc_c2c_open_endpoint_with_event_service(
    struct lwsciipc_c2c_handle **c2cp,
    struct LwSciIpcConfigEntry *entry,
    LwSciEventService *event_service)
{
    struct lwsciipc_c2c_handle *c2ch = NULL;
    LwSciC2cPcieEndpointConfig info = {0};
    int32_t err;
    LwSciError ret;

    ret = lwsciipc_c2c_open_library();
    if (ret != LwSciError_Success) {
        goto fail;
    }

    if (s_C2Csymload == LwBoolTrue) {
        c2ch = (struct lwsciipc_c2c_handle *)calloc(1,
            sizeof(struct lwsciipc_c2c_handle));
        if (c2ch == NULL) {
            LWSCIIPC_ERR_STR("error: lwsciipc_c2c_open_endpoint: calloc");
            ret = LwSciError_InsufficientMemory;
            goto fail;
        }

        /* reserve endpoint mutex and authenticate client */
        ret = lwsciipc_os_get_endpoint_mutex(entry, &c2ch->mutexfd);
        if (ret != LwSciError_Success) {
            goto fail;
        }

        c2ch->entry = entry;

        info.epName = entry->epName;
        info.devId = entry->id;
        info.eventService = event_service;

        err = s_c2c_funcs.open_endpoint_with_event_service(&info, &c2ch->h);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_open_endpoint", err);
        }
        else {
            *c2cp = c2ch;
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_open_endpoint_with_event_service: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

fail:
    if ((ret != LwSciError_Success) && (c2ch != NULL)) {
        free(c2ch);
    }

    return ret;
}

LwSciError lwsciipc_c2c_bind_eventservice(
    struct lwsciipc_c2c_handle *c2ch,
    LwSciEventService *eventService)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.bind_eventservice(c2ch->h, eventService);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_bind_eventservice", err);
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_bind_eventservice: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

LwSciError lwsciipc_c2c_get_eventnotifier(struct lwsciipc_c2c_handle *c2ch,
    LwSciEventNotifier **eventNotifier)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.get_eventnotifier(c2ch->h, eventNotifier);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT("error: lwsciipc_c2c_get_eventnotifier", err);
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_get_eventnotifier: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

LwSciError lwsciipc_c2c_get_endpoint_info_internal(
    const struct lwsciipc_c2c_handle *c2ch,
    LwSciC2cPcieEndpointInernalInfo *c2c_info)
{
    int32_t err;
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        err = s_c2c_funcs.get_endpoint_info_internal(c2ch->h, c2c_info);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT(
                "error: lwsciipc_c2c_get_endpoint_info_internal", err);
        }
    }
    else {
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_get_endpoint_info_internal: "
            "C2C backend is not supported");
        err = ENOTSUP;
    }

    ret = ErrnoToLwSciErr(err);

    return ret;
}

#if (LW_IS_SAFETY == 0)
LwSciError lwsciipc_c2c_get_c2ccopy_funcset(uint32_t backend, void *fn)
{
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        int32_t err;
                err = s_c2c_funcs.get_c2ccopy_funcset((LwSciC2cPcieCopyFuncs*)fn);
                if (err != EOK) {
                    LWSCIIPC_ERR_STRINT(
                    "error: lwsciipc_c2c_get_c2ccopy_funcset", err);
                }
                ret = ErrnoToLwSciErr(err);
    }
    else {
        ret = LwSciError_NotSupported;
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_get_c2ccopy_funcset: "
            "C2C backend is not supported");
    }

    return ret;
}

LwSciError lwsciipc_c2c_validate_c2ccopy_funcset(uint32_t backend,
    const void *fn)
{
    LwSciError ret;

    if (s_C2Csymload == LwBoolTrue) {
        int32_t err;
        err = s_c2c_funcs.validate_c2ccopy_funcset((const LwSciC2cPcieCopyFuncs*)fn);
        if (err != EOK) {
            LWSCIIPC_ERR_STRINT(
                 "error: lwsciipc_c2c_validate_c2ccopy_funcset", err);
        }
        ret = ErrnoToLwSciErr(err);
    }
    else {
        ret = LwSciError_NotSupported;
        LWSCIIPC_ERR_STR(
            "error: lwsciipc_c2c_validate_c2ccopy_funcset: "
            "C2C backend is not supported");
    }

    return ret;
}
#endif
