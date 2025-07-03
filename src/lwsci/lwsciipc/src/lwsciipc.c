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
#include <stdlib.h>
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
#include <ctype.h>
#include <search.h>

#include <lwos_static_analysis.h>
#include <lwsciipc_internal.h>
#if defined(LINUX) || (LW_IS_SAFETY == 0)
#include <lwscic2c_pcie_ipc.h>
#endif /* LINUX || (LW_IS_SAFETY == 0) */
#ifdef __QNX__
#include <lwsciipc_init.h>
#endif /* __QNX__ */

#include "lwsciipc_common.h"
#include "lwsciipc_static_analysis.h"
#include "lwsciipc_os_error.h"
#include "lwsciipc_ivc.h"
#include "lwsciipc_ipc.h"
#include <lw_spelwlation_barrier.h>
#if defined(LINUX) || (LW_IS_SAFETY == 0)
#include "lwsciipc_c2c.h"
#endif /* LINUX || (LW_IS_SAFETY == 0) */
#include "lwsciipc_log.h"


/* maximum usable endpoint counts per process */
static struct lwsciipc_internal_handle *
    s_intHandle[LWSCIIPC_MAX_ENDPOINT];

/* library init done flag which is set in LwSciIpcInit() */
//LwBoolVar doesn't work
static uint32_t s_initDone = LwBoolFalse;

/* global mutex for APIs which don't use endpoint handle
 *
 * s_mutex is used in:
 * LwSciIpcInit()
 * LwSciIpcDeinit()
 * LwSciIpcOpenEndpoint()
 * LwSciIpcOpenEndpointWithEventService()
 * LwSciIpcOpenCfgBlob()
 * LwSciIpcCloseCfgBlob()
 * LwSciIpcEndpointValidateAuthTokenQnx()
 * LwSciIpcEndpointMapVuid()
 */
static pthread_mutex_t s_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint32_t lwsciipc_is_valid_handle(LwSciIpcEndpoint appHandle,
    struct lwsciipc_internal_handle **intHandle);
static void lwsciipc_close_single_endpoint(
    struct lwsciipc_internal_handle *lwsciipch);
static void lwsciipc_close_all_endpoints(void);

/*
 * colwert appHandle to internal handle
 */
static struct lwsciipc_internal_handle *
    lwsciipc_get_int_handle(LwSciIpcEndpoint appHandle)
{
    struct lwsciipc_internal_handle *lwsciipch;
    uint64_t index;

    if ((appHandle == 0UL) || (appHandle >= (uint64_t)LWSCIIPC_MAX_ENDPOINT)) {
        lwsciipch = NULL;
    }
    else {
        index = lw_array_index_no_spelwlate(appHandle, LWSCIIPC_MAX_ENDPOINT);
        lwsciipch = s_intHandle[index];

        /* validate handle */
        if ((lwsciipch != NULL) && (lwsciipch->index != appHandle)) {
            LWSCIIPC_ERR_STR2ULONG("error: handle mismatch",
                lwsciipch->index, appHandle);
            lwsciipch = NULL;
        }
    }

    return lwsciipch;
}

/**
 * Allocate internal handle.
 * It allocates internal handles and checks if the number of endpoint allocated
 * exceeds maximum number.
 * It finds empty internal handle entry and fill it with provided internal
 * handle then return app handle which consists of * index/pid/tid.
 *
 * @param[in]  intHandle  Internal endpoint handle
 * @param[out] appHandle  Index to internal endpoint handle allocated
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_InsufficientMemory Indicates memory allocation failed for the
 *                                   operation.
 */
static LwSciError lwsciipc_alloc_int_handle(
    struct lwsciipc_internal_handle *intHandle,
    LwSciIpcEndpoint *appHandle)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_alloc_int_handle: "
    LwSciError ret = LwSciError_Success;
    uint32_t idx;

    /* 0 handle is not used */
    for (idx = 1U; idx < LWSCIIPC_MAX_ENDPOINT; idx++) {
        if (s_intHandle[idx] == NULL) {
            intHandle->index = idx;
            s_intHandle[idx] = intHandle;
            *appHandle = idx;
#ifdef __QNX__
            LWSCIIPC_DBG_STRINT(LIB_FUNC "index", idx);
            LWSCIIPC_DBG_STR2ULONG(LIB_FUNC "intH, appH",
                (uint64_t)intHandle, (uint64_t)*appHandle);
#endif /* __QNX__ */
            goto done;
        }
    }

    ret = LwSciError_InsufficientMemory;

done:
#if defined(LWSCIIPC_DEBUG)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT("lwsciipc_alloc_int_handle:error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG */

    return ret;
}

/**
 * Test if application endpoint handle is valid or not.
 * If application handle is valid, it's colwerted to internal endpoint handle.
 *
 * @param[in]  appHandle  Endpoint handle from application
 * @param[out] intHandle  Pointer to the internal endpoint handle
 *
 * @return ::
 * - ::LwBoolTrue    Indicates appHandle is valid
 * - ::LwBoolFalse   Indicates appHandle is invalid
 */
static uint32_t lwsciipc_is_valid_handle(LwSciIpcEndpoint appHandle,
    struct lwsciipc_internal_handle **intHandle)
{
    struct lwsciipc_internal_handle *lwsciipch;
    uint32_t ret = LwBoolFalse;

    lwsciipch = lwsciipc_get_int_handle(appHandle);

    if ((lwsciipch == NULL) || (lwsciipch->magic != LW_SCI_IPC_MAGIC)) {
        ret = LwBoolFalse;
    }
    else {
        switch (lwsciipch->type) {
            case LWSCIIPC_BACKEND_IVC :
                if (lwsciipch->ivch == NULL) {
                    ret = LwBoolFalse;
                }
                else {
                    ret = LwBoolTrue;
                }
                break;
            case LWSCIIPC_BACKEND_ITC :
            case LWSCIIPC_BACKEND_IPC :
                if (lwsciipch->ipch == NULL) {
                    ret = LwBoolFalse;
                }
                else {
                    ret = LwBoolTrue;
                }
                break;
#if (LW_IS_SAFETY == 0)
            case LWSCIIPC_BACKEND_C2C :
                if (lwsciipch->c2ch == NULL) {
                    ret = LwBoolFalse;
                }
                else {
                    ret = LwBoolTrue;
                }
                break;
#endif /* (LW_IS_SAFETY == 0) */
            default :
                /* not supoprted backend */
                ret = LwBoolFalse;
                break;
        }
    }

    if (ret == LwBoolTrue) {
        *intHandle = lwsciipch;
    }

    return ret;
}

#ifdef __QNX__
/* Check if ipch is opened with LwSciEventService */
static uint32_t lwsciipc_has_eventservice(
    const struct lwsciipc_internal_handle *ipch)
{
    LwSciEventService *eventService;
    bool ret = LwBoolFalse;

    switch (ipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            eventService = ipch->ivch->eventService;
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            eventService = ipch->ipch->eventService;
            break;
        default : /* dont' care about other backend type */
            eventService = NULL;
            break;
    }

    if (NULL != eventService) {
        ret = LwBoolTrue;
    }
    return ret;
}
#endif /* __QNX__ */

LwSciError LwSciIpcInit(void)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcInit :"
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    err = lwsciipc_os_mutex_lock(&s_mutex);
    report_mutex_errto(err, "mutex_lock", fail2);

    if (s_initDone != LwBoolTrue) {
        ret = lwsciipc_os_open_config();
        if (ret != LwSciError_Success) {
            LWSCIIPC_ERR_STRINT("error: " LIB_FUNC
                "Failed to load ConfigBlob", (int32_t)ret);
            goto fail;
        }

        s_initDone = LwBoolTrue;
    }

fail:
    err = lwsciipc_os_mutex_unlock(&s_mutex);
    update_mutex_errto(err, "mutex_unlock", fail2);

fail2:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

void LwSciIpcDeinit(void)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcDeinit :"
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    err = lwsciipc_os_mutex_lock(&s_mutex);
    log_mutex_errto(err, "mutex_lock", fail);

    if (s_initDone == LwBoolTrue) {
        /* close all opened endpoints */
        lwsciipc_close_all_endpoints();

#if (LW_IS_SAFETY == 0)
        /* unload c2c library */
        lwsciipc_c2c_close_library();
#endif /* (LW_IS_SAFETY == 0) */

        lwsciipc_os_close_config();
        s_initDone = LwBoolFalse;
    }

    err = lwsciipc_os_mutex_unlock(&s_mutex);
    log_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STR(LIB_FUNC "exit");
}

/**
 * Do preliminary job common to all backend.
 * It allocates mutexes for general, read and write operation and calls an
 * internal function to serarch for endpoint with given name.
 *
 * @param[in]  endpoint   The name of the LwSciIpc endpoint to open.
 * @param[in]  handle     Endpoint handle
 * @param[out] lwsciipch  Internal endpoint handle
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success            Indicates a successful operation.
 * - ::LwSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::LwSciError_NotInitialized     Indicates LwSciIpc is uninitialized.
 * - ::LwSciError_NoSuchEntry        Indicates the @a endpoint was not found.
 */
static LwSciError LwSciIpcOpenEndpointPreCommon(const char *endpoint,
    LwSciIpcEndpoint *handle, struct lwsciipc_internal_handle **lwsciipch)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcOpenEndpointPreCommon: "
    struct lwsciipc_internal_handle *ipch = NULL;
    struct LwSciIpcConfigEntry *entry;
    LwSciError ret;
    int32_t err;

    if ((endpoint == NULL) || (handle == NULL) || (lwsciipch == NULL)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid parameter");
        ret = LwSciError_BadParameter;
        goto fail2;
    }

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail2;
    }

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    ipch = (struct lwsciipc_internal_handle *) calloc(1,
        sizeof(struct lwsciipc_internal_handle));
    if (ipch == NULL) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Failed to alloc");
        ret = LwSciError_InsufficientMemory;
        goto fail2;
    }

    err = lwsciipc_os_mutex_init(&ipch->genMutex, NULL);
    report_mutex_errto(err, "GEN mutex_init", fail2);
    err = lwsciipc_os_mutex_init(&ipch->wrMutex, NULL);
    report_mutex_errto(err, "WR mutex_init", de_gen_mtx);
    err = lwsciipc_os_mutex_init(&ipch->rdMutex, NULL);
    report_mutex_errto(err, "RD mutex_init", de_wr_mtx);

    err = lwsciipc_os_mutex_lock(&ipch->genMutex);
    report_mutex_errto(err, "GEN mutex_lock", de_all_mtx);
    err = lwsciipc_os_mutex_lock(&ipch->wrMutex);
    report_mutex_errto(err, "WR mutex_lock", un_gen_mtx);
    err = lwsciipc_os_mutex_lock(&ipch->rdMutex);
    report_mutex_errto(err, "RD mutex_lock", un_wr_mtx);

    ret = lwsciipc_os_get_config_entry(endpoint, &entry);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRINT("error: " LIB_FUNC
            "Failed to find endpoint", (int32_t)ret);
        goto fail;
    }

    ipch->type = entry->backend;
    ipch->entry = entry;
    *lwsciipch = ipch;

    ret = LwSciError_Success;
    goto done;

fail:
    /* clean up mutexes */
    err = lwsciipc_os_mutex_unlock(&ipch->rdMutex);
    log_mutex_err(err, "RD mutex_unlock");
un_wr_mtx:
    err = lwsciipc_os_mutex_unlock(&ipch->wrMutex);
    log_mutex_err(err, "WR mutex_unlock");
un_gen_mtx:
    err = lwsciipc_os_mutex_unlock(&ipch->genMutex);
    log_mutex_err(err, "GEN mutex_unlock");

de_all_mtx:
    err = lwsciipc_os_mutex_destroy(&ipch->rdMutex);
    log_mutex_err(err, "RD mutex_destroy");
de_wr_mtx:
    err = lwsciipc_os_mutex_destroy(&ipch->wrMutex);
    log_mutex_err(err, "WR mutex_destroy");
de_gen_mtx:
    err = lwsciipc_os_mutex_destroy(&ipch->genMutex);
    log_mutex_err(err, "GEN mutex_destroy");

fail2:
    if (ipch != NULL) {
        LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
        free(ipch);
    }
#if defined(LWSCIIPC_DEBUG)
    LWSCIIPC_DBG_STRINT("error: " LIB_FUNC, (int32_t)ret);
#endif /* LWSCIIPC_DEBUG */

done:
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Directive, 4_13), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    return ret;
}

/**
 * Clean up internal handle.
 * It cleans up internal endpoint handles which are used in internal functions
 * for opening endpoint.
 *
 * @param[in]  prev_err   LwSciIpc error code found in previous operation
 * @param[out] lwsciipch  Internal endpoint handle
 *
 * @return ::LwSciError, forward previous error in prev_err
 */
static LwSciError LwSciIpcOpenEndpointPostCommon(LwSciError prev_err,
    struct lwsciipc_internal_handle *lwsciipch)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcOpenEndpointPostCommon: "
    LwSciError ret = prev_err; /* ret is updated by update_mutex_err() */
    int32_t err;

    /* LwSciIpcOpenEndpointPreCommon() success case */
    if (lwsciipch != NULL) {
        err = lwsciipc_os_mutex_unlock(&lwsciipch->rdMutex);
        update_mutex_err(err, "RD mutex_unlock");
        err = lwsciipc_os_mutex_unlock(&lwsciipch->wrMutex);
        update_mutex_err(err, "WR mutex_unlock");
        err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
        update_mutex_err(err, "GEN mutex_unlock");

        if ((prev_err == LwSciError_Success) && (ret == LwSciError_Success)) {
            goto done;
        }
    }
    else {
        goto done;
    }

    /* failed in internal backend API */
    err = lwsciipc_os_mutex_destroy(&lwsciipch->rdMutex);
    log_mutex_err(err, "RD mutex_destroy");
    err = lwsciipc_os_mutex_destroy(&lwsciipch->wrMutex);
    log_mutex_err(err, "WR mutex_destroy");
    err = lwsciipc_os_mutex_destroy(&lwsciipch->genMutex);
    log_mutex_err(err, "GEN mutex_destroy");

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    free(lwsciipch);

done:
    return ret;
}

LwSciError LwSciIpcOpenEndpoint(const char *endpoint, LwSciIpcEndpoint *handle)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcOpenEndpoint: "
    struct lwsciipc_internal_handle *lwsciipch = NULL;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    err = lwsciipc_os_mutex_lock(&s_mutex);
    report_mutex_errto(err, "mutex_lock", fail2);

    ret = LwSciIpcOpenEndpointPreCommon(endpoint, handle, &lwsciipch);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_open_endpoint(&lwsciipch->ivch,
                lwsciipch->entry);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_open_endpoint(&lwsciipch->ipch,
                lwsciipch->entry);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_open_endpoint(&lwsciipch->c2ch,
                lwsciipch->entry);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    if (ret == LwSciError_Success) {
        lwsciipch->magic = LW_SCI_IPC_MAGIC;
        ret = lwsciipc_alloc_int_handle(lwsciipch, handle);
    }

fail:
    ret = LwSciIpcOpenEndpointPostCommon(ret, lwsciipch);
    if (ret != LwSciError_Success) {
        lwsciipc_os_error_2strs("error: " LIB_FUNC, endpoint, (int32_t)ret);
    }

    err = lwsciipc_os_mutex_unlock(&s_mutex);
    update_mutex_errto(err, "mutex_unlock", fail2);

fail2:
    LWSCIIPC_DBG_STR(LIB_FUNC "exit");

    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Directive, 4_13), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
    return ret;
}

LwSciError LwSciIpcOpenEndpointWithEventService(const char *endpoint,
    LwSciIpcEndpoint *handle, LwSciEventService *eventService)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcOpenEndpointWithEventService: "
    struct lwsciipc_internal_handle *lwsciipch = NULL;
    LwSciError ret = LwSciError_Success;
    int32_t err;
#ifdef __QNX__
    struct LwSciQnxEventLoopService *qnxLoopService;
#endif

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    err = lwsciipc_os_mutex_lock(&s_mutex);
    report_mutex_errto(err, "mutex_lock", fail2);

    if (eventService == NULL) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "eventService null");
        ret = LwSciError_BadParameter;
        goto fail;
    }
#ifdef __QNX__
    qnxLoopService = (struct LwSciQnxEventLoopService *)(void *)eventService;
    if (qnxLoopService->magic != LWSCIEVENT_MAGIC) {
        ret = LwSciError_BadParameter;
        goto fail;
    }
#endif

    ret = LwSciIpcOpenEndpointPreCommon(endpoint, handle, &lwsciipch);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_open_endpoint_with_eventservice(
                &lwsciipch->ivch,
                lwsciipch->entry, eventService);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_open_endpoint_with_eventservice(
                &lwsciipch->ipch,
                lwsciipch->entry, eventService);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_open_endpoint_with_event_service(
                &lwsciipch->c2ch,
                lwsciipch->entry, eventService);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    if (ret == LwSciError_Success) {
        lwsciipch->magic = LW_SCI_IPC_MAGIC;
        ret = lwsciipc_alloc_int_handle(lwsciipch, handle);
    }

fail:
    ret = LwSciIpcOpenEndpointPostCommon(ret, lwsciipch);
    if (ret != LwSciError_Success) {
        lwsciipc_os_error_2strs("error: " LIB_FUNC, endpoint, (int32_t)ret);
    }

    err = lwsciipc_os_mutex_unlock(&s_mutex);
    update_mutex_errto(err, "mutex_unlock", fail2);

fail2:
    LWSCIIPC_DBG_STR(LIB_FUNC "exit");

    return ret;
}

LwSciError LwSciIpcBindEventService(LwSciIpcEndpoint handle,
    LwSciEventService *eventService)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcBindEventService: "
    struct lwsciipc_internal_handle *lwsciipch = NULL;
    LwSciError ret = LwSciError_Success;
    int32_t err;
#ifdef __QNX__
    struct LwSciQnxEventLoopService *qnxLoopService;
#endif

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == false) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    if (eventService == NULL) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "eventService null");
        ret = LwSciError_BadParameter;
        goto fail;
    }

#ifdef __QNX__
    qnxLoopService = (struct LwSciQnxEventLoopService *)(void *)eventService;
    if (qnxLoopService->magic != LWSCIEVENT_MAGIC) {
        ret = LwSciError_BadParameter;
        goto fail;
    }
#endif

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    log_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            lwsciipc_ivc_bind_eventservice(lwsciipch->ivch, eventService);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            lwsciipc_ipc_bind_eventservice(lwsciipch->ipch, eventService);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            lwsciipc_c2c_bind_eventservice(lwsciipch->c2ch, eventService);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);

    return ret;
}

/**
 * Close endpoint for inter-VM and intra-VM and free internal handle.
 * This function is to check backend type and perform different clean-up on
 * internal handle to endpoint depending on backend type.
 *
 * @param[in]  lwsciipch  LwSciIpc internal object pointer
 */
static void lwsciipc_close_single_endpoint(
    struct lwsciipc_internal_handle *lwsciipch)
{
#undef LIB_FUNC
#define LIB_FUNC "lwsciipc_close_single_endpoint: "
    int32_t err;

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    log_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            lwsciipc_ivc_close_endpoint(lwsciipch->ivch);
            lwsciipch->ivch = NULL; /* ivch is freed by close_endpoint() */
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            lwsciipc_ipc_close_endpoint(lwsciipch->ipch);
            lwsciipch->ipch = NULL; /* ipch is freed by close_endpoint() */
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            lwsciipc_c2c_close_endpoint(lwsciipch->c2ch);
            lwsciipch->c2ch = NULL; /* c2ch is freed by close_endpoint() */
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            break;
    }

    lwsciipch->magic = 0U;
    s_intHandle[lwsciipch->index] = NULL;

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    log_mutex_err(err, "mutex_unlock");

    err = lwsciipc_os_mutex_destroy(&lwsciipch->rdMutex);
    log_mutex_err(err, "RD mutex_destroy");
    err = lwsciipc_os_mutex_destroy(&lwsciipch->wrMutex);
    log_mutex_err(err, "WR mutex_destroy");
    err = lwsciipc_os_mutex_destroy(&lwsciipch->genMutex);
    log_mutex_err(err, "GEN mutex_destroy");

    lwsciipc_os_debug_2strs(LIB_FUNC, lwsciipch->entry->epName, 0);

fail:
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 21_3), "<QNXBSP>:<lwpu>:<1>:<TID-385>")
    free(lwsciipch);
}

static void lwsciipc_close_all_endpoints(void)
{
    uint32_t idx;

    for (idx = 0; idx < LWSCIIPC_MAX_ENDPOINT; idx++) {
        if (s_intHandle[idx] != NULL) {
            lwsciipc_close_single_endpoint(s_intHandle[idx]);
        }
    }
}

void LwSciIpcCloseEndpoint(LwSciIpcEndpoint handle)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcCloseEndpoint: "
    struct lwsciipc_internal_handle *lwsciipch;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle");
        goto fail;
    }

    lwsciipc_close_single_endpoint(lwsciipch);

    LWSCIIPC_DBG_STR(LIB_FUNC "exit");

fail:
    return;
}

void LwSciIpcResetEndpoint(LwSciIpcEndpoint handle)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcResetEndpoint: "
    struct lwsciipc_internal_handle *lwsciipch;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle");
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    log_mutex_errto(err, "GEN mutex_lock", fail);
    err = lwsciipc_os_mutex_lock(&lwsciipch->wrMutex);
    log_mutex_errto(err, "WR mutex_lock", un_gen_mtx);
    err = lwsciipc_os_mutex_lock(&lwsciipch->rdMutex);
    log_mutex_errto(err, "RD mutex_lock", un_wr_mtx);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            lwsciipc_ivc_reset_endpoint(lwsciipch->ivch);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            lwsciipc_ipc_reset_endpoint(lwsciipch->ipch);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            lwsciipc_c2c_reset_endpoint(lwsciipch->c2ch);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->rdMutex);
    log_mutex_err(err, "RD mutex_unlock");

un_wr_mtx:
    err = lwsciipc_os_mutex_unlock(&lwsciipch->wrMutex);
    log_mutex_err(err, "WR mutex_unlock");

un_gen_mtx:
    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    log_mutex_err(err, "GEN mutex_unlock");

fail:
    LWSCIIPC_DBG_STR(LIB_FUNC "exit");
}

LwSciError LwSciIpcRead(LwSciIpcEndpoint handle, void *buf, size_t size,
    int32_t *bytes)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcRead: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((buf == NULL) || (bytes == NULL) ||
        (size == 0UL) || (size > (size_t)UINT32_MAX) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->rdMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_read(lwsciipch->ivch, buf, (uint32_t)size,
                (uint32_t *)(void *)bytes);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_read(lwsciipch->ipch, buf, (uint32_t)size,
                (uint32_t *)(void *)bytes);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_read(lwsciipch->c2ch, buf, (uint32_t)size,
                (uint32_t *)(void *)bytes);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default    :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->rdMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}

LwSciError LwSciIpcReadPeek(LwSciIpcEndpoint handle, void *buf, int32_t offset,
    int32_t count, int32_t *bytes)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcReadPeek: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((buf == NULL) || (bytes == NULL) || (offset < 0) || (count <= 0) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->rdMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_read_peek(lwsciipch->ivch, buf,
                (uint32_t)offset, (uint32_t)count, (uint32_t *)(void *)bytes);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_read_peek(lwsciipch->ipch, buf,
                (uint32_t)offset, (uint32_t)count, (uint32_t *)(void *)bytes);
            break;
        default    :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->rdMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}

LwSciError LwSciIpcReadGetNextFrame(LwSciIpcEndpoint handle,
    const volatile void **buf)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcReadGetNextFrame: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((buf == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->rdMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            *buf = lwsciipc_ivc_read_get_next_frame(lwsciipch->ivch);
            if (*buf == NULL) {
                ret = lwsciipc_ivc_check_read(lwsciipch->ivch);
            }
            else {
                ret = LwSciError_Success;
            }
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            *buf = lwsciipc_ipc_read_get_next_frame(lwsciipch->ipch);
            if (*buf == NULL) {
                ret = lwsciipc_ipc_check_read(lwsciipch->ipch);
            }
            else {
                ret = LwSciError_Success;
            }
            break;
        default    :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->rdMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}

LwSciError LwSciIpcReadAdvance(LwSciIpcEndpoint handle)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcReadAdvance: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->rdMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_read_advance(lwsciipch->ivch);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_read_advance(lwsciipch->ipch);
            break;
        default    :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->rdMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}

LwSciError LwSciIpcWrite(LwSciIpcEndpoint handle, const void *buf, size_t size,
    int32_t *bytes)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcWrite: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((buf == NULL) || (bytes == NULL) ||
        (size == 0UL) || (size > (size_t)UINT32_MAX) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->wrMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch(lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_write(lwsciipch->ivch, buf, (uint32_t)size,
                (uint32_t *)(void *)bytes);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_write(lwsciipch->ipch, buf, (uint32_t)size,
                (uint32_t *)(void *)bytes);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_write(lwsciipch->c2ch, buf, (uint32_t)size,
                (uint32_t *)(void *)bytes);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->wrMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}

LwSciError LwSciIpcWritePoke(LwSciIpcEndpoint handle, const void *buf,
    int32_t offset, int32_t count, int32_t *bytes)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcWritePoke: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((buf == NULL) || (bytes == NULL) || (offset < 0) || (count <= 0) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->wrMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch(lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_write_poke(lwsciipch->ivch, buf,
                (uint32_t)offset, (uint32_t)count, (uint32_t *)(void *)bytes);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_write_poke(lwsciipch->ipch, buf,
                (uint32_t)offset, (uint32_t)count, (uint32_t *)(void *)bytes);
            break;
        default :
            LWSCIIPC_ERR_STRUINT(
                "error: " LIB_FUNC "Unsupported backend type",
                lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->wrMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}

LwSciError LwSciIpcWriteGetNextFrame(LwSciIpcEndpoint handle, volatile void **buf)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcWriteGetNextFrame: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((buf == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->wrMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            *buf = lwsciipc_ivc_write_get_next_frame(lwsciipch->ivch);
            if (*buf == NULL) {
                ret = lwsciipc_ivc_check_write(lwsciipch->ivch);
            }
            else {
                ret = LwSciError_Success;
            }
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            *buf = lwsciipc_ipc_write_get_next_frame(lwsciipch->ipch);
            if (*buf == NULL) {
                ret = lwsciipc_ipc_check_write(lwsciipch->ipch);
            }
            else {
                ret = LwSciError_Success;
            }
            break;
        default    :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->wrMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}

LwSciError LwSciIpcWriteAdvance(LwSciIpcEndpoint handle)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcWriteAdvance: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->wrMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_write_advance(lwsciipch->ivch);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_write_advance(lwsciipch->ipch);
            break;
        default    :
            LWSCIIPC_ERR_STRUINT(
                "error: " LIB_FUNC "Unsupported backend type",
                lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->wrMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG_RW)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG_RW */

    return ret;
}


LwSciError LwSciIpcGetEndpointInfo(LwSciIpcEndpoint handle,
                LwSciIpcEndpointInfo *info)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcGetEndpointInfo: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((info == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_get_endpoint_info(lwsciipch->ivch, info);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_get_endpoint_info(lwsciipch->ipch, info);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_get_endpoint_info(lwsciipch->c2ch,
                (LwSciC2cPcieEndpointInfo *)info);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

/*
 * This API is <QNX OS> specific.
 */
LwSciError LwSciIpcGetEndpointInfoInternal(LwSciIpcEndpoint handle,
                LwSciIpcEndpointInfoInternal *info)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcGetEndpointInfoInternal: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((info == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
#ifdef __QNX__
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_get_endpoint_info_internal(lwsciipch->ivch, info);
            break;
#endif /* __QNX__ */
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_get_endpoint_info_internal(lwsciipch->c2ch,
                (LwSciC2cPcieEndpointInernalInfo *)&info->c2cInfo);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

#if (LW_IS_SAFETY == 0)
LwSciError LwSciIpcGetCookie(LwSciIpcEndpoint handle,
                LwSciIpcC2cCookie *cookie)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcGetCookie: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((cookie == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_get_cookie(lwsciipch->c2ch, cookie);
            break;
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

LwSciError LwSciIpcSetCookie(LwSciIpcEndpoint handle,
                LwSciIpcC2cCookie cookie)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcSetCookie: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((cookie == 0) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_set_cookie(lwsciipch->c2ch, cookie);
            break;
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}
#endif /* (LW_IS_SAFETY == 0) */

LwSciError LwSciIpcGetEventNotifier(LwSciIpcEndpoint handle,
               LwSciEventNotifier **eventNotifier)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcGetEventNotifier: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((eventNotifier == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_get_eventnotifier(lwsciipch->ivch,
                eventNotifier);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_get_eventnotifier(lwsciipch->ipch,
                eventNotifier);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_get_eventnotifier(lwsciipch->c2ch,
                eventNotifier);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

#ifdef LINUX
/*
 * This API is <LINUX OS> specific.
 */
LwSciError LwSciIpcGetLinuxEventFd(LwSciIpcEndpoint handle, int32_t *fd)
{
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;

    if (s_initDone != LwBoolTrue) {
        lwsciipc_err("LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) ||
    (fd == NULL)) {
        lwsciipc_err("Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    (void)lwsciipc_os_mutex_lock(&lwsciipch->genMutex);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_get_eventfd(lwsciipch->ivch, fd);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_get_eventfd(lwsciipch->ipch, fd);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_get_eventfd(lwsciipch->c2ch, fd);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            lwsciipc_err("Unsupported backend type (%d)", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    (void)lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);

fail:
    return ret;
}
#endif /* LINUX */

LwSciError LwSciIpcGetEvent(LwSciIpcEndpoint handle, uint32_t *events)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcGetEvent: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((events == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex); /* state */
    report_mutex_errto(err, "mutex_lock", fail);

    /* read/write mutexes are handled in each internal backend API */
    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_get_event(lwsciipch->ivch, events, lwsciipch);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_get_event(lwsciipch->ipch, events, lwsciipch);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_get_event(lwsciipch->c2ch, events, lwsciipch);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
#if defined(LWSCIIPC_DEBUG)
    if (ret != LwSciError_Success) {
        LWSCIIPC_DBG_STRINT(LIB_FUNC "error", (int32_t)ret);
    }
#endif /* LWSCIIPC_DEBUG */

    return ret;
}

/*
 * All memory accesses done by tegra_ivc_can_read/write() calls fall into
 * one of the following two categories, so this API doesn’t need mutexes:
 * (1) Reads of constant ivc fields
 *     (nframes, tx_channel, rx_channel)
 * (2) Reads from volatile shared memory state
 *     (channel state, w_count, and r_count fields)
 */
bool LwSciIpcCanRead(LwSciIpcEndpoint handle)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcCanRead: "
    struct lwsciipc_internal_handle *lwsciipch;
    bool ret;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = false;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = false;
        goto fail;
    }

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_can_read(lwsciipch->ivch);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_can_read(lwsciipch->ipch);
            break;
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = false;
            break;
    }

fail:
    return ret;
}

/*
 * All memory accesses done by tegra_ivc_can_read/write() calls fall into
 * one of the following two categories, so this API doesn’t need mutexes:
 * (1) Reads of constant ivc fields
 *     (nframes, tx_channel, rx_channel)
 * (2) Reads from volatile shared memory state
 *     (channel state, w_count, and r_count fields)
 */
bool LwSciIpcCanWrite(LwSciIpcEndpoint handle)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcCanWrite: "
    struct lwsciipc_internal_handle *lwsciipch;
    bool ret;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = false;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = false;
        goto fail;
    }

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_can_write(lwsciipch->ivch);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_can_write(lwsciipch->ipch);
            break;
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = false;
            break;
    }

fail:
    return ret;
}

#ifdef __QNX__
/*
 * This API is <QNX OS> specific.
 */
LwSciError LwSciIpcSetQnxPulseParam(LwSciIpcEndpoint handle,
    int32_t coid, int16_t pulsePriority, int16_t pulseCode,
    void *pulseValue)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcSetQnxPulseParam: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    if (lwsciipc_has_eventservice(lwsciipch) == LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "not compatible with EventSErvice");
        ret = LwSciError_NotSupported;
        goto fail;
    }

    ret = lwsciipc_os_check_pulse_param(coid, pulsePriority, pulseCode);
    if (ret != LwSciError_Success) {
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_set_qnx_pulse_param(lwsciipch->ivch,
                coid, pulsePriority, pulseCode, pulseValue);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_set_qnx_pulse_param(lwsciipch->ipch,
                coid, pulsePriority, pulseCode, pulseValue);
            break;
#if (LW_IS_SAFETY == 0)
        case LWSCIIPC_BACKEND_C2C :
            ret = lwsciipc_c2c_set_qnx_pulse_param(lwsciipch->c2ch,
                coid, pulsePriority, pulseCode, pulseValue);
            break;
#endif /* (LW_IS_SAFETY == 0) */
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}
#endif /* __QNX__ */

#ifdef __QNX__
/*
 * This API is <QNX OS> specific.
 */
LwSciError LwSciIpcOpenCfgBlob(void)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcOpenCfgBlob: "
    LwSciError ret;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    err = lwsciipc_os_mutex_lock(&s_mutex);
    report_mutex_errto(err, "mutex_lock", fail);

    ret =  lwsciipc_os_open_config();

    err = lwsciipc_os_mutex_unlock(&s_mutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}
#endif /* __QNX__ */

#ifdef __QNX__
/*
 * This API is <QNX OS> specific.
 */
LwSciError LwSciIpcGetEndpointAccessInfo(const char *endpoint,
    LwSciIpcEndpointAccessInfo *info)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcGetEndpointAccessInfo: "
    LwSciError ret;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    err = lwsciipc_os_mutex_lock(&s_mutex);
    report_mutex_errto(err, "mutex_lock", fail);

    ret = lwsciipc_os_get_endpoint_access_info(endpoint, info);

    err = lwsciipc_os_mutex_unlock(&s_mutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}
#endif /* __QNX__ */

#ifdef __QNX__
/*
 * This API is <QNX OS> specific.
 */
void LwSciIpcCloseCfgBlob(void)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcCloseCfgBlob: "
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    err = lwsciipc_os_mutex_lock(&s_mutex);
    log_mutex_errto(err, "mutex_lock", fail);

    lwsciipc_os_close_config();

    err = lwsciipc_os_mutex_unlock(&s_mutex);
    log_mutex_errto(err, "mutex_unlock", fail);

fail:
    LWSCIIPC_DBG_STR(LIB_FUNC "exit");
    return;
}
#endif /* __QNX__ */

LwSciError LwSciIpcErrnoToLwSciErr(int32_t err)
{
    return ErrnoToLwSciErr(err);
}

int32_t LwSciIpcLwSciErrToErrno(LwSciError lwSciErr)
{
    LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Directive, 4_7), "<QNXBSP>:<lwpu>:<1>:<TID-361>")
    return LwSciErrToErrno(lwSciErr);
}

/*----------------------------------------------------------------------------
 * LwMap buffer import/export related APIs
 *----------------------------------------------------------------------------
 */

/*
 * Get endpoint authentication token
 *
 * QNX implementation:
 *     The VUID of the LwSciIpcEndpoint.
 * Linux userspace implementation:
 *     The file descriptor of the dev node that represents the particular
 *     LwSciIpcEndpoint.
 * Linux kernel implementation:
 *     The result of casting the LwSciIpcEndpoint to a uintptr_t.
 */
LwSciError LwSciIpcEndpointGetAuthToken(LwSciIpcEndpoint handle,
    LwSciIpcEndpointAuthToken *authToken)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcEndpointGetAuthToken: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((authToken == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_endpoint_get_auth_token(lwsciipch->ivch,
                authToken);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_endpoint_get_auth_token(lwsciipch->ipch,
                authToken);
            break;
        default :
            LWSCIIPC_ERR_STRUINT("error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    if ((ret != LwSciError_Success) && (authToken != NULL)) {
        *authToken = LWSCIIPC_ENDPOINT_AUTHTOKEN_ILWALID;
    }

    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

#ifdef __QNX__
/*
 * Validate authentication token of endpoint and translate it to VUID
 * This API is <QNX OS> specific.
 *
 * QNX:
 *   - The LwSciIpc resource manager creates a ranged ability, "LwSciIpc",
 *     during initialization.
 *   - Each LwSciIpcEndpoint has an associated "LwSciIpc" ability whose
 *     64-bit value is the endpoint's VUID.
 *   - LwSciIpcEndpointValidateVuidQnx() uses iofunc_client_info_able() to
 *     determine whether the client process has the ability.
 * Linux userspace:
 *   - The LwSciIpc kernel driver looks up the auth token (an fd) in the
 *     calling process's fd table, verifies that the struct file is an
 *     LwSciIpc file, and extracts the VUID from it.
 */
LwSciError LwSciIpcEndpointValidateAuthTokenQnx(resmgr_context_t *ctp,
    LwSciIpcEndpointAuthToken authToken,
    LwSciIpcEndpointVuid *localUserVuid)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcEndpointValidateAuthTokenQnx: "
#ifdef USE_IOLAUNCHER_FOR_SELWRITY
    struct _client_able able = {
        .range_lo = authToken,
        .range_hi = authToken
    };
    struct _client_info *infop = NULL;
#endif /* USE_IOLAUNCHER_FOR_SELWRITY */
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if ((ctp == NULL) || (localUserVuid == NULL)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&s_mutex);
    report_mutex_errto(err, "mutex_lock", fail2);

#ifdef USE_IOLAUNCHER_FOR_SELWRITY
    {
        int32_t val;

        val = procmgr_ability_lookup(LWSCIIPC_ABILITY_ID);
        if (val < 0) {
            LWSCIIPC_DBG_STRINT("error: " LIB_FUNC "ability lookup", val);
            ret = LwSciError_NotPermitted;
            goto fail;
        }
        able.ability = (uint32_t)val;

        err = iofunc_client_info_able(ctp, 0, &infop, 0, &able, 1);
        if ((err != EOK) || (able.flags == 0U)) {
            LWSCIIPC_ERR_STR("error: " LIB_FUNC "permission error");
            ret = LwSciError_AccessDenied;
            goto fail;
        }

        err = iofunc_client_info_ext_free(&infop);
        report_os_errto(err, "client_info_ext_free", fail);
    }
#endif /* USE_IOLAUNCHER_FOR_SELWRITY */

    /* vuid is same with authToken for QNX OS */
    *localUserVuid = authToken;

fail:
    if ((ret != LwSciError_Success) && (localUserVuid != NULL)) {
        *localUserVuid = LWSCIIPC_ENDPOINT_VUID_ILWALID;
    }

    err = lwsciipc_os_mutex_unlock(&s_mutex);
    update_mutex_errto(err, "mutex_unlock", fail2);

fail2:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}
#endif /* __QNX__ */

/*
 * Translate VUID(VM unique ID) to topology ID and VUID of peer endpoint.
 * This API is <QNX OS> specific.
 *
 * - If localUserVuid is not the VUID of an LwSciIpcEndpoint in the same VM
 *   as the caller of the mechanism, the operation fails.
 * - Otherwise, if the other end of the LwSciIpcEndpoint represented by
 *   localUserVuid is in the same VM:
 *   - peerTopoId is set to the calling VM's LwSciIpcTopoId
 *   - peerUserVuid is set to the VUID of the other end of the
 *     LwSciIpcEndpoint represented by localUserVuid
 * - Otherwise (if the other end of the LwSciIpcEndpoint represented by
 *   localUserVuid is in a different VM):
 *   - peerTopoId is set to the other VM's LwSciIpcTopoId
 *   - peerUserVuid is set to the VUID in the other VM of the other end of
 *     the LwSciIpcEndpoint in this VM represented by localUserVuid
 */
LwSciError LwSciIpcEndpointMapVuid(LwSciIpcEndpointVuid localUserVuid,
    LwSciIpcTopoId *peerTopoId, LwSciIpcEndpointVuid *peerUserVuid)
{
#ifdef __QNX__
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcEndpointMapVuid: "
    struct LwSciIpcConfigEntry *entry;
    LwSciIpcVUID64 vuid64;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if ((peerTopoId == NULL) || (peerUserVuid == NULL)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid parameter");
        ret = LwSciError_BadParameter;
        goto fail2;
    }

    err = lwsciipc_os_mutex_lock(&s_mutex);
    report_mutex_errto(err, "mutex_lock", fail2);

    /* search vuid */
    ret = lwsciipc_os_get_config_entry_by_vuid(localUserVuid, &entry);
    if (ret != LwSciError_Success) {
        LWSCIIPC_ERR_STRULONG("error: " LIB_FUNC "Not found entry for VUID",
            localUserVuid);
        goto fail;
    }

    switch (entry->backend) {
        case LWSCIIPC_BACKEND_ITC:
        case LWSCIIPC_BACKEND_IPC:
            peerTopoId->SocId = LWSCIIPC_SELF_SOCID;
            peerTopoId->VmId = LWSCIIPC_SELF_VMID;
            *peerUserVuid = (LwSciIpcEndpointVuid)(entry->vuid^1UL);
            break;
        case LWSCIIPC_BACKEND_IVC:
            {
                peerTopoId->SocId = LWSCIIPC_SELF_SOCID;
                peerTopoId->VmId = entry->peerVmid;
                vuid64.value = (LwSciIpcEndpointVuid)entry->vuid;
                vuid64.bit.vmid = entry->peerVmid;
                *peerUserVuid = (vuid64.value);
            }
            break;
        default:
            LWSCIIPC_ERR_STR("error: " LIB_FUNC "Not supported backend");
            ret = LwSciError_NotSupported;
            break;
    }

    LWSCIIPC_DBG_STRULONG(LIB_FUNC "(vuid)", localUserVuid);
    LWSCIIPC_DBG_STRINT(LIB_FUNC "(peer SocId)", peerTopoId->SocId);
    LWSCIIPC_DBG_STRINT(LIB_FUNC "(peer VmId)", peerTopoId->VmId);
    LWSCIIPC_DBG_STRULONG(LIB_FUNC "(peer vuid)", *peerUserVuid);

fail:
    err = lwsciipc_os_mutex_unlock(&s_mutex);
    update_mutex_errto(err, "mutex_unlock", fail2);

fail2:
    if (ret != LwSciError_Success) {
        if (peerTopoId != NULL) {
            peerTopoId->SocId = 0;
            peerTopoId->VmId = 0;
        }
        if (peerUserVuid != NULL) {
            *peerUserVuid = LWSCIIPC_ENDPOINT_VUID_ILWALID;
        }
    }

    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
#endif /* __QNX__ */

#ifdef LINUX
    return LwSciError_NotSupported;
#endif /* LINUX */
}

/*
 * Get VUID(VM unique ID) of endpoint
 */
LwSciError LwSciIpcEndpointGetVuid(LwSciIpcEndpoint handle,
    LwSciIpcEndpointVuid *vuid)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcEndpointGetVuid: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_Success;
    int32_t err;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if ((vuid == NULL) ||
        (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse)) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    err = lwsciipc_os_mutex_lock(&lwsciipch->genMutex);
    report_mutex_errto(err, "mutex_lock", fail);

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_IVC :
            ret = lwsciipc_ivc_endpoint_get_vuid(lwsciipch->ivch,
                vuid);
            break;
        case LWSCIIPC_BACKEND_ITC :
        case LWSCIIPC_BACKEND_IPC :
            ret = lwsciipc_ipc_endpoint_get_vuid(lwsciipch->ipch,
                vuid);
            break;
        default :
            LWSCIIPC_ERR_STRUINT( "error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

    err = lwsciipc_os_mutex_unlock(&lwsciipch->genMutex);
    update_mutex_errto(err, "mutex_unlock", fail);

fail:
    if ((ret != LwSciError_Success) && (vuid != NULL)) {
        *vuid = LWSCIIPC_ENDPOINT_VUID_ILWALID;
    }

    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

#if (LW_IS_SAFETY == 0)
LwSciError LwSciIpcEndpointGetTopoId(LwSciIpcEndpoint handle,
    LwSciIpcTopoId *localTopoId)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcEndpointGetTopoId: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret = LwSciError_BadParameter;

    LWSCIIPC_DBG_STR(LIB_FUNC "enter");

    if (localTopoId == NULL) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_ITC:
        case LWSCIIPC_BACKEND_IPC:
            localTopoId->SocId = LWSCIIPC_SELF_SOCID;
            localTopoId->VmId = LWSCIIPC_SELF_VMID;
            ret = LwSciError_Success;
            break;
        case LWSCIIPC_BACKEND_IVC:
            {
                localTopoId->SocId = LWSCIIPC_SELF_SOCID;
                ret = lwsciipc_os_get_vmid(&localTopoId->VmId);
            }
            break;
        case LWSCIIPC_BACKEND_C2C:
            {
                ret = lwsciipc_os_get_socid(&localTopoId->SocId);
                if (ret != LwSciError_Success) {
                    goto fail;
                }
                ret = lwsciipc_os_get_vmid(&localTopoId->VmId);
            }
            break;
        default:
            LWSCIIPC_ERR_STR("error: " LIB_FUNC "Not supported backend");
            ret = LwSciError_NotSupported;
            break;
    }

fail:
    if (ret != LwSciError_Success) {
        if (localTopoId != NULL) {
            localTopoId->SocId = 0;
            localTopoId->VmId = 0;
        }
    }

    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

LwSciError LwSciIpcGetC2cCopyFuncSet(LwSciIpcEndpoint handle, void *fn)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcGetC2cCopyFuncSet: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_C2C_PCIE :
            ret = lwsciipc_c2c_get_c2ccopy_funcset(lwsciipch->type, fn);
            break;

        default :
            LWSCIIPC_ERR_STRUINT( "error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}

LwSciError LwSciIpcValidateC2cCopyFuncSet(LwSciIpcEndpoint handle,
    const void *fn)
{
#undef LIB_FUNC
#define LIB_FUNC "LwSciIpcValidateC2cCopyFuncSet: "
    struct lwsciipc_internal_handle *lwsciipch;
    LwSciError ret;

    if (s_initDone != LwBoolTrue) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "LwSciIpc is not initialized");
        ret = LwSciError_NotInitialized;
        goto fail;
    }

    if (lwsciipc_is_valid_handle(handle, &lwsciipch) == LwBoolFalse) {
        LWSCIIPC_ERR_STR("error: " LIB_FUNC "Invalid handle or parameter");
        ret = LwSciError_BadParameter;
        goto fail;
    }

    switch (lwsciipch->type) {
        case LWSCIIPC_BACKEND_C2C_PCIE :
            ret = lwsciipc_c2c_validate_c2ccopy_funcset(lwsciipch->type, fn);
            break;

        default :
            LWSCIIPC_ERR_STRUINT( "error: " LIB_FUNC
                "Unsupported backend type", lwsciipch->type);
            ret = LwSciError_NotSupported;
            break;
    }

fail:
    LWSCIIPC_DBG_STRINT(LIB_FUNC "exit", (int32_t)ret);

    return ret;
}
#endif /* (LW_IS_SAFETY == 0) */
