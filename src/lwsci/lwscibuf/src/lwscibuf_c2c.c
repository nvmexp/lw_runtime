/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscilog.h"
#include "lwscicommon_covanalysis.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscibuf_c2c_internal.h"
#include "lwscibuf_c2c_priv.h"
#include "lwscibuf_obj_mgmt.h"

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
static void colwertToC2cFlushRanges(
    const LwSciBufFlushRanges* flushRanges,
    LwSciC2cPcieFlushRange* c2cFlushRanges,
    size_t numFlushRanges)
{
    size_t index = 0UL;

    LWSCI_FNENTRY("");

    for (index = 0UL; index < numFlushRanges; index++) {
        c2cFlushRanges[index].offset = flushRanges[index].offset;
        c2cFlushRanges[index].size = flushRanges[index].size;
    }

    LWSCI_FNEXIT("");
}
#endif

LwSciError LwSciBufOpenIndirectChannelC2c(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciEventService* eventService,
    size_t numRequests,
    size_t numFlushRanges,
    size_t numPreFences,
    size_t numPostFences,
    LwSciC2cHandle* channelHandle)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    struct LwSciIpcEndpointInfo info = {};
    LwSciC2cHandle tmpHandle = NULL;
    int c2cErrorCode = -1;
    LwSciC2cCopyFuncs tmpCopyFuncs = {};

    LWSCI_FNENTRY("");

    if ((0U == ipcEndpoint) || (NULL == eventService) || (0U == numRequests) ||
        (0U == numFlushRanges) || (0U == numPostFences)
         || (NULL == channelHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufOpenIndirectChannelC2c.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciIpcGetEndpointInfo(ipcEndpoint, &info);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate LwSciIpcEndpoint.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *channelHandle = NULL;

    tmpHandle = LwSciCommonCalloc(1U, sizeof(*tmpHandle));
    if (NULL == tmpHandle) {
        LWSCI_ERR_STR("Insufficient system memory when allocating memory for LwSciC2cHandle.");
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    tmpHandle->magic = LWSCIBUF_C2C_CHANNEL_MAGIC;
    tmpHandle->ipcEndpoint = ipcEndpoint;

    err = LwSciIpcGetC2cCopyFuncSet(ipcEndpoint, &tmpCopyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciIpcGetC2cCopyFuncSet failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }
    tmpHandle->copyFuncs = tmpCopyFuncs;

    if (NULL == tmpCopyFuncs.openIndirectChannel) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for opening indirect channel is NULL.");
        LwSciCommonPanic();
    }

    /* TODO: Note that we are passing pcieStreamHandle directly below for
     * 6.0.2.0. In future, if LwSciC2c team provides an abstraction layer then
     * we don't need to specifically pass pcieStreamHandle. If they don't then
     * we should query the IPC backend type here and pass appropriate handle
     * corresponding to appropriate interface.
     */
    c2cErrorCode = tmpCopyFuncs.openIndirectChannel(ipcEndpoint,
                            eventService, numRequests, numFlushRanges,
                            numPreFences, numPostFences,
                            &((tmpHandle->interfaceHandle).pcieStreamHandle));
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not open indirect channel for the interface.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    *channelHandle = tmpHandle;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
        "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_handle:
    LwSciCommonFree(tmpHandle);

#else
    (void)ipcEndpoint;
    (void)eventService;
    (void)numRequests;
    (void)numFlushRanges;
    (void)numPreFences;
    (void)numPostFences;
    (void)channelHandle;

    err = LwSciError_NotSupported;
    goto ret;
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufRegisterSourceObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciBufObj bufObj,
    LwSciC2cBufSourceHandle* sourceHandle)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    int c2cErrorCode = -1;
    /* TODO: This will get replaced by abstract datatype of LwSciC2c provides
     * one or will be defined in the if condition where we switch the
     * backend type.
     */
    LwSciC2cPcieBufRmHandle pcieBufRmHandle = {};
    LwSciBufRmHandle bufRmHandle = {};
    LwSciC2cBufSourceHandle tmpSourceHandle = NULL;
    uint64_t offset = 0UL;
    uint64_t len = 0UL;

    LWSCI_FNENTRY("");

    if ((NULL == channelHandle) || (NULL == bufObj) || (NULL == sourceHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufRegisterSourceObjIndirectChannelC2c.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LWSCIBUF_C2C_CHANNEL_MAGIC != channelHandle->magic) {
        LWSCI_ERR_STR("Invalid channelHandle.");
        LwSciCommonPanic();
    }

    *sourceHandle = NULL;

    tmpSourceHandle = LwSciCommonCalloc(1U, sizeof(*tmpSourceHandle));
    if (NULL == tmpSourceHandle) {
        LWSCI_ERR_STR("Insufficient system memory when allocating memory for LwSciC2cBufSourceHandle.");
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    tmpSourceHandle->magic = LWSCIBUF_C2C_SOURCE_HANDLE_MAGIC;
    tmpSourceHandle->channelHandle = channelHandle;

    err = LwSciBufObjGetMemHandle(bufObj, &bufRmHandle, &offset, &len);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufObjGetMemHandle failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

#if !defined(__x86_64__)
    pcieBufRmHandle.memHandle = bufRmHandle.memHandle;
#else
    /* TODO: Fill X86 handles when LwSciC2c extends LwSciC2cPcieBufRmHandle */
#endif

    /* Take reference to LwSciBufObj. */
    err = LwSciBufObjDup(bufObj, &(tmpSourceHandle->bufObj));
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufObjDup failed when registering source handle.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    /* lwsciipc_c2c_validate_c2ccopy_funcset() not supported.
    err = LwSciIpcValidateC2cCopyFuncSet(channelHandle->ipcEndpoint,
            &channelHandle->copyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not validate copy functions.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }
    */

    if (NULL == channelHandle->copyFuncs.bufMapSourceMemHandle) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for mapping source memhandle is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = channelHandle->copyFuncs.bufMapSourceMemHandle(
                    &pcieBufRmHandle, LWSCIC2C_PCIE_PERM_READWRITE,
                    channelHandle->interfaceHandle.pcieStreamHandle,
                    &(tmpSourceHandle->interfaceSourceHandle.pcieSourceHandle));
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not obtain source handle from the c2c interface.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    if (NULL == channelHandle->copyFuncs.bufRegisterSourceHandle) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for registering source handle is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = channelHandle->copyFuncs.bufRegisterSourceHandle(
                    channelHandle->interfaceHandle.pcieStreamHandle,
                    tmpSourceHandle->interfaceSourceHandle.pcieSourceHandle);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not register source handle to the channel handle for the c2c interface.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    *sourceHandle = tmpSourceHandle;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
        "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_handle:
    LwSciCommonFree(tmpSourceHandle);

#else
    (void)channelHandle;
    (void)bufObj;
    (void)sourceHandle;

    err = LwSciError_NotSupported;
    goto ret;
#endif
ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufRegisterTargetObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciBufObj bufObj,
    LwSciC2cBufTargetHandle* targetHandle)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciC2cBufTargetHandle tmpTargetHandle = NULL;
    LwSciC2cInterfaceTargetHandle tmpInterfaceTargetHandle = {};
    int c2cErrorCode = -1;

    LWSCI_FNENTRY("");

    if ((NULL == channelHandle) || (NULL == bufObj) || (NULL == targetHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufRegisterTargetObjIndirectChannelC2c.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LWSCIBUF_C2C_CHANNEL_MAGIC != channelHandle->magic) {
        LWSCI_ERR_STR("Invalid channelHandle.");
        LwSciCommonPanic();
    }

    *targetHandle = NULL;

    tmpTargetHandle = LwSciCommonCalloc(1U, sizeof(*tmpTargetHandle));
    if (NULL == tmpTargetHandle) {
        LWSCI_ERR_STR("Insufficient system memory when allocating memory for LwSciC2cBufTargetHandle.");
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    tmpTargetHandle->magic = LWSCIBUF_C2C_TARGET_HANDLE_MAGIC;
    tmpTargetHandle->channelHandle = channelHandle;

    /* Take reference to LwSciBufObj. */
    err = LwSciBufObjDup(bufObj, &(tmpTargetHandle->bufObj));
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufObjDup failed when registering source handle.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    err = LwSciBufObjGetC2cInterfaceTargetHandle(bufObj,
            &tmpInterfaceTargetHandle);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufObjGetC2cInterfaceTargetHandle failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    /* lwsciipc_c2c_validate_c2ccopy_funcset() not supported.
    err = LwSciIpcValidateC2cCopyFuncSet(channelHandle->ipcEndpoint,
            &channelHandle->copyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not validate copy functions.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }
    */

    if (NULL == channelHandle->copyFuncs.bufDupTargetHandle) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for duplicating target handle is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = channelHandle->copyFuncs.bufDupTargetHandle(
                    tmpInterfaceTargetHandle.pcieTargetHandle,
                    &tmpTargetHandle->interfaceTargetHandle.pcieTargetHandle);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not duplicate target handle.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    if (NULL == channelHandle->copyFuncs.bufRegisterTargetHandle) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for registering target handle is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = channelHandle->copyFuncs.bufRegisterTargetHandle(
                    channelHandle->interfaceHandle.pcieStreamHandle,
                    tmpTargetHandle->interfaceTargetHandle.pcieTargetHandle);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not register target handle.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_handle;
    }

    *targetHandle = tmpTargetHandle;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
        "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_handle:
    LwSciCommonFree(tmpTargetHandle);

#else
    (void)channelHandle;
    (void)bufObj;
    (void)targetHandle;

    err = LwSciError_NotSupported;
    goto ret;
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufPushCopyIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cBufSourceHandle sourceHandle,
    LwSciC2cBufTargetHandle targetHandle,
    const LwSciBufFlushRanges* flushRanges,
    size_t numFlushRanges)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciC2cPcieFlushRange* c2cFlushRanges = NULL;
    int c2cErrorCode = -1;

    LWSCI_FNENTRY("");

    if ((NULL == channelHandle) || (NULL == sourceHandle) ||
        (NULL == targetHandle) || (NULL == flushRanges) ||
        (0U == numFlushRanges)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufPushCopyIndirectChannelC2c.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((LWSCIBUF_C2C_CHANNEL_MAGIC != channelHandle->magic) ||
        (LWSCIBUF_C2C_SOURCE_HANDLE_MAGIC != sourceHandle->magic) ||
        (LWSCIBUF_C2C_TARGET_HANDLE_MAGIC != targetHandle->magic)) {
        LWSCI_ERR_STR("Invalid handle supplied to LwSciBufPushCopyIndirectChannelC2c.");
        LwSciCommonPanic();
    }

    c2cFlushRanges = LwSciCommonCalloc(numFlushRanges, sizeof(*c2cFlushRanges));
    if (NULL == c2cFlushRanges) {
        LWSCI_ERR_STR("Insufficient system memory encountered in LwSciBufPushCopyIndirectChannelC2c.");
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    colwertToC2cFlushRanges(flushRanges, c2cFlushRanges, numFlushRanges);

    /* lwsciipc_c2c_validate_c2ccopy_funcset() not supported.
    err = LwSciIpcValidateC2cCopyFuncSet(channelHandle->ipcEndpoint,
            &channelHandle->copyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not validate copy functions.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_flushRanges;
    }
    */

    if (NULL == channelHandle->copyFuncs.pushCopyIndirectChannel) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for push copy for channel is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = channelHandle->copyFuncs.pushCopyIndirectChannel(
                    channelHandle->interfaceHandle.pcieStreamHandle,
                    sourceHandle->interfaceSourceHandle.pcieSourceHandle,
                    targetHandle->interfaceTargetHandle.pcieTargetHandle,
                    c2cFlushRanges, numFlushRanges);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not push copy request for the channel.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_flushRanges;
    }

free_flushRanges:
    LwSciCommonFree(c2cFlushRanges);

#else
    (void)channelHandle;
    (void)sourceHandle;
    (void)targetHandle;
    (void)flushRanges;
    (void)numFlushRanges;

    err = LwSciError_NotSupported;
    goto ret;
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufPushSubmitIndirectChannelC2c(
    LwSciC2cHandle channelHandle)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    int c2cErrorCode = -1;

    LWSCI_FNENTRY("");

    if (NULL == channelHandle) {
        LWSCI_ERR_STR("NULL channelHandle supplied.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LWSCIBUF_C2C_CHANNEL_MAGIC != channelHandle->magic) {
        LWSCI_ERR_STR("Invalid channelHandle.");
        LwSciCommonPanic();
    }

    /* lwsciipc_c2c_validate_c2ccopy_funcset() not supported.
    err = LwSciIpcValidateC2cCopyFuncSet(channelHandle->ipcEndpoint,
            &channelHandle->copyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not validate copy functions.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    */

    if (NULL == channelHandle->copyFuncs.pushSubmitIndirectChannel) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for push submit channel is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = channelHandle->copyFuncs.pushSubmitIndirectChannel(
                    channelHandle->interfaceHandle.pcieStreamHandle);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not submit push request for the channel.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#else
    (void)channelHandle;

    err = LwSciError_NotSupported;
    goto ret;
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufCloseIndirectChannelC2c(
    LwSciC2cHandle channelHandle)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    int c2cErrorCode = -1;

    LWSCI_FNENTRY("");

    if (NULL == channelHandle) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("NULL channelHandle passed to LwSciBufCloseIndirectChannelC2c.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LWSCIBUF_C2C_CHANNEL_MAGIC != channelHandle->magic) {
        LWSCI_ERR_STR("Invalid channelHandle.");
        LwSciCommonPanic();
    }

    /* lwsciipc_c2c_validate_c2ccopy_funcset() not supported.
    err = LwSciIpcValidateC2cCopyFuncSet(channelHandle->ipcEndpoint,
            &channelHandle->copyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not validate copy functions.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    */

    if (NULL == channelHandle->copyFuncs.closeIndirectChannel) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for mapping closing channel handle is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = channelHandle->copyFuncs.closeIndirectChannel(
                    channelHandle->interfaceHandle.pcieStreamHandle);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not close indirect channel for the interface.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#else
    (void)channelHandle;

    err = LwSciError_NotSupported;
    goto ret;
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufFreeSourceObjIndirectChannelC2c(
    LwSciC2cBufSourceHandle sourceHandle)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    int c2cErrorCode = -1;

    LWSCI_FNENTRY("");

    if (NULL == sourceHandle) {
        LWSCI_ERR_STR("NULL sourceHandle supplied to LwSciBufFreeSourceObjIndirectChannelC2c.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LWSCIBUF_C2C_SOURCE_HANDLE_MAGIC != sourceHandle->magic) {
        LWSCI_ERR_STR("Invalid sourceHandle.");
        LwSciCommonPanic();
    }

    LwSciBufObjFree(sourceHandle->bufObj);

    /* lwsciipc_c2c_validate_c2ccopy_funcset() not supported.
    err = LwSciIpcValidateC2cCopyFuncSet(
            sourceHandle->channelHandle->ipcEndpoint,
            &sourceHandle->channelHandle->copyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not validate copy functions.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    */

    if (NULL == sourceHandle->channelHandle->copyFuncs.bufFreeSourceHandle) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for freeing source channel is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = sourceHandle->channelHandle->copyFuncs.bufFreeSourceHandle(
                    sourceHandle->interfaceSourceHandle.pcieSourceHandle);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not free source handle for the interface.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#else
    (void)sourceHandle;

    err = LwSciError_NotSupported;
    goto ret;
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufFreeTargetObjIndirectChannelC2c(
    LwSciC2cBufTargetHandle targetHandle)
{
    LwSciError err = LwSciError_Success;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    int c2cErrorCode = -1;

    LWSCI_FNENTRY("");

    if (NULL == targetHandle) {
        LWSCI_ERR_STR("NULL targetHandle supplied to LwSciBufFreeTargetObjIndirectChannelC2c.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LWSCIBUF_C2C_TARGET_HANDLE_MAGIC != targetHandle->magic) {
        LWSCI_ERR_STR("Invalid targetHandle.");
        LwSciCommonPanic();
    }

    LwSciBufObjFree(targetHandle->bufObj);

    /* lwsciipc_c2c_validate_c2ccopy_funcset() not supported.
    err = LwSciIpcValidateC2cCopyFuncSet(
            targetHandle->channelHandle->ipcEndpoint,
            &targetHandle->channelHandle->copyFuncs);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not validate copy functions.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    */

    if (NULL == targetHandle->channelHandle->copyFuncs.bufFreeTargetHandle) {
        LWSCI_ERR_STR("LwSciC2cCopyFuncs for freeing target channel is NULL.");
        LwSciCommonPanic();
    }

    c2cErrorCode = targetHandle->channelHandle->copyFuncs.bufFreeTargetHandle(
                    targetHandle->interfaceTargetHandle.pcieTargetHandle);
    if (0 != c2cErrorCode) {
        LWSCI_ERR_STR("Could not free target handle for the interface.");
        err = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

#else
    (void)targetHandle;

    err = LwSciError_NotSupported;
    goto ret;
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}
