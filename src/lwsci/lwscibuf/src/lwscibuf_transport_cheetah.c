/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/* FIXME: lwscibuf.h and LWSCIIPC_INTERNAL_H are required here so that
 * secure buffer sharing APIs from LwRM will work here. Another option
 * is to copy all the LW_INCLUDES from LwSciIpc and include them in
 * lwscibuf.lwmk. But this may cause issues later on, if lwsciipc starts
 * including more directories.
 */
#include "lwscibuf.h"
#define LWSCIIPC_INTERNAL_H
#include "lwscibuf_dev.h"
#include "lwscicommon_transportutils.h"
#include "lwscicommon_libc.h"
#include "lwscilog.h"
#include "lwscibuf_transport_priv_tegra.h"
#include "lwscibuf_obj_mgmt.h"

#define LWRM_ACC_PERM(perms) \
    (((perms) == LwSciBufAccessPerm_ReadWrite) ? LWOS_MEM_READ_WRITE : \
     ((perms) == LwSciBufAccessPerm_Readonly)? LWOS_MEM_READ : 0U)

LwSciError LwSciBufTransportGetMemHandle(
    LwSciBufObjExportPlatformDescPriv platformDesc,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm impPerm,
    LwSciBufRmHandle* hmem)
{
    LwSciError sciErr = LwSciError_Success;
    LwError lwErr;

    LWSCI_FNENTRY("");
    if (NULL == hmem) {
        LWSCI_ERR_STR("Invalid input args to LwSciBufTransportGetMemHandle.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: Importing LwRmSciIpcId %"PRIu32" on ipcEndpoint: "
               PRIu64," with permissions %d\n",
                        platformDesc.lwRmSciIpcId, ipcEndpoint, impPerm);

    /* FIXME: LwSciBuf doesn't support InterVM. Hence, fix the timeout as
     * 0 for now.
     */
    lwErr = LwRmMemHandleFromSciIpcId(platformDesc.lwRmSciIpcId, ipcEndpoint,
                        LWRM_ACC_PERM(impPerm), 0, &hmem->memHandle);
    if (LwError_Success != lwErr) {
        sciErr = LwSciError_ResourceError;
        LWSCI_ERR_INT("Failed to import memory handle of the buffer. LwError: \n", (int32_t)lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: Imported Handle %"PRIu64"\n", hmem->memHandle);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufTransportGetPlatformDesc(
    LwSciBufRmHandle hmem,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm expPerm,
    LwSciBufObjExportPlatformDescPriv* platformDesc)
{
    LwSciError sciErr = LwSciError_Success;
    LwU32 sciIpcId = 0U;
    LwError lwErr;

    LWSCI_FNENTRY("");
    if (NULL == platformDesc) {
        LWSCI_ERR_STR("Invalid input args to LwSciBufTransportGetPlatformDesc.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs handle %"PRIu64" ipcendpoint %"PRIu64" perms %d \n",
                hmem.memHandle, ipcEndpoint, expPerm);
    lwErr = LwRmMemGetSciIpcId(hmem.memHandle, ipcEndpoint,
                LWRM_ACC_PERM(expPerm), &sciIpcId);
    if (LwError_Success != lwErr) {
        sciErr = LwSciError_ResourceError;
        LWSCI_ERR_INT("Failed to Get LwSciIpc Id for the buffer. LwError: \n", (int32_t)lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    platformDesc->lwRmSciIpcId = sciIpcId;
    LWSCI_INFO("Output: Exported sciIpcId %lu\n", platformDesc->lwRmSciIpcId);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufTransportCreateObjFromMemHandle(
    const LwSciBufRmHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrList reconciledAttrList,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    bool isRemoteObject,
    LwSciC2cCopyFuncs copyFuncs,
    LwSciC2cInterfaceTargetHandle c2cTargetHandle,
#endif
    LwSciBufObj* bufObj)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((0U == len) || (NULL == reconciledAttrList) || (NULL == bufObj)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufTransportCreateObjFromMemHandle.");
        LWSCI_ERR_ULONG("offset: ,", offset);
        LWSCI_ERR_ULONG("len: \n", len);
        LwSciCommonPanic();
    }

    // On CheetAh, we do not need to dupe the handle because if we did, we would
    // leak LwMap handles in cleanup.
    err = LwSciBufObjCreateFromMemHandlePriv(memHandle, offset, len,
            reconciledAttrList, false,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
            isRemoteObject, false, copyFuncs, c2cTargetHandle,
#endif
            bufObj);

    LWSCI_FNEXIT("");
    return err;
}

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
LwSciError LwSciBufTransportSetC2cRmHandle(
    LwSciBufRmHandle bufRmHandle,
    LwSciC2cPcieBufRmHandle* pcieBufRmHandle)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((0U == bufRmHandle.memHandle)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufTransportSetC2cRmHandle.");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (NULL == pcieBufRmHandle) {
        LWSCI_ERR_STR("NULL pcieBufRmHandle supplied.");
        LwSciCommonPanic();
    }

    pcieBufRmHandle->memHandle = bufRmHandle.memHandle;

ret:
    LWSCI_FNEXIT("");
    return err;
}
#endif
