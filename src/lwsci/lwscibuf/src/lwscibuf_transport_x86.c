/*
 * Copyright (c) 2019-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_dev.h"
#include "lwscicommon_transportutils.h"
#include "lwscicommon_libc.h"
#include "lwscilog.h"
#include "lwscibuf_transport_priv_x86.h"
#include "lwscibuf_obj_mgmt.h"

LwSciError LwSciBufTransportGetMemHandle(
    LwSciBufObjExportPlatformDescPriv platformDesc,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm impPerm,
    LwSciBufRmHandle* hmem)
{
    LwSciError sciErr = LwSciError_Success;

    (void)ipcEndpoint;
    (void)impPerm;

    LWSCI_FNENTRY("");
    if (hmem == NULL) {
        LWSCI_ERR("Invalid input args to LwSciBufTransportGetPlatformDesc.\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    hmem->hClient = platformDesc.hClient;
    hmem->hDevice = platformDesc.hDevice;
    hmem->hMemory = platformDesc.hMemory;

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

    (void)ipcEndpoint;
    (void)expPerm;

    LWSCI_FNENTRY("");
    if (platformDesc == NULL) {
        LWSCI_ERR("Invalid input args to LwSciBufTransportGetPlatformDesc.\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    platformDesc->hClient = hmem.hClient;
    platformDesc->hDevice = hmem.hDevice;
    platformDesc->hMemory = hmem.hMemory;

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufTransportCreateObjFromMemHandle(
    const LwSciBufRmHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrList reconciledAttrList,
    bool isRemoteObject,
    LwSciC2cCopyFuncs copyFuncs,
    LwSciC2cInterfaceTargetHandle c2cTargetHandle,
    LwSciBufObj* bufObj)
{
    LwSciError err = LwSciError_Success;
    bool dupHandle = false;

    LWSCI_FNENTRY("");

    /* If 'isRemoteObject' is true, it means there is no backing
     * LwSciBufRmHandle. In such case, set dupHandle to false. Set true,
     * otherwise.
     */
    dupHandle = !(isRemoteObject);

    err = LwSciBufObjCreateFromMemHandlePriv(memHandle, offset, len,
            reconciledAttrList, dupHandle, isRemoteObject, false, copyFuncs,
            c2cTargetHandle, bufObj);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufObjCreateFromMemHandlePriv faile.");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufTransportSetC2cRmHandle(
    LwSciBufRmHandle bufRmHandle,
    LwSciC2cPcieBufRmHandle* pcieBufRmHandle)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* LwSciC2c does not implement x86 interfaces yet. */
    (void)bufRmHandle;
    (void)pcieBufRmHandle;

    LWSCI_FNEXIT("");
    return err;
}
