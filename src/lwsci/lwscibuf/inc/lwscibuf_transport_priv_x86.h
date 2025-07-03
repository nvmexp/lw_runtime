/*
 * lwscibuf_transport_priv_x86.h
 *
 * Transport Layer header file for LwSciBuf
 *
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_TRANSPORT_X86_H
#define INCLUDED_LWSCIBUF_ATTR_TRANSPORT_X86_H

#include "lwscibuf_internal.h"
#include "lwscibuf_obj_mgmt.h"
/*
 * LwSciBuf Object export descriptor platform specific structure
 */
typedef struct {
    LwU32 hClient;
    LwU32 hDevice;
    LwU32 hMemory;
} LwSciBufObjExportPlatformDescPriv;

LwSciError LwSciBufTransportGetMemHandle(
    LwSciBufObjExportPlatformDescPriv platformDesc,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm impPerm,
    LwSciBufRmHandle* hmem);

LwSciError LwSciBufTransportGetPlatformDesc(
    LwSciBufRmHandle hmem,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm expPerm,
    LwSciBufObjExportPlatformDescPriv* platformDesc);

/**
 * Note: The handle lifecycle will be different depending on the platform.
 *
 * @param[in] memHandle is the RM memory handle structure representing the
 * memory containing the memory to be represented by the new LwSciBufObj.
 * @param[in] offset The offset within the memory represented by memHandle
 * of the memory to be represented by the new LwSciBufObj.
 * @param[in] len The length of the memory to be represented by the new
 * LwSciBufObj.
 * @param[in] reconciledAttrList A reconciled attribute list specifying the
 * attributes to assign to the new LwSciBufObj. The same validation of the
 * LwSciBufObj against attrList is done here as in LwSciBufObjIpcImport().
 * @param[in] isRemoteObject boolean flag indicating whether LwSciBufObj being
 *            created is remote or local. True implies that LwSciBufObj is
 *            remote (meaning it is imported from the remote peer for which
 *            there is no backing LwSciBufRmHandle. This can is set to true
 *            only in C2c case when LwSciBufObj allocated by remote Soc peer
 *            is imported), false implies otherwise.
 * @param[in] copyFuncs LwSciC2cCopyFuncs.
 * @param[in] c2cTargetHandle LwSciC2cInterfaceTargetHandle.
 * @param[out] bufObj On success, *bufObj is set to the new LwSciBufObj.
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if bufObj is NULL or reconciledAttrList is NULL
 */
LwSciError LwSciBufTransportCreateObjFromMemHandle(
    const LwSciBufRmHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrList reconciledAttrList,
    bool isRemoteObject,
    LwSciC2cCopyFuncs copyFuncs,
    LwSciC2cInterfaceTargetHandle c2cTargetHandle,
    LwSciBufObj* bufObj);

/**
 * \brief Sets LwSciC2cPcieBufRmHandle corresponding to LwSciBufRmHandle.
 *
 * \param[in] bufRmHandle LwSciBufRmHandle to be colwerted to
 * LwSciC2cPcieBufRmHandle.
 * \param[out] pcieBufRmHandle LwSciC2cPcieBufRmHandle colwerted from
 * LwSciBufRmHandle.
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if @a bufRmHandle is invalid.
 * - Panics if @a pcieBufRamHandle is NULL.
 */
LwSciError LwSciBufTransportSetC2cRmHandle(
    LwSciBufRmHandle bufRmHandle,
    LwSciC2cPcieBufRmHandle* pcieBufRmHandle);

#endif /* INCLUDED_LWSCIBUF_ATTR_TRANSPORT_X86_H */
