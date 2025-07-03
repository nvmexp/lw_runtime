/*
 * lwscibuf_transport_priv_tegra.h
 *
 * Transport Layer header file for LwSciBuf
 *
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_TRANSPORT_TEGRA_H
#define INCLUDED_LWSCIBUF_ATTR_TRANSPORT_TEGRA_H

#include "lwscibuf_internal.h"
#include "lwscibuf_obj_mgmt.h"

/**
 * @defgroup lwscibuf_blanket_statements LwSciBuf blanket statements.
 * Generic statements applicable for LwSciBuf interfaces.
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements LwSciBuf blanket statements
 * \section lwscibuf_element_dependency Dependency on other elements
 *  LwSciBuf calls the following liblwrm_mem interfaces:
 *   - LwRmMemHandleFromSciIpcId() to get LwRmMemHandle.
 *   - LwRmMemGetSciIpcId() to get sciIpcId from LwRmMemHandle.
 *
 * \implements{18842583}
 */

 /**
  * @}
  */

/**
 * CheetAh specific export descriptor.
 * While exporting LwSciBufObj, LwSciBufRmHandle corresponding to it is
 * colwerted to export descriptor.
 * While importing LwSciBufObj, export descriptor is colwerted to
 * LwSciBufRmHandle and LwSciBufObj is created from it.
 *
 * Synchronization: Updates to an instance of this datatype must be
 * externally synchronized
 *
 * \implements{18842241}
 */
typedef struct {
/**
 * sciIpcId provided by liblwrm_memmgr corresponding to given LwSciBufRmHandle,
 * LwSciIpcId and LwSciBufAttrValAccessPerm.
 * This member is initialized by calling liblwrm_mem API when LwSciBufObj is
 * exported. This member along with LwSciIpcId and LwSciBufAttrValAccessPerm
 * are colwerted into LwSciBufRmHandle  using liblwrm_mem API when LwSciBufObj
 * is imported.
 */
    LwU32 lwRmSciIpcId;
} LwSciBufObjExportPlatformDescPriv;

/**
 * @brief Colwerts LwSciBufObjExportPlatformDescPriv into LwSciBufRmHandle by
 * calling liblwrm_mem API.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @param[in] platformDesc: LwSciBufObjExportPlatformDescPriv.
 *            The parameter is valid if it is obtained from successful call to
 *            LwSciBufTransportGetPlatformDesc.
 * @param[in] ipcEndpoint: LwSciIpcEndPoint of the peer.
 * @param[in] impPerm: LwSciBufAttrValAccessPerm. It is valid if input
 *            LwSciBufAttrValAccessPerm <= LwSciBufAttrValAccessPerm with which
 *            LwSciBufObjExportPlatformDescPriv was exported by calling
 *            LwSciBufTransportGetPlatformDesc() during export.
 * @param[out] hmem: LwSciBufRmHandle.
 *
 * @return ::LwSciError, the completion status of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if hmem is NULL
 * - ::LwSciError_ResourceError if LwRmMemHandleFromSciIpcId call not
 *     successful.
 *
 * \implements{18843018}
 */
LwSciError LwSciBufTransportGetMemHandle(
    LwSciBufObjExportPlatformDescPriv platformDesc,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm impPerm,
    LwSciBufRmHandle* hmem);

/**
 * @brief Colwerts LwSciBufRmHandle into LwSciBufObjExportPlatformDescPriv by
 * calling liblwrm_mem API.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the underlying buffer handle is provided via
 *        LwRmMemGetSciIpcId()
 *
 * @param[in] hmem: LwSciBufRmHandle.The parameter is valid if it is obtained
 *            from a successful call to LwRmMemHandleAllocAttr() or it is
 *            obtained from successful call to LwSciBufTransportGetMemHandle()
 *            and has not been deallocated by using LwRmMemHandleFree().
 * @param[in] ipcEndpoint: LwSciIpcEndpoint of the peer.
 * @param[in] expPerm: LwSciBufAttrValAccessPerm. The valid value range is
 *            @a expPerm <= permission of LwRmMemHandle associated with @a hmem.
 * @param[out] platformDesc: LwSciBufObjExportPlatformDescPriv.
 *
 * @return ::LwSciError, the completion status of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if platformDesc is NULL
 * - ::LwSciError_ResourceError if LwRmMemGetSciIpcId call not successful.
 *
 * \implements{18843021}
 */
LwSciError LwSciBufTransportGetPlatformDesc(
    LwSciBufRmHandle hmem,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm expPerm,
    LwSciBufObjExportPlatformDescPriv* platformDesc);

/**
 * @brief Creates LwSciBufObj from LwSciBufRmHandle.
 *
 * This API just calls LwSciBufObjCreateFromMemHandlePriv(), passing
 * false value colweying the function not to duplicate the buffer
 * handle. We do not duplicate the buffer handle when creating
 * LwSciBufObj because buffer handle is already owned by LwSciBuf
 * during importing.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufAttrList is provided via
 *        LwSciBufObjCreateFromMemHandlePriv()
 *      - Conlwrrent access to the underlying buffer handle is provided via
 *        LwSciBufObjCreateFromMemHandlePriv()
 *
 * @param[in] memHandle: buffer handle representing the buffer containing the
 *            buffer to be represented by the new LwSciBufObj. The parameter is
 *            valid if it is obtained from a successful call to
 *            LwRmMemHandleAllocAttr() or it is obtained from successful call
 *            to LwSciBufTransportGetMemHandle() and has not been deallocated
 *            by using LwRmMemHandleFree().
 * @param[in] offset: The offset within the buffer represented by memHandle
 *            to be represented by the new LwSciBufObj. Valid value: 0 to size
 *            of the buffer represented by LwSciBufRmHandle - 1.
 * @param[in] len: The length of the memory to be represented by the new
 *            LwSciBufObj. The size of the buffer represented by memHandle must
 *            be at least offset + len. Valid value: 1 to size of the buffer
 *            represented by LwSciBufRmHandle - offset.
 * @param[in] reconciledAttrList: A reconciled attribute list specifying the
 *            attributes to assign to the new LwSciBufObj.
 * @param[in] isRemoteObject boolean flag indicating whether LwSciBufObj being
 *            created is remote or local. True implies that LwSciBufObj is
 *            remote (meaning it is imported from the remote peer for which
 *            there is no backing LwSciBufRmHandle. This can is set to true
 *            only in C2c case when LwSciBufObj allocated by remote Soc peer
 *            is imported), false implies otherwise.
 * @param[in] copyFuncs LwSciC2cCopyFuncs.
 * @param[in] c2cTargetHandle LwSciC2cInterfaceTargetHandle.
 * @param[out] bufObj: the new LwSciBufObj.
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a memHandle is invalid.
 *    - @a reconciledAttrList is unreconciled.
 *    - @a offset + @a len > buffer size represented by LwSciBufRmHandle.
 *    - @a len > buffer size represented by output attributes for respective
 *      LwSciBufType in @a reconciledAttrList.
 *    - buffer size represented by output attributes for respective
 *      LwSciBufType in @a reconciledAttrList > buffer size represented by
 *      LwSciBufRmHandle.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_Overflow if @a len + @a offset exceeds UINT64_MAX.
 * - LwSciError_ResourceError if any of the following oclwrs:
 *    - LWPU driver stack failed.
 *    - system lacks resource other than memory
 * - LwSciError_IlwalidState if new LwSciBufAttrList cannot be associated with
 *   the LwSciBufModule associated with the given LwSciBufAttrList to create a
 *   new LwSciBufObj
 * - LwSciError_NotSupported if LwSciBufPrivateAttrKey_MemDomain on the
 *   LwSciBufAttrList is not supported.
 * - Panics if:
 *    - @a bufObj is NULL.
 *    - @a reconciledAttrList is NULL.
 *    - @a reconciledAttrList is invalid.
 *    - @a len is 0.
 *
 * \implements{18843024}
 */
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
    LwSciBufObj* bufObj);

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
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
#endif

#endif /* INCLUDED_LWSCIBUF_ATTR_TRANSPORT_TEGRA_H */
