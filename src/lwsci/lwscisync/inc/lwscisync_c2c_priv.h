/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync common C2C definitions</b>
 *
 * @b Description: This file declares C2C related types
 */

#ifndef INCLUDED_LWSCISYNC_C2C_PRIV_H
#define INCLUDED_LWSCISYNC_C2C_PRIV_H

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)

#include "lwsciipc_internal.h"
#include "lwscic2c_pcie_stream.h"

/**
 * C2C specific handle for synchronization objects
 */
typedef union {
    /** PCIe variant of the handle */
    LwSciC2cPcieSyncHandle pcieSyncHandle;
} LwSciC2cInterfaceSyncHandle;

/**
 * @brief typedef copy functions provided by LwSciC2cPCIe into generic type.
 */
typedef LwSciC2cPcieCopyFuncs LwSciC2cCopyFuncs;

/**
 * Calls LwSciSyncHwEngCreateIdWithoutInstance() to create HwEng instance
 * for C2CPCIe. Prepares attributes to be set:
 * - _SignalerPrimitiveInfo and _WaiterPrimitiveInfo to syncpoint
 * - _SignalerPrimitiveCount to 1
 * - _EngineArray to the HwEng instance above
 * - _RequiredPerm to permissions.
 * Sets them in the attribute lists using LwSciSyncAttrListSetAttrs()
 * and LwSciSyncAttrListSetInternalAttrs().
 *
 * \fn LwSciError LwSciSyncFillAttrsIndirectChannelC2c(
 * LwSciIpcEndpoint ipcEndpoint,
 * LwSciSyncAttrList unrecAttrList,
 * LwSciSyncAccessPerm permissions);
 */

/**
 * NULL checks the parameters. Checks magicID of channelHandle. Retrieves
 * the _ActualPerm from the @a syncObj using LwSciSyncObjGetAttrList()
 * and LwSciSyncAttrListGetAttr() and verifies it contains waiting.
 * Also retrieves _EngineArray using LwSciSyncAttrListGetSingleInternalAttr()
 * and checks that it contains exactly one engine and verifies using
 * LwSciSyncHwEngGetNameFromId() that it is _PCIe.
 *
 * Allocates a new syncHandle with LwSciCommonCalloc(). Tries retrieving
 * a LwSciC2cPcieSyncHandle from @a syncObj using LwSciSyncObjGetC2cPcieSyncHandle().
 *
 * If the LwSciError_NotInitialized is returned, it means that this is
 * an engineWritesDoneObj. The function creates the LwSciC2cPcieSyncHandle
 * with copyFuncs.syncCreateLocalHandle(). Registers it with
 * copyFuncs.syncRegisterLocalHandle().
 *
 * If LwSciC2cPcieSyncHandle was retrieved successfully, it means that this is
 * a consReadsDoneProdObj. The function duplicates the handle with
 * copyFuncs.syncDupRemoteHandle().
 * Registers it with copyFuncs.syncRegisterRemoteHandle().
 *
 * In both cases, the function then takes a reference on @a syncObj with
 * LwSciSyncObjRef. It then sets the fields of the output to appropriate values.
 *
 *
 * \fn LwSciError LwSciSyncRegisterWaitObjIndirectChannelC2c(
 * LwSciC2cHandle channelHandle,
 * LwSciSyncObj syncObj,
 * LwSciC2cSyncHandle* syncHandle);
 */

/**
 * NULL checks the parameters. Checks magicID of channelHandle. Retrieves
 * the _ActualPerm from the @a syncObj using LwSciSyncObjGetAttrList()
 * and LwSciSyncAttrListGetAttr() and verifies it contains signaling.
 * Also retrieves _EngineArray using LwSciSyncAttrListGetSingleInternalAttr()
 * and checks that it contains exactly one engine and verifies using
 * LwSciSyncHwEngGetNameFromId() that it is _PCIe.
 *
 * Allocates a new syncHandle with LwSciCommonCalloc(). Tries retrieving
 * a LwSciC2cPcieSyncHandle from @a syncObj using
 * LwSciSyncObjGetC2cPcieSyncHandle().
 *
 * If the LwSciError_NotInitialized is returned, it means that this is
 * a copyDoneProdObj. The function retrieves a LwRmHost1xSyncpointHandle
 * form the  @a syncObj using LwSciSyncCoreObjGetC2cRmHandle(). Creates
 * a LwSciC2cPcieSyncHandle from it with copyFuncs.syncMapLocalMemHandle()
 * and registers it with channelHandle->copyFuncs.syncRegisterLocalHandle().
 *
 * If LwSciC2cPcieSyncHandle was retrieved successfully, it means that this is
 * a copyDoneConsObj. The function duplicates the handle with
 * copyFuncs.syncDupRemoteHandle().
 * Registers it with copyFuncs.syncRegisterRemoteHandle().
 *
 * In both cases, the function then takes a reference on @a syncObj with
 * LwSciSyncObjRef. It then sets the fields of the output to appropriate values.
 *
 * \fn LwSciError LwSciSyncRegisterSignalObjIndirectChannelC2c(
 * LwSciC2cHandle channelHandle,
 * LwSciSyncObj syncObj,
 * LwSciC2cSyncHandle* syncHandle);
 */

/**
 * NULL checks the parameters. Checks the magicID of the @a channelHandle
 * and @a syncHandle. Verifies that @a syncHandle is associated with
 * the @a channelHandle and checks that it has waiting permissions.
 *
 * Tries retrieving the syncObj from @a preFence with
 * LwSciSyncFenceGetSyncObj().
 * If it fails with LwSciError_ClearedFence, it means that the @a preFence
 * is cleared and the function returns.
 *
 * Otherwise it checks that the syncObj is the same that the syncHandle is
 * associated with. Extracts id and value using LwSciSyncFenceExtractFence().
 * Adds the wait to the submission with
 * channelHandle->copyFuncs.pushWaitIndirectChannel().
 *
 * \fn LwSciError LwSciBufPushWaitIndirectChannelC2c(
 * LwSciC2cHandle channelHandle,
 * LwSciC2cSyncHandle syncHandle,
 * const LwSciSyncFence* preFence);
 */

/**
 * NULL checks the parameters. Checks the magicID of the @a channelHandle
 * and @a syncHandle. Verifies that @a syncHandle is associated with
 * the @a channelHandle and checks that it has signaling permissions.
 *
 * Generates the postFence using LwSciSyncObjGenerateFence() on the
 * @a syncHandle's syncObj. Adds the signaling command to the submission with
 * copyFuncs.pushSignalIndirectChannel().
 *
 * \fn LwSciError LwSciBufPushSignalIndirectChannelC2c(
 * LwSciC2cHandle channelHandle,
 * LwSciC2cSyncHandle syncHandle,
 * LwSciSyncFence* postFence);
 */

/**
 * NULL checks the parameter and verifies the magicID.
 *
 * Frees the @a syncHandle's LwSciC2cPcieSyncHandle with
 * copyFuncs.syncFreeHandle(). Releases the reference @a syncHandle has
 * on its syncObj with LwSciSyncObjFree() and finally releases the memory
 * with LwSciCommonFree().
 *
 * \fn LwSciError LwSciSyncFreeObjIndirectChannelC2c(
 * LwSciC2cSyncHandle syncHandle);
 */

#endif /* (LW_IS_SAFETY == 0) && (LW_L4T == 0) */
#endif
