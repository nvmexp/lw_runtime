/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_C2C_INTERNAL_H
#define INCLUDED_LWSCIBUF_C2C_INTERNAL_H

#include "lwsciipc.h"
#include "lwscibuf_internal.h"
#include "lwscisync_internal.h"
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
#include "lwscic2c_pcie_stream.h"
#endif

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * \brief Magic ID identifying the LwScic2cHandle.
 */
#define LWSCIBUF_C2C_CHANNEL_MAGIC (0xFACE0011U)

/**
 * \brief A reference to LwSciC2cHandleRec.
 */
typedef struct LwSciC2cHandleRec* LwSciC2cHandle;

/**
 * \brief A top level container for source buffer in C2c case.
 */
typedef struct LwSciC2cBufSourceHandleRec* LwSciC2cBufSourceHandle;

/**
 * \brief A top level container for target buffer in C2c case.
 */
typedef struct LwSciC2cBufTargetHandleRec* LwSciC2cBufTargetHandle;

/**
 * \brief A top level container for C2c synchronization.
 */
typedef struct LwSciC2cSyncHandleRec* LwSciC2cSyncHandle;

/**
 * \brief Structure defining flush range for the buffer in C2c case.
 */
typedef struct {
    /**
     * Offset of the buffer to be flushed.
     */
    uint64_t offset;
    /**
     * Size of the buffer to be flushed.
     */
    uint64_t size;
} LwSciBufFlushRanges;

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * @brief typedef copy functions provided by LwSciC2cPCIe into generic type.
 */
typedef LwSciC2cPcieCopyFuncs LwSciC2cCopyFuncs;

/**
 * Union abstracting the LwSciC2c handle.
 */
typedef union {
    /**
     * LwSciC2cPcieStreamHandle
     */
    LwSciC2cPcieStreamHandle pcieStreamHandle;
    /* TODO: Either add NPM handle or replace this with abstracted handle if
     * LwSciC2c team agrees to provide it.
     */
} LwSciC2cInterfaceHandle;

/**
 * Actual container referenced by LwSciC2cHandle.
 */
typedef struct LwSciC2cHandleRec {
    /**
     * Magic ID for the sanity check.
     */
    uint32_t magic;

    /**
     * LwSciC2cPlatformHandle
     */
    LwSciC2cInterfaceHandle interfaceHandle;

    /**
     * Set of C2c copy functions obtained from LwSciIpc.
     */
    LwSciC2cCopyFuncs copyFuncs;

    /* LwSciIpcEndpoint */
    LwSciIpcEndpoint ipcEndpoint;
} LwSciC2cHandlePriv;
#endif

/**
 * \brief Opens indirect C2c channel for a given LwSciIpcEndpoint for C2c
 * transfer.
 *
 * \param[in] ipcEndpoint LwSciIpcEndpoint
 * \param[in] eventService LwSciEventService obtained by successfully calling
 *            LwSciEventLoopServiceCreate().
 * \param[in] numRequests Maximum number of requests that can be submitted for
 * copy.
 * \param[in] numFlushRanges Maximum number of buffer flush ranges that can
 * be requested per push call.
 * \param[in] numPreFences Number of pre-fences per submission.
 * \param[in] numPostFences Number of post-fences per submission.
 * \param[out] channelHandle LwSciC2cHandle obtained from successfully opening
 * the channel.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a ipcEndpoint is invalid.
 *      - @a eventService is NULL.
 *      - @a numRequests is zero.
 *      - @a numFlushRanges is zero.
 *      - @a numPreFences is zero.
 *      - @a numPostFences is zero.
 *      - @a channelHandle is NULL.
 * - ::LwSciError_InsufficientMemory if there is no system memory for heap
 * allocation.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufOpenIndirectChannelC2c(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciEventService* eventService,
    size_t numRequests,
    size_t numFlushRanges,
    size_t numPreFences,
    size_t numPostFences,
    LwSciC2cHandle* channelHandle);

/**
 * \brief Registers @a bufObj as source object with @a channelHandle.
 *
 * \param[in] channelHandle LwSciC2cHandle
 * \param[in] bufObj LwSciBufObj to be registered with @a channelHandle.
 * \param[out] sourceHandle LwSciC2cBufSourceHandle registered to
 * @a channelHandle.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a channelHandle is NULL.
 *      - @a bufObj is NULL.
 *      - @a sourceHandle is NULL.
 * - ::LwSciError_InsufficientMemory if there is no system memory for heap
 * allocation.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 * - Panics if any of the following oclwrs:
 *      - @a bufObj is invalid.
 *      - @a channelHandle is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufRegisterSourceObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciBufObj bufObj,
    LwSciC2cBufSourceHandle* sourceHandle);

/**
 * \brief Registers @a bufObj as target object with @a channelHandle.
 *
 * \param[in] channelHandle LwSciC2cHandle
 * \param[in] bufObj LwSciBufObj to be registered with @a channelHandle.
 * \param[out] targetHandle LwSciC2cBufTargetHandle registered to
 * @a channelHandle.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a channelHandle is NULL.
 *      - @a bufObj is NULL.
 *      - @a targetHandle is NULL.
 * - ::LwSciError_InsufficientMemory if there is no system memory for heap
 * allocation.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 * - Panics if any of the following oclwrs:
 *      - @a bufObj is invalid.
 *      - @a channelHandle is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufRegisterTargetObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciBufObj bufObj,
    LwSciC2cBufTargetHandle* targetHandle);

/**
 * \brief Push (queue) the copy command for copying the data from
 * @a sourceHandle to @a targetHandle.
 *
 * \param[in] channelHandle LwSciC2cHandle
 * \param[in] sourceHandle LwSciC2cBufSourceHandle from which copying needs to
 * done.
 * \param[in] targetHandle LwSciC2cBufTargetHandle to which copying needs to
 * be done.
 * \param[in] flushRanges LwSciBufFlushRanges for the buffers to be copied from
 * @a sourceHandle to @a targetHandle.
 * \param[in] numFlushRanges number of flush ranges.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a channelHandle is NULL.
 *      - @a sourceHandle is NULL.
 *      - @a targetHandle is NULL.
 *      - @a flushRanges is NULL.
 *      - @a numFlushRanges is zero.
 * - ::LwSciError_InsufficientMemory if there is no system memory for heap
 * allocation.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 * - Panics if any of the following oclwrs:
 *      - @a channelHandle is invalid.
 *      - @a sourceHandle is invalid.
 *      - @a targetHandle is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciBufPushCopyIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cBufSourceHandle sourceHandle,
    LwSciC2cBufTargetHandle targetHandle,
    const LwSciBufFlushRanges* flushRanges,
    size_t numFlushRanges);

/**
 * \brief Submits the queued copy requests.
 *
 * \param[in] @a channelHandle LwSciC2cHandle
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a channelHandle is NULL.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 * - Panics if @a channelHandle is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciBufPushSubmitIndirectChannelC2c(
    LwSciC2cHandle channelHandle);

/**
 * \brief Closes @a channelHandle.
 *
 * \param[in] channelHandle LwSciC2cHandle to be closed.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a channelHandle is NULL.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 * - Panics if @a channelHandle is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufCloseIndirectChannelC2c(
    LwSciC2cHandle channelHandle);

/**
 * \brief closes @a sourceHandle
 *
 * \param[in] sourceHandle LwSciC2cBufSourceHandle
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a sourceHandle is NULL.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 * - Panics if @a sourceHandle is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufFreeSourceObjIndirectChannelC2c(
    LwSciC2cBufSourceHandle sourceHandle);

/**
 * \brief closes @a targetHandle
 *
 * \param[in] targetHandle LwSciC2cBufTargetHandle
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a targetHandle is NULL.
 * - ::LwSciError_ResourceError if LWPU driver stack failed.
 * - Panics if @a targetHandle is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufFreeTargetObjIndirectChannelC2c(
    LwSciC2cBufTargetHandle targetHandle);

/**
 * \brief Fills appropriate attributes for C2C copy related
 *     to the input ipcEndpoint
 *
 * \param[in] ipcEndpoint a C2C IPC endpoint through which
 * the C2C communication will happen
 * \param[in] unrecAttrList attribute list to be filled
 * \param[in] permissions Permissions to be set in the attribute list
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *           - @a unrecAttrList is NULL,
 *           - @a unrecAttrList is not unreconciled and/or not writable,
 *           - any of LwSciSyncAttrKey_RequiredPerm,
 *             LwSciSyncInternalAttrKey_EngineArray,
 *             LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
 *             LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
 *             LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
 *             is already set in @a unrecAttrList.
 *           - @a ipcEndpoint is invalid
 * - Panics if @a unrecAttrList is not valid.
 */
LwSciError LwSciSyncFillAttrsIndirectChannelC2c(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList unrecAttrList,
    LwSciSyncAccessPerm permissions);

/**
 * \brief Creates a new C2C syncHandle associated with the provided
 *  syncObj and channelHandle to be used for inserting prefences
 *  in C2C submissions.
 *
 * \param[in] channelHandle C2C channel handle for whose submissions
 *            the resulting syncHandle is intended
 * \param[in] syncObj LwSciSyncObj associated with the new syncHandle
 * \param[out] syncHandle the resulting syncHandle
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *           - any parameter is NULL,
 *           - @a syncObj has no waiting permissions
 *           - @a syncObj has a different EngineArray than the one
 *             consisting of only LwSciSyncHwEngName_PCIe
 * - ::LwSciError_InsufficientMemory if there is no memory to allocate
 *   the necessary objects
 * - ::LwSciError_ResourceError if encountered any errors coming
 *   from C2C library
 * - Panics if @a channelHandle or @a syncObj are not valid.
 */
LwSciError LwSciSyncRegisterWaitObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciSyncObj syncObj,
    LwSciC2cSyncHandle* syncHandle);

/**
 * \brief Creates a new C2C syncHandle associated with the provided
 *  syncObj and channelHandle to be used for inserting signal commands
 *  in C2C submissions
 *
 * \param[in] channelHandle C2C channel handle for whose submissions
 *            the resulting syncHandle is intended
 * \param[in] syncObj LwSciSyncObj associated with the new syncHandle
 * \param[out] syncHandle the resulting syncHandle
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *           - any parameter is NULL,
 *           - @a syncObj has no signaling permissions
 *           - @a syncObj has a different EngineArray than the one
 *             consisting of only LwSciSyncHwEngName_PCIe
 * - ::LwSciError_InsufficientMemory if there is no memory to allocate
 *   the necessary objects
 * - ::LwSciError_ResourceError if encountered any errors coming
 *   from C2C library
 * - Panics if @a channelHandle or @a syncObj are not valid.
 */
LwSciError LwSciSyncRegisterSignalObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciSyncObj syncObj,
    LwSciC2cSyncHandle* syncHandle);

/**
 * \brief Inserts a prefence waiting command to the current submission
 * of channelHandle. After submitting, the channel will wait for the preFence
 * to expire before starting the C2C copying operation.
 * If the preFence is cleared, this operation has no effect.
 *
 * \param[in] channelHandle C2C channel handle
 * \param[in] syncHandle previously registered waiting syncHandle
 * \param[in] preFence fence the C2C channel will wait on
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *           - any parameter is NULL,
 *           - @a syncHandle is not associated with @a channelHandle
 *           - @a preFence is not associated with the syncObj
 *             used for creation of @a syncHandle
 * - ::LwSciError_ResourceError if encountered any errors coming
 *   from C2C library
 * - Panics if @a channelHandle or @a syncHandle or LwSciSyncObj associated
 *   with @a preFence are not valid.
 */
LwSciError LwSciBufPushWaitIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cSyncHandle syncHandle,
    const LwSciSyncFence* preFence);

/**
 * \brief Inserts a signaling command to the current submission
 * of channelHandle and generates a postFence. After performing the C2C copy,
 * the channel will signal resulting in expiration of postFence.
 *
 * \param[in] channelHandle C2C channel handle
 * \param[in] syncHandle previously registered signaling syncHandle
 * \param[out] postFence fence corresponding to the event of finishing C2C copy
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *           - any parameter is NULL,
 *           - @a syncHandle is not associated with @a channelHandle
 *           - syncObj associated with @a syncHandle is not a C2C signaler
 * - ::LwSciError_ResourceError if encountered any errors coming
 *   from C2C library
 * - Panics if @a channelHandle or @a syncHandle or syncObj associated
 *   with @a syncHandle are not valid.
 */
LwSciError LwSciBufPushSignalIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cSyncHandle syncHandle,
    LwSciSyncFence* postFence);

/**
 * \brief Frees a syncHandle.
 *
 * \param[in] syncHandle previously registered signaling syncHandle
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *           - @a syncHandle is NULL,
 * - ::LwSciError_ResourceError if encountered any errors coming
 *   from C2C library
 * - Panics if @a syncHandle is not valid.
 */
LwSciError LwSciSyncFreeObjIndirectChannelC2c(
    LwSciC2cSyncHandle syncHandle);

#if defined(__cplusplus)
}
#endif // __cplusplus

#endif /* INCLUDED_LWSCIBUF_C2C_INTERNAL_H */
