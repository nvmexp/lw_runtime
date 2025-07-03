/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync Internal Interface</b>
 *
 * @b Description: This file contains LwSciSync apis exposed to UMDs
 */

#ifndef INCLUDED_LWSCISYNC_INTERNAL_H
#define INCLUDED_LWSCISYNC_INTERNAL_H

/**
 * @defgroup lwsci_sync Synchronization APIs
 *
 * @ingroup lwsci_group_stream
 * @{
 */

#include <lwscisync.h>
#include <lwscierror.h>
#include <lwscibuf.h>
#ifdef LWSCISYNC_EMU_SUPPORT
#include <lwscibuf_internal.h>
#endif

/**
 * \page lwscisync_page_blanket_statements LwSciSync blanket statements
 * \section lwscisync_fence_states Fence states
 * - LwSciSyncFence becomes not cleared if it is modified by a successful
 * LwSciSyncFenceUpdateFence().
 */

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * \brief Types of synchronization primitives.
 *
 * \implements{18840204}
 */
enum LwSciSyncInternalAttrValPrimitiveTypeRec {
    /** For LwSciSync internal use only */
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
    /** Syncpoint */
    LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
    /**
     * 16 bytes semaphore backed by system memory.
     * Contains space for 8-byte timestamp and 4-byte payload..
     */
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
    /**
     * 16 bytes semaphore backed by video memory.
     * Contains space for 8-byte timestamp and 4-byte payload..
     * If used, the LwSciSyncInternalAttrKey_GpuId key must be set as well.
     */
    LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore,
    /**
     * 16 bytes semaphore backed by system memory.
     * Contains space for 8-byte timestamp and 8-byte payload..
     */
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
    /**
     * 16 bytes semaphore backed by video memory.
     * Contains space for 8-byte timestamp and 8-byte payload..
     * If used, the LwSciSyncInternalAttrKey_GpuId key must be set as well.
     */
    LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphorePayload64b,
    /** For LwSciSync internal use only */
    LwSciSyncInternalAttrValPrimitiveType_UpperBound,
};

/**
 * \brief Engines allowed to interact with the LwSciSyncObj.
 *
 * These engines and their UMDs require special treatment and LwSciSync needs
 * to be informed if they are going to be used.
 *
 * \implements{22823488}
 */
typedef enum {
    LwSciSyncHwEngName_LowerBound = 0,
    /** C2cPCIe engine */
    LwSciSyncHwEngName_PCIe = 123,
    LwSciSyncHwEngName_UpperBound = 124,
} LwSciSyncHwEngName;

/**
 * \brief Enum to identify hardware engines from CheetAh or resman world
 *
 * \implements{22823490}
 */
typedef enum {
    LwSciSyncHwEngine_TegraNamespaceId,
    LwSciSyncHwEngine_ResmanNamespaceId,
} LwSciSyncHwEngNamespace;

/**
 * \brief Union specifying revision of the hardware engine accessing LwSciSync.
 *
 * \implements{22823493}
 */
typedef union {
    /** Revision of the non-GPU hardware engine. */
    int32_t engine_rev;

    /** Revision of the GPU hardware engine */
    struct {
        /** GPU HW architecture. It should be initialized as 0x150
         *  by UMDs (example: LwMedia) for CheetAh usecases.
         */
        uint32_t arch;

        /** Implementation version. It should be initialized to zero by UMDs
         *  (example: LwMedia) for CheetAh usecases since it is not being used
         *  for them.
         */
        uint32_t impl;

        /** GPU HW revision. It should be initialized to zero by UMDs (example: LwMedia)
         *  for CheetAh usecases since it is not being used for them.
         */
        uint32_t rev;
    } __attribute__((packed)) gpu;    // For GPUs.
} LwSciSyncHwEngineRevId;

/**
 * \brief Structure identifying information about the hardware engine accessing
 *  LwSciSync. An attribute key, LwSciSyncInternalAttrKey_EngineArray, set to
 *  an array of this structure specifies all the hardware engines whose
 *  constraints should be taken into account while allocating the
 *  synchronization object.
 *
 * \implements{22823495}
 */
typedef struct {
    /** Specifies the hardware engine is from CheetAh or resman
     *  world. It is initialized to LwSciSyncHwEngine_TegraNamespaceId
     *  for CheetAh usecases.
     */
    LwSciSyncHwEngNamespace engNamespace;
    /**
     * Hardware engine ID specifying LwSciSyncHwEngName for which constraints
     * need to be applied. It should be initialized by calling
     * LwSciSyncHwEngCreateIdWithoutInstance().
     */
    int64_t rmModuleID;
    /**
     * This is lwrrently unused.
     */
    uint32_t subEngineID;

    /**
     * Specifies the revision of the hardware engine.
     */
    LwSciSyncHwEngineRevId   rev;
} __attribute__((packed)) LwSciSyncHwEngine;

/**
 * \brief Alias for enum LwSciSyncInternalAttrValPrimitiveTypeRec
 *
 * \implements{18840213}
 */
typedef enum LwSciSyncInternalAttrValPrimitiveTypeRec LwSciSyncInternalAttrValPrimitiveType;

/**
 * \brief Types of LwSciSyncInternalAttr Key.
 *
 * \implements{18840207}
 */
enum LwSciSyncInternalAttrKeyRec {
    /** For LwSciSync internal use only */
    LwSciSyncInternalAttrKey_LowerBound  = (1 << 16),
    /**
     * (LwSciSyncInternalAttrValPrimitiveType[], inout) supported primitive
     * types for signaling.
     * In a reconciled LwSciSyncAttrList, there is exactly one primitive
     * type in this array and it is identical to the (one) primitive type in the
     * value of LwSciSyncInternalAttrKey_WaiterPrimitiveInfo.
     *
     * During reconciliation, the reconciler sets the value of this key to the
     * first common supported LwSciSyncInternalAttrValPrimitiveType in the
     * LwSciSyncInternalAttrKey_SignalerPrimitiveInfo of the LwSciSyncAttrList
     * with LwSciSyncAttrKey_RequiredPerm set to either of
     * LwSciSyncAccessPerm_SignalOnly/LwSciSyncAccessPerm_WaitSignal and the
     * LwSciSyncInternalAttrKey_WaiterPrimitiveInfo of LwSciSyncAttrList(s) with
     * LwSciSyncAttrKey_RequiredPerm set to LwSciSyncAccessPerm_WaitOnly. If
     * the value of the LwSciSyncAttrKey_NeedCpuAccess key is true, then the
     * reconciled primitive type is guaranteed to be one that is CPU accessible.
     *
     * The reconciliation will fail if any of the following conditions are
     * met: The intersection of LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
     * if LwSciSyncAccessPerm_SignalOnly bit is set in
     * LwSciSyncAttrKey_RequiredPerm and LwSciSyncInternalAttrKey_WaiterPrimitiveInfo
     * if LwSciSyncAccessPerm_WaitOnly bit is set in LwSciSyncAttrKey_RequiredPerm
     * of all the input LwSciSyncAttrLists is empty, LwSciSyncAttrKey_NeedCpuAccess
     * is set in any of the input LwSciSyncAttrLists and the above intersection
     * contains no supported primitives.
     *
     * During validation of a reconciled LwSciSyncAttrList against input
     * unreconciled LwSciSyncAttrList(s), validation succeeds if the
     * LwSciSyncInternalAttrValPrimitiveType in
     * LwSciSyncInternalAttrKey_SignalerPrimitiveInfo of the reconciled
     * LwSciSyncAttrList is present in any of the
     * LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
     * LwSciSyncInternalAttrKey_WaiterPrimitiveInfo of the input
     * unreconciled LwSciSyncAttrLists.
     *
     * In unreconciled LwSciSyncAttrList this can be set as internal attribute
     * using LwSciSyncAttrListSetInternalAttrs() or if
     * LwSciSyncAttrKey_NeedCpuAccess is true then this is filled during
     * LwSciSyncAttrListIpcExportUnreconciled().
     */
    LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
    /**
     * (LwSciSyncInternalAttrValPrimitiveType[], inout) supported primitive
     * types for waiting.
     * In a reconciled LwSciSyncAttrList, there is exactly one primitive
     * type in this array and it is identical to the (one) primitive type in the
     * value of LwSciSyncInternalAttrKey_SignalerPrimitiveInfo.
     */
    LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
    /**
     * (uint32_t, inout) The number of independent signaling channels that may
     * conlwrrently need to signal. Expected to be 1 for LwMedia, and the number
     * of GPU channels per LWCA context for LWCA.
     *
     * During reconciliation, the reconciler sets the value of this key to 1 in
     * the reconciled LwSciSyncAttrList if LwSciSyncAttrKey_NeedCpuAccess is
     * true in the signaler's LwSciSyncAttrList. Otherwise it is set to the
     * value set by the UMD. The reconciliation fails if the signaler's
     * LwSciSyncAttrList sets LwSciSyncInternalAttrKey_SignalerPrimitiveCount
     * to 0.
     *
     * During validation of reconciled LwSciSyncAttrList against input
     * unreconciled LwSciSyncAttrList(s), validation succeeds if the following
     * is true:
     * - Value of LwSciSyncInternalAttrKey_SignalerPrimitiveCount in the
     *   reconciled LwSciSyncAttrList is greater than 0.
     * - If the signaler's unreconciled LwSciSyncAttrList has
     *   LwSciSyncAttrKey_CpuNeedAccess attribute set to TRUE then the
     *   reconciled LwSciSyncAttrList must have
     *   LwSciSyncInternalAttrKey_SignalerPrimitiveCount value equal to 1.
     * - If the signaler's unreconciled LwSciSyncAttrList has
     *   LwSciSyncAttrKey_CpuNeedAccess attribute set to FALSE and has
     *   LwSciSyncInternalAttrKey_SignalerPrimitiveCount equal to the
     *   LwSciSyncInternalAttrKey_SignalerPrimitiveCount in reconciled
     *   LwSciSyncAttrList.
     */
    LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
#if (LW_IS_SAFETY == 0)
    /**
     * (LwSciRmGpuId, inout) The GPU to which
     * LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore
     * refers if it is in either of the primitive type arrays.
     */
#else
    /**
     * (LwSciRmGpuId, inout) Unused
     */
#endif
    LwSciSyncInternalAttrKey_GpuId,
#if (LW_IS_SAFETY == 0)
    /**
     * (LwSciSyncAttrValTimestampInfo, inout) Indicates supported timestamp
     * format and scaling info. Only valid for signalers.
     */
#else
    /**
     * (LwSciSyncAttrValTimestampInfo, inout) Unused
     */
#endif
    LwSciSyncInternalAttrKey_SignalerTimestampInfo,
    /**
     * (LwSciSyncAttrValTimestampInfo[], inout) Indicates supported timestamp
     * format and scaling info separately for each corresponding primitive
     * specified in LwSciSyncInternalAttrKey_SignalerPrimitiveInfo. Only valid
     * for signalers.
     *
     * This attribute key should not be used in conjunction with the
     * LwSciSyncInternalAttrKey_SignalerTimestampInfo attribute key.
     */
    LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
#ifdef LWSCISYNC_EMU_SUPPORT
    /**
     * (void*[], inout) External primitive info
     * This is an optional attribute used when the user wants to use their own
     * primitives instead of relying on LwSciSync allocating them when Signaler
     * is not CPU. NULL value for a given primitive type means that the user
     * wants to rely on LwSciSync allocation for that primitive type.
     * In a reconciled LwSciSyncAttrList, there is exactly one primitive info
     * value in this array. It is the value corresponding to the reconciled
     * primitive type in LwSciSyncInternalAttrKey_SignalerPrimitiveInfo.
     *
     * During reconciliation, in unreconciled LwSciSyncAttrList with signaling
     * permission,
     * - if this key is not set and LwSciSyncAttrKey_NeedCpuAccess is unset/false,
     * reconciler does not set value for this key. During LwSciSyncObjAlloc(),
     * LwSciSync will:
     *   - not allocate primitive if reconciled primitive type is syncpoint.
     *   - allocate primitive if reconciled primitive is of semaphore type.
     * - if this key is not set and LwSciSyncAttrKey_NeedCpuAccess is true,
     * reconciler does not set value for this key and LwSciSync will allocate
     * primitive during LwSciSyncObjAlloc().
     * - if this key is set and value is NULL for the reconciled primitive,
     * reconciler does not set value for this key and LwSciSync will allocate
     * primitive during LwSciSyncObjAlloc().
     * - if this key is set and value is not NULL for the reconciled primitive,
     * reconciler sets value of this key to provided value and LwSciSync uses
     * provided external primitive during LwSciSyncObjAlloc().
     *
     * The reconciliation will fail if this key is set and value is not NULL for
     * the reconciled primitive and also the signaler requested CPU access by
     * setting LwSciSyncAttrKey_NeedCpuAccess to true in its unreconciled
     * LwSciSyncAttrList.
     *
     * During validation of reconciled LwSciSyncAttrList against input
     * unreconciled LwSciSyncAttrList(s), validation succeeds, if the signaler's
     * unreconciled LwSciSyncAttrlist:
     * - has this unset then the reconciled list also has it unset
     * - has this set then the reconciled list has this set to the primitive
     * info value in the signaler's unreconciled
     * LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo array's cell
     * corresponding to the reconciled primitive type and unreconciled list
     * has non-CPU access permissions.
     */
#else
    /**
     * (void*[], inout) Unused
     */
#endif
    LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo,
    /**
     * (LwSciSyncHwEngine[], inout) Lists engines that are going to use the
     * LwSciSyncObj. If unset, LwSciSync considers this as an unspecified
     * engine and will not provide any additional treatment that might be
     * needed.
     *
     * During reconciliation, it is set to a union of input EngineArray
     * attributes from unreconciled lists that did not travel.
     *
     * For any peer importing the reconciled LwSciSyncAttrList, this key is set
     * to the requested value set in LwSciSyncInternalAttrKey_EngineArray for
     * that peer.
     *
     * During validation of a reconciled LwSciSyncAttrList against input
     * unreconciled LwSciSyncAttrList(s), validation succeeds only if
     * all engines listed in EngineArray attribute(s) of unreconciled
     * LwSciSyncAttrList(s) are included in the EngineArray of the reconciled
     * LwSciSyncAttrList.*/
    LwSciSyncInternalAttrKey_EngineArray,
    /** For LwSciSync internal use only */
    LwSciSyncInternalAttrKey_UpperBound,
};

/**
 * \brief Alias for enum LwSciSyncInternalAttrKeyRec
 *
 * \implements{18840216}
 */
typedef enum LwSciSyncInternalAttrKeyRec LwSciSyncInternalAttrKey;

/**
 * \brief This structure defines a key/value pair used to get or set
 * the LwSciSyncInternalAttrKey(s) and their corresponding values from or to
 * LwSciSyncAttrList.
 *
 * \implements{18840210}
 */
struct LwSciSyncInternalAttrKeyValuePairRec {
    /** LwSciSyncInternalAttrKey to set */
    LwSciSyncInternalAttrKey attrKey;
    /**
     * Pointer to the attribute values, or NULL if there is no value assigned
     * to attrKey.
     */
    const void* value;
    /** buffer length, or 0 if there is no value assigned to attrKey */
    size_t len;
};

/**
 * \brief Alias for struct LwSciSyncInternalAttrKeyValuePairRec
 *
 * \implements{18840219}
 */
typedef struct LwSciSyncInternalAttrKeyValuePairRec LwSciSyncInternalAttrKeyValuePair;

/**
 * \brief Represents semaphore buffer
 */
typedef struct {
    /** logical id */
    uint64_t id;
    /** LwSciBuf object for semaphore buffer */
    LwSciBufObj bufObj;
    /** offset in semaphore buffer */
    uint64_t offset;
     /** semaphore size */
    uint32_t semaphoreSize;
    /** Is cached in GPU? */
    bool gpuCacheable;
} LwSciSyncSemaphoreInfo;

#ifdef LWSCISYNC_EMU_SUPPORT
/**
 * A set of primitive instances described just by their ids.
 * This is to be used for primitive info of
 * LwSciSyncInternalAttrValPrimitiveType_Syncpoint.
 */
typedef struct {
    /** Primitive type which this primitive info represents */
    LwSciSyncInternalAttrValPrimitiveType primitiveType;
    /** Array of ids */
    uint64_t* ids;
    /** Number of ids in above array. Must not be greater than 8 */
    size_t numIds;
} LwSciSyncSimplePrimitiveInfo;

/** Describes external semaphore primitive info. */
typedef struct {
    /** Primitive type which this primitive info represents */
    LwSciSyncInternalAttrValPrimitiveType primitiveType;
    /** RM memory handle */
    LwSciBufRmHandle memHandle;
    /**
     * The offset within the buffer represented by LwSciBufRmHandle which
     * indicates start of semaphore memory.
     */
    uint64_t offset;
    /**
     * The length of the semaphore memory buffer. The size of the buffer
     * represented by LwSciBufRmHandle must be at least @a offset + @a len.
     * @a len should be at least 16 bytes.
     */
    uint64_t len;
} LwSciSyncSemaphorePrimitiveInfo;
#endif

/**
 * \brief List of timestamp formats
 */
typedef enum {
    /** 64-bit timestamp value (for host1x engines) */
    LwSciSyncTimestampFormat_8Byte,
    /** 16-byte GPU semaphore */
    LwSciSyncTimestampFormat_16Byte,
    /** For use when the backing primitive doesn't support timestamps. */
    LwSciSyncTimestampFormat_Unsupported,
    /** No additional buffer will be allocated for timestamps. Instead, the
     * timestamp is stored along with each primitive. This is only valid for
     * semaphore primitives. */
    LwSciSyncTimestampFormat_EmbeddedInPrimitive,
} __attribute__((packed)) LwSciSyncTimestampFormat;

/**
 * \brief Scaling information
 *
 * If the engine returns scaled timestamps, the actual microseconds value can
 * be retrieved like this:
 *
 * TimestampMicroseconds = sourceOffset + (rawTimestamp *
 *                         scalingFactorNumerator / scalingFactorDenominator)
 */
typedef struct {
    uint64_t scalingFactorNumerator;
    uint64_t scalingFactorDenominator;
    uint64_t sourceOffset;
} __attribute__((packed)) LwSciSyncTimestampScaling;

/**
 * \brief Types of LwSciSyncInternalAttrKey Value - Timestamp info
 */
typedef struct {
    LwSciSyncTimestampFormat format;
    LwSciSyncTimestampScaling scaling;
} __attribute__((packed)) LwSciSyncAttrValTimestampInfo;

/**
 * \brief Represents timestamp buffer info
 */
typedef struct {
    /** LwSciBuf object for timestamp buffer */
    LwSciBufObj bufObj;
    /** timestamp buffer size */
    size_t size;
} LwSciSyncTimestampBufferInfo;

/**
 * \brief Sets the value(s) of LwSciSyncInternalAttrKey(s) in slot 0
 * of the input LwSciSyncAttrList.
 *
 * \param[in] attrList unreconciled LwSciSyncAttrList
 * \param[in] pairArray Array of LwSciSyncInternalAttrKeyValuePair
 * Valid value: pairArray is valid input if it is not NULL and key member of
 * every LwSciSyncInternalAttrKeyValuePair in the array is an input or
 * input/output attribute and it is > LwSciSyncInternalAttrKey_LowerBound and
 * < LwSciSyncInternalAttrKey_UpperBound and value member of every
 * LwSciSyncInternalAttrKeyValuePair in the array is not NULL.
 *
 * \param[in] pairCount number of LwSciSyncInternalAttrKeyValuePairs
              in pairArray
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a attrList is NULL
 *       - @a pairArray is NULL
 *       - @a attrList is not unreconciled and/or not writable,
 *       - @a pairCount is 0
 *       - @a pairArray has duplicate keys
 *       - any of the keys in @a pairArray is not a supported internal key
 *       - any of the values in @a pairArray is NULL
 *       - any of the len(s) in @a pairArray is invalid for a given attribute
 *       - any of the attributes to be written is non-writable in attrList
 * - Panics if @a attrList is invalid
 */
LwSciError LwSciSyncAttrListSetInternalAttrs(
    LwSciSyncAttrList attrList,
    const LwSciSyncInternalAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Gets the value(s) of LwSciSyncInternalAttrKey(s) from slot 0
 * of the input the LwSciSyncAttrList.
 * If an LwSciSyncInternalAttrKey was not set, this function
 * will populate the corresponding value to NULL and length to 0.
 *
 * \param[in] attrList LwSciSyncAttrList for which the values has to be
 * retrieved for the given LwSciSyncInternalAttrKey(s).
 * \param[in,out] pairArray Array of LwSciSyncInternalAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL and key member of every
 * LwSciSyncInternalAttrKeyValuePair in the array > LwSciSyncInternalAttrKey_LowerBound
 * and < LwSciSyncInternalAttrKey_UpperBound.
 * \param[in] pairCount Number of elements/entries in pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 *         - @a pairCount is 0
 *         - any of the keys in @a pairArray is not a supported
 *           LwSciSyncInternalAttrKey
 * - Panics if @a attrList is not valid
 */
LwSciError LwSciSyncAttrListGetInternalAttrs(
    LwSciSyncAttrList attrList,
    LwSciSyncInternalAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Increments the reference count on the input LwSciSyncObj.
 *
 * As a result, in order to fully free the resources managed by this input
 * LwSciSyncObj in the caller's process, an additional free call is required
 * to decrement both reference counts.
 *
 * \param[in] syncObj LwSciSyncObj to increment reference counts on
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a syncObj is NULL
 * - LwSciError_IlwalidState if total number of references to memory
 *   object are INT32_MAX and caller tries to take one more reference
 *   using this API.
 * - Panics if @a syncObj is invalid
 */
LwSciError LwSciSyncObjRef(
    LwSciSyncObj syncObj);

/**
 * \brief Populates the input LwSciSyncFence based on the input id and value.
 *
 * This new LwSciSyncFence is associated with the input LwSciSyncObj. The input
 * LwSciSyncFence is cleared before being populated with the new data.
 *
 * \param[in] syncObj valid LwSciSyncObj
 * \param[in] id LwSciSyncFence identifier
 * Valid value: [0, UINT32_MAX-1] for LwSciSyncInternalAttrValPrimitiveType_Syncpoint.
 * [0, value returned by LwSciSyncObjGetNumPrimitives()-1] for
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore and
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b.
 * \param[in] value LwSciSyncFence value
 * Valid value: [0, UINT32_MAX] for LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * and LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore.
 * [0, UINT64_MAX] for
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b.
 * \param[in,out] syncFence LwSciSyncFence to populate
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if syncObj or syncFence is NULL.
 * - LwSciError_Overflow if id is invalid.
 * - LwSciError_Overflow if value is invalid.
 * - LwSciError_IlwalidState if no more references can be taken on
 *   the syncObj
 * - Panics if syncObj or LwSciSyncObj initially associated with syncFence
 *   is invalid
 */
LwSciError LwSciSyncFenceUpdateFence(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value,
    LwSciSyncFence* syncFence);

/**
 * \brief Populates the input LwSciSyncFence based on the input id, value
 *  and slot index.
 *
 * This new LwSciSyncFence is associated with the input LwSciSyncObj. The input
 * LwSciSyncFence is cleared before being populated with the new data.
 *
 * \param[in] syncObj valid LwSciSyncObj
 * \param[in] id LwSciSyncFence id
 * Valid value: [0, UINT32_MAX-1] for LwSciSyncInternalAttrValPrimitiveType_Syncpoint.
 * [0, value returned by LwSciSyncObjGetNumPrimitives()-1] for
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore and
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b.
 * \param[in] value LwSciSyncFence value
 * Valid value: [0, UINT32_MAX] for LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * and LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore.
 * [0, UINT64_MAX] for
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b.
 * \param[in] slotIndex slot index in the timestamp buffer
 * \param[out] syncFence LwSciSyncFence to populate
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if syncObj or syncFence is NULL.
 * - LwSciError_IlwalidState if no more references can be taken on
 *   the syncObj
 * - Panics if syncObj or LwSciSyncObj initially associated with syncFence
 *   is invalid
 */
LwSciError LwSciSyncFenceUpdateFenceWithTimestamp(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value,
    uint32_t slotIndex,
    LwSciSyncFence* syncFence);

/**
 * \brief Extracts the id and value from the input LwSciSyncFence.
 *
 * \param[in] syncFence LwSciSyncFence from which the id and value should be
 * retrieved
 * \param[out] id LwSciSyncFence id
 * \param[out] value LwSciSyncFence value
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if syncFence is NULL or invalid or id/value are NULL
 * - LwSciError_ClearedFence if syncFence is a valid cleared LwSciSyncFence
 * - Panics if the LwSciSyncObj associated with the syncFence is invalid
 */
LwSciError LwSciSyncFenceExtractFence(
    const LwSciSyncFence* syncFence,
    uint64_t* id,
    uint64_t* value);

/**
 * \brief Extracts the LwSciSyncObj associated with LwSciSyncFence.
 *
 * \param[in] syncFence LwSciSyncFence from which the LwSciSyncObj should be
 * retrieved
 * \param[out] syncObj LwSciSyncObj associated with the LwSciSyncFence
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if syncFence is NULL, or syncObj is NULL, or
 *   syncObj associated with syncFence is NULL.
 * - LwSciError_ClearedFence if syncFence is a valid cleared fence.
 * - Panics if syncObj associated with @a syncFence is invalid
 */
LwSciError LwSciSyncFenceGetSyncObj(
    const LwSciSyncFence* syncFence,
    LwSciSyncObj* syncObj);

/**
 * \brief Returns the semaphore descriptor of the semaphore at position "index"
 * of the supplied LwSciSync object. Used for setting up primitives
 * (e.g., mapping the LwSciBuf object).
 *
 * \param[in] syncObj LwSciSync object
 * \param[in] index index in semaphore array.
 * \parblock
 * Max index value should be below the value obtained from
 * LwSciSyncObjGetNumPrimitives()
 * \endparblock
 * \param[out] semaphoreInfo represents the semaphore allocation data
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a syncObj or @a semaphoreInfo is NULL
 *       - @a index is bigger than the primitive count in @a syncObj
 *       - syncObj's primitive type is not semaphore.
 * - Panics if @a syncObj is invalid.
 */
LwSciError LwSciSyncObjGetSemaphoreInfo(
    const LwSciSyncObj syncObj,
    uint32_t index,
    LwSciSyncSemaphoreInfo* semaphoreInfo);

/*
 * \brief LwSciSync Utility functions
 */

/**
 * \brief Gets the value of an LwSciSyncInternalAttrKey from slot 0 of the
 * input LwSciSyncAttrList.
 * If an LwSciSyncInternalAttrKey was not set, this function
 * will populate value to NULL and length to 0.
 *
 * \param[in] attrList LwSciSyncAttrList for which the value for the
 * LwSciSyncInternalAttrKey has to be retrieved.
 * \param[in] key LwSciSyncInternalAttrKey for which the value has to be
 * retrieved.
 * Valid value: key is valid input if it is an input or input/output attribute
 * and it is > LwSciSyncInternalAttrKey_LowerBound and <
 * LwSciSyncInternalAttrKey_UpperBound
 * \param[out] value pointer where value of attribute is written
 * \param[out] len length of the value
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *   - @a attrList is NULL
 *   - @a value is NULL
 *   - @a len is NULL
 *   - @a key is invalid
 * - Panics if @a attrList is invalid
 */
LwSciError LwSciSyncAttrListGetSingleInternalAttr(
    LwSciSyncAttrList attrList,
    LwSciSyncInternalAttrKey key,
    const void** value,
    size_t* len);

/**
 * \brief Returns the LwSciSyncInternalAttrValPrimitiveType of LwSciSyncObj.
 *
 * Note: Mixed LwSciSyncInternalAttrValPrimitiveTypes in a LwSciSyncObj are not
 * supported so even if there are multiple primitives in a LwSciSyncObj all will
 * be of a single type.
 *
 * Note: This function serves as a colwenience function that retrieves the
 * LwSciSyncInternalAttrKey_SignalerPrimitiveInfo attribute key from the
 * LwSciSync Reconciled LwSciSyncAttrList associated with the LwSciSyncObj using
 * LwSciSyncAttrListGetSingleInternalAttr().
 *
 * \param[in] syncObj A valid LwSciSyncObj.
 * \param[out] primitiveType The LwSciSyncInternalAttrValPrimitiveType associated
 * with the given LwSciSyncObj.
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if @a syncObj is NULL or @a primitiveType is NULL.
 * - Panics if @a syncObj is invalid
 */
LwSciError LwSciSyncObjGetPrimitiveType(
    LwSciSyncObj syncObj,
    LwSciSyncInternalAttrValPrimitiveType* primitiveType);

/**
 * \brief Returns number of primitives in the given LwSciSyncObj.
 *
 * Note: This function serves as a colwenience function that retrieves the
 * value of the LwSciSyncInternalAttrKey_SignalerPrimitiveCount attribute key
 * from the LwSciSync Reconciled LwSciSyncAttrList associated with the
 * LwSciSyncObj using LwSciSyncAttrListGetSingleInternalAttr.
 *
 * \param[in] syncObj A valid LwSciSyncObj.
 * \param[out] numPrimitives Number of primitives count.
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a syncObj is NULL or @a numPrimitives is NULL
 * - Panics if @a syncObj is invalid
 */
LwSciError LwSciSyncObjGetNumPrimitives(
    LwSciSyncObj syncObj,
    uint32_t* numPrimitives);

/**
 * \brief Returns the next timestamp buffer slot index.
 *
 * LwSciSync maintains the buffer index in round-robin manner.
 *
 * \param[in] syncObj LwSciSyncObj
 * \param[out] slotIndex index in the timestamp buffer
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if slotIndex is NULL or timestamps not supported
 *   or syncObj is invalid
 */
LwSciError LwSciSyncObjGetNextTimestampSlot(
    const LwSciSyncObj syncObj,
    uint32_t* slotIndex);

/**
 * \brief Fills the provided timestamp buffer info with information
 * necessary for UMDs to properly interact with timestamps buffer.
 *
 * \param[in] syncObj LwSciSync object
 * \param[out] bufferInfo represents the timestamp buffer allocation info
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if syncObj is invalid or bufferInfo is NULL
 *   or if timestamps not supported in syncObj
*/
LwSciError LwSciSyncObjGetTimestampBufferInfo(
    LwSciSyncObj syncObj,
    LwSciSyncTimestampBufferInfo* bufferInfo);

/**
 * \brief Generates LwSciSync hardware engine ID from the given LwSciSyncHwEngName
 *        without hardware engine instance.
 *
 * @param[in] engName: LwSciSyncHwEngName from which the ID is obtained.
 *  Valid value: LwSciSyncHwEngName enum value > LwSciSyncHwEngName_LowerBound
 *  and < LwSciSyncHwEngName_UpperBound.
 * @param[out] engId: LwSciSync hardware engine ID generated from LwSciSyncHwEngName.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a engName is invalid
 *      - @a engId is NULL
 */
LwSciError LwSciSyncHwEngCreateIdWithoutInstance(
    LwSciSyncHwEngName engName,
    int64_t* engId);

/**
 * \brief Retrieves LwSciSyncHwEngName from LwSciSync hardware engine ID.
 *
 * @param[in] engId: LwSciSync hardware engine ID.
 *  Valid value: engine ID obtained from successful call to
 *  LwSciSyncHwEngCreateIdWithoutInstance().
 * @param[out] engName: LwSciSyncHwEngName retrieved from engine ID.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a engId is invalid
 *       - @a engName is NULL.
 */
LwSciError LwSciSyncHwEngGetNameFromId(
    int64_t engId,
    LwSciSyncHwEngName* engName);

/**
 * \brief Retrieves hardware engine instance from LwSciSync
 *        hardware engine ID.
 *
 * @param[in] engId: LwSciSync hardware engine ID.
 *  Valid value: engine ID obtained from successful call to
 *  LwSciSyncHwEngCreateIdWithoutInstance().
 * @param[out] instance: Hardware engine instance retrieved from
 *  engine ID.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a instance is NULL.
 *       - @a engId is invalid.
 */
LwSciError LwSciSyncHwEngGetInstanceFromId(
    int64_t engId,
    uint32_t* instance);

#if defined(__cplusplus)
}
#endif // __cplusplus
 /** @} */
#endif // INCLUDED_LWSCISYNC_INTERNAL_H
