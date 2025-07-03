/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_INTERFACE_H
#define INCLUDED_LWSCIBUF_ALLOC_INTERFACE_H

#include "lwscibuf_dev.h"
#include "lwscibuf_internal.h"

/**
 * @addtogroup lwscibuf_blanket_statements
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements
 * \section lwscibuf_in_params Input parameters
 * - LwSciBufAllocIfaceType passed as input parameter to an API is valid input
 * if it is greater than or equal to LwSciBufAllocIfaceType_SysMem and less
 * than LwSciBufAllocIfaceType_Max.
 * - Allocation context passed as input parameter to an API is valid if it is
 * not NULL and it is returned from a successful call to
 * LwSciBufAllocIfaceGetAllocContext().
 * - LwSciBufRmHandle passed as input parameter to an API is valid if it is
 * returned from a successful call to LwSciBufAllocIfaceAlloc() and has not
 * been deallocated using LwSciBufAllocIfaceDeAlloc().
 */

/**
 * @}
 */

/**
 * @brief Enum describes the types of allocation interfaces.
 *
 * @implements{18842244}
 */
typedef enum {
	/** This value is used if LwSciBufPrivateAttrKey_MemDomain
	 * is set to LwSciBufMemDomain_Sysmem during reconciliation */
    LwSciBufAllocIfaceType_SysMem = 0x0U,
#if (LW_IS_SAFETY == 0)
    LwSciBufAllocIfaceType_VidMem = 0x1U,
#endif
    LwSciBufAllocIfaceType_Max = 0x2U,
} LwSciBufAllocIfaceType;

/**
 * @brief Enum describes the heap types from which the memory
 * needs to be allocated.
 *
 * @implements{18842247}
 */
typedef enum {
    LwSciBufAllocIfaceHeapType_IOMMU,
    LwSciBufAllocIfaceHeapType_ExternalCarveout,
    LwSciBufAllocIfaceHeapType_IVC,
    LwSciBufAllocIfaceHeapType_VidMem,
    LwSciBufAllocIfaceHeapType_CvsRam,
    LwSciBufAllocIfaceHeapType_Ilwalid,
} LwSciBufAllocIfaceHeapType;

/**
 * @brief Structure specifying allocation properties of the buffer.
 *
 * Synchronization: Access to an instance of this datatype must be
 * externally synchronized
 *
 * @implements{18842250}
 */
typedef struct {
    /**
     * Size of the buffer to be allocated.
     */
    uint64_t size;

    /**
     * Alignment of the buffer to be allocated.
     */
    uint64_t alignment;

    /**
     * Flag to specify whether the CPU caching for the buffer should be enabled
     * or disabled.
     */
    bool coherency;

    /**
     * Array of LwSciBufAllocIfaceHeapType(s).
     */
    LwSciBufAllocIfaceHeapType* heap;

    /**
     * Number of LwSciBufAllocIfaceHeapType(s) held by the heap array.
     */
    uint32_t numHeaps;

    /**
     * Flag to specify whether the CPU access for the buffer is required or not.
     */
    bool cpuMapping;
} LwSciBufAllocIfaceVal;

/**
 * @brief Structure to store parameters which are required to create the
 * allocation context of a LwSciBufAllocIfaceType.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * @implements{18842253}
 */
typedef struct {
#if (LW_IS_SAFETY == 0)
	/**
	 * GPU ID of dGPU from which buffer is allocated.
	 */
    LwSciRmGpuId gpuId;
#endif

    /**
     * Array of GpuIDs which will access the buffer allocated from
     * LwSciBufAllocIfaceType_SysMem
     */
    const LwSciRmGpuId* gpuIds;

    /**
     * Number of LwSciRmGpuId(s) held by the gpuIds array.
     */
    uint64_t gpuIdsCount;
} LwSciBufAllocIfaceAllocContextParams;

/**
 * @brief Creates and returns opaque open context corresponding to
 * specified LwSciBufAllocIfaceType. Calls an allocation interface
 * corresponding to specified LwSciBufAllocIfaceType for the given
 * platform and obtains the opaque context from it.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufAllGpuContext
 *        associated with the LwSciBufDev is never modified after creation
 *        (so there is no data-dependency)
 *
 * @param[in] allocType: LwSciBufAllocIfaceType for which the opaque open
 * context should be created.
 * @param[in] devHandle: LwSciBufDev.
 * @param[out] context: Opaque open context.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - panics if any one of the following:
 *   - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *   - @a devHandle is NULL
 *   - @a context is NULL
 *
 * @implements{18842922}
 */
LwSciError LwSciBufAllocIfaceOpen(
    LwSciBufAllocIfaceType allocType,
    LwSciBufDev devHandle,
    void** context);

/**
 * @brief Allocates a buffer specified by LwSciBufAllocIfaceVal for specified
 * LwSciBufAllocIfaceType. Allocates buffer by calling an allocation interface
 * corresponding to specified LwSciBufAllocIfaceType for the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @param[in] allocType: The LwSciBufAllocIfaceType.
 * @param[in] context: Opaque allocation context.
 * @param[in] iFaceAllocVal: The LwSciBufAllocIfaceVal.
 * Valid value: iFaceAllocVal is valid input if heap member of
 * LwSciBufAllocIfaceVal is not NULL, size and numHeaps members are not zero.
 * @param[in] devHandle: LwSciBufDev.
 * @param[out] rmHandle: LwSciBufRmHandle of the allocated buffer.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - LwSciError_ResourceError if LWPU driver stack failed to allocate the
 *   buffer.
 * - panics if any one of the following:
 *     - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a iFaceAllocVal.size is 0
 *     - @a iFaceAllocVal.heap is NULL
 *     - @a iFaceAllocVal.numHeaps is 0
 *     - @a devHandle is NULL
 *     - @a rmHandle is NULL
 *     - @a if any of the heap in array iFaceAllocVal.heap is
 *           >= LwSciBufAllocIfaceHeapType_Ilwalid
 *
 * @implements{18842928}
 */
LwSciError LwSciBufAllocIfaceAlloc(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufAllocIfaceVal iFaceAllocVal,
    LwSciBufDev devHandle,
    LwSciBufRmHandle *rmHandle);

/**
 * @brief Deallocates the buffer by calling an allocation interface
 * corresponding to specified LwSciBufAllocIfaceType for the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufRmHandle is handled via
 *        LwSciBufSysMemDealloc()
 *
 * @param[in] allocType: The LwSciBufAllocIfaceType.
 * @param[in] context: Opaque allocation context.
 * @param[in] rmHandle: LwSciBufRmHandle of the buffer.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - LwSciError_ResourceError if LWPU driver stack failed to deallocate the
 *   buffer.
 * - panics if any one of the following:
 *   - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *   - @a context is NULL
 *
 * @implements{18842931}
 */
LwSciError LwSciBufAllocIfaceDeAlloc(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufRmHandle rmHandle);

/**
 * @brief Duplicates the buffer handle with the specified LwSciBufAttrValAccessPerm
 * by calling an allocation interface corresponding to specified
 * LwSciBufAllocIfaceType for the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufRmHandle is handled via
 *        LwSciBufSysMemDupHandle()
 *
 * @param[in] allocType: The LwSciBufAllocIfaceType.
 * @param[in] context: Opaque allocation context.
 * @param[in] newPerm: New LwSciBufAttrValAccessPerm.
 * Valid input: LwSciBufAccessPerm_Readonly <= newPerm <=
 * LwSciBufAccessPerm_Auto.
 * @param[in] rmHandle: LwSciBufRmHandle of the buffer to duplicate.
 * @param[out] dupHandle: Duplicated LwSciBufRmHandle.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - LwSciError_ResourceError if LWPU driver stack failed to duplicate
 *   the LwSciBufRmHandle.
 * - panics if any one of the following:
 *     - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a newPerm >= LwSciBufAccessPerm_Ilwalid
 *     - @a dupHandle is NULL
 *
 * @implements{18842934}
 */
LwSciError LwSciBufAllocIfaceDupHandle(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle* dupHandle);

/**
 * @brief Maps the buffer at the given offset and given length with the
 * specified LwSciBufAttrValAccessPerm and gets the CPU virtual address of the
 * mapped buffer by calling an allocation interface corresponding to specified
 * LwSciBufAllocIfaceType for the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufRmHandle is handled via
 *        LwSciBufSysMemMemMap()
 *
 * @param[in] allocType: The LwSciBufAllocIfaceType.
 * @param[in] context: Opaque allocation context.
 * @param[in] rmHandle: LwSciBufRmHandle of the buffer to map.
 * @param[in] offset: The starting offset of the buffer.
 * Valid value: 0 to buffer size - 1.
 * @param[in] len: The length (in bytes) of the buffer to map.
 * Valid value: 1 to buffer size - offset.
 * @param[in] iFaceAccPerm: LwSciBufAttrValAccessPerm of the mapped buffer.
 * Valid input: LwSciBufAccessPerm_Readonly <= iFaceAccPerm <=
 * LwSciBufAccessPerm_Auto.
 * @param[out] ptr: CPU virtual address.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - LwSciError_ResourceError if LWPU driver stack failed to map the
 *   buffer.
 * - panics if any one of the following:
 *     - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a iFaceAccPerm != LwSciBufAccessPerm_Readonly and @a iFaceAccPerm !=
 *       LwSciBufAccessPerm_ReadWrite
 *     - @a len is 0
 *     - @a ptr is NULL
 *
 * @implements{18842937}
 */
LwSciError LwSciBufAllocIfaceMemMap(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm iFaceAccPerm,
    void** ptr);

/**
 * @brief Unmaps the mapped buffer by calling an allocation interface
 * corresponding to specified LwSciBufAllocIfaceType for the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufRmHandle is handled via
 *        LwSciBufSysMemMemUnMap()
 *
 * @param[in] allocType: The LwSciBufAllocIfaceType.
 * @param[in] context: Opaque allocation context.
 * @param[in] rmHandle: LwSciBufRmHandle of the buffer to unmap.
 * @param[in] ptr: CPU virtual address.
 * Valid value: ptr is valid input if it is not NULL and it is returned from a
 * successful call to LwSciBufAllocIfaceMemMap() and has not been unmapped
 * using LwSciBufAllocIfaceMemUnMap().
 * @param[in] size: Length (in bytes) of the mapped buffer.
 * Valid value: 1 to buffer size.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - LwSciError_ResourceError if LWPU driver stack failed to unmap the
 *   buffer.
 * - panics if any one of the following:
 *     - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a size is 0
 *     - @a ptr is NULL
 *
 * @implements{18842940}
 */
LwSciError LwSciBufAllocIfaceMemUnMap(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size);

/**
 * @brief Gets the size of the buffer by calling an allocation interface
 * corresponding to specified LwSciBufAllocIfaceType for the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufRmHandle is handled via
 *        LwSciBufSysMemGetSize()
 *
 * @param[in] allocType: The LwSciBufAllocIfaceType.
 * @param[in] context: Opaque allocation context.
 * @param[in] rmHandle: LwSciBufRmHandle of the buffer.
 * @param[out] size: Size of the buffer.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - LwSciError_ResourceError if LWPU driver stack failed to get the
 *   size of the buffer.
 * - panics if any one of the following:
 *     - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a size is NULL
 *
 * @implements{18842943}
 */
LwSciError LwSciBufAllocIfaceGetSize(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size);

#if (LW_IS_SAFETY == 0)
LwSciError LwSciBufAllocIfaceGetAlignment(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* alignment);

LwSciError LwSciBufAllocIfaceGetHeapType(
    LwSciBufAllocIfaceType allocType,
    void* context,
    LwSciBufRmHandle rmHandle,
    LwSciBufAllocIfaceHeapType* heap);
#endif

/**
 * @brief Flushes the given @c len bytes of the mapped buffer by calling an
 * allocation interface corresponding to specified LwSciBufAllocIfaceType for
 * the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufRmHandle is handled via
 *        LwSciBufSysMemCpuCacheFlush()
 *
 * @param[in] allocType: The LwSciBufAllocIfaceType.
 * @param[in] context: Opaque allocation context.
 * @param[in] rmHandle: LwSciBufRmHandle of the buffer.
 * @param[in] cpuPtr: CPU virtual address.
 * Valid value: cpuPtr is valid input if it is not NULL and it is returned from
 * a successful call to LwSciBufAllocIfaceMemMap() and has not been unmapped
 * using LwSciBufAllocIfaceMemUnMap().
 * @param[in] len: Length (in bytes) of the buffer to flush.
 * Valid value: 1 to buffer size.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - LwSciError_ResourceError if LWPU driver stack failed to flush the
 *   buffer.
 * - panics if any one of the following:
 *     - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a cpuPtr is NULL
 *     - @a len is 0
 *
 * @implements{18842946}
 */
LwSciError LwSciBufAllocIfaceCpuCacheFlush(
    LwSciBufAllocIfaceType allocType,
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len);

/**
 * @brief Closes the opaque open context of the given
 * LwSciBufAllocIfaceType by calling an allocation interface corresponding to
 * specified LwSciBufAllocIfaceType for the given platform.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - User must ensure that context is not used in any of the other APIs
 *        when this API is being called with the same context
 *
 * @param[in] allocType: LwSciBufAllocIfaceType for which the allocation
 * interface context should be closed.
 * @param[in] context: Opaque open context.
 * Valid value: context is valid input if it is not NULL and it is returned
 * from a successful call to LwSciBufAllocIfaceOpen() and has not been
 * closed using LwSciBufAllocIfaceClose().
 * - panics if any one of the following:
 *   - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *   - @a context is NULL
 *
 * @return void
 *
 * @implements{18842949}
 */
void LwSciBufAllocIfaceClose(
    LwSciBufAllocIfaceType allocType,
    void* context);

#if (LW_IS_SAFETY == 0)
void LwSciBufAllocIfacePrintHeapTypes(
    const LwSciBufAllocIfaceHeapType* heaps,
    uint32_t numHeaps);
#endif

/**
 * @brief Creates an opaque allocation context for the specified
 * LwSciBufAllocIfaceType by calling the corresponding allocation
 * interface for the given platform. It colwerts
 * LwSciBufAllocIfaceAllocContextParams into corresponding allocation interface
 * specific allocation context parameters and obtains opaque allocation context
 * by calling corresponding allocation interface.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output opaque open context
 *        is not used by multiple threads at the same time when calling this
 *        API
 *
 * @param[in] allocType: LwSciBufAllocIfaceType for which the allocation
 * context should be created.
 * @param[in] iFaceAllocContextParams: The LwSciBufAllocIfaceAllocContextParams.
 * Valid value: If there are no GPUs in the system, gpuIds and gpuIdsCount
 * members of LwSciBufAllocIfaceAllocContextParams can be filled with NULL and 0
 * respectively, otherwise gpuIds and gpuIdsCount members should be non NULL and
 * non zero respectively.
 * @param[in,out] openContext: Opaque open context.
 * Valid value: openContext is valid input if it is not NULL and it is returned
 * from a successful call to LwSciBufAllocIfaceOpen() and has not been
 * closed using LwSciBufAllocIfaceClose().
 * @param[out] allocContext: Opaque allocation context.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.
 * - panics if any one of the following:
 *     - @a allocType < 0 or >= LwSciBufAllocIfaceType_Max
 *     - @a openContext is NULL
 *     - @a openContext is invalid
 *     - @a allocContext is NULL
 *
 * @implements{18842925}
 */
LwSciError LwSciBufAllocIfaceGetAllocContext(
    LwSciBufAllocIfaceType allocType,
    LwSciBufAllocIfaceAllocContextParams iFaceAllocContextParams,
    void* openContext,
    void** allocContext);

#endif /* INCLUDED_LWSCIBUF_ALLOC_INTERFACE_H */
