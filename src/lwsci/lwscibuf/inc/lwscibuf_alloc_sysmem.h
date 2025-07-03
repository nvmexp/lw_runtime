/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_SYSMEM_H
#define INCLUDED_LWSCIBUF_ALLOC_SYSMEM_H

#include "lwscibuf_dev.h"
#include "lwscibuf_internal.h"

/**
 * @addtogroup lwscibuf_blanket_statements
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements
 */

/**
 * @}
 */

/**
 * \brief Enum describes the heap types from which the buffer
 * needs to be allocated.
 *
 * \implements{18842274}
 */
typedef enum {
    /** Heap type of IOMMU */
    LwSciBufAllocSysMemHeapType_IOMMU,
    /** Heap type of external carveout */
    LwSciBufAllocSysMemHeapType_ExternalCarveout,
    /** Heap type of IVC */
    LwSciBufAllocSysMemHeapType_IVC,
    /** Heap type of VidMem */
    LwSciBufAllocSysMemHeapType_VidMem,
    /** Heap type of CvsRam */
    LwSciBufAllocSysMemHeapType_CvsRam,
    /** Invalid heap type */
    LwSciBufAllocSysMemHeapType_Ilwalid,
} LwSciBufAllocSysMemHeapType;

/**
 * \brief Enum describes which GPUs have access to the allocated buffer.
 *
 * \implements{18842277}
 */
typedef enum {
    /** GPU don't have access */
    LwSciBufAllocSysGpuAccess_None = 0x0U,
    /** Internal GPU have access */
    LwSciBufAllocSysGpuAccess_iGPU,
    /** Desktop GPU have access */
    LwSciBufAllocSysGpuAccess_dGPU,
    /** All of the GPUs have access */
    LwSciBufAllocSysGpuAccess_GPU,
    /** Invalid GPU access */
    LwSciBufAllocSysGpuAccess_Ilwalid,
} LwSciBufAllocSysGpuAccessType;

/**
 * \brief Structure to store parameters which are considered during
 * allocation of the buffer.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842280}
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
     * Array of LwSciBufAllocSysMemHeapType(s).
     */
    LwSciBufAllocSysMemHeapType* heap;

    /**
     * Number of LwSciBufAllocSysMemHeapType(s) held by the heap array.
     */
    uint32_t numHeaps;

    /**
     * Specific number of the heap.
     */
    uint32_t heapNumber;

    /**
     * GPU type which will access the buffer.
     */
    LwSciBufAllocSysGpuAccessType gpuAccess;
} LwSciBufAllocSysMemVal;

/**
 * \brief Structure to store parameters which are required to create the
 * allocation context.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842283}
 */
typedef struct {
    /**
     * Array of GpuIDs which will access the buffer.
     */
    const LwSciRmGpuId* gpuIds;

    /**
     * Number of gpuId held by the @a gpuIds array.
     */
    uint64_t gpuIdsCount;
} LwSciBufAllocSysMemAllocContextParam;

/**
 * \brief Creates and returns opaque open context for
 * LwSciBufAllocIfaceType_SysMem.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufAllGpuContext
 *        associated with the LwSciBufDev is never modified after creation
 *        (so there is no data-dependency)
 *
 * \param[in] devHandle LwSciBufDev.
 * \param[out] context Opaque open context.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - panics if any of the following:
 *     - @a context is NULL
 *     - @a devHandle is NULL
 *
 * \implements{18842640}
 */
LwSciError LwSciBufSysMemOpen(
    LwSciBufDev devHandle,
    void** context);

/**
 * \brief Allocates a buffer from sysmem according to the buffer properties
 * specified in LwSciBufAllocSysMemVal by calling liblwrm_mem allocation
 * interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same input/output allocVal parameter is
 *        not used by multiple threads at the same time
 *
 * \param[in] context Opaque allocation context.
 * \param[in,out] allocVal The LwSciBufAllocSysMemVal.
 *  Valid value: allocVal is valid input if heap member of LwSciBufAllocSysMemVal
 *  is not NULL, size and numHeaps members are not zero, and alignment is a
 *  power of two.
 * \param[in] devHandle The LwSciBufDev.
 * \param[out] rmHandle LwSciBufRmHandle of the allocated buffer.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_ResourceError if liblwrm_mem failed to allocate the buffer.
 * - LwSciError_BadParameter if @a allocVal is invalid
 * - panics if any of the following:
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a allocVal is NULL
 *     - @a allocVal->size is 0
 *     - @a allocVal->heap is NULL
 *     - @a allocVal->numHeaps is 0
 *     - @a rmHandle is NULL
 *     - @a devHandle is NULL
 *
 * \implements{18842646}
 */
LwSciError LwSciBufSysMemAlloc(
    const void* context,
    void* allocVal,
    const LwSciBufDev devHandle,
    LwSciBufRmHandle* rmHandle);

/**
 * \brief Deallocates the buffer by calling cheetah common interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access of the rmHandle is handled via
 *        LwSciBufAllocCommonTegraMemFree()
 *
 * \param[in] context Opaque allocation context.
 * \param[in] rmHandle LwSciBufRmHandle of the buffer.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - panics if any one of the following:
 *     - @a context is NULL.
 *     - @a context is invalid.
 *     - @a rmHandle.memHandle is 0.
 *
 * \implements{18842649}
 */
LwSciError LwSciBufSysMemDealloc(
    void* context,
    LwSciBufRmHandle rmHandle);

/**
 * \brief Duplicates the LwSciBufRmHandle with
 * the given LwSciBufAttrValAccessPerm from input LwScibufRmHandle
 * by calling cheetah common interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the rmHandle is handled via
 *        LwSciBufAllocCommonTegraDupHandle()
 *
 * \param[in] context Opaque allocation context.
 * \param[in] newPerm New LwSciBufAttrValAccessPerm.
 *  Valid value: LwSciBufAccessPerm_Readonly <= newPerm <
 *  LwSciBufAccessPerm_Auto.
 * \param[in] rmHandle LwSciBufRmHandle of the buffer to duplicate.
 * \param[out] dupRmHandle Duplicated LwSciBufRmHandle.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem failed to duplicate
 *   the LwSciBufRmHandle.
 * - panics if any one of the following:
 *     - @a newPerm != LwSciBufAccessPerm_Readonly and @a newPerm !=
 *       LwSciBufAccessPerm_ReadWrite
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a dupRmHandle is NULL
 *     - @a rmHandle.memHandle is 0
 *
 * \implements{18842652}
 */
LwSciError LwSciBufSysMemDupHandle(
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle *dupRmHandle);

/**
 * \brief Maps the buffer at the given offset and given length with the given
 * LwSciBufAttrValAccessPerm and gets the CPU virtual address of the mapped
 * buffer by calling cheetah common interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the rmHandle is handled via
 *        LwSciBufAllocCommonTegraMemMap()
 *
 * \param[in] context Opaque allocation context.
 * \param[in] rmHandle LwSciBufRmHandle of the buffer to map.
 * \param[in] offset The starting offset of the buffer.
 *  Valid value: 0 to buffer size - 1.
 * \param[in] len The length (in bytes) of the buffer to map.
 *  Valid value: 1 to buffer size - offset.
 * \param[in] accPerm LwSciBufAttrValAccessPerm of the mapped buffer.
 *  Valid value: LwSciBufAccessPerm_Readonly <= accPerm <
 *  LwSciBufAccessPerm_Auto.
 * \param[out] ptr CPU virtual address.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem is failed to map the
 *   buffer.
 * - panics if any one of the following:
 *     - @a accPerm != LwSciBufAccessPerm_Readonly and @a accPerm !=
 *       LwSciBufAccessPerm_ReadWrite.
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a rmHandle.memHandle is 0
 *     - @a len is 0
 *     - @a ptr is NULL
 *
 * \implements{18842655}
 */
LwSciError LwSciBufSysMemMemMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void** ptr);

/**
 * \brief Unmap the mapped buffer by calling cheetah common interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the rmHandle is handled via
 *        LwSciBufAllocCommonTegraMemUnmap()
 *
 * \param[in] context Opaque allocation context.
 * \param[in] rmHandle LwSciBufRmHandle of the buffer to unmap.
 * \param[in] ptr CPU virtual address.
 *  Valid value: ptr is valid input if it is not NULL and it is returned from a
 *  successful call to LwSciBufSysMemMemMap() and has not been unmapped
 *  using LwSciBufSysMemMemUnMap().
 * @param[in] size: Length (in bytes) of the mapped buffer.
 *  Valid value: 1 to buffer size.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem failed to unmap the
 *   buffer.
 * - panics if any one of the following:
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a ptr is NULL
 *     - @a size is 0
 *     - @a rmHandle.memHandle is 0
 *
 * \implements{18842658}
 */
LwSciError LwSciBufSysMemMemUnMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size);

/**
 * @brief Gets the size of the buffer by calling cheetah common interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the rmHandle is handled via
 *        LwSciBufAllocCommonTegraGetMemSize()
 *
 * @param[in] context: Opaque allocation context.
 * @param[in] rmHandle: LwSciBufRmHandle of the buffer.
 * @param[out] size: Size of the buffer.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem failed to get the
 *   size of the buffer.
 * - panics if any one of the following:
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a size is NULL
 *     - @a rmHandle.memHandle is 0
 *
 * \implements{18842661}
 */
LwSciError LwSciBufSysMemGetSize(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size);

#if (LW_IS_SAFETY == 0)
LwSciError LwSciBufSysMemGetAlignment(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* alignment);

LwSciError LwSciBufSysMemGetHeapType(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* heapType);
#endif

/**
 * \brief Flushes the given @c len bytes of the mapped buffer by
 * calling cheetah common interface
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the rmHandle is handled via
 *        LwSciBufAllocCommonTegraCpuCacheFlush()
 *
 * \param[in] context: Opaque allocation context.
 * \param[in] rmHandle: LwSciBufRmHandle of the buffer.
 * \param[in] cpuPtr: CPU virtual address.
 *  Valid value: cpuPtr is valid input if it is not NULL and it is returned from
 *  a successful call to LwSciBufSysMemMemMap() and has not been unmapped
 *  using LwSciBufSysMemMemUnMap().
 * \param[in] len: Length (in bytes) of the buffer to flush.
 *  Valid value: 1 to buffer size.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem is failed to flush the
 *   buffer.
 * - panics if any one of the following:
 *     - @a context is NULL
 *     - @a context is invalid
 *     - @a rmHandle.memHandle is 0
 *     - @a cpuPtr is NULL
 *     - @a len is 0
 *
 * \implements{18842664}
 */
LwSciError LwSciBufSysMemCpuCacheFlush(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len);

/**
 * \brief Closes the opaque open context.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input opaque open context is not
 *        used by multiple threads at the same time when this API is called
 *
 * @param[in] context: Opaque allocation context.
 *  Valid value: context is valid input if it is not NULL and it is returned
 *  from a successful call to LwSciBufSysMemOpen() and has not been
 *  closed using LwSciBufSysMemClose().
 *
 * \return void
 *
 * \implements{18842667}
 */
void LwSciBufSysMemClose(
    void* context);

#if (LW_IS_SAFETY == 0)
void LwSciBufAllocSysMemPrintHeapTypes(
    LwSciBufAllocSysMemHeapType* heaps,
    uint32_t numHeaps);

LwSciError LwSciBufAllocSysMemPrintGpuAccessTypes(
    LwSciBufAllocSysGpuAccessType gpuAccess);
#endif

/**
 * \brief Creates an opaque allocation context.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output openContext parameter
 *        is not used by multiple threads at the same time when calling this
 *        API
 *
 * \param[in] allocContextParam: The LwSciBufAllocSysMemAllocContextParam.
 *  Valid value: allocContextParam is valid if it is non-NULL.
 * \param[in,out] openContext: Opaque open context.
 *  Valid value: openContext is valid input if it is not NULL and it is returned
 *  from a successful call to LwSciBufSysMemOpen() and has not been
 *  closed using LwSciBufSysMemClose().
 * \param[out] allocContext: New opaque allocation context.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - panics if any one of the following:
 *     - @a allocContextParam is NULL
 *     - @a openContext is NULL
 *     - @a openContext is invalid
 *     - @a allocContext is NULL
 *
 * \implements{18842643}
 */
LwSciError LwSciBufSysMemGetAllocContext(
    const void* allocContextParam,
    void* openContext,
    void** allocContext);

#endif /* INCLUDED_LWSCIBUF_ALLOC_SYSMEM_H */
