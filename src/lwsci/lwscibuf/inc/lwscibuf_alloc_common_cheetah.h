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
 * \brief <b>LwSciBuf CheetAh Common Allocate Interfaces</b>
 *
 * @b Description: This file contains cheetah common APIs for memory operations
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_COMMON_TERGA_H
#define INCLUDED_LWSCIBUF_ALLOC_COMMON_TERGA_H

#include "lwscibuf.h"
#include "lwscibuf_dev.h"
#include "lwscierror.h"
#include "lwscilog.h"


/**
 * @addtogroup lwscibuf_blanket_statements
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements
 * \section lwscibuf_element_dependency Dependency on other elements
 * LwSciBuf calls below liblwrm_mem interfaces:
 * - LwRmMemHandleAllocAttr() to allocate the buffer.
 * - LwRmMemHandleFree() to deallocate the buffer.
 * - LwRmMemHandleDuplicate() to duplicate the LwRmMemHandle with specified LwSciBufAttrValAccessPerm.
 * - LwRmMemMap() to map the buffer to cpu virtual address.
 * - LwRmMemUnmap() to unmap the buffer from cpu virtual address.
 * - LwRmMemQueryHandleParams() to get size of the buffer.
 * - LwRmMemCacheSyncForDevice() to flush the mapped buffer.
 * LwSciBuf calls below LwSciCommon interfaces:
 * - LwSciCommonPanic() to abort program unexpected internal error.
 *
 * \section lwscibuf_in_params Input parameters
 * - LwRmMemHandle is a valid input if it is obtained from successful call to
 *   LwRmMemHandleAllocAttr() and has not been freed using LwRmMemHandleFree() or it is
 *   retrieved from valid LwSciBufRmHandle().
 */

/**
 * @}
 */

/**
 * \brief Colwerts LwSciBuf defined coherency into
 * LwOsMemAttribute understood by liblwrm_mem.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data dependency)
 *
 * \param[in] lwSciBufCoherency LwSciBuf defined coherency.
 *  Valid value: true or false.
 * \param[out] lwRmCoherency The colwerted buffer coherency attribute to
 * liblwrm_mem.
 *
 * \return void
 * - panics if lwRmCoherency is NULL
 *
 * \implements{18843582}
 */
void LwSciBufAllocCommonTegraColwertCoherency(
    bool lwSciBufCoherency,
    LwOsMemAttribute* lwRmCoherency);

/**
 * \brief Deallocates the buffer by calling liblwrm_mem interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwRmMemHandle is handled via
 *        LwRmMemHandleFree()
 *
 * \param[in] memHandle LwRmMemHandle of the buffer.
 *
 * \return void
 * - panics if memHandle is 0.
 *
 * \implements{18843585}
 */
void LwSciBufAllocCommonTegraMemFree(
    LwRmMemHandle memHandle);

/**
 * \brief Duplicates the LwRmMemHandle with specified LwSciBufAttrValAccessPerm
 * from input LwRmMemHandle by calling liblwrm_mem interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwRmMemHandle is handled via
 *        LwRmMemHandleDuplicate()
 *
 * \param[in] memHandle LwRmMemHandle to be duplicated.
 * \param[in] newPerm New LwSciBufAttrValAccessPerm.
 *  Valid input: LwSciBufAccessPerm_Readonly <= newPerm <
 *  LwSciBufAccessPerm_Auto.
 * \param[out] dupHandle Duplicated LwRmMemHandle.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem failed.
 * - panics if any one of the following:
 *   - @a newPerm != LwSciBufAccessPerm_Readonly and @a newPerm !=
 *     LwSciBufAccessPerm_ReadWrite.
 *   - @a memHandle is 0.
 *   - @a dupHandle is NULL.
 *
 * \implements{18843588}
 */
LwSciError LwSciBufAllocCommonTegraDupHandle(
    LwRmMemHandle memHandle,
    LwSciBufAttrValAccessPerm newPerm,
    LwRmMemHandle* dupHandle);

/**
 * \brief Maps the buffer at the given offset and given length with the
 * specified LwSciBufAttrValAccessPerm and gets the CPU virtual address of
 * the mapped buffer by calling liblwrm_mem interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwRmMemHandle is handled via LwRmMemMap()
 *
 * \param[in] memHandle LwRmMemHandle of the buffer to map.
 * \param[in] offset The starting offset of the buffer.
 *  Valid value: 0 to buffer size - 1.
 * \param[in] len The length (in bytes) of the buffer to map.
 *  Valid value: 1 to buffer size - offset.
 * \param[in] accPerm LwSciBufAttrValAccessPerm of the mapped buffer.
 *  Valid input: LwSciBufAccessPerm_Readonly <= accPerm <
 *  LwSciBufAccessPerm_Auto.
 * \param[out] ptr CPU virtual address.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem failed to map the buffer.
 * - panics if any one of the following:
 *   - @a accPerm != LwSciBufAccessPerm_Readonly and @a accPerm !=
 *     LwSciBufAccessPerm_ReadWrite
 *   - @a memHandle is 0
 *   - @a len is 0
 *   - @a ptr is NULL
 *
 * \implements{18843591}
 */
LwSciError LwSciBufAllocCommonTegraMemMap(
    LwRmMemHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void** ptr);

/**
 * \brief Unmap the mapped buffer by calling liblwrm_mem interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwRmMemHandle is handled via LwRmMemUnmap()
 *
 * \param[in] memHandle LwRmMemHandle of the buffer to unmap.
 * \param[in] ptr CPU virtual address.
 *  Valid value: ptr is valid input if it is not NULL and it is returned from a
 *  successful call to LwSciBufAllocCommonTegraMemMap() or os-specific map function
 *  and has not been unmapped using LwSciBufAllocCommonTegraMemUnmap() or os specific unmap function.
 * @param[in] size: Length (in bytes) of the mapped buffer.
 *  Valid value: 1 to buffer size.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem failed to unmap the buffer.
 * - panics if any one of the following:
 *   - @a memHandle is 0
 *   - @a ptr is NULL
 *   - @a size is 0
 *
 * \implements{18843594}
 */
LwSciError LwSciBufAllocCommonTegraMemUnmap(
    LwRmMemHandle memHandle,
    void* ptr,
    uint64_t size);

/**
 * @brief Gets the size of the buffer by calling liblwrm_mem interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwRmMemHandle is handled via
 *        LwRmMemQueryHandleParams()
 *
 * @param[in] memHandle: LwRmMemHandle of the buffer.
 * @param[out] size: Size of the buffer.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if liblwrm_mem failed to get the
 *   size of the buffer.
 * - panics if any one of the following:
 *   - @a memHandle is 0
 *   - @a size is NULL
 *
 * \implements{18843597}
 */
LwSciError LwSciBufAllocCommonTegraGetMemSize(
    LwRmMemHandle memHandle,
    uint64_t* size);

/**
 * \brief Flushes the given @c len bytes of the mapped buffer by calling liblwrm_mem interface.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwRmMemHandle is handled via
 *        LwRmMemCacheSyncForDevice()
 *
 * \param[in] memHandle: LwRmMemHandle of the buffer.
 * \param[in] ptr: CPU virtual address.
 *  Valid value: ptr is valid input if it is not NULL and it is returned from
 *  a successful call to LwSciBufAllocCommonTegraMemMap() or os-specific map function
 *  and has not been unmapped using LwSciBufAllocCommonTegraMemUnmap() or os-specific unmap function.
 * \param[in] len: Length (in bytes) of the buffer to flush.
 *  Valid value: 1 to buffer size.
 *
 * \return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if any one of the following:
 *   - liblwrm_mem failed to flush the buffer.
 *   - @a len is more than the buffer size.
 * - panics if any one of the following:
 *   - @a memHandle is 0
 *   - @a ptr is NULL
 *   - @a len is 0
 *
 * \implements{18843600}
 */
LwSciError LwSciBufAllocCommonTegraCpuCacheFlush(
    LwRmMemHandle memHandle,
    void* ptr,
    uint64_t len);

#endif /* INCLUDED_LWSCIBUF_ALLOC_COMMON_TERGA_H */
