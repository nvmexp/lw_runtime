/*
 * lwscicommon_transportutils.h
 *
 * Utility Functions for LwSci* Transport
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCICOMMON_TRANSPORTUTILS_H
#define INCLUDED_LWSCICOMMON_TRANSPORTUTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "lwscierror.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup lwscicommon_blanket_statements LwSciCommon blanket statements.
 * Generic statements applicable for LwSciCommon interfaces.
 * @{
 */

/**
 * \page page_blanket_statements LwSciCommon blanket statements
 * \section in_out_params Input/Output parameters
 * - LwSciCommonTransportBuf* passed as an input parameter to an API is a valid input
 *   if it is returned from a successful call to
 *   LwSciCommonTransportAllocTxBufferForKeys or
 *   LwSciCommonTransportGetRxBufferAndParams and not yet been deallocated
 *   using LwSciCommonTransportBufferFree.
 */

/**
 * @}
 */

/**
 * @defgroup lwscicommon_transportutils_api LwSciCommon APIs for transport utils.
 * List of APIs exposed at Inter-Element level.
 * @{
 */

/**
 * \brief Structure contains header information for LwSciCommonTransportBuf*.
 *  This information is filled by exporter side using
 *  LwSciCommonTransportAllocTxBufferForKeys API and is intended to be verified
 *  for compatibility check on importer side after using
 *  LwSciCommonTransportGetRxBufferAndParams API.
 *
 * \implements{21751550}
 * \implements{21755929}
 * \implements{18851127}
 * \implements{18850773}
 */
typedef struct {
    /** Version of the message.
     *  This member is initialized by the user before ilwoking
     *  LwSciCommonTransportAllocTxBufferForKeys API. */
    uint64_t msgVersion;
    /** Magic cookie for sanity check of message.
     *  This member is initialized by the user before ilwoking
     *  LwSciCommonTransportAllocTxBufferForKeys API. */
    uint32_t msgMagic;
    /** Number of key-value pairs encoded in message.
     *  In LwSciCommonTransportAllocTxBufferForKeys API keyCount represents
     *  max number of keys exporter intends to store.
     *  In LwSciCommonTransportGetRxBufferAndParams API keyCount represents
     *  max number of keys stored by exporter */
    uint32_t keyCount;
} LwSciCommonTransportParams;

/**
 * \brief LwSciCommonTransportBuf* represents container for holding key-value
 *  pairs which are exported/imported between two entities. It also stores
 *  information regarding the message version for compatibility checks across
 *  entities which export and import data.
 *  LwSciCommonTransportBuf* also maintains a read pointer internally which keeps
 *  track of next key-value pair to be read when
 *  LwSciCommonTransportGetRxBufferAndParams API is ilwoked.
 *  LwSciCommonTransportBuf* maintains a write pointer which keeps
 *  track of position where next key-value pair will be appended when
 *  LwSciCommonTransportAppendKeyValuePair is ilwoked.
 *
 */
typedef struct LwSciCommonTransportRec LwSciCommonTransportBuf;

/**
 * \brief Creates a new LwSciCommonTransportBuf* which has the capacity to hold
 *  key-value pairs of input provided size. The newly created
 *  LwSciCommonTransportBuf* stores LwSciCommonTransportParams passed by the
 *  user.
 *
 * \param[in] bufParams header information passed by the user which is stored
 *  in LwSciCommonTransportBuf*.
 *  Valid value: keyCount inside @a bufParams is [1, UINT32_MAX].
 * \param[in] totalValueSize max capacity of LwSciCommonTransportBuf* in bytes
 *  to hold key-value pairs.
 *  Valid value: @a totalValueSize is [1, SIZE_MAX]
 * \param[out] txbuf newly created LwSciCommonTransportBuf*.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - Panics if any of the following oclwrs:
 *      - @a totalValueSize is 0
 *      - keyCount inside @a bufParams is 0
 *      - @a txbuf is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same output @a txbuf parameter is not
 *        used by multiple threads at the same time.
 *
 * \implements{18851142}
 */
LwSciError LwSciCommonTransportAllocTxBufferForKeys(
    LwSciCommonTransportParams bufParams,
    size_t totalValueSize,
    LwSciCommonTransportBuf** txbuf);

/**
 * \brief Appends key-value pair in LwSciCommonTransportBuf* if there is capacity
 *  to hold key-value pair and decrements the remaining capacity of
 *  LwSciCommonTransportBuf* by key-value size.
 *
 * \param[in] txbuf LwSciCommonTransportBuf* to which the input key-value pair
 *  will be appended.
 * \param[in] key numerical value of key which needs to be stored.
 *  Valid value: @a key is [0, UINT32_MAX].
 * \param[in] length size of value which needs to be stored.
 *  Valid value: @a length is [1, SIZE_MAX].
 * \param[in] value pointer to the value which needs to be copied in
 *  LwSciCommonTransportBuf*.
 *  Valid value: @a value is not NULL.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_Overflow if internal arithmetic overflow oclwrs.
 * - LwSciError_NoSpace if no space is left in the LwSciCommonTransportBuf*
 *    to append the key-value pair.
 * - Panics if any of the following oclwrs:
 *      - @a length is 0
 *      - @a value is NULL
 *      - @a txbuf is NULL
 *      - @a txbuf is not a valid LwSciCommonTransportBuf*
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *      - Conlwrrent modification of the @a txbuf to append the input
 *        key-value pair is handled via LwSciCommonMemcpyS().
 *      - The user must ensure the input @a txbuf not accessed by other
 *        threads at the same time.
 *
 * \implements{18851148}
 */
LwSciError LwSciCommonTransportAppendKeyValuePair(
    LwSciCommonTransportBuf* txbuf,
    uint32_t key,
    size_t length,
    const void* value);

/**
 * \brief Serializes LwSciCommonTransportBuf* into binary buffer. It also
 *  computes and embeds a checksum value using Cyclic Redundancy Check (CRC)
 *  method from the contents of the LwSciCommonTransportBuf*.
 *
 * \param[in] txbuf LwSciCommonTransportBuf* which needs to be serialized.
 * \param[out] descBufPtr pointer to output serialized binary buffer.
 *  This serialized binary buffer needs to be freed using LwSciCommonFree only.
 * \param[out] descBufSize pointer to size of serialized binary buffer.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a descBufPtr is NULL
 *      - @a descBufSize is NULL
 *      - @a txbuf is NULL
 *      - @a txbuf is not a valid LwSciCommonTransportBuf*
 *      - not all the allocated storage for key-value-pairs was used
 *      - not all the allocated storage for values of key-value-pairs was used
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *      - The user must ensure the input @a txbuf not accessed by other
 *        threads at the same time.
 *
 * \implements{18851157}
 */
void LwSciCommonTransportPrepareBufferForTx(
    LwSciCommonTransportBuf* txbuf,
    void** descBufPtr,
    size_t* descBufSize);

/**
 * \brief Validates data integrity of the input serialized binary buffer by
 *  re-computing checksum value using same Cyclic Redundancy Check (CRC) method
 *  used in LwSciCommonTransportPrepareBufferForTx API and matching the computed
 *  checksum value with the checksum value present in the input serialized
 *  binary buffer. Once verified the input serialized binary buffer is
 *  deserialized into LwSciCommonTransportBuf* and LwSciCommonTransportParams.
 *
 * \param[in] bufPtr pointer to serialized binary buffer.
 *  Valid value: @a bufPtr is not NULL and is a valid serialized binary buffer
 *  returned from LwSciCommonTransportPrepareBufferForTx API.
 * \param[in] bufSize size of serialized binary buffer in bytes.
 *  Valid value: [1, SIZE_MAX].
 * \param[out] rxbuf pointer to deserialized LwSciCommonTransportBuf*.
 * \param[out] params pointer to LwSciCommonTransportParams.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufPtr is NULL
 *      - @a bufPtr is invalid
 *      - @a bufSize is 0
 *      - computed checksum value does not match checksum value stored in
 *        @a bufPtr
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - Panics if any of the following oclwrs:
 *      - @a rxbuf is NULL
 *      - @a params is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *      - The user must ensure the input @a bufPtr not modified by other
 *        threads at the same time.
 *
 * \implements{18851145}
 */
LwSciError LwSciCommonTransportGetRxBufferAndParams(
    const void* bufPtr,
    size_t bufSize,
    LwSciCommonTransportBuf** rxbuf,
    LwSciCommonTransportParams* params);

/**
 * \brief Retrieves next key-value pair from LwSciCommonTransportBuf*.
 * If the retrieved key-value pair is last key-value pair, an output finish flag
 *  is set to true and LwSciCommonTransportBuf* is reset to retrieve the first
 * key-value pair on subsequent call to LwSciCommonTransportGetNextKeyValuePair
 * API.
 *
 * \param[in] rxbuf LwSciCommonTransportBuf* from which next key-value pair is to be
 *  retrieved.
 * \param[out] key pointer to a retrieved key value.
 * \param[out] length pointer to length of the retrieved value in bytes.
 * \param[out] value pointer to a retrieved value.
 * \param[out] rdFinish boolean to indicate the retrieved key-value pair is the
 *  last key-value pair in LwSciCommonTransportBuf*.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_Overflow if internal arithmetic overflow oclwrs.
 * - Panics if any of the following oclwrs:
 *      - @a rdFinish is NULL
 *      - @a rxbuf is NULL
 *      - @a key is NULL
 *      - @a length is NULL
 *      - @a value is NULL
 *      - @a rxbuf is not valid LwSciCommonTransportBuf*
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *      - The user must ensure the input @a rxbuf not accessed by other
 *        threads at the same time.
 *
 * \implements{18851154}
 */
LwSciError LwSciCommonTransportGetNextKeyValuePair(
    LwSciCommonTransportBuf* rxbuf,
    uint32_t* key,
    size_t* length,
    const void** value,
    bool* rdFinish);

/**
 * \brief Deallocates LwSciCommonTransportBuf* previously allocated using
 *  LwSciCommonTransportAllocTxBufferForKeys API so that it is no longer usable.
 *
 * \param[in] buf pointer to buffer which needs to be deallocated.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a buf is NULL
 *      - @a buf is not valid
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *      - The user must ensure that no active operation on object to be freed.
 *      - The user must ensure that two threads are not freeing the same object.
 *      - Conlwrrent deallocatoin of the @a bufPtr is handled via
 *        LwSciCommonFree().
 *
 * \implements{18851151}
 */
void LwSciCommonTransportBufferFree(
    LwSciCommonTransportBuf* buf);

/**
 * @}
 */

#if (LW_IS_SAFETY == 0)
/**
 * \brief Prints binary content of LwSciCommonTransportBuf*.
 *
 * \param[in] buf pointer to buffer object.
 *
 * \return void
 *
 */
void LwSciCommonTransportDumpBuffer(
    LwSciCommonTransportBuf* buf);
#endif

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCICOMMON_TRANSPORTUTILS_H */
