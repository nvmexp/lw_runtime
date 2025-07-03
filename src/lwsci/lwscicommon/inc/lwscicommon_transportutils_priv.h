/*
 * lwscicommon_transportutils_priv.h
 *
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCICOMMON_TRANSPORTUTILS_PRIV_H
#define INCLUDED_LWSCICOMMON_TRANSPORTUTILS_PRIV_H

#include <string.h>
#include <stdlib.h>

#include "lwscicommon_transportutils.h"
#include "lwscicommon_utils.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscilog.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Structure that contains information related to transport object.
 *  It stores state of iterator used to read and write key value pair in
 *  transport object. This structure is allocated, initialized by
 *  LwSciCommonTransportAllocTxBufferForKeys,
 *  LwSciCommonTransportGetRxBufferAndParams and de-initialized, freed by
 *  LwSciCommonTransportBufferFree.
 *
 */
/* This item needs to be manually synced to Jama ID 18851163 */
struct LwSciCommonTransportRec {
    /** Pointer to container where the key value pair is stored. */
    void* bufPtr;
    /** Magic ID to detect if this LwSciCommonTransportRec is valid.
     * This member must be initialized to a particular non-zero constant.
     * It must be changed to a different value when this LwSciCommonTransportRec
     *  is freed.
     * This member must NOT be modified in between allocation and deallocation
     *  of the LwSciCommonTransportRec.
     * Whenever a transport utilities unit API is called with
     *  LwSciCommonTransportRec as input from outside the unit,
     *  the transport utilities unit must validate the magic ID.
     */
    uint32_t magic;
    /** Counter to keep track of max keys to be stored in the transport object.*/
    uint32_t allocatedKeyCount;
    /** Counter to keep track of number of keys read. */
    uint32_t rdKeyCount;
    /** Counter to keep track of number of keys written. */
    uint32_t wrKeyCount;
    /** Total capacity of the transport object requested by user. */
    size_t sizeAllocated;
    /** Counter to keep track of bytes read. */
    size_t sizerd;
    /** Counter to keep track of bytes written. */
    size_t sizewr;
};
typedef struct LwSciCommonTransportRec LwSciCommonTransportBufPriv;

/**
 * \brief Structure that contains information related to serialized binary
 *  buffer. This structure is initialized by
 *  LwSciCommonTransportAllocTxBufferForKeys,
 *  LwSciCommonTransportGetRxBufferAndParams and freed by
 *  LwSciCommonTransportBufferFree.
 *
 */
/* This item needs to be manually synced to Jama ID 21239046 */
typedef struct {
    /** Stores checksum of the binary buffer. This value is computed and set
     *  by LwSciCommonTransportPrepareBufferForTx.*/
    uint64_t checksum;
    /** Magic number provided by user. */
    uint32_t msgMagic;
    /** Version of message provided by user. */
    uint64_t version;
    /** Size of serialized binary buffer. */
    uint64_t size;
    /** Number of keys stored in serialized binary buffer. */
    uint32_t keyCount;
    /** Pointer to key value pair data.
     */
    /* Sized due to MISRA violation fix for rule 18.7.
     *  Do not rely on the size of payload. */
    uint8_t payload[1];
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciCommon-ADV-MISRAC2012-001")
} __attribute__((packed)) LwSciCommonTransportHeader;

/**
 * \brief Key length value structure to store key value pair in serialized
 *  binary buffer.
 *
 */
typedef struct {
    /** Numerical value of key to be stored. */
    uint32_t key;
    /** Length of value to be stored. */
    uint64_t length;
    /** Value data to be stored.
     * Sized due to MISRA violation fix for rule 18.7
     * Do not rely on the size of value.
     */
    uint8_t value[1];
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciCommon-ADV-MISRAC2012-001")
} __attribute__((packed)) LwSciCommonTransportKey;

/**
 * Need this macro as the size of payload and value is declared as 1.
 */
#define ADJUSTED_SIZEOF(x) (sizeof(x) - sizeof(uint8_t))

/**
 * @defgroup lwscicommon_transportutils_api LwSciCommon APIs for transport utils.
 * List of APIs exposed at Inter-Element level.
 * @{
 */

/**
 * LwSciCommonTransportPrepareBufferForTx API uses
 * IEEE-802.3 CRC32 Ethernet Standard to generate 32-bit checksum value.
 *
 * \implements{18851157}
 *
 * \fn void LwSciCommonTransportPrepareBufferForTx(
 *  LwSciCommonTransportBuf txbuf,
 *  void** descBufPtr,
 *  size_t* descBufSize);
 */

/**
 * LwSciCommonTransportGetRxBufferAndParams API uses
 * IEEE-802.3 CRC32 Ethernet Standard to generate 32-bit checksum value.
 *
 * \implements{18851145}
 *
 * \fn LwSciError LwSciCommonTransportGetRxBufferAndParams(
 *  const void* bufPtr,
 *  size_t bufSize,
 *  LwSciCommonTransportBuf* rxbuf,
 *  LwSciCommonTransportParams* params);
 */

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCICOMMON_TRANSPORTUTILS_PRIV_H */
