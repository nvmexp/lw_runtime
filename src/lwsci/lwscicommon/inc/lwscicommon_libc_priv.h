/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCICOMMON_LIBC_PRIV_H
#define INCLUDED_LWSCICOMMON_LIBC_PRIV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Constant value used to initialize magic member of
 *  LwSciCommonAllocHeader. This header is used to identify if the memory was
 *  allocated using LwSciCommonCalloc.
 *
 */
#define LWSCICOMMON_ALLOC_MAGIC 0x10293847U

/**
 * \brief A header structure containing information of system memory allocated
 *  using LwSciCommonCalloc.
 *  This structure is allocated, initialized by LwSciCommonCalloc and
 *  de-initialized, freed by LwSciCommonFree.
 *
 */
/* This item needs to be manually synced to Jama ID 21529454 */
typedef struct {
    /** Magic ID to detect if this LwSciCommonAllocHeader is valid.
     * This member must be initialized to a particular non-zero constant.
     * It must be changed to a different value when this LwSciCommonAllocHeader
     *  is de-initialized.
     * This member must NOT be modified in between allocation and deallocation
     *  of the LwSciCommonAllocHeader.
     * LwSciCommonFree API must validate the magic ID to ensure that the input
     *  system memory was allocated using LwSciCommonCalloc API.
     */
    uint32_t magic;
    /** Size of the memory requested by the user. When LwSciCommonFree API is
     *  called, @a allocSize is used to zero out memory before releasing it to
     *  Operating System.
     */
    uint64_t allocSize;
} LwSciCommonAllocHeader;

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCICOMMON_LIBC_PRIV_H */
