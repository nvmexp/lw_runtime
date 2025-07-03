// Copyright (c) 2019-2020, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.

#ifndef SIVC_INSTANCE_H
#define SIVC_INSTANCE_H

#include <stddef.h>
#include <stdint.h>

#include "ivclib-static-analysis.h"

#ifdef __cplusplus
namespace Ivc {
extern "C" {
#endif

#define SIVC_ALIGN_SHIFT 6U
#define SIVC_ALIGN       (1U << SIVC_ALIGN_SHIFT)

/**
 * @brief Align a number.
 * @param[in] value An unsigned integer
 *
 * @ilwariant The caller must ensure that @p value is less than or equal to
 * (UINT32_MAX - SIVC_ALIGN + 1).
 *
 * @return The closest integer that is greater than or equal to @p value and
 * divisible by SIVC_ALIGN.
 */
uint32_t sivc_align(uint32_t value);

/**
 * @brief Callwlate size of the memory needed for IVC fifo.
 * @param[in] nframes    Number of IVC queue frames
 * @param[in] frame_size Size of one frame in bytes
 *
 * @return Number of bytes needed for IVC fifo memory area, or 0 if fails.
 *
 * Function fails if:
 * @li @p frame_size is not a multiple of SIVC_ALIGN.
 * @li Computed IVC FIFO size exceeds UINT32_MAX
 */
uint32_t sivc_fifo_size(uint32_t nframes, uint32_t frame_size);

struct sivc_queue;
struct sivc_fifo_header;

typedef void (*sivc_notify_function)(struct sivc_queue* queue);
typedef void (*sivc_cache_ilwalidate_function)(const volatile void* addr,
        size_t size);
typedef void (*sivc_cache_flush_function)(const volatile void* addr,
        size_t size);

/* WARNING: This structure is a part of private IVC interface.
 *          You should not access any of its member directly. Please use helper
 *          functions instead.
 *          See sivc_get_nframes() and sivc_get_frame_size().
 */
struct sivc_queue {
    _STAN_A9_6_1_FP_HYP_4836
    volatile struct sivc_fifo_header* recv_fifo;
    _STAN_A9_6_1_FP_HYP_4836
    volatile struct sivc_fifo_header* send_fifo;
    uint32_t w_pos;
    uint32_t r_pos;
    uint32_t nframes;
    uint32_t frame_size;
    _STAN_A9_6_1_FP_HYP_4836
    sivc_notify_function notify;
    _STAN_A9_6_1_FP_HYP_4836
    sivc_cache_ilwalidate_function cache_ilwalidate;
    _STAN_A9_6_1_FP_HYP_4836
    sivc_cache_flush_function cache_flush;
};

/**
 * @brief  Initialize IVC queue control structure.
 * @param[in] queue            IVC queue
 * @param[in] recv_base        Shared memory address of receive IVC FIFO
 * @param[in] send_base        Shared memory address of transmit IVC FIFO
 * @param[in] nframes          Number of frames in a queue
 * @param[in] frame_size       Frame size in bytes
 * @param[in] notify           Notification callback, can be NULL
 * @param[in] cache_ilwalidate Memory cache ilwalidation callback, can be NULL
 * @param[in] cache_flush      Memory cache flush callback, can be NULL
 *
 * IVC queue control structure is considered to be private, even though is
 * is declared in public header. This function should be used to set it up.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * The caller is responsible for ensuring that both IVC queue endpoints
 * are compatibly initialized. Specifically,
 *
 * @li @p recv_base must correspond to the same underlying physical memory
 * as @p send_base for the other IVC queue endpoint.
 * @li @p send_base must correspond to the same underlying physical memory
 * as @p recv_base for the other IVC queue endpoint.
 * @li @p nframes must be identical for both IVC queue endpoints.
 * @li @p frame_size must be identical for both IVC queue endpoints.
 *
 * Function fails if:
 * @li @c -EILWAL    The @p queue is NULL
 * @li @c -EILWAL    @p recv_base or @p send_base are zero
 * @li @c -EILWAL    @p recv_base or @p send_base are not aligned to SIVC_ALIGN
 * @li @c -EILWAL    @p frame_size is not aligned to SIVC_ALIGN
 * @li @c -EILWAL    Receive FIFO memory area and send FIFO memory area overlap
 * @li @c -EILWAL    Expected IVC FIFO size is bigger than 2^32
 */
int sivc_init(struct sivc_queue* queue,
        uintptr_t recv_base, uintptr_t send_base,
        uint32_t nframes, uint32_t frame_size, sivc_notify_function notify,
        sivc_cache_ilwalidate_function cache_ilwalidate,
        sivc_cache_flush_function cache_flush);

#ifdef __cplusplus
} // extern "C"
} // namespace Ivc
#endif

#endif // SIVC_INSTANCE_H
