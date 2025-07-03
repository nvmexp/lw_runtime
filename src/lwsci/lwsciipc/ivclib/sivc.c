// Copyright (c) 2019-2020, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.

#include "sivc.h"
#include "sivc-instance.h"

#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

#include "barrier.h"

// NOTE: UINTPTR_MAX macro is not defined in quickboot

#ifndef UINTPTR_MAX
// The following inline annotation does not suppress the next MISRA 21-1
// violation with Coverity 2019.06 by some reason.
_STAN_21_1_PD_HYP_4974
#define UINTPTR_MAX ((uintptr_t)-1)
#endif

#define SIVC_ALIGN_MASK (SIVC_ALIGN - 1U)

// IVC queue reset protocol.
//
// Each end uses its send_fifo.state to indicate its synchronization state.
// Declaring as anonymous enum so that the enum values can be compared directly
// with int32_t without any explicit colwersion.
enum {
    // This value is zero for backwards compatibility with services that
    // assume queues to be initially zeroed. Such queues are in an
    // initially valid state, but cannot be asynchronously reset, and must
    // maintain a valid state at all times.
    //
    // The transmitting end can enter the established state from the SYNC or
    // ACK state when it observes the receiving endpoint in the ACK or
    // EST state, indicating that it has cleared the counters in our
    // recv_fifo.
    SIVC_STATE_EST = 0,

    // If an endpoint is observed in the SYNC state, the remote endpoint is
    // allowed to clear the counters it owns asynchronously with respect to
    // the current endpoint. Therefore, the current endpoint is no longer
    // allowed to communicate.
    SIVC_STATE_SYNC,

    // When the transmitting end observes the receiving end in the SYNC
    // state, it can clear the w_count and r_count and transition to the ACK
    // state. If the remote endpoint observes us in the ACK state, it can
    // return to the EST state once it has cleared its counters.
    SIVC_STATE_ACK
};

// This structure is divided into two-cache aligned parts, the first is only
// written through the send_fifo pointer, while the second is only written
// through the recv_fifo pointer. This delineates ownership of the cache
// lines, which is critical to performance and necessary in non-cache coherent
// implementations. The transmit fields w_count and state must be located
// in adjacent memory locations with no padding between.
// w_align and r_align fields are to make sure that Tx and Rx fields are in
// different cache lines.
struct __attribute__ ((packed)) sivc_fifo_header {
    // Fields owned by the transmitting end
    uint32_t w_count;
    int32_t state;
    uint8_t w_align[SIVC_ALIGN - sizeof(uint32_t) - sizeof(int32_t)];
    // Fields owned by the receiving end
    uint32_t r_count;
    uint8_t r_align[SIVC_ALIGN - sizeof(uint32_t)];
};

/* NOTE: offsetof macro is broken in quickboot

_Static_assert(
        (offsetof(struct sivc_fifo_header, w_count) & SIVC_ALIGN_MASK) == 0U,
        "Invalid struct sivc_fifo_header");
_Static_assert(
        (offsetof(struct sivc_fifo_header, r_count) & SIVC_ALIGN_MASK) == 0U,
        "Invalid struct sivc_fifo_header");
_Static_assert((sizeof(struct sivc_fifo_header) & SIVC_ALIGN_MASK) == 0U,
        "Invalid struct sivc_fifo_header");

 */

enum Coherency {
    UseCachedData,
    DontUseCachedData
};

// sivc_fifo_send_count() returns the number of frames that are ready to be
// read by the other endpoint. Our cached copy of r_count (owned by the other
// end) may be out-of-date. If coherency is DontUseCachedData, we will ilwalidate
// our cached copy of r_count, forcing a more up-to-date copy to be loaded.
//
// sivc_fifo_recv_count() returns the number of frames that are ready to be
// read by this endpoint. Our cached copy of w_count (owned by the other end)
// may be out-of-date. If coherency is DontUseCachedData, we will ilwalidate
// our cached copy of w_count, forcing a more up-to-date copy to be loaded.
//
// For both of these functions, the return value is an unsigned integer in the
// range [0 .. nframes] which represent the number of frames that are ready to
// be read on their respective FIFO. If the shared state between these two
// endpoints becomes corrupted, the return value may exceed nframes. This may
// indicate an attempted DOS (Denial Of Service) attack by the other end. The
// caller is responsible for detecting a return value that is out of range, and
// handling this condition accordingly.

static inline uint32_t sivc_fifo_send_count(const struct sivc_queue* queue,
        enum Coherency coherency) {
    const volatile struct sivc_fifo_header* fifo = queue->send_fifo;
    if (coherency == DontUseCachedData && queue->cache_ilwalidate != NULL) {
        queue->cache_ilwalidate(&fifo->r_count, sizeof(fifo->r_count));
    }
    // CERT INT30-C Partial Deviation HYP-4978
    return fifo->w_count - fifo->r_count;
}

static inline uint32_t sivc_fifo_recv_count(const struct sivc_queue* queue,
        enum Coherency coherency) {
    const volatile struct sivc_fifo_header* fifo = queue->recv_fifo;
    if (coherency == DontUseCachedData && queue->cache_ilwalidate != NULL) {
        queue->cache_ilwalidate(&fifo->w_count, sizeof(fifo->w_count));
    }
    // CERT INT30-C Partial Deviation HYP-4978
    return fifo->w_count - fifo->r_count;
}

static inline volatile uint8_t* sivc_fifo_frame(
        const volatile struct sivc_fifo_header* fifo,
        uint32_t frame_index, uint32_t frame_size) {
    _STAN_INT30_C_PD_HYP_4978  _STAN_11_8_PD_HYP_4971
    return (volatile uint8_t*)(fifo + 1) + frame_index * frame_size;
}

static inline int sivc_check_read(const struct sivc_queue* queue) {
    // send_fifo->state is set locally, so it is not synchronized with
    // state from the remote peer. The remote peer cannot reset its
    // transmit counters until we've acknowledged its synchronization
    // request, so no additional synchronization is required because an
    // asynchronous transition of recv_fifo->state to IVC_STATE_ACK is not
    // allowed.
    if (queue->send_fifo->state != SIVC_STATE_EST) {
        return -ECONNRESET;
    }

    // If the receive FIFO is not empty, return 0 (success).
    uint32_t count = sivc_fifo_recv_count(queue, UseCachedData);
    if (1 <= count && count <= queue->nframes) {
        return 0;
    }

    // Update w_count; data may have been recently produced by the other end.
    count = sivc_fifo_recv_count(queue, DontUseCachedData);

    if (count > queue->nframes) {
        return -EOVERFLOW;
    }
    return count == 0 ? -ENOMEM : 0;
}

static inline int sivc_check_write(const struct sivc_queue* queue) {
    if (queue->send_fifo->state != SIVC_STATE_EST) {
        return -ECONNRESET;
    }

    // If the send FIFO is less than completely full, return 0 (success).
    uint32_t count = sivc_fifo_send_count(queue, UseCachedData);
    if (count < queue->nframes) {
        return 0;
    }

    // Update r_count; data may have been recently consumed by the other end.
    count = sivc_fifo_send_count(queue, DontUseCachedData);

    if (count > queue->nframes) {
        return -EOVERFLOW;
    }
    return count == queue->nframes ? -ENOMEM : 0;
}

// The caller must subsequently call sivc_set_local_state() to flush w_count.
static inline void sivc_reset_counters(struct sivc_queue* queue) {
    // Order observation of SIVC_STATE_SYNC before stores clearing send_fifo.
    sivc_mb();

    queue->w_pos = 0;
    queue->r_pos = 0;

    // Reset send_fifo counters. The remote end is in the SYNC
    // state and won't make progress until we change our state,
    // so the counters are not in use at this time.
    queue->send_fifo->w_count = 0;
    queue->recv_fifo->r_count = 0;
    if (queue->cache_flush != NULL) {
        queue->cache_flush(&queue->recv_fifo->r_count,
                sizeof(queue->recv_fifo->r_count));
    }

    // Ensure that counters appear cleared before new state can be observed.
    sivc_wmb();
}

static inline volatile uint8_t* sivc_prepare_read(
        struct sivc_queue* queue, uint32_t offset, uint32_t length) {
    sivc_rmb();

    volatile uint8_t* frame = sivc_fifo_frame(queue->recv_fifo,
            queue->r_pos, queue->frame_size);
    if (queue->cache_ilwalidate != NULL) {
        queue->cache_ilwalidate(frame + offset, length);
    }

    return frame + offset;
}

static inline volatile uint8_t* sivc_prepare_write(
        struct sivc_queue* queue, uint32_t offset) {
    return sivc_fifo_frame(queue->send_fifo,
            queue->w_pos, queue->frame_size) + offset;
}

static inline void sivc_do_read_advance(struct sivc_queue* queue) {
    if (queue->r_pos < queue->nframes) {
        queue->r_pos++;
    }
    if (queue->r_pos >= queue->nframes) {
        queue->r_pos = 0U;
    }

    _STAN_INT30_C_PD_HYP_4978
    queue->recv_fifo->r_count++;
    if (queue->cache_flush != NULL) {
        queue->cache_flush(&queue->recv_fifo->r_count,
                sizeof(queue->recv_fifo->r_count));
    }

    // Ensure our write to r_pos oclwrs before our read from w_pos.
    sivc_mb();

    // Notify only upon transition from full to non-full.
    // The available count can only asynchronously increase, so the
    // worst possible side-effect will be a spurious notification.
    uint32_t count = sivc_fifo_recv_count(queue, DontUseCachedData);
    if (count == (queue->nframes - 1U)) {
        if (queue->notify != NULL) {
            queue->notify(queue);
        }
    }
}

static inline void sivc_do_write_advance(struct sivc_queue* queue) {
    if (queue->cache_flush != NULL) {
        volatile uint8_t* frame = sivc_fifo_frame(queue->send_fifo,
                queue->w_pos, queue->frame_size);
        queue->cache_flush(frame, queue->frame_size);
    }

    // Order any possible stores to the frame before update of w_pos.
    sivc_wmb();

    if (queue->w_pos < queue->nframes) {
        queue->w_pos++;
    }
    if (queue->w_pos >= queue->nframes) {
        queue->w_pos = 0U;
    }

    _STAN_INT30_C_PD_HYP_4978
    queue->send_fifo->w_count++;
    if (queue->cache_flush != NULL) {
        queue->cache_flush(&queue->send_fifo->w_count,
                sizeof(queue->send_fifo->w_count));
    }

    // Ensure our write to w_pos oclwrs before our read from r_pos.
    sivc_mb();

    // Notify only upon transition from empty to non-empty.
    // The available count can only asynchronously decrease, so the
    // worst possible side-effect will be a spurious notification.
    if (sivc_fifo_send_count(queue, DontUseCachedData) == 1U) {
        if (queue->notify != NULL) {
            queue->notify(queue);
        }
    }
}

// Write transmitter's state and w_count into shared memory.
// Also notify remote end about state change.
static inline void sivc_set_local_state(struct sivc_queue* queue,
        int32_t state) {
    // Order observation of peer state before storing to send_fifo.
    sivc_mb();

    queue->send_fifo->state = state;

    if (queue->cache_flush != NULL) {
        queue->cache_flush(&queue->send_fifo->w_count,
                sizeof(queue->send_fifo->w_count) +
                sizeof(queue->send_fifo->state));
    }

    // Notify remote end to observe state transition.
    if (queue->notify != NULL) {
        queue->notify(queue);
    }
}

// Copy the receiver's state out of shared memory
static inline int32_t sivc_get_remote_state(const struct sivc_queue* queue) {
    if (queue->cache_ilwalidate != NULL) {
        queue->cache_ilwalidate(&queue->recv_fifo->state,
                sizeof(queue->recv_fifo->state));
    }
    return queue->recv_fifo->state;
}

uint32_t sivc_align(uint32_t value) {
    _STAN_INT30_C_PD_HYP_4978
    return (value + SIVC_ALIGN_MASK) & ~(SIVC_ALIGN_MASK);
}

uint32_t sivc_fifo_size(uint32_t nframes, uint32_t frame_size) {
    if ((frame_size & SIVC_ALIGN_MASK) != 0U) {
        return 0U;
    }

    // Check that overall FIFO size is less than UINT32_MAX.
    if ((frame_size != 0U) && (nframes > UINT32_MAX / frame_size)) {
        return 0U;
    }
    if (UINT32_MAX - (uint32_t)sizeof(struct sivc_fifo_header) <
            nframes * frame_size) {
        return 0U;
    }

    return (uint32_t)sizeof(struct sivc_fifo_header) + nframes * frame_size;
}

uint32_t sivc_get_nframes(const struct sivc_queue* queue) {
    if (queue == NULL) {
        return 0U;
    }
    return queue->nframes;
}

uint32_t sivc_get_frame_size(const struct sivc_queue* queue) {
    if (queue == NULL) {
        return 0U;
    }
    return queue->frame_size;
}

int sivc_init(struct sivc_queue* queue,
        uintptr_t recv_base, uintptr_t send_base,
        uint32_t nframes, uint32_t frame_size, sivc_notify_function notify,
        sivc_cache_ilwalidate_function cache_ilwalidate,
        sivc_cache_flush_function cache_flush) {
    if (queue == NULL) {
        return -EILWAL;
    }

    if ((recv_base == 0) || (send_base == 0)) {
        return -EILWAL;
    }

    uintptr_t fifo_size = sivc_fifo_size(nframes, frame_size);
    if (fifo_size == 0U) {
        return -EILWAL;
    }

    // The headers must at least be aligned enough for counters
    // to be accessed atomically.
    if (((recv_base & SIVC_ALIGN_MASK) != 0U) ||
            ((send_base & SIVC_ALIGN_MASK) != 0U)) {
        return -EILWAL;
    }

    // FIFO regions must not exhibit wrap-around behavior.
    if (UINTPTR_MAX - recv_base < fifo_size) {
        return -EILWAL;
    }
    if (UINTPTR_MAX - send_base < fifo_size) {
        return -EILWAL;
    }

    // FIFO regions must not overlap.
    if (recv_base < send_base) {
        if (recv_base + fifo_size > send_base) {
            return -EILWAL;
        }
    } else {
        if (send_base + fifo_size > recv_base) {
            return -EILWAL;
        }
    }

    queue->recv_fifo = (struct sivc_fifo_header*)recv_base;
    queue->send_fifo = (struct sivc_fifo_header*)send_base;
    queue->w_pos = 0U;
    queue->r_pos = 0U;
    queue->nframes = nframes;
    queue->frame_size = frame_size;
    queue->notify = notify;
    queue->cache_ilwalidate = cache_ilwalidate;
    queue->cache_flush = cache_flush;
    return 0;
}

// Directly peek at the next frame received
const volatile void* sivc_get_read_frame(struct sivc_queue* queue) {
    if (queue == NULL) {
        return NULL;
    }

    if (sivc_check_read(queue) != 0) {
        return NULL;
    }

    return sivc_prepare_read(queue, 0U, queue->frame_size);
}

// Directly poke at the next frame to be sent
volatile void* sivc_get_write_frame(struct sivc_queue* queue) {
    if (queue == NULL) {
        return NULL;
    }

    if (sivc_check_write(queue) != 0) {
        return NULL;
    }

    return sivc_prepare_write(queue, 0U);
}

// Peek in the next receive buffer at offset 'offset', the 'size' bytes
int sivc_read_peek(struct sivc_queue* queue, void* buf,
        uint32_t offset, uint32_t size) {
    if ((queue == NULL) || (buf == NULL)) {
        return -EILWAL;
    }
    if ((UINT32_MAX - offset < size) || (offset + size > queue->frame_size)) {
        return -E2BIG;
    }

    int err = sivc_check_read(queue);
    if (err != 0) {
        return err;
    }

    _STAN_EXP32_C_PD_HYP_4977  _STAN_11_8_PD_HYP_4971  _STAN_21_15_PD_HYP_4973
    (void)memcpy(buf, (uint8_t*)sivc_prepare_read(queue, offset, size), size);

    // Note, no notification

    return 0;
}

// Poke in the next send buffer at offset 'offset, the 'size' bytes
int sivc_write_poke(struct sivc_queue* queue, const void* buf,
        uint32_t offset, uint32_t size) {
    if ((queue == NULL) || (buf == NULL)) {
        return -EILWAL;
    }
    if ((UINT32_MAX - offset < size) || (offset + size > queue->frame_size)) {
        return -E2BIG;
    }

    int err = sivc_check_write(queue);
    if (err != 0) {
        return err;
    }

    _STAN_EXP32_C_PD_HYP_4977  _STAN_11_8_PD_HYP_4971  _STAN_21_15_PD_HYP_4973
    (void)memcpy((uint8_t*)sivc_prepare_write(queue, offset), buf, size);

    // Note, no notification

    return 0;
}

int sivc_read_advance(struct sivc_queue* queue) {
    if (queue == NULL) {
        return -EILWAL;
    }

    // No read barriers or synchronization here: the caller is expected to
    // have already observed the queue non-empty. This check is just to
    // catch programming errors.
    int err = sivc_check_read(queue);
    if (err != 0) {
        return err;
    }

    sivc_do_read_advance(queue);
    return 0;
}

// Advance the tx buffer
int sivc_write_advance(struct sivc_queue* queue) {
    if (queue == NULL) {
        return -EILWAL;
    }

    int err = sivc_check_write(queue);
    if (err != 0) {
        return err;
    }

    sivc_do_write_advance(queue);
    return 0;
}

int sivc_read(struct sivc_queue* queue, void* buf, uint32_t size) {
    if ((queue == NULL) || (buf == NULL)) {
        return -EILWAL;
    }
    if (size > queue->frame_size) {
        return -E2BIG;
    }

    int err = sivc_check_read(queue);
    if (err != 0) {
        return err;
    }

    _STAN_EXP32_C_PD_HYP_4977  _STAN_11_8_PD_HYP_4971  _STAN_21_15_PD_HYP_4973
    (void)memcpy(buf, (uint8_t*)sivc_prepare_read(queue, 0U, size), size);

    sivc_do_read_advance(queue);
    return 0;
}

int sivc_write(struct sivc_queue* queue, const void* buf, uint32_t size) {
    if ((queue == NULL) || (buf == NULL)) {
        return -EILWAL;
    }
    if (size > queue->frame_size) {
        return -E2BIG;
    }

    int err = sivc_check_write(queue);
    if (err != 0) {
        return err;
    }

    volatile uint8_t* frame = sivc_prepare_write(queue, 0U);
    _STAN_EXP32_C_PD_HYP_4977  _STAN_11_8_PD_HYP_4971  _STAN_21_15_PD_HYP_4973
    (void)memcpy((uint8_t*)frame, buf, size);
    _STAN_EXP32_C_PD_HYP_4977  _STAN_11_8_PD_HYP_4971
    (void)memset((uint8_t*)frame + size, 0, (size_t)queue->frame_size - size);

    sivc_do_write_advance(queue);
    return 0;
}

bool sivc_can_read(const struct sivc_queue* queue) {
    if (queue == NULL) {
        return false;
    }
    return sivc_check_read(queue) == 0;
}

bool sivc_can_write(const struct sivc_queue* queue) {
    if (queue == NULL) {
        return false;
    }
    return sivc_check_write(queue) == 0;
}

bool sivc_is_send_fifo_empty(const struct sivc_queue* queue) {
    if (queue == NULL) {
        return false;
    }
    if (queue->send_fifo->state != SIVC_STATE_EST) {
        return false;
    }
    return sivc_fifo_send_count(queue, DontUseCachedData) == 0;
}

void sivc_reset(struct sivc_queue* queue) {
    if (queue == NULL) {
        return;
    }
    sivc_set_local_state(queue, SIVC_STATE_SYNC);
}

int sivc_sync(struct sivc_queue* queue) {
    if ((queue == NULL) || (queue->nframes == 0U)) {
        return -EILWAL;
    }
    queue->w_pos = queue->send_fifo->w_count % queue->nframes;
    queue->r_pos = queue->recv_fifo->r_count % queue->nframes;
    return 0;
}

// IVC state transition table
//
// | Peer | Local | Local Actions                       |
// |------|-------|-------------------------------------|
// | EST  | EST   | <none>                              |
// | EST  | SYNC  | <none>                              |
// | EST  | ACK   |                 move to EST; notify |
// | SYNC | EST   | reset counters; move to ACK; notify |
// | SYNC | SYNC  | reset counters; move to ACK; notify |
// | SYNC | ACK   | reset counters;              notify |
// | ACK  | EST   | <none>                              |
// | ACK  | SYNC  | reset counters; move to EST; notify |
// | ACK  | ACK   |                 move to EST; notify |

// Notification is performed inside sivc_set_local_state()

// Returns 0 when local state is EST, otherwise return -EAGAIN

int sivc_notified(struct sivc_queue* queue) {
    if (queue == NULL) {
        return -EILWAL;
    }

    int32_t peer_state = sivc_get_remote_state(queue);
    int32_t local_state = queue->send_fifo->state;

    if ((local_state == SIVC_STATE_ACK) &&
            ((peer_state == SIVC_STATE_EST) || (peer_state == SIVC_STATE_ACK))) {
        // Move to EST state. We know that we have previously
        // cleared our counters, and we know that the remote end has
        // cleared its counters, so it is safe to start writing/reading
        // on this queue.
        sivc_set_local_state(queue, SIVC_STATE_EST);
    } else if (peer_state == SIVC_STATE_SYNC) {
        sivc_reset_counters(queue);

        // Move to ACK state unconditionally. We have just cleared our counters,
        // so it is now safe for the remote end to start using these values.
        sivc_set_local_state(queue, SIVC_STATE_ACK);
    } else if ((peer_state == SIVC_STATE_ACK) &&
            (local_state == SIVC_STATE_SYNC)) {
        sivc_reset_counters(queue);

        // Move to EST state. We know that the remote end has
        // already cleared its counters, so it is safe to start
        // writing/reading on this queue.
        sivc_set_local_state(queue, SIVC_STATE_EST);
    } else {
        ; // No action required
    }

    return (queue->send_fifo->state == SIVC_STATE_EST) ? 0 : -EAGAIN;
}

bool sivc_need_notify(const struct sivc_queue* queue) {
    if (queue == NULL) {
        return false;
    }
    if (queue->send_fifo->state != SIVC_STATE_EST) {
        return true;
    }
    return sivc_get_remote_state(queue) != SIVC_STATE_EST;
}
