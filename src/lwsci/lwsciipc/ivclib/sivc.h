// Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.

#ifndef SIVC_H
#define SIVC_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
namespace Ivc {
extern "C" {
#endif

struct sivc_queue;

/**
 * @brief Get the number of frames in the IVC queue.
 * @param[in] queue IVC queue endpoint to inspect.
 * @return Number of frames per FIFO/direction. If @p queue is NULL, function
 * returns 0.
 */
uint32_t sivc_get_nframes(const struct sivc_queue* queue);

/**
 * @brief Get the IVC queue frame size.
 * @param[in] queue IVC queue endpoint to inspect.
 * @return Frame size in bytes. If @p queue is NULL, function returns 0.
 */
uint32_t sivc_get_frame_size(const struct sivc_queue* queue);

/**
 * @brief Get a pointer to the current incoming frame.
 * @param[in] queue IVC queue endpoint from which to read.
 *
 * Returns a pointer to the start of the current incoming frame if one
 * is available, but does not release it from the front of the receive FIFO.
 * To release this frame call @c sivc_read_advance, after which do not use
 * the returned pointer. Between these two calls, do not perform any other
 * IVC read operations on this IVC queue endpoint. Do not write through the
 * returned pointer.
 *
 * @c sivc_read is generally recommended over this function. However, this
 * function is better suited for some cirlwmstances.
 *
 * Use this function to avoid an extra copy. For example, when decoding
 * incoming data as a stream (e.g., one byte at a time), use this function
 * to obviate a temporary buffer.
 *
 * Use this function when shared memory has atypical access requirements that
 * are not satisfied by @c memcpy (e.g., specifically-sized load instructions
 * required).
 *
 * Use this function when employing a DMA engine.
 *
 * @note Conlwrrency: This function is considered an IVC read operation, and
 * as such may be called conlwrrently with IVC write operations on the same
 * IVC queue endpoint. Read operation exclusivity should be retained until
 * after calling @c sivc_read_advance.
 *
 * @warning Security: The pointer returned by this function may potentially
 * reference memory shared with another active process. Care must be taken to
 * avoid security vulnerabilities introduced by malicious senders. A safe
 * practice is to read each memory location only once, thus avoiding the
 * race condition known as TOCTTOU (time of check to time of use).
 *
 * @return Pointer to the current incoming frame on success, or NULL on
 * failure.
 *
 * Failure oclwrs if:
 *
 * @li @p queue is NULL
 * @li Queue reset is in progress
 * @li Receive FIFO is empty
 * @li Shared data is corrupted
 */
const volatile void* sivc_get_read_frame(struct sivc_queue* queue);

/**
 * @brief Get a pointer to the current outgoing frame.
 * @param[in] queue IVC queue endpoint into which caller will write.
 *
 * Returns a pointer to the start of the current outgoing frame if one
 * is available, allowing the caller to construct the outgoing message in-place.
 * This function, while analogous to @c sivc_get_read_frame, is generally safer
 * than the later. Do not read through the returned pointer.
 *
 * Once the outgoing message has been fully written, commit the message by
 * calling @c sivc_write_advance, after which do not use the returned pointer.
 * Between calling this function and calling @c sivc_write_advance,
 * do not perform any other IVC write operations on this IVC queue endpoint.
 *
 * This function is partilwlarly useful when frames are allocated to memory
 * with atypical access requirements not satisfied by @c memcpy (e.g.,
 * alignment restrictions or special instructions to access), or must be
 * passed to a DMA engine for optimal throughput.
 *
 * @note Conlwrrency: This function is considered an IVC write operation, and
 * as such may be called conlwrrently with IVC read operations on the same
 * IVC queue endpoint. Write operation exclusivity should be retained until
 * after calling @c sivc_write_advance.
 *
 * @return Pointer to the current outgoing frame on success, or NULL on
 * failure.
 *
 * Failure oclwrs if:
 *
 * @li @p queue is NULL
 * @li Queue reset is in progress
 * @li Transmit FIFO is full, no available frame in which to write
 * @li Shared data is corrupted
 */
volatile void* sivc_get_write_frame(struct sivc_queue* queue);

/**
 * @brief Perform partial read from the current incoming frame.
 * @param[in] queue IVC queue endpoint from which to read.
 * @param[out] buf Buffer to receive the data.
 * @param[in] offset Offset in bytes from start of frame.
 * @param[in] size Number of bytes to read.
 *
 * Copy @p size bytes into @p buf from the current incoming frame starting
 * @p offset bytes from the beginning of the frame. Unlike @c sivc_read,
 * this function does not release the frame from the front of the receive FIFO.
 *
 * @warning Security: The receive frame may be located in memory shared with
 * another active process. Care must be taken to avoid security vulnerabilities
 * introduced by malicious senders. A safe practice is to read each memory
 * location only once, thus avoiding the race condition known as TOCTTOU
 * (time of check to time of use). Any data read a second time must be
 * re-checked.
 *
 * @note Conlwrrency: This function is considered an IVC read operation, and
 * as such may be called conlwrrently with IVC write operations on the same
 * IVC queue endpoint.
 *
 * @warning Memory: This function should not be used on an IVC queue whose
 * shared memory cannot be accessed by @c memcpy with normal load
 * instructions and arbitrary access sizes. Consider using
 * @c sivc_get_read_frame to have direct access to the incoming frame.
 *
 * @deprecated This function is deprecated and may be removed in a future
 * release. Use @c sivc_get_write_frame and @c memcpy instead.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue or @p buf is NULL
 * @li @c -E2BIG     @p offset is greater than the IVC queue frame size
 * @li @c -E2BIG     Sum of @p offset and @p size is greater than the IVC
 * queue frame size
 * @li @c -ECONRESET Queue reset is in progress
 * @li @c -ENOMEM    Receive FIFO is empty
 * @li @c -EOVERFLOW Shared state is corrupted: sivc_reset recommended.
 */
int sivc_read_peek(struct sivc_queue* queue, void* buf,
        uint32_t offset, uint32_t size);

/**
 * @brief Perform partial write to the current outgoing frame.
 * @param[in] queue IVC queue endpoint into which to write.
 * @param[in] buf Buffer containing data to write.
 * @param[in] offset Offset in bytes from start of frame.
 * @param[in] size Number of bytes to write.
 *
 * Copy @p size bytes from @p buf into the current outgoing frame starting
 * at offset @p offset. Unlike @c sivc_write, this function does not advance
 * the transmit FIFO to the next frame. The remainder of the outgoing frame
 * remains uninitialized.
 *
 * @note Conlwrrency: This function is considered an IVC write operation, and
 * as such may be called conlwrrently with IVC read operations on the same
 * IVC queue endpoint.
 *
 * @warning Memory: This function should not be used on an IVC queue whose
 * shared memory cannot be accessed by @c memcpy with normal store
 * instructions and arbitrary access sizes. Consider using
 * @c sivc_get_write_frame to have direct access to the outgoing frame.
 *
 * @deprecated This function is deprecated and may be removed in a future
 * release. Use @c sivc_get_write_frame and @c memcpy instead.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue or @p buf is NULL
 * @li @c -E2BIG     @p offset is greater than the IVC queue frame size
 * @li @c -E2BIG     Sum of @p offset and @p size is greater than the
 * IVC queue frame size
 * @li @c -ECONRESET Queue reset is in progress
 * @li @c -ENOMEM    Transmit FIFO is full
 * @li @c -EOVERFLOW Shared state is corrupted: sivc_reset recommended.
 */
int sivc_write_poke(struct sivc_queue* queue, const void* buf,
        uint32_t offset, uint32_t size);

/**
 * @brief Advance the receive FIFO.
 * @param[in] queue IVC queue endpoint to update.
 *
 * Acknowledge receipt of the current incoming frame and advance the receive
 * FIFO to the next frame. This releases the current frame to be reused by
 * the sender. Any pointer previously returned by @c sivc_get_read_frame
 * for this IVC queue endpoint becomes invalid.
 *
 * @note Conlwrrency: This function is considered an IVC read operation, and
 * as such may be called conlwrrently with IVC write operations on the same
 * IVC queue endpoint.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue is NULL
 * @li @c -ECONRESET Queue reset is in progress
 * @li @c -ENOMEM    Receive FIFO is empty
 * @li @c -EOVERFLOW Shared state is corrupted: sivc_reset recommended.
 */
int sivc_read_advance(struct sivc_queue* queue);

/**
 * @brief Advance the transmit FIFO.
 * @param[in] queue IVC queue endpoint to update.
 *
 * Commit the current outgoing frame and advance the transmit FIFO to the
 * next frame. If the transmit FIFO becomes full, the next outgoing frame
 * will not be available until the other end acknowledges receipt of
 * at least one frame. Any pointer previously returned by
 * @c sivc_get_write_frame for this IVC queue endpoint becomes invalid.
 *
 * @note Conlwrrency: This function is considered an IVC write operation, and
 * as such may be called conlwrrently with IVC read operations on the same
 * IVC queue endpoint.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue is NULL
 * @li @c -ECONRESET Queue reset is in progress
 * @li @c -ENOMEM    Transmit FIFO is full
 * @li @c -EOVERFLOW Shared state is corrupted: sivc_reset recommended.
 */
int sivc_write_advance(struct sivc_queue* queue);

/**
 * @brief Read a frame from the IVC queue.
 * @param[in] queue IVC queue endpoint from which to read.
 * @param[out] buf Buffer into which data shall be copied.
 * @param[in] size Number of bytes to read.
 *
 * Copy @p size bytes into @p buf from the current incoming frame, and
 * advance the receive FIFO to the next frame.
 *
 * @note Conlwrrency: This function is considered an IVC read operation, and
 * as such may be called conlwrrently with IVC write operations on the same
 * IVC queue endpoint.
 *
 * @warning Memory: This function should not be used on an IVC queue whose
 * shared memory cannot be accessed by @c memcpy with normal load
 * instructions and arbitrary access sizes. Consider using
 * @c sivc_get_read_frame to have direct access to the incoming frame.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue or @p buf is NULL
 * @li @c -E2BIG     @p size is greater than the queue frame size
 * @li @c -ECONRESET Queue reset is in progress
 * @li @c -ENOMEM    Queue is empty, no data to read
 * @li @c -EOVERFLOW Shared state is corrupted: sivc_reset recommended.
 */
int sivc_read(struct sivc_queue* queue, void* buf, uint32_t size);

/**
 * @brief Write a frame to the IVC queue.
 * @param[in] queue IVC queue endpoint into which to write.
 * @param[in] buf   Pointer to the data to write.
 * @param[in] size  Number of bytes to write.
 *
 * Copy @p size bytes from @p buf into the current outgoing frame and
 * advance the transmit FIFO to the next frame. If @p size is less than
 * the frame size, the remainder of the outgoing frame will be zero-filled.
 *
 * @note Conlwrrency: This function is considered an IVC write operation, and
 * as such may be called conlwrrently with IVC read operations on the same
 * IVC queue endpoint.
 *
 * @warning Memory: This function should not be used on an IVC queue whose
 * shared memory cannot be accessed by @c memcpy with normal store
 * instructions and arbitrary access sizes. Consider using
 * @c sivc_get_write_frame to have direct access to the outgoing frame.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue or @p buf is NULL
 * @li @c -E2BIG     @p size is greater than the queue frame size
 * @li @c -ECONRESET Queue reset is in progress
 * @li @c -ENOMEM    Queue is full, no data can be written
 * @li @c -EOVERFLOW Shared state is corrupted: sivc_reset recommended.
 */
int sivc_write(struct sivc_queue* queue, const void* buf, uint32_t size);

/**
 * @brief Test whether data are available for reading.
 * @param[in] queue IVC queue endpoint to inspect.
 *
 * Determine whether data to read are available.
 *
 * @note Conlwrrency: The results of this call are valid until the next
 * read operation on this IVC queue endpoint, or the next write operation on
 * the other endpoint.
 *
 * @return @c true if the receive FIFO is not empty, and @c false if
 * the FIFO is empty, @p queue is NULL, or if a reset is in progress.
 */
bool sivc_can_read(const struct sivc_queue* queue);

/**
 * @brief Test whether data can be written.
 * @param[in] queue IVC queue to inspect.
 *
 * Determine whether data can be written.
 *
 * @note Conlwrrency: The results of this call are valid until the next
 * write operation on this IVC queue endpoint, or the next read operation on
 * the other endpoint.
 *
 * @return @c true if the transmit FIFO has room to write another frame, and
 * @c false if @p queue is NULL, the transmit FIFO is full, or if a reset
 * is in progress.
 */
bool sivc_can_write(const struct sivc_queue* queue);

/**
 * @brief Test whether the transmit FIFO is empty.
 * @param[in] queue IVC queue to inspect.
 *
 * Determine whether the transmit FIFO is completely empty.
 *
 * @note Conlwrrency: The results of this call are valid until the next
 * write operation on this IVC queue endpoint, or the next read operation on
 * the other endpoint.
 *
 * @return @c true if the transmit FIFO is empty, and @c false if the FIFO is
 * not empty, @p queue is NULL, or a reset is in progress.
 */
bool sivc_is_send_fifo_empty(const struct sivc_queue* queue);

/**
 * @brief Initiate an IVC queue reset
 * @param[in] queue IVC queue to reset.
 *
 * This function initiates an IVC queue reset. The reset does not complete
 * immediately. Both ends of the IVC queue must make repeated calls to
 * @c sivc_notified in order to make progress. Only one side needs to
 * call @c sivc_reset.
 *
 * @c sivc_reset must be called as part of initialization, before reading or
 * writing will succeed. It may also be called to re-establish a connection
 * (e.g., when one side is asynchronously restarted). Once @c sivc_notified
 * returns success (@c 0), normal read and write operations may be performed.
 *
 * @note Conlwrrency: This function is considered both an IVC read and an IVC
 * write operation, and as such <em>may not</em> be called conlwrrently with
 * any IVC read or IVC write operations on the same IVC queue. The only
 * functions which may be called are
 *
 * @li @c sivc_can_read
 * @li @c sivc_can_write
 * @li @c sivc_is_send_fifo_empty
 * @li @c sivc_need_notify
 * @li @c sivc_notified
 */
void sivc_reset(struct sivc_queue* queue);

/**
 * @brief Re-synchronize an IVC queue across a reboot.
 * @param[in] queue IVC queue to synchronize.
 *
 * This function uses shared state to re-establish synchronization with
 * the other end of the IVC queue.
 *
 * @deprecated This function is deprecated, and will be removed in a future
 * release. Use @c sivc_reset instead.
 *
 * @warning This interface exists to support legacy code, but is considered
 * too fragile to use in new code. Legacy code should migrate towards using
 * @c sivc_reset instead. The conditions necessary for @c sivc_sync to work
 * correctly are diffilwlt to ensure:
 *
 * @li If this function is used in lieu of @c sivc_reset, then the shared
 * memory used for the transmit and receive FIFO headers must be initialized
 * to all zeros before use.
 * @li The number of frames in this queue must be a power of two, or the
 * number of messages sent and the numbers of messages received since the
 * last reset must both be less than 2^32.
 * @li The application-level protocol layered on top of this queue must be
 * completely stateless, or all protocol transactions must be allowed to
 * complete (or otherwise reach a stable state) before calling this function.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue is NULL
 * @li @c -EILWAL    Number of frames in queue is 0
 */
int sivc_sync(struct sivc_queue* queue);

/**
 * @brief Handle incoming notifications from the other IVC queue endpoint.
 * @param[in,out] queue IVC queue receiving a notification.
 *
 * This function must be called at least once for every incoming notification
 * from the other IVC queue endpoint, unless @c sivc_need_notify indicates
 * otherwise. Calling this function (@c sivc_notified) more frequently may
 * result in degraded aggregate performance.
 *
 * @note Conlwrrency: This function is considered both an IVC read and an IVC
 * write operation, and as such <em>may not</em> be called conlwrrently with
 * any IVC read or IVC write operations on the same IVC queue endpoint except:
 *
 * @li @c sivc_can_read
 * @li @c sivc_can_write
 * @li @c sivc_is_send_fifo_empty
 * @li @c sivc_need_notify
 *
 * @return 0 when the connection is successfully established, or a negative
 * error value on failure.
 *
 * Failure oclwrs if:
 *
 * @li @c -EILWAL    @p queue is NULL
 * @li @c -EAGAIN    Queue reset is in progress
 */
int sivc_notified(struct sivc_queue* queue);

/**
 * @brief Determine if @c sivc_notified should be called for a given
 * notification.
 * @param[in] queue IVC queue to test.
 *
 * A notification may occur for one of three reasons:
 *
 * 1. A connection reset is in progress.
 * 2. The transmit FIFO is no longer full.
 * 3. The receive FIFO is no longer empty.
 *
 * Some exelwtion elwironments require taking extra steps to ensure
 * compliance with the conlwrrency rules of the SIVC API (for example,
 * acquire a read-lock, a write-lock, or both).
 *
 * Use this function to determine if a connection reset is in progress.
 * This may aid in reducing the overhead necessary for compliance.
 *
 * @note Conlwrrency: The results of this call are valid until the next
 * invocation of @c sivc_reset or @c sivc_notified.
 *
 * @return @c true if @c sivc_notified must be called, and @c false
 * otherwise. This function also returns @c false if @p queue is NULL.
 */
bool sivc_need_notify(const struct sivc_queue* queue);

#ifdef __cplusplus
} // extern "C"
} // namespace Ivc
#endif

#endif  // SIVC_H
