/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IOV_INL_
#define UCT_IOV_INL_

#include <uct/api/uct.h>
#include <ucs/sys/math.h>
#include <ucs/debug/assert.h>

#include <ucs/sys/iovec.h>
#include <ucs/sys/iovec.inl>


/**
 * Callwlates the total length of the particular UCT IOV data buffer.
 *
 * @param [in]     iov             Pointer to the UCT IOV element.
 *
 * @return The length of the UCT IOV data buffer.
 * @note Lwrrently has no support for the strides. If the strides are
 *       supported, it should be like: length + ((count - 1) * stride)
 */
static UCS_F_ALWAYS_INLINE
size_t uct_iov_get_length(const uct_iov_t *iov)
{
    return iov->count * iov->length;
}

/**
 * Returns the particular UCT IOV data buffer.
 *
 * @param [in]     iov             Pointer to the UCT IOV element.
 *
 * @return The UCT IOV data buffer.
 */
static UCS_F_ALWAYS_INLINE
void *uct_iov_get_buffer(const uct_iov_t *iov)
{
    return iov->buffer;
}

/**
 * Callwlates the total length of the UCT IOV array buffers.
 *
 * @param [in]     iov             Pointer to the array of UCT IOVs.
 * @param [in]     iov_cnt         Number of the elements in the array of UCT IOVs.
 *
 * @return The total length of the array of UCT IOVs.
 */
static UCS_F_ALWAYS_INLINE
size_t uct_iov_total_length(const uct_iov_t *iov, size_t iov_cnt)
{
    return ucs_iov_total_length(iov, iov_cnt, uct_iov_get_length);
}

/**
 * Callwlates the flat offset in the UCT IOV array, which is the total data size
 * before the position of the iterator.
 *
 * @param [in]     iov             Pointer to the array of UCT IOVs.
 * @param [in]     iov_cnt         Number of the elements in the array of UCT IOVs.
 * @param [in]     iov_iter        Pointer to the UCT IOV iterator.
 *
 * @return The flat offset in the UCT IOV array.
 */
static UCS_F_ALWAYS_INLINE
size_t uct_iov_iter_flat_offset(const uct_iov_t *iov, size_t iov_cnt,
                                const ucs_iov_iter_t *iov_iter)
{
    return ucs_iov_iter_flat_offset(iov, iov_cnt, iov_iter,
                                    uct_iov_get_length);
}

/**
 * Fill IOVEC data structure by the data provided in the array of UCT IOVs.
 * The function avoids copying IOVs with zero length.
 *
 * @param [out]    io_vec          Pointer to the resulted array of IOVECs.
 * @param [in/out] io_vec_cnt_p    Pointer to the varibale that holds the number
 *                                 of the elements in the array of IOVECs (input:
 *                                 initial, out: result).
 * @param [in]     uct_iov         Pointer to the array of UCT IOVs.
 * @param [in]     uct_iov_cnt     Number of the elements in the array of UCT IOVs.
 * @param [in]     max_length      Maximal total length of the data that can be
 *                                 placed in the resulted array of IOVECs.
 * @param [in]     uct_iov_iter_p  Pointer to the UCT IOV iterator.
 *
 * @return The amount, in bytes, of the data that is stored in the source
 *         array of IOVs.
 */
static UCS_F_ALWAYS_INLINE
size_t uct_iov_to_iovec(struct iovec *io_vec, size_t *io_vec_cnt_p,
                        const uct_iov_t *uct_iov, size_t uct_iov_cnt,
                        size_t max_length, ucs_iov_iter_t *uct_iov_iter_p)
{
    return ucs_iov_colwerter(io_vec, io_vec_cnt_p,
                             ucs_iovec_set_buffer, ucs_iovec_set_length,
                             uct_iov, uct_iov_cnt,
                             uct_iov_get_buffer, uct_iov_get_length,
                             max_length, uct_iov_iter_p);
}

#endif
