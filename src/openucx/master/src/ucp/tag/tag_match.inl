/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_MATCH_INL_
#define UCP_TAG_MATCH_INL_

#include "tag_match.h"
#include "eager.h"

#include <ucp/tag/offload.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/dt/dt.h>
#include <ucs/debug/log.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/mpool.inl>
#include <inttypes.h>


/* Hash size is a prime number just below 1024. Prime number for even distribution,
 * and small enough to fit L1 cache. */
#define UCP_TAG_MATCH_HASH_SIZE     1021


static UCS_F_ALWAYS_INLINE
int ucp_tag_is_specific_source(ucp_context_t *context, ucp_tag_t tag_mask)
{
    return ((context->config.tag_sender_mask & tag_mask) ==
            context->config.tag_sender_mask);
}

static UCS_F_ALWAYS_INLINE
int ucp_tag_is_match(ucp_tag_t tag, ucp_tag_t exp_tag, ucp_tag_t tag_mask)
{
    /* The bits in which expected and actual tag differ, should not fall
     * inside the mask.
     */
    return ((tag ^ exp_tag) & tag_mask) == 0;
}

static UCS_F_ALWAYS_INLINE size_t
ucp_tag_match_calc_hash(ucp_tag_t tag)
{
    /* Compute two 32-bit modulo and combine their result */
    return ((uint32_t)tag % UCP_TAG_MATCH_HASH_SIZE) ^
           ((uint32_t)(tag >> 32) % UCP_TAG_MATCH_HASH_SIZE);
}

static UCS_F_ALWAYS_INLINE ucp_request_queue_t*
ucp_tag_exp_get_queue_for_tag(ucp_tag_match_t *tm, ucp_tag_t tag)
{
    return &tm->expected.hash[ucp_tag_match_calc_hash(tag)];
}

static UCS_F_ALWAYS_INLINE ucp_request_queue_t*
ucp_tag_exp_get_queue(ucp_tag_match_t *tm, ucp_tag_t tag, ucp_tag_t tag_mask)
{
    if (tag_mask == UCP_TAG_MASK_FULL) {
        return ucp_tag_exp_get_queue_for_tag(tm, tag);
    } else {
        return &tm->expected.wildcard;
    }
}

static UCS_F_ALWAYS_INLINE ucp_request_queue_t*
ucp_tag_exp_get_req_queue(ucp_tag_match_t *tm, ucp_request_t *req)
{
    return ucp_tag_exp_get_queue(tm, req->recv.tag.tag, req->recv.tag.tag_mask);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_exp_push(ucp_tag_match_t *tm, ucp_request_queue_t *req_queue,
                 ucp_request_t *req)
{
    req->recv.tag.sn = tm->expected.sn++;
    ucs_queue_push(&req_queue->queue, &req->recv.queue);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_exp_add(ucp_tag_match_t *tm, ucp_request_t *req)
{
    ucp_tag_exp_push(tm, ucp_tag_exp_get_req_queue(tm, req), req);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_exp_delete(ucp_request_t *req, ucp_tag_match_t *tm,
                   ucp_request_queue_t *req_queue, ucs_queue_iter_t iter)
{
    if (!(req->flags & UCP_REQUEST_FLAG_OFFLOADED)) {
        --tm->expected.sw_all_count;
        --req_queue->sw_count;
        if (req->flags & UCP_REQUEST_FLAG_BLOCK_OFFLOAD) {
            --req_queue->block_count;
        }
    }
    ucs_queue_del_iter(&req_queue->queue, iter);
}

static UCS_F_ALWAYS_INLINE ucp_request_t *
ucp_tag_exp_search(ucp_tag_match_t *tm, ucp_tag_t tag)
{
    ucp_request_queue_t *req_queue;
    ucs_queue_iter_t iter;
    ucp_request_t *req;

    if (ucs_unlikely(!ucs_queue_is_empty(&tm->expected.wildcard.queue))) {
        req_queue = ucp_tag_exp_get_queue_for_tag(tm, tag);
        return ucp_tag_exp_search_all(tm, req_queue, tag);
    }

    /* fast path - wildcard queue is empty, search only the specific queue */
    req_queue = ucp_tag_exp_get_queue_for_tag(tm, tag);
    ucs_queue_for_each_safe(req, iter, &req_queue->queue, recv.queue) {
        req = ucs_container_of(*iter, ucp_request_t, recv.queue);
        ucs_trace_data("checking req %p tag %"PRIx64"/%"PRIx64" with tag %"PRIx64,
                       req, req->recv.tag.tag, req->recv.tag.tag_mask, tag);
        if (ucp_tag_is_match(tag, req->recv.tag.tag, req->recv.tag.tag_mask)) {
            ucs_trace_req("matched received tag %"PRIx64" to req %p", tag, req);
            ucp_tag_exp_delete(req, tm, req_queue, iter);
            return req;
        }
    }
    return NULL;
}

static UCS_F_ALWAYS_INLINE ucp_tag_t ucp_rdesc_get_tag(ucp_recv_desc_t *rdesc)
{
    return ((ucp_tag_hdr_t*)(rdesc + 1))->tag;
}

static UCS_F_ALWAYS_INLINE ucs_list_link_t*
ucp_tag_unexp_get_list_for_tag(ucp_tag_match_t *tm, ucp_tag_t tag)
{
    return &tm->unexpected.hash[ucp_tag_match_calc_hash(tag)];
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_unexp_remove(ucp_recv_desc_t *rdesc)
{
    ucs_list_del(&rdesc->tag_list[UCP_RDESC_HASH_LIST]);
    ucs_list_del(&rdesc->tag_list[UCP_RDESC_ALL_LIST] );
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_unexp_recv(ucp_tag_match_t *tm, ucp_recv_desc_t *rdesc, ucp_tag_t tag)
{
    ucs_list_link_t *hash_list;

    hash_list = ucp_tag_unexp_get_list_for_tag(tm, tag);
    ucs_list_add_tail(hash_list,           &rdesc->tag_list[UCP_RDESC_HASH_LIST]);
    ucs_list_add_tail(&tm->unexpected.all, &rdesc->tag_list[UCP_RDESC_ALL_LIST]);

    ucs_trace_req("unexp "UCP_RECV_DESC_FMT" tag %"PRIx64,
                  UCP_RECV_DESC_ARG(rdesc), tag);
}

static UCS_F_ALWAYS_INLINE ucp_recv_desc_t*
ucp_tag_unexp_list_next(ucp_recv_desc_t *rdesc, int i_list)
{
    return ucs_list_next(&rdesc->tag_list[i_list], ucp_recv_desc_t,
                         tag_list[i_list]);
}

/* search unexpected queue for tag/mask, if found return the received desc,
 * otherwise return NULL
 */
static UCS_F_ALWAYS_INLINE ucp_recv_desc_t*
ucp_tag_unexp_search(ucp_tag_match_t *tm, ucp_tag_t tag, uint64_t tag_mask,
                     int rem, const char *title)
{
    ucp_recv_desc_t *rdesc;
    ucs_list_link_t *list;
    int i_list;

    /* fast check of global unexpected queue */
    if (ucs_list_is_empty(&tm->unexpected.all)) {
        return NULL;
    }

    if (tag_mask == UCP_TAG_MASK_FULL) {
        list = ucp_tag_unexp_get_list_for_tag(tm, tag);
        if (ucs_list_is_empty(list)) {
            return NULL;
        }
        i_list = UCP_RDESC_HASH_LIST;
    } else {
        list   = &tm->unexpected.all;
        i_list = UCP_RDESC_ALL_LIST;
    }

    rdesc = ucs_list_head(list, ucp_recv_desc_t, tag_list[i_list]);
    do {
        ucs_trace_req("searching for tag %"PRIx64"/%"PRIx64" "
                      "checking "UCP_RECV_DESC_FMT" tag %"PRIx64,
                      tag, tag_mask, UCP_RECV_DESC_ARG(rdesc),
                      ucp_rdesc_get_tag(rdesc));
        if (ucp_tag_is_match(ucp_rdesc_get_tag(rdesc), tag, tag_mask)) {
            ucs_trace_req("matched unexp rdesc " UCP_RECV_DESC_FMT " to "
                          "%s tag %"PRIx64"/%"PRIx64, UCP_RECV_DESC_ARG(rdesc),
                          title, tag, tag_mask);
            if (rem) {
                ucp_tag_unexp_remove(rdesc);
            }
            return rdesc;
        }

        rdesc = ucp_tag_unexp_list_next(rdesc, i_list);
    } while (&rdesc->tag_list[i_list] != list);

    return NULL;
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_recv_request_release_non_contig_buffer(ucp_request_t *req)
{
    ucs_assert(!UCP_DT_IS_CONTIG(req->recv.datatype));
    ucs_free(req->recv.tag.non_contig_buf);
    req->recv.tag.non_contig_buf = NULL;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_recv_offload_data(ucp_request_t *req, const void *data,
                              size_t length, unsigned recv_flags)
{
    ucs_status_t status = UCS_OK;
    size_t offset;
    ucp_offload_last_ssend_hdr_t *priv;

    /* Should be used in multi-fragmented flow only */
    ucs_assert(!(recv_flags & UCP_RECV_DESC_FLAG_EAGER_ONLY));

    if (ucs_test_all_flags(recv_flags, UCP_RECV_DESC_FLAG_EAGER_LAST |
                                       UCP_RECV_DESC_FLAG_EAGER_SYNC)) {
        priv = (ucp_offload_last_ssend_hdr_t*)UCS_PTR_BYTE_OFFSET(data,
                                                                  -sizeof(*priv));
        ucp_tag_offload_sync_send_ack(req->recv.worker, priv->ssend_ack.ep_ptr,
                                      priv->ssend_ack.sender_tag, recv_flags);
    }

    if (ucs_unlikely(req->status != UCS_OK)) {
       return req->status;
    }

    /* There is no correct offset in middle headers with tag offload flow.
     * All fragments are always in order - can callwlate offset by
     * subtraction of already received data.
     * NOTE: total length of unexpected eager offload message is not known
     * until last fragment arrives, so it is initialized to SIZE_MAX. */
    offset = SIZE_MAX - req->recv.tag.remaining;

    if (ucs_unlikely(req->recv.length < (length + offset))) {
        /* We have to release non-contig buffer only in case of 
         * this is not the first segment and the datatype is
         * non-contig */
        if ((offset != 0) && !UCP_DT_IS_CONTIG(req->recv.datatype)) {
            ucp_tag_recv_request_release_non_contig_buffer(req);
        }
        return ucp_request_recv_msg_truncated(req, length, offset);
    }

    if (UCP_DT_IS_CONTIG(req->recv.datatype)) {
        ucp_request_unpack_contig(req,
                                  UCS_PTR_BYTE_OFFSET(req->recv.buffer, offset),
                                  data, length);
    } else {
        /* For non-contig data need to assemble the whole message
         * before calling unpack. */
        if (offset == 0) {
            req->recv.tag.non_contig_buf = ucs_malloc(req->recv.length,
                                                      "tag gen buffer");
            if (ucs_unlikely(req->recv.tag.non_contig_buf == NULL)) {
               return UCS_ERR_NO_MEMORY;
            }
        }

        ucp_request_unpack_contig(req,
                                  UCS_PTR_BYTE_OFFSET(req->recv.tag.non_contig_buf,
                                                      offset),
                                  data, length);
    }

    if (recv_flags & UCP_RECV_DESC_FLAG_EAGER_LAST) {
        /* Need to update recv info length. In tag offload protocol we do not
         * know the total message length until the last fragment arrives. */
        req->recv.tag.info.length = offset + length;

        if (!UCP_DT_IS_CONTIG(req->recv.datatype)) {
            status = ucp_request_recv_data_unpack(req, req->recv.tag.non_contig_buf,
                                                  req->recv.tag.info.length,
                                                  0, 1);
            ucp_tag_recv_request_release_non_contig_buffer(req);
        }
    }

    return status;
}

/*
 * process data, complete receive if done
 * @return UCS_OK/ERR - completed, UCS_INPROGRESS - not completed
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_request_process_recv_data(ucp_request_t *req, const void *data,
                                  size_t length, size_t offset, int dereg,
                                  unsigned recv_flags)
{
    ucs_status_t status;
    int last;

    if (recv_flags & UCP_RECV_DESC_FLAG_EAGER_OFFLOAD) {
        req->status = ucp_request_recv_offload_data(req, data, length,
                                                    recv_flags);

        last = recv_flags & UCP_RECV_DESC_FLAG_EAGER_LAST;
    } else {
        last = req->recv.tag.remaining == length;

        /* process data only if the request is not in error state */
        if (ucs_likely(req->status == UCS_OK)) {
            req->status = ucp_request_recv_data_unpack(req, data, length,
                                                       offset, last);
        }
        ucs_assertv(req->recv.tag.remaining >= length,
                    "req->recv.tag.remaining=%zu length=%zu",
                    req->recv.tag.remaining, length);
    }

    req->recv.tag.remaining -= length;

    if (last) {
        status = req->status;
        if (dereg) {
            ucp_request_recv_buffer_dereg(req);
        }
        ucp_request_complete_tag_recv(req, status);
        ucs_assert(status != UCS_INPROGRESS);
        return status;
    } else {
        return UCS_INPROGRESS;
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_recv_request_process_rdesc(ucp_request_t *req, ucp_recv_desc_t *rdesc,
                                   size_t offset)
{
     size_t hdr_len, recv_len;
     ucs_status_t status;

     hdr_len  = rdesc->payload_offset;
     recv_len = rdesc->length - hdr_len;
     status = ucp_tag_request_process_recv_data(req,
                                                UCS_PTR_BYTE_OFFSET(rdesc + 1, hdr_len),
                                                recv_len, offset, 0, rdesc->flags);
     ucp_recv_desc_release(rdesc);
     return status;
}

static UCS_F_ALWAYS_INLINE int
ucp_tag_frag_match_is_unexp(ucp_tag_frag_match_t *frag_list)
{
    /* Hack to reduce memory usage: instead of adding another field to specify
     * which union field is valid, assume that when the unexpected queue field
     * is valid, its ptail field could never be NULL */
    return frag_list->unexp_q.ptail != NULL;
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_frag_match_add_unexp(ucp_tag_frag_match_t *frag_list, ucp_recv_desc_t *rdesc,
                         size_t offset)
{
    ucs_trace_req("unexp frag "UCP_RECV_DESC_FMT" offset %zu",
                  UCP_RECV_DESC_ARG(rdesc), offset);
    ucs_assert(ucp_tag_frag_match_is_unexp(frag_list));
    ucs_queue_push(&frag_list->unexp_q, &rdesc->tag_frag_queue);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_frag_match_init_unexp(ucp_tag_frag_match_t *frag_list)
{
    ucs_queue_head_init(&frag_list->unexp_q);
    ucs_assert(ucp_tag_frag_match_is_unexp(frag_list));
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_frag_hash_init_exp(ucp_tag_frag_match_t *frag_list, ucp_request_t *req)
{
    UCS_STATIC_ASSERT(ucs_offsetof(ucp_tag_frag_match_t, unexp_q.ptail) >=
                      ucs_offsetof(ucp_tag_frag_match_t, exp_req) + sizeof(frag_list->exp_req));
    frag_list->exp_req       = req;
    frag_list->unexp_q.ptail = NULL;
    ucs_assert(!ucp_tag_frag_match_is_unexp(frag_list));
}

#endif
