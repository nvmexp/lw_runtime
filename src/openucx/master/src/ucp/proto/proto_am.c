/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_am.inl"

#include <ucp/tag/offload.h>


static inline size_t ucp_proto_max_packed_size()
{
    return ucs_max(sizeof(ucp_reply_hdr_t),
                   sizeof(ucp_offload_ssend_hdr_t));
}

static size_t ucp_proto_pack(void *dest, void *arg)
{
    ucp_request_t *req = arg;
    ucp_reply_hdr_t *rep_hdr;
    ucp_offload_ssend_hdr_t *off_rep_hdr;

    switch (req->send.proto.am_id) {
    case UCP_AM_ID_EAGER_SYNC_ACK:
    case UCP_AM_ID_RNDV_ATS:
    case UCP_AM_ID_RNDV_ATP:
        rep_hdr = dest;
        rep_hdr->reqptr = req->send.proto.remote_request;
        rep_hdr->status = req->send.proto.status;
        return sizeof(*rep_hdr);
    case UCP_AM_ID_OFFLOAD_SYNC_ACK:
        off_rep_hdr = dest;
        off_rep_hdr->sender_tag = req->send.proto.sender_tag;
        off_rep_hdr->ep_ptr     = ucp_request_get_dest_ep_ptr(req);
        return sizeof(*off_rep_hdr);
    }

    ucs_fatal("unexpected am_id");
    return 0;
}

ucs_status_t
ucp_do_am_single(uct_pending_req_t *self, uint8_t am_id,
                 uct_pack_callback_t pack_cb, ssize_t max_packed_size)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;
    uint64_t *buffer;

    /* if packed data can fit short active message, use it, because it should
     * be faster than bcopy.
     */
    if ((max_packed_size <= UCS_ALLOCA_MAX_SIZE) &&
        (max_packed_size <= ucp_ep_config(ep)->am.max_short)) {
        req->send.lane = ucp_ep_get_am_lane(ep);
        buffer         = ucs_alloca(max_packed_size);
        packed_len     = pack_cb(buffer, req);
        ucs_assertv((packed_len >= 0) && (packed_len <= max_packed_size),
                    "packed_len=%zd max_packed_size=%zu", packed_len,
                    max_packed_size);

        return uct_ep_am_short(ep->uct_eps[req->send.lane], am_id, buffer[0],
                               &buffer[1], packed_len - sizeof(uint64_t));
    } else {
        return ucp_do_am_bcopy_single(self, am_id, pack_cb);
    }
}

ucs_status_t ucp_proto_progress_am_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status = ucp_do_am_single(self, req->send.proto.am_id,
                                           ucp_proto_pack,
                                           ucp_proto_max_packed_size());
    if (status == UCS_OK) {
        req->send.proto.comp_cb(req);
    }
    return status;
}

void ucp_proto_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(req->send.state.uct_comp.count == 0);
    ucp_request_send_buffer_dereg(req); /* TODO register+lane change */
    ucp_request_complete_send(req, status);
}

void ucp_proto_am_zcopy_completion(uct_completion_t *self,
                                    ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    if (req->send.state.dt.offset == req->send.length) {
        ucp_proto_am_zcopy_req_complete(req, status);
    } else if (status != UCS_OK) {
        ucs_assert(req->send.state.uct_comp.count == 0);
        ucs_assert(status != UCS_INPROGRESS);

        /* NOTE: the request is in pending queue if data was not completely sent,
         *       just dereg the buffer here and complete request on purge
         *       pending later.
         */
        ucp_request_send_buffer_dereg(req);
        req->send.state.uct_comp.func = NULL;
    }
}
