/* -*- Mode: C; c-basic-offset:3 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2011-2017 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2011      UT-Battelle, LLC. All rights reserved.
 * Copyright (c) 2014      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      Intel, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "btl_ugni.h"
#include "btl_ugni_frag.h"
#include "btl_ugni_smsg.h"
#include "btl_ugni_prepare.h"

void mca_btl_ugni_wait_list_append (mca_btl_ugni_module_t *ugni_module, mca_btl_base_endpoint_t *endpoint,
                                    mca_btl_ugni_base_frag_t *frag)
{
    BTL_VERBOSE(("wait-listing fragment %p to %s. endpoint state %d\n", (void*)frag, OPAL_NAME_PRINT(endpoint->peer_proc->proc_name), endpoint->state));

    frag->base.des_flags |= MCA_BTL_DES_SEND_ALWAYS_CALLBACK;

    /* queue up request */
    OPAL_THREAD_LOCK(&endpoint->lock);

    opal_list_append (&endpoint->frag_wait_list, (opal_list_item_t *) frag);

    OPAL_THREAD_UNLOCK(&endpoint->lock);

    if (false == endpoint->wait_listed && MCA_BTL_UGNI_EP_STATE_CONNECTED == endpoint->state) {
        OPAL_THREAD_LOCK(&ugni_module->ep_wait_list_lock);
        if (false == endpoint->wait_listed) {
            opal_list_append (&ugni_module->ep_wait_list, &endpoint->super);
            endpoint->wait_listed = true;
        }
        OPAL_THREAD_UNLOCK(&ugni_module->ep_wait_list_lock);
    }
}

int mca_btl_ugni_send (struct mca_btl_base_module_t *btl,
                       struct mca_btl_base_endpoint_t *endpoint,
                       struct mca_btl_base_descriptor_t *descriptor,
                       mca_btl_base_tag_t tag)
{
    mca_btl_ugni_base_frag_t *frag = (mca_btl_ugni_base_frag_t *) descriptor;
    size_t size = frag->segments[0].seg_len + frag->segments[1].seg_len;
    mca_btl_ugni_module_t *ugni_module = (mca_btl_ugni_module_t *) btl;
    int rc;

    /* tag and len are at the same location in eager and smsg frag hdrs */
    frag->hdr.send.lag = (tag << 24) | size;

    BTL_VERBOSE(("btl/ugni sending descriptor %p from %d -> %d. length = %" PRIu64, (void *)descriptor,
                 OPAL_PROC_MY_NAME.vpid, endpoint->peer_proc->proc_name.vpid, size));

    rc = mca_btl_ugni_check_endpoint_state (endpoint);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc || opal_list_get_size (&endpoint->frag_wait_list))) {
        mca_btl_ugni_wait_list_append (ugni_module, endpoint, frag);
        return OPAL_SUCCESS;
    }

    /* add a reference to prevent the fragment from being returned until after the
     * completion flag is checked. */
    ++frag->ref_cnt;
    frag->flags &= ~MCA_BTL_UGNI_FRAG_COMPLETE;

    rc = mca_btl_ugni_send_frag (endpoint, frag);
    if (OPAL_LIKELY(mca_btl_ugni_frag_check_complete (frag))) {
        /* fast path: remote side has received the frag */
        (void) mca_btl_ugni_frag_del_ref (frag, OPAL_SUCCESS);

        return 1;
    }

    if ((OPAL_SUCCESS == rc) && (frag->flags & MCA_BTL_UGNI_FRAG_BUFFERED) && (frag->flags & MCA_BTL_DES_FLAGS_BTL_OWNERSHIP)) {
        /* fast(ish) path: btl owned buffered frag. report send as complete */
        bool call_callback = !!(frag->flags & MCA_BTL_DES_SEND_ALWAYS_CALLBACK);
        frag->flags &= ~MCA_BTL_DES_SEND_ALWAYS_CALLBACK;

        if (call_callback) {
            frag->base.des_cbfunc(&ugni_module->super, frag->endpoint, &frag->base, rc);
        }

        (void) mca_btl_ugni_frag_del_ref (frag, OPAL_SUCCESS);

        return 1;
    }

    /* slow(ish) path: remote side hasn't received the frag. call the frag's callback when
       we get the local smsg/msgq or remote rdma completion */
    frag->base.des_flags |= MCA_BTL_DES_SEND_ALWAYS_CALLBACK;

    mca_btl_ugni_frag_del_ref (frag, OPAL_SUCCESS);

    if (OPAL_UNLIKELY(OPAL_ERR_OUT_OF_RESOURCE == rc)) {
        /* queue up request */
        mca_btl_ugni_wait_list_append (ugni_module, endpoint, frag);
        rc = OPAL_SUCCESS;
    }

    return rc;
}

int mca_btl_ugni_sendi (struct mca_btl_base_module_t *btl,
                        struct mca_btl_base_endpoint_t *endpoint,
                        struct opal_colwertor_t *colwertor,
                        void *header, size_t header_size,
                        size_t payload_size, uint8_t order,
                        uint32_t flags, mca_btl_base_tag_t tag,
                        mca_btl_base_descriptor_t **descriptor)
{
    size_t total_size = header_size + payload_size;
    mca_btl_ugni_base_frag_t *frag = NULL;
    size_t packed_size = payload_size;
    int rc;

    if (OPAL_UNLIKELY(opal_list_get_size (&endpoint->frag_wait_list))) {
        if (NULL != descriptor) {
            *descriptor = NULL;
        }
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    do {
        BTL_VERBOSE(("btl/ugni isend sending fragment from %d -> %d. length = %" PRIu64
                     " endoint state %d", OPAL_PROC_MY_NAME.vpid, endpoint->peer_proc->proc_name.vpid,
                     payload_size + header_size, endpoint->state));

        flags |= MCA_BTL_DES_FLAGS_BTL_OWNERSHIP;

        if (0 == payload_size) {
            frag = (mca_btl_ugni_base_frag_t *) mca_btl_ugni_prepare_src_send_nodata (btl, endpoint, order, header_size,
                                                                                      flags);
        } else {
            frag = (mca_btl_ugni_base_frag_t *) mca_btl_ugni_prepare_src_send_buffered (btl, endpoint, colwertor, order,
                                                                                        header_size, &packed_size, flags);
        }

        assert (packed_size == payload_size);
        if (OPAL_UNLIKELY(NULL == frag || OPAL_SUCCESS != mca_btl_ugni_check_endpoint_state (endpoint))) {
            break;
        }

        frag->hdr.send.lag = (tag << 24) | total_size;
        memcpy (frag->segments[0].seg_addr.pval, header, header_size);

        rc = mca_btl_ugni_send_frag (endpoint, frag);
        if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
            break;
        }

        return OPAL_SUCCESS;
    } while (0);

    if (NULL != descriptor) {
        *descriptor = &frag->base;
    }

    return OPAL_ERR_OUT_OF_RESOURCE;
}

int mca_btl_ugni_progress_send_wait_list (mca_btl_base_endpoint_t *endpoint)
{
    mca_btl_ugni_base_frag_t *frag=NULL;
    int rc;

    do {
        OPAL_THREAD_LOCK(&endpoint->lock);
        frag = (mca_btl_ugni_base_frag_t *) opal_list_remove_first (&endpoint->frag_wait_list);
        OPAL_THREAD_UNLOCK(&endpoint->lock);
        if (NULL == frag) {
            break;
        }
        if (OPAL_LIKELY(!(frag->flags & MCA_BTL_UGNI_FRAG_RESPONSE))) {
            rc = mca_btl_ugni_send_frag (endpoint, frag);
        } else {
            rc = opal_mca_btl_ugni_smsg_send (frag, &frag->hdr.rdma, sizeof (frag->hdr.rdma),
                                              NULL, 0, MCA_BTL_UGNI_TAG_RDMA_COMPLETE);
        }

        if (OPAL_UNLIKELY(OPAL_SUCCESS > rc)) {
            if (OPAL_LIKELY(OPAL_ERR_OUT_OF_RESOURCE == rc)) {
                OPAL_THREAD_LOCK(&endpoint->lock);
                opal_list_prepend (&endpoint->frag_wait_list, (opal_list_item_t *) frag);
                OPAL_THREAD_UNLOCK(&endpoint->lock);
            } else {
                mca_btl_ugni_frag_complete (frag, rc);
            }
            return rc;
        }
    } while(1);

    return OPAL_SUCCESS;
}
