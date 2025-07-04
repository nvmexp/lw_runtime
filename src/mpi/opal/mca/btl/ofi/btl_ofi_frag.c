/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * $COPYRIGHT$
 * Copyright (c) 2018      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2018      Intel Inc. All rights reserved
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "btl_ofi.h"
#include "btl_ofi_frag.h"
#include "btl_ofi_rdma.h"
#include "btl_ofi_endpoint.h"

static void mca_btl_ofi_base_frag_constructor (mca_btl_ofi_base_frag_t *frag)
{
    /* zero everything out */
    memset ((char *) frag + sizeof (frag->base), 0, sizeof (*frag) - sizeof (frag->base));

    frag->base.des_segments = frag->segments;
    frag->base.des_segment_count = 1;
}

static void mca_btl_ofi_base_frag_destructor (mca_btl_ofi_base_frag_t *frag)
{

}

OBJ_CLASS_INSTANCE(mca_btl_ofi_base_frag_t,
                   mca_btl_base_descriptor_t,
                   mca_btl_ofi_base_frag_constructor,
                   mca_btl_ofi_base_frag_destructor);

OBJ_CLASS_INSTANCE(mca_btl_ofi_frag_completion_t,
                   opal_free_list_item_t,
                   NULL,
                   NULL);

mca_btl_ofi_frag_completion_t *mca_btl_ofi_frag_completion_alloc
                                        (mca_btl_base_module_t *btl,
                                         mca_btl_ofi_context_t *context,
                                         mca_btl_ofi_base_frag_t *frag,
                                         int type)
{
    mca_btl_ofi_frag_completion_t *comp;

    comp = (mca_btl_ofi_frag_completion_t*) opal_free_list_get(&context->frag_comp_list);
    comp->base.btl = btl;
    comp->base.my_context = context;
    comp->base.my_list = &context->frag_comp_list;
    comp->base.type = type;

    comp->frag = frag;
    comp->comp_ctx.comp = comp;

    return comp;
}


mca_btl_base_descriptor_t *mca_btl_ofi_alloc(
                                mca_btl_base_module_t *btl,
                                mca_btl_base_endpoint_t *endpoint,
                                uint64_t order, size_t size, uint32_t flags)
{
    mca_btl_ofi_module_t *ofi_btl = (mca_btl_ofi_module_t*) btl;
    mca_btl_ofi_base_frag_t *frag = NULL;
    mca_btl_ofi_context_t *context = get_ofi_context(ofi_btl);

    frag = mca_btl_ofi_frag_alloc(ofi_btl, &context->frag_list, endpoint);

    if (OPAL_LIKELY(frag)) {
        frag->segments[0].seg_addr.pval = frag + 1;
        frag->segments[0].seg_len = size;

        frag->base.des_segment_count = 1;
        frag->base.des_segments = &frag->segments[0];
        frag->base.des_flags = flags;
        frag->base.order     = order;
        frag->hdr.len        = size;
    }

    return (mca_btl_base_descriptor_t*) frag;
}

int mca_btl_ofi_free (mca_btl_base_module_t *btl, mca_btl_base_descriptor_t *des)
{
    /* return the frag to the free list. */
    mca_btl_ofi_frag_return ((mca_btl_ofi_base_frag_t*) des);
    return OPAL_SUCCESS;
}

int mca_btl_ofi_send (mca_btl_base_module_t *btl,
                      mca_btl_base_endpoint_t *endpoint,
                      mca_btl_base_descriptor_t *descriptor,
                      mca_btl_base_tag_t tag)
{
    int rc = 0;
    mca_btl_ofi_context_t *context;
    mca_btl_ofi_module_t *ofi_btl = (mca_btl_ofi_module_t*) btl;
    mca_btl_ofi_endpoint_t *ofi_ep = (mca_btl_ofi_endpoint_t*) endpoint;
    mca_btl_ofi_base_frag_t *frag = (mca_btl_ofi_base_frag_t*) descriptor;
    mca_btl_ofi_frag_completion_t *comp;

    frag->base.des_flags |= MCA_BTL_DES_SEND_ALWAYS_CALLBACK;

    /* This tag is the active message tag for the remote side */
    frag->hdr.tag = tag;

    /* create completion context */
    context = get_ofi_context(ofi_btl);
    comp = mca_btl_ofi_frag_completion_alloc(btl, context, frag,
                                             MCA_BTL_OFI_TYPE_SEND);

    /* send the frag. Note that we start sending from BTL header + payload
     * because we need the other side to have this header information. */
    rc = fi_send(context->tx_ctx,
                 &frag->hdr,
                 sizeof(mca_btl_ofi_header_t) +  frag->hdr.len,
                 NULL,
                 ofi_ep->peer_addr,
                 &comp->comp_ctx);

    if (OPAL_UNLIKELY(FI_SUCCESS != rc)) {
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    MCA_BTL_OFI_NUM_SEND_INC(ofi_btl);
    return OPAL_SUCCESS;
}

int mca_btl_ofi_recv_frag (mca_btl_ofi_module_t *ofi_btl,
                           mca_btl_base_endpoint_t *endpoint,
                           mca_btl_ofi_context_t *context,
                           mca_btl_ofi_base_frag_t *frag)
{
    int rc;
    mca_btl_active_message_callback_t *reg;

    /* Tell PML where the payload is */
    frag->base.des_segments = frag->segments;
    frag->segments[0].seg_addr.pval = frag+1;
    frag->segments[0].seg_len = frag->hdr.len;
    frag->base.des_segment_count = 1;

    /* call the callback */
    reg = mca_btl_base_active_message_trigger + frag->hdr.tag;
    reg->cbfunc (&ofi_btl->super, frag->hdr.tag, &frag->base, reg->cbdata);
    mca_btl_ofi_frag_complete(frag, OPAL_SUCCESS);

    /* repost the recv */
    rc = mca_btl_ofi_post_recvs((mca_btl_base_module_t*) ofi_btl, context, 1);
    if (OPAL_SUCCESS != rc) {
        /* might not be that bad but let's just fail here. */
        BTL_ERROR(("failed reposting receive."));
        MCA_BTL_OFI_ABORT();
    }

    return OPAL_SUCCESS;
}

struct mca_btl_base_descriptor_t *mca_btl_ofi_prepare_src (
                                                mca_btl_base_module_t *btl,
                                                mca_btl_base_endpoint_t *endpoint,
                                                opal_colwertor_t *colwertor,
                                                uint8_t order, size_t reserve,
                                                size_t *size, uint32_t flags)
{
    struct iovec iov;
    size_t length;
    uint32_t iov_count = 1;
    mca_btl_ofi_base_frag_t *frag;

    /* allocate the frag with reserve. */
    frag = (mca_btl_ofi_base_frag_t*) mca_btl_ofi_alloc(btl, endpoint,
                                                        order, reserve, flags);
    if (OPAL_UNLIKELY(NULL == frag)) {
        return NULL;
    }

    /* pack the data after the reserve */
    iov.iov_len = *size;
    iov.iov_base = (IOVBASE_TYPE*)(((unsigned char*)(frag->segments[0].seg_addr.pval)) + reserve);
    opal_colwertor_pack(colwertor, &iov, &iov_count, &length);

    /* pass on frag information */
    frag->base.des_segments = frag->segments;
    frag->base.des_flags = flags;
    frag->base.order = MCA_BTL_NO_ORDER;
    frag->segments[0].seg_len += length;
    frag->hdr.len += length;
    *size = length;

    return &frag->base;
}
