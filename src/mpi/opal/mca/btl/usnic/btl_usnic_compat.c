/*
 * Copyright (c) 2014-2018 Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2015      Intel, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#if BTL_IN_OPAL
#include "opal_config.h"
#include "opal/mca/btl/btl.h"
#else
#include "ompi_config.h"
#include "ompi/mca/btl/btl.h"
#endif

#include "opal/mca/mca.h"
#include "opal_stdint.h"

#include "btl_usnic_compat.h"
#include "btl_usnic_frag.h"
#include "btl_usnic_endpoint.h"
#include "btl_usnic_connectivity.h"
#include "btl_usnic_send.h"

/************************************************************************/

/* v2.0 and beyond */

#if (OPAL_MAJOR_VERSION >= 2)

#include "opal/util/proc.h"

/**************************
 * usNIC BTL-specific functions to hide differences between master and
 * v1.8
 **************************/
void usnic_compat_modex_send(int *rc,
                             mca_base_component_t *component,
                             opal_btl_usnic_modex_t *modexes,
                             size_t size)
{
    OPAL_MODEX_SEND(*rc, OPAL_PMIX_REMOTE, component,
                    modexes, size);
}

void usnic_compat_modex_recv(int *rc,
                             mca_base_component_t *component,
                             opal_proc_t *proc,
                             opal_btl_usnic_modex_t **modexes,
                             size_t *size)
{
    OPAL_MODEX_RECV(*rc, component, &proc->proc_name,
                    (uint8_t**) modexes, size);
}

uint64_t usnic_compat_rte_hash_name(opal_process_name_t *pname)
{
    uint64_t name = pname->jobid;
    name <<= 32;
    name += pname->vpid;
    return name;
}

const char *usnic_compat_proc_name_print(opal_process_name_t *pname)
{
    return OPAL_NAME_PRINT(*pname);
}

/************************************************************************/

/* v1.7, v1.8, and v1.10 (there was no v1.9) */

#elif (OPAL_MAJOR_VERSION == 1 && OPAL_MINOR_VERSION >= 7)

#include "ompi/proc/proc.h"
#include "ompi/mca/rte/rte.h"
#include "ompi/mca/rte/base/base.h"
#include "ompi/runtime/ompi_module_exchange.h"


/**************************
 * Replicate functions that exist on master
 **************************/
char* opal_get_proc_hostname(opal_proc_t *proc)
{
    return proc->proc_hostname;
}

/**************************
 * usNIC BTL-specific functions to hide differences between master and
 * v1.8
 **************************/
void usnic_compat_modex_send(int *rc,
                             mca_base_component_t *component,
                             struct opal_btl_usnic_modex_t *modexes,
                             size_t size)
{
    *rc = ompi_modex_send(component, modexes, size);
}

void usnic_compat_modex_recv(int *rc,
                             mca_base_component_t *component,
                             opal_proc_t *proc,
                             struct opal_btl_usnic_modex_t **modexes,
                             size_t *size)
{
    *rc = ompi_modex_recv(component, proc, (void*) modexes, size);
}

uint64_t usnic_compat_rte_hash_name(opal_process_name_t *pname)
{
    return ompi_rte_hash_name(pname);
}

const char *usnic_compat_proc_name_print(opal_process_name_t *pname)
{
    return OMPI_NAME_PRINT(pname);
}

/*
 * In v1.8, we call ompi_free_list_init, and ignore some of these params
 */
int usnic_compat_free_list_init(opal_free_list_t *free_list,
                                size_t frag_size,
                                size_t frag_alignment,
                                opal_class_t* frag_class,
                                size_t payload_buffer_size,
                                size_t payload_buffer_alignment,
                                int num_elements_to_alloc,
                                int max_elements_to_alloc,
                                int num_elements_per_alloc,
                                struct mca_mpool_base_module_t *mpool,
                                int mpool_reg_flags,
                                void *unused0,
                                opal_free_list_item_init_fn_t item_init,
                                void *ctx)
{
    return ompi_free_list_init_new(free_list,
                                   frag_size,
                                   frag_alignment,
                                   frag_class,
                                   payload_buffer_size,
                                   payload_buffer_alignment,
                                   num_elements_to_alloc,
                                   max_elements_to_alloc,
                                   num_elements_per_alloc,
                                   mpool);
}

static volatile bool agent_thread_time_to_exit = false;
static opal_thread_t agent_thread;
static opal_event_t blocker; // event to block on
static opal_event_base_t *agent_evbase = NULL;

static struct timeval long_timeout = {
    .tv_sec = 3600,
    .tv_usec = 0
};

/*
 * If this event is fired, just restart it so that this event base
 * continues to have something to block on.
 */
static void blocker_timeout_cb(int fd, short args, void *cbdata)
{
    opal_event_add(&blocker, &long_timeout);
}

/*
 * Agent progress thread main entry point
 */
static void *agent_thread_main(opal_object_t *obj)
{
    while (!agent_thread_time_to_exit) {
        opal_event_loop(agent_evbase, OPAL_EVLOOP_ONCE);
    }

    return NULL;
}

opal_event_base_t *opal_progress_thread_init(const char *name)
{
    assert(NULL == name);

    /* Create the event base */
    agent_evbase = opal_event_base_create();
    if (NULL == agent_evbase) {
        return NULL;
    }

    /* add an event to the new event base (if there are no events,
       opal_event_loop() will return immediately) */
    opal_event_set(agent_evbase, &blocker, -1, OPAL_EV_PERSIST,
                   blocker_timeout_cb, NULL);
    opal_event_add(&blocker, &long_timeout);

    /* Spawn the agent thread event loop */
    OBJ_CONSTRUCT(&agent_thread, opal_thread_t);
    agent_thread.t_run = agent_thread_main;
    agent_thread.t_arg = NULL;
    int ret;
    ret = opal_thread_start(&agent_thread);
    if (OPAL_SUCCESS != ret) {
        OPAL_ERROR_LOG(ret);
        ABORT("Failed to start usNIC agent thread");
        /* Will not return */
    }

    return agent_evbase;
}

int opal_progress_thread_finalize(const char *name)
{
    assert(NULL == name);

    agent_thread_time_to_exit = true;

    /* break the event loop - this will cause the loop to exit upon
       completion of any current event */
    opal_event_base_loopbreak(agent_evbase);
    opal_thread_join(&agent_thread, NULL);

    return OPAL_SUCCESS;
}

#endif /* OMPI version */

/************************************************************************/

/* BTL 2.0 and 3.0 compatibilty functions */

/*----------------------------------------------------------------------*/

/* The following functions are common between BTL 2.0 and 3.0 */

/* Responsible for sending "small" frags (reserve + *size <= max_frag_payload)
 * in the same manner as btl_prepare_src.  Must return a smaller amount than
 * requested if the given colwertor cannot process the entire (*size).
 */
static inline opal_btl_usnic_send_frag_t *
prepare_src_small(
    struct opal_btl_usnic_module_t* module,
    struct mca_btl_base_endpoint_t* endpoint,
    struct opal_colwertor_t* colwertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags)
{
    opal_btl_usnic_send_frag_t *frag;
    opal_btl_usnic_small_send_frag_t *sfrag;
    size_t payload_len;

    payload_len = *size + reserve;
    assert(payload_len <= module->max_frag_payload); /* precondition */

    sfrag = opal_btl_usnic_small_send_frag_alloc(module);
    if (OPAL_UNLIKELY(NULL == sfrag)) {
        return NULL;
    }
    frag = &sfrag->ssf_base;

    /* In the case of a colwertor, we will copy the data in now, since that is
     * the cheapest way to discover how much we can actually send (since we know
     * we will pack it anyway later).  The alternative is to do all of the
     * following:
     * 1) clone_with_position(colwertor) and see where the new position ends up
     *    actually being (see opal_btl_usnic_colwertor_pack_peek).  Otherwise we
     *    aren't fulfilling our contract w.r.t. (*size).
     * 2) Add a bunch of branches checking for different cases, both here and in
     *    progress_sends
     * 3) If we choose to defer the packing, we must clone the colwertor because
     *    the PML owns it and might reuse it for another prepare_src call.
     *
     * Two colwertor clones is likely to be at least as slow as just copying the
     * data and might consume a similar amount of memory.  Plus we still have to
     * pack it later to send it.
     *
     * The reason we do not copy non-colwertor buffer at this point is because
     * we might still use INLINE for the send, and in that case we do not want
     * to copy the data at all.
     */
    if (OPAL_UNLIKELY(opal_colwertor_need_buffers(colwertor))) {
        /* put user data just after end of 1st seg (upper layer header) */
        assert(payload_len <= module->max_frag_payload);
        usnic_colwertor_pack_simple(
                colwertor,
                (IOVBASE_TYPE*)(intptr_t)(frag->sf_base.uf_local_seg[0].seg_addr.lval + reserve),
                *size,
                size);
        payload_len = reserve + *size;
        frag->sf_base.uf_base.USNIC_SEND_LOCAL_COUNT = 1;
        /* PML will copy header into beginning of segment */
        frag->sf_base.uf_local_seg[0].seg_len = payload_len;
    } else {
        opal_colwertor_get_lwrrent_pointer(colwertor,
                                           &sfrag->ssf_base.sf_base.uf_local_seg[1].seg_addr.pval);
        frag->sf_base.uf_base.USNIC_SEND_LOCAL_COUNT = 2;
        frag->sf_base.uf_local_seg[0].seg_len = reserve;
        frag->sf_base.uf_local_seg[1].seg_len = *size;
    }

    frag->sf_base.uf_base.des_flags = flags;
    frag->sf_endpoint = endpoint;

    return frag;
}

static void *
pack_chunk_seg_chain_with_reserve(
    struct opal_btl_usnic_module_t* module,
    opal_btl_usnic_large_send_frag_t *lfrag,
    size_t reserve_len,
    opal_colwertor_t *colwertor,
    size_t max_colwertor_bytes,
    size_t *colwertor_bytes_packed)
{
    opal_btl_usnic_chunk_segment_t *seg;
    void *ret_ptr = NULL;
    int n_segs;
    uint8_t *copyptr;
    size_t copylen;
    size_t seg_space;
    size_t max_data;
    bool first_pass;

    assert(NULL != lfrag);
    assert(NULL != colwertor_bytes_packed);

    n_segs = 0;
    *colwertor_bytes_packed = 0;

    first_pass = true;
    while (*colwertor_bytes_packed < max_colwertor_bytes ||
           first_pass) {
        seg = opal_btl_usnic_chunk_segment_alloc(module);
        if (OPAL_UNLIKELY(NULL == seg)) {
            BTL_ERROR(("chunk segment allocation error"));
            abort(); /* XXX */
        }
        ++n_segs;

        seg_space = module->max_chunk_payload;
        copyptr = seg->ss_base.us_payload.raw;

        if (first_pass) {
            /* logic could accommodate >max, but lwrrently doesn't */
            assert(reserve_len <= module->max_chunk_payload);
            ret_ptr = copyptr;
            seg_space -= reserve_len;
            copyptr += reserve_len;
        }

        /* now pack any colwertor data */
        if (*colwertor_bytes_packed < max_colwertor_bytes && seg_space > 0) {
            copylen = max_colwertor_bytes - *colwertor_bytes_packed;
            if (copylen > seg_space) {
                copylen = seg_space;
            }
            usnic_colwertor_pack_simple(colwertor, copyptr, copylen, &max_data);
            seg_space -= max_data;
            *colwertor_bytes_packed += max_data;

            /* If unable to pack any of the remaining bytes, release the
            * most recently allocated segment and finish processing.
            */
            if (seg_space == module->max_chunk_payload) {
                assert(max_data == 0); /* only way this can happen */
                opal_btl_usnic_chunk_segment_return(module, seg);
                break;
            }
        }

        /* bozo checks */
        assert(seg_space >= 0);
        assert(seg_space < module->max_chunk_payload);

        /* append segment of data to chain to send */
        seg->ss_parent_frag = &lfrag->lsf_base;
        seg->ss_len = module->max_chunk_payload - seg_space;
        opal_list_append(&lfrag->lsf_seg_chain, &seg->ss_base.us_list.super);

#if MSGDEBUG1
        opal_output(0, "%s: appending seg=%p, frag=%p, payload=%zd\n",
                    __func__, (void *)seg, (void *)lfrag,
                    (module->max_chunk_payload - seg_space));
#endif

        first_pass = false;
    }

    return ret_ptr;
}

/* Responsible for handling "large" frags (reserve + *size > max_frag_payload)
 * in the same manner as btl_prepare_src.  Must return a smaller amount than
 * requested if the given colwertor cannot process the entire (*size).
 */
static opal_btl_usnic_send_frag_t *
prepare_src_large(
    struct opal_btl_usnic_module_t* module,
    struct mca_btl_base_endpoint_t* endpoint,
    struct opal_colwertor_t* colwertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags)
{
    opal_btl_usnic_send_frag_t *frag;
    opal_btl_usnic_large_send_frag_t *lfrag;
    int rc;

    /* Get holder for the msg */
    lfrag = opal_btl_usnic_large_send_frag_alloc(module);
    if (OPAL_UNLIKELY(NULL == lfrag)) {
        return NULL;
    }
    frag = &lfrag->lsf_base;

    /* The header location goes in SG[0], payload in SG[1].  If we are using a
     * colwertor then SG[1].seg_len is accurate but seg_addr is NULL. */
    frag->sf_base.uf_base.USNIC_SEND_LOCAL_COUNT = 2;

    /* stash header location, PML will write here */
    frag->sf_base.uf_local_seg[0].seg_addr.pval = &lfrag->lsf_ompi_header;
    frag->sf_base.uf_local_seg[0].seg_len = reserve;
    /* make sure upper header small enough */
    assert(reserve <= sizeof(lfrag->lsf_ompi_header));

    if (OPAL_UNLIKELY(opal_colwertor_need_buffers(colwertor))) {
        /* threshold == -1 means always pack eagerly */
        if (mca_btl_usnic_component.pack_lazy_threshold >= 0 &&
            *size >= (size_t)mca_btl_usnic_component.pack_lazy_threshold) {
            MSGDEBUG1_OUT("packing frag %p on the fly", (void *)frag);
            lfrag->lsf_pack_on_the_fly = true;

            /* tell the PML we will absorb as much as possible while still
             * respecting indivisible element boundaries in the colwertor */
            *size = opal_btl_usnic_colwertor_pack_peek(colwertor, *size);

            /* Clone the colwertor b/c we (the BTL) don't own it and the PML
             * might mutate it after we return from this function. */
            rc = opal_colwertor_clone(colwertor, &frag->sf_colwertor,
                                      /*copy_stack=*/true);
            if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
                BTL_ERROR(("unexpected colwertor clone error"));
                abort(); /* XXX */
            }
        }
        else {
            /* pack everything in the colwertor into a chain of segments now,
             * leaving space for the PML header in the first segment */
            lfrag->lsf_base.sf_base.uf_local_seg[0].seg_addr.pval =
                pack_chunk_seg_chain_with_reserve(module, lfrag, reserve,
                                                  colwertor, *size, size);
        }

        /* We set SG[1] to {NULL,bytes_packed} so that various callwlations
         * by both PML and this BTL will be correct.  For example, the PML adds
         * up the bytes in the descriptor segments to determine if an MPI-level
         * request is complete or not. */
        frag->sf_base.uf_local_seg[1].seg_addr.pval = NULL;
        frag->sf_base.uf_local_seg[1].seg_len = *size;
    } else {
        /* colwertor not needed, just save the payload pointer in SG[1] */
        lfrag->lsf_pack_on_the_fly = true;
        opal_colwertor_get_lwrrent_pointer(colwertor,
                                           &frag->sf_base.uf_local_seg[1].seg_addr.pval);
        frag->sf_base.uf_local_seg[1].seg_len = *size;
    }

    frag->sf_base.uf_base.des_flags = flags;
    frag->sf_endpoint = endpoint;

    return frag;
}

/*----------------------------------------------------------------------*/

#if BTL_VERSION == 20

/*
 * BTL 2.0 version of module.btl_prepare_src.
 *
 * Note the "user" data the PML wishes to communicate and return a descriptor
 * that can be used for send or put.  We create a frag (which is also a
 * descriptor by virtue of its base class) and populate it with enough
 * source information to complete a future send/put.
 *
 * We will create either a small send frag if < than an MTU, otherwise a large
 * send frag.  The colwertor will be saved for deferred packing if the user
 * buffer is noncontiguous.  Otherwise it will be saved in one of the
 * descriptor's SGEs.
 *
 * NOTE that the *only* reason this routine is allowed to return a size smaller
 * than was requested is if the colwertor cannot process the entire amount.
 */
mca_btl_base_descriptor_t*
opal_btl_usnic_prepare_src(
    struct mca_btl_base_module_t* base_module,
    struct mca_btl_base_endpoint_t* endpoint,
    struct mca_mpool_base_registration_t* registration,
    struct opal_colwertor_t* colwertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags)
{
    OPAL_THREAD_LOCK(&btl_usnic_lock);
    opal_btl_usnic_module_t *module = (opal_btl_usnic_module_t*) base_module;
    opal_btl_usnic_send_frag_t *frag;
    uint32_t payload_len;
#if MSGDEBUG2
    size_t osize = *size;
#endif

    /* Do we need to check the connectivity?  If enabled, we'll check
       the connectivity at either first send to peer X or first ACK to
       peer X. */
    opal_btl_usnic_check_connectivity(module, endpoint);

    /*
     * if total payload len fits in one MTU use small send, else large
     */
    payload_len = *size + reserve;
    if (payload_len <= module->max_frag_payload) {
        frag = prepare_src_small(module, endpoint, colwertor,
                                 order, reserve, size, flags);
    } else {
        frag = prepare_src_large(module, endpoint, colwertor,
                                 order, reserve, size, flags);
    }

#if MSGDEBUG2
    opal_output(0, "prep_src: %s %s frag %p, size=%d+%u (was %u), colw=%p\n",
                module->linux_device_name,
                (reserve + *size) <= module->max_frag_payload?"small":"large",
                (void *)frag, (int)reserve, (unsigned)*size, (unsigned)osize,
                (void *)colwertor);
#if MSGDEBUG1
    {
        unsigned i;
        mca_btl_base_descriptor_t *desc = &frag->sf_base.uf_base;
        for (i=0; i<desc->USNIC_SEND_LOCAL_COUNT; ++i) {
            opal_output(0, "  %d: ptr:%p len:%d\n", i,
                        (void *)desc->USNIC_SEND_LOCAL[i].seg_addr.pval,
                        desc->USNIC_SEND_LOCAL[i].seg_len);
        }
    }
#endif
#endif

    OPAL_THREAD_UNLOCK(&btl_usnic_lock);
    return &frag->sf_base.uf_base;
}

/*
 * BTL 2.0 prepare_dst function (this function does not exist in BTL
 * 3.0).
 */
mca_btl_base_descriptor_t*
opal_btl_usnic_prepare_dst(
    struct mca_btl_base_module_t* base_module,
    struct mca_btl_base_endpoint_t* endpoint,
    struct mca_mpool_base_registration_t* registration,
    struct opal_colwertor_t* colwertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags)
{
    opal_btl_usnic_put_dest_frag_t *pfrag;
    opal_btl_usnic_module_t *module;
    void *data_ptr;

    module = (opal_btl_usnic_module_t *)base_module;

    /* allocate a fragment for this */
    pfrag = (opal_btl_usnic_put_dest_frag_t *)
        opal_btl_usnic_put_dest_frag_alloc(module);
    if (NULL == pfrag) {
        return NULL;
    }

    /* find start of the data */
    opal_colwertor_get_lwrrent_pointer(colwertor, (void **) &data_ptr);

    /* make a seg entry pointing at data_ptr */
    pfrag->uf_remote_seg[0].seg_addr.pval = data_ptr;
    pfrag->uf_remote_seg[0].seg_len = *size;

    pfrag->uf_base.order       = order;
    pfrag->uf_base.des_flags   = flags;

#if MSGDEBUG2
    opal_output(0, "prep_dst size=%d, addr=%p, pfrag=%p\n", (int)*size,
            data_ptr, (void *)pfrag);
#endif

    return &pfrag->uf_base;
}


/*
 * BTL 2.0 version of module.btl_put.
 *
 * Emulate an RDMA put.  We'll send the remote address
 * across to the other side so it will know where to put the data
 */
int opal_btl_usnic_put(
    struct mca_btl_base_module_t *btl,
    struct mca_btl_base_endpoint_t *endpoint,
    struct mca_btl_base_descriptor_t *desc)
{
    int rc;
    opal_btl_usnic_send_frag_t *frag;

    frag = (opal_btl_usnic_send_frag_t *)desc;

    opal_btl_usnic_compute_sf_size(frag);
    frag->sf_ack_bytes_left = frag->sf_size;

#if MSGDEBUG2
    opal_output(0, "usnic_put, frag=%p, size=%d\n", (void *)frag,
            (int)frag->sf_size);
#if MSGDEBUG1
    { unsigned i;
        for (i=0; i<desc->USNIC_PUT_LOCAL_COUNT; ++i) {
            opal_output(0, "  %d: ptr:%p len:%d%s\n", i,
                    desc->USNIC_PUT_LOCAL[i].seg_addr.pval,
                    desc->USNIC_PUT_LOCAL[i].seg_len,
                    (i==0)?" (put local)":"");
        }
        for (i=0; i<desc->USNIC_PUT_REMOTE_COUNT; ++i) {
            opal_output(0, "  %d: ptr:%p len:%d%s\n", i,
                    desc->USNIC_PUT_REMOTE[i].seg_addr.pval,
                    desc->USNIC_PUT_REMOTE[i].seg_len,
                    (i==0)?" (put remote)":"");
        }
    }
#endif
#endif

    /* RFXX copy out address - why does he not use our provided holder? */
    /* JMS What does this mean? ^^ */
    frag->sf_base.uf_remote_seg[0].seg_addr.pval =
        desc->USNIC_PUT_REMOTE->seg_addr.pval;

    rc = opal_btl_usnic_finish_put_or_send((opal_btl_usnic_module_t *)btl,
                                           (opal_btl_usnic_endpoint_t *)endpoint,
                                           frag,
                                           /*tag=*/MCA_BTL_NO_ORDER);

    return rc;
}

/*----------------------------------------------------------------------*/

#elif BTL_VERSION >= 30

/*
 * BTL 3.0 prepare_src function.
 *
 * This function is only used for sending PML fragments (not putting
 * or getting fragments).
 *
 * Note the "user" data the PML wishes to communicate and return a
 * descriptor.  We create a frag (which is also a descriptor by virtue
 * of its base class) and populate it with enough source information
 * to complete a future send.
 *
 * Recall that the usnic BTL's max_send_size is almost certainly
 * larger than the MTU (by default, max_send_size is either 25K or
 * 150K).  Therefore, the PML may give us a fragment up to
 * max_send_size in this function.  Hence, we make the decision here
 * as to whether it's a "small" fragment (i.e., size <= MTU, meaning
 * that it fits in a single datagram) or a "large" fragment (i.e.,
 * size > MTU, meaning that it must be chunked into multiple
 * datagrams).
 *
 * The colwertor will be saved for deferred packing if the user buffer
 * is noncontiguous.  Otherwise, it will be saved in one of the
 * descriptor's SGEs.
 *
 * NOTE that the *only* reason this routine is allowed to return a size smaller
 * than was requested is if the colwertor cannot process the entire amount.
 */
struct mca_btl_base_descriptor_t *
opal_btl_usnic_prepare_src(struct mca_btl_base_module_t *base_module,
                           struct mca_btl_base_endpoint_t *endpoint,
                           struct opal_colwertor_t *colwertor,
                           uint8_t order,
                           size_t reserve,
                           size_t *size,
                           uint32_t flags)
{
    opal_btl_usnic_module_t *module = (opal_btl_usnic_module_t*) base_module;
    opal_btl_usnic_send_frag_t *frag;
    uint32_t payload_len;
#if MSGDEBUG2
    size_t osize = *size;
#endif

    /* Do we need to check the connectivity?  If enabled, we'll check
       the connectivity at either first send to peer X or first ACK to
       peer X. */
    opal_btl_usnic_check_connectivity(module, endpoint);

    /*
     * if total payload len fits in one MTU use small send, else large
     */
    payload_len = *size + reserve;
    if (payload_len <= module->max_frag_payload) {
        frag = prepare_src_small(module, endpoint, colwertor,
                                 order, reserve, size, flags);
    } else {
        frag = prepare_src_large(module, endpoint, colwertor,
                                 order, reserve, size, flags);
    }

#if MSGDEBUG2
    opal_output(0, "prep_src: %s %s frag %p, size=%d+%u (was %u), colw=%p\n",
                module->linux_device_name,
                (reserve + *size) <= module->max_frag_payload?"small":"large",
                (void *)frag, (int)reserve, (unsigned)*size, (unsigned)osize,
                (void *)colwertor);
#if MSGDEBUG1
    {
        unsigned i;
        mca_btl_base_descriptor_t *desc = &frag->sf_base.uf_base;
        for (i=0; i<desc->USNIC_SEND_LOCAL_COUNT; ++i) {
            opal_output(0, "  %d: ptr:%p len:%d\n", i,
                        (void *)desc->USNIC_SEND_LOCAL[i].seg_addr.pval,
                        desc->USNIC_SEND_LOCAL[i].seg_len);
        }
    }
#endif
#endif

    return &frag->sf_base.uf_base;
}

/*
 * BTL 3.0 version of module.btl_put.
 *
 * Emulate an RDMA put.  We'll send the remote address across to the
 * other side so it will know where to put the data.
 *
 * Note that this function is only ever called with contiguous
 * buffers, so a colwertor is not necessary.
 */
int
opal_btl_usnic_put(struct mca_btl_base_module_t *base_module,
                   struct mca_btl_base_endpoint_t *endpoint,
                   void *local_address, uint64_t remote_address,
                   struct mca_btl_base_registration_handle_t *local_handle,
                   struct mca_btl_base_registration_handle_t *remote_handle,
                   size_t size, int flags, int order,
                   mca_btl_base_rdma_completion_fn_t cbfunc,
                   void *cbcontext, void *cbdata)
{
    opal_btl_usnic_send_frag_t *sfrag;
    opal_btl_usnic_module_t *module = (opal_btl_usnic_module_t*) base_module;

    /* At least for the moment, continue to make a descriptor, like we
       used to in BTL 2.0 */
    if (size <= module->max_frag_payload) {
        /* Small send fragment -- the whole thing fits in one MTU
           (i.e., a single chunk) */
        opal_btl_usnic_small_send_frag_t *ssfrag;
        ssfrag = opal_btl_usnic_small_send_frag_alloc(module);
        if (OPAL_UNLIKELY(NULL == ssfrag)) {
            return OPAL_ERR_OUT_OF_RESOURCE;
        }

        sfrag = &ssfrag->ssf_base;
    } else {
        /* Large send fragment -- need more than one MTU (i.e.,
           multiple chunks) */
        opal_btl_usnic_large_send_frag_t *lsfrag;
        lsfrag = opal_btl_usnic_large_send_frag_alloc(module);
        if (OPAL_UNLIKELY(NULL == lsfrag)) {
            return OPAL_ERR_OUT_OF_RESOURCE;
        }

        lsfrag->lsf_pack_on_the_fly = true;

        sfrag = &lsfrag->lsf_base;
    }

    sfrag->sf_endpoint = endpoint;
    sfrag->sf_size = size;
    sfrag->sf_ack_bytes_left = size;

    opal_btl_usnic_frag_t *frag;
    frag = &sfrag->sf_base;
    frag->uf_local_seg[0].seg_len = size;
    frag->uf_local_seg[0].seg_addr.pval = local_address;
    frag->uf_remote_seg[0].seg_len = size;
    frag->uf_remote_seg[0].seg_addr.pval =
        (void *)(uintptr_t) remote_address;

    mca_btl_base_descriptor_t *desc;
    desc = &frag->uf_base;
    desc->des_segment_count = 1;
    desc->des_segments = &frag->uf_local_seg[0];
    /* This is really the wrong cbfunc type, but we'll cast it to
       the Right type before we use it.  So it'll be ok. */
    desc->des_cbfunc = (mca_btl_base_completion_fn_t) cbfunc;
    desc->des_cbdata = cbdata;
    desc->des_context = cbcontext;
    desc->des_flags = flags;
    desc->order = order;

    int rc;
    rc = opal_btl_usnic_finish_put_or_send(module,
                                           (opal_btl_usnic_endpoint_t *)endpoint,
                                           sfrag,
                                           /*tag=*/MCA_BTL_NO_ORDER);
    return rc;
}

#endif /* BTL_VERSION */
