/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014-2016 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2020      Google, LLC. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "osc_rdma_comm.h"
#include "osc_rdma_lock.h"

#include "mpi.h"

#include "opal/util/sys_limits.h"

static void ompi_osc_rdma_handle_init (ompi_osc_rdma_handle_t *rdma_handle)
{
    rdma_handle->btl_handle = NULL;
    OBJ_CONSTRUCT(&rdma_handle->attachments, opal_list_t);
}

static void ompi_osc_rdma_handle_fini (ompi_osc_rdma_handle_t *rdma_handle)
{
    OPAL_LIST_DESTRUCT(&rdma_handle->attachments);
}

OBJ_CLASS_INSTANCE(ompi_osc_rdma_handle_t, opal_object_t, ompi_osc_rdma_handle_init,
                   ompi_osc_rdma_handle_fini);

OBJ_CLASS_INSTANCE(ompi_osc_rdma_attachment_t, opal_list_item_t, NULL, NULL);

/**
 * ompi_osc_rdma_find_region_containing:
 *
 * @param[in]  regions      sorted list of regions
 * @param[in]  min_index    minimum index to search (call with 0)
 * @param[in]  max_index    maximum index to search (call with length - 1)
 * @param[in]  base         base of region to search for
 * @param[in]  bound        bound of region to search for
 * @param[in]  region_size  size of an ompi_osc_rdma_region_t object
 * @param[out] region_index index of region if found (may be NULL)
 *
 * @returns an index on success or -1 on failure
 *
 * This function searches through a sorted list of rdma regions {regions} and finds
 * the region that contains the region specified by {base} and {bound}. If a
 * matching region is found the index of that region is returned else the function
 * returns -1.
 */
static inline ompi_osc_rdma_region_t *ompi_osc_rdma_find_region_containing (ompi_osc_rdma_region_t *regions, int min_index,
                                                                            int max_index, intptr_t base, intptr_t bound,
                                                                            size_t region_size, int *region_index)
{
    int mid_index = (max_index + min_index) >> 1;
    ompi_osc_rdma_region_t *region = (ompi_osc_rdma_region_t *)((intptr_t) regions + mid_index * region_size);
    intptr_t region_bound;

    if (min_index > max_index) {
        return NULL;
    }

    region_bound = (intptr_t) (region->base + region->len);

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_DEBUG, "checking memory region %p-%p against %p-%p (index %d) (min_index = %d, "
                     "max_index = %d)", (void *) base, (void *) bound, (void *) region->base,
                     (void *)(region->base + region->len), mid_index, min_index, max_index);

    if (region->base > base) {
        return ompi_osc_rdma_find_region_containing (regions, min_index, mid_index-1, base, bound, region_size,
                                                     region_index);
    }

    if (bound <= region_bound) {
        if (region_index) {
            *region_index = mid_index;
        }

        return region;
    }

    return ompi_osc_rdma_find_region_containing (regions, mid_index+1, max_index, base, bound, region_size, region_index);
}

/* binary search for insertion point */
static ompi_osc_rdma_region_t *find_insertion_point (ompi_osc_rdma_region_t *regions, int min_index, int max_index,
                                                     intptr_t base, size_t region_size, int *region_index)
{
    int mid_index = (max_index + min_index) >> 1;
    ompi_osc_rdma_region_t *region = (ompi_osc_rdma_region_t *)((intptr_t) regions + mid_index * region_size);

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "find_insertion_point (%d, %d, %lx, %lu)\n", min_index, max_index, base,
                     region_size);

    if (max_index < min_index) {
        *region_index = min_index;
        return (ompi_osc_rdma_region_t *)((intptr_t) regions + min_index * region_size);
    }

    if (region->base > base || (region->base == base && region->len > region_size)) {
        return find_insertion_point (regions, min_index, mid_index-1, base, region_size, region_index);
    }

    return find_insertion_point (regions, mid_index+1, max_index, base, region_size, region_index);
}

static bool ompi_osc_rdma_find_conflicting_attachment (ompi_osc_rdma_handle_t *handle, intptr_t base, intptr_t bound)
{
    ompi_osc_rdma_attachment_t *attachment;

    OPAL_LIST_FOREACH(attachment, &handle->attachments, ompi_osc_rdma_attachment_t) {
        intptr_t region_bound = attachment->base + attachment->len;
        if (base >= attachment->base && base < region_bound ||
            bound > attachment->base && bound <= region_bound) {
            OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "existing region {%p, %p} overlaps region {%p, %p}",
                             (void *) attachment->base, (void *) region_bound, (void *) base, (void *) bound);
            return true;
        }
    }

    return false;
}

static int ompi_osc_rdma_add_attachment (ompi_osc_rdma_handle_t *handle, intptr_t base, size_t len)
{
    ompi_osc_rdma_attachment_t *attachment = OBJ_NEW(ompi_osc_rdma_attachment_t);
    assert (NULL != attachment);

    if (ompi_osc_rdma_find_conflicting_attachment(handle, base, base + len)) {
        return OMPI_ERR_RMA_ATTACH;
    }

    attachment->base = base;
    attachment->len = len;

    opal_list_append (&handle->attachments, &attachment->super);

    return OMPI_SUCCESS;
}

static int ompi_osc_rdma_remove_attachment (ompi_osc_rdma_handle_t *handle, intptr_t base)
{
    ompi_osc_rdma_attachment_t *attachment;

    OPAL_LIST_FOREACH(attachment, &handle->attachments, ompi_osc_rdma_attachment_t) {
        OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "checking attachment %p against %p",
                         (void *) attachment->base, (void *) base);
        if (attachment->base == (intptr_t) base) {
            opal_list_remove_item (&handle->attachments, &attachment->super);
            OBJ_RELEASE(attachment);
            return OMPI_SUCCESS;
        }
    }

    return OMPI_ERR_NOT_FOUND;
}

int ompi_osc_rdma_attach (struct ompi_win_t *win, void *base, size_t len)
{
    ompi_osc_rdma_module_t *module = GET_MODULE(win);
    const int my_rank = ompi_comm_rank (module->comm);
    ompi_osc_rdma_peer_t *my_peer = ompi_osc_rdma_module_peer (module, my_rank);
    ompi_osc_rdma_region_t *region;
    ompi_osc_rdma_handle_t *rdma_region_handle;
    osc_rdma_counter_t region_count;
    osc_rdma_counter_t region_id;
    intptr_t bound, aligned_base, aligned_bound;
    intptr_t page_size = opal_getpagesize ();
    int region_index, ret;
    size_t aligned_len;

    if (module->flavor != MPI_WIN_FLAVOR_DYNAMIC) {
        return OMPI_ERR_RMA_FLAVOR;
    }

    if (0 == len) {
        /* shot-circuit 0-byte case */
        return OMPI_SUCCESS;
    }

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "attach: %s, %p, %lu", win->w_name, base, (unsigned long) len);

    OPAL_THREAD_LOCK(&module->lock);

    region_count = module->state->region_count & 0xffffffffL;
    region_id    = module->state->region_count >> 32;

    if (region_count == mca_osc_rdma_component.max_attach) {
        OPAL_THREAD_UNLOCK(&module->lock);
        OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "attach: could not attach. max attachment count reached.");
        return OMPI_ERR_RMA_ATTACH;
    }

    /* it is wasteful to register less than a page. this may allow the remote side to access more
     * memory but the MPI standard covers this with calling the calling behavior erroneous */
    bound = (intptr_t) base + len;
    aligned_bound = OPAL_ALIGN((intptr_t) base + len, page_size, intptr_t);
    aligned_base = (intptr_t) base & ~(page_size - 1);
    aligned_len = (size_t)(aligned_bound - aligned_base);

    /* see if a registered region already exists */
    region = ompi_osc_rdma_find_region_containing ((ompi_osc_rdma_region_t *) module->state->regions, 0, region_count - 1,
                                                   aligned_base, aligned_bound, module->region_size, &region_index);
    if (NULL != region) {
        /* validates that the region does not overlap with an existing region even if they are on the same page */
        ret = ompi_osc_rdma_add_attachment (module->dynamic_handles[region_index], (intptr_t) base, len);
        OPAL_THREAD_UNLOCK(&module->lock);
        /* no need to ilwalidate remote caches */
        return ret;
    }

    /* region is in flux */
    module->state->region_count = -1;
    opal_atomic_wmb ();

    ompi_osc_rdma_lock_acquire_exclusive (module, my_peer, offsetof (ompi_osc_rdma_state_t, regions_lock));

    /* do a binary seach for where the region should be inserted */
    if (region_count) {
        region = find_insertion_point ((ompi_osc_rdma_region_t *) module->state->regions, 0, region_count - 1,
                                       (intptr_t) base, module->region_size, &region_index);

        if (region_index < region_count) {
            memmove ((void *) ((intptr_t) region + module->region_size), region,
                     (region_count - region_index) * module->region_size);
            memmove (module->dynamic_handles + region_index + 1, module->dynamic_handles + region_index,
                     (region_count - region_index) * sizeof (module->dynamic_handles[0]));
        }
    } else {
        region_index = 0;
        region = (ompi_osc_rdma_region_t *) module->state->regions;
    }

    region->base = aligned_base;
    region->len  = aligned_len;

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_DEBUG, "attaching dynamic memory region {%p, %p} aligned {%p, %p}, at index %d",
                     base, (void *) bound, (void *) aligned_base, (void *) aligned_bound, region_index);

    /* add RDMA region handle to track this region */
    rdma_region_handle = OBJ_NEW(ompi_osc_rdma_handle_t);
    assert (NULL != rdma_region_handle);

    if (module->selected_btl->btl_register_mem) {
        mca_btl_base_registration_handle_t *handle;

        ret = ompi_osc_rdma_register (module, MCA_BTL_ENDPOINT_ANY, (void *) region->base, region->len,
                                      MCA_BTL_REG_FLAG_ACCESS_ANY, &handle);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
            OPAL_THREAD_UNLOCK(&module->lock);
            OBJ_RELEASE(rdma_region_handle);
            return OMPI_ERR_RMA_ATTACH;
        }

        memcpy (region->btl_handle_data, handle, module->selected_btl->btl_registration_handle_size);
        rdma_region_handle->btl_handle = handle;
    } else {
        rdma_region_handle->btl_handle = NULL;
    }

    ret = ompi_osc_rdma_add_attachment (rdma_region_handle, (intptr_t) base, len);
    assert(OMPI_SUCCESS == ret);
    module->dynamic_handles[region_index] = rdma_region_handle;

#if OPAL_ENABLE_DEBUG
    for (int i = 0 ; i < region_count + 1 ; ++i) {
        region = (ompi_osc_rdma_region_t *) ((intptr_t) module->state->regions + i * module->region_size);

        OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_DEBUG, " dynamic region %d: {%p, %lu}", i,
                         (void *) region->base, (unsigned long) region->len);
    }
#endif

    opal_atomic_mb ();
    /* the region state has changed */
    module->state->region_count = ((region_id + 1) << 32) | (region_count + 1);

    ompi_osc_rdma_lock_release_exclusive (module, my_peer, offsetof (ompi_osc_rdma_state_t, regions_lock));
    OPAL_THREAD_UNLOCK(&module->lock);

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "attach complete");

    return OMPI_SUCCESS;
}


int ompi_osc_rdma_detach (struct ompi_win_t *win, const void *base)
{
    ompi_osc_rdma_module_t *module = GET_MODULE(win);
    const int my_rank = ompi_comm_rank (module->comm);
    ompi_osc_rdma_peer_dynamic_t *my_peer = (ompi_osc_rdma_peer_dynamic_t *) ompi_osc_rdma_module_peer (module, my_rank);
    ompi_osc_rdma_handle_t *rdma_region_handle;
    osc_rdma_counter_t region_count, region_id;
    ompi_osc_rdma_region_t *region;
    void *bound;
    int start_index = INT_MAX, region_index;

    if (module->flavor != MPI_WIN_FLAVOR_DYNAMIC) {
        return OMPI_ERR_WIN;
    }

    OPAL_THREAD_LOCK(&module->lock);

    /* the upper 4 bytes of the region count are an instance counter */
    region_count = module->state->region_count & 0xffffffffL;
    region_id    = module->state->region_count >> 32;

    /* look up the associated region */
    for (region_index = 0 ; region_index < region_count ; ++region_index) {
        rdma_region_handle = module->dynamic_handles[region_index];
        region = (ompi_osc_rdma_region_t *) ((intptr_t) module->state->regions + region_index * module->region_size);
        OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_INFO, "checking attachments at index %d {.base=%p, len=%lu} for attachment %p"
                         ", region handle=%p", region_index, (void *) region->base, region->len, base, rdma_region_handle);

        if (region->base > (uintptr_t) base || (region->base + region->len) < (uintptr_t) base) {
            continue;
        }

        if (OPAL_SUCCESS == ompi_osc_rdma_remove_attachment (rdma_region_handle, (intptr_t) base)) {
            break;
        }
    }

    if (region_index == region_count) {
        OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_INFO, "could not find dynamic memory attachment for %p", base);
        OPAL_THREAD_UNLOCK(&module->lock);
        return OMPI_ERR_BASE;
    }

    if (!opal_list_is_empty (&rdma_region_handle->attachments)) {
        /* another region is referencing this attachment */
        return OMPI_SUCCESS;
    }

    /* lock the region so it can't change while a peer is reading it */
    ompi_osc_rdma_lock_acquire_exclusive (module, &my_peer->super, offsetof (ompi_osc_rdma_state_t, regions_lock));

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_DEBUG, "detaching dynamic memory region {%p, %p} from index %d",
                     base, (void *)((intptr_t) base + region->len), region_index);

    if (module->selected_btl->btl_register_mem) {
        ompi_osc_rdma_deregister (module, rdma_region_handle->btl_handle);

    }

    if (region_index < region_count - 1) {
        size_t end_count = region_count - region_index - 1;
        memmove (module->dynamic_handles + region_index, module->dynamic_handles + region_index + 1,
                 end_count * sizeof (module->dynamic_handles[0]));
        memmove (region, (void *)((intptr_t) region + module->region_size),
                 end_count * module->region_size);
    }

    OBJ_RELEASE(rdma_region_handle);
    module->dynamic_handles[region_count - 1] = NULL;

    module->state->region_count = ((region_id + 1) << 32) | (region_count - 1);

    ompi_osc_rdma_lock_release_exclusive (module, &my_peer->super, offsetof (ompi_osc_rdma_state_t, regions_lock));

    OPAL_THREAD_UNLOCK(&module->lock);

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "detach complete");

    return OMPI_SUCCESS;
}

/**
 * @brief refresh the local view of the dynamic memory region
 *
 * @param[in] module         osc rdma module
 * @param[in] peer           peer object to refresh
 *
 * This function does the work of keeping the local view of a remote peer in sync with what is attached
 * to the remote window. It is called on every address translation since there is no way (lwrrently) to
 * detect that the attached regions have changed. To reduce the amount of data read we first read the
 * region count (which contains an id). If that hasn't changed the region data is not updated. If the
 * list of attached regions has changed then all valid regions are read from the peer while holding
 * their region lock.
 */
static int ompi_osc_rdma_refresh_dynamic_region (ompi_osc_rdma_module_t *module, ompi_osc_rdma_peer_dynamic_t *peer) {
    osc_rdma_counter_t region_count, region_id;
    uint64_t source_address;
    int ret;

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "refreshing dynamic memory regions for target %d", peer->super.rank);

    /* this loop is meant to prevent us from reading data while the remote side is in attach */
    do {
        osc_rdma_counter_t remote_value;

        source_address = (uint64_t)(intptr_t) peer->super.state + offsetof (ompi_osc_rdma_state_t, region_count);
        ret = ompi_osc_get_data_blocking (module, peer->super.state_endpoint, source_address, peer->super.state_handle,
                                          &remote_value, sizeof (remote_value));
        if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
            return ret;
        }

        region_id = remote_value >> 32;
        region_count = remote_value & 0xffffffffl;
        /* check if the region is changing */
    } while (0xffffffffl == region_count);

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_DEBUG, "target region: id 0x%lx, count 0x%lx (cached: 0x%x, 0x%x)",
                     (unsigned long) region_id, (unsigned long) region_count, peer->region_id, peer->region_count);

    if (0 == region_count) {
        return OMPI_ERR_RMA_RANGE;
    }

    /* check if the cached copy is out of date */
    OPAL_THREAD_LOCK(&module->lock);

    if (peer->region_id != region_id) {
        unsigned region_len = module->region_size * region_count;
        void *temp;

        OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_DEBUG, "dynamic memory cache is out of data. reloading from peer");

        /* allocate only enough space for the remote regions */
        temp = realloc (peer->regions, region_len);
        if (NULL == temp) {
            OPAL_THREAD_UNLOCK(&module->lock);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        peer->regions = temp;

        /* lock the region */
        ompi_osc_rdma_lock_acquire_shared (module, &peer->super, 1, offsetof (ompi_osc_rdma_state_t, regions_lock),
                                           OMPI_OSC_RDMA_LOCK_EXCLUSIVE);

        source_address = (uint64_t)(intptr_t) peer->super.state + offsetof (ompi_osc_rdma_state_t, regions);
        ret = ompi_osc_get_data_blocking (module, peer->super.state_endpoint, source_address, peer->super.state_handle,
                                          peer->regions, region_len);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
            OPAL_THREAD_UNLOCK(&module->lock);
            return ret;
        }

        /* release the region lock */
        ompi_osc_rdma_lock_release_shared (module, &peer->super, -1, offsetof (ompi_osc_rdma_state_t, regions_lock));

        /* update cached region ids */
        peer->region_id = region_id;
        peer->region_count = region_count;
    }

    OPAL_THREAD_UNLOCK(&module->lock);

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "finished refreshing dynamic memory regions for target %d", peer->super.rank);

    return OMPI_SUCCESS;
}

int ompi_osc_rdma_find_dynamic_region (ompi_osc_rdma_module_t *module, ompi_osc_rdma_peer_t *peer, uint64_t base, size_t len,
				       ompi_osc_rdma_region_t **region)
{
    ompi_osc_rdma_peer_dynamic_t *dy_peer = (ompi_osc_rdma_peer_dynamic_t *) peer;
    intptr_t bound = (intptr_t) base + len;
    ompi_osc_rdma_region_t *regions;
    int ret, region_count;

    OSC_RDMA_VERBOSE(MCA_BASE_VERBOSE_TRACE, "locating dynamic memory region matching: {%" PRIx64 ", %" PRIx64 "}"
                     " (len %lu)", base, base + len, (unsigned long) len);

    if (!ompi_osc_rdma_peer_local_state (peer)) {
        ret = ompi_osc_rdma_refresh_dynamic_region (module, dy_peer);
        if (OMPI_SUCCESS != ret) {
            return ret;
        }

        regions = dy_peer->regions;
        region_count = dy_peer->region_count;
    } else {
        ompi_osc_rdma_state_t *peer_state = (ompi_osc_rdma_state_t *) peer->state;
        regions = (ompi_osc_rdma_region_t *) peer_state->regions;
        region_count = peer_state->region_count;
    }

    *region = ompi_osc_rdma_find_region_containing (regions, 0, region_count - 1, (intptr_t) base, bound, module->region_size, NULL);
    if (!*region) {
        return OMPI_ERR_RMA_RANGE;
    }

    /* round a matching region */
    return OMPI_SUCCESS;
}
