/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_mm.h"
#include "ucp_context.h"
#include "ucp_worker.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <string.h>
#include <inttypes.h>


static ucp_mem_t ucp_mem_dummy_handle = {
    .address      = NULL,
    .length       = 0,
    .alloc_method = UCT_ALLOC_METHOD_LAST,
    .alloc_md     = NULL,
    .md_map       = 0
};


ucs_status_t ucp_mem_rereg_mds(ucp_context_h context, ucp_md_map_t reg_md_map,
                               void *address, size_t length, unsigned uct_flags,
                               uct_md_h alloc_md, ucs_memory_type_t mem_type,
                               uct_mem_h *alloc_md_memh_p, uct_mem_h *uct_memh,
                               ucp_md_map_t *md_map_p)
{
    unsigned memh_index, prev_memh_index;
    uct_mem_h *prev_uct_memh;
    ucp_md_map_t new_md_map;
    const uct_md_attr_t *md_attr;
    unsigned prev_num_memh;
    unsigned md_index;
    ucs_status_t status;
    ucs_log_level_t level;

    if (reg_md_map == *md_map_p) {
        return UCS_OK; /* shortlwt - no changes required */
    }

    prev_num_memh = ucs_popcount(*md_map_p & reg_md_map);
    prev_uct_memh = ucs_alloca(prev_num_memh * sizeof(*prev_uct_memh));

    /* Go over previous handles, save only the ones we will need */
    memh_index      = 0;
    prev_memh_index = 0;
    ucs_for_each_bit(md_index, *md_map_p) {
        if (reg_md_map & UCS_BIT(md_index)) {
            /* memh still needed, save it */
            ucs_assert(prev_memh_index < prev_num_memh);
            prev_uct_memh[prev_memh_index++] = uct_memh[memh_index];
        } else if (alloc_md == context->tl_mds[md_index].md) {
            /* memh not needed and allocated, return it */
            if (alloc_md_memh_p != NULL) {
                *alloc_md_memh_p = uct_memh[memh_index];
            }
        } else {
            /* memh not needed and registered, deregister it */
            ucs_trace("de-registering memh[%d]=%p from md[%d]", memh_index,
                      uct_memh[memh_index], md_index);
            status = uct_md_mem_dereg(context->tl_mds[md_index].md,
                                      uct_memh[memh_index]);
            if (status != UCS_OK) {
                ucs_warn("failed to dereg from md[%d]=%s: %s", md_index,
                         context->tl_mds[md_index].rsc.md_name,
                         ucs_status_string(status));
            }
        }

        VALGRIND_MAKE_MEM_UNDEFINED(&uct_memh[memh_index],
                                    sizeof(uct_memh[memh_index]));
        ++memh_index;
    }

    /* prev_uct_memh should contain the handles which should be reused */
    ucs_assert(prev_memh_index == prev_num_memh);

    /* Go over requested MD map, and use / register new handles */
    new_md_map      = 0;
    memh_index      = 0;
    prev_memh_index = 0;
    ucs_for_each_bit(md_index, reg_md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        if (*md_map_p & UCS_BIT(md_index)) {
            /* already registered, use previous memh */
            ucs_assert(prev_memh_index < prev_num_memh);
            uct_memh[memh_index++] = prev_uct_memh[prev_memh_index++];
            new_md_map            |= UCS_BIT(md_index);
        } else if (context->tl_mds[md_index].md == alloc_md) {
            /* already allocated, add the memh we got from allocation */
            ucs_assert(alloc_md_memh_p != NULL);
            uct_memh[memh_index++] = *alloc_md_memh_p;
            new_md_map            |= UCS_BIT(md_index);
        } else if (!length) {
            /* don't register zero-length regions */
            continue;
        } else if (md_attr->cap.flags & UCT_MD_FLAG_REG) {
            if (!(md_attr->cap.reg_mem_types & UCS_BIT(mem_type))) {
                status = UCS_ERR_UNSUPPORTED;
            } else {
                ucs_assert(address && length);

                /* MD supports registration, register new memh on it */
                status = uct_md_mem_reg(context->tl_mds[md_index].md, address,
                        length, uct_flags, &uct_memh[memh_index]);
            }

            if (status == UCS_OK) {
                ucs_trace("registered address %p length %zu on md[%d] memh[%d]=%p",
                        address, length, md_index, memh_index,
                        uct_memh[memh_index]);
                new_md_map |= UCS_BIT(md_index);
                ++memh_index;
                continue;
            }

            level = (uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ?
                    UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR;

            ucs_log(level,
                    "failed to register address %p mem_type bit 0x%lx length %zu on "
                    "md[%d]=%s: %s (md reg_mem_types 0x%lx)",
                    address, UCS_BIT(mem_type), length, md_index,
                    context->tl_mds[md_index].rsc.md_name,
                    ucs_status_string(status),
                    md_attr->cap.reg_mem_types);

            if (!(uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS)) {
                goto err_dereg;
            }
        }
    }

    /* Update md_map, note that MDs which did not support registration will be
     * missing from the map.*/
    *md_map_p = new_md_map;
    return UCS_OK;

err_dereg:
    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, alloc_md, mem_type,
                      alloc_md_memh_p, uct_memh, md_map_p);
    return status;

}

/**
 * @return Whether MD number 'md_index' is selected by the configuration as part
 *         of allocation method number 'config_method_index'.
 */
static int ucp_is_md_selected_by_config(ucp_context_h context,
                                        unsigned config_method_index,
                                        unsigned md_index)
{
    const char *cfg_cmpt_name;
    const char *cmpt_name;

    cfg_cmpt_name = context->config.alloc_methods[config_method_index].cmpt_name;
    cmpt_name     = context->tl_mds[md_index].attr.component_name;

    return !strncmp(cfg_cmpt_name, "*",      UCT_COMPONENT_NAME_MAX) ||
           !strncmp(cfg_cmpt_name, cmpt_name, UCT_COMPONENT_NAME_MAX);
}

static ucs_status_t ucp_mem_alloc(ucp_context_h context, size_t length,
                                  unsigned uct_flags, const char *name, ucp_mem_h memh)
{
    uct_allocated_memory_t mem;
    uct_alloc_method_t method;
    unsigned method_index, md_index, num_mds;
    ucs_status_t status;
    uct_md_h *mds;

    mds = ucs_calloc(context->num_mds, sizeof(*mds), "temp mds");
    if (mds == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (method_index = 0; method_index < context->config.num_alloc_methods;
                    ++method_index)
    {
        method = context->config.alloc_methods[method_index].method;

        /* If we are trying MD method, gather all MDs which match the component
         * name specified in the configuration.
         */
        num_mds = 0;
        if (method == UCT_ALLOC_METHOD_MD) {
            for (md_index = 0; md_index < context->num_mds; ++md_index) {
                if (ucp_is_md_selected_by_config(context, method_index, md_index)) {
                    mds[num_mds++] = context->tl_mds[md_index].md;
                }
            }
        }

        status = uct_mem_alloc(memh->address, length, uct_flags, &method, 1, mds,
                               num_mds, name, &mem);
        if (status == UCS_OK) {
            goto allocated;
        }
    }

    status = UCS_ERR_NO_MEMORY;
    goto out;

allocated:
    ucs_debug("allocated memory at %p with method %s, now registering it",
             mem.address, uct_alloc_method_names[mem.method]);
    memh->address      = mem.address;
    memh->length       = mem.length;
    memh->alloc_method = mem.method;
    memh->mem_type     = mem.mem_type;
    memh->alloc_md     = mem.md;
    memh->md_map       = 0;
    status = ucp_mem_rereg_mds(context, UCS_MASK(context->num_mds), memh->address,
                               memh->length, uct_flags | UCT_MD_MEM_FLAG_HIDE_ERRORS,
                               memh->alloc_md, memh->mem_type, &mem.memh,
                               memh->uct, &memh->md_map);
    if (status != UCS_OK) {
        uct_mem_free(&mem);
    }
out:
    ucs_free(mds);
    return status;
}


static inline unsigned
ucp_mem_map_params2uct_flags(ucp_mem_map_params_t *params)
{
    unsigned flags = 0;

    if (params->field_mask & UCP_MEM_MAP_PARAM_FIELD_FLAGS) {
        if (params->flags & UCP_MEM_MAP_NONBLOCK) {
            flags |= UCT_MD_MEM_FLAG_NONBLOCK;
        }

        if (params->flags & UCP_MEM_MAP_FIXED) {
            flags |= UCT_MD_MEM_FLAG_FIXED;
        }
    }

    flags |= UCT_MD_MEM_ACCESS_ALL;
    /* TODO: disable atomic if ucp context does not have it */

    return flags;
}

/* Matrix of behavior
 * |--------------------------------------------------------------------------------|
 * | parameter |                             value                                  |
 * |-----------|--------------------------------------------------------------------|
 * | ALLOCATE  |  0     |     1     |  0  |  0  |  1  |     1     |  0  |     1     |
 * | FIXED     |  0     |     0     |  1  |  0  |  1  |     0     |  1  |     1     |
 * | addr      |  0     |     0     |  0  |  1  |  0  |     1     |  1  |     1     |
 * |-----------|--------|-----------|-----|-----|-----|-----------|-----|-----------|
 * | result    | err if | alloc/reg | err | reg | err | alloc/reg | err | alloc/reg |
 * |           | len >0 |           |     |     |     |  (hint)   |     | (fixed)   |
 * |--------------------------------------------------------------------------------|
 */
static inline ucs_status_t ucp_mem_map_check_and_adjust_params(ucp_mem_map_params_t *params)
{
    if (!(params->field_mask & UCP_MEM_MAP_PARAM_FIELD_LENGTH)) {
        ucs_error("The length value for mapping memory isn't set: %s",
                  ucs_status_string(UCS_ERR_ILWALID_PARAM));
        return UCS_ERR_ILWALID_PARAM;
    }

    /* First of all, define all fields */
    if (!(params->field_mask & UCP_MEM_MAP_PARAM_FIELD_ADDRESS)) {
        params->field_mask |= UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
        params->address = NULL;
    }

    if (!(params->field_mask & UCP_MEM_MAP_PARAM_FIELD_FLAGS)) {
        params->field_mask |= UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params->flags = 0;
    }

    if ((params->flags & UCP_MEM_MAP_FIXED) &&
        (!params->address ||
         ((uintptr_t)params->address % ucs_get_page_size()))) {
        ucs_error("UCP_MEM_MAP_FIXED flag requires page aligned address");
        return UCS_ERR_ILWALID_PARAM;
    }

    /* Now, lets check the rest of erroneous cases from the matrix */
    if (params->address == NULL) {
        if (!(params->flags & UCP_MEM_MAP_ALLOCATE) && (params->length > 0)) {
            ucs_error("Undefined address with nonzero length requires "
                      "UCP_MEM_MAP_ALLOCATE flag");
            return UCS_ERR_ILWALID_PARAM;
        }
    } else if (!(params->flags & UCP_MEM_MAP_ALLOCATE) &&
               (params->flags & UCP_MEM_MAP_FIXED)) {
        ucs_error("Wrong combination of flags when address is defined");
        return UCS_ERR_ILWALID_PARAM;
    }

    return UCS_OK;
}

static inline int ucp_mem_map_is_allocate(ucp_mem_map_params_t *params)
{
    return (params->field_mask & UCP_MEM_MAP_PARAM_FIELD_FLAGS) &&
           (params->flags & UCP_MEM_MAP_ALLOCATE);
}

static ucs_status_t ucp_mem_map_common(ucp_context_h context, void *address,
                                       size_t length, unsigned uct_flags,
                                       int is_allocate, const char *alloc_name,
                                       ucp_mem_h *memh_p)
{
    ucs_status_t            status;
    ucp_mem_h               memh;

    /* Allocate the memory handle */
    ucs_assert(context->num_mds > 0);
    memh = ucs_malloc(sizeof(*memh) + context->num_mds * sizeof(memh->uct[0]),
                      "ucp_memh");
    if (memh == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    memh->address = address;
    memh->length  = length;

    if (is_allocate) {
        ucs_debug("allocating %s at %p length %zu", alloc_name, address, length);
        status = ucp_mem_alloc(context, length, uct_flags, alloc_name, memh);
        if (status != UCS_OK) {
            goto err_free_memh;
        }
    } else {
        memh->mem_type     = ucp_memory_type_detect(context, address, length);
        memh->alloc_method = UCT_ALLOC_METHOD_LAST;
        memh->alloc_md     = NULL;
        memh->md_map       = 0;

        ucs_debug("registering %s %p length %zu mem_type %s", alloc_name,
                  address, length, ucs_memory_type_names[memh->mem_type]);
        status = ucp_mem_rereg_mds(context, UCS_MASK(context->num_mds),
                                   memh->address, memh->length,
                                   uct_flags | UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                   NULL, memh->mem_type, NULL, memh->uct,
                                   &memh->md_map);
        if (status != UCS_OK) {
            goto err_free_memh;
        }
    }

    ucs_debug("%s buffer %p length %zu memh %p md_map 0x%lx",
              (memh->alloc_method == UCT_ALLOC_METHOD_LAST) ? "mapped" : "allocated",
              memh->address, memh->length, memh, memh->md_map);
    *memh_p = memh;
    status  = UCS_OK;
    goto out;

err_free_memh:
    ucs_free(memh);
out:
    return status;
}

static ucs_status_t ucp_mem_unmap_common(ucp_context_h context, ucp_mem_h memh)
{
    uct_allocated_memory_t mem;
    uct_mem_h alloc_md_memh;
    ucs_status_t status;

    ucs_debug("unmapping buffer %p memh %p", memh->address, memh);

    /* Unregister from all memory domains */
    alloc_md_memh = UCT_MEM_HANDLE_NULL;
    status = ucp_mem_rereg_mds(context, 0, NULL, 0, 0, memh->alloc_md, memh->mem_type,
                               &alloc_md_memh, memh->uct, &memh->md_map);
    if (status != UCS_OK) {
        goto out;
    }

    /* If the memory was also allocated, release it */
    if (memh->alloc_method != UCT_ALLOC_METHOD_LAST) {
        mem.address = memh->address;
        mem.length  = memh->length;
        mem.method  = memh->alloc_method;
        mem.md      = memh->alloc_md;  /* May be NULL if method is not MD */
        mem.memh    = alloc_md_memh;   /* May be INVALID if method is not MD */

        status = uct_mem_free(&mem);
        if (status != UCS_OK) {
            goto out;
        }
    }

    ucs_free(memh);
    status = UCS_OK;
out:
    return status;
}

ucs_status_t ucp_mem_map(ucp_context_h context, const ucp_mem_map_params_t *params,
                         ucp_mem_h *memh_p)
{
    ucs_status_t            status;
    ucp_mem_map_params_t    mem_params;

    /* always acquire context lock */
    UCP_THREAD_CS_ENTER(&context->mt_lock);

    mem_params = *params;
    status = ucp_mem_map_check_and_adjust_params(&mem_params);
    if (status != UCS_OK) {
        goto out;
    }

    if (mem_params.length == 0) {
        ucs_debug("mapping zero length buffer, return dummy memh");
        *memh_p = &ucp_mem_dummy_handle;
        status  = UCS_OK;
        goto out;
    }

    status = ucp_mem_map_common(context, mem_params.address, mem_params.length,
                                ucp_mem_map_params2uct_flags(&mem_params),
                                ucp_mem_map_is_allocate(&mem_params),
                                "user memory", memh_p);
out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh)
{
    ucs_status_t status;

    /* always acquire context lock */
    UCP_THREAD_CS_ENTER(&context->mt_lock);

    if (memh == &ucp_mem_dummy_handle) {
        ucs_debug("unmapping zero length buffer (dummy memh, do nothing)");
        status = UCS_OK;
        goto out;
    }

    status = ucp_mem_unmap_common(context, memh);
out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

ucs_status_t ucp_mem_type_reg_buffers(ucp_worker_h worker, void *remote_addr,
                                      size_t length, ucs_memory_type_t mem_type,
                                      ucp_md_index_t md_index, uct_mem_h *memh,
                                      ucp_md_map_t *md_map,
                                      uct_rkey_bundle_t *rkey_bundle)
{
    ucp_context_h context        = worker->context;
    const uct_md_attr_t *md_attr = &context->tl_mds[md_index].attr;
    uct_component_h cmpt;
    ucp_tl_md_t *tl_md;
    ucs_status_t status;
    char *rkey_buffer;

    if (!(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY)) {
        rkey_bundle->handle = NULL;
        rkey_bundle->rkey   = UCT_ILWALID_RKEY;
        status              = UCS_OK;
        goto out;
    }

    tl_md  = &context->tl_mds[md_index];
    cmpt   = context->tl_cmpts[tl_md->cmpt_index].cmpt;

    status = ucp_mem_rereg_mds(context, UCS_BIT(md_index), remote_addr, length,
                               UCT_MD_MEM_ACCESS_ALL |
                               UCT_MD_MEM_FLAG_HIDE_ERRORS,
                               NULL, mem_type, NULL, memh, md_map);
    if (status != UCS_OK) {
        goto out;
    }

    rkey_buffer = ucs_alloca(md_attr->rkey_packed_size);
    status      = uct_md_mkey_pack(tl_md->md, memh[0], rkey_buffer);
    if (status != UCS_OK) {
        ucs_error("failed to pack key from md[%d]: %s",
                  md_index, ucs_status_string(status));
        goto out_dereg_mem;
    }

    status = uct_rkey_unpack(cmpt, rkey_buffer, rkey_bundle);
    if (status != UCS_OK) {
        ucs_error("failed to unpack key from md[%d]: %s",
                  md_index, ucs_status_string(status));
        goto out_dereg_mem;
    }

    return UCS_OK;

out_dereg_mem:
    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL, mem_type, NULL,
                      memh, md_map);
out:
    *memh = UCT_MEM_HANDLE_NULL;
    return status;
}

void ucp_mem_type_unreg_buffers(ucp_worker_h worker, ucs_memory_type_t mem_type,
                                ucp_md_index_t md_index, uct_mem_h *memh,
                                ucp_md_map_t *md_map,
                                uct_rkey_bundle_t *rkey_bundle)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t cmpt_index;

    if (rkey_bundle->rkey != UCT_ILWALID_RKEY) {
        cmpt_index = context->tl_mds[md_index].cmpt_index;
        uct_rkey_release(context->tl_cmpts[cmpt_index].cmpt, rkey_bundle);
    }

    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL, mem_type, NULL,
                      memh, md_map);
}

ucs_status_t ucp_mem_query(const ucp_mem_h memh, ucp_mem_attr_t *attr)
{
    if (attr->field_mask & UCP_MEM_ATTR_FIELD_ADDRESS) {
        attr->address = memh->address;
    }

    if (attr->field_mask & UCP_MEM_ATTR_FIELD_LENGTH) {
        attr->length = memh->length;
    }

    return UCS_OK;
}

static ucs_status_t ucp_advice2uct(unsigned ucp_advice, uct_mem_advice_t *uct_advice)
{
    switch(ucp_advice) {
    case UCP_MADV_NORMAL:
        *uct_advice = UCT_MADV_NORMAL;
        return UCS_OK;
    case UCP_MADV_WILLNEED:
        *uct_advice = UCT_MADV_WILLNEED;
        return UCS_OK;
    }
    return UCS_ERR_ILWALID_PARAM;
}

ucs_status_t
ucp_mem_advise(ucp_context_h context, ucp_mem_h memh,
               ucp_mem_advise_params_t *params)
{
    ucs_status_t status, tmp_status;
    int md_index;
    uct_mem_advice_t uct_advice;
    uct_mem_h uct_memh;

    if (!ucs_test_all_flags(params->field_mask,
                            UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS|
                            UCP_MEM_ADVISE_PARAM_FIELD_LENGTH|
                            UCP_MEM_ADVISE_PARAM_FIELD_ADVICE)) {
        return UCS_ERR_ILWALID_PARAM;
    }

    if ((params->address < memh->address) ||
        (UCS_PTR_BYTE_OFFSET(params->address, params->length) >
         UCS_PTR_BYTE_OFFSET(memh->address, memh->length))) {
        return UCS_ERR_ILWALID_PARAM;
    }

    status = ucp_advice2uct(params->advice, &uct_advice);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("advice buffer %p length %llu memh %p flags %x",
               params->address, (unsigned long long)params->length, memh,
               params->advice);

    if (memh == &ucp_mem_dummy_handle) {
        return UCS_OK;
    }

    UCP_THREAD_CS_ENTER(&context->mt_lock);

    status = UCS_OK;
    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        uct_memh = ucp_memh2uct(memh, md_index);
        if (!(context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_ADVISE) ||
            (uct_memh == NULL)) {
            continue;
        }
        tmp_status = uct_md_mem_advise(context->tl_mds[md_index].md, uct_memh,
                                       params->address, params->length, uct_advice);
        if (tmp_status != UCS_OK) {
            status = tmp_status;
        }
    }

    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

static inline ucs_status_t
ucp_mpool_malloc(ucp_worker_h worker, ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucp_mem_desc_t *chunk_hdr;
    ucp_mem_h memh;
    ucs_status_t status;
    ucp_mem_map_params_t mem_params;

    /* Need to get default flags from ucp_mem_map_params2uct_flags() */
    mem_params.field_mask = 0;
    status = ucp_mem_map_common(worker->context, NULL,
                                *size_p + sizeof(*chunk_hdr),
                                ucp_mem_map_params2uct_flags(&mem_params),
                                1, ucs_mpool_name(mp), &memh);
    if (status != UCS_OK) {
        goto out;
    }

    chunk_hdr       = memh->address;
    chunk_hdr->memh = memh;
    *chunk_p        = chunk_hdr + 1;
    *size_p         = memh->length - sizeof(*chunk_hdr);
out:
    return status;
}

static inline void
ucp_mpool_free(ucp_worker_h worker, ucs_mpool_t *mp, void *chunk)
{
    ucp_mem_desc_t *chunk_hdr;

    chunk_hdr = (ucp_mem_desc_t*)chunk - 1;
    ucp_mem_unmap_common(worker->context, chunk_hdr->memh);
}

void ucp_mpool_obj_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucp_mem_desc_t *elem_hdr  = obj;
    ucp_mem_desc_t *chunk_hdr = (ucp_mem_desc_t*)((ucp_mem_desc_t*)chunk - 1);
    elem_hdr->memh = chunk_hdr->memh;
}

ucs_status_t ucp_reg_mpool_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, reg_mp);

    return ucp_mpool_malloc(worker, mp, size_p, chunk_p);
}

void ucp_reg_mpool_free(ucs_mpool_t *mp, void *chunk)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, reg_mp);

    ucp_mpool_free(worker, mp, chunk);
}

ucs_status_t ucp_frag_mpool_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, rndv_frag_mp);

    return ucp_mpool_malloc(worker, mp, size_p, chunk_p);
}

void ucp_frag_mpool_free(ucs_mpool_t *mp, void *chunk)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, rndv_frag_mp);

    ucp_mpool_free(worker, mp, chunk);
}

void ucp_mem_print_info(const char *mem_size, ucp_context_h context, FILE *stream)
{
    size_t min_page_size, max_page_size;
    ucp_mem_map_params_t mem_params;
    size_t mem_size_value;
    char memunits_str[32];
    ucs_status_t status;
    unsigned md_index;
    ucp_mem_h memh;

    status = ucs_str_to_memunits(mem_size, &mem_size_value);
    if (status != UCS_OK) {
        printf("<Failed to colwert a memunits string>\n");
        return;
    }

    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_params.address    = NULL;
    mem_params.length     = mem_size_value;
    mem_params.flags      = UCP_MEM_MAP_ALLOCATE;

    status = ucp_mem_map(context, &mem_params, &memh);
    if (status != UCS_OK) {
        printf("<Failed to map memory of size %s>\n", mem_size);
        return;
    }

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP memory allocation\n");
    fprintf(stream, "#\n");

    ucs_memunits_to_str(memh->length, memunits_str, sizeof(memunits_str));
    fprintf(stream, "#  allocated %s at address %p with ", memunits_str,
            memh->address);

    if (memh->alloc_md == NULL) {
        fprintf(stream, "%s", uct_alloc_method_names[memh->alloc_method]);
    } else {
        for (md_index = 0; md_index < context->num_mds; ++md_index) {
            if (memh->alloc_md == context->tl_mds[md_index].md) {
                fprintf(stream, "%s", context->tl_mds[md_index].rsc.md_name);
                break;
            }
        }
    }

    ucs_get_mem_page_size(memh->address, memh->length, &min_page_size,
                          &max_page_size);
    ucs_memunits_to_str(min_page_size, memunits_str, sizeof(memunits_str));
    fprintf(stream, ", pagesize: %s", memunits_str);
    if (min_page_size != max_page_size) {
        ucs_memunits_to_str(max_page_size, memunits_str, sizeof(memunits_str));
        fprintf(stream, "-%s", memunits_str);
    }

    fprintf(stream, "\n");
    fprintf(stream, "#  registered on: ");
    ucs_for_each_bit(md_index, memh->md_map) {
        fprintf(stream, "%s ", context->tl_mds[md_index].rsc.md_name);
    }
    fprintf(stream, "\n");
    fprintf(stream, "#\n");

    status = ucp_mem_unmap(context, memh);
    if (status != UCS_OK) {
        printf("<Failed to unmap memory of size %s>\n", mem_size);
    }
}
