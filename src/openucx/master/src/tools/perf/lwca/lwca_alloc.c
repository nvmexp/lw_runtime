/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <lwca.h>
#include <lwda_runtime.h>
#include <ucs/sys/compiler.h>


static ucs_status_t ucx_perf_lwda_init(ucx_perf_context_t *perf)
{
    lwdaError_t cerr;
    unsigned group_index;
    int num_gpus;
    int gpu_index;

    group_index = rte_call(perf, group_index);

    cerr = lwdaGetDeviceCount(&num_gpus);
    if (cerr != lwdaSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    gpu_index = group_index % num_gpus;

    cerr = lwdaSetDevice(gpu_index);
    if (cerr != lwdaSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

static inline ucs_status_t ucx_perf_lwda_alloc(size_t length,
                                               ucs_memory_type_t mem_type,
                                               void **address_p)
{
    lwdaError_t cerr;

    ucs_assert((mem_type == UCS_MEMORY_TYPE_LWDA) ||
               (mem_type == UCS_MEMORY_TYPE_LWDA_MANAGED));

    cerr = ((mem_type == UCS_MEMORY_TYPE_LWDA) ?
            lwdaMalloc(address_p, length) :
            lwdaMallocManaged(address_p, length, lwdaMemAttachGlobal));
    if (cerr != lwdaSuccess) {
        ucs_error("failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_lwda_alloc(const ucx_perf_context_t *perf, size_t length,
                                        void **address_p, ucp_mem_h *memh_p,
                                        int non_blk_flag)
{
    return ucx_perf_lwda_alloc(length, UCS_MEMORY_TYPE_LWDA, address_p);
}

static ucs_status_t ucp_perf_lwda_alloc_managed(const ucx_perf_context_t *perf,
                                                size_t length, void **address_p,
                                                ucp_mem_h *memh_p, int non_blk_flag)
{
    return ucx_perf_lwda_alloc(length, UCS_MEMORY_TYPE_LWDA_MANAGED, address_p);
}

static void ucp_perf_lwda_free(const ucx_perf_context_t *perf,
                               void *address, ucp_mem_h memh)
{
    lwdaFree(address);
}

static inline ucs_status_t
uct_perf_lwda_alloc_reg_mem(const ucx_perf_context_t *perf,
                            size_t length,
                            ucs_memory_type_t mem_type,
                            unsigned flags,
                            uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = ucx_perf_lwda_alloc(length, mem_type, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_md_mem_reg(perf->uct.md, alloc_mem->address,
                            length, flags, &alloc_mem->memh);
    if (status != UCS_OK) {
        lwdaFree(alloc_mem->address);
        ucs_error("failed to register memory");
        return status;
    }

    alloc_mem->mem_type = mem_type;
    alloc_mem->md       = perf->uct.md;

    return UCS_OK;
}

static ucs_status_t uct_perf_lwda_alloc(const ucx_perf_context_t *perf,
                                        size_t length, unsigned flags,
                                        uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_lwda_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_LWDA,
                                       flags, alloc_mem);
}

static ucs_status_t uct_perf_lwda_managed_alloc(const ucx_perf_context_t *perf,
                                                size_t length, unsigned flags,
                                                uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_lwda_alloc_reg_mem(perf, length, UCS_MEMORY_TYPE_LWDA_MANAGED,
                                       flags, alloc_mem);
}

static void uct_perf_lwda_free(const ucx_perf_context_t *perf,
                               uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }

    lwdaFree(alloc_mem->address);
}

static void ucx_perf_lwda_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                                 const void *src, ucs_memory_type_t src_mem_type,
                                 size_t count)
{
    lwdaError_t cerr;

    cerr = lwdaMemcpy(dst, src, count, lwdaMemcpyDefault);
    if (cerr != lwdaSuccess) {
        ucs_error("failed to copy memory: %s", lwdaGetErrorString(cerr));
    }
}

static void* ucx_perf_lwda_memset(void *dst, int value, size_t count)
{
    lwdaError_t cerr;

    cerr = lwdaMemset(dst, value, count);
    if (cerr != lwdaSuccess) {
        ucs_error("failed to set memory: %s", lwdaGetErrorString(cerr));
    }

    return dst;
}

UCS_STATIC_INIT {
    static ucx_perf_allocator_t lwda_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_LWDA,
        .init      = ucx_perf_lwda_init,
        .ucp_alloc = ucp_perf_lwda_alloc,
        .ucp_free  = ucp_perf_lwda_free,
        .uct_alloc = uct_perf_lwda_alloc,
        .uct_free  = uct_perf_lwda_free,
        .memcpy    = ucx_perf_lwda_memcpy,
        .memset    = ucx_perf_lwda_memset
    };
    static ucx_perf_allocator_t lwda_managed_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_LWDA_MANAGED,
        .init      = ucx_perf_lwda_init,
        .ucp_alloc = ucp_perf_lwda_alloc_managed,
        .ucp_free  = ucp_perf_lwda_free,
        .uct_alloc = uct_perf_lwda_managed_alloc,
        .uct_free  = uct_perf_lwda_free,
        .memcpy    = ucx_perf_lwda_memcpy,
        .memset    = ucx_perf_lwda_memset
    };

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_LWDA]         = &lwda_allocator;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_LWDA_MANAGED] = &lwda_managed_allocator;
}
UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_LWDA]         = NULL;
    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_LWDA_MANAGED] = NULL;

}
