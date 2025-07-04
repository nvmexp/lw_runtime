/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) Los Alamos National Security, LLC. 2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "xpmem.h"

#include <uct/sm/mm/base/mm_md.h>
#include <uct/sm/mm/base/mm_iface.h>
#include <ucs/datastruct/khash.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/init_once.h>
#include <ucs/type/spinlock.h>
#include <ucs/memory/rcache.h>
#include <ucs/debug/log.h>


/* XPMEM memory domain configuration */
typedef struct uct_xpmem_md_config {
    uct_mm_md_config_t      super;
} uct_xpmem_md_config_t;

/* Remote process memory */
typedef struct uct_xpmem_remote_mem {
    xpmem_apid_t            apid;
    xpmem_segid_t           xsegid;
    ucs_rcache_t            *rcache;
    int                     refcount;
} uct_xpmem_remote_mem_t;

/* Cache entry for remote memory region */
typedef struct uct_xpmem_remote_region {
    ucs_rcache_region_t     super;
    void                    *attach_address;
    uct_xpmem_remote_mem_t  *rmem;
} uct_xpmem_remote_region_t;

typedef struct uct_xpmem_iface_addr {
    xpmem_segid_t           xsegid;
} UCS_S_PACKED uct_xpmem_iface_addr_t;

typedef struct uct_xpmem_packed_rkey {
    xpmem_segid_t           xsegid;
    uintptr_t               address;
    size_t                  length;
} UCS_S_PACKED uct_xpmem_packed_rkey_t;

KHASH_INIT(xpmem_remote_mem, xpmem_segid_t, uct_xpmem_remote_mem_t*, 1,
           kh_int64_hash_func, kh_int64_hash_equal)

/* Global XPMEM segment which maps the entire process virtual address space */
static ucs_init_once_t uct_xpmem_global_seg_init_once = UCS_INIT_ONCE_INITIALIZER;
static xpmem_segid_t   uct_xpmem_global_xsegid        = -1;

/* Hash of remote regions */
static khash_t(xpmem_remote_mem) uct_xpmem_remote_mem_hash;
static ucs_relwrsive_spinlock_t  uct_xpmem_remote_mem_lock;

static ucs_config_field_t uct_xpmem_md_config_table[] = {
  {"MM_", "", NULL,
   ucs_offsetof(uct_xpmem_md_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_mm_md_config_table)},

  {NULL}
};

UCS_STATIC_INIT {
    ucs_relwrsive_spinlock_init(&uct_xpmem_remote_mem_lock, 0);
    kh_init_inplace(xpmem_remote_mem, &uct_xpmem_remote_mem_hash);
}

UCS_STATIC_CLEANUP {
    uct_xpmem_remote_mem_t *rmem;
    ucs_status_t status;

    kh_foreach_value(&uct_xpmem_remote_mem_hash, rmem, {
        ucs_warn("remote segment id %lx apid %lx is not released, refcount %d",
                 (unsigned long)rmem->xsegid, (unsigned long)rmem->apid,
                 rmem->refcount);
    })
    kh_destroy_inplace(xpmem_remote_mem, &uct_xpmem_remote_mem_hash);

    status = ucs_relwrsive_spinlock_destroy(&uct_xpmem_remote_mem_lock);
    if (status != UCS_OK) {
        ucs_warn("ucs_relwrsive_spinlock_destroy() failed: %s",
                 ucs_status_string(status));
    }
}

static ucs_status_t uct_xpmem_query()
{
    int version;

    version = xpmem_version();
    if (version < 0) {
        ucs_debug("xpmem_version() returned %d (%m), xpmem is unavailable",
                  version);
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_debug("xpmem version: %d", version);
    return UCS_OK;
}

static ucs_status_t uct_xpmem_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    uct_mm_md_query(md, md_attr, 0);

    md_attr->cap.flags         |= UCT_MD_FLAG_REG;
    md_attr->reg_cost.overhead  = 60.0e-9;
    md_attr->reg_cost.growth    = 0;
    md_attr->cap.max_reg        = ULONG_MAX;
    md_attr->cap.reg_mem_types  = UCS_MEMORY_TYPES_CPU_ACCESSIBLE;
    md_attr->rkey_packed_size   = sizeof(uct_xpmem_packed_rkey_t);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE size_t
uct_xpmem_rcache_region_length(uct_xpmem_remote_region_t *xpmem_region)
{
    return xpmem_region->super.super.end - xpmem_region->super.super.start;
}

static ucs_status_t
uct_xpmem_rcache_mem_reg(void *context, ucs_rcache_t *rcache, void *arg,
                         ucs_rcache_region_t *region, uint16_t flags)
{
    uct_xpmem_remote_mem_t    *rmem         = context;
    uct_xpmem_remote_region_t *xpmem_region =
                    ucs_derived_of(region, uct_xpmem_remote_region_t);
    struct xpmem_addr addr;
    size_t length;

    addr.apid   = rmem->apid;
    addr.offset = xpmem_region->super.super.start;
    length      = uct_xpmem_rcache_region_length(xpmem_region);

    xpmem_region->attach_address = xpmem_attach(addr, length, NULL);
    VALGRIND_MAKE_MEM_DEFINED(&xpmem_region->attach_address,
                              sizeof(xpmem_region->attach_address));
    if (xpmem_region->attach_address == MAP_FAILED) {
        ucs_error("failed to attach xpmem apid 0x%lx offset 0x%lx length %zu: %m",
                  (unsigned long)addr.apid, addr.offset, length);
        return UCS_ERR_IO_ERROR;
    }

    xpmem_region->rmem = rmem;

    ucs_trace("xpmem attached apid 0x%lx offset 0x%lx length %zu at %p",
              (unsigned long)addr.apid, addr.offset, length,
              xpmem_region->attach_address);

    VALGRIND_MAKE_MEM_DEFINED(xpmem_region->attach_address, length);
    return UCS_OK;
}

static void uct_xpmem_rcache_mem_dereg(void *context, ucs_rcache_t *rcache,
                                       ucs_rcache_region_t *region)
{
    uct_xpmem_remote_region_t *xpmem_region =
                    ucs_derived_of(region, uct_xpmem_remote_region_t);
    int ret;

    ucs_trace("xpmem detaching address %p", xpmem_region->attach_address);

    ret = xpmem_detach(xpmem_region->attach_address);
    if (ret < 0) {
        ucs_warn("Failed to xpmem_detach: %m");
    }

    xpmem_region->attach_address = NULL;
    xpmem_region->rmem           = NULL;
}

static void uct_xpmem_rcache_dump_region(void *context, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *region, char *buf,
                                         size_t max)
{
    uct_xpmem_remote_mem_t    *rmem         = context;
    uct_xpmem_remote_region_t *xpmem_region =
                    ucs_derived_of(region, uct_xpmem_remote_region_t);

    snprintf(buf, max, "apid 0x%lx attach_addr %p rmem %p",
             (unsigned long)rmem->apid, xpmem_region->attach_address, rmem);
}

static ucs_rcache_ops_t uct_xpmem_rcache_ops = {
    .mem_reg     = uct_xpmem_rcache_mem_reg,
    .mem_dereg   = uct_xpmem_rcache_mem_dereg,
    .dump_region = uct_xpmem_rcache_dump_region
};

static UCS_F_NOINLINE ucs_status_t
uct_xpmem_make_global_xsegid(xpmem_segid_t *xsegid_p)
{
    /* double-checked locking */
    UCS_INIT_ONCE(&uct_xpmem_global_seg_init_once) {
        if (uct_xpmem_global_xsegid < 0) {
            uct_xpmem_global_xsegid = xpmem_make(0, XPMEM_MAXADDR_SIZE,
                                                 XPMEM_PERMIT_MODE, (void*)0600);
            VALGRIND_MAKE_MEM_DEFINED(&uct_xpmem_global_xsegid,
                                      sizeof(uct_xpmem_global_xsegid));
        }
    }

    if (uct_xpmem_global_xsegid < 0) {
        ucs_error("xpmem failed to register process address space: %m");
        return UCS_ERR_IO_ERROR;
    }

    ucs_debug("xpmem registered global segment id 0x%lx",
              (unsigned long)uct_xpmem_global_xsegid);
    *xsegid_p = uct_xpmem_global_xsegid;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_xpmem_get_global_xsegid(xpmem_segid_t *xsegid_p)
{
    if (ucs_unlikely(uct_xpmem_global_xsegid < 0)) {
        return uct_xpmem_make_global_xsegid(xsegid_p);
    }

    *xsegid_p = uct_xpmem_global_xsegid;
    return UCS_OK;
}

/* lock must be held */
static UCS_F_NOINLINE ucs_status_t
uct_xpmem_rmem_add(xpmem_segid_t xsegid, uct_xpmem_remote_mem_t **rmem_p)
{
    ucs_rcache_params_t rcache_params;
    uct_xpmem_remote_mem_t *rmem;
    ucs_status_t status;
    khiter_t khiter;
    int khret;

    rmem = ucs_malloc(sizeof(*rmem), "xpmem_rmem");
    if (rmem == NULL) {
        ucs_error("failed to allocate xpmem rmem");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    rmem->refcount = 0;
    rmem->xsegid   = xsegid;

    rmem->apid = xpmem_get(xsegid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
    VALGRIND_MAKE_MEM_DEFINED(&rmem->apid, sizeof(rmem->apid));
    if (rmem->apid < 0) {
        ucs_error("xpmem_get(segid=0x%lx) failed: %m", (unsigned long)xsegid);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_free;
    }

    rcache_params.region_struct_size = sizeof(uct_xpmem_remote_region_t);
    rcache_params.alignment          = ucs_get_page_size();
    rcache_params.max_alignment      = ucs_get_page_size();
    rcache_params.ucm_events         = 0;
    rcache_params.ucm_event_priority = 0;
    rcache_params.ops                = &uct_xpmem_rcache_ops;
    rcache_params.context            = rmem;

    status = ucs_rcache_create(&rcache_params, "xpmem_remote_mem",
                               ucs_stats_get_root(), &rmem->rcache);
    if (status != UCS_OK) {
        ucs_error("failed to create xpmem remote cache: %s",
                  ucs_status_string(status));
        goto err_release_seg;
    }

    khiter = kh_put(xpmem_remote_mem, &uct_xpmem_remote_mem_hash, xsegid,
                    &khret);
    ucs_assertv_always((khret == 1) || (khret == 2), "khret=%d", khret);
    ucs_assert_always (khiter != kh_end(&uct_xpmem_remote_mem_hash));
    kh_val(&uct_xpmem_remote_mem_hash, khiter) = rmem;

    ucs_trace("xpmem attached to remote segment id 0x%lx apid 0x%lx rcache %p",
              (unsigned long)xsegid, (unsigned long)rmem->apid, rmem->rcache);

    *rmem_p = rmem;
    return UCS_OK;

err_release_seg:
    xpmem_release(rmem->apid);
err_free:
    ucs_free(rmem);
err:
    return status;
}

/* lock must be held */
static UCS_F_NOINLINE void
uct_xpmem_rmem_del(uct_xpmem_remote_mem_t *rmem)
{
    khiter_t khiter;
    int ret;

    ucs_assert(rmem->refcount == 0);

    ucs_trace("detaching remote segment rmem %p apid %lx", rmem,
              (unsigned long)rmem->apid);

    khiter = kh_get(xpmem_remote_mem, &uct_xpmem_remote_mem_hash, rmem->xsegid);
    ucs_assert(kh_val(&uct_xpmem_remote_mem_hash, khiter) == rmem);
    kh_del(xpmem_remote_mem, &uct_xpmem_remote_mem_hash, khiter);

    ucs_rcache_destroy(rmem->rcache);

    ret = xpmem_release(rmem->apid);
    if (ret) {
        ucs_warn("xpmem_release(apid=0x%lx) failed: %m",
                 (unsigned long)rmem->apid);
    }

    ucs_free(rmem);
}

static ucs_status_t
uct_xpmem_rmem_get(xpmem_segid_t xsegid, uct_xpmem_remote_mem_t **rmem_p)
{
    uct_xpmem_remote_mem_t *rmem;
    ucs_status_t status;
    khiter_t khiter;

    ucs_relwrsive_spin_lock(&uct_xpmem_remote_mem_lock);

    khiter = kh_get(xpmem_remote_mem, &uct_xpmem_remote_mem_hash, xsegid);
    if (ucs_likely(khiter != kh_end(&uct_xpmem_remote_mem_hash))) {
        rmem = kh_val(&uct_xpmem_remote_mem_hash, khiter);
    } else {
        status = uct_xpmem_rmem_add(xsegid, &rmem);
        if (status != UCS_OK) {
            *rmem_p = NULL;
            goto out_unlock;
        }
    }

    ++rmem->refcount;
    *rmem_p = rmem;
    status  = UCS_OK;

out_unlock:
    ucs_relwrsive_spin_unlock(&uct_xpmem_remote_mem_lock);
    return status;
}

static void uct_xpmem_rmem_put(uct_xpmem_remote_mem_t *rmem)
{
    ucs_relwrsive_spin_lock(&uct_xpmem_remote_mem_lock);
    if (--rmem->refcount == 0) {
        uct_xpmem_rmem_del(rmem);
    }
    ucs_relwrsive_spin_unlock(&uct_xpmem_remote_mem_lock);
}

static ucs_status_t
uct_xpmem_mem_attach_common(xpmem_segid_t xsegid, uintptr_t remote_address,
                            size_t length, uct_xpmem_remote_region_t **region_p)
{
    ucs_rcache_region_t *rcache_region;
    uct_xpmem_remote_mem_t *rmem;
    uintptr_t start, end;
    ucs_status_t status;

    status = uct_xpmem_rmem_get(xsegid, &rmem);
    if (status != UCS_OK) {
        goto err;
    }

    start = ucs_align_down_pow2(remote_address,          ucs_get_page_size());
    end   = ucs_align_up_pow2  (remote_address + length, ucs_get_page_size());

    status = ucs_rcache_get(rmem->rcache, (void*)start, end - start,
                            PROT_READ|PROT_WRITE, NULL, &rcache_region);
    if (status != UCS_OK) {
        goto err_rmem_put;
    }

    *region_p = ucs_derived_of(rcache_region, uct_xpmem_remote_region_t);
    return UCS_OK;

err_rmem_put:
    uct_xpmem_rmem_put(rmem);
err:
    return status;
}

static void uct_xpmem_mem_detach_common(uct_xpmem_remote_region_t *xpmem_region)
{
    uct_xpmem_remote_mem_t *rmem = xpmem_region->rmem;

    ucs_rcache_region_put(rmem->rcache, &xpmem_region->super);
    uct_xpmem_rmem_put(rmem);
}

static ucs_status_t uct_xmpem_mem_reg(uct_md_h md, void *address, size_t length,
                                      unsigned flags, uct_mem_h *memh_p)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    status = uct_mm_seg_new(address, length, &seg);
    if (status != UCS_OK) {
        return status;
    }

    seg->seg_id  = (uintptr_t)address; /* to be used by mem_attach */
    *memh_p      = seg;
    return UCS_OK;
}

static ucs_status_t uct_xmpem_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_free(seg);
    return UCS_OK;
}

static ucs_status_t
uct_xpmem_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    uct_mm_seg_t                    *seg = memh;
    uct_xpmem_packed_rkey_t *packed_rkey = rkey_buffer;
    xpmem_segid_t xsegid;
    ucs_status_t status;

    ucs_assert((uintptr_t)seg->address == seg->seg_id); /* sanity */

    status = uct_xpmem_get_global_xsegid(&xsegid);
    if (status != UCS_OK) {
        return status;
    }

    packed_rkey->xsegid  = xsegid;
    packed_rkey->address = (uintptr_t)seg->address;
    packed_rkey->length  = seg->length;
    return UCS_OK;
}

static size_t uct_xpmem_iface_addr_length(uct_mm_md_t *md)
{
    return sizeof(uct_xpmem_iface_addr_t);
}

static ucs_status_t uct_xpmem_iface_addr_pack(uct_mm_md_t *md, void *buffer)
{
    uct_xpmem_iface_addr_t *xpmem_iface_addr = buffer;
    xpmem_segid_t xsegid;
    ucs_status_t status;

    status = uct_xpmem_get_global_xsegid(&xsegid);
    if (status != UCS_OK) {
        return status;
    }

    xpmem_iface_addr->xsegid = xsegid;
    return UCS_OK;
}

static ucs_status_t uct_xpmem_mem_attach(uct_mm_md_t *md, uct_mm_seg_id_t seg_id,
                                         size_t length, const void *iface_addr,
                                         uct_mm_remote_seg_t *rseg)
{
    const uct_xpmem_iface_addr_t *xpmem_iface_addr = iface_addr;
    uintptr_t                       remote_address = seg_id;
    uct_xpmem_remote_region_t *xpmem_region;
    ucs_status_t status;
    ptrdiff_t offset;

    ucs_assert(xpmem_iface_addr != NULL);
    status = uct_xpmem_mem_attach_common(xpmem_iface_addr->xsegid,
                                         remote_address, length, &xpmem_region);
    if (status != UCS_OK) {
        return status;
    }

    /* In order to obtain the local access address of the remote segment
     * (rseg->address), we need to callwlate its offset from the beginning of the
     * region on remote side (offset), and then add it to the local base address
     * of the attached region (xpmem_region->attach_address).
     */
    offset        = remote_address - xpmem_region->super.super.start;
    rseg->address = UCS_PTR_BYTE_OFFSET(xpmem_region->attach_address, offset);
    rseg->cookie  = xpmem_region;

    return UCS_OK;
}

static void uct_xpmem_mem_detach(uct_mm_md_t *md,
                                 const uct_mm_remote_seg_t *rseg)
{
    uct_xpmem_mem_detach_common(rseg->cookie);
}

static ucs_status_t
uct_xpmem_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                      uct_rkey_t *rkey_p, void **handle_p)
{
    const uct_xpmem_packed_rkey_t *packed_rkey = rkey_buffer;
    uct_xpmem_remote_region_t *xpmem_region;
    ucs_status_t status;

    status = uct_xpmem_mem_attach_common(packed_rkey->xsegid,
                                         packed_rkey->address,
                                         packed_rkey->length,
                                         &xpmem_region);
    if (status != UCS_OK) {
        return status;
    }

    uct_mm_md_make_rkey(xpmem_region->attach_address,
                        xpmem_region->super.super.start, rkey_p);
    *handle_p = xpmem_region;

    return UCS_OK;
}

static ucs_status_t
uct_xpmem_rkey_release(uct_component_t *component, uct_rkey_t rkey, void *handle)
{
    uct_xpmem_mem_detach_common(handle);
    return UCS_OK;
}

static uct_mm_md_mapper_ops_t uct_xpmem_md_ops = {
    .super = {
        .close                  = uct_mm_md_close,
        .query                  = uct_xpmem_md_query,
        .mem_alloc              = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
        .mem_free               = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
        .mem_advise             = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
        .mem_reg                = uct_xmpem_mem_reg,
        .mem_dereg              = uct_xmpem_mem_dereg,
        .mkey_pack              = uct_xpmem_mkey_pack,
        .is_sockaddr_accessible = (uct_md_is_sockaddr_accessible_func_t)ucs_empty_function_return_zero,
        .detect_memory_type     = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported
    },
   .query                       = uct_xpmem_query,
   .iface_addr_length           = uct_xpmem_iface_addr_length,
   .iface_addr_pack             = uct_xpmem_iface_addr_pack,
   .mem_attach                  = uct_xpmem_mem_attach,
   .mem_detach                  = uct_xpmem_mem_detach,
   .is_reachable                = (uct_mm_mapper_is_reachable_func_t)ucs_empty_function_return_one
};

UCT_MM_TL_DEFINE(xpmem, &uct_xpmem_md_ops, uct_xpmem_rkey_unpack,
                 uct_xpmem_rkey_release, "XPMEM_")
