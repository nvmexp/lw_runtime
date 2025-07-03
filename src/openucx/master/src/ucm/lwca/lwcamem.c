/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/lwca/lwdamem.h>

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucm/util/sys.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>

#include <sys/mman.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>


UCM_DEFINE_REPLACE_DLSYM_FUNC(lwMemFree, LWresult, -1, LWdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwMemFreeHost, LWresult, -1, void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwMemAlloc, LWresult, -1, LWdeviceptr *, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwMemAllocManaged, LWresult, -1, LWdeviceptr *,
                              size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwMemAllocPitch, LWresult, -1, LWdeviceptr *, size_t *,
                              size_t, size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwMemHostGetDevicePointer, LWresult, -1, LWdeviceptr *,
                              void *, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwMemHostUnregister, LWresult, -1, void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwdaFree, lwdaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwdaFreeHost, lwdaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwdaMalloc, lwdaError_t, -1, void**, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwdaMallocManaged, lwdaError_t, -1, void**, size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwdaMallocPitch, lwdaError_t, -1, void**, size_t *,
                              size_t, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwdaHostGetDevicePointer, lwdaError_t, -1, void**,
                              void *, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(lwdaHostUnregister, lwdaError_t, -1, void*)

#if ENABLE_SYMBOL_OVERRIDE
UCM_OVERRIDE_FUNC(lwMemFree,                 LWresult)
UCM_OVERRIDE_FUNC(lwMemFreeHost,             LWresult)
UCM_OVERRIDE_FUNC(lwMemAlloc,                LWresult)
UCM_OVERRIDE_FUNC(lwMemAllocManaged,         LWresult)
UCM_OVERRIDE_FUNC(lwMemAllocPitch,           LWresult)
UCM_OVERRIDE_FUNC(lwMemHostGetDevicePointer, LWresult)
UCM_OVERRIDE_FUNC(lwMemHostUnregister,       LWresult)
UCM_OVERRIDE_FUNC(lwdaFree,                  lwdaError_t)
UCM_OVERRIDE_FUNC(lwdaFreeHost,              lwdaError_t)
UCM_OVERRIDE_FUNC(lwdaMalloc,                lwdaError_t)
UCM_OVERRIDE_FUNC(lwdaMallocManaged,         lwdaError_t)
UCM_OVERRIDE_FUNC(lwdaMallocPitch,           lwdaError_t)
UCM_OVERRIDE_FUNC(lwdaHostGetDevicePointer,  lwdaError_t)
UCM_OVERRIDE_FUNC(lwdaHostUnregister,        lwdaError_t)
#endif


static void ucm_lwda_set_ptr_attr(LWdeviceptr dptr)
{
    if ((void*)dptr == NULL) {
        ucm_trace("skipping lwPointerSetAttribute for null pointer");
        return;
    }

    unsigned int value = 1;
    LWresult ret;
    const char *lw_err_str;

    ret = lwPointerSetAttribute(&value, LW_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
    if (ret != LWDA_SUCCESS) {
        lwGetErrorString(ret, &lw_err_str);
        ucm_warn("lwPointerSetAttribute(%p) failed: %s", (void *) dptr, lw_err_str);
    }
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_alloc(void *addr, size_t length, ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_ALLOC, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_free(void *addr, size_t length, ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_FREE, &event);
}

static void ucm_lwdafree_dispatch_events(void *dptr)
{
    LWresult ret;
    LWdeviceptr pbase;
    size_t psize;

    if (dptr == NULL) {
        return;
    }

    ret = lwMemGetAddressRange(&pbase, &psize, (LWdeviceptr) dptr);
    if (ret == LWDA_SUCCESS) {
        ucs_assert(dptr == (void *)pbase);
    } else {
        ucm_debug("lwMemGetAddressRange(devPtr=%p) failed", (void *)dptr);
        psize = 1; /* set minimum length */
    }

    ucm_dispatch_mem_type_free((void *)dptr, psize, UCS_MEMORY_TYPE_LWDA);
}

LWresult ucm_lwMemFree(LWdeviceptr dptr)
{
    LWresult ret;

    ucm_event_enter();

    ucm_trace("ucm_lwMemFree(dptr=%p)",(void *)dptr);

    ucm_lwdafree_dispatch_events((void *)dptr);

    ret = ucm_orig_lwMemFree(dptr);

    ucm_event_leave();
    return ret;
}

LWresult ucm_lwMemFreeHost(void *p)
{
    LWresult ret;

    ucm_event_enter();

    ucm_trace("ucm_lwMemFreeHost(ptr=%p)", p);

    ucm_dispatch_vm_munmap(p, 0);

    ret = ucm_orig_lwMemFreeHost(p);

    ucm_event_leave();
    return ret;
}

LWresult ucm_lwMemAlloc(LWdeviceptr *dptr, size_t size)
{
    LWresult ret;

    ucm_event_enter();

    ret = ucm_orig_lwMemAlloc(dptr, size);
    if (ret == LWDA_SUCCESS) {
        ucm_trace("ucm_lwMemAlloc(dptr=%p size:%lu)",(void *)*dptr, size);
        ucm_dispatch_mem_type_alloc((void *)*dptr, size, UCS_MEMORY_TYPE_LWDA);
        ucm_lwda_set_ptr_attr(*dptr);
    }

    ucm_event_leave();
    return ret;
}

LWresult ucm_lwMemAllocManaged(LWdeviceptr *dptr, size_t size, unsigned int flags)
{
    LWresult ret;

    ucm_event_enter();

    ret = ucm_orig_lwMemAllocManaged(dptr, size, flags);
    if (ret == LWDA_SUCCESS) {
        ucm_trace("ucm_lwMemAllocManaged(dptr=%p size:%lu, flags:%d)",
                  (void *)*dptr, size, flags);
        ucm_dispatch_mem_type_alloc((void *)*dptr, size,
                                    UCS_MEMORY_TYPE_LWDA_MANAGED);
    }

    ucm_event_leave();
    return ret;
}

LWresult ucm_lwMemAllocPitch(LWdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height,
                             unsigned int ElementSizeBytes)
{
    LWresult ret;

    ucm_event_enter();

    ret = ucm_orig_lwMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    if (ret == LWDA_SUCCESS) {
        ucm_trace("ucm_lwMemAllocPitch(dptr=%p size:%lu)",(void *)*dptr,
                  (WidthInBytes * Height));
        ucm_dispatch_mem_type_alloc((void *)*dptr, WidthInBytes * Height,
                                    UCS_MEMORY_TYPE_LWDA);
        ucm_lwda_set_ptr_attr(*dptr);
    }

    ucm_event_leave();
    return ret;
}

LWresult ucm_lwMemHostGetDevicePointer(LWdeviceptr *pdptr, void *p, unsigned int Flags)
{
    LWresult ret;

    ucm_event_enter();

    ret = ucm_orig_lwMemHostGetDevicePointer(pdptr, p, Flags);
    if (ret == LWDA_SUCCESS) {
        ucm_trace("ucm_lwMemHostGetDevicePointer(pdptr=%p p=%p)",(void *)*pdptr, p);
    }

    ucm_event_leave();
    return ret;
}

LWresult ucm_lwMemHostUnregister(void *p)
{
    LWresult ret;

    ucm_event_enter();

    ucm_trace("ucm_lwMemHostUnregister(ptr=%p)", p);

    ret = ucm_orig_lwMemHostUnregister(p);

    ucm_event_leave();
    return ret;
}

lwdaError_t ucm_lwdaFree(void *devPtr)
{
    lwdaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_lwdaFree(devPtr=%p)", devPtr);

    ucm_lwdafree_dispatch_events((void *)devPtr);

    ret = ucm_orig_lwdaFree(devPtr);

    ucm_event_leave();

    return ret;
}

lwdaError_t ucm_lwdaFreeHost(void *ptr)
{
    lwdaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_lwdaFreeHost(ptr=%p)", ptr);

    ucm_dispatch_vm_munmap(ptr, 0);

    ret = ucm_orig_lwdaFreeHost(ptr);

    ucm_event_leave();
    return ret;
}

lwdaError_t ucm_lwdaMalloc(void **devPtr, size_t size)
{
    lwdaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_lwdaMalloc(devPtr, size);
    if (ret == lwdaSuccess) {
        ucm_trace("ucm_lwdaMalloc(devPtr=%p size:%lu)", *devPtr, size);
        ucm_dispatch_mem_type_alloc(*devPtr, size, UCS_MEMORY_TYPE_LWDA);
        ucm_lwda_set_ptr_attr((LWdeviceptr) *devPtr);
    }

    ucm_event_leave();

    return ret;
}

lwdaError_t ucm_lwdaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    lwdaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_lwdaMallocManaged(devPtr, size, flags);
    if (ret == lwdaSuccess) {
        ucm_trace("ucm_lwdaMallocManaged(devPtr=%p size:%lu flags:%d)",
                  *devPtr, size, flags);
        ucm_dispatch_mem_type_alloc(*devPtr, size, UCS_MEMORY_TYPE_LWDA_MANAGED);
    }

    ucm_event_leave();

    return ret;
}

lwdaError_t ucm_lwdaMallocPitch(void **devPtr, size_t *pitch,
                                size_t width, size_t height)
{
    lwdaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_lwdaMallocPitch(devPtr, pitch, width, height);
    if (ret == lwdaSuccess) {
        ucm_trace("ucm_lwdaMallocPitch(devPtr=%p size:%lu)",*devPtr, (width * height));
        ucm_dispatch_mem_type_alloc(*devPtr, (width * height), UCS_MEMORY_TYPE_LWDA);
        ucm_lwda_set_ptr_attr((LWdeviceptr) *devPtr);
    }

    ucm_event_leave();
    return ret;
}

lwdaError_t ucm_lwdaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    lwdaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_lwdaHostGetDevicePointer(pDevice, pHost, flags);
    if (ret == lwdaSuccess) {
        ucm_trace("ucm_lwMemHostGetDevicePointer(pDevice=%p pHost=%p)", pDevice, pHost);
    }

    ucm_event_leave();
    return ret;
}

lwdaError_t ucm_lwdaHostUnregister(void *ptr)
{
    lwdaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_lwdaHostUnregister(ptr=%p)", ptr);

    ret = ucm_orig_lwdaHostUnregister(ptr);

    ucm_event_leave();
    return ret;
}

static ucm_reloc_patch_t patches[] = {
    {UCS_PP_MAKE_STRING(lwMemFree),                 ucm_override_lwMemFree},
    {UCS_PP_MAKE_STRING(lwMemFreeHost),             ucm_override_lwMemFreeHost},
    {UCS_PP_MAKE_STRING(lwMemAlloc),                ucm_override_lwMemAlloc},
    {UCS_PP_MAKE_STRING(lwMemAllocManaged),         ucm_override_lwMemAllocManaged},
    {UCS_PP_MAKE_STRING(lwMemAllocPitch),           ucm_override_lwMemAllocPitch},
    {UCS_PP_MAKE_STRING(lwMemHostGetDevicePointer), ucm_override_lwMemHostGetDevicePointer},
    {UCS_PP_MAKE_STRING(lwMemHostUnregister),       ucm_override_lwMemHostUnregister},
    {UCS_PP_MAKE_STRING(lwdaFree),                  ucm_override_lwdaFree},
    {UCS_PP_MAKE_STRING(lwdaFreeHost),              ucm_override_lwdaFreeHost},
    {UCS_PP_MAKE_STRING(lwdaMalloc),                ucm_override_lwdaMalloc},
    {UCS_PP_MAKE_STRING(lwdaMallocManaged),         ucm_override_lwdaMallocManaged},
    {UCS_PP_MAKE_STRING(lwdaMallocPitch),           ucm_override_lwdaMallocPitch},
    {UCS_PP_MAKE_STRING(lwdaHostGetDevicePointer),  ucm_override_lwdaHostGetDevicePointer},
    {UCS_PP_MAKE_STRING(lwdaHostUnregister),        ucm_override_lwdaHostUnregister},
    {NULL,                                          NULL}
};

static ucs_status_t ucm_lwdamem_install(int events)
{
    static int ucm_lwdamem_installed = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    ucs_status_t status = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        goto out;
    }

    if (!ucm_global_opts.enable_lwda_reloc) {
        ucm_debug("installing lwdamem relocations is disabled by configuration");
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

    pthread_mutex_lock(&install_mutex);

    if (ucm_lwdamem_installed) {
        goto out_unlock;
    }

    for (patch = patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'", patch->symbol);
            goto out_unlock;
        }
    }

    ucm_debug("lwdaFree hooks are ready");
    ucm_lwdamem_installed = 1;

out_unlock:
    pthread_mutex_unlock(&install_mutex);
out:
    return status;
}

static int ucm_lwdamem_scan_regions_cb(void *arg, void *addr, size_t length,
                                       int prot, const char *path)
{
    static const char *lwda_path_pattern = "/dev/lwpu";
    ucm_event_handler_t *handler         = arg;
    ucm_event_t event;

    /* we are interested in blocks which don't have any access permissions, or
     * mapped to lwpu device.
     */
    if ((prot & (PROT_READ|PROT_WRITE|PROT_EXEC)) &&
        strncmp(path, lwda_path_pattern, strlen(lwda_path_pattern))) {
        return 0;
    }

    ucm_debug("dispatching initial memtype allocation for %p..%p %s",
              addr, UCS_PTR_BYTE_OFFSET(addr, length), path);

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = UCS_MEMORY_TYPE_LAST; /* unknown memory type */

    ucm_event_enter();
    handler->cb(UCM_EVENT_MEM_TYPE_ALLOC, &event, handler->arg);
    ucm_event_leave();

    return 0;
}

static void ucm_lwdamem_get_existing_alloc(ucm_event_handler_t *handler)
{
    if (handler->events & UCM_EVENT_MEM_TYPE_ALLOC) {
        ucm_parse_proc_self_maps(ucm_lwdamem_scan_regions_cb, handler);
    }
}

static ucm_event_installer_t ucm_lwda_initializer = {
    .install            = ucm_lwdamem_install,
    .get_existing_alloc = ucm_lwdamem_get_existing_alloc
};

UCS_STATIC_INIT {
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_lwda_initializer.list);
}

UCS_STATIC_CLEANUP {
    ucs_list_del(&ucm_lwda_initializer.list);
}
