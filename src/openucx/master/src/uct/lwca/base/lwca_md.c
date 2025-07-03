/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lwda_md.h"

#include <ucs/sys/module.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/log.h>
#include <lwda_runtime.h>
#include <lwca.h>


UCS_PROFILE_FUNC(ucs_status_t, uct_lwda_base_detect_memory_type,
                 (md, addr, length, mem_type_p),
                 uct_md_h md, const void *addr, size_t length,
                 ucs_memory_type_t *mem_type_p)
{
    LWmemorytype memType = (LWmemorytype)0;
    uint32_t isManaged   = 0;
    unsigned value       = 1;
    void *attrdata[] = {(void *)&memType, (void *)&isManaged};
    LWpointer_attribute attributes[2] = {LW_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         LW_POINTER_ATTRIBUTE_IS_MANAGED};
    LWresult lw_err;
    const char *lw_err_str;

    if (addr == NULL) {
        *mem_type_p = UCS_MEMORY_TYPE_HOST;
        return UCS_OK;
    }

    lw_err = lwPointerGetAttributes(2, attributes, attrdata, (LWdeviceptr)addr);
    if ((lw_err == LWDA_SUCCESS) && (memType == LW_MEMORYTYPE_DEVICE)) {
        if (isManaged) {
            *mem_type_p = UCS_MEMORY_TYPE_LWDA_MANAGED;
        } else {
            *mem_type_p = UCS_MEMORY_TYPE_LWDA;
            lw_err = lwPointerSetAttribute(&value, LW_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                           (LWdeviceptr)addr);
            if (lw_err != LWDA_SUCCESS) {
                lwGetErrorString(lw_err, &lw_err_str);
                ucs_warn("lwPointerSetAttribute(%p) error: %s", (void*) addr, lw_err_str);
            }
        }
        return UCS_OK;
    }

    return UCS_ERR_ILWALID_ADDR;
}

ucs_status_t
uct_lwda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    lwdaError_t lwdaErr;
    int num_gpus;

    lwdaErr = lwdaGetDeviceCount(&num_gpus);
    if ((lwdaErr != lwdaSuccess) || (num_gpus == 0)) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of lwca */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_lwda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_lwda, 0);
    return UCS_OK;
}
