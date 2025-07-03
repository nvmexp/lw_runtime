/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, LWPU CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCT_LWDA_IPC_MD_H
#define UCT_LWDA_IPC_MD_H

#include <uct/base/uct_md.h>
#include <uct/lwca/base/lwda_md.h>
#include <uct/lwca/base/lwda_iface.h>


/**
 * @brief lwca ipc MD descriptor
 */
typedef struct uct_lwda_ipc_md {
    struct uct_md super;   /**< Domain info */
    LWuuid*       uuid_map;
    char*         peer_accessible_cache;
    int           uuid_map_size;
    int           uuid_map_capacity;
} uct_lwda_ipc_md_t;

/**
 * @brief lwca ipc component extension
 */
typedef struct uct_lwda_ipc_component {
    uct_component_t    super;
    uct_lwda_ipc_md_t* md;
} uct_lwda_ipc_component_t;

extern uct_lwda_ipc_component_t uct_lwda_ipc_component;

/**
 * @brief lwca ipc domain configuration.
 */
typedef struct uct_lwda_ipc_md_config {
    uct_md_config_t super;
} uct_lwda_ipc_md_config_t;


/**
 * @brief lwda_ipc packed and remote key for put/get
 */
typedef struct uct_lwda_ipc_key {
    LWipcMemHandle ph;           /* Memory handle of GPU memory */
    LWdeviceptr    d_bptr;       /* Allocation base address */
    size_t         b_len;        /* Allocation size */
    int            dev_num;      /* GPU Device number */
    LWuuid         uuid;         /* GPU Device UUID */
    LWdeviceptr    d_mapped;     /* Locally mapped device address */
} uct_lwda_ipc_key_t;


#define UCT_LWDA_IPC_GET_DEVICE(_lw_device)                             \
    do {                                                                \
        if (UCS_OK != UCT_LWDADRV_FUNC(lwCtxGetDevice(&_lw_device))) {  \
            return UCS_ERR_IO_ERROR;                                    \
        }                                                               \
    } while(0);

#define UCT_LWDA_IPC_DEVICE_GET_COUNT(_num_device)                        \
    do {                                                                  \
        if (UCS_OK != UCT_LWDADRV_FUNC(lwDeviceGetCount(&_num_device))) { \
            return UCS_ERR_IO_ERROR;                                      \
        }                                                                 \
    } while(0);

#endif
