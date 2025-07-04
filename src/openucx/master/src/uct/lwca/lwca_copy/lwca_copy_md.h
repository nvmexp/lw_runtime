/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_LWDA_COPY_MD_H
#define UCT_LWDA_COPY_MD_H

#include <uct/base/uct_md.h>
#include <uct/lwca/base/lwda_md.h>


extern uct_component_t uct_lwda_copy_component;

/**
 * @brief lwda_copy MD descriptor
 */
typedef struct uct_lwda_copy_md {
    struct uct_md super;   /**< Domain info */
} uct_lwda_copy_md_t;

/**
 * gdr copy domain configuration.
 */
typedef struct uct_lwda_copy_md_config {
    uct_md_config_t super;
} uct_lwda_copy_md_config_t;

#endif
