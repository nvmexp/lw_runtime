/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_LWDA_MD_H
#define UCT_LWDA_MD_H

#include <uct/base/uct_md.h>

ucs_status_t uct_lwda_base_detect_memory_type(uct_md_h md, const void *addr,
                                              size_t length,
                                              ucs_memory_type_t *mem_type_p);

ucs_status_t
uct_lwda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);

#endif
