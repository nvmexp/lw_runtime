/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2011      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2011-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014      Intel, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "orte_config.h"
#include "orte/constants.h"

#include "orte/mca/mca.h"
#include "opal/runtime/opal_params.h"

#include "orte/util/proc_info.h"

#include "grpcomm_rcd.h"

static int my_priority=5;
static int rcd_open(void);
static int rcd_close(void);
static int rcd_query(mca_base_module_t **module, int *priority);
static int rcd_register(void);

/*
 * Struct of function pointers that need to be initialized
 */
orte_grpcomm_base_component_t mca_grpcomm_rcd_component = {
    .base_version = {
        ORTE_GRPCOMM_BASE_VERSION_3_0_0,

        .mca_component_name = "rcd",
        MCA_BASE_MAKE_VERSION(component, ORTE_MAJOR_VERSION, ORTE_MINOR_VERSION,
                              ORTE_RELEASE_VERSION),
        .mca_open_component = rcd_open,
        .mca_close_component = rcd_close,
        .mca_query_component = rcd_query,
        .mca_register_component_params = rcd_register,
    },
    .base_data = {
        /* The component is checkpoint ready */
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },
};

static int rcd_register(void)
{
    mca_base_component_t *c = &mca_grpcomm_rcd_component.base_version;

    /* make the priority adjustable so users can select
     * rcd for use by apps without affecting daemons
     */
    my_priority = 80;
    (void) mca_base_component_var_register(c, "priority",
                                           "Priority of the grpcomm rcd component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &my_priority);
    return ORTE_SUCCESS;
}

/* Open the component */
static int rcd_open(void)
{
    return ORTE_SUCCESS;
}

static int rcd_close(void)
{
    return ORTE_SUCCESS;
}

static int rcd_query(mca_base_module_t **module, int *priority)
{
    *priority = my_priority;
    *module = (mca_base_module_t *)&orte_grpcomm_rcd_module;
    return ORTE_SUCCESS;
}
