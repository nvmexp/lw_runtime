/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2008 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "orte_config.h"
#include "orte/constants.h"

#include "opal/mca/base/base.h"
#include "opal/mca/base/mca_base_var.h"

#include "orte/mca/rmaps/base/rmaps_private.h"
#include "rmaps_mindist.h"

/*
 * Local functions
 */

static int orte_rmaps_mindist_open(void);
static int orte_rmaps_mindist_close(void);
static int orte_rmaps_mindist_query(mca_base_module_t **module, int *priority);
static int orte_rmaps_mindist_register(void);

static int my_priority = 20;

orte_rmaps_base_component_t mca_rmaps_mindist_component = {
    .base_version = {
        ORTE_RMAPS_BASE_VERSION_2_0_0,

        .mca_component_name = "mindist",
        MCA_BASE_MAKE_VERSION(component, ORTE_MAJOR_VERSION, ORTE_MINOR_VERSION,
                              ORTE_RELEASE_VERSION),
        .mca_open_component = orte_rmaps_mindist_open,
        .mca_close_component = orte_rmaps_mindist_close,
        .mca_query_component = orte_rmaps_mindist_query,
        .mca_register_component_params = orte_rmaps_mindist_register,
    },
    .base_data = {
        /* The component is checkpoint ready */
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },
};


static int orte_rmaps_mindist_register(void)
{
    (void) mca_base_component_var_register(&mca_rmaps_mindist_component.base_version,
                                           "priority", "Priority of the mindist rmaps component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &my_priority);
    return ORTE_SUCCESS;
}

/**
  * component open/close/init function
  */
static int orte_rmaps_mindist_open(void)
{
    return ORTE_SUCCESS;
}


static int orte_rmaps_mindist_query(mca_base_module_t **module, int *priority)
{
    /* the RMAPS framework is -only- opened on HNP's,
     * so no need to check for that here
     */

    *priority = my_priority;
    *module = (mca_base_module_t *)&orte_rmaps_mindist_module;
    return ORTE_SUCCESS;
}

/**
 *  Close all subsystems.
 */

static int orte_rmaps_mindist_close(void)
{
    return ORTE_SUCCESS;
}

