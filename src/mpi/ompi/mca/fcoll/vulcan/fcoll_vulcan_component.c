/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2008-2017 University of Houston. All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "ompi_config.h"
#include "fcoll_vulcan.h"
#include "mpi.h"

/*
 * Public string showing the fcoll ompi_vulcan component version number
 */
const char *mca_fcoll_vulcan_component_version_string =
    "Open MPI vulcan collective MCA component version " OMPI_VERSION;

/*
 * Global variables
 */
int mca_fcoll_vulcan_priority = 10;
int mca_fcoll_vulcan_num_groups = 1;
int mca_fcoll_vulcan_write_chunksize = -1;
int mca_fcoll_vulcan_async_io = 0;

/*
 * Local function
 */
static int vulcan_register(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
mca_fcoll_base_component_2_0_0_t mca_fcoll_vulcan_component = {

    /* First, the mca_component_t struct containing meta information
     * about the component itself */

    .fcollm_version = {
        MCA_FCOLL_BASE_VERSION_2_0_0,

        /* Component name and version */
        .mca_component_name = "vulcan",
        MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                              OMPI_RELEASE_VERSION),
        .mca_register_component_params = vulcan_register,
    },
    .fcollm_data = {
        /* The component is checkpoint ready */
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },

    .fcollm_init_query = mca_fcoll_vulcan_component_init_query,
    .fcollm_file_query = mca_fcoll_vulcan_component_file_query,
    .fcollm_file_unquery = mca_fcoll_vulcan_component_file_unquery,
};


static int
vulcan_register(void)
{
    mca_fcoll_vulcan_priority = 10;
    (void) mca_base_component_var_register(&mca_fcoll_vulcan_component.fcollm_version,
                                           "priority", "Priority of the vulcan fcoll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &mca_fcoll_vulcan_priority);

    mca_fcoll_vulcan_num_groups = 1;
    (void) mca_base_component_var_register(&mca_fcoll_vulcan_component.fcollm_version,
                                           "num_groups", "Number of subgroups created by the vulcan component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &mca_fcoll_vulcan_num_groups);

    mca_fcoll_vulcan_write_chunksize = -1;
    (void) mca_base_component_var_register(&mca_fcoll_vulcan_component.fcollm_version,
                                           "write_chunksize", "Chunk size written at once. Default: stripe_size of the file system",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &mca_fcoll_vulcan_write_chunksize);

    mca_fcoll_vulcan_async_io = 0;
    (void) mca_base_component_var_register(&mca_fcoll_vulcan_component.fcollm_version,
                                           "async_io", "Asynchronous I/O support options. 0: Automatic choice (default) "
                                           "1: Asynchronous I/O only. 2: Synchronous I/O only.",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &mca_fcoll_vulcan_async_io);

    return OMPI_SUCCESS;
}
