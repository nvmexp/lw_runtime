/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2012-2013 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017      IBM Corporation. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi/mpi/tool/mpit-internal.h"

#if OPAL_HAVE_WEAK_SYMBOLS && OMPI_PROFILING_DEFINES
#pragma weak MPI_T_category_changed = PMPI_T_category_changed
#endif

#if OMPI_PROFILING_DEFINES
#include "ompi/mpi/tool/profile/defines.h"
#endif


int MPI_T_category_changed(int *stamp)
{
    if (!mpit_is_initialized ()) {
        return MPI_T_ERR_NOT_INITIALIZED;
    }

    ompi_mpit_lock ();
    *stamp = mca_base_var_group_get_stamp ();
    ompi_mpit_unlock ();

    return MPI_SUCCESS;
}
