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
 * Copyright (c) 2009      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012-2013 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi_config.h"
#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/group/group.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Group_translate_ranks = PMPI_Group_translate_ranks
#endif
#define MPI_Group_translate_ranks PMPI_Group_translate_ranks
#endif

static const char FUNC_NAME[] = "MPI_Group_translate_ranks";


int MPI_Group_translate_ranks(MPI_Group group1, int n_ranks, const int ranks1[],
                              MPI_Group group2, int ranks2[])
{
    int err;

    /* check for errors */
    if( MPI_PARAM_CHECK ) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);

        if ((MPI_GROUP_NULL == group1) || (MPI_GROUP_NULL == group2) ||
            (NULL == group1) || (NULL == group2)) {
            return OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD, MPI_ERR_GROUP,
                                          FUNC_NAME);
        }
        if (n_ranks < 0) {
            return OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD, MPI_ERR_GROUP,
                                          FUNC_NAME);
        }
        if (n_ranks > 0 && ((NULL == ranks1) || (NULL == ranks2 ))) {
            return OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD, MPI_ERR_GROUP,
                                          FUNC_NAME);
        }
    }

    if (0 == n_ranks) {
        return MPI_SUCCESS;
    }

    OPAL_CR_ENTER_LIBRARY();

    err = ompi_group_translate_ranks ( group1, n_ranks, ranks1,
                                       group2, ranks2 );
    OMPI_ERRHANDLER_RETURN(err, MPI_COMM_WORLD, err, FUNC_NAME );
}
