/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2007 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/mpi/fortran/base/fint_2_int.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Errhandler_f2c = PMPI_Errhandler_f2c
#endif
#define MPI_Errhandler_f2c PMPI_Errhandler_f2c
#endif

static const char FUNC_NAME[] = "MPI_Errhandler_f2c";


MPI_Errhandler MPI_Errhandler_f2c(MPI_Fint errhandler_f)
{
    int eh_index = OMPI_FINT_2_INT(errhandler_f);

    OPAL_CR_NOOP_PROGRESS();

    /* Error checking */

    if (MPI_PARAM_CHECK) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
    }

    /* Per MPI-2:4.12.4, do not ilwoke an error handler if we get an
       invalid fortran handle.  If we get an invalid fortran handle,
       return an invalid C handle. */

    if (eh_index < 0 ||
        eh_index >=
        opal_pointer_array_get_size(&ompi_errhandler_f_to_c_table)) {
        return NULL;
    }

    return (MPI_Errhandler)opal_pointer_array_get_item(&ompi_errhandler_f_to_c_table,
                                                       eh_index);
}
