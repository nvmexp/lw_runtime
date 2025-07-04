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
 * Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/fortran/mpif-h/bindings.h"
#include "ompi/constants.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mpi/fortran/base/fortran_base_strings.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_TYPE_SET_NAME = ompi_type_set_name_f
#pragma weak pmpi_type_set_name = ompi_type_set_name_f
#pragma weak pmpi_type_set_name_ = ompi_type_set_name_f
#pragma weak pmpi_type_set_name__ = ompi_type_set_name_f

#pragma weak PMPI_Type_set_name_f = ompi_type_set_name_f
#pragma weak PMPI_Type_set_name_f08 = ompi_type_set_name_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_TYPE_SET_NAME,
                           pmpi_type_set_name,
                           pmpi_type_set_name_,
                           pmpi_type_set_name__,
                           pompi_type_set_name_f,
                           (MPI_Fint *type, char *type_name, MPI_Fint *ierr, int name_len),
                           (type, type_name, ierr, name_len) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_TYPE_SET_NAME = ompi_type_set_name_f
#pragma weak mpi_type_set_name = ompi_type_set_name_f
#pragma weak mpi_type_set_name_ = ompi_type_set_name_f
#pragma weak mpi_type_set_name__ = ompi_type_set_name_f

#pragma weak MPI_Type_set_name_f = ompi_type_set_name_f
#pragma weak MPI_Type_set_name_f08 = ompi_type_set_name_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_TYPE_SET_NAME,
                           mpi_type_set_name,
                           mpi_type_set_name_,
                           mpi_type_set_name__,
                           ompi_type_set_name_f,
                           (MPI_Fint *type, char *type_name, MPI_Fint *ierr, int name_len),
                           (type, type_name, ierr, name_len) )
#else
#define ompi_type_set_name_f pompi_type_set_name_f
#endif
#endif


void ompi_type_set_name_f(MPI_Fint *type, char *type_name, MPI_Fint *ierr,
			 int name_len)
{
    int ret, c_ierr;
    char *c_name;
    MPI_Datatype c_type;

    c_type = PMPI_Type_f2c(*type);

    /* Colwert the fortran string */

    if (OMPI_SUCCESS != (ret = ompi_fortran_string_f2c(type_name, name_len,
                                                       &c_name))) {
        c_ierr = OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD, ret,
                                        "MPI_TYPE_SET_NAME");
        if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
        return;
    }

    /* Call the C function */

    c_ierr = PMPI_Type_set_name(c_type, c_name);
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);

    /* Free the C name */

    free(c_name);
}
