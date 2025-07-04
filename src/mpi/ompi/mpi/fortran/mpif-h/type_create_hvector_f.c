/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/fortran/mpif-h/bindings.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_TYPE_CREATE_HVECTOR = ompi_type_create_hvector_f
#pragma weak pmpi_type_create_hvector = ompi_type_create_hvector_f
#pragma weak pmpi_type_create_hvector_ = ompi_type_create_hvector_f
#pragma weak pmpi_type_create_hvector__ = ompi_type_create_hvector_f

#pragma weak PMPI_Type_create_hvector_f = ompi_type_create_hvector_f
#pragma weak PMPI_Type_create_hvector_f08 = ompi_type_create_hvector_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_TYPE_CREATE_HVECTOR,
                           pmpi_type_create_hvector,
                           pmpi_type_create_hvector_,
                           pmpi_type_create_hvector__,
                           pompi_type_create_hvector_f,
                           (MPI_Fint *count, MPI_Fint *blocklength, MPI_Aint *stride, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr),
                           (count, blocklength, stride, oldtype, newtype, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_TYPE_CREATE_HVECTOR = ompi_type_create_hvector_f
#pragma weak mpi_type_create_hvector = ompi_type_create_hvector_f
#pragma weak mpi_type_create_hvector_ = ompi_type_create_hvector_f
#pragma weak mpi_type_create_hvector__ = ompi_type_create_hvector_f

#pragma weak MPI_Type_create_hvector_f = ompi_type_create_hvector_f
#pragma weak MPI_Type_create_hvector_f08 = ompi_type_create_hvector_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_TYPE_CREATE_HVECTOR,
                           mpi_type_create_hvector,
                           mpi_type_create_hvector_,
                           mpi_type_create_hvector__,
                           ompi_type_create_hvector_f,
                           (MPI_Fint *count, MPI_Fint *blocklength, MPI_Aint *stride, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr),
                           (count, blocklength, stride, oldtype, newtype, ierr) )
#else
#define ompi_type_create_hvector_f pompi_type_create_hvector_f
#endif
#endif

void ompi_type_create_hvector_f(MPI_Fint *count, MPI_Fint *blocklength,
			       MPI_Aint *stride, MPI_Fint *oldtype,
			       MPI_Fint *newtype, MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Datatype c_old = PMPI_Type_f2c(*oldtype);
    MPI_Datatype c_new;

    c_ierr = PMPI_Type_create_hvector(OMPI_FINT_2_INT(*count),
                              OMPI_FINT_2_INT(*blocklength),
                              *stride,
                              c_old, &c_new);
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);

    if (MPI_SUCCESS == c_ierr) {
        *newtype = PMPI_Type_c2f(c_new);
    }
}
