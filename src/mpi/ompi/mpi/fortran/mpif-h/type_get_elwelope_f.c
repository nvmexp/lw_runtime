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
 * Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
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
#pragma weak PMPI_TYPE_GET_ELWELOPE = ompi_type_get_elwelope_f
#pragma weak pmpi_type_get_elwelope = ompi_type_get_elwelope_f
#pragma weak pmpi_type_get_elwelope_ = ompi_type_get_elwelope_f
#pragma weak pmpi_type_get_elwelope__ = ompi_type_get_elwelope_f

#pragma weak PMPI_Type_get_elwelope_f = ompi_type_get_elwelope_f
#pragma weak PMPI_Type_get_elwelope_f08 = ompi_type_get_elwelope_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_TYPE_GET_ELWELOPE,
                           pmpi_type_get_elwelope,
                           pmpi_type_get_elwelope_,
                           pmpi_type_get_elwelope__,
                           pompi_type_get_elwelope_f,
                           (MPI_Fint *type, MPI_Fint *num_integers, MPI_Fint *num_addresses, MPI_Fint *num_datatypes, MPI_Fint *combiner, MPI_Fint *ierr),
                           (type, num_integers, num_addresses, num_datatypes, combiner, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_TYPE_GET_ELWELOPE = ompi_type_get_elwelope_f
#pragma weak mpi_type_get_elwelope = ompi_type_get_elwelope_f
#pragma weak mpi_type_get_elwelope_ = ompi_type_get_elwelope_f
#pragma weak mpi_type_get_elwelope__ = ompi_type_get_elwelope_f

#pragma weak MPI_Type_get_elwelope_f = ompi_type_get_elwelope_f
#pragma weak MPI_Type_get_elwelope_f08 = ompi_type_get_elwelope_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_TYPE_GET_ELWELOPE,
                           mpi_type_get_elwelope,
                           mpi_type_get_elwelope_,
                           mpi_type_get_elwelope__,
                           ompi_type_get_elwelope_f,
                           (MPI_Fint *type, MPI_Fint *num_integers, MPI_Fint *num_addresses, MPI_Fint *num_datatypes, MPI_Fint *combiner, MPI_Fint *ierr),
                           (type, num_integers, num_addresses, num_datatypes, combiner, ierr) )
#else
#define ompi_type_get_elwelope_f pompi_type_get_elwelope_f
#endif
#endif


void ompi_type_get_elwelope_f(MPI_Fint *type, MPI_Fint *num_integers,
			     MPI_Fint *num_addresses,
			     MPI_Fint *num_datatypes, MPI_Fint *combiner,
			     MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Datatype c_type = PMPI_Type_f2c(*type);
    OMPI_SINGLE_NAME_DECL(num_integers);
    OMPI_SINGLE_NAME_DECL(num_addresses);
    OMPI_SINGLE_NAME_DECL(num_datatypes);
    OMPI_SINGLE_NAME_DECL(combiner);

    c_ierr = PMPI_Type_get_elwelope(c_type,
                                   OMPI_SINGLE_NAME_COLWERT(num_integers),
                                   OMPI_SINGLE_NAME_COLWERT(num_addresses),
                                   OMPI_SINGLE_NAME_COLWERT(num_datatypes),
                                   OMPI_SINGLE_NAME_COLWERT(combiner));
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);

    if (MPI_SUCCESS == c_ierr) {
        OMPI_SINGLE_INT_2_FINT(num_integers);
        OMPI_SINGLE_INT_2_FINT(num_addresses);
        OMPI_SINGLE_INT_2_FINT(num_datatypes);
        OMPI_SINGLE_INT_2_FINT(combiner);
    }
}
