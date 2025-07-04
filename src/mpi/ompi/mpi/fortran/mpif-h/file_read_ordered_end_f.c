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
 * Copyright (c) 2006-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012      Oracle and/or its affiliates.  All rights reserved.
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
#include "ompi/mpi/fortran/mpif-h/status-colwersion.h"
#include "ompi/mpi/fortran/base/constants.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_FILE_READ_ORDERED_END = ompi_file_read_ordered_end_f
#pragma weak pmpi_file_read_ordered_end = ompi_file_read_ordered_end_f
#pragma weak pmpi_file_read_ordered_end_ = ompi_file_read_ordered_end_f
#pragma weak pmpi_file_read_ordered_end__ = ompi_file_read_ordered_end_f

#pragma weak PMPI_File_read_ordered_end_f = ompi_file_read_ordered_end_f
#pragma weak PMPI_File_read_ordered_end_f08 = ompi_file_read_ordered_end_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_FILE_READ_ORDERED_END,
                           pmpi_file_read_ordered_end,
                           pmpi_file_read_ordered_end_,
                           pmpi_file_read_ordered_end__,
                           pompi_file_read_ordered_end_f,
                           (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr),
                           (fh, buf, status, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_FILE_READ_ORDERED_END = ompi_file_read_ordered_end_f
#pragma weak mpi_file_read_ordered_end = ompi_file_read_ordered_end_f
#pragma weak mpi_file_read_ordered_end_ = ompi_file_read_ordered_end_f
#pragma weak mpi_file_read_ordered_end__ = ompi_file_read_ordered_end_f

#pragma weak MPI_File_read_ordered_end_f = ompi_file_read_ordered_end_f
#pragma weak MPI_File_read_ordered_end_f08 = ompi_file_read_ordered_end_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_FILE_READ_ORDERED_END,
                           mpi_file_read_ordered_end,
                           mpi_file_read_ordered_end_,
                           mpi_file_read_ordered_end__,
                           ompi_file_read_ordered_end_f,
                           (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr),
                           (fh, buf, status, ierr) )
#else
#define ompi_file_read_ordered_end_f pompi_file_read_ordered_end_f
#endif
#endif


void ompi_file_read_ordered_end_f(MPI_Fint *fh, char *buf,
				 MPI_Fint *status, MPI_Fint *ierr)
{
    int c_ierr;
    MPI_File c_fh = PMPI_File_f2c(*fh);
    OMPI_FORTRAN_STATUS_DECLARATION(c_status,c_status2)

    OMPI_FORTRAN_STATUS_SET_POINTER(c_status,c_status2,status)

    c_ierr = PMPI_File_read_ordered_end(c_fh, buf, c_status);
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);

    OMPI_FORTRAN_STATUS_RETURN(c_status,c_status2,status,c_ierr)
}
