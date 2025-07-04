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
#pragma weak PMPI_BUFFER_ATTACH = ompi_buffer_attach_f
#pragma weak pmpi_buffer_attach = ompi_buffer_attach_f
#pragma weak pmpi_buffer_attach_ = ompi_buffer_attach_f
#pragma weak pmpi_buffer_attach__ = ompi_buffer_attach_f

#pragma weak PMPI_Buffer_attach_f = ompi_buffer_attach_f
#pragma weak PMPI_Buffer_attach_f08 = ompi_buffer_attach_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_BUFFER_ATTACH,
                           pmpi_buffer_attach,
                           pmpi_buffer_attach_,
                           pmpi_buffer_attach__,
                           pompi_buffer_attach_f,
                           (char *buffer, MPI_Fint *size, MPI_Fint *ierr),
                           (buffer, size, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_BUFFER_ATTACH = ompi_buffer_attach_f
#pragma weak mpi_buffer_attach = ompi_buffer_attach_f
#pragma weak mpi_buffer_attach_ = ompi_buffer_attach_f
#pragma weak mpi_buffer_attach__ = ompi_buffer_attach_f

#pragma weak MPI_Buffer_attach_f = ompi_buffer_attach_f
#pragma weak MPI_Buffer_attach_f08 = ompi_buffer_attach_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_BUFFER_ATTACH,
                           mpi_buffer_attach,
                           mpi_buffer_attach_,
                           mpi_buffer_attach__,
                           ompi_buffer_attach_f,
                           (char *buffer, MPI_Fint *size, MPI_Fint *ierr),
                           (buffer, size, ierr) )
#else
#define ompi_buffer_attach_f pompi_buffer_attach_f
#endif
#endif


void ompi_buffer_attach_f(char *buffer, MPI_Fint *size, MPI_Fint *ierr)
{
   int c_ierr = PMPI_Buffer_attach(buffer, OMPI_FINT_2_INT(*size));
   if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
}
