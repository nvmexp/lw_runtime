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
#include "ompi/mpi/fortran/base/constants.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_IREDUCE_SCATTER = ompi_ireduce_scatter_f
#pragma weak pmpi_ireduce_scatter = ompi_ireduce_scatter_f
#pragma weak pmpi_ireduce_scatter_ = ompi_ireduce_scatter_f
#pragma weak pmpi_ireduce_scatter__ = ompi_ireduce_scatter_f

#pragma weak PMPI_Ireduce_scatter_f = ompi_ireduce_scatter_f
#pragma weak PMPI_Ireduce_scatter_f08 = ompi_ireduce_scatter_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_IREDUCE_SCATTER,
                            pmpi_ireduce_scatter,
                            pmpi_ireduce_scatter_,
                            pmpi_ireduce_scatter__,
                            pompi_ireduce_scatter_f,
                            (char *sendbuf, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_IREDUCE_SCATTER = ompi_ireduce_scatter_f
#pragma weak mpi_ireduce_scatter = ompi_ireduce_scatter_f
#pragma weak mpi_ireduce_scatter_ = ompi_ireduce_scatter_f
#pragma weak mpi_ireduce_scatter__ = ompi_ireduce_scatter_f

#pragma weak MPI_Ireduce_scatter_f = ompi_ireduce_scatter_f
#pragma weak MPI_Ireduce_scatter_f08 = ompi_ireduce_scatter_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_IREDUCE_SCATTER,
                            mpi_ireduce_scatter,
                            mpi_ireduce_scatter_,
                            mpi_ireduce_scatter__,
                            ompi_ireduce_scatter_f,
                            (char *sendbuf, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, recvbuf, recvcounts, datatype, op, comm, request, ierr) )
#else
#define ompi_ireduce_scatter_f pompi_ireduce_scatter_f
#endif
#endif


void ompi_ireduce_scatter_f(char *sendbuf, char *recvbuf,
                            MPI_Fint *recvcounts, MPI_Fint *datatype,
                            MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request,
                            MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Comm c_comm;
    MPI_Datatype c_type;
    MPI_Request c_request;
    MPI_Op c_op;
    int size;
    OMPI_ARRAY_NAME_DECL(recvcounts);

    c_comm = PMPI_Comm_f2c(*comm);
    c_type = PMPI_Type_f2c(*datatype);
    c_op = PMPI_Op_f2c(*op);

    PMPI_Comm_size(c_comm, &size);
    OMPI_ARRAY_FINT_2_INT(recvcounts, size);

    sendbuf = (char *) OMPI_F2C_IN_PLACE(sendbuf);
    sendbuf = (char *) OMPI_F2C_BOTTOM(sendbuf);
    recvbuf = (char *) OMPI_F2C_BOTTOM(recvbuf);

    c_ierr = PMPI_Ireduce_scatter(sendbuf, recvbuf,
                                 OMPI_ARRAY_NAME_COLWERT(recvcounts),
                                 c_type, c_op, c_comm, &c_request);
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
    if (MPI_SUCCESS == c_ierr) *request = PMPI_Request_c2f(c_request);
}
