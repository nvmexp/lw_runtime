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
 * Copyright (c) 2015-2018 Research Organization for Information Science
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
#include "ompi/mpiext/pcollreq/mpif-h/mpiext_pcollreq_prototypes.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPIX_REDUCE_INIT = ompix_reduce_init_f
#pragma weak pmpix_reduce_init = ompix_reduce_init_f
#pragma weak pmpix_reduce_init_ = ompix_reduce_init_f
#pragma weak pmpix_reduce_init__ = ompix_reduce_init_f

#pragma weak PMPIX_Reduce_init_f = ompix_reduce_init_f
#pragma weak PMPIX_Reduce_init_f08 = ompix_reduce_init_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPIX_REDUCE_INIT,
                            pmpix_reduce_init,
                            pmpix_reduce_init_,
                            pmpix_reduce_init__,
                            pompix_reduce_init_f,
                            (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_REDUCE_INIT = ompix_reduce_init_f
#pragma weak mpix_reduce_init = ompix_reduce_init_f
#pragma weak mpix_reduce_init_ = ompix_reduce_init_f
#pragma weak mpix_reduce_init__ = ompix_reduce_init_f

#pragma weak MPIX_Reduce_init_f = ompix_reduce_init_f
#pragma weak MPIX_Reduce_init_f08 = ompix_reduce_init_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPIX_REDUCE_INIT,
                            mpix_reduce_init,
                            mpix_reduce_init_,
                            mpix_reduce_init__,
                            ompix_reduce_init_f,
                            (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, recvbuf, count, datatype, op, root, comm, info, request, ierr) )
#else
#define ompix_reduce_init_f pompix_reduce_init_f
#endif
#endif


void ompix_reduce_init_f(char *sendbuf, char *recvbuf, MPI_Fint *count,
                         MPI_Fint *datatype, MPI_Fint *op,
                         MPI_Fint *root, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request,
                         MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Datatype c_type;
    MPI_Info c_info;
    MPI_Request c_request;
    MPI_Op c_op;
    MPI_Comm c_comm;

    c_type = PMPI_Type_f2c(*datatype);
    c_op = PMPI_Op_f2c(*op);
    c_comm = PMPI_Comm_f2c(*comm);
    c_info = PMPI_Info_f2c(*info);

    sendbuf = (char *) OMPI_F2C_IN_PLACE(sendbuf);
    sendbuf = (char *) OMPI_F2C_BOTTOM(sendbuf);
    recvbuf = (char *) OMPI_F2C_BOTTOM(recvbuf);

    c_ierr = PMPIX_Reduce_init(sendbuf, recvbuf,
                         OMPI_FINT_2_INT(*count),
                         c_type, c_op,
                         OMPI_FINT_2_INT(*root),
                         c_comm, c_info, &c_request);
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
    if (MPI_SUCCESS == c_ierr) *request = PMPI_Request_c2f(c_request);
}
