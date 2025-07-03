/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
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
 * Copyright (c) 2014-2016 Los Alamos National Security, LLC. All rights
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

#include "ompi/mpi/fortran/mpif-h/bindings.h"
#include "ompi/mpi/fortran/base/constants.h"


#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_RACLWMULATE = ompi_raclwmulate_f
#pragma weak pmpi_raclwmulate = ompi_raclwmulate_f
#pragma weak pmpi_raclwmulate_ = ompi_raclwmulate_f
#pragma weak pmpi_raclwmulate__ = ompi_raclwmulate_f

#pragma weak PMPI_Raclwmulate_f = ompi_raclwmulate_f
#pragma weak PMPI_Raclwmulate_f08 = ompi_raclwmulate_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_RACLWMULATE,
                            pmpi_raclwmulate,
                            pmpi_raclwmulate_,
                            pmpi_raclwmulate__,
                            pompi_raclwmulate_f,
                            (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *request, MPI_Fint *ierr),
                            (origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_RACLWMULATE = ompi_raclwmulate_f
#pragma weak mpi_raclwmulate = ompi_raclwmulate_f
#pragma weak mpi_raclwmulate_ = ompi_raclwmulate_f
#pragma weak mpi_raclwmulate__ = ompi_raclwmulate_f

#pragma weak MPI_Raclwmulate_f = ompi_raclwmulate_f
#pragma weak MPI_Raclwmulate_f08 = ompi_raclwmulate_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_RACLWMULATE,
                            mpi_raclwmulate,
                            mpi_raclwmulate_,
                            mpi_raclwmulate__,
                            ompi_raclwmulate_f,
                            (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *request, MPI_Fint *ierr),
                            (origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request, ierr) )
#else
#define ompi_raclwmulate_f pompi_raclwmulate_f
#endif
#endif


void ompi_raclwmulate_f(char *origin_addr, MPI_Fint *origin_count,
                        MPI_Fint *origin_datatype, MPI_Fint *target_rank,
                        MPI_Aint *target_disp, MPI_Fint *target_count,
                        MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win,
                        MPI_Fint *request, MPI_Fint *ierr)
{
    int ierr_c;

    MPI_Datatype c_origin_datatype = PMPI_Type_f2c(*origin_datatype);
    MPI_Datatype c_target_datatype = PMPI_Type_f2c(*target_datatype);
    MPI_Win c_win = PMPI_Win_f2c(*win);
    MPI_Op c_op = PMPI_Op_f2c(*op);
    MPI_Request c_req;

    ierr_c = PMPI_Raclwmulate(OMPI_F2C_BOTTOM(origin_addr),
                              OMPI_FINT_2_INT(*origin_count),
                              c_origin_datatype,
                              OMPI_FINT_2_INT(*target_rank),
                              *target_disp,
                              OMPI_FINT_2_INT(*target_count),
                              c_target_datatype, c_op, c_win,
                              &c_req);

    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(ierr_c);

    if (MPI_SUCCESS == ierr_c) {
        *request = PMPI_Request_c2f(c_req);
    }
}
