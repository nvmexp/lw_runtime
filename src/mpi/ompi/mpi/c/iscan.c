/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
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
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/op/op.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/memchecker.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Iscan = PMPI_Iscan
#endif
#define MPI_Iscan PMPI_Iscan
#endif

static const char FUNC_NAME[] = "MPI_Iscan";


int MPI_Iscan(const void *sendbuf, void *recvbuf, int count,
              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)
{
    int err;

    SPC_RECORD(OMPI_SPC_ISCAN, 1);

    MEMCHECKER(
        memchecker_datatype(datatype);
        memchecker_comm(comm);
        if (MPI_IN_PLACE != sendbuf) {
            memchecker_call(&opal_memchecker_base_isdefined, sendbuf, count, datatype);
        } else {
            memchecker_call(&opal_memchecker_base_isdefined, recvbuf, count, datatype);
        }
    );

    if (MPI_PARAM_CHECK) {
        char *msg;
        err = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_ilwalid(comm)) {
            return OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD, MPI_ERR_COMM,
                                          FUNC_NAME);
        }

        /* No intercommunicators allowed! (MPI does not define
           MPI_SCAN on intercommunicators) */

        else if (OMPI_COMM_IS_INTER(comm)) {
          err = MPI_ERR_COMM;
        }

        /* Unrooted operation; checks for all ranks */

        else if (MPI_OP_NULL == op || NULL == op) {
          err = MPI_ERR_OP;
        } else if (MPI_IN_PLACE == recvbuf) {
          err = MPI_ERR_ARG;
        } else if (!ompi_op_is_valid(op, datatype, &msg, FUNC_NAME)) {
            int ret = OMPI_ERRHANDLER_ILWOKE(comm, MPI_ERR_OP, msg);
            free(msg);
            return ret;
        } else {
          OMPI_CHECK_DATATYPE_FOR_SEND(err, datatype, count);
        }
        OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
    }

    OPAL_CR_ENTER_LIBRARY();

    /* Call the coll component to actually perform the allgather */

    err = comm->c_coll->coll_iscan(sendbuf, recvbuf, count,
                                  datatype, op, comm,
                                  request,
                                  comm->c_coll->coll_iscan_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ompi_coll_base_retain_op(*request, op, datatype);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
