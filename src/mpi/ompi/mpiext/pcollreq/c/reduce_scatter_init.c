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
 * Copyright (c) 2006-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012-2013 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2016      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
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
#include "ompi/mpiext/pcollreq/c/mpiext_pcollreq_c.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_Reduce_scatter_init = PMPIX_Reduce_scatter_init
#endif
#define MPIX_Reduce_scatter_init PMPIX_Reduce_scatter_init
#endif

static const char FUNC_NAME[] = "MPIX_Reduce_scatter_init";


int MPIX_Reduce_scatter_init(const void *sendbuf, void *recvbuf, const int recvcounts[],
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Info info, MPI_Request *request)
{
    int i, err, size, count;

    SPC_RECORD(OMPI_SPC_REDUCE_SCATTER_INIT, 1);

    MEMCHECKER(
        int rank;
        int count;

        size = ompi_comm_size(comm);
        rank = ompi_comm_rank(comm);
        for (count = i = 0; i < size; ++i) {
            if (0 == recvcounts[i]) {
                count += recvcounts[i];
            }
        }

        memchecker_comm(comm);
        memchecker_datatype(datatype);

        /* check receive buffer of current proccess, whether it's addressable. */
        memchecker_call(&opal_memchecker_base_isaddressable, recvbuf,
                        recvcounts[rank], datatype);

        /* check whether the actual send buffer is defined. */
        if(MPI_IN_PLACE == sendbuf) {
            memchecker_call(&opal_memchecker_base_isdefined, recvbuf, count, datatype);
        } else {
            memchecker_call(&opal_memchecker_base_isdefined, sendbuf, count, datatype);

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

        /* Unrooted operation; same checks for all ranks on both
           intracommunicators and intercommunicators */

        else if (MPI_OP_NULL == op || NULL == op) {
          err = MPI_ERR_OP;
        } else if (!ompi_op_is_valid(op, datatype, &msg, FUNC_NAME)) {
            int ret = OMPI_ERRHANDLER_ILWOKE(comm, MPI_ERR_OP, msg);
            free(msg);
            return ret;
        } else if (NULL == recvcounts) {
          err = MPI_ERR_COUNT;
        } else if (MPI_IN_PLACE == recvbuf) {
          err = MPI_ERR_ARG;
        }
        OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);

        /* Based on the standard each group has to provide the same total
           number of elements, so the size of the recvcounts array depends
           on the number of participants in the local group.  */
        size = ompi_comm_size(comm);
        for (i = 0; i < size; ++i) {
          OMPI_CHECK_DATATYPE_FOR_SEND(err, datatype, recvcounts[i]);
          OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
        }
    }

    /* MPI standard says that reductions have to have a count of at least 1,
     * but some benchmarks (e.g., IMB) calls this function with a count of 0.
     * So handle that case.
     */
    size = ompi_comm_size(comm);
    for (count = i = 0; i < size; ++i) {
        if (0 == recvcounts[i]) {
            ++count;
        }
    }
    if (size == count) {
        err = ompi_request_persistent_noop_create(request);
        OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
    }

    OPAL_CR_ENTER_LIBRARY();

    /* Ilwoke the coll component to perform the back-end operation */

    err = comm->c_coll->coll_reduce_scatter_init(sendbuf, recvbuf, recvcounts,
                                                 datatype, op, comm, info, request,
                                                 comm->c_coll->coll_reduce_scatter_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ompi_coll_base_retain_op(*request, op, datatype);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
