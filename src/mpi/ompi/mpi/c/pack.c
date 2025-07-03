/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
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
#include "opal/datatype/opal_colwertor.h"
#include "ompi/memchecker.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Pack = PMPI_Pack
#endif
#define MPI_Pack PMPI_Pack
#endif

static const char FUNC_NAME[] = "MPI_Pack";


int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype,
             void *outbuf, int outsize, int *position, MPI_Comm comm)
{
    int rc = MPI_SUCCESS, ret;
    opal_colwertor_t local_colwertor;
    struct iovec ilwec;
    unsigned int iov_count;
    size_t size;

    MEMCHECKER(
        memchecker_datatype(datatype);
        memchecker_call(&opal_memchecker_base_isdefined, inbuf, incount, datatype);
        memchecker_comm(comm);
    );

    if (MPI_PARAM_CHECK) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_ilwalid(comm)) {
            return OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD, MPI_ERR_COMM, FUNC_NAME);
        } else if ((NULL == outbuf) || (NULL == position)) {  /* inbuf can be MPI_BOTTOM */
            return OMPI_ERRHANDLER_ILWOKE(comm, MPI_ERR_ARG, FUNC_NAME);
        } else if (incount < 0) {
            return OMPI_ERRHANDLER_ILWOKE(comm, MPI_ERR_COUNT, FUNC_NAME);
        } else if (outsize < 0) {
            return OMPI_ERRHANDLER_ILWOKE(comm, MPI_ERR_ARG, FUNC_NAME);
        }
        OMPI_CHECK_DATATYPE_FOR_SEND(rc, datatype, incount);
        OMPI_ERRHANDLER_CHECK(rc, comm, rc, FUNC_NAME);
        OMPI_CHECK_USER_BUFFER(rc, inbuf, datatype, incount);
        OMPI_ERRHANDLER_CHECK(rc, comm, rc, FUNC_NAME);
    }

    OPAL_CR_ENTER_LIBRARY();

    OBJ_CONSTRUCT( &local_colwertor, opal_colwertor_t );
    /* the resulting colwertor will be set to the position ZERO */
    opal_colwertor_copy_and_prepare_for_send( ompi_mpi_local_colwertor, &(datatype->super),
                                              incount, (void *) inbuf, 0, &local_colwertor );

    /* Check for truncation */
    opal_colwertor_get_packed_size( &local_colwertor, &size );
    if( (*position + size) > (unsigned int)outsize ) {  /* we can cast as we already checked for < 0 */
        OBJ_DESTRUCT( &local_colwertor );
        OPAL_CR_EXIT_LIBRARY();
        return OMPI_ERRHANDLER_ILWOKE(comm, MPI_ERR_TRUNCATE, FUNC_NAME);
    }

    /* Prepare the iovec with all informations */
    ilwec.iov_base = (char*) outbuf + (*position);
    ilwec.iov_len = size;

    /* Do the actual packing */
    iov_count = 1;
    ret = opal_colwertor_pack( &local_colwertor, &ilwec, &iov_count, &size );
    *position += size;
    OBJ_DESTRUCT( &local_colwertor );

    OPAL_CR_EXIT_LIBRARY();

    /* All done.  Note that the colwertor returns 1 upon success, not
       OPAL_SUCCESS. */
    if (1 != ret) {
        rc = OMPI_ERROR;
    }
    OMPI_ERRHANDLER_RETURN(rc, comm, MPI_ERR_UNKNOWN, FUNC_NAME);
}
