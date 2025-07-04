/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2007 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2014-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_BASE_UTIL_EXPORT_H
#define MCA_COLL_BASE_UTIL_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/request/request.h"
#include "ompi/op/op.h"
#include "ompi/mca/pml/pml.h"

BEGIN_C_DECLS

struct ompi_coll_base_nbc_request_t {
    ompi_request_t super;
    union {
        ompi_request_complete_fn_t req_complete_cb;
        ompi_request_free_fn_t req_free;
    } cb;
    void *req_complete_cb_data;
    union {
        struct {
            ompi_op_t *op;
            ompi_datatype_t *datatype;
        } op;
        struct {
            ompi_datatype_t *stype;
            ompi_datatype_t *rtype;
        } types;
        struct {
            opal_object_t *objs[2];
        } objs;
        struct {
            ompi_datatype_t **stypes;
            ompi_datatype_t **rtypes;
        } vecs;
    } data;
};

OMPI_DECLSPEC OBJ_CLASS_DECLARATION(ompi_coll_base_nbc_request_t);

typedef struct ompi_coll_base_nbc_request_t ompi_coll_base_nbc_request_t;

/**
 * A MPI_like function doing a send and a receive simultaneously.
 * If one of the communications results in a zero-byte message the
 * communication is ignored, and no message will cross to the peer.
 */
int ompi_coll_base_sendrecv_actual( const void* sendbuf, size_t scount,
                                    ompi_datatype_t* sdatatype,
                                    int dest, int stag,
                                    void* recvbuf, size_t rcount,
                                    ompi_datatype_t* rdatatype,
                                    int source, int rtag,
                                    struct ompi_communicator_t* comm,
                                    ompi_status_public_t* status );


/**
 * Similar to the function above this implementation of send-receive
 * do not generate communications for zero-bytes messages. Thus, it is
 * improper to use in the context of some algorithms for collective
 * communications.
 */
static inline int
ompi_coll_base_sendrecv( void* sendbuf, size_t scount, ompi_datatype_t* sdatatype,
                          int dest, int stag,
                          void* recvbuf, size_t rcount, ompi_datatype_t* rdatatype,
                          int source, int rtag,
                          struct ompi_communicator_t* comm,
                          ompi_status_public_t* status, int myid )
{
    if ((dest == source) && (source == myid)) {
        return (int) ompi_datatype_sndrcv(sendbuf, (int32_t) scount, sdatatype,
                                          recvbuf, (int32_t) rcount, rdatatype);
    }
    return ompi_coll_base_sendrecv_actual (sendbuf, scount, sdatatype,
                                           dest, stag,
                                           recvbuf, rcount, rdatatype,
                                           source, rtag, comm, status);
}

/**
 * ompi_mirror_perm: Returns mirror permutation of nbits low-order bits
 *                   of x [*].
 * [*] Warren Jr., Henry S. Hacker's Delight (2ed). 2013.
 *     Chapter 7. Rearranging Bits and Bytes.
 */
unsigned int ompi_mirror_perm(unsigned int x, int nbits);

/*
 * ompi_rounddown: Rounds a number down to nearest multiple.
 *     rounddown(10,4) = 8, rounddown(6,3) = 6, rounddown(14,3) = 12
 */
int ompi_rounddown(int num, int factor);

int ompi_coll_base_retain_op( ompi_request_t *request,
                              ompi_op_t *op,
                              ompi_datatype_t *type);

int ompi_coll_base_retain_datatypes( ompi_request_t *request,
                                      ompi_datatype_t *stype,
                                     ompi_datatype_t *rtype);

int ompi_coll_base_retain_datatypes_w( ompi_request_t *request,
                                       ompi_datatype_t *stypes[],
                                       ompi_datatype_t *rtypes[]);

END_C_DECLS
#endif /* MCA_COLL_BASE_UTIL_EXPORT_H */
