/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      QLogic Corporation. All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/communicator/communicator.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/util/show_help.h"

#include "mtl_psm.h"
#include "mtl_psm_types.h"
#include "mtl_psm_request.h"
#include "ompi/mca/mtl/base/mtl_base_datatype.h"

int
ompi_mtl_psm_send(struct mca_mtl_base_module_t* mtl,
                 struct ompi_communicator_t* comm,
                 int dest,
                 int tag,
                 struct opal_colwertor_t *colwertor,
                 mca_pml_base_send_mode_t mode)
{
    psm_error_t err;
    mca_mtl_psm_request_t mtl_psm_request;
    uint64_t mqtag;
    uint32_t flags = 0;
    int ret;
    size_t length;
    ompi_proc_t* ompi_proc = ompi_comm_peer_lookup( comm, dest );
    mca_mtl_psm_endpoint_t* psm_endpoint = ompi_mtl_psm_get_endpoint (mtl, ompi_proc);

    assert(mtl == &ompi_mtl_psm.super);

    mqtag = PSM_MAKE_MQTAG(comm->c_contextid, comm->c_my_rank, tag);

    ret = ompi_mtl_datatype_pack(colwertor,
                                 &mtl_psm_request.buf,
                                 &length,
                                 &mtl_psm_request.free_after);

    if (OMPI_SUCCESS != ret) return ret;

    if (length >= 1ULL << sizeof(uint32_t) * 8) {
            opal_show_help("help-mtl-psm.txt",
                   "message too big", false,
                   length, 1ULL << sizeof(uint32_t) * 8);
            return OMPI_ERROR;
    }

    mtl_psm_request.length = length;
    mtl_psm_request.colwertor = colwertor;
    mtl_psm_request.type = OMPI_MTL_PSM_ISEND;

    if (mode == MCA_PML_BASE_SEND_SYNCHRONOUS)
	flags |= PSM_MQ_FLAG_SENDSYNC;

    err = psm_mq_send(ompi_mtl_psm.mq,
		      psm_endpoint->peer_addr,
		      flags,
		      mqtag,
		      mtl_psm_request.buf,
		      length);

    if (mtl_psm_request.free_after) {
	free(mtl_psm_request.buf);
    }

    return err == PSM_OK ? OMPI_SUCCESS : OMPI_ERROR;
}

int
ompi_mtl_psm_isend(struct mca_mtl_base_module_t* mtl,
                  struct ompi_communicator_t* comm,
                  int dest,
                  int tag,
                  struct opal_colwertor_t *colwertor,
                  mca_pml_base_send_mode_t mode,
                  bool blocking,
                  mca_mtl_request_t * mtl_request)
{
    psm_error_t psm_error;
    uint64_t mqtag;
    uint32_t flags = 0;
    int ret;
    mca_mtl_psm_request_t * mtl_psm_request = (mca_mtl_psm_request_t*) mtl_request;
    size_t length;
    ompi_proc_t* ompi_proc = ompi_comm_peer_lookup( comm, dest );
    mca_mtl_psm_endpoint_t* psm_endpoint = ompi_mtl_psm_get_endpoint (mtl, ompi_proc);

    assert(mtl == &ompi_mtl_psm.super);

    mqtag = PSM_MAKE_MQTAG(comm->c_contextid, comm->c_my_rank, tag);


    ret = ompi_mtl_datatype_pack(colwertor,
                                 &mtl_psm_request->buf,
                                 &length,
                                 &mtl_psm_request->free_after);


    if (OMPI_SUCCESS != ret) return ret;

    if (length >= 1ULL << sizeof(uint32_t) * 8) {
            opal_show_help("help-mtl-psm.txt",
                   "message too big", false,
                   length, 1ULL << sizeof(uint32_t) * 8);
            return OMPI_ERROR;
    }

    mtl_psm_request->length= length;
    mtl_psm_request->colwertor = colwertor;
    mtl_psm_request->type = OMPI_MTL_PSM_ISEND;

    if (mode == MCA_PML_BASE_SEND_SYNCHRONOUS)
	flags |= PSM_MQ_FLAG_SENDSYNC;

    psm_error = psm_mq_isend(ompi_mtl_psm.mq,
			     psm_endpoint->peer_addr,
			     flags,
			     mqtag,
			     mtl_psm_request->buf,
			     length,
			     mtl_psm_request,
			     &mtl_psm_request->psm_request);

    return psm_error == PSM_OK ? OMPI_SUCCESS : OMPI_ERROR;
}
