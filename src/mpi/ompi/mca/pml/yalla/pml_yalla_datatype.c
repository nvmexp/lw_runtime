/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "pml_yalla_datatype.h"
#include "pml_yalla_request.h"


static mca_pml_yalla_colwertor_t *mca_pml_yalla_get_send_colwertor(void *buf, size_t count,
                                                                   ompi_datatype_t *datatype)
{
    mca_pml_yalla_colwertor_t *colwertor = (mca_pml_yalla_colwertor_t *)PML_YALLA_FREELIST_GET(&ompi_pml_yalla.colws);

    colwertor->datatype = datatype;
    OMPI_DATATYPE_RETAIN(datatype);
    opal_colwertor_copy_and_prepare_for_send(ompi_proc_local_proc->super.proc_colwertor,
                                             &datatype->super, count, buf, 0,
                                             &colwertor->colwertor);
    return colwertor;
}

static mca_pml_yalla_colwertor_t *mca_pml_yalla_get_recv_colwertor(void *buf, size_t count,
                                                                   ompi_datatype_t *datatype)
{
    mca_pml_yalla_colwertor_t *colwertor = (mca_pml_yalla_colwertor_t *)PML_YALLA_FREELIST_GET(&ompi_pml_yalla.colws);

    colwertor->datatype = datatype;
    OMPI_DATATYPE_RETAIN(datatype);
    opal_colwertor_copy_and_prepare_for_recv(ompi_proc_local_proc->super.proc_colwertor,
                                             &datatype->super, count, buf, 0,
                                             &colwertor->colwertor);
    return colwertor;
}

static void mca_pml_yalla_noncontig_req_init(mxm_req_base_t *mxm_req,
                                             mca_pml_yalla_colwertor_t *colwertor,
                                             mxm_stream_cb_t stream_cb)
{
    mxm_req->data_type      = MXM_REQ_DATA_STREAM;
    mxm_req->data.stream.cb = stream_cb;
    opal_colwertor_get_packed_size(&colwertor->colwertor, &mxm_req->data.stream.length);
}

static size_t mca_pml_yalla_stream_unpack(void *buffer, size_t length, size_t offset,
                                          opal_colwertor_t *colwertor)
{
    uint32_t iov_count;
    struct iovec iov;

    iov_count    = 1;
    iov.iov_base = buffer;
    iov.iov_len  = length;

    opal_colwertor_set_position(colwertor, &offset);
    opal_colwertor_unpack(colwertor, &iov, &iov_count, &length);
    return length;
}

static size_t mca_pml_yalla_stream_pack(void *buffer, size_t length, size_t offset,
                                        opal_colwertor_t *colwertor)
{
    uint32_t iov_count;
    struct iovec iov;

    iov_count    = 1;
    iov.iov_base = buffer;
    iov.iov_len  = length;

    opal_colwertor_set_position(colwertor, &offset);
    opal_colwertor_pack(colwertor, &iov, &iov_count, &length);
    return length;
}

static size_t mxm_pml_yalla_irecv_stream_cb(void *buffer, size_t length,
                                            size_t offset, void *context)
{
    mca_pml_yalla_base_request_t *req = context;
    return mca_pml_yalla_stream_unpack(buffer, length, offset, &req->colwertor->colwertor);
}

static size_t mxm_pml_yalla_recv_stream_cb(void *buffer, size_t length,
                                           size_t offset, void *context)
{
    mca_pml_yalla_colwertor_t *colwertor = context;
    return mca_pml_yalla_stream_unpack(buffer, length, offset, &colwertor->colwertor);
}

static size_t mxm_pml_yalla_isend_stream_cb(void *buffer, size_t length,
                                            size_t offset, void *context)
{
    mca_pml_yalla_base_request_t *req = context;
    return mca_pml_yalla_stream_pack(buffer, length, offset, &req->colwertor->colwertor);
}

static size_t mxm_pml_yalla_send_stream_cb(void *buffer, size_t length,
                                           size_t offset, void *context)
{
    mca_pml_yalla_colwertor_t *colwertor = context;
    return mca_pml_yalla_stream_pack(buffer, length, offset, &colwertor->colwertor);
}

void mca_pml_yalla_set_noncontig_data_irecv(mxm_req_base_t *mxm_req, void *buf,
                                            size_t count, ompi_datatype_t *datatype,
                                            mca_pml_yalla_recv_request_t *rreq)
{
    rreq->super.colwertor = mca_pml_yalla_get_recv_colwertor(buf, count, datatype);
    mca_pml_yalla_noncontig_req_init(mxm_req, rreq->super.colwertor, mxm_pml_yalla_irecv_stream_cb);
}

void mca_pml_yalla_set_noncontig_data_recv(mxm_req_base_t *mxm_req, void *buf,
                                           size_t count, ompi_datatype_t *datatype)
{
    mca_pml_yalla_colwertor_t *colwertor;

    colwertor = mca_pml_yalla_get_recv_colwertor(buf, count, datatype);
    mca_pml_yalla_noncontig_req_init(mxm_req, colwertor, mxm_pml_yalla_recv_stream_cb);
    mxm_req->context = colwertor;
}

void mca_pml_yalla_set_noncontig_data_isend(mxm_req_base_t *mxm_req, void *buf,
                                            size_t count, ompi_datatype_t *datatype,
                                            mca_pml_yalla_send_request_t *sreq)
{
    sreq->super.colwertor = mca_pml_yalla_get_send_colwertor(buf, count, datatype);
    mca_pml_yalla_noncontig_req_init(mxm_req, sreq->super.colwertor, mxm_pml_yalla_isend_stream_cb);
}

void mca_pml_yalla_set_noncontig_data_send(mxm_req_base_t *mxm_req, void *buf,
                                           size_t count, ompi_datatype_t *datatype)
{
    mca_pml_yalla_colwertor_t *colwertor;

    colwertor = mca_pml_yalla_get_send_colwertor(buf, count, datatype);
    mca_pml_yalla_noncontig_req_init(mxm_req, colwertor, mxm_pml_yalla_send_stream_cb);
    mxm_req->context = colwertor;
}

static void mca_pml_yalla_colwertor_construct(mca_pml_yalla_colwertor_t *colwertor)
{
    OBJ_CONSTRUCT(&colwertor->colwertor, opal_colwertor_t);
}

static void mca_pml_yalla_colwertor_destruct(mca_pml_yalla_colwertor_t *colwertor)
{
    OBJ_DESTRUCT(&colwertor->colwertor);
}

void mca_pml_yalla_init_datatype(void)
{
    PML_YALLA_FREELIST_INIT(&ompi_pml_yalla.colws, mca_pml_yalla_colwertor_t,
                         128, -1, 128);
}

OBJ_CLASS_INSTANCE(mca_pml_yalla_colwertor_t,
                   opal_free_list_item_t,
                   mca_pml_yalla_colwertor_construct,
                   mca_pml_yalla_colwertor_destruct);

