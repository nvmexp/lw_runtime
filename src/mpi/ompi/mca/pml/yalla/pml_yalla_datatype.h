/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef PML_YALLA_DATATYPE_H_
#define PML_YALLA_DATATYPE_H_

#include "pml_yalla.h"

struct pml_yalla_colwertor {
    opal_free_list_item_t     super;
    ompi_datatype_t           *datatype;
    opal_colwertor_t          colwertor;
};

OBJ_CLASS_DECLARATION(mca_pml_yalla_colwertor_t);

#define PML_YALLA_INIT_MXM_REQ_DATA(_req_base, _buf, _count, _dtype, _stream_type, ...) \
    { \
        ptrdiff_t span, gap; \
        \
        if (opal_datatype_is_contiguous_memory_layout(&(_dtype)->super, _count)) { \
            span = opal_datatype_span(&(_dtype)->super, (_count), &gap); \
            (_req_base)->data_type          = MXM_REQ_DATA_BUFFER; \
            (_req_base)->data.buffer.ptr    = (char *)_buf + gap; \
            (_req_base)->data.buffer.length = span; \
        } else { \
            mca_pml_yalla_set_noncontig_data_ ## _stream_type(_req_base, \
                                                              _buf, _count, \
                                                              _dtype, ## __VA_ARGS__); \
        } \
    }

#define PML_YALLA_RESET_PML_REQ_DATA(_pml_req) \
    { \
        if ((_pml_req)->colwertor != NULL) { \
            size_t _position = 0; \
            opal_colwertor_set_position(&(_pml_req)->colwertor->colwertor, &_position); \
        } \
    }


static inline void mca_pml_yalla_colwertor_free(mca_pml_yalla_colwertor_t *colwertor)
{
    opal_colwertor_cleanup(&colwertor->colwertor);
    OMPI_DATATYPE_RELEASE(colwertor->datatype);
    PML_YALLA_FREELIST_RETURN(&ompi_pml_yalla.colws, &colwertor->super);
}

void mca_pml_yalla_set_noncontig_data_irecv(mxm_req_base_t *mxm_req, void *buf,
                                            size_t count, ompi_datatype_t *datatype,
                                            mca_pml_yalla_recv_request_t *rreq);

void mca_pml_yalla_set_noncontig_data_recv(mxm_req_base_t *mxm_req, void *buf,
                                           size_t count, ompi_datatype_t *datatype);

void mca_pml_yalla_set_noncontig_data_isend(mxm_req_base_t *mxm_req, void *buf,
                                            size_t count, ompi_datatype_t *datatype,
                                            mca_pml_yalla_send_request_t *sreq);

void mca_pml_yalla_set_noncontig_data_send(mxm_req_base_t *mxm_req, void *buf,
                                           size_t count, ompi_datatype_t *datatype);

void mca_pml_yalla_init_datatype(void);


#endif /* PML_YALLA_DATATYPE_H_ */
