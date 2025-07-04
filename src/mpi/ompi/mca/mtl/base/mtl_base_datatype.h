/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mca/mca.h"
#include "ompi/mca/mtl/mtl.h"
#include "ompi/mca/mtl/base/base.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/datatype/opal_datatype_internal.h"

#ifndef MTL_BASE_DATATYPE_H_INCLUDED
#define MTL_BASE_DATATYPE_H_INCLUDED

__opal_attribute_always_inline__ static inline int
ompi_mtl_datatype_pack(struct opal_colwertor_t *colwertor,
                       void **buffer,
                       size_t *buffer_len,
                       bool *freeAfter)
{
    struct iovec iov;
    uint32_t iov_count = 1;

#if !(OPAL_ENABLE_HETEROGENEOUS_SUPPORT)
    if (colwertor->pDesc &&
	!(colwertor->flags & COLWERTOR_COMPLETED) &&
	opal_datatype_is_contiguous_memory_layout(colwertor->pDesc,
						  colwertor->count)) {
	    *freeAfter = false;
	    *buffer = colwertor->pBaseBuf;
	    *buffer_len = colwertor->local_size;
	    return OPAL_SUCCESS;
    }
#endif

    opal_colwertor_get_packed_size(colwertor, buffer_len);
    *freeAfter  = false;
    if( 0 == *buffer_len ) {
        *buffer     = NULL;
        return OMPI_SUCCESS;
    }
    iov.iov_len = *buffer_len;
    iov.iov_base = NULL;
    if (opal_colwertor_need_buffers(colwertor)) {
        iov.iov_base = malloc(*buffer_len);
        if (NULL == iov.iov_base) return OMPI_ERR_OUT_OF_RESOURCE;
        *freeAfter = true;
    }

    opal_colwertor_pack( colwertor, &iov, &iov_count, buffer_len );

    *buffer = iov.iov_base;

    return OMPI_SUCCESS;
}


__opal_attribute_always_inline__ static inline int
ompi_mtl_datatype_recv_buf(struct opal_colwertor_t *colwertor,
                           void ** buffer,
                           size_t *buffer_len,
                           bool *free_on_error)
{
    opal_colwertor_get_packed_size(colwertor, buffer_len);
    *free_on_error = false;
    if( 0 == *buffer_len ) {
        *buffer = NULL;
        *buffer_len = 0;
        return OMPI_SUCCESS;
    }
    if (opal_colwertor_need_buffers(colwertor)) {
        *buffer = malloc(*buffer_len);
        *free_on_error = true;
    } else {
        *buffer = colwertor->pBaseBuf +
            colwertor->use_desc->desc[colwertor->use_desc->used].end_loop.first_elem_disp;
    }
    return OMPI_SUCCESS;
}


__opal_attribute_always_inline__ static inline int
ompi_mtl_datatype_unpack(struct opal_colwertor_t *colwertor,
                         void *buffer,
                         size_t buffer_len)
{
    struct iovec iov;
    uint32_t iov_count = 1;

    if (buffer_len > 0 && opal_colwertor_need_buffers(colwertor)) {
        iov.iov_len = buffer_len;
        iov.iov_base = buffer;

        opal_colwertor_unpack(colwertor, &iov, &iov_count, &buffer_len );

        free(buffer);
    }

    return OMPI_SUCCESS;
}

#endif /* MTL_BASE_DATATYPE_H_INCLUDED */
