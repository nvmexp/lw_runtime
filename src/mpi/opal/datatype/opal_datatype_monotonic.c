/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2018-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stddef.h>

#include "opal/constants.h"
#include "opal/datatype/opal_datatype.h"
#include "opal/datatype/opal_datatype_internal.h"
#include "opal/datatype/opal_colwertor.h"

#define OPAL_DATATYPE_MAX_MONOTONIC_IOVEC 32

/**
 * Check if the datatype describes a memory layout where the pointers to
 * the contiguous pieces are always advancing in the same direction, i.e.
 * there is no potential for overlap.
 */
int32_t opal_datatype_is_monotonic(opal_datatype_t* type )
{
    struct iovec iov[OPAL_DATATYPE_MAX_MONOTONIC_IOVEC];
    ptrdiff_t upper_limit = (ptrdiff_t)type->true_lb;  /* as colwersion base will be NULL the first address is true_lb */
    size_t max_data = 0x7FFFFFFF;
    opal_colwertor_t *pColw;
    bool monotonic = true;
    uint32_t iov_count;
    int rc;

    pColw  = opal_colwertor_create( opal_local_arch, 0 );
    if (OPAL_UNLIKELY(NULL == pColw)) {
        return -1;
    }
    rc = opal_colwertor_prepare_for_send( pColw, type, 1, NULL );
    if( OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        OBJ_RELEASE(pColw);
        return -1;
    }

    do {
        iov_count = OPAL_DATATYPE_MAX_MONOTONIC_IOVEC;
        rc = opal_colwertor_raw( pColw, iov, &iov_count, &max_data);
        for (uint32_t i = 0; i < iov_count; i++) {
            if ((ptrdiff_t)iov[i].iov_base < upper_limit) {
                monotonic = false;
                goto cleanup;
            }
            /* The new upper bound is at the end of the iovec */
            upper_limit = (ptrdiff_t)iov[i].iov_base + iov[i].iov_len;
        }
    } while (rc != 1);

  cleanup:
    OBJ_RELEASE( pColw );

    return monotonic;
}
