/*
 * Copyright (c) 2014-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2015 LWPU Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_lwda.h"

#include <stdio.h>

#include "ompi/op/op.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/datatype/opal_datatype_lwda.h"

/*
 *	scan
 *
 *	Function:	- scan
 *	Accepts:	- same arguments as MPI_Scan()
 *	Returns:	- MPI_SUCCESS or error code
 */
int mca_coll_lwda_scan(const void *sbuf, void *rbuf, int count,
                       struct ompi_datatype_t *dtype,
                       struct ompi_op_t *op,
                       struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module)
{
    mca_coll_lwda_module_t *s = (mca_coll_lwda_module_t*) module;
    ptrdiff_t gap;
    char *rbuf1 = NULL, *sbuf1 = NULL, *rbuf2 = NULL;
    size_t bufsize;
    int rc;

    bufsize = opal_datatype_span(&dtype->super, count, &gap);

    if ((MPI_IN_PLACE != sbuf) && (opal_lwda_check_bufs((char *)sbuf, NULL))) {
        sbuf1 = (char*)malloc(bufsize);
        if (NULL == sbuf1) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        opal_lwda_memcpy_sync(sbuf1, sbuf, bufsize);
        sbuf = sbuf1 - gap;
    }

    if (opal_lwda_check_bufs(rbuf, NULL)) {
        rbuf1 = (char*)malloc(bufsize);
        if (NULL == rbuf1) {
            if (NULL != sbuf1) free(sbuf1);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        opal_lwda_memcpy_sync(rbuf1, rbuf, bufsize);
        rbuf2 = rbuf; /* save away original buffer */
        rbuf = rbuf1 - gap;
    }
    rc = s->c_coll.coll_scan(sbuf, rbuf, count, dtype, op, comm,
                             s->c_coll.coll_scan_module);
    if (NULL != sbuf1) {
        free(sbuf1);
    }
    if (NULL != rbuf1) {
        rbuf = rbuf2;
        opal_lwda_memcpy_sync(rbuf, rbuf1, bufsize);
        free(rbuf1);
    }
    return rc;
}
