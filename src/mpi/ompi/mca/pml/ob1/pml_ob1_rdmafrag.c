/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2014      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "pml_ob1.h"
#include "pml_ob1_rdmafrag.h"

static void mca_pml_ob1_rdma_frag_constructor (mca_pml_ob1_rdma_frag_t *frag)
{
    frag->local_handle = NULL;
}

OBJ_CLASS_INSTANCE(
    mca_pml_ob1_rdma_frag_t,
    opal_free_list_item_t,
    mca_pml_ob1_rdma_frag_constructor,
    NULL);
