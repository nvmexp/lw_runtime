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
 * Copyright (c) 2015      Intel, Inc. All rights reserved
 * Copyright (c) 2015-2017 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MTL_PSM2_H_HAS_BEEN_INCLUDED
#define MTL_PSM2_H_HAS_BEEN_INCLUDED

#include "ompi/mca/pml/pml.h"
#include "ompi/mca/mtl/mtl.h"
#include "ompi/mca/mtl/base/base.h"
#include "ompi/proc/proc.h"
#include "opal/datatype/opal_colwertor.h"
#include <psm2.h>
#include <psm2_mq.h>

BEGIN_C_DECLS

/* MPI_THREAD_MULTIPLE_SUPPORT */
extern opal_mutex_t mtl_psm2_mq_mutex;

/* MTL interface functions */
extern int ompi_mtl_psm2_add_procs(struct mca_mtl_base_module_t* mtl,
                          size_t nprocs,
                          struct ompi_proc_t** procs);

extern int ompi_mtl_psm2_del_procs(struct mca_mtl_base_module_t* mtl,
                                 size_t nprocs,
                                 struct ompi_proc_t** procs);

int
ompi_mtl_psm2_send(struct mca_mtl_base_module_t* mtl,
                 struct ompi_communicator_t* comm,
                 int dest,
                 int tag,
                 struct opal_colwertor_t *colwertor,
                 mca_pml_base_send_mode_t mode);

extern int ompi_mtl_psm2_isend(struct mca_mtl_base_module_t* mtl,
                             struct ompi_communicator_t* comm,
                             int dest,
                             int tag,
                             struct opal_colwertor_t *colwertor,
                             mca_pml_base_send_mode_t mode,
                             bool blocking,
                             mca_mtl_request_t * mtl_request);

extern int ompi_mtl_psm2_irecv(struct mca_mtl_base_module_t* mtl,
                             struct ompi_communicator_t *comm,
                             int src,
                             int tag,
                             struct opal_colwertor_t *colwertor,
                             struct mca_mtl_request_t *mtl_request);


extern int ompi_mtl_psm2_iprobe(struct mca_mtl_base_module_t* mtl,
                              struct ompi_communicator_t *comm,
                              int src,
                              int tag,
                              int *flag,
                              struct ompi_status_public_t *status);

extern int ompi_mtl_psm2_imrecv(struct mca_mtl_base_module_t* mtl,
                               struct opal_colwertor_t *colwertor,
                               struct ompi_message_t **message,
                               struct mca_mtl_request_t *mtl_request);

extern int ompi_mtl_psm2_improbe(struct mca_mtl_base_module_t *mtl,
                                struct ompi_communicator_t *comm,
                                int src,
                                int tag,
                                int *matched,
                                struct ompi_message_t **message,
                                struct ompi_status_public_t *status);

extern int ompi_mtl_psm2_cancel(struct mca_mtl_base_module_t* mtl,
                              struct mca_mtl_request_t *mtl_request,
                              int flag);

extern int ompi_mtl_psm2_add_comm(struct mca_mtl_base_module_t *mtl,
                                 struct ompi_communicator_t *comm);

extern int ompi_mtl_psm2_del_comm(struct mca_mtl_base_module_t *mtl,
                                 struct ompi_communicator_t *comm);

extern int ompi_mtl_psm2_finalize(struct mca_mtl_base_module_t* mtl);

int ompi_mtl_psm2_module_init(int local_rank, int num_local_procs);

extern int ompi_mtl_psm2_register_pvars(void);


END_C_DECLS

#endif  /* MTL_PSM2_H_HAS_BEEN_INCLUDED */
