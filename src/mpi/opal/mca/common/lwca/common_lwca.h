/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2011-2015 LWPU Corporation.  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_MCA_COMMON_LWDA_H
#define OPAL_MCA_COMMON_LWDA_H
#include "opal/mca/btl/btl.h"
#include "opal/datatype/opal_colwertor.h"

#define MEMHANDLE_SIZE 8
#define EVTHANDLE_SIZE 8

struct mca_rcache_common_lwda_reg_data_t {
    uint64_t memHandle[MEMHANDLE_SIZE];
    uint64_t evtHandle[EVTHANDLE_SIZE];
    uint64_t event;
    opal_ptr_t memh_seg_addr;
    size_t memh_seg_len;
};
typedef struct mca_rcache_common_lwda_reg_data_t mca_rcache_common_lwda_reg_data_t;

struct mca_rcache_common_lwda_reg_t {
    mca_rcache_base_registration_t base;
    mca_rcache_common_lwda_reg_data_t data;
};
typedef struct mca_rcache_common_lwda_reg_t mca_rcache_common_lwda_reg_t;
extern bool mca_common_lwda_enabled;

OPAL_DECLSPEC void mca_common_lwda_register_mca_variables(void);

OPAL_DECLSPEC void mca_common_lwda_register(void *ptr, size_t amount, char *msg);

OPAL_DECLSPEC void mca_common_lwda_unregister(void *ptr, char *msg);

OPAL_DECLSPEC void mca_common_wait_stream_synchronize(mca_rcache_common_lwda_reg_t *rget_reg);

OPAL_DECLSPEC int mca_common_lwda_memcpy(void *dst, void *src, size_t amount, char *msg,
                                         struct mca_btl_base_descriptor_t *, int *done);

OPAL_DECLSPEC int mca_common_lwda_record_ipc_event(char *msg,
                                               struct mca_btl_base_descriptor_t *frag);
OPAL_DECLSPEC int mca_common_lwda_record_dtoh_event(char *msg,
                                                    struct mca_btl_base_descriptor_t *frag);
OPAL_DECLSPEC int mca_common_lwda_record_htod_event(char *msg,
                                                    struct mca_btl_base_descriptor_t *frag);

OPAL_DECLSPEC void *mca_common_lwda_get_dtoh_stream(void);
OPAL_DECLSPEC void *mca_common_lwda_get_htod_stream(void);

OPAL_DECLSPEC int progress_one_lwda_ipc_event(struct mca_btl_base_descriptor_t **);
OPAL_DECLSPEC int progress_one_lwda_dtoh_event(struct mca_btl_base_descriptor_t **);
OPAL_DECLSPEC int progress_one_lwda_htod_event(struct mca_btl_base_descriptor_t **);

OPAL_DECLSPEC int mca_common_lwda_memhandle_matches(mca_rcache_common_lwda_reg_t *new_reg,
                                                    mca_rcache_common_lwda_reg_t *old_reg);

OPAL_DECLSPEC void mca_common_lwda_construct_event_and_handle(uintptr_t *event, void *handle);
OPAL_DECLSPEC void mca_common_lwda_destruct_event(uintptr_t event);

OPAL_DECLSPEC int lwda_getmemhandle(void *base, size_t, mca_rcache_base_registration_t *newreg,
                                    mca_rcache_base_registration_t *hdrreg);
OPAL_DECLSPEC int lwda_ungetmemhandle(void *reg_data, mca_rcache_base_registration_t *reg);
OPAL_DECLSPEC int lwda_openmemhandle(void *base, size_t size, mca_rcache_base_registration_t *newreg,
                                     mca_rcache_base_registration_t *hdrreg);
OPAL_DECLSPEC int lwda_closememhandle(void *reg_data, mca_rcache_base_registration_t *reg);
OPAL_DECLSPEC int mca_common_lwda_get_device(int *devicenum);
OPAL_DECLSPEC int mca_common_lwda_device_can_access_peer(int *access, int dev1, int dev2);
OPAL_DECLSPEC int mca_common_lwda_stage_one_init(void);
OPAL_DECLSPEC int mca_common_lwda_get_address_range(void *pbase, size_t *psize, void *base);
OPAL_DECLSPEC void mca_common_lwda_fini(void);
#if OPAL_LWDA_GDR_SUPPORT
OPAL_DECLSPEC bool mca_common_lwda_previously_freed_memory(mca_rcache_base_registration_t *reg);
OPAL_DECLSPEC void mca_common_lwda_get_buffer_id(mca_rcache_base_registration_t *reg);
#endif /* OPAL_LWDA_GDR_SUPPORT */
/**
 * Return:   0 if no packing is required for sending (the upper layer
 *             can use directly the pointer to the contiguous user
 *             buffer).
 *           1 if data does need to be packed, i.e. heterogeneous peers
 *             (source arch != dest arch) or non contiguous memory
 *             layout.
 */
static inline int32_t opal_colwertor_lwda_need_buffers( opal_colwertor_t* pColwertor )
{
    int32_t retval;
    uint32_t lwdaflag = pColwertor->flags & COLWERTOR_LWDA; /* Save LWCA flag */
    pColwertor->flags &= ~COLWERTOR_LWDA;              /* Clear LWCA flag if it exists */
    retval = opal_colwertor_need_buffers(pColwertor);
    pColwertor->flags |= lwdaflag; /* Restore LWCA flag */
    return retval;
}

#endif /* OPAL_MCA_COMMON_LWDA_H */
