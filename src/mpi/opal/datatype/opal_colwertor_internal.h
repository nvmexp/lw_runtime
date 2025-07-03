/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2013      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef OPAL_COLWERTOR_INTERNAL_HAS_BEEN_INCLUDED
#define OPAL_COLWERTOR_INTERNAL_HAS_BEEN_INCLUDED

#include "opal_config.h"

#include "opal/datatype/opal_colwertor.h"

BEGIN_C_DECLS

typedef int32_t (*colwersion_fct_t)( opal_colwertor_t* pColwertor, uint32_t count,
                                     const void* from, size_t from_len, ptrdiff_t from_extent,
                                     void* to, size_t to_length, ptrdiff_t to_extent,
                                     ptrdiff_t *advance );

typedef struct opal_colwertor_master_t {
    struct opal_colwertor_master_t* next;
    uint32_t                        remote_arch;
    uint32_t                        flags;
    uint32_t                        hetero_mask;
    const size_t                    remote_sizes[OPAL_DATATYPE_MAX_PREDEFINED];
    colwersion_fct_t*               pFunctions;   /**< the colwertor functions pointer */
} opal_colwertor_master_t;

/*
 * Find or create a new master colwertor based on a specific architecture. The master
 * colwertor hold all informations related to a defined architecture, such as the sizes
 * of the predefined data-types, the colwersion functions, ...
 */
opal_colwertor_master_t* opal_colwertor_find_or_create_master( uint32_t remote_arch );

/*
 * Destroy all pending master colwertors. This function is usually called when we
 * shutdown the data-type engine, once all colwertors have been destroyed.
 */
void opal_colwertor_destroy_masters( void );


END_C_DECLS

#endif  /* OPAL_COLWERTOR_INTERNAL_HAS_BEEN_INCLUDED */
