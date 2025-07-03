/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011      LWPU Corporation.  All rights reserved.
 * Copyright (c) 2013-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2017      Intel, Inc. All rights reserved
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "opal/prefetch.h"
#include "opal/util/arch.h"
#include "opal/util/output.h"

#include "opal/datatype/opal_datatype_internal.h"
#include "opal/datatype/opal_datatype.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/datatype/opal_datatype_checksum.h"
#include "opal/datatype/opal_datatype_prototypes.h"
#include "opal/datatype/opal_colwertor_internal.h"
#if OPAL_LWDA_SUPPORT
#include "opal/datatype/opal_datatype_lwda.h"
#define MEMCPY_LWDA( DST, SRC, BLENGTH, COLWERTOR ) \
    COLWERTOR->cbmemcpy( (DST), (SRC), (BLENGTH), (COLWERTOR) )
#endif

static void opal_colwertor_construct( opal_colwertor_t* colwertor )
{
    colwertor->pStack         = colwertor->static_stack;
    colwertor->stack_size     = DT_STATIC_STACK_SIZE;
    colwertor->partial_length = 0;
    colwertor->remoteArch     = opal_local_arch;
    colwertor->flags          = OPAL_DATATYPE_FLAG_NO_GAPS | COLWERTOR_COMPLETED;
#if OPAL_LWDA_SUPPORT
    colwertor->cbmemcpy       = &opal_lwda_memcpy;
#endif
}


static void opal_colwertor_destruct( opal_colwertor_t* colwertor )
{
    opal_colwertor_cleanup( colwertor );
}

OBJ_CLASS_INSTANCE(opal_colwertor_t, opal_object_t, opal_colwertor_construct, opal_colwertor_destruct );

static opal_colwertor_master_t* opal_colwertor_master_list = NULL;

extern colwersion_fct_t opal_datatype_heterogeneous_copy_functions[OPAL_DATATYPE_MAX_PREDEFINED];
extern colwersion_fct_t opal_datatype_copy_functions[OPAL_DATATYPE_MAX_PREDEFINED];

void opal_colwertor_destroy_masters( void )
{
    opal_colwertor_master_t* master = opal_colwertor_master_list;

    while( NULL != master ) {
        opal_colwertor_master_list = master->next;
        master->next = NULL;
        /* Cleanup the colwersion function if not one of the defaults */
        if( (master->pFunctions != opal_datatype_heterogeneous_copy_functions) &&
            (master->pFunctions != opal_datatype_copy_functions) )
            free( master->pFunctions );

        free( master );
        master = opal_colwertor_master_list;
    }
}

/**
 * Find or create a colwertor suitable for the remote architecture. If there
 * is already a master colwertor for this architecture then return it.
 * Otherwise, create and initialize a full featured master colwertor.
 */
opal_colwertor_master_t* opal_colwertor_find_or_create_master( uint32_t remote_arch )
{
    opal_colwertor_master_t* master = opal_colwertor_master_list;
    int i;
    size_t* remote_sizes;

    while( NULL != master ) {
        if( master->remote_arch == remote_arch )
            return master;
        master = master->next;
    }
    /**
     * Create a new colwertor matching the specified architecture and add it to the
     * master colwertor list.
     */
    master = (opal_colwertor_master_t*)malloc( sizeof(opal_colwertor_master_t) );
    master->next = opal_colwertor_master_list;
    opal_colwertor_master_list = master;
    master->remote_arch = remote_arch;
    master->flags       = 0;
    master->hetero_mask = 0;
    /**
     * Most of the sizes will be identical, so for now just make a copy of
     * the local ones. As master->remote_sizes is defined as being an array of
     * consts we have to manually cast it before using it for writing purposes.
     */
    remote_sizes = (size_t*)master->remote_sizes;
    memcpy(remote_sizes, opal_datatype_local_sizes, sizeof(size_t) * OPAL_DATATYPE_MAX_PREDEFINED);
    /**
     * If the local and remote architecture are the same there is no need
     * to check for the remote data sizes. They will always be the same as
     * the local ones.
     */
    if( master->remote_arch == opal_local_arch ) {
        master->pFunctions = opal_datatype_copy_functions;
        master->flags |= COLWERTOR_HOMOGENEOUS;
        return master;
    }

    /* Find out the remote bool size */
    if( opal_arch_checkmask( &master->remote_arch, OPAL_ARCH_BOOLIS8 ) ) {
        remote_sizes[OPAL_DATATYPE_BOOL] = 1;
    } else if( opal_arch_checkmask( &master->remote_arch, OPAL_ARCH_BOOLIS16 ) ) {
        remote_sizes[OPAL_DATATYPE_BOOL] = 2;
    } else if( opal_arch_checkmask( &master->remote_arch, OPAL_ARCH_BOOLIS32 ) ) {
        remote_sizes[OPAL_DATATYPE_BOOL] = 4;
    } else {
        opal_output( 0, "Unknown sizeof(bool) for the remote architecture\n" );
    }

    /**
     * Now we can compute the colwersion mask. For all sizes where the remote
     * and local architecture differ a colwersion is needed. Moreover, if the
     * 2 architectures don't have the same endianess all data with a length
     * over 2 bytes (with the exception of logicals) have to be byte-swapped.
     */
    for( i = OPAL_DATATYPE_FIRST_TYPE; i < OPAL_DATATYPE_MAX_PREDEFINED; i++ ) {
        if( remote_sizes[i] != opal_datatype_local_sizes[i] )
            master->hetero_mask |= (((uint32_t)1) << i);
    }
    if( opal_arch_checkmask( &master->remote_arch, OPAL_ARCH_ISBIGENDIAN ) !=
        opal_arch_checkmask( &opal_local_arch, OPAL_ARCH_ISBIGENDIAN ) ) {
        uint32_t hetero_mask = 0;

        for( i = OPAL_DATATYPE_FIRST_TYPE; i < OPAL_DATATYPE_MAX_PREDEFINED; i++ ) {
            if( remote_sizes[i] > 1 )
                hetero_mask |= (((uint32_t)1) << i);
        }
        hetero_mask &= ~(((uint32_t)1) << OPAL_DATATYPE_BOOL);
        master->hetero_mask |= hetero_mask;
    }
    master->pFunctions = (colwersion_fct_t*)malloc( sizeof(opal_datatype_heterogeneous_copy_functions) );
    /**
     * Usually the heterogeneous functions are slower than the copy ones. Let's
     * try to minimize the usage of the heterogeneous versions.
     */
    for( i = OPAL_DATATYPE_FIRST_TYPE; i < OPAL_DATATYPE_MAX_PREDEFINED; i++ ) {
        if( master->hetero_mask & (((uint32_t)1) << i) )
            master->pFunctions[i] = opal_datatype_heterogeneous_copy_functions[i];
        else
            master->pFunctions[i] = opal_datatype_copy_functions[i];
    }

    /* We're done so far, return the mater colwertor */
    return master;
}


opal_colwertor_t* opal_colwertor_create( int32_t remote_arch, int32_t mode )
{
    opal_colwertor_t* colwertor = OBJ_NEW(opal_colwertor_t);
    opal_colwertor_master_t* master;

    master = opal_colwertor_find_or_create_master( remote_arch );

    colwertor->remoteArch = remote_arch;
    colwertor->stack_pos  = 0;
    colwertor->flags      = master->flags;
    colwertor->master     = master;

    return colwertor;
}

#define OPAL_COLWERTOR_SET_STATUS_BEFORE_PACK_UNPACK( COLWERTOR, IOV, OUT, MAX_DATA ) \
    do {                                                                \
        /* protect against over packing data */                         \
        if( OPAL_UNLIKELY((COLWERTOR)->flags & COLWERTOR_COMPLETED) ) { \
            (IOV)[0].iov_len = 0;                                       \
            *(OUT) = 0;                                                 \
            *(MAX_DATA) = 0;                                            \
            return 1;  /* nothing to do */                              \
        }                                                               \
        (COLWERTOR)->checksum = OPAL_CSUM_ZERO;                         \
        (COLWERTOR)->csum_ui1 = 0;                                      \
        (COLWERTOR)->csum_ui2 = 0;                                      \
        assert( (COLWERTOR)->bColwerted < (COLWERTOR)->local_size );    \
    } while(0)

/**
 * Return 0 if everything went OK and if there is still room before the complete
 *          colwersion of the data (need additional call with others input buffers )
 *        1 if everything went fine and the data was completly colwerted
 *       -1 something wrong oclwrs.
 */
int32_t opal_colwertor_pack( opal_colwertor_t* pColw,
                             struct iovec* iov, uint32_t* out_size,
                             size_t* max_data )
{
    OPAL_COLWERTOR_SET_STATUS_BEFORE_PACK_UNPACK( pColw, iov, out_size, max_data );

    if( OPAL_LIKELY(pColw->flags & COLWERTOR_NO_OP) ) {
        /**
         * We are doing colwersion on a contiguous datatype on a homogeneous
         * environment. The colwertor contain minimal information, we only
         * use the bColwerted to manage the colwersion.
         */
        uint32_t i;
        unsigned char* base_pointer;
        size_t pending_length = pColw->local_size - pColw->bColwerted;

        *max_data = pending_length;
        opal_colwertor_get_lwrrent_pointer( pColw, (void**)&base_pointer );

        for( i = 0; i < *out_size; i++ ) {
            if( iov[i].iov_len >= pending_length ) {
                goto complete_contiguous_data_pack;
            }
            if( OPAL_LIKELY(NULL == iov[i].iov_base) )
                iov[i].iov_base = (IOVBASE_TYPE *) base_pointer;
            else
#if OPAL_LWDA_SUPPORT
                MEMCPY_LWDA( iov[i].iov_base, base_pointer, iov[i].iov_len, pColw );
#else
                MEMCPY( iov[i].iov_base, base_pointer, iov[i].iov_len );
#endif
            pending_length -= iov[i].iov_len;
            base_pointer += iov[i].iov_len;
        }
        *max_data -= pending_length;
        pColw->bColwerted += (*max_data);
        return 0;

complete_contiguous_data_pack:
        iov[i].iov_len = pending_length;
        if( OPAL_LIKELY(NULL == iov[i].iov_base) )
            iov[i].iov_base = (IOVBASE_TYPE *) base_pointer;
        else
#if OPAL_LWDA_SUPPORT
            MEMCPY_LWDA( iov[i].iov_base, base_pointer, iov[i].iov_len, pColw );
#else
            MEMCPY( iov[i].iov_base, base_pointer, iov[i].iov_len );
#endif
        pColw->bColwerted = pColw->local_size;
        *out_size = i + 1;
        pColw->flags |= COLWERTOR_COMPLETED;
        return 1;
    }

    return pColw->fAdvance( pColw, iov, out_size, max_data );
}


int32_t opal_colwertor_unpack( opal_colwertor_t* pColw,
                               struct iovec* iov, uint32_t* out_size,
                               size_t* max_data )
{
    OPAL_COLWERTOR_SET_STATUS_BEFORE_PACK_UNPACK( pColw, iov, out_size, max_data );

    if( OPAL_LIKELY(pColw->flags & COLWERTOR_NO_OP) ) {
        /**
         * We are doing colwersion on a contiguous datatype on a homogeneous
         * environment. The colwertor contain minimal informations, we only
         * use the bColwerted to manage the colwersion.
         */
        uint32_t i;
        unsigned char* base_pointer;
        size_t pending_length = pColw->local_size - pColw->bColwerted;

        *max_data = pending_length;
        opal_colwertor_get_lwrrent_pointer( pColw, (void**)&base_pointer );

        for( i = 0; i < *out_size; i++ ) {
            if( iov[i].iov_len >= pending_length ) {
                goto complete_contiguous_data_unpack;
            }
#if OPAL_LWDA_SUPPORT
            MEMCPY_LWDA( base_pointer, iov[i].iov_base, iov[i].iov_len, pColw );
#else
            MEMCPY( base_pointer, iov[i].iov_base, iov[i].iov_len );
#endif
            pending_length -= iov[i].iov_len;
            base_pointer += iov[i].iov_len;
        }
        *max_data -= pending_length;
        pColw->bColwerted += (*max_data);
        return 0;

complete_contiguous_data_unpack:
        iov[i].iov_len = pending_length;
#if OPAL_LWDA_SUPPORT
        MEMCPY_LWDA( base_pointer, iov[i].iov_base, iov[i].iov_len, pColw );
#else
        MEMCPY( base_pointer, iov[i].iov_base, iov[i].iov_len );
#endif
        pColw->bColwerted = pColw->local_size;
        *out_size = i + 1;
        pColw->flags |= COLWERTOR_COMPLETED;
        return 1;
    }

    return pColw->fAdvance( pColw, iov, out_size, max_data );
}

static inline int
opal_colwertor_create_stack_with_pos_contig( opal_colwertor_t* pColwertor,
                                             size_t starting_point, const size_t* sizes )
{
    dt_stack_t* pStack;   /* pointer to the position on the stack */
    const opal_datatype_t* pData = pColwertor->pDesc;
    dt_elem_desc_t* pElems;
    size_t count;
    ptrdiff_t extent;

    pStack = pColwertor->pStack;
    /**
     * The prepare function already make the selection on which data representation
     * we have to use: normal one or the optimized version ?
     */
    pElems = pColwertor->use_desc->desc;

    count = starting_point / pData->size;
    extent = pData->ub - pData->lb;

    pStack[0].type     = OPAL_DATATYPE_LOOP;  /* the first one is always the loop */
    pStack[0].count    = pColwertor->count - count;
    pStack[0].index    = -1;
    pStack[0].disp     = count * extent;

    /* now compute the number of pending bytes */
    count = starting_point % pData->size;
    /**
     * We save the current displacement starting from the begining
     * of this data.
     */
    if( OPAL_LIKELY(0 == count) ) {
        pStack[1].type     = pElems->elem.common.type;
        pStack[1].count    = pElems->elem.blocklen;
    } else {
        pStack[1].type  = OPAL_DATATYPE_UINT1;
        pStack[1].count = pData->size - count;
    }
    pStack[1].disp  = count;
    pStack[1].index = 0;  /* useless */

    pColwertor->bColwerted = starting_point;
    pColwertor->stack_pos = 1;
    assert( 0 == pColwertor->partial_length );
    return OPAL_SUCCESS;
}

static inline int
opal_colwertor_create_stack_at_begining( opal_colwertor_t* colwertor,
                                         const size_t* sizes )
{
    dt_stack_t* pStack = colwertor->pStack;
    dt_elem_desc_t* pElems;

    /**
     * The prepare function already make the selection on which data representation
     * we have to use: normal one or the optimized version ?
     */
    pElems = colwertor->use_desc->desc;

    colwertor->stack_pos      = 1;
    colwertor->partial_length = 0;
    colwertor->bColwerted     = 0;
    /**
     * Fill the first position on the stack. This one correspond to the
     * last fake OPAL_DATATYPE_END_LOOP that we add to the data representation and
     * allow us to move quickly inside the datatype when we have a count.
     */
    pStack[0].index = -1;
    pStack[0].count = colwertor->count;
    pStack[0].disp  = 0;
    pStack[0].type  = OPAL_DATATYPE_LOOP;

    pStack[1].index = 0;
    pStack[1].disp = 0;
    if( pElems[0].elem.common.type == OPAL_DATATYPE_LOOP ) {
        pStack[1].count = pElems[0].loop.loops;
        pStack[1].type  = OPAL_DATATYPE_LOOP;
    } else {
        pStack[1].count = (size_t)pElems[0].elem.count * pElems[0].elem.blocklen;
        pStack[1].type  = pElems[0].elem.common.type;
    }
    return OPAL_SUCCESS;
}


int32_t opal_colwertor_set_position_nocheck( opal_colwertor_t* colwertor,
                                             size_t* position )
{
    int32_t rc;

    /**
     * create_stack_with_pos_contig always set the position relative to the ZERO
     * position, so there is no need for special handling. In all other cases,
     * if we plan to rollback the colwertor then first we have to reset it at
     * the beginning.
     */
    if( OPAL_LIKELY(colwertor->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS) ) {
        rc = opal_colwertor_create_stack_with_pos_contig( colwertor, (*position),
                                                          opal_datatype_local_sizes );
    } else {
        if( (0 == (*position)) || ((*position) < colwertor->bColwerted) ) {
            rc = opal_colwertor_create_stack_at_begining( colwertor, opal_datatype_local_sizes );
            if( 0 == (*position) ) return rc;
        }
        rc = opal_colwertor_generic_simple_position( colwertor, position );
        /**
         * If we have a non-contigous send colwertor don't allow it move in the middle
         * of a predefined datatype, it won't be able to copy out the left-overs
         * anyway. Instead force the position to stay on predefined datatypes
         * boundaries. As we allow partial predefined datatypes on the contiguous
         * case, we should be accepted by any receiver colwertor.
         */
        if( COLWERTOR_SEND & colwertor->flags ) {
            colwertor->bColwerted -= colwertor->partial_length;
            colwertor->partial_length = 0;
        }
    }
    *position = colwertor->bColwerted;
    return rc;
}

static size_t
opal_datatype_compute_remote_size( const opal_datatype_t* pData,
                                   const size_t* sizes )
{
    uint32_t typeMask = pData->bdt_used;
    size_t length = 0;

    if (opal_datatype_is_predefined(pData)) {
        return sizes[pData->desc.desc->elem.common.type];
    }

    if( OPAL_UNLIKELY(NULL == pData->ptypes) ) {
        /* Allocate and fill the array of types used in the datatype description */
        opal_datatype_compute_ptypes( (opal_datatype_t*)pData );
    }

    for( int i = OPAL_DATATYPE_FIRST_TYPE; typeMask && (i < OPAL_DATATYPE_MAX_PREDEFINED); i++ ) {
        if( typeMask & ((uint32_t)1 << i) ) {
            length += (pData->ptypes[i] * sizes[i]);
            typeMask ^= ((uint32_t)1 << i);
        }
    }
    return length;
}

/**
 * Compute the remote size. If necessary remove the homogeneous flag
 * and redirect the colwertor description toward the non-optimized
 * datatype representation.
 */
size_t opal_colwertor_compute_remote_size( opal_colwertor_t* pColwertor )
{
    opal_datatype_t* datatype = (opal_datatype_t*)pColwertor->pDesc;
    
    pColwertor->remote_size = pColwertor->local_size;
    if( OPAL_UNLIKELY(datatype->bdt_used & pColwertor->master->hetero_mask) ) {
        pColwertor->flags &= (~COLWERTOR_HOMOGENEOUS);
        if (!(pColwertor->flags & COLWERTOR_SEND && pColwertor->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS)) {
            pColwertor->use_desc = &(datatype->desc);
        }
        if( 0 == (pColwertor->flags & COLWERTOR_HAS_REMOTE_SIZE) ) {
            /* This is for a single datatype, we must update it with the count */
            pColwertor->remote_size = opal_datatype_compute_remote_size(datatype,
                                                                        pColwertor->master->remote_sizes);
            pColwertor->remote_size *= pColwertor->count;
        }
    }
    pColwertor->flags |= COLWERTOR_HAS_REMOTE_SIZE;
    return pColwertor->remote_size;
}

/**
 * This macro will initialize a colwertor based on a previously created
 * colwertor. The idea is the move outside these function the heavy
 * selection of architecture features for the colwertors. I consider
 * here that the colwertor is clean, either never initialized or already
 * cleaned.
 */
#define OPAL_COLWERTOR_PREPARE( colwertor, datatype, count, pUserBuf )  \
    {                                                                   \
        colwertor->local_size = count * datatype->size;                 \
        colwertor->pBaseBuf   = (unsigned char*)pUserBuf;               \
        colwertor->count      = count;                                  \
        colwertor->pDesc      = (opal_datatype_t*)datatype;             \
        colwertor->bColwerted = 0;                                      \
        colwertor->use_desc   = &(datatype->opt_desc);                  \
        /* If the data is empty we just mark the colwertor as           \
         * completed. With this flag set the pack and unpack functions  \
         * will not do anything.                                        \
         */                                                             \
        if( OPAL_UNLIKELY((0 == count) || (0 == datatype->size)) ) {    \
            colwertor->flags |= (OPAL_DATATYPE_FLAG_NO_GAPS | COLWERTOR_COMPLETED | COLWERTOR_HAS_REMOTE_SIZE); \
            colwertor->local_size = colwertor->remote_size = 0;         \
            return OPAL_SUCCESS;                                        \
        }                                                               \
                                                                        \
        /* Grab the datatype part of the flags */                       \
        colwertor->flags     &= COLWERTOR_TYPE_MASK;                    \
        colwertor->flags     |= (COLWERTOR_DATATYPE_MASK & datatype->flags); \
        colwertor->flags     |= (COLWERTOR_NO_OP | COLWERTOR_HOMOGENEOUS); \
                                                                        \
        colwertor->remote_size = colwertor->local_size;                 \
        if( OPAL_LIKELY(colwertor->remoteArch == opal_local_arch) ) {   \
            if( !(colwertor->flags & COLWERTOR_WITH_CHECKSUM) &&        \
                ((colwertor->flags & OPAL_DATATYPE_FLAG_NO_GAPS) || \
                 ((colwertor->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS) && (1 == count))) ) { \
                return OPAL_SUCCESS;                                    \
            }                                                           \
        }                                                               \
                                                                        \
        assert( (colwertor)->pDesc == (datatype) );                     \
        opal_colwertor_compute_remote_size( colwertor );                \
        assert( NULL != colwertor->use_desc->desc );                    \
        /* For predefined datatypes (contiguous) do nothing more */     \
        /* if checksum is enabled then always continue */               \
        if( ((colwertor->flags & (COLWERTOR_WITH_CHECKSUM | OPAL_DATATYPE_FLAG_NO_GAPS)) \
             == OPAL_DATATYPE_FLAG_NO_GAPS) &&                          \
            ((colwertor->flags & (COLWERTOR_SEND | COLWERTOR_HOMOGENEOUS)) == \
             (COLWERTOR_SEND | COLWERTOR_HOMOGENEOUS)) ) {              \
            return OPAL_SUCCESS;                                        \
        }                                                               \
        colwertor->flags &= ~COLWERTOR_NO_OP;                           \
        {                                                               \
            uint32_t required_stack_length = datatype->loops + 1;       \
                                                                        \
            if( required_stack_length > colwertor->stack_size ) {       \
                assert(colwertor->pStack == colwertor->static_stack);   \
                colwertor->stack_size = required_stack_length;          \
                colwertor->pStack     = (dt_stack_t*)malloc(sizeof(dt_stack_t) * \
                                                            colwertor->stack_size ); \
            }                                                           \
        }                                                               \
        opal_colwertor_create_stack_at_begining( colwertor, opal_datatype_local_sizes ); \
    }


int32_t opal_colwertor_prepare_for_recv( opal_colwertor_t* colwertor,
                                         const struct opal_datatype_t* datatype,
                                         size_t count,
                                         const void* pUserBuf )
{
    /* Here I should check that the data is not overlapping */

    colwertor->flags |= COLWERTOR_RECV;
#if OPAL_LWDA_SUPPORT
    if (!( colwertor->flags & COLWERTOR_SKIP_LWDA_INIT )) {
        mca_lwda_colwertor_init(colwertor, pUserBuf);
    }
#endif

    assert(! (colwertor->flags & COLWERTOR_SEND));
    OPAL_COLWERTOR_PREPARE( colwertor, datatype, count, pUserBuf );

#if defined(CHECKSUM)
    if( OPAL_UNLIKELY(colwertor->flags & COLWERTOR_WITH_CHECKSUM) ) {
        if( OPAL_UNLIKELY(!(colwertor->flags & COLWERTOR_HOMOGENEOUS)) ) {
            colwertor->fAdvance = opal_unpack_general_checksum;
        } else {
            if( colwertor->pDesc->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                colwertor->fAdvance = opal_unpack_homogeneous_contig_checksum;
            } else {
                colwertor->fAdvance = opal_generic_simple_unpack_checksum;
            }
        }
    } else
#endif  /* defined(CHECKSUM) */
        if( OPAL_UNLIKELY(!(colwertor->flags & COLWERTOR_HOMOGENEOUS)) ) {
            colwertor->fAdvance = opal_unpack_general;
        } else {
            if( colwertor->pDesc->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                colwertor->fAdvance = opal_unpack_homogeneous_contig;
            } else {
                colwertor->fAdvance = opal_generic_simple_unpack;
            }
        }
    return OPAL_SUCCESS;
}


int32_t opal_colwertor_prepare_for_send( opal_colwertor_t* colwertor,
                                         const struct opal_datatype_t* datatype,
                                         size_t count,
                                         const void* pUserBuf )
{
    colwertor->flags |= COLWERTOR_SEND;
#if OPAL_LWDA_SUPPORT
    if (!( colwertor->flags & COLWERTOR_SKIP_LWDA_INIT )) {
        mca_lwda_colwertor_init(colwertor, pUserBuf);
    }
#endif

    OPAL_COLWERTOR_PREPARE( colwertor, datatype, count, pUserBuf );

#if defined(CHECKSUM)
    if( colwertor->flags & COLWERTOR_WITH_CHECKSUM ) {
        if( COLWERTOR_SEND_COLWERSION == (colwertor->flags & (COLWERTOR_SEND_COLWERSION|COLWERTOR_HOMOGENEOUS)) ) {
            colwertor->fAdvance = opal_pack_general_checksum;
        } else {
            if( datatype->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                if( ((datatype->ub - datatype->lb) == (ptrdiff_t)datatype->size)
                    || (1 >= colwertor->count) )
                    colwertor->fAdvance = opal_pack_homogeneous_contig_checksum;
                else
                    colwertor->fAdvance = opal_pack_homogeneous_contig_with_gaps_checksum;
            } else {
                colwertor->fAdvance = opal_generic_simple_pack_checksum;
            }
        }
    } else
#endif  /* defined(CHECKSUM) */
        if( COLWERTOR_SEND_COLWERSION == (colwertor->flags & (COLWERTOR_SEND_COLWERSION|COLWERTOR_HOMOGENEOUS)) ) {
            colwertor->fAdvance = opal_pack_general;
        } else {
            if( datatype->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                if( ((datatype->ub - datatype->lb) == (ptrdiff_t)datatype->size)
                    || (1 >= colwertor->count) )
                    colwertor->fAdvance = opal_pack_homogeneous_contig;
                else
                    colwertor->fAdvance = opal_pack_homogeneous_contig_with_gaps;
            } else {
                colwertor->fAdvance = opal_generic_simple_pack;
            }
        }
    return OPAL_SUCCESS;
}

/*
 * These functions can be used in order to create an IDENTICAL copy of one colwertor. In this
 * context IDENTICAL means that the datatype and count and all other properties of the basic
 * colwertor get replicated on this new colwertor. However, the references to the datatype
 * are not increased. This function take special care about the stack. If all the cases the
 * stack is created with the correct number of entries but if the copy_stack is true (!= 0)
 * then the content of the old stack is copied on the new one. The result will be a colwertor
 * ready to use starting from the old position. If copy_stack is false then the colwertor
 * is created with a empty stack (you have to use opal_colwertor_set_position before using it).
 */
int opal_colwertor_clone( const opal_colwertor_t* source,
                          opal_colwertor_t* destination,
                          int32_t copy_stack )
{
    destination->remoteArch        = source->remoteArch;
    destination->flags             = source->flags;
    destination->pDesc             = source->pDesc;
    destination->use_desc          = source->use_desc;
    destination->count             = source->count;
    destination->pBaseBuf          = source->pBaseBuf;
    destination->fAdvance          = source->fAdvance;
    destination->master            = source->master;
    destination->local_size        = source->local_size;
    destination->remote_size       = source->remote_size;
    /* create the stack */
    if( OPAL_UNLIKELY(source->stack_size > DT_STATIC_STACK_SIZE) ) {
        destination->pStack = (dt_stack_t*)malloc(sizeof(dt_stack_t) * source->stack_size );
    } else {
        destination->pStack = destination->static_stack;
    }
    destination->stack_size = source->stack_size;

    /* initialize the stack */
    if( OPAL_LIKELY(0 == copy_stack) ) {
        destination->bColwerted = -1;
        destination->stack_pos  = -1;
    } else {
        memcpy( destination->pStack, source->pStack, sizeof(dt_stack_t) * (source->stack_pos+1) );
        destination->bColwerted = source->bColwerted;
        destination->stack_pos  = source->stack_pos;
    }
#if OPAL_LWDA_SUPPORT
    destination->cbmemcpy   = source->cbmemcpy;
#endif
    return OPAL_SUCCESS;
}


void opal_colwertor_dump( opal_colwertor_t* colwertor )
{
    opal_output( 0, "Colwertor %p count %" PRIsize_t " stack position %u bColwerted %" PRIsize_t "\n"
                 "\tlocal_size %" PRIsize_t " remote_size %" PRIsize_t " flags %X stack_size %u pending_length %" PRIsize_t "\n"
                 "\tremote_arch %u local_arch %u\n",
                 (void*)colwertor,
                 colwertor->count, colwertor->stack_pos, colwertor->bColwerted,
                 colwertor->local_size, colwertor->remote_size,
                 colwertor->flags, colwertor->stack_size, colwertor->partial_length,
                 colwertor->remoteArch, opal_local_arch );
    if( colwertor->flags & COLWERTOR_RECV ) opal_output( 0, "unpack ");
    if( colwertor->flags & COLWERTOR_SEND ) opal_output( 0, "pack ");
    if( colwertor->flags & COLWERTOR_SEND_COLWERSION ) opal_output( 0, "colwersion ");
    if( colwertor->flags & COLWERTOR_HOMOGENEOUS ) opal_output( 0, "homogeneous " );
    else opal_output( 0, "heterogeneous ");
    if( colwertor->flags & COLWERTOR_NO_OP ) opal_output( 0, "no_op ");
    if( colwertor->flags & COLWERTOR_WITH_CHECKSUM ) opal_output( 0, "checksum ");
    if( colwertor->flags & COLWERTOR_LWDA ) opal_output( 0, "LWCA ");
    if( colwertor->flags & COLWERTOR_LWDA_ASYNC ) opal_output( 0, "LWCA Async ");
    if( colwertor->flags & COLWERTOR_COMPLETED ) opal_output( 0, "COMPLETED ");

    opal_datatype_dump( colwertor->pDesc );
    if( !((0 == colwertor->stack_pos) &&
          ((size_t)colwertor->pStack[colwertor->stack_pos].index > colwertor->pDesc->desc.length)) ) {
        /* only if the colwertor is completely initialized */
        opal_output( 0, "Actual stack representation\n" );
        opal_datatype_dump_stack( colwertor->pStack, colwertor->stack_pos,
                                  colwertor->pDesc->desc.desc, colwertor->pDesc->name );
    }
}


void opal_datatype_dump_stack( const dt_stack_t* pStack, int stack_pos,
                               const union dt_elem_desc* pDesc, const char* name )
{
    opal_output( 0, "\nStack %p stack_pos %d name %s\n", (void*)pStack, stack_pos, name );
    for( ; stack_pos >= 0; stack_pos-- ) {
        opal_output( 0, "%d: pos %d count %" PRIsize_t " disp %ld ", stack_pos, pStack[stack_pos].index,
                     pStack[stack_pos].count, pStack[stack_pos].disp );
        if( pStack->index != -1 )
            opal_output( 0, "\t[desc count %lu disp %ld extent %ld]\n",
                         (unsigned long)pDesc[pStack[stack_pos].index].elem.count,
                         (long)pDesc[pStack[stack_pos].index].elem.disp,
                         (long)pDesc[pStack[stack_pos].index].elem.extent );
        else
            opal_output( 0, "\n" );
    }
    opal_output( 0, "\n" );
}
