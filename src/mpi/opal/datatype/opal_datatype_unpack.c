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
 * Copyright (c) 2008-2009 Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011      LWPU Corporation.  All rights reserved.
 * Copyright (c) 2013      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stddef.h>
#include <stdio.h>

#include "opal/datatype/opal_colwertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#if OPAL_ENABLE_DEBUG
#include "opal/util/output.h"

#define DO_DEBUG(INST)  if( opal_ddt_unpack_debug ) { INST }
#else
#define DO_DEBUG(INST)
#endif  /* OPAL_ENABLE_DEBUG */

#include "opal/datatype/opal_datatype_checksum.h"
#include "opal/datatype/opal_datatype_unpack.h"
#include "opal/datatype/opal_datatype_prototypes.h"

#if defined(CHECKSUM)
#define opal_unpack_general_function            opal_unpack_general_checksum
#define opal_unpack_homogeneous_contig_function opal_unpack_homogeneous_contig_checksum
#define opal_generic_simple_unpack_function     opal_generic_simple_unpack_checksum
#else
#define opal_unpack_general_function            opal_unpack_general
#define opal_unpack_homogeneous_contig_function opal_unpack_homogeneous_contig
#define opal_generic_simple_unpack_function     opal_generic_simple_unpack
#endif  /* defined(CHECKSUM) */


/**
 * This function will be used to unpack all datatypes that have the contiguous flag set.
 * Several types of datatypes match this criterion, not only the contiguous one, but
 * the ones that have gaps in the beginning and/or at the end but where the data to
 * be unpacked is contiguous. However, this function only work for homogeneous cases
 * and the datatype that are contiguous and where the extent is equal to the size are
 * taken in account directly in the opal_colwertor_unpack function (in colwertor.c) for
 * the homogeneous case.
 */
int32_t
opal_unpack_homogeneous_contig_function( opal_colwertor_t* pColw,
                                         struct iovec* iov,
                                         uint32_t* out_size,
                                         size_t* max_data )
{
    const opal_datatype_t *pData = pColw->pDesc;
    unsigned char *user_memory, *packed_buffer;
    uint32_t iov_idx, i;
    size_t remaining, initial_bytes_colwerted = pColw->bColwerted;
    dt_stack_t* stack = pColw->pStack;
    ptrdiff_t extent = pData->ub - pData->lb;

    DO_DEBUG( opal_output( 0, "unpack_homogeneous_contig( pBaseBuf %p, iov count %d )\n",
                           (void*)pColw->pBaseBuf, *out_size ); );
    if( stack[1].type != opal_datatype_uint1.id ) {
        stack[1].count *= opal_datatype_basicDatatypes[stack[1].type]->size;
        stack[1].type = opal_datatype_uint1.id;
    }

    if( (ptrdiff_t)pData->size == extent ) {
        for( iov_idx = 0; iov_idx < (*out_size); iov_idx++ ) {
            remaining = pColw->local_size - pColw->bColwerted;
            if( 0 == remaining ) break;  /* we're done this time */
            if( remaining > iov[iov_idx].iov_len )
                remaining = iov[iov_idx].iov_len;

            packed_buffer = (unsigned char*)iov[iov_idx].iov_base;
            user_memory = pColw->pBaseBuf + pData->true_lb + pColw->bColwerted;

            /* contiguous data or basic datatype with count */
            OPAL_DATATYPE_SAFEGUARD_POINTER( user_memory, remaining,
                                             pColw->pBaseBuf, pData, pColw->count );
            DO_DEBUG( opal_output( 0, "unpack contig [%d] dest %p src %p length %" PRIsize_t "\n",
                                   iov_idx, (void*)user_memory, (void*)packed_buffer, remaining ); );
            MEMCPY_CSUM( user_memory, packed_buffer, remaining, pColw );
            pColw->bColwerted += remaining; /* how much will get unpacked this time */
        }
    } else {
        for( iov_idx = 0; iov_idx < (*out_size); iov_idx++ ) {
            remaining = pColw->local_size - pColw->bColwerted;
            if( 0 == remaining ) break;  /* we're done this time */
            if( remaining > iov[iov_idx].iov_len )
                remaining = iov[iov_idx].iov_len;

            packed_buffer = (unsigned char*)iov[iov_idx].iov_base;
            user_memory = pColw->pBaseBuf + pData->true_lb + stack[0].disp + stack[1].disp;
            pColw->bColwerted += remaining; /* how much will get unpacked this time */

            for( i = 0; stack[1].count <= remaining; i++ ) {  /* partial or full data */
                OPAL_DATATYPE_SAFEGUARD_POINTER( user_memory, stack[1].count, pColw->pBaseBuf,
                                                 pData, pColw->count );
                DO_DEBUG( opal_output( 0, "unpack gaps [%d] dest %p src %p length %" PRIsize_t " [%d]\n",
                                       iov_idx, (void*)user_memory, (void*)packed_buffer, stack[1].count, i ); );
                MEMCPY_CSUM( user_memory, packed_buffer, stack[1].count, pColw );

                packed_buffer += stack[1].count;
                remaining     -= stack[1].count;

                stack[0].count--;
                stack[0].disp += extent;
                stack[1].count = pData->size;
                stack[1].disp  = 0;

                user_memory = pColw->pBaseBuf + pData->true_lb + stack[0].disp;
            }

            /* Copy the last bits */
            if( 0 != remaining ) {
                OPAL_DATATYPE_SAFEGUARD_POINTER( user_memory, remaining, pColw->pBaseBuf,
                                                 pData, pColw->count );
                DO_DEBUG( opal_output( 0, "unpack gaps [%d] dest %p src %p length %" PRIsize_t " [epilog]\n",
                                       iov_idx, (void*)user_memory, (void*)packed_buffer, remaining ); );
                MEMCPY_CSUM( user_memory, packed_buffer, remaining, pColw );
                stack[1].count -= remaining;
                stack[1].disp  += remaining;  /* keep the += in case we are copying less that the datatype size */
                assert( stack[1].count );
            }
        }
    }
    *out_size = iov_idx; /* we only reach this line after the for loop succesfully complete */
    *max_data = pColw->bColwerted - initial_bytes_colwerted;
    if( pColw->bColwerted == pColw->local_size ) pColw->flags |= COLWERTOR_COMPLETED;
    return !!(pColw->flags & COLWERTOR_COMPLETED);  /* done or not */
}

/**
 * This function handle partial types. Depending on the send operation it might happens
 * that we receive only a partial type (always predefined type). In fact the outcome is
 * that the unpack has to be done in 2 steps. As there is no way to know if the other
 * part of the datatype is already received, we need to use a trick to handle this special
 * case. The trick is to fill the missing part with some well known value, unpack the data
 * as if it was completely received, and then move into the user memory only the bytes
 * that don't match the well known value. This approach work as long as there is no need
 * for more than structural changes. They will not work for cases where we will have to
 * change the content of the data (as in all colwersions that require changing the size
 * of the exponent or mantissa).
 */
static inline void
opal_unpack_partial_datatype( opal_colwertor_t* pColwertor, dt_elem_desc_t* pElem,
                              unsigned char* partial_data,
                              ptrdiff_t start_position, size_t length,
                              unsigned char** user_buffer )
{
    char unused_byte = 0x7F, saved_data[16];
    unsigned char temporary[16], *temporary_buffer = temporary;
    unsigned char* user_data = *user_buffer + pElem->elem.disp;
    size_t count_desc = 1;
    size_t data_length = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;

    DO_DEBUG( opal_output( 0, "unpack partial data start %lu end %lu data_length %lu user %p\n"
                           "\tbColwerted %lu total_length %lu count %ld\n",
                           (unsigned long)start_position, (unsigned long)start_position + length, (unsigned long)data_length, (void*)*user_buffer,
                           (unsigned long)pColwertor->bColwerted, (unsigned long)pColwertor->local_size, pColwertor->count ); );

    /* Find a byte that is not used in the partial buffer */
 find_unused_byte:
    for(size_t i = 0; i < length; i++ ) {
        if( unused_byte == partial_data[i] ) {
            unused_byte--;
            goto find_unused_byte;
        }
    }

    /* Copy and fill the rest of the buffer with the unused byte */
    memset( temporary, unused_byte, data_length );
    MEMCPY( temporary + start_position, partial_data, length );

#if OPAL_LWDA_SUPPORT
    /* In the case where the data is being unpacked from device memory, need to
     * use the special host to device memory copy.  Note this code path was only
     * seen on large receives of noncontiguous data via buffered sends. */
    pColwertor->cbmemcpy(saved_data, user_data, data_length, pColwertor );
#else
    /* Save the content of the user memory */
    MEMCPY( saved_data, user_data, data_length );
#endif

    /* Then unpack the data into the user memory */
    UNPACK_PREDEFINED_DATATYPE( pColwertor, pElem, count_desc,
                                temporary_buffer, *user_buffer, data_length );

    /* reload the length as it is reset by the macro */
    data_length = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;

    /* For every oclwrence of the unused byte move data from the saved
     * buffer back into the user memory.
     */
#if OPAL_LWDA_SUPPORT
    /* Need to copy the modified user_data again so we can see which
     * bytes need to be colwerted back to their original values.  Note
     * this code path was only seen on large receives of noncontiguous
     * data via buffered sends. */
    {
        char resaved_data[16];
        pColwertor->cbmemcpy(resaved_data, user_data, data_length, pColwertor );
        for(size_t i = 0; i < data_length; i++ ) {
            if( unused_byte == resaved_data[i] )
                pColwertor->cbmemcpy(&user_data[i], &saved_data[i], 1, pColwertor);
        }
    }
#else
    for(size_t i = 0; i < data_length; i++ ) {
        if( unused_byte == user_data[i] )
            user_data[i] = saved_data[i];
    }
#endif
}

/* The pack/unpack functions need a cleanup. I have to create a proper interface to access
 * all basic functionalities, hence using them as basic blocks for all colwersion functions.
 *
 * But first let's make some global assumptions:
 * - a datatype (with the flag DT_DATA set) will have the contiguous flags set if and only if
 *   the data is really contiguous (extent equal with size)
 * - for the OPAL_DATATYPE_LOOP type the DT_CONTIGUOUS flag set means that the content of the loop is
 *   contiguous but with a gap in the begining or at the end.
 * - the DT_CONTIGUOUS flag for the type OPAL_DATATYPE_END_LOOP is meaningless.
 */
int32_t
opal_generic_simple_unpack_function( opal_colwertor_t* pColwertor,
                                     struct iovec* iov, uint32_t* out_size,
                                     size_t* max_data )
{
    dt_stack_t* pStack;                /* pointer to the position on the stack */
    uint32_t pos_desc;                 /* actual position in the description of the derived datatype */
    size_t count_desc;                 /* the number of items already done in the actual pos_desc */
    size_t total_unpacked = 0;         /* total size unpacked this time */
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    const opal_datatype_t *pData = pColwertor->pDesc;
    unsigned char *colw_ptr, *iov_ptr;
    size_t iov_len_local;
    uint32_t iov_count;

    DO_DEBUG( opal_output( 0, "opal_colwertor_generic_simple_unpack( %p, {%p, %lu}, %u )\n",
                           (void*)pColwertor, (void*)iov[0].iov_base, (unsigned long)iov[0].iov_len, *out_size ); );

    description = pColwertor->use_desc->desc;

    /* For the first step we have to add both displacement to the source. After in the
     * main while loop we will set back the source_base to the correct value. This is
     * due to the fact that the colwertor can stop in the middle of a data with a count
     */
    pStack     = pColwertor->pStack + pColwertor->stack_pos;
    pos_desc   = pStack->index;
    colw_ptr   = pColwertor->pBaseBuf + pStack->disp;
    count_desc = pStack->count;
    pStack--;
    pColwertor->stack_pos--;
    pElem = &(description[pos_desc]);

    DO_DEBUG( opal_output( 0, "unpack start pos_desc %d count_desc %" PRIsize_t " disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %" PRIsize_t " disp %ld\n",
                           pos_desc, count_desc, (long)(colw_ptr - pColwertor->pBaseBuf),
                           pColwertor->stack_pos, pStack->index, pStack->count, (long)(pStack->disp) ); );

    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
        iov_ptr = (unsigned char *) iov[iov_count].iov_base;
        iov_len_local = iov[iov_count].iov_len;

        if( 0 != pColwertor->partial_length ) {
            size_t element_length = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;
            size_t missing_length = element_length - pColwertor->partial_length;

            assert( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA );
            COMPUTE_CSUM( iov_ptr, missing_length, pColwertor );
            opal_unpack_partial_datatype( pColwertor, pElem,
                                          iov_ptr,
                                          pColwertor->partial_length, (size_t)(element_length - pColwertor->partial_length),
                                          &colw_ptr );
            --count_desc;
            if( 0 == count_desc ) {
                colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                pos_desc++;  /* advance to the next data */
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
            }
            iov_ptr       += missing_length;
            iov_len_local -= missing_length;
            pColwertor->partial_length = 0;  /* nothing more inside */
        }
        if( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
            if( ((size_t)pElem->elem.count * pElem->elem.blocklen) != count_desc ) {
                /* we have a partial (less than blocklen) basic datatype */
                int rc = UNPACK_PARTIAL_BLOCKLEN( pColwertor, pElem, count_desc,
                                                  iov_ptr, colw_ptr, iov_len_local );
                if( 0 == rc )  /* not done */
                    goto complete_loop;
                if( 0 == count_desc ) {
                    colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                    pos_desc++;  /* advance to the next data */
                    UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                }
            }
        }

        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* we have a basic datatype (working on full blocks) */
                UNPACK_PREDEFINED_DATATYPE( pColwertor, pElem, count_desc,
                                            iov_ptr, colw_ptr, iov_len_local );
                if( 0 != count_desc )  /* completed? */
                    goto complete_loop;
                colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                pos_desc++;  /* advance to the next data */
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                DO_DEBUG( opal_output( 0, "unpack end_loop count %" PRIsize_t " stack_pos %d pos_desc %d disp %ld space %lu\n",
                                       pStack->count, pColwertor->stack_pos, pos_desc,
                                       pStack->disp, (unsigned long)iov_len_local ); );
                if( --(pStack->count) == 0 ) { /* end of loop */
                    if( 0 == pColwertor->stack_pos ) {
                        /* we're done. Force the exit of the main for loop (around iovec) */
                        *out_size = iov_count;
                        goto complete_loop;
                    }
                    pColwertor->stack_pos--;
                    pStack--;
                    pos_desc++;
                } else {
                    pos_desc = pStack->index + 1;
                    if( pStack->index == -1 ) {
                        pStack->disp += (pData->ub - pData->lb);
                    } else {
                        assert( OPAL_DATATYPE_LOOP == description[pStack->index].loop.common.type );
                        pStack->disp += description[pStack->index].loop.extent;
                    }
                }
                colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                DO_DEBUG( opal_output( 0, "unpack new_loop count %" PRIsize_t " stack_pos %d pos_desc %d disp %ld space %lu\n",
                                       pStack->count, pColwertor->stack_pos, pos_desc,
                                       pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                ptrdiff_t local_disp = (ptrdiff_t)colw_ptr;
                if( pElem->loop.common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                    UNPACK_CONTIGUOUS_LOOP( pColwertor, pElem, count_desc,
                                            iov_ptr, colw_ptr, iov_len_local );
                    if( 0 == count_desc ) {  /* completed */
                        pos_desc += pElem->loop.items + 1;
                        goto update_loop_description;
                    }
                    /* Save the stack with the correct last_count value. */
                }
                local_disp = (ptrdiff_t)colw_ptr - local_disp;
                PUSH_STACK( pStack, pColwertor->stack_pos, pos_desc, OPAL_DATATYPE_LOOP, count_desc,
                            pStack->disp + local_disp);
                pos_desc++;
            update_loop_description:  /* update the current state */
                colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                DDT_DUMP_STACK( pColwertor->pStack, pColwertor->stack_pos, pElem, "advance loop" );
            }
        }
    complete_loop:
        assert( pElem->elem.common.type < OPAL_DATATYPE_MAX_PREDEFINED );
        if( 0 != iov_len_local ) {
            unsigned char* temp = colw_ptr;
            /* We have some partial data here. Let's copy it into the colwertor
             * and keep it hot until the next round.
             */
            assert( iov_len_local < opal_datatype_basicDatatypes[pElem->elem.common.type]->size );
            COMPUTE_CSUM( iov_ptr, iov_len_local, pColwertor );

            opal_unpack_partial_datatype( pColwertor, pElem,
                                          iov_ptr, 0, iov_len_local,
                                          &temp );
                
            pColwertor->partial_length = iov_len_local;
            iov_len_local = 0;
        }

        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        total_unpacked += iov[iov_count].iov_len;
    }
    *max_data = total_unpacked;
    pColwertor->bColwerted += total_unpacked;  /* update the already colwerted bytes */
    *out_size = iov_count;
    if( pColwertor->bColwerted == pColwertor->remote_size ) {
        pColwertor->flags |= COLWERTOR_COMPLETED;
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pColwertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                colw_ptr - pColwertor->pBaseBuf );
    DO_DEBUG( opal_output( 0, "unpack save stack stack_pos %d pos_desc %d count_desc %" PRIsize_t " disp %ld\n",
                           pColwertor->stack_pos, pStack->index, pStack->count, (long)pStack->disp ); );
    return 0;
}

/*
 *  Remember that the first item in the stack (ie. position 0) is the number
 * of times the datatype is ilwolved in the operation (ie. the count argument
 * in the MPI_ call).
 */
/* Colwert data from multiple input buffers (as received from the network layer)
 * to a contiguous output buffer with a predefined size.
 * return OPAL_SUCCESS if everything went OK and if there is still room before the complete
 *          colwersion of the data (need additional call with others input buffers )
 *        1 if everything went fine and the data was completly colwerted
 *       -1 something wrong oclwrs.
 */
int32_t
opal_unpack_general_function( opal_colwertor_t* pColwertor,
                              struct iovec* iov, uint32_t* out_size,
                              size_t* max_data )
{
    dt_stack_t* pStack;                /* pointer to the position on the stack */
    uint32_t pos_desc;                 /* actual position in the description of the derived datatype */
    size_t count_desc;                 /* the number of items already done in the actual pos_desc */
    uint16_t type = OPAL_DATATYPE_MAX_PREDEFINED; /* type at current position */
    size_t total_unpacked = 0;         /* total size unpacked this time */
    dt_elem_desc_t* description;
    dt_elem_desc_t* pElem;
    const opal_datatype_t *pData = pColwertor->pDesc;
    unsigned char *colw_ptr, *iov_ptr;
    uint32_t iov_count;
    size_t iov_len_local;

    const opal_colwertor_master_t* master = pColwertor->master;
    ptrdiff_t advance;       /* number of bytes that we should advance the buffer */
    size_t rc;

    DO_DEBUG( opal_output( 0, "opal_colwertor_general_unpack( %p, {%p, %lu}, %d )\n",
                           (void*)pColwertor, (void*)iov[0].iov_base, (unsigned long)iov[0].iov_len, *out_size ); );

    description = pColwertor->use_desc->desc;

    /* For the first step we have to add both displacement to the source. After in the
     * main while loop we will set back the source_base to the correct value. This is
     * due to the fact that the colwertor can stop in the middle of a data with a count
     */
    pStack     = pColwertor->pStack + pColwertor->stack_pos;
    pos_desc   = pStack->index;
    colw_ptr   = pColwertor->pBaseBuf + pStack->disp;
    count_desc = pStack->count;
    pStack--;
    pColwertor->stack_pos--;
    pElem = &(description[pos_desc]);

    DO_DEBUG( opal_output( 0, "unpack start pos_desc %d count_desc %" PRIsize_t " disp %ld\n"
                           "stack_pos %d pos_desc %d count_desc %" PRIsize_t " disp %ld\n",
                           pos_desc, count_desc, (long)(colw_ptr - pColwertor->pBaseBuf),
                           pColwertor->stack_pos, pStack->index, pStack->count, (long)(pStack->disp) ); );

    for( iov_count = 0; iov_count < (*out_size); iov_count++ ) {
        iov_ptr = (unsigned char *) iov[iov_count].iov_base;
        iov_len_local = iov[iov_count].iov_len;
        assert( 0 == pColwertor->partial_length );
        while( 1 ) {
            while( pElem->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
                /* now here we have a basic datatype */
                type = description[pos_desc].elem.common.type;
                OPAL_DATATYPE_SAFEGUARD_POINTER( colw_ptr + pElem->elem.disp, pData->size, pColwertor->pBaseBuf,
                                                 pData, pColwertor->count );
                DO_DEBUG( opal_output( 0, "unpack (%p, %ld) -> (%p:%ld, %" PRIsize_t ", %ld) type %s\n",
                                       (void*)iov_ptr, iov_len_local,
                                       (void*)pColwertor->pBaseBuf, colw_ptr + pElem->elem.disp - pColwertor->pBaseBuf,
                                       count_desc, description[pos_desc].elem.extent,
                                       opal_datatype_basicDatatypes[type]->name ); );
                rc = master->pFunctions[type]( pColwertor, count_desc,
                                               iov_ptr, iov_len_local, opal_datatype_basicDatatypes[type]->size,
                                               colw_ptr + pElem->elem.disp,
                                               (pColwertor->pDesc->ub - pColwertor->pDesc->lb) * pColwertor->count,
                                               description[pos_desc].elem.extent, &advance );
                iov_len_local -= advance;  /* decrease the available space in the buffer */
                iov_ptr += advance;        /* increase the pointer to the buffer */
                count_desc -= rc;          /* compute leftovers */
                if( 0 == count_desc ) {  /* completed */
                    colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                    pos_desc++;  /* advance to the next data */
                    UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                    if( 0 == iov_len_local ) goto complete_loop;  /* escape if we're done */
                    continue;
                }
                colw_ptr += rc * description[pos_desc].elem.extent;
                assert( pElem->elem.common.type < OPAL_DATATYPE_MAX_PREDEFINED );
                assert( 0 == iov_len_local );
                if( 0 != iov_len_local ) {
                    unsigned char* temp = colw_ptr;
                    /* We have some partial data here. Let's copy it into the colwertor
                     * and keep it hot until the next round.
                     */
                    assert( iov_len_local < opal_datatype_basicDatatypes[pElem->elem.common.type]->size );
                    COMPUTE_CSUM( iov_ptr, iov_len_local, pColwertor );

                    opal_unpack_partial_datatype( pColwertor, pElem,
                                                  iov_ptr, 0, iov_len_local,
                                                  &temp );

                    pColwertor->partial_length = iov_len_local;
                    iov_len_local = 0;
                }
                goto complete_loop;
            }
            if( OPAL_DATATYPE_END_LOOP == pElem->elem.common.type ) { /* end of the current loop */
                DO_DEBUG( opal_output( 0, "unpack end_loop count %" PRIsize_t " stack_pos %d pos_desc %d disp %ld space %lu\n",
                                       pStack->count, pColwertor->stack_pos, pos_desc,
                                       pStack->disp, (unsigned long)iov_len_local ); );
                if( --(pStack->count) == 0 ) { /* end of loop */
                    if( 0 == pColwertor->stack_pos ) {
                        /* we're done. Force the exit of the main for loop (around iovec) */
                        *out_size = iov_count;
                        goto complete_loop;
                    }
                    pColwertor->stack_pos--;
                    pStack--;
                    pos_desc++;
                } else {
                    pos_desc = pStack->index + 1;
                    if( pStack->index == -1 ) {
                        pStack->disp += (pData->ub - pData->lb);
                    } else {
                        assert( OPAL_DATATYPE_LOOP == description[pStack->index].loop.common.type );
                        pStack->disp += description[pStack->index].loop.extent;
                    }
                }
                colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                DO_DEBUG( opal_output( 0, "unpack new_loop count %" PRIsize_t " stack_pos %d pos_desc %d disp %ld space %lu\n",
                                       pStack->count, pColwertor->stack_pos, pos_desc,
                                       pStack->disp, (unsigned long)iov_len_local ); );
            }
            if( OPAL_DATATYPE_LOOP == pElem->elem.common.type ) {
                PUSH_STACK( pStack, pColwertor->stack_pos, pos_desc, OPAL_DATATYPE_LOOP, count_desc,
                            pStack->disp );
                pos_desc++;
                colw_ptr = pColwertor->pBaseBuf + pStack->disp;
                UPDATE_INTERNAL_COUNTERS( description, pos_desc, pElem, count_desc );
                DDT_DUMP_STACK( pColwertor->pStack, pColwertor->stack_pos, pElem, "advance loop" );
                continue;
            }
        }
    complete_loop:
        iov[iov_count].iov_len -= iov_len_local;  /* update the amount of valid data */
        total_unpacked += iov[iov_count].iov_len;
    }
    *max_data = total_unpacked;
    pColwertor->bColwerted += total_unpacked;  /* update the already colwerted bytes */
    *out_size = iov_count;
    if( pColwertor->bColwerted == pColwertor->remote_size ) {
        pColwertor->flags |= COLWERTOR_COMPLETED;
        return 1;
    }
    /* Save the global position for the next round */
    PUSH_STACK( pStack, pColwertor->stack_pos, pos_desc, pElem->elem.common.type, count_desc,
                colw_ptr - pColwertor->pBaseBuf );
    DO_DEBUG( opal_output( 0, "unpack save stack stack_pos %d pos_desc %d count_desc %" PRIsize_t" disp %ld\n",
                           pColwertor->stack_pos, pStack->index, pStack->count, (long)pStack->disp ); );
    return 0;
}
