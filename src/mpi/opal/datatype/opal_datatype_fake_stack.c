/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2017 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
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
#include <stdlib.h>

#ifdef HAVE_ALLOCA_H
#include <alloca.h>
#endif

#include "opal/datatype/opal_datatype.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/datatype/opal_datatype_internal.h"


extern int opal_colwertor_create_stack_with_pos_general( opal_colwertor_t* colwertor,
                                                         size_t starting_point, const size_t* sizes );

int opal_colwertor_create_stack_with_pos_general( opal_colwertor_t* pColwertor,
                                                  size_t starting_point, const size_t* sizes )
{
    dt_stack_t* pStack;   /* pointer to the position on the stack */
    int pos_desc;         /* actual position in the description of the derived datatype */
    size_t lastLength = 0;
    const opal_datatype_t* pData = pColwertor->pDesc;
    size_t loop_length, *remoteLength, remote_size;
    size_t resting_place = starting_point;
    dt_elem_desc_t* pElems;
    size_t count;

    assert( 0 != starting_point );
    assert( pColwertor->bColwerted != starting_point );
    assert( starting_point <=(pColwertor->count * pData->size) );

    /*opal_output( 0, "Data extent %d size %d count %d total_size %d starting_point %d\n",
                 pData->ub - pData->lb, pData->size, pColwertor->count,
                 pColwertor->local_size, starting_point );*/
    pColwertor->stack_pos = 0;
    pStack = pColwertor->pStack;
    /* Fill the first position on the stack. This one correspond to the
     * last fake OPAL_DATATYPE_END_LOOP that we add to the data representation and
     * allow us to move quickly inside the datatype when we have a count.
     */
    pElems = pColwertor->use_desc->desc;

    if( (pColwertor->flags & COLWERTOR_HOMOGENEOUS) && (pData->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS) ) {
        /* Special case for contiguous datatypes */
        int32_t cnt = (int32_t)(starting_point / pData->size);
        ptrdiff_t extent = pData->ub - pData->lb;

        loop_length = GET_FIRST_NON_LOOP( pElems );
        pStack[0].disp  = pElems[loop_length].elem.disp;
        pStack[0].type  = OPAL_DATATYPE_LOOP;
        pStack[0].count = pColwertor->count - cnt;
        cnt = (int32_t)(starting_point - cnt * pData->size);  /* number of bytes after the loop */
        pStack[1].index    = 0;
        pStack[1].type     = OPAL_DATATYPE_UINT1;
        pStack[1].disp     = pStack[0].disp;
        pStack[1].count    = pData->size - cnt;

        if( (ptrdiff_t)pData->size == extent ) { /* all elements are contiguous */
            pStack[1].disp += starting_point;
        } else {  /* each is contiguous but there are gaps inbetween */
            pStack[1].disp += (pColwertor->count - pStack[0].count) * extent + cnt;
        }

        pColwertor->bColwerted = starting_point;
        pColwertor->stack_pos = 1;
        return OPAL_SUCCESS;
    }

    /* remove from the main loop all the complete datatypes */
    assert (! (pColwertor->flags & COLWERTOR_SEND));
    remote_size    = opal_colwertor_compute_remote_size( pColwertor );
    count          = starting_point / remote_size;
    resting_place -= (remote_size * count);
    pStack->count  = pColwertor->count - count;
    pStack->index  = -1;

    loop_length = GET_FIRST_NON_LOOP( pElems );
    pStack->disp = count * (pData->ub - pData->lb) + pElems[loop_length].elem.disp;

    pos_desc  = 0;
    remoteLength = (size_t*)alloca( sizeof(size_t) * (pColwertor->pDesc->loops + 1));
    remoteLength[0] = 0;  /* initial value set to ZERO */
    loop_length = 0;

    /* The only way to get out of this loop is when we reach the desired position or
     * when we finish the whole datatype.
     */
    while( pos_desc < (int32_t)pColwertor->use_desc->used ) {
        if( OPAL_DATATYPE_END_LOOP == pElems->elem.common.type ) { /* end of the current loop */
            ddt_endloop_desc_t* end_loop = (ddt_endloop_desc_t*)pElems;
            ptrdiff_t extent;

            if( (loop_length * pStack->count) > resting_place ) {
                /* We will stop somewhere on this loop. To avoid moving inside the loop
                 * multiple times, we can compute the index of the loop where we will
                 * stop. Once this index is computed we can then reparse the loop once
                 * until we find the correct position.
                 */
                int32_t cnt = (int32_t)(resting_place / loop_length);
                if( pStack->index == -1 ) {
                    extent = pData->ub - pData->lb;
                } else {
                    assert( OPAL_DATATYPE_LOOP == (pElems - end_loop->items)->loop.common.type );
                    extent = ((ddt_loop_desc_t*)(pElems - end_loop->items))->extent;
                }
                pStack->count -= (cnt + 1);
                resting_place -= cnt * loop_length;
                pStack->disp += (cnt + 1) * extent;
                /* reset the remoteLength as we act as restarting the last loop */
                pos_desc -= (end_loop->items - 1);  /* go back to the first element in the loop */
                pElems -= (end_loop->items - 1);
                remoteLength[pColwertor->stack_pos] = 0;
                loop_length = 0;
                continue;
            }
            /* Not in this loop. Cleanup the stack and advance to the
             * next data description.
             */
            resting_place -= (loop_length * (pStack->count - 1));  /* update the resting place */
            pStack--;
            pColwertor->stack_pos--;
            pos_desc++;
            pElems++;
            remoteLength[pColwertor->stack_pos] += (loop_length * pStack->count);
            loop_length = remoteLength[pColwertor->stack_pos];
            continue;
        }
        if( OPAL_DATATYPE_LOOP == pElems->elem.common.type ) {
            remoteLength[pColwertor->stack_pos] += loop_length;
            PUSH_STACK( pStack, pColwertor->stack_pos, pos_desc, OPAL_DATATYPE_LOOP,
                        pElems->loop.loops, pStack->disp );
            pos_desc++;
            pElems++;
            remoteLength[pColwertor->stack_pos] = 0;
            loop_length = 0;  /* starting a new loop */
        }
        while( pElems->elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {
            /* now here we have a basic datatype */
            const opal_datatype_t* basic_type = BASIC_DDT_FROM_ELEM( (*pElems) );
            lastLength = (size_t)pElems->elem.count * basic_type->size;
            if( resting_place < lastLength ) {
                int32_t cnt = (int32_t)(resting_place / basic_type->size);
                loop_length += (cnt * basic_type->size);
                resting_place -= (cnt * basic_type->size);
                PUSH_STACK( pStack, pColwertor->stack_pos, pos_desc, pElems->elem.common.type,
                            pElems->elem.count - cnt,
                            pElems->elem.disp + cnt * pElems->elem.extent );
                pColwertor->bColwerted = starting_point - resting_place;
                DDT_DUMP_STACK( pColwertor->pStack, pColwertor->stack_pos,
                                pColwertor->pDesc->desc.desc, pColwertor->pDesc->name );
                return OPAL_SUCCESS;
            }
            loop_length += lastLength;
            resting_place -= lastLength;
            pos_desc++;  /* advance to the next data */
            pElems++;
        }
    }

    /* Correctly update the bColwerted field */
    pColwertor->flags |= COLWERTOR_COMPLETED;
    pColwertor->bColwerted = pColwertor->local_size;
    return OPAL_SUCCESS;
}
