/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdPriorityQueue.h
 *
 *  Description              :
 */

#ifndef stdPriorityQueue_INCLUDED
#define stdPriorityQueue_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include <stdTypes.h>
#include <stdStdFun.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types ------------------------------------*/

typedef struct stdPriorityQueueRec *stdPriorityQueue_t;

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new prioq with specified element ordering
 *                   and specified initial size.
 * Parameters      : lessEq    (I) Priority queue's element ordering
 *                   allocSize (I) Initial space for prioq.
 * Function Result : Requested (empty) prioq.
 */
stdPriorityQueue_t STD_CDECL prioqCreate( stdLessEqFun lessEq, SizeT allocSize);


/*
 * Function        : Discard prioq.
 * Parameters      : prioq  (I) PriorityQueue to discard.
 * Function Result :
 */
void STD_CDECL prioqDelete( stdPriorityQueue_t  prioq );


/*
 * Function        : Return number of elements in prioq.
 * Parameters      : prioq  (I) PriorityQueue to inspect.
 * Function Result : Number of elements in  prioq.
 */
SizeT STD_CDECL prioqSize( stdPriorityQueue_t prioq );


/*
 * Function        : Add specified element to the priority queue.
 * Parameters      : element  (I) Element to add.
 *                   prioq    (I) PriorityQueue to modify.
 * Function Result : 
 */
void STD_CDECL prioqAddTo( Pointer element, stdPriorityQueue_t prioq );
#define prioqPush(v,e) prioqAddTo(e,v)


/*
 * Function        : Remove and return the smallest element 
 *                   from specified prioq
 * Parameters      : prioq   (I) PriorityQueue to modify.
 * Function Result : (previous) smallest element of prioq,
 *                   or Nil when prioq was empty
 */
Pointer STD_CDECL prioqPop( stdPriorityQueue_t prioq );


/*
 * Function        : Return the smallest element 
 *                   from specified prioq
 * Parameters      : prioq   (I) PriorityQueue to inspect.
 * Function Result : smallest element of prioq,
 *                   or Nil when prioq was empty
 */
Pointer STD_CDECL prioqTop( stdPriorityQueue_t prioq );


/*
 * Function        : Update the specified priority queue
 *                   after the ordering value of its top
 *                   element has changed
 * Parameters      : prioq   (I) PriorityQueue to modify.
 */
void STD_CDECL prioqTopChanged( stdPriorityQueue_t prioq );



#ifdef __cplusplus
}
#endif

#endif
