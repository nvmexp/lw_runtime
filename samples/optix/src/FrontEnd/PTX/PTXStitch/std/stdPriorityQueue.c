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

/*------------------------------- Includes -----------------------------------*/

#include <stdPriorityQueue.h>

/*--------------------------------- Functions --------------------------------*/

#define stdArray_t   stdPriorityQueue_t
#define stdArrayRec  stdPriorityQueueRec

#define arrayDelete      prioqDelete          
#define arrayAddTo       prioqAddTo           
#define arrayPop         prioqPop             
#define arrayTop         prioqTop             
#define arraySize        prioqSize            
#define IN_PRIO_Q

#include "stdArraySupport.inc"

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new prioq
 *                   and specified initial size.
 * Parameters      : lessEq    (I) Priority queue's element ordering
 *                   allocSize (I) Initial space for prioq.
 * Function Result : Requested (empty) prioq.
 */
stdPriorityQueue_t STD_CDECL prioqCreate( stdLessEqFun lessEq, SizeT allocSize)
{
    stdPriorityQueue_t result = arrayCreate(allocSize);
    
    result->lessEq = lessEq;

    return result;
}


/*
 * Function        : Add specified element to end of specified array.
 * Parameters      : element  (I) Element to add.
 *                   array    (I) Array to modify.
 * Function Result : 
 */
void STD_CDECL arrayAddTo( Pointer element, stdArray_t array )
{
    if (array->size >= array->capacity) {
        arrayGrow (array, array->size);
    }
  
    array->array[array->size++] = element;
    
    bubbleUp(array,array->size-1,array->lessEq);
}



/*
 * Function        : Remove and return the smallest element 
 *                   from specified prioq
 * Parameters      : prioq   (I) PriorityQueue to modify.
 * Function Result : (previous) smallest element of prioq,
 *                   or Nil when prioq was empty
 */
Pointer STD_CDECL arrayPop( stdArray_t array )
{
    if (!array->size) {
        return Nil;
    } else {
        Pointer result = array->array[0];
        
        array->array[0] = array->array[--array->size];
        
        bubbleDown(array,0,array->lessEq);
         
        return result;
    }
}



/*
 * Function        : Return the smallest element 
 *                   from specified prioq
 * Parameters      : prioq   (I) PriorityQueue to inspect.
 * Function Result : smallest element of prioq,
 *                   or Nil when prioq was empty
 */
Pointer STD_CDECL arrayTop( stdArray_t array )
{
    if (!array->size) {
        return Nil;
    } else {
        return array->array[0];
    }
}



/*
 * Function        : Update the specified priority queue
 *                   after the ordering value of its top
 *                   element has changed
 * Parameters      : prioq   (I) PriorityQueue to modify.
 */
void STD_CDECL prioqTopChanged( stdPriorityQueue_t prioq )
{
    bubbleDown(prioq,0,prioq->lessEq);
}


