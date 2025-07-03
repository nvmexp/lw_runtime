/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2011-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdVector.h
 *
 *  Description              :
 *     
 *         This module defines a data type 'vector',
 *
 *         The element type of the vector is represented  
 *         by the generic type 'Pointer'. Obviously, vectors can hold 
 *         (pointers to) memory objects, but as special exception also 
 *         objects of integer type are allowed.
 *
 *         Vector is meant to be a type that can grow dynamically
 *         but index access and traversal is fast.  Deleting individual
 *         elements is not supported, unlike stdList.
 */

/*------------------------------- Includes -----------------------------------*/

#include <stdVector.h>

/*--------------------------------- Functions --------------------------------*/

#define stdArray_t   stdVector_t
#define stdArrayRec  stdVectorRec

#define arrayCreate       vectorCreate          
#define arrayDelete       vectorDelete          
#define arrayClear        vectorClear          
#define arrayAddTo        vectorAddTo           
#define arrayPop          vectorPop             
#define arrayTop          vectorTop             
#define arraySetElement   vectorSetElement      
#define arrayIndex        vectorIndex           
#define arraySize         vectorSize            
#define arrayTraverse     vectorTraverse        
#define arraySort         vectorSort        
#define arrayToArray      vectorToArray
#define arrayStripToArray vectorStripToArray

#include "stdArraySupport.inc"

/*--------------------------------- Functions --------------------------------*/

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
}



/*
 * Function        : Remove and return last element from specified array
 * Parameters      : array   (I) Array to modify.
 * Function Result : (previous) last element of array,
 *                   or Nil when array was empty
 */
Pointer STD_CDECL arrayPop( stdArray_t array )
{
    if (!array->size) {
        return Nil;
    } else {
        return array->array[--array->size];
    }
}

/*
 * Function        : Removes all elements from the vector, leaving the vector with a size of 0.
 * Parameters      : array   (I) Array to clear.
 * Function Result :
 *
 */
void STD_CDECL arrayClear( stdArray_t array )
{
    array->size = 0;
}



/*
 * Function        : Return last element from specified array
 * Parameters      : array   (I) Array to inspect.
 * Function Result : last element of array,
 *                   or Nil when array was empty
 */
Pointer STD_CDECL arrayTop( stdArray_t array )
{
    if (!array->size) {
        return Nil;
    } else {
        return array->array[array->size-1];
    }
}



/*
 * Function        : Set element at specified position in array.
 * Parameters      : array   (I) Array to modify.
 *                   index    (I) Position in array (index origin 0).
 *                   element  (I) Element to add.
 * Function Result : 
 */
void STD_CDECL arraySetElement ( stdArray_t array, SizeT index, Pointer element)
{
    if (index >= array->capacity) {
        arrayGrow (array, index);
    }

    if (index >= array->size) {
        array->size = index + 1; /* +1 cause 0 based */
    }

    array->array[index] = element;
}



/*
 * Function        : Return element at specified position in array.
 * Parameters      : array   (I) Array to inspect.
 *                   index    (I) Position in array (index origin 0).
 * Function Result : Specified element in array, 
 *                   or NULL if index >= size(array).
 */
Pointer STD_CDECL  arrayIndex( stdArray_t array, SizeT index )
{
    if (index >= array->size) {
        return NULL;
    }
  
    return array->array[index];
}



/*
 * Function        : Apply specified function to all elements of the array,
 *                   with specified generic data as additional parameter.
 *                   The array is traversed from head to tail. 
 * Parameters      : array     (I) Array to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result : 
 */
void STD_CDECL  arrayTraverse( stdArray_t  array, stdEltFun traverse, Pointer data )
{
    SizeT i;

    for (i = 0; i < array->size; i++) {
        traverse (array->array[i], data);
    }
}



/*
 * Function        : Sort the vector's elements in increasing
 *                   element order.
 * Parameters      : vector  (I) Vector to sort.
 *                   lessEq  (I) Comparison function defining some total
 *                               ordering of the vector's elements.
 */
void STD_CDECL arraySort( stdArray_t  array, stdLessEqFun lessEq )
{
    SizeT i;

   /*
    * Heapify the array:
    */
    for (i= array->size-1; i>0; i--) {
        NbubbleDown(array,i,array->size,lessEq);
    }

   /*
    * Sort the array:
    */
    for (i= array->size-1; i>0; i--) {
        stdSWAP( array->array[0], array->array[i], Pointer );
        NbubbleDown(array,0,i,lessEq);
    }
}















