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

#ifndef stdVector_INCLUDED
#define stdVector_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include <stdTypes.h>
#include <stdStdFun.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types ------------------------------------*/

typedef struct stdVectorRec *stdVector_t;

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new vector with specified initial size.
 * Parameters      : allocSize (I) Initial space for vector.
 * Function Result : Requested (empty) vector.
 */
stdVector_t STD_CDECL vectorCreate( SizeT allocSize);


/*
 * Function        : Discard vector.
 * Parameters      : vector  (I) Vector to discard.
 * Function Result :
 */
void STD_CDECL  vectorDelete( stdVector_t  vector );

/*
 * Function        : Removes all elements from the vector, leaving the vector with a size of 0.
 * Parameters      : vector (I) to clear
 * Function Result :
 */
void STD_CDECL  vectorClear( stdVector_t  vector );


/*
 * Function        : Add specified element to end of specified vector.
 * Parameters      : element  (I) Element to add.
 *                   vector   (I) Vector to modify.
 * Function Result : 
 */
void STD_CDECL vectorAddTo( Pointer element, stdVector_t vector );
#define vectorPush(v,e) vectorAddTo(e,v)



/*
 * Function        : Remove and return last element from specified vector
 * Parameters      : vector   (I) Vector to modify.
 * Function Result : (previous) last element of vector,
 *                   or Nil when vector was empty
 */
Pointer STD_CDECL vectorPop( stdVector_t vector );



/*
 * Function        : Return last element from specified vector
 * Parameters      : vector   (I) Vector to inspect.
 * Function Result : last element of vector,
 *                   or Nil when vector was empty
 */
Pointer STD_CDECL vectorTop( stdVector_t vector );


/*
 * Function        : Set element at specified position in vector.
 * Parameters      : vector   (I) Vector to modify.
 *                   index    (I) Position in vector (index origin 0).
 *                   element  (I) Element to add.
 * Function Result : 
 */
void STD_CDECL vectorSetElement ( stdVector_t vector, SizeT index, Pointer element);


/*
 * Function        : Return element at specified position in vector.
 * Parameters      : vector   (I) Vector to inspect.
 *                   index    (I) Position in vector (index origin 0).
 * Function Result : Specified element in vector, 
 *                   or NULL if index >= size(vector).
 */
Pointer STD_CDECL vectorIndex( stdVector_t vector, SizeT index );


/*
 * Function        : Return number of elements in vector.
 * Parameters      : vector  (I) Vector to inspect.
 * Function Result : Number of elements in  vector.
 */
SizeT STD_CDECL vectorSize( stdVector_t vector );


/*
 * Function        : Sort the vector's elements in increasing
 *                   element order.
 * Parameters      : vector  (I) Vector to sort.
 *                   lessEq  (I) Comparison function defining some total
 *                               ordering of the vector's elements.
 */
void STD_CDECL vectorSort( stdVector_t vector, stdLessEqFun lessEq );


/*
 * Function        : Apply specified function to all elements of the vector,
 *                   with specified generic data as additional parameter.
 *                   The vector is traversed from head to tail. 
 * Parameters      : vector     (I) Vector to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result : 
 */
void STD_CDECL  vectorTraverse( stdVector_t  vector, stdEltFun traverse, Pointer data );

/*
 * Function        : Return the underlying low level array representation
 *                   of the vector.
 * Parameters      : vector     (I) Vector to query.
 * Function Result : A memory block containing the vector's contents
 */
Pointer* STD_CDECL vectorToArray( stdVector_t  vector );

/*
 * Function        : Return the underlying low level array representation
 *                   of the vector, and delete the vector itself.
 * Parameters      : vector     (I) Vector to strip.
 * Function Result : A memory block containing the vector's contents
 */
Pointer* STD_CDECL vectorStripToArray( stdVector_t  vector );

#ifdef __cplusplus
}
#endif

#endif
