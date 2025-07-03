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
 *  Module name              : stdArraySupport.inc
 *
 *  Description              :
 *     
 *         Implementation templates for stdVector and stdPriorityQueue
 */

/*------------------------------- Includes -----------------------------------*/

#include <stdLocal.h>

/*--------------------------------- Types ------------------------------------*/

struct stdArrayRec {
    Pointer       *array;
    SizeT          size;
    SizeT          capacity;
    
  #ifdef IN_PRIO_Q
    stdLessEqFun   lessEq;
  #endif
};

/*------------------------ Capacity Increase Function ------------------------*/

    static void arrayGrow (stdArray_t array, SizeT index)
    {
        SizeT max = index + 1;

        if (2 * array->capacity > max) {
            max = 2 * array->capacity;
        }

        array->array = stdREALLOC(array->array, max * sizeof(Pointer));

        stdMEMCLEAR_N(array->array + array->capacity, max - array->capacity);

        array->capacity = max;
    }

/*------------------------ 'Heap' Ilwariant Functions ------------------------*/

       /* 
        * Left and right child index
        */
        #define L(i)   ((2*(i))+1)
        #define R(i)   ((2*(i))+2)
        #define P(i)   (((i)-1)/2)

   /*
    * Smallest element at top of tree functions,
    * intended for priority queue functions:
    */
    static inline void bubbleUp( stdArray_t array, SizeT i, stdLessEqFun lessEq)
    {
        SizeT pi= P(i);

        if ( i > 0
          && !lessEq(array->array[pi], array->array[i]) 
           ) {
            stdSWAP( array->array[pi], array->array[i], Pointer );

            bubbleUp( array, pi, lessEq) ;
        }
    }


    static inline void bubbleDown( stdArray_t array, SizeT i, stdLessEqFun lessEq)
    {
        SizeT li= L(i);
        SizeT ri= R(i);

        if (li >= array->size) {
           /*
            * No children exist:
            */
            // Done
        } else 
        if (ri >= array->size) {
           /*
            * Only left child exists:
            */ 
            if ( !lessEq(array->array[i], array->array[li]) ) {
                stdSWAP( array->array[i], array->array[li], Pointer );

                bubbleDown( array, li, lessEq) ;
            }
        } else {
           /*
            * An 'only right child exists' condition
            * cannot occur for priority queues.
            * Hence, left and right child exist here:
            */
            if ( lessEq(array->array[li], array->array[ri]) ) {
                if ( !lessEq( array->array[i], array->array[li]) ) {
                     stdSWAP( array->array[i], array->array[li], Pointer );

                    bubbleDown( array, li, lessEq) ;
                }
            } else {
                if ( !lessEq( array->array[i], array->array[ri]) ) {
                     stdSWAP( array->array[i], array->array[ri], Pointer );

                    bubbleDown( array, ri, lessEq) ;
                }
            }
        }
    }

        /* ----------.---------- */

   /*
    * Largest element at top of tree functions,
    * intended for heapsort:
    */
    static inline void NbubbleDown( stdArray_t array, SizeT i, SizeT size, stdLessEqFun lessEq)
    {
        SizeT li= L(i);
        SizeT ri= R(i);

        if (li >= size) {
           /*
            * No children exist:
            */
            // Done
        } else 
        if (ri >= size) {
           /*
            * Only left child exists:
            */ 
            if ( lessEq( array->array[i], array->array[li]) ) {
                stdSWAP( array->array[i], array->array[li], Pointer );

                NbubbleDown( array, li, size, lessEq) ;
            }
        } else {
           /*
            * An 'only right child exists' condition
            * cannot occur for priority queues.
            * Hence, left and right child exist here:
            */
            if ( !lessEq(array->array[li], array->array[ri]) ) {
                if ( lessEq( array->array[i], array->array[li]) ) {
                    stdSWAP( array->array[i], array->array[li], Pointer );

                    NbubbleDown( array, li, size, lessEq) ;
                }
            } else {
                if ( lessEq( array->array[i], array->array[ri]) ) {
                    stdSWAP( array->array[i], array->array[ri], Pointer );

                    NbubbleDown( array, ri, size, lessEq) ;
                }
            }
        }
    }

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new array with specified initial size.
 * Parameters      : allocSize (I) Initial space for array.
 * Function Result : Requested (empty) array.
 */
stdArray_t STD_CDECL arrayCreate( SizeT allocSize)
{
    stdArray_t result;

    stdNEW   (result);
    stdNEW_N (result->array, allocSize);

    result->size   = 0;
    result->capacity = allocSize;

    return result;
}



/*
 * Function        : Discard array.
 * Parameters      : array  (I) Array to discard.
 * Function Result :
 */
void STD_CDECL arrayDelete( stdArray_t  array )
{
    stdFREE(array->array);
    stdFREE(array);
}



/*
 * Function        : Return number of elements in array.
 * Parameters      : array  (I) Array to inspect.
 * Function Result : Number of elements in  array.
 */
SizeT STD_CDECL arraySize( stdArray_t  array )
{
    return array->size;
}

/*
 * Function        : Return the underlying low level array representation
 *                   of the array.
 * Parameters      : array     (I) Array to query.
 * Function Result : A memory block containing the array's contents
 */
Pointer* STD_CDECL arrayToArray( stdArray_t  array )
{
    Pointer *result = stdCOPY_N(array->array, array->size);

    return result;
}

/*
 * Function        : Return the underlying low level array representation
 *                   of the array, and delete the array itself.
 * Parameters      : array     (I) Array to strip.
 * Function Result : A memory block containing the array's contents
 */
Pointer* STD_CDECL arrayStripToArray( stdArray_t  array )
{
    Pointer *result = arrayToArray(array);

    stdFREE(array);

    return result;
}



