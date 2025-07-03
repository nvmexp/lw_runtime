/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdRangeMap.h
 *
 *  Description              :
 *     
 *         This module defines an abstract data type that associates ranges 
 *         of integers to elements of a 'range' type. It is logically very similar
 *         to an stdMap_t( Int --> <range type>), but it has specific functions
 *         to denote domain values as integer ranges.
 *
 *         The usual rangemap operations are defined, plus a traversal procedure.
 */

#ifndef stdRangeMap_INCLUDED
#define stdRangeMap_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdTypes.h"
#include "stdStdFun.h"
#include "stdWriter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types ------------------------------------*/

typedef struct stdRangeMap     *stdRangeMap_t;
typedef uInt64                  rangemapRange_t;

typedef struct stdRange {
       rangemapRange_t start;
       rangemapRange_t length;
} rangemapDomain_t;



typedef void (STD_CDECL *rangemapDomainFun) ( rangemapDomain_t dom, Pointer data );
typedef void (STD_CDECL *rangemapPairFun)   ( rangemapDomain_t dom, Pointer ran, Pointer data );

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Create new range map.
 * Parameters      : 
 * Function Result : Requested range map.
 */
stdRangeMap_t STD_CDECL rangemapCreate(void);


/*
 * Function        : Discard range map.
 * Parameters      : rangemap  (I) Range map to discard.
 * Function Result : 
 */
void STD_CDECL  rangemapDelete( stdRangeMap_t rangemap );



/*
 * Function        : Apply specified function to all pairs in the specified range map,
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is in the order of the ranges contained by
 *                   the range map. The range map is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   pair is allowed to be removed from the range map by the traversal
 *                   function.
 * Parameters      : rangemap    (I) Range map to traverse.
 *                   traverse    (I) Function to apply to all range map pairs.
 *                   data        (I) Generic data element passed as additional
 *                                   parameter to every invocation of 'traverse'.
 * Function Result :
 */
void STD_CDECL  rangemapTraverse( stdRangeMap_t rangemap, rangemapPairFun traverse, Pointer data );


/*
 * Function        : Apply specified function to the intersection of the specified restricting range
 *                   with all ranges that have a uniform range value in the specified range map, 
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is in the order of the ranges contained by
 *                   the range map. The range map is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   pair is allowed to be removed from the range map by the traversal
 *                   function.
 * Parameters      : rangemap    (I) Range map to traverse.
 *                   restriction (I) Restricting range
 *                   traverse    (I) Function to apply to all range map pairs.
 *                   data        (I) Generic data element passed as additional
 *                                   parameter to every invocation of 'traverse'.
 * Function Result :
 */
void STD_CDECL  rangemapTraverse_R( stdRangeMap_t rangemap, rangemapDomain_t restriction, rangemapPairFun traverse, Pointer data );


/*
 * Function        : Insert range into range map, with associated range element.
 * Parameters      : rangemap  (I) Range map to insert into.
 *                   dom       (I) Range to insert.
 *                   ran       (I) Data to define for range.
 * Function Result : 
 */
void STD_CDECL rangemapDefine( stdRangeMap_t rangemap, rangemapDomain_t dom, Pointer ran );



/*
 * Function        : Remove range from the range map.
 * Parameters      : rangemap  (I) Range map to remove from.
 *                   dom       (I) Range to remove.
 * Function Result : 
 */
void STD_CDECL rangemapUndefine( stdRangeMap_t rangemap, rangemapDomain_t dom );



/*
 * Function        : Insert single value into range map, with associated range element.
 * Parameters      : rangemap  (I) Range map to insert into.
 *                   dom       (I) Domain value to insert.
 *                   ran       (I) Data to define for dom.
 * Function Result : 
 */
void STD_CDECL rangemapDefine1( stdRangeMap_t rangemap, rangemapRange_t dom, Pointer ran );



/*
 * Function        : Remove single value from the range map.
 * Parameters      : rangemap  (I) Range map to remove from.
 *                   dom       (I) Domain value to remove.
 * Function Result : 
 */
void STD_CDECL  rangemapUndefine1( stdRangeMap_t rangemap, rangemapRange_t dom );



/*
 * Function        : Get range element associated with specified integer.
 * Parameters      : rangemap  (I) Range map to inspect.
 *                   dom       (I) Integer to get association from.
 * Function Result : Requested element, or Nil.
 */
Pointer STD_CDECL  rangemapApply( stdRangeMap_t rangemap, rangemapRange_t dom );



/*
 * Function        : Test if rangemap is empty.
 * Parameters      : rangemap  (I) Range map to test.
 * Function Result : True iff. the rangemap does not have any ranges mapped
 */
Bool STD_CDECL  rangemapIsEmpty( stdRangeMap_t rangemap );



/*
 * Function        : Obtain smallest element that is mapped by the specified range map.
 * Parameters      : rangemap   (I) Range map to investigate.
 *                   result     (O) In case elements are mapped by the range map
 *                                  (that is, when this function returns True), 
 *                                  the smallest of these elements will be returned
 *                                  via this parameter. Undefined otherwise.
 * Function Result : True iff. the range map is not empty.
 */
Bool STD_CDECL rangemapSmallestMapped( stdRangeMap_t rangemap, rangemapRange_t *result );



/*
 * Function        : Obtain largest element that is mapped by the specified range map.
 * Parameters      : rangemap   (I) Range map to investigate.
 *                   result     (O) In case elements are mapped by the range map
 *                                  (that is, when this function returns True), 
 *                                  the largest of these elements will be returned
 *                                  via this parameter. Undefined otherwise.
 * Function Result : True iff. the range map is not empty.
 */
Bool STD_CDECL rangemapLargestMapped( stdRangeMap_t rangemap, rangemapRange_t *result );



static inline rangemapDomain_t STD_CDECL rangeCreate( rangemapRange_t start, rangemapRange_t length )
{
    rangemapDomain_t result;
    
    result.start = start;
    result.length= length;
    
    return result;
}


/*
 * Function        : Dump internal representation of specified rangemap to writer object.
 * Parameters      : wr         (I) Writer to print to
 *                   rangemap   (I) Range map to dump.
 */
void STD_CDECL  rangemapPrint( stdWriter_t wr, stdRangeMap_t rangemap );


/*
 * Function        : Return the representation size of the specified rangemap .
 * Parameters      : rangemap   (I) Range map to measure.
 */
SizeT STD_CDECL  rangemapRSize( stdRangeMap_t rangemap );



/*
 * Function        : Return the maximal depth of the specified rangemap .
 * Parameters      : rangemap   (I) Range map to measure.
 */
uInt STD_CDECL  rangemapDepth( stdRangeMap_t rangemap );


#ifdef __cplusplus
}
#endif

#endif
