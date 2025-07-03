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
 *  Module name              : stdAVL.h
 *
 *  Description              :
 *     
 *         This module defines an AVL tree over abstract elements.
 */

#ifndef stdAVL_INCLUDED
#define stdAVL_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"
#include "stdList.h"
#include "stdStdFun.h"
#include "stdWriter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct stdAVLRec  *stdAVL_t;

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : stdAVL creation macro, a shorthand for function avlCreate.
 * Parameters      : type         (I) Name of a type for which functions
 *                                    typeHash, typeEqual and typeLessEq are in scope.
 * Function Result : Requested avl.
 **/
#define   avlXNEW(type)  avlCreate(                    \
                           (stdHashFun  )type##Hash,   \
                           (stdLessEqFun)type##LessEq)

/*
 * Function        : stdAVL creation macro, a shorthand for function avlCreate.
 * Parameters      : type         (I) Name of standard type.
 * Function Result : Requested avl.
 **/
#define   avlNEW(type)   avlXNEW(std##type)

/*
 * Function        : Create new avl.
 * Parameters      : hash         (I) Hash function, mapping the avl element
 *                                    type to an arbitrary integer.
 *                   lesseq       (I) Comparison function for element type.
 * Function Result : Requested avl.
 */
stdAVL_t STD_CDECL avlCreate( stdHashFun hash, stdLessEqFun lesseq);


/*
 * Function        : Create new (empty) avl with parameters identical to specified avl
 * Parameters      : avl          (I) Template avl.
 * Function Result : Requested avl.
 */
stdAVL_t STD_CDECL avlCreateLike( stdAVL_t avl );


/*
 * Function        : Discard avl.
 * Parameters      : avl  (I) avl tree to discard.
 * Function Result :
 */
void STD_CDECL avlDelete( stdAVL_t avl );


/*
 * Function         : Remove all elements from the avl.
 * Parameters       : avl (O) avl tree to empty.
 * Function Result  : True iff. the avl was non-empty when 
 *                    it was passed to this function.
 */
Bool STD_CDECL avlEmpty( stdAVL_t avl );


/*
 * Function        : Apply specified function to all elements in the specified avl,
 *                   with specified generic data element as additional parameter.
 *                   Traversal will be performed in low to high element order. 
 *                   The avl is not allowed to change during traversal.
 *                   Note: the special exception for the other ADTs, namely
 *                         that the current element may be removed during traversal,
 *                         does NOT hold for avl trees.
 * Parameters      : avl        (I) avl tree to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse' .
 * Function Result :
 */
void STD_CDECL avlTraverse( stdAVL_t avl, stdEltFun traverse, Pointer data );


/*
 * Function        : Test oclwrrence in avl.
 * Parameters      : avl  (I) avl tree to test.
 *                   elt  (I) Element to test for oclwrrence.
 * Function Result : The element x in the avl such that avl.equal(x,elt),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL avlElement( stdAVL_t avl, Pointer elt );


/*
 * Function        : Test oclwrrence in avl.
 * Parameters      : avl  (I) avl tree to test.
 *                   elt  (I) Element to test for oclwrrence.
 * Function Result : True if and only if elt is a member of avl.
 */
Bool STD_CDECL avlContains( stdAVL_t avl, Pointer elt );


/*
 * Function        : Insert element into avl.
 * Parameters      : avl  (I) avl tree to insert into.
 *                   elt  (I) Element to insert.
 * Function Result : The element x previously in the avl such that avl.equal(x,elt),
 *                   or Nil otherwise. Note that if such an x oclwrred, it is 
 *                   replaced by the new element, since the avl treats them as
 *                   equal.
 */
Pointer STD_CDECL avlInsert( stdAVL_t avl, Pointer elt );


/*
 * Function        : Remove element from avl.
 * Parameters      : avl  (I) avl tree to remove from.
 *                   elt  (I) Element to remove.
 * Function Result : The element x previously in the avl such that avl.equal(x,elt),
 *                   or Nil otherwise. If such an x oclwrred, it is 
 *                   removed from the avl.
 */
Pointer STD_CDECL avlRemove( stdAVL_t avl, Pointer elt );


/*
 * Function        : Add specified element to specified avl.
 * Parameters      : element  (I)  Element to add.
 *                   avl     (IO)  stdAVL to modify.
 * Function Result : 
 * NB              : This function is an analogon of avlInsert,
 *                   intended as traversal function.
 */
void STD_CDECL avlAddTo( Pointer element, stdAVL_t avl );


/*
 * Function        : Delete specified element from specified avl.
 * Parameters      : element  (I)  Element to delete.
 *                   avl     (IO)  stdAVL to modify.
 * Function Result :
 * NB              : This function is an analogon of avlRemove,
 *                   intended as traversal function.
 */
void STD_CDECL avlDeleteFrom( Pointer element, stdAVL_t avl );


/*
 * Function        : Return number of elements in avl.
 * Parameters      : avl  (I) avl tree to size.
 * Function Result : Number of elements in avl.
 */
SizeT STD_CDECL avlSize( stdAVL_t avl );


/*
 * Function        : Copy a avl.
 * Parameters      : avl    (I) avl tree to copy.
 * Function Result : Copy of avl. The elt objects are not copied! .
 */
stdAVL_t STD_CDECL avlCopy( stdAVL_t avl );


/*
 * Function        : Return an arbitrary element from avl
 *                   (it is not removed).
 * Parameters      : avl  (I) avl tree to return element from.
 * Function Result : An arbitrary element from the avl, or Nil
 *                   if the avl was empty.
 */
Pointer STD_CDECL avlAnyElement( stdAVL_t avl );


/*
 * Function        : Hash value of avl.
 * Parameters      : avl  (I) avl tree to return hash value from.
 * Function Result : Hash value.
 */
uInt STD_CDECL avlHash( stdAVL_t avl );


/*
 * Function        : Compare avls for equality.
 * Parameters      : avl1  (I) avl tree1 to compare.
 *                   avl2  (I) avl tree2 to compare.
 * Function Result : True iff the specified avls contain
 *                   an equal amount of elements, that
 *                   are pairwise 'equal' according to the
 *                   equality function by which the avl
 *                   has been created.
 */
Bool STD_CDECL avlEqual( stdAVL_t avl1, stdAVL_t avl2 );


/*
 * Function        : Create a list form of the avl.
 * Parameters      : avl    (I) avl tree to colwert.
 * Function Result : Colwerted avl.
 */
stdList_t STD_CDECL avlToList( stdAVL_t avl );


/*
 * Function        : Print avl tree via writer object.
 * Parameters      : wr      (I) Writer to print to
 *                   avl     (I) avl tree to print.
 * Function Result : 
 */
void STD_CDECL avlPrint( stdWriter_t wr, stdAVL_t avl );


/*
 * Function        : Validate avl tree's internal representation.
 * Parameters      : avl     (I) avl tree to validate.
 * Function Result : 
 */
void STD_CDECL avlValidate( stdAVL_t avl );


#ifdef __cplusplus
}
#endif

#endif


#if 0
TODO:

/*
 * Function         : Check for overlap of input avls
 * Parameters       : avl1    (I) avl 1
 *                  : avl2    (I) avl 2
 * Function Result  : True iff. the specified avls do overlap
 */
Bool avlOverlaps( stdAVL_t avl1, stdAVL_t avl2 );


/*
 * Function         : Check if avl1 is included as a whole in avl2
 * Parameters       : avl1    (I) avl 1
 *                  : avl2    (I) avl 2
 * Function Result  : True iff avl1 is a subavl of avl2
 */
Bool avlSubset( stdAVL_t avl1, stdAVL_t avl2 );


/*
 * Function         : Check for overlap of input avls.
 * Parameters       : avl1    (I) avl tree 1.
 *                    avl2    (I) avl tree 2.
 * Function Result  : Indication on how the specified avls compare.
 */
stdOverlapKind avlOverlapsHow( stdAVL_t avl1, stdAVL_t avl2 );


/*
 * Function        : Unite specified avl with specified avl.
 * Parameters      : avl1  (I)  The first operand of the union.
 *                   avl2  (I)  The second operand of the union.
 * Function Result : The result of the union.
 */
stdAVL_t avlUnion( stdAVL_t avl1, stdAVL_t avl2 );


/*
 * Function        : Unite specified avl with specified avl.
 * Parameters      : result (IO)  The result/first operand of the union.
 *                   avl     (I)  The second operand of the union.
 * Function Result : True iff. the first operand did change
 */
Bool avlInPlaceUnion( stdAVL_t result, stdAVL_t avl );


/*
 * Function        : Intersect specified avl with specified avl.
 * Parameters      : avl1  (I)  The first operand of the intersect.
 *                   avl2  (I)  The second operand of the intersect.
 * Function Result : The result of the intersection.
 */
stdAVL_t avlIntersection( stdAVL_t avl1, stdAVL_t avl2 );


/*
 * Function        : Intersect specified avl with specified avl.
 * Parameters      : result (IO)  The result/first operand of the intersect.
 *                   avl     (I)  The second operand of the intersect.
 * Function Result : True iff. the first operand did change
 */
Bool avlInPlaceIntersection( stdAVL_t result, stdAVL_t avl );


/*
 * Function        : Diff specified avl with specified avl.
 * Parameters      : avl1  (I)  The first operand of the difference.
 *                   avl2  (I)  The second operand of the difference.
 * Function Result : The result of the difference.
 */
stdAVL_t avlDifference( stdAVL_t avl1, stdAVL_t avl2 );


/*
 * Function        : Diff specified avl with specified avl.
 * Parameters      : result (IO)  The result/first operand of the difference.
 *                   avl     (I)  The second operand of the difference.
 * Function Result : True iff. the first operand did change
 */
Bool avlInPlaceDifference( stdAVL_t result, stdAVL_t avl );

#endif


