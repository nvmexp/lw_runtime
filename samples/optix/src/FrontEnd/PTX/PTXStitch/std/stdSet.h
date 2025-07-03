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
 *  Module name              : stdSet.h
 *
 *  Description              :
 *     
 *         This module defines an abstract data type 'set',
 *         which is implemented as a hash table with a number
 *         of buckets that can be specified at creation.
 *
 *         The element type of the set is represented  
 *         by the generic type 'Pointer', but is further defined
 *         by the equality function specified when creating the set;
 *         Obviously, sets can hold (pointers to) memory objects, 
 *         but as special exception also objects of integer type are allowed.
 *
 *         Set operation performance is further defined how
 *         'well', or how uniformly its hash function spreads 
 *         the element type over the integer domain.
 *         
 *         The usual set operations are defined, plus a traversal
 *         procedure.
 */

#ifndef stdSet_INCLUDED
#define stdSet_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"
#include "stdStdFun.h"
#include "stdWriter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct stdSetRec  *stdSet_t;

/*--------------------------------- Includes ---------------------------------*/

#include "stdList.h"

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Set creation macro, a shorthand for function setCreate.
 * Parameters      : type         (I) Name of a type for which functions
 *                                    typeHash and typeEqual are in scope.
 *                   nrofBuckets  (I) Amount of buckets in underlying hash table.
 * Function Result : Requested set.
 **/
#define   setXNEW(type,nrofBuckets)  setCreate(                   \
                                       (stdHashFun )type##Hash,   \
                                       (stdEqualFun)type##Equal,  \
                                       nrofBuckets)


/*
 * Function        : Set creation macro, a shorthand for function setCreate.
 * Parameters      : type         (I) Name of standard type.
 *                   nrofBuckets  (I) Amount of buckets in underlying hash table.
 * Function Result : Requested set.
 **/
#define   setNEW(type,nrofBuckets)   setXNEW(std##type, nrofBuckets)

/*
 * Function        : Create new set.
 * Parameters      : hash         (I) Hash function, mapping the set element
 *                                    type to an arbitrary integer.
 *                   equal        (I) Equality function for element type.
 *                   nrofBuckets  (I) Amount of buckets in underlying hash table.
 * Function Result : Requested set.
 */
stdSet_t STD_CDECL  setCreate( stdHashFun hash, stdEqualFun equal, uInt nrofBuckets);



/*
 * Function        : Create new (empty) set with element equality 
 *                   and bucket size identical to specified set
 * Parameters      : set          (I) Template set.
 * Function Result : Requested set.
 */
stdSet_t STD_CDECL  setCreateLike( stdSet_t set );



/*
 * Function        : Discard set.
 * Parameters      : set  (I) Set to discard.
 * Function Result :
 */
void STD_CDECL  setDelete( stdSet_t set );



/*
 * Function         : Remove all elements from the set.
 * Parameters       : set (O) Set to empty.
 * Function Result  : True iff. the set was non-empty when 
 *                    it was passed to this function.
 */
Bool STD_CDECL  setEmpty( stdSet_t set );

/*
 * Function         : Tests if set is empty.
 * Parameters       : set (O) to test.
 * Function Result  : True iff. set is empty.
 */
Bool STD_CDECL setIsEmpty ( stdSet_t set );

/*
 * Function        : Apply specified function to all elements in the specified set,
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is unspecified, but reproducible as long
 *                   as the set has not changed. The set is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   element is allowed to be removed from the set by the traversal
 *                   function.
 * Parameters      : set        (I) Set to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse' .
 * Function Result :
 */
void STD_CDECL  setTraverse( stdSet_t set, stdEltFun traverse, Pointer data );




/*
 * Function        : Test oclwrrence in set.
 * Parameters      : set  (I) Set to test.
 *                   elt  (I) Element to test for oclwrrence.
 * Function Result : The element x in the set such that set.equal(x,elt),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL  setElement( stdSet_t set, Pointer elt );


/*
 * Function        : Test oclwrrence in set.
 * Parameters      : set  (I) Set to test.
 *                   elt  (I) Element to test for oclwrrence.
 * Function Result : True if and only if elt is a member of set.
 */
Bool STD_CDECL setContains( stdSet_t set, Pointer elt );



/*
 * Function         : Check for overlap of input sets
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : True iff. the specified sets do overlap
 */
Bool STD_CDECL setOverlaps( stdSet_t set1, stdSet_t set2 );


/*
 * Function         : Check for overlap of input sets.
 * Parameters       : set1    (I) Set 1.
 *                    set2    (I) Set 2.
 * Function Result  : Indication on how the specified sets compare.
 */
stdOverlapKind STD_CDECL setOverlapsHow( stdSet_t set1, stdSet_t set2 );

                
                
/*
 * Function         : Check if set1 is included as a whole in set2
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : True iff set1 is a subset of set2
 */
Bool STD_CDECL setSubset( stdSet_t set1, stdSet_t set2 );



/*
 * Function        : Insert element into set.
 * Parameters      : set  (I) Set to insert into.
 *                   elt  (I) Element to insert.
 * Function Result : The element x previously in the set such that set.equal(x,elt),
 *                   or Nil otherwise. Note that if such an x oclwrred, it is 
 *                   replaced by the new element, since the set treats them as
 *                   equal.
 */
Pointer STD_CDECL  setInsert( stdSet_t set, Pointer elt );



/*
 * Function        : Remove element from set.
 * Parameters      : set  (I) Set to remove from.
 *                   elt  (I) Element to remove.
 * Function Result : The element x previously in the set such that set.equal(x,elt),
 *                   or Nil otherwise. If such an x oclwrred, it is 
 *                   removed from the set.
 */
Pointer STD_CDECL  setRemove( stdSet_t set, Pointer elt );



/*
 * Function        : Add specified element to specified set.
 * Parameters      : element  (I)  Element to add.
 *                   set     (IO)  Set to modify.
 * Function Result : 
 * NB              : This function is an analogon of setInsert,
 *                   intended as traversal function.
 */
void STD_CDECL setAddTo( Pointer element, stdSet_t set );



/*
 * Function        : Delete specified element from specified set.
 * Parameters      : element  (I)  Element to delete.
 *                   set     (IO)  Set to modify.
 * Function Result :
 * NB              : This function is an analogon of setRemove,
 *                   intended as traversal function.
 */
void STD_CDECL setDeleteFrom( Pointer element, stdSet_t set );



/*
 * Function        : Unite specified set with specified set.
 * Parameters      : set1  (I)  The first operand of the union.
 *                   set2  (I)  The second operand of the union.
 * Function Result : The result of the union.
 */
stdSet_t STD_CDECL setUnion( stdSet_t set1, stdSet_t set2 );



/*
 * Function        : Unite specified set with specified set.
 * Parameters      : result (IO)  The result/first operand of the union.
 *                   set     (I)  The second operand of the union.
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL setInPlaceUnion( stdSet_t result, stdSet_t set );



/*
 * Function        : Intersect specified set with specified set.
 * Parameters      : set1  (I)  The first operand of the intersect.
 *                   set2  (I)  The second operand of the intersect.
 * Function Result : The result of the intersection.
 */
stdSet_t STD_CDECL setIntersection( stdSet_t set1, stdSet_t set2 );



/*
 * Function        : Intersect specified set with specified set.
 * Parameters      : result (IO)  The result/first operand of the intersect.
 *                   set     (I)  The second operand of the intersect.
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL setInPlaceIntersection( stdSet_t result, stdSet_t set );



/*
 * Function        : Diff specified set with specified set.
 * Parameters      : set1  (I)  The first operand of the difference.
 *                   set2  (I)  The second operand of the difference.
 * Function Result : The result of the difference.
 */
stdSet_t STD_CDECL setDifference( stdSet_t set1, stdSet_t set2 );



/*
 * Function        : Diff specified set with specified set.
 * Parameters      : result (IO)  The result/first operand of the difference.
 *                   set     (I)  The second operand of the difference.
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL setInPlaceDifference( stdSet_t result, stdSet_t set );



/*
 * Function        : Return number of elements in set.
 * Parameters      : set  (I) Set to size.
 * Function Result : Number of elements in set.
 */
SizeT STD_CDECL setSize( stdSet_t set );



/*
 * Function        : Copy a set.
 * Parameters      : set    (I) Set to copy.
 * Function Result : Copy of set. The elt objects are not copied! .
 */
stdSet_t STD_CDECL setCopy( stdSet_t set );


/*
 * Function        : Return an arbitrary element from set
 *                   (it is not removed).
 * Parameters      : set  (I) Set to return element from.
 * Function Result : An arbitrary element from the set, or Nil
 *                   if the set was empty.
 */
Pointer STD_CDECL  setAnyElement( stdSet_t set );


/*
 * Function        : Hash value of set.
 * Parameters      : set  (I) Set to return hash value from.
 * Function Result : Hash value.
 */
uInt STD_CDECL  setHash( stdSet_t set );


/*
 * Function        : Compare sets for equality.
 * Parameters      : set1  (I) Set1 to compare.
 *                   set2  (I) Set2 to compare.
 * Function Result : True iff the specified sets contain
 *                   an equal amount of elements, that
 *                   are pairwise 'equal' according to the
 *                   equality function by which the set
 *                   has been created.
 */
Bool STD_CDECL  setEqual( stdSet_t set1, stdSet_t set2 );


/*
 * Function        : Create a list form of the set.
 * Parameters      : set    (I) Set to colwert.
 * Function Result : Colwerted set.
 */
stdList_t STD_CDECL setToList( stdSet_t set );


/*
 * Function        : Get hash table parameters of specified set.
 * Parameters      : set    (I) Set to inspect.
 *                   parms  (O) Returned hash table parameters
 * Function Result : 
 */
void STD_CDECL setGetHashTableParameters( stdSet_t set, stdHashTableParameters *parms );


/*
 * Function        : Print hashing performance information via writer object.
 * Parameters      : wr      (I) Writer to print to
 *                   set     (I) Set to print.
 * Function Result : 
 */
void STD_CDECL setPrint( stdWriter_t wr, stdSet_t set );


/*--------------------------------- Iterator --------------------------------*/
// Create iterator type and functions, so can write code like:
// stdSetIterator_t it;
// FOREACH_SET_VALUE(set,it) {
//   v = setValue(it);
// }
typedef struct stdSetIteratorRec *stdSetIterator_t;

stdSetIterator_t setBegin (stdSet_t set);
Bool setAtEnd (stdSetIterator_t *it);
stdSetIterator_t setNext (stdSetIterator_t it);
Pointer setValue (stdSetIterator_t it);

#define FOREACH_SET_VALUE(set,it) \
    for (it = setBegin(set); !setAtEnd(&it); it = setNext(it))

#ifdef __cplusplus
}
#endif

#endif
