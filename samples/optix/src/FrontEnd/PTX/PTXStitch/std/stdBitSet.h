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
 *  Module name              : stdBitSet.h
 *
 *  Description              :
 *     
 *         This module defines an abstract data type 'bitSet'.
 *         A BitSet is a set of bits that grows as needed. The bits are
 *         elemented by non-negative integers.
 *
 */

#ifndef stdBitSet_INCLUDED
#define stdBitSet_INCLUDED

/*--------------------------------- Includes: ------------------------------*/

#include "stdStdFun.h"
#include "stdWriter.h"


#if     defined(__cplusplus)
extern  "C"     {
#endif  /* defined(__cplusplus) */

/*--------------------------------- Types -----------------------------------*/

typedef struct stdBitSet   *stdBitSet_t;

/*--------------------------------- Constants -------------------------------*/

#define   NilBit      -1

/*--------------------------------- Functions -------------------------------*/


/*
 * Function         : Create new bitSet.
 * Parameters       :
 * Function Result  : Requested bitSet.
 */
stdBitSet_t STD_CDECL bitSetCreate( void );


/*
 * Function         : Create new bitset from single element.
 * Parameters       : i  (I) Element to create bitset from.
 * Function Result  : Requested bitSet.
 */
stdBitSet_t STD_CDECL bitSetSingleton( Int i );


/*
 * Function         : Create new bitset initialized to range of values.
 * Parameters       : low  (I) Lowest value of range. 
 *                    high (I) Highest value of range. 
 * Function Result  : Requested bitSet.
 */
stdBitSet_t STD_CDECL bitSetRange( Int low, Int high );


/*
 * Function         : Minimize representation of specified bitset.
 * Parameters       : set  (IO)  Set to purge. 
 * Function Result  : 
 */
void STD_CDECL bitSetPurge( stdBitSet_t set );


/*
 * Function         : Delete a set, free the oclwpied memory.
 * Parameters       : set (O) Bitset to delete.
 * Function Result  : 
 */
void STD_CDECL bitSetDelete( stdBitSet_t set);


/*
 * Function        : Apply specified function to all element of bit in the specified bitSet,
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is unspecified, but reproducible as long
 *                   as the bitSet has not changed. The bitSet is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   element is allowed to be removed from the bitSet by the traversal
 *                   function.
 * Parameters      : set        (I) BitSet to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result :
 */
void STD_CDECL bitSetTraverse( stdBitSet_t set, stdEltFun traverse, Pointer data );


/*
 * Function        : Apply specified function to all [start,end) ranges of conselwtive
 *                   bits in the specified bitSet, with specified generic data element
 *                   as additional parameter.
 *                   The order of traversal is unspecified, but reproducible as long
 *                   as the bitSet has not changed. The bitSet is not allowed to change 
 *                   during traversal, with the special exception that elements in the 
 *                   'current' range may be removed from the bitSet by the traversal
 *                   function.
 * Parameters      : set        (I) bitSet to traverse
 *                   traverse   (I) function to apply to all elements
 *                   data       (I) generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'
 * Function Result : -
 */
void STD_CDECL bitSetTraverseRanges( stdBitSet_t set, stdPairFun f, Pointer data );


/*
 * Function        : Test oclwrrence in bitSet.
 * Parameters      : set     (I) BitSet to test.
 *                   element (I) Element test for oclwrrence.
 * Function Result : True if and only if bit is a member of bitSet.
 */
Bool STD_CDECL bitSetElement( stdBitSet_t set, uInt element );


/*
 * Function        : Adds a bit to specified bitSet.
 * Parameters      : set       (I) BitSet to insert into.
 *                   element   (I) Element of bit to insert.
 * Function Result : True if and only if bit was already in the set.
 */
Bool STD_CDECL bitSetInsert( stdBitSet_t set, uInt element );


/*
 * Function        : Remove a bit from specified bitSet.
 * Parameters      : set     (I) BitSet to remove from.
 *                   element (I) Element of bit to remove.
 * Function Result : True if and only if bit was set.
 */
Bool STD_CDECL bitSetRemove( stdBitSet_t set, uInt element );


/*
 * Function         : Add range of values to bitset.
 * Parameters       : set  (I) Bitset to add to.
 *                    low  (I) Lowest value of range to add.
 *                    high (I) Highest value of range to add.
 * Function Result  :
 */
void STD_CDECL bitSetInsertRange( stdBitSet_t set, Int low, Int high );


/*
 * Function         : Remove range of values from bitset.
 * Parameters       : set  (I) Bitset to remove from.
 *                    low  (I) Lowest value of range to remove.
 *                    high (I) Highest value of range to remove.
 * Function Result  :
 */
void STD_CDECL bitSetRemoveRange( stdBitSet_t set, Int low, Int high );


/*
 * Function         : Check for overlap with range of values.
 * Parameters       : set  (I) bitset to intersect
 *                    low  (I) lowest value of range to intersect
 *                    high (I) highest value of range to intersect
 * Function Result  : True iff. the specified set overlaps with the specified range.
 */
Bool STD_CDECL bitSetOverlapsRange( stdBitSet_t set, Int low, Int high );


/*
 * Function         : Check for overlap with range of values.
 * Parameters       : set  (I) bitset to check
 *                    low  (I) lowest value of range to intersect
 *                    high (I) highest value of range to intersect
 * Function Result  : Indication on how the specified set compares with the specified range.
 */
stdOverlapKind STD_CDECL bitSetOverlapsRangeHow( stdBitSet_t set, Int low, Int high );


/*
 * Function         : Add mask of values to bitset.
 * Parameters       : set     (I) Bitset to add to.
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to add
 * Function Result  :
 */
void STD_CDECL bitSetInsertMask( stdBitSet_t set, uInt offset, uInt64 mask );


/*
 * Function         : Remove mask of values from bitset.
 * Parameters       : set     (I) Bitset to remove from.
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to remove
 * Function Result  :
 */
void STD_CDECL bitSetRemoveMask( stdBitSet_t set, uInt offset, uInt64 mask );


/*
 * Function         : Create a new set that is the intersection of input set and mask of values.
 * Parameters       : set     (I) bitset to intersect
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to intersect
 * Function Result  : Requested bitSet.
 */
stdBitSet_t STD_CDECL bitSetIntersectionMask( stdBitSet_t set, uInt offset, uInt64 mask);


/*
 * Function         : Check for overlap with mask of values.
 * Parameters       : set     (I) bitset to check
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to intersect
 * Function Result  : True iff. the specified set overlaps with the specified mask.
 */
Bool STD_CDECL bitSetOverlapsMask( stdBitSet_t set, uInt offset, uInt64 mask );


/*
 * Function         : Check for overlap with mask of values.
 * Parameters       : set     (I) bitset to check
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to intersect
 * Function Result  : Indication on how the specified set compares with the specified mask.
 */
stdOverlapKind STD_CDECL bitSetOverlapsMaskHow( stdBitSet_t set, uInt offset, uInt64 mask );


/*
 * Function        : Adds a bit to specified bitSet.
 * Parameters      : element (I)  Element of bit to add.
 *                   set     (IO) BitSet to modify.
 * Function Result : 
 * NB              : This function is an analogon of bitSetInsert,
 *                   intended as traversal function.
 */
void STD_CDECL bitSetAddTo( Int element, stdBitSet_t set );


/*
 * Function        : Delete specified element from specified bitSet.
 * Parameters      : element (I)  Element of bit to delete.
 *                   set     (IO) BitSet to modify.
 * Function Result :
 * NB              : This function is an analogon of bitSetRemove,
 *                   intended as traversal function.
 */
void STD_CDECL bitSetDeleteFrom( Int element, stdBitSet_t set );


/*
 * Function         : Create a new set that is the union of input sets.
 * Parameters       : set1    (I) Set 1.
 *                    set2    (I) Set 2.
 * Function Result  : Requested bitSet.
 */
stdBitSet_t STD_CDECL bitSetUnion( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function        : Unite specified bitSet with specified bitSet.
 * Parameters      : result  (IO) The result/first operand of the union.
 *                   set     (I)  The second operand of the union.
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL bitSetInPlaceUnion( stdBitSet_t result, stdBitSet_t set );


/*
 * Function         : Create a new set that is the intersection of input sets.
 * Parameters       : set1    (I) Set 1.
 *                    set2    (I) Set 2.
 * Function Result  : Requested bitSet.
 */
stdBitSet_t STD_CDECL bitSetIntersection( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function        : Intersect specified bitSet with specified bitSet.
 * Parameters      : result  (IO) The result/first operand of the intersect.
 *                   set     (I)  The second operand of the intersect.
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL bitSetInPlaceIntersection( stdBitSet_t result, stdBitSet_t set );


/*
 * Function         : Create a new set with all values in set2 removed from set1.
 * Parameters       : set1    (I) Set 1.
 *                    set2    (I) Set 2.
 * Function Result  : Requested bitSet.
 */
stdBitSet_t STD_CDECL bitSetDifference( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function        : Diff specified bitSet with specified bitSet.
 * Parameters      : result  (IO) The result/first operand of the difference.
 *                   set     (I)  The second operand of the difference.
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL bitSetInPlaceDifference( stdBitSet_t result, stdBitSet_t set );


/*
 * Function        : Return number of elements in bitSet.
 * Parameters      : set  (I) BitSet to size.
 * Function Result : Number of elements in bitSet.
 */
SizeT STD_CDECL bitSetSize( stdBitSet_t set );


/*
 * Function        : Copy a bitSet.
 * Parameters      : set    (I) BitSet to copy.
 * Function Result : Copy of bitSet.
 */
stdBitSet_t STD_CDECL bitSetCopy( stdBitSet_t set );


/*
 * Function        : Hash value of set.
 * Parameters      : set  (I) Set to return hash value from.
 * Function Result : Hash value.
 */
uInt STD_CDECL bitSetHash( stdBitSet_t set );

/*
 * Function        : Compare bitSets for equality.
 * Parameters      : set1     (I) Set1 to compare.
 *                   set2     (I) Set2 to compare.
 * Function Result : True iff the specified bitSets contain
 *                   an equal amount of elements, that
 *                   are pairwise 'equal'.
 */
Bool STD_CDECL bitSetEqual( stdBitSet_t set1, stdBitSet_t set2 );


/*
 * Function        : Return the first/last bit from bitSet (it is not removed).
 * Parameters      : set  (I) BitSet to return bit from.
 * Function Result : Index of the first bit from the set, or NilBit
 *                   if the bitSet was empty.
 */
Int STD_CDECL bitSetFirst( stdBitSet_t set );
Int STD_CDECL bitSetLast ( stdBitSet_t set );
#define  bitSetAnyElement bitSetFirst


/*
 * Function         : Compare 2 sets for inclusion.
 * Parameters       : set1     (I) Set1 to compare.
 *                    set2     (I) Set2 to compare.
 * Function Result  : True iff all of l's elements are also in r.
 */
Bool STD_CDECL bitSetSubset( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function         : Compare 2 sets for equality.
 * Parameters       : set1     (I) Set1 to compare.
 *                    set2     (I) Set2 to compare.
 * Function Result  : True iff set1 and set2 have the same elements.
 */
Bool STD_CDECL bitSetEqual( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function         : Check for overlap of input sets.
 * Parameters       : set1    (I) Set 1.
 *                    set2    (I) Set 2.
 * Function Result  : True iff. the specified sets overlap.
 */
Bool STD_CDECL bitSetOverlaps( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function         : Check for overlap of input sets.
 * Parameters       : set1    (I) Set 1.
 *                    set2    (I) Set 2.
 * Function Result  : Indication on how the specified sets compare.
 */
stdOverlapKind STD_CDECL bitSetOverlapsHow( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function         : Check if set2 contains any bits that are not in set1
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : True iff (set2-set1) is not empty
 */
Bool STD_CDECL bitSetNewBits( stdBitSet_t set1, stdBitSet_t set2);


/*
 * Function         : Remove all elements from the set.
 * Parameters       : set (O) Set to empty.
 * Function Result  : True iff. the set was non-empty when 
 *                    it was passed to this function.
 */
Bool STD_CDECL bitSetEmpty( stdBitSet_t set);


/*
 * Function        : Print set via specified writer object;
 *                   print trailing newline.
 * Parameters      : wr      (I) Writer to print to
 *                   map     (I) Set to print.
 * Function Result : 
 */
void STD_CDECL bitSetPrint( stdWriter_t wr, stdBitSet_t set );


/*
 * Function        : Print set via specified writer object;
 *                   don't print trailing newline.
 * Parameters      : wr      (I) Writer to print to
 *                   map     (I) Set to print.
 * Function Result : 
 */
void STD_CDECL bitSetPrintWN( stdWriter_t wr, stdBitSet_t set );


/*
 * Function        : Print set as hex words via specified writer object
 * Parameters      : wr      (I) Writer to print to
 *                   set     (I) Set to print.
 * Function Result :
 */
void STD_CDECL bitSetPrintWord ( stdWriter_t wr, stdBitSet_t set );



#if     defined(__cplusplus)
}
#endif  /* defined(__cplusplus) */

#endif /* stdBitSet_INCLUDED */


     
   
