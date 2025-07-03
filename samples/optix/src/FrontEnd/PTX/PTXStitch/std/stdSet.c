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
 *  Module name              : stdSet.c
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

/*--------------------------------- Includes ---------------------------------*/

#include "stdSet.h"
#include "stdList.h"
#include "stdLocal.h"
#include "stdStdFun.h"

/*----------------------------------- Types ----------------------------------*/

typedef struct HashBlockRec {
    Pointer       key;
} HashBlockRec, *HashBlock;

/*--------------------------------- Functions --------------------------------*/

#define stdHash_t   stdSet_t
#define stdHashRec  stdSetRec

#define hashCreate                setCreate
#define hashDCreate               setDCreate
#define hashDelete                setDelete
#define hashEmpty                 setEmpty
#define getHashTableParameters    setGetHashTableParameters
#define hashPrint                 setPrint
#define hashHash                  setHash
#define hashSize                  setSize
#define hashCreateLike            setCreateLike

#define printBlock(wr,h)          wtrPrintf(wr," %p", (void*)(h->key) ); 

#define stdHashIterator_t   stdSetIterator_t
#define stdHashIteratorRec  stdSetIteratorRec
#define hashBegin           setBegin
#define hashAtEnd           setAtEnd
#define hashNext            setNext

#include "stdHashTableSupport.inc"

/*--------------------------------- Functions --------------------------------*/

static SizeT nrofCommonElements( stdSet_t set1, stdSet_t set2 )
{
    if (!set1->size || !set2->size) {
        return 0;
    } else {
        SizeT result = 0;

        stdSet_t tmp1 = set1->size < set2->size ? set1 : set2;
        stdSet_t tmp2 = set1->size < set2->size ? set2 : set1;
    
        Int i;
        
        for (i=0; i<tmp1->blocksValidCapacity; i++) {
            uInt32 valid= tmp1->blocksValid[i];
            
            while (valid) {
                HashBlock block;
                uInt      bit   = stdFirstBit32(valid);
                uInt      index = i*stdBITSIZEOF(uInt32) + bit;
                
                valid ^= (1<<bit);
                
                block= &tmp1->blocks[index];
                
                if (setContains(tmp2, block->key)) {
                    result++;
                }
            }
        }

        return result;
    }
}

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Apply specified function to all elements in the specified set,
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is unspecified, but reproducible as long
 *                   as the set has not changed. The set is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   element is allowed to be removed from the set by the traversal
 *                   function.
 * Parameters      : set        (I) set to traverse
 *                   traverse   (I) function to apply to all elements
 *                   data       (I) generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'
 * Function Result : -
 */
void STD_CDECL setTraverse ( stdSet_t set, stdEltFun traverse, Pointer data )
{
    if (set->size) {
        Int i;
        
        for (i=0; i<set->blocksValidCapacity; i++) {
            uInt32 valid= set->blocksValid[i];
            
            while (valid) {
                HashBlock block;
                uInt      bit   = stdFirstBit32(valid);
                uInt      index = i*stdBITSIZEOF(uInt32) + bit;
                
                valid ^= (1<<bit);
                
                block= &set->blocks[index];
                
                traverse(block->key,data);
            }
        }
    }
}



/*
 * Function        : Test oclwrrence in set.
 * Parameters      : set  (I) set to test
 *                   elt  (I) element to test for oclwrrence
 * Function Result : The element x in the set such that set.equal(x,elt),
 *                   or Nil otherwise
 */
Pointer STD_CDECL setElement ( stdSet_t set, Pointer elt )
{
    uInt hashValue;
    HashBlock l= lookup(set,elt,&hashValue);
    
    if (l) {
        return l->key;
    } else {
        return Nil;
    }
}


/*
 * Function        : Test oclwrrence in set.
 * Parameters      : set  (I) set to test
 *                   elt  (I) element to test for oclwrrence
 * Function Result : True if and only if elt is a member of set
 */
Bool STD_CDECL setContains ( stdSet_t set, Pointer elt )
{
    uInt hashValue;
    HashBlock l= lookup(set,elt,&hashValue);
    
    return l != Nil;
}


/*
 * Function        : Insert element into set.
 * Parameters      : set  (I) set to insert into
 *                   elt  (I) element to insert
 * Function Result : The element x previously in the set such that set.equal(x,elt),
 *                   or Nil otherwise. Note that if such an x oclwrred, it is 
 *                   replaced by the new element, since the set treats them as
 *                   equal.
 */
Pointer STD_CDECL setInsert ( stdSet_t set, Pointer elt )
{
    uInt hashValue;
    HashBlock l= lookup(set,elt,&hashValue);
    
    if (l) {
        stdSWAP(l->key,elt,Pointer);

        return elt;

    } else {
        HashBlockRec *newblock = allocBlock(set, hashValue & set->hashMask);

        newblock->key  = elt;

        set->size      += 1;
        set->hashValue ^= hashValue;

        if (set->size > set->rehashSize) {
            rehash(set);
        }

        return Nil;
    }
}




/*
 * Function        : Remove element from set.
 * Parameters      : set  (I) set to remove from
 *                   elt  (I) element to remove
 * Function Result : The element x previously in the set such that set.equal(x,elt),
 *                   or Nil otherwise. If such an x oclwrred, it is 
 *                   removed from the set.
 */
Pointer STD_CDECL setRemove ( stdSet_t set, Pointer elt )
{
    uInt32    hashValue;
    HashBlock l= lookup(set,elt,&hashValue);
    
    if (l) {
        elt = l->key;

        set->size      -= 1;
        set->hashValue ^= hashValue;

        deallocBlock(set, l, hashValue & set->hashMask);            
        
        return elt;

    } else {
        return Nil;
    }
}

                
                
                
/*
 * Function         : Check for overlap of input sets
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : True iff. the specified sets do overlap
 */
Bool STD_CDECL setOverlaps( stdSet_t set1, stdSet_t set2 )
{
    if (set1->size && set2->size) {
        stdSet_t tmp1 = set1->size < set2->size ? set1 : set2;
        stdSet_t tmp2 = set1->size < set2->size ? set2 : set1;
    
        Int i;
        
        for (i=0; i<tmp1->blocksValidCapacity; i++) {
            uInt32 valid= tmp1->blocksValid[i];
            
            while (valid) {
                HashBlock block;
                uInt      bit   = stdFirstBit32(valid);
                uInt      index = i*stdBITSIZEOF(uInt32) + bit;
                
                valid ^= (1<<bit);
                
                block= &tmp1->blocks[index];
                
                if (setContains(tmp2, block->key)) {
                    return True;
                }
            }
        }
    }

    return False;
}

                
                
                
/*
 * Function         : Check for overlap of input sets
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : Indication on how the specified sets compare.
 */
stdOverlapKind STD_CDECL setOverlapsHow( stdSet_t set1, stdSet_t set2 )
{
    SizeT inCommon = nrofCommonElements(set1,set2);

    if (inCommon == 0) { 
        return stdOverlapsNone;
    } else
    
    if ( inCommon == set1->size ) {
       if (inCommon == set2->size) { return stdOverlapsEqual; } else
                                   { return stdOverlapsLT;    }
    } else 
    
    if ( inCommon == set2->size )  { return stdOverlapsGT;    } else
                                   { return stdOverlapsSome;  }
}

                
                
/*
 * Function         : Check if set1 is included as a whole in set2
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : True iff set1 is a subset of set2
 */
Bool STD_CDECL setSubset( stdSet_t set1, stdSet_t set2 )
{
    if (set1 == set2) {
        return True;
    } else 
    if (set1->size > set2->size) {
        return False;
    } else
    if (set1->size == 0) {
        return True;
    } else {
        return nrofCommonElements(set1,set2) == set1->size;
    }
}

                
                
/*
 * Function        : Add specified element to specified set.
 * Parameters      : element  (I)  Element to add
 *                   set     (IO)  Set to modify
 * Function Result : -
 * NB              : This function is an analogon of setInsert,
 *                   intended as traversal function.
 */
void STD_CDECL setAddTo( Pointer element, stdSet_t set )
{
    setInsert(set,element);    
}



/*
 * Function        : Delete specified element from specified set.
 * Parameters      : element  (I)  Element to delete
 *                   set     (IO)  Set to modify
 * Function Result : -
 * NB              : This function is an analogon of setRemove,
 *                   intended as traversal function.
 */
void STD_CDECL setDeleteFrom( Pointer element, stdSet_t set )
{
    setRemove(set,element);    
}



/*
 * Function        : Unite specified set with specified set.
 * Parameters      : set1  (I)  the first operand of the union
 *                   set2  (I)  the second operand of the union
 * Function Result : the result of the union
 */
stdSet_t STD_CDECL setUnion( stdSet_t set1, stdSet_t set2 )
{
    stdSet_t result = setCopy(set1);

    setTraverse(set2, (stdEltFun)setAddTo, result);

    return result;
}


/*
 * Function        : Unite specified set with specified set.
 * Parameters      : result (IO)  the result/first operand of the union
 *                   set     (I)  the second operand of the union
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL setInPlaceUnion( stdSet_t result, stdSet_t set )
{
    SizeT oldSize = result->size;

    setTraverse(set, (stdEltFun)setAddTo, result);

    return result->size != oldSize;
}


/*
 * Function        : diff specified set with specified set.
 * Parameters      : set1  (I)  the first operand of the difference
 *                   set2  (I)  the second operand of the difference
 * Function Result : the result of the difference
 */
stdSet_t STD_CDECL setDifference( stdSet_t set1, stdSet_t set2 )
{
    stdSet_t result = setCopy(set1);
    
    setInPlaceDifference(result,set2);

    return result;
}


/*
 * Function        : diff specified set with specified set.
 * Parameters      : result (IO)  the result/first operand of the difference
 *                   set     (I)  the second operand of the difference
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL setInPlaceDifference( stdSet_t result, stdSet_t set )
{
    SizeT oldSize = result->size;
    
    setTraverse(set,(stdEltFun)setDeleteFrom,result);

    return result->size != oldSize;
}


/*
 * Function        : Intersect specified set with specified set.
 * Parameters      : set1  (I)  the first operand of the intersect
 *                   set2  (I)  the second operand of the intersect
 * Function Result : the result of the intersection
 */
stdSet_t STD_CDECL setIntersection( stdSet_t set1, stdSet_t set2 )
{
    stdSet_t result = setCopy(set1);

    setInPlaceIntersection(result,set2);

    return result;
}


/*
 * Function        : Intersect specified set with specified set.
 * Parameters      : result (IO)  the result/first operand of the intersect
 *                   set     (I)  the second operand of the intersect
 * Function Result : True iff. the first operand did change
 */
Bool STD_CDECL setInPlaceIntersection( stdSet_t result, stdSet_t set )
{
    SizeT oldSize = result->size;

    if (result->size) {
        Int i;
        
        for (i=0; i<result->blocksValidCapacity; i++) {
            uInt32 valid= result->blocksValid[i];
            
            while (valid) {
                HashBlock block;
                uInt      bit   = stdFirstBit32(valid);
                uInt      index = i*stdBITSIZEOF(uInt32) + bit;
                
                valid ^= (1<<bit);
                
                block= &result->blocks[index];
                
                if (!setElement(set, block->key)) {
                  setRemove(result, block->key);
                }
            }
        }
    }

    return result->size != oldSize;
}


/*
 * Function        : Return an arbitrary element from set
 *                   (it is not removed).
 * Parameters      : set  (I) set to return element from
 * Function Result : An arbitrary element from the set, or Nil
 *                   if the set was empty.
 */
Pointer STD_CDECL setAnyElement ( stdSet_t set )
{
    HashBlock l= lookupAny(set);

    if (l) { return l->key; }
      else { return Nil;   }
}


/*
 * Function        : Copy a set
 * Parameters      : set    (I) set to copy
 * Function Result : copy of set. The elt objects are not copied!
 */
stdSet_t STD_CDECL setCopy( stdSet_t set )
{
    stdSet_t newset = setCreateLike(set);

    setTraverse(set, (stdEltFun)setAddTo, newset);

    return newset;
}


/*
 * Function        : Compare sets for equality.
 * Parameters      : set1, set2  (I) sets to compare
 * Function Result : True iff the specified sets contain
 *                   an equal amount of elements, that
 *                   are pairwise 'equal' according to the
 *                   equality function by which the set
 *                   has been created.
 */
Bool STD_CDECL  setEqual( stdSet_t set1, stdSet_t set2 )
{
    if (set1 == set2) {
        return True;
    } else 
    if (set1->hashValue != set2->hashValue) {
        return False;
    } else 
    if (set1->size      != set2->size) {
        return False;
    } else {
        SizeT nrCommon = nrofCommonElements(set1,set2);

        return nrCommon == set1->size
            && nrCommon == set2->size;
    }
}


/*
 * Function        : create a list form of the set
 * Parameters      : set    (I) set to colwert
 * Function Result : colwerted set
 */
stdList_t STD_CDECL setToList( stdSet_t set )
{
    stdList_t list = Nil;

    setTraverse(set, (stdEltFun)listAddTo, &list);

    return list;
}

/*
 * Function         : Tests if set is empty.
 * Parameters       : set (O) to test.
 * Function Result  : True iff. set is empty.
 */
Bool STD_CDECL setIsEmpty( stdSet_t set )
{
    return set->size == 0;
}

/*--------------------------------- Iterator --------------------------------*/

Pointer setValue (stdSetIterator_t it)
{
  HashBlock block = hashBlockValue(it);
  if (block == NULL) return NULL;
  return block->key;
}

