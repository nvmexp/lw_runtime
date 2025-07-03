/*
 *  Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 * 
 *  NOTICE TO USER: The source code, and related code and software
 *  ("Code"), is copyrighted under U.S. and international laws.  
 * 
 *  LWPU Corporation owns the copyright and any patents issued or 
 *  pending for the Code.  
 * 
 *  LWPU CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY 
 *  OF THIS CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS-IS" WITHOUT EXPRESS
 *  OR IMPLIED WARRANTY OF ANY KIND.  LWPU CORPORATION DISCLAIMS ALL
 *  WARRANTIES WITH REGARD TO THE CODE, INCLUDING NON-INFRINGEMENT, AND 
 *  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE.  IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 *  WHATSOEVER ARISING OUT OF OR IN ANY WAY RELATED TO THE USE OR
 *  PERFORMANCE OF THE CODE, INCLUDING, BUT NOT LIMITED TO, INFRINGEMENT,
 *  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 *  NEGLIGENCE OR OTHER TORTIOUS ACTION, AND WHETHER OR NOT THE 
 *  POSSIBILITY OF SUCH DAMAGES WERE KNOWN OR MADE KNOWN TO LWPU
 *  CORPORATION.
 * 
 *  Module name              : stdBitSet.c
 *
 *  Last update              :
 *
 *  Description              :
 *     
 *         This module defines an abstract data type 'bitSet'.
 *         A BitSet is a set of bits that grows as needed. The bits are
 *         indexed by non-negative integers.
 *
 */

/*--------------------------------- Includes: ------------------------------*/

#include "stdBitSet.h"
#include "stdLocal.h"

/*-------------------------------- Constants -------------------------------*/

/*
 * As a space optimization for singleton sets, the following represents values 
 * for the bitset field in the stdBitSet struct for each possible singleton bit set. 
 */
uInt singletons[] = {
         0x00000001,0x00000002,0x00000004,0x00000008,
         0x00000010,0x00000020,0x00000040,0x00000080,
         0x00000100,0x00000200,0x00000400,0x00000800,
         0x00001000,0x00002000,0x00004000,0x00008000,
         0x00010000,0x00020000,0x00040000,0x00080000,
         0x00100000,0x00200000,0x00400000,0x00800000,
         0x01000000,0x02000000,0x04000000,0x08000000,
         0x10000000,0x20000000,0x40000000,0x80000000
};

#define ST(x)      ( ((x)-singletons) < stdNELTS(singletons) )

#define ALL_ONES   ((uInt)-1)

/*--------------------------------- Types -----------------------------------*/

/*
 * Representation of a bit set 
 */
struct stdBitSet {
   Int              size;        /* Number of elements */
   uInt            *bitset;      /* actual bitset      */
   Int              low,high;    /* range of bitset    */
};


/*-------------------------------- Functions -------------------------------*/


/*
 * Function         : replace the bitset of the specified set
 *                    with an uninitialised new set of specified range
 * Parameters       : set  (IO) set to modify
 *                    low  (I)  lower bound of new range
 *                    high (I)  upper bound of new range
 * Function Result  : -
 * Sideeffects      : the old bitset has been deallocated, and
 *                    replaced with an uninitialised new one.
 */
static void create_raw ( stdBitSet_t set, Int low, Int high  )
{ 
    Int   size      = high-low;
    uInt *old_mem   = set->bitset + set->low;
    uInt *new_mem   = stdMALLOC( size * sizeof(uInt32));

    set->low        = low;
    set->high       = high;
    set->bitset     = new_mem - low;

    if (!ST(old_mem)) { stdFREE(old_mem); }
}


/*
 * Function         : lower the lower bound of the bitset by
 *                    specified amount, and initialise to zero
 * Parameters       : set    (IO) set to modify
 *                    amount (I)  nr of words to lower by
 * Function Result  : -
 */
static void prepend ( stdBitSet_t set, Int amount )
{
    Int   old_low    = set->low ;
    Int   old_high   = set->high ;
    uInt *old_bitset = set->bitset;
    uInt *old_mem    = old_bitset + old_low;
    Int   old_size   = old_high   - old_low;

    Int   new_low    = old_low - amount;
    Int   new_high   = old_high;
    Int   new_size   = new_high - new_low;
    uInt *new_mem    = stdMALLOC( new_size * sizeof(uInt32) );
    uInt *new_bitset = new_mem  - new_low;

    stdMEMCLEAR_S( &new_bitset[new_low],                                amount * sizeof(uInt32) );
    stdMEMCOPY_S ( &new_bitset[new_low+amount], &old_bitset[old_low], old_size * sizeof(uInt32) );

    set->low    = new_low;
    set->high   = new_high;
    set->bitset = new_bitset;

    if (!ST(old_mem)) { stdFREE(old_mem); }
}


/*
 * Function         : raise the upper bound of the bitset by
 *                    specified amount, and initialise to zero
 * Parameters       : set    (IO) set to modify
 *                    amount (I)  nr of words to raise by
 * Function Result  : 
 */
static void append  ( stdBitSet_t set, Int amount )
{
    Int   old_low    = set->low ;
    Int   old_high   = set->high ;
    uInt *old_bitset = set->bitset;
    uInt *old_mem    = old_bitset + old_low;
    Int   old_size   = old_high   - old_low;

    Int   new_low    = old_low;
    Int   new_high   = old_high + amount;
    Int   new_size   = new_high - new_low;
    uInt *new_mem    = stdMALLOC( new_size * sizeof(uInt32) );
    uInt *new_bitset = new_mem  - new_low;

    stdMEMCLEAR_S( &new_bitset[new_high-amount],                 amount * sizeof(uInt32) );
    stdMEMCOPY_S ( &new_bitset[new_low], &old_bitset[old_low], old_size * sizeof(uInt32) );

    set->low    = new_low;
    set->high   = new_high;
    set->bitset = new_bitset;

    if (!ST(old_mem)) { stdFREE(old_mem); }
}



/*-------------------------------- Functions -------------------------------*/


/*
 * Function         : Create new bitSet
 * Function Result  : Requested bitSet
 */
stdBitSet_t bitSetCreate( void )
{
    stdBitSet_t result;

    stdNEW(result);

    result->size   = 0;
    result->low    = 0;
    result->high   = 0;
    result->bitset = Nil;

    return result;
}


/*
 * Function         : Remove all elements from the set
 * Parameters       : set (O) set to empty
 * Function Result  : True iff. the set was non-empty when 
 *                    it was passed to this function.
 */
Bool bitSetEmpty( stdBitSet_t set)
{
    Bool result = set->size != 0;
    
    uInt *mem = set->bitset + set->low;
    if (!ST(mem)) { stdFREE(mem); }

    set->size   = 0;
    set->low    = 0;
    set->high   = 0;
    set->bitset = Nil;
    
    return result;
}


/*
 * Function         : Create new bitset from single element
 * Parameters       : i  (I) Element to create bitset from
 * Function Result  : Requested bitSet
 */
stdBitSet_t bitSetSingleton( Int i )
{
    Int index   = i / stdBITSPERINT;
    Int bitcnt  = i % stdBITSPERINT;

    stdBitSet_t result;

    stdNEW(result);

    result->size   = 1;
    result->low    = index;
    result->high   = index+1;
    result->bitset = &singletons[ bitcnt - index ];

    return result;
}


/*
 * Function         : Create new bitset initialized to range of values
 * Parameters       : low  (I) lowest value of range 
 *                    high (I) highest value of range 
 * Function Result  : Requested bitSet
 */
stdBitSet_t bitSetRange( Int low, Int high )
{
    stdBitSet_t result= bitSetCreate();

    bitSetInsertRange( result, low, high );

    return result;
}


/*
 * Function         : delete a set, free the oclwpied memory
 * Parameters       : set (O) bitset to delete
 * Function Result  : -
 */
void bitSetDelete( stdBitSet_t set)
{
    uInt *mem = set->bitset + set->low;
    if (!ST(mem)) { stdFREE(mem); }
    stdFREE( set );
}


/*
 * Function        : Apply specified function to all index of bit in the specified bitSet,
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is unspecified, but reproducible as long
 *                   as the bitSet has not changed. The bitSet is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   element is allowed to be removed from the bitSet by the traversal
 *                   function.
 * Parameters      : set        (I) bitSet to traverse
 *                   traverse   (I) function to apply to all elements
 *                   data       (I) generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'
 * Function Result : -
 */
void  bitSetTraverse( stdBitSet_t set, stdEltFun f, Pointer data )
{
    if (set->size) {
        Int low           = set->low;
        Int offset        = low * stdBITSPERINT;
        Int elements_left = bitSetSize(set);

        while (elements_left) {
            Int bs = set->bitset[low];

            while (bs != 0) { 
                uInt b= stdFirstBit32(bs);

                bs ^= (1<<b);
                elements_left--;

                f((Pointer)(Address)(b+offset),data);
            }

            low++;
            offset += stdBITSPERINT;
        }
    }
}


/*
 * Function        : Test oclwrrence in bitSet.
 * Parameters      : set     (I) bitSet to test
 *                   element (I) element of bit to test for oclwrrence
 * Function Result : True if and only if bit is a member of bitSet.
 */
Bool bitSetElement( stdBitSet_t set, uInt element )
{
    Int index   = element / stdBITSPERINT;
    Int bitcnt  = element % stdBITSPERINT;
    Int bitmask = 1 << bitcnt;

    if (set->bitset == Nil) {
       return False;
    } else 

    if (index < set->low  ) {
       return False;
    } else 

    if (index >= set->high) {
       return False;
    }

    return (bitmask & set->bitset[index]) != 0 ;
}


/*
 * Function        : Adds a bit to specified bitSet.
 * Parameters      : set     (I) bitSet to insert into
 *                   element (I) element to insert
 * Function Result : True if and only if bit was already in the set.
 */
Bool bitSetInsert( stdBitSet_t set, uInt element )
{
    Int index   = element / stdBITSPERINT;
    Int bitcnt  = element % stdBITSPERINT;
    Int bitmask = 1 << bitcnt;

    if (set->high == set->low) {
        set->size   = 1;
        set->low    = index;
        set->high   = index+1;
        set->bitset = &singletons[ bitcnt - index ];
        return True;

    } else {

        if (index < set->low  ) {
            prepend(set, set->low - index);
        } else 

        if (index >= set->high) {
            append(set, index - set->high + 1);
        }

        if ( (bitmask & set->bitset[index]) == 0 ) {
            uInt *mem = set->bitset + set->low;

            if ( ST(mem) ) {
               uInt* new_mem = stdMALLOC( sizeof(uInt32) );
               *new_mem      = set->bitset[set->low];
               set->bitset   = new_mem - set->low;
            }
            
            set->size++;
            set->bitset[index] |= bitmask;
            return False;
        } else {
            return True;
        }
    }
}


/*
 * Function        : Remove a bit from specified bitSet.
 * Parameters      : set     (I) bitSet to remove from
 *                   element (I) element of bit to remove
 * Function Result : True if and only if bit was set.
 */
Bool bitSetRemove( stdBitSet_t set, uInt element )
{
    Int index   = element / stdBITSPERINT;
    Int bitcnt  = element % stdBITSPERINT;
    Int bitmask = 1 << bitcnt;

    if (set->bitset == Nil) {
       return False;
    } else 

    if (index < set->low  ) {
       return False;
    } else 

    if (index >= set->high) {
       return False;
    }

    if ( (bitmask & set->bitset[index]) == 0 ) {
        return False;
    } else 
    if ( set->size == 1 
      && (set->bitset[index] & bitmask)
       ) {
       /* this will deal with 'singletons' sets */
        bitSetEmpty(set);
        return True;
    } else {
        set->size--;
        set->bitset[index] &= ~bitmask;
        return True;
    }
}


/*
 * Function         : Add range of values to bitset
 * Parameters       : set  (I) bitset to add to
 *                    low  (I) lowest value of range to add
 *                    high (I) highest value of range to add
 * Function Result  : -
 */
void bitSetInsertRange( stdBitSet_t set, Int low, Int high )
{
    if (high<low ) { 
        return;
    } else
 
    if (high==low) {
       /* corner case handling for leaving 'singletons' sets intact */
        bitSetInsert(set,low);
        return;
    } else {

        Int index_l   = low / stdBITSPERINT;
        Int bitcnt_l  = low % stdBITSPERINT;

        Int index_h   = high / stdBITSPERINT;
        Int bitcnt_h  = high % stdBITSPERINT;

        Int lowMask, highMask;

        if (set->high == set->low) {
            create_raw(set,index_l,index_h+1);
            stdMEMCLEAR_S(&set->bitset[index_l], (index_h-index_l+1)*sizeof(uInt32) );

        } else {

            if (index_l < set->low  ) {
                prepend(set, set->low - index_l);
            }  

            if (index_h >= set->high) {
                append(set, index_h - set->high + 1);
            }
            
            {
                uInt *mem = set->bitset + set->low;
            
                if ( ST(mem) ) {
                    uInt* new_mem = stdMALLOC( sizeof(uInt32) );
                    *new_mem      = set->bitset[set->low];
                    set->bitset   = new_mem - set->low;
                }
            }
        }


        lowMask  =  ( ALL_ONES <<  bitcnt_l                );
        highMask =  ( ALL_ONES >> (stdBITSPERINT-1-bitcnt_h) );


        if (index_l == index_h) {
            Int bs= (lowMask & highMask) & ~set->bitset[index_l];

            set->size += stdNrofBits32(bs);
            set->bitset[index_l] |= bs;

        } else {
            Int i;
            Int size= set->size;
            Int bs;

            bs   = lowMask & ~set->bitset[index_l];
            size += stdNrofBits32(bs);
            set->bitset[index_l] |= bs;

            bs   = highMask & ~set->bitset[index_h];
            size += stdNrofBits32(bs);
            set->bitset[index_h] |= bs;

            for (i= (index_l+1); i<index_h; i++) {
                bs= ~set->bitset[i];

                size += stdNrofBits32(bs);
                set->bitset[i] = ALL_ONES;
            }

            set->size= size;
        }
    }
}


/*
 * Function         : Check for overlap with range of values.
 * Parameters       : set  (I) bitset to intersect
 *                    low  (I) lowest value of range to intersect
 *                    high (I) highest value of range to intersect
 * Function Result  : True iff the input set contains common elements with value range.
 */
Bool bitSetOverlapsRange( stdBitSet_t set, Int low, Int high )
{
    Int index_l   = low / stdBITSPERINT;
    Int bitcnt_l  = low % stdBITSPERINT;

    Int index_h   = high / stdBITSPERINT;
    Int bitcnt_h  = high % stdBITSPERINT;

    Int lowMask, highMask;


    if ( high      <  low      ) { return False; }
    if ( set->high == set->low ) { return False; }

    if (index_l < set->low  ) {
        index_l  = set->low;
        bitcnt_l = 0;
    }  

    if (index_h >= set->high) {
        index_h  = set->high-1;
        bitcnt_h = sizeof(uInt32)-1;
    }


    lowMask  =  ( ALL_ONES <<  bitcnt_l                );
    highMask =  ( ALL_ONES >> (stdBITSPERINT-1-bitcnt_h) );


    if (index_l == index_h) {
        return ( (lowMask & highMask) & set->bitset[index_l] ) != 0;

    } else {
        Int i;

        if (lowMask  & set->bitset[index_l]) { return True; }
        if (highMask & set->bitset[index_h]) { return True; }
        
        for (i= (index_l+1); i<index_h; i++) {
            if (set->bitset[i]) { return True; }
        }

        return False;
    }
}


/*
 * Function         : Remove range of values from bitset
 * Parameters       : set  (I) bitset to remove from
 *                    low  (I) lowest value of range to remove
 *                    high (I) highest value of range to remove
 * Function Result  : -
 */
void bitSetRemoveRange( stdBitSet_t set, Int low, Int high )
{
    Int index_l   = low / stdBITSPERINT;
    Int bitcnt_l  = low % stdBITSPERINT;

    Int index_h   = high / stdBITSPERINT;
    Int bitcnt_h  = high % stdBITSPERINT;

    Int lowMask, highMask;


    if ( high      <  low      ) { return; }
    if ( set->high == set->low ) { return; }

    if (index_l < set->low  ) {
        index_l  = set->low;
        bitcnt_l = 0;
    }  

    if (index_h >= set->high) {
        index_h  = set->high-1;
        bitcnt_h = sizeof(uInt32)-1;
    }


    lowMask  =  ( ALL_ONES <<  bitcnt_l                );
    highMask =  ( ALL_ONES >> (stdBITSPERINT-1-bitcnt_h) );


    if (index_l == index_h) {
        Int bs= (lowMask & highMask) & set->bitset[index_l];

        if (!bs) {
           /* nothing to do */
        } else
        if (set->size == 1) {
           /* this deals with 'singletons' sets */
            bitSetEmpty(set);
        } else {
            set->size -= stdNrofBits32(bs);
            set->bitset[index_l] &= ~bs;
        }
    } else {
        Int i;
        Int size= set->size;
        Int bs;

        bs   = lowMask & set->bitset[index_l];
        size -= stdNrofBits32(bs);
        set->bitset[index_l] &= ~bs;

        bs   = highMask & set->bitset[index_h];
        size -= stdNrofBits32(bs);
        set->bitset[index_h] &= ~bs;

        for (i= (index_l+1); i<index_h; i++) {
            bs= set->bitset[i];

            size -= stdNrofBits32(bs);
            set->bitset[i] = 0;
        }

        set->size= size;
    }
}


/*
 * Function         : Add mask of values to bitset.
 * Parameters       : set     (I) Bitset to add to.
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to add
 * Function Result  :
 */
void bitSetInsertMask( stdBitSet_t set, uInt offset, uInt64 mask )
{
    while (mask) {
        uInt lo= stdFirstBit64(mask);
        uInt ub;
        
        mask += ((uInt64)1 << lo);
        
        if (mask==0) { 
            ub= stdBITSIZEOF(mask); 
        } else {
            ub= stdFirstBit64(mask);
            mask -= ((uInt64)1 << ub);
        }
        
        bitSetInsertRange(set, offset+lo, offset+ub-1);
    }
}


/*
 * Function         : Remove mask of values from bitset.
 * Parameters       : set     (I) Bitset to remove from.
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to remove
 * Function Result  :
 */
void bitSetRemoveMask( stdBitSet_t set, uInt offset, uInt64 mask )
{
    while (mask) {
        uInt lo= stdFirstBit64(mask);
        uInt ub;
        
        mask += ((uInt64)1 << lo);
        
        if (mask==0) { 
            ub= stdBITSIZEOF(mask); 
        } else {
            ub= stdFirstBit64(mask);
            mask -= ((uInt64)1 << ub);
        }
        
        bitSetRemoveRange(set, offset+lo, offset+ub-1);
    }
}


/*
 * Function         : Check for overlap with mask of values.
 * Parameters       : set     (I) bitset to intersect
 *                    offset  (I) bit position of least order bit in mask.
 *                    mask    (I) bit mask to intersect
 * Function Result  :
 */
Bool bitSetOverlapsMask( stdBitSet_t set, uInt offset, uInt64 mask )
{
    while (mask) {
        uInt lo= stdFirstBit64(mask);
        uInt ub;
        
        mask += ((uInt64)1 << lo);
        
        if (mask==0) { 
            ub= stdBITSIZEOF(mask); 
        } else {
            ub= stdFirstBit64(mask);
            mask -= ((uInt64)1 << ub);
        }
        
        if (bitSetOverlapsRange(set, offset+lo, offset+ub-1)) { return True; }
    }
    
    return False;
}



/*
 * Function        : Adds a bit to specified bitSet.
 * Parameters      : element (I)  element of bit to add
 *                   set     (IO) BitSet to modify
 * Function Result : -
 * NB              : This function is an analogon of bitSetInsert,
 *                   intended as traversal function.
 */
void bitSetAddTo( Int element, stdBitSet_t set )
{
    bitSetInsert(set,element);
}


/*
 * Function        : Delete specified element from specified bitSet.
 * Parameters      : element (I)  element of bit to delete
 *                   set     (IO) BitSet to modify
 * Function Result : -
 * NB              : This function is an analogon of bitSetRemove,
 *                   intended as traversal function.
 */
void bitSetDeleteFrom( Int element, stdBitSet_t set )
{
    bitSetRemove(set,element);
}


/*
 * Function         : Create a new set that is the union of input sets
 * Parameters       : set    (I) set 1
 *                  : set    (I) set 2
 * Function Result  : Requested bitSet
 */
stdBitSet_t bitSetUnion( stdBitSet_t set1, stdBitSet_t set2)
{
    if (set1->size == 0) {
         return bitSetCopy(set2);
    } else 
    
    if (set2->size == 0) {
         return bitSetCopy(set1);
    } else {
    
        Int i,xhigh;
        Int size = 0;
        Int low  = stdMIN(set2->low, set1->low );
        Int high = stdMAX(set2->high,set1->high);

        stdBitSet_t result= bitSetCreate();

        create_raw(result,low,high);
        stdMEMCLEAR_S( &result->bitset[low], (high-low) * sizeof(uInt32) );

        xhigh= set1->high;
        for (i = set1->low; i < xhigh; i++) {
              Int bs= set1->bitset[i];

              size += stdNrofBits32(bs);
              result->bitset[i] = bs;
        }

        xhigh= set2->high;
        for (i = set2->low; i < xhigh; i++) {
              Int bs= set2->bitset[i] & ~result->bitset[i];

              size += stdNrofBits32(bs);
              result->bitset[i] |= bs;
        }

        result->size = size;

        return result;
    }
}


/*
 * Function        : Unite specified bitSet with specified bitSet.
 * Parameters      : result  (IO) the result/first operand of the union
 *                   set     (I)  the second operand of the union
 * Function Result : True iff. the first operand did change
 */
Bool bitSetInPlaceUnion( stdBitSet_t result, stdBitSet_t set )
{
    Int i,xhigh;
    Int size = 0;
    Int low  = stdMIN(result->low, set->low );
    Int high = stdMAX(result->high,set->high);

    if (set->size == 0) {
        return False;
    } else {
        if (result->size == 1) {
            uInt *mem = result->bitset + result->low;

            if ( ST(mem) ) {
               uInt*  new_mem = stdMALLOC( sizeof(uInt32) );
               *new_mem       = result->bitset[result->low];
               result->bitset = new_mem - result->low;
            }
        }

        if (low < result->low  ) {
            prepend(result, result->low - low);
        }  

        if (high > result->high) {
           append(result, high - result->high);
        }

        xhigh= set->high;
        for (i = set->low; i < xhigh; i++) {
          Int bs= set->bitset[i] & ~result->bitset[i];

          size += stdNrofBits32(bs);
          result->bitset[i] |= bs;
        }


        result->size += size;

        return size > 0;
    }
}


/*
 * Function         : Create a new set that is the intersection of input sets
 * Parameters       : set    (I) set 1
 *                  : set    (I) set 2
 * Function Result  : Requested bitSet
 */
stdBitSet_t bitSetIntersection( stdBitSet_t set1, stdBitSet_t set2)
{
    stdBitSet_t result= bitSetCreate();

    Int        low   = stdMAX(set2->low, set1->low );
    Int        high  = stdMIN(set2->high,set1->high);

    if (  ( set1->size != 0 ) 
       && ( set2->size != 0 )
       && ( low < high   )
       ) {
        Int i;
        Int size = 0;

        create_raw(result,low,high);

        for (i = low; i < high; i++) {
              Int bs= set1->bitset[i] & set2->bitset[i];

              size += stdNrofBits32(bs);
              result->bitset[i]= bs;
        }

        result->size= size;
    }

    return result;
}


/*
 * Function        : Intersect specified bitSet set specified bitSet.
 * Parameters      : result  (IO) the result/first operand of the intersect
 *                   set     (I)  the second operand of the intersect
 * Function Result : True iff. the first operand did change
 */
Bool bitSetInPlaceIntersection( stdBitSet_t result, stdBitSet_t set )
{
    if (set->size == 0) {
        uInt oldSize = result->size;
        
        bitSetEmpty(result);
    
        return oldSize > 0;
    } else
    if (result->size == 1) {
        
        bitSetPurge(result);
        
        /* this (also) deals with 'singletons' sets */
        if (result->bitset[result->low] & set->bitset[result->low]) {
            return False;
        } else {
            bitSetEmpty(result);
            return True;
        }
    } else {
        Int i;
        Int size = 0;
        Int low  = stdMAX(result->low, set->low );
        Int high = stdMIN(result->high,set->high);

        if (low>=high) {
            stdMEMCLEAR_S( &result->bitset[result->low], sizeof(uInt32) * ( result->high - result->low ) );
        } else {

            stdMEMCLEAR_S( &result->bitset[result->low], sizeof(uInt32) * ( low - result->low   ) );
            stdMEMCLEAR_S( &result->bitset[high],        sizeof(uInt32) * ( result->high - high ) );
        }

        for (i = low; i < high; i++) {
            Int bs= set->bitset[i] & result->bitset[i];

            size += stdNrofBits32(bs);
            result->bitset[i] = bs;
        }

        if ( result->size == size ) {
            return False;
        } else {
            result->size= size;
            return True;
        }
    }
}


/*
 * Function         : Create a new set with all values in set2 removed from set1
 * Parameters       : set    (I) set 1
 *                  : set    (I) set 2
 * Function Result  : Requested bitSet
 */
stdBitSet_t bitSetDifference( stdBitSet_t set1, stdBitSet_t set2)
{
    stdBitSet_t result= bitSetCopy(set1);

    Int        low   = stdMAX(set2->low, set1->low );
    Int        high  = stdMIN(set2->high,set1->high);

    if (result->size == 1) {
        /* this deals with 'singletons' sets */
        if (  result->low >= set2->low
          &&  result->low <  set2->high
          && (result->bitset[result->low] & set2->bitset[result->low])
             ) {
            bitSetEmpty(result);
        }
    } else 
    if (  ( set2->size != 0 )
       && ( low < high   )
       ) {
        Int i;
        Int size = 0;

        for (i = low; i < high; i++) {
              Int bs= set1->bitset[i] & set2->bitset[i];

              size += stdNrofBits32(bs);
              result->bitset[i] &= ~bs;
        }

        result->size -= size;
    }

    return result;
}


/*
 * Function        : Diff specified bitSet with specified bitSet.
 * Parameters      : result  (IO) the result/first operand of the difference
 *                   set     (I)  the second operand of the difference
 * Function Result : True iff. the first operand did change
 */
Bool  bitSetInPlaceDifference( stdBitSet_t result, stdBitSet_t set )
{
    Int i;
    if (set->size == 0) {
        return False;
    } else
    if (result->size == 1) {
        
        bitSetPurge(result);
        
        /* this deals with 'singletons' sets */
        if (  result->low >= set->low
          &&  result->low <  set->high
          && (result->bitset[result->low] & set->bitset[result->low])
           ) {
            bitSetEmpty(result);
            return True;
        } else {
            return False;
        }
    } else {
        Int size = 0;
        Int low  = stdMAX(result->low, set->low );
        Int high = stdMIN(result->high,set->high);


        for (i = low; i < high; i++) {
            Int bs= set->bitset[i] & result->bitset[i];

            size += stdNrofBits32(bs);
            result->bitset[i] &= ~bs;
        }

        result->size -= size;

        return size > 0;
    }
}


/*
 * Function        : Return number of elements in bitSet
 * Parameters      : set  (I) bitSet to size
 * Function Result : number of elements in bitSet
 */
// OPTIX_HAND_EDIT: Return value changed to SizeT to match declaration in the header.
SizeT bitSetSize( stdBitSet_t set )
{
     return (SizeT)set->size;
}


/*
 * Function        : Copy a bitSet
 * Parameters      : set    (I) bitSet to copy
 * Function Result : copy of bitSet.
 */
stdBitSet_t bitSetCopy( stdBitSet_t set )
{
    stdBitSet_t result;

    bitSetPurge(set);

    result= stdCOPY(set);
   
    if (result->size <= 1) {
       /* 
        * After the purge, we have either a Nil 
        * or a singletons bitset contents, so nothing to
        * copy:
        */
    } else {
        Int   low        = result->low ;
        Int   high       = result->high ;
        Int   size       = high - low;
        uInt *old_bitset = result->bitset;
        uInt *old_mem    = old_bitset + low;
        uInt *new_mem    = stdCOPY_N(old_mem,size);

        result->bitset = new_mem - low;
    }
    
    return result;
}


/*
 * Function         : Minimize representation of specified bitset
 * Parameters       : set  (IO)  set to purge 
 * Function Result  : -
 */
void bitSetPurge( stdBitSet_t set )
{
    if (set->size == 0) {
        bitSetEmpty(set);
    } else {
        Int   old_low    = set->low;
        uInt *old_bitset = set->bitset;
        uInt *old_mem    = old_bitset + old_low;
        Int   low        = set->low;
        Int   high       = set->high;
        
                          /*
                           * Note: the following initialization will
                           *       force a purging into a 'singletons'
                           *       for sets that have only one element:
                           */
        Bool  changed    = (set->size == 1) && !ST(old_mem);

        while ( (low < high) && (old_bitset[low] == 0) ) {
           set->low = ++low;
           changed= True;
        }

        while ( (low < high) && (old_bitset[high-1] == 0) ) {
           set->high= --high;
           changed= True;
        }

        if (changed) {
           Int size = high - low;

           if (size == 1 && set->size == 1) {
               Int bitcnt  = stdFirstBit32(old_bitset[low]);
               set->bitset = &singletons[ bitcnt - low ];
           } else {
               uInt *new_mem    = stdMALLOC( size * sizeof(uInt32) );
               uInt *new_bitset = new_mem  - low;

               stdMEMCOPY_S( &new_bitset[low], &old_bitset[low], size * sizeof(uInt32) );

               set->bitset = new_bitset;
           }

           if (!ST(old_mem)) { stdFREE(old_mem); }
        }
    }
}


/*
 * Function        : Return the first bit from bitSet (it is not removed).
 * Parameters      : set  (I) bitSet to return bit from
 * Function Result : Index of the first bit from the set, or NilBit
 *                   if the bitSet was empty.
 */
Int bitSetFirst( stdBitSet_t set )
{
    Int low           = set->low;
    Int high          = set->high;
    Int offset        = low * stdBITSPERINT;

    while (low < high) {
        Int bs = set->bitset[low++];

        if (bs != 0) { 
            return offset + stdFirstBit32(bs);
        }

        offset += stdBITSPERINT;
    }

    return NilBit;
}


/*
 * Function        : Return the last bit from bitSet (it is not removed).
 * Parameters      : set  (I) bitSet to return bit from
 * Function Result : Index of the first bit from the set, or NilBit
 *                   if the bitSet was empty.
 */
Int bitSetLast( stdBitSet_t set )
{
    Int low           = set->low;
    Int high          = set->high;
    Int offset        = high * stdBITSPERINT;

    while (low < high) {
        Int bs = set->bitset[--high];
        offset -= stdBITSPERINT;

        if (bs != 0) { 
            uInt bit;
            
            do {
                bit = stdFirstBit32(bs);
                bs ^= (1<<bit);
            } while (bs != 0);

            return offset + bit;
        }

    }

    return NilBit;
}


/*
 * Function         : Compare 2 sets for inclusion
 * Parameters       : set1, set2  (I) sets to be compared
 * Function Result  : True iff all of l's elements are also in r
 */
Bool bitSetSubset( stdBitSet_t set1, stdBitSet_t set2)
{
    if (set1->size == 0) {
        return True;
    } 
    else if (set1->size > set2->size) {
        return False;
    } else {

        Int i;
        Int intersection_low  = stdMAX(set1->low,  set2->low );
        Int intersection_high = stdMIN(set1->high, set2->high);
        
        for (i = set1->low; i < intersection_low && set1->bitset[i] == 0; i++) {
            ;
        }
    if ( i != intersection_low )
        return False;
        
        for (i = intersection_high; i < set1->high && set1->bitset[i] == 0; i++) {
            ;
        }
    if ( i != set1->high )
        return False;
        
        for (i = intersection_low; i < intersection_high; i++) {
              Int bsl= set1->bitset[i];
              Int bsr= set2->bitset[i];

              if ( (bsl & bsr) != bsl ) {
                   return False;
              } 
        }
            
        return True;
    }
}


/*
 * Function        : Compare bitSets for equality.
 * Parameters      : set1, set2  (I) bitSets to compare
 * Function Result : True iff the specified bitSets contain
 *                   an equal amount of elements, that
 *                   are pairwise 'equal'.
 */
Bool  bitSetEqual( stdBitSet_t set1, stdBitSet_t set2 )
{
    if (set1 == set2) {
        return True;
    } else

    if (set1->size != set2->size) {
        return False;
    } else {

        Int i;
        Int intersection_size = 0;
        Int intersection_low  = stdMAX(set1->low,  set2->low );
        Int intersection_high = stdMIN(set1->high, set2->high);

        for (i = intersection_low; i < intersection_high; i++) {
              Int bsl= set1->bitset[i];
              Int bsr= set2->bitset[i];

              if ( bsr != bsl ) {
                   return False;
              } else {
                   intersection_size += stdNrofBits32(bsl);
              }
        }

        return (intersection_size == set1->size);
    }
}


/*
 * Function         : Check for overlap of input sets
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : True iff the input sets contain common elements
 */
Bool bitSetOverlaps( stdBitSet_t set1, stdBitSet_t set2)
{
    Int low  = stdMAX(set2->low, set1->low );
    Int high = stdMIN(set2->high,set1->high);

    if (  ( set1->size != 0 ) 
       && ( set2->size != 0 )
       && ( low < high      )
       ) {
        Int i;

        for (i = low; i < high; i++) {
            Int bs= set1->bitset[i] & set2->bitset[i];
            if (bs != 0) { return True; }
        }
    }

    return False;
}


/*
 * Function         : Check if set2 contains any bits that are not in set1
 * Parameters       : set1    (I) set 1
 *                  : set2    (I) set 2
 * Function Result  : True iff (set2-set1) is not empty
 */
Bool bitSetNewBits( stdBitSet_t set1, stdBitSet_t set2)
{
    Int low  = stdMAX(set2->low, set1->low );
    Int high = stdMIN(set2->high,set1->high);

    Int i;

    for (i = set2->low; i < low; i++) {
        Int bs= set2->bitset[i];
        if (bs != 0) { return True; }
    }

    for (i = high; i < set2->high; i++) {
        Int bs= set2->bitset[i];
        if (bs != 0) { return True; }
    }

    for (i = low; i < high; i++) {
        Int bs= set2->bitset[i] & ~set1->bitset[i];
        if (bs != 0) { return True; }
    }
    
    return False;
}


/*
 * Function        : Print set via specified writer object.
 * Parameters      : wr      (I) Writer to print to
 *                   map     (I) Set to print.
 * Function Result : 
 */
    static void printInt( uInt i, stdWriter_t wr )
    { wtrPrintf(wr,"%d ",i); }
    
void bitSetPrint( stdWriter_t wr, stdBitSet_t set )
{
    wtrPrintf(wr,"{ ");
    bitSetTraverse(set,(stdEltFun)printInt,wr);
    wtrPrintf(wr,"}\n");
}







