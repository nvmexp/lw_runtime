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
 *  Module name              : stdRangeMap.c
 *
 *  Description              :
 *     
 *         This module defines an abstract data type that associates ranges 
 *         of integers to elements of a 'range' type. It is logically very similar
 *         to an stdMap_t( Int --> <range type>), but it has specific functions
 *         to denote domain values as integer ranges.
 *
 *         The usual rangemap operations are defined, plus a traversal procedure.
 *
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdRangeMap.h"

/*----------------------------------- Types ----------------------------------*/

#define LOGBLOCKSIZE         4

#define BLOCKSIZE           (1<<LOGBLOCKSIZE)
#define BITS_IN_DOMAIN       stdBITSIZEOF(rangemapRange_t)
#define MAXLOGBLOCKSIZE     (stdROUNDDOWN(BITS_IN_DOMAIN-1,LOGBLOCKSIZE))

typedef struct RangeBlock *RangeBlock_t;

struct RangeBlock {
    rangemapRange_t  start;
    uInt             logIndexSize;
    Bool             isLeaf  [BLOCKSIZE];
    RangeBlock_t     contents[BLOCKSIZE];
};

struct stdRangeMap {
    RangeBlock_t    block;
};

    static inline rangemapRange_t POW2( uInt p ) __ALWAYS_INLINE__;
    static inline rangemapRange_t POW2( uInt p ) { return ((rangemapRange_t)1) << p; }

    static inline rangemapRange_t LOGMOD(rangemapRange_t x, uInt logLen)  
    {
         if (logLen >= BITS_IN_DOMAIN) {
             return x;
         } else {
             return x & (POW2(logLen)-1);
         }    
    }    

    static inline rangemapRange_t LOGDIV(rangemapRange_t x, uInt logLen)  
    {
         if (logLen >= BITS_IN_DOMAIN) {
             return 0;
         } else {
             return x >> logLen;
         }    
    }

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Create new range map.
 * Parameters      : 
 * Function Result : Requested range map.
 */
stdRangeMap_t STD_CDECL rangemapCreate(void)
{
    stdRangeMap_t result;

    stdASSERT( stdMULTIPLEOF( BITS_IN_DOMAIN, LOGBLOCKSIZE ), ("stdRangeMap configuration error") );

    stdNEW(result);

    return result;
}


/*
 * Function        : Discard range map.
 * Parameters      : rangemap  (I) Range map to discard.
 * Function Result : 
 */
    static void blockDelete( RangeBlock_t block )
    {
        Int i;

        if (block) {
            for (i=0; i<BLOCKSIZE; i++) { 
                if (!block->isLeaf[i]) { blockDelete( block->contents[i] ); }
            }

            stdFREE( block );
        }
    }

void STD_CDECL  rangemapDelete( stdRangeMap_t rangemap )
{
    blockDelete(rangemap->block);
    stdFREE(rangemap);
}


/*--------------------------------- Functions --------------------------------*/


    typedef struct {
        rangemapPairFun      traverse;
        Pointer              data;

        rangemapDomain_t     lazy;
        rangemapDomain_t    *restriction;
        Pointer              contents;
    } TraverseRec;


    static inline void flushLazy( TraverseRec *rec )
    {
        if (rec->lazy.length) {
            rec->traverse( rec->lazy, rec->contents, rec->data );
            rec->lazy.length = 0;
        }
    }

    static void blockTraverse( RangeBlock_t block, TraverseRec *rec )
    {
        rangemapRange_t bstart  = block->start;
        rangemapRange_t clength = POW2( block->logIndexSize );
        rangemapRange_t blength = BLOCKSIZE * clength;
        rangemapRange_t bend    = bstart + blength - 1;
        
        rangemapRange_t rstart,  rend;
        Int             istart,  ibound;
        
        if ( !rec->restriction
          || block->logIndexSize == MAXLOGBLOCKSIZE
           ) {
            istart = 0;    ibound = BLOCKSIZE;
            rstart = 0;    rend   = (rangemapRange_t)-1;
        } else {
            rstart = rec->restriction->start;
            rend   = rstart + rec->restriction->length - 1;

            if ( rend < bstart
              || bend < rstart
               ) {
                istart = 0;
                ibound = 0;
            } else {
                
                if (bend <= rend) {
                    ibound = BLOCKSIZE;
                } else {
                    ibound = LOGDIV( rend - bstart, block->logIndexSize) + 1;
                }
                
                if (bstart >= rstart) {
                    istart = 0;
                } else {
                    istart = LOGDIV( rstart - bstart, block->logIndexSize);
                }
            }
        }
                
        {
            Int i;
        
            for ( i=istart; i<ibound; i++ ) { 
                rangemapRange_t cstart   = bstart + i*clength;
                rangemapRange_t cend     = cstart +   clength - 1;
                Pointer         contents = block->contents[i];

                if (!contents) {
                    flushLazy(rec);

                } else
                if (block->isLeaf[i]) {
                    rangemapRange_t  xstart  = stdMAX(cstart, rstart);
                    rangemapRange_t  xend    = stdMIN(cend,   rend  );
                    rangemapRange_t  xlength = xend - xstart + 1;
                
                    if ( rec->lazy.length==0 
                      || rec->contents != contents 
                      || (rec->lazy.start + rec->lazy.length) != cstart
                       ) {
                        flushLazy(rec);
                        rec->lazy.start   = xstart;
                        rec->lazy.length  = xlength;
                        rec->contents     = contents;
                    } else {
                        rec->lazy.length += xlength;
                    }

                } else {
                    blockTraverse( contents, rec );
                }
            }
        }
    }

/*
 * Function        : Apply specified function to all ranges that have a uniform range value 
 *                   in the specified range map, with specified generic data element as 
 *                   additional parameter.
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
void STD_CDECL  rangemapTraverse( stdRangeMap_t rangemap, rangemapPairFun traverse, Pointer data )
{
    TraverseRec rec;

    rec.traverse    = traverse;
    rec.data        = data;
    rec.lazy.length = 0;
    rec.restriction = Nil;

    if (rangemap->block) { 
        blockTraverse( rangemap->block, &rec ); 
        flushLazy(&rec);
    }
}


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
void STD_CDECL  rangemapTraverse_R( stdRangeMap_t rangemap, rangemapDomain_t restriction, rangemapPairFun traverse, Pointer data )
{
    TraverseRec rec;

    rec.traverse    = traverse;
    rec.data        = data;
    rec.lazy.length = 0;
    rec.restriction = &restriction;

    if (rangemap->block) { 
        blockTraverse( rangemap->block, &rec ); 
        flushLazy(&rec);
    }
}

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Obtain smallest element that is mapped by the specified range map.
 * Parameters      : rangemap   (I) Range map to investigate.
 *                   result     (O) In case elements are mapped by the range map
 *                                  (that is, when this function returns True), 
 *                                  the smallest of these elements will be returned
 *                                  via this parameter. Undefined otherwise.
 * Function Result : True iff. the range map is not empty.
 */
    static Bool blockGetSmallest( RangeBlock_t block, rangemapRange_t *result )
    {
        Int  i;
        rangemapRange_t start  = block->start;
        rangemapRange_t length = POW2( block->logIndexSize );

        for (i=0; i<BLOCKSIZE; i++) { 
            Pointer contents= block->contents[i];

            if (contents) {
                if (block->isLeaf[i]) {
                   *result= start;
                    return True;

                } else {
                    return blockGetSmallest( contents, result );
                }
            }

            start += length;
        }
        
        stdASSERT( False, ("Range map blocks should not be completely empty") );
        return False;
    }

Bool STD_CDECL rangemapSmallestMapped( stdRangeMap_t rangemap, rangemapRange_t *result )
{
    if (rangemap->block) { 
        return blockGetSmallest( rangemap->block, result ); 
    } else {
        return False;
    }
}




/*
 * Function        : Obtain largest element that is mapped by the specified range map.
 * Parameters      : rangemap   (I) Range map to investigate.
 *                   result     (O) In case elements are mapped by the range map
 *                                  (that is, when this function returns True), 
 *                                  the largest of these elements will be returned
 *                                  via this parameter. Undefined otherwise.
 * Function Result : True iff. the range map is not empty.
 */
    static Bool blockGetLargest( RangeBlock_t block, rangemapRange_t *result )
    {
        Int  i;
        rangemapRange_t start  = block->start;
        rangemapRange_t length = POW2( block->logIndexSize );

        for (i=0; i<BLOCKSIZE; i++) { 
            Pointer contents= block->contents[i];

            if (contents) {
                if (block->isLeaf[i]) {
                   *result= start;
                    return True;

                } else {
                    return blockGetLargest( contents, result );
                }
            }

            start += length;
        }
        
        stdASSERT( False, ("Range map blocks should not be completely empty") );
        return False;
    }

Bool STD_CDECL rangemapLargestMapped( stdRangeMap_t rangemap, rangemapRange_t *result )
{
    if (rangemap->block) { 
        return blockGetLargest( rangemap->block, result ); 
    } else {
        return False;
    }
}




/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Insert range into range map, with associated range element.
 * Parameters      : rangemap  (I) Range map to insert into.
 *                   dom       (I) Range to insert.
 *                   ran       (I) Data to define for range.
 * Function Result : 
 */
 
   /*
    * Return largest multiple of LOGBLOCKSIZE <= logIndexSize such that
    * [start,length] entirely fits in a block of logIndexSize index size:
    */
    static inline uInt shrinkWrap( rangemapRange_t start, rangemapRange_t length, uInt logIndexSize )
    {
        rangemapRange_t end = start + (length - 1);

        while (logIndexSize && (LOGMOD(end,logIndexSize) - LOGMOD(start,logIndexSize)) == (length-1)) {
            logIndexSize -= LOGBLOCKSIZE;
        }

        return logIndexSize;
    }


    static inline void blockFold( RangeBlock_t *block, Bool *isLeaf, uInt logIndexSize, Bool topLevel )
    {
        RangeBlock_t b = *block; 
        
        if (!topLevel && (logIndexSize == b->logIndexSize) ) {
            uInt         i;
            RangeBlock_t contents = b->contents[0];

            for (i=1; i<BLOCKSIZE; i++) {
                if (b->contents[i] != contents) { return; }
            }

           *block  = contents;
           *isLeaf = True;

            stdFREE(b);
        }
    }


    static void blockDefine( RangeBlock_t *block, Bool *isLeaf, Bool topLevel, rangemapRange_t start, rangemapRange_t length, Pointer ran, uInt logIndexSize )
    {
        RangeBlock_t      result= *block;
        rangemapRange_t   index;
        rangemapRange_t   indexSize;
        rangemapRange_t   f1,f2,f3;
        uInt              lL;
        
        if (length > 0) {        
            if (result == Nil) {
               /*
                * New contents fall entirely in a yet nonexistent block;
                * this is a simple case because no previous block has to 
                * be split into old and new contents.
                * Hence, just create the block. It will be (partially) filled below:
                */
                lL= shrinkWrap( start, length, logIndexSize );

                stdNEW(result);

                result->start        = start - LOGMOD(start, LOGBLOCKSIZE + lL);
                result->logIndexSize = lL;
                
               *block= result;

            } else {
               /*
                * New contents fall within an existing block.
                * the new range is partitioned over its sub-blocks:
                */
                rangemapRange_t bstart  = result->start;
                rangemapRange_t blength = BLOCKSIZE * POW2(result->logIndexSize);
                rangemapRange_t xstart  = stdMIN(start,bstart);
                rangemapRange_t xendm1  = stdMAX(start+length-1, bstart+blength-1);   // Avoid end-of-representation range problems
                rangemapRange_t xlength = xendm1 - xstart + 1;

                lL= shrinkWrap( xstart, xlength, logIndexSize );

                if (lL != result->logIndexSize) {
                    rangemapRange_t bIndex= LOGMOD( LOGDIV(bstart,lL), LOGBLOCKSIZE );

                    stdNEW(result);

                    result->start            = start - LOGMOD(start, LOGBLOCKSIZE + lL);
                    result->logIndexSize     = lL;
                    result->isLeaf  [bIndex] = False;
                    result->contents[bIndex] = *block;

                    blockFold( &result->contents[bIndex], &result->isLeaf[bIndex], result->logIndexSize-LOGBLOCKSIZE, False );

                   *block= result;
                }
            }
            
            
           /*
            * Split the to be mapped range into subranges of sizes f1, f2 and f3 where
            * - f1 is the size of the prefix of the first overlapping subblock that is NOT part of the inserted range
            * - f2 is the size of the intersection of the first overlapping subblock that is part of the inserted range
            * - f3 is the size of the suffix of the last overlapping subbllock that is NOT part of the inserted range
            *
            * Note that some of these three subranges may be empty.
            */
            indexSize = POW2(lL);
            index     = LOGMOD( LOGDIV(start,lL), LOGBLOCKSIZE );

            f1= LOGMOD(start,lL);
            
            if ((f1+length-1) < (indexSize-1)) {       // Avoid end-of-representation range problems
               /*
                * Range falls entirely within one block:
                */
                f2= length; 
                f3= indexSize-(f1+f2);
            } else {
                f2= indexSize-f1; 
                f3= 0;
            }

           /*
            * 1) Handle insertion in first overlapping subblock, in case the first element of this
            *    block is NOT part of the inserted range. If it is, then the next steps will
            *    properly deal with it, and this one can be ignored:
            */
            if (f1) {
                Bool    fract  = result->isLeaf  [index];
                Pointer oldRan = result->contents[index];
                
                if (fract && oldRan) {
                    result->contents[index]= Nil;
                    result->isLeaf  [index]= False;
                    blockDefine( &result->contents[index], &result->isLeaf[index], False, start-f1, f1, oldRan, lL-LOGBLOCKSIZE );
                    blockDefine( &result->contents[index], &result->isLeaf[index], False, start+f2, f3, oldRan, lL-LOGBLOCKSIZE );
                }

                blockDefine( &result->contents[index], &result->isLeaf[index], False, start, f2, ran, lL-LOGBLOCKSIZE );
                
                index++;
                start  += f2;
                length -= f2;
            }

           /*
            * 2) Handle whole-block insertion in the remaining part of the inserted range:
            */
            while (length >= indexSize) {
                Bool fract = result->isLeaf[index];

                if (!fract) {
                    blockDelete( result->contents[index] );
                    result->contents[index]= Nil;
                }

                result->contents[index]= ran;
                result->isLeaf  [index]= True;
                index++;
                start  += indexSize;
                length -= indexSize;
            }

           /*
            * 3) Handle partial block insertion in the last block:
            */
            if (length) {
                Bool    fract  = result->isLeaf  [index];
                Pointer oldRan = result->contents[index];
                
                if (fract && oldRan) {
                    result->contents[index]= Nil;
                    result->isLeaf  [index]= False;
                    blockDefine( &result->contents[index], &result->isLeaf[index], False, start+length, indexSize-length, oldRan, lL-LOGBLOCKSIZE );
                }

                blockDefine( &result->contents[index], &result->isLeaf[index], False, start, length, ran, lL-LOGBLOCKSIZE );
            }

            blockFold( block, isLeaf, logIndexSize, topLevel );
        }
    }

    static void blockUndefine( RangeBlock_t *block, Bool *isLeaf, rangemapRange_t start, rangemapRange_t length );
    
    
    static void checkRange( rangemapDomain_t *dom )
    {
        if (dom->start != 0) {
            rangemapRange_t capacity = -(dom->start);

            if (capacity < dom->length) { 
                stdASSERT( False, ("Specified range wraps around end of memory") );
            }    
        }    
    }


void STD_CDECL rangemapDefine( stdRangeMap_t rangemap, rangemapDomain_t dom, Pointer ran )
{
    checkRange(&dom);
    
    if (ran) {
        blockDefine( &rangemap->block, Nil, True, dom.start, dom.length, ran, MAXLOGBLOCKSIZE );
    } else {
        blockUndefine( &rangemap->block, Nil, dom.start, dom.length );
    }
}

void STD_CDECL rangemapDefine1( stdRangeMap_t rangemap, rangemapRange_t dom, Pointer ran )
{
    if (ran) {
        blockDefine( &rangemap->block, Nil, True, dom, 1, ran, MAXLOGBLOCKSIZE );
    } else {
        blockUndefine( &rangemap->block, Nil, dom, 1 );
    }
}


/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Remove range from the range map.
 * Parameters      : rangemap  (I) Range map to remove from.
 *                   dom       (I) Range to remove.
 * Function Result : 
 */
    static inline void clipRange( rangemapRange_t s1, rangemapRange_t l1, rangemapRange_t *s2, rangemapRange_t *l2) 
    {
        // Be careful to avoid end-of-representation range problems:
        rangemapRange_t e1m1=  s1 +  l1 - 1;
        rangemapRange_t e2m1= *s2 + *l2 - 1;
        
        *s2   = stdMAX( s1,  *s2   );
         e2m1 = stdMIN( e1m1, e2m1 );
         
         if (e2m1 < *s2) { *l2= 0;            }
                    else { *l2= e2m1+1 - *s2; }
    }


    static inline void blockLift( RangeBlock_t *block )
    {
        RangeBlock_t b            = *block;
        RangeBlock_t child        = Nil;
        uInt         nrofChildren = 0;
        uInt         i;

        for (i=0; i<BLOCKSIZE; i++) {
            if (b->isLeaf[i]) {
                return;
            } else
            if (b->contents[i]) {
                nrofChildren++;
                child = b->contents[i];
            }
        }

        if ( nrofChildren <= 1 ) {
            *block  = child;
            stdFREE(b);
        }
    }
    
    
    static void blockUndefine( RangeBlock_t *block, Bool *isLeaf, rangemapRange_t start, rangemapRange_t length )
    {
        RangeBlock_t  result= *block;
       
       /*
        * Strategy largely analogous to blockDefine:
        */ 
        if (length > 0 && result) {        
            uInt            logIndexSize = result->logIndexSize;
            rangemapRange_t indexSize    = POW2(logIndexSize);

            if ((LOGBLOCKSIZE+logIndexSize) < stdBITSIZEOF(rangemapRange_t)) { 
               if (logIndexSize < MAXLOGBLOCKSIZE) {
                   clipRange( result->start, BLOCKSIZE*indexSize, &start, &length ); 
               }
            }
            
            if (length) {
                rangemapRange_t  f1,f2,f3;
                rangemapRange_t  index  = LOGMOD( LOGDIV(start,logIndexSize), LOGBLOCKSIZE );

                f1= LOGMOD(start,logIndexSize);
                
                if ((f1+length-1) < (indexSize-1)) {         // Avoid end-of-representation range problems
                    f2= length; 
                    f3= indexSize-(f1+f2);
                } else {
                    f2= indexSize-f1; 
                    f3= 0;
                }

                if (f1) {
                    Bool    fract  = result->isLeaf  [index];
                    Pointer oldRan = result->contents[index];

                    if (fract) {
                        if (oldRan) {
                            result->contents[index] = Nil;
                            result->isLeaf  [index] = False;
                            blockDefine( &result->contents[index], &result->isLeaf[index], False, start-f1, f1, oldRan, logIndexSize-LOGBLOCKSIZE );
                            blockDefine( &result->contents[index], &result->isLeaf[index], False, start+f2, f3, oldRan, logIndexSize-LOGBLOCKSIZE );
                        }
                    } else {
                        blockUndefine( &result->contents[index], &result->isLeaf[index], start, f2 );
                    }
                                    
                    index++;
                    start  += f2;
                    length -= f2;
                }

                while (length >= indexSize) {
                    Bool  fract = result->isLeaf[index];

                    if (!fract) {
                        blockDelete( result->contents[index] );
                    }

                    result->contents[index]= Nil;
                    result->isLeaf  [index]= False;
                    index++;
                    start  += indexSize;
                    length -= indexSize;
                }

                if (length) {
                    Bool    fract  = result->isLeaf  [index];
                    Pointer oldRan = result->contents[index];

                    if (fract) {
                        if (oldRan) {
                            result->contents[index]= Nil;
                            result->isLeaf[index]= False;
                            blockDefine( &result->contents[index], &result->isLeaf[index], False, start+length, indexSize-length, oldRan, logIndexSize-LOGBLOCKSIZE );
                        }
                    } else {
                        blockUndefine( &result->contents[index], &result->isLeaf[index], start, length );
                    }
                }
                
                blockLift( block );
            }
        }
    }

void STD_CDECL  rangemapUndefine( stdRangeMap_t rangemap, rangemapDomain_t dom )
{
    checkRange(&dom);
    
    blockUndefine( &rangemap->block, Nil, dom.start, dom.length );
}

void STD_CDECL  rangemapUndefine1( stdRangeMap_t rangemap, rangemapRange_t dom )
{
    blockUndefine( &rangemap->block, Nil, dom, 1 );
}


/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Get range element associated with specified domain element.
 * Parameters      : rangemap  (I) Range map to inspect.
 *                   dom       (I) Element to get association from.
 * Function Result : The element y such that there exists a an
 *                   association (x,y) in the range map, with range map.equal(x,dom),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL rangemapApply( stdRangeMap_t rangemap, rangemapRange_t dom )
{
    RangeBlock_t block= rangemap->block;

    while (  block 
         && ( block->logIndexSize == MAXLOGBLOCKSIZE
           || stdINRANGE64(dom,block->start,POW2(LOGBLOCKSIZE+block->logIndexSize))
            )
          ) {
        rangemapRange_t index  = LOGDIV( dom - block->start, block->logIndexSize);

        if (block->isLeaf[index]) {
            return block->contents[index];
        } else {
            block= block->contents[index];
        }
    }

    return Nil;
}


/*
 * Function        : Test if rangemap is empty.
 * Parameters      : rangemap  (I) Range map to test.
 * Function Result : True iff. the rangemap does not have any ranges mapped
 */
Bool STD_CDECL  rangemapIsEmpty( stdRangeMap_t rangemap )
{
    return rangemap->block == Nil;
}


/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Dump internal representation of specified rangemap to writer object.
 * Parameters      : wr         (I) Writer to print to
 *                   rangemap   (I) Range map to dump.
 */
    static void bprint( stdWriter_t wr, RangeBlock_t block, Int level )
    {
        Int  i,j;
        rangemapRange_t start  = block->start;
        rangemapRange_t length = POW2( block->logIndexSize );

        for (j=0; j<=level; j++) { wtrPrintf(wr,"\t"); }
        wtrPrintf(wr,"[ (%d)\n",block->logIndexSize);
        
        for (i=0; i<BLOCKSIZE; i++) { 
            Pointer contents= block->contents[i];

            if (!contents) {
                for (j=0; j<=level; j++) { wtrPrintf(wr,"\t"); }
                wtrPrintf(wr," %4" stdFMT_LLX "-%4" stdFMT_LLX ": ----\n", start, start+length-1);
            } else
            if (block->isLeaf[i]) {
                for (j=0; j<=level; j++) { wtrPrintf(wr,"\t"); }
                wtrPrintf(wr," %4" stdFMT_LLX "-%4" stdFMT_LLX ": 0x%" stdFMT_ADDR "\n", start, start+length-1, (Address)contents);

            } else {
                for (j=0; j<=level; j++) { wtrPrintf(wr,"\t"); }
                wtrPrintf(wr," %4" stdFMT_LLX "-%4" stdFMT_LLX ": *\n", start, start+length-1);
                bprint( wr, contents, level+1 );
            }

            start += length;
        }
        for (j=0; j<=level; j++) { wtrPrintf(wr,"\t"); }
        wtrPrintf(wr,"]\n");
    }

void STD_CDECL  rangemapPrint( stdWriter_t wr, stdRangeMap_t rangemap )
{
    if (rangemap->block) { bprint(wr,rangemap->block,0); }
}


/*
 * Function        : Return the representation size of the specified rangemap .
 * Parameters      : rangemap   (I) Range map to measure.
 */
    static uInt nrofBlocks( RangeBlock_t block )
    {
        uInt i;
        uInt result = 1;
        
        for (i=0; i<BLOCKSIZE; i++) { 
            Pointer contents= block->contents[i];

            if (contents && !block->isLeaf[i]) {
                result += nrofBlocks( contents );
            }
        }

        return result;
    }
    
SizeT STD_CDECL  rangemapRSize( stdRangeMap_t rangemap )
{
    if (rangemap->block) { 
        return sizeof(*rangemap) + sizeof(*rangemap->block) * nrofBlocks(rangemap->block); 
    } else {
        return sizeof(*rangemap);
    }
}


/*
 * Function        : Return the maximal depth of the specified rangemap .
 * Parameters      : rangemap   (I) Range map to measure.
 */
    static uInt dofBlocks( RangeBlock_t block )
    {
        uInt i;
        uInt result = 0;
        
        for (i=0; i<BLOCKSIZE; i++) { 
            Pointer contents= block->contents[i];

            if (contents && !block->isLeaf[i]) {
                uInt depth= dofBlocks( contents );
                result= stdMAX(depth,result);
            }
        }

        return result+1;
    }
    
uInt STD_CDECL  rangemapDepth( stdRangeMap_t rangemap )
{
    if (rangemap->block) { 
        return dofBlocks(rangemap->block); 
    } else {
        return 0;
    }
}


