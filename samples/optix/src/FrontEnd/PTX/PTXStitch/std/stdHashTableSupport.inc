/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2012-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdHashTableSupport.inc
 *
 *  Description              :
 *     
 *         Implementation templates for stdMap and stdSet.
 */

/*----------------------------------- Types ----------------------------------*/

#define REHASH_FACTOR 4

typedef enum {
    MarkNone,
    MarkPointerMap,
    MarkIntegerMap
} HashMarker;

typedef   uInt32  *IntHeap;

struct stdHashRec {
    stdHashFun             hash;
    stdEqualFun            equal;
    /* following 3 fields only used in rare cases where 'data' is set */
    stdHashDFun            hashD;
    stdEqualDFun           equalD;
    Pointer                data;

    uInt32                 hashMask;
    SizeT                  size;
    uInt32                 hashValue;
    SizeT                  rehashSize;
    uInt32                 rover;

    uInt32                 blocksCapacity;
    uInt32                 blocksValidCapacity;

    uInt32                 blocksisStatic       : 2;
    uInt32                 blocksValidisStatic  : 2;

    HashMarker             marker               : 8;

    HashBlockRec          *blocks;
    uInt32                *blocksValid;

    IntHeap               *buckets;
};

/*--------------------------------- Functions --------------------------------*/

/*
 * The following implements heaps of integers as NIL_INT- terminated integer 
 * arrays, with the array capacity as first element. 
 * Such integer heaps are used as buckets in the hash table, and represent
 * indices into the array of preallocated 'blocks' that are also held by this table.
 */

#define NIL_INT 0xffffffff


#define intHeapFree stdFREE

static uInt  intHeapSize( IntHeap l )
{
    uInt result= 0;
    
    if (!l) {
        return 0; 
    } else {
        while (True) {
            uInt index= *(++l);

            if (index == NIL_INT) {
                break;
            } else {
                result++;
            }
        }

        return result;
    }
}

static IntHeap intHeapInsert( IntHeap l, uInt i )
{
    if (!l) {
        uInt c= 1;
    
        stdNEW_N(l, c + 2);

        l[0]= c;
        l[1]= i;
        l[2]= NIL_INT;
    
    } else {
        uInt s     = intHeapSize(l);
        uInt old_c = l[0];
        uInt c     = 2*old_c;
        
        if ((s+1) > old_c) {
            IntHeap t;
            
            stdNEW_N     (t,c+2);
            stdMEMCOPY_N (t+1,l+1,old_c);
            
            t[0]= c;
        
            stdFREE(l);
            l= t;
        }
        
        l[s+1] = i;
        l[s+2] = NIL_INT;
    }
    
    return l;
}

static IntHeap  intHeapRemove( IntHeap l, uInt i )
{
    IntHeap result = l;
    uInt32 *foundAt= Nil;
    
    while (True) {
        uInt index= *(++l);

        if (index == i) {
            foundAt= l;
        } else 
        if (index == NIL_INT) {
          /*
           * Note: the precondition to this function
           *       is that 'i' is in the intHeap.
           *       This means that foundAt must
           *       be non-Nil here:
           */
           *foundAt= *(l-1);
           *(l-1)  = NIL_INT;
           break;
        }
    }
    
    if (result[1]==NIL_INT) {
        stdFREE(result);
        result= Nil;
    }
    
    return result;
}


/*--------------------------------- Functions --------------------------------*/

/*
 * Block allocation and deallocation in the list of preallocated
 * hash blocks. In case we run out of blocks, the entire block array
 * is realloc'd to a larger one (by stdXArrayCheck).
 */

static HashBlock allocBlock( stdHash_t hash, uInt32 bucket )
{
    uInt rover    = hash->rover;
    uInt capacity = hash->blocksValidCapacity;
    
    uInt   i;
    uInt32 free;
    
    for (i=rover; i< capacity; i++) {
        if ( (free = ~hash->blocksValid[i] ) ) { goto found; }
    }
    
    for (i=0; i< rover; i++) {
        if ( (free = ~hash->blocksValid[i] ) ) { goto found; }
    }
    
    i    = capacity;
    free = ~0;
    stdXArrayCheck(hash->blocksValid,i);

  found:    
    
    {
         uInt bit   = stdFirstBit32(free);
         uInt index = i*stdBITSIZEOF(uInt32) + bit;
         
         stdXArrayCheck(hash->blocks,index);
         
         hash->buckets[bucket]= intHeapInsert( hash->buckets[bucket], index );
         
         hash->blocksValid[i] |= (1<<bit);
         
         hash->rover= i;
         
         return &hash->blocks[index];
    }
}

static void deallocBlock( stdHash_t hash, HashBlock block, uInt32 bucket )
{
    uInt index = block - hash->blocks;
               
    hash->buckets[bucket]= intHeapRemove( hash->buckets[bucket], index );
    
    hash->blocksValid[ index/stdBITSIZEOF(uInt32) ] ^= (1<<(index % stdBITSIZEOF(uInt32)));
}

static HashBlock lookupAny( stdHash_t hash )
{
    if (hash->size) {
        Int i;
        
        for (i=0; i<hash->blocksValidCapacity; i++) {
            uInt32 valid= hash->blocksValid[i];
            
            if (valid) {
                uInt  bit   = stdFirstBit32(valid);
                uInt  index = i*stdBITSIZEOF(uInt32) + bit;
                                
                return &hash->blocks[index];
            }
        }
    }
    
    return Nil;
}


/*--------------------------------- Functions --------------------------------*/

#define SEARCH_INT_LIST(hash,l,equal,key,data,equalD) \
{                                                    \
    if (!l) {                                        \
        return Nil;                                  \
    } else {                                         \
        HashBlockRec  *blocks= hash->blocks;         \
                                                     \
        while (True) {                               \
            uInt32 index= *(++l);                    \
                                                     \
            if (index==NIL_INT) {                    \
                return Nil;                          \
            } else {                                 \
                HashBlockRec *block= &blocks[index]; \
                if (data) {                          \
                  if (equalD(block->key,key,data)) { \
                    return block;                    \
                  }                                  \
                } else if (equal(block->key,key)) {  \
                    return block;                    \
                }                                    \
            }                                        \
        }                                            \
    }                                                \
}

#define _PointerEqual(a,b) _stdPointerEqual(a,b)
#define _IntEqual(a,b)     _stdIntEqual(a,b)
static Bool stdDummyEqualD ( Pointer e1, Pointer e2, Pointer data )
{ return False; }


static inline __ALWAYS_INLINE__ HashBlockRec *lookup( stdHash_t hash, Pointer key, uInt32 *hashValue )
{
    IntHeap l;

    switch (hash->marker) {
    case MarkNone       :
     {
       if (hash->data)
           *hashValue = hash->hashD(key, hash->data);        
       else
           *hashValue = hash->hash(key);        
        l         = hash->buckets[*hashValue & hash->hashMask];

        SEARCH_INT_LIST(hash,l,hash->equal,key,hash->data,hash->equalD);
     }
    case MarkPointerMap :
     {
       *hashValue = _stdPointerHash(key);        
        l         = hash->buckets[*hashValue & hash->hashMask];

        SEARCH_INT_LIST(hash,l,_PointerEqual,key,Nil,stdDummyEqualD);
     }
    case MarkIntegerMap :
     {
       *hashValue = _stdIntHash((Address)key);        
        l         = hash->buckets[*hashValue & hash->hashMask];

        SEARCH_INT_LIST(hash,l,_IntEqual,key,Nil,stdDummyEqualD);
     }
    default: 
       *hashValue=0;
        return Nil;
    }
}

/*--------------------------------- Functions --------------------------------*/

static stdHash_t hashCommonCreate( uInt nrofBuckets)
{
    uInt      log, realNrofBuckets;
    stdHash_t result;
                    
    nrofBuckets     = stdMAX(1,nrofBuckets);
    log             = stdLOG2U(nrofBuckets);
    realNrofBuckets = (1<<log);

    stdNEW(result)
    stdNEW_N(result->buckets,realNrofBuckets);

    result->hashMask   = realNrofBuckets-1;
    result->rehashSize = REHASH_FACTOR*realNrofBuckets;
    stdXArrayInit( result->blocks      );
    stdXArrayInit( result->blocksValid );

    return result;
}

stdHash_t STD_CDECL hashCreate( stdHashFun hash, stdEqualFun equal, uInt nrofBuckets)
{
    stdHash_t result = hashCommonCreate(nrofBuckets);
    
    result->hash       = hash;
    result->equal      = equal;
 
    if (hash == (stdHashFun)stdIntHash     && equal == (stdEqualFun)stdIntEqual    ) { result->marker= MarkIntegerMap; }
    if (hash == (stdHashFun)stdPointerHash && equal == (stdEqualFun)stdPointerEqual) { result->marker= MarkPointerMap; }

    return result;
}

stdHash_t STD_CDECL hashDCreate( stdHashDFun hashD, stdEqualDFun equalD, uInt nrofBuckets, Pointer data)
{
    stdHash_t result = hashCommonCreate(nrofBuckets);
    result->data       = data;
    result->hashD      = hashD;
    result->equalD     = equalD;
    return result;
}


void STD_CDECL hashDelete( stdHash_t hash )
{
    Int i= hash->hashMask;        
    
    while (i>=0) {
        intHeapFree(hash->buckets[i]);
        hash->buckets[i] = Nil;
            
        i--;
    }

    stdFREE( hash->buckets     );
    stdFREE( hash->blocks      );
    stdFREE( hash->blocksValid );
    stdFREE( hash              );
}



stdHash_t STD_CDECL hashCreateLike( stdHash_t hash )
{ 
  if (hash->data) 
      return hashDCreate(hash->hashD,hash->equalD,hash->hashMask+1,hash->data);
  else
      return hashCreate(hash->hash,hash->equal,hash->hashMask+1);
}

uInt STD_CDECL  hashHash( stdHash_t hash )
{ return hash->hashValue; }

SizeT STD_CDECL hashSize( stdHash_t hash )
{ return hash->size; }




Bool STD_CDECL hashEmpty( stdHash_t hash )
{
    if (hash->size == 0) {
        return False;
    } else {
        Int i= hash->hashMask;        
    
        while (i>=0) {
            intHeapFree(hash->buckets[i]);
            hash->buckets[i] = Nil;
            
            i--;
        }
        
        stdFREE( hash->blocks       );  stdXArrayInit( hash->blocks      );
        stdFREE( hash->blocksValid  );  stdXArrayInit( hash->blocksValid );
    
        hash->size      = 0;
        hash->hashValue = 0;
        
        return True;
    }
}

static void rehash( stdHash_t hash )
{
    uInt     nrofBuckets = hash->hashMask+1;
    
    uInt     newNrofBuckets = 2*nrofBuckets;
    SizeT    newRehashSize  = 2*hash->rehashSize;
    uInt     newHashMask    = newNrofBuckets-1;
    
    IntHeap *newBuckets     = Nil;
    
    Int      i;        
    
    stdNEW_N(newBuckets,newNrofBuckets);

    i= hash->hashMask;
            
    while (i>=0) {
        intHeapFree(hash->buckets[i]);
        hash->buckets[i] = Nil;

        i--;
    }
    
    stdFREE(hash->buckets);
    
    hash->hashMask    = newHashMask;
    hash->rehashSize  = newRehashSize;
    hash->buckets     = newBuckets;
    
    for (i=0; i<hash->blocksValidCapacity; i++) {
        uInt32 valid= hash->blocksValid[i];
        
        while (valid) {
            HashBlock block;
            uInt32    hashValue;
            uInt      bit   = stdFirstBit32(valid);
            uInt      index = i*stdBITSIZEOF(uInt32) + bit;
            
            valid ^= (1<<bit);
            
            block= &hash->blocks[index];
            
            if (hash->data)
                hashValue = hash->hashD(block->key, hash->data);
            else
                hashValue = hash->hash(block->key);
            
            hash->buckets[ hashValue & hash->hashMask ]= intHeapInsert( hash->buckets[ hashValue & hash->hashMask ],index );
        }
    }
}

void STD_CDECL getHashTableParameters( stdHash_t hash, stdHashTableParameters *parms )
{
    parms->hash        = hash->hash;
    parms->equal       = hash->equal;
    parms->nrofBuckets = hash->hashMask + 1;
    parms->data        = hash->data;
    parms->hashD       = hash->hashD;
    parms->equalD      = hash->equalD;
}


void STD_CDECL hashPrint( stdWriter_t wr, stdHash_t hash )
{
    SizeT size           = hashSize(hash);
    uInt nrBuckets       = hash->hashMask + 1;
    uInt nrFilledBuckets = 0;
    uInt maxBucketSize   = 0;
    uInt i;        
    
    stdXArray(uInt,sizes);
    stdXArrayInit (sizes);
    
    for (i=0; i<nrBuckets; i++) {
        IntHeap l= hash->buckets[i];
        uInt    s= intHeapSize(l);

        stdXArrayCheck(sizes,s);
        sizes[s]++;

        if (s) {
            maxBucketSize= stdMAX( maxBucketSize, s);
            nrFilledBuckets++;

            wtrPrintf(wr,"- %4d : %4d\t|", i, s ); 
                
            while (True) {
                uInt index= *(++l);

                if (index == NIL_INT) {
                    break;
                } else {
                    HashBlock h = &hash->blocks[index];

                    printBlock(wr,h);
                }
            }
                
            wtrPrintf(wr,"\n");
        }
    }

    wtrPrintf(wr," Optimal bucket filling: %f / bucket\n", (Double)size/nrBuckets       );
    wtrPrintf(wr," Current bucket filling: %f / bucket\n", (Double)size/nrFilledBuckets );
    wtrPrintf(wr," Largest bucket filling: %d\n",          maxBucketSize);
    
    wtrPrintf(wr," Bucket fillings       : ");
    for (i=0; i<sizesCapacity; i++) {
        if (sizes[i]) { wtrPrintf(wr,"%d(%d) ",i,sizes[i]); }

    }
    wtrPrintf(wr,"\n" );
}

/*--------------------------------- Iterator --------------------------------*/
struct stdHashIteratorRec {
  stdHash_t hashTable;
  int validBlocksIndex;
  int valid;
};

stdHashIterator_t hashBegin (stdHash_t hash)
{
  stdHashIterator_t it;
  if (hash->size == 0) {
    return NULL;
  }
  stdNEW(it);
  it->hashTable = hash;
  it->validBlocksIndex = 0;
  it->valid = hash->blocksValid[0];
  return it;
}

// pass in pointer to iterator as is freed if at end
Bool hashAtEnd (stdHashIterator_t *it)
{
  if (it == NULL || *it == NULL) return True;
  if ((*it)->validBlocksIndex >= (*it)->hashTable->blocksValidCapacity) {
    // if at end, destroy the iterator
    stdFREE(*it);
    *it = NULL;
    return True;
  }
  return False;
}

stdHashIterator_t hashNext (stdHashIterator_t it)
{
  if (it == NULL) return NULL;
  if (it->validBlocksIndex < it->hashTable->blocksValidCapacity) {
    if (it->valid) {
      uInt bit = stdFirstBit32(it->valid);
      it->valid ^= (1<<bit);
      while (it->valid == 0 
          && it->validBlocksIndex < it->hashTable->blocksValidCapacity) 
      {
        it->validBlocksIndex++;
        it->valid = it->hashTable->blocksValid[it->validBlocksIndex];
      }
    }
  }
  return it;
}

static HashBlock hashBlockValue (stdHashIterator_t it)
{
  if (it == NULL) return NULL;
  if (it->validBlocksIndex < it->hashTable->blocksValidCapacity) {
    if (it->valid) {
      uInt      bit   = stdFirstBit32(it->valid);
      uInt      index = it->validBlocksIndex*stdBITSIZEOF(uInt32) + bit;
      HashBlock block = &it->hashTable->blocks[index];
      return block;
    }
  }
  return NULL;
}

