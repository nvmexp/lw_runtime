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
 *  Module name              : stdMap.c
 *
 *  Description              :
 *     
 *         This module defines an abstract data type 'map',
 *         wich associates elements of a 'domain' type to 
 *         elements of a 'range' type; in other words, it maps
 *         the domain type into the range type.
 *         The map is implemented as a hash table with a number
 *         of buckets that can be specified at creation.
 *
 *         The both the domain- and range type of the map are represented  
 *         by the generic type 'Pointer', but the domain type is further defined
 *         by the equality function specified when creating the map;
 *         Obviously, maps can map (pointers to) memory objects to (pojnters to)
 *         other memory objects, but as special exception objects of integer type 
 *         are allowed for domain- as well as range elements.
 *
 *         Map operation performance is further defined how
 *         'well', or how uniformly its hash function spreads 
 *         the domain type over the integer domain.
 *         
 *         The usual map operations are defined, plus a traversal
 *         procedure.
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdMap.h"
#include "stdList.h"
#include "stdLocal.h"
#include "stdStdFun.h"

/*----------------------------------- Types ----------------------------------*/

typedef struct {
    Pointer       key;
    Pointer       ran;
} HashBlockRec, *HashBlock;

/*--------------------------------- Functions --------------------------------*/

#define stdHash_t   stdMap_t
#define stdHashRec  stdMapRec

#define hashCreate                mapCreate
#define hashDCreate               mapDCreate
#define hashDelete                mapDelete
#define hashEmpty                 mapEmpty
#define getHashTableParameters    mapGetHashTableParameters
#define hashPrint                 mapPrint
#define hashHash                  mapHash
#define hashSize                  mapSize
#define hashCreateLike            mapCreateLike

#define printBlock(wr,h)          wtrPrintf(wr," (%p,%p)", (void*)h->key, (void*)h->ran ); 

#define stdHashIterator_t   stdMapIterator_t
#define stdHashIteratorRec  stdMapIteratorRec
#define hashBegin           mapBegin
#define hashAtEnd           mapAtEnd
#define hashNext            mapNext

#include "stdHashTableSupport.inc"

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Apply specified function to all (domain,range) pairs in the specified map,
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is unspecified, but reproducible as long
 *                   as the map has not changed. The map is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   pair is allowed to be removed from the map by the traversal
 *                   function.
 *                   NOTE: The domain- and range traversal functions can be used as
 *                         shorthand for the main traversal function in case only
 *                         domain-, cq range objects are of interest.
 * Parameters      : map        (I) map to traverse
 *                   traverse   (I) function to apply to all map pairs
 *                   data       (I) generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'
 * Function Result : -
 */
void STD_CDECL mapTraverse( stdMap_t map, stdPairFun traverse, Pointer data )
{
    if (map->size) {
        Int i;
        
        for (i=0; i < (Int)map->blocksValidCapacity; i++) {
            uInt32 valid= map->blocksValid[i];
            
            while (valid) {
                HashBlock block;
                uInt      bit   = stdFirstBit32(valid);
                uInt      index = i*stdBITSIZEOF(uInt32) + bit;
                
                valid ^= (1<<bit);
                
                block= &map->blocks[index];
                
                traverse(block->key,block->ran,data);
            }
        }
    }
}

void STD_CDECL mapDomainTraverse( stdMap_t map, stdEltFun  traverse, Pointer data )
{
    if (map->size) {
        Int i;
        
        for (i=0; i < (Int)map->blocksValidCapacity; i++) {
            uInt32 valid= map->blocksValid[i];
            
            while (valid) {
                HashBlock block;
                uInt      bit   = stdFirstBit32(valid);
                uInt      index = i*stdBITSIZEOF(uInt32) + bit;
                
                valid ^= (1<<bit);
                
                block= &map->blocks[index];
                
                traverse(block->key,data);
            }
        }
    }
}

void STD_CDECL mapRangeTraverse( stdMap_t map, stdEltFun  traverse, Pointer data )
{
    if (map->size) {
        Int i;
        
        for (i=0; i < (Int)map->blocksValidCapacity; i++) {
            uInt32 valid= map->blocksValid[i];
            
            while (valid) {
                HashBlock block;
                uInt      bit   = stdFirstBit32(valid);
                uInt      index = i*stdBITSIZEOF(uInt32) + bit;
                
                valid ^= (1<<bit);
                
                block= &map->blocks[index];
                
                traverse(block->ran,data);
            }
        }
    }
}



/*
 * Function        : Insert association pair into map.
 * Parameters      : map  (I) map to insert into
 *                   dom  (I) domain part of pair to insert
 *                   ran  (I) range part of pair to insert
 * Function Result : The element y such that there previously existed a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise. Note that if such an (x,y) oclwrred, it is 
 *                   replaced by the new association, since the map treats x and dom as
 *                   equal.
 */
Pointer STD_CDECL mapDefine( stdMap_t map, Pointer dom, Pointer ran )
{
    uInt32    hashValue;
    HashBlock l= lookup(map,dom,&hashValue);
    
    if (l) {
        stdSWAP(l->ran,ran,Pointer);

        return ran;

    } else {
        HashBlockRec *nw = allocBlock(map, hashValue & map->hashMask);

        nw->key  = dom;
        nw->ran  = ran;

        map->size      += 1;
        map->hashValue ^= hashValue;

        if (map->size > map->rehashSize) {
            rehash(map);
        }

        return Nil;
    }
}


/*
 * Function        : Multi- insert into map.
 * Parameters      : map  (I) map to insert into
 *                   doms (I) set of domain values
 *                   ran  (I) range value
 * Function Result : Foreach d in domain, the pair (d,ran) has been inserted into the map
 */
    typedef struct  {
        stdMap_t    map;
        Pointer     ran;
    } DefineMultiRec;

    static void STD_CDECL defRan( Pointer dom, DefineMultiRec *rec )
    { mapDefine(rec->map,dom,rec->ran); }

void STD_CDECL mapDefineMulti( stdMap_t map, stdSet_t doms, Pointer ran )
{
    DefineMultiRec rec;
    
    rec.map= map;
    rec.ran= ran;
    
    setTraverse(doms,(stdEltFun)defRan,&rec);
}


/*
 * Function        : Remove an association pair from the map.
 * Parameters      : map  (I)  map to remove from
 *                   dom  (IO) domain part of pair to remove, 
 *                             and will hold the domain part found
 *                             in the map on function return
 * Function Result : The element y such that there previously existed a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise. If such an x oclwrred, its association is 
 *                   removed from the map.
 */
Pointer STD_CDECL mapUndefine1( stdMap_t map, Pointer *dom )
{
    uInt32    hashValue;
    HashBlock l= lookup(map,*dom,&hashValue);
    
    if (l) {
        Pointer ran   = l->ran;
                
       *dom = l->key;

        map->size      -= 1;
        map->hashValue ^= hashValue;

        deallocBlock(map, l, hashValue & map->hashMask);            
        
        return ran;

    } else {
       *dom =  Nil;
        return Nil;
    }
}


/*
 * Function        : Remove an association pair from the map.
 * Parameters      : map  (I) map to remove from
 *                   dom  (I) domain part of pair to remove
 * Function Result : The element y such that there previously existed a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise. If such an x oclwrred, its association is 
 *                   removed from the map.
 */
Pointer STD_CDECL mapUndefine( stdMap_t map, Pointer dom )
{
    return mapUndefine1(map,&dom);
}

                
/*
 * Function        : Remove an association pair from the map.
 * Parameters      : dom  (I) domain part of pair to remove
 *                   map  (I) map to remove from
 *                   
 * Function Result : -
 * NB              : This function is an analogon of mapUndefine,
 *                   intended as traversal function.
 */
void STD_CDECL mapRemoveDomain( Pointer dom, stdMap_t map )
{
    mapUndefine(map, dom);
}
   
                
/*
 * Function        : Get range element associated with specified domain element.
 * Parameters      : map  (I) map to inspect
 *                   dom  (I) element to get association from
 * Function Result : The element y such that there exists a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL mapApply( stdMap_t map, Pointer dom )
{
    uInt32    hashValue;
    HashBlock l= lookup(map,dom,&hashValue);
    
    if (l) {
        return l->ran;
    } else {
        return Nil;
    }
}



/*
 * Function        : Get range element associated with specified domain element.
 * Parameters      : map  (I)  map to inspect
 *                   dom  (IO) element to get association from
 *                             and will hold the domain part found
 *                             in the map on function return.
 * Function Result : The element y such that there exists a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL mapApply1( stdMap_t map, Pointer *dom )
{
    uInt32    hashValue;
    HashBlock l= lookup(map,*dom,&hashValue);
    
    if (l) {
       *dom =  l->key;
        return l->ran;
    } else {
        return Nil;
    }
}



/*
 * Function        : Get range element associated with specified domain element,
 *                   returning specified default value when association is not there.
 * Parameters      : map  (I) Map to inspect.
 *                   dom  (I) Element to get association from.
 *                   dflt (I) Default value to return if no association
 * Function Result : The element y such that there exists a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or 'dflt' otherwise.
 */
Pointer STD_CDECL  mapApplyWD( stdMap_t map, Pointer dom, Pointer dflt )
{
    uInt32    hashValue;
    HashBlock l= lookup(map,dom,&hashValue);
    
    if (l) {
        return l->ran;
    } else {
        return dflt;
    }
}

               
                
/*
 * Function        : Insert association pair into map.
 * Parameters      : dom  (I) domain part of pair to insert
 *                   ran  (I) range part of pair to insert
 *                   map  (I) map to insert into
 * Function Result : -
 * NB              : This function is an analogon of mapDefine,
 *                   intended as traversal function.
 */
void STD_CDECL mapAddTo( Pointer dom, Pointer ran, stdMap_t map )
{ mapDefine(map,dom,ran); }



/*
 * Function        : Insert association pair into ilwerse map.
 * Parameters      : dom  (I) range part of pair to insert
 *                   ran  (I) domain part of pair to insert
 *                   map  (I) map to insert into
 * Function Result : -
 * NB              : This function is an analogon of mapDefine,
 *                   intended as traversal function.
 */
void STD_CDECL mapAddIlwTo( Pointer dom, Pointer ran, stdMap_t map )
{ mapDefine(map,ran,dom); }



/*
 * Function        : Return an arbitrary element from map's domain
 *                   (it is not removed).
 * Parameters      : map  (I) map to return domain element from
 * Function Result : The domain part of an arbitrary pair in the map, or Nil
 *                   if the map was empty.
 */
Pointer STD_CDECL   mapAnyDomain( stdMap_t map )
{
    HashBlock l= lookupAny(map);

    if (l) { return l->key; }
      else { return Nil;   }
}



/*
 * Function        : Return an arbitrary element from map's range
 *                   (it is not removed).
 * Parameters      : map  (I) Map to return range element from.
 * Function Result : The range part of an arbitrary pair in the map, or Nil
 *                   if the map was empty.
 */
Pointer STD_CDECL  mapAnyRange( stdMap_t map )
{
    HashBlock l= lookupAny(map);

    if (l) { return l->ran; }
      else { return Nil;   }
}



/*
 * Function        : Get domain element from map.
 * Parameters      : map  (I) map to inspect
 *                   dom  (I) element to get  from
 * Function Result : The element x with map.equal(x,dom), or Nil otherwise.
 */
Pointer STD_CDECL mapDomain( stdMap_t map, Pointer dom )
{
    uInt32    hashValue;
    HashBlock l= lookup(map,dom,&hashValue);
    
    if (l) {
        return l->key;
    } else {
        return Nil;
    }
}



/*
 * Function        : Test if the map associates the specified element,
 *                   that is, if it oclwrs in the domain of the map.
 * Parameters      : map  (I) map to test
 *                   dom  (I) element to test for oclwrrence in domain
 * Function Result : The element x in the map such that there exists a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise
 */
Bool STD_CDECL mapIsDefined( stdMap_t map, Pointer dom )
{
    uInt32    hashValue;
    HashBlock l= lookup(map,dom,&hashValue);
    
    if (l) {
        return True;
    } else {
        return False;
    }
}




/*
 * Function        : Copy a map
 * Parameters      : map    (I) map to copy
 * Function Result : copy of map. The (dom,ran) objects are not copied!
 */
stdMap_t STD_CDECL mapCopy( stdMap_t map )
{
    stdMap_t nw = mapCreateLike(map);

    mapTraverse(map, (stdPairFun)mapAddTo, nw);
    
    return nw;
}


/*
 * Function        : Compare maps for equality.
 * Parameters      : map1   (I) Map1 to compare.
 *                   map2   (I) Map2 to compare
 *                   equal  (I) Equality function for range element type.
 * Function Result : Colwerted map.
 */
Bool STD_CDECL mapEqual( stdMap_t map1, stdMap_t map2, stdEqualFun equal )
{
    Int i= map1->hashMask;
    
    if (map1            == map2           ) { return True;  }
    if (map1->size      != map2->size     ) { return False; }
    if (map1->hashValue != map2->hashValue) { return False; }
    
    while (i>=0) {
        IntHeap l= map1->buckets[i];
        
        if (l) {        
            while (True) {
                uInt index= *(++l);

                if (index == NIL_INT) {
                    break;
                } else {
                    uInt32    hashValue;
                    HashBlock h1,h2;

                    h1 = &map1->blocks[index];
                    h2 = lookup(map2,h1->key,&hashValue);

                    if (!h2)                          { return False; }
                    if (!equal(h1->ran,h2->ran))      { return False; }
                }
            }
        }
        
        i--;
    }

    return True;

}



/*
 * Function        : create a set containing all elements from 
 *                   the range of the map
 * Parameters      : map    (I) map to colwert
 *                   hash   (I) Hash function for range element type
 *                   equal  (I) Equality function for range element type.
 * Function Result : colwerted map
 */
stdSet_t STD_CDECL mapRangeToSet( stdMap_t map, stdHashFun hash, stdEqualFun equal)
{
    stdSet_t set = setCreate(hash, equal, map->hashMask + 1);

    mapRangeTraverse(map, (stdEltFun)setAddTo, set);
    
    return set;
}



/*
 * Function        : create a set containing all elements from 
 *                   the domain of the map
 * Parameters      : map    (I) map to colwert
 * Function Result : colwerted map
 */
stdSet_t STD_CDECL mapDomainToSet( stdMap_t map )
{
    stdSet_t set = setCreate(map->hash, map->equal, map->hashMask + 1);

    mapDomainTraverse(map, (stdEltFun)setAddTo, set);
    
    return set;
}



/*
 * Function        : Create a list containing all elements from 
 *                   the domain of the map.
 * Parameters      : map    (I) Map to colwert.
 * Function Result : Colwerted map.
 */
stdList_t STD_CDECL mapDomainToList( stdMap_t map )
{
    stdList_t list = Nil;

    mapDomainTraverse(map, (stdEltFun)listAddTo, &list);
    
    return list;
}



/*
 * Function        : Create a list containing all elements from 
 *                   the domain of the map.
 * Parameters      : map    (I) Map to colwert.
 * Function Result : Colwerted map.
 */
stdList_t STD_CDECL mapRangeToList( stdMap_t map )
{
    stdList_t list = Nil;

    mapRangeTraverse(map, (stdEltFun)listAddTo, &list);
    
    return list;
}


/*
 * Function        : Create a list containing all of the map's 
 *                   <domain,range> pairs (as a list).
 * Parameters      : map    (I) Map to colwert.
 * Function Result : Colwerted map.
 */
    static void STD_CDECL pairAddTo( Pointer dom, Pointer ran, stdList_t *l )
    {
        stdList_t pair= Nil;

        listAddTo(ran, &pair);
        listAddTo(dom, &pair);
        listAddTo(pair, l   );
    }
 
stdList_t STD_CDECL mapToList( stdMap_t map )
{
    stdList_t list = Nil;

    mapTraverse(map, (stdPairFun)pairAddTo, &list);
    
    return list;
}

/*--------------------------------- Iterator --------------------------------*/

Pointer mapDomailwalue (stdMapIterator_t it)
{
  HashBlock block = hashBlockValue(it);
  if (block == NULL) return NULL;
  return block->key;
}

Pointer mapRangeValue (stdMapIterator_t it)
{
  HashBlock block = hashBlockValue(it);
  if (block == NULL) return NULL;
  return block->ran;
}


