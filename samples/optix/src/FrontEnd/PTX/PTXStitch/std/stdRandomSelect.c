/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2011-2017, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdRandomSelect.h
 *
 *  Description              :
 *     
 *         Random element selection from data structures.
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdLocal.h"
#include "stdRandomSelect.h"

/*----------------------------------- Types ----------------------------------*/

typedef struct IndexRecRec {
    Pointer *array;
    uInt     size;
} *IndexRec;

struct stdRandomStateRec {
    uInt m_z,m_w;

    stdMap_t  map;
    stdMap_t  rmap;
};

/*--------------------------------- Functions --------------------------------*/

static uInt randomNumber( stdRandomState state )
{
    state->m_z = 36969 * (state->m_z & 65535) + (state->m_z >> 16);
    state->m_w = 18000 * (state->m_w & 65535) + (state->m_w >> 16);

    return (state->m_z << 16) + state->m_w;  /* 32-bit result */
}


static IndexRec getNewIndex( stdMap_t indexMap, Pointer dom, uInt size )
{
    IndexRec result;
    
    stdNEW(result);
    stdNEW_N(result->array,size);
        
    mapDefine(indexMap,dom,result);
    
    return result;
}

static void STD_CDECL addToIndex( Pointer p, IndexRec index )
{
    index->array[ index->size++ ]= p;
}


/*
 * Function        : Create random selection cache
 * Parameters      : size    (I) bucket size of underlying map in which
 *                               the data structures are stored upon which
 *                               random selection takes place
 *                   seed    (I) seed to initialize the underlying
 *                               random number generator
 * Function Result : newly initialized cache
 */
stdRandomState STD_CDECL stdCreateRandomState( uInt size, uInt seed )
{
    stdRandomState result;
    
    if (!seed || !~seed) { seed= 0x1234; }
        
    stdNEW(result);
    
    result->m_z =  seed;
    result->m_w = ~seed;
    
    result->map  = mapNEW(Pointer,size);
    result->rmap = mapNEW(Pointer,size);
    
    return result;
}

    
/*
 * Function        : Delete previously allocated selection cache
 * Parameters      : state   (I) cache to delete
 */
    static void STD_CDECL deleteIndex( IndexRec index )
    {
        if (index) {
          stdFREE(index->array);
          stdFREE(index);
        }
    }

void STD_CDECL stdDeleteRandomState( stdRandomState state )
{
    mapRangeTraverse( state->map,  (stdEltFun)deleteIndex, Nil );
    mapRangeTraverse( state->rmap, (stdEltFun)deleteIndex, Nil );
    
    mapDelete(state->map);
    mapDelete(state->rmap);
    
    stdFREE(state);
}

    
/*
 * Function        : Empty internal selection cache
 * Parameters      : state   (I) cache to clear
 */
void STD_CDECL stdEmptyRandomState( stdRandomState state )
{
    mapRangeTraverse( state->map,  (stdEltFun)deleteIndex, Nil );
    mapRangeTraverse( state->rmap, (stdEltFun)deleteIndex, Nil );
    
    mapEmpty(state->map);
    mapEmpty(state->rmap);
}

    
/*
 * Function        : Return random number within range
 * Parameters      :  state   (I) randomization cache
 *                    bound   (I) upper bound on returned result,
 *                                or 0 for no bound.
 * Function Result : Randomly selected number < bound
 */
uInt32 STD_CDECL stdRandomNumberSelect(stdRandomState state, uInt bound )
{
    uInt32 random= randomNumber(state);
    
    if (bound) { random %= bound; }
    
    return random;
}


/*
 * Function        : Return random element from list
 * Parameters      :  state   (I) randomization cache
 *                    l       (I) list to select from
 * Function Result : Randomly selected list element
 */
Pointer STD_CDECL stdRandomListSelect( stdRandomState state, stdList_t l )
{
    IndexRec index;
    
    index = mapApply(state->map,l);
    
    if (!index) {
        index= getNewIndex( state->map, l, listSize(l) );
        listTraverse(l, (stdEltFun)addToIndex, index);
    }
    
    return  index->array[ randomNumber(state) % index->size ];
}


/*
 * Function        : Return random element from set
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) set to select from
 * Function Result : Randomly selected set element
 */
Pointer STD_CDECL stdRandomSetSelect( stdRandomState state, stdSet_t s )
{
    IndexRec index;
    
    index = mapApply(state->map,s);
    
    if (!index) {
        index= getNewIndex( state->map, s, setSize(s) );
        setTraverse(s, (stdEltFun)addToIndex, index);
    }
    
    return  index->array[ randomNumber(state) % index->size ];
}


/*
 * Function        : Return random element from bit set
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) set to select from
 * Function Result : Randomly selected set element
 */
Pointer STD_CDECL stdRandomBitSetSelect( stdRandomState state, stdBitSet_t s )
{
    IndexRec index;
    
    index = mapApply(state->map,s);
    
    if (!index) {
        index= getNewIndex( state->map, s, bitSetSize(s) );
        bitSetTraverse(s, (stdEltFun)addToIndex, index);
    }
    
    return  index->array[ randomNumber(state) % index->size ];
}


/*
 * Function        : Return random domain element from map
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) map to select from 
 * Function Result : Randomly selected range element 
 */
Pointer STD_CDECL stdRandomDomainSelect( stdRandomState state, stdMap_t m )
{
    IndexRec index;
    
    index = mapApply(state->map,m);
    
    if (!index) {
        index= getNewIndex( state->map, m, mapSize(m) );
        mapDomainTraverse(m, (stdEltFun)addToIndex, index);
    }
    
    return  index->array[ randomNumber(state) % index->size ];
}



/*
 * Function        : Return random range element from map
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) map to select from 
 * Function Result : Randomly selected range element 
 */
Pointer STD_CDECL stdRandomRangeSelect( stdRandomState state, stdMap_t m )
{
    IndexRec index;
    
    index = mapApply(state->rmap,m);
    
    if (!index) {
        index= getNewIndex( state->rmap, m, mapSize(m) );
        mapRangeTraverse(m, (stdEltFun)addToIndex, index);
    }
    
    return  index->array[ randomNumber(state) % index->size ];
}



/*
 * Function        : Forget about data structure from which 
 *                   a random element was previously taken
 * Parameters      :  state   (I) previously inspected map, set or list
 *                    ds      (I) map, set or list 
 */
void STD_CDECL stdUndefineRandomStructure( stdRandomState state, Pointer ds )
{    
    deleteIndex( mapUndefine(state-> map,ds) );
    deleteIndex( mapUndefine(state->rmap,ds) );
}



