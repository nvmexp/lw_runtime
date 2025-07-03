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
 *  Module name              : stdMap.h
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

#ifndef stdMap_INCLUDED
#define stdMap_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"
#include "stdStdFun.h"
#include "stdSet.h"
#include "stdList.h"
#include "stdWriter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct stdMapRec   *stdMap_t;

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Map creation macro, a shorthand for function mapCreate.
 * Parameters      : type         (I) Name of a type for which functions
 *                                    typeHash and typeEqual are in scope.
 *                   nrofBuckets  (I) Amount of buckets in underlying hash table.
 * Function Result : Requested map.
 */
#define   mapXNEW(type,nrofBuckets)  mapCreate(               \
                                       (stdHashFun )type##Hash,   \
                                       (stdEqualFun)type##Equal,  \
                                       nrofBuckets)

/*
 * Function        : Map creation macro, a shorthand for function mapCreate.
 * Parameters      : type         (I) Name of standard type.
 *                   nrofBuckets  (I) Amount of buckets in underlying hash table.
 * Function Result : Requested map.
 */
#define   mapNEW(type,nrofBuckets)   mapXNEW(std##type, nrofBuckets)

/*
 * Function        : Create new map.
 * Parameters      : hash         (I) Hash function, mapping the map's domain
 *                                    type to an arbitrary integer.
 *                   equal        (I) Equality function for domain type.
 *                   nrofBuckets  (I) Amount of buckets in underlying hash table.
 * Function Result : Requested map.
 */
stdMap_t STD_CDECL  mapCreate( stdHashFun hash, stdEqualFun equal, uInt nrofBuckets);

/*
 * Like mapCreate, but also pass generic data element as additional parameter
 * to every invocation of 'equal' and 'hash'.
 */
#define   mapDXNEW(type,nrofBuckets,data)  mapDCreate(                 \
                                           (stdHashDFun )type##Hash,   \
                                           (stdEqualDFun)type##Equal,  \
                                           nrofBuckets,data)

stdMap_t STD_CDECL  mapDCreate( stdHashDFun hash, stdEqualDFun equal, uInt nrofBuckets, Pointer data );


/*
 * Function        : Create new (empty) map with domain equality 
 *                   and bucket size identical to specified map
 * Parameters      : map          (I) Template map.
 * Function Result : Requested map.
 */
stdMap_t STD_CDECL  mapCreateLike( stdMap_t map );




/*
 * Function        : Discard map.
 * Parameters      : map  (I) Map to discard.
 * Function Result : True iff. the map was non-empty when 
 *                   it was passed to this function.
 */
void STD_CDECL  mapDelete( stdMap_t map );



/*
 * Function         : Remove all elements from the map.
 * Parameters       : map (O) Map to empty.
 * Function Result  : 
 */
Bool STD_CDECL  mapEmpty( stdMap_t map );




/*
 * Function        : Apply specified function to all pairs in the specified map,
 *                   with specified generic data element as additional parameter.
 *                   The order of traversal is unspecified, but reproducible as long
 *                   as the map has not changed. The map is not allowed to change 
 *                   during traversal, with the special exception that the 'current'
 *                   pair is allowed to be removed from the map by the traversal
 *                   function.
 * Parameters      : map        (I) Map to traverse.
 *                   traverse   (I) Function to apply to all map pairs.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result :
 */
void STD_CDECL  mapTraverse       ( stdMap_t map, stdPairFun traverse, Pointer data );


/*
 * Function        : Apply specified function to all domain pairs in the specified map,
 *                   with specified generic data element as additional parameter.
 *                   The domain traversal functions can be used as
 *                         shorthand for the main traversal function in case only
 *                         domain objects are of interest.
 * Parameters      : map        (I) Map to traverse.
 *                   traverse   (I) Function to apply to all map pairs.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result :
 */
void STD_CDECL  mapDomainTraverse ( stdMap_t map, stdEltFun  traverse, Pointer data );


/*
 * Function        : Apply specified function to all range pairs in the specified map,
 *                   with specified generic data element as additional parameter.
 *                   The range traversal functions can be used as
 *                         shorthand for the main traversal function in case only
 *                         range objects are of interest.
 * Parameters      : map        (I) Map to traverse.
 *                   traverse   (I) Function to apply to all map pairs.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result :
 */
void STD_CDECL  mapRangeTraverse  ( stdMap_t map, stdEltFun  traverse, Pointer data );




/*
 * Function        : Insert association pair into map.
 * Parameters      : map  (I) Map to insert into.
 *                   dom  (I) Domain part of pair to insert.
 *                   ran  (I) Range part of pair to insert.
 * Function Result : The element y such that there previously existed a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise. Note that if such an (x,y) oclwrred, it is 
 *                   replaced by the new association, since the map treats x and dom as
 *                   equal.
 */
Pointer STD_CDECL mapDefine( stdMap_t map, Pointer dom, Pointer ran );


/*
 * Function        : Multi- insert into map.
 * Parameters      : map  (I) map to insert into
 *                   doms (I) set of domain values
 *                   ran  (I) range value
 * Function Result : Foreach d in domain, the pair (d,ran) has been inserted into the map
 */
void STD_CDECL mapDefineMulti( stdMap_t map, stdSet_t doms, Pointer ran );


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
Pointer STD_CDECL mapUndefine1( stdMap_t map, Pointer *dom );



/*
 * Function        : Remove an association pair from the map.
 * Parameters      : map  (I) Map to remove from.
 *                   dom  (I) Domain part of pair to remove.
 * Function Result : The element y such that there previously existed a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise. If such an x oclwrred, its association is 
 *                   removed from the map.
 */
Pointer STD_CDECL  mapUndefine( stdMap_t map, Pointer dom );



/*
 * Function        : Remove an association pair from the map.
 * Parameters      : dom  (I) domain part of pair to remove
 *                   map  (I) map to remove from
 *                   
 * Function Result : -
 * NB              : This function is an analogon of mapUndefine,
 *                   intended as traversal function.
 */
void STD_CDECL mapRemoveDomain ( Pointer dom, stdMap_t map );



/*
 * Function        : Get range element associated with specified domain element.
 * Parameters      : map  (I) Map to inspect.
 *                   dom  (I) Element to get association from.
 * Function Result : The element y such that there exists a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL  mapApply( stdMap_t map, Pointer dom );




/*
 * Function        : Get range element associated with specified domain element.
 * Parameters      : map  (I)  Map to inspect.
 *                   dom  (IO) Element to get association from, 
 *                             and will hold the domain part found
 *                             in the map on function return.
 * Function Result : The element y such that there exists a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL  mapApply1( stdMap_t map, Pointer *dom );



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
Pointer STD_CDECL  mapApplyWD( stdMap_t map, Pointer dom, Pointer dflt );




/*
 * Function        : Get domain element from map.
 * Parameters      : map  (I) Map to inspect.
 *                   dom  (I) Element to get  from.
 * Function Result : The element x with map.equal(x,dom), or Nil otherwise.
 */
Pointer STD_CDECL mapDomain( stdMap_t map, Pointer dom );



/*
 * Function        : Insert association pair into map.
 * Parameters      : dom  (I) Domain part of pair to insert.
 *                   ran  (I) Range part of pair to insert.
 *                   map  (I) Map to insert into.
 * Function Result :
 * NB              : This function is an analogon of mapDefine,
 *                   intended as traversal function.
 */
void STD_CDECL mapAddTo( Pointer dom, Pointer ran, stdMap_t map );



/*
 * Function        : Insert association pair into ilwerse map.
 * Parameters      : dom  (I) range part of pair to insert
 *                   ran  (I) domain part of pair to insert
 *                   map  (I) map to insert into
 * Function Result : -
 * NB              : This function is an analogon of mapDefine,
 *                   intended as traversal function.
 */
void STD_CDECL mapAddIlwTo( Pointer dom, Pointer ran, stdMap_t map );



/*
 * Function        : Hash value of map.
 * Parameters      : map  (I) Map to return hash value from.
 * Function Result : Hash value
 */
uInt STD_CDECL mapHash( stdMap_t map );



/*
 * Function        : Return an arbitrary element from map's domain
 *                   (it is not removed).
 * Parameters      : map  (I) Map to return domain element from.
 * Function Result : The domain part of an arbitrary pair in the map, or Nil
 *                   if the map was empty.
 */
Pointer STD_CDECL  mapAnyDomain( stdMap_t map );



/*
 * Function        : Return an arbitrary element from map's range
 *                   (it is not removed).
 * Parameters      : map  (I) Map to return range element from.
 * Function Result : The range part of an arbitrary pair in the map, or Nil
 *                   if the map was empty.
 */
Pointer STD_CDECL  mapAnyRange( stdMap_t map );



/*
 * Function        : Test if the map associates the specified element,
 *                   that is, if it oclwrs in the domain of the map.
 * Parameters      : map  (I) Map to test.
 *                   dom  (I) Element to test for oclwrrence in domain.
 * Function Result : The element x in the map such that there exists a an
 *                   association (x,y) in the map, with map.equal(x,dom),
 *                   or Nil otherwise.
 */
Bool STD_CDECL  mapIsDefined( stdMap_t map, Pointer dom );


 
/*
 * Function        : Return number of elements in map.
 * Parameters      : map  (I) Map to size.
 * Function Result : Number of elements in map.
 */
SizeT STD_CDECL mapSize ( stdMap_t map );



/*
 * Function        : Copy a map.
 * Parameters      : map    (I) Map to copy.
 * Function Result : Copy of map. The (dom,ran) objects are not copied! .
 */
stdMap_t STD_CDECL mapCopy( stdMap_t map );



/*
 * Function        : Compare maps for equality.
 * Parameters      : map1   (I) Map1 to compare.
 *                   map2   (I) Map2 to compare
 *                   equal  (I) Equality function for range element type.
 * Function Result : Colwerted map.
 */
Bool STD_CDECL mapEqual( stdMap_t map1, stdMap_t map2, stdEqualFun equal );



/*
 * Function        : Create a set containing all elements from 
 *                   the range of the map.
 * Parameters      : map    (I) Map to colwert.
 *                   hash   (I) Hash function for range element type
 *                   equal  (I) Equality function for range element type.
 * Function Result : Colwerted map.
 */
stdSet_t STD_CDECL mapRangeToSet( stdMap_t map, stdHashFun hash, stdEqualFun equal);


/*
 * Function        : Create a set containing all elements from 
 *                   the domain of the map.
 * Parameters      : map    (I) Map to colwert.
 * Function Result : Colwerted map.
 */
stdSet_t STD_CDECL mapDomainToSet( stdMap_t map );



/*
 * Function        : Create a list containing all elements from 
 *                   the domain of the map.
 * Parameters      : map    (I) Map to colwert.
 * Function Result : Colwerted map.
 */
stdList_t STD_CDECL mapDomainToList( stdMap_t map );


/*
 * Function        : Create a list containing all elements from 
 *                   the domain of the map.
 * Parameters      : map    (I) Map to colwert.
 * Function Result : Colwerted map.
 */
stdList_t STD_CDECL mapRangeToList( stdMap_t map );


/*
 * Function        : Create a list containing all of the map's 
 *                   <domain,range> pairs (as a list).
 * Parameters      : map    (I) Map to colwert.
 * Function Result : Colwerted map.
 */
stdList_t STD_CDECL mapToList( stdMap_t map );


/*
 * Function        : Get hash table parameters of specified map.
 * Parameters      : map    (I) Map to inspect.
 *                   parms  (O) Returned hash table parameters
 * Function Result : 
 */
void STD_CDECL mapGetHashTableParameters( stdMap_t map, stdHashTableParameters *parms );


/*
 * Function        : Print hashing performance information via writer object.
 * Parameters      : wr      (I) Writer to print to
 *                   map     (I) Map to print.
 * Function Result : 
 */
void STD_CDECL mapPrint( stdWriter_t wr, stdMap_t map );

/*--------------------------------- Iterator --------------------------------*/
// Create iterator type and functions, so can write code like:
// stdMapIterator_t it;
// FOREACH_MAP_VALUE(map,it) {
//   d = mapDomailwalue(it);
//   r = mapRangeValue(it);
// }
typedef struct stdMapIteratorRec *stdMapIterator_t;

stdMapIterator_t mapBegin (stdMap_t map);
Bool mapAtEnd (stdMapIterator_t *it);
stdMapIterator_t mapNext (stdMapIterator_t it);
Pointer mapDomailwalue (stdMapIterator_t it);
Pointer mapRangeValue (stdMapIterator_t it);

#define FOREACH_MAP_VALUE(map,it) \
    for (it = mapBegin(map); !mapAtEnd(&it); it = mapNext(it))

#ifdef __cplusplus
}
#endif

#endif
