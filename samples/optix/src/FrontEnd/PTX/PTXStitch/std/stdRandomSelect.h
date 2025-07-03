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

#ifndef stdRandomSelect_INCLUDED
#define stdRandomSelect_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdMap.h"
#include "stdSet.h"
#include "stdBitSet.h"
#include "stdList.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

/*
 * Caching handle, to hold state for speeding up 
 * subsequent random selections from same data structure:
 */
typedef struct stdRandomStateRec  *stdRandomState;

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create random selection cache
 * Parameters      : size    (I) bucket size of underlying map in which
 *                               the data structures are stored upon which
 *                               random selection takes place
 *                   seed    (I) seed to initialize the underlying
 *                               random number generator
 * Function Result : newly initialized cache
 */
stdRandomState STD_CDECL stdCreateRandomState( uInt size, uInt seed );

/*
 * Function        : Delete previously allocated selection cache
 * Parameters      : state   (I) cache to delete
 */
void STD_CDECL stdDeleteRandomState( stdRandomState state );
    
/*
 * Function        : Empty internal selection cache
 * Parameters      : state   (I) cache to clear
 */
void STD_CDECL stdEmptyRandomState( stdRandomState state );

/*
 * Function        : Return random number within range
 * Parameters      :  state   (I) randomization cache
 *                    bound   (I) upper bound on returned result,
 *                                or 0 for no bound.
 * Function Result : Randomly selected number < bound
 */
uInt32 STD_CDECL stdRandomNumberSelect(stdRandomState state, uInt bound );

/*
 * Function        : Return random element from list
 * Parameters      :  state   (I) randomization cache
 *                    l       (I) list to select from
 * Function Result : Randomly selected list element
 */
Pointer STD_CDECL stdRandomListSelect( stdRandomState state, stdList_t l );

/*
 * Function        : Return random element from set
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) set to select from
 * Function Result : Randomly selected set element
 */
Pointer STD_CDECL stdRandomSetSelect( stdRandomState state, stdSet_t s );

/*
 * Function        : Return random element from bit set
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) set to select from
 * Function Result : Randomly selected set element
 */
Pointer STD_CDECL stdRandomBitSetSelect( stdRandomState state, stdBitSet_t s );

/*
 * Function        : Return random domain element from map
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) map to select from 
 * Function Result : Randomly selected range element 
 */
Pointer STD_CDECL stdRandomDomainSelect( stdRandomState state, stdMap_t m );

/*
 * Function        : Return random range element from map
 * Parameters      :  state   (I) randomization cache
 *                    s       (I) map to select from 
 * Function Result : Randomly selected range element 
 */
Pointer STD_CDECL stdRandomRangeSelect( stdRandomState state, stdMap_t m );

/*
 * Function        : Forget about data structure from which 
 *                   a random element was previously taken
 * Parameters      :  state   (I) previously inspected map, set or list
 *                    ds      (I) map, set or list 
 */
void STD_CDECL stdUndefineRandomStructure( stdRandomState state, Pointer ds );


#ifdef __cplusplus
}
#endif

#endif
