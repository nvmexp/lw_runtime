/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdUtils.h
 *
 *  Description              :
 *     
 */

#ifndef stdUtils_INCLUDED
#define stdUtils_INCLUDED

/*---------------------------------- Includes --------------------------------*/

#include "stdStdFun.h"
#include "stdMap.h"
#include "stdSet.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Discard variously structured maps.
 * Parameters      : map  (I) Map to discard.
 * Function Result : 
 */
// x --> SET(y)
void STD_CDECL mapDeleteSetMap   ( stdMap_t map );

// x --> BITSET(y)
void STD_CDECL mapDeleteBitSetMap   ( stdMap_t map );

// x --> (y-->z)
void STD_CDECL mapDeleteMapMap   ( stdMap_t map );

// x --> (y-->(z-->w))
void STD_CDECL mapDeleteMapMapMap( stdMap_t map );

/*
 * Function        : Empty variously structured maps.
 * Parameters      : map  (I) Map to discard.
 * Function Result : 
 */
// x --> SET(y)
void STD_CDECL mapEmptySetMap   ( stdMap_t map );

// x --> BITSET(y)
void STD_CDECL mapEmptyBitSetMap   ( stdMap_t map );

// x --> (y-->z)
void STD_CDECL mapEmptyMapMap   ( stdMap_t map );

// x --> (y-->(z-->w))
void STD_CDECL mapEmptyMapMapMap( stdMap_t map );


/*
 * Function        : Tokenize string according to specified separator characters.
 * Parameters      : value      (I) String to split.
 *                   separators (I) String containing separator characters to split on.
 *                   emptyFields(I) Pass False to skip empty tokens
 *                   doEscapes  (I) Pass False iff backslash characters need to be filtered.
 *                   fun        (I) Callback function for feeding each encountered token.
 *                   data       (I) Additional user specified data element to callback function.
 *                   keepQuote  (I) False: obeying quote characters
 *                                  True:  keeping quote characters even if not do escape
 *                   keepBraces (I) False: skipping brace characters
 *                                  True:  keeping brace characters even if not do escape
 * Function Result :
 */
void STD_CDECL stdTokenizeString(String value, cString separators, Bool emptyFields, Bool doEscapes, stdEltFun fun, Pointer data, Bool keepQuote, Bool keepBraces);

#ifdef __cplusplus
}
#endif

#endif
