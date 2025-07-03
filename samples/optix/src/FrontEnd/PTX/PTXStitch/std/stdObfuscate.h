/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2017, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdObfuscate.h
 *
 *  Description              :
 *
 */

#ifndef stdObfuscate_INCLUDED
#define stdObfuscate_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdTypes.h"


#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct stdObfuscationStateRec  *stdObfuscationState;

/*-------------------------------- Functions ---------------------------------*/

/*
 * Function         : Create new obfuscation engine from specified seed
 * Parameters       : seed         (I) Initializer for pseudo- random number
 *                                      generator hidden by this module.
 * Function Result  : New obfuscation state
 */
stdObfuscationState STD_CDECL stdCreateObfuscation( uInt32 seed );


/*
 * Function         : Clone specified obfuscation engine
 * Parameters       : state        (I) obfuscation state to clone
 * Function Result  : Cloned obfuscation state
 */
stdObfuscationState STD_CDECL stdCloneObfuscation( stdObfuscationState obfuscation );


/*
 * Function         : Delete obfuscation engine
 * Parameters       : obfuscation  (I) Obfuscation engine to delete
 */
void STD_CDECL stdDeleteObfuscation( stdObfuscationState obfuscation );


/*
 * Function         : Obfuscate character using specified obfuscaton state.
 * Parameters       : state        (I) obfuscation state to use
 *                    c            (I) character to obfuscate using next
 *                                     pseudo random number.
 * Function Result  : Obfuscated c.
 */
Char STD_CDECL stdObfuscate( stdObfuscationState state, Char c );


/*
 * Function         : Obfuscate character buffer using specified obfuscaton state.
 * Parameters       : state        (I) obfuscation state to use
 *                    buf          (I) character buffer to obfuscate using next
 *                                     sequence of pseudo random numbers.
 *                    size         (I) size of buffer
 * Function Result  : Obfuscated c.
 */
void STD_CDECL stdObfuscateBuffer( stdObfuscationState state, Char *buf, uInt size );

/*
 * Function         : Deobfuscate character using specified obfuscaton state.
 * Parameters       : state        (I) obfuscation state to use
 *                    c            (I) character to obfuscate using next
 *                                     pseudo random number.
 * Function Result  : Obfuscated c.
 */
Char STD_CDECL stdDeobfuscate( stdObfuscationState state, Char c );


/*
 * Function         : Deobfuscate character buffer using specified obfuscaton state.
 * Parameters       : state        (I) obfuscation state to use
 *                    buf          (I) character buffer to obfuscate using next
 *                                     sequence of pseudo random numbers.
 *                    size         (I) size of buffer
 * Function Result  : Obfuscated c.
 */
void STD_CDECL stdDeobfuscateBuffer( stdObfuscationState state, Char *buf, uInt size );


#if     defined(__cplusplus)
}
#endif 

#endif
