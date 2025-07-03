/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2019, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdString.h
 *
 *  Description              :
 *
 *  Implementation of an abstract type 'string', based upon a list
 *  of small blocks. strings do not run out of space while stdMALLOC
 *  returns Nil.
 */

#ifndef stdString_INCLUDED
#define stdString_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdTypes.h"
#include "stdLocal.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types ------------------------------------*/

typedef struct stdString  *stdString_t;

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : String creation macro, a shorthand for function stringCreate.
 * Parameters      :
 * Function Result : Requested (empty) string.
 **/
#define   stringNEW()   stringCreate(128)



/*
 * Function        : Check for empty string, a shorthand for stringSize.
 * Parameters      :
 * Function Result : Requested (empty) string.
 **/
#define   stringIsEmpty(string)   (stringSize(string) == 0)



/*
 * Function        : Create new string.
 * Parameters      : szofBuckets  (I) Minimal size of memory blocks.
 * Function Result : Requested (empty) string.
 */
stdString_t STD_CDECL stringCreate( SizeT szofBuckets );



/*
 * Function        : Discard string.
 * Parameters      : string  (I) String to discard.
 * Function Result : 
 */
void STD_CDECL  stringDelete( stdString_t string );



/*
 * Function        : Make string empty.
 * Parameters      : string  (I) String to empty.
 * Function Result : 
 */
void STD_CDECL  stringEmpty( stdString_t string );



/*
 * Function        : Returns a C string represention of string.
 * Parameters      : string     (I) String to make into C string.
 * Function Result : A C string created with stdMALLOC. Caller is
 *                   responsible for freeing memory.
 */
String STD_CDECL  stringToBuf( stdString_t string );



/*
 * Function        : Returns a C string represention of string,
 *                   and delete string itself.
 *                   This function is a colwenient shorthand for
 *                   the combination stringToBuf and stringDelete.
 * Parameters      : string     (I) string to make into C string
 * Function Result : A C string created with stdMALLOC. Caller is
 *                   responsible for freeing memory.
 */
String STD_CDECL stringStripToBuf( stdString_t string );



/*
 * Function        : Append a string with a C string.
 * Parameters      : string      (IO) String to append to.
 *                   buf         (I)  Its donor.
 * Function Result : 
 */
void STD_CDECL  stringAddBuf( stdString_t string, cString buf );



/*
 * Function        : Append a string with a prefix of a C string.
 * Parameters      : string      (IO) String to append to.
 *                   buf         (I)  Its donor; len characters will
 *                                    be added to string, none of
 *                                    which should be null.
 *                   len         (I)  The number of characters from
 *                                    buf to be added.
 * Function Result : 
 */
void STD_CDECL  stringAddBufLen( stdString_t string, cString buf, SizeT len );



/*
 * Function        : Append a string with a single character.
 * Parameters      : string      (IO) String to append to.
 *                   c           (I)  Its donor.
 * Function Result : 
 */
void STD_CDECL  stringAddChar( stdString_t string, Char c );



/*
 * Function        : Append a string with formatted text.
 * Parameters      : string      (IO) String to append to.
 *                   format      (I)  The 'sprintf' format.
 *                   ...         (I)  Its donor.
 * Function Result : Number of characters added to string
 */
SizeT STD_CDECL __CHECK_FORMAT__(printf, 2, 3) stringAddFormat( stdString_t string, cString format, ... );



/*
 * Function        : Append a string with formatted text.
 * Parameters      : string      (IO) String to append to.
 *                   format      (I)  The 'sprintf' format.
 *                   arg         (I)  Its donor.
 * Function Result : Number of characters added to string
 */
SizeT STD_CDECL  stringAddVFormat( stdString_t string, cString format, va_list arg );



/*
 * Function        : Concatenate two strings.
 * Parameters      : toString    (IO) String to append to.
 *                   fromString  (I)  Its donor.
 * Function Result :
 */
void STD_CDECL  stringAddString( stdString_t toString, stdString_t fromString );



/*
 * Function        : Copy a string.
 * Parameters      : string      (I) String to be copied.
 * Function Result : The copied string.
 */
stdString_t STD_CDECL stringCopy( stdString_t string );



/*
 * Function        : Size of string.
 * Parameters      : string  (I)  String to size.
 * Function Result : String size (without '\0').
 */
SizeT STD_CDECL  stringSize( stdString_t string );


#ifdef __cplusplus
}
#endif

#endif  /* stdString_INCLUDED */
