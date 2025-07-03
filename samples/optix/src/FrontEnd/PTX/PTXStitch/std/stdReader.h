/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdReader.h
 *
 *  Description              :
 *
 */

#ifndef stdReader_INCLUDED
#define stdReader_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdString.h"
#include "stdLocal.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types ------------------------------------*/

typedef struct stdReader  *stdReader_t;

typedef uInt (STD_CDECL *rdrReaderFun )( Pointer data, Byte *buffer, uInt amount );
typedef void (STD_CDECL *rdrResetFun  )( Pointer data );
typedef void (STD_CDECL *rdrCleanupFun)( Pointer data );

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new reader object.
 * Parameters      : read    (I) Reader function to encapsulate
 *                   reset   (I) Reset function to encapsulate
 *                   cleanup (I) Function to cleanup state upon reader deletion, 
 *                                or Nil when not appropriate
 *                   data    (I) Reader state to read from
 * Function Result : Requested (empty) string.
 */
stdReader_t STD_CDECL rdrCreate( rdrReaderFun read, rdrResetFun reset, rdrCleanupFun cleanup, Pointer data );


/*
 * Function        : Discard reader object.
 * Parameters      : r       (I) Reader to discard.
 * Function Result : 
 */
void STD_CDECL  rdrDelete( stdReader_t r );


/*
 * Function        : Reset reader object to set read position to '0'.
 * Parameters      : r       (I)  Reader to reset.
 * Function Result : 
 */
void STD_CDECL  rdrReset( stdReader_t r );


/*
 * Function        : Read block of data from reader object.
 * Parameters      : r       (I)  Reader to read from.
 *                   buffer  (I)  Buffer to read into.
 *                   amount  (I)  Maximal number of bytes to read.
 * Function Result : Actual number of bytes read
 */
uInt STD_CDECL  rdrRead( stdReader_t r, Byte *buffer, uInt amount );


/*--------------------------------- Utilities --------------------------------*/

/*
 * Function        : Wrap reader object around opened file.
 * Parameters      : f       (I)  File handle to wrap.
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateFileReader(FILE *f);


/*
 * Function        : Open the specified file, and return 
 *                   a wrapped reader object around it.
 * Parameters      : name    (I)  Name of file to open.
 * Function Result : Wrapping new reader object, 
 *                   OR Nil in case of error in opening the
 *                   file, in which case an error is issued
 *                   via msgReport.
 */
stdReader_t STD_CDECL rdrCreateFileNameReader( cString name );


/*
 * Function        : Wrap reader object around text string.
 * Parameters      : s       (I)  String to wrap.
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateStringReader(cString s);


/*
 * Function        : Wrap reader object around text string
 * Parameters      : s       (I)  String to wrap.
 *                   size    (I)  Expected string size, ignoring Nils.
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateSizedStringReader(cString s, uInt size);


/*
 * Function        : Wrap deobfuscating reader around existing reader object
 * Parameters      : reader  (I)  Reader object to wrap.
 *                   seed    (I)  Obfuscation seed
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateObfuscatedReader(stdReader_t reader, uInt32 seed);


#ifdef __cplusplus
}
#endif

#endif  /* stdReader_INCLUDED */
