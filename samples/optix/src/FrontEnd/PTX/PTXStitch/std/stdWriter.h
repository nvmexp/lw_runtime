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
 *  Module name              : stdWriter.h
 *
 *  Description              :
 *
 */

#ifndef stdWriter_INCLUDED
#define stdWriter_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdString.h"
#include "stdLocal.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types ------------------------------------*/

typedef struct stdWriter  *stdWriter_t;

typedef SizeT (STD_CDECL *wtrWriterFun )( Pointer data, Byte *buffer, SizeT amount );
typedef void (STD_CDECL *wtrResetFun  )( Pointer data );
typedef void (STD_CDECL *wtrCleanupFun)( Pointer data );

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new writer object.
 * Parameters      : write   (I) Writer function to encapsulate
 *                   reset   (I) Reset function to encapsulate
 *                   cleanup (I) Function to cleanup state upon writer deletion, 
 *                                or Nil when not appropriate
 *                   data    (I) Writer state to print to
 * Function Result : Requested (empty) string.
 */
stdWriter_t STD_CDECL wtrCreate( wtrWriterFun write, wtrResetFun reset, wtrCleanupFun cleanup, Pointer data );


/*
 * Function        : Discard writer object.
 * Parameters      : w       (I) Writer to discard, 
 *                               or Nil for trivial stdout writer
 * Function Result : 
 */
void STD_CDECL  wtrDelete( stdWriter_t w );


/*
 * Function        : Reset writer object to set print position to '0'.
 * Parameters      : w       (I)  Writer to reset.
 * Function Result : 
 */
void STD_CDECL  wtrReset( stdWriter_t w );


/*
 * Function        : Write block of data to writer object.
 * Parameters      : w       (I)  Writer to write to.
 *                   buffer  (I)  Buffer to write from.
 *                   amount  (I)  Maximal number of bytes to write.
 * Function Result : Actual number of bytes written
 */
SizeT STD_CDECL  wtrWrite( stdWriter_t w, Byte *buffer, SizeT amount );


/*
 * Function        : Print formatted text to writer object.
 * Parameters      : w       (I)  Writer to print to.
 *                   format  (I)  The 'sprintf' format.
 *                   ...     (I)  Format data.
 * Function Result : Number of characters printed
 */
SizeT STD_CDECL  __CHECK_FORMAT__(printf, 2, 3) wtrPrintf( stdWriter_t w, cString format, ... );


/*
 * Function        : Print formatted text to writer object.
 * Parameters      : w       (I)  Writer to print to.
 *                   format  (I)  The 'sprintf' format.
 *                   arg     (I)  Format data.
 * Function Result : Number of characters printed
 */
SizeT STD_CDECL  wtrVPrintf( stdWriter_t w, cString format, va_list arg );


/*--------------------------------- Utilities --------------------------------*/

/*
 * Function        : Return writer that sinks its output
 * Function Result : Requested new writer object.
 */
stdWriter_t STD_CDECL wtrCreateNullWriter(void);


/*
 * Function        : Wrap writer object around opened file.
 * Parameters      : f       (I)  File handle to wrap.
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateFileWriter(FILE *f);


/*
 * Function        : Open the specified file, and return 
 *                   a wrapped writer object around it.
 * Parameters      : name    (I)  Name of file to open.
 * Function Result : Wrapping new writer object, 
 *                   OR Nil in case of error in opening the
 *                   file, in which case an error is issued
 *                   via msgReport.
 */
stdWriter_t STD_CDECL wtrCreateFileNameWriter( cString name );


/*
 * Function        : Wrap writer object around string.
 * Parameters      : s       (I)  String to wrap.
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateStringWriter(stdString_t s);

/*
 * Function        : Wrap writer object around raw pointer.
 * Parameters      : ptr     (I)  Pointer to wrap.
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateRawWriter(Pointer ptr);

/*
 * Function        : Wrap obfuscating writer around existing writer object
 * Parameters      : writer  (I)  Writer object to wrap.
 *                   seed    (I)  Obfuscation seed
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateObfuscatedWriter(stdWriter_t writer, uInt32 seed);


/*
 * Function        : Wrap tab colwerting writer around existing writer object
 * Parameters      : writer  (I)  Writer object to wrap.
 *                   tablen  (I)  Tab length to be used
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateTabColwWriter(stdWriter_t writer, uInt tablen);


#ifdef __cplusplus
}
#endif

#endif
