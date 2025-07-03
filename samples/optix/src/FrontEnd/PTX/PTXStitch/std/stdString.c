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
 *  Module name              : stdString.c
 *
 *  Description              :
 *
 *  Implementation of an abstract type 'string', based upon a list
 *  of small blocks. strings do not run out of space while stdMALLOC
 *  returns non-zero.
 */

/*------------------------------- Includes -----------------------------------*/

#include "stdString.h"
#include "stdList.h"
#include "stdLocal.h"

/*---------------------------------- Types -----------------------------------*/

typedef struct stdSBucket *stdSBucket_t;

struct stdSBucket {
    SizeT         size;
    SizeT         bytesLeft;
    String        buf;
};


struct stdString {
    SizeT        szofBuckets;
    SizeT        size;
    listXList    (buckets);
    stdSBucket_t  lastBucket;
};


/*---------------------------- Bucket functions ------------------------------*/

static stdSBucket_t bucketNew(SizeT size)
{
    stdSBucket_t result;

    stdNEW(result);

    result->size      =
    result->bytesLeft = size;
    result->buf       = stdMALLOC(size);

    return result;
}

static void STD_CDECL bucketDelete(stdSBucket_t bucket)
{
    stdFREE( bucket->buf );
    stdFREE( bucket );
}

static void STD_CDECL bucketToBuf(stdSBucket_t bucket, String* buf)
{
    SizeT toCopy = bucket->size - bucket->bytesLeft;

    stdMEMCOPY_S(*buf, bucket->buf, toCopy);

    *buf += toCopy;
}


/*---------------------------- String functions ------------------------------*/

/*
 * Function        : Create new string
 * Parameters      : szofBuckets  (I) Minimal size of memory blocks
 * Function Result : Requested (empty) string
 */
stdString_t STD_CDECL stringCreate ( SizeT szofBuckets )
{
    stdString_t result;

    stdNEW(result);

    result->szofBuckets  = szofBuckets;
    result->size         = 0;
    result->lastBucket   = Nil;

    listXInit(result->buckets);

    return result;
}



/*
 * Function        : Discard string
 * Parameters      : string  (I) string to discard
 * Function Result : -
 */
void STD_CDECL  stringDelete ( stdString_t string )
{
    if (string->buckets != Nil) {
        listTraverse( string->buckets, (stdEltFun)bucketDelete, Nil );
        listDelete  ( string->buckets );
    }
    stdFREE( string );
}



/*
 * Function        : Make string empty
 * Parameters      : string  (I) string to empty
 * Function Result : -
 */
void STD_CDECL  stringEmpty ( stdString_t string )
{
    if (string->buckets != Nil) {
        listTraverse( string->buckets, (stdEltFun)bucketDelete, Nil );
        listDelete  ( string->buckets );
    }
    string->size       = 0;
    string->lastBucket = Nil;

    listXInit(string->buckets);
}



/*
 * Function        : Append a string with a prefix of a C string
 * Parameters      : string      (IO) String to append to
 *                   buf         (I)  its donor; len characters will
 *                                    be added to string, none of
 *                                    which should be null
 *                   len         (I)  the number of characters from
 *                                    buf to be added
 * Function Result : -
 */
void STD_CDECL  stringAddBufLen( stdString_t string, cString buf, SizeT len )
{
    stdSBucket_t toBucket = string->lastBucket;

    if (toBucket != Nil) {

        SizeT toCopy    = stdMIN(len, toBucket->bytesLeft);
        SizeT bufOffset = toBucket->size - toBucket->bytesLeft;

        stdMEMCOPY_S(toBucket->buf + bufOffset, buf, toCopy);

        toBucket->bytesLeft -= toCopy;
        buf                 += toCopy;
        len                 -= toCopy;
        string->size        += toCopy;
    }

    if (len > 0) {

        toBucket = bucketNew(stdMAX(len, string->szofBuckets));

        stdMEMCOPY_S (toBucket->buf, buf, len);
        listXPutAfter(string->buckets, toBucket);

        toBucket->bytesLeft -= len;
        string->size        += len;
        string->lastBucket   = toBucket;
    }
}



/*
 * Function        : Returns a C string represention of string
 * Parameters      : string     (I) string to make into C string
 * Function Result : A C string created with stdMALLOC. Caller is
 *                   responsible for freeing memory.
 */
String STD_CDECL  stringToBuf( stdString_t string )
{
    String buf = stdMALLOC(string->size + 1);

    listTraverse(string->buckets, (stdEltFun)bucketToBuf, &buf);
    *buf = 0;

    return buf - string->size;
}



/*
 * Function        : Returns a C string represention of string,
 *                   and delete string itself.
 *                   This function is a colwenient shorthand for
 *                   the combination stringToBuf and stringDelete.
 * Parameters      : string     (I) string to make into C string
 * Function Result : A C string created with stdMALLOC. Caller is
 *                   responsible for freeing memory.
 */
String STD_CDECL stringStripToBuf( stdString_t string )
{
    String result= stringToBuf(string);
    stringDelete(string);
    return result;
}



/*
 * Function        : Append a string with a C string
 * Parameters      : string      (IO) String to append to
 *                   buf         (I)  its donor
 * Function Result : -
 */
void STD_CDECL  stringAddBuf( stdString_t string, cString buf )
{
    stringAddBufLen(string, buf, strlen(buf));
}



/*
 * Function        : Append a string with a single character
 * Parameters      : string      (IO) String to append to
 *                   c           (I)  its donor
 * Function Result : -
 */
void STD_CDECL  stringAddChar( stdString_t string, Char c )
{
    stringAddBufLen(string, &c, 1);
}



/*
 * Function        : Append a string with formatted text.
 * Parameters      : string      (IO) String to append to
 *                   format      (I)  'sprintf' format
 *                   ...         (I)  its donor
 * Function Result : Number of characters added to string
 */
SizeT STD_CDECL  stringAddFormat( stdString_t string, cString format, ... )
{
    SizeT     result;
    va_list  ap;
    
    va_start (ap, format);

    result= stringAddVFormat(string,format,ap);

    va_end (ap);  
    
    return result;
}



/*
 * Function        : Append a string with formatted text.
 * Parameters      : string      (IO) String to append to.
 *                   format      (I)  The 'sprintf' format.
 *                   arg         (I)  Its donor.
 * Function Result : Number of characters added to string
 */

    #define DEF_SIZE 1024

SizeT STD_CDECL  stringAddVFormat( stdString_t string, cString format, va_list arg )
{
    Char  defaultBuffer[DEF_SIZE];
    Char *buffer         = defaultBuffer;
    SizeT bufferCapacity = DEF_SIZE;
    Int   formatted;

    #ifdef STD_OS_win32
    formatted = vsnprintf(buffer, bufferCapacity, format, arg);
    if (formatted == -1) {
        formatted = _vscprintf(format, arg);
    }
    #else
    {
       /*
        * NOTE: in gcc/glibc, vsnprintf overwrites 'arg'
        */
        va_list argCopy;
        va_copy(argCopy, arg);
        formatted = vsnprintf(buffer, bufferCapacity, format, argCopy);
        va_end(argCopy);
    }
    #endif

    if (formatted >= bufferCapacity) {
        bufferCapacity = formatted + 1;
        buffer         = stdMALLOC(bufferCapacity);
        formatted      = vsprintf(buffer, format, arg);
    }

    stringAddBufLen(string, buffer, (SizeT)formatted);

    if (buffer != defaultBuffer) { stdFREE(buffer); }

    return (SizeT)formatted;
}



/*
 * Function        : Concatenate two strings
 * Parameters      : toString    (IO) String to append to
 *                   fromString  (I)  its donor
 * Function Result : -
 */
void STD_CDECL  stringAddString( stdString_t toString, stdString_t fromString )
{
    String s = stringToBuf(fromString);

    stringAddBufLen(toString, s, fromString->size);
    stdFREE(s);
}




/*
 * Function        : Copy a string
 * Parameters      : string      (I) string to be copied
 * Function Result : The copied string
 */
stdString_t STD_CDECL stringCopy( stdString_t string )
{
    stdString_t result = stringNEW( );

    stringAddString(result, string);

    return result;
}



/*
 * Function        : Size of string
 * Parameters      : string  (I)  string to size
 * Function Result : string size (without '\0')
 */
SizeT STD_CDECL  stringSize( stdString_t string )
{
    return string->size;
}
