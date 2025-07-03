/*
 * Copyright (c) 2007 - 2009 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __STR_UTIL_H__
#define __STR_UTIL_H__

#include <stdarg.h>
#include "printf_like.h"

// Not using this yet, but we might want it some day
#ifndef __scanflike
# if __GNUC__ > 2 || __GNUC__ == 2 && __GNUC_MINOR__ >= 7
#  define __scanflike(fmtarg, firstvararg) \
          __attribute__((__format__ (__scanf__, fmtarg, firstvararg)))
# else
#  define __scanflike(fmtarg, firstvararg)
# endif
#endif

// Print the formatted output to the given buffer buf
// size is the total length of the buffer
//
// If the appended string is too long to fit in the buffer including the null
// terminated string, -1 is returned.  Otherwise, the number of characters
// written is returned.
//
// Buf is guaranteed to be null terminated after this call returns.
int lwog_vsnprintf(char *buf, size_t size, const char *fmt, va_list args);
int lwog_snprintf(char *buf, size_t size, const char *fmt, ...) __printflike(3,4);

// Catenate as much of the src string as will fit in dest
// size is the total length of the dest buffer
//
// This differs from the standard strncat where n specifies an upper-bound on
// the number of characters to be copied from src.
//
// Returns a pointer to the dest buffer.
//
// Buf is guaranteed to be null terminated after this call returns, just like
// for strncat().
char *lwog_strncat(char *dest, char *src, size_t size);

// Safe replacement for strcat.
//
// Calls strncat with size, but asserts if strlen(src) >= size
char *lwog_strcat(char *dest, size_t size, const char *src);

// Catenate the formatted output to the given buffer buf, which must already
// contain a valid null-terminated string.
// size is the total length of the buffer
//
// If the appended string is too long to fit in the buffer including the null
// terminated string, -1 is returned.  Otherwise, the number of new characters
// is returned.
//
// Buf is guaranteed to be null terminated after this call returns.
int vstrncatf(char *buf, size_t size, const char *fmt, va_list args);
int strncatf(char *buf, size_t size, const char *fmt, ...) __printflike(3,4);

void lwogOutputDebugString(const char *str);
// Output the given format and arguments printf-wise to the debugger
int vdebugprintf(const char *fmt, va_list args);
int debugprintf(const char *fmt, ...) __printflike(1,2);

// Output to both stdout (and flush), also output to debugger
int vxdebugprintf(int level, const char *fmt, va_list args);
int xdebugprintf(int level, const char *fmt, ...) __printflike(2,3);

// Decode the given exception code into a string
const char *getExceptionString(unsigned int code);

// Stick the commandline arguments (including the exe as argv[0]) into a
// single string.
//
// Returns a string allocated via __LWOG_MALLOC.  This string should be freed
// via __LWOG_FREE when no longer needed.
char *makeCmdlineString(int argc, char** argv);

// SBuilder:  Structure for incrementally adding to a string with overflow 
// detection.
typedef struct SBuilder {
    char *buffer;
    size_t size;
    size_t maxsize;
} SBuilder;

// Initialize the string builder to have an empty buffer at <buffer> with size
// <maxsize>.
void sbuilderinit(SBuilder *string, char *buffer, size_t maxsize);

// Do a printf to add to the end of the current string buffer.  If the
// provided buffer, that is treated as a test bug and an assert will fire.
void sbuilderprint(SBuilder *string, const char *fmt, ...);

#endif // #ifdef __STR_UTIL_H__
