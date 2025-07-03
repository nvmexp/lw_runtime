/*
 * Copyright (c) 2002 - 2011 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include <assert.h>
#include <stdlib.h>
#include "str_util.h"
#include <time.h>
#include <string.h>
#include <stdio.h>
#if defined(LW_LINUX) || defined(LW_MACINTOSH_OSX) || defined(LW_SUNOS)
#include <unistd.h>
#endif
#include "cmdline.h"

// Utility string functions

//========================================================================
// snprintf - internal helper defines
//========================================================================

// the MS versions of these act like the old glibc versions - if the resulting
// string is too long to fit, the result is truncated, without a trailing
// null, and a value <0 is returned.
// (The newer glibc version return the length that would have been written,
// allowing one to pass NULL in as the buffer and then dynamically allocate an
// appropriately sized buffer.)
#if defined(_WIN32)
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#endif

//========================================================================
// lwog_snprintf - sanitized sprintf
//========================================================================

// Print the formatted output to the given buffer buf
// size is the total length of the buffer
//
// If the appended string is too long to fit in the buffer including the null
// terminated string, -1 is returned.  Otherwise, the number of characters
// written is returned.
//
// Buf is guaranteed to be null terminated after this call returns.
int lwog_vsnprintf(char *buf, size_t size, const char *fmt, va_list args)
{
    int ret = vsnprintf(buf, size, fmt, args);

    if (ret >= (int)size || ret < 0) {
        // We've overflowed.  Replace the last byte with a null terminator.
        buf[size - 1] = 0;
        // signal overflow uniformly for both Windows and Glibc
        ret = -1;
    }

    return ret;
}

int lwog_snprintf(char *buf, size_t size, const char *fmt, ...)
{
    int ret;

    va_list args;
    va_start(args, fmt);
    ret = lwog_vsnprintf(buf, size, fmt, args);
    va_end(args);

    return ret;
}

//========================================================================
// lwog_strncat - sanitized strncat
//========================================================================

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
char *lwog_strncat(char *dest, char *src, size_t size)
{
    // count less 1 to leave space for the null terminator
    return strncat(dest, src, size - strlen(dest) - 1);
}

// Different from strncat because the size of the destination buffer is the 2nd, not 3rd parameter.
// This parameter makes it easier to fix an existing strcat call.
char *lwog_strcat(char *dest, size_t size, const char *src)
{
    size_t bytesLeft = (size - 1) - strlen(dest);  // -1 for null terminator
    return strncat(dest, src, bytesLeft);
}

//========================================================================
// strncatf - safe concatenating sprintf
//========================================================================

// Catenate the formatted output to the given buffer buf, which must already
// contain a valid null-terminated string.
// size is the total length of the buffer
//
// If the appended string is too long to fit in the buffer including the null
// terminated string, -1 is returned.  Otherwise, the number of new characters
// is returned.
//
// Buf is guaranteed to be null terminated after this call returns.
int vstrncatf(char *buf, size_t size, const char *fmt, va_list args)
{
    size_t startlen = strlen(buf);
    size_t avail = size - startlen;  // including space for null termination

    return lwog_vsnprintf(buf + startlen, avail, fmt, args);
}

int strncatf(char *buf, size_t size, const char *fmt, ...)
{
    int ret;

    va_list args;
    va_start(args, fmt);
    ret = vstrncatf(buf, size, fmt, args);
    va_end(args);

    return ret;
}


