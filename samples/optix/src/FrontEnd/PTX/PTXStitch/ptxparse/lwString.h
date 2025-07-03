// **************************************************************************
//
//       Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
//
//     NOTICE TO USER:   The source code  is copyrighted under  U.S. and
//     international laws.  Users and possessors of this source code are
//     hereby granted a nonexclusive,  royalty-free copyright license to
//     use this code in individual and commercial software.
//
//     Any use of this source code must include,  in the user dolwmenta-
//     tion and  internal comments to the code,  notices to the end user
//     as follows:
//
//       Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
//
//     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY
//     OF  THIS SOURCE  CODE  FOR ANY PURPOSE.  IT IS  PROVIDED  "AS IS"
//     WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPOR-
//     ATION DISCLAIMS ALL WARRANTIES  WITH REGARD  TO THIS SOURCE CODE,
//     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGE-
//     MENT,  AND FITNESS  FOR A PARTICULAR PURPOSE.   IN NO EVENT SHALL
//     LWPU, CORPORATION  BE LIABLE FOR ANY SPECIAL,  INDIRECT,  INCI-
//     DENTAL, OR CONSEQUENTIAL DAMAGES,  OR ANY DAMAGES  WHATSOEVER RE-
//     SULTING FROM LOSS OF USE,  DATA OR PROFITS,  WHETHER IN AN ACTION
//     OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF
//     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.
//
//     U.S. Government  End  Users.   This source code  is a "commercial
//     item,"  as that  term is  defined at  48 C.F.R. 2.101 (OCT 1995),
//     consisting  of "commercial  computer  software"  and  "commercial
//     computer  software  documentation,"  as such  terms  are  used in
//     48 C.F.R. 12.212 (SEPT 1995)  and is provided to the U.S. Govern-
//     ment only as  a commercial end item.   Consistent with  48 C.F.R.
//     12.212 and  48 C.F.R. 227.7202-1 through  227.7202-4 (JUNE 1995),
//     all U.S. Government End Users  acquire the source code  with only
//     those rights set forth herein.
//
// **************************************************************************
//
//  Module: lwString.h
//      our very own string utils (necessary because the analogs
//      don't exist for kernel mode drivers. ugh)
//
// **************************************************************************
//
//  History:
//      Craig Duttweiler    February 2005   broken out of lwUtil.c
//
// **************************************************************************

#ifndef _LWSTRING_H_
#define _LWSTRING_H_

// strupr needs the 'POSIX' name in Windows, the
// normal one otherwise
#if defined(LW_WINDOWS)
#define lwStrUpr                    _strupr
#else // defined(LW_WINDOWS)
#define lwStrUpr                    strupr
#endif // defined(LW_WINDOWS)

#if defined(LW_PARSEASM) || defined(LW_GLSLC) || defined(LW_IN_CGC)

#define lwStrCat                    strcat
#define lwStrCmp                    strcmp
#define lwStrCaseCmp                _stricmp
#define lwStrCpy                    strcpy
#define lwStrNCpy                   strncpy
#define lwStrLen                    strlen
#define lwSprintf                   sprintf
#define lwSprintfVAL(X, Y, Z)       vsprintf((X), (Y), (Z))
#define lwWStrCat                   wcscat
#define lwWStrLen                   wcslen
#define lwWStrCpy                   wcscpy

#elif defined(IS_OPENGL)

// This is added here to play nicely with module branching
#if !defined(__GL_WSTRCAT)
#define __GL_WSTRCAT(X, Y)          (wcscat((X), (Y)))
#endif // !defined(__GL_WSTRCAT)

#if !defined(__GL_WSTRCPY)
#define __GL_WSTRCPY(X, Y)          (wcscpy((X), (Y)))
#endif // !defined(__GL_WSTRCPY)

#if !defined(__GL_WSTRLEN)
#define __GL_WSTRLEN(X)             (wcslen((X)))
#endif // !defined(__GL_WSTRLEN)

#define lwStrCat(X, Y)              __GL_STRCAT((X), (Y))
#define lwStrCmp(X, Y)              __GL_STRCMP((X), (Y))
#define lwStrCaseCmp(X, Y)          __GL_STRCASECMP((X), (Y))
#define lwStrCpy(X, Y)              __GL_STRCPY((X), (Y))
#define lwStrNCpy(X, Y, C)          __GL_STRNCPY((X), (Y), (C))
#define lwStrLen(X)                 __GL_STRLEN((X))
#define lwSprintf                   __GL_SPRINTF
#define lwSprintfVAL(X, Y, Z)       __GL_VSPRINTF((X), (Y), (Z))
#define lwWStrCat(X, Y)             __GL_WSTRCAT((X), (Y))
#define lwWStrLen(X)                __GL_WSTRLEN((X))
#define lwWStrCpy(X, Y)             __GL_WSTRCPY((X), (Y))

#else  // D3D

#ifdef __cplusplus
    #define LWSTR_EXTERN_C    extern "C"
#else
    #define LWSTR_EXTERN_C    extern
#endif

LWSTR_EXTERN_C int   LW_CDECLCALL lwStrCmp     (const char *szStr1, const char *szStr2);
LWSTR_EXTERN_C int   LW_CDECLCALL lwStrCaseCmp (const char *szStr1, const char *szStr2);
LWSTR_EXTERN_C int   LW_CDECLCALL lwStrNCmp    (const char *szStr1, const char *szStr2, int n);
LWSTR_EXTERN_C int   LW_CDECLCALL lwStrLen     (const char *szStr);
LWSTR_EXTERN_C int   LW_CDECLCALL lwStrNLen    (const char *szStr, int n);
LWSTR_EXTERN_C char* LW_CDECLCALL lwStrCpy     (char *szDst, const char *szSrc);
LWSTR_EXTERN_C void  LW_CDECLCALL lwStrNCpy    (char *szDst, const char *szSrc, size_t n);
LWSTR_EXTERN_C char* LW_CDECLCALL lwStrCat     (char *szStr1, const char *szStr2);
LWSTR_EXTERN_C char* LW_CDECLCALL lwStrChr     (const char *szStr, LwU8 c);
LWSTR_EXTERN_C char* LW_CDECLCALL lwSubStr     (const char *szStr, const char *szSubStr);
LWSTR_EXTERN_C char* LW_CDECLCALL lwStrRChr    (const char *szStr, LwU8 c);
LWSTR_EXTERN_C int   LW_CDECLCALL lwSprintfVAL (char *szDest, const char *szFormat, va_list vaArgs);
LWSTR_EXTERN_C int   LW_CDECLCALL lwSprintf    (char *szDest, const char *szFormat, ...);
LWSTR_EXTERN_C int   LW_CDECLCALL lwStrMatch   (const char *s, const char *mask);

LWSTR_EXTERN_C wchar_t* LW_CDECLCALL lwWStrCat (wchar_t *wszStr1, const wchar_t *wszStr2);
LWSTR_EXTERN_C int      LW_CDECLCALL lwWStrLen (const wchar_t *wszStr);
LWSTR_EXTERN_C wchar_t* LW_CDECLCALL lwWStrCpy  (wchar_t *wszDst, const wchar_t *wszSrc);

#endif  // parseasm/ogl/d3d

#endif  // _LWSTRING_H_

