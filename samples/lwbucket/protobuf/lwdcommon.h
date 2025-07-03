 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: lwd_common.h                                                      *|
|*                                                                            *|
|*          Common lwdebug-related header.                                    *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _LWD_COMMON_H_
#define _LWD_COMMON_H_

#include "lwtypes.h"

// Return status
typedef LwU32 LWD_STATUS;

// Printf function pointer
typedef int (*LWD_PRINTF)(const char *, ...);
extern LWD_PRINTF lwdPrintf;

// Export settings
#if defined(_WIN32) || defined(_WIN64)
#define LWD_EXPORT __cdecl
#else
#define LWD_EXPORT
#endif

// LWD Error Codes
// 
// When adding a new error code, keep in mind that LWD_LAST_ERROR needs
// to be the last element in the list (and have the same value as 
// its preceeding element)
#define LWD_OK                                            0
#define LWD_ERROR_ILWALID_HANDLE                          1
#define LWD_ERROR_ILWALID_ARGUMENT                        2
#define LWD_ERROR_ILWALID_ZIP_MODE                        3
#define LWD_ERROR_INSUFFICIENT_RESOURCES                  4
#define LWD_ERROR_CRYPT                                   5
#define LWD_ERROR_ZIP                                     6
#define LWD_ERROR_IO                                      7
#define LWD_ERROR_NO_OPEN_FILE                            8
#define LWD_ERROR_NO_OPEN_ZIP_FILE                        9
#define LWD_ERROR_ZIP_FILE_OPEN                          10
#define LWD_ERROR_MALFORMED_ZIP_DATA                     11
#define LWD_ERROR_UNREAD_DATA                            12
#define LWD_ERROR_UNDEFINED_STATUS                       13
#define LWD_ERROR_OS                                     14
#define LWD_LAST_ERROR                                   14

// Common interface
LWD_STATUS  LWD_EXPORT lwdCmn_GetErrorMessage(LWD_STATUS, char **);
void        LWD_EXPORT lwdCmn_SetPrintf();

#endif /* _LWD_COMMON_H_ */
