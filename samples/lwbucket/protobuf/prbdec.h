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
|*  Module: prbdec.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _PRBDEC_H_
#define _PRBDEC_H_

/*****************************************************************************
*
*   Header: prbdec.h
*
*   Description:
*       Lightweight protobuf decoder.
*
*   Revision History:
*       Original -- 10/09 Ben Harris
*
******************************************************************************/

#include "lwdcommon.h"
#include "prbrt.h"

// assert required protobuf options are enabled
#if !PRB_FIELD_DEFAULTS
#error PRB_FIELD_DEFAULTS must be enabled for protobuf decoder
#endif

// Set optional defines to their defaults if not already set.

// Add in optional fields with their default values
#ifndef PRB_ADD_OPTIONAL_DEFAULT_FIELDS
#define PRB_ADD_OPTIONAL_DEFAULT_FIELDS 1
#endif

// Print message length
#ifndef PRB_PRINT_MESSAGE_LENGTH
#define PRB_PRINT_MESSAGE_LENGTH 1
#endif

// Disable custom print routines
#ifndef PRB_LWSTOM_PRINT_ROUTINES
#define PRB_LWSTOM_PRINT_ROUTINES 0
#endif

// runtime protobuf structures
typedef struct
{
    const LwU8 *base;
    const LwU8 *pos;
    const LwU8 *end;
} PRB_DECODE_BUF;

typedef struct
{
    const PRB_FIELD_DESC *desc;
    PRB_VALUE *values;
    LwU32 count;
} PRB_FIELD;

typedef struct
{
    const PRB_MSG_DESC *desc;
    PRB_FIELD *fields;
    LwU32 mergedMsgLen;
} PRB_MSG;

// core decode API
PRB_STATUS          LWD_EXPORT prbCreateMsg(PRB_MSG *pMsg, const PRB_MSG_DESC *pMsgDesc);
void                LWD_EXPORT prbDestroyMsg(PRB_MSG *pMsg);
PRB_STATUS          LWD_EXPORT prbDecodeMsg(PRB_MSG *pMsg, const void *data, LwU32 length);
const PRB_FIELD *   LWD_EXPORT prbGetField(const PRB_MSG *pMsg, const PRB_FIELD_DESC *pFieldDesc);
#if PRB_FIELD_NAMES
const PRB_FIELD *   LWD_EXPORT prbGetFieldByName(const PRB_MSG *pMsg, const char *fieldName);
#endif

// printing utiltities
#if PRB_MESSAGE_NAMES && PRB_FIELD_NAMES
void                LWD_EXPORT prbPrintMsg(const PRB_MSG *pMsg, LwU32 indentLevel);
void                LWD_EXPORT prbPrintMsgOutline(const PRB_MSG *pMsg, LwU32 indentLevel);
void                LWD_EXPORT prbPrintField(const PRB_FIELD *pField, const LwU32 *pIndex, LwU32 indentLevel);
#endif
const char *        LWD_EXPORT prbGetEnumValueName(const PRB_ENUM_DESC *pEnumDesc, int value);

// debug settings
void                LWD_EXPORT prbSetDebugPrintFlag(LwBool bDbgPrint);

// utilities for decoding
#if PRB_MESSAGE_NAMES
const PRB_MSG_DESC* LWD_EXPORT prbGetMsgDescByName(const PRB_MSG_DESC **msgDescs, const char *name);
#endif
const PRB_MSG *     LWD_EXPORT prbGetMsg(const PRB_MSG *pMsg, const PRB_MSG_DESC *pMsgDesc);
const PRB_MSG *     LWD_EXPORT prbGetMsgFromField(const PRB_FIELD *pField, const PRB_MSG_DESC *pMsgDesc);

// prototypes for custom print routines
#if PRB_LWSTOM_PRINT_ROUTINES
LwBool LWD_EXPORT lwstomPrintMsg(const PRB_MSG *pMsg, LwU32 indentLevel);
LwBool LWD_EXPORT lwstomPrintField(const PRB_FIELD *pField, const LwU32 *pIndex, LwU32 indentLevel);
#endif // PRB_LWSTOM_PRINT_ROUTINES

#endif // _PRBDEC_H_
