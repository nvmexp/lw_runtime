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
|*  Module: helper.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _HELPER_H
#define _HELPER_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define MAX_TYPE_STRING                 128                 // Maximum type string
#define MAX_NAME_STRING                 256                 // Maximum name string
#define MAX_DESCRIPTION_STRING          1024                // Maximum description string

#define ILWALID_INDEX                   (static_cast<ULONG>(-1))

#define ILWALID_PROCESS_ADDRESS         0xffffffffffffffff  // Invalid process address value
#define ILWALID_THREAD_ADDRESS          0xffffffffffffffff  // Invalid thread address value

#define ILWALID_BOOLEAN_VALUE           0xffffffffffffffff  // Invalid boolean value
#define ILWALID_VERBOSE_VALUE           0xffffffffffffffff  // Invalid verbose value

#define DEBUG_MODULE_NAME               "dbgeng.dll"        // Module used to determine debugger version

//******************************************************************************
//
//  Structures
//
//******************************************************************************
typedef struct _BIT_TABLE               // Bit table structure (Matching bit fields)
{
    char       *pString;                // Pointer to string value
    ULONG64     ulValue;                // Bit value for matching string

} BIT_TABLE, *PBIT_TABLE;

typedef struct _BOOLEAN_ENTRY           // Boolean table entry structure
{
    char       *pString;                // Pointer to boolean string value
    bool        bValue;                 // Boolean data value

} BOOLEAN_ENTRY, *PBOOLEAN_ENTRY;

typedef struct _VERBOSE_ENTRY           // Verbose table entry structure
{
    char       *pString;                // Pointer to verbose string value
    ULONG64     ulValue;                // Verbose data value

} VERBOSE_ENTRY, *PVERBOSE_ENTRY;

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  bool            isLowercase(const char* pString);
extern  bool            isUppercase(const char* pString);

extern  bool            isRegularExpression(const char* pString);
extern  CString         getRegularExpression(const char* pString);

extern  bool            isStringExpression(const char* pString);
extern  CString         getStringExpression(const char* pString);

extern  bool            getEnableValue(bool bEnable = true);
extern  bool            getDisableValue(bool bDisable = false);

extern  const CString&  getBoolString(bool bBool);
extern  const CString&  getEnabledString(bool bEnable);
extern  const CString&  getErrorString(bool bError);

extern  float           getTimeValue(float fTime);
extern  const CString&  getTimeUnit(float fTime);
extern  const CString&  getTimeColor(float fTime);

extern  float           getFreqValue(float fFreq);
extern  const CString&  getFreqUnit(float fFreq);

extern  float           getPwrValue(float fPwr);
extern  const CString&  getPwrUnit(float fPwr);

extern  float           getSizeValue(float fSize);
extern  const CString&  getSizeUnit(float fSize);
extern  const CString&  getSizeColor(float fSize);

extern  const CString&  getDataTypeString(DataType dataType);

extern  ULONG64         getDebuggerVersion();

extern  CString         getTemporaryPath();

extern  CString         subExpression(const char* pString, const regmatch_t* pRegMatch, ULONG ulSubexpression);

extern  CString         centerString(const CString& sString, ULONG ulWidth);
extern  void            headerString(const CString& sString, ULONG ulWidth, CString& sHeader, CString& sDash, ULONG ulSpacing = 0);
extern  void            titleString(const CString& sString, ULONG ulWidth, CString& sTitle, ULONG ulSpacing = 0);

extern  ULONG           fieldWidth(const CMemberField& memberField);
extern  ULONG           memberWidth(const CMember& member);

extern  bool            fileExists(const char* pFile);

extern  ULONG64         elapsedTime(ULONG64 ulStartTime, ULONG64 ulEndTime);

extern  ULONG64         extractBitfield(ULONG64 ulValue, ULONG ulPosition, ULONG ulWidth);

extern  ULONG64         boolealwalue(const char* pBooleanString);
extern  ULONG64         verboseValue(const char* pVerboseString);

extern  ULONG64         factorial(ULONG ulValue);

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _HELPER_H
