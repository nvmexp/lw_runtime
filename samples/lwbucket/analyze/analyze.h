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
|*  Module: analyze.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _ANALYZE_H
#define _ANALYZE_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define ILWALID_PHASE                                   (static_cast<FA_EXTENSION_PLUGIN_PHASE>(0))
#define LWIDIA_TAG_START                                (DEBUG_FLR_LWSTOM_ANALYSIS_TAG_MIN + 0x08000000)

//******************************************************************************
//
//  Structures
//
//******************************************************************************
typedef struct _TAG_ENTRY
{
    int                 tag;
    const char         *szTagName;
    const char         *szTagDescription;

} TAG_ENTRY, *PTAG_ENTRY;

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  HRESULT         setTagString(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, FA_TAG tag, CString sString, CString sName, CString sDesc);
extern  HRESULT         setTagUlong(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, FA_TAG tag, ULONG ulValue, CString sName, CString sDesc);
extern  HRESULT         setTagUlong64(PDEBUG_FAILURE_ANALYSIS2 pAnalysis, IDebugFAEntryTags* pTagControl, FA_TAG tag, ULONG64 ulValue, CString sName, CString sDesc);

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _ANALYZE_H
