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
|*  Module: extension.h                                                       *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _EXTENSION_H
#define _EXTENSION_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************


//******************************************************************************
//
// class CExtension
//
//******************************************************************************

class CExtension
{
private:
static  CExtension*     m_pFirstExtension;
static  CExtension*     m_pLastExtension;
static  ULONG           m_ulExtensionCount;

        CExtension*     m_pPrevExtension;
        CExtension*     m_pNextExtension;

        CString         m_ExtensionName;
        CString         m_ExtensionPath;
        ULONG64         m_hExtension;
        bool            m_bLoaded;
        bool            m_bExisting;

        void            addExtension(CExtension *pExtension);
        void            delExtension(CExtension *pExtension);






public:
                        CExtension(const char* pszExtensionName);
virtual                ~CExtension();

const   CString&        extensionName() const       { return m_ExtensionName; }
const   CString&        extensionPath() const       { return m_ExtensionPath; }




static  CExtension*     findExtension(const char* pszExtensionName);

static  CExtension*     firstExtension()            { return m_pFirstExtension; }
static  CExtension*     lastExtension()             { return m_pLastExtension; }
static  ULONG           extensionCount()            { return m_ulExtensionCount; }

        CExtension*     prevExtension()             { return m_pPrevExtension; }
        CExtension*     nextExtension()             { return m_pNextExtension; }

}; // class CExtension

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _EXTENSION_H
