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
|*  Module: annotation.cpp                                                    *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

CString
getAnnotation
(
    POINTER             ptrAddress
)
{
    CString             sAnnotation;
    WCHAR               wBuffer[1024];
    ULONG               ulAnnotationSize;
    ULONG               ulAnnotationLength;
    ULONG               ulStringLength;
    ULONG               ulAnnotationPosition;
    HRESULT             hResult;

    // Try to get the annotation string(s) for the given address (Annotation address is sign extended)
    hResult = GetSymbolInformation(DEBUG_SYMINFO_GET_SYMBOL_NAME_BY_OFFSET_AND_TAG_WIDE,
                                   ptrAddress.pointer(),
                                   SymTagAnnotation,
                                   wBuffer,
                                   sizeof(wBuffer),
                                   &ulAnnotationSize,
                                   NULL,
                                   0,
                                   NULL);
    if (SUCCEEDED(hResult))
    {
        // Loop computing the length of the annotation ANSI string(s)
        ulAnnotationLength   = 0;
        ulAnnotationPosition = 0;
        while (ulAnnotationPosition < (ulAnnotationSize / sizeof(WCHAR)))
        {
            // Get length of the next annotation UNICODE string (Includes terminator)
            ulStringLength = WideCharToMultiByte(CP_UTF8, 0, &wBuffer[ulAnnotationPosition], -1, 0, 0, NULL, NULL);
            if (ulStringLength > 1)
            {
                // Check for not the first UNICODE string (Needs separator)
                if (ulAnnotationPosition != 0)
                {
                    // Update the annotation length and position
                    ulAnnotationLength   += ulStringLength;
                    ulAnnotationPosition += ulStringLength;
                }
                else    // First UNICODE string (No separator)
                {
                    // Update the annotation length and position
                    ulAnnotationLength   += ulStringLength - 1;
                    ulAnnotationPosition += ulStringLength;
                }
            }
            else    // No more UNICODE strings
            {
                // Exit the length loop
                break;
            }
        }
        // Check for annotation present
        if (ulAnnotationLength != 0)
        {
            // Resize string large enough to hold the ANSI string
            sAnnotation.resize(ulAnnotationLength + 1);

            // Loop colwerting annotation UNICODE string(s) to ANSI
            ulAnnotationPosition = 0;
            while (ulAnnotationPosition < (ulAnnotationSize / sizeof(WCHAR)))
            {
                // Check for not the first UNICODE string (Needs separator)
                if (ulAnnotationPosition != 0)
                {
                    // Append blank separator onto ANSI string
                    sAnnotation.append(1, ' ');
                }
                // Colwert next UNICODE annotation string to ANSI
                ulStringLength = WideCharToMultiByte(CP_UTF8, 0, &wBuffer[ulAnnotationPosition], -1, sAnnotation.data(ulAnnotationPosition), ulAnnotationLength, NULL, NULL);
                if (ulStringLength != 0)
                {
                    // Update the annotation position
                    ulAnnotationPosition += ulStringLength;
                }
                else    // Error colwerting next UNICODE string
                {
                    // Exit the colwersion loop
                    break;
                }
            }
        }
    }
    return sAnnotation;

} // getAnnotation

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
