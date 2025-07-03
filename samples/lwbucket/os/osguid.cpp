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
|*  Module: osguid.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#include "osprecomp.h"

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// Locals
//
//******************************************************************************
// GUID Type Helpers
CMemberType     CGuid::m_GuidType                           (&osKernel(), "GUID", "_GUID");

// GUID Field Helpers
CMemberField    CGuid::m_Data1Field                         (&GuidType(), false, NULL, "Data1");
CMemberField    CGuid::m_Data2Field                         (&GuidType(), false, NULL, "Data2");
CMemberField    CGuid::m_Data3Field                         (&GuidType(), false, NULL, "Data3");
CMemberField    CGuid::m_Data4Field                         (&GuidType(), false, NULL, "Data4");

//******************************************************************************

CGuid::CGuid
(
    POINTER             ptrGuid
)
:   m_ptrGuid(ptrGuid),
    INIT(Data1),
    INIT(Data2),
    INIT(Data3),
    INIT(Data4)
{
    // Get the GUID information
    READ(Data1, ptrGuid);
    READ(Data2, ptrGuid);
    READ(Data3, ptrGuid);
    READ(Data4, ptrGuid);

} // CGuid

//******************************************************************************

CGuid::~CGuid()
{

} // ~CGuid

//******************************************************************************

CString
CGuid::guidString() const
{
    CString             sGuid(MAX_GUID_STRING);

    // Format and return the GUID string {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
    sGuid.sprintf("{%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}",
                  Data1(), Data2(), Data3(), Data4(0), Data4(1), Data4(2), Data4(3), Data4(4), Data4(5), Data4(6), Data4(7));

    return sGuid;

} // guidString

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
