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
|*  Module: dbginput.cpp                                                      *|
|*                                                                            *|
 \****************************************************************************/
#include "dbgprecomp.h"

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
// Undefine any interface macros for routines that this code will declare
//
//******************************************************************************


//******************************************************************************
//
//  CLwstomDebugInputCallbacks Interface
//
//******************************************************************************

STDMETHODIMP
CLwstomDebugInputCallbacks::QueryInterface
(
    THIS_
    __in  REFIID        InterfaceId,
    __out PVOID*        Interface
)
{
    HRESULT             hResult = E_NOINTERFACE;

    assert(Interface != NULL);

    if (IsEqualIID(InterfaceId , __uuidof(IUnknown)) || IsEqualIID(InterfaceId , __uuidof(IDebugInputCallbacks)))
    {
        *Interface = (IDebugInputCallbacks *)this;
        AddRef ( ) ;

        hResult = S_OK;
    }
    return hResult;

} // QueryInterface

//******************************************************************************

STDMETHODIMP_(ULONG)
CLwstomDebugInputCallbacks::AddRef
(
    THIS
)
{
    // Increment and return the new interface reference count
    return InterlockedIncrement(&m_lRefCount);

} // AddRef

//******************************************************************************

STDMETHODIMP_(ULONG)
CLwstomDebugInputCallbacks::Release
(
    THIS
)
{
    LONG                lRefCount;

    // Decrement the interface reference count
    lRefCount = InterlockedDecrement(&m_lRefCount);

    // Free the interface if no longer referenced
    if (lRefCount == 0)
    {
        delete this;
    }
    return lRefCount;

} // Release

//******************************************************************************

STDMETHODIMP
CLwstomDebugInputCallbacks::StartInput
(
    THIS_
    __in  ULONG         BufferSize
)
{
    UNREFERENCED_PARAMETER(BufferSize);

    HRESULT             hResult = S_OK;





    return hResult;

} // StartInput

//******************************************************************************

STDMETHODIMP
CLwstomDebugInputCallbacks::EndInput
(
    THIS_
)
{
    HRESULT             hResult = S_OK;





    return hResult;

} // EndInput

//******************************************************************************

CLwstomDebugInputCallbacks::CLwstomDebugInputCallbacks()
:   m_lRefCount(0),
    m_pDebugInputCallbacks(NULL)
{
    HRESULT             hResult;

    // Try to get existing debug input callbacks
    hResult = GetInputCallbacks(&m_pDebugInputCallbacks);
    if (SUCCEEDED(hResult))
    {
        // Try to setup the custom debug input callbacks
        hResult = SetInputCallbacks(funcptr(PDEBUG_INPUT_CALLBACKS, this));
    }

} // CLwstomDebugInputCallbacks

//******************************************************************************

CLwstomDebugInputCallbacks::~CLwstomDebugInputCallbacks()
{
    // Restore the original input callbacks (ignore any errors)
    SetInputCallbacks(m_pDebugInputCallbacks);

} // ~CLwstomDebugInputCallbacks

//******************************************************************************
//
// Custom Debug Input Interface wrappers
//
//******************************************************************************

HRESULT
StartInput
(
    ULONG               BufferSize
)
{
    PLWSTOM_DEBUG_INPUT_CALLBACKS pDbgInput = dbgInput();
    HRESULT             hResult = E_FAIL;

    // Call the StartInput debug input method (If created)
    if (pDbgInput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        hResult = pDbgInput->StartInput(BufferSize);

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // StartInput

//******************************************************************************

HRESULT
EndInput()
{
    PLWSTOM_DEBUG_INPUT_CALLBACKS pDbgInput = dbgInput();
    HRESULT             hResult = E_FAIL;

    // Call the EndInput debug input method (If created)
    if (pDbgInput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        hResult = pDbgInput->EndInput();

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // EndInput

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
