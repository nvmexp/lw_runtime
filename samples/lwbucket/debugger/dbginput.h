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
|*  Module: dbginput.h                                                        *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGINPUT_H
#define _DBGINPUT_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
//  CLwstomDebugInputCallbacks Interface
//
//******************************************************************************
#ifdef __cplusplus
extern "C" {
#endif

/* {57826f36-de22-429c-a59a-67b220c19d98} */
DEFINE_GUID(IID_ILwstomDebugInputCallbacks, 0x57826f36, 0xde22, 0x429c,
            0xa5, 0x9a, 0x67, 0xb2, 0x20, 0xc1, 0x9d, 0x98) ;

typedef interface DECLSPEC_UUID("57826f36-de22-429c-a59a-67b220c19d98")
    ILwstomDebugInputCallbacks * PLWSTOM_DEBUG_INPUT_CALLBACKS ;

#undef INTERFACE
#define INTERFACE ILwstomDebugInputCallbacks
DECLARE_INTERFACE_(ILwstomDebugInputCallbacks, IDebugInputCallbacks)
{
    // IUnknown.
    STDMETHOD(QueryInterface)( THIS_
                               __in  REFIID InterfaceId ,
                               __out PVOID* Interface   ) PURE ;

    STDMETHOD_(ULONG, AddRef)( THIS ) PURE ;

    STDMETHOD_(ULONG, Release)( THIS ) PURE ;

    // IDebugInputCallbacks.

    // A call to the StartInput method is a request for
    // a line of input from any client.  The returned input
    // should always be zero-terminated.  The buffer size
    // provided is only a guideline.  A client can return
    // more if necessary and the engine will truncate it
    // before returning from IDebugControl::Input.
    // The return value is ignored.
    STDMETHOD(StartInput)( THIS_
                           __in ULONG BufferSize ) PURE;

    // The return value is ignored.
    STDMETHOD(EndInput)( THIS ) PURE;

    // ILwstomDebugInputCallbacks.


}; // ILwstomDebugInputCallbacks

#ifdef __cplusplus
}
#endif

// CLwstomDebugInputCallbacks class
class CLwstomDebugInputCallbacks : public ILwstomDebugInputCallbacks
{
// Private Data
private:
    LONG                    m_lRefCount;

    PDEBUG_INPUT_CALLBACKS  m_pDebugInputCallbacks;

// Public Interface Methods
//////////////////////////////////////////////////////////////////////*/
public:
    // IUnknown.
    STDMETHOD(QueryInterface)( THIS_
                               __in  REFIID InterfaceId ,
                               __out PVOID* Interface   ) ;

    STDMETHOD_(ULONG, AddRef)( THIS ) ;

    STDMETHOD_(ULONG, Release)( THIS ) ;

    // IDebugInputCallbacks.

    // A call to the StartInput method is a request for
    // a line of input from any client.  The returned input
    // should always be zero-terminated.  The buffer size
    // provided is only a guideline.  A client can return
    // more if necessary and the engine will truncate it
    // before returning from IDebugControl::Input.
    // The return value is ignored.
    STDMETHOD(StartInput)( THIS_
                           __in ULONG BufferSize ) ;

    // The return value is ignored.
    STDMETHOD(EndInput)( THIS ) ;

    // ILwstomDebugInputCallbacks.


// Public Class Methods
public:
            CLwstomDebugInputCallbacks();
           ~CLwstomDebugInputCallbacks();

}; // CLwstomDebugInputCallbacks

// Custom Debug Input Callbacks Interface
HRESULT                 StartInput(ULONG BufferSize);
HRESULT                 EndInput();

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGINPUT_H
