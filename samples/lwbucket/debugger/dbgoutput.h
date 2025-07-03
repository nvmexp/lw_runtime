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
|*  Module: dbgoutput.h                                                       *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGOUTPUT_H
#define _DBGOUTPUT_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
//  CLwstomDebugOutputCallbacks Interface
//
//******************************************************************************
#ifdef __cplusplus
extern "C" {
#endif

/* {c39055fc-b44e-4ff4-a080-e27b197dc400} */
DEFINE_GUID(IID_ILwstomDebugOutputCallbacks, 0xc39055fc, 0xb44e, 0x4ff4,
            0xa0, 0x80, 0xe2, 0x7b, 0x19, 0x7d, 0xc4, 0x00 ) ;

typedef interface DECLSPEC_UUID("c39055fc-b44e-4ff4-a080-e27b197dc400")
    ILwstomDebugOutputCallbacks * PLWSTOM_DEBUG_OUTPUT_CALLBACKS ;

#undef INTERFACE
#define INTERFACE ILwstomDebugOutputCallbacks
DECLARE_INTERFACE_(ILwstomDebugOutputCallbacks, IDebugOutputCallbacks2)
{
    // IUnknown.
    STDMETHOD(QueryInterface)( THIS_
                               __in  REFIID InterfaceId ,
                               __out PVOID* Interface   ) PURE ;

    STDMETHOD_(ULONG, AddRef)( THIS ) PURE ;

    STDMETHOD_(ULONG, Release)( THIS ) PURE ;

    // IDebugOutputCallbacks.

    // This method is only called if the supplied mask
    // is allowed by the clients output control.
    // The return value is ignored.
    STDMETHOD(Output)( THIS_
                       __in ULONG Mask ,
                       __in PCSTR Text  ) PURE ;

    // IDebugOutputCallbacks2.

    // The engine calls GetInterestMask once when
    // the callbacks are set for a client.
    STDMETHOD(GetInterestMask)( THIS_
                                __out PULONG Mask ) PURE;
    
    STDMETHOD(Output2)( THIS_
                        __in ULONG Which,
                        __in ULONG Flags,
                        __in ULONG64 Arg,
                        __in_opt PCWSTR Text ) PURE;

    // ILwstomDebugOutputCallbacks.

    // Gets the current display output state.
    STDMETHOD_(BOOL, GetDisplayState)( THIS ) ;

    // Gets the current capture output state.
    STDMETHOD_(BOOL, GetCaptureState)( THIS_ ) PURE ;

    // Gets the current DML output state.
    STDMETHOD_(BOOL, GetDmlState)( THIS_ ) PURE ;

    // Pass in TRUE to display output, FALSE to not display output.
    STDMETHOD(SetDisplayState)( THIS_
                                __in BOOL bDisplayState ) ;

    // Pass in TRUE to capture output, FALSE to not capture output.
    STDMETHOD(SetCaptureState)( THIS_
                                __in BOOL bCaptureState ) PURE ;

    // Pass in TRUE to enable DML, FALSE to disable DML.
    STDMETHOD(SetDmlState)( THIS_
                            __in BOOL bDmlState ) PURE ;

    // Get the current size of the output capture buffer.
    STDMETHOD(GetCaptureSize)( THIS_
                              __out PULONG pSize ) PURE ;

    // Gets the last captured output.
    STDMETHOD(GetCaptureOutput)( THIS_
                                __out PSTR pBuffer ) PURE ;

    // Clear the capture buffer.
    STDMETHOD(ClearCapture)( THIS_ ) PURE ;

}; // ILwstomDebugOutputCallbacks

#ifdef __cplusplus
}
#endif

// CLwstomDebugOutputCallbacks class
class CLwstomDebugOutputCallbacks : public ILwstomDebugOutputCallbacks
{
// Private Data
private:
    LONG                    m_lRefCount;
    ULONG                   m_ulMask;

    BOOL                    m_bOutputEnable;
    BOOL                    m_bDisplayState;
    BOOL                    m_bCaptureState;
    BOOL                    m_bDmlState;

    char*                   m_pCaptureBuffer;
    ULONG                   m_ulBufferOffset;
    ULONG                   m_ulBufferSize;

    PDEBUG_OUTPUT_CALLBACKS m_pDebugOutputCallbacks;
    PDEBUG_OUTPUT_CALLBACKS2 m_pDebugOutputCallbacks2;

// Public Interface Methods
//////////////////////////////////////////////////////////////////////*/
public:
    // IUnknown.
    STDMETHOD(QueryInterface)( THIS_
                               __in  REFIID InterfaceId ,
                               __out PVOID* Interface   ) ;

    STDMETHOD_(ULONG, AddRef)( THIS ) ;

    STDMETHOD_(ULONG, Release)( THIS ) ;

    // IDebugOutputCallbacks.

    // This method is only called if the supplied mask
    // is allowed by the clients output control.
    // The return value is ignored.
    STDMETHOD(Output)( THIS_
                       __in ULONG Mask ,
                       __in PCSTR Text  ) ;

    // IDebugOutputCallbacks2.

    // The engine calls GetInterestMask once when
    // the callbacks are set for a client.
    STDMETHOD(GetInterestMask)( THIS_
                                __out PULONG Mask ) ;
    
    STDMETHOD(Output2)( THIS_
                        __in ULONG Which,
                        __in ULONG Flags,
                        __in ULONG64 Arg,
                        __in_opt PCWSTR Text ) ;

    // ILwstomDebugOutputCallbacks.

    // Gets the current display output state.
    STDMETHOD_(BOOL, GetDisplayState)( THIS ) ;

    // Gets the current capture output state.
    STDMETHOD_(BOOL, GetCaptureState)( THIS ) ;

    // Gets the current DML output state.
    STDMETHOD_(BOOL, GetDmlState)( THIS ) ;

    // Pass in TRUE to display output, FALSE to not display output.
    STDMETHOD(SetDisplayState)( THIS_
                                __in BOOL bDisplayState ) ;

    // Pass in TRUE to capture output, FALSE to not capture output.
    STDMETHOD(SetCaptureState)( THIS_
                                __in BOOL bCaptureState ) ;

    // Pass in TRUE to enable DML, FALSE to disable DML.
    STDMETHOD(SetDmlState)( THIS_
                            __in BOOL bDmlState ) ;

    // Get the current size of the output capture buffer.
    STDMETHOD(GetCaptureSize)( THIS_
                              __out PULONG pSize ) ;

    // Gets the last captured output.
    STDMETHOD(GetCaptureOutput)( THIS_
                                __out PSTR pBuffer ) ;

    // Clear the capture buffer.
    STDMETHOD(ClearCapture)( THIS ) ;

// Public Class Methods
public:
            CLwstomDebugOutputCallbacks();
           ~CLwstomDebugOutputCallbacks();

}; // CLwstomDebugOutputCallbacks

// Custom Debug Output Callbacks Interface
BOOL                    GetDisplayState();
BOOL                    GetCaptureState();
BOOL                    GetDmlState();
HRESULT                 SetDisplayState(BOOL bDisplayState);
HRESULT                 SetCaptureState(BOOL bCaptureState);
HRESULT                 SetDmlState(BOOL bDmlState);
HRESULT                 GetCaptureSize(PULONG pSize);
HRESULT                 GetCaptureOutput(PSTR pBuffer);
HRESULT                 ClearCapture();

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGOUTPUT_H
