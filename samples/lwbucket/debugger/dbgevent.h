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
|*  Module: dbgevent.h                                                        *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGEVENT_H
#define _DBGEVENT_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
//  Constants
//
//******************************************************************************




//******************************************************************************
//
//  Functions
//
//******************************************************************************





//******************************************************************************
//
//  CLwstomDebugEventCallbacks Interface
//
//******************************************************************************
#ifdef __cplusplus
extern "C" {
#endif

/* {5a379682-5a9a-11e1-9037-c06b4824019b} */
DEFINE_GUID(IID_ILwstomDebugEventCallbacks, 0x5a379682, 0x5a9a, 0x11e1,
            0x90, 0x37, 0xc0, 0x6b, 0x48, 0x24, 0x01, 0x9b) ;

typedef interface DECLSPEC_UUID("5a379682-5a9a-11e1-9037-c06b4824019b")
    ILwstomDebugEventCallbacks * PLWSTOM_DEBUG_EVENT_CALLBACKS ;

#undef INTERFACE
#define INTERFACE ILwstomDebugEventCallbacks
DECLARE_INTERFACE_(ILwstomDebugEventCallbacks, IDebugEventCallbacks)
{
    // IUnknown.
    STDMETHOD(QueryInterface)(
        THIS_
        __in REFIID InterfaceId,
        __out PVOID* Interface
        ) PURE;
    STDMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;
    STDMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // IDebugEventCallbacks.

    // The engine calls GetInterestMask once when
    // the event callbacks are set for a client.
    STDMETHOD(GetInterestMask)(
        THIS_
        __out PULONG Mask
        ) PURE;

    // A breakpoint event is generated when
    // a breakpoint exception is received and
    // it can be mapped to an existing breakpoint.
    // The callback method is given a reference
    // to the breakpoint and should release it when
    // it is done with it.
    STDMETHOD(Breakpoint)(
        THIS_
        __in PDEBUG_BREAKPOINT Bp
        ) PURE;

    // Exceptions include breaks which cannot
    // be mapped to an existing breakpoint
    // instance.
    STDMETHOD(Exception)(
        THIS_
        __in PEXCEPTION_RECORD64 Exception,
        __in ULONG FirstChance
        ) PURE;

    // Any of these values can be zero if they
    // cannot be provided by the engine.
    // Lwrrently the kernel does not return thread
    // or process change events.
    STDMETHOD(CreateThread)(
        THIS_
        __in ULONG64 Handle,
        __in ULONG64 DataOffset,
        __in ULONG64 StartOffset
        ) PURE;
    STDMETHOD(ExitThread)(
        THIS_
        __in ULONG ExitCode
        ) PURE;

    // Any of these values can be zero if they
    // cannot be provided by the engine.
    STDMETHOD(CreateProcess)(
        THIS_
        __in ULONG64 ImageFileHandle,
        __in ULONG64 Handle,
        __in ULONG64 BaseOffset,
        __in ULONG ModuleSize,
        __in_opt PCSTR ModuleName,
        __in_opt PCSTR ImageName,
        __in ULONG CheckSum,
        __in ULONG TimeDateStamp,
        __in ULONG64 InitialThreadHandle,
        __in ULONG64 ThreadDataOffset,
        __in ULONG64 StartOffset
        ) PURE;
    STDMETHOD(ExitProcess)(
        THIS_
        __in ULONG ExitCode
        ) PURE;

    // Any of these values may be zero.
    STDMETHOD(LoadModule)(
        THIS_
        __in ULONG64 ImageFileHandle,
        __in ULONG64 BaseOffset,
        __in ULONG ModuleSize,
        __in_opt PCSTR ModuleName,
        __in_opt PCSTR ImageName,
        __in ULONG CheckSum,
        __in ULONG TimeDateStamp
        ) PURE;
    STDMETHOD(UnloadModule)(
        THIS_
        __in_opt PCSTR ImageBaseName,
        __in ULONG64 BaseOffset
        ) PURE;

    STDMETHOD(SystemError)(
        THIS_
        __in ULONG Error,
        __in ULONG Level
        ) PURE;

    // Session status is synchronous like the other
    // wait callbacks but it is called as the state
    // of the session is changing rather than at
    // specific events so its return value does not
    // influence waiting.  Implementations should just
    // return DEBUG_STATUS_NO_CHANGE.
    // Also, because some of the status
    // notifications are very early or very
    // late in the session lifetime there may not be
    // current processes or threads when the notification
    // is generated.
    STDMETHOD(SessionStatus)(
        THIS_
        __in ULONG Status
        ) PURE;

    // The following callbacks are informational
    // callbacks notifying the provider about
    // changes in debug state.  The return value
    // of these callbacks is ignored.  Implementations
    // can not call back into the engine.

    // Debuggee state, such as registers or data spaces,
    // has changed.
    STDMETHOD(ChangeDebuggeeState)(
        THIS_
        __in ULONG Flags,
        __in ULONG64 Argument
        ) PURE;
    // Engine state has changed.
    STDMETHOD(ChangeEngineState)(
        THIS_
        __in ULONG Flags,
        __in ULONG64 Argument
        ) PURE;
    // Symbol state has changed.
    STDMETHOD(ChangeSymbolState)(
        THIS_
        __in ULONG Flags,
        __in ULONG64 Argument
        ) PURE;

    // ILwstomDebugEventCallbacks.

    // Set the event interest mask.
    STDMETHOD(SetInterestMask)( 
        THIS_
        __in ULONG ulMask
        ) PURE;

}; // ILwstomDebugEventCallbacks

#ifdef __cplusplus
}
#endif

// CLwstomDebugEventCallbacks class
class CLwstomDebugEventCallbacks : public ILwstomDebugEventCallbacks
{
// Private Data
private:
    LONG                    m_lRefCount;

    ULONG                   m_ulEventMask;

    PDEBUG_EVENT_CALLBACKS  m_pDebugEventCallbacks;

// Public Interface Methods
//////////////////////////////////////////////////////////////////////*/
public:
    // IUnknown.
    STDMETHOD(QueryInterface)(
        THIS_
        __in REFIID InterfaceId,
        __out PVOID* Interface
        );
    STDMETHOD_(ULONG, AddRef)(
        THIS
        );
    STDMETHOD_(ULONG, Release)(
        THIS
        );

    // IDebugEventCallbacks.

    // The engine calls GetInterestMask once when
    // the event callbacks are set for a client.
    STDMETHOD(GetInterestMask)(
        THIS_
        __out PULONG Mask
        );

    // A breakpoint event is generated when
    // a breakpoint exception is received and
    // it can be mapped to an existing breakpoint.
    // The callback method is given a reference
    // to the breakpoint and should release it when
    // it is done with it.
    STDMETHOD(Breakpoint)(
        THIS_
        __in PDEBUG_BREAKPOINT Bp
        );

    // Exceptions include breaks which cannot
    // be mapped to an existing breakpoint
    // instance.
    STDMETHOD(Exception)(
        THIS_
        __in PEXCEPTION_RECORD64 Exception,
        __in ULONG FirstChance
        );

    // Any of these values can be zero if they
    // cannot be provided by the engine.
    // Lwrrently the kernel does not return thread
    // or process change events.
    STDMETHOD(CreateThread)(
        THIS_
        __in ULONG64 Handle,
        __in ULONG64 DataOffset,
        __in ULONG64 StartOffset
        );
    STDMETHOD(ExitThread)(
        THIS_
        __in ULONG ExitCode
        );

    // Any of these values can be zero if they
    // cannot be provided by the engine.
    STDMETHOD(CreateProcess)(
        THIS_
        __in ULONG64 ImageFileHandle,
        __in ULONG64 Handle,
        __in ULONG64 BaseOffset,
        __in ULONG ModuleSize,
        __in_opt PCSTR ModuleName,
        __in_opt PCSTR ImageName,
        __in ULONG CheckSum,
        __in ULONG TimeDateStamp,
        __in ULONG64 InitialThreadHandle,
        __in ULONG64 ThreadDataOffset,
        __in ULONG64 StartOffset
        );
    STDMETHOD(ExitProcess)(
        THIS_
        __in ULONG ExitCode
        );

    // Any of these values may be zero.
    STDMETHOD(LoadModule)(
        THIS_
        __in ULONG64 ImageFileHandle,
        __in ULONG64 BaseOffset,
        __in ULONG ModuleSize,
        __in_opt PCSTR ModuleName,
        __in_opt PCSTR ImageName,
        __in ULONG CheckSum,
        __in ULONG TimeDateStamp
        );
    STDMETHOD(UnloadModule)(
        THIS_
        __in_opt PCSTR ImageBaseName,
        __in ULONG64 BaseOffset
        );

    STDMETHOD(SystemError)(
        THIS_
        __in ULONG Error,
        __in ULONG Level
        );

    // Session status is synchronous like the other
    // wait callbacks but it is called as the state
    // of the session is changing rather than at
    // specific events so its return value does not
    // influence waiting.  Implementations should just
    // return DEBUG_STATUS_NO_CHANGE.
    // Also, because some of the status
    // notifications are very early or very
    // late in the session lifetime there may not be
    // current processes or threads when the notification
    // is generated.
    STDMETHOD(SessionStatus)(
        THIS_
        __in ULONG Status
        );

    // The following callbacks are informational
    // callbacks notifying the provider about
    // changes in debug state.  The return value
    // of these callbacks is ignored.  Implementations
    // can not call back into the engine.

    // Debuggee state, such as registers or data spaces,
    // has changed.
    STDMETHOD(ChangeDebuggeeState)(
        THIS_
        __in ULONG Flags,
        __in ULONG64 Argument
        );
    // Engine state has changed.
    STDMETHOD(ChangeEngineState)(
        THIS_
        __in ULONG Flags,
        __in ULONG64 Argument
        );
    // Symbol state has changed.
    STDMETHOD(ChangeSymbolState)(
        THIS_
        __in ULONG Flags,
        __in ULONG64 Argument
        );

    // ILwstomDebugEventCallbacks.

    // Set the event interest mask.
    STDMETHOD(SetInterestMask)( 
        THIS_
        __in ULONG ulMask
        );

// Public Class Methods
public:
            CLwstomDebugEventCallbacks();
           ~CLwstomDebugEventCallbacks();

}; // CLwstomDebugEventCallbacks

//******************************************************************************
//
// class CEvent
//
// Class for dealing with debugger extension events
//
//******************************************************************************
class CEvent
{
private:
static  CEvent*         m_pFirstEvent;              // Pointer to first event
static  CEvent*         m_pLastEvent;               // Pointer to last event
static  ULONG           m_ulEventCount;             // Event count

        CEvent*         m_pPrevEvent;               // Pointer to previous event
        CEvent*         m_pNextEvent;               // Pointer to next event

        void            addEvent(CEvent* pEvent);
        void            removeEvent(CEvent* pEvent);

protected:
                        CEvent();

public:
virtual                ~CEvent();

        // Debugger extension event methods
virtual HRESULT         breakpoint(PDEBUG_BREAKPOINT Bp) const;
virtual HRESULT         exception(PEXCEPTION_RECORD64 Exception, ULONG FirstChance) const;

virtual HRESULT         createThread(ULONG64 Handle, ULONG64 DataOffset, ULONG64 StartOffset) const;
virtual HRESULT         exitThread(ULONG ExitCode) const;

virtual HRESULT         createProcess(ULONG64 ImageFileHandle, ULONG64 Handle, ULONG64 BaseOffset, ULONG ModuleSize, PCSTR ModuleName, PCSTR ImageName, ULONG CheckSum, ULONG TimeDateStamp, ULONG64 InitialThreadHandle, ULONG64 ThreadDataOffset, ULONG64 StartOffset) const;
virtual HRESULT         exitProcess(ULONG ExitCode) const;

virtual HRESULT         loadModule(ULONG64 ImageFileHandle, ULONG64 BaseOffset, ULONG ModuleSize, PCSTR ModuleName, PCSTR ImageName, ULONG CheckSum, ULONG TimeDateStamp) const;
virtual HRESULT         unloadModule(PCSTR ImageBaseName, ULONG64 BaseOffset) const;

virtual HRESULT         systemError(ULONG Error, ULONG Level) const;

virtual HRESULT         sessionStatus(ULONG Status) const;

virtual HRESULT         changeDebuggeeState(ULONG Flags, ULONG64 Argument) const;
virtual HRESULT         changeEngineState(ULONG Flags, ULONG64 Argument) const;
virtual HRESULT         changeSymbolState(ULONG Flags, ULONG64 Argument) const;

static const CEvent*    firstEvent()                { return m_pFirstEvent; }
static const CEvent*    lastEvent()                 { return m_pLastEvent; }

        CEvent*         prevEvent() const           { return m_pPrevEvent; }
        CEvent*         nextEvent() const           { return m_pNextEvent; }

}; // class CEvent

// Custom Debug Event Callbacks Interface
HRESULT                 SetInterestMask(ULONG ulMask);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGEVENT_H
