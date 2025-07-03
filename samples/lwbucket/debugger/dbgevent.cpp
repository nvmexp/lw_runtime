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
|*  Module: dbgevent.cpp                                                      *|
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
//  Forwards
//
//******************************************************************************
static const CEvent*    firstEvent();               // First event entry

//******************************************************************************
//
//  Locals
//
//******************************************************************************
CEvent* CEvent::m_pFirstEvent  = NULL;              // First event entry
CEvent* CEvent::m_pLastEvent   = NULL;              // Last event entry
ULONG   CEvent::m_ulEventCount = 0;                 // Event count value

//******************************************************************************
//
//  CLwstomDebugEventCallbacks Interface
//
//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::QueryInterface
(
    THIS_
    __in  REFIID        InterfaceId,
    __out PVOID*        Interface
)
{
    HRESULT             hResult = E_NOINTERFACE;

    assert(Interface != NULL);

    if (IsEqualIID(InterfaceId , __uuidof(IUnknown)) || IsEqualIID(InterfaceId , __uuidof(IDebugEventCallbacks)))
    {
        *Interface = (IDebugEventCallbacks *)this;
        AddRef ( ) ;

        hResult = S_OK;
    }
    return hResult;

} // QueryInterface

//******************************************************************************

STDMETHODIMP_(ULONG)
CLwstomDebugEventCallbacks::AddRef
(
    THIS
)
{
    // Increment and return the new interface reference count
    return InterlockedIncrement(&m_lRefCount);

} // AddRef

//******************************************************************************

STDMETHODIMP_(ULONG)
CLwstomDebugEventCallbacks::Release
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
CLwstomDebugEventCallbacks::GetInterestMask
(
    THIS_
    __out PULONG        Mask
)
{
    HRESULT             hResult = S_OK;

    assert(Mask != NULL);

    // Get the event interest mask
    *Mask = m_ulEventMask;

    return hResult;

} // GetInterestMask

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::Breakpoint
(
    THIS_
    __in  PDEBUG_BREAKPOINT Bp
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    assert(Bp != NULL);

    // Loop calling breakpoint events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next breakpoint event
            hStatus = pEvent->breakpoint(Bp);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // Breakpoint

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::Exception
(
    THIS_
    __in PEXCEPTION_RECORD64 Exception,
    __in ULONG          FirstChance
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling exception events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next exception event
            hStatus = pEvent->exception(Exception, FirstChance);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // Exception

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::CreateThread
(
    THIS_
    __in  ULONG64       Handle,
    __in  ULONG64       DataOffset,
    __in  ULONG64       StartOffset
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling create thread events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next create thread event
            hStatus = pEvent->createThread(Handle, DataOffset, StartOffset);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // CreateThread

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::ExitThread
(
    THIS_
    __in  ULONG         ExitCode
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling exit thread events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next exit thread event
            hStatus = pEvent->exitThread(ExitCode);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // ExitThread

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::CreateProcess
(
    THIS_
    __in     ULONG64    ImageFileHandle,
    __in     ULONG64    Handle,
    __in     ULONG64    BaseOffset,
    __in     ULONG      ModuleSize,
    __in_opt PCSTR      ModuleName,
    __in_opt PCSTR      ImageName,
    __in     ULONG      CheckSum,
    __in     ULONG      TimeDateStamp,
    __in     ULONG64    InitialThreadHandle,
    __in     ULONG64    ThreadDataOffset,
    __in     ULONG64    StartOffset
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling create process events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next create process event
            hStatus = pEvent->createProcess(ImageFileHandle, Handle, BaseOffset, ModuleSize, ModuleName, ImageName, CheckSum, TimeDateStamp, InitialThreadHandle, ThreadDataOffset, StartOffset);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // CreateProcess

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::ExitProcess
(
    THIS_
    __in  ULONG         ExitCode
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling exit process events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next exit process event
            hStatus = pEvent->exitProcess(ExitCode);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // ExitProcess

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::LoadModule
(
    THIS_
    __in     ULONG64    ImageFileHandle,
    __in     ULONG64    BaseOffset,
    __in     ULONG      ModuleSize,
    __in_opt PCSTR      ModuleName,
    __in_opt PCSTR      ImageName,
    __in     ULONG      CheckSum,
    __in     ULONG      TimeDateStamp
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling load module events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next load module event
            hStatus = pEvent->loadModule(ImageFileHandle, BaseOffset, ModuleSize, ModuleName, ImageName, CheckSum, TimeDateStamp);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // LoadModule

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::UnloadModule
(
    THIS_
    __in_opt PCSTR      ImageBaseName,
    __in     ULONG64    BaseOffset
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling unload module events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next unload module event
            hStatus = pEvent->unloadModule(ImageBaseName, BaseOffset);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // UnloadModule

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::SystemError
(
    THIS_
    __in  ULONG         Error,
    __in  ULONG         Level
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling system error events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next system error event
            hStatus = pEvent->systemError(Error, Level);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // SystemError

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::SessionStatus
(
    THIS_
    __in  ULONG         Status
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling session status events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next session status event
            hStatus = pEvent->sessionStatus(Status);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // SessionStatus

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::ChangeDebuggeeState
(
    THIS_
    __in  ULONG         Flags,
    __in  ULONG64       Argument
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling change debuggee state events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next change debuggee state event
            hStatus = pEvent->changeDebuggeeState(Flags, Argument);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // ChangeDebuggeeState

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::ChangeEngineState
(
    THIS_
    __in  ULONG         Flags,
    __in  ULONG64       Argument
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling change engine state events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next change engine state event
            hStatus = pEvent->changeEngineState(Flags, Argument);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // ChangeEngineState

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::ChangeSymbolState
(
    THIS_
    __in  ULONG         Flags,
    __in  ULONG64       Argument
)
{
    const CEvent       *pEvent = firstEvent();
    HRESULT             hStatus;
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Loop calling change symbol state events
    while (pEvent != NULL)
    {
        // Don't let exceptions in one event affect another
        try
        {
            // Call the next change symbol state event
            hStatus = pEvent->changeSymbolState(Flags, Argument);
            if (hStatus != DEBUG_STATUS_NO_CHANGE)
            {
                // Save the new debug status value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing events)
            exception.dPrintf();
        }
        // Move to the next event
        pEvent = pEvent->nextEvent();
    }
    return hResult;

} // ChangeSymbolState

//******************************************************************************

STDMETHODIMP
CLwstomDebugEventCallbacks::SetInterestMask
(
    THIS_
    __in ULONG          ulMask
)
{
    HRESULT             hResult = S_OK;

    // Set the event interest mask
    m_ulEventMask = ulMask;

    // Need to reset the custom debug event callbacks (to force a get of new interest mask)
    hResult = SetEventCallbacks(funcptr(PDEBUG_EVENT_CALLBACKS, this));

    return hResult;

} // SetInterestMask

//******************************************************************************

CLwstomDebugEventCallbacks::CLwstomDebugEventCallbacks()
:   m_lRefCount(0),
    m_ulEventMask(0),
    m_pDebugEventCallbacks(NULL)
{
    HRESULT             hResult;

    // Try to get existing debug event callbacks
    hResult = GetEventCallbacks(&m_pDebugEventCallbacks);
    if (SUCCEEDED(hResult))
    {
        // Set interest mask to all events (Will setup the event callbacks)
        hResult = SetInterestMask(DEBUG_EVENT_BREAKPOINT            |
                                  DEBUG_EVENT_EXCEPTION             |
                                  DEBUG_EVENT_CREATE_THREAD         |
                                  DEBUG_EVENT_EXIT_THREAD           |
                                  DEBUG_EVENT_CREATE_PROCESS        |
                                  DEBUG_EVENT_EXIT_PROCESS          |
                                  DEBUG_EVENT_LOAD_MODULE           |
                                  DEBUG_EVENT_UNLOAD_MODULE         |
                                  DEBUG_EVENT_SYSTEM_ERROR          |
                                  DEBUG_EVENT_SESSION_STATUS        |
                                  DEBUG_EVENT_CHANGE_DEBUGGEE_STATE |
                                  DEBUG_EVENT_CHANGE_ENGINE_STATE   |
                                  DEBUG_EVENT_CHANGE_SYMBOL_STATE);
    }

} // CLwstomDebugEventCallbacks

//******************************************************************************

CLwstomDebugEventCallbacks::~CLwstomDebugEventCallbacks()
{
    // Restore the original event callbacks [May be none] (ignore any errors)
    SetEventCallbacks(m_pDebugEventCallbacks);

} // ~CLwstomDebugEventCallbacks

//******************************************************************************

CEvent::CEvent()
:   m_pPrevEvent(NULL),
    m_pNextEvent(NULL)
{
    // Add this event to the event list
    addEvent(this);

} // CEvent

//******************************************************************************

CEvent::~CEvent()
{
    // Remove this event from the event list
    removeEvent(this);

} // ~CEvent

//******************************************************************************

void
CEvent::addEvent
(
    CEvent             *pEvent
)
{
    assert(pEvent != NULL);

    // Check for first event
    if (m_pFirstEvent == NULL)
    {
        // Set first and last event to this event
        m_pFirstEvent = pEvent;
        m_pLastEvent  = pEvent;
    }
    else    // Adding new event to event list
    {
        // Add this event to the end of the event list
        pEvent->m_pPrevEvent = m_pLastEvent;
        pEvent->m_pNextEvent = NULL;

        m_pLastEvent->m_pNextEvent = pEvent;

        m_pLastEvent = pEvent;
    }
    // Increment the event count
    m_ulEventCount++;

} // addEvent

//******************************************************************************

void
CEvent::removeEvent
(
    CEvent             *pEvent
)
{
    assert(pEvent != NULL);

    // Remove this event from the event list
    if (pEvent->m_pPrevEvent != NULL)
    {
        pEvent->m_pPrevEvent->m_pNextEvent = pEvent->m_pNextEvent;
    }
    if (pEvent->m_pNextEvent != NULL)
    {
        pEvent->m_pNextEvent->m_pPrevEvent = pEvent->m_pPrevEvent;
    }
    // Check for first event
    if (m_pFirstEvent == pEvent)
    {
        // Update first event
        m_pFirstEvent = pEvent->m_pNextEvent;
    }
    // Check for last event
    if (m_pLastEvent == pEvent)
    {
        // Update last event
        m_pLastEvent = pEvent->m_pPrevEvent;
    }
    // Decrement the event count
    m_ulEventCount--;

} // removeEvent

//******************************************************************************

HRESULT
CEvent::breakpoint
(
    PDEBUG_BREAKPOINT   Bp
) const
{
    UNREFERENCED_PARAMETER(Bp);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // breakpoint

//******************************************************************************

HRESULT
CEvent::exception
(
    PEXCEPTION_RECORD64 Exception,
    ULONG               FirstChance
) const
{
    UNREFERENCED_PARAMETER(Exception);
    UNREFERENCED_PARAMETER(FirstChance);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // exception

//******************************************************************************

HRESULT
CEvent::createThread
(
    ULONG64             Handle,
    ULONG64             DataOffset,
    ULONG64             StartOffset
) const
{
    UNREFERENCED_PARAMETER(Handle);
    UNREFERENCED_PARAMETER(DataOffset);
    UNREFERENCED_PARAMETER(StartOffset);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // createThread

//******************************************************************************

HRESULT
CEvent::exitThread
(
    ULONG               ExitCode
) const
{
    UNREFERENCED_PARAMETER(ExitCode);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // exitThread

//******************************************************************************

HRESULT
CEvent::createProcess
(
    ULONG64             ImageFileHandle,
    ULONG64             Handle,
    ULONG64             BaseOffset,
    ULONG               ModuleSize,
    PCSTR               ModuleName,
    PCSTR               ImageName,
    ULONG               CheckSum,
    ULONG               TimeDateStamp,
    ULONG64             InitialThreadHandle,
    ULONG64             ThreadDataOffset,
    ULONG64             StartOffset
) const
{
    UNREFERENCED_PARAMETER(ImageFileHandle);
    UNREFERENCED_PARAMETER(Handle);
    UNREFERENCED_PARAMETER(BaseOffset);
    UNREFERENCED_PARAMETER(ModuleSize);
    UNREFERENCED_PARAMETER(ModuleName);
    UNREFERENCED_PARAMETER(ImageName);
    UNREFERENCED_PARAMETER(CheckSum);
    UNREFERENCED_PARAMETER(TimeDateStamp);
    UNREFERENCED_PARAMETER(InitialThreadHandle);
    UNREFERENCED_PARAMETER(ThreadDataOffset);
    UNREFERENCED_PARAMETER(StartOffset);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // createProcess

//******************************************************************************

HRESULT
CEvent::exitProcess
(
    ULONG               ExitCode
) const
{
    UNREFERENCED_PARAMETER(ExitCode);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // exitProcess

//******************************************************************************

HRESULT
CEvent::loadModule
(
    ULONG64             ImageFileHandle,
    ULONG64             BaseOffset,
    ULONG               ModuleSize,
    PCSTR               ModuleName,
    PCSTR               ImageName,
    ULONG               CheckSum,
    ULONG               TimeDateStamp
) const
{
    UNREFERENCED_PARAMETER(ImageFileHandle);
    UNREFERENCED_PARAMETER(BaseOffset);
    UNREFERENCED_PARAMETER(ModuleSize);
    UNREFERENCED_PARAMETER(ModuleName);
    UNREFERENCED_PARAMETER(ImageName);
    UNREFERENCED_PARAMETER(CheckSum);
    UNREFERENCED_PARAMETER(TimeDateStamp);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // loadModule

//******************************************************************************

HRESULT
CEvent::unloadModule
(
    PCSTR               ImageBaseName,
    ULONG64             BaseOffset
) const
{
    UNREFERENCED_PARAMETER(ImageBaseName);
    UNREFERENCED_PARAMETER(BaseOffset);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // unloadModule

//******************************************************************************

HRESULT
CEvent::systemError
(
    ULONG               Error,
    ULONG               Level
) const
{
    UNREFERENCED_PARAMETER(Error);
    UNREFERENCED_PARAMETER(Level);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // systemError

//******************************************************************************

HRESULT
CEvent::sessionStatus
(
    ULONG               Status
) const
{
    UNREFERENCED_PARAMETER(Status);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // sessionStatus

//******************************************************************************

HRESULT
CEvent::changeDebuggeeState
(
    ULONG               Flags,
    ULONG64             Argument
) const
{
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(Argument);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // changeDebuggeeState

//******************************************************************************

HRESULT
CEvent::changeEngineState
(
    ULONG               Flags,
    ULONG64             Argument
) const
{
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(Argument);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // changeEngineState

//******************************************************************************

HRESULT
CEvent::changeSymbolState
(
    ULONG               Flags,
    ULONG64             Argument
) const
{
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(Argument);

    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    return hResult;

} // changeSymbolState

//******************************************************************************

static const CEvent*
firstEvent()
{
    // Return the first event routine
    return CEvent::firstEvent();

} // firstEvent

//******************************************************************************
//
// Custom Debug Event Interface wrappers
//
//******************************************************************************

HRESULT
SetInterestMask
(
    ULONG               ulMask
)
{
    PLWSTOM_DEBUG_EVENT_CALLBACKS pDbgEvent = dbgEvent();
    HRESULT             hResult = E_NOINTERFACE;

    // Check for event interface
    if (pDbgEvent != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetInterestMask debug event method
        hResult = pDbgEvent->SetInterestMask(ulMask);

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // SetInterestMask

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
