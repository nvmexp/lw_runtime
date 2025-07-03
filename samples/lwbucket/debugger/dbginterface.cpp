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
|*  Module: dbginterface.cpp                                                  *|
|*                                                                            *|
 \****************************************************************************/
#include "dbgprecomp.h"

//******************************************************************************
//
// Globals
//
//******************************************************************************
// Global WDBG Extension API's
wdbgexts::WINDBG_EXTENSION_APIS wdbgexts::ExtensionApis;    // WBDG extension API's

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
// Forwards
//
//******************************************************************************
//static  HRESULT getDebugClientInterface(PDEBUG_CLIENT pClient, PDEBUG_CLIENT* pDbgClient, IID* pDbgClientId);

static  HRESULT createDebugClientInterface(PDEBUG_CLIENT* pDbgClient, IID* pDbgClientId);
static  HRESULT createDebugControlInterface(PDEBUG_CONTROL* pDbgControl, IID* pDbgControlId);
static  HRESULT createDebugDataSpacesInterface(PDEBUG_DATA_SPACES* pDbgDataSpaces, IID* pDbgDataSpacesId);
static  HRESULT createDebugRegistersInterface(PDEBUG_REGISTERS* pDbgRegisters, IID* pDbgRegistersId);
static  HRESULT createDebugSymbolsInterface(PDEBUG_SYMBOLS* pDbgSymbols, IID* pDbgSymbolsId);
static  HRESULT createDebugSystemObjectsInterface(PDEBUG_SYSTEM_OBJECTS* pDbgSystemObjects, IID* pDbgSystemObjectsId);
static  HRESULT createDebugAdvancedInterface(PDEBUG_ADVANCED* pDbgAdvanced, IID* pDbgAdvancedId);

static  HRESULT createDebugInputInterface(PDEBUG_INPUT_CALLBACKS* pDbgInput);
static  HRESULT createDebugOutputInterface(PDEBUG_OUTPUT_CALLBACKS* pDbgOutput);
static  HRESULT createDebugEventInterface(PDEBUG_EVENT_CALLBACKS* pDbgEvent);

static  PDEBUG_CLIENT                   dbgClient();
static  PDEBUG_CONTROL                  dbgControl();
static  PDEBUG_DATA_SPACES              dbgDataSpaces();
static  PDEBUG_REGISTERS                dbgRegisters();
static  PDEBUG_SYMBOLS                  dbgSymbols();
static  PDEBUG_SYSTEM_OBJECTS           dbgSystemObjects();
static  PDEBUG_ADVANCED                 dbgAdvanced();
static  PDEBUG_SYMBOL_GROUP2            dbgSymbolGroup();
static  PDEBUG_BREAKPOINT3              dbgBreakpoint();

static  GUID&                           dbgClientId();
static  GUID&                           dbgControlId();
static  GUID&                           dbgDataSpacesId();
static  GUID&                           dbgRegistersId();
static  GUID&                           dbgSymbolsId();
static  GUID&                           dbgSystemObjectsId();
static  GUID&                           dbgAdvancedId();

//******************************************************************************
//
// Locals
//
//******************************************************************************
// Storage for the debugger interfaces
static  DEBUG_INTERFACES                s_DbgInterfaces = {0};

static  CRITICAL_SECTION                s_DbgInterfaceLock;

static  GUID                            s_DbgUnknownId;

//******************************************************************************

HRESULT
createInterface
(
    REFIID              refInterfaceId,
    PVOID              *ppInterface
)
{
    CLwstomDebugInputCallbacks* pLwstomDebugInputCallbacks;
    CLwstomDebugOutputCallbacks* pLwstomDebugOutputCallbacks;
    CLwstomDebugEventCallbacks* pLwstomDebugEventCallbacks;
    HRESULT             hResult = E_NOINTERFACE;

    assert(ppInterface != NULL);

    // Check to see if this is the custom debug input interface
    if (IsEqualIID(refInterfaceId, __uuidof(ILwstomDebugInputCallbacks)))
    {
        // Check to see if custom debug input interface not already created
        pLwstomDebugInputCallbacks = reinterpret_cast<CLwstomDebugInputCallbacks*>(*ppInterface);
        if (pLwstomDebugInputCallbacks == NULL)
        {
            // Try to create the custom debug input interface
            try
            {
                pLwstomDebugInputCallbacks = new CLwstomDebugInputCallbacks();
                *ppInterface               = pLwstomDebugInputCallbacks;
            }
            catch (CMemoryException& exception)
            {
                UNREFERENCED_PARAMETER(exception);

                return hResult;
            }
        }
        // Indicate custom debug input interface created
        hResult = S_OK;
    }
    // Check to see if this is the custom debug output interface
    else if (IsEqualIID(refInterfaceId, __uuidof(ILwstomDebugOutputCallbacks)))
    {
        // Check to see if custom debug output interface not already created
        pLwstomDebugOutputCallbacks = reinterpret_cast<CLwstomDebugOutputCallbacks*>(*ppInterface);
        if (pLwstomDebugOutputCallbacks == NULL)
        {
            // Try to create the custom debug output interface
            try
            {
                pLwstomDebugOutputCallbacks = new CLwstomDebugOutputCallbacks();
                *ppInterface                = pLwstomDebugOutputCallbacks;
            }
            catch (CMemoryException& exception)
            {
                UNREFERENCED_PARAMETER(exception);

                return hResult;
            }
        }
        // Indicate custom debug output interface created
        hResult = S_OK;
    }
    // Check to see if this is the custom debug event interface
    else if (IsEqualIID(refInterfaceId, __uuidof(ILwstomDebugEventCallbacks)))
    {
        // Check to see if custom debug event interface not already created
        pLwstomDebugEventCallbacks = reinterpret_cast<CLwstomDebugEventCallbacks*>(*ppInterface);
        if (pLwstomDebugEventCallbacks == NULL)
        {
            // Try to create the custom debug event interface
            try
            {
                pLwstomDebugEventCallbacks = new CLwstomDebugEventCallbacks();
                *ppInterface               = pLwstomDebugEventCallbacks;
            }
            catch (CMemoryException& exception)
            {
                UNREFERENCED_PARAMETER(exception);

                return hResult;
            }
        }
        // Indicate custom debug event interface created
        hResult = S_OK;
    }
    return hResult;

} // createInterface

//******************************************************************************

HRESULT
releaseInterface
(
    REFIID              refInterfaceId,
    PVOID              *ppInterface
)
{
    CLwstomDebugInputCallbacks* pLwstomDebugInputCallbacks;
    CLwstomDebugOutputCallbacks* pLwstomDebugOutputCallbacks;
    CLwstomDebugEventCallbacks* pLwstomDebugEventCallbacks;
    HRESULT             hResult = E_NOINTERFACE;

    assert(ppInterface != NULL);

    // Check to see if this is the custom debug input interface
    if (IsEqualIID(refInterfaceId, __uuidof(ILwstomDebugInputCallbacks)))
    {
        // Check to see if custom debug input interface created
        pLwstomDebugInputCallbacks = reinterpret_cast<CLwstomDebugInputCallbacks*>(*ppInterface);
        if (pLwstomDebugInputCallbacks != NULL)
        {
            // Dereference the custom debug input interface (free if no more references)
            if (pLwstomDebugInputCallbacks->Release() == 0)
            {
                *ppInterface = NULL;
            }
        }
        hResult = S_OK;
    }
    // Check to see if this is the custom debug output interface
    else if (IsEqualIID(refInterfaceId, __uuidof(ILwstomDebugOutputCallbacks)))
    {
        // Check to see if custom debug output interface created
        pLwstomDebugOutputCallbacks = reinterpret_cast<CLwstomDebugOutputCallbacks*>(*ppInterface);
        if (pLwstomDebugOutputCallbacks != NULL)
        {
            // Dereference the custom debug output interface (free if no more references)
            if (pLwstomDebugOutputCallbacks->Release() == 0)
            {
                *ppInterface = NULL;
            }
        }
        hResult = S_OK;
    }
    // Check to see if this is the custom debug event interface
    else if (IsEqualIID(refInterfaceId, __uuidof(ILwstomDebugEventCallbacks)))
    {
        // Check to see if custom debug event interface created
        pLwstomDebugEventCallbacks = reinterpret_cast<CLwstomDebugEventCallbacks*>(*ppInterface);
        if (pLwstomDebugEventCallbacks != NULL)
        {
            // Dereference the custom debug event interface (free if no more references)
            if (pLwstomDebugEventCallbacks->Release() == 0)
            {
                *ppInterface = NULL;
            }
        }
        hResult = S_OK;
    }
    return hResult;

} // releaseInterface

//******************************************************************************

PDEBUG_BREAKPOINT3
getBreakpointInterface
(
    ULONG               Id
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = NULL;

    // Try to get the requested breakpoint interface
    if (SUCCEEDED(GetBreakpointById(Id, reinterpret_cast<PDEBUG_BREAKPOINT*>(&s_DbgInterfaces.pDbgBreakpoint))))
    {
        // Get the debug breakpoint interface
        pDbgBreakpoint = s_DbgInterfaces.pDbgBreakpoint;
    }
    return pDbgBreakpoint;

} // getBreakpointInterface

//******************************************************************************

void
setBreakpointInterface
(
    PDEBUG_BREAKPOINT3  pDbgBreakpoint
)
{
    // Set the debug breakpoint interface
    s_DbgInterfaces.pDbgBreakpoint = pDbgBreakpoint;

    return;

} // setBreakpointInterface

//******************************************************************************

// Initialize all the DbgEng interfaces
void
initializeInterfaces()
{
    HRESULT             hResult = E_FAIL;

    try
    {
        // Check for debug client not yet initialized
        if (s_DbgInterfaces.pDbgClient == NULL)
        {
            // Acquire the debug interface (Single thread initialization)
            acquireDebugInterface();

            // Try to create a new client interface
            hResult = createDebugClientInterface(&s_DbgInterfaces.pDbgClient, &s_DbgInterfaces.dbgClientId);
            if (SUCCEEDED(hResult))
            {
                // Try to initialize all the debugger interfaces
                THROW_ON_FAIL(createDebugControlInterface(&s_DbgInterfaces.pDbgControl, &s_DbgInterfaces.dbgControlId));
                THROW_ON_FAIL(createDebugDataSpacesInterface(&s_DbgInterfaces.pDbgDataSpaces, &s_DbgInterfaces.dbgDataSpacesId));
                THROW_ON_FAIL(createDebugRegistersInterface(&s_DbgInterfaces.pDbgRegisters, &s_DbgInterfaces.dbgRegistersId));
                THROW_ON_FAIL(createDebugSymbolsInterface(&s_DbgInterfaces.pDbgSymbols, &s_DbgInterfaces.dbgSymbolsId));
                THROW_ON_FAIL(createDebugSystemObjectsInterface(&s_DbgInterfaces.pDbgSystemObjects, &s_DbgInterfaces.dbgSystemObjectsId));
                THROW_ON_FAIL(createDebugAdvancedInterface(&s_DbgInterfaces.pDbgAdvanced, &s_DbgInterfaces.dbgAdvancedId));

                // Try to initialize the custom interfaces
                THROW_ON_FAIL(createDebugInputInterface(&s_DbgInterfaces.pDbgInput));
                THROW_ON_FAIL(createDebugOutputInterface(&s_DbgInterfaces.pDbgOutput));
                THROW_ON_FAIL(createDebugEventInterface(&s_DbgInterfaces.pDbgEvent));

                // Make sure the progress state is reset
                progressReset();

                // Get the current scope symbol group (Don't throw an error as there might be no current scope)
                GetScopeSymbolGroup2(DEBUG_SCOPE_GROUP_ALL, NULL, &s_DbgInterfaces.pDbgSymbolGroup);

                // Try to get the WDBG extension API's
                ExtensionApis.nSize = sizeof(ExtensionApis);
                THROW_ON_FAIL(GetWindbgExtensionApis64(&ExtensionApis));
            }
            else    // Failed to create new client interface
            {
                // Throw exception
                throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                "Failed to create new debug client");
            }
            // Release the debug interface
            releaseDebugInterface();
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Release all the debugger interfaces and throw exception
        releaseInterfaces();
        throw;
    }

} // initializeInterfaces

//******************************************************************************

// Releases all the DbgEng interfaces
void
releaseInterfaces()
{
    // Catch any errors from debugger API calls
    try
    {
        // Make sure no more events are processed (Set event interest mask to 0)
        SetInterestMask(0);
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    // Release all the debugger interfaces
    safeRelease(s_DbgInterfaces.pDbgSymbolGroup);

    safeRelease(s_DbgInterfaces.pDbgAdvanced);
    safeRelease(s_DbgInterfaces.pDbgSystemObjects);
    safeRelease(s_DbgInterfaces.pDbgSymbols);
    safeRelease(s_DbgInterfaces.pDbgRegisters);
    safeRelease(s_DbgInterfaces.pDbgDataSpaces);
    safeRelease(s_DbgInterfaces.pDbgControl);

    safeRelease(s_DbgInterfaces.pDbgEvent);
    safeRelease(s_DbgInterfaces.pDbgOutput);
    safeRelease(s_DbgInterfaces.pDbgInput);

    safeRelease(s_DbgInterfaces.pDbgClient);

} // releaseInterfaces

//******************************************************************************

void
initializeDebugInterface()
{
    // Initialize the debugger interface critical section
    InitializeCriticalSection(&s_DbgInterfaceLock);

} // initializeDebugInterface

//******************************************************************************

void
uninitializeDebugInterface()
{
    // Delete the debug interface critical section
    DeleteCriticalSection(&s_DbgInterfaceLock);

} // uninitializeDebugInterface

//******************************************************************************

void
acquireDebugInterface()
{
    // Acquire the debug interface critical section
    EnterCriticalSection(&s_DbgInterfaceLock);

} // acquireDebugInterface

//******************************************************************************

void
releaseDebugInterface()
{
    // Release the debug interface critical section
    LeaveCriticalSection(&s_DbgInterfaceLock);

} // releaseDebugInterface

//******************************************************************************

static HRESULT
createDebugClientInterface
(
    PDEBUG_CLIENT      *pDbgClient,
    IID                *pDbgClientId
)
{
    HRESULT             hResult = E_NOINTERFACE;

    assert(pDbgClient != NULL);
    assert(pDbgClientId != NULL);

    // Try to create the debug client 5 interface
    hResult = DebugCreate(__uuidof(IDebugClient5), pvoidptr(pDbgClient));
    if (SUCCEEDED(hResult))
    {
        // Save the actual debug client interface UUID
        *pDbgClientId = __uuidof(IDebugClient5);
    }
    else    // Unable to create debug client 5 interface
    {
        // Try to create the debug client 4 interface
        hResult = DebugCreate(__uuidof(IDebugClient4), pvoidptr(pDbgClient));
        if (SUCCEEDED(hResult))
        {
            // Save the actual debug client interface UUID
            *pDbgClientId = __uuidof(IDebugClient4);
        }
        else    // Unable to create debug client 4 interface
        {
            // Try to create the debug client 3 interface
            hResult = DebugCreate(__uuidof(IDebugClient3), pvoidptr(pDbgClient));
            if (SUCCEEDED(hResult))
            {
                // Save the actual debug client interface UUID
                *pDbgClientId = __uuidof(IDebugClient3);
            }
            else    // Unable to create debug client 3 interface
            {
                // Try to create the debug client 2 interface
                hResult = DebugCreate(__uuidof(IDebugClient2), pvoidptr(pDbgClient));
                if (SUCCEEDED(hResult))
                {
                    // Save the actual debug client interface UUID
                    *pDbgClientId = __uuidof(IDebugClient2);
                }
                else    // Unable to create debug client 2 interface
                {
                    // Try to create the base debug client interface
                    hResult = DebugCreate(__uuidof(IDebugClient), pvoidptr(pDbgClient));
                    if (SUCCEEDED(hResult))
                    {
                        // Save the actual debug client interface UUID
                        *pDbgClientId = __uuidof(IDebugClient);
                    }
                }
            }
        }
    }
    return hResult;

} // createDebugClientInterface

//******************************************************************************

bool
isDebugClientInterface()
{
    PDEBUG_CLIENT       pDbgClient = dbgClient();
    GUID&               dbgClientId = dbg::dbgClientId();
    bool                bDebugClientInterface = false;

    // Check for debug client interface
    if (pDbgClient != NULL)
    {
        // Check for valid debug client interface
        if (IsEqualIID(dbgClientId, __uuidof(IDebugClient5))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient4))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient3))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient2))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient)))
        {
            // Indicate this is a valid debug client interface
            bDebugClientInterface = true;
        }
    }
    return bDebugClientInterface;

} // isDebugClientInterface

//******************************************************************************

bool
isDebugClient2Interface()
{
    PDEBUG_CLIENT       pDbgClient = dbgClient();
    GUID&               dbgClientId = dbg::dbgClientId();
    bool                bDebugClient2Interface = false;

    // Check for debug client interface
    if (pDbgClient != NULL)
    {
        // Check for valid debug client 2 interface
        if (IsEqualIID(dbgClientId, __uuidof(IDebugClient5))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient4))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient3))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient2)))
        {
            // Indicate this is a valid debug client 2 interface
            bDebugClient2Interface = true;
        }
    }
    return bDebugClient2Interface;

} // isDebugClient2Interface

//******************************************************************************

bool
isDebugClient3Interface()
{
    PDEBUG_CLIENT       pDbgClient = dbgClient();
    GUID&               dbgClientId = dbg::dbgClientId();
    bool                bDebugClient3Interface = false;

    // Check for debug client interface
    if (pDbgClient != NULL)
    {
        // Check for valid debug client 3 interface
        if (IsEqualIID(dbgClientId, __uuidof(IDebugClient5))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient4))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient3)))
        {
            // Indicate this is a valid debug client 3 interface
            bDebugClient3Interface = true;
        }
    }
    return bDebugClient3Interface;

} // isDebugClient3Interface

//******************************************************************************

bool
isDebugClient4Interface()
{
    PDEBUG_CLIENT       pDbgClient = dbgClient();
    GUID&               dbgClientId = dbg::dbgClientId();
    bool                bDebugClient4Interface = false;

    // Check for debug client interface
    if (pDbgClient != NULL)
    {
        // Check for valid debug client 4 interface
        if (IsEqualIID(dbgClientId, __uuidof(IDebugClient5))  ||
            IsEqualIID(dbgClientId, __uuidof(IDebugClient4)))
        {
            // Indicate this is a valid debug client 4 interface
            bDebugClient4Interface = true;
        }
    }
    return bDebugClient4Interface;

} // isDebugClient4Interface

//******************************************************************************

bool
isDebugClient5Interface()
{
    PDEBUG_CLIENT       pDbgClient = dbgClient();
    GUID&               dbgClientId = dbg::dbgClientId();
    bool                bDebugClient5Interface = false;

    // Check for debug client interface
    if (pDbgClient != NULL)
    {
        // Check for valid debug client 5 interface
        if (IsEqualIID(dbgClientId, __uuidof(IDebugClient5)))
        {
            // Indicate valid debug client 5 interface
            bDebugClient5Interface = true;
        }
    }
    return bDebugClient5Interface;

} // isDebugClient5Interface

//******************************************************************************

PDEBUG_CLIENT
debugClientInterface()
{
    PDEBUG_CLIENT       pDbgClient = NULL;

    // Check for debug client interface
    if (isDebugClientInterface())
    {
        // Get the debug client interface
        pDbgClient = reinterpret_cast<PDEBUG_CLIENT>(dbgClient());
    }
    return pDbgClient;

} // debugClientInterface

//******************************************************************************

PDEBUG_CLIENT2
debugClient2Interface()
{
    PDEBUG_CLIENT2      pDbgClient2 = NULL;

    // Check for debug client 2 interface
    if (isDebugClient2Interface())
    {
        // Get the debug client 2 interface
        pDbgClient2 = reinterpret_cast<PDEBUG_CLIENT2>(dbgClient());
    }
    return pDbgClient2;

} // debugClient2Interface

//******************************************************************************

PDEBUG_CLIENT3
debugClient3Interface()
{
    PDEBUG_CLIENT3      pDbgClient3 = NULL;

    // Check for debug client 3 interface
    if (isDebugClient3Interface())
    {
        // Get the debug client 3 interface
        pDbgClient3 = reinterpret_cast<PDEBUG_CLIENT3>(dbgClient());
    }
    return pDbgClient3;

} // debugClient3Interface

//******************************************************************************

PDEBUG_CLIENT4
debugClient4Interface()
{
    PDEBUG_CLIENT4      pDbgClient4 = NULL;

    // Check for debug client 4 interface
    if (isDebugClient4Interface())
    {
        // Get the debug client 4 interface
        pDbgClient4 = reinterpret_cast<PDEBUG_CLIENT4>(dbgClient());
    }
    return pDbgClient4;

} // debugClient4Interface

//******************************************************************************

PDEBUG_CLIENT5
debugClient5Interface()
{
    PDEBUG_CLIENT5      pDbgClient5 = NULL;

    // Check for debug client 5 interface
    if (isDebugClient5Interface())
    {
        // Get the debug client 5 interface
        pDbgClient5 = reinterpret_cast<PDEBUG_CLIENT5>(dbgClient());
    }
    return pDbgClient5;

} // debugClient5Interface

//******************************************************************************

static HRESULT
createDebugControlInterface
(
    PDEBUG_CONTROL     *pDbgControl,
    IID                *pDbgControlId
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pDbgControl != NULL);
    assert(pDbgControlId != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Try to create the debug control 4 interface
        hResult = pDbgClient->QueryInterface(__uuidof(IDebugControl4), pvoidptr(pDbgControl));
        if (SUCCEEDED(hResult))
        {
            // Save the actual debug control interface UUID
            *pDbgControlId = __uuidof(IDebugControl4);
        }
        else    // Unable to create debug control 4 interface
        {
            // Try to create the debug control 3 interface
            hResult = pDbgClient->QueryInterface(__uuidof(IDebugControl3), pvoidptr(pDbgControl));
            if (SUCCEEDED(hResult))
            {
                // Save the actual debug control interface UUID
                *pDbgControlId = __uuidof(IDebugControl3);
            }
            else    // Unable to create debug control 3 interface
            {
                // Try to create the debug control 2 interface
                hResult = pDbgClient->QueryInterface(__uuidof(IDebugControl2), pvoidptr(pDbgControl));
                if (SUCCEEDED(hResult))
                {
                    // Save the actual debug control interface UUID
                    *pDbgControlId = __uuidof(IDebugControl2);
                }
                else    // Unable to create debug control 2 interface
                {
                    // Try to create the base debug control interface
                    hResult = pDbgClient->QueryInterface(__uuidof(IDebugControl), pvoidptr(pDbgControl));
                    if (SUCCEEDED(hResult))
                    {
                        // Save the actual debug control interface UUID
                        *pDbgControlId = __uuidof(IDebugControl);
                    }
                }
            }
        }
    }
    return hResult;

} // createDebugControlInterface

//******************************************************************************

bool
isDebugControlInterface()
{
    PDEBUG_CONTROL      pDbgControl = dbgControl();
    GUID&               dbgControlId = dbg::dbgControlId();
    bool                bDebugControlInterface = false;

    // Check for debug control interface
    if (pDbgControl != NULL)
    {
        // Check for valid debug control interface
        if (IsEqualIID(dbgControlId, __uuidof(IDebugControl4))  ||
            IsEqualIID(dbgControlId, __uuidof(IDebugControl3))  ||
            IsEqualIID(dbgControlId, __uuidof(IDebugControl2))  ||
            IsEqualIID(dbgControlId, __uuidof(IDebugControl)))
        {
            // Indicate this is a valid debug control interface
            bDebugControlInterface = true;
        }
    }
    return bDebugControlInterface;

} // isDebugControlInterface

//******************************************************************************

bool
isDebugControl2Interface()
{
    PDEBUG_CONTROL      pDbgControl = dbgControl();
    GUID&               dbgControlId = dbg::dbgControlId();
    bool                bDebugControl2Interface = false;

    // Check for debug control interface
    if (pDbgControl != NULL)
    {
        // Check for valid debug control 2 interface
        if (IsEqualIID(dbgControlId, __uuidof(IDebugControl4))  ||
            IsEqualIID(dbgControlId, __uuidof(IDebugControl3))  ||
            IsEqualIID(dbgControlId, __uuidof(IDebugControl2)))
        {
            // Indicate this is a valid debug control 2 interface
            bDebugControl2Interface = true;
        }
    }
    return bDebugControl2Interface;

} // isDebugControl2Interface

//******************************************************************************

bool
isDebugControl3Interface()
{
    PDEBUG_CONTROL      pDbgControl = dbgControl();
    GUID&               dbgControlId = dbg::dbgControlId();
    bool                bDebugControl3Interface = false;

    // Check for debug control interface
    if (pDbgControl != NULL)
    {
        // Check for valid debug control 3 interface
        if (IsEqualIID(dbgControlId, __uuidof(IDebugControl4))  ||
            IsEqualIID(dbgControlId, __uuidof(IDebugControl3)))
        {
            // Indicate this is a valid debug control 3 interface
            bDebugControl3Interface = true;
        }
    }
    return bDebugControl3Interface;

} // isDebugControl3Interface

//******************************************************************************

bool
isDebugControl4Interface()
{
    PDEBUG_CONTROL      pDbgControl = dbgControl();
    GUID&               dbgControlId = dbg::dbgControlId();
    bool                bDebugControl4Interface = false;

    // Check for debug control interface
    if (pDbgControl != NULL)
    {
        // Check for valid debug control 4 interface
        if (IsEqualIID(dbgControlId, __uuidof(IDebugControl4)))
        {
            // Indicate this is a valid debug control 4 interface
            bDebugControl4Interface = true;
        }
    }
    return bDebugControl4Interface;

} // isDebugControl4Interface

//******************************************************************************

PDEBUG_CONTROL
debugControlInterface()
{
    PDEBUG_CONTROL      pDbgControl = NULL;

    // Check for debug control interface
    if (isDebugControlInterface())
    {
        // Get the debug control interface
        pDbgControl = reinterpret_cast<PDEBUG_CONTROL>(dbgControl());
    }
    return pDbgControl;

} // debugControlInterface

//******************************************************************************

PDEBUG_CONTROL2
debugControl2Interface()
{
    PDEBUG_CONTROL2     pDbgControl2 = NULL;

    // Check for debug control 2 interface
    if (isDebugControl2Interface())
    {
        // Get the debug control 2 interface
        pDbgControl2 = reinterpret_cast<PDEBUG_CONTROL2>(dbgControl());
    }
    return pDbgControl2;

} // debugControl2Interface

//******************************************************************************

PDEBUG_CONTROL3
debugControl3Interface()
{
    PDEBUG_CONTROL3     pDbgControl3 = NULL;

    // Check for debug control 3 interface
    if (isDebugControl3Interface())
    {
        // Get the debug control 3 interface
        pDbgControl3 = reinterpret_cast<PDEBUG_CONTROL3>(dbgControl());
    }
    return pDbgControl3;

} // debugControl3Interface

//******************************************************************************

PDEBUG_CONTROL4
debugControl4Interface()
{
    PDEBUG_CONTROL4     pDbgControl4 = NULL;

    // Check for debug control 4 interface
    if (isDebugControl4Interface())
    {
        // Get the debug control 4 interface
        pDbgControl4 = reinterpret_cast<PDEBUG_CONTROL4>(dbgControl());
    }
    return pDbgControl4;

} // debugControl4Interface

//******************************************************************************

static HRESULT
createDebugDataSpacesInterface
(
    PDEBUG_DATA_SPACES *pDbgDataSpaces,
    IID                *pDbgDataSpacesId
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pDbgDataSpaces != NULL);
    assert(pDbgDataSpacesId != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Try to create the debug data spaces 4 interface
        hResult = pDbgClient->QueryInterface(__uuidof(IDebugDataSpaces4), pvoidptr(pDbgDataSpaces));
        if (SUCCEEDED(hResult))
        {
            // Save the actual debug data spaces interface UUID
            *pDbgDataSpacesId = __uuidof(IDebugDataSpaces4);
        }
        else    // Unable to create debug data spaces 4 interface
        {
            // Try to create the debug data spaces 3 interface
            hResult = pDbgClient->QueryInterface(__uuidof(IDebugDataSpaces3), pvoidptr(pDbgDataSpaces));
            if (SUCCEEDED(hResult))
            {
                // Save the actual debug data spaces interface UUID
                *pDbgDataSpacesId = __uuidof(IDebugDataSpaces3);
            }
            else    // Unable to create debug data spaces 3 interface
            {
                // Try to create the debug data spaces 2 interface
                hResult = pDbgClient->QueryInterface(__uuidof(IDebugDataSpaces2), pvoidptr(pDbgDataSpaces));
                if (SUCCEEDED(hResult))
                {
                    // Save the actual debug data spaces interface UUID
                    *pDbgDataSpacesId = __uuidof(IDebugDataSpaces2);
                }
                else    // Unable to create debug data spaces 2 interface
                {
                    // Try to create the base debug data spaces interface
                    hResult = pDbgClient->QueryInterface(__uuidof(IDebugDataSpaces), pvoidptr(pDbgDataSpaces));
                    if (SUCCEEDED(hResult))
                    {
                        // Save the actual debug data spaces interface UUID
                        *pDbgDataSpacesId = __uuidof(IDebugDataSpaces);
                    }
                }
            }
        }
    }
    return hResult;

} // createDebugDataSpacesInterface

//******************************************************************************

bool
isDebugDataSpacesInterface()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = dbgDataSpaces();
    GUID&               dbgDataSpacesId = dbg::dbgDataSpacesId();
    bool                bDebugDataSpacesInterface = false;

    // Check for debug data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Check for valid debug data spaces interface
        if (IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces4))  ||
            IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces3))  ||
            IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces2))  ||
            IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces)))
        {
            // Indicate this is a valid debug data spaces interface
            bDebugDataSpacesInterface = true;
        }
    }
    return bDebugDataSpacesInterface;

} // isDebugDataSpacesInterface

//******************************************************************************

bool
isDebugDataSpaces2Interface()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = dbgDataSpaces();
    GUID&               dbgDataSpacesId = dbg::dbgDataSpacesId();
    bool                bDebugDataSpaces2Interface = false;

    // Check for debug data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Check for valid debug data spaces 2 interface
        if (IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces4))  ||
            IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces3))  ||
            IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces2)))
        {
            // Indicate this is a valid debug data spaces 2 interface
            bDebugDataSpaces2Interface = true;
        }
    }
    return bDebugDataSpaces2Interface;

} // isDebugDataSpaces2Interface

//******************************************************************************

bool
isDebugDataSpaces3Interface()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = dbgDataSpaces();
    GUID&               dbgDataSpacesId = dbg::dbgDataSpacesId();
    bool                bDebugDataSpaces3Interface = false;

    // Check for debug data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Check for valid debug data spaces 3 interface
        if (IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces4))  ||
            IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces3)))
        {
            // Indicate this is a valid debug data spaces 3 interface
            bDebugDataSpaces3Interface = true;
        }
    }
    return bDebugDataSpaces3Interface;

} // isDebugDataSpaces3Interface

//******************************************************************************

bool
isDebugDataSpaces4Interface()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = dbgDataSpaces();
    GUID&               dbgDataSpacesId = dbg::dbgDataSpacesId();
    bool                bDebugDataSpaces4Interface = false;

    // Check for debug data spaces interface
    if (pDbgDataSpaces != NULL)
    {
        // Check for valid debug data spaces 4 interface
        if (IsEqualIID(dbgDataSpacesId, __uuidof(IDebugDataSpaces4)))
        {
            // Indicate this is a valid debug data spaces 4 interface
            bDebugDataSpaces4Interface = true;
        }
    }
    return bDebugDataSpaces4Interface;

} // isDebugDataSpaces4Interface

//******************************************************************************

PDEBUG_DATA_SPACES
debugDataSpacesInterface()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = NULL;

    // Check for debug data spaces interface
    if (isDebugDataSpacesInterface())
    {
        // Get the debug data spaces interface
        pDbgDataSpaces = reinterpret_cast<PDEBUG_DATA_SPACES>(dbgDataSpaces());
    }
    return pDbgDataSpaces;

} // debugDataSpacesInterface

//******************************************************************************

PDEBUG_DATA_SPACES2
debugDataSpaces2Interface()
{
    PDEBUG_DATA_SPACES2 pDbgDataSpaces2 = NULL;

    // Check for debug data spaces 2 interface
    if (isDebugDataSpaces2Interface())
    {
        // Get the debug data spaces 2 interface
        pDbgDataSpaces2 = reinterpret_cast<PDEBUG_DATA_SPACES2>(dbgDataSpaces());
    }
    return pDbgDataSpaces2;

} // debugDataSpaces2Interface

//******************************************************************************

PDEBUG_DATA_SPACES3
debugDataSpaces3Interface()
{
    PDEBUG_DATA_SPACES3 pDbgDataSpaces3 = NULL;

    // Check for debug data spaces 3 interface
    if (isDebugDataSpaces3Interface())
    {
        // Get the debug data spaces 3 interface
        pDbgDataSpaces3 = reinterpret_cast<PDEBUG_DATA_SPACES3>(dbgDataSpaces());
    }
    return pDbgDataSpaces3;

} // debugDataSpaces3Interface

//******************************************************************************

PDEBUG_DATA_SPACES4
debugDataSpaces4Interface()
{
    PDEBUG_DATA_SPACES4 pDbgDataSpaces4 = NULL;

    // Check for debug data spaces 4 interface
    if (isDebugDataSpaces4Interface())
    {
        // Get the debug data spaces 4 interface
        pDbgDataSpaces4 = reinterpret_cast<PDEBUG_DATA_SPACES4>(dbgDataSpaces());
    }
    return pDbgDataSpaces4;

} // debugDataSpaces4Interface

//******************************************************************************

static HRESULT
createDebugRegistersInterface
(
    PDEBUG_REGISTERS   *pDbgRegisters,
    IID                *pDbgRegistersId
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pDbgRegisters != NULL);
    assert(pDbgRegistersId != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Try to create the debug registers 2 interface
        hResult = pDbgClient->QueryInterface(__uuidof(IDebugRegisters2), pvoidptr(pDbgRegisters));
        if (SUCCEEDED(hResult))
        {
            // Save the actual debug registers interface UUID
            *pDbgRegistersId = __uuidof(IDebugRegisters2);
        }
        else    // Unable to create debug registers 2 interface
        {
            // Try to create the base debug registers interface
            hResult = pDbgClient->QueryInterface(__uuidof(IDebugRegisters), pvoidptr(pDbgRegisters));
            if (SUCCEEDED(hResult))
            {
                // Save the actual debug registers interface UUID
                *pDbgRegistersId = __uuidof(IDebugRegisters);
            }
        }
    }
    return hResult;

} // createDebugRegistersInterface

//******************************************************************************

bool
isDebugRegistersInterface()
{
    PDEBUG_REGISTERS    pDbgRegisters = dbgRegisters();
    GUID&               dbgRegistersId = dbg::dbgRegistersId();
    bool                bDebugRegistersInterface = false;

    // Check for debug registers interface
    if (pDbgRegisters != NULL)
    {
        // Check for valid debug registers interface
        if (IsEqualIID(dbgRegistersId, __uuidof(IDebugRegisters2))    ||
            IsEqualIID(dbgRegistersId, __uuidof(IDebugRegisters)))
        {
            // Indicate this is a valid debug registers interface
            bDebugRegistersInterface = true;
        }
    }
    return bDebugRegistersInterface;

} // isDebugRegistersInterface

//******************************************************************************

bool
isDebugRegisters2Interface()
{
    PDEBUG_REGISTERS    pDbgRegisters = dbgRegisters();
    GUID&               dbgRegistersId = dbg::dbgRegistersId();
    bool                bDebugRegisters2Interface = false;

    // Check for debug registers interface
    if (pDbgRegisters != NULL)
    {
        // Check for valid debug registers 2 interface
        if (IsEqualIID(dbgRegistersId, __uuidof(IDebugRegisters2)))
        {
            // Indicate this is a valid debug registers 2 interface
            bDebugRegisters2Interface = true;
        }
    }
    return bDebugRegisters2Interface;

} // isDebugRegisters2Interface

//******************************************************************************

PDEBUG_REGISTERS
debugRegistersInterface()
{
    PDEBUG_REGISTERS    pDbgRegisters = NULL;

    // Check for debug registers interface
    if (isDebugRegistersInterface())
    {
        // Get the debug registers interface
        pDbgRegisters = reinterpret_cast<PDEBUG_REGISTERS>(dbgRegisters());
    }
    return pDbgRegisters;

} // debugRegistersInterface

//******************************************************************************

PDEBUG_REGISTERS2
debugRegisters2Interface()
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = NULL;

    // Check for debug registers 2 interface
    if (isDebugRegisters2Interface())
    {
        // Get the debug registers 2 interface
        pDbgRegisters2 = reinterpret_cast<PDEBUG_REGISTERS2>(dbgRegisters());
    }
    return pDbgRegisters2;

} // debugRegisters2Interface

//******************************************************************************

static HRESULT
createDebugSymbolsInterface
(
    PDEBUG_SYMBOLS     *pDbgSymbols,
    IID                *pDbgSymbolsId
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pDbgSymbols != NULL);
    assert(pDbgSymbolsId != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Try to create the debug symbols 3 interface
        hResult = pDbgClient->QueryInterface(__uuidof(IDebugSymbols3), pvoidptr(pDbgSymbols));
        if (SUCCEEDED(hResult))
        {
            // Save the actual debug symbols interface UUID
            *pDbgSymbolsId = __uuidof(IDebugSymbols3);
        }
        else    // Unable to create debug symbols 3 interface
        {
            // Try to create the debug symbols 2 interface
            hResult = pDbgClient->QueryInterface(__uuidof(IDebugSymbols2), pvoidptr(pDbgSymbols));
            if (SUCCEEDED(hResult))
            {
                // Save the actual debug symbols interface UUID
                *pDbgSymbolsId = __uuidof(IDebugSymbols2);
            }
            else    // Unable to create debug symbols 2 interface
            {
                // Try to create the base debug symbols interface
                hResult = pDbgClient->QueryInterface(__uuidof(IDebugSymbols), pvoidptr(pDbgSymbols));
                if (SUCCEEDED(hResult))
                {
                    // Save the actual debug symbols interface UUID
                    *pDbgSymbolsId = __uuidof(IDebugSymbols);
                }
            }
        }
    }
    return hResult;

} // createDebugSymbolsInterface

//******************************************************************************

bool
isDebugSymbolsInterface()
{
    PDEBUG_SYMBOLS      pDbgSymbols = dbgSymbols();
    GUID&               dbgSymbolsId = dbg::dbgSymbolsId();
    bool                bDebugSymbolsInterface = false;

    // Check for debug symbols interface
    if (pDbgSymbols != NULL)
    {
        // Check for valid debug symbols interface
        if (IsEqualIID(dbgSymbolsId, __uuidof(IDebugSymbols3))    ||
            IsEqualIID(dbgSymbolsId, __uuidof(IDebugSymbols2))    ||
            IsEqualIID(dbgSymbolsId, __uuidof(IDebugSymbols)))
        {
            // Indicate this is a valid debug symbols interface
            bDebugSymbolsInterface = true;
        }
    }
    return bDebugSymbolsInterface;

} // isDebugSymbolsInterface

//******************************************************************************

bool
isDebugSymbols2Interface()
{
    PDEBUG_SYMBOLS      pDbgSymbols = dbgSymbols();
    GUID&               dbgSymbolsId = dbg::dbgSymbolsId();
    bool                bDebugSymbols2Interface = false;

    // Check for debug symbols interface
    if (pDbgSymbols != NULL)
    {
        // Check for valid debug symbols 2 interface
        if (IsEqualIID(dbgSymbolsId, __uuidof(IDebugSymbols3))  ||
            IsEqualIID(dbgSymbolsId, __uuidof(IDebugSymbols2)))
        {
            // Indicate this is a valid debug symbols 2 interface
            bDebugSymbols2Interface = true;
        }
    }
    return bDebugSymbols2Interface;

} // isDebugSymbols2Interface

//******************************************************************************

bool
isDebugSymbols3Interface()
{
    PDEBUG_SYMBOLS      pDbgSymbols = dbgSymbols();
    GUID&               dbgSymbolsId = dbg::dbgSymbolsId();
    bool                bDebugSymbols3Interface = false;

    // Check for debug symbols interface
    if (pDbgSymbols != NULL)
    {
        // Check for valid debug symbols 3 interface
        if (IsEqualIID(dbgSymbolsId, __uuidof(IDebugSymbols3)))
        {
            // Indicate this is a valid debug symbols 3 interface
            bDebugSymbols3Interface = true;
        }
    }
    return bDebugSymbols3Interface;

} // isDebugSymbols3Interface

//******************************************************************************

PDEBUG_SYMBOLS
debugSymbolsInterface()
{
    PDEBUG_SYMBOLS      pDbgSymbols = NULL;

    // Check for debug symbols interface
    if (isDebugSymbolsInterface())
    {
        // Get the debug symbols interface
        pDbgSymbols = reinterpret_cast<PDEBUG_SYMBOLS>(dbgSymbols());
    }
    return pDbgSymbols;

} // debugSymbolsInterface

//******************************************************************************

PDEBUG_SYMBOLS2
debugSymbols2Interface()
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = NULL;

    // Check for debug symbols 2 interface
    if (isDebugSymbols2Interface())
    {
        // Get the debug symbols 2 interface
        pDbgSymbols2 = reinterpret_cast<PDEBUG_SYMBOLS2>(dbgSymbols());
    }
    return pDbgSymbols2;

} // debugSymbols2Interface

//******************************************************************************

PDEBUG_SYMBOLS3
debugSymbols3Interface()
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = NULL;

    // Check for debug symbols 3 interface
    if (isDebugSymbols3Interface())
    {
        // Get the debug symbols 3 interface
        pDbgSymbols3 = reinterpret_cast<PDEBUG_SYMBOLS3>(dbgSymbols());
    }
    return pDbgSymbols3;

} // debugSymbols3Interface

//******************************************************************************

static HRESULT
createDebugSystemObjectsInterface
(
    PDEBUG_SYSTEM_OBJECTS *pDbgSystemObjects,
    IID                *pDbgSystemObjectsId
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pDbgSystemObjects != NULL);
    assert(pDbgSystemObjectsId != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Try to create the debug system objects 4 interface
        hResult = pDbgClient->QueryInterface(__uuidof(IDebugSystemObjects4), pvoidptr(pDbgSystemObjects));
        if (SUCCEEDED(hResult))
        {
            // Save the actual debug system objects interface UUID
            *pDbgSystemObjectsId = __uuidof(IDebugSystemObjects4);
        }
        else    // Unable to create debug system objects 4 interface
        {
            // Try to create the debug system objects 3 interface
            hResult = pDbgClient->QueryInterface(__uuidof(IDebugSystemObjects3), pvoidptr(pDbgSystemObjects));
            if (SUCCEEDED(hResult))
            {
                // Save the actual debug system objects interface UUID
                *pDbgSystemObjectsId = __uuidof(IDebugSystemObjects3);
            }
            else    // Unable to create debug system objects 3 interface
            {
                // Try to create the debug system object 2 interface
                hResult = pDbgClient->QueryInterface(__uuidof(IDebugSystemObjects2), pvoidptr(pDbgSystemObjects));
                if (SUCCEEDED(hResult))
                {
                    // Save the actual debug system objects interface UUID
                    *pDbgSystemObjectsId = __uuidof(IDebugSystemObjects2);
                }
                else    // Unable to create debug system objects 2 interface
                {
                    // Try to create the base debug system objects interface
                    hResult = pDbgClient->QueryInterface(__uuidof(IDebugSystemObjects), pvoidptr(pDbgSystemObjects));
                    if (SUCCEEDED(hResult))
                    {
                        // Save the actual debug system objects interface UUID
                        *pDbgSystemObjectsId = __uuidof(IDebugSystemObjects);
                    }
                }
            }
        }
    }
    return hResult;

} // createDebugSystemObjectsInterface

//******************************************************************************

bool
isDebugSystemObjectsInterface()
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = dbgSystemObjects();
    GUID&               dbgSystemObjectsId = dbg::dbgSystemObjectsId();
    bool                bDebugSystemObjectsInterface = false;

    // Check for debug system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Check for valid debug system objects interface
        if (IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects4))    ||
            IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects3))    ||
            IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects2))    ||
            IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects)))
        {
            // Indicate this is a valid debug system objects spaces interface
            bDebugSystemObjectsInterface = true;
        }
    }
    return bDebugSystemObjectsInterface;

} // isDebugSystemObjectsInterface

//******************************************************************************

bool
isDebugSystemObjects2Interface()
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = dbgSystemObjects();
    GUID&               dbgSystemObjectsId = dbg::dbgSystemObjectsId();
    bool                bDebugSystemObjects2Interface = false;

    // Check for debug system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Check for valid debug system objects 2 interface
        if (IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects4))    ||
            IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects3))    ||
            IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects2)))
        {
            // Indicate this is a valid debug system objects 2 interface
            bDebugSystemObjects2Interface = true;
        }
    }
    return bDebugSystemObjects2Interface;

} // isDebugSystemObjects2Interface

//******************************************************************************

bool
isDebugSystemObjects3Interface()
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = dbgSystemObjects();
    GUID&               dbgSystemObjectsId = dbg::dbgSystemObjectsId();
    bool                bDebugSystemObjects3Interface = false;

    // Check for debug system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Check for valid debug system objects 3 interface
        if (IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects4))    ||
            IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects3)))
        {
            // Indicate this is a valid debug system objects 3 interface
            bDebugSystemObjects3Interface = true;
        }
    }
    return bDebugSystemObjects3Interface;

} // isDebugSystemObjects3Interface

//******************************************************************************

bool
isDebugSystemObjects4Interface()
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = dbgSystemObjects();
    GUID&               dbgSystemObjectsId = dbg::dbgSystemObjectsId();
    bool                bDebugSystemObjects4Interface = false;

    // Check for debug system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Check for valid debug system objects 4 interface
        if (IsEqualIID(dbgSystemObjectsId, __uuidof(IDebugSystemObjects4)))
        {
            // Indicate this is a valid debug system objects 4 interface
            bDebugSystemObjects4Interface = true;
        }
    }
    return bDebugSystemObjects4Interface;

} // isDebugSystemObjects4Interface

//******************************************************************************

PDEBUG_SYSTEM_OBJECTS
debugSystemObjectsInterface()
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = NULL;

    // Check for debug system objects interface
    if (isDebugSystemObjectsInterface())
    {
        // Get the debug system objects interface
        pDbgSystemObjects = reinterpret_cast<PDEBUG_SYSTEM_OBJECTS>(dbgSystemObjects());
    }
    return pDbgSystemObjects;

} // debugSystemObjectsInterface

//******************************************************************************

PDEBUG_SYSTEM_OBJECTS2
debugSystemObjects2Interface()
{
    PDEBUG_SYSTEM_OBJECTS2 pDbgSystemObjects2 = NULL;

    // Check for debug system objects 2 interface
    if (isDebugSystemObjects2Interface())
    {
        // Get the debug system objects 2 interface
        pDbgSystemObjects2 = reinterpret_cast<PDEBUG_SYSTEM_OBJECTS2>(dbgSystemObjects());
    }
    return pDbgSystemObjects2;

} // debugSystemObjects2Interface

//******************************************************************************

PDEBUG_SYSTEM_OBJECTS3
debugSystemObjects3Interface()
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = NULL;

    // Check for debug system objects 3 interface
    if (isDebugSystemObjects3Interface())
    {
        // Get the debug system objects 3 interface
        pDbgSystemObjects3 = reinterpret_cast<PDEBUG_SYSTEM_OBJECTS3>(dbgSystemObjects());
    }
    return pDbgSystemObjects3;

} // debugSystemObjects3Interface

//******************************************************************************

PDEBUG_SYSTEM_OBJECTS4
debugSystemObjects4Interface()
{
    PDEBUG_SYSTEM_OBJECTS4 pDbgSystemObjects4 = NULL;

    // Check for debug system objects 4 interface
    if (isDebugSystemObjects4Interface())
    {
        // Get the debug system objects 4 interface
        pDbgSystemObjects4 = reinterpret_cast<PDEBUG_SYSTEM_OBJECTS4>(dbgSystemObjects());
    }
    return pDbgSystemObjects4;

} // debugSystemObjects4Interface

//******************************************************************************

static HRESULT
createDebugAdvancedInterface
(
    PDEBUG_ADVANCED    *pDbgAdvanced,
    IID                *pDbgAdvancedId
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pDbgAdvanced != NULL);
    assert(pDbgAdvancedId != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Try to create the debug advanced 3 interface
        hResult = pDbgClient->QueryInterface(__uuidof(IDebugAdvanced3), pvoidptr(pDbgAdvanced));
        if (SUCCEEDED(hResult))
        {
            // Save the actual debug advanced interface UUID
            *pDbgAdvancedId = __uuidof(IDebugAdvanced3);
        }
        else    // Unable to create debug advanced 3 interface
        {
            // Try to create the debug advanced 2 interface
            hResult = pDbgClient->QueryInterface(__uuidof(IDebugAdvanced2), pvoidptr(pDbgAdvanced));
            if (SUCCEEDED(hResult))
            {
                // Save the actual debug advanced interface UUID
                *pDbgAdvancedId = __uuidof(IDebugAdvanced2);
            }
            else    // Unable to create debug advanced 2 interface
            {
                // Try to create the base debug advanced interface
                hResult = pDbgClient->QueryInterface(__uuidof(IDebugAdvanced), pvoidptr(pDbgAdvanced));
                if (SUCCEEDED(hResult))
                {
                    // Save the actual debug advanced interface UUID
                    *pDbgAdvancedId = __uuidof(IDebugAdvanced);
                }
            }
        }
    }
    return hResult;

} // createDebugAdvancedInterface

//******************************************************************************

static PDEBUG_CLIENT
dbgClient()
{
    PDEBUG_CLIENT       pDbgClient = NULL;

    // Get the debug client interface
    pDbgClient = s_DbgInterfaces.pDbgClient;

    return pDbgClient;

} // dbgClient

//******************************************************************************

static PDEBUG_CONTROL
dbgControl()
{
    PDEBUG_CONTROL      pDbgControl = NULL;

    // Get the debug control interface
    pDbgControl = s_DbgInterfaces.pDbgControl;

    return pDbgControl;

} // dbgControl

//******************************************************************************

static PDEBUG_DATA_SPACES
dbgDataSpaces()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces = NULL;

    // Get the debug data spaces interface
    pDbgDataSpaces = s_DbgInterfaces.pDbgDataSpaces;

    return pDbgDataSpaces;

} // dbgDataSpaces

//******************************************************************************

static PDEBUG_REGISTERS
dbgRegisters()
{
    PDEBUG_REGISTERS    pDbgRegisters = NULL;

    // Get the debug registers interface
    pDbgRegisters = s_DbgInterfaces.pDbgRegisters;

    return pDbgRegisters;

} // dbgRegisters

//******************************************************************************

static PDEBUG_SYMBOLS
dbgSymbols()
{
    PDEBUG_SYMBOLS      pDbgSymbols = NULL;

    // Get the debug symbols interface
    pDbgSymbols = s_DbgInterfaces.pDbgSymbols;

    return pDbgSymbols;

} // dbgSymbols

//******************************************************************************

static PDEBUG_SYSTEM_OBJECTS
dbgSystemObjects()
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = NULL;

    // Get the debug system objects interface
    pDbgSystemObjects = s_DbgInterfaces.pDbgSystemObjects;

    return pDbgSystemObjects;

} // dbgSystemObjects

//******************************************************************************

static PDEBUG_ADVANCED
dbgAdvanced()
{
    PDEBUG_ADVANCED     pDbgAdvanced = NULL;

    // Get the debug advanced interface
    pDbgAdvanced = s_DbgInterfaces.pDbgAdvanced;

    return pDbgAdvanced;

} // dbgAdvanced

//******************************************************************************

static PDEBUG_SYMBOL_GROUP2
dbgSymbolGroup()
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = NULL;

    // Get the debug symbol group interface
    pDbgSymbolGroup = s_DbgInterfaces.pDbgSymbolGroup;

    return pDbgSymbolGroup;

} // dbgSymbolGroup

//******************************************************************************

static PDEBUG_BREAKPOINT3
dbgBreakpoint()
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = NULL;

    // Get the debug breakpoint interface
    pDbgBreakpoint = s_DbgInterfaces.pDbgBreakpoint;

    return pDbgBreakpoint;

} // dbgBreakpoint

//******************************************************************************

static GUID&
dbgClientId()
{
    PDEBUG_CLIENT       pDbgClient;

    // Get the debug client interface
    pDbgClient = s_DbgInterfaces.pDbgClient;
    if (pDbgClient != NULL)
    {
        // Return debug client interface GUID
        return s_DbgInterfaces.dbgClientId;
    }
    // Return unknown interface GUID (All zeros)
    return s_DbgUnknownId;

} // dbgClientId

//******************************************************************************

static GUID&
dbgControlId()
{
    PDEBUG_CONTROL      pDbgControl;

    // Get the debug control interface
    pDbgControl = s_DbgInterfaces.pDbgControl;
    if (pDbgControl != NULL)
    {
        // Return debug control interface GUID
        return s_DbgInterfaces.dbgControlId;
    }
    // Return unknown interface GUID (All zeros)
    return s_DbgUnknownId;

} // dbgControlId

//******************************************************************************

static GUID&
dbgDataSpacesId()
{
    PDEBUG_DATA_SPACES  pDbgDataSpaces;

    // Get the debug data spaces interface
    pDbgDataSpaces = s_DbgInterfaces.pDbgDataSpaces;
    if (pDbgDataSpaces != NULL)
    {
        // Return debug data spaces interface GUID
        return s_DbgInterfaces.dbgDataSpacesId;
    }
    // Return unknown interface GUID (All zeros)
    return s_DbgUnknownId;

} // dbgDataSpacesId

//******************************************************************************

static GUID&
dbgRegistersId()
{
    PDEBUG_REGISTERS    pDbgRegisters;

    // Get the debug registers interface
    pDbgRegisters = s_DbgInterfaces.pDbgRegisters;
    if (pDbgRegisters != NULL)
    {
        // Return debug registers interface GUID
        return s_DbgInterfaces.dbgRegistersId;
    }
    // Return unknown interface GUID (All zeros)
    return s_DbgUnknownId;

} // dbgRegistersId

//******************************************************************************

static GUID&
dbgSymbolsId()
{
    PDEBUG_SYMBOLS      pDbgSymbols;

    // Get the debug symbols interface
    pDbgSymbols = s_DbgInterfaces.pDbgSymbols;
    if (pDbgSymbols != NULL)
    {
        // Return debug symbols interface GUID
        return s_DbgInterfaces.dbgSymbolsId;
    }
    // Return unknown interface GUID (All zeros)
    return s_DbgUnknownId;

} // dbgSymbolsId

//******************************************************************************

static GUID&
dbgSystemObjectsId()
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects;

    // Get the debug system objects interface
    pDbgSystemObjects = s_DbgInterfaces.pDbgSystemObjects;
    if (pDbgSystemObjects != NULL)
    {
        // Return debug system objects interface GUID
        return s_DbgInterfaces.dbgSystemObjectsId;
    }
    // Return unknown interface GUID (All zeros)
    return s_DbgUnknownId;

} // dbgSystemObjectsId

//******************************************************************************

static GUID&
dbgAdvancedId()
{
    PDEBUG_ADVANCED     pDbgAdvanced;

    // Get the debug advanced interface
    pDbgAdvanced = s_DbgInterfaces.pDbgAdvanced;
    if (pDbgAdvanced != NULL)
    {
        // Return debug advanced interface GUID
        return s_DbgInterfaces.dbgAdvancedId;
    }
    // Return unknown interface GUID (All zeros)
    return s_DbgUnknownId;

} // dbgAdvancedId

//******************************************************************************

PLWSTOM_DEBUG_INPUT_CALLBACKS
dbgInput()
{
    PDEBUG_INPUT_CALLBACKS pDbgInput = s_DbgInterfaces.pDbgInput;

    // Return debug input callbacks as interface
    return (funcptr(PLWSTOM_DEBUG_INPUT_CALLBACKS, pDbgInput));

} // dbgInput

//******************************************************************************

PLWSTOM_DEBUG_OUTPUT_CALLBACKS
dbgOutput()
{
    PDEBUG_OUTPUT_CALLBACKS pDbgOutput = s_DbgInterfaces.pDbgOutput;

    // Return debug output callbacks as interface
    return (funcptr(PLWSTOM_DEBUG_OUTPUT_CALLBACKS, pDbgOutput));

} // dbgOutput

//******************************************************************************

PLWSTOM_DEBUG_EVENT_CALLBACKS
dbgEvent()
{
    PDEBUG_EVENT_CALLBACKS pDbgEvent = s_DbgInterfaces.pDbgEvent;

    // Return debug event callbacks as interface
    return (funcptr(PLWSTOM_DEBUG_EVENT_CALLBACKS, pDbgEvent));

} // dbgEvent

//******************************************************************************

bool
isDebugAdvancedInterface()
{
    PDEBUG_ADVANCED     pDbgAdvanced = dbgAdvanced();
    GUID&               dbgAdvancedId = dbg::dbgAdvancedId();
    bool                bDebugAdvancedInterface = false;

    // Check for debug advanced interface
    if (pDbgAdvanced != NULL)
    {
        // Check for valid debug advanced interface
        if (IsEqualIID(dbgAdvancedId, __uuidof(IDebugAdvanced3))  ||
            IsEqualIID(dbgAdvancedId, __uuidof(IDebugAdvanced2))  ||
            IsEqualIID(dbgAdvancedId, __uuidof(IDebugAdvanced)))
        {
            // Indicate this is a valid debug advanced interface
            bDebugAdvancedInterface = true;
        }
    }
    return bDebugAdvancedInterface;

} // isDebugAdvancedInterface

//******************************************************************************

bool
isDebugAdvanced2Interface()
{
    PDEBUG_ADVANCED     pDbgAdvanced = dbgAdvanced();
    GUID&               dbgAdvancedId = dbg::dbgAdvancedId();
    bool                bDebugAdvanced2Interface = false;

    // Check for debug advanced interface
    if (pDbgAdvanced != NULL)
    {
        // Check for valid debug advanced 2 interface
        if (IsEqualIID(dbgAdvancedId, __uuidof(IDebugAdvanced3))  ||
            IsEqualIID(dbgAdvancedId, __uuidof(IDebugAdvanced2)))
        {
            // Indicate this is a valid debug advanced 2 interface
            bDebugAdvanced2Interface = true;
        }
    }
    return bDebugAdvanced2Interface;

} // isDebugAdvanced2Interface

//******************************************************************************

bool
isDebugAdvanced3Interface()
{
    PDEBUG_ADVANCED     pDbgAdvanced = dbgAdvanced();
    GUID&               dbgAdvancedId = dbg::dbgAdvancedId();
    bool                bDebugAdvanced3Interface = false;

    // Check for debug advanced interface
    if (pDbgAdvanced != NULL)
    {
        // Check for valid debug advanced 3 interface
        if (IsEqualIID(dbgAdvancedId, __uuidof(IDebugAdvanced3)))
        {
            // Indicate this is a valid debug advanced 3 interface
            bDebugAdvanced3Interface = true;
        }
    }
    return bDebugAdvanced3Interface;

} // isDebugAdvanced3Interface

//******************************************************************************

PDEBUG_ADVANCED
debugAdvancedInterface()
{
    PDEBUG_ADVANCED     pDbgAdvanced = NULL;

    // Check for debug advanced interface
    if (isDebugAdvancedInterface())
    {
        // Get the debug advanced interface
        pDbgAdvanced = reinterpret_cast<PDEBUG_ADVANCED>(dbgAdvanced());
    }
    return pDbgAdvanced;

} // debugAdvancedInterface

//******************************************************************************

PDEBUG_ADVANCED2
debugAdvanced2Interface()
{
    PDEBUG_ADVANCED2    pDbgAdvanced2 = NULL;

    // Check for debug advanced 2 interface
    if (isDebugAdvanced2Interface())
    {
        // Get the debug advanced 2 interface
        pDbgAdvanced2 = reinterpret_cast<PDEBUG_ADVANCED2>(dbgAdvanced());
    }
    return pDbgAdvanced2;

} // debugAdvanced2Interface

//******************************************************************************

PDEBUG_ADVANCED3
debugAdvanced3Interface()
{
    PDEBUG_ADVANCED3    pDbgAdvanced3 = NULL;

    // Check for debug advanced 3 interface
    if (isDebugAdvanced3Interface())
    {
        // Get the debug advanced 3 interface
        pDbgAdvanced3 = reinterpret_cast<PDEBUG_ADVANCED3>(dbgAdvanced());
    }
    return pDbgAdvanced3;

} // debugControl3Interface

//******************************************************************************

PDEBUG_SYMBOL_GROUP2
debugSymbolGroupInterface()
{
    // Simply return the symbol group interface (May be none)
    return dbgSymbolGroup();

} // debugSymbolGroupInterface

//******************************************************************************

PDEBUG_BREAKPOINT3
debugBreakpointInterface()
{
    // Simply return the breakpoint interface (May be none)
    return dbgBreakpoint();

} // debugBreakpointInterface

//******************************************************************************

HRESULT
createDebugInputInterface
(
    PDEBUG_INPUT_CALLBACKS *pDbgInput
)
{
    HRESULT             hResult;

    // Try to create the custom debug input callback interface
    hResult = createInterface(__uuidof(ILwstomDebugInputCallbacks), pvoidptr(pDbgInput));

    return hResult;

} // createDebugInputInterface

//******************************************************************************

HRESULT
createDebugOutputInterface
(
    PDEBUG_OUTPUT_CALLBACKS *pDbgOutput
)
{
    HRESULT             hResult;

    // Try to create the custom debug output callback interface
    hResult = createInterface(__uuidof(ILwstomDebugOutputCallbacks), pvoidptr(pDbgOutput));

    return hResult;

} // createDebugOutputInterface

//******************************************************************************

HRESULT
createDebugEventInterface
(
    PDEBUG_EVENT_CALLBACKS *pDbgEvent
)
{
    HRESULT             hResult;

    // Try to create the custom debug event callback interface
    hResult = createInterface(__uuidof(ILwstomDebugEventCallbacks), pvoidptr(pDbgEvent));

    return hResult;

} // createDebugEventInterface

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
