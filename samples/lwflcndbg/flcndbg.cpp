/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2008 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004
// flcndbg.cpp
// 
//*****************************************************

//
// includes
//
#include "lwwatch.h"
#include "lwwatch2.h"
#include "os.h"

//
// Globals
//
WINDBG_EXTENSION_APIS   ExtensionApis;

// 
// lwwatch2.0
//
USHORT ExtMajorVersion = 3; 
USHORT ExtMinorVersion = 0;

//
// Used to determine OS version information.
//
PDEBUG_CLIENT                       g_ExtClient          = NULL;
PDEBUG_CONTROL                      g_ExtControl         = NULL;
PDEBUG_SYMBOLS2                     g_ExtSymbols         = NULL;
PDEBUG_DATA_SPACES4                 g_ExtMemory          = NULL;
PDEBUG_REGISTERS2                   g_ExtRegisters       = NULL;
PLWWATCH_DEBUG_OUTPUT_CALLBACK      g_ExtOutputCallbacks = NULL;


BOOL DllInit
(
    HANDLE hModule,
    DWORD  dwReason,
    DWORD  dwReserved
)
{
    switch (dwReason)
    {
        case DLL_THREAD_ATTACH:
            break;

        case DLL_THREAD_DETACH:
            break;

        case DLL_PROCESS_DETACH:
            break;

        case DLL_PROCESS_ATTACH:
            break;
    }

    return TRUE;
}

//----------------------------------------------------
// WinDbgExtensionDllInit2
// + WinDbgExtensionDllInit used to the first function
//   called by the debugger engine when the dll extension
//   was loaded. Not so in lwwatch2.0 
//----------------------------------------------------
VOID
WinDbgExtensionDllInit2
(
    ULONG MinorVersion
)
{
    //
    // print out start of welcoming message.
    //
    dprintf("flcndbg: **********************************************\n");
    dprintf("flcndbg: Initializing flcndbg...\n");
    dprintf("flcndbg: **********************************************\n");

    // 
    // If the build number is under 2600 (WinXP Gold)
    // assume we're using Win2K.
    //
    if(OS_IS_VISTA_OR_HIGHER(MinorVersion))
    {
        dprintf("flcndbg: Target machine is running Vista or higher OS.\n");
    }
    else if(OS_IS_WINXP_OR_HIGHER(MinorVersion))
    {
        dprintf("flcndbg: Target machine is running WinXP or higher OS.\n");
    }
    else if(OS_IS_WIN2K_OR_HIGHER(MinorVersion))
    {
        dprintf("flcndbg: Target machine is running Win2K.\n");
    }
    else
    {
        dprintf("flcndbg: Target machine is running WinNT.\n");
    }

    return;
}


//----------------------------------------------------
// ExtQuery
// + Queries for all debugger interfaces.
//
//----------------------------------------------------
HRESULT
ExtQuery
(
    void
)
{
    HRESULT Hr;

    //
    // Fill up g_ExtControl
    //    
    if ((Hr = g_ExtClient->QueryInterface(__uuidof(IDebugControl),  
                    (void **)&g_ExtControl)) != S_OK) 
    { 
        goto Fail; 
    }

    //
    // Setup DebugSymbols interface
    //
    if ((Hr = g_ExtClient->QueryInterface(__uuidof(IDebugSymbols2),
                    (void **)&g_ExtSymbols)) != S_OK)
    {
        goto Fail;
    }

    //
    // Setup DebugDataSpaces interface
    //
    if ((Hr = g_ExtClient->QueryInterface(__uuidof(IDebugDataSpaces4),
                    (void **)&g_ExtMemory)) != S_OK)
    {
        goto Fail;
    }

   //
    // Setup DebugRegisters interface
    //
    if ((Hr = g_ExtClient->QueryInterface(__uuidof(IDebugRegisters2),
                    (void **)&g_ExtRegisters)) != S_OK)
    {
        goto Fail;
    }

    //
    // Provide our Output callback to the engine so that it notifies us when
    // is produces our desired output
    //
    if ((Hr = g_ExtClient->SetOutputCallbacks(g_ExtOutputCallbacks))!=S_OK)
    { 
        goto Fail; 
    }
    
    //
    // But we don't need any output to start with. So Set Output Mask
    // appropriately.
    //    
    if ((Hr = g_ExtClient->SetOutputMask(DEBUG_OUTPUT_NONE))!= S_OK) 
    { 
        goto Fail; 
    } 
    
    return S_OK;

 Fail:
    ExtRelease();
    return Hr;
}


//----------------------------------------------------
// ExtRelease
// + Cleans up all debugger interfaces.
//
//----------------------------------------------------
void
ExtRelease
(
 void
)
{
    if (g_ExtControl)
    { 
        g_ExtControl->Release(); 
        g_ExtControl = NULL; 
    }

    if (g_ExtSymbols)
    { 
        g_ExtSymbols->Release(); 
        g_ExtSymbols = NULL; 
    }

    if (g_ExtMemory)
    { 
        g_ExtMemory->Release(); 
        g_ExtMemory = NULL; 
    }

    if (g_ExtRegisters)
    { 
        g_ExtRegisters->Release(); 
        g_ExtRegisters = NULL; 
    }
}

//----------------------------------------------------
// DebugExtensionUninitialize
// + This callback function is called by the engine after
//   loading a DbgEng extension DLL.
//----------------------------------------------------
extern "C" HRESULT CALLBACK
DebugExtensionInitialize
(
    OUT PULONG  Version,
    OUT PULONG  Flags
)
{
    PDEBUG_CLIENT DebugClient; 
    PDEBUG_CONTROL DebugControl; 
    ULONG m_ulExeCpuType;
    HRESULT Hr = S_OK;

    //
    // 1. WINDBG VERSION CAUTION : CoCreateInstance is another much more
    // complicated way of creating COM objects. Lwrrently, instantiating our
    // output callback class (OutputCb) using new operator works fine for our
    // requirements. It might break with future versions of windbg/windows which
    // may expect COM objects to be created using CoCreateInstance though this
    // is very unlikely. 
    //
    // 2. object automatically deleted in OutputCb::Release. No need to
    // explicitly delete
    //
    g_ExtOutputCallbacks = new OutputCb();
 
    //
    // Provide our extension dll version to the debugger engine
    //
    *Version = DEBUG_EXTENSION_VERSION(ExtMajorVersion, ExtMinorVersion);
    
    //
    // From 'Using the Debugger Engine and Extension API'
    // Flags - Set this to zero. (Reserved for future use.) 
    //   
    *Flags = 0;

    // 
    // Get a new DebugClient from engine
    //
    if ((Hr = DebugCreate(__uuidof(IDebugClient),
                          (void **)&DebugClient)) != S_OK)
    {
        return Hr;
    }

    //
    // Get a temporary DebugControl object to do the stuff inside the if block
    //
    if ((Hr = DebugClient->QueryInterface(__uuidof(IDebugControl),
                                  (void **)&DebugControl)) == S_OK)
    {
        //
        // Get the windbg-style extension APIS
        //
        ExtensionApis.nSize = sizeof (ExtensionApis);
        Hr = DebugControl->GetWindbgExtensionApis64(&ExtensionApis);

        //
        // Get Build number
        //
        ULONG PlatformId=0;
        ULONG ServicePackStringUsed = 0, ServicePackNumber = 0;
        ULONG BuildStringUsed = 0;
        
        //
        // You can find GetSystemVersion doc in dbgeng.h
        //
        DebugControl->GetSystemVersion(&PlatformId, &OSMajorVersion, 
                &OSMinorVersion, 0, 0, &ServicePackStringUsed, 
                &ServicePackNumber, 0, 0, &BuildStringUsed);
      
        // Determine Processor Pointer Size
        DebugControl->GetExelwtingProcessorType(&m_ulExeCpuType);

        //
        // Release temporary DebugControl object now that we done whatever we
        // wanted to do in DebugExtensionInitialize
        //
        DebugControl->Release();
        DebugControl = NULL;
    }

    //
    // Save DebugClient object in a global for future use
    //
    g_ExtClient = DebugClient;

    //
    // CAUTION: Call WinDbgExtensionDllInit2 at last since dprintf should not be
    // called before setting ExtensionApis and WinDbgExtensionDllInit2 calls
    // dprintf.
    //
    WinDbgExtensionDllInit2(OSMinorVersion);

    return Hr;
}

//----------------------------------------------------
// DebugExtensionUninitialize
// + Called when the debugger engine unloads the
//   extension
//----------------------------------------------------
extern "C" void CALLBACK
DebugExtensionUninitialize
(
 void
)
{
    if (g_ExtClient != NULL)
        g_ExtClient->Release();

    g_ExtClient = NULL;
    return; 
}
