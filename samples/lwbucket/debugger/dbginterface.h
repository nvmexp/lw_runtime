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
|*  Module: dbginterface.h                                                    *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGINTERFACE_H
#define _DBGINTERFACE_H

//******************************************************************************
//
// Globals
//
//******************************************************************************
// Global WDBG Extension API's
extern wdbgexts::WINDBG_EXTENSION_APIS  wdbgexts::ExtensionApis;// WBDG extension API's

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
//  Macros
//
//******************************************************************************
#define DBG_INTERFACES(Pointer)         (reinterpret_cast<PDEBUG_INTERFACES>(voidptr(Pointer)))

//******************************************************************************
//
//  Template safeRelease (Safe debugger interface release)
//
//******************************************************************************
template <typename T>
void
safeRelease(T*& pInterface)
{
    if ((pInterface != NULL) && isInitialized())
    {
        pInterface->Release();
    }
    pInterface = NULL;

} // safeRelease

//******************************************************************************
//
// Structures
//
//******************************************************************************
typedef struct _DEBUG_INTERFACES
{
    // DbgEng Interfaces
    PDEBUG_CLIENT           pDbgClient;                         // Debug client interface
    PDEBUG_CONTROL          pDbgControl;                        // Debug control interface
    PDEBUG_DATA_SPACES      pDbgDataSpaces;                     // Debug data spaces interface
    PDEBUG_REGISTERS        pDbgRegisters;                      // Debug registers interface
    PDEBUG_SYMBOLS          pDbgSymbols;                        // Debug symbols interface
    PDEBUG_SYSTEM_OBJECTS   pDbgSystemObjects;                  // Debug system objects interface
    PDEBUG_ADVANCED         pDbgAdvanced;                       // Debug advanced interface
    PDEBUG_SYMBOL_GROUP2    pDbgSymbolGroup;                    // Debug symbol group interface
    PDEBUG_BREAKPOINT3      pDbgBreakpoint;                     // Debug breakpoint interface

    // DbgEng Interface ID's
    IID                     dbgClientId;                        // Debug client inteface ID (UUID)
    IID                     dbgControlId;                       // Debug control inteface ID (UUID)
    IID                     dbgDataSpacesId;                    // Debug data spaces inteface ID (UUID)
    IID                     dbgRegistersId;                     // Debug registers inteface ID (UUID)
    IID                     dbgSymbolsId;                       // Debug symbols inteface ID (UUID)
    IID                     dbgSystemObjectsId;                 // Debug system objects inteface ID (UUID)
    IID                     dbgAdvancedId;                      // Debug advanced inteface ID (UUID)

    // Custom Interfaces
    PDEBUG_INPUT_CALLBACKS  pDbgInput;                          // Custom debug input interface
    PDEBUG_OUTPUT_CALLBACKS pDbgOutput;                         // Custom debug output interface
    PDEBUG_EVENT_CALLBACKS  pDbgEvent;                          // Custom debug event interface

}  DEBUG_INTERFACES, *PDEBUG_INTERFACES;

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  HRESULT                         createInterface(REFIID refInterfaceId, PVOID *ppInterface);
extern  HRESULT                         releaseInterface(REFIID refInterfaceId, PVOID *ppInterface);

extern  PDEBUG_BREAKPOINT3              getBreakpointInterface(ULONG Id);
extern  void                            setBreakpointInterface(PDEBUG_BREAKPOINT3 pDbgBreakpoint);

extern  void                            initializeInterfaces();
extern  void                            releaseInterfaces();

extern  void                            initializeDebugInterface();
extern  void                            uninitializeDebugInterface();

extern  void                            acquireDebugInterface();
extern  void                            releaseDebugInterface();

extern  bool                            isDebugClientInterface();
extern  bool                            isDebugClient2Interface();
extern  bool                            isDebugClient3Interface();
extern  bool                            isDebugClient4Interface();
extern  bool                            isDebugClient5Interface();

extern  PDEBUG_CLIENT                   debugClientInterface();
extern  PDEBUG_CLIENT2                  debugClient2Interface();
extern  PDEBUG_CLIENT3                  debugClient3Interface();
extern  PDEBUG_CLIENT4                  debugClient4Interface();
extern  PDEBUG_CLIENT5                  debugClient5Interface();

extern  bool                            isDebugControlInterface();
extern  bool                            isDebugControl2Interface();
extern  bool                            isDebugControl3Interface();
extern  bool                            isDebugControl4Interface();

extern  PDEBUG_CONTROL                  debugControlInterface();
extern  PDEBUG_CONTROL2                 debugControl2Interface();
extern  PDEBUG_CONTROL3                 debugControl3Interface();
extern  PDEBUG_CONTROL4                 debugControl4Interface();

extern  bool                            isDebugDataSpacesInterface();
extern  bool                            isDebugDataSpaces2Interface();
extern  bool                            isDebugDataSpaces3Interface();
extern  bool                            isDebugDataSpaces4Interface();

extern  PDEBUG_DATA_SPACES              debugDataSpacesInterface();
extern  PDEBUG_DATA_SPACES2             debugDataSpaces2Interface();
extern  PDEBUG_DATA_SPACES3             debugDataSpaces3Interface();
extern  PDEBUG_DATA_SPACES4             debugDataSpaces4Interface();

extern  bool                            isDebugRegistersInterface();
extern  bool                            isDebugRegisters2Interface();

extern  PDEBUG_REGISTERS                debugRegistersInterface();
extern  PDEBUG_REGISTERS2               debugRegisters2Interface();

extern  bool                            isDebugSymbolsInterface();
extern  bool                            isDebugSymbols2Interface();
extern  bool                            isDebugSymbols3Interface();

extern  PDEBUG_SYMBOLS                  debugSymbolsInterface();
extern  PDEBUG_SYMBOLS2                 debugSymbols2Interface();
extern  PDEBUG_SYMBOLS3                 debugSymbols3Interface();

extern  bool                            isDebugSystemObjectsInterface();
extern  bool                            isDebugSystemObjects2Interface();
extern  bool                            isDebugSystemObjects3Interface();
extern  bool                            isDebugSystemObjects4Interface();

extern  PDEBUG_SYSTEM_OBJECTS           debugSystemObjectsInterface();
extern  PDEBUG_SYSTEM_OBJECTS2          debugSystemObjects2Interface();
extern  PDEBUG_SYSTEM_OBJECTS3          debugSystemObjects3Interface();
extern  PDEBUG_SYSTEM_OBJECTS4          debugSystemObjects4Interface();

extern  bool                            isDebugAdvancedInterface();
extern  bool                            isDebugAdvanced2Interface();
extern  bool                            isDebugAdvanced3Interface();

extern  PDEBUG_ADVANCED                 debugAdvancedInterface();
extern  PDEBUG_ADVANCED2                debugAdvanced2Interface();
extern  PDEBUG_ADVANCED3                debugAdvanced3Interface();

extern  PDEBUG_SYMBOL_GROUP2            debugSymbolGroupInterface();

extern  PDEBUG_BREAKPOINT3              debugBreakpointInterface();

extern  PLWSTOM_DEBUG_INPUT_CALLBACKS   dbgInput();
extern  PLWSTOM_DEBUG_OUTPUT_CALLBACKS  dbgOutput();
extern  PLWSTOM_DEBUG_EVENT_CALLBACKS   dbgEvent();

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGINTERFACE_H
