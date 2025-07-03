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
|*  Module: event.cpp                                                         *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************
//
// Forwards
//
//******************************************************************************







//******************************************************************************
//
// Locals
//
//******************************************************************************
static  CExtensionEvent s_extensionEvent;           // Extension event handler







//******************************************************************************

HRESULT
CExtensionEvent::breakpoint
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
CExtensionEvent::exception
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
CExtensionEvent::createThread
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
CExtensionEvent::exitThread
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
CExtensionEvent::createProcess
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
CExtensionEvent::exitProcess
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
CExtensionEvent::loadModule
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
CExtensionEvent::unloadModule
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
CExtensionEvent::systemError
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
CExtensionEvent::sessionStatus
(
    ULONG               Status
) const
{
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Switch on the session status value
    switch(Status)
    {
        case DEBUG_SESSION_ACTIVE:                          /* A debuggee has been discovered for the session */

            // A debuggee has been discovered for the session
            break;

        case DEBUG_SESSION_END_SESSION_ACTIVE_TERMINATE:    /* Session has ended (Terminate) */

            // Session has ended (Terminate)
            break;

        case DEBUG_SESSION_END_SESSION_ACTIVE_DETACH:       /* Session has ended (Detach) */

            // Session has ended (Detach)
            break;

        case DEBUG_SESSION_END_SESSION_PASSIVE:             /* Session has ended (Passive) */

            // Session has ended (Passive)
            break;

        case DEBUG_SESSION_END:                             /* Debuggee has run to completion (User mode only) */

            // Debuggee has run to completion (User mode only)
            break;

        case DEBUG_SESSION_REBOOT:                          /* Target machine has rebooted */

            // Target machine has rebooted
            break;

        case DEBUG_SESSION_HIBERNATE:                       /* Target machine has hibernated */

            // Target machine has hibernated
            break;

        case DEBUG_SESSION_FAILURE:                         /* Engine was unable to continue the session */

            // Engine was unable to continue the session
            break;

        default:                                            /* Unknown session status */

            // Unknown session status
            break;
    }
    return hResult;

} // sessionStatus

//******************************************************************************

HRESULT
CExtensionEvent::changeDebuggeeState
(
    ULONG               Flags,
    ULONG64             Argument
) const
{
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Check on which debuggee state has changed (via the flags)
    if ((Flags & DEBUG_CDS_REGISTERS) == DEBUG_CDS_REGISTERS)
    {
        // Check for a specific vs. general registers change
        if (Argument == DEBUG_ANY_ID)
        {
            // Registers have changed
        }
        else    // Specific register change
        {
            // Register has changed (Argument is register index)
        }
    }
    else if ((Flags & DEBUG_CDS_DATA) == DEBUG_CDS_DATA)
    {
        // Check for a specific vs. general data space change
        if (Argument == DEBUG_ANY_ID)
        {
            // Data spaces have changed
        }
        else    // Specific data space change
        {
            // Data space has changed (Argument is data space index)
        }
    }
    else if ((Flags & DEBUG_CDS_REFRESH) == DEBUG_CDS_REFRESH)
    {
        // Switch on the requested GUI refresh request
        switch(Argument)
        {
            case DEBUG_CDS_REFRESH_EVALUATE:                    /* Evaluate GUI refresh request */

                // Evaluate GUI refresh
                break;

            case DEBUG_CDS_REFRESH_EXELWTE:                     /* Execute GUI refresh request */

                // Execute GUI refresh
                break;

            case DEBUG_CDS_REFRESH_EXELWTECOMMANDFILE:          /* Execute command file GUI refresh request */

                // Execute command file GUI refresh
                break;

            case DEBUG_CDS_REFRESH_ADDBREAKPOINT:               /* Add breakpoint GUI refresh request */

                // Add breakpoint GUI refresh
                break;

            case DEBUG_CDS_REFRESH_REMOVEBREAKPOINT:            /* Remove breakpoint GUI refresh request */

                // Remove breakpoint GUI refresh
                break;

            case DEBUG_CDS_REFRESH_WRITEVIRTUAL:                /* Write virtual GUI refresh request */

                // Write virtual GUI refresh
                break;

            case DEBUG_CDS_REFRESH_WRITEVIRTUALUNCACHED:        /* Write virtual uncached GUI refresh request */

                // Write virtual uncached GUI refresh
                break;

            case DEBUG_CDS_REFRESH_WRITEPHYSICAL:               /* Write physical GUI refresh request */

                // Write physical GUI refresh
                break;

            case DEBUG_CDS_REFRESH_WRITEPHYSICAL2:              /* Write physical2 GUI refresh request */

                // Write physical2 GUI refresh
                break;

            case DEBUG_CDS_REFRESH_SETVALUE:                    /* Set value GUI refresh request */

                // Set value GUI refresh
                break;

            case DEBUG_CDS_REFRESH_SETVALUE2:                   /* Set value2 GUI refresh request */

                // Set value2 GUI refresh
                break;

            case DEBUG_CDS_REFRESH_SETSCOPE:                    /* Set scope GUI refresh request */

                // Set scope GUI refresh
                break;

            case DEBUG_CDS_REFRESH_SETSCOPEFRAMEBYINDEX:        /* Set scope frame by index GUI refresh request */

                // Set scope frame by index GUI refresh
                break;

            case DEBUG_CDS_REFRESH_SETSCOPEFROMJITDEBUGINFO:    /* Set scope from JIT debug info GUI refresh request */

                // Set scope from JIT debug info GUI refresh
                break;

            case DEBUG_CDS_REFRESH_SETSCOPEFROMSTOREDEVENT:     /* Set scope from stored event GUI refresh request */

                // Set scope from stored event GUI refresh
                break;

            default:                                            /* Unknown GUI refresh request */

                // Unknown GUI refresh
                break;
        }
    }
    return hResult;

} // changeDebuggeeState

//******************************************************************************

HRESULT
CExtensionEvent::changeEngineState
(
    ULONG               Flags,
    ULONG64             Argument
) const
{
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Check on which engine state has changed (via the flags)
    if ((Flags & DEBUG_CES_LWRRENT_THREAD) == DEBUG_CES_LWRRENT_THREAD)
    {
        // Check for no current thread specified
        if (Argument == DEBUG_ANY_ID)
        {
            // Current thread changed (No current thread)
        }
        else    // New current thread ID
        {
            // Current thread changed (Argument is new thread ID)
        }
    }
    else if ((Flags & DEBUG_CES_EFFECTIVE_PROCESSOR) == DEBUG_CES_EFFECTIVE_PROCESSOR)
    {
        // Effective processor has changed (Argument is new effective processor)
    }
    else if ((Flags & DEBUG_CES_BREAKPOINTS) == DEBUG_CES_BREAKPOINTS)
    {
        // Check for specific vs. general breakpoint change
        if (Argument == DEBUG_ANY_ID)
        {
            // Breakpoints have changed
        }
        else    // General breakpoint change
        {
            // Breakpoint has changed (Argument is breakpoint ID)
        }
    }
    else if ((Flags & DEBUG_CES_CODE_LEVEL) == DEBUG_CES_CODE_LEVEL)
    {
        // Code interpretation level changed (Argument is new code interpretation level)
    }
    else if ((Flags & DEBUG_CES_EXELWTION_STATUS) == DEBUG_CES_EXELWTION_STATUS)
    {
        // Exelwtion status has changed (Argument is new exelwtion status)
    }
    else if ((Flags & DEBUG_CES_ENGINE_OPTIONS) == DEBUG_CES_ENGINE_OPTIONS)
    {
        // Engine options have change (Argument is new engine options)
    }
    else if ((Flags & DEBUG_CES_LOG_FILE) == DEBUG_CES_LOG_FILE)
    {
        // Check for log file opened or closed
        if (tobool(Argument))
        {
            // Log file has been opened
        }
        else    // Log file closed
        {
            // Log file has been closed
        }
    }
    else if ((Flags & DEBUG_CES_RADIX) == DEBUG_CES_RADIX)
    {
        // Debug radix value has changed (Argument is new radix)
    }
    else if ((Flags & DEBUG_CES_EVENT_FILTERS) == DEBUG_CES_EVENT_FILTERS)
    {
        // Check for specific vs. general filter change
        if (Argument == DEBUG_ANY_ID)
        {
            // Event filters changed
        }
        else    // General filter change
        {
            // Event filter change (Argument is event filter index)
        }
    }
    else if ((Flags & DEBUG_CES_PROCESS_OPTIONS) == DEBUG_CES_PROCESS_OPTIONS)
    {
        // Process options have changed (Argument is new process options)
    }
    else if ((Flags & DEBUG_CES_EXTENSIONS) == DEBUG_CES_EXTENSIONS)
    {
        // Extensions have been added or removed (Does argument mean anything?)
    }
    else if ((Flags & DEBUG_CES_SYSTEMS) == DEBUG_CES_SYSTEMS)
    {
        // Systems have been added or removed (Argument is new system ID)
    }
    else if ((Flags & DEBUG_CES_ASSEMBLY_OPTIONS) == DEBUG_CES_ASSEMBLY_OPTIONS)
    {
        // Assembly options have changed (Argument is new assembly options)
    }
    else if ((Flags & DEBUG_CES_EXPRESSION_SYNTAX) == DEBUG_CES_EXPRESSION_SYNTAX)
    {
        // Expression syntax has changed (Argument is new expression syntax)
    }
    else if ((Flags & DEBUG_CES_TEXT_REPLACEMENTS) == DEBUG_CES_TEXT_REPLACEMENTS)
    {
        // Text replacements have changed (Does argument mean anything?)
    }
    return hResult;

} // changeEngineState

//******************************************************************************

HRESULT
CExtensionEvent::changeSymbolState
(
    ULONG               Flags,
    ULONG64             Argument
) const
{
    HRESULT             hResult = DEBUG_STATUS_NO_CHANGE;

    // Check on which symbol state has changed (via the flags)
    if ((Flags & DEBUG_CSS_LOADS) == DEBUG_CSS_LOADS)
    {
        // Check for a specific module loaded
        if (Argument != 0)
        {
            // Specific module loaded
        }
        else    // Modules loaded
        {
            // Several modules loaded
        }
    }
    else if ((Flags & DEBUG_CSS_UNLOADS) == DEBUG_CSS_UNLOADS)
    {
        // Check for a specific module unloaded
        if (Argument != 0)
        {
            // Specific module unloaded
        }
        else    // Modules unloaded
        {
            // Several modules unloaded
        }
    }
    else if ((Flags & DEBUG_CSS_SCOPE) == DEBUG_CSS_SCOPE)
    {
        // Current symbol scope changed (Does argument mean anything?)
    }
    else if ((Flags & DEBUG_CSS_PATHS) == DEBUG_CSS_PATHS)
    {
        // Symbol path changed
    }
    else if ((Flags & DEBUG_CSS_SYMBOL_OPTIONS) == DEBUG_CSS_SYMBOL_OPTIONS)
    {
        // Symbol options have changed (Argument is new symbol options)
    }
    else if ((Flags & DEBUG_CSS_TYPE_OPTIONS) == DEBUG_CSS_TYPE_OPTIONS)
    {
        // Type options have change (Argument is new type options)
    }
    return hResult;

} // changeSymbolState

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
