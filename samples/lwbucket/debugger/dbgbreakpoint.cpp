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
|*  Module: dbgbreakpoint.cpp                                                 *|
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
// Debugger Breakpoint Interface wrappers
//
//******************************************************************************

HRESULT
GetId
(
    PULONG              Id
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetId breakpoint method
        hResult = pDbgBreakpoint->GetId(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetId %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetId
            {
                dPrintf("%s GetId %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetId

//******************************************************************************

HRESULT
GetType
(
    PULONG              BreakType,
    PULONG              ProcType
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(BreakType != NULL);
    assert(ProcType != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetType breakpoint method
        hResult = pDbgBreakpoint->GetType(BreakType, ProcType);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetType %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetType
            {
                dPrintf("%s GetType %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetType

//******************************************************************************

HRESULT
GetAdder
(
    PDEBUG_CLIENT      *Adder
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Adder != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetAddr breakpoint method
        hResult = pDbgBreakpoint->GetAdder(Adder);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetAdder %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetAdder
            {
                dPrintf("%s GetAdder %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetAdder

//******************************************************************************

HRESULT
GetFlags
(
    PULONG              Flags
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Flags != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFLags breakpoint method
        hResult = pDbgBreakpoint->GetFlags(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFlags
            {
                dPrintf("%s GetFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFlags

//******************************************************************************

HRESULT
AddFlags
(
    ULONG               Flags
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddFlags breakpoint method
        hResult = pDbgBreakpoint->AddFlags(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddFlags
            {
                dPrintf("%s AddFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddFlags

//******************************************************************************

HRESULT
RemoveFlags
(
    ULONG               Flags
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveFlags breakpoint method
        hResult = pDbgBreakpoint->RemoveFlags(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveFlags
            {
                dPrintf("%s RemoveFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveFlags

//******************************************************************************

HRESULT
SetFlags
(
    ULONG               Flags
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetFlags breakpoint method
        hResult = pDbgBreakpoint->SetFlags(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetFlags
            {
                dPrintf("%s SetFlags %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetFlags

//******************************************************************************

HRESULT
GetOffset
(
    PULONG64            Offset
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffset breakpoint method
        hResult = pDbgBreakpoint->GetOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffset %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffset
            {
                dPrintf("%s GetOffset %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffset

//******************************************************************************

HRESULT
SetOffset
(
    ULONG64             Offset
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOffset breakpoint method
        hResult = pDbgBreakpoint->SetOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOffset %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOffset
            {
                dPrintf("%s SetOffset %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOffset

//******************************************************************************

HRESULT
GetDataParameters
(
    PULONG              Size,
    PULONG              AccessType
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Size != NULL);
    assert(AccessType != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDataParameters breakpoint method
        hResult = pDbgBreakpoint->GetDataParameters(Size, AccessType);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDataParameters %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDataParameters
            {
                dPrintf("%s GetDataParameters %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDataParameters

//******************************************************************************

HRESULT
SetDataParameters
(
    ULONG               Size,
    ULONG               AccessType
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetDataParameters breakpoint method
        hResult = pDbgBreakpoint->SetDataParameters(Size, AccessType);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetDataParameters %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetDataParameters
            {
                dPrintf("%s SetDataParameters %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetDataParameters

//******************************************************************************

HRESULT
GetPassCount
(
    PULONG              Count
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Count != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPassCount breakpoint method
        hResult = pDbgBreakpoint->GetPassCount(Count);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPassCount %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPassCount
            {
                dPrintf("%s GetPassCount %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPassCount

//******************************************************************************

HRESULT
SetPassCount
(
    ULONG               Count
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetPassCount breakpoint method
        hResult = pDbgBreakpoint->SetPassCount(Count);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetPassCount %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetPassCount
            {
                dPrintf("%s SetPassCount %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetPassCount

//******************************************************************************

HRESULT
GetLwrrentPassCount
(
    PULONG              Count
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Count != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentPassCount breakpoint method
        hResult = pDbgBreakpoint->GetLwrrentPassCount(Count);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentPassCount %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentPassCount
            {
                dPrintf("%s GetLwrrentPassCount %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentPassCount

//******************************************************************************

HRESULT
GetMatchThreadId
(
    PULONG              Thread
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Thread != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetMatchThreadId breakpoint method
        hResult = pDbgBreakpoint->GetMatchThreadId(Thread);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetMatchThreadId %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetMatchThreadId
            {
                dPrintf("%s GetMatchThreadId %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetMatchThreadId

//******************************************************************************

HRESULT
SetMatchThreadId
(
    ULONG               Thread
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetMatchThreadId breakpoint method
        hResult = pDbgBreakpoint->SetMatchThreadId(Thread);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetMatchThreadId %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetMatchThreadId
            {
                dPrintf("%s SetMatchThreadId %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetMatchThreadId

//******************************************************************************

HRESULT
GetCommand
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetCommand breakpoint method
        hResult = pDbgBreakpoint->GetCommand(Buffer, BufferSize, CommandSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetCommand %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetCommand
            {
                dPrintf("%s GetCommand %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetCommand

//******************************************************************************

HRESULT
SetCommand
(
    PSTR                Buffer
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetCommand breakpoint method
        hResult = pDbgBreakpoint->SetCommand(Buffer);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetCommand %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetCommand
            {
                dPrintf("%s SetCommand %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetCommand

//******************************************************************************

HRESULT
GetOffsetExpression
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              ExpressionSize
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetExpression breakpoint method
        hResult = pDbgBreakpoint->GetOffsetExpression(Buffer, BufferSize, ExpressionSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetExpression %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetExpression
            {
                dPrintf("%s GetOffsetExpression %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetExpression

//******************************************************************************

HRESULT
SetOffsetExpression
(
    PCSTR               Expression
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Expression != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOffsetExpression breakpoint method
        hResult = pDbgBreakpoint->SetOffsetExpression(Expression);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOffsetExpression %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOffsetExpression
            {
                dPrintf("%s SetOffsetExpression %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOffsetExpression

//******************************************************************************

HRESULT
GetParameters
(
    PDEBUG_BREAKPOINT_PARAMETERS Params
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Params != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetParameters breakpoint method
        hResult = pDbgBreakpoint->GetParameters(Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetParameters %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetParameters
            {
                dPrintf("%s GetParameters %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetParameters

//******************************************************************************

HRESULT
GetCommandWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetCommandWide breakpoint method
        hResult = pDbgBreakpoint->GetCommandWide(Buffer, BufferSize, CommandSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetCommandWide %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetCommandWide
            {
                dPrintf("%s GetLwrrentTimeDate %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetCommandWide

//******************************************************************************

HRESULT
SetCommandWide
(
    PCWSTR              Command
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Command != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetCommandWide breakpoint method
        hResult = pDbgBreakpoint->SetCommandWide(Command);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetCommandWide %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetCommandWide
            {
                dPrintf("%s SetCommandWide %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetCommandWide

//******************************************************************************

HRESULT
GetOffsetExpressionWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              ExpressionSize
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetExpressionWide breakpoint method
        hResult = pDbgBreakpoint->GetOffsetExpressionWide(Buffer, BufferSize, ExpressionSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetExpressionWide %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetExpressionWide
            {
                dPrintf("%s GetOffsetExpressionWide %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetExpressionWide

//******************************************************************************

HRESULT
SetOffsetExpressionWide
(
    PCWSTR              Expression
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Expression != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOffsetExpressionWide breakpoint method
        hResult = pDbgBreakpoint->SetOffsetExpressionWide(Expression);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOffsetExpressionWide %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOffsetExpressionWide
            {
                dPrintf("%s SetOffsetExpressionWide %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOffsetExpression

//******************************************************************************

HRESULT
GetGuid
(
    LPGUID              Guid
)
{
    PDEBUG_BREAKPOINT3  pDbgBreakpoint = debugBreakpointInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Guid != NULL);

    // Check for valid debug breakpoint interface
    if (pDbgBreakpoint != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetGuid breakpoint method
        hResult = pDbgBreakpoint->GetGuid(Guid);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng breakpoint interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_BREAKPOINT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetGuid %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetGuid
            {
                dPrintf("%s GetGuid %s = 0x%08x\n", DML(bold("DbgBreakpoint:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetGuid

//******************************************************************************

HRESULT
GetType
(
    ULONG               Id,
    PULONG              BreakType,
    PULONG              ProcType
)
{
    HRESULT             hResult;

    assert(BreakType != NULL);
    assert(ProcType != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetType breakpoint method
    hResult = GetType(BreakType, ProcType);

    return hResult;

} // GetType

//******************************************************************************

HRESULT
GetAdder
(
    ULONG               Id,
    PDEBUG_CLIENT*      Adder
)
{
    HRESULT             hResult;

    assert(Adder != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetAdder breakpoint method
    hResult = GetAdder(Adder);

    return hResult;

} // GetAdder

//******************************************************************************

HRESULT
GetFlags
(
    ULONG               Id,
    PULONG              Flags
)
{
    HRESULT             hResult;

    assert(Flags != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetFlags breakpoint method
    hResult = GetFlags(Flags);

    return hResult;

} // GetFlags

//******************************************************************************

HRESULT
AddFlags
(
    ULONG               Id,
    ULONG               Flags
)
{
    HRESULT             hResult;

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the AddFlags breakpoint method
    hResult = AddFlags(Flags);

    return hResult;

} // AddFlags

//******************************************************************************

HRESULT
RemoveFlags
(
    ULONG               Id,
    ULONG               Flags
)
{
    HRESULT             hResult;

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the RemoveFlags breakpoint method
    hResult = RemoveFlags(Flags);

    return hResult;

} // RemoveFlags

//******************************************************************************

HRESULT
SetFlags
(
    ULONG               Id,
    ULONG               Flags
)
{
    HRESULT             hResult;

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetFlags breakpoint method
    hResult = SetFlags(Flags);

    return hResult;

} // SetFlags

//******************************************************************************

HRESULT
GetOffset
(
    ULONG               Id,
    PULONG64            Offset
)
{
    HRESULT             hResult;

    assert(Offset != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetOffset breakpoint method
    hResult = GetOffset(Offset);

    return hResult;

} // GetOffset

//******************************************************************************

HRESULT
SetOffset
(
    ULONG               Id,
    ULONG64             Offset
)
{
    HRESULT             hResult;

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetOffset breakpoint method
    hResult = SetOffset(Offset);

    return hResult;

} // SetOffset

//******************************************************************************

HRESULT
GetDataParameters
(
    ULONG               Id,
    PULONG              Size,
    PULONG              AccessType
)
{
    HRESULT             hResult;

    assert(Size != NULL);
    assert(AccessType != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetDataParameters breakpoint method
    hResult = GetDataParameters(Size, AccessType);

    return hResult;

} // GetDataParameters

//******************************************************************************

HRESULT
SetDataParameters
(
    ULONG               Id,
    ULONG               Size,
    ULONG               AccessType
)
{
    HRESULT             hResult;

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetDataParameters breakpoint method
    hResult = SetDataParameters(Size, AccessType);

    return hResult;

} // SetDataParameters

//******************************************************************************

HRESULT
GetPassCount
(
    ULONG               Id,
    PULONG              Count
)
{
    HRESULT             hResult;

    assert(Count != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetPassCount breakpoint method
    hResult = GetPassCount(Count);

    return hResult;

} // GetPassCount

//******************************************************************************

HRESULT
SetPassCount
(
    ULONG               Id,
    ULONG               Count
)
{
    HRESULT             hResult;

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetPassCount breakpoint method
    hResult = SetPassCount(Count);

    return hResult;

} // SetPassCount

//******************************************************************************

HRESULT
GetLwrrentPassCount
(
    ULONG               Id,
    PULONG              Count
)
{
    HRESULT             hResult;

    assert(Count != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetLwrrentPassCount breakpoint method
    hResult = GetLwrrentPassCount(Count);

    return hResult;

} // GetLwrrentPassCount

//******************************************************************************

HRESULT
GetMatchThreadId
(
    ULONG               Id,
    PULONG              Thread
)
{
    HRESULT             hResult;

    assert(Thread != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetMatchThreadId breakpoint method
    hResult = GetMatchThreadId(Thread);

    return hResult;

} // GetMatchThreadId

//******************************************************************************

HRESULT
SetMatchThreadId
(
    ULONG               Id,
    ULONG               Thread
)
{
    HRESULT             hResult;

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetMatchThreadId breakpoint method
    hResult = SetMatchThreadId(Thread);

    return hResult;

} // SetMatchThreadId

//******************************************************************************

HRESULT
GetCommand
(
    ULONG               Id,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    HRESULT             hResult;

    assert(Buffer != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetCommand breakpoint method
    hResult = GetCommand(Buffer, BufferSize, CommandSize);

    return hResult;

} // GetCommand

//******************************************************************************

HRESULT
SetCommand
(
    ULONG               Id,
    PSTR                Buffer
)
{
    HRESULT             hResult;

    assert(Buffer != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetCommand breakpoint method
    hResult = SetCommand(Buffer);

    return hResult;

} // SetCommand

//******************************************************************************

HRESULT
GetOffsetExpression
(
    ULONG               Id,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              ExpressionSize
)
{
    HRESULT             hResult;

    assert(Buffer != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetOffsetExpression breakpoint method
    hResult = GetOffsetExpression(Buffer, BufferSize, ExpressionSize);

    return hResult;

} // GetOffsetExpression

//******************************************************************************

HRESULT
SetOffsetExpression
(
    ULONG               Id,
    PCSTR               Expression
)
{
    HRESULT             hResult;

    assert(Expression != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetOffsetExpression breakpoint method
    hResult = SetOffsetExpression(Expression);

    return hResult;

} // SetOffsetExpression

//******************************************************************************

HRESULT
GetParameters
(
    ULONG               Id,
    PDEBUG_BREAKPOINT_PARAMETERS Params
)
{
    HRESULT             hResult;

    assert(Params != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetParameters breakpoint method
    hResult = GetParameters(Params);

    return hResult;

} // GetParameters

//******************************************************************************

HRESULT
GetCommandWide
(
    ULONG               Id,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    HRESULT             hResult;

    assert(Buffer != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetCommandWide breakpoint method
    hResult = GetCommandWide(Buffer, BufferSize, CommandSize);

    return hResult;

} // GetCommandWide

//******************************************************************************

HRESULT
SetCommandWide
(
    ULONG               Id,
    PCWSTR              Command
)
{
    HRESULT             hResult;

    assert(Command != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetCommandWide breakpoint method
    hResult = SetCommandWide(Command);

    return hResult;

} // SetCommandWide

//******************************************************************************

HRESULT
GetOffsetExpressionWide
(
    ULONG               Id,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              ExpressionSize
)
{
    HRESULT             hResult;

    assert(Buffer != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetOffsetExpressionWide breakpoint method
    hResult = GetOffsetExpressionWide(Buffer, BufferSize, ExpressionSize);

    return hResult;

} // GetOffsetExpressionWide

//******************************************************************************

HRESULT
SetOffsetExpressionWide
(
    ULONG               Id,
    PCWSTR              Expression
)
{
    HRESULT             hResult;

    assert(Expression != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the SetOffsetExpressionWide breakpoint method
    hResult = SetOffsetExpressionWide(Expression);

    return hResult;

} // SetOffsetExpressionWide

//******************************************************************************

HRESULT
GetGuid
(
    ULONG               Id,
    LPGUID              Guid
)
{
    HRESULT             hResult;

    assert(Guid != NULL);

    // Try to set the requested breakpoint interface
    setBreakpointInterface(getBreakpointInterface(Id));

    // Call the GetGuid breakpoint method
    hResult = GetGuid(Guid);

    return hResult;

} // GetGuid

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
