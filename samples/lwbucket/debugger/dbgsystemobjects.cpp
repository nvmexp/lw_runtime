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
|*  Module: dbgsystemobjects.cpp                                              *|
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
// Debugger System Objects Interface wrappers
//
//******************************************************************************

HRESULT
GetEventThread
(
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventThread system object method
        hResult = pDbgSystemObjects->GetEventThread(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventThread %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventThread
            {
                dPrintf("%s GetEventThread %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventThread

//******************************************************************************

HRESULT
GetEventProcess
(
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventProcess system object method
        hResult = pDbgSystemObjects->GetEventProcess(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventProcess %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventProcess
            {
                dPrintf("%s GetEventProcess %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventProcess

//******************************************************************************

HRESULT
GetLwrrentThreadId
(
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentThreadId system object method
        hResult = pDbgSystemObjects->GetLwrrentThreadId(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentThreadId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentThreadId
            {
                dPrintf("%s GetLwrrentThreadId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentThreadId

//******************************************************************************

HRESULT
SetLwrrentThreadId
(
    ULONG               Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetLwrrentThreadId system object method
        hResult = pDbgSystemObjects->SetLwrrentThreadId(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetLwrrentThreadId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetLwrrentThreadId
            {
                dPrintf("%s SetLwrrentThreadId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetLwrrentThreadId

//******************************************************************************

HRESULT
GetLwrrentProcessId
(
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessId system object method
        hResult = pDbgSystemObjects->GetLwrrentProcessId(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessId
            {
                dPrintf("%s GetLwrrentProcessId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessId

//******************************************************************************

HRESULT
SetLwrrentProcessId
(
    ULONG               Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetLwrrentProcessId system object method
        hResult = pDbgSystemObjects->SetLwrrentProcessId(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetLwrrentProcessId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetLwrrentProcessId
            {
                dPrintf("%s SetLwrrentProcessId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetLwrrentProcessId

//******************************************************************************

HRESULT
GetNumberThreads
(
    PULONG              Number
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberThreads system object method
        hResult = pDbgSystemObjects->GetNumberThreads(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberThreads %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberThreads
            {
                dPrintf("%s GetNumberThreads %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberThreads

//******************************************************************************

HRESULT
GetTotalNumberThreads
(
    PULONG              Total,
    PULONG              LargestProcess
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Total != NULL);
    assert(LargestProcess != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTotalNumberThreads system object method
        hResult = pDbgSystemObjects->GetTotalNumberThreads(Total, LargestProcess);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTotalNumberThreads %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTotalNumberThreads
            {
                dPrintf("%s GetTotalNumberThreads %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTotalNumberThreads

//******************************************************************************

HRESULT
GetThreadIdsByIndex
(
    ULONG               Start,
    ULONG               Count,
    PULONG              Ids,
    PULONG              SysIds
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetThreadIdsByIndex system object method
        hResult = pDbgSystemObjects->GetThreadIdsByIndex(Start, Count, Ids, SysIds);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetThreadIdsByIndex %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetThreadIdsByIndex
            {
                dPrintf("%s GetThreadIdsByIndex %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetThreadIdsByIndex

//******************************************************************************

HRESULT
GetThreadIdByProcessor
(
    ULONG               Processor,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetThreadIdByProcessor system object method
        hResult = pDbgSystemObjects->GetThreadIdByProcessor(Processor, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetThreadIdByProcessor %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetThreadIdByProcessor
            {
                dPrintf("%s GetThreadIdByProcessor %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetThreadIdByProcessor

//******************************************************************************

HRESULT
GetLwrrentThreadDataOffset
(
    PULONG64            Offset
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentThreadDataOffset system object method
        hResult = pDbgSystemObjects->GetLwrrentThreadDataOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentThreadDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentThreadDataOffset
            {
                dPrintf("%s GetLwrrentThreadDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentThreadDataOffset

//******************************************************************************

HRESULT
GetThreadIdByDataOffset
(
    ULONG64             Offset,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetThreadIdByDataOffset system object method
        hResult = pDbgSystemObjects->GetThreadIdByDataOffset(Offset, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetThreadIdByDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetThreadIdByDataOffset
            {
                dPrintf("%s GetThreadIdByDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetThreadIdByDataOffset

//******************************************************************************

HRESULT
GetLwrrentThreadTeb
(
    PULONG64            Offset
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentThreadTeb system object method
        hResult = pDbgSystemObjects->GetLwrrentThreadTeb(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentThreadTeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentThreadTeb
            {
                dPrintf("%s GetLwrrentThreadTeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentThreadTeb

//******************************************************************************

HRESULT
GetThreadIdByTeb
(
    ULONG64             Offset,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetThreadIdByTeb system object method
        hResult = pDbgSystemObjects->GetThreadIdByTeb(Offset, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetThreadIdByTeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetThreadIdByTeb
            {
                dPrintf("%s GetThreadIdByTeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetThreadIdByTeb

//******************************************************************************

HRESULT
GetLwrrentThreadSystemId
(
    PULONG              SysId
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(SysId != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentThreadSystemId system object method
        hResult = pDbgSystemObjects->GetLwrrentThreadSystemId(SysId);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentThreadSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentThreadSystemId
            {
                dPrintf("%s GetLwrrentThreadSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentThreadSystemId

//******************************************************************************

HRESULT
GetThreadIdBySystemId
(
    ULONG               SysId,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetThreadIdBySystemId system object method
        hResult = pDbgSystemObjects->GetThreadIdBySystemId(SysId, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetThreadIdBySystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetThreadIdBySystemId
            {
                dPrintf("%s GetThreadIdBySystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetThreadIdBySystemId

//******************************************************************************

HRESULT
GetLwrrentThreadHandle
(
    PULONG64            Handle
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Handle != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentThreadHandle system object method
        hResult = pDbgSystemObjects->GetLwrrentThreadHandle(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentThreadHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentThreadHandle
            {
                dPrintf("%s GetLwrrentThreadHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentThreadHandle

//******************************************************************************

HRESULT
GetThreadIdByHandle
(
    ULONG64             Handle,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetThreadIdByHandle system object method
        hResult = pDbgSystemObjects->GetThreadIdByHandle(Handle, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetThreadIdByHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetThreadIdByHandle
            {
                dPrintf("%s GetThreadIdByHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetThreadIdByHandle

//******************************************************************************

HRESULT
GetNumberProcesses
(
    PULONG              Number
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberProcesses system object method
        hResult = pDbgSystemObjects->GetNumberProcesses(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberProcesses %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberProcesses
            {
                dPrintf("%s GetNumberProcesses %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberProcesses

//******************************************************************************

HRESULT
GetProcessIdsByIndex
(
    ULONG               Start,
    ULONG               Count,
    PULONG              Ids,
    PULONG              SysIds
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessIdsByIndex system object method
        hResult = pDbgSystemObjects->GetProcessIdsByIndex(Start, Count, Ids, SysIds);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessIdsByIndex %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessIdsByIndex
            {
                dPrintf("%s GetProcessIdsByIndex %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessIdsByIndex

//******************************************************************************

HRESULT
GetLwrrentProcessDataOffset
(
    PULONG64            Offset
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessDataOffset system object method
        hResult = pDbgSystemObjects->GetLwrrentProcessDataOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessDataOffset
            {
                dPrintf("%s GetLwrrentProcessDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessDataOffset

//******************************************************************************

HRESULT
GetProcessIdByDataOffset
(
    ULONG64             Offset,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessIdByDataOffset system object method
        hResult = pDbgSystemObjects->GetProcessIdByDataOffset(Offset, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessIdByDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessIdByDataOffset
            {
                dPrintf("%s GetProcessIdByDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessIdByDataOffset

//******************************************************************************

HRESULT
GetLwrrentProcessPeb
(
    PULONG64            Offset
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessPeb system object method
        hResult = pDbgSystemObjects->GetLwrrentProcessPeb(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessPeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessPeb
            {
                dPrintf("%s GetLwrrentProcessPeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessPeb

//******************************************************************************

HRESULT
GetProcessIdByPeb
(
    ULONG64             Offset,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessIdByPeb system object method
        hResult = pDbgSystemObjects->GetProcessIdByPeb(Offset, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessIdByPeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessIdByPeb
            {
                dPrintf("%s GetProcessIdByPeb %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessIdByPeb

//******************************************************************************

HRESULT
GetLwrrentProcessSystemId
(
    PULONG              SysId
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(SysId != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessSystemId system object method
        hResult = pDbgSystemObjects->GetLwrrentProcessSystemId(SysId);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessSystemId
            {
                dPrintf("%s GetLwrrentProcessSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessSystemId

//******************************************************************************

HRESULT
GetProcessIdBySystemId
(
    ULONG               SysId,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessIdBySystemId system object method
        hResult = pDbgSystemObjects->GetProcessIdBySystemId(SysId, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessIdBySystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessIdBySystemId
            {
                dPrintf("%s GetProcessIdBySystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessIdBySystemId

//******************************************************************************

HRESULT
GetLwrrentProcessHandle
(
    PULONG64            Handle
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Handle != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessHandle system object method
        hResult = pDbgSystemObjects->GetLwrrentProcessHandle(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessHandle
            {
                dPrintf("%s GetLwrrentProcessHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessHandle

//******************************************************************************

HRESULT
GetProcessIdByHandle
(
    ULONG64             Handle,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessIdByHandle system object method
        hResult = pDbgSystemObjects->GetProcessIdByHandle(Handle, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessIdByHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessIdByHandle
            {
                dPrintf("%s GetProcessIdByHandle %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessIdByHandle

//******************************************************************************

HRESULT
GetLwrrentProcessExelwtableName
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              ExeSize
)
{
    PDEBUG_SYSTEM_OBJECTS pDbgSystemObjects = debugSystemObjectsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessExelwtableName system object method
        hResult = pDbgSystemObjects->GetLwrrentProcessExelwtableName(Buffer, BufferSize, ExeSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessExelwtableName %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessExelwtableName
            {
                dPrintf("%s GetLwrrentProcessExelwtableName %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessExelwtableName

//******************************************************************************

HRESULT
GetLwrrentProcessUpTime
(
    PULONG              UpTime
)
{
    PDEBUG_SYSTEM_OBJECTS2 pDbgSystemObjects2 = debugSystemObjects2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(UpTime != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessUpTime system object method
        hResult = pDbgSystemObjects2->GetLwrrentProcessUpTime(UpTime);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessUpTime %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessUpTime
            {
                dPrintf("%s GetLwrrentProcessUpTime %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessUpTime

//******************************************************************************

HRESULT
GetImplicitThreadDataOffset
(
    PULONG64            Offset
)
{
    PDEBUG_SYSTEM_OBJECTS2 pDbgSystemObjects2 = debugSystemObjects2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetImplicitThreadDataOffset system object method
        hResult = pDbgSystemObjects2->GetImplicitThreadDataOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetImplicitThreadDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetImplicitThreadDataOffset
            {
                dPrintf("%s GetImplicitThreadDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetImplicitThreadDataOffset

//******************************************************************************

HRESULT
SetImplicitThreadDataOffset
(
    ULONG64             Offset
)
{
    PDEBUG_SYSTEM_OBJECTS2 pDbgSystemObjects2 = debugSystemObjects2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetImplicitThreadDataOffset system object method
        hResult = pDbgSystemObjects2->SetImplicitThreadDataOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetImplicitThreadDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetImplicitThreadDataOffset
            {
                dPrintf("%s SetImplicitThreadDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetImplicitThreadDataOffset

//******************************************************************************

HRESULT
GetImplicitProcessDataOffset
(
    PULONG64            Offset
)
{
    PDEBUG_SYSTEM_OBJECTS2 pDbgSystemObjects2 = debugSystemObjects2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetImplicitProcessDataOffset system object method
        hResult = pDbgSystemObjects2->GetImplicitProcessDataOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetImplicitProcessDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetImplicitProcessDataOffset
            {
                dPrintf("%s GetImplicitProcessDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetImplicitProcessDataOffset

//******************************************************************************

HRESULT
SetImplicitProcessDataOffset
(
    ULONG64             Offset
)
{
    PDEBUG_SYSTEM_OBJECTS2 pDbgSystemObjects2 = debugSystemObjects2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetImplicitProcessDataOffset system object method
        hResult = pDbgSystemObjects2->SetImplicitProcessDataOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetImplicitProcessDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetImplicitProcessDataOffset
            {
                dPrintf("%s SetImplicitProcessDataOffset %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetImplicitProcessDataOffset

//******************************************************************************

HRESULT
GetEventSystem
(
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventSystem system object method
        hResult = pDbgSystemObjects3->GetEventSystem(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventSystem %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventSystem
            {
                dPrintf("%s GetEventSystem %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventSystem

//******************************************************************************

HRESULT
GetLwrrentSystemId
(
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentSystemId system object method
        hResult = pDbgSystemObjects3->GetLwrrentSystemId(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentSystemId
            {
                dPrintf("%s GetLwrrentSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentSystemId

//******************************************************************************

HRESULT
SetLwrrentSystemId
(
    ULONG               Id
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetLwrrentSystemId system object method
        hResult = pDbgSystemObjects3->SetLwrrentSystemId(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetLwrrentSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetLwrrentSystemId
            {
                dPrintf("%s SetLwrrentSystemId %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetLwrrentSystemId

//******************************************************************************

HRESULT
GetNumberSystems
(
    PULONG              Number
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberSystems system object method
        hResult = pDbgSystemObjects3->GetNumberSystems(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberSystems %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberSystems
            {
                dPrintf("%s GetNumberSystems %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberSystems

//******************************************************************************

HRESULT
GetSystemIdsByIndex
(
    ULONG               Start,
    ULONG               Count,
    PULONG              Ids
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Ids != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemIdsByIndex system object method
        hResult = pDbgSystemObjects3->GetSystemIdsByIndex(Start, Count, Ids);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemIdsByIndex %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemIdsByIndex
            {
                dPrintf("%s GetSystemIdsByIndex %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemIdsByIndex

//******************************************************************************

HRESULT
GetTotalNumberThreadsAndProcesses
(
    PULONG              TotalThreads,
    PULONG              TotalProcesses,
    PULONG              LargestProcessThreads,
    PULONG              LargestSystemThreads,
    PULONG              LargestSystemProcesses
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(TotalThreads != NULL);
    assert(TotalProcesses != NULL);
    assert(LargestProcessThreads != NULL);
    assert(LargestSystemThreads != NULL);
    assert(LargestSystemProcesses != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTotalNumberThreadsAndProcesses system object method
        hResult = pDbgSystemObjects3->GetTotalNumberThreadsAndProcesses(TotalThreads, TotalProcesses, LargestProcessThreads, LargestSystemThreads, LargestSystemProcesses);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTotalNumberThreadsAndProcesses %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTotalNumberThreadsAndProcesses
            {
                dPrintf("%s GetTotalNumberThreadsAndProcesses %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTotalNumberThreadsAndProcesses

//******************************************************************************

HRESULT
GetLwrrentSystemServer
(
    PULONG64            Server
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Server != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentSystemServer system object method
        hResult = pDbgSystemObjects3->GetLwrrentSystemServer(Server);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentSystemServer %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentSystemServer
            {
                dPrintf("%s GetLwrrentSystemServer %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentSystemServer

//******************************************************************************

HRESULT
GetSystemByServer
(
    ULONG64             Server,
    PULONG              Id
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemByServer system object method
        hResult = pDbgSystemObjects3->GetSystemByServer(Server, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemByServer %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemByServer
            {
                dPrintf("%s GetSystemByServer %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemByServer

//******************************************************************************

HRESULT
GetLwrrentSystemServerName
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYSTEM_OBJECTS3 pDbgSystemObjects3 = debugSystemObjects3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentSystemServerName system object method
        hResult = pDbgSystemObjects3->GetLwrrentSystemServerName(Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentSystemServerName %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentSystemServerName
            {
                dPrintf("%s GetLwrrentSystemServerName %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentSystemServerName

//******************************************************************************

HRESULT
GetLwrrentProcessExelwtableNameWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              ExeSize
)
{
    PDEBUG_SYSTEM_OBJECTS4 pDbgSystemObjects4 = debugSystemObjects4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentProcessExelwtableNameWide system object method
        hResult = pDbgSystemObjects4->GetLwrrentProcessExelwtableNameWide(Buffer, BufferSize, ExeSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentProcessExelwtableNameWide %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentProcessExelwtableNameWide
            {
                dPrintf("%s GetLwrrentProcessExelwtableNameWide %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentProcessExelwtableNameWide

//******************************************************************************

HRESULT
GetLwrrentSystemServerNameWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYSTEM_OBJECTS4 pDbgSystemObjects4 = debugSystemObjects4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid system objects interface
    if (pDbgSystemObjects4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentSystemServerNameWide system object method
        hResult = pDbgSystemObjects4->GetLwrrentSystemServerNameWide(Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng system objects interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYSTEM_OBJECTS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentSystemServerNameWide %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentSystemServerNameWide
            {
                dPrintf("%s GetLwrrentSystemServerNameWide %s = 0x%08x\n", DML(bold("DbgSystemObjects:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentSystemServerNameWide

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
