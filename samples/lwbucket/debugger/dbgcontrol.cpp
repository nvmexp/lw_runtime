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
|*  Module: dbgcontrol.cpp                                                    *|
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
// Debugger Control Interface wrappers
//
//******************************************************************************

HRESULT
GetInterrupt()
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetInterrupt control method
        hResult = pDbgControl->GetInterrupt();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetInterrupt %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetInterrupt
            {
                dPrintf("%s GetInterrupt %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetInterrupt

//******************************************************************************

HRESULT
SetInterrupt
(
    ULONG               Flags
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetInterrupt control method
        hResult = pDbgControl->SetInterrupt(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetInterrupt %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetInterrupt
            {
                dPrintf("%s SetInterrupt %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetInterrupt

//******************************************************************************

HRESULT
GetInterruptTimeout
(
    PULONG              Seconds
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Seconds != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetInterruptTimeout control method
        hResult = pDbgControl->GetInterruptTimeout(Seconds);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetInterruptTimeout %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetInterruptTimeout
            {
                dPrintf("%s GetInterruptTimeout %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetInterruptTimeout

//******************************************************************************

HRESULT
SetInterruptTimeout
(
    ULONG               Seconds
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetInterruptTimeout control method
        hResult = pDbgControl->SetInterruptTimeout(Seconds);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetInterruptTimeout %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetInterruptTimeout
            {
                dPrintf("%s SetInterruptTimeout %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetInterruptTimeout

//******************************************************************************

HRESULT
GetLogFile
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              FileSize,
    PBOOL               Append
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Append != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLogFile control method
        hResult = pDbgControl->GetLogFile(Buffer, BufferSize, FileSize, Append);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLogFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLogFile
            {
                dPrintf("%s GetLogFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLogFile

//******************************************************************************

HRESULT
OpenLogFile
(
    PCSTR               File,
    BOOL                Append
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OpenLogFile control method
        hResult = pDbgControl->OpenLogFile(File, Append);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OpenLogFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OpenLogFile
            {
                dPrintf("%s OpenLogFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OpenLogFile

//******************************************************************************

HRESULT
CloseLogFile()
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CloseLogFile control method
        hResult = pDbgControl->CloseLogFile();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CloseLogFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CloseLogFile
            {
                dPrintf("%s CloseLogFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CloseLogFile

//******************************************************************************

HRESULT
GetLogMask
(
    PULONG              Mask
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Mask != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLogMask control method
        hResult = pDbgControl->GetLogMask(Mask);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLogMask %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLogMask
            {
                dPrintf("%s GetLogMask %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLogMask

//******************************************************************************

HRESULT
SetLogMask
(
    ULONG               Mask
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetLogMask control method
        hResult = pDbgControl->SetLogMask(Mask);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetLogMask %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetLogMask
            {
                dPrintf("%s SetLogMask %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetLogMask

//******************************************************************************

HRESULT
Input
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              InputSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the Input control method
        hResult = pDbgControl->Input(Buffer, BufferSize, InputSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng input)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_INPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Input %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Input
            {
                dPrintf("%s Input %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Input

//******************************************************************************

HRESULT
ReturnInput
(
    PCSTR               Buffer
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReturnInput control method
        hResult = pDbgControl->ReturnInput(Buffer);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng input)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_INPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReturnInput %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReturnInput
            {
                dPrintf("%s ReturnInput %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReturnInput

//******************************************************************************

HRESULT
Output
(
    ULONG               Mask,
    PCSTR               Format,
    ...
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputVaList control method
        hResult = pDbgControl->OutputVaList(Mask, Format, reinterpret_cast<va_list>(&Format + 1));

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Output %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Output
            {
                dPrintf("%s Output %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Output

//******************************************************************************

HRESULT
OutputVaList
(
    ULONG               Mask,
    PCSTR               Format,
    va_list             Args
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputVaList control method
        hResult = pDbgControl->OutputVaList(Mask, Format, Args);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputVaList %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputVaList
            {
                dPrintf("%s OutputVaList %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputVaList

//******************************************************************************

HRESULT
ControlledOutput
(
    ULONG               OutputControl,
    ULONG               Mask,
    PCSTR               Format,
    ...
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ControlledOutputVaList control method
        hResult = pDbgControl->ControlledOutputVaList(OutputControl, Mask, Format, reinterpret_cast<va_list>(&Format + 1));

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ControlledOutput %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ControlledOutput
            {
                dPrintf("%s ControlledOutput %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ControlledOutput

//******************************************************************************

HRESULT
ControlledOutputVaList
(
    ULONG               OutputControl,
    ULONG               Mask,
    PCSTR               Format,
    va_list             Args
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ControlledOutputVaList control method
        hResult = pDbgControl->ControlledOutputVaList(OutputControl, Mask, Format, Args);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ControlledOutputVaList %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ControlledOutputVaList
            {
                dPrintf("%s ControlledOutputVaList %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ControlledOutputVaList

//******************************************************************************

HRESULT
OutputPrompt
(
    ULONG               OutputControl,
    PCSTR               Format,
    ...
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputPromptVaList control method
        hResult = pDbgControl->OutputPromptVaList(OutputControl, Format, reinterpret_cast<va_list>(&Format + 1));

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputPrompt %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputPrompt
            {
                dPrintf("%s OutputPrompt %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputPrompt

//******************************************************************************

HRESULT
OutputPromptVaList
(
    ULONG               OutputControl,
    PCSTR               Format,
    va_list             Args
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputPromptVaList control method
        hResult = pDbgControl->OutputPromptVaList(OutputControl, Format, Args);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputPromptVaList %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputPromptVaList
            {
                dPrintf("%s OutputPromptVaList %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputPromptVaList

//******************************************************************************

HRESULT
GetPromptText
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              TextSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPromptText control method
        hResult = pDbgControl->GetPromptText(Buffer, BufferSize, TextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPromptText %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPromptText
            {
                dPrintf("%s GetPromptText %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPromptText

//******************************************************************************

HRESULT
OutputLwrrentState
(
    ULONG               OutputControl,
    ULONG               Flags
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputLwrrentState control method
        hResult = pDbgControl->OutputLwrrentState(OutputControl, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputLwrrentState %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputLwrrentState
            {
                dPrintf("%s OutputLwrrentState %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputLwrrentState

//******************************************************************************

HRESULT
OutputVersionInformation
(
    ULONG               OutputControl
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputVersionInformation control method
        hResult = pDbgControl->OutputVersionInformation(OutputControl);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputVersionInformation %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputVersionInformation
            {
                dPrintf("%s OutputVersionInformation %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputVersionInformation

//******************************************************************************

HRESULT
GetNotifyEventHandle
(
    PULONG64            Handle
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Handle != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNotifyEventHandle control method
        hResult = pDbgControl->GetNotifyEventHandle(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNotifyEventHandle %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNotifyEventHandle
            {
                dPrintf("%s GetNotifyEventHandle %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNotifyEventHandle

//******************************************************************************

HRESULT
SetNotifyEventHandle
(
    ULONG64             Handle
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetNotifyEventHandle control method
        hResult = pDbgControl->SetNotifyEventHandle(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetNotifyEventHandle %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetNotifyEventHandle
            {
                dPrintf("%s SetNotifyEventHandle %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetNotifyEventHandle

//******************************************************************************

HRESULT
Assemble
(
    ULONG64             Offset,
    PCSTR               Instr,
    PULONG64            EndOffset
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Instr != NULL);
    assert(EndOffset != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the Assemble control method
        hResult = pDbgControl->Assemble(Offset, Instr, EndOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Assemble %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Assemble
            {
                dPrintf("%s Assemble %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Assemble

//******************************************************************************

HRESULT
Disassemble
(
    ULONG64             Offset,
    ULONG               Flags,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              DisassemblySize,
    PULONG64            EndOffset
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the Disassemble control method
        hResult = pDbgControl->Disassemble(Offset, Flags, Buffer, BufferSize, DisassemblySize, EndOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Disassemble %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Disassemble
            {
                dPrintf("%s Disassemble %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Disassemble

//******************************************************************************

HRESULT
GetDisassembleEffectiveOffset
(
    PULONG64            Offset
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDisassembleEffectiveOffset control method
        hResult = pDbgControl->GetDisassembleEffectiveOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDisassembleEffectiveOffset %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDisassembleEffectiveOffset
            {
                dPrintf("%s GetDisassembleEffectiveOffset %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDisassembleEffectiveOffset

//******************************************************************************

HRESULT
OutputDisassembly
(
    ULONG               OutputControl,
    ULONG64             Offset,
    ULONG               Flags,
    PULONG64            EndOffset
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(EndOffset != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputDisassembly control method
        hResult = pDbgControl->OutputDisassembly(OutputControl, Offset, Flags, EndOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputDisassembly %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputDisassembly
            {
                dPrintf("%s OutputDisassembly %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputDisassembly

//******************************************************************************

HRESULT
OutputDisassemblyLines
(
    ULONG               OutputControl,
    ULONG               PreviousLines,
    ULONG               TotalLines,
    ULONG64             Offset,
    ULONG               Flags,
    PULONG              OffsetLine,
    PULONG64            StartOffset,
    PULONG64            EndOffset,
    PULONG64            LineOffsets
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputDisassemblyLines control method
        hResult = pDbgControl->OutputDisassemblyLines(OutputControl, PreviousLines, TotalLines, Offset, Flags, OffsetLine, StartOffset, EndOffset, LineOffsets);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputDisassemblyLines %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputDisassemblyLines
            {
                dPrintf("%s OutputDisassemblyLines %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputDisassemblyLines

//******************************************************************************

HRESULT
GetNearInstruction
(
    ULONG64             Offset,
    LONG                Delta,
    PULONG64            NearOffset
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(NearOffset != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNearInstruction control method
        hResult = pDbgControl->GetNearInstruction(Offset, Delta, NearOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNearInstruction %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNearInstruction
            {
                dPrintf("%s GetNearInstruction %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNearInstruction

//******************************************************************************

HRESULT
GetStackTrace
(
    ULONG64             FrameOffset,
    ULONG64             StackOffset,
    ULONG64             InstructionOffset,
    PDEBUG_STACK_FRAME  Frames,
    ULONG               FramesSize,
    PULONG              FramesFilled
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetStackTrace control method
        hResult = pDbgControl->GetStackTrace(FrameOffset, StackOffset, InstructionOffset, Frames, FramesSize, FramesFilled);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetStackTrace
            {
                dPrintf("%s GetStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetStackTrace

//******************************************************************************

HRESULT
GetReturnOffset
(
    PULONG64            Offset
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetReturnOffset control method
        hResult = pDbgControl->GetReturnOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetReturnOffset %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetReturnOffset
            {
                dPrintf("%s GetReturnOffset %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetReturnOffset

//******************************************************************************

HRESULT
OutputStackTrace
(
    ULONG               OutputControl,
    PDEBUG_STACK_FRAME  Frames,
    ULONG               FramesSize,
    ULONG               Flags
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputStackTrace control method
        hResult = pDbgControl->OutputStackTrace(OutputControl, Frames, FramesSize, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputStackTrace
            {
                dPrintf("%s OutputStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputStackTrace

//******************************************************************************

HRESULT
GetDebuggeeType
(
    PULONG              Class,
    PULONG              Qualifier
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Class != NULL);
    assert(Qualifier != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDebuggeeType control method
        hResult = pDbgControl->GetDebuggeeType(Class, Qualifier);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDebuggeeType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDebuggeeType
            {
                dPrintf("%s GetDebuggeeType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDebuggeeType

//******************************************************************************

HRESULT
GetActualProcessorType
(
    PULONG              Type
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetActualProcessorType control method
        hResult = pDbgControl->GetActualProcessorType(Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetActualProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetActualProcessorType
            {
                dPrintf("%s GetActualProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetActualProcessorType

//******************************************************************************

HRESULT
GetExelwtingProcessorType
(
    PULONG              Type
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExelwtingProcessorType control method
        hResult = pDbgControl->GetExelwtingProcessorType(Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExelwtingProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExelwtingProcessorType
            {
                dPrintf("%s GetExelwtingProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExelwtingProcessorType

//******************************************************************************

HRESULT
GetNumberPossibleExelwtingProcessorTypes
(
    PULONG              Number
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberPossibleExelwtingProcessorTypes control method
        hResult = pDbgControl->GetNumberPossibleExelwtingProcessorTypes(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberPossibleExelwtingProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberPossibleExelwtingProcessorTypes
            {
                dPrintf("%s GetNumberPossibleExelwtingProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberPossibleExelwtingProcessorTypes

//******************************************************************************

HRESULT
GetPossibleExelwtingProcessorTypes
(
    ULONG               Start,
    ULONG               Count,
    PULONG              Types
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Types != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPossibleExelwtingProcessorTypes control method
        hResult = pDbgControl->GetPossibleExelwtingProcessorTypes(Start, Count, Types);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPossibleExelwtingProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPossibleExelwtingProcessorTypes
            {
                dPrintf("%s GetPossibleExelwtingProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPossibleExelwtingProcessorTypes

//******************************************************************************

HRESULT
GetNumberProcessors
(
    PULONG              Number
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberProcessors control method
        hResult = pDbgControl->GetNumberProcessors(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberProcessors %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberProcessors
            {
                dPrintf("%s GetNumberProcessors %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberProcessors

//******************************************************************************

HRESULT
GetSystemVersion
(
    PULONG              PlatformId,
    PULONG              Major,
    PULONG              Minor,
    PSTR                ServicePackString,
    ULONG               ServicePackStringSize,
    PULONG              ServicePackStringUsed,
    PULONG              ServicePackNumber,
    PSTR                BuildString,
    ULONG               BuildStringSize,
    PULONG              BuildStringUsed
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(PlatformId != NULL);
    assert(Major != NULL);
    assert(Minor != NULL);
    assert(ServicePackNumber != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemVersion control method
        hResult = pDbgControl->GetSystemVersion(PlatformId, Major, Minor, ServicePackString, ServicePackStringSize, ServicePackStringUsed, ServicePackNumber, BuildString, BuildStringSize, BuildStringUsed);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemVersion %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemVersion
            {
                dPrintf("%s GetSystemVersion %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemVersion

//******************************************************************************

HRESULT
GetPageSize
(
    PULONG              Size
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Size != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPageSize control method
        hResult = pDbgControl->GetPageSize(Size);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPageSize %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPageSize
            {
                dPrintf("%s GetPageSize %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPageSize

//******************************************************************************

HRESULT
IsPointer64Bit()
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the IsPointer64Bit control method
        hResult = pDbgControl->IsPointer64Bit();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s IsPointer64Bit %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed IsPointer64Bit
            {
                dPrintf("%s IsPointer64Bit %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // IsPointer64Bit

//******************************************************************************

HRESULT
ReadBugCheckData
(
    PULONG              Code,
    PULONG64            Arg1,
    PULONG64            Arg2,
    PULONG64            Arg3,
    PULONG64            Arg4
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Code != NULL);
    assert(Arg1 != NULL);
    assert(Arg2 != NULL);
    assert(Arg3 != NULL);
    assert(Arg4 != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadBugCheckData control method
        hResult = pDbgControl->ReadBugCheckData(Code, Arg1, Arg2, Arg3, Arg4);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadBugCheckData %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadBugCheckData
            {
                dPrintf("%s ReadBugCheckData %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadBugCheckData

//******************************************************************************

HRESULT
GetNumberSupportedProcessorTypes
(
    PULONG              Number
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberSupportedProcessorTypes control method
        hResult = pDbgControl->GetNumberSupportedProcessorTypes(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberSupportedProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberSupportedProcessorTypes
            {
                dPrintf("%s GetNumberSupportedProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberSupportedProcessorTypes

//******************************************************************************

HRESULT
GetSupportedProcessorTypes
(
    ULONG               Start,
    ULONG               Count,
    PULONG              Types
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Types != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSupportedProcessorTypes control method
        hResult = pDbgControl->GetSupportedProcessorTypes(Start, Count, Types);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSupportedProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSupportedProcessorTypes
            {
                dPrintf("%s GetSupportedProcessorTypes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSupportedProcessorTypes

//******************************************************************************

HRESULT
GetProcessorTypeNames
(
    ULONG               Type,
    PSTR                FullNameBuffer,
    ULONG               FullNameBufferSize,
    PULONG              FullNameSize,
    PSTR                AbbrevNameBuffer,
    ULONG               AbbrevNameBufferSize,
    PULONG              AbbrevNameSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessorTypeNames control method
        hResult = pDbgControl->GetProcessorTypeNames(Type, FullNameBuffer, FullNameBufferSize, FullNameSize, AbbrevNameBuffer, AbbrevNameBufferSize, AbbrevNameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessorTypeNames %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessorTypeNames
            {
                dPrintf("%s GetProcessorTypeNames %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessorTypeNames

//******************************************************************************

HRESULT
GetEffectiveProcessorType
(
    PULONG              Type
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEffectiveProcessorType control method
        hResult = pDbgControl->GetEffectiveProcessorType(Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEffectiveProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEffectiveProcessorType
            {
                dPrintf("%s GetEffectiveProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEffectiveProcessorType

//******************************************************************************

HRESULT
SetEffectiveProcessorType
(
    ULONG               Type
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetEffectiveProcessorType control method
        hResult = pDbgControl->SetEffectiveProcessorType(Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetEffectiveProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetEffectiveProcessorType
            {
                dPrintf("%s SetEffectiveProcessorType %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetEffectiveProcessorType

//******************************************************************************

HRESULT
GetExelwtionStatus
(
    PULONG              Status
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Status != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExelwtionStatus control method
        hResult = pDbgControl->GetExelwtionStatus(Status);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExelwtionStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExelwtionStatus
            {
                dPrintf("%s GetExelwtionStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExelwtionStatus

//******************************************************************************

HRESULT
SetExelwtionStatus
(
    ULONG               Status
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetExelwtionStatus control method
        hResult = pDbgControl->SetExelwtionStatus(Status);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetExelwtionStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetExelwtionStatus
            {
                dPrintf("%s SetExelwtionStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetExelwtionStatus

//******************************************************************************

HRESULT
GetCodeLevel
(
    PULONG              Level
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Level != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetCodeLevel control method
        hResult = pDbgControl->GetCodeLevel(Level);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetCodeLevel %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetCodeLevel
            {
                dPrintf("%s GetCodeLevel %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetCodeLevel

//******************************************************************************

HRESULT
SetCodeLevel
(
    ULONG               Level
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetCodeLevel control method
        hResult = pDbgControl->SetCodeLevel(Level);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetCodeLevel %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetCodeLevel
            {
                dPrintf("%s SetCodeLevel %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetCodeLevel

//******************************************************************************

HRESULT
GetEngineOptions
(
    PULONG              Options
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEngineOptions control method
        hResult = pDbgControl->GetEngineOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEngineOptions
            {
                dPrintf("%s GetEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEngineOptions

//******************************************************************************

HRESULT
AddEngineOptions
(
    ULONG               Options
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddEngineOptions control method
        hResult = pDbgControl->AddEngineOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddEngineOptions
            {
                dPrintf("%s AddEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddEngineOptions

//******************************************************************************

HRESULT
RemoveEngineOptions
(
    ULONG               Options
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveEngineOptions control method
        hResult = pDbgControl->RemoveEngineOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveEngineOptions
            {
                dPrintf("%s RemoveEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveEngineOptions

//******************************************************************************

HRESULT
SetEngineOptions
(
    ULONG               Options
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetEngineOptions control method
        hResult = pDbgControl->SetEngineOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetEngineOptions
            {
                dPrintf("%s SetEngineOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetEngineOptions

//******************************************************************************

HRESULT
GetSystemErrorControl
(
    PULONG              OutputLevel,
    PULONG              BreakLevel
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(OutputLevel != NULL);
    assert(BreakLevel != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemErrorControl control method
        hResult = pDbgControl->GetSystemErrorControl(OutputLevel, BreakLevel);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemErrorControl %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemErrorControl
            {
                dPrintf("%s GetSystemErrorControl %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemErrorControl

//******************************************************************************

HRESULT
SetSystemErrorControl
(
    ULONG               OutputLevel,
    ULONG               BreakLevel
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSystemErrorControl control method
        hResult = pDbgControl->SetSystemErrorControl(OutputLevel, BreakLevel);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSystemErrorControl %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSystemErrorControl
            {
                dPrintf("%s SetSystemErrorControl %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSystemErrorControl

//******************************************************************************

HRESULT
GetTextMacro
(
    ULONG               Slot,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              MacroSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTextMacro control method
        hResult = pDbgControl->GetTextMacro(Slot, Buffer, BufferSize, MacroSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTextMacro %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTextMacro
            {
                dPrintf("%s GetTextMacro %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTextMacro

//******************************************************************************

HRESULT
SetTextMacro
(
    ULONG               Slot,
    PCSTR               Macro
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Macro != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetTextMacro control method
        hResult = pDbgControl->SetTextMacro(Slot, Macro);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetTextMacro %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetTextMacro
            {
                dPrintf("%s SetTextMacro %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetTextMacro

//******************************************************************************

HRESULT
GetRadix
(
    PULONG              Radix
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Radix != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetRadix control method
        hResult = pDbgControl->GetRadix(Radix);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetRadix %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetRadix
            {
                dPrintf("%s GetRadix %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetRadix

//******************************************************************************

HRESULT
SetRadix
(
    ULONG               Radix
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetRadix control method
        hResult = pDbgControl->SetRadix(Radix);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetRadix %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetRadix
            {
                dPrintf("%s SetRadix %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetRadix

//******************************************************************************

HRESULT
Evaluate
(
    PCSTR               Expression,
    ULONG               DesiredType,
    PDEBUG_VALUE        Value,
    PULONG              RemainderIndex
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Expression != NULL);
    assert(Value != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the Evaluate control method
        hResult = pDbgControl->Evaluate(Expression, DesiredType, Value, RemainderIndex);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Evaluate %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Evaluate
            {
                dPrintf("%s Evaluate %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Evaluate

//******************************************************************************

HRESULT
CoerceValue
(
    PDEBUG_VALUE        In,
    ULONG               OutType,
    PDEBUG_VALUE        Out
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(In != NULL);
    assert(Out != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CoerceValue control method
        hResult = pDbgControl->CoerceValue(In, OutType, Out);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CoerceValue %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CoerceValue
            {
                dPrintf("%s CoerceValue %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CoerceValue

//******************************************************************************

HRESULT
CoerceValues
(
    ULONG               Count,
    PDEBUG_VALUE        In,
    PULONG              OutTypes,
    PDEBUG_VALUE        Out
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(In != NULL);
    assert(OutTypes != NULL);
    assert(Out != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CoerceValues control method
        hResult = pDbgControl->CoerceValues(Count, In, OutTypes, Out);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CoerceValues %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CoerceValues
            {
                dPrintf("%s CoerceValues %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CoerceValues

//******************************************************************************

HRESULT
Execute
(
    ULONG               OutputControl,
    PCSTR               Command,
    ULONG               Flags
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Command != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the Execute control method
        hResult = pDbgControl->Execute(OutputControl, Command, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Execute %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Execute
            {
                dPrintf("%s Execute %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Execute

//******************************************************************************

HRESULT
ExelwteCommandFile
(
    ULONG               OutputControl,
    PCSTR               CommandFile,
    ULONG               Flags
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(CommandFile != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ExelwteCommandFile control method
        hResult = pDbgControl->ExelwteCommandFile(OutputControl, CommandFile, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ExelwteCommandFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ExelwteCommandFile
            {
                dPrintf("%s ExelwteCommandFile %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ExelwteCommandFile

//******************************************************************************

HRESULT
GetNumberBreakpoints
(
    PULONG              Number
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberBreakpoints control method
        hResult = pDbgControl->GetNumberBreakpoints(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberBreakpoints %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberBreakpoints
            {
                dPrintf("%s GetNumberBreakpoints %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberBreakpoints

//******************************************************************************

HRESULT
GetBreakpointByIndex
(
    ULONG               Index,
    PDEBUG_BREAKPOINT*  Bp
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetBreakpointByIndex control method
        hResult = pDbgControl->GetBreakpointByIndex(Index, Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetBreakpointByIndex %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetBreakpointByIndex
            {
                dPrintf("%s GetBreakpointByIndex %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetBreakpointByIndex

//******************************************************************************

HRESULT
GetBreakpointById
(
    ULONG               Id,
    PDEBUG_BREAKPOINT*  Bp
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetBreakpointById control method
        hResult = pDbgControl->GetBreakpointById(Id, Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetBreakpointById %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetBreakpointById
            {
                dPrintf("%s GetBreakpointById %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetBreakpointById

//******************************************************************************

HRESULT
GetBreakpointParameters
(
    ULONG               Count,
    PULONG              Ids,
    ULONG               Start,
    PDEBUG_BREAKPOINT_PARAMETERS Params
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetBreakpointParameters control method
        hResult = pDbgControl->GetBreakpointParameters(Count, Ids, Start, Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetBreakpointParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetBreakpointParameters
            {
                dPrintf("%s GetBreakpointParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetBreakpointParameters

//******************************************************************************

HRESULT
AddBreakpoint
(
    ULONG               Type,
    ULONG               DesiredId,
    PDEBUG_BREAKPOINT*  Bp
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddBreakpoint control method
        hResult = pDbgControl->AddBreakpoint(Type, DesiredId, Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddBreakpoint %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddBreakpoint
            {
                dPrintf("%s AddBreakpoint %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddBreakpoint

//******************************************************************************

HRESULT
RemoveBreakpoint
(
    PDEBUG_BREAKPOINT   Bp
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveBreakpoint control method
        hResult = pDbgControl->RemoveBreakpoint(Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveBreakpoint %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveBreakpoint
            {
                dPrintf("%s RemoveBreakpoint %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveBreakpoint

//******************************************************************************

HRESULT
AddExtension
(
    PCSTR               Path,
    ULONG               Flags,
    PULONG64            Handle
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);
    assert(Handle != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddExtension control method
        hResult = pDbgControl->AddExtension(Path, Flags, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddExtension %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddExtension
            {
                dPrintf("%s AddExtension %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddExtension

//******************************************************************************

HRESULT
RemoveExtension
(
    ULONG64             Handle
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveExtension control method
        hResult = pDbgControl->RemoveExtension(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveExtension %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveExtension
            {
                dPrintf("%s RemoveExtension %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveExtension

//******************************************************************************

HRESULT
GetExtensionByPath
(
    PCSTR               Path,
    PULONG64            Handle
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);
    assert(Handle != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExtensionByPath control method
        hResult = pDbgControl->GetExtensionByPath(Path, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExtensionByPath %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExtensionByPath
            {
                dPrintf("%s GetExtensionByPath %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExtensionByPath

//******************************************************************************

HRESULT
CallExtension
(
    ULONG64             Handle,
    PCSTR               Function,
    PCSTR               Arguments
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Function != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CallExtension control method
        hResult = pDbgControl->CallExtension(Handle, Function, Arguments);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CallExtension %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CallExtension
            {
                dPrintf("%s CallExtension %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CallExtension

//******************************************************************************

HRESULT
GetExtensionFunction
(
    ULONG64             Handle,
    PCSTR               FuncName,
    FARPROC*            Function
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(FuncName != NULL);
    assert(Function != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExtensionFunction control method
        hResult = pDbgControl->GetExtensionFunction(Handle, FuncName, Function);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExtensionFunction %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExtensionFunction
            {
                dPrintf("%s GetExtensionFunction %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExtensionFunction

//******************************************************************************

HRESULT
GetWindbgExtensionApis32
(
    PWINDBG_EXTENSION_APIS32 Api
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Api != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetWindbgExtensionApis32 control method
        hResult = pDbgControl->GetWindbgExtensionApis32(Api);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetWindbgExtensionApis32 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetWindbgExtensionApis32
            {
                dPrintf("%s GetWindbgExtensionApis32 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetWindbgExtensionApis32

//******************************************************************************

HRESULT
GetWindbgExtensionApis64
(
    PWINDBG_EXTENSION_APIS64 Api
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Api != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetWindbgExtensionApis64 control method
        hResult = pDbgControl->GetWindbgExtensionApis64(Api);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetWindbgExtensionApis64 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetWindbgExtensionApis64
            {
                dPrintf("%s GetWindbgExtensionApis64 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetWindbgExtensionApis64

//******************************************************************************

HRESULT
GetNumberEventFilters
(
    PULONG              SpecificEvents,
    PULONG              SpecificExceptions,
    PULONG              ArbitraryExceptions
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(SpecificEvents != NULL);
    assert(SpecificExceptions != NULL);
    assert(ArbitraryExceptions != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberEventFilters control method
        hResult = pDbgControl->GetNumberEventFilters(SpecificEvents, SpecificExceptions, ArbitraryExceptions);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberEventFilters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberEventFilters
            {
                dPrintf("%s GetNumberEventFilters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberEventFilters

//******************************************************************************

HRESULT
GetEventFilterText
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              TextSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventFilterText control method
        hResult = pDbgControl->GetEventFilterText(Index, Buffer, BufferSize, TextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventFilterText %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventFilterText
            {
                dPrintf("%s GetEventFilterText %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventFilterText

//******************************************************************************

HRESULT
GetEventFilterCommand
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventFilterCommand control method
        hResult = pDbgControl->GetEventFilterCommand(Index, Buffer, BufferSize, CommandSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventFilterCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventFilterCommand
            {
                dPrintf("%s GetEventFilterCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventFilterCommand

//******************************************************************************

HRESULT
SetEventFilterCommand
(
    ULONG               Index,
    PCSTR               Command
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Command != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetEventFilterCommand control method
        hResult = pDbgControl->SetEventFilterCommand(Index, Command);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetEventFilterCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetEventFilterCommand
            {
                dPrintf("%s SetEventFilterCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetEventFilterCommand

//******************************************************************************

HRESULT
GetSpecificFilterParameters
(
    ULONG               Start,
    ULONG               Count,
    PDEBUG_SPECIFIC_FILTER_PARAMETERS Params
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Params != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSpecificFilterParameters control method
        hResult = pDbgControl->GetSpecificFilterParameters(Start, Count, Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSpecificFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSpecificFilterParameters
            {
                dPrintf("%s GetSpecificFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSpecificFilterParameters

//******************************************************************************

HRESULT
SetSpecificFilterParameters
(
    ULONG               Start,
    ULONG               Count,
    PDEBUG_SPECIFIC_FILTER_PARAMETERS Params
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Params != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSpecificFilterParameters control method
        hResult = pDbgControl->SetSpecificFilterParameters(Start, Count, Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSpecificFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSpecificFilterParameters
            {
                dPrintf("%s SetSpecificFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSpecificFilterParameters

//******************************************************************************

HRESULT
GetSpecificFilterArgument
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              ArgumentSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSpecificFilterArgument control method
        hResult = pDbgControl->GetSpecificFilterArgument(Index, Buffer, BufferSize, ArgumentSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSpecificFilterArgument %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSpecificFilterArgument
            {
                dPrintf("%s GetSpecificFilterArgument %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSpecificFilterArgument

//******************************************************************************

HRESULT
SetSpecificFilterArgument
(
    ULONG               Index,
    PCSTR               Argument
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Argument != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSpecificFilterArgument control method
        hResult = pDbgControl->SetSpecificFilterArgument(Index, Argument);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSpecificFilterArgument %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSpecificFilterArgument
            {
                dPrintf("%s SetSpecificFilterArgument %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSpecificFilterArgument

//******************************************************************************

HRESULT
GetExceptionFilterParameters
(
    ULONG               Count,
    PULONG              Codes,
    ULONG               Start,
    PDEBUG_EXCEPTION_FILTER_PARAMETERS Params
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Params != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExceptionFilterParameters control method
        hResult = pDbgControl->GetExceptionFilterParameters(Count, Codes, Start, Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExceptionFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExceptionFilterParameters
            {
                dPrintf("%s GetExceptionFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExceptionFilterParameters

//******************************************************************************

HRESULT
SetExceptionFilterParameters
(
    ULONG               Count,
    PDEBUG_EXCEPTION_FILTER_PARAMETERS Params
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Params != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetExceptionFilterParameters control method
        hResult = pDbgControl->SetExceptionFilterParameters(Count, Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetExceptionFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetExceptionFilterParameters
            {
                dPrintf("%s SetExceptionFilterParameters %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetExceptionFilterParameters

//******************************************************************************

HRESULT
GetExceptionFilterSecondCommand
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExceptionFilterSecondCommand control method
        hResult = pDbgControl->GetExceptionFilterSecondCommand(Index, Buffer, BufferSize, CommandSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExceptionFilterSecondCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExceptionFilterSecondCommand
            {
                dPrintf("%s GetExceptionFilterSecondCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExceptionFilterSecondCommand

//******************************************************************************

HRESULT
SetExceptionFilterSecondCommand
(
    ULONG               Index,
    PCSTR               Command
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Command != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetExceptionFilterSecondCommand control method
        hResult = pDbgControl->SetExceptionFilterSecondCommand(Index, Command);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetExceptionFilterSecondCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetExceptionFilterSecondCommand
            {
                dPrintf("%s SetExceptionFilterSecondCommand %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetExceptionFilterSecondCommand

//******************************************************************************

HRESULT
WaitForEvent
(
    ULONG               Flags,
    ULONG               Timeout
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WaitForEvent control method
        hResult = pDbgControl->WaitForEvent(Flags, Timeout);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WaitForEvent %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WaitForEvent
            {
                dPrintf("%s WaitForEvent %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WaitForEvent

//******************************************************************************

HRESULT
GetLastEventInformation
(
    PULONG              Type,
    PULONG              ProcessId,
    PULONG              ThreadId,
    PVOID               ExtraInformation,
    ULONG               ExtraInformationSize,
    PULONG              ExtraInformationUsed,
    PSTR                Description,
    ULONG               DescriptionSize,
    PULONG              DescriptionUsed
)
{
    PDEBUG_CONTROL      pDbgControl = debugControlInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);
    assert(ProcessId != NULL);
    assert(ThreadId != NULL);

    // Check for valid debug control interface
    if (pDbgControl != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLastEventInformation control method
        hResult = pDbgControl->GetLastEventInformation(Type, ProcessId, ThreadId, ExtraInformation, ExtraInformationSize, ExtraInformationUsed, Description, DescriptionSize, DescriptionUsed);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLastEventInformation %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLastEventInformation
            {
                dPrintf("%s GetLastEventInformation %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLastEventInformation

//******************************************************************************

HRESULT
GetLwrrentTimeDate
(
    PULONG              TimeDate
)
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(TimeDate != NULL);

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentTimeDate control method
        hResult = pDbgControl2->GetLwrrentTimeDate(TimeDate);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentTimeDate %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentTimeDate
            {
                dPrintf("%s GetLwrrentTimeDate %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentTimeDate

//******************************************************************************

HRESULT
GetLwrrentSystemUpTime
(
    PULONG              UpTime
)
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(UpTime != NULL);

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentSystemUpTime control method
        hResult = pDbgControl2->GetLwrrentSystemUpTime(UpTime);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentSystemUpTime %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentSystemUpTime
            {
                dPrintf("%s GetLwrrentSystemUpTime %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentSystemUpTime

//******************************************************************************

HRESULT
GetDumpFormatFlags
(
    PULONG              FormatFlags
)
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(FormatFlags != NULL);

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDumpFormatFlags control method
        hResult = pDbgControl2->GetDumpFormatFlags(FormatFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDumpFormatFlags %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDumpFormatFlags
            {
                dPrintf("%s GetDumpFormatFlags %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDumpFormatFlags

//******************************************************************************

HRESULT
GetNumberTextReplacements
(
    PULONG              NumRepl
)
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(NumRepl != NULL);

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberTextReplacements control method
        hResult = pDbgControl2->GetNumberTextReplacements(NumRepl);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberTextReplacements %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberTextReplacements
            {
                dPrintf("%s GetNumberTextReplacements %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberTextReplacements

//******************************************************************************

HRESULT
GetTextReplacement
(
    PCSTR               SrcText,
    ULONG               Index,
    PSTR                SrcBuffer,
    ULONG               SrcBufferSize,
    PULONG              SrcSize,
    PSTR                DstBuffer,
    ULONG               DstBufferSize,
    PULONG              DstSize
)
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTextReplacement control method
        hResult = pDbgControl2->GetTextReplacement(SrcText, Index, SrcBuffer, SrcBufferSize, SrcSize, DstBuffer, DstBufferSize, DstSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTextReplacement %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTextReplacement
            {
                dPrintf("%s GetTextReplacement %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTextReplacement

//******************************************************************************

HRESULT
SetTextReplacement
(
    PCSTR               SrcText,
    PCSTR               DstText
)
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(SrcText != NULL);

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetTextReplacement control method
        hResult = pDbgControl2->SetTextReplacement(SrcText, DstText);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetTextReplacement %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetTextReplacement
            {
                dPrintf("%s SetTextReplacement %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetTextReplacement

//******************************************************************************

HRESULT
RemoveTextReplacements()
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveTextReplacements control method
        hResult = pDbgControl2->RemoveTextReplacements();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveTextReplacements %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveTextReplacements
            {
                dPrintf("%s RemoveTextReplacements %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveTextReplacements

//******************************************************************************

HRESULT
OutputTextReplacements
(
    ULONG               OutputControl,
    ULONG               Flags
)
{
    PDEBUG_CONTROL2     pDbgControl2 = debugControl2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputTextReplacements control method
        hResult = pDbgControl2->OutputTextReplacements(OutputControl, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputTextReplacements %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputTextReplacements
            {
                dPrintf("%s OutputTextReplacements %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputTextReplacements

//******************************************************************************

HRESULT
GetAssemblyOptions
(
    PULONG              Options
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetAssemblyOptions control method
        hResult = pDbgControl3->GetAssemblyOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetAssemblyOptions
            {
                dPrintf("%s GetAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetAssemblyOptions

//******************************************************************************

HRESULT
AddAssemblyOptions
(
    ULONG               Options
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddAssemblyOptions control method
        hResult = pDbgControl3->AddAssemblyOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddAssemblyOptions
            {
                dPrintf("%s AddAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddAssemblyOptions

//******************************************************************************

HRESULT
RemoveAssemblyOptions
(
    ULONG               Options
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveAssemblyOptions control method
        hResult = pDbgControl3->RemoveAssemblyOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveAssemblyOptions
            {
                dPrintf("%s RemoveAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveAssemblyOptions

//******************************************************************************

HRESULT
SetAssemblyOptions
(
    ULONG               Options
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetAssemblyOptions control method
        hResult = pDbgControl3->SetAssemblyOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetAssemblyOptions
            {
                dPrintf("%s SetAssemblyOptions %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetAssemblyOptions

//******************************************************************************

HRESULT
GetExpressionSyntax
(
    PULONG              Flags
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Flags != NULL);

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExpressionSyntax control method
        hResult = pDbgControl3->GetExpressionSyntax(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExpressionSyntax %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExpressionSyntax
            {
                dPrintf("%s GetExpressionSyntax %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExpressionSyntax

//******************************************************************************

HRESULT
SetExpressionSyntax
(
    ULONG               Flags
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetExpressionSyntax control method
        hResult = pDbgControl3->SetExpressionSyntax(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetExpressionSyntax %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetExpressionSyntax
            {
                dPrintf("%s SetExpressionSyntax %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetExpressionSyntax

//******************************************************************************

HRESULT
SetExpressionSyntaxByName
(
    PCSTR               AbbrevName
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(AbbrevName != NULL);

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetExpressionSyntaxByName control method
        hResult = pDbgControl3->SetExpressionSyntaxByName(AbbrevName);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetExpressionSyntaxByName %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetExpressionSyntaxByName
            {
                dPrintf("%s SetExpressionSyntaxByName %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetExpressionSyntaxByName

//******************************************************************************

HRESULT
GetNumberExpressionSyntaxes
(
    PULONG              Number
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberExpressionSyntaxes control method
        hResult = pDbgControl3->GetNumberExpressionSyntaxes(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberExpressionSyntaxes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberExpressionSyntaxes
            {
                dPrintf("%s GetNumberExpressionSyntaxes %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberExpressionSyntaxes

//******************************************************************************

HRESULT
GetExpressionSyntaxNames
(
    ULONG               Index,
    PSTR                FullNameBuffer,
    ULONG               FullNameBufferSize,
    PULONG              FullNameSize,
    PSTR                AbbrevNameBuffer,
    ULONG               AbbrevNameBufferSize,
    PULONG              AbbrevNameSize
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExpressionSyntaxNames control method
        hResult = pDbgControl3->GetExpressionSyntaxNames(Index, FullNameBuffer, FullNameBufferSize, FullNameSize, AbbrevNameBuffer, AbbrevNameBufferSize, AbbrevNameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExpressionSyntaxNames %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExpressionSyntaxNames
            {
                dPrintf("%s GetExpressionSyntaxNames %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExpressionSyntaxNames

//******************************************************************************

HRESULT
GetNumberEvents
(
    PULONG              Events
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Events != NULL);

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberEvents control method
        hResult = pDbgControl3->GetNumberEvents(Events);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberEvents %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberEvents
            {
                dPrintf("%s GetNumberEvents %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberEvents

//******************************************************************************

HRESULT
GetEventIndexDescription
(
    ULONG               Index,
    ULONG               Which,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              DescSize
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventIndexDescription control method
        hResult = pDbgControl3->GetEventIndexDescription(Index, Which, Buffer, BufferSize, DescSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventIndexDescription %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventIndexDescription
            {
                dPrintf("%s GetEventIndexDescription %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventIndexDescription

//******************************************************************************

HRESULT
GetLwrrentEventIndex
(
    PULONG              Index
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Index != NULL);

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentEventIndex control method
        hResult = pDbgControl3->GetLwrrentEventIndex(Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentEventIndex %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentEventIndex
            {
                dPrintf("%s GetLwrrentEventIndex %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentEventIndex

//******************************************************************************

HRESULT
SetNextEventIndex
(
    ULONG               Relation,
    ULONG               Value,
    PULONG              NextIndex
)
{
    PDEBUG_CONTROL3     pDbgControl3 = debugControl3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(NextIndex != NULL);

    // Check for valid debug control interface
    if (pDbgControl3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetNextEventIndex control method
        hResult = pDbgControl3->SetNextEventIndex(Relation, Value, NextIndex);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetNextEventIndex %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetNextEventIndex
            {
                dPrintf("%s SetNextEventIndex %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetNextEventIndex

//******************************************************************************

HRESULT
GetLogFileWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              FileSize,
    PBOOL               Append
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Append != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLogFileWide control method
        hResult = pDbgControl4->GetLogFileWide(Buffer, BufferSize, FileSize, Append);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLogFileWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLogFileWide
            {
                dPrintf("%s GetLogFileWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLogFileWide

//******************************************************************************

HRESULT
OpenLogFileWide
(
    PCWSTR              File,
    BOOL                Append
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OpenLogFileWide control method
        hResult = pDbgControl4->OpenLogFileWide(File, Append);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OpenLogFileWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OpenLogFileWide
            {
                dPrintf("%s OpenLogFileWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OpenLogFileWide

//******************************************************************************

HRESULT
InputWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              InputSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the InputWide control method
        hResult = pDbgControl4->InputWide(Buffer, BufferSize, InputSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s InputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed InputWide
            {
                dPrintf("%s InputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // InputWide

//******************************************************************************

HRESULT
ReturnInputWide
(
    PCWSTR              Buffer
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReturnInputWide control method
        hResult = pDbgControl4->ReturnInputWide(Buffer);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng input)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_INPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReturnInputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReturnInputWide
            {
                dPrintf("%s ReturnInputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReturnInputWide

//******************************************************************************

HRESULT
OutputWide
(
    ULONG               Mask,
    PCWSTR              Format,
    ...
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputVaListWide control method
        hResult = pDbgControl4->OutputVaListWide(Mask, Format, reinterpret_cast<va_list>(&Format + 1));

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputWide
            {
                dPrintf("%s OutputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputWide

//******************************************************************************

HRESULT
OutputVaListWide
(
    ULONG               Mask,
    PCWSTR              Format,
    va_list             Args
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputVaListWide control method
        hResult = pDbgControl4->OutputVaListWide(Mask, Format, Args);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputVaListWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputVaListWide
            {
                dPrintf("%s OutputVaListWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputVaListWide

//******************************************************************************

HRESULT
ControlledOutputWide
(
    ULONG               OutputControl,
    ULONG               Mask,
    PCWSTR              Format,
    ...
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ControlledOutputVaListWide control method
        hResult = pDbgControl4->ControlledOutputVaListWide(OutputControl, Mask, Format, reinterpret_cast<va_list>(&Format + 1));

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ControlledOutputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ControlledOutputWide
            {
                dPrintf("%s ControlledOutputWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ControlledOutputWide

//******************************************************************************

HRESULT
ControlledOutputVaListWide
(
    ULONG               OutputControl,
    ULONG               Mask,
    PCWSTR              Format,
    va_list             Args
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ControlledOutputVaListWide control method
        hResult = pDbgControl4->ControlledOutputVaListWide(OutputControl, Mask, Format, Args);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ControlledOutputVaListWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ControlledOutputVaListWide
            {
                dPrintf("%s ControlledOutputVaListWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ControlledOutputVaListWide

//******************************************************************************

HRESULT
OutputPromptWide
(
    ULONG               OutputControl,
    PCWSTR              Format,
    ...
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputPromptVaListWide control method
        hResult = pDbgControl4->OutputPromptVaListWide(OutputControl, Format, reinterpret_cast<va_list>(&Format + 1));

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputPromptWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputPromptWide
            {
                dPrintf("%s OutputPromptWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputPromptWide

//******************************************************************************

HRESULT
OutputPromptVaListWide
(
    ULONG               OutputControl,
    PCWSTR              Format,
    va_list             Args
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputPromptVaListWide control method
        hResult = pDbgControl4->OutputPromptVaListWide(OutputControl, Format, Args);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested (or DbgEng output)
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL) || VERBOSE_LEVEL(VERBOSE_DBGENG_OUTPUT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputPromptVaListWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputPromptVaListWide
            {
                dPrintf("%s OutputPromptVaListWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputPromptVaListWide

//******************************************************************************

HRESULT
GetPromptTextWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              TextSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPromptTextWide control method
        hResult = pDbgControl4->GetPromptTextWide(Buffer, BufferSize, TextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPromptTextWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPromptTextWide
            {
                dPrintf("%s GetPromptTextWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPromptTextWide

//******************************************************************************

HRESULT
AssembleWide
(
    ULONG64             Offset,
    PCWSTR              Instr,
    PULONG64            EndOffset
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Instr != NULL);
    assert(EndOffset != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AssembleWide control method
        hResult = pDbgControl4->AssembleWide(Offset, Instr, EndOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AssembleWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AssembleWide
            {
                dPrintf("%s AssembleWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AssembleWide

//******************************************************************************

HRESULT
DisassembleWide
(
    ULONG64             Offset,
    ULONG               Flags,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              DisassemblySize,
    PULONG64            EndOffset
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(EndOffset != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the DisassembleWide control method
        hResult = pDbgControl4->DisassembleWide(Offset, Flags, Buffer, BufferSize, DisassemblySize, EndOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s DisassembleWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed DisassembleWide
            {
                dPrintf("%s DisassembleWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // DisassembleWide

//******************************************************************************

HRESULT
GetProcessorTypeNamesWide
(
    ULONG               Type,
    PWSTR               FullNameBuffer,
    ULONG               FullNameBufferSize,
    PULONG              FullNameSize,
    PWSTR               AbbrevNameBuffer,
    ULONG               AbbrevNameBufferSize,
    PULONG              AbbrevNameSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessorTypeNamesWide control method
        hResult = pDbgControl4->GetProcessorTypeNamesWide(Type, FullNameBuffer, FullNameBufferSize, FullNameSize, AbbrevNameBuffer, AbbrevNameBufferSize, AbbrevNameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessorTypeNamesWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessorTypeNamesWide
            {
                dPrintf("%s GetProcessorTypeNamesWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessorTypeNamesWide

//******************************************************************************

HRESULT
GetTextMacroWide
(
    ULONG               Slot,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              MacroSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTextMacroWide control method
        hResult = pDbgControl4->GetTextMacroWide(Slot, Buffer, BufferSize, MacroSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTextMacroWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTextMacroWide
            {
                dPrintf("%s GetTextMacroWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTextMacroWide

//******************************************************************************

HRESULT
SetTextMacroWide
(
    ULONG               Slot,
    PCWSTR              Macro
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Macro != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetTextMacroWide control method
        hResult = pDbgControl4->SetTextMacroWide(Slot, Macro);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetTextMacroWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetTextMacroWide
            {
                dPrintf("%s SetTextMacroWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetTextMacroWide

//******************************************************************************

HRESULT
EvaluateWide
(
    PCWSTR              Expression,
    ULONG               DesiredType,
    PDEBUG_VALUE        Value,
    PULONG              RemainderIndex
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Expression != NULL);
    assert(Value != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the EvaluateWide control method
        hResult = pDbgControl4->EvaluateWide(Expression, DesiredType, Value, RemainderIndex);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s EvaluateWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed EvaluateWide
            {
                dPrintf("%s EvaluateWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // EvaluateWide

//******************************************************************************

HRESULT
ExelwteWide
(
    ULONG               OutputControl,
    PCWSTR              Command,
    ULONG               Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Command != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ExelwteWide control method
        hResult = pDbgControl4->ExelwteWide(OutputControl, Command, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ExelwteWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ExelwteWide
            {
                dPrintf("%s ExelwteWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ExelwteWide

//******************************************************************************

HRESULT
ExelwteCommandFileWide
(
    ULONG               OutputControl,
    PCWSTR              CommandFile,
    ULONG               Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(CommandFile != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ExelwteCommandFileWide control method
        hResult = pDbgControl4->ExelwteCommandFileWide(OutputControl, CommandFile, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ExelwteCommandFileWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ExelwteCommandFileWide
            {
                dPrintf("%s ExelwteCommandFileWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ExelwteCommandFileWide

//******************************************************************************

HRESULT
GetBreakpointByIndex2
(
    ULONG               Index,
    PDEBUG_BREAKPOINT2* Bp
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetBreakpointByIndex2 control method
        hResult = pDbgControl4->GetBreakpointByIndex2(Index, Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetBreakpointByIndex2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetBreakpointByIndex2
            {
                dPrintf("%s GetBreakpointByIndex2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetBreakpointByIndex2

//******************************************************************************

HRESULT
GetBreakpointById2
(
    ULONG               Id,
    PDEBUG_BREAKPOINT2* Bp
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetBreakpointById2 control method
        hResult = pDbgControl4->GetBreakpointById2(Id, Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetBreakpointById2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetBreakpointById2
            {
                dPrintf("%s GetBreakpointById2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetBreakpointById2

//******************************************************************************

HRESULT
AddBreakpoint2
(
    ULONG               Type,
    ULONG               DesiredId,
    PDEBUG_BREAKPOINT2* Bp
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddBreakpoint2 control method
        hResult = pDbgControl4->AddBreakpoint2(Type, DesiredId, Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddBreakpoint2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddBreakpoint2
            {
                dPrintf("%s AddBreakpoint2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddBreakpoint2

//******************************************************************************

HRESULT
RemoveBreakpoint2
(
    PDEBUG_BREAKPOINT2  Bp
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Bp != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveBreakpoint2 control method
        hResult = pDbgControl4->RemoveBreakpoint2(Bp);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveBreakpoint2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveBreakpoint2
            {
                dPrintf("%s RemoveBreakpoint2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveBreakpoint2

//******************************************************************************

HRESULT
AddExtensionWide
(
    PCWSTR              Path,
    ULONG               Flags,
    PULONG64            Handle
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);
    assert(Handle != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddExtensionWide control method
        hResult = pDbgControl4->AddExtensionWide(Path, Flags, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddExtensionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddExtensionWide
            {
                dPrintf("%s AddExtensionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddExtensionWide

//******************************************************************************

HRESULT
GetExtensionByPathWide
(
    PCWSTR              Path,
    PULONG64            Handle
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);
    assert(Handle != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExtensionByPathWide control method
        hResult = pDbgControl4->GetExtensionByPathWide(Path, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExtensionByPathWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExtensionByPathWide
            {
                dPrintf("%s GetExtensionByPathWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExtensionByPathWide

//******************************************************************************

HRESULT
CallExtensionWide
(
    ULONG64             Handle,
    PCWSTR              Function,
    PCWSTR              Arguments
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Function != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CallExtensionWide control method
        hResult = pDbgControl4->CallExtensionWide(Handle, Function, Arguments);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CallExtensionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CallExtensionWide
            {
                dPrintf("%s CallExtensionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CallExtensionWide

//******************************************************************************

HRESULT
GetExtensionFunctionWide
(
    ULONG64             Handle,
    PCWSTR              FuncName,
    FARPROC*            Function
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(FuncName != NULL);
    assert(Function != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExtensionFunctionWide control method
        hResult = pDbgControl4->GetExtensionFunctionWide(Handle, FuncName, Function);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExtensionFunctionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExtensionFunctionWide
            {
                dPrintf("%s GetExtensionFunctionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExtensionFunctionWide

//******************************************************************************

HRESULT
GetEventFilterTextWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              TextSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventFilterTextWide control method
        hResult = pDbgControl4->GetEventFilterTextWide(Index, Buffer, BufferSize, TextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventFilterTextWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventFilterTextWide
            {
                dPrintf("%s GetEventFilterTextWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventFilterTextWide

//******************************************************************************

HRESULT
GetEventFilterCommandWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventFilterCommandWide control method
        hResult = pDbgControl4->GetEventFilterCommandWide(Index, Buffer, BufferSize, CommandSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventFilterCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventFilterCommandWide
            {
                dPrintf("%s GetEventFilterCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventFilterCommandWide

//******************************************************************************

HRESULT
SetEventFilterCommandWide
(
    ULONG               Index,
    PCWSTR              Command
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Command != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetEventFilterCommandWide control method
        hResult = pDbgControl4->SetEventFilterCommandWide(Index, Command);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetEventFilterCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetEventFilterCommandWide
            {
                dPrintf("%s SetEventFilterCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetEventFilterCommandWide

//******************************************************************************

HRESULT
GetSpecificFilterArgumentWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              ArgumentSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSpecificFilterArgumentWide control method
        hResult = pDbgControl4->GetSpecificFilterArgumentWide(Index, Buffer, BufferSize, ArgumentSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSpecificFilterArgumentWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSpecificFilterArgumentWide
            {
                dPrintf("%s GetSpecificFilterArgumentWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSpecificFilterArgumentWide

//******************************************************************************

HRESULT
SetSpecificFilterArgumentWide
(
    ULONG               Index,
    PCWSTR              Argument
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Argument != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSpecificFilterArgumentWide control method
        hResult = pDbgControl4->SetSpecificFilterArgumentWide(Index, Argument);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSpecificFilterArgumentWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSpecificFilterArgumentWide
            {
                dPrintf("%s SetSpecificFilterArgumentWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSpecificFilterArgumentWide

//******************************************************************************

HRESULT
GetExceptionFilterSecondCommandWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              CommandSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExceptionFilterSecondCommandWide control method
        hResult = pDbgControl4->GetExceptionFilterSecondCommandWide(Index, Buffer, BufferSize, CommandSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExceptionFilterSecondCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExceptionFilterSecondCommandWide
            {
                dPrintf("%s GetExceptionFilterSecondCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExceptionFilterSecondCommandWide

//******************************************************************************

HRESULT
SetExceptionFilterSecondCommandWide
(
    ULONG               Index,
    PCWSTR              Command
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Command != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetExceptionFilterSecondCommandWide control method
        hResult = pDbgControl4->SetExceptionFilterSecondCommandWide(Index, Command);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetExceptionFilterSecondCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetExceptionFilterSecondCommandWide
            {
                dPrintf("%s SetExceptionFilterSecondCommandWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetExceptionFilterSecondCommandWide

//******************************************************************************

HRESULT
GetLastEventInformationWide
(
    PULONG              Type,
    PULONG              ProcessId,
    PULONG              ThreadId,
    PVOID               ExtraInformation,
    ULONG               ExtraInformationSize,
    PULONG              ExtraInformationUsed,
    PWSTR               Description,
    ULONG               DescriptionSize,
    PULONG              DescriptionUsed
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);
    assert(ProcessId != NULL);
    assert(ThreadId != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLastEventInformationWide control method
        hResult = pDbgControl4->GetLastEventInformationWide(Type, ProcessId, ThreadId, ExtraInformation, ExtraInformationSize, ExtraInformationUsed, Description, DescriptionSize, DescriptionUsed);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLastEventInformationWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLastEventInformationWide
            {
                dPrintf("%s GetLastEventInformationWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLastEventInformationWide

//******************************************************************************

HRESULT
GetTextReplacementWide
(
    PCWSTR              SrcText,
    ULONG               Index,
    PWSTR               SrcBuffer,
    ULONG               SrcBufferSize,
    PULONG              SrcSize,
    PWSTR               DstBuffer,
    ULONG               DstBufferSize,
    PULONG              DstSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTextReplacementWide control method
        hResult = pDbgControl4->GetTextReplacementWide(SrcText, Index, SrcBuffer, SrcBufferSize, SrcSize, DstBuffer, DstBufferSize, DstSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTextReplacementWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTextReplacementWide
            {
                dPrintf("%s GetTextReplacementWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTextReplacementWide

//******************************************************************************

HRESULT
SetTextReplacementWide
(
    PCWSTR              SrcText,
    PCWSTR              DstText
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(SrcText != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetTextReplacementWide control method
        hResult = pDbgControl4->SetTextReplacementWide(SrcText, DstText);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetTextReplacementWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetTextReplacementWide
            {
                dPrintf("%s SetTextReplacementWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetTextReplacementWide

//******************************************************************************

HRESULT
SetExpressionSyntaxByNameWide
(
    PCWSTR              AbbrevName
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(AbbrevName != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetExpressionSyntaxByNameWide control method
        hResult = pDbgControl4->SetExpressionSyntaxByNameWide(AbbrevName);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetExpressionSyntaxByNameWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetExpressionSyntaxByNameWide
            {
                dPrintf("%s SetExpressionSyntaxByNameWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetExpressionSyntaxByNameWide

//******************************************************************************

HRESULT
GetExpressionSyntaxNamesWide
(
    ULONG               Index,
    PWSTR               FullNameBuffer,
    ULONG               FullNameBufferSize,
    PULONG              FullNameSize,
    PWSTR               AbbrevNameBuffer,
    ULONG               AbbrevNameBufferSize,
    PULONG              AbbrevNameSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExpressionSyntaxNamesWide control method
        hResult = pDbgControl4->GetExpressionSyntaxNamesWide(Index, FullNameBuffer, FullNameBufferSize, FullNameSize, AbbrevNameBuffer, AbbrevNameBufferSize, AbbrevNameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExpressionSyntaxNamesWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExpressionSyntaxNamesWide
            {
                dPrintf("%s GetExpressionSyntaxNamesWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExpressionSyntaxNamesWide

//******************************************************************************

HRESULT
GetEventIndexDescriptionWide
(
    ULONG               Index,
    ULONG               Which,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              DescSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventIndexDescriptionWide control method
        hResult = pDbgControl4->GetEventIndexDescriptionWide(Index, Which, Buffer, BufferSize, DescSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventIndexDescriptionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventIndexDescriptionWide
            {
                dPrintf("%s GetEventIndexDescriptionWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventIndexDescriptionWide

//******************************************************************************

HRESULT
GetLogFile2
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              FileSize,
    PULONG              Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Flags != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLogFile2 control method
        hResult = pDbgControl4->GetLogFile2(Buffer, BufferSize, FileSize, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLogFile2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLogFile2
            {
                dPrintf("%s GetLogFile2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLogFile2

//******************************************************************************

HRESULT
OpenLogFile2
(
    PCSTR               File,
    ULONG               Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OpenLogFile2 control method
        hResult = pDbgControl4->OpenLogFile2(File, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OpenLogFile2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OpenLogFile2
            {
                dPrintf("%s OpenLogFile2 %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OpenLogFile2

//******************************************************************************

HRESULT
GetLogFile2Wide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              FileSize,
    PULONG              Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Flags != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLogFile2Wide control method
        hResult = pDbgControl4->GetLogFile2Wide(Buffer, BufferSize, FileSize, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLogFile2Wide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLogFile2Wide
            {
                dPrintf("%s GetLogFile2Wide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLogFile2Wide

//******************************************************************************

HRESULT
OpenLogFile2Wide
(
    PCWSTR              File,
    ULONG               Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OpenLogFile2Wide control method
        hResult = pDbgControl4->OpenLogFile2Wide(File, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OpenLogFile2Wide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OpenLogFile2Wide
            {
                dPrintf("%s OpenLogFile2Wide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OpenLogFile2Wide

//******************************************************************************

HRESULT
GetSystemVersiolwalues
(
    PULONG              PlatformId,
    PULONG              Win32Major,
    PULONG              Win32Minor,
    PULONG              KdMajor,
    PULONG              KdMinor
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(PlatformId != NULL);
    assert(Win32Major != NULL);
    assert(Win32Minor != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemVersiolwalues control method
        hResult = pDbgControl4->GetSystemVersiolwalues(PlatformId, Win32Major, Win32Minor, KdMajor, KdMinor);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemVersiolwalues %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemVersiolwalues
            {
                dPrintf("%s GetSystemVersiolwalues %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemVersiolwalues

//******************************************************************************

HRESULT
GetSystemVersionString
(
    ULONG               Which,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemVersionString control method
        hResult = pDbgControl4->GetSystemVersionString(Which, Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemVersionString %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemVersionString
            {
                dPrintf("%s GetSystemVersionString %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemVersionString

//******************************************************************************

HRESULT
GetSystemVersionStringWide
(
    ULONG               Which,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemVersionStringWide control method
        hResult = pDbgControl4->GetSystemVersionStringWide(Which, Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemVersionStringWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemVersionStringWide
            {
                dPrintf("%s GetSystemVersionStringWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemVersionStringWide

//******************************************************************************

HRESULT
GetContextStackTrace
(
    PVOID               StartContext,
    ULONG               StartContextSize,
    PDEBUG_STACK_FRAME  Frames,
    ULONG               FramesSize,
    PVOID               FrameContexts,
    ULONG               FrameContextsSize,
    ULONG               FrameContextsEntrySize,
    PULONG              FramesFilled
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetContextStackTrace control method
        hResult = pDbgControl4->GetContextStackTrace(StartContext, StartContextSize, Frames, FramesSize, FrameContexts, FrameContextsSize, FrameContextsEntrySize, FramesFilled);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetContextStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetContextStackTrace
            {
                dPrintf("%s GetContextStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetContextStackTrace

//******************************************************************************

HRESULT
OutputContextStackTrace
(
    ULONG               OutputControl,
    PDEBUG_STACK_FRAME  Frames,
    ULONG               FramesSize,
    PVOID               FrameContexts,
    ULONG               FrameContextsSize,
    ULONG               FrameContextsEntrySize,
    ULONG               Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Frames != NULL);
    assert(FrameContexts != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputContextStackTrace control method
        hResult = pDbgControl4->OutputContextStackTrace(OutputControl, Frames, FramesSize, FrameContexts, FrameContextsSize, FrameContextsEntrySize, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputContextStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputContextStackTrace
            {
                dPrintf("%s OutputContextStackTrace %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputContextStackTrace

//******************************************************************************

HRESULT
GetStoredEventInformation
(
    PULONG              Type,
    PULONG              ProcessId,
    PULONG              ThreadId,
    PVOID               Context,
    ULONG               ContextSize,
    PULONG              ContextUsed,
    PVOID               ExtraInformation,
    ULONG               ExtraInformationSize,
    PULONG              ExtraInformationUsed
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);
    assert(ProcessId != NULL);
    assert(ThreadId != NULL);

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetStoredEventInformation control method
        hResult = pDbgControl4->GetStoredEventInformation(Type, ProcessId, ThreadId, Context, ContextSize, ContextUsed, ExtraInformation, ExtraInformationSize, ExtraInformationUsed);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetStoredEventInformation %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetStoredEventInformation
            {
                dPrintf("%s GetStoredEventInformation %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetStoredEventInformation

//******************************************************************************

HRESULT
GetManagedStatus
(
    PULONG              Flags,
    ULONG               WhichString,
    PSTR                String,
    ULONG               StringSize,
    PULONG              StringNeeded
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetManagedStatus control method
        hResult = pDbgControl4->GetManagedStatus(Flags, WhichString, String, StringSize, StringNeeded);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetManagedStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetManagedStatus
            {
                dPrintf("%s GetManagedStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetManagedStatus

//******************************************************************************

HRESULT
GetManagedStatusWide
(
    PULONG              Flags,
    ULONG               WhichString,
    PWSTR               String,
    ULONG               StringSize,
    PULONG              StringNeeded
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetManagedStatusWide control method
        hResult = pDbgControl4->GetManagedStatusWide(Flags, WhichString, String, StringSize, StringNeeded);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetManagedStatusWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetManagedStatusWide
            {
                dPrintf("%s GetManagedStatusWide %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetManagedStatusWide

//******************************************************************************

HRESULT
ResetManagedStatus
(
    ULONG               Flags
)
{
    PDEBUG_CONTROL4     pDbgControl4 = debugControl4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug control interface
    if (pDbgControl4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ResetManagedStatus control method
        hResult = pDbgControl4->ResetManagedStatus(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng control interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CONTROL))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ResetManagedStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ResetManagedStatus
            {
                dPrintf("%s ResetManagedStatus %s = 0x%08x\n", DML(bold("DbgControl:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ResetManagedStatus

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
