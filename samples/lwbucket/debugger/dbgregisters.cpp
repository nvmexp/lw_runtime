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
|*  Module: dbgregisters.cpp                                                  *|
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
// Debugger Registers Interface wrappers
//
//******************************************************************************

HRESULT
GetNumberRegisters
(
    PULONG              Number
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberRegisters registers method
        hResult = pDbgRegisters->GetNumberRegisters(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberRegisters %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberRegisters
            {
                dPrintf("%s GetNumberRegisters %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberRegisters

//******************************************************************************

HRESULT
GetDescription
(
    ULONG               Register,
    PSTR                NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PDEBUG_REGISTER_DESCRIPTION Desc
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDescription registers method
        hResult = pDbgRegisters->GetDescription(Register, NameBuffer, NameBufferSize, NameSize, Desc);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDescription %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDescription
            {
                dPrintf("%s GetDescription %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDescription

//******************************************************************************

HRESULT
GetIndexByName
(
    PCSTR               Name,
    PULONG              Index
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(Index != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetIndexByName registers method
        hResult = pDbgRegisters->GetIndexByName(Name, Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetIndexByName %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetIndexByName
            {
                dPrintf("%s GetIndexByName %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetIndexByName

//******************************************************************************

HRESULT
GetValue
(
    ULONG               Register,
    PDEBUG_VALUE        Value
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Value != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetValue registers method
        hResult = pDbgRegisters->GetValue(Register, Value);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetValue %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetValue
            {
                dPrintf("%s GetValue %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetValue

//******************************************************************************

HRESULT
SetValue
(
    ULONG               Register,
    PDEBUG_VALUE        Value
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Value != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetValue registers method
        hResult = pDbgRegisters->SetValue(Register, Value);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetValue %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetValue
            {
                dPrintf("%s SetValue %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetValue

//******************************************************************************

HRESULT
GetValues
(
    ULONG               Count,
    PULONG              Indices,
    ULONG               Start,
    PDEBUG_VALUE        Values
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Values != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetValues registers method
        hResult = pDbgRegisters->GetValues(Count, Indices, Start, Values);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetValues
            {
                dPrintf("%s GetValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetValues

//******************************************************************************

HRESULT
SetValues
(
    ULONG               Count,
    PULONG              Indices,
    ULONG               Start,
    PDEBUG_VALUE        Values
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Values != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetValues registers method
        hResult = pDbgRegisters->SetValues(Count, Indices, Start, Values);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetValues
            {
                dPrintf("%s SetValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetValues

//******************************************************************************

HRESULT
OutputRegisters
(
    ULONG               OutputControl,
    ULONG               Flags
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputRegisters registers method
        hResult = pDbgRegisters->OutputRegisters(OutputControl, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputRegisters %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputRegisters
            {
                dPrintf("%s OutputRegisters %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputRegisters

//******************************************************************************

HRESULT
GetInstructionOffset
(
    PULONG64            Offset
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetInstructionOffset registers method
        hResult = pDbgRegisters->GetInstructionOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetInstructionOffset %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetInstructionOffset
            {
                dPrintf("%s GetInstructionOffset %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetInstructionOffset

//******************************************************************************

HRESULT
GetStackOffset
(
    PULONG64            Offset
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetStackOffset registers method
        hResult = pDbgRegisters->GetStackOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetStackOffset %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetStackOffset
            {
                dPrintf("%s GetStackOffset %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetStackOffset

//******************************************************************************

HRESULT
GetFrameOffset
(
    PULONG64            Offset
)
{
    PDEBUG_REGISTERS    pDbgRegisters = debugRegistersInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid registers interface
    if (pDbgRegisters != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFrameOffset registers method
        hResult = pDbgRegisters->GetFrameOffset(Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFrameOffset %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFrameOffset
            {
                dPrintf("%s GetFrameOffset %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFrameOffset

//******************************************************************************

HRESULT
GetDescriptionWide
(
    ULONG               Register,
    PWSTR               NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PDEBUG_REGISTER_DESCRIPTION Desc
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDescriptionWide registers method
        hResult = pDbgRegisters2->GetDescriptionWide(Register, NameBuffer, NameBufferSize, NameSize, Desc);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDescriptionWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDescriptionWide
            {
                dPrintf("%s GetDescriptionWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDescriptionWide

//******************************************************************************

HRESULT
GetIndexByNameWide
(
    PCWSTR              Name,
    PULONG              Index
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(Index != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetIndexByNameWide registers method
        hResult = pDbgRegisters2->GetIndexByNameWide(Name, Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetIndexByNameWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetIndexByNameWide
            {
                dPrintf("%s GetIndexByNameWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetIndexByNameWide

//******************************************************************************

HRESULT
GetNumberPseudoRegisters
(
    PULONG              Number
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberPseudoRegisters registers method
        hResult = pDbgRegisters2->GetNumberPseudoRegisters(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberPseudoRegisters %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberPseudoRegisters
            {
                dPrintf("%s GetNumberPseudoRegisters %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberPseudoRegisters

//******************************************************************************

HRESULT
GetPseudoDescription
(
    ULONG               Register,
    PSTR                NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PULONG64            TypeModule,
    PULONG              TypeId
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPseudoDescription registers method
        hResult = pDbgRegisters2->GetPseudoDescription(Register, NameBuffer, NameBufferSize, NameSize, TypeModule, TypeId);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPseudoDescription %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPseudoDescription
            {
                dPrintf("%s GetPseudoDescription %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPseudoDescription

//******************************************************************************

HRESULT
GetPseudoDescriptionWide
(
    ULONG               Register,
    PWSTR               NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PULONG64            TypeModule,
    PULONG              TypeId
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPseudoDescriptionWide registers method
        hResult = pDbgRegisters2->GetPseudoDescriptionWide(Register, NameBuffer, NameBufferSize, NameSize, TypeModule, TypeId);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPseudoDescriptionWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPseudoDescriptionWide
            {
                dPrintf("%s GetPseudoDescriptionWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPseudoDescriptionWide

//******************************************************************************

HRESULT
GetPseudoIndexByName
(
    PCSTR               Name,
    PULONG              Index
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(Index != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPseudoIndexByName registers method
        hResult = pDbgRegisters2->GetPseudoIndexByName(Name, Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPseudoIndexByName %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPseudoIndexByName
            {
                dPrintf("%s GetPseudoIndexByName %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPseudoIndexByName

//******************************************************************************

HRESULT
GetPseudoIndexByNameWide
(
    PCWSTR              Name,
    PULONG              Index
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(Index != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPseudoIndexByNameWide registers method
        hResult = pDbgRegisters2->GetPseudoIndexByNameWide(Name, Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPseudoIndexByNameWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPseudoIndexByNameWide
            {
                dPrintf("%s GetPseudoIndexByNameWide %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPseudoIndexByNameWide

//******************************************************************************

HRESULT
GetPseudoValues
(
    ULONG               Source,
    ULONG               Count,
    PULONG              Indices,
    ULONG               Start,
    PDEBUG_VALUE        Values
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Values != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetPseudoValues registers method
        hResult = pDbgRegisters2->GetPseudoValues(Source, Count, Indices, Start, Values);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetPseudoValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetPseudoValues
            {
                dPrintf("%s GetPseudoValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetPseudoValues

//******************************************************************************

HRESULT
SetPseudoValues
(
    ULONG               Source,
    ULONG               Count,
    PULONG              Indices,
    ULONG               Start,
    PDEBUG_VALUE        Values
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Values != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetPseudoValues registers method
        hResult = pDbgRegisters2->SetPseudoValues(Source, Count, Indices, Start, Values);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetPseudoValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetPseudoValues
            {
                dPrintf("%s SetPseudoValues %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetPseudoValues

//******************************************************************************

HRESULT
GetValues2
(
    ULONG               Source,
    ULONG               Count,
    PULONG              Indices,
    ULONG               Start,
    PDEBUG_VALUE        Values
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Values != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetValues2 registers method
        hResult = pDbgRegisters2->GetValues2(Source, Count, Indices, Start, Values);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetValues2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetValues2
            {
                dPrintf("%s GetValues2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetValues2

//******************************************************************************

HRESULT
SetValues2
(
    ULONG               Source,
    ULONG               Count,
    PULONG              Indices,
    ULONG               Start,
    PDEBUG_VALUE        Values
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Values != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetValues2 registers method
        hResult = pDbgRegisters2->SetValues2(Source, Count, Indices, Start, Values);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetValues2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetValues2
            {
                dPrintf("%s SetValues2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetValues2

//******************************************************************************

HRESULT
OutputRegisters2
(
    ULONG               OutputControl,
    ULONG               Source,
    ULONG               Flags
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputRegisters2 registers method
        hResult = pDbgRegisters2->OutputRegisters2(OutputControl, Source, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputRegisters2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputRegisters2
            {
                dPrintf("%s OutputRegisters2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputRegisters2

//******************************************************************************

HRESULT
GetInstructionOffset2
(
    ULONG               Source,
    PULONG64            Offset
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetInstructionOffset2 registers method
        hResult = pDbgRegisters2->GetInstructionOffset2(Source, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetInstructionOffset2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetInstructionOffset2
            {
                dPrintf("%s GetInstructionOffset2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetInstructionOffset2

//******************************************************************************

HRESULT
GetStackOffset2
(
    ULONG               Source,
    PULONG64            Offset
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetStackOffset2 registers method
        hResult = pDbgRegisters2->GetStackOffset2(Source, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetStackOffset2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetStackOffset2
            {
                dPrintf("%s GetStackOffset2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetStackOffset2

//******************************************************************************

HRESULT
GetFrameOffset2
(
    ULONG               Source,
    PULONG64            Offset
)
{
    PDEBUG_REGISTERS2   pDbgRegisters2 = debugRegisters2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for valid registers interface
    if (pDbgRegisters2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFrameOffset2 registers method
        hResult = pDbgRegisters2->GetFrameOffset2(Source, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng registers interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_REGISTERS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFrameOffset2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFrameOffset2
            {
                dPrintf("%s GetFrameOffset2 %s = 0x%08x\n", DML(bold("DbgRegisters:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFrameOffset2

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
