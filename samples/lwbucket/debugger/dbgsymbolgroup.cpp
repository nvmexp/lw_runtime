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
|*  Module: dbgsymbolgroup.cpp                                                *|
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
// Debugger Symbol Group Interface wrappers
//
//******************************************************************************

HRESULT
GetNumberSymbols
(
    PULONG              Number
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberSymbols symbol group method
        hResult = pDbgSymbolGroup->GetNumberSymbols(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberSymbols %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberSymbols
            {
                dPrintf("%s GetNumberSymbols %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberSymbols

//******************************************************************************

HRESULT
AddSymbol
(
    PCSTR               Name,
    PULONG              Index
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(Index != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddSymbol symbol group method
        hResult = pDbgSymbolGroup->AddSymbol(Name, Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddSymbol %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddSymbol
            {
                dPrintf("%s AddSymbol %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddSymbol

//******************************************************************************

HRESULT
RemoveSymbolByName
(
    PCSTR               Name
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveSymbolByName symbol group method
        hResult = pDbgSymbolGroup->RemoveSymbolByName(Name);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveSymbolByName %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveSymbolByName
            {
                dPrintf("%s RemoveSymbolByName %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveSymbolByName

//******************************************************************************

HRESULT
RemoveSymbolByIndex
(
    ULONG               Index
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveSymbolByIndex symbol group method
        hResult = pDbgSymbolGroup->RemoveSymbolByIndex(Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveSymbolByIndex %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveSymbolByIndex
            {
                dPrintf("%s RemoveSymbolByIndex %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveSymbolByIndex

//******************************************************************************

HRESULT
GetSymbolName
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolName symbol group method
        hResult = pDbgSymbolGroup->GetSymbolName(Index, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolName %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolName
            {
                dPrintf("%s GetSymbolName %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolName

//******************************************************************************

HRESULT
GetSymbolParameters
(
    ULONG               Start,
    ULONG               Count,
    PDEBUG_SYMBOL_PARAMETERS Params
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Params != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolParameters symbol group method
        hResult = pDbgSymbolGroup->GetSymbolParameters(Start, Count, Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolParameters %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolParameters
            {
                dPrintf("%s GetSymbolParameters %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolParameters

//******************************************************************************

HRESULT
ExpandSymbol
(
    ULONG               Index,
    BOOL                Expand
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ExpandSymbol symbol group method
        hResult = pDbgSymbolGroup->ExpandSymbol(Index, Expand);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ExpandSymbol %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ExpandSymbol
            {
                dPrintf("%s ExpandSymbol %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ExpandSymbol

//******************************************************************************

HRESULT
OutputSymbols
(
    ULONG               OutputControl,
    ULONG               Flags,
    ULONG               Start,
    ULONG               Count
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputSymbols symbol group method
        hResult = pDbgSymbolGroup->OutputSymbols(OutputControl, Flags, Start, Count);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputSymbols %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputSymbols
            {
                dPrintf("%s OutputSymbols %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputSymbols

//******************************************************************************

HRESULT
WriteSymbol
(
    ULONG               Index,
    PCSTR               Value
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Value != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteSymbol symbol group method
        hResult = pDbgSymbolGroup->WriteSymbol(Index, Value);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteSymbol %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteSymbol
            {
                dPrintf("%s WriteSymbol %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteSymbol

//******************************************************************************

HRESULT
OutputAsType
(
    ULONG               Index,
    PCSTR               Type
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputAsType symbol group method
        hResult = pDbgSymbolGroup->OutputAsType(Index, Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputAsType %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputAsType
            {
                dPrintf("%s OutputAsType %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputAsType

//******************************************************************************

HRESULT
AddSymbolWide
(
    PCWSTR              Name,
    PULONG              Index
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(Index != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddSymbolWide symbol group method
        hResult = pDbgSymbolGroup->AddSymbolWide(Name, Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddSymbolWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddSymbolWide
            {
                dPrintf("%s AddSymbolWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddSymbolWide

//******************************************************************************

HRESULT
RemoveSymbolByNameWide
(
    PCWSTR              Name
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveSymbolByNameWide symbol group method
        hResult = pDbgSymbolGroup->RemoveSymbolByNameWide(Name);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveSymbolByNameWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveSymbolByNameWide
            {
                dPrintf("%s RemoveSymbolByNameWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveSymbolByNameWide

//******************************************************************************

HRESULT
GetSymbolNameWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolNameWide symbol group method
        hResult = pDbgSymbolGroup->GetSymbolNameWide(Index, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolNameWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolNameWide
            {
                dPrintf("%s GetSymbolNameWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolNameWide

//******************************************************************************

HRESULT
WriteSymbolWide
(
    ULONG               Index,
    PCWSTR              Value
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Value != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteSymbolWide symbol group method
        hResult = pDbgSymbolGroup->WriteSymbolWide(Index, Value);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteSymbolWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteSymbolWide
            {
                dPrintf("%s WriteSymbolWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteSymbolWide

//******************************************************************************

HRESULT
OutputAsTypeWide
(
    ULONG               Index,
    PCWSTR              Type
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputAsTypeWide symbol group method
        hResult = pDbgSymbolGroup->OutputAsTypeWide(Index, Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputAsTypeWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputAsTypeWide
            {
                dPrintf("%s OutputAsTypeWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputAsTypeWide

//******************************************************************************

HRESULT
GetSymbolTypeName
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolTypeName symbol group method
        hResult = pDbgSymbolGroup->GetSymbolTypeName(Index, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolTypeName %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolTypeName
            {
                dPrintf("%s GetSymbolTypeName %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolTypeName

//******************************************************************************

HRESULT
GetSymbolTypeNameWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolTypeNameWide symbol group method
        hResult = pDbgSymbolGroup->GetSymbolTypeNameWide(Index, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolTypeNameWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolTypeNameWide
            {
                dPrintf("%s GetSymbolTypeNameWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolTypeNameWide

//******************************************************************************

HRESULT
GetSymbolSize
(
    ULONG               Index,
    PULONG              Size
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Size != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolSize symbol group method
        hResult = pDbgSymbolGroup->GetSymbolSize(Index, Size);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolSize %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolSize
            {
                dPrintf("%s GetSymbolSize %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolSize

//******************************************************************************

HRESULT
GetSymbolOffset
(
    ULONG               Index,
    PULONG64            Offset
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Offset != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolOffset symbol group method
        hResult = pDbgSymbolGroup->GetSymbolOffset(Index, Offset);
        if (SUCCEEDED(hResult))
        {
            // Make sure offset is the correct size for target system (32/64 bit)
            *Offset = TARGET(*Offset);
        }
        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolOffset %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolOffset
            {
                dPrintf("%s GetSymbolOffset %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolOffset

//******************************************************************************

HRESULT
GetSymbolRegister
(
    ULONG               Index,
    PULONG              Register
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Register != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolRegister symbol group method
        hResult = pDbgSymbolGroup->GetSymbolRegister(Index, Register);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolRegister %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolRegister
            {
                dPrintf("%s GetSymbolRegister %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolRegister

//******************************************************************************

HRESULT
GetSymbolValueText
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolValueText symbol group method
        hResult = pDbgSymbolGroup->GetSymbolValueText(Index, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolValueText %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolValueText
            {
                dPrintf("%s GetSymbolValueText %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolValueText

//******************************************************************************

HRESULT
GetSymbolValueTextWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolValueTextWide symbol group method
        hResult = pDbgSymbolGroup->GetSymbolValueTextWide(Index, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolValueTextWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolValueTextWide
            {
                dPrintf("%s GetSymbolValueTextWide %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolValueTextWide

//******************************************************************************

HRESULT
GetSymbolEntryInformation
(
    ULONG               Index,
    PDEBUG_SYMBOL_ENTRY Entry
)
{
    PDEBUG_SYMBOL_GROUP2 pDbgSymbolGroup = debugSymbolGroupInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Entry != NULL);

    // Check for extension initialized and valid symbol group interface
    if (isInitialized() && (pDbgSymbolGroup != NULL))
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntryInformation symbol group method
        hResult = pDbgSymbolGroup->GetSymbolEntryInformation(Index, Entry);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbol group interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOL_GROUP))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntryInformation %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntryInformation
            {
                dPrintf("%s GetSymbolEntryInformation %s = 0x%08x\n", DML(bold("DbgSymbolGroup:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntryInfo

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
