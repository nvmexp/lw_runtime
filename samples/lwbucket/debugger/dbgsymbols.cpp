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
|*  Module: dbgsymbols.cpp                                                    *|
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
// Debugger Symbols Interface wrappers
//
//******************************************************************************

HRESULT
GetSymbolOptions
(
    PULONG              Options
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolOptions symbols method
        hResult = pDbgSymbols->GetSymbolOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolOptions
            {
                dPrintf("%s GetSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolOptions

//******************************************************************************

HRESULT
AddSymbolOptions
(
    ULONG               Options
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddSymbolOptions symbols method
        hResult = pDbgSymbols->AddSymbolOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddSymbolOptions
            {
                dPrintf("%s AddSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddSymbolOptions

//******************************************************************************

HRESULT
RemoveSymbolOptions
(
    ULONG               Options
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveSymbolOptions symbols method
        hResult = pDbgSymbols->RemoveSymbolOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveSymbolOptions
            {
                dPrintf("%s RemoveSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveSymbolOptions

//******************************************************************************

HRESULT
SetSymbolOptions
(
    ULONG               Options
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSymbolOptions symbols method
        hResult = pDbgSymbols->SetSymbolOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSymbolOptions
            {
                dPrintf("%s SetSymbolOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSymbolOptions

//******************************************************************************

HRESULT
GetNameByOffset
(
    ULONG64             Offset,
    PSTR                NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PULONG64            Displacement
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNameByOffset symbols method
        hResult = pDbgSymbols->GetNameByOffset(Offset, NameBuffer, NameBufferSize, NameSize, Displacement);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNameByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNameByOffset
            {
                dPrintf("%s GetNameByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNameByOffset

//******************************************************************************

HRESULT
GetOffsetByName
(
    PCSTR               Symbol,
    PULONG64            Offset
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);
    assert(Offset != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetByName symbols method
        hResult = pDbgSymbols->GetOffsetByName(Symbol, Offset);
        if (SUCCEEDED(hResult))
        {
            // Make sure offset is the correct size for target system (32/64 bit)
            *Offset = TARGET(*Offset);
        }
        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetByName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetByName
            {
                dPrintf("%s GetOffsetByName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetByName

//******************************************************************************

HRESULT
GetNearNameByOffset
(
    ULONG64             Offset,
    LONG                Delta,
    PSTR                NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PULONG64            Displacement
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNearNameByOffset symbols method
        hResult = pDbgSymbols->GetNearNameByOffset(Offset, Delta, NameBuffer, NameBufferSize, NameSize, Displacement);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNearNameByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNearNameByOffset
            {
                dPrintf("%s GetNearNameByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNearNameByOffset

//******************************************************************************

HRESULT
GetLineByOffset
(
    ULONG64             Offset,
    PULONG              Line,
    PSTR                FileBuffer,
    ULONG               FileBufferSize,
    PULONG              FileSize,
    PULONG64            Displacement
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLineByOffset symbols method
        hResult = pDbgSymbols->GetLineByOffset(Offset, Line, FileBuffer, FileBufferSize, FileSize, Displacement);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLineByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLineByOffset
            {
                dPrintf("%s GetLineByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLineByOffset

//******************************************************************************

HRESULT
GetOffsetByLine
(
    ULONG               Line,
    PCSTR               File,
    PULONG64            Offset
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);
    assert(Offset != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetByLine symbols method
        hResult = pDbgSymbols->GetOffsetByLine(Line, File, Offset);
        if (SUCCEEDED(hResult))
        {
            // Make sure offset is the correct size for target system (32/64 bit)
            *Offset = TARGET(*Offset);
        }
        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetByLine %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetByLine
            {
                dPrintf("%s GetOffsetByLine %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetByLine

//******************************************************************************

HRESULT
GetNumberModules
(
    PULONG              Loaded,
    PULONG              Unloaded
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Loaded != NULL);
    assert(Unloaded != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberModules symbols method
        hResult = pDbgSymbols->GetNumberModules(Loaded, Unloaded);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberModules %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberModules
            {
                dPrintf("%s GetNumberModules %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberModules

//******************************************************************************

HRESULT
GetModuleByIndex
(
    ULONG               Index,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Base != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleByIndex symbols method
        hResult = pDbgSymbols->GetModuleByIndex(Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleByIndex %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleByIndex
            {
                dPrintf("%s GetModuleByIndex %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleByIndex

//******************************************************************************

HRESULT
GetModuleByModuleName
(
    PCSTR               Name,
    ULONG               StartIndex,
    PULONG              Index,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleByModuleName symbols method
        hResult = pDbgSymbols->GetModuleByModuleName(Name, StartIndex, Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleByModuleName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleByModuleName
            {
                dPrintf("%s GetModuleByModuleName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleByModuleName

//******************************************************************************

HRESULT
GetModuleByOffset
(
    ULONG64             Offset,
    ULONG               StartIndex,
    PULONG              Index,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleByOffset symbols method
        hResult = pDbgSymbols->GetModuleByOffset(Offset, StartIndex, Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleByOffset
            {
                dPrintf("%s GetModuleByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleByOffset

//******************************************************************************

HRESULT
GetModuleNames
(
    ULONG               Index,
    ULONG64             Base,
    PSTR                ImageNameBuffer,
    ULONG               ImageNameBufferSize,
    PULONG              ImageNameSize,
    PSTR                ModuleNameBuffer,
    ULONG               ModuleNameBufferSize,
    PULONG              ModuleNameSize,
    PSTR                LoadedImageNameBuffer,
    ULONG               LoadedImageNameBufferSize,
    PULONG              LoadedImageNameSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleNames symbols method
        hResult = pDbgSymbols->GetModuleNames(Index, Base, ImageNameBuffer, ImageNameBufferSize, ImageNameSize, ModuleNameBuffer, ModuleNameBufferSize, ModuleNameSize, LoadedImageNameBuffer, LoadedImageNameBufferSize, LoadedImageNameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleNames %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleNames
            {
                dPrintf("%s GetModuleNames %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleNames

//******************************************************************************

HRESULT
GetModuleParameters
(
    ULONG               Count,
    PULONG64            Bases,
    ULONG               Start,
    PDEBUG_MODULE_PARAMETERS Params
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Params != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleParameters symbols method
        hResult = pDbgSymbols->GetModuleParameters(Count, Bases, Start, Params);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleParameters %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleParameters
            {
                dPrintf("%s GetModuleParameters %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleParameters

//******************************************************************************

HRESULT
GetSymbolModule
(
    PCSTR               Symbol,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);
    assert(Base != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolModule symbols method
        hResult = pDbgSymbols->GetSymbolModule(Symbol, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolModule
            {
                dPrintf("%s GetSymbolModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolModule

//******************************************************************************

HRESULT
GetTypeName
(
    ULONG64             Module,
    ULONG               TypeId,
    PSTR                NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTypeName symbols method
        hResult = pDbgSymbols->GetTypeName(Module, TypeId, NameBuffer, NameBufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTypeName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTypeName
            {
                dPrintf("%s GetTypeName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTypeName

//******************************************************************************

HRESULT
GetTypeId
(
    ULONG64             Module,
    PCSTR               Name,
    PULONG              TypeId
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(TypeId != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTypeId symbols method
        hResult = pDbgSymbols->GetTypeId(Module, Name, TypeId);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTypeId %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTypeId
            {
                dPrintf("%s GetTypeId %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTypeId

//******************************************************************************

HRESULT
GetTypeSize
(
    ULONG64             Module,
    ULONG               TypeId,
    PULONG              Size
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Size != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTypeSize symbols method
        hResult = pDbgSymbols->GetTypeSize(Module, TypeId, Size);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTypeSize %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTypeSize
            {
                dPrintf("%s GetTypeSize %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTypeSize

//******************************************************************************

HRESULT
GetFieldOffset
(
    ULONG64             Module,
    ULONG               TypeId,
    PCSTR               Field,
    PULONG              Offset
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Field != NULL);
    assert(Offset != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFieldOffset symbols method
        hResult = pDbgSymbols->GetFieldOffset(Module, TypeId, Field, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFieldOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFieldOffset
            {
                dPrintf("%s GetFieldOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFieldOffset

//******************************************************************************

HRESULT
GetSymbolTypeId
(
    PCSTR               Symbol,
    PULONG              TypeId,
    PULONG64            Module
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);
    assert(TypeId != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolTypeId symbols method
        hResult = pDbgSymbols->GetSymbolTypeId(Symbol, TypeId, Module);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolTypeId %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolTypeId
            {
                dPrintf("%s GetSymbolTypeId %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolTypeId

//******************************************************************************

HRESULT
GetOffsetTypeId
(
    ULONG64             Offset,
    PULONG              TypeId,
    PULONG64            Module
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(TypeId != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetTypeId symbols method
        hResult = pDbgSymbols->GetOffsetTypeId(Offset, TypeId, Module);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetTypeId %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetTypeId
            {
                dPrintf("%s GetOffsetTypeId %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetTypeId

//******************************************************************************

HRESULT
ReadTypedDataVirtual
(
    ULONG64             Offset,
    ULONG64             Module,
    ULONG               TypeId,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadTypedDataVirtual symbols method
        hResult = pDbgSymbols->ReadTypedDataVirtual(Offset, Module, TypeId, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng virtual read interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadTypedDataVirtual %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadTypedDataVirtual
            {
                dPrintf("%s ReadTypedDataVirtual %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadTypedDataVirtual

//******************************************************************************

HRESULT
WriteTypedDataVirtual
(
    ULONG64             Offset,
    ULONG64             Module,
    ULONG               TypeId,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteTypedDataVirtual symbols method
        hResult = pDbgSymbols->WriteTypedDataVirtual(Offset, Module, TypeId, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng virtual write interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_VIRTUAL_WRITE))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteTypedDataVirtual %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteTypedDataVirtual
            {
                dPrintf("%s WriteTypedDataVirtual %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteTypedDataVirtual

//******************************************************************************

HRESULT
OutputTypedDataVirtual
(
    ULONG               OutputControl,
    ULONG64             Offset,
    ULONG64             Module,
    ULONG               TypeId,
    ULONG               Flags
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputTypedDataVirtual symbols method
        hResult = pDbgSymbols->OutputTypedDataVirtual(OutputControl, Offset, Module, TypeId, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputTypedDataVirtual %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputTypedDataVirtual
            {
                dPrintf("%s OutputTypedDataVirtual %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputTypedDataVirtual

//******************************************************************************

HRESULT
ReadTypedDataPhysical
(
    ULONG64             Offset,
    ULONG64             Module,
    ULONG               TypeId,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesRead
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReadTypedDataPhysical symbols method
        hResult = pDbgSymbols->ReadTypedDataPhysical(Offset, Module, TypeId, Buffer, BufferSize, BytesRead);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng physical read interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_PHYSICAL_READ))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReadTypedDataPhysical %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReadTypedDataPhysical
            {
                dPrintf("%s ReadTypedDataPhysical %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReadTypedDataPhysical

//******************************************************************************

HRESULT
WriteTypedDataPhysical
(
    ULONG64             Offset,
    ULONG64             Module,
    ULONG               TypeId,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BytesWritten
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Buffer != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteTypedDataPhysical symbols method
        hResult = pDbgSymbols->WriteTypedDataPhysical(Offset, Module, TypeId, Buffer, BufferSize, BytesWritten);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteTypedDataPhysical %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteTypedDataPhysical
            {
                dPrintf("%s WriteTypedDataPhysical %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteTypedDataPhysical

//******************************************************************************

HRESULT
OutputTypedDataPhysical
(
    ULONG               OutputControl,
    ULONG64             Offset,
    ULONG64             Module,
    ULONG               TypeId,
    ULONG               Flags
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputTypedDataPhysical symbols method
        hResult = pDbgSymbols->OutputTypedDataPhysical(OutputControl, Offset, Module, TypeId, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputTypedDataPhysical %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputTypedDataPhysical
            {
                dPrintf("%s OutputTypedDataPhysical %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputTypedDataPhysical

//******************************************************************************

HRESULT
GetScope
(
    PULONG64            InstructionOffset,
    PDEBUG_STACK_FRAME  ScopeFrame,
    PVOID               ScopeContext,
    ULONG               ScopeContextSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetScope symbols method
        hResult = pDbgSymbols->GetScope(InstructionOffset, ScopeFrame, ScopeContext, ScopeContextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetScope %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetScope
            {
                dPrintf("%s GetScope %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetScope

//******************************************************************************

HRESULT
SetScope
(
    ULONG64             InstructionOffset,
    PDEBUG_STACK_FRAME  ScopeFrame,
    PVOID               ScopeContext,
    ULONG               ScopeContextSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetScope symbols method
        hResult = pDbgSymbols->SetScope(InstructionOffset, ScopeFrame, ScopeContext, ScopeContextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetScope %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetScope
            {
                dPrintf("%s SetScope %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetScope

//******************************************************************************

HRESULT
ResetScope()
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ResetScope symbols method
        hResult = pDbgSymbols->ResetScope();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ResetScope %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ResetScope
            {
                dPrintf("%s ResetScope %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ResetScope

//******************************************************************************

HRESULT
GetScopeSymbolGroup
(
    ULONG               Flags,
    PDEBUG_SYMBOL_GROUP Update,
    PDEBUG_SYMBOL_GROUP* Symbols
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbols != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetScopeSymbolGroup symbols method
        hResult = pDbgSymbols->GetScopeSymbolGroup(Flags, Update, Symbols);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetScopeSymbolGroup %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetScopeSymbolGroup
            {
                dPrintf("%s GetScopeSymbolGroup %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetScopeSymbolGroup

//******************************************************************************

HRESULT
CreateSymbolGroup
(
    PDEBUG_SYMBOL_GROUP* Group
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Group != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateSymbolGroup symbols method
        hResult = pDbgSymbols->CreateSymbolGroup(Group);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateSymbolGroup %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateSymbolGroup
            {
                dPrintf("%s CreateSymbolGroup %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateSymbolGroup

//******************************************************************************

HRESULT
StartSymbolMatch
(
    PCSTR               Pattern,
    PULONG64            Handle
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Pattern != NULL);
    assert(Handle != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the StartSymbolMatch symbols method
        hResult = pDbgSymbols->StartSymbolMatch(Pattern, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s StartSymbolMatch %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed StartSymbolMatch
            {
                dPrintf("%s StartSymbolMatch %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // StartSymbolMatch

//******************************************************************************

HRESULT
GetNextSymbolMatch
(
    ULONG64             Handle,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              MatchSize,
    PULONG64            Offset
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNextSymbolMatch symbols method
        hResult = pDbgSymbols->GetNextSymbolMatch(Handle, Buffer, BufferSize, MatchSize, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNextSymbolMatch %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNextSymbolMatch
            {
                dPrintf("%s GetNextSymbolMatch %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNextSymbolMatch

//******************************************************************************

HRESULT
EndSymbolMatch
(
    ULONG64             Handle
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the EndSymbolMatch symbols method
        hResult = pDbgSymbols->EndSymbolMatch(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s EndSymbolMatch %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed EndSymbolMatch
            {
                dPrintf("%s EndSymbolMatch %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // EndSymbolMatch

//******************************************************************************

HRESULT
Reload
(
    PCSTR               Module
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Module != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the Reload symbols method
        hResult = pDbgSymbols->Reload(Module);

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Reload %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Reload
            {
                dPrintf("%s Reload %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Reload

//******************************************************************************

HRESULT
GetSymbolPath
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              PathSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolPath symbols method
        hResult = pDbgSymbols->GetSymbolPath(Buffer, BufferSize, PathSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolPath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolPath
            {
                dPrintf("%s GetSymbolPath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolPath

//******************************************************************************

HRESULT
SetSymbolPath
(
    PCSTR               Path
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSymbolPath symbols method
        hResult = pDbgSymbols->SetSymbolPath(Path);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSymbolPath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSymbolPath
            {
                dPrintf("%s SetSymbolPath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSymbolPath

//******************************************************************************

HRESULT
AppendSymbolPath
(
    PCSTR               Addition
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Addition != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AppendSymbolPath symbols method
        hResult = pDbgSymbols->AppendSymbolPath(Addition);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AppendSymbolPath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AppendSymbolPath
            {
                dPrintf("%s AppendSymbolPath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AppendSymbolPath

//******************************************************************************

HRESULT
GetImagePath
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              PathSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetImagePath symbols method
        hResult = pDbgSymbols->GetImagePath(Buffer, BufferSize, PathSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetImagePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetImagePath
            {
                dPrintf("%s GetImagePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetImagePath

//******************************************************************************

HRESULT
SetImagePath
(
    PCSTR               Path
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetImagePath symbols method
        hResult = pDbgSymbols->SetImagePath(Path);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetImagePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetImagePath
            {
                dPrintf("%s SetImagePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetImagePath

//******************************************************************************

HRESULT
AppendImagePath
(
    PCSTR               Addition
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Addition != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AppendImagePath symbols method
        hResult = pDbgSymbols->AppendImagePath(Addition);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AppendImagePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AppendImagePath
            {
                dPrintf("%s AppendImagePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AppendImagePath

//******************************************************************************

HRESULT
GetSourcePath
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              PathSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourcePath symbols method
        hResult = pDbgSymbols->GetSourcePath(Buffer, BufferSize, PathSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourcePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourcePath
            {
                dPrintf("%s GetSourcePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourcePath

//******************************************************************************

HRESULT
GetSourcePathElement
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              ElementSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourcePathElement symbols method
        hResult = pDbgSymbols->GetSourcePathElement(Index, Buffer, BufferSize, ElementSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourcePathElement %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourcePathElement
            {
                dPrintf("%s GetSourcePathElement %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourcePathElement

//******************************************************************************

HRESULT
SetSourcePath
(
    PCSTR               Path
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSourcePath symbols method
        hResult = pDbgSymbols->SetSourcePath(Path);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSourcePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSourcePath
            {
                dPrintf("%s SetSourcePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSourcePath

//******************************************************************************

HRESULT
AppendSourcePath
(
    PCSTR               Addition
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Addition != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AppendSourcePath symbols method
        hResult = pDbgSymbols->AppendSourcePath(Addition);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AppendSourcePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AppendSourcePath
            {
                dPrintf("%s AppendSourcePath %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AppendSourcePath

//******************************************************************************

HRESULT
FindSourceFile
(
    ULONG               StartElement,
    PCSTR               File,
    ULONG               Flags,
    PULONG              FoundElement,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              FoundSize
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the FindSourceFile symbols method
        hResult = pDbgSymbols->FindSourceFile(StartElement, File, Flags, FoundElement, Buffer, BufferSize, FoundSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s FindSourceFile %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed FindSourceFile
            {
                dPrintf("%s FindSourceFile %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // FindSourceFile

//******************************************************************************

HRESULT
GetSourceFileLineOffsets
(
    PCSTR               File,
    PULONG64            Buffer,
    ULONG               BufferLines,
    PULONG              FileLines
)
{
    PDEBUG_SYMBOLS      pDbgSymbols = debugSymbolsInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceFileLineOffsets symbols method
        hResult = pDbgSymbols->GetSourceFileLineOffsets(File, Buffer, BufferLines, FileLines);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceFileLineOffsets %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceFileLineOffsets
            {
                dPrintf("%s GetSourceFileLineOffsets %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceFileLineOffsets

//******************************************************************************

HRESULT
GetModuleVersionInformation
(
    ULONG               Index,
    ULONG64             Base,
    PCSTR               Item,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              VerInfoSize
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Item != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleVersionInformation symbols method
        hResult = pDbgSymbols2->GetModuleVersionInformation(Index, Base, Item, Buffer, BufferSize, VerInfoSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleVersionInformation %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleVersionInformation
            {
                dPrintf("%s GetModuleVersionInformation %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleVersionInformation

//******************************************************************************

HRESULT
GetModuleNameString
(
    ULONG               Which,
    ULONG               Index,
    ULONG64             Base,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleNameString symbols method
        hResult = pDbgSymbols2->GetModuleNameString(Which, Index, Base, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleNameString %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleNameString
            {
                dPrintf("%s GetModuleNameString %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleNameString

//******************************************************************************

HRESULT
GetConstantName
(
    ULONG64             Module,
    ULONG               TypeId,
    ULONG64             Value,
    PSTR                NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetConstantName symbols method
        hResult = pDbgSymbols2->GetConstantName(Module, TypeId, Value, NameBuffer, NameBufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetConstantName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetConstantName
            {
                dPrintf("%s GetConstantName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetConstantName

//******************************************************************************

HRESULT
GetFieldName
(
    ULONG64             Module,
    ULONG               TypeId,
    ULONG               FieldIndex,
    PSTR                NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFieldName symbols method
        hResult = pDbgSymbols2->GetFieldName(Module, TypeId, FieldIndex, NameBuffer, NameBufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFieldName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFieldName
            {
                dPrintf("%s GetFieldName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFieldName

//******************************************************************************

HRESULT
GetTypeOptions
(
    PULONG              Options
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTypeOptions symbols method
        hResult = pDbgSymbols2->GetTypeOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTypeOptions
            {
                dPrintf("%s GetTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTypeOptions

//******************************************************************************

HRESULT
AddTypeOptions
(
    ULONG               Options
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddTypeOptions symbols method
        hResult = pDbgSymbols2->AddTypeOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddTypeOptions
            {
                dPrintf("%s AddTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddTypeOptions

//******************************************************************************

HRESULT
RemoveTypeOptions
(
    ULONG               Options
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveTypeOptions symbols method
        hResult = pDbgSymbols2->RemoveTypeOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveTypeOptions
            {
                dPrintf("%s RemoveTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveTypeOptions

//******************************************************************************

HRESULT
SetTypeOptions
(
    ULONG               Options
)
{
    PDEBUG_SYMBOLS2     pDbgSymbols2 = debugSymbols2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetTypeOptions symbols method
        hResult = pDbgSymbols2->SetTypeOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetTypeOptions
            {
                dPrintf("%s SetTypeOptions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetTypeOptions

//******************************************************************************

HRESULT
GetNameByOffsetWide
(
    ULONG64             Offset,
    PWSTR               NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PULONG64            Displacement
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNameByOffsetWide symbols method
        hResult = pDbgSymbols3->GetNameByOffsetWide(Offset, NameBuffer, NameBufferSize, NameSize, Displacement);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNameByOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNameByOffsetWide
            {
                dPrintf("%s GetNameByOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNameByOffsetWide

//******************************************************************************

HRESULT
GetOffsetByNameWide
(
    PCWSTR              Symbol,
    PULONG64            Offset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);
    assert(Offset != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetByNameWide symbols method
        hResult = pDbgSymbols3->GetOffsetByNameWide(Symbol, Offset);
        if (SUCCEEDED(hResult))
        {
            // Make sure offset is the correct size for target system (32/64 bit)
            *Offset = TARGET(*Offset);
        }
        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetByNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetByNameWide
            {
                dPrintf("%s GetOffsetByNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetByNameWide

//******************************************************************************

HRESULT
GetNearNameByOffsetWide
(
    ULONG64             Offset,
    LONG                Delta,
    PWSTR               NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize,
    PULONG64            Displacement
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNearNameByOffsetWide symbols method
        hResult = pDbgSymbols3->GetNearNameByOffsetWide(Offset, Delta, NameBuffer, NameBufferSize, NameSize, Displacement);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNearNameByOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNearNameByOffsetWide
            {
                dPrintf("%s GetNearNameByOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNearNameByOffsetWide

//******************************************************************************

HRESULT
GetLineByOffsetWide
(
    ULONG64             Offset,
    PULONG              Line,
    PWSTR               FileBuffer,
    ULONG               FileBufferSize,
    PULONG              FileSize,
    PULONG64            Displacement
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLineByOffsetWide symbols method
        hResult = pDbgSymbols3->GetLineByOffsetWide(Offset, Line, FileBuffer, FileBufferSize, FileSize, Displacement);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLineByOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLineByOffsetWide
            {
                dPrintf("%s GetLineByOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLineByOffsetWide

//******************************************************************************

HRESULT
GetOffsetByLineWide
(
    ULONG               Line,
    PCWSTR              File,
    PULONG64            Offset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);
    assert(Offset != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOffsetByLineWide symbols method
        hResult = pDbgSymbols3->GetOffsetByLineWide(Line, File, Offset);
        if (SUCCEEDED(hResult))
        {
            // Make sure offset is the correct size for target system (32/64 bit)
            *Offset = TARGET(*Offset);
        }
        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOffsetByLineWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOffsetByLineWide
            {
                dPrintf("%s GetOffsetByLineWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOffsetByLineWide

//******************************************************************************

HRESULT
GetModuleByModuleNameWide
(
    PCWSTR              Name,
    ULONG               StartIndex,
    PULONG              Index,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleByModuleNameWide symbols method
        hResult = pDbgSymbols3->GetModuleByModuleNameWide(Name, StartIndex, Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleByModuleNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleByModuleNameWide
            {
                dPrintf("%s GetModuleByModuleNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleByModuleNameWide

//******************************************************************************

HRESULT
GetSymbolModuleWide
(
    PCWSTR              Symbol,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);
    assert(Base != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolModuleWide symbols method
        hResult = pDbgSymbols3->GetSymbolModuleWide(Symbol, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolModuleWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolModuleWide
            {
                dPrintf("%s GetSymbolModuleWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolModuleWide

//******************************************************************************

HRESULT
GetTypeNameWide
(
    ULONG64             Module,
    ULONG               TypeId,
    PWSTR               NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTypeNameWide symbols method
        hResult = pDbgSymbols3->GetTypeNameWide(Module, TypeId, NameBuffer, NameBufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTypeNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTypeNameWide
            {
                dPrintf("%s GetTypeNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTypeNameWide

//******************************************************************************

HRESULT
GetTypeIdWide
(
    ULONG64             Module,
    PCWSTR              Name,
    PULONG              TypeId
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);
    assert(TypeId != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetTypeIdWide symbols method
        hResult = pDbgSymbols3->GetTypeIdWide(Module, Name, TypeId);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetTypeIdWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetTypeIdWide
            {
                dPrintf("%s GetTypeIdWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetTypeIdWide

//******************************************************************************

HRESULT
GetFieldOffsetWide
(
    ULONG64             Module,
    ULONG               TypeId,
    PCWSTR              Field,
    PULONG              Offset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Field != NULL);
    assert(Offset != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFieldOffsetWide symbols method
        hResult = pDbgSymbols3->GetFieldOffsetWide(Module, TypeId, Field, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFieldOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFieldOffsetWide
            {
                dPrintf("%s GetFieldOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFieldOffsetWide

//******************************************************************************

HRESULT
GetSymbolTypeIdWide
(
    PCWSTR              Symbol,
    PULONG              TypeId,
    PULONG64            Module
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);
    assert(TypeId != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolTypeIdWide symbols method
        hResult = pDbgSymbols3->GetSymbolTypeIdWide(Symbol, TypeId, Module);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolTypeIdWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolTypeIdWide
            {
                dPrintf("%s GetSymbolTypeIdWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolTypeIdWide

//******************************************************************************

HRESULT
GetScopeSymbolGroup2
(
    ULONG               Flags,
    PDEBUG_SYMBOL_GROUP2 Update,
    PDEBUG_SYMBOL_GROUP2* Symbols
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbols != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetScopeSymbolGroup2 symbols method
        hResult = pDbgSymbols3->GetScopeSymbolGroup2(Flags, Update, Symbols);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetScopeSymbolGroup2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetScopeSymbolGroup2
            {
                dPrintf("%s GetScopeSymbolGroup2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetScopeSymbolGroup2

//******************************************************************************

HRESULT
CreateSymbolGroup2
(
    PDEBUG_SYMBOL_GROUP2* Group
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Group != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateSymbolGroup2 symbols method
        hResult = pDbgSymbols3->CreateSymbolGroup2(Group);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateSymbolGroup2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateSymbolGroup2
            {
                dPrintf("%s CreateSymbolGroup2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateSymbolGroup2

//******************************************************************************

HRESULT
StartSymbolMatchWide
(
    PCWSTR              Pattern,
    PULONG64            Handle
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Pattern != NULL);
    assert(Handle != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the StartSymbolMatchWide symbols method
        hResult = pDbgSymbols3->StartSymbolMatchWide(Pattern, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s StartSymbolMatchWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed StartSymbolMatchWide
            {
                dPrintf("%s StartSymbolMatchWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // StartSymbolMatchWide

//******************************************************************************

HRESULT
GetNextSymbolMatchWide
(
    ULONG64             Handle,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              MatchSize,
    PULONG64            Offset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNextSymbolMatchWide symbols method
        hResult = pDbgSymbols3->GetNextSymbolMatchWide(Handle, Buffer, BufferSize, MatchSize, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNextSymbolMatchWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNextSymbolMatchWide
            {
                dPrintf("%s GetNextSymbolMatchWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNextSymbolMatchWide

//******************************************************************************

HRESULT
ReloadWide
(
    PCWSTR              Module
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Module != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ReloadWide symbols method
        hResult = pDbgSymbols3->ReloadWide(Module);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ReloadWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ReloadWide
            {
                dPrintf("%s ReloadWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ReloadWide

//******************************************************************************

HRESULT
GetSymbolPathWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              PathSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolPathWide symbols method
        hResult = pDbgSymbols3->GetSymbolPathWide(Buffer, BufferSize, PathSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolPathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolPathWide
            {
                dPrintf("%s GetSymbolPathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolPathWide

//******************************************************************************

HRESULT
SetSymbolPathWide
(
    PCWSTR              Path
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSymbolPathWide symbols method
        hResult = pDbgSymbols3->SetSymbolPathWide(Path);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSymbolPathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSymbolPathWide
            {
                dPrintf("%s SetSymbolPathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSymbolPathWide

//******************************************************************************

HRESULT
AppendSymbolPathWide
(
    PCWSTR              Addition
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Addition != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AppendSymbolPathWide symbols method
        hResult = pDbgSymbols3->AppendSymbolPathWide(Addition);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AppendSymbolPathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AppendSymbolPathWide
            {
                dPrintf("%s AppendSymbolPathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AppendSymbolPathWide

//******************************************************************************

HRESULT
GetImagePathWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              PathSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetImagePathWide symbols method
        hResult = pDbgSymbols3->GetImagePathWide(Buffer, BufferSize, PathSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetImagePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetImagePathWide
            {
                dPrintf("%s GetImagePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetImagePathWide

//******************************************************************************

HRESULT
SetImagePathWide
(
    PCWSTR              Path
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetImagePathWide symbols method
        hResult = pDbgSymbols3->SetImagePathWide(Path);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetImagePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetImagePathWide
            {
                dPrintf("%s SetImagePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetImagePathWide

//******************************************************************************

HRESULT
AppendImagePathWide
(
    PCWSTR              Addition
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Addition != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AppendImagePathWide symbols method
        hResult = pDbgSymbols3->AppendImagePathWide(Addition);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AppendImagePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AppendImagePathWide
            {
                dPrintf("%s AppendImagePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AppendImagePathWide

//******************************************************************************

HRESULT
GetSourcePathWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              PathSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourcePathWide symbols method
        hResult = pDbgSymbols3->GetSourcePathWide(Buffer, BufferSize, PathSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourcePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourcePathWide
            {
                dPrintf("%s GetSourcePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourcePathWide

//******************************************************************************

HRESULT
GetSourcePathElementWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              ElementSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourcePathElementWide symbols method
        hResult = pDbgSymbols3->GetSourcePathElementWide(Index, Buffer, BufferSize, ElementSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourcePathElementWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourcePathElementWide
            {
                dPrintf("%s GetSourcePathElementWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourcePathElementWide

//******************************************************************************

HRESULT
SetSourcePathWide
(
    PCWSTR              Path
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Path != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetSourcePathWide symbols method
        hResult = pDbgSymbols3->SetSourcePathWide(Path);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetSourcePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetSourcePathWide
            {
                dPrintf("%s SetSourcePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetSourcePathWide

//******************************************************************************

HRESULT
AppendSourcePathWide
(
    PCWSTR              Addition
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Addition != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AppendSourcePathWide symbols method
        hResult = pDbgSymbols3->AppendSourcePathWide(Addition);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AppendSourcePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AppendSourcePathWide
            {
                dPrintf("%s AppendSourcePathWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AppendSourcePathWide

//******************************************************************************

HRESULT
FindSourceFileWide
(
    ULONG               StartElement,
    PCWSTR              File,
    ULONG               Flags,
    PULONG              FoundElement,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              FoundSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the FindSourceFileWide symbols method
        hResult = pDbgSymbols3->FindSourceFileWide(StartElement, File, Flags, FoundElement, Buffer, BufferSize, FoundSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s FindSourceFileWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed FindSourceFileWide
            {
                dPrintf("%s FindSourceFileWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // FindSourceFileWide

//******************************************************************************

HRESULT
GetSourceFileLineOffsetsWide
(
    PCWSTR              File,
    PULONG64            Buffer,
    ULONG               BufferLines,
    PULONG              FileLines
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceFileLineOffsetsWide symbols method
        hResult = pDbgSymbols3->GetSourceFileLineOffsetsWide(File, Buffer, BufferLines, FileLines);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceFileLineOffsetsWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceFileLineOffsetsWide
            {
                dPrintf("%s GetSourceFileLineOffsetsWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceFileLineOffsetsWide

//******************************************************************************

HRESULT
GetModuleVersionInformationWide
(
    ULONG               Index,
    ULONG64             Base,
    PCWSTR              Item,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              VerInfoSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Item != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleVersionInformationWide symbols method
        hResult = pDbgSymbols3->GetModuleVersionInformationWide(Index, Base, Item, Buffer, BufferSize, VerInfoSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleVersionInformationWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleVersionInformationWide
            {
                dPrintf("%s GetModuleVersionInformationWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleVersionInformationWide

//******************************************************************************

HRESULT
GetModuleNameStringWide
(
    ULONG               Which,
    ULONG               Index,
    ULONG64             Base,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleNameStringWide symbols method
        hResult = pDbgSymbols3->GetModuleNameStringWide(Which, Index, Base, Buffer, BufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleNameStringWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleNameStringWide
            {
                dPrintf("%s GetModuleNameStringWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleNameStringWide

//******************************************************************************

HRESULT
GetConstantNameWide
(
    ULONG64             Module,
    ULONG               TypeId,
    ULONG64             Value,
    PWSTR               NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetConstantNameWide symbols method
        hResult = pDbgSymbols3->GetConstantNameWide(Module, TypeId, Value, NameBuffer, NameBufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetConstantNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetConstantNameWide
            {
                dPrintf("%s GetConstantNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetConstantNameWide

//******************************************************************************

HRESULT
GetFieldNameWide
(
    ULONG64             Module,
    ULONG               TypeId,
    ULONG               FieldIndex,
    PWSTR               NameBuffer,
    ULONG               NameBufferSize,
    PULONG              NameSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFieldNameWide symbols method
        hResult = pDbgSymbols3->GetFieldNameWide(Module, TypeId, FieldIndex, NameBuffer, NameBufferSize, NameSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFieldNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFieldNameWide
            {
                dPrintf("%s GetFieldNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFieldNameWide

//******************************************************************************

HRESULT
IsManagedModule
(
    ULONG               Index,
    ULONG64             Base
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the IsManagedModule symbols method
        hResult = pDbgSymbols3->IsManagedModule(Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s IsManagedModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed IsManagedModule
            {
                dPrintf("%s IsManagedModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // IsManagedModule

//******************************************************************************

HRESULT
GetModuleByModuleName2
(
    PCSTR               Name,
    ULONG               StartIndex,
    ULONG               Flags,
    PULONG              Index,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleByModuleName2 symbols method
        hResult = pDbgSymbols3->GetModuleByModuleName2(Name, StartIndex, Flags, Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleByModuleName2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleByModuleName2
            {
                dPrintf("%s GetModuleByModuleName2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleByModuleName2

//******************************************************************************

HRESULT
GetModuleByModuleName2Wide
(
    PCWSTR              Name,
    ULONG               StartIndex,
    ULONG               Flags,
    PULONG              Index,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleByModuleName2Wide symbols method
        hResult = pDbgSymbols3->GetModuleByModuleName2Wide(Name, StartIndex, Flags, Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleByModuleName2Wide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleByModuleName2Wide
            {
                dPrintf("%s GetModuleByModuleName2Wide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleByModuleName2Wide

//******************************************************************************

HRESULT
GetModuleByOffset2
(
    ULONG64             Offset,
    ULONG               StartIndex,
    ULONG               Flags,
    PULONG              Index,
    PULONG64            Base
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetModuleByOffset2 symbols method
        hResult = pDbgSymbols3->GetModuleByOffset2(Offset, StartIndex, Flags, Index, Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetModuleByOffset2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetModuleByOffset2
            {
                dPrintf("%s GetModuleByOffset2 %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetModuleByOffset2

//******************************************************************************

HRESULT
AddSyntheticModule
(
    ULONG64             Base,
    ULONG               Size,
    PCSTR               ImagePath,
    PCSTR               ModuleName,
    ULONG               Flags
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(ImagePath != NULL);
    assert(ModuleName != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddSyntheticModule symbols method
        hResult = pDbgSymbols3->AddSyntheticModule(Base, Size, ImagePath, ModuleName, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddSyntheticModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddSyntheticModule
            {
                dPrintf("%s AddSyntheticModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddSyntheticModule

//******************************************************************************

HRESULT
AddSyntheticModuleWide
(
    ULONG64             Base,
    ULONG               Size,
    PCWSTR              ImagePath,
    PCWSTR              ModuleName,
    ULONG               Flags
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(ImagePath != NULL);
    assert(ModuleName != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddSyntheticModuleWide symbols method
        hResult = pDbgSymbols3->AddSyntheticModuleWide(Base, Size, ImagePath, ModuleName, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddSyntheticModuleWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddSyntheticModuleWide
            {
                dPrintf("%s AddSyntheticModuleWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddSyntheticModuleWide

//******************************************************************************

HRESULT
RemoveSyntheticModule
(
    ULONG64             Base
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveSyntheticModule symbols method
        hResult = pDbgSymbols3->RemoveSyntheticModule(Base);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveSyntheticModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveSyntheticModule
            {
                dPrintf("%s RemoveSyntheticModule %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveSyntheticModule

//******************************************************************************

HRESULT
GetLwrrentScopeFrameIndex
(
    PULONG              Index
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Index != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetLwrrentScopeFrameIndex symbols method
        hResult = pDbgSymbols3->GetLwrrentScopeFrameIndex(Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetLwrrentScopeFrameIndex %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetLwrrentScopeFrameIndex
            {
                dPrintf("%s GetLwrrentScopeFrameIndex %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetLwrrentScopeFrameIndex

//******************************************************************************

HRESULT
SetScopeFrameByIndex
(
    ULONG               Index
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetScopeFrameByIndex symbols method
        hResult = pDbgSymbols3->SetScopeFrameByIndex(Index);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetScopeFrameByIndex %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetScopeFrameByIndex
            {
                dPrintf("%s SetScopeFrameByIndex %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetScopeFrameByIndex

//******************************************************************************

HRESULT
SetScopeFromJitDebugInfo
(
    ULONG               OutputControl,
    ULONG64             InfoOffset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetScopeFromJitDebugInfo symbols method
        hResult = pDbgSymbols3->SetScopeFromJitDebugInfo(OutputControl, InfoOffset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetScopeFromJitDebugInfo %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetScopeFromJitDebugInfo
            {
                dPrintf("%s SetScopeFromJitDebugInfo %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetScopeFromJitDebugInfo

//******************************************************************************

HRESULT
SetScopeFromStoredEvent()
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetScopeFromStoredEvent symbols method
        hResult = pDbgSymbols3->SetScopeFromStoredEvent();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetScopeFromStoredEvent %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetScopeFromStoredEvent
            {
                dPrintf("%s SetScopeFromStoredEvent %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetScopeFromStoredEvent

//******************************************************************************

HRESULT
OutputSymbolByOffset
(
    ULONG               OutputControl,
    ULONG               Flags,
    ULONG64             Offset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputSymbolByOffset symbols method
        hResult = pDbgSymbols3->OutputSymbolByOffset(OutputControl, Flags, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputSymbolByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputSymbolByOffset
            {
                dPrintf("%s OutputSymbolByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputSymbolByOffset

//******************************************************************************

HRESULT
GetFunctionEntryByOffset
(
    ULONG64             Offset,
    ULONG               Flags,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              BufferNeeded
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFunctionEntryByOffset symbols method
        hResult = pDbgSymbols3->GetFunctionEntryByOffset(Offset, Flags, Buffer, BufferSize, BufferNeeded);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFunctionEntryByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFunctionEntryByOffset
            {
                dPrintf("%s GetFunctionEntryByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFunctionEntryByOffset

//******************************************************************************

HRESULT
GetFieldTypeAndOffset
(
    ULONG64             Module,
    ULONG               ContainerTypeId,
    PCSTR               Field,
    PULONG              FieldTypeId,
    PULONG              Offset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Field != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFieldTypeAndOffset symbols method
        hResult = pDbgSymbols3->GetFieldTypeAndOffset(Module, ContainerTypeId, Field, FieldTypeId, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFieldTypeAndOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFieldTypeAndOffset
            {
                dPrintf("%s GetFieldTypeAndOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFieldTypeAndOffset

//******************************************************************************

HRESULT
GetFieldTypeAndOffsetWide
(
    ULONG64             Module,
    ULONG               ContainerTypeId,
    PCWSTR              Field,
    PULONG              FieldTypeId,
    PULONG              Offset
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Field != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetFieldTypeAndOffsetWide symbols method
        hResult = pDbgSymbols3->GetFieldTypeAndOffsetWide(Module, ContainerTypeId, Field, FieldTypeId, Offset);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetFieldTypeAndOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetFieldTypeAndOffsetWide
            {
                dPrintf("%s GetFieldTypeAndOffsetWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetFieldTypeAndOffsetWide

//******************************************************************************

HRESULT
AddSyntheticSymbol
(
    ULONG64             Offset,
    ULONG               Size,
    PCSTR               Name,
    ULONG               Flags,
    PDEBUG_MODULE_AND_ID Id
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddSyntheticSymbol symbols method
        hResult = pDbgSymbols3->AddSyntheticSymbol(Offset, Size, Name, Flags, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddSyntheticSymbol %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddSyntheticSymbol
            {
                dPrintf("%s AddSyntheticSymbol %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddSyntheticSymbol

//******************************************************************************

HRESULT
AddSyntheticSymbolWide
(
    ULONG64             Offset,
    ULONG               Size,
    PCWSTR              Name,
    ULONG               Flags,
    PDEBUG_MODULE_AND_ID Id
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Name != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddSyntheticSymbolWide symbols method
        hResult = pDbgSymbols3->AddSyntheticSymbolWide(Offset, Size, Name, Flags, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddSyntheticSymbolWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddSyntheticSymbolWide
            {
                dPrintf("%s AddSyntheticSymbolWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddSyntheticSymbolWide

//******************************************************************************

HRESULT
RemoveSyntheticSymbol
(
    PDEBUG_MODULE_AND_ID Id
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveSyntheticSymbol symbols method
        hResult = pDbgSymbols3->RemoveSyntheticSymbol(Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveSyntheticSymbol %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveSyntheticSymbol
            {
                dPrintf("%s RemoveSyntheticSymbol %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveSyntheticSymbol

//******************************************************************************

HRESULT
GetSymbolEntriesByOffset
(
    ULONG64             Offset,
    ULONG               Flags,
    PDEBUG_MODULE_AND_ID Ids,
    PULONG64            Displacements,
    ULONG               IdsCount,
    PULONG              Entries
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntriesByOffset symbols method
        hResult = pDbgSymbols3->GetSymbolEntriesByOffset(Offset, Flags, Ids, Displacements, IdsCount, Entries);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntriesByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntriesByOffset
            {
                dPrintf("%s GetSymbolEntriesByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntriesByOffset

//******************************************************************************

HRESULT
GetSymbolEntriesByName
(
    PCSTR               Symbol,
    ULONG               Flags,
    PDEBUG_MODULE_AND_ID Ids,
    ULONG               IdsCount,
    PULONG              Entries
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntriesByName symbols method
        hResult = pDbgSymbols3->GetSymbolEntriesByName(Symbol, Flags, Ids, IdsCount, Entries);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntriesByName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntriesByName
            {
                dPrintf("%s GetSymbolEntriesByName %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntriesByName

//******************************************************************************

HRESULT
GetSymbolEntriesByNameWide
(
    PCWSTR              Symbol,
    ULONG               Flags,
    PDEBUG_MODULE_AND_ID Ids,
    ULONG               IdsCount,
    PULONG              Entries
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Symbol != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntriesByNameWide symbols method
        hResult = pDbgSymbols3->GetSymbolEntriesByNameWide(Symbol, Flags, Ids, IdsCount, Entries);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntriesByNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntriesByNameWide
            {
                dPrintf("%s GetSymbolEntriesByNameWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntriesByNameWide

//******************************************************************************

HRESULT
GetSymbolEntryByToken
(
    ULONG64             ModuleBase,
    ULONG               Token,
    PDEBUG_MODULE_AND_ID Id
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntryByToken symbols method
        hResult = pDbgSymbols3->GetSymbolEntryByToken(ModuleBase, Token, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntryByToken %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntryByToken
            {
                dPrintf("%s GetSymbolEntryByToken %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntryByToken

//******************************************************************************

HRESULT
GetSymbolEntryInformation
(
    PDEBUG_MODULE_AND_ID Id,
    PDEBUG_SYMBOL_ENTRY Info
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);
    assert(Info != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntryInformation symbols method
        hResult = pDbgSymbols3->GetSymbolEntryInformation(Id, Info);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntryInformation %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntryInformation
            {
                dPrintf("%s GetSymbolEntryInformation %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntryInformation

//******************************************************************************

HRESULT
GetSymbolEntryString
(
    PDEBUG_MODULE_AND_ID Id,
    ULONG               Which,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntryString symbols method
        hResult = pDbgSymbols3->GetSymbolEntryString(Id, Which, Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntryString %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntryString
            {
                dPrintf("%s GetSymbolEntryString %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntryString

//******************************************************************************

HRESULT
GetSymbolEntryStringWide
(
    PDEBUG_MODULE_AND_ID Id,
    ULONG               Which,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntryStringWide symbols method
        hResult = pDbgSymbols3->GetSymbolEntryStringWide(Id, Which, Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntryStringWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntryStringWide
            {
                dPrintf("%s GetSymbolEntryStringWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntryStringWide

//******************************************************************************

HRESULT
GetSymbolEntryOffsetRegions
(
    PDEBUG_MODULE_AND_ID Id,
    ULONG               Flags,
    PDEBUG_OFFSET_REGION Regions,
    ULONG               RegionsCount,
    PULONG              RegionsAvail
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Id != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntryOffsetRegions symbols method
        hResult = pDbgSymbols3->GetSymbolEntryOffsetRegions(Id, Flags, Regions, RegionsCount, RegionsAvail);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntryOffsetRegions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntryOffsetRegions
            {
                dPrintf("%s GetSymbolEntryOffsetRegions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntryOffsetRegions

//******************************************************************************

HRESULT
GetSymbolEntryBySymbolEntry
(
    PDEBUG_MODULE_AND_ID FromId,
    ULONG               Flags,
    PDEBUG_MODULE_AND_ID ToId
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(FromId != NULL);
    assert(ToId != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolEntryBySymbolEntry symbols method
        hResult = pDbgSymbols3->GetSymbolEntryBySymbolEntry(FromId, Flags, ToId);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolEntryBySymbolEntry %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolEntryBySymbolEntry
            {
                dPrintf("%s GetSymbolEntryBySymbolEntry %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolEntryBySymbolEntry

//******************************************************************************

HRESULT
GetSourceEntriesByOffset
(
    ULONG64             Offset,
    ULONG               Flags,
    PDEBUG_SYMBOL_SOURCE_ENTRY Entries,
    ULONG               EntriesCount,
    PULONG              EntriesAvail
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceEntriesByOffset symbols method
        hResult = pDbgSymbols3->GetSourceEntriesByOffset(Offset, Flags, Entries, EntriesCount, EntriesAvail);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceEntriesByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceEntriesByOffset
            {
                dPrintf("%s GetSourceEntriesByOffset %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceEntriesByOffset

//******************************************************************************

HRESULT
GetSourceEntriesByLine
(
    ULONG               Line,
    PCSTR               File,
    ULONG               Flags,
    PDEBUG_SYMBOL_SOURCE_ENTRY Entries,
    ULONG               EntriesCount,
    PULONG              EntriesAvail
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceEntriesByLine symbols method
        hResult = pDbgSymbols3->GetSourceEntriesByLine(Line, File, Flags, Entries, EntriesCount, EntriesAvail);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceEntriesByLine %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceEntriesByLine
            {
                dPrintf("%s GetSourceEntriesByLine %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceEntriesByLine

//******************************************************************************

HRESULT
GetSourceEntriesByLineWide
(
    ULONG               Line,
    PCWSTR              File,
    ULONG               Flags,
    PDEBUG_SYMBOL_SOURCE_ENTRY Entries,
    ULONG               EntriesCount,
    PULONG              EntriesAvail
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceEntriesByLineWide symbols method
        hResult = pDbgSymbols3->GetSourceEntriesByLineWide(Line, File, Flags, Entries, EntriesCount, EntriesAvail);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceEntriesByLineWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceEntriesByLineWide
            {
                dPrintf("%s GetSourceEntriesByLineWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceEntriesByLineWide

//******************************************************************************

HRESULT
GetSourceEntryString
(
    PDEBUG_SYMBOL_SOURCE_ENTRY Entry,
    ULONG               Which,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Entry != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceEntryString symbols method
        hResult = pDbgSymbols3->GetSourceEntryString(Entry, Which, Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceEntryString %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceEntryString
            {
                dPrintf("%s GetSourceEntryString %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceEntryString

//******************************************************************************

HRESULT
GetSourceEntryStringWide
(
    PDEBUG_SYMBOL_SOURCE_ENTRY Entry,
    ULONG               Which,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Entry != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceEntryStringWide symbols method
        hResult = pDbgSymbols3->GetSourceEntryStringWide(Entry, Which, Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceEntryStringWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceEntryStringWide
            {
                dPrintf("%s GetSourceEntryStringWide %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceEntryStringWide

//******************************************************************************

HRESULT
GetSourceEntryOffsetRegions
(
    PDEBUG_SYMBOL_SOURCE_ENTRY Entry,
    ULONG               Flags,
    PDEBUG_OFFSET_REGION Regions,
    ULONG               RegionsCount,
    PULONG              RegionsAvail
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Entry != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceEntryOffsetRegions symbols method
        hResult = pDbgSymbols3->GetSourceEntryOffsetRegions(Entry, Flags, Regions, RegionsCount, RegionsAvail);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceEntryOffsetRegions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceEntryOffsetRegions
            {
                dPrintf("%s GetSourceEntryOffsetRegions %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceEntryOffsetRegions

//******************************************************************************

HRESULT
GetSourceEntryBySourceEntry
(
    PDEBUG_SYMBOL_SOURCE_ENTRY FromEntry,
    ULONG               Flags,
    PDEBUG_SYMBOL_SOURCE_ENTRY ToEntry
)
{
    PDEBUG_SYMBOLS3     pDbgSymbols3 = debugSymbols3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(FromEntry != NULL);
    assert(ToEntry != NULL);

    // Check for valid symbols interface
    if (pDbgSymbols3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceEntryBySourceEntry symbols method
        hResult = pDbgSymbols3->GetSourceEntryBySourceEntry(FromEntry, Flags, ToEntry);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng symbols interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_SYMBOLS))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceEntryBySourceEntry %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceEntryBySourceEntry
            {
                dPrintf("%s GetSourceEntryBySourceEntry %s = 0x%08x\n", DML(bold("DbgSymbols:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceEntryBySourceEntry

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
