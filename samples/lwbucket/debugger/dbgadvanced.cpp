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
|*  Module: dbgadvanced.cpp                                                   *|
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
// Debugger Advanced Interface wrappers
//
//******************************************************************************

HRESULT
GetThreadContext
(
    PVOID               Context,
    ULONG               ContextSize
)
{
    PDEBUG_ADVANCED     pDbgAdvanced = debugAdvancedInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Context != NULL);

    // Check for valid advanced interface
    if (pDbgAdvanced != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetThreadContext advanced method
        hResult = pDbgAdvanced->GetThreadContext(Context, ContextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetThreadContext %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetThreadContext
            {
                dPrintf("%s GetThreadContext %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetThreadContext

//******************************************************************************

HRESULT
SetThreadContext
(
    PVOID               Context,
    ULONG               ContextSize
)
{
    PDEBUG_ADVANCED     pDbgAdvanced = debugAdvancedInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Context != NULL);

    // Check for valid advanced interface
    if (pDbgAdvanced != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetThreadContext advanced method
        hResult = pDbgAdvanced->SetThreadContext(Context, ContextSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetThreadContext %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetThreadContext
            {
                dPrintf("%s SetThreadContext %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetThreadContext

//******************************************************************************

HRESULT
Request
(
    ULONG               Request,
    PVOID               InBuffer,
    ULONG               InBufferSize,
    PVOID               OutBuffer,
    ULONG               OutBufferSize,
    PULONG              OutSize
)
{
    PDEBUG_ADVANCED2    pDbgAdvanced2 = debugAdvanced2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid advanced interface
    if (pDbgAdvanced2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the Request advanced method
        hResult = pDbgAdvanced2->Request(Request, InBuffer, InBufferSize, OutBuffer, OutBufferSize, OutSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s Request %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed Request
            {
                dPrintf("%s Request %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // Request

//******************************************************************************

HRESULT
GetSourceFileInformation
(
    ULONG               Which,
    PSTR                SourceFile,
    ULONG64             Arg64,
    ULONG               Arg32,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              InfoSize
)
{
    PDEBUG_ADVANCED2    pDbgAdvanced2 = debugAdvanced2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(SourceFile != NULL);

    // Check for valid advanced interface
    if (pDbgAdvanced2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceFileInformation advanced method
        hResult = pDbgAdvanced2->GetSourceFileInformation(Which, SourceFile, Arg64, Arg32, Buffer, BufferSize, InfoSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceFileInformation %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceFileInformation
            {
                dPrintf("%s GetSourceFileInformation %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceFileInformation

//******************************************************************************

HRESULT
FindSourceFileAndToken
(
    ULONG               StartElement,
    ULONG64             ModAddr,
    PCSTR               File,
    ULONG               Flags,
    PVOID               FileToken,
    ULONG               FileTokenSize,
    PULONG              FoundElement,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              FoundSize
)
{
    PDEBUG_ADVANCED2    pDbgAdvanced2 = debugAdvanced2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid advanced interface
    if (pDbgAdvanced2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the FindSourceFileAndToken advanced method
        hResult = pDbgAdvanced2->FindSourceFileAndToken(StartElement, ModAddr, File, Flags, FileToken, FileTokenSize, FoundElement, Buffer, BufferSize, FoundSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s FindSourceFileAndToken %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed FindSourceFileAndToken
            {
                dPrintf("%s FindSourceFileAndToken %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // FindSourceFileAndToken

//******************************************************************************

HRESULT
GetSymbolInformation
(
    ULONG               Which,
    ULONG64             Arg64,
    ULONG               Arg32,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              InfoSize,
    PSTR                StringBuffer,
    ULONG               StringBufferSize,
    PULONG              StringSize
)
{
    PDEBUG_ADVANCED2    pDbgAdvanced2 = debugAdvanced2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid advanced interface
    if (pDbgAdvanced2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolInformation advanced method
        hResult = pDbgAdvanced2->GetSymbolInformation(Which, Arg64, Arg32, Buffer, BufferSize, InfoSize, StringBuffer, StringBufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolInformation %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolInformation
            {
                dPrintf("%s GetSymbolInformation %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolInformation

//******************************************************************************

HRESULT
GetSystemObjectInformation
(
    ULONG               Which,
    ULONG64             Arg64,
    ULONG               Arg32,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              InfoSize
)
{
    PDEBUG_ADVANCED2    pDbgAdvanced2 = debugAdvanced2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid advanced interface
    if (pDbgAdvanced2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSystemObjectInformation advanced method
        hResult = pDbgAdvanced2->GetSystemObjectInformation(Which, Arg64, Arg32, Buffer, BufferSize, InfoSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSystemObjectInformation %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSystemObjectInformation
            {
                dPrintf("%s GetSystemObjectInformation %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSystemObjectInformation

//******************************************************************************

HRESULT
GetSourceFileInformationWide
(
    ULONG               Which,
    PWSTR               SourceFile,
    ULONG64             Arg64,
    ULONG               Arg32,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              InfoSize
)
{
    PDEBUG_ADVANCED3    pDbgAdvanced3 = debugAdvanced3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(SourceFile != NULL);

    // Check for valid advanced interface
    if (pDbgAdvanced3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSourceFileInformationWide advanced method
        hResult = pDbgAdvanced3->GetSourceFileInformationWide(Which, SourceFile, Arg64, Arg32, Buffer, BufferSize, InfoSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSourceFileInformationWide %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSourceFileInformationWide
            {
                dPrintf("%s GetSourceFileInformationWide %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSourceFileInformationWide

//******************************************************************************

HRESULT
FindSourceFileAndTokenWide
(
    ULONG               StartElement,
    ULONG64             ModAddr,
    PCWSTR              File,
    ULONG               Flags,
    PVOID               FileToken,
    ULONG               FileTokenSize,
    PULONG              FoundElement,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              FoundSize
)
{
    PDEBUG_ADVANCED3    pDbgAdvanced3 = debugAdvanced3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(File != NULL);

    // Check for valid advanced interface
    if (pDbgAdvanced3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the FindSourceFileAndTokenWide advanced method
        hResult = pDbgAdvanced3->FindSourceFileAndTokenWide(StartElement, ModAddr, File, Flags, FileToken, FileTokenSize, FoundElement, Buffer, BufferSize, FoundSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s FindSourceFileAndTokenWide %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed FindSourceFileAndTokenWide
            {
                dPrintf("%s FindSourceFileAndTokenWide %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // FindSourceFileAndTokenWide

//******************************************************************************

HRESULT
GetSymbolInformationWide
(
    ULONG               Which,
    ULONG64             Arg64,
    ULONG               Arg32,
    PVOID               Buffer,
    ULONG               BufferSize,
    PULONG              InfoSize,
    PWSTR               StringBuffer,
    ULONG               StringBufferSize,
    PULONG              StringSize
)
{
    PDEBUG_ADVANCED3    pDbgAdvanced3 = debugAdvanced3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid advanced interface
    if (pDbgAdvanced3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetSymbolInformationWide advanced method
        hResult = pDbgAdvanced3->GetSymbolInformationWide(Which, Arg64, Arg32, Buffer, BufferSize, InfoSize, StringBuffer, StringBufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng advanced interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_ADVANCED))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetSymbolInformationWide %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetSymbolInformationWide
            {
                dPrintf("%s GetSymbolInformationWide %s = 0x%08x\n", DML(bold("DbgAdvanced:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetSymbolInformationWide

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
