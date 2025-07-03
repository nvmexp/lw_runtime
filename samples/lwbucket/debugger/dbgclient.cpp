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
|*  Module: dbgclient.cpp                                                     *|
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
// Debugger Client Interface wrappers
//
//******************************************************************************

HRESULT
AttachKernel
(
    ULONG               Flags,
    PCSTR               ConnectOptions
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AttachKernel client method
        hResult = pDbgClient->AttachKernel(Flags, ConnectOptions);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AttachKernel %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AttachKernel
            {
                dPrintf("%s AttachKernel %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AttachKernel

//******************************************************************************

HRESULT
GetKernelConnectionOptions
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              OptionsSize
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetKernelConnectionOptions client method
        hResult = pDbgClient->GetKernelConnectionOptions(Buffer, BufferSize, OptionsSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetKernelConnectionOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AttachKernel
            {
                dPrintf("%s GetKernelConnectionOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetKernelConnectionOptions

//******************************************************************************

HRESULT
SetKernelConnectionOptions
(
    PCSTR               Options
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetKernelConnectionOptionsWide client method
        hResult = pDbgClient->SetKernelConnectionOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetKernelConnectionOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetKernelConnectionOptions
            {
                dPrintf("%s SetKernelConnectionOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetKernelConnectionOptions

//******************************************************************************

HRESULT
StartProcessServer
(
    ULONG               Flags,
    PCSTR               Options,
    PVOID               Reserved
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the StartProcessServer client method
        hResult = pDbgClient->StartProcessServer(Flags, Options, Reserved);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s StartProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed StartProcessServer
            {
                dPrintf("%s StartProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // StartProcessServer

//******************************************************************************

HRESULT
ConnectProcessServer
(
    PCSTR               RemoteOptions,
    PULONG64            Server
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(RemoteOptions != NULL);
    assert(Server != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ConnectProcessServer client method
        hResult = pDbgClient->ConnectProcessServer(RemoteOptions, Server);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ConnectProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ConnectProcessServer
            {
                dPrintf("%s ConnectProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ConnectProcessServer

//******************************************************************************

HRESULT
DisconnectProcessServer
(
    ULONG64             Server
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the DisconnectProcessServer client method
        hResult = pDbgClient->DisconnectProcessServer(Server);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s DisconnectProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed DisconnectProcessServer
            {
                dPrintf("%s DisconnectProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // DisconnectProcessServer

//******************************************************************************

HRESULT
GetRunningProcessSystemIds
(
    ULONG64             Server,
    PULONG              Ids,
    ULONG               Count,
    PULONG              ActualCount
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetRunningProcessSystemIds client method
        hResult = pDbgClient->GetRunningProcessSystemIds(Server, Ids, Count, ActualCount);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetRunningProcessSystemIds %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AttachKernel
            {
                dPrintf("%s GetRunningProcessSystemIds %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetRunningProcessSystemIds

//******************************************************************************

HRESULT
GetRunningProcessSystemIdByExelwtableName
(
    ULONG64             Server,
    PCSTR               ExeName,
    ULONG               Flags,
    PULONG              Id
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(ExeName != NULL);
    assert(Id != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetRunningProcessSystemIdByExelwtableName client method
        hResult = pDbgClient->GetRunningProcessSystemIdByExelwtableName(Server, ExeName, Flags, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetRunningProcessSystemIdByExelwtableName %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetRunningProcessSystemIdByExelwtableName
            {
                dPrintf("%s GetRunningProcessSystemIdByExelwtableName %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetRunningProcessSystemIdByExelwtableName

//******************************************************************************

HRESULT
GetRunningProcessDescription
(
    ULONG64             Server,
    ULONG               SystemId,
    ULONG               Flags,
    PSTR                ExeName,
    ULONG               ExeNameSize,
    PULONG              ActualExeNameSize,
    PSTR                Description,
    ULONG               DescriptionSize,
    PULONG              ActualDescriptionSize
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetRunningProcessDescription client method
        hResult = pDbgClient->GetRunningProcessDescription(Server, SystemId, Flags, ExeName, ExeNameSize, ActualExeNameSize, Description, DescriptionSize, ActualDescriptionSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetRunningProcessDescription %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetRunningProcessDescription
            {
                dPrintf("%s GetRunningProcessDescription %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetRunningProcessDescription

//******************************************************************************

HRESULT
AttachProcess
(
    ULONG64             Server,
    ULONG               ProcessId,
    ULONG               AttachFlags
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AttachProcess client method
        hResult = pDbgClient->AttachProcess(Server, ProcessId, AttachFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AttachProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AttachProcess
            {
                dPrintf("%s AttachProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AttachProcess

//******************************************************************************

HRESULT
CreateProcess
(
    ULONG64             Server,
    PSTR                CommandLine,
    ULONG               CreateFlags
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(CommandLine != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcess client method
        hResult = pDbgClient->CreateProcess(Server, CommandLine, CreateFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcess
            {
                dPrintf("%s CreateProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcess

//******************************************************************************

HRESULT
CreateProcessAndAttach
(
    ULONG64             Server,
    PSTR                CommandLine,
    ULONG               CreateFlags,
    ULONG               ProcessId,
    ULONG               AttachFlags
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcessAndAttach client method
        hResult = pDbgClient->CreateProcessAndAttach(Server, CommandLine, CreateFlags, ProcessId, AttachFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcessAndAttach %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcessAndAttach
            {
                dPrintf("%s CreateProcessAndAttach %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcessAndAttach

//******************************************************************************

HRESULT
GetProcessOptions
(
    PULONG              Options
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetProcessOptions client method
        hResult = pDbgClient->GetProcessOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetProcessOptions
            {
                dPrintf("%s GetProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetProcessOptions

//******************************************************************************

HRESULT
AddProcessOptions
(
    ULONG               Options
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddProcessOptions client method
        hResult = pDbgClient->AddProcessOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddProcessOptions
            {
                dPrintf("%s AddProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddProcessOptions

//******************************************************************************

HRESULT
RemoveProcessOptions
(
    ULONG               Options
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the RemoveProcessOptions client method
        hResult = pDbgClient->RemoveProcessOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s RemoveProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed RemoveProcessOptions
            {
                dPrintf("%s RemoveProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // RemoveProcessOptions

//******************************************************************************

HRESULT
SetProcessOptions
(
    ULONG               Options
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetProcessOptions client method
        hResult = pDbgClient->SetProcessOptions(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetProcessOptions
            {
                dPrintf("%s SetProcessOptions %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetProcessOptions

//******************************************************************************

HRESULT
OpenDumpFile
(
    PCSTR               DumpFile
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(DumpFile != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OpenDumpFile client method
        hResult = pDbgClient->OpenDumpFile(DumpFile);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OpenDumpFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OpenDumpFile
            {
                dPrintf("%s OpenDumpFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OpenDumpFile

//******************************************************************************

HRESULT
WriteDumpFile
(
    PCSTR               DumpFile,
    ULONG               Qualifier
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(DumpFile != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteDumpFile client method
        hResult = pDbgClient->WriteDumpFile(DumpFile, Qualifier);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteDumpFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteDumpFile
            {
                dPrintf("%s WriteDumpFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteDumpFile

//******************************************************************************

HRESULT
ConnectSession
(
    ULONG               Flags,
    ULONG               HistoryLimit
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ConnectSession client method
        hResult = pDbgClient->ConnectSession(Flags, HistoryLimit);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ConnectSession %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ConnectSession
            {
                dPrintf("%s ConnectSession %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ConnectSession

//******************************************************************************

HRESULT
StartServer
(
    PCSTR               Options
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the StartServer client method
        hResult = pDbgClient->StartServer(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s StartServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed StartServer
            {
                dPrintf("%s StartServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // StartServer

//******************************************************************************

HRESULT
OutputServers
(
    ULONG               OutputControl,
    PCSTR               Machine,
    ULONG               Flags
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Machine != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputServers client method
        hResult = pDbgClient->OutputServers(OutputControl, Machine, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputServers %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputServers
            {
                dPrintf("%s OutputServers %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputServers

//******************************************************************************

HRESULT
TerminateProcesses()
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the TerminateProcesses() client method
        hResult = pDbgClient->TerminateProcesses();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s TerminateProcesses %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed TerminateProcesses
            {
                dPrintf("%s TerminateProcesses %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // TerminateProcesses

//******************************************************************************

HRESULT
DetachProcesses()
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the DetachProcesses() client method
        hResult = pDbgClient->DetachProcesses();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s DetachProcesses %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed DetachProcesses
            {
                dPrintf("%s DetachProcesses %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // DetachProcesses

//******************************************************************************

HRESULT
EndSession
(
    ULONG               Flags
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the EndSession client method
        hResult = pDbgClient->EndSession(Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s EndSession %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed EndSession
            {
                dPrintf("%s EndSession %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // EndSession

//******************************************************************************

HRESULT
GetExitCode
(
    PULONG              Code
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Code != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetExitCode client method
        hResult = pDbgClient->GetExitCode(Code);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetExitCode %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetExitCode
            {
                dPrintf("%s GetExitCode %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetExitCode

//******************************************************************************

HRESULT
DispatchCallbacks
(
    ULONG               Timeout
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the DispatchCallbacks client method
        hResult = pDbgClient->DispatchCallbacks(Timeout);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s DispatchCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed DispatchCallbacks
            {
                dPrintf("%s DispatchCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // DispatchCallbacks

//******************************************************************************

HRESULT
ExitDispatch
(
    PDEBUG_CLIENT       Client
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Client != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ExitDispatch client method
        hResult = pDbgClient->ExitDispatch(Client);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ExitDispatch %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ExitDispatch
            {
                dPrintf("%s ExitDispatch %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ExitDispatch

//******************************************************************************

HRESULT
CreateClient
(
    PDEBUG_CLIENT*      Client
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Client != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateClient client method
        hResult = pDbgClient->CreateClient(Client);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateClient %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateClient
            {
                dPrintf("%s CreateClient %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateClient

//******************************************************************************

HRESULT
GetInputCallbacks
(
    PDEBUG_INPUT_CALLBACKS* Callbacks
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Callbacks != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetInputCallbacks client method
        hResult = pDbgClient->GetInputCallbacks(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetInputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetInputCallbacks
            {
                dPrintf("%s GetInputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetInputCallbacks

//******************************************************************************

HRESULT
SetInputCallbacks
(
    PDEBUG_INPUT_CALLBACKS Callbacks
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetInputCallbacks client method
        hResult = pDbgClient->SetInputCallbacks(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetInputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetInputCallbacks
            {
                dPrintf("%s SetInputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetInputCallbacks

//******************************************************************************

HRESULT
GetOutputCallbacks
(
    PDEBUG_OUTPUT_CALLBACKS* Callbacks
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Callbacks != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOutputCallbacks client method
        hResult = pDbgClient->GetOutputCallbacks(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOutputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOutputCallbacks
            {
                dPrintf("%s GetOutputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOutputCallbacks

//******************************************************************************

HRESULT
SetOutputCallbacks
(
    PDEBUG_OUTPUT_CALLBACKS Callbacks
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOutputCallbacks client method
        hResult = pDbgClient->SetOutputCallbacks(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOutputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOutputCallbacks
            {
                dPrintf("%s SetOutputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOutputCallbacks

//******************************************************************************

HRESULT
GetOutputMask
(
    PULONG              Mask
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Mask != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOutputMask client method
        hResult = pDbgClient->GetOutputMask(Mask);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOutputMask
            {
                dPrintf("%s GetOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOutputMask

//******************************************************************************

HRESULT
SetOutputMask
(
    ULONG               Mask
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOutputMask client method
        hResult = pDbgClient->SetOutputMask(Mask);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOutputMask
            {
                dPrintf("%s SetOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOutputMask

//******************************************************************************

HRESULT
GetOtherOutputMask
(
    PDEBUG_CLIENT       Client,
    PULONG              Mask
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Client != NULL);
    assert(Mask != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOtherOutputMask client method
        hResult = pDbgClient->GetOtherOutputMask(Client, Mask);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOtherOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOtherOutputMask
            {
                dPrintf("%s GetOtherOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOtherOutputMask

//******************************************************************************

HRESULT
SetOtherOutputMask
(
    PDEBUG_CLIENT       Client,
    ULONG               Mask
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Client != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOtherOutputMask client method
        hResult = pDbgClient->SetOtherOutputMask(Client, Mask);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOtherOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOtherOutputMask
            {
                dPrintf("%s SetOtherOutputMask %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOtherOutputMask

//******************************************************************************

HRESULT
GetOutputWidth
(
    PULONG              Columns
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Columns != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOutputWidth client method
        hResult = pDbgClient->GetOutputWidth(Columns);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOutputWidth %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOutputWidth
            {
                dPrintf("%s GetOutputWidth %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOutputWidth

//******************************************************************************

HRESULT
SetOutputWidth
(
    ULONG               Columns
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOutputWidth client method
        hResult = pDbgClient->SetOutputWidth(Columns);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOutputWidth %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOutputWidth
            {
                dPrintf("%s SetOutputWidth %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOutputWidth

//******************************************************************************

HRESULT
GetOutputLinePrefix
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              PrefixSize
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOutputLinePrefix client method
        hResult = pDbgClient->GetOutputLinePrefix(Buffer, BufferSize, PrefixSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOutputLinePrefix
            {
                dPrintf("%s GetOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOutputLinePrefix

//******************************************************************************

HRESULT
SetOutputLinePrefix
(
    PCSTR               Prefix
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOutputLinePrefix client method
        hResult = pDbgClient->SetOutputLinePrefix(Prefix);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOutputLinePrefix
            {
                dPrintf("%s SetOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOutputLinePrefix

//******************************************************************************

HRESULT
GetIdentity
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              IdentitySize
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetIdentity client method
        hResult = pDbgClient->GetIdentity(Buffer, BufferSize, IdentitySize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetIdentity %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetIdentity
            {
                dPrintf("%s GetIdentity %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetIdentity

//******************************************************************************

HRESULT
OutputIdentity
(
    ULONG               OutputControl,
    ULONG               Flags,
    PCSTR               Format
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputIdentity client method
        hResult = pDbgClient->OutputIdentity(OutputControl, Flags, Format);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputIdentity %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputIdentity
            {
                dPrintf("%s OutputIdentity %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputIdentity

//******************************************************************************

HRESULT
GetEventCallbacks
(
    PDEBUG_EVENT_CALLBACKS* Callbacks
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Callbacks != NULL);

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventCallbacks client method
        hResult = pDbgClient->GetEventCallbacks(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventCallbacks
            {
                dPrintf("%s GetEventCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventCallbacks

//******************************************************************************

HRESULT
SetEventCallbacks
(
    PDEBUG_EVENT_CALLBACKS Callbacks
)
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetEventCallbacks client method
        hResult = pDbgClient->SetEventCallbacks(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetEventCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetEventCallbacks
            {
                dPrintf("%s SetEventCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetEventCallbacks

//******************************************************************************

HRESULT
FlushCallbacks()
{
    PDEBUG_CLIENT       pDbgClient = debugClientInterface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the FlushCallbacks client method
        hResult = pDbgClient->FlushCallbacks();

        // Release the debug interface
        releaseDebugInterface();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s FlushCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed FlushCallbacks
            {
                dPrintf("%s FlushCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // FlushCallbacks

//******************************************************************************

HRESULT
WriteDumpFile2
(
    PCSTR               DumpFile,
    ULONG               Qualifier,
    ULONG               FormatFlags,
    PCSTR               Comment
)
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(DumpFile != NULL);

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteDumpFile2 client method
        hResult = pDbgClient2->WriteDumpFile2(DumpFile, Qualifier, FormatFlags, Comment);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteDumpFile2 %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteDumpFile2
            {
                dPrintf("%s WriteDumpFile2 %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteDumpFile2

//******************************************************************************

HRESULT
AddDumpInformationFile
(
    PCSTR               InfoFile,
    ULONG               Type
)
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(InfoFile != NULL);

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddDumpInformationFile client method
        hResult = pDbgClient2->AddDumpInformationFile(InfoFile, Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddDumpInformationFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddDumpInformationFile
            {
                dPrintf("%s AddDumpInformationFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddDumpInformationFile

//******************************************************************************

HRESULT
EndProcessServer
(
    ULONG64             Server
)
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the EndProcessServer client method
        hResult = pDbgClient2->EndProcessServer(Server);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s EndProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed EndProcessServer
            {
                dPrintf("%s EndProcessServer %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // EndProcessServer

//******************************************************************************

HRESULT
WaitForProcessServerEnd
(
    ULONG               Timeout
)
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WaitForProcessServerEnd client method
        hResult = pDbgClient2->WaitForProcessServerEnd(Timeout);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WaitForProcessServerEnd %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WaitForProcessServerEnd
            {
                dPrintf("%s WaitForProcessServerEnd %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WaitForProcessServerEnd

//******************************************************************************

HRESULT
IsKernelDebuggerEnabled()
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the IsKernelDebuggerEnabled client method
        hResult = pDbgClient2->IsKernelDebuggerEnabled();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s IsKernelDebuggerEnabled %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed IsKernelDebuggerEnabled
            {
                dPrintf("%s IsKernelDebuggerEnabled %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // IsKernelDebuggerEnabled

//******************************************************************************

HRESULT
TerminateLwrrentProcess()
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the TerminateLwrrentProcess client method
        hResult = pDbgClient2->TerminateLwrrentProcess();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s TerminateLwrrentProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed TerminateLwrrentProcess
            {
                dPrintf("%s TerminateLwrrentProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // TerminateLwrrentProcess

//******************************************************************************

HRESULT
DetachLwrrentProcess()
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the DetachLwrrentProcess client method
        hResult = pDbgClient2->DetachLwrrentProcess();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s DetachLwrrentProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed DetachLwrrentProcess
            {
                dPrintf("%s DetachLwrrentProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // DetachLwrrentProcess

//******************************************************************************

HRESULT
AbandonLwrrentProcess()
{
    PDEBUG_CLIENT2      pDbgClient2 = debugClient2Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient2 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AbandonLwrrentProcess client method
        hResult = pDbgClient2->AbandonLwrrentProcess();

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AbandonLwrrentProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AbandonLwrrentProcess
            {
                dPrintf("%s AbandonLwrrentProcess %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AbandonLwrrentProcess

//******************************************************************************

HRESULT
GetRunningProcessSystemIdByExelwtableNameWide
(
    ULONG64             Server,
    PCWSTR              ExeName,
    ULONG               Flags,
    PULONG              Id
)
{
    PDEBUG_CLIENT3      pDbgClient3 = debugClient3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(ExeName != NULL);
    assert(Id != NULL);

    // Check for valid debug client interface
    if (pDbgClient3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetRunningProcessSystemIdByExelwtableNameWide client method
        hResult = pDbgClient3->GetRunningProcessSystemIdByExelwtableNameWide(Server, ExeName, Flags, Id);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetRunningProcessSystemIdByExelwtableNameWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetRunningProcessSystemIdByExelwtableNameWide
            {
                dPrintf("%s GetRunningProcessSystemIdByExelwtableNameWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetRunningProcessSystemIdByExelwtableNameWide

//******************************************************************************

HRESULT
GetRunningProcessDescriptionWide
(
    ULONG64             Server,
    ULONG               SystemId,
    ULONG               Flags,
    PWSTR               ExeName,
    ULONG               ExeNameSize,
    PULONG              ActualExeNameSize,
    PWSTR               Description,
    ULONG               DescriptionSize,
    PULONG              ActualDescriptionSize
)
{
    PDEBUG_CLIENT3      pDbgClient3 = debugClient3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetRunningProcessDescriptionWide client method
        hResult = pDbgClient3->GetRunningProcessDescriptionWide(Server, SystemId, Flags, ExeName, ExeNameSize, ActualExeNameSize, Description, DescriptionSize, ActualDescriptionSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetRunningProcessDescriptionWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetRunningProcessDescriptionWide
            {
                dPrintf("%s GetRunningProcessDescriptionWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetRunningProcessDescriptionWide

//******************************************************************************

HRESULT
CreateProcessWide
(
    ULONG64             Server,
    PWSTR               CommandLine,
    ULONG               CreateFlags
)
{
    PDEBUG_CLIENT3      pDbgClient3 = debugClient3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(CommandLine != NULL);

    // Check for valid debug client interface
    if (pDbgClient3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcessWide client method
        hResult = pDbgClient3->CreateProcessWide(Server, CommandLine, CreateFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcessWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcessWide
            {
                dPrintf("%s CreateProcessWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcessWide

//******************************************************************************

HRESULT
CreateProcessAndAttachWide
(
    ULONG64             Server,
    PWSTR               CommandLine,
    ULONG               CreateFlags,
    ULONG               ProcessId,
    ULONG               AttachFlags
)
{
    PDEBUG_CLIENT3      pDbgClient3 = debugClient3Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient3 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcessAndAttachWide client method
        hResult = pDbgClient3->CreateProcessAndAttachWide(Server, CommandLine, CreateFlags, ProcessId, AttachFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcessAndAttachWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcessAndAttachWide
            {
                dPrintf("%s CreateProcessAndAttachWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcessAndAttachWide

//******************************************************************************

HRESULT
OpenDumpFileWide
(
    PCWSTR              FileName,
    ULONG64             FileHandle
)
{
    PDEBUG_CLIENT4      pDbgClient4 = debugClient4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OpenDumpFileWide client method
        hResult = pDbgClient4->OpenDumpFileWide(FileName, FileHandle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OpenDumpFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OpenDumpFileWide
            {
                dPrintf("%s OpenDumpFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OpenDumpFileWide

//******************************************************************************

HRESULT
WriteDumpFileWide
(
    PCWSTR              FileName,
    ULONG64             FileHandle,
    ULONG               Qualifier,
    ULONG               FormatFlags,
    PCWSTR              Comment
)
{
    PDEBUG_CLIENT4      pDbgClient4 = debugClient4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the WriteDumpFileWide client method
        hResult = pDbgClient4->WriteDumpFileWide(FileName, FileHandle, Qualifier, FormatFlags, Comment);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s WriteDumpFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed WriteDumpFileWide
            {
                dPrintf("%s WriteDumpFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // WriteDumpFileWide

//******************************************************************************

HRESULT
AddDumpInformationFileWide
(
    PCWSTR              FileName,
    ULONG64             FileHandle,
    ULONG               Type
)
{
    PDEBUG_CLIENT4      pDbgClient4 = debugClient4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AddDumpInformationFileWide client method
        hResult = pDbgClient4->AddDumpInformationFileWide(FileName, FileHandle, Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AddDumpInformationFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AddDumpInformationFileWide
            {
                dPrintf("%s AddDumpInformationFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AddDumpInformationFileWide

//******************************************************************************

HRESULT
GetNumberDumpFiles
(
    PULONG              Number
)
{
    PDEBUG_CLIENT4      pDbgClient4 = debugClient4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Number != NULL);

    // Check for valid debug client interface
    if (pDbgClient4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberDumpFiles client method
        hResult = pDbgClient4->GetNumberDumpFiles(Number);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberDumpFiles %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberDumpFiles
            {
                dPrintf("%s GetNumberDumpFiles %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberDumpFiles

//******************************************************************************

HRESULT
GetDumpFile
(
    ULONG               Index,
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              NameSize,
    PULONG64            Handle,
    PULONG              Type
)
{
    PDEBUG_CLIENT4      pDbgClient4 = debugClient4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);

    // Check for valid debug client interface
    if (pDbgClient4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDumpFile client method
        hResult = pDbgClient4->GetDumpFile(Index, Buffer, BufferSize, NameSize, Handle, Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDumpFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDumpFile
            {
                dPrintf("%s GetDumpFile %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDumpFile

//******************************************************************************

HRESULT
GetDumpFileWide
(
    ULONG               Index,
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              NameSize,
    PULONG64            Handle,
    PULONG              Type
)
{
    PDEBUG_CLIENT4      pDbgClient4 = debugClient4Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Type != NULL);

    // Check for valid debug client interface
    if (pDbgClient4 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDumpFileWide client method
        hResult = pDbgClient4->GetDumpFileWide(Index, Buffer, BufferSize, NameSize, Handle, Type);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetDumpFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetDumpFileWide
            {
                dPrintf("%s GetDumpFileWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetDumpFileWide

//******************************************************************************

HRESULT
AttachKernelWide
(
    ULONG               Flags,
    PCWSTR              ConnectOptions
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the AttachKernelWide client method
        hResult = pDbgClient5->AttachKernelWide(Flags, ConnectOptions);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s AttachKernelWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed AttachKernelWide
            {
                dPrintf("%s AttachKernelWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // AttachKernelWide

//******************************************************************************

HRESULT
GetKernelConnectionOptionsWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              OptionsSize
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetKernelConnectionOptionsWide client method
        hResult = pDbgClient5->GetKernelConnectionOptionsWide(Buffer, BufferSize, OptionsSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetKernelConnectionOptionsWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetKernelConnectionOptionsWide
            {
                dPrintf("%s GetKernelConnectionOptionsWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetKernelConnectionOptionsWide

//******************************************************************************

HRESULT
SetKernelConnectionOptionsWide
(
    PCWSTR              Options
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetKernelConnectionOptionsWide client method
        hResult = pDbgClient5->SetKernelConnectionOptionsWide(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetKernelConnectionOptionsWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetKernelConnectionOptionsWide
            {
                dPrintf("%s SetKernelConnectionOptionsWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetKernelConnectionOptionsWide

//******************************************************************************

HRESULT
StartProcessServerWide
(
    ULONG               Flags,
    PCWSTR              Options,
    PVOID               Reserved
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the StartProcessServerWide client method
        hResult = pDbgClient5->StartProcessServerWide(Flags, Options, Reserved);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s StartProcessServerWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed StartProcessServerWide
            {
                dPrintf("%s StartProcessServerWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // StartProcessServerWide

//******************************************************************************

HRESULT
ConnectProcessServerWide
(
    PCWSTR              RemoteOptions,
    PULONG64            Server
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(RemoteOptions != NULL);
    assert(Server != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ConnectProcessServerWide client method
        hResult = pDbgClient5->ConnectProcessServerWide(RemoteOptions, Server);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s ConnectProcessServerWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed ConnectProcessServerWide
            {
                dPrintf("%s ConnectProcessServerWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // ConnectProcessServerWide

//******************************************************************************

HRESULT
StartServerWide
(
    PCWSTR              Options
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Options != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the StartServerWide client method
        hResult = pDbgClient5->StartServerWide(Options);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s StartServerWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed StartServerWide
            {
                dPrintf("%s StartServerWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // StartServerWide

//******************************************************************************

HRESULT
OutputServersWide
(
    ULONG               OutputControl,
    PCWSTR              Machine,
    ULONG               Flags
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Machine != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputServersWide client method
        hResult = pDbgClient5->OutputServersWide(OutputControl, Machine, Flags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputServersWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputServersWide
            {
                dPrintf("%s OutputServersWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputServersWide

//******************************************************************************

HRESULT
GetOutputCallbacksWide
(
    PDEBUG_OUTPUT_CALLBACKS_WIDE* Callbacks
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Callbacks != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOutputCallbacksWide client method
        hResult = pDbgClient5->GetOutputCallbacksWide(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOutputCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOutputCallbacksWide
            {
                dPrintf("%s GetOutputCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOutputCallbacksWide

//******************************************************************************

HRESULT
SetOutputCallbacksWide
(
    PDEBUG_OUTPUT_CALLBACKS_WIDE Callbacks
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Callbacks != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOutputCallbacksWide client method
        hResult = pDbgClient5->SetOutputCallbacksWide(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOutputCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOutputCallbacksWide
            {
                dPrintf("%s SetOutputCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOutputCallbacksWide

//******************************************************************************

HRESULT
GetOutputLinePrefixWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              PrefixSize
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetOutputLinePrefixWide client method
        hResult = pDbgClient5->GetOutputLinePrefixWide(Buffer, BufferSize, PrefixSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetOutputLinePrefixWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetOutputLinePrefixWide
            {
                dPrintf("%s GetOutputLinePrefixWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetOutputLinePrefixWide

//******************************************************************************

HRESULT
SetOutputLinePrefixWide
(
    PCWSTR              Prefix
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetOutputLinePrefixWide client method
        hResult = pDbgClient5->SetOutputLinePrefixWide(Prefix);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetOutputLinePrefixWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetOutputLinePrefixWide
            {
                dPrintf("%s SetOutputLinePrefixWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetOutputLinePrefixWide

//******************************************************************************

HRESULT
GetIdentityWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              IdentitySize
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetIdentityWide client method
        hResult = pDbgClient5->GetIdentityWide(Buffer, BufferSize, IdentitySize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetIdentityWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetIdentityWide
            {
                dPrintf("%s GetIdentityWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetIdentityWide

//******************************************************************************

HRESULT
OutputIdentityWide
(
    ULONG               OutputControl,
    ULONG               Flags,
    PCWSTR              Format
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Format != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the OutputIdentityWide client method
        hResult = pDbgClient5->OutputIdentityWide(OutputControl, Flags, Format);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s OutputIdentityWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed OutputIdentityWide
            {
                dPrintf("%s OutputIdentityWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // OutputIdentityWide

//******************************************************************************

HRESULT
GetEventCallbacksWide
(
    PDEBUG_EVENT_CALLBACKS_WIDE* Callbacks
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Callbacks != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetEventCallbacksWide client method
        hResult = pDbgClient5->GetEventCallbacksWide(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetEventCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetEventCallbacksWide
            {
                dPrintf("%s GetEventCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetEventCallbacksWide

//******************************************************************************

HRESULT
SetEventCallbacksWide
(
    PDEBUG_EVENT_CALLBACKS_WIDE Callbacks
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Callbacks != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetEventCallbacksWide client method
        hResult = pDbgClient5->SetEventCallbacksWide(Callbacks);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetEventCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetEventCallbacksWide
            {
                dPrintf("%s SetEventCallbacksWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetEventCallbacksWide

//******************************************************************************

HRESULT
CreateProcess2
(
    ULONG64             Server,
    PSTR                CommandLine,
    PVOID               OptionsBuffer,
    ULONG               OptionsBufferSize,
    PCSTR               InitialDirectory,
    PCSTR               Environment
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(InitialDirectory != NULL);
    assert(Environment != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcess2 client method
        hResult = pDbgClient5->CreateProcess2(Server, CommandLine, OptionsBuffer, OptionsBufferSize, InitialDirectory, Environment);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcess2 %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcess2
            {
                dPrintf("%s CreateProcess2 %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcess2

//******************************************************************************

HRESULT
CreateProcess2Wide
(
    ULONG64             Server,
    PWSTR               CommandLine,
    PVOID               OptionsBuffer,
    ULONG               OptionsBufferSize,
    PCWSTR              InitialDirectory,
    PCWSTR              Environment
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(CommandLine != NULL);
    assert(OptionsBuffer != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcess2Wide client method
        hResult = pDbgClient5->CreateProcess2Wide(Server, CommandLine, OptionsBuffer, OptionsBufferSize, InitialDirectory, Environment);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcess2Wide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcess2Wide
            {
                dPrintf("%s CreateProcess2Wide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcess2Wide

//******************************************************************************

HRESULT
CreateProcessAndAttach2
(
    ULONG64             Server,
    PSTR                CommandLine,
    PVOID               OptionsBuffer,
    ULONG               OptionsBufferSize,
    PCSTR               InitialDirectory,
    PCSTR               Environment,
    ULONG               ProcessId,
    ULONG               AttachFlags
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(OptionsBuffer != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcessAndAttach2 client method
        hResult = pDbgClient5->CreateProcessAndAttach2(Server, CommandLine, OptionsBuffer, OptionsBufferSize, InitialDirectory, Environment, ProcessId, AttachFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcessAndAttach2 %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcessAndAttach2
            {
                dPrintf("%s CreateProcessAndAttach2 %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcessAndAttach2

//******************************************************************************

HRESULT
CreateProcessAndAttach2Wide
(
    ULONG64             Server,
    PWSTR               CommandLine,
    PVOID               OptionsBuffer,
    ULONG               OptionsBufferSize,
    PCWSTR              InitialDirectory,
    PCWSTR              Environment,
    ULONG               ProcessId,
    ULONG               AttachFlags
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(OptionsBuffer != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the CreateProcessAndAttach2Wide client method
        hResult = pDbgClient5->CreateProcessAndAttach2Wide(Server, CommandLine, OptionsBuffer, OptionsBufferSize, InitialDirectory, Environment, ProcessId, AttachFlags);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s CreateProcessAndAttach2Wide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed CreateProcessAndAttach2Wide
            {
                dPrintf("%s CreateProcessAndAttach2Wide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // CreateProcessAndAttach2Wide

//******************************************************************************

HRESULT
PushOutputLinePrefix
(
    PCSTR               NewPrefix,
    PULONG64            Handle
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Handle != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the PushOutputLinePrefix client method
        hResult = pDbgClient5->PushOutputLinePrefix(NewPrefix, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s PushOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed PushOutputLinePrefix
            {
                dPrintf("%s PushOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // PushOutputLinePrefix

//******************************************************************************

HRESULT
PushOutputLinePrefixWide
(
    PCWSTR              NewPrefix,
    PULONG64            Handle
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Handle != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the PushOutputLinePrefixWide client method
        hResult = pDbgClient5->PushOutputLinePrefixWide(NewPrefix, Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s PushOutputLinePrefixWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed PushOutputLinePrefixWide
            {
                dPrintf("%s PushOutputLinePrefixWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // PushOutputLinePrefixWide

//******************************************************************************

HRESULT
PopOutputLinePrefix
(
    ULONG64             Handle
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the PopOutputLinePrefix client method
        hResult = pDbgClient5->PopOutputLinePrefix(Handle);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s PopOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed PopOutputLinePrefix
            {
                dPrintf("%s PopOutputLinePrefix %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // PopOutputLinePrefix

//******************************************************************************

HRESULT
GetNumberInputCallbacks
(
    PULONG              Count
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Count != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberInputCallbacks client method
        hResult = pDbgClient5->GetNumberInputCallbacks(Count);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberInputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberInputCallbacks
            {
                dPrintf("%s GetNumberInputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberInputCallbacks

//******************************************************************************

HRESULT
GetNumberOutputCallbacks
(
    PULONG              Count
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Count != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberOutputCallbacks client method
        hResult = pDbgClient5->GetNumberOutputCallbacks(Count);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberOutputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberOutputCallbacks
            {
                dPrintf("%s GetNumberOutputCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberOutputCallbacks

//******************************************************************************

HRESULT
GetNumberEventCallbacks
(
    ULONG               EventFlags,
    PULONG              Count
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(Count != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetNumberEventCallbacks client method
        hResult = pDbgClient5->GetNumberEventCallbacks(EventFlags, Count);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetNumberEventCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetNumberEventCallbacks
            {
                dPrintf("%s GetNumberEventCallbacks %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetNumberEventCallbacks

//******************************************************************************

HRESULT
GetQuitLockString
(
    PSTR                Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetQuitLockString client method
        hResult = pDbgClient5->GetQuitLockString(Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetQuitLockString %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetQuitLockString
            {
                dPrintf("%s GetQuitLockString %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetQuitLockString

//******************************************************************************

HRESULT
SetQuitLockString
(
    PCSTR               String
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(String != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetQuitLockString client method
        hResult = pDbgClient5->SetQuitLockString(String);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetQuitLockString %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetQuitLockString
            {
                dPrintf("%s SetQuitLockString %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetQuitLockString

//******************************************************************************

HRESULT
GetQuitLockStringWide
(
    PWSTR               Buffer,
    ULONG               BufferSize,
    PULONG              StringSize
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetQuitLockStringWide client method
        hResult = pDbgClient5->GetQuitLockStringWide(Buffer, BufferSize, StringSize);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s GetQuitLockStringWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed GetQuitLockStringWide
            {
                dPrintf("%s GetQuitLockStringWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // GetQuitLockStringWide

//******************************************************************************

HRESULT
SetQuitLockStringWide
(
    PCWSTR              String
)
{
    PDEBUG_CLIENT5      pDbgClient5 = debugClient5Interface();
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = E_NOINTERFACE;

    assert(String != NULL);

    // Check for valid debug client interface
    if (pDbgClient5 != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetQuitLockStringWide client method
        hResult = pDbgClient5->SetQuitLockStringWide(String);

        // Release the debug interface
        releaseDebugInterface();

        // Check the progress indicator
        progressCheck();

        // Display DbgEng client interface if requested
        if (VERBOSE_LEVEL(VERBOSE_DBGENG_CLIENT))
        {
            // Clear verbose value so nested tracing calls don't happen
            setCommandValue(VerboseOption, 0);

            if (SUCCEEDED(hResult))
            {
                dPrintf("%s SetQuitLockStringWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", GREEN)), hResult);
            }
            else    // Failed SetQuitLockStringWide
            {
                dPrintf("%s SetQuitLockStringWide %s = 0x%08x\n", DML(bold("DbgClient:")), DML(foreground("hResult", RED)), hResult);
            }
            // Restore verbose value so tracing calls resume
            setCommandValue(VerboseOption, ulVerboseValue);
        }
    }
    return breakCheck(hResult);

} // SetQuitLockStringWide

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
