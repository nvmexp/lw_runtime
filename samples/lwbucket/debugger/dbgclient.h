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
|*  Module: dbgclient.h                                                       *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGCLIENT_H
#define _DBGCLIENT_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
// Debugger Interface wrappers
//
//******************************************************************************
// Debugger Client Interface
HRESULT                 AttachKernel(ULONG Flags, PCSTR ConnectOptions);
HRESULT                 GetKernelConnectionOptions(PSTR Buffer, ULONG BufferSize, PULONG OptionsSize);
HRESULT                 SetKernelConnectionOptions(PCSTR Options);
HRESULT                 StartProcessServer(ULONG Flags, PCSTR Options, PVOID Reserved);
HRESULT                 ConnectProcessServer(PCSTR RemoteOptions, PULONG64 Server);
HRESULT                 DisconnectProcessServer(ULONG64 Server);
HRESULT                 GetRunningProcessSystemIds(ULONG64 Server, PULONG Ids, ULONG Count, PULONG ActualCount);
HRESULT                 GetRunningProcessSystemIdByExelwtableName(ULONG64 Server, PCSTR ExeName, ULONG Flags, PULONG Id);
HRESULT                 GetRunningProcessDescription(ULONG64 Server, ULONG SystemId, ULONG Flags, PSTR ExeName, ULONG ExeNameSize, PULONG ActualExeNameSize, PSTR Description, ULONG DescriptionSize, PULONG ActualDescriptionSize);
HRESULT                 AttachProcess(ULONG64 Server, ULONG ProcessId, ULONG AttachFlags);
HRESULT                 CreateProcessClient(ULONG64 Server, PSTR CommandLine, ULONG CreateFlags);
HRESULT                 CreateProcessAndAttach(ULONG64 Server, PSTR CommandLine, ULONG CreateFlags, ULONG ProcessId, ULONG AttachFlags);
HRESULT                 GetProcessOptions(PULONG Options);
HRESULT                 AddProcessOptions(ULONG Options);
HRESULT                 RemoveProcessOptions(ULONG Options);
HRESULT                 SetProcessOptions(ULONG Options);
HRESULT                 OpenDumpFile(PCSTR DumpFile);
HRESULT                 WriteDumpFile(PCSTR DumpFile, ULONG Qualifier);
HRESULT                 ConnectSession(ULONG Flags, ULONG HistoryLimit);
HRESULT                 StartServer(PCSTR Options);
HRESULT                 OutputServers(ULONG OutputControl, PCSTR Machine, ULONG Flags);
HRESULT                 TerminateProcesses();
HRESULT                 DetachProcesses();
HRESULT                 EndSession(ULONG Flags);
HRESULT                 GetExitCode(PULONG Code);
HRESULT                 DispatchCallbacks(ULONG Timeout);
HRESULT                 ExitDispatch(PDEBUG_CLIENT Client);
HRESULT                 CreateClient(PDEBUG_CLIENT* Client);
HRESULT                 GetInputCallbacks(PDEBUG_INPUT_CALLBACKS* Callbacks);
HRESULT                 SetInputCallbacks(PDEBUG_INPUT_CALLBACKS Callbacks);
HRESULT                 GetOutputCallbacks(PDEBUG_OUTPUT_CALLBACKS* Callbacks);
HRESULT                 SetOutputCallbacks(PDEBUG_OUTPUT_CALLBACKS Callbacks);
HRESULT                 GetOutputMask(PULONG Mask);
HRESULT                 SetOutputMask(ULONG Mask);
HRESULT                 GetOtherOutputMask(PDEBUG_CLIENT Client, PULONG Mask);
HRESULT                 SetOtherOutputMask(PDEBUG_CLIENT Client, ULONG Mask);
HRESULT                 GetOutputWidth(PULONG Columns);
HRESULT                 SetOutputWidth(ULONG Columns);
HRESULT                 GetOutputLinePrefix(PSTR Buffer, ULONG BufferSize, PULONG PrefixSize);
HRESULT                 SetOutputLinePrefix(PCSTR Prefix);
HRESULT                 GetIdentity(PSTR Buffer, ULONG BufferSize, PULONG IdentitySize);
HRESULT                 OutputIdentity(ULONG OutputControl, ULONG Flags, PCSTR Format);
HRESULT                 GetEventCallbacks(PDEBUG_EVENT_CALLBACKS* Callbacks);
HRESULT                 SetEventCallbacks(PDEBUG_EVENT_CALLBACKS Callbacks);
HRESULT                 FlushCallbacks();
// Debugger Client 2 Interface
HRESULT                 WriteDumpFile2(PCSTR DumpFile, ULONG Qualifier, ULONG FormatFlags, PCSTR Comment);
HRESULT                 AddDumpInformationFile(PCSTR InfoFile, ULONG Type);
HRESULT                 EndProcessServer(ULONG64 Server);
HRESULT                 WaitForProcessServerEnd(ULONG Timeout);
HRESULT                 IsKernelDebuggerEnabled();
HRESULT                 TerminateLwrrentProcess();
HRESULT                 DetachLwrrentProcess();
HRESULT                 AbandonLwrrentProcess();
// Debugger Client 3 Interface
HRESULT                 GetRunningProcessSystemIdByExelwtableNameWide(ULONG64 Server, PCWSTR ExeName, ULONG Flags, PULONG Id);
HRESULT                 GetRunningProcessDescriptionWide(ULONG64 Server, ULONG SystemId, ULONG Flags, PWSTR ExeName, ULONG ExeNameSize, PULONG ActualExeNameSize, PWSTR Description, ULONG DescriptionSize, PULONG ActualDescriptionSize);
HRESULT                 CreateProcessWide(ULONG64 Server, PWSTR CommandLine, ULONG CreateFlags);
HRESULT                 CreateProcessAndAttachWide(ULONG64 Server, PWSTR CommandLine, ULONG CreateFlags, ULONG ProcessId, ULONG AttachFlags);
// Debugger Client 4 Interface
HRESULT                 OpenDumpFileWide(PCWSTR FileName, ULONG64 FileHandle);
HRESULT                 WriteDumpFileWide(PCWSTR FileName, ULONG64 FileHandle, ULONG Qualifier, ULONG FormatFlags, PCWSTR Comment);
HRESULT                 AddDumpInformationFileWide(PCWSTR FileName, ULONG64 FileHandle, ULONG Type);
HRESULT                 GetNumberDumpFiles(PULONG Number);
HRESULT                 GetDumpFile(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG NameSize, PULONG64 Handle, PULONG Type);
HRESULT                 GetDumpFileWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG NameSize, PULONG64 Handle, PULONG Type);
// Debugger Client 5 Interface
HRESULT                 AttachKernelWide(ULONG Flags, PCWSTR ConnectOptions);
HRESULT                 GetKernelConnectionOptionsWide(PWSTR Buffer, ULONG BufferSize, PULONG OptionsSize);
HRESULT                 SetKernelConnectionOptionsWide(PCWSTR Options);
HRESULT                 StartProcessServerWide(ULONG Flags, PCWSTR Options, PVOID Reserved);
HRESULT                 ConnectProcessServerWide(PCWSTR RemoteOptions, PULONG64 Server);
HRESULT                 StartServerWide(PCWSTR Options);
HRESULT                 OutputServersWide(ULONG OutputControl, PCWSTR Machine, ULONG Flags);
HRESULT                 GetOutputCallbacksWide(PDEBUG_OUTPUT_CALLBACKS_WIDE* Callbacks);
HRESULT                 SetOutputCallbacksWide(PDEBUG_OUTPUT_CALLBACKS_WIDE Callbacks);
HRESULT                 GetOutputLinePrefixWide(PWSTR Buffer, ULONG BufferSize, PULONG PrefixSize);
HRESULT                 SetOutputLinePrefixWide(PCWSTR Prefix);
HRESULT                 GetIdentityWide(PWSTR Buffer, ULONG BufferSize, PULONG IdentitySize);
HRESULT                 OutputIdentityWide(ULONG OutputControl, ULONG Flags, PCWSTR Format);
HRESULT                 GetEventCallbacksWide(PDEBUG_EVENT_CALLBACKS_WIDE* Callbacks);
HRESULT                 SetEventCallbacksWide(PDEBUG_EVENT_CALLBACKS_WIDE Callbacks);
HRESULT                 CreateProcess2(ULONG64 Server, PSTR CommandLine, PVOID OptionsBuffer, ULONG OptionsBufferSize, PCSTR InitialDirectory, PCSTR Environment);
HRESULT                 CreateProcess2Wide(ULONG64 Server, PWSTR CommandLine, PVOID OptionsBuffer, ULONG OptionsBufferSize, PCWSTR InitialDirectory, PCWSTR Environment);
HRESULT                 CreateProcessAndAttach2(ULONG64 Server, PSTR CommandLine, PVOID OptionsBuffer, ULONG OptionsBufferSize, PCSTR InitialDirectory, PCSTR Environment, ULONG ProcessId, ULONG AttachFlags);
HRESULT                 CreateProcessAndAttach2Wide(ULONG64 Server, PWSTR CommandLine, PVOID OptionsBuffer, ULONG OptionsBufferSize, PCWSTR InitialDirectory, PCWSTR Environment, ULONG ProcessId, ULONG AttachFlags);
HRESULT                 PushOutputLinePrefix(PCSTR NewPrefix, PULONG64 Handle);
HRESULT                 PushOutputLinePrefixWide(PCWSTR NewPrefix, PULONG64 Handle);
HRESULT                 PopOutputLinePrefix(ULONG64 Handle);
HRESULT                 GetNumberInputCallbacks(PULONG Count);
HRESULT                 GetNumberOutputCallbacks(PULONG Count);
HRESULT                 GetNumberEventCallbacks(ULONG EventFlags, PULONG Count);
HRESULT                 GetQuitLockString(PSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 SetQuitLockString(PCSTR String);
HRESULT                 GetQuitLockStringWide(PWSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 SetQuitLockStringWide(PCWSTR String);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGCLIENT_H
