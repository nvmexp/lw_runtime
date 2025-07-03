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
|*  Module: dbgsystemobjects.h                                                *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGSYSTEMOBJECTS_H
#define _DBGSYSTEMOBJECTS_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger System Objects Interface
HRESULT                 GetEventThread(PULONG Id);
HRESULT                 GetEventProcess(PULONG Id);
HRESULT                 GetLwrrentThreadId(PULONG Id);
HRESULT                 SetLwrrentThreadId(ULONG Id);
HRESULT                 GetLwrrentProcessId(PULONG Id);
HRESULT                 SetLwrrentProcessId(ULONG Id);
HRESULT                 GetNumberThreads(PULONG Number);
HRESULT                 GetTotalNumberThreads(PULONG Total, PULONG LargestProcess);
HRESULT                 GetThreadIdsByIndex(ULONG Start, ULONG Count, PULONG Ids, PULONG SysIds);
HRESULT                 GetThreadIdByProcessor(ULONG Processor, PULONG Id);
HRESULT                 GetLwrrentThreadDataOffset(PULONG64 Offset);
HRESULT                 GetThreadIdByDataOffset(ULONG64 Offset, PULONG Id);
HRESULT                 GetLwrrentThreadTeb(PULONG64 Offset);
HRESULT                 GetThreadIdByTeb(ULONG64 Offset, PULONG Id);
HRESULT                 GetLwrrentThreadSystemId(PULONG SysId);
HRESULT                 GetThreadIdBySystemId(ULONG SysId, PULONG Id);
HRESULT                 GetLwrrentThreadHandle(PULONG64 Handle);
HRESULT                 GetThreadIdByHandle(ULONG64 Handle, PULONG Id);
HRESULT                 GetNumberProcesses(PULONG Number);
HRESULT                 GetProcessIdsByIndex(ULONG Start, ULONG Count, PULONG Ids, PULONG SysIds);
HRESULT                 GetLwrrentProcessDataOffset(PULONG64 Offset);
HRESULT                 GetProcessIdByDataOffset(ULONG64 Offset, PULONG Id);
HRESULT                 GetLwrrentProcessPeb(PULONG64 Offset);
HRESULT                 GetProcessIdByPeb(ULONG64 Offset, PULONG Id);
HRESULT                 GetLwrrentProcessSystemId(PULONG SysId);
HRESULT                 GetProcessIdBySystemId(ULONG SysId, PULONG Id);
HRESULT                 GetLwrrentProcessHandle(PULONG64 Handle);
HRESULT                 GetProcessIdByHandle(ULONG64 Handle, PULONG Id);
HRESULT                 GetLwrrentProcessExelwtableName(PSTR Buffer, ULONG BufferSize, PULONG ExeSize);
// Debugger System Objects 2 Interface
HRESULT                 GetLwrrentProcessUpTime(PULONG UpTime);
HRESULT                 GetImplicitThreadDataOffset(PULONG64 Offset);
HRESULT                 SetImplicitThreadDataOffset(ULONG64 Offset);
HRESULT                 GetImplicitProcessDataOffset(PULONG64 Offset);
HRESULT                 SetImplicitProcessDataOffset(ULONG64 Offset);
// Debugger System Objects 3 Interface
HRESULT                 GetEventSystem(PULONG Id);
HRESULT                 GetLwrrentSystemId(PULONG Id);
HRESULT                 SetLwrrentSystemId(ULONG Id);
HRESULT                 GetNumberSystems(PULONG Number);
HRESULT                 GetSystemIdsByIndex(ULONG Start, ULONG Count, PULONG Ids);
HRESULT                 GetTotalNumberThreadsAndProcesses(PULONG TotalThreads, PULONG TotalProcesses, PULONG LargestProcessThreads, PULONG LargestSystemThreads, PULONG LargestSystemProcesses);
HRESULT                 GetLwrrentSystemServer(PULONG64 Server);
HRESULT                 GetSystemByServer(ULONG64 Server, PULONG Id);
HRESULT                 GetLwrrentSystemServerName(PSTR Buffer, ULONG BufferSize, PULONG NameSize);
// Debugger System Objects 4 Interface
HRESULT                 GetLwrrentProcessExelwtableNameWide(PWSTR Buffer, ULONG BufferSize, PULONG ExeSize);
HRESULT                 GetLwrrentSystemServerNameWide(PWSTR Buffer, ULONG BufferSize, PULONG NameSize);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGSYSTEMOBJECTS_H
