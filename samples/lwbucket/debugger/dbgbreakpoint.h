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
|*  Module: dbgbreakpoint.h                                                   *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGBREAKPOINT_H
#define _DBGBREAKPOINT_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger Breakpoint Interface
HRESULT                 GetId(PULONG Id);
HRESULT                 GetType(PULONG BreakType, PULONG ProcType);
HRESULT                 GetAdder(PDEBUG_CLIENT* Adder);
HRESULT                 GetFlags(PULONG Flags);
HRESULT                 AddFlags(ULONG Flags);
HRESULT                 RemoveFlags(ULONG Flags);
HRESULT                 SetFlags(ULONG Flags);
HRESULT                 GetOffset(PULONG64 Offset);
HRESULT                 SetOffset(ULONG64 Offset);
HRESULT                 GetDataParameters(PULONG Size, PULONG AccessType);
HRESULT                 SetDataParameters(ULONG Size, ULONG AccessType);
HRESULT                 GetPassCount(PULONG Count);
HRESULT                 SetPassCount(ULONG Count);
HRESULT                 GetLwrrentPassCount(PULONG Count);
HRESULT                 GetMatchThreadId(PULONG Thread);
HRESULT                 SetMatchThreadId(ULONG Thread);
HRESULT                 GetCommand(PSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetCommand(PSTR Buffer);
HRESULT                 GetOffsetExpression(PSTR Buffer, ULONG BufferSize, PULONG ExpressionSize);
HRESULT                 SetOffsetExpression(PCSTR Expression);
HRESULT                 GetParameters(PDEBUG_BREAKPOINT_PARAMETERS Params);
// Debugger Breakpoint 2 Interface
HRESULT                 GetCommandWide(PWSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetCommandWide(PCWSTR Command);
HRESULT                 GetOffsetExpressionWide(PWSTR Buffer, ULONG BufferSize, PULONG ExpressionSize);
HRESULT                 SetOffsetExpressionWide(PCWSTR Expression);
// Debugger Breakpoint 3 Interface
HRESULT                 GetGuid(LPGUID Guid);

// Debugger Breakpoint Wrappers
HRESULT                 GetType(ULONG Id, PULONG BreakType, PULONG ProcType);
HRESULT                 GetAdder(ULONG Id, PDEBUG_CLIENT* Adder);
HRESULT                 GetFlags(ULONG Id, PULONG Flags);
HRESULT                 AddFlags(ULONG Id, ULONG Flags);
HRESULT                 RemoveFlags(ULONG Id, ULONG Flags);
HRESULT                 SetFlags(ULONG Id, ULONG Flags);
HRESULT                 GetOffset(ULONG Id, PULONG64 Offset);
HRESULT                 SetOffset(ULONG Id, ULONG64 Offset);
HRESULT                 GetDataParameters(ULONG Id, PULONG Size, PULONG AccessType);
HRESULT                 SetDataParameters(ULONG Id, ULONG Size, ULONG AccessType);
HRESULT                 GetPassCount(ULONG Id, PULONG Count);
HRESULT                 SetPassCount(ULONG Id, ULONG Count);
HRESULT                 GetLwrrentPassCount(ULONG Id, PULONG Count);
HRESULT                 GetMatchThreadId(ULONG Id, PULONG Thread);
HRESULT                 SetMatchThreadId(ULONG Id, ULONG Thread);
HRESULT                 GetCommand(ULONG Id, PSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetCommand(ULONG Id, PSTR Buffer);
HRESULT                 GetOffsetExpression(ULONG Id, PSTR Buffer, ULONG BufferSize, PULONG ExpressionSize);
HRESULT                 SetOffsetExpression(ULONG Id, PCSTR Expression);
HRESULT                 GetParameters(ULONG Id, PDEBUG_BREAKPOINT_PARAMETERS Params);
// Debugger Breakpoint 2 Wrappers
HRESULT                 GetCommandWide(ULONG Id, PWSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetCommandWide(ULONG Id, PCWSTR Command);
HRESULT                 GetOffsetExpressionWide(ULONG Id, PWSTR Buffer, ULONG BufferSize, PULONG ExpressionSize);
HRESULT                 SetOffsetExpressionWide(ULONG Id, PCWSTR Expression);
// Debugger Breakpoint 3 Wrappers
HRESULT                 GetGuid(ULONG Id, LPGUID Guid);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGBREAKPOINT_H
