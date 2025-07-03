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
|*  Module: dbgadvanced.h                                                     *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGADVANCED_H
#define _DBGADVANCED_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger Advanced Interface
HRESULT                 GetThreadContext(PVOID Context, ULONG ContextSize);
HRESULT                 SetThreadContext(PVOID Context, ULONG ContextSize);
// Debugger Advanced 2 Interface
HRESULT                 Request(ULONG Request, PVOID InBuffer, ULONG InBufferSize, PVOID OutBuffer, ULONG OutBufferSize, PULONG OutSize);
HRESULT                 GetSourceFileInformation(ULONG Which, PSTR SourceFile, ULONG64 Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize);
HRESULT                 FindSourceFileAndToken(ULONG StartElement, ULONG64 ModAddr, PCSTR File, ULONG Flags, PVOID FileToken, ULONG FileTokenSize, PULONG FoundElement, PSTR Buffer, ULONG BufferSize, PULONG FoundSize);
HRESULT                 GetSymbolInformation(ULONG Which, ULONG64 Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize, PSTR StringBuffer, ULONG StringBufferSize, PULONG StringSize);
HRESULT                 GetSystemObjectInformation(ULONG Which, ULONG64 Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize);
// Debugger Advanced 3 Interface
HRESULT                 GetSourceFileInformationWide(ULONG Which, PWSTR SourceFile, ULONG64 Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize);
HRESULT                 FindSourceFileAndTokenWide(ULONG StartElement, ULONG64 ModAddr, PCWSTR File, ULONG Flags, PVOID FileToken, ULONG FileTokenSize, PULONG FoundElement, PWSTR Buffer, ULONG BufferSize, PULONG FoundSize);
HRESULT                 GetSymbolInformationWide(ULONG Which, ULONG64 Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize, PWSTR StringBuffer, ULONG StringBufferSize, PULONG StringSize);

// Debugger Advanced 2 Interface Wrappers
static inline HRESULT   GetSourceFileInformation(ULONG Which, PSTR SourceFile, POINTER Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize)
                            { return GetSourceFileInformation(Which, SourceFile, Arg64.ptr(), Arg32, Buffer, BufferSize, InfoSize); }
static inline HRESULT   FindSourceFileAndToken(ULONG StartElement, POINTER ModAddr, PCSTR File, ULONG Flags, PVOID FileToken, ULONG FileTokenSize, PULONG FoundElement, PSTR Buffer, ULONG BufferSize, PULONG FoundSize)
                            { return FindSourceFileAndToken(StartElement, ModAddr.ptr(), File, Flags, FileToken, FileTokenSize, FoundElement, Buffer, BufferSize, FoundSize); }
static inline HRESULT   GetSymbolInformation(ULONG Which, POINTER Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize, PSTR StringBuffer, ULONG StringBufferSize, PULONG StringSize)
                            { return GetSymbolInformation(Which, Arg64.ptr(), Arg32, Buffer, BufferSize, InfoSize, StringBuffer, StringBufferSize, StringSize); }
static inline HRESULT   GetSystemObjectInformation(ULONG Which, POINTER Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize)
                            { return GetSystemObjectInformation(Which, Arg64.ptr(), Arg32, Buffer, BufferSize, InfoSize); }
// Debugger Advanced 3 Interface Wrappers
static inline HRESULT   GetSourceFileInformationWide(ULONG Which, PWSTR SourceFile, POINTER Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize)
                            { return GetSourceFileInformationWide(Which, SourceFile, Arg64.ptr(), Arg32, Buffer, BufferSize, InfoSize); }
static inline HRESULT   FindSourceFileAndTokenWide(ULONG StartElement, POINTER ModAddr, PCWSTR File, ULONG Flags, PVOID FileToken, ULONG FileTokenSize, PULONG FoundElement, PWSTR Buffer, ULONG BufferSize, PULONG FoundSize)
                            { return FindSourceFileAndTokenWide(StartElement, ModAddr.ptr(), File, Flags, FileToken, FileTokenSize, FoundElement, Buffer, BufferSize, FoundSize); }
static inline HRESULT   GetSymbolInformationWide(ULONG Which, POINTER Arg64, ULONG Arg32, PVOID Buffer, ULONG BufferSize, PULONG InfoSize, PWSTR StringBuffer, ULONG StringBufferSize, PULONG StringSize)
                            { return GetSymbolInformationWide(Which, Arg64.ptr(), Arg32, Buffer, BufferSize, InfoSize, StringBuffer, StringBufferSize, StringSize); }

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGADVANCED_H
