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
|*  Module: dbgsymbolgroup.h                                                  *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGSYMBOLGROUP_H
#define _DBGSYMBOLGROUP_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger Symbol Group Interface
HRESULT                 GetNumberSymbols(PULONG Number);
HRESULT                 AddSymbol(PCSTR Name, PULONG Index);
HRESULT                 RemoveSymbolByName(PCSTR Name);
HRESULT                 RemoveSymbolByIndex(ULONG Index);
HRESULT                 GetSymbolName(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 GetSymbolParameters(ULONG Start, ULONG Count, PDEBUG_SYMBOL_PARAMETERS Params);
HRESULT                 ExpandSymbol(ULONG Index, BOOL Expand);
HRESULT                 OutputSymbols(ULONG OutputControl, ULONG Flags, ULONG Start, ULONG Count);
HRESULT                 WriteSymbol(ULONG Index, PCSTR Value);
HRESULT                 OutputAsType(ULONG Index, PCSTR Type);
// Debugger Symbol Group 2 Interface
HRESULT                 AddSymbolWide(PCWSTR Name, PULONG Index);
HRESULT                 RemoveSymbolByNameWide(PCWSTR Name);
HRESULT                 GetSymbolNameWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 WriteSymbolWide(ULONG Index, PCWSTR Value);
HRESULT                 OutputAsTypeWide(ULONG Index, PCWSTR Type);
HRESULT                 GetSymbolTypeName(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 GetSymbolTypeNameWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 GetSymbolSize(ULONG Index, PULONG Size);
HRESULT                 GetSymbolOffset(ULONG Index, PULONG64 Offset);
HRESULT                 GetSymbolRegister(ULONG Index, PULONG Register);
HRESULT                 GetSymbolValueText(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 GetSymbolValueTextWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 GetSymbolEntryInformation(ULONG Index, PDEBUG_SYMBOL_ENTRY Entry);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGSYMBOLGROUP_H
