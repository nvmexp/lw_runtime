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
|*  Module: dbgregisters.h                                                    *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGREGISTERS_H
#define _DBGREGISTERS_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger Registers Interface
HRESULT                 GetNumberRegisters(PULONG Number);
HRESULT                 GetDescription(ULONG Register, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PDEBUG_REGISTER_DESCRIPTION Desc);
HRESULT                 GetIndexByName(PCSTR Name, PULONG Index);
HRESULT                 GetValue(ULONG Register, PDEBUG_VALUE Value);
HRESULT                 SetValue(ULONG Register, PDEBUG_VALUE Value);
HRESULT                 GetValues(ULONG Count, PULONG Indices, ULONG Start, PDEBUG_VALUE Values);
HRESULT                 SetValues(ULONG Count, PULONG Indices, ULONG Start, PDEBUG_VALUE Values);
HRESULT                 OutputRegisters(ULONG OutputControl, ULONG Flags);
HRESULT                 GetInstructionOffset(PULONG64 Offset);
HRESULT                 GetStackOffset(PULONG64 Offset);
HRESULT                 GetFrameOffset(PULONG64 Offset);
// Debugger Registers 2 Interface
HRESULT                 GetDescriptionWide(ULONG Register, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PDEBUG_REGISTER_DESCRIPTION Desc);
HRESULT                 GetIndexByNameWide(PCWSTR Name, PULONG Index);
HRESULT                 GetNumberPseudoRegisters(PULONG Number);
HRESULT                 GetPseudoDescription(ULONG Register, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 TypeModule, PULONG TypeId);
HRESULT                 GetPseudoDescriptionWide(ULONG Register, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 TypeModule, PULONG TypeId);
HRESULT                 GetPseudoIndexByName(PCSTR Name, PULONG Index);
HRESULT                 GetPseudoIndexByNameWide(PCWSTR Name, PULONG Index);
HRESULT                 GetPseudoValues(ULONG Source, ULONG Count, PULONG Indices, ULONG Start, PDEBUG_VALUE Values);
HRESULT                 SetPseudoValues(ULONG Source, ULONG Count, PULONG Indices, ULONG Start, PDEBUG_VALUE Values);
HRESULT                 GetValues2(ULONG Source, ULONG Count, PULONG Indices, ULONG Start, PDEBUG_VALUE Values);
HRESULT                 SetValues2(ULONG Source, ULONG Count, PULONG Indices, ULONG Start, PDEBUG_VALUE Values);
HRESULT                 OutputRegisters2(ULONG OutputControl, ULONG Source, ULONG Flags);
HRESULT                 GetInstructionOffset2(ULONG Source, PULONG64 Offset);
HRESULT                 GetStackOffset2(ULONG Source, PULONG64 Offset);
HRESULT                 GetFrameOffset2(ULONG Source, PULONG64 Offset);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGREGISTERS_H
