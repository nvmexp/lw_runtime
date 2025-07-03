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
|*  Module: dbgdataspaces.h                                                   *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGDATASPACES_H
#define _DBGDATASPACES_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger Data Spaces Interface
HRESULT                 ReadVirtual(ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WriteVirtual(ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 SearchVirtual(ULONG64 Offset, ULONG64 Length, PVOID Pattern, ULONG PatternSize, ULONG PatternGranularity, PULONG64 MatchOffset);
HRESULT                 ReadVirtualUncached(ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WriteVirtualUncached(ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 ReadPointersVirtual(ULONG Count, ULONG64 Offset, PULONG64 Ptrs);
HRESULT                 WritePointersVirtual(ULONG Count, ULONG64 Offset, PULONG64 Ptrs);
HRESULT                 ReadPhysical(ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WritePhysical(ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 ReadControl(ULONG Processor, ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WriteControl(ULONG Processor, ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 ReadIo(ULONG InterfaceType, ULONG BusNumber, ULONG AddressSpace, ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WriteIo(ULONG InterfaceType, ULONG BusNumber, ULONG AddressSpace, ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 ReadMsr(ULONG Msr, PULONG64 Value);
HRESULT                 WriteMsr(ULONG Msr, ULONG64 Value);
HRESULT                 ReadBusData(ULONG BusDataType, ULONG BusNumber, ULONG SlotNumber, ULONG Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WriteBusData(ULONG BusDataType, ULONG BusNumber, ULONG SlotNumber, ULONG Offset, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 CheckLowMemory();
HRESULT                 ReadDebuggerData(ULONG Index, PVOID Buffer, ULONG BufferSize, PULONG DataSize);
HRESULT                 ReadProcessorSystemData(ULONG Processor, ULONG Index, PVOID Buffer, ULONG BufferSize, PULONG DataSize);
// Debugger Data Spaces 2 Interface
HRESULT                 VirtualToPhysical(ULONG64 Virtual, PULONG64 Physical);
HRESULT                 GetVirtualTranslationPhysicalOffsets(ULONG64 Virtual, PULONG64 Offsets, ULONG OffsetsSize, PULONG Levels);
HRESULT                 ReadHandleData(ULONG64 Handle, ULONG DataType, PVOID Buffer, ULONG BufferSize, PULONG DataSize);
HRESULT                 FillVirtual(ULONG64 Start, ULONG Size, PVOID Pattern, ULONG PatternSize, PULONG Filled);
HRESULT                 FillPhysical(ULONG64 Start, ULONG Size, PVOID Pattern, ULONG PatternSize, PULONG Filled);
HRESULT                 QueryVirtual(ULONG64 Offset, PMEMORY_BASIC_INFORMATION64 Info);
// Debugger Data Spaces 3 Interface
HRESULT                 ReadImageNtHeaders(ULONG64 ImageBase, PIMAGE_NT_HEADERS64 Headers);
HRESULT                 ReadTagged(LPGUID Tag, ULONG Offset, PVOID Buffer, ULONG BufferSize, PULONG TotalSize);
HRESULT                 StartEnumTagged(PULONG64 Handle);
HRESULT                 GetNextTagged(ULONG64 Handle, LPGUID Tag, PULONG Size);
HRESULT                 EndEnumTagged(ULONG64 Handle);
// Debugger Data Spaces 4 Interface
HRESULT                 GetOffsetInformation(ULONG Space, ULONG Which, ULONG64 Offset, PVOID Buffer, ULONG BufferSize, PULONG InfoSize);
HRESULT                 GetNextDifferentlyValidOffsetVirtual(ULONG64 Offset, PULONG64 NextOffset);
HRESULT                 GetValidRegiolwirtual(ULONG64 Base, ULONG Size, PULONG64 ValidBase, PULONG ValidSize);
HRESULT                 SearchVirtual2(ULONG64 Offset, ULONG64 Length, ULONG Flags, PVOID Pattern, ULONG PatternSize, ULONG PatternGranularity, PULONG64 MatchOffset);
HRESULT                 ReadMultiByteStringVirtual(ULONG64 Offset, ULONG MaxBytes, PSTR Buffer, ULONG BufferSize, PULONG StringBytes);
HRESULT                 ReadMultiByteStringVirtualWide(ULONG64 Offset, ULONG MaxBytes, ULONG CodePage, PWSTR Buffer, ULONG BufferSize, PULONG StringBytes);
HRESULT                 ReadUnicodeStringVirtual(ULONG64 Offset, ULONG MaxBytes, ULONG CodePage, PSTR Buffer, ULONG BufferSize, PULONG StringBytes);
HRESULT                 ReadUnicodeStringVirtualWide(ULONG64 Offset, ULONG MaxBytes, PWSTR Buffer, ULONG BufferSize, PULONG StringBytes);
HRESULT                 ReadPhysical2(ULONG64 Offset, ULONG Flags, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WritePhysical2(ULONG64 Offset, ULONG Flags, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGDATASPACES_H
