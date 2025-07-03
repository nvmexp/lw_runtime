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
|*  Module: dbgsymbols.h                                                      *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGSYMBOLS_H
#define _DBGSYMBOLS_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger Symbols Interface
HRESULT                 GetSymbolOptions(PULONG Options);
HRESULT                 AddSymbolOptions(ULONG Options);
HRESULT                 RemoveSymbolOptions(ULONG Options);
HRESULT                 SetSymbolOptions(ULONG Options);
HRESULT                 GetNameByOffset(ULONG64 Offset, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement);
HRESULT                 GetOffsetByName(PCSTR Symbol, PULONG64 Offset);
HRESULT                 GetNearNameByOffset(ULONG64 Offset, LONG Delta, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement);
HRESULT                 GetLineByOffset(ULONG64 Offset, PULONG Line, PSTR FileBuffer, ULONG FileBufferSize, PULONG FileSize, PULONG64 Displacement);
HRESULT                 GetOffsetByLine(ULONG Line, PCSTR File, PULONG64 Offset);
HRESULT                 GetNumberModules(PULONG Loaded, PULONG Unloaded);
HRESULT                 GetModuleByIndex(ULONG Index, PULONG64 Base);
HRESULT                 GetModuleByModuleName(PCSTR Name, ULONG StartIndex, PULONG Index, PULONG64 Base);
HRESULT                 GetModuleByOffset(ULONG64 Offset, ULONG StartIndex, PULONG Index, PULONG64 Base);
HRESULT                 GetModuleNames(ULONG Index, ULONG64 Base, PSTR ImageNameBuffer, ULONG ImageNameBufferSize, PULONG ImageNameSize, PSTR ModuleNameBuffer, ULONG ModuleNameBufferSize, PULONG ModuleNameSize, PSTR LoadedImageNameBuffer, ULONG LoadedImageNameBufferSize, PULONG LoadedImageNameSize);
HRESULT                 GetModuleParameters(ULONG Count, PULONG64 Bases, ULONG Start, PDEBUG_MODULE_PARAMETERS Params);
HRESULT                 GetSymbolModule(PCSTR Symbol, PULONG64 Base);
HRESULT                 GetTypeName(ULONG64 Module, ULONG TypeId, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize);
HRESULT                 GetTypeId(ULONG64 Module, PCSTR Name, PULONG TypeId);
HRESULT                 GetTypeSize(ULONG64 Module, ULONG TypeId, PULONG Size);
HRESULT                 GetFieldOffset(ULONG64 Module, ULONG TypeId, PCSTR Field, PULONG Offset);
HRESULT                 GetSymbolTypeId(PCSTR Symbol, PULONG TypeId, PULONG64 Module);
HRESULT                 GetOffsetTypeId(ULONG64 Offset, PULONG TypeId, PULONG64 Module);
HRESULT                 ReadTypedDataVirtual(ULONG64 Offset, ULONG64 Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WriteTypedDataVirtual(ULONG64 Offset, ULONG64 Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 OutputTypedDataVirtual(ULONG OutputControl, ULONG64 Offset, ULONG64 Module, ULONG TypeId, ULONG Flags);
HRESULT                 ReadTypedDataPhysical(ULONG64 Offset, ULONG64 Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesRead);
HRESULT                 WriteTypedDataPhysical(ULONG64 Offset, ULONG64 Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten);
HRESULT                 OutputTypedDataPhysical(ULONG OutputControl, ULONG64 Offset, ULONG64 Module, ULONG TypeId, ULONG Flags);
HRESULT                 GetScope(PULONG64 InstructionOffset, PDEBUG_STACK_FRAME ScopeFrame, PVOID ScopeContext, ULONG ScopeContextSize);
HRESULT                 SetScope(ULONG64 InstructionOffset, PDEBUG_STACK_FRAME ScopeFrame, PVOID ScopeContext, ULONG ScopeContextSize);
HRESULT                 ResetScope();
HRESULT                 GetScopeSymbolGroup(ULONG Flags, PDEBUG_SYMBOL_GROUP Update, PDEBUG_SYMBOL_GROUP* Symbols);
HRESULT                 CreateSymbolGroup(PDEBUG_SYMBOL_GROUP* Group);
HRESULT                 StartSymbolMatch(PCSTR Pattern, PULONG64 Handle);
HRESULT                 GetNextSymbolMatch(ULONG64 Handle, PSTR Buffer, ULONG BufferSize, PULONG MatchSize, PULONG64 Offset);
HRESULT                 EndSymbolMatch(ULONG64 Handle);
HRESULT                 Reload(PCSTR Module);
HRESULT                 GetSymbolPath(PSTR Buffer, ULONG BufferSize, PULONG PathSize);
HRESULT                 SetSymbolPath(PCSTR Path);
HRESULT                 AppendSymbolPath(PCSTR Addition);
HRESULT                 GetImagePath(PSTR Buffer, ULONG BufferSize, PULONG PathSize);
HRESULT                 SetImagePath(PCSTR Path);
HRESULT                 AppendImagePath(PCSTR Addition);
HRESULT                 GetSourcePath(PSTR Buffer, ULONG BufferSize, PULONG PathSize);
HRESULT                 GetSourcePathElement(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG ElementSize);
HRESULT                 SetSourcePath(PCSTR Path);
HRESULT                 AppendSourcePath(PCSTR Addition);
HRESULT                 FindSourceFile(ULONG StartElement, PCSTR File, ULONG Flags, PULONG FoundElement, PSTR Buffer, ULONG BufferSize, PULONG FoundSize);
HRESULT                 GetSourceFileLineOffsets(PCSTR File, PULONG64 Buffer, ULONG BufferLines, PULONG FileLines);
// Debugger Symbols 2 Interface
HRESULT                 GetModuleVersionInformation(ULONG Index, ULONG64 Base, PCSTR Item, PVOID Buffer, ULONG BufferSize, PULONG VerInfoSize);
HRESULT                 GetModuleNameString(ULONG Which, ULONG Index, ULONG64 Base, PSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 GetConstantName(ULONG64 Module, ULONG TypeId, ULONG64 Value, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize);
HRESULT                 GetFieldName(ULONG64 Module, ULONG TypeId, ULONG FieldIndex, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize);
HRESULT                 GetTypeOptions(PULONG Options);
HRESULT                 AddTypeOptions(ULONG Options);
HRESULT                 RemoveTypeOptions(ULONG Options);
HRESULT                 SetTypeOptions(ULONG Options);
// Debugger Symbols 3 Interface
HRESULT                 GetNameByOffsetWide(ULONG64 Offset, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement);
HRESULT                 GetOffsetByNameWide(PCWSTR Symbol, PULONG64 Offset);
HRESULT                 GetNearNameByOffsetWide(ULONG64 Offset, LONG Delta, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement);
HRESULT                 GetLineByOffsetWide(ULONG64 Offset, PULONG Line, PWSTR FileBuffer, ULONG FileBufferSize, PULONG FileSize, PULONG64 Displacement);
HRESULT                 GetOffsetByLineWide(ULONG Line, PCWSTR File, PULONG64 Offset);
HRESULT                 GetModuleByModuleNameWide(PCWSTR Name, ULONG StartIndex, PULONG Index, PULONG64 Base);
HRESULT                 GetSymbolModuleWide(PCWSTR Symbol, PULONG64 Base);
HRESULT                 GetTypeNameWide(ULONG64 Module, ULONG TypeId, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize);
HRESULT                 GetTypeIdWide(ULONG64 Module, PCWSTR Name, PULONG TypeId);
HRESULT                 GetFieldOffsetWide(ULONG64 Module, ULONG TypeId, PCWSTR Field, PULONG Offset);
HRESULT                 GetSymbolTypeIdWide(PCWSTR Symbol, PULONG TypeId, PULONG64 Module);
HRESULT                 GetScopeSymbolGroup2(ULONG Flags, PDEBUG_SYMBOL_GROUP2 Update, PDEBUG_SYMBOL_GROUP2* Symbols);
HRESULT                 CreateSymbolGroup2(PDEBUG_SYMBOL_GROUP2* Group);
HRESULT                 StartSymbolMatchWide(PCWSTR Pattern, PULONG64 Handle);
HRESULT                 GetNextSymbolMatchWide(ULONG64 Handle, PWSTR Buffer, ULONG BufferSize, PULONG MatchSize, PULONG64 Offset);
HRESULT                 ReloadWide(PCWSTR Module);
HRESULT                 GetSymbolPathWide(PWSTR Buffer, ULONG BufferSize, PULONG PathSize);
HRESULT                 SetSymbolPathWide(PCWSTR Path);
HRESULT                 AppendSymbolPathWide(PCWSTR Addition);
HRESULT                 GetImagePathWide(PWSTR Buffer, ULONG BufferSize, PULONG PathSize);
HRESULT                 SetImagePathWide(PCWSTR Path);
HRESULT                 AppendImagePathWide(PCWSTR Addition);
HRESULT                 GetSourcePathWide(PWSTR Buffer, ULONG BufferSize, PULONG PathSize);
HRESULT                 GetSourcePathElementWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG ElementSize);
HRESULT                 SetSourcePathWide(PCWSTR Path);
HRESULT                 AppendSourcePathWide(PCWSTR Addition);
HRESULT                 FindSourceFileWide(ULONG StartElement, PCWSTR File, ULONG Flags, PULONG FoundElement, PWSTR Buffer, ULONG BufferSize, PULONG FoundSize);
HRESULT                 GetSourceFileLineOffsetsWide(PCWSTR File, PULONG64 Buffer, ULONG BufferLines, PULONG FileLines);
HRESULT                 GetModuleVersionInformationWide(ULONG Index, ULONG64 Base, PCWSTR Item, PVOID Buffer, ULONG BufferSize, PULONG VerInfoSize);
HRESULT                 GetModuleNameStringWide(ULONG Which, ULONG Index, ULONG64 Base, PWSTR Buffer, ULONG BufferSize, PULONG NameSize);
HRESULT                 GetConstantNameWide(ULONG64 Module, ULONG TypeId, ULONG64 Value, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize);
HRESULT                 GetFieldNameWide(ULONG64 Module, ULONG TypeId, ULONG FieldIndex, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize);
HRESULT                 IsManagedModule(ULONG Index, ULONG64 Base);
HRESULT                 GetModuleByModuleName2(PCSTR Name, ULONG StartIndex, ULONG Flags, PULONG Index, PULONG64 Base);
HRESULT                 GetModuleByModuleName2Wide(PCWSTR Name, ULONG StartIndex, ULONG Flags, PULONG Index, PULONG64 Base);
HRESULT                 GetModuleByOffset2(ULONG64 Offset, ULONG StartIndex, ULONG Flags, PULONG Index, PULONG64 Base);
HRESULT                 AddSyntheticModule(ULONG64 Base, ULONG Size, PCSTR ImagePath, PCSTR ModuleName, ULONG Flags);
HRESULT                 AddSyntheticModuleWide(ULONG64 Base, ULONG Size, PCWSTR ImagePath, PCWSTR ModuleName, ULONG Flags);
HRESULT                 RemoveSyntheticModule(ULONG64 Base);
HRESULT                 GetLwrrentScopeFrameIndex(PULONG Index);
HRESULT                 SetScopeFrameByIndex(ULONG Index);
HRESULT                 SetScopeFromJitDebugInfo(ULONG OutputControl, ULONG64 InfoOffset);
HRESULT                 SetScopeFromStoredEvent();
HRESULT                 OutputSymbolByOffset(ULONG OutputControl, ULONG Flags, ULONG64 Offset);
HRESULT                 GetFunctionEntryByOffset(ULONG64 Offset, ULONG Flags, PVOID Buffer, ULONG BufferSize, PULONG BufferNeeded);
HRESULT                 GetFieldTypeAndOffset(ULONG64 Module, ULONG ContainerTypeId, PCSTR Field, PULONG FieldTypeId, PULONG Offset);
HRESULT                 GetFieldTypeAndOffsetWide(ULONG64 Module, ULONG ContainerTypeId, PCWSTR Field, PULONG FieldTypeId, PULONG Offset);
HRESULT                 AddSyntheticSymbol(ULONG64 Offset, ULONG Size, PCSTR Name, ULONG Flags, PDEBUG_MODULE_AND_ID Id);
HRESULT                 AddSyntheticSymbolWide(ULONG64 Offset, ULONG Size, PCWSTR Name, ULONG Flags, PDEBUG_MODULE_AND_ID Id);
HRESULT                 RemoveSyntheticSymbol(PDEBUG_MODULE_AND_ID Id);
HRESULT                 GetSymbolEntriesByOffset(ULONG64 Offset, ULONG Flags, PDEBUG_MODULE_AND_ID Ids, PULONG64 Displacements, ULONG IdsCount, PULONG Entries);
HRESULT                 GetSymbolEntriesByName(PCSTR Symbol, ULONG Flags, PDEBUG_MODULE_AND_ID Ids, ULONG IdsCount, PULONG Entries);
HRESULT                 GetSymbolEntriesByNameWide(PCWSTR Symbol, ULONG Flags, PDEBUG_MODULE_AND_ID Ids, ULONG IdsCount, PULONG Entries);
HRESULT                 GetSymbolEntryByToken(ULONG64 ModuleBase, ULONG Token, PDEBUG_MODULE_AND_ID Id);
HRESULT                 GetSymbolEntryInformation(PDEBUG_MODULE_AND_ID Id, PDEBUG_SYMBOL_ENTRY Info);
HRESULT                 GetSymbolEntryString(PDEBUG_MODULE_AND_ID Id, ULONG Which, PSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 GetSymbolEntryStringWide(PDEBUG_MODULE_AND_ID Id, ULONG Which, PWSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 GetSymbolEntryOffsetRegions(PDEBUG_MODULE_AND_ID Id, ULONG Flags, PDEBUG_OFFSET_REGION Regions, ULONG RegionsCount, PULONG RegionsAvail);
HRESULT                 GetSymbolEntryBySymbolEntry(PDEBUG_MODULE_AND_ID FromId, ULONG Flags, PDEBUG_MODULE_AND_ID ToId);
HRESULT                 GetSourceEntriesByOffset(ULONG64 Offset, ULONG Flags, PDEBUG_SYMBOL_SOURCE_ENTRY Entries, ULONG EntriesCount, PULONG EntriesAvail);
HRESULT                 GetSourceEntriesByLine(ULONG Line, PCSTR File, ULONG Flags, PDEBUG_SYMBOL_SOURCE_ENTRY Entries, ULONG EntriesCount, PULONG EntriesAvail);
HRESULT                 GetSourceEntriesByLineWide(ULONG Line, PCWSTR File, ULONG Flags, PDEBUG_SYMBOL_SOURCE_ENTRY Entries, ULONG EntriesCount, PULONG EntriesAvail);
HRESULT                 GetSourceEntryString(PDEBUG_SYMBOL_SOURCE_ENTRY Entry, ULONG Which, PSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 GetSourceEntryStringWide(PDEBUG_SYMBOL_SOURCE_ENTRY Entry, ULONG Which, PWSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 GetSourceEntryOffsetRegions(PDEBUG_SYMBOL_SOURCE_ENTRY Entry, ULONG Flags, PDEBUG_OFFSET_REGION Regions, ULONG RegionsCount, PULONG RegionsAvail);
HRESULT                 GetSourceEntryBySourceEntry(PDEBUG_SYMBOL_SOURCE_ENTRY FromEntry, ULONG Flags, PDEBUG_SYMBOL_SOURCE_ENTRY ToEntry);

// Debugger Symbols Interface Wrappers
static inline HRESULT   GetNameByOffset(POINTER Offset, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement)
                            { return GetNameByOffset(Offset.ptr(), NameBuffer, NameBufferSize, NameSize, Displacement); }
static inline HRESULT   GetNearNameByOffset(POINTER Offset, LONG Delta, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement)
                            { return GetNearNameByOffset(Offset.ptr(), Delta, NameBuffer, NameBufferSize, NameSize, Displacement); }
static inline HRESULT   GetLineByOffset(POINTER Offset, PULONG Line, PSTR FileBuffer, ULONG FileBufferSize, PULONG FileSize, PULONG64 Displacement)
                            { return GetLineByOffset(Offset.ptr(), Line, FileBuffer, FileBufferSize, FileSize, Displacement); }
static inline HRESULT   GetModuleByOffset(POINTER Offset, ULONG StartIndex, PULONG Index, PULONG64 Base)
                            { return GetModuleByOffset(Offset.ptr(), StartIndex, Index, Base); }
static inline HRESULT   GetModuleNames(ULONG Index, POINTER Base, PSTR ImageNameBuffer, ULONG ImageNameBufferSize, PULONG ImageNameSize, PSTR ModuleNameBuffer, ULONG ModuleNameBufferSize, PULONG ModuleNameSize, PSTR LoadedImageNameBuffer, ULONG LoadedImageNameBufferSize, PULONG LoadedImageNameSize)
                            { return GetModuleNames(Index, Base.ptr(), ImageNameBuffer, ImageNameBufferSize, ImageNameSize, ModuleNameBuffer, ModuleNameBufferSize, ModuleNameSize, LoadedImageNameBuffer, LoadedImageNameBufferSize, LoadedImageNameSize); }
static inline HRESULT   GetTypeName(POINTER Module, ULONG TypeId, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize)
                            { return GetTypeName(Module.ptr(), TypeId, NameBuffer, NameBufferSize, NameSize); }
static inline HRESULT   GetTypeId(POINTER Module, PCSTR Name, PULONG TypeId)
                            { return GetTypeId(Module.ptr(), Name, TypeId); }
static inline HRESULT   GetTypeSize(POINTER Module, ULONG TypeId, PULONG Size)
                            { return GetTypeSize(Module.ptr(), TypeId, Size); }
static inline HRESULT   GetFieldOffset(POINTER Module, ULONG TypeId, PCSTR Field, PULONG Offset)
                            { return GetFieldOffset(Module.ptr(), TypeId, Field, Offset); }
static inline HRESULT   GetOffsetTypeId(POINTER Offset, PULONG TypeId, PULONG64 Module)
                            { return GetOffsetTypeId(Offset.ptr(), TypeId, Module); }
static inline HRESULT   ReadTypedDataVirtual(POINTER Offset, ULONG64 Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesRead)
                            { return ReadTypedDataVirtual(Offset.ptr(), Module, TypeId, Buffer, BufferSize, BytesRead); }
static inline HRESULT   WriteTypedDataVirtual(POINTER Offset, ULONG64 Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten)
                            { return WriteTypedDataVirtual(Offset.ptr(), Module, TypeId, Buffer, BufferSize, BytesWritten); }
static inline HRESULT   OutputTypedDataVirtual(ULONG OutputControl, POINTER Offset, ULONG64 Module, ULONG TypeId, ULONG Flags)
                            { return OutputTypedDataVirtual(OutputControl, Offset.ptr(), Module, TypeId, Flags); }
static inline HRESULT   ReadTypedDataPhysical(POINTER Offset, POINTER Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesRead)
                            { return ReadTypedDataPhysical(Offset.ptr(), Module.ptr(), TypeId, Buffer, BufferSize, BytesRead); }
static inline HRESULT   WriteTypedDataPhysical(POINTER Offset, POINTER Module, ULONG TypeId, PVOID Buffer, ULONG BufferSize, PULONG BytesWritten)
                            { return WriteTypedDataPhysical(Offset.ptr(), Module.ptr(), TypeId, Buffer, BufferSize, BytesWritten); }
static inline HRESULT   OutputTypedDataPhysical(ULONG OutputControl, POINTER Offset, POINTER Module, ULONG TypeId, ULONG Flags)
                            { return OutputTypedDataPhysical(OutputControl, Offset.ptr(), Module.ptr(), TypeId, Flags); }
static inline HRESULT   SetScope(POINTER InstructionOffset, PDEBUG_STACK_FRAME ScopeFrame, PVOID ScopeContext, ULONG ScopeContextSize)
                            { return SetScope(InstructionOffset.ptr(), ScopeFrame, ScopeContext, ScopeContextSize); }
static inline HRESULT   GetNextSymbolMatch(POINTER Handle, PSTR Buffer, ULONG BufferSize, PULONG MatchSize, PULONG64 Offset)
                            { return GetNextSymbolMatch(Handle.ptr(), Buffer, BufferSize, MatchSize, Offset); }
static inline HRESULT   EndSymbolMatch(POINTER Handle)
                            { return EndSymbolMatch(Handle.ptr()); }
static inline HRESULT   GetModuleVersionInformation(ULONG Index, POINTER Base, PCSTR Item, PVOID Buffer, ULONG BufferSize, PULONG VerInfoSize)
                            { return GetModuleVersionInformation(Index, Base.ptr(), Item, Buffer, BufferSize, VerInfoSize); }
static inline HRESULT   GetModuleNameString(ULONG Which, ULONG Index, POINTER Base, PSTR Buffer, ULONG BufferSize, PULONG NameSize)
                            { return GetModuleNameString(Which, Index, Base.ptr(), Buffer, BufferSize, NameSize); }
static inline HRESULT   GetConstantName(POINTER Module, ULONG TypeId, ULONG64 Value, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize)
                            { return GetConstantName(Module.ptr(), TypeId, Value, NameBuffer, NameBufferSize, NameSize); }
static inline HRESULT   GetFieldName(POINTER Module, ULONG TypeId, ULONG FieldIndex, PSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize)
                            { return GetFieldName(Module.ptr(), TypeId, FieldIndex, NameBuffer, NameBufferSize, NameSize); }
static inline HRESULT   GetNameByOffsetWide(POINTER Offset, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement)
                            { return GetNameByOffsetWide(Offset.ptr(), NameBuffer, NameBufferSize, NameSize, Displacement); }
static inline HRESULT   GetNearNameByOffsetWide(POINTER Offset, LONG Delta, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize, PULONG64 Displacement)
                            { return GetNearNameByOffsetWide(Offset.ptr(), Delta, NameBuffer, NameBufferSize, NameSize, Displacement); }
static inline HRESULT   GetLineByOffsetWide(POINTER Offset, PULONG Line, PWSTR FileBuffer, ULONG FileBufferSize, PULONG FileSize, PULONG64 Displacement)
                            { return GetLineByOffsetWide(Offset.ptr(), Line, FileBuffer, FileBufferSize, FileSize, Displacement); }
static inline HRESULT   GetTypeNameWide(POINTER Module, ULONG TypeId, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize)
                            { return GetTypeNameWide(Module.ptr(), TypeId, NameBuffer, NameBufferSize, NameSize); }
static inline HRESULT   GetTypeIdWide(POINTER Module, PCWSTR Name, PULONG TypeId)
                            { return GetTypeIdWide(Module.ptr(), Name, TypeId); }
static inline HRESULT   GetFieldOffsetWide(POINTER Module, ULONG TypeId, PCWSTR Field, PULONG Offset)
                            { return GetFieldOffsetWide(Module.ptr(), TypeId, Field, Offset); }
static inline HRESULT   GetNextSymbolMatchWide(POINTER Handle, PWSTR Buffer, ULONG BufferSize, PULONG MatchSize, PULONG64 Offset)
                            { return GetNextSymbolMatchWide(Handle.ptr(), Buffer, BufferSize, MatchSize, Offset); }
static inline HRESULT   GetModuleVersionInformationWide(ULONG Index, POINTER Base, PCWSTR Item, PVOID Buffer, ULONG BufferSize, PULONG VerInfoSize)
                            { return GetModuleVersionInformationWide(Index, Base.ptr(), Item, Buffer, BufferSize, VerInfoSize); }
static inline HRESULT   GetModuleNameStringWide(ULONG Which, ULONG Index, POINTER Base, PWSTR Buffer, ULONG BufferSize, PULONG NameSize)
                            { return GetModuleNameStringWide(Which, Index, Base.ptr(), Buffer, BufferSize, NameSize); }
static inline HRESULT   GetConstantNameWide(POINTER Module, ULONG TypeId, ULONG64 Value, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize)
                            { return GetConstantNameWide(Module.ptr(), TypeId, Value, NameBuffer, NameBufferSize, NameSize); }
static inline HRESULT   GetFieldNameWide(POINTER Module, ULONG TypeId, ULONG FieldIndex, PWSTR NameBuffer, ULONG NameBufferSize, PULONG NameSize)
                            { return GetFieldNameWide(Module.ptr(), TypeId, FieldIndex, NameBuffer, NameBufferSize, NameSize); }
static inline HRESULT   IsManagedModule(ULONG Index, POINTER Base)
                            { return IsManagedModule(Index, Base.ptr()); }
static inline HRESULT   GetModuleByOffset2(POINTER Offset, ULONG StartIndex, ULONG Flags, PULONG Index, PULONG64 Base)
                            { return GetModuleByOffset2(Offset.ptr(), StartIndex, Flags, Index, Base); }
static inline HRESULT   AddSyntheticModule(POINTER Base, ULONG Size, PCSTR ImagePath, PCSTR ModuleName, ULONG Flags)
                            { return AddSyntheticModule(Base.ptr(), Size, ImagePath, ModuleName, Flags); }
static inline HRESULT   AddSyntheticModuleWide(POINTER Base, ULONG Size, PCWSTR ImagePath, PCWSTR ModuleName, ULONG Flags)
                            { return AddSyntheticModuleWide(Base.ptr(), Size, ImagePath, ModuleName, Flags); }
static inline HRESULT   RemoveSyntheticModule(POINTER Base)
                            { return RemoveSyntheticModule(Base.ptr()); }
static inline HRESULT   SetScopeFromJitDebugInfo(ULONG OutputControl, POINTER InfoOffset)
                            { return SetScopeFromJitDebugInfo(OutputControl, InfoOffset.ptr()); }
static inline HRESULT   OutputSymbolByOffset(ULONG OutputControl, ULONG Flags, POINTER Offset)
                            { return OutputSymbolByOffset(OutputControl, Flags, Offset.ptr()); }
static inline HRESULT   GetFunctionEntryByOffset(POINTER Offset, ULONG Flags, PVOID Buffer, ULONG BufferSize, PULONG BufferNeeded)
                            { return GetFunctionEntryByOffset(Offset.ptr(), Flags, Buffer, BufferSize, BufferNeeded); }
static inline HRESULT   GetFieldTypeAndOffset(POINTER Module, ULONG ContainerTypeId, PCSTR Field, PULONG FieldTypeId, PULONG Offset)
                            { return GetFieldTypeAndOffset(Module.ptr(), ContainerTypeId, Field, FieldTypeId, Offset); }
static inline HRESULT   GetFieldTypeAndOffsetWide(POINTER Module, ULONG ContainerTypeId, PCWSTR Field, PULONG FieldTypeId, PULONG Offset)
                            { return GetFieldTypeAndOffsetWide(Module.ptr(), ContainerTypeId, Field, FieldTypeId, Offset); }
static inline HRESULT   AddSyntheticSymbol(POINTER Offset, ULONG Size, PCSTR Name, ULONG Flags, PDEBUG_MODULE_AND_ID Id)
                            { return AddSyntheticSymbol(Offset.ptr(), Size, Name, Flags, Id); }
static inline HRESULT   AddSyntheticSymbolWide(POINTER Offset, ULONG Size, PCWSTR Name, ULONG Flags, PDEBUG_MODULE_AND_ID Id)
                            { return AddSyntheticSymbolWide(Offset.ptr(), Size, Name, Flags, Id); }
static inline HRESULT   GetSymbolEntriesByOffset(POINTER Offset, ULONG Flags, PDEBUG_MODULE_AND_ID Ids, PULONG64 Displacements, ULONG IdsCount, PULONG Entries)
                            { return GetSymbolEntriesByOffset(Offset.ptr(), Flags, Ids, Displacements, IdsCount, Entries); }
static inline HRESULT   GetSourceEntriesByOffset(POINTER Offset, ULONG Flags, PDEBUG_SYMBOL_SOURCE_ENTRY Entries, ULONG EntriesCount, PULONG EntriesAvail)
                            { return GetSourceEntriesByOffset(Offset.ptr(), Flags, Entries, EntriesCount, EntriesAvail); }

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGSYMBOLS_H
