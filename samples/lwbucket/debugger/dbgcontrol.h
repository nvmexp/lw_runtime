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
|*  Module: dbgcontrol.h                                                      *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGCONTROL_H
#define _DBGCONTROL_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
// Debugger Control Interface
HRESULT                 GetInterrupt();
HRESULT                 SetInterrupt(ULONG Flags);
HRESULT                 GetInterruptTimeout(PULONG Seconds);
HRESULT                 SetInterruptTimeout(ULONG Seconds);
HRESULT                 GetLogFile(PSTR Buffer, ULONG BufferSize, PULONG FileSize, PBOOL Append);
HRESULT                 OpenLogFile(PCSTR File, BOOL Append);
HRESULT                 CloseLogFile();
HRESULT                 GetLogMask(PULONG Mask);
HRESULT                 SetLogMask(ULONG Mask);
HRESULT                 Input(PSTR Buffer, ULONG BufferSize, PULONG InputSize);
HRESULT                 ReturnInput(PCSTR Buffer);
HRESULT                 Output(ULONG Mask, PCSTR Format, ...);
HRESULT                 OutputVaList(ULONG Mask, PCSTR Format, va_list Args);
HRESULT                 ControlledOutput(ULONG OutputControl, ULONG Mask, PCSTR Format, ...);
HRESULT                 ControlledOutputVaList(ULONG OutputControl, ULONG Mask, PCSTR Format, va_list Args);
HRESULT                 OutputPrompt(ULONG OutputControl, PCSTR Format, ...);
HRESULT                 OutputPromptVaList(ULONG OutputControl, PCSTR Format, va_list Args);
HRESULT                 GetPromptText(PSTR Buffer, ULONG BufferSize, PULONG TextSize);
HRESULT                 OutputLwrrentState(ULONG OutputControl, ULONG Flags);
HRESULT                 OutputVersionInformation(ULONG OutputControl);
HRESULT                 GetNotifyEventHandle(PULONG64 Handle);
HRESULT                 SetNotifyEventHandle(ULONG64 Handle);
HRESULT                 Assemble(ULONG64 Offset, PCSTR Instr, PULONG64 EndOffset);
HRESULT                 Disassemble(ULONG64 Offset, ULONG Flags, PSTR Buffer, ULONG BufferSize, PULONG DisassemblySize, PULONG64 EndOffset);
HRESULT                 GetDisassembleEffectiveOffset(PULONG64 Offset);
HRESULT                 OutputDisassembly(ULONG OutputControl, ULONG64 Offset, ULONG Flags, PULONG64 EndOffset);
HRESULT                 OutputDisassemblyLines(ULONG OutputControl, ULONG PreviousLines, ULONG TotalLines, ULONG64 Offset, ULONG Flags, PULONG OffsetLine, PULONG64 StartOffset, PULONG64 EndOffset, PULONG64 LineOffsets);
HRESULT                 GetNearInstruction(ULONG64 Offset, LONG Delta, PULONG64 NearOffset);
HRESULT                 GetStackTrace(ULONG64 FrameOffset, ULONG64 StackOffset, ULONG64 InstructionOffset, PDEBUG_STACK_FRAME Frames, ULONG FramesSize, PULONG FramesFilled);
HRESULT                 GetReturnOffset(PULONG64 Offset);
HRESULT                 OutputStackTrace(ULONG OutputControl, PDEBUG_STACK_FRAME Frames, ULONG FramesSize, ULONG Flags);
HRESULT                 GetDebuggeeType(PULONG Class, PULONG Qualifier);
HRESULT                 GetActualProcessorType(PULONG Type);
HRESULT                 GetExelwtingProcessorType(PULONG Type);
HRESULT                 GetNumberPossibleExelwtingProcessorTypes(PULONG Number);
HRESULT                 GetPossibleExelwtingProcessorTypes(ULONG Start, ULONG Count, PULONG Types);
HRESULT                 GetNumberProcessors(PULONG Number);
HRESULT                 GetSystemVersion(PULONG PlatformId, PULONG Major, PULONG Minor, PSTR ServicePackString, ULONG ServicePackStringSize, PULONG ServicePackStringUsed, PULONG ServicePackNumber, PSTR BuildString, ULONG BuildStringSize, PULONG BuildStringUsed);
HRESULT                 GetPageSize(PULONG Size);
HRESULT                 IsPointer64Bit();
HRESULT                 ReadBugCheckData(PULONG Code, PULONG64 Arg1, PULONG64 Arg2, PULONG64 Arg3, PULONG64 Arg4);
HRESULT                 GetNumberSupportedProcessorTypes(PULONG Number);
HRESULT                 GetSupportedProcessorTypes(ULONG Start, ULONG Count, PULONG Types);
HRESULT                 GetProcessorTypeNames(ULONG Type, PSTR FullNameBuffer, ULONG FullNameBufferSize, PULONG FullNameSize, PSTR AbbrevNameBuffer, ULONG AbbrevNameBufferSize, PULONG AbbrevNameSize);
HRESULT                 GetEffectiveProcessorType(PULONG Type);
HRESULT                 SetEffectiveProcessorType(ULONG Type);
HRESULT                 GetExelwtionStatus(PULONG Status);
HRESULT                 SetExelwtionStatus(ULONG Status);
HRESULT                 GetCodeLevel(PULONG Level);
HRESULT                 SetCodeLevel(ULONG Level);
HRESULT                 GetEngineOptions(PULONG Options);
HRESULT                 AddEngineOptions(ULONG Options);
HRESULT                 RemoveEngineOptions(ULONG Options);
HRESULT                 SetEngineOptions(ULONG Options);
HRESULT                 GetSystemErrorControl(PULONG OutputLevel, PULONG BreakLevel);
HRESULT                 SetSystemErrorControl(ULONG OutputLevel, ULONG BreakLevel);
HRESULT                 GetTextMacro(ULONG Slot, PSTR Buffer, ULONG BufferSize, PULONG MacroSize);
HRESULT                 SetTextMacro(ULONG Slot, PCSTR Macro);
HRESULT                 GetRadix(PULONG Radix);
HRESULT                 SetRadix(ULONG Radix);
HRESULT                 Evaluate(PCSTR Expression, ULONG DesiredType, PDEBUG_VALUE Value, PULONG RemainderIndex);
HRESULT                 CoerceValue(PDEBUG_VALUE In, ULONG OutType, PDEBUG_VALUE Out);
HRESULT                 CoerceValues(ULONG Count, PDEBUG_VALUE In, PULONG OutTypes, PDEBUG_VALUE Out);
HRESULT                 Execute(ULONG OutputControl, PCSTR Command, ULONG Flags);
HRESULT                 ExelwteCommandFile(ULONG OutputControl, PCSTR CommandFile, ULONG Flags);
HRESULT                 GetNumberBreakpoints(PULONG Number);
HRESULT                 GetBreakpointByIndex(ULONG Index, PDEBUG_BREAKPOINT* Bp);
HRESULT                 GetBreakpointById(ULONG Id, PDEBUG_BREAKPOINT* Bp);
HRESULT                 GetBreakpointParameters(ULONG Count, PULONG Ids, ULONG Start, PDEBUG_BREAKPOINT_PARAMETERS Params);
HRESULT                 AddBreakpoint(ULONG Type, ULONG DesiredId, PDEBUG_BREAKPOINT* Bp);
HRESULT                 RemoveBreakpoint(PDEBUG_BREAKPOINT Bp);
HRESULT                 AddExtension(PCSTR Path, ULONG Flags, PULONG64 Handle);
HRESULT                 RemoveExtension(ULONG64 Handle);
HRESULT                 GetExtensionByPath(PCSTR Path, PULONG64 Handle);
HRESULT                 CallExtension(ULONG64 Handle, PCSTR Function, PCSTR Arguments);
HRESULT                 GetExtensionFunction(ULONG64 Handle, PCSTR FuncName, FARPROC* Function);
HRESULT                 GetWindbgExtensionApis32(PWINDBG_EXTENSION_APIS32 Api);
HRESULT                 GetWindbgExtensionApis64(PWINDBG_EXTENSION_APIS64 Api);
HRESULT                 GetNumberEventFilters(PULONG SpecificEvents, PULONG SpecificExceptions, PULONG ArbitraryExceptions);
HRESULT                 GetEventFilterText(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG TextSize);
HRESULT                 GetEventFilterCommand(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetEventFilterCommand(ULONG Index, PCSTR Command);
HRESULT                 GetSpecificFilterParameters(ULONG Start, ULONG Count, PDEBUG_SPECIFIC_FILTER_PARAMETERS Params);
HRESULT                 SetSpecificFilterParameters(ULONG Start, ULONG Count, PDEBUG_SPECIFIC_FILTER_PARAMETERS Params);
HRESULT                 GetSpecificFilterArgument(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG ArgumentSize);
HRESULT                 SetSpecificFilterArgument(ULONG Index, PCSTR Argument);
HRESULT                 GetExceptionFilterParameters(ULONG Count, PULONG Codes, ULONG Start, PDEBUG_EXCEPTION_FILTER_PARAMETERS Params);
HRESULT                 SetExceptionFilterParameters(ULONG Count, PDEBUG_EXCEPTION_FILTER_PARAMETERS Params);
HRESULT                 GetExceptionFilterSecondCommand(ULONG Index, PSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetExceptionFilterSecondCommand(ULONG Index, PCSTR Command);
HRESULT                 WaitForEvent(ULONG Flags, ULONG Timeout);
HRESULT                 GetLastEventInformation(PULONG Type, PULONG ProcessId, PULONG ThreadId, PVOID ExtraInformation, ULONG ExtraInformationSize, PULONG ExtraInformationUsed, PSTR Description, ULONG DescriptionSize, PULONG DescriptionUsed);
// Debugger Control 2 Interface
HRESULT                 GetLwrrentTimeDate(PULONG TimeDate);
HRESULT                 GetLwrrentSystemUpTime(PULONG UpTime);
HRESULT                 GetDumpFormatFlags(PULONG FormatFlags);
HRESULT                 GetNumberTextReplacements(PULONG NumRepl);
HRESULT                 GetTextReplacement(PCSTR SrcText, ULONG Index, PSTR SrcBuffer, ULONG SrcBufferSize, PULONG SrcSize, PSTR DstBuffer, ULONG DstBufferSize, PULONG DstSize);
HRESULT                 SetTextReplacement(PCSTR SrcText, PCSTR DstText);
HRESULT                 RemoveTextReplacements();
HRESULT                 OutputTextReplacements(ULONG OutputControl, ULONG Flags);
// Debugger Control 3 Interface
HRESULT                 GetAssemblyOptions(PULONG Options);
HRESULT                 AddAssemblyOptions(ULONG Options);
HRESULT                 RemoveAssemblyOptions(ULONG Options);
HRESULT                 SetAssemblyOptions(ULONG Options);
HRESULT                 GetExpressionSyntax(PULONG Flags);
HRESULT                 SetExpressionSyntax(ULONG Flags);
HRESULT                 SetExpressionSyntaxByName(PCSTR AbbrevName);
HRESULT                 GetNumberExpressionSyntaxes(PULONG Number);
HRESULT                 GetExpressionSyntaxNames(ULONG Index, PSTR FullNameBuffer, ULONG FullNameBufferSize, PULONG FullNameSize, PSTR AbbrevNameBuffer, ULONG AbbrevNameBufferSize, PULONG AbbrevNameSize);
HRESULT                 GetNumberEvents(PULONG Events);
HRESULT                 GetEventIndexDescription(ULONG Index, ULONG Which, PSTR Buffer, ULONG BufferSize, PULONG DescSize);
HRESULT                 GetLwrrentEventIndex(PULONG Index);
HRESULT                 SetNextEventIndex(ULONG Relation, ULONG Value, PULONG NextIndex);
// Debugger Control 4 Interface
HRESULT                 GetLogFileWide(PWSTR Buffer, ULONG BufferSize, PULONG FileSize, PBOOL Append);
HRESULT                 OpenLogFileWide(PCWSTR File, BOOL Append);
HRESULT                 InputWide(PWSTR Buffer, ULONG BufferSize, PULONG InputSize);
HRESULT                 ReturnInputWide(PCWSTR Buffer);
HRESULT                 OutputWide(ULONG Mask, PCWSTR Format, ...);
HRESULT                 OutputVaListWide(ULONG Mask, PCWSTR Format, va_list Args);
HRESULT                 ControlledOutputWide(ULONG OutputControl, ULONG Mask, PCWSTR Format, ...);
HRESULT                 ControlledOutputVaListWide(ULONG OutputControl, ULONG Mask, PCWSTR Format, va_list Args);
HRESULT                 OutputPromptWide(ULONG OutputControl, PCWSTR Format, ...);
HRESULT                 OutputPromptVaListWide(ULONG OutputControl, PCWSTR Format, va_list Args);
HRESULT                 GetPromptTextWide(PWSTR Buffer, ULONG BufferSize, PULONG TextSize);
HRESULT                 AssembleWide(ULONG64 Offset, PCWSTR Instr, PULONG64 EndOffset);
HRESULT                 DisassembleWide(ULONG64 Offset, ULONG Flags, PWSTR Buffer, ULONG BufferSize, PULONG DisassemblySize, PULONG64 EndOffset);
HRESULT                 GetProcessorTypeNamesWide(ULONG Type, PWSTR FullNameBuffer, ULONG FullNameBufferSize, PULONG FullNameSize, PWSTR AbbrevNameBuffer, ULONG AbbrevNameBufferSize, PULONG AbbrevNameSize);
HRESULT                 GetTextMacroWide(ULONG Slot, PWSTR Buffer, ULONG BufferSize, PULONG MacroSize);
HRESULT                 SetTextMacroWide(ULONG Slot, PCWSTR Macro);
HRESULT                 EvaluateWide(PCWSTR Expression, ULONG DesiredType, PDEBUG_VALUE Value, PULONG RemainderIndex);
HRESULT                 ExelwteWide(ULONG OutputControl, PCWSTR Command, ULONG Flags);
HRESULT                 ExelwteCommandFileWide(ULONG OutputControl, PCWSTR CommandFile, ULONG Flags);
HRESULT                 GetBreakpointByIndex2(ULONG Index, PDEBUG_BREAKPOINT2* Bp);
HRESULT                 GetBreakpointById2(ULONG Id, PDEBUG_BREAKPOINT2* Bp);
HRESULT                 AddBreakpoint2(ULONG Type, ULONG DesiredId, PDEBUG_BREAKPOINT2* Bp);
HRESULT                 RemoveBreakpoint2(PDEBUG_BREAKPOINT2 Bp);
HRESULT                 AddExtensionWide(PCWSTR Path, ULONG Flags, PULONG64 Handle);
HRESULT                 GetExtensionByPathWide(PCWSTR Path, PULONG64 Handle);
HRESULT                 CallExtensionWide(ULONG64 Handle, PCWSTR Function, PCWSTR Arguments);
HRESULT                 GetExtensionFunctionWide(ULONG64 Handle, PCWSTR FuncName, FARPROC* Function);
HRESULT                 GetEventFilterTextWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG TextSize);
HRESULT                 GetEventFilterCommandWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetEventFilterCommandWide(ULONG Index, PCWSTR Command);
HRESULT                 GetSpecificFilterArgumentWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG ArgumentSize);
HRESULT                 SetSpecificFilterArgumentWide(ULONG Index, PCWSTR Argument);
HRESULT                 GetExceptionFilterSecondCommandWide(ULONG Index, PWSTR Buffer, ULONG BufferSize, PULONG CommandSize);
HRESULT                 SetExceptionFilterSecondCommandWide(ULONG Index, PCWSTR Command);
HRESULT                 GetLastEventInformationWide(PULONG Type, PULONG ProcessId, PULONG ThreadId, PVOID ExtraInformation, ULONG ExtraInformationSize, PULONG ExtraInformationUsed, PWSTR Description, ULONG DescriptionSize, PULONG DescriptionUsed);
HRESULT                 GetTextReplacementWide(PCWSTR SrcText, ULONG Index, PWSTR SrcBuffer, ULONG SrcBufferSize, PULONG SrcSize, PWSTR DstBuffer, ULONG DstBufferSize, PULONG DstSize);
HRESULT                 SetTextReplacementWide(PCWSTR SrcText, PCWSTR DstText);
HRESULT                 SetExpressionSyntaxByNameWide(PCWSTR AbbrevName);
HRESULT                 GetExpressionSyntaxNamesWide(ULONG Index, PWSTR FullNameBuffer, ULONG FullNameBufferSize, PULONG FullNameSize, PWSTR AbbrevNameBuffer, ULONG AbbrevNameBufferSize, PULONG AbbrevNameSize);
HRESULT                 GetEventIndexDescriptionWide(ULONG Index, ULONG Which, PWSTR Buffer, ULONG BufferSize, PULONG DescSize);
HRESULT                 GetLogFile2(PSTR Buffer, ULONG BufferSize, PULONG FileSize, PULONG Flags);
HRESULT                 OpenLogFile2(PCSTR File, ULONG Flags);
HRESULT                 GetLogFile2Wide(PWSTR Buffer, ULONG BufferSize, PULONG FileSize, PULONG Flags);
HRESULT                 OpenLogFile2Wide(PCWSTR File, ULONG Flags);
HRESULT                 GetSystemVersiolwalues(PULONG PlatformId, PULONG Win32Major, PULONG Win32Minor, PULONG KdMajor, PULONG KdMinor);
HRESULT                 GetSystemVersionString(ULONG Which, PSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 GetSystemVersionStringWide(ULONG Which, PWSTR Buffer, ULONG BufferSize, PULONG StringSize);
HRESULT                 GetContextStackTrace(PVOID StartContext, ULONG StartContextSize, PDEBUG_STACK_FRAME Frames, ULONG FramesSize, PVOID FrameContexts, ULONG FrameContextsSize, ULONG FrameContextsEntrySize, PULONG FramesFilled);
HRESULT                 OutputContextStackTrace(ULONG OutputControl, PDEBUG_STACK_FRAME Frames, ULONG FramesSize, PVOID FrameContexts, ULONG FrameContextsSize, ULONG FrameContextsEntrySize, ULONG Flags);
HRESULT                 GetStoredEventInformation(PULONG Type, PULONG ProcessId, PULONG ThreadId, PVOID Context, ULONG ContextSize, PULONG ContextUsed, PVOID ExtraInformation, ULONG ExtraInformationSize, PULONG ExtraInformationUsed);
HRESULT                 GetManagedStatus(PULONG Flags, ULONG WhichString, PSTR String, ULONG StringSize, PULONG StringNeeded);
HRESULT                 GetManagedStatusWide(PULONG Flags, ULONG WhichString, PWSTR String, ULONG StringSize, PULONG StringNeeded);
HRESULT                 ResetManagedStatus(ULONG Flags);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGCONTROL_H
