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
|*  Module: dbgname.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGNAME_H
#define _DBGNAME_H

//******************************************************************************
//
// dbg namespace entries
//
//******************************************************************************
// In dbgadvanced
using dbg::GetThreadContext;
using dbg::SetThreadContext;

using dbg::Request;
using dbg::GetSourceFileInformation;
using dbg::FindSourceFileAndToken;
using dbg::GetSymbolInformation;
using dbg::GetSystemObjectInformation;

using dbg::GetSourceFileInformationWide;
using dbg::FindSourceFileAndTokenWide;
using dbg::GetSymbolInformationWide;

using dbg::GetSourceFileInformation;
using dbg::FindSourceFileAndToken;
using dbg::GetSymbolInformation;
using dbg::GetSystemObjectInformation;

using dbg::GetSourceFileInformationWide;
using dbg::FindSourceFileAndTokenWide;
using dbg::GetSymbolInformationWide;

// In dbgbreakpoint.h
using dbg::GetId;
using dbg::GetType;
using dbg::GetAdder;
using dbg::GetFlags;
using dbg::AddFlags;
using dbg::RemoveFlags;
using dbg::SetFlags;
using dbg::GetOffset;
using dbg::SetOffset;
using dbg::GetDataParameters;
using dbg::SetDataParameters;
using dbg::GetPassCount;
using dbg::SetPassCount;
using dbg::GetLwrrentPassCount;
using dbg::GetMatchThreadId;
using dbg::SetMatchThreadId;
using dbg::GetCommand;
using dbg::SetCommand;
using dbg::GetOffsetExpression;
using dbg::SetOffsetExpression;
using dbg::GetParameters;

using dbg::GetCommandWide;
using dbg::SetCommandWide;
using dbg::GetOffsetExpressionWide;
using dbg::SetOffsetExpressionWide;

using dbg::GetGuid;

using dbg::GetType;
using dbg::GetAdder;
using dbg::GetFlags;
using dbg::AddFlags;
using dbg::RemoveFlags;
using dbg::SetFlags;
using dbg::GetOffset;
using dbg::SetOffset;
using dbg::GetDataParameters;
using dbg::SetDataParameters;
using dbg::GetPassCount;
using dbg::SetPassCount;
using dbg::GetLwrrentPassCount;
using dbg::GetMatchThreadId;
using dbg::SetMatchThreadId;
using dbg::GetCommand;
using dbg::SetCommand;
using dbg::GetOffsetExpression;
using dbg::SetOffsetExpression;
using dbg::GetParameters;

using dbg::GetCommandWide;
using dbg::SetCommandWide;
using dbg::GetOffsetExpressionWide;
using dbg::SetOffsetExpressionWide;

using dbg::GetGuid;

// In dbgclient.h
using dbg::AttachKernel;
using dbg::GetKernelConnectionOptions;
using dbg::SetKernelConnectionOptions;
using dbg::StartProcessServer;
using dbg::ConnectProcessServer;
using dbg::DisconnectProcessServer;
using dbg::GetRunningProcessSystemIds;
using dbg::GetRunningProcessSystemIdByExelwtableName;
using dbg::GetRunningProcessDescription;
using dbg::AttachProcess;
using dbg::CreateProcessClient;
using dbg::CreateProcessAndAttach;
using dbg::GetProcessOptions;
using dbg::AddProcessOptions;
using dbg::RemoveProcessOptions;
using dbg::SetProcessOptions;
using dbg::OpenDumpFile;
using dbg::WriteDumpFile;
using dbg::ConnectSession;
using dbg::StartServer;
using dbg::OutputServers;
using dbg::TerminateProcesses;
using dbg::DetachProcesses;
using dbg::EndSession;
using dbg::GetExitCode;
using dbg::DispatchCallbacks;
using dbg::ExitDispatch;
using dbg::CreateClient;
using dbg::GetInputCallbacks;
using dbg::SetInputCallbacks;
using dbg::GetOutputCallbacks;
using dbg::SetOutputCallbacks;
using dbg::GetOutputMask;
using dbg::SetOutputMask;
using dbg::GetOtherOutputMask;
using dbg::SetOtherOutputMask;
using dbg::GetOutputWidth;
using dbg::SetOutputWidth;
using dbg::GetOutputLinePrefix;
using dbg::SetOutputLinePrefix;
using dbg::GetIdentity;
using dbg::OutputIdentity;
using dbg::GetEventCallbacks;
using dbg::SetEventCallbacks;
using dbg::FlushCallbacks;

using dbg::WriteDumpFile2;
using dbg::AddDumpInformationFile;
using dbg::EndProcessServer;
using dbg::WaitForProcessServerEnd;
using dbg::IsKernelDebuggerEnabled;
using dbg::TerminateLwrrentProcess;
using dbg::DetachLwrrentProcess;
using dbg::AbandonLwrrentProcess;

using dbg::GetRunningProcessSystemIdByExelwtableNameWide;
using dbg::GetRunningProcessDescriptionWide;
using dbg::CreateProcessWide;
using dbg::CreateProcessAndAttachWide;

using dbg::OpenDumpFileWide;
using dbg::WriteDumpFileWide;
using dbg::AddDumpInformationFileWide;
using dbg::GetNumberDumpFiles;
using dbg::GetDumpFile;
using dbg::GetDumpFileWide;

using dbg::AttachKernelWide;
using dbg::GetKernelConnectionOptionsWide;
using dbg::SetKernelConnectionOptionsWide;
using dbg::StartProcessServerWide;
using dbg::ConnectProcessServerWide;
using dbg::StartServerWide;
using dbg::OutputServersWide;
using dbg::GetOutputCallbacksWide;
using dbg::SetOutputCallbacksWide;
using dbg::GetOutputLinePrefixWide;
using dbg::SetOutputLinePrefixWide;
using dbg::GetIdentityWide;
using dbg::OutputIdentityWide;
using dbg::GetEventCallbacksWide;
using dbg::SetEventCallbacksWide;
using dbg::CreateProcess2;
using dbg::CreateProcess2Wide;
using dbg::CreateProcessAndAttach2;
using dbg::CreateProcessAndAttach2Wide;
using dbg::PushOutputLinePrefix;
using dbg::PushOutputLinePrefixWide;
using dbg::PopOutputLinePrefix;
using dbg::GetNumberInputCallbacks;
using dbg::GetNumberOutputCallbacks;
using dbg::GetNumberEventCallbacks;
using dbg::GetQuitLockString;
using dbg::SetQuitLockString;
using dbg::GetQuitLockStringWide;
using dbg::SetQuitLockStringWide;

// In dbgcontrol.h
using dbg::GetInterrupt;
using dbg::SetInterrupt;
using dbg::GetInterruptTimeout;
using dbg::SetInterruptTimeout;
using dbg::GetLogFile;
using dbg::OpenLogFile;
using dbg::CloseLogFile;
using dbg::GetLogMask;
using dbg::SetLogMask;
using dbg::Input;
using dbg::ReturnInput;
using dbg::Output;
using dbg::OutputVaList;
using dbg::ControlledOutput;
using dbg::ControlledOutputVaList;
using dbg::OutputPrompt;
using dbg::OutputPromptVaList;
using dbg::GetPromptText;
using dbg::OutputLwrrentState;
using dbg::OutputVersionInformation;
using dbg::GetNotifyEventHandle;
using dbg::SetNotifyEventHandle;
using dbg::Assemble;
using dbg::Disassemble;
using dbg::GetDisassembleEffectiveOffset;
using dbg::OutputDisassembly;
using dbg::OutputDisassemblyLines;
using dbg::GetNearInstruction;
using dbg::GetStackTrace;
using dbg::GetReturnOffset;
using dbg::OutputStackTrace;
using dbg::GetDebuggeeType;
using dbg::GetActualProcessorType;
using dbg::GetExelwtingProcessorType;
using dbg::GetNumberPossibleExelwtingProcessorTypes;
using dbg::GetPossibleExelwtingProcessorTypes;
using dbg::GetNumberProcessors;
using dbg::GetSystemVersion;
using dbg::GetPageSize;
using dbg::IsPointer64Bit;
using dbg::ReadBugCheckData;
using dbg::GetNumberSupportedProcessorTypes;
using dbg::GetSupportedProcessorTypes;
using dbg::GetProcessorTypeNames;
using dbg::GetEffectiveProcessorType;
using dbg::SetEffectiveProcessorType;
using dbg::GetExelwtionStatus;
using dbg::SetExelwtionStatus;
using dbg::GetCodeLevel;
using dbg::SetCodeLevel;
using dbg::GetEngineOptions;
using dbg::AddEngineOptions;
using dbg::RemoveEngineOptions;
using dbg::SetEngineOptions;
using dbg::GetSystemErrorControl;
using dbg::SetSystemErrorControl;
using dbg::GetTextMacro;
using dbg::SetTextMacro;
using dbg::GetRadix;
using dbg::SetRadix;
using dbg::Evaluate;
using dbg::CoerceValue;
using dbg::CoerceValues;
using dbg::Execute;
using dbg::ExelwteCommandFile;
using dbg::GetNumberBreakpoints;
using dbg::GetBreakpointByIndex;
using dbg::GetBreakpointById;
using dbg::GetBreakpointParameters;
using dbg::AddBreakpoint;
using dbg::RemoveBreakpoint;
using dbg::AddExtension;
using dbg::RemoveExtension;
using dbg::GetExtensionByPath;
using dbg::CallExtension;
using dbg::GetExtensionFunction;
using dbg::GetWindbgExtensionApis32;
using dbg::GetWindbgExtensionApis64;
using dbg::GetNumberEventFilters;
using dbg::GetEventFilterText;
using dbg::GetEventFilterCommand;
using dbg::SetEventFilterCommand;
using dbg::GetSpecificFilterParameters;
using dbg::SetSpecificFilterParameters;
using dbg::GetSpecificFilterArgument;
using dbg::SetSpecificFilterArgument;
using dbg::GetExceptionFilterParameters;
using dbg::SetExceptionFilterParameters;
using dbg::GetExceptionFilterSecondCommand;
using dbg::SetExceptionFilterSecondCommand;
using dbg::WaitForEvent;
using dbg::GetLastEventInformation;

using dbg::GetLwrrentTimeDate;
using dbg::GetLwrrentSystemUpTime;
using dbg::GetDumpFormatFlags;
using dbg::GetNumberTextReplacements;
using dbg::GetTextReplacement;
using dbg::SetTextReplacement;
using dbg::RemoveTextReplacements;
using dbg::OutputTextReplacements;

using dbg::GetAssemblyOptions;
using dbg::AddAssemblyOptions;
using dbg::RemoveAssemblyOptions;
using dbg::SetAssemblyOptions;
using dbg::GetExpressionSyntax;
using dbg::SetExpressionSyntax;
using dbg::SetExpressionSyntaxByName;
using dbg::GetNumberExpressionSyntaxes;
using dbg::GetExpressionSyntaxNames;
using dbg::GetNumberEvents;
using dbg::GetEventIndexDescription;
using dbg::GetLwrrentEventIndex;
using dbg::SetNextEventIndex;

using dbg::GetLogFileWide;
using dbg::OpenLogFileWide;
using dbg::InputWide;
using dbg::ReturnInputWide;
using dbg::OutputWide;
using dbg::OutputVaListWide;
using dbg::ControlledOutputWide;
using dbg::ControlledOutputVaListWide;
using dbg::OutputPromptWide;
using dbg::OutputPromptVaListWide;
using dbg::GetPromptTextWide;
using dbg::AssembleWide;
using dbg::DisassembleWide;
using dbg::GetProcessorTypeNamesWide;
using dbg::GetTextMacroWide;
using dbg::SetTextMacroWide;
using dbg::EvaluateWide;
using dbg::ExelwteWide;
using dbg::ExelwteCommandFileWide;
using dbg::GetBreakpointByIndex2;
using dbg::GetBreakpointById2;
using dbg::AddBreakpoint2;
using dbg::RemoveBreakpoint2;
using dbg::AddExtensionWide;
using dbg::GetExtensionByPathWide;
using dbg::CallExtensionWide;
using dbg::GetExtensionFunctionWide;
using dbg::GetEventFilterTextWide;
using dbg::GetEventFilterCommandWide;
using dbg::SetEventFilterCommandWide;
using dbg::GetSpecificFilterArgumentWide;
using dbg::SetSpecificFilterArgumentWide;
using dbg::GetExceptionFilterSecondCommandWide;
using dbg::SetExceptionFilterSecondCommandWide;
using dbg::GetLastEventInformationWide;
using dbg::GetTextReplacementWide;
using dbg::SetTextReplacementWide;
using dbg::SetExpressionSyntaxByNameWide;
using dbg::GetExpressionSyntaxNamesWide;
using dbg::GetEventIndexDescriptionWide;
using dbg::GetLogFile2;
using dbg::OpenLogFile2;
using dbg::GetLogFile2Wide;
using dbg::OpenLogFile2Wide;
using dbg::GetSystemVersiolwalues;
using dbg::GetSystemVersionString;
using dbg::GetSystemVersionStringWide;
using dbg::GetContextStackTrace;
using dbg::OutputContextStackTrace;
using dbg::GetStoredEventInformation;
using dbg::GetManagedStatus;
using dbg::GetManagedStatusWide;
using dbg::ResetManagedStatus;

// In dbgdataspaces.h
using dbg::ReadVirtual;
using dbg::WriteVirtual;
using dbg::SearchVirtual;
using dbg::ReadVirtualUncached;
using dbg::WriteVirtualUncached;
using dbg::ReadPointersVirtual;
using dbg::WritePointersVirtual;
using dbg::ReadPhysical;
using dbg::WritePhysical;
using dbg::ReadControl;
using dbg::WriteControl;
using dbg::ReadIo;
using dbg::WriteIo;
using dbg::ReadMsr;
using dbg::WriteMsr;
using dbg::ReadBusData;
using dbg::WriteBusData;
using dbg::CheckLowMemory;
using dbg::ReadDebuggerData;
using dbg::ReadProcessorSystemData;

using dbg::VirtualToPhysical;
using dbg::GetVirtualTranslationPhysicalOffsets;
using dbg::ReadHandleData;
using dbg::FillVirtual;
using dbg::FillPhysical;
using dbg::QueryVirtual;

using dbg::ReadImageNtHeaders;
using dbg::ReadTagged;
using dbg::StartEnumTagged;
using dbg::GetNextTagged;
using dbg::EndEnumTagged;

using dbg::GetOffsetInformation;
using dbg::GetNextDifferentlyValidOffsetVirtual;
using dbg::GetValidRegiolwirtual;
using dbg::SearchVirtual2;
using dbg::ReadMultiByteStringVirtual;
using dbg::ReadMultiByteStringVirtualWide;
using dbg::ReadUnicodeStringVirtual;
using dbg::ReadUnicodeStringVirtualWide;
using dbg::ReadPhysical2;
using dbg::WritePhysical2;

// In dbgevent.h
using dbg::CEvent;

using dbg::SetInterestMask;

// In dbghook.h
using dbg::CHook;

using dbg::callInitializeHooks;
using dbg::callNotifyHooks;
using dbg::callUninitializeHooks;

// In dbginput.h
using dbg::StartInput;
using dbg::EndInput;

// In dbgInterface.h
using dbg::createInterface;
using dbg::releaseInterface;

using dbg::getBreakpointInterface;

using dbg::initializeDebugInterface;
using dbg::uninitializeDebugInterface;

using dbg::acquireDebugInterface;
using dbg::releaseDebugInterface;

using dbg::initializeInterfaces;
using dbg::releaseInterfaces;

using dbg::isDebugClientInterface;
using dbg::isDebugClient2Interface;
using dbg::isDebugClient3Interface;
using dbg::isDebugClient4Interface;
using dbg::isDebugClient5Interface;

using dbg::debugClientInterface;
using dbg::debugClient2Interface;
using dbg::debugClient3Interface;
using dbg::debugClient4Interface;
using dbg::debugClient5Interface;

using dbg::isDebugControlInterface;
using dbg::isDebugControl2Interface;
using dbg::isDebugControl3Interface;
using dbg::isDebugControl4Interface;

using dbg::debugControlInterface;
using dbg::debugControl2Interface;
using dbg::debugControl3Interface;
using dbg::debugControl4Interface;

using dbg::isDebugDataSpacesInterface;
using dbg::isDebugDataSpaces2Interface;
using dbg::isDebugDataSpaces3Interface;
using dbg::isDebugDataSpaces4Interface;

using dbg::debugDataSpacesInterface;
using dbg::debugDataSpaces2Interface;
using dbg::debugDataSpaces3Interface;
using dbg::debugDataSpaces4Interface;

using dbg::isDebugRegistersInterface;
using dbg::isDebugRegisters2Interface;

using dbg::debugRegistersInterface;
using dbg::debugRegisters2Interface;

using dbg::isDebugSymbolsInterface;
using dbg::isDebugSymbols2Interface;
using dbg::isDebugSymbols3Interface;

using dbg::debugSymbolsInterface;
using dbg::debugSymbols2Interface;
using dbg::debugSymbols3Interface;

using dbg::isDebugSystemObjectsInterface;
using dbg::isDebugSystemObjects2Interface;
using dbg::isDebugSystemObjects3Interface;
using dbg::isDebugSystemObjects4Interface;

using dbg::debugSystemObjectsInterface;
using dbg::debugSystemObjects2Interface;
using dbg::debugSystemObjects3Interface;
using dbg::debugSystemObjects4Interface;

using dbg::isDebugAdvancedInterface;
using dbg::isDebugAdvanced2Interface;
using dbg::isDebugAdvanced3Interface;

using dbg::debugAdvancedInterface;
using dbg::debugAdvanced2Interface;
using dbg::debugAdvanced3Interface;

using dbg::debugSymbolGroupInterface;

using dbg::debugBreakpointInterface;
using dbg::setBreakpointInterface;

using dbg::dbgInput;
using dbg::dbgOutput;
using dbg::dbgEvent;

// In dbgoutput.h
using dbg::GetDisplayState;
using dbg::GetCaptureState;
using dbg::GetDmlState;
using dbg::SetDisplayState;
using dbg::SetCaptureState;
using dbg::SetDmlState;
using dbg::GetCaptureSize;
using dbg::GetCaptureOutput;
using dbg::ClearCapture;

// In dbgregisters.h
using dbg::GetNumberRegisters;
using dbg::GetDescription;
using dbg::GetIndexByName;
using dbg::GetValue;
using dbg::SetValue;
using dbg::GetValues;
using dbg::SetValues;
using dbg::OutputRegisters;
using dbg::GetInstructionOffset;
using dbg::GetStackOffset;
using dbg::GetFrameOffset;

using dbg::GetDescriptionWide;
using dbg::GetIndexByNameWide;
using dbg::GetNumberPseudoRegisters;
using dbg::GetPseudoDescription;
using dbg::GetPseudoDescriptionWide;
using dbg::GetPseudoIndexByName;
using dbg::GetPseudoIndexByNameWide;
using dbg::GetPseudoValues;
using dbg::SetPseudoValues;
using dbg::GetValues2;
using dbg::SetValues2;
using dbg::OutputRegisters2;
using dbg::GetInstructionOffset2;
using dbg::GetStackOffset2;
using dbg::GetFrameOffset2;

// In dbgsymbolgroup
using dbg::GetNumberSymbols;
using dbg::AddSymbol;
using dbg::RemoveSymbolByName;
using dbg::RemoveSymbolByIndex;
using dbg::GetSymbolName;
using dbg::GetSymbolParameters;
using dbg::ExpandSymbol;
using dbg::OutputSymbols;
using dbg::WriteSymbol;
using dbg::OutputAsType;

using dbg::AddSymbolWide;
using dbg::RemoveSymbolByNameWide;
using dbg::GetSymbolNameWide;
using dbg::WriteSymbolWide;
using dbg::OutputAsTypeWide;
using dbg::GetSymbolTypeName;
using dbg::GetSymbolTypeNameWide;
using dbg::GetSymbolSize;
using dbg::GetSymbolOffset;
using dbg::GetSymbolRegister;
using dbg::GetSymbolValueText;
using dbg::GetSymbolValueTextWide;
using dbg::GetSymbolEntryInformation;

// In dbgsymbols.h
using dbg::GetSymbolOptions;
using dbg::AddSymbolOptions;
using dbg::RemoveSymbolOptions;
using dbg::SetSymbolOptions;
using dbg::GetNameByOffset;
using dbg::GetOffsetByName;
using dbg::GetNearNameByOffset;
using dbg::GetLineByOffset;
using dbg::GetOffsetByLine;
using dbg::GetNumberModules;
using dbg::GetModuleByIndex;
using dbg::GetModuleByModuleName;
using dbg::GetModuleByOffset;
using dbg::GetModuleNames;
using dbg::GetModuleParameters;
using dbg::GetSymbolModule;
using dbg::GetTypeName;
using dbg::GetTypeId;
using dbg::GetTypeSize;
using dbg::GetFieldOffset;
using dbg::GetSymbolTypeId;
using dbg::GetOffsetTypeId;
using dbg::ReadTypedDataVirtual;
using dbg::WriteTypedDataVirtual;
using dbg::OutputTypedDataVirtual;
using dbg::ReadTypedDataPhysical;
using dbg::WriteTypedDataPhysical;
using dbg::OutputTypedDataPhysical;
using dbg::GetScope;
using dbg::SetScope;
using dbg::ResetScope;
using dbg::GetScopeSymbolGroup;
using dbg::CreateSymbolGroup;
using dbg::StartSymbolMatch;
using dbg::GetNextSymbolMatch;
using dbg::EndSymbolMatch;
using dbg::Reload;
using dbg::GetSymbolPath;
using dbg::SetSymbolPath;
using dbg::AppendSymbolPath;
using dbg::GetImagePath;
using dbg::SetImagePath;
using dbg::AppendImagePath;
using dbg::GetSourcePath;
using dbg::GetSourcePathElement;
using dbg::SetSourcePath;
using dbg::AppendSourcePath;
using dbg::FindSourceFile;
using dbg::GetSourceFileLineOffsets;

using dbg::GetModuleVersionInformation;
using dbg::GetModuleNameString;
using dbg::GetConstantName;
using dbg::GetFieldName;
using dbg::GetTypeOptions;
using dbg::AddTypeOptions;
using dbg::RemoveTypeOptions;
using dbg::SetTypeOptions;

using dbg::GetNameByOffsetWide;
using dbg::GetOffsetByNameWide;
using dbg::GetNearNameByOffsetWide;
using dbg::GetLineByOffsetWide;
using dbg::GetOffsetByLineWide;
using dbg::GetModuleByModuleNameWide;
using dbg::GetSymbolModuleWide;
using dbg::GetTypeNameWide;
using dbg::GetTypeIdWide;
using dbg::GetFieldOffsetWide;
using dbg::GetSymbolTypeIdWide;
using dbg::GetScopeSymbolGroup2;
using dbg::CreateSymbolGroup2;
using dbg::StartSymbolMatchWide;
using dbg::GetNextSymbolMatchWide;
using dbg::ReloadWide;
using dbg::GetSymbolPathWide;
using dbg::SetSymbolPathWide;
using dbg::AppendSymbolPathWide;
using dbg::GetImagePathWide;
using dbg::SetImagePathWide;
using dbg::AppendImagePathWide;
using dbg::GetSourcePathWide;
using dbg::GetSourcePathElementWide;
using dbg::SetSourcePathWide;
using dbg::AppendSourcePathWide;
using dbg::FindSourceFileWide;
using dbg::GetSourceFileLineOffsetsWide;
using dbg::GetModuleVersionInformationWide;
using dbg::GetModuleNameStringWide;
using dbg::GetConstantNameWide;
using dbg::GetFieldNameWide;
using dbg::IsManagedModule;
using dbg::GetModuleByModuleName2;
using dbg::GetModuleByModuleName2Wide;
using dbg::GetModuleByOffset2;
using dbg::AddSyntheticModule;
using dbg::AddSyntheticModuleWide;
using dbg::RemoveSyntheticModule;
using dbg::GetLwrrentScopeFrameIndex;
using dbg::SetScopeFrameByIndex;
using dbg::SetScopeFromJitDebugInfo;
using dbg::SetScopeFromStoredEvent;
using dbg::OutputSymbolByOffset;
using dbg::GetFunctionEntryByOffset;
using dbg::GetFieldTypeAndOffset;
using dbg::GetFieldTypeAndOffsetWide;
using dbg::AddSyntheticSymbol;
using dbg::AddSyntheticSymbolWide;
using dbg::RemoveSyntheticSymbol;
using dbg::GetSymbolEntriesByOffset;
using dbg::GetSymbolEntriesByName;
using dbg::GetSymbolEntriesByNameWide;
using dbg::GetSymbolEntryByToken;
using dbg::GetSymbolEntryInformation;
using dbg::GetSymbolEntryString;
using dbg::GetSymbolEntryStringWide;
using dbg::GetSymbolEntryOffsetRegions;
using dbg::GetSymbolEntryBySymbolEntry;
using dbg::GetSourceEntriesByOffset;
using dbg::GetSourceEntriesByLine;
using dbg::GetSourceEntriesByLineWide;
using dbg::GetSourceEntryString;
using dbg::GetSourceEntryStringWide;
using dbg::GetSourceEntryOffsetRegions;
using dbg::GetSourceEntryBySourceEntry;

using dbg::GetNameByOffset;
using dbg::GetNearNameByOffset;
using dbg::GetLineByOffset;
using dbg::GetModuleByOffset;
using dbg::GetModuleNames;
using dbg::GetTypeName;
using dbg::GetTypeId;
using dbg::GetTypeSize;
using dbg::GetFieldOffset;
using dbg::GetOffsetTypeId;
using dbg::ReadTypedDataVirtual;
using dbg::WriteTypedDataVirtual;
using dbg::OutputTypedDataVirtual;
using dbg::ReadTypedDataPhysical;
using dbg::WriteTypedDataPhysical;
using dbg::OutputTypedDataPhysical;
using dbg::SetScope;
using dbg::GetNextSymbolMatch;
using dbg::EndSymbolMatch;
using dbg::GetModuleVersionInformation;
using dbg::GetModuleNameString;
using dbg::GetConstantName;
using dbg::GetFieldName;
using dbg::GetNameByOffsetWide;
using dbg::GetNearNameByOffsetWide;
using dbg::GetLineByOffsetWide;
using dbg::GetTypeNameWide;
using dbg::GetTypeIdWide;
using dbg::GetFieldOffsetWide;
using dbg::GetNextSymbolMatchWide;
using dbg::GetModuleVersionInformationWide;
using dbg::GetModuleNameStringWide;
using dbg::GetConstantNameWide;
using dbg::GetFieldNameWide;
using dbg::IsManagedModule;
using dbg::GetModuleByOffset2;
using dbg::AddSyntheticModule;
using dbg::AddSyntheticModuleWide;
using dbg::RemoveSyntheticModule;
using dbg::SetScopeFromJitDebugInfo;
using dbg::OutputSymbolByOffset;
using dbg::GetFunctionEntryByOffset;
using dbg::GetFieldTypeAndOffset;
using dbg::GetFieldTypeAndOffsetWide;
using dbg::AddSyntheticSymbol;
using dbg::AddSyntheticSymbolWide;
using dbg::GetSymbolEntriesByOffset;
using dbg::GetSourceEntriesByOffset;

// In dbgsystemobjects
using dbg::GetEventThread;
using dbg::GetEventProcess;
using dbg::GetLwrrentThreadId;
using dbg::SetLwrrentThreadId;
using dbg::GetLwrrentProcessId;
using dbg::SetLwrrentProcessId;
using dbg::GetNumberThreads;
using dbg::GetTotalNumberThreads;
using dbg::GetThreadIdsByIndex;
using dbg::GetThreadIdByProcessor;
using dbg::GetLwrrentThreadDataOffset;
using dbg::GetThreadIdByDataOffset;
using dbg::GetLwrrentThreadTeb;
using dbg::GetThreadIdByTeb;
using dbg::GetLwrrentThreadSystemId;
using dbg::GetThreadIdBySystemId;
using dbg::GetLwrrentThreadHandle;
using dbg::GetThreadIdByHandle;
using dbg::GetNumberProcesses;
using dbg::GetProcessIdsByIndex;
using dbg::GetLwrrentProcessDataOffset;
using dbg::GetProcessIdByDataOffset;
using dbg::GetLwrrentProcessPeb;
using dbg::GetProcessIdByPeb;
using dbg::GetLwrrentProcessSystemId;
using dbg::GetProcessIdBySystemId;
using dbg::GetLwrrentProcessHandle;
using dbg::GetProcessIdByHandle;
using dbg::GetLwrrentProcessExelwtableName;

using dbg::GetLwrrentProcessUpTime;
using dbg::GetImplicitThreadDataOffset;
using dbg::SetImplicitThreadDataOffset;
using dbg::GetImplicitProcessDataOffset;
using dbg::SetImplicitProcessDataOffset;

using dbg::GetEventSystem;
using dbg::GetLwrrentSystemId;
using dbg::SetLwrrentSystemId;
using dbg::GetNumberSystems;
using dbg::GetSystemIdsByIndex;
using dbg::GetTotalNumberThreadsAndProcesses;
using dbg::GetLwrrentSystemServer;
using dbg::GetSystemByServer;
using dbg::GetLwrrentSystemServerName;

using dbg::GetLwrrentProcessExelwtableNameWide;
using dbg::GetLwrrentSystemServerNameWide;

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGNAME_H
