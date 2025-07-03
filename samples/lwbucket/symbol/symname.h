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
|*  Module: symname.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _SYMNAME_H
#define _SYMNAME_H

//******************************************************************************
//
// sym namespace entries
//
//******************************************************************************
// In symtypes.h
using sym::CData;
using sym::CModule;
using sym::CType;
using sym::CField;
using sym::CEnum;
using sym::CValue;
using sym::CGlobal;
using sym::CMember;
using sym::CClass;
using sym::CSymbolProcess;
using sym::CSymbolSession;
using sym::CModuleInstance;
using sym::CTypeInstance;
using sym::CFieldInstance;
using sym::CEnumInstance;
using sym::CGlobalInstance;
using sym::CSessionMember;
using sym::CProcessMember;
using sym::CMemberType;
using sym::CMemberField;
using sym::CSymbolSet;

using sym::CModulePtr;
using sym::CTypePtr;
using sym::CFieldPtr;
using sym::CEnumPtr;
using sym::CGlobalPtr;
using sym::CSymbolSetPtr;

using sym::CSymbolProcessPtr;
using sym::CSymbolSessionPtr;

using sym::CModuleArray;
using sym::CTypeArray;
using sym::CFieldArray;
using sym::CEnumArray;
using sym::CGlobalArray;

using sym::CProcessObject;
using sym::CSessionObject;

using sym::CProcessList;
using sym::CSessionList;

using sym::DataType;

using sym::SYM_INFO;
using sym::TYPE_INFO;
using sym::FINDCHILDREN_PARAMS;

// In symdbghelp.h
using sym::initializeDbgHelp;
using sym::uninitializeDbgHelp;

using sym::symbolHandle;
using sym::symbolReset;

using sym::symProperty;
using sym::symTag;
using sym::symName;
using sym::symLength;
using sym::symType;
using sym::symTypeId;
using sym::symBaseType;
using sym::symArrayIndexTypeId;
using sym::symDataKind;
using sym::symAddressOffset;
using sym::symOffset;
using sym::symValue;
using sym::symCount;
using sym::symChildrenCount;
using sym::symBitPosition;
using sym::symVirtualBaseClass;
using sym::symVirtualTableShapeId;
using sym::symVirtualBasePointerOffset;
using sym::symClassParentId;
using sym::symNested;
using sym::symSymIndex;
using sym::symLexicalParent;
using sym::symAddress;
using sym::symThisAdjust;
using sym::symUdtKind;
using sym::symEquivTo;
using sym::symCallingColwention;
using sym::symCloseEquivTo;
using sym::symGtiExReqsValid;
using sym::symVirtualBaseOffset;
using sym::symVirtualBaseDispIndex;
using sym::symIsReference;
using sym::symIndirectVirtualBaseClass;
using sym::symVirtualBaseTableType;

using sym::pointerType;
using sym::pointerCount;

using sym::arrayType;
using sym::arrayDimensions;

using sym::symbolTypeName;
using sym::symbolTagName;
using sym::callingColwentionName;
using sym::udtKindName;

using sym::enumDirTree;
using sym::enumDirTreeW;
using sym::makeSureDirectoryPathExists;
using sym::searchTreeForFile;
using sym::searchTreeForFileW;

using sym::enumerateLoadedModules;
using sym::enumerateLoadedModules64;
using sym::enumerateLoadedModulesW64;
using sym::enumerateLoadedModulesEx;
using sym::enumerateLoadedModulesExW;
using sym::findDebugInfoFile;
using sym::findDebugInfoFileEx;
using sym::findDebugInfoFileExW;
using sym::findExelwtableImage;
using sym::findExelwtableImageEx;
using sym::findExelwtableImageExW;
#ifdef _WIN64
using sym::stackWalk64;
#else
using sym::stackWalk;
using sym::stackWalk64;
#endif
using sym::symSetParentWindow;
using sym::unDecorateSymbolName;
using sym::unDecorateSymbolNameW;

using sym::imageDirectoryEntryToData;
using sym::imageDirectoryEntryToDataEx;
using sym::imageNtHeader;
using sym::imageRvaToSection;
using sym::imageRvaToVa;

using sym::symAddSymbol;
using sym::symAddSymbolW;
using sym::symCleanup;
using sym::symDeleteSymbol;
using sym::symDeleteSymbolW;
#ifdef _WIN64
using sym::symEnumerateModules64;
#else
using sym::symEnumerateModules;
using sym::symEnumerateModules64;
#endif
using sym::symEnumerateModulesW64;
using sym::symEnumLines;
using sym::symEnumLinesW;
using sym::symEnumProcesses;
using sym::symEnumSourceFiles;
using sym::symEnumSourceFilesW;
using sym::symEnumSourceLines;
using sym::symEnumSourceLinesW;
using sym::symEnumSymbols;
using sym::symEnumSymbolsW;
using sym::symEnumSymbolsForAddr;
using sym::symEnumSymbolsForAddrW;
using sym::symEnumTypes;
using sym::symEnumTypesW;
using sym::symEnumTypesByName;
using sym::symEnumTypesByNameW;
using sym::symFindDebugInfoFile;
using sym::symFindDebugInfoFileW;
using sym::symFindExelwtableImage;
using sym::symFindExelwtableImageW;
using sym::symFindFileInPath;
using sym::symFindFileInPathW;
using sym::symFromAddr;
using sym::symFromAddrW;
using sym::symFromIndex;
using sym::symFromIndexW;
using sym::symFromName;
using sym::symFromNameW;
using sym::symFromToken;
using sym::symFromTokenW;
using sym::symFunctionTableAccess;
using sym::symFunctionTableAccess64;
using sym::symGetFileLineOffsets64;
using sym::symGetHomeDirectory;
using sym::symGetHomeDirectoryW;
using sym::symGetLineFromAddr;
using sym::symGetLineFromAddr64;
using sym::symGetLineFromAddrW64;
using sym::symGetLineFromName;
using sym::symGetLineFromName64;
using sym::symGetLineFromNameW64;
using sym::symGetLineNext;
using sym::symGetLineNext64;
using sym::symGetLineNextW64;
using sym::symGetLinePrev;
using sym::symGetLinePrev64;
using sym::symGetLinePrevW64;
#ifdef _WIN64
using sym::symGetModuleBase64;
#else
using sym::symGetModuleBase;
using sym::symGetModuleBase64;
#endif
using sym::symGetModuleInfo;
using sym::symGetModuleInfoW;
using sym::symGetModuleInfo64;
using sym::symGetModuleInfoW64;
using sym::symGetOmaps;
using sym::symGetOptions;
using sym::symGetScope;
using sym::symGetScopeW;
using sym::symGetSearchPath;
using sym::symGetSearchPathW;
using sym::symGetSymbolFile;
using sym::symGetSymbolFileW;
using sym::symGetTypeFromName;
using sym::symGetTypeFromNameW;
using sym::symGetTypeInfo;
using sym::symGetTypeInfoEx;
using sym::symGetUnwindInfo;
using sym::symInitialize;
using sym::symInitializeW;
#ifdef _WIN64
using sym::symLoadModule64;
#else
using sym::symLoadModule;
using sym::symLoadModule64;
#endif
using sym::symLoadModuleEx;
using sym::symLoadModuleExW;
using sym::symMatchFileName;
using sym::symMatchFileNameW;
using sym::symMatchString;
using sym::symMatchStringA;
using sym::symMatchStringW;
using sym::symNext;
using sym::symNextW;
using sym::symPrev;
using sym::symPrevW;
using sym::symRefreshModuleList;
#ifdef _WIN64
using sym::symRegisterCallback64;
using sym::symRegisterFunctionEntryCallback64;
#else
using sym::symRegisterCallback;
using sym::symRegisterFunctionEntryCallback;
using sym::symRegisterCallback64;
using sym::symRegisterFunctionEntryCallback64;
#endif
using sym::symRegisterCallbackW64;
using sym::symSearch;
using sym::symSearchW;
using sym::symSetContext;
using sym::symSetHomeDirectory;
using sym::symSetHomeDirectoryW;
using sym::symSetOptions;
using sym::symSetScopeFromAddr;
using sym::symSetScopeFromIndex;
using sym::symSetSearchPath;
using sym::symSetSearchPathW;
#ifdef _WIN64
using sym::symUnDName64;
using sym::symUnloadModule64;
#else
using sym::symUnDName;
using sym::symUnloadModule;
using sym::symUnDName64;
using sym::symUnloadModule64;
#endif

using sym::symSrvDeltaName;
using sym::symSrvDeltaNameW;
using sym::symSrvGetFileIndexes;
using sym::symSrvGetFileIndexesW;
using sym::symSrvGetFileIndexInfo;
using sym::symSrvGetFileIndexInfoW;
using sym::symSrvGetFileIndexString;
using sym::symSrvGetFileIndexStringW;
using sym::symSrvGetSupplement;
using sym::symSrvGetSupplementW;
using sym::symSrvIsStore;
using sym::symSrvIsStoreW;
using sym::symSrvStoreFile;
using sym::symSrvStoreFileW;
using sym::symSrvStoreSupplement;
using sym::symSrvStoreSupplementW;

using sym::symGetSourceFile;
using sym::symGetSourceFileW;
using sym::symEnumSourceFileTokens;
using sym::symGetSourceFileFromToken;
using sym::symGetSourceFileFromTokenW;
using sym::symGetSourceFileToken;
using sym::symGetSourceFileTokenW;
using sym::symGetSourceVarFromToken;
using sym::symGetSourceVarFromTokenW;

using sym::symGetSymFromAddr;
using sym::symGetSymFromName;
using sym::symGetSymNext;
using sym::symGetSymPrev;
using sym::symGetSymFromAddr64;
using sym::symGetSymFromName64;
using sym::symGetSymNext64;
using sym::symGetSymPrev64;

// In symhandler.h
using sym::initializeSymbols;
using sym::uninitializeSymbols;

using sym::acquireSymbolOperation;
using sym::releaseSymbolOperation;
using sym::symbolOperation;

using sym::reloadSymbols;
using sym::resetSymbols;

using sym::symbolName;
using sym::symbolDump;

// In symmodule.h
using sym::ModuleType;
using sym::KernelModule;
using sym::UserModule;

using sym::firstModule;
using sym::lastModule;
using sym::modulesCount;

using sym::firstKernelModule;
using sym::lastKernelModule;
using sym::kernelModuleCount;

using sym::firstUserModule;
using sym::lastUserModule;
using sym::userModuleCount;

using sym::findModule;
using sym::findKernelModule;
using sym::findUserModule;

using sym::loadModuleInformation;
using sym::resetModuleInformation;

using sym::loadModuleSymbols;
using sym::unloadModuleSymbols;
using sym::reloadModuleSymbols;

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _SYMNAME_H
