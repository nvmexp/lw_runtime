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
|*  Module: symdbghelp.h                                                      *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _SYMDBGHELP_H
#define _SYMDBGHELP_H

//******************************************************************************
//
//  sym namespace
//
//******************************************************************************
namespace sym
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define MAX_SYMBOL_PATH         1024                // Maximum symbol path
#define MAX_TYPE_LENGTH         256                 // Maximum type length

#define DEFAULT_SYMBOL_PATH     "http://DispSym/Sym;http://msdl.microsoft.com/download/symbols"

//******************************************************************************
//
//  Unions
//
//******************************************************************************
typedef union _TYPE_INFO
{
    DWORD           dwSymTag;
    WCHAR          *pSymName;
    ULONG64         ulLength;
    DWORD           dwType;
    DWORD           dwTypeId;
    DWORD           dwBaseType;
    DWORD           dwArrayIndexTypeId;
    TI_FINDCHILDREN_PARAMS *pFindChildrenParams;
    DWORD           dwDataKind;
    DWORD           dwAddressOffset;
    DWORD           dwOffset;
    VARIANT         vValue;
    DWORD           dwCount;
    DWORD           dwChildrenCount;
    DWORD           dwBitPosition;
    BOOL            bVirtualBaseClass;
    DWORD           dwVirtualTableShapeId;
    DWORD           dwVirtualBasePointerOffset;
    DWORD           dwClassParentId;
    DWORD           dwNested;
    DWORD           dwSymIndex;
    DWORD           dwLexicalParent;
    ULONG64         ulAddress;
    DWORD           dwThisAdjust;
    DWORD           dwUdtKind;
    DWORD           dwEquiv;
    DWORD           dwCallingColwention;
    DWORD           dwCloseEquiv;
    ULONG64         ulGtiExReqsValid;
    DWORD           dwVirtualBaseOffset;
    DWORD           dwVirtualBaseDispIndex;
    BOOLEAN         bReference;
    BOOL            bIndirectVirtualBaseClass;
    DWORD           dwVirtualBaseTableType;

} TYPE_INFO, *PTYPE_INFO;

//******************************************************************************
//
//  Type Definitions
//
//******************************************************************************






//******************************************************************************
//
//  Forwards
//
//******************************************************************************






//******************************************************************************
//
// Structures
//
//******************************************************************************







//******************************************************************************
//
// Macros
//
//******************************************************************************




//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  HRESULT         initializeDbgHelp();
extern  HRESULT         uninitializeDbgHelp();

extern  HANDLE          symbolHandle();
extern  HRESULT         symbolReset();

extern  bool            symProperty(const CModule* pModule, DWORD dwIndex, IMAGEHLP_SYMBOL_TYPE_INFO property);
extern  DWORD           symTag(const CModule* pModule, DWORD dwIndex);
extern  CString         symName(const CModule* pModule, DWORD dwIndex);
extern  ULONG64         symLength(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symType(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symTypeId(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symBaseType(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symArrayIndexTypeId(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symDataKind(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symAddressOffset(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symOffset(const CModule* pModule, DWORD dwIndex);
extern  VARIANT         symValue(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symCount(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symChildrenCount(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symBitPosition(const CModule* pModule, DWORD dwIndex);
extern  bool            symVirtualBaseClass(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symVirtualTableShapeId(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symVirtualBasePointerOffset(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symClassParentId(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symNested(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symSymIndex(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symLexicalParent(const CModule* pModule, DWORD dwIndex);
extern  ULONG64         symAddress(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symThisAdjust(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symUdtKind(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symEquivTo(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symCallingColwention(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symCloseEquivTo(const CModule* pModule, DWORD dwIndex);
extern  ULONG64         symGtiExReqsValid(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symVirtualBaseOffset(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symVirtualBaseDispIndex(const CModule* pModule, DWORD dwIndex);
extern  bool            symIsReference(const CModule* pModule, DWORD dwIndex);
extern  bool            symIndirectVirtualBaseClass(const CModule* pModule, DWORD dwIndex);
extern  DWORD           symVirtualBaseTableType(const CModule* pModule, DWORD dwIndex);

extern  DWORD           pointerType(const CModule* pModule, DWORD dwIndex);
extern  DWORD           pointerCount(const CModule* pModule, DWORD dwIndex);

extern  DWORD           arrayType(const CModule* pModule, DWORD dwIndex);
extern  DWORD           arrayDimensions(const CModule* pModule, DWORD dwIndex);

extern  const char*     symbolTypeName(DWORD dwSymType);
extern  const char*     symbolTagName(DWORD dwSymTag);
extern  const char*     callingColwentionName(DWORD dwCallingColwention);
extern  const char*     udtKindName(DWORD dwUdtKind);

// General DbgHelp routines
extern  BOOL            enumDirTree(PCSTR RootPath, PCSTR InputPathName, PSTR OutputPathBuffer, PENUMDIRTREE_CALLBACK cb, PVOID data);
extern  BOOL            enumDirTreeW(PCWSTR RootPath, PCWSTR InputPathName, PWSTR OutputPathBuffer, PENUMDIRTREE_CALLBACKW cb, PVOID data);
extern  BOOL            makeSureDirectoryPathExists(PCSTR DirPath);
extern  BOOL            searchTreeForFile(PCSTR RootPath, PCSTR InputPathName, PSTR OutputPathBuffer);
extern  BOOL            searchTreeForFileW(PCWSTR RootPath, PCWSTR InputPathName, PWSTR OutputPathBuffer);

// Debugger DbgHelp routines
extern  BOOL            enumerateLoadedModules(PENUMLOADED_MODULES_CALLBACK EnumLoadedModulesCallback, PVOID UserContext);
extern  BOOL            enumerateLoadedModules64(PENUMLOADED_MODULES_CALLBACK64 EnumLoadedModulesCallback, PVOID UserContext);
extern  BOOL            enumerateLoadedModulesW64(PENUMLOADED_MODULES_CALLBACKW64 EnumLoadedModulesCallback, PVOID UserContext);
extern  BOOL            enumerateLoadedModulesEx(PENUMLOADED_MODULES_CALLBACK64 EnumLoadedModulesCallback, PVOID UserContext);
extern  BOOL            enumerateLoadedModulesExW(PENUMLOADED_MODULES_CALLBACKW64 EnumLoadedModulesCallback, PVOID UserContext);
extern  HANDLE          findDebugInfoFile(PCSTR FileName, PCSTR SymbolPath, PSTR DebugFilePath);
extern  HANDLE          findDebugInfoFileEx(PCSTR FileName, PCSTR SymbolPath, PSTR DebugFilePath, PFIND_DEBUG_FILE_CALLBACK Callback, PVOID CallerData);
extern  HANDLE          findDebugInfoFileExW(PCWSTR FileName, PCWSTR SymbolPath, PWSTR DebugFilePath, PFIND_DEBUG_FILE_CALLBACKW Callback, PVOID CallerData);
extern  HANDLE          findExelwtableImage(PCSTR FileName, PCSTR SymbolPath, PSTR ImageFilePath);
extern  HANDLE          findExelwtableImageEx(PCSTR FileName, PCSTR SymbolPath, PSTR ImageFilePath, PFIND_EXE_FILE_CALLBACK Callback, PVOID CallerData);
extern  HANDLE          findExelwtableImageExW(PCWSTR FileName, PCWSTR SymbolPath, PWSTR ImageFilePath, PFIND_EXE_FILE_CALLBACKW Callback, PVOID CallerData);
#ifdef _WIN64
#define StackWalk StackWalk64
#else
extern  BOOL            stackWalk(DWORD MachineType, HANDLE hProcess, HANDLE hThread, LPSTACKFRAME StackFrame, PVOID ContextRecord, PREAD_PROCESS_MEMORY_ROUTINE ReadMemoryRoutine, PFUNCTION_TABLE_ACCESS_ROUTINE FunctionTableAccessRoutine, PGET_MODULE_BASE_ROUTINE GetModuleBaseRoutine, PTRANSLATE_ADDRESS_ROUTINE TranslateAddress);
#endif
extern  BOOL            stackWalk64(DWORD MachineType, HANDLE hProcess, HANDLE hThread, LPSTACKFRAME64 StackFrame, PVOID ContextRecord, PREAD_PROCESS_MEMORY_ROUTINE64 ReadMemoryRoutine, PFUNCTION_TABLE_ACCESS_ROUTINE64 FunctionTableAccessRoutine, PGET_MODULE_BASE_ROUTINE64 GetModuleBaseRoutine, PTRANSLATE_ADDRESS_ROUTINE64 TranslateAddress);
extern  BOOL            symSetParentWindow(HWND hwnd);
extern  DWORD           unDecorateSymbolName(PCSTR name, PSTR outputString, DWORD maxStringLength, DWORD flags);
extern  DWORD           unDecorateSymbolNameW(PCWSTR name, PWSTR outputString, DWORD maxStringLength, DWORD flags);

// Image access DbgHelp routines
extern  PVOID           imageDirectoryEntryToData(PVOID Base, BOOLEAN MappedAsImage, USHORT DirectoryEntry, PULONG Size);
extern  PVOID           imageDirectoryEntryToDataEx(PVOID Base, BOOLEAN MappedAsImage, USHORT DirectoryEntry, PULONG Size, PIMAGE_SECTION_HEADER *FoundHeader);
extern  PIMAGE_NT_HEADERS imageNtHeader(PVOID Base);
extern  PIMAGE_SECTION_HEADER imageRvaToSection(PIMAGE_NT_HEADERS NtHeaders, PVOID Base, ULONG Rva);
extern  PVOID           imageRvaToVa(PIMAGE_NT_HEADERS NtHeaders, PVOID Base, ULONG Rva, PIMAGE_SECTION_HEADER *LastRvaSection);

// Symbol handler DbgHelp routines
extern  BOOL            symAddSymbol(ULONG64 BaseOfDll, PCSTR Name, DWORD64 Address, DWORD Size, DWORD Flags);
extern  BOOL            symAddSymbolW(ULONG64 BaseOfDll, PCWSTR Name, DWORD64 Address, DWORD Size, DWORD Flags);
extern  BOOL            symCleanup();
extern  BOOL            symDeleteSymbol(ULONG64 BaseOfDll, PCSTR Name, DWORD64 Address, DWORD Flags);
extern  BOOL            symDeleteSymbolW(ULONG64 BaseOfDll, PCWSTR Name, DWORD64 Address, DWORD Flags);
#ifdef _WIN64
#define SymEnumerateModules SymEnumerateModules64
#else
extern  BOOL            symEnumerateModules(PSYM_ENUMMODULES_CALLBACK EnumModulesCallback, PVOID UserContext);
#endif
extern  BOOL            symEnumerateModules64(PSYM_ENUMMODULES_CALLBACK64 EnumModulesCallback, PVOID UserContext);
extern  BOOL            symEnumerateModulesW64(PSYM_ENUMMODULES_CALLBACKW64 EnumModulesCallback, PVOID UserContext);
extern  BOOL            symEnumLines(ULONG64 Base, PCSTR Obj, PCSTR File, PSYM_ENUMLINES_CALLBACK EnumLinesCallback, PVOID UserContext);
extern  BOOL            symEnumLinesW(ULONG64 Base, PCWSTR Obj, PCWSTR File, PSYM_ENUMLINES_CALLBACKW EnumLinesCallback, PVOID UserContext);
extern  BOOL            symEnumProcesses(PSYM_ENUMPROCESSES_CALLBACK EnumProcessesCallback, PVOID UserContext);
extern  BOOL            symEnumSourceFiles(ULONG64 ModBase, PCSTR Mask, PSYM_ENUMSOURCEFILES_CALLBACK cbSrcFiles, PVOID UserContext);
extern  BOOL            symEnumSourceFilesW(ULONG64 ModBase, PCWSTR Mask, PSYM_ENUMSOURCEFILES_CALLBACKW cbSrcFiles, PVOID UserContext);
extern  BOOL            symEnumSourceLines(ULONG64 Base, PCSTR Obj, PCSTR File, DWORD Line, DWORD Flags, PSYM_ENUMLINES_CALLBACK EnumLinesCallback, PVOID UserContext);
extern  BOOL            symEnumSourceLinesW(ULONG64 Base, PCWSTR Obj, PCWSTR File, DWORD Line, DWORD Flags, PSYM_ENUMLINES_CALLBACKW EnumLinesCallback, PVOID UserContext);
extern  BOOL            symEnumSymbols(ULONG64 BaseOfDll, PCSTR Mask, PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback, PVOID UserContext);
extern  BOOL            symEnumSymbolsW(ULONG64 BaseOfDll, PCWSTR Mask, PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback, PVOID UserContext);
extern  BOOL            symEnumSymbolsForAddr(DWORD64 Address, PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback, PVOID UserContext);
extern  BOOL            symEnumSymbolsForAddrW(DWORD64 Address, PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback, PVOID UserContext);
extern  BOOL            symEnumTypes(ULONG64 BaseOfDll, PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback, PVOID UserContext);
extern  BOOL            symEnumTypesW(ULONG64 BaseOfDll, PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback, PVOID UserContext);
extern  BOOL            symEnumTypesByName(ULONG64 BaseOfDll, PCSTR mask, PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback, PVOID UserContext);
extern  BOOL            symEnumTypesByNameW(ULONG64 BaseOfDll, PCWSTR mask, PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback, PVOID UserContext);
extern  HANDLE          symFindDebugInfoFile(PCSTR FileName, PSTR DebugFilePath, PFIND_DEBUG_FILE_CALLBACK Callback, PVOID CallerData);
extern  HANDLE          symFindDebugInfoFileW(PCWSTR FileName, PWSTR DebugFilePath, PFIND_DEBUG_FILE_CALLBACKW Callback, PVOID CallerData);
extern  HANDLE          symFindExelwtableImage(PCSTR FileName, PSTR ImageFilePath, PFIND_EXE_FILE_CALLBACK Callback, PVOID CallerData);
extern  HANDLE          symFindExelwtableImageW(PCWSTR FileName, PWSTR ImageFilePath, PFIND_EXE_FILE_CALLBACKW Callback, PVOID CallerData);
extern  BOOL            symFindFileInPath(PCSTR SearchPath, PCSTR FileName, PVOID id, DWORD two, DWORD three, DWORD flags, PSTR FoundFile, PFINDFILEINPATHCALLBACK callback, PVOID context);
extern  BOOL            symFindFileInPathW(PCWSTR SearchPath, PCWSTR FileName, PVOID id, DWORD two, DWORD three, DWORD flags, PWSTR FoundFile, PFINDFILEINPATHCALLBACKW callback, PVOID context);
extern  BOOL            symFromAddr(DWORD64 Address, PDWORD64 Displacement, PSYMBOL_INFO Symbol);
extern  BOOL            symFromAddrW(DWORD64 Address, PDWORD64 Displacement, PSYMBOL_INFOW Symbol);
extern  BOOL            symFromIndex(ULONG64 BaseOfDll, DWORD Index, PSYMBOL_INFO Symbol);
extern  BOOL            symFromIndexW(ULONG64 BaseOfDll, DWORD Index, PSYMBOL_INFOW Symbol);
extern  BOOL            symFromName(PCSTR Name, PSYMBOL_INFO Symbol);
extern  BOOL            symFromNameW(PCWSTR Name, PSYMBOL_INFOW Symbol);
extern  BOOL            symFromToken(DWORD64 Base, DWORD Token, PSYMBOL_INFO Symbol);
extern  BOOL            symFromTokenW(DWORD64 Base, DWORD Token, PSYMBOL_INFOW Symbol);
extern  PVOID           symFunctionTableAccess(DWORD AddrBase);
extern  PVOID           symFunctionTableAccess64(DWORD64 AddrBase);
extern  ULONG           symGetFileLineOffsets64(PCSTR ModuleName, PCSTR FileName, PDWORD64 Buffer, ULONG BufferLines);
extern  PCHAR           symGetHomeDirectory(DWORD type, PSTR dir, size_t size);
extern  PWSTR           symGetHomeDirectoryW(DWORD type, PWSTR dir, size_t size);
#ifdef _WIN64
#define symGetLineFromAddr symGetLineFromAddr64
#define symGetLineFromAddrW symGetLineFromAddrW64
#else
extern  BOOL            symGetLineFromAddr(DWORD dwAddr, PDWORD pdwDisplacement, PIMAGEHLP_LINE Line);
#endif
extern  BOOL            symGetLineFromAddr64(DWORD64 qwAddr, PDWORD pdwDisplacement, PIMAGEHLP_LINE64 Line64);
extern  BOOL            symGetLineFromAddrW64(DWORD64 dwAddr, PDWORD pdwDisplacement, PIMAGEHLP_LINEW64 Line);
#ifdef _WIN64
#define symGetLineFromName symGetLineFromName64
#else
extern  BOOL            symGetLineFromName(PCSTR ModuleName, PCSTR FileName, DWORD dwLineNumber, PLONG plDisplacement, PIMAGEHLP_LINE Line);
#endif
extern  BOOL            symGetLineFromName64(PCSTR ModuleName, PCSTR FileName, DWORD dwLineNumber, PLONG plDisplacement, PIMAGEHLP_LINE64 Line);
extern  BOOL            symGetLineFromNameW64(PCWSTR ModuleName, PCWSTR FileName, DWORD dwLineNumber, PLONG plDisplacement, PIMAGEHLP_LINEW64 Line);
#ifdef _WIN64
#define symGetLineNext symGetLineNext64
#define symGetLineNextW symGetLineNextW64
#else
extern  BOOL            symGetLineNext(PIMAGEHLP_LINE Line);
#endif
extern  BOOL            symGetLineNext64(PIMAGEHLP_LINE64 Line);
extern  BOOL            symGetLineNextW64(PIMAGEHLP_LINEW64 Line);
#ifdef _WIN64
#define symGetLinePrev symGetLinePrev64
#define symGetLinePrevW symGetLinePrevW64
#else
extern  BOOL            symGetLinePrev(PIMAGEHLP_LINE Line);
#endif
extern  BOOL            symGetLinePrev64(PIMAGEHLP_LINE64 Line);
extern  BOOL            symGetLinePrevW64(PIMAGEHLP_LINEW64 Line);
#ifdef _WIN64
#define SymGetModuleBase SymGetModuleBase64
#else
extern  DWORD           symGetModuleBase(DWORD dwAddr);
#endif
extern  DWORD64         symGetModuleBase64(DWORD64 qwAddr);
#ifdef _WIN64
#define symGetModuleInfo symGetModuleInfo64
#define symGetModuleInfoW symGetModuleInfoW64
#else
extern  BOOL            symGetModuleInfo(DWORD dwAddr, PIMAGEHLP_MODULE ModuleInfo);
extern  BOOL            symGetModuleInfoW(DWORD dwAddr, PIMAGEHLP_MODULEW ModuleInfo);
#endif
extern  BOOL            symGetModuleInfo64(DWORD64 qwAddr, PIMAGEHLP_MODULE64 ModuleInfo);
extern  BOOL            symGetModuleInfoW64(DWORD64 qwAddr, PIMAGEHLP_MODULEW64 ModuleInfo);
extern  BOOL            symGetOmaps(DWORD64 BaseOfDll, POMAP *OmapTo, PDWORD64 cOmapTo, POMAP *OmapFrom, PDWORD64 cOmapFrom);
extern  DWORD           symGetOptions(VOID);
extern  BOOL            symGetScope(ULONG64 BaseOfDll, DWORD Index, PSYMBOL_INFO Symbol);
extern  BOOL            symGetScopeW(ULONG64 BaseOfDll, DWORD Index, PSYMBOL_INFOW Symbol);
extern  BOOL            symGetSearchPath(PSTR SearchPath, DWORD SearchPathLength);
extern  BOOL            symGetSearchPathW(PWSTR SearchPath, DWORD SearchPathLength);
extern  BOOL            symGetSymbolFile(PCSTR SymPath, PCSTR ImageFile, DWORD Type, PSTR SymbolFile, size_t cSymbolFile, PSTR DbgFile, size_t cDbgFile);
extern  BOOL            symGetSymbolFileW(PCWSTR SymPath, PCWSTR ImageFile, DWORD Type, PWSTR SymbolFile, size_t cSymbolFile, PWSTR DbgFile, size_t cDbgFile);
extern  BOOL            symGetTypeFromName(ULONG64 BaseOfDll, PCSTR Name, PSYMBOL_INFO Symbol);
extern  BOOL            symGetTypeFromNameW(ULONG64 BaseOfDll, PCWSTR Name, PSYMBOL_INFOW Symbol);
extern  BOOL            symGetTypeInfo(DWORD64 ModBase, ULONG TypeId, IMAGEHLP_SYMBOL_TYPE_INFO GetType, PVOID pInfo);
extern  BOOL            symGetTypeInfoEx(DWORD64 ModBase, PIMAGEHLP_GET_TYPE_INFO_PARAMS Params);
extern  BOOL            symGetUnwindInfo(DWORD64 Address, PVOID Buffer, PULONG Size);
extern  BOOL            symInitialize(PCSTR UserSearchPath, BOOL fIlwadeProcess);
extern  BOOL            symInitializeW(PCWSTR UserSearchPath, BOOL fIlwadeProcess);
#ifdef _WIN64
#define SymLoadModule SymLoadModule64
#else
extern  DWORD           symLoadModule(HANDLE hFile, PCSTR ImageName, PCSTR ModuleName, DWORD BaseOfDll, DWORD SizeOfDll);
#endif
extern  DWORD64         symLoadModule64(HANDLE hFile, PCSTR ImageName, PCSTR ModuleName, DWORD64 BaseOfDll, DWORD SizeOfDll);
extern  DWORD64         symLoadModuleEx(HANDLE hFile, PCSTR ImageName, PCSTR ModuleName, DWORD64 BaseOfDll, DWORD DllSize, PMODLOAD_DATA Data, DWORD Flags);
extern  DWORD64         symLoadModuleExW(HANDLE hFile, PCWSTR ImageName, PCWSTR ModuleName, DWORD64 BaseOfDll, DWORD DllSize, PMODLOAD_DATA Data, DWORD Flags);
extern  BOOL            symMatchFileName(PCSTR FileName, PCSTR Match, PSTR *FileNameStop, PSTR *MatchStop);
extern  BOOL            symMatchFileNameW(PCWSTR FileName, PCWSTR Match, PWSTR *FileNameStop, PWSTR *MatchStop);
extern  BOOL            symMatchString(PCSTR string, PCSTR expression, BOOL fCase);
extern  BOOL            symMatchStringA(PCSTR string, PCSTR expression, BOOL fCase);
extern  BOOL            symMatchStringW(PCWSTR string, PCWSTR expression, BOOL fCase);
extern  BOOL            symNext(PSYMBOL_INFO si);
extern  BOOL            symNextW(PSYMBOL_INFOW siw);
extern  BOOL            symPrev(PSYMBOL_INFO si);
extern  BOOL            symPrevW(PSYMBOL_INFOW siw);
extern  BOOL            symRefreshModuleList();
#ifdef _WIN64
#define SymRegisterCallback SymRegisterCallback64
#define SymRegisterFunctionEntryCallback SymRegisterFunctionEntryCallback64
#else
extern  BOOL            symRegisterCallback(PSYMBOL_REGISTERED_CALLBACK CallbackFunction, PVOID UserContext);
extern  BOOL            symRegisterFunctionEntryCallback(PSYMBOL_FUNCENTRY_CALLBACK CallbackFunction, PVOID UserContext);
#endif
extern  BOOL            symRegisterCallback64(PSYMBOL_REGISTERED_CALLBACK64 CallbackFunction, ULONG64 UserContext);
extern  BOOL            symRegisterCallbackW64(PSYMBOL_REGISTERED_CALLBACK64 CallbackFunction, ULONG64 UserContext);
extern  BOOL            symRegisterFunctionEntryCallback64(PSYMBOL_FUNCENTRY_CALLBACK64 CallbackFunction, ULONG64 UserContext);
extern  BOOL            symSearch(ULONG64 BaseOfDll, DWORD Index, DWORD SymTag, PCSTR Mask, DWORD64 Address, PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback, PVOID UserContext, DWORD Options);
extern  BOOL            symSearchW(ULONG64 BaseOfDll, DWORD Index, DWORD SymTag, PCWSTR Mask, DWORD64 Address, PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback, PVOID UserContext, DWORD Options);
extern  BOOL            symSetContext(PIMAGEHLP_STACK_FRAME StackFrame, PIMAGEHLP_CONTEXT Context);
extern  PCHAR           symSetHomeDirectory(PCSTR dir);
extern  PWSTR           symSetHomeDirectoryW(PCWSTR dir);
extern  DWORD           symSetOptions(DWORD SymOptions);
extern  BOOL            symSetScopeFromAddr(ULONG64 Address);
extern  BOOL            symSetScopeFromIndex(ULONG64 BaseOfDll, DWORD Index);
extern  BOOL            symSetSearchPath(PCSTR SearchPath);
extern  BOOL            symSetSearchPathW(PCWSTR SearchPath);
#ifdef _WIN64
#define SymUnDName SymUnDName64
#define symUnloadModule symUnloadModule64
#else
extern  BOOL            symUnDName(PIMAGEHLP_SYMBOL sym, PSTR UnDecName, DWORD UnDecNameLength);
extern  BOOL            symUnloadModule(DWORD BaseOfDll);
#endif
extern  BOOL            symUnDName64(PIMAGEHLP_SYMBOL64 sym, PSTR UnDecName, DWORD UnDecNameLength);
extern  BOOL            symUnloadModule64(DWORD64 BaseOfDll);

// Symbol server DbgHelp routines
extern  PCSTR           symSrvDeltaName(PCSTR SymPath, PCSTR Type, PCSTR File1, PCSTR File2);
extern  PCWSTR          symSrvDeltaNameW(PCWSTR SymPath, PCWSTR Type, PCWSTR File1, PCWSTR File2);
extern  BOOL            symSrvGetFileIndexes(PCSTR File, GUID *Id, PDWORD Val1, PDWORD Val2, DWORD Flags);
extern  BOOL            symSrvGetFileIndexesW(PCWSTR File, GUID *Id, PDWORD Val1, PDWORD Val2, DWORD Flags);
extern  BOOL            symSrvGetFileIndexInfo(PCSTR File, PSYMSRV_INDEX_INFO Info, DWORD Flags);
extern  BOOL            symSrvGetFileIndexInfoW(PCWSTR File, PSYMSRV_INDEX_INFOW Info, DWORD Flags);
extern  BOOL            symSrvGetFileIndexString(PCSTR SrvPath, PCSTR File, PSTR Index, size_t Size, DWORD Flags);
extern  BOOL            symSrvGetFileIndexStringW(PCWSTR SrvPath, PCWSTR File, PWSTR Index, size_t Size, DWORD Flags);
extern  PCSTR           symSrvGetSupplement(PCSTR SymPath, PCSTR Node, PCSTR File);
extern  PCWSTR          symSrvGetSupplementW(PCWSTR SymPath, PCWSTR Node, PCWSTR File);
extern  BOOL            symSrvIsStore(PCSTR path);
extern  BOOL            symSrvIsStoreW(PCWSTR path);
extern  PCSTR           symSrvStoreFile(PCSTR SrvPath, PCSTR File, DWORD Flags);
extern  PCWSTR          symSrvStoreFileW(PCWSTR SrvPath, PCWSTR File, DWORD Flags);
extern  PCSTR           symSrvStoreSupplement(PCSTR SrvPath, PCSTR Node, PCSTR File, DWORD Flags);
extern  PCWSTR          symSrvStoreSupplementW(PCWSTR SymPath, PCWSTR Node, PCWSTR File, DWORD Flags);

// Source server DbgHelp routines
extern  BOOL            symGetSourceFile(ULONG64 Base, PCSTR Params, PCSTR FileSpec, PSTR FilePath, DWORD Size);
extern  BOOL            symGetSourceFileW(ULONG64 Base, PCWSTR Params, PCWSTR FileSpec, PWSTR FilePath, DWORD Size);
extern  BOOL            symEnumSourceFileTokens(ULONG64 Base, PENUMSOURCEFILETOKENSCALLBACK Callback);
extern  BOOL            symGetSourceFileFromToken(PVOID Token, PCSTR Params, PSTR FilePath, DWORD Size);
extern  BOOL            symGetSourceFileFromTokenW(PVOID Token, PCWSTR Params, PWSTR FilePath, DWORD Size);
extern  BOOL            symGetSourceFileToken(ULONG64 Base, PCSTR FileSpec, PVOID *Token, DWORD *Size);
extern  BOOL            symGetSourceFileTokenW(ULONG64 Base, PCWSTR FileSpec, PVOID *Token, DWORD *Size);
extern  BOOL            symGetSourceVarFromToken(PVOID Token, PCSTR Params, PCSTR VarName, PSTR Value, DWORD Size);
extern  BOOL            symGetSourceVarFromTokenW(PVOID Token, PCWSTR Params, PCWSTR VarName, PWSTR Value, DWORD Size);

// Obsolete DbgHelp routines
#ifdef _WIN64
#define symGetSymFromAddr symGetSymFromAddr64
#define symGetSymFromName symGetSymFromName64
#define symGetSymNext symGetSymNext64
#define symGetSymPrev symGetSymPrev64
#else
extern  BOOL            symGetSymFromAddr(DWORD dwAddr, PDWORD pdwDisplacement, PIMAGEHLP_SYMBOL Symbol);
extern  BOOL            symGetSymFromName(PCSTR Name, PIMAGEHLP_SYMBOL Symbol);
extern  BOOL            symGetSymNext(PIMAGEHLP_SYMBOL Symbol);
extern  BOOL            symGetSymPrev(PIMAGEHLP_SYMBOL Symbol);
#endif
extern  BOOL            symGetSymFromAddr64(DWORD64 qwAddr, PDWORD64 pdwDisplacement, PIMAGEHLP_SYMBOL64  Symbol);
extern  BOOL            symGetSymFromName64(PCSTR Name, PIMAGEHLP_SYMBOL64 Symbol);
extern  BOOL            symGetSymNext64(PIMAGEHLP_SYMBOL64 Symbol);
extern  BOOL            symGetSymPrev64(PIMAGEHLP_SYMBOL64 Symbol);

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _SYMDBGHELP_H
