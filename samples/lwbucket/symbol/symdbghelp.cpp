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
|*  Module: symdbghelp.cpp                                                    *|
|*                                                                            *|
 \****************************************************************************/
#include "symprecomp.h"

//******************************************************************************
//
//  sym namespace
//
//******************************************************************************
namespace sym
{

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  void            acquireDbgHelpInterface();
static  void            releaseDbgHelpInterface();

//******************************************************************************
//
// Locals
//
//******************************************************************************
static  HANDLE          s_hSymbol               = 0;    // DbgHelp symbol handle

static const char*      s_SymbolType[]          = {
/* 0x00000000 */                                   "SymNone",
/* 0x00000001 */                                   "SymCoff",
/* 0x00000002 */                                   "SymCv",
/* 0x00000003 */                                   "SymPdb",
/* 0x00000004 */                                   "SymExport",
/* 0x00000005 */                                   "SymDeferred",
/* 0x00000006 */                                   "SymSym",
/* 0x00000007 */                                   "SymDia",
/* 0x00000008 */                                   "SymVirtual",
                                                  };
static const char*      s_SymbolTypeUnknown     = "SymUnknown";

static const char*      s_SymbolTag[]           = {
/* 0x00000000 */                                   "SymTagNull",
/* 0x00000001 */                                   "SymTagExe",
/* 0x00000002 */                                   "SymTagCompiland",
/* 0x00000003 */                                   "SymTagCompilandDetails",
/* 0x00000004 */                                   "SymTagCompilandElw",
/* 0x00000005 */                                   "SymTagFunction",
/* 0x00000006 */                                   "SymTagBlock",
/* 0x00000007 */                                   "SymTagData",
/* 0x00000008 */                                   "SymTagAnnotation",
/* 0x00000009 */                                   "SymTagLabel",
/* 0x0000000a */                                   "SymTagPublicSymbol",
/* 0x0000000b */                                   "SymTagUDT",
/* 0x0000000c */                                   "SymTagEnum",
/* 0x0000000d */                                   "SymTagFunctionType",
/* 0x0000000e */                                   "SymTagPointerType",
/* 0x0000000f */                                   "SymTagArrayType",
/* 0x00000010 */                                   "SymTagBaseType",
/* 0x00000011 */                                   "SymTagTypedef",
/* 0x00000012 */                                   "SymTagBaseClass",
/* 0x00000013 */                                   "SymTagFriend",
/* 0x00000014 */                                   "SymTagFunctionArgType",
/* 0x00000015 */                                   "SymTagFuncDebugStart",
/* 0x00000016 */                                   "SymTagFuncDebugEnd",
/* 0x00000017 */                                   "SymTagUsingNamespace",
/* 0x00000018 */                                   "SymTagVTableShape",
/* 0x00000019 */                                   "SymTagVTable",
/* 0x0000001a */                                   "SymTagLwstom",
/* 0x0000001b */                                   "SymTagThunk",
/* 0x0000001c */                                   "SymTagLwstomType",
/* 0x0000001d */                                   "SymTagManagedType",
/* 0x0000001e */                                   "SymTagDimension",
/* 0x0000001f */                                   "SymTagCallSite",
                                                  };
static const char*      s_SymbolTagUnknown      = "SymTagUnknown";

static const char*      s_CallingColwention[]   = {
/* 0x00000000 */                                   "NEAR C",
/* 0x00000001 */                                   "FAR C",
/* 0x00000002 */                                   "NEAR PASCAL",
/* 0x00000003 */                                   "FAR PASCAL",
/* 0x00000004 */                                   "NEAR FAST",
/* 0x00000005 */                                   "FAR FAST",
/* 0x00000006 */                                   "SKIPPED",
/* 0x00000007 */                                   "NEAR STD",
/* 0x00000008 */                                   "FAR STD",
/* 0x00000009 */                                   "NEAR SYS",
/* 0x0000000a */                                   "FAR SYS",
/* 0x0000000b */                                   "THIS CALL",
/* 0x0000000c */                                   "MIPS CALL",
/* 0x0000000d */                                   "GENERIC",
/* 0x0000000e */                                   "ALPHA CALL",
/* 0x0000000f */                                   "PPC CALL",
/* 0x00000010 */                                   "SH CALL",
/* 0x00000011 */                                   "ARM CALL",
/* 0x00000012 */                                   "AM33 CALL",
/* 0x00000013 */                                   "TRICORE CALL",
/* 0x00000014 */                                   "SH5 CALL",
/* 0x00000015 */                                   "M32R CALL",
/* 0x00000016 */                                   "CLR CALL",
                                                  };
static const char*      s_CallingColwentionUnknown = "UNKNOWN";

static const char*      s_UdtKind[]             = {
/* 0x00000000 */                                   "struct",
/* 0x00000001 */                                   "class",
/* 0x00000002 */                                   "union",
/* 0x00000003 */                                   "interface",
                                                  };
static const char*      s_UdtKindUnknown        = "unknown";

static  CRITICAL_SECTION    dbgHelpInterfaceLock;

//******************************************************************************

HRESULT
initializeDbgHelp()
{
    HRESULT             hResult = S_OK;

    // Initialize the DbgHelp interface critical section
    InitializeCriticalSection(&dbgHelpInterfaceLock);

    return hResult;

} // initializeDbgHelp

//******************************************************************************

HRESULT
uninitializeDbgHelp()
{
    HRESULT             hResult = S_OK;

    // Delete the DbgHelp interface critical section
    DeleteCriticalSection(&dbgHelpInterfaceLock);

    return hResult;

} // uninitializeDbgHelp

//******************************************************************************

void
acquireDbgHelpInterface()
{
    // Acquire the DbgHelp interface critical section
    EnterCriticalSection(&dbgHelpInterfaceLock);

} // acquireDbgHelpInterface

//******************************************************************************

void
releaseDbgHelpInterface()
{
    // Release the DbgHelp interface critical section
    LeaveCriticalSection(&dbgHelpInterfaceLock);

} // releaseDbgHelpInterface

//******************************************************************************

BOOL
enumDirTree
(
    PCSTR               RootPath,
    PCSTR               InputPathName,
    PSTR                OutputPathBuffer,
    PENUMDIRTREE_CALLBACK cb,
    PVOID               data
)
{
    BOOL                bResult = FALSE;

    assert(RootPath != NULL);
    assert(InputPathName != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call enumerate directory tree routine
        bResult = EnumDirTree(s_hSymbol, RootPath, InputPathName, OutputPathBuffer, cb, data);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // enumDirTree

//******************************************************************************

BOOL
enumDirTreeW
(
    PCWSTR              RootPath,
    PCWSTR              InputPathName,
    PWSTR               OutputPathBuffer,
    PENUMDIRTREE_CALLBACKW cb,
    PVOID               data
)
{
    BOOL                bResult = FALSE;

    assert(RootPath != NULL);
    assert(InputPathName != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call enumerate directory tree routine
        bResult = EnumDirTreeW(s_hSymbol, RootPath, InputPathName, OutputPathBuffer, cb, data);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // enumDirTreeW

//******************************************************************************

BOOL
makeSureDirectoryPathExists
(
    PCSTR               DirPath
)
{
    BOOL                bResult = FALSE;

    assert(DirPath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call make sure directory path exists routine
        bResult = MakeSureDirectoryPathExists(DirPath);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // makeSureDirectoryPathExists

//******************************************************************************

BOOL
searchTreeForFile
(
    PCSTR               RootPath,
    PCSTR               InputPathName,
    PSTR                OutputPathBuffer
)
{
    BOOL                bResult = FALSE;

    assert(RootPath != NULL);
    assert(InputPathName != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call search tree for file routine
        bResult = SearchTreeForFile(RootPath, InputPathName, OutputPathBuffer);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // searchTreeForFile

//******************************************************************************

BOOL
searchTreeForFileW
(
    PCWSTR              RootPath,
    PCWSTR              InputPathName,
    PWSTR               OutputPathBuffer
)
{
    BOOL                bResult = FALSE;

    assert(RootPath != NULL);
    assert(InputPathName != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call search tree for file routine
        bResult = SearchTreeForFileW(RootPath, InputPathName, OutputPathBuffer);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // searchTreeForFileW

//******************************************************************************

BOOL
enumerateLoadedModules
(
    PENUMLOADED_MODULES_CALLBACK EnumLoadedModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumLoadedModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call enumerate loaded modules routine
        bResult = EnumerateLoadedModules(s_hSymbol, EnumLoadedModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // enumerateLoadedModules

//******************************************************************************

BOOL
enumerateLoadedModules64
(
    PENUMLOADED_MODULES_CALLBACK64 EnumLoadedModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumLoadedModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call enumerate loaded modules routine
        bResult = EnumerateLoadedModules64(s_hSymbol, EnumLoadedModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // enumerateLoadedModules64

//******************************************************************************

BOOL
enumerateLoadedModulesW64
(
    PENUMLOADED_MODULES_CALLBACKW64 EnumLoadedModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumLoadedModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call enumerate loaded modules routine
        bResult = EnumerateLoadedModulesW64(s_hSymbol, EnumLoadedModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // enumerateLoadedModulesW64

//******************************************************************************

BOOL
enumerateLoadedModulesEx
(
    PENUMLOADED_MODULES_CALLBACK64 EnumLoadedModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumLoadedModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call enumerate loaded modules routine
        bResult = EnumerateLoadedModulesEx(s_hSymbol, EnumLoadedModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // enumerateLoadedModulesEx

//******************************************************************************

BOOL
enumerateLoadedModulesExW
(
    PENUMLOADED_MODULES_CALLBACKW64 EnumLoadedModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumLoadedModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call enumerate loaded modules routine
        bResult = EnumerateLoadedModulesExW(s_hSymbol, EnumLoadedModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // enumerateLoadedModulesExW

//******************************************************************************

HANDLE
findDebugInfoFile
(
    PCSTR               FileName,
    PCSTR               SymbolPath,
    PSTR                DebugFilePath
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(SymbolPath != NULL);
    assert(DebugFilePath != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call find debug information file routine
        hResult = FindDebugInfoFile(FileName, SymbolPath, DebugFilePath);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // findDebugInfoFile

//******************************************************************************

HANDLE
findDebugInfoFileEx
(
    PCSTR               FileName,
    PCSTR               SymbolPath,
    PSTR                DebugFilePath,
    PFIND_DEBUG_FILE_CALLBACK Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(SymbolPath != NULL);
    assert(DebugFilePath != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call find debug information file routine
        hResult = FindDebugInfoFileEx(FileName, SymbolPath, DebugFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // findDebugInfoFileEx

//******************************************************************************

HANDLE
findDebugInfoFileExW
(
    PCWSTR              FileName,
    PCWSTR              SymbolPath,
    PWSTR               DebugFilePath,
    PFIND_DEBUG_FILE_CALLBACKW Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(SymbolPath != NULL);
    assert(DebugFilePath != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call find debug information file routine
        hResult = FindDebugInfoFileExW(FileName, SymbolPath, DebugFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // findDebugInfoFileExW

//******************************************************************************

HANDLE
findExelwtableImage
(
    PCSTR               FileName,
    PCSTR               SymbolPath,
    PSTR                ImageFilePath
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(SymbolPath != NULL);
    assert(ImageFilePath != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call find exelwtable image routine
        hResult = FindExelwtableImage(FileName, SymbolPath, ImageFilePath);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // findExelwtableImage

//******************************************************************************

HANDLE
findExelwtableImageEx
(
    PCSTR               FileName,
    PCSTR               SymbolPath,
    PSTR                ImageFilePath,
    PFIND_EXE_FILE_CALLBACK Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(SymbolPath != NULL);
    assert(ImageFilePath != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call find exelwtable image routine
        hResult = FindExelwtableImageEx(FileName, SymbolPath, ImageFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // findExelwtableImageEx

//******************************************************************************

HANDLE
findExelwtableImageExW
(
    PCWSTR              FileName,
    PCWSTR              SymbolPath,
    PWSTR               ImageFilePath,
    PFIND_EXE_FILE_CALLBACKW Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(SymbolPath != NULL);
    assert(ImageFilePath != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call find exelwtable image routine
        hResult = FindExelwtableImageExW(FileName, SymbolPath, ImageFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // findExelwtableImageExW

//******************************************************************************
#ifndef _WIN64
BOOL
stackWalk
(
    DWORD               MachineType,
    HANDLE              hProcess,
    HANDLE              hThread,
    LPSTACKFRAME        StackFrame,
    PVOID               ContextRecord,
    PREAD_PROCESS_MEMORY_ROUTINE ReadMemoryRoutine,
    PFUNCTION_TABLE_ACCESS_ROUTINE FunctionTableAccessRoutine,
    PGET_MODULE_BASE_ROUTINE GetModuleBaseRoutine,
    PTRANSLATE_ADDRESS_ROUTINE TranslateAddress
)
{
    BOOL                bResult = FALSE;

    assert(StackFrame != NULL);
    assert(ContextRecord != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call stack walk routine
        bResult = StackWalk(MachineType, hProcess, hThread, StackFrame, ContextRecord, ReadMemoryRoutine, FunctionTableAccessRoutine, GetModuleBaseRoutine, TranslateAddress);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // stackWalk
#endif
//******************************************************************************

BOOL
stackWalk64
(
    DWORD               MachineType,
    HANDLE              hProcess,
    HANDLE              hThread,
    LPSTACKFRAME64      StackFrame,
    PVOID               ContextRecord,
    PREAD_PROCESS_MEMORY_ROUTINE64 ReadMemoryRoutine,
    PFUNCTION_TABLE_ACCESS_ROUTINE64 FunctionTableAccessRoutine,
    PGET_MODULE_BASE_ROUTINE64 GetModuleBaseRoutine,
    PTRANSLATE_ADDRESS_ROUTINE64 TranslateAddress
)
{
    BOOL                bResult = FALSE;

    assert(StackFrame != NULL);
    assert(ContextRecord != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call stack walk routine
        bResult = StackWalk64(MachineType, hProcess, hThread, StackFrame, ContextRecord, ReadMemoryRoutine, FunctionTableAccessRoutine, GetModuleBaseRoutine, TranslateAddress);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // stackWalk64

//******************************************************************************

BOOL
symSetParentWindow
(
    HWND                hwnd
)
{
    BOOL                bResult = FALSE;

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set parent window routine
        bResult = SymSetParentWindow(hwnd);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSetParentWindow

//******************************************************************************

DWORD
unDecorateSymbolName
(
    PCSTR               name,
    PSTR                outputString,
    DWORD               maxStringLength,
    DWORD               flags
)
{
    DWORD               dwResult = 0;

    assert(name != NULL);
    assert(outputString);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call undecorate symbol name routine
        dwResult = UnDecorateSymbolName(name, outputString, maxStringLength, flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return dwResult;

} // unDecorateSymbolName

//******************************************************************************

DWORD
unDecorateSymbolNameW
(
    PCWSTR              name,
    PWSTR               outputString,
    DWORD               maxStringLength,
    DWORD               flags
)
{
    DWORD               dwResult = 0;

    assert(name != NULL);
    assert(outputString);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call undecorate symbol name routine
        dwResult = UnDecorateSymbolNameW(name, outputString, maxStringLength, flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return dwResult;

} // unDecorateSymbolNameW

//******************************************************************************

PVOID
imageDirectoryEntryToData
(
    PVOID               Base,
    BOOLEAN             MappedAsImage,
    USHORT              DirectoryEntry,
    PULONG              Size
)
{
    PVOID               pResult = NULL;

    assert(Base != NULL);
    assert(Size != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call image directory entry to data routine
        pResult = ImageDirectoryEntryToData(Base, MappedAsImage, DirectoryEntry, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // imageDirectoryEntryToData

//******************************************************************************

PVOID
imageDirectoryEntryToDataEx
(
    PVOID               Base,
    BOOLEAN             MappedAsImage,
    USHORT              DirectoryEntry,
    PULONG              Size,
    PIMAGE_SECTION_HEADER *FoundHeader
)
{
    PVOID               pResult = NULL;

    assert(Base != NULL);
    assert(Size != NULL);

    // Check for extension initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call image directory entry to data routine
        pResult = ImageDirectoryEntryToDataEx(Base, MappedAsImage, DirectoryEntry, Size, FoundHeader);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // imageDirectoryEntryToDataEx

//******************************************************************************

PIMAGE_NT_HEADERS
imageNtHeader
(
    PVOID               Base
)
{
    PIMAGE_NT_HEADERS   pResult = NULL;

    assert(Base != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call image NT header routine
        pResult = ImageNtHeader(Base);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // imageNtHeader

//******************************************************************************

PIMAGE_SECTION_HEADER
imageRvaToSection
(
    PIMAGE_NT_HEADERS   NtHeaders,
    PVOID               Base,
    ULONG               Rva
)
{
    PIMAGE_SECTION_HEADER pResult = NULL;

    assert(NtHeaders != NULL);
    assert(Base != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call image RVA to section routine
        pResult = ImageRvaToSection(NtHeaders, Base, Rva);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // imageRvaToSection

//******************************************************************************

PVOID
imageRvaToVa
(
    PIMAGE_NT_HEADERS   NtHeaders,
    PVOID               Base,
    ULONG               Rva,
    PIMAGE_SECTION_HEADER *LastRvaSection
)
{
    PVOID               pResult = NULL;

    assert(NtHeaders != NULL);
    assert(Base != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call image RBA to VA routine
        pResult = ImageRvaToVa(NtHeaders, Base, Rva, LastRvaSection);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // imageRvaToVa

//******************************************************************************

BOOL
symAddSymbol
(
    ULONG64             BaseOfDll,
    PCSTR               Name,
    DWORD64             Address,
    DWORD               Size,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(Name != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol add symbol routine
        bResult = SymAddSymbol(s_hSymbol, BaseOfDll, Name, Address, Size, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symAddSymbol

//******************************************************************************

BOOL
symAddSymbolW
(
    ULONG64             BaseOfDll,
    PCWSTR              Name,
    DWORD64             Address,
    DWORD               Size,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(Name != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol add symbol routine
        bResult = SymAddSymbolW(s_hSymbol, BaseOfDll, Name, Address, Size, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symAddSymbolW

//******************************************************************************

BOOL
symCleanup()
{
    BOOL                bResult = FALSE;

    // Symbol system should be initialized
    assert(s_hSymbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol cleanup routine
        bResult = SymCleanup(s_hSymbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();
    }
    // Clear the symbol handle
    s_hSymbol = NULL;

    // Check the progress indicator
    progressCheck();

    // Perform status check after every DbgHlp call
    statusCheck();

    // Return the result
    return bResult;

} // symCleanup

//******************************************************************************

BOOL
symDeleteSymbol
(
    ULONG64             BaseOfDll,
    PCSTR               Name,
    DWORD64             Address,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol delete symbol routine
        bResult = SymDeleteSymbol(s_hSymbol, BaseOfDll, Name, Address, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symDeleteSymbol

//******************************************************************************

BOOL
symDeleteSymbolW
(
    ULONG64             BaseOfDll,
    PCWSTR              Name,
    DWORD64             Address,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol delete symbol routine
        bResult = SymDeleteSymbolW(s_hSymbol, BaseOfDll, Name, Address, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symDeleteSymbolW

//******************************************************************************
#ifndef _WIN64
BOOL
symEnumerateModules
(
    PSYM_ENUMMODULES_CALLBACK EnumModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate modules routine
        bResult = SymEnumerateModules(s_hSymbol, EnumModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumerateModules
#endif
//******************************************************************************

BOOL
symEnumerateModules64
(
    PSYM_ENUMMODULES_CALLBACK64 EnumModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate modules routine
        bResult = SymEnumerateModules64(s_hSymbol, EnumModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumerateModules64

//******************************************************************************

BOOL
symEnumerateModulesW64
(
    PSYM_ENUMMODULES_CALLBACKW64 EnumModulesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumModulesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate modules routine
        bResult = SymEnumerateModulesW64(s_hSymbol, EnumModulesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumerateModulesW64

//******************************************************************************

BOOL
symEnumLines
(
    ULONG64             Base,
    PCSTR               Obj,
    PCSTR               File,
    PSYM_ENUMLINES_CALLBACK EnumLinesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumLinesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate lines routine
        bResult = SymEnumLines(s_hSymbol, Base, Obj, File, EnumLinesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumLines

//******************************************************************************

BOOL
symEnumLinesW
(
    ULONG64             Base,
    PCWSTR              Obj,
    PCWSTR              File,
    PSYM_ENUMLINES_CALLBACKW EnumLinesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumLinesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate lines routine
        bResult = SymEnumLinesW(s_hSymbol, Base, Obj, File, EnumLinesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumLinesW

//******************************************************************************

BOOL
symEnumProcesses
(
    PSYM_ENUMPROCESSES_CALLBACK EnumProcessesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumProcessesCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate processes routine
        bResult = SymEnumProcesses(EnumProcessesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumProcesses

//******************************************************************************

BOOL
symEnumSourceFiles
(
    ULONG64             ModBase,
    PCSTR               Mask,
    PSYM_ENUMSOURCEFILES_CALLBACK cbSrcFiles,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(cbSrcFiles != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate source files routine
        bResult = SymEnumSourceFiles(s_hSymbol, ModBase, Mask, cbSrcFiles, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSourceFiles

//******************************************************************************

BOOL
symEnumSourceFilesW
(
    ULONG64             ModBase,
    PCWSTR              Mask,
    PSYM_ENUMSOURCEFILES_CALLBACKW cbSrcFiles,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(cbSrcFiles != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate source files routine
        bResult = SymEnumSourceFilesW(s_hSymbol, ModBase, Mask, cbSrcFiles, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSourceFilesW

//******************************************************************************

BOOL
symEnumSourceLines
(
    ULONG64             Base,
    PCSTR               Obj,
    PCSTR               File,
    DWORD               Line,
    DWORD               Flags,
    PSYM_ENUMLINES_CALLBACK EnumLinesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate source lines routine
        bResult = SymEnumSourceLines(s_hSymbol, Base, Obj, File, Line, Flags, EnumLinesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSourceLines

//******************************************************************************

BOOL
symEnumSourceLinesW
(
    ULONG64             Base,
    PCWSTR              Obj,
    PCWSTR              File,
    DWORD               Line,
    DWORD               Flags,
    PSYM_ENUMLINES_CALLBACKW EnumLinesCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate source lines routine
        bResult = SymEnumSourceLinesW(s_hSymbol, Base, Obj, File, Line, Flags, EnumLinesCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSourceLinesW

//******************************************************************************

BOOL
symEnumSymbols
(
    ULONG64             BaseOfDll,
    PCSTR               Mask,
    PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate symbols routine
        bResult = SymEnumSymbols(s_hSymbol, BaseOfDll, Mask, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSymbols

//******************************************************************************

BOOL
symEnumSymbolsW
(
    ULONG64             BaseOfDll,
    PCWSTR              Mask,
    PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate symbols routine
        bResult = SymEnumSymbolsW(s_hSymbol, BaseOfDll, Mask, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSymbolsW

//******************************************************************************

BOOL
symEnumSymbolsForAddr
(
    DWORD64             Address,
    PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate symbols for address routine
        bResult = SymEnumSymbolsForAddr(s_hSymbol, Address, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSymbolsForAddr

//******************************************************************************

BOOL
symEnumSymbolsForAddrW
(
    DWORD64             Address,
    PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate symbols for address routine
        bResult = SymEnumSymbolsForAddrW(s_hSymbol, Address, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSymbolsForAddrW

//******************************************************************************

BOOL
symEnumTypes
(
    ULONG64             BaseOfDll,
    PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate types routine
        bResult = SymEnumTypes(s_hSymbol, BaseOfDll, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumTypes

//******************************************************************************

BOOL
symEnumTypesW
(
    ULONG64             BaseOfDll,
    PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate types routine
        bResult = SymEnumTypesW(s_hSymbol, BaseOfDll, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumTypesW

//******************************************************************************

BOOL
symEnumTypesByName
(
    ULONG64             BaseOfDll,
    PCSTR               mask,
    PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate types by name routine
        bResult = SymEnumTypesByName(s_hSymbol, BaseOfDll, mask, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumTypesByName

//******************************************************************************

BOOL
symEnumTypesByNameW
(
    ULONG64             BaseOfDll,
    PCWSTR              mask,
    PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate types by name routine
        bResult = SymEnumTypesByNameW(s_hSymbol, BaseOfDll, mask, EnumSymbolsCallback, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumTypesByNameW

//******************************************************************************

HANDLE
symFindDebugInfoFile
(
    PCSTR               FileName,
    PSTR                DebugFilePath,
    PFIND_DEBUG_FILE_CALLBACK Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(DebugFilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol find debug information file routine
        hResult = SymFindDebugInfoFile(s_hSymbol, FileName, DebugFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // symFindDebugInfoFile

//******************************************************************************

HANDLE
symFindDebugInfoFileW
(
    PCWSTR              FileName,
    PWSTR               DebugFilePath,
    PFIND_DEBUG_FILE_CALLBACKW Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(DebugFilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol find debug information file routine
        hResult = SymFindDebugInfoFileW(s_hSymbol, FileName, DebugFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // symFindDebugInfoFileW

//******************************************************************************

HANDLE
symFindExelwtableImage
(
    PCSTR               FileName,
    PSTR                ImageFilePath,
    PFIND_EXE_FILE_CALLBACK Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(ImageFilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol find exelwtable image routine
        hResult = SymFindExelwtableImage(s_hSymbol, FileName, ImageFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // symFindExelwtableImage

//******************************************************************************

HANDLE
symFindExelwtableImageW
(
    PCWSTR              FileName,
    PWSTR               ImageFilePath,
    PFIND_EXE_FILE_CALLBACKW Callback,
    PVOID               CallerData
)
{
    HANDLE              hResult = NULL;

    assert(FileName != NULL);
    assert(ImageFilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol find exelwtable image routine
        hResult = SymFindExelwtableImageW(s_hSymbol, FileName, ImageFilePath, Callback, CallerData);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return hResult;

} // symFindExelwtableImageW

//******************************************************************************

BOOL
symFindFileInPath
(
    PCSTR               SearchPath,
    PCSTR               FileName,
    PVOID               id,
    DWORD               two,
    DWORD               three,
    DWORD               flags,
    PSTR                FoundFile,
    PFINDFILEINPATHCALLBACK callback,
    PVOID               context
)
{
    BOOL                bResult = FALSE;

    assert(FileName != NULL);
    assert(FoundFile != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol find file in path routine
        bResult = SymFindFileInPath(s_hSymbol, SearchPath, FileName, id, two, three, flags, FoundFile, callback, context);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFindFileInPath

//******************************************************************************

BOOL
symFindFileInPathW
(
    PCWSTR              SearchPath,
    PCWSTR              FileName,
    PVOID               id,
    DWORD               two,
    DWORD               three,
    DWORD               flags,
    PWSTR               FoundFile,
    PFINDFILEINPATHCALLBACKW callback,
    PVOID               context
)
{
    BOOL                bResult = FALSE;

    assert(FileName != NULL);
    assert(FoundFile != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol find file in path routine
        bResult = SymFindFileInPathW(s_hSymbol, SearchPath, FileName, id, two, three, flags, FoundFile, callback, context);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFindFileInPathW

//******************************************************************************

BOOL
symFromAddr
(
    DWORD64             Address,
    PDWORD64            Displacement,
    PSYMBOL_INFO        Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from address routine
        bResult = SymFromAddr(s_hSymbol, Address, Displacement, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromAddr

//******************************************************************************

BOOL
symFromAddrW
(
    DWORD64             Address,
    PDWORD64            Displacement,
    PSYMBOL_INFOW       Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from address routine
        bResult = SymFromAddrW(s_hSymbol, Address, Displacement, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromAddrW

//******************************************************************************

BOOL
symFromIndex
(
    ULONG64             BaseOfDll,
    DWORD               Index,
    PSYMBOL_INFO        Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from index routine
        bResult = SymFromIndex(s_hSymbol, BaseOfDll, Index, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromIndex

//******************************************************************************

BOOL
symFromIndexW
(
    ULONG64             BaseOfDll,
    DWORD               Index,
    PSYMBOL_INFOW       Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from index routine
        bResult = SymFromIndexW(s_hSymbol, BaseOfDll, Index, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromIndexW

//******************************************************************************

BOOL
symFromName
(
    PCSTR               Name,
    PSYMBOL_INFO        Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Name != NULL);
    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from name routine
        bResult = SymFromName(s_hSymbol, Name, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromName

//******************************************************************************

BOOL
symFromNameW
(
    PCWSTR              Name,
    PSYMBOL_INFOW       Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Name != NULL);
    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from name routine
        bResult = SymFromNameW(s_hSymbol, Name, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromNameW

//******************************************************************************

BOOL
symFromToken
(
    DWORD64             Base,
    DWORD               Token,
    PSYMBOL_INFO        Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from token routine
        bResult = SymFromToken(s_hSymbol, Base, Token, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromToken

//******************************************************************************

BOOL
symFromTokenW
(
    DWORD64             Base,
    DWORD               Token,
    PSYMBOL_INFOW       Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol from token routine
        bResult = SymFromTokenW(s_hSymbol, Base, Token, Symbol);
        if (bResult)
        {
            // Check for an invalid type index (Index already a type)
            if (Symbol->TypeIndex == 0)
            {
                // Save symbol index as type index
                Symbol->TypeIndex = Symbol->Index;
            }
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symFromTokenW

//******************************************************************************

PVOID
symFunctionTableAccess
(
    DWORD               AddrBase
)
{
    PVOID               pResult = NULL;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol function table access routine
        pResult = SymFunctionTableAccess64(s_hSymbol, AddrBase);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symFunctionTableAccess

//******************************************************************************

PVOID
symFunctionTableAccess64
(
    DWORD64             AddrBase
)
{
    PVOID               pResult = NULL;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol function table access routine
        pResult = SymFunctionTableAccess64(s_hSymbol, AddrBase);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symFunctionTableAccess64

//******************************************************************************

ULONG
symGetFileLineOffsets64
(
    PCSTR               ModuleName,
    PCSTR               FileName,
    PDWORD64            Buffer,
    ULONG               BufferLines
)
{
    ULONG               ulResult = 0;

    assert(FileName != NULL);
    assert(Buffer != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get file line offset routine
        ulResult = SymGetFileLineOffsets64(s_hSymbol, ModuleName, FileName, Buffer, BufferLines);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return ulResult;

} // symGetFileLineOffsets64

//******************************************************************************

PCHAR
symGetHomeDirectory
(
    DWORD               type,
    PSTR                dir,
    size_t              size
)
{
    PCHAR               pResult = NULL;

    assert(dir != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get home directory routine
        pResult = SymGetHomeDirectory(type, dir, size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symGetHomeDirectory

//******************************************************************************

PWSTR
symGetHomeDirectoryW
(
    DWORD               type,
    PWSTR               dir,
    size_t              size
)
{
    PWSTR               pResult = NULL;

    assert(dir != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get home directory routine
        pResult = SymGetHomeDirectoryW(type, dir, size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symGetHomeDirectoryW

//******************************************************************************
#ifndef _WIN64
BOOL
symGetLineFromAddr
(
    DWORD               dwAddr,
    PDWORD              pdwDisplacement,
    PIMAGEHLP_LINE      Line
)
{
    BOOL                bResult = FALSE;

    assert(pdwDisplacement != NULL);
    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line from address routine
        bResult = SymGetLineFromAddr(s_hSymbol, dwAddr, pdwDisplacement, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineFromAddr
#endif
//******************************************************************************

BOOL
symGetLineFromAddr64
(
    DWORD64             qwAddr,
    PDWORD              pdwDisplacement,
    PIMAGEHLP_LINE64    Line64
)
{
    BOOL                bResult = FALSE;

    assert(pdwDisplacement != NULL);
    assert(Line64 != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line from address routine
        bResult = SymGetLineFromAddr64(s_hSymbol, qwAddr, pdwDisplacement, Line64);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineFromAddr64

//******************************************************************************

BOOL
symGetLineFromAddrW64
(
    DWORD64             dwAddr,
    PDWORD              pdwDisplacement,
    PIMAGEHLP_LINEW64   Line
)
{
    BOOL                bResult = FALSE;

    assert(pdwDisplacement != NULL);
    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line from address routine
        bResult = SymGetLineFromAddrW64(s_hSymbol, dwAddr, pdwDisplacement, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineFromAddrW64

//******************************************************************************
#ifndef _WIN64
BOOL
symGetLineFromName
(
    PCSTR               ModuleName,
    PCSTR               FileName,
    DWORD               dwLineNumber,
    PLONG               plDisplacement,
    PIMAGEHLP_LINE      Line
)
{
    BOOL                bResult = FALSE;

    assert(plDisplacement != NULL);
    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line from name routine
        bResult = SymGetLineFromName(s_hSymbol, ModuleName, FileName, dwLineNumber, plDisplacement, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineFromName
#endif
//******************************************************************************

BOOL
symGetLineFromName64
(
    PCSTR               ModuleName,
    PCSTR               FileName,
    DWORD               dwLineNumber,
    PLONG               plDisplacement,
    PIMAGEHLP_LINE64    Line
)
{
    BOOL                bResult = FALSE;

    assert(plDisplacement != NULL);
    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line from name routine
        bResult = SymGetLineFromName64(s_hSymbol, ModuleName, FileName, dwLineNumber, plDisplacement, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineFromName64

//******************************************************************************

BOOL
symGetLineFromNameW64
(
    PCWSTR              ModuleName,
    PCWSTR              FileName,
    DWORD               dwLineNumber,
    PLONG               plDisplacement,
    PIMAGEHLP_LINEW64   Line
)
{
    BOOL                bResult = FALSE;

    assert(plDisplacement != NULL);
    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line from name routine
        bResult = SymGetLineFromNameW64(s_hSymbol, ModuleName, FileName, dwLineNumber, plDisplacement, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineFromNameW64

//******************************************************************************
#ifndef _WIN64
BOOL
symGetLineNext
(
    PIMAGEHLP_LINE      Line
)
{
    BOOL                bResult = FALSE;

    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line next routine
        bResult = SymGetLineNext(s_hSymbol, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineNext
#endif
//******************************************************************************

BOOL
symGetLineNext64
(
    PIMAGEHLP_LINE64    Line
)
{
    BOOL                bResult = FALSE;

    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line next routine
        bResult = SymGetLineNext64(s_hSymbol, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineNext64

//******************************************************************************

BOOL
symGetLineNextW64
(
    PIMAGEHLP_LINEW64   Line
)
{
    BOOL                bResult = FALSE;

    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line next routine
        bResult = SymGetLineNextW64(s_hSymbol, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLineNextW64

//******************************************************************************
#ifndef _WIN64
BOOL
symGetLinePrev
(
    PIMAGEHLP_LINE      Line
)
{
    BOOL                bResult = FALSE;

    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line previous routine
        bResult = SymGetLinePrev(s_hSymbol, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLinePrev
#endif
//******************************************************************************

BOOL
symGetLinePrev64
(
    PIMAGEHLP_LINE64    Line
)
{
    BOOL                bResult = FALSE;

    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line previous routine
        bResult = SymGetLinePrev64(s_hSymbol, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLinePrev64

//******************************************************************************

BOOL
symGetLinePrevW64
(
    PIMAGEHLP_LINEW64   Line
)
{
    BOOL                bResult = FALSE;

    assert(Line != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get line previous routine
        bResult = SymGetLinePrevW64(s_hSymbol, Line);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetLinePrevW64

//******************************************************************************
#ifndef _WIN64
DWORD
symGetModuleBase
(
    DWORD               dwAddr
)
{
    DWORD               dwResult = 0;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get module base routine
        dwResult = SymGetModuleBase(s_hSymbol, dwAddr);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return dwResult;

} // symGetModuleBase
#endif
//******************************************************************************

DWORD64
symGetModuleBase64
(
    DWORD64             qwAddr
)
{
    DWORD64             qwResult = 0;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get module base routine
        qwResult = SymGetModuleBase64(s_hSymbol, qwAddr);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return qwResult;

} // symGetModuleBase64

//******************************************************************************
#ifndef _WIN64
BOOL
symGetModuleInfo
(
    DWORD               dwAddr,
    PIMAGEHLP_MODULE    ModuleInfo
)
{
    BOOL                bResult = FALSE;

    assert(ModuleInfo != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get module information routine
        bResult = SymGetModuleInfo(s_hSymbol, dwAddr, ModuleInfo);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetModuleInfo
#endif
//******************************************************************************
#ifndef _WIN64
BOOL
symGetModuleInfoW
(
    DWORD               dwAddr,
    PIMAGEHLP_MODULEW   ModuleInfo
)
{
    BOOL                bResult = FALSE;

    assert(ModuleInfo != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get module information routine
        bResult = SymGetModuleInfoW(s_hSymbol, dwAddr, ModuleInfo);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetModuleInfoW
#endif
//******************************************************************************

BOOL
symGetModuleInfo64
(
    DWORD64             qwAddr,
    PIMAGEHLP_MODULE64  ModuleInfo
)
{
    BOOL                bResult = FALSE;

    assert(ModuleInfo != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get module information routine
        bResult = SymGetModuleInfo64(s_hSymbol, qwAddr, ModuleInfo);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetModuleInfo64

//******************************************************************************

BOOL
symGetModuleInfoW64
(
    DWORD64             qwAddr,
    PIMAGEHLP_MODULEW64 ModuleInfo
)
{
    BOOL                bResult = FALSE;

    assert(ModuleInfo != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get module information routine
        bResult = SymGetModuleInfoW64(s_hSymbol, qwAddr, ModuleInfo);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetModuleInfoW64

//******************************************************************************

BOOL
symGetOmaps
(
    DWORD64             BaseOfDll,
    POMAP              *OmapTo,
    PDWORD64            cOmapTo,
    POMAP              *OmapFrom,
    PDWORD64            cOmapFrom
)
{
    BOOL                bResult = FALSE;

    assert(OmapTo != NULL);
    assert(cOmapTo != NULL);
    assert(OmapFrom != NULL);
    assert(cOmapFrom != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get omaps routine
        bResult = SymGetOmaps(s_hSymbol, BaseOfDll, OmapTo, cOmapTo, OmapFrom, cOmapFrom);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetOmaps

//******************************************************************************

DWORD
symGetOptions(VOID)
{
    DWORD               dwResult = 0;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get options routine
        dwResult = SymGetOptions();

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return dwResult;

} // symGetOptions

//******************************************************************************

BOOL
symGetScope
(
    ULONG64             BaseOfDll,
    DWORD               Index,
    PSYMBOL_INFO        Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get scope routine
        bResult = SymGetScope(s_hSymbol, BaseOfDll, Index, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetScope

//******************************************************************************

BOOL
symGetScopeW
(
    ULONG64             BaseOfDll,
    DWORD               Index,
    PSYMBOL_INFOW       Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get scope routine
        bResult = SymGetScopeW(s_hSymbol, BaseOfDll, Index, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetScopeW

//******************************************************************************

BOOL
symGetSearchPath
(
    PSTR                SearchPath,
    DWORD               SearchPathLength
)
{
    BOOL                bResult = FALSE;

    assert(SearchPath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get search path routine
        bResult = SymGetSearchPath(s_hSymbol, SearchPath, SearchPathLength);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSearchPath

//******************************************************************************

BOOL
symGetSearchPathW
(
    PWSTR               SearchPath,
    DWORD               SearchPathLength
)
{
    BOOL                bResult = FALSE;

    assert(SearchPath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get search path routine
        bResult = SymGetSearchPathW(s_hSymbol, SearchPath, SearchPathLength);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSearchPathW

//******************************************************************************

BOOL
symGetSymbolFile
(
    PCSTR               SymPath,
    PCSTR               ImageFile,
    DWORD               Type,
    PSTR                SymbolFile,
    size_t              cSymbolFile,
    PSTR                DbgFile,
    size_t              cDbgFile
)
{
    BOOL                bResult = FALSE;

    assert(ImageFile != NULL);
    assert(SymbolFile != NULL);
    assert(DbgFile != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol file routine
        bResult = SymGetSymbolFile(s_hSymbol, SymPath, ImageFile, Type, SymbolFile, cSymbolFile, DbgFile, cDbgFile);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymbolFile

//******************************************************************************

BOOL
symGetSymbolFileW
(
    PCWSTR              SymPath,
    PCWSTR              ImageFile,
    DWORD               Type,
    PWSTR               SymbolFile,
    size_t              cSymbolFile,
    PWSTR               DbgFile,
    size_t              cDbgFile
)
{
    BOOL                bResult = FALSE;

    assert(ImageFile != NULL);
    assert(SymbolFile != NULL);
    assert(DbgFile != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol file routine
        bResult = SymGetSymbolFileW(s_hSymbol, SymPath, ImageFile, Type, SymbolFile, cSymbolFile, DbgFile, cDbgFile);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymbolFileW

//******************************************************************************

BOOL
symGetTypeFromName
(
    ULONG64             BaseOfDll,
    PCSTR               Name,
    PSYMBOL_INFO        Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Name != NULL);
    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get type from name routine
        bResult = SymGetTypeFromName(s_hSymbol, BaseOfDll, Name, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetTypeFromName

//******************************************************************************

BOOL
symGetTypeFromNameW
(
    ULONG64             BaseOfDll,
    PCWSTR              Name,
    PSYMBOL_INFOW       Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Name != NULL);
    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get type from name routine
        bResult = SymGetTypeFromNameW(s_hSymbol, BaseOfDll, Name, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetTypeFromNameW

//******************************************************************************

BOOL
symGetTypeInfo
(
    DWORD64             ModBase,
    ULONG               TypeId,
    IMAGEHLP_SYMBOL_TYPE_INFO GetType,
    PVOID               pInfo
)
{
    BOOL                bResult = FALSE;

    assert(pInfo != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get type information routine
        bResult = SymGetTypeInfo(s_hSymbol, ModBase, TypeId, GetType, pInfo);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetTypeInfo

//******************************************************************************

BOOL
symGetTypeInfoEx
(
    DWORD64             ModBase,
    PIMAGEHLP_GET_TYPE_INFO_PARAMS Params
)
{
    BOOL                bResult = FALSE;

    assert(Params != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get type information extended routine
        bResult = SymGetTypeInfoEx(s_hSymbol, ModBase, Params);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetTypeInfoEx

//******************************************************************************

BOOL
symGetUnwindInfo
(
    DWORD64             Address,
    PVOID               Buffer,
    PULONG              Size
)
{
    BOOL                bResult = FALSE;

    assert(Buffer != NULL);
    assert(Size != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get unwind information routine
        bResult = SymGetUnwindInfo(s_hSymbol, Address, Buffer, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetUnwindInfo

//******************************************************************************

BOOL
symInitialize
(
    PCSTR               UserSearchPath,
    BOOL                fIlwadeProcess
)
{
    HANDLE              hHandle;
    BOOL                bResult = FALSE;

    // Symbol system should not be initialized
    assert(s_hSymbol == NULL);

    // Check for extension initialized and no symbol handle (Not initialized)
    if (isInitialized() && (s_hSymbol == NULL))
    {
        // Use the module handle as the symbol handle
        hHandle = getModuleHandle();

        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call the symbol initialize routine
        bResult = SymInitialize(hHandle, UserSearchPath, fIlwadeProcess);
        if (bResult)
        {
            // Save the symbol handle value
            s_hSymbol = hHandle;
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symInitialize

//******************************************************************************

BOOL
symInitializeW
(
    PCWSTR              UserSearchPath,
    BOOL                fIlwadeProcess
)
{
    HANDLE              hHandle;
    BOOL                bResult = FALSE;

    // Symbol system should not be initialized
    assert(s_hSymbol == NULL);

    // Check for extension initialized and no symbol handle (Not initialized)
    if (isInitialized() && (s_hSymbol == NULL))
    {
        // Use the module handle as the symbol handle
        hHandle = getModuleHandle();

        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call the symbol initialize routine
        bResult = SymInitializeW(hHandle, UserSearchPath, fIlwadeProcess);
        if (bResult)
        {
            // Save the symbol handle value
            s_hSymbol = hHandle;
        }
        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symInitializeW

//******************************************************************************
#ifndef _WIN64
DWORD
symLoadModule
(
    HANDLE              hFile,
    PCSTR               ImageName,
    PCSTR               ModuleName,
    DWORD               BaseOfDll,
    DWORD               SizeOfDll
)
{
    DWORD               dwResult = 0;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol load module routine
        dwResult = SymLoadModule(s_hSymbol, hFile, ImageName, ModuleName, BaseOfDll, SizeOfDll);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return dwResult;

} // symLoadModule
#endif
//******************************************************************************

DWORD64
symLoadModule64
(
    HANDLE              hFile,
    PCSTR               ImageName,
    PCSTR               ModuleName,
    DWORD64             BaseOfDll,
    DWORD               SizeOfDll
)
{
    DWORD64             qwResult = 0;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol load module routine
        qwResult = SymLoadModule64(s_hSymbol, hFile, ImageName, ModuleName, BaseOfDll, SizeOfDll);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return qwResult;

} // symLoadModule64

//******************************************************************************

DWORD64
symLoadModuleEx
(
    HANDLE              hFile,
    PCSTR               ImageName,
    PCSTR               ModuleName,
    DWORD64             BaseOfDll,
    DWORD               DllSize,
    PMODLOAD_DATA       Data,
    DWORD               Flags
)
{
    DWORD64             qwResult = 0;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol load module extended routine
        qwResult = SymLoadModuleEx(s_hSymbol, hFile, ImageName, ModuleName, BaseOfDll, DllSize, Data, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return qwResult;

} // symLoadModuleEx

//******************************************************************************

DWORD64
symLoadModuleExW
(
    HANDLE              hFile,
    PCWSTR              ImageName,
    PCWSTR              ModuleName,
    DWORD64             BaseOfDll,
    DWORD               DllSize,
    PMODLOAD_DATA       Data,
    DWORD               Flags
)
{
    DWORD64             qwResult = 0;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol load module extended routine
        qwResult = SymLoadModuleExW(s_hSymbol, hFile, ImageName, ModuleName, BaseOfDll, DllSize, Data, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return qwResult;

} // symLoadModuleExW

//******************************************************************************

BOOL
symMatchFileName
(
    PCSTR               FileName,
    PCSTR               Match,
    PSTR               *FileNameStop,
    PSTR               *MatchStop
)
{
    BOOL                bResult = FALSE;

    assert(FileName != NULL);
    assert(Match != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol match filename routine
        bResult = SymMatchFileName(FileName, Match, FileNameStop, MatchStop);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symMatchFileName

//******************************************************************************

BOOL
symMatchFileNameW
(
    PCWSTR              FileName,
    PCWSTR              Match,
    PWSTR              *FileNameStop,
    PWSTR              *MatchStop
)
{
    BOOL                bResult = FALSE;

    assert(FileName != NULL);
    assert(Match != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol match filename routine
        bResult = SymMatchFileNameW(FileName, Match, FileNameStop, MatchStop);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symMatchFileNameW

//******************************************************************************

BOOL
symMatchString
(
    PCSTR               string,
    PCSTR               expression,
    BOOL                fCase
)
{
    BOOL                bResult = FALSE;

    assert(string != NULL);
    assert(expression != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol match string routine
        bResult = SymMatchString(string, expression, fCase);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symMatchString

//******************************************************************************

BOOL
symMatchStringA
(
    PCSTR               string,
    PCSTR               expression,
    BOOL                fCase
)
{
    BOOL                bResult = FALSE;

    assert(string != NULL);
    assert(expression != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol match string routine
        bResult = SymMatchStringA(string, expression, fCase);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symMatchStringA

//******************************************************************************

BOOL
symMatchStringW
(
    PCWSTR              string,
    PCWSTR              expression,
    BOOL                fCase
)
{
    BOOL                bResult = FALSE;

    assert(string != NULL);
    assert(expression != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol match string routine
        bResult = SymMatchStringW(string, expression, fCase);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symMatchStringW

//******************************************************************************

BOOL
symNext
(
    PSYMBOL_INFO        si
)
{
    BOOL                bResult = FALSE;

    assert(si != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol next routine
        bResult = SymNext(s_hSymbol, si);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symNext

//******************************************************************************

BOOL
symNextW
(
    PSYMBOL_INFOW       siw
)
{
    BOOL                bResult = FALSE;

    assert(siw != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol next routine
        bResult = SymNextW(s_hSymbol, siw);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symNextW

//******************************************************************************

BOOL
symPrev
(
    PSYMBOL_INFO        si
)
{
    BOOL                bResult = FALSE;

    assert(si != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol previous routine
        bResult = SymPrev(s_hSymbol, si);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symPrev

//******************************************************************************

BOOL
symPrevW
(
    PSYMBOL_INFOW       siw
)
{
    BOOL                bResult = FALSE;

    assert(siw != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol previous routine
        bResult = SymPrevW(s_hSymbol, siw);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symPrevW

//******************************************************************************

BOOL
symRefreshModuleList()
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol refresh module list routine
        bResult = SymRefreshModuleList(s_hSymbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symRefreshModuleList

//******************************************************************************
#ifndef _WIN64
BOOL
symRegisterCallback
(
    PSYMBOL_REGISTERED_CALLBACK CallbackFunction,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(CallbackFunction != NULL);

    // Symbol handler should be initialized
    assert(s_hSymbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call the symbol register callback routine
        bResult = SymRegisterCallback(s_hSymbol, CallbackFunction, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symRegisterCallback
#endif
//******************************************************************************

BOOL
symRegisterCallback64
(
    PSYMBOL_REGISTERED_CALLBACK64 CallbackFunction,
    ULONG64             UserContext
)
{
    BOOL                bResult = FALSE;

    assert(CallbackFunction != NULL);

    // Symbol handler should be initialized
    assert(s_hSymbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call the symbol register callback routine
        bResult = SymRegisterCallback64(s_hSymbol, CallbackFunction, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symRegisterCallback64

//******************************************************************************

BOOL
symRegisterCallbackW64
(
    PSYMBOL_REGISTERED_CALLBACK64 CallbackFunction,
    ULONG64             UserContext
)
{
    BOOL                bResult = FALSE;

    assert(CallbackFunction != NULL);

    // Symbol handler should be initialized
    assert(s_hSymbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call the symbol register callback routine
        bResult = SymRegisterCallbackW64(s_hSymbol, CallbackFunction, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symRegisterCallbackW64

//******************************************************************************
#ifndef _WIN64
BOOL
symRegisterFunctionEntryCallback
(
    PSYMBOL_FUNCENTRY_CALLBACK CallbackFunction,
    PVOID               UserContext
)
{
    BOOL                bResult = FALSE;

    assert(CallbackFunction != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol register function entry callback routine
        bResult = SymRegisterFunctionEntryCallback(s_hSymbol, CallbackFunction, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symRegisterFunctionEntryCallback
#endif
//******************************************************************************

BOOL
symRegisterFunctionEntryCallback64
(
    PSYMBOL_FUNCENTRY_CALLBACK64 CallbackFunction,
    ULONG64             UserContext
)
{
    BOOL                bResult = FALSE;

    assert(CallbackFunction != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol register function entry callback routine
        bResult = SymRegisterFunctionEntryCallback64(s_hSymbol, CallbackFunction, UserContext);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symRegisterFunctionEntryCallback64

//******************************************************************************

BOOL
symSearch
(
    ULONG64             BaseOfDll,
    DWORD               Index,
    DWORD               SymTag,
    PCSTR               Mask,
    DWORD64             Address,
    PSYM_ENUMERATESYMBOLS_CALLBACK EnumSymbolsCallback,
    PVOID               UserContext,
    DWORD               Options
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol search routine
        bResult = SymSearch(s_hSymbol, BaseOfDll, Index, SymTag, Mask, Address, EnumSymbolsCallback, UserContext, Options);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSearch

//******************************************************************************

BOOL
symSearchW
(
    ULONG64             BaseOfDll,
    DWORD               Index,
    DWORD               SymTag,
    PCWSTR              Mask,
    DWORD64             Address,
    PSYM_ENUMERATESYMBOLS_CALLBACKW EnumSymbolsCallback,
    PVOID               UserContext,
    DWORD               Options
)
{
    BOOL                bResult = FALSE;

    assert(EnumSymbolsCallback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol search routine
        bResult = SymSearchW(s_hSymbol, BaseOfDll, Index, SymTag, Mask, Address, EnumSymbolsCallback, UserContext, Options);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSearchW

//******************************************************************************

BOOL
symSetContext
(
    PIMAGEHLP_STACK_FRAME StackFrame,
    PIMAGEHLP_CONTEXT   Context
)
{
    BOOL                bResult = FALSE;

    assert(StackFrame != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set context routine
        bResult = SymSetContext(s_hSymbol, StackFrame, Context);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSetContext

//******************************************************************************

PCHAR
symSetHomeDirectory
(
    PCSTR               dir
)
{
    PCHAR               pResult = NULL;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set home directory routine
        pResult = SymSetHomeDirectory(s_hSymbol, dir);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSetHomeDirectory

//******************************************************************************

PWSTR
symSetHomeDirectoryW
(
    PCWSTR              dir
)
{
    PWSTR               pResult = NULL;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set home directory routine
        pResult = SymSetHomeDirectoryW(s_hSymbol, dir);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSetHomeDirectoryW

//******************************************************************************

DWORD
symSetOptions
(
    DWORD               SymOptions
)
{
    DWORD               dwResult = 0;

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set options routine
        dwResult = SymSetOptions(SymOptions);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return dwResult;

} // symSetOptions

//******************************************************************************

BOOL
symSetScopeFromAddr
(
    ULONG64             Address
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set scope from address routine
        bResult = SymSetScopeFromAddr(s_hSymbol, Address);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSetScopeFromAddr

//******************************************************************************

BOOL
symSetScopeFromIndex
(
    ULONG64             BaseOfDll,
    DWORD               Index
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set scope from index routine
        bResult = SymSetScopeFromIndex(s_hSymbol, BaseOfDll, Index);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSetScopeFromIndex

//******************************************************************************

BOOL
symSetSearchPath
(
    PCSTR               SearchPath
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set search path routine
        bResult = SymSetSearchPath(s_hSymbol, SearchPath);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSetSearchPath

//******************************************************************************

BOOL
symSetSearchPathW
(
    PCWSTR              SearchPath
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol set search path routine
        bResult = SymSetSearchPathW(s_hSymbol, SearchPath);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSetSearchPathW

//******************************************************************************
#ifndef _WIN64
BOOL
symUnDName
(
    PIMAGEHLP_SYMBOL    sym,
    PSTR                UnDecName,
    DWORD               UnDecNameLength
)
{
    BOOL                bResult = FALSE;

    assert(sym != NULL);
    assert(UnDecName != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol undecorate name routine
        bResult = SymUnDName(sym, UnDecName, UnDecNameLength);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symUnDName
#endif
//******************************************************************************

BOOL
symUnDName64
(
    PIMAGEHLP_SYMBOL64  sym,
    PSTR                UnDecName,
    DWORD               UnDecNameLength
)
{
    BOOL                bResult = FALSE;

    assert(sym != NULL);
    assert(UnDecName != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol undecorate name routine
        bResult = SymUnDName64(sym, UnDecName, UnDecNameLength);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symUnDName64

//******************************************************************************
#ifndef _WIN64
BOOL
symUnloadModule
(
    DWORD               BaseOfDll
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol unload module routine
        bResult = SymUnloadModule(s_hSymbol, BaseOfDll);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symUnloadModule
#endif
//******************************************************************************

BOOL
symUnloadModule64
(
    DWORD64             BaseOfDll
)
{
    BOOL                bResult = FALSE;

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol unload module routine
        bResult = SymUnloadModule64(s_hSymbol, BaseOfDll);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symUnloadModule64

//******************************************************************************

PCSTR
symSrvDeltaName
(
    PCSTR               SymPath,
    PCSTR               Type,
    PCSTR               File1,
    PCSTR               File2
)
{
    PCSTR               pResult = NULL;

    assert(Type != NULL);
    assert(File1 != NULL);
    assert(File2 != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server delta name routine
        pResult = SymSrvDeltaName(s_hSymbol, SymPath, Type, File1, File2);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvDeltaName

//******************************************************************************

PCWSTR
symSrvDeltaNameW
(
    PCWSTR              SymPath,
    PCWSTR              Type,
    PCWSTR              File1,
    PCWSTR              File2
)
{
    PCWSTR              pResult = NULL;

    assert(Type != NULL);
    assert(File1 != NULL);
    assert(File2 != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server delta name routine
        pResult = SymSrvDeltaNameW(s_hSymbol, SymPath, Type, File1, File2);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvDeltaNameW

//******************************************************************************

BOOL
symSrvGetFileIndexes
(
    PCSTR               File,
    GUID               *Id,
    PDWORD              Val1,
    PDWORD              Val2,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(File != NULL);
    assert(Id != NULL);
    assert(Val1 != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get file indexes routine
        bResult = SymSrvGetFileIndexes(File, Id, Val1, Val2, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvGetFileIndexes

//******************************************************************************

BOOL
symSrvGetFileIndexesW
(
    PCWSTR              File,
    GUID               *Id,
    PDWORD              Val1,
    PDWORD              Val2,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(File != NULL);
    assert(Id != NULL);
    assert(Val1 != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get file indexes routine
        bResult = SymSrvGetFileIndexesW(File, Id, Val1, Val2, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvGetFileIndexesW

//******************************************************************************

BOOL
symSrvGetFileIndexInfo
(
    PCSTR               File,
    PSYMSRV_INDEX_INFO  Info,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(File != NULL);
    assert(Info != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get file index information routine
        bResult = SymSrvGetFileIndexInfo(File, Info, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvGetFileIndexInfo

//******************************************************************************

BOOL
symSrvGetFileIndexInfoW
(
    PCWSTR              File,
    PSYMSRV_INDEX_INFOW Info,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(File != NULL);
    assert(Info != NULL);

    // Check for extension initialized
    if (isInitialized())
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get file index information routine
        bResult = SymSrvGetFileIndexInfoW(File, Info, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvGetFileIndexInfoW

//******************************************************************************

BOOL
symSrvGetFileIndexString
(
    PCSTR               SrvPath,
    PCSTR               File,
    PSTR                Index,
    size_t              Size,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(File != NULL);
    assert(Index != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get file index string routine
        bResult = SymSrvGetFileIndexString(s_hSymbol, SrvPath, File, Index, Size, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvGetFileIndexString

//******************************************************************************

BOOL
symSrvGetFileIndexStringW
(
    PCWSTR              SrvPath,
    PCWSTR              File,
    PWSTR               Index,
    size_t              Size,
    DWORD               Flags
)
{
    BOOL                bResult = FALSE;

    assert(File != NULL);
    assert(Index != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get file index string routine
        bResult = SymSrvGetFileIndexStringW(s_hSymbol, SrvPath, File, Index, Size, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvGetFileIndexStringW

//******************************************************************************

PCSTR
symSrvGetSupplement
(
    PCSTR               SymPath,
    PCSTR               Node,
    PCSTR               File
)
{
    PCSTR               pResult = NULL;

    assert(Node != NULL);
    assert(File != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get supplement routine
        pResult = SymSrvGetSupplement(s_hSymbol, SymPath, Node, File);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvGetSupplement

//******************************************************************************

PCWSTR
symSrvGetSupplementW
(
    PCWSTR              SymPath,
    PCWSTR              Node,
    PCWSTR              File
)
{
    PCWSTR              pResult = NULL;

    assert(Node != NULL);
    assert(File != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server get supplement routine
        pResult = SymSrvGetSupplementW(s_hSymbol, SymPath, Node, File);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvGetSupplementW

//******************************************************************************

BOOL
symSrvIsStore
(
    PCSTR               path
)
{
    BOOL                bResult = FALSE;

    assert(path != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server is store routine
        bResult = SymSrvIsStore(s_hSymbol, path);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvIsStore

//******************************************************************************

BOOL
symSrvIsStoreW
(
    PCWSTR              path
)
{
    BOOL                bResult = FALSE;

    assert(path != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server is store routine
        bResult = SymSrvIsStoreW(s_hSymbol, path);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symSrvIsStoreW

//******************************************************************************

PCSTR
symSrvStoreFile
(
    PCSTR               SrvPath,
    PCSTR               File,
    DWORD               Flags
)
{
    PCSTR               pResult = NULL;

    assert(File != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server store file routine
        pResult = SymSrvStoreFile(s_hSymbol, SrvPath, File, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvStoreFile

//******************************************************************************

PCWSTR
symSrvStoreFileW
(
    PCWSTR              SrvPath,
    PCWSTR              File,
    DWORD               Flags
)
{
    PCWSTR              pResult = NULL;

    assert(File != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server store file routine
        pResult = SymSrvStoreFileW(s_hSymbol, SrvPath, File, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvStoreFileW

//******************************************************************************

PCSTR
symSrvStoreSupplement
(
    PCSTR               SrvPath,
    PCSTR               Node,
    PCSTR               File,
    DWORD               Flags
)
{
    PCSTR               pResult = NULL;

    assert(Node != NULL);
    assert(File != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server store supplement routine
        pResult = SymSrvStoreSupplement(s_hSymbol, SrvPath, Node, File, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvStoreSupplement

//******************************************************************************

PCWSTR
symSrvStoreSupplementW
(
    PCWSTR              SymPath,
    PCWSTR              Node,
    PCWSTR              File,
    DWORD               Flags
)
{
    PCWSTR              pResult = NULL;

    assert(Node != NULL);
    assert(File != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol server store supplement routine
        pResult = SymSrvStoreSupplementW(s_hSymbol, SymPath, Node, File, Flags);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return pResult;

} // symSrvStoreSupplementW

//******************************************************************************

BOOL
symGetSourceFile
(
    ULONG64             Base,
    PCSTR               Params,
    PCSTR               FileSpec,
    PSTR                FilePath,
    DWORD               Size
)
{
    BOOL                bResult = FALSE;

    assert(FileSpec != NULL);
    assert(FilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source file routine
        bResult = SymGetSourceFile(s_hSymbol, Base, Params, FileSpec, FilePath, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceFile

//******************************************************************************

BOOL
symGetSourceFileW
(
    ULONG64             Base,
    PCWSTR              Params,
    PCWSTR              FileSpec,
    PWSTR               FilePath,
    DWORD               Size
)
{
    BOOL                bResult = FALSE;

    assert(FileSpec != NULL);
    assert(FilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source file routine
        bResult = SymGetSourceFileW(s_hSymbol, Base, Params, FileSpec, FilePath, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceFileW

//******************************************************************************

BOOL
symEnumSourceFileTokens
(
    ULONG64             Base,
    PENUMSOURCEFILETOKENSCALLBACK Callback
)
{
    BOOL                bResult = FALSE;

    assert(Callback != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol enumerate source file tokens routine
        bResult = SymEnumSourceFileTokens(s_hSymbol, Base, Callback);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symEnumSourceFileTokens

//******************************************************************************

BOOL
symGetSourceFileFromToken
(
    PVOID               Token,
    PCSTR               Params,
    PSTR                FilePath,
    DWORD               Size
)
{
    BOOL                bResult = FALSE;

    assert(Token != NULL);
    assert(FilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source file from token routine
        bResult = SymGetSourceFileFromToken(s_hSymbol, Token, Params, FilePath, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceFileFromToken

//******************************************************************************

BOOL
symGetSourceFileFromTokenW
(
    PVOID               Token,
    PCWSTR              Params,
    PWSTR               FilePath,
    DWORD               Size
)
{
    BOOL                bResult = FALSE;

    assert(Token != NULL);
    assert(FilePath != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source file from token routine
        bResult = SymGetSourceFileFromTokenW(s_hSymbol, Token, Params, FilePath, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceFileFromTokenW

//******************************************************************************

BOOL
symGetSourceFileToken
(
    ULONG64             Base,
    PCSTR               FileSpec,
    PVOID              *Token,
    DWORD              *Size
)
{
    BOOL                bResult = FALSE;

    assert(FileSpec != NULL);
    assert(Token != NULL);
    assert(Size != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source file token routine
        bResult = SymGetSourceFileToken(s_hSymbol, Base, FileSpec, Token, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceFileToken

//******************************************************************************

BOOL
symGetSourceFileTokenW
(
    ULONG64             Base,
    PCWSTR              FileSpec,
    PVOID              *Token,
    DWORD              *Size
)
{
    BOOL                bResult = FALSE;

    assert(FileSpec != NULL);
    assert(Token != NULL);
    assert(Size != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source file token routine
        bResult = SymGetSourceFileTokenW(s_hSymbol, Base, FileSpec, Token, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceFileTokenW

//******************************************************************************

BOOL
symGetSourceVarFromToken
(
    PVOID               Token,
    PCSTR               Params,
    PCSTR               VarName,
    PSTR                Value,
    DWORD               Size
)
{
    BOOL                bResult = FALSE;

    assert(Token != NULL);
    assert(VarName != NULL);
    assert(Value != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source variable from routine
        bResult = SymGetSourceVarFromToken(s_hSymbol, Token, Params, VarName, Value, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceVarFromToken

//******************************************************************************

BOOL
symGetSourceVarFromTokenW
(
    PVOID               Token,
    PCWSTR              Params,
    PCWSTR              VarName,
    PWSTR               Value,
    DWORD               Size
)
{
    BOOL                bResult = FALSE;

    assert(Token != NULL);
    assert(VarName != NULL);
    assert(Value != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get source variable from routine
        bResult = SymGetSourceVarFromTokenW(s_hSymbol, Token, Params, VarName, Value, Size);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSourceVarFromTokenW

//******************************************************************************
#ifndef _WIN64
BOOL
symGetSymFromAddr
(
    DWORD               dwAddr,
    PDWORD              pdwDisplacement,
    PIMAGEHLP_SYMBOL    Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol from address routine
        bResult = SymGetSymFromAddr(s_hSymbol, dwAddr, pdwDisplacement, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymFromAddr
#endif
//******************************************************************************
#ifndef _WIN64
BOOL
symGetSymFromName
(
    PCSTR               Name,
    PIMAGEHLP_SYMBOL    Symbol
)
{
    BOOL                bResult= FALSE;

    assert(Name != NULL);
    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol from name routine
        bResult = SymGetSymFromName(s_hSymbol, Name, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymFromName
#endif
//******************************************************************************
#ifndef _WIN64
BOOL
symGetSymNext
(
    PIMAGEHLP_SYMBOL    Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol next routine
        bResult = SymGetSymNext(s_hSymbol, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymNext
#endif
//******************************************************************************
#ifndef _WIN64
BOOL
symGetSymPrev
(
    PIMAGEHLP_SYMBOL    Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol previous routine
        bResult = SymGetSymPrev(s_hSymbol, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymPrev
#endif
//******************************************************************************

BOOL
symGetSymFromAddr64
(
    DWORD64             qwAddr,
    PDWORD64            pdwDisplacement,
    PIMAGEHLP_SYMBOL64  Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol from address routine
        bResult = SymGetSymFromAddr64(s_hSymbol, qwAddr, pdwDisplacement, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymFromAddr64

//******************************************************************************

BOOL
symGetSymFromName64
(
    PCSTR               Name,
    PIMAGEHLP_SYMBOL64  Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Name != NULL);
    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol from name routine
        bResult = SymGetSymFromName64(s_hSymbol, Name, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymFromName64

//******************************************************************************

BOOL
symGetSymNext64
(
    PIMAGEHLP_SYMBOL64  Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol next routine
        bResult = SymGetSymNext64(s_hSymbol, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymNext64

//******************************************************************************

BOOL
symGetSymPrev64
(
    PIMAGEHLP_SYMBOL64  Symbol
)
{
    BOOL                bResult = FALSE;

    assert(Symbol != NULL);

    // Check for extension and symbol system initialized
    if (isInitialized() && (s_hSymbol != NULL))
    {
        // Acquire the DbgHelp interface
        acquireDbgHelpInterface();

        // Call symbol get symbol previous routine
        bResult = SymGetSymPrev64(s_hSymbol, Symbol);

        // Release the DbgHelp interface
        releaseDbgHelpInterface();

        // Check the progress indicator
        progressCheck();

        // Perform status check after every DbgHlp call
        statusCheck();
    }
    // Return the result
    return bResult;

} // symGetSymPrev64

//******************************************************************************

bool
symProperty
(
    const CModule      *pModule,
    DWORD               dwIndex,
    IMAGEHLP_SYMBOL_TYPE_INFO property
)
{
    TYPE_INFO           typeInfo;
    bool                bProperty;

    // Check to see if the index symbol supports this property
    bProperty = tobool(symGetTypeInfo(pModule->address(), dwIndex, property, &typeInfo));

    return bProperty;

} // symProperty

//******************************************************************************

DWORD
symTag
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwSymTag = 0;

    // Try to get the index symbol tag
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_SYMTAG, &typeInfo))
    {
        // Save the symbol tag value
        dwSymTag = typeInfo.dwSymTag;
    }
    else    // Unable to get index symbol tag
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol tag (TI_GET_SYMTAG) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol tag (TI_GET_SYMTAG) for index %d", dwIndex);
        }
    }
    return dwSymTag;

} // symTag

//******************************************************************************

CString
symName
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    CString             sTypeName(MAX_TYPE_LENGTH);

    // Try to get the symbol type name
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_SYMNAME, &typeInfo))
    {
        // Colwert symbol type name to ANSI
        sTypeName.sprintf("%ls", typeInfo.pSymName);

        // Free the name buffer
        LocalFree(typeInfo.pSymName);
    }
    else    // Unable to get symbol type name
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol name (TI_GET_SYMNAME) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol name (TI_GET_SYMNAME) for index %d", dwIndex);
        }
    }
    return sTypeName;

} // symName

//******************************************************************************

ULONG64
symLength
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    ULONG64             ulLength = 0;

    // Try to get the index symbol length
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_LENGTH, &typeInfo))
    {
        // Save the symbol length value
        ulLength = typeInfo.ulLength;
    }
    else    // Unable to get index symbol length
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol length (TI_GET_LENGTH) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol length (TI_GET_LENGTH) for index %d", dwIndex);
        }
    }
    return ulLength;

} // symLength

//******************************************************************************

DWORD
symType
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwType = 0;

    // Try to get the index symbol type
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_TYPE, &typeInfo))
    {
        // Save the symbol type value
        dwType = typeInfo.dwType;
    }
    else    // Unable to get index symbol type
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol type (TI_GET_TYPE) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol type (TI_GET_TYPE) for index %d", dwIndex);
        }
    }
    return dwType;

} // symType

//******************************************************************************

DWORD
symTypeId
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwTypeId = 0;

    // Try to get the index symbol type
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_TYPEID, &typeInfo))
    {
        // Save the symbol type value
        dwTypeId = typeInfo.dwTypeId;
    }
    else    // Unable to get index symbol type ID
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol type ID (TI_GET_TYPEID) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol type ID (TI_GET_TYPEID) for index %d", dwIndex);
        }
    }
    return dwTypeId;

} // symTypeId

//******************************************************************************

DWORD
symBaseType
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwBaseType = 0;

    // Try to get the index symbol type
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_BASETYPE, &typeInfo))
    {
        // Save the symbol base type value
        dwBaseType = typeInfo.dwBaseType;
    }
    else    // Unable to get index symbol base type
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol base type (TI_GET_BASETYPE) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol base type (TI_GET_BASETYPE) for index %d", dwIndex);
        }
    }
    return dwBaseType;

} // symBaseType

//******************************************************************************

DWORD
symArrayIndexTypeId
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwArrayIndexTypeId = 0;

    // Try to get the index symbol array index type ID
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_ARRAYINDEXTYPEID, &typeInfo))
    {
        // Save the symbol array index type ID value
        dwArrayIndexTypeId = typeInfo.dwArrayIndexTypeId;
    }
    else    // Unable to get index symbol array index type ID
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol array index type ID (TI_GET_ARRAYINDEXTYPEID) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol array index type ID (TI_GET_ARRAYINDEXTYPEID) for index %d", dwIndex);
        }
    }
    return dwArrayIndexTypeId;

} // symArrayIndexTypeId

//******************************************************************************

DWORD
symDataKind
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwDataKind = 0;

    // Try to get the index symbol data kind
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_DATAKIND, &typeInfo))
    {
        // Save the symbol data kind value
        dwDataKind = typeInfo.dwDataKind;
    }
    else    // Unable to get index symbol data kind
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol data kind (TI_GET_DATAKIND) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol data kind (TI_GET_DATAKIND) for index %d", dwIndex);
        }
    }
    return dwDataKind;

} // symDataKind

//******************************************************************************

DWORD
symAddressOffset
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwAddressOffset = 0;

    // Try to get the index symbol address offset
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_ADDRESSOFFSET, &typeInfo))
    {
        // Save the symbol address offset value
        dwAddressOffset = typeInfo.dwAddressOffset;
    }
    else    // Unable to get index symbol address offset
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol address offset (TI_GET_ADDRESSOFFSET) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol address offset (TI_GET_ADDRESSOFFSET) for index %d", dwIndex);
        }
    }
    return dwAddressOffset;

} // symAddressOffset

//******************************************************************************

DWORD
symOffset
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwOffset = 0;

    // Try to get the index symbol offset
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_OFFSET, &typeInfo))
    {
        // Save the symbol offset value
        dwOffset = typeInfo.dwOffset;
    }
    else    // Unable to get index symbol offset
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol offset (TI_GET_OFFSET) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol offset (TI_GET_OFFSET) for index %d", dwIndex);
        }
    }
    return dwOffset;

} // symOffset

//******************************************************************************

VARIANT
symValue
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    VARIANT             vValue;

    // Try to get the index symbol value
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_VALUE, &typeInfo))
    {
        // Save the symbol type value
        vValue = typeInfo.vValue;
    }
    else    // Unable to get index symbol value
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol value (TI_GET_VALUE) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol value (TI_GET_VALUE) for index %d", dwIndex);
        }
    }
    return vValue;

} // symValue

//******************************************************************************

DWORD
symCount
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwCount = 0;

    // Try to get the index symbol count
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_COUNT, &typeInfo))
    {
        // Save the symbol count value
        dwCount = typeInfo.dwCount;
    }
    else    // Unable to get index symbol count
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol count (TI_GET_COUNT) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol count (TI_GET_COUNT) for index %d", dwIndex);
        }
    }
    return dwCount;

} // symCount

//******************************************************************************

DWORD
symChildrenCount
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwChildrenCount = 0;

    // Try to get the index symbol children count
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_CHILDRENCOUNT, &typeInfo))
    {
        // Save the symbol children count value
        dwChildrenCount = typeInfo.dwChildrenCount;
    }
    else    // Unable to get index symbol children count
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol children count (TI_GET_CHILDRENCOUNT) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol children count (TI_GET_CHILDRENCOUNT) for index %d", dwIndex);
        }
    }
    return dwChildrenCount;

} // symChildrenCount

//******************************************************************************

DWORD
symBitPosition
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwBitPosition = 0;

    // Try to get the index symbol bit position
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_BITPOSITION, &typeInfo))
    {
        // Save the symbol bit position value
        dwBitPosition = typeInfo.dwBitPosition;
    }
    else    // Unable to get index symbol bit position
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol bit position (TI_GET_BITPOSITION) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol bit position (TI_GET_BITPOSITION) for index %d", dwIndex);
        }
    }
    return dwBitPosition;

} // symBitPosition

//******************************************************************************

bool
symVirtualBaseClass
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    bool                bVirtualBaseClass = false;

    // Try to get the index symbol virtual base class
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_VIRTUALBASECLASS, &typeInfo))
    {
        // Save the symbol virtual base class value
        bVirtualBaseClass = tobool(typeInfo.bVirtualBaseClass);
    }
    else    // Unable to get index symbol virtual base class
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base class (TI_GET_VIRTUALBASECLASS) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base class (TI_GET_VIRTUALBASECLASS) for index %d", dwIndex);
        }
    }
    return bVirtualBaseClass;

} // symVirtualBaseClass

//******************************************************************************

DWORD
symVirtualTableShapeId
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwVirtualTableShapeId = 0;

    // Try to get the index symbol virtual table shape ID
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_VIRTUALTABLESHAPEID, &typeInfo))
    {
        // Save the symbol virtual table shape ID value
        dwVirtualTableShapeId = typeInfo.dwVirtualTableShapeId;
    }
    else    // Unable to get index symbol virtual table shape ID
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual table shape ID (TI_GET_VIRTUALTABLESHAPEID) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual table shape ID (TI_GET_VIRTUALTABLESHAPEID) for index %d", dwIndex);
        }
    }
    return dwVirtualTableShapeId;

} // symVirtualTableShapeId

//******************************************************************************

DWORD
symVirtualBasePointerOffset
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwVirtualBasePointerOffset = 0;

    // Try to get the index symbol virtual base pointer offset
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_VIRTUALBASEPOINTEROFFSET, &typeInfo))
    {
        // Save the symbol virtual base pointer offset value
        dwVirtualBasePointerOffset = typeInfo.dwVirtualBasePointerOffset;
    }
    else    // Unable to get index symbol virtual base pointer offse
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base pointer offset (TI_GET_VIRTUALBASEPOINTEROFFSET) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base pointer offset (TI_GET_VIRTUALBASEPOINTEROFFSET) for index %d", dwIndex);
        }
    }
    return dwVirtualBasePointerOffset;

} // symVirtualBasePointerOffset

//******************************************************************************

DWORD
symClassParentId
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwClassParentId = 0;

    // Try to get the index symbol class parent ID
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_CLASSPARENTID, &typeInfo))
    {
        // Save the symbol class parent ID value
        dwClassParentId = typeInfo.dwClassParentId;
    }
    else    // Unable to get index symbol class parent ID
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol class parent ID (TI_GET_CLASSPARENTID) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol class parent ID (TI_GET_CLASSPARENTID) for index %d", dwIndex);
        }
    }
    return dwClassParentId;

} // symClassParentId

//******************************************************************************

DWORD
symNested
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwNested = 0;

    // Try to get the index symbol nested
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_NESTED, &typeInfo))
    {
        // Save the symbol nested value
        dwNested = typeInfo.dwNested;
    }
    else    // Unable to get index symbol nested
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol nested (TI_GET_NESTED) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol nested (TI_GET_NESTED) for index %d", dwIndex);
        }
    }
    return dwNested;

} // symNested

//******************************************************************************

DWORD
symSymIndex
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwSymIndex = 0;

    // Try to get the index symbol symbol index
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_SYMINDEX, &typeInfo))
    {
        // Save the symbol index value
        dwSymIndex = typeInfo.dwSymIndex;
    }
    else    // Unable to get index symbol index
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol index (TI_GET_SYMINDEX) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol index (TI_GET_SYMINDEX) for index %d", dwIndex);
        }
    }
    return dwSymIndex;

} // symSymIndex

//******************************************************************************

DWORD
symLexicalParent
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwLexicalParent = 0;

    // Try to get the index symbol type
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_LEXICALPARENT, &typeInfo))
    {
        // Save the symbol lexical parent value
        dwLexicalParent = typeInfo.dwLexicalParent;
    }
    else    // Unable to get index symbol lexical parent
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol lexical parent (TI_GET_LEXICALPARENT) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol lexical parent (TI_GET_LEXICALPARENT) for index %d", dwIndex);
        }
    }
    return dwLexicalParent;

} // symLexicalParent

//******************************************************************************

ULONG64
symAddress
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    ULONG64             ulAddress = 0;

    // Try to get the index symbol address
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_ADDRESS, &typeInfo))
    {
        // Save the symbol address value
        ulAddress = typeInfo.ulAddress;
    }
    else    // Unable to get index symbol address
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol address (TI_GET_ADDRESS) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol address (TI_GET_ADDRESS) for index %d", dwIndex);
        }
    }
    return ulAddress;

} // symAddress

//******************************************************************************

DWORD
symThisAdjust
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwThisAdjust = 0;

    // Try to get the index symbol this adjust
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_THISADJUST, &typeInfo))
    {
        // Save the symbol this adjust value
        dwThisAdjust = typeInfo.dwThisAdjust;
    }
    else    // Unable to get index symbol this adjust
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol this adjust (TI_GET_THISADJUST) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol this adjust (TI_GET_THISADJUST) for index %d", dwIndex);
        }
    }
    return dwThisAdjust;

} // symThisAdjust

//******************************************************************************

DWORD
symUdtKind
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwUdtKind = 0;

    // Try to get the index symbol UDT kind
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_UDTKIND, &typeInfo))
    {
        // Save the symbol UDT kind value
        dwUdtKind = typeInfo.dwUdtKind;
    }
    else    // Unable to get index symbol UDT kibd
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol UDT kind (TI_GET_UDTKIND) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol UDT kind (TI_GET_UDTKIND) for index %d", dwIndex);
        }
    }
    return dwUdtKind;

} // symUdtKind

//******************************************************************************

DWORD
symEquivTo
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwEquiv = 0;

    // Try to get the index symbol equiv to
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_IS_EQUIV_TO, &typeInfo))
    {
        // Save the symbol equiv to value
        dwEquiv = typeInfo.dwEquiv;
    }
    else    // Unable to get index symbol equiv to
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol equiv to (TI_IS_EQUIV_TO) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol equiv to (TI_IS_EQUIV_TO) for index %d", dwIndex);
        }
    }
    return dwEquiv;

} // symEquivTo

//******************************************************************************

DWORD
symCallingColwention
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwCallingColwention = 0;

    // Try to get the index symbol calling convention
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_CALLING_COLWENTION, &typeInfo))
    {
        // Save the symbol calling convention value
        dwCallingColwention = typeInfo.dwCallingColwention;
    }
    else    // Unable to get index symbol calling convention
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol calling convention (TI_GET_CALLING_COLWENTION) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol calling convention (TI_GET_CALLING_COLWENTION) for index %d", dwIndex);
        }
    }
    return dwCallingColwention;

} // symCallingColwention

//******************************************************************************

DWORD
symCloseEquivTo
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwCloseEquiv = 0;

    // Try to get the index symbol close equiv to
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_IS_CLOSE_EQUIV_TO, &typeInfo))
    {
        // Save the symbol close equiv value
        dwCloseEquiv = typeInfo.dwCloseEquiv;
    }
    else    // Unable to get index symbol close equiv to
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol close equiv to (TI_IS_CLOSE_EQUIV_TO) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol close equiv to (TI_IS_CLOSE_EQUIV_TO) for index %d", dwIndex);
        }
    }
    return dwCloseEquiv;

} // symCloseEquivTo

//******************************************************************************

ULONG64
symGtiExReqsValid
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    ULONG64             ulGtiExReqsValid = 0;

    // Try to get the index symbol GTIEx request valid
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GTIEX_REQS_VALID, &typeInfo))
    {
        // Save the symbol GTIEx request valid
        ulGtiExReqsValid = typeInfo.ulGtiExReqsValid;
    }
    else    // Unable to get index symbol GTIEx request valid
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol GTIEx request valid (TI_GTIEX_REQS_VALID) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol GTIEx request valid (TI_GTIEX_REQS_VALID) for index %d", dwIndex);
        }
    }
    return ulGtiExReqsValid;

} // symGtiExReqsValid

//******************************************************************************

DWORD
symVirtualBaseOffset
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwVirtualBaseOffset = 0;

    // Try to get the index symbol virtual base offset
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_VIRTUALBASEOFFSET, &typeInfo))
    {
        // Save the symbol virtual base offset value
        dwVirtualBaseOffset = typeInfo.dwVirtualBaseOffset;
    }
    else    // Unable to get index symbol virtual base offset
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base offset (TI_GET_VIRTUALBASEOFFSET) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base offset (TI_GET_VIRTUALBASEOFFSET) for index %d", dwIndex);
        }
    }
    return dwVirtualBaseOffset;

} // symVirtualBaseOffset

//******************************************************************************

DWORD
symVirtualBaseDispIndex
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwVirtualBaseDispIndex = 0;

    // Try to get the index symbol virtual base displacement index
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_VIRTUALBASEDISPINDEX, &typeInfo))
    {
        // Save the symbol virtual base displacement index value
        dwVirtualBaseDispIndex = typeInfo.dwVirtualBaseDispIndex;
    }
    else    // Unable to get index symbol virtual base displacement index
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base displacement index (TI_GET_VIRTUALBASEDISPINDEX) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base displacement index (TI_GET_VIRTUALBASEDISPINDEX) for index %d", dwIndex);
        }
    }
    return dwVirtualBaseDispIndex;

} // symVirtualBaseDispIndex

//******************************************************************************

bool
symIsReference
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    bool                bReference = false;

    // Try to get the index symbol is reference
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_IS_REFERENCE, &typeInfo))
    {
        // Save the symbol is reference value
        bReference = tobool(typeInfo.bReference);
    }
    else    // Unable to get index symbol type
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol is reference (TI_GET_IS_REFERENCE) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol is reference (TI_GET_IS_REFERENCE) for index %d", dwIndex);
        }
    }
    return bReference;

} // symIsReference

//******************************************************************************

bool
symIndirectVirtualBaseClass
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    bool                bIndirectVirtualBaseClass = false;

    // Try to get the index symbol is indirect virtual base class
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_INDIRECTVIRTUALBASECLASS, &typeInfo))
    {
        // Save the symbol is indirect virtual base class value
        bIndirectVirtualBaseClass = tobool(typeInfo.bIndirectVirtualBaseClass);
    }
    else    // Unable to get index symbol is indirect virtual base class
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol is indirect virtual base class (TI_GET_INDIRECTVIRTUALBASECLASS) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol is indirect virtual base class for (TI_GET_INDIRECTVIRTUALBASECLASS) index %d", dwIndex);
        }
    }
    return bIndirectVirtualBaseClass;

} // symIndirectVirtualBaseClass

//******************************************************************************

DWORD
symVirtualBaseTableType
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    TYPE_INFO           typeInfo;
    SYM_INFO            symbolInfo;
    HRESULT             hResult;
    DWORD               dwVirtualBaseTableType = 0;

    // Try to get the index symbol virtual base table type
    if (symGetTypeInfo(pModule->address(), dwIndex, TI_GET_VIRTUALBASETABLETYPE, &typeInfo))
    {
        // Save the symbol virtual base table type value
        dwVirtualBaseTableType = typeInfo.dwVirtualBaseTableType;
    }
    else    // Unable to get index symbol virtual base table type
    {
        // Get symGetTypeInfo error result
        hResult = HRESULT_FROM_WIN32(GetLastError());

        // Initialize the symbol information structure
        memset(&symbolInfo, 0, sizeof(symbolInfo));

        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the symbol information (for symbol name)
        if (symFromIndex(pModule->address(), dwIndex, &symbolInfo))
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base table type (TI_GET_VIRTUALBASETABLETYPE) for %s (%d)", symbolInfo.Name, dwIndex);
        }
        else    // Unable to get symbol information
        {
            // Throw symbol exception with the error
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting symbol virtual base table type (TI_GET_VIRTUALBASETABLETYPE) for index %d", dwIndex);
        }
    }
    return dwVirtualBaseTableType;

} // symVirtualBaseTableType

//******************************************************************************

DWORD
pointerType
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    DWORD               dwTypeId = dwIndex;

    assert(pModule != NULL);

    // This should be a pointer type symbol
    assert(symTag(pModule, dwIndex) == SymTagPointerType);

    // Loop thru the pointer redirections
    while (symTag(pModule, dwTypeId) == SymTagPointerType)
    {
        // Move to the next type ID
        dwTypeId = symTypeId(pModule, dwTypeId);
    }
    // Return the actual pointer type
    return dwTypeId;

} // pointerType

//******************************************************************************

DWORD
pointerCount
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    DWORD               dwTypeId = dwIndex;
    DWORD               dwPointers = 0;

    assert(pModule != NULL);

    // This should be a pointer type symbol
    assert(symTag(pModule, dwIndex) == SymTagPointerType);

    // Loop computing the number of pointer redirections
    while (symTag(pModule, dwTypeId) == SymTagPointerType)
    {
        // Increment pointer count
        dwPointers++;

        // Move to the next type ID
        dwTypeId = symTypeId(pModule, dwTypeId);
    }
    // Return the number of pointers
    return dwPointers;

} // pointerCount

//******************************************************************************

DWORD
arrayType
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    DWORD               dwTypeId = dwIndex;

    assert(pModule != NULL);

    // This should be an array type symbol
    assert(symTag(pModule, dwIndex) == SymTagArrayType);

    // Loop thru the array dimensions
    while (symTag(pModule, dwTypeId) == SymTagArrayType)
    {
        // Move to the next type ID
        dwTypeId = symTypeId(pModule, dwTypeId);
    }
    // Return the actual array type
    return dwTypeId;

} // arrayType

//******************************************************************************

DWORD
arrayDimensions
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    DWORD               dwTypeId = dwIndex;
    DWORD               dwDimensions = 0;

    assert(pModule != NULL);

    // This should be an array type symbol
    assert(symTag(pModule, dwIndex) == SymTagArrayType);

    // Loop computing the number of array dimensions
    while (symTag(pModule, dwTypeId) == SymTagArrayType)
    {
        // Increment array dimensions
        dwDimensions++;

        // Move to the next type ID
        dwTypeId = symTypeId(pModule, dwTypeId);
    }
    // Return the number of dimensions
    return dwDimensions;

} // arrayDimensions

//******************************************************************************

const char*
symbolTypeName
(
    DWORD               dwSymType
)
{
    const char         *pSymbolTypeName;

    // Check for a valid symbol type
    if (dwSymType < countof(s_SymbolType))
    {
        // Get the symbol type name
        pSymbolTypeName = s_SymbolType[dwSymType];
    }
    else    // Invalid/unknown symbol type
    {
        // Get unknown symbol type
        pSymbolTypeName = s_SymbolTypeUnknown;
    }
    return pSymbolTypeName;

} // symbolTypeName

//******************************************************************************

const char*
symbolTagName
(
    DWORD               dwSymTag
)
{
    const char         *pSymbolTagName;

    // Check for a valid symbol tag
    if (dwSymTag < countof(s_SymbolTag))
    {
        // Get the symbol tag name
        pSymbolTagName = s_SymbolTag[dwSymTag];
    }
    else    // Invalid/unknown symbol tag
    {
        // Get unknwon symbol tag
        pSymbolTagName = s_SymbolTagUnknown;
    }
    return pSymbolTagName;

} // symbolTagName

//******************************************************************************

const char*
callingColwentionName
(
    DWORD               dwCallingColwention
)
{
    const char         *pCallingColwentionName;

    // Check for a valid calling convention
    if (dwCallingColwention < countof(s_CallingColwention))
    {
        // Get the calling convention name
        pCallingColwentionName = s_CallingColwention[dwCallingColwention];
    }
    else    // Invalid/unknown calling convention
    {
        // Get unknown calling convention
        pCallingColwentionName = s_CallingColwentionUnknown;
    }
    return pCallingColwentionName;

} // callingColwentionName

//******************************************************************************

const char*
udtKindName
(
    DWORD               dwUdtKind
)
{
    const char         *pUdtKindName;

    // Check for a valid user defined type kind
    if (dwUdtKind < countof(s_UdtKind))
    {
        // Get the user defined type name
        pUdtKindName = s_UdtKind[dwUdtKind];
    }
    else    // Invalid/unknown UDT kind
    {
        // Get unknown user defined type kind
        pUdtKindName = s_UdtKindUnknown;
    }
    return pUdtKindName;

} // udtKindName

//******************************************************************************

HANDLE
symbolHandle()
{
    // Return the symbol handle
    return s_hSymbol;

} // symbolHandle

//******************************************************************************

HRESULT
symbolReset()
{
    CharArray           aSymbolPath;
    ULONG               ulPathLength = MAX_SYMBOL_PATH;
    HRESULT             hResult = S_OK;

    // Only reset symbol handler if initialized
    if (s_hSymbol != NULL)
    {
        // Try to get the current debugger symbol path length
        hResult = GetSymbolPath(NULL, 0, &ulPathLength);
        if (SUCCEEDED(hResult))
        {
            // Make sure we allocate enough space for the default path
            ulPathLength = max(ulPathLength, MAX_SYMBOL_PATH);
        }
        else    // Unable to get symbol path length
        {
            // Display warning and setup default symbol path
            dPrintf("Unable to get symbol path length, defaulting to %d\n", MAX_SYMBOL_PATH);
            ulPathLength = MAX_SYMBOL_PATH;
        }
        // Allocate space for the symbol path
        aSymbolPath = new char[ulPathLength];

        // Try to get the current debugger symbol path
        hResult = GetSymbolPath(aSymbolPath.ptr(), ulPathLength, &ulPathLength);
        if (!SUCCEEDED(hResult))
        {
            // Display warning and setup default path
            dPrintf("Unable to get current sympath defaulting to %s\n", DEFAULT_SYMBOL_PATH);
            strncpy(aSymbolPath.ptr(), DEFAULT_SYMBOL_PATH, ulPathLength);
        }
        // Try to set the new symbol search path
        if (SymSetSearchPath(s_hSymbol, aSymbolPath.ptr()))
        {
            // Indicate symbol handler reset
            hResult = S_OK;
        }
        else    // Failed to set new symbol path
        {
            // Get the failure code
            hResult = HRESULT_FROM_WIN32(GetLastError());
        }
    }
    return hResult;

} // symbolReset

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
