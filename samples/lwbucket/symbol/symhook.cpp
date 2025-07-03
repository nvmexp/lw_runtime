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
|*  Module: symhook.cpp                                                       *|
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
extern "C"
{
static  BOOL CALLBACK   symbolCallback(HANDLE hProcess, ULONG ActionCode, ULONG64 CallbackData, ULONG64 UserContext);
}

//******************************************************************************
//
// Locals
//
//******************************************************************************
static  CSymbolHook     s_symbolHook;                   // Symbol hook handler

static  DWORD           s_dwSymbolOptions = 0;          // Symbol options

//******************************************************************************

HRESULT
CSymbolHook::initialize
(
    const PULONG        pVersion,
    const PULONG        pFlags
)
{
    UNREFERENCED_PARAMETER(pVersion);
    UNREFERENCED_PARAMETER(pFlags);

    CharArray           aSymbolPath;
    ULONG               ulPathLength = MAX_SYMBOL_PATH;
    HRESULT             hResult = S_OK;

    assert(pVersion != NULL);
    assert(pFlags != NULL);

    // Catch any initialization failures
    try
    {
        // Initialize symbol operations
        initializeSymbols();

        // Initialize the DbgHelp interface
        initializeDbgHelp();

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
            dbgPrintf("Unable to get symbol path length, defaulting to %d\n", MAX_SYMBOL_PATH);
            ulPathLength = MAX_SYMBOL_PATH;
        }
        // Allocate space for the symbol path
        aSymbolPath = new char[ulPathLength];

        // Try to get the current debugger symbol path
        hResult = GetSymbolPath(aSymbolPath.ptr(), ulPathLength, &ulPathLength);
        if (!SUCCEEDED(hResult))
        {
            // Display warning and setup default path
            dbgPrintf("Unable to get current sympath defaulting to %s\n", DEFAULT_SYMBOL_PATH);
            strncpy(aSymbolPath.ptr(), DEFAULT_SYMBOL_PATH, ulPathLength);
        }
        // Try to initialize the symbol handler
        if (symInitialize(aSymbolPath.ptr(), FALSE))
        {
            // Get the current symbol options
            s_dwSymbolOptions = symGetOptions();

            // Setup the new symbol options
            symSetOptions(SYMOPT_CASE_INSENSITIVE       |
                          SYMOPT_UNDNAME                |
                          SYMOPT_DEFERRED_LOADS         |
                          SYMOPT_LOAD_LINES             |
                          SYMOPT_OMAP_FIND_NEAREST      |
                          SYMOPT_NO_UNQUALIFIED_LOADS   |
                          SYMOPT_FAIL_CRITICAL_ERRORS   |
                          SYMOPT_AUTO_PUBLICS           |
                          SYMOPT_NO_IMAGE_SEARCH);

            // Register the symbol callback handler
            if (!symRegisterCallback64(symbolCallback, NULL))
            {
                // Get and throw the failure
                hResult = GetLastError();

                throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                 "Failed to register symbol callback handler (0x%08x)\n",
                                 hResult);
            }
        }
        else    // Failed to initialize symbol handler
        {
            // Get and throw the failure
            hResult = GetLastError();

            throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                             "Failed to initialize symbol handler (0x%08x)\n",
                             hResult);
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Unintiialize the symbol handler
        uninitialize();

        throw;
    }
    return hResult;

} // initialize

//******************************************************************************

void
CSymbolHook::notify
(
    ULONG               Notify,
    ULONG64             Argument
)
{
    UNREFERENCED_PARAMETER(Argument);

    CEffectiveProcessor effectiveProcessor(actualMachine());

    try
    {
        // Switch on the notify type
        switch(Notify)
        {
            case DEBUG_NOTIFY_SESSION_ACTIVE:



                break;

            case DEBUG_NOTIFY_SESSION_INACTIVE:

#pragma message("  Should probably destroy all symbol sets if session going away")

                break;

            case DEBUG_NOTIFY_SESSION_ACCESSIBLE:



                break;

            case DEBUG_NOTIFY_SESSION_INACCESSIBLE:



                break;
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Simply ignore error any exceptions here (Hopefully none)
    }

} // notify

//******************************************************************************

void
CSymbolHook::uninitialize(void)
{
    // Check for symbol system initialized
    if (symbolHandle() != NULL)
    {
        // Restore the original symbol options
        symSetOptions(s_dwSymbolOptions);
        s_dwSymbolOptions = 0;

        // Cleanup the symbol handler
        symCleanup();
    }
    // Uninitialize the DbgHelp interface
    uninitializeDbgHelp();

    // Uninitialize symbol operations
    uninitializeSymbols();

} // uninitialize

//******************************************************************************

extern "C"
{
static BOOL CALLBACK
symbolCallback
(
    HANDLE              hProcess,
    ULONG               ActionCode,
    ULONG64             CallbackData,
    ULONG64             UserContext
)
{
    UNREFERENCED_PARAMETER(hProcess);
    UNREFERENCED_PARAMETER(UserContext);

    IMAGEHLP_DEFERRED_SYMBOL_LOAD64 *pImageHlpDeferredSymbolLoad;
    IMAGEHLP_DUPLICATE_SYMBOL64 *pImageHlpDuplicateSymbol;
    IMAGEHLP_CBA_EVENT *pImageHlpCbaEvent;
    IMAGEHLP_CBA_READ_MEMORY *pImageHlpCbaReadMemory;
    char               *pVerboseInformation;
    BOOL                bResult = FALSE;

    // Switch on the action code
    switch(ActionCode)
    {
        case CBA_DEFERRED_SYMBOL_LOAD_START:

            // Setup pointer to ImageHlp deferred symbol load structure
            pImageHlpDeferredSymbolLoad = reinterpret_cast<IMAGEHLP_DEFERRED_SYMBOL_LOAD64 *>(static_cast<ULONG_PTR>(CallbackData));



            break;

        case CBA_DEFERRED_SYMBOL_LOAD_COMPLETE:

            // Setup pointer to ImageHlp deferred symbol load structure
            pImageHlpDeferredSymbolLoad = reinterpret_cast<IMAGEHLP_DEFERRED_SYMBOL_LOAD64 *>(static_cast<ULONG_PTR>(CallbackData));



            break;

        case CBA_DEFERRED_SYMBOL_LOAD_FAILURE:

            // Setup pointer to ImageHlp deferred symbol load structure
            pImageHlpDeferredSymbolLoad = reinterpret_cast<IMAGEHLP_DEFERRED_SYMBOL_LOAD64 *>(static_cast<ULONG_PTR>(CallbackData));



            break;

        case CBA_SYMBOLS_UNLOADED:


            break;

        case CBA_DUPLICATE_SYMBOL:

            // Setup pointer to ImageHlp duplicate symbol structure
            pImageHlpDuplicateSymbol = reinterpret_cast<IMAGEHLP_DUPLICATE_SYMBOL64 *>(static_cast<ULONG_PTR>(CallbackData));



            break;

        case CBA_READ_MEMORY:

            // Setup pointer to ImageHlp CBA read memory structure
            pImageHlpCbaReadMemory = reinterpret_cast<IMAGEHLP_CBA_READ_MEMORY *>(static_cast<ULONG_PTR>(CallbackData));
            if (pImageHlpCbaReadMemory != NULL)
            {
                // Try to read the requested memory from the target system
                // Perform this read as *uncached* (As we could be nested from a DbgHelp call)
                *pImageHlpCbaReadMemory->bytesread = readCpuVirtual(TARGET(pImageHlpCbaReadMemory->addr),
                                                                    pImageHlpCbaReadMemory->buf,
                                                                    pImageHlpCbaReadMemory->bytes,
                                                                    UNCACHED);

                // Check for successful read
                if (*pImageHlpCbaReadMemory->bytesread == pImageHlpCbaReadMemory->bytes)
                {
                    // Indicate success
                    bResult = TRUE;
                }
            }
            break;

        case CBA_DEFERRED_SYMBOL_LOAD_CANCEL:


            break;

        case CBA_SET_OPTIONS:


            break;

        case CBA_EVENT:

            // Setup pointer to ImageHlp CBA event structure
            pImageHlpCbaEvent = reinterpret_cast<IMAGEHLP_CBA_EVENT *>(static_cast<ULONG_PTR>(CallbackData));



            break;

        case CBA_DEFERRED_SYMBOL_LOAD_PARTIAL:

            // Setup pointer to ImageHlp deferred symbol load structure
            pImageHlpDeferredSymbolLoad = reinterpret_cast<IMAGEHLP_DEFERRED_SYMBOL_LOAD64 *>(static_cast<ULONG_PTR>(CallbackData));



            break;

        case CBA_DEBUG_INFO:

            // Get pointer to verbose information string
            pVerboseInformation = reinterpret_cast<char *>(static_cast<ULONG_PTR>(CallbackData));



            break;

        default:


            break;
    }
    return bResult;

} // symbolCallback
}

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
