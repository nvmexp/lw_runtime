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
|*  Module: lwbucket.cpp                                                      *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************
//
//  Forwards
//
//******************************************************************************
DEBUGGER_INITIALIZE     DebugExtensionInitialize(PULONG pVersion, PULONG pFlags);
DEBUGGER_NOTIFY         DebugExtensionNotify(ULONG Notify, ULONG64 Argument);
DEBUGGER_UNINITIALIZE   DebugExtensionUninitialize(void);

static  HRESULT         displayLargeInteger(ULONG64 Address, PSTR Buffer, PULONG BufferSize);
static  HRESULT         displaySystemTime(ULONG64 Address, PSTR Buffer, PULONG BufferSize);

static  BOOL            processAttach();
static  BOOL            threadAttach();
static  BOOL            processDetach();
static  BOOL            threadDetach();

static  bool            checkSymbols();
char*                   terminate(char* pBuffer, char* pStart, ULONG ulSize);

//******************************************************************************
//
//  Locals
//
//******************************************************************************
// Thread local storage values
static  DWORD                   s_dwTlsIndex;                   // Thread local storage index
static  CThreadStorage*         s_pFirstThreadStorage  = NULL;  // First thread storage entry
static  CThreadStorage*         s_pLastThreadStorage   = NULL;  // Last thread storage entry
static  ULONG                   s_ulThreadStorageCount = 0;     // Thread storage count value

// Debug extension status values
static  bool                    s_bAttached    = false;         // DLL attached to process
static  ULONG                   s_ulReferenceCount = 0;         // Debugger extension reference count

// Debug session status values
static  bool                     s_bActive;                     // Debug session active flag
static  bool                     s_bAccessible;                 // Debug session accessible flag
static  bool                     s_bConnected;                  // Debug session connected flag
static  ULONG                    s_ulDebugClass;                // Debug session class
static  ULONG                    s_ulDebugQualifier;            // Debug session qualifier
static  ULONG                    s_ulActualMachine;             // Actual machine type
static  ULONG                    s_ulExelwtingMachine;          // Exelwting machine type
static  ULONG64                  s_ulDebuggerVersion;           // Debugger engine version

//  Address size/mask values
static  ULONG64                  s_ulAddressMask;               // Address mask value
static  ULONG                    s_ulAddressSize;               // Address size value

// LwExt debugger extension information
static  char                     s_LwExtPrefix[MAX_PATH] = "";  // LwExt extension prefix
static  char                     s_LwExtPath[MAX_PATH]   = "";  // LwExt extension path

// Known Structures
static  const KNOWN_STRUCTS      s_KnownStructs[] = {
                                                    {"_LARGE_INTEGER", TRUE,  displayLargeInteger},
                                                    {"_SYSTEMTIME",    FALSE, displaySystemTime},
                                                    {NULL,             FALSE, NULL}
                                                   };

static  ULONG                   s_SizeTable[] = {               /* Data format size table */
                                                 1,             /* BYTE format size */
                                                 2,             /* WORD format size */
                                                 4,             /* DWORD format size */
                                                 8,             /* QWORD format size */
                                                 4,             /* Single format size (float) */
                                                 8              /* Double format size (float) */
                                                  };

// Enumerations
CEnum                   s_ThreadInfoClassEnum       (&osKernel(), "_THREADINFOCLASS",  "THREADINFOCLASS");
CEnum                   s_ProcessInfoClassEnum      (&osKernel(), "_PROCESSINFOCLASS", "PROCESSINFOCLASS");

//******************************************************************************
//
// Main DLL entry point (Used for thread local storage)
//
//******************************************************************************
 
BOOL WINAPI
DllMain
(
    HINSTANCE           hDllInstance,
    DWORD               dwReason,
    LPVOID              lpReserved
)
{
    UNREFERENCED_PARAMETER(hDllInstance);
    UNREFERENCED_PARAMETER(lpReserved);

    BOOL                bResult = TRUE;

    // Switch on the reason this routine is being called
    switch (dwReason)
    {
        case DLL_PROCESS_ATTACH:                    // DLL is loading due to process initialization or LoadLibrary

            // Perform the process attach
            bResult = processAttach();
            if (bResult)
            {
                // Also perform a thread attach for initial process
                bResult = threadAttach();
            }
            break;

        case DLL_THREAD_ATTACH:                     // Attached process is creating a new thread

            // Perform the thread attach
            bResult = threadAttach();

            break;

        case DLL_THREAD_DETACH:                     // Thread of an attached process is terminating

            // Perform the thread detach
            bResult = threadDetach();

            break; 

        case DLL_PROCESS_DETACH:                    // DLL is unloading due to process termination or FreeLibrary

            // Perform the process detach
            bResult = processDetach();
            if (bResult)
            {
                // Also perform a thread detach for final process
                bResult = threadDetach();
            }
            break;

        default:                                    // Unknown reason

            // Indicate error (this shouldn't happen)
            bResult = FALSE;

            break;
    }
    return bResult;

} // DllMain

//******************************************************************************

static BOOL
processAttach()
{
    BOOL                bResult = TRUE;

    // Check for thread local storage requested
    if (s_ulThreadStorageCount != 0)
    {
        // Try to allocate a thread local storage index
        s_dwTlsIndex = TlsAlloc();
        if (s_dwTlsIndex == TLS_OUT_OF_INDEXES)
        {
            // Indicate failure
            bResult = FALSE;
        }
    }
    // Check for no thread local storage failure
    if (bResult)
    {
        // Initialize the debug interface
        initializeDebugInterface();

        // Attach thread for this process (Indicate DLL attached to a process)
        s_bAttached = true;
    }
    return bResult;

} // processAttach

//******************************************************************************

static BOOL
threadAttach()
{
    void               *pThreadStorage;

    // Check for thread local storage requested
    if (s_ulThreadStorageCount != 0)
    {
        // Initialize the TLS index for this thread (Allocate thread storage)
        pThreadStorage = malloc(s_pFirstThreadStorage->totalSize());
        if (pThreadStorage != NULL)
        {
            // Clear the thread local storage memory
            memset(pThreadStorage, 0, s_pFirstThreadStorage->totalSize());

            // Set the thread local storage value
            TlsSetValue(s_dwTlsIndex, pThreadStorage);
        }
    }
    return TRUE;

} // threadAttach

//******************************************************************************

static BOOL
processDetach()
{
    // Check for debugger extension still initialized (Abnormal debugger termination)
    if (isInitialized())
    {
        // Call debugger extension uninitialize routine
        DebugExtensionUninitialize();
    }
    // Uninitialize the debug interface
    uninitializeDebugInterface();

    // Detach thread for this process (Indicate DLL no longer attached to a process)
    s_bAttached = false;

    return TRUE;

} // processDetach

//******************************************************************************

static BOOL
threadDetach()
{
    void               *pThreadStorage;

    // Check for thread local storage requested
    if (s_ulThreadStorageCount != 0)
    {
        // Release any thread local storage allocated
        pThreadStorage = TlsGetValue(s_dwTlsIndex);
        if (pThreadStorage != NULL)
        {
            // Free the thread local storage
            free(pThreadStorage); 

            // Clear thread local storage value
            TlsSetValue(s_dwTlsIndex, NULL);
        }
    }
    return TRUE;

} // threadDetach

//******************************************************************************

CEffectiveProcessor::CEffectiveProcessor
(
    ULONG               ulEffectiveProcessor
)
{
    // Get the current effective processor type
    GetEffectiveProcessorType(&m_ulEffectiveProcessor);

    // Check to see if an effective processor type was requested
    if (ulEffectiveProcessor != IMAGE_FILE_MACHINE_UNKNOWN)
    {
        // Check to see if requested processor doesn't match current processor
        if (ulEffectiveProcessor != m_ulEffectiveProcessor)
        {
            // Set the requested effective processor type
            SetEffectiveProcessorType(ulEffectiveProcessor);

            // Update current pointer size/mask (Changes based on effective processor)
            updatePointerSize();

            // Set default input type based on pointer size (32/64 bit)
            if (pointerSize() == 32)
            {
                // Set default input type to 32-bit
                setInputType(DEBUG_VALUE_INT32);
            }
            else    // 64-bit pointers
            {
                // Set default input type to 64-bit
                setInputType(DEBUG_VALUE_INT64);
            }
        }
    }

} // CEffectiveProcessor

//******************************************************************************

CEffectiveProcessor::~CEffectiveProcessor()
{
    ULONG               ulEffectiveProcessor;

    // Get the current effective processor type
    GetEffectiveProcessorType(&ulEffectiveProcessor);

    // Check to see if current effective processor doesn't match original type
    if (ulEffectiveProcessor != effectiveProcessor())
    {
        // Restore the original effective processor type
        SetEffectiveProcessorType(effectiveProcessor());

        // Update current pointer size/mask (Changes based on effective processor)
        updatePointerSize();

        // Set default input type based on pointer size (32/64 bit)
        if (pointerSize() == 32)
        {
            // Set default input type to 32-bit
            setInputType(DEBUG_VALUE_INT32);
        }
        else    // 64-bit pointers
        {
            // Set default input type to 64-bit
            setInputType(DEBUG_VALUE_INT64);
        }
    }

} // ~CEffectiveProcessor

//******************************************************************************

CThreadStorage::CThreadStorage
(
    ULONG               ulStorageSize
)
:   m_pPrevThreadStorage(NULL),
    m_pNextThreadStorage(NULL),
    m_ulStorageSize(ulStorageSize),
    m_ulStorageOffset(0)
{
    // Shouldn't request 0 size
    assert(ulStorageSize != 0);

    // Add this thread storage to the thread storage list
    addThreadStorage(this);

    // Compute the actual offset for this thread storage
    if (m_pPrevThreadStorage != NULL)
    {
        m_ulStorageOffset = m_pPrevThreadStorage->storageOffset() + m_pPrevThreadStorage->storageSize();
    }

} // CThreadStorage

//******************************************************************************

CThreadStorage::~CThreadStorage()
{

} // ~CThreadStorage

//******************************************************************************

void
CThreadStorage::addThreadStorage
(
    CThreadStorage     *pThreadStorage
)
{
    assert(pThreadStorage != NULL);

    // Check for first thread storage
    if (s_pFirstThreadStorage == NULL)
    {
        // Set first and last hook to this hook
        s_pFirstThreadStorage = pThreadStorage;
        s_pLastThreadStorage  = pThreadStorage;
    }
    else    // Adding new thread storage to thread storage list
    {
        // Add this thread storage to the end of the thread storage list
        pThreadStorage->m_pPrevThreadStorage = s_pLastThreadStorage;
        pThreadStorage->m_pNextThreadStorage = NULL;

        s_pLastThreadStorage->m_pNextThreadStorage = pThreadStorage;

        s_pLastThreadStorage = pThreadStorage;
    }
    // Increment the thread storage count
    s_ulThreadStorageCount++;

} // addThreadStorage

//******************************************************************************

void*
CThreadStorage::threadStorage() const
{
    void               *pStorage = NULL;

    // Check for thread local storage available
    if (s_dwTlsIndex != 0)
    {
        // Check for thread local storage actually allocated
        pStorage = TlsGetValue(s_dwTlsIndex);
        if (pStorage != NULL)
        {
            // Compute the address of this thread storage
            pStorage = static_cast<char*>(pStorage) + storageOffset();
        }
    }
    return pStorage;

} // threadStorage

//******************************************************************************

ULONG
CThreadStorage::totalSize() const
{
    CThreadStorage     *pThreadStorage = s_pFirstThreadStorage;
    ULONG               ulTotalSize = 0;

    // Loop callwlating the total thread storage size
    while(pThreadStorage != NULL)
    {
        ulTotalSize    += pThreadStorage->storageSize();
        pThreadStorage  = pThreadStorage->nextThreadStorage();
    }
    return ulTotalSize;

} // totalSize

//******************************************************************************

DEBUGGER_INITIALIZE
DebugExtensionInitialize
(
    PULONG              pVersion,
    PULONG              pFlags
)
{
    HRESULT             hResult = S_OK;

    assert(pVersion != NULL);
    assert(pFlags != NULL);

    // Check to make sure this extension isn't already initialized
    if (!isInitialized())
    {
        // Set the LwExt debugger extension version and flags
        *pVersion = DEBUG_EXTENSION_VERSION(LWEXT_MAJOR_VERSION, LWEXT_MINOR_VERSION);
        *pFlags   = DEBUG_EXTINIT_HAS_COMMAND_HELP;

        // Initialize the debug session variables
        s_bActive            = false;
        s_bAccessible        = false;
        s_bConnected         = false;
        s_ulDebugClass       = 0;
        s_ulDebugQualifier   = 0;
        s_ulActualMachine    = 0;
        s_ulExelwtingMachine = 0;

        // Catch any initialization errors
        try
        {
            // Increment the extension reference count (Will indicate extension is initialized)
            s_ulReferenceCount++;

            // Get the debugger engine version
            s_ulDebuggerVersion = getDebuggerVersion();

            // Initialize the debugger interfaces
            initializeInterfaces();

            // Turn DML on by default
            dmlState(true);

            // Try to get the LwExt debugger extension information
            hResult = getExtensionInfo(LWEXT_MODULE_NAME, s_LwExtPrefix, s_LwExtPath);
            if (!SUCCEEDED(hResult))
            {
                // Display warning and ignore the error
                dPrintf("WARNING - Unable to get %s extension information\n", LWEXT_MODULE_NAME);
                hResult = S_OK;
            }
            // Check for no extension errors
            if (SUCCEEDED(hResult))
            {
                // Call initialize hooks if no initialize errors
                hResult = callInitializeHooks(pVersion, pFlags);
                if (FAILED(hResult))
                {
                    // Call uninitialize hooks if initialize hooks failed
                    callUninitializeHooks();
                }
            }
            // Clear initialized flag if initialization failed
            if (FAILED(hResult))
            {
                // Release the debugger interfaces
                releaseInterfaces();

                // Decrement the extension reference count
                s_ulReferenceCount--;
            }
        }
        catch (CException& exception)
        {
            // Display exception message and set error
            exception.dPrintf();
            hResult = exception.hResult();
            if (FAILED(hResult))
            {
                // Release the debugger interfaces
                releaseInterfaces();

                // Decrement the extension reference count
                s_ulReferenceCount--;
            }
        }
    }
    else    // Debugger extension is already initialized
    {
        // Just increment the extension reference count
        s_ulReferenceCount++;
    }
    return hResult;

} // DebugExtensionInitialize

//******************************************************************************

DEBUGGER_NOTIFY
DebugExtensionNotify
(
    ULONG               Notify,
    ULONG64             Argument
)
{
    HRESULT             hResult = S_OK;

    // Switch on the notify type
    switch(Notify)
    {
        case DEBUG_NOTIFY_SESSION_ACTIVE:

            // Indicate that a debug session is active
            s_bActive = true;

            break;

        case DEBUG_NOTIFY_SESSION_INACTIVE:

            // Indicate that no debug session is inactive (Disconnected)
            s_bActive            = false;
            s_bConnected         = false;
            s_bAccessible        = false;
            s_ulDebugClass       = 0;
            s_ulDebugQualifier   = 0;
            s_ulActualMachine    = 0;
            s_ulExelwtingMachine = 0;

            break;

        case DEBUG_NOTIFY_SESSION_ACCESSIBLE:

            // Indicate the debug session is accessible
            s_bAccessible = true;

            // Catch any notify errors
            try
            {
                // Check for the first connection since activation
                if (!s_bConnected)
                {
                    // Try to get the debug class and qualifier
                    hResult = GetDebuggeeType(&s_ulDebugClass, &s_ulDebugQualifier);
                    if (SUCCEEDED(hResult))
                    {
                        // Try to get the actual processor type (Using the debug control interface)
                        hResult = GetActualProcessorType(&s_ulActualMachine);
                        if (SUCCEEDED(hResult))
                        {
                            // Try to get the exelwting processor type (Using the debug control interface)
                            hResult = GetExelwtingProcessorType(&s_ulExelwtingMachine);
                            if (SUCCEEDED(hResult))
                            {
                                // Update current pointer size/mask (Changes based on effective processor)
                                updatePointerSize();

                                // Set default input type based on pointer size (32/64 bit)
                                if (pointerSize() == 32)
                                {
                                    // Set default input type to 32-bit
                                    setInputType(DEBUG_VALUE_INT32);
                                }
                                else    // 64-bit pointers
                                {
                                    // Set default input type to 64-bit
                                    setInputType(DEBUG_VALUE_INT64);
                                }
                                // Indicate the debug session is connected
                                s_bConnected = true;
                            }
                        }
                    }
                }
            }
            catch (CException& exception)
            {
                // Display exception message
                exception.dPrintf();
            }
            break;

        case DEBUG_NOTIFY_SESSION_INACCESSIBLE:

            // Indicate the debug session is inaccessible (Running)
            s_bAccessible = false;

            break;
    }
    // Call notify hooks
    callNotifyHooks(Notify, Argument);

} // DebugExtensionNotify

//******************************************************************************

DEBUGGER_UNINITIALIZE
DebugExtensionUninitialize
(
    void
)
{
    // Catch any uninitialization errors
    try
    {
        // Check for debugger extension initialized
        if (isInitialized())
        {
            // Check for time to uninitialize debugger extension (Final reference count being removed)
            if (s_ulReferenceCount == 1)
            {
                // Call uninitialize hooks
                callUninitializeHooks();

                // Release the debugger interfaces
                releaseInterfaces();
            }
            // Decrement the extension reference count
            s_ulReferenceCount--;
        }
    }
    catch (CException& exception)
    {
        // Display exception message
        exception.dPrintf();

        // Release the debugger interfaces
        releaseInterfaces();

        // Decrement the extension reference count
        s_ulReferenceCount--;
    }

} // DebugExtensionUninitialize

//******************************************************************************

DEBUGGER_KNOWN_OUTPUT
KnownStructOutput
(
    ULONG               Flag,
    ULONG64             Address,
    PSTR                StructName,
    PSTR                Buffer,
    PULONG              BufferSize
)
{
    ULONG               ulRemaining;
    ULONG               ulSize;
    ULONG               ulIndex;
    ULONG               ulLength;
    PSTR                pBuffer;
    HRESULT             hResult = S_OK;

    // Switch on the flag value
    switch(Flag)
    {
        case DEBUG_KNOWN_STRUCT_GET_NAMES:

            // Setup to copy known structure names (May fail)
            pBuffer     = Buffer;
            ulRemaining = *BufferSize;
            ulSize      = 0;

            // Loop through the known structure names
            for (ulIndex = 0; s_KnownStructs[ulIndex].pStructName != NULL; ulIndex++)
            {
                // Get length of the next known structure name (Include the terminator)
                ulLength = static_cast<ULONG>(strlen(s_KnownStructs[ulIndex].pStructName)) + 1;

                // Check for no errors yet (On copy or buffer too small)
                if (SUCCEEDED(hResult))
                {
                    // Copy next name to buffer if enough room available
                    if (ulRemaining > ulLength)
                    {
                        // Copy the next known structure name and move to the next
                        strcpy(pBuffer, s_KnownStructs[ulIndex].pStructName);
                        pBuffer     += ulLength;
                        ulRemaining -= ulLength;
                    }
                    else    // Not enough room in the buffer (Set failure)
                    {
                        hResult = S_FALSE;
                    }
                }
                // Update  the known structures name size
                ulSize += ulLength;
            }
            // Terminate the known structures multi-string and save size
            *pBuffer    = 0;
            *BufferSize = ulSize + 1;

            break;

        case DEBUG_KNOWN_STRUCT_GET_SINGLE_LINE_OUTPUT:

            // Default to no matching known structure name (Invalid name)
            hResult = E_ILWALIDARG;

            // Loop through the known structure names (Looking for a match)
            for (ulIndex = 0; s_KnownStructs[ulIndex].pStructName != NULL; ulIndex++)
            {
                // Check the next known structure name
                if (strcmp(StructName, s_KnownStructs[ulIndex].pStructName) == 0)
                {
                    // Call the function to display this known structure
                    hResult = (s_KnownStructs[ulIndex].pfnDisplayKnown)(Address, Buffer, BufferSize);
                    break;
                }
            }
            break;

        case DEBUG_KNOWN_STRUCT_SUPPRESS_TYPE_NAME:

            // Default to no matching known structure name (Invalid name)
            hResult = E_ILWALIDARG;

            // Loop through the known structure names
            for (ulIndex = 0; s_KnownStructs[ulIndex].pStructName != NULL; ulIndex++)
            {
                // Check the next known structure name
                if (strcmp(StructName, s_KnownStructs[ulIndex].pStructName) == 0)
                {
                    // Check if name should be suppressed
                    if (s_KnownStructs[ulIndex].bSuppressName)
                    {
                        hResult = S_OK;
                    }
                    else    // Display the structure name
                    {
                        hResult = S_FALSE;
                    }
                    break;
                }
            }
            break;

        default:

            // Set error result (Unknown flag value)
            hResult = E_ILWALIDARG;

            break;
    }
    return hResult;

} // KnownStructOutput

//******************************************************************************

static HRESULT
displayLargeInteger
(
    ULONG64             Address,
    PSTR                Buffer,
    PULONG              BufferSize
)
{
    UNREFERENCED_PARAMETER(BufferSize);

    ULONG64             ulData;
    ULONG               ulRead;
    HRESULT             hResult = S_OK;

    assert(Buffer != NULL);
    assert(BufferSize != NULL);

    // Try to read the large integer from memory (Uncached)
    hResult = ReadVirtualUncached(Address, &ulData, sizeof(ulData), &ulRead);
    if (SUCCEEDED(hResult))
    {
        // Check for able to read all the data
        if (ulRead == sizeof(ulData))
        {
            // Format the large integer into the given buffer
            sprintf(Buffer, " { %x`%08x }", (ULONG) (ulData >> 32), (ULONG) ulData);
        }
        else    // Unable to read all the data
            hResult = E_ILWALIDARG;
    }
    return hResult;

} // displayLargeInteger

//******************************************************************************

static HRESULT
displaySystemTime
(
    ULONG64             Address,
    PSTR                Buffer,
    PULONG              BufferSize
)
{
    UNREFERENCED_PARAMETER(BufferSize);

    SYSTEMTIME          SystemTime;
    ULONG               ulRead;
    HRESULT             hResult = S_OK;

    assert(Buffer != NULL);
    assert(BufferSize != NULL);

    // Try to read the system time from memory (Uncached)
    hResult = ReadVirtualUncached(Address, (PVOID) &SystemTime, sizeof(SystemTime), &ulRead);
    if (SUCCEEDED(hResult))
    {
        // Check for able to read all the data
        if (ulRead == sizeof(SystemTime))
        {
            // Format the system time into the given buffer
            sprintf(Buffer, " { %02d:%02d:%02d %02d/%02d/%04d }",
                    SystemTime.wHour,  SystemTime.wMinute, SystemTime.wSecond,
                    SystemTime.wMonth, SystemTime.wDay,    SystemTime.wYear);
        }
        else    // Unable to read all the data
            hResult = E_ILWALIDARG;
    }
    return hResult;

} // displayLargeInteger

//******************************************************************************

HRESULT
initializeGlobals
(
    bool                bThrow
)
{
    CString             commandString(MAX_COMMAND_STRING);
    CString             reloadString;
    HRESULT             hResult = S_OK;

    // Reset DML state in case it got left messed up
    dmlReset();

    // Clear verbose value to stop all debug tracing
    setCommandValue(VerboseOption, 0);

    // Check for no kernel mode driver information
    if (!kmDriver().isLoaded())
    {
        // Set error indicating kernel mode driver not found
        hResult = ERROR_FILE_NOT_FOUND;

        // Check for exception requested (Otherwise simply return the error)
        if (bThrow)
        {
            if (dmlState())
            {
                // Build command string and reload string for OS symbol reload
                commandString.assign(".reload /f ");
                commandString.append(osKernel().name());

                reloadString.assign("Press ");
                reloadString.append(exec("here", commandString));
                reloadString.append(" to reload OS symbols");

                throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get module information for '%s'.\n"
                                 "Please make sure OS symbols are loaded properly.\n"
                                 "%s for '%s'.",
                                 kmDriver().name(), reloadString, kmDriver().name());
            }
            else    // Plain text only
            {
                throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get module information for '%s'.\n"
                                 "Please make sure OS symbols are loaded properly.",
                                 kmDriver().name());
            }
        }
        else    // Do not throw an error
        {
            // Simply return the error code
            return hResult;
        }
    }
    // Check for valid kernel mode driver symbols
    if (!checkSymbols())
    {
        // Set error indicating kernel driver symbols not found
        hResult = ERROR_FILE_NOT_FOUND;

        // Check for exception requested (Otherwise simply return the error)
        if (bThrow)
        {
            if (dmlState())
            {
                // Build command string and reload string for driver symbol reload
                commandString.assign(".reload /f ");
                commandString.append(kmDriver().name());
                commandString.append(".sys");

                reloadString.assign("Press ");
                reloadString.append(exec("here", commandString));
                reloadString.append(" to reload driver symbols");

                throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                 ": Invalid driver symbols for '%s'.\n"
                                 "Please make sure driver symbols are loaded properly.\n"
                                 "%s for '%s'.",
                                 kmDriver().name(), reloadString, kmDriver().name());
            }
            else    // Plain text only
            {
                throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                 ": Invalid driver symbols for '%s'.\n"
                                 "Please make sure driver symbols are loaded properly.",
                                 kmDriver().name());
            }
        }
        else    // Do not throw an error
        {
            // Simply return the error code
            return hResult;
        }
    }
    // Return initialization result
    return hResult;

} // initializeGlobals

//******************************************************************************

void
initializeArguments
(
    PCSTR               args,
    char               *pCommand,
    int                *argc,
    char              **argv
)
{
    LONG                lOption;
    HRESULT             hResult;

    assert(args != NULL);
    assert(pCommand != NULL);
    assert(argc != NULL);
    assert(argv != NULL);

    // Reset the debug indentation
    dbgResetIndent();

    // Loop resetting all the argument options
    for (lOption = 0; lOption < OptionCount; lOption++)
    {
        // Initialize the command options, values, and masks
        resetOption(lOption);
        setCommandValue(lOption, 0);
        setMaskValue(lOption, INITIAL_MASK);

        // Initialize the search options and values
        resetSearch(lOption);
        setSearchValue(lOption, 0);

        // Initialize the verbose options
        resetVerbose(lOption);

        // Initialize the count options and values
        resetCount(lOption);
        setCountValue(lOption, 0);

        // Initialize the sort options and values
        resetSort(lOption);
        setSortValue(lOption, UnknownOption);
        setSortOrder(lOption, false);
    }
    // Initiailze the sort count
    setSortCount(0);

    // Try to parse the command arguments (Setup argc and argv)
    hResult = parseArguments(const_cast<char *>(args), pCommand, argc, argv);
    if (FAILED(hResult))
    {
        throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                         ": Couldn't parse '%s' command arguments",
                         pCommand);
    }
    // Initialize option parsing
    initializeOptions();

} // initializeArguments

//******************************************************************************

static bool
checkSymbols()
{
    SYM_TYPE            symType;
    bool                bValid = true;

    // Get the current OS kernel symbol type
    symType = osKernel().symType();

    // Only need to perform symbol load if no/deferred symbols
    if ((symType == SymNone) || (symType == SymDeferred))
    {
        // Assume symbols are not valid
        bValid = false;

        // Force OS kernel symbols to loaded
        osKernel().loadSymbols(FORCE_SYMBOL_LOAD);

        // Switch on the OS kernel symbol type (Updated)
        switch(osKernel().symType())
        {
            case SymNone:                   // No driver sybols

                // Print message indicating no OS symbols
                dPrintf("No symbols found for module '%s'!\n", osKernel().name());

                break;

            case SymExport:                 // Export only driver symbols

                // Print message indicating export only OS symbols
                dPrintf("Only export symbols found for module '%s'!\n", osKernel().name());

                break;

            case SymCoff:                   // Unsupport symbol types
            case SymCv:
            case SymSym:
            case SymDia:
            case SymVirtual:

                // Print message indicating unsupported OS symbols
                dPrintf("Unsupport symbol type %s (%d) for module '%s'!\n", symbolTypeName(osKernel().symType()), osKernel().symType(), osKernel().name());

                break;

            case SymPdb:                    // PDB symbol type (Correct type)

                // Check for private symbol (Type information) available
                if (osKernel().typeInfo())
                {
                    // Indicate OS symbols are valid
                    bValid = true;
                }
                else    // Only public (stripped) symbols
                {
                    // Print message indicating only public driver symbols
                    dPrintf("Only public symbols found for module '%s'!\n", osKernel().name());
                }
                break;

            default:                        // Unknown symbol type value

                // Print message indicating unknown OS symbol type
                dPrintf("Unknown symbol type (%d) for module '%s'!\n", osKernel().symType(), osKernel().name());

                break;
        }
    }
    return bValid;

} // checkSymbols

//******************************************************************************

void
updatePointerSize()
{
    // Check the current context pointer size
    if (IsPointer64Bit() == S_OK)
    {
        // Setup the address mask and size for 64-bit
        s_ulAddressMask = 0xffffffffffffffff;
        s_ulAddressSize = 64;
    }
    else    // 32-bit pointers
    {
        // Setup the address mask and size for 32-bit
        s_ulAddressMask = 0x00000000ffffffff;
        s_ulAddressSize = 32;
    }

} // updatePointerSize

//******************************************************************************

ULONG
pointerSize()
{
    // Return the pointer size
    return s_ulAddressSize;

} // pointerSize

//******************************************************************************

ULONG64
pointerMask()
{
    // Return the pointer mask
    return s_ulAddressMask;

} // pointerMask

//******************************************************************************

ULONG
pointerWidth()
{
    // Return the pointer width (Hexadecimal characters)
    return (pointerSize() / 4);

} // pointerWidth

//******************************************************************************

bool
isInitialized()
{
    // Return debugger extension initialized status (Reference count != 0)
    return (s_ulReferenceCount != 0);

} // isInitialized

//******************************************************************************

bool
isActive()
{
    // Return debug session active status
    return s_bActive;

} // isActive

//******************************************************************************

bool
isAccessible()
{
    // Return debug session accessible status
    return s_bAccessible;

} // isAccessible

//******************************************************************************

bool
isConnected()
{
    // Return debug session connected status
    return s_bConnected;

} // isConnected

//******************************************************************************

ULONG
debugClass()
{
    // Return the debug session debug class
    return s_ulDebugClass;

} // debugClass

//******************************************************************************

ULONG
debugQualifier()
{
    // Return the debug session debug qualifier
    return s_ulDebugQualifier;

} // debugQualifier

//******************************************************************************

ULONG
actualMachine()
{
    // Return the debug session actual machine
    return s_ulActualMachine;

} // actualMachine

//******************************************************************************

ULONG
exelwtingMachine()
{
    // Return the debug session exevuting machine
    return s_ulExelwtingMachine;

} // exelwtingMachine

//******************************************************************************

bool
isMachine32Bit
(
    ULONG               ulMachine
)
{
    // Return indicator if machine is 32-bit
    return ((ulMachine == IMAGE_FILE_MACHINE_I386)  ||
            (ulMachine == IMAGE_FILE_MACHINE_ARM)   ||
            (ulMachine == IMAGE_FILE_MACHINE_THUMB) ||
            (ulMachine == IMAGE_FILE_MACHINE_ARMNT));

} // isMachine32Bit

//******************************************************************************

bool
isMachine64Bit
(
    ULONG               ulMachine
)
{
    // Return indicator if machine is 64-bit
    return ((ulMachine == IMAGE_FILE_MACHINE_AMD64) ||
            (ulMachine == IMAGE_FILE_MACHINE_IA64));

} // isMachine64Bit

//******************************************************************************

bool
is32Bit()
{
    // Return indicator if 32-bit machine
    return isMachine32Bit(exelwtingMachine());

} // is32Bit

//******************************************************************************

bool
is64Bit()
{
    // Return indicator if 64-bit machine
    return isMachine64Bit(exelwtingMachine());

} // is64Bit

//******************************************************************************

bool
is32on64()
{
    // Return indicator if 32-bit on a 64-bit machine
    return (isMachine32Bit(exelwtingMachine()) && isMachine64Bit(actualMachine()));

} // is32on64

//******************************************************************************

bool
isUserMode()
{
    // Return indictor of user mode
    return (debugClass() == DEBUG_CLASS_USER_WINDOWS);

} // isUserMode

//******************************************************************************

bool
isKernelMode()
{
    // Return indicator of kernel mode
    return (debugClass() == DEBUG_CLASS_KERNEL);

} // isKernelMode

//******************************************************************************

bool
isDumpFile()
{
    // Return indicator of dump file
    return ((debugQualifier() == DEBUG_DUMP_SMALL)      ||
            (debugQualifier() == DEBUG_DUMP_DEFAULT)    ||
            (debugQualifier() == DEBUG_DUMP_FULL)       ||
            (debugQualifier() == DEBUG_DUMP_IMAGE_FILE) ||
            (debugQualifier() == DEBUG_DUMP_TRACE_LOG)  ||
            (debugQualifier() == DEBUG_DUMP_WINDOWS_CE));

} // isDumpFile

//******************************************************************************

ULONG64
debuggerVersion()
{
    // Return the debugger engine version (Complete value)
    return s_ulDebuggerVersion;

} // debuggerVersion

//******************************************************************************

ULONG
debuggerMajorVersion()
{
    // Return the debugger engine major version
    return static_cast<ULONG>((s_ulDebuggerVersion >> 48) & 0xffff);

} // debuggerMajorVersion

//******************************************************************************

ULONG
debuggerMinorVersion()
{
    // Return the debugger engine minor version
    return static_cast<ULONG>((s_ulDebuggerVersion >> 32) & 0xffff);

} // debuggerMinorVersion

//******************************************************************************

ULONG
debuggerReleaseNumber()
{
    // Return the debugger engine release number
    return static_cast<ULONG>((s_ulDebuggerVersion >> 16) & 0xffff);

} // debuggerReleaseNumber

//******************************************************************************

ULONG
debuggerBuildNumber()
{
    // Return the debugger engine build number
    return static_cast<ULONG>(s_ulDebuggerVersion & 0xffff);

} // debuggerBuildNumber

//******************************************************************************

bool
isDebuggerVersion
(
    ULONG               ulMajorVersion,
    ULONG               ulMinorVersion,
    ULONG               ulReleaseNumber,
    ULONG               ulBuildNumber
)
{
    bool                bIsDebuggerVersion = false;

    // Make sure we have a valid debugger version
    if (debuggerVersion() != 0)
    {
        // Check the major debugger version first
        if (debuggerMajorVersion() > ulMajorVersion)
        {
            // Indicate this is at least the requested debugger version
            bIsDebuggerVersion = true;
        }
        else if (debuggerMajorVersion() == ulMajorVersion)
        {
            // Check the minor debugger version next
            if (debuggerMinorVersion() > ulMinorVersion)
            {
                // Indicate this is at least the requested debugger version
                bIsDebuggerVersion = true;
            }
            else if (debuggerMinorVersion() == ulMinorVersion)
            {
                // Check the debugger release number next
                if (debuggerReleaseNumber() > ulReleaseNumber)
                {
                    // Indicate this is at least the requested debugger version
                    bIsDebuggerVersion = true;
                }
                else if (debuggerReleaseNumber() == ulReleaseNumber)
                {
                    // Finally check the debugger build number
                    if (debuggerBuildNumber() >= ulBuildNumber)
                    {
                        // Indicate this is at least the requested debugger version
                        bIsDebuggerVersion = true;
                    }
                }
            }
        }
    }
    else    // Invalid debugger version (None)
    {
        // Assume for now this debugger version is valid
        bIsDebuggerVersion = true;
    }
    return bIsDebuggerVersion;

} // isDebuggerVersion

//******************************************************************************

bool
isKernelModeAddress
(
    POINTER             ptrAddress
)
{
    // Check the current context pointer size (Hack for now)
    if (pointerSize() == 64)
    {
        // Check for address >= 0x8000000000000000
        return (ptrAddress > 0x8000000000000000);
    }
    else    // 32-bit pointer
    {
        // Check for address >= 0x80000000
        return (ptrAddress > 0x80000000);
    }

} // isKernelModeAddress

//******************************************************************************

HRESULT 
breakCheck
(
    HRESULT             hResult
)
{
    // Check for a ctrl-break from user
    if (userBreak(hResult))
    {
        // Don't throw a nested break exception
        if (CBreakException::reference() == 0)
        {
            // Throw a break exception
            throw CBreakException(hResult);
        }
    }
    return hResult;

} // breakCheck

//******************************************************************************

HRESULT
statusCheck()
{
    ULONG64             ulVerboseValue = commandValue(VerboseOption);
    HRESULT             hResult = S_OK;

    try
    {
        // Clear verbose value so nested tracing calls don't happen
        setCommandValue(VerboseOption, 0);

        // Check for user ctrl-c/ctrl-break
        hResult = GetInterrupt();
        if (hResult == S_OK)
        {
            // Set status to indicate ctrl-c/ctrl-break
            hResult = STATUS_CONTROL_BREAK_EXIT;
        }
        else    // No ctrl-c/ctrl-break
        {
            hResult = S_OK;
        }
        // Restore verbose value so tracing calls resume
        setCommandValue(VerboseOption, ulVerboseValue);
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Restore verbose value so tracing calls resume
        setCommandValue(VerboseOption, ulVerboseValue);

        throw;
    }
    return breakCheck(hResult);

} // statusCheck

//******************************************************************************

bool
userBreak
(
    HRESULT                 hResult
)
{
    bool                    bUserBreak;

    // Check for a user control-break result
    if (hResult == STATUS_CONTROL_BREAK_EXIT)
    {
        bUserBreak = true;
    }
    else
    {
        bUserBreak = false;
    }
    return bUserBreak;

} // userBreak

//******************************************************************************

bool
ignoreError
(
    HRESULT                 hResult
)
{
    bool                    bIgnoreError;

    // Check for result codes to ignore (User break and E_FAIL)
    if ((hResult == STATUS_CONTROL_BREAK_EXIT) || (hResult == E_FAIL))
    {
        bIgnoreError = true;
    }
    else
    {
        bIgnoreError = false;
    }
    return bIgnoreError;

} // ignoreError

//******************************************************************************

const char*
lwExtPrefix()
{
    // Return the LwExt extension prefix
    return s_LwExtPrefix;

} // lwExtPrefix

//******************************************************************************

const char*
lwExtPath()
{
    // Return the LwExt extension path
    return s_LwExtPath;

} // lwExtPath

//******************************************************************************

HANDLE
getModuleHandle()
{
    CString             sExtension;
    HANDLE              hHandle = NULL;

    // Build the extension string
    sExtension = lwExtPath();
    sExtension.append(LWEXT_MODULE_NAME);
    sExtension.append(DEBUG_FILE_EXTENSION);

    // Try to get the extension module handle
    hHandle = GetModuleHandle(sExtension);

    return hHandle;

} // getModuleHandle

//******************************************************************************

HRESULT
getExtensionInfo
(
    const CString&      sModule,
    char               *pPrefix,
    char               *pPath
)
{
    regex_t             reExt = {0};
    regex_t             reImg = {0};
    regex_t             rePath = {0};
    regex_t             reFile = {0};
    regex_t             reModule = {0};
    regmatch_t          reMatch[25];
    regex_t            *pRegEx = NULL;
    const char         *pRegExpr = NULL;
    char               *pBuffer = NULL;
    char               *pStart;
    char               *pEnd;
    CString             sFilename("");
    CString             sName("");
    CString             sRoot("");
    CString             sPrefix("");
    CString             sPath("");
    CString             sExpression;
    bool                bFoundChain = false;
    ULONG               ulSize;
    int                 reResult;
    HRESULT             hResult = E_FAIL;

    assert(pPrefix != NULL);
    assert(pPath != NULL);

    try
    {
        // Try to compile extension chain regular expression
        reResult = regcomp(&reExt, EXTEXPR, REG_EXTENDED + REG_ICASE);
        if (reResult == REG_NOERROR)
        {
            // Try to compile image regular expression
            reResult = regcomp(&reImg, IMGEXPR, REG_EXTENDED + REG_ICASE);
            if (reResult == REG_NOERROR)
            {
                // Try to compile path regular expression
                reResult = regcomp(&rePath, PATHEXPR, REG_EXTENDED + REG_ICASE);
                if (reResult == REG_NOERROR)
                {
                    // Try to compile filename regular expression
                    reResult = regcomp(&reFile, FILEEXPR, REG_EXTENDED + REG_ICASE);
                    if (reResult == REG_NOERROR)
                    {
                        // Build regular expression for module name
                        sExpression  = "^";
                        sExpression += sModule;
                        sExpression += "$";

                        // Try to compile module regular expression
                        reResult = regcomp(&reModule, sExpression, REG_EXTENDED + REG_ICASE);
                        if (reResult == REG_NOERROR)
                        {
                            // Indicate all regular expressions have compiled successfully
                            hResult = S_OK;
                        }
                        else    // Error compiling module regular expression
                        {
                            // Save regular expression error
                            pRegEx   = &reModule;
                            pRegExpr = sExpression;
                        }
                    }
                    else    // Error compiling filename regular expression
                    {
                        // Save regular expression error
                        pRegEx   = &reFile;
                        pRegExpr = FILEEXPR;
                    }
                }
                else    // Error compiling path regular expression
                {
                    // Save regular expression error
                    pRegEx   = &rePath;
                    pRegExpr = PATHEXPR;
                }
            }
            else    // Error compiling image regular expression
            {
                // Save regular expression error
                pRegEx   = &reImg;
                pRegExpr = IMGEXPR;
            }
        }
        else    // Error compiling extension chain regular expression
        {
            // Save regular expression error
            pRegEx   = &reExt;
            pRegExpr = EXTEXPR;
        }
        // Check for no errors compiling the regular expressions
        if (SUCCEEDED(hResult))        
        {
            // Clear the capture buffer
            ClearCapture();

            // Setup to capture extension information
            FlushCallbacks();
            SetCaptureState(TRUE);

            // Try to get the extension information
            Execute(DEBUG_OUTCTL_THIS_CLIENT | DEBUG_OUTCTL_NOT_LOGGED, ".chain", DEBUG_EXELWTE_NOT_LOGGED | DEBUG_EXELWTE_NO_REPEAT);

            // Turn output capture off (Make sure output is flushed)
            FlushCallbacks();
            SetCaptureState(FALSE);

            // Try to get the captured output
            hResult = getCaptureBuffer(&pBuffer, &ulSize);
            if (SUCCEEDED(hResult))
            {
                // Check for captured output present
                if (pBuffer != NULL)
                {
                    // Loop processing captured output lines
                    hResult = E_FAIL;
                    pStart  = pBuffer;
                    pEnd    = terminate(pBuffer, pStart, ulSize);
                    while (static_cast<ULONG>(pStart - pBuffer) < ulSize)
                    {
                        // Check for start of extension chain found
                        if (bFoundChain)
                        {
                            // Check to see if this is an image line (May not actually have image in it)
                            reResult = regexec(&reImg, pStart, countof(reMatch), reMatch, 0);
                            if (reResult == REG_NOERROR)
                            {
                                // Get and parse the image filename
                                sFilename = subExpression(pStart, reMatch, IMG_FILENAME);

                                reResult = regexec(&reFile, sFilename, countof(reMatch), reMatch, 0);
                                if (reResult == REG_NOERROR)
                                {
                                    // Get the name and check against requested module
                                    sName = subExpression(sFilename, reMatch, FILE_FILE);
                                    reResult = regexec(&reModule, sName, 0, NULL, 0);
                                    if (reResult == REG_NOERROR)
                                    {
                                        // Build the module prefix from the filename
                                        sRoot   = subExpression(sFilename, reMatch, FILE_ROOT);
                                        sPath   = subExpression(sFilename, reMatch, FILE_PATH);
                                        sPrefix = sRoot + sPath;

                                        // Check for valid prefix (Save if valid)
                                        if (sPrefix.length() < MAX_PATH)
                                        {
                                            // Copy the extension prefix
                                            strcpy(pPrefix, sPrefix);
                                        }
                                        // Skip to the next line (Extension path)
                                        pStart = pEnd + 1;
                                        pEnd   = terminate(pBuffer, pStart, ulSize);

                                        // Check to see if this is a path line
                                        reResult = regexec(&rePath, pStart, countof(reMatch), reMatch, 0);
                                        if (reResult == REG_NOERROR)
                                        {
                                            // Get and parse the path filename
                                            sFilename = subExpression(pStart, reMatch, PATH_FILENAME);
                                            reResult = regexec(&reFile, sFilename, countof(reMatch), reMatch, 0);
                                            if (reResult == REG_NOERROR)
                                            {
                                                // Build the module path from the filename
                                                sRoot = subExpression(sFilename, reMatch, FILE_ROOT);
                                                sPath = sRoot + subExpression(sFilename, reMatch, FILE_PATH);

                                                // Check for valid path (Save if valid)
                                                if (sPath.length() < MAX_PATH)
                                                {
                                                    // Copy the extension path
                                                    strcpy(pPath, sPath);
                                                }
                                                // Indicate extension information found
                                                hResult = S_OK;
                                                break;
                                            }
                                        }
                                        else    // Not the path line
                                        {
                                            // Simply break out of the loop and fail
                                            break;
                                        }
                                    }
                                    else    // Not the requested module
                                    {
                                        // Skip this line (not the requested module)
                                        pStart = pEnd + 1;
                                        pEnd   = terminate(pBuffer, pStart, ulSize);

                                        // Skip the next line (if present, should be path line)
                                        if (static_cast<ULONG>(pStart - pBuffer) < ulSize)
                                        {
                                            pStart = pEnd + 1;
                                            pEnd   = terminate(pBuffer, pStart, ulSize);
                                        }
                                    }
                                }
                                else    // Invalid image filename (Shouldn't happen)
                                {
                                    // Simply break out of the loop and fail
                                    break;
                                }
                            }
                            else    // Not an image line (Format change?)
                            {
                                // Skip this line (Invalid image filename)
                                pStart = pEnd + 1;
                                pEnd   = terminate(pBuffer, pStart, ulSize);
                            }
                        }
                        else    // Not at start of extension chain yet
                        {
                            // Check to see if this is the start of extension DLL chain
                            reResult = regexec(&reExt, pStart, countof(reMatch), reMatch, 0);
                            if (reResult == REG_NOERROR)
                            {
                                // Set flag indicating start of extension chain
                                bFoundChain = true;
                            }
                            // Move to the next line
                            pStart = pEnd + 1;
                            pEnd   = terminate(pBuffer, pStart, ulSize);
                        }
                    }
                    // Free the capture output buffer
                    delete pBuffer;
                    pBuffer = NULL;
                }
            }
        }
        else    // Error compiling regular expression
        {
            // Display regular expression error to the user
            dPrintf("%s\n", regString(reResult, pRegEx, pRegExpr));
        }
    }
    catch (...)
    {
        // Free capture buffer if allocated and throw error
        if (pBuffer != NULL)
        {
            delete pBuffer;
            pBuffer = NULL;
        }
        throw;
    }
    return hResult;

} // getExtensionInfo

//******************************************************************************

char*
terminate
(
    char               *pBuffer,
    char               *pStart,
    ULONG               ulSize
)
{
    char               *pEnd;

    // Try to find the end of the line (from the start)
    pEnd = strchr(pStart, EOL);
    if (pEnd != NULL)
    {
        // Terminate the string at the end
        *pEnd = NULL;
    }
    else    // No end of line found
    {
        // Point to end of the buffer
        pEnd = pBuffer + ulSize - 1;
    }
    return pEnd;

} // terminate

//******************************************************************************

CString
buildDbgCommand
(
    const CString&      sString
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the debug command string
    sCommand.sprintf("%s", sString);

    return sCommand;

} // buildDbgCommand

//******************************************************************************

CString
buildDbgCommand
(
    const CString&      sString,
    const CString&      sOptions
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the debug command string
    sCommand.sprintf("%s %s", sString, sOptions);

    return sCommand;

} // buildDbgCommand

//******************************************************************************

CString
buildDotCommand
(
    const CString&      sString
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the dot command string
    sCommand.sprintf(".%s", sString);

    return sCommand;

} // buildDotCommand

//******************************************************************************

CString
buildDotCommand
(
    const CString&      sString,
    const CString&      sOptions
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the dot command string
    sCommand.sprintf(".%s %s", sString, sOptions);

    return sCommand;

} // buildDotCommand

//******************************************************************************

CString
buildExtCommand
(
    const CString&      sString
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the extension command string
    sCommand.sprintf("!%s", sString);

    return sCommand;

} // buildExtCommand

//******************************************************************************

CString
buildExtCommand
(
    const CString&      sString,
    const CString&      sOptions
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the extension command string
    sCommand.sprintf("!%s %s", sString, sOptions);

    return sCommand;

} // buildExtCommand

//******************************************************************************

CString
buildModCommand
(
    const CString&      sString,
    const CString&      sModule
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the module command string
    sCommand.sprintf("!%s.%s", sModule, sString);

    return sCommand;

} // buildModCommand

//******************************************************************************

CString
buildModCommand
(
    const CString&      sString,
    const CString&      sModule,
    const CString&      sOptions
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the module command string
    sCommand.sprintf("!%s.%s %s", sModule, sString, sOptions);

    return sCommand;

} // buildModCommand

//******************************************************************************

CString
buildModPathCommand
(
    const CString&      sString,
    const CString&      sPath,
    const CString&      sModule
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the module path command string
    sCommand.sprintf("!%s%s.%s", sPath, sModule, sString);

    return sCommand;

} // buildModPathCommand

//******************************************************************************

CString
buildModPathCommand
(
    const CString&      sString,
    const CString&      sPath,
    const CString&      sModule,
    const CString&      sOptions
)
{
    CString             sCommand(MAX_COMMAND_STRING);

    // Build and return the command string
    sCommand.sprintf("!%s%s.%s %s", sPath, sModule, sString, sOptions);

    return sCommand;

} // buildModPathCommand

//******************************************************************************

CString
buildLwExtCommand
(
    const CString&      sString
)
{
    // Return an lWpu debugger extension command
    return buildModPathCommand(sString, s_LwExtPrefix, LWEXT_MODULE_NAME);

} // buildLwExtCommand

//******************************************************************************

CString
buildLwExtCommand
(
    const CString&      sString,
    const CString&      sOptions
)
{
    // Return an lWpu debugger extension command
    return buildModPathCommand(sString, s_LwExtPrefix, LWEXT_MODULE_NAME, sOptions);

} // buildLwExtCommand

//******************************************************************************

ULONG
dataSize
(
    DataFormat          dataFormat
)
{
    // Return the data size (Bytes) for the given data format
    return s_SizeTable[dataFormat];

} // dataSize

//******************************************************************************
//
//  Undefine new so we can define the new/delete operator overload functions
//
//******************************************************************************
#undef new

void* _cdecl
operator new
(
    size_t              size,
    const char         *pFunction,
    const char         *pFile,
    int                 nLine
)
{
    void               *pMemory = NULL;

    // Try to allocate the requested memory
    pMemory = malloc(size);
    if (pMemory == NULL)
    {
        // Throw a memory exception
        throw CMemoryException(size, pFunction, pFile, nLine);
    }
    return pMemory;

} // operator new

//******************************************************************************

void* _cdecl
operator new[]
(
    size_t              size,
    const char         *pFunction,
    const char         *pFile,
    int                 nLine
)
{
    void               *pMemory = NULL;

    // Try to allocate the requested memory
    pMemory = malloc(size);
    if (pMemory == NULL)
    {
        // Throw a memory exception
        throw CMemoryException(size, pFunction, pFile, nLine);
    }
    return pMemory;

} // operator new

//******************************************************************************

void _cdecl
operator delete
(
    void               *pMemory,
    const char         *pFunction,
    const char         *pFile,
    int                 nLine
)
{
    UNREFERENCED_PARAMETER(pFunction);
    UNREFERENCED_PARAMETER(pFile);
    UNREFERENCED_PARAMETER(nLine);

    // Free the allocated memory
    free(pMemory);

} // operator delete

//******************************************************************************

void _cdecl
operator delete[]
(
    void               *pMemory,
    const char         *pFunction,
    const char         *pFile,
    int                 nLine
)
{
    UNREFERENCED_PARAMETER(pFunction);
    UNREFERENCED_PARAMETER(pFile);
    UNREFERENCED_PARAMETER(nLine);

    // Free the allocated memory
    free(pMemory);

} // operator delete

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
