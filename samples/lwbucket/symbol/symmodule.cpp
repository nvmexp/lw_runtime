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
|*  Module: symmodule.cpp                                                     *|
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
static bool             isModuleCached(const CModule* pModule);
static bool             isModuleCached(const CModuleInstance* pModule);

static void             findModuleInformation(const CModule* pModule);
static void             findModuleInformation(const CModuleInstance* pModule);

//******************************************************************************

CModule::CModule
(
    const char         *pName,
    const char         *pFullName,
    ModuleType          moduleType
)
:   m_pPrevModule(NULL),
    m_pNextModule(NULL),
    m_pPrevKernelModule(NULL),
    m_pNextKernelModule(NULL),
    m_pPrevUserModule(NULL),
    m_pNextUserModule(NULL),
    m_pFirstType(NULL),
    m_pLastType(NULL),
    m_ulTypesCount(0),
    m_pFirstField(NULL),
    m_pLastField(NULL),
    m_ulFieldsCount(0),
    m_pFirstEnum(NULL),
    m_pLastEnum(NULL),
    m_ulEnumsCount(0),
    m_pFirstGlobal(NULL),
    m_pLastGlobal(NULL),
    m_ulGlobalsCount(0),
    m_pName(pName),
    m_pFullName(pFullName),
    m_ModuleType(moduleType),
    m_ulIndex(UNCACHED_INDEX),
    m_ulAddress(UNCACHED_ADDRESS)
{
    assert(pName != NULL);

    // Initialize the imagehlp module structure
    memset(&m_ImageHlpModule, 0, sizeof(m_ImageHlpModule));
    m_ImageHlpModule.SizeOfStruct = sizeof(m_ImageHlpModule);

    // Add this module to the module list
    addModule(this);

} // CModule

//******************************************************************************

CModule::~CModule()
{

} // ~CModule

//******************************************************************************

bool
CModule::isLoaded() const
{
    bool                bLoaded = false;

    // Make sure module information is loaded
    loadInformation();

    // Check to see if the module is loaded (Address is non-zero)
    if (address() != NOT_PRESENT_ADDRESS)
    {
        // Set module loaded indicator
        bLoaded = true;
    }
    return bLoaded;

} // isLoaded

//******************************************************************************

bool
CModule::hasSymbols() const
{
    bool                bSymbols = false;

    // Module must be loaded to have symbols
    if (isLoaded())
    {
        // Make sure module symbols are loaded
        loadSymbols(FORCE_SYMBOL_LOAD);

        // Check to see if the module has symbols
        if (symType() != SymNone)
        {
            // Indicate module has symbols
            bSymbols = true;
        }
    }
    return bSymbols;

} // hasSymbols

//******************************************************************************

HRESULT
CModule::loadInformation() const
{
    CString             sReload;
    CString             sOptions(MAX_COMMAND_STRING);
    HRESULT             hResult = S_OK;

    // Only load module information if not cached
    if (!isCached())
    {
        // Try to get the module information
        hResult = GetModuleByModuleName(name(), 0, &m_ulIndex, &m_ulAddress);
        if (!SUCCEEDED(hResult))
        {
            // Build command to reload module information (Requires full module name)
            // (Don't use Reload API as that will display errors if module not loaded)
            sOptions.sprintf("%s", fullName());
            sReload = buildDotCommand("reload", sOptions);

            // Try issuing reload command to sync engine module list with the target
            hResult = Execute(DEBUG_OUTCTL_THIS_CLIENT | DEBUG_OUTCTL_NOT_LOGGED, sReload, DEBUG_EXELWTE_NOT_LOGGED | DEBUG_EXELWTE_NO_REPEAT);
            if (SUCCEEDED(hResult))
            {
                // Try to get module information one more time
                hResult = GetModuleByModuleName(name(), 0, &m_ulIndex, &m_ulAddress);
                if (!SUCCEEDED(hResult))
                {
                    // Set the module index and address (Cached but not loaded)
                    m_ulIndex   = NOT_PRESENT_INDEX;
                    m_ulAddress = NOT_PRESENT_ADDRESS;
                }
            }
            else    // Failed to reload module information
            {
                // Set the module index and address (Cached but not loaded)
                m_ulIndex   = NOT_PRESENT_INDEX;
                m_ulAddress = NOT_PRESENT_ADDRESS;
            }
        }
    }
    return hResult;

} // loadInformation

//******************************************************************************

HRESULT
CModule::resetInformation() const
{
    HRESULT             hResult = S_OK;

    // Only reset module information if cached
    if (isCached())
    {
        // Check to see if this module is loaded
        if (isLoaded())
        {
            // Check to see if symbols are loaded (Need to be unloaded)
            if (symType() != SymNone)
            {
                // Unload the module symbols
                hResult = unloadSymbols();
            }
        }
        // Reset module index and address to uncached values
        m_ulIndex   = UNCACHED_INDEX;
        m_ulAddress = UNCACHED_ADDRESS;

        // Initialize the imagehlp module structure
        memset(&m_ImageHlpModule, 0, sizeof(m_ImageHlpModule));
        m_ImageHlpModule.SizeOfStruct = sizeof(m_ImageHlpModule);
    }
    return hResult;

} // resetInformation

//******************************************************************************

HRESULT
CModule::loadSymbols
(
    bool                bForce
) const
{
    DWORD               dwSymOpt = 0;
    HRESULT             hResult = S_OK;

    // Only try to load module symbols if module loaded
    if (isLoaded())
    {
        // Check to see if module symbols not loaded or forced defer
        if ((symType() == SymNone) || ((symType() == SymDeferred) && bForce))
        {
            // Check for forced symbol load (Need to update symopts if so)
            if (bForce)
            {
                // Get the current symbol options
                dwSymOpt = SymGetOptions();

                // Turn off the deferred symbol loads
                SymSetOptions(dwSymOpt & ~SYMOPT_DEFERRED_LOADS);
            }
            // Save the effective processor mode (In case we have to change it for symbol load)
            CEffectiveProcessor EffectiveProcessor;

            // If kernel module make sure we are in the proper processor mode
            if (isKernelModule())
            {
                SetEffectiveProcessorType(actualMachine());
            }
            // Try to load the symbols for the given module
            if (SymLoadModuleEx(symbolHandle(), NULL, name(), name(), address(), 0, NULL, 0))
            {
                // Update the module information
                SymGetModuleInfo64(symbolHandle(), address(), &m_ImageHlpModule);
            }
            else    // Unable to load module symbols
            {
                // Get the error result
                hResult = GetLastError();
                if (!SUCCEEDED(hResult))
                {
                    // Display warning to the user
                    dprintf("Unable to load symbols for module '%s' (0x%08x)!\n", name(), hResult);
                }
            }
            // Check for forced symbol load (Need to restore symopts if so)
            if (bForce)
            {
                // Restore the original symbol options
                SymSetOptions(dwSymOpt);
            }
        }
        // Check for deferred symbols (Need to update module information)
        if (symType() == SymDeferred)
        {
            // Update module information to see if symbols are now loaded
            SymGetModuleInfo64(symbolHandle(), address(), &m_ImageHlpModule);
        }
    }
    return hResult;

} // loadSymbols

//******************************************************************************

HRESULT
CModule::unloadSymbols() const
{
    HRESULT             hResult = S_OK;

    // Only try to unload module symbols if module loaded
    if (isLoaded())
    {
        // Check to see if module symbols loaded
        if (symType() != SymNone)
        {
            // Try to unload the symbols for the given module
            if (SymUnloadModule64(symbolHandle(), address()))
            {
                // Reset the module information
                memset(&m_ImageHlpModule, 0, sizeof(m_ImageHlpModule));
                m_ImageHlpModule.SizeOfStruct = sizeof(m_ImageHlpModule);
            }
            else    // Unable to unload module symbols
            {
                // Get the error result
                hResult = GetLastError();
                if (!SUCCEEDED(hResult))
                {
                    // Display warning to the user
                    dprintf("Unable to unload symbols for module '%s' (0x%08x)!\n", name(), hResult);
                }
            }
        }
    }
    return hResult;

} // unloadModuleSymbols

//******************************************************************************

HRESULT
CModule::reloadSymbols
(
    bool                bForce
) const
{
    HRESULT             hResult;

    // First attempt to unload the symbols for the module
    hResult = unloadSymbols();
    if (SUCCEEDED(hResult))
    {
        // Now try to load the symbols for the module
        hResult = loadSymbols(bForce);
    }
    return hResult;

} // reloadSymbols

//******************************************************************************

void
CModule::addModule
(
    CModule            *pModule
)
{
    assert(pModule != NULL);

    // Check for first module
    if (m_pFirstModule == NULL)
    {
        // Set first and last module to this module
        m_pFirstModule = pModule;
        m_pLastModule  = pModule;
    }
    else    // Adding new module to module list
    {
        // Add this module to the end of the module list
        pModule->m_pPrevModule = m_pLastModule;
        pModule->m_pNextModule = NULL;

        m_pLastModule->m_pNextModule = pModule;

        m_pLastModule = pModule;
    }
    // Increment the modules count
    m_ulModulesCount++;

    // Check to see if this is a kernel module
    if (isKernelModule())
    {
        // Check for first kernel module
        if (m_pFirstKernelModule == NULL)
        {
            // Set first and last kernel module to this module
            m_pFirstKernelModule = pModule;
            m_pLastKernelModule  = pModule;
        }
        else    // Adding new kernel module to kernel module list
        {
            // Add this kernel module to the end of the kernel module list
            pModule->m_pPrevKernelModule = m_pLastKernelModule;
            pModule->m_pNextKernelModule = NULL;

            m_pLastKernelModule->m_pNextKernelModule = pModule;

            m_pLastKernelModule = pModule;
        }
        // Set module instance to kernel module count (and increment kernel module count)
        m_ulInstance = m_ulKernelModuleCount++;
    }
    else    // Not a kernel module (Must be a user module)
    {
        // Check for first user module
        if (m_pFirstUserModule == NULL)
        {
            // Set first and last user module to this module
            m_pFirstUserModule = pModule;
            m_pLastUserModule  = pModule;
        }
        else    // Adding new user module to user module list
        {
            // Add this user module to the end of the user module list
            pModule->m_pPrevUserModule = m_pLastUserModule;
            pModule->m_pNextUserModule = NULL;

            m_pLastUserModule->m_pNextUserModule = pModule;

            m_pLastUserModule = pModule;
        }
        // Set module instance to user module count (and increment user module count)
        m_ulInstance = m_ulUserModuleCount++;
    }

} // addModule

//******************************************************************************

void
CModule::addType
(
    CType              *pType
) const
{
    assert(pType != NULL);

    // Check for first type
    if (m_pFirstType == NULL)
    {
        // Set first and last type to this type
        m_pFirstType = pType;
        m_pLastType  = pType;
    }
    else    // Adding new type to type list
    {
        // Add this type to the end of the type list
        pType->m_pPrevModuleType = m_pLastType;
        pType->m_pNextModuleType = NULL;

        m_pLastType->m_pNextModuleType = pType;

        m_pLastType = pType;
    }
    // Set type instance to module type count (and increment types count)
    pType->m_ulInstance = m_ulTypesCount++;

} // addType

//******************************************************************************

void
CModule::addField
(
    CField             *pField
) const
{
    assert(pField != NULL);

    // Check for first field
    if (m_pFirstField == NULL)
    {
        // Set first and last field to this field
        m_pFirstField = pField;
        m_pLastField  = pField;
    }
    else    // Adding new field to field list
    {
        // Add this field to the end of the field list
        pField->m_pPrevModuleField = m_pLastField;
        pField->m_pNextModuleField = NULL;

        m_pLastField->m_pNextModuleField = pField;

        m_pLastField = pField;
    }
    // Set field instance to module field count (and increment fields count)
    pField->m_ulInstance = m_ulFieldsCount++;

} // addField

//******************************************************************************

void
CModule::addEnum
(
    CEnum              *pEnum
) const
{
    assert(pEnum != NULL);

    // Check for first enum
    if (m_pFirstEnum == NULL)
    {
        // Set first and last enum to this enum
        m_pFirstEnum = pEnum;
        m_pLastEnum  = pEnum;
    }
    else    // Adding new enum to enum list
    {
        // Add this enum to the end of the enum list
        pEnum->m_pPrevModuleEnum = m_pLastEnum;
        pEnum->m_pNextModuleEnum = NULL;

        m_pLastEnum->m_pNextModuleEnum = pEnum;

        m_pLastEnum = pEnum;
    }
    // Set enum instance to module enum count (and increment enums count)
    pEnum->m_ulInstance = m_ulEnumsCount++;

} // addEnum

//******************************************************************************

void
CModule::addGlobal
(
    CGlobal            *pGlobal
) const
{
    assert(pGlobal != NULL);

    // Check for first global
    if (m_pFirstGlobal == NULL)
    {
        // Set first and last global to this global
        m_pFirstGlobal = pGlobal;
        m_pLastGlobal  = pGlobal;
    }
    else    // Adding new global to global list
    {
        // Add this global to the end of the global list
        pGlobal->m_pPrevModuleGlobal = m_pLastGlobal;
        pGlobal->m_pNextModuleGlobal = NULL;

        m_pLastGlobal->m_pNextModuleGlobal = pGlobal;

        m_pLastGlobal = pGlobal;
    }
    // Set global instance to module global count (and increment globals count)
    pGlobal->m_ulInstance = m_ulGlobalsCount++;

} // addGlobal

//******************************************************************************

CModuleInstance::CModuleInstance
(
    const CModule      *pModule,
    const CSymbolSession *pSession
)
:   m_pModule(pModule),
    m_ulIndex(UNCACHED_INDEX),
    m_ulAddress(UNCACHED_ADDRESS),
    m_pSymbolSet(NULL)
{
    assert(pModule != NULL);
    assert(pSession != NULL);

    // This had better be a kernel module
    assert(pModule->isKernelModule());

    // Initialize the imagehlp module structure
    memset(&m_ImageHlpModule, 0, sizeof(m_ImageHlpModule));
    m_ImageHlpModule.SizeOfStruct = sizeof(m_ImageHlpModule);

    // Try to create the symbol set for this module
    m_pSymbolSet = new CSymbolSet(this, pSession);

} // CModuleInstance

//******************************************************************************

CModuleInstance::CModuleInstance
(
    const CModule      *pModule,
    const CSymbolProcess *pProcess
)
:   m_pModule(pModule),
    m_ulIndex(UNCACHED_INDEX),
    m_ulAddress(UNCACHED_ADDRESS),
    m_pSymbolSet(NULL)
{
    assert(pModule != NULL);
    assert(pProcess != NULL);

    // This had better be a user module
    assert(pModule->isUserModule());

    // Initialize the imagehlp module structure
    memset(&m_ImageHlpModule, 0, sizeof(m_ImageHlpModule));
    m_ImageHlpModule.SizeOfStruct = sizeof(m_ImageHlpModule);

    // Try to create the symbol set for this module
    m_pSymbolSet = new CSymbolSet(this, pProcess);

} // CModuleInstance

//******************************************************************************

CModuleInstance::~CModuleInstance()
{

} // ~CModuleInstance

//******************************************************************************

const CSymbolSession*
CModuleInstance::symbolSession() const
{
    // Return the session for this symbol set
    return m_pSymbolSet->session();

} // symbolSession

//******************************************************************************

const CSymbolProcess*
CModuleInstance::symbolProcess() const
{
    // Return the process for this symbol set
    return m_pSymbolSet->process();

} // symbolProcess

//******************************************************************************

bool
CModuleInstance::validContext() const
{
    bool                bValidContext = true;

    // Check for a kernel (session) or user (process) module
    if (isKernelModule())
    {
        // Check to make sure symbol session matches module session
        if (symbolSession() != getLwrrentSession())
        {
            // Indicate wrong session context
            bValidContext = false;
        }
    }
    else    // Not a kernel module, user module
    {
        // Check to make sure current session/process matches module session/process
        if (symbolSession() == getLwrrentSession())
        {
            // Correct session context, check for correct process context
            if (symbolProcess() != symbolSession()->getLwrrentProcess())
            {
                // Indicate wrong process context
                bValidContext = false;
            }
        }
        else    // Incorrect session context
        {
            // Indicate wrong session context
            bValidContext = false;
        }
    }
    return bValidContext;

} // validContext

//******************************************************************************

bool
CModuleInstance::isLoaded() const
{
    bool                bLoaded = false;

    // Make sure module information is loaded
    loadInformation();

    // Check to see if the module is loaded (Address is non-zero)
    if (address() != NOT_PRESENT_ADDRESS)
    {
        // Set module loaded indicator
        bLoaded = true;
    }
    return bLoaded;

} // isLoaded

//******************************************************************************

bool
CModuleInstance::hasSymbols() const
{
    bool                bSymbols = false;

    // Module must be loaded to have symbols
    if (isLoaded())
    {
        // Make sure module symbols are loaded
        loadSymbols(FORCE_SYMBOL_LOAD);

        // Check to see if the module has symbols
        if (symType() != SymNone)
        {
            // Indicate module has symbols
            bSymbols = true;
        }
    }
    return bSymbols;

} // hasSymbols

//******************************************************************************

HRESULT
CModuleInstance::loadInformation() const
{
    CString             sReload;
    CString             sOptions(MAX_COMMAND_STRING);
    HRESULT             hResult = S_OK;

    // Only load module information if not cached
    if (!isCached())
    {
        // Try to get the module information
        hResult = GetModuleByModuleName(name(), 0, &m_ulIndex, &m_ulAddress);
        if (!SUCCEEDED(hResult))
        {
            // Build command to reload module information (Requires full module name)
            // (Don't use Reload API as that will display errors if module not loaded)
            sOptions.sprintf("%s", fullName());
            sReload = buildDotCommand("reload", sOptions);

            // Try issuing reload command to sync engine module list with the target
            hResult = Execute(DEBUG_OUTCTL_THIS_CLIENT | DEBUG_OUTCTL_NOT_LOGGED, sReload, DEBUG_EXELWTE_NOT_LOGGED | DEBUG_EXELWTE_NO_REPEAT);
            if (SUCCEEDED(hResult))
            {
                // Try to get module information one more time
                hResult = GetModuleByModuleName(name(), 0, &m_ulIndex, &m_ulAddress);
                if (!SUCCEEDED(hResult))
                {
                    // Set the module index and address (Cached but not loaded)
                    m_ulIndex   = NOT_PRESENT_INDEX;
                    m_ulAddress = NOT_PRESENT_ADDRESS;
                }
            }
            else    // Failed to reload module information
            {
                // Set the module index and address (Cached but not loaded)
                m_ulIndex   = NOT_PRESENT_INDEX;
                m_ulAddress = NOT_PRESENT_ADDRESS;
            }
        }
    }
    return hResult;

} // loadInformation

//******************************************************************************

HRESULT
CModuleInstance::resetInformation() const
{
    HRESULT             hResult = S_OK;

    // Only reset module information if cached
    if (isCached())
    {
        // Check to see if this module is loaded
        if (isLoaded())
        {
            // Check to see if symbols are loaded (Need to be unloaded)
            if (symType() != SymNone)
            {
                // Unload the module symbols
                hResult = unloadSymbols();
            }
        }
        // Reset module index and address to uncached values
        m_ulIndex   = UNCACHED_INDEX;
        m_ulAddress = UNCACHED_ADDRESS;

        // Initialize the imagehlp module structure
        memset(&m_ImageHlpModule, 0, sizeof(m_ImageHlpModule));
        m_ImageHlpModule.SizeOfStruct = sizeof(m_ImageHlpModule);
    }
    return hResult;

} // resetInformation

//******************************************************************************

HRESULT
CModuleInstance::loadSymbols
(
    bool                bForce
) const
{
    DWORD               dwSymOpt = 0;
    HRESULT             hResult = S_OK;

    // Only try to load module symbols if module loaded
    if (isLoaded())
    {
        // Check to see if module symbols not loaded or forced defer
        if ((symType() == SymNone) || ((symType() == SymDeferred) && bForce))
        {
            // Check for forced symbol load (Need to update symopts if so)
            if (bForce)
            {
                // Get the current symbol options
                dwSymOpt = SymGetOptions();

                // Turn off the deferred symbol loads
                SymSetOptions(dwSymOpt & ~SYMOPT_DEFERRED_LOADS);
            }
            // Save the effective processor mode (In case we have to change it for symbol load)
            CEffectiveProcessor EffectiveProcessor;

            // If kernel module make sure we are in the proper processor mode
            if (isKernelModule())
            {
                SetEffectiveProcessorType(actualMachine());
            }
            // Try to load the symbols for the given module
            if (SymLoadModuleEx(symbolHandle(), NULL, name(), name(), address(), 0, NULL, 0))
            {
                // Update the module information
                SymGetModuleInfo64(symbolHandle(), address(), &m_ImageHlpModule);
            }
            else    // Unable to load module symbols
            {
                // Get the error result
                hResult = GetLastError();
                if (!SUCCEEDED(hResult))
                {
                    // Display warning to the user
                    dprintf("Unable to load symbols for module '%s' (0x%08x)!\n", name(), hResult);
                }
            }
            // Check for forced symbol load (Need to restore symopts if so)
            if (bForce)
            {
                // Restore the original symbol options
                SymSetOptions(dwSymOpt);
            }
        }
        // Check for deferred symbols (Need to update module information)
        if (symType() == SymDeferred)
        {
            // Update module information to see if symbols are now loaded
            SymGetModuleInfo64(symbolHandle(), address(), &m_ImageHlpModule);
        }
    }
    return hResult;

} // loadSymbols

//******************************************************************************

HRESULT
CModuleInstance::unloadSymbols() const
{
    HRESULT             hResult = S_OK;

    // Only try to unload module symbols if module loaded
    if (isLoaded())
    {
        // Check to see if module symbols loaded
        if (symType() != SymNone)
        {
            // Try to unload the symbols for the given module
            if (SymUnloadModule64(symbolHandle(), address()))
            {
                // Reset the module information
                memset(&m_ImageHlpModule, 0, sizeof(m_ImageHlpModule));
                m_ImageHlpModule.SizeOfStruct = sizeof(m_ImageHlpModule);
            }
            else    // Unable to unload module symbols
            {
                // Get the error result
                hResult = GetLastError();
                if (!SUCCEEDED(hResult))
                {
                    // Display warning to the user
                    dprintf("Unable to unload symbols for module '%s' (0x%08x)!\n", name(), hResult);
                }
            }
        }
    }
    return hResult;

} // unloadModuleSymbols

//******************************************************************************

HRESULT
CModuleInstance::reloadSymbols
(
    bool                bForce
) const
{
    HRESULT             hResult;

    // First attempt to unload the symbols for the module
    hResult = unloadSymbols();
    if (SUCCEEDED(hResult))
    {
        // Now try to load the symbols for the module
        hResult = loadSymbols(bForce);
    }
    return hResult;

} // reloadSymbols

//******************************************************************************

const CModule*
findModule
(
    ULONG64             ulAddress
)
{
    const CModule      *pModule = firstModule();

    // Loop looking for the requested module
    while (pModule != NULL)
    {
        // Check to see if this is the requested module
        if (pModule->address() == ulAddress)
        {
            // Found the requested module, stop the search
            break;
        }
        else    // Not the requested module
        {
            // Move to the next module in the list
            pModule = pModule->nextModule();
        }
    }
    return pModule;

} // findModule

//******************************************************************************

const CModule*
findModule
(
    CString             sModule
)
{
    const CModule      *pModule = firstModule();

    // Colwert module name to lower case
    sModule.lower();

    // Loop looking for the given module name
    while (pModule != NULL)
    {
        // Check for matching module name
        if (strcmp(sModule, pModule->name()) == 0)
        {
            // Found matching module, exit the search
            break;
        }
        // Check for matching full module name
        if (strcmp(sModule, pModule->fullName()) == 0)
        {
            // Found matching module, exit the search
            break;
        }
        // Move to the next module
        pModule = pModule->nextModule();
    }
    return pModule;

} // findModule

//******************************************************************************

const CModule*
findKernelModule
(
    ULONG64             ulAddress
)
{
    const CModule      *pModule = firstKernelModule();

    // Loop looking for the requested module
    while (pModule != NULL)
    {
        // Check to see if this is the requested module
        if (pModule->address() == ulAddress)
        {
            // Found the requested module, stop the search
            break;
        }
        else    // Not the requested module
        {
            // Move to the next kernel module in the list
            pModule = pModule->nextKernelModule();
        }
    }
    return pModule;

} // findKernelModule

//******************************************************************************

const CModule*
findKernelModule
(
    CString             sModule
)
{
    const CModule      *pModule = firstKernelModule();

    // Colwert module name to lower case
    sModule.lower();

    // Loop looking for the given module name
    while (pModule != NULL)
    {
        // Check for matching module name
        if (strcmp(sModule, pModule->name()) == 0)
        {
            // Found matching module, exit the search
            break;
        }
        // Check for matching full module name
        if (strcmp(sModule, pModule->fullName()) == 0)
        {
            // Found matching module, exit the search
            break;
        }
        // Move to the next kernel module
        pModule = pModule->nextKernelModule();
    }
    return pModule;

} // findKernelModule

//******************************************************************************

const CModule*
findUserModule
(
    ULONG64             ulAddress
)
{
    const CModule      *pModule = firstUserModule();

    // Loop looking for the requested module
    while (pModule != NULL)
    {
        // Check to see if this is the requested module
        if (pModule->address() == ulAddress)
        {
            // Found the requested module, stop the search
            break;
        }
        else    // Not the requested module
        {
            // Move to the next user module in the list
            pModule = pModule->nextUserModule();
        }
    }
    return pModule;

} // findUserModule

//******************************************************************************

const CModule*
findUserModule
(
    CString             sModule
)
{
    const CModule      *pModule = firstUserModule();

    // Colwert module name to lower case
    sModule.lower();

    // Loop looking for the given module name
    while (pModule != NULL)
    {
        // Check for matching module name
        if (strcmp(sModule, pModule->name()) == 0)
        {
            // Found matching module, exit the search
            break;
        }
        // Check for matching full module name
        if (strcmp(sModule, pModule->fullName()) == 0)
        {
            // Found matching module, exit the search
            break;
        }
        // Move to the next user module
        pModule = pModule->nextUserModule();
    }
    return pModule;

} // findKernelModule

//******************************************************************************

void
loadModuleInformation()
{
    const CModule      *pModule;

    // Loop getting information for all modules
    pModule = firstModule();
    while (pModule != NULL)
    {
        // Load the next module information
        pModule->loadInformation();

        // Move to the next module
        pModule = pModule->nextModule();
    }

} // loadModuleInformation

//******************************************************************************

void
resetModuleInformation()
{
    const CModule      *pModule;

    // Loop resetting information for all modules
    pModule = firstModule();
    while (pModule != NULL)
    {
        // Reset the module information
        pModule->resetInformation();

        // Move to the next module
        pModule = pModule->nextModule();
    }

} // resetModuleInformation

//******************************************************************************

HRESULT
loadModuleSymbols
(
    bool                bForce
)
{
    const CModule      *pModule;
    HRESULT             hTemp;
    HRESULT             hResult = S_OK;

    // Make sure all kernel modules are loaded in the proper mode (Actual machine type)
    CEffectiveProcessor EffectiveProcessor(actualMachine());

    // Loop loading all the modules
    pModule = firstModule();
    while (pModule != NULL)
    {
        // Try to load the symbols for this module
        hTemp = pModule->loadSymbols(bForce);
        if (!SUCCEEDED(hTemp))
        {
            // Update the result code (At least one failure)
            hResult = hTemp;
        }
        // Move to the next module
        pModule = pModule->nextModule();
    }
    return hResult;

} // loadModuleSymbols

//******************************************************************************

HRESULT
unloadModuleSymbols()
{
    const CModule      *pModule;
    HRESULT             hTemp;
    HRESULT             hResult = S_OK;

    // Loop unloading all the modules
    pModule = firstModule();
    while (pModule != NULL)
    {
        // Try to unload the symbols for this module
        hTemp = pModule->unloadSymbols();
        if (!SUCCEEDED(hTemp))
        {
            // Update the result code (At least one failure)
            hResult = hTemp;
        }
        // Move to the next module
        pModule = pModule->nextModule();
    }
    return hResult;

} // unloadModuleSymbols

//******************************************************************************

HRESULT
reloadModuleSymbols
(
    bool                bForce
)
{
    const CModule      *pModule;
    HRESULT             hTemp;
    HRESULT             hResult = S_OK;

    // Loop reloading all the modules
    pModule = firstModule();
    while (pModule != NULL)
    {
        // First try to unload the symbols for this module
        hTemp = pModule->unloadSymbols();
        if (SUCCEEDED(hTemp))
        {
            // Try to load the symbols for this module
            hTemp = pModule->loadSymbols(bForce);
            if (!SUCCEEDED(hTemp))
            {
                // Update the result code (At least one failure)
                hResult = hTemp;
            }
        }
        else    // Failed to unload module symbols
        {
            // Update the result code (At least one failure)
            hResult = hTemp;
        }
        // Move to the next module
        pModule = pModule->nextModule();
    }
    return hResult;

} // reloadModuleSymbols

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
