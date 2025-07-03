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
|*  Module: symsession.cpp                                                    *|
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
// Locals
//
//******************************************************************************
static  CSessionHook    s_sessionHook;

// Current session
static  CSymbolSessionPtr s_pLwrrentSession;

// CSession object tracking
CSessionList            CSymbolSession::m_SessionList;

//******************************************************************************

CSymbolSession::CSymbolSession
(
    CSessionList       *pSessionList,
    ULONG               ulSession
)
:   LWnqObj(pSessionList, ulSession),
    m_ulSession(ulSession),
    m_aModules(NULL)
{
    const CModule      *pKernel;

    assert(pSessionList != NULL);

    // Try to allocate module instances for this session
    m_aModules = new CModulePtr[kernelModuleCount()];

    // Loop initializing all the module instances (Kernel)
    pKernel = firstKernelModule();
    while (pKernel != NULL)
    {
        // Validate and create the next module instance
        assert(pKernel->instance() < kernelModuleCount());

        m_aModules[pKernel->instance()] = new CModuleInstance(pKernel, this);

        // Move to the next kernel module
        pKernel = pKernel->nextKernelModule();
    }

} // CSymbolSession

//******************************************************************************

CSymbolSession::~CSymbolSession()
{
    CSymbolProcess     *pProcess = NULL;

    // Clear the current process
    m_pLwrrentProcess = NULL;

    // Check for processes present
    if (processCount() != 0)
    {
        // Loop destroying all the processes
        pProcess = m_ProcessList.firstObject();
        while (pProcess != NULL)
        {
            // Destroy the current process
            destroyProcess(pProcess->ptrProcess());

            // Get the new first process (None if none left)
            pProcess = m_ProcessList.firstObject();
        }
    }

} // ~CSymbolSession

//******************************************************************************

const CModuleInstance*
CSymbolSession::module
(
    ULONG               ulInstance
) const
{
    // Check for a valid module index
    if (ulInstance >= kernelModuleCount())
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid module instance (%d >= %d) for session %d",
                               ulInstance, kernelModuleCount(), session());
    }
    // Return the requested module instance
    return m_aModules[ulInstance];

} // module

//******************************************************************************

CSymbolSessionPtr
CSymbolSession::createSession
(
    ULONG               ulSession
)
{
    CSymbolSessionPtr   pSession;

    // Check to see if this session already exists
    pSession = findObject(sessionList(), ulSession);
    if (pSession == NULL)
    {
        // Try to create the new session object
        pSession = new CSymbolSession(sessionList(), ulSession);
    }
    return pSession;

} // createSession

//******************************************************************************

CSymbolProcessPtr
CSymbolSession::createProcess
(
    POINTER             ptrProcess
)
{
    CSymbolProcessPtr   pProcess;

    // Check for valid process address given
    if (ptrProcess != NULL)
    {
        // Check to see if this process already exists
        pProcess = findProcess(ptrProcess);
        if (pProcess == NULL)
        {
            // Create the new session process
            pProcess = CSymbolProcess::createProcess(this, ptrProcess);
            if (pProcess != NULL)
            {
                // Acquire a reference to this process (create/hold)
                pProcess->acquire();
            }
        }
    }
    return pProcess;

} // createProcess

//******************************************************************************

void
CSymbolSession::destroyProcess
(
    POINTER             ptrProcess
)
{
    CSymbolProcessPtr   pProcess;

    // Check to see if this process exists
    pProcess = findProcess(ptrProcess);
    if (pProcess != NULL)
    {
        // Check for destroying the current process
        if (pProcess == m_pLwrrentProcess)
        {
            // Set current process to none
            m_pLwrrentProcess = NULL;
        }
        // Release the reference to this process (destroy/release)
        pProcess->release();
        pProcess = NULL;
    }

} // destroyProcess

//******************************************************************************

CSymbolSessionPtr
CSymbolSession::findSession
(
    ULONG               ulSession
)
{
    CSymbolSessionPtr   pSession;

    // Try to find the requested session
    pSession = findObject(sessionList(), ulSession);

    return pSession;

} // findSession

//******************************************************************************

CSymbolProcessPtr
CSymbolSession::findProcess
(
    POINTER             ptrProcess
)
{
    CSymbolProcessPtr   pProcess;

    // Try to find the requested process
    pProcess = CSymbolProcess::findObject(processList(), ptrProcess);

    return pProcess;

} // findProcess

//******************************************************************************

void
CSymbolSession::setLwrrentProcess
(
    CSymbolProcessPtr   pProcess
)
{
    // Check to make sure this session is the current session
    if (this == s_pLwrrentSession)
    {
        // Check to see if this process is not already the current process
        if (pProcess != m_pLwrrentProcess)
        {
            // Set the new current process
            m_pLwrrentProcess = pProcess;

#pragma message("  Actually switch processes here")

        }
    }
    else    // This is not the current session
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to set current process (0x%*I64x) for non-current session (%d != %d)",
                               PTR(pProcess->ptrProcess()), session(), (s_pLwrrentSession != NULL) ? s_pLwrrentSession->session() : -1);
    }

} // setLwrrentProcess

//******************************************************************************

CSymbolSessionPtr
createSession
(
    ULONG               ulSession
)
{
    CSymbolSessionPtr   pSession;

    // Check to see if this session already exists
    pSession = CSymbolSession::findSession(ulSession);
    if (pSession == NULL)
    {
        // Try to create the requested session
        pSession = CSymbolSession::createSession(ulSession);
        if (pSession != NULL)
        {
            // Acquire a reference to this session (create/hold)
            pSession->acquire();
        }
    }
    return pSession;

} // createSession

//******************************************************************************

void
destroySession
(
    ULONG               ulSession
)
{
    CSymbolSessionPtr   pSession;

    // Check to see if this session exists
    pSession = CSymbolSession::findSession(ulSession);
    if (pSession != NULL)
    {
        // Release the reference to this session (destroy/release)
        pSession->release();
        pSession = NULL;
    }

} // destroySession

//******************************************************************************

CSymbolSessionPtr
getLwrrentSession()
{
    // Return the current session
    return s_pLwrrentSession;

} // getLwrrentSession

//******************************************************************************

void
setLwrrentSession
(
    CSymbolSessionPtr   pSession
)
{
    // Check to see if this session is not already the current session
    if (pSession != s_pLwrrentSession)
    {
        // Set the new current session
        s_pLwrrentSession = pSession;

#pragma message("  Actually switch sessions here")

    }

} // setLwrrentSession

//******************************************************************************

HRESULT
CSessionHook::initialize
(
    const PULONG        pVersion,
    const PULONG        pFlags
)
{
    UNREFERENCED_PARAMETER(pVersion);
    UNREFERENCED_PARAMETER(pFlags);

    // Nothing to do on initialization
    return S_OK;

} // initialize

//******************************************************************************

void
CSessionHook::notify
(
    ULONG               Notify,
    ULONG64             Argument
)
{
    UNREFERENCED_PARAMETER(Notify);
    UNREFERENCED_PARAMETER(Argument);






} // notify

//******************************************************************************

void
CSessionHook::uninitialize(void)
{
    const CSymbolSession *pSession = NULL;

    // Clear the current session
    s_pLwrrentSession = NULL;

    // Check for sessions present
    if (CSymbolSession::sessionCount() != 0)
    {
        // Loop destroying all the sessions
        pSession = pSession->firstSession();
        while (pSession != NULL)
        {
            // Destroy the current session
            destroySession(pSession->session());

            // Get the new first session (None if none left)
            pSession = pSession->firstSession();
        }
    }

} // uninitialize

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
