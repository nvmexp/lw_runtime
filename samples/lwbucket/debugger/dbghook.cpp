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
|*  Module: dbghook.cpp                                                       *|
|*                                                                            *|
 \****************************************************************************/
#include "dbgprecomp.h"

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
//  Forwards
//
//******************************************************************************
static CHook*           firstHook();                // First hook entry

//******************************************************************************
//
//  Locals
//
//******************************************************************************
CHook*  CHook::m_pFirstHook  = NULL;                // First hook entry
CHook*  CHook::m_pLastHook   = NULL;                // Last hook entry
ULONG   CHook::m_ulHookCount = 0;                   // Hook count value

//******************************************************************************

CHook::CHook()
:   m_pPrevHook(NULL),
    m_pNextHook(NULL)
{
    // Add this hook to the hook list
    addHook(this);

} // CHook

//******************************************************************************

CHook::~CHook()
{
    // Remove this hook from the hook list
    removeHook(this);

} // ~CHook

//******************************************************************************

void
CHook::addHook
(
    CHook              *pHook
)
{
    assert(pHook != NULL);

    // Check for first hook
    if (m_pFirstHook == NULL)
    {
        // Set first and last hook to this hook
        m_pFirstHook = pHook;
        m_pLastHook  = pHook;
    }
    else    // Adding new hook to hook list
    {
        // Add this hook to the end of the hook list
        pHook->m_pPrevHook = m_pLastHook;
        pHook->m_pNextHook = NULL;

        m_pLastHook->m_pNextHook = pHook;

        m_pLastHook = pHook;
    }
    // Increment the hook count
    m_ulHookCount++;

} // addHook

//******************************************************************************

void
CHook::removeHook
(
    CHook              *pHook
)
{
    assert(pHook != NULL);

    // Remove this hook from the hook list
    if (pHook->m_pPrevHook != NULL)
    {
        pHook->m_pPrevHook->m_pNextHook = pHook->m_pNextHook;
    }
    if (pHook->m_pNextHook != NULL)
    {
        pHook->m_pNextHook->m_pPrevHook = pHook->m_pPrevHook;
    }
    // Check for first hook
    if (m_pFirstHook == pHook)
    {
        // Update first hook
        m_pFirstHook = pHook->m_pNextHook;
    }
    // Check for last hook
    if (m_pLastHook == pHook)
    {
        // Update last hook
        m_pLastHook = pHook->m_pPrevHook;
    }
    // Decrement the hook count
    m_ulHookCount--;

} // removeHook

//******************************************************************************

HRESULT
CHook::initialize
(
    const PULONG        pVersion,
    const PULONG        pFlags
)
{
    UNREFERENCED_PARAMETER(pVersion);
    UNREFERENCED_PARAMETER(pFlags);

    HRESULT             hResult = S_OK;

    assert(pVersion != NULL);
    assert(pFlags != NULL);

    return hResult;

} // initialize

//******************************************************************************

void
CHook::notify
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
CHook::uninitialize(void)
{

} // uninitialize

//******************************************************************************

HRESULT
callInitializeHooks
(
    const PULONG        pVersion,
    const PULONG        pFlags
)
{
    CHook              *pHook = firstHook();
    HRESULT             hStatus;
    HRESULT             hResult = S_OK;

    assert(pVersion != NULL);
    assert(pFlags != NULL);

    // Loop calling initialize hooks
    while (pHook != NULL)
    {
        // Don't let exceptions in one hook affect another
        try
        {
            // Call the next initialize hook
            hStatus = pHook->initialize(pVersion, pFlags);
            if (FAILED(hStatus))
            {
                // Save failed result value
                hResult = hStatus;
            }
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing hooks)
            exception.dPrintf();
        }
        // Move to the next hook
        pHook = pHook->nextHook();
    }
    return hResult;

} // callInitializeHooks

//******************************************************************************

void
callNotifyHooks
(
    ULONG               Notify,
    ULONG64             Argument
)
{
    CHook              *pHook = firstHook();

    // Loop calling notify hooks
    while (pHook != NULL)
    {
        // Don't let exceptions in one hook affect another
        try
        {
            // Call the next notify hook
            pHook->notify(Notify, Argument);
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing hooks)
            exception.dPrintf();
        }
        // Move to the next hook
        pHook = pHook->nextHook();
    }

} // callNotifyHooks

//******************************************************************************

void
callUninitializeHooks(void)
{
    CHook              *pHook = firstHook();

    // Loop calling uninitialize hooks
    while (pHook != NULL)
    {
        // Don't let exceptions in one hook affect another
        try
        {
            // Call the next uninitialize hook
            pHook->uninitialize();
        }
        catch (CException& exception)
        {
            // Display exception message (But continue processing hooks)
            exception.dPrintf();
        }
        // Move to the next hook
        pHook = pHook->nextHook();
    }

} // callUninitializeHooks

//******************************************************************************

static CHook*
firstHook()
{
    // Return the first hook routine
    return CHook::firstHook();

} // firstHook

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
