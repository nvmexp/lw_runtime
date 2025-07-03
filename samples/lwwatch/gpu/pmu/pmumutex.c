/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   pmumutex.c
 * @brief  Provides all the fundamental logic and interfaces (public and
 *         private) for acquiring/releasing PMU mutexes.
 *
 * The PMU PWR block provides an array of mutex registers to help facilitate
 * software coordination between the PMU RTOS application and the RM. This
 * module provides the basic logic required to manage each mutex register as a
 * discrete mutex object. Two public interfaces are provided: one for acquiring
 * a mutex and another for releasing a mutex. Callers must refer to each mutex
 * using the mutex's logical mutex identifer.  'rmpmucmdif.h' provides an
 * enumeration containing all valid logical identifiers.  Each logical
 * identifer will be internally mapped to a corresponding physical mutex
 * register index.  This component is responsible for maintain this
 * logical-to-physical mapping.
 *
 * The following properties are supported by this mutex implementation:
 *    - sense of ownership
 *    - reentrant - can be acquired multiple times by the same owner/thread
 *    - must be released the same number of times as acquired
 */

#include "pmu.h"

/************************** PUBLIC PMU INTERFACES ****************************/

/*!
 * Attempt to acquire a PMU mutex.
 *
 * @param[in]      mutexIndex  The logical identifier for the mutex to lock
 * @param[in]      retryCount  
 *     The number of retry attempts to make if the mutex is initially
 *     unavailable. This number does not include the initial attempt.
 *
 * @param[in,out]  pToken      Pointer to token/id used to lock the mutex
 *
 * @return 'LW_OK'
 *    If the mutex was successfully acquired
 *
 * @return 'LW_ERR_ILWALID_ARGUMENT'
 *    If the token is NULL or if the mutex-id is invalid
 *
 * @return 'LW_ERR_STATE_IN_USE'
 *    If the mutex is lwrrently acquired by another client (and not the
 *    requestor) and non-blocking behavior was requested.
 *
 * @return 'LW_ERR_GENERIC'
 *    If the mutex does not exist, or if we could not determine the mutex
 *    owner and thus did not acquire the mutex.
 *
 * @sa pmuReleaseMutex
 */
LW_STATUS
pmuAcquireMutex
(
    LwU32  mutexIndex,
    LwU16  retryCount,
    LwU32 *pToken
)
{
    LW_STATUS  status = LW_OK;
    LwU32      owner;

    // 'pToken' may never be NULL
    if (pToken == NULL)
    {
        dprintf("LWRM: %s: Mutex acquire-token may never be NULL.\n",
                __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    // query to determine the current owner of the mutex
    status = pPmu[indexGpu].pmuMutex_QueryOwnerByIndex(mutexIndex, &owner);
    if (status != LW_OK)
    {
        dprintf("LWRM: %s: Cannot determine mutex owner (status=0x%x).\n",
                __FUNCTION__, status);
        return LW_ERR_GENERIC;
    }

    // attempt to acquire the mutex
    for (;;)
    {
        status = pPmu[indexGpu].pmuMutex_AcquireByIndex(mutexIndex, pToken);
        if (status != LW_ERR_STATE_IN_USE)
        {
            break;
        }
        if (retryCount-- == 0)
        {
            status = LW_ERR_STATE_IN_USE;
            break;
        }
    }
    return status;
}


/*!
 * Release a previously acquired PMU mutex.
 *
 * @param[in]      mutexIndex  The logical identifier for the mutex to release
 * @param[in,out]  pToken
 *     Pointer to the token which is set to INVALID once we see the reference
 *     count is zero as a result of a release operation
 *
 * @return
 *     'LW_OK' upon successful release of the mutex (this includes the case
 *     where the reference count was decremented but remains non-zero).
 *
 * @return
 *     'LW_ERR_GENERIC' if the provided mutex identifier does not match the id of
 *     the mutex's current owner (if owned).
 *
 * @return
 *     'LW_ERR_GENERIC' if the mutex is lwrrently unowned
 *
 * @return
 *     'LW_ERR_GENERIC' Other unexpected failures returned from the HAL-layer.
 *
 * @sa pmuAcquireMutex
 */
LW_STATUS
pmuReleaseMutex
(
    LwU32  mutexIndex,
    LwU32 *pToken
)
{
    LW_STATUS status;
    LwU32     owner;

    // 'pToken' may never be NULL
    if (pToken == NULL)
    {
        dprintf("LWRM: %s: Mutex acquire-token may never be NULL.\n",
                __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    // query to determine the current owner of the mutex
    status = pPmu[indexGpu].pmuMutex_QueryOwnerByIndex(mutexIndex, &owner);
    if (status != LW_OK)
    {
        dprintf("LWRM: %s: Cannot determine mutex owner (status=0x%x).\n",
                __FUNCTION__, status);
        return LW_ERR_GENERIC;
    }

    //
    // Never allow the RM to release a mutex that it does not own or that it
    // was not asked to release.
    //
    if (*pToken != owner)
    {
        dprintf("LWRM: %s: Attempt made to release a mutex that the "
                "does not own.\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    status = pPmu[indexGpu].pmuMutex_ReleaseByIndex(mutexIndex, *pToken);
    if (status == LW_OK)
    {
        *pToken = PMU_ILWALID_MUTEX_OWNER_ID;
    }
    else
    {
        dprintf("LWRM: %s: Unexpected error oclwrred while releasing"
                " PMU mutex.  (owner-id=0x%x, status=0x%x).\n",
                __FUNCTION__, *pToken, status);
    }
    return LW_OK;
}

