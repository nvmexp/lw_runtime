/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "sec2.h"

#include "g_sec2_private.h"

#include "volta/gv100/dev_sec_pri.h"
#include "volta/gv100/dev_master.h"
#include "sec2/sec2ifcmn.h"

static LW_STATUS
_sec2MutexIdGen_GV100(LwU32 *pOwnerId);

static void
_sec2MutexIdRel_GV100(LwU32 ownerId);


LW_STATUS
sec2LockEmem_GV100(LwU32 *pMutexId)
{
    LW_STATUS status    = LW_OK;
    LwU32     timeoutUs = SEC2_MUTEX_TIMEOUT_US;

    // Acquire EMEM lock
    do
    {
        status = pSec2[indexGpu].sec2AcquireMutexByIndex(SEC2_MUTEX_EMEM, pMutexId);
        if (status == LW_OK)
        {
            break;
        }
        else
        {
            dprintf("lw: %s: error in acquiring EMEM lock (non-timeout)\n", __FUNCTION__);
            return status;
        }
        osPerfDelay(0x10);
        timeoutUs -= 0x10;
    } while (timeoutUs > 0);

    if (status == LW_ERR_STATE_IN_USE)
    {
        dprintf("lw: %s: timeout in acquiring EMEM lock\n", __FUNCTION__);
    }

    return status;
}


LW_STATUS
sec2UnlockEmem_GV100(LwU32 mutexId)
{
    pSec2[indexGpu].sec2ReleaseMutexByIndex(SEC2_MUTEX_EMEM, mutexId);
    return LW_OK;
}

/*!
 * Generate a unique identifier that may be used for locking the SEC2's HW
 * mutexes.
 *
 * @param[out]  pOwnerId  Pointer to write with the generated owner identifier
 *
 * @return 'LW_OK' if a owner identifier was successfully generated.
 */
static LW_STATUS
_sec2MutexIdGen_GV100
(
    LwU32    *pOwnerId
)
{
    LwU32      reg32;
    LW_STATUS  status = LW_OK;

    //
    // Generate a owner ID by reading the MUTEX_ID register (this register
    // has a read side-effect; avoid unnecessary reads).
    //
    reg32 = GPU_REG_RD32(LW_PSEC_MUTEX_ID);

    //
    // Hardware will return _NOT_AVAIL if all identifiers have been used/
    // consumed. Also check against _INIT since zero is not a valid identifier
    // either (zero is used to release mutexes so it cannot be used as an ID).
    //
    if ((!FLD_TEST_DRF(_PSEC, _MUTEX_ID, _VALUE, _INIT     , reg32)) &&
        (!FLD_TEST_DRF(_PSEC, _MUTEX_ID, _VALUE, _NOT_AVAIL, reg32)))
    {
        *pOwnerId = DRF_VAL(_PSEC, _MUTEX_ID, _VALUE, reg32);
    }
    else
    {
        status = LW_ERR_INSUFFICIENT_RESOURCES;
        //
        // A common failure for power-management features is to have BAR0
        // return all f's for all accesses. When this happens, this function
        // tends to fail first, leading to false bugs filed.
        // Check against that here by looking at the entire return value (and
        // not just the eight-bit mutex-ID field).
        //
        if (reg32 == 0xFFFFFFFF)
        {
            dprintf("lw: The SEC2 mutex ID generator returned "
                    "0xFFFFFFFF suggesting there may be an error with "
                    "BAR0. Verify BAR0 is functional before filing a "
                    "bug.\n");
        }
        // otherwise, there is a real leak with the mutex identifiers
        else
        {
            dprintf("lw: Failed to generate a mutex identifier. "
                     "Hardware indicates that all identifiers have been "
                     "consumed.\n");
        }
    }
    return status;
}

/*!
 * Release the given owner identifier thus making it available to other
 * clients.
 *
 * @param[in]  ownerId  The owner identifier to release
 */
static void
_sec2MutexIdRel_GV100
(
    LwU32     ownerId
)
{
    GPU_FLD_WR_DRF_NUM(_PSEC, _MUTEX_ID_RELEASE, _VALUE, ownerId);
    return;
}

/*!
 * Attempts to acquire the SEC2 mutex as specified by the given physical
 * mutex-index.
 *
 * @param[in]   physMutexId  The physical mutex-index for the mutex to acquire
 * @param[out]  pOwnerId     Pointer to the ID to write with the ID generated
 *                           upon a successful lock of the mutex.  This value
 *                           will remain unchanged upon failure.
 *
 * @return 'LW_OK'               if the mutex was successfully acquired
 *         'LW_ERR_STATE_IN_USE' if mutex was in use
 *
 */
LW_STATUS
sec2AcquireMutexByIndex_GV100
(
    LwU32     physMutexId,
    LwU32    *pOwnerId
)
{
    LwU32      ownerId;
    LwU32      value;
    LW_STATUS  status;

    // generate a unique mutex identifier
    status = _sec2MutexIdGen_GV100(&ownerId);
    if (status != LW_OK)
    {
        dprintf("lw: error generating a mutex identifer.\n");
        return status;
    }

    //
    // Write the ID into the mutex register to attempt an "acquire"
    // of the mutex.
    //
    GPU_REG_IDX_WR_DRF_NUM(_PSEC, _MUTEX, physMutexId, _VALUE, ownerId);

    //
    // Read the register back to see if the ID stuck.  If the value
    // read back matches the ID written, the mutex was successfully
    // acquired.  Otherwise, release the ID and return an error.
    //
    value = GPU_REG_IDX_RD_DRF(_PSEC, _MUTEX, physMutexId, _VALUE);
    if (value == ownerId)
    {
        *pOwnerId = ownerId;
    }
    else
    {
        dprintf("lw: Cannot acquire mutex index %d (owned by %d).\n",
                physMutexId, value);

        _sec2MutexIdRel_GV100(ownerId);
        status = LW_ERR_STATE_IN_USE;
    }
    return status;
}

/*!
 * Attempts to release the SEC2 mutex as specified by the given physical
 * mutex-index.  It is the caller's responsibility to ensure that the mutex was
 * acquired before calling this function.  This function simply performs a
 * "release" operation on the given mutex, and frees the owner-ID.
 *
 * @param[in]  physMutexId  The physical mutex-index for the mutex to release
 * @param[in]  ownerId      The ID returned when the mutex was initially
 *                          acquired.
 *
 * @return 'LW_OK' if the mutex and owner ID were successfully released.
 */
void
sec2ReleaseMutexByIndex_GV100
(
    LwU32     physMutexId,
    LwU32     ownerId
)
{
    GPU_REG_IDX_WR_DRF_DEF(_PSEC, _MUTEX, physMutexId, _VALUE, _INITIAL_LOCK);

    // release the mutex identifer
    _sec2MutexIdRel_GV100(ownerId);
}
