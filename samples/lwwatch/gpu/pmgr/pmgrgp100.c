/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "pmgr.h"

#include "g_pmgr_private.h"

#include "pascal/gp100/dev_pmgr.h"

static LW_STATUS
_pmgrMutexIdGen_GP100(LwU32 *pOwnerId);

static void
_pmgrMutexIdRel_GP100(LwU32 ownerId);


/*!
 * Generate a unique identifier that may be used for locking the PMGR's HW
 * mutexes.
 *
 * @param[out]  pOwnerId  Pointer to write with the generated owner identifier
 *
 * @return 'LW_OK' if a owner identifier was successfully generated.
 */
static LW_STATUS
_pmgrMutexIdGen_GP100
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
    reg32 = GPU_REG_RD32(LW_PMGR_MUTEX_ID_ACQUIRE);

    //
    // Hardware will return _NOT_AVAIL if all identifiers have been used/
    // consumed. Also check against _INIT since zero is not a valid identifier
    // either (zero is used to release mutexes so it cannot be used as an ID).
    //
    if ((!FLD_TEST_DRF(_PMGR, _MUTEX_ID_ACQUIRE, _VALUE, _INIT, reg32)) &&
        (!FLD_TEST_DRF(_PMGR, _MUTEX_ID_ACQUIRE, _VALUE, _NOT_AVAIL, reg32)))
    {
        *pOwnerId = DRF_VAL(_PMGR, _MUTEX_ID_ACQUIRE, _VALUE, reg32);
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
            dprintf("lw: %s: The PMGR mutex ID generator returned "
                    "0xFFFFFFFF suggesting there may be an error with "
                    "BAR0. Verify BAR0 is functional before filing a "
                    "bug.\n", __FUNCTION__);
        }
        // otherwise, there is a real leak with the mutex identifiers
        else
        {
            dprintf("lw: %s: Failed to generate a mutex identifier. "
                     "Hardware indicates that all identifiers have been "
                     "consumed.\n", __FUNCTION__);
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
_pmgrMutexIdRel_GP100
(
    LwU32     ownerId
)
{
    GPU_FLD_WR_DRF_NUM(_PMGR, _MUTEX_ID_RELEASE, _VALUE, ownerId);
    return;
}

/*!
 * Attempts to acquire the PMGR mutex as specified by the given physical
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
pmgrMutexAcquireByIndex_GP100
(
    LwU32     physMutexId,
    LwU32    *pOwnerId
)
{
    LwU32      ownerId;
    LwU32      value;
    LW_STATUS  status;

    // generate a unique mutex identifier
    status = _pmgrMutexIdGen_GP100(&ownerId);
    if (status != LW_OK)
    {
        dprintf("lw: %s: error generating a mutex identifer.\n", __FUNCTION__);
        return status;
    }

    //
    // Write the ID into the mutex register to attempt an "acquire"
    // of the mutex.
    //
    GPU_REG_IDX_WR_DRF_NUM(_PMGR, _MUTEX_REG, physMutexId, _VALUE, ownerId);

    //
    // Read the register back to see if the ID stuck.  If the value
    // read back matches the ID written, the mutex was successfully
    // acquired.  Otherwise, release the ID and return an error.
    //
    value = GPU_REG_IDX_RD_DRF(_PMGR, _MUTEX_REG, physMutexId, _VALUE);
    if (value == ownerId)
    {
        *pOwnerId = ownerId;
    }
    else
    {
        dprintf("lw: %s: Cannot acquire mutex index %d (owned by %d).\n",
                __FUNCTION__, physMutexId, value);

        _pmgrMutexIdRel_GP100(ownerId);
        status = LW_ERR_STATE_IN_USE;
    }
    return status;
}

/*!
 * Attempts to release the PMGR mutex as specified by the given physical
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
pmgrMutexReleaseByIndex_GP100
(
    LwU32     physMutexId,
    LwU32     ownerId
)
{
    GPU_REG_IDX_WR_DRF_DEF(_PMGR, _MUTEX_REG, physMutexId, _VALUE, _INITIAL_LOCK);

    // release the mutex identifer
    _pmgrMutexIdRel_GP100(ownerId);
}
