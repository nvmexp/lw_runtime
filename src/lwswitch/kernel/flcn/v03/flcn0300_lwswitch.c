/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   flcn0300_lwswitch.c
 * @brief  Provides the implementation for all falcon 3.0 HAL interfaces.
 */

#include "lwmisc.h"
#include "common_lwswitch.h"

#include "flcn/flcnable_lwswitch.h"
#include "flcn/flcn_lwswitch.h"

#include "lwswitch/lr10/dev_falcon_v4.h"

/*!
 * @brief Get information about the falcon core
 *
 * @param[in] device lwswitch_device pointer
 * @param[in] pFlcn  FLCN pointer
 *
 * @returns nothing
 */
static void
_flcnGetCoreInfo_v03_00
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LwU32 hwcfg1 = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_HWCFG1);

    if (FLD_TEST_DRF(_PFALCON, _FALCON_HWCFG1, _SELWRITY_MODEL, _HEAVY, hwcfg1))
    {
        LWSWITCH_PRINT(device, INFO,
                    "%s: Engine '%s' is using the heavy security model\n",
                    __FUNCTION__, flcnGetName_HAL(device, pFlcn));
    }

    // Save off the security model.
    pFlcn->selwrityModel = DRF_VAL(_PFALCON, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1);

    // Combine Falcon core revision and subversion for easy version comparison.
    pFlcn->coreRev = flcnableReadCoreRev(device, pFlcn->pFlcnable);

    pFlcn->supportsDmemApertures = FLD_TEST_DRF(_PFALCON, _FALCON_HWCFG1, _DMEM_APERTURES, _ENABLE, hwcfg1);
}

/**
 * @brief   set hal function pointers for functions defined in v03_00 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcn   The flcn for which to set hals
 */
void
flcnSetupHal_v03_00
(
    PFLCN            pFlcn
)
{
    flcn_hal *pHal = pFlcn->pHal;

    pHal->getCoreInfo = _flcnGetCoreInfo_v03_00;
}

