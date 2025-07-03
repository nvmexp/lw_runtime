/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch PG helper.  
//
//*****************************************************

//
// includes
//
#include "elpg.h"
#include "hal.h"
#include "volta/gv100/dev_gc6_island.h"


LwU32
elpgBsiRamRead_GV100(
    LwU32   offset,
    LwU32   size,
    LwU32  *buffer)
{
    LwU32 saveReg, tmp, idx;

    saveReg = GPU_REG_RD32(LW_PGC6_BSI_RAMCTRL);
    tmp = 0;

    tmp = FLD_SET_DRF(_PGC6, _BSI_RAMCTRL, _RAUTOINCR, _ENABLE, tmp);
    tmp = FLD_SET_DRF_NUM(_PGC6, _BSI_RAMCTRL, _ADDR, offset, tmp);
    GPU_REG_WR32(LW_PGC6_BSI_RAMCTRL, tmp);

    for (idx =0; idx < size; idx++)
    {
        buffer[idx] = GPU_REG_RD32(LW_PGC6_BSI_RAMDATA);
    }

    GPU_REG_WR32(LW_PGC6_BSI_RAMCTRL, saveReg);

    return size;
}
