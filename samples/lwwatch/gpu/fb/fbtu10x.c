/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbtu10x.c
//
//*****************************************************

//
// includes
//
#include "turing/tu102/dev_ltc.h"
#include "turing/tu102/dev_graphics_nobundle.h"
#include "turing/tu102/dev_graphics_nobundle_addendum.h"
#include "chip.h"
#include "hal.h"
#include "fb.h"
#include "priv.h"
#include "turing/tu102/hwproject.h"

//-----------------------------------------------------
// _fbReadDSColorFormatZBCindex_TU10X(LwU32 zbcIndex)
//
//-----------------------------------------------------
static LwU32
_fbReadDSColorFormatZBCindex_TU10X
(
    LwU32 zbcIndex
)
{
    LwU32 format;
    LwU32 index;
    LwU32 rangeIndex, indexModuloFour;

    index = zbcIndex - 1;
    rangeIndex = index / 4;
    indexModuloFour = index % 4;

    format = GPU_REG_RD32(LW_PGRAPH_PRI_GPC_SWDX_DSS_ZBC_C_FORMAT(rangeIndex));

    switch(indexModuloFour)
    {
        case 0:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_C_01_TO_04_FORMAT, _ENTRY_01, format);
            break;
        case 1:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_C_01_TO_04_FORMAT, _ENTRY_02, format);
            break;
        case 2:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_C_01_TO_04_FORMAT, _ENTRY_03, format);
            break;
        case 3:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_C_01_TO_04_FORMAT, _ENTRY_04, format);
            break;
    }

    return format;
}

//-----------------------------------------------------
// fbReadDSDepthZBCindex_TU10X(LwU32 zbcIndex)
//
//-----------------------------------------------------
LW_STATUS
fbReadDSDepthZBCindex_TU10X
(
    LwU32 zbcIndex
)
{
    LwU32 dataZBC;
    LwU32 format;
    LwU32 index;
    LwU32 rangeIndex, indexModuloFour;

    index = zbcIndex - 1;
    rangeIndex = index / 4;
    indexModuloFour = index % 4;

    dataZBC = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_Z, _VAL,
                      GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_SWDX_DSS_ZBC_Z(index)));
    format = GPU_REG_RD32(LW_PGRAPH_PRI_GPC_SWDX_DSS_ZBC_Z_FORMAT(rangeIndex));

    switch(indexModuloFour)
    {
        case 0:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_Z_01_TO_04_FORMAT, _ENTRY_01, format);
            break;
        case 1:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_Z_01_TO_04_FORMAT, _ENTRY_02, format);
            break;
        case 2:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_Z_01_TO_04_FORMAT, _ENTRY_03, format);
            break;
        case 3:
            format = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_Z_01_TO_04_FORMAT, _ENTRY_04, format);
            break;
    }

     dprintf("lw: Depth Value 0x%x\n", dataZBC);
     dprintf("lw: FB Depth Format Value is 0x%x\n", format);

    return LW_OK;
}

//-----------------------------------------------------
// fbReadDSColorZBCindex_TU10X(LwU32 zbcIndex)
//
//-----------------------------------------------------
LW_STATUS
fbReadDSColorZBCindex_TU10X
(
    LwU32 zbcIndex
)
{
    LwU32 dataZBC[4] = {0, 0, 0, 0};
    LwU32 format = 0;

    dataZBC[0] = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_COLOR_R, _VAL,
                         GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_SWDX_DSS_ZBC_COLOR_R(zbcIndex - 1)));
    dataZBC[1] = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_COLOR_G, _VAL,
                         GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_SWDX_DSS_ZBC_COLOR_G(zbcIndex - 1)));
    dataZBC[2] = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_COLOR_B, _VAL,
                         GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_SWDX_DSS_ZBC_COLOR_B(zbcIndex - 1)));
    dataZBC[3] = DRF_VAL(_PGRAPH, _PRI_GPC0_SWDX_DSS_ZBC_COLOR_A, _VAL,
                         GPU_REG_RD32(LW_PGRAPH_PRI_GPC0_SWDX_DSS_ZBC_COLOR_A(zbcIndex - 1)));

    format     = _fbReadDSColorFormatZBCindex_TU10X(zbcIndex);

    dprintf("lw: R Color Value 0x%x\n", dataZBC[0]);
    dprintf("lw: G Color Value 0x%x\n", dataZBC[1]);
    dprintf("lw: B Color Value 0x%x\n", dataZBC[2]);
    dprintf("lw: A Color Value 0x%x\n", dataZBC[3]);
    dprintf("lw: FB Color Format Value is 0x%x\n", format);

    return LW_OK;
}

LwU32
fbGetActiveLtsMaskForLTC_TU102(LwU32 ltcIdx)
{
    LwU32 regVal;
    LwU32 ltsDisableMask = 0;
    LwU32 numLTSPerLTC;

    regVal = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_CBC_PARAM + (ltcIdx * LW_LTC_PRI_STRIDE));
    numLTSPerLTC = DRF_VAL(_PLTCG, _LTC0_LTS0_CBC_PARAM, _SLICES_PER_LTC, regVal);

    // One slice per LTC is not viable for Turing.
    assert(numLTSPerLTC == LW_PLTCG_LTC0_LTS0_CBC_PARAM_SLICES_PER_LTC_2 ||
            numLTSPerLTC == LW_PLTCG_LTC0_LTS0_CBC_PARAM_SLICES_PER_LTC_4);

    regVal = GPU_REG_RD32(LW_PLTCG_LTC0_MISC_LTC_CFG + (ltcIdx * LW_LTC_PRI_STRIDE));

    if (FLD_TEST_DRF(_PLTCG_LTC0, _MISC_LTC_CFG,
                    _FS_LOWER_LTS_PAIR, _DISABLED, regVal))
    {
        ltsDisableMask |= 0x3; // LWBIT32(2) - 1;
    }

    if (FLD_TEST_DRF(_PLTCG_LTC0, _MISC_LTC_CFG,
                    _FS_UPPER_LTS_PAIR, _DISABLED, regVal))
    {
        ltsDisableMask |= 0xC; // ((LWBIT32(2) - 1) << (2));
    }

    return ~ltsDisableMask & (LWBIT32(numLTSPerLTC) - 1);
}