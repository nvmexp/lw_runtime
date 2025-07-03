/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FERMI_CLK_H_
#define _FERMI_CLK_H_

#define LW_PTRIM_CLK_NAMEMAP_INDEX_HOSTCLK    0x0040
#define FERMI_VCLK_NAME_MAP_INDEX_TO_HEADNUM(nameMapIndex)  (nameMapIndex - LW_PVTRIM_CLK_NAMEMAP_INDEX_VCLK(0))
#define RM_ASSERT(x)
#define DBG_BREAKPOINT()

typedef enum
{
    spread_Center = 0,
    spread_Down   = 1
} SPREADTYPE;

typedef struct
{
    BOOL       bSDM;
    BOOL       bSSC;
    LwS32      SDM;
    LwS32      SSCMin;
    LwS32      SSCMax;
    SPREADTYPE spreadType;
} PLLSPREADPARAMS;

LwU32 clkGenReadClk_FERMI(LwU32 nameMapIndex);

#endif  // _FERMI_CLK_H_
