/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2015 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//
// includes
//

#include "pascal/gp102/hwproject.h"
#include "chip.h"
#include "pascal/gp102/hwproject.h"
#include "pascal/gp102/dev_top.h"
#include "pascal/gp102/dev_fuse.h"
#include "pascal/gp102/dev_graphics_nobundle.h"

#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes


LwU32 grGetMaxTpcPerGpc_GP102()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetNumberPesPerGpc_GP102(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

/*!
 * Colwerts the "disable_mask" partition FS config
 * into "enable_mask" partition FS config.
 */
static LwU32 extractAndIlwert ( LwU32 nActive, LwU32 nMax )
{
    LwU32 mask = BITMASK (nMax);
    return (( ~nActive ) & mask);
}

// Gets active CE config.
LwBool grGetActiveCeConfig_GP102(LwU32 *activeCeConfig, LwU32 *maxNumberOfCes)
{
    //
    // This fuse is removed from GP10X+, CE will no longer be disabled in HW
    // See bug 200025697
    //
    *activeCeConfig = 0x0;

    *maxNumberOfCes = GPU_REG_RD32(LW_PTOP_SCAL_NUM_CES);
    if ((*maxNumberOfCes & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_CES register read gave 0x%x value.\n", *maxNumberOfCes);
        return LW_FALSE;
    }
    *activeCeConfig = extractAndIlwert (*activeCeConfig,*maxNumberOfCes);
    return LW_TRUE;
}

