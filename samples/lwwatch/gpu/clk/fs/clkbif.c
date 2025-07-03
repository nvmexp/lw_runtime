/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file
 * @see https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @author Daniel Worpell
 * @author Chandrabhanu Mahapatra
 */

/* ------------------------- System Includes -------------------------------- */
/* ------------------------- Application Includes --------------------------- */

#include "clk.h"
#include "clk/fs/clkbif.h"
#include "hopper/gh100/dev_top.h"
#include "os.h"
#include "hal.h"
#include "bif.h"
#include "print.h"

/* ------------------------- Type Definitions ------------------------------- */
/* ------------------------- External Definitions --------------------------- */
/* ------------------------- Static Variables ------------------------------- */
/* ------------------------- Global Variables ------------------------------- */

CLK_DEFINE_VTABLE__FREQSRC(Bif);

/* ------------------------- Macros and Defines ----------------------------- */
/* ------------------------- Prototypes ------------------------------------- */
/* ------------------------- Public Functions ------------------------------- */
/* ------------------------- Virtual Implemenations ------------------------- */

/*!
 * @see         clkReadAndPrint_FreqSrc
 * @brief       Get the link speed from the previous phase else from hardware
 *
 * @memberof    ClkBif
 *
 * @param[in]   this        Instance of ClkBif from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
void
clkReadAndPrint_Bif
(
    ClkBif     *this,
    LwU32      *pFreqKHz
)
{
    LwU32 platformData;
    LwU32 lwrrentGenSpeed = 0;
    //
    // Skip these registers on FMODEL since they are not modelled.
    // Assume a reasonable default instead.
    //
    platformData = GPU_REG_RD32(LW_PTOP_PLATFORM);
    if (FLD_TEST_DRF(_PTOP, _PLATFORM, _TYPE, _FMODEL, platformData))
    {
        *pFreqKHz = RM_PMU_BIF_LINK_SPEED_GEN1PCIE;
    }
    else
    {
        // Get the gen speed.
        pBif[indexGpu].bifGetBusGenSpeed(&lwrrentGenSpeed);
        //
        // Since there is no actual frequency with gen speed, we use the link
        // speed as the frequency.  This is kinda hacky, but colwenient since
        // we can reuse the interface used by OBJPERF, various MODS tests, etc.
        //
        *pFreqKHz = lwrrentGenSpeed;
    }
    dprintf("lw: %s: lwrrentGenSpeed: %u\n", CLK_NAME(this), *pFreqKHz);
}
