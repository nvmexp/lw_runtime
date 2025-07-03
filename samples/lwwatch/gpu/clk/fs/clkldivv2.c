/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file
 * @see         https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @author      Daniel Worpell
 * @author      Antone Vogt-Varvak
 */

/* ------------------------- System Includes -------------------------------- */
/* ------------------------- Application Includes --------------------------- */

#include "clk.h"
#include "os.h"
#include "print.h"
#include "hopper/gh100/dev_trim.h"
#include "hopper/gh100/dev_trim_addendum.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"
#include "clk/fs/clkldivv2.h"
#include "clk/generic_dev_trim.h"


/* ------------------------- Type Definitions ------------------------------- */
/* ------------------------- External Definitions --------------------------- */
/* ------------------------- Static Variables ------------------------------- */
/* ------------------------- Global Variables ------------------------------- */

CLK_DEFINE_VTABLE__FREQSRC(LdivV2);


/* ------------------------- Macros and Defines ----------------------------- */
/* ------------------------- Prototypes ------------------------------------- */
/* ------------------------- Public Functions ------------------------------- */
/* ------------------------- Virtual Implemenations ------------------------- */

/*!
 * @brief       Read the coefficients from hardware
 * @memberof    ClkLdivV2
 * @see         ClkFreqSrc_Virtual::clkRead
 *
 * @details     Determine the current selected input by reading the hardware,
 *              then daisy-chain through the input node to get the frequency.
 *              Set all values of the phase array to align with the current
 *              hardware state.
 *
 * @param[in]   this        Instance of ClkLdivV2 from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
void
clkReadAndPrint_LdivV2
(
    ClkLdivV2           *this,
    LwU32               *pFreqKHz
)
{
    LwU32               ldivRegData;
    LwU8                ldivValue;

    // Read the divider value and the input frequency.
    ldivRegData = GPU_REG_RD32(this->ldivRegAddr);
    clkReadAndPrint_Wire(&(this->super), pFreqKHz);
    ldivValue = DRF_VAL(_PTRIM, _SYS_CLK_LDIV, _V2, ldivRegData) + 1;

    if (ldivValue == 0)
    {
        dprintf("lw: ERROR: %s: Callwlated ldivider=0\n", CLK_NAME(this));
        dprintf("lw: %s: Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->ldivRegAddr, ldivRegData);
        *pFreqKHz = 0;
    }
    else
    {
        // Compute the output value if this LDIV unit.
        *pFreqKHz = *pFreqKHz / ldivValue;
        dprintf("lw: %s: Freq=%uKHz Div=%u\n", CLK_NAME(this), *pFreqKHz, ldivValue);

    }
}
