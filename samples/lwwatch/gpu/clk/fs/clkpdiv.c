/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file
 * @see     https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @see     Bug 2557340: Turn off VPLL, HUBPLL and DISPPLL based on usecase
 * @see     Bug 2571031: Ampere DISPCLK/HUBCLK: Switching to bypass after initialization
 * @see     Perforce:  hw/doc/gpu/SOC/Clocks/Documentation/Display/Unified_Display_Clocking_Structure.vsdx
 * @author  Daniel Worpell
 * @author  Antone Vogt Varvak
 */

/* ------------------------- System Includes -------------------------------- */
/* ------------------------- Application Includes --------------------------- */

#include "clk.h"
#include "clk/fs/clkpdiv.h"
#include "clk/generic_dev_trim.h"
#include "print.h"

/* ------------------------- Type Definitions ------------------------------- */
/* ------------------------- External Definitions --------------------------- */
/* ------------------------- Static Variables ------------------------------- */
/* ------------------------- Global Variables ------------------------------- */

CLK_DEFINE_VTABLE__FREQSRC(PDiv);

/* ------------------------- Macros and Defines ----------------------------- */
/* ------------------------- Prototypes ------------------------------------- */
/* ------------------------- Public Functions ------------------------------- */
/* ------------------------- Virtual Implemenations ------------------------- */

/*!
 * @brief       Get the frequency.
 * @memberof    ClkPDiv
 * @see         ClkFreqSrc_Virtual::clkRead
 *
 * @details     This implementation reads the divider after calling clkRead
 *              on the input.  From that it computes the output frequency.
 *
 * @param[in]   this        Instance of ClkPDiv from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
void
clkReadAndPrint_PDiv
(
    ClkPDiv     *this,
    LwU32       *pFreqKHz
)
{
    LwU32 divValue;
    LwU32 regData;
    
    // Read the input frequency.
    clkReadAndPrint_Wire(&(this->super), pFreqKHz);

    // Read the divider value.
    regData = GPU_REG_RD32(this->regAddr);

    // Extract the divider value.
    divValue = (regData >> this->base) & (LWBIT32(this->size) - 1);

    //
    // On some dividers, zero means powered-off, but even if that's not the
    // case on this divider, we can't divide by zero and expect a good result.
    //
    if (divValue == 0)
    {
        dprintf("lw: ERROR: %s: divider=0 or powered-off\n", CLK_NAME(this));
        dprintf("lw: %s: Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->regAddr, regData);
        *pFreqKHz = 0;
    }
    else
    {
        // Compute the output value if this LDIV unit.
        *pFreqKHz /= divValue;
        dprintf("lw: %s: Freq=%uKHz Div=%u\n", CLK_NAME(this), *pFreqKHz, divValue);

    }
}
