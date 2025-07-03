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
 * @see     https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @author Daniel Worpell
 * @author Eric Colter
 */

/* ------------------------- System Includes -------------------------------- */
/* ------------------------- Application Includes --------------------------- */

#include "clk.h"
#include "os.h"
#include "print.h"
#include "clk/fs/clkxtal.h"

/* ------------------------- Type Definitions ------------------------------- */
/* ------------------------- External Definitions --------------------------- */
/* ------------------------- Static Variables ------------------------------- */
/* ------------------------- Global Variables ------------------------------- */

//! Virtual table
CLK_DEFINE_VTABLE__FREQSRC(Xtal);

/* ------------------------- Macros and Defines ----------------------------- */
/* ------------------------- Private Functions ------------------------------ */
/* ------------------------- Prototypes ------------------------------------- */
/* ------------------------- Public Functions ------------------------------- */

/*!
 * @see         ClkReadAndPrint_FreqSrc_VIP
 * @brief       Get the frequency in KHz of the crystal.
 * @memberof    ClkXtal
 *
 * @param[in]   this      instance of ClkXtal to read the frequency of
 * @param[out]  pFreqKHz  pointer to variable which will hold callwlated frequency 
 */
void
clkReadAndPrint_Xtal
(
    ClkXtal     *this,
    LwU32       *pFreqKHz
)
{
    *pFreqKHz = this->freqKHz;
    dprintf("lw: %s: Freq=%uKHz\n", CLK_NAME(this), *pFreqKHz);
}
