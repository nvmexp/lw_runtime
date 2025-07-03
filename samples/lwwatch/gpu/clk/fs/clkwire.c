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
 * @author  Daniel Worpell
 * @author  Eric Colter
 * @author  Antone Vogt-Varvak
 */

#include "clk/fs/clkwire.h"
#include "print.h"

/*******************************************************************************
    Virtual Table
*******************************************************************************/
CLK_DEFINE_VTABLE__FREQSRC(Wire);
/*******************************************************************************
    Virtual Function Implementation
*******************************************************************************/

/*!
 * @see         ClkFreqSrc_Virtual::clkRead
 * @brief       Get the frequency in KHz of the input to the wire.
 *
 * @memberof    ClkWire
 *
 * @param[in]   this        Instance of ClkWire from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
void
clkReadAndPrint_Wire
(
    ClkWire     *this,
    LwU32       *pFreqKHz
)
{
    //
    // If a cycle has been detected, we need to break the relwrsive chain of
    // calls.
    //
    if (this->super.bCycle)
    {
        //
        // If this is part of the active path, and a cycle is detected, then
        // there is a big problem.  The hardware is in a bad state.
        //
        dprintf("lw: ERROR: %s Cycle Detected in Schematic Diagram", CLK_NAME(this));
        return;
    }
    
    this->super.bCycle = LW_TRUE;
    clkReadAndPrint_FreqSrc_VIP(this->pInput, pFreqKHz);
    this->super.bCycle = LW_FALSE;
}
