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
 * @author  Lawrence Chang
 * @author  Manisha Choudhury
 */

/* ------------------------- System Includes -------------------------------- */
/* ------------------------- Application Includes --------------------------- */

#include "clk.h"
#include "clk/fs/clkmux.h"
#include "print.h"


/* ------------------------- Macros and Defines ----------------------------- */
/* ------------------------- Type Definitions ------------------------------- */
/* ------------------------- External Definitions --------------------------- */
/* ------------------------- Static Variables ------------------------------- */
/* ------------------------- Global Variables ------------------------------- */

CLK_DEFINE_VTABLE__FREQSRC(Mux);


/* ------------------------- Prototypes ------------------------------------- */

static LW_INLINE LwU8 clkReadAndPrint_FindMatch_Mux(const ClkMux *this, LwU32 muxRegData);

/* ------------------------- Public Functions ------------------------------- */
/* ------------------------- Virtual Implemenations ------------------------- */

/*******************************************************************************
    Reading
*******************************************************************************/

/*!
 * @memberof    ClkMux
 * @brief       Find a match in the field value map.
 * @see         ClkMux::muxValueMap
 *
 * @details     This function uses 'muxValueMap' to translate the data read
 *              from the mux register into the node of the selected input.
 *
 *              If there is no match, CLK_SIGNAL_NODE_INDETERMINATE is
 *              returned.
 *
 * @param[in]   this            This ClkMux object
 * @param[in]   muxRegData      Register data to find
 *
 * @return                                      Node of the matching element.
 * @retval      CLK_SIGNAL_NODE_INDETERMINATE   There is no match.
 */
static LW_INLINE LwU8
clkReadAndPrint_FindMatch_Mux
(   const ClkMux *this,
    LwU32 muxRegData
)
{
    LwU8 i;
    LwU8 count = this->count;

    // Search the map.
    for (i = 0; i < count; ++i)
    {
        if (CLK_MATCHES__FIELDVALUE(this->muxValueMap[i], muxRegData))
        {
            return i;
        }
    }

    // There is no match.
    return CLK_SIGNAL_NODE_INDETERMINATE;
}


/*!
 * @see         ClkFreqSrc_Virtual::clkRead
 * @brief       read the frequency of the active input from hardware
 *
 * @details     This function read the mux's register to determine which of
 *              its input nodes is selected.  It then reads that input node
 *              and returns its clock signal.
 * 
 * @param[in]   this        Instance of ClkMux from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
void clkReadAndPrint_Mux
(
    ClkMux      *this,
    LwU32       *pFreqKHz
)
{
    LwU32           muxRegData;
    LwU8            node;       // Index of the input node

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
    // Read the value of the register and get the node from it
    muxRegData = GPU_REG_RD32(this->muxRegAddr);
    node = clkReadAndPrint_FindMatch_Mux(this, muxRegData);
    
    //
    // If value in the register is not in the map, either it is gated, or the
    // hardware is lwrrently in an invalid state.
    //
    if (node == CLK_SIGNAL_NODE_INDETERMINATE)
    {
        if (!CLK_MATCHES__FIELDVALUE(this->muxGateField, muxRegData))
        {
            dprintf("lw: ERROR: %s has an unsupported field in it's CFG register\n", CLK_NAME(this));
            dprintf("lw: %s: Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->muxRegAddr, muxRegData);
        }
        else
        {
            dprintf("lw: WARNING: %s is gated\n", CLK_NAME(this));
        }
    }

    // Make sure the input is hooked up to something
    else if (this->input[node] == NULL)
    {
        dprintf("lw: ERROR: %s node %u is unsupported in LwWatch\n", CLK_NAME(this), node);
        dprintf("lw: %s: Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->muxRegAddr, muxRegData);
    }

    //
    // Read the input selected by this mux (if any).  That input is active only
    // if this mux is along the active path.
    //
    else
    {
        clkReadAndPrint_FreqSrc_VIP(this->input[node], pFreqKHz);
        dprintf("lw: %s: Freq=%uKHz Node=%u\n", CLK_NAME(this), *pFreqKHz, node);
    }
    this->super.bCycle = LW_FALSE;            
}
