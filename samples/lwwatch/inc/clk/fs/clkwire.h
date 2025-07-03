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


#ifndef CLK3_FS_WIRE_H
#define CLK3_FS_WIRE_H


/* ------------------------ Includes --------------------------------------- */

#include "clk/fs/clkfreqsrc.h"


/* ------------------------ Macros ----------------------------------------- */
/* ------------------------ Datatypes -------------------------------------- */

typedef struct ClkWire              ClkWire;
typedef        ClkFreqSrc_Virtual   ClkWire_Virtual;


/*!
 * @extends     ClkFreqSrc
 * @brief       Single-Input Frequency Source
 * @protected
 *
 * @details     Objects of this class do nothing except daisy-chain.  The output
 *              signal always equals the output of its input.  It is called
 *              ClkWire as a metaphor because wires don't do anything to the
 *              signal in hardware (we hope).
 *
 * @note        abstract:  This class does not have a vtable if its own.
 */
struct ClkWire
{
/*!
 * @brief       Inherited state
 *
 * @ilwariant   Inherited state must always be first.
 */
    ClkFreqSrc  super;

/*!
 * @brief       Input node per the schematic dag
 *
 * @note        init state: The value for this member is set during initialization
 *              and does not change after that.
 */
    ClkFreqSrc* pInput;
};


/* ------------------------ External Definitions --------------------------- */

extern ClkWire_Virtual clkVirtual_Wire;
/* ------------------------ Function Prototypes ---------------------------- */

/*!
 * @brief       Implementation of the virtual function
 * @memberof    ClkWire
 * @protected

 */
void clkReadAndPrint_Wire(ClkWire *this, LwU32 *pFreqKHz);


/* ------------------------ Include Derived Types -------------------------- */
#endif // CLK3_FS_WIRE_H

