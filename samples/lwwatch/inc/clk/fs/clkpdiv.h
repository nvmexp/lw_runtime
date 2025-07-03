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
 * @see     Perforce:  hw/doc/gpu/SOC/Clocks/Documentation/Display/Unified_Display_Clocking_Structure.vsdx
 * @author  Daniel Worpell
 * @author  Antone Vogt-Varvak
 */

#ifndef CLK3_FS_ROPDIV_H
#define CLK3_FS_ROPDIV_H


/* ------------------------ Includes --------------------------------------- */
#include "clk/fs/clkfreqsrc.h"
#include "clk/fs/clkwire.h"


/* ------------------------ Macros ----------------------------------------- */
/* ------------------------ Datatypes -------------------------------------- */

// Same vtable as base class
typedef         ClkWire_Virtual ClkPDiv_Virtual;


/*!
 * @extends     ClkWire
 * @version     Clocks 3.1
 * @brief       Simple Read-Only Divider
 * @protected
 *
 * @details     Objects if this class represent a simple divider that is not
 *              programmed.  The clock signal passing through this divider is
 *              divided by the numerical value in the register field.  (This
 *              differs from LDIVs, for example.)
 */
typedef struct
{
/*!
 * @brief       Inherited state
 *
 * @ilwariant   Inherited state must always be first.
 */
    ClkWire     super;

/*!
 * @brief       Configuration register address
 *
 * @details     For SPPLLs, the name of this register in dev_trim is
 *              LW_PVTRIM_SYS_SPPLLn_COEFF2 where 'n' is the SPPLL number.
 *
 *              For REFPLL, the name in dev_fbpa is LW_PFB_FBPA_REFMPLL_COEFF.
 *
 *              For HBMPLL, the name in dev_fbpa is LW_PFB_FBPA_FBIO_COEFF_CFG(i)
 *              for some 'i' based on floorsweeping.
 *
 *              The generic name for the register is LW_PTRIM_SYS_PLL_COEFF.
 *
 *              This member can not be marked 'const' because the GH100 MCLK
 *              constructor (for example) assigns this value at runtime based
 *              on floorsweeping.  Nonetheless, it is effectively 'const'
 *              after construction.
 */
    LwU32       regAddr;

/*!
 * @brief       Position of divider field
 *
 * @details     For SPPLLs, the name of this field in dev_trim is
 *              LW_PVTRIM_SYS_SPPLL0_COEFF2_PDIVA or _PDIVB.
 */
    LwU8  base;

/*!
 * @brief       Size of divider field
 *
 * @details     For SPPLLs, the name of this field in dev_trim is
 *              LW_PVTRIM_SYS_SPPLL0_COEFF2_PDIVA or _PDIVB.
 */
    LwU8  size;
} ClkPDiv;


/* ------------------------ External Definitions --------------------------- */

/*!
 * @brief       Virtual table
 * @memberof    ClkPDiv
 * @protected
 */
extern ClkPDiv_Virtual clkVirtual_PDiv;


/* ------------------------ Function Prototypes ---------------------------- */

/*!
 * @brief       Implementation of the virtual function
 * @memberof    ClkPDiv
 * @protected
 */
void clkReadAndPrint_PDiv(ClkPDiv *this, LwU32 *pFreqKHz);

/* ------------------------ Include Derived Types -------------------------- */

#endif // CLK3_FS_PDIV_H

