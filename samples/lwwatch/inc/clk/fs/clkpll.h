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
 * @see     //hw/libs/common/analog/lwpu/doc/
 * @see     //hw/doc/gpu/volta/volta/physical/clocks/Volta_clocks_block_diagram.vsd
 * @author  Daniel Worpell
 * @author  Eric Colter
 * @author  Antone Vogt-Varvak
 * @author  Ming Zhong
 * @author  Prafull Gupta
 */


#ifndef CLK3_FS_PLL_H
#define CLK3_FS_PLL_H

/* ------------------------ Includes --------------------------------------- */

#include "clk/fs/clkfreqsrc.h"
#include "clk/fs/clkwire.h"

/* ------------------------ Macros ----------------------------------------- */
/* ------------------------ Datatypes -------------------------------------- */

typedef ClkWire_Virtual ClkPll_Virtual;


/*!
 * @extends     ClkWire
 * @brief       Phase Locked Loop
 * @version     Clocks 3.1 and after
 * @protected
 *
 * @details     Objects of this class represtent a phase-locked loop in the hardware.
 */
typedef struct
{
/*!
 * @brief       Inherited state
 *
 * @details     ClkPll inherits from ClkWire since it is the class that handles
 *              objects with exactly one potential input, which is useful to
 *              separate out the lone-potential-input logic (e.g. daisy-chaining
 *              along the schematic dag) from the PLL-specific logic.
 */
    ClkWire     super;

/*!
 * @brief       Configuration register
 *
 * @details     The name of this register in the manuals is LW_PVTRIM_SYS_xxx_CFG.
 *              where xxx is the name of the PLL (with some exceptions).
 *
 *              This member can not be 'const' because there is floorsweeping
 *              logic for GA100 in 'clkConstruct_SchematicDag'.  However, its
 *              values does not change after construction.
 *
 *              The generic name for the register is LW_PTRIM_SYS_PLL_CFG.
 */
    LwU32 cfgRegAddr;

/*!
 * @brief       Coefficient register
 *
 * @details     The name of this register in the manuals is LW_PVTRIM_SYS_xxx_COEFF.
 *              where xxx is the name of the PLL (with some exceptions).
 *
 *              This member can not be 'const' because there is floorsweeping
 *              logic for GA100 in 'clkConstruct_SchematicDag'.  However, its
 *              values does not change after construction.
 *
 *              The generic name for the register is LW_PTRIM_SYS_PLL_COEFF.
 */
    LwU32 coeffRegAddr;

/*!
 * @brief       PLL contains a PLDIV
 *
 * @details     When false, there is no PLDIV, which is equivalent to a PLDIV
 *              programmed to 1.
 *
 * @note        bPldivExists and bDiv2Exists are mutually exclusive and shoud
 *              not both be set to true.
 */
    LwBool bPldivExists;

/*!
 * @brief       PLL contains a DIV2
 *
 * @details     DRAMPLL contains a glitchy DIV2 divider in place of the more
 *              common PLDIV.
 *
 * @note        bPldivExists and bDiv2Exists are mutually exclusive and should
 *              not both be set to true.
 */
    LwBool bDiv2Exists;
} ClkPll;


/* ------------------------ External Definitions --------------------------- */
/*!
 * @brief       Virtual table
 * @memberof    ClkPll
 * @protected
 */
extern ClkPll_Virtual clkVirtual_Pll;


/* ------------------------ Function Prototypes ---------------------------- */

/*!
 * @brief       Implementation of the virtual function
 * @memberof    ClkPll
 * @protected
 */
void clkReadAndPrint_Pll(ClkPll *this, LwU32 *pFreqKHz);


/* ------------------------ Include Derived Types -------------------------- */

#endif // CLK3_FS_PLL_H

