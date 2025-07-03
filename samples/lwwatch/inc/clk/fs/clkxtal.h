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


#ifndef CLK3_FS_XTAL_H
#define CLK3_FS_XTAL_H


/* ------------------------ Includes --------------------------------------- */

#include "clk/fs/clkfreqsrc.h"


/* ------------------------ Macros ----------------------------------------- */
/* ------------------------ Datatypes -------------------------------------- */

typedef struct ClkXtal              ClkXtal;
typedef        ClkFreqSrc_Virtual   ClkXtal_Virtual;


/*!
 * @class       ClkXtal
 * @extends     ClkFreqSrc
 * @brief       Zero-Input Frequency Source
 * @protected
 *
 * @details     Objects of this class represent crystals, but can also be used
 *              any place where we want a hard-coded frequency to be reported.
 *
 *              Furthermore, this class can be extended for any frequency source
 *              that takes no input clock signals.
 *
 * @todo        The xtal may have different frequencies depending on the SKU.
 *              This should be determined at runtime by checking the
 *              LW_PEXTDEV_BOOT_0_STRAP_CRYSTAL register, and then setting the
 *              ClkXtal's final pointer to point to a ClkXtal with the correct
 *              frequency.
 */
struct ClkXtal
{
/*!
 * @brief       Inherited state
 *
 * @ilwariant   Inherited state must always be first.
 */
    ClkFreqSrc  super;

/*!
* @brief        Crystal oscillation frequency in KHz
*/
    LwU32 freqKHz;
};



/* ------------------------ External Definitions --------------------------- */

/*!
 * @brief       Virtual table
 * @memberof    ClkXtal
 * @protected
 */
extern ClkXtal_Virtual clkVirtual_Xtal;


/* ------------------------ Function Prototypes ---------------------------- */

/*!
 * @brief       Implementation of the virtual function
 * @memberof    ClkXtal
 * @protected
 */
void clkReadAndPrint_Xtal(ClkXtal *this, LwU32 *pFreqKHz);


/* ------------------------ Include Derived Types -------------------------- */

#endif // CLK3_FS_XTAL_H

