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
 * @see     https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @author  Daniel Worpell
 * @author  Chandrabhanu Mahapatra
 */


#ifndef CLK3_FS_BIF_H
#define CLK3_FS_BIF_H


/* ------------------------ Includes --------------------------------------- */

#include "clk/fs/clkfreqsrc.h"
#include "pmu/pmuifbif.h"

/* ------------------------ Macros ----------------------------------------- */
/* ------------------------ Datatypes -------------------------------------- */

typedef struct ClkBif               ClkBif;
typedef        ClkFreqSrc_Virtual   ClkBif_Virtual;


/*!
 * @class       ClkBif
 * @extends     ClkFreqSrc
 * @brief       Bus Inferface Gen Speed (PCIE)
 * @protected
 *
 * @details     This class calls into OBJBIF to get/set the PCIE link speed
 *              (genspeed).
 *
 *              Unlink other ClkFreqSrc classes, ClkSignal::freqSrc contains
 *              the genspeed rather than a frequency.  As of July 2015, this
 *              class supports Gen1 through Gen4, a superset of OBJBIF.
 *
 *              Typically, an objects of this class is the root for the
 *              PCIEGenClk domain, symbolized by clkWhich_PCIEGenClk and
 *              LW2080_CTRL_CLK_DOMAIN_PCIEGENCLK
 */
struct ClkBif
{
    /*!
     * @brief       Inherited state
     *
     * @ilwariant   Inherited state must always be first.
     */
    ClkFreqSrc super;

    /*!
     * @brief       Gen Speed lwrrently programmed
     */
    RM_PMU_BIF_LINK_SPEED lwrrentGenSpeed;
};


/* ------------------------ External Definitions --------------------------- */

/*!
 * @brief       Virtual table
 * @memberof    ClkBif
 * @protected
 */
extern ClkBif_Virtual clkVirtual_Bif;


/* ------------------------ Function Prototypes ---------------------------- */

/*!
 * @brief       Implementation of the virtual function
 * @memberof    ClkBif
 * @protected
 */
void clkReadAndPrint_Bif(ClkBif *this, LwU32 *pFreqKHz);

/* ------------------------ Include Derived Types -------------------------- */

#endif // CLK3_FS_BIF_H

