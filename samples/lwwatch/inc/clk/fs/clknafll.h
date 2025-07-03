/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
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
 */


#ifndef CLK3_FS_NAFLL_H
#define CLK3_FS_NAFLL_H


/* ------------------------ Includes --------------------------------------- */
#include "clk/fs/clkfreqsrc.h"

/* ------------------------ Macros ----------------------------------------- */
/* ------------------------ Datatypes -------------------------------------- */
typedef struct ClkNafll                 ClkNafll;
typedef        ClkFreqSrc_Virtual       ClkNafll_Virtual;


/*!
 * @class       ClkNafll
 * @extends     ClkFreqSrc
 * @version     Clocks 3.1 and after
 * @brief       NAFLL frequency source programming.
 * @protected
 *
 * @details     This class calls into NAFLL routines for programming NAFLL
 *              clock domains.
 *              https://wiki.lwpu.com/engwiki/index.php/Clocks/AVFS
 */
struct ClkNafll
{
    /*!
     * @brief       Inherited state
     *
     * @ilwariant   Inherited state must always be first.
     */
    ClkFreqSrc                            super;
    
    /*!
     * @brief       Coefficent Register Address
     *
     * @details     This variable holds the register address of the PRI which
     *              holds the MDIV and PDIV values for this nafll. E.G.
     *              LW_PTRIM_GPC_GPCNAFLL_COEFF(0) for GH100
     */    
    LwU32                                 coeffRegAddr;
    
    /*!
     * @brief       SW_FREQ_REQUEST Register Address
     *
     * @details     This variable holds the register address of the PRI which
     *              holds the current Override Mode for this Nafll. 
     *              E.g. LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(0) for GH100
     */   
    LwU32                                 swfreqRegAddr;

    /*!
     * @brief       LUT_STATUS Register Address
     *
     * @details     This variable holds the register address of the PRI which
     *              holds the current effective NDIV value for the Nafll.
     *              E.g. LW_PTRIM_GPC_GPCLUT_STATUS(0) for GH100
     */   
    LwU32                                 lutStatusRegAddr;

    /*!
     * @brief       Reference Clock Register Address
     *
     * @details     This variable holds the register address of the PRI which
     *              holds Divider value for the input reference clock to this
     *              Nafll. E.g. LW_PTRIM_SYS_AVFS_REFCLK_CORE_CONTROL for GH100
     */

    LwU32                                 refdivRegAddr;
    /*!
     * The global ID @ref LW2080_CTRL_CLK_NAFLL_ID_<xyz> of this
     * NAFLL device.
     *
     * NAFLL souce will use this @ref id to get the pointe to
     * @ref CLK_NAFLL_DEVICE struct containing static params
     * of NAFLL device.
     */
    LwU8                                  id;
};



/* ------------------------ External Definitions --------------------------- */

/*!
 * @brief       Virtual table
 * @memberof    ClkNafll
 * @protected
 */
extern ClkNafll_Virtual clkVirtual_Nafll;


/* ------------------------ Function Prototypes ---------------------------- */

/*!
 * @brief       Implementation of the virtual function
 * @memberof    ClkNafll
 * @protected
 */
void clkReadAndPrint_Nafll(ClkNafll *this, LwU32 *pFreqKHz);



/* ------------------------ Include Derived Types -------------------------- */

#endif // CLK3_FS_NAFLL_H

