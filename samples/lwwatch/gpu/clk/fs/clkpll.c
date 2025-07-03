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
 * @author  Ming Zhong
 * @author  Prafull Gupta
 */

/* ------------------------- System Includes -------------------------------- */
/* ------------------------- Application Includes --------------------------- */

#include "clk.h"
#include "os.h"
#include "print.h"
#include "hopper/gh100/dev_trim.h"
#include "hopper/gh100/dev_trim_addendum.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"
#include "clk/fs/clkpll.h"
#include "clk/generic_dev_trim.h"

/* ------------------------- Macros and Defines ----------------------------- */

#define DELAY_READY_NS 5000
#define CLK_RETRY_COUNT_MAX_PLL 5

/*!
 * @brief: True if the new config is better than the old (prioritize accuracy)
 */
#define CONFIG_COMPARE(new, old)                                               \
     ((new).deltaKHz            <   (old).deltaKHz          ||                 \
     ((new).deltaKHz            ==  (old).deltaKHz          &&                 \
      (new).status == FLCN_OK   &&  (old).status != FLCN_OK))


/* ------------------------- Type Definitions ------------------------------- */
CLK_DEFINE_VTABLE__FREQSRC(Pll);


/* ------------------------- Private Functions ------------------------------ */
/* ------------------------- Prototypes ------------------------------------- */
/* ------------------------- Public Functions ------------------------------- */
/* ------------------------- Virtual Implemenations ------------------------- */

/*******************************************************************************
    Reading
*******************************************************************************/

/*!
 * @see         ClkFreqSrc_Virtual::clkRead
 * @memberof    ClkPll
 * @brief       Read the coefficients from hardware
 *
 * @details     Determine the current selected input by reading the hardware,
 *              then daisy-chain through the input node to get the frequency.
 *
 * @param[in]   this        Instance of ClkPDiv from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
void
clkReadAndPrint_Pll
(
    ClkPll      *this,
    LwU32       *pFreqKHz
)
{
    LwU32 cfgRegData;
    LwU32 mdiv = 0;
    LwU32 ndiv = 0;
    LwU32 pldiv = 0;
    
    // If this pll isn't on, set coefficients to be zero
    cfgRegData = GPU_REG_RD32(this->cfgRegAddr);
    if (FLD_TEST_DRF(_PTRIM, _SYS_PLL_CFG, _ENABLE, _NO, cfgRegData))
    {
        dprintf("lw: %s: Not Enabled\n", CLK_NAME(this));
        *pFreqKHz = 0;
    }

    //
    // If the pll is on, then read the coefficients from hardware, and then
    // read the input
    //
    else
    {
        //
        // Analog PLLs have one register for coefficients, Hybrids have two.
        // MDIV and PLDIV are in the same register/field for all PLL types.
        //
        LwU32 coeffRegData = GPU_REG_RD32(this->coeffRegAddr);
        mdiv  = DRF_VAL(_PTRIM, _SYS_PLL_COEFF, _MDIV, coeffRegData);
        ndiv  = DRF_VAL(_PTRIM, _SYS_APLL_COEFF, _NDIV, coeffRegData);
        pldiv = 1;
        
        //
        // DRAMPLL does not have a PLDIV.  Instead it has a SEL_DIVBY2 which acts
        // as a glitchy version of PLDIV that always divides by 2.
        //
        if (this->bDiv2Exists &&
            FLD_TEST_DRF(_PFB, _FBPA_PLL_COEFF, _SEL_DIVBY2, _ENABLE, coeffRegData))
        {
            pldiv = 2;
        }

        //
        // If this PLL contains a PLDIV, then get its value.
        //
        else if (this->bPldivExists)
        {
            pldiv = DRF_VAL(_PTRIM, _SYS_PLL_COEFF, _PLDIV,  coeffRegData);
        }
        // This PLL has only a VCO, so set the divider value to 1.

        clkReadAndPrint_Wire(&this->super, pFreqKHz);
        
        if (mdiv == 0 || pldiv == 0)
        {
            dprintf("lw: ERROR: %s: %s=0\n", CLK_NAME(this), (mdiv == 0 ? "mdiv":"pdiv"));
            dprintf("lw: %s: (CFG)   Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->cfgRegAddr,   cfgRegData);
            dprintf("lw: %s: (COEFF) Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->coeffRegAddr, coeffRegData);
            *pFreqKHz = 0;
        }
        else
        {
            *pFreqKHz *= ndiv;
            *pFreqKHz /= mdiv;
            *pFreqKHz /= pldiv;
            if (this->bDiv2Exists)
                dprintf("lw: %s: Freq=%uKHz ndiv=%u mdiv=%u div2=%s\n", CLK_NAME(this), *pFreqKHz, ndiv, mdiv, (pldiv == 1? "disabled" : "enabled"));
            else if (this->bPldivExists)
                dprintf("lw: %s: Freq=%uKHz ndiv=%u mdiv=%u pldiv=%u\n", CLK_NAME(this), *pFreqKHz, ndiv, mdiv, pldiv);
            else
                dprintf("lw: %s: Freq=%uKHz ndiv=%u mdiv=%u\n", CLK_NAME(this), *pFreqKHz, ndiv, mdiv);
        }
    }
}
