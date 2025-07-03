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
 * @see    https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @author Daniel Worpell
 */

/* ------------------------- System Includes -------------------------------- */

#include "clk/fs/clknafll.h"
#include "os.h"
#include "clk.h"
#include "print.h"
#include "hopper/gh100/dev_trim.h"
#include "hopper/gh100/dev_trim_addendum.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"


/* ------------------------- Application Includes --------------------------- */
/* ------------------------- Type Definitions ------------------------------- */
/* ------------------------- External Definitions --------------------------- */
/* ------------------------- Static Variables ------------------------------- */
/* ------------------------- Global Variables ------------------------------- */

CLK_DEFINE_VTABLE__FREQSRC(Nafll);


/* ------------------------- Macros and Defines ----------------------------- */
/* ------------------------- Private Functions ------------------------------ */
/* ------------------------- Type Definitions ------------------------------- */
/* ------------------------- External Definitions --------------------------- */
/* ------------------------- Macros and Defines ----------------------------- */
/* ------------------------- Static Variables ------------------------------- */
/* ------------------------- Prototypes ------------------------------------- */
/* ------------------------- Public Functions ------------------------------- */

/* ------------------------- Prototypes ------------------------------------- */
/* ------------------------- Public Functions ------------------------------- */
/* ------------------------- Virtual Implemenations ------------------------- */

/*!
 * @see         clkReadAndPrint_FreqSrc
 * @brief       Get the link speed from the previous phase else from hardware
 *
 * @memberof    ClkNafll
 *
 * @param[in]   this        Instance of ClkNafll from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
void
clkReadAndPrint_Nafll
(
    ClkNafll    *this,
    LwU32       *pFreqKHz
)
{
    LwU8  overrideMode;
    LwU32 coeffRegData;
    LwU32 refRegData;
    LwU32 swfreqRegData;
    LwU32 lutStatusRegData;

    LwU32 pdiv;
    LwU32 ndiv;
    LwU32 mdiv;
    LwU32 inputRefClkDivVal;
    char  *modeString; 
    const LwU32 inputRefClkFreqMHz = 810;   
    
    // Read COEFF reg for MDIV and PDIV 
    coeffRegData = GPU_REG_RD32(this->coeffRegAddr);
    mdiv = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSNAFLL_COEFF, _MDIV, coeffRegData);
    pdiv = DRF_VAL(_PTRIM, _SYS_NAFLL_SYSNAFLL_COEFF, _PDIV, coeffRegData);
    
    // Read _SW_FREQ_REQ register to get overrideMode
    swfreqRegData = GPU_REG_RD32(this->swfreqRegAddr);
    overrideMode = DRF_VAL(_PTRIM_SYS, _NAFLL_SYSLUT_SW_FREQ_REQ, _SW_OVERRIDE_NDIV, swfreqRegData);

    //
    // Effective Ndiv value comes from the LUT_STATUS register since we do not
    // explicitly program the ndiv while in VR mode, its picked up from the LUT
    //
    lutStatusRegData = GPU_REG_RD32(this->lutStatusRegAddr);
    ndiv = DRF_VAL(_PTRIM_SYS, _NAFLL_SYSLUT_STATUS, _NDIV, lutStatusRegData);

    // Parse the programmed regime (to be filled in later).
    switch (overrideMode)
    {
        case LW2080_CTRL_CLK_NAFLL_SW_OVERRIDE_LUT_USE_HW_REQ:
        {
            modeString = "_LUT_USE_HW_REQ";
            break;
        }
        case LW2080_CTRL_CLK_NAFLL_SW_OVERRIDE_LUT_USE_SW_REQ:
        {
            modeString = "_LUT_USE_SW_REQ";
            break;
        }
        case LW2080_CTRL_CLK_NAFLL_SW_OVERRIDE_LUT_USE_MIN:
        {
            modeString = "_LUT_USE_MIN";
            break;
        }
        default:
        {
            modeString = "_LUT_USE_DEFAULT";
            break;
        }
    }

    // Read the NAFLL 
    refRegData = GPU_REG_RD32(this->refdivRegAddr);
    inputRefClkDivVal = DRF_VAL(_PTRIM_SYS, _AVFS_REFCLK_CORE_CONTROL, _AVFS_REFCLK_DIVIDE_CORE, refRegData);

    //Handle 0-valued divider errors
    if (inputRefClkDivVal == 0 || mdiv == 0 || pdiv == 0)
    {
        dprintf("lw: ERROR: %s: %s=0\n", CLK_NAME(this), 
            (inputRefClkDivVal == 0 ? "refClkDiv" : (mdiv == 0 ? "mdiv" :" pdiv")) );
        dprintf("lw: %s: (REFCLK) Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->refdivRegAddr, refRegData);
        dprintf("lw: %s: (COEFF)  Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->coeffRegAddr, coeffRegData);
        dprintf("lw: %s: (LUTSTATUS) Addr=0x%08x Data=0x%08x\n", CLK_NAME(this), this->lutStatusRegAddr, lutStatusRegData);
        *pFreqKHz = 0;
        return;
    }

    // Everything looks good, refclk_divider is 1 more than the field indicates
    inputRefClkDivVal += 1;

    *pFreqKHz =((inputRefClkFreqMHz * ndiv) / (mdiv * inputRefClkDivVal * pdiv));
    *pFreqKHz *= 1000;
    
    dprintf("lw: %s: RefClk-Freq=%uKHz RefClk-div=%u ndiv=%u mdiv=%u pdiv=%u \n", CLK_NAME(this), inputRefClkFreqMHz, inputRefClkDivVal, ndiv, mdiv, pdiv);
    dprintf("lw: %s: Freq=%uKHz override=%s\n", CLK_NAME(this), *pFreqKHz, modeString);
}
