/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file
 * @see         https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @author      Daniel Worpell
 * @author      Eric Colter
 * @author      Antone Vogt-Varvak
 *
 * @details     The main purpose of this file is to declare and define the
 *              schematic dag for chips which
 *              - Are Hopper and later (using SWDIVs),
 *              - Are displayless, and
 *              - Use HBM (High-Bandwidth Memory).
 *
 *              More generally, everything family-specific in Clocks 3.x goes
 *              in this file.  Nothing else should contain chip-specific data
 *              or logic since these data constructors contain flags to control
 *              specific behaviour.
 *
 *              In general, the schematic dag is declared and defined in
 *              "clk3/sd/clksd[chip].c", where [chip] is the first chip of
 *              the series of chips using that schematic.
 *
 * @warning     After modifying this file, please check it by running:
 *              //sw/dev/gpu_drv/chips_a/pmu_sw/prod_app/clk/clk3/clksdcheck.c
 *              See source code comments for instructions and information.
 */

/***************************************************************
 * Includes
 ***************************************************************/
// Frequency Source Objects
#include "clk/fs/clkwire.h"
#include "clk/fs/clkpll.h"
#include "clk/fs/clkldivv2.h"
#include "clk/fs/clkmux.h"
#include "clk/fs/clkpdiv.h"
#include "clk/fs/clkxtal.h"
#include "clk/fs/clknafll.h"
#include "clk/fs/clkbif.h"

// Basic clocks stuff
#include "print.h"
#include "ctrl/ctrl2080/ctrl2080clkavfs.h"

// The manual
#include "lwctassert.h"
#include "hopper/gh100/dev_fuse.h"
#include "hopper/gh100/dev_fbpa.h"
#include "hopper/gh100/dev_fbpa_addendum.h"
#include "hopper/gh100/dev_top.h"
#include "hopper/gh100/dev_trim.h"
#include "hopper/gh100/dev_trim_addendum.h"
#include "hopper/gh100/hwproject.h"
#include "clk/generic_dev_trim.h"


/***************************************************************
 * Feature Check
 ***************************************************************/


/***************************************************************
 * Schematic Structure
 ***************************************************************/

/*!
 * @brief       Schematic structure for this litter
 * @see         clkSchematicDag
 * @see         clkFreqDomainArray
 * @see         clkConstruct_SchematicDag
 * @private
 *
 * @details     Why isn't this in the header file?  Because this structure is
 *              entirely private.  There is only one instance of this struct,
 *              'clkSchematicDag'.  Nothing (other than initialization in this
 *              source file) references any of the symbols defined herein.
 *
 *              All other logic accesses these fields via 'clkFreqDomainArray'
 *              and/or similar pointers.
 *
 *              Why is this struct called 'ClkSchematicDag' instead of
 *              'ClkSchematicDag_[chip]'?  Because PMU builds are per-litter
 *              and there is one of these structures per litter.
 */
typedef struct
{   
    //
    // PCIE Gen (BIF)
    //
    struct
    {
        ClkWire   domain; 
        struct
        {
            ClkFreqSrc super;
        } bif;
    } pciegenclk;
    
    struct 
    {
        ClkWire                         domain;
        ClkNafll                        nafll[LW_PTRIM_GPC_GPCNAFLL_CFG1__SIZE_1];
    } gpcclk;
    
    // SYSCLK, XBARCLK, LWDCLK 
    struct 
    {
        ClkWire                         domain;
        ClkNafll                        nafll;
    } sysclk, xbarclk, lwdclk;
    
    // MCLK on displayless GH100 is for HBM memory (not GDDR/SDDR).
    struct
    {
        ClkWire                         domain;
        ClkFieldValue                   valueMap[2];
        ClkMux                          mainMux;    // Main multiplexer
        ClkFreqSrc                     *input[2];   // Main mux inputs

        struct
        {
            ClkPDiv                   pdiv;         // mclk.hbmpll.pdiv
            ClkPll                      vco;        // mclk.hbmpll.vco
        } hbmpll;
        
        struct
        {
            ClkFieldValue               valueMap[4];            
            ClkMux                      mux;        // mclk.swdiv.mux
            ClkFreqSrc                 *input[4];   // mclk.swdiv.input
            ClkLdivV2                   div2;       // mclk.swdiv.div2
            ClkLdivV2                   div3;       // mclk.swdiv.div3
        } swdiv;
    } mclk;

    //
    // Read-Only Domains
    //
    struct
    {
        ClkWire                         domain;
        struct
        {
            ClkFieldValue               valueMap[4];            
            ClkLdivV2                   div2;
            ClkLdivV2                   div3;
            ClkMux                      mux;
            ClkFreqSrc*                 input[4];
        } swdiv;
    } utilsclk;

    struct
    {
        ClkWire                         domain;
        struct
        {
            ClkFieldValue               valueMap[4];            
            ClkLdivV2                   div2;
            ClkLdivV2                   div3;
            ClkMux                      mux;
            ClkFreqSrc*                 input[4];
        } swdiv;
    } pwrclk;

    //
    // Frequency Source Objects Shared Among Domains
    //
    struct
    {
        ClkWire                         readonly;   // defrost.readonly
        ClkPll                          vco;        // defrost.pll
    } defrost;                                      // Sometimes called XTAL4X

    ClkXtal                             xtal;       // xtal

} ClkSchematicDag_GH100;


/***************************************************************
 * Schematic DAG
 ***************************************************************/

/*!
 * @brief   The schematic itself
 *
 * @details The type of each object this structure is a struct whose name is
 *          'Clk[leafname]' where [leafname] is the base name of a leaf
 *          class such as 'Pll', 'Mux', 'ClkMclkFreqDomain' etc.
 *
 *          For the most part, these data do not change after initialization.
 *          Much of these data are initialized at compile-time.
 */
ClkSchematicDag_GH100 clkSchematicDag_GH100;

//
// TODO: Revert below function to dot-assignment in the initializer once all
//       build elwironemnts are updated to at least C99.
//
/*!
 * @brief   Assigns values to schematic dag
 *
 * @details This function is a WAR, hopefully temporarily, which assigns 
 *          schematic dag fields. The reason this function is needed is
 *          because not all build elwironments are working at the C99
 *          standard, which is the first version which allows for dot
 *          assignment of struct fields. Normally, these values are assigned
 *          at compile time using dot assignment in the initialzier above.
 *          However, because not all build elwironments in use C99, we use
 *          static function to assign these values as opposed to using 
 *          positional assignment in the initializer. 
 */

static void clkConstructSchematicDag_GH100()
{
    /* --------------------- MUX VALUE MAP DEFINITIONS --------------------- */

    //
    // mclk_swdiv MuxValueMap
    //
    clkSchematicDag_GH100.mclk.swdiv.valueMap[0].mask    = DRF_SHIFTMASK(LW_PTRIM_SYS_SWDIV4_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.mclk.swdiv.valueMap[0].value   = DRF_DEF(_PTRIM, _SYS_SWDIV4, _CLOCK_SOURCE_SEL, _SOURCE(0));

    clkSchematicDag_GH100.mclk.swdiv.valueMap[1].mask    = DRF_SHIFTMASK(LW_PTRIM_SYS_SWDIV4_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.mclk.swdiv.valueMap[1].value   = DRF_DEF(_PTRIM, _SYS_SWDIV4, _CLOCK_SOURCE_SEL, _SOURCE(1));

    clkSchematicDag_GH100.mclk.swdiv.valueMap[2].mask    = DRF_SHIFTMASK(LW_PTRIM_SYS_SWDIV4_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.mclk.swdiv.valueMap[2].value   = DRF_DEF(_PTRIM, _SYS_SWDIV4, _CLOCK_SOURCE_SEL, _SOURCE(2));

    clkSchematicDag_GH100.mclk.swdiv.valueMap[3].mask    = DRF_SHIFTMASK(LW_PTRIM_SYS_SWDIV4_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.mclk.swdiv.valueMap[3].value   = DRF_DEF(_PTRIM, _SYS_SWDIV4, _CLOCK_SOURCE_SEL, _SOURCE(3));
    
    //
    // mclk_mainMux MuxValueMap
    //
    // 0 bypasspll=1 BYPASSCLK
    clkSchematicDag_GH100.mclk.valueMap[0].mask           = DRF_SHIFTMASK(LW_PFB_FBPA_HBMPLL_CFG_BYPASSPLL);
    clkSchematicDag_GH100.mclk.valueMap[0].value          = DRF_SHIFTMASK(LW_PFB_FBPA_HBMPLL_CFG_BYPASSPLL) &
                                                                DRF_DEF(_PFB, _FBPA_HBMPLL_CFG, _BYPASSPLL, _BYPASSCLK);

    // 1 bypasspll=0 VCO
    clkSchematicDag_GH100.mclk.valueMap[1].mask           = DRF_SHIFTMASK(LW_PFB_FBPA_HBMPLL_CFG_BYPASSPLL);
    clkSchematicDag_GH100.mclk.valueMap[1].value          = DRF_SHIFTMASK(LW_PFB_FBPA_HBMPLL_CFG_BYPASSPLL) &
                                                                 DRF_DEF(_PFB, _FBPA_HBMPLL_CFG, _BYPASSPLL, _VCO);

    //
    // pwrclk MuxValueMap
    //
    // 0: xtal 27MHz
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[0].mask   = DRF_SHIFTMASK(LW_PTRIM_SYS_PWRCLK_OUT_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[0].value  = DRF_DEF(_PTRIM, _SYS_PWRCLK_OUT_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _XTAL_0);

    // 1: xtal 27MHz w/o BYPASS FSM
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[1].mask   = DRF_SHIFTMASK(LW_PTRIM_SYS_PWRCLK_OUT_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[1].value  = DRF_DEF(_PTRIM, _SYS_PWRCLK_OUT_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _XTAL_1);

    // 2: defrost divider w/o BYPASS_FSM (automatic)
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[2].mask   = DRF_SHIFTMASK(LW_PTRIM_SYS_PWRCLK_OUT_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[2].value  = DRF_DEF(_PTRIM, _SYS_PWRCLK_OUT_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _DEFROST_CLK);

    // 3: XTAL divider w/o BYPASS_FSM
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[3].mask   = DRF_SHIFTMASK(LW_PTRIM_SYS_PWRCLK_OUT_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.pwrclk.swdiv.valueMap[3].value  = DRF_DEF(_PTRIM, _SYS_PWRCLK_OUT_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _XTAL_PLL_REFCLK);
    //
    // utilsclk MuxValueMap
    //
    // 0: xtal 27MHz w/o divider
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[0].mask  = DRF_SHIFTMASK(LW_PTRIM_SYS_UNIV_SEC_CLK_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[0].value = DRF_DEF(_PTRIM, _SYS_UNIV_SEC_CLK_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _XTAL_PLL_REFCLK_0);

    // 1: xtal 27MHz w/o divider
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[1].mask  = DRF_SHIFTMASK(LW_PTRIM_SYS_UNIV_SEC_CLK_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[1].value = DRF_DEF(_PTRIM, _SYS_UNIV_SEC_CLK_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _XTAL_PLL_REFCLK_1);

    // 2: defrost with divider
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[2].mask  = DRF_SHIFTMASK(LW_PTRIM_SYS_UNIV_SEC_CLK_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[2].value = DRF_DEF(_PTRIM, _SYS_UNIV_SEC_CLK_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _DEFROST_CLK);

    // 3: XTAL with divider
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[3].mask  = DRF_SHIFTMASK(LW_PTRIM_SYS_UNIV_SEC_CLK_SWITCH_DIVIDER_CLOCK_SOURCE_SEL);
    clkSchematicDag_GH100.utilsclk.swdiv.valueMap[3].value = DRF_DEF(_PTRIM, _SYS_UNIV_SEC_CLK_SWITCH_DIVIDER, _CLOCK_SOURCE_SEL, _XTAL_PLL_REFCLK_2);
   
    /* ------------------------ UTILSCLK ----------------------------------- */
    /*
     *                                     |\
     *                                     | \
     *       xtal ----- 0 -----------------|0 |
     *                                     |  |
     * pex refclk ----- 1 -----------------|1 |
     *                          ______     |  |
     *                         |      |    |  |
     *    defrost ----- 2 -----| DIV  |----|2 |----> OUTPUT
     *                         |______|    |  |
     *                                     |  |
     *                          ______     |  |
     *                         |      |    |  |
     *       xtal ----- 3 -----| DIV  |----|3 |
     *                         |______|    | /
     *                                     |/
     *
     */

    clkSchematicDag_GH100.utilsclk.domain.super.pVirtual              = &clkVirtual_Wire;
    clkSchematicDag_GH100.utilsclk.domain.super.name                  = "utilsclk";
    clkSchematicDag_GH100.utilsclk.domain.pInput                      = &clkSchematicDag_GH100.utilsclk.swdiv.mux.super;

    clkSchematicDag_GH100.utilsclk.swdiv.mux.super.pVirtual           = &clkVirtual_Mux;
    clkSchematicDag_GH100.utilsclk.swdiv.mux.super.name               = "utilsclk.swdiv.mux";
    clkSchematicDag_GH100.utilsclk.swdiv.mux.muxRegAddr               = LW_PTRIM_SYS_UNIV_SEC_CLK_SWITCH_DIVIDER;
    clkSchematicDag_GH100.utilsclk.swdiv.mux.muxValueMap              = clkSchematicDag_GH100.utilsclk.swdiv.valueMap;
    clkSchematicDag_GH100.utilsclk.swdiv.mux.input                    = clkSchematicDag_GH100.utilsclk.swdiv.input;
    clkSchematicDag_GH100.utilsclk.swdiv.mux.count                    = LW_ARRAY_ELEMENTS(clkSchematicDag_GH100.utilsclk.swdiv.input);
//  .utilsclk.swdiv.mux.glitchy                  = LW_FALSE, glitchy field not useful in LwWatch

    clkSchematicDag_GH100.utilsclk.swdiv.input[0u]                    = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.utilsclk.swdiv.input[1u]                    = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.utilsclk.swdiv.input[2u]                    = &clkSchematicDag_GH100.utilsclk.swdiv.div2.super.super;
    clkSchematicDag_GH100.utilsclk.swdiv.input[3u]                    = &clkSchematicDag_GH100.utilsclk.swdiv.div3.super.super;

    clkSchematicDag_GH100.utilsclk.swdiv.div2.super.super.pVirtual    = &clkVirtual_LdivV2;
    clkSchematicDag_GH100.utilsclk.swdiv.div2.super.pInput            = &clkSchematicDag_GH100.defrost.readonly.super;
    clkSchematicDag_GH100.utilsclk.swdiv.div2.ldivRegAddr             = LW_PTRIM_SYS_UNIV_SEC_CLK_SWITCH_DIVIDER;
    clkSchematicDag_GH100.utilsclk.swdiv.div2.super.super.name        = "utilsclk.swdiv.div2";

    clkSchematicDag_GH100.utilsclk.swdiv.div3.super.super.pVirtual    = &clkVirtual_LdivV2;
    clkSchematicDag_GH100.utilsclk.swdiv.div3.super.pInput            = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.utilsclk.swdiv.div3.ldivRegAddr             = LW_PTRIM_SYS_UNIV_SEC_CLK_SWITCH_DIVIDER;
    clkSchematicDag_GH100.utilsclk.swdiv.div3.super.super.name        = "utilsclk.swdiv.div3";



    /* ------------------------ PWRCLK ------------------------------------- */
    /*
     *
     *                                   |\
     *                                   | \
     *     xtal ----- 0 -----------------|0 |
     *                                   |  |
     *     xtal ----- 1 -----------------|1 |
     *                        ______     |  |
     *                       |      |    |  |
     *  defrost ----- 2 -----| DIV  |----|2 |----> OUTPUT
     *                       |______|    |  |
     *                                   |  |
     *                        ______     |  |
     *                       |      |    |  |
     *     xtal ----- 3 -----| DIV  |----|3 |
     *                       |______|    | /
     *                                   |/
     *
     */

    clkSchematicDag_GH100.pwrclk.domain.super.pVirtual                = &clkVirtual_Wire;
    clkSchematicDag_GH100.pwrclk.domain.super.name                    = "pwrclk";
    clkSchematicDag_GH100.pwrclk.domain.pInput                        = &clkSchematicDag_GH100.pwrclk.swdiv.mux.super;

    clkSchematicDag_GH100.pwrclk.swdiv.div2.super.super.pVirtual      = &clkVirtual_LdivV2;
    clkSchematicDag_GH100.pwrclk.swdiv.div2.super.pInput              = &clkSchematicDag_GH100.defrost.readonly.super;
    clkSchematicDag_GH100.pwrclk.swdiv.div2.ldivRegAddr               = LW_PTRIM_SYS_PWRCLK_OUT_SWITCH_DIVIDER;
    clkSchematicDag_GH100.pwrclk.swdiv.div2.super.super.name          = "pwrclk.swdiv.div2";

    clkSchematicDag_GH100.pwrclk.swdiv.div3.super.super.pVirtual      = &clkVirtual_LdivV2;
    clkSchematicDag_GH100.pwrclk.swdiv.div3.super.pInput              = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.pwrclk.swdiv.div3.ldivRegAddr               = LW_PTRIM_SYS_PWRCLK_OUT_SWITCH_DIVIDER;
    clkSchematicDag_GH100.pwrclk.swdiv.div3.super.super.name          = "pwrclk.swdiv.div3";

    clkSchematicDag_GH100.pwrclk.swdiv.mux.super.pVirtual             = &clkVirtual_Mux;
    clkSchematicDag_GH100.pwrclk.swdiv.mux.muxRegAddr                 = LW_PTRIM_SYS_PWRCLK_OUT_SWITCH_DIVIDER;
    clkSchematicDag_GH100.pwrclk.swdiv.mux.muxValueMap                = clkSchematicDag_GH100.pwrclk.swdiv.valueMap;
    clkSchematicDag_GH100.pwrclk.swdiv.mux.input                      = clkSchematicDag_GH100.pwrclk.swdiv.input;
    clkSchematicDag_GH100.pwrclk.swdiv.mux.count                      = LW_ARRAY_ELEMENTS(clkSchematicDag_GH100.pwrclk.swdiv.input);
//  .pwrclk.swdiv.mux.glitchy                    = LW_FALSE, glitchy field not useful on LwWatch
    clkSchematicDag_GH100.pwrclk.swdiv.mux.super.name                 = "pwrclk.swdiv.mux";


    clkSchematicDag_GH100.pwrclk.swdiv.input[0u]                      = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.pwrclk.swdiv.input[1u]                      = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.pwrclk.swdiv.input[2u]                      = &clkSchematicDag_GH100.pwrclk.swdiv.div2.super.super;
    clkSchematicDag_GH100.pwrclk.swdiv.input[3u]                      = &clkSchematicDag_GH100.pwrclk.swdiv.div3.super.super;

    /* ------------------------ PCIEGENCLK ---------------------------------- */

    clkSchematicDag_GH100.pciegenclk.domain.super.pVirtual           = &clkVirtual_Wire;
    clkSchematicDag_GH100.pciegenclk.domain.super.name               = "pciegenclk";
    clkSchematicDag_GH100.pciegenclk.domain.pInput                   = &clkSchematicDag_GH100.pciegenclk.bif.super;
    clkSchematicDag_GH100.pciegenclk.bif.super.pVirtual              = &clkVirtual_Bif;
    clkSchematicDag_GH100.pciegenclk.bif.super.name                  = "pciegenclk.bif";

    /* ------------------------ GPCCLK ------------------------------------- */

    clkSchematicDag_GH100.gpcclk.domain.super.pVirtual                = &clkVirtual_Wire;
    clkSchematicDag_GH100.gpcclk.domain.super.name                    = "gpcclk";
    clkSchematicDag_GH100.gpcclk.domain.pInput                        = &clkSchematicDag_GH100.gpcclk.nafll[0].super,   // Runtime override for each NAFLL source

    clkSchematicDag_GH100.gpcclk.nafll[0].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC0;
    clkSchematicDag_GH100.gpcclk.nafll[0].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[0].super.name                 = "gpcclk.nafll.0";
    clkSchematicDag_GH100.gpcclk.nafll[0].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(0);
    clkSchematicDag_GH100.gpcclk.nafll[0].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(0);
    clkSchematicDag_GH100.gpcclk.nafll[0].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(0);
    clkSchematicDag_GH100.gpcclk.nafll[0].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(0);
    
    clkSchematicDag_GH100.gpcclk.nafll[1].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC1;
    clkSchematicDag_GH100.gpcclk.nafll[1].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[1].super.name                 = "gpcclk.nafll.1";
    clkSchematicDag_GH100.gpcclk.nafll[1].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(1);
    clkSchematicDag_GH100.gpcclk.nafll[1].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(1);
    clkSchematicDag_GH100.gpcclk.nafll[1].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(1);
    clkSchematicDag_GH100.gpcclk.nafll[1].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(1);
    
    clkSchematicDag_GH100.gpcclk.nafll[2].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC2;
    clkSchematicDag_GH100.gpcclk.nafll[2].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[2].super.name                 = "gpcclk.nafll.2";
    clkSchematicDag_GH100.gpcclk.nafll[2].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(2);
    clkSchematicDag_GH100.gpcclk.nafll[2].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(2);
    clkSchematicDag_GH100.gpcclk.nafll[2].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(2);
    clkSchematicDag_GH100.gpcclk.nafll[2].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(2);
    
    clkSchematicDag_GH100.gpcclk.nafll[3].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC3;
    clkSchematicDag_GH100.gpcclk.nafll[3].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[3].super.name                 = "gpcclk.nafll.3";
    clkSchematicDag_GH100.gpcclk.nafll[3].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(3);
    clkSchematicDag_GH100.gpcclk.nafll[3].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(3);
    clkSchematicDag_GH100.gpcclk.nafll[3].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(3);
    clkSchematicDag_GH100.gpcclk.nafll[3].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(3);
    
    clkSchematicDag_GH100.gpcclk.nafll[4].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC4;
    clkSchematicDag_GH100.gpcclk.nafll[4].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[4].super.name                 = "gpcclk.nafll.4";
    clkSchematicDag_GH100.gpcclk.nafll[4].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(4);
    clkSchematicDag_GH100.gpcclk.nafll[4].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(4);
    clkSchematicDag_GH100.gpcclk.nafll[4].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(4);
    clkSchematicDag_GH100.gpcclk.nafll[4].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(4);
    
    clkSchematicDag_GH100.gpcclk.nafll[5].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC5;
    clkSchematicDag_GH100.gpcclk.nafll[5].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[5].super.name                 = "gpcclk.nafll.5";
    clkSchematicDag_GH100.gpcclk.nafll[5].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(5);
    clkSchematicDag_GH100.gpcclk.nafll[5].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(5);
    clkSchematicDag_GH100.gpcclk.nafll[5].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(5);
    clkSchematicDag_GH100.gpcclk.nafll[5].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(5);
    
    clkSchematicDag_GH100.gpcclk.nafll[6].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC6;
    clkSchematicDag_GH100.gpcclk.nafll[6].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[6].super.name                 = "gpcclk.nafll.6";
    clkSchematicDag_GH100.gpcclk.nafll[6].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(6);
    clkSchematicDag_GH100.gpcclk.nafll[6].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(6);
    clkSchematicDag_GH100.gpcclk.nafll[6].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(6);
    clkSchematicDag_GH100.gpcclk.nafll[6].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(6);
    
    clkSchematicDag_GH100.gpcclk.nafll[7].id                         = LW2080_CTRL_CLK_NAFLL_ID_GPC7;
    clkSchematicDag_GH100.gpcclk.nafll[7].super.pVirtual             = &clkVirtual_Nafll;
    clkSchematicDag_GH100.gpcclk.nafll[7].super.name                 = "gpcclk.nafll.7";
    clkSchematicDag_GH100.gpcclk.nafll[7].coeffRegAddr               = LW_PTRIM_GPC_GPCNAFLL_COEFF(7);
    clkSchematicDag_GH100.gpcclk.nafll[7].swfreqRegAddr              = LW_PTRIM_GPC_GPCLUT_SW_FREQ_REQ(7);
    clkSchematicDag_GH100.gpcclk.nafll[7].lutStatusRegAddr           = LW_PTRIM_GPC_GPCLUT_STATUS(7);
    clkSchematicDag_GH100.gpcclk.nafll[7].refdivRegAddr              = LW_PTRIM_GPC_CLK_SRC_CONTROL(7);

    /* ------------------------ SYSCLK ------------------------------------- */

    clkSchematicDag_GH100.sysclk.domain.super.pVirtual                = &clkVirtual_Wire;
    clkSchematicDag_GH100.sysclk.domain.super.name                    = "sysclk";
    clkSchematicDag_GH100.sysclk.domain.pInput                        = &clkSchematicDag_GH100.sysclk.nafll.super;

    clkSchematicDag_GH100.sysclk.nafll.id                            = LW2080_CTRL_CLK_NAFLL_ID_SYS;
    clkSchematicDag_GH100.sysclk.nafll.super.pVirtual                = &clkVirtual_Nafll;
    clkSchematicDag_GH100.sysclk.nafll.super.name                    = "sysclk.nafll";
    clkSchematicDag_GH100.sysclk.nafll.coeffRegAddr                  = LW_PTRIM_SYS_NAFLL_SYSNAFLL_COEFF;
    clkSchematicDag_GH100.sysclk.nafll.swfreqRegAddr                 = LW_PTRIM_SYS_NAFLL_SYSLUT_SW_FREQ_REQ;
    clkSchematicDag_GH100.sysclk.nafll.lutStatusRegAddr              = LW_PTRIM_SYS_NAFLL_SYSLUT_STATUS;
    clkSchematicDag_GH100.sysclk.nafll.refdivRegAddr                 = LW_PTRIM_SYS_AVFS_REFCLK_CORE_CONTROL;
    
    /* ------------------------ XBARCLK ------------------------------------- */

    clkSchematicDag_GH100.xbarclk.domain.super.pVirtual               = &clkVirtual_Wire;
    clkSchematicDag_GH100.xbarclk.domain.super.name                   = "xbarclk";
    clkSchematicDag_GH100.xbarclk.domain.pInput                       = &clkSchematicDag_GH100.xbarclk.nafll.super;

    clkSchematicDag_GH100.xbarclk.nafll.id                           = LW2080_CTRL_CLK_NAFLL_ID_XBAR;
    clkSchematicDag_GH100.xbarclk.nafll.super.pVirtual               = &clkVirtual_Nafll;
    clkSchematicDag_GH100.xbarclk.nafll.super.name                   = "xbarclk.nafll";
    clkSchematicDag_GH100.xbarclk.nafll.coeffRegAddr                 = LW_PTRIM_SYS_NAFLL_XBARNAFLL_COEFF;
    clkSchematicDag_GH100.xbarclk.nafll.swfreqRegAddr                = LW_PTRIM_SYS_NAFLL_XBARLUT_SW_FREQ_REQ;
    clkSchematicDag_GH100.xbarclk.nafll.lutStatusRegAddr             = LW_PTRIM_SYS_NAFLL_XBARLUT_STATUS;
    clkSchematicDag_GH100.xbarclk.nafll.refdivRegAddr                = LW_PTRIM_SYS_AVFS_REFCLK_CORE_CONTROL;

    /* ------------------------ LWDCLK ------------------------------------- */

    clkSchematicDag_GH100.lwdclk.domain.super.pVirtual                = &clkVirtual_Wire;
    clkSchematicDag_GH100.lwdclk.domain.super.name                    = "lwdclk";
    clkSchematicDag_GH100.lwdclk.domain.pInput                         = &clkSchematicDag_GH100.lwdclk.nafll.super;

    clkSchematicDag_GH100.lwdclk.nafll.id                            = LW2080_CTRL_CLK_NAFLL_ID_LWD;
    clkSchematicDag_GH100.lwdclk.nafll.super.pVirtual                = &clkVirtual_Nafll;
    clkSchematicDag_GH100.lwdclk.nafll.super.name                    = "lwdclk.nafll";
    clkSchematicDag_GH100.lwdclk.nafll.coeffRegAddr                  = LW_PTRIM_SYS_NAFLL_LWDNAFLL_COEFF;
    clkSchematicDag_GH100.lwdclk.nafll.swfreqRegAddr                 = LW_PTRIM_SYS_NAFLL_LWDLUT_SW_FREQ_REQ;
    clkSchematicDag_GH100.lwdclk.nafll.lutStatusRegAddr              = LW_PTRIM_SYS_NAFLL_LWDLUT_STATUS;
    clkSchematicDag_GH100.lwdclk.nafll.refdivRegAddr                 = LW_PTRIM_SYS_AVFS_REFCLK_CORE_CONTROL;

    /* ------------------------ MCLK --------------------------------------- */
    /*
     * The block diagram of schema defined for HBM (high-bandwisth memory) MCLK:
     *                  ____________
     *                 |     |      |
     *      XTAL ------| VCO | PDIV |------+
     *                 |_____|______|      |        ______
     *                     HPMBLL          |       |      | bypass in
     *                                     +-------|1     | PLL register
     *                                             |      |
     *                                  ______     |      |
     *      XTAL ------------------+   |      |    | SW2  |---> OUTPUT
     *               _______       +---|0     |    |      |
     *              |       |      |   |      |    |      |
     *   DEFROST ---| LDIV2 |---+  +---|1     |    |      |
     *              |_______|   |      |  SW4 |----|0     |
     *               _______    +------|2     |    |______|
     *              |       |          |      |
     *      XTAL ---| LDIV3 |----------|3     |
     *              |_______|          |______|
     *           \_____________________________/
     *                        SWDIV
     *
     * Per bug 2622250, we must use unicast registers for FBIO instead of broadcast.
     * As such, clkConstruct_SchematicDag sets the addresses based on floorsweeping.
     * See also bug 2369474.
     *
     * Starting with Hopper, SWDIVs (switch dividers) have a finite state machine
     * that orders the programming of the switch and dividers.  One feature of
     * the FSM is that ldiv2 and ldiv3 have the same divider value controlled
     * by the same register and field.  See bug 2991907/55 and in Perforce at
     * hw/doc/gpu/SOC/Clocks/Documentation/POR/GH100/Divider_switch.docx
     *
     * HBMCLK/HBMPLL are read-only.
     */
    clkSchematicDag_GH100.mclk.domain.super.pVirtual                 = &clkVirtual_Wire;
    clkSchematicDag_GH100.mclk.domain.super.name                     = "mclk";
    clkSchematicDag_GH100.mclk.domain.pInput                         = &clkSchematicDag_GH100.mclk.mainMux.super;

    clkSchematicDag_GH100.mclk.input[0]                              = &clkSchematicDag_GH100.mclk.swdiv.mux.super;
    clkSchematicDag_GH100.mclk.input[1]                              = &clkSchematicDag_GH100.mclk.hbmpll.pdiv.super.super;
                                                
    clkSchematicDag_GH100.mclk.mainMux.super.pVirtual                = &clkVirtual_Mux;
//  .mclk.mainMux.muxRegAddr                    = LW_PFB_FBPA_i_FBIO_HBMPLL_CFG,    Based on floorsweeping
    clkSchematicDag_GH100.mclk.mainMux.muxValueMap                   = clkSchematicDag_GH100.mclk.valueMap,    // Either DDR or HBM
    clkSchematicDag_GH100.mclk.mainMux.input                         = clkSchematicDag_GH100.mclk.input;
    clkSchematicDag_GH100.mclk.mainMux.count                         = LW_ARRAY_ELEMENTS(clkSchematicDag_GH100.mclk.input),            // Either DDR or HBM
//  .mclk.mainMux.glitchy                       = LW_FALSE, glitchy field not useful on LwWatch
    clkSchematicDag_GH100.mclk.mainMux.super.name                    = "mclk.mainMux.name";

    clkSchematicDag_GH100.mclk.hbmpll.pdiv.super.super.pVirtual      = &clkVirtual_PDiv;
    clkSchematicDag_GH100.mclk.hbmpll.pdiv.super.pInput              = &clkSchematicDag_GH100.mclk.hbmpll.vco.super.super;
    clkSchematicDag_GH100.mclk.hbmpll.pdiv.super.super.name          = "mclk.hbmpll.pdiv";
//  .mclk.hbmpll.pdiv.regAddr                   = LW_PFB_FBPA_i_FBIO_COEFF_CFG,    Based on floorsweeping
    clkSchematicDag_GH100.mclk.hbmpll.pdiv.base                      = DRF_BASE(LW_PFB_FBPA_FBIO_HBMPLL_COEFF_PLDIV);
    clkSchematicDag_GH100.mclk.hbmpll.pdiv.size                      = DRF_SIZE(LW_PFB_FBPA_FBIO_HBMPLL_COEFF_PLDIV);
    

    clkSchematicDag_GH100.mclk.hbmpll.vco.super.super.pVirtual       = &clkVirtual_Pll;
    clkSchematicDag_GH100.mclk.hbmpll.vco.super.pInput               = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.mclk.hbmpll.vco.super.super.name           = "mclk.hbmpll.vco";
//  .mclk.hbmpll.vco.cfgRegAddr                 = LW_PFB_FBPA_i_FBIO_HBMPLL_CFG,    Based on floorsweeping
//  .mclk.hbmpll.vco.coeffRegAddr               = LW_PFB_FBPA_i_FBIO_HBMPLL_COEFF,  Based on floorsweeping
//  .mclk.hbmpll.vco.source                     = LW2080_CTRL_CLK_PROG_1X_SOURCE_PLL, source not useful on LwWatch
    clkSchematicDag_GH100.mclk.hbmpll.vco.bDiv2Exists                = LW_FALSE;
    clkSchematicDag_GH100.mclk.hbmpll.vco.bPldivExists               = LW_FALSE;

    clkSchematicDag_GH100.mclk.swdiv.mux.super.pVirtual              = &clkVirtual_Mux;
    clkSchematicDag_GH100.mclk.swdiv.mux.super.name                  = "mclk.swdiv.mux";
    clkSchematicDag_GH100.mclk.swdiv.mux.muxRegAddr                  = LW_PTRIM_SYS_DRAMCLK_ALT_SWITCH_DIVIDER;
    clkSchematicDag_GH100.mclk.swdiv.mux.muxValueMap                 = clkSchematicDag_GH100.mclk.swdiv.valueMap;
    clkSchematicDag_GH100.mclk.swdiv.mux.input                       = clkSchematicDag_GH100.mclk.swdiv.input;
    clkSchematicDag_GH100.mclk.swdiv.mux.count                       = LW_ARRAY_ELEMENTS(clkSchematicDag_GH100.mclk.swdiv.input);
//  .mclk.swdiv.mux.glitchy                     = LW_FALSE, glitchy field not useful in LwWatch

    clkSchematicDag_GH100.mclk.swdiv.input[0]                        = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.mclk.swdiv.input[1]                        = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.mclk.swdiv.input[2]                        = &clkSchematicDag_GH100.mclk.swdiv.div2.super.super;
    clkSchematicDag_GH100.mclk.swdiv.input[3]                        = &clkSchematicDag_GH100.mclk.swdiv.div3.super.super;

    clkSchematicDag_GH100.mclk.swdiv.div2.super.super.pVirtual       = &clkVirtual_LdivV2;
    clkSchematicDag_GH100.mclk.swdiv.div2.super.pInput               = &clkSchematicDag_GH100.defrost.readonly.super;
    clkSchematicDag_GH100.mclk.swdiv.div2.super.super.name           = "mclk.swdiv.div2";
    clkSchematicDag_GH100.mclk.swdiv.div2.ldivRegAddr                = LW_PTRIM_SYS_DRAMCLK_ALT_SWITCH_DIVIDER;

    clkSchematicDag_GH100.mclk.swdiv.div3.super.super.pVirtual       = &clkVirtual_LdivV2;
    clkSchematicDag_GH100.mclk.swdiv.div3.super.pInput               = &clkSchematicDag_GH100.xtal.super;
    clkSchematicDag_GH100.mclk.swdiv.div3.super.super.name           = "mclk.swdiv.div3";
    clkSchematicDag_GH100.mclk.swdiv.div3.ldivRegAddr                = LW_PTRIM_SYS_DRAMCLK_ALT_SWITCH_DIVIDER;
    {
    //
    // Per bug 2622250, we must use unicast registers for FBIO instead of broadcast.
    // We may choose any nonfloorswept registers assuming that all FBIOs have
    // been programmed the same by the FbFalcon ucode.  See also bug 2369474.
    // This issue is specific to HBM since DDR does not use FBIO.
    // See wiki.lwpu.com/gpuhwvoltaindex.php/GV100_HBM_Bringup#Serial_priv_bus
    //
#if LW_FUSE_STATUS_OPT_FBIO_IDX_ENABLE != 0 || LW_FUSE_STATUS_OPT_FBIO_IDX_DISABLE != 1
#error Logic below uses the ~ operator under the assumption that LW_FUSE_STATUS_OPT_FBIO_IDX_ENABLE is zero.
#endif
        // Choose the lowest-orderded FBIO that has not been floorswept.
        LwU32 fbioStatusData     = GPU_REG_RD32(LW_FUSE_STATUS_OPT_FBIO);
        LwU32 fbioFloorsweptMask = DRF_VAL(_FUSE, _STATUS_OPT_FBIO, _DATA, fbioStatusData);
        LwU32 fbioValidMask      = LWBIT32(GPU_REG_RD32(LW_PTOP_SCAL_NUM_FBPAS)) - 1;
        LwU32 fbioActiveMask     = (~fbioFloorsweptMask) & fbioValidMask;
        LwU8  fbioToUse          = (LwU8) BIT_IDX_32(LOWESTBIT(fbioActiveMask));

        // Set the register addresses to use the chosen FBIO.
        clkSchematicDag_GH100.mclk.mainMux.muxRegAddr      = LW_PFB_FBPA_UC_FBIO_HBMPLL_CFG(fbioToUse);
        clkSchematicDag_GH100.mclk.hbmpll.pdiv.regAddr     =
        clkSchematicDag_GH100.mclk.hbmpll.vco.coeffRegAddr = LW_PFB_FBPA_UC_FBIO_HBMPLL_COEFF(fbioToUse);
    }

    /* ------------------------ DEFROST ------------------------------------- */

    //
    // The following assumptions apply about 'defrost':
    // - It is not programmed after initialization (read-only);
    // - Fractional NDIV is not being used; and
    // - PLDIV is not being used.
    //
    // NOTE: November 2019: Unlike GA10x, LW_PTRIM_SYS_XTAL4X_CFG contains
    // a _PLLOUT_DIV field which is not lwrrently POR.  If it becomes POR;
    // a ClkPDiv object (or similar) would be necessary.
    //
    // See hw/libs/common/analog/lwpu/doc/HPLL16G_SSD_DYN_A1.doc
    //
    // LW2080_CTRL_CLK_PROG_1X_SOURCE_PLL is used because DEFROST_CLK is the
    // only PLL for UTILSCLK and PWRCLK.  For these domains, there is no
    // "bypass" per se.  Moreover, DEFROST_CLK is not used in any other domain.
    //
    clkSchematicDag_GH100.defrost.readonly.super.pVirtual            = &clkVirtual_Wire;
    clkSchematicDag_GH100.defrost.readonly.pInput                    = &clkSchematicDag_GH100.defrost.vco.super.super;
    clkSchematicDag_GH100.defrost.readonly.super.name                = "defrost";

    clkSchematicDag_GH100.defrost.vco.super.super.pVirtual           = &clkVirtual_Pll;
    clkSchematicDag_GH100.defrost.vco.super.super.name               = "defrost.pll";
    clkSchematicDag_GH100.defrost.vco.super.pInput                   = &clkSchematicDag_GH100.xtal.super;
#ifdef LW_PTRIM_SYS_DEFROST_COEFF
    clkSchematicDag_GH100.defrost.vco.coeffRegAddr                   = LW_PTRIM_SYS_DEFROST_COEFF;
#else       // TODO: Delete once HW changes are checked in
    clkSchematicDag_GH100.defrost.vco.coeffRegAddr                   = LW_PTRIM_SYS_XTAL4X_COEFF;
#endif
//  .defrost.vco.source                         = LW2080_CTRL_CLK_PROG_1X_SOURCE_PLL, source not useful on LwWatch
    clkSchematicDag_GH100.defrost.vco.bPldivExists                   = LW_FALSE;
    clkSchematicDag_GH100.defrost.vco.bDiv2Exists                    = LW_FALSE;

    /* ------------------------ XTAL ---------------------------------------- */

    clkSchematicDag_GH100.xtal.super.pVirtual                        = &clkVirtual_Xtal;
    clkSchematicDag_GH100.xtal.super.name                            = "xtal";
    clkSchematicDag_GH100.xtal.freqKHz                               = 27000;
}

/*!
 * @brief       Table of all domains in the schematic
 *
 * @details     It is essential that C99 designated initializers are used here.
 *              See doxygen flowerbox for this array in 'clkdomain.h' for details.
 */
ClkFreqSrc *clkFreqSrcArray_GH100[] =
{
    &clkSchematicDag_GH100.utilsclk.domain.super,
    &clkSchematicDag_GH100.utilsclk.swdiv.mux.super,
    &clkSchematicDag_GH100.utilsclk.swdiv.div2.super.super,
    &clkSchematicDag_GH100.utilsclk.swdiv.div3.super.super,
    &clkSchematicDag_GH100.pwrclk.domain.super,
    &clkSchematicDag_GH100.pwrclk.swdiv.div2.super.super,
    &clkSchematicDag_GH100.pwrclk.swdiv.div3.super.super,
    &clkSchematicDag_GH100.pwrclk.swdiv.mux.super,
    &clkSchematicDag_GH100.pciegenclk.domain.super,
    &clkSchematicDag_GH100.pciegenclk.bif.super,
    &clkSchematicDag_GH100.gpcclk.domain.super,
    &clkSchematicDag_GH100.gpcclk.nafll[0].super,
    &clkSchematicDag_GH100.gpcclk.nafll[1].super,
    &clkSchematicDag_GH100.gpcclk.nafll[2].super,
    &clkSchematicDag_GH100.gpcclk.nafll[3].super,
    &clkSchematicDag_GH100.gpcclk.nafll[4].super,
    &clkSchematicDag_GH100.gpcclk.nafll[5].super,
    &clkSchematicDag_GH100.gpcclk.nafll[6].super,
    &clkSchematicDag_GH100.gpcclk.nafll[7].super,
    &clkSchematicDag_GH100.sysclk.domain.super,
    &clkSchematicDag_GH100.sysclk.nafll.super,
    &clkSchematicDag_GH100.xbarclk.domain.super,
    &clkSchematicDag_GH100.xbarclk.nafll.super,
    &clkSchematicDag_GH100.lwdclk.domain.super,
    &clkSchematicDag_GH100.lwdclk.nafll.super,
    &clkSchematicDag_GH100.mclk.domain.super,
    &clkSchematicDag_GH100.mclk.mainMux.super,
    &clkSchematicDag_GH100.mclk.hbmpll.pdiv.super.super,
    &clkSchematicDag_GH100.mclk.hbmpll.vco.super.super,
    &clkSchematicDag_GH100.mclk.swdiv.mux.super,
    &clkSchematicDag_GH100.mclk.swdiv.div2.super.super,
    &clkSchematicDag_GH100.mclk.swdiv.div3.super.super,
    &clkSchematicDag_GH100.defrost.readonly.super,
    &clkSchematicDag_GH100.defrost.vco.super.super,
    &clkSchematicDag_GH100.xtal.super,
    NULL
};

/*!
 * @brief   Returns an array of all FreqSrc objects in schematic dag
 *
 * @details This function returns the array of FreqSrc objects which
 *          are in the schematic dag. 
 *
 *          This function contains a static Boolean, which is meant
 *          to prevent the schematic dag from being built more than
 *          once. This Boolean can be removed once at the same time
 *          that clkBuildSchematicDag() is removed.
 *
 */
ClkFreqSrc **clkGetFreqSrcArray_GH100
(
)
{
    //
    // TODO: Remove these variables once Windows LwWatch build environment is
    // running at C99 and the clkBuildSchematicDag() function is removed
    //
    static LwBool schematicDagConstructed = LW_FALSE;

    if (schematicDagConstructed == LW_FALSE)
    {
        dprintf("Building Schematic dag for GH100\n");
        clkConstructSchematicDag_GH100();
        schematicDagConstructed = LW_TRUE;
    }
    return (ClkFreqSrc **) clkFreqSrcArray_GH100;
}
