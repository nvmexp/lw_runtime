#include "chip.h"
#include "elpg.h"
#include "pascal/gp104/dev_perf.h"
#include "pascal/gp104/dev_lw_xve.h"
#include "pascal/gp104/dev_sec_pri.h"
#include "disp/v02_01/dev_disp.h"
#include "pascal/gp104/dev_pwr_pri.h"
#include "pascal/gp104/dev_ihub.h"

#include "g_elpg_private.h"

// XVE Register Read in GPU Space
#define CFG_RD32(a)   ELPG_REG_RD32(DRF_BASE(LW_PCFG) + a)

//*****************************************************************************
// Static functions
//*****************************************************************************

/*!
 * @brief: Helper function to get the IdleSuppCtrlIndex, used by 
 *         PG-ENG(GR/MS/DI) for start/stop functionality
 *
 * @param[in]:  elpgId               Engine ID
 * @param[out]: *idleSuppCtrlIndex   Address to store the index
 *
 * @returns: LW_ERR_NOT_SUPPORTED  un-supported Engine/Null pointer received
 *           LW_OK                 otherwise
*/
LW_STATUS _elpgGetIdleSuppCtrlIdx_GP10X(LwU32 elpgId, LwU32 *idleSuppCtrlIndex)
{
    LW_STATUS status = LW_OK;
    if (idleSuppCtrlIndex == NULL)
    {
        dprintf("ERROR - Null ptr passed!\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    switch(elpgId)
    {
        // Needed to include the PG_IDLE_MASK_SUPP_IDX_* to be used with
        // LW_PPWR_PMU_IDLE_CTRL_SUPP register. Since the IDs are in PMU
        // code and not accessible to LwWatch, using hard-coded values for
        // the same. It is a potential bug.
        case ELPG_ENGINE_ID_GRAPHICS:
            *idleSuppCtrlIndex = 0;
            break;
        case ELPG_ENGINE_ID_MS:
            *idleSuppCtrlIndex = 4;
            break;
        case ELPG_ENGINE_ID_DI:
            *idleSuppCtrlIndex = 7;
            break;
        default:
            return LW_ERR_NOT_SUPPORTED;
    }
    return status;
}

/*!
 * @brief: Helper function to get the IdleSuppCtrlIndex, used by 
 *         PG-ENG(GR/MS/DI) for start/stop functionality
 *
 * @param[in]:  elpgId               Engine ID
 * @param[out]: *idleSuppCtrlIndex   Address to store the index
 *
 * @returns: LW_ERR_NOT_SUPPORTED  un-supported Engine/Null pointer received
 *           LW_OK                 otherwise
*/
LW_STATUS _elpgDisplayIdleCtrlSupp_GP10X(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 val;
    LwU32 regIdleCtrlSupp;
    LwU32 idleSuppCtrlIndex = 0;
    LW_STATUS status = LW_OK;

    _elpgGetIdleSuppCtrlIdx_GP10X(elpgId, &idleSuppCtrlIndex);
    //
    // LW_PPWR_PMU_IDLE_CTRL_SUPP denotes whether ELPG0 and ELPG4 have their
    // idle signal overridden for their associated engines.
    //
    regIdleCtrlSupp = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_CTRL_SUPP(idleSuppCtrlIndex));

    // Determine whether the idle signal is overridden.
    val = regIdleCtrlSupp & DRF_SHIFTMASK(LW_PPWR_PMU_IDLE_CTRL_SUPP_VALUE);
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_IDLE_CTRL_SUPP(%d), elpgId, val);

    //
    // NOTE that only "one" of the following prints will succeed since only
    // one state can be active at a time.
    //
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _NEVER , regIdleCtrlSupp, "NEVER");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _IDLE  , regIdleCtrlSupp, "IDLE");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _BUSY  , regIdleCtrlSupp, "BUSY");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _ALWAYS, regIdleCtrlSupp, "ALWAYS");
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIdleCtrlSupp;

    return status;
}

//*****************************************************************************
// Global functions
//*****************************************************************************

/*!
 * @brief: The HAL function to display the interrupt
 *         enablement status of supported PG_ENG.
 *
 * @param[in]     elpgID    PG-ENG engine ID
 * @param[out]    regVal    Pointer to grab the Lwrrently Enabled
 *                          Interrupts for PG-ENG(elpgID) (if needed)
 *
 * @returns LW_OK
 */
LW_STATUS elpgDisplayPgIntrEn_GP10X(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regIntrEn;
    LW_STATUS status = LW_OK;
    LwU32 val;

    //
    // LW_PPWR_PMU_ELPG_INTREN holds the interrupt enable bits for PG_ENG
    //
    regIntrEn = ELPG_REG_RD32(LW_PPWR_PMU_PG_INTREN(elpgId));

    // Determine which PG_ENG interrupts are enabled.
    val = regIntrEn & (DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_PG_ON)       |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_PG_ON_DONE)  |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_CTX_RESTORE) |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_CFG_ERR)     |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_ENG_RST)     |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_IDLE_SNAP));
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_INTREN(%d), elpgId, val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _PG_ON      , regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _PG_ON_DONE , regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _CTX_RESTORE, regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _CFG_ERR    , regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _ENG_RST    , regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _IDLE_SNAP  , regIntrEn, "ENABLED", "DISABLED");

    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntrEn;

    return status;
}

/*!
 * @brief: The HAL function to display the interrupt
 *         pending status to PG_ENG(GR/MS/DI)
 *
 * The following function checks if there is an interrupt pending for 
 * any PG_ENG(GR/MS/DI).
 * In case, some Interrupts are found pending, following actions are taken:
 *    1. For CTX_RESTORE interrupt, check the request source: RM/HOST
 *    2. For CFG_ERR, raise an error signal
 *    3. FOr IDLE_SNAP, dump the LW_PPWR_PMU_IDLE_STATUS/_1 registers
 *
 * @param[in]     elpgID    PG-ENG engine ID
 * @param[out]    regVal    Pointer to grab the Current Pending 
 *                          Interrupt state (if needed)
 *
 * @returns LW_ERR_GENERIC    in case of Config Error
 *          LW_OK             otherwise
 */
LW_STATUS elpgDisplayPgIntrStat_GP10X(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regPgoff;
    LwU32 regIntrStat;
    LwU32 regIdleSnap;
    LW_STATUS status = LW_OK;

    // Determine which of the enabled interrupts are pending, if any.
    regIntrStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_INTRSTAT(elpgId));
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_INTRSTAT(%d), elpgId, regIntrStat);

    if (regIntrStat == 0)
    {
        PRINT_INDENTED_STRING("NO INTERRUPTS PENDING");
    }
    else
    {
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_INTRSTAT, _INTR       , _SET, regIntrStat, "PENDING");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_INTRSTAT, _PG_ON      , _SET, regIntrStat, "PENDING");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_INTRSTAT, _PG_ON_DONE , _SET, regIntrStat, "PENDING");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_INTRSTAT, _CTX_RESTORE, _SET, regIntrStat, "PENDING");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_INTRSTAT, _CFG_ERR    , _SET, regIntrStat, "PENDING");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_INTRSTAT, _ENG_RST    , _SET, regIntrStat, "PENDING");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_INTRSTAT, _IDLE_SNAP  , _SET, regIntrStat, "PENDING");

        //
        // If a CTX_RESTORE interrupt is pending, determine whether the request
        // to turn off power-gating came from RM or HOST.
        //
        if (FLD_TEST_DRF(_PPWR, _PMU_PG_INTRSTAT, _CTX_RESTORE, _SET,
                         regIntrStat))
        {
            //
            // LW_PPWR_PMU_ELPG_PGOFF holds the power-gate control bits for
            // PG_ENG(0).
            //
            regPgoff = ELPG_REG_RD32(LW_PPWR_PMU_PG_PGOFF);
            if (FLD_IDX_TEST_DRF(_PPWR, _PMU_PG_PGOFF, _REQSRC, elpgId, _RM,
                                 regPgoff))
            {
                PRINT_NEWLINE;
                PRINT_INDENTED_STRING("_CTX_RESTORE request came from RM");
            }
            else if (
                FLD_IDX_TEST_DRF(_PPWR, _PMU_PG_PGOFF, _REQSRC, elpgId, _HOST,
                                 regPgoff))
            {
                PRINT_NEWLINE;
                PRINT_INDENTED_STRING("_CTX_RESTORE request came from HOST");
            }
        }

        //
        // If a CFG_ERR interrupt is pending, report an error as there is some
        // incorrect configuration done for PG-ENG(elpgId)
        //
        if (FLD_TEST_DRF(_PPWR, _PMU_PG_INTRSTAT, _CFG_ERR, _SET, regIntrStat))
        {
            PRINT_NEWLINE;
            PRINT_INDENTED_ERROR("_CFG_ERR set");
            status = LW_ERR_GENERIC;
        }

        if (FLD_TEST_DRF(_PPWR, _PMU_PG_INTRSTAT, _IDLE_SNAP, _SET, regIntrStat))
        {
            PRINT_NEWLINE;
            PRINT_INDENTED_STRING("_IDLE_SNAP set, Dumping IDLE_STATUS and IDLE_STATUS_1 registers");
            regIdleSnap = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS);
            dprintf("\t\tIDLE_STATUS   : 0x%x", regIdleSnap);
            regIdleSnap = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_1);
            dprintf("\t\tIDLE_STATUS_1 : 0x%x", regIdleSnap);
        }
    }
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntrStat;

    return status;
}

/*!
 * @brief: The HAL function to display the SM state of PG-ENG(GR/MS/DI)
 *
 * The following function prints the Engine State, 
 * as well as the HW Controller state of PG-ENG(elpgId)
 *
 * @param[in]     elpgID    PG-ENG engine ID
 * @param[out]    regVal    Pointer to grab the PG-Engine SM state (if needed)
 *
 * @returns LW_OK           operation successful.
 *          LW_ERR_GENERIC  unknown PG-Engine State
 */
LW_STATUS elpgDisplayPgStat_GP10X(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regStat;
    LW_STATUS status = LW_OK;
    LwU32 val;

    //
    // The LW_PPWR_PMU_PG_STAT register holds the engine Power Status
    // and the State of the actual internal PG-ENG HW controller.
    //
    regStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_STAT(elpgId));

    // Determine the current state of the PG_ENG state machine.
    val = regStat & DRF_SHIFTMASK(LW_PPWR_PMU_PG_STAT_EST);
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_STAT(%d), elpgId, val);

    if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _IDLE, regStat)         ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _POWERINGDOWN, regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE, regStat)       ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _POWERINGUP, regStat)   ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _PWROFF, regStat)       ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _PWROFF_ACTIVE, regStat))
    {
        //
        // NOTE that only "one" of the following prints will succeed since
        // only one state can be active at a time.
        //
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _IDLE          , regStat, "IDLE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _POWERINGDOWN  , regStat, "POWERINGDOWN");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE        , regStat, "ACTIVE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _POWERINGUP    , regStat, "POWERINGUP");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _PWROFF        , regStat, "PWROFF");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _PWROFF_ACTIVE , regStat, "PWROFF_ACTIVE");
    }
    else
    {
        // Report the unrecognized state as an error.
        PRINT_NEWLINE;
        PRINT_INDENTED_ERROR("Unknown LW_PPWR_PMU_PG_STAT_EST value");
        status = LW_ERR_GENERIC;
    }

    // Determine the current State of the internal PG-Eng HW controller.
    val = regStat & DRF_SHIFTMASK(LW_PPWR_PMU_PG_STAT_PGST);
    if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _IDLE           , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _PG_ON          , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_ON       , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_ON_DELAY , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _PWROFF_SEQ     , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _PG_ON_DONE     , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _PWRON_SEQ      , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _ENG_RST        , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_OFF      , regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_OFF_DELAY, regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _DAT_RESTORE    , regStat))
    {
        //
        // NOTE that only "one" of the following prints will succeed since
        // only one state can be active at a time.
        //
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _IDLE           , regStat, "IDLE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _PG_ON          , regStat, "PG_ON");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_ON       , regStat, "CLAMP ON");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_ON_DELAY , regStat, "CLAMP ON DELAY");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _PWROFF_SEQ     , regStat, "PWROFF SEQUENCE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _PG_ON_DONE     , regStat, "PG-ON DONE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _PWRON_SEQ      , regStat, "PWRON SEQUENCE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _ENG_RST        , regStat, "ENG RESET");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_OFF      , regStat, "CLAMP OFF");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _CLAMP_OFF_DELAY, regStat, "CLAMP OFF DELAY");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _PGST, _DAT_RESTORE    , regStat, "DATA RESTORE");
    }
    else
    {
        // Report the unrecognized state as an error.
        PRINT_NEWLINE;
        PRINT_INDENTED_ERROR("Unknown LW_PPWR_PMU_PG_STAT_PGST value");
        status = LW_ERR_GENERIC;
    }

    // Dumping the Idle Flip status
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _IDLE_FLIPPED, _ASSERTED, regStat, "ASSERTED");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _IDLE_FLIPPED, _DEASSERTED, regStat, "NOT_ASSERTED");
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal!= NULL)
        *regVal= regStat;

    return status;
}

/*!
 * @brief: The HAL function to print the Second Level Interrupt
*          pending in PMU due to PG-ENG(GR/MS/DI) or HOLDOFF wakeup.
 *
 * The Second level Interrupt status to PMU are stored in 
 * LW_PPWR_PMU_INTR_1 register
 *
 * @param[out]    regVal    Pointer to grab the Interrupt status (if needed)
 *
 * @returns LW_OK
 *
 */
LW_STATUS elpgDisplayPmuIntr1_GP10X(LwU32 *regVal)
{
    LwU32 regIntr1;
    LW_STATUS status = LW_OK;
    LwU32 val;

    regIntr1 = ELPG_REG_RD32(LW_PPWR_PMU_INTR_1);
    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_HOLDOFF);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_INTR_1 (HOLDOFF)", val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _HOLDOFF, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_0);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_INTR_1 (PG_ENG(GR))", val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_0, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_4);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_INTR_1 (PG_ENG(MS))", val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_4, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_3);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_INTR_1 (PG_ENG(DI))", val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_3, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntr1;

    return status;
}

/* ------------------------- HAL Functions ------------------------------- */

/*!
 * @brief: The HAL function to start the PG-ENG(GR/MS/DI).
 *
 * This function starts the desired Pg Engine. The start mechanism
 * is as follows:
 *    1. Set the PG-ENG to IDLE state by updating
 *       LW_PPWR_PMU_IDLE_CTRL_SUPP_VALUE bit to _IDLE
 *
 * @param[in]    elpgId    Engine ID
 *
 * @returns LW_OK                  for successful completion
 *          LW_ERR_NOT_SUPPORTED   for incorrect Engine ID
 *          LW_ERR_ILWALID_STATE   if Engine is in invalid State
 *
 */
LW_STATUS elpgStart_GP10X(LwU32 elpgId)
{
    LW_STATUS status            = LW_OK;
    LwU32     regIdleSupp       = 0;
    LwU32     idleSuppCtrlIndex = 0;

    status = _elpgGetIdleSuppCtrlIdx_GP10X(elpgId, &idleSuppCtrlIndex);

    // Starting the PG-Engine by setting the LW_PPWR_PMU_IDLE_CTRL_SUPP_VALUE
    // for PG_ENG(ID) to IDLE from NEVER, as set during elpgStop.
    regIdleSupp = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_CTRL_SUPP(idleSuppCtrlIndex));
    regIdleSupp = FLD_SET_DRF(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _IDLE, regIdleSupp);
    ELPG_REG_WR32(LW_PPWR_PMU_IDLE_CTRL_SUPP(idleSuppCtrlIndex), regIdleSupp);

    PRINT_NEWLINE;
    PRINT_STRING("**************************************************************************");
    dprintf     ("lw: PG-Eng(%d) has been started. Now verifying that the changes went through.\n", elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    // Checking if the PG engine was successfully started.
    regIdleSupp = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_CTRL_SUPP(idleSuppCtrlIndex));
    if(FLD_TEST_DRF(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _IDLE, regIdleSupp))
    {
        PRINT_NEWLINE;
        PRINT_STRING("**************************************************************************");
        dprintf     ("lw: PG-Eng(%d) successful start has been verified.\n", elpgId);
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;
    }
    else
    {
        PRINT_NEWLINE;
        PRINT_STRING("**************************************************************************");
        dprintf     ("lw: Failed to start PG-Eng(%d) engine!!!\n", elpgId);
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;
        status = LW_ERR_ILWALID_STATE;
    }

    dprintf("lw: Current State of LW_PPWR_PMU_IDLE_CTRL_SUPP(%d) is:\n", idleSuppCtrlIndex);
    _elpgDisplayIdleCtrlSupp_GP10X(elpgId, NULL);

    return status;
}

/*!
 * @brief: The HAL function to stop the PG-ENG(GR/MS/DI).
 *
 * This function stops the desired Pg Engine. The Stop Mechanism
 * is as follows:
 *    1. Set the LW_PPWR_PMU_IDLE_STATUS_1_SW0 bit to BUSY
 *       This will wake any engine that is power-gated.
 *
 *    2. Wait for PG-Engine to exit from engaged state.
 *       This is done by checking Engine State bit
 *       of LW_PPWR_PMU_PG_STAT_EST for respective engine to
 *       get to IDLE or ACTIVE state, and the HW controller
 *       for that engine is in IDLE state
 *
 *    3. Set the PG-ENG to DISABLE state by updating
 *       LW_PPWR_PMU_IDLE_CTRL_SUPP_VALUE bit to NEVER
 *
 *    4. Reset the idle signal of step-1 to IDLE.
 *
 * @param[in]    elpgId    Engine ID
 *
 * @returns LW_OK                  for successful completion
 *          LW_ERR_NOT_SUPPORTED   for incorrect Engine ID
 *
 */
LW_STATUS elpgStop_GP10X(LwU32 elpgId)
{
    LW_STATUS status        = LW_OK;
    LwU32 regIdleStatus     = 0;
    LwU32 regIdleSupp       = 0;
    LwU32 regPgStat         = 0;
    LwBool bLoopCtrl        = LW_TRUE;
    LwU32 idleSuppCtrlIndex = 0;
    LwU32 i = 0;

    // First check if the engine is supported or not.
    if (pElpg[indexGpu].elpgIsEngineSupported(elpgId) != LW_TRUE)
    {
        PRINT_INDENTED_ERROR("PG_ENG on Specified Engine is not supported\n");
        return LW_ERR_NOT_SUPPORTED;
    }

    status = _elpgGetIdleSuppCtrlIdx_GP10X(elpgId, &idleSuppCtrlIndex);
    if (status != LW_OK)
    {
        dprintf("\tSuppCtrlIdx is not initialized for PG_ENG(%d)\n", elpgId);
        return status;
    }

    // Step-1: Set the Activity of the PG Ctrls to Busy.
    regIdleStatus = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_1);
    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_GRAPHICS))
    {
        dprintf("lw: Using _SW0 bit for waking up the PG_ENG(%d).\n", elpgId);
        regIdleStatus = FLD_SET_DRF(_PPWR, _PMU_IDLE_STATUS_1, _SW0, _BUSY, regIdleStatus);
    }
    else
    {
        dprintf("lw: Using _SW1 bit for waking up the PG_ENG(%d).\n", elpgId);
        regIdleStatus = FLD_SET_DRF(_PPWR, _PMU_IDLE_STATUS_1, _SW1, _BUSY, regIdleStatus);
    }
    ELPG_REG_WR32(LW_PPWR_PMU_IDLE_STATUS_1, regIdleStatus);

    // Step-2: Waiting for PG_ENG(elpgId) to exit from engaged state.
    do
    {
        regPgStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_STAT(elpgId));
        bLoopCtrl = (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _IDLE, regPgStat) ||
                     FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE, regPgStat)) &&
                    FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _IDLE, regPgStat);
    } while(!bLoopCtrl);

    // Step-3: Stopping the engine.
    regIdleSupp = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_CTRL_SUPP(idleSuppCtrlIndex));
    regIdleSupp = FLD_SET_DRF(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _NEVER, regIdleSupp);
    ELPG_REG_WR32(LW_PPWR_PMU_IDLE_CTRL_SUPP(idleSuppCtrlIndex), regIdleSupp);

    // Step-4: Set the Activity of the PG Ctrls to IDLE.
    regIdleStatus = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_1);
    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_GRAPHICS))
    {
        regIdleStatus = FLD_SET_DRF(_PPWR, _PMU_IDLE_STATUS_1, _SW0, _IDLE, regIdleStatus);
    }
    else
    {
        regIdleStatus = FLD_SET_DRF(_PPWR, _PMU_IDLE_STATUS_1, _SW1, _IDLE, regIdleStatus);
    }
    ELPG_REG_WR32(LW_PPWR_PMU_IDLE_STATUS_1, regIdleStatus);

    dprintf("lw: Current State of LW_PPWR_PMU_IDLE_CTRL_SUPP(%d) is:\n", idleSuppCtrlIndex);
    _elpgDisplayIdleCtrlSupp_GP10X(elpgId, NULL);

    dprintf("lw: Stopping of the PG_ENG(%d) is complete.\n", elpgId);
    return status;
}

/*!
 * @brief: The HAL function checks if the Engine PG-ENG(GR/MS/DI)
*          is supported on current chip or not.
 *
 * The task is accomplished by checking the corresponding bit from
 * LW_PPWR_PMU_PG_CTRL register for Enablement status.
 *
 * @param[in]    elpgId    Engine ID
 *
 * @returns LW_TRUE    if feature is supported
 *          LW_FALSE   otherwise
 */
LwBool elpgIsEngineSupported_GP10X(LwU32 elpgId)
{
    LwU32  regStat;
    LwBool bSupported = LW_FALSE;

    regStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_CTRL);

    switch (elpgId)
    {
        case ELPG_ENGINE_ID_GRAPHICS:
        {
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_CTRL, _ENG_0, _ENABLE, regStat))
            {
                bSupported =  LW_TRUE;
            }
            break;
        }
        case ELPG_ENGINE_ID_MS:
        {
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_CTRL, _ENG_4, _ENABLE, regStat))
            {
                bSupported =  LW_TRUE;
            }
            break;
        }
        case ELPG_ENGINE_ID_DI:
        {
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_CTRL, _ENG_3, _ENABLE, regStat))
            {
                bSupported =  LW_TRUE;
            }
            break;
        }
        default:
        {
            //dprintf("(ERROR - invalid id!)\n");
            PRINT_NEWLINE;
            return LW_FALSE;
        }
    }

    return bSupported;
}

/*!
 * @brief: The HAL function to display the Blocker and Wakeup interrupt status for MS
 *
 * @param[in]     elpgID    PG-ENG engine ID
 *
 * @returns LW_ERR_GENERIC    in case of Config Error
 *          LW_OK             otherwise
 */

LW_STATUS elpgDisplayMsBlockerAndInterruptState_GP10X(void)
{
    LwU32     regVal;

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the state of PRIV Blocker for MS");
    PRINT_STRING("**************************************************************************");

    // Read the XVE Blocker Register
    regVal = CFG_RD32(LW_XVE_BAR_BLOCKER);
    PRINT_REG_AND_VALUE("LW_XVE_BAR_BLOCKER", regVal);
    PRINT_NEWLINE;

    if (!IsGP100()) {
        regVal = ELPG_REG_RD32(LW_PSEC_BLOCKER_BAR0_CTRL);
        PRINT_REG_AND_VALUE("LW_PSEC_BLOCKER_BAR0_CTRL", regVal);
        PRINT_NEWLINE;
    }

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the state of ISO Blocker for MS");
    PRINT_STRING("**************************************************************************");

    // Read the ISO HUB  Blocker Register
    regVal = ELPG_REG_RD32(LW_PDISP_ISOHUB_FB_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PDISP_ISOHUB_FB_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the state of NISO Blocker for MS");
    PRINT_STRING("**************************************************************************");

    // Read the PERF NISO Blocker Register
    regVal = ELPG_REG_RD32(LW_PERF_PMASYS_SYS_FB_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PERF_PMASYS_SYS_FB_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    // Read the DISB NISO Blocker Register
    regVal = ELPG_REG_RD32(LW_PDISP_DMI_FB_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PDISP_DMI_FB_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    // Read the SEC2 FB Blocker Register
    regVal = ELPG_REG_RD32(LW_PSEC_FALCON_ENGCTL);
    PRINT_REG_AND_VALUE("LW_PSEC_FALCON_ENGCTL", regVal);
    PRINT_NEWLINE;

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the MS Wakeup Interrupt Status");
    PRINT_STRING("**************************************************************************");

    regVal = ELPG_REG_RD32(LW_PPWR_PMU_GPIO_1_INTR_RISING_EN);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_GPIO_1_INTR_RISING_EN", regVal);
    PRINT_NEWLINE;

    regVal = ELPG_REG_RD32(LW_PPWR_PMU_GPIO_1_INTR_RISING);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_GPIO_1_INTR_RISING", regVal);
    PRINT_NEWLINE;

    return LW_OK;
}

/*!
 * @brief: The HAL function to print the IDLE Signal Status.
 *
 * @returns LW_OK
 */
LW_STATUS elpgDisplayPmuPgIdleSignal_GP10X(void)
{
    LwU32 regVal;

    regVal = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_IDLE_STATUS", regVal);
    PRINT_NEWLINE;

    regVal = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_1);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_IDLE_STATUS_1", regVal);
    PRINT_NEWLINE;
    return LW_OK;
}
