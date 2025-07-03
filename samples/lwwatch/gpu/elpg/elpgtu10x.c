#include "elpg.h"
#include "turing/tu102/dev_perf.h"
#include "turing/tu102/dev_lw_xve.h"
#include "disp/v04_00/dev_disp.h"
#include "turing/tu102/dev_pwr_pri.h"
#include "turing/tu102/dev_sec_pri.h"
#include "turing/tu102/dev_gsp.h"
#include "turing/tu102/dev_hshub.h"

#include "g_elpg_private.h"

// XVE Register Read in GPU Space
#define CFG_RD32(a)   ELPG_REG_RD32(DRF_BASE(LW_PCFG) + a)

//*****************************************************************************
// Static functions
//*****************************************************************************

/*!
 * @brief: HAL function to display current SW CLIENT status for supported
 *         PG_ENG/LPWR_ENG
 *
 * @param[in]:  elpgId    Engine ID
 */
LW_STATUS elpgDisplaySwClientStatus_TU10X(LwU32 elpgId)
{
    LW_STATUS Status = LW_OK;
    LwU32     regSwClientStatus;

    regSwClientStatus = ELPG_REG_RD32(LW_PPWR_PMU_PG_SW_CLIENT_STATUS);
    if (FLD_IDX_TEST_DRF(_PPWR, _PMU_PG_SW_CLIENT_STATUS, _ST_ENG, elpgId, _BUSY, regSwClientStatus))
    {
        PRINT_STRING_AND_FIELD_VALUE(SW_CLIENT(%d) is %s, elpgId, "BUSY");
    }
    else
    {
        PRINT_STRING_AND_FIELD_VALUE(SW_CLIENT(%d) is %s, elpgId, "IDLE");
    }
    PRINT_NEWLINE;

    return Status;
}

//*****************************************************************************
// Global functions
//*****************************************************************************

/*!
 * @brief: The HAL function to display the Configuration and status of PG CTRL
 *
 * @param[in]     elpgID    PG-ENG engine ID
 *
 * @returns LW_ERR_GENERIC    in case of Config Error
 *          LW_OK             otherwise
 */

LW_STATUS elpgDisplayPgEngConfig_TU10X(LwU32 elpgId)
{
    LwU32 regCtrl;
    LwU32 regIdleCount;
    LwU32 regIdleMask;
    LwU32 regIdleMask1;
    LwU32 regIdleMask2;
    LwU32 regTriggerMask;
    LwU32 regTriggerMask1;
    LwU32 regTriggerMask2;

    // Read the PG Ctrl Register
    regCtrl = ELPG_REG_RD32(LW_PPWR_PMU_PG_CTRL(elpgId));

    // Read the IDLE Mask configuration
    regIdleMask = ELPG_REG_RD32(LW_PPWR_PMU_PG_IDLE_MASK(elpgId));
    regIdleMask1 = ELPG_REG_RD32(LW_PPWR_PMU_PG_IDLE_MASK_1(elpgId));
    regIdleMask2 = ELPG_REG_RD32(LW_PPWR_PMU_PG_IDLE_MASK_2(elpgId));

    // Read the IDLE Counter register
    regIdleCount = ELPG_REG_RD32(LW_PPWR_PMU_PG_IDLE_CNT(elpgId));

    // Read the Precondition registers
    regTriggerMask  = ELPG_REG_RD32(LW_PPWR_PMU_PG_ON_TRIGGER_MASK(elpgId));
    regTriggerMask1 = ELPG_REG_RD32(LW_PPWR_PMU_PG_ON_TRIGGER_MASK_1(elpgId));
    regTriggerMask2 = ELPG_REG_RD32(LW_PPWR_PMU_PG_ON_TRIGGER_MASK_2(elpgId));

    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_CTRL(%d), elpgId, regCtrl);
    //
    // Dump the PG CTRL Type i.e PG_ENG or LPWR ENG
    //
    // NOTE that only "one" of the following prints will succeed since only
    // one state can be active at a time.
    //
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_CTRL, _ENG_TYPE, _PG, regCtrl, "PG_ENG");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_CTRL, _ENG_TYPE, _LPWR, regCtrl, "LPWR_ENG");
    //
    // Dump the idle counter state for requested PG Control
    //
    // NOTE that only "one" of the following prints will succeed since only
    // one state can be active at a time.
    //
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_CTRL, _IDLE_MASK_VALUE, _NEVER , regCtrl, "NEVER");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_CTRL, _IDLE_MASK_VALUE, _IDLE  , regCtrl, "IDLE");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_CTRL, _IDLE_MASK_VALUE, _BUSY  , regCtrl, "BUSY");
    PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_CTRL, _IDLE_MASK_VALUE, _ALWAYS, regCtrl, "ALWAYS");
    PRINT_NEWLINE;

    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_IDLE_MASK(%d), elpgId, regIdleMask);
    PRINT_NEWLINE;
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_IDLE_MASK_1(%d), elpgId, regIdleMask1);
    PRINT_NEWLINE;
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_IDLE_MASK_2(%d), elpgId, regIdleMask2);
    PRINT_NEWLINE;

    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_IDLE_CNT(%d), elpgId, regIdleCount);
    PRINT_NEWLINE;

    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_ON_TRIGGER_MASK(%d), elpgId, regTriggerMask);
    PRINT_NEWLINE;
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_ON_TRIGGER_MASK_1(%d), elpgId, regTriggerMask1);
    PRINT_NEWLINE;
    PRINT_REG_IDX_AND_VALUE(LW_PPWR_PMU_PG_ON_TRIGGER_MASK_2(%d), elpgId, regTriggerMask2);
    PRINT_NEWLINE;

    return LW_OK;
}

/*!
 * @brief: The HAL function to display the Blocker and Wakeup interrupt status for MS
 *
 * @param[in]     elpgID    PG-ENG engine ID
 *
 * @returns LW_ERR_GENERIC    in case of Config Error
 *          LW_OK             otherwise
 */

LW_STATUS elpgDisplayMsBlockerAndInterruptState_TU10X(void)
{
    LwU32     regVal;

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the state of PRIV Blocker for MS");
    PRINT_STRING("**************************************************************************");

    // Read the XVE Blocker Register
    regVal = CFG_RD32(LW_XVE_BAR_BLOCKER);
    PRINT_REG_AND_VALUE("LW_XVE_BAR_BLOCKER", regVal);
    PRINT_NEWLINE;

    // Read the SEC2 Priv  Blocker Register
    regVal = ELPG_REG_RD32(LW_PSEC_PRIV_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PSEC_PRIV_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    // Read the SEC2 Priv Blocker Status Register
    regVal = ELPG_REG_RD32(LW_PSEC_PRIV_BLOCKER_STAT);
    PRINT_REG_AND_VALUE("LW_PSEC_PRIV_BLOCKER_STAT", regVal);
    PRINT_NEWLINE;

    // Read the GSP PRIV Blocker Register
    regVal = ELPG_REG_RD32(LW_PGSP_PRIV_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PGSP_PRIV_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    // Read the GSP Priv Blocker Status Register
    regVal = ELPG_REG_RD32(LW_PGSP_PRIV_BLOCKER_STAT);
    PRINT_REG_AND_VALUE("LW_PGSP_PRIV_BLOCKER_STAT", regVal);
    PRINT_NEWLINE;

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the state of ISO Blocker for MS");
    PRINT_STRING("**************************************************************************");

    // Read the ISO HUB  Blocker Register
    regVal = ELPG_REG_RD32(LW_PDISP_IHUB_COMMON_FB_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PDISP_IHUB_COMMON_FB_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the state of NISO Blocker for MS");
    PRINT_STRING("**************************************************************************");

    // Read the PERF NISO Blocker Register
    regVal = ELPG_REG_RD32(LW_PERF_PMASYS_SYS_FB_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PERF_PMASYS_SYS_FB_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    // Read the DISB NISO Blocker Register
    regVal = ELPG_REG_RD32(LW_PDISP_FE_FB_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PDISP_FE_FB_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    // Read the HSHUB Blocker Register
    regVal = ELPG_REG_RD32(LW_PFB_HSHUB_IG_SYS_FB_BLOCKER_CTRL);
    PRINT_REG_AND_VALUE("LW_PFB_HSHUB_IG_SYS_FB_BLOCKER_CTRL", regVal);
    PRINT_NEWLINE;

    // Read the SEC2 FB Blocker Register
    regVal = ELPG_REG_RD32(LW_PSEC_FALCON_ENGCTL);
    PRINT_REG_AND_VALUE("LW_PSEC_FALCON_ENGCTL", regVal);
    PRINT_NEWLINE;

    // Read the GSP FB Blocker Register
    regVal = ELPG_REG_RD32(LW_PGSP_FALCON_ENGCTL);
    PRINT_REG_AND_VALUE("LW_PGSP_FALCON_ENGCTL", regVal);
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
 * @brief: The HAL function to display the interrupt
 *         pending status to PG_ENG
 *
 * The following function checks if there is an interrupt pending for
 * any supported PG_ENG/LPWR_ENG.
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
LW_STATUS elpgDisplayPgIntrStat_TU10X(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regPgoff;
    LwU32 regIntrStat;
    LwU32 regIdleSnap;
    LwU32 val;
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
            // CTX_RESTORE interrupt can be pending only for _PG_ENG type state machine
            regPgoff = ELPG_REG_RD32(LW_PPWR_PMU_PG_CTRL(elpgId));
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_CTRL, _ENG_TYPE, _PG, regPgoff))
            {
                if (FLD_TEST_DRF(_PPWR, _PMU_PG_CTRL_PGOFF, _REQSRC, _RM,
                                 regPgoff))
                {
                    PRINT_NEWLINE;
                    PRINT_INDENTED_STRING("_CTX_RESTORE request came from RM");
                }
                else if (FLD_TEST_DRF(_PPWR, _PMU_PG_CTRL_PGOFF, _REQSRC, _HOST,
                                      regPgoff))
                {
                    PRINT_NEWLINE;
                    PRINT_INDENTED_STRING("_CTX_RESTORE request came from HOST");
                }
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
            // Set IDX to current engine and enable SNAP_READ
            val = ELPG_REG_RD32(LW_PPWR_PMU_PG_MISC);
            val = FLD_SET_DRF_NUM(_PPWR, _PMU_PG_MISC,
                                  _IDLE_SNAP_IDX, elpgId, val);
            val = FLD_SET_DRF(_PPWR, _PMU_PG_MISC, _IDLE_SNAP_READ, _ENABLE, val);
            ELPG_REG_WR32(LW_PPWR_PMU_PG_MISC, val);

            PRINT_NEWLINE;
            PRINT_INDENTED_STRING("_IDLE_SNAP set, Dumping IDLE_STATUS and IDLE_STATUS_1/_2 registers");
            regIdleSnap = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS);
            dprintf("\t\tIDLE_STATUS   : 0x%x", regIdleSnap);
            regIdleSnap = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_1);
            dprintf("\t\tIDLE_STATUS_1 : 0x%x", regIdleSnap);
            regIdleSnap = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_2);
            dprintf("\t\tIDLE_STATUS_2 : 0x%x", regIdleSnap);

            // Disable SNAP_READ
            val = FLD_SET_DRF(_PPWR, _PMU_PG_MISC, _IDLE_SNAP_READ, _DISABLE, val);
            ELPG_REG_WR32(LW_PPWR_PMU_PG_MISC, val);
        }
    }
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntrStat;

    return status;
}

/*!
 * @brief: The HAL function to print the Second Level Interrupt
 *         pending in PMU due to PG-ENG(GR PG/GR RG/EI/MS) or HOLDOFF wakeup.
 *
 * The Second level Interrupt status to PMU are stored in 
 * LW_PPWR_PMU_INTR_1 register
 *
 * @param[out]    regVal    Pointer to grab the Interrupt status (if needed)
 *
 * @returns LW_OK
 *
 */
LW_STATUS elpgDisplayPmuIntr1_TU10X(LwU32 *regVal)
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
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[0], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_0, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_1);
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[1], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_1, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_2);
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[2], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_2, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_3);
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[3], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_3, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_4);
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[4], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_4, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntr1;

    return status;
}

/* ------------------------- HAL Functions ------------------------------- */

/*!
 * @brief: The HAL function to start the supported PG-ENG.
 *
 * This function starts the desired Pg Engine. The start mechanism
 * is as follows:
 *    1. Set the PG-ENG to IDLE state by updating
 *       PMU_PG_SW_CLIENT_7_ENG_BUSY_CLR bit to _TRIGGER
 *       (SW CLIENT_7 is used by LW_WATCH)
 *
 * @param[in]    elpgId    Engine ID
 *
 * @returns LW_OK                  for successful completion
 *          LW_ERR_NOT_SUPPORTED   for incorrect Engine ID
 *          LW_ERR_ILWALID_STATE   if Engine is in invalid State
 *
 */
LW_STATUS elpgStart_TU10X(LwU32 elpgId)
{
    LW_STATUS status            = LW_OK;
    LwU32     regIdleStatus     = 0;

    // Set the SW CLIENT Back to Idle which will allow the PG ENG to engage
    regIdleStatus = ELPG_REG_RD32(LW_PPWR_PMU_PG_SW_CLIENT_7);
    if (FLD_IDX_TEST_DRF(_PPWR, _PMU_PG_SW_CLIENT, _ENG_ST, elpgId, _BUSY, regIdleStatus))
    {
        regIdleStatus = FLD_IDX_SET_DRF(_PPWR, _PMU_PG_SW_CLIENT, _ENG_BUSY_CLR, elpgId, _TRIGGER, regIdleStatus);
        ELPG_REG_WR32(LW_PPWR_PMU_PG_SW_CLIENT_7, regIdleStatus);
    }

    PRINT_NEWLINE;
    PRINT_STRING("**************************************************************************");
    dprintf     ("lw:  PG-Eng(%d) has been started. Now verifying that the changes went through.\n", elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    // Checking if the PG engine was successfully started.
    regIdleStatus = ELPG_REG_RD32(LW_PPWR_PMU_PG_SW_CLIENT_7);
    if(FLD_IDX_TEST_DRF(_PPWR, _PMU_PG_SW_CLIENT, _ENG_ST, elpgId, _IDLE, regIdleStatus))
    {
        PRINT_NEWLINE;
        PRINT_STRING("**************************************************************************");
        dprintf     ("lw:  PG-Eng(%d) successful start has been verified.\n", elpgId);
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;
    }
    else
    {
        PRINT_NEWLINE;
        PRINT_STRING("**************************************************************************");
        dprintf     ("lw:  Failed to start PG-Eng(%d) engine!!!\n", elpgId);
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;
        status = LW_ERR_ILWALID_STATE;
    }

    // Display SW Client status
    pElpg[indexGpu].elpgDisplaySwClientStatus(elpgId);

    return status;
}

/*!
 * @brief: The HAL function to stop the PG-ENG(GR/MS/DI).
 *
 * This function stops the desired Pg Engine. The Stop Mechanism
 * is as follows:
 *    1. Set the PMU_PG_SW_CLIENT_7_ENG_BUSY_SET bit to _TRIGGER
 *       This will wake any engine that is power-gated.
 *       (SW CLIENT_7 is used by LW_WATCH)
 *
 *    2. Wait for PG-Engine to exit from engaged state.
 *       This is done by checking Engine State bit
 *       of LW_PPWR_PMU_PG_STAT_EST for respective engine to
 *       get to IDLE or ACTIVE state, and the HW controller
 *       for that engine is in IDLE state
 *
 * @param[in]    elpgId    Engine ID
 *
 * @returns LW_OK                  for successful completion
 *          LW_ERR_NOT_SUPPORTED   for incorrect Engine ID
 *
 */
LW_STATUS elpgStop_TU10X(LwU32 elpgId)
{
    LW_STATUS status            = LW_OK;
    LwU32     regIdleStatus     = 0;
    LwU32     regPgCtrl         = 0;
    LwU32     regPgStat         = 0;
    LwBool    bLoopCtrl         = LW_TRUE;
    LwU32     idleSuppCtrlIndex = 0;
    LwU32     i                 = 0;

    // First check if the engine is supported or not.
    if (pElpg[indexGpu].elpgIsEngineSupported(elpgId) != LW_TRUE)
    {
        PRINT_INDENTED_ERROR("PG_ENG on Specified Engine is not supported\n");
        return LW_ERR_NOT_SUPPORTED;
    }

    // Step-1: Set the Activity of the PG Ctrls to Busy.
    regIdleStatus = ELPG_REG_RD32(LW_PPWR_PMU_PG_SW_CLIENT_7);
    dprintf("lw: Using SW Client 7 for waking up the PG_ENG(%d).\n", elpgId);
    regIdleStatus = FLD_IDX_SET_DRF(_PPWR, _PMU_PG_SW_CLIENT, _ENG_BUSY_SET, elpgId, _TRIGGER, regIdleStatus);
    ELPG_REG_WR32(LW_PPWR_PMU_PG_SW_CLIENT_7, regIdleStatus);

    // Step-2: Waiting for PG_ENG(elpgId) to exit from engaged state.
    do
    {
        regPgStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_STAT(elpgId));
        bLoopCtrl = (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _IDLE, regPgStat) ||
                     FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE, regPgStat)) &&
                    FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _PGST, _IDLE, regPgStat);
    } while(!bLoopCtrl);

    dprintf("lw: Current State of LW_PPWR_PMU_PG_CTRL_IDLE_MASK_VALUE(%d) is:\n", elpgId);
    PRINT_NEWLINE;

    // Get SW Client status
    pElpg[indexGpu].elpgDisplaySwClientStatus(elpgId);

    dprintf("lw: Stopping of the PG_ENG(%d) is complete.\n", elpgId);
    PRINT_NEWLINE;

    return status;
}

/*!
 * @brief: The HAL function checks if the requested LPWR Engine
*          is supported on current chip or not.
 *
 * The task is accomplished by checking the corresponding bit from
 * LW_PPWR_PMU_PG_CTRL(i) register for Enablement status.
 *
 * @param[in]    elpgId    Engine ID
 *
 * @returns LW_TRUE    if feature is supported
 *          LW_FALSE   otherwise
 */
LwBool elpgIsEngineSupported_TU10X(LwU32 elpgId)
{
    LwU32  regStat;
    LwBool bSupported = LW_FALSE;

    if (elpgId >= ELPG_ENGINE_ID_ILWALID)
    {
        dprintf("lw: Error: Invalid PgEng Id %d\n", elpgId);
        return LW_FALSE;
    }

    regStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_CTRL(elpgId));

    if (FLD_TEST_DRF(_PPWR, _PMU_PG_CTRL, _ENG, _ENABLE, regStat))
    {
        bSupported =  LW_TRUE;
    }

    return bSupported;
}

/*!
 * @brief: The HAL function to print the IDLE Signal Status.
 *
 * @returns LW_OK
 */
LW_STATUS elpgDisplayPmuPgIdleSignal_TU10X(void)
{
    LwU32 regVal;

    regVal = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_IDLE_STATUS", regVal);
    PRINT_NEWLINE;

    regVal = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_1);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_IDLE_STATUS_1", regVal);
    PRINT_NEWLINE;

    regVal = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_STATUS_2);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_IDLE_STATUS_2", regVal);
    PRINT_NEWLINE;

    return LW_OK;
}

/*!
 * @brief: The HAL function to print the state of LPWR sequencer.
 * 
 * @returns LW_OK
 */
LW_STATUS elpgDisplaySequencerState_TU10X(LwU32 elpgId)
{
    LW_STATUS Status = LW_OK;
    LwU32     RegSeqStatusVal;

    if(elpgId != ELPG_ENGINE_ID_GRAPHICS &&
       elpgId != ELPG_ENGINE_ID_MS)
    {
        // Sequencer is supported only for GR and MS.
        return Status;
    }

    PRINT_STRING("**************************************************************************");
    PRINT_STRING_AND_FIELD_VALUE(Dumping the state of LPWR sequencer for LPWR_ENG(%d), elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    RegSeqStatusVal = ELPG_REG_RD32(LW_PPWR_PMU_RAM_STATUS);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_RAM_STATUS", RegSeqStatusVal);

    switch(elpgId)
    {
        case ELPG_ENGINE_ID_GRAPHICS:
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR, _POWER_ON, RegSeqStatusVal, "POWER_ON");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR, _RPPG_ENTERING, RegSeqStatusVal, "RPPG_ENTERING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR, _RPPG, RegSeqStatusVal, "RPPG");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR, _RPPG_EXITING, RegSeqStatusVal, "RPPG_EXITING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR, _SD_ENTERING, RegSeqStatusVal, "SD_ENTERING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR, _SD, RegSeqStatusVal, "SD");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR, _SD_EXITING, RegSeqStatusVal, "SD_EXITING");
            break;

        case ELPG_ENGINE_ID_MS:
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS, _POWER_ON, RegSeqStatusVal, "POWER_ON");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS, _RPPG_ENTERING, RegSeqStatusVal, "RPPG_ENTERING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS, _RPPG, RegSeqStatusVal, "RPPG");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS, _RPPG_EXITING, RegSeqStatusVal, "RPPG_EXITING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS, _SD_ENTERING, RegSeqStatusVal, "SD_ENTERING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS, _SD, RegSeqStatusVal, "SD");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS, _SD_EXITING, RegSeqStatusVal, "SD_EXITING");
            break;

        default:
            break;
    }
    PRINT_NEWLINE;

    return Status;
}
