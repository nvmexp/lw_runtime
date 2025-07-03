/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "elpg.h"
#include "ada/ad102/dev_lw_xve.h"
#include "ada/ad102/dev_pwr_pri.h"

// XVE Register Read in GPU Space
#define CFG_RD32(a)   ELPG_REG_RD32(DRF_BASE(LW_PCFG) + a)


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

LW_STATUS elpgDisplayPgEngConfig_AD102(LwU32 elpgId)
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

LW_STATUS elpgDisplayPgIntrStat_AD102(LwU32 elpgId, LwU32 *regVal)
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
LW_STATUS elpgStart_AD102(LwU32 elpgId)
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
LW_STATUS elpgStop_AD102(LwU32 elpgId)
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
 * @brief: The HAL function to print the Second Level Interrupt
 *         pending in PMU due to PG-ENG(GR PG/GR RG/EI/MS/DIFR) or HOLDOFF wakeup.
 *
 * The Second level Interrupt status to PMU are stored in
 * LW_PPWR_PMU_INTR_1 register
 *
 * @param[out]    regVal    Pointer to grab the Interrupt status (if needed)
 *
 * @returns LW_OK
 *
 */
LW_STATUS elpgDisplayPmuIntr1_AD102(LwU32 *regVal)
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

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_5);
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[5], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_5, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_6);
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[6], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_6, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_ELPG_7);
    PRINT_REG_ENG_AND_VALUE(LW_PPWR_PMU_INTR_1 LPWR_ENG(%s), LpwrEng[7], val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _ELPG_7, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntr1;

    return status;
}
