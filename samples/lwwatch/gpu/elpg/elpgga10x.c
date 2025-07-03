#include "elpg.h"
#include "ampere/ga102/dev_pwr_pri.h"
#include "ampere/ga102/dev_perf.h"
#include "ampere/ga102/dev_lw_xve.h"
#include "disp/v04_00/dev_disp.h"
#include "ampere/ga102/dev_sec_pri.h"
#include "ampere/ga102/dev_gsp.h"
#include "ampere/ga102/dev_hshub_SW.h"

#include "g_elpg_private.h"

// XVE Register Read in GPU Space
#define CFG_RD32(a)   ELPG_REG_RD32(DRF_BASE(LW_PCFG) + a)

/* ------------------------- HAL Functions ------------------------------- */

/*!
 * @brief: The HAL function to print the state of LPWR sequencer for GR and MS.
 * 
 * @returns LW_OK
 */
LW_STATUS elpgDisplaySequencerState_GA10X(LwU32 elpgId)
{
    LW_STATUS Status = LW_OK;
    LwU32     RegSeqStatusVal;

    if(elpgId != ELPG_ENGINE_ID_GRAPHICS &&
       elpgId != ELPG_ENGINE_ID_MS)
    {
        // Sequencer is supported for GR and MS.
        return Status;
    }

    PRINT_STRING("**************************************************************************");
    PRINT_STRING_AND_FIELD_VALUE(Dumping the state of LPWR sequencer for LPWR_ENG(%d), elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    RegSeqStatusVal = ELPG_REG_RD32(LW_PPWR_PMU_RAM_STATUS);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_RAM_STATUS", RegSeqStatusVal);
    PRINT_NEWLINE;

    switch(elpgId)
    {
        case ELPG_ENGINE_ID_GRAPHICS:

            // Determine the state of GR FSM
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_FSM, _POWER_ON, RegSeqStatusVal, "POWER_ON");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_FSM, _LOW_POWER_ENTERING, RegSeqStatusVal, "LOW_POWER_ENTERING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_FSM, _LOW_POWER, RegSeqStatusVal, "LOW_POWER");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_FSM, _LOW_POWER_EXITING , RegSeqStatusVal, "LOW_POWER_EXITING");

            // Determine the state of GR profile
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_PROFILE, _POWER_ON, RegSeqStatusVal, "POWER_ON");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_PROFILE, _PROFILE0, RegSeqStatusVal, "PROFILE0");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_PROFILE, _PROFILE1, RegSeqStatusVal, "PROFILE1");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _GR_PROFILE, _PROFILE2, RegSeqStatusVal, "PROFILE2");
            break;

        case ELPG_ENGINE_ID_MS:

            // Determine the state of MS FSM
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_FSM, _POWER_ON, RegSeqStatusVal, "POWER_ON");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_FSM, _LOW_POWER_ENTERING, RegSeqStatusVal, "LOW_POWER_ENTERING");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_FSM, _LOW_POWER, RegSeqStatusVal, "LOW_POWER");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_FSM, _LOW_POWER_EXITING , RegSeqStatusVal, "LOW_POWER_EXITING");

            // Determine the state of MS profile
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_PROFILE, _POWER_ON, RegSeqStatusVal, "POWER_ON");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_PROFILE, _PROFILE0, RegSeqStatusVal, "PROFILE0");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_PROFILE, _PROFILE1, RegSeqStatusVal, "PROFILE1");
            PRINT_DRF_IF_SET_TO(_PPWR, _PMU_RAM_STATUS, _MS_PROFILE, _PROFILE2, RegSeqStatusVal, "PROFILE2");
            break;

        default:
            break;
    }
    PRINT_NEWLINE;

    return Status;
}

/*!
 * @brief: The HAL function to display the Blocker and Wakeup interrupt status for MS
 *
 * @param[in]     elpgID    PG-ENG engine ID
 *
 * @returns LW_ERR_GENERIC    in case of Config Error
 *          LW_OK             otherwise
 */

LW_STATUS elpgDisplayMsBlockerAndInterruptState_GA10X(void)
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
    regVal = ELPG_REG_RD32(LW_PFB_HSHUB_IG_SYS_FB_BLOCKER_CTRL(0));
    PRINT_REG_AND_VALUE("LW_PFB_HSHUB_IG_SYS_FB_BLOCKER_CTRL(0)", regVal);
    PRINT_NEWLINE;

    // Read the HSHUB Blocker Register
    regVal = ELPG_REG_RD32(LW_PFB_HSHUB_IG_SYS_FB_BLOCKER_CTRL(1));
    PRINT_REG_AND_VALUE("LW_PFB_HSHUB_IG_SYS_FB_BLOCKER_CTRL(1)", regVal);
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
