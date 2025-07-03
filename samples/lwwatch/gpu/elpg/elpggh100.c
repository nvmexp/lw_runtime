#include "elpg.h"
#include "hopper/gh100/dev_perf.h"
#include "disp/v04_00/dev_disp.h"
#include "hopper/gh100/dev_pwr_pri.h"
#include "hopper/gh100/dev_sec_pri.h"
#include "hopper/gh100/dev_gsp.h"
#include "hopper/gh100/dev_hshub_SW.h"

#include "g_elpg_private.h"

/*!
 * @brief: The HAL function to display the Blocker and Wakeup interrupt status for MS
 *
 * @param[in]     elpgID    PG-ENG engine ID
 *
 * @returns LW_ERR_GENERIC    in case of Config Error
 *          LW_OK             otherwise
 */

LW_STATUS elpgDisplayMsBlockerAndInterruptState_GH100(void)
{
    LwU32     regVal;

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Dumping the state of PRIV Blocker for MS");
    PRINT_STRING("**************************************************************************");

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
