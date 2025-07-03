/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "hal.h"
#include "elpg.h"
#include "kepler/gk104/dev_fifo.h"
#include "kepler/gk104/dev_pri_ringstation_sys.h"
#include "kepler/gk104/dev_pwr_pri.h"
#include "kepler/gk104/dev_therm.h"
#include "kepler/gk104/dev_top.h"
#include "kepler/gk104/dev_ihub.h"

#include "g_elpg_private.h"         // (rmconfig) implementation prototypes


//*****************************************************************************
// Static functions
//*****************************************************************************

LW_STATUS _elpgDisplayIdleCtrlSupp_GK104(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 val;
    LwU32 regIdleCtrlSupp;
    LW_STATUS status = LW_OK;

    //
    // LW_PPWR_PMU_IDLE_CTRL_SUPP denotes whether ELPG0 and ELPG4 have their
    // idle signal overridden for their associated engines.
    //
    regIdleCtrlSupp = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_CTRL_SUPP(elpgId));

    // Determine whether the idle signal is overridden.
    val = regIdleCtrlSupp & DRF_SHIFTMASK(LW_PPWR_PMU_IDLE_CTRL_SUPP_VALUE);
    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_IDLE_CTRL_SUPP(0) (ELPG0)", val);
    else if (elpgId == ELPG_ENGINE_ID_MS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_IDLE_CTRL_SUPP(4) (ELPG4)", val);

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

/*!
 * Helper function to stop PG on given engine
 * 
 * This function contains sequence for disabling PG on particular engine.
 * 
 * @param[in]   elpgId     Engine ID
 * 
 * @returns     LW_ERR_GENERIC   If fails to disable PG
 *              LW_OK      Otherwise
 */
LW_STATUS _elpgStop_GK104(LwU32 elpgId)
{
    LW_STATUS status = LW_OK;

    //
    // Disable ELPG and wake up its associated engine.
    //

    // The engine should be powered up before disabling idle monitoring
    status = pElpg[indexGpu].elpgPowerUp(elpgId);
    if (status != LW_OK)
        return status;

    // Write LW_PPWR_PMU_IDLE_CTRL_SUPP in order to disable idle-monitoring.
    ELPG_REG_WR32(LW_PPWR_PMU_IDLE_CTRL_SUPP(elpgId),
                         LW_PPWR_PMU_IDLE_CTRL_SUPP_VALUE_NEVER);

    // Output register values that have to do with idle monitoring.
    status = _elpgDisplayIdleCtrlSupp_GK104(elpgId, NULL);

    if (status != LW_OK)
    {
        return status;
    }

    PRINT_STRING("**************************************************************************");
    dprintf     ("lw:  ELPG%d PG_ON interrupt has been disabled to prevent power-gating.\n", elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    return status;
}


//*****************************************************************************
// Global functions
//*****************************************************************************

LW_STATUS elpgDisplayPrivAccessConfig_GK104(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 bitmask;
    LwU32 regConfig;
    LW_STATUS status = LW_OK;
    LwU32 val;

    //
    // The following is a 64-bit register. The argument "0" indicates that only
    // the lower 32-bits are read.
    //
    // When the graphics engine (GR) has PRI access blocked, the following
    // register has this field unset:
    //     LW_PPRIV_SYS_PRIV_FS_CONFIG(0)
    //         LW_PPRIV_SYS_PRI_MASTER_fecs2gr_pri
    //
    // When the video engines (MSVLD, MSPDEC, MSPPP) have PRI access blocked,
    // the following     register has these fields unset:
    //     LW_PPRIV_SYS_PRIV_FS_CONFIG(0)
    //         LW_PPRIV_SYS_PRI_MASTER_fecs2mspdec_pri
    //         LW_PPRIV_SYS_PRI_MASTER_fecs2msppp_pri
    //         LW_PPRIV_SYS_PRI_MASTER_fecs2msvld_pri
    //
    regConfig = ELPG_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0));

    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
    {
        bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2gr_pri);
        val = regConfig & bitmask;
        PRINT_REG_AND_VALUE("LW_PPRIV_SYS_PRIV_FS_CONFIG(0) (GR)", val);

        if (regConfig & bitmask)
            PRINT_FIELD_AND_STRING("fecs2gr_pri", "NOT GATED");
        else
            PRINT_FIELD_AND_STRING("fecs2gr_pri", "GATED");
    }
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regConfig;

    return status;
}

//*****************************************************************************

LW_STATUS elpgDisplayPgIntrEn_GK104(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regIntrEn;
    LW_STATUS status = LW_OK;
    LwU32 val;

    //
    // LW_PPWR_PMU_ELPG_INTREN holds the interrupt enable bits for both ELPG0
    // and ELPG1.
    //
    regIntrEn = ELPG_REG_RD32(LW_PPWR_PMU_PG_INTREN(elpgId));

    // Determine which ELPG0 interrupts are enabled.
    val = regIntrEn & (DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_PG_ON)       |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_PG_ON_DONE)  |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_CTX_RESTORE) |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_CFG_ERR)     |
                       DRF_SHIFTMASK(LW_PPWR_PMU_PG_INTREN_ENG_RST));
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_ELPG_INTREN", val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _PG_ON      , regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _PG_ON_DONE , regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _CTX_RESTORE, regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _CFG_ERR    , regIntrEn, "ENABLED", "DISABLED");
    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_PG_INTREN, _ENG_RST    , regIntrEn, "ENABLED", "DISABLED");

    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntrEn;

    return status;
}

//*****************************************************************************

LW_STATUS elpgDisplayPgIntrStat_GK104(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regPgoff;
    LwU32 regIntrStat;
    LW_STATUS status = LW_OK;

    // Determine which of the enabled interrupts are pending, if any.
    regIntrStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_INTRSTAT(elpgId));
    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_PG_INTRSTAT0", regIntrStat);
    else if (elpgId == ELPG_ENGINE_ID_MS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_PG_INTRSTAT4", regIntrStat);

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

        //
        // If a CTX_RESTORE interrupt is pending, determine whether the request
        // to turn off power-gating came from RM or HOST.
        //
        if (FLD_TEST_DRF(_PPWR, _PMU_PG_INTRSTAT, _CTX_RESTORE, _SET,
                         regIntrStat))
        {
            //
            // LW_PPWR_PMU_ELPG_PGOFF holds the power-gate control bits for
            // both ELPG0 and ELPG1.
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
        // If a CFG_ERR interrupt is pending, report an error as ELPG0 has been
        // configured incorrectly.
        //
        if (FLD_TEST_DRF(_PPWR, _PMU_PG_INTRSTAT, _CFG_ERR, _SET, regIntrStat))
        {
            PRINT_NEWLINE;
            PRINT_INDENTED_ERROR("_CFG_ERR set");
            status = LW_ERR_GENERIC;
        }
    }
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntrStat;

    return status;
}

//*****************************************************************************

LW_STATUS elpgDisplayPgOff_GK104(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regPgoff;
    LW_STATUS status = LW_OK;
    LwU32 val;

    //
    // LW_PPWR_PMU_ELPG_PGOFF holds the power-gate control bits for both ELPG0
    // and ELPG1.
    //
    regPgoff = ELPG_REG_RD32(LW_PPWR_PMU_PG_PGOFF);

    val = regPgoff & (DRF_SHIFTMASK(LW_PPWR_PMU_PG_PGOFF_ENG(elpgId)) |
                      DRF_SHIFTMASK(LW_PPWR_PMU_PG_PGOFF_REQSRC(elpgId)));
    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_PG_PGOFF (GR ELPG)", val);
    else if (elpgId == ELPG_ENGINE_ID_MS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_PG_PGOFF (MSCG)", val);

    PRINT_DRF_IDX_CONDITIONALLY(_PPWR, _PMU_PG_PGOFF, _ENG   , elpgId, regPgoff, "START", "DONE");
    PRINT_DRF_IDX_CONDITIONALLY(_PPWR, _PMU_PG_PGOFF, _REQSRC, elpgId, regPgoff, "HOST" , "RM");
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regPgoff;

    return status;
}

//*****************************************************************************

LW_STATUS elpgDisplayPgStat_GK104(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regStat;
    LW_STATUS status = LW_OK;
    LwU32 val;

    //
    // LW_PPWR_PMU_ELPG_STAT holds the state machine bits for both ELPG0 and
    // ELPG1.
    //
    regStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_STAT(elpgId));

    // Determine the current state of the ELPG state machine.
    val = regStat & DRF_SHIFTMASK(LW_PPWR_PMU_PG_STAT_EST);
    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_PG_STAT (GR ELPG)", val);
    else if (elpgId == ELPG_ENGINE_ID_MS)
        PRINT_REG_AND_VALUE("LW_PPWR_PMU_PG_STAT (MSCG)", val);

    if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _IDLE, regStat)         ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _POWERINGDOWN, regStat) ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE, regStat)       ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _POWERINGUP, regStat)   ||
        FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _PWROFF, regStat))
    {
        //
        // NOTE that only "one" of the following prints will succeed since
        // only one state can be active at a time.
        //
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _IDLE        , regStat, "IDLE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _POWERINGDOWN, regStat, "POWERINGDOWN");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE      , regStat, "ACTIVE");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _POWERINGUP  , regStat, "POWERINGUP");
        PRINT_DRF_IF_SET_TO(_PPWR, _PMU_PG_STAT, _EST, _PWROFF      , regStat, "PWROFF");
    }
    else
    {
        // Report the unrecognized state as an error.
        PRINT_NEWLINE;
        PRINT_INDENTED_ERROR("Unknown LW_PPWR_PMU_PG_STAT_EST value");
        status = LW_ERR_GENERIC;
    }
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal!= NULL)
        *regVal= regStat;

    return status;
}

//*****************************************************************************

LW_STATUS elpgDisplayPmuIntr1_GK104(LwU32 *regVal)
{
    LwU32 regIntr1;
    LW_STATUS status = LW_OK;
    LwU32 val;

    regIntr1 = ELPG_REG_RD32(LW_PPWR_PMU_INTR_1);
    val = regIntr1 & DRF_SHIFTMASK(LW_PPWR_PMU_INTR_1_HOLDOFF);
    PRINT_REG_AND_VALUE("LW_PPWR_PMU_INTR_1 (HOLDOFF)", val);

    PRINT_DRF_CONDITIONALLY(_PPWR, _PMU_INTR_1, _HOLDOFF, regIntr1, "PENDING", "NOT PENDING");
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regIntr1;

    return status;
}

//*****************************************************************************

LW_STATUS elpgDisplayEngHoldoffEnableStatus_GK104(LwU32 elpgId, LwU32 *regVal)
{
    LwU32 regHoldoff;
    LwU32 regStat;
    LW_STATUS status = LW_OK;
    LwU32 val;

    // LW_THERM_ENG_HOLDOFF denotes which engines are in "holdoff" mode.
    regHoldoff = ELPG_REG_RD32(LW_THERM_ENG_HOLDOFF);

    //
    // LW_PPWR_PMU_ELPG_STAT holds the state machine bits for both ELPG0 and
    // ELPG1.
    //
    regStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_STAT(elpgId));

    //
    // Determine if the engine(s) associated with the given elpgId are in
    // "holdoff" mode. An engine in "holdoff" mode means that methods to or a
    // context switch operation for that engine is lwrrently blocked.
    //

    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
    {
        // Determine if the graphics engine is in "holdoff" mode.
        val = regHoldoff & (DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(0)) |
                            DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(7)));
        PRINT_REG_AND_VALUE("LW_THERM_ENG_HOLDOFF (GR/GRCOPY)", val);

        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG0, regHoldoff,
                                "BLOCKED (GR)", "NOT BLOCKED (GR)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG7, regHoldoff,
                                "BLOCKED (GRCOPY)", "NOT BLOCKED (GRCOPY)");


        if (FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 0, _BLOCKED, regHoldoff) ||
            FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 7, _BLOCKED, regHoldoff))
        {

            //
            // If the graphics engine is blocked, then the state of the engine
            // controlled by ELPG0 must "not" be IDLE or ACTIVE.
            //
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _IDLE, regStat) ||
                FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE, regStat))
            {
                // Report the incorrect state denoted by ELPG0.
                PRINT_NEWLINE;
                PRINT_INDENTED_ERROR("_ENG0 (GR/GRCOPY) BLOCKED BUT ELPG0 IS " \
                                     "IDLE/ACTIVE");
                status = LW_ERR_GENERIC;
            }
        }
        else
        {
            //
            // If the graphics engine is "not" blocked, then the state of the
            // engine controlled by ELPG0 must "not" be PWROFF.
            //
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _PWROFF, regStat))
            {
                // Report the incorrect state denoted by ELPG0.
                PRINT_NEWLINE;
                PRINT_INDENTED_ERROR("_ENG0 (GR/GRCOPY) NOT BLOCKED BUT ELPG0 " \
                                     "IS PWROFF");
                status = LW_ERR_GENERIC;
            }
        }
    }
    else if (elpgId == ELPG_ENGINE_ID_MS)
    {
        val = regHoldoff & (DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(1)) |
                            DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(2)) |
                            DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(3)) |
                            DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(4)) |
                            DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(5)) |
                            DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_ENG(6)));
        PRINT_REG_AND_VALUE("LW_THERM_ENG_HOLDOFF (MS)", val);

        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG1, regHoldoff, "BLOCKED (MSPDEC)", "NOT BLOCKED (MSPDEC)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG2, regHoldoff, "BLOCKED (MSPPP)" , "NOT BLOCKED (MSPPP)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG3, regHoldoff, "BLOCKED (MSVLD)" , "NOT BLOCKED (MSVLD)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG4, regHoldoff, "BLOCKED (CE0)" , "NOT BLOCKED (CE0)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG5, regHoldoff, "BLOCKED (CE1)" , "NOT BLOCKED (CE1)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG6, regHoldoff, "BLOCKED (MSENC)" , "NOT BLOCKED (MSENC)");

        if (FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 1, _BLOCKED, regHoldoff) ||
            FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 2, _BLOCKED, regHoldoff) ||
            FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 3, _BLOCKED, regHoldoff) ||
            FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 4, _BLOCKED, regHoldoff) ||
            FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 5, _BLOCKED, regHoldoff) ||
            FLD_IDX_TEST_DRF(_THERM, _ENG_HOLDOFF, _ENG, 6, _BLOCKED, regHoldoff))
        {
            //
            // If a holdoffs for MSCG are blocked, then the state of the engine
            // controlled by ELPG4 must "not" be IDLE or ACTIVE.
            //
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _IDLE, regStat) ||
                FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE, regStat))
            {
                // Report the incorrect state denoted by ELPG1.
                PRINT_NEWLINE;
                PRINT_INDENTED_ERROR("_ENG1/2/3/4/5/6 (ALL VIDEO and CE0/1) BLOCKED BUT MS IS " \
                                     "IDLE/ACTIVE");
                status = LW_ERR_GENERIC;
            }
        }
        else
        {
            //
            // If a holdoffs for MSCG are "not" blocked, then the state of the
            // engines controlled by ELPG4 must "not" be PWROFF.
            //
            if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _PWROFF, regStat))
            {
                // Report the incorrect state denoted by ELPG4.
                PRINT_NEWLINE;
                PRINT_INDENTED_ERROR("_ENG1/2/3/4/5/6 (ALL VIDEO and CE0/1) BLOCKED BUT MS IS " \
                                     "IS PWROFF");
                status = LW_ERR_GENERIC;
            }
        }
    }
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regHoldoff;

    return status;
}

LW_STATUS elpgDisplayEngHoldoffPendingStatus_GK104
(
    LwU32  elpgId,
    LwU32 *regVal
)
{
    LwU32 regStatus;
    LW_STATUS status = LW_OK;
    LwU32 val;

    //
    // LW_THERM_ENG_HOLDOFF denotes which engines in "holdoff" mode have work
    // pending against them.
    //
    regStatus = ELPG_REG_RD32(LW_THERM_ENG_HOLDOFF_STATUS);

    //
    // Work pending against an engine means that a method for or a context
    // switch to the engine is blocked.
    //

    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
    {
        // Determine if the graphics engine has work pending.
        val = regStatus & DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(0));
        PRINT_REG_AND_VALUE("LW_THERM_ENG_HOLDOFF_STATUS (GR)", val);
        val = regStatus & DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(7));
        PRINT_REG_AND_VALUE("LW_THERM_ENG_HOLDOFF_STATUS (GRCOPY)", val);

        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF_STATUS, _ENG0, regStatus, "PENDING (GR)", "NOT PENDING (GR)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF_STATUS, _ENG7, regStatus, "PENDING (GRCOPY)", "NOT PENDING (GRCOPY)");
    }

    else if (elpgId == ELPG_ENGINE_ID_MS)
    {
        // Determine if a video engine is in "holdoff" mode.
        val = regStatus & (DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(1)) |
                           DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(2)) |
                           DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(3)) |
                           DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(4)) |
                           DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(5)) |
                           DRF_SHIFTMASK(LW_THERM_ENG_HOLDOFF_STATUS_ENG(6)));
        PRINT_REG_AND_VALUE("LW_THERM_ENG_HOLDOFF_STATUS (MSCG)", val);

        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG1, regStatus, "BLOCKED (MSPDEC)", "NOT PENDING (MSPDEC)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG2, regStatus, "BLOCKED (MSPPP)" , "NOT PENDING (MSPPP)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG3, regStatus, "BLOCKED (MSVLD)" , "NOT PENDING (MSVLD)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG4, regStatus, "BLOCKED (CE0)" , "NOT PENDING (CE0)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG5, regStatus, "BLOCKED (CE1)" , "NOT PENDING (CE1)");
        PRINT_DRF_CONDITIONALLY(_THERM, _ENG_HOLDOFF, _ENG6, regStatus, "BLOCKED (MSENC)" , "NOT PENDING (MSENC)");
    }
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regStatus;

    return status;
}

LW_STATUS elpgDisplayEngHoldoffStatus_GK104(LwU32 *regVal)
{
    LwU32 regHoldoff;
    LwU32 regStatus;
    LW_STATUS status = LW_OK;

    // LW_THERM_ENG_HOLDOFF denotes which engines are in "holdoff" mode.
    regHoldoff = ELPG_REG_RD32(LW_THERM_ENG_HOLDOFF);

    PRINT_REG_AND_VALUE("LW_THERM_ENG_HOLDOFF", regHoldoff);
    PRINT_NEWLINE;

    regStatus = ELPG_REG_RD32(LW_THERM_ENG_HOLDOFF_STATUS);
    PRINT_REG_AND_VALUE("LW_THERM_ENG_HOLDOFF_STATUS", regStatus);
    PRINT_NEWLINE;

    // Return back the register value if desired by the caller.
    if (regVal != NULL)
        *regVal = regHoldoff;

    return status;
}

LW_STATUS elpgPowerUp_GK104(LwU32 elpgId)
{
    LwU32 i;
    LwU32 regPgOff;
    LwU32 regStat;

    PRINT_NEWLINE;
    PRINT_STRING("**************************************************************************");
    dprintf     ("lw:  NOTE: ELPG%d must be in one of these states for it to be stopped:\n", elpgId);
    dprintf     ("lw:  IDLE, ACTIVE, POWERINGUP, or PWROFF (check with !elpg -status).\n");
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    //
    // LW_PPWR_PMU_PG_STAT holds the state machine bits for both ELPG0 and
    // ELPG4.
    //
    regStat = ELPG_REG_RD32(LW_PPWR_PMU_PG_STAT(elpgId));

    // Ensure that ELPG is in the correct state first.
    if (!(FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _IDLE, regStat)       ||
          FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _ACTIVE, regStat)     ||
          FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _POWERINGUP, regStat) ||
          FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _PWROFF, regStat)))
    {
        PRINT_INDENTED_ERROR("ELPG IS NOT IN A VALID STATE TO BE STOPPED");
        PRINT_NEWLINE;

        return LW_ERR_GENERIC;
    }

    PRINT_STRING("**************************************************************************");
    dprintf     ("lw:  Disabling ELPG%d and powering-on associated engine...\n", elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    if (elpgId == ELPG_ENGINE_ID_MS)
    {
        //
        // Disable MSCG in ISO-HUB if its not already disabled. This will
        // assert mspg_wake interrupt which will wake MS. Thus, no need to
        // write LW_PPWR_PMU_PGOFF.
        //
        regPgOff = ELPG_REG_RD32(LW_PDISP_ISOHUB_MISC_CTL);
        if (FLD_TEST_DRF(_PDISP, _ISOHUB_MISC_CTL, _MSPG, _ENABLE, regPgOff))
        {
            regPgOff = FLD_SET_DRF(_PDISP, _ISOHUB_MISC_CTL, _MSPG, _DISABLE,
                                   regPgOff);
            ELPG_REG_WR32(LW_PDISP_ISOHUB_MISC_CTL, regPgOff);

            // Display power-on request status
            pElpg[indexGpu].elpgDisplayPgOff(elpgId, &regPgOff);
        }
    }
    else // For GR and other(if supported) ELPG
    {
        //
        // Write LW_PPWR_PMU_PG_PGOFF if ELPG is in _PWROFF state
        //
        if (FLD_TEST_DRF(_PPWR, _PMU_PG_STAT, _EST, _PWROFF, regStat))
        {
            regPgOff = ELPG_REG_RD32(LW_PPWR_PMU_PG_PGOFF);

            //
            // Write LW_PPWR_PMU_ELPG_PGOFF in order to power-on the graphics
            // engine. The register is written in accordance with comments in
            // "dev_pwr_pri.ref".
            //
            for (i = 0; i < LW_PPWR_PMU_PG_PGOFF_ENG__SIZE_1; i++)
            {
                if (i == elpgId)
                {
                    regPgOff = FLD_IDX_SET_DRF(_PPWR, _PMU_PG_PGOFF, _ENG, i, _START, regPgOff);
                    regPgOff = FLD_IDX_SET_DRF(_PPWR, _PMU_PG_PGOFF, _REQSRC, elpgId, _HOST , regPgOff);
                }
                else
                {
                    regPgOff = FLD_IDX_SET_DRF(_PPWR, _PMU_PG_PGOFF, _ENG, i, _DONE , regPgOff);
                }
            }
            ELPG_REG_WR32(LW_PPWR_PMU_PG_PGOFF, regPgOff);

            // Ensure the power-on request went through.
            pElpg[indexGpu].elpgDisplayPgOff(elpgId, &regPgOff);
        }
    }

    return LW_OK;
}

/* ------------------------- HAL Functions ------------------------------- */

LW_STATUS elpgStart_GK104(LwU32 elpgId)
{
    LW_STATUS status = LW_OK;
    LwU32 regPgOn;

    PRINT_NEWLINE;
    PRINT_STRING("**************************************************************************");
    dprintf     ("lw:  NOTE: ELPG%d should have previously been stopped with !elpg -stop %d\n", elpgId, elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    if (pElpg[indexGpu].elpgIsEngineSupported(elpgId) != LW_TRUE)
    {
        PRINT_INDENTED_ERROR("ELPG on Specified Engine is not supported\n");
        return LW_ERR_NOT_SUPPORTED;
    }

    if (elpgId == ELPG_ENGINE_ID_GRAPHICS)
    {
        PRINT_NEWLINE;
        PRINT_STRING("**************************************************************************");
        dprintf     ("lw: \"!elpg -start 0\" option is not supported. (Bug 928692)\n");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        return LW_ERR_GENERIC;
    }
    // Enable idle monitoring for the given ELPG id.
    ELPG_REG_WR32(LW_PPWR_PMU_IDLE_CTRL_SUPP(elpgId),
                         LW_PPWR_PMU_IDLE_CTRL_SUPP_VALUE_IDLE);

    // Enable MSCG in ISO-HUB, if it is not enabled
    if (elpgId == ELPG_ENGINE_ID_MS)
    {
        regPgOn = ELPG_REG_RD32(LW_PDISP_ISOHUB_MISC_CTL);
        if (FLD_TEST_DRF(_PDISP, _ISOHUB_MISC_CTL, _MSPG, _DISABLE, regPgOn))
        {
            regPgOn = FLD_SET_DRF(_PDISP, _ISOHUB_MISC_CTL, _MSPG, _ENABLE,
                                  regPgOn);
            ELPG_REG_WR32(LW_PDISP_ISOHUB_MISC_CTL, regPgOn);
        }
    }

    // Output register values that have to do with idle monitoring.
    status = _elpgDisplayIdleCtrlSupp_GK104(elpgId, NULL);
    if (status != LW_OK)
        return status;

    PRINT_STRING("**************************************************************************");
    dprintf     ("lw:  ELPG%d PG_ON interrupt has been enabled to allow for power-gating.\n", elpgId);
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    return status;
}

/*!
 * This fuction handles PG disable request.
 *
 * We have new power feature named MSCG from Kepler+ GPUs. So, before disabling
 * GR ELPG LwWatch should consider state of MS. We need to ensure that MS is
 * disabled before disabling GR to avoid underflow. This fuction take care of
 * this.
 *
 * @param[in]   elpgId     Engine ID
 * 
 * @returns     LW_ERR_NOT_SUPPORTED    If engine is not supported
 *              LW_ERR_GENERIC            If fails to disable PG
 *              LW_OK               Otherwise
 */
LW_STATUS elpgStop_GK104(LwU32 elpgId)
{
    LW_STATUS status = LW_OK;
    LwBool bMSDisabled = LW_FALSE;
    LwU32 regIdleSupp;

    // First check whether given engine is supported or not
    if (pElpg[indexGpu].elpgIsEngineSupported(elpgId) != LW_TRUE)
    {
        PRINT_INDENTED_ERROR("ELPG on Specified Engine is not supported\n");
        return LW_ERR_NOT_SUPPORTED;
    }

    //
    // Before disabling GR ELPG check status of MS. If MS is supported and not
    // disabled then first disable MS then disable GR. 
    //
    if ((elpgId == ELPG_ENGINE_ID_GRAPHICS) &&
        (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_MS)))
    {
        regIdleSupp = ELPG_REG_RD32(LW_PPWR_PMU_IDLE_CTRL_SUPP(ELPG_ENGINE_ID_MS));

        if (!FLD_TEST_DRF(_PPWR, _PMU_IDLE_CTRL_SUPP, _VALUE, _NEVER,
                                                         regIdleSupp))
        {
            PRINT_STRING("**************************************************************************");
            dprintf     ("lw:  To avoid underflow we need to disable MSCG before disabling GR ELPG.\n");
            PRINT_STRING("**************************************************************************");

            status = _elpgStop_GK104(ELPG_ENGINE_ID_MS);
            if (status != LW_OK)
                return status;

            bMSDisabled = LW_TRUE;
        }
    }

    status = _elpgStop_GK104(elpgId);
    if (status != LW_OK)
    {
        // Restore state of MS if LwWatch fails to disable GR
        if(bMSDisabled)
            pElpg[indexGpu].elpgStart(ELPG_ENGINE_ID_MS);
    }

    return status;
}

LwBool elpgIsEngineSupported_GK104(LwU32 elpgId)
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
        default:
        {
            //dprintf("(ERROR - invalid id!)\n");
            PRINT_NEWLINE;
            return LW_FALSE;
        }
    }

    return bSupported;
}
