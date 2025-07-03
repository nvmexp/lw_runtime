/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "os.h"
#include "hal.h"
#include "elpg.h"

//*****************************************************************************
// Global functions
//*****************************************************************************

void elpgDisplayHelp(void)
{
    dprintf("ELPG Routines:\n");
    dprintf(" elpg -help                - Displays the ELPG command help menu.\n");
    dprintf(" elpg -status              - Dumps ELPG state.\n");
    dprintf(" elpg -start <elpgId>      - Restarts usage of the given ELPG controller (0 or 1).\n");
    dprintf("                             NOTE: This command should only be called after !elpg -stop <elpgId> has run.\n");
    dprintf(" elpg -stop <elpgId>       - Stops usage of the given ELPG controller (0 or 1) and powers-on its associated engine.\n");
    dprintf(" elpg -dumpLog             - Dumps all the entries of the PG log stored in DMEM.\n");
    dprintf("\n");
}


LW_STATUS elpgGetStatus(void)
{
    LwU32 err;
    LW_STATUS status = LW_OK;

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_GRAPHICS))
    {
        //*************************************************************************
        // ELPG0 - Controls power gating for graphics engine (GR).
        //*************************************************************************

        PRINT_NEWLINE;
        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(0) (Graphics - GR) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(0) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_GRAPHICS, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(0) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_GRAPHICS, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_GRAPHICS, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_VIDEO))
    {
        //*************************************************************************
        // ELPG1 - Controls power gating for video engines (MSVLD, MSPDEC, MSPPP),
        //         all at the same time.
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking ELPG1 (Video - MSVLD, MSPDEC, MSPPP) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the ELPG1 state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_VIDEO, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which ELPG1 interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_VIDEO, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_VIDEO, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_MS))
    {
        //*************************************************************************
        // PG_ENG(4) - Controls clock gating for memory system 
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(4) (MSCG) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(4) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_MS, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(4) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_MS, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_MS, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_DI))
    {
        //*************************************************************************
        // PG_ENG(3) - Gates all clock in GPU and lowers LWVDD voltage.
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(3) (DI) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(3) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_DI, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(3) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_DI, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_DI, NULL);
        if (err != LW_OK)
            status = err;
    }

        if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_MS_LTC))
    {
        //*************************************************************************
        // PG_ENG(4) - Controls clock gating for memory system.
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(4) (MS LTC) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(4) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_MS_LTC, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(4) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_MS_LTC, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_MS_LTC, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_MS_PASSIVE))
    {
        //*************************************************************************
        // PG_ENG(4) - Controls clock gating for memory system.
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(4) (MS PASSIVE) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(4) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_MS_PASSIVE, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(4) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_MS_PASSIVE, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_MS_PASSIVE, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_EI_PASSIVE))
    {
        //*************************************************************************
        // PG_ENG(3) - Dummy EI sequence or characterization profiling on different chips at different PSTATEs
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(3) (EI PASSIVE) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(3) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_EI_PASSIVE, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(3) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_EI_PASSIVE, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_EI_PASSIVE, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_DIFR_PREFETCH))
    {
        //*************************************************************************
        // PG_ENG(5) - Controls clock gating for DIFR PREFETCH
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(5) (DIFR PREFETCH) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(5) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_DIFR_PREFETCH, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(5) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_DIFR_PREFETCH, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_DIFR_PREFETCH, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_DIFR_SW_ASR))
    {
        //*************************************************************************
        // PG_ENG(6) - Controls clock gating for DIFR Software ASR
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(6) (DIFR SW ASR) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(6) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_DIFR_SW_ASR, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(6) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_DIFR_SW_ASR, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_DIFR_SW_ASR, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_DIFR_CG))
    {
        //*************************************************************************
        // PG_ENG(7) - Controls clock gating for DIFR. 
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking PG_ENG(7) (DIFR CG) for interrupts and status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine the current state of the PG_ENG(7) state machine.
        err = pElpg[indexGpu].elpgDisplayPgStat(ELPG_ENGINE_ID_DIFR_CG, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which PG_ENG(7) interrupts are enabled.
        err = pElpg[indexGpu].elpgDisplayPgIntrEn(ELPG_ENGINE_ID_DIFR_CG, NULL);
        if (err != LW_OK)
            status = err;

        // Determine which of the enabled interrupts are pending, if any.
        err = pElpg[indexGpu].elpgDisplayPgIntrStat(ELPG_ENGINE_ID_DIFR_CG, NULL);
        if (err != LW_OK)
            status = err;
    }

    // Check engines for Holdoff and PRI access status

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_GRAPHICS))
    {
        //*************************************************************************
        // Graphics (GR) Engine (Engine 0)
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking Graphics engine (engine 0) holdoff and pri access status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine if the graphics engine is in "holdoff" mode.
        err = pElpg[indexGpu].elpgDisplayEngHoldoffEnableStatus(ELPG_ENGINE_ID_GRAPHICS, NULL);
        if (err != LW_OK)
            status = err;

        //
        // In case the graphics engine is in "holdoff" mode, determine if it has
        // work pending (methods or a context switch blocked).
        //
        err = pElpg[indexGpu].elpgDisplayEngHoldoffPendingStatus(ELPG_ENGINE_ID_GRAPHICS, NULL);
        if (err != LW_OK)
            status = err;

        // Determine if the graphics engine has its PRI access blocked.
        err = pElpg[indexGpu].elpgDisplayPrivAccessConfig(ELPG_ENGINE_ID_GRAPHICS, NULL);
        if (err != LW_OK)
            status = err;
    }

    if (pElpg[indexGpu].elpgIsEngineSupported(ELPG_ENGINE_ID_VIDEO))
    {
        //*************************************************************************
        // Video (GR) Engine (Engines 3,4,5)
        //*************************************************************************

        PRINT_STRING("**************************************************************************");
        PRINT_STRING("Checking Video engines (engines 3,4,5) holdoff and pri access status...");
        PRINT_STRING("**************************************************************************");
        PRINT_NEWLINE;

        // Determine if the video engines are in "holdoff" mode.
        err = pElpg[indexGpu].elpgDisplayEngHoldoffEnableStatus(ELPG_ENGINE_ID_VIDEO, NULL);
        if (err != LW_OK)
            status = err;

        //
        // In case the video engine is in "holdoff" mode, determine if it has work
        // pending (methods or a context switch blocked).
        //
        err = pElpg[indexGpu].elpgDisplayEngHoldoffPendingStatus(ELPG_ENGINE_ID_VIDEO, NULL);
        if (err != LW_OK)
            status = err;

        // Determine if the video engine has its PRI access blocked.
        err = pElpg[indexGpu].elpgDisplayPrivAccessConfig(ELPG_ENGINE_ID_VIDEO, NULL);
        if (err != LW_OK)
            status = err;
    }

    //*************************************************************************
    // PMU to RM Interrupt Status
    //*************************************************************************

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Checking whether an interrupt is pending from PMU to RM...");
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    err = pElpg[indexGpu].elpgDisplayPmuIntr1(NULL);
    if (err != LW_OK)
        status = err;

    return status;
}
