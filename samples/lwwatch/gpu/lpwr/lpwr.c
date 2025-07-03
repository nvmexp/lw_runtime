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
// Global defines
//*****************************************************************************
// GR PG      - Graphics Power Gating.
// GR PASSIVE - Skeletal/Minimal Graphics FSM for decoupled MSCG support.
// GR RG      - GPC Rail Gating.
// EI         - Engine Idle.
// MS         - Memory Subsystem Clock Gating.
char *LpwrEng[ELPG_MAX_SUPPORTED_ENGINES] = {"GR PG","GR PASSIVE" ,"GR RG", "EI", "MS", "DIFR PREFETCH", "DIFR SW ASR" , "DIFR CG"};

//*****************************************************************************
// Global functions
//*****************************************************************************

void lpwrDisplayHelp(void)
{
    dprintf("LPWR Routines:\n");
    dprintf(" lpwr -help                - Displays the ELPG command help menu.\n");
    dprintf(" lpwr -status              - Dumps ELPG state.\n");
    dprintf(" lpwr -start <elpgId>      - Restarts usage of the given ELPG controller (0 or 1).\n");
    dprintf("                             NOTE: This command should only be called after !lpwr -stop <elpgId> has run.\n");
    dprintf(" lpwr -stop <elpgId>       - Stops usage of the given ELPG controller (0 or 1) and powers-on its associated engine.\n");
    dprintf(" lpwr -dumpLog             - Dumps all the entries of the PG log stored in DMEM.\n");
    dprintf("\n");
}

/**
 * @brief Lwwatch Extension to display all LPWR FSM data.
 *
 * This extension dumps all the information for all supported PG/LPWR engines.
 * Any generic data to be dumped, should be added in this extension.
 * Most frequently, required data should also be included in lpwrGetFsmState.
 * 
 * @param   void
 * 
 * @return  LW_OK if success
 *          LW error code otherwise.
 **/
LW_STATUS lpwrGetStatus(void)
{
    LwU32 LpwrEngId;
    LwU32 Rval;
    LW_STATUS Status = LW_OK;

    //*************************************************************************
    // EI is not supported on pre Turing chips.
    //*************************************************************************
    if (!IsTU102orLater())
    {
        LpwrEng[ELPG_ENGINE_ID_EI] = "DI";
    }

    //*************************************************************************
    // Lpwr Generic Status Dump
    //*************************************************************************

    PRINT_STRING("****************************************************************************");
    PRINT_STRING("LPWR Generic Information Dump - Idle Signal Status                          ");
    PRINT_STRING("****************************************************************************");
    PRINT_NEWLINE;

    Rval = pElpg[indexGpu].elpgDisplayPmuPgIdleSignal();
    if (Rval != LW_OK)
    {
        Status = Rval;
    }

    //*************************************************************************
    // PG ENG/LPWR ENG Dump
    //*************************************************************************

    PRINT_NEWLINE;
    PRINT_STRING("**************************************************************************");
    PRINT_STRING("LPWR PgCtrl Status                                                        ");
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    for(LpwrEngId = ELPG_ENGINE_ID_GRAPHICS; LpwrEngId < ELPG_MAX_SUPPORTED_ENGINES; LpwrEngId++)
    {
        if(pElpg[indexGpu].elpgIsEngineSupported(LpwrEngId))
        {
            PRINT_STRING("**************************************************************************");
            PRINT_STRING_AND_FIELD_VALUE(Checking PG_ENG(%d)/LPWR_ENG(%d) (%s) for interrupts and Status...,
                                                                   LpwrEngId, LpwrEngId, LpwrEng[LpwrEngId]);
            PRINT_STRING("**************************************************************************");
            PRINT_NEWLINE;

            // PG ENG Configuration stat
            Rval = pElpg[indexGpu].elpgDisplayPgEngConfig(LpwrEngId);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Determine the current state of the PG_ENG state machine.
            Rval = pElpg[indexGpu].elpgDisplayPgStat(LpwrEngId, NULL);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Determine which PG_ENG interrupts are enabled.
            Rval = pElpg[indexGpu].elpgDisplayPgIntrEn(LpwrEngId, NULL);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Determine which of the enabled interrupts are pending, if any.
            Rval = pElpg[indexGpu].elpgDisplayPgIntrStat(LpwrEngId, NULL);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Determine the Sequencer state.
            Rval = pElpg[indexGpu].elpgDisplaySequencerState(LpwrEngId);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Dump the MSCG Blocker Status.
            if(LpwrEngId == ELPG_ENGINE_ID_MS)
            {
                Rval = pElpg[indexGpu].elpgDisplayMsBlockerAndInterruptState();
                if (Rval != LW_OK)
                {
                    Status = Rval;
                }
            }
        }
    }

    PRINT_NEWLINE;
    // Determine if the graphics engine is in "holdoff" mode.
    Rval = pElpg[indexGpu].elpgDisplayEngHoldoffStatus(NULL);
    if (Rval != LW_OK)
    {
        Status = Rval;
    }

    //*************************************************************************
    // PMU to RM Interrupt Status
    //*************************************************************************

    PRINT_STRING("**************************************************************************");
    PRINT_STRING("Checking whether an interrupt is pending from PMU to RM...");
    PRINT_STRING("**************************************************************************");
    PRINT_NEWLINE;

    Rval = pElpg[indexGpu].elpgDisplayPmuIntr1(NULL);
    if (Rval != LW_OK)
    {
        Status = Rval;
    }

    return Status;
}

/**
 * @brief Lwwatch Extension to display frequent LPWR FSM state.
 *
 * Display the PG status, Interrupt enabled and status and SW Client status.
 * This extension dumps most frequently used data for LPWR FSM.
 * Generic data should be added only in lpwrState extension.
 *
 * @param   void
 * 
 * @return  LW_OK if success
 *          LW error code otherwise.
 **/
LW_STATUS lpwrGetFsmState(void)
{
    LwU32 elpgId;
    LwU32 Rval;
    LwU32 Status = LW_OK;

    for(elpgId = ELPG_ENGINE_ID_GRAPHICS; elpgId < ELPG_MAX_SUPPORTED_ENGINES; elpgId++)
    {
        if(pElpg[indexGpu].elpgIsEngineSupported(elpgId))
        {
            PRINT_STRING("****************************************************************************");
            PRINT_STRING_AND_FIELD_VALUE(Dumping LPWR ENG(%d) %s interrupts and Status, elpgId, LpwrEng[elpgId]);
            PRINT_STRING("****************************************************************************");

            // Determine the current state of the PG_ENG state machine.
            Rval = pElpg[indexGpu].elpgDisplayPgStat(elpgId, NULL);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Determine which LPWR_ENG interrupts are enabled.
            Rval = pElpg[indexGpu].elpgDisplayPgIntrEn(elpgId, NULL);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Determine any of the enabled interrupts are pending.
            Rval = pElpg[indexGpu].elpgDisplayPgIntrStat(elpgId, NULL);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }

            // Get SW Client status.
            Rval = pElpg[indexGpu].elpgDisplaySwClientStatus(elpgId);
            if (Rval != LW_OK)
            {
                Status = Rval;
            }
        }
    }

    return Status;
}
