/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004
// lwwatchMods register reading routines
//*****************************************************

#include "lwwatch2.h"
#include "regMods.h"
#include "os.h" //For LW_ERROR definition

//
// Used for printing error messages in functions below.
//
#define DEBUGGER_ENGINE_ERROR_CHECK_STRING(funcName)                \
    "lw: " __FILE__ ":\n"                                           \
    "lw: " __FUNCTION__ ": " funcName " returned error. Register\n" \
    "lw: reading/writing MAY be incorrect. Error code 1. Error \n"  \
    "lw: codes in docs/README.lwwatchMods.txt\n"

#define DEBUGGER_ENGINE_ERROR_CHECK(funcName) \
        do \
        { \
            if (S_OK  != Hr) \
                dprintf((DEBUGGER_ENGINE_ERROR_CHECK_STRING(funcName))); \
        } while (0)


//
// Global used for RegRd/RegWr and Output communication
//
ReadStruct readStruct;

//
// lwwatchFp - used to store debug messages
//
#ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
extern FILE *lwwatchFp;
#endif


//-----------------------------------------------------
// RegRd 
// - Assumes that Address is a valid address. Sanity 
//   checking of Address should be done by callers
//-----------------------------------------------------
extern "C"
UINT32 RegRd
(
    char *command, 
    READ_SIZE _readSizeCmd, 
    UINT32 Address
)
{
    HRESULT Hr;
    char commandToWindbg[256];
    
    //
    // The output callback function uses this global to determine how many bits
    // to read.
    //
    readStruct.readSizeCmd = _readSizeCmd;
    
    Hr = ExtQuery();
    DEBUGGER_ENGINE_ERROR_CHECK("ExtQuery");
    if (Hr)
    {
        dprintf(__FUNCTION__ " failed.\n");
        switch(readStruct.readSizeCmd)
        {
            case READ32:
                return ILWALID_REG32;
            case READ16:
                return ILWALID_REG16;
            case READ08:
                return ILWALID_REG08;
            default:
                dprintf(__FUNCTION__ " Bug detected.\n");
                return ILWALID_REG32;
        }

    }

    //
    // Before we start calling the output callback, set readIsValid to be
    // false.
    //
    SET_READ_ILWALID();

    //
    // Set output mask so that debugger engine's output (something
    // like ".call returns unsigned int 0xdeadbeef") triggers a call
    // to output callback function.
    //
    Hr |= g_ExtClient->SetOutputMask(DEBUG_OUTPUT_NORMAL);
    DEBUGGER_ENGINE_ERROR_CHECK("SetOutputMask");

    //
    // Prepare register reading windbg command
    //
    sprintf(commandToWindbg,".call %s(%lu)", command, Address);

    //
    // Now ask the debugger engine to execute it in MODs address space for us
    //
    
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && debugOCMFlag>=1) 
            fprintf(lwwatchFp,"RegRd: Calling Exelwte1 with %s\n",
                    commandToWindbg);
    #endif
   
    Hr = g_ExtControl->Execute(
            DEBUG_OUTCTL_OVERRIDE_MASK | DEBUG_OUTCTL_THIS_CLIENT, 
            (PCSTR)commandToWindbg, 
            DEBUG_EXELWTE_DEFAULT);
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtControl->Exelwte1");
    
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && debugOCMFlag>=1) 
            fprintf(lwwatchFp,"RegRd: Calling Exelwte2 with %s\n",
                    FORCE_REG_OP_STRING);
    #endif

    //
    // To force the debugger engine to call the output callback for the above
    // command, we need it to Execute the command contained in
    // FORCE_REG_OP_STRING below
    //
    Hr = g_ExtControl->Execute(
            DEBUG_OUTCTL_OVERRIDE_MASK | DEBUG_OUTCTL_THIS_CLIENT,
            FORCE_REG_OP_STRING, 
            DEBUG_EXELWTE_DEFAULT);
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtControl->Exelwte2");

    //
    // This should force the debugger engine to call the output callback
    // function
    //
    Hr = g_ExtClient->FlushCallbacks();
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtClient->FlushCallbacks");

    //
    // At this point, the engine should have called the output callback function
    // Output. Control returns here after calling the output callback.
    //
    
    //
    // We don't want the engine to call us anymore. So, set the appropriate mask
    //
    Hr = g_ExtClient->SetOutputMask(DEBUG_OUTPUT_NONE);
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtClient->SetOutputMask");

    ExtRelease();

    //
    // If the output callback was successful in reading register values, then
    // the readIsValid() should be true and we return the number of bits asked
    // for by callers of RegRd
    //
    if (IS_READ_VALID())
    {
        //
        // set readIsValid to be false. In some rare cases, the output callback 
        // is called by the engine with garbage strings. We don't something to
        // be read during that call. The output callback will not be triggered.
        //
        SET_READ_ILWALID();

        switch(readStruct.readSizeCmd)
        {
            case READ32:
                return readStruct.readU032;
            case READ16:
                return readStruct.readU016;
            case READ08:
                return readStruct.readU008;
            default:
                dprintf(__FILE__ ": " __FUNCTION__ ": Bug.\n");
                return ILWALID_REG32;
        }
    }
    else    
        return ILWALID_REG32;    
}

//-----------------------------------------------------
// RegWr 
// - Assumes that Address is a valid address. Sanity
//   checking of Address should be done by callers.
//   Works similar to RegRd above.
//-----------------------------------------------------
extern "C"
BOOL 
RegWr
(
    char *command, 
    UINT32 Address, 
    UINT32 value
)
{
    HRESULT Hr;
    char commandToWindbg[256];
    
    Hr = ExtQuery();
    DEBUGGER_ENGINE_ERROR_CHECK("ExtQuery");
    if (Hr)
    {
        dprintf(__FUNCTION__ " failed.\n");
        return FALSE;
    }

    Hr = g_ExtClient->SetOutputMask(DEBUG_OUTPUT_NORMAL);
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtClient->SetOutputMask");

    //
    // Prepare register reading windbg command
    //
    sprintf(commandToWindbg,".call %s(%lu,%lu)", command, Address, value);
    
    //
    // Signal Output Callback to not attempt to read anything
    //
    readStruct.readSizeCmd = READ00;

    Hr |= g_ExtControl->Execute(
            DEBUG_OUTCTL_OVERRIDE_MASK | DEBUG_OUTCTL_THIS_CLIENT, 
            (PCSTR)commandToWindbg, 
            DEBUG_EXELWTE_DEFAULT);
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtControl->Execute");

    Hr |= g_ExtControl->Execute(
            DEBUG_OUTCTL_OVERRIDE_MASK | DEBUG_OUTCTL_THIS_CLIENT,
            FORCE_REG_OP_STRING, 
            DEBUG_EXELWTE_DEFAULT);
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtControl->Execute");

    Hr |= g_ExtClient->FlushCallbacks();
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtClient->FlushCallbacks");

    Hr |= g_ExtClient->SetOutputMask(DEBUG_OUTPUT_NONE);
    DEBUGGER_ENGINE_ERROR_CHECK("g_ExtClient->SetOutputMask");

    ExtRelease();
    
    if (Hr) return FALSE;
    else    return TRUE;    
}
