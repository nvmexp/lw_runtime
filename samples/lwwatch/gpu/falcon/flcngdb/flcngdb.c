/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcngdb.c
 * @brief WinDbg and GDB Extension for Falcon.
 * 
 */

#include "flcngdb.h"
#include "flcngdbUtilsWrapper.h"

// needed for source code window UI
#ifdef WIN32
    #include "flcngdbUI.h"
#endif

#include "pmu.h"
#include "dpu.h"


#include "kepler/gk107/dev_pwr_pri.h"

// TODO : Need to check for falcon5, But for now for debugger falcon 4 reg address space will work 
// for falcon5.
#include "kepler/gk107/dev_falcon_v4.h" 

/* ------------------------ Structures ------------------------------------- */

// saves the CPP Flcngdb class pointer between sessions
CFlcngdbUtils* pFlcngdbUtilsCls = NULL;

// holds the register mapping for the current falcon
FLCNGDB_REGISTER_MAP flcngdbRegisterMap;

// saves the interrupt status between sessions so they can be restored
LwU32 flcngdbSavedInterrupts;

static void flcngdbPrepareICD();
static LwU32 flcngdbQueryICDStatus();
static LwBool flcngdbWaitForBP(LwS32 timeoutMillis);
static LwBool flcngdbWriteICDCmd(LwU32 regVal);
static int flcngdbResumeExelwtion();
static void flcngdbWaitForICDDataReady();
static void flcngdbAdvanceOneInstruction();
static void flcngdbAdvanceToNext(CFlcngdbUtils* pCls);
static void flcngdbClearBp();
static LwU32 flcngdbGetBpAddress();
static LwU32 flcngdbReadWordDMEM(LwU32 address);
static void flcngdbDumpSymbol(const char* symbolName);
static void flcngdbSetBpByAddr(LwU32 addr);
static void flcngdbDisableInterrupts();
static void flcngdbRestoreInterrupts();
static LwU32 flcngdbReadFalcPc();
static LwU32 flcngdbGetAddressFromFunctionName(const char* name);
static LwU32 flcngdbGetAddressFromFileAndLine(const char *file, unsigned int line);
static void flcngdbDisplaySource(CFlcngdbUtils* pCls);
static void flcngdbPrintHelp();
static void flcngdbCleanAll();



// put the ICD into appropriate mode for debugging
static void flcngdbPrepareICD()
{
    // set EMASK
    LwU32 regVal = 0;

    // 
    // enable EMASK and set to drop into ICD on IBRK
    // 0x00800007
    // 
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _OPC, _EMASK, regVal);
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _EMASK_EXC_IBREAK, _TRUE,
                         regVal);

    flcngdbWriteICDCmd(regVal);
}

// returns the ICD status register after querying
static LwU32 flcngdbQueryICDStatus()
{
    LwU32 regVal = 0;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;
    LwBool result = FALSE;

    // RSTAT with IDX 4 gives us the current processor and ICD status
    // 0x0000040E
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _OPC, _RSTAT, regVal);
    regVal = FLD_SET_DRF_NUM(_PPWR_FALCON, _ICD_CMD, _IDX, 0x4, regVal);

    result = flcngdbWriteICDCmd(regVal);

    flcngdbWaitForICDDataReady();

    return FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_RDATA);
}

//
// busy loop to wait for the falcon to halt and drop into ICD (for any reason)
// returns FALSE only if timeout, TRUE if it hit anything
// 
static LwBool flcngdbWaitForBP(LwS32 timeoutMillis)
{
    LwU32 regVal = 0;
    LwU32 pc;
    LwU32 bp;
    LwBool bHasTimeout;

    if (timeoutMillis != 0)
        bHasTimeout = TRUE;
    else
        bHasTimeout = FALSE;

    while (TRUE)
    {
        pc = flcngdbReadFalcPc();
        dprintf("PC: %x\n", pc);
        regVal = flcngdbQueryICDStatus();
        dprintf("RDATA: %x\n", regVal);
        if (FLD_TEST_DRF(_PPWR_FALCON, _ICD_RDATA, _RSTAT4_ICD_STATE,
                         _FULL_DBG_MODE, regVal))
        {
            bp = flcngdbGetBpAddress();
            if (pc != bp)
            {
                dprintf("PC halted at a point that is not the BP (IBRKPT1), PC: %u, BP: %u\n", pc, bp);
            } 
            else
            {   
                //
                // if we hit the breakpoint, we want to disable the interrupt
                // so that when you step there isnt a huge number of timer
                // interrupts that takes you into code sections you dont want
                //
                // TODO: This can be changed later when stack trace is supported
                //       in which case we dont need to disable the interrupt
                //       anymore since its all source level stepping by then
                // 
                dprintf("Breakpoint 1 hit at %u\n", pc);
                flcngdbDisableInterrupts();
            }
            break;
        }
        osPerfDelay(100 * 1000);
        //
        // if there is a timeout on the wait then decrease time and see
        // if we should exit
        if (bHasTimeout)
        {
            timeoutMillis -= 100;
            if (timeoutMillis <= 0)
            {
                return FALSE;
            }
        }
    }

    return TRUE;
}

// write a ICD command and confirm that the command is properly accepted
static LwBool flcngdbWriteICDCmd(LwU32 regVal)
{
    LwU32 engineBase = flcngdbRegisterMap.registerBase; 

    // write to the ICD command register and wait for a while
    FLCN_REG_WR32(LW_PFALCON_FALCON_ICD_CMD, regVal);

    osPerfDelay(FLCNGDB_CMD_WAIT_TIME * 1000);

    // see if there was an error exelwting the last command
    regVal = FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_CMD);

    // regVal & 0x4000 != 0
    if (FLD_TEST_DRF(_PPWR_FALCON, _ICD_CMD, _ERROR, _TRUE, regVal))
    {
        dprintf("The last ICD command could not be completed\n");
        return FALSE;
    }

    return TRUE;
}

// single instruction step
static void flcngdbAdvanceOneInstruction()
{
    // write command 0x5
    LwU32 regVal = 0;
    LwBool result =  FALSE;

    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _OPC, _STEP, regVal);

    result = flcngdbWriteICDCmd(regVal);
    if (!result)
    {
        dprintf("flcngdbAdvance not performed\n");
    }
}

// single step until line number changes
//
// TODO: this function does not yet behave like an actual step since it will
//       stop on any new line that belonged to a new function
//       Once stacktrace support is in this can be changed
// 
static void flcngdbAdvanceToNext(CFlcngdbUtils* pCls)
{
    LwU32 pc;
    char oldF[FLCNGDB_FILENAME_MAX_LEN];
    char newF[FLCNGDB_FILENAME_MAX_LEN];
    LwU32 oldLine;
    LwU32 newLine;
    LwBool bRun = TRUE;
    LwBool bIgnore = FALSE;

    pc = flcngdbReadFalcPc();
    flcngdbGetFileMatchingInfo(pc, oldF, &oldLine, pFlcngdbUtilsCls);
    newLine = oldLine;
    strcpy(newF, oldF);

    // if we cannot resolve the first line the let the user step to a line that can be
    // resolved before allowing next
    if (oldLine == 0)
    {
        dprintf("Current PC has no line number matching information,"
                "step (s) to a line that does before using next\n");
        return;
    }

    while (bRun)
    {
        oldLine = newLine;
        strcpy(oldF, newF);
        flcngdbAdvanceOneInstruction();
        pc = flcngdbReadFalcPc();
        dprintf("PC: %x\n", pc);
        flcngdbGetFileMatchingInfo(pc, newF, &newLine, pFlcngdbUtilsCls);
        dprintf("Matching is %s:%u\n", newF, newLine);

        //
        // if we could not matching the previous PC to a line, single step again anyway
        // (i.e. we will always stop on a pc that has a source matching
        // 
        if ((newLine == 0) || ((newLine == oldLine) && (strcmp(newF, oldF) == 0)) || bIgnore)
        {
            bRun = TRUE;
        } else
        {
            bRun = FALSE;
        }
    }
}

// resume the CPU from a halt state
static int flcngdbResumeExelwtion()
{
    LwU32 regVal = 0;

    //
    // enable interrupts again (that was disabled when th BP was hit in waitForBP)
    // and then send the run command to ICD
    // 
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _OPC, _RUN, regVal);
    flcngdbRestoreInterrupts();

    return flcngdbWriteICDCmd(regVal);
}

// busy loop to wait for ICD data return. This can be called after a data read
// instruction that returns data to RDATA
// 
static void flcngdbWaitForICDDataReady()
{
    LwU32 regVal = 0;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    //
    // busy loop until ICD has data
    // TODO: Implement timeout here
    // 
    while (1)
    {
        regVal = FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_CMD);
        // test for valid return data bit set (regVal & 0x8000 != 0)
        if (FLD_TEST_DRF(_PPWR_FALCON, _ICD_CMD, _RDVLD, _TRUE, regVal))
            break;
        osPerfDelay(10 * 1000);
    }
}

// clear breakpoint
static void flcngdbClearBp()
{
    LwU32 engineBase = flcngdbRegisterMap.registerBase;
    FLCN_REG_WR32(LW_PFALCON_FALCON_IBRKPT1, 0x0);
}

// insert a breakpoint at address addr, the breakpoint is set to enabled
// and exceptiosn suppress
static void flcngdbSetBpByAddr(LwU32 addr)
{
    LwU32 regVal = 0;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    //
    // bp = addr & 0x00FFFFFF | 0xA0000000;
    // set the PC
    // 
    regVal = FLD_SET_DRF_NUM(_PPWR_FALCON, _IBRKPT1, _PC, addr, regVal);

    //
    // set Enable and Suppress flags (we want to drop into ICD without
    // triggering exception interrupt)
    // 
    regVal = FLD_SET_DRF(_PPWR_FALCON, _IBRKPT1, _SUPPRESS, _ENABLE, regVal);
    regVal = FLD_SET_DRF(_PPWR_FALCON, _IBRKPT1, _EN, _ENABLE, regVal);

    FLCN_REG_WR32(LW_PFALCON_FALCON_IBRKPT1, regVal);
    dprintf("Set IBPT1 to %x\n", FLCN_REG_RD32(LW_PFALCON_FALCON_IBRKPT1));
}

// return the lwrrently set breakpoint address
static LwU32 flcngdbGetBpAddress()
{
    LwU32 regVal;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    regVal = FLCN_REG_RD32(LW_PFALCON_FALCON_IBRKPT1);

    return(DRF_NUM(_PPWR_FALCON, _IBRKPT1, _PC, regVal));
}

// save the current interrupt register and mask all interrupts
static void flcngdbDisableInterrupts()
{
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    // read the interrupt enable mask
    flcngdbSavedInterrupts = FLCN_REG_RD32(LW_PFALCON_FALCON_IRQMASK);

    FLCN_REG_WR32(LW_PFALCON_FALCON_IRQMCLR, 0xFFFFFFFF);
    dprintf("Disabling interrupts (%x)\n", flcngdbSavedInterrupts);
}

// restore the previously saved interrupt register
static void flcngdbRestoreInterrupts()
{
    LwU32 engineBase = flcngdbRegisterMap.registerBase;
    FLCN_REG_WR32(LW_PFALCON_FALCON_IRQMSET, flcngdbSavedInterrupts);
    dprintf("Restoring interrupts (%x)\n", flcngdbSavedInterrupts);
}

// return the PC of the falcon
static LwU32 flcngdbReadFalcPc()
{
    LwU32 regVal = 0;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    //LW_PFALCON_FALCON_ICD_CMD_OPC_RREG | LW_PFALCON_FALCON_PC_IDX | (0x2 << 6);
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _OPC, _RREG, regVal);
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _IDX, _PC, regVal);

    FLCN_REG_WR32(LW_PFALCON_FALCON_ICD_CMD, regVal);

    flcngdbWaitForICDDataReady();

    return FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_RDATA); 
}

#ifndef WIN32
//
// function reads the next line (delimited by \n) from fp
// only needed for non-windows lwwatch, the windows version has UI support
// 
char* flcngdbReadLine(FILE* fp, LwBool returnData)
{
    char* buffer;
    char line[128];
    char* t;

    buffer = calloc(1,1);
    if (!buffer)
    {
        dprintf("Could not allocate buffer\n");
        return NULL;
    }

    for (; fgets(line, sizeof line, fp); strcat(buffer, line))
    {
        t = strchr(line, '\n');
        buffer = (char*) realloc(buffer, strlen(buffer) + strlen(line) + 1);
        if (!buffer)
        {
            dprintf("Could not reallocate buffer\n");
            return NULL;
        }
        if (t)
        {
            *t = '\0';
            return strcat(buffer, line);
        }
    }

    return buffer;
}
#endif

// read one word from DMEM
static LwU32 flcngdbReadWordDMEM(LwU32 address)
{
    LwU32  regVal = 0;
    LwU32  engineBase = flcngdbRegisterMap.registerBase;
    LwBool result = FALSE;

    // put the destination into the ADDR register
    FLCN_REG_WR32(LW_PFALCON_FALCON_ICD_ADDR, address);

    // issue a read command 0x0000000A | (2 << 6)
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _OPC, _RDM, regVal);
    regVal = FLD_SET_DRF(_PPWR_FALCON, _ICD_CMD, _SZ, _W, regVal);

    result = flcngdbWriteICDCmd(regVal);
    // if (result == FALSE)
    //     return 0;

    flcngdbWaitForICDDataReady();

    return FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_RDATA);
}

// dumps a global symbol
static void flcngdbDumpSymbol(const char* symbolName)
{
    // Reason why pmusym functions not reused:
    //
    // 1. this is a stub function for now. It is mimicing the function of pmusym dump but
    //    will be changed to use DWARF debug_info to dump the actual structure of the vars
    // 2. This works on more than just the PMU, it is supposed to be expanded onto the DPU
    //    too
    // 3. Falcon is halted at this point
    // 

    LwU32 startAddress;
    LwU32 endAddress;
    LwU32 len;
    LwU32 i;
    LwU32 c = 0;

    flcngdbGetSymbolInfo(symbolName, &startAddress, &endAddress, pFlcngdbUtilsCls);
    if (startAddress == FLCGDB_ERR_ILWALID_VALUE)
    {
        dprintf("Symbol not found (%s)\n", symbolName);
        return;
    }

    len = endAddress - startAddress + 1;
    // not formatting the string, its temporary anyway (replacing with var dump)
    dprintf("Content:\n");
    for (i = startAddress; i < (startAddress + len); i+=4 )
    {
        // mask to get rid of the leading 1 for DMEM
        dprintf("%x ", flcngdbReadWordDMEM(0x00FFFFFF & i));
        // we want to print the dump out in blocks
        c++;
        if (c == 4)
        {
            dprintf("\n");
            c = 0;
        }

    }
    dprintf("\n");
}

static void flcngdbCleanSession(const char *sessionID)
{
    if (flcngdbDeleteSession(sessionID, pFlcngdbUtilsCls) != 0)
    {
        dprintf("flcngdb :Error given session name doesn't exist \n");
    } else
    {
        dprintf("flcngdb: Seesion deleted \n");
    }
}

//
// displays the source code corresponding to the current PC
// OS dependent: Windows has UI, other OSes will print to console
// 
static void flcngdbDisplaySource(CFlcngdbUtils* pCls)
{
    LwU32 pc;
    LwU32 line;
    char f[FLCNGDB_FILENAME_MAX_LEN];
#ifndef WIN32
    FILE* fp;
    LwU32 start;
    LwU32 end;
    LwU32 i;
    char* tLine;
#endif

    // get the current PC
    pc = flcngdbReadFalcPc();

    // find the filename and line number corresponding to the pc
    flcngdbGetFileMatchingInfo(pc, f, &line, pFlcngdbUtilsCls);
    if (line == 0)
    {
        dprintf("Source cannot be found (PC %x does not match table)\n", pc);
        return;
    }
    dprintf("Loading source from %s:%u\n", f, line);

    // symbol line number starts at 1
    line--;

    //
    // if we are not on Windows then we will print the source code into
    // the current console window, otherwise we can display the soruce in GUI
    // 
#ifdef WIN32
    flcngdbUiLoadFileFlcngdbWindow(f);
    flcngdbUiCenterOnLineFlcngdbWindow(line);
#else
    fp = fopen(f, "r");
    if (fp == NULL)
    {
        dprintf("Source file %s could not be opened\n", f);
        return;
    }

    // clip the line number to min 0
    if (line < FLCNGDB_LOAD_SOURCE_LINES)
    {
        start = 0;
    } else
    {
        start = line - FLCNGDB_LOAD_SOURCE_LINES;
    }
    end = line + FLCNGDB_LOAD_SOURCE_LINES;

    // print the actual source code
    dprintf("\n-------------------------------------------------\n");
    for (i=0; i<end; ++i)
    {
        tLine = flcngdbReadLine(fp, TRUE);
        if ((start <= i) && (i <= end))
        {
            if (i == line)
            {
                dprintf("bp> ");
            } else
            {
                dprintf("    ");
            }
            dprintf("%s\n", tLine);
        }

        if (feof(fp))
        {
            break;
        }
    }
    dprintf("-------------------------------------------------\n");

    fclose(fp);
#endif
}

// returns the address of a function (first address)
static LwU32 flcngdbGetAddressFromFunctionName(const char* name)
{
    LwU32 startAddress;
    LwU32 endAddress;

    flcngdbGetFunctionInfo(name, &startAddress, &endAddress, pFlcngdbUtilsCls);
    if (startAddress == FLCGDB_ERR_ILWALID_VALUE)
    {
        dprintf("Function '%s' not found in symbols\n", name);
        return 0;
    }

    return startAddress;
}

//
// returns the address based on filename and line number
// 
static LwU32 flcngdbGetAddressFromFileAndLine(const char *file, unsigned int line)
{
    LwU32 pc;

    flcngdbGetPcFromFileMatchingInfo(file, line, &pc, pFlcngdbUtilsCls);
    if (pc == FLCGDB_ERR_ILWALID_VALUE)
    {
        dprintf("File %s not found in symbols\n", file);
        return 0;
    }
    return pc;
}

// prints help
void flcngdbPrintHelp()
{
    dprintf("lw: flcngdb help file\n");
    dprintf("USAGE: <COMMAND> [ARGS]\n\n");
    dprintf("COMMAND  ARGUMENTS             DESCRIPTION\n");
    dprintf("~~~~~~~  ~~~~~~~~~             ~~~~~~~~~~~\n\n");
    dprintf("   ?                           Prints this help file.\n");
    dprintf("   bl <file name>:<line>       Set breakpoint at line of specified file\n");
    dprintf("   bf <function name>          Set breakpoint at function\n");
    dprintf("   bp <address>                Set breakpoint at function\n");
    dprintf("   bc                          Clear bp, will re-enable interrupts\n");
    dprintf("   p                           Print variable\n");
    dprintf("   s                           Step into\n");
    dprintf("   n                           Step over (next)\n");
    dprintf("   c                           Continue\n");
    dprintf("   w millis                    Wait for a bp to hit, will disable interrupts "
            "on hit. Will timeout after millis\n");
    dprintf("   cleansession sessionId      deletes session\n");
    dprintf("   q                           Quit the debugging sessoin\n");
    dprintf("\n");
    dprintf("Example 1 - Setting a breakpoint and resuming:\n");
    dprintf("Break into WinDBG or GDB , go into flcngdb, set up breakpoint, use q to exit, "
            "F5 or C to continue the target\n");
    dprintf("Once you have reached the area where the breakpoint should have hit, break into windbg,"
            "go into flcngdb again, use w to latch onto the breakpoint, "
            "the use any of the debug commands to debug\n");
    dprintf("\n");
    dprintf("Example 2 - Clearing a breakpoint:\n");
    dprintf("While in flcngdb, use bc to clear breakpoint, "
            "c to continue, q to exit flcngdb "
            "and F5 or C to continue target\n");
}

void flcngdbCleanAll()
{
    // 1. Clear the break points.
    // 2 .Delete flcnGDBUtils and class and delete all session data.
    // 3. close the source windows if any.
    // TBD : have this part of WinDbg exit routine. GDB exit.
    // 
    dprintf("lw: flcngdb Clean up \n");

}
// Flcngdb main entry point (main loop)
void flcngdbMenu(char* sessionID, char* pSymbPath)
{
    BOOL bDone = FALSE;
    char input1024[FLCNGDB_MAX_INPUT_LEN];

    int cmd;
    int lastCmd = FALCGDB_CMD_NONE_MATCH;
    char tTmpStr[FLCNGDB_CMD_MAX_LEN];
    LwU32 tTmpInt;
    LwU32 address;

    //
    // pCLS is a pointer to the FlcngdbUtils C++ class, used to provide STL based
    // data structures and other Auxiliary services
    // 
    if (pFlcngdbUtilsCls == NULL)
    {
        dprintf("Creating FlcngdbUtils\n");
        pFlcngdbUtilsCls = flcngdbCreateUtils();
    }

    // check if session exist, create new if not
    if (flcngdbChangeSession(sessionID, pFlcngdbUtilsCls) != 0)
    {
        dprintf("Creating new session\n");
        flcngdbAddSession(sessionID, pFlcngdbUtilsCls);

        // load the symbols into session
        flcngdbLoadSymbols(pSymbPath, pFlcngdbUtilsCls);

        switch (flcngdbGetCrntSessionEngId(pFlcngdbUtilsCls))
        {
            case FLCNGDB_SESSION_ENGINE_PMU:
                dprintf("Creating new PMU session\n");
                pPmu[indexGpu].pmuFlcngdbGetRegMap(&flcngdbRegisterMap);
                break;
            case FLCNGDB_SESSION_ENGINE_DPU:
                dprintf("Creating new DPU session\n");
                pDpu[indexGpu].dpuFlcngdbGetRegMap(&flcngdbRegisterMap);
                break;
            case FLCNGDB_SESSION_ENGINE_NONE:
                // error
                break;
            default : 
                break; // error
        }

        // save the map into current session
        flcngdbSetRegisterMap(&flcngdbRegisterMap, pFlcngdbUtilsCls);
    } else
    {
        // we loaded a session, now we load out the register map from session
        flcngdbGetRegisterMap(&flcngdbRegisterMap, pFlcngdbUtilsCls);
    }

    dprintf("Session loaded\n");

#ifdef WIN32
    //
    // open a window to display source code (this has to be recreate each time we
    // re-enter the debugger (otherwise it would freeze anyway since the message queue
    // isnt being processed when this debugger is nt running
    // 
    flcngdbUiCreateFlcngdbWindow();
#endif

    dprintf("lw: Starting flcngdb Menu. (Type '?' for help)\n");

    // prepare the ICD debugger by setting EMASK to catch IBRKPTs
    flcngdbPrepareICD();

    while (!bDone)
    {
        dprintf("\n");

        if (osGetInputLine((LwU8 *)"flcngdb> ", (LwU8 *)input1024, sizeof(input1024)))
        {
            // parse the current command string
            flcngdbSetLwrrentCmdString(input1024, pFlcngdbUtilsCls);
            cmd = flcngdbGetCmd(pFlcngdbUtilsCls);

            // if the user pressed Enter then we will execute the last command
            if (cmd == FALCGDB_CMD_LAST_CMD)
            {
                cmd = lastCmd;
            }

            switch (cmd)
            {
                default:
                    // intentional fall-through
                case FALCGDB_CMD_NONE_MATCH:
                    dprintf("******* Unknown command!  Printing help...*******\n\n");
                    // intentional fall-through
                case FALCGDB_CMD_HELP:
                    flcngdbPrintHelp();
                    break;

                case FALCGDB_CMD_BF:
                    flcngdbGetCmdNextStr(tTmpStr, pFlcngdbUtilsCls);
                    address = flcngdbGetAddressFromFunctionName(tTmpStr);
                    if (address > 0)
                    {
                        dprintf("Breakpoint at %u (%s)\n", address, tTmpStr);
                        flcngdbSetBpByAddr(address);
                    } else
                    {
                        dprintf("Invalid breakpoint at %u\n", address);
                    }

                    break;

                case FALCGDB_CMD_BL:
                    flcngdbGetCmdNextStr(tTmpStr, pFlcngdbUtilsCls);
                    flcngdbGetCmdNextUInt(&tTmpInt, pFlcngdbUtilsCls);
                    address = flcngdbGetAddressFromFileAndLine(tTmpStr, tTmpInt);
                    if (address > 0)
                    {
                        dprintf("Breakpoint at %u (%s:%u)\n", address, tTmpStr, tTmpInt);
                        flcngdbSetBpByAddr(address);
                    } else
                    {
                        dprintf("Invalid breakpoint at %u\n", address);
                    }

                    break;

                case FALCGDB_CMD_BP:
                    flcngdbGetCmdNextHexAsUInt(&address, pFlcngdbUtilsCls);
                    if (address > 0)
                    {
                        dprintf("Breakpoint at %u\n", address);
                        flcngdbSetBpByAddr(address);
                    } else
                    {
                        dprintf("Invalid breakpoint at %u\n", address);
                    }

                    break;

                case FALCGDB_CMD_P:
                    flcngdbGetCmdNextStr(tTmpStr, pFlcngdbUtilsCls);
                    flcngdbDumpSymbol(tTmpStr);

                    break;

                case FALCGDB_CMD_QUIT:
                    bDone = TRUE;

                    break;

                case FALCGDB_CMD_CONTINUE:
                    dprintf("Continuing exelwtion\n");
                    flcngdbResumeExelwtion();

                    break;

                case FALCGDB_CMD_CLEAR_BP:
                    dprintf("Clearing breakpoint\n");
                    flcngdbClearBp();

                    break;

                case FALCGDB_CMD_WAIT:
                    dprintf("Waiting for breakpoint to hit\n");
                    flcngdbGetCmdNextUInt(&tTmpInt, pFlcngdbUtilsCls);
                    if (flcngdbWaitForBP(tTmpInt))
                        flcngdbDisplaySource(pFlcngdbUtilsCls);
                    break;

                case FALCGDB_CMD_STEP_INTO:
                    flcngdbAdvanceOneInstruction();
                    flcngdbDisplaySource(pFlcngdbUtilsCls);
                    break;

                case FALCGDB_CMD_STEP_OVER:
                    flcngdbAdvanceToNext(pFlcngdbUtilsCls);
                    flcngdbDisplaySource(pFlcngdbUtilsCls);
                    break;

                case FALCGDB_CMD_CLEAN_SESSION:
                    flcngdbGetCmdNextStr(tTmpStr, pFlcngdbUtilsCls);
                    flcngdbCleanSession(tTmpStr);
                    break;
            }

            // remember the last exelwted command
            lastCmd = cmd;
            memset(input1024, 0, sizeof(input1024));
        }
    }

// close the UI window if we are in Windows
#ifdef WIN32
    dprintf("Closing UI\n");
    flcngdbUiCloseFlcngdbWindow();
#endif

    dprintf("********************leaving debugging session*******************\n");
}


