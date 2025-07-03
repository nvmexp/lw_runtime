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
#include <stdarg.h>
#include <string.h>
#include "flcngdb.h"
#include "flcngdbUtilsWrapper.h"

// needed for source code window UI
#ifdef WIN32
    #include "flcngdbUI.h"
    #include <io.h>
#else
    #include <unistd.h>
#endif

#include "pmu.h"
#include "dpu.h"

#include "kepler/gk107/dev_pwr_pri.h"

// TODO : Need to check for falcon5, But for now for debugger falcon 4 reg address space will work 
// for falcon5.
#include "maxwell/gm107/dev_falcon_v4.h" 

/* ------------------------ Structures ------------------------------------- */

// saves the CPP Flcngdb class pointer between sessions
CFlcngdbUtils* pFlcngdbUtilsCls = NULL;

// holds the register mapping for the current falcon
FLCNGDB_REGISTER_MAP flcngdbRegisterMap;

// saves the interrupt status between sessions so they can be restored
LwU32 flcngdbSavedInterrupts;
LwU32 flcngdbMaxBreakpoints;
static LwBool flcngdbWaitForBP(LwS32 timeoutMillis);
static LwBool flcngdbWriteICDCmd(LwU32 regVal);
static int flcngdbResumeExelwtion();
static void flcngdbDisplaySource();
static void flcngdbWaitForICDDataReady();
static void flcngdbAdvanceOneInstruction();
static void flcngdbAdvanceToNext();
static void flcngdbClearBp(LwU32 bpIndex);
static void flcngdbDumpSymbol(const char* symbolName);
static void flcngdbSetBpByAddr(LwU32 addr);
static void flcngdbRestoreInterrupts();
static LwU32 flcngdbGetAddressFromFunctionName(const char* name);
static LwU32 flcngdbGetAddressFromFileAndLine(const char *file, unsigned int line);
static void flcngdbPrintHelp();
static void flcngdbCleanAll();
static void flcngdbSetSourcePath(const char* sourcePath);
static void flcngdbDumpGlobalSymbol(const char* symbolName);
static void flcngdbShowSingleBpInfo(LwU32 bpIndex, LwU32 regVal);
static void flcngdbShowAllBps();
static void flcngdbEnableBp(LwU32 bpIndex);
static void flcngdbDisableBp(LwU32 bpIndex);
static void flcngdbPrintStack();
static LwBool flcngdbExecFileVerify(const char* exePath);

int flcngdbPrintf( const char * format, ... )
{
    int chars_written = 0;
    char tstr[1024] = {0};

    va_list arglist;
    va_start(arglist, format);
    vsprintf(tstr, format, arglist);
    va_end(arglist);

    dprintf("%s", tstr);

    return chars_written;
}

// returns the ICD status register after querying
LwU32 flcngdbQueryICDStatus()
{
    LwU32 regVal = 0;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;
    LwBool result = FALSE;
    // RSTAT with IDX 4 gives us the current processor and ICD status
    // 0x0000040E
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _RSTAT, regVal);
    regVal = FLD_SET_DRF_NUM(_PPWR_FALCON, _ICD_CMD, _IDX, 0x4, regVal);

    result = flcngdbWriteICDCmd(regVal);

    flcngdbWaitForICDDataReady();
    return FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_RDATA);
}

//
// Print stack trace
// TO-DO: Print value of function paramters
// 
static void flcngdbPrintStack()
{

    char funcName[FLCNGDB_FUNCNAME_MAX_LEN];
    char fileName[FLCNGDB_FILENAME_MAX_LEN];
    LwU32 line, regVal, maxIdx, idx, pc;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    regVal = FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEIDX);
    maxIdx = DRF_VAL(_PPWR, _FALCON_TRACEIDX, _MAXIDX, regVal);

    for (idx = 0; idx < maxIdx; idx++)
    {
        regVal =  FLD_SET_DRF_NUM(_PPWR, _FALCON_TRACEIDX, _IDX, idx, regVal);
        FLCN_REG_WR32(LW_PFALCON_FALCON_TRACEIDX, regVal);

        // Read the PC from the TRACEPC register
        regVal = FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEPC);
        pc = DRF_VAL(_PPWR, _FALCON_TRACEPC, _PC, regVal);

        // Get the function name from the PC
        flcngdbGetFunctionFromPc(funcName, pc, pFlcngdbUtilsCls);

        // Get the filename and line number corresponding to the PC
        flcngdbGetFileMatchingInfo(pc, fileName, &line, pFlcngdbUtilsCls);

        dprintf("#%d 0x%x in %s() at %s:%d\n", idx, pc, 
                funcName, (line)? fileName:"Source not found", line);
    }

}

// Displays information for a single breakpoint
static void flcngdbShowSingleBpInfo(LwU32 bpIndex, LwU32 regVal)
{
    char funcName[FLCNGDB_FUNCNAME_MAX_LEN];
    char fileName[FLCNGDB_FILENAME_MAX_LEN];
    LwBool bEnabled;
    LwU32 line;
    LwU32 pc;

    // Get the PC value
    pc = DRF_NUM(_PPWR_FALCON, _IBRKPT1, _PC, regVal);

    // Get the enabled/disabled state of the breakpoint
    bEnabled = FLD_TEST_DRF(_PPWR_FALCON, _IBRKPT1, _EN, _ENABLE, regVal);

    // Get the function name from the pc value
    flcngdbGetFunctionFromPc(funcName, pc, pFlcngdbUtilsCls);

    // Get the filename and line number corresponding to the pc
    flcngdbGetFileMatchingInfo(pc, fileName, &line, pFlcngdbUtilsCls);
    if (line == 0)
    {
        dprintf("Source cannot be found (PC %x does not match table)\n", pc);
        return;
    }

    dprintf("%d   breakpoint   %c      0x%x    in %s at %s:%d\n", bpIndex, 
             (bEnabled)?'y':'n', pc, funcName, fileName, line);  
}

// Displays information for all breakpoints
void flcngdbShowAllBps()
{
    LwU32 regVal = 0;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;
    LwU32 i;

    dprintf("Num    Type      Enb    Address    What\n");
    dprintf("~~~    ~~~~      ~~~    ~~~~~~~    ~~~~\n");

    for (i = 0; i < flcngdbMaxBreakpoints; i++)
    {
        regVal = FLCN_REG_RD32(IBRKPT_REG_GET(i));
        // Display only if a breakpoint has been set.
        if (!(FLD_TEST_DRF(_PPWR_FALCON, _IBRKPT1, _PC, _INIT, regVal)))
        {
            flcngdbShowSingleBpInfo(i + 1, regVal);
        }
    }
}

LwBool flcngdbIsInDebugMode()
{
    LwU32 icdStatus = 0; 
    icdStatus = flcngdbQueryICDStatus();
    
    if (icdStatus != 0)
    {
        return FLD_TEST_DRF(_PPWR_FALCON, _ICD_RDATA, _RSTAT4_ICD_STATE,
            _FULL_DBG_MODE, icdStatus);
    }

    return LW_FALSE;
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
    LwU32 i;

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
            for (i = 0; i < flcngdbMaxBreakpoints; i++)
            {
                bp = flcngdbGetBpAddress(i);
                if (pc == bp)
                {
                    //
                    // if we hit the breakpoint, we want to disable the interrupt
                    // so that when you step there isnt a huge number of timer
                    // interrupts that takes you into code sections you don't want
                    //
                    // TODO: This can be changed later when stack trace is supported
                    //       in which case we dont need to disable the interrupt
                    //       anymore since its all source level stepping by then
                    // 
                    dprintf("Breakpoint %d hit at %u\n", i + 1, pc);
                    flcngdbDisableInterrupts();
                    break;
                } 
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

// Enables a breakpoint 
static void flcngdbEnableBp(LwU32 bpIndex)
{
    LwU32 regVal;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    if (bpIndex < 1 || bpIndex > flcngdbMaxBreakpoints)
    {
        dprintf("Invalid breakpoint\n");
        dprintf("Usage: be <bpIndex>\n");
        return;
    }

    regVal = FLCN_REG_RD32(IBRKPT_REG_GET(bpIndex - 1));
    FLD_SET_DRF_DEF(_PPWR_FALCON, _IBRKPT1, _EN, _ENABLE, regVal);

    FLCN_REG_WR32(IBRKPT_REG_GET(bpIndex - 1), regVal);
    dprintf("Enabling IBPT%d\n", bpIndex);
}

// Disables a breakpoint
static void flcngdbDisableBp(LwU32 bpIndex)
{
    LwU32 regVal;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    if (bpIndex < 1 || bpIndex > flcngdbMaxBreakpoints)
    {
        dprintf("Invalid breakpoint\n");
        dprintf("Usage: bd <bpIndex>\n");
        return;
    }

    regVal = FLCN_REG_RD32(IBRKPT_REG_GET(bpIndex - 1));
    FLD_SET_DRF_DEF(_PPWR_FALCON, _IBRKPT1, _EN, _DISABLE, regVal);

    FLCN_REG_WR32(IBRKPT_REG_GET(bpIndex - 1), regVal);
    dprintf("Disabling IBPT%d\n", bpIndex);
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

    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _STEP, regVal);

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
static void flcngdbAdvanceToNext()
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
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _RUN, regVal);
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

void flcngdbSleep(LwU32 MicroSeconds)
{
    osPerfDelay(MicroSeconds);
}

// clear breakpoint
static void flcngdbClearBp(LwU32 bpIndex)
{
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    if (bpIndex < 1 || bpIndex > flcngdbMaxBreakpoints)
    {
        dprintf("Invalid breakpoint\n");
        dprintf("Usage: bc <bpIndex>\n");
        return;
    }

    FLCN_REG_WR32(IBRKPT_REG_GET(bpIndex - 1), 0x0);
}

// insert a breakpoint at address addr, the breakpoint is set to enabled
// and exceptiosn suppress
static void flcngdbSetBpByAddr(LwU32 addr)
{
    LwU32 regVal = 0;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;
    LwU32 i;
    LwU32 readVal;

    // 
    // clear the EMASK since don't drop into ICD , let the exception 
    // handler execute and break point get drop into.
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _EMASK, regVal);
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _EMASK_EXC_IBREAK, _FALSE, regVal);
    flcngdbWriteICDCmd(regVal);

    //
    // bp = addr & 0x00FFFFFF | 0xA0000000;
    // set the PC
    regVal = FLD_SET_DRF_NUM(_PPWR_FALCON, _IBRKPT1, _PC, addr, regVal);

    //
    // set Enable and Suppress flags (we want to drop into ICD without
    // triggering exception interrupt)
    // 
    FLD_SET_DRF_DEF(_PPWR_FALCON, _IBRKPT1, _SUPPRESS, _DISABLE, regVal);
    FLD_SET_DRF_DEF(_PPWR_FALCON, _IBRKPT1, _EN, _ENABLE, regVal);

    for (i = 0; i < flcngdbMaxBreakpoints; i++)
    {
        //
        // Read the register to check if it is already in use
        // The PC field should be all zeros if not in use
        // 
        readVal = FLCN_REG_RD32(IBRKPT_REG_GET(i));
        if (FLD_TEST_DRF(_PPWR_FALCON, _IBRKPT1, _PC, _INIT, readVal))
        {
            FLCN_REG_WR32(IBRKPT_REG_GET(i), regVal);
            dprintf("Set IBPT%d to %x\n", i + 1, FLCN_REG_RD32(IBRKPT_REG_GET(i)));
            return;
        }
    }
    dprintf("Exceeded maximum limit of %d breakpoints\n", flcngdbMaxBreakpoints);
}

// return the lwrrently set breakpoint address
LwU32 flcngdbGetBpAddress(LwU32 bpIndex)
{
    LwU32 regVal;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    regVal = FLCN_REG_RD32(IBRKPT_REG_GET(bpIndex));

    return (DRF_NUM(_PPWR_FALCON, _IBRKPT1, _PC, regVal));
}

// save the current interrupt register and mask all interrupts
void flcngdbDisableInterrupts()
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
LwU32 flcngdbReadFalcPc()
{
    LwU32 regVal = 0;
    LwBool result = FALSE;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    //LW_PFALCON_FALCON_ICD_CMD_OPC_RREG | LW_PFALCON_FALCON_PC_IDX | (0x2 << 6);
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _RREG, regVal);
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _IDX, _PC, regVal);

    FLCN_REG_WR32(LW_PFALCON_FALCON_ICD_CMD, regVal);

    result = flcngdbWriteICDCmd(regVal);
    // see if there was an error exelwting the last command
    // regVal & 0x4000 != 0
    // test for valid return data bit set (regVal & 0x8000 != 0)
     flcngdbWaitForICDDataReady();
   return FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_RDATA); 
}
//
// This function reads the register data from given regIndex 
//
LwU32 flcngdbReadReg(LwU32 regIndex)
{
    LwU32 regVal = 0;
    LwBool result = FALSE;
    LwU32 engineBase = flcngdbRegisterMap.registerBase;

    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _RREG, regVal);

    regVal = FLD_SET_DRF_NUM(_PPWR_FALCON, _ICD_CMD, _IDX, regIndex, regVal);

    result = flcngdbWriteICDCmd(regVal);

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

// Read one word from DMEM
LwU32 flcngdbReadWordDMEM(LwU32 address)
{
	LwU32 engineBase = flcngdbRegisterMap.registerBase;
	LwU32 reg32;
	LwU32 dmemdata;

	// Mask the block and offset for the given DMEM address
	reg32 = address & (DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_OFFS) |
                       DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_BLK));

    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(0), reg32  | 
				  DRF_DEF(_PFALCON, _FALCON_DMEMC, _AINCR, _INIT));

    dmemdata = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMD(0));

	return dmemdata;

}

static void flcngdbReadDMEMTest(LwU32 startAddress, LwU32 length)
{
    LwU32 address;
    LwU32 i;
	LwU32 dmemdata;

    address = startAddress;
    
    dprintf("Dumping DMEM: Start Address: 0x%x Length: %d locations\n",
			 startAddress & 0x00FFFFFF, length);

    for (i = 0; i < length; i++)
    {
        // PC is 24 bits long.
        dmemdata = flcngdbReadWordDMEM(0x00FFFFFF & address);
        if (i % 4 == 0)
        {
            dprintf("\n");
        }
        dprintf("%08x ", dmemdata);
        address += 4;
    }
    dprintf("\n");
}

void flcngdbReadDMEM(char *pDataBuff, LwU32 startAddress, LwU32 length)
{
    LwU32 endAddress;
    LwU32 i;
    LwU32 c=0;
    LwU32 tmpLength = 0;

    endAddress = startAddress + length;
    tmpLength = length;

    if( pDataBuff == NULL) 
    {
        dprintf("flcngdbReadDMEM : error \n");
        return;
    }

    // The data of falcon stored in Big Endian and Register reads of data 
    // are liitle endian , so the Byte order may mismatch if the byte length is more then
    // 32 bit, and memcpy will takecare of copy of data.
    // Big endian issue.
    for (i = startAddress; i < (endAddress + 1); i+=4 )
    {
        // mask to get rid of the leading 1 for DMEM
        LwU32 buffer = flcngdbReadWordDMEM(0x00FFFFFF & i);

        dprintf("%08x \n", buffer);

        if (tmpLength > 4)
        {
            memcpy((void*)pDataBuff, (void *)&buffer, tmpLength);
        }
        else
        {
            memcpy((void*)pDataBuff, (void *)&buffer, 4);
        }

        c++;
        
        if (c == 4)
        {
            dprintf("\n");
            c = 0;
        }


        tmpLength -= 4;
    }
    return;
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
    LwU32 pc;
    LwU32 c = 0;

    // this is for global symbols, if not found here .. next is local symbol. 
    if (0)
    {
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
    }

    pc = flcngdbReadFalcPc();
    flcngdbPrintLocalData(symbolName, pc, pFlcngdbUtilsCls);
    dprintf("\n");
}

static void flcngdbCleanSession(const char *sessionID)
{
    if (flcngdbDeleteSession(sessionID, pFlcngdbUtilsCls) != 0)
    {
        dprintf("flcngdb :Error given session name doesn't exist \n");
    } 
    else
    {
        dprintf("flcngdb: Session deleted \n");
    }
}

//
// displays the source code corresponding to the current PC
// OS dependent: Windows has UI, other OSes will print to console
// 
static void flcngdbDisplaySource(void)
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
    dprintf("flcndbg: flcngdb help file\n");
    dprintf("USAGE: <COMMAND> [ARGS]\n\n");
    dprintf("COMMAND  ARGUMENTS               DESCRIPTION\n");
    dprintf("~~~~~~~  ~~~~~~~~~               ~~~~~~~~~~~\n\n");
    dprintf("   ?                             Prints this help file.\n");
    dprintf("   bl <file name>:<line>         Set breakpoint at line of specified file\n");
    dprintf("   bf <function name>            Set breakpoint at function\n");
    dprintf("   bp <hex-address>              Set breakpoint at address ex: bp 120d2\n");
    dprintf("   bc <bpIndex>                  Clear bp, will re-enable interrupts. ex: bc 1\n");
    dprintf("   show                          Show breakpoints\n");
    dprintf("   be <bpIndex>                  Enables the breakpoint. ex: be 1\n");
    dprintf("   bd <bpIndex>                  Disables the breakpoint. ex: bd 1\n");
	dprintf("   bt                            Prints the stack trace.\n");
    dprintf("   p                             Prints local variable\n");
    dprintf("   pg                            Prints global variable\n");
    dprintf("   s                             Step into\n");
    dprintf("   n                             Step over (next)\n");
    dprintf("   c                             Continue\n");
    dprintf("   dir <dirpath1;dirpath2>       Separator is ;\n");
    dprintf("   w millis                      Wait for a bp to hit, will disable interrupts "
            "on hit. Will timeout after millis\n");
    dprintf("   cleansession sessionId        Deletes session\n");
    dprintf("   rr regindex                   Reads ICD reg\n");
    dprintf("   rdmem <hex-address> <length>  Reads DMEM for the local variable data\n");
    dprintf("   q                             Quit the debugging sessoin\n");
    dprintf("\n");
    dprintf("Example 1 - Setting a breakpoint and resuming:\n");
    dprintf("Break into WinDBG or GDB , go into flcngdb, set up breakpoint, use q to exit, "
            "F5 or C to continue the target\n");
    dprintf("Once you have reached the area where the breakpoint hits automatically, now"
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
    dprintf("flcndbg: flcngdb Clean up \n");

}

//
// Verify that the user mentioned .out file for symbol loading
// and the file actually exists at that location.
// 
static LwBool flcngdbExecFileVerify(const char *exePath)
{
    char *pDot = NULL;
    LwBool bFound = LW_FALSE;
	LwU32 result;

	if (exePath == NULL)
    {
		dprintf("Path to .out file missing\n");
		return LW_FALSE;
    }

	// Check if the file has .out suffix. Fail otherwise.
	pDot = strrchr(exePath, '.');
    if (strcmp(pDot + 1, "out"))
    {
        dprintf("Invalid file! Please verify that full path to .out file is provided.\n");
		return bFound;
    }

	// Now check if file exists at the location provided.
#ifdef WIN32
    result = _access(exePath, 0);
#else
    result = access(exePath, F_OK);
#endif
    if (result == 0)
    {
        bFound = LW_TRUE;
    }
    else
    {
        dprintf("File does not exist at the location provided.\n");
    }
    return bFound;
}

// Flcngdb main entry point (main loop)
void flcngdbMenu(char* sessionID, char* exePath)
{
    FLCNGDB_FP_TABLE funcPtable;

    if (!flcngdbExecFileVerify(exePath))
    {
        return;
    }

    if (pFlcngdbUtilsCls == NULL)
    {
        dprintf("Creating FlcngdbUtils\n");
        pFlcngdbUtilsCls = flcngdbCreateUtils();
    }

    // assign and Init the function pointers.
    funcPtable.flcngdbRegRd32 = &flcngdbReadReg;
    funcPtable.flcngdbReadDMEM = &flcngdbReadDMEM;
    funcPtable.flcngdbReadWordDMEM = &flcngdbReadWordDMEM;
    funcPtable.dbgPrintf = &flcngdbPrintf;

    flcngdbInitFunctionPointers(&funcPtable, pFlcngdbUtilsCls);

    // check if session exist, create new if not
    if (flcngdbChangeSession(sessionID, pFlcngdbUtilsCls) != 0)
    {
        dprintf("Creating new session\n");
        flcngdbAddSession(sessionID, exePath, pFlcngdbUtilsCls);

        // load the symbols into session
        flcngdbLoadSymbols(pFlcngdbUtilsCls);

        switch (flcngdbGetCrntSessionEngId(pFlcngdbUtilsCls))
        {
            case FLCNGDB_SESSION_ENGINE_PMU:
                dprintf("Creating new PMU session\n");
                pPmu[indexGpu].pmuFlcngdbGetRegMap(&flcngdbRegisterMap);
                flcngdbMaxBreakpoints = pPmu[indexGpu].pmuFlcngdbMaxBreakpointsGet();
                break;
            case FLCNGDB_SESSION_ENGINE_DPU:
                dprintf("Creating new DPU session\n");
                pDpu[indexGpu].dpuFlcngdbGetRegMap(&flcngdbRegisterMap);
                flcngdbMaxBreakpoints = pDpu[indexGpu].dpuFlcngdbMaxBreakpointsGet();
                break;
            case FLCNGDB_SESSION_ENGINE_NONE:
                // error
                break;
            default : 
                break; // error
        }

        // save the map into current session
        flcngdbSetRegisterMap(&flcngdbRegisterMap, pFlcngdbUtilsCls);
    } 
    else
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

    // accept the inputs and do the processing.
    flcngdbAcceptInputs();
}

void flcngdbShowSourceInWindow()
{
#ifdef WIN32
    flcngdbUiCreateFlcngdbWindow();
#endif
    flcngdbDisplaySource();
}

void flcngdbSetSourcePath(const char* sourcePath)
{
     flcngdbSetSourceDirPath(sourcePath, pFlcngdbUtilsCls);
}

void flcngdbDumpGlobalSymbol(const char* symbolName)
{
    flcngdbPrintGlobalVarData(symbolName, pFlcngdbUtilsCls);
}

void flcngdbAcceptInputs()
{

    BOOL bDone = FALSE;
    char input1024[FLCNGDB_MAX_INPUT_LEN];

    int cmd;
    int lastCmd = FALCGDB_CMD_NONE_MATCH;
    char tTmpStr[FLCNGDB_CMD_MAX_LEN];
    LwU32 tTmpInt;
    LwU32 tTmpInt2;
    LwU32 address;
    LwU32 regvalue;


    dprintf("flcndbg: Starting flcngdb Menu. (Type '?' for help)\n");

    while (!bDone)
    {
        dprintf("\n");

        if (osGetInputLine((U008 *)"flcngdb> ", (U008 *)input1024, sizeof(input1024)))
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
                    flcngdbGetCmdNextUInt(&tTmpInt, pFlcngdbUtilsCls);
                    dprintf("Clearing breakpoint\n");
                    flcngdbClearBp(tTmpInt);
                    break;

                case FALCGDB_CMD_WAIT:
                    dprintf("Waiting for breakpoint to hit\n");
                    flcngdbGetCmdNextUInt(&tTmpInt, pFlcngdbUtilsCls);
                    if (flcngdbWaitForBP(tTmpInt))
                    {
                        flcngdbDisplaySource();
                    }
                    break;

                case FALCGDB_CMD_STEP_INTO:
                    flcngdbAdvanceOneInstruction();
                    flcngdbDisplaySource();
                    break;

                case FALCGDB_CMD_STEP_OVER:
                    flcngdbAdvanceToNext();
                    flcngdbDisplaySource();
                    break;

                case FALCGDB_CMD_CLEAN_SESSION:
                    flcngdbGetCmdNextStr(tTmpStr, pFlcngdbUtilsCls);
                    flcngdbCleanSession(tTmpStr);
                    break;

                case FALCGDB_CMD_READ_REG:
                    flcngdbGetCmdNextUInt(&tTmpInt, pFlcngdbUtilsCls);;
                    regvalue = flcngdbReadReg(tTmpInt);
                    dprintf("regread : %d \n", regvalue);
                    break;

                case FALCGDB_CMD_READ_DMEM:
                    flcngdbGetCmdNextHexAsUInt(&tTmpInt, pFlcngdbUtilsCls);
                    flcngdbGetCmdNextUInt(&tTmpInt2, pFlcngdbUtilsCls);
                    flcngdbReadDMEMTest(tTmpInt, tTmpInt2);
                    break;

                case FALCGDB_CMD_SET_DIR_PATH:
                    flcngdbGetCmdNextStr(tTmpStr, pFlcngdbUtilsCls);
                    flcngdbSetSourcePath(tTmpStr);
                    break;

                case FALCGDB_CMD_PRINT_GLOBAL_SYMBOL:
                    flcngdbGetCmdNextStr(tTmpStr, pFlcngdbUtilsCls);
                    flcngdbDumpGlobalSymbol(tTmpStr);
                    break;

                case FALCGDB_CMD_SHOW_BP_INFO:
                    flcngdbShowAllBps();
                    break;
                
                case FALCGDB_CMD_ENABLE_BP:
                    flcngdbGetCmdNextUInt(&tTmpInt, pFlcngdbUtilsCls);
                    dprintf("Enabling breakpoint\n");
                    flcngdbEnableBp(tTmpInt);
                    break;

                case FALCGDB_CMD_DISABLE_BP:
                    flcngdbGetCmdNextUInt(&tTmpInt, pFlcngdbUtilsCls);
                    dprintf("Disabling breakpoint\n");
                    flcngdbDisableBp(tTmpInt);
                    break;

			    case FALCGDB_CMD_PRINT_STACK:
				    flcngdbPrintStack();
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


