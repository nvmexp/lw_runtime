/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// exts.c
//
//*****************************************************

//
// includes
//
#include "lwwatch.h"
#include "vgpu.h"

#include <stdio.h>
#include <string.h>

#include "os.h"
#include "hal.h"

#if defined(USERMODE) && defined(LW_WINDOWS) && !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#include <usermode.h>
#include <wdbgexts.h>
#include <dbgeng.h>
#endif

#ifndef CLIENT_SIDE_RESMAN
#if defined(USERMODE)

#elif defined(LW_WINDOWS)
#include "lwwatch.h"

#elif LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
#include "lwstring.h"
#endif
#endif

#include "exts.h"

// lw includes
//
#include "print.h"
#include "parse.h"
#include "help.h"
#include "chip.h"
#include "bif.h"
#include "fb.h"
#include "fifo.h"
#include "clk.h"
#include "disp.h"
#include "diags.h"
#include "inst.h"
#include "dcb.h"
#include "elpg.h"
#include "gr.h"
#include "heap.h"
#include "i2c.h"
#include "intr.h"
#include  "intr_private.h"
#include "msa.h"
#include "exts.h"
#include "mmu.h"
#include "methodParse.h"
#include "lwwatch_pex.h"
#include "pmu.h"
#include "sig.h"
#include "falcon.h"
#include "flcnrtos.h"
#include "gpuanalyze.h"
#include "vic.h"
#include "msdec.h"
#include "falctrace.h"
#include "msenc.h"
#include "ofa.h"
#include "hda.h"
#include "sec.h"
#include "virtOp.h"
#include "ce.h"
#include "dpu.h"
#include "fecs.h"
#include "vmem.h"
#include "virt.h"
#include "hwprod.h"
#include "sig.h"
#include "cipher.h"
#include "bus.h"
#include "mc.h"
#include "dpaux.h"
#include "smbpbi.h"
#include "seq.h"
#include "tegrasys.h"
#include "lwdec.h"
#include "lwjpg.h"
#include "acr.h"
#include "psdl.h"
#include "falcphys.h"
#include "sec2.h"
#include "deviceinfo.h"
#include "lwlink.h"
#include "ibmnpu.h"
#include "dpmsg.h"
#include "lwsr_parsereg.h"
#include "vpr.h"
#include "gsp.h"
#include "riscv.h"
#include "fbflcn.h"
#include "dpparsedpcd.h"
#include "l2ila.h"
#include "dfdasm.h"

#if defined(LW_WINDOWS)  &&  !defined(LW_MODS)
#include "lwoca.h"
#endif // LW_WINDOWS

#if defined(LWDEBUG_SUPPORTED)
#include "lwdump_priv.h"
#include "prbdec.h"
#include "g_lwdebug_pb.h"
#include "g_regs_pb.h"
#endif

#include "g_intr_hal.h"

#include "br04.h"
#include "priv.h"

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) || defined(CLIENT_SIDE_RESMAN)
typedef char* PCSTR;
#endif

extern LwU8 lwsr_message[12];  //  message = ((TFC|TFRD) | (R) | (PVER | PVID | PDID))
extern LwU8 lwsr_digest[20];   //  digest = the result of HMAC-SHA1 computation
extern LwU8 lwsr_key[16];      //  key = private vendor key

//
// Checks for NULL ptr's and empty before calling GetExpression
// prevents Windbg access violations where we call GetExpression without
// checking these conditions
//
LwU64 GetSafeExpression(PCSTR lpExpression)
{
    LwU64 Value = 0;

    if (!lpExpression || (*lpExpression == '\0'))
        return Value;

    Value = GetExpression(lpExpression);

    return Value;
}

BOOL GetSafeExpressionEx
(
    PCSTR Expression,
    LwU64* Value,
    PCSTR* Remainder
)
{
    BOOL bResult = FALSE;

    if (!Expression || (*Expression == '\0'))
        return bResult;

    bResult = GetExpressionEx(Expression, Value, Remainder);

    return bResult;
}
//
//
//
//
LwU64 GetSafeExpressionUnextended(PCSTR lpExpression)
{
    LwU64 Value = 0;

    Value = GetSafeExpression(lpExpression);

    // Make sure the value isn't sign-extended
    if (((Value >> 32) == 0xffffffff) && ((Value & 0x80000000) == 0x80000000))
        Value &= 0xffffffff;

    return Value;
}

BOOL GetSafeExpressionExUnextended
(
    PCSTR Expression,
    LwU64* Value,
    PCSTR* Remainder
)
{
    BOOL bResult = FALSE;

    bResult = GetExpressionEx(Expression, Value, Remainder);
    if (bResult)
    {
        // Make sure the value isn't sign-extended
        if (((*Value >> 32) == 0xffffffff) && ((*Value & 0x80000000) == 0x80000000))
            *Value &= 0xffffffff;
    }
    return bResult;
}

//-----------------------------------------------------
// GetDevTunnelAddr
//
//-----------------------------------------------------
PhysAddr GetDevTunnelAddr(char *devName, PhysAddr lwReg)
{
    PDEVICE_RELOCATION pDev = NULL;
    LwU64 addr = 0;
    LwU32 devInd = 0;

    devInd = pTegrasys[indexGpu].tegrasysGetDeviceBroadcastIndex(&TegraSysObj[indexGpu], devName);
    pDev = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], devName, devInd);
    if (!pDev)
    {
        dprintf("Didn't find device %s at index %d.\n", devName, devInd);
        return 0;
    }

    addr = pDev->start + lwReg;

    return addr;
}

//-----------------------------------------------------
// init [lwBar0] [lwBar1]
// - If lwBar0 is 0 or not specified, we will search
//   for LW devices.
//-----------------------------------------------------
DECLARE_API( init )
{
    LwU64 bar0;
    usingMods = 0;
    lwMode = MODE_LIVE;

#ifdef CLIENT_SIDE_RESMAN
    if(lwBar0 != 0)
    {
        dprintf("init was already called once.\n");
        return;
    }
#endif

    // Disable multi-GPU if previously enabled for it
    memset(&multiGpuBar0,  0x0, sizeof(multiGpuBar0));
    memset(&multiGpuBar1,  0x0, sizeof(multiGpuBar1));
    memset(&multiPmcBoot0, 0x0, sizeof(multiPmcBoot0));

    //
    // Windows Dump?   If so use registers/mem from Proto Buffer
    //
#if defined(LW_WINDOWS)  &&  !defined(LW_MODS)
    if (IsWindowsDumpFile())
    {
        if (dumpModeInit(NULL, NULL) == LW_OK)
        {
            lwMode = MODE_DUMP;
            dprintf("lw: LwWatch initialized for read-only dump session from OCA Proto Buffer.\n");
        }
        else
        {
            dprintf("lw: ERROR: LwWatch FAILED to initialize for read-only dump session from OCA Proto Buffer.\n");
            lwBar0 = 0;
            lwMode = MODE_NONE;
        }
        return;
    }
#endif

    // set the lwaddr - lwBar0 is global
    lwBar0 = (PhysAddr) GetSafeExpressionUnextended(args);

    if (GetSafeExpressionExUnextended(args, &bar0, &args))
    {
        lwBar1 = (PhysAddr)GetSafeExpressionUnextended(args);
        lwBar0 = (PhysAddr)bar0;
    }

    if (lwBar0 <= 0xff)
        lwBar0 <<= 24;

    if (lwBar1 <= 0xff)
        lwBar1 <<= 24;

    initLwWatch();
    dprintf("\n");

    if (!IsTegra()) {
        initSec2ObjBaseAddr();
    }
}

#ifndef USERMODE
//-----------------------------------------------------
// modsinit
// - Initialize lwwatchMods
//-----------------------------------------------------
DECLARE_API( modsinit )
{
    LwU8 userSuppliedFlag = (LwU8) GetSafeExpression(args);

    //
    // We are using Mods
    //
    usingMods = 1;

    //
    // If user specifies nothing/0, debugOCMFlag = 2 i.e. log messages
    // with maximum level (default)
    // if user specifies 1, debugOCMFlag = 1 i.e. log level 1 messages
    // if user specifies 2, debugOCMFlag = 2 i.e. log level 2 messages
    // if user specifies 10, debugOCMFlag = 0 i.e. don't log
    //
    switch(userSuppliedFlag)
    {
        case 0:
        case 2:
            debugOCMFlag = 2;
            break;
        case 1:
            debugOCMFlag = 1;
            break;
        default:
            dprintf("lw: modsinit: flag not yet supported. Defaulting to 0.\n");
            // NOTE: Don't put a break here
        case 10:
            debugOCMFlag = 0;
            break;
    }

    initLwWatch();
}

#endif //!USERMODE

static BOOL addMultiGpu(LwU64 bar0, LwU32 i)
{
    if (bar0 <= 0xff)
    {
        bar0 <<= 24;
    }

    if (i >= MAX_GPUS)
    {
        dprintf("lw: multigpu only supports up to %u GPUs (i = %u)!\n",
                MAX_GPUS, i);
        return FALSE;
    }

    //
    // If SMC is enabled on the first GPU, any SMC related commands
    // run only on that GPU are disabled on all other GPUS.
    //
    if (i == 0)
    {
        if(pGr[indexGpu].grGetSmcState())
        {
            dprintf("lw: Commands will run only on a single GPU since SMC is enabled\n");
        }
    }
    else
    {
        if(pGr[indexGpu].grGetSmcState())
        {
            dprintf("lw: Error: SMC is not supported in multigpu environment\n");
            return FALSE;
        }
    }

    if (bar0)
    {
        LwU32    orig_verboseLevel = verboseLevel;

        // Setup for this GPU
        lwBar0 = bar0;

        //
        // LW_PMC_BOOT_0 is needed for HAL wire up, so bypass the
        // GPU_REG_RD32 mechanism and just read straight from the GPU.
        //
        pmc_boot0 = osRegRd32(0x0 /* LW_PMC_BOOT_0 */);

        // Initialize the HAL for this GPU
        verboseLevel = 0;
        osInitHal();
        verboseLevel = orig_verboseLevel;

        // Save information for this GPU
        multiGpuBar0[i]  = (PhysAddr) lwBar0;
        multiGpuBar1[i]  = (PhysAddr) lwBar1;
        multiPmcBoot0[i] = pmc_boot0;

        // Display the new GPU device added
        dprintf("lw: device %u, lwBar0: " PhysAddr_FMT, i, multiGpuBar0[i]);
        dprintf(" lwBar1: " PhysAddr_FMT " ", multiGpuBar1[i]);
        dprintf(" LW_PMC_BOOT_0: 0x%08x\n", multiPmcBoot0[i]);
    }
    else
    {
        if (i != 0)
        {
            dprintf("lw: device %d lwBar0 supplied was 0x0!\n", i);
        }
        return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------
// msi
// - Prints the MSI enable status for each GPU 
//-----------------------------------------------------
DECLARE_API( msi )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pBif[indexGpu].bifGetMsiInfo();
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// multigpu [d0Bar0] [d1Bar0] [d2Bar0] ...
//-----------------------------------------------------
DECLARE_API( multigpu )
{
    LwU32 i = 0;
    LwU64 bar0;

    // Initialize the multi-GPU data (Setup by this command)
    memset(&multiGpuBar0,  0x0, sizeof(multiGpuBar0));
    memset(&multiGpuBar1,  0x0, sizeof(multiGpuBar1));
    memset(&multiPmcBoot0, 0x0, sizeof(multiPmcBoot0));

    // This command is only lwrrently supported on big GPU
    if (!IsTegra())
    {
        for (i = 0; GetSafeExpressionEx(args, &bar0, &args); i++)
        {
            if (!addMultiGpu(bar0, i))
            {
                return;
            }
        }
        // Initialize HAL for first GPU in case commands don't support multi-GPU (No MGPU_LOOP_START/END)
        lwBar0    = multiGpuBar0[0];
        lwBar1    = multiGpuBar1[0];
        pmc_boot0 = multiPmcBoot0[0];

        osInitHal();
    }
    else    // CheetAh system
    {
        dprintf("This command is not supported on CheetAh systems!\n");
    }
}

//-----------------------------------------------------
// s_elw
// - Sets the elw to the given value
//-----------------------------------------------------
DECLARE_API( s_elw )
{
    if (args)
    {
#ifdef WIN32
        LwU8 *elwString;
        LwU8 *elwValue;

        elwString = (LwU8*) args;
        elwValue = strchr(elwString, ' ');
        if (elwValue)
        {
            *elwValue = '=';
            _putelw(elwString);
        }
#endif // WIN32
    }
}

//-----------------------------------------------------
// g_elw
// - Prings the value of the given elw
//-----------------------------------------------------
DECLARE_API( g_elw )
{
    char *elwValue;

    if (args)
    {
        elwValue = getelw(args);
        if (elwValue)
        {
            dprintf("lw: %s = %s\n", args, elwValue);
            dprintf("\n");
        }
        else
        {
            dprintf("lw: %s was not found!\n", args);
        }
    }
}

//-----------------------------------------------------------------------------------
// rb [-grIdx] <grIdx> <offset>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - reads a byte from BAR0+offset
//-----------------------------------------------------------------------------------
DECLARE_API( rb )
{
    PhysAddr lwReg = 0;
    LwU8 regVal;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    lwReg = (LwU32) GetSafeExpression(args);

    //With SMC mode enabled, a BAR0 window must be set to read the correct PGRAPH register offset
    if((pGr[indexGpu].grPgraphOffset( (LwU32) lwReg )) && (pGr[indexGpu].grGetSmcState()))
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }
        else
        {
            grIdx = *param -'0';
            // Set the window by passing grIdx
            pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        }
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    regVal = GPU_REG_RD08(lwReg);

    dprintf(PhysAddr_FMT ": 0x%02x\n", (lwBar0 + lwReg), regVal);
    dprintf("\n");

    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------------------------------------
// wb [-grIdx] <grIdx> <offset> <value>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - writes a byte to BAR0+offset
//-----------------------------------------------------------------------------------
DECLARE_API( wb )
{
    PhysAddr lwReg = 0;
    LwU8 newRegVal = 0;
    LwU8 byteOffset;
    LwU32 lwrrRegVal;
    LwU32 shift;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if (GetSafeExpressionEx(args, &lwReg, &args))
    {
        newRegVal = (LwU8) GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.wb <addr> <regVal08>\n");
        return;
    }

    //With SMC mode enabled, a BAR0 window must be set to read the correct PGRAPH register offset
    if((pGr[indexGpu].grPgraphOffset( (LwU32) lwReg )) && (pGr[indexGpu].grGetSmcState()))
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }
        else
        {
            grIdx = *param -'0';
            // Set the window by passing grIdx
            pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        }
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    // Find the byte offset for ANDing
    byteOffset = (LwU8) (((LwU32) lwReg) % 4);
    lwReg -= byteOffset;
    shift = byteOffset * 8;

    lwrrRegVal = GPU_REG_RD32((LwU32) lwReg);
    lwrrRegVal &= ~(0xFF << shift);
    lwrrRegVal |= ((newRegVal & 0xFF) << shift);

    GPU_REG_WR32( (LwU32) lwReg, lwrrRegVal);
    dprintf("\n");

    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------------------------------------
// rw [-grIdx] <grIdx> <offset>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - reads a word from BAR0+offset
//-----------------------------------------------------------------------------------
DECLARE_API( rw )
{
    PhysAddr lwReg;
    LwU8 byteOffset;
    LwU32 regVal;
    LwU32 shift;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    lwReg = (LwU32) GetSafeExpression(args);

    if (lwReg % 2 != 0)
    {
        dprintf("lw: Please enter a word aligned address\n\n");
        return;
    }

    //With SMC mode enabled, a BAR0 window must be set to read the correct PGRAPH register offset
    if((pGr[indexGpu].grPgraphOffset( (LwU32) lwReg )) && (pGr[indexGpu].grGetSmcState()))
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }
        else
        {
            grIdx = *param -'0';
            // Set the window by passing grIdx
            pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        }
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    byteOffset = (LwU8) (((LwU32) lwReg) % 4);
    lwReg -= byteOffset;
    shift = byteOffset * 8;

    regVal = GPU_REG_RD32((LwU32) lwReg);
    regVal = (regVal >> shift) & 0xFFFF;

    dprintf(PhysAddr_FMT ": 0x%04x\n\n", (lwBar0 + lwReg + byteOffset), regVal);

    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//------------------------------------------------------------------------------------
// ww [-grIdx] <grIdx> <offset> <value>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - writes a word to BAR0+offset
//------------------------------------------------------------------------------------
DECLARE_API( ww )
{
    PhysAddr lwReg = 0;
    LwU32 newRegVal = 0;
    LwU8 byteOffset;
    LwU32 lwrrRegVal;
    LwU32 shift;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if (GetSafeExpressionEx(args, &lwReg, &args))
    {
        if (lwReg % 2 != 0)
        {
            dprintf("lw: Please enter a word aligned address\n\n");
            return;
        }
        newRegVal = (LwU32) GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.ww <addr> <regVal16>\n\n");
        return;
    }

    //With SMC mode enabled, a BAR0 window must be set to read the correct PGRAPH register offset
    if((pGr[indexGpu].grPgraphOffset( (LwU32) lwReg )) && (pGr[indexGpu].grGetSmcState()))
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }
        else
        {
            grIdx = *param -'0';
            // Set the window by passing grIdx
            pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        }
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    // Find the byte offset for ANDing
    byteOffset = (LwU8) (((LwU32) lwReg) % 4);
    lwReg -= byteOffset;
    shift = byteOffset * 8;

    lwrrRegVal = GPU_REG_RD32((LwU32) lwReg);
    lwrrRegVal &= ~(0xFFFF << shift);
    lwrrRegVal |= ((0xFFFF & newRegVal) << shift);

    GPU_REG_WR32( (LwU32) lwReg, lwrrRegVal);
    dprintf("\n");

    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//------------------------------------------------------------------------------------
// rd [-p] [-a] [-d] <dev> [-i] <inst> [-l] [-grIdx] <grIdx> <addr> [length]
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - reads a lw register
//------------------------------------------------------------------------------------
DECLARE_API( rd )
{
    PhysAddr lwReg = 0;
    LwU32 sizeInBytes = 0, i;
    char *param;
    char devName[16];
    LwU32 devInst = 0;
    BOOL isParse, isListAll;
    BOOL isDevSpec = FALSE;
    BOOL isInstSpec = FALSE;
    BOOL isListDevs = FALSE;
    BOOL isGrIndex = FALSE;
    BOOL isSafeExpression = FALSE;
    LwU32 grIdx = 0;

    CHECK_INIT(MODE_LIVE | MODE_DUMP);

    isParse = parseCmd(args, "p", 0, NULL);
    isListAll = parseCmd(args, "a", 0, NULL);
    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    isSafeExpression = GetSafeExpressionEx(args, &lwReg, &args);

    if (isVirtual())
        setLwwatchMode(LW_TRUE);

    //With SMC mode enabled, a BAR0 window must be set to read the correct PGRAPH register offset
    if((pGr[indexGpu].grPgraphOffset( (LwU32) lwReg )) && (pGr[indexGpu].grGetSmcState()))
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }
        else
        {
            grIdx = *param -'0';
            // Set the window by passing grIdx
            pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        }
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    if (isSafeExpression)
    {
        sizeInBytes = (LwU32) GetSafeExpression(args);
        sizeInBytes = (sizeInBytes + 3) & ~3ULL;
    }

    isInstSpec = parseCmd(args, "i", 1, &param);
    if (isInstSpec)
    {
        devInst = LwU64_LO32(GetSafeExpression(param));
    }

    isListDevs = parseCmd(args, "l", 0, NULL);
    if (isListDevs)
    {
        tegrasysListAllDevs(&TegraSysObj[indexGpu]);
    }

    if (IsTegra())
    {
        isDevSpec = parseCmd(args, "d", 0, NULL);
    }

    if (isDevSpec)
    {
        isListDevs = parseCmd(args, "l", 0, NULL);
        if (isListDevs)
        {
            tegrasysListAllDevs(&TegraSysObj[indexGpu]);
        }
        else
        {
            // Get the device name from the args
            // Find the Base Address of the device from the TEGRA_DEV_TABLE
            // Get the offset to read from the rest of the args
            // Now do a Physical read of the computed address
            //
            args = getToken(args, devName, NULL);

            // Check if device is powered on and not in reset before trying to touch it
            if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], devName, devInst)==TRUE)
            {
                lwReg = GetDevTunnelAddr(devName, lwReg);
                if (lwReg != 0)
                {
                    printDataByType((LwU32) lwReg, sizeInBytes, SYSTEM_PHYS, 4);
                }
            }
            else
            {
                dprintf("lw: Register 0x%08x read failed. %s(%d) powered off/in reset\n",
                    (LwU32) lwReg, devName, devInst);
            }
        }
    }
    else
    {
        if ((lwReg & 3) != 0)
        {
            dprintf("lw: Please enter a dword aligned address\n\n");
            goto bad_usage;
            return;
        }

        if (sizeInBytes == 0)
        {
            sizeInBytes = 4;
        }

        MGPU_LOOP_START;
        {
            if (isParse)
            {
                for (i = 0; i < sizeInBytes; i += 4)
                {
                    parseManualReg( (LwU32) (lwReg + i),
                                    isVirtualWithSriov() ? GPU_REG_RD32_DIRECT((LwU32) (lwReg + i)) :
                                                    GPU_REG_RD32( (LwU32) (lwReg + i) ),
                                    isListAll);
                    if (osCheckControlC())
                    {
                        break;
                    }
                }
            }
            else
            {
                printDataByType((LwU32) lwReg, sizeInBytes, REGISTER, 4);
            }
        }
        MGPU_LOOP_END;
    }

    if (isVirtual())
        setLwwatchMode(LW_FALSE);

bad_usage:
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}


//------------------------------------------------------------------------------------
// wr [-d] <dev> [-i] <inst> [-grIdx] <grIdx> <addr> <val>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - writes a lw register
//------------------------------------------------------------------------------------
DECLARE_API( wr )
{
    LwU64 lwReg = 0;
    LwU32 regVal = 0;
    char devName[16];
    LwU32 devInst = 0;
    PDEVICE_RELOCATION pDev = NULL;
    BOOL isDevSpec = FALSE;
    BOOL isInstSpec = FALSE;
    char *param;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;

    CHECK_INIT(MODE_LIVE);

    isInstSpec = parseCmd(args, "i", 1, &param);
    if (isInstSpec)
    {
        devInst = LwU64_LO32(GetSafeExpression(param));
    }

    if (IsTegra())
    {
        isDevSpec = parseCmd(args, "d", 0, NULL);
    }

    if (isDevSpec)
    {
        //
        // Get the device name from the args
        // Find the Base Address of the device from the TEGRA_DEV_TABLE
        // Get the offset to write and the value to write from the rest of the args
        // Now do a Physical write of the computed address
        //
        args = getToken(args, devName, NULL);
    }

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if (GetSafeExpressionEx(args, &lwReg, &args))
    {
        if (isDevSpec)
        {
            if (!isInstSpec)
            {
                devInst = pTegrasys[indexGpu].tegrasysGetDeviceBroadcastIndex(&TegraSysObj[indexGpu], devName);
            }
            pDev = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], devName, devInst);
            if (!pDev)
            {
                dprintf("lw: %s: Didn't find device %s(%d).\n", __FUNCTION__, devName, devInst);
//                return 0;
            }
            else
            {
                lwReg = GetDevTunnelAddr(devName, lwReg);
            }

            // Check if device is powered on and not in reset before trying to touch it
            if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], devName, devInst)==FALSE)
            {
                dprintf("lw: Register 0x%08x write failed. %s(%d) powered off/in reset\n",
                    (LwU32) lwReg, devName, devInst);
                return;
            }
        }
        regVal = (LwU32) GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.wr <lwReg32> <regVal32>\n");
        return;
    }

    if (lwReg == 0)
    {
        return;
    }

    if ((lwReg & 3) != 0)
    {
        dprintf("lw: Please enter a dword aligned address\n\n");
        return;
    }
    //With SMC mode enabled, a BAR0 window must be set to read the correct PGRAPH register offset
    if((pGr[indexGpu].grPgraphOffset( (LwU32) lwReg )) && (pGr[indexGpu].grGetSmcState()))
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }
        else
        {
            grIdx = *param -'0';
            // Set the window by passing grIdx
            pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        }
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    isVirtualWithSriov() ? GPU_REG_WR32_DIRECT((LwU32) lwReg, regVal) :
                           GPU_REG_WR32((LwU32) lwReg, regVal);

    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------
// tmdsrd <link> <reg>
// - reads indexed tmds register
//-----------------------------------------------------
DECLARE_API( tmdsrd )
{
    LwU64 value;
    LwU32 link = 0;
    LwU32 reg = 0;
    LwU32 val;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &value, &args))
    {
        reg = (LwU32) GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.tmdsrd <link> <reg>\n");
        return;
    }

    link = (LwU32)value;

    //
    // Printing three args using dprintf doesn't work.
    // So don't try anything like the one below.
    // dprintf("Link 0x%x, reg 0x%x: 0x%x\n", reg, link, val);
    //
    MGPU_LOOP_START;
    {
        val = TMDS_RD(link, reg);
        dprintf("Link 0x%x ", link);
        dprintf("reg 0x%x, val: 0x%x\n", reg, val);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// tmdswr <link> <reg> <value>
// - writes indexed tmds reg
//-----------------------------------------------------
DECLARE_API( tmdswr )
{
    LwU64 link = 0;
    LwU64 reg = 0;
    LwU32 val;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &link, &args) &&
        GetSafeExpressionEx(args, &reg, &args))
    {
        val = (LwU32) GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.tmdswr <link> <reg> <value>\n");
        return;
    }

    MGPU_LOOP_START;
    {
        TMDS_WR((LwU32)link, (LwU32)reg, val);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

#if defined(LWDEBUG_SUPPORTED)
//-----------------------------------------------------
// dump [-gpu <index>] [-c <component>] [-f <filename>]
// - dump GPU debug info to a zip file
//-----------------------------------------------------
DECLARE_API( dump )
{
#if defined(WIN32) && !defined(USERMODE)

    char *pParams;
    char *pComponentName = NULL;
    char *pFilename = "dump.zip";
    LwU64 gpuIndex = 0;
    LW_STATUS status;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "check", 0, NULL))
    {
        status = lwDumpCheck();
        if (status != LW_OK)
        {
            dprintf("lw: Error: lwDumpCheck failed, status=%d\n", status);
        }
    }
    else
    {
        if (parseCmd(args, "gpu", 1, &pParams))
        {
            GetSafeExpressionEx(pParams, &gpuIndex, &pParams);
        }
        parseCmd(args, "c", 1, &pComponentName);
        parseCmd(args, "f", 1, &pFilename);

        status = lwDumpSetup((LwU32)gpuIndex, pComponentName, pFilename);
        if (status != LW_OK)
        {
            dprintf("lw: Error: lwDumpSetup failed, status=%d\n", status);
        }
    }

#else // WIN32 && !USERMODE
    dprintf("lw: Error: LwDump not implemented on this platform.\n");
#endif
}

void printPrbMessage(char *msgName, void* data, LwU32 length)
{
    const PRB_MSG_DESC *msgDescs[] =
    {
        REGS_REGSANDMEM,
        LWDEBUG_LWDUMP,
        NULL
    };
    const PRB_MSG_DESC* msgDesc = NULL;
    PRB_MSG msg;

    msgDesc = prbGetMsgDescByName(msgDescs, msgName);
    if (msgDesc == NULL)
    {
        dprintf("lw: Unrecognized message name '%s'\n", msgName);
    }
    else
    {
        // Set LWD's printf function to "dprintf" so that below commands will
        // be outputted correctly.
        lwdCmn_SetPrintf(dprintf);

        prbCreateMsg(&msg, msgDesc);
        prbDecodeMsg(&msg, data, length);
        prbPrintMsg(&msg, 0);
        prbDestroyMsg(&msg);
    }
}

//-----------------------------------------------------
// prbdec <message_name>
//        (<virtual_address> <length> | -f <filename>)
// - decodes a memory location or dump file as a binary
//   protobuf message
//-----------------------------------------------------
DECLARE_API( prbdec )
{
    char *params;
    char msgName[128];
    char *filename = NULL;
    PhysAddr address = 0;
    long fpos;
    LwU64 length = 0;
    void *data = NULL;
    FILE *pFile = NULL;
    LwBool validParams = FALSE;

    CHECK_INIT(MODE_LIVE);

    params = getToken(args, msgName, NULL);
    if (params != NULL)
    {
        {
            if (parseCmd(params, "f", 1, &filename))
            {
                pFile = fopen(filename, "rb");
                if (pFile == NULL)
                {
                    dprintf("lw: Could not open file '%s'\n", filename);
                }
                else
                {
                    fseek(pFile, 0, SEEK_END);
                    fpos = ftell(pFile);
                    if (fpos > 0)
                    {
                        fseek(pFile, 0, SEEK_SET);
                        length = fpos;
                        data = malloc((size_t)length);
                        if (data != NULL)
                        {
                            length = (LwU64)fread(data, 1, (size_t)length, pFile);
                            if (length < (LwU64)fpos)
                            {
                                free(data);
                                data = NULL;
                            }
                        }
                    }
                    fclose(pFile);
                    validParams = TRUE;
                }
            }
            else if (GetSafeExpressionEx(params, &address, &params))
            {
                length = (LwU64)GetSafeExpression(params);
                if (length != 0)
                {
                    data = malloc((size_t)length);
                    if (data != NULL)
                        osReadMemByType(address, data, length, &length, SYSTEM_VIRT);
                    validParams = TRUE;
                }
            }
        }
    }

    if (validParams)
    {
        if (data != NULL)
        {
            printPrbMessage(msgName, data, (LwU32)length);
        }
    }
    else
    {
        dprintf("lw: Usage: !lw.prbdec <message_name> "
                "(<virtual_address> <length> | -f <filename>)\n");
    }

    free(data);
}

//-----------------------------------------------------
// dumpinit
// Intializes LwWatch dump mode from a protobuf message file.
//-----------------------------------------------------
DECLARE_API( dumpinit )
{
    char zipName[128];
    char innerName[128];
    char argBuf[128];
    LW_STATUS status;
    lwMode = MODE_DUMP;

    // Empty strings
    zipName[0] = '\0';
    innerName[0] = '\0';

    // Get filenames for zip file and inner file
    while (args != NULL)
    {
        args = getToken(args, argBuf, NULL);
        if (argBuf[0] == '\0')
        {
            break;
        }
        else
        {
            strcpy(zipName, argBuf);
            if (args != NULL)
            {
                args = getToken(args, innerName, NULL);
            }
        }
    }

    status = dumpModeInit(zipName, innerName);
    if (status == LW_OK)
    {
        dprintf("lw: LwWatch initialized for read-only dump session.\n");
    }
    else
    {
        lwBar0 = 0;
        lwMode = MODE_NONE;
    }
}

//-----------------------------------------------------
// Prints a protobuf field from the root LwDump message.
//   i.e. -field gpu_general.rm_data.gpuInstance
//   Without <field> the entire LwDump message is printed.
//-----------------------------------------------------
DECLARE_API( dumpprint )
{
    CHECK_INIT(MODE_DUMP);

    if (dumpModePrint(args) != LW_OK)
    {
        dprintf("lw: Usage: !lw.dumpprint[<message_field>]\n");
    }
}

//-----------------------------------------------------
// Dump mode keeps track of memory reads that are
//   not present in the dump file. This subcommand
//   prints all missing memory ranges in the format
//   of a live system dump call (for colwenience).
//-----------------------------------------------------
DECLARE_API( dumpfeedback )
{
    char *filename = "rawmem.txt";

    CHECK_INIT(MODE_DUMP);

    parseCmd(args, "f", 1, &filename);
    dumpModeFeedback(filename);
}

#else
void printPrbMessage(char *msgName, void* data, LwU32 length)
{
}
#endif // LWDEBUG_SUPPORTED

/**
 * @brief      Parse arguments to get memory space type
 *
 * @details    Parse first part of the arguments to "gv" functions
 *             Supported memory space types:
 *             smmu/iommu, gmmu, bar1, bar2, ifb, pmu, fla, all
 *
 * @param      args       The arguments
 * @param      pVMemType  The vitrual memory type
 * @param      pId        Id, type depending on the memory type
 *
 * @return     LW_OK on success, ERROR code otherwise
 */
static LW_STATUS gvParseVMem(char **args, VMemTypes *pVMemType, VMEM_INPUT_TYPE* pId)
{
    char   *params;
    LwU32   numChannels;
    LwU64   id = 0;
    LW_STATUS status = LW_OK;

    // -smmu/iommu <asid>
    if (parseCmd(*args, "smmu", 1, &params) || parseCmd(*args, "iommu", 1, &params))
    {
        VMEM_INPUT_TYPE_IOMMU *pIommu = (VMEM_INPUT_TYPE_IOMMU*)pId;
        *pVMemType = VMEM_TYPE_IOMMU;

        GetSafeExpressionEx(params, &id, &params);

        // Save the ASID value
        pIommu->asId = (LwU32)id;
        dprintf("lw: ASID %d\n", (LwU32)id);
    }
    // -gmmu
    else if (parseCmd(*args, "gmmu", 0, &params))
    {
        // -gmmu -ch <engine_string> <chid> OR
        // -gmmu -ch <type_enum> <instance_id> <chid>
        if (parseCmd(*args, "ch", 1, &params))
        {
            VMEM_INPUT_TYPE_CHANNEL *pCh = (VMEM_INPUT_TYPE_CHANNEL*)pId;
            *pVMemType = VMEM_TYPE_CHANNEL;

            // initialize
            pCh->chId = 0;
            pCh->rlId = 0;

            if ((params[0] >= '0') && (params[0] <= '9'))
            {
                // params is a number, so assume (type, inst) combination
                LwU64 typeEnum;
                LwU64 instId;

                GetSafeExpressionEx(params, &typeEnum, &params);
                GetSafeExpressionEx(*args, &instId, args);

                dprintf("lw: engine typeEnum %lld, instId %lld\n", typeEnum, instId);
                if ((typeEnum > LW_U32_MAX) || (instId > LW_U32_MAX))
                {
                    dprintf("ERROR: typeEnum and/or instId out of 32 bit range.\n");
                    return LW_ERR_ILWALID_ARGUMENT;
                }

                status = pFifo[indexGpu].fifoXlateFromDevTypeAndInstId((LwU32)typeEnum, (LwU32)instId, ENGINE_INFO_TYPE_RUNLIST, &pCh->rlId);
                if (status != LW_OK)
                {
                    dprintf("ERROR: failed to fetch engine information from typeEnum, instId\n");
                    return LW_ERR_ILWALID_ARGUMENT;
                }
            }
            else
            {
                // params is a string, so assuming engine name string
                status = pFifo[indexGpu].fifoXlateFromEngineString(params, ENGINE_INFO_TYPE_RUNLIST, &pCh->rlId);
                if (status != LW_OK)
                {
                    dprintf("Engine name %s does not map to a valid engine\n", params);
                    return LW_ERR_ILWALID_ARGUMENT;
                }

                dprintf("lw: engine name %s\n", params);
            }

            GetSafeExpressionEx(*args, &id, args);
            dprintf("lw: ChId 0x%llx\n", id);
            if (id > LW_U32_MAX)
            {
                dprintf("ERROR: chId out of 32 bit range\n");
                return LW_ERR_ILWALID_ARGUMENT;
            }

            // Save the CHID value
            pCh->chId = (LwU32)id;
            //
            // Validate channel value
            // (if possible, i.e. fifoGetNumChannels implemented)
            //
            numChannels = (LwU32)pFifo[indexGpu].fifoGetNumChannels(pCh->rlId);
            if ((numChannels > 0) && (pCh->chId >= numChannels))
            {
                dprintf("ERROR: ChId is too large. supported numChannels = 0x%x\n", numChannels);
                return LW_ERR_ILWALID_ARGUMENT;
            }

            dprintf("lw: Host runlistId %d\n", pCh->rlId);
        }
        // - gmmu -instptr <instptr> <-vidmem/-sysmem>
        else if (parseCmd(*args, "instptr", 1, &params))
        {
            VMEM_INPUT_TYPE_INST *pInst = (VMEM_INPUT_TYPE_INST*)pId;
            *pVMemType = VMEM_TYPE_INST_BLK;

            GetSafeExpressionEx(params, &id, &params);

            // Save the instance ptr value
            pInst->instPtr = id;
            dprintf("lw: Instance Ptr 0x%llx\n", id);

            if (parseCmd(*args, "vidmem", 0, &params))
            {
                pInst->targetMemType = FRAMEBUFFER;
            }
            else if (parseCmd(*args, "sysmem", 0, &params))
            {
                pInst->targetMemType = SYSTEM_PHYS;
            }
            else
            {
                dprintf("Invalid aperture\n");
                return LW_ERR_ILWALID_ARGUMENT;
            }
        }
        else
        {
            if(IsGA100orLater())
            {
                dprintf("Engine_string-Chid or (engine type enum, instance id)-chid or InstPtr-Aperture combination needs to be provided\n");
                return LW_ERR_ILWALID_ARGUMENT;
            }
            else
            {
                VMEM_INPUT_TYPE_CHANNEL *pCh = (VMEM_INPUT_TYPE_CHANNEL*)pId;
                *pVMemType = VMEM_TYPE_CHANNEL;

                GetSafeExpressionEx(*args, &id, args);

                // Save CHID in pre ampere case
                pCh->chId = (LwU32)id;
                //
                // Validate channel value
                // (if possible, i.e. fifoGetNumChannels implemented)
                //
                numChannels = (LwU32)pFifo[indexGpu].fifoGetNumChannels(pCh->rlId);
                if ((numChannels > 0) && (pCh->chId >= numChannels))
                {
                    dprintf("ERROR: Invalid CHID %d\n", pCh->chId);
                    return LW_ERR_ILWALID_ARGUMENT;
                }

                dprintf("lw: Channel Id = %d\n", pCh->chId);
            }
        }
    }
    else
    {
        if (parseCmd(*args, "bar1", 0, NULL))
        {
            *pVMemType = VMEM_TYPE_BAR1;
            dprintf("lw: bar1\n");
        }
        else if (parseCmd(*args, "bar2", 0, NULL))
        {
            *pVMemType = VMEM_TYPE_BAR2;
            dprintf("lw: bar2\n");
        }
        else if (parseCmd(*args, "ifb", 0, NULL))
        {
            *pVMemType = VMEM_TYPE_IFB;
            dprintf("lw: ifb\n");
        }
        else if (parseCmd(*args, "pmu", 0, NULL))
        {
            *pVMemType = VMEM_TYPE_PMU;
            dprintf("lw: pmu\n");
        }
        else if (parseCmd(*args, "fla", 1, &params))
        {
            VMEM_INPUT_TYPE_FLA *pFla = (VMEM_INPUT_TYPE_FLA*)pId;
            *pVMemType = VMEM_TYPE_FLA;

            GetSafeExpressionEx(params, &id, &params);
            pFla->flaImbAddr = id;
            dprintf("lw: FLA IMB Address 0x%llx\n", pFla->flaImbAddr);

            if (parseCmd(*args, "vidmem", 0, &params))
            {
                pFla->targetMemType = FRAMEBUFFER;
            }
            else if (parseCmd(*args, "sysmem", 0, &params))
            {
                pFla->targetMemType = SYSTEM_PHYS;
            }
            else
            {
                dprintf("Invalid aperture\n");
                return LW_ERR_ILWALID_ARGUMENT;
            }
        }
        // Retaining this for legacy reasons, assume it's just a chId
        else
        {
            if(IsGA100orLater())
            {
                dprintf("Engine_string-Chid or (engine type enum, instance id)-Chid or InstPtr-Aperture combination needs to be provided\n");
                return LW_ERR_ILWALID_ARGUMENT;
            }
            else
            {
                if (GetSafeExpressionEx(*args, &id, args))
                {
                    VMEM_INPUT_TYPE_CHANNEL *pCh = (VMEM_INPUT_TYPE_CHANNEL*)pId;
                    *pVMemType = VMEM_TYPE_CHANNEL;

                    // Save the channel ID value
                    pCh->chId = (LwU32)id;

                    //
                    // Validate channel value
                    // (if possible, i.e. fifoGetNumChannels implemented)
                    //
                    numChannels = (LwU32)pFifo[indexGpu].fifoGetNumChannels(pCh->rlId);
                    if ((numChannels > 0) && (pCh->chId >= numChannels))
                    {
                        dprintf("ERROR: Invalid CHID %d\n", pCh->chId);
                        return LW_ERR_ILWALID_ARGUMENT;
                    }
                    dprintf("lw: Channel %d\n", pCh->chId);
                }
                else    // Invalid arguments
                {
                    dprintf("ERROR: Unrecognized VMEM argument\n");
                    return LW_ERR_ILWALID_ARGUMENT;
                }
            }
        }
    }
    return LW_OK;
}

//--------------------------------------------------------------
// gvrd -smmu/iommu <asId> <vAddr> [length] OR
// gvrd -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> [length] OR
// gvrd -gmmu <-ch <engine_string> <chid>> <vAddr> [length] OR
// gvrd -gmmu <-ch <type_enum> <instance_id> <chid>> <vAddr> [length] OR
// gvrd -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> [length]
//--------------------------------------------------------------
DECLARE_API( gvrd )
{
    VMEM_INPUT_TYPE  Id;
    LwU64            vAddr   = 0;
    LwU64            length  = 0x80;
    LwU64            value;
    char             *buffer = NULL;
    VMemSpace        vMemSpace;
    VMemTypes        vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gvrd_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gvrd does not support FLA");
        return;
    }

    if (GetSafeExpressionEx(args, &vAddr, &args))
    {
        // we have at least two arguments
        value = GetSafeExpression(args);
        if (value > 0)
        {
            length = (value + 3) & ~3ULL;

            if (length < 4)
            {
                length = 4;
            }
        }
    }
    else
    {
        goto gvrd_bad_usage;
        return;
    }

    buffer = (char*)malloc((size_t)length);
    if (buffer == NULL)
    {
        dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
        return;
    }

    MGPU_LOOP_START;
    {
        if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
        {
            if (vmemRead(&vMemSpace, vAddr, (LwU32)length, (void*)buffer) == LW_OK)
            {
                printBuffer(buffer, (LwU32)length, vAddr, 4);
            }
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }

    free(buffer);
    return;

gvrd_bad_usage:
    dprintf("lw: Usage: !lw.gvrd -smmu/iommu <asId> <vAddr> [length] OR\n");
    dprintf("!lw.gvrd -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> [length] OR\n");
    dprintf("!lw.gvrd -gmmu <-ch <rlid> <chid>> <vAddr> [length] OR\n");
    dprintf("!lw.gvrd -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> [length]\n");
}

//-----------------------------------------------------------------------
// gvwr -smmu/iommu <asId> <vAddr> <data> [data] ... OR
// gvwr -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> <data> [data] ... OR
// gvwr -gmmu <-ch <engine_string> <chid>> <vAddr> <data> [data] ... OR
// gvwr -gmmu <-ch <type_enum> <instance_id> <chid>> <vAddr> <data> [data] ... OR
// gvwr -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> <data> [data] ...
//-----------------------------------------------------------------------
DECLARE_API( gvwr )
{
    VMEM_INPUT_TYPE Id;
    LwU64           vAddr    = 0;
    LwU64           data     = 0;
    LwU32          *buffer   = NULL;
    LwU32          *buffer2  = NULL;
    LwU32           size     = 1;
    LwU32           count    = 0;
    VMemSpace       vMemSpace;
    VMemTypes       vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    //
    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gvwr_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gvwr does not support FLA");
        return;
    }

    if (GetSafeExpressionEx(args, &vAddr, &args))
    {
        // Allocate buffer
        buffer = (LwU32*)malloc(size*sizeof(LwU32));
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }
        // Loop through data
        while (GetSafeExpressionEx(args, &data, &args))
        {
            buffer[count++] = (LwU32)data;
            // Double buffer size
            if (count >= size)
            {
                size *= 2;
                buffer2 = (LwU32*)realloc(buffer, size*sizeof(LwU32));
                if (buffer2 == NULL)
                {
                    free(buffer);
                    dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
                    return;
                }
                buffer = buffer2;
            }
        }
        buffer[count++] = (LwU32)data;
    }
    else
    {
        goto gvwr_bad_usage;
        return;
    }

    MGPU_LOOP_START;
    {
        if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
        {
            vmemWrite(&vMemSpace, vAddr, (LwU32)(count*sizeof(LwU32)), buffer);
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }

    free(buffer);
    return;

gvwr_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf("!lw.gvwr -smmu/iommu <asId> <vAddr> <data> [data] ... OR\n");
    dprintf("!lw.gvwr -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> <data> [data] ... OR\n");
    dprintf("!lw.gvwr -gmmu <-ch <engine_string> <chid>> <vAddr> <data> [data] ... OR\n");
    dprintf("!lw.gvwr -gmmu <-ch <type_enum> <instance_id> <chid>> <vAddr> <data> [data] ... OR\n");
    dprintf("!lw.gvwr -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> <data> [data] ...\n");
}

//----------------------------------------------------------------------
// gvfill -smmu/iommu <asId> <vAddr> <data> <length> OR
// gvfill -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> <data> <length> OR
// gvfill -gmmu <-ch <engine_string> <chid>> <vAddr> <data> <length> OR
// gvfill -gmmu <-ch <type_enum> <instance_id> <chid>> <vAddr> <data> <length> OR
// gvfill -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> <data> <length>
//-----------------------------------------------------------------------
DECLARE_API( gvfill )
{
    VMEM_INPUT_TYPE Id;
    LwU64           vAddr    = 0;
    LwU64           data     = 0;
    LwU64           length   = 0;
    VMemSpace       vMemSpace;
    VMemTypes       vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    //
    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gvfill_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gvfill does not support FLA");
        return;
    }

    if (GetSafeExpressionEx(args, &vAddr, &args) &&
        GetSafeExpressionEx(args, &data,  &args))
    {
        // we have at least four arguments
        length = GetSafeExpression(args);
        length = (length + 3) & ~3ULL;
    }
    else
    {
        goto gvfill_bad_usage;
        return;
    }

    if (length < 4)
    {
        length = 4;
    }

    MGPU_LOOP_START;
    {
        if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
        {
            vmemFill(&vMemSpace, vAddr, (LwU32)length, (LwU32)data);
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;

    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }
    return;

gvfill_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf(" !lw.gvfill -smmu <asId> <vAddr> <data> <length> OR\n");
    dprintf("!lw.gvfill -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> <data> <length> OR\n");
    dprintf("!lw.gvfill -gmmu <-ch <engine_string> <chid>> <vAddr> <data> <length> OR\n");
    dprintf("!lw.gvfill -gmmu <-ch <type_enum> <instance_id> <chid>> <vAddr> <data> <length> OR\n");
    dprintf("!lw.gvfill -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> <data> <length>\n");
}

//------------------------------------------------------------------------------
// gvcp -smmu/iommu <asId> <vSrcAddr> <vDstAddr> [length] OR
// gvcp -gmmu <-bar1/-bar2/-ifb/<chid> <vSrcAddr> <vDstAddr> [length] OR
// gvcp -gmmu <-ch <engine_string> <chid>> <vSrcAddr> <vDstAddr> [length] OR
// gvcp -gmmu <-ch <type_enum> <instance_id> <chid>> <vSrcAddr> <vDstAddr> [length] OR
// gvcp -gmmu -instptr <instptr> <-vidmem/-sysmem> <vSrcAddr> <vDstAddr> [length]
//------------------------------------------------------------------------------
DECLARE_API( gvcp )
{
    VMEM_INPUT_TYPE Id;
    LwU64           vSrcAddr = 0;
    LwU64           vDstAddr = 0;
    LwU64           length   = 0;
    void           *buffer   = NULL;
    VMemSpace       vMemSpace;
    VMemTypes       vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    /* TODO: Surely both parameters need full address qualification */

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gvcp_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gvcp does not support FLA");
        return;
    }

   if (GetSafeExpressionEx(args, &vSrcAddr, &args))
    {
        if (GetSafeExpressionEx(args, &vDstAddr, &args))
        {
            // we have at least four arguments
            length = GetSafeExpression(args);
            length = (length + 3) & ~3ULL;
        }
    }
    else
    {
        goto gvcp_bad_usage;
        return;
    }

    if (length < 4)
    {
        length = 4;
    }

    buffer = malloc((size_t)length);
    if (buffer == NULL)
    {
        dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
        return;
    }

    MGPU_LOOP_START;
    {
        if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
        {
            if (vmemRead(&vMemSpace, vSrcAddr, (LwU32)length, buffer) == LW_OK)
            {
                vmemWrite(&vMemSpace, vDstAddr, (LwU32)length, buffer);
            }
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;
    free(buffer);
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }

    return;

gvcp_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf("!lw.gvcp -smmu/iommu <asId> <vSrcAddr> <vDstAddr> [length] OR\n");
    dprintf("!lw.gvcp -gmmu <-bar1/-bar2/-ifb/<chid> <vSrcAddr> <vDstAddr> [length] OR\n");
    dprintf("!lw.gvcp -gmmu <-ch <engine_string> <chid>> <vSrcAddr> <vDstAddr> [length] OR\n");
    dprintf("!lw.gvcp -gmmu <-ch <type_enum> <instance_id> <chid>> <vSrcAddr> <vDstAddr> [length] OR\n");
    dprintf("!lw.gvcp -gmmu -instptr <instptr> <-vidmem/-sysmem> <vSrcAddr> <vDstAddr> [length]\n");
}

//-----------------------------------------------------
// gvdisp ChId vAddr width height blockWidth blockHeight blockDepth format
// - read the
//-----------------------------------------------------
DECLARE_API( gvdisp )
{
#if defined(WIN32)
    VMEM_INPUT_TYPE Id;
    LwU64 vAddr;
    LwU64 width;
    LwU64 height;
    LwU64 blockWidth;
    LwU64 blockHeight;
    LwU64 blockDepth;
    LwU64 format;
    VMemTypes vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
        goto gvdisp_bad_usage;

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gvdisp does not support FLA");
        return;
    }

    if (!GetSafeExpressionEx(args, &vAddr, &args))
        goto gvdisp_bad_usage;
    if (!GetSafeExpressionEx(args, &width, &args))
        goto gvdisp_bad_usage;
    if (!GetSafeExpressionEx(args, &height, &args))
        goto gvdisp_bad_usage;
    if (!GetSafeExpressionEx(args, &blockWidth, &args))
        goto gvdisp_bad_usage;
    if (!GetSafeExpressionEx(args, &blockHeight, &args))
        goto gvdisp_bad_usage;
    if (!GetSafeExpressionEx(args, &blockDepth, &args))
        goto gvdisp_bad_usage;
    format = GetSafeExpression(args);

    MGPU_LOOP_START;
    {
        pVirt[indexGpu].virtDisplayVirtual(vMemType, Id.ch.chId, vAddr, (LwU32)width, (LwU32)height,
                                           (LwU32)blockWidth, (LwU32) blockHeight, (LwU32) blockDepth, (LwU32)format);
        dprintf("\n");
    }
    MGPU_LOOP_END;

    return;

gvdisp_bad_usage:
    dprintf(g_szGvDispUsage);
#else
    dprintf("lw: %s - Only supported on WIN32...\n", __FUNCTION__);
#endif
}

//-----------------------------------------------------
// gco2bl
//-----------------------------------------------------
DECLARE_API( gco2bl )
{
#if defined(WIN32)
    LwU64 cx;
    LwU64 cy;
    LwU64 width;
    LwU64 height;
    LwU64 logBlockWidth;
    LwU64 logBlockHeight;
    LwU64 logBlockDepth;
    LwU64 logGobWidth;
    LwU64 logGobHeight;
    LwU64 logGobDepth;
    LwU64 format;
    LwU32 offsetBL;

    CHECK_INIT(MODE_LIVE);

    if (!GetSafeExpressionEx(args, &cx, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &cy, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &width, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &height, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logBlockWidth, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logBlockHeight, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logBlockDepth, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logGobWidth, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logGobHeight, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logGobDepth, &args))
        goto bad_usage;
    format = GetSafeExpression(args);

    offsetBL = coordToBL((LwU32)cx, (LwU32)cy, (LwU32)width, (LwU32)height,
                (LwU32)logBlockWidth, (LwU32)logBlockHeight, (LwU32)logBlockDepth,
                (LwU32)logGobWidth, (LwU32)logGobHeight, (LwU32)logGobDepth, (LwU32)format);

    dprintf("BL offset: 0x%08x\n", offsetBL);

    return;

bad_usage:
    dprintf(g_szGCo2blUsage);
#else
    dprintf("lw: %s - Only supported on WIN32...\n", __FUNCTION__);
#endif
}

//-----------------------------------------------------
// gbl2co
//-----------------------------------------------------
DECLARE_API( gbl2co )
{
#if defined(WIN32)
    LwU64 offsetBL;
    LwU64 width;
    LwU64 height;
    LwU64 logBlockWidth;
    LwU64 logBlockHeight;
    LwU64 logBlockDepth;
    LwU64 logGobWidth;
    LwU64 logGobHeight;
    LwU64 logGobDepth;
    LwU64 format;
    LwU32 cx, cy;

    CHECK_INIT(MODE_LIVE);

    if (!GetSafeExpressionEx(args, &offsetBL, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &width, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &height, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logBlockWidth, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logBlockHeight, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logBlockDepth, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logGobWidth, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logGobHeight, &args))
        goto bad_usage;
    if (!GetSafeExpressionEx(args, &logGobDepth, &args))
        goto bad_usage;
    format = GetSafeExpression(args);

    blToCoord((LwU32)offsetBL, &cx, &cy, (LwU32)width, (LwU32)height,
        (LwU32)logBlockWidth, (LwU32)logBlockHeight, (LwU32)logBlockDepth,
        (LwU32)logGobWidth, (LwU32)logGobHeight, (LwU32)logGobDepth, (LwU32)format);

    dprintf("Coords: x 0x%x (%u), y 0x%x (%u)\n", cx, cx, cy, cy);

    return;

bad_usage:
    dprintf(g_szGBl2coUsage);
#else
    dprintf("lw: %s - Only supported on WIN32...\n", __FUNCTION__);
#endif
}

//-----------------------------------------------------
// gvdiss chId vAddr <length> <shaderType>
// - Read the memory at vAddr and try to decode it
//   as a shader using sass
//-----------------------------------------------------
DECLARE_API( gvdiss )
{
    LwU64   shaderType;
    LwU64   chId;
    LwU64   vAddr;
    LwU64   Length;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &chId, &args) &&
        GetSafeExpressionEx(args, &vAddr, &args) &&
        GetSafeExpressionEx(args, &Length, &args))
    {
        Length = (Length + 3) & ~3ULL;
        shaderType = GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.gvdiss <chId> <vAddr> <length> <shaderType>\n"
                "for shader type, use the following enum\n"
                "0 - vertex shader\n"
                "1 - geometry shader\n"
                "2 - pixel shader\n"
                "3 - compute shader\n");
        return;
    }

    MGPU_LOOP_START;
    {
        pFb[indexGpu].fbDisassembleVirtual((LwU32)chId, vAddr, (LwU32)Length, (LwU32) shaderType);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// cpudisp <pAddr> <width> <height> <blockWidth> <blockHeight> <blockDepth> <format>
//-----------------------------------------------------
DECLARE_API( cpudisp )
{
    LwU64 pAddr;
    LwU64 width;
    LwU64 height;
    LwU64 blockWidth;
    LwU64 blockHeight;
    LwU64 blockDepth;
    LwU64 format;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &pAddr, &args) &&
        GetSafeExpressionEx(args, &width, &args) &&
        GetSafeExpressionEx(args, &height, &args) &&
        GetSafeExpressionEx(args, &blockWidth, &args) &&
        GetSafeExpressionEx(args, &blockHeight, &args) &&
        GetSafeExpressionEx(args, &blockDepth, &args))
    {
        format = GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.cpudisp <pAddr> <width> <height> "
                "<blockWidth> <blockHeight> <blockDepth> "
                "<format>\n");
        return;
    }

    MGPU_LOOP_START;
    {
        pVirt[0].virtDisplayVirtual(MT_CPUADDRESS, 0, pAddr, (LwU32)width, (LwU32)height,
                              (LwU32)blockWidth, (LwU32) blockHeight, (LwU32) blockDepth, (LwU32)format);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// fbrd <fbOffset> [fbLength] [size]
// - read the FB from fbOffset to fbOffset + fbLength
//-----------------------------------------------------
DECLARE_API( fbrd )
{
    LwU64   fbOffset = 0;
    LwU64   fbLength = 0x80;
    LwU64   fbSize = 4;
    char*   buffer = NULL;

    CHECK_INIT(MODE_LIVE | MODE_DUMP);

    if (!args || *args == '\0')
    {
        dprintf("lw: Usage: !lw.fbrd <fbOffset> [fbLength] [size]\n");
        return;
    }
    else if (GetSafeExpressionEx(args, &fbOffset, &args))
    {
        if (GetSafeExpressionEx(args, &fbLength, &args))
        {
            fbSize = GetSafeExpression(args);
        }
    }

    // Ensure fbSize is one of the valid values
    if ((fbSize != 1) && (fbSize != 2) && (fbSize != 4))
    {
        fbSize = 4;
    }

    // Align fbOffset to a multiple of fbSize
    fbOffset &= ~(fbSize - 1);

    // Align fbLength to a multiple of fbSize
    fbLength = (fbLength + (fbSize - 1)) & ~(fbSize - 1);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(GUEST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        switch (lwMode)
        {
            case MODE_LIVE:
                // Allocate buffer
                buffer = (char*)malloc((size_t)fbLength);
                if (buffer)
                {
                    // Read from FB at fbOffset
                    if (pFb[indexGpu].fbRead(fbOffset, (void*)buffer, (LwU32)fbLength) != LW_ERR_GENERIC)
                    {
                        printBuffer(buffer, (LwU32)fbLength, fbOffset, (LwU8)fbSize);
                    }
                    // Free buffer
                    free(buffer);
                }
                break;
            case MODE_DUMP:
                dumpModeReadFb(fbOffset, (LwU32)fbLength, (LwU8)fbSize);
                break;
            case MODE_NONE:
            default:
                break;
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }
}

//-----------------------------------------------------
// fbwr <fboffset> <val32> [val32] ...
// - writes value(s) to a fb 32-bit lw memory location
//-----------------------------------------------------
DECLARE_API( fbwr )
{
    LwU64 fbOffset = 0;
    LwU64 data = 0;
    LwU32* buffer = NULL;
    LwU32* buffer2 = NULL;
    LwU32 size = 1;
    LwU32 count = 0;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &fbOffset, &args))
    {
        // Allocate buffer
        buffer = (LwU32*)malloc(size*sizeof(LwU32));
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Loop through data
        while (GetSafeExpressionEx(args, &data, &args))
        {
            buffer[count++] = (LwU32)data;
            // Double buffer size
            if (count >= size)
            {
                size *= 2;
                buffer2 = (LwU32*)realloc((void*)buffer, size*sizeof(LwU32));
                if (buffer2 == NULL)
                {
                    free(buffer);
                    dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
                    return;
                }
                buffer = buffer2;
            }
        }
        buffer[count++] = (LwU32)data;

    }
    else
    {
        dprintf("lw: Usage: !lw.fbwr <fboffset> <val32> [val32] ...\n");
        return;
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(GUEST_PHYSICAL);
    }

    // Write buffer to FB
    MGPU_LOOP_START;
    {
        pFb[indexGpu].fbWrite(fbOffset, buffer, (LwU32)(count*sizeof(LwU32)));
        dprintf("\n");
    }
    MGPU_LOOP_END;

    // Free buffer
    free(buffer);
    buffer = NULL;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }
}


//-----------------------------------------------------
// fbfill <fboffset> <val32> <fbLength>
// - fills value to a fb 32-bit lw memory location
//-----------------------------------------------------
DECLARE_API( fbfill )
{
    LwU64 fbOffset = 0;
    LwU64 value = 0;
    LwU64 fbLength = 0;
    LwU32 val32 = 0;
    LwU32* buffer = NULL;
    LwU32 i = 0;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &fbOffset, &args) &&
        GetSafeExpressionEx(args, &value, &args))
    {
        fbLength = GetSafeExpression(args);

        // Align fbLength to a multiple of 4
        fbLength = (fbLength + 3) & ~3ULL;

        // Cast to LwU32
        val32 = (LwU32)value;
    }
    else
    {
        dprintf("lw: Usage: !lw.fbfill <fbOffset> <val32> <fbLength>\n");
        return;
    }

    // Ensure length at least 4
    if (fbLength < 4)
    {
        fbLength = 4;
    }

    // Allocate buffer
    buffer = (LwU32*)malloc((size_t)fbLength);
    if (buffer)
    {
        // Fill buffer with val32
        for (i = 0; i < fbLength/4; i++)
        {
            buffer[i] = val32;
        }
        // Write buffer to FB
        MGPU_LOOP_START;
        {
            pFb[indexGpu].fbWrite(fbOffset, buffer, (LwU32)fbLength);
            dprintf("\n");
        }
        MGPU_LOOP_END;

        // Free buffer
        free(buffer);
        buffer = NULL;
    }
}

//-----------------------------------------------------
// fbcp <fbSrcOffset> <fbDstOffset> [fbLength]
// - copy the FB from fbSrcOffset to fbDstOffset
//-----------------------------------------------------
DECLARE_API( fbcp )
{
    LwU64   fbSrcOffset = 0;
    LwU64   fbDstOffset = 0;
    LwU64   fbLength = 0;
    LwU32*   buffer = NULL;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &fbSrcOffset, &args))
    {
        if (GetSafeExpressionEx(args, &fbDstOffset, &args))
        {
            fbLength = GetSafeExpression(args);

            // Align fbLength to a multiple of 4
            fbLength = (fbLength + 3) & ~3ULL;
        }
    }
    else
    {
        dprintf("lw: Usage: !lw.fbcp <fbSrcOffset> <fbDstOffset> [fbLength]\n");
        return;
    }

    // Ensure length at least 4
    if (fbLength < 4)
    {
        fbLength = 4;
    }

    // Allocate buffer
    buffer = (LwU32*)malloc((size_t)fbLength);
    if (buffer)
    {
        MGPU_LOOP_START;
        {
            // Read from FB at fbSrcOffset
            if (pFb[indexGpu].fbRead(fbSrcOffset, buffer, (LwU32)fbLength) != LW_ERR_GENERIC)
            {
                // Write to FB at fbDstOffset
                pFb[indexGpu].fbWrite(fbDstOffset, buffer, (LwU32)fbLength);
            }
            dprintf("\n");
        }
        MGPU_LOOP_END;

        // Free buffer
        free(buffer);
        buffer = NULL;
    }
}

//-----------------------------------------------------
// pcieinfo, pcieinfo all, pcieinfo <domain> <bus> <dev> <function>
// - get the current PCI-Express info
//-----------------------------------------------------
DECLARE_API( pcieinfo )
{
    LwU64 domain, bus, dev, func;

    CHECK_INIT(MODE_LIVE);

    if (args && strcmp(args, "all") == 0)
    {
        getPexAllInfo();
    }
    else if (GetSafeExpressionEx(args, &domain, &args) &&
             GetSafeExpressionEx(args, &bus, &args) &&
             GetSafeExpressionEx(args, &dev, &args))
    {
        func = GetSafeExpression(args);
        getPexSpecifiedInfo((LwU16)domain, (LwU8)bus, (LwU8)dev, (LwU8)func);
    }
    else
    {
        getPexGPUInfo();
    }
    dprintf("\n");
}

//---------------------------------------------------------------------------
// pcie3evtlogdmp
// - The new gen3 design has an LA which logs XVE events to an on chip ram.
// - This extension will dump the event log.
//---------------------------------------------------------------------------
DECLARE_API( pcie3evtlogdmp )
{
    CHECK_INIT(MODE_LIVE);

    if (IsGM107orLater())
    {
        printPcie3EvtLogDmp();
    }
    else
    {
        dprintf("This feature only applies to GM107 and later GPUs.");
    }
}

//-----------------------------------------------------
// pcie <domain> <bus> <dev> <func> [offset] [length]
// - get the current PCI-Express info
//-----------------------------------------------------
DECLARE_API( pcie )
{
    LwU64 domain, bus, dev, func;
    LwU64 offset = 0;
    LwU64 length = 64;
    BOOL basicArgRead = FALSE;
    BOOL ilwalidArg = FALSE;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &domain, &args) &&
        GetSafeExpressionEx(args, &bus, &args) &&
        GetSafeExpressionEx(args, &dev, &args))
    {
        if ((domain >= PCIE_MAX_DOMAIN) ||
            (bus >= PCIE_MAX_BUS) || (dev >= PCIE_MAX_DEV))
            ilwalidArg = TRUE;
        else if (GetSafeExpressionEx(args, &func, &args))
        {
            if (func >= PCIE_MAX_FUNC)
                ilwalidArg = TRUE;
            else if (GetSafeExpressionEx(args, &offset, &args))
            {
                length = GetSafeExpression(args);
                if ((offset >= PCIE_MAX_OFFSET) ||
                    (offset + length) >= PCIE_MAX_OFFSET)
                    ilwalidArg = TRUE;
            }
        }
        basicArgRead = TRUE;
    }

    if (ilwalidArg)
        dprintf("lw: Bad Arguments\n");
    else if(basicArgRead)
        printPci((LwU16)domain, (LwU8)bus, (LwU8)dev, (LwU8)func, (LwU32)offset, (LwU32)length);
    else
        dprintf("lw: Usage: !lw.pcie <domain> <bus> <dev> <func> [offset] [length]\n");

    dprintf("\n");
}

//-----------------------------------------------------
// clocks [-disp]
// - get the current mem and lw clocks
//-----------------------------------------------------
DECLARE_API( clocks )
{
    BOOL isDisp = FALSE;

    CHECK_INIT(MODE_LIVE);

    isDisp = parseCmd(args, "disp", 0, NULL);

    MGPU_LOOP_START;
    {
        if (isDisp)
        {
            pDisp[indexGpu].dispReadDispClkSettings();
            pDisp[indexGpu].dispReadPixelClkSettings();
            pDisp[indexGpu].dispReadSorClkSettings();
        }
        else
        {
            pClk[indexGpu].clkGetClocks();
            dprintf("\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// clockregs
// - get all of the current clock regs
//-----------------------------------------------------
DECLARE_API( clockregs )
{
    CHECK_INIT(MODE_LIVE);

    pClk[indexGpu].clkGetClockRegs();
    dprintf("\n");
}

//-----------------------------------------------------
// plls
//-----------------------------------------------------
DECLARE_API( plls )
{
    CHECK_INIT(MODE_LIVE);

    pClk[indexGpu].clkGetPlls();
    dprintf("\n");
}

//-----------------------------------------------------
// pllregs
// - get all of the current pll regs
//-----------------------------------------------------
DECLARE_API( pllregs )
{
    CHECK_INIT(MODE_LIVE);

    pClk[indexGpu].clkGetPllRegs();
    dprintf("\n");
}

//-----------------------------------------------------
// powergate
// - get all of the current clock regs
//-----------------------------------------------------
DECLARE_API( powergate )
{
    CHECK_INIT(MODE_LIVE);

    pClk[indexGpu].clkGetPowerGateRegs();
    dprintf("\n");
}

//-----------------------------------------------------
// cluster
//
//-----------------------------------------------------
DECLARE_API( cluster )
{
    CHECK_INIT(MODE_LIVE);

    pClk[indexGpu].clkGetCpuClusterInfo();
    dprintf("\n");
}

//-----------------------------------------------------
//-----------------------------------------------------
// cntrfreq -inpcnt -domain <val>
// - use counters to determine frequency of various clocks
// - inpcnt is used to provide number of input xtal clocks to count
//   the desired clocks and determine the frequency
// - domain is used to read a specific domain, if requested
//   'all' otherwise
//   <val> can be DRAM|GPC|DISP|HOST|HUB|LWD|PWR|SYS|SPPLL0|SPPLL1|UTILS|VCLK|XBAR
//-----------------------------------------------------
DECLARE_API( cntrfreq )
{
    LwU32 clkSel;
    char  clkDomainName[CLK_DOMAIN_NAME_STR_LEN] = "all";
    char *params;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "domain", 1, &params))
    {
        getToken(params, clkDomainName, NULL);
    }
    clkSel = (LwU32)GetSafeExpression(args);

    MGPU_LOOP_START;
    {
        pClk[indexGpu].clkCounterFrequency(clkSel, clkDomainName);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// clklutread <nafllId> {tempIdx}
// - prints out the programmed LUT values for the given NAFLL ID
//   and an optional temperature index
// - nafllId is from GPC0|GPC1|GPC2|GPC3|GPC4|GPC5|GPCS|SYS|XBAR|LWD|HOST
// - optional temperature index, default 'current temperature index'
//-----------------------------------------------------
DECLARE_API( clklutread )
{
    LwU32 nafllId;
    LwU32 tempIdx = CLK_LUT_TEMP_IDX_ILWALID;
    char  nafllName[8];

    CHECK_INIT(MODE_LIVE);

    // Get the NAFLL name
    args = getToken(args, nafllName, NULL);

    // See if temperature index is passed in as well
    if (strcmp(args, "") != 0)
    {
        tempIdx = (LwU32)GetSafeExpression(args);
        if (tempIdx > CLK_LUT_TEMP_IDX_MAX)
        {
            dprintf("Invalid temperature index (%d) specified, using the programmed value instead!!!\n", tempIdx);
        }
    }

    // Translate the NAFLL name to NAFLL ID
    if (strncmp(nafllName, "GPC0", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC0;
    }
    else if (strncmp(nafllName, "GPC1", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC1;
    }
    else if (strncmp(nafllName, "GPC2", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC2;
    }
    else if (strncmp(nafllName, "GPC3", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC3;
    }
    else if (strncmp(nafllName, "GPC4", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC4;
    }
    else if (strncmp(nafllName, "GPC5", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC5;
    }
    else if (strncmp(nafllName, "GPC6", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC6;
    }
    else if (strncmp(nafllName, "GPC7", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPC7;
    }
    else if (strncmp(nafllName, "GPCS", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_GPCS;
    }
    else if (strncmp(nafllName, "SYS", 3) == 0)
    {
        nafllId = CLK_NAFLL_ID_SYS;
    }
    else if (strncmp(nafllName, "XBAR", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_XBAR;
    }
    else if (strncmp(nafllName, "LWD", 3) == 0)
    {
        nafllId = CLK_NAFLL_ID_LWD;
    }
    else if (strncmp(nafllName, "HOST", 4) == 0)
    {
        nafllId = CLK_NAFLL_ID_HOST;
    }
    else
    {
        nafllId = CLK_NAFLL_ID_UNDEFINED;
        dprintf("ERROR: Invalid NAFLL ID!!!\n");

        dprintf("lw: Usage: !lw.clklutread <nafllId> {tempIdx}\n"
                "where nafllId is one of: GPC0|GPC1|GPC2|GPC3|GPC4|GPC5|GPC6|GPC7|GPCS|SYS|XBAR|LWD|HOST\n"
                "optional temperature index, default 'current temperature index'\n");
        return;
    }

    MGPU_LOOP_START;
    {
        pClk[indexGpu].clkNafllLutRead(nafllId, tempIdx);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// class <classNum>
// - prints out the class name
//-----------------------------------------------------
DECLARE_API( classname )
{
    LwU32 classNum;
    classNum = (LwU32) GetSafeExpression(args);
    dprintf("lw: class 0x%03x", classNum);
    printClassName(classNum);
    dprintf("\n");
}

//-----------------------------------------------------
// gpde -smmu/iommu <asId> <begin> [end] OR
// gpde -gmmu <-bar1/-bar2/-ifb/<chid> <begin> [end] OR
// gpde -gmmu <-ch <engine_string> <chid>> <begin> [end] OR
// gpde -gmmu <-ch <type_enum> <instance_id> <chid>> <begin> [end] OR
// gpde -gmmu -instptr <instptr> <-vidmem/-sysmem> <begin> [end]
//-----------------------------------------------------
DECLARE_API( gpde )
{
    VMEM_INPUT_TYPE Id;
    LwU64           begin;
    LwU64           end;
    VMemSpace       vMemSpace;
    VMemTypes       vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gpde_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gpde does not support FLA");
        return;
    }

    if (GetSafeExpressionEx(args, &begin, &args))
    {
        if (!GetSafeExpressionEx(args, &end, &args))
    {
            end = begin;
        }
    }
    else
    {
        goto gpde_bad_usage;
        return;
    }

    // we have at least two arguments
    MGPU_LOOP_START;
    {
        if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
        {
            vmemDoPdeDump(&vMemSpace, (LwU32)begin, (LwU32)end);
            dprintf("\n");
        }
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }

    return;

gpde_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf("!lw.gpde -smmu/iommu <asId> <begin> [end] OR\n");
    dprintf("!lw.gpde -gmmu <-bar1/-bar2/-ifb/<chid> <begin> [end] OR\n");
    dprintf("!lw.gpde -gmmu <-ch <engine_string> <chid>> <begin> [end] OR\n");
    dprintf("!lw.gpde -gmmu <-ch <type_enum> <instance_id> <chid>> <begin> [end] OR\n");
    dprintf("!lw.gpde -gmmu -instptr <instptr> <-vidmem/-sysmem> <begin> [end]\n");
}

//-------------------------------------------------------------------
// gpte -smmu/iommu <asId> <pde_id> <begin> [end] OR
// gpte -gmmu <-bar1/-bar2/-ifb/<chid> <pde_id> <begin> [end] OR
// gpte -gmmu <-ch <engine_string> <chid>> <pde_id> <begin> [end] OR
// gpte -gmmu <-ch <type_enum> <instance_id> <chid>> <pde_id> <begin> [end] OR
// gpte -gmmu -instptr <instptr> <-vidmem/-sysmem> <pde_id> <begin> [end]
//-------------------------------------------------------------------
DECLARE_API( gpte )
{
    VMEM_INPUT_TYPE Id;
    LwU64           pde_id;
    LwU64           begin;
    LwU64           end;
    VMemSpace       vMemSpace;
    VMemTypes       vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gpte_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gpte does not support FLA");
        return;
    }

    if (GetSafeExpressionEx(args, &pde_id, &args))
    {
        if (GetSafeExpressionEx(args, &begin, &args))
        {
            if (!GetSafeExpressionEx(args, &end, &args))
            {
                end = begin;
            }
        }
        else
        {
            goto gpte_bad_usage;
            return;
        }

        // we have at least three arguments
        MGPU_LOOP_START;
        {
            if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
            {
                pVmem[indexGpu].vmemDoPteDump(&vMemSpace, (LwU32)pde_id, (LwU32)begin, (LwU32)end);
                dprintf("\n");
            }
        }
        MGPU_LOOP_END;

        //
        // turn off LwWatch mode and address type if we're running a VGPU, set address
        // type to invalid for good measure
        //
        if (isVirtual() == LW_TRUE)
        {
            setLwwatchAddrType(INVALID);
            setLwwatchMode(LW_FALSE);
        }

        return;
    }
    else
    {
        goto gpte_bad_usage;
        return;
    }

    return;

gpte_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf("!lw.gpte -smmu/iommu <asId> <pde_id> <begin> [end] OR\n");
    dprintf("!lw.gpte -gmmu <-bar1/-bar2/-ifb/<chid> <pde_id> <begin> [end] OR\n");
    dprintf("!lw.gpte -gmmu <-ch <engine_string> <chid>> <pde_id> <begin> [end] OR\n");
    dprintf("!lw.gpte -gmmu <-ch <type_enum> <instance_id> <chid>> <pde_id> <begin> [end] OR\n");
    dprintf("!lw.gpte -gmmu -instptr <instptr> <-vidmem/-sysmem> <pde_id> <begin> [end]\n");
}

#if !defined(USERMODE)
#if !LWWATCHCFG_IS_PLATFORM(UNIX)
//-----------------------------------------------------
// ptov <physAddr>
// - colwert a phys addr to a virt addr
//-----------------------------------------------------
DECLARE_API( ptov )
{
    LwU64 physAddr = 0;
    LwU64 virtAddr = 0;
    LwU32 flags = 0;

    if (GetSafeExpressionEx(args, &physAddr, &args))
    {
        flags = (LwU32)GetSafeExpression(args);
    }

    virtAddr = physToVirt(physAddr, flags);

    dprintf("lw: virtAddr:          " LwU64_FMT "\n", virtAddr);
    dprintf("\n");
}
#endif

#if !LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(UNIX_HWSNOOP)
//-----------------------------------------------------
// vtop <virtAddr>
// - colwert a virt addr to a phys addr
//-----------------------------------------------------
DECLARE_API( vtop )
{
    LwU64 virtAddr = 0;
    LwU64 physAddr = 0;
    LwU32 pid = 0;

    if (GetSafeExpressionEx(args, &virtAddr, &args))
    {
        pid = (LwU32)GetSafeExpression(args);
    }

    physAddr = virtToPhys(virtAddr, pid);

    dprintf("lw: physAddr:          " LwU64_FMT "\n", physAddr);
    dprintf("\n");
}
#endif
#endif // !USERMODE

//--------------------------------------------------------------------------------------------
// gvtop <-smmu/iommu asId> <virtAddr>OR
// gvtop -gmmu <-bar1/-bar2/-fla/-ifb/<chid>> <virtAddr> OR
// gvtop -fla <flaImbAddr> <-vidmem/-sysmem> <virtAddr> OR
// gvtop -gmmu <-ch <engine_string> <virtAddr> OR
// gvtop -gmmu <-ch <type_enum> <instance_id> <chid>> <virtAddr> OR
// gvtop -gmmu -instptr <instptr> <-vidmem/-sysmem> <virtAddr>
// - colwert a GPU virtual address to physical address
//--------------------------------------------------------------------------------------------
DECLARE_API( gvtop )
{
    VMEM_INPUT_TYPE   Id;
    LwU64             virtAddr;
    VMemSpace         vMemSpace;
    VMemTypes         vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    //
    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    // parse memory space type
    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gvtop_bad_usage;
        return;
    }

    // parse virtual address
    GetSafeExpressionEx(args, &virtAddr, &args);
    dprintf("lw: Virtual address 0x%llx\n", virtAddr);

    MGPU_LOOP_START;
    {
        if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
        {
            vmemVToP(&vMemSpace, virtAddr, NULL, NULL, LW_TRUE);
        }
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU,
    // set address type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }

    return;

gvtop_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf("!lw.gvtop -smmu/iommu <asId> <virtAddr> OR\n");
    dprintf("!lw.gvtop -gmmu <-bar1/-bar2/-ifb/<chid> <virtAddr> OR\n");
    dprintf("!lw.gvtop -fla <flaImbAddr> <-vidmem/-sysmem> <virtAddr> OR\n");
    dprintf("!lw.gvtop -gmmu <-ch <engine_string> <chid>> <virtAddr> OR\n");
    dprintf("!lw.gvtop -gmmu <-ch <type_enum> <instance_id> <chid>> <virtAddr> OR\n");
    dprintf("!lw.gvtop -gmmu -instptr <instptr> <-vidmem/-sysmem> <virtAddr>\n");
}

//---------------------------------------------------------------------------
// gptov -smmu/iommu <asId> <physAddr> <-vidmem/-sysmem> OR
// gptov -gmmu <-bar1/-bar2/-ifb/<chid> <physAddr> <-vidmem/-sysmem> OR
// gptov -gmmu <-ch <engine_string> <chid>> <physAddr> <-vidmem/-sysmem> OR
// gptov -gmmu <-ch <type_enum> <instance_id> <chid>> <physAddr> <-vidmem/-sysmem> OR
// gptov -gmmu -instptr <instptr> <-vidmem/-sysmem> <physAddr> <-vidmem/-sysmem>
// - colwert a physical address to GPU virtual address
//----------------------------------------------------------------------------
DECLARE_API( gptov )
{
    VMEM_INPUT_TYPE Id;
    LwU64           physAddr;
    VMemTypes       vMemType;
    BOOL            vidMem = TRUE;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gptov_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gptov does not support FLA");
        return;
    }

    if (GetSafeExpressionEx(args, &physAddr, &args))
    {
        if (parseCmd(args, "vidmem", 0, NULL))
        {
            vidMem = TRUE;
        }
        else if (parseCmd(args, "sysmem", 0, NULL))
        {
            vidMem = FALSE;
        }
        else
        {
            goto gptov_bad_usage;
            return;
        }

        MGPU_LOOP_START;
        {
            pVmem[indexGpu].vmemPToV(vMemType, &Id, physAddr, vidMem);
        }
        MGPU_LOOP_END;
    }
    else
    {
        goto gptov_bad_usage;
        return;
    }

    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_FALSE);
        setLwwatchAddrType(INVALID);
    }

    return;

gptov_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf("!lw.gptov -smmu/iommu <asId> <physAddr> <-vidmem/-sysmem> OR\n");
    dprintf("!lw.gptov -gmmu <-bar1/-bar2/-ifb/<chid> <physAddr> <-vidmem/-sysmem> OR\n");
    dprintf("!lw.gptov -gmmu <-ch <engine_string> <chid>> <physAddr> <-vidmem/-sysmem> OR\n");
    dprintf("!lw.gptov -gmmu <-ch <type_enum> <instance_id> <chid>> <physAddr> <-vidmem/-sysmem> OR\n");
    dprintf("!lw.gptov -gmmu -instptr <instptr> <-vidmem/-sysmem> <physAddr> <-vidmem/-sysmem>\n");
}

//-----------------------------------------------------------------------------
// gvpte -smmu/iommu <asId> <vAddr> OR
// gvpte -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> OR
// gvpte -gmmu <-ch <engine_string> <chid>> <vAddr> OR
// gvpte -gmmu <-ch <type_enum> <instance_id> <chid>> <vAddr> OR
// gvpte -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr>
// - Displays the PTE corresponding to the specified virtual address
//-----------------------------------------------------------------------------
DECLARE_API( gvpte )
{
    VMEM_INPUT_TYPE   Id;
    LwU64             virtAddr;
    VMemSpace         vMemSpace;
    VMemTypes         vMemType;

    memset(&Id, 0, sizeof(Id));

    CHECK_INIT(MODE_LIVE);

    if (vmemIsGvpteDeprecated())
    {
        dprintf("lw: gvpte is deprecated from Hopper+, please use gvtop.\n");
        return;
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch
    // mode bits, before gvParseMem which needs plugin reg access
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (gvParseVMem(&args, &vMemType, &Id) != LW_OK)
    {
        goto gvpte_bad_usage;
        return;
    }

    if (vMemType == VMEM_TYPE_FLA)
    {
        dprintf("lw: gvpte does not support FLA");
        return;
    }

    virtAddr = GetSafeExpression(args);
    dprintf("lw: Virtual address 0x%llx\n", virtAddr);

    MGPU_LOOP_START;
    {
        if (vmemGet(&vMemSpace, vMemType, &Id) == LW_OK)
        {
            vmemVToP(&vMemSpace, virtAddr, NULL, NULL, LW_TRUE);
        }
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }

    return;

gvpte_bad_usage:
    dprintf("lw: Usage:\n");
    dprintf("!lw.gvpte -smmu/iommu <asId> <vAddr> OR\n");
    dprintf("!lw.gvpte -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> OR\n");
    dprintf("!lw.gvpte -gmmu <-ch <engine_string> <chid>> <vAddr> OR\n");
    dprintf("!lw.gvpte -gmmu <-ch <type_enum> <instance_id> <chid>> <vAddr> OR\n");
    dprintf("!lw.gvpte -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr>\n");
}

//-----------------------------------------------------
// getcr <reg> <optHd>
// - read a cr register
//-----------------------------------------------------
DECLARE_API( getcr )
{
    LwU64 crReg = 0;
    LwU8 crVal = 0;
    LwU32 crtcOffset = 0;

    CHECK_INIT(MODE_LIVE);

    // read in the crReg and crtcOffset
    if (GetSafeExpressionEx(args, &crReg, &args))
    {
        crtcOffset = (LwU32) GetSafeExpression(args);
        if (crtcOffset == 1)
            crtcOffset = 0x2000;    // head 1
        else
            crtcOffset = 0;         // head 0
    }
    else
    {
        dprintf("lw: Usage: !lw.getcr <crReg> <optHd>\n");
        return;
    }

    if (crReg != 0x44)
        dprintf("lw: crtcOffset: 0x%04x\n", crtcOffset);

    MGPU_LOOP_START;
    {
        crVal = REG_RDCR((LwU8) crReg, crtcOffset);
        dprintf("lw: crVal = 0x%02x\n", crVal);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// setcr <reg> <val> <optHd>
// - crtcOffset is based off cr44
// - write a cr register
//-----------------------------------------------------
DECLARE_API( setcr )
{
    LwU64 values[2];
    LwU8 crReg = 0;
    LwU8 crVal = 0;
    LwU32 crtcOffset = 0;

    CHECK_INIT(MODE_LIVE);

    // read in the crReg, CrVal, and Head
    if (GetSafeExpressionEx(args, &values[0], &args) &&
        GetSafeExpressionEx(args, &values[1], &args))
    {
        crtcOffset = (LwU8)GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.setcr <crReg> <crVal> <optHd>\n");
        return;
    }

    crReg = (LwU8)values[0];
    crVal = (LwU8)values[1];

    if (crtcOffset == 0)
        crtcOffset = 0x0;
    else
        crtcOffset = 0x2000;

    if (crReg != 0x44)
        dprintf("lw: crtcOffset: 0x%04x\n", crtcOffset);

    MGPU_LOOP_START;
    {
        // write the value
        REG_WRCR(crReg, crVal, crtcOffset);

        // read out the value to make sure it took
        crVal = REG_RDCR(crReg, crtcOffset);
        dprintf("lw: crVal = 0x%02x\n", crVal);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pbinfo
// - get the current pb information
//-----------------------------------------------------
DECLARE_API( pbinfo )
{
    CHECK_INIT(MODE_LIVE);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(HOST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        pFifo[indexGpu].fifoGetPbInfo();
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }
}

//-----------------------------------------------------
// pbdma <pbdmaId>
// - Displays the PBDMA regs corresponding to the specified pbdma unit id
//-----------------------------------------------------
DECLARE_API( pbdma )
{
    LwU64 pbdmaId;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.pbdma <pbdmaId>\n");
        return;
    }

    GetSafeExpressionEx(args, &pbdmaId, &args);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(HOST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        pFifo[indexGpu].fifoDumpPbdmaRegs((LwU32)pbdmaId);
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }
}

//-----------------------------------------------------
// channelram <runlistId>
// - Displays the Channel Ram regs for all active chids of specified runlist
// or in case when a argument is passed, prints all channel ram regs.
//-----------------------------------------------------
DECLARE_API( channelram )
{
    LwU64 runlistId;

    CHECK_INIT(MODE_LIVE);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        if (isVirtualWithSriov() == LW_TRUE)
        {
            setLwwatchAddrType(GUEST_PHYSICAL);
        }
        else
        {
            setLwwatchAddrType(HOST_PHYSICAL);
        }
    }

    if (parseCmd(args, "a", 0, NULL))
    {
        runlistId = -1;
    }
    else
    {
        GetSafeExpressionEx(args, &runlistId, &args);
    }

    MGPU_LOOP_START;
    {
        pFifo[indexGpu].fifoDumpChannelRamRegs((LwS32)runlistId);
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }
}

//------------------------------------------------------------
// runlist <runlistId>
// - gets runlist info of specified runlist or in case when
// "a" or no argument is passed, it prints all active runlists.
//------------------------------------------------------------
DECLARE_API( runlist )
{
    LwU64 runlistId;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "a", 0, NULL) || !GetSafeExpressionEx(args, &runlistId, &args))
    {
        runlistId = RUNLIST_ALL;
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(HOST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        pFifo[indexGpu].fifoDumpRunlist((LwU32)runlistId);
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }

}

//-----------------------------------------------------
// fifoinfo
//
//-----------------------------------------------------
DECLARE_API( fifoinfo )
{
    CHECK_INIT(MODE_LIVE);

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(HOST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        pFifo[indexGpu].fifoGetInfo();
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU, set address
    // type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }

}

#ifndef USERMODE

//-----------------------------------------------------
// pbch <chid> <optOffset> <optLength>
// - dump the pb for the given chid
//-----------------------------------------------------
DECLARE_API( pbch )
{
    BOOL isParse = FALSE;
    LwU64 chid = ~0;
    LwU64 pbOffset = 0;
    LwU32 lengthInBytes = 0;

    CHECK_INIT(MODE_LIVE);

    isParse = parseCmd(args, "p", 0, NULL);

    if (GetSafeExpressionEx(args, &chid, &args))
    {
        if (GetSafeExpressionEx(args, &pbOffset, &args))
        {
            pbOffset = (pbOffset + 3) & ~3ULL;
            lengthInBytes = (LwU32)GetSafeExpression(args);
            lengthInBytes = (lengthInBytes + 3) & ~3ULL;
        }
    }

    MGPU_LOOP_START;
    {
        pFifo[indexGpu].fifoDumpPb((LwU32)chid, (LwU32)pbOffset, lengthInBytes, isParse);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pb [-p] <optOffset> <optLength>
// - dump the pb for the current chid
//-----------------------------------------------------
DECLARE_API( pb )
{
    LwU32 lengthInBytes = 0;
    LwU64 pbOffset = 0;
    BOOL isParse = FALSE;

    CHECK_INIT(MODE_LIVE);

    isParse = parseCmd(args, "p", 0, NULL);

    // read in the location and size to dump
    if (GetSafeExpressionEx(args, &pbOffset, &args))
    {
        pbOffset = (pbOffset + 3) & ~3ULL;
        lengthInBytes = (LwU32)GetSafeExpression(args);
        lengthInBytes = (lengthInBytes + 3) & ~3ULL;
    }

    MGPU_LOOP_START;
    {
        pFifo[indexGpu].fifoDumpPb(LWRRENT_CHANNEL, (LwU32) pbOffset, lengthInBytes, isParse);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// i2c
// - bring up the i2c menu
//-----------------------------------------------------
DECLARE_API( i2c )
{
    CHECK_INIT(MODE_LIVE);

    i2cMenu();
    dprintf("\n");
}

#endif

//---------------------------------------------------------
// msa
// - bring up the msa menu to program MSA values through SW
//---------------------------------------------------------
DECLARE_API( msa )
{
    CHECK_INIT(MODE_LIVE);

    if (IsGM107orLater())
    {
        msaMenu();
    }
    else
    {
        dprintf("This feature only applies to GK110 and later GPUs.");
    }

    dprintf("\n");
}

/*
//-----------------------------------------------------
// i2cinfo
//
//-----------------------------------------------------
DECLARE_API( i2cinfo )
{
    LwU32 isParse = parseCmd(args, "p", 0, NULL);
    LwU32 port = (LwU32) GetSafeExpression(args);

    CHECK_INIT(MODE_LIVE);

    hwI2cGetStatus_G98(port, isParse);
    dprintf("\n");
}
*/

//-----------------------------------------------------
// dsiinfo
//
//-----------------------------------------------------
DECLARE_API( dsiinfo )
{
    LwU32 index = (LwU32)GetSafeExpression(args);

    CHECK_INIT(MODE_LIVE);

    pDisp[indexGpu].dispGetDsiInfo(index);
    dprintf("\n");
}

//-----------------------------------------------------------------------------------------------------
// getGPUInstanceIndex
// - Using source symbol to find the corresponding GPU instance index from from the given BAR0 address.
// - Return the GPU instance index
//-----------------------------------------------------------------------------------------------------
static BOOL getGPUInstanceIndex(LwU64 targetAddr, LwU8 *pInstanceIdx)
{
    LwU8  index = 0;
    LwU64 bar0 = 0;
    char  cmdStr[70];

    do
    {
        sprintf(cmdStr, "@@(lwlddmkm!LwDBPtr_Table[%d]->halHwInfo.pGpu->busInfo.gpuPhysAddr)", index);

        bar0 = GetSafeExpression(cmdStr);
        if (targetAddr == bar0)
        {
            *pInstanceIdx = index;
            return TRUE;
        }

        index++;

    } while (bar0);

    return FALSE;
}

//-----------------------------------------------------
// dcb
// - dump the dcb
//-----------------------------------------------------
DECLARE_API( dcb )
{
    char *params;
    char  cmdStr[70];
    LwU8  index = 0;
    LwU32 flags = 0;
    LwU64 addr = 0;
    LwU64 mode = 0;

    CHECK_INIT(MODE_LIVE);

    dprintf("lw: Usage: !lw.dcb [-f <flags>] [-m <mode> <data>] \n\n");

    if (parseCmd(args, "f", 1, &params))
    {
        flags = (LwU32) GetSafeExpression(params);

        if (flags >= 0x40)
        {
            dcbusage();
            return;
        }
    }

    // -m 0: Dump DCB from vbios image located at bar0+0x7e000.
    // -m 1: Dump image using source symbols to get from lwrrently selected GPU's pVbios->pImage.
    // -m 2 address: Dump DCB from vbios image located at address.
    if (parseCmd(args, "m", 1, &params))
    {
        GetSafeExpressionEx(params, &mode, &params);
    }

    if ((mode == 0) || (mode == 1))
    {
        MGPU_LOOP_START;
        {
            if (mode == 1)
            {
                if (getGPUInstanceIndex(lwBar0, &index))
                {
                    sprintf(cmdStr, "@@(lwlddmkm!LwDBPtr_Table[%d]->halHwInfo.pGpu->pVbios->pImage)", index);

                    addr = GetSafeExpression(cmdStr);
                    if (!addr)
                    {
                        dprintf("lw: Cannot find the image from source symbol LwDBPtr_Table[%d]\n", index);
                        continue;
                    }
                }
                else
                {
                    dprintf("lw: Cannot find the GPU instance index for Bar0(" PhysAddr_FMT ") from the source symbol\n", lwBar0);
                }
            }

            getDCB(flags, addr);
        }
        MGPU_LOOP_END;
    }
    else
    {
        if (mode == 2)
        {
            // After parseCmd, params doesn't keep address information but orginal args has.
            GetSafeExpressionEx(args, &addr, &params);
            getDCB(flags, addr);
        }
        else
        {
            dprintf("lw: Specified mode doesn't exist\n");
        }
    }
}

void slivberrmsg (void)
{
    dprintf("Error! No SliBars defined!\n");
    dprintf("lw: Usage: !lw.slivb <multiGpuBar0> <multiGpuBar0> ...\n");
    dprintf("           !lw.slivb r <reg offset>\n");
    dprintf("           !lw.slivb w <reg offset> <data>\n");
    dprintf("           !lw.slivb m <reg offset> <high bit> <low bit> <data>\n");
}

//-----------------------------------------------------
// slivb
// - dump the SLI Video Bridge affected registers
//-----------------------------------------------------
DECLARE_API( slivb )
{
    LwU32 reg, data, high, low, mod;
    LwU32 UseCR = 0;
    LwU64 data64;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        slivberrmsg();
        return;
    }

    if ( (args[0] == 'R') ||
         (args[0] == 'r') )
    {
        if (multiGpuBar0[0] == 0)
        {
            slivberrmsg();
        }

        if ( (args[1] == 'C') ||
             (args[1] == 'c') )
        {
            UseCR = 1;
            args = &args[2];
        }
        else
        {
            args = &args[1];
        }

        GetSafeExpressionEx(args, &data64, &args);
        reg  = (LwU32) data64;

        MGPU_LOOP_START;
        {
            if (UseCR)
            {
                dprintf(PhysAddr_FMT ": Cr%x 0x%02x\n", lwBar0, reg, REG_RDCR((LwU8)reg, 0));
            }
            else
            {
                dprintf(PhysAddr_FMT ": 0x%08x\n", lwBar0 | reg, GPU_REG_RD32(reg));
            }
        }
        MGPU_LOOP_END;
    }
    else if ( (args[0] == 'W') ||
              (args[0] == 'w') )
    {
        if (multiGpuBar0[0] == 0)
        {
            slivberrmsg();
        }

        if ( (args[1] == 'C') ||
             (args[1] == 'c') )
        {
            UseCR = 1;
            args = &args[2];
        }
        else
        {
            args = &args[1];
        }

        GetSafeExpressionEx(args, &data64, &args);
        reg  = (LwU32) data64;
        GetSafeExpressionEx(args, &data64, &args);
        data  = (LwU32) data64;

        MGPU_LOOP_START;
        {
            if (UseCR)
            {
                REG_WRCR((LwU8)reg, (LwU8)data, 0);

                dprintf(PhysAddr_FMT ": Cr%x 0x%02x\n", lwBar0, reg, REG_RDCR((LwU8)reg, 0));
            }
            else
            {
                GPU_REG_WR32(reg, data);

                dprintf(PhysAddr_FMT ": 0x%08x\n", lwBar0 | reg, GPU_REG_RD32(reg));
            }
        }
        MGPU_LOOP_END;
    }
    else if ( (args[0] == 'M') ||
              (args[0] == 'm') )
    {
        if (multiGpuBar0[0] == 0)
        {
            slivberrmsg();
        }

        if ( (args[1] == 'C') ||
             (args[1] == 'c') )
        {
            UseCR = 1;
            args = &args[2];
        }
        else
        {
            args = &args[1];
        }

        GetSafeExpressionEx(args, &data64, &args);
        reg  = (LwU32) data64;
        GetSafeExpressionEx(args, &data64, &args);
        high = (LwU32) data64;
        GetSafeExpressionEx(args, &data64, &args);
        low  = (LwU32) data64;
        GetSafeExpressionEx(args, &data64, &args);
        mod  = (LwU32) data64;

        dprintf("    Mask: 0x%08x\n", ((0xFFFFFFFF>>(31-(high-low)))<<low));

        MGPU_LOOP_START;
        {
            if (UseCR)
            {
                data = (LwU32)REG_RDCR((LwU8)reg, 0);
                data &= ~((0xFFFFFFFF>>(31-(high-low)))<<low);
                mod  &= (0xFFFFFFFF>>(31-(high-low)));
                data |= (mod<<low);

                REG_WRCR((LwU8)reg, (LwU8)data, 0);

                dprintf(PhysAddr_FMT ": Cr%x 0x%02x\n", lwBar0, reg, REG_RDCR((LwU8)reg, 0));
            }
            else
            {
                data = GPU_REG_RD32(reg);
                data &= ~((0xFFFFFFFF>>(31-(high-low)))<<low);
                mod  &= (0xFFFFFFFF>>(31-(high-low)));
                data |= (mod<<low);
                GPU_REG_WR32(reg, data);

                dprintf(PhysAddr_FMT ": 0x%08x\n", lwBar0 | reg, GPU_REG_RD32(reg));
            }
        }
        MGPU_LOOP_END;
    }

}

//----------------------------------------------------------------------------------
// diag [-grIdx] <grIdx>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - reports the general status of the GPU
//----------------------------------------------------------------------------------
DECLARE_API( diag )
{
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    LWWATCHDIAGSTRUCT diagStruct;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    //With SMC mode enabled, a BAR0 window must be set
    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    diagFillStruct(&diagStruct);

    pClk[indexGpu].clkGetClocks();
    dprintf("\n");

    diagMaster(&diagStruct);
    dprintf("\n");

    diagFifo(&diagStruct);
    dprintf("\n");

    diagGraphics(&diagStruct, grIdx);
    dprintf("\n");
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

/**
 * @brief      Parse arguments to get channel info
 *
 * @param      args       The arguments
 * @param      pChId      parsed channel ID
 *
 * @return     LW_OK on success, ERROR code otherwise
 */
static LW_STATUS fifoctxParseCh(char **args, ChannelId *pChId)
{
    char   *params;
    LwU32   numChannels;
    LwU64   id = 0;
    LW_STATUS status = LW_OK;

    if (pChId == NULL)
        return LW_ERR_ILWALID_ARGUMENT;

    // defaults
    pChId->bRunListValid = LW_FALSE;
    pChId->bChramPriBaseValid = LW_FALSE;
    pChId->id = 0;

    // -ch <engine_string> <chid> OR
    // -ch <type_enum> <instance_id> <chid>
    if (parseCmd(*args, "ch", 1, &params))
    {
        if ((params[0] >= '0') && (params[0] <= '9'))
        {
            // params is a number, so assume (type, inst) combination
            LwU64 typeEnum;
            LwU64 instId;

            GetSafeExpressionEx(params, &typeEnum, &params);
            GetSafeExpressionEx(*args, &instId, args);

            dprintf("lw: engine typeEnum %lld, instId %lld\n", typeEnum, instId);
            if ((typeEnum > LW_U32_MAX) || (instId > LW_U32_MAX))
            {
                dprintf("ERROR: typeEnum and/or instId out of 32 bit range.\n");
                return LW_ERR_ILWALID_ARGUMENT;
            }

            status = pFifo[indexGpu].fifoXlateFromDevTypeAndInstId((LwU32)typeEnum, (LwU32)instId, ENGINE_INFO_TYPE_RUNLIST, &pChId->runlistId);
            if (status != LW_OK)
            {
                dprintf("ERROR: failed to fetch engine information from typeEnum, instId\n");
                return LW_ERR_ILWALID_ARGUMENT;
            }

            pChId->bRunListValid = LW_TRUE;
        }
        else
        {
            // params is a string, so assuming engine name string
            status = pFifo[indexGpu].fifoXlateFromEngineString(params, ENGINE_INFO_TYPE_RUNLIST, &pChId->runlistId);
            if (status != LW_OK)
            {
                dprintf("Engine name %s does not map to a valid engine\n", params);
                return LW_ERR_ILWALID_ARGUMENT;
            }

            dprintf("lw: engine name %s\n", params);
            pChId->bRunListValid = LW_TRUE;
        }

        GetSafeExpressionEx(*args, &id, args);
        dprintf("lw: ChId 0x%llx\n", id);
        if (id > LW_U32_MAX)
        {
            dprintf("ERROR: chId out of 32 bit range\n");
            return LW_ERR_ILWALID_ARGUMENT;
        }

        // Save the CHID value
        pChId->id = (LwU32)id;
        //
        // Validate channel value
        // (if possible, i.e. fifoGetNumChannels implemented)
        //
        numChannels = (LwU32)pFifo[indexGpu].fifoGetNumChannels(pChId->runlistId);
        if ((numChannels > 0) && (pChId->id >= numChannels))
        {
            dprintf("ERROR: ChId is too large. supported numChannels = 0x%x\n", numChannels);
            return LW_ERR_ILWALID_ARGUMENT;
        }

        dprintf("lw: Host runlistId %d\n", pChId->runlistId);
    }
    else
    {
        LwU64 id;
        GetSafeExpressionEx(*args, &id, args);
        pChId->id = (LwU32)id;

        //
        // Validate channel value
        // (if possible, i.e. fifoGetNumChannels implemented)
        //
        numChannels = (LwU32)pFifo[indexGpu].fifoGetNumChannels(pChId->runlistId);
        if ((numChannels > 0) && (pChId->id >= numChannels))
        {
            dprintf("ERROR: ChId is too large. supported numChannels = 0x%x\n", numChannels);
            return LW_ERR_ILWALID_ARGUMENT;
        }

        dprintf("lw: ChId 0x%x\n", pChId->id);
    }

    if (IsGA100orLater() && !pChId->bRunListValid)
    {
        dprintf("Runlist id is required on Ampere+ and could not be derived.\b");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    return LW_OK;
}

//-----------------------------------------------------
// Usage:
// fifoctx -ch <engine_string> <chid>
// fifoctx -ch <type_enum> <instance_id> <chid>
// Deprecated Ampere +:
// fifoctx <chId>
// - dump the fifo ctx for a given chId / runlist
//-----------------------------------------------------
DECLARE_API( fifoctx )
{
    ChannelId channelId;

    CHECK_INIT(MODE_LIVE);

    if (fifoctxParseCh(&args, &channelId) != LW_OK)
    {
        return;
    }

    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(HOST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        pInstmem[indexGpu].instmemDumpFifoCtx(&channelId);
        dprintf("\n");
    }
    MGPU_LOOP_END;

    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }
}

//-----------------------------------------------------
// insttochid <inst> [target]
// - Prints out the chid for the given engine instance and target
//-----------------------------------------------------
DECLARE_API( insttochid )
{
    LwU64 inst = 0;
    LwU32 target = 0;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &inst, &args))
        target = (LwU32) GetSafeExpression(args);

    pInstmem[indexGpu].instmemGetChidFromInst((LwU32)inst, target);
    dprintf("\n");
}

//-----------------------------------------------------
// veidinfo <chId><veid>
// - dump the veid info for a given chId
//-----------------------------------------------------
DECLARE_API( veidinfo )
{
    LwU64     arg1;
    LwU32     veid;
    ChannelId channelId;

    CHECK_INIT(MODE_LIVE);

    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;

    if (GetSafeExpressionEx(args, &arg1, &args))
    {
        veid = (LwU32) GetSafeExpression(args);
        channelId.id = (LwU32) arg1;
    }
    else
    {
        dprintf("lw: Usage: !lw.veidinfo <chid> <veid>\n\n");
        return;
    }

    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(HOST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        pInstmem[indexGpu].instmemGetSubctxPDB(&channelId, veid);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// grctx <chid>
//
//-----------------------------------------------------
DECLARE_API( grctx )
{
    LwU32 chId;

    CHECK_INIT(MODE_LIVE);

    chId = (LwU32) GetSafeExpression(args);

    pGr[indexGpu].grGetChannelCtxInfo(chId);
    dprintf("\n");
}

//-----------------------------------------------------
// subch <subch>
//
//-----------------------------------------------------
DECLARE_API( subch )
{
    LwU32 subCh;

    CHECK_INIT(MODE_LIVE);

    subCh = (LwU32)GetSafeExpression(args);
    if (subCh >= 0xffff)
    {
        pGr[indexGpu].grDumpSubCh();
    }
    else
    {
        pGr[indexGpu].grGetSubChInfo((LwU32)subCh);
    }
    dprintf("\n");
}

//----------------------------------------------------------------------------------
// gr [-p] [-grIdx] <grIdx>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - dump graphics fifo
//----------------------------------------------------------------------------------
DECLARE_API( gr )
{
    BOOL isParse = FALSE;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    CHECK_INIT(MODE_LIVE);

    isParse = parseCmd(args, "p", 0, NULL);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    //With SMC mode enabled, a BAR0 window must be set
    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    pGr[indexGpu].grDumpFifo(isParse, grIdx);
    dprintf("\n");
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------
// grinfo
// - dump graphics info
//-----------------------------------------------------
DECLARE_API( grinfo )
{

    CHECK_INIT(MODE_LIVE);

    pGr[indexGpu].grGetInfo();
    dprintf("\n");
}

//-----------------------------------------------------
// tiling
// - get graphics tiling info
//-----------------------------------------------------
DECLARE_API( tiling )
{
    CHECK_INIT(MODE_LIVE);

    pGr[indexGpu].grGetTilingInfoLww();
    dprintf("\n");
}

//-----------------------------------------------------------------------------
// zlwll [-grIdx] <grIdx>
// <grIdx> - Graphics index required to access registers when SMC is enabled
// - get graphics zlwll info
//-----------------------------------------------------------------------------
DECLARE_API( zlwll )
{
    char *param;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    pGr[indexGpu].grGetZlwllInfoLww( grIdx );
    dprintf("\n");
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);

}

//-----------------------------------------------------
// zlwllram
//
// ----------------------------------------------------
DECLARE_API( zlwllram )
{
    LwU64 select;
    LwU64 size;
    LwU64 addr;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &select, &args) &&
        GetSafeExpressionEx(args, &addr, &args) &&
        ((size = GetSafeExpression(args)) != 0))
    {
        size = (size + 3) & ~3ULL;
        pGr[indexGpu].grDumpZlwllRam((LwU32)addr, (LwU32)size, (LwU32)select);
    }
    else
    {
        dprintf("lw: Usage: !lw.zcram <select> <addr> <size>\n");
    }
}

//-----------------------------------------------------------------------------------
// grstatus [-grIdx] <grIdx>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - get graphics status info
//-----------------------------------------------------------------------------------
DECLARE_API( grstatus )
{
    LwBool isFullPrint = FALSE;
    LwBool bForceEnable = FALSE;
    LwBool isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;
    CHECK_INIT(MODE_LIVE);

    isFullPrint  = parseCmd(args, "a", 0, NULL);
    bForceEnable = parseCmd(args, "f", 0, NULL);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    //With SMC mode enabled, a BAR0 window must be set
    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchMode(LW_TRUE);
        setLwwatchAddrType(HOST_PHYSICAL);
    }

    MGPU_LOOP_START;
    {
        pGr[indexGpu].grGetStatus(isFullPrint, bForceEnable, grIdx);
        dprintf("\n");
    }
    MGPU_LOOP_END;

    //
    // turn off LwWatch mode and address type if we're running a VGPU,
    // set address type to invalid for good measure
    //
    if (isVirtual() == LW_TRUE)
    {
        setLwwatchAddrType(INVALID);
        setLwwatchMode(LW_FALSE);
    }

    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------------------------------------
// grwarppc <repeat> [<gpcId> <tpcId>] [-grIdx] <grIdx>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
//-----------------------------------------------------------------------------------
DECLARE_API( grwarppc )
{
    BOOL targetSet = FALSE;
    LwU64 targetGpcId = 0;
    LwU64 targetTpcId = 0;
    LwU32 gpcId = 0;
    LwU32 tpcId = 0;
    LwU32 nGPC;
    LwU32 nTPC;
    LwU64 repeat = 8;
    char *param;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    if (GetSafeExpressionEx(args, &repeat, &args))
    {
        if (GetSafeExpressionEx(args, &targetGpcId, &args))
        {
            if (GetSafeExpressionEx(args, &targetTpcId, &args)) {
                targetSet = TRUE;
            } else {
                goto bad_usage;
            }

        }
        else
        {
            targetSet = FALSE;
        }
    }

    MGPU_LOOP_START;

    nGPC = pGr[indexGpu].grGetNumActiveGpc( grIdx );

    for (gpcId = 0; gpcId < nGPC; gpcId += 1)
    {
        nTPC = pGr[indexGpu].grGetNumTpcForGpc(gpcId, grIdx);

        for (tpcId = 0; tpcId < nTPC; tpcId += 1)
        {
            if (targetSet)
            {
                if (gpcId != targetGpcId || tpcId != targetTpcId)
                {
                    continue;
                }
            }
            dprintf("%s GPC/TPC %d/%d Warp PCs\n", GpuArchitecture(), gpcId, tpcId);
            pGr[indexGpu].grDumpWarpPc(gpcId, tpcId, (LwU32)repeat, LW_TRUE);
        }
    }
    dprintf("\n");
    MGPU_LOOP_END;

    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
    return;

bad_usage:
    dprintf("grwarppc <repeat> [<gpcId> <tpcId>]\n");
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------
// ssclass <classNum> <optDis>
// - enable single step for the given class
// - optDis of 1 means disable single step
//-----------------------------------------------------
DECLARE_API( ssclass )
{
    LwU64 classNum;
    LwU32 optDis = 0;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &classNum, &args))
    {
        optDis = (LwU32) GetSafeExpression(args);
    }

    pGr[indexGpu].grEnableSingleStep((LwU32) classNum, !optDis);
    dprintf("\n");
}

//-----------------------------------------------------
// limiterror
// - Analyze a Gr limit error
//-----------------------------------------------------
DECLARE_API( limiterror )
{
    CHECK_INIT(MODE_LIVE);

    pGr[indexGpu].grAnalyzeLimitError();
    dprintf("\n");
}

//-----------------------------------------------------
// surface
// - Prints out tiling and surface regs
//-----------------------------------------------------
DECLARE_API( surface )
{
    CHECK_INIT(MODE_LIVE);

    pGr[indexGpu].grPrintSurfaceRegs();
    dprintf("\n");
}

//-----------------------------------------------------
// launchcheck
//
//-----------------------------------------------------
DECLARE_API( launchcheck )
{
    CHECK_INIT(MODE_LIVE);

    pGr[indexGpu].grLauchCheck();
    dprintf("\n");
}

//-----------------------------------------------------
// perfmon
//
//-----------------------------------------------------
DECLARE_API( perfmon )
{
    CHECK_INIT(MODE_LIVE);

    pGr[indexGpu].grGetPerfMonitorInfo();
    dprintf("\n");
}

//-----------------------------------------------------
// MSDEC dmem (includes MSPPP,MSVLD, and MSPDEC)
//
//-----------------------------------------------------
DECLARE_API( msdec )
{
    LwU64 idec=0;
    LwU64 iDmemSize=0;
    LwU64 iImemSize=0;
    LwU64 iChId = 0;
    BOOL isParse = FALSE;
    BOOL isPrintFCB = FALSE;

    CHECK_INIT(MODE_LIVE);

    // flow control buffer is to be printed out if the -f option has
    // been selected
    isPrintFCB = parseCmd(args, "f", 0, NULL);
    if (isPrintFCB)
    {
        if (!GetSafeExpressionEx(args, &iChId, &args))
        {
            // if channel id has not been specified, let us assume that it
            // is 3 generally the case for XP
            iChId = 3;
        }
    }
    else
    {
        // methods and data are to be parsed and  printed if the -p option
        // has been selected
        isParse = parseCmd(args, "p", 0, NULL);
    }

    if (GetSafeExpressionEx(args, &idec, &args))
    {
        if (GetSafeExpressionEx(args, &iDmemSize, &args))
        {
            iImemSize = GetSafeExpression(args);
        }
    }

    pMsdec[indexGpu].msdecGetInfo((LwU32)idec, (LwU32)iDmemSize, (LwU32)iImemSize, (BOOL)isParse, (LwU32)iChId, (BOOL)isPrintFCB);
    dprintf("\n");
}
//-----------------------------------------------------
// mcinfo
//
//-----------------------------------------------------
DECLARE_API( mcinfo )
{
    CHECK_INIT(MODE_LIVE);

    pMc[indexGpu].mcGetInfo();
    dprintf("\n");
}

#if !defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX)
//-----------------------------------------------------
// heap <optHeapAddr>
// - dumps the heap
//-----------------------------------------------------
DECLARE_API( heap )
{
    LwU64 heapAddr;
    LwU64 heapAddr0;
    LwU32 SortOption = 0, OwnerFilter = 0, i;
    char *param;

    if (parseCmd(args, "s", 1, &param))
        SortOption = (LwU32) GetSafeExpression(param);
    if (parseCmd(args, "o", 1, &param))
    {
        for (i = 0; i < 4; i++)
        {
            if (param[i] == '\0')
                break;
            OwnerFilter |= (param[i] << (i*8));
        }
    }

    heapAddr = (LwU64) GetSafeExpression(args);

    heapAddr0 = GetSafeExpression(LW_VAR("LwDBPtr_Table[0]->halHwInfo.pGpu->pFb->pHeap"));
    if (heapAddr0)          // It is R60 version.
    {
        if (!heapAddr)
            heapAddr = heapAddr0;

        dprintf("lw: heapAddr:          " LwU64_FMT "\n", heapAddr);
        dumpHeap_R60((void *)(LwUPtr)heapAddr, SortOption, OwnerFilter);
    }
    else
    {
        dprintf("Failed to print heap, your driver probably doesn't match.");
    }

    dprintf("\n");
}

//-----------------------------------------------------
// pma
// - dumps PMA state
//-----------------------------------------------------
DECLARE_API( pma )
{
    LwU64 pmaAddr;

    pmaAddr = GetSafeExpression(LW_VAR("&LwDBPtr_Table[0]->halHwInfo.pGpu->pFb->pHeap->pmaObject"));
    if (pmaAddr)
    {
        dprintf("lw: pmaAddr:          " LwU64_FMT "\n", pmaAddr);
        dumpPMA((void *)(LwUPtr)pmaAddr);
    }
    else
    {
        dprintf("Failed to print PMA state, your driver probably doesn't match.");
    }

    dprintf("\n");
}
#endif

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(USERMODE) && \
!defined(CLIENT_SIDE_RESMAN)

//-----------------------------------------------------
// Extension to dump stacktrace
//-----------------------------------------------------
DECLARE_API ( stack )
{
    EXTSTACKTRACE64 stk[20];
    LwU32 frames, i;
    CHAR Buffer[256];
    LwU64 displacement;

    // Get stacktrace for current thread
    frames = StackTrace( 0, 0, 0, stk, 20 );

    if (!frames)
    {
        dprintf("Stacktrace failed\n");
    }

    for (i = 0; i < frames; i++)
    {
        if (i == 0)
        {
            dprintf( "ChildEBP RetAddr  Args to Child\n" );
        }

        Buffer[0] = '!';
        GetSymbol(stk[i].ProgramCounter, (PUCHAR)Buffer, &displacement);

        dprintf( "%08p %08p %08p %08p %08p %s",
                 stk[i].FramePointer,
                 stk[i].ReturnAddress,
                 stk[i].Args[0],
                 stk[i].Args[1],
                 stk[i].Args[2],
                 Buffer
                 );

        if (displacement)
        {
            dprintf( "+0x%p", displacement );
        }

        dprintf( "\n" );
    }
}

#endif // !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(...)

//-----------------------------------------------------
// dispParseArgs
// - Yet another arguments packer
// Rationale: display commands take arguments in a different
//            way from the old commands, and the existing
//            functions do not nicely fit into our needs.
//-----------------------------------------------------
#define DISP_MAXARGV (44)
#define DISP_MAXTOK (512)

static int dArgc;
static char dArgv[DISP_MAXARGV][DISP_MAXTOK];

static void dispParseArgs(char *line);
#define PA_DBG (0)
static void dispParseArgs(char *line)
{
    char c, buf[DISP_MAXTOK], *pBuf;

    dArgc = 0;
    if (!line || *line == '\0') {
        return;
    }

    if (PA_DBG) {
        dprintf("::%s::\n", line);
    }

    // skip whitespaces
    while (*line == ' ' || *line == '\t')
        line++;
    // copy arguments until it hits the end of string
    pBuf = buf;
    while (dArgc < DISP_MAXARGV) {
        c = *line;
        // assume these are the only delimeters
        if (c != ' ' && c != '\t' && c != '\0') {
            *pBuf++ = c;
            line++;
        }else {
            *pBuf = '\0';
            strncpy(dArgv[dArgc], buf, DISP_MAXTOK);
            dArgv[dArgc][DISP_MAXTOK-1] = '\0';
            dArgc++;
            while (*line == ' ' || *line == '\t')
                line++;

            if (*line == '\0')
                break;
            pBuf = buf;
        }
    }

    // DEBUG
    if (PA_DBG) {
        int i;
        dprintf("dArgc = %d" , dArgc);
        for (i = 0; i < dArgc ; i++)
            dprintf(", dArgv[%d] = %s",i,dArgv[i]);
        dprintf("\n");
    }
}

static void dispDchnExceptPrintUsage()
{
    dprintf("lw: Usage: !lw.dchnexcept <chName> -n<errorNum>\n");
    dprintf("expected chName: win/core\n");
    dprintf("expected errorNum: decimal input\n");
}

DECLARE_API ( dchnexcept )
{
    char *chName = NULL;
    LwBool isCore    = LW_FALSE;
    LwBool isWindow  = LW_FALSE;
    LwBool isNumProv = LW_FALSE;
    LwU32 errNum = 0;
    LwU32 temp;
    LwU32 i;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc > 0  && dArgc < 3)
    {
        chName = dArgv[0];
        for (i=0; i < strlen(dArgv[0]); i++)
        {
            *(chName + i) = tolower(*(chName + i));
        }

        if (strcmp("core", chName) == 0)
        {
            isCore = LW_TRUE;
        }
        else if (strcmp("win", chName) == 0)
        {
            isWindow = LW_TRUE;
        }
        else
        {
            dispDchnExceptPrintUsage();
            return;
        }

        if (dArgc > 1)
        {
            if (sscanf(dArgv[1], "-n%d", &temp))
            {
                errNum = temp;
                isNumProv = LW_TRUE;
            }
            else
            {
                dispDchnExceptPrintUsage();
                return;
            }
        }
    }
    else if (dArgc == 0)
    {
        isCore   = LW_TRUE;
        isWindow = LW_TRUE;
    }
    else
    {
        dispDchnExceptPrintUsage();
        return;
    }

    MGPU_LOOP_START;
    {
        if (isCore && isWindow && !isNumProv)
        {
            //print for all pending Exceptions.
            pDisp[indexGpu].dispDumpPendingExcHls(isCore, isWindow);
            pDisp[indexGpu].dispDumpPendingExcHls(isCore, isWindow);
        }
        else if ((isCore || isWindow) && !isNumProv)
        {
            //
            //print all pending exceptions for the channel type provided.
            //at this point only core or window will be true not both.
            //
            pDisp[indexGpu].dispDumpPendingExcHls(isCore, isWindow);
        }
        else if ((isCore || isWindow) && isNumProv)
        {
            // print only the requested errNum from the requested channel type.
            pDisp[indexGpu].dispParseHls(isCore, errNum);
        }
    }
    MGPU_LOOP_END;

    return;
}

//-----------------------------------------------------
// dchnstate <chName> [-h<hd>/-w<wd>]
// - dumps out each channel's state.
//-----------------------------------------------------
DECLARE_API ( dchnstate )
{
    char *chName = NULL;
    LwS32 chNum = 0;
    LwS32 headNum = -1;
    LwS32 winNum = -1;
    LwU32 temp;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    dprintf("----------------------------------------------------------------------------------------------\n");
    dprintf("Ch#\tName\tHead\t%8s%5s", "", "STATE");
    dprintf("\t%13s", "SUPERVISOR");
    dprintf("\t%11s \t%2s", "EXCEPTION", "");
    dprintf("%6s%2s\n", "CHNCTL", "");
    dprintf("----------------------------------------------------------------------------------------------\n");

    if ( dArgc > 0 )
    {
        chName = dArgv[0];
        if ( dArgc > 1 )
        {
            if (sscanf(dArgv[1], "-h%d", &temp))
            {
                headNum = temp;
            }
            else if (sscanf(dArgv[1], "-w%d", &temp))
            {
                winNum = temp;
                headNum = winNum;
            }
            else
            {
                dprintf("lw: Usage: !lw.dchnstate <chName> [-h<hd>/-w<wd>] \n");
                return ;
            }

            if (!strcmp("win", chName) || !strcmp("winim", chName))
            {
                if (winNum == -1)
                {
                    dprintf("lw: Usage: !lw.dchnstate <win/winim> [-w<wd>] \n");
                    return ;
                }
            }
        }

        MGPU_LOOP_START;
        {
            if ( (chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum)) != -1)
            {
                int i = chNum;
                int howmany = chNum + 1;
                // if dArgc == 1, print all heads
                if (dArgc == 1)
                {
                    howmany = pDisp[indexGpu].dispGetMaxChan();
                }
                for (i = chNum; i < howmany ; i++)
                {
                    pDisp[indexGpu].dispPrintChanState(i);
                }
            }
        }
        MGPU_LOOP_END;
    }
    else
    {
        LwU32 i,k;
        MGPU_LOOP_START;
        {
            k = pDisp[indexGpu].dispGetMaxChan();
            for (i = 0; i < k; i++)
            {
                pDisp[indexGpu].dispPrintChanState(i);
            }
        }
        MGPU_LOOP_END;
    }
}

//-----------------------------------------------------
// dchnstate <chName> [-h<hd>/-w<wd>]
// - dumps out each channel's state.
//-----------------------------------------------------
DECLARE_API( dchnnum )
{
    char *chName = NULL;
    LwU32 headNum = -1;
    LwU32 winNum = -1;
    LwS32 ret;
    LwU32 temp;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    if ( dArgc > 0 )
    {
        chName = dArgv[0];

        if (dArgc > 1)
        {
            if (sscanf(dArgv[1], "-h%d", &temp))
            {
                headNum = temp;
            }
            else if (sscanf(dArgv[1], "-w%d", &temp))
            {
                winNum = temp;
                headNum = winNum;
            }
            else
            {
                dprintf("lw: Usage: !lw.dchnnum <chName> [-h<hd>/-w<wd>] \n");
                return ;
            }

            if (!strcmp("win", chName) || !strcmp("winim", chName))
            {
                if (winNum == -1)
                {
                    dprintf("lw: Usage: !lw.dchnnum <win/winim> [-w<wd>] \n");
                    return ;
                }
            }
        }

        MGPU_LOOP_START;
        {
            if ((ret = pDisp[indexGpu].dispGetChanNum(chName, headNum)) != -1)
                dprintf("Channel Number : %d\n", ret);
        }
        MGPU_LOOP_END;
    }
    else
    {
        dprintf("lw: Usage: !lw.dchnnum <chName> [-h<hd>/-w<wd>] \n");
    }
}

//-----------------------------------------------------
// dchnname <chNum>
// - dumps out channel number name.
//-----------------------------------------------------
DECLARE_API( dchnname )
{
    int channelNum;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    if (!dArgc || (sscanf(dArgv[0], "%d", &channelNum) != 1))
    {
        dprintf("lw: Usage: !lw.dchnname <chNum>\n");
        return;
    }
    MGPU_LOOP_START;
    {
        pDisp[indexGpu].dispPrintChanName(channelNum);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dchnmstate <chanName> [-h<hd>/-w<wd>]
//
//-----------------------------------------------------
DECLARE_API( dchnmstate )
{
    char *chName = NULL;
    LwS32 headNum = -1;
    LwS32 winNum = -1;
    int argNum, temp;
    BOOL printHeadless = TRUE;
    BOOL headlessExplicit = FALSE;
    BOOL printRegsWithoutEquivMethod = FALSE;

    LwU32 minChan = 0, maxChan = 0;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    for (argNum = 0; argNum < dArgc; ++argNum)
    {
        if (sscanf(dArgv[argNum], "-h%d", &temp))
        {
            headNum = temp;
        }
        else if (sscanf(dArgv[argNum], "-w%d", &temp))
        {
            winNum = temp;
        }
        else if (strcmp(dArgv[argNum], "-noheadless") == 0)
        {
            printHeadless = FALSE;
            headlessExplicit = TRUE;
        }
        else if (strcmp(dArgv[argNum], "-headless") == 0)
        {
            printHeadless = TRUE;
            headlessExplicit = TRUE;
        }
        else if (strcmp(dArgv[argNum], "-nonmethods") == 0)
        {
            printRegsWithoutEquivMethod = TRUE;
        }
        else if (pDisp[indexGpu].dispGetChanNum(dArgv[argNum], 0) != -1)
        {
            chName = dArgv[argNum];
        }
        else
        {
            dprintf("Bad argument %s\n", dArgv[argNum]);
            dprintf("lw: Usage: !lw.dchnmstate [chName] [-h<hd>/-w<wd] [-[no]headless] [-nonmethods]\n");
            return;
        }
    }

    if (!headlessExplicit)
    {
        // If not explicitly requested, show headless only if no head or window specified
        printHeadless = headNum < 0 && winNum < 0;
    }

    pDisp[indexGpu].dispDumpChannelState(chName, headNum, winNum, printHeadless, printRegsWithoutEquivMethod);
}

//-----------------------------------------------------
// dchlwal <chName> <head/win/sor number> <method offset> [-assy/-armed]
// Dumps display channel method value by method offset in class file(clc57d.h)
//   Example:-
//         1. To print core channel HEAD_SET_PROCAMP value for head 0
//           > dchlwal core 0 0x00002000
//             0x0
//         2. To print window channel SET_SIZE value for window 2
//           > dchlwal win 2 0x00000224
//             0x10e01e00
// -----------------------------------------------------
DECLARE_API( dchlwal )
{
    const char *errorStr;
    int i;
    BOOL foundRecognizedChannel;
    const char * const recognisedChannelNames[] = {"core", "win", "winimm", "base", "ovly"};
    char *chName;
    LwU32 chNum;
    char *headWinSorStr;
    LwU32 headWinSorNum;
    char *offsetStr;
    LwU32 offset;
    BOOL isArmed;
    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    errorStr = "lw: Usage: !lw.dchlwal <chName> <head/win/sor number> <method offset>  [-assy/-armed]. Default prints armed state\n"
               "            chName can either be core, win, winimm, base, ovly.\n"
               "            Example:-\n"
               "              1. To print core channel HEAD_SET_PROCAMP value for head 0\n"
               "                !lw.dchlwal core 0 0x00002000\n\n"
               "              2. To print window channel SET_SIZE value for window 2\n"
               "                !lw.dchlwal win 2 0x00000224\n";
    if (!(dArgc == 3 || dArgc == 4))
    {
        printf("Wrong number of arguments %d\n", dArgc);
        dprintf(errorStr);
        return;
    }

    chName = dArgv[0];
    // Validate channel name
    foundRecognizedChannel = FALSE;
    for (i = 0; i < LW_ARRAY_ELEMENTS(recognisedChannelNames); i++)
    {
        if (strcmp(chName, recognisedChannelNames[i]) == 0)
        {
            foundRecognizedChannel = TRUE;
            break;
        }
    }
    if (!foundRecognizedChannel)
    {
        dprintf("Unrecognized channel name %s\n", chName);
        dprintf(errorStr);
        return;
    }

    headWinSorStr = dArgv[1];
    headWinSorNum = 0;
    if (!(sscanf(headWinSorStr, "%d", &headWinSorNum) == 1))
    {
        dprintf("Unrecognized address %s", headWinSorStr);
        dprintf(errorStr);
        return;
    }

    offsetStr = dArgv[2];
    // Validate addr. colwert offsetStr to offset(LwU32)
    offset = 0;
    if (!(sscanf(offsetStr, "0x%x", &offset) == 1))
    {
        dprintf("Unrecognized address %s", offsetStr);
        dprintf(errorStr);
        return;
    }

    isArmed = TRUE;
    if (dArgc == 4)
    {
        if (strcmp(dArgv[3], "-armed") == 0)
        {
            isArmed = TRUE;
        }
        else if (strcmp(dArgv[3], "-assy") == 0)
        {
            isArmed = FALSE;
        }
        else
        {
            dprintf("Bad argument %s\n", dArgv[3]);
            dprintf(errorStr);
            return;
        }
    }

    // Dump channel register value
    chNum = pDisp[indexGpu].dispGetChanNum(chName, headWinSorNum);
    dprintf("0x%x\n", pDisp[indexGpu].dispGetChannelStateCacheValue(chNum, isArmed, offset));
}

//-----------------------------------------------------
// ddsc [-h<hd>]
//
//-----------------------------------------------------
DECLARE_API( ddsc )
{
    LwU32 headNum = 0xFFFFFFFF;
    int argNum, temp;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    for (argNum = 0; argNum < dArgc; ++argNum)
    {
        if (sscanf(dArgv[argNum], "-h%d", &temp))
        {
            headNum = temp;
        }
        else
        {
            dprintf("Bad argument %s\n", dArgv[argNum]);
            dprintf("lw: Usage: !lw.ddsc [-h<hd>]\n");
            return;
        }
    }

    pDisp[indexGpu].dispReadDscStatus(headNum);
}

//-----------------------------------------------------
// dlowpower [-h<hd>]
//
//-----------------------------------------------------
DECLARE_API( dlowpower )
{
    int argNum;
    BOOL bAnalyzeMscg = FALSE;
    BOOL bClearCounter = FALSE;
    BOOL bPrintCounters = FALSE;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    for (argNum = 0; argNum < dArgc; ++argNum)
    {
        if (strcmp(dArgv[argNum], "-analyze") == 0)
        {
            bAnalyzeMscg = TRUE;
        }
        else if (strcmp(dArgv[argNum], "-clear") == 0)
        {
            bClearCounter = TRUE;
        }
        else if (strcmp(dArgv[argNum], "-counters") == 0)
        {
            bPrintCounters = TRUE;
        }
        else
        {
            dprintf("Bad argument %s\n", dArgv[argNum]);
            dprintf("lw: Usage: !lw.dlowpower [-analyze] [-clear] [-counters]\n");
            return;
        }
    }
    if (bAnalyzeMscg)
    {
        pDisp[indexGpu].dispAnalyzeDisplayLowPowerMscg();
    }
    else if (bClearCounter)
    {
        pDisp[indexGpu].dispClearDisplayLowPowerMscgCounters();
    }
    else if (bPrintCounters)
    {
        pDisp[indexGpu].dispPrintDisplayLowPowerMscgCounters();
    }
    else
    {
        pDisp[indexGpu].dispReadDisplayLowPowerStatus();
    }
}

//-----------------------------------------------------
// dgetdbgmode <chanName> [-h<hd>/-w<wd>]
//
//-----------------------------------------------------
DECLARE_API( dgetdbgmode )
{
    char *chName = NULL;
    LwS32 headNum = -1;
    LwS32 winNum = -1;
    LwU32 temp;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    if ( dArgc > 0 )
    {
        chName = dArgv[0];
        if (sscanf(dArgv[1], "-h%d", &temp))
        {
            headNum = temp;
        }
        else if (sscanf(dArgv[1], "-w%d", &temp))
        {
            winNum = temp;
            headNum = winNum;
        }
        else
        {
            dprintf("lw: Usage: !lw.dgetdbgmode [chName] [-h<hd>/-w<wd>]\n");
            return ;
        }

        if (!strcmp("win", chName) || !strcmp("winimm", chName))
        {
            if (winNum == -1)
            {
                dprintf("lw: Usage: !lw.dgetdbgmode <win/winim> [-w<wd>] \n");
                return ;
            }
        }
    }

    pDisp[indexGpu].dispDumpGetDebugMode(chName, headNum, dArgc);
}

//-----------------------------------------------------
// dsetdbgmode <chanName> [-h<hd>/-w<wd>] <1/0>
//
//-----------------------------------------------------
DECLARE_API( dsetdbgmode )
{
    char *chName = NULL;
    LwU32 headNum = 0;
    LwS32 chNum =0;
    LwS32 winNum =0;
    LwS32 debugMode;
    LwU32 temp;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    if ( dArgc > 1 )
    {
        chName = dArgv[0];

        // we started not fully relying on HAL interface...
        // if it's core, it's okay to omit head #
        if (!strcmp("core", chName) && dArgc == 2)
        {
            if ( !sscanf(dArgv[1], "%d", &debugMode) &&
                    !(debugMode == 1 || debugMode == 0))
            {
                dprintf("Enable : 1, Disable : 0\n");
                return ;
            }
        }
        else if (dArgc == 3)
        {
            if (sscanf(dArgv[1], "-h%d", &temp))
            {
                headNum = temp;
            }
            else if (sscanf(dArgv[1], "-w%d", &temp))
            {
                winNum = temp;
                headNum = winNum;
            }
            else
            {
                dprintf("lw: Usage: !lw.dsetdbgmode [chName] [-h<hd>/-w<wd>] <1/0>\n");
                return ;
            }

            if (!strcmp("win", chName) || !strcmp("winimm", chName))
            {
                if (winNum == -1)
                {
                    dprintf("lw: Usage: !lw.dsetdbgmode <win/winim> [-h<hd>/-w<wd>] <1/0>\n");
                    return ;
                }
            }

            if ( !sscanf(dArgv[2], "%d", &debugMode) &&
                    !(debugMode == 1 || debugMode == 0))
            {
                dprintf("Enable : 1, Disable : 0\n");
                return ;
            }
        }
        else
        {
            dprintf("lw: Usage: !lw.dsetdbgmode [chName] [-h<hd>/-w<wd>] <1/0>\n");
            return;
        }
    }
    else
    {
        dprintf("lw: Usage: !lw.dsetdbgmode [chName] [-h<hd>/-w<wd>] <1/0>\n");
        return;
    }

    pDisp[indexGpu].dispDumpSetDebugMode(chName, headNum, debugMode);
}

#define LW_SHIFTMASK_VAL(v,h,l) (((v)& (0xFFFFFFFF>>(31-((h) % 32)+((l) % 32)))) << ((l) % 32))
#define LW_MASK(h,l) ((0xFFFFFFFF>>(31-((h) % 32)+((l) % 32))) << ((l) % 32))

#define MTD_USAGE \
            "Usage: dinjectmethod <chanName> -h<headNum>/-w<winNum> method1 data1 ...\n"\
                    "       mthd format : METHOD_NAME[@n[@n]]\n"\
                    "       mthd format : Method Address\n"\
                    "       Try to add '*' to the end of method or field name\n"
//-----------------------------------------------------
// dinjectmethod
//
//-----------------------------------------------------
DECLARE_API( dinjectmethod )
{

    char *chName = NULL;
    int i, escape =  0, omitHd = 0, lwrArg = 0;
    size_t len;
    // MinArg is 1, after getting rid of <DBG Auto Restore:1/0>
    int minArg = 1;
    LwS32 chInstance = -1;
    LwS32 chNum =0;
    LwS32 debugMode = 0;
    // Setting debugRestore to "Always ON"
    const LwS32 restore = 1;
    LwU32 offset, hbit, lbit, data, sc, temp;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc >= minArg)
    {
        chName = dArgv[lwrArg++];

        // get head, in case of "core", head can be omitted
        omitHd = !strcmp(chName, "core");
        if(!omitHd && (dArgc < ++minArg))
        {
            goto print_usage;
        }

        // get head
        if (sscanf(dArgv[lwrArg], "-h%d", &temp))
        {
            // advance lwrArg if really used
            chInstance = temp;
            lwrArg++;
        }
        else if (omitHd)
        {
            chInstance = 0;
        }
        else if (sscanf(dArgv[lwrArg], "-w%d", &temp))
        {
            chInstance = temp;
            lwrArg++;
        }
        else
        {
            dprintf("headNum/winNum should be a number\n");
            return ;
        }

        if (!strcmp("win", chName) || !strcmp("winim", chName))
        {
            if (chInstance == -1)
            {
                goto print_usage;
            }
        }

        // check for escape, '*' or '!' (old)
        len = strlen(dArgv[dArgc - 1]);
        for (i = 0; i < (int)len; i++)
        {
            if (dArgv[dArgc - 1][i] == '*' || dArgv[dArgc - 1][i] == '!')
            {
                escape = 1;
                break;
            }
        }
    }
    else
    {
        dprintf(MTD_USAGE);
        return;
    }

    MGPU_LOOP_START;
    {
        if ((chNum = pDisp[indexGpu].dispGetChanNum(chName, chInstance)) == -1)
        {
            dprintf("use correct chanName and head number\n");
            return;
        }

        if (escape)
        {
            // ignore returns..
            pDisp[indexGpu].dispMatchMethod(chNum, chInstance, dArgv[dArgc-1], &offset, &hbit, &lbit, &sc);
        }
        else
        {
            // auto restore
            if (restore)
            {
                debugMode = pDisp[indexGpu].dispGetDebugMode(chNum);
            }

            // enable
            pDisp[indexGpu].dispSetDebugMode(chNum, 1);

            for ( i = lwrArg; i < dArgc ; ++i)
            {
                unsigned int val = 0;

                // dispMatchMethod - screwy, but we have to pass headNum(chInstance) since
                //                   some core methods work on both heads. -- G80
                if (pDisp[indexGpu].dispMatchMethod(chNum, chInstance, dArgv[i], &offset, &hbit, &lbit, &sc))
                {
                    dprintf("ERROR\n");
                    break ;
                }

                if ( i + 1 >= dArgc)
                {
                    dprintf("Data required for method %s\n", dArgv[i]);
                    break;
                }
                if (sc)
                {
                    dprintf(".. Using State Cache register @ 0x%08x\n", sc);
                    val = GPU_REG_RD32(sc);
                    val = val & ~(LW_MASK(hbit, lbit));
                }
                else
                {
                    dprintf("   This method doesn't seem to have the related SC register\n");
                }

                data = strtoul(dArgv[++i], NULL, 0);
                data = val | LW_SHIFTMASK_VAL(data, hbit, lbit);
                // DEBUG
                // dprintf("offset: 0x%x, hbit: %d, lbit :%d, data : 0x%x sc : 0x%x\n", offset, hbit, lbit, data, sc);

                if (pDisp[indexGpu].dispInjectMethod(chNum, offset, data))
                {
                    dprintf("inject method failed\n");
                    break;
                }
            }
            // restore the debug mode
            if (restore)
            {
                pDisp[indexGpu].dispSetDebugMode(chNum, debugMode);
            }
        }
    }
    MGPU_LOOP_END;
    return;
print_usage:
    dprintf(MTD_USAGE);
    return;

}

//-----------------------------------------------------
// dreadasy
//
//-----------------------------------------------------
DECLARE_API( dreadasy )
{
    dprintf("Not implemented!\n");
}

//-----------------------------------------------------
// dreadarm
//
//-----------------------------------------------------
DECLARE_API( dreadarm )
{
    dprintf("Not implemented!\n");
}

#define DEFAULT_PB_DUMP_SIZE        0x20
#define DISP_PUSH_BUFFER_SIZE       4096
//-----------------------------------------------------
// ddumppb
//
//-----------------------------------------------------
DECLARE_API( ddumppb )
{
    LwS32 channelNum = 0;
    LwS32 i;
    LW_STATUS status;
    LwS32 chInstance = -1;
    LwU32 temp = 0;
    LwS32 Offset = 0;
    LwS32 numDwords = DEFAULT_PB_DUMP_SIZE;
    char *usage = "Usage: ddumppb <channelName> [-p] -h<headNum>/-w<winnum> [numDwords/-f] [-o<OffsetDwords>]\n";
    char *channelName = NULL;
    BOOL isParse = FALSE;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc < 1)
    {
        dprintf("%s\n", usage);
        return;
    }

    // parsing the command line arguments
    channelName = dArgv[0];
    for (i = 1; i < dArgc; i++)
    {
        if (strcmp(dArgv[i], "-p" ) == 0)
        {
            isParse = TRUE;
        }
        else if (strcmp(dArgv[i], "-f" ) == 0)
        {
            numDwords = DISP_PUSH_BUFFER_SIZE/4;
        }
        else if (sscanf(dArgv[i], "-h%d",&temp))
        {
            chInstance = temp;
            if ( (chInstance >= (LwS32)pDisp[indexGpu].dispGetNumHeads()) || (chInstance <= -1) )
            {
                chInstance = 0;
                dprintf("Wrong head!!! hence Setting Head = %d\n\n",chInstance);
            }
        }
        else if (sscanf(dArgv[i], "-w%d",&temp))
        {
            chInstance = temp;
        }
        else if (sscanf(dArgv[i], "-o%x", &Offset) )
        {
            if ( (Offset >= DISP_PUSH_BUFFER_SIZE / 4) || (Offset <= -1) )
            {
                Offset = 0;
                dprintf("Offset out of range!!! Hence Setting Offset = %x Dwords\n\n",Offset);
            }
        }
        else if((numDwords == DEFAULT_PB_DUMP_SIZE) && (sscanf(dArgv[i], "%x", &numDwords)== 1))
        {
            if ( numDwords <= 0 )
            {
                numDwords = DEFAULT_PB_DUMP_SIZE;
            }
        }
        else
        {
            dprintf("%s\n",usage);
            return;
        }
    }

    if (!strcmp("win", channelName) || !strcmp("winim", channelName))
    {
        if (chInstance == -1)
        {
            dprintf("%s\n",usage);
            return ;
        }
    }

    //
    // get head, in case of "core", head can be omitted
    //
    if (strcmp(channelName, "core") == 0)
    {
        chInstance = 0;
    }

    //
    // check if numDwords after offset exceeds the push buffer limit (4k)
    //
    if (numDwords > DISP_PUSH_BUFFER_SIZE / 4)
    {
        numDwords = DISP_PUSH_BUFFER_SIZE / 4;
    }

    MGPU_LOOP_START;
    {
        if ((channelNum = pDisp[indexGpu].dispGetChanNum(channelName, chInstance)) == -1)
        {
            dprintf("Use correct channelName and head number\n");
            return;
        }
        status = pDisp[indexGpu].dispDumpPB(channelNum, chInstance, numDwords, Offset, isParse);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// danalyzeblank <headNum>
//
//-----------------------------------------------------
DECLARE_API( danalyzeblank )
{
    LwS32 status = 0;
    LwS32 head = 0;

    char *usage = "danalyzeblank -h<headNum>";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc > 0)
    {
        if (!sscanf(dArgv[0],"-h%d", &head))
        {
            dprintf("Error : HeadNum should be a number, -h#\n");
            dprintf("%s\n", usage);
            return;
        }
    }
    MGPU_LOOP_START;
    {
        status = pDisp[indexGpu].dispAnalyzeBlank(head);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// danalyzehang
//
//-----------------------------------------------------
DECLARE_API( danalyzehang )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pDisp[indexGpu].dispAnalyzeHang();
    }
    MGPU_LOOP_END;
}


//Helper function for dintr
void PrintUsageOfdintr ()
{
    dprintf("Usage:\n");
    dprintf ("%-6s%-15s:%s\n", "dintr", "", "Prints dispatch info and supported events and interrupts");
    dprintf ("%-6s%-15s:%s\n", "dintr", "-all", "Prints dispatch info and all available events and interrupts");
    dprintf ("%-6s%-15s:%s\n", "dintr", "-evt", "Prints only supported events");
    dprintf ("%-6s%-15s:%s\n", "dintr", "-intr", "Prints only supported interrupts");
    dprintf ("%-6s%-15s:%s\n", "dintr", "-dispatch", "Prints only dispatch info");
}

//-----------------------------------------------------
// dintr
//
//-----------------------------------------------------
DECLARE_API( dintr )
{
    //The arguments for dintr are not used in the pre-LwDisplay implementation. They are only of relevance for Volta and later.
    LwU32 all = 0;
    LwU32 evt = 1;
    LwU32 intr = 1;
    LwU32 dispatch = 1;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc > 1)
    {
        dprintf("Bad argument\n");
        PrintUsageOfdintr();
        return;
    }

    if (dArgc == 1)
    {
        if (strcmp(dArgv[0], "-all") == 0)
        {
            all = 1;
            evt = 1;
            intr = 1;
            dispatch = 1;
        }
        else if (strcmp(dArgv[0], "-evt") == 0)
        {
            all = 0;
            evt = 1;
            intr = 0;
            dispatch = 0;
        }
        else if (strcmp(dArgv[0], "-intr") == 0)
        {
            all = 0;
            evt = 0;
            intr = 1;
            dispatch = 0;
        }
        else if (strcmp(dArgv[0], "-dispatch") == 0)
        {
            all = 0;
            evt = 0;
            intr = 0;
            dispatch = 1;
        }
        else
        {
            dprintf("Bad argument \"%s\"\n", dArgv[0]);
            PrintUsageOfdintr();
            return;
        }
    }

    MGPU_LOOP_START;
    {
        pDisp[indexGpu].dispAnalyzeInterrupts(all, evt, intr, dispatch);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dtim
//
//-----------------------------------------------------
DECLARE_API( dtim )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pDisp[indexGpu].dispTimings();
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// drdhdmidecoder <orNum> <address>
//
//-----------------------------------------------------
DECLARE_API( drdhdmidecoder )
{
    LwU32 orNum = 0;
    LwU32 addr = 0;

    char *usage = "Usage: emulation only: drdhdmidecoder <orNum> <address>\n";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc != 2)
    {
        dprintf("Incorrect number of arguments to %s. Bailing out early\n", __FUNCTION__);
        dprintf("%s\n", usage);
        return;
    }

    if (!sscanf(dArgv[0], "%d", &orNum))
    {
        dprintf("Invalid argument %s to %s. Bailing out early\n", dArgv[0], __FUNCTION__);
        return;
    }

    if (!sscanf(dArgv[1], "%x", &addr))
    {
        dprintf("Invalid argument %s to %s. Bailing out early\n", dArgv[1], __FUNCTION__);
        return;
    }

    MGPU_LOOP_START;
    {
        pDisp[indexGpu].dispReadHDMIDecoder(orNum, addr);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dwrhdmidecoder <orNum> <address> <data>
//
//-----------------------------------------------------
DECLARE_API( dwrhdmidecoder )
{
    LwU32 orNum = 0;
    LwU32 addr = 0;
    LwU32 wrData = 0;

    char *usage = "Usage: emulation only: dwrhdmidecoder <orNum> <address> <data>\n";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc != 3)
    {
        dprintf("Incorrect number of arguments to %s. Bailing out early\n", __FUNCTION__);
        dprintf("%s\n", usage);
        return;
    }

    if (!sscanf(dArgv[0], "%d", &orNum))
    {
        dprintf("Invalid argument %s to %s. Bailing out early\n", dArgv[0], __FUNCTION__);
        return;
    }

    if (!sscanf(dArgv[1], "%x", &addr))
    {
        dprintf("Invalid argument %s to %s. Bailing out early\n", dArgv[1], __FUNCTION__);
        return;
    }

    if (!sscanf(dArgv[2], "%x", &wrData))
    {
        dprintf("Invalid argument %s to %s. Bailing out early\n", dArgv[2], __FUNCTION__);
        return;
    }

    MGPU_LOOP_START;
    {
        if (pDisp[indexGpu].dispWriteHDMIDecoder(orNum, addr, wrData))
        {
            pDisp[indexGpu].dispReadHDMIDecoder(orNum, addr);
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// ddispowner
//
//-----------------------------------------------------
DECLARE_API( ddispowner )
{

    int owner;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        owner = pDisp[indexGpu].dispDispOwner();
        if (owner == 0)
            dprintf("Display Owner : DRIVER\n");
        else if (owner == 1)
            dprintf("Display Owner : VBIOS\n");
        else
            dprintf("Display Owner : ERROR\n");

        pDisp[indexGpu].dispPrintScanoutOwner();
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dorstate
//
//-----------------------------------------------------
DECLARE_API( dorstate )
{
    dprintf("Not implemented!\n");
}

//-----------------------------------------------------
// dsetorstate
//
//-----------------------------------------------------
DECLARE_API( dsetorstate )
{
    dprintf("Not implemented!\n");
}

//-----------------------------------------------------
// dsorpadlinkconn
//
//-----------------------------------------------------
DECLARE_API( dsorpadlinkconn )
{

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pDisp[indexGpu].dispOrPadlinkConnection();
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// checkprodvals
//
//-----------------------------------------------------
DECLARE_API ( checkprodvals )
{

    FILE *pFile;

    CHECK_INIT(MODE_LIVE);

    if (!args || *args == '\0')
    {
        dprintf("lw: %s: please provide one argument for command %s\n",
                __FUNCTION__, args);
        return;
    }

    if( (pFile = fopen(args,"rb" )) == NULL)
    {
        dprintf("lw: %s: could not open the file %s\n", __FUNCTION__, args);
        return;
    }

    pHwprod[indexGpu].hwprodCheckVals(pFile);
}

//-----------------------------------------------------
// Parses arguments to "sigdump"
//  Looks for:
// sigdump -f <gpc><tpc><fbp> [-sfile <filename>]
// Example : sigdump -f 121
// If any one is not present; sigdump_floorsweep is aborted
//
//-----------------------------------------------------
static LwU32 sigdumpParseOption
(
    char *args,
    BOOL *bFs,
    LwU32* gpc,
    LwU32* tpc,
    LwU32* fbp,
    char **sfile
)
{
    char *params;
    LwU64 config = 0;
    LwS32 ind = 0;

    *bFs = FALSE;
    dispParseArgs(args);
    while(ind < dArgc)
    {
        if(strcmp(dArgv[ind], "-sfile")==0)
        {
            if((ind+1) == dArgc)
            {
                dprintf("lw: File name missing\n");
                return LW_ERR_GENERIC;
            }
            *sfile = (char *)dArgv[ind+1];
            break;
        }
        ++ind;
    }

    if (parseCmd(args, "f", 1, &params))
    {
        GetSafeExpressionEx(params, &config, &params);
        //If config is 0, this is consider as a special case where signals other than  gpc, tpc &fbp are to be dumped. Dont quit.
        if(!config)
        {
            *gpc = *tpc = *fbp = 0;
            *bFs = TRUE;
        }
        else if (config > 999 || config < 100)
        {
            dprintf("lw: Improper config\n");
            return LW_ERR_GENERIC;
        }
        else
        {
            *gpc = (LwU32)(config/100);
            *tpc = (LwU32)((config%100)/10);
            *fbp = (LwU32)((config%100)%10);
            if ((*gpc) && (*tpc) && (*fbp))
            {
                *bFs = TRUE;
            }
        }
    }
    else if (parseCmd(args, "xbar", 0, NULL))
    {
        //temp hack to use alternate xbar signals
        *gpc = 0;
        *tpc = 0;
        *fbp = 0;
        *bFs = TRUE;
    }

    return LW_OK;
}

//-----------------------------------------------------
// Parses arguments to "sigdump" for Maxwell and later
//
//-----------------------------------------------------
static LwU32 sigdumpParseOptionMaxwellAndLater
(
    char *args,
    BOOL *bFs,
    LwU32* gpc,
    LwU32* tpc,
    LwU32* fbp,
    char **ofile,
    char **chipletKeyword,
    char **chipletNumKeyword,
    char **domainKeyword,
    char **domainNumKeyword,
    char **instanceNumKeyword,
    LwU32 *n,
    int *regWriteOptimization,
    int *markerValuesCheck,
    int *regWriteCheck,
    int *verifySigdump,
    int *engineStatusVerbose,
    int *priCheckVerbose,
    int *multiSignalOptimization
)
{
    LwS32 ind = 0;
    char *p;

    *bFs = FALSE;
    dispParseArgs(args);
    while(ind < dArgc)
    {
        if ((dArgv[ind][0] == '-')                       &&
            (strcmp (dArgv[ind], "-help")                &&
             strcmp (dArgv[ind], "-out")                 &&
             strcmp (dArgv[ind], "-unoptimized")         &&
             strcmp (dArgv[ind], "-verify")              &&
             strcmp (dArgv[ind], "-enginestatusverbose") &&
             strcmp (dArgv[ind], "-pricheckverbose")     &&
             strcmp (dArgv[ind], "-nocheck")             &&
             strcmp (dArgv[ind], "-checkwrites")         &&
             strcmp (dArgv[ind], "-repeat")              &&
             strcmp (dArgv[ind], "-chiplet")             &&
             strcmp (dArgv[ind], "-chipletNum")          &&
             strcmp (dArgv[ind], "-domain")              &&
             strcmp (dArgv[ind], "-domainNum")           &&
             strcmp (dArgv[ind], "-instanceNum")         &&
             strcmp (dArgv[ind], "-multisignal")
            ))
        {
            dprintf ("lw: Wrong command-line argument!\n");
            return LW_ERR_GENERIC;
        }
        if (strcmp (dArgv[ind], "-help") == 0)
        {
            return LW_ERR_GENERIC;
        }
        else if ((strcmp (dArgv[ind], "-out") == 0))
        {
            if(((ind + 1) == dArgc) || (dArgv[ind + 1][0] == '-'))
            {
                dprintf("lw: Output file name missing!\n");
                return LW_ERR_GENERIC;
            }
            *ofile = (char *)dArgv[ind + 1];
        }
        else if (strcmp (dArgv[ind], "-unoptimized") == 0)
        {
            *regWriteOptimization = 0;
        }
        else if (strcmp (dArgv[ind], "-verify") == 0)
        {
            *verifySigdump = 1;
        }
        else if (strcmp (dArgv[ind], "-enginestatusverbose") == 0)
        {
            *engineStatusVerbose = 1;
        }
        else if (strcmp (dArgv[ind], "-pricheckverbose") == 0)
        {
            *priCheckVerbose = 1;
        }
        else if (strcmp (dArgv[ind], "-nocheck") == 0)
        {
            *markerValuesCheck = 0;
        }
        else if (strcmp (dArgv[ind], "-checkwrites") == 0)
        {
            *regWriteCheck = 1;
        }
        else if (strcmp (dArgv[ind], "-multisignal") == 0)
        {
            *multiSignalOptimization = 1;
        }
        else if (strcmp (dArgv[ind], "-chiplet") == 0)
        {
            if ((ind + 1) == dArgc)
            {
                dprintf("lw: Keyword missing\n");
                return LW_ERR_GENERIC;
            }
            *chipletKeyword = (char *)dArgv[ind + 1];
        }
        else if (strcmp (dArgv[ind], "-chipletNum") == 0)
        {
            if ((ind + 1) == dArgc)
            {
                dprintf("lw: Keyword missing\n");
                return LW_ERR_GENERIC;
            }
            *chipletNumKeyword = (char *)dArgv[ind + 1];
        }
        else if (strcmp (dArgv[ind], "-domain") == 0)
        {
            if ((ind + 1) == dArgc)
            {
                dprintf("lw: Keyword missing\n");
                return LW_ERR_GENERIC;
            }
            *domainKeyword = (char *)dArgv[ind + 1];
        }
        else if (strcmp (dArgv[ind], "-domainNum") == 0)
        {
            if ((ind + 1) == dArgc)
            {
                dprintf("lw: Keyword missing\n");
                return LW_ERR_GENERIC;
            }
            *domainNumKeyword = (char *)dArgv[ind + 1];
        }
        else if (strcmp (dArgv[ind], "-instanceNum") == 0)
        {
            if ((ind + 1) == dArgc)
            {
                dprintf("lw: Keyword missing\n");
                return LW_ERR_GENERIC;
            }
            *instanceNumKeyword = (char *)dArgv[ind + 1];
        }
        else if (strcmp (dArgv[ind], "-repeat") == 0)
        {
            if ((ind + 1) == dArgc)
            {
                dprintf ("lw: Please enter the number of back-to-back sigdumps requested and try again!\n");
                return LW_ERR_GENERIC;
            }
            *n = strtol (dArgv[ind+1], &p, 10);
            if (!(*n))
            {
                dprintf ("lw: Please enter a non-negative, non-zero value with \"-repeat\"!\n");
                return LW_ERR_GENERIC;
            }
        }
        // End here.
        ++ind;
    }
    return LW_OK;
}

//------------------------------------------------
// dump out signals.
//
// For the floorswept version;
// sigdump -f <gpc><tpc><fbp>
// Example : sigdump -f 121
//--------------------------------------------------
DECLARE_API( sigdump )
{
    //TODO : Sigdump needs to be updated so as to support SMC

    FILE *fp;
    char *sfile = NULL;
    char *ofile = NULL;
    char *chipletKeyword = NULL;
    char *chipletNumKeyword = NULL;
    char *domainKeyword = NULL;
    char *domainNumKeyword = NULL;
    char *instanceNumKeyword = NULL;
    char *prefix = NULL;
    BOOL bFs;
    LwU32 gpc, tpc, fbp;
    LwU32 n = 1;                        // Number of back-to-back sigdumps.
    LwU32 i;
    int regWriteOptimization = 1;       // To activate register write optimization by default.
    int regWriteCheck = 0;              // Deactivate checks on register writes by default.
    int markerValuesCheck = 1;          // Activate checks on fixed marker signal values by default.
    int verifySigdump = 0;              // Do not generate "sigdump_verif.txt" by default.
    int engineStatusVerbose = 0;        // Do not print ENGINE_STATUS not going to idle cases by default.
    int priCheckVerbose = 0;            // Do not print PRI check failures by default.
    int multiSignalOptimization = 0;    // Multi-signal optimization disabled by default.

    char cat_ofile[100];
    char fName[90];
    char fExtn[11];

    CHECK_INIT(MODE_LIVE);

    if (IsGM107orLater ())
    {
        if (LW_ERR_GENERIC == sigdumpParseOptionMaxwellAndLater (args,
                                                                 &bFs,
                                                                 &gpc,
                                                                 &tpc,
                                                                 &fbp,
                                                                 &ofile,
                                                                 &chipletKeyword,
                                                                 &chipletNumKeyword,
                                                                 &domainKeyword,
                                                                 &domainNumKeyword,
                                                                 &instanceNumKeyword,
                                                                 &n,
                                                                 &regWriteOptimization,
                                                                 &markerValuesCheck,
                                                                 &regWriteCheck,
                                                                 &verifySigdump,
                                                                 &engineStatusVerbose,
                                                                 &priCheckVerbose,
                                                                 &multiSignalOptimization))
        {
#if defined(_WIN32) || defined(_WIN64)
            prefix = "lw: !lw.";
#else
            prefix = "lw: ";
#endif
            dprintf("lw: Usage:    \n");

            dprintf("%ssigdump                           // Sigdump default\n", prefix);
            dprintf("%ssigdump -help                     // Display sigdump help\n", prefix);
            dprintf("%ssigdump -unoptimized              // Dump with unoptimized register writes (default is optimized writes)\n", prefix);
            dprintf("%ssigdump -checkwrites              // Dump with checks on register writes\n", prefix);
            dprintf("%ssigdump -nocheck                  // Dump without checks on fixed marker and/or static pattern signals (default is to check)\n", prefix);
            dprintf("%ssigdump -verify                   // Dump and also generate \"sigdump_verif.txt\" containing the sequence of register programming for each signal\n", prefix);
            dprintf("%ssigdump -enginestatusverbose      // Dump with prints if an engine does not go to EMPTY after 20 polls\n", prefix);
            dprintf("%ssigdump -pricheckverbose          // Dump with error prints if PRI reads read 0xBADFxxxx values\n", prefix);
            dprintf("%ssigdump -multisignal              // Dump with multi-signal optimization (default is disabled)\n", prefix);
            dprintf("%ssigdump -out <filename>           // Dump results into <filename>\n", prefix);
            dprintf("%ssigdump -repeat <number>          // Dump <number> back-to-back sigdumps; \"_<dumpID>\" will be appended to the dump filename\n", prefix);
            dprintf("%ssigdump -chiplet <keyword>        // Dump only signals with chiplet = <keyword> (simple string comparison)\n", prefix);
            dprintf("%ssigdump -chipletNum <keyword>     // Dump only signals with chiplet# = <keyword> (simple string comparison)\n", prefix);
            dprintf("%ssigdump -domain <keyword>         // Dump only signals with domain = <keyword> (simple string comparison)\n", prefix);
            dprintf("%ssigdump -domainNum <keyword>      // Dump only signals with domain# = <keyword> (simple string comparison)\n", prefix);
            dprintf("%ssigdump -instanceNum <keyword>    // Dump only signals with instance# = <keyword> (simple string comparison)\nlw:\n", prefix);

            return;
        }

        if (ofile != NULL)
        {
            if (strstr (ofile, ".") == NULL)
            {
                strcpy(fName, ofile);
                strcpy (fExtn, "txt");
            }
            else
            {
                strcpy(fName, strtok (ofile, "."));
                strcpy(fExtn, strtok (NULL,"."));
            }
        }

        for (i = 1; i <= n; i++)
        {
            if (ofile != NULL)                          // If sigdump is ilwoked with custom output filename.
            {
                if (n == 1)                             // If single sigdump is requested.
                {
                    sprintf(cat_ofile,"%s.%s",fName,fExtn);
                }
                else
                {
                    sprintf(cat_ofile,"%s_%u.%s",fName,i,fExtn);
                }
            }
            else
            {
                strcpy(fName,"sigdump");
                strcpy(fExtn,"txt");
                if (n == 1)
                {
                    sprintf(cat_ofile,"%s.%s",fName,fExtn);
                }
                else
                {
                    sprintf(cat_ofile,"%s_%u.%s",fName,i,fExtn);
                }
            }
            fp = fopen(cat_ofile, "w+");
            if (fp == NULL)
            {
                dprintf("lw: Unable to open %s to write to\n",cat_ofile);
                return;
            }
            if (n > 1)
            {
                dprintf("lw: Sigdump iteration: %u of %u\n", i, n);
            }
            dprintf("lw: %s created in the current working directory.\n",cat_ofile);
            pSig[indexGpu].sigGetSigdump( fp,
                                          regWriteOptimization,
                                          regWriteCheck,
                                          markerValuesCheck,
                                          verifySigdump,
                                          engineStatusVerbose,
                                          priCheckVerbose,
                                          multiSignalOptimization,
                                          chipletKeyword,
                                          chipletNumKeyword,
                                          domainKeyword,
                                          domainNumKeyword,
                                          instanceNumKeyword
                                          );
            fclose(fp);
        }
    }
    else {
        fp = fopen("sigdump.txt", "w");
        if (fp == NULL) {
            dprintf("lw: Unable to open sigdump.txt\n");
            return;
        }
        else
        {
            dprintf("lw: ===========================================================================================================================\n");
            dprintf("lw:  SIGDUMP NOTE:\n");
            dprintf("lw:  SIGDUMP NOTE: SIGDUMP SIGNAL CONTENT MAY BE CORRUPTED BY GPU POWER FEATURES!!!  See GK110 Bug http://lwbugs/1028519 \n");
            dprintf("lw:  SIGDUMP NOTE:\n");
            dprintf("lw:  SIGDUMP NOTE: The following links provide details on how to statically disable lwrrently supported power features:\n");
            dprintf("lw:  SIGDUMP NOTE:  https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/RegKeys\n");
            dprintf("lw:  SIGDUMP NOTE:  https://wiki.lwpu.com/gpuhwkepler/index.php/Emulation_Feature_Enablement_Plan#Regkey \n");
            dprintf("lw:  SIGDUMP NOTE:\n");
            dprintf("lw:  SIGDUMP NOTE: When providing sigdump results, be sure to inform consumers of power feature disablement (or lack thereof).\n");
            dprintf("lw:  SIGDUMP NOTE:\n");
            dprintf("lw: ===========================================================================================================================\n");
            dprintf("lw: sigdump.txt created in the current working directory.\n");

            if (LW_ERR_GENERIC == sigdumpParseOption(args, &bFs, &gpc, &tpc, &fbp, &sfile))
            {
                dprintf("lw: Usage:    \n");
                dprintf("lw: !lw.sigdump                  //sigdump default\n");
                dprintf("lw: !lw.sigdump -f <xyz>         //fs config xyz (gpc,tpc,fbp)\n");
                dprintf("lw: !lw.sigdump -xbar            //specially for xbar signals(fermi)\n");
                dprintf("lw: !lw.sigdump -sfile <file>    //Dump only the signals present in the file. Can be used with other options\n");
            }
            else if (bFs)
            {
                if(sfile)
                    pSig[indexGpu].sigGetSelectSigdumpFloorsweep(fp, gpc, tpc, fbp, sfile);
                else
                    pSig[indexGpu].sigGetSigdumpFloorsweep(fp, gpc, tpc, fbp);
            }
            else
            {
                if(sfile)
                    pSig[indexGpu].sigGetSelectSigdump(fp, sfile);
                else
                    pSig[indexGpu].sigGetSigdump(fp,
                                                 regWriteOptimization,
                                                 regWriteCheck,
                                                 markerValuesCheck,
                                                 verifySigdump,
                                                 engineStatusVerbose,
                                                 priCheckVerbose,
                                                 multiSignalOptimization,
                                                 chipletKeyword,
                                                 chipletNumKeyword,
                                                 domainKeyword,
                                                 domainNumKeyword,
                                                 instanceNumKeyword
                                                 );
            }
            fclose(fp);
        }
    }
}

#if defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(LW_WINDOWS)
//-----------------------------------------------------
// tests
//
//-----------------------------------------------------
DECLARE_API( tests )
{
    getUserTestsInput();
}
#endif

//-----------------------------------------------------
// verbose
//
//-----------------------------------------------------
DECLARE_API( verbose )
{
    LwU32 level;
    level = (LwU32) GetSafeExpression(args);
    verboseLevel = level;
    dprintf("lw: verboseLevel is lwrrently set to: 0x%x\n", verboseLevel);
    dprintf("\n");
}

#if !defined(USERMODE)
//-----------------------------------------------------
// pd [-grIdx N] <register>
// - dumps a bar0 register by name or offset
//-----------------------------------------------------
void priv_dump(const char *);
DECLARE_API( pd )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        char *param;
        LwBool isGrReg = FALSE, isGrIndex = FALSE;
        LwU32 grIdx = 0;

        //With SMC mode enabled, the TOOL_WINDOW should be updated
        isGrIndex = parseCmd(args, "grIdx", 1, &param);

        if (args[0] >= '0' && args[0] <= '9') {
            LwU32 lwReg = (LwU32)GetSafeExpression(args);
            isGrReg = pGr[indexGpu].grPgraphOffset(lwReg);
        } else {
            isGrReg = strncmp(args, "LW_PGRAPH", strlen("LW_PGRAPH")) == 0;
        }

        if (isGrReg && pGr[indexGpu].grGetSmcState())
        {
            if(!isGrIndex)
            {
                dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
                return;
            }

            grIdx = *param -'0';
            // Set the window by passing grIdx
            pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        }
        else if (isGrIndex && *param != '0')
        {
            dprintf("lw: non-zero grIndex specified without SMC\n");
            return;
        }

        priv_dump(args);
    }
    MGPU_LOOP_END;
}

//---------------------------------------------------
// pe <register> data
// - writes to a bar0 register by name or address
// ---------------------------------------------------
void priv_emit(const char *);
DECLARE_API( pe )
{
    CHECK_INIT(MODE_LIVE);

    priv_emit(args);
}

#endif

void display_number(LwU32 num)
{
    dprintf(" dec: %d\n hex: %x\n", num, num);
    dprintf(" bin:  3           2              1\n");
    dprintf(" bin: 1098 7654 3210 9876   5432 1098 7654 3210\n");
    dprintf(" bin: %u%u%u%u %u%u%u%u %u%u%u%u %u%u%u%u | %u%u%u%u %u%u%u%u %u%u%u%u %u%u%u%u\n\n",
            (num >> 31) & 1, (num >> 30) & 1, (num >> 29) & 1, (num >> 28) & 1,
            (num >> 27) & 1, (num >> 26) & 1, (num >> 25) & 1, (num >> 24) & 1,
            (num >> 23) & 1, (num >> 22) & 1, (num >> 21) & 1, (num >> 20) & 1,
            (num >> 19) & 1, (num >> 18) & 1, (num >> 17) & 1, (num >> 16) & 1,
            (num >> 15) & 1, (num >> 14) & 1, (num >> 13) & 1, (num >> 12) & 1,
            (num >> 11) & 1, (num >> 10) & 1, (num >>  9) & 1, (num >>  8) & 1,
            (num >>  7) & 1, (num >>  6) & 1, (num >>  5) & 1, (num >>  4) & 1,
            (num >>  3) & 1, (num >>  2) & 1, (num >>  1) & 1,  num & 1);
}

#ifdef USERMODE // habals
extern const char *GetArg(void);

//these dont exist on windows
//but defined in efi/strings
#if defined(WIN32)
extern char *strsep(char **stringp, const char *delim);
extern char *win_strdup(const char *s);
#undef strdup
#define strdup win_strdup
#endif
#endif

//---------------------------------------------------
// chex
//
// ---------------------------------------------------
DECLARE_API( chex )
{
#if defined(USERMODE) // habals
    char *c_arg;

    while((c_arg = (char*) GetArg()) != NULL)
    {
        //
        // allow value:first_bit-last_bit
        // like 20:5:4
        //
        if(strchr(c_arg, ':') != NULL) {
            char *c_copy = strdup(c_arg);
            char *c_ptr, *c_num, *c_hi, *c_lo;

            c_ptr = c_copy;
            c_num = strsep(&c_ptr, ":");
            c_hi = strsep(&c_ptr, ":");
            c_lo = strsep(&c_ptr, ":");

            if(c_num != NULL &&
               c_hi != NULL &&
               c_lo != NULL)
            {
                LwU32 num = strtoul(c_num, NULL, 16);
                LwU32 hi = strtoul(c_hi, NULL, 0);
                LwU32 lo = strtoul(c_lo, NULL, 0);

                // allow either number:4:3  or  number:3:4
                if(hi >= lo)
                    num = REF_VAL(hi:lo, num);
                else
                    num = REF_VAL(lo:hi, num);

                display_number(num);
            }
            else
                dprintf("err: arg %s invalid, try something like 0xABC:7:4 or 0xABC:4:7\n");

            free(c_copy);
        } else {
            LwU32 num = strtoul(c_arg, NULL, 16);
            display_number(num);
        }
    }
#endif
}

//---------------------------------------------------
// cdec
//
// ---------------------------------------------------
DECLARE_API( cdec )
{
#if defined(USERMODE) // habals
   char *c_arg;

    while((c_arg = (char*) GetArg()) != NULL)
    {

        // allow value:first_bit-last_bit
        // like 20:5:4

        if(strchr(c_arg, ':') != NULL) {
            char *c_copy = strdup(c_arg);
            char *c_ptr, *c_num, *c_hi, *c_lo;

            c_ptr = c_copy;
            c_num = strsep(&c_ptr, ":");
            c_hi = strsep(&c_ptr, ":");
            c_lo = strsep(&c_ptr, ":");

            if(c_num != NULL &&
               c_hi != NULL &&
               c_lo != NULL)
            {
                LwU32 num = strtoul(c_num, NULL, 0);
                LwU32 hi = strtoul(c_hi, NULL, 0);
                LwU32 lo = strtoul(c_lo, NULL, 0);

                // allow either number:4:3  or  number:3:4
                if(hi >= lo)
                    num = REF_VAL(hi:lo, num);
                else
                    num = REF_VAL(lo:hi, num);

                display_number(num);
            }
            else
                dprintf("err: arg %s invalid, try something like 0xABC:7:4 or 0xABC:4:7\n");

            free(c_copy);
        } else {
            LwU32 num = strtoul(c_arg, NULL, 0);
            display_number(num);
        }
    }
#endif // USERMODE && EFI
}

//-----------------------------------------------------
// dhdorconn
// - prints out what OR each head is driving
//-----------------------------------------------------
DECLARE_API( dhdorconn )
{
    BOOL ascii=FALSE;
    char *usage ="dhdorconn [-ascii]";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    if (dArgc > 0)
    {
        if (parseCmd(args, "ascii", 0, NULL))
            ascii=TRUE;
        else
        {
            dprintf("%s\n",usage);
            return;
        }
    }
    pDisp[indexGpu].dispHeadORConnection();
    if (ascii == TRUE)
        dispHeadORConnectionAscii();
    return;
}

//-----------------------------------------------------
// ddesc
// - prints out the ctx dma description of the specified
//   handle from display instance memory
//-----------------------------------------------------
DECLARE_API( ddesc )
{
    char *chName = NULL;
    LwU32 headNum = 0;
    LwS32 argNum, chNum = 0;
    LwS32 temp;
    LwU32 handle = 0;
    int retVal;
    BOOL searchAllHandles = TRUE;
    char *usage = "Usage: ddesc [handle] [channelName] [-h<0|1>]\n";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc > 3)
    {
        dprintf("Incorrect number of arguments to %s. Bailing out early\n", __FUNCTION__);
        dprintf("%s\n", usage);
        return;
    }

    for (argNum = 0; argNum < dArgc; ++argNum)
    {
        if (sscanf(dArgv[argNum], "-h%d", &temp))
        {
            headNum = temp;
        }
        else if ((chNum = pDisp[indexGpu].dispGetChanNum(dArgv[argNum], headNum)) != -1)
        {
            chName = dArgv[argNum];
        }
        else
        {
            retVal = sscanf(dArgv[argNum], "%x", &handle);
            if ((retVal == 0) || (retVal == EOF))
            {
                dprintf("Invalid argument %s. Bailing out early\n", dArgv[argNum]);
                return;
            }
            else
            {
                searchAllHandles = FALSE;
            }
        }
    }

    if (chName != NULL)
    {
        if (strcmp(chName, "core") == 0)
        {
            headNum = 0;
        }

        chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum);
        if (chNum == -1)
        {
            dprintf("Bad head argument %d.\n", headNum);
            return;
        }
    }
    else
    {
            chNum = -1;
    }
    pDisp[indexGpu].dispCtxDmaDescription(handle, chNum, searchAllHandles);
}

//-----------------------------------------------------
// dpauxrd
// - performs DPAUX read transaction.
//-----------------------------------------------------
LwBool dpDecodeAux(LwU32 addr, LwU8 *data, LwU32 length, LwU8 version);
DECLARE_API( dpauxrd )
{
    LwU32 port, addr, length;
    LwU8 *data;
    DP_RELATIVE_ADDRESS dpRelAddr;
    LwBool decode;
    char *usage = "Usage: dpauxrd [-d] <physical port>"
                  "[.concat port 1][.concat port 2][...] "
                  "<DPCD offset> [length]\n";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    decode = !strcmp(dArgv[0], "-d");

    if ((!decode && dArgc != 2 && dArgc != 3) || (decode && dArgc != 3 && dArgc != 4))
    {
        dprintf("Incorrect number of arguments to %s. Bailing out early\n",
                __FUNCTION__);
        dprintf("%s\n", usage);
        return;
    }

    if (!dpmsgGetPortAddressFromText(&port, &dpRelAddr, dArgv[decode ? 1 : 0]))
    {
        dprintf("Port is invalid in %s. Bailing out early\n", __FUNCTION__);
        return;
    }
    sscanf(dArgv[decode ? 2 : 1], "%x", &addr);

    if (dArgc == (decode ? 4 : 3))
        sscanf(dArgv[decode ? 3 : 2], "%x", &length);
    else
        length = 1;

    if ((addr + length - 1) > MAX_DP_AUX_ADDRESS)
    {
        dprintf("Address to read is out of bound in %s. Bailing out early\n",
                __FUNCTION__);
        return;
    }

    data = (LwU8 *)malloc(length);
    if (data == NULL)
    {
        dprintf("Failed to alloc resource in %s. Bailing out early\n",
                __FUNCTION__);
    }

    if (length == dpmsgDpcdRead(port, &dpRelAddr, addr, data, length))
    {
        LwU32 i, j, k;
        // print 16 bytes per line.
        for (i = 0; i < length;)
        {
            if (length - i > 16)
                j = 16;
            else
                j = length - i;

            dprintf("#%05x ", addr + i);
            for (k = 0; k < j; k++)
                dprintf("%02x ", data[i + k]);
            dprintf("\n");
            i += j;
        }
    }

    if (decode)
    {
        LwU8 version;
        if (1 == dpmsgDpcdRead(port, &dpRelAddr, 0, &version, 1))
        {
            dpDecodeAux(addr, data, length, version);
        }
        else
        {
            dprintf("DP version could not be retreived. Some labels may be incorrect.\n");
            dpDecodeAux(addr, data, length, 0);
        }
    }

    free(data);
}

//-----------------------------------------------------
// dpauxwr
// - performs DPAUX write transaction.
//-----------------------------------------------------
DECLARE_API( dpauxwr )
{
    LwU32 port, addr, length, scanData;
    LwU8 *data, i;
    DP_RELATIVE_ADDRESS dpRelAddr;
    char *usage = "Usage: dpauxwr <physical port>"
                  "[.concat port 1][.concat port 2][...] "
                  "<DPCD offset> <data 1> [data 2] ...\n";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    if (dArgc < 3)
    {
        dprintf("Incorrect number of arguments to %s. Bailing out early\n",
                __FUNCTION__);
        dprintf("%s\n", usage);
        return;
    }

    if (!dpmsgGetPortAddressFromText(&port, &dpRelAddr, dArgv[0]))
    {
        dprintf("Port is invalid in %s. Bailing out early\n", __FUNCTION__);
        return;
    }
    sscanf(dArgv[1], "%x", &addr);

    // Data to write begins from 3rd arg
    length = dArgc - 2;
    if ((addr + length - 1) > MAX_DP_AUX_ADDRESS)
    {
        dprintf("Address to write is out of bound in %s. Bailing out early\n",
                __FUNCTION__);
        return;
    }

    data = (LwU8*)malloc(length);
    if (data == NULL)
    {
        dprintf("Failed to alloc resource in %s. Bailing out early\n",
                __FUNCTION__);
        return;
    }
    for (i = 0; i < length; i++)
    {
        sscanf(dArgv[2 + i], "%x", &scanData);
        data[i] = (LwU8)scanData;
    }

    dpmsgDpcdWrite(port, &dpRelAddr, addr, data, length);
    free(data);
}

//-------------------------------------------------------
// dpinfo
// - Prints out display port source and sink device info.
//-------------------------------------------------------
DECLARE_API( dpinfo )
{
    LwS32 port, sorIndex, dpIndex;
    char *usage = "Usage: dpinfo [physical port] [<sorIndex> <dpIndex>]\n";

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    switch (dArgc)
    {
        case 0:
            pDisp[indexGpu].dispDisplayPortEnum();
            dprintf("\n%s\n", usage);
            return;
        case 1:
            sscanf(dArgv[0], "%x", &port);
            sorIndex = 0;
            dpIndex = LW_MAX_SUBLINK;
            break;
        case 3:
            sscanf(dArgv[0], "%x", &port);
            sscanf(dArgv[1], "%x", &sorIndex);
            sscanf(dArgv[2], "%x", &dpIndex);
            break;
        default:
            dprintf("Incorrect number of arguments to %s. Bailing out early\n"
                    "%s\n", __FUNCTION__, usage);
            return;
    }

    pDisp[indexGpu].dispDisplayPortInfo(port, sorIndex, dpIndex);
}

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(CLIENT_SIDE_RESMAN)

//-------------------------------------------------------
// br04init
//
//-------------------------------------------------------
DECLARE_API( br04init )
{
    LwU32 PcieConfigSpaceBase;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc < 1 || !sscanf(dArgv[0],"%i", &PcieConfigSpaceBase))
    {
        PcieConfigSpaceBase = 0xe0000000;
    }

    dprintf("Using 0x%08x as PcieConfigSpaceBase\n", PcieConfigSpaceBase);

    br04Init(PcieConfigSpaceBase);
}

//-------------------------------------------------------
// br04topology
//
//-------------------------------------------------------
DECLARE_API( br04topology )
{
    CHECK_INIT(MODE_LIVE);

    br04DisplayTopology();
}

//-------------------------------------------------------
// br04dump
//
//-------------------------------------------------------
DECLARE_API( br04dump )
{
    LwU32 bid;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc < 1 || !sscanf(dArgv[0],"%d", &bid))
    {
        dprintf("Invalid Board ID\n");
        return;
    }

    br04DumpBoard(bid);
}

//-------------------------------------------------------
// br04port
//
//-------------------------------------------------------
DECLARE_API( br04port )
{
    LwU32 bid, portid;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    if (dArgc < 1 || !sscanf(dArgv[0],"%d", &bid))
    {
        dprintf("Invalid Board ID\n");
        return;
    }

    if (dArgc < 2 || !sscanf(dArgv[1],"%d", &portid) || portid >= 4)
    {
        dprintf("Invalid Port ID\n");
        return;
    }

    br04DumpPort(bid, portid);
}

#endif

//-----------------------------------------------------
//  A built-in help for the extension dll
//-----------------------------------------------------
DECLARE_API ( help )
{
    printHelpMenu();
}

//---------------------------------------------------------
// dpudmemrd <offset> [length(bytes)] [port] [size(bytes)]
// - read the DMEM in the range offset-offset+length
//---------------------------------------------------------
DECLARE_API( dpudmemrd )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32  engineBase      = 0x0;
    LwU64  offset          = 0x0;
    LwU64  lengthInBytes   = 0x80;
    LwU64  port            = 0x3;
    LwU64  size            = 0x4;
    LwU32  memSize         = 0x0;
    LwU32  numPorts        = 0x0;
    LwU32  length          = 0x0;
    LwU32* buffer          = NULL;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.dpudmemrd <offset> [length(bytes)] " \
                "[port] [size]\n");
        return;
    }

    if (args[0] == '\0')
    {
        dprintf("lw: Usage: !lw.dpudmemrd <offset> [length(bytes)] " \
                "[port] [size]\n");
        dprintf("lw: No args specified, defaulted to offset" \
                                " 0x%04x and length 0x%04x bytes.\n",
                                 (LwU32)offset, (LwU32)lengthInBytes);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetSafeExpressionEx(args, &offset, &args))
    {
        if (GetSafeExpressionEx(args, &lengthInBytes, &args))
        {
            if (GetSafeExpressionEx(args, &port, &args))
            {
                GetSafeExpressionEx(args, &size, &args);
            }
        }
    }

    // Tidy up the length to be 4-byte aligned
    lengthInBytes = (lengthInBytes + 3) & ~3ULL;
    offset = offset & ~3ULL;


    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pDpu[indexGpu].dpuGetFalconCoreIFace();
        pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnDmemGetSize(engineBase);
        numPorts   = pFCIF->flcnDmemGetNumPorts(engineBase);
        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%x is too large (DMEM size 0x%x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the DMEM

        length = pFCIF->flcnDmemRead(engineBase,
                                       (LwU32)offset,
                                       LW_FALSE,
                                       (LwU32)lengthInBytes / sizeof(LwU32),
                                       (LwU32)port, buffer);

        // Dump out the DMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping DMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, (LwU8)size);
        }

        // Cleanup after ourselves
        free(buffer);
    }
    MGPU_LOOP_END;
}

//--------------------------------------------------------------------------------------------
// dpudmemwr <offset> <value> [-w width(bytes)] [length(units of width)] [-p <port>] [-s <size>]
// - write 'value' of 'width' to DMEM in the range offset-offset+length
//--------------------------------------------------------------------------------------------
DECLARE_API( dpudmemwr )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32   engineBase    = 0x0;
    LwU64   offset        = 0x0;
    LwU64   length        = 0x1;
    LwU64   width         = 0x4;
    LwU64   value         = 0x0;
    LwU64   port          = 0x3;
    LwU64   size          = 0x4;
    LwU32   memSize       = 0x0;
    LwU32   numPorts      = 0x0;
    LwU32   bytesWritten  = 0x0;
    LwU32   endOffset     = 0x0;
    char   *pParams;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Offset not specified.\n");
        dprintf("lw: Usage: !lw.dpudmemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }

    // extract optional arguments first
    if (parseCmd(args, "w", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &width, &pParams);
    }

    if (parseCmd(args, "l", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &length, &pParams);
    }

    if (parseCmd(args, "p", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &port, &pParams);
    }

    if (parseCmd(args, "s", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &size, &pParams);
    }

    // Read in the <offset> <value>, in that order, if present
    if (!GetSafeExpressionEx(args, &offset, &args))
    {
        dprintf("lw: Value not specified.\n");
        dprintf("lw: Usage: !lw.dpudmemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }
    GetSafeExpressionEx(args, &value, &args);

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pDpu[indexGpu].dpuGetFalconCoreIFace();
        pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize  = pFCIF->flcnDmemGetSize(engineBase);
        numPorts = pFCIF->flcnDmemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the DMEM

        dprintf("lw:\tWriting DMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at DMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pFCIF->flcnDmemWrite(engineBase,
                                              (LwU32)offset,
                                              LW_FALSE,
                                              (LwU32)value,
                                              (LwU32)width,
                                              (LwU32)length,
                                              (LwU32)port);

        if (bytesWritten == ((LwU32)width * (LwU32)length))
        {
            dprintf("lw:\n");
            endOffset = (LwU32)offset + ((LwU32)width * (LwU32)length);
            flcnDmemDump(pFEIF,
                         (LwU32)offset & ~0x3,
                         endOffset - ((LwU32)offset & ~0x3),
                         (LwU8)port,
                         (LwU8)size);
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dpuimemrd <offset> [length(bytes)] [port]
// - read the IMEM in the range offset-offset+length
//-----------------------------------------------------
DECLARE_API( dpuimemrd )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32    engineBase      = 0x0;
    LwU64    offset          = 0x0;
    LwU64    lengthInBytes   = 0x80;
    LwU64    port            = 0x0;
    LwU32    memSize         = 0x0;
    LwU32    numPorts        = 0x0;
    LwU32    length          = 0x0;
    LwU32   *buffer          = NULL;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.dpuimemrd <offset> [length(bytes)] [port]\n");
        return;
    }

    if (args[0] == '\0')
    {
        dprintf("lw: Usage: !lw.dpuimemrd <offset> [length(bytes)] [port]\n");
        dprintf("lw: No args specified, defaulted to offset" \
                                " 0x%04x and length 0x%04x bytes.\n",
                                 (LwU32)offset, (LwU32)lengthInBytes);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetSafeExpressionEx(args, &offset, &args))
    {
        if (GetSafeExpressionEx(args, &lengthInBytes, &args))
        {
            GetSafeExpressionEx(args, &port, &args);
        }
    }

    // Tidy up the length and offset to be 4-byte aligned
    lengthInBytes   = (lengthInBytes + 3) & ~3ULL;
    offset          = offset & ~3ULL;

    // Get the size of the IMEM and number of IMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pDpu[indexGpu].dpuGetFalconCoreIFace();
        pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnImemGetSize(engineBase);
        numPorts   = pFCIF->flcnImemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%04x is too large (IMEM size 0x%04x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the IMEM
        length = pFCIF->flcnImemRead(engineBase,
                                       (LwU32)offset,
                                       (LwU32)lengthInBytes / sizeof(LwU32),
                                       (LwU32)port, buffer);

        // Dump out the IMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping DPU IMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, 0x4);
        }

        // Cleanup after ourselves
        free((void*)buffer);
    }
    MGPU_LOOP_END;
}

//--------------------------------------------------------------------------------------------
// dpuimemwr <offset> <value> [width(bytes)] [length(units of width)] [-p <port>]
// - write 'value' to IMEM in the range offset-offset+length
//--------------------------------------------------------------------------------------------
DECLARE_API( dpuimemwr )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32  engineBase    = 0x0;
    LwU64  offset        = 0x0;
    LwU64  length        = 0x1;
    LwU64  width         = 0x4;
    LwU64  value         = 0x0;
    LwU64  port          = 0x0;
    LwU32  memSize       = 0x0;
    LwU32  numPorts      = 0x0;
    LwU32  bytesWritten  = 0x0;
    char  *pParams;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Offset not specified.\n");
        dprintf("lw: Usage: !lw.dpuimemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>]\n");
        return;
    }

    // extract optional arguments first
    if (parseCmd(args, "w", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &width, &pParams);
    }

    if (parseCmd(args, "l", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &length, &pParams);
    }

    if (parseCmd(args, "p", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &port, &pParams);
    }

    // Read in the <offset> <value>, in that order, if present
    if (!GetSafeExpressionEx(args, &offset, &args))
    {
        dprintf("lw: Value not specified.\n");
        dprintf("lw: Usage: !lw.dpuimemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>]\n");
        return;
    }
    GetSafeExpressionEx(args, &value, &args);

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pDpu[indexGpu].dpuGetFalconCoreIFace();
        pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnImemGetSize(engineBase);
        numPorts   = pFCIF->flcnImemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the IMEM

        dprintf("lw:\tWriting IMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at IMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pFCIF->flcnImemWrite(
                           engineBase,
                           (LwU32)offset,
                           (LwU32)value,
                           (LwU32)width,
                           (LwU32)length,
                           (LwU32)port);


        dprintf("lw: number of bytes written: 0x%x\n", bytesWritten);
    }
    MGPU_LOOP_END;
}


//-----------------------------------------------------
// @brief dpuqueues  [queue id]
//        read the DPU queues or a specified queue
//
// A specific queue ID may be specified.
//-----------------------------------------------------
DECLARE_API( dpuqueues )
{
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    BOOL            bListAll  = TRUE;
    LwU64           userId    = ~0x0;
    LwU32           queueId   = 0x0;
    LwU32           numQueues = 0x0;
    FLCN_QUEUE      queue;
    LwU32 numCmdQs = 0x0;
    LwU32 numMsgQs = 0x0;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.dpuqueues [queue id]\n");
        return;
    }

    GetSafeExpressionEx(args, &userId, &args);
    bListAll = (userId == (LwU64)(~0));

    MGPU_LOOP_START;
    {
        pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
        numQueues = pFEIF->flcnEngQueueGetNum();
        if (!bListAll)
        {
            if ((LwU32)userId >= numQueues)
            {
                dprintf("lw: %s: 0x%x is not a valid queue ID (max 0x%x)\n",
                        __FUNCTION__, (LwU32)userId, numQueues - 1);
                return;
            }
            queueId   = (LwU32)userId;
            numQueues = queueId + 1;
        }
        else
        {
            queueId = 0;
        }

        numCmdQs = pDpu[indexGpu].dpuGetCmdQNum();
        numMsgQs = pDpu[indexGpu].dpuGetMsgQNum();
        dprintf("lw: numCmdQs = %d, numMsgQs = %d\n", numCmdQs, numMsgQs);

        for (; queueId < numQueues; queueId++)
        {
            if (pFEIF->flcnEngQueueRead(queueId, &queue))
            {
                if (queueId < numCmdQs)
                {
                    flcnQueueDump(FALSE, &queue, "DPU CMD");
                }
                else
                {
                    flcnQueueDump(FALSE, &queue, "DPU MSG");
                }
            }
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;

}

//-----------------------------------------------------
// dpuimblk [block index]
//        List all current block status.
//
// If no block index is specified, the status for all blocks
// is printed. Otherwise only the specified block status is
// printed. An invalid block index results in an error
// message.
//-----------------------------------------------------
DECLARE_API( dpuimblk )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32       engineBase          = 0x0;
    BOOL        bListAll            = TRUE;
    LwU64       userIndex           = ~0x0;
    LwU32       blockIndex          = 0x0;
    LwU32       numBlocks           = 0x0;
    LwU32       tagWidth            = 0x0;
    FLCN_BLOCK  blockInfo;


    CHECK_INIT(MODE_LIVE);

    // Get the block index
    GetSafeExpressionEx(args, &userIndex, &args);
    bListAll    = (userIndex == (LwU64)(~0));
    blockIndex  = (LwU32)userIndex;

    MGPU_LOOP_START;
    {
        pFCIF       = pDpu[indexGpu].dpuGetFalconCoreIFace();
        pFEIF       = pDpu[indexGpu].dpuGetFalconEngineIFace();
        engineBase  = pFEIF->flcnEngGetFalconBase();
        numBlocks   = pFCIF->flcnImemGetNumBlocks(engineBase);
        tagWidth    = pFCIF->flcnImemGetTagWidth(engineBase);


        if (!bListAll)
        {
            if (blockIndex >= numBlocks)
            {
                dprintf("lw: Block index 0x%x is invalid (max 0x%x).\n",
                        blockIndex, numBlocks - 1);
                return;
            }
            else
            {
                numBlocks = blockIndex + 1;
            }
        }
        else
        {
            blockIndex = 0;
        }

        dprintf("lw:\tDumping IMEM code block status\n");
        dprintf("lw:\t--------------------------------------------------\n");

        // Loop through all the blocks dumping out information
        for (; blockIndex < numBlocks; blockIndex++)
        {
            if (pFCIF->flcnImemBlk(engineBase, blockIndex, &blockInfo))
            {
                dprintf("lw:\tBlock 0x%02x: tag=0x%02x, valid=%d, pending=%d, "
                        "secure=%d\n",
                        blockIndex, blockInfo.tag, blockInfo.bValid,
                        blockInfo.bPending, blockInfo.bSelwre);
            }

        }

        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dpuimmap [-s (skip unmapped blocks)] [start tag] [end tag]
//
// Print out the tag to block mappings starting from
// start tag and until end tag (inclusive). This provides
// a way to see the mapping of "virtual" IMEM into physical
// IMEM code blocks.
//
// If no inputs are given, then we will try to automatically
// determine the area of "virtual" tag memory to print out.
//-----------------------------------------------------
DECLARE_API( dpuimmap )
{
    const   FLCN_CORE_IFACES    *pFCIF  = NULL;
    const   FLCN_ENGINE_IFACES  *pFEIF  = NULL;
    extern  POBJFLCN            thisFlcn;
    POBJFLCN        pTmpFlcn            = NULL;
    LwU32           engineBase          = 0x0;
    LwU32           maxTag              = 0x0;
    LwU32           minTag              = ~0x0;
    LwU32           tagWidth            = 0x0;
    LwU32           maxAllowedTag       = 0x0;
    LwU64           userMax             = 0x0;
    LwU64           userMin             = 0x0;
    LwU32           numBlocks           = 0x0;
    LwU32           blockIndex          = 0x0;
    BOOL            bArguments          = FALSE;
    FLCN_BLOCK      blockInfo;
    POBJFLCN        pOrigFlcn           = NULL;
    BOOL            bSkipUnmappedTags   = FALSE;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "s", 0, NULL))
    {
        bSkipUnmappedTags = TRUE;
    }

    // Get the pretty arguments
    if (GetSafeExpressionEx(args, &userMin, &args))
    {
        bArguments  = TRUE;
        GetSafeExpressionEx(args, &userMax, &args);
        maxTag      = (LwU32)userMax;
        minTag      = (LwU32)userMin;
    }

    // Make sure the argument make some sense
    if (bArguments && (minTag > maxTag))
    {
        dprintf("lw: \"start tag\" must be smaller than \"end tag\"\n");
        dprintf("lw: Usage: !lw.dpuimmap [-s] [start tag] [end tag]\n");
        return;
    }

    MGPU_LOOP_START;
    {
        pFCIF       = pDpu[indexGpu].dpuGetFalconCoreIFace();
        pFEIF       = pDpu[indexGpu].dpuGetFalconEngineIFace();
        engineBase  = pFEIF->flcnEngGetFalconBase();
        numBlocks   = pFCIF->flcnImemGetNumBlocks(engineBase);
        tagWidth    = pFCIF->flcnImemGetTagWidth(engineBase);
        maxAllowedTag   = BIT(tagWidth) - 1;

        //
        // Loop through the block mappings looking for max/min tag
        // Only do this if the user did not specify a start/end tag
        //
        if (!bArguments)
        {
            for (blockIndex = 0; blockIndex < numBlocks; blockIndex++)
            {
                if (pFCIF->flcnImemBlk(engineBase, blockIndex, &blockInfo))
                {
                    if (blockInfo.tag < minTag)
                        minTag = blockInfo.tag;
                    if (blockInfo.tag > maxTag)
                        maxTag = blockInfo.tag;
                }
            }
        }
        else
        {
            if ((maxTag > maxAllowedTag) || (minTag > maxAllowedTag))
            {
                dprintf("lw: %s: Tag was larger than maximum allowed (max 0x%02x)\n",
                        __FUNCTION__, maxAllowedTag);
                return;
            }
        }

        pTmpFlcn = thisFlcn;
        thisFlcn = dpuGetFalconObject();

        if (!thisFlcn)
        {
            dprintf("lw: Falcon object is not supported, nothing can be dumped\n");
            thisFlcn = pTmpFlcn;
            return;
        }

        flcnImemMapDump(minTag, maxTag, bSkipUnmappedTags);

        thisFlcn = pTmpFlcn;
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dpuimtag <code address>
//        List status of block mapped to address.
//
// Block address, or "code addres", will be used to find
// an IMEM code block mapping.
//-----------------------------------------------------
DECLARE_API( dpuimtag )
{
    const FLCN_CORE_IFACES   *pFCIF         = NULL;
    const FLCN_ENGINE_IFACES *pFEIF         = NULL;
    LwU32       engineBase                  = 0x0;
    LwU64       addr                        = 0x0;
    LwU32       tagWidth                    = 0x0;
    LwU32       maxAddr                     = 0x0;
    FLCN_TAG    tagInfo;

    CHECK_INIT(MODE_LIVE);

    if (!args || args[0] == '\0')
    {
        dprintf("lw:\tUsage: !lw.dpuimtag <code addr>\n");
        return;
    }

    // Get the block address to look up
    GetSafeExpressionEx(args, &addr, &args);

    MGPU_LOOP_START;
    {
        //
        // Get the tag width and notify user when code address entered
        // is larger than the maximum allowed by tag width.
        // Note that bits 0-7 are offsets into the block and 8-7+TagWidth
        // define the tag bits.
        //
        pFCIF       = pDpu[indexGpu].dpuGetFalconCoreIFace();
        pFEIF       = pDpu[indexGpu].dpuGetFalconEngineIFace();
        engineBase  = pFEIF->flcnEngGetFalconBase();
        tagWidth    = pFCIF->flcnImemGetTagWidth(engineBase);


        maxAddr = BIT(tagWidth+8) - 1;

        if (addr > maxAddr)
        {
            dprintf("lw: %s: Address 0x%04x is too large (max 0x%04x)\n",
                    __FUNCTION__, (LwU32)addr, maxAddr);
            return;
        }

        dprintf("lw: Dumping block info for address 0x%x, Tag=0x%x\n",
                (LwU32)addr, ((LwU32)addr) >> 8);

        if (pFCIF->flcnImemTag(engineBase, (LwU32)addr, &tagInfo))
        {
            switch (tagInfo.mapType)
            {
            case FALCON_TAG_UNMAPPED:
                dprintf("lw:\tTag 0x%02x: Not mapped to a block", (LwU32)addr >> 8);
                break;
            case FALCON_TAG_MULTI_MAPPED:
            case FALCON_TAG_MAPPED:
                dprintf("lw:\tTag 0x%02x: block=0x%02x, valid=%d, pending=%d, secure=%d",
                        tagInfo.blockInfo.tag, tagInfo.blockInfo.blockIndex,
                        tagInfo.blockInfo.bValid, tagInfo.blockInfo.bPending,
                        tagInfo.blockInfo.bSelwre);
                break;
            }
            if (tagInfo.mapType == FALCON_TAG_MULTI_MAPPED)
            {
                dprintf(" (multiple)");
            }
            dprintf("\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dpumutex [mutex id]
//        dump the contents of DPU mutices or a specific mutex
//
// A specific mutex Id may be specified.
//-----------------------------------------------------
DECLARE_API( dpumutex )
{
    BOOL    bListAll    = TRUE;
    BOOL    bFree       = FALSE;
    LwU64   lockId      = ~0;
    LwU32   id          =  0;
    LwU32   mutex       =  0;
    LwU32   numLocks    =  0;

    CHECK_INIT(MODE_LIVE);

    GetSafeExpressionEx(args, &lockId, &args);
    bListAll = (lockId == (LwU64)(~0));

    MGPU_LOOP_START;
    {
        numLocks = pDpu[indexGpu].dpuMutexGetNum();
        if (numLocks == 0)
        {
            dprintf("lw:\tDpu mutex is lwrrently not supported on this gpu.\n");
            return;
        }
        if (!bListAll)
        {
            id          = (LwU32)lockId;
            numLocks    = id + 1;
        }
        for (; id < numLocks; id++)
        {
            if (pDpu[indexGpu].dpuMutexRead(id, &mutex, &bFree))
            {
                dprintf("lw:\tMutex 0x%x: 0x%08x - %s\n", id, mutex,
                        bFree ? "Available" : "Taken");
            }
            else
            {
                dprintf("lw:\tMutex 0x%x is invalid.\n", id);
            }
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dputcb <tcbAddress> [port] [size] [options]
//    Dump the contents of the DPU TCB at address
//    'tcbAddress' including the full stack.
//    '-a' or '-l' to list info on all TCBs
//    '-v' for verbose output when using '-a' or '-l'
//-----------------------------------------------------
DECLARE_API( dputcb )
{
    extern POBJFLCN thisFlcn;
    FLCN_RTOS_TCB  tcb;
    POBJFLCN pTmpFlcn   = NULL;
    LwU64    tcbAddress = 0x0;
    LwU64    size       = 0x4;
    LwU64    port       = 0x1;
    BOOL     bVerbose   = FALSE;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Usage: !lw.dputcb [-v (verbose)] "
                "[-c] [-a] [-l] [tcbAddress] [port] [size].\n");
        return;
    }

    if (parseCmd(args, "v", 0, NULL))
    {
        bVerbose = TRUE;
    }

    // Handle the "-c" case for current TCB.
    if (parseCmd(args, "c", 0, NULL))
    {
        if (GetSafeExpressionEx(args, &port, &args))
        {
            GetSafeExpressionEx(args, &size, &args);
        }
        MGPU_LOOP_START;
        {
            pTmpFlcn = thisFlcn;
            thisFlcn = dpuGetFalconObject();
            if (thisFlcn && thisFlcn->pFCIF)
            {
                if (!thisFlcn->bSymLoaded)
                {
                    dprintf("lw: symbol not loaded, loading automatically...\n");
                    flcnExec("load", thisFlcn);
                }
                if (flcnRtosTcbGetLwrrent(&tcb, (LwU32)port))
                {
                    dprintf("lw: dumping current TCB\n");
                    flcnRtosTcbDump(&tcb, FALSE, (LwU32)port, (LwU8)size);
                }
            }
            thisFlcn = pTmpFlcn;
        }
        MGPU_LOOP_END;
        return;
    }

    // Handle the "-a", or "-l" case for all TCBs
    if ((parseCmd(args, "a", 0, NULL)) ||
        (parseCmd(args, "l", 0, NULL)))
    {
        if (GetSafeExpressionEx(args, &port, &args))
        {
            GetSafeExpressionEx(args, &size, &args);
        }
        MGPU_LOOP_START;
        {
            pTmpFlcn = thisFlcn;
            thisFlcn = dpuGetFalconObject();
            if (thisFlcn && thisFlcn->pFCIF)
            {
                if (!thisFlcn->bSymLoaded)
                {
                    dprintf("lw: symbol not loaded, loading automatically...\n");
                    flcnExec("load", thisFlcn);
                }
                dprintf("lw: dumping TCBs\n");
                flcnRtosTcbDumpAll(!bVerbose);
            }
            thisFlcn = pTmpFlcn;
        }
        MGPU_LOOP_END;
        return;
    }

    // For a specific tcb address
    if (GetSafeExpressionEx(args, &tcbAddress, &args))
    {
        if (GetSafeExpressionEx(args, &port, &args))
        {
            GetSafeExpressionEx(args, &size, &args);
        }
    }
    MGPU_LOOP_START;
    {
        pTmpFlcn = thisFlcn;
        thisFlcn = dpuGetFalconObject();
        if (thisFlcn && thisFlcn->pFCIF)
        {
            flcnRtosTcbGet((LwU32)tcbAddress, 0, &tcb);
            dprintf("lw: dumping tcb at address 0x%x\n", (LwU32)tcbAddress);
            flcnRtosTcbDump(&tcb, FALSE, (LwU32)port, (LwU8)size);
        }
        thisFlcn = pTmpFlcn;
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dpusym
//
//---------------------------------------------------------
DECLARE_API( dpusym )
{
    extern POBJFLCN thisFlcn;
    POBJFLCN    pTmpFlcn    = NULL;
    char        *pFilename  = NULL;
    char        *pParams;
    BOOL        bIgnoreCase = FALSE;
    LwU64       address     = 0x0;
    FLCN_SYM    *pMatches   = NULL;
    LwU32       ucodeVersion= 0x0;
    LwU32       i           = 1;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Usage: !dpusym [-l [fileName]] [-u]"
                "[-n <addr>] [-i]");
        return;
    }

    if (parseCmd(args, "l", 1, &pFilename))
    {
        MGPU_LOOP_START;
        {
            pTmpFlcn = thisFlcn;
            thisFlcn = dpuGetFalconObject();
            if (thisFlcn && thisFlcn->pFCIF)
            {
                ucodeVersion = thisFlcn->pFCIF->\
                                flcnUcodeGetVersion(thisFlcn->engineBase);
                flcnSymLoad(pFilename, ucodeVersion);
            }
            thisFlcn = pTmpFlcn;
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "l", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            pTmpFlcn = thisFlcn;
            thisFlcn = dpuGetFalconObject();
            if (thisFlcn)
            {
                flcnExec("load", thisFlcn);
            }
            thisFlcn = pTmpFlcn;
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "u", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            pTmpFlcn = thisFlcn;
            thisFlcn = dpuGetFalconObject();
            if (thisFlcn)
            {
                flcnExec("unload", thisFlcn);
            }
            thisFlcn = pTmpFlcn;
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "n", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &address, &pParams);
        MGPU_LOOP_START;
        {
            pTmpFlcn = thisFlcn;
            thisFlcn = dpuGetFalconObject();
            if (thisFlcn)
            {
                pMatches = flcnSymResolve((LwU32)address);
                if (pMatches != NULL)
                {
                    while (pMatches != NULL)
                    {
                        flcnSymPrintBrief(pMatches, i++);
                        pMatches = pMatches->pTemp;
                    }
                }
                else
                {
                    dprintf("lw: Error: no matches found. Symbol out of range?\n");
                }
            }
            thisFlcn = pTmpFlcn;
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "i", 0, NULL))
    {
        bIgnoreCase = TRUE;
    }

    if (strlen(pParams) > 0)
    {
        MGPU_LOOP_START;
        {
            pTmpFlcn = thisFlcn;
            thisFlcn = dpuGetFalconObject();
            if (thisFlcn)
            {
                flcnSymDump(pParams, bIgnoreCase);
            }
            thisFlcn = pTmpFlcn;
        }
        MGPU_LOOP_END;
        return;
    }
}

//-----------------------------------------------------
// dpusched
//
//-----------------------------------------------------
DECLARE_API( dpusched )
{
    extern POBJFLCN thisFlcn;
    POBJFLCN    pTmpFlcn    = NULL;
    BOOL        bTable      = TRUE;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "h", 0, NULL))
    {
        dprintf("lw: Usage: !lw.dpusched [-h] [-l (list format)]\n");
        return;
    }

    if (parseCmd(args, "l", 0, NULL))
    {
        bTable = FALSE;
    }

    MGPU_LOOP_START;
    {
        pTmpFlcn = thisFlcn;
        thisFlcn = dpuGetFalconObject();
        if (thisFlcn && thisFlcn->pFCIF && thisFlcn->pFEIF)
        {
            if (!thisFlcn->bSymLoaded)
            {
                dprintf("lw: symbol not loaded, loading automatically...\n");
                flcnExec("load", thisFlcn);
            }
            flcnRtosSchedDump(bTable);
        }
        thisFlcn = pTmpFlcn;
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dpuevtq
//
//-----------------------------------------------------
DECLARE_API( dpuevtq )
{
    extern POBJFLCN thisFlcn;
    POBJFLCN pTmpFlcn;
    char    *pParams;
    BOOL     bAll    = FALSE;
    char    *pSym    = NULL;
    LwU64    qAddr   = 0x00;

    // The tasks lwrrently might run on DPU
    char *pQueues[] =
    {
        "DpuMgmtCmdDispQueue"   ,
        "RegCacheQueue"         ,
        "LwDisplayQueue"        ,
        "BrightcQueue"          ,
    };

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "h", 0, NULL))
    {
        dprintf("lw: Usage: !dpuevtq [-h] [-a] [-s <symbol>] [-n <addr>]\n");
        dprintf("lw:                 - Dump out the DPU RTOS event queues\n");
        dprintf("lw:                   + '-h' : print usage\n");
        dprintf("lw:                   + '-a' : dump all info on all queues\n");
        dprintf("lw:                   + '-s' : dump info on a specific queue (identified by symbol name)\n");
        dprintf("lw:                   + '-n' : dump info on a specific queue (identified by queue address)\n");
        return;
    }

    if (parseCmd(args, "a", 0, NULL))
    {
        bAll = TRUE;
    }

    parseCmd(args, "s", 1, &pSym);

    if (parseCmd(args, "n", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &qAddr, &pParams);
    }

    MGPU_LOOP_START;
    {
        pTmpFlcn = thisFlcn;
        thisFlcn = dpuGetFalconObject();
        if (thisFlcn && thisFlcn->pFCIF && thisFlcn->pFEIF)
        {
            if (!thisFlcn->bSymLoaded)
            {
                dprintf("lw: symbol not loaded, loading automatically...\n");
                flcnExec("load", thisFlcn);
            }

            if (bAll)
            {
                flcnRtosEventQueueDumpAll(FALSE);
            }
            else if (pSym != NULL)
            {
                flcnRtosEventQueueDumpBySymbol(pSym);
            }
            else if (qAddr != 0)
            {
                flcnRtosEventQueueDumpByAddr((LwU32)qAddr);
            }
            else
            {
                flcnRtosEventQueueDumpAll(TRUE);
            }
        }
        thisFlcn = pTmpFlcn;
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------------------------------
// dpuqboot <program>
//  - DPU Quick Boot : Bootstraps a simple DPU program.
//    <program> : A simple DPU binary to bootstrap and run.
//-----------------------------------------------------------------------------
DECLARE_API( dpuqboot )
{
    const FLCN_ENGINE_IFACES *pFEIF = NULL;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Usage: !lw.dpuqboot <program> \n");
        return;
    }

    pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
    pDpu[indexGpu].dpuRst();
    flcnSimpleBootstrap(pFEIF, args);
}

//-----------------------------------------------------------------------------
// dpusanitytest [options]
//  - Runs DPU sanity test.
//    + '-t' <test#> : execute command on a single test program.
//    + '-v' <level> : verbose level. 0-3 where 0 - mute (default), 3 - noisy.
//    + '-i'         : prints description of available pmu sanity tests
//    + '-n'         : returns the number of tests available. <testnum> ignored
//-----------------------------------------------------------------------------
DECLARE_API( dpusanitytest )
{
    char     *pParams;
    LwU64     verbose      = 0;
    LwU64     testnum      = 0;
    LwU32     totalNumTest = 0;
    BOOL      bAllTest     = TRUE;
    LwU32     i;
    LW_STATUS status;

    CHECK_INIT(MODE_LIVE);

    // Handle the "-t" case
    if (parseCmd(args, "t", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &testnum, &pParams);
        bAllTest = FALSE;
    }

    // Handle the "-v" case
    if (parseCmd(args, "v", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &verbose, &pParams);
    }

    // Handle the "-n" case
    if (parseCmd(args, "n", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            dprintf("# of available DPU tests: %d\n",
                    pDpu[indexGpu].dpuSanityTestGetNum());
        }
        MGPU_LOOP_END;
        return;
    }

    // Handle the "-i" case
    if (parseCmd(args, "i", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            totalNumTest = pDpu[indexGpu].dpuSanityTestGetNum();
            dprintf("  # | Test Names  A - AUTO         D - DESTRUCTIVE X - ARG REQUIRED\n");
            dprintf("    |             O - OPTIONAL ARG P - PROD UCODE  V - VERIF UCODE\n");
            dprintf("----+-----------------------------------------------------------------\n");
            if (bAllTest)
            {
                for (i = 0; i < totalNumTest; i++)
                {
                    dprintf("%3x | %s\n", i,
                            pDpu[indexGpu].dpuSanityTestGetInfo(i, VB1));
                }
            }
            else
            {
                    dprintf("%3x | %s\n", (LwU32) testnum,
                            pDpu[indexGpu].dpuSanityTestGetInfo((LwU32) testnum, VB1));
            }
        }
        MGPU_LOOP_END;
        return;
    }

    dprintf("========================================================\n");
    dprintf("                    DPU SANITY TEST                     \n");
    dprintf("========================================================\n");
    if (bAllTest)
    {
        MGPU_LOOP_START;
        {
            totalNumTest = pDpu[indexGpu].dpuSanityTestGetNum();
            for (i = 0; i < totalNumTest; i++)
            {
                char * _errstr[] = {"PASS!", "RETRY", "FAIL!", "SKIP!"};
                LwU32  _errno;

                if (pDpu[indexGpu].dpuSanityTestGetFlags(i) & DPU_TEST_AUTO)
                {
                    status =
                        pDpu[indexGpu].dpuSanityTestRun(i, (LwU32) verbose, NULL);

                    if      (status == LW_OK)    _errno = 0;
                    else if (status == LW_ERR_BUSY_RETRY) _errno = 1;
                    else                         _errno = 2;
                }
                else
                {
                    _errno = 3;
                }

                dprintf("%2x. %42s   [%s]\n", i,
                        pDpu[indexGpu].dpuSanityTestGetInfo(i, VB0),
                        _errstr[_errno]);
            }
        }
        MGPU_LOOP_END;
    }
    else
    {
        MGPU_LOOP_START;
        {
            totalNumTest = pDpu[indexGpu].dpuSanityTestGetNum();
            if (testnum < totalNumTest)
            {
                dprintf("Running a single test : [%d] \"%s\"\n", (LwU32) testnum,
                        pDpu[indexGpu].dpuSanityTestGetInfo((LwU32) testnum, VB0));
                status = pDpu[indexGpu].dpuSanityTestRun((LwU32) testnum,
                                (LwU32) verbose, NULL);
                if (status == LW_OK)
                {
                    dprintf(":::::: PASS!!\n");
                }
                else if (status == LW_ERR_BUSY_RETRY)
                {
                    dprintf(":::::: RETRY!!\n");
                    dprintf(":::::: This is often returned when the test requires the DPU be in\n"
                            "       some specific state. Run the test with -v 2 to find out how to\n"
                            "       get around this problem.\n");
                }
                else
                {
                    dprintf(":::::: FAIL!!\n");
                }
            }
            else
            {
                dprintf("Test number %u is invalid.\n", (LwU32)testnum);
            }
        }
        MGPU_LOOP_END;
    }
}

//---------------------------------------------------------
// pmudmemrd <offset> [length(bytes)] [port] [size(bytes)]
// - read the DMEM in the range offset-offset+length
//---------------------------------------------------------
DECLARE_API( pmudmemrd )
{
    LwU64   offset          = 0x0;
    LwU64   lengthInBytes   = 0x80;
    LwU64   port            = 0x3;
    LwU64   size            = 0x4;
    LwU32   memSize         = 0x0;
    LwU32   numPorts        = 0x0;
    LwU32   length          = 0x0;
    LwU32*   buffer          = NULL;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.pmudmemrd <offset> [length(bytes)] " \
                "[port] [size]\n");
        return;
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetSafeExpressionEx(args, &offset, &args))
    {
        if (GetSafeExpressionEx(args, &lengthInBytes, &args))
        {
            if (GetSafeExpressionEx(args, &port, &args))
            {
                GetSafeExpressionEx(args, &size, &args);
            }
        }
    }

    // Tidy up the length to be 4-byte aligned
    lengthInBytes = (lengthInBytes + 3) & ~3ULL;
    offset = offset & ~3ULL;

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        memSize = pPmu[indexGpu].pmuDmemGetSize();
        numPorts = pPmu[indexGpu].pmuDmemGetNumPorts();

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%x is too large (DMEM size 0x%x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the DMEM
        length = pPmu[indexGpu].pmuDmemRead((LwU32)offset,
                                    LW_FALSE,
                                    (LwU32)lengthInBytes / sizeof(LwU32),
                                    (LwU32)port, buffer);

        // Dump out the DMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping DMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, (LwU8)size);
        }

        // Cleanup after ourselves
        free(buffer);
    }
    MGPU_LOOP_END;
}

//--------------------------------------------------------------------------------------------
// pmudmemwr <offset> <value> [width(bytes)] [length(units of width)] [-p <port>] [-s <size>]
// - write 'value' to DMEM in the range offset-offset+length
//--------------------------------------------------------------------------------------------
DECLARE_API( pmudmemwr )
{
    LwU64   offset        = 0x0;
    LwU64   length        = 0x1;
    LwU64   width         = 0x4;
    LwU64   value         = 0x0;
    LwU64   port          = 0x3;
    LwU64   size          = 0x4;
    LwU32   memSize       = 0x0;
    LwU32   numPorts      = 0x0;
    LwU32   bytesWritten  = 0;
    LwU32   endOffset     = 0;
    char   *pParams;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Offset not specified.\n");
        dprintf("lw: Usage: !lw.pmudmemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }

    // extract optional arguments first
    if (parseCmd(args, "w", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &width, &pParams);
    }

    if (parseCmd(args, "l", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &length, &pParams);
    }

    if (parseCmd(args, "p", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &port, &pParams);
    }

    if (parseCmd(args, "s", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &size, &pParams);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (!GetSafeExpressionEx(args, &offset, &args))
    {
        dprintf("lw: Value not specified.\n");
        dprintf("lw: Usage: !lw.pmudmemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }
    GetSafeExpressionEx(args, &value, &args);

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        memSize  = pPmu[indexGpu].pmuDmemGetSize();
        numPorts = pPmu[indexGpu].pmuDmemGetNumPorts();

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the DMEM
        dprintf("lw:\tWriting DMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at DMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pPmu[indexGpu].pmuDmemWrite(
                           (LwU32)offset,
                           LW_FALSE,
                           (LwU32)value,
                           (LwU32)width,
                           (LwU32)length,
                           (LwU32)port);

        if (bytesWritten == ((LwU32)width * (LwU32)length))
        {
            dprintf("lw:\n");
            endOffset = (LwU32)offset + ((LwU32)width * (LwU32)length);
            pmuDmemDump((LwU32)offset & ~0x3,
                        endOffset - ((LwU32)offset & ~0x3),
                        (LwU8)port,
                        (LwU8)size);
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// bsiramrd <offset> [length(bytes)]
// - read the BSI RAM in the range offset-offset+length
//-----------------------------------------------------
DECLARE_API( bsiramrd )
{
    LwU64   offset          = 0x0;
    LwU64   lengthInBytes   = 0x80;
    LwU32   memSize         = 0x0;
    LwU32   length          = 0x0;
    LwU32*   buffer          = NULL;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.bsiramrd <offset> [length(bytes)] \n");
        return;
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetSafeExpressionEx(args, &offset, &args))
    {
        GetSafeExpressionEx(args, &lengthInBytes, &args);
    }

    // Tidy up the length and offset to be 4-byte aligned
    lengthInBytes   = (lengthInBytes + 3) & ~3ULL;
    offset          = offset & ~3ULL;

    // Get the size of the BSI
    MGPU_LOOP_START;
    {
        memSize     = pElpg[indexGpu].elpgBsiRamSize();

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%04x is too large (BSI RAM size 0x%04x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the BSI
        length = pElpg[indexGpu].elpgBsiRamRead((LwU32)offset,
                                    (LwU32)lengthInBytes / sizeof(LwU32),
                                    buffer);

        // Dump out the BSI
        if (length > 0)
        {
            dprintf("lw:\tDumping BSI RAM from 0x%04x-0x%04x\n",
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)));
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, 0x4);
        }
        // Cleanup after ourselves
        free((void*)buffer);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmuimemrd <offset> [length(bytes)] [port]
// - read the IMEM in the range offset-offset+length
//-----------------------------------------------------
DECLARE_API( pmuimemrd )
{
    LwU64   offset          = 0x0;
    LwU64   lengthInBytes   = 0x80;
    LwU64   port            = 0x0;
    LwU32   memSize         = 0x0;
    LwU32   numPorts        = 0x0;
    LwU32   length          = 0x0;
    LwU32*   buffer          = NULL;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.pmuimemrd <offset> [length(bytes)] [port]\n");
        return;
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetSafeExpressionEx(args, &offset, &args))
    {
        if (GetSafeExpressionEx(args, &lengthInBytes, &args))
        {
            GetSafeExpressionEx(args, &port, &args);
        }
    }

    // Tidy up the length and offset to be 4-byte aligned
    lengthInBytes   = (lengthInBytes + 3) & ~3ULL;
    offset          = offset & ~3ULL;

    // Get the size of the IMEM and number of IMEM ports
    MGPU_LOOP_START;
    {
        memSize     = pPmu[indexGpu].pmuImemGetSize();
        numPorts    = pPmu[indexGpu].pmuImemGetNumPorts();

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%04x is too large (IMEM size 0x%04x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the IMEM
        length = pPmu[indexGpu].pmuImemRead((LwU32)offset,
                                    (LwU32)lengthInBytes / sizeof(LwU32),
                                    (LwU32)port, buffer);

        // Dump out the IMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping IMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, 0x4);
        }
        // Cleanup after ourselves
        free((void*)buffer);
    }
    MGPU_LOOP_END;
}

//--------------------------------------------------------------------------------------------
// pmuimemwr <offset> <value> [width(bytes)] [length(units of width)] [-p <port>] [-s <size>]
// - write 'value' to IMEM in the range offset-offset+length
//--------------------------------------------------------------------------------------------
DECLARE_API( pmuimemwr )
{
    LwU64   offset        = 0x0;
    LwU64   length        = 0x1;
    LwU64   width         = 0x4;
    LwU64   value         = 0x0;
    LwU64   port          = 0x0;
    LwU64   size          = 0x4;
    LwU32   memSize       = 0x0;
    LwU32   numPorts      = 0x0;
    LwU32   bytesWritten  = 0;
    char   *pParams;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Offset not specified.\n");
        dprintf("lw: Usage: !lw.pmuimemwr <offset> <value> [width(bytes)] "
                "[length(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }

    // extract optional arguments first
    if (parseCmd(args, "w", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &width, &pParams);
    }

    if (parseCmd(args, "l", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &length, &pParams);
    }

    if (parseCmd(args, "p", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &port, &pParams);
    }

    if (parseCmd(args, "s", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &size, &pParams);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (!GetSafeExpressionEx(args, &offset, &args))
    {
        dprintf("lw: Value not specified.\n");
        dprintf("lw: Usage: !lw.pmuimemwr <offset> <value> [width(bytes)] "
                "[length(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }
    GetSafeExpressionEx(args, &value, &args);

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        memSize  = pPmu[indexGpu].pmuImemGetSize();
        numPorts = pPmu[indexGpu].pmuImemGetNumPorts();

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the IMEM
        dprintf("lw:\tWriting IMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at IMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pPmu[indexGpu].pmuImemWrite(
                           (LwU32)offset,
                           (LwU32)value,
                           (LwU32)width,
                           (LwU32)length,
                           (LwU32)port);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// @brief pmuqueues [-p] [queue id]
//        read the PMU queues or a specified queue
//
// A specific queue ID may be specified.
//-----------------------------------------------------
DECLARE_API( pmuqueues )
{
    BOOL        bListAll  = TRUE;
    LwU64       userId    = ~0x0;
    LwU32       queueId   = 0x0;
    LwU32       numQueues = 0x0;
    FLCN_QUEUE  queue;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.pmuqueues [-p] [-u] [queue id]\n");
        return;
    }

    GetSafeExpressionEx(args, &userId, &args);
    bListAll = (userId == (LwU64)(~0));

    MGPU_LOOP_START;
    {
        numQueues = pPmu[indexGpu].pmuQueueGetNum();
        if (!bListAll)
        {
            if ((LwU32)userId >= numQueues)
            {
                dprintf("lw: %s: 0x%x is not a valid queue ID (max 0x%x)\n",
                        __FUNCTION__, (LwU32)userId, numQueues - 1);
                return;
            }
            queueId   = (LwU32)userId;
            numQueues = queueId + 1;
        }
        else
        {
            queueId = 0;
        }
        for (; queueId < numQueues; queueId++)
        {
            if (pPmu[indexGpu].pmuQueueRead(queueId, &queue))
            {
                flcnQueueDump(FALSE, &queue, "PMU");
            }
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmuimmap [start tag] [end tag]
//
// Print out the tag to block mappings starting from
// start tag and until end tag (inclusive). This provides
// a way to see the mapping of "virtual" IMEM into physical
// IMEM code blocks.
//
// If no inputs are given, then we will try to automatically
// determine the area of "virtual" tag memory to print out.
//-----------------------------------------------------
DECLARE_API( pmuimmap )
{
    LwU32       maxTag          = 0x0;
    LwU32       minTag          = ~0x0;
    LwU32       tagWidth        = 0x0;
    LwU32       maxAllowedTag   = 0x0;
    LwU64       userMax         = 0x0;
    LwU64       userMin         = 0x0;
    LwU32       numBlocks       = 0x0;
    LwU32       blockIndex      = 0x0;
    BOOL        bArguments      = FALSE;
    PmuBlock    blockInfo;
    BOOL        bSkipUnmappedTags = FALSE;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "s", 0, NULL))
    {
        bSkipUnmappedTags = TRUE;
    }

    // Get the pretty arguments
    if (GetSafeExpressionEx(args, &userMin, &args))
    {
        bArguments  = TRUE;
        GetSafeExpressionEx(args, &userMax, &args);
        maxTag      = (LwU32)userMax;
        minTag      = (LwU32)userMin;
    }

    // Make sure the argument make some sense
    if (bArguments && (minTag > maxTag))
    {
        dprintf("lw: \"start tag\" must be smaller than \"end tag\"\n");
        dprintf("lw: Usage: !lw.pmuimmap [start tag] [end tag]\n");
        return;
    }

    MGPU_LOOP_START;
    {
        numBlocks       = pPmu[indexGpu].pmuImemGetNumBlocks();
        tagWidth        = pPmu[indexGpu].pmuImemGetTagWidth();
        maxAllowedTag   = BIT(tagWidth) - 1;

        //
        // Loop through the block mappings looking for max/min tag
        // Only do this if the user did not specify a start/end tag
        //
        if (!bArguments)
        {
            for (blockIndex = 0; blockIndex < numBlocks; blockIndex++)
            {
                if (pPmu[indexGpu].pmuImblk(blockIndex, &blockInfo))
                {
                    if (blockInfo.tag < minTag)
                        minTag = blockInfo.tag;
                    if (blockInfo.tag > maxTag)
                        maxTag = blockInfo.tag;
                }
            }
        }
        else
        {
            if ((maxTag > maxAllowedTag) || (minTag > maxAllowedTag))
            {
                dprintf("lw: %s: Tag was larger than maximum allowed (max 0x%02x)\n",
                        __FUNCTION__, maxAllowedTag);
                return;
            }
        }

        pmuImemMapDump(minTag, maxTag, bSkipUnmappedTags);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmuimblk [block index]
//        List all current block status.
//
// If a block index is specified, the status for all blocks
// is printed. Otherwise only the specified block status is
// printed. An invalid block index results in an error
// message.
//-----------------------------------------------------
DECLARE_API( pmuimblk )
{
    BOOL        bListAll    = TRUE;
    LwU64       userIndex   = ~0x0;
    LwU32       blockIndex  = 0x0;
    LwU32       numBlocks   = 0x0;
    LwU32       tagWidth    = 0x0;
    PmuBlock    blockInfo;

    CHECK_INIT(MODE_LIVE);

    // Get the block index
    GetSafeExpressionEx(args, &userIndex, &args);
    bListAll    = (userIndex == (LwU64)(~0));
    blockIndex  = (LwU32)userIndex;

    MGPU_LOOP_START;
    {
        numBlocks = pPmu[indexGpu].pmuImemGetNumBlocks();
        tagWidth  = pPmu[indexGpu].pmuImemGetTagWidth();

        if (!bListAll)
        {
            if (blockIndex >= numBlocks)
            {
                dprintf("lw: Block index 0x%x is invalid (max 0x%x).\n",
                        blockIndex, numBlocks - 1);
                return;
            }
            else
            {
                numBlocks = blockIndex + 1;
            }
        }
        else
        {
            blockIndex = 0;
        }

        dprintf("lw:\tDumping IMEM code block status\n");
        dprintf("lw:\t#Blocks=0x%x  Tag Width=%d\n", numBlocks, tagWidth);
        dprintf("lw:\t--------------------------------------------------\n");

        // Loop through all the blocks dumping out information
        for (; blockIndex < numBlocks; blockIndex++)
        {
            if (pPmu[indexGpu].pmuImblk(blockIndex, &blockInfo))
            {
                dprintf("lw:\tBlock 0x%02x: tag=0x%02x, valid=%d, pending=%d, "
                        "secure=%d\n",
                        blockIndex, blockInfo.tag, blockInfo.bValid,
                        blockInfo.bPending, blockInfo.bSelwre);
            }
        }

        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmuimtag <block address>
//        List status of block mapped to address.
//
// Block address, or "code addres", will be used to find
// an IMEM code block mapping.
//-----------------------------------------------------
DECLARE_API( pmuimtag )
{
    LwU64       addr        = 0x0;
    LwU32       tagWidth    = 0x0;
    LwU32       maxAddr     = 0x0;
    PmuTagBlock tagMapping;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw:\tUsage: !lw.pmuimtag <block addr>\n");
        return;
    }

    // Get the block address to look up
    GetSafeExpressionEx(args, &addr, &args);

    MGPU_LOOP_START;
    {
        //
        // Get the tag width and notify user when code address entered
        // is larger than the maximum allowed by tag width.
        // Note that bits 0-7 are offsets into the block and 8-7+TagWidth
        // define the tag bits.
        //
        tagWidth = pPmu[indexGpu].pmuImemGetTagWidth();
        maxAddr = BIT(tagWidth+8) - 1;

        if (addr > maxAddr)
        {
            dprintf("lw: %s: Address 0x%04x is too large (max 0x%04x)\n",
                    __FUNCTION__, (LwU32)addr, maxAddr);
            return;
        }

        if (pPmu[indexGpu].pmuImtag((LwU32)addr, &tagMapping))
        {
            switch (tagMapping.mapType)
            {
            case PMU_TAG_UNMAPPED:
                dprintf("lw:\tTag 0x%02x: Not mapped to a block", (LwU32)addr >> 8);
                break;
            case PMU_TAG_MULTI_MAPPED:
            case PMU_TAG_MAPPED:
                dprintf("lw:\tTag 0x%02x: block=0x%02x, valid=%d, pending=%d, secure=%d",
                        tagMapping.blockInfo.tag, tagMapping.blockInfo.blockIndex,
                        tagMapping.blockInfo.bValid, tagMapping.blockInfo.bPending,
                        tagMapping.blockInfo.bSelwre);
                break;
            }
            if (tagMapping.mapType == PMU_TAG_MULTI_MAPPED)
            {
                dprintf(" (multiple)");
            }
            dprintf("\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmumutex [mutex id]
//        dump the contents of PMU mutices or a specific mutex
//
// A specific mutex Id may be specified.
//-----------------------------------------------------
DECLARE_API( pmumutex )
{
    BOOL listAll   = TRUE;
    BOOL free      = FALSE;
    LwU64 lockId = ~0;
    LwU32 id        =  0;
    LwU32 mutex     =  0;
    LwU32 numLocks  =  0;

    CHECK_INIT(MODE_LIVE);

    GetSafeExpressionEx(args, &lockId, &args);
    listAll = (lockId == (LwU64)(~0));

    MGPU_LOOP_START;
    {
        numLocks = pPmu[indexGpu].pmuMutexGetNum();
        if (!listAll)
        {
            id          = (LwU32)lockId;
            numLocks    = id + 1;
        }
        for (; id < numLocks; id++)
        {
            if (pPmu[indexGpu].pmuMutexRead(id, &mutex, &free))
            {
                dprintf("lw:\tMutex 0x%x: 0x%08x - %s\n", id, mutex,
                        free ? "Available" : "Taken");
            }
            else
            {
                dprintf("lw:\tMutex 0x%x is invalid.\n", id);
            }
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmutcb <tcbAddress> [port] [size] [options]
//    Dump the contents of the PMU TCB at address
//    'tcbAddress' including the full stack.
//    '-a' or '-l' to list info on all TCBs
//    '-v' for verbose output when using '-a' or '-l'
//-----------------------------------------------------
DECLARE_API( pmutcb )
{
    LwU64    tcbAddress = 0x0;
    LwU64    size       = 0x4;
    LwU64    port       = 0x1;
    BOOL     bVerbose   = FALSE;
    PMU_TCB  tcb;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Usage: !lw.pmutcb [tcbAddress] [port] [size] [-a] "
                "[-l] [-v].\n");
        return;
    }

    if (parseCmd(args, "v", 0, NULL))
    {
        bVerbose = TRUE;
    }

    // Handle the "-c" case for current TCB.
    if (parseCmd(args, "c", 0, NULL))
    {
        if (GetSafeExpressionEx(args, &port, &args))
        {
            GetSafeExpressionEx(args, &size, &args);
        }
        MGPU_LOOP_START;
        {
            if (pmuTcbGetLwrrent(&tcb, (LwU32)port))
            {
                dprintf("lw:========================================\n");
                pmuTcbDump(&tcb, FALSE, (LwU32)port, (LwU8)size);
                dprintf("lw:========================================\n");
            }
        }
        MGPU_LOOP_END;
        return;
    }

    // Handle the "-a" case for all TCBs
    if ((parseCmd(args, "a", 0, NULL)) ||
        (parseCmd(args, "l", 0, NULL)))
    {
        if (GetSafeExpressionEx(args, &port, &args))
        {
            GetSafeExpressionEx(args, &size, &args);
        }
        MGPU_LOOP_START;
        {
            if (pmuTcbGetLwrrent(&tcb, (LwU32)port))
            {
                pmuTcbDumpAll(!bVerbose);
            }
        }
        MGPU_LOOP_END;
        return;
    }

    if (GetSafeExpressionEx(args, &tcbAddress, &args))
    {
        if (GetSafeExpressionEx(args, &port, &args))
        {
            GetSafeExpressionEx(args, &size, &args);
        }
    }

    MGPU_LOOP_START;
    {
        if (pPmu[indexGpu].pmuTcbGet((LwU32)tcbAddress, 0, &tcb))
        {
            pmuTcbDump(&tcb, FALSE, (LwU32)port, (LwU8)size);
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmusched
//-----------------------------------------------------
DECLARE_API( pmusched )
{
    BOOL  bTable  = TRUE;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "h", 0, NULL))
    {
        dprintf("lw: Usage: !lw.pmusched [-h] [-l]\n");
        return;
    }

    if (parseCmd(args, "l", 0, NULL))
    {
        bTable = FALSE;
    }

    MGPU_LOOP_START;
    {
        pmuSchedDump(bTable);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmuevtq
//-----------------------------------------------------
DECLARE_API( pmuevtq )
{
    char    *pParams;
    LwBool   bAll    = LW_FALSE;
    char    *pSym    = NULL;
    LwU64    qAddr   = 0x00;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "h", 0, NULL))
    {
        dprintf("lw: Usage: !pmuevtq [-h] [-a] [-s <symbol>] [-n <address>]\n");
        dprintf("lw:                 - Dump out the PMU RTOS event queues\n");
        dprintf("lw:                   + '-h' : print usage\n");
        dprintf("lw:                   + '-a' : dump info on all known event queues\n");
        dprintf("lw:                   + '-s' : dump info on a specific queue (identified by symbol name)\n");
        dprintf("lw:                   + '-n' : dump info on a specific queue (identified by queue address)\n");
        return;
    }

    if (parseCmd(args, "a", 0, NULL))
    {
        bAll = LW_TRUE;
    }

    parseCmd(args, "s", 1, &pSym);

    if (parseCmd(args, "n", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &qAddr, &pParams);
    }

    MGPU_LOOP_START;
    {
        if (bAll)
        {
            pmuEventQueueDumpAll(LW_FALSE);
        }
        else if (pSym != NULL)
        {
            pmuEventQueueDumpBySymbol(pSym);
        }
        else if (qAddr != 0)
        {
            pmuEventQueueDumpByAddr((LwU32)qAddr);
        }
        else
        {
            pmuEventQueueDumpAll(LW_TRUE);
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// pmufuse
//---------------------------------------------------------
DECLARE_API ( pmufuse )
{
    LwU32  retCode = 0;

    CHECK_INIT(MODE_LIVE);

    retCode = pPmu[indexGpu].pmuVerifyFuse(indexGpu);
    return;
}

//-----------------------------------------------------
// pmuswak
//---------------------------------------------------------
DECLARE_API ( pmuswak )
{
    LwBool bDetail = LW_FALSE;
    char  *pFilename = NULL;

    MGPU_LOOP_START;
    {
        if (parseCmd(args, "d", 0, NULL))
        {
            bDetail = LW_TRUE;
        }

        if (parseCmd(args, "f", 1, &pFilename))
        {
            bDetail = LW_TRUE;
        }
        pmuswakExec(pFilename, bDetail);
    }
    MGPU_LOOP_END;
    return;
}

//-----------------------------------------------------
// pmusym
//---------------------------------------------------------
DECLARE_API( pmusym )
{
    char    *pFilename   = NULL;
    char    *pParams = NULL;
    BOOL     bIgnoreCase = FALSE;
    LwU64    address;
    LwU32    ucodeVersion;
    PMU_SYM *pMatches;
    LwU32    i = 1;

    CHECK_INIT(MODE_LIVE);

    ucodeVersion = pPmu[indexGpu].pmuUcodeGetVersion();

    if (parseCmd(args, "l", 1, &pFilename))
    {
        MGPU_LOOP_START;
        {
            pmuSymLoad(pFilename, ucodeVersion);
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "l", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            pmuSymLoad(NULL, ucodeVersion);
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "u", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            pmuSymUnload();
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "n", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &address, &pParams);
        MGPU_LOOP_START;
        {
            pMatches = pmuSymResolve((LwU32)address);
            if (pMatches != NULL)
            {
                while (pMatches != NULL)
                {
                    pmuSymPrintBrief(pMatches, i++);
                    pMatches = pMatches->pTemp;
                }
            }
            else
            {
                dprintf("lw: Error: no matches found. Symbol out of range?\n");
            }
        }
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(args, "i", 1, &pParams))
    {
        bIgnoreCase = TRUE;
    }
    else
    {
        pParams = args;
    }

    if ((pParams != NULL ) && (strlen(pParams) > 0))
    {
        MGPU_LOOP_START;
        {
            pmuSymDump(pParams, bIgnoreCase);
        }
        MGPU_LOOP_END;
        return;
    }
}

//-----------------------------------------------------
// pmust
//---------------------------------------------------------
DECLARE_API( pmust )
{
    char    *pFilename   = NULL;
    LwU32    ucodeVersion;
    BOOL     bStatus;

    CHECK_INIT(MODE_LIVE);

    ucodeVersion = pPmu[indexGpu].pmuUcodeGetVersion();

    if (parseCmd(args, "l", 1, &pFilename))
    {
        MGPU_LOOP_START;
        {
            bStatus = pmustLoad(pFilename, ucodeVersion, TRUE);
        }
        MGPU_LOOP_END;

        if (!bStatus)
            dprintf("lw: Unable to load the pmu objdump file at %s\n", pFilename);
        return;
    }

    if (parseCmd(args, "l", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            bStatus = pmustLoad(NULL, ucodeVersion, TRUE);
        }
        MGPU_LOOP_END;
        if (!bStatus)
            dprintf("lw: Unable to load the pmu objdump file from lwsym\n");
        return;
    }

    if (parseCmd(args, "u", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            bStatus = pmustUnload();
        }
        MGPU_LOOP_END;
        if (!bStatus)
            dprintf("lw: Unable to unload the pmu objdump\n");
        return;
    }

    /* otherwise we just print out the stack trace */
    {
        PMU_TCB tcb;
        LwU32   port = 0x1;

        if (pmuTcbGetLwrrent(&tcb, port) && (pmuTcbValidate(&tcb) == 0))
        {
            if (tcb.tcbVer == PMU_TCB_VER_0)
            {
                pmustPrintStacktrace(tcb.pmuTcb.pmuTcb0.pStack, tcb.pmuTcb.pmuTcb0.stackDepth, ucodeVersion);
            }
            else
            {
                PMU_TCB_PVT* pPrivTcb = NULL;
                LwU32 pStack = 0;
                LwU32 stackSize = 0;

                switch (tcb.tcbVer)
                {
                    case PMU_TCB_VER_0:
                        // Satisfy -Werror=switch error on DVS
                        break;
                    case PMU_TCB_VER_1:
                        pmuTcbGetPriv(&pPrivTcb, (LwU32)tcb.pmuTcb.pmuTcb1.pvTcbPvt, port);
                        break;
                    case PMU_TCB_VER_2:
                        pmuTcbGetPriv(&pPrivTcb, (LwU32)tcb.pmuTcb.pmuTcb2.pvTcbPvt, port);
                        break;
                    case PMU_TCB_VER_3:
                        pmuTcbGetPriv(&pPrivTcb, (LwU32)tcb.pmuTcb.pmuTcb3.pvTcbPvt, port);
                        break;
                    case PMU_TCB_VER_4:
                        pmuTcbGetPriv(&pPrivTcb, (LwU32)tcb.pmuTcb.pmuTcb4.pvTcbPvt, port);
                        break;
                    case PMU_TCB_VER_5:
                        pmuTcbGetPriv(&pPrivTcb, (LwU32)tcb.pmuTcb.pmuTcb5.pvTcbPvt, port);
                        break;
                }

                // switch on private tcb version
                switch (pPrivTcb->tcbPvtVer)
                {
                    case FLCN_TCB_PVT_VER_0:
                        pStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt0.pStack;
                        stackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt0.stackSize;
                        break;

                    case FLCN_TCB_PVT_VER_1:
                        pStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt1.pStack;
                        stackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt1.stackSize;
                        break;

                    case FLCN_TCB_PVT_VER_2:
                        pStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt2.pStack;
                        stackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt2.stackSize;
                        break;

                    case FLCN_TCB_PVT_VER_3:
                        pStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt3.pStack;
                        stackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt3.stackSize;
                        break;

                    case FLCN_TCB_PVT_VER_4:
                        pStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt4.pStack;
                        stackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt4.stackSize;
                        break;

                    case FLCN_TCB_PVT_VER_5:
                        pStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt5.pStack;
                        stackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt5.stackSize;
                        break;

                    case FLCN_TCB_PVT_VER_6:
                        pStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt6.pStack;
                        stackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt6.stackSize;
                        break;

                    default:
                        dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                        break;
                }

                //
                // In recent TCB versions we move stack bottom to the RTOS TCB from RM_RTOS_TCB_PVT
                // When we remove fields from RM_RTOS_TCB_PVT we clean this all up
                //
                if (PMU_TCB_VER_3 == tcb.tcbVer)
                {
                    pStack = tcb.pmuTcb.pmuTcb3.pcStackBaseAddress;
                }
                if (PMU_TCB_VER_4 == tcb.tcbVer)
                {
                    pStack = tcb.pmuTcb.pmuTcb4.pcStackBaseAddress;
                }
                if (PMU_TCB_VER_5 == tcb.tcbVer)
                {
                    pStack = tcb.pmuTcb.pmuTcb5.pcStackBaseAddress;
                }

                pmustPrintStacktrace(pStack, stackSize, ucodeVersion);
                free(pPrivTcb);
            }
        }
        else
        {
            // We cannot get the current tcb, so just let stack trace to figure out as many as possible
            dprintf("Cannot retrieve the current tcb or tcb is corrupted, simply parse the stack as much as pmust can\n\n");
            pmustPrintStacktrace(0, 0, ucodeVersion);
        }
    }
}

//-----------------------------------------------------
// pmu
//---------------------------------------------------------
DECLARE_API( pmu )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pmuExec(args);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// flcn
//---------------------------------------------------------
DECLARE_API( flcn )
{
    static POBJFLCN pCntFlcn = NULL;
           POBJFLCN pTmpFlcn = NULL;
    char  *pStart = NULL;
    char   token[64];
    LwBool bEngineChanged = LW_FALSE;
    LwBool bCmdLine = LW_FALSE;

    CHECK_INIT(MODE_LIVE);

    //
    // Command Parsing:
    //
    // 1. Check if the first token is the <engine>
    //    If it does, change the target engine pCntFlcn and ignore the rest
    //    parameters.
    // 2. Check if this is command line format.  Ex. !flcn -pmu <command>
    // 3. Before pass the rest command to flcnExec(), check if pCntFlcn is not
    //    NULL and the function interfaces are ready
    //

    //
    // #1. Check if the first token is the <engine>
    //
    getToken(args, token, &pStart);
    if (pStart)
    {
        //
        // only when there exists the next token, make pParams point
        // to the start of the next token
        //
        args = pStart;
    }

    if (strcmp(token, "dpu") == 0)
    {
        pCntFlcn = dpuGetFalconObject();
        bEngineChanged = LW_TRUE;
    }
    else if (strcmp(token, "pmu") == 0)
    {
        pCntFlcn = pmuGetFalconObject();
        bEngineChanged = LW_TRUE;
    }
    else if (strcmp(token, "sec2") == 0)
    {
        pCntFlcn = sec2GetFalconObject();
        bEngineChanged = LW_TRUE;
    }
    else if (strcmp(token, "gsp") == 0)
    {
        pCntFlcn = gspGetFalconObject();
        bEngineChanged = LW_TRUE;
    }
    else if (strcmp(token, "gsplite") == 0)
    {
        pCntFlcn = gspGetFalconObject();
        bEngineChanged = LW_TRUE;
    }
    else if (strcmp(token, "fbflcn") == 0)
    {
        pCntFlcn = fbflcnGetFalconObject();
        bEngineChanged = LW_TRUE;
    }
    else if (strcmp(token, "ofa") == 0)
    {
        pCntFlcn = ofaGetFalconObject();
        bEngineChanged = LW_TRUE;
    }
    else if (strcmp(token, "help") == 0)
    {
        dprintf("lw: Usage:\n");
        dprintf("lw: !flcn <engine>             - Set the default engine \n");
        dprintf("lw: !flcn <command>            - Execute <command> using the default engine\n");
        dprintf("lw: !flcn -<engine> <command>  - Command line mode, execute <command> using <engine>\n");
        dprintf("lw:\n");
        dprintf("lw: Supported engines:\n");
        dprintf("lw: pmu, dpu, sec2, gsplite, gsp, fbflcn, ofa\n");
        dprintf("lw:\n");
        FLCN_PRINT_USAGE_MESSAGE();

        return;
    }

    if (bEngineChanged)
    {
        if(!pCntFlcn)
        {
            dprintf("lw: !flcn <engine> fails \n");
        }
        else if( pCntFlcn->pFEIF == NULL || pCntFlcn->pFCIF == NULL )
        {
            dprintf("lw: !flcn doesn't support %s on this GPU. \n",pCntFlcn->engineName);
            pCntFlcn = NULL;
        }
        else
        {
            dprintf("lw: Switch Falcon engine to %s\n",pCntFlcn->engineName);
        }

        return;
    }

    //
    // #2. For command line: !flcn -<engine name> <command>
    //
    if (parseCmd(args, "dpu", 0, NULL))
    {
        pTmpFlcn = pCntFlcn;
        pCntFlcn = dpuGetFalconObject();
        bCmdLine = LW_TRUE;
    }
    else if (parseCmd(args, "pmu", 0, NULL))
    {
        pTmpFlcn = pCntFlcn;
        pCntFlcn = pmuGetFalconObject();
        bCmdLine = LW_TRUE;
    }
    else if (parseCmd(args, "sec2", 0, NULL))
    {
        pTmpFlcn = pCntFlcn;
        pCntFlcn = sec2GetFalconObject();
        bCmdLine = LW_TRUE;
    }
    else if (parseCmd(args, "gsp", 0, NULL))
    {
        pTmpFlcn = pCntFlcn;
        pCntFlcn = gspGetFalconObject();
        bCmdLine = LW_TRUE;
    }
    else if (parseCmd(args, "gsplite", 0, NULL))
    {
        pTmpFlcn = pCntFlcn;
        pCntFlcn = gspGetFalconObject();
        bCmdLine = LW_TRUE;
    }
    else if (parseCmd(args, "fbflcn", 0, NULL))
    {
        pTmpFlcn = pCntFlcn;
        pCntFlcn = fbflcnGetFalconObject();
        bCmdLine = LW_TRUE;
    }
    else if (parseCmd(args, "ofa", 0, NULL))
    {
        pTmpFlcn = pCntFlcn;
        pCntFlcn = ofaGetFalconObject();
        bCmdLine = LW_TRUE;
    }

    //
    // #3. check if the falcon engine is set and the interfaces are ready
    //
    if ((!pCntFlcn || !pCntFlcn->pFCIF || !pCntFlcn->pFEIF))
    {
        if(bCmdLine)
        {
            dprintf("lw: !flcn doesn't support this falcon on this GPU. \n");
            pCntFlcn = pTmpFlcn;
        }
        else
        {
            dprintf("lw: The target Falcon engine is not set, please use \n"
                    "lw: !flcn <engine> to set the target engine.\n"
                    "lw: Current supported engines: \"pmu\", \"dpu\", \"sec2\", \"gsplite\", \"gsp\", \"fbflcn\", \"ofa\" \n");
        }
        return;
    }

    //
    // Execute the command
    //
    MGPU_LOOP_START;
    {
        flcnExec(args, pCntFlcn);
    }
    MGPU_LOOP_END;

    //
    // Switch back to original falcon engine after command line exelwtion
    //
    if (bCmdLine)
    {
        pCntFlcn = pTmpFlcn;
    }
}

//-----------------------------------------------------
// rv - RISC-V handling
//---------------------------------------------------------
DECLARE_API( rv )
{
    CHECK_INIT(MODE_LIVE);

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) || LWWATCHCFG_IS_PLATFORM(WINDOWS)
    riscvMain(args);
#else
    dprintf("Error: RISCV is not supported on this OS.\n");
#endif
}

DECLARE_API( rvgdb )
{
    CHECK_INIT(MODE_LIVE);

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) || LWWATCHCFG_IS_PLATFORM(WINDOWS)
    riscvGdbMain(args);
#else
    dprintf("Error: RISCV is not supported on this OS.\n");
#endif
}

//-----------------------------------------------------------------------------
// pmusanitytest [options]
//  - Runs PMU sanity test.
//    + '-t' <test#> : execute command on a single test program.
//    + '-v' <level> : verbose level. 0-3 where 0 - mute (default), 3 - noisy.
//    + '-i'         : prints description of available pmu sanity tests
//    + '-n'         : returns the number of tests available. <testnum> ignored
//-----------------------------------------------------------------------------
DECLARE_API( pmusanitytest )
{
    char     *pParams;
    LwU64     verbose      = 0;
    LwU64     testnum      = 0;
    LwU32     totalNumTest = 0;
    BOOL      bAllTest     = TRUE;
    LwU32     i;
    LW_STATUS status;

    CHECK_INIT(MODE_LIVE);

    // Handle the "-t" case
    if (parseCmd(args, "t", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &testnum, &pParams);
        bAllTest = FALSE;
    }

    // Handle the "-v" case
    if (parseCmd(args, "v", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &verbose, &pParams);
    }

    // Handle the "-n" case
    if (parseCmd(args, "n", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            dprintf("# of available PMU tests: %d\n",
                    pPmu[indexGpu].pmuSanityTestGetNum());
        }
        MGPU_LOOP_END;
        return;
    }

    // Handle the "-i" case
    if (parseCmd(args, "i", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            totalNumTest = pPmu[indexGpu].pmuSanityTestGetNum();
            dprintf("  # | Test Names  A - AUTO         D - DESTRUCTIVE X - ARG REQUIRED\n");
            dprintf("    |             O - OPTIONAL ARG P - PROD UCODE  V - VERIF UCODE\n");
            dprintf("----+-----------------------------------------------------------------\n");
            if (bAllTest)
            {
                for (i = 0; i < totalNumTest; i++)
                {
                    dprintf("%3x | %s\n", i,
                            pPmu[indexGpu].pmuSanityTestGetInfo(i, VB1));
                }
            }
            else
            {
                    dprintf("%3x | %s\n", (LwU32) testnum,
                            pPmu[indexGpu].pmuSanityTestGetInfo((LwU32) testnum, VB1));
            }
        }
        MGPU_LOOP_END;
        return;
    }

    dprintf("========================================================\n");
    dprintf("                    PMU SANITY TEST                     \n");
    dprintf("========================================================\n");
    if (bAllTest)
    {
        MGPU_LOOP_START;
        {
            totalNumTest = pPmu[indexGpu].pmuSanityTestGetNum();
            for (i = 0; i < totalNumTest; i++)
            {
                char * _errstr[] = {"PASS!", "RETRY", "FAIL!", "SKIP!"};
                LwU32  _errno;

                if (pPmu[indexGpu].pmuSanityTestGetFlags(i) & PMU_TEST_AUTO)
                {
                    status =
                        pPmu[indexGpu].pmuSanityTestRun(i, (LwU32) verbose, NULL);

                    if      (status == LW_OK)    _errno = 0;
                    else if (status == LW_ERR_BUSY_RETRY) _errno = 1;
                    else                         _errno = 2;
                }
                else
                {
                    _errno = 3;
                }

                dprintf("%2x. %42s   [%s]\n", i,
                        pPmu[indexGpu].pmuSanityTestGetInfo(i, VB0),
                        _errstr[_errno]);
            }
        }
        MGPU_LOOP_END;
    }
    else
    {
        MGPU_LOOP_START;
        {
            totalNumTest = pPmu[indexGpu].pmuSanityTestGetNum();
            if (testnum < totalNumTest)
            {
                dprintf("Running a single test : [%d] \"%s\"\n", (LwU32) testnum,
                        pPmu[indexGpu].pmuSanityTestGetInfo((LwU32) testnum, VB0));
                status = pPmu[indexGpu].pmuSanityTestRun((LwU32) testnum,
                                (LwU32) verbose, NULL);
                if (status == LW_OK)
                {
                    dprintf(":::::: PASS!!\n");
                }
                else if (status == LW_ERR_BUSY_RETRY)
                {
                    dprintf(":::::: RETRY!!\n");
                    dprintf(":::::: This is often returned when the test requires the PMU be in\n"
                            "       some specific state. Run the test with -v 2 to find out how to\n"
                            "       get around this problem.\n");
                }
                else
                {
                    dprintf(":::::: FAIL!!\n");
                }
            }
            else
            {
                dprintf("Test number %u is invalid.\n", (LwU32)testnum);
            }
        }
        MGPU_LOOP_END;
    }
}

//-----------------------------------------------------------------------------
// pmuqboot <program>
//  - PMU Quick Boot : Bootstraps a simple PMU program.
//    <program> : A simple PMU binary to bootstrap and run.
//-----------------------------------------------------------------------------
DECLARE_API( pmuqboot )
{
    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Usage: !lw.pmuqboot <program> \n");
        return;
    }
    pmuSimpleBootstrap(args);
}

//-----------------------------------------------------------------------------
// smbpbi
//-----------------------------------------------------------------------------
DECLARE_API( smbpbi )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        smbpbiExec(args);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------------------------------
// seq
//-----------------------------------------------------------------------------
DECLARE_API( seq )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        seqExec(args);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// fbstate
// - prints the state of FB: r/w test
//-----------------------------------------------------
DECLARE_API( fbstate )
{
    LW_STATUS  status;

    CHECK_INIT(MODE_LIVE | MODE_DUMP);

    MGPU_LOOP_START;
    {
        dprintf("\n\tlw: ******** Checking bus interrupts... ********\n");

        if (LW_OK == pBus[indexGpu].busTestBusInterrupts())
        {
            dprintf("lw: ******** PASSED: no pending interrupts. ******** \n");
        }
        else
        {
            dprintf("lw: ******** FAILED: interrupts are pending. ******** \n");
        }

        dprintf("\n\tlw: ******** TLB test... ********\n");

        if (LW_OK == pFb[indexGpu].fbTestTLB())
        {
            dprintf("lw: ******** TLB test PASSED. ******** \n");
        }
        else
        {
            dprintf("lw: ******** TLB test FAILED. ******** \n");
        }

        dprintf("\n\tlw: ******** Frame Buffer test... ********\n");

        if (lwMode == MODE_DUMP)
        {
            dprintf("lw: ******** Frame Buffer test SKIPPED. ******** \n");
        }
        else if (LW_OK == pFb[indexGpu].fbTest())
        {
            dprintf("lw: ******** Frame Buffer test PASSED. ******** \n");
        }
        else
        {
            dprintf("lw: ******** Frame Buffer test FAILED. ******** \n");
        }

        dprintf("\n\tlw: ******** System Memory test... ********\n");

        if ((lwMode == MODE_DUMP) || ((status = pFb[indexGpu].fbTestSysmem()) == LW_ERR_NOT_SUPPORTED))
        {
            dprintf("lw: ******** System Memory test SKIPPED. ******** \n");
        }
        else if (status == LW_OK)
        {
            dprintf("lw: ******** System Memory test PASSED. ******** \n");
        }
        else
        {
            dprintf("lw: ******** System Memory test FAILED. ******** \n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// ptevalidate
// - validates PT entries
//-----------------------------------------------------
DECLARE_API( ptevalidate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** PTE validation test... ********\n");

        if(  LW_OK == pMmu[indexGpu].mmuPteValidate() )
        {
            dprintf("lw: ******** PTEs Validation test PASSED. ********\n");
        }
        else
        {
            dprintf("lw: ******** PTEs Validation test FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;

}

//-----------------------------------------------------
// pdecheck
// - check all pte entries for given <chid,pde>
//-----------------------------------------------------
DECLARE_API( pdecheck )
{
    LwU64 chId, pdeId;

    CHECK_INIT(MODE_LIVE);

    if (GetSafeExpressionEx(args, &chId, &args))
    {
        pdeId = GetSafeExpression(args);
    }
    else
    {
        dprintf("lw: Usage: !lw.pdecheck <chId> <pdeId>\n");
        return;
    }

    MGPU_LOOP_START;
    {
        if(  LW_OK == pMmu[indexGpu].mmuPdeCheck( (LwU32)chId, (LwU32)pdeId) )
        {
            dprintf("lw: ******** PDE Validation PASSED. ********\n");
        }
        else
        {
            dprintf("lw: ******** PDE Validation FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;

}

//-----------------------------------------------------
// hoststate
// - tests if host is in valid state
//-----------------------------------------------------
DECLARE_API( hoststate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** Host state test... ********\n");

        if(  LW_OK == pFifo[indexGpu].fifoTestHostState() )
        {
            dprintf("lw: ******** Host state test PASSED. ********\n");
        }
        else
        {
            dprintf("lw: ******** Host state test FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------------------------------------
// grstate [-grIdx] <grIdx>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - tests if graphics is in valid state
//-----------------------------------------------------------------------------------
DECLARE_API( grstate )
{
    FILE    *fp;

    // Parameters passed on to sigdump are hardcoded here. We should allow the user to override these parameters.
    int regWriteOptimization = 1;
    int regWriteCheck = 0;
    int markerValuesCheck = 1;
    int verifySigdump = 0;
    int engineStatusVerbose = 0;
    int priCheckVerbose = 0;
    int multiSignalOptimization = 0;
    char *chipletKeyword = NULL;
    char *chipletNumKeyword = NULL;
    char *domainKeyword = NULL;
    char *domainNumKeyword = NULL;
    char *instanceNumKeyword = NULL;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    CHECK_INIT(MODE_LIVE | MODE_DUMP);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** Graphics state test... ********\n");

        if ( LW_OK == pGr[indexGpu].grTestGraphicsState( grIdx ) )
        {
            dprintf("lw: ******** Graphics state test PASSED. ********\n");
        }
        else
        {
            dprintf("lw: ******** Graphics state test FAILED. ********\n");

            if (lwMode == MODE_LIVE)
            {
                //sigdump on error
                fp = fopen("sigdump.txt", "w");
                if (fp == NULL)
                {
                    dprintf("lw: Unable to open sigdump.txt\n");
                }
                else
                {
                    dprintf("lw: sigdump.txt created in the current working directory.\n");
                    pSig[indexGpu].sigGetSigdump(fp,
                                                 regWriteOptimization,
                                                 regWriteCheck,
                                                 markerValuesCheck,
                                                 verifySigdump,
                                                 engineStatusVerbose,
                                                 priCheckVerbose,
                                                 multiSignalOptimization,
                                                 chipletKeyword,
                                                 chipletNumKeyword,
                                                 domainKeyword,
                                                 domainNumKeyword,
                                                 instanceNumKeyword
                                                 );
                    fclose(fp);
                }
            }
        }
    }
    MGPU_LOOP_END;
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------
// msdecstate
// - tests if msdec engines (vld, dec, ppp) are in valid state
//-----------------------------------------------------
DECLARE_API( msdecstate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** MSDEC state test... ********\n");

        //test all engines
        if(  LW_OK == msdecTestMsdecState(3) )
        {
            dprintf("\n\t lw: ******** MSDEC state test PASSED. ********\n");
        }
        else
        {
            dprintf("\n\t lw: ******** MSDEC state test FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// vic
// - VIC engine commands
//-----------------------------------------------------
DECLARE_API( vic )
{
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bPriv, bImem, bDmem, bState, bHwCfg, bSpr;

    CHECK_INIT(MODE_LIVE);

    bHelp      = parseCmd(args, "help",      0, NULL);
    bSupported = parseCmd(args, "supported", 0, NULL);
    bPriv      = parseCmd(args, "priv",      0, NULL);
    bImem      = parseCmd(args, "imem",      0, NULL);
    bDmem      = parseCmd(args, "dmem",      0, NULL);
    bState     = parseCmd(args, "state",     0, NULL);
    bHwCfg     = parseCmd(args, "hwcfg",     0, NULL);
    bSpr       = parseCmd(args, "spr",       0, NULL);

    if (bHelp)
    {
        vicDisplayHelp();
        return;
    }

    MGPU_LOOP_START;
    {
        if (bSupported)
        {
            vicIsSupported(indexGpu);
            bHandled = TRUE;
        }

        // Check if device is powered on and not in reset before trying to touch it
        if (pTegrasys[indexGpu].tegrasysDeviceOn(&TegraSysObj[indexGpu], "VIC", 0)==TRUE)
        {
                if (bPriv)
            {
                vicDumpPriv(indexGpu);
                bHandled = TRUE;
            }
                else if (bImem)
            {
                vicDumpImem(indexGpu);
                bHandled = TRUE;
            }
                else if (bDmem)
            {
                vicDumpDmem(indexGpu);
                bHandled = TRUE;
            }
                else if (bState)
            {
                vicTestState(indexGpu);
                bHandled = TRUE;
            }
                else if (bHwCfg)
            {
                vicDisplayHwcfg(indexGpu);
                bHandled = TRUE;
            }
                else if (bSpr)
            {
                vicDisplayFlcnSPR(indexGpu);
                bHandled = TRUE;
            }
        }
        else
        {
            dprintf("lw: 'VIC' powered off/in reset\n");
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        vicDisplayHelp();
    }
}

//-----------------------------------------------------
// elpgstate
// - tests if elpg state is valid
//-----------------------------------------------------
DECLARE_API( elpgstate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** ELPG state test... ********\n");

        if(  LW_OK == pPmu[indexGpu].pmuTestElpgState() )
        {
            dprintf("\n\t lw: ******** ELPG state test PASSED. ********\n");
        }
        else
        {
            dprintf("\n\t lw: ******** ELPG state test FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// lpwrstate
// - tests if lpwr state is valid
//-----------------------------------------------------
DECLARE_API( lpwrstate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** LPWR state test... ********\n");

        if(  LW_OK == pPmu[indexGpu].pmuTestLpwrState() )
        {
            dprintf("\n\t lw: ******** LPWR state test PASSED. ********\n");
        }
        else
        {
            dprintf("\n\t lw: ******** LPWR state test FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// lpwrfsmstate
// - tests if lpwr FSM (Finite state machine) sequencer state is valid
//-----------------------------------------------------
DECLARE_API( lpwrfsmstate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** Lpwr FSM State ********\n");

        if(LW_OK == lpwrGetFsmState())
        {
            dprintf("\n\t lw: ******** LPWR FSM state test PASSED. ********\n");
        }
        else
        {
            dprintf("\n\t lw: ******** LPWR FSM state test FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------------------------------------
// gpuanalyze [-grIdx] <grIdx>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
//-----------------------------------------------------------------------------------
DECLARE_API( gpuanalyze )
{
    char *param;
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    MGPU_LOOP_START;
    {
        dprintf("\n\t lw: ******** GPU Analyze... ********\n");

        if(  LW_OK == gpuAnalyze( grIdx ) )
        {
            dprintf("\n\t lw: ******** GPU Analyze PASSED. ********\n");
        }
        else
        {
            dprintf("\nlw: ******** GPU Analyze FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------
// cestate
//
//-----------------------------------------------------
DECLARE_API( cestate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n\tlw: ******** CE state test... ********\n");

        if ( LW_OK == pCe[indexGpu].ceTestCeState( indexGpu, ~0 ) )
        {
            dprintf("\n\tlw: ******** CE state test PASSED. ********\n");
        }
        else
        {
            dprintf("\n\tlw: ******** CE state test FAILED. ********\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// dispstate
//
//-----------------------------------------------------
DECLARE_API( dispstate )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("\n                   <<<< DISPLAY STATE TEST >>>>                   ");

        if ( LW_OK == pDisp[indexGpu].dispTestDisplayState() )
        {
            dprintf("\n                 >>>> DISPLAY STATE TEST PASSED <<<<                 \n");
        }
        else
        {
            dprintf("\n                 >>>> DISPLAY STATE TEST FAILED <<<<                 \n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// zbc [table index]
//
//-----------------------------------------------------
DECLARE_API( zbc )
{
    LwU64 index = ~0;

    CHECK_INIT(MODE_LIVE);

    GetSafeExpressionEx(args, &index, &args);

    //
    // If no index is specified or if index == 0,
    // then dump the whole ZBC Color and Depth DS/L2 Table.
    //

    pFb[indexGpu].fbReadZBC((LwU32)index);
    dprintf("\n");
}

//-----------------------------------------------------
// falctrace
// - falctrace commands
//-----------------------------------------------------
DECLARE_API( falctrace )
{
    char    *pParams;
    LwBool   bIsSysMem;
    LwBool   bIsPhyAddr;
    LwU64    addr;
    LwU64    size;
    LwU64    numEntries = 0;
    LwU32    engineId   = MSDEC_UNKNOWN;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd(args, "pmu", 0, NULL))
    {
        // Setup PMU engine and it's defaults
        engineId   = MSDEC_PMU;
        bIsSysMem  = TRUE;
        bIsPhyAddr = FALSE;
    }
    else if (parseCmd(args, "dpu", 0, NULL))
    {
        // Setup DPU engine and it's defaults
        engineId   = MSDEC_DPU;
        bIsSysMem  = FALSE;
        bIsPhyAddr = TRUE;
    }
    else if (parseCmd(args, "msenc", 0, NULL))
    {
        // Setup MSDEC/MSENC engine and it's defaults
        engineId   = MSDEC_MSENC;
        bIsSysMem  = FALSE;
        bIsPhyAddr = TRUE;
    }
    else
    {
        dprintf("Error: no engine specified.  See !help for usage " \
                "information.\n");
        return;
    }
    // Check for explicit entry count
    if (parseCmd(args, "n", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &numEntries, &pParams);
    }
    // Dump with default entry count
    else if (parseCmd(args, "n", 0, NULL))
    {
        dprintf("Warning: entry count omitted; reverting to default. See " \
                "!help for usage information.\n");
    }
    // Engine initialization w/two arguments (Address and Size)
    if (parseCmd(args, "i", 2, &pParams))
    {
        // Get the user supplied address and size parameters
        GetSafeExpressionEx(pParams, &addr, &pParams);
        GetSafeExpressionEx(pParams, &size, &pParams);

        // Check for override to default memory space
        if (parseCmd(args, "phy", 0, &pParams))
        {
            bIsPhyAddr = TRUE;
        }
        else if (parseCmd(args, "vir", 0, &pParams))
        {
            bIsPhyAddr = FALSE;
        }
        // Check for override to default aperture
        if (parseCmd(args, "sys", 0, &pParams))
        {
            bIsSysMem = TRUE;
        }
        else if (parseCmd(args, "vid", 0, &pParams))
        {
            bIsSysMem = FALSE;
        }
        MGPU_LOOP_START;
        {
            falctraceInitEngine(engineId, bIsPhyAddr, bIsSysMem, addr, (LwU32)size);
            falctraceDump(engineId, (LwU32)numEntries);
        }
        MGPU_LOOP_END;
    }
    // Engine initialization w/no arguments
    else if (parseCmd(args, "i", 0, &pParams))
    {
        switch (engineId)
        {
            case MSDEC_PMU:
            {
                dprintf("Info: initialization arguments not specified; "
                        "will attempt initialization using defaults.\n");

                MGPU_LOOP_START;
                {
                    // Default address and size (Memory space and aperture already defaulted)
                    addr = pPmu[indexGpu].pmuGetVABase() + 0x40000;
                    size  = 0x8000;

                    dprintf("lw:\n");
                    dprintf("lw: vaddr = " LwU40_FMT "\n", addr);
                    dprintf("lw: size  = 0x%x\n", (LwU32)size);
                    dprintf("lw:\n");
                    dprintf("lw: Dumping tail\n");
                    dprintf("lw:\n");
                    falctraceInitEngine(engineId, bIsPhyAddr, bIsSysMem, addr, (LwU32)size);
                    falctraceDump(engineId, (LwU32)numEntries);
                }
                MGPU_LOOP_END;
                break;
            }
            case MSDEC_DPU:
            default:
            {
                dprintf("Error: insufficient initializion arguements. See "
                        "!help for usage information.\n");
            }
        }
        return;
    }
    // Falcon trace dump without initialization
    else
    {
        // Dump the requested number of entries
        MGPU_LOOP_START;
        {
            falctraceDump(engineId, (LwU32)numEntries);
        }
        MGPU_LOOP_END;
    }
}

//-----------------------------------------------------
// Parses arguments
//  Looks for:
// l2ilwalidate -t <scalingFactor>
//
// Default 1
//-----------------------------------------------------
static LwU32 l2ParseOption
(
    char *args,
    LwU32* scale
)
{
    char *params;
    LwU64 config = 0;

    if (parseCmd(args, "t", 1, &params))
    {
        GetSafeExpressionEx(params, &config, &params);
        if (config == 0)
        {
            dprintf(" lw: improper scale factor\n");
            return LW_ERR_GENERIC;
        }

        *scale = (LwU32)config;
        return LW_OK;
    }
    return LW_ERR_GENERIC;
}

//-----------------------------------------------------
// L2 ilwalidate
// Timeout is timeout * scalingFactor
// Default is 1 sec with scaling factor = 1
//
//-----------------------------------------------------
DECLARE_API( l2ilwalidate )
{
    LwU32 scale;

    CHECK_INIT(MODE_LIVE);

    if (LW_ERR_GENERIC == l2ParseOption(args, &scale))
    {
        scale = 1;
    }

    MGPU_LOOP_START;
    {
        if ( LW_OK == pFb[indexGpu].fbL2IlwalEvict(scale) )
        {
            dprintf("\n\tlw: ** L2 Ilwalidation done **\n");
        }
        else
        {
            dprintf("\n\tlw: ** L2 Ilwalidation FAILED. **\n");
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// Parses arguments
//  Looks for:
// l2ilwalidate -t <scalingFactor>
//
// Default 1
//-----------------------------------------------------
static LwU32 fbMonitorParseOption
(
    char *args,
    LwU32* pFbps,
    BOOL* pbRead
)
{
    char *params;
    LwU64 temp = 0;
    *pbRead = FALSE;

    if (parseCmd(args, "read", 1, &params))
    {
        *pbRead = TRUE;
         GetSafeExpressionEx(params, &temp, &params);
         *pFbps     = (LwU32)temp;
         return LW_OK;
    }
    else if (parseCmd(args, "setup", 1, &params))
    {
        *pbRead = FALSE;
        GetSafeExpressionEx(params, &temp, &params);
        *pFbps     = (LwU32)temp;
        return LW_OK;
    }
    else
        return LW_ERR_GENERIC;
}

//-----------------------------------------------------
// fb access monitor
//
//-----------------------------------------------------
DECLARE_API( fbmonitor )
{
    LwU32 nFbps;
    BOOL bRead;

    CHECK_INIT(MODE_LIVE);

    if (LW_ERR_GENERIC == fbMonitorParseOption(args, &nFbps, &bRead))
    {
        dprintf("lw: Usage: fbmonitor -setup <num of fbp>\n");
        dprintf("lw: Usage: fbmonitor -read <num of fbp>\n");
        dprintf("lw: all args needed.\n");
        return;
    }

    MGPU_LOOP_START;
    {
        if (nFbps != 0)
        {
            pFb[indexGpu].fbMonitorAccess(nFbps, bRead);
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// l2state <fbpartitions>
// - dump the L2 state for given number of fb partitions
//-----------------------------------------------------
DECLARE_API( l2state )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pFb[indexGpu].fbL2State();
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// ismemreq <fbpartitions>
// - check whether there have been mem req pending
//-----------------------------------------------------
DECLARE_API( ismemreq )
{
    LwU32 nFbp;

    CHECK_INIT(MODE_LIVE);

    nFbp = (LwU32) GetSafeExpression(args);

    if (nFbp == 0)
    {
        dprintf("lw: Give num of fb partitions in arg.\n");
        dprintf("lw: ismemreq <nFbp>\n");
        return;
    }
    MGPU_LOOP_START;
    {
        pFb[indexGpu].fbIsMemReq(nFbp);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// msenc
// - MSENC engine commands
//-----------------------------------------------------
DECLARE_API( msenc )
{
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bPriv, bImem, bDmem, bState, bHwCfg, bSpr, bFuse;
    LwU32 dmemSize = 0;
    LwU32 imemSize = 0;
    char  *params;

    CHECK_INIT(MODE_LIVE);

    bHelp      = parseCmd(args, "help",      0, NULL);
    bSupported = parseCmd(args, "supported", 0, NULL);
    bPriv      = parseCmd(args, "priv",      0, NULL);
    bFuse      = parseCmd(args, "fuse",      0, NULL);
    bImem      = parseCmd(args, "imem",      1, &params);
    if (bImem)
    {
        imemSize = (LwU32) GetSafeExpression(params);
    }

    bDmem = parseCmd(args, "dmem", 1, &params);
    if (bDmem)
    {
        dmemSize = (LwU32) GetSafeExpression(params);
    }

    bState     = parseCmd(args, "state",     0, NULL);
    bHwCfg     = parseCmd(args, "hwcfg",     0, NULL);
    bSpr       = parseCmd(args, "spr",       0, NULL);

    if (bHelp)
    {
        msencDisplayHelp();
        return;
    }

    MGPU_LOOP_START;
    {
        lwencId  = (LwU32) GetSafeExpression(args);
        if (bSupported)
        {
            msencIsSupported(indexGpu);
            bHandled = TRUE;
        }
        else if (bPriv)
        {
            msencDumpPriv(indexGpu);
            bHandled = TRUE;
        }
        else if (bFuse)
        {
            msencDumpFuse(indexGpu);
            bHandled = TRUE;
        }
        else if (bImem)
        {
            msencDumpImem(indexGpu, (LwU32)imemSize);
            bHandled = TRUE;
        }
        else if (bDmem)
        {
            msencDumpDmem(indexGpu, dmemSize);
            bHandled = TRUE;
        }
        else if (bState)
        {
            msencTestState(indexGpu);
            bHandled = TRUE;
        }
        else if (bHwCfg)
        {
            msencDisplayHwcfg(indexGpu);
            bHandled = TRUE;
        }
        else if (bSpr)
        {
            msencDisplayFlcnSPR(indexGpu);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        msencDisplayHelp();
    }
}

//-----------------------------------------------------
// ofa
// - OFA engine commands
//-----------------------------------------------------
DECLARE_API(ofa)
{
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bPriv, bImem, bDmem, bState, bHwCfg, bSpr, bFuse;
    LwU32 dmemSize = 0;
    LwU32 imemSize = 0;
    LwU32 offs2MthdOffs = ILWALID_OFFSET;
    char  *params;

    CHECK_INIT(MODE_LIVE);

    bHelp = parseCmd(args, "help", 0, NULL);
    bSupported = parseCmd(args, "supported", 0, NULL);
    bPriv = parseCmd(args, "priv", 0, NULL);
    bFuse = parseCmd(args, "fuse", 0, NULL);
    bImem = parseCmd(args, "imem", 1, &params);
    if (bImem)
    {
        imemSize = (LwU32)GetSafeExpression(params);
    }

    bDmem = parseCmd(args, "dmem", 1, &params);
    if (bDmem)
    {
        dmemSize = (LwU32)GetSafeExpression(params);
        if (parseCmd(args, "offs2mthdoffs", 1, &params))
        {
            offs2MthdOffs = (LwU32)GetSafeExpression(params);
        }
    }

    bState = parseCmd(args, "state", 0, NULL);
    bHwCfg = parseCmd(args, "hwcfg", 0, NULL);
    bSpr = parseCmd(args, "spr", 0, NULL);

    if (bHelp)
    {
        ofaDisplayHelp();
        return;
    }

    MGPU_LOOP_START;
    {
        ofaId = (LwU32)GetSafeExpression(args);
        if (bSupported)
        {
            ofaIsGpuSupported(indexGpu);
            bHandled = TRUE;
        }
        else if (bPriv)
        {
            ofaDumpPriv(indexGpu);
            bHandled = TRUE;
        }
        else if (bFuse)
        {
            ofaDumpFuse(indexGpu);
            bHandled = TRUE;
        }
        else if (bImem)
        {
            ofaDumpImem(indexGpu, (LwU32)imemSize);
            bHandled = TRUE;
        }
        else if (bDmem)
        {
            ofaDumpDmem(indexGpu, dmemSize, offs2MthdOffs);
            bHandled = TRUE;
        }
        else if (bState)
        {
            ofaTestState(indexGpu);
            bHandled = TRUE;
        }
        else if (bHwCfg)
        {
            ofaDisplayHwcfg(indexGpu);
            bHandled = TRUE;
        }
        else if (bSpr)
        {
            ofaDisplayFlcnSPR(indexGpu);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        ofaDisplayHelp();
    }
}

//-----------------------------------------------------
// hda
// - HDA engine commands
//-----------------------------------------------------
DECLARE_API( hda )
{
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bImem, bDmem, bState;
    LwU32 dmemSize = 0;
    LwU32 imemSize = 0;
    char  *params;

    CHECK_INIT(MODE_LIVE);

    bHelp      = parseCmd(args, "help",      0, NULL);
    bSupported = parseCmd(args, "supported", 0, NULL);
    bImem      = parseCmd(args, "imem",      1, &params);
    if (bImem)
    {
        imemSize = (LwU32) GetSafeExpression(params);
    }
    bDmem      = parseCmd(args, "dmem",      1, &params);
    if (bDmem)
    {
        dmemSize = (LwU32) GetSafeExpression(params);
    }
    bState     = parseCmd(args, "state",     0, NULL);

    if (bHelp)
    {
        hdaDisplayHelp();
        return;
    }

    MGPU_LOOP_START;
    {
        if (bSupported)
        {
            hdaIsSupported(indexGpu);
            bHandled = TRUE;
        }
        else if (bImem)
        {
            hdaDumpImem(indexGpu, (LwU32)imemSize);
            bHandled = TRUE;
        }
        else if (bDmem)
        {
            hdaDumpDmem(indexGpu, dmemSize);
            bHandled = TRUE;
        }
        else if (bState)
        {
            hdaTestState(indexGpu);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        hdaDisplayHelp();
    }
}

//-----------------------------------------------------
// sec
// - SEC engine commands
//-----------------------------------------------------
DECLARE_API( sec )
{
    char *params;
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bPriv, bImem, bDmem, bState, bHwCfg, bSpr;
    LwU32 dmemSize = 0;
    LwU32 imemSize = 0;

    CHECK_INIT(MODE_LIVE);

    bHelp      = parseCmd(args, "help",      0, NULL);
    bSupported = parseCmd(args, "supported", 0, NULL);
    bPriv      = parseCmd(args, "priv",      0, NULL);
    bImem      = parseCmd(args, "imem",      1, &params);
    if (bImem)
    {
        imemSize = (LwU32) GetSafeExpression(params);
    }
    bDmem      = parseCmd(args, "dmem",      1, &params);
    if (bDmem)
    {
        dmemSize = (LwU32) GetSafeExpression(params);
    }
    bState     = parseCmd(args, "state",     0, NULL);
    bHwCfg     = parseCmd(args, "hwcfg",     0, NULL);
    bSpr       = parseCmd(args, "spr",       0, NULL);

    if (bHelp)
    {
        secDisplayHelp();
        return;
    }

    MGPU_LOOP_START;
    {
        if (bSupported)
        {
            secIsSupported(indexGpu);
            bHandled = TRUE;
        }
        else if (bPriv)
        {
            secDumpPriv(indexGpu);
            bHandled = TRUE;
        }
        else if (bImem)
        {
            secDumpImem(indexGpu, (LwU32)imemSize);
            bHandled = TRUE;
        }
        else if (bDmem)
        {
            secDumpDmem(indexGpu, dmemSize);
            bHandled = TRUE;
        }
        else if (bState)
        {
            secTestState(indexGpu);
            bHandled = TRUE;
        }
        else if (bHwCfg)
        {
            secDisplayHwcfg(indexGpu);
            bHandled = TRUE;
        }
        else if (bSpr)
        {
            secDisplayFlcnSPR(indexGpu);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        secDisplayHelp();
    }
}

//-----------------------------------------------------
// dchlwars
//
//-----------------------------------------------------
DECLARE_API( dchlwars )
{
    char *chName = NULL;
    LwS32 chNum = 0, argNum;
    LwU32 headNum = 0, temp = 0;
    LwU32 i, k;
    BOOL printHeadless = TRUE;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    for (argNum = 0; argNum < dArgc; ++argNum)
    {
        if (sscanf(dArgv[argNum], "-h%d", &temp))
        {
            headNum = temp;
        }
        else if (strcmp(dArgv[argNum], "-noheadless") == 0)
        {
            printHeadless = FALSE;
        }
        else if ( (chNum = pDisp[indexGpu].dispGetChanNum(dArgv[argNum], temp)) != -1)
        {
            chName = dArgv[argNum];
        }
        else
        {
            dprintf("Bad argument %s\n", dArgv[argNum]);
            dprintf("lw: Usage: !lw.dchlwars [chName] [-h<hd>] [-noheadless]\n");
            return;
        }
    }
    if (headNum > pDisp[indexGpu].dispGetNumHeads())
    {
        dprintf("Invalid Head %d\n",headNum);
        return;
    }

    if ( chName != NULL )
    {
        if ( (chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum)) != -1)
        {
            pDisp[indexGpu].dispPrintChalwars(chNum, printHeadless);
        }
    }
    else
    {
        k = pDisp[indexGpu].dispGetMaxChan();
        for (i = 0; i < k; i++)
        {
            pDisp[indexGpu].dispPrintChalwars(i, printHeadless);
        }
    }
}

//-----------------------------------------------------
// pgob
// - Read PG On Boot related registers/fuse
//-----------------------------------------------------
DECLARE_API( pgob )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pMc[indexGpu].mcReadPgOnBootStatus();
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// ecc
// - Read ECC related registers
//-----------------------------------------------------
DECLARE_API( ecc )
{
    BOOL isFullPrint = FALSE;

    CHECK_INIT(MODE_LIVE);

    isFullPrint = parseCmd(args, "a", 0, NULL);

    MGPU_LOOP_START;
    {
        pFb[indexGpu].fbGetEccInfo(isFullPrint);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}


//-----------------------------------------------------------------------------------
// Tex Parity[-grIdx] <grIdx>
// <grIdx> - Graphics index required to access PGRAPH registers when SMC is enabled
// - Read all Tex Parity relevant info
//-----------------------------------------------------------------------------------
DECLARE_API( texparity )
{
    BOOL isGrIndex = FALSE;
    LwU32 grIdx = 0;
    char *param;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
    }
    else if (isGrIndex && *param != '0')
    {
        dprintf("lw: non-zero grIndex specified without SMC\n");
        return;
    }

    MGPU_LOOP_START;
    {
        pGr[indexGpu].grGetTexParityInfo( grIdx );
        dprintf("\n");
    }
    MGPU_LOOP_END;
    pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
}

//-----------------------------------------------------
// privhistory
// - Dump the Priv History Buffer (GF119, Kepler)
//-----------------------------------------------------
DECLARE_API( privhistory )
{
    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        pPriv[indexGpu].privDumpPriHistoryBuffer();
    }
    MGPU_LOOP_END;
}

//-------------------------------------------
// dsli
// - Display SLI exchanges for debug
//-------------------------------------------
DECLARE_API( dsli )
{
    LwU32 verbose = 0;

    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);
    if (dArgc > 0)
    {
        verbose = (strcmp(dArgv[0],  "-v") ? 0 : 1);
    }
    MGPU_LOOP_START;
    {
        pDisp[indexGpu].dispDumpSliConfig(verbose);
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// elpg
// - Runs various elpg related functions
//-----------------------------------------------------
DECLARE_API( elpg )
{
    char *params;
    LW_STATUS status;
    LwU64 value;

    CHECK_INIT(MODE_LIVE);

    // Dump ELPG state
    if (parseCmd(args, "status", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            status = elpgGetStatus();
            if (status != LW_OK)
            {
                dprintf("lw:  **************************************************************************\n");
                dprintf("lw:  ERROR reported above by elpg -status...\n");
                dprintf("lw:  **************************************************************************\n");
                dprintf("lw:  \n");
            }
        }
        MGPU_LOOP_END;
    }
    // Restart using the given ELPG controller.
    else if (parseCmd(args, "start", 1, &params))
    {
        MGPU_LOOP_START;
        {
            value = GetSafeExpression(params);
            if (value != (LwU32)value)
            {
                dprintf("lw:  ERROR: value must be a 32-bit number\n");
                status = LW_ERR_GENERIC;
            }
            else
            {
                status = pElpg[indexGpu].elpgStart((LwU32)value);
            }

            if (status != LW_OK)
            {
                dprintf("lw:  **************************************************************************\n");
                dprintf("lw:  ERROR reported above by elpg -start...\n");
                dprintf("lw:  **************************************************************************\n");
                dprintf("lw:  \n");
            }
            else
            {
                dprintf("lw:  ELPG%d has been started.\n", (LwU32)value);
                            dprintf("lw:  For RM ELPG, continue the debugger to allow its associated engine(s) to power-gate...\n");
                dprintf("lw:  \n");
            }
        }
        MGPU_LOOP_END;
    }
    //
    // Stop using the given ELPG controller and force its associated engine to
    // wakeup.
    //
    // NOTE: This command disables ELPG 'power-gate on' interrupts (PG_ON and
    // PG_ON_DONE).
    //
    else if (parseCmd(args, "stop", 1, &params))
    {
        MGPU_LOOP_START;
        {
            value = GetSafeExpression(params);
            if (value != (LwU32)value)
            {
                dprintf("lw:  ERROR: value must be a 32-bit number\n");
                status = LW_ERR_GENERIC;
            }
            else
            {
                status = pElpg[indexGpu].elpgStop((LwU32)value);
            }

            if (status != LW_OK)
            {
                dprintf("lw:  **************************************************************************\n");
                dprintf("lw:  ERROR reported above by elpg -stop...\n");
                dprintf("lw:  **************************************************************************\n");
                dprintf("lw:  \n");
            }
            else
            {
                dprintf("lw:  ELPG%d has been stopped.\n", (LwU32)value);
                dprintf("lw:  For RM ELPG, continue the debugger to allow its associated engine(s) to power-on...\n");
                dprintf("lw:  \n");
            }
        }
        MGPU_LOOP_END;
    }
    // Dump PG log
    else if (parseCmd(args, "dumpLog", 0, NULL))
    {
        MGPU_LOOP_START;
        {
            elpgDumpPgLog();
        }
        MGPU_LOOP_END;
    }
    // Display the ELPG command help menu
    else
    {
        elpgDisplayHelp();
    }
}

//-----------------------------------------------------
// ce
// - CE engine commands
//-----------------------------------------------------
DECLARE_API( ce )
{
    LwU32 argCe = 0, numArgs = 0, temp = 0, indexCe;
    char argCommend[128], argExtra[128];
    BOOL bHandled = FALSE;
    BOOL bAllEngines = FALSE;
    BOOL bValidArgs = FALSE;

    CHECK_INIT(MODE_LIVE);

    // Read our command line and parse, look for a lone command => all engines
    numArgs = sscanf(args, "%s %s", argCommend, argExtra);
    if ((numArgs == 1) && (argCommend[0]  != '\0'))
    {

        bAllEngines = TRUE;
        bValidArgs = TRUE;
    }

    if (!bValidArgs)
    {
        // Read our command line and parse, look for a command and index number
        numArgs = sscanf(args, "%s %d", argCommend, &temp);
        if ((numArgs == 2) && (argCommend[0] != '\0'))
        {
            argCe = temp;
            bValidArgs = TRUE;

            if (!ceIsSupported( indexGpu, argCe ))
            {
                return;
            }
        }
    }

    if (bValidArgs)
    {
        if (strcmp(argCommend, "help")==0)
        {
            ceDisplayHelp();
            return;
        }

        MGPU_LOOP_START;
        {
            if (strcmp(argCommend, "pcelcemap")==0)
            {
                cePrintPceLceMap(indexGpu);
                bHandled = TRUE;
            }

            for (indexCe = 0;;indexCe++)
            {
                if (ceIsValid(indexGpu,indexCe))
                {
                    if (bAllEngines || (indexCe == argCe))
                    {
                        if (strcmp(argCommend, "supported")==0)
                        {
                            ceIsSupported(indexGpu, indexCe);
                            bHandled = TRUE;
                        }
                        else if (strcmp(argCommend, "priv")==0)
                        {
                            ceDumpPriv(indexGpu, indexCe);
                            bHandled = TRUE;
                        }
                        else if (strcmp(argCommend, "state")==0)
                        {
                            ceTestState(indexGpu, indexCe);
                            bHandled = TRUE;
                        }
                    }
                }
                else
                {
                    bHandled = TRUE;
                    break;
                }
            }
        }
        MGPU_LOOP_END;
    }
    else
    {
        dprintf("error: Invalid arguments: %s\n", args);
        ceDisplayHelp();
        return;
    }

    if (!bHandled)
    {
        dprintf("error: Unknown CE command: %s\n", argCommend);
        ceDisplayHelp();
    }
}


//-----------------------------------------------------
// hdcp
// display HDCP information
//-----------------------------------------------------
DECLARE_API( hdcp )
{
    CHECK_INIT(MODE_LIVE);

    dispParseArgs(args);

    switch (dArgc)
    {
    case 1:
        if(!strcmp("status", args))
        {
            if(LW_ERR_NOT_SUPPORTED == pDisp[indexGpu].dispHdcpPrintStatus())
            {
                dprintf("HDCP: not supported chip\n");
            }    
            return;
        }
        else if(!strcmp("--help", args))
        {
           hdcpDisplayHelp();
           return;
        }
        goto hdcp_bad_usage;
        break;

    case 2:
         if(!strcmp("keydecryption", dArgv[0]) && !strcmp("status", dArgv[1]))
        {
            if(LW_ERR_NOT_SUPPORTED == pDisp[indexGpu].dispHdcpKeydecryptionStatus())
            {
                dprintf("HDCP not supported\n");
            }
            return;
        }
        else if(!strcmp("keydecryption", dArgv[0]) && !strcmp("trigger", dArgv[1]))
        {
            if(LW_ERR_NOT_SUPPORTED == pDisp[indexGpu].dispHdcpKeydecryptionTrigger())
            {
                dprintf("HDCP not supported\n");
            }
            return;
        }    
        goto hdcp_bad_usage;
        break;

    case 3:
        if(!strcmp("status", dArgv[0]) && !strcmp("SOR", struppr(dArgv[1])))
        {
            if((strlen(dArgv[2]) == 1) && 
               ((!strcmp("*", dArgv[2])) || ((strcmp("0", dArgv[2]) <= 0 ) && (strcmp("7", dArgv[2]) >= 0))))
            {
                if(LW_ERR_NOT_SUPPORTED == pDisp[indexGpu].dispPrintHdcp22Status(dArgv[2]))
                {
                   dprintf("HDCP 2.2 : not supported chip\n");
                }
            }
            else
            {
               dprintf("Invalid args! \n");
               dprintf("Usage: \"hdcp status SOR [0-7]\" prints the register info of a particular SOR or,\
                       \n       \"hdcp status SOR *\" prints the register info of all the attached SORs\n\n");
            }
            return;
        }
        goto hdcp_bad_usage;
        break;

    default:
        goto hdcp_bad_usage;
        break;
    }

    hdcp_bad_usage:
    dprintf("Invalid args: For usage run \"hdcp --help\"\n\n");
}

//-----------------------------------------------------
// Falcon Debugger
//-----------------------------------------------------
DECLARE_API( flcngdb )
{
    char* sessionName;
    char* symPath;

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

    dprintf("Error: FLCNGDB not supported on this platform\n");

#else

    CHECK_INIT(MODE_LIVE);

    // Lwrrently flcngdb supported only on lwwatch on mods and WinDbg
    dispParseArgs(args);

    if (dArgc < 1)
    {
        dprintf("lw: Usage: !lw.flcngdb <sessionID> [symbol file full path]\n");
        dprintf("\t-symbol file full path: no whilespace should be used\n");
        dprintf("\t-A new session will be created for new sessionIDs\n");
        return ;
    }

    sessionName = dArgv[0];
    symPath = dArgv[1];
    flcngdbMenu(sessionName, symPath);

#endif
}

//---------------------------------------------------------
// fecsdmemrd <offset> [length(bytes)] [port] [size(bytes)]
// - read the DMEM in the range offset-offset+length
//---------------------------------------------------------
DECLARE_API( fecsdmemrd )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32  engineBase      = 0x0;
    LwU64  offset          = 0x0;
    LwU64  lengthInBytes   = 0x80;
    LwU64  port            = 0x0;
    LwU64  size            = 0x4;
    LwU32  memSize         = 0x0;
    LwU32  numPorts        = 0x0;
    LwU32  length          = 0x0;
    LwU32* buffer          = NULL;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.fecsdmemrd <offset> [length(bytes)] " \
                "[port] [size]\n");
        return;
    }

    if (args[0] == '\0')
    {
        dprintf("lw: Usage: !lw.fecsdmemrd <offset> [length(bytes)] " \
                "[port] [size]\n");
        dprintf("lw: No args specified, defaulted to offset" \
                                " 0x%04x and length 0x%04x bytes.\n",
                                 (LwU32)offset, (LwU32)lengthInBytes);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetSafeExpressionEx(args, &offset, &args))
    {
        if (GetSafeExpressionEx(args, &lengthInBytes, &args))
        {
            if (GetSafeExpressionEx(args, &port, &args))
            {
                GetSafeExpressionEx(args, &size, &args);
            }
        }
    }

    // Tidy up the length to be 4-byte aligned
    lengthInBytes = (lengthInBytes + 3) & ~3ULL;
    offset = offset & ~3ULL;


    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pFecs[indexGpu].fecsGetFalconCoreIFace();
        pFEIF = pFecs[indexGpu].fecsGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnDmemGetSize(engineBase);
        numPorts   = pFCIF->flcnDmemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%x is too large (DMEM size 0x%x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the DMEM

        length = pFCIF->flcnDmemRead(engineBase,
                                       (LwU32)offset,
                                       LW_FALSE,
                                       (LwU32)lengthInBytes / sizeof(LwU32),
                                       (LwU32)port, buffer);

        // Dump out the DMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping DMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, (LwU8)size);
        }

        // Cleanup after ourselves
        free(buffer);
    }
    MGPU_LOOP_END;
}

//--------------------------------------------------------------------------------------------
// fecsdmemwr <offset> <value> [-w width(bytes)] [length(units of width)] [-p <port>] [-s <size>]
// - write 'value' of 'width' to DMEM in the range offset-offset+length
//--------------------------------------------------------------------------------------------
DECLARE_API( fecsdmemwr )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32   engineBase    = 0x0;
    LwU64   offset        = 0x0;
    LwU64   length        = 0x1;
    LwU64   width         = 0x4;
    LwU64   value         = 0x0;
    LwU64   port          = 0x0;
    LwU64   size          = 0x4;
    LwU32   memSize       = 0x0;
    LwU32   numPorts      = 0x0;
    LwU32   bytesWritten  = 0x0;
    LwU32   endOffset     = 0x0;
    char   *pParams;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Offset not specified.\n");
        dprintf("lw: Usage: !lw.fecsdmemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }

    // extract optional arguments first
    if (parseCmd(args, "w", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &width, &pParams);
    }

    if (parseCmd(args, "l", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &length, &pParams);
    }

    if (parseCmd(args, "p", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &port, &pParams);
    }

    if (parseCmd(args, "s", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &size, &pParams);
    }

    // Read in the <offset> <value>, in that order, if present
    if (!GetSafeExpressionEx(args, &offset, &args))
    {
        dprintf("lw: Value not specified.\n");
        dprintf("lw: Usage: !lw.fecsdmemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>] [-s <size>]\n");
        return;
    }
    GetSafeExpressionEx(args, &value, &args);

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pFecs[indexGpu].fecsGetFalconCoreIFace();
        pFEIF = pFecs[indexGpu].fecsGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize  = pFCIF->flcnDmemGetSize(engineBase);
        numPorts = pFCIF->flcnDmemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the DMEM

        dprintf("lw:\tWriting DMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at DMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pFCIF->flcnDmemWrite(engineBase,
                                              (LwU32)offset,
                                              LW_FALSE,
                                              (LwU32)value,
                                              (LwU32)width,
                                              (LwU32)length,
                                              (LwU32)port);

        if (bytesWritten == ((LwU32)width * (LwU32)length))
        {
            dprintf("lw:\n");
            endOffset = (LwU32)offset + ((LwU32)width * (LwU32)length);
            flcnDmemDump(pFEIF,
                         (LwU32)offset & ~0x3,
                         endOffset - ((LwU32)offset & ~0x3),
                         (LwU8)port,
                         (LwU8)size);
        }
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// fecsimemrd <offset> [length(bytes)] [port]
// - read the IMEM in the range offset-offset+length
//-----------------------------------------------------
DECLARE_API( fecsimemrd )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32    engineBase      = 0x0;
    LwU64    offset          = 0x0;
    LwU64    lengthInBytes   = 0x80;
    LwU64    port            = 0x0;
    LwU32    memSize         = 0x0;
    LwU32    numPorts        = 0x0;
    LwU32    length          = 0x0;
    LwU32   *buffer          = NULL;

    CHECK_INIT(MODE_LIVE);

    if (!args)
    {
        dprintf("lw: Usage: !lw.fecsimemrd <offset> [length(bytes)] [port]\n");
        return;
    }

    if (args[0] == '\0')
    {
        dprintf("lw: Usage: !lw.fecsimemrd <offset> [length(bytes)] [port]\n");
        dprintf("lw: No args specified, defaulted to offset" \
                                " 0x%04x and length 0x%04x bytes.\n",
                                 (LwU32)offset, (LwU32)lengthInBytes);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetSafeExpressionEx(args, &offset, &args))
    {
        if (GetSafeExpressionEx(args, &lengthInBytes, &args))
        {
            GetSafeExpressionEx(args, &port, &args);
        }
    }

    // Tidy up the length and offset to be 4-byte aligned
    lengthInBytes   = (lengthInBytes + 3) & ~3ULL;
    offset          = offset & ~3ULL;

    // Get the size of the IMEM and number of IMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pFecs[indexGpu].fecsGetFalconCoreIFace();
        pFEIF = pFecs[indexGpu].fecsGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnImemGetSize(engineBase);
        numPorts   = pFCIF->flcnImemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%04x is too large (IMEM size 0x%04x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the IMEM
        length = pFCIF->flcnImemRead(engineBase,
                                       (LwU32)offset,
                                       (LwU32)lengthInBytes / sizeof(LwU32),
                                       (LwU32)port, buffer);

        // Dump out the IMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping FECS IMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, 0x4);
        }

        // Cleanup after ourselves
        free((void*)buffer);
    }
    MGPU_LOOP_END;
}

//--------------------------------------------------------------------------------------------
// fecsimemwr <offset> <value> [width(bytes)] [length(units of width)] [-p <port>]
// - write 'value' to IMEM in the range offset-offset+length
//--------------------------------------------------------------------------------------------
DECLARE_API( fecsimemwr )
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32  engineBase    = 0x0;
    LwU64  offset        = 0x0;
    LwU64  length        = 0x1;
    LwU64  width         = 0x4;
    LwU64  value         = 0x0;
    LwU64  port          = 0x0;
    LwU32  memSize       = 0x0;
    LwU32  numPorts      = 0x0;
    LwU32  bytesWritten  = 0x0;
    char  *pParams;

    CHECK_INIT(MODE_LIVE);

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Offset not specified.\n");
        dprintf("lw: Usage: !lw.fecsimemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>]\n");
        return;
    }

    // extract optional arguments first
    if (parseCmd(args, "w", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &width, &pParams);
    }

    if (parseCmd(args, "l", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &length, &pParams);
    }

    if (parseCmd(args, "p", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &port, &pParams);
    }

    // Read in the <offset> <value>, in that order, if present
    if (!GetSafeExpressionEx(args, &offset, &args))
    {
        dprintf("lw: Value not specified.\n");
        dprintf("lw: Usage: !lw.fecsimemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>]\n");
        return;
    }
    GetSafeExpressionEx(args, &value, &args);

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF = pFecs[indexGpu].fecsGetFalconCoreIFace();
        pFEIF = pFecs[indexGpu].fecsGetFalconEngineIFace();
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnImemGetSize(engineBase);
        numPorts   = pFCIF->flcnImemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the IMEM

        dprintf("lw:\tWriting IMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at IMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pFCIF->flcnImemWrite(
                           engineBase,
                           (LwU32)offset,
                           (LwU32)value,
                           (LwU32)width,
                           (LwU32)length,
                           (LwU32)port);


        dprintf("lw: number of bytes written: 0x%x\n", bytesWritten);
    }
    MGPU_LOOP_END;
}

//-----------------------------------------------------
// lwdec
// - LWDEC engine commands
//-----------------------------------------------------
DECLARE_API( lwdec )
{
    char *params;
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bPriv, bImem, bDmem, bState, bHwCfg, bSpr, bFuse;
    LwU32 dmemSize = 0;
    LwU32 imemSize = 0;
    LwU32 lwdecId = 0;

    CHECK_INIT(MODE_LIVE);

    bHelp      = parseCmd(args, "help",      0, NULL);
    bSupported = parseCmd(args, "supported", 0, NULL);
    bPriv      = parseCmd(args, "priv",      0, NULL);
    bFuse      = parseCmd(args, "fuse",      0, NULL);
    bImem      = parseCmd(args, "imem",      1, &params);
    if (bImem)
    {
        imemSize = (LwU32) GetSafeExpression(params);
    }
    bDmem      = parseCmd(args, "dmem",      1, &params);
    if (bDmem)
    {
        dmemSize = (LwU32) GetSafeExpression(params);
    }
    bState     = parseCmd(args, "state",     0, NULL);
    bHwCfg     = parseCmd(args, "hwcfg",     0, NULL);
    bSpr       = parseCmd(args, "spr",       0, NULL);

    if (bHelp)
    {
        lwdecDisplayHelp();
        return;
    }

    lwdecId  = (LwU32) GetSafeExpression(args);

    MGPU_LOOP_START;
    {
        if (bSupported)
        {
            lwdecIsSupported(indexGpu, lwdecId);
            bHandled = TRUE;
        }
        else if (bPriv)
        {
            lwdecDumpPriv(indexGpu, lwdecId);
            bHandled = TRUE;
        }
        else if (bFuse)
        {
            lwdecDumpFuse(indexGpu);
            bHandled = TRUE;
        }
        else if (bImem)
        {
            lwdecDumpImem(indexGpu, lwdecId, (LwU32)imemSize);
            bHandled = TRUE;
        }
        else if (bDmem)
        {
            lwdecDumpDmem(indexGpu, lwdecId, dmemSize);
            bHandled = TRUE;
        }
        else if (bState)
        {
            lwdecTestState(indexGpu, lwdecId);
            bHandled = TRUE;
        }
        else if (bHwCfg)
        {
            lwdecDisplayHwcfg(indexGpu, lwdecId);
            bHandled = TRUE;
        }
        else if (bSpr)
        {
            lwdecDisplayFlcnSPR(indexGpu, lwdecId);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        lwdecDisplayHelp();
    }
}

//-----------------------------------------------------
// lwjpg
// - LWJPG engine commands
//-----------------------------------------------------
DECLARE_API(lwjpg)
{
    char *params;
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bPriv, bImem, bDmem, bState, bHwCfg, bSpr, bFuse;
    LwU32 dmemSize = 0;
    LwU32 imemSize = 0;
    LwU32 lwjpgId = 0;

    CHECK_INIT(MODE_LIVE);

    bHelp = parseCmd(args, "help", 0, NULL);
    bSupported = parseCmd(args, "supported", 0, NULL);
    bPriv = parseCmd(args, "priv", 0, NULL);
    bFuse = parseCmd(args, "fuse", 0, NULL);
    bImem = parseCmd(args, "imem", 1, &params);
    if (bImem)
    {
        imemSize = (LwU32)GetSafeExpression(params);
    }
    bDmem = parseCmd(args, "dmem", 1, &params);
    if (bDmem)
    {
        dmemSize = (LwU32)GetSafeExpression(params);
    }
    bState = parseCmd(args, "state", 0, NULL);
    bHwCfg = parseCmd(args, "hwcfg", 0, NULL);
    bSpr = parseCmd(args, "spr", 0, NULL);

    if (bHelp)
    {
        lwjpgDisplayHelp();
        return;
    }

    lwjpgId = (LwU32)GetSafeExpression(args);

    MGPU_LOOP_START;
    {
        if (bSupported)
        {
            lwjpgIsSupported(indexGpu, lwjpgId);
            bHandled = TRUE;
        }
        else if (bPriv)
        {
            lwjpgDumpPriv(indexGpu, lwjpgId);
            bHandled = TRUE;
        }
        else if (bFuse)
        {
            lwjpgDumpFuse(indexGpu, lwjpgId);
            bHandled = TRUE;
        }
        else if (bImem)
        {
            lwjpgDumpImem(indexGpu, lwjpgId, (LwU32)imemSize);
            bHandled = TRUE;
        }
        else if (bDmem)
        {
            lwjpgDumpDmem(indexGpu, lwjpgId, dmemSize);
            bHandled = TRUE;
        }
        else if (bState)
        {
            lwjpgTestState(indexGpu, lwjpgId);
            bHandled = TRUE;
        }
        else if (bHwCfg)
        {
            lwjpgDisplayHwcfg(indexGpu, lwjpgId);
            bHandled = TRUE;
        }
        else if (bSpr)
        {
            lwjpgDisplayFlcnSPR(indexGpu, lwjpgId);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        lwjpgDisplayHelp();
    }
}

//-----------------------------------------------------
// LwSym
// - Commands for LwSym debug metadata distribution
//-----------------------------------------------------
// Only enabled a windbg extension right now
#include "lwsym.h"

#if LWSYM_ENABLED

LWSYM_PACKAGE *pLwsymPackage = NULL;

DECLARE_API( lwsym )
{
    char *param = (char *)args;
    LWSYM_STATUS status;
    CHECK_INIT(MODE_LIVE);

    if (parseCmd((char *)args, "unload", 0, &param))
    {
        lwsymFreePackage(pLwsymPackage);
        pLwsymPackage = NULL;
        dprintf("lw: LwSym package unloaded.\n");
        return;
    }

    if (parseCmd((char *)args, "load", 1, &param))
    {
        FILE *f = fopen(param, "rb");
        if (f == NULL)
        {
            dprintf("lw: Error opening file %s\n", param);
            return;
        }
        lwsymFreePackage(pLwsymPackage);
        status = lwsymReadPackage(&pLwsymPackage, f);
        if (status != LWSYM_OK)
            dprintf("lw: %s\n", lwsymGetStatusMessage(status));
        else
            dprintf("lw: LwSym package read from %s\n", param);

        fclose(f);
        return;
    }

    if (lwsymInit() == NULL)
        return;

    if (parseCmd((char *)args, "list", 0, &param))
    {
        char **names = NULL;
        LwU32 numSegments = 0, i;
        lwsymGetSegmentNames(pLwsymPackage, &names, &numSegments);

        dprintf("lw: Segments in loaded LwSym package:\n");
        for (i = 0; i < numSegments; i++)
        {
            LwU32 size;
            lwsymGetSegmentSize(pLwsymPackage, i, &size);
            dprintf("lw: - %s%-32s [%8d bytes]\n", LWSYM_VIRUTAL_PATH, names[i], size);
            free(names[i]);
        }
        free(names);
    }
    else if (parseCmd((char *)args, "extract", 1, &param))
    {
        FILE *fout = fopen(param, "wb");
        if (fout == NULL)
        {
            dprintf("lw: Unable to create file: %s\n", param);
        }
        else
        {
            status = lwsymWritePackage(pLwsymPackage, fout);
            if (status == LWSYM_OK)
                dprintf("lw: LwSym package extracted to %s\n", param);
            else
                dprintf("lw: %s\n", lwsymGetStatusMessage(status));
            fclose(fout);
        }
    }
}
#endif // LWSYM_ENABLED

//-----------------------------------------------------
// acr
// - acr commands
//-----------------------------------------------------
DECLARE_API( acr )
{
    char *params;
    params  = (char *)args;
    CHECK_INIT(MODE_LIVE);

    if (parseCmd(params, "help", 0, &params))
    {
        acrDisplayHelp();
    }
    else if (parseCmd(params, "supported", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrIsSupported(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if (parseCmd(params, "lsfstatus", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrLsfStatus(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if ( parseCmd(params, "getregioninfo", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrGetRegionInfo(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if ( parseCmd(params, "regionstatus", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrRegionStatus(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if ( parseCmd(params, "dmemprotection", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrDmemProtection(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if ( parseCmd(params, "imemprotection", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrImemProtection(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if ( parseCmd(params, "getmwprinfo", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrGetMultipleWprInfo(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if ( parseCmd(params, "mwprstatus", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrVerifyMultipleWprStatus(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else if ( parseCmd(params, "sharedwprstatus", 0, &params))
    {
        MGPU_LOOP_START;
        {
            acrGetSharedWprStatus(indexGpu);
        }
        MGPU_LOOP_END;
    }
    else
    {
        acrDisplayHelp();
    }
}

//-----------------------------------------------------
// psdl
// - psdl commands
//-----------------------------------------------------
DECLARE_API( psdl )
{
    char *pFalconName = NULL;
    char *pFilename   = NULL;
    void  *pData      = NULL;
    FILE  *pFile      = NULL;
    LwU32 length      = 0;
    LwU32 fpos        = 0;

    dprintf("lw:\t======================================================================\n");
    // Check if PSDL is supported or not
    if (pPsdl[indexGpu].psdlIsSupported() == LW_FALSE)
    {
        dprintf("lw:\tPSDL ->  Not supported\n");
        goto label_return;
    }

    // Lwrrently flcngdb supported only on lwwatch on mods and WinDbg
    dispParseArgs((char *) args);

    if (dArgc == 1)
    {
        if (!strcmp(dArgv[0], "ecid"))
        {
            pPsdl[indexGpu].psdlPrintEcid(indexGpu);
            goto label_return;
        }
        else
        {
            goto printhelp;
        }
    }
    else if (dArgc == 2)
    {

        pFalconName = dArgv[0];
        pFilename   = dArgv[1];

        // Open PSDL license
        pFile = fopen(pFilename, "rb");
        if (pFile == NULL)
        {
            dprintf("lw:\tPSDL -> Opening license file \"'%s'\" FAILED\n", pFilename);
        }
        else
        {
            fseek(pFile, 0, SEEK_END);
            fpos = ftell(pFile);
            if (fpos > 0)
            {
                fseek(pFile, 0, SEEK_SET);
                length = fpos;
                pData = malloc(length);
                if (pData != NULL)
                {
                    length = (LwU32)fread(pData, 1, length, pFile);
                    if (length < (LwU32)fpos)
                    {
                        dprintf("lw:\tPSDL -> Cert read FAILED\n");
                        free(pData);
                        pData = NULL;
                    }
                }
                else
                {
                    dprintf("lw:\tPSDL -> Memory allocation FAILED\n");
                }
            }
            fclose(pFile);
        }

        if (!pData)
        {
            goto printhelp;
        }

        if (!strcmp(pFalconName, "sec2"))
        {
            //dprintf("lw:\t\tPSDL -> Not enabled for SEC2 yet. FAIL\n");
            pPsdl[indexGpu].psdlUseSec2(indexGpu, pData, length);
        }
        else if (!strcmp(pFalconName, "pmu"))
        {
            pPsdl[indexGpu].psdlUsePmu(indexGpu, pData, length);
        }
        else if (!strcmp(pFalconName, "lwdec"))
        {
            //dprintf("lw:\t\tPSDL -> Not enabled for LWDEC yet. FAIL\n");
            pPsdl[indexGpu].psdlUseLwdec(indexGpu, pData, length);
        }
        else
        {
            dprintf("lw:\t\tPSDL -> Invalid falcon identifier \n");
            goto printhelp;
        }
    }
    else
    {
        goto printhelp;
    }

label_return:
    dprintf("lw:\t======================================================================\n\n");
    return;

printhelp:
    dprintf("lw:\tUsage: !lw.psdl <pmu|sec2|lwdec> <PSDL cert abspath>\n");
    dprintf("lw:\tUsage: !lw.psdl <ecid>\n");
    dprintf("lw:\t======================================================================\n\n");
    return;

}

//-----------------------------------------------------
// falcphysdmacheck
// - falcphysdmacheck commands
//-----------------------------------------------------
DECLARE_API( falcphysdmacheck )
{
    char *params;
    BOOL bHandled = FALSE;

    params  = (char *)args;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        if (parseCmd(params, "help", 0, &params))
        {
            falcphysDisplayHelp();
            return;
        }

        if (parseCmd(params, "supported", 0, &params))
        {
            falcphysIsSupported(indexGpu);
            bHandled = TRUE;
        }
        else if (parseCmd(params, "accesscheck", 0, &params))
        {
            falcphysDmaAccessCheck(indexGpu);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        falcphysDisplayHelp();
    }
}

//-----------------------------------------------------
// deviceinfo
// - print the entries from device info
//-----------------------------------------------------
DECLARE_API( deviceinfo )
{
    CHECK_INIT(MODE_LIVE);

    if (isVirtual())
        setLwwatchMode(LW_TRUE);

    MGPU_LOOP_START;
    {
        deviceInfoDump();
        dprintf("\n");
    }
    MGPU_LOOP_END;

    if (isVirtual())
        setLwwatchMode(LW_FALSE);
}

// -----------------------------------------------------
// lwsrinfo <port> <verbose level>
// - LwSR commands
// - displays SRC basic register analysis
// -----------------------------------------------------
DECLARE_API( lwsrinfo )
{
    LwU32 port = 10;   // random value
    LwU32 verbose_level = 0;
    char *usage = "Usage   : !lw.lwsrinfo <port> <verbose level>\n";

    CHECK_INIT(MODE_LIVE);

    // Parse commandline
    dispParseArgs(args);

    if (dArgc < 1)
    {
        dprintf("Too few arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    if (dArgc == 1)
    {
        dprintf("Defaulted to verbose level 0\n");
        verbose_level = 0;
        dprintf("%s\n", usage);
    }
    else
        sscanf(dArgv[1], "%x", &verbose_level);

    if (dArgc > 2)
    {
        dprintf("Too many arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    sscanf(dArgv[0], "%x", &port);
    dprintf("DEBUG   : Port %x\n", port);
    dprintf("DEBUG   : Verbose level %x\n",verbose_level);

    // Call lwsrinfo entry function
    lwsrinfo_entryfunction(port,verbose_level);
}

// -----------------------------------------------------
// lwsrcap <port>
// - LwSR commands
// - displays SRC capabilities register analysis
// -----------------------------------------------------
DECLARE_API( lwsrcap )
{
    LwU32 port;
    char *usage = "Usage   : !lw.lwsrcap <port>\n";

    CHECK_INIT(MODE_LIVE);

    // Parse commandline to get port number
    dispParseArgs(args);
    // dprintf("dArgc = %d\n", dArgc);

    if (dArgc > 1)
    {
        dprintf("Too many arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    if (dArgc < 1)
    {
        dprintf("Too few arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    sscanf(dArgv[0], "%x", &port);
    dprintf("DEBUG   : Port %x\n", port);

    // Call lwsrcap entry function
    lwsrcap_entryfunction(port);
}

// -----------------------------------------------------
// lwsrtiming <port>
// - LwSR commands
// - displays SRC capabilities register analysis
// -----------------------------------------------------
DECLARE_API( lwsrtiming )
{
    LwU32 port;
    char *usage = "Usage   : !lw.lwsrtiming <port>\n";

    CHECK_INIT(MODE_LIVE);

    // Parse commandline to get port number
    dispParseArgs(args);
    // dprintf("dArgc = %d\n", dArgc);

    if (dArgc > 1)
    {
        dprintf("Too many arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    if (dArgc < 1)
    {
        dprintf("Too few arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    sscanf(dArgv[0], "%x", &port);
    dprintf("DEBUG   : Port %x\n", port);

    // Call lwsrtiming entry function
    lwsrtiming_entryfunction(port);
}

// -----------------------------------------------------
// lwsrmutex <port> <sub_func_option> <16 byte key>
// - LwSR commands
// - mutex check and unlock
// -----------------------------------------------------
DECLARE_API( lwsrmutex )
{
    LwU32 port;
    LwU32 sub_function_option = 0;
    LwU8 index = 0;

    char *usage = "Usage   : !lw.lwsrmutex <port> <function option> <16 byte key>\n";

    CHECK_INIT(MODE_LIVE);

    // Parse commandline
    dispParseArgs(args);
    // dprintf("dArgc = %d\n", dArgc);

    if (dArgc < 2)
    {
        dprintf("Too few arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    if (dArgc > 18)
    {
        dprintf("Too many arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    // Parse argument list to get port, user option and secret vendor key
    sscanf(dArgv[0], "%x", &port);
    dprintf("DEBUG   : Port %x\n", port);

    sscanf(dArgv[1], "%x", &sub_function_option);
    dprintf("DEBUG   : Option entered = %x\n\n",sub_function_option);
    if (sub_function_option!= 1 && sub_function_option!= 2)
    {
        dprintf("ERROR   : Bad 'sub_func_option' argument. Choose proper option\n");
        dprintf("HELP    : Input '1' - To check SPRN randomness\n");
        dprintf("HELP    : Input '2' - To unlock LWSR mutex\n\n");
        return;
    }

    // Append zeros in key for bytes not entered by user
    // Workaround for subfunction to be reused IN SAME DEBUG SESSION (initialize/reset key to 0)
    for(index = 0 ; index < 16; index++)
        lwsr_key[index] = 0;

    // Accept vendor key as user input
    if (dArgc > 2)
        for(index = 0 ; index < dArgc-2; index++)
            sscanf(dArgv[index+2], "%x", (LwU32*)&lwsr_key[index]);

    // Call lwsrmutex entry function
    lwsrmutex_entryfunction(port, sub_function_option, dArgc);
}

// -----------------------------------------------------
// lwsrsetrr <port> <sub_func_option> <16 byte key>
// - LwSR commands
// - mutex check and unlock
// -----------------------------------------------------
DECLARE_API( lwsrsetrr )
{
    LwU32 port;
    LwU32 refresh_rate ;

    char *usage = "Usage   : !lw.lwsrsetrr <port> <refresh_rate>\n";

    CHECK_INIT(MODE_LIVE);

    // Parse commandline
    dispParseArgs(args);

    if (dArgc < 2)
    {
        dprintf("Too few arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    if (dArgc > 2)
    {
        dprintf("Too many arguments\n%sCall !lw.help for a command list\n\n", usage);
        return;
    }

    // Parse argument list to get port, user option and secret vendor key
    sscanf(dArgv[0], "%x", &port);
    dprintf("DEBUG   : Port %x\n", port);
    sscanf(dArgv[1], "%d", &refresh_rate);

    // Call lwsrmutex entry function
    lwsrsetrr_entryfunction(port, refresh_rate);
}

//-----------------------------------------------------
// LwLog
// - Commands for decoding and printing LwLog
//-----------------------------------------------------
// Only enabled as windbg extension right now
#include "lwlog.h"

#if LWLOG_CMD_ENABLED

#define LWLOG_DECODE_BUFFER_SIZE (1024*1024*4)
DECLARE_API( lwlog )
{
    char *param = (char *)args;
    FILE *file = NULL;

    CHECK_INIT(MODE_LIVE);

    if (parseCmd((char *)args, "load", 1, &param))
    {
        lwlogLoadDatabase(param);
    }
    else if (parseCmd((char *)args, "load", 0, &param))
    {
        lwlogLoadDatabase(NULL);
    }
    else if (parseCmd((char *)args, "unload", 0, &param))
    {
        lwlogFreeAll();
    }
    else if (parseCmd((char *)args, "help", 0, &param))
    {
        dprintf("%s", LWLOG_HELP);
    }
    else if (parseCmd((char *)args, "list", 1, &param))
    {
        LwU32 n;
        // If a number is specified, print only that buffer.
        if (sscanf(param, "%d", &n) == 1)
            lwlogPrintBufferInfo(n, NULL);
        else if (!strcmp(param, "print"))
            lwlogPrintAllPrintBuffersInfo();
        else
            lwlogPrintAllBuffersInfo();
    }
    else if (parseCmd((char *)args, "list", 0, &param))
    {
        lwlogPrintAllBuffersInfo();
    }
    else if ((param = strstr((char *)args, "-filter")))
    {
        char *s = param + strlen("-filter");

        if (strstr(s, "help"))
            dprintf(LWLOG_FILTER_HELP);
        else if (strstr(s, "list"))
            lwlogListFilters();
        else
            lwlogApplyFilter(s);
    }
    else
    {
        LwU32 bufferNum = ~0;
        LwU32 numToDecode = 0;
        char filename[FILENAME_MAX] = {0};
        LwBool bHtml;
        LwBool bDecode = LW_FALSE;

        // Parameters used by multiple commands - Specifying buffer and outfile.
        if (parseCmd((char *)args, "b", 1, &param))
        {
            sscanf(param, "%d", &bufferNum);
        }
        if (parseCmd((char *)args, "f", 1, &param))
        {
            strncpy(filename, param, FILENAME_MAX);
            file = fopen(filename, "wb");
            if (file == NULL)
            {
                dbgprintf("lw: Unable to create file %s.\n", filename);
                goto LWLOG_END;
            }
        }
        bHtml = parseCmd((char *)args, "html", 0, &param);

        if (parseCmd((char *)args, "dump", 1, &param))
        {
            if (file == NULL)
            {
                dbgprintf("lw: Please specify the output file with -f\n");
                goto LWLOG_END;
            }
            lwlogInit();
            if (!strcmp(param, "all") || (bufferNum == ~0))
            {
                lwlogDumpAll(file);
            }
            else
            {
                lwlogDumpHeader(bufferNum, file);
                lwlogDumpBuffer(bufferNum, file);
            }

            dbgprintf("lw: LwLog dump stored to %s\n", filename);
            goto LWLOG_END;
        }
        else if (parseCmd((char *)args, "decode", 1, &param) ||
                 parseCmd((char *)args, "d",      1, &param))
        {
            sscanf(param, "%d", &numToDecode);
            bDecode = LW_TRUE;
        }
        else if (parseCmd((char *)args, "decode", 0, &param) ||
                 parseCmd((char *)args, "d",      0, &param))
        {
            bDecode = LW_TRUE;
            numToDecode = 0;
        }

        if (bDecode)
        {
            char *decoded;

            if (bufferNum == ~0)
            {
                dbgprintf("lw: Please specify a buffer number using -b.\n");
                goto LWLOG_END;
            }

            decoded = (char *)malloc(LWLOG_DECODE_BUFFER_SIZE);
            lwlogDecodePrintBuffer(bufferNum, decoded, bHtml, numToDecode);

            if (file == NULL)
            {
                size_t i, buflen;
                char *head;

                // It seems dprintf() (on Windows) cannot print a large string, so here
                // we separate the buffer into small pieces and print them one by one.
                buflen = strlen(decoded);
                head = decoded;
                for (i = 0; i < buflen; i++)
                {
                    if (decoded[i] == '\n')
                    {
                        decoded[i] = 0;
                        dbgprintf("%s\n", head);
                        head = &decoded[i+1];
                    }
                }
            }
            else
            {
                fprintf(file, "%s", decoded);
                dbgprintf("lw: Logs decoded to %s\n", filename);
            }
            free(decoded);
        }
        else
        {
            dprintf("%s\n", LWLOG_HELP);
            lwlogPrintAllBuffersInfo();
        }
    }

LWLOG_END:
    if (file)
    {
        fclose(file);
    }
}
#endif // LWLOG_CMD_ENABLED



//-----------------------------------------------------
// lwlink
// - display lwlink status
//-----------------------------------------------------
DECLARE_API( lwlink )
{
    char *params;

    params = (char *)args;
    CHECK_INIT(MODE_LIVE);

    if (parseCmd(params, "status", 0, &params))
    {
        // Dump LWLink status
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintStatus(FALSE);
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(params, "state", 0, &params))
    {
        // Dump LWLink status
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintStatus(TRUE);
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(params, "progCntrs", 1, &params))
    {
        // Program the LWLTL counters for the given link
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkProgramTlCounters((LwU32) GetSafeExpression(params));
        MGPU_LOOP_END;
        return;
    }
    else if (parseCmd(params, "progCntrs", 0, &params))
    {
        // Program the LWLTL counters for all the links
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkProgramTlCounters(-1);
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(params, "resetCntrs", 1, &params))
    {
        // Reset the LWLTL counters for the given link
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkResetTlCounters((LwU32) GetSafeExpression(params));
        MGPU_LOOP_END;
        return;
    }
    else if (parseCmd(params, "resetCntrs", 0, &params))
    {
        // Reset the LWLTL counters for all the links
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkResetTlCounters(-1);
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(params, "readCntrs", 1, &params))
    {
        // Read the LWLTL counters for the given link
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkReadTlCounters((LwU32) GetSafeExpression(params));
        MGPU_LOOP_END;
        return;
    }
    else if (parseCmd(params, "readCntrs", 0, &params))
    {
        // Read the LWLTL counters for all the links
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkReadTlCounters(-1);
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(params, "v", 0, &params) || parseCmd(params, "p", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintVerbose(parseCmd(params, "p", 0, &params));
        MGPU_LOOP_END;
        return;
    }

    if (parseCmd(params, "dumpuphys", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkDumpUPhy();
        MGPU_LOOP_END;
        return;
    }
    if(parseCmd(params, "dumpAlt", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkDumpAltTraining();
        MGPU_LOOP_END;
    }
    if(parseCmd(params,"dumpAlt",1,&params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkDumpAltTrainingLink((LwU32) GetSafeExpression(params));
        MGPU_LOOP_END;
    }
    // Anything else gets help
    pLwlink[indexGpu].lwlinkPrintHelp();
    return;
}

//-----------------------------------------------------
// ibmnpu
// - NPU debug utilities
//-----------------------------------------------------
DECLARE_API( ibmnpu )
{
    char *params;
    BOOL verbose = FALSE;

    params = (char *)args;
    CHECK_INIT(MODE_LIVE);

    // Parse options
    if (parseCmd(params, "v", 0, &params))
    {
        verbose = TRUE;
    }

    // Parse commands
    if (parseCmd(params, "links", 0, &params))
    {
        ibmnpuCommandLINKS();
    }
    else if (parseCmd(params, "devices", 0, &params))
    {
        ibmnpuCommandDEVICES();
    }
    else if (parseCmd(params, "read", 4, &params))
    {
        LwU32 npu;
        LwU32 brick;
        LwU32 type;
        LwU64 offset;
        dprintf("%s\n", params);
        sscanf(params, "%u %u %u %li", &npu, &brick, &type, (unsigned long *)&offset);
        dprintf("%lx\n", (unsigned long)offset);
        ibmnpuCommandREAD(npu, brick, type, offset);
    }
    else if (parseCmd(params, "write", 5, &params))
    {
        LwU32 npu;
        LwU32 brick;
        LwU32 type;
        LwU64 offset;
        LwU64 data;
        sscanf(params, "%u %u %u %li %li",
                &npu,
                &brick,
                &type,
                (unsigned long *)&offset,
                (unsigned long *)&data);
        ibmnpuCommandWRITE(npu, brick, type, offset, data);
    }
    else if (parseCmd(params, "ctrl", 3, &params))
    {
        LwS32 npu;
        LwS32 brick;
        LwU32 proc;
        sscanf(params, "%d %d %d", &npu, &brick, &proc);
        ibmnpuCommandCTRL(npu, brick, proc);
    }
    else if (parseCmd(params, "ctrl", 1, &params))
    {
        ibmnpuCommandCTRL(-1, -1, (LwU32) GetSafeExpression(params));
    }
    else if (parseCmd(params, "dumpdlpl", 2, &params))
    {
        LwS32 npu;
        LwS32 brick;
        sscanf(params, "%d %d", &npu, &brick);

        ibmnpuCommandDUMPDLPL(npu, brick);
    }
    else if (parseCmd(params, "dumpdlpl", 0, &params))
    {
        ibmnpuCommandDUMPDLPL(-1, -1);
    }
    else if (parseCmd(params, "dumpntl", 2, &params))
    {
        LwS32 npu;
        LwS32 brick;
        sscanf(params, "%d %d", &npu, &brick);

        ibmnpuCommandDUMPNTL(npu, brick);
    }
    else if (parseCmd(params, "dumpntl", 0, &params))
    {
        ibmnpuCommandDUMPNTL(-1, -1);
    }
    else if (parseCmd(params, "dumpuphys", 2, &params))
    {
        LwS32 npu;
        LwS32 brick;
        sscanf(params, "%d %d", &npu, &brick);

        ibmnpuCommandDUMPUPHYS(npu, brick);
    }
    else if (parseCmd(params, "dumpuphys", 0, &params))
    {
        ibmnpuCommandDUMPUPHYS(-1, -1);
    }
    else
    {
        ibmnpuPrintHelp();
    }
}

//-----------------------------------------------------
// hshub
//-----------------------------------------------------
DECLARE_API( hshub )
{
    char *params;

    params = (char *)args;
    CHECK_INIT(MODE_LIVE);

    if (parseCmd(params, "config", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintHshubConfig();
        MGPU_LOOP_END;
        return;
    }
    if (parseCmd(params, "idleStatus", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintHshubIdleStatus();
        MGPU_LOOP_END;
        return;
    }
    if (parseCmd(params, "logErrors", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkLogHshubErrors();
        MGPU_LOOP_END;
        return;
    }
    if (parseCmd(params, "reqTimeoutInfo", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintHshubReqTimeoutInfo();
        MGPU_LOOP_END;
        return;
    }
    if (parseCmd(params, "enableLogging", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkEnableHshubLogging();
        MGPU_LOOP_END;
        return;
    }
    if (parseCmd(params, "readTimeoutInfo", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintHshubReadTimeoutInfo();
        MGPU_LOOP_END;
        return;
    }
    if (parseCmd(params, "connCfg", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintHshubConnectionCfg();
        MGPU_LOOP_END;
        return;
    }
    if (parseCmd(params, "muxCfg", 0, &params))
    {
        MGPU_LOOP_START;
        pLwlink[indexGpu].lwlinkPrintHshubMuxConfig();
        MGPU_LOOP_END;
        return;
    }
}

//-----------------------------------------------------
// vpr
// - vpr commands
//-----------------------------------------------------
DECLARE_API( vpr )
{
    char *params;
    BOOL bHandled = FALSE;
    BOOL bHelp, bSupported, bLwrrangeinmmu, bMaxrangeinbsi, bLwrrangeinbsi, bPrintmemlockstatus, bMemlockrange,
         bPrintBsiType1LockStatus, bHwfuseversions, bUcodeversions;

    params  = (char *)args;
    CHECK_INIT(MODE_LIVE);

    bHelp                      = parseCmd(params, "help",                                        0, NULL);
    bSupported                 = parseCmd(params, "supported",                                   0, NULL);
    bLwrrangeinmmu             = parseCmd(params, "lwrrangeinmmu",                               0, NULL);
    bMaxrangeinbsi             = parseCmd(params, "maxrangeinbsi",                               0, NULL);
    bLwrrangeinbsi             = parseCmd(params, "lwrrangeinbsi",                               0, NULL);
    bPrintmemlockstatus        = parseCmd(params, "getmemlockstatus",                            0, NULL);
    bMemlockrange              = parseCmd(params, "memlockrange",                                0, NULL);
    bPrintBsiType1LockStatus   = parseCmd(params, "getHdcpType1LockStatusInBSISelwreScratch",    0, NULL);
    bHwfuseversions            = parseCmd(params, "hwfuseversions",                              0, NULL);
    bUcodeversions             = parseCmd(params, "ucodeversions",                               0, NULL);

    if (bHelp)
    {
        vprDisplayHelp();
        return;
    }

    MGPU_LOOP_START;
    {
        if(bSupported)
        {
            vprIsSupported(indexGpu);
            bHandled = TRUE;
        }

        if(bLwrrangeinmmu)
        {
            vprMmuLwrrentRangeInfo(indexGpu, LW_TRUE);
            bHandled = TRUE;
        }
        if(bMaxrangeinbsi)
        {
            vprBsiMaxRangeInfo(indexGpu, LW_TRUE);
            bHandled = TRUE;
        }
        if(bLwrrangeinbsi)
        {
            vprBsiLwrrentRangeInfo(indexGpu, LW_TRUE);
            bHandled = TRUE;
        }
        if(bPrintmemlockstatus)
        {
            vprPrintMemLockStatus(indexGpu, LW_TRUE);
            bHandled = TRUE;
        }
        if(bMemlockrange)
        {
            vprMemLockRangeInfo(indexGpu, LW_TRUE);
            bHandled = TRUE;
        }
        if(bPrintBsiType1LockStatus)
        {
            vprPrintBsiType1LockStatus(indexGpu, LW_TRUE);
            bHandled = TRUE;
        }
        if(bHwfuseversions)
        {
            vprGetHwFuseVersions(indexGpu);
            bHandled = TRUE;
        }
        if(bUcodeversions)
        {
            vprGetUcodeVersions(indexGpu);
            bHandled = TRUE;
        }
    }
    MGPU_LOOP_END;

    if (!bHandled)
    {
        vprGetAllInfo(indexGpu);
    }
}

//-------------------------------------------------------------------------
// smcengineinfo [-grIdx] <grIdx>
// - gives the smc info of the specified grIdx
// - eg usage: lw smcengineinfo "-grIdx 0" - to print partition
//                                           info associated with grIdx 0
//-------------------------------------------------------------------------

DECLARE_API( smcengineinfo )
{
    LwU32 grIdx = 0;
    char *param;
    BOOL isGrIndex = FALSE;

    CHECK_INIT(MODE_LIVE);

    isGrIndex = parseCmd(args, "grIdx", 1, &param);

    if(pGr[indexGpu].grGetSmcState())
    {
        if(!isGrIndex)
        {
            dprintf("lw: Missing GrIndex when SMC is enabled. Use -grIdx <grIdx>\n");
            return;
        }

        grIdx = *param -'0';
        if(grIdx > MAX_GR_IDX)
        {
            dprintf("lw: Error: grIdx provided is out of range\n");
            return;
        }
        // Set the window by passing grIdx
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_TRUE);
        pGr[indexGpu].grGetSmcEngineInfo();
        pGr[indexGpu].grConfigBar0Window(grIdx, LW_FALSE);
    }
    else
    {
        dprintf("lw: Error: smcengineinfo cannot be used when SMC is disabled\n");
        return;
    }

}

//--------------------------------------------------------------------
// smcpartitioninfo [-swizzid] <swizzId>
// - gives the partition info of the specified swizzId
// - if no swizzId is given, it prints info about all partitions
// - eg usage: lw smcpartitioninfo "" - to print all partitions' info
//           : lw smcpartitioninfo "-swizzid 1" - to print info about partition
//                                                with swizzId 1
//--------------------------------------------------------------------

DECLARE_API( smcpartitioninfo )
{
    LwU32 swizzId = ILWALID_SWIZZID;
    char *param;
    BOOL isSwizzId = FALSE;

    CHECK_INIT(MODE_LIVE);

    if(pGr[indexGpu].grGetSmcState())
    {
        isSwizzId = parseCmd(args, "swizzid", 1, &param);

        if(isSwizzId)
        {
            swizzId = *param -'0';
            if(swizzId > MAX_SWIZZID)
            {
                dprintf("lw: Error: swizzId provided is out of range\n");
                return;
            }
        }
        else
        {
            dprintf("lw: Printing information about all partitions since no swizzId is mentioned\n");
        }
        pGr[indexGpu].grGetSmcPartitionInfo(swizzId);
    }
    else
    {
        dprintf("lw: Error: smcpartitioninfo cannot be used when SMC is disabled\n");
    }
}

//----------------------------------------------------------------------------
// intr [-enum]
// - print out info about the LW_CTRL tree
// - devinit should be finished before usings this
//   + '-enum' : enumerates all interrupts
//----------------------------------------------------------------------------

DECLARE_API( intr )
{
    char   *param;

    CHECK_INIT(MODE_LIVE);

    if (!IsTU102orLater())
    {
        dprintf("lw: Error: -intr is not supported on pre-Turing\n");
        dprintf("lw: Error: See -diag for info on interrupts pre-Turing\n");
        return;
    }

    if (!IsGA100orLater())
    {
        dprintf("lw: Warning: See -diag for info on interrupts that may report to LW_PMC_INTR in Turing\n");
    }

    if (!args || *args == '\0')
    {
        intrPrintHelp();
        return;
    }

    // enumerate interrupts
    if (parseCmd(args, "enum", 0, NULL))
    {
        intrEnum();
    }
    else if (parseCmd(args, "status", 1, &param))
    {
        if (strncmp(param, "g", 1) == 0)
        {
            intrEnumPending(INTR_ILWALID_GFID, GSP);
            pIntr[indexGpu].intrDumpRawRegs(INTR_ILWALID_GFID, GSP);
        }
        else
        {
            LwU32 gfid = 0;
            if (!parseCmd(args, "gfid", 1, &param) || !sscanf(param, "%d", &gfid))
            {
                gfid = 0;
            }

            intrEnumPending(gfid, CPU);
            pIntr[indexGpu].intrDumpRawRegs(gfid, CPU);
        }
    }
    else if (parseCmd(args, "status", 0, NULL))
    {
        intrEnumPending(INTR_DEFAULT_GFID, CPU);
        pIntr[indexGpu].intrDumpRawRegs(INTR_DEFAULT_GFID, CPU);
    }
    else if (parseCmd(args, "set", 1, &param))
    {
        LwU32 vect, gfid;

        if (!sscanf(param, "%d", &vect))
        {
            dprintf("lw: Error: invalid vector number\n");
            return;
        }

        if (!parseCmd(args, "gfid", 1, &param))
        {
            gfid = INTR_DEFAULT_GFID;
        }
        else if (sscanf(param, "%d", &gfid) <= 0)
        {
            gfid = INTR_ILWALID_GFID;
        }

        if (!IsGH100orLater())
        {
            // All interrupts are pulse-based in Hopper+
            dprintf("lw: Remember, you cannot set level-based interrupts\n");
        }
        pIntr[indexGpu].intrSetInterrupt(vect, gfid);
    }
    else if (parseCmd(args, "clear", 1, &param))
    {
        LwU32 vect, gfid;
        if (!sscanf(param, "%d", &vect))
        {
            dprintf("invalid vector number\n");
            return;
        }

        if (!parseCmd(args, "gfid", 1, &param))
        {
            gfid = INTR_DEFAULT_GFID;
        }
        else if (sscanf(param, "%d", &gfid) <= 0)
        {
            gfid = INTR_ILWALID_GFID;
        }

        dprintf("lw: Remember, you cannot clear level-based interrupts\n");
        pIntr[indexGpu].intrClearInterrupt(vect, gfid);
    }
    else if (parseCmd(args, "enable", 1, &param))
    {
        LwU32 vect, gfid;
        LwBool bGsp = CPU;

        if (!sscanf(param, "%d", &vect))
        {
            dprintf("lw: Error: invalid vector number\n");
            return;
        }

        if (!parseCmd(args, "gfid", 1, &param))
        {
            gfid = INTR_DEFAULT_GFID;
        }
        else if (sscanf(param, "%d", &gfid) <= 0)
        {
            if (param[0] == 'g')
            {
                bGsp = GSP;
            }
            gfid = INTR_ILWALID_GFID;
        }

        pIntr[indexGpu].intrEnableInterrupt(vect, gfid, bGsp);
    }
    else if (parseCmd(args, "disable", 1, &param))
    {
        LwU32 vect, gfid;
        LwBool bGsp = CPU;

        if (!sscanf(param, "%d", &vect))
        {
            dprintf("lw: Error: invalid vector number\n");
            return;
        }

        if (!parseCmd(args, "gfid", 1, &param))
        {
            gfid = INTR_DEFAULT_GFID;
        }
        else if (sscanf(param, "%d", &gfid) <= 0)
        {
            if (param[0] == 'g')
            {
                bGsp = GSP;
            }
            gfid = INTR_ILWALID_GFID;
        }

        pIntr[indexGpu].intrDisableInterrupt(vect, gfid, bGsp);
    }
    else
    {
        intrPrintHelp();
    }
    return;
}

//-----------------------------------------------------
// clkread <clkname|list>
// - traverses schematic diagram to from clkname back, or
// - if clkname is the string "list" then all clock names
// - will be printed which are present in the schematic.
//-----------------------------------------------------
DECLARE_API ( clkread )
{ 
    LwU32 i;                     
    LwU32 freqKHz;               
    ClkFreqSrc *pClock;         
    LwBool listNames;            
    ClkFreqSrc **clkFreqSrcArray;
    char requestName[256]  = {0};

    listNames = LW_FALSE;
    clkFreqSrcArray = pClk[indexGpu].clkGetFreqSrcArray();

    // 
    // Check if clkFreqSrcArray is NULL, then clkread is
    // not supported on this chip
    //
    if (clkFreqSrcArray == NULL)
    {
        dprintf("clkread is not supported on this chip\n");
        return;
    }

    args = getToken(args, requestName, NULL);
    if (requestName[0] == '\0')
    {
        dprintf("An empty name is not supported\n");
        return;
    }
    else if (strncmp("list", requestName, 4) == 0)
    {
        listNames = LW_TRUE;
    }

    i = 0;
    pClock = clkFreqSrcArray[i];
    while (pClock != NULL)
    {
        if (listNames)
        {
            dprintf("lw: %s\n", CLK_NAME(pClock));
        }
        else if (strncmp(pClock->name, requestName, 256) == 0)
        {
            break;
        }
        pClock = clkFreqSrcArray[++i];     
    }
    if (!pClock)
    {
        if (!listNames)
        {
            dprintf("No matching name \"%s\" found in Schematic Diagram.\n", requestName);
        }
        return;
    }
    clkReadAndPrint_FreqSrc_VIP(pClock, &freqKHz);
    dprintf("lw: %s: Freq=%uKHz\n", CLK_NAME(pClock), freqKHz);
}
//-----------------------------------------------------
// version
// - Display LwWatch extension version
//-----------------------------------------------------
DECLARE_API( version )
{
    // Display debugger extension version information
    dprintf("LwWatch version %d.%d\n", LWWATCH_MAJOR_VERSION, LWWATCH_MINOR_VERSION);
}

//-----------------------------------------------------
// l2ila
//-----------------------------------------------------
DECLARE_API(l2ila)
{
    L2ILAConfig config;
    L2ILAArguments L2ILA_args;
    char defaultLogFile[20] = "l2ila.json";
    char commandStr[20];
    char *param = NULL;

    memset(&config, 0, sizeof(L2ILAConfig));
    memset(&L2ILA_args, 0, sizeof(L2ILAArguments));

    dprintf("L2ila plugin is going to be deprecated soon! Please use dfdasm tool instead!\n");

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Usage: l2ila [-v (verbose)] [-k <logfile>] "
                "[-o <outfile>] [command] [inputscriptfile].\n");
        return;
    }

    // parse optional args
    if (parseCmd(args, "v", 0, NULL))
    {
        L2ILA_args.verbose = 1;
        dprintf("Got verbose.\n");
    }

    if (parseCmd(args, "k", 1, &param))
    {
        L2ILA_args.keep = 1;
        strcpy(L2ILA_args.logFname, param);
        dprintf("Got keep with fname %s.\n", L2ILA_args.logFname);
    }

    if (!parseCmd(args, "o", 1, &param))
    {
        strcpy(L2ILA_args.outFname, defaultLogFile);
    } 
    else
    {
        strcpy(L2ILA_args.outFname, param);
    } 
    dprintf("Got out with fname %s.\n", L2ILA_args.outFname);

    // parse command arg
    args = getToken(args, commandStr, NULL);

    if (strcmp(commandStr, "config") == 0) 
    {
        L2ILA_args.command = COMMAND_CONFIG;
        dprintf("Got COMMAND_CONFIG.\n");
    }
    else if (strcmp(commandStr, "arm") == 0) 
    {
        L2ILA_args.command = COMMAND_ARM;
        dprintf("Got COMMAND_ARM.\n");
    }
    else if (strcmp(commandStr, "disarm") == 0) 
    {
        L2ILA_args.command = COMMAND_DISARM;
        dprintf("Got COMMAND_DISARM.\n");
    }
    else if (strcmp(commandStr, "status") == 0) 
    {
        L2ILA_args.command = COMMAND_STATUS;
        dprintf("Got COMMAND_STATUS.\n");
    }
    else if (strcmp(commandStr, "capture") == 0)
    {
        L2ILA_args.command = COMMAND_CAPTURE;
        dprintf("Got COMMAND_CAPTURE.\n");
    }
    else
    {
        dprintf("ERROR!! Unsupported command!! \n"
                "Valid commands: config, arm, disarm, status, capture\n"
                "Usage: l2ila [-v (verbose)] [-k <logfile>] "
                "[-o <outfile>] [command] [inputscriptfile].\n");
        return;
    }

    // parse input script file
    args = getToken(args, L2ILA_args.inFname, NULL);

    if (!L2ILA_args.inFname)
    {
        dprintf("ERROR!! No input script file found!! \n"
                "Usage: l2ila [-v (verbose)] [-k <logfile>] "
                "[-o <outfile>] [command] [inputscriptfile].\n");
        return;
    }

    // check files openable
    L2ILA_args.inFile = fopen(L2ILA_args.inFname, "r");
    if (!L2ILA_args.inFile) 
    {
        dprintf("ERROR!! Cannot open input script file!! \n"
                "Usage: l2ila [-v (verbose)] [-k <logfile>] "
                "[-o <outfile>] [command] [inputscriptfile].\n");        
        L2ILACleanup(&L2ILA_args);
        return;
    }
    if (L2ILA_args.keep)
    {
        L2ILA_args.logFile = fopen(L2ILA_args.logFname, "a");
        if (!L2ILA_args.logFile)
        {
            dprintf("ERROR!! Cannot open log file for keep!! \n"
                    "Usage: l2ila [-v (verbose)] [-k <logfile>] "
                    "[-o <outfile>] [command] [inputscriptfile].\n");           
            L2ILACleanup(&L2ILA_args);
            return;
        }
        fprintf(L2ILA_args.logFile, "====================================================\n");
        fprintf(L2ILA_args.logFile, "Script %s called\n", L2ILA_args.inFname);
        fprintf(L2ILA_args.logFile, "====================================================\n");
    }
    if (L2ILA_args.command == COMMAND_CAPTURE)
    {
        L2ILA_args.outFile = fopen(L2ILA_args.outFname, "w");
        if (!L2ILA_args.inFile)
        {
            dprintf("ERROR!! Cannot open output file for capture!! \n"
                    "Usage: l2ila [-v (verbose)] [-k <logfile>] "
                    "[-o <outfile>] [command] [inputscriptfile].\n");
            L2ILACleanup(&L2ILA_args);
            return;
        }
    }

    // parse input script file
    if (!ParseConfigFromFile(&config, &L2ILA_args))
    {
        dprintf("ERROR!! Cannot parse input script file!! \n"
                "Usage: l2ila [-v (verbose)] [-k <logfile>] "
                "[-o <outfile>] [command] [inputscriptfile].\n");
        L2ILACleanup(&L2ILA_args);
        return;
    }

    // call execute function
    if (L2ILA_args.command == COMMAND_CONFIG)
    {
        ExelwteConfig(&config, &L2ILA_args);
    }
    if (L2ILA_args.command == COMMAND_ARM)
    {
        ExelwteArm(&config, &L2ILA_args);
    }
    if (L2ILA_args.command == COMMAND_DISARM)
    {
        ExelwteDisarm(&config, &L2ILA_args);
    }
    if (L2ILA_args.command == COMMAND_STATUS)
    {
        ExelwteStatus(&config, &L2ILA_args);
    }
    if (L2ILA_args.command == COMMAND_CAPTURE)
    {
        ExelwteCapture(&config, &L2ILA_args);
    }

    // cleanup
    L2ILACleanup(&L2ILA_args);
}


//-----------------------------------------------------
// DFD Assembly
//-----------------------------------------------------
DECLARE_API(dfdasm)
{
    char logFname[256] = "dfdasm.log";
    char command[100];
    char asmFname[256];
    char *param = NULL;
    int verbose = 0;
    int verboseLog = 0;
    int test = 0;

    dprintf("DFD Assembly Runtime\n");

    if ((args == NULL) || (args[0] == '\0'))
    {
        dprintf("lw: Usage: dfdasm [-h] [-v] [-t] [-o <logfile>] <dfdasm file> <command>\n");
        dprintf("    For more info use -h option to see the helper message");
        return;
    }

    if (parseCmd(args, "h", 0, NULL))
    {
        dprintf(" dfdasm [-h] [-v] [vl] [-t] [-o <logfile>] <dfdasm> <command>\n");
        dprintf("                           - Plugin to execute DFD Assembly code\n");
        dprintf("                             dfdasm is an abstraction layer between dfd tools and hardware environment\n");
        dprintf("                             More info: https://confluence.lwpu.com/display/GPUPSIDBG/Lwwatch+DFD+Assembly\n");
        dprintf("                             + -h: Print this help message and exit\n");
        dprintf("                             + -v: Verbose\n");
        dprintf("                             + -vl: Verbose log. This will print all log method and command calls to the console\n");
        dprintf("                             + -t: Test mode. Under this mode all PRI traffic will be handled internally as variables and no actual PRI requests will be sent to HW\n");
        dprintf("                             + -o <logfile>: Store log in <outfile>. Default to dfdasm.log\n");
        dprintf("                             + dfdasm: DFD asm file to execute from\n");
        dprintf("                             + command: Command in dfd asm file to execute\n");
        dprintf("\n");
        return;
    }

    // parse optional args
    if (parseCmd(args, "v", 0, NULL))
    {
        dprintf("Got verbose.\n");
        verbose = 1;
    }

    // parse optional args
    if (parseCmd(args, "vl", 0, NULL))
    {
        dprintf("Got verbose log.\n");
        verboseLog = 1;
    }

    // parse optional args
    if (parseCmd(args, "t", 0, NULL))
    {
        dprintf("Got test.\n");
        test = 1;
    }

    if (parseCmd(args, "o", 1, &param))
    {
        strcpy(logFname, param);
    }

    // parse asmfname and command
    args = getToken(args, asmFname, NULL);
    args = getToken(args, command, NULL);
    dprintf("Got log with fname %s.\n", logFname);
    dprintf("Got dfdasm with fname %s.\n", asmFname);
    dprintf("Got command %s.\n", command);

    // call dfdasm
    runDfdAsm(asmFname, logFname, command, verbose, verboseLog, test);
}
