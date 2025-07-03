/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003 by LWPU Corporation.  All rights reserved.  All
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
// dac.h
//
//*****************************************************

#ifndef _DAC_H_
#define _DAC_H_

#include "os.h"

//
// defines
//
#define MAX_CRTCS               0x2
#define MAX_LINKS               0x4
#define REGS_MAX_INDEX          64

#define FP_SKIP_TMDS_LINKS      0x1
#define FP_DECIMAL_TIMINGS      0x2

#define QFPDUMP_PRINT_BYTE      0x1
#define QFPDUMP_DECIMAL_TIMINGS 0x2

#define VGAT_HT                 0x00000000
#define VGAT_HDE                0x00000001
#define VGAT_HBS                0x00000002
#define VGAT_HBE                0x00000003
#define VGAT_HRS                0x00000004
#define VGAT_HRE                0x00000005
#define VGAT_VT                 0x00000006
#define VGAT_VDE                0x00000007
#define VGAT_VBS                0x00000008
#define VGAT_VBE                0x00000009
#define VGAT_VRS                0x0000000A
#define VGAT_VRE                0x0000000B
#define VGAT_ALL                0x0000000F

//
// prototypes
// 
VOID    dumpVGARegs(void);
VOID    VGATimings(U032, U032, U032, U032);
VOID    dumpPalette(U032);
VOID    dumpHWSEQRam(void);
VOID    dumpTvRegs(U032 crtcOffset);
U032    tvoRegRd(U032 reg);
VOID    tvoRegWr(U032 reg, U032 val);
U008    UnlockExtendedCRTCs(U032 crtcOffset);
VOID    RestoreExtendedCRTCs(U008 crLock, U032 crtcOffset);
U032    SetBroadcastBit(U032 state);
VOID    dumpSLIVideoBridgeRegisters(void);
VOID    quickFPRegPerHeadDump(char *initStr, U032 Regs[MAX_CRTCS][REGS_MAX_INDEX], U032, U032, U032);


#endif // _DAC_H_
