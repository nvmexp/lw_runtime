/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2007 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _CHIP_H_
#define _CHIP_H_

#include "os.h"
#define MAX_PATHS   5

// ---- TAHOMAHACK START ----
extern LwU32 isTegraHack;
extern BOOL bIsSocBrdg;

VOID mcGetInfo_T30(void);
VOID mcGetInfo_T114(void);
VOID mcGetInfo_T124(void);
VOID dispGetDcCmdInfo_T20(U032 dc);
VOID dispGetDcComInfo_T20(U032 dc);
VOID dispGetDcDispInfo_T20(U032 dc);
VOID dispGetDcWinAInfo_T20(U032 dc);
VOID dispGetDcWinBInfo_T20(U032 dc);
VOID dispGetDcWinCInfo_T20(U032 dc);
VOID dispGetTvDacInfo_T20(void);
VOID dispGetPinMuxInfo_T20(void);
VOID dispGetClocks_T20(void);
VOID dispGetAllInfo_T20(void);
VOID dispGetHdmiRegs_T20(void);
VOID intrGetCtlrInfo_T20(U032 ictlr);
VOID gpioGetInfo_T20(U032 gpioctrl);
VOID gpioGetAllInfo_T20(void);
VOID dispGetDsiInfo_T30(U032 index);
VOID dispGetHdmiRegs_T30(void);
LwBool socbrdgIsBridgeDevid(LwU32 deviceId);
VOID mpeGetInfo_T20(void);


// ---- TAHOMAHACK END ----


//
// chip routines - chip.c
//
VOID    GetChipAndRevision(U032 *pChip, U032 *pRevision);

BOOL    IsTegra(void);
BOOL    IsT30(void);
BOOL    IsT114(void);
BOOL    IsT124(void);
BOOL    IsT148(void);
BOOL    IsT210(void);
BOOL    IsSocBrdg(void);
BOOL    IsRSX(void);
BOOL    IsLW11(void);
BOOL    IsLW15(void);
BOOL    IsLW17(void);
BOOL    IsLW18(void);
BOOL    IsLW31(void);
BOOL    IsLW36(void);
BOOL    IsLW40(void);
BOOL    IsLW41(void);
BOOL    IsLW43(void);
BOOL    IsLW44(void);
BOOL    IsLW46(void);
BOOL    IsLW47(void);
BOOL    IsLW49(void);
BOOL    IsLW4C(void);
BOOL    IsLW4E(void);
BOOL    IsLW63(void);
BOOL    IsLW67(void);
BOOL    IsG78(void);
BOOL    IsLW50(void);
BOOL    IsLW50orBetter(void);
BOOL    IsG82(void);
BOOL    IsG84(void);
BOOL    IsG86(void);
BOOL    IsG92(void);
BOOL    IsG94(void);
BOOL    IsG96(void);
BOOL    IsG98(void);
BOOL    IsGT200(void);
BOOL    IsGT206(void);
BOOL    IsiGT206(void);
BOOL    IsMCP77(void);
BOOL    IsMCP78(void);
BOOL    IsMCP79(void);
BOOL    IsGT215(void);
BOOL    IsGT216(void);
BOOL    IsGT218(void);
BOOL    IsiGT21A(void);
BOOL    IsMCP89(void);
BOOL    IsGT2XX(void);
BOOL    IsGT21X(void);
BOOL    IsGF100(void);
BOOL    IsGF100B(void);
BOOL    IsGF104(void);
BOOL    IsGF106(void);
BOOL    IsGF108(void);
BOOL    IsGF110D(void);
BOOL    IsGF110F(void);
BOOL    IsGF110F2(void);
BOOL    IsGF110F3(void);
BOOL    IsGF117(void);
BOOL    IsGF119(void);
BOOL    IsGK104(void);
BOOL    IsGK106(void);
BOOL    IsGK107(void);
BOOL    IsGK107B(void);
BOOL    IsGK110(void);
BOOL    IsGK20A(void);
BOOL    IsGK208(void);
BOOL    IsGM107(void);
BOOL    IsGM200(void);
BOOL    IsGK104orLater(void);
BOOL    IsGK110orLater(void);
BOOL    IsGK208orLater(void);
BOOL    IsGM107orLater(void);
BOOL    IsGM200orLater(void);


BOOL    IsLW10Arch(void);
BOOL    IsLW20Arch(void);
BOOL    IsLW30Arch(void);
BOOL    IsLW40Arch(void);
BOOL    IsLW50Arch(void);
BOOL    IsG80Arch(void);
BOOL    IsG90Arch(void);
BOOL    IsGT200Arch(void);
BOOL    IsGF100Arch(void);
BOOL    IsGF110Arch(void);
BOOL    IsGK100Arch(void);
BOOL    IsGK110Arch(void);
BOOL    IsGK200Arch(void);
BOOL    IsGM100Arch(void);
BOOL    IsGM200Arch(void);

BOOL    IsLW17orBetter(void);
BOOL    IsLW18orBetter(void);
BOOL    IsLW25orBetter(void);
BOOL    IsLW30orBetter(void);
BOOL    IsLW40orBetter(void);
BOOL    IsLW41orBetter(void);

U032    GetNumCrtcs(void);
U032    GetNumLinks(void);
U032    EnableHead(U032 Head);
U032    GetMaxCrtcReg(void);
U032    GetMaxTMDSReg(void);

BOOL    GetManualsDir(char **pChipInfo, char *pChipClassNum, int *pNumOfPaths);
BOOL    GetDispManualsDir(char *dispManualPath);
BOOL    GetClassNum(char *pClassNum);

#endif // _CHIP_H_
