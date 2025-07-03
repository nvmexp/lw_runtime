/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
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
// ---- TAHOMAHACK END ----

//
// chip routines - chip.c
//
void    GetChipAndRevision(LwU32 *pChip, LwU32 *pRevision);

BOOL    IsTegra(void);
BOOL    IsT124(void);
BOOL    IsT210(void);
BOOL    IsT186(void);
BOOL    IsT194(void);
BOOL    IsT234(void);
BOOL    IsGM107(void);
BOOL    IsGM200(void);
BOOL    IsGM204(void);
BOOL    IsGM206(void);
BOOL    IsGP100(void);
BOOL    IsGP102(void);
BOOL    IsGP104(void);
BOOL    IsGP106(void);
BOOL    IsGP107(void);
BOOL    IsGP108(void);
BOOL    IsGV100(void);
BOOL    IsTU102(void);
BOOL    IsTU104(void);
BOOL    IsTU106(void);
BOOL    IsTU116(void);
BOOL    IsTU117(void);
BOOL    IsGA100(void);
BOOL    IsGA102(void);
BOOL    IsGA103(void);
BOOL    IsGA104(void);
BOOL    IsGA106(void);
BOOL    IsGA107(void);
BOOL    IsAD102(void);
BOOL    IsAD103(void);
BOOL    IsAD104(void);
BOOL    IsAD106(void);
BOOL    IsAD107(void);
BOOL    IsGH100(void);
BOOL    IsGH202(void);
BOOL    IsGB100(void);
BOOL    IsG000(void);
BOOL    IsGM107orLater(void);
BOOL    IsGM200orLater(void);
BOOL    IsGP100orLater(void);
BOOL    IsGP102orLater(void);
BOOL    IsGP104orLater(void);
BOOL    IsGP106orLater(void);
BOOL    IsGP107orLater(void);
BOOL    IsGP108orLater(void);
BOOL    IsGV100orLater(void);
BOOL    IsTU102orLater(void);
BOOL    IsTU104orLater(void);
BOOL    IsTU106orLater(void);
BOOL    IsTU116orLater(void);
BOOL    IsTU117orLater(void);
BOOL    IsGA100orLater(void);
BOOL    IsGA102orLater(void);
BOOL    IsAD102orLater(void);
BOOL    IsGH100orLater(void);
BOOL    IsGB100orLater(void);
BOOL    IsG000orLater(void);
BOOL    IsGM100Arch(void);
BOOL    IsGM200Arch(void);
BOOL    IsGP100Arch(void);
BOOL    IsGV100Arch(void);
BOOL    IsTU100Arch(void);
BOOL    IsGA100Arch(void);
BOOL    IsGH100Arch(void);
BOOL    IsGB100Arch(void);
BOOL    IsG000Arch(void);

char*   GpuArchitecture(void);
char*   GpuImplementation(void);

BOOL    GetManualsDir(char **pChipInfo, char *pChipClassNum, int *pNumOfPaths);
BOOL    GetDispManualsDir(char *dispManualPath);
BOOL    GetClassNum(char *pClassNum);

#endif // _CHIP_H_
