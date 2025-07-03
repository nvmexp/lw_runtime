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
// osMods: Prototypes for code in osMods.c
// 
//*****************************************************

#ifndef _OSMODS_H_
#define _OSMODS_H_

extern U032 usingMods;

VOID initLwWatchMods(void);
BOOL QueryPerformanceFrequencyMods( LARGE_INTEGER *freq  );

U032 REG_RD32Mods (LwU64 Address);
U008 REG_RD08Mods (LwU64 Address);
VOID REG_WR32Mods (LwU64 Address, U032 Data);
VOID REG_WR08Mods (LwU64 Address, U008 Data);

U008 FB_RD08Mods  (U032 Address);
U016 FB_RD16Mods  (U032 Address);
U032 FB_RD32Mods  (U032 Address);

/* Temporary functions--See comment in osMods.c */
U008 SYSMEM_RD08Mods(LwU64 pa);
U016 SYSMEM_RD16Mods(LwU64 pa);
U032 SYSMEM_RD32Mods(LwU64 pa);

VOID FB_WR08Mods  (U032 Address, U008 Data);
VOID FB_WR16Mods  (U032 Address, U016 Data);
VOID FB_WR32Mods  (U032 Address, U032 Data);

U032 readVirtualMemMods (ULONG64 address, PVOID buf, ULONG size, PULONG pSizer,
                         MEM_TYPE memoryType);
U032 readFBMemMods      (ULONG64 address, PVOID buf, ULONG size, PULONG pSizer);

#endif // _OSMODS_H_
