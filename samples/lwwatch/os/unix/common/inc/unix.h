/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __UNIX_H__
#define __UNIX_H__

#if !defined(_OS_H_)
#error "don't #include unix.h directly, #include os.h instead!"
#endif

void *start_command(const char *);
void complete_command(void *);

void main_loop(int (*callback)(char *name, char *args));

#define min(a,b) ((a) < (b) ? (a) : (b))
#define __cdecl
#define dprintf printf

void  initLwWatch(void);
LW_STATUS  readPhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *read);
LW_STATUS  writePhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *written);
LW_STATUS  readVirtMem(LwU64 address, void *buf, LwU64 bytes, LwU64 *read);
void  ScanLWTopology(LwU32 pcie_config_base_address);
LwU64 virtToPhys(LwU64 virtAddr, LwU32 pid);
LwU64 physToVirt(LwU64 physAddr, LwU32 flags);
LW_STATUS  osGetLwrrentTick(LwU64 *pTimeInNs);

LwU64 GetExpression(const char *args);
BOOL GetExpressionEx(const char *args, LwU64 *val, char **endp);

#endif /* __UNIX_H__ */
