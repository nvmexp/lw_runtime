/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2007 by LWPU Corporation.  All rights reserved.  All
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

typedef long long LONG64;
typedef unsigned long long ULONG64;
typedef long LONG;
typedef void * PVOID;
typedef unsigned long * PULONG;

void *start_command(const char *);
void complete_command(void *);

void main_loop(int (*callback)(char *name, char *args));

#define min(a,b) ((a) < (b) ? (a) : (b))
#define __cdecl
#define dprintf printf

VOID  initLwWatch(VOID);
VOID  WriteIoSpace(ULONG offset, ULONG value, PULONG size);
VOID  ReadIoSpace(ULONG offset, PULONG value, PULONG size);
U032  readPhysicalMem(ULONG64 address, PVOID buffer, ULONG bytes, U032 *read);
U032  writePhysicalMem(ULONG64 address, PVOID buffer, ULONG bytes, U032 *written);
U032  readVirtMem(ULONG64 address, PVOID buf, ULONG bytes, PULONG read);
VOID  ScanLWTopology(U032 pcie_config_base_address);
LwU64 virtToPhys(LwU64 virtAddr, LwU32 pid);
LwU64 physToVirt(LwU64 physAddr, LwU32 flags);

U032 GetExpression(const char *args);
BOOL GetExpressionEx(const char *args, ULONG64 *val, char **endp);

#endif /* __UNIX_H__ */
