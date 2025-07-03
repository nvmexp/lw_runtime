/*
 * Copyright 2003-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
#ifndef VGPU_LWWATCH_CONFIG
#define VGPU_LWWATCH_CONFIG

#include "lwtypes.h"
#include "os.h"  // For typedef of PhysAddr

#define INVALID        0x00000000
#define HOST_PHYSICAL  0x00000001
#define HOST_VIRTUAL   0x00000002
#define GUEST_PHYSICAL 0x00000003
#define GUEST_VIRTUAL  0x00000004

LwBool isVirtual(void);
LwBool isVirtualWithSriov(void);
void setLwwatchMode(LwBool isEnabled);
void setLwwatchAddrType(LwU32 addrType);
LwBool getLwwatchMode(void);
LwU32 getLwwatchAddrType(void);
LwU32 pfRegRead(PhysAddr reg);
void pfRegWrite(PhysAddr reg, LwU32 value);
LwU32 getGfid();

#endif
