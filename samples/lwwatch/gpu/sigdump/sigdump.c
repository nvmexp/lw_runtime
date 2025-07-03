/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "chip.h"
#include "sig.h"

// NOTE: The order should match the enum in sig.h
static const char* instanceNames[NUMINSTANCES]=
{
    "gpc",
    "gpc_tpc",
    "fbp",
    "sys",
    "sys_mxbar_cq_daisy",
    "sys_wxbar_cq_daisy",
    "sys_mxbar_cq_daisy_unrolled",
    "gpc_dtpc",   // added in gf117
    "gpc_ppc",
    "sys_mxbar_cs_daisy",
    "sys_wxbar_cs_daisy",
};

static const char illegalInstance[] = "unsupported_instance";

void RegWrite(LwU32 addr, LwU32 value, LwU32 mask)
{
    LwU32 temp;
    temp = GPU_REG_RD32(addr);
    temp &= ~mask;
    temp |= value;
    GPU_REG_WR32(addr, temp);
}

LwU32 RegBitRead(LwU32 addr, LwU32 bit)
{
    LwU32  temp;
    temp = GPU_REG_RD32(addr);
    return ((temp & (1<<bit)) >> bit) & 1;
}

LwBool RegRead(LwU32 *readVal, LwU32 addr, LwBool printsEnabled)
{
    LwU32 temp;
    LwBool readSuccess;

    temp = GPU_REG_RD32(addr);
    if ((temp & 0xFFFF0000) == 0xBADF0000)
    {
        if (printsEnabled == LW_TRUE)
        {
            dprintf ("WARNING: Pri read @(0x%X) gave 0x%X value!\n", addr, temp);
        }
        readSuccess = LW_FALSE;             // Pri read failed.
    }
    else
    {
        *readVal = temp;
        readSuccess = LW_TRUE;              // Pri read succeeded.
    }
    return readSuccess;
}

LwBool RegBitfieldRead(LwU32 *readVal, LwU32 addr, LwU32 lsb, LwU32 msb, LwBool printsEnabled)
{
    LwBool readSuccess;

    readSuccess = RegRead(readVal, addr, printsEnabled);
    if (readSuccess)
    {
        // Mask generation trick fails for 32b left shift, which is an attempt
        // to return the whole register anyway
        if ((msb - lsb + 1) < 32)
        {
            *readVal = (*readVal >> lsb) & ((1 << (msb - lsb + 1)) - 1);
        }
    }
    return readSuccess;
}

const char* sigGetInstanceName(LwU32 instance)
{
    if (instance >= NUMINSTANCES)
    {
        return illegalInstance;
    }
    return instanceNames[instance];
}

#ifdef SIGDUMP_ENABLE

void OutputSignal(FILE *fp, char *name, LwU32 value)
{
     fprintf(fp, "%s=%x\n", name, value);
}

#endif  //SIGDUMP_ENABLE
