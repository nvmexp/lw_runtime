/*
 * Copyright 2003-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// library for LwWatch on VGX guest
// justint@lwpu.com
// vgpu.c
//
//*****************************************************

#include "vgpu.h"
#include "vgpu/dev_vgpu.h"
#include "turing/tu102/dev_master.h"
#include "os.h"
#include "hal.h"
#include "chip.h"

#if defined(USERMODE) && defined(LW_WINDOWS) && !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#include <usermode.h>
#include <wdbgexts.h>
#include <dbgeng.h>
#endif

//
// uses the same algorithm RM uses to figures out it is on a VGPU. It is desirable
// to bypass RM for debugging purposes.
//
LwBool isVirtual(void)
{
    LwU32 boot1;
    LwU32 vgpuConfig;
    
    if (IsTegra())
    {
        return LW_FALSE;
    }

    // check if we're a VF
    boot1 = GPU_REG_RD32_DIRECT(LW_PMC_BOOT_1);
    if (FLD_TEST_DRF(_PMC,_BOOT_1,_VGPU,_VF, boot1)) 
        return LW_TRUE;

    // check if we're running paravirtualization
    else if (FLD_TEST_DRF(_PMC,_BOOT_1,_VGPU,_PV, boot1)) 
    {
        vgpuConfig = GPU_REG_RD32_DIRECT(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG);
    
        //
        // check if we're running VGPU mode of virtualization as opposed to RTLSIM
        // or FMODEL. if we're running VGPU, we need to raise the LwWatch flag to
        // notify VGX to let us to privilaged accesses (e.g. accessing regions outside
        // of guest granted FB)
        //
        if (!(FLD_TEST_DRF(_VGPU, _CONFIG, _SIMULATION, _FMODEL, vgpuConfig)) &&
            !(FLD_TEST_DRF(_VGPU, _CONFIG, _SIMULATION, _RTLSIM, vgpuConfig)))
        {
            return LW_TRUE;
        }
    }
    
    return LW_FALSE;
}

LwBool isVirtualWithSriov(void)
{
    LwU32 boot1;

    if (IsTegra())
    {
        return LW_FALSE;
    }
    // check if we're a VF
    boot1 = GPU_REG_RD32_DIRECT(LW_PMC_BOOT_1);
    if (FLD_TEST_DRF(_PMC, _BOOT_1, _VGPU, _VF, boot1))
        return LW_TRUE;
    return LW_FALSE;
}

// Read a PF register via the emulated register pair from plugin
// Relies on setLwwatchMode being true usually from exts.c, else plugin will return 0
LwU32 pfRegRead(PhysAddr reg)
{
    LwU32 regValue = 0;
    GPU_REG_WR32_DIRECT(DRF_BASE(LW_VGPU_EMU) + LW_VGPU_LWWATCH_REG_ADDR, (LwU32)reg);
    regValue = GPU_REG_RD32_DIRECT(DRF_BASE(LW_VGPU_EMU) + LW_VGPU_LWWATCH_REG_VALUE);

    return regValue;
}

// Write a PF register via the emulated register pair from plugin
// Relies on setLwwatchMode being true
void pfRegWrite(PhysAddr reg, LwU32 value)
{
    GPU_REG_WR32_DIRECT(DRF_BASE(LW_VGPU_EMU) + LW_VGPU_LWWATCH_REG_ADDR, (LwU32)reg);
    GPU_REG_WR32_DIRECT(DRF_BASE(LW_VGPU_EMU) + LW_VGPU_LWWATCH_REG_VALUE, value);

}

// Get GFID from plugin using the new LW_VGPU_LWWATCH_INFO_TYPE access
LwU32 getGfid()
{
    LwU32 regValue = 0;
    GPU_REG_WR32_DIRECT(DRF_BASE(LW_VGPU_EMU) + LW_VGPU_LWWATCH_INFO_TYPE, 
                    (LwU32)LW_VGPU_LWWATCH_INFO_TYPE_GFID);
    regValue = GPU_REG_RD32_DIRECT(DRF_BASE(LW_VGPU_EMU) + 
                    LW_VGPU_LWWATCH_INFO_VALUE);
    return regValue;
}

//
//
// sets or unsets the LwWatch mode flag
//
void setLwwatchMode(LwBool isEnabled)
{
    LwU32 lwwatchConfig;
    LwU32 regBase;

    regBase = isVirtualWithSriov() ? DRF_BASE(LW_VGPU_EMU) : DRF_BASE(LW_VGPU);
    lwwatchConfig = GPU_REG_RD32_DIRECT(regBase + LW_VGPU_CONFIG);
    lwwatchConfig &= ~DRF_SHIFTMASK(LW_VGPU_CONFIG_LWWATCH_MODE);
    lwwatchConfig |= (isEnabled ? 0x1ULL : 0x0ULL) << DRF_SHIFT(LW_VGPU_CONFIG_LWWATCH_MODE);
    
    GPU_REG_WR32_DIRECT(regBase + LW_VGPU_CONFIG, lwwatchConfig);
}

//
// sets the address type bits in the virtual registers denoting what kind of address
// LwWatch is passing down
//
void setLwwatchAddrType(LwU32 addrType)
{
    LwU32 lwwatchConfig;

    LwU32 regBase = isVirtualWithSriov() ? DRF_BASE(LW_VGPU_EMU) : DRF_BASE(LW_VGPU);

    lwwatchConfig = GPU_REG_RD32_DIRECT(regBase + LW_VGPU_CONFIG);
    lwwatchConfig &= ~DRF_SHIFTMASK(LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE);
    
    switch (addrType)
    {
        case GUEST_PHYSICAL:
            lwwatchConfig |= LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_GUEST_PHYSICAL << 
                             DRF_SHIFT(LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE);
            break;
        case GUEST_VIRTUAL:
            lwwatchConfig |= LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_GUEST_VIRTUAL << 
                             DRF_SHIFT(LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE);
            break;
        case HOST_PHYSICAL:
            lwwatchConfig |= LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_HOST_PHYSICAL << 
                             DRF_SHIFT(LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE);
            break;
        case HOST_VIRTUAL:
            lwwatchConfig |= LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_HOST_VIRTUAL <<
                             DRF_SHIFT(LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE);
            break;
        default:
            lwwatchConfig |= LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_ILWALID <<
                             DRF_SHIFT(LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE);
            break;
    }
    
    GPU_REG_WR32_DIRECT(regBase + LW_VGPU_CONFIG, lwwatchConfig);
}

LwBool getLwwatchMode(void)
{
    LwU32 regBase = isVirtualWithSriov() ? DRF_BASE(LW_VGPU_EMU) : DRF_BASE(LW_VGPU);

    LwU32 lwwatchConfig = GPU_REG_RD32_DIRECT(regBase + LW_VGPU_CONFIG);
    return (LW_VGPU_CONFIG_LWWATCH_MODE_TRUE == lwwatchConfig) ? LW_TRUE : LW_FALSE;
}

LwU32 getLwwatchAddrType(void)
{
    LwU32 lwwatchConfig;
    LwU32 regBase = isVirtualWithSriov() ? DRF_BASE(LW_VGPU_EMU) : DRF_BASE(LW_VGPU); 

    lwwatchConfig = GPU_REG_RD32_DIRECT(regBase + LW_VGPU_CONFIG);
    lwwatchConfig = DRF_VAL(_VGPU, _CONFIG_LWWATCH, _ADDR_TYPE, lwwatchConfig);
    
    switch (lwwatchConfig)
    {
        case LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_GUEST_PHYSICAL:
            return GUEST_PHYSICAL;
        case LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_GUEST_VIRTUAL:
            return GUEST_VIRTUAL;
        case LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_HOST_PHYSICAL:
            return HOST_PHYSICAL;
        case LW_VGPU_CONFIG_LWWATCH_ADDR_TYPE_HOST_VIRTUAL:
            return HOST_VIRTUAL;
        default:
            return INVALID;
    }
}
