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
#include "kepler/gk107/dev_master.h"
#include "os.h"
#include "hal.h"

#if defined(USERMODE) && defined(LW_WINDOWS)
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
    
    // check if we're running some sort of virtualization
    boot1 = GPU_REG_RD32(LW_PMC_BOOT_1);
    if (FLD_TEST_DRF(_PMC,_BOOT_1,_VGPU16,_VIRTUAL, boot1))
    {
        vgpuConfig = GPU_REG_RD32(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG);
    
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

//
// sets or unsets the LwWatch mode flag
//
void setLwwatchMode(LwBool isEnabled)
{
    LwU32 lwwatchConfig;

    lwwatchConfig = GPU_REG_RD32(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG);
    lwwatchConfig &= ~DRF_SHIFTMASK(LW_VGPU_CONFIG_LWWATCH_MODE);
    lwwatchConfig |= (isEnabled ? 0x1ULL : 0x0ULL) << DRF_SHIFT(LW_VGPU_CONFIG_LWWATCH_MODE);
    
    GPU_REG_WR32(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG, lwwatchConfig);
}

//
// sets the address type bits in the virtual registers denoting what kind of address
// LwWatch is passing down
//
void setLwwatchAddrType(LwU32 addrType)
{
    LwU32 lwwatchConfig;
    
    lwwatchConfig = GPU_REG_RD32(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG);
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
    
    GPU_REG_WR32(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG, lwwatchConfig);
}

LwBool getLwwatchMode(void)
{
    LwU32 lwwatchConfig = GPU_REG_RD32(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG);
    return (LW_VGPU_CONFIG_LWWATCH_MODE_TRUE == lwwatchConfig) ? LW_TRUE : LW_FALSE;
}

LwU32 getLwwatchAddrType(void)
{
    LwU32 lwwatchConfig;
    
    lwwatchConfig = GPU_REG_RD32(DRF_BASE(LW_VGPU) + LW_VGPU_CONFIG);
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
