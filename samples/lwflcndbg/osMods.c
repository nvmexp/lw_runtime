/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2012 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004
// os specific routines for lwwatchMods
//
//*****************************************************

//-----------------------------------------------------
// Note regarding register reading/writing routines in 
// this file: 
// + They do not check for / propogate error
//   conditions to their callers. This is a deliberate
//   design decision to keep things clean for callers.
//   But, when a error condition is detected, a message
//   is printed out in RegRd which should alert the user
//   that something went wrong. 
//-----------------------------------------------------

//
// includes
//
#include "regMods.h"
#include "os.h"


//-----------------------------------------------------
// QueryPerformanceFrequencyMods 
// + Retrieve the processor clock frequency
//-----------------------------------------------------
BOOL QueryPerformanceFrequencyMods
( 
    LARGE_INTEGER *freq  
)
{
    dprintf("lw: %s - not implemented!\n", __FUNCTION__);
    return FALSE;
}

//-----------------------------------------------------
// initLwWatchMods
// 
//-----------------------------------------------------
VOID initLwWatchMods()
{
    LARGE_INTEGER liFreq;

    if(!FindLWDevice())
    {
        dprintf("lw: No devices found.\n");
        return;
    }

    //
    // Common OS initialization
    //
    osInit();

    //
    // Override what the common OS init did
    // Mods does not provide access to lwBar0/lwBar1
    // Set it to zero so that lwwatch that uses it does not
    // cause harm by reading/addresses bogus addresses
    //
    lwBar1 = 0;

    //
    // Get the frequency of the processor here.
    //
    if (QueryPerformanceFrequencyMods(&liFreq))
    {
        CPUFreq = liFreq.LowPart;
    }
    else
    {
        dprintf("lw: QueryPerformanceFrequencyMods call failed!\n");
    }

    dprintf("lw: Call !lw.help for a command list.\n");
}

//-----------------------------------------------------
// Register reading routines
// 
//-----------------------------------------------------
U008 
REG_RD08Mods
(
    LwU64 Address
)
{
    //assert(0 == LwU64_HI32(Address));
    #ifdef FMODEL_REG08_BUG
    
        U032 regVal;
        U008 byteOffset, shift;
        
        byteOffset = (U008) ((Address) % 4);
        Address -= byteOffset;
        shift = byteOffset * 8;
        
        regVal = REG_RD32(Address);
        regVal = (regVal >> shift) & 0xFF;

        return (U008)regVal;
    
    #else

        return (U008)RegRd("GpuRegRd08", READ08, LwU64_LO32(Address));    

    #endif
}

U032 
REG_RD32Mods
(
    LwU64 Address
)
{
    assert(0 == LwU64_HI32(Address));
    return (U032)RegRd("GpuRegRd32", READ32, LwU64_LO32(Address));    
}

//-----------------------------------------------------
// Framebuffer reading routines
//
//-----------------------------------------------------

U008 
FB_RD08Mods
(
    U032 Address
)
{
    return (U008)RegRd("GpuFbRd08", READ08, Address);    
}

U016 
FB_RD16Mods
(
    U032 Address
)
{
    return (U016)RegRd("GpuFbRd16", READ16, Address);    
}

U032 
FB_RD32Mods
(
    U032 Address
)
{
    return RegRd("GpuFbRd32", READ32, Address);    
}

//-----------------------------------------------------
// Register writing routines
// 
//-----------------------------------------------------

VOID 
REG_WR08Mods
(
    LwU64 Address, 
    U008 Data
)
{
    BOOL retval = RegWr("GpuRegWr08", (LwU32) Address, Data);
    assert(0 == LwU64_HI32(Address));
    if (!retval) dprintf(__FUNCTION__ ": Register write may be erroneous");
}

VOID 
REG_WR32Mods
(
    LwU64 Address, 
    U032 Data
)
{
    BOOL retval = RegWr("GpuRegWr32", (LwU32) Address, Data);
    assert(0 == LwU64_HI32(Address));
    if (!retval) dprintf(__FUNCTION__ ": Register write may be erroneous");
}

//-----------------------------------------------------
// FB writing routines
// 
//-----------------------------------------------------

VOID 
FB_WR08Mods
(
    U032 Address, 
    U008 Data
)
{
    BOOL retval = RegWr("GpuFbWr08", Address, Data);
    if (!retval) dprintf(__FUNCTION__ ": Register write may be erroneous");
}

VOID 
FB_WR16Mods
(
    U032 Address, 
    U016 Data
)
{
    BOOL retval = RegWr("GpuFbWr16", Address, Data);
    if (!retval) dprintf(__FUNCTION__ ": Register write may be erroneous");
}

VOID 
FB_WR32Mods
(
    U032 Address, 
    U032 Data
)
{
    BOOL retval = RegWr("GpuFbWr32", Address, Data);
    if (!retval) dprintf(__FUNCTION__ ": Register write may be erroneous");
}

U008 RD_PHYS08Mods
(
    LwU64 pa
)
{
    return (U008)RegRd("Platform::PhysRd08", READ08, (U032)pa);
}

U016 RD_PHYS16Mods
(
    LwU64 pa
)
{
    return (U016)RegRd("Platform::PhysRd16", READ16, (U032)pa);
}

U032 RD_PHYS32Mods
(
    LwU64 pa
)
{
    return (U032)RegRd("Platform::PhysRd32", READ32, (U032)pa);
}

/*
Note: The following functions really should be replaced with RD_PHYSXX,
      but, I don't want to change RD_PHYSXX functions until I can 
      verify corrrectness in all instances that they are used, and I
      want to get something checked in to use in the lab.  In the 
      next revision, these functions will be gone.
*/
U008 SYSMEM_RD08Mods
(
    LwU64 pa
)
{
    return (U008)RegRd("Platform::PhysRd08", READ08, (U032)pa);
}

U016 SYSMEM_RD16Mods
(
    LwU64 pa
)
{
    return (U016)RegRd("Platform::PhysRd16", READ16, (U032)pa);
}

U032 SYSMEM_RD32Mods
(
    LwU64 pa
)
{
    return (U032)RegRd("Platform::PhysRd32", READ32, (U032)pa);
}
