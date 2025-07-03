/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grt124.c
//
//*****************************************************

//
// includes
//
#include "chip.h"
// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/t12x/t124/
#include "t12x/t124/dev_graphics_nobundle.h"
#include "t12x/t124/dev_fifo.h"

#include "t12x/t124/dev_top.h"
#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes


// LW_PFIFO_ENGINE_STATUS() has a different stride starting on Kepler
void grPrintEngineGraphicsStatus_T124(void)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];

    // A little tricky: print address of this reg as a string in order
    // to use priv_dump, which will print all of the fields
    // automatically.  It won't print the fact that _ENGINE_GRAPHICS
    // is the register being shown (most likely the value will be 0),
    // so print that beforehand to make it clear.  priv_dump will
    // recognize "LW_PFIFO_ENGINE_STATUS(0)" as an address, but not
    // "LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS)".  Using them
    // this way expands and checks the values at compile time.
    dprintf("LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS):\n");
    sprintf( buffer, "0x%08x", LW_PFIFO_ENGINE_STATUS(LW_PTOP_DEVICE_INFO_TYPE_ENUM_GRAPHICS) );
    priv_dump( buffer );
}
