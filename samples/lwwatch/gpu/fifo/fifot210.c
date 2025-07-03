
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "hal.h"
#include "fifo.h"
#include "t21x/t210/hwproject.h"

/*!
 * @return The number of PBDMAs provided by the chip.
 */
LwU32 fifoGetNumPbdma_T210(void)
{
    return LW_HOST_NUM_PBDMA;
}
