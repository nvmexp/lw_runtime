/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "fifo.h"
#include "kepler/gk208/hwproject.h"
#include "kepler/gk208/dev_fifo.h"

/*!
 * @return The number of PBDMAs provided by the chip.
 */
LwU32 fifoGetNumPbdma_GK208(void)
{
    return LW_HOST_NUM_PBDMA;
}

/*!
 * @return The maximum number of channels provided by the chip.
 */
LwU32 fifoGetNumChannels_GK208(LwU32 runlistId)
{
    // Unused pre-Ampere
    (void) runlistId;

    return LW_PCCSR_CHANNEL__SIZE_1;
}

LwU32 fifoGetNumEng_GK208(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}
