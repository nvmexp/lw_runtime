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
#include "maxwell/gm107/dev_fifo.h"
#include "maxwell/gm107/dev_top.h"


//
// Map the engine names, from //hw/lwgpu/defs/projects.spec, to the values in
// LW_PTOP_INFO_TYPE_ENUM. COPY1 and LWENC1 are unused on some Maxwell chips.
//
// The array index is used to represent the engines in runlist2EngMask[].
//
static EngineNameValue engName2DeviceInfo[] =
{
    {{"GR0",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_GRAPHICS }, 0, ENGINE_TAG_GR},      // bit 0 in runlist2EngMask
    {{"LWDEC",  LW_PTOP_DEVICE_INFO_TYPE_ENUM_LWDEC    }, 0, ENGINE_TAG_UNKNOWN}, // bit 1 in runlist2EngMask
    {{"LWENC0", LW_PTOP_DEVICE_INFO_TYPE_ENUM_LWENC0   }, 0, ENGINE_TAG_UNKNOWN}, // bit 2 in runlist2EngMask
    {{"SEC0",   LW_PTOP_DEVICE_INFO_TYPE_ENUM_SEC      }, 0, ENGINE_TAG_UNKNOWN}, // bit 3 in runlist2EngMask
    {{"CE0",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_COPY0    }, 0, ENGINE_TAG_CE},      // bit 4 in runlist2EngMask
    {{"CE1",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_COPY1    }, 0, ENGINE_TAG_CE},      // bit 5 in runlist2EngMask
    {{"LWENC1", LW_PTOP_DEVICE_INFO_TYPE_ENUM_LWENC1   }, 0, ENGINE_TAG_UNKNOWN}, // bit 6 in runlist2EngMask
    {{"CE2",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_COPY2    }, 0, ENGINE_TAG_CE},      // bit 7 in runlist2EngMask
    {{"",       DEVICE_INFO_TYPE_ILWALID               }, 0, ENGINE_TAG_ILWALID}
};

void * fifoGetEngNames_GM107(void)
{
    return engName2DeviceInfo;
}

LwU32 fifoGetNumEng_GM107(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}

LwU32 fifoGetNumChannels_GM107(LwU32 runlistId)
{
    // Unused pre-Ampere
    (void) runlistId;
    return LW_PCCSR_CHANNEL__SIZE_1;
}

//
// TODO:
// Use LW_PTOP_DEVICE_INFO_ENGINE_ENUM values to index into
// LW_PFIFO_ENGINE_STATUS(i). Will need to set up a mapping
// similar to RM mapping first.
//
void fifoCheckEngStates_GM107(gpu_state* pGpuState)
{
    PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN();
}
