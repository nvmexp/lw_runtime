/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2019 by LWPU Corporation.  All rights reserved.  All
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
#include "pascal/gp100/dev_fifo.h"
#include "pascal/gp100/dev_ram.h"
#include "pascal/gp100/dev_top.h"
#include "mmu.h"
#include "vmem.h"

//
// Map the engine names, from projects.spec, to the values in
// LW_PTOP_DEVICE_INFO_TYPE_ENUM.
// New Engine enum types for Pascal
//        LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE
//        LW_PTOP_DEVICE_INFO_TYPE_ENUM_LWENC
//
// The array index is used to represent the engines in runlist2EngMask[].
//
static EngineNameValue engName2DeviceInfo[] =
{
    {{"GR0",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_GRAPHICS}, 0, ENGINE_TAG_GR},      // bit 0  in runlist2EngMask
    {{"LWDEC",  LW_PTOP_DEVICE_INFO_TYPE_ENUM_LWDEC   }, 0, ENGINE_TAG_UNKNOWN}, // bit 1  in runlist2EngMask
    {{"LWENC0", LW_PTOP_DEVICE_INFO_TYPE_ENUM_LWENC   }, 0, ENGINE_TAG_UNKNOWN}, // bit 2  in runlist2EngMask
    {{"SEC0",   LW_PTOP_DEVICE_INFO_TYPE_ENUM_SEC     }, 0, ENGINE_TAG_UNKNOWN}, // bit 3  in runlist2EngMask
    {{"CE0",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 0, ENGINE_TAG_CE},      // bit 4  in runlist2EngMask
    {{"CE1",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 1, ENGINE_TAG_CE},      // bit 5  in runlist2EngMask
    {{"LWENC1", LW_PTOP_DEVICE_INFO_TYPE_ENUM_LWENC   }, 1, ENGINE_TAG_UNKNOWN}, // bit 6  in runlist2EngMask
    {{"CE2",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 2, ENGINE_TAG_CE},      // bit 7  in runlist2EngMask
    {{"CE3",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 3, ENGINE_TAG_CE},      // bit 8  in runlist2EngMask
    {{"CE4",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 4, ENGINE_TAG_CE},      // bit 9  in runlist2EngMask
    {{"CE5",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 5, ENGINE_TAG_CE},      // bit 10 in runlist2EngMask
    {{"IOCTRL", LW_PTOP_DEVICE_INFO_TYPE_ENUM_IOCTRL  }, 0, ENGINE_TAG_UNKNOWN}, // bit 11 in runlist2EngMask
    {{"",       DEVICE_INFO_TYPE_ILWALID              }, 0, ENGINE_TAG_ILWALID}
};

LwU32 fifoGetNumEng_GP100(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}

/*!
 * @return The address limit for the address space used by a channel.
 */
LwU64
fifoGetChannelAddressLimit_GP100(ChannelId *pChannelId)
{
    LwU32       buf = 0;
    LwU64       instMemAddr = 0;
    LwU64       result = 0;

    readFn_t    readFn = NULL;

    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId, &readFn, NULL, NULL);
    if (readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return LW_ERR_NOT_SUPPORTED;
    }

    readFn(instMemAddr + SF_OFFSET(LW_RAMIN_ADR_LIMIT_HI), &result, 4);
    result <<= 32;
    readFn(instMemAddr + SF_OFFSET(LW_RAMIN_ADR_LIMIT_LO), &buf, 4);
    result += buf;

    return result;
}

LwU32 fifoGetPbdmaConfigSize_GP100(void)
{
    // The maximum PBDMA count per engine is 2 on Pascal / Volta / Turing
    return 2;
}

/*!
 * @return The number of PBDMAs provided by the chip.
 */
LwU32 fifoGetNumPbdma_GP100(void)
{
    LwU32 cfg = GPU_REG_RD32(LW_PFIFO_CFG0);

    return DRF_VAL(_PFIFO, _CFG0, _NUM_PBDMA, cfg);;
}

void * fifoGetEngNames_GP100(void)
{
    return engName2DeviceInfo;
}

/**
 * @brief Get the table entry for the engine from device info
 *
 * @param[in]  engNames
 * @param[in]  deviceInfoType
 * @param[in]  pDeviceInfoData
 * @param[in]  bDataValid
 */
LwU32
fifoGetTableEntry_GP100
(
    EngineNameValue *engNames,
    LwU32      deviceInfoType,
    LwU32      instId,
    LwBool     bDataValid
)
{
    LwU32 tblIdx = 0;
    while (engNames[tblIdx].nameValue.value != DEVICE_INFO_TYPE_ILWALID)
    {
        if (bDataValid && (deviceInfoType == engNames[tblIdx].nameValue.value)
             && (instId == engNames[tblIdx].instanceId))
        {
            return tblIdx;
        }
        tblIdx++;
    }
    return tblIdx;
}

LwU32
fifoGetInstanceIdFromDeviceInfoData_GP100
(
    LwU32 deviceInfoData
)
{
    return (DRF_VAL(_PTOP, _DEVICE_INFO, _DATA_INST_ID, deviceInfoData));
}
