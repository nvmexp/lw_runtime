/* _LWRM_COPYRIGHT_BEGIN_
 *
 *  Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 *  information contained herein is proprietary and confidential to LWPU
 *  Corporation.  Any use, reproduction, or disclosure without the written
 *  permission of LWPU Corporation is prohibited.
 *
 *  _LWRM_COPYRIGHT_END_
 */

#include "hopper/gh100/dev_top.h"
#include "hopper/gh100/dev_runlist.h"
#include "hopper/gh100/dev_ram.h"
#include "fifo.h"
#include "vmem.h"

#include "g_fifo_private.h"        // (rmconfig) implementation prototypes


void
fifoDumpEngStates_GH100
(
    ChannelId *pChannelId,
    gpu_state* pGpuState
)
{
    LwU32 buf;
    LwU64 instMemAddr = 0;
    LwU64 ctxSpace = 0;
    LwU64 lowBits;
    LwU64 highBits;
    readFn_t readFn = NULL;

    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId,
                                                                   &readFn,
                                                                   NULL, NULL);
    if (readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return;
    }

    dprintf("\n");

    readFn(instMemAddr + SF_OFFSET(LW_RAMIN_ENGINE_WFI_TARGET), &buf, 4);
    dprintf(" + %-36s :\n", "ENGINE");
    dprintf("   + %-34s : %03x\n", "TARGET",
            SF_VAL(_RAMIN, _ENGINE_WFI_TARGET, buf));
    dprintf("   + %-34s : %03x\n", "MODE",
            SF_VAL(_RAMIN, _ENGINE_WFI_MODE, buf));
    lowBits = SF_VAL(_RAMIN, _ENGINE_WFI_PTR_LO, buf) << 12;

    readFn(instMemAddr + SF_OFFSET(LW_RAMIN_ENGINE_WFI_PTR_HI), &buf, 4);
    highBits = SF_VAL(_RAMIN, _ENGINE_WFI_PTR_HI, buf);
    ctxSpace = lowBits + (highBits << 32);
    dprintf("   + %-34s : " LwU64_FMT "\n", "CTX SPACE", ctxSpace);
    dprintf("\n");

    pFifo[indexGpu].fifoDumpEngineStatus();
}


/**
 * Reads data about esched runlist whose runlist id is @p runlistId.
 *
 * @param in Id of hardware scheduler runlist (further in description runlist).
 * @param out pEngRunlistPtr - base of the runlist pointer.
 * @param out pRunListLenght - length of the runlist.
 * @param out pTgtAperture - target of the runlist.
 * @param out ppRunlistBuffer - allocated buffer in whose memory runlist data will be stored.
 *
 * @note if any of the parameters is NULL, specified parameter will not be initialized.
 */
LW_STATUS
fifoReadRunlistInfo_GH100
(
    LwU32   runlistId,
    LwU64  *pEngRunlistPtr,
    LwU32  *pRunlistLength,
    LwU32  *pTgtAperture,
    LwU32 **ppRunlistBuffer
)
{
    LW_STATUS status;
    LwU32 runlistPriBase;
    LwU32 runlistLength;
    LwU32 tgtAperture;
    LwU32 engRunlistBaseLo;
    LwU32 engRunlistBaseHi;
    LwU32 engRunlistSubmit;
    LwU64 runlistBasePtr;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, runlistId,
                                                 ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);
    if (status != LW_OK)
    {
        return status;
    }
    engRunlistBaseLo = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_SUBMIT_BASE_LO);
    engRunlistBaseHi = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_SUBMIT_BASE_HI);
    engRunlistSubmit = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_SUBMIT);

    runlistBasePtr = ((LwU64)DRF_VAL(_RUNLIST, _SUBMIT_BASE_HI, _PTR_HI, engRunlistBaseHi)
                        << DRF_SIZE(LW_RUNLIST_SUBMIT_BASE_LO_PTR_LO))
                     | (LwU64)DRF_VAL(_RUNLIST, _SUBMIT_BASE_LO, _PTR_LO, engRunlistBaseLo);
    runlistBasePtr <<= LW_RUNLIST_SUBMIT_BASE_LO_PTR_ALIGN_SHIFT;
    runlistLength = DRF_VAL(_RUNLIST, _SUBMIT, _LENGTH, engRunlistSubmit);
    tgtAperture = DRF_VAL(_RUNLIST, _SUBMIT_BASE_LO, _TARGET, engRunlistBaseLo);
    if (pEngRunlistPtr)
    {
        *pEngRunlistPtr = runlistBasePtr;
    }
    if (pRunlistLength)
    {
        *pRunlistLength = runlistLength;
    }
    if (pTgtAperture)
    {
        *pTgtAperture = tgtAperture;
    }

    if (NULL != ppRunlistBuffer)
    {
        status = pFifo[indexGpu].fifoAllocateAndFetchRunlist(runlistBasePtr, runlistLength,
                                                             tgtAperture, ppRunlistBuffer);
        if (LW_OK != status)
        {
            if (runlistLength != LW_RUNLIST_SUBMIT_LENGTH_ZERO)
            {
                dprintf("**ERROR: Could not fetch runlist info\n");
                return status;
            }

            *ppRunlistBuffer = NULL;
            return LW_OK;
        }
    }

    return LW_OK;
}


/**
 * Map the engine names, from projects.spec, to the values in
 * LW_PTOP_DEVICE_INFO2_TYPE_ENUM.
 *
 * See Chapter 7: Device Info table in dev_ptop.ref
 *
 * The array index is used to represent the engines in runListsEngines.
 */
static EngineNameValue engName2DeviceInfo[] =
{
    {{"GR0",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 0, ENGINE_TAG_GR},
    {{"GR1",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 1, ENGINE_TAG_GR},
    {{"GR2",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 2, ENGINE_TAG_GR},
    {{"GR3",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 3, ENGINE_TAG_GR},
    {{"GR4",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 4, ENGINE_TAG_GR},
    {{"GR5",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 5, ENGINE_TAG_GR},
    {{"GR6",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 6, ENGINE_TAG_GR},
    {{"GR7",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 7, ENGINE_TAG_GR},
    {{"CE0",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 0, ENGINE_TAG_CE},
    {{"CE1",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 1, ENGINE_TAG_CE},
    {{"CE2",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 2, ENGINE_TAG_CE},
    {{"CE3",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 3, ENGINE_TAG_CE},
    {{"CE4",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 4, ENGINE_TAG_CE},
    {{"CE5",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 5, ENGINE_TAG_CE},
    {{"CE6",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 6, ENGINE_TAG_CE},
    {{"CE7",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 7, ENGINE_TAG_CE},
    {{"CE8",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 8, ENGINE_TAG_CE},
    {{"CE9",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 9, ENGINE_TAG_CE},
    {{"LWENC0",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWENC    }, 0, ENGINE_TAG_LWENC},
    {{"LWENC1",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWENC    }, 1, ENGINE_TAG_LWENC},
    {{"LWENC2",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWENC    }, 2, ENGINE_TAG_LWENC},
    {{"LWDEC0",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 0, ENGINE_TAG_LWDEC},
    {{"LWDEC1",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 1, ENGINE_TAG_LWDEC},
    {{"LWDEC2",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 2, ENGINE_TAG_LWDEC},
    {{"LWDEC3",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 3, ENGINE_TAG_LWDEC},
    {{"LWDEC4",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 4, ENGINE_TAG_LWDEC},
    {{"LWDEC5",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 5, ENGINE_TAG_LWDEC},
    {{"LWDEC6",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 6, ENGINE_TAG_LWDEC},
    {{"LWDEC7",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 7, ENGINE_TAG_LWDEC},
    {{"SEC0",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_SEC      }, 0, ENGINE_TAG_SEC2},
    {{"LWJPG0",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 0, ENGINE_TAG_LWJPG},
    {{"LWJPG1",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 1, ENGINE_TAG_LWJPG},
    {{"LWJPG2",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 2, ENGINE_TAG_LWJPG},
    {{"LWJPG3",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 3, ENGINE_TAG_LWJPG},
    {{"LWJPG4",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 4, ENGINE_TAG_LWJPG},
    {{"LWJPG5",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 5, ENGINE_TAG_LWJPG},
    {{"LWJPG6",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 6, ENGINE_TAG_LWJPG},
    {{"LWJPG7",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 7, ENGINE_TAG_LWJPG},
    {{"IOCTRL0", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_IOCTRL   }, 0, ENGINE_TAG_IOCTRL},
    {{"IOCTRL1", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_IOCTRL   }, 1, ENGINE_TAG_IOCTRL},
    {{"IOCTRL2", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_IOCTRL   }, 2, ENGINE_TAG_IOCTRL},
    {{"OFA",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_OFA      }, 0, ENGINE_TAG_OFA},
    {{"GSP",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GSP      }, 0, ENGINE_TAG_GSP},
    {{"FLA",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_FLA      }, 0, ENGINE_TAG_FLA},
    {{"HSHUB0",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_HSHUB    }, 0, ENGINE_TAG_HSHUB},
    {{"HSHUB1",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_HSHUB    }, 1, ENGINE_TAG_HSHUB},
    {{"HSHUB2",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_HSHUB    }, 2, ENGINE_TAG_HSHUB},
    {{"HSHUB3",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_HSHUB    }, 3, ENGINE_TAG_HSHUB},
    {{"HSHUB4",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_HSHUB    }, 4, ENGINE_TAG_HSHUB},
    {{"C2C0",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_C2C      }, 0, ENGINE_TAG_C2C},
    {{"C2C1",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_C2C      }, 1, ENGINE_TAG_C2C},
    {{"FSP",     LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_FSP      }, 0, ENGINE_TAG_FSP},
    {{"",        DEVICE_INFO_TYPE_ILWALID                    }, 0, ENGINE_TAG_ILWALID}
};

void *fifoGetEngNames_GH100(void)
{
    return engName2DeviceInfo;
}
