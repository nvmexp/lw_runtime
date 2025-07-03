/* _LWRM_COPYRIGHT_BEGIN_
 *
 *  Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 *  information contained herein is proprietary and confidential to LWPU
 *  Corporation.  Any use, reproduction, or disclosure without the written
 *  permission of LWPU Corporation is prohibited.
 *
 *  _LWRM_COPYRIGHT_END_
 */


#include "ampere/ga100/dev_top.h"
#include "ampere/ga100/dev_runlist.h"
#include "ampere/ga100/dev_pbdma.h"
#include "ampere/ga100/dev_ctrl.h"
#include "ampere/ga100/dev_ram.h"
#include "fb.h"
#include "priv.h"
#include "vmem.h"
#include "vgpu.h"

#include "g_fifo_private.h"        // (rmconfig) implementation prototypes


/// @brief This macro is needed here to parse multi-word/array defines from HW
#define SF_ARR32_VAL(s,f,arr) \
    (((arr)[SF_INDEX(LW ## s ## f)] >> SF_SHIFT(LW ## s ## f)) & SF_MASK(LW ## s ## f))

static void      _readDeviceInfoCfg(void);
static LW_STATUS _getInstancePointer(ChannelId *pChannelId,
                                     LwU64     *pInstancePtr,
                                     LwU32     *pInstanceTarget);
static LwBool    checkFifoIntrInfo_GA100(LwU32 runlistId);
static void      getFifoIntrInfo_GA100(LwU32 runlistId);
static void      _printRunlistEngines(LwU32 runlistId);


static LwBool checkFifoIntrInfo_GA100(LwU32 runlistId)
{
    LwU32 intr;
    LwU32 lockedMask;
    LwU32 locklessMask;

    LwU32 status;
    LwU32 runlistPriBase;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
        runlistId, ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);
    if (status != LW_OK)
    {
        dprintf("***ERROR Could not get runlist PRI base\n");
        return LW_FALSE;
    }

    intr = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_INTR_0);
    dprintf("LW_RUNLIST_INTR_0:                     0x%08x\n", intr);

    lockedMask = GPU_REG_RD32(runlistPriBase +
                              LW_RUNLIST_INTR_0_EN_SET_TREE(0));
    dprintf("LW_RUNLIST_INTR_0_EN_SET_TREE(0):      0x%08x\n", lockedMask);

    locklessMask = GPU_REG_RD32(runlistPriBase +
                                LW_RUNLIST_INTR_0_EN_SET_TREE(1));
    dprintf("LW_RUNLIST_INTR_0_EN_SET_TREE(1):      0x%08x\n", locklessMask);

    return !!(intr & (lockedMask | locklessMask));
}

static void getFifoIntrInfo_GA100(LwU32 runlistId)
{
    LwU32 intr;
    LwU32 lockedMask;
    LwU32 locklessMask;

    LwU32 status;
    LwU32 runlistPriBase;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
        runlistId, ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);
    if (status != LW_OK)
    {
        dprintf("***ERROR Could not get runlist PRI base\n");
        return;
    }

    intr = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_INTR_0);
    dprintf("LW_RUNLIST_INTR_0:                     0x%08x\n", intr);
    priv_dump("LW_RUNLIST_INTR_0");

    lockedMask = GPU_REG_RD32(runlistPriBase +
                              LW_RUNLIST_INTR_0_EN_SET_TREE(0));
    dprintf("LW_RUNLIST_INTR_0_EN_SET_TREE(0):      0x%08x\n", lockedMask);
    priv_dump("LW_RUNLIST_INTR_0_EN_SET_TREE(0)");

    locklessMask = GPU_REG_RD32(runlistPriBase +
                                LW_RUNLIST_INTR_0_EN_SET_TREE(1));
    dprintf("LW_RUNLIST_INTR_0_EN_SET_TREE(1):      0x%08x\n", locklessMask);
    priv_dump("LW_RUNLIST_INTR_0_EN_SET_TREE(1)");

    if (verboseLevel)
    {
        pFifo[indexGpu].fifoDumpFifoIntrInfo(runlistId, intr, lockedMask);
        pFifo[indexGpu].fifoDumpFifoIntrInfo(runlistId, intr, locklessMask);
    }
}

//-----------------------------------------------------
// fifoDumpFifoIntrInfo_GA100
//
// This function dumps information about various pending
// interrupts based on interrupt mask.
//-----------------------------------------------------
void fifoDumpFifoIntrInfo_GA100(LwU32 runlistId, LwU32 intr, LwU32 intrMsk)
{
    LwU32 regRead;
    LwU32 status;
    LwU32 runlistPriBase;

    intr &= intrMsk;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
        runlistId, ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);
    if (status != LW_OK)
    {
        dprintf("***ERROR Could not get runlist PRI base\n");
        return;
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _CTXSW_TIMEOUT_ENG0, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistPriBase +
                               LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(0));
        dprintf(" + LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(0): 0x%08x\n",
                regRead);
        priv_dump("LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(0)");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _CTXSW_TIMEOUT_ENG1, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistPriBase +
                               LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(1));
        dprintf(" + LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(1): 0x%08x\n",
                regRead);
        priv_dump("LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(1)");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _CTXSW_TIMEOUT_ENG2, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistPriBase +
                               LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(2));
        dprintf(" + LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(2): 0x%08x\n",
                regRead);
        priv_dump("LW_RUNLIST_ENGINE_CTXSW_TIMEOUT_INFO(2)");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _RUNLIST_IDLE, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistId + LW_RUNLIST_INFO);
        dprintf(" + LW_RUNLIST_INTR_0_RUNLIST_IDLE_PENDING\n");
        dprintf(" + LW_RUNLIST_INFO                       : 0x%08x\n", regRead);
        priv_dump("LW_RUNLIST_INFO");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _RUNLIST_AND_ENG_IDLE, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistId + LW_RUNLIST_INFO);
        dprintf(" + LW_RUNLIST_INTR_0_RUNLIST_AND_ENG_IDLE_PENDING\n");
        dprintf(" + LW_RUNLIST_INFO                       : 0x%08x\n", regRead);
        priv_dump("LW_RUNLIST_INFO");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _RUNLIST_ACQUIRE, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistId + LW_RUNLIST_INFO);
        dprintf(" + LW_RUNLIST_INTR_0_RUNLIST_ACQUIRE_PENDING\n");
        dprintf(" + LW_RUNLIST_INFO                       : 0x%08x\n", regRead);
        priv_dump("LW_RUNLIST_INFO");
    }

    if (intr &
        DRF_DEF(_RUNLIST, _INTR_0, _RUNLIST_ACQUIRE_AND_ENG_IDLE, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistId + LW_RUNLIST_INFO);
        dprintf(" + LW_RUNLIST_INTR_0_RUNLIST_ACQUIRE_AND_ENG_IDLE_PENDING\n");
        dprintf(" + LW_RUNLIST_INFO                       : 0x%08x\n", regRead);
        priv_dump("LW_RUNLIST_INFO");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _BAD_TSG, _PENDING))
    {
        regRead = GPU_REG_RD32(runlistId + LW_RUNLIST_INTR_BAD_TSG);
        dprintf(" + LW_RUNLIST_INTR_BAD_TSG               : 0x%08x\n", regRead);
        priv_dump("LW_RUNLIST_INTR_BAD_TSG");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _TSG_PREEMPT_COMPLETE, _PENDING))
    {
        dprintf(" + LW_RUNLIST_INTR_0_TSG_PREEMPT_COMPLETE\n");
    }

    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _RUNLIST_PREEMPT_COMPLETE, _PENDING))
    {
        dprintf(" + LW_RUNLIST_INTR_0_RUNLIST_PREEMPT_COMPLETE\n");
    }

    // Dump PBDMA info
    if (intr & DRF_DEF(_RUNLIST, _INTR_0, _PBDMA0_INTR_TREE_0, _PENDING) ||
        intr & DRF_DEF(_RUNLIST, _INTR_0, _PBDMA1_INTR_TREE_0, _PENDING) ||
        intr & DRF_DEF(_RUNLIST, _INTR_0, _PBDMA0_INTR_TREE_1, _PENDING) ||
        intr & DRF_DEF(_RUNLIST, _INTR_0, _PBDMA1_INTR_TREE_1, _PENDING))
    {
        pFifo[indexGpu].fifoGetPerRunlistIntrPbdmaInfo(runlistId);
    }
}

static char* getOpcodeStr(LwU32 opcode)
{
    switch (opcode)
    {
        case LW_PPBDMA_GP_ENTRY1_OPCODE_NOP:
            return "_NOP";
        case LW_PPBDMA_GP_ENTRY1_OPCODE_ILLEGAL:
            return "_ILLEGAL";
        case LW_PPBDMA_GP_ENTRY1_OPCODE_GP_CRC:
            return "_GP_CRC";
        case LW_PPBDMA_GP_ENTRY1_OPCODE_PB_CRC:
            return "_PB_CRC";
        default:
            return "unknown";
    }
}

/**
 * Map the engine names, from projects.spec, to the values in
 * LW_PTOP_DEVICE_INFO2_TYPE_ENUM.
 *
 * The array index is used to represent the engines in runListsEngines.
 */
static EngineNameValue engName2DeviceInfo[] =
{
    {{"GR0",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 0, ENGINE_TAG_GR},
    {{"GR1",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 1, ENGINE_TAG_GR},
    {{"GR2",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 2, ENGINE_TAG_GR},
    {{"GR3",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 3, ENGINE_TAG_GR},
    {{"GR4",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 4, ENGINE_TAG_GR},
    {{"GR5",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 5, ENGINE_TAG_GR},
    {{"GR6",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 6, ENGINE_TAG_GR},
    {{"GR7",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GRAPHICS }, 7, ENGINE_TAG_GR},
    {{"CE0",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 0, ENGINE_TAG_CE},
    {{"CE1",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 1, ENGINE_TAG_CE},
    {{"CE2",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 2, ENGINE_TAG_CE},
    {{"CE3",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 3, ENGINE_TAG_CE},
    {{"CE4",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 4, ENGINE_TAG_CE},
    {{"CE5",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 5, ENGINE_TAG_CE},
    {{"CE6",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 6, ENGINE_TAG_CE},
    {{"CE7",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 7, ENGINE_TAG_CE},
    {{"CE8",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 8, ENGINE_TAG_CE},
    {{"CE9",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE      }, 9, ENGINE_TAG_CE},
    {{"LWENC0", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWENC    }, 0, ENGINE_TAG_LWENC},
    {{"LWENC1", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWENC    }, 1, ENGINE_TAG_LWENC},
    {{"LWENC2", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWENC    }, 2, ENGINE_TAG_LWENC},
    {{"LWDEC0", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 0, ENGINE_TAG_LWDEC},
    {{"LWDEC1", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 1, ENGINE_TAG_LWDEC},
    {{"LWDEC2", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 2, ENGINE_TAG_LWDEC},
    {{"LWDEC3", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 3, ENGINE_TAG_LWDEC},
    {{"LWDEC4", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWDEC    }, 4, ENGINE_TAG_LWDEC},
    {{"SEC0",   LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_SEC      }, 0, ENGINE_TAG_SEC2},
    {{"LWJPG",  LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LWJPG    }, 0, ENGINE_TAG_LWJPG},
    {{"IOCTRL", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_IOCTRL   }, 0, ENGINE_TAG_IOCTRL},
    {{"IOCTRL", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_IOCTRL   }, 1, ENGINE_TAG_IOCTRL},
    {{"IOCTRL", LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_IOCTRL   }, 2, ENGINE_TAG_IOCTRL},
    {{"OFA",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_OFA      }, 0, ENGINE_TAG_OFA},
    {{"GSP",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_GSP      }, 0, ENGINE_TAG_GSP},
    {{"FLA",    LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_FLA      }, 0, ENGINE_TAG_FLA},
    {{"",       DEVICE_INFO_TYPE_ILWALID                    }, 0, ENGINE_TAG_ILWALID}
};

/*!
 * Get information about PBDMA interrupts for PBDMAS of given runlist
 */
void fifoGetPerRunlistIntrPbdmaInfo_GA100(LwU32 runlistId)
{
    const DeviceInfoEngine *pEngine;
    LW_STATUS status;
    LwU32 i;
    LwU32 intr;
    LwU32 engineIdx;
    LwU32 runlistPriBase;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
                                                 runlistId,
                                                 ENGINE_INFO_TYPE_ILWALID,
                                                 &engineIdx);
    if (status != LW_OK)
    {
        dprintf("**ERROR: Cannot get deviceInfo engine entry for runlist %s\n",
                __FUNCTION__);
        return;
    }

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
        runlistId, ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);
    if (status != LW_OK)
    {
        dprintf("***ERROR Could not get runlist PRI base\n");
        return;
    }

    intr = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_INTR_0);
    dprintf(" + LW_RUNLIST_INTR_0                   : 0x%08x\n", intr);

    pEngine = &deviceInfo.pEngines[engineIdx];
    for (i = 0; i < pEngine->numPbdmas; ++i)
    {
        if (intr & (LW_RUNLIST_INTR_0_PBDMAi_INTR_TREE_j_PENDING
                    << DRF_SHIFT(LW_RUNLIST_INTR_0_PBDMAi_INTR_TREE_j(i, 0))) ||
            intr & (LW_RUNLIST_INTR_0_PBDMAi_INTR_TREE_j_PENDING
                    << DRF_SHIFT(LW_RUNLIST_INTR_0_PBDMAi_INTR_TREE_j(i, 1))))
        {
            dprintf("    + LW_RUNLIST_INTR_0_PBDMA%d_INTR_TREE* : _PENDING\n",
                    i);
            pFifo[indexGpu].fifoDumpPerPbdmaIntrInfo(pEngine->pPbdmaIds[i]);
        }
    }
}

/*!
 * Get maximum number of engines. For Ampere+, this per-runlist max number of
 * engines.
 */
LwU32 fifoGetNumEng_GA100(void)
{
    return LW_RUNLIST_ENGINE_STATUS0__SIZE_1;
}

/*!
 * Dump status information about each active engine.
 */
void fifoDumpEngineStatus_GA100(void)
{
    LwU32 engineId;
    LwU32 runlistId;
    LwU32 engineStatusSize = pFifo[indexGpu].fifoGetNumEng();
    LwU32 *runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();

    for (runlistId = 0; runlistId < pFifo[indexGpu].fifoGetRunlistMaxNumber();
         runlistId++)
    {
        LwU32 runlistPriBase;
        LW_STATUS status;

        if (runListsEngines[runlistId] == 0)
        {
            continue;
        }

        status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
            runlistId, ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);

        if (status != LW_OK)
        {
            dprintf("**ERROR: Cannot get runlist PRI base %s\n", __FUNCTION__);
            return;
        }

        for (engineId = 0; engineId < engineStatusSize; engineId++)
        {
            LwU32 val = GPU_REG_RD32(runlistPriBase +
                                     LW_RUNLIST_ENGINE_STATUS0(engineId));

            if (SF_VAL(_RUNLIST, _ENGINE_STATUS0_CTX_STATUS, val) ==
                LW_RUNLIST_ENGINE_STATUS0_CTX_STATUS_VALID)
            {
                dprintf("RUNLIST ID %d\n", runlistId);
                dprintf("PER-RUNLIST ENGINE NUMBER %d\n", engineId);
                dprintf("   + LW_RUNLIST_ENGINE_STATUS0_TSGID\n");

                dprintf("   + DIRECT STATUS:                    %d\n",
                        SF_VAL(_RUNLIST, _ENGINE_STATUS0_ENGINE, val));
                dprintf("   + TSG ID:                       %03d\n",
                        SF_VAL(_RUNLIST, _ENGINE_STATUS0_TSGID, val));
            }
        }
    }
}

//-----------------------------------------------------
// PBDMA_CACHE1(n,id)
//
// This macro is used by fifoDumpPbdmaRegsCache1_GA100()
// to dump out LW_PBDMA_METHOD/DATA0/1/2/3 info.
//-----------------------------------------------------
#define PBDMA_CACHE1(n, id)                                   \
do                                                            \
{                                                             \
    regRead = GPU_REG_RD32(LW_PPBDMA_METHOD##n(id));          \
    addr    = DRF_VAL(_PPBDMA_METHOD, n, _ADDR, regRead);     \
                                                              \
    subch = DRF_VAL(_PPBDMA_METHOD, n, _SUBCH, regRead);      \
                                                              \
    first = DRF_VAL(_PPBDMA_METHOD, n, _FIRST, regRead);      \
                                                              \
    valid   = DRF_VAL(_PPBDMA_METHOD, n, _VALID, regRead);    \
    regRead = GPU_REG_RD32(LW_PPBDMA_DATA##n(id));            \
    value   = DRF_VAL(_PPBDMA_DATA, n, _VALUE, regRead);      \
                                                              \
    dprintf("%d    0x%04x    %d      ", n, addr << 2, subch); \
    dprintf("%s     %s",                                      \
            pbdmaTrueFalse[first].strName,                    \
            pbdmaTrueFalse[valid].strName);                   \
    dprintf("    0x%08x\n", value);                           \
} while (0)


//-----------------------------------------------------
// fifoDumpPbdmaRegsCache1_GA100(LwU32 pbdmaId)
//
// This function dumps out pbdma cache1 regs for given pbdmaId
// Uses DUMP_METHOD and DUMP_DATA macros
//-----------------------------------------------------
void fifoDumpPbdmaRegsCache1_GA100(LwU32 pbdmaId)
{
    LwU32 regRead;
    LwU32 addr;
    LwU32 subch;
    LwU32 first;
    LwU32 valid;
    LwU32 value;

    NameValue pbdmaTrueFalse[2] = {{"False", 0x0}, {"True ", 0x1}};

    dprintf("\n");
    dprintf(" PBDMA %d CACHE1 :  \n", pbdmaId);
    dprintf("      ----------- METHOD -------------   ---DATA---\n");
    dprintf("N     ADDR    SUBCH    FIRST     VALID      VALUE \n");
    dprintf("--    ----    -----    -----     -----      -----\n");
    PBDMA_CACHE1(0, pbdmaId);
    PBDMA_CACHE1(1, pbdmaId);
    PBDMA_CACHE1(2, pbdmaId);
    PBDMA_CACHE1(3, pbdmaId);
}

/*!
 * Checks if a runlist length is valid (non-zero)
 */
LwBool fifoIsRunlistLengthValid_GA100(LwU32 runlistLength)
{
    return LW_RUNLIST_SUBMIT_LENGTH_ZERO < runlistLength &&
           runlistLength <= LW_RUNLIST_SUBMIT_LENGTH_MAX;
}

void fifoGetChannelOffsetMask_GA100(LwU32 runlistId, LwU32 *pOffset, LwU32 *pMask)
{

    LwU32 val;
    LwU32 runlistPriBase;
    LW_STATUS status;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
        runlistId, ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);

    if (status != LW_OK)
    {
        dprintf("**ERROR: Cannot get runlist PRI base %s\n", __FUNCTION__);
        return;
    }

    val = GPU_REG_RD32(runlistPriBase +
                       LW_RUNLIST_VIRTUAL_CHANNEL_CFG(getGfid()));

    *pMask = DRF_VAL(_RUNLIST, _VIRTUAL_CHANNEL_CFG, _MASK, val);
    *pOffset = DRF_VAL(_RUNLIST, _VIRTUAL_CHANNEL_CFG, _SET,  val);
}

void *fifoGetEngNames_GA100(void)
{
    return engName2DeviceInfo;
}

LwU32 fifoGetPbdmaConfigSize_GA100(void)
{
    return LW_RUNLIST_PBDMA_CONFIG__SIZE_1;
}

/*!
 * Gets information about interrupts for a pbdma id.
 */
void fifoDumpPerPbdmaIntrInfo_GA100(LwU32 pbdmaId)
{
    char buffer[FIFO_REG_NAME_BUFFER_LEN];
    buffer[0] = '\0';

    // INTR_0
    dprintf("       LW_PPBDMA_INTR_0_EN_SET_TREE 0  : 0x%08x\n",
            GPU_REG_RD32(LW_PPBDMA_INTR_0_EN_SET_TREE(pbdmaId, 0)));
    dprintf("       LW_PPBDMA_INTR_0_EN_SET_TREE 1  : 0x%08x\n",
            GPU_REG_RD32(LW_PPBDMA_INTR_0_EN_SET_TREE(pbdmaId, 1)));
    dprintf("       LW_PPBDMA_INTR_0                : 0x%08x\n",
            GPU_REG_RD32(LW_PPBDMA_INTR_0(pbdmaId)));
    dprintf("\n");

    sprintf(buffer, "LW_PPBDMA_INTR_0_EN_SET_TREE*(%d,0)", pbdmaId);
    priv_dump(buffer);

    sprintf(buffer, "LW_PPBDMA_INTR_0_EN_SET_TREE*(%d,1)", pbdmaId);
    priv_dump(buffer);

    sprintf(buffer, "LW_PPBDMA_INTR_0*(%d)", pbdmaId);
    priv_dump(buffer);

    // INTR_1
    dprintf("       LW_PPBDMA_INTR_1_EN_SET_TREE 0  : 0x%08x\n",
            GPU_REG_RD32(LW_PPBDMA_INTR_1_EN_SET_TREE(pbdmaId, 0)));
    dprintf("       LW_PPBDMA_INTR_1_EN_SET_TREE 1  : 0x%08x\n",
            GPU_REG_RD32(LW_PPBDMA_INTR_1_EN_SET_TREE(pbdmaId, 1)));
    dprintf("       LW_PPBDMA_INTR_1                : 0x%08x\n",
            GPU_REG_RD32(LW_PPBDMA_INTR_1(pbdmaId)));
    dprintf("\n");

    sprintf(buffer, "LW_PPBDMA_INTR_1_EN_SET_TREE*(%d,0)", pbdmaId);
    priv_dump(buffer);

    sprintf(buffer, "LW_PPBDMA_INTR_1_EN_SET_TREE*(%d,1)", pbdmaId);
    priv_dump(buffer);

    sprintf(buffer, "LW_PPBDMA_INTR_1*(%d)", pbdmaId);
    priv_dump(buffer);
}

//-----------------------------------------------------
// fifoGetPbdmaState_GA100
//
// Prints information for each PBDMA unit.
// If the PBDMA is bound to a channel then prints
// things like the GP GET and PUT pointers. For this
// it needs to freeze the host and unfreeze it after
// reading what is needed.
//-----------------------------------------------------
void
fifoGetPbdmaState_GA100(void)
{
    LwU32 i;
    LwU32 buf;
    LwU64 gpBase;
    LwU32 chid;
    const LwU32 num_pbdma = pFifo[indexGpu].fifoGetNumPbdma();

    dprintf("\n");

    for(i=0; i < num_pbdma; i++)
    {
        buf = GPU_REG_RD32(LW_PPBDMA_CHANNEL(i));
        chid = SF_VAL(_PPBDMA, _CHANNEL_CHID, buf);

        if (buf == 0xbadf1002)
        {
            dprintf(" + PBDMA %03d floorswept, ignoring\n", i);
            continue;
        }
        if(chid == LW_PPBDMA_CHANNEL_VALID_FALSE)
        {
            dprintf(" + PBDMA %03d ON CHANNEL:               NULL\n", i);
        }
        else
        {
            if (isVirtualWithSriov())
            {
                LwU32 runlistId;
                if (pFifo[indexGpu].fifoEngineDataXlate(
                        ENGINE_INFO_TYPE_PBDMA_ID, i,
                        ENGINE_INFO_TYPE_RUNLIST, &runlistId) != LW_OK)
                {
                    dprintf("**ERROR: Cannot find runlist ID for PBDMA %d\n",
                            i);
                    continue;
                }

                if ((chid < pFifo[indexGpu].fifoGetSchidForVchid(0)) ||
                    (chid > pFifo[indexGpu].fifoGetSchidForVchid(0) +
                            pFifo[indexGpu].fifoGetNumChannels(runlistId)))
                {
                    // Not this VF's channel. Skip it.
                    continue;
                }
                else
                {
                    chid -= pFifo[indexGpu].fifoGetSchidForVchid(0);
                }
            }
            dprintf(" + PBDMA %03d ON CHANNEL:               %03d\n", i, chid);
            buf = GPU_REG_RD32(LW_PPBDMA_GP_BASE_HI(i));
            gpBase = SF_VAL(_PPBDMA, _GP_BASE_HI_OFFSET, buf);
            gpBase <<= 32;
            buf = GPU_REG_RD32(LW_PPBDMA_GP_BASE(i));
            buf = SF_VAL(_PPBDMA, _GP_BASE_OFFSET, buf) << 3;
            gpBase += buf;

            dprintf("   + GP_BASE:                          " LwU64_FMT "\n", gpBase);

            buf = GPU_REG_RD32(LW_PPBDMA_GP_GET(i));
            buf = SF_VAL(_PPBDMA, _GP_PUT_ENTRY, buf);
            dprintf("   + GP GET POINTER:                   0x%08x\n", buf);

            buf = GPU_REG_RD32(LW_PPBDMA_GP_PUT(i));
            buf = SF_VAL(_PPBDMA, _GP_GET_ENTRY, buf);
            dprintf("   + GP PUT POINTER:                   0x%08x\n", buf);

            buf = GPU_REG_RD32(LW_PPBDMA_GP_FETCH(i));
            buf = SF_VAL(_PPBDMA, _GP_FETCH_ENTRY, buf);
            dprintf("   + GP FETCH POINTER:                 0x%08x\n", buf);
        }

        buf = GPU_REG_RD32(LW_PPBDMA_TARGET(i));
        dprintf(" + PBDMA %03d ON ENGINE:                %03d\n", i, SF_VAL(_PPBDMA, _TARGET_ENGINE, buf));
    }
}

//-----------------------------------------------------
// fifoDumpPb_GA100
//
// This function dumps the pushbuffer for each PBDMA
// unit that has a channel bound to it. It finds the
// GPFIFO GET pointer location and starts reading
// GP entries until it reaches the GPFIFO PUT pointer.
// For each entry it dumps the pushbuffer segment that
// corresponds to this entry.
//-----------------------------------------------------
void fifoDumpPb_GA100(LwU32 chid, LwU32 pbOffset, LwU32 sizeInBytes, LwU32 printParsed)
{
    LwU32       i,j;

    LwU32       buf;
    LwU32       lowerBits;
    LwU32       higherBits;
    LwU32       gpGet;
    LwU32       gpPut;
    LwU32       chId;

    LwU64       gpBase;
    LwU64       realGpGet;
    LwU64       pbGetAddr;
    LwU64       pbGetLength;
    LwU32       operand = 0;
    LwU32       opcode  = 0;
    const LwU32 num_pbdma = pFifo[indexGpu].fifoGetNumPbdma();

    // for each PBDMA unit
    for(i=0; i < num_pbdma; i++)
    {
        buf = GPU_REG_RD32(LW_PPBDMA_CHANNEL(i));

        // if there is a channel bound to this PBDMA unit
        if(SF_VAL(_PPBDMA, _CHANNEL_VALID, buf) != LW_PPBDMA_CHANNEL_VALID_FALSE)
        {
            chId = SF_VAL(_PPBDMA, _CHANNEL_CHID, buf);

            // Check for a CHID specified and if so matching this PBDMA
            if ((chid == ~0) || (chid == chId))
            {
                dprintf("+ PBDMA #%03d:       CHANNEL #%03d\n", i, chId);

                // get GP base
                buf = GPU_REG_RD32(LW_PPBDMA_GP_BASE_HI(i));
                gpBase = SF_VAL(_PPBDMA, _GP_BASE_HI_OFFSET, buf);
                gpBase <<= 32;
                buf = GPU_REG_RD32(LW_PPBDMA_GP_BASE(i));
                buf = SF_VAL(_PPBDMA, _GP_BASE_OFFSET, buf) << 3;
                gpBase += buf;

                // get GP get pointer
                buf = GPU_REG_RD32(LW_PPBDMA_GP_GET(i));
                gpGet = SF_VAL(_PPBDMA, _GP_GET_ENTRY, buf);

                // get GP put pointer
                buf = GPU_REG_RD32(LW_PPBDMA_GP_PUT(i));
                gpPut = SF_VAL(_PPBDMA, _GP_PUT_ENTRY, buf);

                // this should not be here, just put it to test the pb command
                //if(gpGet>0 && gpGet == gpPut)
                //    gpGet--;

                dprintf("    GPFIFO BASE ADDR:             " LwU64_FMT "\n", gpBase);
                dprintf("    GPFIFO GET:                   0x%08x\n", gpGet);
                dprintf("    GPFIFO PUT:                   0x%08x\n", gpPut);

                // go through all the GP entries
                for (; gpGet < gpPut; gpGet++)
                {
                    // get real GP get pointer address
                    realGpGet = gpGet * LW_PPBDMA_GP_ENTRY__SIZE + gpBase;

                    // read from GPFIFO the higher bits from the PB get address
                    // and the length of that segment in the PB
                    readGpuVirtualAddr_GK104(chId, realGpGet+4, &higherBits, 4);
                    pbGetAddr = SF_VAL(_PPBDMA, _GP_ENTRY1_GET_HI, higherBits);
                    pbGetAddr <<= 32;
                    pbGetLength = SF_VAL(_PPBDMA, _GP_ENTRY1_LENGTH, higherBits);

                    // read from GPFIFO the lower bits from the PB get address
                    // the address in PB is 2-bits aligned, so shift this value
                    readGpuVirtualAddr_GK104(chId, realGpGet, &lowerBits, 4);
                    pbGetAddr += SF_VAL(_PPBDMA, _GP_ENTRY0_GET, lowerBits) << 2;

                    // print the contents of this GP entry

                    dprintf("    GP ENTRY 0x%08x CONTENTS: 0x%08x%08x\n", gpGet, lowerBits, higherBits);
                    dprintf("    GP ENTRY ADDRESS:             " LwU64_FMT "\n", realGpGet);
                    dprintf("    PB GET ADDRESS:               " LwU64_FMT "\n", pbGetAddr);
                    dprintf("    PB GET LENGTH:                " LwU64_FMT "\n", pbGetLength);

                    if (pbGetLength == 0)
                    {
                        opcode = SF_VAL(_PPBDMA, _GP_ENTRY1_OPCODE, higherBits);
                        operand = SF_VAL(_PPBDMA, _GP_ENTRY0_OPERAND, lowerBits);
                        dprintf("    GP ENTRY : control entry\n");
                        dprintf("    GP ENTRY OPERAND:             0x%x\n", operand);
                        dprintf("    GP ENTRY OPERAND:             %s\n", getOpcodeStr(opcode));
                        continue;
                    }
                    dprintf("    PB DATA FOR THIS ENTRY:\n");

                    for(j=0; j < pbGetLength; j++)
                    {
                        if(j%4 == 0)
                        {
                            dprintf("        " LwU64_FMT ":", pbGetAddr+j*4);
                        }

                        readGpuVirtualAddr_GK104(chId, pbGetAddr+j*4, &buf, 4);
                        dprintf(" %08x", buf);

                        // do that newline separately for better output
                        if(j%4 == 3)
                        {
                            dprintf("\n");
                        }
                    }

                    dprintf("\n");
                }
            }
        }
    }
}

static void _readDeviceInfoCfg(void)
{
    LwU32 deviceInfoCfgReg;
    if (!deviceInfo.cfg.bValid)
    {
        deviceInfoCfgReg = GPU_REG_RD32(LW_PTOP_DEVICE_INFO_CFG);
        deviceInfo.cfg.version = DRF_VAL(_PTOP, _DEVICE_INFO_CFG, _VERSION, deviceInfoCfgReg);
        deviceInfo.cfg.maxDevices = DRF_VAL(_PTOP, _DEVICE_INFO_CFG, _MAX_DEVICES, deviceInfoCfgReg);
        deviceInfo.cfg.maxRowsPerDevice = DRF_VAL(_PTOP, _DEVICE_INFO_CFG, _MAX_ROWS_PER_DEVICE, deviceInfoCfgReg);
        deviceInfo.cfg.numRows = DRF_VAL(_PTOP, _DEVICE_INFO_CFG, _NUM_ROWS, deviceInfoCfgReg);
        deviceInfo.cfg.bValid = LW_TRUE;
    }
}

LwU32 fifoGetDeviceInfoNumRows_GA100(void)
{
    _readDeviceInfoCfg();
    return deviceInfo.cfg.numRows;
}

LwU32 fifoGetDeviceInfoMaxDevices_GA100(void)
{
    _readDeviceInfoCfg();
    return deviceInfo.cfg.maxDevices;
}

/**
 * @note This number should be increased if maximum number of runlists is increased.
 * At the time of writing the code this was sufficient, however it can change later in the future,
 * and this should be updated. There is not a value in dev_runlist.h that defines maximum number of runlists,
 * so we used maximum number of devices.
 */
LwU32 fifoGetRunlistMaxNumber_GA100(void)
{
    return LW_PTOP_DEVICE_INFO_CFG_MAX_DEVICES_STATIC;
}

LwU32 fifoGetTableEntry_GA100
(
    EngineNameValue *engNames,
    LwU32            deviceInfoType,
    LwU32            instId,
    LwBool           bDataValid
)
{
    LwU32 tableIndex;

    for (tableIndex = 0; engNames[tableIndex].nameValue.value != DEVICE_INFO_TYPE_ILWALID; tableIndex++)
    {
        if ((engNames[tableIndex].nameValue.value == deviceInfoType) &&
            (engNames[tableIndex].instanceId == instId))
        {
            break;
        }
    }
    return tableIndex;
}

LW_STATUS
fifoParseDeviceInfoAndGetEngine_GA100
(
    DeviceInfoEngine *pEngine,
    LwU32            *pDeviceAclwm,
    LwU32            *pLookup
)
{
    EngineNameValue *engNames;
    LwU32 tableIndex;
    engNames = pFifo[indexGpu].fifoGetEngNames();
    tableIndex = pFifo[indexGpu].fifoGetTableEntry(engNames,
                                                   SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_TYPE_ENUM, pDeviceAclwm),
                                                   SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_INSTANCE_ID, pDeviceAclwm),
                                                   LW_TRUE);
    if (engNames[tableIndex].nameValue.value == DEVICE_INFO_TYPE_ILWALID)
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    *pLookup = tableIndex;
    pEngine->engineName = engNames[tableIndex].nameValue.strName;
    pEngine->engineData[ENGINE_INFO_TYPE_ENGINE_TAG] = engNames[tableIndex].engineTag;
    pEngine->engineData[ENGINE_INFO_TYPE_INST_ID] = engNames[tableIndex].instanceId;
    pEngine->engineData[ENGINE_INFO_TYPE_ENGINE_TYPE] = SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_TYPE_ENUM, pDeviceAclwm);
    pEngine->bHostEng = SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_IS_ENGINE, pDeviceAclwm);
    pEngine->engineData[ENGINE_INFO_TYPE_RESET] = SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_RESET_ID, pDeviceAclwm);
    pEngine->engineData[ENGINE_INFO_TYPE_MMU_FAULT_ID] = SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_FAULT_ID, pDeviceAclwm);

    if (pEngine->bHostEng)
    {
        const LwU32 runListPriBase =
            SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_RUNLIST_PRI_BASE, pDeviceAclwm) << SF_SHIFT(LW_PTOP_DEVICE_INFO2_DEV_RUNLIST_PRI_BASE);
        const LwU32 runlistEngineId = SF_ARR32_VAL(_PTOP, _DEVICE_INFO2_DEV_RLENG_ID, pDeviceAclwm);
        LwU32 pbdmaIds[LW_RUNLIST_PBDMA_CONFIG__SIZE_1];
        LwU32 pbdmaFaultIds[LW_RUNLIST_PBDMA_CONFIG__SIZE_1];
        LwU32 i;
        LwU32 doorbellConfig;
        LwU32 channelConfig;
        LwU32 engineStatusDebug;
        LwLength pPbdmaIdsAllocSize;

        doorbellConfig = GPU_REG_RD32(runListPriBase + LW_RUNLIST_DOORBELL_CONFIG);
        channelConfig = GPU_REG_RD32(runListPriBase + LW_RUNLIST_CHANNEL_CONFIG);
        engineStatusDebug = GPU_REG_RD32(runListPriBase + LW_RUNLIST_ENGINE_STATUS_DEBUG(runlistEngineId));
        pEngine->engineData[ENGINE_INFO_TYPE_RUNLIST_PRI_BASE] = runListPriBase;
        pEngine->engineData[ENGINE_INFO_TYPE_RUNLIST_ENGINE_ID] = runlistEngineId;
        pEngine->engineData[ENGINE_INFO_TYPE_RUNLIST] = DRF_VAL(_RUNLIST, _DOORBELL_CONFIG, _ID, doorbellConfig);
        pEngine->engineData[ENGINE_INFO_TYPE_CHRAM_PRI_BASE] =
            DRF_VAL(_RUNLIST, _CHANNEL_CONFIG, _CHRAM_BAR0_OFFSET, channelConfig) << DRF_BASE(LW_RUNLIST_CHANNEL_CONFIG_CHRAM_BAR0_OFFSET);
        pEngine->engineData[ENGINE_INFO_TYPE_FIFO_TAG] = DRF_VAL(_RUNLIST, _ENGINE_STATUS_DEBUG, _ENGINE_ID, engineStatusDebug);

        for (i = 0; i < LW_ARRAY_ELEMENTS(pbdmaIds); i++)
        {
            LwU32 pbdmaConfig;
            pbdmaConfig = GPU_REG_RD32(runListPriBase + LW_RUNLIST_PBDMA_CONFIG(i));

            if (!FLD_TEST_DRF(_RUNLIST, _PBDMA_CONFIG, _VALID, _TRUE, pbdmaConfig))
            {
                break;
            }

            pbdmaIds[i] = DRF_VAL(_RUNLIST, _PBDMA_CONFIG, _PBDMA_ID, pbdmaConfig);

            pbdmaConfig = GPU_REG_RD32(LW_PPBDMA_CFG0(pbdmaIds[i]));
            pbdmaFaultIds[i] = DRF_VAL(_PPBDMA, _CFG0, _PBDMA_FAULT_ID, pbdmaConfig);
        }

        // Fix PBDMA IDs for GRCE (LCE 0 and 1)
        if (pEngine->engineData[ENGINE_INFO_TYPE_ENGINE_TAG] == ENGINE_TAG_CE &&
            (pEngine->engineData[ENGINE_INFO_TYPE_INST_ID] == 0 ||
             pEngine->engineData[ENGINE_INFO_TYPE_INST_ID] == 1))
        {
            LwU32 instanceId = pEngine->engineData[ENGINE_INFO_TYPE_INST_ID];
            pbdmaIds[0] = pbdmaIds[instanceId];
            pbdmaFaultIds[0] = pbdmaFaultIds[instanceId];
            i = 1;
        }

        pEngine->numPbdmas = i;
        pPbdmaIdsAllocSize = sizeof(*pEngine->pPbdmaIds) * pEngine->numPbdmas;
        if (pPbdmaIdsAllocSize != 0)
        {
            pEngine->pPbdmaIds = malloc(pPbdmaIdsAllocSize);
            pEngine->pPbdmaFaultIds = malloc(pPbdmaIdsAllocSize);
            if (NULL == pEngine->pPbdmaIds || NULL == pEngine->pPbdmaFaultIds)
            {
                free(pEngine->pPbdmaIds);
                free(pEngine->pPbdmaFaultIds);
                return LW_ERR_NO_MEMORY;
            }
            memcpy(pEngine->pPbdmaIds, pbdmaIds, pPbdmaIdsAllocSize);
            memcpy(pEngine->pPbdmaFaultIds, pbdmaFaultIds, pPbdmaIdsAllocSize);
        }
    }

    return LW_OK;
}

/**
 * @brief Parses device info and passes useful information to memory.
 * If it finds an engine which is not registered in engName2DeviceInfo, it will prompt a message.
 *
 * @return
 *  LW_OK:
 *      If everything exelwted correctly.
 *  LW_ERR_NO_MEMORY:
 *      If there is no memory to allocate appropriate structures/arrays.
 *
 * @note This function is version of RM's.
 */

LW_STATUS fifoGetDeviceInfo_GA100(void)
{
    LW_STATUS status = LW_OK;
    LwLength deviceAclwmAllocSize;
    LwU32 *pDeviceAclwm;
    LwU32 deviceAclwmIndex;
    LwU32 rowIdx;
    LwU32 engineIndex;
    LwBool bInChain;
    LwU32 *runListsEngines;
    LwU32 lookup;
    LwU32 maxPbdmaId = 0;
    LwU32 i;
    if (deviceInfo.bInitialized)
    {
        return LW_OK;
    }
    _readDeviceInfoCfg();
    status = deviceInfoAlloc();
    if (LW_OK != status)
    {
        return status;
    }
    runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();
    if (NULL == runListsEngines)
    {
        return LW_ERR_NO_MEMORY;
    }

    deviceAclwmAllocSize = deviceInfo.cfg.maxRowsPerDevice * sizeof(*pDeviceAclwm);
    pDeviceAclwm = malloc(deviceAclwmAllocSize);
    if (NULL == pDeviceAclwm)
    {
        status = LW_ERR_NO_MEMORY;
        return status;
    }
    memset(pDeviceAclwm, 0, deviceAclwmAllocSize);
    bInChain = LW_FALSE;
    deviceAclwmIndex = 0;
    engineIndex = 0;

    for (rowIdx = 0; rowIdx < deviceInfo.cfg.numRows; rowIdx++)
    {
        LwU32 rowValue;
        LwBool bValid;
        LwBool bInChainNew;

        rowValue = GPU_REG_RD32(LW_PTOP_DEVICE_INFO2(rowIdx));
        bValid = !FLD_TEST_DRF(_PTOP, _DEVICE_INFO2, _ROW_VALUE, _ILWALID, rowValue);
        bInChainNew = FLD_TEST_DRF(_PTOP, _DEVICE_INFO2, _ROW_CHAIN, _MORE, rowValue);
        deviceInfo.pRows[rowIdx].value = rowValue;
        deviceInfo.pRows[rowIdx].bValid = bValid;
        deviceInfo.pRows[rowIdx].bInChain = bInChainNew;
        if (!bValid && !bInChain)
        {
            continue;
        }

        pDeviceAclwm[deviceAclwmIndex] = rowValue;
        deviceAclwmIndex++;
        bInChain = bInChainNew;
        if (!bInChain)
        {
            status = pFifo[indexGpu].fifoParseDeviceInfoAndGetEngine(
                                        deviceInfo.pEngines + engineIndex,
                                        pDeviceAclwm, &lookup);

            if (LW_ERR_NOT_SUPPORTED == status)
            {
                dprintf("WARNING: An engine that has no support in lwwatch detected in %s \n", __FUNCTION__);
            }
            else
            {
                if (deviceInfo.pEngines[engineIndex].bHostEng)
                {
                    LwU32 engineRunListId = deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_RUNLIST];
                    if (engineRunListId < pFifo[indexGpu].fifoGetRunlistMaxNumber())
                    {
                        runListsEngines[engineRunListId] |= LWBIT(lookup);
                    }
                    else
                    {
                        dprintf("**ERROR: ENGINE_INFO_TYPE_RUNLIST exceeds %d\n",
                            pFifo[indexGpu].fifoGetRunlistMaxNumber() - 1);
                    }
                }

                for (i = 0; i < deviceInfo.pEngines[engineIndex].numPbdmas; i++)
                {
                    maxPbdmaId = LW_MAX(maxPbdmaId, deviceInfo.pEngines[engineIndex].pPbdmaIds[i]);
                }

                // Append to device map
                //
                engineIndex++;
            }
            deviceAclwmIndex = 0;
            memset(pDeviceAclwm, 0, deviceAclwmAllocSize);
        }
    }
    deviceInfo.enginesCount = engineIndex;
    deviceInfo.maxPbdmas = maxPbdmaId + 1;
    free(pDeviceAclwm);
    deviceInfo.bInitialized = LW_TRUE;

    return LW_OK;
}

LwU32 fifoGetNumChannels_GA100(LwU32 runlistId)
{
    LwU32 offset, mask;
    if (isVirtualWithSriov())
    {
        pFifo[indexGpu].fifoGetChannelOffsetMask(runlistId, &offset, &mask);
        return mask + 1;
    }
    return LW_CHRAM_CHANNEL__SIZE_1;
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
fifoReadRunlistInfo_GA100
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

LW_STATUS
fifoValidateRunlistInfoForAlloc_GA100
(
    LwU32   **ppOutRunlist,
    LwU64     engRunlistPtr,
    LwU32     runlistLength,
    LwU32     tgtAperture,
    readFn_t *pReadFunc
)
{
    if (NULL == ppOutRunlist)
    {
        dprintf("**ERROR: %s:%d: Must pass runlist output.\n", __FILE__, __LINE__);
        return LW_ERR_GENERIC;
    }
    // The ordering is like this because we wanted to avoid printfs of erros when channelram extension is called for runlist with 0 size.
    if (runlistLength == LW_RUNLIST_SUBMIT_LENGTH_ZERO)
    {
        return LW_ERR_GENERIC;
    }

    if (!pFifo[indexGpu].fifoIsRunlistLengthValid(runlistLength))
    {
        dprintf("**ERROR: LW_RUNLIST_SUBMIT_LENGTH Not allocatable\n");
        return LW_ERR_GENERIC;
    }

    if (0ULL == engRunlistPtr)
    {
        dprintf("**ERROR: LW_RUNLIST_SUBMIT_BASE_LO_PTR_LO_NULL and LW_RUNLIST_SUBMIT_BASE_HI_PTR_HI_NULL\n");
        return LW_ERR_GENERIC;
    }

    if (NULL == pReadFunc)
    {
        dprintf("**ERROR: LW_PFIFO_ENG_RUNLIST_ALLOCATOR FUNCTION PTR is NULL\n");
        return LW_ERR_GENERIC;
    }

    switch (tgtAperture)
    {
        case LW_RUNLIST_SUBMIT_BASE_LO_TARGET_VID_MEM :
            *pReadFunc = pFb[indexGpu].fbRead;
        break;

        case LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_COHERENT :
            *pReadFunc = readSystem;
        break;

        case LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_NONCOHERENT :
            *pReadFunc = readSystem;
        break;

        default :
            return LW_ERR_GENERIC;
    }

    return LW_OK;
}


static LW_STATUS _getInstancePointer(ChannelId *pChannelId, LwU64 *pInstancePtr, LwU32 *pTarget)
{
    LwU32 channelId;
    LwU32 *pEntry;
    LwU32 *pRunlistBuffer;
    LwU32 runlistLength = (LwU32)(-1);
    LwU32 i;
    LW_STATUS status;
    LwU32 instancePtrLow;
    LwU32 instancePtrHigh;
    LwU32 instancePtrTarget;
    LwU32 runlistEntrySize = pFifo[indexGpu].fifoRunlistGetEntrySizeLww();
    if ((NULL == pChannelId) || (!pChannelId->bRunListValid) || (NULL == pInstancePtr) || (NULL == pTarget))
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }
    status = pFifo[indexGpu].fifoReadRunlistInfo(pChannelId->runlistId, NULL, &runlistLength, NULL, &pRunlistBuffer);
    if (LW_OK != status)
    {
        return status;
    }
    *pInstancePtr = 0ULL;

    for (i = 0; i < runlistLength; i++)
    {
        pEntry = &pRunlistBuffer[(i * runlistEntrySize)/sizeof(LwU32)];
        if (!FLD_TEST_DRF(_RAMRL, _ENTRY, _TYPE, _CHAN, *pEntry))
        {
            continue;
        }
        channelId = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_CHID, pEntry);
        if (channelId != pFifo[indexGpu].fifoGetSchidForVchid(pChannelId->id))
        {
            continue;
        }
        instancePtrLow = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_INST_PTR_LO, pEntry);
        instancePtrHigh = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_INST_PTR_HI, pEntry);

        // High concatenate low have more than 32 bits, therefore 64-bit colwersion is needed.
        *pInstancePtr =  ((LwU64)instancePtrHigh << DRF_SIZE(LW_RAMRL_ENTRY_CHAN_INST_PTR_LO))
                          | (LwU64)instancePtrLow;
        *pInstancePtr <<= LW_RAMRL_ENTRY_CHAN_INST_PTR_ALIGN_SHIFT;

        instancePtrTarget = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_INST_TARGET, pEntry);
        *pTarget = instancePtrTarget;

    }
    if (runlistLength != LW_RUNLIST_SUBMIT_LENGTH_ZERO)
    {
        free(pRunlistBuffer);
    }
    return LW_OK;
}

/**
 * @brief   Populates array with valid chids of a runlist
 * @param   numChannels Maximum number of channels.
 * @param   pChidArr    Array of valid chids. Must be at least numChannels
 *                      wide.
 * @param   runlistId   Runlist ID
 */
static LW_STATUS
fifoFindChannels
(
    LwU32   *pChidArr,
    LwU32    runlistId,
    LwU32   *chanCnt,
    LwU32    numChannels
)
{
    LwU32 chramPriBase;
    LwU32 i;
    LwU32 chidIdx = 0;

    LwU64 runlistBasePtr;
    LwU32 runlistTgt;
    LwU32 runlistLength;
    LwU32 *pRunlistBuffer = NULL;

    LwU32 *pEntry;
    LwU32 runlistEntrySize;
    LwU32 channelId;

    LW_STATUS status = LW_OK;

    runlistEntrySize = pFifo[indexGpu].fifoRunlistGetEntrySizeLww();
    if (numChannels == 0)
    {
        goto finish;
    }

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, runlistId,
                                                 ENGINE_INFO_TYPE_CHRAM_PRI_BASE, &chramPriBase);
    if (status != LW_OK)
    {
        dprintf("Error: fifoEngineDataXlate failed with status code %d\n", status);
        goto finish;
    }

    status = pFifo[indexGpu].fifoReadRunlistInfo(runlistId, &runlistBasePtr, &runlistLength,
                                                 &runlistTgt, &pRunlistBuffer);

    if ((status != LW_OK) || (runlistLength == LW_RUNLIST_SUBMIT_LENGTH_ZERO))
    {
        goto cleanup;
    }

    for (i = 0; i < runlistLength; i++)
    {
        pEntry = &pRunlistBuffer[(i * runlistEntrySize)/sizeof(LwU32)];
        if (!FLD_TEST_DRF(_RAMRL, _ENTRY, _TYPE, _CHAN, *pEntry))
        {
            continue;
        }

        if (chidIdx == numChannels)
        {
            dprintf("Warning: More channels found than expected. Channel list is truncated to %d channels.\n", numChannels);
            break;
        }
        channelId = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_CHID, pEntry);
        pChidArr[chidIdx] = channelId;

        chidIdx++;
    }

cleanup:
    free(pRunlistBuffer);

finish:
    *chanCnt = chidIdx;
    return status;
}

/**
 *  bChramPriBaseValid = LW_TRUE when we pass chramPriBase for caching purposes.
 *  bRunListValid = LW_TRUE the function will return a pointer to the instance block of a runlist whose id is passed.
 *  If we do not pass runlistId and we do not passs chramPriBase, then the address
 *  of channel id will be absolute.
 */
LW_STATUS
fifoGetChannelInstForChid_GA100
(
    ChannelId   *pChannelId,
    ChannelInst *pChannelInst
)
{
    LwU64 instancePtr = 0ULL;
    LwU32 regChannel;
    LwU32 chramPriBase = 0;
    LwU32 instTarget = LW_U32_MAX;
    LwU32 instPtrTarget = LW_U32_MAX;
    LW_STATUS status;

    if ((pChannelInst == NULL) || (pChannelId == NULL))
    {
        dprintf("**ERROR: Invalid argument for %s\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (pChannelId->bChramPriBaseValid)
    {
        chramPriBase = pChannelId->chramPriBase;
    }
    else if (pChannelId->bRunListValid)
    {
        status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, pChannelId->runlistId,
                                                     ENGINE_INFO_TYPE_CHRAM_PRI_BASE, &chramPriBase);
        if (LW_OK != status)
        {
            return status;
        }
    }
    if (pChannelId->bRunListValid)
    {
        status = pFifo[indexGpu].fifoReadRunlistInfo(pChannelId->runlistId, NULL,
                                                     NULL, &instTarget, NULL);
        if (LW_OK != status)
        {
            return status;
        }
        status = _getInstancePointer(pChannelId, &instancePtr, &instPtrTarget);
        if (LW_OK != status)
        {
            return status;
        }
    }

    regChannel = GPU_REG_RD32(chramPriBase + LW_CHRAM_CHANNEL(pFifo[indexGpu].fifoGetSchidForVchid(pChannelId->id)));
    pChannelInst->regChannel = regChannel;
    pChannelInst->instPtr = instancePtr;
    pChannelInst->target = instPtrTarget;

    pChannelInst->state = 0;
    pChannelInst->state |= FLD_TEST_DRF(_CHRAM, _CHANNEL, _BUSY, _TRUE, regChannel) ?
                            LW_PFIFO_CHANNEL_STATE_BUSY : 0;

    pChannelInst->state |= FLD_TEST_DRF(_CHRAM, _CHANNEL, _ENABLE, _IN_USE, regChannel) ?
                            LW_PFIFO_CHANNEL_STATE_ENABLE : 0;

    pChannelInst->state |= FLD_TEST_DRF(_CHRAM, _CHANNEL, _ACQUIRE_FAIL, _TRUE, regChannel) ?
                        LW_PFIFO_CHANNEL_STATE_ACQ_PENDING : 0;

    pChannelInst->state |= FLD_TEST_DRF(_CHRAM, _CHANNEL, _PENDING, _TRUE, regChannel) ?
                            LW_PFIFO_CHANNEL_STATE_PENDING : 0;

    return LW_OK;
}

LW_STATUS fifoDumpEschedChannelRamRegsById_GA100(LwU32 runlistId)
{
    LwU32 displayCount;
    LwU32 chid;
    LwU32 ctr;
    LwU32 chanCnt;
    LwU32 numChannels;
    LwU32 chramPriBase;
    LwU32 runlistTgt;
    BOOL bIsBusy;
    ChannelInst channelInst;
    ChannelId channelId;
    LwU64 runlistBasePtr;
    LwU32 *pChidArr;
    char *aperture;
    LW_STATUS status;

    targetMem[LW_RUNLIST_SUBMIT_BASE_LO_TARGET_VID_MEM].strName = "VID";
    targetMem[LW_RUNLIST_SUBMIT_BASE_LO_TARGET_VID_MEM].value =
        LW_RUNLIST_SUBMIT_BASE_LO_TARGET_VID_MEM;

    targetMem[LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_COHERENT].strName = "SYS_CO";
    targetMem[LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_COHERENT].value =
        LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_COHERENT;

    targetMem[LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_NONCOHERENT].strName = "SYS_NON";
    targetMem[LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_NONCOHERENT].value =
        LW_RUNLIST_SUBMIT_BASE_LO_TARGET_SYS_MEM_NONCOHERENT;

    numChannels = pFifo[indexGpu].fifoGetNumChannels(runlistId);
    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, runlistId,
                                                 ENGINE_INFO_TYPE_CHRAM_PRI_BASE, &chramPriBase);
    if (LW_OK != status)
    {
        return status;
    }

    status = pFifo[indexGpu].fifoReadRunlistInfo(runlistId, &runlistBasePtr, NULL,
                                                 &runlistTgt, NULL);
    if (LW_OK != status)
    {
        return status;
    }
    if (targetMem[runlistTgt].strName == NULL)
    {
        targetMem[runlistTgt].strName = "?";
    }
    dprintf(" ChramPriBase: " LwU64_FMT "\n", (LwU64)chramPriBase);
    dprintf(" Runlist base pointer: " LwU64_FMT "\n", runlistBasePtr);
    dprintf(" Target: %s\n", targetMem[runlistTgt].strName);

    channelId.bChramPriBaseValid  = LW_TRUE;
    channelId.chramPriBase = chramPriBase;
    channelId.bRunListValid = LW_TRUE;
    channelId.runlistId = runlistId;

    pChidArr = calloc(numChannels, sizeof(LwU32));
    if (pChidArr == NULL)
    {
        status = LW_ERR_NO_MEMORY;
        goto cleanup;
    }

    status = fifoFindChannels(pChidArr, runlistId, &chanCnt, numChannels);

    if (status != LW_OK)
    {
        goto cleanup;
    }

    ctr = 0;
    displayCount = 0;

    for (ctr = 0; ctr < chanCnt; ctr++)
    {
        chid = pChidArr[ctr];
        channelId.id = chid;

        status = pFifo[indexGpu].fifoGetChannelInstForChid(&channelId, &channelInst);
        if (LW_OK != status)
        {
            dprintf("Error: could not get channel instance pointer\n");
            continue;
        }
        if (!(channelInst.state & LW_PFIFO_CHANNEL_STATE_ENABLE))
        {
            ctr++;
            continue;
        }

        if ((displayCount % 10) == 0)
        {
            dprintf("\n");
            dprintf("%-4s %-18s %-18s %-12s %-30s\n",
                    "chId", "Instance Ptr", "Inst Ptr Target", "BUSY", "STATUS");
            dprintf("%-4s %-18s %-18s %-12s %-30s\n", "----", "---------------", "---------------", "----", "------");
        }

        bIsBusy = (channelInst.state & LW_PFIFO_CHANNEL_STATE_BUSY) != 0;

        switch (channelInst.target)
        {
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_VID_MEM:
                aperture = "(video)";
                break;
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_COHERENT:
                aperture = "(syscoh)";
                break;
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_NONCOHERENT:
                aperture = "(sysnon)";
                break;
            default:
                aperture = "(?)";
        }

        dprintf("%-4d " LwU64_FMT "    %-14s" " %-12s",
                chid, channelInst.instPtr, aperture,
                (bIsBusy ? "TRUE" : "FALSE"));

        dprintf("(");
        if (FLD_TEST_DRF(_CHRAM, _CHANNEL, _PENDING, _TRUE, channelInst.regChannel))
            dprintf("PENDING ");
        if (FLD_TEST_DRF(_CHRAM, _CHANNEL, _CTX_RELOAD, _TRUE, channelInst.regChannel))
            dprintf("CTX_RELOAD ");
        if (FLD_TEST_DRF(_CHRAM, _CHANNEL, _PBDMA_BUSY, _TRUE, channelInst.regChannel))
            dprintf("PBDMA_BUSY ");
        if (FLD_TEST_DRF(_CHRAM, _CHANNEL, _ENG_BUSY, _TRUE, channelInst.regChannel))
            dprintf("ENG_BUSY ");
        if (FLD_TEST_DRF(_CHRAM, _CHANNEL, _ACQUIRE_FAIL, _TRUE, channelInst.regChannel))
            dprintf("ACQUIRE_FAIL ");
        dprintf(")\n");

        displayCount++;

        ctr++;
        chid = pChidArr[ctr];
    }

    if (displayCount == 0)
    {
        dprintf("Could not find channels IN_USE\n");
    }

cleanup:
    free(pChidArr);
    return status;
}

static void _printRunlistEngines(LwU32 runlistId)
{
    LwU32 mask;
    LwU32 j;
    LwU32 *runListsEngines;
    EngineNameValue *engNames;

    runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();
    if (NULL == runListsEngines)
    {
        dprintf("**ERROR: Device info is not called %s\n", __FUNCTION__);
        return;
    }
    mask = runListsEngines[runlistId];
    if (mask == 0)
    {
        return;
    }

    engNames = (EngineNameValue*)pFifo[indexGpu].fifoGetEngNames();
    dprintf("\n");
    dprintf(" Runlist %d\n", runlistId);
    dprintf(" Engine(s):");
    for (j = 0; mask != 0; j++, mask >>= 1)
    {
        if ((mask & 0x1) != 0)
        {
            dprintf("%s", engNames[j].nameValue.strName);
        }
    }
    dprintf("\n");

}

/**
 * Dumps the channel ram register info for active chid for the specified esched.
 */
static LW_STATUS _dumpRunlistById(LwU32 runlistId)
{
    LwU32 *runListsEngines;

    runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();
    if (pFifo[indexGpu].fifoGetRunlistMaxNumber() <= runlistId)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (runListsEngines[runlistId] == 0)
    {
        return LW_OK;
    }

    _printRunlistEngines(runlistId);
    pFifo[indexGpu].fifoDumpEschedChannelRamRegsById(runlistId);
    return LW_OK;
}

/**
 * Dumps the channel ram register info for active chid across all escheds if channelId = -1,
 * otherwise it dumps only for the specified esched.
 */
LW_STATUS fifoDumpChannelRamRegs_GA100(LwS32 runlistIdPar)
{
    LwU32 runlistId;
    LW_STATUS status;

    status = pFifo[indexGpu].fifoGetDeviceInfo();
    if (LW_OK != status)
    {
        return status;
    }

    if (runlistIdPar == -1)
    {
        for (runlistId = 0; runlistId < pFifo[indexGpu].fifoGetRunlistMaxNumber(); runlistId++)
        {
            _dumpRunlistById(runlistId);
        }
    }
    else
    {
        status = _dumpRunlistById((LwU32)runlistIdPar);
        if (LW_OK != status)
        {
            dprintf("***ERROR runlist is not defined in device info or it does not exist\n");
            return status;
        }
    }
    return LW_OK;
}

/*!
 * Fetches the runlist for the specified runlistId.
 */
LW_STATUS fifoDumpEngRunlistById_GA100(LwU32 runlistId)
{
    LwU32     runlistLength = 0;
    LwU32     tgtAperture   = 0;
    LwU64     engRunlistPtr = 0;
    LwU32    *runlistBuffer = NULL;
    LwU32     chramPriBase;
    LwU32     i;
    LW_STATUS status;
    LwU32     rlEntrySize = pFifo[indexGpu].fifoRunlistGetEntrySizeLww();
    BOOL      printRawChram = TRUE;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, runlistId,
                                                 ENGINE_INFO_TYPE_CHRAM_PRI_BASE, &chramPriBase);
    if (LW_OK != status)
    {
        return status;
    }

    status = pFifo[indexGpu].fifoReadRunlistInfo(runlistId, &engRunlistPtr, &runlistLength,
                                                 &tgtAperture, &runlistBuffer);
    if (LW_OK != status)
    {
        dprintf("**ERROR: Could not read runlist info\n");
        return status;
    }

    dprintf("  Base Addr             : " LwU64_FMT "\n", engRunlistPtr);
    dprintf("  Length                : 0x%x (%d)\n", runlistLength, runlistLength);
    dprintf("  ChannelRam PRI Base   : 0x%x\n", chramPriBase);
    dprintf("                                                                              "); pFifo[indexGpu].fifoPrintChramStatusHeader(0, printRawChram); dprintf("\n");
    dprintf("                                                                              "); pFifo[indexGpu].fifoPrintChramStatusHeader(1, printRawChram); dprintf("\n");
    dprintf("                                                                              "); pFifo[indexGpu].fifoPrintChramStatusHeader(2, printRawChram); dprintf("\n");
    dprintf("                                                                              "); pFifo[indexGpu].fifoPrintChramStatusHeader(3, printRawChram); dprintf("\n");
    dprintf(" Entry                 RunQ                                                   "); pFifo[indexGpu].fifoPrintChramStatusHeader(4, printRawChram); dprintf("\n");
    dprintf("  |   TYPE ID           | Len Instance Pointer        USERD                   "); pFifo[indexGpu].fifoPrintChramStatusHeader(5, printRawChram); dprintf("\n");
    dprintf(" ---- ---- ---          - --- ----------------        -----                   "); pFifo[indexGpu].fifoPrintChramStatusHeader(6, printRawChram); dprintf("\n");

    for (i = 0; i < runlistLength; i++)
    {
        LwU32 *entry = &runlistBuffer[(i * rlEntrySize)/sizeof(LwU32)];

        dprintf(" %4d", i);

        pFifo[indexGpu].fifoPrintRunlistEntry(chramPriBase, entry);
        dprintf("\n");
    }

    free(runlistBuffer);
    return LW_OK;
}

void fifoPrintChramStatusHeader_GA100(LwU32 line, BOOL printRaw)
{
    switch(line)
    {
        case 0: dprintf("Enabled     OnPBDMA      "); break;
        case 1: dprintf("|SemAcqFail |PBDMAFault  "); break;
        case 2: dprintf("||CtxReload ||PBDMABusy  "); break;
        case 3: dprintf("|||Pending  |||OnEng     "); break;
        case 4: dprintf("||||Busy    ||||EngFault "); break;
        case 5: dprintf("|||||Next   |||||EngBusy "); break;
        case 6: dprintf("------      ------       "); break;
    }

    if (printRaw)
    {
        switch (line) {
            case 5: dprintf("Raw CHRAM  "); break;
            case 6: dprintf("---------  "); break;
            default:
                dprintf("%10s ", "");
                break;
        }
    }
}

const char* _boolToChramStatusStr(BOOL val)
{
    return val ? "Y" : "n";
}

void fifoPrintChramStatusDecode_GA100(LwU32 regChannel, BOOL invalid, BOOL printRaw)
{
    BOOL bIsBusy, bIsEnabled, bIsPending, bIsOnPBDMA;
    BOOL bIsOnENG, bIsNext, bIsCtxReload, bIsSemAcqFail;
    BOOL bIsPBDMABusy, bIsEngBusy, bIsPBDMAFault, bIsEngFault;

    bIsBusy    = DRF_VAL(_CHRAM, _CHANNEL, _BUSY, regChannel);
    bIsEnabled = DRF_VAL(_CHRAM, _CHANNEL, _ENABLE, regChannel);
    bIsPending = DRF_VAL(_CHRAM, _CHANNEL, _PENDING, regChannel);
    bIsOnPBDMA = DRF_VAL(_CHRAM, _CHANNEL, _ON_PBDMA, regChannel);

    bIsOnENG      = DRF_VAL(_CHRAM, _CHANNEL, _ON_ENG, regChannel);
    bIsNext       = DRF_VAL(_CHRAM, _CHANNEL, _NEXT, regChannel);
    bIsCtxReload  = DRF_VAL(_CHRAM, _CHANNEL, _CTX_RELOAD, regChannel);
    bIsSemAcqFail = DRF_VAL(_CHRAM, _CHANNEL, _ACQUIRE_FAIL, regChannel);

    bIsPBDMABusy  = DRF_VAL(_CHRAM, _CHANNEL, _PBDMA_BUSY, regChannel);
    bIsEngBusy    = DRF_VAL(_CHRAM, _CHANNEL, _ENG_BUSY, regChannel);
    bIsPBDMAFault = DRF_VAL(_CHRAM, _CHANNEL, _PBDMA_FAULTED, regChannel);
    bIsEngFault   = DRF_VAL(_CHRAM, _CHANNEL, _ENG_FAULTED, regChannel);

    if (invalid) {
        dprintf("------      ------       ");
        if (printRaw) {
            dprintf("%-10s ", "-");
        }
    } else {
        dprintf("%s%s%s%s%s%s      %s%s%s%s%s%s       ",
                _boolToChramStatusStr(bIsEnabled), _boolToChramStatusStr(bIsSemAcqFail), _boolToChramStatusStr(bIsCtxReload),
                _boolToChramStatusStr(bIsPending), _boolToChramStatusStr(bIsBusy), _boolToChramStatusStr(bIsNext),
                _boolToChramStatusStr(bIsOnPBDMA), _boolToChramStatusStr(bIsPBDMAFault), _boolToChramStatusStr(bIsPBDMABusy),
                _boolToChramStatusStr(bIsOnENG), _boolToChramStatusStr(bIsEngFault), _boolToChramStatusStr(bIsEngBusy));
        if (printRaw) {
            dprintf("0x%-8x ", regChannel);
        }
    }
}

void fifoPrintRunlistEntry_GA100(LwU32 chramPriBase, LwU32 *entry)
{
    BOOL printRawChram = TRUE;

    if (!entry)
    {
        dprintf("Invalid entry");
        return;
    }

    if(DRF_VAL(_RAMRL, _ENTRY, _TYPE, entry[0]) == LW_RAMRL_ENTRY_TYPE_TSG)
    {
        LwU32 tsgId = pFifo[indexGpu].fifoGetTsgIdFromRunlistEntry(entry);
        LwU32 tsgLen = DRF_VAL(_RAMRL, _ENTRY_TSG, _LENGTH, entry[1]);
        {
            char tsgIdDecStr[8];
            char tsgIdStr[32];

            // Maintain the fields to be seprated by space so the output can be parsed easier
            sprintf(tsgIdDecStr, "(%d)", tsgId);
            sprintf(tsgIdStr, "0x%-3x %-6s", tsgId, tsgIdDecStr);

            dprintf(" TSG  %12s - %-3d %-23s %-23s ",
                    tsgIdStr, tsgLen, "-", "-");
            pFifo[indexGpu].fifoPrintChramStatusDecode(0, TRUE, printRawChram);
        }
    }
    else
    {
        LwU32 chanId   = DRF_VAL(_RAMRL, _ENTRY_CHAN, _CHID, entry[2]);
        LwU32 runqueue = DRF_VAL(_RAMRL, _ENTRY_CHAN, _RUNQUEUE_SELECTOR, entry[0]);

        LwU32 instancePtrTarget = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_INST_TARGET, entry);
        LwU32 instancePtrLow = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_INST_PTR_LO, entry);
        LwU32 instancePtrHigh = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_INST_PTR_HI, entry);
        LwU64 instancePtr = (((LwU64)instancePtrHigh << DRF_SIZE(LW_RAMRL_ENTRY_CHAN_INST_PTR_LO))
                             | (LwU64)instancePtrLow) << LW_RAMRL_ENTRY_CHAN_INST_PTR_ALIGN_SHIFT;
        const char* instancePtrTargetStr = NULL;

        LwU32 userdTarget = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_USERD_TARGET, entry);
        LwU32 userdPtrLow = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_USERD_PTR_LO, entry);
        LwU32 userdPtrHigh = SF_ARR32_VAL(_RAMRL, _ENTRY_CHAN_USERD_PTR_HI, entry);
        LwU64 userdPtr = (((LwU64)userdPtrHigh << DRF_SIZE(LW_RAMRL_ENTRY_CHAN_USERD_PTR_LO))
                          | (LwU64)userdPtrLow);
        const char* userdTargetStr = NULL;

        LwU32 regChannel;

        regChannel = GPU_REG_RD32(chramPriBase + LW_CHRAM_CHANNEL(chanId));

        switch (instancePtrTarget)
        {
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_VID_MEM:
                instancePtrTargetStr = "(video)";
                break;
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_COHERENT:
                instancePtrTargetStr = "(syscoh)";
                break;
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_NONCOHERENT:
                instancePtrTargetStr = "(sysnon)";
                break;
        }

        switch (userdTarget)
        {
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_VID_MEM:
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_VID_MEM_LWLINK_COHERENT:
                userdTargetStr = "(video)";
                break;
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_SYS_MEM_COHERENT:
                userdTargetStr = "(syscoh)";
                break;
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_SYS_MEM_NONCOHERENT:
                userdTargetStr = "(sysnon)";
                break;
        }

        {
            char channelIdDecStr[8];
            char channelIdStr[32];

            // Maintain the fields to be seprated by space so the output can be parsed easier
            sprintf(channelIdDecStr, "(%d)", chanId);
            sprintf(channelIdStr, "0x%-3x %-6s", chanId, channelIdDecStr);

            dprintf(" CHAN %12s %d -   0x%012llx %-8s 0x%012llx %-8s ",
                    channelIdStr, runqueue, instancePtr, instancePtrTargetStr, userdPtr, userdTargetStr);

            pFifo[indexGpu].fifoPrintChramStatusDecode(regChannel, FALSE, printRawChram);
        }
    }
}

/**
 * fifoGetInfo_GA100
 *
 * This function dumps all the information gathered
 * about the fifo.
 */
LW_STATUS fifoGetInfo_GA100(void)
{
    LW_STATUS status;
    LwU32 i;
    LwU32 buf;
    LwU64 pde;
    LwU64 instMemAddr;
    LwU32 runlistId;
    readFn_t readFn = NULL;
    LwU32 *runListsEngines;
    ChannelId channelId;
    ChannelInst channelInst;
    status = pFifo[indexGpu].fifoGetDeviceInfo();
    if (LW_OK != status)
    {
        return status;
    }
    runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();

    // Channel state
    channelId.bRunListValid = LW_TRUE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    for (runlistId = 0; runlistId < pFifo[indexGpu].fifoGetRunlistMaxNumber(); runlistId++)
    {
        LwU32 numChannels;
        if (runListsEngines[runlistId] == 0)
        {
            continue;
        }
        _printRunlistEngines(runlistId);
        channelId.runlistId = runlistId;
        numChannels = pFifo[indexGpu].fifoGetNumChannels(runlistId);
        for (i = 0; i < numChannels; i++)
        {
            channelId.id = i;
            status = pFifo[indexGpu].fifoGetChannelInstForChid(&channelId, &channelInst);
            if (status == LW_OK && channelInst.state & LW_PFIFO_CHANNEL_STATE_ENABLE)
            {
                dprintf("CHANNEL:            %03d\n", i);
                if (verboseLevel > 0)
                {
                    // get instance memory address for this channel
                    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(&channelId, &readFn, NULL, NULL);
                    if (readFn == NULL)
                    {
                        dprintf("**ERROR: NULL value of readFn.\n");
                        return LW_ERR_NOT_SUPPORTED;
                    }

                    // print some info for the PBDMA unit for this channel
                    // read from RAMFC
                    pFifo[indexGpu].fifoGetGpInfoByChId(&channelId);

                    // find channel page directory base address
                    pde = pMmu[indexGpu].mmuGetPDETableStartAddress(instMemAddr, readFn);
                    dprintf(" + PDB ADDRESS:                        " LwU64_FMT "\n", pde);

                    // check where the PD lives
                    readFn(instMemAddr + SF_OFFSET(LW_RAMIN_PAGE_DIR_BASE_TARGET), &buf, 4);
                    if(SF_VAL(_RAMIN, _PAGE_DIR_BASE_TARGET, buf) == LW_RAMIN_PAGE_DIR_BASE_TARGET_VID_MEM)
                        dprintf(" + PDB APERTURE:                       VID_MEM\n");
                    else
                        if(SF_VAL(_RAMIN, _PAGE_DIR_BASE_TARGET, buf) ==
                            LW_RAMIN_PAGE_DIR_BASE_TARGET_SYS_MEM_COHERENT)
                            dprintf(" + PDB APERTURE:                       SYS_MEM_COHERENT\n");
                        else
                            dprintf(" + PDB APERTURE:                       SYS_MEM_NONCOHERENT\n");

                    pFifo[indexGpu].fifoDumpSubctxInfo(&channelId);

                    // get the engine states
                    pFifo[indexGpu].fifoDumpEngStates(&channelId, NULL);
                    dprintf("\n");
                }
            }
        }

        // Check for interrupts
        dprintf("\nChecking pending interrupts...\n");

        if (checkFifoIntrInfo_GA100(runlistId))
        {
            getFifoIntrInfo_GA100(runlistId);
        }
        else
        {
            dprintf("No interrupts are pending\n");
        }
    }
    dprintf("\n");

    pFifo[indexGpu].fifoGetPbdmaState();
    dprintf("\n");

    return LW_OK;
}

/**
 * fifoTestHostState_GF100
 *
 * Check if channels are idle
 * for a given channel :
 *     1. Check for DMA pending
 *     2. Check for interrupts
 *     3. Show engine states
 *     4. Show semaphore info.
 */

LW_STATUS fifoTestHostState_GA100(void)
{
    LW_STATUS status = LW_OK;
    LwU32 chid;
    LwBool bIsAcquirePending = LW_FALSE;
    ChannelId channelId;
    ChannelInst channelInst;
    LwU32 runlistId;
    LwU32 *runListsEngines;
    status = pFifo[indexGpu].fifoGetDeviceInfo();
    if (LW_OK != status)
    {
        return status;
    }
    runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();

    // @TODO: add check and print whether the all channels are idle.
    // Maybe even replace static allChannelsIdlefrom fifogf100.c with hal function.

    dprintf("\nChecking pending channels...\n");

    channelId.bRunListValid = LW_TRUE;
    channelId.bChramPriBaseValid  = LW_FALSE;

    for (runlistId = 0; runlistId < pFifo[indexGpu].fifoGetRunlistMaxNumber(); runlistId++)
    {
        LwU32 numChannels;
        if (runListsEngines[runlistId] == 0)
        {
            continue;
        }
        _printRunlistEngines(runlistId);

        channelId.runlistId = runlistId;
        numChannels = pFifo[indexGpu].fifoGetNumChannels(runlistId);
        for (chid = 0; chid < numChannels; chid++)
        {
            channelId.id = chid;
            pFifo[indexGpu].fifoGetChannelInstForChid(&channelId, &channelInst);

            if (channelInst.state & LW_PFIFO_CHANNEL_STATE_ENABLE)
            {
                dprintf("\nChannel #%03d:              0x%08x\n",
                    chid, channelInst.regChannel);

                //1. check if pending
                if (channelInst.state & LW_PFIFO_CHANNEL_STATE_PENDING)
                {
                    dprintf("LW_PFIFO_CHANNEL_PENDING_TRUE\n");
                }
            }
        }

        //2. Check for interrupts
        dprintf("\nChecking pending interrupts...\n");

        if (checkFifoIntrInfo_GA100(runlistId))
        {
            getFifoIntrInfo_GA100(runlistId);
        }
        else
        {
            dprintf("No interrupts are pending\n");
        }
    }

    // @TODO: Add version for ampere of this function.
    //3. Engine state
    dprintf("\nDumping engine states...\n");
    pFifo[indexGpu].fifoCheckEngStates(&gpuState);

    //4. Semaphores
    dprintf("\nChecking semaphores state...\n");
    for (runlistId = 0; runlistId < pFifo[indexGpu].fifoGetRunlistMaxNumber(); runlistId++)
    {
        LwU32 numChannels;
        if (runListsEngines[runlistId] == 0)
        {
            continue;
        }
        _printRunlistEngines(runlistId);
        channelId.runlistId = runlistId;
        numChannels = pFifo[indexGpu].fifoGetNumChannels(runlistId);
        for (chid = 0; chid < numChannels; chid++)
        {
            channelId.id = chid;
            pFifo[indexGpu].fifoGetChannelInstForChid(&channelId, &channelInst);

            if ((channelInst.state & LW_PFIFO_CHANNEL_STATE_ENABLE) &&
                (channelInst.state & LW_PFIFO_CHANNEL_STATE_ACQ_PENDING))
            {
                dprintf("**ERROR: CHANNEL_ACQUIRE_PENDING_ON set on channel #%03d\n", chid);
                addUnitErr("\t CHANNEL_ACQUIRE_PENDING_ON set on channel #%03d\n", chid);

                bIsAcquirePending = LW_TRUE;
                status = LW_ERR_GENERIC;
            }
        }
    }

    if (!bIsAcquirePending)
        dprintf("No semaphore acquire pending on any channel.\n");

    return status;
}

/*!
 * @return The number of PBDMAs provided by the chip.
 */
LwU32 fifoGetNumPbdma_GA100(void)
{
    // Force read fifoGetDeviceInfo to initialize maxPbdmas
    LW_STATUS status = pFifo[indexGpu].fifoGetDeviceInfo();
    if (status != LW_OK)
    {
        dprintf("**ERROR: Unable to initialize device info. numPbdmas will be inaclwrate.\n");
        assert(status == LW_OK);
    }

    return deviceInfo.maxPbdmas;
}
