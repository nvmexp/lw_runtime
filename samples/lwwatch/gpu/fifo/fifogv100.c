/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015-2022 by LWPU Corporation.  All rights reserved.  All
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
#include "mmu.h"
#include "vmem.h"
#include "gr.h"
#include "volta/gv100/dev_fifo.h"
#include "volta/gv100/dev_ram.h"
#include "volta/gv100/dev_pbdma.h"
#include "volta/gv100/dev_top.h"

/// @brief This macro is needed here to parse multi-word/array defines from HW
#define SF_ARR32_VAL(s,f,arr) \
    (((arr)[SF_INDEX(LW ## s ## f)] >> SF_SHIFT(LW ## s ## f)) & SF_MASK(LW ## s ## f))

//
// Map the engine names, from projects.spec, to the values in
// LW_PTOP_DEVICE_INFO_TYPE_ENUM.
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
    {{"CE6",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 6, ENGINE_TAG_CE},      // bit 11 in runlist2EngMask
    {{"CE7",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 7, ENGINE_TAG_CE},      // bit 12 in runlist2EngMask
    {{"CE8",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE     }, 8, ENGINE_TAG_CE},      // bit 13 in runlist2EngMask
    {{"IOCTRL", LW_PTOP_DEVICE_INFO_TYPE_ENUM_IOCTRL  }, 0, ENGINE_TAG_UNKNOWN}, // bit 14 in runlist2EngMask
    {{"",       DEVICE_INFO_TYPE_ILWALID              }, 0, ENGINE_TAG_ILWALID}
};

void * fifoGetEngNames_GV100(void)
{
    return engName2DeviceInfo;
}

LwU32 fifoGetNumEng_GV100(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}

/*!
 * Dumps the Subcontext info
 */
void fifoDumpSubctxInfo_GV100(ChannelId *pChannelId)
{
    LwU32    buf, veid;
    LwU64    instMemAddr;
    readFn_t readFn = NULL;

    // get instance memory address for this channel
    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId, &readFn, NULL, NULL);
    if (readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return;
    }

    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_SET_CHANNEL_INFO), &buf, 4);
    veid = DRF_VAL(_PPBDMA, _SET_CHANNEL_INFO, _VEID, buf);
    dprintf(" + VEID:                               %d\n", veid);
}

/**
 * Get all the entries from device info
 */
LwU32 fifoRunlistGetEntrySizeLww_GV100(void)
{
    return LW_RAMRL_ENTRY_SIZE;
}


LwBool fifoIsChidInEntry_GV100(LwU32* entry, LwU32 chid)
{
    if(entry == NULL)
    {
        return FALSE;
    }

    return (chid == DRF_VAL(_RAMRL, _ENTRY, _CHAN_CHID, entry[0]));
}

void fifoPrintRunlistEntry_GV100(LwU32 unused, LwU32 *entry)
{
    if (!entry)
    {
        dprintf("Invalid entry");
        return;
    }

    if(DRF_VAL(_RAMRL, _ENTRY, _TYPE, entry[0]) == LW_RAMRL_ENTRY_TYPE_TSG)
    {
        LwU32 tsgId = DRF_VAL(_RAMRL, _ENTRY_TSG, _TSGID, entry[2]);
        LwU32 tsgLen = DRF_VAL(_RAMRL, _ENTRY_TSG, _LENGTH, entry[1]);
        dprintf("TSG         0x%-3x(%4d)", tsgId, tsgId);
        dprintf("       Len =0x%-3x", tsgLen);
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

        switch (instancePtrTarget)
        {
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_VID_MEM:
                instancePtrTargetStr = "video";
                break;
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_COHERENT:
                instancePtrTargetStr = "syscoh";
                break;
            case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_NONCOHERENT:
                instancePtrTargetStr = "sysnoncoh";
                break;
        }

        switch (userdTarget)
        {
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_VID_MEM:
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_VID_MEM_LWLINK_COHERENT:
                userdTargetStr = "video";
                break;
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_SYS_MEM_COHERENT:
                userdTargetStr = "syscoh";
                break;
            case LW_RAMRL_ENTRY_CHAN_USERD_TARGET_SYS_MEM_NONCOHERENT:
                userdTargetStr = "sysnoncoh";
                break;
        }

        dprintf("CHAN        0x%-3x(%4d)",chanId, chanId);
        dprintf("       RunQ=0x%-3x InstPtr=0x%016llx (%s) USERD=0x%llx (%s)",
                runqueue, instancePtr, instancePtrTargetStr, userdPtr, userdTargetStr);
    }
}

//-----------------------------------------------------
// fifoDumpFifoIntrInfo_GV100
//
// This function dumps information about various pending
// interrupts based on interrupt mask.
//-----------------------------------------------------
void fifoDumpFifoIntrInfo_GV100(LwU32 runlistId, LwU32 intr, LwU32 intrMsk)
{
    LwU32 regRead;

    // Unused pre-AMPERE
    (void) runlistId;

    intr &= intrMsk;

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _BIND_ERROR, _PENDING))
    {
        regRead = GPU_REG_RD32(LW_PFIFO_INTR_BIND_ERROR);
        dprintf(" + LW_PFIFO_INTR_BIND_ERROR              : 0x%08x\n", regRead);
        priv_dump("LW_PFIFO_INTR_BIND_ERROR");
    }

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _SCHED_ERROR, _PENDING))
    {
        regRead = GPU_REG_RD32(LW_PFIFO_INTR_SCHED_ERROR);
        dprintf(" + LW_PFIFO_INTR_SCHED_ERROR              : 0x%08x\n", regRead);
        priv_dump("LW_PFIFO_INTR_SCHED_ERROR");
    }

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _CHSW_ERROR, _PENDING))
    {
        regRead = GPU_REG_RD32(LW_PFIFO_INTR_CHSW_ERROR);
        dprintf(" + LW_PFIFO_INTR_CHSW_ERROR              : 0x%08x\n", regRead);
        priv_dump("LW_PFIFO_INTR_CHSW_ERROR");
    }

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _LB_ERROR, _PENDING))
    {
        regRead = GPU_REG_RD32(LW_PFIFO_INTR_LB_ERROR);
        dprintf(" + LW_PFIFO_INTR_LB_ERROR              : 0x%08x\n", regRead);
        priv_dump("LW_PFIFO_INTR_LB_ERROR");
    }

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _RUNLIST_EVENT, _PENDING))
    {
        dprintf(" + LW_PFIFO_INTR_0_RUNLIST_EVENT_PENDING\n");
    }

    // Dump PBDMA info
    if (intr & DRF_DEF(_PFIFO, _INTR_0, _PBDMA_INTR, _PENDING))
    {
        pFifo[indexGpu].fifoGetIntrPbdmaInfo();
    }
}


/*!
 * Decodes and prints a register using priv_dump.
 */
#define PRINT_PBDMA_REG_PD(d, r, nPbdma)        do{ \
    sprintf(buffer, "LW%s%s(%d)", #d, #r, (LwU32)nPbdma); \
    dprintf("%s\n", buffer); \
    priv_dump(buffer); \
}while(0)

//-----------------------------------------------------
//
// Dumps the pbdma registers for the given pbdma id.
//-----------------------------------------------------
void fifoDumpPbdmaRegs_GV100(LwU32 pbdmaId)
{
    char buffer[FIFO_REG_NAME_BUFFER_LEN];
    const LwU32 num_pbdma = pFifo[indexGpu].fifoGetNumPbdma();

    buffer[0] = '\0';

    if (pbdmaId >= num_pbdma )
    {
        dprintf("ERROR: pbdmaId >= LW_HOST_NUM_PBDMA (%d)\n", num_pbdma);
        return;
    }

    PRINT_PBDMA_REG_PD( _PPBDMA, _TARGET, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _STATUS, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _CONTROL, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _CHANNEL, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _GP_BASE, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _GP_BASE_HI, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _GP_FETCH, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _PB_FETCH, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _PB_FETCH_HI, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _GP_GET, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _GP_PUT, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _GET, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _GET_HI, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _PUT, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _REF, pbdmaId );

    PRINT_PBDMA_REG_PD( _PPBDMA, _SEM_ADDR_HI, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _SEM_ADDR_LO, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _SEM_PAYLOAD_HI, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _SEM_PAYLOAD_LO, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _SEM_EXELWTE, pbdmaId );

    //LW_PPBDMA CACHE1
    pFifo[indexGpu].fifoDumpPbdmaRegsCache1(pbdmaId);

}

/*!
 * @return TSGID from provided runlist entry.
 */
LwU32 fifoGetTsgIdFromRunlistEntry_GV100(LwU32 *entry)
{
    return DRF_VAL(_RAMRL, _ENTRY_TSG, _TSGID, entry[2]);
}