/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "turing/tu102/dev_fifo.h"
#include "turing/tu102/dev_ram.h"
#include "turing/tu102/dev_ctrl.h"
#include "turing/tu102/dev_pbdma.h"
#include "fb.h"
#include "mmu.h"
#include "vmem.h"
#include "vgpu.h"

#include "gpuanalyze.h"
#include "deviceinfo.h"

#include "g_fifo_private.h"        // (rmconfig) implementation prototypes

LwU32 fifoGetNumEng_TU102(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}

void
fifoDumpEngStates_TU102
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

//-----------------------------------------------------
// PBDMA_CACHE1(n,id)
//
// This macro is used by fifoDumpPbdmaRegsCache1_TU102()
// to dump out LW_PBDMA_METHOD/DATA0/1/2/3 info.
//-----------------------------------------------------
#define PBDMA_CACHE1(n, id)                                \
do                                                         \
{                                                          \
    regRead = GPU_REG_RD32(LW_PPBDMA_METHOD##n(id));       \
    incr    = DRF_VAL(_PPBDMA_METHOD, n, _INCR, regRead);  \
                                                           \
    addr = DRF_VAL(_PPBDMA_METHOD, n, _ADDR, regRead);     \
                                                           \
    subch = DRF_VAL(_PPBDMA_METHOD, n, _SUBCH, regRead);   \
                                                           \
    first = DRF_VAL(_PPBDMA_METHOD, n, _FIRST, regRead);   \
                                                           \
    dual = DRF_VAL(_PPBDMA_METHOD, n, _DUAL, regRead);     \
                                                           \
    valid   = DRF_VAL(_PPBDMA_METHOD, n, _VALID, regRead); \
    regRead = GPU_REG_RD32(LW_PPBDMA_DATA##n(id));         \
    value   = DRF_VAL(_PPBDMA_DATA, n, _VALUE, regRead);   \
                                                           \
    dprintf("%d    %s    0x%04x    %d      ",              \
            n,                                             \
            pbdmaTrueFalse[incr].strName,                  \
            addr << 2,                                     \
            subch);                                        \
    dprintf("%s     %s    %s",                             \
            pbdmaTrueFalse[first].strName,                 \
            pbdmaTrueFalse[dual].strName,                  \
            pbdmaTrueFalse[valid].strName);                \
    dprintf("    0x%08x\n", value);                        \
} while (0)

//-----------------------------------------------------
// fifoDumpPbdmaRegsCache1_TU102(LwU32 pbdmaId)
//
// This function dumps out pbdma cache1 regs for given pbdmaId
// Uses DUMP_METHOD and DUMP_DATA macros
//-----------------------------------------------------
void fifoDumpPbdmaRegsCache1_TU102(LwU32 pbdmaId)
{
    LwU32 regRead;
    LwU32 incr;
    LwU32 addr;
    LwU32 subch;
    LwU32 first;
    LwU32 dual;
    LwU32 valid;
    LwU32 value;

    NameValue pbdmaTrueFalse[2] =
    {
        {"False",0x0},
        {"True ",0x1}
    };

    dprintf("\n");
    dprintf(" PBDMA %d CACHE1 :  \n", pbdmaId);
    dprintf("      -------------------- METHOD ----------------------   ---DATA---\n");
    dprintf("N     INCR     ADDR    SUBCH    FIRST     DUAL     VALID      VALUE \n");
    dprintf("--    ----     ----    -----    -----     -----    -----      -----\n");
    PBDMA_CACHE1(0, pbdmaId);
    PBDMA_CACHE1(1, pbdmaId);
    PBDMA_CACHE1(2, pbdmaId);
    PBDMA_CACHE1(3, pbdmaId);
}

LW_STATUS
fifoReadRunlistInfo_TU102
(
    LwU32   runlistId,
    LwU64  *pEngRunlistPtr,
    LwU32  *pRunlistLength,
    LwU32  *pTgtAperture,
    LwU32 **ppRunlistBuffer
)
{
    LwU64 engRunlistPtr;
    LwU32 runlistLength;
    LwU32 tgtAperture;
    LW_STATUS status;
    LwU32 engRunlistBaseLo = GPU_REG_RD32(LW_PFIFO_RUNLIST_BASE_LO(runlistId));
    LwU32 engRunlistBaseHi = GPU_REG_RD32(LW_PFIFO_RUNLIST_BASE_HI(runlistId));
    LwU32 engRunlistSubmit = GPU_REG_RD32(LW_PFIFO_RUNLIST_SUBMIT(runlistId));

    engRunlistPtr = ((LwU64)DRF_VAL(_PFIFO, _RUNLIST_BASE_HI, _PTR_HI, engRunlistBaseHi) << DRF_SIZE(LW_PFIFO_RUNLIST_BASE_LO_PTR_LO)) |
                                    (LwU64)DRF_VAL(_PFIFO, _RUNLIST_BASE_LO, _PTR_LO, engRunlistBaseLo);
    engRunlistPtr <<= LW_PFIFO_RUNLIST_BASE_PTR_ALIGN_SHIFT;
    runlistLength = DRF_VAL(_PFIFO, _RUNLIST_SUBMIT, _LENGTH, engRunlistSubmit);
    tgtAperture = (DRF_VAL(_PFIFO, _RUNLIST_BASE_LO, _TARGET, engRunlistBaseLo));

    if (pEngRunlistPtr)
    {
        *pEngRunlistPtr = engRunlistPtr;
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
        status = pFifo[indexGpu].fifoAllocateAndFetchRunlist(engRunlistPtr, runlistLength,
                                                             tgtAperture, ppRunlistBuffer);
        if (LW_OK != status)
        {
            dprintf("**ERROR: Could not fetch runlist info\n");
            return status;
        }
    }

    return LW_OK;
}

/*!
 * Checks if a runlist length is valid (non-zero)
 */
LwBool fifoIsRunlistLengthValid_TU102(LwU32 runlistLength)
{
    return LW_PFIFO_RUNLIST_SUBMIT_LENGTH_ZERO < runlistLength &&
           runlistLength <= LW_PFIFO_RUNLIST_SUBMIT_LENGTH_MAX;
}

LW_STATUS
fifoValidateRunlistInfoForAlloc_TU102
(
    LwU32   **ppOutRunlist,
    LwU64     engRunlistPtr,
    LwU32     runlistLength,
    LwU32     tgtAperture,
    readFn_t *pReadFunc
)
{
    if ( ppOutRunlist == NULL)
    {
        dprintf("**ERROR: %s:%d: Must pass runlist output.\n", __FILE__, __LINE__);
        return LW_ERR_GENERIC;
    }

    if (engRunlistPtr == 0) // edit
    {
        dprintf("**ERROR: LW_PFIFO_ENG_RUNLIST_BASE_PTR_NULL\n");
        return LW_ERR_GENERIC;
    }

    if (!pFifo[indexGpu].fifoIsRunlistLengthValid(runlistLength))
    {
        dprintf("**ERROR: LW_PFIFO_RUNLIST_SUBMIT_LENGTH Not allocatable\n");
        return LW_ERR_GENERIC;
    }

    if (pReadFunc == NULL)
    {
        dprintf("**ERROR: LW_PFIFO_ENG_RUNLIST_ALLOCATOR FUNCTION PTR is NULL\n");
        return LW_ERR_GENERIC;
    }

    switch (tgtAperture)
    {
        case LW_PFIFO_RUNLIST_BASE_LO_TARGET_VID_MEM :
            *pReadFunc = pFb[indexGpu].fbRead;
        break;

        case LW_PFIFO_RUNLIST_BASE_LO_TARGET_SYS_MEM_COHERENT :
            *pReadFunc = readSystem;
        break;

        case LW_PFIFO_RUNLIST_BASE_LO_TARGET_SYS_MEM_NONCOHERENT :
            *pReadFunc = readSystem;
        break;

        default :
            return LW_ERR_GENERIC;
    }

    return LW_OK;
}

void
fifoGetChannelOffsetMask_TU102
(
    LwU32 runlistId,
    LwU32 *pOffset,
    LwU32 *pMask
)
{
    LwU32 val;

    // Unused pre-AMPERE
    (void) runlistId;

    val = GPU_REG_RD32(LW_CTRL_VIRTUAL_CHANNEL_CFG(getGfid()));
    // Checking the value is needed to WAR HW bug 2431475.
    // Do we need a loop?
    assert ((val & 0xBAD00000) != 0xBAD00000);

    *pMask = DRF_VAL(_CTRL, _VIRTUAL_CHANNEL_CFG, _MASK, val);
    *pOffset = DRF_VAL(_CTRL, _VIRTUAL_CHANNEL_CFG, _SET,  val);
}

/*!
 * @return The maximum number of channels provided by the chip
 */
LwU32 fifoGetNumChannels_TU102(LwU32 runlistId)
{
    LwU32 mask, offset;
    if (isVirtualWithSriov())
    {
        pFifo[indexGpu].fifoGetChannelOffsetMask(runlistId, &offset, &mask);
        return mask + 1;
    }
    return LW_PCCSR_CHANNEL__SIZE_1;
}


LwU32
fifoGetSchidForVchid_TU102
(
    LwU32   vChid
)
{
    LwU32 mask, offset;
    if (!isVirtualWithSriov())
    {
        return vChid;
    }

    pFifo[indexGpu].fifoGetChannelOffsetMask(RUNLIST_ALL, &offset, &mask);

    return ((vChid & mask) | offset);
}

