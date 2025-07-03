/* _LWRM_COPYRIGHT_BEGIN_
 *
 *  Copyright 2021-2022 by LWPU Corporation.  All rights reserved.  All
 *  information contained herein is proprietary and confidential to LWPU
 *  Corporation.  Any use, reproduction, or disclosure without the written
 *  permission of LWPU Corporation is prohibited.
 *
 *  _LWRM_COPYRIGHT_END_
 */
#include "blackwell/gb100/dev_runlist.h"
#include "blackwell/gb100/dev_ram.h"
#include "blackwell/gb100/dev_pbdma.h"

#include "fifo.h"
#include "vgpu.h"
#include "vmem.h"

#include "g_fifo_private.h"        // (rmconfig) implementation prototypes

/// @brief This macro is needed here to parse multi-word/array defines from HW
#define SF_ARR32_VAL(s,f,arr) \
    (((arr)[SF_INDEX(LW ## s ## f)] >> SF_SHIFT(LW ## s ## f)) & SF_MASK(LW ## s ## f))

/* adding these temporary defines for HW changes transition in TSG Header */
// TODO(ksaini) : remove these defines once HW changes are in. 
#define LW_RAMRL_ENTRY_TSG_TSGID_DWORD1                  (23+1*32):(12+1*32) /* RWXUF */

//-----------------------------------------------------
// PBDMA_CACHE1(n,id)
//
// This macro is used by fifoDumpPbdmaRegsCache1_GB100()
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

//-----------------------------------------------------
// fifoGetChannelBaseCount_GB100
//
// It retruns the virtual channel config values of the runlist
//-----------------------------------------------------
LW_STATUS fifoGetChannelBaseCount_GB100(LwU32 runlistId, LwU32 *pBase, LwU32 *pCount)
{
    LwU32      val;
    LwU32      runlistPriBase;
    LW_STATUS  status;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST,
        runlistId, ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);

    if (status != LW_OK)
    {
        dprintf("**ERROR: Cannot get runlist PRI base %s\n", __FUNCTION__);
        return status;
    }

    val = GPU_REG_RD32(runlistPriBase +
                       LW_RUNLIST_VIRTUAL_CHANNEL_CFG(getGfid()));

    *pBase  = DRF_VAL(_RUNLIST, _VIRTUAL_CHANNEL_CFG, _BASE,  val);
    *pCount = DRF_VAL(_RUNLIST, _VIRTUAL_CHANNEL_CFG, _COUNT, val);

    return LW_OK;
}

//-----------------------------------------------------
// fifoDumpPbdmaRegsCache1_GB100(LwU32 pbdmaId)
//
// This function dumps out pbdma cache1 regs for given pbdmaId
// Uses DUMP_METHOD and DUMP_DATA macros
//-----------------------------------------------------
void fifoDumpPbdmaRegsCache1_GB100(LwU32 pbdmaId)
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

//-----------------------------------------------------
// fifoGetPbdmaState_GB100
//
// Prints information for each PBDMA unit.
// If the PBDMA is bound to a channel then prints
// things like the GP GET and PUT pointers. For this
// it needs to freeze the host and unfreeze it after
// reading what is needed.
//-----------------------------------------------------
void
fifoGetPbdmaState_GB100(void)
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

        if(chid == LW_PPBDMA_STATUS_SCHED_CHAN_STATUS_ILWALID)
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
        }
    }
}

//-----------------------------------------------------
// fifoDumpPb_GB100
//
// This function dumps the pushbuffer for each PBDMA
// unit that has a channel bound to it. It finds the
// GPFIFO GET pointer location and starts reading
// GP entries until it reaches the GPFIFO PUT pointer.
// For each entry it dumps the pushbuffer segment that
// corresponds to this entry.
//-----------------------------------------------------
void fifoDumpPb_GB100(LwU32 chid, LwU32 pbOffset, LwU32 sizeInBytes, LwU32 printParsed)
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
        buf = GPU_REG_RD32(LW_PPBDMA_STATUS_SCHED(i));

        // if there is a channel bound to this PBDMA unit
        if(SF_VAL(_PPBDMA, _STATUS_SCHED_CHAN_STATUS, buf) != LW_PPBDMA_STATUS_SCHED_CHAN_STATUS_ILWALID)
        {
            buf = GPU_REG_RD32(LW_PPBDMA_CHANNEL(i));
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

/**
 *  bChramPriBaseValid = LW_TRUE when we pass chramPriBase for caching purposes.
 *  bRunListValid = LW_TRUE the function will return a pointer to the instance block of a runlist whose id is passed.
 *  If we do not pass runlistId and we do not passs chramPriBase, then the address
 *  of channel id will be absolute.
 */
LW_STATUS
fifoGetChannelInstForChid_GB100
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

LW_STATUS fifoDumpEschedChannelRamRegsById_GB100(LwU32 runlistId)
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

LwU64
fifoGetGpBaseByChId_GB100
(
    ChannelId *pChannelId
)
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

    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_GP_BASE_HI), &buf, 4);
    result = SF_VAL(_PPBDMA, _GP_BASE_HI_OFFSET, buf);
    result <<= 32;
    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_GP_BASE), &buf, 4);
    buf = SF_VAL(_PPBDMA, _GP_BASE_OFFSET, buf) << 3;
    result += buf;

    return result;
}

//-----------------------------------------------------
// fifoGetGpInfoByChId_GB100
//
// Prints information about a channel. This information
// is from the RAMFC and can be not relevant at some
// moments.
//-----------------------------------------------------
void fifoGetGpInfoByChId_GB100(ChannelId *pChannelId)
{
    LwU32       buf = 0;
    LwU32       gpGet;
    LwU32       gpPut;

    LwU64       fifoBase;
    LwU64       instMemAddr;

    readFn_t readFn = NULL;

    // get instance memory address for the channel
    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId, &readFn, NULL, NULL);
    if (readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return;
    }

    dprintf(" + INST MEM ADDR:                      " LwU64_FMT "\n", instMemAddr);

    // read information about context state from the RAMFC
    // GPFIFO base
    fifoBase = pFifo[indexGpu].fifoGetGpBaseByChId(pChannelId);
    dprintf(" + RAMFC GP BASE ADDR:                 " LwU64_FMT "\n", fifoBase);

    // GPFIFO GET pointer
    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_GP_GET), &buf, 4);
    gpGet = SF_VAL(_RAMFC, _GP_GET, buf);
    dprintf(" + RAMFC GP GET:                       " LwU64_FMT "\n", (LwU64)gpGet&0xffffffff);

    // GPFIFO PUT pointer
    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_DEBUG_STATE(LW_RAMFC_DEBUG_STATE_RAMFC_INDEX_gp_put)), &buf, 4);
    gpPut = SF_INDEX_VAL(_RAMFC, _DEBUG_STATE, LW_RAMFC_DEBUG_STATE_RAMFC_INDEX_gp_put, buf);
    dprintf(" + RAMFC GP PUT:                       " LwU64_FMT "\n", (LwU64)gpPut&0xffffffff);
}

/*!
 * Dumps the Subcontext info
 */
void fifoDumpSubctxInfo_GB100(ChannelId *pChannelId)
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

/*!
 * @return The maximum number of channels provided by the chip
 */
LwU32 fifoGetNumChannels_GB100(LwU32 runlistId)
{
    LwU32 base;
    LwU32 count;

    if (isVirtualWithSriov())
    {
        pFifo[indexGpu].fifoGetChannelBaseCount(runlistId, &base, &count);
        return count;
    }

    return LW_CHRAM_CHANNEL__SIZE_1;
}

/*!
 * @return system chid for the virtual chid
 */
LwU32 fifoGetSchidForVchid_GB100(LwU32   vChid)
{
    LwU32 base;
    LwU32 count;

    if (!isVirtualWithSriov())
    {
        return vChid;
    }

    pFifo[indexGpu].fifoGetChannelBaseCount(RUNLIST_ALL, &base, &count);

    if (count == 0)
    {
        dprintf("**ERROR: count value is 0 \n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    return vChid + base;
}

/*!
 * @return TSGID from provided runlist entry.
 */
LwU32 fifoGetTsgIdFromRunlistEntry_GB100(LwU32 *entry)
{
    return DRF_VAL(_RAMRL, _ENTRY_TSG, _TSGID_DWORD1, entry[1]);
}