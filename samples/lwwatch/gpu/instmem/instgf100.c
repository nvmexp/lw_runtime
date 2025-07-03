/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//

#include "fermi/gf100/dev_bus.h"
#include "fermi/gf100/dev_mmu.h"
#include "fermi/gf100/dev_fb.h"
#include "fermi/gf100/dev_ram.h"
#include "fermi/gf100/dev_fifo.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "mmu.h"
#include "inst.h"
#include "fb.h"
#include "fifo.h"
#include "print.h"
#include "vmem.h"

#include "hwref/lwutil.h"
#include "class/cl906f.h"

#include "g_instmem_private.h"      // (rmconfig) implementation prototypes



void* getUserdDInfo_GF100(LwU32 chid);
static LW_STATUS getRamfcStartAddr_GF100 (ChannelId *pChannelId, LwU32* pRamfcSize, readFn_t* memReadFunc, LwU64* pRamfcBase);
static LW_STATUS _readBar1_GF100(LwU64 offset, void* buf, LwU32 size, LwU32 *pSizer);
static void dumpUserd_GF100(void* pUserd);

/**
 * Prints the DWORD at a memory location corresponding to each formattedMemoryEntry.
 */
void PrintFormattedMemory(formattedMemoryEntry* entry, LwU32 numEntries, void* pMemory)
{
    LwU32 i;
    LwU32 byte_offset = 0;

    for (i=0; i < numEntries; i++)
    {
        byte_offset = entry[i].bit_offset >> 5;

        dprintf("lw: 0x%02x %-40s 0x%08x\n", byte_offset,
                entry[i].name,
                ((LwU32*)pMemory)[byte_offset]);
    }
}

static LW_STATUS _getUserDInfo(LwU32 chid, void** ppUserDInfo, const LwU32 userdSize, const LwU32 channelStride)
{
    void* pUserd = NULL;
    LwU32 data32 = 0;
    LW_STATUS  status = LW_OK;
    LwU64  basePtr = 0;
    LwU32 sizeRead = 0;

    data32 = GPU_REG_RD32(LW_PFIFO_BAR1_BASE);
    if(SF_VAL(_PFIFO, _BAR1_BASE_VALID, data32) == LW_PFIFO_BAR1_BASE_VALID_FALSE)
    {
        status = LW_ERR_GENERIC;
        dprintf("lw: LW_PFIFO_BAR1_BASE_VALID : _FALSE\n");
        goto ERROR_END;
    }
    basePtr = DRF_VAL(_PFIFO, _BAR1_BASE, _PTR, data32) << LW_PFIFO_BAR1_BASE_PTR_ALIGN_SHIFT;
    basePtr += channelStride*chid;

    pUserd = malloc(userdSize);
    status = _readBar1_GF100(basePtr, (void*)pUserd, userdSize, &sizeRead);
    if ((status == LW_ERR_GENERIC) || (sizeRead != userdSize))
    {
        goto ERROR_END;
    }

    *ppUserDInfo = pUserd;
    return LW_OK;

ERROR_END:
   if (pUserd)
   {
       free(pUserd);
   }
   return status;
}

static LW_STATUS _readBar1_GF100(LwU64 offset, void* buf, LwU32 size, LwU32 *pSizer)
{
    LwU64 bar1PhysAddr = 0;
    LwU64 tmpSizer = 0;
    LW_STATUS ret;

    if (pSizer == NULL)
    {
        return LW_ERR_GENERIC;
    }

    if (lwBar1 == 0)
    {
        dprintf("lw: %s lwBar1=0 ; probly not initialized. Aborting\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    bar1PhysAddr = lwBar1 & ~(BIT(8)-1);
    ret = readPhysicalMem(bar1PhysAddr + offset, buf, size, &tmpSizer);
    *pSizer = (LwU32)tmpSizer;
    return ret;
}

//-----------------------------------------------------
// getUserDInfo_GF100
// wrapper for _getUserDInfo
// Fetches the BAR1 mapped USERD ram info
// Info is Upto date as opposed to fetching from RAMFC
// Info based on class 906f
//-----------------------------------------------------
void* getUserdDInfo_GF100(LwU32 chid)
{
    void* pUserd = NULL;
    LW_STATUS status = LW_OK;
    const LwU32 numChannels = pFifo[indexGpu].fifoGetNumChannels(RUNLIST_ALL);
    LwU32 userdSize = 0;
    LwU32 channelStride = 0;

    pInstmem[indexGpu].instmemGetUserdParams(&userdSize, &channelStride);

    if (chid > numChannels)
    {
        return NULL;
    }

    status = _getUserDInfo(chid, (void**)&pUserd, userdSize, channelStride);
    if (status == LW_ERR_GENERIC)
    {
        dprintf("lw: Could not fetch USERD; rely on RAMFC for now.\n");
        return NULL;
    }

    return ((void*)pUserd);
}

//-----------------------------------------------------
// instmemDumpFifoCtx_GF100
// Prints out the fifo context data for a given chId
//-----------------------------------------------------
void instmemDumpFifoCtx_GF100(ChannelId *pChannelId)
{
    LW_STATUS status;
    LwU32 ramFcSize;
    LwU64  ramfcBase;
    readFn_t memReadFunc;
    LwU32* pFifoCtxVals = NULL;
    void* pUserd = NULL;
    LwU32 userdSize = 0;
    LwU32 channelStride = 0;

    pInstmem[indexGpu].instmemGetUserdParams(&userdSize, &channelStride);

    status = getRamfcStartAddr_GF100(pChannelId, &ramFcSize, &memReadFunc, &ramfcBase);
    if (status != LW_OK)
    {
        dprintf("lw: %s - could not find Ramfc start address\n", __FUNCTION__);
        goto EXIT_ERROR;
    }

    pFifoCtxVals = (LwU32*) malloc(ramFcSize);
    if (!pFifoCtxVals)
    {
        dprintf("lw: %s - alloc mem for fifoctx failed\n", __FUNCTION__);
        goto EXIT_ERROR;
    }

    //fetch ramfc data
    status = memReadFunc(ramfcBase, (void*)pFifoCtxVals, ramFcSize);
    if (status != LW_OK)
    {
        goto EXIT_ERROR;
    }

    //lwwatch global
    if (verboseLevel)
    {
        printBuffer((char*)pFifoCtxVals, ramFcSize, ramfcBase, 4);
        dprintf("\n");
    }

    pFifo[indexGpu].fifoDumpCtxRegisters(pFifoCtxVals);

    //dump the userd info as well
    status = _getUserDInfo(pChannelId->id, (void**)&pUserd, userdSize, channelStride);
    if (status == LW_ERR_GENERIC)
    {
        dprintf("lw: Could not fetch USERD; rely on RAMFC for now.\n");
        goto EXIT_ERROR;
    }

    //else dump USERD info
    dumpUserd_GF100(pUserd);

EXIT_ERROR:

    if (pFifoCtxVals)
    {
        free(pFifoCtxVals);
    }

    if (pUserd)
    {
        free((void*)pUserd);
    }
}

/*!
 * Prints out the USERD structure.
 * The structure itself may vary from chip to chip.
 */
static void dumpUserd_GF100( void* pUserd )
{
    formattedMemoryEntry* pEntry;
    LwU32 numEntries;
    pInstmem[indexGpu].instmemGetUserdFormattedMemory(&pEntry, &numEntries);

    dprintf("lw: ==================================\n");
    dprintf("lw: Fifoctx info from USERD: \n");
    PrintFormattedMemory(pEntry, numEntries, pUserd);
}


//-----------------------------------------------------
// getRamfcStartAddr_GF100
// Fetches the Ramfc start address , size and read function
//-----------------------------------------------------
static LW_STATUS
getRamfcStartAddr_GF100
(
    ChannelId *pChannelId,
    LwU32 *pRamfcSize,
    readFn_t *memReadFunc,
    LwU64 *pRamfcBase
)
{
    if (pRamfcSize == NULL || memReadFunc == NULL || pRamfcBase == NULL || pChannelId == NULL)
    {
        return LW_ERR_GENERIC;
    }

    *pRamfcSize = LW_RAMFC_SIZE_VAL;
    *pRamfcBase = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId, memReadFunc, NULL, NULL);

    //go to the base of Ramfc
    *pRamfcBase += SF_OFFSET(LW_RAMIN_RAMFC);

    return LW_OK;
}


void
instmemDumpPdeForChannel_GF100
(
    LwU32   chId,
    LwU32   begin,
    LwU32   end
)
{
    VMemSpace   vMemSpace;
    VMEM_INPUT_TYPE Id;
    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chId;

    if ((vmemGet(&vMemSpace, VMEM_TYPE_CHANNEL, &Id) == LW_OK))
    {
        vmemDoPdeDump(&vMemSpace, begin, end);
    }
}

void
instmemDumpPteForChannel_GF100
(
    LwU32   chId,
    LwU32   pdeIndex,
    LwU32   begin,
    LwU32   end
)
{
    VMemSpace   vMemSpace;
    VMEM_INPUT_TYPE Id;
    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chId;

    if (vmemGet(&vMemSpace, VMEM_TYPE_CHANNEL, &Id) == LW_OK)
    {
        pVmem[indexGpu].vmemDoPteDump(&vMemSpace, pdeIndex, begin, end);
    }
}

#define ALIGNMENT                0xc
#define ILWALID_INST            ~0

/*!
 *  Fetches the chid corrsponding to the given Instance
 *
 *  @param[in] inst - instance ptr
 *  @param[in] target - the mem aperture of the instance
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
LW_STATUS instmemGetChidFromInst_GF100(LwU32 inst, LwU32 target)
{
    LwU32 i;
    LwU64 instMemStart;
    LwU32 aperture;
    LwU64 instAddr = inst << ALIGNMENT;
    const LwU32 numChannels = pFifo[indexGpu].fifoGetNumChannels(RUNLIST_ALL);
    ChannelId channelId;

    if (inst == 0)
    {
        dprintf("lw: inst was 0x0 in %s...\n", __FUNCTION__);
        return (ILWALID_INST);
    }

    // Go through all the fifo contexts and look for a match
    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    for (i = 0; i < numChannels; i++)
    {
        channelId.id = i;
        instMemStart = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(&channelId, NULL, NULL, (MEM_TYPE*)&aperture);
        if ((instMemStart == instAddr) && (target == aperture))
        {
            dprintf("lw: Chid:      0x%04x - for the given context\n", i);
            break;
        }
    }

    if (i != numChannels)
        return i;

    dprintf("lw: %s did not find a match...\n", __FUNCTION__);
    return (ILWALID_INST);
}


//-----------------------------------------------------
// instmemSetStartAddress_GF100
//-----------------------------------------------------
void instmemSetStartAddress_GF100(void)
{
    // For Fermi+, VBIOS is shadowed at bar0+7e0000.
    hal.instStartAddr        = (lwBar0 + 0x7E0000);
}
