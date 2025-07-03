/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//

#include "kepler/gk104/dev_fifo.h"
#include "kepler/gk104/dev_ram.h"
#include "kepler/gk104/dev_pbdma.h"
#include "kepler/gk104/dev_top.h"
#include "gr.h"
#include "fb.h"
#include "mmu.h"
#include "vmem.h"
#include "kepler/gk104/hwproject.h"
#include "inst.h"
#include "vgpu.h"

#include "gpuanalyze.h"
#include "deviceinfo.h"

#include "g_fifo_private.h"        // (rmconfig) implementation prototypes
#include "g_fifo_hal.h"


static BOOL checkFifoIntrInfo_GK104(void);
static LW_STATUS allChannelsIdle(BOOL* bAllIdle);
static void getFifoIntrInfo_GK104(void);
static void _printRunlistEngines(LwU32 runlistId);

NameValue targetMem[4];
NameValue trueFalse[2] =
{
    {"FALSE",0x0},
    {"TRUE",0x1}
};

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
void fifoDumpPbdmaRegs_GK104(LwU32 pbdmaId)
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

    PRINT_PBDMA_REG_PD( _PPBDMA, _SEMAPHOREA, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _SEMAPHOREB, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _SEMAPHOREC, pbdmaId );
    PRINT_PBDMA_REG_PD( _PPBDMA, _SEMAPHORED, pbdmaId );

    //LW_PPBDMA CACHE1
    pFifo[indexGpu].fifoDumpPbdmaRegsCache1(pbdmaId);
}

//-----------------------------------------------------
// fifoGetPbInfo_GK104
//
// Dumps the pbdma registers for all pbdma's.
//
// LW_HOST_NUM_PBDMA is #defined to be 6 in case LW_HOST_NUM_PBDMA is not already #defined.
//
//-----------------------------------------------------
void fifoGetPbInfo_GK104(void)
{
    LwU32 pbdmaId;
    const LwU32 num_pbdma = pFifo[indexGpu].fifoGetNumPbdma();

    dprintf("  Use pbdma<pbdmaId> for a specific pbdma info.\n");

    for (pbdmaId=0;pbdmaId<num_pbdma;pbdmaId++)
    {
        dprintf("\n  pbdma  id = %d     \n", pbdmaId);
        pFifo[indexGpu].fifoDumpPbdmaRegs(pbdmaId);
    }
}

LwU64
fifoGetGpBaseByChId_GK104
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

/*!
 * @return The address limit for the address space used by a channel.
 */
LwU64
fifoGetChannelAddressLimit_GK104(ChannelId *pChannelId)
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

//-----------------------------------------------------
// getFifoIntrInfo_GK104
//
// Gets information about various interrupts.
//-----------------------------------------------------
static void getFifoIntrInfo_GK104(void)
{
    LwU32 intr;
    LwU32 stallMsk;
    LwU32 nonStallMsk;

    intr = GPU_REG_RD32(LW_PFIFO_INTR_0);
    dprintf("LW_PFIFO_INTR_0:                       0x%08x\n", intr);
    priv_dump("LW_PFIFO_INTR_0");

    stallMsk = GPU_REG_RD32(LW_PFIFO_INTR_EN_0);
    dprintf("LW_PFIFO_INTR_EN_0:                    0x%08x\n", stallMsk);
    priv_dump("LW_PFIFO_INTR_EN_0");

    nonStallMsk = GPU_REG_RD32(LW_PFIFO_INTR_EN_1);
    dprintf("LW_PFIFO_INTR_EN_1:                    0x%08x\n", nonStallMsk);
    priv_dump("LW_PFIFO_INTR_EN_0");

    if (verboseLevel)
    {
        // Dump *stalling* interrupt info.
        pFifo[indexGpu].fifoDumpFifoIntrInfo(RUNLIST_ALL, intr, stallMsk);

        // Dump *non-stalling* interrupt info.
        pFifo[indexGpu].fifoDumpFifoIntrInfo(RUNLIST_ALL, intr, nonStallMsk);
    }
}

//-----------------------------------------------------
// fifoDumpFifoIntrInfo_GK104
//
// This function dumps information about various pending
// interrupts based on interrupt mask.
//-----------------------------------------------------
void fifoDumpFifoIntrInfo_GK104(LwU32 runlistId, LwU32 intr, LwU32 intrMsk)
{
    LwU32 regRead;
    // Unused pre-Ampere
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

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _FB_FLUSH_TIMEOUT, _PENDING))
    {
        dprintf(" + LW_PFIFO_INTR_0_FB_FLUSH_TIMEOUT_PENDING\n");
    }

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _LB_ERROR, _PENDING))
    {
        regRead = GPU_REG_RD32(LW_PFIFO_INTR_LB_ERROR);
        dprintf(" + LW_PFIFO_INTR_LB_ERROR              : 0x%08x\n", regRead);
        priv_dump("LW_PFIFO_INTR_LB_ERROR");
    }

    if (intr & DRF_DEF(_PFIFO, _INTR_0, _DROPPED_MMU_FAULT, _PENDING))
    {
        dprintf(" + LW_PFIFO_INTR_DROPPED_MMU_FAULT_PENDING\n");
    }

    // Dump MMU fault info
    if (intr & DRF_DEF(_PFIFO, _INTR_0, _MMU_FAULT, _PENDING))
    {
        priv_dump("LW_PFIFO_INTR_MMU_FAULT*");
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

//-----------------------------------------------------
// fifoGetIntrPbdmaInfo_GK104
//
// Gets information about various pbdma interrupts.
//-----------------------------------------------------
void fifoGetIntrPbdmaInfo_GK104(void)
{
    LwU32 regRead;
    LwU32 pbdmaId;
    const LwU32 num_pbdma = pFifo[indexGpu].fifoGetNumPbdma();

    regRead = GPU_REG_RD32(LW_PFIFO_INTR_PBDMA_ID);
    dprintf(" + LW_PFIFO_INTR_PBDMA_ID              : 0x%08x\n", regRead);

    for (pbdmaId=0;pbdmaId<num_pbdma;pbdmaId++)
    {
        if (regRead & BIT(pbdmaId))
        {
            dprintf ("    + LW_PFIFO_INTR_PBDMA_ID_%-9d : _PENDING\n", pbdmaId);
            pFifo[indexGpu].fifoDumpPerPbdmaIntrInfo(pbdmaId);
        }
    }
}

LwU32 fifoGetPbdmaConfigSize_GK104(void)
{
    // The maximum PBDMA count per engine is 1 on Kepler / Maxwell
    return 1;
}

//-----------------------------------------------------
// fifoGetPbdmaState_GK104
//
// Prints information for each PBDMA unit.
// If the PBDMA is bound to a channel then prints
// things like the GP GET and PUT pointers. For this
// it needs to freeze the host and unfreeze it after
// reading what is needed.
//-----------------------------------------------------
void
fifoGetPbdmaState_GK104(void)
{
    LwU32 i;
    LwU32 buf;
    LwU64 gpBase;
    const LwU32 num_pbdma = pFifo[indexGpu].fifoGetNumPbdma();
    LwU32 chid;

    dprintf("\n");

    for(i=0; i < num_pbdma; i++)
    {
        buf = GPU_REG_RD32(LW_PPBDMA_CHANNEL(i));
        chid = SF_VAL(_PPBDMA, _CHANNEL_CHID, buf);

        if(chid == LW_PPBDMA_CHANNEL_VALID_FALSE)
        {
            dprintf(" + PBDMA %03d ON CHANNEL:               NULL\n", i);
        }
        else
        {
            if (isVirtualWithSriov())
            {
                if ((chid < pFifo[indexGpu].fifoGetSchidForVchid(0)) ||
                    (chid > pFifo[indexGpu].fifoGetSchidForVchid(0) +
                            pFifo[indexGpu].fifoGetNumChannels(RUNLIST_ALL)))
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

        if(SF_VAL(_PPBDMA, _TARGET_ENGINE, buf) == LW_PPBDMA_TARGET_ENGINE_SW)
        {
            dprintf(" + PBDMA %03d ON ENGINE:                SW\n", i);
        }
        else
        {
            dprintf(" + PBDMA %03d ON ENGINE:                %03d\n", i, SF_VAL(_PPBDMA, _TARGET_ENGINE, buf));
        }
    }
}

//-----------------------------------------------------
// fifoGetGpInfoByChId_GK104
//
// Prints information about a channel. This information
// is from the RAMFC and can be not relevant at some
// moments.
//-----------------------------------------------------
void fifoGetGpInfoByChId_GK104(ChannelId *pChannelId)
{
    LwU32       buf = 0;
    LwU32       gpGet;
    LwU32       gpPut;

    LwU64       fifoBase;
    LwU64       gpLimit;
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
    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_GP_PUT), &buf, 4);
    gpPut = SF_VAL(_RAMFC, _GP_PUT, buf);
    dprintf(" + RAMFC GP PUT:                       " LwU64_FMT "\n", (LwU64)gpPut&0xffffffff);

    // GPFIFO limit
    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_GP_BASE_HI), &buf, 4);
    gpLimit = 1ull << SF_VAL(_PPBDMA, _GP_BASE_HI_LIMIT2, buf);
    dprintf(" + RAMFC GP LIMIT:                     " LwU64_FMT "\n", gpLimit);
}

//-----------------------------------------------------
// fifoGetInfo_GK104
//
// This function dumps all the information gathered
// about the fifo.
//-----------------------------------------------------
LW_STATUS
fifoGetInfo_GK104(void)
{
    LwU32 i;
    LwU32 buf;

    LwU64 pde;
    LwU64 addrLimit;
    LwU64 instMemAddr;
    BOOL  bIsInterrupt;

    readFn_t readFn = NULL;
    ChannelId channelId;
    ChannelInst channelInst;
    const LwU32 numChannels = pFifo[indexGpu].fifoGetNumChannels(RUNLIST_ALL);

    // Channel state
    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    for (i = 0; i < numChannels; i++)
    {
        channelId.id = i;
        pFifo[indexGpu].fifoGetChannelInstForChid(&channelId, &channelInst);
        if (channelInst.state & LW_PFIFO_CHANNEL_STATE_ENABLE)
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

                addrLimit = pFifo[indexGpu].fifoGetChannelAddressLimit(&channelId);
                dprintf(" + ADDR SPACE LIMIT:                   " LwU64_FMT "\n", addrLimit);

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
    dprintf("\n");

    // info about each pbdma unit that is bound to a channel
    pFifo[indexGpu].fifoGetPbdmaState();

    // Check for interrupts
    dprintf("\nChecking pending interrupts...\n");
    bIsInterrupt = checkFifoIntrInfo_GK104();

    if (bIsInterrupt)
    {
        getFifoIntrInfo_GK104();
    }
    else
    {
        dprintf("No interrupts are pending\n");
    }
    return LW_OK;
}

LwBool fifoIsChidInEntry_GK104(LwU32* entry, LwU32 chid)
{
    if(entry == NULL)
    {
        return FALSE;
    }

    return (chid == DRF_VAL(_RAMRL, _ENTRY, _CHID, entry[0]));
}

//-----------------------------------------------------
// Gets information about various interrupts.
//-----------------------------------------------------
static BOOL checkFifoIntrInfo_GK104(void)
{
    LwU32 intr;
    LwU32 stallMsk;
    LwU32 nonStallMsk;

    intr = GPU_REG_RD32(LW_PFIFO_INTR_0);
    dprintf("LW_PFIFO_INTR_0:                       0x%08x\n", intr);

    stallMsk = GPU_REG_RD32(LW_PFIFO_INTR_EN_0);
    dprintf("LW_PFIFO_INTR_EN_0:                    0x%08x\n", stallMsk);

    nonStallMsk = GPU_REG_RD32(LW_PFIFO_INTR_EN_1);
    dprintf("LW_PFIFO_INTR_EN_1:                    0x%08x\n", nonStallMsk);

    //
    // Returns true if any of the stalling or non-stalling interrupts
    // is enabled and pending.
    //
    return ((intr & stallMsk) || (intr & nonStallMsk));
}

static LW_STATUS allChannelsIdle(BOOL* bAllIdle)
{
    LwU32 reg;

    if (bAllIdle == NULL)
    {
        return LW_ERR_GENERIC;
    }

    reg = GPU_REG_RD32(LW_PFIFO_SCHED_STATUS);
    *bAllIdle = (LW_PFIFO_SCHED_STATUS_ALL_CHANNELS_IDLE == DRF_VAL(_PFIFO, _SCHED_STATUS, _ALL_CHANNELS, reg));

    return LW_OK;
}



//-----------------------------------------------------
// fifoTestHostState_GK104
//
// Check if channels are idle
// for a given channel :
//     1. Check for DMA pending
//     2. Check if channel is on runlist
//     3. Check for interrupts
//     4. Show engine states
//     5. Show semaphore info.
//-----------------------------------------------------

LW_STATUS fifoTestHostState_GK104( void )
{
    LW_STATUS status = LW_OK;
    LwU32   chid;
    BOOL    bAllChannelsIdle;
    BOOL    bIsChOnRunlist = TRUE;
    BOOL    bIsInterrupt;
    BOOL    bIsAcquirePending = FALSE;
    ChannelId channelId;
    ChannelInst channelInst;
    const LwU32 numChannels = pFifo[indexGpu].fifoGetNumChannels(RUNLIST_ALL);

    if (LW_ERR_GENERIC == allChannelsIdle(&bAllChannelsIdle))
    {
        dprintf("Assuming all channels are not idle.\n");
        bAllChannelsIdle = FALSE;
    }

    //check if all channels are idle
    if (bAllChannelsIdle == FALSE)
    {
        dprintf("Not all channels are idle\n");
    }

    dprintf("\nChecking pending channels...\n");

    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
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

                //2. check if channel is on runlist
                status = pFifo[indexGpu].fifoIsChannelOnRunlist( chid, &bIsChOnRunlist );
                if (status == LW_ERR_GENERIC)
                {
                    dprintf("**ERROR: fifoIsChannelOnRunlist\n");
                    return status;
                }

                if (!bIsChOnRunlist)
                {
                    dprintf("**ERROR: Channel #%03d is pending, but not on "
                        "runlist. Dumping fifo info\n", chid);
                    addUnitErr("\t Channel #%03d is pending, but not on runlist\n",
                        chid);

                    pFifo[indexGpu].fifoGetInfo();
                    status = LW_ERR_GENERIC;
                }
                else
                {
                    dprintf("Channel #%03d is pending and is on runlist\n",
                        chid);
                }
            }
        }
    }

    if (!(channelInst.state & LW_PFIFO_CHANNEL_STATE_PENDING))
    {
        dprintf("No channels are pending\n");
    }

    //3. Check for interrupts
    dprintf("\nChecking pending interrupts...\n");
    bIsInterrupt = checkFifoIntrInfo_GK104();

    if (bIsInterrupt)
    {
        getFifoIntrInfo_GK104();
    }
    else
    {
        dprintf("No interrupts are pending\n");
    }

    //4. Engine state
    dprintf("\nDumping engine states...\n");
    pFifo[indexGpu].fifoCheckEngStates(&gpuState);

    //5. Semaphores
    dprintf("\nChecking semaphores state...\n");

    for (chid = 0; chid < numChannels; chid++)
    {
        channelId.id = chid;
        pFifo[indexGpu].fifoGetChannelInstForChid(&channelId, &channelInst);

        if ((channelInst.state & LW_PFIFO_CHANNEL_STATE_ENABLE) &&
            (channelInst.state & LW_PFIFO_CHANNEL_STATE_ACQ_PENDING))
        {
            dprintf("**ERROR: CHANNEL_ACQUIRE_PENDING_ON set on channel #%03d\n", chid);
            addUnitErr("\t CHANNEL_ACQUIRE_PENDING_ON set on channel #%03d\n", chid);

            bIsAcquirePending = TRUE;
            status = LW_ERR_GENERIC;
        }
    }

    if (!bIsAcquirePending)
        dprintf("No semaphore acquire pending on any channel.\n");

    return status;
}


/**
 * Prints useful information about the target, mode, and context
 * space of an engine.
 * @param eng The engine's name
 * @param state_reg WFI if restoring from idle state,
 *                  FG if restoring from busy state
 */
#define DUMP_ENG_STATE_SINGLE(eng, state_reg, state_label) do { \
        dprintf(" + %-36s : %-4s\n", #eng, state_label); \
        dprintf("   + %-34s : %03x\n", \
                "TARGET", SF_VAL(_RAMIN, _## eng ## _ ## state_reg ## _TARGET, buf)); \
        dprintf("   + %-34s : %03x\n", \
                "MODE", SF_VAL(_RAMIN, _## eng ## _ ## state_reg ## _MODE, buf)); \
        lowBits = SF_VAL(_RAMIN, _## eng ## _ ## state_reg ## _PTR_LO, buf) << 12; \
        \
        readFn( instMemAddr + SF_OFFSET(LW_RAMIN_ ## eng ## _ ## state_reg ## _PTR_HI), &buf, 4); \
        highBits = SF_VAL(_RAMIN, _ ## eng ## _ ## state_reg ## _PTR_HI, buf); \
        ctxSpace = lowBits + (highBits << 32); \
        dprintf("   + %-34s : " LwU64_FMT "\n", "CTX SPACE", ctxSpace); \
}while(0)

/**
 * Prints the appropriate engine state based on whether the
 * engine is idle or busy. Sets a gpuState flag to true if
 * the engine is busy.
 */
#define DUMP_ENG_STATE(eng, gpuStateBusyVar) do { \
    readFn( instMemAddr + SF_OFFSET(LW_RAMIN_ ## eng ## _CS), &buf, 4 ); \
    if(SF_VAL(_RAMIN, _ ## eng ## _CS, buf) == LW_RAMIN_ ## eng ## _CS_WFI) \
    { \
        DUMP_ENG_STATE_SINGLE( eng, WFI, "IDLE" ); \
    } \
    else \
    { \
        readFn( instMemAddr + SF_OFFSET(LW_RAMIN_ ## eng ## _FG_TARGET), &buf, 4); \
        DUMP_ENG_STATE_SINGLE( eng, FG, "BUSY" ); \
        if (bFillState) \
        { \
            pGpuState->gpuStateBusyVar = TRUE; \
        } \
    } \
}while(0)

#define ALIGNADDR                   2

static LW_STATUS dumpEngRunlistById_GK104(LwU32 runlistId);
static BOOL isChannelOnRunlistEngId_GK104(LwU32 runlistId, LwU32 chid);

//
// Map the engine names, from projects.spec, to the values in
// LW_PTOP_INFO_TYPE_ENUM.
//
// The array index is used to represent the engines in runListsEngines[].
//
static EngineNameValue engName2DeviceInfo[] =
{
    {{"GR0",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_GRAPHICS }, 0, ENGINE_TAG_GR},
    {{"CE0",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_COPY0    }, 0, ENGINE_TAG_CE},
    {{"CE1",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_COPY1    }, 0, ENGINE_TAG_CE},
    {{"CE2",    LW_PTOP_DEVICE_INFO_TYPE_ENUM_COPY2    }, 0, ENGINE_TAG_CE},
    {{"MSPDEC", LW_PTOP_DEVICE_INFO_TYPE_ENUM_MSPDEC   }, 0, ENGINE_TAG_UNKNOWN},
    {{"MSPPP",  LW_PTOP_DEVICE_INFO_TYPE_ENUM_MSPPP    }, 0, ENGINE_TAG_UNKNOWN},
    {{"MSVLD",  LW_PTOP_DEVICE_INFO_TYPE_ENUM_MSVLD    }, 0, ENGINE_TAG_UNKNOWN},
    {{"MSENC",  LW_PTOP_DEVICE_INFO_TYPE_ENUM_MSENC    }, 0, ENGINE_TAG_UNKNOWN},
    {{"",       DEVICE_INFO_TYPE_ILWALID               }, 0, ENGINE_TAG_ILWALID}
};

/**
 *  @brief Represents a row in the device info table.
 */
typedef struct DeviceInfoRowType
{
    const char *typeName;
    LwU32       type;
} DeviceInfoRowType;
/**
 * @brief Represents a list of possible row types and their meaningful names.
 */
static DeviceInfoRowType DEVICE_INFO_ROW_TYPE[] =
{
    {"NOT VALID  ", LW_PTOP_DEVICE_INFO_ENTRY_NOT_VALID},
    {"DATA       ", LW_PTOP_DEVICE_INFO_ENTRY_DATA},
    {"ENUM       ", LW_PTOP_DEVICE_INFO_ENTRY_ENUM},
    {"ENGINE TYPE", LW_PTOP_DEVICE_INFO_ENTRY_ENGINE_TYPE}
};

static LwU32 *runListsEngines;

void * fifoGetEngNames_GK104(void)
{
    return engName2DeviceInfo;
}

LwU32 fifoGetDeviceInfoNumRows_GK104(void)
{
    return LW_PTOP_DEVICE_INFO__SIZE_1;
}

LwU32 fifoGetDeviceInfoMaxDevices_GK104(void)
{
    return LW_PTOP_DEVICE_INFO__SIZE_1;
}


LwU32 fifoGetRunlistMaxNumber_GK104(void)
{
    return 16;
}

/**
 * @brief Returns initialized allocated value of runListsEngines whose all masks are initialized to zeros.
 * If runListsEngines was already initialized it will return the initialized one.
 *
 * @return:
 *    Initialized runListsEngines
 *    NULL:
 *       When there is insufficient available memory.
 *
 * @note If a runlistsEngines 32-bit size of one mask isn't enough to store all the engines new implementation of
 * this function should be added. In that case return type should change to void* or it should be opaque structure.
 */
LwU32 *fifoGetRunlistsEngines_GK104(void)
{
    if (NULL == runListsEngines)
    {
        LwLength allocSize = pFifo[indexGpu].fifoGetRunlistMaxNumber() *
                             sizeof(*runListsEngines);
        runListsEngines = malloc(allocSize);
        if (NULL == runListsEngines)
        {
            return runListsEngines;
        }
        memset(runListsEngines, 0, allocSize);
    }
    return runListsEngines;
}

/**
 * @brief Gets type of a row that was parsed.
 *
 * @note Row types are obsolete since introduction of version 2 of deviceinfo.
 */
static const char *getDeviceInfoRowType(LwU32 type)
{
    LwLength idx;
    const char* typeName = DEVICE_INFO_ROW_TYPE[0].typeName;
    for (idx = 0; idx < sizeof(DEVICE_INFO_ROW_TYPE) / sizeof(*DEVICE_INFO_ROW_TYPE); idx++)
    {
        if (DEVICE_INFO_ROW_TYPE[idx].type == type)
        {
            typeName = DEVICE_INFO_ROW_TYPE[idx].typeName;
            break;
        }
    }
    return typeName;
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
 * Prints the DWORD at a memory location corresponding to each formattedMemoryEntry.
 */
static void _fifoPrintFormattedMemory(formattedMemoryEntry* entry, LwU32 numEntries, void* pMemory)
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


//-----------------------------------------------------
// fifoDumpCtxRegisters_GK104
// Prints the fifo ctx data based on register names in reference manualf
//-----------------------------------------------------
void fifoDumpCtxRegisters_GK104(LwU32 *fifoCtxVals)
{
    formattedMemoryEntry* pEntry;
    LwU32 numEntries;
    pInstmem[indexGpu].instmemGetRamfcFormattedMemory(&pEntry, &numEntries);
    _fifoPrintFormattedMemory(pEntry, numEntries, fifoCtxVals);
}


//-----------------------------------------------------
// fifoDumpPb_GK104
//
// This function dumps the pushbuffer for each PBDMA
// unit that has a channel bound to it. It finds the
// GPFIFO GET pointer location and starts reading
// GP entries until it reaches the GPFIFO PUT pointer.
// For each entry it dumps the pushbuffer segment that
// corresponds to this entry.
//-----------------------------------------------------
void fifoDumpPb_GK104(LwU32 chid, LwU32 pbOffset, LwU32 sizeInBytes, LwU32 printParsed)
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
                    pbGetAddr += SF_VAL(_PPBDMA, _GP_ENTRY0_GET, lowerBits) << ALIGNADDR;

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

/*!
 * @return The number of PBDMAs provided by the chip.
 */
LwU32 fifoGetNumPbdma_GK104(void)
{
    return LW_HOST_NUM_PBDMA;
}

LW_STATUS fifoGetChannelInstForChid_GK104(ChannelId *pChannelId, ChannelInst *pChannelInst)
{
    LwU32 regReadInst, regRead;

    if ((pChannelInst == NULL) || (pChannelId == NULL))
    {
        dprintf("**ERROR: Invalid argument for %s\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }
    else
    {
        LwU32 channelId = pFifo[indexGpu].fifoGetSchidForVchid(pChannelId->id);
        regRead = GPU_REG_RD32(LW_PCCSR_CHANNEL(channelId));
        regReadInst = GPU_REG_RD32(LW_PCCSR_CHANNEL_INST(channelId));

        pChannelInst->instPtr = DRF_VAL(_PCCSR, _CHANNEL_INST, _PTR, regReadInst);
        pChannelInst->instPtr <<= LW_PCCSR_CHANNEL_INST_PTR_ALIGN_SHIFT;

        pChannelInst->target = DRF_VAL(_PCCSR, _CHANNEL_INST, _TARGET, regReadInst);

        // In some places, we want to print these information out.
        pChannelInst->regChannel = regRead;

        if (DRF_VAL(_PCCSR, _CHANNEL, _STATUS, regRead) != LW_PCCSR_CHANNEL_STATUS_PENDING_ACQUIRE)
            pChannelInst->state &= ~LW_PFIFO_CHANNEL_STATE_ACQ_PENDING;
        else
            pChannelInst->state |= LW_PFIFO_CHANNEL_STATE_ACQ_PENDING;

        if (DRF_VAL(_PCCSR, _CHANNEL, _ENABLE, regRead) == LW_PCCSR_CHANNEL_ENABLE_NOT_IN_USE)
            pChannelInst->state &= ~LW_PFIFO_CHANNEL_STATE_ENABLE;
        else
            pChannelInst->state |= LW_PFIFO_CHANNEL_STATE_ENABLE;

        if (DRF_VAL(_PCCSR, _CHANNEL_INST, _BIND, regReadInst) == LW_PCCSR_CHANNEL_INST_BIND_FALSE)
            pChannelInst->state &= ~LW_PFIFO_CHANNEL_STATE_BIND;
        else
            pChannelInst->state |= LW_PFIFO_CHANNEL_STATE_BIND;

        if (DRF_VAL(_PCCSR, _CHANNEL, _STATUS, regRead) == LW_PCCSR_CHANNEL_STATUS_PENDING)
            pChannelInst->state &= ~LW_PFIFO_CHANNEL_STATE_PENDING;
        else
            pChannelInst->state |= LW_PFIFO_CHANNEL_STATE_PENDING;

        if (DRF_VAL(_PCCSR, _CHANNEL, _BUSY, regRead) == LW_PCCSR_CHANNEL_BUSY_FALSE)
            pChannelInst->state &= ~LW_PFIFO_CHANNEL_STATE_BUSY;
        else
            pChannelInst->state |= LW_PFIFO_CHANNEL_STATE_BUSY;
    }

    return LW_OK;
}

/*!
 * @return The maximum number of channels provided by the chip.
 */
LwU32 fifoGetNumChannels_GK104(LwU32 runlistId)
{
    // Unused pre-Ampere
    (void) runlistId;

    return LW_PCCSR_CHANNEL__SIZE_1;
}

static void _printRunlistEngines(LwU32 runlistId)
{
    LwU32 mask;
    LwU32 j;
    LwU32 *runListsEngines;
    EngineNameValue *engNames;

    runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();

    // Since fifoGetDeviceInfo is called before fifoGetRunlistsEngines,
    // we should ideally never run into the below condition.
    if (NULL == runListsEngines)
    {
        dprintf("**ERROR: Device info table is not parsed %s\n", __FUNCTION__);
        return;
    }

    if (pFifo[indexGpu].fifoGetRunlistMaxNumber() <= runlistId)
    {
        dprintf("**ERROR: Invalid runlistId %s\n", __FUNCTION__);
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

/*!
 * Iterates through runlist2EngMask helper array
 * and calls dumpEngRunlistById.
 */
LW_STATUS fifoDumpRunlist_GK104(LwU32 runlistIdPar)
{
    LwU32 runlistId = 0;
    LwU32 engineTag;
    LW_STATUS status;

    status = pFifo[indexGpu].fifoGetDeviceInfo();
    if (LW_OK != status)
    {
        return status;
    }

    if(runlistIdPar == RUNLIST_ALL)
    {
        for (runlistId = 0; runlistId < pFifo[indexGpu].fifoGetRunlistMaxNumber(); runlistId++)
        {
            //
            // RUNLIST->ENGINE_TAG colwersion for runlistId = 0 is not unique because
            // GR and GRCEs share the same runlist id. The check below is only a sanity check
            // to make sure the runlistId exists/is valid before printing its info.
            //
            status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, runlistId,
                                                         ENGINE_INFO_TYPE_ENGINE_TAG, &engineTag);
            if (status == LW_OK)
            {
                _printRunlistEngines(runlistId);
                status = pFifo[indexGpu].fifoDumpEngRunlistById(runlistId);
                if (LW_OK != status)
                {
                    dprintf("***ERROR runlist %d is not defined in device info or it does not exist\n", runlistId);
                    return status;
                }
            }
        }
    }
    else
    {
        if (pFifo[indexGpu].fifoGetRunlistMaxNumber() <= runlistIdPar)
        {
            return LW_ERR_ILWALID_ARGUMENT;
        }

        status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, runlistIdPar,
                                                     ENGINE_INFO_TYPE_ENGINE_TAG, &engineTag);
        if (status == LW_OK)
        {
            _printRunlistEngines(runlistIdPar);
            status = pFifo[indexGpu].fifoDumpEngRunlistById(runlistIdPar);
            if (LW_OK != status)
            {
                dprintf("***ERROR runlist %d is not defined in device info or it does not exist\n",
                    (LwU32)runlistIdPar);
                return status;
            }
        }
        else
        {
            dprintf("Invalid runlistId\n");
        }
    }
    return LW_OK;
}

LwU32 fifoGetNumEng_GK104(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}

/*!
 * Checks if a runlist length is valid (non-zero)
 */
LwBool fifoIsRunlistLengthValid_GK104(LwU32 runlistLength)
{
    return LW_PFIFO_ENG_RUNLIST_LENGTH_ZERO < runlistLength &&
           runlistLength <= LW_PFIFO_ENG_RUNLIST_LENGTH_MAX;
}

/*!
 * Allocates memory for a runlist and fetches it from memory.
 * If the allocation is successful, the caller is responsible
 * for freeing the runlist.
 *
 * @param ppOutRunlist Allocated runlist output (required)
 */
LW_STATUS
fifoAllocateAndFetchRunlist_GK104
(
    LwU64    engRunlistPtr,
    LwU32    runlistLength,
    LwU32    tgtAperture,
    LwU32**  ppOutRunlist
)
{
    LwU32*      runlistBuffer = 0;
    readFn_t    readFn = NULL;
    LW_STATUS   status;
    LwU32       rlEntrySize = pFifo[indexGpu].fifoRunlistGetEntrySizeLww();
    LwU32       oldAddrtype;

    status = pFifo[indexGpu].fifoValidateRunlistInfoForAlloc(ppOutRunlist,
                                             engRunlistPtr,
                                             runlistLength,
                                             tgtAperture,
                                             &readFn);
    if (readFn == NULL)
    {
        //
        // If runlist length is zero, we expect to have no readFn
        // This isn't worth an mcheck exception to HAL out 0 as
        // LW_PFIFO_RUNLIST_LENGTH_ZERO
        //
        if (runlistLength != 0)
        {
            dprintf("**ERROR: NULL value of readFn.\n");
        }
        return LW_ERR_NOT_SUPPORTED;
    }

    if (status != LW_OK)
    {
        return LW_ERR_GENERIC;
    }

    runlistBuffer = (LwU32*)malloc(runlistLength * rlEntrySize);
    if (runlistBuffer == NULL)
    {
        dprintf("**ERROR: %s: Failed to allocate memory\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    oldAddrtype = getLwwatchAddrType();
    setLwwatchAddrType(HOST_PHYSICAL);
    status = readFn(engRunlistPtr, (void*)runlistBuffer, runlistLength * rlEntrySize);
    if (status != LW_OK)
    {
        dprintf("**ERROR: Could not fetch runlist entry\n");
        free(runlistBuffer);
        return LW_ERR_GENERIC;
    }
    setLwwatchAddrType(oldAddrtype);

    *ppOutRunlist = runlistBuffer;

    return LW_OK;
}

void fifoPrintRunlistEntry_GK104(LwU32 unused, LwU32 *entry)
{
    if (!entry)
    {
        dprintf("Invalid entry");
        return;
    }

    if(DRF_VAL(_RAMRL, _ENTRY, _TYPE, entry[0]) == LW_RAMRL_ENTRY_TYPE_TSG)
    {
        LwU32 tsgId = DRF_VAL(_RAMRL, _ENTRY, _ID, entry[0]);
        LwU32 tsgLen = DRF_VAL(_RAMRL, _ENTRY, _TSG_LENGTH, entry[0]);
        dprintf("TSG         0x%-3x(%4d)", tsgId, tsgId);
        dprintf("       Len=0x%-3x", tsgLen);
    }
    else
    {
        LwU32 chanId   = DRF_VAL(_RAMRL, _ENTRY, _CHID, entry[0]);

        dprintf("CHAN        0x%-3x(%4d)",chanId, chanId);
    }
}

/*!
 * Fetches the runlist for the specified runlistId.
 */
LW_STATUS fifoDumpEngRunlistById_GK104(LwU32 runlistId)
{
    LwU32     runlistLength = 0;
    LwU32     tgtAperture   = 0;
    LwU64     engRunlistPtr = 0;
    LwU32    *runlistBuffer;
    LwU32     i;
    LW_STATUS status;
    LwU32     rlEntrySize = pFifo[indexGpu].fifoRunlistGetEntrySizeLww();

    status = pFifo[indexGpu].fifoReadRunlistInfo(runlistId, &engRunlistPtr, &runlistLength,
                                                 &tgtAperture, &runlistBuffer);
    if (LW_OK != status)
    {
        dprintf("**ERROR: Could not read runlist info\n");
        return status;
    }

    dprintf("   Runlist Base Addr     : " LwU64_FMT "\n", engRunlistPtr);
    dprintf("   Max Runlist Entries   : 0x%x (%d)\n", runlistLength, runlistLength);
    dprintf("   ***runlist contents***\n");
    dprintf("   Entry       TYPE        ID                FIELDS\n");
    dprintf("   -----       ----        -r-\n");

    for (i = 0; i < runlistLength; i++)
    {
        LwU32 *entry = &runlistBuffer[(i * rlEntrySize)/sizeof(LwU32)];

        dprintf("   0x%-10x", i);

        pFifo[indexGpu].fifoPrintRunlistEntry(0, entry);
        dprintf("\n");
    }

    free(runlistBuffer);
    return LW_OK;
}

/*!
 * Checks if chid is on runlist. Result returned in *isOnRunlist.
 * @return LW_OK/ERROR
 */
LW_STATUS fifoIsChannelOnRunlist_GK104( LwU32 chid, BOOL *isOnRunlist )
{
    LwU32 i;
    LwU32 *runListsEngines;
    dprintf("Searching channel in Per Engine Runlists...\n");

    runListsEngines = pFifo[indexGpu].fifoGetRunlistsEngines();
    if (NULL == runListsEngines)
    {
        return LW_ERR_NO_MEMORY;
    }
    for (i=0; i < pFifo[indexGpu].fifoGetRunlistMaxNumber(); i++)
    {
        if (runListsEngines[i] != 0)
        {
            if (isChannelOnRunlistEngId_GK104(i, chid))
            {
                *isOnRunlist = TRUE;
                return LW_OK;
            }
        }
    }

    return LW_OK;
}

static BOOL isChannelOnRunlistEngId_GK104(LwU32 runlistId, LwU32 chid)
{
    LwU32       runlistLength = 0;
    LwU64       engRunlistPtr = 0;
    LwU32*       runlistBuffer;
    LwU32        tgtAperture;
    LW_STATUS        status;
    LwU32       i;
    LwU32*       lwrrEntry = NULL;
    LwU32       rlEntrySize = pFifo[indexGpu].fifoRunlistGetEntrySizeLww();

    dprintf("Searching channel in Runlist for runlistId : %d ...\n", runlistId);

    status = pFifo[indexGpu].fifoReadRunlistInfo(runlistId, &engRunlistPtr, &runlistLength, &tgtAperture, &runlistBuffer);
    if ( status != LW_OK )
    {
        dprintf("**ERROR: Could not read runlist info\n");
        return status;
    }

    dprintf("   Runlist Base Addr     : " LwU64_FMT "\n", engRunlistPtr);
    dprintf("   Max Runlist Entries   : 0x%x (%d)\n", runlistLength, runlistLength);

    for (i=0; i<runlistLength; i++)
    {
        lwrrEntry = (LwU32*)(&runlistBuffer[(i * rlEntrySize)/sizeof(LwU32)]);
        if (pFifo[indexGpu].fifoIsChidInEntry(lwrrEntry, chid))
        {
           dprintf("Found channel!\n");
           break;
        }
    }

    free(runlistBuffer);
    return status;
}


/*!
 * Dump status information about each active engine.
 */
void fifoDumpEngineStatus_GK104()
{
    LwU32 i;
    LwU32 val;
    LwU32 engineIdType;
    LwU32 engineStatusSize = pFifo[indexGpu].fifoGetNumEng();

    for(i=0;i<engineStatusSize;i++)
    {
        val = GPU_REG_RD32(LW_PFIFO_ENGINE_STATUS(i));

        if(SF_VAL(_PFIFO, _ENGINE_STATUS_CTX_STATUS, val) == LW_PFIFO_ENGINE_STATUS_CTX_STATUS_VALID)
        {
            dprintf("ENGINE NUMBER %d\n",i);
            engineIdType = SF_VAL(_PFIFO, _ENGINE_STATUS_ID_TYPE, val);
            if ( engineIdType == LW_PFIFO_ENGINE_STATUS_ID_TYPE_CHID )
            {
                dprintf("   + LW_PFIFO_ENGINE_STATUS_ID_TYPE_CHID\n");
            }
            else if ( engineIdType == LW_PFIFO_ENGINE_STATUS_ID_TYPE_TSGID )
            {
                dprintf("   + LW_PFIFO_ENGINE_STATUS_ID_TYPE_TSGID\n");
            }

            dprintf("   + DIRECT STATUS:                    %d\n", SF_VAL(_PFIFO, _ENGINE_STATUS_ENGINE, val));
            dprintf("   + CHANNEL ID:                       %03d\n", SF_VAL(_PFIFO, _ENGINE_STATUS_ID, val));
        }
    }
}

/*!
 * Dumps the channel ram register info for active chid
 */
LW_STATUS fifoDumpChannelRamRegs_GK104(LwS32 runlistIdPar)
{
    LwU32 regRead, val;
    LwU32 displayCount, chid;
    ChannelId channelId;
    ChannelInst channelInst;
    BOOL bIsBind = FALSE;
    BOOL bIsBusy = FALSE;
    const LwU32 numChannels = pFifo[indexGpu].fifoGetNumChannels(RUNLIST_ALL);
    const char* channelStatus;
    targetMem[LW_PCCSR_CHANNEL_INST_TARGET_VID_MEM].strName = "VID";
    targetMem[LW_PCCSR_CHANNEL_INST_TARGET_VID_MEM].value =
        LW_PCCSR_CHANNEL_INST_TARGET_VID_MEM;

    targetMem[LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_COHERENT].strName = "SYS_CO";
    targetMem[LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_COHERENT].value =
        LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_COHERENT;

    targetMem[LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_NONCOHERENT].strName = "SYS_NON";
    targetMem[LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_NONCOHERENT].value =
        LW_PCCSR_CHANNEL_INST_TARGET_SYS_MEM_NONCOHERENT;

    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    for (chid = 0, displayCount = 0; chid < numChannels; chid++)
    {
        channelId.id = chid;
        pFifo[indexGpu].fifoGetChannelInstForChid(&channelId, &channelInst);

        if (!(channelInst.state & LW_PFIFO_CHANNEL_STATE_ENABLE))
        {
            continue;
        }

        if ((displayCount % 10) == 0)
        {
            dprintf("\n");
            dprintf("%-4s %-18s %-12s %-12s %-12s %-30s\n",
                    "chId", "PTR",  "TARGET",  "BIND", "BUSY", "STATUS");
            dprintf("%-4s %-18s %-12s %-12s %-12s %-30s\n",
                    "----", "---",  "------",  "----", "----", "------");
        }

        if (targetMem[channelInst.target].strName == NULL)
        {
            targetMem[channelInst.target].strName = "?";
        }

        bIsBind = (channelInst.state & LW_PFIFO_CHANNEL_STATE_BIND) != 0;
        bIsBusy = (channelInst.state & LW_PFIFO_CHANNEL_STATE_BUSY) != 0;

        dprintf("%-4d " LwU64_FMT, chid, channelInst.instPtr);
        dprintf(" %-12s %-12s %-12s", targetMem[channelInst.target].strName,
            trueFalse[bIsBind].strName,
            trueFalse[bIsBusy].strName);

        regRead = GPU_REG_RD32(LW_PCCSR_CHANNEL(pFifo[indexGpu].fifoGetSchidForVchid(chid)));
        val = DRF_VAL(_PCCSR, _CHANNEL, _STATUS, regRead);
        switch(val)
        {
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _IDLE);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _PENDING);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _PENDING_CTX_RELOAD);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _PENDING_ACQUIRE);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _PENDING_ACQ_CTX_RELOAD);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _ON_PBDMA);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _ON_PBDMA_AND_ENG);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _ON_ENG);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _ON_ENG_PENDING_ACQUIRE);
            REG_CASE_CHANNEL_STATUS(channelStatus, _PCCSR, _CHANNEL, _STATUS, _ON_ENG_PENDING);
            default:
                channelStatus = "_UNKNOWN";
        }
        dprintf(" %-30s\n", channelStatus);

        displayCount++;
    }

    if (displayCount == 0)
    {
        dprintf("Could not find channels IN_USE\n");
    }
    return LW_OK;
}

/*!
 * Gets information about interrupts for a pbdma id.
 */
void fifoDumpPerPbdmaIntrInfo_GK104(LwU32 pbdmaId)
{
    // INTR_0
    LwU32 regRead0,regReadEn0,regRead,regReadEn;
    char buffer[FIFO_REG_NAME_BUFFER_LEN];
    buffer[0] = '\0';
    regReadEn0 = GPU_REG_RD32(LW_PPBDMA_INTR_EN_0(pbdmaId));
    regRead0 = GPU_REG_RD32(LW_PPBDMA_INTR_0(pbdmaId));
    dprintf("       LW_PPBDMA_INTR_EN_0             : 0x%08x\n", regReadEn0);
    dprintf("       LW_PPBDMA_INTR_0                : 0x%08x\n", regRead0);
    dprintf("\n");

    sprintf(buffer, "LW_PPBDMA_INTR_EN_0*(%d)", pbdmaId);
    priv_dump(buffer);

    sprintf(buffer, "LW_PPBDMA_INTR_0*(%d)", pbdmaId);
    priv_dump(buffer);

    // INTR_1
    regReadEn = GPU_REG_RD32(LW_PPBDMA_INTR_EN_1(pbdmaId));
    regRead = GPU_REG_RD32(LW_PPBDMA_INTR_1(pbdmaId));
    dprintf("       LW_PPBDMA_INTR_EN_1             : 0x%08x\n", regReadEn);
    dprintf("       LW_PPBDMA_INTR_1                : 0x%08x\n", regRead);
    dprintf("\n");

    sprintf(buffer, "LW_PPBDMA_INTR_EN_1*(%d)", pbdmaId);
    priv_dump(buffer);

    sprintf(buffer, "LW_PPBDMA_INTR_1*(%d)", pbdmaId);
    priv_dump(buffer);

}

void fifoCheckEngStates_GK104(gpu_state* pGpuState)
{
    LwU32 i;
    LwU32 regValue;
    LwU32 engineStatusSize = pFifo[indexGpu].fifoGetNumEng();

    for(i=0; i<engineStatusSize; i++)
    {
        regValue = GPU_REG_RD32(LW_PFIFO_ENGINE_STATUS(i));
        if(SF_VAL(_PFIFO_ENGINE_STATUS, _ENGINE, regValue) == LW_PFIFO_ENGINE_STATUS_ENGINE_BUSY)
        {
            switch (i)
            {
                case LW_PFIFO_ENGINE_GRAPHICS:
                    dprintf("Gr busy\n");
                    pGpuState->busyGr = TRUE;
                    break;
                case LW_PFIFO_ENGINE_COPY0:
                    dprintf("Ce0 busy\n");
                    pGpuState->busyCe0 = TRUE;
                    break;
                case LW_PFIFO_ENGINE_COPY1:
                    dprintf("Ce1 busy\n");
                    pGpuState->busyCe1 = TRUE;
                    break;
                case LW_PFIFO_ENGINE_GRCOPY:
                    dprintf("Ce2 busy\n");
                    pGpuState->busyCe2 = TRUE;
                    break;
                case LW_PFIFO_ENGINE_MSPDEC:
                    dprintf("Mspdec busy\n");
                    pGpuState->busyMspdec = TRUE;
                    break;
                case LW_PFIFO_ENGINE_MSPPP:
                    dprintf("Msppp busy\n");
                    pGpuState->busyMsppp = TRUE;
                    break;
                case LW_PFIFO_ENGINE_MSVLD:
                    dprintf("Msvld busy\n");
                    pGpuState->busyMsvld = TRUE;
                    break;
                case LW_PFIFO_ENGINE_MSENC:
                    dprintf("Msenc busy\n");
                    pGpuState->busyMsenc = TRUE;
                    break;
                default:
                    break;
            }
        }
    }
    dprintf("\n");
}

void fifoDumpEngStates_GK104(ChannelId *pChannelId, gpu_state* pGpuState)
{
    LwU32 buf;
    LwU64 instMemAddr = 0;
    LwU64 ctxSpace = 0;
    LwU64 lowBits;
    LwU64 highBits;
    readFn_t readFn = NULL;
    BOOL bFillState = !(pGpuState == NULL);    //for gpuAnalyze

    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId, &readFn, NULL, NULL);
    if (readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return;
    }

    dprintf("\n");

    DUMP_ENG_STATE( GR, busyGr);
    DUMP_ENG_STATE( MSPDEC, busyMspdec );
    DUMP_ENG_STATE( MSPPP, busyMsppp );
    DUMP_ENG_STATE( MSVLD, busyMsvld );
    DUMP_ENG_STATE( MSENC, busyMsenc );

    dprintf("\n");

    pFifo[indexGpu].fifoDumpEngineStatus();
}

/**
 * @brief Get the table entry for engine from device info
 *
 * @param[in]  engNames
 * @param[in]  deviceInfoType
 * @param[in]  pDeviceInfoData
 * @param[in]  bDataValid
 */
LwU32
fifoGetTableEntry_GK104
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
        if (engNames[tblIdx].nameValue.value == deviceInfoType)
        {
            return tblIdx;
        }
        tblIdx++;
    }
    return tblIdx;
}

/**
 * Get all the entries from device info.
 * Read LW_PTOP_DEVICE_INFO to map runlist IDs to engines.
 * This is a very simplified version of fifoConstructEngineList_v02_01() in RM.
 */
LW_STATUS fifoGetDeviceInfo_GK104(void)
{
    LwU32  i, tblIdx;
    LwU32  engineIndex = 0;
    LwBool bDataValid = LW_FALSE;
    LwU32  deviceInfoData = 0;
    LwU32  deviceInfoType = DEVICE_INFO_TYPE_ILWALID;
    LwU32  deviceInfoRunlist = DEVICE_INFO_RUNLIST_ILWALID;
    LwU32  pbdmaMap[LW_HOST_MAX_NUM_PBDMA];
    LwU32  grPbdmaMask = 0xFFFFFFFF;
    LwU32  pbdmaClearMask = 0;
    LwU32  numPbdma;
    LwU32 *runListsEngines;
    LW_STATUS status;
    EngineNameValue *engNames;

    if (deviceInfo.bInitialized)
    {
        return LW_OK;
    }
    status = deviceInfoAlloc();
    if (LW_OK != status)
    {
        return status;
    }
    numPbdma = pFifo[indexGpu].fifoGetNumPbdma();
    runListsEngines = (LwU32*)pFifo[indexGpu].fifoGetRunlistsEngines();
    if (NULL == runListsEngines)
    {
        return LW_ERR_NO_MEMORY;
    }
    engNames = (EngineNameValue*)pFifo[indexGpu].fifoGetEngNames();

    for (i = 0; i < numPbdma; i++)
    {
        pbdmaMap[i] = GPU_REG_RD32(LW_PFIFO_PBDMA_MAP(i));
    }
    deviceInfo.cfg.version = 1;
    for (i = 0; i < pFifo[indexGpu].fifoGetDeviceInfoNumRows(); ++i)
    {
        LwU32 tableEntry = GPU_REG_RD32(LW_PTOP_DEVICE_INFO(i));

        //
        // Save the device info entry
        //
        deviceInfo.pRows[i].value = tableEntry;
        deviceInfo.pRows[i].bInChain = FLD_TEST_DRF(_PTOP, _DEVICE_INFO, _CHAIN, _ENABLE, tableEntry);
        deviceInfo.pRows[i].type = getDeviceInfoRowType(DRF_VAL(_PTOP, _DEVICE_INFO, _ENTRY, tableEntry));
        // Clear the bottom 2 bits (_VALUE overlaps _ENTRY).
        deviceInfo.pRows[i].data = DRF_VAL(_PTOP, _DEVICE_INFO, _VALUE, tableEntry) &
                                 DRF_SHIFTMASK(LW_PTOP_DEVICE_INFO_DATA);
        if (FLD_TEST_DRF(_PTOP, _DEVICE_INFO, _ENTRY, _NOT_VALID, tableEntry))
        {
            deviceInfo.pRows[i].bValid = LW_FALSE;
            continue;
        }
        deviceInfo.pRows[i].bValid = LW_TRUE;

        switch (DRF_VAL(_PTOP, _DEVICE_INFO, _ENTRY, tableEntry))
        {
            case LW_PTOP_DEVICE_INFO_ENTRY_DATA:
                deviceInfoData = tableEntry;
                bDataValid = LW_TRUE;
                break;

            case LW_PTOP_DEVICE_INFO_ENTRY_ENUM:
                deviceInfoRunlist                                             = DRF_VAL(_PTOP, _DEVICE_INFO, _RUNLIST_ENUM, tableEntry);
                deviceInfo.pEngines[engineIndex].bHostEng                              = FLD_TEST_DRF(_PTOP, _DEVICE_INFO, _ENGINE, _VALID, tableEntry);
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_FIFO_TAG] = DRF_VAL(_PTOP, _DEVICE_INFO, _ENGINE_ENUM, tableEntry);
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_RUNLIST]  = deviceInfoRunlist;
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_RESET]    = DRF_VAL(_PTOP, _DEVICE_INFO, _RESET_ENUM, tableEntry);
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_INTR]     = DRF_VAL(_PTOP, _DEVICE_INFO, _INTR_ENUM, tableEntry);
                break;

            case LW_PTOP_DEVICE_INFO_ENTRY_ENGINE_TYPE:
                deviceInfoType = DRF_VAL(_PTOP, _DEVICE_INFO, _TYPE_ENUM, tableEntry);
                break;

            default:
                break;
        }

        if (FLD_TEST_DRF(_PTOP, _DEVICE_INFO, _CHAIN, _DISABLE, tableEntry))
        {
            tblIdx = pFifo[indexGpu].fifoGetTableEntry(engNames,
                                                       deviceInfoType,
                                                       pFifo[indexGpu].fifoGetInstanceIdFromDeviceInfoData(deviceInfoData),
                                                       bDataValid);

            if (engNames[tblIdx].nameValue.value != DEVICE_INFO_TYPE_ILWALID)
            {
                LwU32 idx, pbdmaMask = 0;

                deviceInfo.pEngines[engineIndex].engineName = engNames[tblIdx].nameValue.strName;
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_ENUM] = deviceInfoType;
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_INST_ID] = engNames[tblIdx].instanceId;

                if (deviceInfo.pEngines[engineIndex].bHostEng)
                {
                    for(idx = 0; idx < numPbdma; idx++)
                    {
                        if (BIT(deviceInfoRunlist) & pbdmaMap[idx])
                        {
                            pbdmaMask |= BIT(idx);
                        }
                    }
                }

                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_PBDMA_MASK] = pbdmaMask;
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_ENGINE_TYPE] = engNames[tblIdx].nameValue.value;
                deviceInfo.pEngines[engineIndex].engineData[ENGINE_INFO_TYPE_ENGINE_TAG] = engNames[tblIdx].engineTag;

                if (engNames[tblIdx].engineTag == ENGINE_TAG_GR)
                {
                    grPbdmaMask = pbdmaMask;
                }

                if(deviceInfo.pEngines[engineIndex].bHostEng)
                {
                    if(deviceInfoRunlist < pFifo[indexGpu].fifoGetRunlistMaxNumber())
                    {
                        runListsEngines[deviceInfoRunlist] |= LWBIT(tblIdx);
                    }
                    else
                    {
                        dprintf("**ERROR: LW_PTOP_DEVICE_INFO_RUNLIST_ENUM exceeds %d\n",
                                pFifo[indexGpu].fifoGetRunlistMaxNumber() - 1);
                    }
                }

                engineIndex++;
                deviceInfo.enginesCount = engineIndex;
            }

            bDataValid = LW_FALSE;
            deviceInfoType = DEVICE_INFO_TYPE_ILWALID;
            deviceInfoRunlist = DEVICE_INFO_RUNLIST_ILWALID;
        }
    }

    /**
     * Corrects CE entries so that only the pbdma that feeds the CE is in the mask.
     *
     * Pbdma masks are generated by looking at the runlist for a given engine without taking
     * the runqueue into account. Thus, CEs may report multiple pbdmas even though the CE to
     * pbdma mapping is one to one. We go through all the CE entries to remove the extra
     * pbdmas from the CE pbdma mask. Hardware guarantees that lower CEs correspond to lower
     * pbdmas.
     */
    for (i = 0; i < engineIndex; i++)
    {
        DeviceInfoEngine *pEngine = &deviceInfo.pEngines[i];

        if (pEngine->engineData[ENGINE_INFO_TYPE_ENGINE_TAG] == ENGINE_TAG_CE)
        {
            LwU32 pbdmaMask = pEngine->engineData[ENGINE_INFO_TYPE_PBDMA_MASK];

            if (pbdmaMask == grPbdmaMask)
            {
                // Zero out all previously seen pbdmas.
                // Add the current one to the clear mask.
                // Set the tables entry to the lowest pbdma bit of the remaining pbdmas
                pbdmaMask &= ~pbdmaClearMask;
                pbdmaMask = LOWESTBIT(pbdmaMask);
                pbdmaClearMask |= pbdmaMask;
                pEngine->engineData[ENGINE_INFO_TYPE_PBDMA_MASK] = pbdmaMask;
            }
        }

        // Use the final PBDMA mask to fill the PBDMA ID / fault ID array
        if (pEngine->bHostEng)
        {
            LwU32 pbdmaIds[32];
            LwU32 pbdmaId = 0;
            LwU32 pbdmaCount = 0;
            LwU32 pbdmaMask = pEngine->engineData[ENGINE_INFO_TYPE_PBDMA_MASK];
            size_t pPbdmaIdsAllocSize;

            while (pbdmaMask)
            {
                if (pbdmaMask & (1 << pbdmaId)) {
                    pbdmaIds[pbdmaCount] = pbdmaId;

                    // Clear the bit
                    pbdmaMask ^= (1 << pbdmaId);

                    pbdmaCount += 1;
                }

                pbdmaId += 1;
            }

            pPbdmaIdsAllocSize = sizeof(*pEngine->pPbdmaIds) * pbdmaCount;
            pEngine->numPbdmas = pbdmaCount;
            pEngine->pPbdmaIds = malloc(pPbdmaIdsAllocSize);
            if (NULL == pEngine->pPbdmaIds)
            {
                return LW_ERR_NO_MEMORY;
            }
            pEngine->pPbdmaFaultIds = NULL;

            memcpy(pEngine->pPbdmaIds, pbdmaIds, pPbdmaIdsAllocSize);
        }
    }

    deviceInfo.bInitialized = LW_TRUE;

    return LW_OK;
}

/**
 * Get all the entries from device info
 */
LwU32 fifoRunlistGetEntrySizeLww_GK104(void)
{
    return LW_RAMRL_ENTRY_SIZE;
}


//-----------------------------------------------------
// PBDMA_CACHE1(n,id)
//
// This macro is used by fifoDumpPbdmaRegsCache1_GK104()
// to dump out LW_PBDMA_METHOD/DATA0/1/2/3 info.
//-----------------------------------------------------
#define PBDMA_CACHE1(n,id)                                  \
do{                                                         \
    regRead = GPU_REG_RD32(LW_PPBDMA_METHOD##n(id));        \
    incr = DRF_VAL(_PPBDMA_METHOD, n, _INCR, regRead);      \
                                                            \
    priv = DRF_VAL(_PPBDMA_METHOD, n, _PRIV, regRead);      \
                                                            \
    addr = DRF_VAL(_PPBDMA_METHOD, n, _ADDR, regRead);      \
                                                            \
    subch = DRF_VAL(_PPBDMA_METHOD, n, _SUBCH, regRead);    \
                                                            \
    first = DRF_VAL(_PPBDMA_METHOD, n, _FIRST, regRead);    \
                                                            \
    dual = DRF_VAL(_PPBDMA_METHOD, n, _DUAL, regRead);      \
                                                            \
    valid = DRF_VAL(_PPBDMA_METHOD, n, _VALID, regRead);    \
    regRead = GPU_REG_RD32(LW_PPBDMA_DATA##n(id));          \
    value = DRF_VAL(_PPBDMA_DATA, n, _VALUE, regRead);      \
                                                            \
    dprintf("%d    %s    %s   0x%04x    %d      ",          \
                  n, pbdmaTrueFalse[incr].strName, privUserKernel[priv].strName, addr << 2, subch);             \
    dprintf("%s     %s    %s", pbdmaTrueFalse[first].strName,                                                   \
                               pbdmaTrueFalse[dual].strName, pbdmaTrueFalse[valid].strName);                    \
    dprintf("    0x%08x\n", value);                                                                             \
}while(0)


//-----------------------------------------------------
// fifoDumpPbdmaRegsCache1_GK104(LwU32 pbdmaId)
//
// This function dumps out pbdma cache1 regs for given pbdmaId
// Uses DUMP_METHOD and DUMP_DATA macros
//-----------------------------------------------------
void fifoDumpPbdmaRegsCache1_GK104(LwU32 pbdmaId)
{
    LwU32 regRead;
    LwU32 incr;
    LwU32 priv;
    LwU32 addr;
    LwU32 subch;
    LwU32 first;
    LwU32 dual;
    LwU32 valid;
    LwU32 value;

    NameValue privUserKernel[2] =
    {
        {" User ",0x0},
        {"Kernel",0x1}
    };

    NameValue pbdmaTrueFalse[2] =
    {
        {"False",0x0},
        {"True ",0x1}
    };


    dprintf("\n");
    dprintf(" PBDMA %d CACHE1 :  \n", pbdmaId);
    dprintf("      -------------------------- METHOD -------------------------   ---DATA---\n");
    dprintf("N     INCR     PRIV     ADDR    SUBCH    FIRST     DUAL     VALID      VALUE \n");
    dprintf("--    ----     ----     ----    -----    -----     -----    -----      -----\n");
    PBDMA_CACHE1(0, pbdmaId);
    PBDMA_CACHE1(1, pbdmaId);
    PBDMA_CACHE1(2, pbdmaId);
    PBDMA_CACHE1(3, pbdmaId);

}

LW_STATUS
fifoReadRunlistInfo_GK104
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
    LwU32 engRunlist     = GPU_REG_RD32(LW_PFIFO_ENG_RUNLIST(runlistId));
    LwU32 engRunlistBase = GPU_REG_RD32(LW_PFIFO_ENG_RUNLIST_BASE(runlistId));

    engRunlistPtr = (LwU64)DRF_VAL(_PFIFO, _ENG_RUNLIST_BASE, _PTR, engRunlistBase);
    engRunlistPtr <<= LW_RAMRL_BASE_SHIFT;
    runlistLength = DRF_VAL(_PFIFO, _ENG_RUNLIST, _LENGTH, engRunlist);
    tgtAperture = DRF_VAL(_PFIFO, _ENG_RUNLIST_BASE, _TARGET, engRunlistBase);

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
            dprintf("**LW_ERR_GENERICOR: Could not fetch runlist info\n");
            return status;
        }
    }

    return LW_OK;
}

LW_STATUS
fifoValidateRunlistInfoForAlloc_GK104
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

    if (engRunlistPtr == LW_PFIFO_ENG_RUNLIST_BASE_PTR_NULL)
    {
        dprintf("**ERROR: LW_PFIFO_ENG_RUNLIST_BASE_PTR_NULL\n");
        return LW_ERR_GENERIC;
    }

    if (runlistLength == LW_PFIFO_ENG_RUNLIST_LENGTH_ZERO)
    {
        dprintf("**ERROR: LW_PFIFO_ENG_RUNLIST_LENGTH_ZERO\n");
        return LW_ERR_GENERIC;
    }

    if (pReadFunc == NULL)
    {
        dprintf("**ERROR: LW_PFIFO_ENG_RUNLIST_ALLOCATOR FUNCTION PTR is NULL\n");
        return LW_ERR_GENERIC;
    }

    switch (tgtAperture)
    {
        case LW_PFIFO_ENG_RUNLIST_BASE_TARGET_VID_MEM :
            *pReadFunc = pFb[indexGpu].fbRead;
        break;

        case LW_PFIFO_ENG_RUNLIST_BASE_TARGET_SYS_MEM_COHERENT :
            *pReadFunc = readSystem;
        break;

        case LW_PFIFO_ENG_RUNLIST_BASE_TARGET_SYS_MEM_NONCOHERENT :
            *pReadFunc = readSystem;
        break;

        default :
            dprintf("**ERROR: Invalid target aperture for runlist\n");
            return LW_ERR_GENERIC;
    }

    return LW_OK;
}

LW_STATUS
fifoXlateFromDevTypeAndInstId_GK104
(
    LwU32  devType,
    LwU32  instId,
    LwU32  outType,
    LwU32 *pOutVal
)
{
    LwU32 idxEngine;
    LW_STATUS status = pFifo[indexGpu].fifoGetDeviceInfo();

    if (status != LW_OK)
    {
        dprintf("**ERROR: failed to parse device info\n");
        return status;
    }

    // check for validity of outType
    if (outType >= ENGINE_INFO_TYPE_ILWALID)
    {
        dprintf("**ERROR: outType %d exceeds the available engine info types\n", outType);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (pOutVal == NULL)
    {
        dprintf("**ERROR: invalid pointer. pOutVal is NULL\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    for (idxEngine = 0; idxEngine < deviceInfo.enginesCount; idxEngine++)
    {
        if ((deviceInfo.pEngines[idxEngine].engineData[ENGINE_INFO_TYPE_ENGINE_TYPE] == devType) &&
            (deviceInfo.pEngines[idxEngine].engineData[ENGINE_INFO_TYPE_INST_ID] == instId))
        {
            *pOutVal = deviceInfo.pEngines[idxEngine].engineData[outType];
            return LW_OK;
        }
    }

    dprintf("**ERROR: Failed to find entry for specified (dev type = 0x%x, inst Id = %d) combination\n",
        devType, instId);
    return LW_ERR_OBJECT_NOT_FOUND;
}

LW_STATUS
fifoXlateFromEngineString_GK104
(
    char*  engName,
    LwU32  outType,
    LwU32 *pOutVal
)
{
    LwU32 tableIndex;
    EngineNameValue *engNames;

    // check for validity of outType
    if (outType >= ENGINE_INFO_TYPE_ILWALID)
    {
        dprintf("**ERROR: outType %d exceeds the available engine info types\n", outType);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (pOutVal == NULL)
    {
        dprintf("**ERROR: invalid pointer. pOutVal is NULL\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    engNames = pFifo[indexGpu].fifoGetEngNames();

    for (tableIndex = 0; engNames[tableIndex].nameValue.value != DEVICE_INFO_TYPE_ILWALID; tableIndex++)
    {
        if (strcasecmp(engNames[tableIndex].nameValue.strName, engName) == 0)
        {
            break;
        }
    }

    if (engNames[tableIndex].nameValue.value == DEVICE_INFO_TYPE_ILWALID)
    {
        dprintf("**ERROR: Failed to find entry for specified engine string %s\n", engName);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    return pFifo[indexGpu].fifoXlateFromDevTypeAndInstId(engNames[tableIndex].nameValue.value, // dev_type
                                                         engNames[tableIndex].instanceId,      // inst_id
                                                         outType, pOutVal);
}

/**
 * Goes through the engines looking for the first one whose data of type
 * @p dataTypeIn is @p searchedVal, returning @p pOutputVal engnine's data of
 * type dataTypeOut.
 */
LW_STATUS
fifoEngineDataXlate_GK104
(
    ENGINE_INFO_TYPE dataTypeIn,
    LwU32            searchedVal,
    ENGINE_INFO_TYPE dataTypeOut,
    LwU32           *pOutputVal
)
{
    LwU32 idxEngine;

    if (NULL == pOutputVal)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (dataTypeOut == ENGINE_INFO_TYPE_PBDMA_ID)
    {
        //
        // PBDMA_ID can only be a search parameter as one engine can have
        // multiple PBDMAs
        //
        return LW_ERR_ILWALID_ARGUMENT;
    }

    for (idxEngine = 0; idxEngine < deviceInfo.enginesCount; idxEngine++)
    {
        if (dataTypeIn == ENGINE_INFO_TYPE_PBDMA_ID)
        {
            LwU32 idxPbdma;
            for (idxPbdma = 0;
                 idxPbdma < deviceInfo.pEngines[idxEngine].numPbdmas;
                 idxPbdma++)
            {
                if (deviceInfo.pEngines[idxEngine].pPbdmaIds[idxPbdma] == searchedVal)
                {
                    *pOutputVal = deviceInfo.pEngines[idxEngine].engineData[dataTypeOut];
                    return LW_OK;
                }
            }
        }
        else if (deviceInfo.pEngines[idxEngine].engineData[dataTypeIn] == searchedVal)
        {
            if (dataTypeOut == ENGINE_INFO_TYPE_ILWALID)
            {
                *pOutputVal = idxEngine;
            }
            else
            {
                *pOutputVal = deviceInfo.pEngines[idxEngine].engineData[dataTypeOut];
            }
            return LW_OK;
        }
    }
    return LW_ERR_OBJECT_NOT_FOUND;
}

LwU32
fifoGetSchidForVchid_GK104
(
    LwU32   vChid
)
{
    // Pre-TURING doesn't have SRIOV, so directly return Vchid as Schid
    return vChid;
}
