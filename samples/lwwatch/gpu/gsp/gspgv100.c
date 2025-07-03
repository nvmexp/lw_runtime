/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "gsp.h"
#include "hal.h"
#include "lwsym.h"

#include "g_gsp_private.h"

#include "volta/gv100/dev_gsp.h"
#include "volta/gv100/dev_master.h"


// TODO: replace with proper define once it is known
#define GSP_MUTEX_EMEM 0

static void _gspEmemWriteToEmemOffset(LwU32 off, LwU32 val, LwU32 width, LwU32 port);
static LW_STATUS _gspMutexIdGen(LwU32 *pOwnerId);
static void _gspMutexIdRel(LwU32 ownerId);
static LW_STATUS _gspMutexAcquireByIndex(LwU32 physMutexId, LwU32 *pOwnerId);
static void _gspMutexReleaseByIndex(LwU32 physMutexId, LwU32 ownerId);
const char* gspGetEngineName_GV100(void);
static const char* gspGetSymFilePath(void);

static FLCN_ENGINE_IFACES flcnEngineIfaces_gsp =
{
    gspGetFalconCoreIFace_STUB,         // flcnEngGetCoreIFace
    gspGetFalconBase_STUB,              // flcnEngGetFalconBase
    gspGetEngineName_GV100,             // flcnEngGetEngineName
    gspUcodeName_STUB,                  // flcnEngUcodeName
    gspGetSymFilePath,            // flcnEngGetSymFilePath
    gspQueueGetNum_STUB,                // flcnEngQueueGetNum
    gspQueueRead_STUB,                  // flcnEngQueueRead
    gspGetDmemAccessPort,               // flcnEngGetDmemAccessPort
    gspIsDmemRangeAccessible_STUB,      // flcnEngIsDmemRangeAccessible
    gspEmemGetSize_STUB,                // flcnEngEmemGetSize
    gspEmemGetOffsetInDmemVaSpace_STUB, // flcnEngEmemGetOffsetInDmemVaSpace
    gspEmemGetNumPorts_STUB,            // flcnEngEmemGetNumPorts
    gspEmemRead_STUB,                   // flcnEngEmemRead
    gspEmemWrite_STUB,                  // flcnEngEmemWrite
};

const char* gspGetEngineName_GV100(void)
{
    return "GSPLITE";
}

static const char* gspGetSymFilePath(void)
{
    return DIR_SLASH "gsplite" DIR_SLASH "bin";
}

LW_STATUS gspFillSymPath_GV100(OBJFLCN *gspFlcn)
{
    sprintf(gspFlcn[indexGpu].symPath, "%s%s", LWSYM_VIRUTAL_PATH, "gsplite/");
    gspFlcn[indexGpu].bSympathSet = TRUE;
    return LW_OK;
}

LwU32 gspQueueGetNum_GV100()
{
    return LW_PGSP_QUEUE_HEAD__SIZE_1 + LW_PGSP_MSGQ_HEAD__SIZE_1;
}

/*!
 *  Read the contents of a specific queue into queue. Message queue Id comes
 *  sequentially after the command queues. pQueue->id will be filled out
 *  automatically as well.
 *
 *  @param queueId Id of queue to get data for. If invalid, then this function
 *                 will return FALSE.
 *  @param pQueue  Pointer to queue structure to fill up.
 *
 *  @return FALSE if queueId is invalid or queue is NULL; TRUE on success.
 */
LwBool
gspQueueRead_GV100
(
    LwU32         queueId,
    PFLCN_QUEUE   pQueue
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pGsp[indexGpu].gspGetFalconEngineIFace();
    const FLCN_CORE_IFACES   *pFCIF =
        pGsp[indexGpu].gspGetFalconCoreIFace();
    LwU32 engineBase = pFEIF->flcnEngGetFalconBase();
    LwU32 numQueues;
    LwU32 numCmdQs = LW_PGSP_QUEUE_HEAD__SIZE_1;
    LwU32 sizeInWords;
    LwU32 ememPort;

    numQueues = pGsp[indexGpu].gspQueueGetNum();
    if (queueId >= numQueues || !pQueue)
    {
        return LW_FALSE;
    }

    //
    // The "message" queues comes right after the command queues,
    // so we use a special case to get the information
    //
    if (queueId < LW_PGSP_QUEUE_HEAD__SIZE_1)
    {
        pQueue->head = GPU_REG_RD32(LW_PGSP_QUEUE_HEAD(queueId));
        pQueue->tail = GPU_REG_RD32(LW_PGSP_QUEUE_TAIL(queueId));
    }
    else
    {
        pQueue->head = GPU_REG_RD32(LW_PGSP_MSGQ_HEAD(queueId-numCmdQs));
        pQueue->tail = GPU_REG_RD32(LW_PGSP_MSGQ_TAIL(queueId-numCmdQs));
    }

    //
    // At the momement, we assume that tail <= head. This is not the case since
    // the queue wraps-around. Unfortunatly, we have no way of knowing the size
    // or offsets of the queues, and thus renders the parsing slightly
    // impossible. Lwrrently do not support.
    //
    if (pQueue->head < pQueue->tail)
    {
        dprintf("lw: Queue 0x%x is lwrrently in a wrap-around state.\n",
                queueId);
        dprintf("lw:     tail=0x%04x, head=0x%04x\n", pQueue->tail, pQueue->head);
        dprintf("lw:     It is lwrrently not possible to parse this queue.\n");
        return LW_FALSE;
    }

    sizeInWords    = (pQueue->head - pQueue->tail) / sizeof(LwU32);
    pQueue->length = sizeInWords;
    pQueue->id     = queueId;

    //
    // If the queue happens to be larger than normally allowed, print out an
    // error message and return an error.
    //
    if (sizeInWords >= LW_FLCN_MAX_QUEUE_SIZE)
    {
        dprintf("lw: %s: GSPLITE queue 0x%x is larger than configured to read:\n",
                __FUNCTION__, queueId);
        dprintf("lw:     Queue Size: 0x%x     Supported Size: 0x%x\n",
                (LwU32)(sizeInWords * sizeof(LwU32)), (LwU32)(LW_FLCN_MAX_QUEUE_SIZE * sizeof(LwU32)));
        dprintf("lw:     Make LW_FLCN_MAX_QUEUE_SIZE larger and re-compile LW_WATCH\n");
        return LW_FALSE;
    }

    ememPort = pFCIF->flcnDmemGetNumPorts(engineBase) - 1;

    // Simply read the queue into the buffer if it is initialized
    if (pQueue->tail > pFEIF->flcnEngEmemGetOffsetInDmemVaSpace())
    {
        pFEIF->flcnEngEmemRead(pQueue->tail, sizeInWords, ememPort,
                               pQueue->data);
    }
    return LW_TRUE;
}

LwU32 gspEmemGetNumPorts_GV100(void)
{
    return LW_PGSP_EMEMD__SIZE_1;
}

LwU32 gspEmemGetSize_GV100(void)
{
    return GPU_REG_RD_DRF(_PGSP, _HWCFG, _EMEM_SIZE) * FLCN_BLK_ALIGNMENT;
}

/*!
 *  @return Offset of EMEM in DMEM VA space.
 *          Should be located directly above addressable (virtual) DMEM.
 */
LwU32
gspEmemGetOffsetInDmemVaSpace_GV100()
{
    // START_EMEM = DMEM_VA_MAX = 2^(DMEM_TAG_WIDTH + 8)
    return 1 << (GPU_REG_RD_DRF(_PGSP, _FALCON_HWCFG1, _DMEM_TAG_WIDTH) +
                 FALCON_DMEM_BLKSIZE);
}

const FLCN_ENGINE_IFACES *gspGetFalconEngineIFace_GV100(void)
{
    const FLCN_CORE_IFACES   *pFCIF = pGsp[indexGpu].gspGetFalconCoreIFace();
          FLCN_ENGINE_IFACES *pFEIF = NULL;

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if (pFCIF)
    {
        pFEIF = &flcnEngineIfaces_gsp;

        pFEIF->flcnEngGetCoreIFace               = pGsp[indexGpu].gspGetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase              = pGsp[indexGpu].gspGetFalconBase;
        pFEIF->flcnEngQueueGetNum                = pGsp[indexGpu].gspQueueGetNum;
        pFEIF->flcnEngQueueRead                  = pGsp[indexGpu].gspQueueRead;
        pFEIF->flcnEngIsDmemRangeAccessible      = pGsp[indexGpu].gspIsDmemRangeAccessible;
        pFEIF->flcnEngEmemGetSize                = pGsp[indexGpu].gspEmemGetSize;
        pFEIF->flcnEngEmemGetOffsetInDmemVaSpace = pGsp[indexGpu].gspEmemGetOffsetInDmemVaSpace;
        pFEIF->flcnEngEmemGetNumPorts            = pGsp[indexGpu].gspEmemGetNumPorts;
        pFEIF->flcnEngEmemRead                   = pGsp[indexGpu].gspEmemRead;
        pFEIF->flcnEngEmemWrite                  = pGsp[indexGpu].gspEmemWrite;
    }
    return pFEIF;
}

/*!
 *  @brief Reads data from EMEM
 *
 *  Read length words of EMEM starting at address 'addr'.
 *  Offset will automatically be truncated down to 4-byte aligned value.
 *  If length spans out of the range of the EMEM, it will automatically
 *  be truncated to fit into the EMEM range.
 *
 *  The address must be located in the EMEM region located directly above the
 *  maximum virtual address of DMEM.
 *
 *  @param addr        The address for the source of the copy.
 *  @param length      Number of 4-byte words to read.
 *  @param port        Port to read from.
 *  @param pBuf        Buffer to store EMEM into.
 *
 *  @return 0 on error, or number of 4-byte words read.
 */
LwU32
gspEmemRead_GV100
(
    LwU32 addr,
    LwU32 length,
    LwU32 port,
    LwU32 *pBuf
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pGsp[indexGpu].gspGetFalconEngineIFace();
    LwU32     ememSize    = pFEIF->flcnEngEmemGetSize();
    LwU32     startEmem   = pFEIF->flcnEngEmemGetOffsetInDmemVaSpace();
    LwU32     endEmem     = startEmem + ememSize;
    LwU32     timeoutUs   = GSP_MUTEX_TIMEOUT_US;
    LwU32     ememcOrig   = 0x0;
    LwU32     ememc       = 0x0;
    LwU32     i           = 0x0;
    LwU32     mutexId;
    LW_STATUS status;


    addr &= ~(sizeof(LwU32) - 1);

    // Fail if the port specified is not valid
    if (port >= pGsp[indexGpu].gspEmemGetNumPorts())
    {
        dprintf("lw: Only %d ports supported. Accessed port=%d\n",
                pGsp[indexGpu].gspEmemGetNumPorts(), port);
        return 0;
    }

    //
    // Verify that the address is located in EMEM, above addressable DMEM
    // START_EMEM = DMEM_VA_MAX = 2^(DMEM_TAG_WIDTH + 8)
    //
    if (addr < startEmem || addr >= endEmem)
    {
        dprintf("lw: 0x%0x Not in EMEM aperature [0x%x,0x%x)\n",
                addr, startEmem, endEmem);
        return 0;
    }

    // Truncate the length down to a reasonable size
    if ((addr + (length * sizeof(LwU32))) > endEmem)
    {
        length = (endEmem - addr) / sizeof(LwU32);
        dprintf("lw:\twarning: length truncated to fit into EMEM range.\n");
    }

    // Colwert address from offset in DMEM to offset in EMEM
    addr -= startEmem;

    // Acquire EMEM lock
    do
    {
        status = _gspMutexAcquireByIndex(GSP_MUTEX_EMEM, &mutexId);
        if (status == LW_OK)
        {
            break;
        }
        else if (status != LW_OK && status != LW_ERR_STATE_IN_USE)
        {
            dprintf("lw: error in acquiring EMEM lock (non-timeout)\n");
            return 0;
        }
        osPerfDelay(0x10);
        timeoutUs -= 0x10;
    } while (timeoutUs > 0);
    if (status == LW_ERR_STATE_IN_USE)
    {
        dprintf("lw: timeout in acquiring EMEM lock\n");
        return 0;
    }

    //
    // Build the EMEMC command that auto-increments on each read
    // We take the address and mask it off with OFFSET and BLOCK region.
    //
    // Note: We also remember and restore the original command value
    //
    ememc = addr & (DRF_SHIFTMASK(LW_PGSP_EMEMC_OFFS) |
                    DRF_SHIFTMASK(LW_PGSP_EMEMC_BLK));
    // mark auto-increment on read
    ememc |= DRF_DEF(_PGSP, _EMEMC, _AINCR, _TRUE);

    // Store the original EMEMC command
    ememcOrig = GPU_REG_RD32(LW_PGSP_EMEMC(port));

    // Perform the actual EMEM read operations
    GPU_REG_WR32(LW_PGSP_EMEMC(port), ememc);
    for (i = 0; i < length; i++)
    {
        pBuf[i] = GPU_REG_RD32(LW_PGSP_EMEMD(port));
    }

    // Restore the original EMEMC command
    GPU_REG_WR32(LW_PGSP_EMEMC(port), ememcOrig);

    // Release EMEM lock
    _gspMutexReleaseByIndex(GSP_MUTEX_EMEM, mutexId);

    return length;
}

/*!
 * Write 'val' to EMEM located at DMEM address 'addr'.
 *
 * The address must be located in the EMEM region located directly above the
 * maximum virtual address of DMEM. 'val' may be a byte, half-word, or word
 * based on 'width'. There are no alignment restrictions on the address.
 *
 * A length of 1 will perform a single write.  Lengths greater than one may
 * be used to stream writes multiple times to adjacent locations in DMEM.
 *
 * @param[in]  addr    DMEM address to write 'val'
 * @param[in]  val     Value to write
 * @param[in]  width   width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  length  Number of writes to perform
 * @param[in]  port    EMEM port to use for when reading and writing EMEM
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32 gspEmemWrite_GV100
(
    LwU32 addr,
    LwU32 val,
    LwU32 width,
    LwU32 length,
    LwU32 port
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pGsp[indexGpu].gspGetFalconEngineIFace();
    LwU32     ememSize  = pFEIF->flcnEngEmemGetSize();
    LwU32     startEmem = pFEIF->flcnEngEmemGetOffsetInDmemVaSpace();
    LwU32     endEmem   = startEmem + ememSize;
    LwU32     ememcOrig = 0x0;
    LwU32     timeoutUs = GSP_MUTEX_TIMEOUT_US;
    LwU32     i;
    LwU32     mutexId;
    LW_STATUS status;

    // Fail if the port specified is not valid
    if (port >= pGsp[indexGpu].gspEmemGetNumPorts())
    {
        dprintf("lw: Only %d ports supported. Accessed port=%d\n",
                pGsp[indexGpu].gspEmemGetNumPorts(), port);
        return 0;
    }

    //
    // Verify that the address is located in EMEM, above addressable DMEM
    // START_EMEM = DMEM_VA_MAX = 2^(DMEM_TAG_WIDTH + 8)
    //
    if (addr < startEmem || addr >= endEmem)
    {
        dprintf("lw: 0x%0x Not in EMEM aperature [0x%x,0x%x)\n",
                addr, startEmem, endEmem);
        return 0;
    }

    // Width must be 1, 2, or 4.
    if ((width != 1) && (width != 2) && (width != 4))
    {
        dprintf("lw: Error: Width (%u) must be 1, 2, or 4\n", width);
        return 0;
    }

    // Fail if the write will go out-of-bounds
    if ((addr + (length * width)) > endEmem)
    {
        dprintf("lw: Error: Cannot write to EMEM. Writes go out-of-"
                "bounds.\n");
        return 0;
    }

    //
    // Verify that the value to be written will not be truncated due to the
    // transfer width.
    //
    if (width < 4)
    {
        if (val & ~((1 << (8 * width)) - 1))
        {
            dprintf("lw: Error: Value (0x%x) exceeds max-size (0x%x) for width "
                   "(%d byte(s)).\n", val, ((1 << (8 * width)) - 1), width);
            return 0;
        }
    }

    // Colwert address from offset in DMEM to offset in EMEM
    addr -= startEmem;

    // Acquire EMEM lock
    do
    {
        status = _gspMutexAcquireByIndex(GSP_MUTEX_EMEM, &mutexId);
        if (status == LW_OK)
        {
            break;
        }
        else if (status != LW_OK && status != LW_ERR_STATE_IN_USE)
        {
            dprintf("lw: error in acquiring EMEM lock (non-timeout)\n");
            return 0;
        }
        osPerfDelay(0x10);
        timeoutUs -= 0x10;
    } while (timeoutUs > 0);
    if (status == LW_ERR_STATE_IN_USE)
    {
        dprintf("lw: timeout in acquiring EMEM lock\n");
        return 0;
    }

    // Save the current EMEMC register value (to be restored later)
    ememcOrig = GPU_REG_RD32(LW_PGSP_EMEMC(port));

    // Write to EMEM
    for (i = 0; i < length; i++)
    {
        _gspEmemWriteToEmemOffset(addr + (i * width), val, width, port);
    }
    // Restore the original EMEMC command
    GPU_REG_WR32(LW_PGSP_EMEMC(port), ememcOrig);

    // Release EMEM lock
    _gspMutexReleaseByIndex(GSP_MUTEX_EMEM, mutexId);

    return length * width;
}

/*!
 * @brief Write 'val' to EMEM located at EMEM offset 'offset'.
 *
 * @param[in]  offset  EMEM offset to write 'val'
 * @param[in]  val     Value to write
 * @param[in]  width   width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  port    EMEM port to use for when reading and writing EMEM
 */
static void
_gspEmemWriteToEmemOffset
(
    LwU32  offset,
    LwU32  val,
    LwU32  width,
    LwU32  port
)
{
    LwU32 ememc;
    LwU32 unaligned;
    LwU32 offsetAligned;
    LwU32 data32;
    LwU32 andMask;
    LwU32 lshift;
    LwU32 overflow = 0;
    LwU32 val2     = 0;

    //
    // EMEM transfers are always in 4-byte alignments/chunks. Callwlate the
    // misalignment and the aligned starting offset of the transfer.
    //
    unaligned     = offset & 0x3;
    offsetAligned = offset & ~0x3;
    lshift        = unaligned * 8;
    andMask       = (LwU32)~(((((LwU64)1) << (8 * width)) - 1) << lshift);

    if ((unaligned + width) > 4)
    {
        overflow = unaligned + width - 4;
        val2     = (val >> (8 * (width - overflow)));
    }

    ememc = offsetAligned & (DRF_SHIFTMASK(LW_PGSP_EMEMC_OFFS) |
                             DRF_SHIFTMASK(LW_PGSP_EMEMC_BLK));
    // mark auto-increment on write
    ememc |= DRF_DEF(_PGSP, _EMEMC, _AINCW, _TRUE);
    GPU_REG_WR32(LW_PGSP_EMEMC(port), ememc);

    //
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = GPU_REG_RD32(LW_PGSP_EMEMD(port));
    data32 &= andMask;
    data32 |= (val << lshift);
    GPU_REG_WR32(LW_PGSP_EMEMD(port), data32);

    if (overflow != 0)
    {
        offsetAligned += 4;
        andMask      = ~((1 << (8 * overflow)) - 1);

        data32  = GPU_REG_RD32(LW_PGSP_EMEMD(port));
        data32 &= andMask;
        data32 |= val2;
        GPU_REG_WR32(LW_PGSP_EMEMD(port), data32);
    }
}

/*!
 *  Reset gsp
 */
LW_STATUS gspMasterReset_GV100()
{
    LwU32 reg32;
    LwU32 timeoutUs = GSP_RESET_TIMEOUT_US;

    if (GPU_REG_RD_DRF(_PGSP, _FALCON_RESET_PRIV_LEVEL_MASK, _WRITE_PROTECTION_LEVEL0) ==
        LW_PGSP_FALCON_RESET_PRIV_LEVEL_MASK_WRITE_PROTECTION_LEVEL0_ENABLE)
    {
        // Reset the GSP side
        reg32 = GPU_REG_RD32(LW_PGSP_FALCON_ENGINE);
        reg32 = FLD_SET_DRF(_PGSP, _FALCON_ENGINE, _RESET, _TRUE, reg32);
        GPU_REG_WR32(LW_PGSP_FALCON_ENGINE, reg32);

        // Take it out of reset
        reg32 = GPU_REG_RD32(LW_PGSP_FALCON_ENGINE);
        reg32 = FLD_SET_DRF(_PGSP, _FALCON_ENGINE, _RESET, _FALSE, reg32);
        GPU_REG_WR32(LW_PGSP_FALCON_ENGINE, reg32);
    }
    else
    {
        dprintf("lw: Don't have permission to write to secure reset register.\n");
    }
    
    // TODO: Is it possible to reset GSP with PMC registers?

    // Wait for SCRUBBING to complete
    while (timeoutUs > 0)
    {
        reg32 = GPU_REG_RD32(LW_PGSP_FALCON_DMACTL);

        if (FLD_TEST_DRF(_PGSP, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, reg32) &&
            FLD_TEST_DRF(_PGSP, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, reg32))
        {
            break;
        }
        osPerfDelay(20);
        timeoutUs -= 20;
    }

    if (timeoutUs == 0)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

/*!
 * @return Falcon core interface
 */
const FLCN_CORE_IFACES *
gspGetFalconCoreIFace_GV100()
{
    return &flcnCoreIfaces_v06_00;
}

/*!
 * @return Get the number of DMEM carveouts
 */
LwU32 gspGetDmemNumPrivRanges_GV100(void)
{
    // No carveouts for now
    return 0;
}

/*!
 * @return LW_TRUE  DMEM range is accessible
 *         LW_FALSE DMEM range is inaccessible
 */
LwBool
gspIsDmemRangeAccessible_GV100
(
    LwU32 blkLo,
    LwU32 blkHi
)
{
    dprintf("lw: %s is in LS mode. There are no DMEM carveouts that CPU "
            "can access when %s is not in NS mode. \n",
            pGsp[indexGpu].gspGetEngineName(), pGsp[indexGpu].gspGetEngineName());
    return LW_FALSE;
}

/*!
 * @brief Checks if SEC2 DEBUG fuse is blown or not
 *
 */
LwBool
gspIsDebugMode_GV100()
{
    LwU32 ctlStat =  GPU_REG_RD32(LW_PGSP_SCP_CTL_STAT);

    return !FLD_TEST_DRF(_PGSP, _SCP_CTL_STAT, _DEBUG_MODE, _DISABLED, ctlStat);
}

/*!
 * @return The falcon base address of PMU
 */
LwU32
gspGetFalconBase_GV100()
{
    return DEVICE_BASE(LW_PGSP);
}


/*!
 * Generate a unique identifier that may be used for locking the GSP's HW
 * mutexes.
 *
 * @param[out]  pOwnerId  Pointer to write with the generated owner identifier
 *
 * @return 'LW_OK' if a owner identifier was successfully generated.
 */
static LW_STATUS _gspMutexIdGen(LwU32 *pOwnerId)
{
    LwU32      reg32;
    LW_STATUS  status = LW_OK;

    //
    // Generate a owner ID by reading the MUTEX_ID register (this register
    // has a read side-effect; avoid unnecessary reads).
    //
    reg32 = GPU_REG_RD32(LW_PGSP_MUTEX_ID);

    //
    // Hardware will return _NOT_AVAIL if all identifiers have been used/
    // consumed. Also check against _INIT since zero is not a valid identifier
    // either (zero is used to release mutexes so it cannot be used as an ID).
    //
    if ((!FLD_TEST_DRF(_PGSP, _MUTEX_ID, _VALUE, _INIT     , reg32)) &&
        (!FLD_TEST_DRF(_PGSP, _MUTEX_ID, _VALUE, _NOT_AVAIL, reg32)))
    {
        *pOwnerId = DRF_VAL(_PGSP, _MUTEX_ID, _VALUE, reg32);
    }
    else
    {
        status = LW_ERR_INSUFFICIENT_RESOURCES;
        //
        // A common failure for power-management features is to have BAR0
        // return all f's for all accesses. When this happens, this function
        // tends to fail first, leading to false bugs filed.
        // Check against that here by looking at the entire return value (and
        // not just the eight-bit mutex-ID field).
        //
        if (reg32 == 0xFFFFFFFF)
        {
            dprintf("lw: The %s mutex ID generator returned "
                    "0xFFFFFFFF suggesting there may be an error with "
                    "BAR0. Verify BAR0 is functional before filing a "
                    "bug.\n", pGsp[indexGpu].gspGetEngineName());
        }
        // otherwise, there is a real leak with the mutex identifiers
        else
        {
            dprintf("lw: Failed to generate a mutex identifier. "
                     "Hardware indicates that all identifiers have been "
                     "consumed.\n");
        }
    }
    return status;
}

/*!
 * Release the given owner identifier thus making it available to other
 * clients.
 *
 * @param[in]  ownerId  The owner identifier to release
 */
static void _gspMutexIdRel(LwU32 ownerId)
{
    GPU_FLD_WR_DRF_NUM(_PGSP, _MUTEX_ID_RELEASE, _VALUE, ownerId);
    return;
}

/*!
 * Attempts to acquire the GSP mutex as specified by the given physical
 * mutex-index.
 *
 * @param[in]   physMutexId  The physical mutex-index for the mutex to acquire
 * @param[out]  pOwnerId     Pointer to the ID to write with the ID generated
 *                           upon a successful lock of the mutex.  This value
 *                           will remain unchanged upon failure.
 *
 * @return 'LW_OK'               if the mutex was successfully acquired
 *         'LW_ERR_STATE_IN_USE' if mutex was in use
 *
 */
static LW_STATUS _gspMutexAcquireByIndex(LwU32 physMutexId, LwU32 *pOwnerId)
{
    LwU32      ownerId;
    LwU32      value;
    LW_STATUS  status;

    // generate a unique mutex identifier
    status = _gspMutexIdGen(&ownerId);
    if (status != LW_OK)
    {
        dprintf("lw: error generating a mutex identifer.\n");
        return status;
    }

    //
    // Write the ID into the mutex register to attempt an "acquire"
    // of the mutex.
    //
    GPU_REG_IDX_WR_DRF_NUM(_PGSP, _MUTEX, physMutexId, _VALUE, ownerId);

    //
    // Read the register back to see if the ID stuck.  If the value
    // read back matches the ID written, the mutex was successfully
    // acquired.  Otherwise, release the ID and return an error.
    //
    value = GPU_REG_IDX_RD_DRF(_PGSP, _MUTEX, physMutexId, _VALUE);
    if (value == ownerId)
    {
        *pOwnerId = ownerId;
    }
    else
    {
        dprintf("lw: Cannot acquire mutex index %d (owned by %d).\n",
                physMutexId, value);

        _gspMutexIdRel(ownerId);
        status = LW_ERR_STATE_IN_USE;
    }
    return status;
}

/*!
 * Attempts to release the SEC2 mutex as specified by the given physical
 * mutex-index.  It is the caller's responsibility to ensure that the mutex was
 * acquired before calling this function.  This function simply performs a
 * "release" operation on the given mutex, and frees the owner-ID.
 *
 * @param[in]  physMutexId  The physical mutex-index for the mutex to release
 * @param[in]  ownerId      The ID returned when the mutex was initially
 *                          acquired.
 *
 * @return 'LW_OK' if the mutex and owner ID were successfully released.
 */
static void _gspMutexReleaseByIndex(LwU32 physMutexId, LwU32 ownerId)
{
    GPU_REG_IDX_WR_DRF_DEF(_PGSP, _MUTEX, physMutexId, _VALUE, _INITIAL_LOCK);

    // release the mutex identifer
    _gspMutexIdRel(ownerId);
}

LwBool gspIsSupported_GV100(void)
{
    return TRUE;
}
