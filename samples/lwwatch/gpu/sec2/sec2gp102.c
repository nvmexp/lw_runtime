/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "sec2.h"
#include "pmgr.h"

#include "g_sec2_private.h"

#include "pascal/gp102/dev_pri_ringstation_sys.h"
#include "pascal/gp102/dev_sec_pri.h"
#include "pascal/gp102/dev_master.h"
#include "pascal/gp102/dev_sec_addendum.h"
//
// Adding this file because  LW_PFALCON_FALCON_RESET_PRIV_LEVEL_MASK
// not defined on pascal FALCON manual but corresponding PSEC defination is present.
//
#include "volta/gv11b/dev_falcon_v4.h"
#include "sec2/sec2ifcmn.h"

static void
_sec2EmemWriteToEmemOffset_GP102(LwU32 off, LwU32 val, LwU32 width, LwU32 port);

static FLCN_ENGINE_IFACES flcnEngineIfaces_sec2 =
{
    sec2GetFalconCoreIFace_STUB,         // flcnEngGetCoreIFace
    sec2GetFalconBase_STUB,              // flcnEngGetFalconBase
    sec2GetEngineName,                   // flcnEngGetEngineName
    sec2UcodeName_STUB,                  // flcnEngUcodeName
    sec2GetSymFilePath,                  // flcnEngGetSymFilePath
    sec2QueueGetNum_STUB,                // flcnEngQueueGetNum
    sec2QueueRead_STUB,                  // flcnEngQueueRead
    sec2GetDmemAccessPort,               // flcnEngGetDmemAccessPort
    sec2IsDmemRangeAccessible_STUB,      // flcnEngIsDmemRangeAccessible
    sec2EmemGetSize_STUB,                // flcnEngEmemGetSize
    sec2EmemGetOffsetInDmemVaSpace_STUB, // flcnEngEmemGetOffsetInDmemVaSpace
    sec2EmemGetNumPorts_STUB,            // flcnEngEmemGetNumPorts
    sec2EmemRead_STUB,                   // flcnEngEmemRead
    sec2EmemWrite_STUB,                  // flcnEngEmemWrite
}; // flcnEngineIfaces_sec2

const FLCN_ENGINE_IFACES *
sec2GetFalconEngineIFace_GP102()
{
    const FLCN_CORE_IFACES   *pFCIF = pSec2[indexGpu].sec2GetFalconCoreIFace();
    FLCN_ENGINE_IFACES       *pFEIF = NULL;

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if (pFCIF)
    {
        pFEIF = &flcnEngineIfaces_sec2;

        pFEIF->flcnEngGetCoreIFace               = pSec2[indexGpu].sec2GetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase              = pSec2[indexGpu].sec2GetFalconBase;
        pFEIF->flcnEngQueueGetNum                = pSec2[indexGpu].sec2QueueGetNum;
        pFEIF->flcnEngQueueRead                  = pSec2[indexGpu].sec2QueueRead;
        pFEIF->flcnEngIsDmemRangeAccessible      = pSec2[indexGpu].sec2IsDmemRangeAccessible;
        pFEIF->flcnEngEmemGetSize                = pSec2[indexGpu].sec2EmemGetSize;
        pFEIF->flcnEngEmemGetOffsetInDmemVaSpace = pSec2[indexGpu].sec2EmemGetOffsetInDmemVaSpace;
        pFEIF->flcnEngEmemGetNumPorts            = pSec2[indexGpu].sec2EmemGetNumPorts;
        pFEIF->flcnEngEmemRead                   = pSec2[indexGpu].sec2EmemRead;
        pFEIF->flcnEngEmemWrite                  = pSec2[indexGpu].sec2EmemWrite;
    }
    return pFEIF;
}

LwU32
sec2QueueGetNum_GP102()
{
    return ( pSec2[indexGpu].sec2GetQueueHeadSize() +  pSec2[indexGpu].sec2GetMsgqHeadSize());
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
sec2QueueRead_GP102
(
    LwU32         queueId,
    PFLCN_QUEUE   pQueue
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pSec2[indexGpu].sec2GetFalconEngineIFace();
    const FLCN_CORE_IFACES   *pFCIF =
        pSec2[indexGpu].sec2GetFalconCoreIFace();
    LwU32 engineBase = pFEIF->flcnEngGetFalconBase();
    LwU32 numQueues;
    LwU32 numCmdQs =  pSec2[indexGpu].sec2GetQueueHeadSize();
    LwU32 sizeInWords;
    LwU32 ememPort;

    numQueues = pSec2[indexGpu].sec2QueueGetNum();
    if (queueId >= numQueues || !pQueue)
    {
        return LW_FALSE;
    }

    //
    // The "message" queues comes right after the command queues,
    // so we use a special case to get the information
    //
    if (queueId < numCmdQs)
    {
        pQueue->head = GPU_REG_RD32( pSec2[indexGpu].sec2GetQueueHead(queueId));
        pQueue->tail = GPU_REG_RD32( pSec2[indexGpu].sec2GetQueueTail(queueId));
    }
    else
    {
        pQueue->head = GPU_REG_RD32( pSec2[indexGpu].sec2GetMsgqHead(queueId-numCmdQs));
        pQueue->tail = GPU_REG_RD32( pSec2[indexGpu].sec2GetMsgqTail(queueId-numCmdQs));
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
        dprintf("lw: %s: SEC2 queue 0x%x is larger than configured to read:\n",
                __FUNCTION__, queueId);
        dprintf("lw:     Queue Size: 0x%x     Supported Size: 0x%x\n",
                (LwU32)(sizeInWords * sizeof(LwU32)), (LwU32)(LW_FLCN_MAX_QUEUE_SIZE * sizeof(LwU32)));
        dprintf("lw:     Make LW_FLCN_MAX_QUEUE_SIZE larger and re-compile LW_WATCH\n");
        return LW_FALSE;
    }

    ememPort = pSec2[indexGpu].sec2GetEmemPortId();

    // Simply read the queue into the buffer if it is initialized
    if (pQueue->tail > pFEIF->flcnEngEmemGetOffsetInDmemVaSpace())
    {
        pFEIF->flcnEngEmemRead(pQueue->tail, sizeInWords, ememPort,
                               pQueue->data);
    }
    return LW_TRUE;
}

LwU32
sec2EmemGetNumPorts_GP102()
{
    return LW_PSEC_EMEMD__SIZE_1;
}

LwU32
sec2EmemGetSize_GP102()
{
    return GPU_REG_RD_DRF(_PSEC, _HWCFG, _EMEM_SIZE) * FLCN_BLK_ALIGNMENT;
}

/*!
 *  @return Offset of EMEM in DMEM VA space.
 *          Should be located directly above addressable (virtual) DMEM.
 */
LwU32
sec2EmemGetOffsetInDmemVaSpace_GP102()
{
    LwU32 hwcfg1 = pObjSec2->readRegAddr(LW_PFALCON_FALCON_HWCFG1);
    // START_EMEM = DMEM_VA_MAX = 2^(DMEM_TAG_WIDTH + 8)
    return 1 << (DRF_VAL(_PSEC, _FALCON_HWCFG1, _DMEM_TAG_WIDTH, hwcfg1) +
                 FALCON_DMEM_BLKSIZE);
}


LW_STATUS
sec2LockEmem_GP102(LwU32 *pMutexId)
{
    LW_STATUS status    = LW_OK;
    LwU32     timeoutUs = SEC2_MUTEX_TIMEOUT_US;

    // Acquire EMEM lock
    do
    {
        status = pPmgr[indexGpu].pmgrMutexAcquireByIndex(PMGR_MUTEX_ID_SEC2_EMEM_ACCESS, pMutexId);
        if (status == LW_OK)
        {
            break;
        }
        else
        {
            dprintf("lw: %s: error in acquiring EMEM lock (non-timeout)\n", __FUNCTION__);
            return status;
        }
        osPerfDelay(0x10);
        timeoutUs -= 0x10;
    } while (timeoutUs > 0);

    if (status == LW_ERR_STATE_IN_USE)
    {
        dprintf("lw: %s: timeout in acquiring EMEM lock\n", __FUNCTION__);
    }

    return status;
}


LW_STATUS
sec2UnlockEmem_GP102(LwU32 mutexId)
{
    pPmgr[indexGpu].pmgrMutexReleaseByIndex(PMGR_MUTEX_ID_SEC2_EMEM_ACCESS, mutexId);
    return LW_OK;
}


/*!
 *  @brief Reads data from EMEM
 *
 *  Read length words of EMEM starting at DMEM address 'addr'.
 *  Offset will automatically be truncated down to 4-byte aligned value.
 *  If length spans out of the range of the EMEM, it will automatically
 *  be truncated to fit into the EMEM range.
 *
 *  The address must be located in the EMEM region located directly above the
 *  maximum virtual address of DMEM.
 *
 *  @param addr        The DMEM address for the source of the copy.
 *  @param length      Number of 4-byte words to read.
 *  @param port        Port to read from.
 *  @param pBuf        Buffer to store EMEM into.
 *
 *  @return 0 on error, or number of 4-byte words read.
 */
LwU32
sec2EmemRead_GP102
(
    LwU32 addr,
    LwU32 length,
    LwU32 port,
    LwU32 *pBuf
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pSec2[indexGpu].sec2GetFalconEngineIFace();
    LwU32     ememSize    = pFEIF->flcnEngEmemGetSize();
    LwU32     startEmem   = pFEIF->flcnEngEmemGetOffsetInDmemVaSpace();
    LwU32     endEmem     = startEmem + ememSize;
    LwU32     ememcOrig   = 0x0;
    LwU32     ememc       = 0x0;
    LwU32     i           = 0x0;
    LwU32     mutexId;

    addr &= ~(sizeof(LwU32) - 1);

    // Fail if the port specified is not valid
    if (port >= pSec2[indexGpu].sec2GetEmemcSize())
    {
        dprintf("lw: Only %d ports supported. Accessed port=%d\n",
                pSec2[indexGpu].sec2GetEmemcSize(), port);
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
    if (LW_OK != pSec2[indexGpu].sec2LockEmem(&mutexId))
    {
        dprintf("lw: %s: Could not acquire EMEM lock for reading, aborting read operation\n", __FUNCTION__);
        return 0;
    }

    //
    // Build the EMEMC command that auto-increments on each read
    // We take the address and mask it off with OFFSET and BLOCK region.
    //
    // Note: We also remember and restore the original command value
    //
    ememc = addr & (DRF_SHIFTMASK(LW_PSEC_EMEMC_OFFS) |
                    DRF_SHIFTMASK(LW_PSEC_EMEMC_BLK));
    // mark auto-increment on read
    ememc |= DRF_DEF(_PSEC, _EMEMC, _AINCR, _TRUE);

    // Store the original EMEMC command
    ememcOrig = GPU_REG_RD32( pSec2[indexGpu].sec2GetEmemc(port));

    // Perform the actual EMEM read operations
    GPU_REG_WR32( pSec2[indexGpu].sec2GetEmemc(port), ememc);
    for (i = 0; i < length; i++)
    {
        pBuf[i] = GPU_REG_RD32( pSec2[indexGpu].sec2GetEmemd(port));
    }

    // Restore the original EMEMC command
    GPU_REG_WR32( pSec2[indexGpu].sec2GetEmemc(port), ememcOrig);

    // Release EMEM lock
    if (LW_OK != pSec2[indexGpu].sec2UnlockEmem(mutexId))
    {
        dprintf("lw: %s: Could not unlock EMEM after reading, this is bad for RM and lwwatch\n", __FUNCTION__);
    }

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
LwU32
sec2EmemWrite_GP102
(
    LwU32 addr,
    LwU32 val,
    LwU32 width,
    LwU32 length,
    LwU32 port
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pSec2[indexGpu].sec2GetFalconEngineIFace();
    LwU32     ememSize  = pFEIF->flcnEngEmemGetSize();
    LwU32     startEmem = pFEIF->flcnEngEmemGetOffsetInDmemVaSpace();
    LwU32     endEmem   = startEmem + ememSize;
    LwU32     ememcOrig = 0x0;
    LwU32     mutexId;
    LwU32     i;

    // Fail if the port specified is not valid
    if (port >= pSec2[indexGpu].sec2GetEmemcSize())
    {
        dprintf("lw: Only %d ports supported. Accessed port=%d\n",
                pSec2[indexGpu].sec2GetEmemcSize(), port);
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
    if (LW_OK != pSec2[indexGpu].sec2LockEmem(&mutexId))
    {
        dprintf("lw: %s: Could not acquire EMEM lock for writing, aborting write operation\n", __FUNCTION__);
        return 0;
    }

    // Save the current EMEMC register value (to be restored later)
    ememcOrig = GPU_REG_RD32( pSec2[indexGpu].sec2GetEmemc(port));

    // Write to EMEM
    for (i = 0; i < length; i++)
    {
        _sec2EmemWriteToEmemOffset_GP102(addr + (i * width), val, width, port);
    }
    // Restore the original EMEMC command
    GPU_REG_WR32( pSec2[indexGpu].sec2GetEmemc(port), ememcOrig);

    // Release EMEM lock
    if (LW_OK != pSec2[indexGpu].sec2UnlockEmem(mutexId))
    {
        dprintf("lw: %s: Could not unlock EMEM after writing, this is bad for RM and lwwatch\n", __FUNCTION__);
    }

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
_sec2EmemWriteToEmemOffset_GP102
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

    ememc = offsetAligned & (DRF_SHIFTMASK(LW_PSEC_EMEMC_OFFS) |
                             DRF_SHIFTMASK(LW_PSEC_EMEMC_BLK));
    // mark auto-increment on write
    ememc |= DRF_DEF(_PSEC, _EMEMC, _AINCW, _TRUE);
    GPU_REG_WR32( pSec2[indexGpu].sec2GetEmemc(port), ememc);

    //
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = GPU_REG_RD32( pSec2[indexGpu].sec2GetEmemd(port));
    data32 &= andMask;
    data32 |= (val << lshift);
    GPU_REG_WR32( pSec2[indexGpu].sec2GetEmemd(port), data32);

    if (overflow != 0)
    {
        offsetAligned += 4;
        andMask      = ~((1 << (8 * overflow)) - 1);

        data32  = GPU_REG_RD32( pSec2[indexGpu].sec2GetEmemd(port));
        data32 &= andMask;
        data32 |= val2;
        GPU_REG_WR32( pSec2[indexGpu].sec2GetEmemd(port), data32);
    }
}


/*!
 * Attempts to acquire the SEC2 mutex as specified by the given physical
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
LW_STATUS
sec2AcquireMutexByIndex_GP102
(
    LwU32     physMutexId,
    LwU32    *pOwnerId
)
{
    LwU32 trap18 = GPU_REG_RD32(LW_PPRIV_SYS_PRI_DECODE_TRAP18_MATCH);
    if (DRF_VAL (_PPRIV, _SYS_PRI_DECODE_TRAP18_MATCH, _ADDR, trap18) != pSec2[indexGpu].sec2GetMutexId())
        dprintf("lw: %s: ERROR: SEC2 mutex should be protected by decode traps\n", __FUNCTION__);
    else
        dprintf("lw: %s: SEC2 mutex cannot be used on GP10X because it is protected at level3 by decode traps.\n", __FUNCTION__);
    return LW_ERR_NOT_SUPPORTED;
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
void
sec2ReleaseMutexByIndex_GP102
(
    LwU32     physMutexId,
    LwU32     ownerId
)
{
    dprintf("lw: %s: SEC2 mutex cannot be used on GP10X because it is protected at level3 by decode traps.\n", __FUNCTION__);
    return;
}


/*!
 * @return Falcon core interface
 */
const FLCN_CORE_IFACES *
sec2GetFalconCoreIFace_GP104()
{
    return &flcnCoreIfaces_v06_00;
}

/*!
 * Returns LW_PSEC_QUEUE_HEAD__SIZE_1 value
 */
LwU32
sec2GetQueueHeadSize_GP102()
{
    return LW_PSEC_QUEUE_HEAD__SIZE_1;
}

/*!
 * Returns LW_PSEC_MSGQ_HEAD__SIZE_1 value
 */
LwU32
sec2GetMsgqHeadSize_GP102()
{
    return LW_PSEC_MSGQ_HEAD__SIZE_1;
}

/*!
 * Returns LW_PSEC_QUEUE_HEAD value
 */
LwU32
sec2GetQueueHead_GP102(LwU32 queueId)
{
    return LW_PSEC_QUEUE_HEAD(queueId);
}

/*!
 * Returns LW_PSEC_QUEUE_TAIL value
 */
LwU32
sec2GetQueueTail_GP102(LwU32 queueId)
{
    return LW_PSEC_QUEUE_TAIL(queueId);
}

/*!
 * Returns LW_PSEC_MSGQ_HEAD value
 */
LwU32
sec2GetMsgqHead_GP102(LwU32 queueId)
{
    return LW_PSEC_MSGQ_HEAD(queueId);
}

/*!
 * Returns LW_PSEC_MSGQ_TAIL value
 */
LwU32
sec2GetMsgqTail_GP102(LwU32 queueId)
{
    return LW_PSEC_MSGQ_TAIL(queueId);
}

/*!
 * Returns LW_PSEC_EMEMC__SIZE_1 value
 */
LwU32
sec2GetEmemcSize_GP102()
{
    return LW_PSEC_EMEMC__SIZE_1;
}

/*!
 * Returns LW_PSEC_EMEMC(i) value
 */
LwU32
sec2GetEmemc_GP102(LwU32 port)
{
    return LW_PSEC_EMEMC(port);
}

/*!
 * Returns LW_PSEC_EMEMD(i) value
 */
LwU32
sec2GetEmemd_GP102(LwU32 port)
{
    return LW_PSEC_EMEMD(port);
}

/*!
 * Returns the physical address of LW_PSEC_MUTEX_ID register.
 */
LwU32
sec2GetMutexId_GP102()
{
    return LW_PSEC_MUTEX_ID;
}

LwU32
sec2GetEmemPortId_GP102()
{
    return LW_SEC2_EMEM_ACCESS_PORT_RM;
}
