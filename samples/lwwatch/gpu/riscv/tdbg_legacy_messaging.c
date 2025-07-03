/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "riscv_taskdbg.h"
#include "tdbg_legacy_messaging.h"

#include "riscv_prv.h"
#include "g_riscv_hal.h"

#if DEBUG_LEGACY_MESSAGING > 1
static void debugQueues(LwU32 addr, LwU32 size, LwU32 head, LwU32 tail)
{
    // TODO: may need to be GPU_REG_RD32
    LwU32 h = bar0ReadLegacy(head) - addr;
    LwU32 t = bar0ReadLegacy(tail) - addr;
    LW_STATUS retVal;
    LwU8 *data = malloc(size);
    LwU32 i;
    if (data == 0)
    {
        dprintf("malloc failed\n");
        return;
    }
    retVal = pRiscv[indexGpu].riscvEmemRead(addr, size, data, backdoor_no);
    dprintf("POINTERS: head=%x (emem_offset=%x) tail=%x (emem_offset=%x)\nDUMP:", h, h - addr, t, t - addr);
    for (i = 0; i < size; i+=4)
    {
        if (i%16 == 0)
        dprintf("\n");
        dprintf("%02x %02x %02x %02x ",
        *(LwU32 *)(data+i) >> 0 & 0xFF,
        *(LwU32 *)(data+i) >> 8 & 0xFF,
        *(LwU32 *)(data+i) >> 16 & 0xFF,
        *(LwU32 *)(data+i) >> 24 & 0xFF);
    }
    dprintf("\n");
    free(data);
}
#endif

LwBool riscvEmemCommandReady()
{
    //simple command queue only allows one command in-flight for now.
    if (GPU_REG_RD32(pRiscv[indexGpu]._queueHead()) != GPU_REG_RD32(pRiscv[indexGpu]._queueTail()))
        return LW_FALSE;
    else
        return LW_TRUE;
}

LwBool riscvEmemMessageExists()
{
    if (GPU_REG_RD32(pRiscv[indexGpu]._msgqHead()) != GPU_REG_RD32(pRiscv[indexGpu]._msgqTail()))
        return LW_TRUE;
    else
        return LW_FALSE;
}

LW_STATUS riscvPollCommandReady(Session *pSession)
{
    LwU32 timeout = RISCV_LEGACY_MESSAGING_TIMEOUT_MS;
    LwU32 step = 1;
    while (!riscvEmemCommandReady())
    {
        riscvDelay(1);
        --timeout;
        if (!timeout)
        {
            dprintf("RISC-V not ready to receive commands.\n");
#if DEBUG_LEGACY_MESSAGING > 1
            debugQueues(pSession->cmdQueueAddr, pSession->cmdQueueSize,
                pRiscv[indexGpu]._queueHead(), pRiscv[indexGpu]._queueTail());
#endif
            return LW_ERR_TIMEOUT;
        }
#if DEBUG_LEGACY_MESSAGING > 0
        if ((RISCV_LEGACY_MESSAGING_TIMEOUT_MS - timeout) % step == 0)
        {
            step = step < 500 ? step * 2 : 500;
            dprintf("riscvPollCmdReady loops=%u\n", RISCV_LEGACY_MESSAGING_TIMEOUT_MS-timeout);
        }
#endif
    }
    return LW_OK;
}

LW_STATUS riscvPollMsgReady(Session *pSession)
{
    LwU32 timeout = RISCV_LEGACY_MESSAGING_TIMEOUT_MS;
    LwU32 step = 1;
    while (!riscvEmemMessageExists())
    {
        riscvDelay(1);
        timeout --;
        if (!timeout)
        {
            dprintf("RISC-V did not return message.\n");
#if DEBUG_LEGACY_MESSAGING > 1
            debugQueues(pSession->msgQueueAddr, pSession->msgQueueSize,
                pRiscv[indexGpu]._msgqHead(), pRiscv[indexGpu]._msgqTail());
#endif
            return LW_ERR_TIMEOUT;
        }
#if DEBUG_LEGACY_MESSAGING > 0
        if ((RISCV_LEGACY_MESSAGING_TIMEOUT_MS-timeout)%step == 0)
        {
            step = step < 500 ? step * 2 : 500;
            dprintf("riscvPollMsgReady loops=%u\n", RISCV_LEGACY_MESSAGING_TIMEOUT_MS-timeout);
        }
#endif
    }
    return LW_OK;
}

static LW_STATUS riscvFindQueues(Session *pSession)
{
    LW_STATUS retVal = LW_ERR_OBJECT_NOT_FOUND;
    LwU64 i;
    LwBool bFoundDebugger = LW_FALSE;
    LwU32 *emem;
    const RiscVInstance *pInstance = pSession->pInstance;

    if (pInstance->riscv_emem_size == 0)
    {
        dprintf("PMU is not supported.\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    emem = malloc(pInstance->riscv_emem_size * sizeof(LwU8));
    if (emem == 0)
    {
        return LW_ERR_NO_MEMORY;
    }
    retVal = pRiscv[indexGpu].riscvEmemRead(0, pInstance->riscv_emem_size, emem, backdoor_no);
    if (retVal != LW_OK)
        goto riscvFindQueuesExit;

    retVal = LW_ERR_OBJECT_NOT_FOUND;

    for (i=0; i<pInstance->riscv_emem_size/sizeof(LwU32); i++)
    {
        if (emem[i] == *(LwU32 *)"tdb1")
        {
            bFoundDebugger = LW_TRUE;
            break;
        }
    }
    if (bFoundDebugger)
    {
        LwU32 commandQueueAddress = emem[i+1] >> 12;
        LwU32 commandQueueSize = emem[i+1] & 0xFFF;
        LwU32 messageQueueAddress = emem[i+2] >> 12;
        LwU32 messageQueueSize = emem[i+2] & 0xFFF;
        if (emem[i+3] != *(LwU32 *)"Meow")
        {
            dprintf("WARNING: Meow not detected, probable version mismatch or corruption.\n");
        }
        if ((commandQueueSize > pInstance->riscv_emem_size) ||
            (commandQueueSize < 64) ||
            (messageQueueSize > pInstance->riscv_emem_size) ||
            (messageQueueSize < 64))
        {
            dprintf("Queue lengths invalid.\n");
            retVal = LW_ERR_ILWALID_STATE;
            goto riscvFindQueuesExit;
        }
        pSession->cmdQueueAddr = commandQueueAddress;
        pSession->cmdQueueSize = commandQueueSize;
        pSession->msgQueueAddr = messageQueueAddress;
        pSession->msgQueueSize = messageQueueSize;
        dprintf("Found command queue of size %u at offset 0x%08x\n", pSession->cmdQueueSize, pSession->cmdQueueAddr);
        dprintf("Found message queue of size %u at offset 0x%08x\n", pSession->msgQueueSize, pSession->msgQueueAddr);
        retVal = LW_OK;
    }
    else
    {
        dprintf("Cannot find task debugger configuration information (needed to communicate with task debugger).\n");
    }
    riscvFindQueuesExit:
    free(emem);
    return retVal;
}

// PRIVATE FUNCTIONS ABOVE
// PUBLIC FUNCTIONS BELOW

void riscvEmemEmergencyReset()
{
    GPU_REG_WR32(pRiscv[indexGpu]._queueHead(), *(LwU32*)"meow"); //send interrupt
    dprintf("Emergency reset issued.\n");
}

LW_STATUS riscvSendCommand(Session *pSession, TaskDebuggerPacket *pTaskDbgCmd)
{
    LW_STATUS retVal;
    LwU32 len = pTaskDbgCmd->header.dataSize + HEADERSIZE;
    LwU32 cmd_queue_base = pSession->cmdQueueAddr;
    LwU32 cmd_queue_size = pSession->cmdQueueSize;
    LwU8 *pBufferU8, *pBuffer;
    LwU32 head, offset;

    if (cmd_queue_size > 0x10000 || cmd_queue_size == 0)
    {
        retVal = riscvFindQueues(pSession);
        if (retVal != LW_OK)
            return retVal;
        cmd_queue_base = pSession->cmdQueueAddr;
        cmd_queue_size = pSession->cmdQueueSize;
    }

    if (GPU_REG_RD32(pRiscv[indexGpu]._queueHead()) == 0)
    {
        dprintf("Cmd Queue ptr is zero. Likely stale debugger cfg info in EMEM.\n");
        pSession->cmdQueueAddr = 0;
        pSession->cmdQueueSize = 0;
        pSession->msgQueueAddr = 0;
        pSession->msgQueueSize = 0;
        return LW_ERR_OBJECT_NOT_FOUND;
    }

    if (len & 0x3)
    {
        len += 0x3 + 1 - (len & 0x3);
        dprintf("cmd size not multiple of 4, padding\n"); //can pad instead
        // return LW_ERR_ILWALID_ARGUMENT;
    }
    if (len > (cmd_queue_size-4))
    {
        dprintf("cmd size larger than cmd queue buffer size\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    pBufferU8 = (LwU8 *) pTaskDbgCmd;
#if DEBUG_LEGACY_MESSAGING > 0
    dprintf("CMD size=%u", len);
#endif
#if DEBUG_LEGACY_MESSAGING > 1
    for (head = 0; head < len; head+=4)
    {
        if (head%16 == 0)
            dprintf("\n    ");
        dprintf("%02x %02x %02x %02x ",
            *(LwU32 *)(pBufferU8+head) >> 0 & 0xFF,
            *(LwU32 *)(pBufferU8+head) >> 8 & 0xFF,
            *(LwU32 *)(pBufferU8+head) >> 16 & 0xFF,
            *(LwU32 *)(pBufferU8+head) >> 24 & 0xFF);
    }
    dprintf("\n");
#endif
    retVal = riscvPollCommandReady(pSession);
    if (retVal != LW_OK)
        return retVal;

    offset = (GPU_REG_RD32(pRiscv[indexGpu]._queueHead()) & (pRiscvInstance->riscv_emem_size-1)) - cmd_queue_base;
    if (offset > cmd_queue_size)
    {
        dprintf("Command queue head register out of range.\n");
#if DEBUG_LEGACY_MESSAGING > 1
        debugQueues(pSession->cmdQueueAddr, pSession->cmdQueueSize,
                    pRiscv[indexGpu]._queueHead(), pRiscv[indexGpu]._queueTail());
#endif
        return LW_ERR_ILWALID_STATE;
    }
    offset = offset > len ? offset - len : offset - len + cmd_queue_size; // preserve tail ptr, only move head ptr
    pTaskDbgCmd->header.magic = 0x4454; //TD
    pBuffer = (LwU8 *) pTaskDbgCmd;

    if ((offset + len) > (cmd_queue_size))
    {
        pRiscv[indexGpu].riscvEmemWrite(cmd_queue_base+offset, cmd_queue_size-offset, pBuffer, backdoor_no);
        pRiscv[indexGpu].riscvEmemWrite(cmd_queue_base+0, len-(cmd_queue_size-offset), pBuffer+(cmd_queue_size-offset), backdoor_no);
        GPU_REG_WR32(pRiscv[indexGpu]._queueTail(), cmd_queue_base+offset+len-cmd_queue_size);
    }
    else
    {
        pRiscv[indexGpu].riscvEmemWrite(cmd_queue_base+offset, len, pBuffer, backdoor_no);
        GPU_REG_WR32(pRiscv[indexGpu]._queueTail(), cmd_queue_base+offset+len);
    }
    GPU_REG_WR32(pRiscv[indexGpu]._queueHead(),offset+cmd_queue_base); //send interrupt
    return LW_OK;
}

LW_STATUS riscvRecvMessage(Session *pSession, void *pBuffer)
{
    LW_STATUS retVal;
    LwU32 len;
    LwU32 head, tail;
    LwU8 *pBufferU8;

    LwU32 msg_queue_base = pSession->msgQueueAddr;
    LwU32 msg_queue_size = pSession->msgQueueSize;

    if (msg_queue_size > 0x10000 || msg_queue_size == 0)
    {
        retVal = riscvFindQueues(pSession);
        if (retVal != LW_OK)
            return LW_ERR_ILWALID_STATE;
        msg_queue_base = pSession->msgQueueAddr;
        msg_queue_size = pSession->msgQueueSize;
    }
    if (GPU_REG_RD32(pRiscv[indexGpu]._msgqHead()) == 0)
    {
        dprintf("Msg Queue ptr is zero. Likely stale debugger cfg info in EMEM.\n");
        pSession->cmdQueueAddr = 0;
        pSession->cmdQueueSize = 0;
        pSession->msgQueueAddr = 0;
        pSession->msgQueueSize = 0;
        return LW_ERR_OBJECT_NOT_FOUND;
    }

    retVal = riscvPollMsgReady(pSession);
    if (retVal != LW_OK)
        return retVal;

    head = (GPU_REG_RD32(pRiscv[indexGpu]._msgqHead()) & (pRiscvInstance->riscv_emem_size-1)) - msg_queue_base;
    tail = (GPU_REG_RD32(pRiscv[indexGpu]._msgqTail()) & (pRiscvInstance->riscv_emem_size-1)) - msg_queue_base;
    if (tail&3)
    {
        dprintf("debugger task protocol violation\n");
    }
    len = (tail > head) ? tail - head : tail - head + msg_queue_size;
#if DEBUG_LEGACY_MESSAGING > 0
    dprintf("MSG(%02x|%02x) size=%u", head, tail, len);
#endif
    pBufferU8 = pBuffer;

    if ((head + len) > (msg_queue_size))
    {
        pRiscv[indexGpu].riscvEmemRead(msg_queue_base+head, msg_queue_size-head, pBufferU8, backdoor_no);
        pRiscv[indexGpu].riscvEmemRead(msg_queue_base+0, len-(msg_queue_size-head), pBufferU8+(msg_queue_size-head), backdoor_no);
    }
    else
    {
        pRiscv[indexGpu].riscvEmemRead(msg_queue_base+head, len, pBuffer, backdoor_no);
    }
#if DEBUG_LEGACY_MESSAGING > 1
    for (head = 0; head < len; head+=4)
    {
        if (head%16 == 0)
            dprintf("\n    ");
        dprintf("%02x %02x %02x %02x ",
            *(LwU32 *)(pBufferU8+head) >> 0 & 0xFF,
            *(LwU32 *)(pBufferU8+head) >> 8 & 0xFF,
            *(LwU32 *)(pBufferU8+head) >> 16 & 0xFF,
            *(LwU32 *)(pBufferU8+head) >> 24 & 0xFF);
    }
    dprintf("\n");
#endif
    if ( ((TaskDebuggerPacket *) pBuffer)->header.magic != 0x6474) //td
    {
        dprintf("warning: message magic incorrect\n");
    }
    GPU_REG_WR32(pRiscv[indexGpu]._msgqHead(), tail + msg_queue_base); //mark message as received

    return LW_OK;
}

//END LEGACY MESSAGING
