/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <string.h>
#include "lwmisc.h"
#include "riscv_prv.h"
#include "riscv_dbgint.h"

#include "g_riscv_hal.h"

LW_STATUS riscvWaitForHalt(unsigned timeoutUs)
{
    while (timeoutUs)
    {
        if (pRiscv[indexGpu].riscvIsInIcd() || riscvIsInactive())
            return LW_OK;
        riscvDelay(1);
        --timeoutUs;
    }
    dprintf("Failed waiting for halt.\n");

    return LW_ERR_TIMEOUT;
}

LwBool riscvIsInactive(void)
{
    return !pRiscv[indexGpu].riscvIsActive();
}

LwBool riscvIsActive(void)
{
    return pRiscv[indexGpu].riscvIsActive();
}

LwBool riscvIsRunning(void)
{
    return pRiscv[indexGpu].riscvIsActive() && !pRiscv[indexGpu].riscvIsInIcd();
}

LW_STATUS riscvClearAllBreakpoints(void)
{
    int i;
    LW_STATUS retVal = LW_ERR_GENERIC;
    for (i = 0; i < TRIGGERS_MAX; i++)
    {
        retVal = pRiscv[indexGpu].riscvClearBreakpoint(i);
        if (retVal != LW_OK)
            return retVal;
    }
    return LW_OK;
}

LW_STATUS riscvDumpDmesg(LwBool bFlush, LwBool bIcd)
{
    RiscvDbgDmesgHdr hdr;
    LwU32 buffer_start;
    char *buffer = NULL; // encoded buffer copy
    char *buffer_dec = NULL; // decoded buffer
    int dec_offset = 0;
    LW_STATUS ret = LW_OK;

    if (bIcd && !pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Target must be in ICD.\n");
        return LW_ERR_ILWALID_STATE;
    }

    memset(&hdr, 0, sizeof(hdr));
    if (bIcd)
        ret = riscvMemRead(pRiscvInstance->riscv_dmem_start + pRiscvInstance->riscv_dmesg_hdr_addr, sizeof(hdr),
                           &hdr, MEM_FORCE_ICD_ACCESS);
    else
        ret = pRiscv[indexGpu].riscvDmemRead((LwU32)pRiscvInstance->riscv_dmesg_hdr_addr,
                                             sizeof(hdr), &hdr, BACKDOOR_NO);
    if (ret != LW_OK)
    {
        dprintf("Failed looking for dmesg buffer...\n");
        return ret;
    }

    // Sanity checks
    if (hdr.magic != RISCV_DMESG_MAGIC)
    {
        dprintf("Missing magic number.\n");
        return LW_ERR_ILWALID_STATE;
    }

    if (hdr.buffer_size == 0 || hdr.buffer_size > (pRiscvInstance->riscv_dmem_size - sizeof(hdr)))
    {
        dprintf("Buffer size is invalid.");
        return LW_ERR_ILWALID_STATE;
    }

    if (hdr.write_offset > hdr.buffer_size || hdr.read_offset > hdr.buffer_size)
    {
        dprintf("Read or write pointer are invalid.\n");
        return LW_ERR_ILWALID_STATE;
    }

    buffer_start = (LwU32)pRiscvInstance->riscv_dmesg_hdr_addr - hdr.buffer_size;

    dprintf("Found debug buffer at 0x%x, size 0x%x, w %d, r %d, magic %x\n",
            buffer_start, hdr.buffer_size, hdr.write_offset, hdr.read_offset,
            hdr.magic);

    if (hdr.read_offset == hdr.write_offset)
    {
        dprintf("Buffer is empty.\n");
        return LW_OK;
    }

    buffer = malloc(hdr.buffer_size);
    buffer_dec = malloc(hdr.buffer_size + 1); // for an additional NULL terminator

    if (!buffer || !buffer_dec)
    {
        dprintf("Run out of memory allocating dmesg buffers..\n");
        goto out;
    }

    memset(buffer, 0, hdr.buffer_size);
    memset(buffer_dec, 0, hdr.buffer_size + 1);

    if (hdr.write_offset > hdr.read_offset) // no wrap
    {
        if (bIcd)
            ret = riscvMemRead(pRiscvInstance->riscv_dmem_start + buffer_start + hdr.read_offset,
                hdr.write_offset - hdr.read_offset, buffer + hdr.read_offset, MEM_FORCE_ICD_ACCESS);
        else
            ret = pRiscv[indexGpu].riscvDmemRead(buffer_start + hdr.read_offset,
                                                 hdr.write_offset - hdr.read_offset,
                                                 buffer + hdr.read_offset, BACKDOOR_NO);
    }
    else // (hdr.write_offset < hdr.read_offset) //wraps
    {
        if (bIcd)
        {
            ret = riscvMemRead(pRiscvInstance->riscv_dmem_start + buffer_start + hdr.read_offset,
                hdr.buffer_size - hdr.read_offset, buffer + hdr.read_offset, MEM_FORCE_ICD_ACCESS);
            if (ret != LW_OK)
            {
                dprintf("Failed reading message buffer.\n");
                goto out;
            }
            ret = riscvMemRead(pRiscvInstance->riscv_dmem_start + buffer_start,
                hdr.write_offset, buffer, MEM_FORCE_ICD_ACCESS);
        }
        else
        {
            ret = pRiscv[indexGpu].riscvDmemRead(buffer_start + hdr.read_offset,
                                                 hdr.buffer_size - hdr.read_offset,
                                                 buffer + hdr.read_offset, BACKDOOR_NO);
            if (ret != LW_OK)
            {
                dprintf("Failed reading message buffer.\n");
                goto out;
            }
            ret = pRiscv[indexGpu].riscvDmemRead(buffer_start, hdr.write_offset, buffer, BACKDOOR_NO);
        }
    }

    if (ret != LW_OK)
    {
        dprintf("Failed reading message buffer.\n");
        goto out;
    }

    while (hdr.read_offset != hdr.write_offset)
    {
        buffer_dec[dec_offset] = buffer[hdr.read_offset];
        if (buffer_dec[dec_offset] == '\0')
            buffer_dec[dec_offset] = '\n';
        dec_offset++;
        hdr.read_offset = (hdr.read_offset + 1) % hdr.buffer_size;
    }

    // Print buffer
    dprintf("DMESG buffer \n--------\n%s\n------\n", buffer_dec);

    // Flush buffer if requested
    if (bFlush)
    {
        LwU32 offs = (LwU32)((LwUPtr)&hdr.read_offset - (LwUPtr)&hdr);

        dprintf("Clearing dmesg buffer...\n");
        if (bIcd)
            ret = riscvMemWrite(pRiscvInstance->riscv_dmem_start + pRiscvInstance->riscv_dmesg_hdr_addr + offs,
                                sizeof(hdr.read_offset), &hdr.read_offset,
                                MEM_FORCE_ICD_ACCESS);
        else
            ret = pRiscv[indexGpu].riscvDmemWrite((LwU32)pRiscvInstance->riscv_dmesg_hdr_addr + offs,
                                                  sizeof(hdr.read_offset), &hdr.read_offset, BACKDOOR_NO);
        if (ret != LW_OK)
        {
            dprintf("Failed to write back buffer pointers.\n");
            goto out;
        }
    }

out:
    if (buffer)
        free(buffer);
    if (buffer_dec)
        free(buffer_dec);
    return ret;
}
