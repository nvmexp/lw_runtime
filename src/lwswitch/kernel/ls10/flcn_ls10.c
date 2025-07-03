
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "flcn/flcn_lwswitch.h"

#include "lwswitch/ls10/dev_falcon_v4.h"
#include "lwswitch/ls10/dev_riscv_pri.h"

//
// Functions shared with LR10
//

void flcnSetupHal_LR10(PFLCN pFlcn);

static LwU32
_flcnRiscvRegRead_LS10
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            offset
)
{
    // Probably should perform some checks on the offset, the device, and the engine descriptor
    return lwswitch_reg_read_32(device, LW_FALCON2_SOE_BASE + offset);
}

static void
_flcnRiscvRegWrite_LS10
(
    lwswitch_device    *device,
    PFLCN               pFlcn,
    LwU32               offset,
    LwU32               data
)
{
    // Probably should perform some checks on the offset, the device, and the engine descriptor
    lwswitch_reg_write_32(device, LW_FALCON2_SOE_BASE + offset, data);
}

/*!
 * @brief Retrieve the size of the falcon data memory.
 *
 * @param[in]  pGpu             OBJGPU  pointer
 * @param[in]  pFlcn            Falcon object pointer
 * @param[in]  bFalconReachable If set, returns size that can be reached by Falcon
 *
 * @return IMEM size in bytes
 */
static LwU32
_flcnDmemSize_LS10
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LwU32 data = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_HWCFG3);
    return (DRF_VAL(_PFALCON, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, data) <<
            FALCON_DMEM_BLKSIZE2);
}

/*
 * @brief Get the destination after masking
 * off all but the OFFSET and BLOCK in IMEM
 *
 * @param[in] dst   Destination in IMEM
 *
 * @returns dst with only OFFSET and BLOCK bits set
 */
static LwU32
_flcnSetImemAddr_LS10
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst
)
{
    return (dst & (DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_OFFS) |
            DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_BLK)));
}

/*!
 *
 * @brief Copy contents of pSrc to IMEM
 *
 * @param[in] pGpu          OBJGPU pointer
 * @param[in] pFlcn         Falcon object pointer
 * @param[in] dst           Destination in IMEM
 * @param[in] pSrc          IMEM contents
 * @param[in] sizeInBytes   Total IMEM size in bytes
 * @param[in] bSelwre       LW_TRUE if IMEM is signed
 * @param[in] tag           IMEM tag
 * @param[in] port          PMB port to copy IMEM
 *
 * @returns void
 */
static void
_flcnImemCopyTo_LS10
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst,
    LwU8            *pSrc,
    LwU32            sizeBytes,
    LwBool           bSelwre,
    LwU32            tag,
    LwU8             port
)
{
    LwU32  numWords;
    LwU32 *pData  = (LwU32 *)pSrc;
    LwU32  reg32  = 0;
    LwU32  index;

    // simply return in the copy-size is zero
    if (sizeBytes == 0)
    {
        LWSWITCH_PRINT(device, ERROR, "Zero-byte copy requested.\n");
        LWSWITCH_ASSERT(0);
        return;
    }

    // the destination must be 4-byte aligned
    if (dst & 0x3)
    {
        LWSWITCH_PRINT(device, ERROR, "Destination not 4-byte aligned. dst=0x%08x\n", dst);
        LWSWITCH_ASSERT(0);
        return;
    }

    // callwlate the number of words and bytes
    numWords = sizeBytes >> 2;

    // mask off all but the OFFSET and BLOCK in DMEM offset
    reg32 = flcnSetImemAddr_HAL(device, pFlcn, dst);
    reg32 = FLD_SET_DRF_NUM(_PFALCON_FALCON, _IMEMC, _AINCW, 0x1, reg32);
    reg32 = FLD_SET_DRF_NUM(_PFALCON_FALCON, _IMEMC, _SELWRE, bSelwre, reg32);
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMC(port), reg32);
    tag = tag >> 8;
    for (index = 0; index < numWords; index++)
    {
        if ((index % 64) == 0)
        {
            flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMT(0), DRF_NUM(_PFALCON_FALCON,
                    _IMEMT, _TAG, tag));
            tag++;
        }
        flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMD(port), DRF_NUM(_PFALCON_FALCON, _IMEMD,
                     _DATA, pData[index]));
    }
    return;
}

/*!
 *
 * @brief Mask the DMEM destination to have only the BLK and OFFSET bits set
 *
 * @param[in] dst           Destination in DMEM
 *
 * @returns masked destination value in DMEM
 */

static LwU32
_flcnSetDmemAddr_LS10
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst
)
{
    return (dst & (DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_OFFS) |
                   DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_BLK)));
}

/*!
 * Depending on the direction of the copy, copies 'sizeBytes' to/from 'pBuf'
 * from/to DMEM offset 'dmemAddr' using DMEM access port 'port'.
 *
 * @param[in]  pGpu       GPU object pointer
 * @param[in]  pFlcn      Falcon object pointer
 * @param[in]  dmemAddr   The DMEM offset for the copy
 * @param[in]  pBuf       The pointer to the buffer containing the data to copy
 * @param[in]  sizeBytes  The number of bytes to copy
 * @param[in]  port       The DMEM port index to use when accessing DMEM
 * @param[in]  bCopyFrom  Boolean representing the copy direction (to/from DMEM)
 *
 * @return LW_OK                    if the data was successfully copied
 *         LW_ERR_ILWALID_ARGUMENT  if the input argument(s) is/are invalid
 */
static LW_STATUS
_flcnDmemTransfer_LS10
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dmemAddr,
    LwU8            *pBuf,
    LwU32            sizeBytes,
    LwU8             port,
    LwBool           bCopyFrom
)
{
    LwU32  numWords;
    LwU32  numBytes;
    LwU32 *pData = (LwU32 *)pBuf;
    LwU32  reg32;
    LwU32  i;

    // simply return if the copy-size is zero
    if (sizeBytes == 0)
    {
        LWSWITCH_PRINT(device, ERROR, "Zero-byte copy requested\n");
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    // the DMEM address must be 4-byte aligned
    if (!LW_IS_ALIGNED(dmemAddr, FLCN_DMEM_ACCESS_ALIGNMENT))
    {
        LWSWITCH_PRINT(device, ERROR, "Source not 4-byte aligned. dmemAddr=0x%08x\n", dmemAddr);
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    // callwlate the number of words and bytes
    numWords = sizeBytes >> 2;
    numBytes = sizeBytes & LWSWITCH_MASK_BITS(2);

    // mask off all but the OFFSET and BLOCK in DMEM offset
    reg32 = flcnSetDmemAddr_HAL(device, pFlcn, dmemAddr);

    if (bCopyFrom)
    {
        // mark auto-increment on read
        reg32 = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCR, 0x1, reg32);
    }
    else
    {
        // mark auto-increment on write
        reg32 = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCW, 0x1, reg32);
    }

    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_DMEMC(port), reg32);

    // directly copy as many words as possible
    for (i = 0; i < numWords; i++)
    {
        if (bCopyFrom)
        {
            pData[i] = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_DMEMD(port));
        }
        else
        {
            flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_DMEMD(port), pData[i]);
        }
    }

    // Check if there are left over bytes to copy
    if (numBytes > 0)
    {
        LwU32 bytesCopied = numWords << 2;

        //
        // Read the contents first. If we're copying to the DMEM, we've set
        // autoincrement on write, so reading does not modify the pointer. We
        // can, thus, do a read/modify/write without needing to worry about the
        // pointer having moved forward. There is no special explanation needed
        // if we're copying from the DMEM since this is the last access to HW
        // in that case.
        //
        reg32 = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_DMEMD(port));

        if (bCopyFrom)
        {
            // Copy byte-by-byte into the buffer as required
            for (i = 0; i < numBytes; i++)
            {
                pBuf[bytesCopied + i] = ((LwU8 *)&reg32)[i];
            }
        }
        else
        {
            // Modify what we read byte-by-byte before writing to dmem
            for (i = 0; i < numBytes; i++)
            {
                ((LwU8 *)&reg32)[i] = pBuf[bytesCopied + i];
            }
            flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_DMEMD(port), reg32);
        }
    }
    return LWL_SUCCESS;
}

static void
_flcnDbgInfoCaptureRiscvPcTrace_LS10
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LwU32 ctl, ridx, widx, count, bufferSize;
    LwBool full;

    flcnRiscvRegWrite_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACECTL,
        DRF_DEF(_PRISCV_RISCV, _TRACECTL, _MODE, _FULL) |
        DRF_DEF(_PRISCV_RISCV, _TRACECTL, _UMODE_ENABLE, _TRUE) |
        DRF_DEF(_PRISCV_RISCV, _TRACECTL, _MMODE_ENABLE, _TRUE) |
        DRF_DEF(_PRISCV_RISCV, _TRACECTL, _INTR_ENABLE, _FALSE) |
        DRF_DEF(_PRISCV_RISCV, _TRACECTL, _HIGH_THSHD, _INIT));

    ctl = flcnRiscvRegRead_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACECTL);

    full = FLD_TEST_DRF_NUM(_PRISCV_RISCV, _TRACECTL,_FULL, 1, ctl);

    if (full)
    {
        LWSWITCH_PRINT(device, INFO, "%s: Trace buffer full. Entries may have been lost.\n", __FUNCTION__);
    }

    // Reset and disable buffer, we don't need it during dump
    flcnRiscvRegWrite_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACECTL, 0);

    widx = flcnRiscvRegRead_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACE_WTIDX);
    widx = DRF_VAL(_PRISCV_RISCV, _TRACE_WTIDX, _WTIDX, widx);

    ridx = flcnRiscvRegRead_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACE_RDIDX);
    bufferSize = DRF_VAL(_PRISCV_RISCV, _TRACE_RDIDX, _MAXIDX, ridx);
    ridx = DRF_VAL(_PRISCV_RISCV, _TRACE_RDIDX, _RDIDX, ridx);

    count = widx > ridx ? widx - ridx : bufferSize + widx - ridx;

    //
    // Trace buffer is full when write idx == read idx and full is set,
    // otherwise it is empty.
    //
    if (widx == ridx && !full)
        count = 0;

    if (count)
    {
        LwU32 entry;
        LWSWITCH_PRINT(device, INFO, "%s: Tracebuffer has %d entries. Starting with latest.\n", __FUNCTION__, count);
        ridx = widx;
        for (entry = 0; entry < count; ++entry)
        {
            LwU64 pc;

            ridx = ridx > 0 ? ridx - 1 : bufferSize - 1;
            flcnRiscvRegWrite_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACE_RDIDX, DRF_NUM(_PRISCV_RISCV, _TRACE_RDIDX, _RDIDX, ridx));
            pc = flcnRiscvRegRead_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACEPC_HI);
            pc = (pc << 32) | flcnRiscvRegRead_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACEPC_LO);
            LWSWITCH_PRINT(device, INFO, "%s: TRACE[%d] = 0x%16llx\n", __FUNCTION__, entry, pc);
        }
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "%s: Trace buffer is empty.\n", __FUNCTION__);
    }

    // reset trace buffer
    flcnRiscvRegWrite_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACE_RDIDX, 0);
    flcnRiscvRegWrite_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACE_WTIDX, 0);

    // Clear full and empty bits
    ctl = FLD_SET_DRF_NUM(_PRISCV_RISCV, _TRACECTL, _FULL, 0, ctl);
    ctl = FLD_SET_DRF_NUM(_PRISCV_RISCV, _TRACECTL, _EMPTY, 0, ctl);
    flcnRiscvRegWrite_HAL(device, pFlcn, LW_PRISCV_RISCV_TRACECTL, ctl);
}

/**
 * @brief   set hal function pointers for functions defined in
 *          LS10 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcn   The flcn for which to set hals
 */
void
flcnSetupHal_LS10
(
    PFLCN            pFlcn
)
{
    flcn_hal *pHal = pFlcn->pHal;

    flcnSetupHal_LR10(pFlcn);
    pHal->riscvRegRead                  =  _flcnRiscvRegRead_LS10;
    pHal->riscvRegWrite                 =  _flcnRiscvRegWrite_LS10;
    pHal->dmemTransfer                  =  _flcnDmemTransfer_LS10;
    pHal->setDmemAddr                   =  _flcnSetDmemAddr_LS10;
    pHal->imemCopyTo                    =  _flcnImemCopyTo_LS10;
    pHal->setImemAddr                   =  _flcnSetImemAddr_LS10;
    pHal->dmemSize                      =  _flcnDmemSize_LS10;
    pHal->dbgInfoCaptureRiscvPcTrace    =  _flcnDbgInfoCaptureRiscvPcTrace_LS10;
}
