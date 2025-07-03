/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "bios_lwswitch.h"
#include "error_lwswitch.h"
#include "rmsoecmdif.h"
#include "lwswitch/lr10/dev_ext_devices.h"

#include "flcn/flcn_lwswitch.h"

#include "rmflcncmdif_lwswitch.h"

static LwlStatus
_lwswitch_core_bios_read
(
    lwswitch_device *device,
    LwU8 readType,
    LwU32 reqSize,
    LwU8 *pData
)
{
#define MAX_READ_SIZE 0x2000
    RM_FLCN_CMD_SOE     cmd;
    LWSWITCH_TIMEOUT    timeout;
    LwU32               cmdSeqDesc = 0;
    LW_STATUS           status;
    FLCN               *pFlcn = NULL;
    RM_SOE_CORE_CMD_BIOS *pParams = &cmd.cmd.core.bios;
    LwU64 dmaHandle = 0;
    LwU8 *pReadBuffer = NULL;
    LwU32 spiReadCnt = 0;
    LwU32 offset = 0;
    LwU32 bufferSize = (reqSize < SOE_DMA_MAX_SIZE) ? SOE_DMA_MAX_SIZE : MAX_READ_SIZE;

    // Create DMA mapping for SOE CORE transactions
    status = lwswitch_os_alloc_contig_memory(device->os_handle,
                 (void**)&pReadBuffer, bufferSize, (device->dma_addr_width == 32));
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to allocate contig memory\n");
        return status;
    }

    status = lwswitch_os_map_dma_region(device->os_handle,
                                        pReadBuffer,
                                        &dmaHandle,
                                        bufferSize,
                                        LWSWITCH_DMA_DIR_TO_SYSMEM);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to map dma region to sysmem\n");
        lwswitch_os_free_contig_memory(device->os_handle, pReadBuffer, bufferSize);
        return status;
    }

    pFlcn = device->pSoe->pFlcn;

    while (offset < reqSize)
    {
        lwswitch_os_memset(pReadBuffer, 0, bufferSize);
        lwswitch_os_memset(&cmd, 0, sizeof(cmd));

        cmd.hdr.unitId = RM_SOE_UNIT_CORE;
        cmd.hdr.size   = sizeof(cmd);
        cmd.cmd.core.bios.cmdType = readType;
        RM_FLCN_U64_PACK(&pParams->dmaHandle, &dmaHandle);
        pParams->offset = offset;
        pParams->sizeInBytes = LW_MIN((reqSize - offset), MAX_READ_SIZE);
        cmdSeqDesc = 0;

        status = lwswitch_os_sync_dma_region_for_device(device->os_handle, dmaHandle,
                        bufferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);
        if (status != LW_OK)
        {
            lwswitch_os_unmap_dma_region(device->os_handle, pReadBuffer, dmaHandle,
                    bufferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);
            lwswitch_os_free_contig_memory(device->os_handle, pReadBuffer, bufferSize);
            LWSWITCH_PRINT(device, ERROR, "Failed to yield to DMA controller\n");
            return status;
        }

        lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 4, &timeout);

        status = flcnQueueCmdPostBlocking(device, pFlcn,
                                    (PRM_FLCN_CMD)&cmd,
                                    NULL,   // pMsg             - not used for now
                                    NULL,   // pPayload         - not used for now
                                    SOE_RM_CMDQ_LOG_ID,
                                    &cmdSeqDesc,
                                    &timeout);
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: CORE read failed. rc:%d\n",
                            __FUNCTION__, status);
            break;
        }

        status = lwswitch_os_sync_dma_region_for_cpu(device->os_handle, dmaHandle,
                            bufferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR, "DMA controller failed to yield back\n");
            break;
        }

        if (readType == RM_SOE_CORE_CMD_READ_BIOS)
        {
            lwswitch_os_memcpy(((LwU8*)&pData[offset]), pReadBuffer, pParams->sizeInBytes);
        }
        else if (readType == RM_SOE_CORE_CMD_READ_BIOS_SIZE)
        {
            lwswitch_os_memcpy(((LwU8*)pData), pReadBuffer, reqSize);
            break;
        }

        offset += pParams->sizeInBytes;
        spiReadCnt++;
    }

    lwswitch_os_unmap_dma_region(device->os_handle, pReadBuffer, dmaHandle,
        bufferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);

    lwswitch_os_free_contig_memory(device->os_handle, pReadBuffer, bufferSize);
    return status;
}

LwlStatus
lwswitch_bios_read
(
    lwswitch_device *device,
    LwU32 size,
    void *pData
)
{
    return _lwswitch_core_bios_read(device, RM_SOE_CORE_CMD_READ_BIOS, size, (LwU8*)pData);
}

LwlStatus
lwswitch_bios_read_size
(
    lwswitch_device *device,
    LwU32 *pSize
)
{
    if (pSize == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    return _lwswitch_core_bios_read(device, RM_SOE_CORE_CMD_READ_BIOS_SIZE, sizeof(LwU32), (LwU8*)pSize);
}

/*!
 * @brief Retrieves BIOS Image via SOE's CORE task
 * This function needs SOE to be initialized for the Util task to respond.
 * Upon success the BIOS Image will be place in device.biosImage
 * @param[in/out] device - pointer to the lwswitch device.
 */
LwlStatus
lwswitch_bios_get_image
(
    lwswitch_device *device
)
{
    LwU8 *pBiosRawBuffer = NULL;
    LwlStatus status = LWL_SUCCESS;
    LwU32 biosSize = 0;

    if (device->biosImage.pImage != NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "LWRM: %s: bios already available, skip reading"
                    "\n", __FUNCTION__);

        return LWL_SUCCESS;
    }

    if (!device->pSoe)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: SOE not initialized yet. \n",
                __FUNCTION__);
        return LWL_SUCCESS;
    }

    status = lwswitch_bios_read_size(device, &biosSize);
    if (status != LWL_SUCCESS || biosSize == 0)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "LWRM: %s: bios read size failed"
                    "\n", __FUNCTION__);
        return status;
    }

    LWSWITCH_PRINT(device, SETUP,
                    "LWRM: %s: BIOS Size = 0x%x"
                    "\n", __FUNCTION__, biosSize);

    pBiosRawBuffer = (LwU8*) lwswitch_os_malloc(biosSize);
    if (pBiosRawBuffer == NULL)
    {
            LWSWITCH_PRINT(device, ERROR,
                    "%s : failed memory allocation"
                    "\n", __FUNCTION__);

        return -LWL_NO_MEM;
    }

    lwswitch_os_memset(pBiosRawBuffer, 0, biosSize);

    status = lwswitch_bios_read(device, biosSize, pBiosRawBuffer);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, " Failed to retrieve BIOS image, Code 0x%x\n", status);
        lwswitch_os_free(pBiosRawBuffer);
        return status;
    }

    device->biosImage.pImage = pBiosRawBuffer;
    device->biosImage.size   = biosSize;

    return LWL_SUCCESS;
}
