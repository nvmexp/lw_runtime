/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwlink_common.h"
#include "common_lwswitch.h"
#include "error_lwswitch.h"
#include "haldef_lwswitch.h"

#include "inforom/inforom_lwswitch.h"

#include "soe/soeififr.h"
#include "rmsoecmdif.h"
#include "flcn/flcn_lwswitch.h"
#include "rmflcncmdif_lwswitch.h"

// Interface functions
static LwlStatus _lwswitch_inforom_unpack_object(const char *, LwU8 *, LwU32 *);
static LwlStatus _lwswitch_inforom_pack_object(const char *, LwU32 *, LwU8 *);
static void _lwswitch_inforom_string_copy(inforom_U008 *pSrc, LwU8 *pDst, LwU32 size);
static LwlStatus _lwswitch_inforom_read_file(lwswitch_device *device,
                                            const char objectName[INFOROM_FS_FILE_NAME_SIZE],
                                            LwU32 packedObjectSize, LwU8 *pPackedObject);
static LwlStatus _lwswitch_inforom_write_file(lwswitch_device *device,
                                            const char objectName[INFOROM_FS_FILE_NAME_SIZE],
                                            LwU32 packedObjectSize,
                                            LwU8 *pPackedObject);

/*!
 * Interface to copy string of inforom object.
 * inforom_U008 is LwU32, and we use 0xff bits to store the character.
 * Therefore we need a special copy API.
 *
 * @param[in]       pSrc          Source pointer
 * @param[out]      pDst          Destination pointer
 * @param[in]       length        Length of the string
 */
static void
_lwswitch_inforom_string_copy
(
    inforom_U008   *pSrc,
    LwU8           *pDst,
    LwU32           length
)
{
    LwU32 i;

    for (i = 0; i < length; ++i)
    {
        pDst[i] = (LwU8)(pSrc[i] & 0xff);
    }
}

static LwlStatus
_lwswitch_inforom_calc_packed_object_size
(
    const char *objectFormat,
    LwU16      *pPackedObjectSize
)
{
    LwU16 count;
    char type;
    LwU16 packedObjectSize = 0;

    while ((type = *objectFormat++) != '\0')
    {
        count = 0;
        while ((type >= '0') && (type <= '9'))
        {
            count *= 10;
            count += (type - '0');
            type = *objectFormat++;
        }
        count = (count > 0) ? count : 1;

        switch (type)
        {
            case INFOROM_FMT_S08:
                packedObjectSize += count;
                break;
            case INFOROM_FMT_U04:
                if (count % 2)
                    return -LWL_ERR_ILWALID_STATE;
                packedObjectSize += (count / 2);
                break;
            case INFOROM_FMT_U08:
                packedObjectSize += count;
                break;
            case INFOROM_FMT_U16:
                packedObjectSize += (count * 2);
                break;
            case INFOROM_FMT_U24:
                packedObjectSize += (count * 3);
                break;
            case INFOROM_FMT_U32:
                packedObjectSize += (count * 4);
                break;
            case INFOROM_FMT_U64:
                packedObjectSize += (count * 8);
                break;
            case INFOROM_FMT_BINARY:
                packedObjectSize += count;
                break;
            default:
                return -LWL_BAD_ARGS;
        }
    }

    *pPackedObjectSize = packedObjectSize;

    return LWL_SUCCESS;
}

static LW_INLINE void
_lwswitch_inforom_unpack_uint_field
(
    LwU8  **ppPackedObject,
    LwU32 **ppObject,
    LwU8    width
)
{
    LwU8 i;
    LwU64 field = 0;

    if (width > 8)
    {
        return;
    }

    for (i = 0; i < width; i++, (*ppPackedObject)++)
    {
        field |= (((LwU64)**ppPackedObject) << (8 * i));
    }

    if (width <= 4)
    {
        **ppObject = (LwU32)field;
        (*ppObject)++;
    }
    else
    {
        **(LwU64 **)ppObject = field;
        *ppObject += 2;
    }
}

static LwlStatus
_lwswitch_inforom_unpack_object
(
    const char *objectFormat,
    LwU8       *pPackedObject,
    LwU32      *pObject
)
{
    LwU16 count;
    char type;
    LwU64 field;

    while ((type = *objectFormat++) != '\0')
    {
        count = 0;
        while ((type >= '0') && (type <= '9'))
        {
            count *= 10;
            count += (type - '0');
            type = *objectFormat++;
        }
        count = (count > 0) ? count : 1;

        for (; count > 0; count--)
        {
            switch (type)
            {
                case INFOROM_FMT_S08:
                    field = *pPackedObject++;
                    field |= ((field & 0x80) ? ~0xff : 0);
                    *pObject++ = (LwU32)field;
                    break;
                case INFOROM_FMT_U04:
                    // Extract two nibbles per byte, and adjust count accordingly
                    if (count % 2)
                        return -LWL_ERR_ILWALID_STATE;
                    field = *pPackedObject++;
                    *pObject++ = (LwU32)(field & 0x0f);
                    *pObject++ = (LwU32)((field & 0xf0) >> 4);
                    count--;
                    break;
                case INFOROM_FMT_U08:
                    _lwswitch_inforom_unpack_uint_field(&pPackedObject, &pObject, 1);
                    break;
                case INFOROM_FMT_U16:
                    _lwswitch_inforom_unpack_uint_field(&pPackedObject, &pObject, 2);
                    break;
                case INFOROM_FMT_U24:
                    _lwswitch_inforom_unpack_uint_field(&pPackedObject, &pObject, 3);
                    break;
                case INFOROM_FMT_U32:
                    _lwswitch_inforom_unpack_uint_field(&pPackedObject, &pObject, 4);
                    break;
                case INFOROM_FMT_U64:
                    _lwswitch_inforom_unpack_uint_field(&pPackedObject, &pObject, 8);
                    break;
                case INFOROM_FMT_BINARY:
                    lwswitch_os_memcpy(pObject, pPackedObject, count);
                    pObject += LW_CEIL(count, 4);
                    pPackedObject += count;
                    // Adjust count to exit the loop.
                    count = 1;
                    break;
                default:
                    return -LWL_BAD_ARGS;
            }
        }
    }

    return LWL_SUCCESS;
}

static LW_INLINE void
_lwswitch_inforom_pack_uint_field
(
    LwU8  **ppPackedObject,
    LwU32 **ppObject,
    LwU8    width
)
{
    LwU8 i;
    LwU64 field = (width <= 4) ? **ppObject : **((LwU64 **)ppObject);

    if (width > 8)
    {
        return;
    }

    for (i = 0; i < width; i++, (*ppPackedObject)++)
    {
        **ppPackedObject = (LwU8)((field >> (8 * i)) & 0xff);
    }

    if (width <= 4)
    {
        (*ppObject)++;
    }
    else
    {
        *ppObject += 2;
    }
}

static LwlStatus
_lwswitch_inforom_pack_object
(
    const char *objectFormat,
    LwU32      *pObject,
    LwU8       *pPackedObject
)
{
    LwU16 count;
    char type;
    LwU64 field;

    while ((type = *objectFormat++) != '\0')
    {
        count = 0;
        while ((type >= '0') && (type <= '9'))
        {
            count *= 10;
            count += (type - '0');
            type = *objectFormat++;
        }
        count = (count > 0) ? count : 1;

        for (; count > 0; count--)
        {
            switch (type)
            {
                case INFOROM_FMT_S08:
                    field = *pObject++;
                    *pPackedObject++ = (LwS8)field;
                    break;
                case INFOROM_FMT_U04:
                    // Encode two nibbles per byte, and adjust count accordingly
                    if (count % 2)
                        return -LWL_ERR_ILWALID_STATE;
                    field = (*pObject++) & 0xf;
                    field |= (((*pObject++) & 0xf) << 4);
                    *pPackedObject++ = (LwU8)field;
                    count--;
                    break;
                case INFOROM_FMT_U08:
                    _lwswitch_inforom_pack_uint_field(&pPackedObject, &pObject, 1);
                    break;
                case INFOROM_FMT_U16:
                    _lwswitch_inforom_pack_uint_field(&pPackedObject, &pObject, 2);
                    break;
                case INFOROM_FMT_U24:
                    _lwswitch_inforom_pack_uint_field(&pPackedObject, &pObject, 3);
                    break;
                case INFOROM_FMT_U32:
                    _lwswitch_inforom_pack_uint_field(&pPackedObject, &pObject, 4);
                    break;
                case INFOROM_FMT_U64:
                    _lwswitch_inforom_pack_uint_field(&pPackedObject, &pObject, 8);
                    break;
                case INFOROM_FMT_BINARY:
                    lwswitch_os_memcpy(pPackedObject, pObject, count);
                    pObject += LW_CEIL(count, 4);
                    pPackedObject += count;
                    // Adjust count to exit the loop.
                    count = 1;
                    break;
                default:
                    return -LWL_BAD_ARGS;
            }
        }
    }

    return LWL_SUCCESS;
}

/*!
 * Read and unpack an object from the InfoROM filesystem.
 *
 * @param[in]  device           switch device pointer
 * @param[in]  pInforom         INFOROM object pointer
 * @param[in]  objectName       Name of the object to read from the InfoROM
 * @param[in]  pObjectFormat    Ascii-string describing the layout of the
 *                              object to read. Used to callwlate the packed
 *                              object size and to unpack the data.
 * @param[out] pPackedObject    Written with the packed object read from the
 *                              InfoROM. It is assumed that this is large
 *                              enough to hold the packed data size computed
 *                              from the pObjectFormat string. This argument
 *                              cannot be NULL.
 * @param[out] pObject          Written with the unpacked object read from the
 *                              InfoROM. It is assumed that this is large
 *                              enough to hold the unpacked data size computed
 *                              from the pObjectFormat string. This argument
 *                              may be NULL.
 *
 * @return LWL_SUCCESS
 *      Object successfully read, and unpacked if necessary
 * @return -LWL_BAD_ARGS
 *      If one of the required pointer arguments is NULL
 * @return -LWL_ERR_NOT_SUPPORTED
 *      The InfoROM filesystem image is not supported
 * @return Other error
 *      If packed size determination fails, object unpacking fails, or there
 *      is a filesystem adapter failure in reading any packed data, it may
 *      result in other error values.
 */

LwlStatus
lwswitch_inforom_read_object
(
    lwswitch_device *device,
    const char   objectName[3],
    const char  *pObjectFormat,
    LwU8        *pPackedObject,
    void        *pObject
)
{
    struct inforom      *pInforom = device->pInforom;
    LwlStatus           status;
    LwU16               packedSize;
    LwU16               fileSize;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = _lwswitch_inforom_calc_packed_object_size(pObjectFormat, &packedSize);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    status = _lwswitch_inforom_read_file(device, objectName, packedSize, pPackedObject);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "InfoROM FS read for %c%c%c failed! rc:%d\n",
                        objectName[0], objectName[1], objectName[2], status);
        return status;
    }

    //
    // Verify a couple things about the object data:
    //  1. The size in the header matches the callwlated packed size.
    //  2. The type is as it was expected
    //
    fileSize = INFOROM_FS_FILE_SIZE(pPackedObject);
    if (packedSize != fileSize)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "LWRM: %s: object %c%c%c was found, but discarded due to "
                    "a size mismatch! (Expected = 0x%X bytes, Actual = 0x%X "
                    "bytes)\n", __FUNCTION__,
                    objectName[0], objectName[1], objectName[2],
                    packedSize, fileSize);
        return -LWL_ERR_ILWALID_STATE;
    }

    if (!INFOROM_FS_FILE_NAMES_MATCH(pPackedObject, objectName))
    {
        LWSWITCH_PRINT(device, ERROR,
                    "LWRM: %s: object %c%c%c was found, but discarded due to "
                    "a type mismatch in the header!\n", __FUNCTION__,
                    objectName[0], objectName[1], objectName[2]);
        return -LWL_ERR_ILWALID_STATE;
    }

    if (pObject != NULL)
    {
        status = _lwswitch_inforom_unpack_object(pObjectFormat, pPackedObject, pObject);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
    }

    return status;
}

static LwlStatus
_lwswitch_inforom_read_file
(
    lwswitch_device *device,
    const char objectName[INFOROM_FS_FILE_NAME_SIZE],
    LwU32 packedObjectSize,
    LwU8 *pPackedObject
)
{
    LwlStatus status = LWL_SUCCESS;
    void *pDmaBuf;
    LwU64 dmaHandle;
    LwU32 fsRet;
    FLCN *pFlcn = device->pSoe->pFlcn;
    RM_FLCN_CMD_SOE soeCmd;
    RM_SOE_IFR_CMD *pIfrCmd = &soeCmd.cmd.ifr;
    RM_SOE_IFR_CMD_PARAMS *pParams = &pIfrCmd->params;
    LwU32 cmdSeqDesc;
    LWSWITCH_TIMEOUT timeout;
    // The first 4 bytes are reserved for status/debug data from SOE
    LwU32 transferSize = packedObjectSize + sizeof(LwU32);

    status = lwswitch_os_alloc_contig_memory(device->os_handle, &pDmaBuf, transferSize,
                                            (device->dma_addr_width == 32));
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to allocate contig memory\n", __FUNCTION__);
        return status;
    }

    status = lwswitch_os_map_dma_region(device->os_handle, pDmaBuf, &dmaHandle,
                                        transferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to map DMA region\n", __FUNCTION__);
        goto ifr_dma_free_and_exit;
    }

    lwswitch_os_memset(&soeCmd, 0, sizeof(soeCmd));
    soeCmd.hdr.unitId = RM_SOE_UNIT_IFR;
    soeCmd.hdr.size = sizeof(soeCmd);
    pIfrCmd->cmdType = RM_SOE_IFR_READ;

    RM_FLCN_U64_PACK(&pParams->dmaHandle, &dmaHandle);
    lwswitch_os_memcpy(pParams->fileName, objectName, INFOROM_FS_FILE_NAME_SIZE);
    pParams->offset = 0;
    pParams->sizeInBytes = packedObjectSize;

    //SOE will copy entire file into SYSMEM
    lwswitch_os_memset(pDmaBuf, 0, transferSize);

    cmdSeqDesc = 0;
    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS * 100, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn, (PRM_FLCN_CMD)&soeCmd, NULL, NULL,
                                          SOE_RM_CMDQ_LOG_ID, &cmdSeqDesc, &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: DMA transfer failed\n", __FUNCTION__);
        goto ifr_dma_unmap_and_exit;
    }

    status = lwswitch_os_sync_dma_region_for_cpu(device->os_handle, dmaHandle,
                                                        transferSize,
                                                        LWSWITCH_DMA_DIR_TO_SYSMEM);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to sync DMA region\n", __FUNCTION__);
        goto ifr_dma_unmap_and_exit;
    }

    lwswitch_os_memcpy(pPackedObject, (LwU8 *)pDmaBuf + sizeof(LwU32), packedObjectSize);

    fsRet = *(LwU32*)pDmaBuf;
    if (fsRet != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: FS error %x. Filename: %3s\n", __FUNCTION__, fsRet,
                        pParams->fileName);
    }

ifr_dma_unmap_and_exit:
    lwswitch_os_unmap_dma_region(device->os_handle, pDmaBuf, dmaHandle,
                                        transferSize, LWSWITCH_DMA_DIR_FROM_SYSMEM);
ifr_dma_free_and_exit:
    lwswitch_os_free_contig_memory(device->os_handle, pDmaBuf, transferSize);

    return status;
}

static LwlStatus
_lwswitch_inforom_write_file
(
    lwswitch_device *device,
    const char objectName[INFOROM_FS_FILE_NAME_SIZE],
    LwU32 packedObjectSize,
    LwU8 *pPackedObject
)
{
    LwlStatus status = LWL_SUCCESS;
    void *pDmaBuf;
    LwU64 dmaHandle;
    LwU32 fsRet;
    FLCN *pFlcn = device->pSoe->pFlcn;
    RM_FLCN_CMD_SOE soeCmd;
    RM_SOE_IFR_CMD *pIfrCmd = &soeCmd.cmd.ifr;
    RM_SOE_IFR_CMD_PARAMS *pParams = &pIfrCmd->params;
    LwU32 cmdSeqDesc;
    LWSWITCH_TIMEOUT timeout;
    // The first 4 bytes are reserved for status/debug data from SOE
    LwU32 transferSize = packedObjectSize + sizeof(LwU32);

    status = lwswitch_os_alloc_contig_memory(device->os_handle, &pDmaBuf, transferSize,
                                            (device->dma_addr_width == 32));
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to allocate contig memory\n", __FUNCTION__);
        return status;
    }

    status = lwswitch_os_map_dma_region(device->os_handle, pDmaBuf, &dmaHandle,
                                        transferSize, LWSWITCH_DMA_DIR_BIDIRECTIONAL);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to map DMA region\n", __FUNCTION__);
        goto ifr_dma_free_and_exit;
    }

    lwswitch_os_memset(&soeCmd, 0, sizeof(soeCmd));
    soeCmd.hdr.unitId = RM_SOE_UNIT_IFR;
    soeCmd.hdr.size = sizeof(soeCmd);
    pIfrCmd->cmdType = RM_SOE_IFR_WRITE;

    RM_FLCN_U64_PACK(&pParams->dmaHandle, &dmaHandle);
    lwswitch_os_memcpy(pParams->fileName, objectName, INFOROM_FS_FILE_NAME_SIZE);
    pParams->offset = 0;
    pParams->sizeInBytes = packedObjectSize;

    //SOE will copy entire file from SYSMEM
    lwswitch_os_memset(pDmaBuf, 0, transferSize);
    lwswitch_os_memcpy((LwU8 *)pDmaBuf + sizeof(LwU32), pPackedObject, packedObjectSize);

    status = lwswitch_os_sync_dma_region_for_device(device->os_handle, dmaHandle,
                                                        transferSize,
                                                        LWSWITCH_DMA_DIR_BIDIRECTIONAL);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to sync DMA region\n", __FUNCTION__);
        goto ifr_dma_unmap_and_exit;
    }

    cmdSeqDesc = 0;
    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS * 100, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn, (PRM_FLCN_CMD)&soeCmd, NULL, NULL,
                                          SOE_RM_CMDQ_LOG_ID, &cmdSeqDesc, &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: DMA transfer failed\n", __FUNCTION__);
        goto ifr_dma_unmap_and_exit;
    }

    status = lwswitch_os_sync_dma_region_for_cpu(device->os_handle, dmaHandle,
                                                        transferSize,
                                                        LWSWITCH_DMA_DIR_BIDIRECTIONAL);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to sync DMA region\n", __FUNCTION__);
        goto ifr_dma_unmap_and_exit;
    }

    fsRet = *(LwU32*)pDmaBuf;
    if (fsRet != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: FS returned %x. Filename: %3s\n", __FUNCTION__, fsRet,
                        pParams->fileName);
    }

ifr_dma_unmap_and_exit:
    lwswitch_os_unmap_dma_region(device->os_handle, pDmaBuf, dmaHandle,
                                        packedObjectSize, LWSWITCH_DMA_DIR_FROM_SYSMEM);
ifr_dma_free_and_exit:
    lwswitch_os_free_contig_memory(device->os_handle, pDmaBuf, transferSize);

    if (status != LW_OK)
    {
        return status;
    }

    return status;
}

/*!
 * Pack and write an object to the InfoROM filesystem.
 *
 * @param[in]     device            switch device pointer
 * @param[in]     pInforom          INFOROM object pointer
 * @param[in]     objectName        Name of the object to write to the InfoROM
 * @param[in]     pObjectFormat     Ascii-string describing the layout of the
 *                                  object to write. Used to callwlate the
 *                                  packed object size and to pack the data.
 * @param[in]     pObject           Contains the unpacked object to write to
 *                                  the InfoROM. It is assumed that this is
 *                                  large enough to hold the unpacked data
 *                                  size computed from the pObjectFormat
 *                                  string. This argument may not be NULL.
 * @param[in|out] pOldPackedObject  As input, contains the old packed data of
 *                                  the object, to be used to determine if any
 *                                  parts of the write can be avoided. This
 *                                  argument may be NULL.
 *
 * @return LWL_SUCCESS
 *      If the object data is successfully written
 * @return -LWL_BAD_ARGS
 *      If any of the required pointers are NULL
 * @return -LWL_ERR_NOT_SUPPORTED
 *      If the InfoROM filesystem image is not supported
 * @return Other error
 *      If dynamic memory allocation fails, packed size determination fails,
 *      object packing fails, or if there is a filesystem adapter failure in
 *      writing the packed data, it may result in other error values.
 */

LwlStatus
lwswitch_inforom_write_object
(
    lwswitch_device *device,
    const char   objectName[3],
    const char  *pObjectFormat,
    void        *pObject,
    LwU8        *pOldPackedObject
)
{
    struct inforom      *pInforom = device->pInforom;
    LwlStatus status;
    LwU8 *pPackedObject;
    LwU16 packedObjectSize;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = _lwswitch_inforom_calc_packed_object_size(pObjectFormat,
                &packedObjectSize);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if (packedObjectSize > INFOROM_MAX_PACKED_SIZE)
    {
        LWSWITCH_ASSERT(packedObjectSize > INFOROM_MAX_PACKED_SIZE);
        return -LWL_ERR_ILWALID_STATE;
    }

    // Allocate a buffer to pack the object into
    pPackedObject = lwswitch_os_malloc(packedObjectSize);
    if (!pPackedObject)
    {
        return -LWL_NO_MEM;
    }

    status = _lwswitch_inforom_pack_object(pObjectFormat, pObject, pPackedObject);
    if (status != LWL_SUCCESS)
    {
        goto done;
    }

    status = _lwswitch_inforom_write_file(device, objectName, packedObjectSize, pPackedObject);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "InfoROM FS write for %c%c%c failed! rc:%d\n",
                        objectName[0], objectName[1], objectName[2], status);
        goto done;
    }

done:
    lwswitch_os_free(pPackedObject);
    return status;
}

/*!
 * @brief Looks for an object of type pType in the object cache.
 */
static LwlStatus
_lwswitch_inforom_get_cached_object
(
    struct inforom               *pInforom,
    const char                   *pType,
    INFOROM_OBJECT_HEADER_V1_00 **ppHeader
)
{
    struct INFOROM_OBJECT_CACHE_ENTRY *pCacheEntry = pInforom->pObjectCache;

    while (pCacheEntry != NULL)
    {
        if (INFOROM_FS_FILE_NAMES_MATCH(pType, pCacheEntry->header.type))
        {
            *ppHeader = &pCacheEntry->header;
            return LWL_SUCCESS;
        }

        pCacheEntry = pCacheEntry->pNext;
    }

    return -LWL_NOT_FOUND;
}

/*!
 * @brief Adds an object's unpacked header and packed data to the object cache.
 *
 * @param[in]  pInforom         INFOROM object pointer
 * @param[in]  pHeader          A pointer to the object's unpacked header
 *
 * @return LWL_SUCCESS
 *      If the object information is successfully added to the object cache
 * @return Other error
 *      If dynamic memory allocation of the cache entry fails
 */
LwlStatus lwswitch_inforom_add_object
(
    struct inforom              *pInforom,
    INFOROM_OBJECT_HEADER_V1_00 *pHeader
)
{
    struct INFOROM_OBJECT_CACHE_ENTRY *pCacheEntry = NULL;

    if (!pInforom || !pHeader)
    {
        return -LWL_ERR_ILWALID_STATE;
    }

    // Allocate a new cache entry
    pCacheEntry = lwswitch_os_malloc(sizeof(struct INFOROM_OBJECT_CACHE_ENTRY));
    if (!pCacheEntry)
    {
        return -LWL_NO_MEM;
    }

    lwswitch_os_memset(pCacheEntry, 0,
                sizeof(struct INFOROM_OBJECT_CACHE_ENTRY));

    lwswitch_os_memcpy(&pCacheEntry->header, pHeader,
                sizeof(INFOROM_OBJECT_HEADER_V1_00));

    pCacheEntry->pNext     = pInforom->pObjectCache;
    pInforom->pObjectCache = pCacheEntry;

    return LWL_SUCCESS;
}

/*!
 * Get the version/subversion of an object from the Inforom.
 *
 * @param[in]  device       switch device pointer
 * @param[in]  pInforom     INFOROM object pointer
 * @param[in]  objectName   The name of the object to get the version info of
 * @param[out] pVersion     The version of the named object
 * @param[out] pSubVersion  The subversion of the named object
 *
 * @return LWL_SUCCESS
 *      Version information successfully read from the inforom.
 *
 * @return -LWL_ERR_NOT_SUPPORTED
 *      The InfoROM filesystem could not be used.
 *
 * @return Other error
 *      From @inforomReadObject if the object was not cached and could not be
 *      read from the filesystem.
 */
LwlStatus
lwswitch_inforom_get_object_version_info
(
    lwswitch_device *device,
    const char  objectName[3],
    LwU8       *pVersion,
    LwU8       *pSubVersion
)
{
    LwlStatus status = LWL_SUCCESS;
    struct inforom      *pInforom = device->pInforom;
    LwU8 packedHeader[INFOROM_OBJECT_HEADER_V1_00_PACKED_SIZE];
    INFOROM_OBJECT_HEADER_V1_00 *pHeader = NULL;
    INFOROM_OBJECT_HEADER_V1_00 header;
    LwU8 *pFile;
    LwU16 fileSize;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pFile = NULL;

    // First, check the cache for the object in question
    status = _lwswitch_inforom_get_cached_object(pInforom, objectName, &pHeader);
    if (status != LWL_SUCCESS)
    {
        //
        // The object wasn't cached, so we need to read it from
        // the filesystem. Since just the header is read, no checksum
        // verification is performed.
        //
        status = _lwswitch_inforom_read_file(device, objectName,
                    INFOROM_OBJECT_HEADER_V1_00_PACKED_SIZE, packedHeader);
        if (status != LWL_SUCCESS)
        {
            goto done;
        }

        // Unpack the header
        status = _lwswitch_inforom_unpack_object(INFOROM_OBJECT_HEADER_V1_00_FMT,
                    packedHeader, (LwU32 *)&header);
        if (status != LWL_SUCCESS)
        {
            goto done;
        }

        pHeader  = &header;

        //
        // Verify that the file is not corrupt, by attempting to read in the
        // entire file. We only need to do this for objects that weren't
        // cached, as we assume that cached objects were validated when they
        // were added to the cache.
        //
        fileSize = (LwU16)pHeader->size;
        if ((fileSize == 0) || (fileSize > INFOROM_MAX_PACKED_SIZE))
        {
            status = -LWL_ERR_ILWALID_STATE;
            goto done;
        }

        pFile = lwswitch_os_malloc(fileSize);
        if (!pFile)
        {
            status = -LWL_NO_MEM;
            goto done;
        }

        status = _lwswitch_inforom_read_file(device, objectName, fileSize, pFile);
done:
        if (pFile != NULL)
        {
            lwswitch_os_free(pFile);
        }
    }

    if (status == LWL_SUCCESS && pHeader != NULL)
    {
        *pVersion    = (LwU8)(pHeader->version & 0xFF);
        *pSubVersion = (LwU8)(pHeader->subversion & 0xFF);
    }

    return status;
}

/*!
 *  Fill in the static identification data structure for the use by the SOE
 *  to be passed on to a BMC over the I2CS interface.
 *
 * @param[in]      device       switch device pointer
 * @param[in]      pInforom     INFOROM object pointer
 * @param[in, out] pData        Target data structure pointer (the structure
 *                              must be zero initialized by the caller)
 */
void
lwswitch_inforom_read_static_data
(
    lwswitch_device            *device,
    struct inforom             *pInforom,
    RM_SOE_SMBPBI_INFOROM_DATA *pData
)
{
#define _INFOROM_TO_SOE_STRING_COPY(obj, irName, soeName)                                   \
{                                                                                           \
    LwU32   _i;                                                                             \
    ct_assert(LW_ARRAY_ELEMENTS(pInforom->obj.object.irName) <=                             \
              LW_ARRAY_ELEMENTS(pData->obj.soeName));                                       \
    for (_i = 0; _i < LW_ARRAY_ELEMENTS(pInforom->obj.object.irName); ++_i)                 \
    {                                                                                       \
        pData->obj.soeName[_i] = (LwU8)(pInforom->obj.object.irName[_i] & 0xff);            \
    }                                                                                       \
    if (LW_ARRAY_ELEMENTS(pInforom->obj.object.irName) <                                    \
        LW_ARRAY_ELEMENTS(pData->obj.soeName))                                              \
    {                                                                                       \
        do                                                                                  \
        {                                                                                   \
            pData->obj.soeName[_i++] = 0;                                                   \
        }                                                                                   \
        while (_i < LW_ARRAY_ELEMENTS(pData->obj.soeName));                                 \
    }                                                                                       \
}

    if (pInforom->OBD.bValid)
    {
        pData->OBD.bValid = LW_TRUE;
        pData->OBD.buildDate = (LwU32)pInforom->OBD.object.buildDate;
        _lwswitch_inforom_string_copy(pInforom->OBD.object.marketingName,
                                      pData->OBD.marketingName,
                                      LW_ARRAY_ELEMENTS(pData->OBD.marketingName));

        _lwswitch_inforom_string_copy(pInforom->OBD.object.serialNumber,
                                      pData->OBD.serialNum,
                                      LW_ARRAY_ELEMENTS(pData->OBD.serialNum));

        //
        // boardPartNum requires special handling, as its size exceeds that
        // of its InfoROM representation
        //
        _INFOROM_TO_SOE_STRING_COPY(OBD, productPartNumber, boardPartNum);
    }

    if (pInforom->OEM.bValid)
    {
        pData->OEM.bValid = LW_TRUE;
        _lwswitch_inforom_string_copy(pInforom->OEM.object.oemInfo,
                                      pData->OEM.oemInfo,
                                      LW_ARRAY_ELEMENTS(pData->OEM.oemInfo));
    }

    if (pInforom->IMG.bValid)
    {
        pData->IMG.bValid = LW_TRUE;
        _lwswitch_inforom_string_copy(pInforom->IMG.object.version,
                                      pData->IMG.inforomVer,
                                      LW_ARRAY_ELEMENTS(pData->IMG.inforomVer));
    }

#undef _INFOROM_TO_SOE_STRING_COPY
}

/*!
 *
 * Wrapper to read an inforom object into system and cache the header
 *
 */
LwlStatus
lwswitch_inforom_load_object
(
    lwswitch_device *device,
    struct inforom  *pInforom,
    const char   objectName[3],
    const char  *pObjectFormat,
    LwU8        *pPackedObject,
    void        *pObject
)
{
    LwlStatus status;

    status = lwswitch_inforom_read_object(device, objectName, pObjectFormat,
                                        pPackedObject, pObject);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to read %c%c%c object, rc:%d\n",
                    objectName[0], objectName[1], objectName[2], status);
        return status;
    }

    status = lwswitch_inforom_add_object(pInforom,
                        (INFOROM_OBJECT_HEADER_V1_00 *)pObject);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "Failed to cache %c%c%c object header, rc:%d\n",
                    objectName[0], objectName[1], objectName[2], status);
        return status;
    }

    return status;
}

/*!
 * @brief Inforom State Initialization
 *
 * Initializes the filesystem layer of the InfoROM so that InfoROM objects can
 * be read. Also load certain InfoROM objects that are neded as early as possible
 * in the initialization path (See bug 992278).
 *
 * @param[in]     device    switch device pointer
 * @param[in/out] pInforom  INFOROM object pointer.
 *
 * @return LWL_SUCCESS
 *      If the filesystem layer is initialized successfully
 * @return -LWL_ERR_NOT_SUPPORTED
 *      If an adapter could not be set up for the InfoROM image device.
 * @return Other error
 *      From attempting to determine the image device location of the InfoROM
 *      or constructing a filesystem adapter for the image.
 */
LwlStatus
lwswitch_initialize_inforom
(
    lwswitch_device *device
)
{
    struct inforom      *pInforom;

    pInforom = lwswitch_os_malloc(sizeof(struct inforom));
    if (!pInforom)
    {
        return -LWL_NO_MEM;
    }
    lwswitch_os_memset(pInforom, 0, sizeof(struct inforom));

    device->pInforom = pInforom;

    return LWL_SUCCESS;
}

/*!
 * @brief Tears down the state of the InfoROM.
 *
 * This includes tearing down the HAL, the FSAL, and freeing any objects left
 * in the object cache.
 *
 * @param[in]  device       switch device pointer
 * @param[in]  pInforom     INFOROM object pointer
 */
void
lwswitch_destroy_inforom
(
    lwswitch_device *device
)
{
    struct inforom                    *pInforom = device->pInforom;
    struct INFOROM_OBJECT_CACHE_ENTRY *pCacheEntry;
    struct INFOROM_OBJECT_CACHE_ENTRY *pTmpCacheEntry;

    if (pInforom)
    {
        pCacheEntry = pInforom->pObjectCache;
        while (pCacheEntry != NULL)
        {
            pTmpCacheEntry = pCacheEntry;
            pCacheEntry = pCacheEntry->pNext;
            lwswitch_os_free(pTmpCacheEntry);
        }

        lwswitch_os_free(pInforom);
        device->pInforom = NULL;
    }
}

void
lwswitch_inforom_post_init
(
    lwswitch_device *device
)
{
    return;
}

LwlStatus
lwswitch_initialize_inforom_objects
(
    lwswitch_device *device
)
{
    LwlStatus status;
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    // RO objects
    status = lwswitch_inforom_read_only_objects_load(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "Failed to load RO objects, rc:%d\n",
                    status);
    }

    // LWL object
    status = lwswitch_inforom_lwlink_load(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "Failed to load LWL object, rc:%d\n",
                    status);
    }

    status = lwswitch_inforom_ecc_load(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "Failed to load ECC object, rc:%d\n",
                    status);
    }

    status = lwswitch_inforom_oms_load(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "Failed to load OMS object, rc:%d\n",
                    status);
    }

    status = lwswitch_inforom_bbx_load(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "Failed to load BBX object, rc: %d\n",
                    status);
    }

    status = lwswitch_inforom_dem_load(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "Failed to load DEM object, rc: %d\n",
                    status);
    }

    return LWL_SUCCESS;
}

void
lwswitch_destroy_inforom_objects
(
    lwswitch_device *device
)
{
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL)
    {
        return;
    }

    // BBX object
    lwswitch_inforom_bbx_unload(device);

    // ECC object
    lwswitch_inforom_ecc_unload(device);

    // LWL object
    lwswitch_inforom_lwlink_unload(device);

    // OMS object
    lwswitch_inforom_oms_unload(device);
}
