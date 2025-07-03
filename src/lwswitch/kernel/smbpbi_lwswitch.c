/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021  by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwfixedtypes.h"
#include "common_lwswitch.h"
#include "error_lwswitch.h"
#include "rmsoecmdif.h"
#include "smbpbi_lwswitch.h"
#include "lwswitch/lr10/dev_ext_devices.h"

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "cci/cci_lwswitch.h"
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "flcn/flcn_lwswitch.h"

#include "rmflcncmdif_lwswitch.h"

#define GET_PFIFO_FROM_DEVICE(dev)  (&(dev)->pSmbpbi->sharedSurface->inforomObjects.DEM.object.v1)

#define DEM_FIFO_SIZE   INFOROM_DEM_OBJECT_V1_00_FIFO_SIZE
#define DEM_FIFO_PTR(x) ((x) % DEM_FIFO_SIZE)
#define DEM_PTR_DIFF(lwr, next) (((next) > (lwr)) ? ((next) - (lwr)) :      \
                    (DEM_FIFO_SIZE - ((lwr) - (next))))
#define DEM_BYTES_OCLWPIED(pf) DEM_PTR_DIFF((pf)->readOffset, (pf)->writeOffset)
//
// See how much space is available in the FIFO.
// Must leave 1 word free so the write pointer does not
// catch up with the read pointer. That would be indistinguishable
// from an empty FIFO.
//
#define DEM_BYTES_AVAILABLE(pf) (DEM_PTR_DIFF((pf)->writeOffset, (pf)->readOffset) - \
                                 sizeof(LwU32))
#define DEM_RECORD_SIZE_MAX (sizeof(LW_MSGBOX_DEM_RECORD)   \
                            + LW_MSGBOX_MAX_DRIVER_EVENT_MSG_TXT_SIZE)
#define DEM_RECORD_SIZE_MIN (sizeof(LW_MSGBOX_DEM_RECORD) + 1)

#define FIFO_REC_LOOP_ITERATOR  _lwrPtr
#define FIFO_REC_LOOP_REC_PTR   _recPtr
#define FIFO_REC_LOOP_REC_SIZE  _recSize
#define FIFO_REC_LOOP_START(pf, cond)                                                           \
{                                                                                               \
    LwU16                           _nextPtr;                                                   \
    for (FIFO_REC_LOOP_ITERATOR = (pf)->readOffset; cond; FIFO_REC_LOOP_ITERATOR = _nextPtr)    \
    {                                                                                           \
        LW_MSGBOX_DEM_RECORD    *FIFO_REC_LOOP_REC_PTR = (LW_MSGBOX_DEM_RECORD *)               \
                                            ((pf)->fifoBuffer + FIFO_REC_LOOP_ITERATOR);        \
        LwU16                   FIFO_REC_LOOP_REC_SIZE =                                        \
                                    FIFO_REC_LOOP_REC_PTR->recordSize * sizeof(LwU32);

#define FIFO_REC_LOOP_END                                                                       \
        _nextPtr = DEM_FIFO_PTR(FIFO_REC_LOOP_ITERATOR + FIFO_REC_LOOP_REC_SIZE);               \
    }                                                                                           \
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define SMBPBI_CCI_POLLING_RATE_HZ  5

#define SMBPBI_CCI_READ_TEMP_MAX_TRIES  3
static LwlStatus _lwswitch_smbpbi_cci_init(lwswitch_device *device);
static LwlStatus _lwswitch_smbpbi_cci_update_hottest(lwswitch_device *device, PCCI pCci, PSOE_SMBPBI_OSFP_DATA pSsod);
static LwlStatus _lwswitch_smbpbi_cci_update_led_state(lwswitch_device *device, PCCI pCci, PSOE_SMBPBI_OSFP_DATA pSsod);
static void _lwswitch_smbpbi_cci_poll_callback(lwswitch_device *device);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

static void _smbpbiDemInit(lwswitch_device *device, struct smbpbi *pSmbpbi, struct INFOROM_DEM_OBJECT_V1_00 *pFifo);
static void _lwswitch_smbpbi_dem_flush(lwswitch_device *device);


LwlStatus
lwswitch_smbpbi_init
(
    lwswitch_device *device
)
{
    LW_STATUS                  status;
    LwU64                      dmaHandle;
    void                      *cpuAddr;

    if (!device->pSoe)
    {
        return -LWL_ERR_ILWALID_STATE;
    }

    // Create DMA mapping for SMBPBI transactions
    status = lwswitch_os_alloc_contig_memory(device->os_handle, &cpuAddr,
                                        sizeof(SOE_SMBPBI_SHARED_SURFACE),
                                        (device->dma_addr_width == 32));
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to allocate contig memory, rc:%d\n",
                    status);
        return status;
    }

    lwswitch_os_memset(cpuAddr, 0, sizeof(SOE_SMBPBI_SHARED_SURFACE));

    status = lwswitch_os_map_dma_region(device->os_handle, cpuAddr, &dmaHandle,
                        sizeof(SOE_SMBPBI_SHARED_SURFACE),
                        LWSWITCH_DMA_DIR_BIDIRECTIONAL);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Failed to map dma region for SMBPBI shared surface, rc:%d\n",
            status);
        goto os_map_dma_region_fail;
    }

    device->pSmbpbi = lwswitch_os_malloc(sizeof(struct smbpbi));
    if (!device->pSmbpbi)
    {
        status = -LWL_NO_MEM;
        goto smbpbi_init_fail;
    }

    device->pSmbpbi->sharedSurface = cpuAddr;
    device->pSmbpbi->dmaHandle = dmaHandle;

    return LWL_SUCCESS;

smbpbi_init_fail:
    lwswitch_os_unmap_dma_region(device->os_handle, cpuAddr, dmaHandle,
        sizeof(SOE_SMBPBI_SHARED_SURFACE), LWSWITCH_DMA_DIR_BIDIRECTIONAL);
os_map_dma_region_fail:
    lwswitch_os_free_contig_memory(device->os_handle, cpuAddr, sizeof(SOE_SMBPBI_SHARED_SURFACE));

    return status;
}

LwlStatus
lwswitch_smbpbi_post_init
(
    lwswitch_device * device
)
{
    struct smbpbi             *pSmbpbi = device->pSmbpbi;
    FLCN                      *pFlcn;
    LwU64                      dmaHandle;
    RM_FLCN_CMD_SOE            cmd;
    LWSWITCH_TIMEOUT           timeout;
    LwU32                      cmdSeqDesc;
    RM_SOE_SMBPBI_CMD_INIT    *pInitCmd = &cmd.cmd.smbpbiCmd.init;
    LwlStatus                  status;

    if (!device->pSmbpbi || !device->pInforom)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    status = _lwswitch_smbpbi_cci_init(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to init SMBPBI/CCI, rc:%d\n",
                    status);
        return status;
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    // Populate shared surface with static InfoROM data
    lwswitch_inforom_read_static_data(device, device->pInforom,
                                      &device->pSmbpbi->sharedSurface->inforomObjects);

    pFlcn = device->pSoe->pFlcn;
    dmaHandle = pSmbpbi->dmaHandle;

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));
    cmd.hdr.unitId = RM_SOE_UNIT_SMBPBI;
    cmd.hdr.size   = RM_SOE_CMD_SIZE(SMBPBI, INIT);
    cmd.cmd.smbpbiCmd.cmdType = RM_SOE_SMBPBI_CMD_ID_INIT;
    RM_FLCN_U64_PACK(&pInitCmd->dmaHandle, &dmaHandle);

    //
    // Make the interval twice the heartbeat period to avoid
    // skew between driver and soe threads
    //
    pInitCmd->driverPollingPeriodUs = (LWSWITCH_HEARTBEAT_INTERVAL_NS / 1000) * 2;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn,
                                (PRM_FLCN_CMD)&cmd,
                                NULL,   // pMsg             - not used for now
                                NULL,   // pPayload         - not used for now
                                SOE_RM_CMDQ_LOG_ID,
                                &cmdSeqDesc,
                                &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: SMBPBI Init command failed. rc:%d\n",
                        __FUNCTION__, status);
        return status;
    }

    lwswitch_lib_smbpbi_log_sxid(device, LWSWITCH_ERR_NO_ERROR,
                                 "LWSWITCH SMBPBI server is online.");

    LWSWITCH_PRINT(device, INFO, "%s: SMBPBI POST INIT completed\n", __FUNCTION__);

    return LWL_SUCCESS;
}

static void
_lwswitch_smbpbi_send_unload
(
    lwswitch_device *device
)
{
    FLCN                      *pFlcn;
    RM_FLCN_CMD_SOE            cmd;
    LWSWITCH_TIMEOUT           timeout;
    LwU32                      cmdSeqDesc;
    LwlStatus                  status;

    pFlcn = device->pSoe->pFlcn;

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));
    cmd.hdr.unitId = RM_SOE_UNIT_SMBPBI;
    cmd.hdr.size   = RM_SOE_CMD_SIZE(SMBPBI, UNLOAD);
    cmd.cmd.smbpbiCmd.cmdType = RM_SOE_SMBPBI_CMD_ID_UNLOAD;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn,
                                (PRM_FLCN_CMD)&cmd,
                                NULL,   // pMsg             - not used for now
                                NULL,   // pPayload         - not used for now
                                SOE_RM_CMDQ_LOG_ID,
                                &cmdSeqDesc,
                                &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: SMBPBI unload command failed. rc:%d\n",
                        __FUNCTION__, status);
    }
}

void
lwswitch_smbpbi_unload
(
    lwswitch_device *device
)
{
    if (device->pSmbpbi)
    {
        _lwswitch_smbpbi_send_unload(device);
        _lwswitch_smbpbi_dem_flush(device);
    }
}

void
lwswitch_smbpbi_destroy
(
    lwswitch_device *device
)
{
    if (device->pSmbpbi)
    {
        lwswitch_os_unmap_dma_region(device->os_handle,
                            device->pSmbpbi->sharedSurface,
                            device->pSmbpbi->dmaHandle,
                            sizeof(SOE_SMBPBI_SHARED_SURFACE),
                            LWSWITCH_DMA_DIR_BIDIRECTIONAL);
        lwswitch_os_free_contig_memory(device->os_handle, device->pSmbpbi->sharedSurface,
                            sizeof(SOE_SMBPBI_SHARED_SURFACE));
        lwswitch_os_free(device->pSmbpbi);
        device->pSmbpbi = NULL;
    }
}

LwlStatus
lwswitch_smbpbi_refresh_ecc_counts
(
    lwswitch_device *device
)
{
    PRM_SOE_SMBPBI_INFOROM_DATA pObjs;
    struct inforom              *pInforom = device->pInforom;
    LwU64                       corCnt;
    LwU64                       uncCnt;

    if ((device->pSmbpbi == NULL) || (device->pSmbpbi->sharedSurface == NULL))
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (pInforom == NULL || pInforom->pEccState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    device->hal.lwswitch_inforom_ecc_get_total_errors(device, pInforom->pEccState->pEcc,
                                                      &corCnt, &uncCnt);

    pObjs = &device->pSmbpbi->sharedSurface->inforomObjects;
    LwU64_ALIGN32_PACK(&(pObjs->ECC.correctedTotal), &corCnt);
    LwU64_ALIGN32_PACK(&(pObjs->ECC.uncorrectedTotal), &uncCnt);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_inforom_dem_load
(
    lwswitch_device *device
)
{
    LwlStatus                           status;
    LwU8                                version = 0;
    LwU8                                subversion = 0;
    struct inforom                      *pInforom = device->pInforom;
    LwU8                                *pPackedObject = NULL;
    struct INFOROM_DEM_OBJECT_V1_00     *pFifo;

    if ((pInforom == NULL) || (device->pSmbpbi == NULL) ||
        (device->pSmbpbi->sharedSurface == NULL))
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pFifo = GET_PFIFO_FROM_DEVICE(device);

    status = lwswitch_inforom_get_object_version_info(device, "DEM", &version,
                                                    &subversion);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "no DEM object found, rc:%d\n", status);
        goto lwswitch_inforom_dem_load_fail;
    }

    if (!INFOROM_OBJECT_SUBVERSION_SUPPORTS_LWSWITCH(subversion))
    {
        LWSWITCH_PRINT(device, WARN, "DEM v%u.%u not supported\n",
                    version, subversion);
        status = -LWL_ERR_NOT_SUPPORTED;
        goto lwswitch_inforom_dem_load_fail;
    }

    LWSWITCH_PRINT(device, INFO, "DEM v%u.%u found\n", version, subversion);

    if (version != 1)
    {
        LWSWITCH_PRINT(device, WARN, "DEM v%u.%u not supported\n",
                    version, subversion);
        status = -LWL_ERR_NOT_SUPPORTED;
        goto lwswitch_inforom_dem_load_fail;
    }

    pPackedObject = lwswitch_os_malloc(INFOROM_DEM_OBJECT_V1_00_PACKED_SIZE);

    if (pPackedObject == NULL)
    {
        status = -LWL_NO_MEM;
        goto lwswitch_inforom_dem_load_fail;
    }

    status = lwswitch_inforom_load_object(device, pInforom, "DEM",
                                        INFOROM_DEM_OBJECT_V1_00_FMT,
                                        pPackedObject,
                                        pFifo);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to load DEM object, rc: %d\n",
                        status);
        goto lwswitch_inforom_dem_load_fail;
    }

lwswitch_inforom_dem_load_fail:

    if (pPackedObject)
    {
        lwswitch_os_free(pPackedObject);
    }

    //
    // Mark the cached DEM as usable for Xid logging, even if we were
    // unable to find it in the InfoROM image.
    //

    device->pSmbpbi->sharedSurface->inforomObjects.DEM.bValid = LW_TRUE;

    _smbpbiDemInit(device, device->pSmbpbi, pFifo);

    return status;
}

/*!
 * Validate/Initialize the Driver Event Message (SXid) FIFO buffer
 *
 * @param[in]       device     device object pointer
 * @param[in]       pSmbpbi    SMBPBI object pointer
 * @param[in,out]   pFifo      DEM object pointer
 *
 * @return  void
 */
static void
_smbpbiDemInit
(
    lwswitch_device                     *device,
    struct smbpbi                       *pSmbpbi,
    struct INFOROM_DEM_OBJECT_V1_00     *pFifo
)
{
    LwU8                            msgLeft;
    unsigned                        recordsHeld = 0;
    LwU16                           FIFO_REC_LOOP_ITERATOR;
    LwU16                           bytesOclwpied;
    LwU16                           bytesSeen;
    LwBool                          status = LW_FALSE;

    // validate the FIFO buffer

    if ((DEM_FIFO_PTR(pFifo->writeOffset) != pFifo->writeOffset) ||
        (DEM_FIFO_PTR(pFifo->readOffset) != pFifo->readOffset)   ||
        ((pFifo->writeOffset % sizeof(LwU32)) != 0)              ||
        ((pFifo->readOffset % sizeof(LwU32)) != 0))
    {
        goto smbpbiDemInit_exit;
    }

    if (pFifo->writeOffset == pFifo->readOffset)
    {
        // The FIFO is empty
        status = LW_TRUE;
        goto smbpbiDemInit_exit;
    }

    //
    // This HAL extracts from a scratch register the count of DEM messages
    // in the FIFO that has not yet been requested by the SMBPBI client.
    // If the FIFO holds more messages than that, it means those in excess
    // of this count have been delivered to the client by PreOS app.
    //
    if (device->hal.lwswitch_smbpbi_get_dem_num_messages(device, &msgLeft) != LWL_SUCCESS)
    {
        // assume the maximum
        msgLeft = ~0;
    }

    if (msgLeft == 0)
    {
        // Nothing of value in the FIFO. Lets reset it explicitly.
        status = LW_TRUE;
        pFifo->writeOffset = 0;
        pFifo->readOffset = 0;
        goto smbpbiDemInit_exit;
    }

    //
    // Count the messages in the FIFO, while also checking the structure
    // for integrity. Reset the FIFO in case any corruption is found.
    //
    bytesOclwpied = DEM_BYTES_OCLWPIED(pFifo);

    bytesSeen = 0;
    FIFO_REC_LOOP_START(pFifo, bytesSeen < bytesOclwpied)
        if ((_recSize > DEM_RECORD_SIZE_MAX) ||
            (FIFO_REC_LOOP_REC_SIZE < DEM_RECORD_SIZE_MIN))
        {
            goto smbpbiDemInit_exit;
        }

        bytesSeen += FIFO_REC_LOOP_REC_SIZE;
        ++recordsHeld;
    FIFO_REC_LOOP_END

    if ((bytesSeen != bytesOclwpied) || (msgLeft > recordsHeld))
    {
        goto smbpbiDemInit_exit;
    }

    //
    // Advance the FIFO read ptr in order to remove those messages that
    // have already been delivered to the client.
    //
    FIFO_REC_LOOP_START(pFifo, recordsHeld > msgLeft)
        --recordsHeld;
    FIFO_REC_LOOP_END

    pFifo->readOffset =  FIFO_REC_LOOP_ITERATOR;
    status = LW_TRUE;

smbpbiDemInit_exit:

    if (!status)
    {
        // Reset the FIFO
        pFifo->writeOffset = 0;
        pFifo->readOffset = 0;
        pFifo->seqNumber = 0;
    }
}

static void
_lwswitch_smbpbi_dem_flush(lwswitch_device *device)
{
    LwU8                                *pPackedObject = NULL;
    struct INFOROM_DEM_OBJECT_V1_00     *pFifo;
    LwlStatus                           status = LWL_SUCCESS;

    pPackedObject = lwswitch_os_malloc(INFOROM_DEM_OBJECT_V1_00_PACKED_SIZE);

    if (pPackedObject == NULL)
    {
        status = -LWL_NO_MEM;
        goto _lwswitch_smbpbi_dem_flush_exit;
    }

    pFifo = GET_PFIFO_FROM_DEVICE(device);

    status = lwswitch_inforom_write_object(device, "DEM",
                                        INFOROM_DEM_OBJECT_V1_00_FMT,
                                        pFifo,
                                        pPackedObject);

_lwswitch_smbpbi_dem_flush_exit:
    lwswitch_os_free(pPackedObject);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "DEM object write failed, status=%d\n",
                       status);
    }
}

/*!
 * A helper to create a new DEM FIFO record
 *
 * @param[in,out]   pFifo           DEM object pointer
 * @param[in]       num             Xid number
 * @param[in]       osErrorString   text message to store
 * @param[in]       msglen          message size
 * @param[out]      pRecSize        new record size in bytes
 *
 * @return          ptr to the new record
 * @return          NULL if there's no room in the FIFO
 *                       or dynamic allocation error
 */
static LW_MSGBOX_DEM_RECORD *
_makeNewRecord
(
    INFOROM_DEM_OBJECT_V1_00    *pFifo,
    LwU32                       num,
    LwU8                        *osErrorString,
    LwU32                       msglen,
    LwU32                       *pRecSize
)
{
    LW_MSGBOX_DEM_RECORD                *pNewRec;

    *pRecSize = LW_MIN(sizeof(LW_MSGBOX_DEM_RECORD) + msglen,
                     DEM_RECORD_SIZE_MAX);

    if ((*pRecSize > DEM_BYTES_AVAILABLE(pFifo)) ||
        ((pNewRec = lwswitch_os_malloc(*pRecSize)) == NULL))
    {
        return NULL;
    }

    // Fill the new record.
    lwswitch_os_memset(pNewRec, 0, *pRecSize);
    pNewRec->recordSize = LW_UNSIGNED_DIV_CEIL(*pRecSize, sizeof(LwU32));
    pNewRec->xidId = num;
    pNewRec->seqNumber = pFifo->seqNumber++;
    pNewRec->timeStamp = lwswitch_os_get_platform_time() / LWSWITCH_NSEC_PER_SEC;

    if (msglen > LW_MSGBOX_MAX_DRIVER_EVENT_MSG_TXT_SIZE)
    {
        // The text string is too long. Truncate and notify the client.
        pNewRec->flags = FLD_SET_DRF(_MSGBOX, _DEM_RECORD_FLAGS,
                                       _TRUNC, _SET, pNewRec->flags);
        msglen = LW_MSGBOX_MAX_DRIVER_EVENT_MSG_TXT_SIZE - 1;
    }

    lwswitch_os_memcpy(pNewRec->textMessage, osErrorString, msglen);

    return pNewRec;
}

/*!
 * A helper to add the new record to the DEM FIFO
 *
 * @param[in,out]   pFifo           DEM object pointer
 * @param[in]       pNewRec         the new record
 * @param[in]       recSize         new record size in bytes
 *
 * @return          void
 */
static void
_addNewRecord
(
    INFOROM_DEM_OBJECT_V1_00    *pFifo,
    LW_MSGBOX_DEM_RECORD        *pNewRec,
    LwU32                       recSize
)
{
    LwU16   rem;
    LwU16   lwrPtr;
    LwU16   copySz;
    LwU8    *srcPtr;

    // Copy the new record into the FIFO, handling a possible wrap-around.
    rem = recSize;
    lwrPtr = pFifo->writeOffset;
    srcPtr = (LwU8 *)pNewRec;
    while (rem > 0)
    {
        copySz = LW_MIN(rem, DEM_FIFO_SIZE - lwrPtr);
        lwswitch_os_memcpy(pFifo->fifoBuffer + lwrPtr, srcPtr, copySz);
        rem -= copySz;
        srcPtr += copySz;
        lwrPtr = DEM_FIFO_PTR(lwrPtr + copySz);
    }

    // Advance the FIFO write ptr.
    pFifo->writeOffset = DEM_FIFO_PTR(pFifo->writeOffset +
                                      (pNewRec->recordSize * sizeof(LwU32)));
}

/*!
 * Add a Driver Event Message (SXid) to the InfoROM DEM FIFO buffer
 *
 * @param[in]   device          device object pointer
 * @param[in]   num             Xid number
 * @param[in]   msglen          message size
 * @param[in]   osErrorString   text message to store
 *
 * @return  void
 */
void
lwswitch_smbpbi_log_message
(
    lwswitch_device *device,
    LwU32           num,
    LwU32           msglen,
    LwU8            *osErrorString
)
{
    INFOROM_DEM_OBJECT_V1_00            *pFifo;
    LwU32                               recSize;
    LwU16                               FIFO_REC_LOOP_ITERATOR;
    LW_MSGBOX_DEM_RECORD                *pNewRec;

    if ((device->pSmbpbi == NULL) ||
        (device->pSmbpbi->sharedSurface == NULL))
    {
        return;
    }

    pFifo = GET_PFIFO_FROM_DEVICE(device);

    pNewRec = _makeNewRecord(pFifo, num, osErrorString, msglen, &recSize);

    if (pNewRec != NULL)
    {
        _addNewRecord(pFifo, pNewRec, recSize);
        lwswitch_os_free(pNewRec);
    }
    else
    {
        //
        // We are unable to log this message. Mark the latest record
        // with a flag telling the client that message(s) were dropped.
        //

        LwU16                   bytesOclwpied = DEM_BYTES_OCLWPIED(pFifo);
        LwU16                   bytesSeen;
        LW_MSGBOX_DEM_RECORD    *pLastRec = NULL;

        // Find the newest record
        bytesSeen = 0;
        FIFO_REC_LOOP_START(pFifo, bytesSeen < bytesOclwpied)
            pLastRec = FIFO_REC_LOOP_REC_PTR;
            bytesSeen += FIFO_REC_LOOP_REC_SIZE;
        FIFO_REC_LOOP_END

        if (pLastRec != NULL)
        {
            pLastRec->flags = FLD_SET_DRF(_MSGBOX, _DEM_RECORD_FLAGS,
                                           _OVFL, _SET, pLastRec->flags);
        }
    }

    return;
}

LwlStatus
lwswitch_smbpbi_set_link_error_info
(
    lwswitch_device *device,
    LWSWITCH_LINK_TRAINING_ERROR_INFO *pLinkTrainingErrorInfo,
    LWSWITCH_LINK_RUNTIME_ERROR_INFO  *pLinkRuntimeErrorInfo
)
{
    FLCN                                   *pFlcn;
    RM_FLCN_CMD_SOE                        cmd;
    LWSWITCH_TIMEOUT                       timeout;
    LwU32                                  cmdSeqDesc;
    RM_SOE_SMBPBI_CMD_SET_LINK_ERROR_INFO *pSetCmd = &cmd.cmd.smbpbiCmd.linkErrorInfo;
    LwlStatus                              status;

    if (!device->pSmbpbi)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pFlcn = device->pSoe->pFlcn;

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));
    cmd.hdr.unitId = RM_SOE_UNIT_SMBPBI;
    cmd.hdr.size   = RM_SOE_CMD_SIZE(SMBPBI, SET_LINK_ERROR_INFO);
    cmd.cmd.smbpbiCmd.cmdType = RM_SOE_SMBPBI_CMD_ID_SET_LINK_ERROR_INFO;

    pSetCmd->trainingErrorInfo.isValid = pLinkTrainingErrorInfo->isValid;
    pSetCmd->runtimeErrorInfo.isValid  = pLinkRuntimeErrorInfo->isValid;

    RM_FLCN_U64_PACK(&pSetCmd->trainingErrorInfo.attemptedTrainingMask0,
                     &pLinkTrainingErrorInfo->attemptedTrainingMask0);
    RM_FLCN_U64_PACK(&pSetCmd->trainingErrorInfo.trainingErrorMask0,
                     &pLinkTrainingErrorInfo->trainingErrorMask0);
    RM_FLCN_U64_PACK(&pSetCmd->runtimeErrorInfo.mask0, &pLinkRuntimeErrorInfo->mask0);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn,
                                 (PRM_FLCN_CMD)&cmd,
                                 NULL,   // pMsg            - not used for now
                                 NULL,   // pPayload        - not used for now
                                 SOE_RM_CMDQ_LOG_ID,
                                 &cmdSeqDesc,
                                 &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s SMBPBI Set Link Error Info command failed. rc:%d\n",
                       __FUNCTION__, status);
        return status;
    }

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* ADD NEW UNPUBLISHED CODE BELOW THIS LINE */

static LwlStatus
_lwswitch_smbpbi_cci_init
(
    lwswitch_device *device
)
{
    PCCI                            pCci = device->pCci;
    PSOE_SMBPBI_OSFP_DATA           pSsod;
    LwU32                           xcvrMaskAll = 0;
    LwU32                           xcvrMaskPresent = 0;
    LWSWITCH_CCI_GET_FW_REVISIONS   revisions[LWSWITCH_CCI_FW_IMAGE_COUNT];
    unsigned                        xcvrIdx;
    LwlStatus                       status;

    if ((pCci == NULL) || (device->pSmbpbi == NULL) ||
        (device->pSmbpbi->sharedSurface == NULL))
    {
        return LWL_SUCCESS;
    }

    status = cciGetXcvrMask(device, &xcvrMaskAll, &xcvrMaskPresent);

    if (status == -LWL_ERR_NOT_SUPPORTED)
    {
        return LWL_SUCCESS;
    }
    else if (status != LWL_SUCCESS)
    {
        return status;
    }
    else if (xcvrMaskAll == 0)
    {
        return LWL_SUCCESS;
    }

    pSsod = &device->pSmbpbi->sharedSurface->osfpData;

    pSsod->pingPongBuffIdx = 0;
    pSsod->pingPongBuff[0].xcvrMask.all = (LwU16)xcvrMaskAll;
    pSsod->pingPongBuff[0].xcvrMask.present = (LwU16)xcvrMaskPresent;

    status = _lwswitch_smbpbi_cci_update_hottest(device, pCci, pSsod);

    switch (status)
    {
        case -LWL_ERR_NOT_SUPPORTED:
            return LWL_SUCCESS;
        case LWL_SUCCESS:
            break;
        default:
            return status;
    }

    status = cciRomCache(device, 0);

    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s Caching FRU EEPROM failed. rc:%d\n",
                       __FUNCTION__, status);
    }

    FOR_EACH_INDEX_IN_MASK(16, xcvrIdx, xcvrMaskPresent)
    {
        LwU8    serialNumber[LW_MSGBOX_OPTICAL_XCEIVER_INFO_DATA_SIZE_SERIAL_NUMBER];
        LwU8    partNumber[LW_MSGBOX_OPTICAL_XCEIVER_INFO_DATA_SIZE_PART_NUMBER];
        LwU8    hwRevision[LW_MSGBOX_OPTICAL_XCEIVER_INFO_DATA_SIZE_HW_REVISION];
        LwU8    *pFru;

        status = cciGetXcvrFWRevisions(device, 0, xcvrIdx, revisions);

        if (status == LWL_SUCCESS)
        {
            unsigned    imgIdx;

            for (imgIdx = 0; imgIdx < LWSWITCH_CCI_FW_IMAGE_COUNT; ++imgIdx)
            {
                if (FLD_TEST_DRF(SWITCH_CCI, _FW_FLAGS, _ACTIVE, _YES,
                                revisions[imgIdx].flags))
                {
                    pSsod->info[xcvrIdx].firmwareVersion[0] =
                                        revisions[imgIdx].major;
                    pSsod->info[xcvrIdx].firmwareVersion[1] =
                                        revisions[imgIdx].minor;
                    *(LwU16 *)&pSsod->info[xcvrIdx].firmwareVersion[2] =
                                        revisions[imgIdx].build;
                    break;
                }
            }
        }

        status = cciGetXcvrStaticIdInfo(device, 0, xcvrIdx,
                                        serialNumber, partNumber, hwRevision, &pFru);
        if (status == LWL_SUCCESS)
        {
            lwswitch_os_memcpy(pSsod->info[xcvrIdx].serialNumber,
                                serialNumber,
                                sizeof(pSsod->info[xcvrIdx].serialNumber));
            lwswitch_os_memcpy(pSsod->info[xcvrIdx].partNumber,
                                partNumber,
                                sizeof(pSsod->info[xcvrIdx].partNumber));
            lwswitch_os_memcpy(pSsod->info[xcvrIdx].hardwareRevision,
                                hwRevision,
                                sizeof(pSsod->info[xcvrIdx].hardwareRevision));
            if (pFru != NULL)
            {
                lwswitch_os_memcpy(pSsod->info[xcvrIdx].fruEeprom,
                                    pFru,
                                    sizeof(pSsod->info[xcvrIdx].fruEeprom));
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    FOR_EACH_INDEX_IN_MASK(16, xcvrIdx, xcvrMaskAll)
    {
        status = cciGetXcvrLedState(device, 0, xcvrIdx, &pSsod->ledState[xcvrIdx]);

        if (status != LWL_SUCCESS)
        {
            pSsod->ledState[xcvrIdx] = 0;
            LWSWITCH_PRINT(device, ERROR, "%s reading LED state failed. rc:%d\n",
                       __FUNCTION__,
                       status);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    status = cciRegisterCallback(device, LWSWITCH_CCI_CALLBACK_SMBPBI,
                                 _lwswitch_smbpbi_cci_poll_callback,
                                 SMBPBI_CCI_POLLING_RATE_HZ);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s Registering callback failed rc:%d\n",
                       __FUNCTION__, status);
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_smbpbi_cci_update_led_state
(
    lwswitch_device         *device,
    PCCI                    pCci,
    PSOE_SMBPBI_OSFP_DATA   pSsod
)
{
    LwU8        buffIdx;
    LwU32       osfp;
    LwlStatus   status = LWL_SUCCESS;

    buffIdx = pSsod->pingPongBuffIdx;
    FOR_EACH_INDEX_IN_MASK(16, osfp, pSsod->pingPongBuff[buffIdx].xcvrMask.all)
    {
        status = cciGetXcvrLedState(device, 0, osfp, &pSsod->ledState[osfp]);
        if (status != LWL_SUCCESS)
        {
            goto _lwswitch_smbpbi_cci_update_led_state_exit;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

_lwswitch_smbpbi_cci_update_led_state_exit:
    return status;
}

static LwlStatus
_lwswitch_smbpbi_cci_update_hottest
(
    lwswitch_device         *device,
    PCCI                    pCci,
    PSOE_SMBPBI_OSFP_DATA   pSsod
)
{
    LwS16       *pOsfpTemperatures;
    LwU32       osfp;
    LwU32       xcvrMaskAll;
    LwU32       xcvrMaskPresent;
    LwU8        buffIdx;
    LwlStatus   status = LWL_SUCCESS;
    
    cciDetectXcvrsPresent(device);
    status = cciGetXcvrMask(device, NULL, &xcvrMaskPresent);

    if (status != LWL_SUCCESS)
    {   
        xcvrMaskPresent = 0;
    }

    buffIdx = (pSsod->pingPongBuffIdx + 1) % 2;
    pOsfpTemperatures = pSsod->pingPongBuff[buffIdx].temperature;

    FOR_EACH_INDEX_IN_MASK(16, osfp, xcvrMaskPresent)
    {
        LwTemp      temperature;
        unsigned    i;

        for (i = 0; i < SMBPBI_CCI_READ_TEMP_MAX_TRIES; ++i)
        {
            status = cciGetXcvrTemperature(device, 0, osfp, &temperature);

            if ((status == LWL_SUCCESS) || (status == -LWL_ERR_NOT_SUPPORTED))
            {
                break;
            }
        }
        
        if (status == LWL_SUCCESS)
        {
            pOsfpTemperatures[osfp] = (LwS16)temperature;
        }
        else if (status != -LWL_ERR_NOT_SUPPORTED)
        {
            pOsfpTemperatures[osfp] = LW_S16_MIN;
            cciSetXcvrPresent(device, osfp, LW_FALSE);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    status = cciGetXcvrMask(device, &xcvrMaskAll, &xcvrMaskPresent);

    if (status == LWL_SUCCESS)
    {
        pSsod->pingPongBuff[buffIdx].xcvrMask.all = (LwU16)xcvrMaskAll;
        pSsod->pingPongBuff[buffIdx].xcvrMask.present = (LwU16)xcvrMaskPresent;
        pSsod->pingPongBuffIdx = buffIdx;
    }
    
    return LWL_SUCCESS;
}

static void
_lwswitch_smbpbi_cci_process_cmd_fifo
(
    lwswitch_device         *device,
    PCCI                     pCci,
    PSOE_SMBPBI_OSFP_DATA    pSsod
)
{
    SOE_SMBPBI_OSFP_CMD_FIFO    *pFifo = &pSsod->cmdFifo;
    LwU8                         cmd;
    LwBool                       bSetLocate;

    while (OSFP_FIFO_BYTES_OCLWPIED(pFifo) > 0)
    {
        cmd = osfp_fifo_read_element(pFifo);

        switch(DRF_VAL(_SOE_SMBPBI_OSFP_FIFO, _CMD, _OPCODE, cmd))
        {
            case LW_SOE_SMBPBI_OSFP_FIFO_CMD_OPCODE_LED_LOCATE_OFF:
            {
                bSetLocate = LW_FALSE;
                break;
            }

            case LW_SOE_SMBPBI_OSFP_FIFO_CMD_OPCODE_LED_LOCATE_ON:
            {
                bSetLocate = LW_TRUE;
                break;
            }

            default:
            {
                LWSWITCH_PRINT(device, ERROR, "%s: Illegal FIFO command %02x\n",
                       __FUNCTION__, cmd);
                continue;
            }
        }

        (void)cciSetXcvrLedState(device, 0,
                                 DRF_VAL(_SOE_SMBPBI_OSFP_FIFO, _CMD, _ARG, cmd),
                                 bSetLocate);
    }
}

static void
_lwswitch_smbpbi_cci_poll_callback
(
    lwswitch_device *device
)
{
    PCCI                    pCci = device->pCci;
    PSOE_SMBPBI_OSFP_DATA   pSsod = &device->pSmbpbi->sharedSurface->osfpData;

    (void)_lwswitch_smbpbi_cci_update_hottest(device, pCci, pSsod);
    (void)_lwswitch_smbpbi_cci_process_cmd_fifo(device, pCci, pSsod);
    (void)_lwswitch_smbpbi_cci_update_led_state(device, pCci, pSsod);
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
