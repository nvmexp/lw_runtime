/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "regkey_lwswitch.h"
#include "lwVer.h"
#include "inforom/inforom_lwswitch.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
void
lwswitch_bbx_collect_lwrrent_time
(
    lwswitch_device     *device,
    void                *pBbxState
)
{
    LwU64 time_ns;

    time_ns = lwswitch_os_get_platform_time();
    ((INFOROM_BBX_STATE*)pBbxState)->timeLwrr = (LwU32)(time_ns / LWSWITCH_INTERVAL_1SEC_IN_NS);
}

static void
_lwswitch_bbx_collect_driver_version
(
    lwswitch_device     *device,
    INFOROM_BBX_STATE   *pBbxState
)
{
    const char *verDrvStr = LW_VERSION_STRING;
    LwU64       verNum    = 0;
    char        verChar;

    /*
     * Driver version is limited to 6 bytes
     * 2 ^ 48 = 281e+12 => 14 full decimal digits and partial 15th digit
     * Use 15th digit as a flag to indicate driver version truncation
     */
    while ((verChar = *verDrvStr++) != '\0')
    {
        if ((verChar >= '0') && (verChar <= '9'))
        {
            verNum = verNum * 10 + (verChar - '0');

            if (verNum >= INFOROM_BBX_OBJ_V1_00_SYSTEM_DRIVER_MAX)
            {
                verNum %= INFOROM_BBX_OBJ_V1_00_SYSTEM_DRIVER_MAX;
                verNum += INFOROM_BBX_OBJ_V1_00_SYSTEM_DRIVER_MAX;
            }
        }
    }

    pBbxState->systemState.v1_0.systemDriverLo = LwU64_LO32(verNum);
    pBbxState->systemState.v1_0.systemDriverHi = LwU32_LO16(LwU64_HI32(verNum));
}

static void
_lwswitch_bbx_collect_vbios_version
(
    lwswitch_device     *device,
    INFOROM_BBX_STATE   *pBbxState
)
{
    LwlStatus status;
    LWSWITCH_GET_BIOS_INFO_PARAMS biosParams;

    status = device->hal.lwswitch_ctrl_get_bios_info(device, &biosParams);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "Failed to collect BIOS version for BBX (rc:%d)\n", status);
        return;
    }

    pBbxState->systemState.v1_0.systemVbios = biosParams.version >> 8;
    pBbxState->systemState.v1_0.systemVbiosOem = biosParams.version & 0xff;
}

static void
_lwswitch_bbx_collect_os_version
(
    lwswitch_device *device,
    INFOROM_BBX_STATE *pBbxState
)
{
    LwlStatus status;
    LwU32     majorVer;
    LwU32     minorVer;
    LwU32     buildNum;
    LwU8      osType;

#if (defined(_WIN32) || defined(_WIN64))
        osType = INFOROM_BBX_OBJ_V1_00_SYSTEM_OS_TYPE_WIN;
#else
        osType = INFOROM_BBX_OBJ_V1_00_SYSTEM_OS_TYPE_UNIX;
#endif

    status = lwswitch_os_get_os_version(&majorVer, &minorVer, &buildNum);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "%s: Failed to get OS version, status=%d\n",
                    __FUNCTION__, status);
        return;
    }

    if ((majorVer > 0xff) || (minorVer > 0xff) || (buildNum > 0xffff))
    {
        LWSWITCH_PRINT(device, ERROR,
                    "Unexpected OS versions found. majorVer: 0x%x minorVer: 0x%x buildNum: 0x%x\n",
                    majorVer, minorVer, buildNum);
        return;
    }

    pBbxState->systemState.v1_0.systemOsType = osType;
    pBbxState->systemState.v1_0.systemOs =
        REF_NUM(INFOROM_BBX_OBJ_V1_00_SYSTEM_OS_MAJOR, majorVer) |
        REF_NUM(INFOROM_BBX_OBJ_V1_00_SYSTEM_OS_MINOR, minorVer) |
        REF_NUM(INFOROM_BBX_OBJ_V1_00_SYSTEM_OS_BUILD, buildNum);
}

static LwlStatus
_lwswitch_bbx_collect_temperature
(
    lwswitch_device     *device,
    INFOROM_BBX_STATE   *pBbxState,
    LwTemp              *pTempLwrr
)
{
    LwlStatus status;
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS tempInfo;

    tempInfo.channelMask = LWBIT(LWSWITCH_THERM_CHANNEL_LR10_TSENSE_OFFSET_MAX);
    status = device->hal.lwswitch_ctrl_therm_read_temperature(device,
                                                            &tempInfo);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "Failed to collect temperature for BBX, rc: %d\n", status);
        return status;
    }

    *pTempLwrr = tempInfo.temperature[LWSWITCH_THERM_CHANNEL_LR10_TSENSE_OFFSET_MAX];

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_inforom_bbx_add_data
(
    lwswitch_device     *device,
    INFOROM_BBX_STATE   *pBbxState,
    INFOROM_BBX_DATA    *pData
)
{
    LwlStatus status;

    if (!pBbxState->bValid)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    switch (pData->type)
    {
        case INFOROM_BBX_DATA_TYPE_TIME:
            status = device->hal.lwswitch_bbx_add_data_time(device,
                                                        pBbxState, pData);
            break;

        case INFOROM_BBX_DATA_TYPE_XID:
            status = device->hal.lwswitch_bbx_add_sxid(device, pBbxState,
                                                    pData);
            break;

        case INFOROM_BBX_DATA_TYPE_TEMPERATURE:
            status = device->hal.lwswitch_bbx_add_temperature(device, pBbxState,
                                                    pData);
            break;
        default:
            status = -LWL_BAD_ARGS;
            break;
    }

    return status;
}

static void
_lwswitch_bbx_get_static_params
(
    lwswitch_device   *device,
    struct inforom    *pInforom,
    PINFOROM_BBX_STATE pBbxState
)
{
    INFOROM_BBX_DATA bbxData;

    lwswitch_os_memset(&bbxData, 0, sizeof(bbxData));

    lwswitch_bbx_collect_lwrrent_time(device, pBbxState);

    _lwswitch_bbx_collect_driver_version(device, pBbxState);

    _lwswitch_bbx_collect_os_version(device, pBbxState);

    _lwswitch_bbx_collect_vbios_version(device, pBbxState);

    if (_lwswitch_bbx_collect_temperature(device, pBbxState,
                                    &bbxData.temperature.value) != LWL_SUCCESS)
    {
        bbxData.temperature.value = 0;
    }
    device->hal.lwswitch_bbx_set_initial_temperature(device, pBbxState, &bbxData);
}

/*!
 * BBX routine to collect dynamic LWSwitch parameters, called periodically.
 */
static void
_lwswitch_bbx_get_dynamic_params
(
    lwswitch_device   *device,
    struct inforom    *pInforom,
    INFOROM_BBX_STATE *pBbxState
)
{
    INFOROM_BBX_DATA bbxData;

    // Collect and add current time
    lwswitch_bbx_collect_lwrrent_time(device, pBbxState);
    lwswitch_os_memset(&bbxData, 0, sizeof(bbxData));
    bbxData.type = INFOROM_BBX_DATA_TYPE_TIME;
    bbxData.time.sec = pBbxState->timeLwrr;
    _lwswitch_inforom_bbx_add_data(device, pBbxState, &bbxData);

    // Collect and add temperature
    lwswitch_os_memset(&bbxData, 0, sizeof(bbxData));
    if (_lwswitch_bbx_collect_temperature(device, pBbxState,
                                    &bbxData.temperature.value) == LWL_SUCCESS)
    {
        bbxData.type = INFOROM_BBX_DATA_TYPE_TEMPERATURE;
        _lwswitch_inforom_bbx_add_data(device, pBbxState, &bbxData);
    }
}

static LwlStatus
_lwswitch_bbx_flush_object
(
    lwswitch_device   *device,
    struct inforom    *pInforom,
    INFOROM_BBX_STATE *pBbxState
)
{
    LwlStatus status;

    if (!pBbxState->bFlushImmediately)
    {
        return LWL_SUCCESS;
    }

    status = lwswitch_inforom_write_object(device, "BBX", pBbxState->uPFmt.pFmt,
                pBbxState->pObject, pBbxState->pPackedObject);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "BBX object write failed, status=%d\n",
                  status);
        return status;
    }

    pBbxState->bFlushImmediately  = LW_FALSE;
    pBbxState->timeSinceLastFlush = 0;
    return status;
}

static void
_lwswitch_bbx_update_params
(
    lwswitch_device *device
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_BBX_STATE *pBbxState;

    if (pInforom == NULL)
    {
        return;
    }

    pBbxState = pInforom->pBbxState;
    if (pBbxState == NULL)
    {
        return;
    }

    if (!pBbxState->bValid)
    {
        return;
    }

    pBbxState->timeSinceLastFlush += 1;

    _lwswitch_bbx_get_dynamic_params(device, pInforom, pBbxState);

    // Periodically flush the BBX back to Inforom.
    pBbxState->bFlushImmediately = (pBbxState->timeSinceLastFlush >= pBbxState->flushInterval &&
         device->regkeys.inforom_bbx_periodic_flush == LW_SWITCH_REGKEY_INFOROM_BBX_ENABLE_PERIODIC_FLUSHING_ENABLE);

    _lwswitch_bbx_flush_object(device, pInforom, pBbxState);
}

/*!
 * Start the periodic data collection for the BBX object.
 */
static LwlStatus
_lwswitch_bbx_start
(
    lwswitch_device *device,
    struct inforom  *pInforom
)
{
    INFOROM_BBX_STATE *pBbxState = pInforom->pBbxState;
    LwlStatus status;

    status = device->hal.lwswitch_bbx_setup_prologue(device, pBbxState);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "BBX setup prologue failed, status=%d\n",
                  status);
        pBbxState->bValid = LW_FALSE;
        return status;
    }

    _lwswitch_bbx_get_static_params(device, pInforom, pBbxState);

    status = device->hal.lwswitch_bbx_setup_epilogue(device, pBbxState);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "BBX setup epilogue failed, status=%d\n",
                    status);
    }

    lwswitch_task_create(device, &_lwswitch_bbx_update_params,
                    LWSWITCH_INTERVAL_1SEC_IN_NS, 0);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_inforom_bbx_add_sxid
(
    lwswitch_device *device,
    LwU32            exceptionType,
    LwU32            data0,
    LwU32            data1,
    LwU32            data2
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_BBX_STATE *pBbxState;
    INFOROM_BBX_DATA bbxData;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pBbxState = pInforom->pBbxState;
    if (pBbxState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwswitch_os_memset(&bbxData, 0, sizeof(bbxData));
    bbxData.type = INFOROM_BBX_DATA_TYPE_XID;
    bbxData.xid.XidNumber = exceptionType;
    bbxData.xid.data[0]   = data0;
    bbxData.xid.data[1]   = data1;
    bbxData.xid.data[2]   = data2;
    _lwswitch_inforom_bbx_add_data(device, pBbxState, &bbxData);

    return LWL_SUCCESS;
}

void
lwswitch_inforom_bbx_unload
(
    lwswitch_device *device
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_BBX_STATE *pBbxState;

    if (pInforom == NULL)
    {
        return;
    }

    pBbxState = pInforom->pBbxState;
    if (pBbxState == NULL)
    {
        return;
    }

    pBbxState->bFlushImmediately |=
        (pBbxState->timeSinceLastFlush >= pBbxState->flushMinDuration);

    _lwswitch_bbx_flush_object(device, pInforom, pBbxState);

    lwswitch_os_free(pBbxState->pPackedObject);
    lwswitch_os_free(pBbxState->pObject);
    lwswitch_os_free(pBbxState);
    pInforom->pBbxState = NULL;
}

LwlStatus
lwswitch_inforom_bbx_load
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwU8      version;
    LwU8      subversion;
    struct inforom *pInforom = device->pInforom;
    INFOROM_BBX_STATE *pBbxState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = lwswitch_inforom_get_object_version_info(device, "BBX", &version,
                                                    &subversion);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, WARN, "no BBX object found, rc:%d\n", status);
        return LWL_SUCCESS;
    }

    if (!INFOROM_OBJECT_SUBVERSION_SUPPORTS_LWSWITCH(subversion))
    {
        LWSWITCH_PRINT(device, WARN, "BBX v%u.%u not supported\n",
                    version, subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    LWSWITCH_PRINT(device, INFO, "BBX v%u.%u found\n", version, subversion);

    pBbxState = lwswitch_os_malloc(sizeof(INFOROM_BBX_STATE));
    if (pBbxState == NULL)
    {
        return -LWL_NO_MEM;
    }
    lwswitch_os_memset(pBbxState, 0, sizeof(INFOROM_BBX_STATE));

    switch (version)
    {
        case 2:
            pBbxState->uPFmt.pFmt = INFOROM_BBX_OBJ_V2_S0_FMT;
            pBbxState->pPackedObject = lwswitch_os_malloc(INFOROM_BBX_OBJ_V2_S0_PKD_SIZE);
            if (pBbxState->pPackedObject == NULL)
            {
                status = -LWL_NO_MEM;
                goto lwswitch_bbx_load_object_fail;
            }

            pBbxState->pObject = lwswitch_os_malloc(sizeof(INFOROM_BBX_OBJECT));
            if (pBbxState->pObject == NULL)
            {
                status = -LWL_NO_MEM;
                lwswitch_os_free(pBbxState->pPackedObject);
                goto lwswitch_bbx_load_object_fail;
            }
            break;

        default:
            LWSWITCH_PRINT(device, ERROR, "BBX object invalid version (%d.%d)\n",
                      version, subversion);
            status = -LWL_ERR_NOT_SUPPORTED;
            goto lwswitch_bbx_load_object_fail;
            break;
    }

    status = lwswitch_inforom_load_object(device, pInforom, "BBX",
                        pBbxState->uPFmt.pFmt, pBbxState->pPackedObject,
                        pBbxState->pObject);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "BBX object read failed, status=%d",
                        status);
        goto lwswitch_bbx_load_fail;
    }

    pBbxState->flushInterval = device->regkeys.inforom_bbx_write_periodicity;
    pBbxState->flushMinDuration = device->regkeys.inforom_bbx_write_min_duration;
    pBbxState->bValid = LW_TRUE;
    pInforom->pBbxState = pBbxState;

    _lwswitch_bbx_start(device, pInforom);

    return LWL_SUCCESS;

lwswitch_bbx_load_fail:
        lwswitch_os_free(pBbxState->pPackedObject);
        lwswitch_os_free(pBbxState->pObject);
lwswitch_bbx_load_object_fail:
        lwswitch_os_free(pBbxState);

    return status;
}

LwlStatus
lwswitch_inforom_bbx_get_sxid
(
    lwswitch_device *device,
    LWSWITCH_GET_SXIDS_PARAMS *params
)
{
    struct inforom *pInforom = device->pInforom;

    if ((pInforom == NULL) || (pInforom->pBbxState == NULL))
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return device->hal.lwswitch_inforom_bbx_get_sxid(device, params);
}
#else
void
lwswitch_bbx_collect_lwrrent_time
(
    lwswitch_device     *device,
    void                *pBbxState
)
{
    return;
}

LwlStatus
lwswitch_inforom_bbx_add_sxid
(
    lwswitch_device *device,
    LwU32            exceptionType,
    LwU32            data0,
    LwU32            data1,
    LwU32            data2
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

void
lwswitch_inforom_bbx_unload
(
    lwswitch_device *device
)
{
    return;
}

LwlStatus
lwswitch_inforom_bbx_load
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_bbx_get_sxid
(
    lwswitch_device *device,
    LWSWITCH_GET_SXIDS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
