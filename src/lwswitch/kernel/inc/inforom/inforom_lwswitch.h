/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _INFOROM_LWSWITCH_H_
#define _INFOROM_LWSWITCH_H_

#include "inforom/ifrstruct.h"
#include "inforom/omsdef.h"
#include "lw_list.h"
#include "smbpbi_shared_lwswitch.h"

#define INFOROM_MAX_PACKED_SIZE (32*1024)

#define INFOROM_FS_FILE_SIZE(pPackedFile) \
    (((pPackedFile)[INFOROM_OBJECT_HEADER_V1_00_SIZE_OFFSET]) | \
     ((pPackedFile)[INFOROM_OBJECT_HEADER_V1_00_SIZE_OFFSET + 1] << 8))
#define INFOROM_FS_FILE_NAMES_MATCH(fileName1, fileName2) \
    ((((LwU8)((fileName1)[0])) == ((LwU8)((fileName2)[0]))) && \
     (((LwU8)((fileName1)[1])) == ((LwU8)((fileName2)[1]))) && \
     (((LwU8)((fileName1)[2])) == ((LwU8)((fileName2)[2]))))

#define INFOROM_FS_COPY_FILE_NAME(destName, srcName) \
{                                                    \
    (destName)[0] = (srcName)[0];                    \
    (destName)[1] = (srcName)[1];                    \
    (destName)[2] = (srcName)[2];                    \
}

struct INFOROM_OBJECT_CACHE_ENTRY
{
    INFOROM_OBJECT_HEADER_V1_00         header;
    struct INFOROM_OBJECT_CACHE_ENTRY  *pNext;
};

struct inforom
{
    // InfoROM Objects
    // RO objects - statically allocated as the InfoROM should always contain
    // these objects.
    struct
    {
        LwBool                      bValid;
        LwU8                        packedObject[INFOROM_OBD_OBJECT_V1_XX_PACKED_SIZE];
        INFOROM_OBD_OBJECT_V1_XX    object;
    } OBD;

    struct
    {
        LwBool                      bValid;
        LwU8                        packedObject[INFOROM_OEM_OBJECT_V1_00_PACKED_SIZE];
        INFOROM_OEM_OBJECT_V1_00    object;
    } OEM;

    struct
    {
        LwBool                      bValid;
        LwU8                        packedObject[INFOROM_IMG_OBJECT_V1_00_PACKED_SIZE];
        INFOROM_IMG_OBJECT_V1_00    object;
    } IMG;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_STATE               *pLwlinkState;
    INFOROM_BBX_STATE                  *pBbxState;
#endif
    INFOROM_ECC_STATE                  *pEccState;
    INFOROM_OMS_STATE                  *pOmsState;

    //
    // descriptor cache for all the inforom objects. This is to handle inforom objects in a generic way.
    //
    struct INFOROM_OBJECT_CACHE_ENTRY   *pObjectCache;
};

// Generic InfoROM APIs
LwlStatus lwswitch_initialize_inforom(lwswitch_device *device);
LwlStatus lwswitch_inforom_read_object(lwswitch_device* device,
                const char *objectName, const char *pObjectFormat,
                LwU8 *pPackedObject, void *pObject);
LwlStatus lwswitch_inforom_write_object(lwswitch_device* device,
                const char *objectName, const char *pObjectFormat,
                void *pObject, LwU8 *pOldPackedObject);
void lwswitch_destroy_inforom(lwswitch_device *device);
LwlStatus lwswitch_inforom_add_object(struct inforom *pInforom,
                                    INFOROM_OBJECT_HEADER_V1_00 *pHeader);
LwlStatus lwswitch_inforom_get_object_version_info(lwswitch_device *device,
                const char *objectName, LwU8 *pVersion, LwU8 *pSubVersion);
void *lwswitch_add_halinfo_node(LWListPtr head, int type, int size);
void *lwswitch_get_halinfo_node(LWListPtr head, int type);
void lwswitch_inforom_post_init(lwswitch_device *device);
LwlStatus lwswitch_initialize_inforom_objects(lwswitch_device *device);
void lwswitch_destroy_inforom_objects(lwswitch_device *device);
LwlStatus lwswitch_inforom_load_object(lwswitch_device* device,
                struct inforom *pInforom, const char *objectName,
                const char *pObjectFormat, LwU8 *pPackedObject, void *pObject);
void lwswitch_inforom_read_static_data(lwswitch_device *device,
                struct inforom  *pInforom, RM_SOE_SMBPBI_INFOROM_DATA *pData);

// InfoROM RO APIs
LwlStatus lwswitch_inforom_read_only_objects_load(lwswitch_device *device);

// InfoROM LWL APIs
LwlStatus lwswitch_inforom_lwlink_load(lwswitch_device *device);
void lwswitch_inforom_lwlink_unload(lwswitch_device *device);
LwlStatus lwswitch_inforom_lwlink_flush(lwswitch_device *device);
LwlStatus lwswitch_inforom_lwlink_get_minion_data(lwswitch_device *device,
                                            LwU8 linkId, LwU32 *seedData);
LwlStatus lwswitch_inforom_lwlink_set_minion_data(lwswitch_device *device,
                                LwU8 linkId, LwU32 *seedData, LwU32 size);
LwlStatus lwswitch_inforom_lwlink_log_error_event(lwswitch_device *device, void *error_event);
LwlStatus lwswitch_inforom_lwlink_get_max_correctable_error_rate(lwswitch_device *device,
                LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params);
LwlStatus lwswitch_inforom_lwlink_get_errors(lwswitch_device *device,
                                LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params);

// InfoROM ECC APIs
LwlStatus lwswitch_inforom_ecc_load(lwswitch_device *device);
void lwswitch_inforom_ecc_unload(lwswitch_device *device);
LwlStatus lwswitch_inforom_ecc_flush(lwswitch_device *device);
LwlStatus lwswitch_inforom_ecc_log_err_event(lwswitch_device *device,
                                INFOROM_LWS_ECC_ERROR_EVENT *err_event);
LwlStatus lwswitch_inforom_ecc_get_errors(lwswitch_device *device,
                                LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *params);

// InfoROM OMS APIs
LwlStatus lwswitch_inforom_oms_load(lwswitch_device *device);
void lwswitch_inforom_oms_unload(lwswitch_device *device);
LwlStatus lwswitch_inforom_oms_set_device_disable(lwswitch_device *device,
                                        LwBool bDisable);
LwlStatus lwswitch_inforom_oms_get_device_disable(lwswitch_device *device,
                                        LwBool *pBDisabled);

// InfoROM BBX APIs
LwlStatus lwswitch_inforom_bbx_load(lwswitch_device *device);
void lwswitch_inforom_bbx_unload(lwswitch_device * device);
LwlStatus lwswitch_inforom_bbx_add_sxid(lwswitch_device *device,
                                    LwU32 exceptionType, LwU32 data0,
                                    LwU32 data1, LwU32 data2);
void lwswitch_bbx_collect_lwrrent_time(lwswitch_device *device,
                            void *pBbxState);
LwlStatus lwswitch_inforom_bbx_get_sxid(lwswitch_device *device,
                            LWSWITCH_GET_SXIDS_PARAMS *params);

// InfoROM DEM APIs
LwlStatus lwswitch_inforom_dem_load(lwswitch_device *device);
void lwswitch_inforom_dem_unload(lwswitch_device * device);
#endif // _INFOROM_LWSWITCH_H_
