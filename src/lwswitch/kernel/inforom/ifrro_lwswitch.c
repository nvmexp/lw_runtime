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
#include "inforom/inforom_lwswitch.h"

LwlStatus
lwswitch_inforom_read_only_objects_load
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

    status = lwswitch_inforom_load_object(device, pInforom, "OBD",
                                        INFOROM_OBD_OBJECT_V1_XX_FMT,
                                        pInforom->OBD.packedObject,
                                        &pInforom->OBD.object);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to load OBD object, rc:%d\n",
                    status);
    }
    else
    {
        pInforom->OBD.bValid = LW_TRUE;
    }

    status = lwswitch_inforom_load_object(device, pInforom, "OEM",
                                        INFOROM_OEM_OBJECT_V1_00_FMT,
                                        pInforom->OEM.packedObject,
                                        &pInforom->OEM.object);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to load OEM object, rc:%d\n",
                    status);
    }
    else
    {
        pInforom->OEM.bValid = LW_TRUE;
    }

    status = lwswitch_inforom_load_object(device, pInforom, "IMG",
                                        INFOROM_IMG_OBJECT_V1_00_FMT,
                                        pInforom->IMG.packedObject,
                                        &pInforom->IMG.object);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to load IMG object, rc:%d\n",
                    status);
    }
    else
    {
        pInforom->IMG.bValid = LW_TRUE;
    }

    return LWL_SUCCESS;
}
