/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "common_lwswitch.h"
#include "error_lwswitch.h"
#include "rmsoecmdif.h"
#include "spi_lwswitch.h"
#include "flcn/flcn_lwswitch.h"
#include "rmflcncmdif_lwswitch.h"

LwlStatus
lwswitch_spi_init
(
    lwswitch_device *device
)
{
    RM_FLCN_CMD_SOE     cmd;
    LWSWITCH_TIMEOUT    timeout;
    LwU32               cmdSeqDesc;
    LW_STATUS           status;
    FLCN               *pFlcn;

    if (!device->pSoe)
    {
        return -LWL_ERR_ILWALID_STATE;
    }

    pFlcn = device->pSoe->pFlcn;

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));
    cmd.hdr.unitId = RM_SOE_UNIT_SPI;
    cmd.hdr.size   = sizeof(cmd);
    cmd.cmd.spi.cmdType = RM_SOE_SPI_INIT;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * 30, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn,
                                (PRM_FLCN_CMD)&cmd,
                                NULL,   // pMsg             - not used for now
                                NULL,   // pPayload         - not used for now
                                SOE_RM_CMDQ_LOG_ID,
                                &cmdSeqDesc,
                                &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: SPI INIT failed. rc:%d\n",
                        __FUNCTION__, status);
    }

    return status;
}
