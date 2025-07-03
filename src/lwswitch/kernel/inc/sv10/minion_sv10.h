/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _MINION_SV10_H_
#define _MINION_SV10_H_

#include "sv10.h"

#define FALCON_IMEM_BLK_SIZE_BYTES_SV10              256

#define FALCON_CODE_HDR_OS_CODE_OFFSET_SV10          0
#define FALCON_CODE_HDR_OS_CODE_SIZE_SV10            1
#define FALCON_CODE_HDR_OS_DATA_OFFSET_SV10          2
#define FALCON_CODE_HDR_OS_DATA_SIZE_SV10            3
#define FALCON_CODE_HDR_NUM_APPS_SV10                4
#define FALCON_CODE_HDR_APP_CODE_START_SV10          5
#define FALCON_CODE_HDR_APP_DATA_START_SV10          ( FALCON_CODE_HDR_APP_CODE_START_SV10 + (FALCON_CODE_HDR_NUM_APPS_SV10 * 2))
#define FALCON_CODE_HDR_CODE_OFFSET_SV10             0
#define FALCON_CODE_HDR_CODE_SIZE_SV10               1
#define FALCON_CODE_HDR_DATA_OFFSET_SV10             0
#define FALCON_CODE_HDR_DATA_SIZE_SV10               1

typedef const struct
{
    LwU32 osCodeOffset;
    LwU32 osCodeSize;
    LwU32 osDataOffset;
    LwU32 osDataSize;
    LwU32 numApps;
    LwU32 appCodeStart;
    LwU32 appDataStart;
    LwU32 codeOffset;
    LwU32 codeSize;
    LwU32 dataOffset;
    LwU32 dataSize;
} FALCON_UCODE_HDR_INFO_SV10, *PFALCON_UCODE_HDR_INFO_SV10;

#define LWSWITCH_MINION_LINK_WR32_SV10(_d, _physlinknum, _dev, _reg, _data)   \
    lwswitch_reg_write_32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(device)->link[_physlinknum].engMINION->uc_addr + LW##_dev##_reg, _data);    \
    ((void)(_d))

#define LWSWITCH_MINION_LINK_RD32_SV10(_d, _physlinknum, _dev, _reg)          \
    (                                                               \
        lwswitch_reg_read_32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(device)->link[_physlinknum].engMINION->uc_addr + LW##_dev##_reg)    \
    );                                                              \
    ((void)(_d))

#define LWSWITCH_MINION_WR32_SV10(_d, _instance, _dev, _reg, _data)                                       \
    LWSWITCH_SUBENG_WR32_SV10(_d, SIOCTRL, , _instance, MINION, , 0, uc, _dev, _reg, _data)

#define LWSWITCH_MINION_RD32_SV10(_d, _instance, _dev, _reg)                                              \
    LWSWITCH_SUBENG_RD32_SV10(_d, SIOCTRL, _instance, MINION, 0, _dev, _reg)

#define LWSWITCH_MINION_WR32_BCAST_SV10(_d, _dev, _reg, _data)                                            \
    LWSWITCH_BCAST_WR32_SV10(_d, SIOCTRL, MINION, , _dev, _reg, _data)


//
// Internal function declarations
//
LwlStatus lwswitch_init_minion_sv10(lwswitch_device *device);
LwlStatus lwswitch_minion_send_command_sv10(lwswitch_device *device, LwU32 linkNumber, LwU32 command, LwU32 scratch0);
LwlStatus lwswitch_minion_get_dl_status_sv10(lwswitch_device *device, LwU32 linkId, LwU32 statusIdx, LwU32 statusArgs, LwU32 *statusData);
LwlStatus lwswitch_set_minion_initialized_sv10(lwswitch_device *device, LwU32 idx_minion, LwBool initialized);
LwBool    lwswitch_is_minion_initialized_sv10(lwswitch_device *device, LwU32 idx_minion);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus lwswitch_ctrl_config_eom_sv10(lwswitch_device *device, LWSWITCH_CTRL_CONFIG_EOM *p);
LwlStatus lwswitch_ctrl_read_uphy_pad_lane_reg_sv10(lwswitch_device *device, LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG *p);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#endif //_MINION_SV10_H_
