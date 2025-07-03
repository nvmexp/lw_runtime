/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _MINION_LR10_H_
#define _MINION_LR10_H_

#include "lr10.h"

// TODO modify these for LR10
#define FALCON_IMEM_BLK_SIZE_BYTES_LR10              256

#define FALCON_CODE_HDR_OS_CODE_OFFSET_LR10          0
#define FALCON_CODE_HDR_OS_CODE_SIZE_LR10            1
#define FALCON_CODE_HDR_OS_DATA_OFFSET_LR10          2
#define FALCON_CODE_HDR_OS_DATA_SIZE_LR10            3
#define FALCON_CODE_HDR_NUM_APPS_LR10                4
#define FALCON_CODE_HDR_APP_CODE_START_LR10          5
#define FALCON_CODE_HDR_APP_DATA_START_LR10          ( FALCON_CODE_HDR_APP_CODE_START_LR10 + (FALCON_CODE_HDR_NUM_APPS_LR10 * 2))
#define FALCON_CODE_HDR_CODE_OFFSET_LR10             0
#define FALCON_CODE_HDR_CODE_SIZE_LR10               1
#define FALCON_CODE_HDR_DATA_OFFSET_LR10             0
#define FALCON_CODE_HDR_DATA_SIZE_LR10               1

#define LW_MINION_LWLINK_DL_STAT_ARGS_LANEID  15:12
#define LW_MINION_LWLINK_DL_STAT_ARGS_ADDRS   11:0

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
} FALCON_UCODE_HDR_INFO_LR10, *PFALCON_UCODE_HDR_INFO_LR10;

#define LWSWITCH_MINION_LINK_RD32_LR10(_d, _physlinknum, _dev, _reg) \
    LWSWITCH_LINK_RD32_LR10(_d, _physlinknum, MINION, _dev, _reg)

#define LWSWITCH_MINION_LINK_WR32_LR10(_d, _physlinknum, _dev, _reg, _data) \
    LWSWITCH_LINK_WR32_LR10(_d, _physlinknum, MINION, _dev, _reg, _data)

#define LWSWITCH_MINION_WR32_LR10(_d, _instance, _dev, _reg, _data)         \
       LWSWITCH_ENG_WR32_LR10(_d, MINION, , _instance, _dev, _reg, _data)

#define LWSWITCH_MINION_RD32_LR10(_d, _instance, _dev, _reg)                \
    LWSWITCH_ENG_RD32_LR10(_d, MINION, _instance, _dev, _reg)

#define LWSWITCH_MINION_WR32_BCAST_LR10(_d, _dev, _reg, _data)              \
    LWSWITCH_BCAST_WR32_LR10(_d, MINION, _dev, _reg, _data)

#define LWSWITCH_MINION_GET_LOCAL_LINK_ID(_physlinknum) \
    (_physlinknum%LWSWITCH_LINKS_PER_MINION)

//
// Internal function declarations
//
LwlStatus lwswitch_init_minion_lr10(lwswitch_device *device);
LwlStatus lwswitch_minion_send_command_lr10(lwswitch_device *device, LwU32 linkNumber, LwU32 command, LwU32 scratch0);
LwlStatus lwswitch_minion_get_dl_status_lr10(lwswitch_device *device, LwU32 linkId, LwU32 statusIdx, LwU32 statusArgs, LwU32 *statusData);
LwlStatus lwswitch_minion_get_initoptimize_status_lr10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_minion_get_initnegotiate_status_lr10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_minion_get_rxdet_status_lr10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_minion_set_rx_term_lr10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_minion_restore_seed_data_lr10(lwswitch_device *device, LwU32 linkId, LwU32 *seedData);
LwlStatus lwswitch_minion_save_seed_data_lr10(lwswitch_device *device, LwU32 linkId, LwU32 *seedData);
LwU32     lwswitch_minion_get_line_rate_Mbps_lr10(lwswitch_device *device, LwU32 linkId);
LwU32     lwswitch_minion_get_data_rate_KiBps_lr10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_set_minion_initialized_lr10(lwswitch_device *device, LwU32 idx_minion, LwBool initialized);
LwBool    lwswitch_is_minion_initialized_lr10(lwswitch_device *device, LwU32 idx_minion);
LwlStatus lwswitch_minion_clear_dl_error_counters_lr10(lwswitch_device *device, LwU32 linkId);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus lwswitch_ctrl_config_eom_lr10(lwswitch_device *device, LWSWITCH_CTRL_CONFIG_EOM *p);
LwlStatus lwswitch_ctrl_read_uphy_pad_lane_reg_lr10(lwswitch_device *device, LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG *p);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#endif //_MINION_LR10_H_
