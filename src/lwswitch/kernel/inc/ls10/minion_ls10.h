/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _MINION_LS10_H_
#define _MINION_LS10_H_

#include "ls10.h"

#define FALCON_IMEM_BLK_SIZE_BYTES_LS10              256

#define FALCON_CODE_HDR_OS_CODE_OFFSET_LS10          0
#define FALCON_CODE_HDR_OS_CODE_SIZE_LS10            1
#define FALCON_CODE_HDR_OS_DATA_OFFSET_LS10          2
#define FALCON_CODE_HDR_OS_DATA_SIZE_LS10            3
#define FALCON_CODE_HDR_NUM_APPS_LS10                4
#define FALCON_CODE_HDR_APP_CODE_START_LS10          5
#define FALCON_CODE_HDR_APP_DATA_START_LS10          ( FALCON_CODE_HDR_APP_CODE_START_LS10 + (FALCON_CODE_HDR_NUM_APPS_LS10 * 2))
#define FALCON_CODE_HDR_CODE_OFFSET_LS10             0
#define FALCON_CODE_HDR_CODE_SIZE_LS10               1
#define FALCON_CODE_HDR_DATA_OFFSET_LS10             0
#define FALCON_CODE_HDR_DATA_SIZE_LS10               1

#define LW_MINION_LWLINK_DL_STAT_ARGS_LANEID  15:12
#define LW_MINION_LWLINK_DL_STAT_ARGS_ADDRS   11:0

//
// Internal function declarations
//
LwlStatus lwswitch_minion_get_dl_status_ls10(lwswitch_device *device, LwU32 linkId, LwU32 statusIdx, LwU32 statusArgs, LwU32 *statusData);
LwlStatus lwswitch_set_minion_initialized_ls10(lwswitch_device *device, LwU32 idx_minion, LwBool initialized);
LwBool    lwswitch_is_minion_initialized_ls10(lwswitch_device *device, LwU32 idx_minion);
LwlStatus lwswitch_init_minion_ls10(lwswitch_device *device);
LwlStatus lwswitch_minion_send_command_ls10(lwswitch_device *device, LwU32 linkNumber, LwU32 command, LwU32 scratch0);
LwlStatus lwswitch_minion_riscv_get_physical_address_ls10(lwswitch_device *device,LwU32 idx_minion, LwU32 target, LwLength offset, LwU64 *pRiscvPa);
LwlStatus lwswitch_minion_set_sim_mode_ls10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_minion_set_smf_settings_ls10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_minion_select_uphy_tables_ls10(lwswitch_device *device, lwlink_link *link);
LwBool    lwswitch_minion_is_riscv_active_ls10(lwswitch_device *device, LwU32 idx_minion);
LwlStatus lwswitch_minion_clear_dl_error_counters_ls10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_minion_send_inband_data_ls10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_minion_receive_inband_data_ls10(lwswitch_device *device, LwU32 linkId);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus lwswitch_ctrl_config_eom_ls10(lwswitch_device *device, LWSWITCH_CTRL_CONFIG_EOM *p);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#endif //_MINION_LS10_H_
