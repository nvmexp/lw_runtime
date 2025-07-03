/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _HALDEF_LWSWITCH_H_
#define _HALDEF_LWSWITCH_H_

#include "ctrl_dev_internal_lwswitch.h"
#include "ctrl_dev_lwswitch.h"

#include "inforom/ifrstruct.h"
#include "inforom/omsdef.h"

//
// List of functions halified in the LWSwitch Driver
//
// Note: All hal fns must be implemented for each chip.
//       There is no automatic stubbing here.
//
// This 'xmacro' list is fed into generator macros which then use the
// _FUNCTION_LIST to generate HAL declarations, function prototypes, and HAL
// construction.  Xmacros are a useful way to maintain consistency between
// parallel lists.
// The components of the _FUNCTION_LIST are similar to a function prototype
// declaration, with the addition of an '_arch' parameter suffixed on to it
// which is used on some _FUNCTION_LIST expansions to generate arch-specific
// information.
// The format of each line is:
//     _op(return type, function name, (parameter list), _arch)
//

#define LWSWITCH_HAL_FUNCTION_LIST(_op, _arch)                                          \
    _op(LwlStatus, lwswitch_initialize_device_state, (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_destroy_device_state,    (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_determine_platform,      (lwswitch_device *device), _arch)  \
    _op(LwU32,     lwswitch_get_num_links,           (lwswitch_device *device), _arch)  \
    _op(LwBool,    lwswitch_is_link_valid,           (lwswitch_device *device, LwU32 link_id), _arch)  \
    _op(void,      lwswitch_set_fatal_error,         (lwswitch_device *device, LwBool device_fatal, LwU32 link_id), _arch)  \
    _op(LwU32,     lwswitch_get_swap_clk_default,    (lwswitch_device *device), _arch)  \
    _op(LwU32,     lwswitch_get_latency_sample_interval_msec, (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_internal_latency_bin_log,(lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_ecc_writeback_task,      (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_monitor_thermal_alert,   (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_hw_counter_shutdown,     (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_get_rom_info,            (lwswitch_device *device, LWSWITCH_EEPROM_TYPE *eeprom), _arch)  \
    _op(void,      lwswitch_lib_enable_interrupts,   (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_lib_disable_interrupts,  (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_lib_check_interrupts,    (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_lib_service_interrupts,  (lwswitch_device *device), _arch)  \
    _op(LwU64,     lwswitch_hw_counter_read_counter, (lwswitch_device *device), _arch)  \
    _op(LwBool,    lwswitch_is_link_in_use,          (lwswitch_device *device, LwU32 link_id), _arch)  \
    _op(LwlStatus, lwswitch_reset_and_drain_links,   (lwswitch_device *device, LwU64 link_mask), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_info,           (lwswitch_device *device, LWSWITCH_GET_INFO *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_lwlink_status,  (lwswitch_device *device, LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_counters,       (lwswitch_device *device, LWSWITCH_LWLINK_GET_COUNTERS_PARAMS *ret), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_switch_port_config,    (lwswitch_device *device, LWSWITCH_SET_SWITCH_PORT_CONFIG *p), _arch)  \
    _op(LwlStatus, lwswitch_set_nport_port_config,   (lwswitch_device *device, LWSWITCH_SET_SWITCH_PORT_CONFIG *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_ingress_request_table, (lwswitch_device *device, LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *params), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_ingress_request_table, (lwswitch_device *device, LWSWITCH_SET_INGRESS_REQUEST_TABLE *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_ingress_request_valid, (lwswitch_device *device, LWSWITCH_SET_INGRESS_REQUEST_VALID *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_ingress_response_table, (lwswitch_device *device, LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *params), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_ingress_response_table, (lwswitch_device *device, LWSWITCH_SET_INGRESS_RESPONSE_TABLE *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_ganged_link_table,      (lwswitch_device *device, LWSWITCH_SET_GANGED_LINK_TABLE *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_remap_policy,  (lwswitch_device *device, LWSWITCH_SET_REMAP_POLICY *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_remap_policy,  (lwswitch_device *device, LWSWITCH_GET_REMAP_POLICY_PARAMS *params), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_remap_policy_valid, (lwswitch_device *device, LWSWITCH_SET_REMAP_POLICY_VALID *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_routing_id,    (lwswitch_device *device, LWSWITCH_SET_ROUTING_ID *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_routing_id,    (lwswitch_device *device, LWSWITCH_GET_ROUTING_ID_PARAMS *params), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_routing_id_valid, (lwswitch_device *device, LWSWITCH_SET_ROUTING_ID_VALID *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_routing_lan,   (lwswitch_device *device, LWSWITCH_SET_ROUTING_LAN *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_routing_lan,   (lwswitch_device *device, LWSWITCH_GET_ROUTING_LAN_PARAMS *params), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_routing_lan_valid, (lwswitch_device *device, LWSWITCH_SET_ROUTING_LAN_VALID *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_internal_latency, (lwswitch_device *device, LWSWITCH_GET_INTERNAL_LATENCY *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_latency_bins,  (lwswitch_device *device, LWSWITCH_SET_LATENCY_BINS *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_ingress_reqlinkid, (lwswitch_device *device, LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *params), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_register_read,     (lwswitch_device *device, LWSWITCH_REGISTER_READ *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_register_write,    (lwswitch_device *device, LWSWITCH_REGISTER_WRITE *p), _arch)  \
    _op(LwlStatus, lwswitch_pex_get_counter,        (lwswitch_device *device, LwU32 counterType, LwU32 *pCount), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_pex_get_lane_counters, (lwswitch_device *device, LWSWITCH_PEX_GET_LANE_COUNTERS_PARAMS *pParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_i2c_get_port_info, (lwswitch_device *device, LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS *pParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_i2c_indexed,       (lwswitch_device *device, LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_therm_read_temperature, (lwswitch_device *device, LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_therm_get_temperature_limit, (lwswitch_device *device, LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS *info), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_throughput_counters, (lwswitch_device *device, LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_corelib_add_link,       (lwlink_link *link), _arch)  \
    _op(LwlStatus, lwswitch_corelib_remove_link,    (lwlink_link *link), _arch)  \
    _op(LwlStatus, lwswitch_corelib_set_dl_link_mode, (lwlink_link *link, LwU64 mode, LwU32 flags), _arch)  \
    _op(LwlStatus, lwswitch_corelib_get_dl_link_mode, (lwlink_link *link, LwU64 *mode), _arch)  \
    _op(LwlStatus, lwswitch_corelib_set_tl_link_mode, (lwlink_link *link, LwU64 mode, LwU32 flags), _arch)  \
    _op(LwlStatus, lwswitch_corelib_get_tl_link_mode, (lwlink_link *link, LwU64 *mode), _arch)  \
    _op(LwlStatus, lwswitch_corelib_set_tx_mode,    (lwlink_link *link, LwU64 mode, LwU32 flags), _arch)  \
    _op(LwlStatus, lwswitch_corelib_get_tx_mode,    (lwlink_link *link, LwU64 *mode, LwU32 *subMode), _arch)  \
    _op(LwlStatus, lwswitch_corelib_set_rx_mode,    (lwlink_link *link, LwU64 mode, LwU32 flags), _arch)  \
    _op(LwlStatus, lwswitch_corelib_get_rx_mode,    (lwlink_link *link, LwU64 *mode, LwU32 *subMode), _arch)  \
    _op(LwlStatus, lwswitch_corelib_set_rx_detect,  (lwlink_link *link, LwU32 flags), _arch)  \
    _op(LwlStatus, lwswitch_corelib_get_rx_detect,  (lwlink_link *link), _arch)  \
    _op(LwlStatus, lwswitch_corelib_write_discovery_token, (lwlink_link *link, LwU64 token), _arch)  \
    _op(LwlStatus, lwswitch_corelib_read_discovery_token,  (lwlink_link *link, LwU64 *token), _arch)  \
    _op(void,      lwswitch_corelib_training_complete, (lwlink_link *link), _arch)  \
    _op(LwU32,     lwswitch_get_device_dma_width,   (lwswitch_device *device), _arch)  \
    _op(LwU32,     lwswitch_get_link_ip_version,    (lwswitch_device *device, LwU32 link_id), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_fom_values,    (lwswitch_device *device, LWSWITCH_GET_FOM_VALUES_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_deassert_link_reset,    (lwswitch_device *device, lwlink_link *link), _arch)  \
    _op(LwBool,    lwswitch_is_soe_supported,       (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_soe_set_ucode_core,     (lwswitch_device *device, LwBool bFalcon), _arch)  \
    _op(LwlStatus, lwswitch_init_soe, (lwswitch_device *device), _arch)  \
    _op(LwBool,    lwswitch_is_inforom_supported,   (lwswitch_device *device), _arch)  \
    _op(LwBool,    lwswitch_is_spi_supported,       (lwswitch_device *device), _arch)  \
    _op(LwBool,    lwswitch_is_smbpbi_supported,   (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_soe_prepare_for_reset, (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_post_init_device_setup, (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_post_init_blacklist_device_setup, (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_setup_link_system_registers, (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_get_lwlink_ecc_errors,  (lwswitch_device *device, LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS *p), _arch)  \
    _op(void,      lwswitch_save_lwlink_seed_data_from_minion_to_inforom, (lwswitch_device *device, LwU32 linkId), _arch)  \
    _op(void,      lwswitch_store_seed_data_from_inforom_to_corelib, (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_inforom_ecc_log_error_event, (lwswitch_device *device, INFOROM_ECC_OBJECT *pEccGeneric, INFOROM_LWS_ECC_ERROR_EVENT *error_event), _arch)  \
    _op(void,      lwswitch_oms_set_device_disable, (INFOROM_OMS_STATE *pOmsState, LwBool bForceDeviceDisable), _arch)  \
    _op(LwBool,    lwswitch_oms_get_device_disable, (INFOROM_OMS_STATE *pOmsState), _arch)  \
    _op(LwlStatus, lwswitch_inforom_lwl_get_minion_data, (lwswitch_device *device, void *pLwlGeneric, LwU8 linkId, LwU32 *seedData), _arch)  \
    _op(LwlStatus, lwswitch_inforom_lwl_set_minion_data, (lwswitch_device *device, void *pLwlGeneric, LwU8 linkId, LwU32 *seedData, LwU32 size, LwBool *bDirty), _arch)  \
    _op(LwlStatus, lwswitch_inforom_lwl_log_error_event, (lwswitch_device *device, void *pLwlGeneric, void *error_event, LwBool *bDirty), _arch)  \
    _op(LwlStatus, lwswitch_inforom_lwl_update_link_correctable_error_info, (lwswitch_device *device, void *pLwlGeneric, void *pData, LwU8 linkId, LwU8 lwliptInstance, LwU8 localLinkIdx, void *pErrorCounts, LwBool *bDirty), _arch)  \
    _op(LwlStatus, lwswitch_inforom_lwl_get_max_correctable_error_rate,  (lwswitch_device *device, LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_inforom_lwl_get_errors,  (lwswitch_device *device, LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_inforom_ecc_get_errors,  (lwswitch_device *device, LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *p), _arch)  \
    _op(void,      lwswitch_load_uuid,              (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_i2c_set_hw_speed_mode,  (lwswitch_device *device, LwU32 port, LwU32 speedMode), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_bios_info,     (lwswitch_device *device, LWSWITCH_GET_BIOS_INFO_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_read_oob_blacklist_state, (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_write_fabric_state,     (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_initialize_oms_state,   (lwswitch_device *device, INFOROM_OMS_STATE *pOmsState), _arch)  \
    _op(LwlStatus, lwswitch_oms_inforom_flush,      (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_inforom_ecc_get_total_errors,   (lwswitch_device *device, INFOROM_ECC_OBJECT *pEccGeneric, LwU64 *corCount, LwU64 *uncCount), _arch)  \
    _op(LwlStatus, lwswitch_bbx_setup_prologue,     (lwswitch_device *device, void *pInforomBbxState), _arch)  \
    _op(LwlStatus, lwswitch_bbx_setup_epilogue,     (lwswitch_device *device, void *pInforomBbxState), _arch)  \
    _op(LwlStatus, lwswitch_bbx_add_data_time,      (lwswitch_device *device, void *pInforomBbxState, void *pInforomBbxData), _arch)  \
    _op(LwlStatus, lwswitch_bbx_add_sxid,   (lwswitch_device *device, void *pInforomBbxState, void *pInforomBbxData), _arch)  \
    _op(LwlStatus, lwswitch_bbx_add_temperature,    (lwswitch_device *device, void *pInforomBbxState, void *pInforomBbxData), _arch)  \
    _op(void,      lwswitch_bbx_set_initial_temperature,    (lwswitch_device *device, void *pInforomBbxState, void *pInforomBbxData), _arch)  \
    _op(LwlStatus, lwswitch_inforom_bbx_get_sxid,  (lwswitch_device *device, LWSWITCH_GET_SXIDS_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_smbpbi_get_dem_num_messages,    (lwswitch_device *device, LwU8 *pMsgCount), _arch)  \
    _op(LwlStatus, lwswitch_set_minion_initialized, (lwswitch_device *device, LwU32 idx_minion, LwBool initialized), _arch)  \
    _op(LwBool,    lwswitch_is_minion_initialized,  (lwswitch_device *device, LwU32 idx_minion), _arch)  \
    _op(LwlStatus, lwswitch_get_link_public_id, (lwswitch_device *device, LwU32 linkId, LwU32 *publicId), _arch)  \
    _op(LwlStatus, lwswitch_get_link_local_idx, (lwswitch_device *device, LwU32 linkId, LwU32 *localLinkIdx), _arch)  \
    _op(LwlStatus, lwswitch_set_training_error_info, (lwswitch_device *device, LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *pLinkTrainingErrorInfoParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_fatal_error_scope, (lwswitch_device *device, LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *pParams), _arch)  \
    _op(void,      lwswitch_init_scratch,       (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_device_discovery,   (lwswitch_device *device, LwU32 discovery_offset), _arch)  \
    _op(void,      lwswitch_filter_discovery,   (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_process_discovery,  (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_init_minion,        (lwswitch_device *device), _arch)  \
    _op(LwU32,     lwswitch_get_eng_base,   (lwswitch_device *device, LWSWITCH_ENGINE_ID eng_id, LwU32 eng_bcast, LwU32 eng_instance), _arch)  \
    _op(LwU32,     lwswitch_get_eng_count,  (lwswitch_device *device, LWSWITCH_ENGINE_ID eng_id, LwU32 eng_bcast), _arch)  \
    _op(LwU32,     lwswitch_eng_rd,         (lwswitch_device *device, LWSWITCH_ENGINE_ID eng_id, LwU32 eng_bcast, LwU32 eng_instance, LwU32 offset), _arch)  \
    _op(void,      lwswitch_eng_wr,         (lwswitch_device *device, LWSWITCH_ENGINE_ID eng_id, LwU32 eng_bcast, LwU32 eng_instance, LwU32 offset, LwU32 data), _arch)  \
    _op(LwU32,     lwswitch_get_link_eng_inst,  (lwswitch_device *device, LwU32 link_id, LWSWITCH_ENGINE_ID eng_id), _arch)  \
    _op(void *,    lwswitch_alloc_chipdevice,   (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_init_thermal,       (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_init_pll_config,    (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_init_pll,           (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_init_clock_gating,  (lwswitch_device *device), _arch)  \
    _op(LwU32,     lwswitch_read_physical_id,   (lwswitch_device *device), _arch)  \
    _op(LwU32,     lwswitch_get_caps_lwlink_version,    (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_initialize_interrupt_tree,  (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_init_dlpl_interrupts,       (lwlink_link *link), _arch)  \
    _op(LwlStatus, lwswitch_initialize_pmgr,    (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_initialize_ip_wrappers,     (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_initialize_route,   (lwswitch_device *device), _arch)  \
    _op(void,      lwswitch_soe_unregister_events,      (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_soe_register_event_callbacks,  (lwswitch_device *device), _arch)  \
    _op(LWSWITCH_BIOS_LWLINK_CONFIG *, lwswitch_get_bios_lwlink_config, (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_minion_send_command, (lwswitch_device *device, LwU32 linkNumber, LwU32 command, LwU32 scratch0), _arch)  \
    _op(LwlStatus, lwswitch_init_nport,      (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_init_nxbar,      (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_clear_nport_rams,       (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_pri_ring_init,          (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_get_soe_ucode_binaries, (lwswitch_device *device, const LwU32 **soe_ucode_data, const LwU32 **soe_ucode_header), _arch)  \
    _op(LwlStatus, lwswitch_get_remap_table_selector,   (lwswitch_device *device, LWSWITCH_TABLE_SELECT_REMAP table_selector, LwU32 *remap_ram_sel), _arch)  \
    _op(LwU32,     lwswitch_get_ingress_ram_size,   (lwswitch_device *device, LwU32 ingress_ram_selector), _arch)  \
    _op(LwlStatus, lwswitch_minion_get_dl_status,   (lwswitch_device *device, LwU32 linkId, LwU32 statusIdx, LwU32 statusArgs, LwU32 *statusData), _arch)  \
    _op(void,      lwswitch_corelib_get_uphy_load, (lwlink_link *link, LwBool *bUnlocked), _arch) \
    _op(LwBool,    lwswitch_is_i2c_supported, (lwswitch_device *device), _arch) \
    _op(LwlStatus, lwswitch_poll_sublink_state, (lwswitch_device *device, lwlink_link *link), _arch)\
    _op(void,      lwswitch_setup_link_loopback_mode, (lwswitch_device *device, LwU32 linkNumber), _arch)\
    _op(void,      lwswitch_reset_persistent_link_hw_state, (lwswitch_device *device, LwU32 linkNumber), _arch)\
    _op(void,      lwswitch_store_topology_information, (lwswitch_device *device, lwlink_link *link), _arch) \
    _op(void,      lwswitch_init_lpwr_regs, (lwlink_link *link), _arch) \
    _op(LwlStatus, lwswitch_set_training_mode, (lwswitch_device *device), _arch) \
    _op(LwU32,     lwswitch_get_sublink_width, (lwswitch_device *device, LwU32 linkNumber), _arch) \
    _op(LwBool,    lwswitch_i2c_is_device_access_allowed, (lwswitch_device *device, LwU32 port, LwU8 addr, LwBool bIsRead), _arch) \
    _op(LwlStatus, lwswitch_parse_bios_image, (lwswitch_device *device), _arch) \
    _op(LwBool,    lwswitch_is_link_in_reset, (lwswitch_device *device, lwlink_link *link), _arch) \
    _op(void,      lwswitch_init_buffer_ready, (lwswitch_device *device, lwlink_link * link, LwBool bNportBufferReady), _arch) \
    _op(LwlStatus, lwswitch_ctrl_get_lwlink_lp_counters, (lwswitch_device *device, LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *p), _arch) \
    _op(LwlStatus, lwswitch_ctrl_set_residency_bins, (lwswitch_device *device, LWSWITCH_SET_RESIDENCY_BINS *p), _arch) \
    _op(LwlStatus, lwswitch_ctrl_get_residency_bins, (lwswitch_device *device, LWSWITCH_GET_RESIDENCY_BINS *p), _arch) \
    _op(void,      lwswitch_apply_recal_settings, (lwswitch_device *device, lwlink_link *), _arch) \
    _op(LwlStatus, lwswitch_service_lwldl_fatal_link, (lwswitch_device *device, LwU32 lwliptInstance, LwU32 link), _arch) \
    _op(LwlStatus, lwswitch_ctrl_get_rb_stall_busy, (lwswitch_device *device, LWSWITCH_GET_RB_STALL_BUSY *p), _arch) \
    _op(LwlStatus, lwswitch_service_minion_link, (lwswitch_device *device, LwU32 link_id), _arch) \
    _op(LwlStatus, lwswitch_ctrl_inband_send_data, (lwswitch_device *device, LWSWITCH_INBAND_SEND_DATA_PARAMS *p), _arch) \
    _op(LwlStatus, lwswitch_ctrl_inband_read_data, (lwswitch_device *device, LWSWITCH_INBAND_READ_DATA_PARAMS *p), _arch) \
    _op(LwlStatus, lwswitch_ctrl_inband_flush_data, (lwswitch_device *device, LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p), _arch) \
    _op(LwlStatus, lwswitch_ctrl_inband_pending_data_stats, (lwswitch_device *device, LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p), _arch) \
    _op(LwU32,     lwswitch_read_iddq_dvdd, (lwswitch_device *device), _arch) \

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWSWITCH_HAL_FUNCTION_LIST_UNPUBLISHED(_op, _arch) \
    _op(LwlStatus, lwswitch_get_board_id, (lwswitch_device *device, LwU16 *boardId), _arch) \
    _op(LwlStatus, lwswitch_is_link_in_repeater_mode,   (lwswitch_device *device, LwU32 link_id, LwBool *isRepeaterMode), _arch)  \
    _op(void,      lwswitch_fetch_active_repeater_mask,(lwswitch_device *device), _arch)  \
    _op(LwU64,     lwswitch_get_active_repeater_mask,  (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_corelib_set_optical_infinite_mode, (lwlink_link *link, LwBool bEnable), _arch) \
    _op(LwlStatus, lwswitch_corelib_enable_optical_maintenance, (lwlink_link *link, LwBool bTx), _arch)  \
    _op(LwlStatus, lwswitch_corelib_set_optical_iobist, (lwlink_link *link, LwBool bEnable), _arch) \
    _op(LwlStatus, lwswitch_corelib_set_optical_pretrain, (lwlink_link *link, LwBool bTx, LwBool bEnable), _arch) \
    _op(LwlStatus, lwswitch_corelib_check_optical_pretrain, (lwlink_link *link, LwBool bTx, LwBool* bSuccess), _arch) \
    _op(LwlStatus, lwswitch_corelib_init_optical_links, (lwlink_link *link), _arch) \
    _op(LwlStatus, lwswitch_corelib_set_optical_force_eq, (lwlink_link *link, LwBool bEnable), _arch) \
    _op(LwlStatus, lwswitch_corelib_check_optical_eom_status, (lwlink_link *link, LwBool* bEomLow), _arch) \
    _op(LwBool,    lwswitch_is_cci_supported,       (lwswitch_device *device), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_port_test_mode, (lwswitch_device *device, LWSWITCH_SET_PORT_TEST_MODE *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_jtag_chain_read,   (lwswitch_device *device, LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_jtag_chain_write,  (lwswitch_device *device, LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_pex_clear_counters, (lwswitch_device *device, LWSWITCH_PEX_CLEAR_COUNTERS_PARAMS *pParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_i2c_get_dev_info,  (lwswitch_device *device, LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS *pParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_therm_read_voltage, (lwswitch_device *device, LWSWITCH_CTRL_GET_VOLTAGE_PARAMS *info), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_config_eom,        (lwswitch_device *device, LWSWITCH_CTRL_CONFIG_EOM *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_inject_link_error, (lwswitch_device *device, LWSWITCH_INJECT_LINK_ERROR *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_lwlink_caps,    (lwswitch_device *device, LWSWITCH_GET_LWLINK_CAPS_PARAMS *ret), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_clear_counters,     (lwswitch_device *device, LWSWITCH_LWLINK_CLEAR_COUNTERS_PARAMS *ret), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_err_info,       (lwswitch_device *device, LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS *ret), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_irq_info,      (lwswitch_device *device, LWSWITCH_GET_IRQ_INFO_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_read_uphy_pad_lane_reg, (lwswitch_device *device, LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_pex_set_eom,       (lwswitch_device *device, LWSWITCH_PEX_CTRL_EOM *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_pex_get_eom_status,(lwswitch_device *device, LWSWITCH_PEX_GET_EOM_STATUS_PARAMS *pParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_uphy_dln_cfg_space,    (lwswitch_device *device, LWSWITCH_GET_PEX_UPHY_DLN_CFG_SPACE_PARAMS *params), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_force_thermal_slowdown,    (lwswitch_device *device, LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_pcie_link_speed, (lwswitch_device *device, LWSWITCH_SET_PCIE_LINK_SPEED_PARAMS *pParams), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_set_mc_rid_table,  (lwswitch_device *device, LWSWITCH_SET_MC_RID_TABLE_PARAMS *p), _arch)  \
    _op(LwlStatus, lwswitch_ctrl_get_mc_rid_table,  (lwswitch_device *device, LWSWITCH_GET_MC_RID_TABLE_PARAMS *p), _arch)  \

#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#define LWSWITCH_HAL_FUNCTION_LIST_LWCFG_LS10(_op, _arch) \
   _op(LwlStatus, lwswitch_launch_ALI, (lwswitch_device *device), _arch) \
    
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
//
// Declare HAL function pointer table
//
// This macro takes the xmacro _FUNCTION_LIST and uses some components in it to
// automatically generate the HAL structure declaration in a form:
//     LwU32    (*function_foo1)(lwswitch_device device);
//     void     (*function_foo2)(lwswitch_device device, LwU32 parameter1);
//     LwlStatus (*function_foo3)(lwswitch_device device, LwU32 parameter1, void *parameter2);
//

#define DECLARE_HAL_FUNCTIONS(_return, _function, _params, _arch)   \
    _return (*_function)_params;

typedef struct lwswitch_hal_functions
{
    LWSWITCH_HAL_FUNCTION_LIST(DECLARE_HAL_FUNCTIONS, HAL)
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LWSWITCH_HAL_FUNCTION_LIST_UNPUBLISHED(DECLARE_HAL_FUNCTIONS, HAL)
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_HAL_FUNCTION_LIST_LWCFG_LS10(DECLARE_HAL_FUNCTIONS, HAL)
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

} lwswitch_hal;

//
// Fill in HAL function pointer table
//
// This macro takes the xmacro _FUNCTION_LIST and uses some components in it to
// automatically generate all the HAL function fill-in assignments for a given
// architecture.
//

#define CREATE_HAL_FUNCTIONS(_return, _function, _params, _arch)    \
    device->hal._function = _function##_##_arch;                    \

#define LWSWITCH_INIT_HAL(device, arch)                             \
    LWSWITCH_HAL_FUNCTION_LIST(CREATE_HAL_FUNCTIONS, arch)          \

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWSWITCH_INIT_HAL_UNPUBLISHED(device, arch)                         \
    LWSWITCH_HAL_FUNCTION_LIST_UNPUBLISHED(CREATE_HAL_FUNCTIONS, arch)      \

#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#define LWSWITCH_INIT_HAL_LWCFG_LS10(device, arch)                         \
    LWSWITCH_HAL_FUNCTION_LIST_LWCFG_LS10(CREATE_HAL_FUNCTIONS, arch)      \

#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

//
// Declare HAL function dispatch functions
//
// This macro takes the xmacro _FUNCTION_LIST and uses some components in it to
// automatically generate the function prototypes for the dispatcher functions
// that dereference the correct arch HAL function.
//

#define DECLARE_HAL_DISPATCHERS(_return, _function, _params, _arch) \
    _return _function _params;

LWSWITCH_HAL_FUNCTION_LIST(DECLARE_HAL_DISPATCHERS, unused_argument)
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LWSWITCH_HAL_FUNCTION_LIST_UNPUBLISHED(DECLARE_HAL_DISPATCHERS, unused_argument)
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LWSWITCH_HAL_FUNCTION_LIST_LWCFG_LS10(DECLARE_HAL_DISPATCHERS, unused_argument)
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

// HAL functions
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
void lwswitch_setup_hal_sv10(lwswitch_device *device);
#endif
void lwswitch_setup_hal_lr10(lwswitch_device *device);
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
void lwswitch_setup_hal_ls10(lwswitch_device *device);
#endif

#endif //_HALDEF_LWSWITCH_H_
