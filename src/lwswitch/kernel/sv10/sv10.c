/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwlink_export.h"
#include "common_lwswitch.h"
#include "error_lwswitch.h"
#include "regkey_lwswitch.h"
#include "haldef_lwswitch.h"
#include "lwSha1.h"
#include "sv10/sv10.h"
#include "sv10/clock_sv10.h"
#include "sv10/bus_sv10.h"
#include "sv10/minion_sv10.h"
#include "sv10/fuse_sv10.h"
#include "sv10/pmgr_sv10.h"
#include "sv10/therm_sv10.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "sv10/jtag_sv10.h"
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "sv10/inforom_sv10.h"
#include "sv10/smbpbi_sv10.h"

#include "lwswitch/svnp01/dev_pri_ringmaster.h"
#include "lwswitch/svnp01/dev_pri_ringstation_sys.h"
#include "lwswitch/svnp01/dev_lws_master.h"
#include "lwswitch/svnp01/dev_lws_master_addendum.h"
#include "lwswitch/svnp01/dev_npg_ip.h"
#include "lwswitch/svnp01/dev_npgperf_ip.h"
#include "lwswitch/svnp01/dev_nport_ip.h"
#include "lwswitch/svnp01/dev_nport_ip_addendum.h"
#include "lwswitch/svnp01/dev_lwlipt_ip.h"
#include "lwswitch/svnp01/dev_lwltlc_ip.h"
#include "lwswitch/svnp01/dev_lwltlc_ip_addendum.h"
#include "lwswitch/svnp01/dev_lwl_ip.h"
#include "lwswitch/svnp01/dev_lwlctrl_ip.h"
#include "lwswitch/svnp01/dev_minion_ip.h"
#include "lwswitch/svnp01/dev_swx_ip.h"
#include "lwswitch/svnp01/dev_afs_ip.h"
#include "lwswitch/svnp01/dev_route_ip.h"
#include "lwswitch/svnp01/dev_route_ip_addendum.h"
#include "lwswitch/svnp01/dev_ingress_ip.h"
#include "lwswitch/svnp01/dev_ingress_ip_addendum.h"
#include "lwswitch/svnp01/dev_egress_ip.h"
#include "lwswitch/svnp01/dev_lwlsaw_ip.h"
#include "lwswitch/svnp01/dev_lws.h"
#include "lwswitch/svnp01/dev_ftstate_ip.h"
#include "lwswitch/svnp01/dev_timer.h"
#include "lwswitch/svnp01/dev_pmgr.h"
#include "lwswitch/svnp01/dev_fuse.h"
#include "lwswitch/svnp01/dev_lwlsaw_ip_addendum.h"

#define INGRESS_MAP_TABLE_SIZE (1 << DRF_SIZE(LW_INGRESS_REQRSPMAPADDR_TABLE_INDEX))

#define ROUTE_GANG_TABLE_SIZE (1 << DRF_SIZE(LW_ROUTE_REG_TABLE_ADDRESS_INDEX))

#define LW_PTOP_DISCOVERY_TABLE(i) (0x0002c000+(i)*4)

// Forward declarations
static LwU32
lwswitch_get_num_links_sv10
(
    lwswitch_device *device
)
{
    return LWSWITCH_NUM_LINKS_SV10;
}

static LwU32
_lwswitch_get_num_vcs_sv10
(
    lwswitch_device *device
)
{
    return LWSWITCH_NUM_VCS_SV10;
}

static void
lwswitch_destroy_link_info_sv10
(
    lwswitch_device *device
)
{
    LwU32   idx_link;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    for (idx_link=0; idx_link < LWSWITCH_NUM_LINKS_SV10; idx_link++)
    {
        if (NULL != chip_device->link[idx_link].ingress_req_table)
        {
            lwswitch_os_free(chip_device->link[idx_link].ingress_req_table);
            chip_device->link[idx_link].ingress_req_table = NULL;
        }

        if (NULL != chip_device->link[idx_link].ingress_res_table)
        {
            lwswitch_os_free(chip_device->link[idx_link].ingress_res_table);
            chip_device->link[idx_link].ingress_res_table = NULL;
        }

        if (NULL != chip_device->link[idx_link].ganged_link_table)
        {
            lwswitch_os_free(chip_device->link[idx_link].ganged_link_table);
            chip_device->link[idx_link].ganged_link_table = NULL;
        }
    }
}

static void
_lwswitch_init_link_info
(
    lwswitch_device *device,
    LwU32 link_id
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    lwswitch_os_memset(chip_device->link[link_id].ingress_req_table, 0,
                  sizeof(INGRESS_REQUEST_RESPONSE_ENTRY_SV10) * INGRESS_MAP_TABLE_SIZE);
    lwswitch_os_memset(chip_device->link[link_id].ingress_res_table, 0,
                  sizeof(INGRESS_REQUEST_RESPONSE_ENTRY_SV10) * INGRESS_MAP_TABLE_SIZE);
    lwswitch_os_memset(chip_device->link[link_id].ganged_link_table, 0,
                  sizeof(ROUTE_GANG_ENTRY_SV10) * ROUTE_GANG_TABLE_SIZE);
}

static LwlStatus
_lwswitch_construct_link_info
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwlStatus retval = LWL_SUCCESS;
    LwU32   idx_link;

    for (idx_link=0; idx_link < LWSWITCH_NUM_LINKS_SV10; idx_link++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, idx_link))
        {
            continue;
        }

        chip_device->link[idx_link].ingress_req_table =
            lwswitch_os_malloc(sizeof(INGRESS_REQUEST_RESPONSE_ENTRY_SV10) * INGRESS_MAP_TABLE_SIZE);

        chip_device->link[idx_link].ingress_res_table =
            lwswitch_os_malloc(sizeof(INGRESS_REQUEST_RESPONSE_ENTRY_SV10) * INGRESS_MAP_TABLE_SIZE);

        chip_device->link[idx_link].ganged_link_table =
            lwswitch_os_malloc(sizeof(ROUTE_GANG_ENTRY_SV10) * ROUTE_GANG_TABLE_SIZE);

        if ((NULL == chip_device->link[idx_link].ingress_req_table) ||
            (NULL == chip_device->link[idx_link].ingress_res_table) ||
            (NULL == chip_device->link[idx_link].ganged_link_table))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to allocate req/res backing stores\n",
                __FUNCTION__);
            retval = -LWL_NO_MEM;
            break;
        }

        _lwswitch_init_link_info(device, idx_link);
    }

    if (retval != LWL_SUCCESS)
    {
        lwswitch_destroy_link_info_sv10(device);
    }

    return retval;
}

static void
lwswitch_determine_platform_sv10
(
    lwswitch_device *device
)
{
    LwU32 value;

    //
    // Determine which model we are using SMC_BOOT_2 and OS query
    //
    value = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_2);
    device->is_emulation = FLD_TEST_DRF(_PSMC, _BOOT_2, _EMULATION, _YES, value);

    if (!IS_EMULATION(device))
    {
        // If we are not on fmodel, we must be on RTL sim or silicon.
        if (FLD_TEST_DRF(_PSMC, _BOOT_2, _FMODEL, _YES, value))
        {
            device->is_fmodel = LW_TRUE;
        }
        else
        {
            device->is_rtlsim = LW_TRUE;

            // Let OS code finalize RTL sim vs silicon
            lwswitch_os_override_platform(device->os_handle, &device->is_rtlsim);
        }
    }

#if defined(LWLINK_PRINT_ENABLED)
    {
        const char *build;
        const char *mode;

        build = "HW";
        if (IS_FMODEL(device))
            mode = "fmodel";
        else if (IS_RTLSIM(device))
            mode = "rtlsim";
        else if (IS_EMULATION(device))
            mode = "emulation";
        else
            mode = "silicon";

        LWSWITCH_PRINT(device, SETUP,
            "%s: build: %s platform: %s\n",
             __FUNCTION__, build, mode);
    }
#endif // LWLINK_PRINT_ENABLED
}

static void
lwswitch_set_ingress_table_sv10
(
    lwswitch_device *device,
    LwU32            portNum,
    LwU32            firstIndex,
    LwU32            numEntries,
    LwBool           tableSelect
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 i;
    INGRESS_REQUEST_RESPONSE_ENTRY_SV10 *table;

    table = &chip_device->link[portNum].ingress_req_table[firstIndex];

    if (tableSelect == LW_INGRESS_REQRSPMAPADDR_TABLE_SELECT_RESPONSE)
    {
        table = &chip_device->link[portNum].ingress_res_table[firstIndex];
    }

    LWSWITCH_LINK_WR32_SV10(device, portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_INDEX, firstIndex)   |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_SELECT, tableSelect) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    for (i = 0; i < numEntries; i++)
    {
        LWSWITCH_LINK_WR32_SV10(device, portNum, NPORT, _INGRESS, _REQRSPMAPDATA1,
            table[i].ingress_reqresmapdata1);
        LWSWITCH_LINK_WR32_SV10(device, portNum, NPORT, _INGRESS, _REQRSPMAPDATA2,
            table[i].ingress_reqresmapdata2);
        LWSWITCH_LINK_WR32_SV10(device, portNum, NPORT, _INGRESS, _REQRSPMAPDATA3,
            table[i].ingress_reqresmapdata3);

        // Write last and auto-increment
        LWSWITCH_LINK_WR32_SV10(device, portNum, NPORT, _INGRESS, _REQRSPMAPDATA0,
            table[i].ingress_reqresmapdata0);
    }
}

static void
_lwswitch_ingress_ecc_writeback
(
    lwswitch_device *device,
    LwU32 port
)
{
    LwU32 val;
    LWSWITCH_RAW_ERROR_LOG_TYPE error_info = {{ 0 }};

    val = LWSWITCH_LINK_RD32_SV10(device, port, NPORT, _INGRESS, _ECC_ERROR_COUNT);

    if (FLD_TEST_DRF(_INGRESS, _ECC_ERROR_COUNT, _ERROR_COUNT, __PROD, val))
    {
        return;
    }

    // Log error count.
    error_info.data[0] = val;

    val = LWSWITCH_LINK_RD32_SV10(device, port, NPORT, _INGRESS, _ECC_ERROR_ADDRESS);

    // Log error address.
    error_info.data[1] = val;

    lwswitch_set_ingress_table_sv10(device, port,
        DRF_VAL(_INGRESS, _REQRSPMAPADDR, _TABLE_INDEX, val),
        1,
        DRF_VAL(_INGRESS, _REQRSPMAPADDR, _TABLE_SELECT, val));

    LWSWITCH_LINK_WR32_SV10(device, port, NPORT, _INGRESS, _ECC_ERROR_COUNT,
        DRF_DEF(_INGRESS, _ECC_ERROR_COUNT, _ERROR_COUNT, __PROD));

    LWSWITCH_REPORT_CORRECTABLE_LINK_DATA(device, port,
        _HW_NPORT_INGRESS_ECCSOFTLIMITERR, &error_info,
        "ingress single bit ECC");
}

//
// HW ECC WB mechanism is broken. See bugs 1835914, 1833572, 1849183 and 1973042.
// Hence, SW needs restore the data for both ingress and route RAMs by
// periodically checking if the ECC errors have oclwrred.
//
static void
lwswitch_ecc_writeback_task_sv10
(
    lwswitch_device *device
)
{
    LwU32 port;

    for (port = 0; port < LWSWITCH_NUM_LINKS_SV10; port++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, port))
        {
            continue;
        }

        _lwswitch_ingress_ecc_writeback(device, port);
    }
}

//
// Data collector which runs on a background thread, collecting latency stats
//
// Latency stats are periodically snapped based on the setting of WINDOW_LIMIT.
// Once the bin counts are snapped, they need to be read out before the next
// WINDOW_LIMIT interval.  There is no flag that a new set of counts have been
// snapped, so we need to track what the last set was and compare against the
// current counts to know if we need to add these to the aggregated counts we
// are aclwmulating.
//
static void
lwswitch_internal_latency_bin_log_sv10
(
    lwswitch_device *device
)
{
    LwU32 idx_nport;
    LwU32 idx_vc;
    LwU64 latency;
    LwU64 time_nsec;
    LwBool update_aclwm_latency = LW_FALSE;
    LwU32 window_limit;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link_type;    // Access or trunk link
    LwBool vc_valid;

    if (chip_device->latency_stats == NULL)
    {
        // Latency stat buffers not allocated yet
        return;
    }

    time_nsec = lwswitch_os_get_platform_time();

    // TODO: Compare time stamp to insure we didn't miss snap
    for (idx_nport=0; idx_nport < LWSWITCH_NUM_LINKS_SV10; idx_nport++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, idx_nport))
        {
            for (idx_vc=0; idx_vc < LWSWITCH_NUM_VCS_SV10; idx_vc++)
            {
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].low = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].medium = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].high = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].panic = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].count = 0;
            }
            continue;
        }

        link_type = LWSWITCH_LINK_RD32_SV10(device, idx_nport, NPORT, _NPORT, _CTRL);

        //
        // TODO: Check _STARTCOUNTER and don't log if counter not enabled.
        // Lwrrently all counters are always enabled
        //

        //
        // Snap the current counts, then see if they have updated compared to the previous
        // sample period.  The counters latch at _WINDOWLIMIT.
        //
        for (idx_vc=0; idx_vc < LWSWITCH_NUM_VCS_SV10; idx_vc++)
        {
            vc_valid = LW_TRUE;

#if !defined(LWCPU_PPC64LE)
            // VCs DNGRD(1), ATR(2), ATSD(3), and PROBE(4) are only relevant on Power9 fabrics
            if ((idx_vc == LW_NPORT_VC_MAPPING_DNGRD) ||
                (idx_vc == LW_NPORT_VC_MAPPING_ATR) ||
                (idx_vc == LW_NPORT_VC_MAPPING_ATSD) ||
                (idx_vc == LW_NPORT_VC_MAPPING_PROBE))
            {
                vc_valid = LW_FALSE;
            }
#endif  //!defined(LWCPU_PPC64LE)

            // VCs CREQ1(6) and RSP1(7) are only relevant when trunk & VCs are enabled
            if (FLD_TEST_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _ACCESSLINK, link_type) &&
                ((idx_vc == LW_NPORT_VC_MAPPING_CREQ1) ||
                (idx_vc == LW_NPORT_VC_MAPPING_RSP1)))
            {
                vc_valid = LW_FALSE;
            }

            // If the VC is not being used, skip reading and checking it
            if (!vc_valid)
            {
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].low = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].medium = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].high = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].panic = 0;
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].count = 0;

                continue;
            }

            latency = LWSWITCH_OFF_RD32(device, chip_device->link[idx_nport].engNPORT->uc_addr +
                LW_NPORT_PORTSTAT_SV10(_COUNT, _LOW, idx_vc));
            chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].low = latency;
            update_aclwm_latency |=
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].low !=
                chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].low;

            latency = LWSWITCH_OFF_RD32(device, chip_device->link[idx_nport].engNPORT->uc_addr +
                LW_NPORT_PORTSTAT_SV10(_COUNT, _MEDIUM, idx_vc));
            chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].medium = latency;
            update_aclwm_latency |=
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].medium !=
                chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].medium;

            latency = LWSWITCH_OFF_RD32(device, chip_device->link[idx_nport].engNPORT->uc_addr +
                LW_NPORT_PORTSTAT_SV10(_COUNT, _HIGH, idx_vc));
            chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].high = latency;
            update_aclwm_latency |=
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].high !=
                chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].high;

            latency = LWSWITCH_OFF_RD32(device, chip_device->link[idx_nport].engNPORT->uc_addr +
                LW_NPORT_PORTSTAT_SV10(_COUNT, _PANIC, idx_vc));
            chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].panic = latency;
            update_aclwm_latency |=
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].panic !=
                chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].panic;

            latency = LWSWITCH_OFF_RD32(device, chip_device->link[idx_nport].engNPORT->uc_addr +
                LW_NPORT_PORTSTAT_SV10(_PACKET, _COUNT, idx_vc));
            chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].count = latency;
            update_aclwm_latency |=
                chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport].count !=
                chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].count;
        }
    }

    //
    // Program _WINDOWLIMIT (in clock ticks) to the sampling interval
    // Snap time 0xFFFFFFFF in switch clocks (1.64GHz) is ~2.6 seconds
    //
    window_limit = device->switch_pll.freq_khz * chip_device->latency_stats->sample_interval_msec;

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _PORTSTAT_WINDOW_LIMIT,
        DRF_NUM(_NPORT, _PORTSTAT_WINDOW_LIMIT, _WINDOWLIMIT, window_limit));

    if (update_aclwm_latency)
    {
        for (idx_vc=0; idx_vc < LWSWITCH_NUM_VCS_SV10; idx_vc++)
        {
            // Note the time of this snap
            chip_device->latency_stats->latency[idx_vc].last_read_time_nsec = time_nsec;
            chip_device->latency_stats->latency[idx_vc].count++;

            for (idx_nport=0; idx_nport < LWSWITCH_NUM_LINKS_SV10; idx_nport++)
            {
                chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport] =
                    chip_device->latency_stats->latency[idx_vc].lwrr_latency[idx_nport];

                chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].low +=
                     chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].low;

                chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].medium +=
                    chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].medium;

                chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].high +=
                    chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].high;

                chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].panic +=
                     chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].panic;

                chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].count +=
                    chip_device->latency_stats->latency[idx_vc].last_latency[idx_nport].count;
            }
        }
    }
}

static void
_lwswitch_init_mc_enable_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 engine_enable_mask;
    LwU32 engine_disable_mask;
    LwU32 i, j;
    LwU32 idx_link;

    // Enable all the top level units
    LWSWITCH_REG_WR32(device, _PSMC, _ENABLE,
        DRF_DEF(_PSMC, _ENABLE, _SWX, _ENABLE) |
        DRF_DEF(_PSMC, _ENABLE, _SAW, _ENABLE) |
        DRF_DEF(_PSMC, _ENABLE, _PRIV_RING, _ENABLE) |
        DRF_DEF(_PSMC, _ENABLE, _PERFMON, _ENABLE));

    //
    // At this point the list of discovered devices has been cross-referenced
    // with the ROM configuration, platform configuration, and regkey override.
    // The LWLIPT & NPORT enable filtering done here further updates the MMIO
    // information based on KVM.
    //

    // Enable the LWLIPT units that have been discovered
    engine_enable_mask = 0;
    for (i = 0; i < NUM_SIOCTRL_ENGINE_SV10; i++)
    {
        if (chip_device->subengSIOCTRL[i].subengSIOCTRL[0].valid)
        {
            engine_enable_mask |= LWBIT(i);
        }
    }
    LWSWITCH_REG_WR32(device, _PSMC, _ENABLE_LWLIPT, engine_enable_mask);

    //
    // In bare metal we write ENABLE_LWLIPT to enable the units that aren't
    // disabled by ROM configuration, platform configuration, or regkey override.
    // If we are running inside a VM, the hypervisor has already set ENABLE_LWLIPT
    // and write protected it.  Reading ENABLE_LWLIPT tells us which units we 
    // are allowed to use inside this VM.
    //
    engine_disable_mask = ~LWSWITCH_REG_RD32(device, _PSMC, _ENABLE_LWLIPT);
    engine_disable_mask &= LWBIT(NUM_SIOCTRL_ENGINE_SV10) - 1;
    FOR_EACH_INDEX_IN_MASK(32, i, engine_disable_mask)
    {
        chip_device->subengSIOCTRL[i].subengSIOCTRL[0].valid = LW_FALSE;
        for (j = 0; j < NUM_DLPL_INSTANCES_SV10; j++)
        {
            idx_link = i * NUM_DLPL_INSTANCES_SV10 + j;
            if (idx_link < LWSWITCH_NUM_LINKS_SV10)
            {
                chip_device->link[idx_link].valid = LW_FALSE;
                chip_device->link[idx_link].engDLPL->valid = LW_FALSE;
                chip_device->link[idx_link].engLWLTLC->valid = LW_FALSE;
                chip_device->link[idx_link].engTX_PERFMON->valid = LW_FALSE;
                chip_device->link[idx_link].engRX_PERFMON->valid = LW_FALSE;
                chip_device->link[idx_link].engSIOCTRL->valid = LW_FALSE;
                chip_device->link[idx_link].engMINION->valid = LW_FALSE;
                chip_device->link[idx_link].engLWLIPT->valid = LW_FALSE;
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    // Enable the NPORT units that have been discovered
    engine_enable_mask = 0;
    for (i = 0; i < NUM_NPG_ENGINE_SV10; i++)
    {
        if (chip_device->subengNPG[i].subengNPG[0].valid)
        {
            engine_enable_mask |= LWBIT(i);
        }
    }
    LWSWITCH_REG_WR32(device, _PSMC, _ENABLE_NPG, engine_enable_mask);

    //
    // In bare metal we write ENABLE_NPG to enable the units that aren't
    // disabled by ROM configuration, platform configuration, or regkey override.
    // If we are running inside a VM, the hypervisor has already set ENABLE_NPG
    // and write protected it.  Reading ENABLE_NPG tells us which units we 
    // are allowed to use inside this VM.
    //
    engine_disable_mask = ~LWSWITCH_REG_RD32(device, _PSMC, _ENABLE_NPG);
    engine_disable_mask &= LWBIT(NUM_NPG_ENGINE_SV10) - 1;
    FOR_EACH_INDEX_IN_MASK(32, i, engine_disable_mask)
    {
        chip_device->subengNPG[i].subengNPG[0].valid = LW_FALSE;
        for (j = 0; j < NUM_NPORT_INSTANCES_SV10; j++)
        {
            idx_link = i * NUM_NPORT_INSTANCES_SV10 + j;
            if (idx_link < LWSWITCH_NUM_LINKS_SV10)
            {
                chip_device->link[idx_link].valid = LW_FALSE;
                chip_device->link[idx_link].engNPORT->valid = LW_FALSE;
                chip_device->link[idx_link].engNPORT_PERFMON->valid = LW_FALSE;
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

static void
_lwswitch_init_debug_reset
(
    lwswitch_device *device
)
{
    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _GLOBAL_TRIGGER_ENABLE,
        DRF_DEF(_LWLCTRL, _GLOBAL_TRIGGER_ENABLE, _TRIGGERENABLETX, _INIT) |
        DRF_DEF(_LWLCTRL, _GLOBAL_TRIGGER_ENABLE, _TRIGGERENABLERX, _INIT) |
        DRF_DEF(_LWLCTRL, _GLOBAL_TRIGGER_ENABLE, _TRIGGERENABLESYS, _INIT));

    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _DEBUG_RESET,
        DRF_VAL(_LWLCTRL, _DEBUG_RESET, _LINK, 0) |
        DRF_VAL(_LWLCTRL, _DEBUG_RESET, _COMMON, 0));

    LWSWITCH_FLUSH_MMIO(device);

    lwswitch_os_sleep(1);

    LWSWITCH_SWX_BCAST_WR32_SV10(device, _SWX, _DEBUG_RESET,
        DRF_VAL(_SWX, _DEBUG_RESET, _ALL_AFS, 0));

    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _DEBUG_RESET,
        DRF_DEF(_LWLCTRL, _DEBUG_RESET, _LINK, _INIT) |
        DRF_DEF(_LWLCTRL, _DEBUG_RESET, _COMMON, _INIT));

    LWSWITCH_FLUSH_MMIO(device);

    lwswitch_os_sleep(1);

    LWSWITCH_SWX_BCAST_WR32_SV10(device, _SWX, _DEBUG_RESET,
        DRF_DEF(_SWX, _DEBUG_RESET, _ALL_AFS, _INIT));

    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _GLOBAL_TRIGGER_ENABLE,
        DRF_NUM(_LWLCTRL, _GLOBAL_TRIGGER_ENABLE, _TRIGGERENABLETX, 3) |
        DRF_NUM(_LWLCTRL, _GLOBAL_TRIGGER_ENABLE, _TRIGGERENABLESYS, 1));

    LWSWITCH_SIM_FLUSH_MMIO(device);
}

static void
_lwswitch_init_saw_reset
(
    lwswitch_device *device
)
{
    LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _DEBUG_RESET,
        DRF_DEF(_LWLSAW, _DEBUG_RESET, _SAW, _ASSERT));

    LWSWITCH_FLUSH_MMIO(device);

    // TODO: Delay 50us required, but the lwswitch_os_sleep is only 1msec granularity
    lwswitch_os_sleep(1);

    LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _DEBUG_RESET,
        DRF_DEF(_LWLSAW, _DEBUG_RESET, _SAW, _DEASSERT));

    LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _WARMRESET,
        DRF_DEF(_LWLSAW, _WARMRESET, _SAWWARMRESET, _DEASSERT));

    LWSWITCH_FLUSH_MMIO(device);
}

//
// Bring units out of warm reset on boot.  Used by driver load.
//
static void
_lwswitch_init_warm_reset
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 i;
    LwU32 enable_mask;

    for (i = 0; i < NUM_NPG_ENGINE_SV10; i++)
    {
        enable_mask =
            (chip_device->subengNPG[i].subengNPORT[0].valid ? LWBIT(0) : 0x0) |
            (chip_device->subengNPG[i].subengNPORT[1].valid ? LWBIT(1) : 0x0) |
            (chip_device->subengNPG[i].subengNPORT[2].valid ? LWBIT(2) : 0x0) |
            (chip_device->subengNPG[i].subengNPORT[3].valid ? LWBIT(3) : 0x0);

        LWSWITCH_NPG_WR32_SV10(device, i, _NPG, _WARMRESET,
            DRF_NUM(_NPG, _WARMRESET, _NPORTDISABLE, ~enable_mask) |
            DRF_NUM(_NPG, _WARMRESET, _NPORTWARMRESET, enable_mask));
    }

    for (i = 0; i < NUM_SIOCTRL_ENGINE_SV10; i++)
    {
        enable_mask =
            (chip_device->subengSIOCTRL[i].subengDLPL[0].valid ? LWBIT(0) : 0x0) |
            (chip_device->subengSIOCTRL[i].subengDLPL[1].valid ? LWBIT(1) : 0x0);

        LWSWITCH_SIOCTRL_WR32_SV10(device, i, _LWLCTRL, _RESET,
            DRF_NUM(_LWLCTRL, _RESET, _LINKDISABLE, ~enable_mask) |
            DRF_NUM(_LWLCTRL, _RESET, _LINKRESET, enable_mask));
    }

    LWSWITCH_FLUSH_MMIO(device);

    lwswitch_os_sleep(1);

    for (i = 0; i < NUM_SIOCTRL_ENGINE_SV10; i++)
    {
        enable_mask =
            (chip_device->subengSIOCTRL[i].subengDLPL[0].valid ? LWBIT(0) : 0x0) |
            (chip_device->subengSIOCTRL[i].subengDLPL[1].valid ? LWBIT(1) : 0x0);

        LWSWITCH_SIOCTRL_WR32_SV10(device, i, _LWLCTRL, _CLKCROSS_RESET,
            DRF_NUM(_LWLCTRL, _CLKCROSS_RESET, _CLKCROSSRESET, enable_mask));
    }
}

static void
_lwswitch_init_saw
(
    lwswitch_device *device
)
{
    // Enable PORTSTAT latency block
    LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _GLBLLATENCYTIMERCTRL,
        DRF_DEF(_LWLSAW, _GLBLLATENCYTIMERCTRL, _ENABLE, _ENABLE));

    //
    // Willow_Driver_Tables.xlsx requests
    // using these values rather than __PROD until silicon is characterized
    // and __PROD values updated.
    //

    // NOTE: Requested variance from __PROD
    LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _OVERTEMPONCNTR,
        DRF_NUM(_LWLSAW, _OVERTEMPONCNTR, _COUNT, 0x0));

    // NOTE: Requested variance from __PROD
    LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _OVERTEMPOFFCNTR,
        DRF_NUM(_LWLSAW, _OVERTEMPOFFCNTR, _COUNT, 0xFFFFFFFF));

    LWSWITCH_FLUSH_MMIO(device);
}

static void
_lwswitch_init_swx
(
    lwswitch_device *device
)
{
    LwU32 swx_dbi = device->regkeys.crossbar_DBI;
    LwU32 xsu_ctrl;

    LWSWITCH_SWX_BCAST_WR32_SV10(device, _SWX, _DBI,
        DRF_NUM(_SWX, _DBI, _ENC_EN, swx_dbi));

    switch (device->regkeys.bandwidth_shaper)
    {
        case LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_XSD:
            xsu_ctrl =
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_SHAPER_ENABLE, __DISABLE) |
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_TXN_FAIR_MODE, __PROD)    |
                DRF_DEF(_AFS, _XSU_CTRL, _XSD_SHAPER_ENABLE,    __ENABLE);
            LWSWITCH_PRINT(device, SETUP,
                "Bandwidth shaper: %s (ctrl=0x%x)\n",
                "XSD", xsu_ctrl);
        break;

        case LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_BUCKET_BW:
            xsu_ctrl =
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_SHAPER_ENABLE, __ENABLE)  |
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_TXN_FAIR_MODE, __DISABLE) |
                DRF_DEF(_AFS, _XSU_CTRL, _XSD_SHAPER_ENABLE,    __DISABLE);
            LWSWITCH_PRINT(device, SETUP,
                "Bandwidth shaper: %s (ctrl=0x%x)\n",
                "BUCKET_BW", xsu_ctrl);
        break;

        case LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_BUCKET_TX_FAIR:
            xsu_ctrl =
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_SHAPER_ENABLE, __ENABLE) |
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_TXN_FAIR_MODE, __ENABLE) |
                DRF_DEF(_AFS, _XSU_CTRL, _XSD_SHAPER_ENABLE,    __DISABLE);
            LWSWITCH_PRINT(device, SETUP,
                "Bandwidth shaper: %s (ctrl=0x%x)\n",
                "BUCKET_TX_FAIR", xsu_ctrl);
        break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "Unknown bandwidth_shaper (0x%x)\n",
                device->regkeys.bandwidth_shaper);
            // Deliberate fallthrough
        case LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_PROD:
            xsu_ctrl =
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_SHAPER_ENABLE, __PROD) |
                DRF_DEF(_AFS, _XSU_CTRL, _BUCKET_TXN_FAIR_MODE, __PROD) |
                DRF_DEF(_AFS, _XSU_CTRL, _XSD_SHAPER_ENABLE,    __PROD);
            LWSWITCH_PRINT(device, SETUP,
                "Bandwidth shaper: %s (ctrl=0x%x)\n",
                "PROD", xsu_ctrl);
        break;
    }
    LWSWITCH_AFS_MC_BCAST_WR32_SV10(device, _AFS, _XSU_CTRL, xsu_ctrl);

    LWSWITCH_AFS_MC_BCAST_WR32_SV10(device, _AFS, _XSU_CYA,
        DRF_DEF(_AFS, _XSU_CYA, _XSD_CNT_BASED_BUCKET_INHIBIT, __PROD) |
        DRF_DEF(_AFS, _XSU_CYA, _XSD_OUTSTANDING_FLIT_CNT,     __PROD));

    LWSWITCH_AFS_MC_BCAST_WR32_SV10(device, _AFS, _ERR_CYA,
        DRF_DEF(_AFS, _ERR_CYA, _SRCID_TAGGING_EN, __PROD));

    LWSWITCH_AFS_MC_BCAST_WR32_SV10(device, _AFS, _PMON_CFG0,
        DRF_DEF(_AFS, _PMON_CFG0, _PMON_ENABLE, __PROD) |
        DRF_DEF(_AFS, _PMON_CFG0, _SRC_SEL, _INIT) |
        DRF_DEF(_AFS, _PMON_CFG0, _VC_SEL, _INIT) |
        DRF_DEF(_AFS, _PMON_CFG0, _PKT_SEL, _INIT));

    LWSWITCH_FLUSH_MMIO(device);
}

static void
_lwswitch_init_npg_multicast
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 idx_npg;
    LwU32 nport_mask;

    //
    // Walk the NPGs and build the mask of extant NPORTs
    //
    for (idx_npg=0; idx_npg < NUM_NPG_ENGINE_SV10; idx_npg++)
    {
        if (chip_device->engNPG[idx_npg].valid &&
            chip_device->subengNPG[idx_npg].subengNPG[0].valid)
        {
            nport_mask =
                (chip_device->subengNPG[idx_npg].subengNPORT[0].valid ? LWBIT(0) : 0x0) |
                (chip_device->subengNPG[idx_npg].subengNPORT[1].valid ? LWBIT(1) : 0x0) |
                (chip_device->subengNPG[idx_npg].subengNPORT[2].valid ? LWBIT(2) : 0x0) |
                (chip_device->subengNPG[idx_npg].subengNPORT[3].valid ? LWBIT(3) : 0x0);

            LWSWITCH_NPG_WR32_SV10(device, idx_npg, _NPG, _CTRL_PRI_MULTICAST,
                DRF_NUM(_NPG, _CTRL_PRI_MULTICAST, _NPORT_ENABLE, nport_mask) |
                DRF_DEF(_NPG, _CTRL_PRI_MULTICAST, _READ_MODE, _AND_ALL_BUSSES));

            LWSWITCH_NPGPERF_WR32_SV10(device, idx_npg, _NPGPERF, _CTRL_PRI_MULTICAST,
                DRF_NUM(_NPGPERF, _CTRL_PRI_MULTICAST, _NPORT_ENABLE, nport_mask) |
                DRF_DEF(_NPGPERF, _CTRL_PRI_MULTICAST, _READ_MODE, _AND_ALL_BUSSES));
        }
    }
}

static void
_lwswitch_init_route_control
(
    lwswitch_device *device
)
{
    LwU32 val, i;

    for (i = 0; i < LWSWITCH_NUM_LINKS_SV10; i++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, i))
        {
            continue;
        }

        val = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _ROUTE, _ROUTE_CONTROL);

        val |= DRF_DEF(_ROUTE, _ROUTE_CONTROL, _SWECCENB, _HWGEN)                 |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _ECCWRITEBACKEN, _DISABLE)         |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _DEBUGENB, _DISABLE)               |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _URRESPENB, _ENABLE)               |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _ECCENB, __PROD)                   |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _ECCCNTRENB, __PROD)               |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _ECCCOUNTSINGLEBIT0, __PROD)       |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _ECCCOUNTDOUBLEBITHEADER0, __PROD) |
               DRF_DEF(_ROUTE, _ROUTE_CONTROL, _STOREANDFORWARD, __PROD);

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _ROUTE, _ROUTE_CONTROL, val);

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _ROUTE, _ECC_ERROR_COUNT, 0x0);

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _ROUTE, _ECC_ERROR_LIMIT, 0x1);
    }
}

static void
_lwswitch_init_egress_control
(
    lwswitch_device *device
)
{
    LwU32 val;
    LwU32 i;

    for (i = 0; i < LWSWITCH_NUM_LINKS_SV10; i++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, i))
        {
            continue;
        }

        val = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _CTRL);

        val = FLD_SET_DRF(_EGRESS, _CTRL, _GENFLUSHERRRSP, _ENABLE, val);
        val = FLD_SET_DRF(_EGRESS, _CTRL, _ECCENABLE0, _ENABLE, val);
        val = FLD_SET_DRF(_EGRESS, _CTRL, _ECCENABLE1, _ENABLE, val);

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _EGRESS, _CTRL, val);
    }

    LWSWITCH_SIM_FLUSH_MMIO(device);
}

static void
_lwswitch_init_ingress_control
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 i;
    LwU32 val;

    for (i = 0; i < LWSWITCH_NUM_LINKS_SV10; i++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, i))
        {
            continue;
        }

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _INGRESS, _CTRL,
            DRF_DEF(_INGRESS, _CTRL, _SWECCENB0, _DISABLE)        |
            DRF_DEF(_INGRESS, _CTRL, _ECCWRITEBACKENB0, _DISABLE) |
            DRF_DEF(_INGRESS, _CTRL, _ECCENB0, _ENABLE)           |
            DRF_DEF(_INGRESS, _CTRL, _ECCCOUNTERENB0, __PROD)     |
            DRF_DEF(_INGRESS, _CTRL, _ECCCOUNTSINGLEBIT0, __PROD) |
            DRF_DEF(_INGRESS, _CTRL, _ADDRMAPENB, __PROD));

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _INGRESS, _ECC_ERROR_COUNT,
            DRF_DEF(_INGRESS, _ECC_ERROR_COUNT, _ERROR_COUNT, __PROD));

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _INGRESS, _ECC_ERROR_LIMIT,
            DRF_DEF(_INGRESS, _ECC_ERROR_LIMIT, _ERROR_LIMIT, __PROD));

        val = (chip_device->link[i].engNPORT->instance << 8);

        LWSWITCH_LINK_WR32_SV10(device, i, NPORT, _INGRESS, _REQLINKID,
            DRF_NUM(_INGRESS, _REQLINKID, _REQLINKID, val));
    }
}

static void
_lwswitch_init_bubble_squash
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 i;
    LwU32 val;
    LwU32 link_clock_khz;
    LwU32 bubble_squash;
    LwU32 bubble_squash_default =
        DRF_MASK(LW_LWLCTRL_CLKCROSS_0_BUBBLE_SQUASH_TERM_COUNT_TERM_COUNT);
    LwU32 pll_freq_khz = device->switch_pll.freq_khz;
    LwU32 rxlinkclk_khz;

    // update bubble squash
    for (i = 0; i < LWSWITCH_NUM_LINKS_SV10; i++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, SIOCTRL, i))
        {
            continue;
        }

        //
        // Link rate is selected per link pair, so really even & odd link
        // rate will both be set to the rate selected for the even link.
        //
        if (device->regkeys.lwlink_speed_control == LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_20G)
        {
            // Select 20Gbps
            link_clock_khz = 20000000;
        }
        else
        {
            // Select 25.781Gbps
            link_clock_khz = 25781000;
        }

        chip_device->link[i].link_clock_khz = link_clock_khz;

        rxlinkclk_khz = link_clock_khz / 16;

        //
        // Callwlate bubble squash term count.
        //
        if (rxlinkclk_khz >= pll_freq_khz)
        {
            bubble_squash = bubble_squash_default;
        }
        else
        {
            bubble_squash = pll_freq_khz / (pll_freq_khz - rxlinkclk_khz);
            bubble_squash = LW_MAX(bubble_squash, 0x3) - 2;
            bubble_squash = LW_MIN(bubble_squash, bubble_squash_default);
        }

        val = DRF_NUM(_LWLCTRL, _CLKCROSS_0_BUBBLE_SQUASH_TERM_COUNT,
            _TERM_COUNT, bubble_squash);

        if (i & 0x1)
        {
            // link 1
            LWSWITCH_LINK_WR32_SV10(device, i, SIOCTRL, _LWLCTRL,
                _CLKCROSS_1_BUBBLE_SQUASH_TERM_COUNT, val);
        }
        else
        {
            // link 0
            LWSWITCH_LINK_WR32_SV10(device, i, SIOCTRL, _LWLCTRL,
                _CLKCROSS_0_BUBBLE_SQUASH_TERM_COUNT, val);
        }
    }
}

static void
_lwswitch_init_lwlipt
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 sioctrl;
    LwU32 nport_dbi = (device->regkeys.link_DBI == LW_SWITCH_REGKEY_LINK_DBI_ENABLE ?
                    DRF_MASK(LW_LWLCTRL_LWLIPT_GLOBAL_CTRL_NPORT_DBI_ENB): 0);
    LwU32 link_mask;

    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _LWLIPT_GLOBAL_CTRL,
        DRF_DEF(_LWLCTRL, _LWLIPT_GLOBAL_CTRL, _NPORT_BUBBLE_SQUASH_ENB, __PROD) |
        DRF_NUM(_LWLCTRL, _LWLIPT_GLOBAL_CTRL, _NPORT_DBI_ENB, nport_dbi) |
        DRF_DEF(_LWLCTRL, _LWLIPT_GLOBAL_CTRL, _NPORT_DATA_BE_FLIT_GATE_ENB, __PROD));

    for (sioctrl=0; sioctrl < chip_device->numSIOCTRL; sioctrl++)
    {
        if ((chip_device->engSIOCTRL[sioctrl].valid) &&
            (chip_device->subengSIOCTRL[sioctrl].subengLWLIPT[0].valid))
        {
            link_mask =
                (chip_device->subengSIOCTRL[sioctrl].subengDLPL[0].valid ? LWBIT(0) : 0x0) |
                (chip_device->subengSIOCTRL[sioctrl].subengDLPL[1].valid ? LWBIT(1) : 0x0);

            LWSWITCH_LWLIPT_WR32_SV10(device, sioctrl, _LWLIPT, _CTRL_PRI_MULTICAST,
                DRF_NUM(_LWLIPT, _CTRL_PRI_MULTICAST, _TL_ENABLE, link_mask) |
                DRF_NUM(_LWLIPT, _CTRL_PRI_MULTICAST, _DL_ENABLE, link_mask) |
                DRF_DEF(_LWLIPT, _CTRL_PRI_MULTICAST, _READ_MODE, _AND_ALL_BUSSES));
        }
    }

    LWSWITCH_SIM_FLUSH_MMIO(device);

    _lwswitch_init_bubble_squash(device);
}

static void
_lwswitch_init_portstat_counters
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwlStatus retval;
    LwU32 window_limit;
    LwU32 idx_channel;
    LWSWITCH_SET_LATENCY_BINS default_latency_bins;

    //
    // PORTSTAT_WINDOWLIMIT = 0xFFFFFFFF in switch clocks (1.64GHz) is ~2.6 seconds
    // Sample interval must be <= 2600ms
    //
    chip_device->latency_stats->sample_interval_msec = 2000; // 2 second sample interval

    //
    // These bin thresholds are values provided by Arch based off
    // switch latency expectations.
    //
    for (idx_channel=0; idx_channel < LWSWITCH_MAX_VCS; idx_channel++)
    {
        default_latency_bins.bin[idx_channel].lowThreshold = 120;    // 120ns
        default_latency_bins.bin[idx_channel].medThreshold = 200;    // 200ns
        default_latency_bins.bin[idx_channel].hiThreshold  = 1000;   // 1us
    }

    retval = lwswitch_ctrl_set_latency_bins(device, &default_latency_bins);
    LWSWITCH_ASSERT(retval == LWL_SUCCESS);

    //
    // Program _WINDOWLIMIT (in clock ticks) to the sampling interval
    // Snap time 0xFFFFFFFF in switch clocks (1.64GHz) is ~2.6 seconds
    //
    window_limit = device->switch_pll.freq_khz * chip_device->latency_stats->sample_interval_msec;

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _PORTSTAT_WINDOW_LIMIT,
        DRF_NUM(_NPORT, _PORTSTAT_WINDOW_LIMIT, _WINDOWLIMIT, window_limit));

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _PORTSTAT_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_CONTROL, _CONTSWEEPMODE, _CONTINUOUS) |
        DRF_DEF(_NPORT, _PORTSTAT_CONTROL, _RANGESELECT, _BITS13TO0));

     LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _PORTSTAT_SOURCE_FILTER,
         DRF_NUM(_NPORT, _PORTSTAT_SOURCE_FILTER, _SRCFILTERBIT, 0x3FFFF));

     // NPORT PORTSTAT enable
     LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _PORTSTAT_SNAP_CONTROL,
         DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE));
}

static void _lwswitch_reset_nport_debug_state_sv10
(
    lwswitch_device *device
)
{
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _EGRESS, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _EGRESS, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_C_FIRST_NPORT, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_C_STATUS_NPORT, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_UC_FIRST_NPORT, ~0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_UC_STATUS_NPORT, ~0);

    LWSWITCH_FLUSH_MMIO(device);
}

static void
_lwswitch_init_cmd_routing
(
    lwswitch_device *device
)
{
    LwU32 val ;

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _CMD_ROUTE_TABLE0, 0);

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _CMD_ROUTE_TABLE1, 0);

    // Set RANDOM policy only for reponses.
    val = DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN16, _RANDOM) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN17, _RANDOM) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN18, _RANDOM) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN19, _RANDOM) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN20, _RANDOM) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN21, _RANDOM);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _CMD_ROUTE_TABLE2, val);

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _CMD_ROUTE_TABLE3, 0);

    LWSWITCH_SIM_FLUSH_MMIO(device);
}

#define LWSWITCH_CLEAR_CRUMBSTORE(device, state, val)                       \
do                                                                          \
{                                                                           \
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, state, _RAM_ADDRESS, val);    \
                                                                            \
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, state, _RAM_DATA1, 0);        \
                                                                            \
    for (i = 0; i < (1 << DRF_SIZE(LW## state ##_RAM_ADDRESS_ADDR)); i++)   \
    {                                                                       \
        LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, state, _RAM_DATA0, 0);    \
    }                                                                       \
                                                                            \
    LWSWITCH_FLUSH_MMIO(device);                                            \
} while(0)

static void
_lwswitch_clear_flush_state_ram
(
    lwswitch_device *device
)
{
    LwU32 val;
    LwU32 i;

    // Clear crumbstore
    val = DRF_NUM(_FSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_NUM(_FSTATE, _RAM_ADDRESS, _CRUMBTAGSTORESEL, 1) |
          DRF_NUM(_FSTATE, _RAM_ADDRESS, _AUTO_INCR, 1);

    LWSWITCH_CLEAR_CRUMBSTORE(device, _FSTATE, val);
}

//
// See IAS section 7.5.19.2.2.1 TagPool initialization
// Also, see Bug 1913897 for more info.
//
static void
_lwswitch_init_flush_state
(
    lwswitch_device *device
)
{
    LwU32 val;

    _lwswitch_clear_flush_state_ram(device);

    // This step should reset the RDWR RAM pointers.
    val = DRF_DEF(_NPORT, _TAGPOOLENTRYCOUNT_6, _LIMIT, _INIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _TAGPOOLENTRYCOUNT_6, val);

    val = DRF_DEF(_FSTATE, _TAGPOOLWATERMARK, _DEPTH, __PROD);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _TAGPOOLWATERMARK, val);

    //
    // Toggle HWINIT bit 1->0->1, to start HWINIT mode for flushstate.
    //
    val = DRF_NUM(_FSTATE, _FLUSHSTATECONTROL, _HWINIT_TAGPOOLRAM, 0x1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _FLUSHSTATECONTROL, val);

    val = DRF_NUM(_FSTATE, _FLUSHSTATECONTROL, _HWINIT_TAGPOOLRAM, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _FLUSHSTATECONTROL, val);

    val = DRF_DEF(_FSTATE, _FLUSHSTATECONTROL, _SWECCENB, _DISABLE)   |
          DRF_DEF(_FSTATE, _FLUSHSTATECONTROL, _ECCENABLE_0, _ENABLE) |
          DRF_DEF(_FSTATE, _FLUSHSTATECONTROL, _ECCENABLE_1, _ENABLE) |
          DRF_DEF(_FSTATE, _FLUSHSTATECONTROL, _ECCENABLE_2, _ENABLE) |
          DRF_DEF(_FSTATE, _FLUSHSTATECONTROL, _HWINIT_TAGPOOLRAM, _HWINIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _FLUSHSTATECONTROL, val);

    LWSWITCH_SIM_FLUSH_MMIO(device);
}

//
// If any tag counts are not zero, write NPORT.RESTORETAGPOOLCOUNTS corresponding
// to the VC counter which is not zero until the register corresponding to that
// VC is zero. Each write to NPORT.RESTORETAGPOOLCOUNTS.StepXXXXTagCount decrements
// the shadow count (total tags used) by one and increments the available tag count
// by one.
//
#define LWSWITCH_RESTORE_TAG_COUNT(device, link, tag_count, vc)               \
do                                                                            \
{                                                                             \
    val = DRF_NUM(_NPORT, _RESTORETAGPOOLCOUNTS, vc, 0x1);                    \
    for (j = 0; j < tag_count; j++)                                           \
    {                                                                         \
        LWSWITCH_LINK_WR32_SV10(device, link, NPORT, _NPORT, _RESTORETAGPOOLCOUNTS, val);   \
    }                                                                         \
} while(0)

static void
_lwswitch_restore_tag_counts
(
    lwswitch_device *device
)
{
    LwU32 val;
    LwU32 i;
    LwU32 j;
    LwU32 tag_count;

    for (i = 0; i < LWSWITCH_NUM_LINKS_SV10; i++)
    {
        if (LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, i))
        {
            // CREQ0 tag counts (VC0)
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS0);
            LWSWITCH_RESTORE_TAG_COUNT(device, i, tag_count, _STEPCREQTAGCOUNT);
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS0);
            LWSWITCH_ASSERT(tag_count == 0);

            // DOWNGRADE tag counts (VC1)
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS1);
            LWSWITCH_RESTORE_TAG_COUNT(device, i, tag_count, _STEPDOWNGRADETAGCOUNT);
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS1);
            LWSWITCH_ASSERT(tag_count == 0);

            // ATR tag counts (VC2)
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS2);
            LWSWITCH_RESTORE_TAG_COUNT(device, i, tag_count, _STEPATRTAGCOUNT);
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS2);
            LWSWITCH_ASSERT(tag_count == 0);

            // ATSD tag counts (VC3)
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS3);
            LWSWITCH_RESTORE_TAG_COUNT(device, i, tag_count, _STEPATRTAGCOUNT);
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS3);
            LWSWITCH_ASSERT(tag_count == 0);

            // PROBE tag counts (VC4)
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS4);
            LWSWITCH_RESTORE_TAG_COUNT(device, i, tag_count, _STEPPROBETAGCOUNT);
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _EGRESS, _TAGCOUNTS4);
            LWSWITCH_ASSERT(tag_count == 0);

            // CREQ transdone tag counts
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _ROUTE, _TDCREDIT);
            tag_count = DRF_VAL(_ROUTE, _TDCREDIT, _CREQ_COUNT, tag_count);
            LWSWITCH_RESTORE_TAG_COUNT(device, i, tag_count, _STEPCREQTDTAGCOUNT);
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _ROUTE, _TDCREDIT);
            tag_count = DRF_VAL(_ROUTE, _TDCREDIT, _CREQ_COUNT, tag_count);
            LWSWITCH_ASSERT(tag_count == 0);

            // DOWNGRADE transdone tag counts
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _ROUTE, _TDCREDIT);
            tag_count = DRF_VAL(_ROUTE, _TDCREDIT, _DGD_COUNT, tag_count);
            LWSWITCH_RESTORE_TAG_COUNT(device, i, tag_count, _STEPDGDTDTAGCOUNT);
            tag_count = LWSWITCH_LINK_RD32_SV10(device, i, NPORT, _ROUTE, _TDCREDIT);
            tag_count = DRF_VAL(_ROUTE, _TDCREDIT, _DGD_COUNT, tag_count);
            LWSWITCH_ASSERT(tag_count == 0);
        }
    }
}

static void
_lwswitch_clear_tag_state_ram
(
    lwswitch_device *device
)
{
    LwU32 val;
    LwU32 i;

    // Clear crumbstore for CREQ (VC0)
    val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _CRUMBTAGSTORESEL, 1) |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1)        |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _VC, _VC0_CREQ);
    LWSWITCH_CLEAR_CRUMBSTORE(device, _TSTATE, val);

    // Clear crumbstore for DOWNGRADE (VC1)
    val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _CRUMBTAGSTORESEL, 1) |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1)        |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _VC, _VC1_DOWNGRADE);
    LWSWITCH_CLEAR_CRUMBSTORE(device, _TSTATE, val);

    // Clear crumbstore for ATR (VC2)
    val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _CRUMBTAGSTORESEL, 1) |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1)        |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _VC, _VC2_ATR);
    LWSWITCH_CLEAR_CRUMBSTORE(device, _TSTATE, val);

    // Clear crumbstore for ATSD (VC3)
    val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _CRUMBTAGSTORESEL, 1) |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1)        |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _VC, _VC3_ATSD);
    LWSWITCH_CLEAR_CRUMBSTORE(device, _TSTATE, val);

    // Clear crumbstore for PROBE (VC4)
    val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _CRUMBTAGSTORESEL, 1) |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1)        |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _VC, _VC4_PROBE);
    LWSWITCH_CLEAR_CRUMBSTORE(device, _TSTATE, val);

    // Clear crumbstore for TRANSDONE (VC5)
    val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _CRUMBTAGSTORESEL, 1) |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1)        |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _VC, _VC5_TRANSDONE);
    LWSWITCH_CLEAR_CRUMBSTORE(device, _TSTATE, val);
}

//
// See IAS section 7.5.19.2.2.1 TagPool initialization
// Also, see Bug 1913897 for more info.
//
static void
_lwswitch_init_tag_state
(
    lwswitch_device *device
)
{
    LwU32 val = 0;

    _lwswitch_restore_tag_counts(device);

    _lwswitch_clear_tag_state_ram(device);

    // This should reset the RDWR RAM pointers.
    val = DRF_DEF(_NPORT, _TAGPOOLENTRYCOUNT_0, _LIMIT, _INIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _TAGPOOLENTRYCOUNT_0, val);

    val = DRF_DEF(_NPORT, _TAGPOOLENTRYCOUNT_1, _LIMIT, _INIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _TAGPOOLENTRYCOUNT_1, val);

    val = DRF_DEF(_NPORT, _TAGPOOLENTRYCOUNT_2, _LIMIT, _INIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _TAGPOOLENTRYCOUNT_2, val);

    val = DRF_DEF(_NPORT, _TAGPOOLENTRYCOUNT_3, _LIMIT, _INIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _TAGPOOLENTRYCOUNT_3, val);

    val = DRF_DEF(_NPORT, _TAGPOOLENTRYCOUNT_4, _LIMIT, _INIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _TAGPOOLENTRYCOUNT_4, val);

    val = DRF_DEF(_NPORT, _TAGPOOLENTRYCOUNT_5, _LIMIT, _INIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _TAGPOOLENTRYCOUNT_5, val);

    // Toggle HWINIT bit 1->0->1, to start HWINIT mode for tagstate.
    val = DRF_NUM(_TSTATE, _TAGSTATECONTROL, _HWINIT_TAGPOOLRAM, 0x1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _TAGSTATECONTROL, val);

    val = DRF_NUM(_TSTATE, _TAGSTATECONTROL, _HWINIT_TAGPOOLRAM, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _TAGSTATECONTROL, val);

    val = DRF_DEF(_TSTATE, _TAGSTATECONTROL, _SWECCENB, _DISABLE)   |
          DRF_DEF(_TSTATE, _TAGSTATECONTROL, _ECCENABLE_0, _ENABLE) |
          DRF_DEF(_TSTATE, _TAGSTATECONTROL, _ECCENABLE_1, _ENABLE) |
          DRF_DEF(_TSTATE, _TAGSTATECONTROL, _HWINIT_TAGPOOLRAM, _HWINIT);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _TAGSTATECONTROL, val);

    LWSWITCH_SIM_FLUSH_MMIO(device);
}

static void
_lwswitch_init_ingress_next_hops
(
    lwswitch_device *device
)
{
    LwU32 i;
    LwU32 val;

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATRMAPDATA0, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATRMAPDATA1, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATRMAPDATA2, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATRMAPDATA3, 0x0);

    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _CPUVIRTMAPDATA0, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _CPUVIRTMAPDATA1, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _CPUVIRTMAPDATA2, 0x0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _CPUVIRTMAPDATA3, 0x0);

#define LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(regnum)                                          \
do {                                                                                          \
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATSDRESULTMAPDATA0_ ## regnum, 0x0);           \
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATSDRESULTMAPDATA1_ ## regnum, 0x0);           \
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATSDRESULTMAPDATA2_ ## regnum, 0x0);           \
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATSDRESULTMAPDATA3_ ## regnum, 0x0);           \
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ATSDMATCHMAPDATA_ ## regnum, 0x0);             \
} while(0)

    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(0);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(1);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(2);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(3);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(4);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(5);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(6);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(7);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(8);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(9);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(10);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(11);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(12);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(13);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(14);
    LWSWITCH_INIT_ATSD_INGRESS_REGISTERS(15);

    //
    // On emulation and rtlsim, routing table init adds a huge runtime penalty.
    // Hence, enable for silicon only!
    //
    if (!(IS_RTLSIM(device) || IS_EMULATION(device)))
    {
        val = DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_INDEX, 0)         |
              DRF_DEF(_INGRESS, _REQRSPMAPADDR, _TABLE_SELECT, _REQUEST) |
              DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1);

        LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _REQRSPMAPADDR, val);

        for (i = 0; i < INGRESS_MAP_TABLE_SIZE; i++)
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _REQRSPMAPDATA0, 0);
        }

        val = DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_INDEX, 0)          |
              DRF_DEF(_INGRESS, _REQRSPMAPADDR, _TABLE_SELECT, _RESPONSE) |
              DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1);

        LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _REQRSPMAPADDR, val);

        for (i = 0; i < INGRESS_MAP_TABLE_SIZE; i++)
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _REQRSPMAPDATA0, 0);
        }
    }

    LWSWITCH_SIM_FLUSH_MMIO(device);
}

void
lwswitch_set_ganged_link_table_sv10
(
    lwswitch_device *device,
    LwU32            port,
    LwU32            firstIndex,
    LwU32            numEntries
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 i;
    ROUTE_GANG_ENTRY_SV10 *table;

    LWSWITCH_LINK_WR32_SV10(device, port, NPORT, _ROUTE, _REG_TABLE_ADDRESS,
        DRF_NUM(_ROUTE, _REG_TABLE_ADDRESS, _INDEX, firstIndex) |
        DRF_NUM(_ROUTE, _REG_TABLE_ADDRESS, _AUTO_INCR, 1));

    table = &chip_device->link[port].ganged_link_table[firstIndex];

    for (i = 0; i < numEntries; i++)
    {
        LWSWITCH_LINK_WR32_SV10(device, port, NPORT, _ROUTE, _REG_TABLE_DATA0,
            DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _DATA, table[i].regtabledata0));
    }

    LWSWITCH_FLUSH_MMIO(device);
}

static void
_lwswitch_init_ganged_link_routing
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32        gang_index, gang_size;
    LwU32        gang_entry;
    LwU32        block_index;
    const LwU32  block_count = 8;
    LwU32        block_size = (ROUTE_GANG_TABLE_SIZE / block_count);
    LwU32        link;
    LwU32        table_index;

    //
    // Refer to switch IAS 7.5.18.2.4 and 7.5.19.5.2.2
    // Also hw\doc\gpu\volta\lwswitch\design\IAS\lwswitch_pri_addr.xlsm "route" tab
    //
    // The ganged link routing table is composed of 256 entries divided into 8 sections.
    // Each section specifies how requests should be routed through the ganged links.
    // Each 32-bit entry is composed of eight 4-bit fields specifying the set of of links
    // to distribute through.  More complex spray patterns could be constructed, but for
    // now initialize it with a uniform distribution pattern.
    // If NPORT 0..7 are ganged together, their routing tables would be loaded with:
    // Typically the first section would be filled with (0,1,2,3,4,5,6,7)
    // Typically the second section would be filled with (0,0,0,0,0,0,0,0)
    // Typically the third section would be filled with (0,1,0,1,0,1,0,1)
    //  :
    // The last section would typically be filled with (0,1,2,3,4,5,6,0),(1,2,3,4,5,6,0,1),...
    // Note that section 0 corresponds with 8 ganged links.  Section N corresponds with
    // N ganged links.
    //

    for (link = 0; link < LWSWITCH_NUM_LINKS_SV10; link++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, link))
        {
            continue;
        }

        table_index = 0;

        for (block_index = 0; block_index < block_count; block_index++)
        {
            gang_size = ((block_index==0) ? 8 : block_index);

            for (gang_index = 0; gang_index < block_size; gang_index++)
            {
                gang_entry =
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(0), (8*gang_index+0) % gang_size) |
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(1), (8*gang_index+1) % gang_size) |
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(2), (8*gang_index+2) % gang_size) |
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(3), (8*gang_index+3) % gang_size) |
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(4), (8*gang_index+4) % gang_size) |
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(5), (8*gang_index+5) % gang_size) |
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(6), (8*gang_index+6) % gang_size) |
                    DRF_NUM(_ROUTE, _REG_TABLE_DATA0, _GANG(7), (8*gang_index+7) % gang_size);

                chip_device->link[link].ganged_link_table[table_index].regtabledata0 = gang_entry;
                table_index++;
            }
        }

        lwswitch_set_ganged_link_table_sv10(device, link, 0, ROUTE_GANG_TABLE_SIZE);
    }

    // Setup address hashing function to distribute across all the ganged links
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _FLOWIDADDRMASKLO,
        DRF_DEF(_INGRESS, _FLOWIDADDRMASKLO, _FLOWIDMASK, __PROD));
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _FLOWIDADDRMASKHI,
        DRF_DEF(_INGRESS, _FLOWIDADDRMASKHI, _FLOWIDMASK, __PROD));

    LWSWITCH_SIM_FLUSH_MMIO(device);
}

/*
 * @Brief : Send priv ring command and wait for completion
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 * @param[in] cmd           encoded priv ring command
 */
LwlStatus
lwswitch_ring_master_cmd_sv10
(
    lwswitch_device *device,
    LwU32 cmd
)
{
    LwU32 value;
    LWSWITCH_TIMEOUT timeout;

    LWSWITCH_REG_WR32(device, _PPRIV_MASTER, _RING_COMMAND, cmd);
    LWSWITCH_FLUSH_MMIO(device);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        value = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_COMMAND);
        if (FLD_TEST_DRF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _NO_CMD, value))
        {
            break;
        }
    }
    while (!lwswitch_timeout_check(&timeout));

    if (!FLD_TEST_DRF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _NO_CMD, value))
    {
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    return LWL_SUCCESS;
}

/*
 * @brief Determines if a link's lanes are reversed
 *
 * @param[in] device    a reference to the device to query
 * @param[in] linkId    Target link ID
 *
 * @return LW_TRUE if a link's lanes are reversed
 */
static LwBool
_lwswitch_link_lane_reversed
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwU32 regData;

    regData = LWSWITCH_LINK_RD32_SV10(device, linkId, DLPL, _PLWL_SL1, _CONFIG_RX);

    // HW may reverse the lane ordering or it may be overridden by SW.
    if (FLD_TEST_DRF(_PLWL_SL1, _CONFIG_RX, _REVERSAL_OVERRIDE, _ON, regData))
    {
        // Overridden
        if (FLD_TEST_DRF(_PLWL_SL1, _CONFIG_RX, _LANE_REVERSE, _ON, regData))
        {
            return LW_TRUE;
        }
        else
        {
            return LW_FALSE;
        }
    }
    else
    {
        // Sensed in HW
        if (FLD_TEST_DRF(_PLWL_SL1, _CONFIG_RX, _HW_LANE_REVERSE, _ON, regData))
        {
            return LW_TRUE;
        }
        else
        {
            return LW_FALSE;
        }
    }

    return LW_FALSE;
}

/*
 * @brief Process the information read from ROM tables and apply it to device
 * settings.
 *
 * @param[in] device    a reference to the device to query
 * @param[in] firmware  Information parsed from ROM tables
 */
static void
_lwswitch_process_firmware_info_sv10
(
    lwswitch_device *device,
    LWSWITCH_FIRMWARE *firmware
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 idx_link;

    if (device->firmware.firmware_size == 0)
    {
        return;
    }

    if (device->firmware.lwlink.link_config_found)
    {
        //
        // If the link enables were not already overridden by regkey, then
        // apply the ROM link enables
        //
        if (device->regkeys.link_enable_mask == LW_U32_MAX)
        {
            for (idx_link = 0; idx_link < lwswitch_get_num_links(device); idx_link++)
            {
                if ((device->firmware.lwlink.link_enable_mask & LWBIT64(idx_link)) == 0)
                {
                    chip_device->link[idx_link].valid = LW_FALSE;
                }
            }
        }
    }
}

void *
lwswitch_alloc_chipdevice_sv10
(
    lwswitch_device *device
)
{
    void *chip_device;

    chip_device = lwswitch_os_malloc(sizeof(sv10_device));
    if (NULL != chip_device)
    {
        lwswitch_os_memset(chip_device, 0, sizeof(sv10_device));
    }

    device->chip_id = LW_PSMC_BOOT_42_CHIP_ID_SVNP01;
    return(chip_device);
}

static LwlStatus
lwswitch_initialize_pmgr_sv10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_IMPLEMENTED;
}

static LwlStatus
lwswitch_initialize_ip_wrappers_sv10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_IMPLEMENTED;
}

static LwlStatus
lwswitch_initialize_route_sv10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*
 * @Brief : Initializes an LWSwitch hardware state
 *          Refer to switch IAS 7.5.11.1 LWSwitch init sequence.
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 *
 * @returns                 LWL_SUCCESS if the action succeeded
 *                          -LWL_BAD_ARGS if bad arguments provided
 *                          -LWL_PCI_ERROR if bar info unable to be retrieved
 */
static LwlStatus
lwswitch_initialize_device_state_sv10
(
    lwswitch_device *device
)
{
    LwlStatus retval = LWL_SUCCESS;
    LwBool enumerated = LW_FALSE;
    LwU32 value;
    LwU32 i;
    sv10_device *chip_device;

    // alloc sv10_device structure
    device->chip_device = lwswitch_alloc_chipdevice(device);
    if (NULL == device->chip_device)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwswitch_os_malloc during chip_device creation failed!\n");
        retval = -LWL_NO_MEM;
        goto lwswitch_initialize_device_state_exit;
    }
    chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    // Make sure interrupts are disabled before we enable interrupts with the OS.
    lwswitch_lib_disable_interrupts(device);

    LWSWITCH_FLUSH_MMIO(device);

    //
    // Sometimes on RTL simulation we see the priv ring initialization fail.
    // Retry up to 3 times until this issue is root caused. Bug 1826216.
    //
    for (i = 0; !enumerated && (i < 3); i++)
    {
        value = DRF_DEF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _ENUMERATE_AND_START_RING);
        retval = lwswitch_ring_master_cmd_sv10(device, value);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: PRIV ring enumeration failed\n",
                __FUNCTION__);
            continue;
        }

        value = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_START_RESULTS);
        if (!FLD_TEST_DRF(_PPRIV_MASTER, _RING_START_RESULTS, _CONNECTIVITY, _PASS, value))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: PRIV ring connectivity failed\n",
                __FUNCTION__);
            continue;
        }

        value = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_INTERRUPT_STATUS0);
        if (value)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: LW_PPRIV_MASTER_RING_INTERRUPT_STATUS0 = %x\n",
                __FUNCTION__, value);

            if ((!FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                    _RING_START_CONN_FAULT, 0, value)) ||
                (!FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                    _DISCONNECT_FAULT, 0, value))      ||
                (!FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                    _OVERFLOW_FAULT, 0, value)))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: PRIV ring error interrupt\n",
                    __FUNCTION__);
            }

            (void)lwswitch_ring_master_cmd_sv10(device,
                    DRF_DEF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _ACK_INTERRUPT));

            continue;
        }

        enumerated = LW_TRUE;
    }

    if (!enumerated)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Cannot enumerate PRIV ring!\n",
            __FUNCTION__);
        retval = -LWL_INITIALIZATION_TOTAL_FAILURE;
        goto lwswitch_initialize_device_state_exit;
    }

    chip_device->overrides.WAR_Bug_200241882_AFS_interrupt_bits = LW_TRUE;

    LWSWITCH_PRINT(device, SETUP,
        "%s: Enabled links: 0x%x\n",
        __FUNCTION__,
        device->regkeys.link_enable_mask & ((1 << LWSWITCH_NUM_LINKS_SV10) - 1));

    retval = lwswitch_device_discovery_sv10(device, LW_PTOP_DISCOVERY_TABLE(0));
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Engine discovery failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    lwswitch_filter_discovery_sv10(device);

    retval = lwswitch_process_discovery_sv10(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Discovery processing failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    // Initialize PMGR to setup I2C before examining EEPROM
    lwswitch_init_pmgr_sv10(device);
    //
    // Ideally checking if an attached EEPROM contains configuration info
    // happens before probing the PMGR I2C bus, but the detection scheme
    // requires probing the I2C bus to identify the board to set the I2C
    // voltage correctly so that we can then read the EEPROM.
    //
    lwswitch_read_rom_tables(device, &device->firmware);
    _lwswitch_process_firmware_info_sv10(device, &device->firmware);

    lwswitch_init_pmgr_devices_sv10(device);

    retval = lwswitch_init_pll_config(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: failed\n", __FUNCTION__);
        retval = -LWL_INITIALIZATION_TOTAL_FAILURE;
        goto lwswitch_initialize_device_state_exit;
    }

    //
    // PLL init should be done *first* before other hardware init
    //
    retval = lwswitch_init_pll_sv10(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: PLL init failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    retval = lwswitch_init_thermal(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Thermal init failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    chip_device->latency_stats = lwswitch_os_malloc(sizeof(LWSWITCH_LATENCY_STATS_SV10));

    //
    // Now that software knows the devices and addresses, it must take all
    // the wrapper modules out of reset.  It does this by writing to the
    // PMC module enable registers.
    //
    _lwswitch_init_mc_enable_sv10(device);

    retval = _lwswitch_construct_link_info(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Link info construction failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    _lwswitch_init_saw_reset(device);

    // Interrupt initialization needs a higher PRI timeout
    if (IS_EMULATION(device) || IS_RTLSIM(device))
    {
        LWSWITCH_REG_WR32(device, _PTIMER, _PRI_TIMEOUT,
            DRF_NUM(_PTIMER, _PRI_TIMEOUT, _PERIOD, 0x200) |
            DRF_DEF(_PTIMER, _PRI_TIMEOUT, _EN, _ENABLED));
    }

    _lwswitch_init_debug_reset(device);

    _lwswitch_init_warm_reset(device);

    _lwswitch_init_saw(device);

    _lwswitch_init_swx(device);

    _lwswitch_init_npg_multicast(device);

    _lwswitch_reset_nport_debug_state_sv10(device);

    lwswitch_init_hw_counter_sv10(device);

    _lwswitch_init_lwlipt(device);

    retval = lwswitch_init_minion_sv10(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Init MINIONs failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    _lwswitch_init_flush_state(device);

    _lwswitch_init_tag_state(device);

    _lwswitch_init_ingress_next_hops(device);

    _lwswitch_init_ingress_control(device);

    _lwswitch_init_ganged_link_routing(device);

    _lwswitch_init_cmd_routing(device);

    _lwswitch_init_route_control(device);

    _lwswitch_init_egress_control(device);

    _lwswitch_init_portstat_counters(device);

    lwswitch_init_clock_gating(device);

    lwswitch_initialize_interrupt_tree(device);

    return LWL_SUCCESS;

lwswitch_initialize_device_state_exit:
    lwswitch_destroy_device_state(device);

    return retval;
}

/*
 * @Brief : Destroys an LwSwitch hardware state
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
static void
lwswitch_destroy_device_state_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (chip_device != NULL)
    {
        lwswitch_destroy_link_info_sv10(device);

        if ((chip_device->latency_stats) != NULL)
        {
            lwswitch_os_free(chip_device->latency_stats);
        }

        lwswitch_free_chipdevice(device);
    }

    lwswitch_i2c_destroy(device);
}

static void
_lwswitch_set_lwlink_caps
(
    LwU32 *pCaps
)
{
    LwU8 tempCaps[LWSWITCH_LWLINK_CAPS_TBL_SIZE];

    lwswitch_os_memset(tempCaps, 0, sizeof(tempCaps));

    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _VALID);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _SUPPORTED);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _P2P_SUPPORTED);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _P2P_ATOMICS);

    // Assume IBM P9 for PPC -- TODO Xavier support.
#if defined(LWCPU_PPC64LE)
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _SYSMEM_ACCESS);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _SYSMEM_ATOMICS);
#endif

    lwswitch_os_memcpy(pCaps, tempCaps, sizeof(tempCaps));
}

static LwlStatus
lwswitch_ctrl_get_lwlink_status_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret
)
{
    LwlStatus retval = LWL_SUCCESS;
    lwlink_link *link;
    LwU8 i;
    LwU32 data;
    lwlink_conn_info conn_info = {0};
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    ret->enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, i, ret->enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_NUM_LINKS_SV10);

        link = lwswitch_get_link(device, i);

        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        //
        // Call the core library to get the remote end information. On the first
        // invocation this will also trigger link training, if link-training is
        // not externally managed by FM. Therefore it is necessary that this be
        // before link status on the link is populated since this call will
        // actually change link state.
        //
        if (device->regkeys.external_fabric_mgmt)
        {
            lwlink_lib_get_remote_conn_info(link, &conn_info);
        }
        else
        {
            lwlink_lib_discover_and_get_remote_conn_info(link, &conn_info,
                                                         LWLINK_STATE_CHANGE_SYNC);
        }

        // Set LWLINK per-link caps
        _lwswitch_set_lwlink_caps(&ret->linkInfo[i].capsTbl);

        //TODO figure out what phyType is. GPU is LWHS.
        ret->linkInfo[i].phyType = LWSWITCH_LWLINK_STATUS_PHY_LWHS;
        ret->linkInfo[i].subLinkWidth = 8;

        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_STATE);
        ret->linkInfo[i].linkState = DRF_VAL(_PLWL, _LINK_STATE, _STATE, data);

        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _SLSM_STATUS_TX);
        ret->linkInfo[i].txSublinkStatus =
            DRF_VAL(_PLWL_SL0, _SLSM_STATUS_TX, _PRIMARY_STATE, data);

        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _SLSM_STATUS_RX);
        ret->linkInfo[i].rxSublinkStatus =
            DRF_VAL(_PLWL_SL1, _SLSM_STATUS_RX, _PRIMARY_STATE, data);

        ret->linkInfo[i].bLaneReversal = _lwswitch_link_lane_reversed(device, i);

        ret->linkInfo[i].lwlinkVersion = LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_2_0;
        ret->linkInfo[i].nciVersion = LWSWITCH_LWLINK_STATUS_NCI_VERSION_2_0;
        ret->linkInfo[i].phyVersion = LWSWITCH_LWLINK_STATUS_LWHS_VERSION_1_0;

        if (conn_info.bConnected)
        {
            ret->linkInfo[i].connected = LWSWITCH_LWLINK_STATUS_CONNECTED_TRUE;
            ret->linkInfo[i].remoteDeviceLinkNumber = (LwU8)conn_info.linkNumber;

            ret->linkInfo[i].remoteDeviceInfo.domain = conn_info.domain;
            ret->linkInfo[i].remoteDeviceInfo.bus = conn_info.bus;
            ret->linkInfo[i].remoteDeviceInfo.device = conn_info.device;
            ret->linkInfo[i].remoteDeviceInfo.function = conn_info.function;
            ret->linkInfo[i].remoteDeviceInfo.pciDeviceId = conn_info.pciDeviceId;
            ret->linkInfo[i].remoteDeviceInfo.deviceType = conn_info.deviceType;

            if (0 != conn_info.pciDeviceId)
            {
                ret->linkInfo[i].remoteDeviceInfo.deviceIdFlags =
                    FLD_SET_DRF(SWITCH_LWLINK, _DEVICE_INFO, _DEVICE_ID_FLAGS,
                         _PCI, ret->linkInfo[i].remoteDeviceInfo.deviceIdFlags);
            }

            // Does not use loopback
            ret->linkInfo[i].loopProperty =
                LWSWITCH_LWLINK_STATUS_LOOP_PROPERTY_NONE;
        }
        else
        {
            ret->linkInfo[i].connected =
                LWSWITCH_LWLINK_STATUS_CONNECTED_FALSE;
            ret->linkInfo[i].remoteDeviceInfo.deviceType =
                LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_NONE;
        }

        // Set the device information for the local end of the link
        ret->linkInfo[i].localDeviceInfo.domain = device->lwlink_device->pciInfo.domain;
        ret->linkInfo[i].localDeviceInfo.bus = device->lwlink_device->pciInfo.bus;
        ret->linkInfo[i].localDeviceInfo.device = device->lwlink_device->pciInfo.device;
        ret->linkInfo[i].localDeviceInfo.function = device->lwlink_device->pciInfo.function;
        ret->linkInfo[i].localDeviceInfo.pciDeviceId = 0xdeadbeef; // TODO
        ret->linkInfo[i].localDeviceLinkNumber = i;
        ret->linkInfo[i].localDeviceInfo.deviceType =
            LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_SWITCH;

        // Clock data
        ret->linkInfo[i].lwlinkLinkClockKHz = chip_device->link[i].link_clock_khz;

        // All GV100 based designs use 156.25 MHz refclk
        ret->linkInfo[i].lwlinkRefClkSpeedKHz = 156250;

        ret->linkInfo[i].lwlinkCommonClockSpeedKHz =
            ret->linkInfo[i].lwlinkLinkClockKHz / 16;

        ret->linkInfo[i].lwlinkLinkClockMhz =
            ret->linkInfo[i].lwlinkLinkClockKHz / 1000;
        ret->linkInfo[i].lwlinkRefClkSpeedMhz =
            ret->linkInfo[i].lwlinkRefClkSpeedKHz / 1000;
        ret->linkInfo[i].lwlinkCommonClockSpeedMhz =
             ret->linkInfo[i].lwlinkCommonClockSpeedKHz / 1000;

        ret->linkInfo[i].lwlinkRefClkType = LWSWITCH_LWLINK_REFCLK_TYPE_LWHS;
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return retval;
}

static LwlStatus
lwswitch_ctrl_get_counters_sv10
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_GET_COUNTERS_PARAMS *ret
)
{
    lwlink_link *link;
    LwU8   i;
    LwU32  counterMask;
    LwU32  data;
    LwU32  val;
    LwU64  tx0TlCount;
    LwU64  tx1TlCount;
    LwU64  rx0TlCount;
    LwU64  rx1TlCount;
    LwU32  laneId;
    LwBool bLaneReversed;

    ct_assert(LWSWITCH_NUM_LANES_SV10 <= LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE);

    link = lwswitch_get_link(device, ret->linkId);
    if ((link == NULL) ||
        !LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        return -LWL_BAD_ARGS;
    }

    counterMask = ret->counterMask;

    // Common usage allows one of these to stand for all of them
    if (counterMask & (LWSWITCH_LWLINK_COUNTER_TL_TX0 |
                       LWSWITCH_LWLINK_COUNTER_TL_TX1 |
                       LWSWITCH_LWLINK_COUNTER_TL_RX0 |
                       LWSWITCH_LWLINK_COUNTER_TL_RX1))
    {
        tx0TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR0_LO),
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR0_HI));
        if (LWBIT64(63) & tx0TlCount)
        {
            ret->bTx0TlCounterOverflow = LW_TRUE;
            tx0TlCount &= ~(LWBIT64(63));
        }

        tx1TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR1_LO),
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR1_HI));
        if (LWBIT64(63) & tx1TlCount)
        {
            ret->bTx1TlCounterOverflow = LW_TRUE;
            tx1TlCount &= ~(LWBIT64(63));
        }

        rx0TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR0_LO),
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR0_HI));
        if (LWBIT64(63) & rx0TlCount)
        {
            ret->bRx0TlCounterOverflow = LW_TRUE;
            rx0TlCount &= ~(LWBIT64(63));
        }

        rx1TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR1_LO),
            LWSWITCH_LINK_OFFSET_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR1_HI));
        if (LWBIT64(63) & rx1TlCount)
        {
            ret->bRx1TlCounterOverflow = LW_TRUE;
            rx1TlCount &= ~(LWBIT64(63));
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_TX0)] = tx0TlCount;
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_TX1)] = tx1TlCount;
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_RX0)] = rx0TlCount;
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_RX1)] = rx1TlCount;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)
    {
        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _ERROR_COUNT1);

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)]
            = DRF_VAL(_PLWL_SL1, _ERROR_COUNT1, _FLIT_CRC_ERRORS, data);
    }

    data = 0x0;
    bLaneReversed = _lwswitch_link_lane_reversed(device, link->linkNumber);

    for (laneId = 0; laneId < LWSWITCH_NUM_LANES_SV10; laneId++)
    {
        //
        // HW may reverse the lane ordering or it may be overridden by SW.
        // If so, ilwert the interpretation of the lane CRC errors.
        //
        i = (LwU8)((bLaneReversed) ? (LWSWITCH_NUM_LANES_SV10 - 1) - laneId : laneId);

        if (i < 4)
        {
            data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _ERROR_COUNT2_LANECRC);
        }
        else
        {
            data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _ERROR_COUNT3_LANECRC);
        }

        if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(laneId))
        {
            val = BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(laneId));

            switch (i)
            {
                case 0:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT2_LANECRC, _L0, data);
                    break;
                case 1:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT2_LANECRC, _L1, data);
                    break;
                case 2:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT2_LANECRC, _L2, data);
                    break;
                case 3:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT2_LANECRC, _L3, data);
                    break;
                case 4:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT3_LANECRC, _L4, data);
                    break;
                case 5:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT3_LANECRC, _L5, data);
                    break;
                case 6:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT3_LANECRC, _L6, data);
                    break;
                case 7:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_PLWL_SL1, _ERROR_COUNT3_LANECRC, _L7, data);
                    break;
            }
        }
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)
    {
        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _ERROR_COUNT4);

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)]
            = DRF_VAL(_PLWL_SL0, _ERROR_COUNT4, _REPLAY_EVENTS, data);
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)
    {
        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _ERROR_COUNT1);

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)]
            = DRF_VAL(_PLWL, _ERROR_COUNT1, _RECOVERY_EVENTS, data);
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS)
    {
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS)] = 0;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL)
    {
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL)] = 0;
    }

    return LWL_SUCCESS;
}

static LwU32
_lwswitch_get_info_chip_id
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _CHIP_ID, val));
}

static LwU32
_lwswitch_get_info_revision_major
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _MAJOR_REVISION, val));
}

static LwU32
_lwswitch_get_info_revision_minor
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _MINOR_REVISION, val));
}

static LwU32
_lwswitch_get_info_revision_minor_ext
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _MINOR_EXTENDED_REVISION, val));
}

static LwU32
_lwswitch_get_info_foundry
(
    lwswitch_device *device
)
{
    LwU32 data = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_0);

    return (DRF_VAL(_PSMC, _BOOT_0, _FOUNDRY, data));
}

static LwU32
_lwswitch_get_info_voltage
(
    lwswitch_device *device
)
{
    LwU32 data;
    LwU32 vdd_vid = 0;
    LwU32 vdd_vref = 0;
    LwU32 voltage;

    // E3600 specific
    data = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_INPUT_CNTL_8);
    vdd_vid = DRF_VAL(_PMGR, _GPIO_INPUT_CNTL_8, _READ, data);

    data = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_INPUT_CNTL_13);
    vdd_vref = DRF_VAL(_PMGR, _GPIO_INPUT_CNTL_13, _READ, data);

    if (vdd_vid == 1)
    {
        if (vdd_vref == 1)
        {
            voltage = 800;          // 800mV
        }
        else
        {
            voltage = 1000;          // 1000mV
        }
    }
    else
    {
        if (vdd_vref == 1)
        {
            voltage = 0;            // ?mV
        }
        else
        {
            voltage = 1200;         // 1200mV
        }
    }

    return voltage;
}

/*
 * CTRL_LWSWITCH_GET_INFO
 *
 * Query for miscellaneous information analogous to LW2080_CTRL_GPU_INFO
 * This provides a single API to query for multiple pieces of miscellaneous
 * information via a single call.
 *
 */

static LwlStatus
lwswitch_ctrl_get_info_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_INFO *p
)
{
    LwlStatus retval = LWL_SUCCESS;
    LwU32 i;

    if (p->count > LWSWITCH_GET_INFO_COUNT_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid args\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(p->info, 0, sizeof(LwU32)*LWSWITCH_GET_INFO_COUNT_MAX);

    for (i = 0; i < p->count; i++)
    {
        switch (p->index[i])
        {
            case LWSWITCH_GET_INFO_INDEX_ARCH:
                p->info[i] = device->chip_arch;
                break;
            case LWSWITCH_GET_INFO_INDEX_PLATFORM:
                p->info[i] = LWSWITCH_GET_INFO_INDEX_PLATFORM_SILICON;
                break;
            case LWSWITCH_GET_INFO_INDEX_IMPL:
                p->info[i] = device->chip_impl;
                break;
            case LWSWITCH_GET_INFO_INDEX_CHIPID:
                p->info[i] = _lwswitch_get_info_chip_id(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REVISION_MAJOR:
                p->info[i] = _lwswitch_get_info_revision_major(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REVISION_MINOR:
                p->info[i] = _lwswitch_get_info_revision_minor(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REVISION_MINOR_EXT:
                p->info[i] = _lwswitch_get_info_revision_minor_ext(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_FOUNDRY:
                p->info[i] = _lwswitch_get_info_foundry(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_FAB:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_FAB_CODE);
                break;
            case LWSWITCH_GET_INFO_INDEX_LOT_CODE_0:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_LOT_CODE_0);
                break;
            case LWSWITCH_GET_INFO_INDEX_LOT_CODE_1:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_LOT_CODE_1);
                break;
            case LWSWITCH_GET_INFO_INDEX_WAFER:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_WAFER_ID);
                break;
            case LWSWITCH_GET_INFO_INDEX_XCOORD:
                {
                    LwS32 xcoord = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_X_COORDINATE);
                    // X coordinate is two's complement, colwert to signed 32-bit
                    p->info[i] = xcoord -
                        2 * (xcoord & (1<<(DRF_SIZE(LW_FUSE_OPT_X_COORDINATE_DATA) - 1)));
                }
                break;
            case LWSWITCH_GET_INFO_INDEX_YCOORD:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_Y_COORDINATE);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO_REV:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_SPEEDO_REV);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO0:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_SPEEDO0);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO1:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_SPEEDO1);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO2:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_SPEEDO2);
                break;
            case LWSWITCH_GET_INFO_INDEX_IDDQ:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_IDDQ);
                break;
            case LWSWITCH_GET_INFO_INDEX_IDDQ_REV:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_IDDQ_REV);
                break;
            case LWSWITCH_GET_INFO_INDEX_ATE_REV:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_LWST_ATE_REV);
                break;
            case LWSWITCH_GET_INFO_INDEX_VENDOR_CODE:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_VENDOR_CODE);
                break;
            case LWSWITCH_GET_INFO_INDEX_OPS_RESERVED:
                p->info[i] = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_OPS_RESERVED);
                break;
            case LWSWITCH_GET_INFO_INDEX_DEVICE_ID:
                p->info[i] = device->lwlink_device->pciInfo.pciDeviceId;
                break;
            case LWSWITCH_GET_INFO_INDEX_NUM_PORTS:
                p->info[i] = lwswitch_get_num_links_sv10(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0:
                p->info[i] = LwU64_LO32(lwswitch_get_enabled_link_mask(device));
                break;
            case LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32:
                p->info[i] = LwU64_HI32(lwswitch_get_enabled_link_mask(device));
                break;
            case LWSWITCH_GET_INFO_INDEX_NUM_VCS:
                p->info[i] = _lwswitch_get_num_vcs_sv10(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_TABLE_SIZE:
                p->info[i] = 0;
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTA_TABLE_SIZE:
                p->info[i] = 0;
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTB_TABLE_SIZE:
                p->info[i] = 0;
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_MULTICAST_TABLE_SIZE:
                p->info[i] = 0;
                break;
            case LWSWITCH_GET_INFO_INDEX_ROUTING_ID_TABLE_SIZE:
                p->info[i] = 0;
                break;
            case LWSWITCH_GET_INFO_INDEX_ROUTING_LAN_TABLE_SIZE:
                p->info[i] = 0;
                break;
            case LWSWITCH_GET_INFO_INDEX_FREQ_KHZ:
                p->info[i] = device->switch_pll.freq_khz;
                break;
            case LWSWITCH_GET_INFO_INDEX_VCOFREQ_KHZ:
                p->info[i] = device->switch_pll.vco_freq_khz;
                break;
            case LWSWITCH_GET_INFO_INDEX_VOLTAGE_MVOLT:
                p->info[i] = _lwswitch_get_info_voltage(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID:
                p->info[i] = lwswitch_read_physical_id(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_DOMAIN:
                p->info[i] = device->lwlink_device->pciInfo.domain;
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_BUS:
                p->info[i] = device->lwlink_device->pciInfo.bus;
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_DEVICE:
                p->info[i] = device->lwlink_device->pciInfo.device;
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_FUNCTION:
                p->info[i] = device->lwlink_device->pciInfo.function;
                break;
            case LWSWITCH_GET_INFO_INDEX_INFOROM_LWL_SUPPORTED:
                p->info[i] = LW_FALSE;
                break;
            case LWSWITCH_GET_INFO_INDEX_INFOROM_BBX_SUPPORTED:
                p->info[i] = LW_FALSE;
                break;
            default:
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Undefined LWSWITCH_GET_INFO_INDEX 0x%x\n",
                    __FUNCTION__,
                    p->index[i]);
                retval = -LWL_BAD_ARGS;
                break;
        }
    }

    return retval;
}

LwlStatus
lwswitch_set_nport_port_config_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_SWITCH_PORT_CONFIG *p
)
{
    switch (p->type)
    {
        case CONNECT_ACCESS_GPU:
        case CONNECT_ACCESS_CPU:
        case CONNECT_ACCESS_SWITCH:
            LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _NPORT, _CTRL,
                DRF_NUM(_NPORT, _CTRL, _TRUNKLINKENB, 0x0));
            break;
        case CONNECT_TRUNK_SWITCH:
            LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _NPORT, _CTRL,
                DRF_NUM(_NPORT, _CTRL, _TRUNKLINKENB, 0x1));
            break;
        default:
            return -LWL_BAD_ARGS;
    }

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_set_switch_port_config_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_SWITCH_PORT_CONFIG *p
)
{
    lwlink_link *link;
    LwU32 temp;
    LwU32 val;
    LwlStatus status;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, p->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    if (p->enableVC1 && (p->type != CONNECT_TRUNK_SWITCH))
    {
        return -LWL_BAD_ARGS;
    }

    status = lwswitch_set_nport_port_config(device, p);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if (p->requesterLinkID > DRF_MASK(LW_INGRESS_REQLINKID_REQLINKID))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid requester link ID 0x%x\n",
            __FUNCTION__, p->requesterLinkID);
        return -LWL_BAD_ARGS;
    }

    LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _REQLINKID,
        DRF_NUM(_INGRESS, _REQLINKID, _REQLINKID, p->requesterLinkID));

    link = lwswitch_get_link(device, (LwU8)p->portNum);
    if (link == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid link\n",
            __FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    //
    // If ac_coupled_mask is configured during lwswitch_create_link,
    // give preference to it.
    //
    if (device->regkeys.ac_coupled_mask ||
        device->firmware.lwlink.link_ac_coupled_mask)
    {
        if (link->ac_coupled != p->acCoupled)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: port[%d]: Unsupported AC coupled change (%s)\n",
                __FUNCTION__, p->portNum, p->acCoupled ? "AC" : "DC");
            return -LWL_BAD_ARGS;
        }
    }

    link->ac_coupled = p->acCoupled;

    // If _BUFFER_RDY is asserted, credits are locked.
    val = LWSWITCH_LINK_RD32_SV10(device, p->portNum, NPORT, _NPORT, _CTRL_BUFFER_READY);
    if (FLD_TEST_DRF(_NPORT, _CTRL_BUFFER_READY, _BUFFERRDY, _ENABLE, val))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: port[%d]: BUFFERRDY already enabled.\n",
            __FUNCTION__, p->portNum);
        return LWL_SUCCESS;
    }

    // If enabling VC1, assign half the slots to VC0 and half to VC1
    temp = LW_LWLTLC_TX_CTRL_LINK_CONFIG_STICKYPARTITION_MAX;
    if (p->enableVC1)
    {
        temp /= 2;
    }
    val = LWSWITCH_LINK_RD32_SV10(device, p->portNum, LWLTLC, _LWLTLC_TX, _CTRL_LINK_CONFIG);
    val = FLD_SET_DRF_NUM(_LWLTLC_TX, _CTRL_LINK_CONFIG, _STICKYPARTITION, temp, val);
    LWSWITCH_LINK_WR32_SV10(device, p->portNum, LWLTLC, _LWLTLC_TX, _CTRL_LINK_CONFIG, val);

    if (p->enableVC1)
    {
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_0,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_0, _VCMAX, 0xA0));        // 160
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_1,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_1, _VCMAX, 0xA1));        // 1
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_2,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_2, _VCMAX, 0xA2));        // 1
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_3,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_3, _VCMAX, 0xA3));        // 1
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_4,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_4, _VCMAX, 0xA4));        // 1
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_5,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_5, _VCMAX, 0x142));       // 158
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_6,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_6, _VCMAX, 0x1E0));       // 158
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_7,
            DRF_NUM(_ROUTE, _BUFFER_SZ_VC_7, _VCMAX, 0x27E));       // 158

        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_0,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_0, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_0, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_1,
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_1, _VCMAXHEADERINIT, 0x1) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_1, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_2,
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_2, _VCMAXHEADERINIT, 0x1) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_2, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_3,
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_3, _VCMAXHEADERINIT, 0x1) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_3, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_4,
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_4, _VCMAXHEADERINIT, 0x1) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_4, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_5,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_5, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_5, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_6,
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_6, _VCMAXHEADERINIT, 0x49) |
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_6, _VCMAXDATAINIT,   0x55));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_7,
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_7, _VCMAXHEADERINIT, 0x49) |
            DRF_NUM(_INGRESS, _BUFFER_CREDIT_VC_7, _VCMAXDATAINIT,   0x55));
    }
    else
    {
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_0,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_0, _VCMAX, _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_1,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_1, _VCMAX, _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_2,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_2, _VCMAX, _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_3,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_3, _VCMAX, _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_4,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_4, _VCMAX, _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_5,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_5, _VCMAX, _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_6,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_6, _VCMAX, _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _ROUTE, _BUFFER_SZ_VC_7,
            DRF_DEF(_ROUTE, _BUFFER_SZ_VC_7, _VCMAX, _INIT));

        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_0,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_0, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_0, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_1,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_1, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_1, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_2,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_2, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_2, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_3,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_3, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_3, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_4,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_4, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_4, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_5,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_5, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_5, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_6,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_6, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_6, _VCMAXDATAINIT,   _INIT));
        LWSWITCH_LINK_WR32_SV10(device, p->portNum, NPORT, _INGRESS, _BUFFER_CREDIT_VC_7,
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_7, _VCMAXHEADERINIT, _INIT) |
            DRF_DEF(_INGRESS, _BUFFER_CREDIT_VC_7, _VCMAXDATAINIT,   _INIT));
    }

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_get_ingress_request_table_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *params
)
{
    LwU32 table_idx, sparse_count;
    LWSWITCH_INGRESS_REQUEST_IDX_ENTRY *sparse_entries;
    INGRESS_REQUEST_RESPONSE_ENTRY_SV10 lwr_entry;

    /* 8K req/rsp table sizes */
    ct_assert(INGRESS_MAP_TABLE_SIZE ==
              (1 << DRF_SIZE(LW_INGRESS_REQRSPMAPADDR_TABLE_INDEX)));

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, params->portNum) ||
        params->firstIndex >= INGRESS_MAP_TABLE_SIZE)
    {
        return -LWL_BAD_ARGS;
    }

    sparse_entries = params->entries;
    table_idx = params->firstIndex;
    sparse_count = 0;

    /* set table offset */
    LWSWITCH_LINK_WR32_SV10(
        device, params->portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_INDEX, table_idx) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_SELECT,
                    LW_INGRESS_REQRSPMAPADDR_TABLE_SELECT_REQUEST) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    while (sparse_count < LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX &&
           table_idx < INGRESS_MAP_TABLE_SIZE)
    {
        /* read one entry */
        lwr_entry.ingress_reqresmapdata0 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA0);
        lwr_entry.ingress_reqresmapdata1 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA1);
        lwr_entry.ingress_reqresmapdata2 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA2);
        lwr_entry.ingress_reqresmapdata3 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA3);

        /* add to sparse list if nonzero */
        if (lwr_entry.ingress_reqresmapdata0 || lwr_entry.ingress_reqresmapdata1 ||
                lwr_entry.ingress_reqresmapdata2 || lwr_entry.ingress_reqresmapdata3)
        {
            sparse_entries[sparse_count].entry.mappedAddress = DRF_VAL(_INGRESS,
                _REQRSPMAPDATA0, _MAPADDR, lwr_entry.ingress_reqresmapdata0);
            sparse_entries[sparse_count].entry.routePolicy =  DRF_VAL(_INGRESS,
                _REQRSPMAPDATA0, _RPOLICY, lwr_entry.ingress_reqresmapdata0);
            sparse_entries[sparse_count].entry.entryValid = DRF_VAL(_INGRESS,
                _REQRSPMAPDATA0, _ACLVALID, lwr_entry.ingress_reqresmapdata0);

            sparse_entries[sparse_count].entry.vcModeValid7_0 =
                lwr_entry.ingress_reqresmapdata1;
            sparse_entries[sparse_count].entry.vcModeValid15_8 =
                lwr_entry.ingress_reqresmapdata2;
            sparse_entries[sparse_count].entry.vcModeValid17_16 =
                lwr_entry.ingress_reqresmapdata3;

            sparse_entries[sparse_count].idx = table_idx;

            sparse_count++;
        }

        table_idx++;
    }

    params->nextIndex = table_idx;
    params->numEntries = sparse_count;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_set_ingress_request_table_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_REQUEST_TABLE *p
)
{
    LwU32 i;
    LWSWITCH_INGRESS_REQUEST_ENTRY *entries;
    INGRESS_REQUEST_RESPONSE_ENTRY_SV10 *table = NULL;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, p->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    // 8K req/rsp table sizes
    LWSWITCH_ASSERT(INGRESS_MAP_TABLE_SIZE ==
        (1 << DRF_SIZE(LW_INGRESS_REQRSPMAPADDR_TABLE_INDEX)));

    LWSWITCH_ASSERT(NULL != chip_device->link[p->portNum].ingress_req_table);
    if ((p->firstIndex >= INGRESS_MAP_TABLE_SIZE) ||
        (p->numEntries > LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > INGRESS_MAP_TABLE_SIZE))
    {
        return -LWL_BAD_ARGS;
    }

    table = &chip_device->link[p->portNum].ingress_req_table[p->firstIndex];
    entries = p->entries;

    for (i = 0; i < p->numEntries; i++)
    {
        table[i].ingress_reqresmapdata0 =
            DRF_NUM(_INGRESS, _REQRSPMAPDATA0, _MAPADDR, entries[i].mappedAddress) |
            DRF_NUM(_INGRESS, _REQRSPMAPDATA0, _RPOLICY, entries[i].routePolicy)   |
            DRF_NUM(_INGRESS, _REQRSPMAPDATA0, _ACLVALID, entries[i].entryValid);

        table[i].ingress_reqresmapdata1 = entries[i].vcModeValid7_0;

        table[i].ingress_reqresmapdata2 = entries[i].vcModeValid15_8;

        table[i].ingress_reqresmapdata3 = entries[i].vcModeValid17_16;
    }

    lwswitch_set_ingress_table_sv10(device, p->portNum, p->firstIndex,
        p->numEntries, LW_INGRESS_REQRSPMAPADDR_TABLE_SELECT_REQUEST);

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_set_ingress_request_valid_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_REQUEST_VALID *p
)
{
    LwU32 i;
    INGRESS_REQUEST_RESPONSE_ENTRY_SV10 *table = NULL;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, p->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    // 8K req/rsp table sizes
    LWSWITCH_ASSERT(INGRESS_MAP_TABLE_SIZE ==
         (1 << DRF_SIZE(LW_INGRESS_REQRSPMAPADDR_TABLE_INDEX)));

    LWSWITCH_ASSERT(NULL != chip_device->link[p->portNum].ingress_req_table);

    if ((p->firstIndex >= INGRESS_MAP_TABLE_SIZE) ||
        (p->numEntries > LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > INGRESS_MAP_TABLE_SIZE))
    {
        return -LWL_BAD_ARGS;
    }

    table = &chip_device->link[p->portNum].ingress_req_table[p->firstIndex];

    //
    // Iterate through the cached table, setting the valid bit as directed and
    // writing the corresponding request table entry in HW.  Note that we don't
    // mark the response table similarly.  It shouldn't need to be ilwalidated.
    //
    for (i = 0; i < p->numEntries; i++)
    {
        table[i].ingress_reqresmapdata0 =
            FLD_SET_DRF_NUM(_INGRESS, _REQRSPMAPDATA0, _ACLVALID,
                p->entryValid[i], table[i].ingress_reqresmapdata0);
    }

    lwswitch_set_ingress_table_sv10(device, p->portNum, p->firstIndex,
        p->numEntries, LW_INGRESS_REQRSPMAPADDR_TABLE_SELECT_REQUEST);

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_get_ingress_response_table_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *params
)
{
    LwU32 table_idx, sparse_count;
    LWSWITCH_INGRESS_RESPONSE_IDX_ENTRY *sparse_entries;
    INGRESS_REQUEST_RESPONSE_ENTRY_SV10 lwr_entry;

    /* 8K req/rsp table sizes */
    ct_assert(INGRESS_MAP_TABLE_SIZE ==
              (1 << DRF_SIZE(LW_INGRESS_REQRSPMAPADDR_TABLE_INDEX)));

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, params->portNum) ||
        params->firstIndex >= INGRESS_MAP_TABLE_SIZE)
    {
        return -LWL_BAD_ARGS;
    }

    sparse_entries = params->entries;
    table_idx = params->firstIndex;
    sparse_count = 0;

    /* set table offset */
    LWSWITCH_LINK_WR32_SV10(
        device, params->portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_INDEX, table_idx) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _TABLE_SELECT,
                    LW_INGRESS_REQRSPMAPADDR_TABLE_SELECT_RESPONSE) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    while (sparse_count < LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX &&
           table_idx < INGRESS_MAP_TABLE_SIZE)
    {
        /* read one entry */
        lwr_entry.ingress_reqresmapdata0 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA0);
        lwr_entry.ingress_reqresmapdata1 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA1);
        lwr_entry.ingress_reqresmapdata2 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA2);
        lwr_entry.ingress_reqresmapdata3 = LWSWITCH_LINK_RD32_SV10(device,
            params->portNum, NPORT, _INGRESS, _REQRSPMAPDATA3);

        /* add to sparse list if nonzero */
        if (lwr_entry.ingress_reqresmapdata0 || lwr_entry.ingress_reqresmapdata1 ||
            lwr_entry.ingress_reqresmapdata2 || lwr_entry.ingress_reqresmapdata3)
        {
            sparse_entries[sparse_count].entry.routePolicy =  DRF_VAL(_INGRESS,
                _REQRSPMAPDATA0, _RPOLICY, lwr_entry.ingress_reqresmapdata0);
            sparse_entries[sparse_count].entry.entryValid = DRF_VAL(_INGRESS,
                _REQRSPMAPDATA0, _ACLVALID, lwr_entry.ingress_reqresmapdata0);

            sparse_entries[sparse_count].entry.vcModeValid7_0 =
                lwr_entry.ingress_reqresmapdata1;
            sparse_entries[sparse_count].entry.vcModeValid15_8 =
                lwr_entry.ingress_reqresmapdata2;
            sparse_entries[sparse_count].entry.vcModeValid17_16 =
                lwr_entry.ingress_reqresmapdata3;

            sparse_entries[sparse_count].idx = table_idx;

            sparse_count++;
        }

        table_idx++;
    }

    params->nextIndex = table_idx;
    params->numEntries = sparse_count;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_set_ingress_response_table_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE *p
)
{
    LwU32 i;
    LWSWITCH_INGRESS_RESPONSE_ENTRY *entries;
    INGRESS_REQUEST_RESPONSE_ENTRY_SV10 *table = NULL;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, p->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    // 8K req/rsp table sizes
    LWSWITCH_ASSERT(INGRESS_MAP_TABLE_SIZE ==
         (1 << DRF_SIZE(LW_INGRESS_REQRSPMAPADDR_TABLE_INDEX)));

    if ((p->firstIndex >= INGRESS_MAP_TABLE_SIZE) ||
        (p->numEntries > LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > INGRESS_MAP_TABLE_SIZE))
    {
        return -LWL_BAD_ARGS;
    }

    table = &chip_device->link[p->portNum].ingress_res_table[p->firstIndex];
    entries = p->entries;

    for (i = 0; i < p->numEntries; i++)
    {
        table[i].ingress_reqresmapdata0 =
            DRF_NUM(_INGRESS, _REQRSPMAPDATA0, _MAPADDR, 0)                      |
            DRF_NUM(_INGRESS, _REQRSPMAPDATA0, _RPOLICY, entries[i].routePolicy) |
            DRF_NUM(_INGRESS, _REQRSPMAPDATA0, _ACLVALID, entries[i].entryValid);

        table[i].ingress_reqresmapdata1 = entries[i].vcModeValid7_0;

        table[i].ingress_reqresmapdata2 = entries[i].vcModeValid15_8;

        table[i].ingress_reqresmapdata3 = entries[i].vcModeValid17_16;
    }

    lwswitch_set_ingress_table_sv10(device, p->portNum, p->firstIndex,
        p->numEntries, LW_INGRESS_REQRSPMAPADDR_TABLE_SELECT_RESPONSE);

    return LWL_SUCCESS;
}

/*
 * CTRL_LWSWITCH_SET_GANGED_LINK_TABLE
 *
 * Allows a client to overwrite the default ganged link distribution table
 * which uniformly distributes traffic over the set of links in a gang.
 * Refer to switch IAS 7.5.18.2.4 and 7.5.19.5.2.2
 * Also hw\doc\gpu\volta\lwswitch\design\IAS\lwswitch_pri_addr.xlsm "route" tab
 * Also _lwswitch_init_ganged_link_routing
 * For more details regarding the function.
 */
static LwlStatus
lwswitch_ctrl_set_ganged_link_table_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_GANGED_LINK_TABLE *p
)
{
    LwU32 link;
    LwU32 i;
    ROUTE_GANG_ENTRY_SV10 *table;
    LwlStatus retval = LWL_SUCCESS;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    FOR_EACH_INDEX_IN_MASK(32, link, p->link_mask)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, link))
        {
            return -LWL_BAD_ARGS;
        }

        table = chip_device->link[link].ganged_link_table;

        for (i = 0; i < ROUTE_GANG_TABLE_SIZE; i++)
        {
            table[i].regtabledata0 = p->entries[i];
        }

        lwswitch_set_ganged_link_table_sv10(device, link, 0, ROUTE_GANG_TABLE_SIZE);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return retval;
}

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY
 */
static LwlStatus
lwswitch_ctrl_set_remap_policy_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_SET_REMAP_POLICY should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_GET_REMAP_POLICY
 */
static LwlStatus
lwswitch_ctrl_get_remap_policy_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_REMAP_POLICY_PARAMS *params
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_GET_REMAP_POLICY should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY_VALID
 */
static LwlStatus
lwswitch_ctrl_set_remap_policy_valid_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY_VALID *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_SET_REMAP_POLICY_VALID should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
lwswitch_ctrl_get_fom_values_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_FOM_VALUES_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_SET_ROUTING_ID
 */
static LwlStatus
lwswitch_ctrl_set_routing_id_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_ID *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_SET_ROUTING_ID should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_GET_ROUTING_ID
 */
static LwlStatus
lwswitch_ctrl_get_routing_id_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_ROUTING_ID_PARAMS *params
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_GET_ROUTING_ID should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_SET_ROUTING_ID_VALID
 */
static LwlStatus
lwswitch_ctrl_set_routing_id_valid_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_ID_VALID *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_SET_ROUTING_ID_VALID should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_SET_ROUTING_LAN
 */
static LwlStatus
lwswitch_ctrl_set_routing_lan_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_LAN *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_SET_ROUTING_LAN should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_GET_ROUTING_LAN
 */
static LwlStatus
lwswitch_ctrl_get_routing_lan_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_ROUTING_LAN_PARAMS *params
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_GET_ROUTING_LAN should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_SET_ROUTING_LAN_VALID
 */
static LwlStatus
lwswitch_ctrl_set_routing_lan_valid_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_LAN_VALID *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "_SET_ROUTING_LAN_VALID should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
lwswitch_ctrl_get_internal_latency_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_INTERNAL_LATENCY *p
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 idx_nport;
    LwU32 vc_selector = p->vc_selector;
    LwlStatus retval = LWL_SUCCESS;

    lwswitch_os_memset(p, 0, sizeof(*p));

    p->vc_selector = vc_selector;

    if (vc_selector >= LWSWITCH_NUM_VCS_SV10)
    {
        // Unrecognized VC
        return -LWL_BAD_ARGS;
    }

    for (idx_nport=0; idx_nport < LWSWITCH_NUM_LINKS_SV10; idx_nport++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, idx_nport))
        {
            continue;
        }

        p->egressHistogram[idx_nport].low =
            chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].low;
        p->egressHistogram[idx_nport].medium =
            chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].medium;
        p->egressHistogram[idx_nport].high =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].high;
        p->egressHistogram[idx_nport].panic =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].panic;
    }

    p->elapsed_time_msec =
      (chip_device->latency_stats->latency[vc_selector].last_read_time_nsec -
       chip_device->latency_stats->latency[vc_selector].start_time_nsec)/1000000ULL;

    chip_device->latency_stats->latency[vc_selector].start_time_nsec =
        chip_device->latency_stats->latency[vc_selector].last_read_time_nsec;

    chip_device->latency_stats->latency[vc_selector].count = 0;

    // Clear aclwm_latency[]
    for (idx_nport=0; idx_nport < LWSWITCH_NUM_LINKS_SV10; idx_nport++)
    {
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].low = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].medium = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].high = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].panic = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].count = 0;
    }

    return retval;
}

static LwlStatus
lwswitch_ctrl_set_latency_bins_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_LATENCY_BINS *p
)
{
    LwU32 switchpll_hz = device->switch_pll.freq_khz * 1000;
    LwU32 idx_channel;
    const LwU32 min_threshold = 10;       // Must be > zero to avoid div by zero
    const LwU32 max_threshold = 10000;

    // Quick input validation and ns to register value colwersion
    for (idx_channel = 0; idx_channel < LWSWITCH_MAX_VCS; idx_channel++)
    {
        if ((p->bin[idx_channel].lowThreshold > max_threshold)                    ||
            (p->bin[idx_channel].lowThreshold < min_threshold)                    ||
            (p->bin[idx_channel].medThreshold > max_threshold)                    ||
            (p->bin[idx_channel].medThreshold < min_threshold)                    ||
            (p->bin[idx_channel].hiThreshold  > max_threshold)                    ||
            (p->bin[idx_channel].hiThreshold  < min_threshold)                    ||
            (p->bin[idx_channel].lowThreshold > p->bin[idx_channel].medThreshold) ||
            (p->bin[idx_channel].medThreshold > p->bin[idx_channel].hiThreshold))
        {
            return -LWL_BAD_ARGS;
        }

        p->bin[idx_channel].lowThreshold =
            switchpll_hz / (1000000000 / p->bin[idx_channel].lowThreshold);
        p->bin[idx_channel].medThreshold =
            switchpll_hz / (1000000000 / p->bin[idx_channel].medThreshold);
        p->bin[idx_channel].hiThreshold =
            switchpll_hz / (1000000000 / p->bin[idx_channel].hiThreshold);
    }

    for (idx_channel = 0; idx_channel < LWSWITCH_MAX_VCS; idx_channel++)
    {
        LWSWITCH_BCAST_OFF_WR32_SV10(device, NPG, NPORT, _MULTICAST,
            LW_NPORT_PORTSTAT_SV10(_LIMIT, _LOW, idx_channel),    p->bin[idx_channel].lowThreshold);
        LWSWITCH_BCAST_OFF_WR32_SV10(device, NPG, NPORT, _MULTICAST,
            LW_NPORT_PORTSTAT_SV10(_LIMIT, _MEDIUM, idx_channel), p->bin[idx_channel].medThreshold);
        LWSWITCH_BCAST_OFF_WR32_SV10(device, NPG, NPORT, _MULTICAST,
            LW_NPORT_PORTSTAT_SV10(_LIMIT, _HIGH, idx_channel),   p->bin[idx_channel].hiThreshold);
    }

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_deassert_link_reset_sv10
(
    lwswitch_device *device,
    lwlink_link     *link
)
{
    // NOP
    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
// Ported from lwlinkInjectTlcRxFatalError_GV100
static LwlStatus
_lwswitch_inject_link_error_fatal
(
    lwswitch_device *device,
    LwU32            link
)
{
    LwU32 tempRegVal = 0;

    // Choice of error is somewhat arbitrary
    tempRegVal =
        FLD_SET_DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXRAMDATAPARITYERR, 0x1, 0);

    LWSWITCH_LINK_WR32_SV10(device, link, LWLTLC, _LWLTLC_RX, _ERR_INJECT_0, tempRegVal);

    return LWL_SUCCESS;
}

// Ported from lwlinkInjectRcvy_GV100
static LwlStatus
_lwswitch_inject_link_error_rcvy
(
    lwswitch_device *device,
    LwU32            link
)
{
    LwlStatus  retval  = LWL_SUCCESS;
    LwU32 tempRegVal;
    LWSWITCH_TIMEOUT timeout;

    tempRegVal = LWSWITCH_LINK_RD32_SV10(device, link, DLPL, _PLWL, _LINK_STATE);
    tempRegVal = DRF_VAL(_PLWL, _LINK_STATE, _STATE, tempRegVal);
    if (tempRegVal != LW_PLWL_LINK_STATE_STATE_ACTIVE)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Link %d: RCVY_AC can only be injected in ACTIVE state.\n",
            link);
        return -LWL_ERR_ILWALID_STATE;
    }

    tempRegVal = LWSWITCH_LINK_RD32_SV10(device, link, DLPL, _PLWL, _LINK_CHANGE);
    tempRegVal = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _NEWSTATE, _RCVY_AC, tempRegVal);
    tempRegVal = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, tempRegVal);
    tempRegVal = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _ACTION, _LTSSM_CHANGE, tempRegVal);
    LWSWITCH_LINK_WR32_SV10(device, link, DLPL, _PLWL, _LINK_CHANGE, tempRegVal);

    retval = -LWL_IO_ERROR;
    lwswitch_timeout_create(8 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    do
    {
        tempRegVal = LWSWITCH_LINK_RD32_SV10(device, link, DLPL, _PLWL, _LINK_CHANGE);

        if (FLD_TEST_DRF(_PLWL, _LINK_CHANGE, _STATUS, _DONE, tempRegVal))
        {
            retval  = LWL_SUCCESS;
            break;
        }
        else if ((FLD_TEST_DRF(_PLWL, _LINK_CHANGE, _STATUS, _FAULT, tempRegVal)) ||
                 (FLD_TEST_DRF(_PLWL, _LINK_CHANGE, _STATUS, _ABORT, tempRegVal)))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: link = 0x%x,  Fault or Abort\n",
                __FUNCTION__,
                link);
            retval = -LWL_ERR_ILWALID_STATE;
            break;
        }
    } while (!lwswitch_timeout_check(&timeout));

    return retval;
}
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

//
// MODS-only IOCTLS
//

/*
 * REGISTER_READ/_WRITE
 * Provides direct access to the MMIO space for trusted clients like MODS.
 * This API should not be exposed to unselwre clients.
 */

/*
 * _lwswitch_get_engine_base
 * Used by REGISTER_READ/WRITE API.  Looks up an engine based on device/instance
 * and returns the base address in BAR0.
 *
 * register_rw_engine   [in] REGISTER_RW_ENGINE_*
 * instance             [in] physical instance of device
 * bcast                [in] FALSE: find unicast base address
 *                           TRUE:  find broadcast base address
 * base_addr            [out] base address in BAR0 of requested device
 *
 * Returns              LWL_SUCCESS: Device base address successfully found
 *                      else device lookup failed
 */
static LwlStatus
_lwswitch_get_engine_base
(
    lwswitch_device *device,
    LwU32   register_rw_engine,     // REGISTER_RW_ENGINE_*
    LwU32   instance,               // device instance
    LwBool  bcast,
    LwU32   *base_addr
)
{
    LwU32 engine_idx;
    LwU32 base = 0;
    ENGINE_DESCRIPTOR_TYPE_SV10  *engine = NULL;
    ENGINE_DESCRIPTOR_TYPE_SV10  *lwrr_engine;
    LwlStatus retval = LWL_SUCCESS;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    // Find the engine descriptor matching the request
    engine = NULL;

    switch (register_rw_engine)
    {
        case REGISTER_RW_ENGINE_RAW:
            // Special case raw IO
            if ((instance != 0) ||
                (bcast != LW_FALSE))
            {
                retval = -LWL_BAD_ARGS;
            }
        break;

        case REGISTER_RW_ENGINE_CLKS:
        case REGISTER_RW_ENGINE_FUSE:
        case REGISTER_RW_ENGINE_JTAG:
        case REGISTER_RW_ENGINE_PMGR:
        case REGISTER_RW_ENGINE_XP3G:
            //
            // Legacy devices are always single-instance, unicast-only.
            // These manuals are BAR0 offset-based, not IP-based.  Treat them
            // the same as RAW.
            //
            if ((instance != 0) ||
                (bcast != LW_FALSE))
            {
                retval = -LWL_BAD_ARGS;
            }
            register_rw_engine = REGISTER_RW_ENGINE_RAW;
        break;

        case REGISTER_RW_ENGINE_SAW:
            for (engine_idx=0; engine_idx < NUM_SAW_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->engSAW[engine_idx];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_XVE:
            for (engine_idx=0; engine_idx < NUM_XVE_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->engXVE[engine_idx];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_SIOCTRL:
            for (engine_idx=0; engine_idx < NUM_SIOCTRL_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->subengSIOCTRL[engine_idx].subengSIOCTRL[0];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_MINION:
            for (engine_idx=0; engine_idx < NUM_SIOCTRL_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->subengSIOCTRL[engine_idx].subengMINION[0];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLIPT:
            for (engine_idx=0; engine_idx < NUM_SIOCTRL_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->subengSIOCTRL[engine_idx].subengLWLIPT[0];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLTLC:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine = chip_device->link[engine_idx].engLWLTLC;
                if ((lwrr_engine != NULL) &&
                    (lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_DLPL:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine = chip_device->link[engine_idx].engDLPL;
                if ((lwrr_engine != NULL) &&
                    (lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_TX_PERFMON:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine = chip_device->link[engine_idx].engTX_PERFMON;
                if ((lwrr_engine != NULL) &&
                    (lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_RX_PERFMON:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine = chip_device->link[engine_idx].engRX_PERFMON;
                if ((lwrr_engine != NULL) &&
                    (lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPG:
            for (engine_idx=0; engine_idx < NUM_NPG_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->subengNPG[engine_idx].subengNPG[0];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine = chip_device->link[engine_idx].engNPORT;
                if ((lwrr_engine != NULL) &&
                    (lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPG_PERFMON:
            for (engine_idx=0; engine_idx < NUM_NPG_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->subengNPG[engine_idx].subengNPG_PERFMON[0];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT_PERFMON:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine = chip_device->link[engine_idx].engNPORT_PERFMON;
                if ((lwrr_engine != NULL) &&
                    (lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_SWX:
            for (engine_idx=0; engine_idx < NUM_SWX_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->subengSWX[engine_idx].subengSWX[0];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_SWX_PERFMON:
            for (engine_idx=0; engine_idx < NUM_SWX_ENGINE_SV10; engine_idx++)
            {
                lwrr_engine = &chip_device->subengSWX[engine_idx].subengSWX_PERFMON[0];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_AFS:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine =
                     &chip_device->subengSWX[engine_idx / NUM_AFS_INSTANCES_SV10].
                        subengAFS[engine_idx % NUM_AFS_INSTANCES_SV10];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        case REGISTER_RW_ENGINE_AFS_PERFMON:
            for (engine_idx=0; engine_idx < LWSWITCH_NUM_LINKS_SV10; engine_idx++)
            {
                lwrr_engine =
                    &chip_device->subengSWX[engine_idx / NUM_AFS_PERFMON_INSTANCES_SV10].
                        subengAFS_PERFMON[engine_idx % NUM_AFS_PERFMON_INSTANCES_SV10];
                if ((lwrr_engine->valid) &&
                    (lwrr_engine->instance == instance))
                {
                    engine = lwrr_engine;
                    break;
                }
            }
        break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: unknown REGISTER_RW_ENGINE 0x%x\n",
                __FUNCTION__,
                register_rw_engine);
            engine = NULL;
        break;
    }

    if (register_rw_engine == REGISTER_RW_ENGINE_RAW)
    {
        // Raw IO -- client provides full BAR0 offset
        base = 0;
    }
    else
    {
        // Check engine descriptor was found and valid
        if (engine == NULL)
        {
            retval = -LWL_BAD_ARGS;
            LWSWITCH_PRINT(device, ERROR,
                "%s: invalid REGISTER_RW_ENGINE/instance 0x%x(%d)\n",
                __FUNCTION__,
                register_rw_engine,
                instance);
        }
        else if (!engine->valid)
        {
            retval = -LWL_UNBOUND_DEVICE;
            LWSWITCH_PRINT(device, ERROR,
                "%s: REGISTER_RW_ENGINE/instance 0x%x(%d) disabled or invalid\n",
                __FUNCTION__,
                register_rw_engine,
                instance);
        }
        else
        {
            if (bcast)
            {
                //
                // Caveat emptor: A read of a broadcast register is
                // implementation-specific.
                //
                base = engine->bc_addr;
            }
            else
            {
                base = engine->uc_addr;
            }

            if (base == 0)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: REGISTER_RW_ENGINE/instance 0x%x(%d) has %s base address 0!\n",
                    __FUNCTION__,
                    register_rw_engine,
                    instance,
                    (bcast ? "BCAST" : "UNICAST" ));
                retval = -LWL_IO_ERROR;
            }
        }
    }

    *base_addr = base;
    return retval;
}

/*
 * CTRL_LWSWITCH_REGISTER_READ
 *
 * This provides direct access to the MMIO space for trusted clients like
 * MODS.
 * This API should not be exposed to unselwre clients.
 */

static LwlStatus
lwswitch_ctrl_register_read_sv10
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_READ *p
)
{
    LwU32 base;
    LwU32 data;
    LwlStatus retval = LWL_SUCCESS;

    retval = _lwswitch_get_engine_base(device, p->engine, p->instance, LW_FALSE, &base);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    // Make sure target offset isn't out-of-range
    if ((base + p->offset) >= device->lwlink_device->pciInfo.bars[0].barSize)
    {
        return -LWL_IO_ERROR;
    }

    //
    // Some legacy device manuals are not 0-based (IP style).
    //
    data = LWSWITCH_OFF_RD32(device, base + p->offset);
    p->val = data;

    return LWL_SUCCESS;
}

/*
 * CTRL_LWSWITCH_REGISTER_WRITE
 *
 * This provides direct access to the MMIO space for trusted clients like
 * MODS.
 * This API should not be exposed to unselwre clients.
 */

static LwlStatus
lwswitch_ctrl_register_write_sv10
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_WRITE *p
)
{
    LwU32 base;
    LwlStatus retval = LWL_SUCCESS;

    retval = _lwswitch_get_engine_base(device, p->engine, p->instance, p->bcast, &base);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    // Make sure target offset isn't out-of-range
    if ((base + p->offset) >= device->lwlink_device->pciInfo.bars[0].barSize)
    {
        return -LWL_IO_ERROR;
    }

    //
    // Some legacy device manuals are not 0-based (IP style).
    //
    LWSWITCH_OFF_WR32(device, base + p->offset, p->val);

    return LWL_SUCCESS;
}

static LwBool
lwswitch_is_link_valid_sv10
(
    lwswitch_device *device,
    LwU32            link_id
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (link_id < lwswitch_get_num_links_sv10(device))
    {
        return chip_device->link[link_id].valid;
    }
    else
    {
        LWSWITCH_ASSERT(link_id < lwswitch_get_num_links_sv10(device));
        return LW_FALSE;
    }
}

static void
lwswitch_set_fatal_error_sv10
(
    lwswitch_device *device,
    LwBool           device_fatal,
    LwU32            link_id
)
{
    LwU32 reg;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (device_fatal)
    {
        reg = LWSWITCH_SAW_RD32_SV10(device, _LWLSAW, _SCRATCH_WARM);
        reg = FLD_SET_DRF_NUM(_LWLSAW, _SCRATCH_WARM, _DEVICE_RESET_REQUIRED,
                              1, reg);

        LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _SCRATCH_WARM, reg);
    }

    if (!lwswitch_is_link_valid(device, link_id))
    {
        // Called with invalid link ID.  Abort
        LWSWITCH_ASSERT(lwswitch_is_link_valid(device, link_id));
        return;
    }
    else
    {
        chip_device->link[link_id].fatal_error_oclwrred = LW_TRUE;

        if (!device_fatal)
        {
            reg = LWSWITCH_LINK_RD32_SV10(device, link_id, NPORT, _NPORT, _SCRATCH_WARM);
            reg = FLD_SET_DRF_NUM(_NPORT, _SCRATCH_WARM, _PORT_RESET_REQUIRED,
                                  1, reg);

            LWSWITCH_LINK_WR32_SV10(device, link_id, NPORT, _NPORT, _SCRATCH_WARM, reg);
        }
    }
}

static LwU32
lwswitch_get_latency_sample_interval_msec_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    return chip_device->latency_stats->sample_interval_msec;
}

static LwU32
lwswitch_get_swap_clk_default_sv10
(
    lwswitch_device *device
)
{
    return LW_PBUS_LWHS_REFCLK_PAD_CTRL_SWAP_CLK_DEFAULT;
}

static LwlStatus
lwswitch_ctrl_get_ingress_reqlinkid_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *params
)
{
    LwU32 reg;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, params->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    reg = LWSWITCH_LINK_RD32_SV10(device, params->portNum, NPORT, _INGRESS,
        _REQLINKID);

    params->requesterLinkID = DRF_VAL(_INGRESS, _REQLINKID, _REQLINKID, reg);

    return LWL_SUCCESS;
}

static LwBool
lwswitch_is_link_in_use_sv10
(
    lwswitch_device *device,
    LwU32 link_id
)
{
    LwU32 data;

    data = LWSWITCH_LINK_RD32_SV10(device, link_id,
                                   DLPL, _PLWL, _LINK_STATE);

    return (DRF_VAL(_PLWL, _LINK_STATE, _STATE, data) !=
            LW_PLWL_LINK_STATE_STATE_INIT);
}

const static LwU32 nport_reg_addr[] =
{
    LW_NPORT_CTRL,
    LW_NPORT_CTRL_STOP,
    LW_NPORT_PORTSTAT_CONTROL,
    LW_NPORT_PORTSTAT_SNAP_CONTROL,
    LW_NPORT_PORTSTAT_WINDOW_LIMIT,
    LW_NPORT_PORTSTAT_LIMIT_LOW_0,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_0,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_0,
    LW_NPORT_PORTSTAT_LIMIT_LOW_1,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_1,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_1,
    LW_NPORT_PORTSTAT_LIMIT_LOW_2,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_2,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_2,
    LW_NPORT_PORTSTAT_LIMIT_LOW_3,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_3,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_3,
    LW_NPORT_PORTSTAT_LIMIT_LOW_4,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_4,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_4,
    LW_NPORT_PORTSTAT_LIMIT_LOW_5,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_5,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_5,
    LW_NPORT_PORTSTAT_LIMIT_LOW_6,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_6,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_6,
    LW_NPORT_PORTSTAT_LIMIT_LOW_7,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_7,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_7,
    LW_NPORT_PORTSTAT_SOURCE_FILTER,
    LW_NPORT_DEBUG_MUX_SELECT,
    LW_NPORT_DEBUG_MUX_CTRL,
    LW_NPORT_RESTORETAGPOOLCOUNTS,
    LW_NPORT_TAGPOOLENTRYCOUNT_0,
    LW_NPORT_TAGPOOLENTRYCOUNT_1,
    LW_NPORT_TAGPOOLENTRYCOUNT_2,
    LW_NPORT_TAGPOOLENTRYCOUNT_3,
    LW_NPORT_TAGPOOLENTRYCOUNT_4,
    LW_NPORT_TAGPOOLENTRYCOUNT_5,
    LW_NPORT_TAGPOOLENTRYCOUNT_6,
    LW_NPORT_ERR_UC_SEVERITY_NPORT,
    LW_NPORT_ERR_CONTROL_NPORT,
    LW_FSTATE_FLUSHSTATECONTROL,
    LW_FSTATE_TAGPOOLWATERMARK,
    LW_FSTATE_ECC_ERROR_COUNT_CRUMBSTORE,
    LW_FSTATE_ECC_ERROR_LIMIT_CRUMBSTORE,
    LW_FSTATE_ECC_ERROR_COUNT_TAGSTORE,
    LW_FSTATE_ECC_ERROR_LIMIT_TAGSTORE,
    LW_FSTATE_ECC_ERROR_COUNT_FLUSHREQ,
    LW_FSTATE_ECC_ERROR_LIMIT_FLUSHREQ,
    LW_FSTATE_ERR_LOG_EN_0,
    LW_FSTATE_ERR_CONTAIN_EN_0,
    LW_FSTATE_ERR_INJECT_0,
    LW_TSTATE_TAGSTATECONTROL,
    LW_TSTATE_ECC_ERROR_COUNT_CRUMBSTORE,
    LW_TSTATE_ECC_ERROR_LIMIT_CRUMBSTORE,
    LW_TSTATE_ECC_ERROR_COUNT_TAGSTORE,
    LW_TSTATE_ECC_ERROR_LIMIT_TAGSTORE,
    LW_TSTATE_ERR_LOG_EN_0,
    LW_TSTATE_ERR_CONTAIN_EN_0,
    LW_TSTATE_ERR_INJECT_0,
    LW_INGRESS_CTRL,
    LW_INGRESS_REQLINKID,
    LW_INGRESS_FLOWIDADDRMASKLO,
    LW_INGRESS_FLOWIDADDRMASKHI,
    LW_INGRESS_ECC_ERROR_COUNT,
    LW_INGRESS_ECC_ERROR_LIMIT,
    LW_INGRESS_ATRMAPDATA0,
    LW_INGRESS_ATRMAPDATA1,
    LW_INGRESS_ATRMAPDATA2,
    LW_INGRESS_ATRMAPDATA3,
    LW_INGRESS_CPUVIRTMAPDATA0,
    LW_INGRESS_CPUVIRTMAPDATA1,
    LW_INGRESS_CPUVIRTMAPDATA2,
    LW_INGRESS_CPUVIRTMAPDATA3,
    LW_INGRESS_ATSDRESULTMAPDATA0_0,
    LW_INGRESS_ATSDRESULTMAPDATA1_0,
    LW_INGRESS_ATSDRESULTMAPDATA2_0,
    LW_INGRESS_ATSDRESULTMAPDATA3_0,
    LW_INGRESS_ATSDRESULTMAPDATA0_1,
    LW_INGRESS_ATSDRESULTMAPDATA1_1,
    LW_INGRESS_ATSDRESULTMAPDATA2_1,
    LW_INGRESS_ATSDRESULTMAPDATA3_1,
    LW_INGRESS_ATSDRESULTMAPDATA0_2,
    LW_INGRESS_ATSDRESULTMAPDATA1_2,
    LW_INGRESS_ATSDRESULTMAPDATA2_2,
    LW_INGRESS_ATSDRESULTMAPDATA3_2,
    LW_INGRESS_ATSDRESULTMAPDATA0_3,
    LW_INGRESS_ATSDRESULTMAPDATA1_3,
    LW_INGRESS_ATSDRESULTMAPDATA2_3,
    LW_INGRESS_ATSDRESULTMAPDATA3_3,
    LW_INGRESS_ATSDRESULTMAPDATA0_4,
    LW_INGRESS_ATSDRESULTMAPDATA1_4,
    LW_INGRESS_ATSDRESULTMAPDATA2_4,
    LW_INGRESS_ATSDRESULTMAPDATA3_4,
    LW_INGRESS_ATSDRESULTMAPDATA0_5,
    LW_INGRESS_ATSDRESULTMAPDATA1_5,
    LW_INGRESS_ATSDRESULTMAPDATA2_5,
    LW_INGRESS_ATSDRESULTMAPDATA3_5,
    LW_INGRESS_ATSDRESULTMAPDATA0_6,
    LW_INGRESS_ATSDRESULTMAPDATA1_6,
    LW_INGRESS_ATSDRESULTMAPDATA2_6,
    LW_INGRESS_ATSDRESULTMAPDATA3_6,
    LW_INGRESS_ATSDRESULTMAPDATA0_7,
    LW_INGRESS_ATSDRESULTMAPDATA1_7,
    LW_INGRESS_ATSDRESULTMAPDATA2_7,
    LW_INGRESS_ATSDRESULTMAPDATA3_7,
    LW_INGRESS_ATSDRESULTMAPDATA0_8,
    LW_INGRESS_ATSDRESULTMAPDATA1_8,
    LW_INGRESS_ATSDRESULTMAPDATA2_8,
    LW_INGRESS_ATSDRESULTMAPDATA3_8,
    LW_INGRESS_ATSDRESULTMAPDATA0_9,
    LW_INGRESS_ATSDRESULTMAPDATA1_9,
    LW_INGRESS_ATSDRESULTMAPDATA2_9,
    LW_INGRESS_ATSDRESULTMAPDATA3_9,
    LW_INGRESS_ATSDRESULTMAPDATA0_10,
    LW_INGRESS_ATSDRESULTMAPDATA1_10,
    LW_INGRESS_ATSDRESULTMAPDATA2_10,
    LW_INGRESS_ATSDRESULTMAPDATA3_10,
    LW_INGRESS_ATSDRESULTMAPDATA0_11,
    LW_INGRESS_ATSDRESULTMAPDATA1_11,
    LW_INGRESS_ATSDRESULTMAPDATA2_11,
    LW_INGRESS_ATSDRESULTMAPDATA3_11,
    LW_INGRESS_ATSDRESULTMAPDATA0_12,
    LW_INGRESS_ATSDRESULTMAPDATA1_12,
    LW_INGRESS_ATSDRESULTMAPDATA2_12,
    LW_INGRESS_ATSDRESULTMAPDATA3_12,
    LW_INGRESS_ATSDRESULTMAPDATA0_13,
    LW_INGRESS_ATSDRESULTMAPDATA1_13,
    LW_INGRESS_ATSDRESULTMAPDATA2_13,
    LW_INGRESS_ATSDRESULTMAPDATA3_13,
    LW_INGRESS_ATSDRESULTMAPDATA0_14,
    LW_INGRESS_ATSDRESULTMAPDATA1_14,
    LW_INGRESS_ATSDRESULTMAPDATA2_14,
    LW_INGRESS_ATSDRESULTMAPDATA3_14,
    LW_INGRESS_ATSDRESULTMAPDATA0_15,
    LW_INGRESS_ATSDRESULTMAPDATA1_15,
    LW_INGRESS_ATSDRESULTMAPDATA2_15,
    LW_INGRESS_ATSDRESULTMAPDATA3_15,
    LW_INGRESS_ATSDMATCHMAPDATA_0,
    LW_INGRESS_ATSDMATCHMAPDATA_1,
    LW_INGRESS_ATSDMATCHMAPDATA_2,
    LW_INGRESS_ATSDMATCHMAPDATA_3,
    LW_INGRESS_ATSDMATCHMAPDATA_4,
    LW_INGRESS_ATSDMATCHMAPDATA_5,
    LW_INGRESS_ATSDMATCHMAPDATA_6,
    LW_INGRESS_ATSDMATCHMAPDATA_7,
    LW_INGRESS_ATSDMATCHMAPDATA_8,
    LW_INGRESS_ATSDMATCHMAPDATA_9,
    LW_INGRESS_ATSDMATCHMAPDATA_10,
    LW_INGRESS_ATSDMATCHMAPDATA_11,
    LW_INGRESS_ATSDMATCHMAPDATA_12,
    LW_INGRESS_ATSDMATCHMAPDATA_13,
    LW_INGRESS_ATSDMATCHMAPDATA_14,
    LW_INGRESS_ATSDMATCHMAPDATA_15,
    LW_INGRESS_RATELIMITCONTROL_0,
    LW_INGRESS_RATELIMITPARAM_0,
    LW_INGRESS_RATELIMITUPTHRESHOLD_0,
    LW_INGRESS_RATELIMITDOWNTHRESHOLD_0,
    LW_INGRESS_RATELIMITCONTROL_1,
    LW_INGRESS_RATELIMITPARAM_1,
    LW_INGRESS_RATELIMITUPTHRESHOLD_1,
    LW_INGRESS_RATELIMITDOWNTHRESHOLD_1,
    LW_INGRESS_BUFFER_CREDIT_VC_0,
    LW_INGRESS_BUFFER_CREDIT_VC_1,
    LW_INGRESS_BUFFER_CREDIT_VC_2,
    LW_INGRESS_BUFFER_CREDIT_VC_3,
    LW_INGRESS_BUFFER_CREDIT_VC_4,
    LW_INGRESS_BUFFER_CREDIT_VC_5,
    LW_INGRESS_BUFFER_CREDIT_VC_6,
    LW_INGRESS_BUFFER_CREDIT_VC_7,
    LW_INGRESS_ERR_LOG_EN_0,
    LW_INGRESS_ERR_CONTAIN_EN_0,
    LW_INGRESS_ERR_INJECT_0,
    LW_EGRESS_CTRL,
    LW_EGRESS_REMAPBDF,
    LW_EGRESS_TXCDCWATERMARK,
    LW_EGRESS_ECC_CORRECTABLE_COUNT_0,
    LW_EGRESS_ECC_CORRECTABLE_ERROR_LIMIT_0,
    LW_EGRESS_ECC_CORRECTABLE_COUNT_1,
    LW_EGRESS_ECC_CORRECTABLE_ERROR_LIMIT_1,
    LW_EGRESS_BUFFER_SZ_VC0,
    LW_EGRESS_BUFFER_SZ_VC1,
    LW_EGRESS_BUFFER_SZ_VC2,
    LW_EGRESS_BUFFER_SZ_VC3,
    LW_EGRESS_BUFFER_SZ_VC4,
    LW_EGRESS_BUFFER_SZ_VC5,
    LW_EGRESS_BUFFER_SZ_VC6,
    LW_EGRESS_BUFFER_SZ_VC7,
    LW_EGRESS_ENDPOINT_ADDRMATCH,
    LW_EGRESS_ENDPOINT_ADDRMASK,
    LW_EGRESS_ERR_LOG_EN_0,
    LW_EGRESS_ERR_CONTAIN_EN_0,
    LW_EGRESS_ERR_INJECT_0,
    LW_ROUTE_ROUTE_CONTROL,
    LW_ROUTE_DEST_PORT,
    LW_ROUTE_ECC_ERROR_COUNT,
    LW_ROUTE_ECC_ERROR_LIMIT,
    LW_ROUTE_ECC_ERROR_ADDRESS,
    LW_ROUTE_CMD_ROUTE_TABLE0,
    LW_ROUTE_CMD_ROUTE_TABLE1,
    LW_ROUTE_CMD_ROUTE_TABLE2,
    LW_ROUTE_CMD_ROUTE_TABLE3,
    LW_ROUTE_RMTBUFCREDITREG,
    LW_ROUTE_RMTBUFCREDITLIMIT,
    LW_ROUTE_BUFFER_SZ_VC_0,
    LW_ROUTE_BUFFER_SZ_VC_1,
    LW_ROUTE_BUFFER_SZ_VC_2,
    LW_ROUTE_BUFFER_SZ_VC_3,
    LW_ROUTE_BUFFER_SZ_VC_4,
    LW_ROUTE_BUFFER_SZ_VC_5,
    LW_ROUTE_BUFFER_SZ_VC_6,
    LW_ROUTE_BUFFER_SZ_VC_7,
    LW_ROUTE_ERR_LOG_EN_0,
    LW_ROUTE_ERR_CONTAIN_EN_0,
    LW_ROUTE_ERR_INJECT_0,
    LW_ROUTE_TDRSV,
};

static LwlStatus
_lwswitch_drain_port
(
    lwswitch_device *device,
    LwU32 link_id
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    // There are 4 NPORTs (links) per NPG.
    LwU32 npg = link_id / NUM_NPORT_INSTANCES_SV10;
    LwU32 nport = link_id % NUM_NPORT_INSTANCES_SV10;
    LwU32 *nport_reg_val = NULL;
    LwU32 i;
    LwU32 val;
    LwU32 reg_count = LW_ARRAY_ELEMENTS(nport_reg_addr);
    LwU32 nport_mask;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, NPORT, link_id))
    {
        return -LWL_BAD_ARGS;
    }

    nport_reg_val = lwswitch_os_malloc(sizeof(nport_reg_addr));
    if (nport_reg_val == NULL)
    {
        return -LWL_NO_MEM;
    }

    LWSWITCH_FLUSH_MMIO(device);

    // Backup NPORT state
    for (i = 0; i < reg_count; i++)
    {
        nport_reg_val[i] = LWSWITCH_SUBENG_OFF_RD32_SV10(device, NPG, npg, NPORT,
                                                         nport, nport_reg_addr[i]);
    }

    // WAR for bug 1853124. See comment 25.
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_LOG_EN_0, 0);

    LWSWITCH_FLUSH_MMIO(device);

    // Drain the port.
    val = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _NPORT, _CTRL);
    val = FLD_SET_DRF(_NPORT, _CTRL, _EGDRAINENB, _ENABLE, val);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _CTRL, val);

    LWSWITCH_FLUSH_MMIO(device);

    // Wait for packets to drain (safe-guard).
    lwswitch_os_sleep(1);

    // Reset the port to disable drain. (active-low).
    val = LWSWITCH_NPG_RD32_SV10(device, npg, _NPG, _WARMRESET);
    nport_mask = DRF_VAL(_NPG, _WARMRESET, _NPORTWARMRESET, val);
    nport_mask &= ~LWBIT(nport);
    val = FLD_SET_DRF_NUM(_NPG, _WARMRESET, _NPORTWARMRESET, nport_mask, val);
    LWSWITCH_NPG_WR32_SV10(device, npg, _NPG, _WARMRESET, val);

    lwswitch_os_sleep(1);

    // Take the port out of reset.
    nport_mask |= LWBIT(nport);
    val = FLD_SET_DRF_NUM(_NPG, _WARMRESET, _NPORTWARMRESET, nport_mask, val);
    LWSWITCH_NPG_WR32_SV10(device, npg, _NPG, _WARMRESET, val);

    LWSWITCH_FLUSH_MMIO(device);

    // Restore NPORT state.
    for (i = 0; i < reg_count; i++)
    {
        LWSWITCH_SUBENG_OFF_WR32_SV10(device, NPG, , npg, NPORT, , nport, uc,
                                      nport_reg_addr[i], nport_reg_val[i]);
    }

    LWSWITCH_FLUSH_MMIO(device);

    // Clear buffer ready.
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _CTRL_BUFFER_READY, 0);

    // Clear debug reset state.
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_FIRST_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_STATUS_0, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_C_FIRST_NPORT, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_C_STATUS_NPORT, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_UC_FIRST_NPORT, ~0);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_UC_STATUS_NPORT, ~0);

    // Restore interrupt mask.
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_REPORT_EN_0,
                             chip_device->intr_mask.ingress);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_REPORT_EN_0,
                             chip_device->intr_mask.route);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_REPORT_EN_0,
                             chip_device->intr_mask.fstate);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_REPORT_EN_0,
                             chip_device->intr_mask.tstate);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_REPORT_EN_0,
                             chip_device->intr_mask.egress);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_UC_MASK_NPORT,
                             chip_device->intr_mask.nport_uc);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_C_MASK_NPORT,
                             chip_device->intr_mask.nport_c);

    lwswitch_os_free(nport_reg_val);

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_reset_link_pair
(
    lwswitch_device *device,
    LwU32 link_id
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link_pair[NUM_DLPL_INSTANCES_SV10];
    LwU32 lwlipt_intr_ctr;
    LwU32 lwlipt_err_uc_severity;
    LwU32 sioctrl = link_id / NUM_DLPL_INSTANCES_SV10;
    LwU32 lwlipt = link_id / NUM_DLPL_INSTANCES_SV10;

    // There are two links per SIOCTRL.
    link_pair[0] = sioctrl * NUM_DLPL_INSTANCES_SV10 + 0;
    link_pair[1] = sioctrl * NUM_DLPL_INSTANCES_SV10 + 1;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link_pair[0]) ||
        !LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link_pair[1]))
    {
        return -LWL_BAD_ARGS;
    }

    // Disable DLPL interrupts (will be re-enabled by _lwswitch_init_link)
    LWSWITCH_LINK_WR32_SV10(device, link_pair[0], DLPL, _PLWL, _INTR_STALL_EN, 0x0);
    LWSWITCH_LINK_WR32_SV10(device, link_pair[0], DLPL, _PLWL, _INTR_NONSTALL_EN, 0x0);
    LWSWITCH_LINK_WR32_SV10(device, link_pair[1], DLPL, _PLWL, _INTR_STALL_EN, 0x0);
    LWSWITCH_LINK_WR32_SV10(device, link_pair[1], DLPL, _PLWL, _INTR_NONSTALL_EN, 0x0);

    // Issue LANE_DISABLE (safe-guard)
    (void)lwswitch_minion_send_command_sv10(device, link_pair[0],
                        LW_MINION_LWLINK_DL_CMD_COMMAND_LANEDISABLE, 0);

    (void)lwswitch_minion_send_command_sv10(device, link_pair[1],
                        LW_MINION_LWLINK_DL_CMD_COMMAND_LANEDISABLE, 0);

    // Disable CLKCROSS error logging. See 1828802.
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl,
                            _LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, 0x0);
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl,
                            _LWLCTRL, _CLKCROSS_1_ERR_LOG_EN_0, 0x0);

    // Disable LWLIPT interrupts. WAR for bug 1815118 (see comment 56.)
    lwlipt_intr_ctr = LWSWITCH_LWLIPT_RD32_SV10(device, lwlipt,
                            _LWLIPT, _INTR_CONTROL_LINK0);
    lwlipt_err_uc_severity = LWSWITCH_LWLIPT_RD32_SV10(device, lwlipt,
                            _LWLIPT, _ERR_UC_SEVERITY_LINK0);

    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                         _LWLIPT, _INTR_CONTROL_LINK0, 0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                         _LWLIPT, _INTR_CONTROL_LINK1, 0x0);

    LWSWITCH_FLUSH_MMIO(device);

    // Issue LINK_RESET on both links.
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl, _LWLCTRL, _RESET,
                    DRF_NUM(_LWLCTRL, _RESET, _LINKRESET, 0x0));
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl, _LWLCTRL, _DEBUG_RESET,
                    DRF_NUM(_LWLCTRL, _DEBUG_RESET, _LINK, 0x0) |
                    DRF_NUM(_LWLCTRL, _DEBUG_RESET, _COMMON, 0x0));

    // Issue CLOCKCROSS_RESET on both links.
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl, _LWLCTRL, _CLKCROSS_RESET,
                    DRF_NUM(_LWLCTRL, _CLKCROSS_RESET, _CLKCROSSRESET, 0x0));

    LWSWITCH_FLUSH_MMIO(device);

    // 1ms as a port reset delay. (as per manuals is is 8 usec.)
    lwswitch_os_sleep(1);

    //
    // Mask LWLIPT interrupts and clear LWLIPT status registers.
    // WAR for bug 1815118 (see comment 56.)
    //
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_STATUS_LINK0, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_STATUS_LINK1, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_MASK_LINK0, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_MASK_LINK1, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_SEVERITY_LINK0, 0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_SEVERITY_LINK1, 0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_FIRST_LINK0, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_FIRST_LINK1, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_ADVISORY_LINK0,0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_UC_ADVISORY_LINK1, 0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_C_STATUS_LINK0, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_C_STATUS_LINK1, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_C_MASK_LINK0, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_C_MASK_LINK1, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_C_FIRST_LINK0, ~0x0);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt, _LWLIPT, _ERR_C_FIRST_LINK1, ~0x0);

    LWSWITCH_FLUSH_MMIO(device);

    // Take both links out of CROSSCLOCK_RESET.
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl, _LWLCTRL, _CLKCROSS_RESET,
                    DRF_NUM(_LWLCTRL, _CLKCROSS_RESET, _CLKCROSSRESET, 0x3));

    // Take both links out of LINK_RESET.
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl, _LWLCTRL, _RESET,
                    DRF_NUM(_LWLCTRL, _RESET, _LINKRESET, 0x3));
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl, _LWLCTRL, _DEBUG_RESET,
                    DRF_NUM(_LWLCTRL, _DEBUG_RESET, _LINK, 0x3) |
                    DRF_NUM(_LWLCTRL, _DEBUG_RESET, _COMMON, 0x1));

    LWSWITCH_FLUSH_MMIO(device);

    // Re-enable LWLIPT interrupts.
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _ERR_UC_MASK_LINK0, chip_device->intr_mask.lwlipt_uc);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _ERR_UC_MASK_LINK1, chip_device->intr_mask.lwlipt_uc);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _ERR_UC_SEVERITY_LINK0, lwlipt_err_uc_severity);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _ERR_UC_SEVERITY_LINK1, lwlipt_err_uc_severity);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _ERR_C_MASK_LINK0, chip_device->intr_mask.lwlipt_c);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _ERR_C_MASK_LINK1, chip_device->intr_mask.lwlipt_c);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _INTR_CONTROL_LINK0, lwlipt_intr_ctr);
    LWSWITCH_LWLIPT_WR32_SV10(device, lwlipt,
                    _LWLIPT, _INTR_CONTROL_LINK1, lwlipt_intr_ctr);

    // Re-enable CLKCROSS error logging.
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl,
                _LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, chip_device->intr_mask.clkcross);
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl,
                _LWLCTRL, _CLKCROSS_1_ERR_LOG_EN_0, chip_device->intr_mask.clkcross);
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl,
                _LWLCTRL, _CLKCROSS_0_ERR_REPORT_EN_0, chip_device->intr_mask.clkcross);
    LWSWITCH_SIOCTRL_WR32_SV10(device, sioctrl,
                _LWLCTRL, _CLKCROSS_1_ERR_REPORT_EN_0, chip_device->intr_mask.clkcross);

    // Clear NPORT buffer ready.
    LWSWITCH_LINK_WR32_SV10(device, link_pair[0], NPORT, _NPORT, _CTRL_BUFFER_READY,
                DRF_NUM(_NPORT, _CTRL_BUFFER_READY, _BUFFERRDY, 0x0));
    LWSWITCH_LINK_WR32_SV10(device, link_pair[1], NPORT, _NPORT, _CTRL_BUFFER_READY,
                DRF_NUM(_NPORT, _CTRL_BUFFER_READY, _BUFFERRDY, 0x0));

    LWSWITCH_FLUSH_MMIO(device);

    return LWL_SUCCESS;
}

static void
_lwswitch_reset_sw_state
(
    lwswitch_device *device,
    LwU32 link_id
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    chip_device->link[link_id].fatal_error_oclwrred = LW_FALSE;
    chip_device->link[link_id].ingress_packet_latched = LW_FALSE;
    chip_device->link[link_id].egress_packet_latched = LW_FALSE;

    _lwswitch_init_link_info(device, link_id);
}

static LwlStatus
lwswitch_reset_and_drain_links_sv10
(
    lwswitch_device *device,
    LwU64 link_mask
)
{
    LwlStatus status = LWL_SUCCESS;
    LwlStatus temp_status;
    LwU8 link_id;
    lwlink_link *even_link;
    lwlink_link *odd_link;
    LwU64 mask = link_mask;

    if (link_mask == 0)
    {
        return -LWL_BAD_ARGS;
    }

    // For SV10, we must reset and drain both odd-even links together.
    for (link_id = 0; mask != 0; link_id += 2, mask >>= 2)
    {
        if ((mask & 0x3) != 0 && (mask & 0x3) != 0x3)
        {
            return -LWL_BAD_ARGS;
        }

        if (link_id >= LWSWITCH_NUM_LINKS_SV10)
        {
            return -LWL_BAD_ARGS;
        }
    }

    mask = link_mask;

    for (link_id = 0; mask != 0; link_id += 2, mask >>= 2)
    {
        if ((mask & 0x3) == 0)
        {
            continue;
        }

        // Unregister links to make them unusable while reset is in progress.
        even_link = lwswitch_get_link(device, link_id);
        if (even_link != NULL)
        {
            lwlink_lib_unregister_link(even_link);
        }

        odd_link = lwswitch_get_link(device, link_id + 1);
        if (odd_link != NULL)
        {
            lwlink_lib_unregister_link(odd_link);
        }

        // Reset and drain links.
        status = _lwswitch_reset_link_pair(device, link_id);

        temp_status = _lwswitch_drain_port(device, link_id);
        status = (status == LWL_SUCCESS) ? temp_status : status;
        temp_status = _lwswitch_drain_port(device, link_id + 1);
        status = (status == LWL_SUCCESS) ? temp_status : status;

        // Clear SW state only if the reset is successful.
        if (status == LWL_SUCCESS)
        {
            _lwswitch_reset_sw_state(device, link_id);
            _lwswitch_reset_sw_state(device, link_id + 1);
        }

        //
        // Re-register links.
        //
        // Note that we deliberately check NULL pointers here because the link
        // SW state can be unregistered via CTRL_LWSWITCH_UNREGISTER_LINK.
        //
        if (even_link != NULL)
        {
            temp_status = lwlink_lib_register_link(device->lwlink_device,
                                                   even_link);
            if (temp_status != LWL_SUCCESS)
            {
                lwswitch_destroy_link(even_link);
                status = (status == LWL_SUCCESS) ? temp_status : status;
            }
        }

        if (odd_link != NULL)
        {
            temp_status = lwlink_lib_register_link(device->lwlink_device,
                                                   odd_link);
            if (temp_status != LWL_SUCCESS)
            {
                lwswitch_destroy_link(odd_link);
                status = (status == LWL_SUCCESS) ? temp_status : status;
            }
        }

        // Initialize select scratch registers to 0x0
        lwswitch_init_scratch_sv10(device);

        if (status != LWL_SUCCESS)
        {
            break;
        }
    }

    return status;
}

static LwU32
lwswitch_get_device_dma_width_sv10
(
    lwswitch_device *device
)
{
    return 0;
}

static LwlStatus
lwswitch_get_lwlink_ecc_errors_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwU32
lwswitch_get_link_ip_version_sv10
(
    lwswitch_device *device,
    LwU32            link_id
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    LWSWITCH_ASSERT(link_id < lwswitch_get_num_links_sv10(device));
    return chip_device->link[link_id].engDLPL->version;
}

static LwlStatus
lwswitch_ctrl_get_throughput_counters_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwBool
lwswitch_is_soe_supported_sv10
(
    lwswitch_device *device
)
{
    return LW_FALSE;
}

LwlStatus
lwswitch_ctrl_get_bios_info_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_BIOS_INFO_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwBool
lwswitch_is_inforom_supported_sv10
(
    lwswitch_device *device
)
{
    return LW_FALSE;
}

static LwBool
lwswitch_is_spi_supported_sv10
(
    lwswitch_device *device
)
{
    return LW_FALSE;
}

static LwBool
lwswitch_is_smbpbi_supported_sv10
(
    lwswitch_device *device
)
{
    return LW_FALSE;
}

static LwlStatus
lwswitch_soe_prepare_for_reset_sv10
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
lwswitch_soe_set_ucode_core_sv10
(
    lwswitch_device *device,
    LwBool bFalcon
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwU32
_lwswitch_sha1_copy
(
    LwU8 *pBuff,
    LwU32 index,
    LwU32 size,
    void *pInfo
)
{
    LwU8 *pBytes = pInfo;
    lwswitch_os_memcpy(pBuff, pBytes + index, size);
    return size;
}

static void
_lwswitch_construct_uuid_sv10
(
    lwswitch_device *device,
    LwU64 *ecid,
    LwU32 ecidSize
)
{
    typedef struct
    {
        LwU8 digest[LW_SHA1_DIGEST_LENGTH];
        struct
        {
            LwU64 ecid[2];   // LWSWITCH ECID value
            LwU32 boot0Val;  // LW_PSMC_BOOT_0 value (Chip ID information)
            LwU32 boot42Val; // LW_PSMC_BOOT_42 value (Chip ID information)
        } seed;
    } UUID_CONTEXT;

    UUID_CONTEXT uuidCtx;

    //
    // populate the UUID seed
    //
    uuidCtx.seed.boot0Val  = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_0);
    uuidCtx.seed.boot42Val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);
    lwswitch_os_memcpy(uuidCtx.seed.ecid, ecid, ecidSize);

    //
    // Generate the LWSwitch UUID; note that UUID strings only use the
    // first 16 bytes of the 20-byte SHA-1 digest.
    //
    sha1Generate(uuidCtx.digest, &(uuidCtx.seed), sizeof(uuidCtx.seed),
                _lwswitch_sha1_copy);

    lwswitch_os_memcpy(&device->uuid.uuid, uuidCtx.digest, LW_UUID_LEN);
}

/*
 * @Brief : Additional setup needed after device initialization
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
LwlStatus
lwswitch_post_init_device_setup_sv10
(
    lwswitch_device *device
)
{
    return LWL_SUCCESS;
}

/*
 * @Brief : Setting up system registers after device initialization
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
LwlStatus
lwswitch_setup_link_system_registers_sv10
(
    lwswitch_device *device
)
{
    return LWL_SUCCESS;
}

/*
 * @Brief : Additional setup needed after blacklisted device initialization
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
void
lwswitch_post_init_blacklist_device_setup_sv10
(
    lwswitch_device *device
)
{
    // NOP
}

void lwswitch_load_uuid_sv10
(
    lwswitch_device *device
)
{
    LwU32 lotCode0;
    LwU32 lotCode1;
    LwU32 fabCode;
    LwU32 waferId;
    LwU32 vendorCode;
    LwU32 xcoord;
    LwU32 ycoord;
    LwU32 offset;
    LwU64 ecidData[2];

    lotCode0   = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_LOT_CODE_0);
    lotCode1   = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_LOT_CODE_1);
    fabCode    = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_FAB_CODE);
    waferId    = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_WAFER_ID);
    vendorCode = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_VENDOR_CODE);
    xcoord     = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_X_COORDINATE);
    ycoord     = lwswitch_fuse_opt_read_sv10(device, LW_FUSE_OPT_Y_COORDINATE);

    // Aggregate the components into the ECID
    ecidData[0] = (LwU64) vendorCode;
    offset = DRF_SIZE(LW_FUSE_OPT_VENDOR_CODE_DATA);

    ecidData[0] |= (LwU64) lotCode0 << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_LOT_CODE_0_DATA);

    ecidData[0] |= (LwU64) lotCode1 << offset;

    ecidData[1] = (LwU64) fabCode;
    offset = DRF_SIZE(LW_FUSE_OPT_FAB_CODE_DATA);

    ecidData[1] |= (LwU64) waferId << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_WAFER_ID_DATA);

    ecidData[1] |= (LwU64) xcoord << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_X_COORDINATE_DATA);

    ecidData[1] |= (LwU64) ycoord << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_Y_COORDINATE_DATA);

    _lwswitch_construct_uuid_sv10(device, ecidData, sizeof(ecidData));
}

LwlStatus
lwswitch_read_oob_blacklist_state_sv10
(
    lwswitch_device *device
)
{
    /*
     * SV10 does not have the scratch register or OMS InfoROM object for
     * fabric state.  So, just return not supported.
     */

    if (device == NULL)
        return -LWL_BAD_ARGS;

    return -LWL_ERR_NOT_SUPPORTED;
}
LwlStatus
lwswitch_write_fabric_state_sv10
(
    lwswitch_device *device
)
{
    /*
     * SV10 does not have the scratch register or OMS InfoROM object for
     * fabric state.  So, just return not supported.
     */
    if (device == NULL)
        return -LWL_BAD_ARGS;

    return -LWL_ERR_NOT_SUPPORTED;
}

LwU32
lwswitch_get_eng_base_sv10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return LWSWITCH_BASE_ADDR_ILWALID;
}

LwU32
lwswitch_get_eng_count_sv10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return 0;
}

LwU32
lwswitch_eng_rd_sv10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance,
    LwU32 offset
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return 0xBADFBADF;
}

void
lwswitch_eng_wr_sv10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance,
    LwU32 offset,
    LwU32 data
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return;
}

LwU32
lwswitch_get_link_eng_inst_sv10
(
    lwswitch_device *device,
    LwU32 link_id,
    LWSWITCH_ENGINE_ID eng_id
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return LWSWITCH_ENGINE_INSTANCE_ILWALID;
}

LwU32
lwswitch_get_caps_lwlink_version_sv10
(
    lwswitch_device *device
)
{
    return LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_2_0;
}

void
lwswitch_soe_unregister_events_sv10
(
    lwswitch_device *device
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
}

LwlStatus
lwswitch_soe_register_event_callbacks_sv10
(
    lwswitch_device *device
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_SUPPORTED;
}

LWSWITCH_BIOS_LWLINK_CONFIG *
lwswitch_get_bios_lwlink_config_sv10
(
    lwswitch_device *device
)
{
    return NULL;
}

static LwlStatus
lwswitch_clear_nport_rams_sv10
(
    lwswitch_device *device
)
{
    // Not supported in SV10, and should never be called
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_pri_ring_init_sv10
(
    lwswitch_device *device
)
{
    LWSWITCH_ASSERT(0);

    // Should not be called on SV10
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_get_soe_ucode_binaries_sv10
(
    lwswitch_device *device,
    const LwU32 **soe_ucode_data,
    const LwU32 **soe_ucode_header
)
{
    LWSWITCH_ASSERT(0);

    // Not supported in SV10
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_get_remap_table_selector_sv10
(
    lwswitch_device *device,
    LWSWITCH_TABLE_SELECT_REMAP table_selector,
    LwU32 *remap_ram_sel
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwU32
lwswitch_get_ingress_ram_size_sv10
(
    lwswitch_device *device,
    LwU32 ingress_ram_selector      // LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECT*
)
{
    //
    // This could report the REQ/RSP RAM sizes, but this function is not used
    // for those tables.
    //
    LwU32 ram_size = 0;
    return ram_size;
}

/*
 * CTRL_LWSWITCH_SET_RESIDENCY_BINS
 */
static LwlStatus
lwswitch_ctrl_set_residency_bins_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_RESIDENCY_BINS *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "SET_RESIDENCY_BINS should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_GET_RESIDENCY_BINS
 */
static LwlStatus
lwswitch_ctrl_get_residency_bins_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_RESIDENCY_BINS *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "GET_RESIDENCY_BINS should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_GET_RB_STALL_BUSY
 */
static LwlStatus
lwswitch_ctrl_get_rb_stall_busy_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_RB_STALL_BUSY *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "GET_RB_STALL_BUSY should not be called on SV10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_SEND_DATA
 */
LwlStatus
lwswitch_ctrl_inband_send_data_sv10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_SEND_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_RECEIVE_DATA
 */
LwlStatus
lwswitch_ctrl_inband_read_data_sv10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_READ_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_FLUSH_DATA
 */
LwlStatus
lwswitch_ctrl_inband_flush_data_sv10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_PENDING_MESSAGES_STATS
 */
LwlStatus
lwswitch_ctrl_inband_pending_data_stats_sv10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwU32
lwswitch_read_iddq_dvdd_sv10
(
    lwswitch_device *device
)
{
    return 0;
}

LwlStatus
lwswitch_init_nxbar_sv10
(
    lwswitch_device *device
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
* @brief: This function will try to save the last valid seeds from MINION into InfoROM
* @params[in] device        reference to current lwswitch device
* @params[in] linkId        link we want to save seed data for
* return                    sv10 is not running lwlink3.0+ and will not need to support this
*/
void lwswitch_save_lwlink_seed_data_from_minion_to_inforom_sv10
(
    lwswitch_device *device,
    LwU32 linkId
)
{
    return;
}

void
lwswitch_store_seed_data_from_inforom_to_corelib_sv10
(
    lwswitch_device *device
)
{
    // Not supported in SV10
    return;
}

/*
* @brief: This function retrieves the LWLIPT public ID for a given global link idx
* @params[in]  device        reference to current lwswitch device
* @params[in]  linkId        link to retrieve LWLIPT public ID from
* @params[out] publicId      Public ID of LWLIPT owning linkId
*/
LwlStatus lwswitch_get_link_public_id_sv10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *publicId
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
* @brief: This function retrieves the internal link idx for a given global link idx
* @params[in]  device        reference to current lwswitch device
* @params[in]  linkId        link to retrieve LWLIPT public ID from
* @params[out] localLinkIdx  Internal link index of linkId
*/
LwlStatus lwswitch_get_link_local_idx_sv10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *localLinkIdx
)
{
    if (!device->hal.lwswitch_is_link_valid(device, linkId) ||
        (localLinkIdx == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    *localLinkIdx = LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_SV10(linkId);

    return LWL_SUCCESS;
}

LwlStatus lwswitch_set_training_error_info_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *pLinkTrainingErrorInfoParams
)
{
    return LWL_SUCCESS;
}

LwlStatus lwswitch_ctrl_get_fatal_error_scope_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *pParams
)
{
    LwU32 linkId;
    LwU32 reg = LWSWITCH_SAW_RD32_SV10(device, _LWLSAW, _SCRATCH_WARM);

    pParams->device = FLD_TEST_DRF_NUM(_LWLSAW, _SCRATCH_WARM, _DEVICE_RESET_REQUIRED,
                                       1, reg);

    for (linkId = 0; linkId < LWSWITCH_MAX_PORTS; linkId++)
    {
        if ((linkId >= LWSWITCH_NUM_LINKS_SV10) || !device->hal.lwswitch_is_link_valid(device, linkId))
        {
            pParams->port[linkId] = LW_FALSE;
            continue;
        }

        reg = LWSWITCH_LINK_RD32_SV10(device, linkId, NPORT, _NPORT, _SCRATCH_WARM);

        pParams->port[linkId] = FLD_TEST_DRF_NUM(_NPORT, _SCRATCH_WARM,
                                                 _PORT_RESET_REQUIRED, 1, reg);
    }

    return LW_OK;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus lwswitch_ctrl_set_mc_rid_table_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_MC_RID_TABLE_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus lwswitch_ctrl_get_mc_rid_table_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_MC_RID_TABLE_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

void lwswitch_init_scratch_sv10
(
    lwswitch_device *device
)
{
    LwU32 linkId;
    LwU32 reg;

    reg = LWSWITCH_SAW_RD32_SV10(device, _LWLSAW, _SCRATCH_WARM);
    if (reg == LW_LWLSAW_SCRATCH_WARM_DATA_INIT)
    {
        LWSWITCH_SAW_WR32_SV10(device, _LWLSAW, _SCRATCH_WARM, 0);
    }

    for (linkId = 0; linkId < LWSWITCH_NUM_LINKS_SV10; linkId++)
    {
        reg = LWSWITCH_LINK_RD32_SV10(device, linkId, NPORT, _NPORT, _SCRATCH_WARM);
        if (reg == LW_NPORT_SCRATCH_WARM_DATA_INIT)
        {
            LWSWITCH_LINK_WR32_SV10(device, linkId, NPORT, _NPORT, _SCRATCH_WARM, 0);
        }
    }
}

LwlStatus lwswitch_init_nport_sv10
(
    lwswitch_device *device
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_SUPPORTED;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus
lwswitch_launch_ALI_sv10
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif

LwlStatus
lwswitch_set_training_mode_sv10
(
    lwswitch_device *device
)
{
    return LWL_SUCCESS;
}

LwlStatus 
lwswitch_parse_bios_image_sv10
(
    lwswitch_device *device
)
{
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_get_lwlink_lp_counters_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* TRY TO ADD NEW UNPUBLISHED CODE BELOW THIS LINE */

LwBool
lwswitch_is_cci_supported_sv10
(
    lwswitch_device *device
)
{
    return LW_FALSE;
}

LwlStatus
lwswitch_get_board_id_sv10
(
    lwswitch_device *device,
    LwU16 *boardId
)
{
    return LWL_SUCCESS;
}

/*
 * @brief: This function returns current link repeater mode state for a given global link idx
 * @params[in]  device          reference to current lwswitch device
 * @params[in]  linkId          link to retrieve repeater mode state from
 * @params[out] isRepeaterMode  pointer to Repeater Mode boolean
 */
LwlStatus lwswitch_is_link_in_repeater_mode_sv10
(
    lwswitch_device *device,
    LwU32           linkId,
    LwBool          *isRepeaterMode
)
{
    *isRepeaterMode = LW_FALSE;
    return LWL_SUCCESS;
}

void
lwswitch_fetch_active_repeater_mask_sv10
(
    lwswitch_device *device
)
{
    return;
}

LwU64
lwswitch_get_active_repeater_mask_sv10
(
    lwswitch_device *device
)
{
    return 0x0;
}

LwBool
lwswitch_cci_is_optical_link_sv10
(
    lwswitch_device *device,
    LwU32 linkNumber
)
{
    LWSWITCH_ASSERT(0);
    return LW_FALSE;
}

LwlStatus
lwswitch_init_cci_sv10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, ERROR,
        "%s: CCI is not yet supported on sv10.\n",
        __FUNCTION__);
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
lwswitch_ctrl_set_port_test_mode_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_PORT_TEST_MODE *p
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    lwlink_link *link;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, MINION, p->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    link = lwswitch_get_link(device, (LwU8)p->portNum);
    if (link == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid link\n",
            __FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    if ((p->nea) && (p->ned))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NEA and NED can not both be enabled simultaneously.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    // Near End Analog 
    chip_device->link[p->portNum].nea = p->nea;

    // Near End Digital 
    chip_device->link[p->portNum].ned = p->ned;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_jtag_chain_read_sv10
(
    lwswitch_device *device,
    LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain
)
{
    LwlStatus retval = LWL_SUCCESS;

    // Compute the amount of the buffer that we will actually use.
    LwU32 dataArrayLen = jtag_chain->chainLen / 32 + !!(jtag_chain->chainLen % 32);

    // The buffer must be at least large enough to hold the chain.
    if (jtag_chain->dataArrayLen < dataArrayLen)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Buffer too small for chain.  buffer=%u  required=%u  chain=%u.\n",
            __FUNCTION__,
            jtag_chain->dataArrayLen, dataArrayLen, jtag_chain->chainLen);
        return -LWL_BAD_ARGS;
    }

    // We need a buffer from which to read.
    if (jtag_chain->data == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: data: Required parameter is NULL.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    retval = lwswitch_jtag_read_seq_sv10(device,
                 jtag_chain->chainLen,
                 jtag_chain->chipletSel,
                 jtag_chain->instrId,
                 jtag_chain->data,
                 dataArrayLen);

    return retval;
}

static LwlStatus
lwswitch_ctrl_jtag_chain_write_sv10
(
    lwswitch_device *device,
    LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain
)
{
    LwlStatus retval = LWL_SUCCESS;

    // Compute the amount of the bufer that we will actuall use.
    LwU32 dataArrayLen = jtag_chain->chainLen / 32 + !!(jtag_chain->chainLen % 32);

    // The buffer must be at least large enough to hold the chain.
    if (jtag_chain->dataArrayLen < dataArrayLen)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Buffer too small for chain.  buffer=%u  required=%u  chain=%u.\n",
            __FUNCTION__,
            jtag_chain->dataArrayLen, dataArrayLen, jtag_chain->chainLen);
        return -LWL_BAD_ARGS;
    }

    // We need a buffer from which to read.
    if (jtag_chain->data == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: data: Required parameter is NULL.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    retval = lwswitch_jtag_write_seq_sv10(device,
                  jtag_chain->chainLen,
                  jtag_chain->chipletSel,
                  jtag_chain->instrId,
                  jtag_chain->data,
                  dataArrayLen);

    return retval;
}

/*
 * @brief Inject an LWLink error on a link or links
 *
 * Errors are injected asynchronously.  This is a MODS test-only API and is an
 * exception to the locking rules regarding LWLink corelib callbacks (sv10_link.c).
 *
 * @param[in] device            LwSwitch device to contain this link
 * @param[in] p                 LWSWITCH_INJECT_LINK_ERROR
 *
 * @returns                     LWL_SUCCESS if action succeeded,
 *                              -LWL_ERR_ILWALID_STATE invalid link
 */
static LwlStatus
lwswitch_ctrl_inject_link_error_sv10
(
    lwswitch_device *device,
    LWSWITCH_INJECT_LINK_ERROR *p
)
{
    LwlStatus  retval  = LWL_SUCCESS;
    LwlStatus  link_retval;
    LwU32      i;

    FOR_EACH_INDEX_IN_MASK(64, i, p->linkMask)
    {
        if (!(LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, i) &&
            LWSWITCH_IS_LINK_ENG_VALID_SV10(device, LWLTLC, i)))
        {
            return -LWL_BAD_ARGS;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    FOR_EACH_INDEX_IN_MASK(64, i, p->linkMask)
    {
        if (p->bFatalError)
        {
            link_retval = _lwswitch_inject_link_error_fatal(device, i);
        }
        else
        {
            link_retval = _lwswitch_inject_link_error_rcvy(device, i);
        }

        if (link_retval != LWL_SUCCESS)
        {
            retval = link_retval;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return retval;
}

static LwlStatus
lwswitch_ctrl_get_lwlink_caps_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_CAPS_PARAMS *ret
)
{
    LwlStatus retval = LWL_SUCCESS;

    _lwswitch_set_lwlink_caps(&ret->capsTbl);

    ret->lowestLwlinkVersion    = LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_2_0;
    ret->highestLwlinkVersion   = LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_2_0;
    ret->lowestNciVersion       = LWSWITCH_LWLINK_CAPS_NCI_VERSION_2_0;
    ret->highestNciVersion      = LWSWITCH_LWLINK_CAPS_NCI_VERSION_2_0;

    ret->enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    return retval;
}

static LwlStatus
lwswitch_ctrl_clear_counters_sv10
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_CLEAR_COUNTERS_PARAMS *ret
)
{
    lwlink_link *link;
    LwU8 i;
    LwU32 counterMask;
    LwU32 data;

    counterMask = ret->counterMask;

    // Common usage allows one of these to stand for all of them
    if ((counterMask) & ( LWSWITCH_LWLINK_COUNTER_TL_TX0
                        | LWSWITCH_LWLINK_COUNTER_TL_TX1
                        | LWSWITCH_LWLINK_COUNTER_TL_RX0
                        | LWSWITCH_LWLINK_COUNTER_TL_RX1
                        ))
    {
        counterMask |= ( LWSWITCH_LWLINK_COUNTER_TL_TX0
                       | LWSWITCH_LWLINK_COUNTER_TL_TX1
                       | LWSWITCH_LWLINK_COUNTER_TL_RX0
                       | LWSWITCH_LWLINK_COUNTER_TL_RX1
                       );
    }

    // Common usage allows one of these to stand for all of them
    if ((counterMask) & ( LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7
                        | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY
                        | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY
                        ))
    {
        counterMask |= ( LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7
                       | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY
                       | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY
                       );
    }

    FOR_EACH_INDEX_IN_MASK(64, i, ret->linkMask)
    {
        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            continue;
        }

        if (LWSWITCH_IS_LINK_ENG_VALID_SV10(device, LWLTLC, link->linkNumber))
        {
            // TX
            data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR_CTRL);
            if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_TX0)
            {
                data = FLD_SET_DRF_NUM(_LWLTLC_TX, _DEBUG_TP_CNTR_CTRL, _RESETTX0, 0x1, data);
            }
            if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_TX1)
            {
                data = FLD_SET_DRF_NUM(_LWLTLC_TX, _DEBUG_TP_CNTR_CTRL, _RESETTX1, 0x1, data);
            }
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR_CTRL, data);

            // RX
            data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR_CTRL);
            if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_RX0)
            {
                data = FLD_SET_DRF_NUM(_LWLTLC_RX, _DEBUG_TP_CNTR_CTRL, _RESETRX0, 0x1, data);
            }
            if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_RX1)
            {
                data = FLD_SET_DRF_NUM(_LWLTLC_RX, _DEBUG_TP_CNTR_CTRL, _RESETRX1, 0x1, data);
            }
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR_CTRL, data);
        }

        if (LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
        {
            if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)
            {
                data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _ERROR_COUNT_CTRL);
                data = FLD_SET_DRF(_PLWL_SL1, _ERROR_COUNT_CTRL, _CLEAR_FLIT_CRC, _CLEAR, data);
                data = FLD_SET_DRF(_PLWL_SL1, _ERROR_COUNT_CTRL, _CLEAR_RATES, _CLEAR, data);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _ERROR_COUNT_CTRL, data);
            }

            if (counterMask & (LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7))
            {
                data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _ERROR_COUNT_CTRL);
                data = FLD_SET_DRF(_PLWL_SL1, _ERROR_COUNT_CTRL, _CLEAR_LANE_CRC, _CLEAR, data);
                data = FLD_SET_DRF(_PLWL_SL1, _ERROR_COUNT_CTRL, _CLEAR_RATES, _CLEAR, data);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _ERROR_COUNT_CTRL, data);
            }

            if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)
            {
                data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _ERROR_COUNT_CTRL);
                data = FLD_SET_DRF(_PLWL_SL0, _ERROR_COUNT_CTRL, _CLEAR_REPLAY, _CLEAR, data);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _ERROR_COUNT_CTRL, data);
            }

            if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)
            {
                data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _ERROR_COUNT_CTRL);
                data = FLD_SET_DRF(_PLWL, _ERROR_COUNT_CTRL, _CLEAR_RECOVERY, _CLEAR, data);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _ERROR_COUNT_CTRL, data);
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_get_err_info_sv10
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS *ret
)
{
    lwlink_link *link;
    LwU32 data;
    LwU8 i;

    ret->linkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, i, ret->linkMask)
    {
        link = lwswitch_get_link(device, i);
        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        // TODO LWpu TL not supported
        LWSWITCH_PRINT(device, WARN,
            "%s WARNING: Lwpu %s register %s does not exist!\n",
            __FUNCTION__, "LWLTL", "LW_LWLTL_TL_ERRLOG_REG");

        LWSWITCH_PRINT(device, WARN,
            "%s WARNING: Lwpu %s register %s does not exist!\n",
            __FUNCTION__, "LWLTL", "LW_LWLTL_TL_INTEN_REG");

        ret->linkErrInfo[i].TLErrlog = 0x0;
        ret->linkErrInfo[i].TLIntrEn = 0x0;

        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _SLSM_STATUS_TX);
        ret->linkErrInfo[i].DLSpeedStatusTx =
            DRF_VAL(_PLWL_SL0, _SLSM_STATUS_TX, _PRIMARY_STATE, data);

        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _SLSM_STATUS_RX);
        ret->linkErrInfo[i].DLSpeedStatusRx =
            DRF_VAL(_PLWL_SL1, _SLSM_STATUS_RX, _PRIMARY_STATE, data);

        data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _INTR);
        ret->linkErrInfo[i].bExcessErrorDL =
            !!DRF_VAL(_PLWL, _INTR, _RX_SHORT_ERROR_RATE, data);

        if (ret->linkErrInfo[i].bExcessErrorDL)
        {
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _INTR,
                DRF_NUM(_PLWL, _INTR, _RX_SHORT_ERROR_RATE, 0x1));
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_get_irq_info_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_IRQ_INFO_PARAMS *p
)
{
    // Set the mask to AND out during servicing in order to avoid int storm.
    p->maskInfoList[0].irqPendingOffset = LW_PSMC_INTR_LEGACY;
    p->maskInfoList[0].irqEnabledOffset = LW_PSMC_INTR_EN_LEGACY;
    p->maskInfoList[0].irqEnableOffset  = LW_PSMC_INTR_EN_SET_LEGACY;
    p->maskInfoList[0].irqDisableOffset = LW_PSMC_INTR_EN_CLR_LEGACY;

    p->maskInfoList[1].irqPendingOffset = LW_PSMC_INTR_FATAL;
    p->maskInfoList[1].irqEnabledOffset = LW_PSMC_INTR_EN_FATAL;
    p->maskInfoList[1].irqEnableOffset  = LW_PSMC_INTR_EN_SET_FATAL;
    p->maskInfoList[1].irqDisableOffset = LW_PSMC_INTR_EN_CLR_FATAL;

    p->maskInfoList[2].irqPendingOffset = LW_PSMC_INTR_NONFATAL;
    p->maskInfoList[2].irqEnabledOffset = LW_PSMC_INTR_EN_NONFATAL;
    p->maskInfoList[2].irqEnableOffset  = LW_PSMC_INTR_EN_SET_NONFATAL;
    p->maskInfoList[2].irqDisableOffset = LW_PSMC_INTR_EN_CLR_NONFATAL;

    p->maskInfoList[3].irqPendingOffset = LW_PSMC_INTR_CORRECTABLE;
    p->maskInfoList[3].irqEnabledOffset = LW_PSMC_INTR_EN_CORRECTABLE;
    p->maskInfoList[3].irqEnableOffset  = LW_PSMC_INTR_EN_SET_CORRECTABLE;
    p->maskInfoList[3].irqDisableOffset = LW_PSMC_INTR_EN_CLR_CORRECTABLE;

    p->maskInfoCount                    = 4;

    return LWL_SUCCESS;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

LwlStatus
lwswitch_init_soe_sv10
(
    lwswitch_device *device
)
{
    // Not supported in SV10
    LWSWITCH_ASSERT(0);
    return -LWL_ERR_NOT_SUPPORTED;
}

//
// This function auto creates the sv10 HAL connectivity from the LWSWITCH_INIT_HAL
// macro in haldef_lwswitch.h
//
// Note: All hal fns must be implemented for each chip.
//       There is no automatic stubbing here.
//
void lwswitch_setup_hal_sv10(lwswitch_device *device)
{
    device->chip_arch = LWSWITCH_GET_INFO_INDEX_ARCH_SV10;
    device->chip_impl = LWSWITCH_GET_INFO_INDEX_IMPL_SV10;

    LWSWITCH_INIT_HAL(device, sv10);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LWSWITCH_INIT_HAL_UNPUBLISHED(device, sv10);                             
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_INIT_HAL_LWCFG_LS10(device, sv10);                             
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

}
