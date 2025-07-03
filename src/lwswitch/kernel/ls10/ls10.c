/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
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

#include "ls10/ls10.h"
#include "ls10/clock_ls10.h"
#include "ls10/bus_ls10.h"
#include "ls10/fuse_ls10.h"
#include "ls10/inforom_ls10.h"
#include "ls10/minion_ls10.h"
#include "ls10/pmgr_ls10.h"
#include "ls10/therm_ls10.h"
#include "ls10/smbpbi_ls10.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "ls10/multicast_ls10.h"
#endif
#include "ls10/soe_ls10.h"

#include "lwswitch/ls10/dev_lws_top.h"
#include "lwswitch/ls10/dev_pri_masterstation_ip.h"
#include "lwswitch/ls10/dev_pri_hub_sys_ip.h"
#include "lwswitch/ls10/dev_fuse.h"
#include "lwswitch/ls10/dev_lwlsaw_ip.h"
#include "lwswitch/ls10/dev_lwltlc_ip.h"
#include "lwswitch/ls10/dev_lwldl_ip.h"
#include "lwswitch/ls10/dev_nport_ip.h"
#include "lwswitch/ls10/dev_route_ip.h"
#include "lwswitch/ls10/dev_nport_ip_addendum.h"
#include "lwswitch/ls10/dev_route_ip_addendum.h"
#include "lwswitch/ls10/dev_ingress_ip.h"
#include "lwswitch/ls10/dev_egress_ip.h"
#include "lwswitch/ls10/dev_tstate_ip.h"
#include "lwswitch/ls10/dev_sourcetrack_ip.h"
#include "lwswitch/ls10/dev_cpr_ip.h"
#include "lwswitch/ls10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/ls10/dev_minion_ip.h"
#include "lwswitch/ls10/dev_minion_ip_addendum.h" 
#include "lwswitch/ls10/dev_multicasttstate_ip.h"
#include "lwswitch/ls10/dev_reductiontstate_ip.h"

void *
lwswitch_alloc_chipdevice_ls10
(
    lwswitch_device *device
)
{
    void *chip_device;

    chip_device = lwswitch_os_malloc(sizeof(ls10_device));
    if (NULL != chip_device)
    {
        lwswitch_os_memset(chip_device, 0, sizeof(ls10_device));
    }

    device->chip_id = LW_PMC_BOOT_42_CHIP_ID_LS10;
    return(chip_device);
}

/*
 * @Brief : Initializes the PRI Ring
 *
 * @Description : An example of a function that we'd like to generate from SU.
 *
 * @paramin device    a reference to the device to initialize
 *
 * @returns             LWL_SUCCESS if the action succeeded
 */
LwlStatus
lwswitch_pri_ring_init_ls10
(
    lwswitch_device *device
)
{
    LwU32 checked_data;
    LwU32 command;
    LwBool keepPolling;
    LWSWITCH_TIMEOUT timeout;

    if (!IS_FMODEL(device))
    {
        command = LWSWITCH_ENG_RD32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRI_RING_INIT);
        if (FLD_TEST_DRF(_PPRIV_SYS, _PRI_RING_INIT, _STATUS, _ALIVE, command))
        {
            // _STATUS == ALIVE. Skipping
            return LWL_SUCCESS;
        }
        if (!FLD_TEST_DRF(_PPRIV_SYS, _PRI_RING_INIT, _STATUS, _ALIVE_IN_SAFE_MODE, command))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: -- Initial _STATUS (0x%x) != _ALIVE_IN_SAFE_MODE --\n",
                __FUNCTION__, DRF_VAL(_PPRIV_SYS, _PRI_RING_INIT, _STATUS, command));
            return -LWL_ERR_GENERIC;
        }

        // .Switch PRI Ring Init Sequence

        // *****

        // . [SW] Enumerate and start the PRI Ring

        LWSWITCH_ENG_WR32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRI_RING_INIT,
                               DRF_DEF(_PPRIV_SYS, _PRI_RING_INIT, _CMD, _ENUMERATE_AND_START));

        // . [SW] Wait for the command to complete

        if (IS_EMULATION(device))
        {
            lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
        }
        else
        {
            lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
        }

        do
        {
            keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;
            command = LWSWITCH_ENG_RD32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRI_RING_INIT);

            if ( FLD_TEST_DRF(_PPRIV_SYS,_PRI_RING_INIT,_CMD,_NONE,command) )
            {
                break;
            }
            if ( keepPolling == LW_FALSE )
            {
                LWSWITCH_PRINT(device, ERROR, "%s: -- Timeout waiting for _CMD == _NONE --\n", __FUNCTION__);
                return -LWL_ERR_GENERIC;
            }
        }
        while (keepPolling);

        // . [SW] Confirm PRI Ring initialized properly. Exelwting four reads to introduce a delay.

        if (IS_EMULATION(device))
        {
            lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
        }
        else
        {
            lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
        }

        do
        {
            keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;
            command = LWSWITCH_ENG_RD32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRI_RING_INIT);

            if ( FLD_TEST_DRF(_PPRIV_SYS, _PRI_RING_INIT, _STATUS, _ALIVE, command) )
            {
                break;
            }
            if ( keepPolling == LW_FALSE )
            {
                LWSWITCH_PRINT(device, ERROR, "%s: -- Timeout waiting for _STATUS == _ALIVE --\n", __FUNCTION__);
                return -LWL_ERR_GENERIC;
            }
        }
        while (keepPolling);

        // . [SW] PRI Ring Interrupt Status0 and Status1 should be clear unless there was an error.

        checked_data = LWSWITCH_ENG_RD32(device, PRI_MASTER_RS, , 0, _PPRIV_MASTER, _RING_INTERRUPT_STATUS0);
        if ( !FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0, _DISCONNECT_FAULT, 0x0, checked_data) )
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PPRIV_MASTER,_RING_INTERRUPT_STATUS0,_DISCONNECT_FAULT != 0x0\n", __FUNCTION__);
        }
        if ( !FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0, _GBL_WRITE_ERROR_FBP, 0x0, checked_data) )
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PPRIV_MASTER,_RING_INTERRUPT_STATUS0,_GBL_WRITE_ERROR_FBP != 0x0\n", __FUNCTION__);
        }
        if ( !FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0, _GBL_WRITE_ERROR_SYS, 0x0, checked_data) )
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PPRIV_MASTER,_RING_INTERRUPT_STATUS0,_GBL_WRITE_ERROR_SYS != 0x0\n", __FUNCTION__);
        }
        if ( !FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0, _OVERFLOW_FAULT, 0x0, checked_data) )
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PPRIV_MASTER,_RING_INTERRUPT_STATUS0,_OVERFLOW_FAULT != 0x0\n", __FUNCTION__);
        }
        if ( !FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0, _RING_START_CONN_FAULT, 0x0, checked_data) )
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PPRIV_MASTER,_RING_INTERRUPT_STATUS0,_RING_START_CONN_FAULT != 0x0\n", __FUNCTION__);
        }

        checked_data = LWSWITCH_ENG_RD32(device, PRI_MASTER_RS, , 0, _PPRIV_MASTER, _RING_INTERRUPT_STATUS1);
        if ( !FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS1, _GBL_WRITE_ERROR_GPC, 0x0, checked_data) )
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PPRIV_MASTER,_RING_INTERRUPT_STATUS1,_GBL_WRITE_ERROR_GPC != 0x0\n", __FUNCTION__);
        }

        // *****
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Destroys an LwSwitch hardware state
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
void
lwswitch_destroy_device_state_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    if (lwswitch_is_soe_supported(device))
    {
        lwswitch_soe_unregister_events(device);
    }

    if (chip_device != NULL)
    {
        if ((chip_device->latency_stats) != NULL)
        {
            lwswitch_os_free(chip_device->latency_stats);
        }

        if ((chip_device->ganged_link_table) != NULL)
        {
            lwswitch_os_free(chip_device->ganged_link_table);
        }

        lwswitch_free_chipdevice(device);
    }

    lwswitch_i2c_destroy(device);

    return;
}

LwlStatus
lwswitch_initialize_pmgr_ls10
(
    lwswitch_device *device
)
{
    // Init PMGR info
    lwswitch_init_pmgr_ls10(device);
    lwswitch_init_pmgr_devices_ls10(device);

    return LWL_SUCCESS;
}


LwlStatus
lwswitch_initialize_ip_wrappers_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_TIMEOUT timeout;
    LwBool keepPolling;
    LwU32 val;
    LwlStatus status = LWL_SUCCESS;

    //
    // Now that software knows the devices and addresses, it must take all
    // the wrapper modules out of reset.
    //

    // Enable LWLW CPR
    lwswitch_timeout_create(5 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);

    LWSWITCH_ENG_WR32(device, CPR, _BCAST, 0, _CPR_SYS, _RESET_CTRL,
        DRF_DEF(_CPR_SYS, _RESET_CTRL, _RESET_N, _DEASSERTED));
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        val = LWSWITCH_ENG_RD32(device, CPR, _BCAST, 0, _CPR_SYS, _RESET_CTRL);
        if (FLD_TEST_DRF(_CPR_SYS, _RESET_CTRL, _STATUS, _DEASSERTED, val))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);
    if (!FLD_TEST_DRF(_CPR_SYS, _RESET_CTRL, _STATUS, _DEASSERTED, val))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout waiting for CPR to come out of reset (_RESET_CTRL = 0x%x)\n",
            __FUNCTION__, val);
        return -LWL_IO_ERROR;
    }

    LWSWITCH_ENG_WR32(device, CPR, _BCAST, 0, _CPR_SYS, _CTLCLK_CTRL,
        DRF_DEF(_CPR_SYS, _CTLCLK_CTRL, _CTLCLK_SEL, _ON));
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        val = LWSWITCH_ENG_RD32(device, CPR, _BCAST, 0, _CPR_SYS, _CTLCLK_CTRL);
        if (FLD_TEST_DRF(_CPR_SYS, _CTLCLK_CTRL, _CTLCLK_STS, _ON, val))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);
    if (!FLD_TEST_DRF(_CPR_SYS, _CTLCLK_CTRL, _CTLCLK_STS, _ON, val))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout waiting for CPR clock to start (_CTLCLK_CTRL = 0x%x)\n",
            __FUNCTION__, val);
        return -LWL_IO_ERROR;;
    }

    // TODO: Enable LWLIPT & NPG and update discovered data for VMs
    status = lwswitch_lws_top_prod_ls10(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: TOP PROD initialization failed.\n",
            __FUNCTION__);
        return status;
    }

    status = lwswitch_npg_prod_ls10(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPG PROD initialization failed.\n",
            __FUNCTION__);
        return status;
    }

    status = lwswitch_apply_prod_lwlw_ls10(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: LWLW PROD initialization failed.\n",
            __FUNCTION__);
        return status;
    }

    status = lwswitch_apply_prod_nxbar_ls10(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NXBAR PROD initialization failed.\n",
            __FUNCTION__);
        return status;
    }

    return status;
}

void
lwswitch_set_ganged_link_table_ls10
(
    lwswitch_device *device,
    LwU32            firstIndex,
    LwU64           *ganged_link_table,
    LwU32            numEntries
)
{
    LwU32 i;

    LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _ROUTE, _REG_TABLE_ADDRESS,
        DRF_NUM(_ROUTE, _REG_TABLE_ADDRESS, _INDEX, firstIndex) |
        DRF_NUM(_ROUTE, _REG_TABLE_ADDRESS, _AUTO_INCR, 1));

    for (i = 0; i < numEntries; i++)
    {
        LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _ROUTE, _REG_TABLE_DATA1,
            LwU64_HI32(ganged_link_table[i]));

        // HW will fill in the ECC
        LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _ROUTE, _REG_TABLE_DATA2,
            0);

        //
        // Writing DATA0 triggers the latched data to be written to the table
        // So write it last
        //
        LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _ROUTE, _REG_TABLE_DATA0,
            LwU64_LO32(ganged_link_table[i]));
    }
}

static LwlStatus
_lwswitch_init_ganged_link_routing_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32        gang_size;
    LwU64        gang_entry;
    LwU32        glt_entries = 16;
    LwU32        glt_size = (LW_ROUTE_REG_TABLE_ADDRESS_INDEX_GLTAB_DEPTH + 1);
    LwU64        *ganged_link_table = NULL;
    LwU32        i;
    LwU32        glt_index;

    //
    // Refer to Laguna Seca IAS 16.2.4.12 Ganged RAM Table Format
    // https://p4viewer.lwpu.com/get/hw/doc/gpu/hopper/laguna/design/IAS/arch/publish/working/laguna_4P0_Full.html#ROUTE_REG_REG_TABLE_ADDRESS
    //
    // The ganged link routing table is composed of 256 entries of 64-bits in
    // size.  Each entry is divided into 16 4-bit fields GLX(i), where GLX(x)
    // contains the distribution pattern for x ports.  Zero ports is not a
    // valid configuration, so GLX(0) corresponds with 16 ports.
    // Each GLX(i) column therefore should contain a uniform distribution
    // pattern for i ports.
    //
    // The ganged link routing table will be loaded with following values:
    // (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
    // (1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1),
    // (2,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2),
    // (3,0,1,0,3,3,3,3,3,3,3,3,3,3,3,3),
    //  :
    // (E,0,0,2,2,4,2,2,6,2,4,1,2,7,2,E),
    // (F,0,1,0,3,0,3,3,7,3,5,2,3,8,3,0)
    //
    // Refer table 22: Definition of size bits used with Ganged Link Number Table.
    //

    //Alloc memory for Ganged Link Table
    ganged_link_table = lwswitch_os_malloc(glt_size * sizeof(gang_entry));
    if (ganged_link_table == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Failed to allocate memory for GLT!!\n");
        return -LWL_NO_MEM;
    }

    for (glt_index = 0; glt_index < glt_size; glt_index++)
    {
        gang_entry = 0;
        for (i = 0; i < glt_entries; i++)
        {
            gang_size = ((i==0) ? 16 : i);
            gang_entry |=
                DRF_NUM64(_ROUTE, _REG_TABLE_DATA0, _GLX(i), glt_index % gang_size);
        }

        ganged_link_table[glt_index] = gang_entry;
    }

    lwswitch_set_ganged_link_table_ls10(device, 0, ganged_link_table, glt_size);

    chip_device->ganged_link_table = ganged_link_table;

    return LWL_SUCCESS;
}

static void
_lwswitch_init_cmd_routing_ls10
(
    lwswitch_device *device
)
{
    LwU32 val;

    //Set Hash policy for the requests.
    val = DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN1, _SPRAY) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN2, _SPRAY) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN4, _SPRAY) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN7, _SPRAY);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _ROUTE, _CMD_ROUTE_TABLE0, val);

    // Set Random policy for reponses.
    val = DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN16, _RANDOM) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN17, _RANDOM);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _ROUTE, _CMD_ROUTE_TABLE2, val);
}

static LwlStatus
_lwswitch_init_portstat_counters_ls10
(
    lwswitch_device *device
)
{
    LwlStatus retval;
    LwU32 idx_channel;
    LWSWITCH_SET_LATENCY_BINS default_latency_bins;
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    chip_device->latency_stats = lwswitch_os_malloc(sizeof(LWSWITCH_LATENCY_STATS_LS10));
    if (chip_device->latency_stats == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed allocate memory for latency stats\n",
            __FUNCTION__);
        return -LWL_NO_MEM;
    }

    lwswitch_os_memset(chip_device->latency_stats, 0, sizeof(LWSWITCH_LATENCY_STATS_LS10));

    //
    // These bin thresholds are values provided by Arch based off
    // switch latency expectations.
    //
    for (idx_channel=0; idx_channel < LWSWITCH_NUM_VCS_LS10; idx_channel++)
    {
        default_latency_bins.bin[idx_channel].lowThreshold = 120;    // 120ns
        default_latency_bins.bin[idx_channel].medThreshold = 200;    // 200ns
        default_latency_bins.bin[idx_channel].hiThreshold  = 1000;   // 1us
    }

    //
    // 6 hour sample interval
    // The 48-bit counters can theoretically rollover after ~12 hours of full
    // throttle traffic.
    //
    chip_device->latency_stats->sample_interval_msec = 6 * 60 * 60 * 1000;

    retval = lwswitch_ctrl_set_latency_bins(device, &default_latency_bins);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to set latency bins\n",
            __FUNCTION__);
        LWSWITCH_ASSERT(0);
        return retval;
    }

    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _NPORT, _PORTSTAT_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_CONTROL, _RANGESELECT, _BITS13TO0));

    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _NPORT, _PORTSTAT_SOURCE_FILTER_0,
        DRF_NUM(_NPORT, _PORTSTAT_SOURCE_FILTER_0, _SRCFILTERBIT, 0xFFFFFFFF));

    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _NPORT, _PORTSTAT_SOURCE_FILTER_1,
        DRF_NUM(_NPORT, _PORTSTAT_SOURCE_FILTER_1, _SRCFILTERBIT, 0xFFFFFFFF));

    LWSWITCH_SAW_WR32_LS10(device, _LWLSAW, _GLBLLATENCYTIMERCTRL,
        DRF_DEF(_LWLSAW, _GLBLLATENCYTIMERCTRL, _ENABLE, _ENABLE));

    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _NPORT, _PORTSTAT_SNAP_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _DISABLE));

    // Start & Clear Residency Counters
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL,
        DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL,
        DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));

    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL,
        DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL,
        DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));

    // Start & Clear Stall/Busy Counters
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL,
        DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL,
        DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));

    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL,
        DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL,
        DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
        DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_initialize_route_ls10
(
    lwswitch_device *device
)
{
    LwlStatus retval;

    retval = _lwswitch_init_ganged_link_routing_ls10(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to initialize GLT\n",
            __FUNCTION__);
        goto lwswitch_initialize_route_exit;
    }

    _lwswitch_init_cmd_routing_ls10(device);

    // Initialize Portstat Counters
    retval = _lwswitch_init_portstat_counters_ls10(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to initialize portstat counters\n",
            __FUNCTION__);
        goto lwswitch_initialize_route_exit;
    }

    // TODO: Setup multicast/reductions

lwswitch_initialize_route_exit:
    return retval;
}

LwlStatus
lwswitch_ctrl_get_counters_ls10
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
    LwlStatus status;
    LwBool minion_enabled;

    ct_assert(LWSWITCH_NUM_LANES_LS10 <= LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE);

    link = lwswitch_get_link(device, ret->linkId);
    if ((link == NULL) ||
        !LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber))
    {
        return -LWL_BAD_ARGS;
    }

    minion_enabled = lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, link->linkNumber, MINION));

    counterMask = ret->counterMask;

    // Common usage allows one of these to stand for all of them
    if (counterMask & (LWSWITCH_LWLINK_COUNTER_TL_TX0 |
                       LWSWITCH_LWLINK_COUNTER_TL_TX1 |
                       LWSWITCH_LWLINK_COUNTER_TL_RX0 |
                       LWSWITCH_LWLINK_COUNTER_TL_RX1))
    {
        tx0TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_LO(0)),
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_HI(0)));
        if (LWBIT64(63) & tx0TlCount)
        {
            ret->bTx0TlCounterOverflow = LW_TRUE;
            tx0TlCount &= ~(LWBIT64(63));
        }

        tx1TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_LO(1)),
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_HI(1)));
        if (LWBIT64(63) & tx1TlCount)
        {
            ret->bTx1TlCounterOverflow = LW_TRUE;
            tx1TlCount &= ~(LWBIT64(63));
        }

        rx0TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_LO(0)),
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_HI(0)));
        if (LWBIT64(63) & rx0TlCount)
        {
            ret->bRx0TlCounterOverflow = LW_TRUE;
            rx0TlCount &= ~(LWBIT64(63));
        }

        rx1TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_LO(1)),
            LWSWITCH_LINK_OFFSET_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_HI(1)));
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
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_RX01, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _RX01, _FLIT_CRC_ERRORS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)]
            = data;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_MASKED)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_RX02, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _RX02, _MASKED_CRC_ERRORS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_MASKED)]
            = data;
    }
    data = 0x0;
    bLaneReversed = lwswitch_link_lane_reversed_ls10(device, link->linkNumber);

    for (laneId = 0; laneId < LWSWITCH_NUM_LANES_LS10; laneId++)
    {
        //
        // HW may reverse the lane ordering or it may be overridden by SW.
        // If so, ilwert the interpretation of the lane CRC errors.
        //
        i = (LwU8)((bLaneReversed) ? (LWSWITCH_NUM_LANES_LS10 - 1) - laneId : laneId);

        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_DB01, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(laneId))
        {
            val = BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(laneId));

            switch (i)
            {
                case 0:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_LWLSTAT, _DB01, _ERROR_COUNT_ERR_LANECRC_L0, data);
                    break;
                case 1:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_LWLSTAT, _DB01, _ERROR_COUNT_ERR_LANECRC_L1, data);
                    break;
            }
        }
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_TX09, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _TX09, _REPLAY_EVENTS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)]
            = data;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_LNK1, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _LNK1, _ERROR_COUNT1_RECOVERY_EVENTS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)]
            = data;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_RX00, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _RX00, _REPLAY_EVENTS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY)]
            = data;
    }

    if ((counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS) ||
        (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL))
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                LW_LWLSTAT_DB11, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        if (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS)
        {
            ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS)]
                = DRF_VAL(_LWLSTAT_DB11, _COUNT_PHY_REFRESH, _PASS, data);
        }

        if (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL)
        {
            ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL)]
                = DRF_VAL(_LWLSTAT_DB11, _COUNT_PHY_REFRESH, _FAIL, data);
        }
    }

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
static void
lwswitch_ctrl_clear_throughput_counters_ls10
(
    lwswitch_device *device,
    lwlink_link     *link,
    LwU32           counterMask
)
{
    LwU32 data;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLTLC, link->linkNumber))
    {
        return;
    }

    //
    // Common usage allows one of these to stand for all of them
    // If one field is defined: perform a clear on counters 0 & 1
    //

    if ((counterMask) & ( LWSWITCH_LWLINK_COUNTER_TL_TX0 |
                          LWSWITCH_LWLINK_COUNTER_TL_TX1 |
                          LWSWITCH_LWLINK_COUNTER_TL_RX0 |
                          LWSWITCH_LWLINK_COUNTER_TL_RX1 ))
    {
        // TX 0
        data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0(0));
        data = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _RESET, 0x1, data);
        LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0(0), data);

        // TX 1
        data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0(1));
        data = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _RESET, 0x1, data);
        LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0(1), data);

        // RX 0
        data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0(0));
        data = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _RESET, 0x1, data);
        LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0(0), data);

        // RX 1
        data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0(1));
        data = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _RESET, 0x1, data);
        LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0(1), data);
    }
}

static void
lwswitch_ctrl_clear_lp_counters_ls10
(
    lwswitch_device *device,
    lwlink_link     *link,
    LwU32           counterMask
)
{
    LwlStatus status;

    // Clears all LP counters
    if (counterMask & LWSWITCH_LWLINK_LP_COUNTERS_DL)
    {
        status = lwswitch_minion_send_command(device, link->linkNumber,
            LW_MINION_LWLINK_DL_CMD_COMMAND_DLSTAT_CLR_DLLPCNT, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "%s : Failed to clear lp counts to MINION for link # %d\n",
                __FUNCTION__, link->linkNumber);
        }
    }
}

static LwlStatus
lwswitch_ctrl_clear_dl_error_counters_ls10
(
    lwswitch_device *device,
    lwlink_link     *link,
    LwU32            counterMask
)
{
    LwU32           data;

    if ((!counterMask) ||
        (!(counterMask & (LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_ECC_COUNTS |
                          LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY |
                          LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY))))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Link%d: No error count clear request, counterMask (0x%x). Returning!\n",
            __FUNCTION__, link->linkNumber, counterMask);
        return LWL_SUCCESS;
    }

    // With Minion initialized, send command to minion
    if (lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, link->linkNumber, MINION)))
    {
        return lwswitch_minion_clear_dl_error_counters_ls10(device, link->linkNumber);
    }

    // With Minion not-initialized, perform with the registers
    if (counterMask & (LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6 |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7 |
                       LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY      |
                       LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY    |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT    |
                       LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_MASKED  ))
    {
        data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLDL, _LWLDL_RX, _ERROR_COUNT_CTRL);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_LANE_CRC, _CLEAR, data);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_FLIT_CRC, _CLEAR, data);
        data = FLD_SET_DRF(_LWLDL_TX, _ERROR_COUNT_CTRL, _CLEAR_REPLAY, _CLEAR, data);
        data = FLD_SET_DRF(_LWLDL_TOP, _ERROR_COUNT_CTRL, _CLEAR_RECOVERY, _CLEAR, data);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_RATES, _CLEAR, data);
        LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLDL, _LWLDL_RX, _ERROR_COUNT_CTRL, data);
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_ECC_COUNTS)
    {
        data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLDL, _LWLDL_RX, _ERROR_COUNT_CTRL);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_LANE_CRC, _CLEAR, data);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_RATES, _CLEAR, data);        
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_ECC_COUNTS, _CLEAR, data);
    }

    return LWL_SUCCESS;
}
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

static void
_lwswitch_portstat_reset_latency_counters_ls10
(
    lwswitch_device *device
)
{
    // Set SNAPONDEMAND from 0->1 to reset the counters
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _NPORT, _PORTSTAT_SNAP_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _ENABLE));

    // Set SNAPONDEMAND back to 0.
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _NPORT, _PORTSTAT_SNAP_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _DISABLE));
}

//
// Data collector which runs on a background thread, collecting latency stats.
//
// The latency counters have a maximum window period of about 12 hours
// (2^48 clk cycles). The counters reset after this period. So SW snaps
// the bins and records latencies every 6 hours. Setting SNAPONDEMAND from 0->1
// snaps the  latency counters and updates them to PRI registers for
// the SW to read. It then resets the counters to start collecting fresh latencies.
//

void
lwswitch_internal_latency_bin_log_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 idx_nport;
    LwU32 idx_vc;
    LwBool vc_valid;
    LwU64 lo, hi;
    LwU64 latency;
    LwU64 time_nsec;
    LwU32 link_type;    // Access or trunk link
    LwU64 last_visited_time_nsec;

    if (chip_device->latency_stats == NULL)
    {
        // Latency stat buffers not allocated yet
        return;
    }

    time_nsec = lwswitch_os_get_platform_time();
    last_visited_time_nsec = chip_device->latency_stats->last_visited_time_nsec;

    // Update last visited time
    chip_device->latency_stats->last_visited_time_nsec = time_nsec;

    // Compare time stamp and reset the counters if the snap is missed
    if (!IS_RTLSIM(device) || !IS_FMODEL(device))
    {
        if ((last_visited_time_nsec != 0) &&
            ((time_nsec - last_visited_time_nsec) >
             chip_device->latency_stats->sample_interval_msec * LWSWITCH_INTERVAL_1MSEC_IN_NS))
        {
            LWSWITCH_PRINT(device, ERROR,
                "Latency metrics recording interval missed.  Resetting counters.\n");
            _lwswitch_portstat_reset_latency_counters_ls10(device);
            return;
        }
    }

    for (idx_nport=0; idx_nport < LWSWITCH_LINK_COUNT(device); idx_nport++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, idx_nport))
        {
            continue;
        }

        // Setting SNAPONDEMAND from 0->1 snaps the latencies and resets the counters
        LWSWITCH_LINK_WR32_LS10(device, idx_nport, NPORT, _NPORT, _PORTSTAT_SNAP_CONTROL,
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _ENABLE));

        //
        // TODO: Check _STARTCOUNTER and don't log if counter not enabled.
        // Lwrrently all counters are always enabled
        //

        link_type = LWSWITCH_LINK_RD32_LS10(device, idx_nport, NPORT, _NPORT, _CTRL);
        for (idx_vc = 0; idx_vc < LWSWITCH_NUM_VCS_LS10; idx_vc++)
        {
            vc_valid = LW_FALSE;

            // VC's CREQ0(0) and RSP0(5) are relevant on access links.
            if (FLD_TEST_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _ACCESSLINK, link_type) &&
                ((idx_vc == LW_NPORT_VC_MAPPING_CREQ0) ||
                (idx_vc == LW_NPORT_VC_MAPPING_RSP0)))
            {
                vc_valid = LW_TRUE;
            }

            // VC's CREQ0(0), RSP0(5), CREQ1(6) and RSP1(7) are relevant on trunk links.
            if (FLD_TEST_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _TRUNKLINK, link_type) &&
                ((idx_vc == LW_NPORT_VC_MAPPING_CREQ0)  ||
                 (idx_vc == LW_NPORT_VC_MAPPING_RSP0)   ||
                 (idx_vc == LW_NPORT_VC_MAPPING_CREQ1)  ||
                 (idx_vc == LW_NPORT_VC_MAPPING_RSP1)))
            {
                vc_valid = LW_TRUE;
            }

            // If the VC is not being used, skip reading it
            if (!vc_valid)
            {
                continue;
            }

            lo = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _LOW, _0, idx_vc);
            hi = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _LOW, _1, idx_vc);
            latency = lo | (hi << 32);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].low += latency;

            lo = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _MEDIUM, _0, idx_vc);
            hi = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _MEDIUM, _1, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].medium += latency;
            latency = lo | (hi << 32);

            lo = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _HIGH, _0, idx_vc);
            hi = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _HIGH, _1, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].high += latency;
            latency = lo | (hi << 32);

            lo = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _PANIC, _0, idx_vc);
            hi = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _COUNT, _PANIC, _1, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].panic += latency;
            latency = lo | (hi << 32);

            lo = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _PACKET, _COUNT, _0, idx_vc);
            hi = LWSWITCH_NPORT_PORTSTAT_RD32_LS10(device, idx_nport, _PACKET, _COUNT, _1, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].count += latency;
            latency = lo | (hi << 32);

            // Note the time of this snap
            chip_device->latency_stats->latency[idx_vc].last_read_time_nsec = time_nsec;
            chip_device->latency_stats->latency[idx_vc].count++;
        }

        // Disable SNAPONDEMAND after fetching the latencies
        LWSWITCH_LINK_WR32_LS10(device, idx_nport, NPORT, _NPORT, _PORTSTAT_SNAP_CONTROL,
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _DISABLE));
    }
}

static LwlStatus
lwswitch_ctrl_set_ganged_link_table_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_GANGED_LINK_TABLE *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_set_nport_port_config_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_SWITCH_PORT_CONFIG *p
)
{
    LwU32   val;

    if (p->requesterLinkID >= LWBIT(
        DRF_SIZE(LW_NPORT_REQLINKID_REQROUTINGID) +
        DRF_SIZE(LW_NPORT_REQLINKID_REQROUTINGID_UPPER)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid requester RID 0x%x\n",
            __FUNCTION__, p->requesterLinkID);
        return -LWL_BAD_ARGS;
    }

    if (p->requesterLanID > DRF_MASK(LW_NPORT_REQLINKID_REQROUTINGLAN))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid requester RLAN 0x%x\n",
            __FUNCTION__, p->requesterLanID);
        return -LWL_BAD_ARGS;
    }

    val = LWSWITCH_LINK_RD32(device, p->portNum, NPORT, _NPORT, _CTRL);
    switch (p->type)
    {
        case CONNECT_ACCESS_GPU:
        case CONNECT_ACCESS_CPU:
        case CONNECT_ACCESS_SWITCH:
            val = FLD_SET_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _ACCESSLINK, val);
            break;
        case CONNECT_TRUNK_SWITCH:
            val = FLD_SET_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _TRUNKLINK, val);
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: invalid type #%d\n",
                __FUNCTION__, p->type);
            return -LWL_BAD_ARGS;
    }

    // _ENDPOINT_COUNT deprecated on LS10

    LWSWITCH_LINK_WR32(device, p->portNum, NPORT, _NPORT, _CTRL, val);

    LWSWITCH_LINK_WR32(device, p->portNum, NPORT, _NPORT, _REQLINKID,
        DRF_NUM(_NPORT, _REQLINKID, _REQROUTINGID, p->requesterLinkID) |
        DRF_NUM(_NPORT, _REQLINKID, _REQROUTINGID_UPPER,
            p->requesterLinkID >> DRF_SIZE(LW_NPORT_REQLINKID_REQROUTINGID)) |
        DRF_NUM(_NPORT, _REQLINKID, _REQROUTINGLAN, p->requesterLanID));

    return LWL_SUCCESS;
}

/*
 * @brief Returns the ingress requester link id.
 *
 * @param[in] device            lwswitch device
 * @param[in] params            LWSWITCH_GET_INGRESS_REQLINKID_PARAMS
 *
 * @returns                     LWL_SUCCESS if action succeeded,
 *                              -LWL_ERR_ILWALID_STATE invalid link
 */
LwlStatus
lwswitch_ctrl_get_ingress_reqlinkid_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *params
)
{
    LwU32 regval;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, params->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    regval = LWSWITCH_NPORT_RD32_LS10(device, params->portNum, _NPORT, _REQLINKID);
    params->requesterLinkID = DRF_VAL(_NPORT, _REQLINKID, _REQROUTINGID, regval) |
        (DRF_VAL(_NPORT, _REQLINKID, _REQROUTINGID_UPPER, regval) <<
            DRF_SIZE(LW_NPORT_REQLINKID_REQROUTINGID));

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_get_internal_latency_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_INTERNAL_LATENCY *pLatency
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 vc_selector = pLatency->vc_selector;
    LwU32 idx_nport;

    // Validate VC selector
    if (vc_selector >= LWSWITCH_NUM_VCS_LS10)
    {
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(pLatency, 0, sizeof(*pLatency));
    pLatency->vc_selector = vc_selector;

    // Snap up-to-the moment stats
    lwswitch_internal_latency_bin_log(device);

    for (idx_nport=0; idx_nport < LWSWITCH_LINK_COUNT(device); idx_nport++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, idx_nport))
        {
            continue;
        }

        pLatency->egressHistogram[idx_nport].low =
            chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].low;
        pLatency->egressHistogram[idx_nport].medium =
            chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].medium;
        pLatency->egressHistogram[idx_nport].high =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].high;
        pLatency->egressHistogram[idx_nport].panic =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].panic;
        pLatency->egressHistogram[idx_nport].count =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].count;
    }

    pLatency->elapsed_time_msec =
      (chip_device->latency_stats->latency[vc_selector].last_read_time_nsec -
       chip_device->latency_stats->latency[vc_selector].start_time_nsec)/1000000ULL;

    chip_device->latency_stats->latency[vc_selector].start_time_nsec =
        chip_device->latency_stats->latency[vc_selector].last_read_time_nsec;

    chip_device->latency_stats->latency[vc_selector].count = 0;

    // Clear aclwm_latency[]
    for (idx_nport = 0; idx_nport < LWSWITCH_LINK_COUNT(device); idx_nport++)
    {
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].low = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].medium = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].high = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].panic = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].count = 0;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_set_latency_bins_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_LATENCY_BINS *pLatency
)
{
    LwU32 vc_selector;
    const LwU32 freq_mhz = 1330;
    const LwU32 switchpll_hz = freq_mhz * 1000000ULL; // TODO: Verify this against POR clocks
    const LwU32 min_threshold = 10;   // Must be > zero to avoid div by zero
    const LwU32 max_threshold = 10000;

    // Quick input validation and ns to register value colwersion
    for (vc_selector = 0; vc_selector < LWSWITCH_NUM_VCS_LS10; vc_selector++)
    {
        if ((pLatency->bin[vc_selector].lowThreshold > max_threshold)                           ||
            (pLatency->bin[vc_selector].lowThreshold < min_threshold)                           ||
            (pLatency->bin[vc_selector].medThreshold > max_threshold)                           ||
            (pLatency->bin[vc_selector].medThreshold < min_threshold)                           ||
            (pLatency->bin[vc_selector].hiThreshold  > max_threshold)                           ||
            (pLatency->bin[vc_selector].hiThreshold  < min_threshold)                           ||
            (pLatency->bin[vc_selector].lowThreshold > pLatency->bin[vc_selector].medThreshold) ||
            (pLatency->bin[vc_selector].medThreshold > pLatency->bin[vc_selector].hiThreshold))
        {
            return -LWL_BAD_ARGS;
        }

        pLatency->bin[vc_selector].lowThreshold =
            switchpll_hz / (1000000000 / pLatency->bin[vc_selector].lowThreshold);
        pLatency->bin[vc_selector].medThreshold =
            switchpll_hz / (1000000000 / pLatency->bin[vc_selector].medThreshold);
        pLatency->bin[vc_selector].hiThreshold =
            switchpll_hz / (1000000000 / pLatency->bin[vc_selector].hiThreshold);

        LWSWITCH_PORTSTAT_BCAST_WR32_LS10(device, _LIMIT, _LOW,    vc_selector, pLatency->bin[vc_selector].lowThreshold);
        LWSWITCH_PORTSTAT_BCAST_WR32_LS10(device, _LIMIT, _MEDIUM, vc_selector, pLatency->bin[vc_selector].medThreshold);
        LWSWITCH_PORTSTAT_BCAST_WR32_LS10(device, _LIMIT, _HIGH,   vc_selector, pLatency->bin[vc_selector].hiThreshold);
    }

    return LWL_SUCCESS;
}

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
_lwswitch_get_engine_base_ls10
(
    lwswitch_device *device,
    LwU32   register_rw_engine,     // REGISTER_RW_ENGINE_*
    LwU32   instance,               // device instance
    LwBool  bcast,
    LwU32   *base_addr
)
{
    LwU32 base = 0;
    ENGINE_DISCOVERY_TYPE_LS10  *engine = NULL;
    LwlStatus retval = LWL_SUCCESS;
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

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

        case REGISTER_RW_ENGINE_FUSE:
        case REGISTER_RW_ENGINE_JTAG:
        case REGISTER_RW_ENGINE_PMGR:
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
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, SAW, instance))
                {
                    engine = &chip_device->engSAW[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_SOE:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, SOE, instance))
                {
                    engine = &chip_device->engSOE[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_SE:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, SE, instance))
                {
                    engine = &chip_device->engSE[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_CLKS_SYS:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, CLKS_SYS, instance))
                {
                    engine = &chip_device->engCLKS_SYS[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_CLKS_SYSB:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, CLKS_SYSB, instance))
                {
                    engine = &chip_device->engCLKS_SYSB[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_CLKS_P0:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, CLKS_P0_BCAST, instance))
                {
                    engine = &chip_device->engCLKS_P0_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, CLKS_P0, instance))
                {
                    engine = &chip_device->engCLKS_P0[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_XPL:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, XPL, instance))
                {
                    engine = &chip_device->engXPL[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_XTL:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, XTL, instance))
                {
                    engine = &chip_device->engXTL[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLW:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLW_BCAST, instance))
                {
                    engine = &chip_device->engLWLW_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLW, instance))
                {
                    engine = &chip_device->engLWLW[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_MINION:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, MINION_BCAST, instance))
                {
                    engine = &chip_device->engMINION_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, MINION, instance))
                {
                    engine = &chip_device->engMINION[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLIPT:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLIPT_BCAST, instance))
                {
                    engine = &chip_device->engLWLIPT_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLIPT, instance))
                {
                    engine = &chip_device->engLWLIPT[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLTLC:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLTLC_BCAST, instance))
                {
                    engine = &chip_device->engLWLTLC_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLTLC, instance))
                {
                    engine = &chip_device->engLWLTLC[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLTLC_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLTLC_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engLWLTLC_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLTLC_MULTICAST, instance))
                {
                    engine = &chip_device->engLWLTLC_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TX_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TX_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engTX_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TX_PERFMON, instance))
                {
                    engine = &chip_device->engTX_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TX_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TX_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engTX_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TX_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engTX_PERFMON_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_RX_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, RX_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engRX_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, RX_PERFMON, instance))
                {
                    engine = &chip_device->engRX_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_RX_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, RX_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engRX_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, RX_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engRX_PERFMON_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPG:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPG_BCAST, instance))
                {
                    engine = &chip_device->engNPG_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPG, instance))
                {
                    engine = &chip_device->engNPG[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT, instance))
                {
                    engine = &chip_device->engNPORT[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT_MULTICAST, instance))
                {
                    engine = &chip_device->engNPORT_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPG_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPG_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engNPG_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPG_PERFMON, instance))
                {
                    engine = &chip_device->engNPG_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT_PERFMON, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NPORT_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLIPT_LNK:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLIPT_LNK_BCAST, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLIPT_LNK, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLIPT_LNK_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLIPT_LNK_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLIPT_LNK_MULTICAST, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLDL:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLDL_BCAST, instance))
                {
                    engine = &chip_device->engLWLDL_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLDL, instance))
                {
                    engine = &chip_device->engLWLDL[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLDL_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLDL_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engLWLDL_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, LWLDL_MULTICAST, instance))
                {
                    engine = &chip_device->engLWLDL_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NXBAR:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NXBAR_BCAST, instance))
                {
                    engine = &chip_device->engNXBAR_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NXBAR, instance))
                {
                    engine = &chip_device->engNXBAR[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NXBAR_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NXBAR_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engNXBAR_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, NXBAR_PERFMON, instance))
                {
                    engine = &chip_device->engNXBAR_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE_BCAST, instance))
                {
                    engine = &chip_device->engTILE_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE, instance))
                {
                    engine = &chip_device->engTILE[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engTILE_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE_MULTICAST, instance))
                {
                    engine = &chip_device->engTILE_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engTILE_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE_PERFMON, instance))
                {
                    engine = &chip_device->engTILE_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engTILE_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LS10(device, TILE_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engTILE_PERFMON_MULTICAST[instance];
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
            if (bcast && (engine->disc_type == DISCOVERY_TYPE_BROADCAST))
            {
                //
                // Caveat emptor: A read of a broadcast register is
                // implementation-specific.
                //
                base = engine->info.bc.bc_addr;
            }
            else if ((!bcast) && (engine->disc_type == DISCOVERY_TYPE_UNICAST))
            {
                base = engine->info.uc.uc_addr;
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
lwswitch_ctrl_register_read_ls10
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_READ *p
)
{
    LwU32 base;
    LwU32 data;
    LwlStatus retval = LWL_SUCCESS;

    retval = _lwswitch_get_engine_base_ls10(device, p->engine, p->instance, LW_FALSE, &base);
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
lwswitch_ctrl_register_write_ls10
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_WRITE *p
)
{
    LwU32 base;
    LwlStatus retval = LWL_SUCCESS;

    retval = _lwswitch_get_engine_base_ls10(device, p->engine, p->instance, p->bcast, &base);
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

LwlStatus
lwswitch_get_lwlink_ecc_errors_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS *params
)
{
    LwU32 statData;
    LwU8 i, j;
    LwlStatus status;
    LwBool bLaneReversed;

    lwswitch_os_memset(params->errorLink, 0, sizeof(params->errorLink));

    FOR_EACH_INDEX_IN_MASK(64, i, params->linkMask)
    {
        lwlink_link         *link;
        LWSWITCH_LANE_ERROR *errorLane;
        LwU8                offset;
        LwBool              minion_enabled;
        LwU32               sublinkWidth;

        link = lwswitch_get_link(device, i);
        sublinkWidth = device->hal.lwswitch_get_sublink_width(device, i);

        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber) ||
            (i >= LWSWITCH_LINK_COUNT(device)))
        {
            return -LWL_BAD_ARGS;
        }

        minion_enabled = lwswitch_is_minion_initialized(device,
            LWSWITCH_GET_LINK_ENG_INST(device, link->linkNumber, MINION));

        bLaneReversed = lwswitch_link_lane_reversed_ls10(device, link->linkNumber);

        for (j = 0; j < LWSWITCH_NUM_LANES_LS10; j++)
        {
            if (minion_enabled && (j < sublinkWidth))
            {
                status = lwswitch_minion_get_dl_status(device, i,
                                        (LW_LWLSTAT_RX12 + j), 0, &statData);

                if (status != LWL_SUCCESS)
                {
                    return status;
                }
                offset = bLaneReversed ? ((sublinkWidth - 1) - j) : j;
                errorLane                = &params->errorLink[i].errorLane[offset];
                errorLane->valid         = LW_TRUE;
            }
            else
            {
                // MINION disabled
                statData                 = 0;
                offset                   = j;
                errorLane                = &params->errorLink[i].errorLane[offset];
                errorLane->valid         = LW_FALSE;
            }

            errorLane->eccErrorValue = DRF_VAL(_LWLSTAT, _RX12, _ECC_CORRECTED_ERR_L0_VALUE, statData);
            errorLane->overflowed    = DRF_VAL(_LWLSTAT, _RX12, _ECC_CORRECTED_ERR_L0_OVER, statData);
        }

        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, i,
                                    LW_LWLSTAT_RX11, 0, &statData);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
        }
        else 
        {
            statData = 0;
        }

        params->errorLink[i].eccDecFailed           = DRF_VAL(_LWLSTAT, _RX11, _ECC_DEC_FAILED_VALUE, statData);
        params->errorLink[i].eccDecFailedOverflowed = DRF_VAL(_LWLSTAT, _RX11, _ECC_DEC_FAILED_OVER, statData);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

LwU32
lwswitch_get_num_links_ls10
(
    lwswitch_device *device
)
{
    return LWSWITCH_NUM_LINKS_LS10;
}

static LwU32
lwswitch_get_latency_sample_interval_msec_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    return chip_device->latency_stats->sample_interval_msec;
}

static LwU32
lwswitch_get_device_dma_width_ls10
(
    lwswitch_device *device
)
{
    return DMA_ADDR_WIDTH_LS10;
}

static LwU32
lwswitch_get_link_ip_version_ls10
(
    lwswitch_device *device,
    LwU32            link_id
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 lwldl_instance;

    lwldl_instance = LWSWITCH_GET_LINK_ENG_INST(device, link_id, LWLDL);
    if (LWSWITCH_ENG_IS_VALID(device, LWLDL, lwldl_instance))
    {
        return chip_device->engLWLDL[lwldl_instance].version;
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: LWLink[0x%x] LWLDL instance invalid\n",
            __FUNCTION__, link_id);
        return 0;
    }
}

//
// TODO : Set this to 'LW_TRUE' once ls10-fmodel issue is resolved.
//        Tracked in bugs 3144287 and 200634104.
//
static LwBool
lwswitch_is_soe_supported_ls10
(
    lwswitch_device *device
)
{
    if (IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, INFO, "SOE is not yet supported on fmodel\n");
        return LW_FALSE;
    }

    return LW_TRUE;
}

/*
 * @Brief : Execute SOE pre-reset sequence for secure reset.
 *
 * Stubbing SOE pre-reset sequence on ls10.
 *
 * TODO: Review pre-reset sequence for ls10. Tracked in Bug#3450681
 *
 */
LwlStatus
lwswitch_soe_prepare_for_reset_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, INFO,
        "Skipping pre-reset sequence on LS10\n");
    return LWL_SUCCESS;
}

/*
 * @Brief : Checks if Inforom is supported
 *
 * Stubbing SOE Inforom support on ls10.
 *
 * TODO: Re-enable Inforom for ls10. Tracked in Bug#3450683
 *
 */
LwBool
lwswitch_is_inforom_supported_ls10
(
    lwswitch_device *device
)
{
    if (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, INFO,
            "INFOROM is not supported on non-silicon platform\n");
        return LW_FALSE;
    }

    if (!lwswitch_is_soe_supported(device))
    {
        LWSWITCH_PRINT(device, INFO,
            "INFOROM is not supported since SOE is not supported\n");
        return LW_FALSE;
    }

    LWSWITCH_PRINT(device, INFO,
        "INFOROM is not supported on LS10\n");
    return LW_FALSE;
}

/*
 * @Brief : Checks if Spi is supported
 *
 * Stubbing SOE Spi support on ls10.
 *
 * TODO: Re-enable Spi for ls10. Tracked in Bug#3450682
 *
 */
LwBool
lwswitch_is_spi_supported_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, INFO,
        "SPI is not supported on LS10\n");

    return LW_FALSE;
}

/*
 * @Brief : Check if SMBPBI is supported
 *
 */
LwBool
lwswitch_is_smbpbi_supported_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, INFO,
        "SMBPBI is not supported on LS10\n");

    return LW_FALSE;
}

/*
 * @Brief : Additional setup needed after blacklisted device initialization
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
void
lwswitch_post_init_blacklist_device_setup_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    return;
}

/*
* @brief: This function retrieves the LWLIPT public ID for a given global link idx
* @params[in]  device        reference to current lwswitch device
* @params[in]  linkId        link to retrieve LWLIPT public ID from
* @params[out] publicId      Public ID of LWLIPT owning linkId
*/
LwlStatus lwswitch_get_link_public_id_ls10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *publicId
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*
* @brief: This function retrieves the internal link idx for a given global link idx
* @params[in]  device        reference to current lwswitch device
* @params[in]  linkId        link to retrieve LWLIPT public ID from
* @params[out] localLinkIdx  Internal link index of linkId
*/
LwlStatus lwswitch_get_link_local_idx_ls10
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

    *localLinkIdx = LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LS10(linkId);

    return LWL_SUCCESS;
}

LwlStatus lwswitch_set_training_error_info_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *pLinkTrainingErrorInfoParams
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus lwswitch_ctrl_get_fatal_error_scope_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *pParams
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY
 */

LwlStatus
lwswitch_get_remap_table_selector_ls10
(
    lwswitch_device *device,
    LWSWITCH_TABLE_SELECT_REMAP table_selector,
    LwU32 *remap_ram_sel
)
{
    LwU32 ram_sel = 0;

    switch (table_selector)
    {
        case LWSWITCH_TABLE_SELECT_REMAP_PRIMARY:
            ram_sel = LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSNORMREMAPRAM;
            break;
        case LWSWITCH_TABLE_SELECT_REMAP_EXTA:
            ram_sel = LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSEXTAREMAPRAM;
            break;
        case LWSWITCH_TABLE_SELECT_REMAP_EXTB:
            ram_sel = LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSEXTBREMAPRAM;
            break;
        case LWSWITCH_TABLE_SELECT_REMAP_MULTICAST:
            ram_sel = LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECT_MULTICAST_REMAPRAM;
            break;
        default:
            LWSWITCH_PRINT(device, ERROR, "%s: invalid remap table selector (0x%x)\n",
                __FUNCTION__, table_selector);
            return -LWL_ERR_NOT_SUPPORTED;
            break;
    }

    if (remap_ram_sel)
    {
        *remap_ram_sel = ram_sel;
    }

    return LWL_SUCCESS;
}

LwU32
lwswitch_get_ingress_ram_size_ls10
(
    lwswitch_device *device,
    LwU32 ingress_ram_selector      // LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECT*
)
{
    LwU32 ram_size = 0;

    switch (ingress_ram_selector)
    {
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSNORMREMAPRAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_NORMREMAPTAB_DEPTH + 1;
            break;
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSEXTAREMAPRAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_EXTAREMAPTAB_DEPTH + 1;
            break;
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSEXTBREMAPRAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_EXTBREMAPTAB_DEPTH + 1;
            break;
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRIDROUTERAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_RID_TAB_DEPTH + 1;
            break;
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRLANROUTERAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_RLAN_TAB_DEPTH + 1;
            break;
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECT_MULTICAST_REMAPRAM:
            ram_size = LW_INGRESS_MCREMAPTABADDR_RAM_ADDRESS_MCREMAPTAB_DEPTH + 1;
            break;
        default:
            LWSWITCH_PRINT(device, ERROR, "%s: Unsupported ingress RAM selector (0x%x)\n",
                __FUNCTION__, ingress_ram_selector);
            break;
    }

    return ram_size;
}

static void
_lwswitch_set_remap_policy_ls10
(
    lwswitch_device *device,
    LwU32 portNum,
    LwU32 remap_ram_sel,
    LwU32 firstIndex,
    LwU32 numEntries,
    LWSWITCH_REMAP_POLICY_ENTRY *remap_policy
)
{
    LwU32 i;
    LwU32 remap_address;
    LwU32 address_base;
    LwU32 address_limit;
    LwU32 rfunc;

    LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, firstIndex) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, remap_ram_sel) |
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, _ENABLE));

    for (i = 0; i < numEntries; i++)
    {
        // Set each field if enabled, else set it to 0.
        remap_address = DRF_VAL64(_INGRESS, _REMAP, _ADDR_PHYS_LS10, remap_policy[i].address);
        address_base = DRF_VAL64(_INGRESS, _REMAP, _ADR_BASE_PHYS_LS10, remap_policy[i].addressBase);
        address_limit = DRF_VAL64(_INGRESS, _REMAP, _ADR_LIMIT_PHYS_LS10, remap_policy[i].addressLimit);
        rfunc = remap_policy[i].flags &
            (
                LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE
            );
        // Handle re-used RFUNC[5] conflict between Limerock and Laguna Seca
        if (rfunc & LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE)
        {
            //
            // RFUNC[5] Limerock functionality was deprecated and replaced with
            // a new function in Laguna Seca.  So fix RFUNC if needed.
            //

            rfunc &= ~LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE;
            rfunc |= LWBIT(5);
        }

        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA1,
            DRF_NUM(_INGRESS, _REMAPTABDATA1, _REQCTXT_MSK, remap_policy[i].reqCtxMask) |
            DRF_NUM(_INGRESS, _REMAPTABDATA1, _REQCTXT_CHK, remap_policy[i].reqCtxChk));
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA2,
            DRF_NUM(_INGRESS, _REMAPTABDATA2, _REQCTXT_REP, remap_policy[i].reqCtxRep));
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA3,
            DRF_NUM(_INGRESS, _REMAPTABDATA3, _ADR_BASE, address_base) |
            DRF_NUM(_INGRESS, _REMAPTABDATA3, _ADR_LIMIT, address_limit));
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA4,
            DRF_NUM(_INGRESS, _REMAPTABDATA4, _TGTID, remap_policy[i].targetId) |
            DRF_NUM(_INGRESS, _REMAPTABDATA4, _RFUNC, rfunc));
        // Get the upper bits of address_base/_limit
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA5,
            DRF_NUM(_INGRESS, _REMAPTABDATA5, _ADR_BASE,
                (address_base >> DRF_SIZE(LW_INGRESS_REMAPTABDATA3_ADR_BASE))) |
            DRF_NUM(_INGRESS, _REMAPTABDATA5, _ADR_LIMIT,
                (address_limit >> DRF_SIZE(LW_INGRESS_REMAPTABDATA3_ADR_LIMIT))));

        // Write last and auto-increment
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA0,
            DRF_NUM(_INGRESS, _REMAPTABDATA0, _RMAP_ADDR, remap_address) |
            DRF_NUM(_INGRESS, _REMAPTABDATA0, _IRL_SEL, remap_policy[i].irlSelect) |
            DRF_NUM(_INGRESS, _REMAPTABDATA0, _ACLVALID, remap_policy[i].entryValid));
    }
}

static void
_lwswitch_set_mc_remap_policy_ls10
(
    lwswitch_device *device,
    LwU32 portNum,
    LwU32 firstIndex,
    LwU32 numEntries,
    LWSWITCH_REMAP_POLICY_ENTRY *remap_policy
)
{
    LwU32 i;
    LwU32 remap_address;
    LwU32 address_base;
    LwU32 address_limit;
    LwU32 rfunc;
    LwU32 reflective;

    LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _MCREMAPTABADDR,
        DRF_NUM(_INGRESS, _MCREMAPTABADDR, _RAM_ADDRESS, firstIndex) |
        DRF_DEF(_INGRESS, _MCREMAPTABADDR, _AUTO_INCR, _ENABLE));

    for (i = 0; i < numEntries; i++)
    {
        // Set each field if enabled, else set it to 0.
        remap_address = DRF_VAL64(_INGRESS, _REMAP, _ADDR_PHYS_LS10, remap_policy[i].address);
        address_base = DRF_VAL64(_INGRESS, _REMAP, _ADR_BASE_PHYS_LS10, remap_policy[i].addressBase);
        address_limit = DRF_VAL64(_INGRESS, _REMAP, _ADR_LIMIT_PHYS_LS10, remap_policy[i].addressLimit);
        rfunc = remap_policy[i].flags &
            (
                LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE
            );
        // Handle re-used RFUNC[5] conflict between Limerock and Laguna Seca
        if (rfunc & LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE)
        {
            //
            // RFUNC[5] Limerock functionality was deprecated and replaced with
            // a new function in Laguna Seca.  So fix RFUNC if needed.
            //

            rfunc &= ~LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE;
            rfunc |= LWBIT(5);
        }
        reflective = (remap_policy[i].flags & LWSWITCH_REMAP_POLICY_FLAGS_REFLECTIVE ? 1 : 0);

        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _MCREMAPTABDATA1,
            DRF_NUM(_INGRESS, _MCREMAPTABDATA1, _REQCTXT_MSK, remap_policy[i].reqCtxMask) |
            DRF_NUM(_INGRESS, _MCREMAPTABDATA1, _REQCTXT_CHK, remap_policy[i].reqCtxChk));
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _MCREMAPTABDATA2,
            DRF_NUM(_INGRESS, _MCREMAPTABDATA2, _REQCTXT_REP, remap_policy[i].reqCtxRep));
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _MCREMAPTABDATA3,
            DRF_NUM(_INGRESS, _MCREMAPTABDATA3, _ADR_BASE, address_base) |
            DRF_NUM(_INGRESS, _MCREMAPTABDATA3, _ADR_LIMIT, address_limit));
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _MCREMAPTABDATA4,
            DRF_NUM(_INGRESS, _MCREMAPTABDATA4, _MCID, remap_policy[i].targetId) |
            DRF_NUM(_INGRESS, _MCREMAPTABDATA4, _RFUNC, rfunc) |
            DRF_NUM(_INGRESS, _MCREMAPTABDATA4, _ENB_REFLECT_MEM, reflective));
        // Get the upper bits of address_base/_limit
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _MCREMAPTABDATA5,
            DRF_NUM(_INGRESS, _MCREMAPTABDATA5, _ADR_BASE,
                (address_base >> DRF_SIZE(LW_INGRESS_MCREMAPTABDATA3_ADR_BASE))) |
            DRF_NUM(_INGRESS, _MCREMAPTABDATA5, _ADR_LIMIT,
                (address_limit >> DRF_SIZE(LW_INGRESS_MCREMAPTABDATA3_ADR_LIMIT))));

        // Write last and auto-increment
        LWSWITCH_LINK_WR32_LS10(device, portNum, NPORT, _INGRESS, _MCREMAPTABDATA0,
            DRF_NUM(_INGRESS, _MCREMAPTABDATA0, _RMAP_ADDR, remap_address) |
            DRF_NUM(_INGRESS, _MCREMAPTABDATA0, _IRL_SEL, remap_policy[i].irlSelect) |
            DRF_NUM(_INGRESS, _MCREMAPTABDATA0, _ACLVALID, remap_policy[i].entryValid));
    }
}

LwlStatus
lwswitch_ctrl_set_remap_policy_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY *p
)
{
    LwU32 i;
    LwU32 rfunc;
    LwU32 remap_ram_sel = ~0;
    LwU32 ram_size;
    LwlStatus retval = LWL_SUCCESS;

    //
    // This function is used to read both normal and multicast REMAP table,
    // so guarantee table definitions are identical.
    //
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA0_RMAP_ADDR) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA0_RMAP_ADDR));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA0_IRL_SEL) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA0_IRL_SEL));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA1_REQCTXT_MSK) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA1_REQCTXT_MSK));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA1_REQCTXT_CHK) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA1_REQCTXT_CHK));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA2_REQCTXT_REP) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA2_REQCTXT_REP));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA3_ADR_BASE) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA3_ADR_BASE));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA3_ADR_LIMIT) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA3_ADR_LIMIT));
//    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA4_TGTID) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA4_MCID));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA4_RFUNC) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA4_RFUNC));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA5_ADR_BASE) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA5_ADR_BASE));
    ct_assert(DRF_SIZE(LW_INGRESS_REMAPTABDATA5_ADR_LIMIT) == DRF_SIZE(LW_INGRESS_MCREMAPTABDATA5_ADR_LIMIT));

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "NPORT port #%d not valid\n",
            p->portNum);
        return -LWL_BAD_ARGS;
    }

    retval = lwswitch_get_remap_table_selector(device, p->tableSelect, &remap_ram_sel);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Remap table #%d not supported\n",
            p->tableSelect);
        return retval;
    }
    ram_size = lwswitch_get_ingress_ram_size(device, remap_ram_sel);

    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_REMAP_POLICY_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "remapPolicy[%d..%d] overflows range %d..%d or size %d.\n",
            p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_REMAP_POLICY_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    for (i = 0; i < p->numEntries; i++)
    {
        if (p->tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
        {
            if (p->remapPolicy[i].targetId &
                ~DRF_MASK(LW_INGRESS_MCREMAPTABDATA4_MCID))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "remapPolicy[%d].targetId 0x%x out of valid MCID range (0x%x..0x%x)\n",
                    i, p->remapPolicy[i].targetId,
                    0, DRF_MASK(LW_INGRESS_MCREMAPTABDATA4_MCID));
                return -LWL_BAD_ARGS;
            }
        }
        else
        {
            if (p->remapPolicy[i].targetId &
                ~DRF_MASK(LW_INGRESS_REMAPTABDATA4_TGTID))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "remapPolicy[%d].targetId 0x%x out of valid TGTID range (0x%x..0x%x)\n",
                    i, p->remapPolicy[i].targetId,
                    0, DRF_MASK(LW_INGRESS_REMAPTABDATA4_TGTID));
                return -LWL_BAD_ARGS;
            }
        }

        if (p->remapPolicy[i].irlSelect &
            ~DRF_MASK(LW_INGRESS_REMAPTABDATA0_IRL_SEL))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].irlSelect 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].irlSelect,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA0_IRL_SEL));
            return -LWL_BAD_ARGS;
        }

        rfunc = p->remapPolicy[i].flags &
            (
                LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE |
                LWSWITCH_REMAP_POLICY_FLAGS_REFLECTIVE
            );
        if (rfunc != p->remapPolicy[i].flags)
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].flags 0x%x has undefined flags (0x%x)\n",
                i, p->remapPolicy[i].flags,
                p->remapPolicy[i].flags ^ rfunc);
            return -LWL_BAD_ARGS;
        }
        if ((rfunc & LWSWITCH_REMAP_POLICY_FLAGS_REFLECTIVE) &&
            (p->tableSelect != LWSWITCH_TABLE_SELECT_REMAP_MULTICAST))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].flags: REFLECTIVE mapping only supported for MC REMAP\n",
                i);
            return -LWL_BAD_ARGS;
        }

        // Validate that only bits 51:39 are used
        if (p->remapPolicy[i].address &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADDR_PHYS_LS10))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].address 0x%llx & ~0x%llx != 0\n",
                i, p->remapPolicy[i].address,
                DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADDR_PHYS_LS10));
            return -LWL_BAD_ARGS;
        }

        if (p->remapPolicy[i].reqCtxMask &
           ~DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_MSK))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].reqCtxMask 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].reqCtxMask,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_MSK));
            return -LWL_BAD_ARGS;
        }

        if (p->remapPolicy[i].reqCtxChk &
            ~DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_CHK))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].reqCtxChk 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].reqCtxChk,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_CHK));
            return -LWL_BAD_ARGS;
        }

        if (p->remapPolicy[i].reqCtxRep &
            ~DRF_MASK(LW_INGRESS_REMAPTABDATA2_REQCTXT_REP))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].reqCtxRep 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].reqCtxRep,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA2_REQCTXT_REP));
            return -LWL_BAD_ARGS;
        }

        // Validate that only bits 38:21 are used
        if (p->remapPolicy[i].addressBase &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_BASE_PHYS_LS10))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].addressBase 0x%llx & ~0x%llx != 0\n",
                i, p->remapPolicy[i].addressBase,
                DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_BASE_PHYS_LS10));
            return -LWL_BAD_ARGS;
        }

        // Validate that only bits 38:21 are used
        if (p->remapPolicy[i].addressLimit &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_LIMIT_PHYS_LS10))
        {
            LWSWITCH_PRINT(device, ERROR,
                 "remapPolicy[%d].addressLimit 0x%llx & ~0x%llx != 0\n",
                 i, p->remapPolicy[i].addressLimit,
                 DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_LIMIT_PHYS_LS10));
            return -LWL_BAD_ARGS;
        }

        // Validate base & limit describe a region
        if (p->remapPolicy[i].addressBase > p->remapPolicy[i].addressLimit)
        {
            LWSWITCH_PRINT(device, ERROR,
                 "remapPolicy[%d].addressBase/Limit invalid: 0x%llx > 0x%llx\n",
                 i, p->remapPolicy[i].addressBase, p->remapPolicy[i].addressLimit);
            return -LWL_BAD_ARGS;
        }

        // Validate limit - base doesn't overflow 64G
        if ((p->remapPolicy[i].addressLimit - p->remapPolicy[i].addressBase) &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_OFFSET_PHYS_LS10))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].addressLimit 0x%llx - addressBase 0x%llx overflows 64GB\n",
                i, p->remapPolicy[i].addressLimit, p->remapPolicy[i].addressBase);
            return -LWL_BAD_ARGS;
        }

        // AddressOffset is deprecated in LS10 and later
        if (p->remapPolicy[i].addressOffset != 0)
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].addressOffset deprecated\n",
                i);
            return -LWL_BAD_ARGS;
        }
    }

    if (p->tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
    {
        _lwswitch_set_mc_remap_policy_ls10(device, p->portNum, p->firstIndex, p->numEntries, p->remapPolicy);
    }
    else
    {
        _lwswitch_set_remap_policy_ls10(device, p->portNum, remap_ram_sel, p->firstIndex, p->numEntries, p->remapPolicy);
    }

    return retval;
}

/*
 * CTRL_LWSWITCH_GET_REMAP_POLICY
 */

#define LWSWITCH_NUM_REMAP_POLICY_REGS_LS10 6

LwlStatus
lwswitch_ctrl_get_remap_policy_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_REMAP_POLICY_PARAMS *params
)
{
    LWSWITCH_REMAP_POLICY_ENTRY *remap_policy;
    LwU32 remap_policy_data[LWSWITCH_NUM_REMAP_POLICY_REGS_LS10]; // 6 word/REMAP table entry
    LwU32 table_index;
    LwU32 remap_count;
    LwU32 remap_address;
    LwU32 address_base;
    LwU32 address_limit;
    LwU32 remap_ram_sel;
    LwU32 ram_size;
    LwlStatus retval;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, params->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "NPORT port #%d not valid\n",
            params->portNum);
        return -LWL_BAD_ARGS;
    }

    retval = lwswitch_get_remap_table_selector(device, params->tableSelect, &remap_ram_sel);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Remap table #%d not supported\n",
            params->tableSelect);
        return retval;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, remap_ram_sel);
    if ((params->firstIndex >= ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: remapPolicy first index %d out of range[%d..%d].\n",
            __FUNCTION__, params->firstIndex, 0, ram_size - 1);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(params->entry, 0, (LWSWITCH_REMAP_POLICY_ENTRIES_MAX *
        sizeof(LWSWITCH_REMAP_POLICY_ENTRY)));

    table_index = params->firstIndex;
    remap_policy = params->entry;
    remap_count = 0;

    /* set table offset */
    if (params->tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
    {
        LWSWITCH_LINK_WR32_LS10(device, params->portNum, NPORT, _INGRESS, _MCREMAPTABADDR,
            DRF_NUM(_INGRESS, _MCREMAPTABADDR, _RAM_ADDRESS, params->firstIndex) |
            DRF_DEF(_INGRESS, _MCREMAPTABADDR, _AUTO_INCR, _ENABLE));
    }
    else
    {
        LWSWITCH_LINK_WR32_LS10(device, params->portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
            DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, params->firstIndex) |
            DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, remap_ram_sel) |
            DRF_DEF(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, _ENABLE));
    }

    while (remap_count < LWSWITCH_REMAP_POLICY_ENTRIES_MAX &&
        table_index < ram_size)
    {
        if (params->tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
        {
            remap_policy_data[0] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _MCREMAPTABDATA0);
            remap_policy_data[1] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _MCREMAPTABDATA1);
            remap_policy_data[2] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _MCREMAPTABDATA2);
            remap_policy_data[3] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _MCREMAPTABDATA3);
            remap_policy_data[4] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _MCREMAPTABDATA4);
            remap_policy_data[5] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _MCREMAPTABDATA5);
        }
        else
        {
            remap_policy_data[0] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA0);
            remap_policy_data[1] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA1);
            remap_policy_data[2] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA2);
            remap_policy_data[3] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA3);
            remap_policy_data[4] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA4);
            remap_policy_data[5] = LWSWITCH_LINK_RD32_LS10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA5);
        }

        /* add to remap_entries list if nonzero */
        if (remap_policy_data[0] || remap_policy_data[1] || remap_policy_data[2] ||
            remap_policy_data[3] || remap_policy_data[4] || remap_policy_data[5])
        {
            remap_policy[remap_count].irlSelect =
                DRF_VAL(_INGRESS, _REMAPTABDATA0, _IRL_SEL, remap_policy_data[0]);

            remap_policy[remap_count].entryValid =
                DRF_VAL(_INGRESS, _REMAPTABDATA0, _ACLVALID, remap_policy_data[0]);

            remap_address =
                DRF_VAL(_INGRESS, _REMAPTABDATA0, _RMAP_ADDR, remap_policy_data[0]);

            remap_policy[remap_count].address =
                DRF_NUM64(_INGRESS, _REMAP, _ADDR_PHYS_LS10, remap_address);

            remap_policy[remap_count].reqCtxMask =
                DRF_VAL(_INGRESS, _REMAPTABDATA1, _REQCTXT_MSK, remap_policy_data[1]);

            remap_policy[remap_count].reqCtxChk =
                DRF_VAL(_INGRESS, _REMAPTABDATA1, _REQCTXT_CHK, remap_policy_data[1]);

            remap_policy[remap_count].reqCtxRep =
                DRF_VAL(_INGRESS, _REMAPTABDATA2, _REQCTXT_REP, remap_policy_data[2]);

            remap_policy[remap_count].addressOffset = 0;

            address_base =
                DRF_VAL(_INGRESS, _REMAPTABDATA3, _ADR_BASE, remap_policy_data[3]) |
                (DRF_VAL(_INGRESS, _REMAPTABDATA5, _ADR_BASE, remap_policy_data[5]) <<
                    DRF_SIZE(LW_INGRESS_REMAPTABDATA3_ADR_BASE));

            remap_policy[remap_count].addressBase =
                DRF_NUM64(_INGRESS, _REMAP, _ADR_BASE_PHYS_LS10, address_base);

            address_limit =
                DRF_VAL(_INGRESS, _REMAPTABDATA3, _ADR_LIMIT, remap_policy_data[3]) |
                (DRF_VAL(_INGRESS, _REMAPTABDATA5, _ADR_LIMIT, remap_policy_data[5]) <<
                    DRF_SIZE(LW_INGRESS_REMAPTABDATA3_ADR_LIMIT));

            remap_policy[remap_count].addressLimit =
                DRF_NUM64(_INGRESS, _REMAP, _ADR_LIMIT_PHYS_LS10, address_limit);

            if (params->tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
            {
                remap_policy[remap_count].targetId =
                    DRF_VAL(_INGRESS, _MCREMAPTABDATA4, _MCID, remap_policy_data[4]);
            }
            else
            {
                remap_policy[remap_count].targetId =
                    DRF_VAL(_INGRESS, _REMAPTABDATA4, _TGTID, remap_policy_data[4]);
            }

            remap_policy[remap_count].flags =
                DRF_VAL(_INGRESS, _REMAPTABDATA4, _RFUNC, remap_policy_data[4]);
            // Handle re-used RFUNC[5] conflict between Limerock and Laguna Seca
            if (remap_policy[remap_count].flags & LWBIT(5))
            {
                remap_policy[remap_count].flags &= ~LWBIT(5);
                remap_policy[remap_count].flags |= LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE;
            }
            if (params->tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
            {
                if (FLD_TEST_DRF_NUM(_INGRESS, _MCREMAPTABDATA4, _ENB_REFLECT_MEM, 1, remap_policy_data[4]))
                {
                    remap_policy[remap_count].flags |= LWSWITCH_REMAP_POLICY_FLAGS_REFLECTIVE;
                }
            }

            remap_count++;
        }

        table_index++;
    }

    params->nextIndex = table_index;
    params->numEntries = remap_count;

    return LWL_SUCCESS;
}

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY_VALID
 */
LwlStatus
lwswitch_ctrl_set_remap_policy_valid_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY_VALID *p
)
{
    LwU32 remap_ram;
    LwU32 ram_address = p->firstIndex;
    LwU32 remap_policy_data[LWSWITCH_NUM_REMAP_POLICY_REGS_LS10]; // 6 word/REMAP table entry
    LwU32 i;
    LwU32 remap_ram_sel;
    LwU32 ram_size;
    LwlStatus retval;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT port #%d not valid\n",
            __FUNCTION__, p->portNum);
        return -LWL_BAD_ARGS;
    }

    retval = lwswitch_get_remap_table_selector(device, p->tableSelect, &remap_ram_sel);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Remap table #%d not supported\n",
            p->tableSelect);
        return retval;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, remap_ram_sel);
    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_REMAP_POLICY_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: remapPolicy[%d..%d] overflows range %d..%d or size %d.\n",
            __FUNCTION__, p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_REMAP_POLICY_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    if (p->tableSelect == LWSWITCH_TABLE_SELECT_REMAP_MULTICAST)
    {
        for (i = 0; i < p->numEntries; i++)
        {
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABADDR,
                DRF_NUM(_INGRESS, _MCREMAPTABADDR, _RAM_ADDRESS, ram_address++) |
                DRF_DEF(_INGRESS, _MCREMAPTABADDR, _AUTO_INCR, _DISABLE));

            remap_policy_data[0] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA0);
            remap_policy_data[1] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA1);
            remap_policy_data[2] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA2);
            remap_policy_data[3] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA3);
            remap_policy_data[4] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA4);
            remap_policy_data[5] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA5);

            // Set valid bit in REMAPTABDATA0.
            remap_policy_data[0] = FLD_SET_DRF_NUM(_INGRESS, _MCREMAPTABDATA0, _ACLVALID, p->entryValid[i], remap_policy_data[0]);

            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA5, remap_policy_data[5]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA4, remap_policy_data[4]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA3, remap_policy_data[3]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA2, remap_policy_data[2]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA1, remap_policy_data[1]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _MCREMAPTABDATA0, remap_policy_data[0]);
        }
    }
    else
    {
        // Select REMAP POLICY RAM and disable Auto Increment.
        remap_ram =
            DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, remap_ram_sel) |
            DRF_DEF(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, _DISABLE);

        for (i = 0; i < p->numEntries; i++)
        {
            /* set the ram address */
            remap_ram = FLD_SET_DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, ram_address++, remap_ram);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _REQRSPMAPADDR, remap_ram);

            remap_policy_data[0] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA0);
            remap_policy_data[1] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA1);
            remap_policy_data[2] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA2);
            remap_policy_data[3] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA3);
            remap_policy_data[4] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA4);
            remap_policy_data[5] = LWSWITCH_LINK_RD32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA5);

            // Set valid bit in REMAPTABDATA0.
            remap_policy_data[0] = FLD_SET_DRF_NUM(_INGRESS, _REMAPTABDATA0, _ACLVALID, p->entryValid[i], remap_policy_data[0]);

            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA5, remap_policy_data[5]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA4, remap_policy_data[4]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA3, remap_policy_data[3]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA2, remap_policy_data[2]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA1, remap_policy_data[1]);
            LWSWITCH_LINK_WR32_LS10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA0, remap_policy_data[0]);
        }
    }

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus lwswitch_ctrl_set_mc_rid_table_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_MC_RID_TABLE_PARAMS *p
)
{
    LwlStatus ret;
    LWSWITCH_MC_RID_ENTRY_LS10 table_entry;
    LwU32 entries_used = 0;

    if (!lwswitch_is_link_valid(device, p->portNum))
        return -LWL_BAD_ARGS;

    // check if link is invalid or repeater
    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: NPORT invalid for port %d\n",
                       __FUNCTION__, p->portNum);
        return -LWL_BAD_ARGS;
    }

    // range check index
    if (p->extendedTable && (p->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDEXTTAB_DEPTH))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for extended table\n",
                        __FUNCTION__, p->index);
        return -LWL_BAD_ARGS;
    }

    if (p->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDTAB_DEPTH)
    {
         LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for main table\n",
                        __FUNCTION__, p->index);
         return -LWL_BAD_ARGS;
    }

    // if !entryValid, zero the table and return
    if (!p->entryValid)
        return lwswitch_mc_ilwalidate_mc_rid_entry_ls10(device, p->portNum, p->index,
                                                        p->extendedTable, LW_TRUE);

    // range check mcSize
    if ((p->mcSize == 0) || (p->mcSize > LWSWITCH_NUM_LINKS_LS10))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: mcSize %d is invalid\n", __FUNCTION__, p->mcSize);
        return -LWL_BAD_ARGS;
    }

    // extended table cannot have an extended ptr
    if (p->extendedTable && p->extendedValid)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: extendedTable cannot have an extendedValid ptr\n",
                        __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    // set up table entry fields
    table_entry.index = (LwU8)p->index;
    table_entry.use_extended_table = p->extendedTable;
    table_entry.mcpl_size = (LwU8)p->mcSize;
    table_entry.num_spray_groups = (LwU8)p->numSprayGroups;
    table_entry.ext_ptr = (LwU8)p->extendedPtr;
    table_entry.no_dyn_rsp = p->noDynRsp;
    table_entry.ext_ptr_valid = p->extendedValid;
    table_entry.valid = p->entryValid;

    // build the directive list, remaining range checks are performed inside
    ret = lwswitch_mc_build_mcp_list_ls10(device, p->ports, p->portsPerSprayGroup, p->replicaOffset,
                                          p->replicaValid, p->vcHop, &table_entry, &entries_used);

    LWSWITCH_PRINT(device, INFO, "lwswitch_mc_build_mcp_list_ls10() returned %d, entries used: %d\n",
                   ret, entries_used);

    if (ret != LWL_SUCCESS)
        return ret;

    // program the table
    ret = lwswitch_mc_program_mc_rid_entry_ls10(device, p->portNum, &table_entry, entries_used);

    return ret;
}

LwlStatus lwswitch_ctrl_get_mc_rid_table_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_MC_RID_TABLE_PARAMS *p
)
{
    LwU32 ret;
    LWSWITCH_MC_RID_ENTRY_LS10 table_entry;
    LwU32 port = p->portNum;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, NPORT, port))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: NPORT invalid for port %d\n",
                       __FUNCTION__, port);
        return -LWL_BAD_ARGS;
    }

    // range check index
    if (p->extendedTable && (p->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDEXTTAB_DEPTH))
    {
        LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for extended table\n",
                        __FUNCTION__, p->index);
        return -LWL_BAD_ARGS;
    }

    if (p->index > LW_ROUTE_RIDTABADDR_INDEX_MCRIDTAB_DEPTH)
    {
         LWSWITCH_PRINT(device, ERROR, "%s: index %d out of range for main table\n",
                        __FUNCTION__, p->index);
         return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(&table_entry, 0, sizeof(LWSWITCH_MC_RID_ENTRY_LS10));

    table_entry.index = (LwU8)p->index;
    table_entry.use_extended_table = p->extendedTable;

    ret = lwswitch_mc_read_mc_rid_entry_ls10(device, port, &table_entry);
    if (ret != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: lwswitch_mc_read_mc_rid_entry_ls10() returned %d\n",
                        __FUNCTION__, ret);
        return ret;
    }

    lwswitch_os_memset(p, 0, sizeof(LWSWITCH_GET_MC_RID_TABLE_PARAMS));

    p->portNum = port;
    p->index = table_entry.index;
    p->extendedTable = table_entry.use_extended_table;

    ret = lwswitch_mc_unwind_directives_ls10(device, table_entry.directives, p->ports,
                                                p->vcHop, p->portsPerSprayGroup, p->replicaOffset,
                                                p->replicaValid);
    if (ret != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: lwswitch_mc_unwind_directives_ls10() returned %d\n",
                        __FUNCTION__, ret);
        return ret;
    }

    p->mcSize = table_entry.mcpl_size;
    p->numSprayGroups = table_entry.num_spray_groups;
    p->extendedPtr = table_entry.ext_ptr;
    p->noDynRsp = table_entry.no_dyn_rsp;
    p->extendedValid = table_entry.ext_ptr_valid;
    p->entryValid = table_entry.valid;

    return LWL_SUCCESS;
}
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

void lwswitch_init_scratch_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
}

static LWSWITCH_ENGINE_DESCRIPTOR_TYPE *
_lwswitch_get_eng_descriptor_ls10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine = NULL;

    if (eng_id >= LWSWITCH_ENGINE_ID_SIZE)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Engine_ID 0x%x out of range 0..0x%x\n",
            __FUNCTION__,
            eng_id, LWSWITCH_ENGINE_ID_SIZE-1);
        return NULL;
    }

    engine = &(chip_device->io.common[eng_id]);
    if (eng_id != engine->eng_id)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Requested Engine_ID 0x%x does not equal found Engine_ID 0x%x (%s)\n",
            __FUNCTION__,
            eng_id, engine->eng_id, engine->eng_name);
    }
    LWSWITCH_ASSERT(eng_id == engine->eng_id);

    return engine;
}

LwU32
lwswitch_get_eng_base_ls10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance
)
{
    LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine;
    LwU32 base_addr = LWSWITCH_BASE_ADDR_ILWALID;

    engine = _lwswitch_get_eng_descriptor_ls10(device, eng_id);
    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x[%d] %s not found\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        return LWSWITCH_BASE_ADDR_ILWALID;
    }

    if ((eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) &&
        (eng_instance < engine->eng_count))
    {
        base_addr = engine->uc_addr[eng_instance];
    }
    else if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST)
    {
        base_addr = engine->bc_addr;
    }
    else if ((eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) &&
        (eng_instance < engine->mc_addr_count))
    {
        base_addr = engine->mc_addr[eng_instance];
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unknown address space type 0x%x (not UC, BC, or MC)\n",
            __FUNCTION__,
            eng_bcast);
    }

    // The NPORT engine can be marked as invalid when it is in Repeater Mode
    if (base_addr == LWSWITCH_BASE_ADDR_ILWALID)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: ID 0x%x[%d] %s invalid address\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
    }

    return base_addr;
}

LwU32
lwswitch_get_eng_count_ls10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast
)
{
    LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine;
    LwU32 eng_count = 0;

    engine = _lwswitch_get_eng_descriptor_ls10(device, eng_id);
    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x %s not found\n",
            __FUNCTION__,
            eng_id,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        return 0;
    }

    if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST)
    {
        eng_count = engine->eng_count;
    }
    else if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST)
    {
        if (engine->bc_addr == LWSWITCH_BASE_ADDR_ILWALID)
        {
            eng_count = 0;
        }
        else
        {
            eng_count = 1;
        }
    }
    else if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST)
    {
        eng_count = engine->mc_addr_count;
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unknown address space type 0x%x (not UC, BC, or MC)\n",
            __FUNCTION__,
            eng_bcast);
    }

    return eng_count;
}

LwU32
lwswitch_eng_rd_ls10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance,
    LwU32 offset
)
{
    LwU32 base_addr = LWSWITCH_BASE_ADDR_ILWALID;
    LwU32 data;

    base_addr = lwswitch_get_eng_base_ls10(device, eng_id, eng_bcast, eng_instance);
    if (base_addr == LWSWITCH_BASE_ADDR_ILWALID)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x[%d] %s invalid address\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        LWSWITCH_ASSERT(base_addr != LWSWITCH_BASE_ADDR_ILWALID);
        return 0xBADFBADF;
    }

    data = lwswitch_reg_read_32(device, base_addr + offset);

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    {
        LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine = _lwswitch_get_eng_descriptor_ls10(device, eng_id);

        LWSWITCH_PRINT(device, MMIO,
            "%s: ENG_RD %s(0x%x)[%d] @0x%08x+0x%06x = 0x%08x\n",
            __FUNCTION__,
            engine->eng_name, engine->eng_id,
            eng_instance,
            base_addr, offset,
            data);
    }
#endif  //defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

    return data;
}

void
lwswitch_eng_wr_ls10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance,
    LwU32 offset,
    LwU32 data
)
{
    LwU32 base_addr = LWSWITCH_BASE_ADDR_ILWALID;

    base_addr = lwswitch_get_eng_base_ls10(device, eng_id, eng_bcast, eng_instance);
    if (base_addr == LWSWITCH_BASE_ADDR_ILWALID)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x[%d] %s invalid address\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        LWSWITCH_ASSERT(base_addr != LWSWITCH_BASE_ADDR_ILWALID);
        return;
    }

    lwswitch_reg_write_32(device, base_addr + offset,  data);

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    {
        LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine = _lwswitch_get_eng_descriptor_ls10(device, eng_id);

        LWSWITCH_PRINT(device, MMIO,
            "%s: ENG_WR %s(0x%x)[%d] @0x%08x+0x%06x = 0x%08x\n",
            __FUNCTION__,
            engine->eng_name, engine->eng_id,
            eng_instance,
            base_addr, offset,
            data);
    }
#endif  //defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
}

LwU32
lwswitch_get_link_eng_inst_ls10
(
    lwswitch_device *device,
    LwU32 link_id,
    LWSWITCH_ENGINE_ID eng_id
)
{
    LwU32   eng_instance = LWSWITCH_ENGINE_INSTANCE_ILWALID;

    if (link_id >= LWSWITCH_LINK_COUNT(device))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link ID 0x%x out-of-range [0x0..0x%x]\n",
            __FUNCTION__,
            link_id, LWSWITCH_LINK_COUNT(device)-1);
        return LWSWITCH_ENGINE_INSTANCE_ILWALID;
    }

    switch (eng_id)
    {
        case LWSWITCH_ENGINE_ID_NPG:
            eng_instance = link_id / LWSWITCH_LINKS_PER_NPG_LS10;
            break;
        case LWSWITCH_ENGINE_ID_LWLIPT:
            eng_instance = link_id / LWSWITCH_LINKS_PER_LWLIPT_LS10;
            break;
        case LWSWITCH_ENGINE_ID_LWLW:
        case LWSWITCH_ENGINE_ID_LWLW_PERFMON:
            eng_instance = link_id / LWSWITCH_LINKS_PER_LWLW_LS10;
            break;
        case LWSWITCH_ENGINE_ID_MINION:
            eng_instance = link_id / LWSWITCH_LINKS_PER_MINION_LS10;
            break;
        case LWSWITCH_ENGINE_ID_NPORT:
        case LWSWITCH_ENGINE_ID_LWLTLC:
        case LWSWITCH_ENGINE_ID_LWLDL:
        case LWSWITCH_ENGINE_ID_LWLIPT_LNK:
        case LWSWITCH_ENGINE_ID_NPORT_PERFMON:
        case LWSWITCH_ENGINE_ID_RX_PERFMON:
        case LWSWITCH_ENGINE_ID_TX_PERFMON:
            eng_instance = link_id;
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: link ID 0x%x has no association with EngID 0x%x\n",
                __FUNCTION__,
                link_id, eng_id);
            eng_instance = LWSWITCH_ENGINE_INSTANCE_ILWALID;
            break;
    }

    return eng_instance;
}

LwU32
lwswitch_get_caps_lwlink_version_ls10
(
    lwswitch_device *device
)
{
    return LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_4_0;
}


LWSWITCH_BIOS_LWLINK_CONFIG *
lwswitch_get_bios_lwlink_config_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    return &chip_device->bios_config;
}


static void
_lwswitch_init_nport_ecc_control_ls10
(
    lwswitch_device *device
)
{
    // Set ingress ECC error limits
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER,
        DRF_NUM(_INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set egress ECC error limits
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _EGRESS, _ERR_NXBAR_ECC_ERROR_COUNTER,
        DRF_NUM(_EGRESS, _ERR_NXBAR_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _EGRESS, _ERR_NXBAR_ECC_ERROR_COUNTER_LIMIT, 1);

    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _EGRESS, _ERR_RAM_OUT_ECC_ERROR_COUNTER,
        DRF_NUM(_EGRESS, _ERR_RAM_OUT_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _EGRESS, _ERR_RAM_OUT_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set route ECC error limits
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _ROUTE, _ERR_LWS_ECC_ERROR_COUNTER,
        DRF_NUM(_ROUTE, _ERR_LWS_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _ROUTE, _ERR_LWS_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set tstate ECC error limits
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER,
        DRF_NUM(_TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT, 1);

    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER,
        DRF_NUM(_TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set sourcetrack ECC error limits to _PROD value
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _SOURCETRACK, _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT,
        DRF_NUM(_SOURCETRACK, _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _SOURCETRACK, _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT, 1);

    // Enable ECC/parity
    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _INGRESS, _ERR_ECC_CTRL,
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _NCISOC_HDR_ECC_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _REMAPTAB_ECC_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _RIDTAB_ECC_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _RLANTAB_ECC_ENABLE, __PROD));

    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _EGRESS, _ERR_ECC_CTRL,
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NXBAR_ECC_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NXBAR_PARITY_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _RAM_OUT_ECC_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NCISOC_ECC_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, __PROD));

    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _ROUTE, _ERR_ECC_CTRL,
        DRF_DEF(_ROUTE, _ERR_ECC_CTRL, _GLT_ECC_ENABLE, __PROD) |
        DRF_DEF(_ROUTE, _ERR_ECC_CTRL, _LWS_ECC_ENABLE, __PROD));

    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _TSTATE, _ERR_ECC_CTRL,
        DRF_DEF(_TSTATE, _ERR_ECC_CTRL, _CRUMBSTORE_ECC_ENABLE, __PROD) |
        DRF_DEF(_TSTATE, _ERR_ECC_CTRL, _TAGPOOL_ECC_ENABLE, __PROD));

    LWSWITCH_ENG_WR32(device, NPORT, _BCAST, 0, _SOURCETRACK, _ERR_ECC_CTRL,
        DRF_DEF(_SOURCETRACK, _ERR_ECC_CTRL, _CREQ_TCEN0_CRUMBSTORE_ECC_ENABLE, __PROD));
}

LwlStatus
lwswitch_init_nport_ls10
(
    lwswitch_device *device
)
{
    LwU32 data32, timeout;
    LwU32 idx_nport;
    LwU32 num_nports;

    num_nports = LWSWITCH_ENG_COUNT(device, NPORT, );

    for (idx_nport = 0; idx_nport < num_nports; idx_nport++)
    {
        // Find the first valid nport
        if (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_nport))
        {
            break;
        }
    }

    // This is a valid case, since all NPORTs can be in Repeater mode.
    if (idx_nport == num_nports)
    {
        LWSWITCH_PRINT(device, INFO, "%s: No valid nports found! Skipping.\n", __FUNCTION__);
        return LWL_SUCCESS;
    }

    _lwswitch_init_nport_ecc_control_ls10(device);

    if (DRF_VAL(_SWITCH_REGKEY, _ATO_CONTROL, _DISABLE, device->regkeys.ato_control) ==
        LW_SWITCH_REGKEY_ATO_CONTROL_DISABLE_TRUE)
    {
        // ATO Disable
        data32 = LWSWITCH_NPORT_RD32_LS10(device, idx_nport, _TSTATE, _TAGSTATECONTROL);
        data32 = FLD_SET_DRF(_TSTATE, _TAGSTATECONTROL, _ATO_ENB, _OFF, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _TSTATE, _TAGSTATECONTROL, data32);
    }
    else
    {
        // ATO Enable
        data32 = LWSWITCH_NPORT_RD32_LS10(device, idx_nport, _TSTATE, _TAGSTATECONTROL);
        data32 = FLD_SET_DRF(_TSTATE, _TAGSTATECONTROL, _ATO_ENB, _ON, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _TSTATE, _TAGSTATECONTROL, data32);

        // ATO Timeout value
        timeout = DRF_VAL(_SWITCH_REGKEY, _ATO_CONTROL, _TIMEOUT, device->regkeys.ato_control);
        if (timeout != LW_SWITCH_REGKEY_ATO_CONTROL_TIMEOUT_DEFAULT)
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _TSTATE, _ATO_TIMER_LIMIT,
                DRF_NUM(_TSTATE, _ATO_TIMER_LIMIT, _LIMIT, timeout));
        }
    }

    if (DRF_VAL(_SWITCH_REGKEY, _STO_CONTROL, _DISABLE, device->regkeys.sto_control) ==
        LW_SWITCH_REGKEY_STO_CONTROL_DISABLE_TRUE)
    {
        // STO Disable
        data32 = LWSWITCH_NPORT_RD32_LS10(device, idx_nport, _SOURCETRACK, _CTRL);
        data32 = FLD_SET_DRF(_SOURCETRACK, _CTRL, _STO_ENB, _DISABLED, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _SOURCETRACK, _CTRL, data32);
    }
    else
    {
        // STO Enable
        data32 = LWSWITCH_NPORT_RD32_LS10(device, idx_nport, _SOURCETRACK, _CTRL);
        data32 = FLD_SET_DRF(_SOURCETRACK, _CTRL, _STO_ENB, _ENABLED, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _SOURCETRACK, _CTRL, data32);

        // STO Timeout value
        timeout = DRF_VAL(_SWITCH_REGKEY, _STO_CONTROL, _TIMEOUT, device->regkeys.sto_control);
        if (timeout != LW_SWITCH_REGKEY_STO_CONTROL_TIMEOUT_DEFAULT)
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _SOURCETRACK, _MULTISEC_TIMER0,
                DRF_NUM(_SOURCETRACK, _MULTISEC_TIMER0, _TIMERVAL0, timeout));
        }
    }

    //
    // Bug 3115824
    // Clear CONTAIN_AND_DRAIN during init for links in reset.
    // Since SBR does not clear CONTAIN_AND_DRAIN, this will clear the bit
    // when the driver is reloaded after an SBR. If the driver has been reloaded
    // without an SBR, then CONTAIN_AND_DRAIN will be re-triggered.
    //
    LWSWITCH_NPORT_MC_BCAST_WR32_LS10(device, _NPORT, _CONTAIN_AND_DRAIN,
        DRF_DEF(_NPORT, _CONTAIN_AND_DRAIN, _CLEAR, _ENABLE));

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_init_nxbar_ls10
(
    lwswitch_device *device
)
{
    LwlStatus status = LWL_SUCCESS;

    status = lwswitch_apply_prod_nxbar_ls10(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NXBAR PRODs failed\n",
            __FUNCTION__);
        return status;
    }

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_clear_nport_rams_ls10
(
    lwswitch_device *device
)
{
    LwU32 idx_nport;
    LwU64 nport_mask = 0;
    LwU32 zero_init_mask;
    LwU32 val;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    // Build the mask of available NPORTs
    for (idx_nport = 0; idx_nport < LWSWITCH_ENG_COUNT(device, NPORT, ); idx_nport++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_nport))
        {
            nport_mask |= LWBIT64(idx_nport);
        }
    }

    // Start the HW zero init
    zero_init_mask =
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_0, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _LINKTABLEINIT, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _REMAPTABINIT,  _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _RIDTABINIT,    _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _RLANTABINIT,   _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _MCREMAPTABINIT, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _MCTAGSTATEINIT, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _RDTAGSTATEINIT, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _MCREDSGTINIT,  _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _MCREDBUFINIT,  _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _MCRIDINIT,     _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _EXTMCRIDINIT,  _HWINIT);

    LWSWITCH_BCAST_WR32_LS10(device, NPORT, _NPORT, _INITIALIZATION,
        zero_init_mask);

    lwswitch_timeout_create(25 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        // Check each enabled NPORT that is still pending until all are done
        for (idx_nport = 0; idx_nport < LWSWITCH_ENG_COUNT(device, NPORT, ); idx_nport++)
        {
            if (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_nport) && (nport_mask & LWBIT64(idx_nport)))
            {
                val = LWSWITCH_ENG_RD32_LS10(device, NPORT, idx_nport, _NPORT, _INITIALIZATION);
                if (val == zero_init_mask)
                {
                    nport_mask &= ~LWBIT64(idx_nport);
                }
            }
        }

        if (nport_mask == 0)
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (nport_mask != 0)
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: Timeout waiting for LW_NPORT_INITIALIZATION (0x%llx)\n",
            __FUNCTION__, nport_mask);
        return -LWL_ERR_ILWALID_STATE;
    }

    return LWL_SUCCESS;
}

/*
 * CTRL_LWSWITCH_SET_RESIDENCY_BINS
 */
static LwlStatus
lwswitch_ctrl_set_residency_bins_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_RESIDENCY_BINS *p
)
{
    LwU64 threshold;
    LwU64 max_threshold;

    if (p->bin.lowThreshold > p->bin.hiThreshold )
    {
        LWSWITCH_PRINT(device, ERROR,
            "SET_RESIDENCY_BINS: Low threshold (%d) > Hi threshold (%d)\n",
            p->bin.lowThreshold, p->bin.hiThreshold);
        return -LWL_BAD_ARGS;
    }

    if (p->table_select == LWSWITCH_TABLE_SELECT_MULTICAST)
    {
        max_threshold = DRF_MASK(LW_MULTICASTTSTATE_STAT_RESIDENCY_BIN_CTRL_HIGH_LIMIT);

        threshold = (LwU64) p->bin.hiThreshold * 1333 / 1000;
        if (threshold > max_threshold)
        {
            LWSWITCH_PRINT(device, ERROR,
                "SET_RESIDENCY_BINS: Threshold overflow.  %u > %llu max\n",
                p->bin.hiThreshold, max_threshold * 1000 / 1333);
            return -LWL_BAD_ARGS;
        }
        LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH,
            DRF_NUM(_MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH, _LIMIT, (LwU32)threshold));

        threshold = (LwU64)p->bin.lowThreshold * 1333 / 1000;
        LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW,
            DRF_NUM(_MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW, _LIMIT, (LwU32)threshold));
    }
    else if (p->table_select == LWSWITCH_TABLE_SELECT_REDUCTION)
    {
        max_threshold = DRF_MASK(LW_REDUCTIONTSTATE_STAT_RESIDENCY_BIN_CTRL_HIGH_LIMIT);

        threshold = (LwU64) p->bin.hiThreshold * 1333 / 1000;
        if (threshold > max_threshold)
        {
            LWSWITCH_PRINT(device, ERROR,
                "SET_RESIDENCY_BINS: Threshold overflow.  %u > %llu max\n",
                p->bin.hiThreshold, max_threshold * 1000 / 1333);
            return -LWL_BAD_ARGS;
        }
        LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH,
            DRF_NUM(_REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH, _LIMIT, (LwU32)threshold));

        threshold = (LwU64)p->bin.lowThreshold * 1333 / 1000;
        LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW,
            DRF_NUM(_REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW, _LIMIT, (LwU32)threshold));
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "SET_RESIDENCY_BINS: Invalid table %d\n", p->table_select);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return LWL_SUCCESS;
}

#define LWSWITCH_RESIDENCY_BIN_SIZE                                 \
    ((LW_MULTICASTTSTATE_STAT_RESIDENCY_COUNT_CTRL_INDEX_MAX + 1) / \
     LW_MULTICASTTSTATE_STAT_RESIDENCY_COUNT_CTRL_INDEX_MCID_STRIDE)

/*
 * CTRL_LWSWITCH_GET_RESIDENCY_BINS
 */
static LwlStatus
lwswitch_ctrl_get_residency_bins_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_RESIDENCY_BINS *p
)
{
    LwU64 val;
    LwU64 val_hi;
    LwU32 i;
    LwU64 threshold;

    ct_assert(
        LW_MULTICASTTSTATE_STAT_RESIDENCY_COUNT_CTRL_INDEX_MCID_STRIDE ==
        LW_REDUCTIONTSTATE_STAT_RESIDENCY_COUNT_CTRL_INDEX_MCID_STRIDE);
    ct_assert(
        LW_MULTICASTTSTATE_STAT_RESIDENCY_COUNT_CTRL_INDEX_MAX ==
        LW_REDUCTIONTSTATE_STAT_RESIDENCY_COUNT_CTRL_INDEX_MAX);

    ct_assert(LWSWITCH_RESIDENCY_BIN_SIZE == LWSWITCH_RESIDENCY_SIZE);

    if (!lwswitch_is_link_valid(device, p->link))
    {
        LWSWITCH_PRINT(device, ERROR,
            "GET_RESIDENCY_BINS: Invalid link %d\n", p->link);
        return -LWL_BAD_ARGS;
    }

    if (p->table_select == LWSWITCH_TABLE_SELECT_MULTICAST)
    {
        // Snap the histogram
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL,
            DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));

        // Read high/low thresholds and colwery clocks to nsec
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW);
        threshold = DRF_VAL(_MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW, _LIMIT, val);
        p->bin.lowThreshold = threshold * 1000 / 1333;

        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH);
        threshold = DRF_VAL(_MULTICASTTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH, _LIMIT, val);
        p->bin.hiThreshold = threshold * 1000 / 1333;

        LWSWITCH_NPORT_WR32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_CTRL,
            DRF_NUM(_MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_CTRL, _INDEX, 0) |
            DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_CTRL, _AUTOINCR, _ON));
        for (i = 0; i < LWSWITCH_RESIDENCY_BIN_SIZE; i++)
        {
            // Low
            val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            p->residency[i].low = (val_hi << 32) | val;

            // Medium
            val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            p->residency[i].medium = (val_hi << 32) | val;

            // High
            val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            p->residency[i].high = (val_hi << 32) | val;
        }

        // Reset the histogram
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL,
            DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_MULTICASTTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));

    }
    else if (p->table_select == LWSWITCH_TABLE_SELECT_REDUCTION)
    {
        // Snap the histogram
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL,
            DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));

        // Read high/low thresholds and colwery clocks to nsec
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW);
        threshold = DRF_VAL(_REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_LOW, _LIMIT, val);
        p->bin.lowThreshold = threshold * 1000 / 1333;

        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH);
        threshold = DRF_VAL(_REDUCTIONTSTATE, _STAT_RESIDENCY_BIN_CTRL_HIGH, _LIMIT, val);
        p->bin.hiThreshold = threshold * 1000 / 1333;

        LWSWITCH_NPORT_WR32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_CTRL,
            DRF_NUM(_REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_CTRL, _INDEX, 0) |
            DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_CTRL, _AUTOINCR, _ON));
        for (i = 0; i < LWSWITCH_RESIDENCY_BIN_SIZE; i++)
        {
            // Low
            val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            p->residency[i].low = (val_hi << 32) | val;

            // Medium
            val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            p->residency[i].medium = (val_hi << 32) | val;

            // High
            val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_COUNT_DATA);
            p->residency[i].high = (val_hi << 32) | val;
        }

        // Reset the histogram
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL,
            DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_REDUCTIONTSTATE, _STAT_RESIDENCY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "GET_RESIDENCY_BINS: Invalid table %d\n", p->table_select);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return LWL_SUCCESS;
}

/*
 * CTRL_LWSWITCH_GET_RB_STALL_BUSY
 */
static LwlStatus
lwswitch_ctrl_get_rb_stall_busy_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_RB_STALL_BUSY *p
)
{
    LwU64 val;
    LwU64 val_hi;

    if (!lwswitch_is_link_valid(device, p->link))
    {
        LWSWITCH_PRINT(device, ERROR,
            "GET_RB_STALL_BUSY: Invalid link %d\n", p->link);
        return -LWL_BAD_ARGS;
    }

    if (p->table_select == LWSWITCH_TABLE_SELECT_MULTICAST)
    {
        // Snap the histogram
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL,
            DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));

        //
        // VC0
        // 

        // Total time
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_WINDOW_0_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_WINDOW_0_HIGH);
        p->vc0.time = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Busy
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_BUSY_TIMER_0_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_BUSY_TIMER_0_HIGH);
        p->vc0.busy = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Stall
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_STALL_TIMER_0_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_STALL_TIMER_0_HIGH);
        p->vc0.stall = ((val_hi << 32) | val) * 1000 / 1333;     // in ns

        //
        // VC1
        // 

        // Total time
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_WINDOW_1_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_WINDOW_1_HIGH);
        p->vc1.time = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Busy
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_BUSY_TIMER_1_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_BUSY_TIMER_1_HIGH);
        p->vc1.busy = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Stall
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_STALL_TIMER_1_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_STALL_TIMER_1_HIGH);
        p->vc1.stall = ((val_hi << 32) | val) * 1000 / 1333;     // in ns

        // Reset the busy/stall counters
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL,
            DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_MULTICASTTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));
    }
    else if (p->table_select == LWSWITCH_TABLE_SELECT_REDUCTION)
    {
        // Snap the histogram
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL,
            DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _ENABLE));
        //
        // VC0
        // 

        // Total time
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_WINDOW_0_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_WINDOW_0_HIGH);
        p->vc0.time = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Busy
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_BUSY_TIMER_0_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_BUSY_TIMER_0_HIGH);
        p->vc0.busy = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Stall
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_STALL_TIMER_0_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_STALL_TIMER_0_HIGH);
        p->vc0.stall = ((val_hi << 32) | val) * 1000 / 1333;     // in ns

        //
        // VC1
        // 

        // Total time
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_WINDOW_1_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_WINDOW_1_HIGH);
        p->vc1.time = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Busy
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_BUSY_TIMER_1_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_BUSY_TIMER_1_HIGH);
        p->vc1.busy = ((val_hi << 32) | val) * 1000 / 1333;      // in ns

        // Stall
        val = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_STALL_TIMER_1_LOW);
        val_hi = LWSWITCH_NPORT_RD32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_STALL_TIMER_1_HIGH);
        p->vc1.stall = ((val_hi << 32) | val) * 1000 / 1333;     // in ns

        // Reset the histogram
        LWSWITCH_NPORT_WR32_LS10(device, p->link, _REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL,
            DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _ENABLE_TIMER, _ENABLE) |
            DRF_DEF(_REDUCTIONTSTATE, _STAT_STALL_BUSY_CONTROL, _SNAP_ON_DEMAND, _DISABLE));
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "GET_RB_STALL_BUSY: Invalid table %d\n", p->table_select);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return LWL_SUCCESS;
}

LwU32
lwswitch_read_iddq_dvdd_ls10
(
    lwswitch_device *device
)
{
    return lwswitch_fuse_opt_read_ls10(device, LW_FUSE_OPT_IDDQ_DVDD);
}

void
lwswitch_load_uuid_ls10
(
    lwswitch_device *device
)
{
    //
    // Read 128-bit UUID from secure scratch registers which must be
    // populated by firmware.
    // TODO: FSP does not implement this, so this hack uses physical ID of LS10
    // on the board to serve as a proxy for UUID.
    // Bug #3526744
    //

    device->uuid.uuid[0] = lwswitch_read_physical_id(device);
    device->uuid.uuid[1] = 0;
    device->uuid.uuid[2] = 0;
    device->uuid.uuid[3] = 0;

    LWSWITCH_PRINT(device, WARN, "%s: Temporarily reporting physical_id=0x%x for UUID=%08x'%08x'%08x'%08x\n",
         __FUNCTION__,
         device->uuid.uuid[0],
         device->uuid.uuid[0], device->uuid.uuid[1], device->uuid.uuid[2], device->uuid.uuid[3]);
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus
lwswitch_launch_ALI_ls10
(
    lwswitch_device *device
)
{
    LwU64 enabledLinkMask;
    LwU64 forcedConfgLinkMask = 0;
    LwlStatus status  = LWL_SUCCESS;
    LwBool bEnableAli = LW_FALSE;
    LwU64 i           = 0;
    lwlink_link *link;

    enabledLinkMask   = lwswitch_get_enabled_link_mask(device);
    forcedConfgLinkMask = ((LwU64)device->regkeys.chiplib_forced_config_link_mask) +
                ((LwU64)device->regkeys.chiplib_forced_config_link_mask2 << 32);


    //
    // Lwrrently, we don't support a mix of forced/auto config links
    // return early
    //
    if (forcedConfgLinkMask != 0)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

#ifdef INCLUDE_LWLINK_LIB
    bEnableAli = device->lwlink_device->enableALI;
#endif

    if (!bEnableAli)
    {
        LWSWITCH_PRINT(device, INFO,
                "%s: ALI not supported on the given device\n",
                __FUNCTION__);
        return LWL_ERR_GENERIC;
    }



    FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

        link = lwswitch_get_link(device, i);

        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLIPT_LNK, link->linkNumber) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        if (!lwswitch_is_link_in_reset(device, link))
        {
            continue;
        }

        LWSWITCH_PRINT(device, INFO,
                "%s: ALI launching on link: 0x%llx\n",
                __FUNCTION__, i);

        // Apply appropriate SIMMODE settings
        status = lwswitch_minion_set_sim_mode_ls10(device, link);
        if (status != LWL_SUCCESS)
        {
            return LW_ERR_LWLINK_CONFIGURATION_ERROR;
        }

        // Apply appropriate SMF settings
        status = lwswitch_minion_set_smf_settings_ls10(device, link);
        if (status != LWL_SUCCESS)
        {
            return LW_ERR_LWLINK_CONFIGURATION_ERROR;
        }

        // Apply appropriate UPHY Table settings
        status = lwswitch_minion_select_uphy_tables_ls10(device, link);
        if (status != LWL_SUCCESS)
        {
            return LW_ERR_LWLINK_CONFIGURATION_ERROR;
        }

        // Before INITPHASE1, apply NEA setting
        lwswitch_setup_link_loopback_mode(device, link->linkNumber);

        //
        // Request active, but don't block. FM will come back and check
        // active link status by blocking on this TLREQ's completion
        //
        status = lwswitch_request_tl_link_state_ls10(link,
                LW_LWLIPT_LNK_CTRL_LINK_STATE_REQUEST_REQUEST_ACTIVE,
                LW_FALSE);

        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: TL link state request to active for ALI failed for link: 0x%llx\n",
                __FUNCTION__, i);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

LwlStatus
lwswitch_set_training_mode_ls10
(
    lwswitch_device *device
)
{
    LwU64 enabledLinkMask, forcedConfgLinkMask;

    LwU32 regVal;
    LwU64 i = 0;
    lwlink_link *link;

    enabledLinkMask     = lwswitch_get_enabled_link_mask(device);
    forcedConfgLinkMask = ((LwU64)device->regkeys.chiplib_forced_config_link_mask) +
                ((LwU64)device->regkeys.chiplib_forced_config_link_mask2 << 32);

    //
    // Lwrrently, we don't support a mix of forced/auto config links
    // return early
    //
    if (forcedConfgLinkMask != 0)
    {
        LWSWITCH_PRINT(device, INFO,
                "%s: Forced-config set, skipping setting up link training selection\n",
                __FUNCTION__);
        return LWL_SUCCESS;
    }

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS) || LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    if (device->regkeys.link_training_mode == LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_ALI)
    {
        //
        // If ALI is force enabled then check to make sure ALI is supported
        // and write to the SYSTEM_CTRL register to force it to enabled
        //
        FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
        {
            LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

            link = lwswitch_get_link(device, i);

            if ((link == NULL) ||
                !LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLIPT_LNK, link->linkNumber) ||
                (i >= LWSWITCH_LWLINK_MAX_LINKS))
            {
                continue;
            }

            regVal = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_CAP_LOCAL_LINK_CHANNEL);

            if (!FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CAP_LOCAL_LINK_CHANNEL, _ALI_SUPPORT, _SUPPORTED, regVal))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: ALI training not supported! Regkey forcing ALI will be ignored\n",__FUNCTION__);
                return -LWL_ERR_NOT_SUPPORTED;
            }

            if (FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2_LOCK, _ALI_ENABLE, _LOCKED, regVal))
            {
                LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_SYSTEM_LINK_CHANNEL_CTRL2_LOCK_CLEAR, 0x2);
            }

            LWSWITCH_PRINT(device, INFO,
                "%s: ALI training set on link: 0x%llx\n",
                __FUNCTION__, i);

            regVal = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_SYSTEM_LINK_CHANNEL_CTRL2);

            regVal = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2, _ALI_ENABLE, _ENABLE, regVal);
            LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_SYSTEM_LINK_CHANNEL_CTRL2, regVal);

        }
        FOR_EACH_INDEX_IN_MASK_END;
    }
    else if (device->regkeys.link_training_mode == LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_NON_ALI)
    {
        // If non-ALI is force enabled then disable ALI
        FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
        {
            LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

            link = lwswitch_get_link(device, i);

            if ((link == NULL) ||
                !LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLIPT_LNK, link->linkNumber) ||
                (i >= LWSWITCH_LWLINK_MAX_LINKS))
            {
                continue;
            }

            regVal = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_SYSTEM_LINK_CHANNEL_CTRL2_LOCK);

            if (FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2_LOCK, _ALI_ENABLE, _LOCKED, regVal))
            {
                LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_SYSTEM_LINK_CHANNEL_CTRL2_LOCK_CLEAR, 0x2);
            }

            regVal = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_SYSTEM_LINK_CHANNEL_CTRL2);

            regVal = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2, _ALI_ENABLE, _DISABLE, regVal);
            LWSWITCH_LINK_WR32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_SYSTEM_LINK_CHANNEL_CTRL2, regVal);

        }
        FOR_EACH_INDEX_IN_MASK_END;

    }
    else
#endif
    {
        // Else sanity check the SYSTEM register settings
        FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
        {
            LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

            link = lwswitch_get_link(device, i);

            if ((link == NULL) ||
                !LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLIPT_LNK, link->linkNumber) ||
                (i >= LWSWITCH_LWLINK_MAX_LINKS))
            {
                continue;
            }

            regVal = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                        _CTRL_CAP_LOCAL_LINK_CHANNEL);

            if (!FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CAP_LOCAL_LINK_CHANNEL, _ALI_SUPPORT, _SUPPORTED, regVal))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: ALI training not supported! Non-ALI will be used as the default.\n",__FUNCTION__);
                return LWL_SUCCESS;
            }
        }
        FOR_EACH_INDEX_IN_MASK_END;

#ifdef INCLUDE_LWLINK_LIB
        device->lwlink_device->enableALI = LW_TRUE;
#endif
    }

    return LWL_SUCCESS;
}

static void
_lwswitch_get_lwlink_power_state_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret
)
{
    lwlink_link *link;
    LwU32 linkState;
    LwU32 linkPowerState;
    LwU8 i;

    // Determine power state for each enabled link
    FOR_EACH_INDEX_IN_MASK(64, i, ret->enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

        link = lwswitch_get_link(device, i);

        if ((link == NULL) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        linkState = ret->linkInfo[i].linkState;

        switch (linkState)
        {
            case LWSWITCH_LWLINK_STATUS_LINK_STATE_ACTIVE:
                linkPowerState = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK, _PWRM_CTRL);

                if (FLD_TEST_DRF(_LWLIPT_LNK, _PWRM_CTRL, _L1_LWRRENT_STATE, _L1, linkPowerState))
                {
                    linkPowerState = LWSWITCH_LWLINK_STATUS_LINK_POWER_STATE_L1;
                }
                else
                {
                    linkPowerState = LWSWITCH_LWLINK_STATUS_LINK_POWER_STATE_L0;
                }
                break;

            default:
                linkPowerState  = LWSWITCH_LWLINK_STATUS_LINK_POWER_STATE_ILWALID;
                break;
        }

        ret->linkInfo[i].linkPowerState = linkPowerState;
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

LwlStatus
lwswitch_ctrl_get_lwlink_status_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret
)
{
    LwlStatus retval = LWL_SUCCESS;

    retval = lwswitch_ctrl_get_lwlink_status_lr10(device, ret);

    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    _lwswitch_get_lwlink_power_state_ls10(device, ret);

    return retval;
}

LwlStatus
lwswitch_parse_bios_image_ls10
(
    lwswitch_device *device
)
{
    return lwswitch_parse_bios_image_lr10(device);
}

/*
 * CTRL_LWSWITCH_INBAND_SEND_DATA
 */
LwlStatus
lwswitch_ctrl_inband_send_data_ls10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_SEND_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*
 * CTRL_LWSWITCH_INBAND_READ_DATA
 */
LwlStatus
lwswitch_ctrl_inband_read_data_ls10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_READ_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*
 * CTRL_LWSWITCH_INBAND_FLUSH_DATA
 */
LwlStatus
lwswitch_ctrl_inband_flush_data_ls10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*
 * CTRL_LWSWITCH_INBAND_FLUSH_DATA
 */
LwlStatus
lwswitch_ctrl_inband_pending_data_stats_ls10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_ctrl_get_lwlink_lp_counters_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *params
)
{
    LwU32 counterValidMaskOut;
    LwU32 counterValidMask;
    LwU32 cntIdx;
    LW_STATUS status;
    LwU32 statData;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, params->linkId))
    {
        return -LWL_BAD_ARGS;
    }

    counterValidMaskOut = 0;
    counterValidMask = params->counterValidMask;

    cntIdx = CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_LWHS;
    if (counterValidMask & LWBIT32(cntIdx))
    {
        status = lwswitch_minion_get_dl_status(device, params->linkId,
                                LW_LWLSTAT_TX01, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        params->counterValues[cntIdx] = DRF_VAL(_LWLSTAT_TX01, _COUNT_TX_STATE,
                                                _LWHS_VALUE, statData);
        counterValidMaskOut |= LWBIT32(cntIdx);
    }

    cntIdx = CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_OTHER;
    if (counterValidMask & LWBIT32(cntIdx))
    {
        status = lwswitch_minion_get_dl_status(device, params->linkId,
                                LW_LWLSTAT_TX02, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        params->counterValues[cntIdx] = DRF_VAL(_LWLSTAT_TX02, _COUNT_TX_STATE,
                                                _OTHER_VALUE, statData);
        counterValidMaskOut |= LWBIT32(cntIdx);
    }

    cntIdx = CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_ENTER;
    if (counterValidMask & LWBIT32(cntIdx))
    {
        status = lwswitch_minion_get_dl_status(device, params->linkId,
                                LW_LWLSTAT_TX06, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        params->counterValues[cntIdx] = DRF_VAL(_LWLSTAT_TX06, _NUM_LCL,
                                                _LP_ENTER_VALUE, statData);
        counterValidMaskOut |= LWBIT32(cntIdx);
    }

    cntIdx = CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_EXIT;
    if (counterValidMask & LWBIT32(cntIdx))
    {
        status = lwswitch_minion_get_dl_status(device, params->linkId,
                                LW_LWLSTAT_TX05, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        params->counterValues[cntIdx] = DRF_VAL(_LWLSTAT_TX05, _NUM_LCL,
                                                _LP_EXIT_VALUE, statData);
        counterValidMaskOut |= LWBIT32(cntIdx);
    }

    cntIdx = CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_SLEEP;
    if (counterValidMask & LWBIT32(cntIdx))
    {
        status = lwswitch_minion_get_dl_status(device, params->linkId,
                                LW_LWLSTAT_TX10, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        params->counterValues[cntIdx] = DRF_VAL(_LWLSTAT_TX10, _COUNT_TX_STATE,
                                                _SLEEP_VALUE, statData);
        counterValidMaskOut |= LWBIT32(cntIdx);
    }
    
    params->counterValidMask = counterValidMaskOut;

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* TRY TO ADD NEW UNPUBLISHED CODE BELOW THIS LINE */

LwlStatus
lwswitch_get_board_id_ls10
(
    lwswitch_device *device,
    LwU16 *boardId
)
{
    return lwswitch_get_board_id_lr10(device, boardId);
}

/*
 * @brief: This function returns current link repeater mode state for a given global link idx
 * @params[in]   device           reference to current lwswitch device
 * @params[in]   linkId           link to retrieve repeater mode state from
 * @params[out]  isRepeaterMode   pointer to Repeater Mode boolean
 */
LwlStatus
lwswitch_is_link_in_repeater_mode_ls10
(
    lwswitch_device *device,
    LwU32           link_id,
    LwBool          *isRepeaterMode
)
{
    lwlink_link *link;

    if (!device->hal.lwswitch_is_link_valid(device, link_id))
    {
        return -LWL_BAD_ARGS;
    }

    link = lwswitch_get_link(device, link_id);
    if (link == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    *isRepeaterMode = device->link[link->linkNumber].bIsRepeaterMode;

    return LWL_SUCCESS;
}

LwBool
lwswitch_is_cci_supported_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_PRINT(device, WARN, "%s: CCI is not yet supported on LS10\n", __FUNCTION__);
    return LW_FALSE;
}

static LwlStatus
lwswitch_ctrl_clear_counters_ls10
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_CLEAR_COUNTERS_PARAMS *ret
)
{
    lwlink_link *link;
    LwU8 i;
    LwU32 counterMask;
    LwlStatus status = LWL_SUCCESS;

    counterMask = ret->counterMask;

    FOR_EACH_INDEX_IN_MASK(64, i, ret->linkMask)
    {
        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            continue;
        }

        counterMask = ret->counterMask;

        if ((counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS) ||
            (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL))
        {
            status = lwswitch_minion_send_command(device, link->linkNumber,
                LW_MINION_LWLINK_DL_CMD_COMMAND_DLSTAT_CLR_MINION_MISCCNT, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR, "%s : Failed to clear misc count to MINION for link # %d\n",
                    __FUNCTION__, link->linkNumber);
            }
            counterMask &=
                ~(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS | LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL);
        }

        lwswitch_ctrl_clear_throughput_counters_ls10(device, link, counterMask);
        lwswitch_ctrl_clear_lp_counters_ls10(device, link, counterMask);
        status = lwswitch_ctrl_clear_dl_error_counters_ls10(device, link, counterMask);
        // Return early with failure on clearing through minion
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failure on clearing link counter mask 0x%x on link %d\n",
                __FUNCTION__, counterMask, link->linkNumber);
            break;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return status;
}

LwlStatus
lwswitch_ctrl_get_irq_info_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_IRQ_INFO_PARAMS *p
)
{
    p->maskInfoCount                    = 0;
    // Set the mask to AND out during servicing in order to avoid int storm.
//    p->maskInfoList[0].irqPendingOffset = LW_PSMC_INTR_LEGACY;
//    p->maskInfoList[0].irqEnabledOffset = LW_PSMC_INTR_EN_LEGACY;
//    p->maskInfoList[0].irqEnableOffset  = LW_PSMC_INTR_EN_SET_LEGACY;
//    p->maskInfoList[0].irqDisableOffset = LW_PSMC_INTR_EN_CLR_LEGACY;
//    p->maskInfoCount                    = 1;

    return LWL_SUCCESS;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

//
// This function auto creates the ls10 HAL connectivity from the LWSWITCH_INIT_HAL
// macro in haldef_lwswitch.h
//
// Note: All hal fns must be implemented for each chip.
//       There is no automatic stubbing here.
//
void lwswitch_setup_hal_ls10(lwswitch_device *device)
{
    device->chip_arch = LWSWITCH_GET_INFO_INDEX_ARCH_LS10;
    device->chip_impl = LWSWITCH_GET_INFO_INDEX_IMPL_LS10;

    LWSWITCH_INIT_HAL(device, ls10);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LWSWITCH_INIT_HAL_UNPUBLISHED(device, ls10);                             
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_INIT_HAL_LWCFG_LS10(device, ls10);                             
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}
