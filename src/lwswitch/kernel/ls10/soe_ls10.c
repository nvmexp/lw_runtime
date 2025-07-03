/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "soe/soe_lwswitch.h"
#include "soe/soe_priv_lwswitch.h"

#include "rmlsfm.h"

#include "lwlink_export.h"
#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/soe_ls10.h"
#include "lr10/soe_lr10.h"
#include "soe/bin/g_soeuc_ls10_dbg.h"
#include "soe/bin/g_soeriscvuc_ls10_combined.h"
#include "lwswitch/ls10/dev_soe_ip.h"
#include "lwswitch/ls10/dev_soe_ip_addendum.h"
#include "lwswitch/ls10/dev_falcon_v4.h"
#include "lwswitch/ls10/lw_soe_riscv_address_map.h"

#include "flcn/flcnable_lwswitch.h"
#include "flcn/flcn_lwswitch.h"
#include "rmflcncmdif_lwswitch.h"
#include "soe/soeifcmn.h"
#include "regkey_lwswitch.h"

LwlStatus
lwswitch_get_soe_ucode_binaries_ls10
(
    lwswitch_device *device,
    const LwU32 **soe_ucode_data,
    const LwU32 **soe_ucode_header
)
{
    LwU32 debug_mode;

    if (!soe_ucode_data || !soe_ucode_header)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Bad agrs!\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    debug_mode = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_SCP, _CTL_STAT);
    debug_mode = DRF_VAL(_SOE, _SCP_CTL_STAT, _DEBUG_MODE, debug_mode);

    if (debug_mode)
    {
        *soe_ucode_data = soe_ucode_data_ls10_dbg;
        *soe_ucode_header = soe_ucode_header_ls10_dbg;
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: No soe-ls10 prod binaries!\n",
            __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

/**
 * @brief   Sets pEngDeslwc and pEngDescBc to the discovered
 * engine that matches this flcnable instance
 *
 * @param[in]   device       lwswitch_device pointer
 * @param[in]   pSoe         SOE pointer
 * @param[out]  pEngDeslwc  pointer to the UniCast Engine
 *       Descriptor
 * @param[out]  pEngDescBc  pointer to the BroadCast Engine
 *       Descriptor
 */
static void
_soeFetchEngines_LS10
(
    lwswitch_device         *device,
    FLCNABLE                *pSoe,
    ENGINE_DESCRIPTOR_TYPE *pEngDeslwc,
    ENGINE_DESCRIPTOR_TYPE *pEngDescBc
)
{
    pEngDeslwc->initialized = LW_FALSE;
    if (LWSWITCH_ENG_IS_VALID(device, SOE, 0))
    {
        pEngDeslwc->base = LWSWITCH_GET_ENG(device, SOE, , 0);
    }
    else
    {
        pEngDeslwc->base = 0;
    }

    pEngDescBc->initialized = LW_FALSE;
    pEngDescBc->base = 0;
}

/*
 * @Brief : Selects SOE core (Falcon or Riscv)
 *
 * @param[in] device Bootstrap SOE on this device
 *
 *
 */
LwlStatus
lwswitch_soe_set_ucode_core_ls10
(
    lwswitch_device *device,
    LwBool bFalcon
)
{
    LWSWITCH_TIMEOUT timeout;
    LwU32 bcrCtrl;
    LwBool keepPolling;

    bcrCtrl = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE, _RISCV_BCR_CTRL);

    // Check if BCR is locked to RISCV when falcon boot is requested
    if (FLD_TEST_DRF(_SOE, _RISCV_BCR_CTRL, _VALID, _TRUE, bcrCtrl) &&
        FLD_TEST_DRF(_SOE, _RISCV_BCR_CTRL, _CORE_SELECT, _RISCV, bcrCtrl) &&
        bFalcon)
    {
        LWSWITCH_PRINT(device, ERROR,
            "BCR ctrl is locked to RISCV. Soe boot can fail!\n");
    }

    // Check if BCR is locked to FALCON when riscv boot is requested
    if (FLD_TEST_DRF(_SOE, _RISCV_BCR_CTRL, _VALID, _TRUE, bcrCtrl) &&
        FLD_TEST_DRF(_SOE, _RISCV_BCR_CTRL, _CORE_SELECT, _FALCON, bcrCtrl) &&
        !bFalcon)
    {
        LWSWITCH_PRINT(device, ERROR,
            "BCR ctrl is locked to FALCON. Soe boot can fail!\n");
    }

    // Select core FALCON vs RISCV
    bcrCtrl = bFalcon ? 
        DRF_DEF(_SOE_RISCV, _BCR_CTRL, _CORE_SELECT, _FALCON) :
        DRF_DEF(_SOE_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV);
    LWSWITCH_SOE_WR32_LS10(device, 0, _SOE_RISCV, _BCR_CTRL, bcrCtrl);

    // Wait for the switch to happen.
    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        bcrCtrl = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE, _RISCV_BCR_CTRL);

        if (FLD_TEST_DRF(_SOE_RISCV, _BCR_CTRL, _VALID, _TRUE, bcrCtrl))
        {
            LWSWITCH_PRINT(device, INFO,
                "%s : Core set to %s!\n",
                __FUNCTION__, bFalcon ? "FALCON" : "RISCV");
            return LWL_SUCCESS;    
        }

        if (!keepPolling)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for BCR core switch to complete!"
                " BCR_CTRL = 0x%08x\n",
                __FUNCTION__, bcrCtrl);
            LWSWITCH_ASSERT(0);
            break;
        }

        lwswitch_os_sleep(1);
    } while (keepPolling);
        
    return -LWL_ERR_ILWALID_STATE;
}

/*
 * @Brief : Send a test command to SOE
 *
 * @param   device  The lwswitch device
 */
static LW_STATUS
_lwswitch_soe_send_test_cmd
(
    lwswitch_device *device
)
{
    RM_FLCN_CMD_SOE     cmd;
    LWSWITCH_TIMEOUT    timeout;
    LwU32               cmdSeqDesc;
    LW_STATUS           status;

    FLCN *pFlcn = device->pSoe->pFlcn;

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));

    cmd.hdr.unitId = RM_SOE_UNIT_NULL;
    // sending nothing but a header for UNIT_NULL
    cmd.hdr.size   = RM_FLCN_QUEUE_HDR_SIZE;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 5, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn,
                                (PRM_FLCN_CMD)&cmd,
                                NULL,   // pMsg             - not used for now
                                NULL,   // pPayload         - not used for now
                                SOE_RM_CMDQ_LOG_ID,
                                &cmdSeqDesc,
                                &timeout);

    LWSWITCH_ASSERT(status == LWL_SUCCESS);
    return status;
}

LwlStatus
_lwswitch_reset_soe
(
    lwswitch_device *device
)
{
    LwU32 val = 0;
    LWSWITCH_TIMEOUT timeout;
    LwBool keepPolling;

    LWSWITCH_PRINT(device, INFO, "%s: Issuing reset\n", __FUNCTION__);

    // Assert reset
    LWSWITCH_SOE_WR32_LS10(device, 0, _SOE, _FALCON_ENGINE,
        DRF_DEF(_SOE_FALCON, _ENGINE, _RESET, _TRUE));

    lwswitch_os_sleep(10);

    // Unassert reset
    LWSWITCH_SOE_WR32_LS10(device, 0, _SOE, _FALCON_ENGINE,
        DRF_DEF(_SOE_FALCON, _ENGINE, _RESET, _FALSE));

    // Wait for the switch to happen.
    lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        val = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE, _FALCON_ENGINE);

        if (FLD_TEST_DRF(_SOE_FALCON, _ENGINE, _RESET_STATUS, _DEASSERTED, val))
        {
            LWSWITCH_PRINT(device, INFO, "%s: Reset Done.\n", __FUNCTION__);
            break;
        }

        if (!keepPolling)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for LW_SOE_FALCON_ENGINE_RESET_STATUS to be deasserted!"
                " val = 0x%08x\n",
                __FUNCTION__, val);
            LWSWITCH_ASSERT(0);
            return -LWL_ERR_ILWALID_STATE;
        }

        lwswitch_os_sleep(1);
    } while (keepPolling);

    return LWL_SUCCESS;
}

/*
 * @Brief : Bootstrap SOE associated with the link
 *
 * @param[in] device Bootstrap SOE on this device
 * @param[in] link   Bootstrap SOE associated with the link
 */
static LwlStatus
_lwswitch_soe_sideload_riscv_ucode
(
    lwswitch_device *device
)
{
    LWSWITCH_TIMEOUT timeout;
    LwBool keepPolling;
    LwU32 data;
    PFLCN pFlcn = device->pSoe->pFlcn;
    LwU32 codeOffset   = g_soeriscvuc_ls10_combined_desc.monitorCodeOffset;
    LwU32 codeSize     = g_soeriscvuc_ls10_combined_desc.monitorCodeSize;
    LwU32 dataOffset   = g_soeriscvuc_ls10_combined_desc.monitorDataOffset;
    LwU32 dataSize     = g_soeriscvuc_ls10_combined_desc.monitorDataSize;
    LwU32 manifestSize = g_soeriscvuc_ls10_combined_desc.manifestSize;
    LwU32 dmemSize     = flcnDmemSize_HAL(device, pFlcn);
    LwU32 manifestOffset = g_soeriscvuc_ls10_combined_desc.manifestOffset;
    LwU32 manifestDest = dmemSize - manifestSize;
    LwU32 *pUcode          = (LwU32*) g_soeriscvuc_ls10_combined_image;

    if (!LWSWITCH_ENG_VALID_LS10(device, SOE, 0))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: SOE is invalid\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    flcnImemCopyTo_HAL(device, pFlcn, 0, (LwU8*)&pUcode[codeOffset / 4], codeSize, LW_FALSE, 0, 0);
    if (dataSize > 0)
    {
        flcnDmemTransfer_HAL(device, pFlcn, 0, (LwU8*)&pUcode[dataOffset / 4], dataSize, 0, LW_FALSE);
    }

    if (manifestSize > 0)
    {
        flcnDmemTransfer_HAL(device, pFlcn, manifestDest, (LwU8*)&pUcode[manifestOffset / 4], manifestSize, 0, LW_FALSE);
    }

    // Set boot vector
    LWSWITCH_SOE_WR32_LS10(device, 0, _SOE_RISCV, _BOOT_VECTOR_HI, 0);
    LWSWITCH_SOE_WR32_LS10(device, 0, _SOE_RISCV, _BOOT_VECTOR_LO, LwU64_LO32(LW_RISCV_AMAP_IROM_START));

    // Select RISCV Core
    LWSWITCH_SOE_WR32_LS10(device, 0, _SOE_RISCV, _BCR_CTRL,
        DRF_DEF(_SOE_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV) |
        DRF_NUM(_SOE_RISCV, _BCR_CTRL, _VALID, 1) |
        DRF_NUM(_SOE_RISCV, _BCR_CTRL, _BRFETCH, 0));

    // Start CPU
    LWSWITCH_SOE_WR32_LS10(device, 0, _SOE_RISCV, _CPUCTL,
        DRF_DEF(_SOE_RISCV, _CPUCTL, _STARTCPU, _TRUE));

    lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;
        data = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_RISCV, _BR_RETCODE);
        if (FLD_TEST_DRF(_SOE, _RISCV_BR_RETCODE, _RESULT, _PASS, data))
        {
            // BROM finished with result: PASS
            break;
        }
        else if (FLD_TEST_DRF(_SOE, _RISCV_BR_RETCODE, _RESULT, _FAIL, data))
        {
            LWSWITCH_PRINT(device, SETUP, "%s: BROM finished with result: FAIL\n", __FUNCTION__);
            break;
        }

        lwswitch_os_sleep(1);
    } while(keepPolling);

    // Ensure the CPU has started
    data = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE, _RISCV_CPUCTL);
    if (FLD_TEST_DRF(_SOE, _RISCV_CPUCTL, _ACTIVE_STAT, _ACTIVE, data))
    {
        LWSWITCH_PRINT(device, SETUP, "%s: SOE started RISCV Core successfully\n", __FUNCTION__);
    }
    else
    {
        LWSWITCH_PRINT(device, SETUP, "%s: SOE failed to start in RISCV\n", __FUNCTION__);
        return -LW_ERR_NOT_READY;
    }
    return LWL_SUCCESS;
}

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
/*!
 * Helper function to dump some registers for debug.
 *
 * @param[in]  device  lwswitch_device pointer
 */
static void
dumpDebugRegisters
(
    lwswitch_device *device
)
{
    LwU32    value;

    // Mail boxes and CPU control
    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _MAILBOX0);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_MAILBOX0: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _MAILBOX1);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_MAILBOX1: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _CPUCTL);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_CPUCTL: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _DEBUGINFO);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_DEBUGINFO: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _EXCI);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_EXCI: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _SCTL);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_SCTL: 0x%x\n",
                   __FUNCTION__,
                   value);

    // EXTIO interrupts
    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_MISC, _EXTIO_IRQSTAT);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_MISC_EXTIO_IRQSTAT: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_MISC, _EXTIO_IRQMASK);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_MISC_EXTIO_IRQMASK: 0x%x\n",
                   __FUNCTION__,
                   value);

    // Falcon top level interrupts
    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _IRQSTAT);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQSTAT: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _IRQMODE);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQMODE: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _IRQMASK);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQMASK: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _IRQDEST);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQDEST: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _IRQDEST2);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQDEST2: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _IRQSCMASK);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQSCMASK: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _RESET_PRIV_LEVEL_MASK);
    LWSWITCH_PRINT(device, ERROR,
                "%s: LW_SOE_FALCON_RESET_PRIV_LEVEL_MASK: 0x%x\n",
                __FUNCTION__,
                value);
    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _IRQTMR_PRIV_LEVEL_MASK);
    LWSWITCH_PRINT(device, ERROR,
                "%s: LW_SOE_FALCON_IRQTMR_PRIV_LEVEL_MASK: 0x%x\n",
                __FUNCTION__,
                value);
    value = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _EXE_PRIV_LEVEL_MASK);
    LWSWITCH_PRINT(device, ERROR,
                "%s: LW_SOE_FALCON_EXE_PRIV_LEVEL_MASK: 0x%x\n",
                __FUNCTION__,
                value);
}
#endif // defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

/*
 * @Brief : Bootstrap SOE RISCV on the specified device
 *
 * @param[in] device Bootstrap SOE on this device
 */
LwlStatus
lwswitch_init_soe_ls10
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LWSWITCH_TIMEOUT timeout;
    LwBool keepPolling;
    LwU32 data;
    FLCN *pFlcn = device->pSoe->pFlcn;

    // Boot falcon by default
    if (device->regkeys.soe_boot_core == LW_SWITCH_REGKEY_SOE_BOOT_CORE_DEFAULT)
    {
        device->regkeys.soe_boot_core = LW_SWITCH_REGKEY_SOE_BOOT_CORE_FALCON;
    }

    // For falcon boot, revert to lr10.
    if (device->regkeys.soe_boot_core == LW_SWITCH_REGKEY_SOE_BOOT_CORE_FALCON)
    {
        return lwswitch_init_soe_lr10(device);
    }

    status = _lwswitch_reset_soe(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_RESET,
            "Failed to reset SOE(0)\n");
        return status;
    }

    // Set SOE ucode core to falcon
    status = lwswitch_soe_set_ucode_core(device, LW_FALSE);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_BOOTSTRAP,
            "Failed to boot SOE(0)\n");
        return status;
    }

    status = _lwswitch_soe_sideload_riscv_ucode(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_BOOTSTRAP,
            "Failed to boot SOE(1)\n");
        goto lwswitch_init_soe_fail;
    }

    //
    // We will exit this if we recieve bootstrap signal OR
    // if we timeout waiting for bootstrap signal       OR
    // if bootstrap fails
    //
    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 5, &timeout);
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        data = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_MAILBOX1);
        if (data == SOE_BOOTSTRAP_SUCCESS)
        {
            pFlcn->engDeslwc.initialized = LW_TRUE;
            break;
        }

        //
        // Check if SOE has halted unexpectedly.
        //
        // The explicit check is required because the interrupts
        // are not yet enabled as the device is still initializing.
        //
        if (soeIsCpuHalted_HAL(device, ((PSOE)pFlcn->pFlcnable)))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: SOE HALTED.\n",
                __FUNCTION__);
            status = -LWL_ERR_ILWALID_STATE;
            goto lwswitch_bootstrap_soe_fail;
        }

        if (!keepPolling)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for SOE to complete bootstrap!"
                "LW_PFALCON_FALCON_MAILBOX1 = 0x%08x\n",
                __FUNCTION__, data);
            status = -LWL_ERR_ILWALID_STATE;
            goto lwswitch_bootstrap_soe_fail;
        }

        lwswitch_os_sleep(1);
    } while (keepPolling);

    // Sanity the command and message queues as a final check
    if (_lwswitch_soe_send_test_cmd(device) != LW_OK)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_BOOTSTRAP,
            "Failed to boot SOE(2)\n");
        status = -LWL_ERR_ILWALID_STATE;
        goto lwswitch_init_soe_fail;
    }

    // Register SOE callbacks
    status = lwswitch_soe_register_event_callbacks(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_COMMAND_QUEUE,
            "Failed to register SOE events\n");
        return status;
    }

    LWSWITCH_PRINT(device, SETUP,
                   "%s: SOE successfully bootstrapped.\n",
                   __FUNCTION__);

    return LWL_SUCCESS;

lwswitch_bootstrap_soe_fail :
    LWSWITCH_PRINT(device, SETUP,
            "%s: Failed to bootstrap SOE.\n",
            __FUNCTION__);
    LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_BOOTSTRAP,
            "Failed to boot SOE(3)\n");
    // Log any failures SOE may have had during bootstrap
    (void)soeService_HAL(device, ((PSOE)pFlcn->pFlcnable));

lwswitch_init_soe_fail :
#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
        dumpDebugRegisters(device);
#endif // defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    LWSWITCH_ASSERT(0);
    return status;
}

/*!
 * @brief Determine if the SOE RISCV CPU is halted
 *
 * @param[in] device         lwswitch_device  pointer
 * @param[in] pSoe           SOE  pointer
 *
 * @return LwBool reflecting the SOE Riscv CPU halted state
 */
static LwBool
_soeIsCpuHalted_LS10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    LwU32 data;

    if (device->regkeys.soe_boot_core == LW_SWITCH_REGKEY_SOE_BOOT_CORE_RISCV)
    {
        data = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_RISCV, _CPUCTL);
    }
    else
    {
        data = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_FALCON, _CPUCTL);
    }

    return FLD_TEST_DRF(_PFALCON, _FALCON_CPUCTL, _HALTED, _TRUE, data);
}

static LwU32
_soeIntrStatus_LS10
(
    lwswitch_device *device
)
{
    LwU32 irq, mask, dest;
    FLCN *pFlcn = device->pSoe->pFlcn;

    irq = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQSTAT);

    if (device->regkeys.soe_boot_core == LW_SWITCH_REGKEY_SOE_BOOT_CORE_RISCV)
    {
        mask = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_RISCV, _IRQMASK);
        dest = LWSWITCH_SOE_RD32_LS10(device, 0, _SOE_RISCV, _IRQDEST);
    }
    else
    {
        mask = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQMASK);
        dest = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQDEST);
    }

    return (irq & mask & dest);
}

/*!
 * @brief Top level service routine
 *
 * @param[in] device         lwswitch_device  pointer
 * @param[in] pSoe           SOE  pointer
 *
 * @return 32-bit interrupt status AFTER all known interrupt-sources were
 *         serviced.
 */
static LwU32
_soeService_LS10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    LwBool  bRecheckMsgQ    = LW_FALSE;
    LwU32   clearBits       = 0;
    LwU32   intrStatus;
    PFLCN   pFlcn  = ENG_GET_FLCN(pSoe);

    if (pFlcn == NULL)
    {
        LWSWITCH_ASSERT(0);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    // Get the IRQ status and mask the sources not directed to host.
    intrStatus =  _soeIntrStatus_LS10(device);

    // Exit if there is nothing to do
    if (intrStatus == 0)
    {
       return 0;
    }

    // Service pending interrupts
    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _WDTMR, _TRUE))
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_WATCHDOG,
            "SOE Watchdog error\n");
        LWSWITCH_PRINT(device, INFO,
                    "%s: Watchdog timer fired. We do not support this "
                    "yet.\n", __FUNCTION__);
        LWSWITCH_ASSERT(0);

        clearBits |= DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _WDTMR, _SET);
    }

    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _EXTERR, _TRUE))
    {
        clearBits |= DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _EXTERR, _SET);

        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_EXTERR, "SOE EXTERR\n");
    }

    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _HALT, _TRUE))
    {
        clearBits |= DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _HALT, _SET);

        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_HALT, "SOE HALTED\n");
        soeServiceHalt_HAL(device, pSoe);
    }

    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _SWGEN0, _TRUE))
    {
        clearBits |= DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _SWGEN0, _SET);

        LWSWITCH_PRINT(device, INFO,
                    "%s: Received a message from SOE via SWGEN0\n",
                    __FUNCTION__);
        soeProcessMessages(device, pSoe);
        bRecheckMsgQ = LW_TRUE;
    }

    if (intrStatus & DRF_DEF(_PFALCON, _FALCON_IRQSTAT, _SWGEN1, _TRUE))
    {
        clearBits |= DRF_DEF(_PFALCON, _FALCON_IRQSCLR, _SWGEN1, _SET);

        LWSWITCH_PRINT(device, INFO,
                    "%s: Received a SWGEN1 interrupt\n",
                    __FUNCTION__);
    }

    // Clear any sources that were serviced and get the new status.
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQSCLR, clearBits);

    // Re-read interrupt status before retriggering to return correct value
    intrStatus =  _soeIntrStatus_LS10(device);

    //
    // If we just processed a SWGEN0 message queue interrupt, peek
    // into the message queue and see if any messages were missed the last time
    // the queue was purged (above). If it is not empty, re-generate SWGEN0
    // (since it is now cleared) and exit. As long as an interrupt is pending,
    // this function will be re-entered and the message(s) will be processed.
    //
    if (bRecheckMsgQ)
    {
        PFALCON_QUEUE_INFO      pQueueInfo;
        FLCNQUEUE              *pMsgQ;

        pQueueInfo = pFlcn->pQueueInfo;

        LWSWITCH_ASSERT(pQueueInfo != NULL);
        LWSWITCH_ASSERT(pQueueInfo->pQueues != NULL);

        pMsgQ = &pQueueInfo->pQueues[SOE_RM_MSGQ_LOG_ID];

        if (!pMsgQ->isEmpty(device, pFlcn, pMsgQ))
        {
           // It is not necessary to RMW IRQSSET (zeros are ignored)
           flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQSSET,
                            DRF_DEF(_PFALCON, _FALCON_IRQSSET, _SWGEN0, _SET));
        }
    }

    flcnIntrRetrigger_HAL(device, pFlcn);

    return intrStatus;
}

/*!
 * Called by soeService to handle a SOE halt. This function will dump the
 * current status of SOE and then trap the CPU for further inspection for a
 * debug build.
 *
 * @param[in]  device  lwswitch_device pointer
 * @param[in]  pSoe    SOE object pointer
 */
static void
_soeServiceHalt_LS10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    PFLCN    pFlcn = ENG_GET_FLCN(pSoe);

    LWSWITCH_PRINT(device, ERROR,
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!                   ** SOE HALTED **                !!\n"
                "!! Please file a bug with the following information. !!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n");

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    dumpDebugRegisters(device);
#endif // defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

    if (device->regkeys.soe_boot_core == LW_SWITCH_REGKEY_SOE_BOOT_CORE_RISCV)
        flcnDbgInfoCaptureRiscvPcTrace_HAL(device, pFlcn);
    else
        flcnDbgInfoCapturePcTrace_HAL(device, pFlcn);
    LWSWITCH_ASSERT(0);
}

/**
 * @brief   set hal function pointers for functions defined in LR10 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcnable   The flcnable for which to set hals
 */
void
soeSetupHal_LS10
(
    SOE *pSoe
)
{
    soe_hal *pHal = pSoe->base.pHal;
    flcnable_hal *pParentHal = (flcnable_hal *)pHal;

    soeSetupHal_LR10(pSoe);

    pParentHal->fetchEngines = _soeFetchEngines_LS10;

    pHal->isCpuHalted        = _soeIsCpuHalted_LS10;
    pHal->service            = _soeService_LS10;
    pHal->serviceHalt        = _soeServiceHalt_LS10;
}
