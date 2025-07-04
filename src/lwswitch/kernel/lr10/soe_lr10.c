/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "soe/soe_lwswitch.h"
#include "soe/soe_priv_lwswitch.h"
#include "soe/soebif.h"
#include "rmlsfm.h"

#include "lwlink_export.h"
#include "common_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/soe_lr10.h"
#include "soe/bin/g_soeuc_lr10_dbg.h"
#include "soe/bin/g_soeuc_lr10_prd.h"
#include "soe/soeifcmn.h"
#include "lwswitch/lr10/dev_soe_ip.h"
#include "lwswitch/lr10/dev_soe_ip_addendum.h"
#include "lwswitch/lr10/dev_falcon_v4.h"
#include "lwswitch/lr10/dev_lwlsaw_ip.h"
#include "lwswitch/lr10/dev_therm.h"
#include "regkey_lwswitch.h"

#include "flcn/flcnable_lwswitch.h"
#include "flcn/flcn_lwswitch.h"

#include "rmflcncmdif_lwswitch.h"

/*
 * @Brief : Selects SOE core (Falcon or Riscv)
 *
 * @param[in] device Bootstrap SOE on this device
 *
 * Does nothing on LR10
 */
LwlStatus
lwswitch_soe_set_ucode_core_lr10
(
    lwswitch_device *device,
    LwBool bFalcon
)
{
    return LWL_SUCCESS;
}

/*
 * @Brief : Reset SOE at the engine level.
 *
 * @param[in] device Reset SOE on this device
 */
static LwlStatus
_lwswitch_reset_soe
(
    lwswitch_device *device
)
{
    LwU32 value;
    LwlStatus status;

    // Assert reset
    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _ENGINE);
    value = FLD_SET_DRF(_SOE, _FALCON, _ENGINE_RESET, _TRUE, value);
    LWSWITCH_SOE_WR32_LR10(device, 0, _SOE_FALCON, _ENGINE, value);

    //
    // TODO: Track down correct delay, if any.
    // Lwrrently GPU does not enforce a delay, use 1ms for now.
    //
    lwswitch_os_sleep(1);

    // Unassert reset
    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _ENGINE);
    value = FLD_SET_DRF(_SOE, _FALCON, _ENGINE_RESET, _FALSE, value);
    LWSWITCH_SOE_WR32_LR10(device, 0, _SOE_FALCON, _ENGINE, value);

    // Set SOE ucode core to falcon
    status = lwswitch_soe_set_ucode_core(device, LW_TRUE);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Failed to set SOE core\n");
        return status;
    }

    // Wait for reset to complete
    if (flcnWaitForResetToFinish_HAL(device, device->pSoe->pFlcn) != LW_OK)
    {
        // TODO: Fix up LW_STATUS translation, anything but LW_OK is a failure.
        return LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Copy the soe ucode to IMEM and DMEM and write soe ucode entrypoint
 *          to boot vector register.
 *
 * @param[in] device  Copy ucode to this device's SOE
 */
static LwlStatus
_lwswitch_soe_copy_ucode_cpubitbang
(
    lwswitch_device                *device,
    const LwU32                    *soe_ucode_data,
    const LwU32                    *soe_ucode_header
)
{
    const PSOE_UCODE_HDR_INFO_LR10 pUcodeHeader =
        (PSOE_UCODE_HDR_INFO_LR10) &(soe_ucode_header[0]);

    LwU32 dataSize, data, i, appCount;
    LwU32 appCodeStartOffset, appCodeSize, appCodeImemOffset;
    LwU32 appDataStartOffset, appDataSize, appDataDmemOffset;
    LwU32 appCodeIsSelwre;
    LwU16 tag;
    FLCN *pFlcn = device->pSoe->pFlcn;

    dataSize = sizeof(soe_ucode_data[0]);

    // Initialize address of IMEM to 0x0 and set auto-increment on write
    data = 0;
    data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_IMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_IMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_IMEMC, _AINCW, 0x1, data);
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMC(0), data);

    for (appCount = 0; appCount < pUcodeHeader -> numApps; appCount++)
    {
        appCodeStartOffset = pUcodeHeader -> apps[appCount].appCodeStartOffset;
        appCodeSize = pUcodeHeader -> apps[appCount].appCodeSize;
        appCodeImemOffset = pUcodeHeader -> apps[appCount].appCodeImemOffset;
        appCodeIsSelwre = pUcodeHeader -> apps[appCount].appCodeIsSelwre;
        appDataStartOffset = pUcodeHeader -> apps[appCount].appDataStartOffset;
        appDataSize = pUcodeHeader -> apps[appCount].appDataSize;
        appDataDmemOffset = pUcodeHeader -> apps[appCount].appDataDmemOffset;

        if(appCodeSize > 0)
        {
            // Mark the following code as secure or unselwre
            data = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMC(0));
            data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_IMEMC, _SELWRE, appCodeIsSelwre, data);
            flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMC(0), data);
            // Initialize IMEM tag.
            // Writes to IMEM don't work if we don't do this
            tag = (LwU16)(appCodeImemOffset >> 8);
            flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMT(0), (LwU32) tag);

            // Copy over IMEM part of the ucode and tag along the way
            for (i = 0; i < (appCodeSize / dataSize); i++)
            {
                // Increment tag for after every block (256 bytes)
                if (i && ((i % ((256/dataSize))) == 0))
                {
                    tag++;
                    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMT(0), (LwU32) tag);
                }

                // Copy IMEM DWORD by DWORD
                data = soe_ucode_data[(appCodeStartOffset / dataSize) + i];
                flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_IMEMD(0), data);
            }
        }

        if(appDataSize > 0)
        {
            // Initialize address of DMEM to appDataDmemOffset and set auto-increment on write
            data = 0;
            data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMEMC, _OFFS, (appDataDmemOffset&0xFF), data);
            data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMEMC, _BLK, appDataDmemOffset>>8, data);
            data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCW, 0x1, data);
            flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_DMEMC(0), data);

            // Copy over DMEM part of the ucode
            for (i = 0; i < (appDataSize / dataSize); i++)
            {
                // Copy DMEM DWORD by DWORD
                data = soe_ucode_data[appDataStartOffset/dataSize + i];
                flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_DMEMD(0), data);
            }
        }
    }

    //
    // In this ucode load path, we bit bang, we do not use DMA,
    // so set REQUIRE_CTX to FALSE. This must be set before we start SOE.
    //
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_DMACTL,
                     DRF_NUM(_PFALCON_FALCON, _DMACTL, _REQUIRE_CTX, LW_FALSE));

    // Write soe ucode entrypoint to boot vector register
    data = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_BOOTVEC);
    data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_BOOTVEC, _VEC, pUcodeHeader->codeEntryPoint, data);
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_BOOTVEC, data);

    return LWL_SUCCESS;
}

/*
 * @Brief : Send a command to pFlcn for testing (this function is temporary)
 *
 * @param   device  The lwswitch device
 * @param   pFlcn   The flcn
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
    if (status != LW_OK)
    {
        LWSWITCH_ASSERT(status == LW_OK);
        return status;
    }

    return status;
}

LwlStatus
lwswitch_get_soe_ucode_binaries_lr10
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
            "%s: SOE get ucode binaries BadArgs!\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    debug_mode = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_SCP, _CTL_STAT);
    debug_mode = DRF_VAL(_SOE, _SCP_CTL_STAT, _DEBUG_MODE, debug_mode);

    if (debug_mode)
    {
        *soe_ucode_data = soe_ucode_data_lr10_dbg;
        *soe_ucode_header = soe_ucode_header_lr10_dbg;
    }
    else
    {
        *soe_ucode_data = soe_ucode_data_lr10_prd;
        *soe_ucode_header = soe_ucode_header_lr10_prd;
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Load soe ucode image into SOE Falcon
 *
 * @param   device            The lwswitch device
 */
static LwlStatus
_lwswitch_load_soe_ucode_image
(
    lwswitch_device *device
)
{
    LwlStatus status;
    const LwU32 *soe_ucode_data;
    const LwU32 *soe_ucode_header;

    status = lwswitch_get_soe_ucode_binaries(device, &soe_ucode_data, &soe_ucode_header);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get SOE ucode binaries!\n",
            __FUNCTION__);
        return status;
    }

    status = _lwswitch_soe_copy_ucode_cpubitbang(device, soe_ucode_data,
                                                     soe_ucode_header);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to copy SOE ucode!\n",
            __FUNCTION__);
        return status;
    }

    return status;
}

/*
 * @Brief : Bootstrap SOE
 *
 * @param[in] device Bootstrap SOE on this device
 */
static LwlStatus
_lwswitch_soe_bootstrap
(
    lwswitch_device *device
)
{
    LWSWITCH_TIMEOUT timeout;
    LwU32 data;
    FLCN *pFlcn;

    // POR requires SOE, no SOE, total failure.
    if (!LWSWITCH_ENG_IS_VALID(device, SOE, 0))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: SOE is not present, failing driver load.\n",
            __FUNCTION__);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    pFlcn = device->pSoe->pFlcn;

    //
    // Start the SOE Falcon
    //
    data = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_CPUCTL);
    data = FLD_SET_DRF(_PFALCON, _FALCON_CPUCTL, _STARTCPU, _TRUE, data);
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_CPUCTL, data);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 20, &timeout);

    //
    // We will exit this if we recieve bootstrap signal OR
    // if we timeout waiting for bootstrap signal       OR
    // if bootstrap fails
    //
    while (1)
    {
        data = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_MAILBOX1);
        if (data == SOE_BOOTSTRAP_SUCCESS)
        {
            pFlcn->engDeslwc.initialized = LW_TRUE;
            return LWL_SUCCESS;
        }

        //
        // Check if SOE has halted unexpectedly.
        //
        // The explicit check is required because the interrupts
        // are not yet enabled as the device is still initializing.
        //
        if (soeIsCpuHalted_HAL(device, ((PSOE)pFlcn->pFlcnable)))
        {
            break;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for SOE to complete bootstrap!"
                "LW_PFALCON_FALCON_MAILBOX1 = 0x%08x\n",
                __FUNCTION__, data);
            LWSWITCH_ASSERT(0);
            break;
        }
    }

    LWSWITCH_PRINT(device, SETUP,
            "%s: Failed to bootstrap SOE.\n",
            __FUNCTION__);

    // Log any failures SOE may have had during bootstrap
    (void)soeService_HAL(device, ((PSOE)pFlcn->pFlcnable));
    return -LWL_ERR_ILWALID_STATE;
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
    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _MAILBOX0);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_MAILBOX0: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _MAILBOX1);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_MAILBOX1: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _CPUCTL);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_CPUCTL: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _DEBUGINFO);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_DEBUGINFO: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _EXCI);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_EXCI: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _SCTL);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_SCTL: 0x%x\n",
                   __FUNCTION__,
                   value);

    // Legacy steering and interrupts
    value = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _STEER_INTR_LEGACY);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_LWLSAW_LWSPMC_STEER_INTR_LEGACY: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_SOE_EN_LEGACY);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_LWLSAW_LWSPMC_INTR_SOE_EN_LEGACY: 0x%x\n",
                   __FUNCTION__,
                   value);

    // Correctable steering and interrupts
    value = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _STEER_INTR_CORRECTABLE);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_LWLSAW_LWSPMC_STEER_INTR_CORRECTABLE: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_SOE_EN_CORRECTABLE);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_LWLSAW_LWSPMC_INTR_SOE_EN_CORRECTABLE: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_SOE_LEGACY);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_LWLSAW_LWSPMC_INTR_SOE_LEGACY: 0x%x\n",
                   __FUNCTION__,
                   value);

    // EXTIO interrupts
    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_MISC, _EXTIO_IRQSTAT);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_MISC_EXTIO_IRQSTAT: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_MISC, _EXTIO_IRQMASK);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_MISC_EXTIO_IRQMASK: 0x%x\n",
                   __FUNCTION__,
                   value);


    // Falcon top level interrupts
    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQSTAT);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQSTAT: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQSTAT_ALIAS);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQSTAT_ALIAS: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQMODE);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQMODE: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQMASK);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQMASK: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQDEST);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQDEST: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQDEST2);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQDEST2: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQSCMASK);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: LW_SOE_FALCON_IRQSCMASK: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_REG_RD32(device, _THERM, _MSGBOX_COMMAND);
    LWSWITCH_PRINT(device, ERROR,
                   "%s: MSGBOX_COMMAND: 0x%x\n",
                   __FUNCTION__,
                   value);

    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _RESET_PRIV_LEVEL_MASK);
    LWSWITCH_PRINT(device, ERROR,
                "%s: LW_SOE_FALCON_RESET_PRIV_LEVEL_MASK: 0x%x\n",
                __FUNCTION__,
                value);
    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _IRQTMR_PRIV_LEVEL_MASK);
    LWSWITCH_PRINT(device, ERROR,
                "%s: LW_SOE_FALCON_IRQTMR_PRIV_LEVEL_MASK: 0x%x\n",
                __FUNCTION__,
                value);
    value = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _EXE_PRIV_LEVEL_MASK);
    LWSWITCH_PRINT(device, ERROR,
                "%s: LW_SOE_FALCON_EXE_PRIV_LEVEL_MASK: 0x%x\n",
                __FUNCTION__,
                value);
}
#endif // defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

/*
 * @Brief : Request SOE GFW image to exit and halt
 *
 *     i)   Query for SOE firmware validation state.
 *     ii)  Request for SOE to exit and halt.
 *     iii) Wait fot IFR to complete and exit by halting SOE.
 */
static LwlStatus
_lwswitch_soe_request_gfw_image_halt
(
    lwswitch_device *device
)
{
    LwU32 val;
    LWSWITCH_TIMEOUT timeout;
    FLCN* pFlcn = device->pSoe->pFlcn;

    //
    // Poll for firmware boot state.
    // GFW takes around 150ms to finish it's sequence.
    //
    lwswitch_timeout_create(1000 * LW_GFW_SOE_EXIT_AND_HALT_TIMEOUT, &timeout);
    do
    {
        val = LWSWITCH_REG_RD32(device, _GFW, _SOE_BOOT);
        if (FLD_TEST_DRF(_GFW, _SOE_BOOT, _PROGRESS, _COMPLETED, val) &&
            !FLD_TEST_DRF(_GFW, _SOE_BOOT, _VALIDATION_STATUS, _IN_PROGRESS, val))
        {
            break;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_TIMEOUT,
                "SOE reset timeout error(0)\n");
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for SOE GFW boot to complete. rc = 0x%x.\n",
                __FUNCTION__, val);
            return -LWL_ERR_ILWALID_STATE;
        }

        lwswitch_os_sleep(5);
    } while (LW_TRUE);

    // Check for firmware validation status.
    if (!FLD_TEST_DRF(_GFW, _SOE_BOOT, _VALIDATION_STATUS,
        _PASS_TRUSTED, val))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: SOE Firmware validation failed. rc = 0x%x\n",
            __FUNCTION__, val);
        return -LWL_ERR_ILWALID_STATE;
    }

    // Request SOE GFW image to exit and halt.
    val = LWSWITCH_REG_RD32(device, _GFW, _SOE_EXIT_AND_HALT);
    val = FLD_SET_DRF(_GFW, _SOE_EXIT_AND_HALT, _REQUESTED, _YES, val);
    LWSWITCH_REG_WR32(device, _GFW, _SOE_EXIT_AND_HALT, val);

    //
    // Wait for SOE to halt.
    //
    do
    {
        val = flcnRegRead_HAL(device, pFlcn, LW_SOE_FALCON_CPUCTL);
        if (FLD_TEST_DRF(_SOE, _FALCON_CPUCTL, _HALTED, _TRUE, val))
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: Handshake with SOE GFW successful.\n",
                __FUNCTION__);
            break;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_TIMEOUT,
                "SOE reset timeout error(1)\n");
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for SOE GFW image to exit and halt.\n",
                __FUNCTION__);
            return -LWL_ERR_ILWALID_STATE;
        }

        lwswitch_os_sleep(5);
    } while (LW_TRUE);

    return LWL_SUCCESS;
}

void
lwswitch_soe_unregister_events_lr10
(
    lwswitch_device *device
)
{
    PFLCN pFlcn = device->pSoe->pFlcn;
    PSOE   pSoe  = (PSOE)device->pSoe;
    LW_STATUS status;

    // un-register thermal callback funcion
    status = flcnQueueEventUnregister(device, pFlcn,
                                      pSoe->thermEvtDesc);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to un-register thermal event handler.\n",
            __FUNCTION__);
    }
}

/*
 * @Brief : Register callback functions for events
 *          and messages from SOE.
 */
LwlStatus
lwswitch_soe_register_event_callbacks_lr10
(
    lwswitch_device *device
)
{
    PFLCN pFlcn = device->pSoe->pFlcn;
    PSOE   pSoe  = (PSOE)device->pSoe;
    LW_STATUS status;

    // Register Thermal callback funcion
    status = flcnQueueEventRegister(
                 device, pFlcn,
                 RM_SOE_UNIT_THERM,
                 NULL,
                 lwswitch_therm_soe_callback_lr10,
                 NULL,
                 &pSoe->thermEvtDesc);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to register thermal event handler.\n",
            __FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Request SOE old driver image to provide L0 write permissions
 *          for reset registers to perform reset and boot up the new image.
 */
static LwlStatus
_lwswitch_soe_request_reset_permissions
(
    lwswitch_device *device
)
{
    LWSWITCH_TIMEOUT timeout;
    LwU32 reset_plm, engctl_plm;

    // Request reset access.
    LWSWITCH_REG_WR32(device, _SOE, _RESET_SEQUENCE,
        DRF_DEF(_SOE, _RESET_SEQUENCE, _REQUESTED, _YES));

    // Poll on reset PLMs.
    lwswitch_timeout_create(20 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    do
    {
        // Verify if SOE has given L0 write access for reset registers.
        reset_plm = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _RESET_PRIV_LEVEL_MASK);
        engctl_plm = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE_FALCON, _ENGCTL_PRIV_LEVEL_MASK);

        if (FLD_TEST_DRF(_SOE_FALCON, _RESET_PRIV_LEVEL_MASK,  _WRITE_PROTECTION_LEVEL0, _ENABLE, reset_plm) &&
            FLD_TEST_DRF(_SOE_FALCON, _ENGCTL_PRIV_LEVEL_MASK, _WRITE_PROTECTION_LEVEL0, _ENABLE, engctl_plm))
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: Got write access for reset registers from SOE.\n",
                __FUNCTION__);
            break;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_TIMEOUT,
                "SOE reset timeout error(2)\n");
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for SOE to provide write access for reset registers.\n",
                __FUNCTION__);

            LWSWITCH_PRINT(device, ERROR,
                "%s: LW_SOE_FALCON_RESET_PRIV_LEVEL_MASK = 0x%x, LW_SOE_FALCON_ENGCTL_PRIV_LEVEL_MASK = 0x%x.\n",
                __FUNCTION__, reset_plm, engctl_plm);

            return -LWL_ERR_ILWALID_STATE;
        }

        lwswitch_os_sleep(1);
    } while (LW_TRUE);

    return LWL_SUCCESS;
}

/*
 * @Brief : Execute SOE pre-reset sequence for secure reset.
 */
LwlStatus
lwswitch_soe_prepare_for_reset_lr10
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwU32 val;

    if (IS_FMODEL(device) || IS_RTLSIM(device) || IS_EMULATION(device))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Skipping SOE pre-reset sequence on pre-silicon.\n",
            __FUNCTION__);
        return LWL_SUCCESS;
    }

    val = LWSWITCH_REG_RD32(device, _GFW, _SOE_PROGRESS_CODE);
    if (!FLD_TEST_DRF(_GFW, _SOE_PROGRESS_CODE, _VALUE, _COMPLETED, val))
    {
        // Request SOE GFW to exit and halt.
        status = _lwswitch_soe_request_gfw_image_halt(device);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: GFW shutdown request failed!\n",
                __FUNCTION__);
        }
    }
    else
    {
        //
        // The SOE image from previous driver load needs to be reset.
        // Request reset permissions from that SOE image to perform the reset.
        //
        status = _lwswitch_soe_request_reset_permissions(device);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: SOE reset request failed!\n",
                __FUNCTION__);
        }
    }

    return status;
}

/*
 * @Brief : Bootstrap SOE on the specified device
 *
 * @param[in] device Bootstrap SOE on this device
 */
LwlStatus
lwswitch_init_soe_lr10
(
    lwswitch_device *device
)
{
    LwlStatus status;

    // Prepare SOE for reset.
    status = lwswitch_soe_prepare_for_reset(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_RESET,
            "Failed to reset SOE(0)\n");
        return status;
    }

    // Reset SOE
    status = _lwswitch_reset_soe(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_RESET,
            "Failed to reset SOE(1)\n");
        return status;
    }

    // Load SOE
    status = _lwswitch_load_soe_ucode_image(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_BOOTSTRAP,
            "Failed to boot SOE(0)\n");
        return status;
    }

    // Start SOE
   status = _lwswitch_soe_bootstrap(device);
   if (status != LWL_SUCCESS)
   {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_BOOTSTRAP,
            "Failed to boot SOE(1)\n");
        return status;
    }

    // Sanity the command and message queues as a final check
    if (_lwswitch_soe_send_test_cmd(device) != LW_OK)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_BOOTSTRAP,
            "Failed to boot SOE(2)\n");

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
        dumpDebugRegisters(device);
#endif // defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

        return -LWL_ERR_ILWALID_STATE;
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

    return status;
}

/**
 * @brief  SOE construct
 *
 * @param[in] device            lwswitch_device  pointer
 * @param[in] pFlcnable         SOE pointer
 *
 * @return LW_OK
 */
static LW_STATUS
_soeConstruct_LR10
(
    lwswitch_device *device,
    FLCNABLE        *pFlcnable
)
{
    SOE                *pSoe  = (PSOE)pFlcnable;
    FLCN               *pFlcn = ENG_GET_FLCN(pFlcnable);
    PFALCON_QUEUE_INFO  pQueueInfo;
    LW_STATUS           status;

    LWSWITCH_ASSERT(pFlcn != NULL);

    //
    // Set SOE specific Falcon state
    // This is where any default Falcon state should be overridden if necessary.
    //
    pFlcn->name                    = "SOE";
    pFlcn->pFlcnable               = pFlcnable;
    pFlcn->bQueuesEnabled          = LW_TRUE;
    pFlcn->numQueues               = SOE_QUEUE_NUM;
    pFlcn->numSequences            = RM_SOE_MAX_NUM_SEQUENCES;
    pFlcn->bEmemEnabled            = LW_TRUE;
    pFlcn->engineTag               = ENG_TAG_SOE;

    lwswitch_os_memset(pSoe->seqInfo, 0, sizeof(pSoe->seqInfo));

    // Do the HAL dependent init for Falcon
    status = flcnConstruct_HAL(device, pFlcn);

    pQueueInfo = pFlcn->pQueueInfo;
    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueueInfo->pQueues != NULL);

    //
    // Fill in the Message Queue handling details
    //
    pQueueInfo->maxUnitId          = RM_SOE_UNIT_END;
    pQueueInfo->maxMsgSize         = sizeof(RM_FLCN_MSG_SOE);
    pQueueInfo->initEventUnitId    = RM_SOE_UNIT_INIT;

    return status;
}

/**
 * @brief  SOE destruct
 *
 * @param[in] device            lwswitch_device  pointer
 * @param[in] pFlcnable         SOE pointer
 */
static void
_soeDestruct_LR10
(
    lwswitch_device *device,
    FLCNABLE        *pFlcnable
)
{
    flcnDestruct_HAL(device, ENG_GET_FLCN(pFlcnable));
}

 /*!
 * @brief Sets up the external configuration for accessing registers,etc.
 *
 * @param[in] device         lwswitch_device pointer
 * @param[in] pSoe           FLCNABLE pointer
 * @param[in] pConfig        FALCON_EXTERNAL_CONFIG pointer
 *
 * @returns void.
 */
static void
_soeGetExternalConfig_LR10
(
    lwswitch_device        *device,
    FLCNABLE               *pSoe,
    PFALCON_EXTERNAL_CONFIG pConfig
)
{
    PFLCN               pFlcn = ENG_GET_FLCN(pSoe);
    PFALCON_QUEUE_INFO  pQueueInfo;

    LWSWITCH_ASSERT(pFlcn != NULL);

    pConfig->bResetInPmc    = LW_TRUE;
    pConfig->fbifBase       = LW_SOE_FBIF_TRANSCFG(0);

    pQueueInfo = pFlcn->pQueueInfo;
    LWSWITCH_ASSERT(pQueueInfo != NULL);

    // Populate the falcon queue details
    pQueueInfo->cmdQHeadSize        = LW_SOE_QUEUE_HEAD__SIZE_1;
    pQueueInfo->cmdQTailSize        = LW_SOE_QUEUE_TAIL__SIZE_1;
    pQueueInfo->msgQHeadSize        = LW_SOE_MSGQ_HEAD__SIZE_1;
    pQueueInfo->msgQTailSize        = LW_SOE_MSGQ_TAIL__SIZE_1;

    pQueueInfo->cmdQHeadBaseAddress = LW_SOE_QUEUE_HEAD(0);
    pQueueInfo->cmdQHeadStride      = LW_SOE_QUEUE_HEAD(1) - LW_SOE_QUEUE_HEAD(0);
    pQueueInfo->cmdQTailBaseAddress = LW_SOE_QUEUE_TAIL(0);
    pQueueInfo->cmdQTailStride      = LW_SOE_QUEUE_TAIL(1) - LW_SOE_QUEUE_TAIL(0);
    pQueueInfo->msgQHeadBaseAddress = LW_SOE_MSGQ_HEAD(0);
    pQueueInfo->msgQTailBaseAddress = LW_SOE_MSGQ_TAIL(0);

    pQueueInfo->maxCmdQueueIndex    = SOE_RM_CMDQ_LOG_ID__LAST;
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
_soeService_LR10
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
    intrStatus = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQSTAT) &
                 flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQMASK) &
                 flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQDEST);

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
        soeServiceExterr_HAL(device, pSoe);
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
    intrStatus = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQSTAT) &
                 flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQMASK) &
                 flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_IRQDEST);

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
_soeServiceHalt_LR10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    PFLCN    pFlcn = ENG_GET_FLCN(pSoe);
    LwU32    value;

    LWSWITCH_PRINT(device, ERROR,
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!                   ** SOE HALTED **                !!\n"
                "!! Please file a bug with the following information. !!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n");

    // TODO: Break out the register dumps to specific hals
#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
        dumpDebugRegisters(device);
        flcnDbgInfoCapturePcTrace_HAL(device, pFlcn);
#endif // defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

    //
    // If the halt was related to security, we store the information in
    // MAILBOX0. Print out an error that clearly indicates the reason for the
    // halt.
    //
    value = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_MAILBOX0);

    if (value == LSF_FALCON_MODE_TOKEN_FLCN_INSELWRE)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "SOE HAS HALTED BECAUSE IT IS NOT RUNNING IN "
                    "SECURE MODE\n");
    }

    LWSWITCH_ASSERT(0);
}

/*!
 * Depending on the direction of the copy, copies 'sizeBytes' to/from 'pBuf'
 * from/to DMEM offset dmemAddr. Note the below statement about dmemAddr.
 * The address must be located in the EMEM region located directly above the
 * maximum virtual address of DMEM.
 *
 * @param[in]   device      lwswitch_device pointer
 * @param[in]   pSoe        SOE pointer
 * @param[in]   dmemAddr    The DMEM address for the copy
 * @param[out]  pBuf        pPointer to the buffer containing the data to copy
 * @param[in]   sizeBytes   The number of bytes to copy from EMEM
 * @param[in]   port        EMEM port
 * @param[in]   bCopyFrom   Boolean representing the copy direction (to/from EMEM)
 */
static void
_soeEmemTransfer_LR10
(
    lwswitch_device *device,
    PSOE             pSoe,
    LwU32            dmemAddr,
    LwU8            *pBuf,
    LwU32            sizeBytes,
    LwU8             port,
    LwBool           bCopyFrom
)
{
    LwU32       numWords;
    LwU32       numBytes;
    LwU32      *pData = (LwU32 *)pBuf;
    LwU32       startEmem;
    LwU32       endEmem;
    LwU32       reg32;
    LwU32       i;
    LwU32       ememCOffset;
    LwU32       ememDOffset;
    LwU32       maxEmemPorts = soeEmemPortSizeGet_HAL(device, pSoe);
    PFLCN       pFlcn        = ENG_GET_FLCN(pSoe);
    LW_STATUS status;

    if (pFlcn == NULL)
    {
        LWSWITCH_ASSERT(pFlcn != NULL);
        return;
    }

    status = soeEmemPortToRegAddr_HAL(device, pSoe, port, &ememCOffset, &ememDOffset);
    if (status != LW_OK)
    {
        LWSWITCH_ASSERT(status == LW_OK);
        return;
    }

    // Simply return if the copy-size is zero
    if (sizeBytes == 0)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: zero-byte copy requested.\n", __FUNCTION__);
        LWSWITCH_ASSERT(0);
        return;
    }

    // The source must be 4-byte aligned
    if (!LW_IS_ALIGNED(dmemAddr, FLCN_DMEM_ACCESS_ALIGNMENT))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Address is not 4-byte aligned. dmemAddr=0x%08x\n",
             __FUNCTION__, dmemAddr);
        LWSWITCH_ASSERT(0);
        return;
    }

    // Check the port. Only one port for SOE LR10.
    if (port >= maxEmemPorts)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: only %d ports supported. Accessed port=%d\n",
            __FUNCTION__, maxEmemPorts, port);
        LWSWITCH_ASSERT(0);
        return;
    }

    //
    // Verify that the dmemAddr address is located in EMEM, above addressable DMEM,
    // and that the copy does not overshoot the end of EMEM.
    //
    startEmem = soeGetEmemStartOffset_HAL(device, pSoe);

    // END_EMEM = START_EMEM + SIZE_EMEM (the size of EMEM is given in blocks)
    endEmem = startEmem + soeGetEmemSize_HAL(device, pSoe);

    if (dmemAddr < startEmem || (dmemAddr + sizeBytes) > endEmem)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: copy must be in EMEM aperature [0x%x, 0x%x)\n",
            __FUNCTION__, startEmem, endEmem);
        LWSWITCH_ASSERT(0);
        return;
    }

    // Colwert to EMEM offset for use by EMEMC/EMEMD
    dmemAddr -= startEmem;

    // Callwlate the number of words and bytes
    numWords = sizeBytes >> 2;
    numBytes = sizeBytes & 0x3;

    // Mask off all but the OFFSET and BLOCK in EMEM offset
    reg32 = dmemAddr & (DRF_SHIFTMASK(LW_SOE_EMEMC_OFFS) |
                   DRF_SHIFTMASK(LW_SOE_EMEMC_BLK));

    if (bCopyFrom)
    {
        // mark auto-increment on read
        reg32 = FLD_SET_DRF(_SOE, _EMEMC, _AINCR, _TRUE, reg32);
    }
    else
    {
        // mark auto-increment on write
        reg32 = FLD_SET_DRF(_SOE, _EMEMC, _AINCW, _TRUE, reg32);
    }
    flcnRegWrite_HAL(device, pFlcn, ememCOffset, reg32);

    // Directly copy as many words as possible
    for (i = 0; i < numWords; i++)
    {
        if (bCopyFrom)
        {
            pData[i] = flcnRegRead_HAL(device, pFlcn, ememDOffset);
        }
        else
        {
            flcnRegWrite_HAL(device, pFlcn, ememDOffset, pData[i]);
        }
    }

    // Check if there are leftover bytes to copy
    if (numBytes > 0)
    {
        LwU32 bytesCopied = numWords << 2;

        //
        // Read the contents first. If we're copying to the EMEM, we've set
        // autoincrement on write, so reading does not modify the pointer. We
        // can, thus, do a read/modify/write without needing to worry about the
        // pointer having moved forward. There is no special explanation needed
        // if we're copying from the EMEM since this is the last access to HW
        // in that case.
        //
        reg32 = flcnRegRead_HAL(device, pFlcn, ememDOffset);
        if (bCopyFrom)
        {
            for (i = 0; i < numBytes; i++)
            {
                pBuf[bytesCopied + i] = ((LwU8 *)&reg32)[i];
            }
        }
        else
        {
            for (i = 0; i < numBytes; i++)
            {
                ((LwU8 *)&reg32)[i] = pBuf[bytesCopied + i];
            }
            flcnRegWrite_HAL(device, pFlcn, ememDOffset, reg32);
        }
    }
}

/*!
 * Get the EMEM size in bytes
 *
 * @param[in]  device      lwswitch_device pointer
 * @param[in]  pSoe        SOE pointer
 */
static LwU32
_soeGetEmemSize_LR10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    LwU32 data = flcnRegRead_HAL(device, ENG_GET_FLCN(pSoe), LW_SOE_HWCFG);
    return DRF_VAL(_SOE, _HWCFG, _EMEM_SIZE, data) * FLCN_BLK_ALIGNMENT;
}

/*!
 * Get the EMEM start offset in DMEM VA space
 *
 * @param[in]  device      lwswitch_device pointer
 * @param[in]  pSoe        SOE pointer
 */
static LwU32
_soeGetEmemStartOffset_LR10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    //
    // EMEM is mapped at the top of DMEM VA space
    // START_EMEM = DMEM_VA_MAX = 2^(DMEM_TAG_WIDTH + 8)
    //
    LwU32 data = flcnRegRead_HAL(device, ENG_GET_FLCN(pSoe), LW_SOE_FALCON_HWCFG1);
    return (1 << (DRF_VAL(_SOE, _FALCON_HWCFG1, _DMEM_TAG_WIDTH, data) + 8));
}

/*!
 * Get the EMEMC/D register addresses for the specified port
 *
 * @param[in]  device      lwswitch_device pointer
 * @param[in]  pSoe        SOE pointer
 * @param[in]  port        EMEM port number
 * @param[out] pEmemCAddr  BAR0 address of the specified EMEMC port
 * @param[out] pEmemDAddr  BAR0 address of the specified EMEMD port
 */
static LW_STATUS
_soeEmemPortToRegAddr_LR10
(
    lwswitch_device *device,
    PSOE             pSoe,
    LwU32            port,
    LwU32           *pEmemCAddr,
    LwU32           *pEmemDAddr
)
{
    if (!pEmemCAddr || !pEmemDAddr)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (pEmemCAddr)
    {
        *pEmemCAddr = LW_SOE_EMEMC(port);
    }
    if (pEmemDAddr)
    {
        *pEmemDAddr = LW_SOE_EMEMD(port);
    }
    return LW_OK;
}

/*!
 * Called by soeService to handle a SOE exterr. This function will dump the
 * current status of SOE and then trap the CPU for further inspection for a
 * debug build.
 *
 * @param[in]  device  lwswitch_device object pointer
 * @param[in]  pSoe    SOE object pointer
 */
static void
_soeServiceExterr_LR10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    PFLCN pFlcn = ENG_GET_FLCN(pSoe);
    LwU32 extErrAddrOffset = 0, extErrStatOffset = 0;
    LwU32 exterrStatVal;

    LWSWITCH_PRINT(device, ERROR,
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!                   ** SOE EXTERR **                !!\n"
                "!! Please file a bug with the following information. !!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n");

    if (pFlcn == NULL)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    LWSWITCH_PRINT(device, ERROR,
        "<<<<<<<<<<< SOE DEBUG INFORMATION >>>>>>>>>>>\n");
    LWSWITCH_PRINT(device, ERROR,
        "OS VERSION (FALCON_OS): %u\n",
        flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_OS));

    if (soeGetExtErrRegAddrs_HAL(device, pSoe, &extErrAddrOffset, &extErrStatOffset) != LW_OK)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    LWSWITCH_PRINT(device, ERROR,
        "EXTERRADDR            : %u\n",
         flcnRegRead_HAL(device, pFlcn, extErrAddrOffset));

    exterrStatVal = flcnRegRead_HAL(device, pFlcn, extErrStatOffset);
    LWSWITCH_PRINT(device, ERROR,
        "EXTERRSTAT            : %u\n", exterrStatVal);
    LWSWITCH_PRINT(device, ERROR,
        "(AT PC)               : 0x%08X\n",
        DRF_VAL(_SOE, _FALCON_EXTERRSTAT, _PC, exterrStatVal));

    //
    // HW will continue to assert this interrupt as long as the _VALID bit is
    // set. Clear it to allow reporting of further failures since we have
    // already alerted the user that a transaction has failed.
    //
     flcnRegWrite_HAL(device, pFlcn, extErrStatOffset, FLD_SET_DRF(_SOE, _FALCON_EXTERRSTAT, _VALID, _FALSE, exterrStatVal));

    // Break to allow the user to inspect this on a debug build.
    LWSWITCH_ASSERT(0);
}

/*!
 * Get the bar0 offsets of LW_SOE_FALCON_EXTERRADDR and/or LW_SOE_FALCON_EXTERRSTAT
 *
 * @param[in]  device      lwswitch_device pointer
 * @param[in]  pSoe        SOE pointer
 * @param[out] pExtErrAddr BAR0 offset of LW_SOE_FALCON_EXTERRADDR
 * @param[out] pExtErrAddr BAR0 offset of LW_SOE_FALCON_EXTERRSTAT
 */
static LW_STATUS
_soeGetExtErrRegAddrs_LR10
(
    lwswitch_device *device,
    PSOE             pSoe,
    LwU32           *pExtErrAddr,
    LwU32           *pExtErrStat
)
{
     if (!pExtErrAddr || !pExtErrStat)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (pExtErrAddr)
    {
        *pExtErrAddr = LW_SOE_FALCON_EXTERRADDR;
    }
    if (pExtErrStat)
    {
        *pExtErrStat = LW_SOE_FALCON_EXTERRSTAT;
    }
    return LW_OK;
}

/*
 * Get number of EMEM ports
 *
 * @param[in]  device  lwswitch_device pointer
 * @param[in]  pSoe    SOE pointer
 */
static LwU32
_soeEmemPortSizeGet_LR10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    return LW_SOE_EMEMC__SIZE_1;
}

/**
 * @brief   sets pEngDeslwc and pEngDescBc to the discovered
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
_soeFetchEngines_LR10
(
    lwswitch_device         *device,
    FLCNABLE                *pSoe,
    ENGINE_DESCRIPTOR_TYPE  *pEngDeslwc,
    ENGINE_DESCRIPTOR_TYPE  *pEngDescBc
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

/*!
 * @brief Determine if the SOE Falcon CPU is halted
 *
 * @param[in] device         lwswitch_device  pointer
 * @param[in] pSoe           SOE  pointer
 *
 * @return LwBool reflecting the SOE Falcon CPU halted state
 */
static LwBool
_soeIsCpuHalted_LR10
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    LwU32 data = flcnRegRead_HAL(device, ENG_GET_FLCN(pSoe), LW_PFALCON_FALCON_CPUCTL);
    return (FLD_TEST_DRF(_PFALCON, _FALCON_CPUCTL, _HALTED, _TRUE, data));
}

static LwlStatus
_soeDmaStartTest
(
    lwswitch_device *device,
    void            *cpuAddr,
    LwU64           dmaHandle,
    LwU16           xferSize,
    LwU8            subCmd
)
{
    FLCN *pFlcn       = device->pSoe->pFlcn;
    LwU32               cmdSeqDesc;
    LW_STATUS           status;
    RM_FLCN_CMD_SOE     cmd;
    RM_SOE_CORE_CMD_DMA_TEST *pDmaCmd;
    LWSWITCH_TIMEOUT    timeout;

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));

    cmd.hdr.unitId = RM_SOE_UNIT_CORE;
    cmd.hdr.size   = sizeof(cmd);

    pDmaCmd = &cmd.cmd.core.dma_test;
    RM_FLCN_U64_PACK(&pDmaCmd->dmaHandle, &dmaHandle);
    pDmaCmd->xferSize    = xferSize;
    pDmaCmd->dataPattern = SOE_DMA_TEST_XFER_PATTERN;
    pDmaCmd->cmdType = RM_SOE_CORE_CMD_DMA_SELFTEST;
    pDmaCmd->subCmdType  = subCmd;

    cmdSeqDesc = 0;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 5, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn,
                                (PRM_FLCN_CMD)&cmd,
                                NULL,   // pMsg
                                NULL,   // pPayload
                                SOE_RM_CMDQ_LOG_ID,
                                &cmdSeqDesc,
                                &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to send DMA test command to SOE\n");
        return -LWL_ERR_ILWALID_STATE;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_soeValidateDmaTestResult
(
    lwswitch_device *device,
    void            *cpuAddr,
    LwU16           xferSize
)
{
    LwU16 iter;

    // Verify data written by SOE DMA matches what we expect.
    for (iter = 0; iter < SOE_DMA_TEST_BUF_SIZE; iter++)
    {
        LwU8 data = ((LwU8*) cpuAddr)[iter];

        // SOE would only touch data as much as the xfer size.
        if (iter < xferSize)
        {
            if (data != SOE_DMA_TEST_XFER_PATTERN)
            {
                LWSWITCH_PRINT(device, ERROR, "Incorrect data byte at offset %d = 0x%04x"
                                " for xfersize = %d\n", iter, data, xferSize);
                return -LWL_ERR_ILWALID_STATE;
            }
        }
        // We expect the rest of the data to be at init value.
        else
        {
            if (data != SOE_DMA_TEST_INIT_PATTERN)
            {
                LWSWITCH_PRINT(device, ERROR, "Incorrect data byte at offset %d = 0x%04x"
                                " for xferSize = 0x%04x\n", iter, data, xferSize);
                return -LWL_ERR_ILWALID_STATE;
            }
        }
    }

    return LWL_SUCCESS;
}

static LwlStatus
_soeDmaSelfTest
(
    lwswitch_device *device
)
{
    LwlStatus   ret;
    void        *cpuAddr;
    LwU64       dmaHandle;
    LwU16       xferSize;

    ret = lwswitch_os_alloc_contig_memory(device->os_handle, &cpuAddr, SOE_DMA_TEST_BUF_SIZE,
                                            (device->dma_addr_width == 32));

    if (ret != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "lwswitch_os_alloc_contig_memory returned %d\n", ret);
        return ret;
    }

    // SOE DMA Write test

    lwswitch_os_memset(cpuAddr, SOE_DMA_TEST_INIT_PATTERN, SOE_DMA_TEST_BUF_SIZE);

    ret = lwswitch_os_map_dma_region(device->os_handle, cpuAddr, &dmaHandle,
                                        SOE_DMA_TEST_BUF_SIZE, LWSWITCH_DMA_DIR_TO_SYSMEM);

    if (ret != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "lwswitch_os_map_dma_region returned %d\n", ret);
        goto _soeDmaSelfTest_exit;
    }

    // SOE DMA transfer sizes are in powers of 2.
    for (xferSize = SOE_DMA_MIN_SIZE; xferSize <= SOE_DMA_MAX_SIZE; xferSize <<= 1)
    {
        ret = lwswitch_os_sync_dma_region_for_device(device->os_handle, dmaHandle, SOE_DMA_TEST_BUF_SIZE,
                                                        LWSWITCH_DMA_DIR_TO_SYSMEM);

        if (ret != LWL_SUCCESS)
            break;

        ret = _soeDmaStartTest(device, cpuAddr, dmaHandle, xferSize,
                                RM_SOE_DMA_WRITE_TEST_SUBCMD);

        if (ret != LWL_SUCCESS)
            break;

        ret = lwswitch_os_sync_dma_region_for_cpu(device->os_handle, dmaHandle, SOE_DMA_TEST_BUF_SIZE,
                                                    LWSWITCH_DMA_DIR_TO_SYSMEM);

        if (ret != LWL_SUCCESS)
            break;

        ret = _soeValidateDmaTestResult(device, cpuAddr, xferSize);

        if (ret != LWL_SUCCESS)
            break;

        lwswitch_os_memset(cpuAddr, SOE_DMA_TEST_INIT_PATTERN, SOE_DMA_TEST_BUF_SIZE);
    }

    lwswitch_os_unmap_dma_region(device->os_handle, cpuAddr, dmaHandle,
                                    SOE_DMA_TEST_BUF_SIZE, LWSWITCH_DMA_DIR_TO_SYSMEM);

    if (ret != LWL_SUCCESS)
        goto _soeDmaSelfTest_exit;

    // SOE DMA read test

    lwswitch_os_memset(cpuAddr, SOE_DMA_TEST_INIT_PATTERN, SOE_DMA_TEST_BUF_SIZE);

    //
    // 4B/8B reads will overfetch 16B from PCIe. The Falcon logic ignores the extra
    // data. In the case of this test the reads only occur from the start of
    // a DMA mapped buffer which is larger than 16B, hence the selftest does
    // not need special handling for this behavior. However this will need to
    // be handled for other cases where SW cannot guarentee that the overfetch
    // will not exceed mapped regions.
    //
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    // https://p4viewer.lwpu.com/get/hw/doc/gpu/ampere/limerock/design/IAS/arch/limerock/publish/working/limerock_3P0_Full.html#_soe_dma
#endif
    //

    ret = lwswitch_os_map_dma_region(device->os_handle, cpuAddr, &dmaHandle,
                                        SOE_DMA_TEST_BUF_SIZE, LWSWITCH_DMA_DIR_FROM_SYSMEM);

    if (ret != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "lwswitch_os_map_dma_region returned %d\n", ret);
        goto _soeDmaSelfTest_exit;
    }

    for (xferSize = SOE_DMA_MIN_SIZE; xferSize <= SOE_DMA_MAX_SIZE; xferSize <<= 1)
    {
        ret = lwswitch_os_sync_dma_region_for_cpu(device->os_handle, dmaHandle, SOE_DMA_TEST_BUF_SIZE,
                                                    LWSWITCH_DMA_DIR_FROM_SYSMEM);

        if (ret != LWL_SUCCESS)
            break;

        // Fill in relevant data for the read test.
        lwswitch_os_memset(cpuAddr, SOE_DMA_TEST_XFER_PATTERN, xferSize);

        ret = lwswitch_os_sync_dma_region_for_device(device->os_handle, dmaHandle, SOE_DMA_TEST_BUF_SIZE,
                                                        LWSWITCH_DMA_DIR_FROM_SYSMEM);

        if (ret != LWL_SUCCESS)
            break;

        ret = _soeDmaStartTest(device, cpuAddr, dmaHandle, xferSize,
                                RM_SOE_DMA_READ_TEST_SUBCMD);

        if (ret != LWL_SUCCESS)
            break;
    }

    lwswitch_os_unmap_dma_region(device->os_handle, cpuAddr, dmaHandle,
                                    SOE_DMA_TEST_BUF_SIZE, LWSWITCH_DMA_DIR_FROM_SYSMEM);

_soeDmaSelfTest_exit:

    lwswitch_os_free_contig_memory(device->os_handle, cpuAddr, SOE_DMA_TEST_BUF_SIZE);

    return ret;
}

static LwlStatus
_soeTestDma_LR10
(
    lwswitch_device *device
)
{
    LwlStatus retval;

    retval = _soeDmaSelfTest(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "SOE DMA selftest failed\n");
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "SOE DMA selftest succeeded\n");
    }

    return retval;
}

/*!
 * @brief Send the EOM parameters to SOE
 *
 * @param[in]   device    lwswitch device pointer
 * @param[in]   mode      Node of EOM
 * @param[in]   nblks     Number of blocks
 * @param[in]   nerrs     Number of Errors.
 *
 * @return      LWL_SUCCESS on success
 */
static LwlStatus
_soeSetPexEOM_LR10
(
    lwswitch_device *device,
    LwU8 mode,
    LwU8 nblks,
    LwU8 nerrs,
    LwU8 berEyeSel
)
{
    FLCN               *pFlcn = device->pSoe->pFlcn;
    LwU32               cmdSeqDesc = 0;
    LW_STATUS           status = LW_OK;
    RM_FLCN_CMD_SOE     cmd;
    RM_SOE_BIF_CMD_EOM *pBifCmd = NULL;
    LWSWITCH_TIMEOUT    timeout = {0};

   lwswitch_os_memset(&cmd, 0, sizeof(cmd));

    cmd.hdr.unitId = RM_SOE_UNIT_BIF;
    cmd.hdr.size   = RM_SOE_CMD_SIZE(BIF, EOM);
    cmd.cmd.bif.cmdType = RM_SOE_BIF_CMD_UPDATE_EOM;
    //
    // We use SOE to set the EOM UPHY register since its Decode trapped and
    // hence CPU accessible.
    //
    pBifCmd = &cmd.cmd.bif.eomctl;
    pBifCmd->mode  = mode;
    pBifCmd->nblks = nblks;
    pBifCmd->nerrs = nerrs;
    pBifCmd->berEyeSel = berEyeSel;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 5, &timeout);

    status = flcnQueueCmdPostBlocking(device,
                 pFlcn,
                 (PRM_FLCN_CMD)&cmd,
                 NULL,   // pMsg
                 NULL,   // pPayload
                 SOE_RM_CMDQ_LOG_ID,
                 &cmdSeqDesc,
                 &timeout);

    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to Set EOM via SOE, Error 0x%x\n",
            __FUNCTION__, status);
        return -LWL_ERR_ILWALID_STATE;
    }

    return LWL_SUCCESS;
}

/*!
 * @brief Send the EOM parameters to SOE
 *
 * @param[in]   device    lwswitch device pointer
 * @param[in]   mode      Node of EOM
 * @param[in]   nblks     Number of blocks
 * @param[in]   nerrs     Number of Errors.
 *
 * @return      LWL_SUCCESS on success
 */
static LwlStatus
_soeGetPexEomStatus_LR10
(
    lwswitch_device *device,
    LwU8 mode,
    LwU8 nblks,
    LwU8 nerrs,
    LwU8 berEyeSel,
    LwU32 laneMask,
    LwU16 *pEomStatus
)
{
    FLCN *pFlcn = device->pSoe->pFlcn;
    LwU32 cmdSeqDesc = 0;
    LW_STATUS status = LW_OK;
    RM_FLCN_CMD_SOE cmd;
    RM_SOE_BIF_CMD_EOM_STATUS *pBifCmd = NULL;
    LWSWITCH_TIMEOUT timeout = { 0 };
    LwU64 dmaHandle = 0;
    LwU8 *pReadBuffer = NULL;
    LwU32 bufferSize = BIF_MAX_PCIE_LANES * sizeof(LwU16);

    if (bufferSize > SOE_DMA_MAX_SIZE)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Buffer size too large\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    // Create DMA mapping
    status = lwswitch_os_alloc_contig_memory(device->os_handle,
                 (void**)&pReadBuffer, bufferSize, (device->dma_addr_width == 32));
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to allocate contig memory\n",
            __FUNCTION__);
        return status;
    }

    status = lwswitch_os_map_dma_region(device->os_handle,
                                        pReadBuffer,
                                        &dmaHandle,
                                        bufferSize,
                                        LWSWITCH_DMA_DIR_TO_SYSMEM);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to map dma region to sysmem\n");
        lwswitch_os_free_contig_memory(device->os_handle, pReadBuffer, bufferSize);
        return status;
    }

   lwswitch_os_memset(&cmd, 0, sizeof(cmd));
   lwswitch_os_memset(pReadBuffer, 0, bufferSize);

    cmd.hdr.unitId = RM_SOE_UNIT_BIF;
    cmd.hdr.size   = RM_SOE_CMD_SIZE(BIF, EOM_STATUS);
    cmd.cmd.bif.cmdType = RM_SOE_BIF_CMD_GET_EOM_STATUS;

    pBifCmd = &cmd.cmd.bif.eomStatus;
    pBifCmd->mode  = mode;
    pBifCmd->nblks = nblks;
    pBifCmd->nerrs = nerrs;
    pBifCmd->berEyeSel = berEyeSel;
    pBifCmd->laneMask = laneMask;
    RM_FLCN_U64_PACK(&pBifCmd->dmaHandle, &dmaHandle);

    status = lwswitch_os_sync_dma_region_for_device(device->os_handle, dmaHandle,
        bufferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to yield to DMA controller\n");
        goto _soeGetPexEomStatus_LR10_exit;
    }

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 5, &timeout);

    status = flcnQueueCmdPostBlocking(device,
                 pFlcn,
                 (PRM_FLCN_CMD)&cmd,
                 NULL,   // pMsg
                 NULL,   // pPayload
                 SOE_RM_CMDQ_LOG_ID,
                 &cmdSeqDesc,
                 &timeout);

    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to Get EOM status via SOE, Error 0x%x\n",
            __FUNCTION__, status);
        status = -LWL_ERR_ILWALID_STATE;
        goto _soeGetPexEomStatus_LR10_exit;
    }

    status = lwswitch_os_sync_dma_region_for_cpu(device->os_handle, dmaHandle,
        bufferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "DMA controller failed to yield back\n");
        goto _soeGetPexEomStatus_LR10_exit;
    }

    lwswitch_os_memcpy(((LwU8*)pEomStatus), pReadBuffer, bufferSize);

_soeGetPexEomStatus_LR10_exit :
    lwswitch_os_unmap_dma_region(device->os_handle, pReadBuffer, dmaHandle,
        bufferSize, LWSWITCH_DMA_DIR_TO_SYSMEM);
    lwswitch_os_free_contig_memory(device->os_handle, pReadBuffer, bufferSize);

    return status;
}

/*!
 * @brief Get the register values of UPHY registers
 *
 * Read the register value from a scratch register updated by SOE.
 *
 * @param[in]   device            lwswitch device pointer
 * @param[in]   regAddress        Register address whose value is to be retrieved
 * @param[in]   laneSelectMask    Mask of lanes to read from
 * @param[out]  *pRegValue        Value of register address
 *
 * Read the register value from a scratch register updated by SOE.
 *
 * @return      LWL_SUCCESS on success
 */
static LwlStatus
_soeGetUphyDlnCfgSpace_LR10
(
    lwswitch_device *device,
    LwU32 regAddress,
    LwU32 laneSelectMask,
    LwU16 *pRegValue
)
{
    FLCN               *pFlcn = device->pSoe->pFlcn;
    LwU32               cmdSeqDesc = 0;
    LW_STATUS           status = LW_OK;
    RM_FLCN_CMD_SOE     cmd;
    RM_SOE_BIF_CMD_UPHY_DLN_CFG_SPACE *pBifCmd = NULL;
    LWSWITCH_TIMEOUT    timeout = { 0 };

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));

    cmd.hdr.unitId = RM_SOE_UNIT_BIF;
    cmd.hdr.size = RM_SOE_CMD_SIZE(BIF, UPHY_DLN_CFG_SPACE);
    cmd.cmd.bif.cmdType = RM_SOE_BIF_CMD_GET_UPHY_DLN_CFG_SPACE;

    pBifCmd = &cmd.cmd.bif.cfgctl;
    pBifCmd->regAddress = regAddress;
    pBifCmd->laneSelectMask = laneSelectMask;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    status = flcnQueueCmdPostBlocking(device,
                 pFlcn,
                 (PRM_FLCN_CMD)&cmd,
                 NULL,   // pMsg
                 NULL,   // pPayload
                 SOE_RM_CMDQ_LOG_ID,
                 &cmdSeqDesc,
                 &timeout);

    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to execute BIF GET_UPHY_DLN_CFG_SPACE via SOE, Error 0x%x\n",
            __FUNCTION__, status);
        return -LWL_ERR_ILWALID_STATE;
    }

    *pRegValue = LWSWITCH_SOE_RD32_LR10(device, 0, _SOE, _MAILBOX(0));

    return LWL_SUCCESS;
}

static LwlStatus
_soeForceThermalSlowdown_LR10
(
    lwswitch_device *device,
    LwBool slowdown,
    LwU32  periodUs
)
{
    FLCN               *pFlcn = device->pSoe->pFlcn;
    LwU32               cmdSeqDesc = 0;
    LW_STATUS           status = LW_OK;
    RM_FLCN_CMD_SOE     cmd;
    LWSWITCH_TIMEOUT    timeout = {0};

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));
    cmd.hdr.unitId = RM_SOE_UNIT_THERM;
    cmd.hdr.size = sizeof(cmd);
    cmd.cmd.therm.cmdType = RM_SOE_THERM_FORCE_SLOWDOWN;
    cmd.cmd.therm.slowdown.slowdown = slowdown;
    cmd.cmd.therm.slowdown.periodUs = periodUs;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    status = flcnQueueCmdPostBlocking(device, pFlcn,
                                  (PRM_FLCN_CMD)&cmd,
                                  NULL,   // pMsg             - not used for now
                                  NULL,   // pPayload         - not used for now
                                  SOE_RM_CMDQ_LOG_ID,
                                  &cmdSeqDesc,
                                  &timeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Thermal slowdown failed. rc:%d\n",
            __FUNCTION__, status);

        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_soeSetPcieLinkSpeed_LR10
(
    lwswitch_device *device,
    LwU32 linkSpeed
)
{
    FLCN               *pFlcn = device->pSoe->pFlcn;
    LwU32               cmdSeqDesc = 0;
    LW_STATUS           status = LW_OK;
    RM_FLCN_CMD_SOE     cmd;
    RM_SOE_BIF_CMD_PCIE_LINK_SPEED *pBifCmd = NULL;
    LWSWITCH_TIMEOUT    timeout = { 0 };

    lwswitch_os_memset(&cmd, 0, sizeof(cmd));

    cmd.hdr.unitId = RM_SOE_UNIT_BIF;
    cmd.hdr.size = RM_SOE_CMD_SIZE(BIF, PCIE_LINK_SPEED);
    cmd.cmd.bif.cmdType = RM_SOE_BIF_CMD_SET_PCIE_LINK_SPEED;

    pBifCmd = &cmd.cmd.bif.speedctl;
    pBifCmd->linkSpeed = linkSpeed;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    status = flcnQueueCmdPostBlocking(device,
                 pFlcn,
                 (PRM_FLCN_CMD)&cmd,
                 NULL,   // pMsg
                 NULL,   // pPayload
                 SOE_RM_CMDQ_LOG_ID,
                 &cmdSeqDesc,
                 &timeout);

    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to execute BIF SET_PCIE_LINK_SPEED via SOE, Error 0x%x\n",
            __FUNCTION__, status);
        return -LWL_ERR_ILWALID_STATE;
    }

    return LWL_SUCCESS;
}

/*!
 * @brief   set hal function pointers for functions defined in LR10 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcnable   The flcnable for which to set hals
 */
void
soeSetupHal_LR10
(
    SOE *pSoe
)
{
    soe_hal *pHal = pSoe->base.pHal;
    flcnable_hal *pParentHal = (flcnable_hal *)pHal;
    //set any functions we want to override
    pParentHal->construct            = _soeConstruct_LR10;
    pParentHal->destruct             = _soeDestruct_LR10;
    pParentHal->getExternalConfig    = _soeGetExternalConfig_LR10;
    pParentHal->fetchEngines         = _soeFetchEngines_LR10;

    // set any functions specific to SOE
    pHal->service               = _soeService_LR10;
    pHal->serviceHalt           = _soeServiceHalt_LR10;
    pHal->getEmemSize           = _soeGetEmemSize_LR10;
    pHal->ememTransfer          = _soeEmemTransfer_LR10;
    pHal->getEmemSize           = _soeGetEmemSize_LR10;
    pHal->getEmemStartOffset    = _soeGetEmemStartOffset_LR10;
    pHal->ememPortToRegAddr     = _soeEmemPortToRegAddr_LR10;
    pHal->serviceExterr         = _soeServiceExterr_LR10;
    pHal->getExtErrRegAddrs     = _soeGetExtErrRegAddrs_LR10;
    pHal->ememPortSizeGet       = _soeEmemPortSizeGet_LR10;
    pHal->isCpuHalted           = _soeIsCpuHalted_LR10;
    pHal->testDma               = _soeTestDma_LR10;
    pHal->setPexEOM             = _soeSetPexEOM_LR10;
    pHal->getUphyDlnCfgSpace    = _soeGetUphyDlnCfgSpace_LR10;
    pHal->forceThermalSlowdown  = _soeForceThermalSlowdown_LR10;
    pHal->setPcieLinkSpeed      = _soeSetPcieLinkSpeed_LR10;
    pHal->getPexEomStatus       = _soeGetPexEomStatus_LR10;
}
