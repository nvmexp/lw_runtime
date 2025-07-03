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
#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/minion_ls10.h"
#include "ls10/minion_production_ucode_ls10_dbg.h"
#include "ls10/minion_ucode_riscv_ls10_no_manifest_dbg.h"
#include "ls10/minion_ucode_riscv_ls10_manifest_dbg.h"
#include "regkey_lwswitch.h"

#include "lwswitch/ls10/dev_minion_ip.h"
#include "lwswitch/ls10/dev_minion_ip_addendum.h"
#include "lwswitch/ls10/dev_ingress_ip.h"
#include "lwswitch/ls10/dev_egress_ip.h"
#include "lwswitch/ls10/dev_riscv_pri.h"
#include "lwswitch/ls10/lw_minion_riscv_address_map.h"

#include "flcn/flcn_lwswitch.h"

/*
 * @Brief : Find the link # of the first valid link for the given MINION
 *
 *
 * @param[in] device Bootstrap MINIONs on this device
 */
static LwU32
_lwswitch_minion_get_first_valid_link
(
    lwswitch_device *device,
    LwU32            minionInstIdx
)
{
    LwU32 i;

    // Search across links connected to a single MINION
    for (i = 0; i < LWSWITCH_LINKS_PER_MINION_LS10; ++i)
    {
        //
        // If the globalLink has a valid MINION instance (i.e. link is marked
        // as valid) then break
        //
        if(LWSWITCH_IS_LINK_ENG_VALID_LS10(device, MINION,
                    minionInstIdx*LWSWITCH_LINKS_PER_MINION_LS10 + i))
        {
            break;
        }
    }

    return i;
}

/*
 * @Brief : Check if MINION is already running.
 *
 * The function assumes that if one of MINIONs is running, all of them are
 * running. This approach needs to be fixed.
 *
 * TODO: Refactor minion code to check for each minion's status individually.
 *
 * @param[in] device Bootstrap MINIONs on this device
 */
static LwBool
_lwswitch_check_running_minions
(
    lwswitch_device *device
)
{
    LwU32  data, i;
    LwBool bMinionRunning = LW_FALSE;

    for (i = 0; i < NUM_MINION_ENGINE_LS10; i++)
    {
        if (!LWSWITCH_ENG_VALID_LS10(device, MINION, i))
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s: MINION instance %d is not valid.\n",
                 __FUNCTION__, i);
            continue;
        }

        data = LWSWITCH_MINION_RD32_LS10(device, i, _CMINION, _FALCON_IRQSTAT);
        if (FLD_TEST_DRF(_CMINION, _FALCON_IRQSTAT, _HALT, _FALSE, data))
        {
            data = LWSWITCH_MINION_RD32_LS10(device, i, _MINION, _MINION_STATUS);
            if (FLD_TEST_DRF(_MINION,  _MINION_STATUS, _STATUS, _BOOT, data))
            {
                //
                // Set initialized flag if MINION is running.
                // We don't want to bootstrap a falcon that is already running.
                //
                lwswitch_set_minion_initialized(device, i, LW_TRUE);

                LWSWITCH_PRINT(device, SETUP,
                    "%s: MINION instance %d is already bootstrapped.\n",
                    __FUNCTION__, i);
                bMinionRunning = LW_TRUE;
            }
        }
    }

    return bMinionRunning;
}

/*
 * @Brief : MINION set ucode target routine
 *          Identifies which target to boot Minion with
 *
 * @param[in] device      The Lwswitch device
 * @param[in] idx_minion  MINION instance to use
 */
static LwlStatus
_lwswitch_minion_set_ucode_target
(
    lwswitch_device *device,
    LwU32 idx_minion
)
{
    LwlStatus status = LWL_SUCCESS;
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    LwU32 bcrCtrl = LWSWITCH_MINION_RD32_LS10(device, idx_minion, _CMINION, _RISCV_BCR_CTRL);

    if (!LWSWITCH_ENG_VALID_LS10(device, MINION, idx_minion))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: MINION instance %d is not valid.\n",
             __FUNCTION__, idx_minion);
        return LWL_SUCCESS;
    }

    if (device->regkeys.set_ucode_target == LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_RISCV ||
        device->regkeys.set_ucode_target == LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_RISCV_MANIFEST)
    {
        if (FLD_TEST_DRF(_CMINION, _RISCV_BCR_CTRL, _VALID, _TRUE, bcrCtrl) &&
            !FLD_TEST_DRF(_CMINION, _RISCV_BCR_CTRL, _CORE_SELECT, _RISCV, bcrCtrl))
        {
            LWSWITCH_PRINT(device, ERROR,
                      "Minion RISCV boot requested, but not allowed via BCR_CTRL!!\n");
            status = -LWL_ERR_ILWALID_STATE;
            return status;
        }

        LWSWITCH_MINION_WR32_LS10(device, idx_minion, _CMINION_RISCV, _BCR_CTRL,
                        DRF_DEF(_CMINION_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV));

        chip_device->minionEngArch = LW_UPROC_ENGINE_ARCH_FALCON_RISCV;

        if (device->regkeys.set_ucode_target == LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_RISCV_MANIFEST)
        {
            chip_device->riscvManifestBoot = LW_TRUE;
        }
    }
    else if (device->regkeys.set_ucode_target == LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_FALCON)
    {
        if (FLD_TEST_DRF(_CMINION, _RISCV_BCR_CTRL, _VALID, _TRUE, bcrCtrl) &&
            !FLD_TEST_DRF(_CMINION, _RISCV_BCR_CTRL, _CORE_SELECT, _FALCON, bcrCtrl))
        {
            LWSWITCH_PRINT(device, ERROR,
                      "Minion FALCON boot requested, but not allowed via BCR_CTRL!!\n");
            status = -LWL_ERR_ILWALID_STATE;
            return status;
        }

        LWSWITCH_MINION_WR32_LS10(device, idx_minion, _CMINION_RISCV, _BCR_CTRL,
                        DRF_DEF(_CMINION_RISCV, _BCR_CTRL, _CORE_SELECT, _FALCON));

        chip_device->minionEngArch = LW_UPROC_ENGINE_ARCH_FALCON;
    }
    else
    {
        // Use CORE_SELECT to determine which core to boot
        if (FLD_TEST_DRF(_CMINION, _RISCV_BCR_CTRL, _CORE_SELECT, _RISCV, bcrCtrl))
        {
            LWSWITCH_PRINT(device, SETUP,
                      "Minion RISCV boot is enabled by default\n");

            chip_device->minionEngArch = LW_UPROC_ENGINE_ARCH_FALCON_RISCV;
        }
        else
        {
            LWSWITCH_PRINT(device, SETUP,
                      "Minion FALCON boot is enabled by default\n");

            chip_device->minionEngArch = LW_UPROC_ENGINE_ARCH_FALCON;
        }
    }
    return status;
}

/*
 * @Brief : MINION pre init routine
 *          Waits for scrubbing to finish
 *
 * @param[in] device  MINIONs on this device
 */
static LwlStatus
_lwswitch_minion_pre_init
(
    lwswitch_device *device
)
{
    LwU32            data;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwU32            idx_minion;
    LwlStatus        status = LWL_SUCCESS;
    LwU32            falconIntrMask, falconIntrDest;
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    // Find first valid MINION instance
    for (idx_minion = 0; idx_minion < NUM_MINION_ENGINE_LS10; idx_minion++)
    {
        if (LWSWITCH_ENG_VALID_LS10(device, MINION, idx_minion))
        {
            break;
        }
    }
    if (idx_minion >= NUM_MINION_ENGINE_LS10)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: No MINIONs instantiated.  Skipping MINION pre-init\n",
            __FUNCTION__);
        goto _lwswitch_minion_pre_init_exit;
    }

    status = _lwswitch_minion_set_ucode_target(device, idx_minion);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION core_select failed\n",
            __FUNCTION__);
        return status;
    }

    // Since we are not using Falcon DMA to load ucode, set REQUIRE_CTX to FALSE
    LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_DMACTL, 0x0);

    //
    // Initialize falcon specific interrupts before MINION is loaded.
    // Once MINION is loaded, these registers get locked down.
    //

    // falcon interrupt mask is set through IRQMSET
    falconIntrMask = (DRF_DEF(_CMINION, _FALCON_IRQMSET, _WDTMR, _SET) |
                      DRF_DEF(_CMINION, _FALCON_IRQMSET, _HALT, _SET)  |
                      DRF_DEF(_CMINION, _FALCON_IRQMSET, _EXTERR, _SET)|
                      DRF_DEF(_CMINION, _FALCON_IRQMSET, _SWGEN0, _SET)|
                      DRF_DEF(_CMINION, _FALCON_IRQMSET, _SWGEN1, _SET));

    // falcon interrupt routing to the HOST
    falconIntrDest = (DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_WDTMR,  _HOST) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_HALT,   _HOST) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_EXTERR, _HOST) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_SWGEN0, _HOST) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_SWGEN1,   _HOST)        |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_WDTMR,  _HOST_NORMAL) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_HALT,   _HOST_NORMAL) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_EXTERR, _HOST_NORMAL) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_SWGEN0, _HOST_NORMAL) |
                      DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_SWGEN1, _HOST_NORMAL));

    LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IRQMSET, falconIntrMask);
    LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IRQDEST, falconIntrDest);
    chip_device->intr_minion_dest = falconIntrDest;

    //
    // As soon as we access any falcon reg (above), the scrubber will start scrubbing
    // IMEM and DMEM. Wait for the scrubber to finish scrubbing.
    //
    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(30*LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    }
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        // Check if scrubbing was done for first enabled MINION
        data = LWSWITCH_MINION_RD32_LS10(device, idx_minion, _CMINION, _FALCON_DMACTL);
        if (FLD_TEST_DRF(_CMINION, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, data) &&
            FLD_TEST_DRF(_CMINION, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, data))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (!FLD_TEST_DRF(_CMINION, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, data) ||
        !FLD_TEST_DRF(_CMINION, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, data))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout waiting for scrubbing to finish on MINION %d.\n",
            __FUNCTION__, idx_minion);
        status = -LWL_ERR_ILWALID_STATE;
        return status;
    }

_lwswitch_minion_pre_init_exit:
    return status;
}

/*
 * @Brief : Copy the minion ucode to IMEM and DMEM in broadcast mode
 *
 * @param[in] device  Copy ucode to all MINIONS associated with the device
 */
static LwlStatus
_lwswitch_minion_copy_ucode_bc
(
    lwswitch_device                *device,
    const LwU32                    *minion_ucode_data,
    const LwU32                    *minion_ucode_header
)
{
    const LwU32 *pHeader = &minion_ucode_header[0];

    LwU32 osCodeOffset   = 0;
    LwU32 osCodeSize     = 0;
    LwU32 osDataOffset   = 0;
    LwU32 osDataSize     = 0;
    LwU32 manifestOffset = 0;
    LwU32 manifestSize   = 0;
    LwU32 numApps        = 0;

    LwU32 data, i, app, dataSize;
    LwU16 tag;
    LwU32 idx_minion;
    LwU32 appDataOffset, appDataSize;
    LwU32 appCodeOffset, appCodeSize;
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    // Find first valid MINION instance
    for (idx_minion = 0; idx_minion < NUM_MINION_ENGINE_LS10; idx_minion++)
    {
        if (LWSWITCH_ENG_VALID_LS10(device, MINION, idx_minion))
        {
            break;
        }
    }

    if (idx_minion >= NUM_MINION_ENGINE_LS10)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: No MINIONs instantiated. Skipping MINION ucode load\n",
            __FUNCTION__);
        goto _lwswitch_minion_copy_ucode_bc_exit;
    }

    if (chip_device->minionEngArch == LW_UPROC_ENGINE_ARCH_FALCON_RISCV)
    {
         PRISCV_UCODE_HDR_INFO_LS10 pUcodeHeader =
            (PRISCV_UCODE_HDR_INFO_LS10) &minion_ucode_header[0];

        osCodeOffset   = pUcodeHeader->monitorCodeOffset;
        osCodeSize     = pUcodeHeader->monitorCodeSize;
        osDataOffset   = pUcodeHeader->monitorDataOffset;
        osDataSize     = pUcodeHeader->monitorDataSize;
        manifestOffset = pUcodeHeader->manifestOffset;
        manifestSize   = pUcodeHeader->manifestSize;
    }
    else
    {
         PFALCON_UCODE_HDR_INFO_LS10 pUcodeHeader =
            (PFALCON_UCODE_HDR_INFO_LS10) &minion_ucode_header[0];

        osCodeOffset   = pUcodeHeader->osCodeOffset;
        osCodeSize     = pUcodeHeader->osCodeSize;
        osDataOffset   = pUcodeHeader->osDataOffset;
        osDataSize     = pUcodeHeader->osDataSize;
        numApps        = pUcodeHeader->numApps;
    }

    dataSize = sizeof(minion_ucode_data[0]);

    // Initialize address of IMEM to 0x0 and set auto-increment on write
    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_IMEMC, _AINCW, _TRUE, data);
    LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMC(0), data);

    //
    // Initialize IMEM tag to 0 explicitly even though power-on value is 0.
    // Writes to IMEM don't work if we don't do this
    //
    tag = 0;
    LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMT(0), tag);

    // Copy over IMEM part of the ucode and tag along the way
    for (i = 0; i < (osCodeSize / dataSize) ; i++)
    {
        // Increment tag for after every block (256 bytes)
        if (i && ((i % (FALCON_IMEM_BLK_SIZE_BYTES_LS10 / dataSize)) == 0))
        {
            tag++;
            LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMT(0), (LwU32) tag);
        }

        // Copy IMEM DWORD by DWORD
        data = minion_ucode_data[(osCodeOffset / dataSize) + i];
        LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMD(0), data);
    }

    // Initialize address of DMEM to 0x0 and set auto-increment on write
    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_DMEMC, _AINCW, _TRUE, data);
    LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_DMEMC(0), data);

    // Copy over DMEM part of the ucode
    for (i = 0; i < (osDataSize / dataSize) ; i++)
    {
        // Copy DMEM DWORD by DWORD
        data = minion_ucode_data[(osDataOffset / dataSize) + i];
        LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_DMEMD(0), data);
    }

    // 
    // Falcon Case
    // Copy over any apps in the ucode with the appropriate tags
    //
    if (numApps)
    {
        for (app = 0; app < numApps ; app++)
        {
            // Index into the app code info
            appCodeOffset = pHeader[FALCON_CODE_HDR_APP_CODE_START_LS10 + 2*app];
            appCodeSize   = pHeader[FALCON_CODE_HDR_APP_CODE_START_LS10 + 2*app + 1];

            // Index into the app data info using appCodeStart offset as a base
            appDataOffset = pHeader[FALCON_CODE_HDR_APP_CODE_START_LS10 +
                2*numApps + 2*app];
            appDataSize   = pHeader[FALCON_CODE_HDR_APP_CODE_START_LS10 +
                2*numApps + 2*app + 1];


            // Mark the following IMEM blocks as secure
            data = LWSWITCH_MINION_RD32_LS10(device, idx_minion, _CMINION, _FALCON_IMEMC(0));
            data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _SELWRE, 0x1, data);
            LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMC(0), data);

            // Copy to IMEM and tag along the way
            tag = (LwU16)(appCodeOffset >> 8);
            LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMT(0), tag);

            // Copy app code to IMEM picking up where last load left off
            for (i = 0; i < (appCodeSize / dataSize); i++)
            {
                if (i && ((i % (FALCON_IMEM_BLK_SIZE_BYTES_LS10 / dataSize)) == 0))
                {
                    tag++;
                    LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMT(0), tag);
                }

                data = minion_ucode_data[(appCodeOffset / dataSize) + i];
                LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_IMEMD(0), data);
            }
        }

        // Copy app data to DMEM picking up where last load left off
        for (i = 0; i < (appDataSize / dataSize); i++)
        {
            data = minion_ucode_data[(appDataOffset/ dataSize) + i];
            LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_DMEMD(0), data);
        }
    }

    //
    // RISCV Case (Manifest)
    // Note: We are not copying over imem for the RISCV Manifest case. That is
    // because manifest as a secure app is dmem only.
    //
    if (chip_device->riscvManifestBoot)
    {
        // Set address of DMEM to Manifest offset and set auto-increment on write
        data = manifestOffset - osDataOffset;
        data = FLD_SET_DRF(_CMINION, _FALCON_DMEMC, _AINCW, _TRUE, data);
        LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_DMEMC(0), data);

        // Copy app data to DMEM picking up where last load left off
        for (i = 0; i < (manifestSize / dataSize); i++)
        {
            data = minion_ucode_data[(manifestOffset/ dataSize) + i];
            LWSWITCH_MINION_WR32_BCAST_LS10(device, _CMINION, _FALCON_DMEMD(0), data);
        }
    }

_lwswitch_minion_copy_ucode_bc_exit:
    return LWL_SUCCESS;
}

/*
 * @brief : Print MINION ucode (first 8 DWORDS).
 *          This is used for diagnostic purposes only.
 *
 * @param[in] device Print ucode for a MINION on this device
 * @param[in] link   Print ucode for MINION associated with the link
 */
static void
_lwswitch_minion_print_ucode
(
    lwswitch_device *device,
    LwU32            instance
)
{
#if defined(DEBUG) || defined(DEVELOP) || defined(LW_MODS)
    LwU32 data, i;
    LwU32 buf[8];

    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_IMEMC, _AINCR, _TRUE, data);
    LWSWITCH_MINION_WR32_LS10(device, instance, _CMINION, _FALCON_IMEMC(0), data);

    LWSWITCH_PRINT(device, SETUP, "MINION IMEMD = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LS10(device, instance, _CMINION, _FALCON_IMEMD(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);

    LWSWITCH_PRINT(device, SETUP, "MINION IMEMC = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LS10(device, instance, _CMINION, _FALCON_IMEMC(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);

    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_DMEMC, _AINCR, _TRUE, data);
    LWSWITCH_MINION_WR32_LS10(device, instance, _CMINION, _FALCON_DMEMC(0), data);

    LWSWITCH_PRINT(device, SETUP, "MINION DMEMD = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LS10(device, instance, _CMINION, _FALCON_DMEMD(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);

    LWSWITCH_PRINT(device, SETUP, "MINION DMEMC = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LS10(device, instance, _CMINION, _FALCON_DMEMC(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
#endif  //defined(DEBUG) || defined(DEVELOP) || defined(LW_MODS)
}

LwlStatus
lwswitch_minion_get_dl_status_ls10
(
    lwswitch_device *device,
    LwU32            linkId,
    LwU32            statusIdx,
    LwU32            statusArgs,
    LwU32           *statusData
)
{
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwU32            regData, localLinkNumber;
    localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION_LS10;

    if (LWSWITCH_IS_LINK_ENG_VALID_LS10(device, MINION, linkId) &&
        !lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION %d is not initialized for link %08x.\n",
            __FUNCTION__, LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
            linkId);
        return -LWL_ERR_ILWALID_STATE;
    }

    // Query the DL status interface to get the data
    LWSWITCH_MINION_LINK_WR32_LS10(device, linkId, _MINION, _LWLINK_DL_STAT(localLinkNumber),
            DRF_NUM(_MINION, _LWLINK_DL_STAT, _ARGS, statusArgs) |
            DRF_NUM(_MINION, _LWLINK_DL_STAT, _STATUSIDX, statusIdx));

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(20 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    }

    // Poll for READY bit to be set
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        regData = LWSWITCH_MINION_LINK_RD32_LS10(device, linkId, _MINION, _LWLINK_DL_STAT(localLinkNumber));
        if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_STAT, _READY, 1, regData))
        {
            *statusData = LWSWITCH_MINION_LINK_RD32_LS10(device, linkId, _MINION, _LWLINK_DL_STATDATA(localLinkNumber));
            return LWL_SUCCESS;
        }
        if (IS_FMODEL(device) || IS_RTLSIM(device))
        {
            lwswitch_os_sleep(1);
        }
    }
    while (keepPolling);

    LWSWITCH_PRINT(device, ERROR,
        "%s: Timeout waiting for DL_STAT request to complete"
        " LW_MINION_LWLINK_DL_STAT(%d) = 0x%08x\n",
        __FUNCTION__, linkId, regData);
    return -LWL_ERR_ILWALID_STATE;
}

/*
 * @Brief : Send MINION DL CMD for a particular link
 *
 * @param[in] device     Send command to MINION on this device
 * @param[in] linkNumber DLCMD will be sent on this link number
 *
 * @return           Returns true if the DLCMD passed
 */
LwlStatus
lwswitch_minion_send_command_ls10
(
    lwswitch_device *device,
    LwU32            linkNumber,
    LwU32            command,
    LwU32            scratch0
)
{
    LwU32            data = 0, localLinkNumber, statData = 0;
    LwU32            ingressEccRegVal = 0, egressEccRegVal = 0;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    localLinkNumber = linkNumber % LWSWITCH_LINKS_PER_MINION_LS10;

    if (LWSWITCH_IS_LINK_ENG_VALID_LS10(device, MINION, linkNumber) &&
        !lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, linkNumber, MINION)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION %d is not initialized for link %08x.\n",
            __FUNCTION__, LWSWITCH_GET_LINK_ENG_INST(device, linkNumber, MINION),
            linkNumber);
        return LWL_SUCCESS;
    }

    data = LWSWITCH_MINION_LINK_RD32_LS10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber));
    if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT, 1, data))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION %d is in fault state. LW_MINION_LWLINK_DL_CMD(%d) = %08x\n",
            __FUNCTION__, LWSWITCH_GET_LINK_ENG_INST(device, linkNumber, MINION),
            linkNumber, data);
        return -LWL_ERR_ILWALID_STATE;
    }

    // Write to minion scratch if needed by command
    switch (command)
    {
        case LW_MINION_LWLINK_DL_CMD_COMMAND_CONFIGEOM:
            data = 0;
            data = FLD_SET_DRF_NUM(_MINION, _MISC_0, _SCRATCH_SWRW_0, scratch0, data);
            LWSWITCH_MINION_WR32_LS10(device,
                LWSWITCH_GET_LINK_ENG_INST(device, linkNumber, MINION), _MINION, _MISC_0, data);
            break;
        case LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHASE1:
            //
            // WAR bug 2708497
            // Before INITPHASE1, we must clear these values, then set back to
            // _PROD after the call
            // LW_INGRESS_ERR_ECC_CTRL_NCISOC_PARITY_ENABLE
            // LW_EGRESS_ERR_ECC_CTRL_NCISOC_PARITY_ENABLE
            //

            ingressEccRegVal = LWSWITCH_NPORT_RD32_LS10(device, linkNumber, _INGRESS, _ERR_ECC_CTRL);
            LWSWITCH_NPORT_WR32_LS10(device, linkNumber, _INGRESS, _ERR_ECC_CTRL,
                FLD_SET_DRF(_INGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, _DISABLE, ingressEccRegVal));

            egressEccRegVal = LWSWITCH_NPORT_RD32_LS10(device, linkNumber, _EGRESS, _ERR_ECC_CTRL);
            LWSWITCH_NPORT_WR32_LS10(device, linkNumber, _EGRESS, _ERR_ECC_CTRL,
                FLD_SET_DRF(_EGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, _DISABLE, egressEccRegVal));
            break;
        default:
            break;
    }

    data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _COMMAND, command, data);
    data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT,   1,   data);
    LWSWITCH_MINION_LINK_WR32_LS10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber), data);

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    }

    //
    // We will exit this if the command is successful OR
    // if timeout waiting for the READY bit to be set OR
    // if it generates a MINION FAULT
    //
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        data = LWSWITCH_MINION_LINK_RD32_LS10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber));
        if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_CMD, _READY, 1, data))
        {
            // The command has completed, success?
            if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT, 1, data))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: LWLink MINION command faulted!"
                    " LW_MINION_LWLINK_DL_CMD(%d) = 0x%08x\n",
                    __FUNCTION__, linkNumber, data);

                // Pull fault code and subcode
                if (lwswitch_minion_get_dl_status(device, linkNumber,
                            LW_LWLSTAT_MN00, 0, &statData) == LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Minion DLCMD Fault code = 0x%x, Sub-code = 0x%x\n",
                        __FUNCTION__,
                        DRF_VAL(_LWLSTAT, _MN00, _LINK_INTR_CODE, statData),
                        DRF_VAL(_LWLSTAT, _MN00, _LINK_INTR_SUBCODE, statData));
                }
                else
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Failed to get code and subcode from DLSTAT, link %d\n",
                        __FUNCTION__, linkNumber);
                }

                // Clear the fault and return
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Clearing LWLink MINION fault for link %d\n",
                    __FUNCTION__, linkNumber);

                data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT, 1, 0x0);
                LWSWITCH_MINION_LINK_WR32_LS10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber), data);
                return -LWL_ERR_ILWALID_STATE;
            }
            else
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s: LWLink MINION command %x was sent successfully for link %d\n",
                    __FUNCTION__, command, linkNumber);
                break;
            }
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (!FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_CMD, _READY, 1, data))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout waiting for LWLink MINION command to complete!"
            " LW_MINION_LWLINK_DL_CMD(%d) = 0x%08x\n",
            __FUNCTION__, linkNumber, data);
        return -LWL_ERR_ILWALID_STATE;
    }

    if (command == LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHASE1)
    {
        //
        // WAR bug 2708497
        // Before INITPHASE1, we must clear these values, then set back to
        // _PROD after the call
        // LW_INGRESS_ERR_ECC_CTRL_NCISOC_PARITY_ENABLE
        // LW_EGRESS_ERR_ECC_CTRL_NCISOC_PARITY_ENABLE
        //
        LWSWITCH_NPORT_WR32_LS10(device, linkNumber, _INGRESS, _ERR_ECC_CTRL, ingressEccRegVal);
        LWSWITCH_NPORT_WR32_LS10(device, linkNumber, _EGRESS,  _ERR_ECC_CTRL, egressEccRegVal);
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Test MINION by sending SWINTR DLCMD
 *
 * @param[in] device Send command to MINION on this device
 * @param[in] link   DLCMD will be sent on this link
 *
 * @return           Returns true if the DLCMD passed
 */
static LwBool
_lwswitch_minion_test_dlcmd
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    LwU32 interrupts, localLinkNumber;
    localLinkNumber = linkNumber % LWSWITCH_LINKS_PER_MINION_LS10;

    if (lwswitch_minion_send_command(device, linkNumber,
        LW_MINION_LWLINK_DL_CMD_COMMAND_SWINTR, 0) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: SWINTR DL CMD failed for link %d.\n",
            __FUNCTION__, linkNumber);
        return LW_FALSE;
    }

    interrupts = LWSWITCH_MINION_LINK_RD32_LS10(device, linkNumber,
                    _MINION, _LWLINK_LINK_INTR(localLinkNumber));

    if (DRF_VAL(_MINION, _LWLINK_LINK_INTR, _CODE, interrupts) ==
        LW_MINION_LWLINK_LINK_INTR_CODE_SWREQ)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Received NON-FATAL INTR_CODE = SWREQ, SUBCODE = 0x%x."
            " SWINTR DLCMD was exelwted successfully.\n",
            __FUNCTION__,
            DRF_VAL(_MINION, _LWLINK_LINK_INTR, _SUBCODE, interrupts));

        // clear the interrupt
        interrupts = DRF_NUM(_MINION, _LWLINK_LINK_INTR, _STATE, 1);
        LWSWITCH_MINION_LINK_WR32_LS10(device, linkNumber, _MINION,
                                       _LWLINK_LINK_INTR(localLinkNumber), interrupts);

        return LW_TRUE;
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: No SWINTR interrupt received. DL CMD failed for link %d.\n",
            __FUNCTION__, linkNumber);
        return LW_FALSE;
    }

    return LW_TRUE;
}

static void
_lwswitch_print_minion_info
(
    lwswitch_device *device,
    LwU32 id
)
{
#if defined(DEVELOP) || defined(DEBUG)
    LwU32   falcon_os;
    LwU32   falcon_mailbox;
    LwU32   falcon_sctl;
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    falcon_os = LWSWITCH_MINION_RD32_LS10(device, id, _CMINION, _FALCON_OS);
    falcon_mailbox = LWSWITCH_MINION_RD32_LS10(device, id, _CMINION, _FALCON_MAILBOX1);
    falcon_sctl = LWSWITCH_MINION_RD32_LS10(device, id, _CMINION, _FALCON_SCTL);

    if (chip_device->minionEngArch == LW_UPROC_ENGINE_ARCH_FALCON_RISCV)
    {
        // Dump the ucode ID string epilog
        LWSWITCH_PRINT(device, SETUP,
            "MINION RISCV ucode version info: Ucode v%d.%d  Phy v%d\n",
            (falcon_os >> 16) & 0xFFFF,
            falcon_os & 0xFFFF,
            falcon_mailbox);        
    }
    else
    {
        // Dump the ucode ID string epilog
        LWSWITCH_PRINT(device, SETUP,
            "MINION Falcon ucode version info: Ucode v%d.%d  Phy v%d\n",
            (falcon_os >> 16) & 0xFFFF,
            falcon_os & 0xFFFF,
            falcon_mailbox);
    }

    // Display security level info at info level, very useful for logs.
    LWSWITCH_PRINT(device, SETUP,
       "%s: LW_CMINION_FALCON_SCTL : 0x%08X\n",
       __FUNCTION__, falcon_sctl);
#endif
}

/*
 * @Brief : Bootstrap MINION associated with the link
 *
 * @param[in] device Bootstrap MINION on this device
 * @param[in] link   Bootstrap MINION associated with the link
 */
static LwlStatus
_lwswitch_minion_bootstrap
(
    lwswitch_device *device
)
{
    LwU32            data, i;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwlStatus        status = LWL_SUCCESS;
    ls10_device      *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32            firstValidLink;

    for (i = 0; i < NUM_MINION_ENGINE_LS10 ; i++)
    {
        if (!LWSWITCH_ENG_VALID_LS10(device, MINION, i))
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: MINION[%d] is not valid.  Skipping\n",
                __FUNCTION__, i);
            continue;
        }

        if (lwswitch_is_minion_initialized(device, i))
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: MINION[%d] is already bootstrapped.\n",
                __FUNCTION__, i);
            continue;
        }

        firstValidLink = _lwswitch_minion_get_first_valid_link(device, i);
        if (firstValidLink == LWSWITCH_LINKS_PER_MINION_LS10)
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: MINION instane %d, has no valid links associated with it. Skipping bootstrapping the instance and marking as un-initialized.\n",
                __FUNCTION__, i);
            lwswitch_set_minion_initialized(device, i, LW_FALSE);
            continue;
        }

        // Verify if the ucode was written properly
        _lwswitch_minion_print_ucode(device, i);

        if (chip_device->minionEngArch == LW_UPROC_ENGINE_ARCH_FALCON_RISCV)
        {
            LwU64 phyAddr;
            LwU32 val;
            if (chip_device->riscvManifestBoot)
            {
                // For manifest boot, use IROM addr
                lwswitch_minion_riscv_get_physical_address_ls10(device, i, LW_RISCV_MEM_IROM, 0, &phyAddr);
            }
            else
            {
                // For non-manifest boot, use IMEM addr
                lwswitch_minion_riscv_get_physical_address_ls10(device, i, LW_RISCV_MEM_IMEM, 0, &phyAddr);
            }

            // Set boot vector
            LWSWITCH_MINION_WR32_LS10(device, i, _CMINION_RISCV, _BOOT_VECTOR_HI, LwU64_HI32(phyAddr));
            LWSWITCH_MINION_WR32_LS10(device, i, _CMINION_RISCV, _BOOT_VECTOR_LO, LwU64_LO32(phyAddr));

            // Select RISCV Core
            LWSWITCH_MINION_WR32_LS10(device, i, _CMINION_RISCV, _BCR_CTRL,
                        DRF_DEF(_CMINION_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV));

            // Start CPU
            LWSWITCH_MINION_WR32_LS10(device, i, _CMINION_RISCV, _CPUCTL,
                        DRF_DEF(_CMINION_RISCV, _CPUCTL, _STARTCPU, _TRUE));

            if (chip_device->riscvManifestBoot)
            {
                do
                {
                    val = LWSWITCH_MINION_RD32_LS10(device, i, _CMINION_RISCV, _BR_RETCODE);
                    if (FLD_TEST_DRF(_CMINION, _RISCV_BR_RETCODE, _RESULT, _PASS, val))
                    {
                        // BROM finished with result: PASS
                        break;
                    }
                    else if (FLD_TEST_DRF(_CMINION, _RISCV_BR_RETCODE, _RESULT, _FAIL, val))
                    {
                        LWSWITCH_PRINT(device, SETUP, "%s: BROM finished with result: FAIL\n", __FUNCTION__);
                        break;
                    }
                    lwswitch_os_sleep(1);
                }while(1);
            }

            // Ensure the CPU has started
            if (lwswitch_minion_is_riscv_active_ls10(device, i))
            {
                LWSWITCH_PRINT(device, SETUP, "%s: MINION %d started RISCV Core successfully\n", __FUNCTION__, i);
            }
            else
            {
                return LW_ERR_NOT_READY;
            }
        }
        else
        {
            // Write boot vector to 0x0
            data = LWSWITCH_MINION_RD32_LS10(device, i, _CMINION, _FALCON_BOOTVEC);
            data = FLD_SET_DRF_NUM(_CMINION, _FALCON_BOOTVEC, _VEC, 0x0, data);
            LWSWITCH_MINION_WR32_LS10(device, i, _CMINION, _FALCON_BOOTVEC, data);

            //
            // Start the Falcon
            // If a falcon is managed (and thus supports secure mode), we need to
            // write LW_PFALCON_FALCON_CPUCTL_ALIAS_STARTCPU = _TRUE.
            // Below write is a nop in secure mode.
            //
            data = LWSWITCH_MINION_RD32_LS10(device, i, _CMINION, _FALCON_CPUCTL);
            data = FLD_SET_DRF(_CMINION, _FALCON_CPUCTL, _STARTCPU, _TRUE, data);
            LWSWITCH_MINION_WR32_LS10(device, i, _CMINION, _FALCON_CPUCTL, data);
        }

        if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
        {
            lwswitch_timeout_create(30*LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
        }
        else
        {
            lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
        }

        //
        // We will exit this if we recieve bootstrap signal OR
        // if we timeout waiting for bootstrap signal       OR
        // if bootstrap fails
        //
        do
        {
            keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

            data = LWSWITCH_MINION_RD32_LS10(device, i, _MINION, _MINION_STATUS);

            // The INIT sequence has completed, success?
            if (FLD_TEST_DRF(_MINION, _MINION_STATUS, _STATUS, _BOOT, data))
            {
                // MINION Init succeeded.
                LWSWITCH_PRINT(device, SETUP,
                    "%s: LWLink MINION %d bootstrap complete signal received.\n",
                    __FUNCTION__, i);

                _lwswitch_print_minion_info(device, i);
                break;
            }

            //
            // Check if any falcon interrupts are hit & pending.
            // TODO: Check return status of the call below
            //
            lwswitch_minion_service_falcon_interrupts_ls10(device, i);

            lwswitch_os_sleep(1);
        }
        while (keepPolling);

        if (!FLD_TEST_DRF(_MINION, _MINION_STATUS, _STATUS, _BOOT, data))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for LWLink MINION %d to complete bootstrap!"
                "LW_CMINION_MINION_STATUS = 0x%08x\n",
                __FUNCTION__, i, data);
            // Bug 2974064: Review this timeout handling (fall through)
        }
        lwswitch_set_minion_initialized(device, i, LW_TRUE);

        // Run a test DLCMD to see if MINION is accepting commands.
        if (!_lwswitch_minion_test_dlcmd(device, (i * LWSWITCH_LINKS_PER_MINION_LS10)))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unable to bootstrap MINION %d.\n",
                __FUNCTION__, i);
            lwswitch_set_minion_initialized(device, i, LW_FALSE);
            return -LWL_ERR_ILWALID_STATE;
        }
        else
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s: MINION %d successfully bootstrapped and accepting DLCMDs.\n",
                __FUNCTION__, i);
            lwswitch_set_minion_initialized(device, i, LW_TRUE);
        }
    }

    return status;
}

/*
 * @Brief : Load minion ucode from regkeys
 *          Overrides minion image from the regkeys
 *
 * @param   device  The lwswitch device
 */
static LwlStatus
_lwswitch_load_minion_ucode_image_from_regkeys
(
    lwswitch_device *device
)
{
    LwlStatus status = LWL_SUCCESS;

    LwU32 *data = NULL;
    LwU32 *header = NULL;
    LwU32 data_size;
    LwU32 header_size;

    if (!LW_SWITCH_REGKEY_PRIVATE_ALLOWED)
    {
        // Regkey override of ucode image only allowed on internal use debug drivers.
        return -LWL_ERR_GENERIC;
    }

    status = lwswitch_os_read_registry_dword(device->os_handle,
                    LW_SWITCH_REGKEY_MINION_SET_UCODE_HDR_SIZE, &header_size);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    status = lwswitch_os_read_registry_dword(device->os_handle,
                    LW_SWITCH_REGKEY_MINION_SET_UCODE_DATA_SIZE, &data_size);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if (header_size == 0 || data_size == 0)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Failed to query ucode size via regkey.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    header = lwswitch_os_malloc(header_size);
    if (header == NULL)
    {
        status = -LWL_NO_MEM;
        goto done;
    }

    data = lwswitch_os_malloc(data_size);
    if (data == NULL)
    {
        status = -LWL_NO_MEM;
        goto done;
    }

    status = lwswitch_os_read_registery_binary(device->os_handle,
                    LW_SWITCH_REGKEY_MINION_SET_UCODE_HDR, (LwU8*)header, header_size);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Failed to query ucode header.\n",
            __FUNCTION__);
        goto done;
    }

    status = lwswitch_os_read_registery_binary(device->os_handle,
                    LW_SWITCH_REGKEY_MINION_SET_UCODE_DATA, (LwU8*)data, data_size);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Failed to query ucode data.\n",
            __FUNCTION__);
        goto done;
    }

    // Copy the ucode to IMEM and DMEM by using backdoor PMB access
    status = _lwswitch_minion_copy_ucode_bc(device, data, header);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to copy MINION ucode in broadcast mode!\n",
            __FUNCTION__);
        goto done;
    }
    else
    {
        LWSWITCH_PRINT(device, SETUP,
            "Successfully loaded MINION microcode override.\n");
    }

done:
    if (header != NULL)
    {
        lwswitch_os_free(header);
    }

    if (data != NULL)
    {
        lwswitch_os_free(data);
    }

    return status;
}

/*
 * @Brief : Load minion ucode image
 *
 * @param   device            The lwswitch device
 */
static LwlStatus
_lwswitch_load_minion_ucode_image
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwU32 data;
    LwBool bDebugMode = LW_FALSE;

    // load ucode image via regkey
    status = _lwswitch_load_minion_ucode_image_from_regkeys(device);
    if (status == LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Successfully loaded ucode via regkey\n",
            __FUNCTION__);
        return status;
    }

    //
    // Determine if _dbg or _prod ucode needs to be loaded
    // Read from MINION 0 - we don't support MINIONs being in different debug modes
    //
    data = LWSWITCH_MINION_RD32_LS10(device, 0, _MINION, _MINION_DEVICES_3);
    bDebugMode = (data & BIT(1)) ? LW_TRUE : LW_FALSE;

    //
    // If ucode load fails via regkey fallback to the default ucode.
    // Copy the ucode to IMEM and DMEM by using backdoor PMB access
    //
    if (bDebugMode)
    {
        ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

        if (chip_device->minionEngArch == LW_UPROC_ENGINE_ARCH_FALCON_RISCV)
        {
            if (chip_device->riscvManifestBoot)
            {
                status = _lwswitch_minion_copy_ucode_bc(device, minion_ucode_data_ls10_dbg_riscv, minion_ucode_header_ls10_dbg_riscv);
            }
            else
            {
                status = _lwswitch_minion_copy_ucode_bc(device, minion_ucode_data_ls10_no_manifest_dbg_riscv, minion_ucode_header_ls10_no_manifest_dbg_riscv);
            }

            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Unable to copy dbg MINION ucode in broadcast mode!\n",
                    __FUNCTION__);
                return status;
            }
        }
        else
        {
            status = _lwswitch_minion_copy_ucode_bc(device, minion_ucode_data_ls10_dbg,  minion_ucode_header_ls10_dbg);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Unable to copy dbg MINION ucode in broadcast mode!\n",
                    __FUNCTION__);
                return status;
            }            
        }
    }
    else
    {
//        status = _lwswitch_minion_copy_ucode_bc(device, minion_ucode_data_ls10_prod, minion_ucode_header_ls10_prod);
        status = _lwswitch_minion_copy_ucode_bc(device, minion_ucode_data_ls10_dbg, minion_ucode_header_ls10_dbg);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unable to copy prod MINION ucode in broadcast mode!\n",
                __FUNCTION__);
            return status;
        }
    }

    return status;
}

/*
 * @Brief : Bootstrap all MINIONs on the specified device
 *
 * @param[in] device Bootstrap MINIONs on this device
 */
LwlStatus
lwswitch_init_minion_ls10
(
    lwswitch_device *device
)
{
    LwlStatus status = LWL_SUCCESS;

    if (_lwswitch_check_running_minions(device))
    {
        return LWL_SUCCESS;
    }

    status = _lwswitch_minion_pre_init(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION pre init failed\n",
            __FUNCTION__);
        return status;
    }

    // Load MINION
    status = _lwswitch_load_minion_ucode_image(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to load MINION ucode image!\n",
            __FUNCTION__);
        return status;
    }

    status = _lwswitch_minion_bootstrap(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to bootstrap MINION!\n",
            __FUNCTION__);
        return status;
    }

    return status;
}

/*
* @brief: This function will restore seed data back into MINION for training
* @params[in] device        reference to current lwswitch device
* @params[in] linkId        link we want to save seed data for
* @params[in] seedData      referene to a buffer to read
*/
LwlStatus
lwswitch_minion_restore_seed_data_ls10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *seedData
)
{
    LwlStatus status = LWL_SUCCESS;
    LwU32 localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION_LS10;
    LwU32 size = seedData[0];
    LwU32 i;

    if (device->regkeys.minion_cache_seeds == LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_DISABLE)
    {
        return status;
    }

    if (size == 0)
    {
        LWSWITCH_PRINT(device, INFO, "%s : Bad seed data, got numEntries of 0. Skipping seed restore (%s):(%d).\n",
            __FUNCTION__, device->name, linkId);
        return LWL_SUCCESS;
    }

    // Send minion the size of the buffer, where a pre-poulated size of 0 denotes skipping restoring on MINION's end
    LWSWITCH_MINION_LINK_WR32_LS10(device, linkId, _MINION_LWLINK_DL_CMD,
        _DATA(localLinkNumber), size);

    status = lwswitch_minion_send_command(device, linkId,
        LW_MINION_LWLINK_DL_CMD_COMMAND_WRITE_PHY_TRAINING_PARAMS,0);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "%s : Failed to send size of training seed buffer to MINION for (%s):(%d).\n",
            __FUNCTION__, device->name, linkId);
        return status;
    }

    for (i = 1; i < size+1; i++)
    {
        LWSWITCH_PRINT(device, INFO, "%s : Link %d seed data entry %d = 0x%x\n",
            __FUNCTION__, linkId, i, seedData[i]);

        LWSWITCH_MINION_LINK_WR32_LS10(device, linkId, _MINION_LWLINK_DL_CMD,
            _DATA(localLinkNumber), seedData[i]);

        status = lwswitch_minion_send_command(device, linkId,
            LW_MINION_LWLINK_DL_CMD_COMMAND_WRITE_PHY_TRAINING_PARAMS,0);

        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, INFO, "%s : Could not send part of data buffer to MINION for (%s):(%d).\n",
                __FUNCTION__, device->name, linkId);
            return status;
        }
    }
    return status;
}

/*
* @brief: This function will poll MINION to save known good seed values
* @params[in] device        reference to current lwswitch device
* @params[in] linkId        link we want to save seed data for
* @params[in] seedData      referene to a buffer to populate
*/
LwlStatus
lwswitch_minion_save_seed_data_ls10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *seedData
)
{
    LwlStatus status = LW_OK;
    LwU32 localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION_LS10;
    LwU32 size, statData, i;

    if (device->regkeys.minion_cache_seeds == LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_DISABLE)
    {
        return status;
    }

    status = lwswitch_minion_get_dl_status(device, linkId, LW_LWLSTAT_UC01, 0, &statData);

    if(status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO,"%s : Failed to poll DLSTAT register for (%s):(%d)\n",
                        __FUNCTION__, device->name, linkId);
    }
    else
    {
        statData = DRF_VAL(_LWLSTAT, _UC01, _TRAINING_BUFFER_STATUS, statData);
        // Not lwrrently in hwref, 0xF = good A0 training
        if (statData != 0XF)
        {
            LWSWITCH_PRINT(device, INFO,"%s : Previously loaded training seeds were rejected by MINION for (%s):(%d)\n",
                        __FUNCTION__, device->name, linkId);
        }
    }

    status = lwswitch_minion_send_command(device, linkId,
        LW_MINION_LWLINK_DL_CMD_COMMAND_READ_PHY_TRAINING_PARAMS,0);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO,"%s : Failed to poll MINION for size of training seed buffer for (%s):(%d)\n",
                        __FUNCTION__, device->name, linkId);
        size = 0;
        goto done;
    }

    // read in the size of the buffer being read in from MNINION
    size = LWSWITCH_MINION_LINK_RD32_LS10(device, linkId, _MINION_LWLINK_DL_CMD,
        _DATA(localLinkNumber));

    // if bad data zero out the size
    if (size > LWLINK_MAX_SEED_NUM)
    {
        LWSWITCH_PRINT(device, INFO,"%s : MINION returned bad seed buffer size for link (%s):(%d)\n",
                        __FUNCTION__, device->name, linkId);
        size = 0;
        status = LWL_ERR_GENERIC;
        goto done;
    }

    for (i = 1; i < size+1; i++)
    {
        status = lwswitch_minion_send_command(device, linkId,
            LW_MINION_LWLINK_DL_CMD_COMMAND_READ_PHY_TRAINING_PARAMS,0);

        // if minion DL command fails ilwalidate buffer by writing back a size of 0
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, INFO,"%s : Failed to poll MINION for part of training seed buffer for (%s):(%d)\n",
                        __FUNCTION__, device->name, linkId);
            size = 0;
            goto done;
        }

        seedData[i] = LWSWITCH_MINION_LINK_RD32_LS10(device, linkId, _MINION_LWLINK_DL_CMD,
                                                          _DATA(localLinkNumber));

        LWSWITCH_PRINT(device, INFO,"%s : Seed Data for link (%s):(%d) entry %d = %d\n",
                       __FUNCTION__, device->name, linkId, i, seedData[i]);
    }

done:

    //
    // first entry into the seed data buffer is always the size of the buffer
    // if we fail to get good data (i.e. size to large or fail to poll from minion) this entry will be 0
    //
    seedData[0] = size;
    return status;
}

LwlStatus
lwswitch_set_minion_initialized_ls10
(
    lwswitch_device *device,
    LwU32 idx_minion,
    LwBool initialized
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    if (!LWSWITCH_ENG_VALID_LS10(device, MINION, idx_minion))
    {
        return -LWL_BAD_ARGS;
    }

    chip_device->engMINION[idx_minion].initialized = initialized;
    return LWL_SUCCESS;
}

LwBool
lwswitch_is_minion_initialized_ls10
(
    lwswitch_device *device,
    LwU32 idx_minion
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);

    if (!LWSWITCH_ENG_VALID_LS10(device, MINION, idx_minion))
    {
        return LW_FALSE;
    }
    return (chip_device->engMINION[idx_minion].initialized != 0);
}

LwlStatus
lwswitch_minion_set_sim_mode_ls10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    LwlStatus status = LWL_SUCCESS;
    LwU32 dlcmd;
    LwU32 linkNumber = link->linkNumber;
    LwU32 localLinkNumber = linkNumber % LWSWITCH_LINKS_PER_MINION_LS10;

    switch (device->regkeys.set_simmode)
    {
        case LW_SWITCH_REGKEY_MINION_SET_SIMMODE_FAST:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_FAST;
            break;
        case LW_SWITCH_REGKEY_MINION_SET_SIMMODE_MEDIUM:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_MEDIUM;
            break;
        case LW_SWITCH_REGKEY_MINION_SET_SIMMODE_SLOW:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_SLOW;
            break;
        default:
            return LWL_SUCCESS;
    }

    status = lwswitch_minion_send_command(device, linkNumber, dlcmd, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: DLCMD 0x%x failed on link: %d\n",
                __FUNCTION__, dlcmd, linkNumber);
        return status;
    }

    // Setting RXCAL_EN_ALARM timer value
    LWSWITCH_MINION_LINK_WR32_LS10(device, linkNumber, _MINION,
            _LWLINK_DL_CMD_DATA(localLinkNumber),
            LW_MINION_DL_CMD_DATA_RXCAL_EN_ALARM);

    status = lwswitch_minion_send_command(device, linkNumber,
        LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_RXCAL_EN_ALARM, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: DLCMD DBG_SETSIMMODE_RXCAL_EN_ALARM failed on link: %d\n",
                __FUNCTION__, linkNumber);
        return status;
    }

    // Setting INIT_CAL_DONE timer value
    LWSWITCH_MINION_LINK_WR32_LS10(device, linkNumber, _MINION,
            _LWLINK_DL_CMD_DATA(localLinkNumber),
            LW_MINION_DL_CMD_DATA_INIT_CAL_DONE);

    status = lwswitch_minion_send_command(device, linkNumber,
        LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_INIT_CAL_DONE, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: DLCMD DBG_SETSIMMODE_INIT_CAL_DONE failed on link: %d\n",
                __FUNCTION__, linkNumber);
        return status;
    }

    return status;
}

LwlStatus
lwswitch_minion_set_smf_settings_ls10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    LwlStatus status = LWL_SUCCESS;
    LwU32 dlcmd;
    LwU32 linkNumber = link->linkNumber;

    switch (device->regkeys.set_smf_settings)
    {
        case LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_SLOW:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_SMF_VALUES_SLOW;
            break;
        case LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_MEDIUM:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_SMF_VALUES_MEDIUM;
            break;
        case LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_FAST:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_SMF_VALUES_FAST;
            break;
        case LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_MEDIUM_SERIAL:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_SMF_VALUES_MEDIUM_SERIAL;
            break;
        default:
            return LWL_SUCCESS;
    }

    status = lwswitch_minion_send_command(device, linkNumber, dlcmd, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: DLCMD 0x%x failed on link: %d\n",
                __FUNCTION__, dlcmd, linkNumber);
        return status;
    }

    return status;
}

LwlStatus
lwswitch_minion_select_uphy_tables_ls10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    LwlStatus status = LWL_SUCCESS;
    LwU32 dlcmd;
    LwU32 linkNumber = link->linkNumber;

    switch (device->regkeys.select_uphy_tables)
    {
        case LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES_SHORT:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_UPHY_TABLES_SHORT;
            break;
        case LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES_FAST:
            dlcmd = LW_MINION_LWLINK_DL_CMD_COMMAND_DBG_SETSIMMODE_UPHY_TABLES_FAST;
            break;
        default:
            return LWL_SUCCESS;
    }

    status = lwswitch_minion_send_command(device, linkNumber, dlcmd, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: DLCMD 0x%x failed on link: %d\n",
                __FUNCTION__, dlcmd, linkNumber);
        return status;
    }

    return status;
}


LwlStatus
lwswitch_minion_get_rxdet_status_ls10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwU32            statData;
    LwlStatus        status;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(30*LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(20 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    }

    // Poll for READY bit to be set
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        // Check RXDET status on MINION DL STAT interface
        status = lwswitch_minion_get_dl_status(device, linkId, LW_LWLSTAT_LNK2, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }

        if (FLD_TEST_DRF(_LWLSTAT, _LNK2, _RXDET_LINK_STATUS, _FOUND, statData))
        {
            LWSWITCH_PRINT(device, INFO,
                    "%s: RXDET LINK_STATUS = FOUND on link: %d\n",
                    __FUNCTION__, linkId);

            // Retrieve which lanes were found (should be all)
            device->link[linkId].lane_rxdet_status_mask =
                    DRF_VAL(_LWLSTAT, _LNK2, _RXDET_LANE_STATUS, statData);

            //
            // MINION doesn't have knowledge of lane reversal and therefore
            // reports logical lanes. We must reverse the bitmask here if applicable
            // since RM reports physical lanes.
            //
            if (lwswitch_link_lane_reversed_lr10(device, linkId))
            {
                LWSWITCH_REVERSE_BITMASK_32(LWSWITCH_NUM_LANES_LS10,
                        device->link[linkId].lane_rxdet_status_mask);
            }

            return LWL_SUCCESS;
        }

        if (FLD_TEST_DRF(_LWLSTAT, _LNK2, _RXDET_LINK_STATUS, _TIMEOUT, statData))
        {
            LWSWITCH_PRINT(device, ERROR,
                    "%s: RXDET LINK_STATUS = TIMEOUT on link: %d\n",
                    __FUNCTION__, linkId);

            // Retrieve which lanes were found
            device->link[linkId].lane_rxdet_status_mask =
                    DRF_VAL(_LWLSTAT, _LNK2, _RXDET_LANE_STATUS, statData);

            //
            // MINION doesn't have knowledge of lane reversal and therefore
            // reports logical lanes. We must reverse the bitmask here if applicable
            // since RM reports physical lanes.
            //
            if (lwswitch_link_lane_reversed_lr10(device, linkId))
            {
                LWSWITCH_REVERSE_BITMASK_32(LWSWITCH_NUM_LANES_LS10,
                        device->link[linkId].lane_rxdet_status_mask);
            }

            return -LWL_ERR_ILWALID_STATE;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    LWSWITCH_PRINT(device, ERROR,
        "%s: Timeout waiting for RXDET STATUS on link: %d\n",
        __FUNCTION__, linkId);

    return -LWL_ERR_ILWALID_STATE;
}

/*
 * @Brief : Get the RISCV physical address of the target
 *
 * @param[in] device      The Lwswitch device
 * @param[in] idx_minion  MINION instance to use
 * @param[in] target      RISCV target
 * @param[in] offset      Offset for which PA is needed
 * @param[out] pRiscvPa   RISCV physical address
 */
LwlStatus
lwswitch_minion_riscv_get_physical_address_ls10
(
    lwswitch_device *device,
    LwU32           idx_minion,
    LwU32           target,
    LwLength        offset,
    LwU64           *pRiscvPa
)
{
    LwU64 rangeStart;
    LwU64 rangeSize;

    if (pRiscvPa == NULL)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }
    switch(target)
    {
        case LW_RISCV_MEM_IROM:
            rangeStart = LW_RISCV_AMAP_IROM_START;
            rangeSize  = LW_RISCV_AMAP_IROM_SIZE;
            break;
        case LW_RISCV_MEM_IMEM:
            rangeStart = LW_RISCV_AMAP_IMEM_START;
            rangeSize  = LW_RISCV_AMAP_IMEM_SIZE;
            break;
        case LW_RISCV_MEM_DMEM:
            rangeStart = LW_RISCV_AMAP_DMEM_START;
            rangeSize  = LW_RISCV_AMAP_DMEM_SIZE;
            break;
        case LW_RISCV_MEM_EMEM:
            rangeStart = LW_RISCV_AMAP_EMEM_START;
            rangeSize  = LW_RISCV_AMAP_EMEM_SIZE;
            break;
        case LW_RISCV_MEM_PRIV:
            rangeStart = LW_RISCV_AMAP_PRIV_START;
            rangeSize  = LW_RISCV_AMAP_PRIV_SIZE;
            break;
        case LW_RISCV_MEM_FBGPA:
            rangeStart = LW_RISCV_AMAP_FBGPA_START;
            rangeSize  = LW_RISCV_AMAP_FBGPA_SIZE;
            break;
        case LW_RISCV_MEM_SYSGPA:
            rangeStart = LW_RISCV_AMAP_SYSGPA_START;
            rangeSize  = LW_RISCV_AMAP_SYSGPA_SIZE;
            break;
        case LW_RISCV_MEM_GVA:
            rangeStart = LW_RISCV_AMAP_GVA_START;
            rangeSize  = LW_RISCV_AMAP_GVA_SIZE;
            break;
        default:
            return LW_ERR_ILWALID_INDEX;
    }

    if (offset > rangeSize)
    {
        return LW_ERR_ILWALID_OFFSET;
    }

    *pRiscvPa = rangeStart + offset;
    return LWL_SUCCESS;
}

/*
 * @Brief : Check if the RISCV CPU has started
 *
 * @param[in] device      The Lwswitch device
 * @param[in] idx_minion  MINION instance to use
 */
LwBool
lwswitch_minion_is_riscv_active_ls10
(
    lwswitch_device *device,
    LwU32           idx_minion
)
{
    LwU32 val;

    val = LWSWITCH_MINION_RD32_LS10(device, idx_minion, _CMINION_RISCV, _CPUCTL);

    return FLD_TEST_DRF(_CMINION, _RISCV_CPUCTL, _ACTIVE_STAT, _ACTIVE, val);

}

LwlStatus
lwswitch_minion_clear_dl_error_counters_ls10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwlStatus status;

    status = lwswitch_minion_send_command(device, linkId,
                                               LW_MINION_LWLINK_DL_CMD_COMMAND_DLSTAT_CLR_DLERRCNT, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s : Failed to clear error count to MINION for link # %d\n",
            __FUNCTION__, linkId);
    }
    return status;
}

LwlStatus
lwswitch_minion_send_inband_data_ls10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwlStatus status = LWL_SUCCESS;
#if defined(INCLUDE_LWLINK_LIB)
    LwU32 numEntries, i;
    LwU32 localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION_LS10;

    LwU32 *sendBuffer = device->link[linkId].inBandData.sendBuffer;

    numEntries = sendBuffer[0] & 0x3F;

    if (numEntries == 0)
    {
        LWSWITCH_PRINT(device, ERROR, "Bad Inband data, got numEntries of 0. Skipping Inband Send\n");
        return -LWL_ERR_ILWALID_STATE;
    }

    // Write a 256 Byte buffer via DLCMD

    LWSWITCH_MINION_WR32_LS10(device, 
                LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
                _MINION, _LWLINK_DL_CMD_DATA(localLinkNumber), sendBuffer[0]);

    status = lwswitch_minion_send_command(device, linkId,
        LW_MINION_LWLINK_DL_CMD_COMMAND_READ_RX_BUFFER_START,0);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Link %d Inband Buffer transfer for entry %d failed\n", linkId, 0);
        return status;
    }

    for (i = 1; i < numEntries; i++)
    {
        LWSWITCH_MINION_WR32_LS10(device, 
                    LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
                    _MINION, _LWLINK_DL_CMD_DATA(localLinkNumber), sendBuffer[i]);

        status = lwswitch_minion_send_command(device, linkId,
                LW_MINION_LWLINK_DL_CMD_COMMAND_WRITE_TX_BUFFER_MIDDLE, 0);

        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "Link %d Inband Buffer transfer for entry %d failed\n", linkId, i);
            return status;
        }
    }

    LWSWITCH_MINION_WR32_LS10(device, 
                LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
                _MINION, _LWLINK_DL_CMD_DATA(localLinkNumber), sendBuffer[numEntries]);

    status = lwswitch_minion_send_command(device, linkId,
            LW_MINION_LWLINK_DL_CMD_COMMAND_WRITE_TX_BUFFER_END, 0);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Link %d Inband Buffer transfer for entry %d failed\n", linkId, numEntries);
        return status;
    }

    device->link[linkId].inBandData.bIsSenderMinion = LW_TRUE;

#endif
    return status;
}

LwlStatus
lwswitch_minion_receive_inband_data_ls10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwlStatus status = LWL_SUCCESS;
#if defined(INCLUDE_LWLINK_LIB)
    LwU32 numEntries = 0;
    LwU32 i;
    LwU32 localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION_LS10;

    LwU32 *receiveBuffer = device->link[linkId].inBandData.receiveBuffer;

    if (device->link[linkId].inBandData.bTransferFail)
    {
        goto exit;
    }

    status = lwswitch_minion_send_command(device, linkId,
            LW_MINION_LWLINK_DL_CMD_COMMAND_READ_RX_BUFFER_START,0);
    if (status != LW_OK)
        goto exit;

    receiveBuffer[0] =
                LWSWITCH_MINION_RD32_LS10(device, 
                    LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
                    _MINION, _LWLINK_DL_CMD_DATA(localLinkNumber));

    numEntries = receiveBuffer[0] & 0x3F;

    LWSWITCH_ASSERT(numEntries < LWSWITCH_MAX_INBAND_BUFFER_ENTRIES);

    for (i = 1; i < numEntries; i++)
    {
        status = lwswitch_minion_send_command(device, linkId,
                LW_MINION_LWLINK_DL_CMD_COMMAND_READ_RX_BUFFER_MIDDLE,0);
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR, "Link %d Inband Buffer receive for entry %d failed\n", linkId, i);
            goto exit;
        }

        receiveBuffer[i] =
                    LWSWITCH_MINION_RD32_LS10(device, 
                        LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
                        _MINION, _LWLINK_DL_CMD_DATA(localLinkNumber));
    }

    status = lwswitch_minion_send_command(device, linkId,
            LW_MINION_LWLINK_DL_CMD_COMMAND_READ_RX_BUFFER_END,0);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "Link %d Inband Buffer receive for entry %d failed\n", linkId, numEntries);
        goto exit;
    }

    receiveBuffer[numEntries] =
                    LWSWITCH_MINION_RD32_LS10(device, 
                        LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
                        _MINION, _LWLINK_DL_CMD_DATA(localLinkNumber));

exit:
    numEntries = 0;
    // memset other elements to zero
    lwswitch_os_memset(receiveBuffer, 0, sizeof(*receiveBuffer));
    device->link[linkId].inBandData.bIsSenderMinion = LW_FALSE;

#endif
    return status;
}

 #if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus
lwswitch_ctrl_config_eom_ls10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_CONFIG_EOM *p
)
{
    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, p->link))
    {
        return -LWL_BAD_ARGS;
    }

    return lwswitch_minion_send_command(device, p->link, LW_MINION_LWLINK_DL_CMD_COMMAND_CONFIGEOM, p->params);
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
