/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwlink_export.h"
#include "common_lwswitch.h"
#include "sv10/sv10.h"
#include "sv10/minion_sv10.h"
#include "sv10/minion_production_ucode_sv10.h"

#include "lwswitch/svnp01/dev_minion_ip.h"
#include "lwswitch/svnp01/dev_lwl_ip.h"

#define LWSWITCH_MINION_DEBUG 0

// The UCODE ID string dump constants (this will come from a shared header TBD)
#define RM_MINION_FALCON_UCODE_ID_MAX_LENGTH       (256)
#define RM_MINION_FALCON_UCODE_ID_DMEM_OFFSET      (0)
#define RM_MINION_FALCON_UCODE_ID_DMEM_PORT        (0)

// TODO : pull from manuals
#define LW_MINION_LWLINK_DL_CMD_COMMAND_READREGPADLANE  0x00000044
#define LW_MINION_UCODE_READREGPADLANE_ADDR             7:0
#define LW_MINION_UCODE_READREGPADLANE_LANE             18:16

/*
 * @Brief : Copy the minion ucode to IMEM and DMEM in broadcast mode
 *
 * @param[in] device  Copy ucode to all MINIONS associated with the device
 */
static LwlStatus
_lwswitch_minion_copy_ucode_bc
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    const PFALCON_UCODE_HDR_INFO_SV10 pUcodeHeader =
        (PFALCON_UCODE_HDR_INFO_SV10) &minion_ucode_header_sv10[0];
    const LwU32 *pHeader = &minion_ucode_header_sv10[0];

    LwU32 data, i, app, dataSize;
    LwU32 appCodeOffset, appCodeSize, appDataOffset, appDataSize;
    LwU16 tag;
    LwU32 idx_minion;

    // Find first valid MINION instance
    for (idx_minion = 0; idx_minion < NUM_SIOCTRL_ENGINE_SV10; idx_minion++)
    {
        if (chip_device->subengSIOCTRL[idx_minion].subengMINION[0].valid)
        {
            break;
        }
    }
    if (idx_minion >= NUM_SIOCTRL_ENGINE_SV10)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: No MINIONs instantiated.  Skipping MINION ucode load\n",
            __FUNCTION__);
        goto _lwswitch_minion_copy_ucode_bc_exit;
    }

    dataSize = sizeof(minion_ucode_data_sv10[0]);

    // Initialize address of IMEM to 0x0 and set auto-increment on write
    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_IMEMC, _AINCW, _TRUE, data);
    LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMC(0), data);

    //
    // Initialize IMEM tag to 0 explicitly even though power-on value is 0.
    // Writes to IMEM don't work if we don't do this
    //
    tag = 0;
    LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMT(0), tag);

    // Copy over IMEM part of the ucode and tag along the way
    for (i = 0; i < (pUcodeHeader->osCodeSize / dataSize) ; i++)
    {
        // Increment tag for after every block (256 bytes)
        if (i && ((i % (FALCON_IMEM_BLK_SIZE_BYTES_SV10 / dataSize)) == 0))
        {
            tag++;
            LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMT(0), (LwU32) tag);
        }

        // Copy IMEM DWORD by DWORD
        data = minion_ucode_data_sv10[(pUcodeHeader->osCodeOffset / dataSize) + i];
        LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMD(0), data);
    }

    // Initialize address of DMEM to 0x0 and set auto-increment on write
    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_DMEMC, _AINCW, _TRUE, data);
    LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_DMEMC(0), data);

    // Copy over DMEM part of the ucode
    for (i = 0; i < (pUcodeHeader->osDataSize / dataSize) ; i++)
    {
        // Copy DMEM DWORD by DWORD
        data = minion_ucode_data_sv10[(pUcodeHeader->osDataOffset / dataSize) + i];
        LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_DMEMD(0), data);
    }

    // Copy over any apps in the ucode with the appropriate tags
    if (pUcodeHeader->numApps)
    {
        for (app = 0; app < pUcodeHeader->numApps ; app++)
        {
            // Index into the app code info
            appCodeOffset = pHeader[FALCON_CODE_HDR_APP_CODE_START_SV10 + 2*app];
            appCodeSize   = pHeader[FALCON_CODE_HDR_APP_CODE_START_SV10 + 2*app + 1];

            // Index into the app data info using appCodeStart offset as a base
            appDataOffset = pHeader[FALCON_CODE_HDR_APP_CODE_START_SV10 +
                2*pUcodeHeader->numApps + 2*app];
            appDataSize   = pHeader[FALCON_CODE_HDR_APP_CODE_START_SV10 +
                2*pUcodeHeader->numApps + 2*app + 1];

            // Mark the following IMEM blocks as secure
            data = LWSWITCH_MINION_RD32_SV10(device, idx_minion, _CMINION, _FALCON_IMEMC(0));
            data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _SELWRE, 0x1, data);
            LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMC(0), data);

            // Copy to IMEM and tag along the way
            tag = (LwU16)(appCodeOffset >> 8);
            LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMT(0), tag);

            // Copy app code to IMEM picking up where last load left off
            for (i = 0; i < (appCodeSize / dataSize); i++)
            {
                if (i && ((i % (FALCON_IMEM_BLK_SIZE_BYTES_SV10 / dataSize)) == 0))
                {
                    tag++;
                    LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMT(0), tag);
                }

                data = minion_ucode_data_sv10[(appCodeOffset / dataSize) + i];
                LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_IMEMD(0), data);
            }

            // Copy app data to DMEM picking up where last load left off
            for (i = 0; i < (appDataSize / dataSize); i++)
            {
                data = minion_ucode_data_sv10[appDataOffset + i];
                LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_DMEMD(0), data);
            }
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
lwswitch_minion_print_ucode
(
    lwswitch_device *device,
    LwU32            instance
)
{
#if defined(DEBUG) || defined(DEVELOP) || defined(LW_MODS)
    LwU32 data, i;

    if (!LWSWITCH_MINION_DEBUG)
    {
        return;
    }

    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_IMEMC, _AINCR, _TRUE, data);
    LWSWITCH_MINION_WR32_SV10(device, instance, _CMINION, _FALCON_IMEMC(0), data);

    LWSWITCH_PRINT(device, SETUP, "MINION IMEMD = ");
    for (i = 0; i < 8 ; i++)
    {
        data = LWSWITCH_MINION_RD32_SV10(device, instance, _CMINION, _FALCON_IMEMD(0));
        LWSWITCH_PRINT(device, SETUP, " 0x%08x ", data);
    }
    LWSWITCH_PRINT(device, SETUP, "\n");

    LWSWITCH_PRINT(device, SETUP, "MINION IMEMC = ");
    for (i = 0; i < 8 ; i++)
    {
        data = LWSWITCH_MINION_RD32_SV10(device, instance, _CMINION, _FALCON_IMEMC(0));
        LWSWITCH_PRINT(device, SETUP, " 0x%08x ", data);
    }
    LWSWITCH_PRINT(device, SETUP, "\n");

    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_DMEMC, _AINCR, _TRUE, data);
    LWSWITCH_MINION_WR32_SV10(device, instance, _CMINION, _FALCON_DMEMC(0), data);

    LWSWITCH_PRINT(device, SETUP, "MINION DMEMD = ");
    for (i = 0; i < 8 ; i++)
    {
        data = LWSWITCH_MINION_RD32_SV10(device, instance, _CMINION, _FALCON_DMEMD(0));
        LWSWITCH_PRINT(device, SETUP, " 0x%08x ", data);
    }
    LWSWITCH_PRINT(device, SETUP, "\n");

    LWSWITCH_PRINT(device, SETUP, "MINION DMEMC = ");
    for (i = 0; i < 8 ; i++)
    {
        data = LWSWITCH_MINION_RD32_SV10(device, instance, _CMINION, _FALCON_DMEMC(0));
        LWSWITCH_PRINT(device, SETUP, " 0x%08x ", data);
    }
    LWSWITCH_PRINT(device, SETUP, "\n");
#endif  //defined(DEBUG) || defined(DEVELOP) || defined(LW_MODS)
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
lwswitch_minion_send_command_sv10
(
    lwswitch_device *device,
    LwU32            linkNumber,
    LwU32            command,
    LwU32            scratch0
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32            data = 0, link;
    LWSWITCH_TIMEOUT timeout;

    link = linkNumber % 2;

    if (!chip_device->link[linkNumber].engMINION->initialized)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION %d is not initialized for link %08x.\n",
            __FUNCTION__, chip_device->link[linkNumber].engMINION->instance,
            linkNumber);
        return LWL_SUCCESS;
    }

    data = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _LWLINK_DL_CMD(link));
    if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT, 1, data))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION %d is in fault state. LW_MINION_LWLINK_DL_CMD(%d) = %08x\n",
            __FUNCTION__, chip_device->link[linkNumber].engMINION->instance,
            linkNumber, data);
        return -LWL_ERR_ILWALID_STATE;
    }

    // Write to minion scratch if needed by command
    switch (command)
    {
        case LW_MINION_LWLINK_DL_CMD_COMMAND_CONFIGEOM:
        case LW_MINION_LWLINK_DL_CMD_COMMAND_READREGPADLANE:
            data = 0;
            data = FLD_SET_DRF_NUM(_MINION, _MISC_0, _SCRATCH_SWRW_0, scratch0, data);
            LWSWITCH_MINION_WR32_SV10(device, chip_device->link[linkNumber].engMINION->instance,
                _MINION, _MISC_0, data);
            break;
        default:
            break;
    }

    data = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _LWLINK_DL_CMD(link));
    data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _COMMAND, command, data);
    data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT,   1,   data);
    LWSWITCH_MINION_LINK_WR32_SV10(device, linkNumber, _MINION, _LWLINK_DL_CMD(link), data);

    lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);

    //
    // We will exit this if the command is successful OR
    // if timeout waiting for the READY bit to be set OR
    // if it generates a MINION FAULT
    //
    while (1)
    {
        data = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _LWLINK_DL_CMD(link));
        if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_CMD, _READY, 1, data))
        {
            // The command has completed, success?
            if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT, 1, data))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: LWLink MINION command faulted!"
                    " LW_MINION_LWLINK_DL_CMD(%d) = 0x%08x\n",
                    __FUNCTION__, linkNumber, data);

                // Clear the fault and return
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Clearing LWLink MINION fault for link %d\n",
                    __FUNCTION__, linkNumber);

                data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT, 1, data);
                LWSWITCH_MINION_LINK_WR32_SV10(device, linkNumber, _MINION, _LWLINK_DL_CMD(link), data);
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

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for LWLink MINION command to complete!"
                " LW_MINION_LWLINK_DL_CMD(%d) = 0x%08x\n",
                __FUNCTION__, linkNumber, data);
            return -LWL_ERR_ILWALID_STATE;
        }
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
    LwU32 interrupts, link;
    link = linkNumber % 2;

    if (lwswitch_minion_send_command_sv10(device, linkNumber,
        LW_MINION_LWLINK_DL_CMD_COMMAND_SWINTR, 0) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: SWINTR DL CMD failed for link %d.\n",
            __FUNCTION__, linkNumber);
        return LW_FALSE;
    }

    interrupts = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber,
                    _MINION, _LWLINK_LINK_INTR(link));

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
        LWSWITCH_MINION_LINK_WR32_SV10(device, linkNumber, _MINION,
                                       _LWLINK_LINK_INTR(link), interrupts);

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

    falcon_os = LWSWITCH_MINION_RD32_SV10(device, id, _CMINION, _FALCON_OS);
    falcon_mailbox = LWSWITCH_MINION_RD32_SV10(device, id, _CMINION, _FALCON_MAILBOX1);
    falcon_sctl = LWSWITCH_MINION_RD32_SV10(device, id, _CMINION, _FALCON_SCTL);

    // Dump the ucode ID string epilog
    LWSWITCH_PRINT(device, SETUP,
        "MINION Falcon ucode version info: Ucode v%d.%d  Phy v%d\n",
        (falcon_os >> 16) & 0xFFFF,
        falcon_os & 0xFFFF,
        falcon_mailbox);

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
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32            data, i;
    LWSWITCH_TIMEOUT timeout;

    for (i = 0; i < (chip_device->numSIOCTRL * NUM_MINION_INSTANCES_SV10) ; i++)
    {
        if (!chip_device->engSIOCTRL[i].valid ||
            !chip_device->subengSIOCTRL[i].subengMINION[0].valid)
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

        // Verify if the ucode was written properly
        lwswitch_minion_print_ucode(device, i);

        // Write boot vector to 0x0
        data = LWSWITCH_MINION_RD32_SV10(device, i, _CMINION, _FALCON_BOOTVEC);
        data = FLD_SET_DRF_NUM(_CMINION, _FALCON_BOOTVEC, _VEC, 0x0, data);
        LWSWITCH_MINION_WR32_SV10(device, i, _CMINION, _FALCON_BOOTVEC, data);

        //
        // Start the Falcon
        // If a falcon is managed (and thus supports secure mode), we need to
        // write LW_PFALCON_FALCON_CPUCTL_ALIAS_STARTCPU = _TRUE.
        // Below write is a nop in secure mode.
        //
        data = LWSWITCH_MINION_RD32_SV10(device, i, _CMINION, _FALCON_CPUCTL);
        data = FLD_SET_DRF(_CMINION, _FALCON_CPUCTL, _STARTCPU, _TRUE, data);
        LWSWITCH_MINION_WR32_SV10(device, i, _CMINION, _FALCON_CPUCTL, data);

        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);

        //
        // We will exit this if we receive bootstrap signal OR
        // if we timeout waiting for bootstrap signal       OR
        // if bootstrap fails
        //
        while (1)
        {
            data = LWSWITCH_MINION_RD32_SV10(device, i, _MINION, _MINION_STATUS);

            // The INIT sequence has completed, success?
            if (FLD_TEST_DRF(_MINION, _MINION_STATUS, _STATUS, _BOOT, data))
            {
                // MINION Init succeeded.
                lwswitch_set_minion_initialized(device, i, LW_TRUE);

                LWSWITCH_PRINT(device, SETUP,
                    "%s: LWLink MINION %d bootstrap complete signal received.\n",
                    __FUNCTION__, i);

                _lwswitch_print_minion_info(device, i);
                break;
            }

            if (lwswitch_timeout_check(&timeout))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: LWLink MINION %d hit the timeout -"
                    " LW_CMINION_MINION_STATUS = 0x%08x.\n",
                    __FUNCTION__, i, data);
                break;
            }
        }

        //
        // MINION commands can be submitted only if it is initialized. Hence,
        // check its state and test DLCMD.
        //
        // Note that _lwswitch_minion_test_dlcmd checks for
        // `subengMINION[0].initialized = LW_TRUE`.
        //
        if (lwswitch_is_minion_initialized(device, i))
        {
            if (_lwswitch_minion_test_dlcmd(device, (i * 2)))
            {
                LWSWITCH_PRINT(device, SETUP,
                    "%s: LWLink MINION %d successfully bootstrapped and accepting DLCMDs.\n",
                    __FUNCTION__, i);
            }
            else
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: LWLink MINION %d failed to execute test DLCMD.\n",
                    __FUNCTION__, i);
                lwswitch_set_minion_initialized(device, i, LW_FALSE);
            }
        }

        // MINION failed to initialize, error out.
        if (!lwswitch_is_minion_initialized(device, i))
        {
            //
            // Check if any falcon interrupts are hit & pending.
            //
            // The below call reports SXids for pending interrupts to help with
            // debugging. The explicit check is required because the interrupts
            // are not yet enabled as the device is still initializing.
            //
            (void)lwswitch_minion_service_falcon_interrupts_sv10(device, i);
            return -LWL_ERR_ILWALID_STATE;
        }
    }

    return LWL_SUCCESS;
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
    sv10_device     *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32            data;
    LWSWITCH_TIMEOUT timeout;
    LwU32            idx_minion;
    LwlStatus        status = LWL_SUCCESS;

    // Find first valid MINION instance
    for (idx_minion = 0; idx_minion < NUM_SIOCTRL_ENGINE_SV10; idx_minion++)
    {
        if (chip_device->subengSIOCTRL[idx_minion].subengMINION[0].valid)
        {
            break;
        }
    }
    if (idx_minion >= NUM_SIOCTRL_ENGINE_SV10)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: No MINIONs instantiated.  Skipping MINION pre-init\n",
            __FUNCTION__);
        goto _lwswitch_minion_pre_init_exit;
    }

    // Since we are not using Falcon DMA to load ucode, set REQUIRE_CTX to FALSE
    LWSWITCH_MINION_WR32_BCAST_SV10(device, _CMINION, _FALCON_DMACTL, 0x0);

    //
    // As soon as we access any falcon reg (above), the scrubber will start scrubbing
    // IMEM and DMEM. Wait for the scrubber to finish scrubbing.
    //
    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    while (1)
    {
        // Check if scrubbing was done for first enabled MINION
        data = LWSWITCH_MINION_RD32_SV10(device, idx_minion, _CMINION, _FALCON_DMACTL);
        if (FLD_TEST_DRF(_CMINION, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, data) &&
            FLD_TEST_DRF(_CMINION, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, data))
        {
            break;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for scrubbing to finish on MINION 0.\n",
                __FUNCTION__);
            status = -LWL_ERR_ILWALID_STATE;
            return status;
        }
    }

_lwswitch_minion_pre_init_exit:
    return status;
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
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32  data, i;
    LwBool bMinionRunning = LW_FALSE;

    for (i = 0; i < (chip_device->numSIOCTRL * NUM_MINION_INSTANCES_SV10); i++)
    {
        if (!chip_device->engSIOCTRL[i].valid ||
            !chip_device->subengSIOCTRL[i].subengMINION[0].valid)
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s: MINION instance %d is not valid.\n",
                 __FUNCTION__, i);
            continue;
        }

        data = LWSWITCH_MINION_RD32_SV10(device, i, _CMINION, _FALCON_IRQSTAT);
        if (FLD_TEST_DRF(_CMINION, _FALCON_IRQSTAT, _HALT, _FALSE, data))
        {
            data = LWSWITCH_MINION_RD32_SV10(device, i, _MINION, _MINION_STATUS);
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
 * @Brief : Bootstrap all MINIONs on the specified device
 *
 * @param[in] device Bootstrap MINIONs on this device
 */
LwlStatus
lwswitch_init_minion_sv10
(
    lwswitch_device *device
)
{
    LwlStatus status = LWL_SUCCESS;

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        return LWL_SUCCESS;
    }

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

    // Copy the ucode to IMEM and DMEM by using backdoor PMB access
    status = _lwswitch_minion_copy_ucode_bc(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to copy MINION ucode in broadcast mode!\n",
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

LwlStatus
lwswitch_set_minion_initialized_sv10
(
    lwswitch_device *device,
    LwU32 idx_minion,
    LwBool initialized
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (!((idx_minion < NUM_SIOCTRL_ENGINE_SV10) &&
          (chip_device->subengSIOCTRL[idx_minion].subengMINION[0].valid)))
    {
        return -LWL_BAD_ARGS;
    }

    chip_device->subengSIOCTRL[idx_minion].subengMINION[0].initialized = initialized;
    return LWL_SUCCESS;
}

LwBool
lwswitch_is_minion_initialized_sv10
(
    lwswitch_device *device,
    LwU32 idx_minion
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (!((idx_minion < NUM_SIOCTRL_ENGINE_SV10) &&
          (chip_device->subengSIOCTRL[idx_minion].subengMINION[0].valid)))
    {
        return LW_FALSE;
    }
    return (chip_device->subengSIOCTRL[idx_minion].subengMINION[0].initialized != 0);
}

LwlStatus
lwswitch_minion_get_dl_status_sv10
(
    lwswitch_device *device,
    LwU32            linkId,
    LwU32            statusIdx,
    LwU32            statusArgs,
    LwU32           *statusData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus
lwswitch_ctrl_config_eom_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_CONFIG_EOM *p
)
{
    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, p->link))
    {
        return -LWL_BAD_ARGS;
    }

    return lwswitch_minion_send_command_sv10(device, p->link, LW_MINION_LWLINK_DL_CMD_COMMAND_CONFIGEOM, p->params);
}

LwlStatus
lwswitch_ctrl_read_uphy_pad_lane_reg_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG *p
)
{
    LwU32 read_uphy_param;
    LwU32 val = 0;
    LwlStatus status = LWL_SUCCESS;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, p->link))
    {
        return -LWL_BAD_ARGS;
    }

    if (p->lane >= 8)
    {
        return -LWL_BAD_ARGS;
    }

    read_uphy_param = DRF_NUM(_MINION, _UCODE_READREGPADLANE, _LANE, p->lane) |
                      DRF_NUM(_MINION, _UCODE_READREGPADLANE, _ADDR, p->addr);

    status = lwswitch_minion_send_command_sv10(device, p->link,
                                             LW_MINION_LWLINK_DL_CMD_COMMAND_READREGPADLANE,
                                             read_uphy_param);

    if (status == LWL_SUCCESS)
    {
        val = LWSWITCH_LINK_RD32_SV10(device, p->link, DLPL, _PLWL, _BR0_PAD_CTL_7(p->lane));
        p->phy_config_data = DRF_VAL(_PLWL, _BR0_PAD_CTL_7, _CFG_RDATA, val);
    }

    return status;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
