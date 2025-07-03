/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwlink_export.h"
#include "common_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/minion_lr10.h"
#include "lr10/minion_production_ucode_lr10_dbg.h"
#include "lr10/minion_production_ucode_lr10_prod.h"
#include "regkey_lwswitch.h"

#include "lwswitch/lr10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/lr10/dev_minion_ip.h"
#include "lwswitch/lr10/dev_minion_ip_addendum.h"
#include "lwswitch/lr10/dev_ingress_ip.h"
#include "lwswitch/lr10/dev_egress_ip.h"

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

    for (i = 0; i < LWSWITCH_ENG_COUNT(device, MINION, ); i++)
    {
        if (!LWSWITCH_ENG_IS_VALID(device, MINION, i))
        {
            LWSWITCH_PRINT(device, SETUP,
                "%s: MINION instance %d is not valid.\n",
                 __FUNCTION__, i);
            continue;
        }

        data = LWSWITCH_MINION_RD32_LR10(device, i, _CMINION, _FALCON_IRQSTAT);
        if (FLD_TEST_DRF(_CMINION, _FALCON_IRQSTAT, _HALT, _FALSE, data))
        {
            data = LWSWITCH_MINION_RD32_LR10(device, i, _MINION, _MINION_STATUS);
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
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    // Find first valid MINION instance
    for (idx_minion = 0; idx_minion < LWSWITCH_ENG_COUNT(device, MINION, ); idx_minion++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, MINION, idx_minion))
        {
            break;
        }
    }
    if (idx_minion >= LWSWITCH_ENG_COUNT(device, MINION, ))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: No MINIONs instantiated.  Skipping MINION pre-init\n",
            __FUNCTION__);
        goto _lwswitch_minion_pre_init_exit;
    }

    // Since we are not using Falcon DMA to load ucode, set REQUIRE_CTX to FALSE
    LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_DMACTL, 0x0);

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

    LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IRQMSET, falconIntrMask);
    LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IRQDEST, falconIntrDest);
    chip_device->intr_minion_dest = falconIntrDest;

    //
    // As soon as we access any falcon reg (above), the scrubber will start scrubbing
    // IMEM and DMEM. Wait for the scrubber to finish scrubbing.
    //
    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(10*LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    }
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        // Check if scrubbing was done for first enabled MINION
        data = LWSWITCH_MINION_RD32_LR10(device, idx_minion, _CMINION, _FALCON_DMACTL);
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
    const PFALCON_UCODE_HDR_INFO_LR10 pUcodeHeader =
        (PFALCON_UCODE_HDR_INFO_LR10) &minion_ucode_header[0];
    const LwU32 *pHeader = &minion_ucode_header[0];

    LwU32 data, i, app, dataSize;
    LwU32 appCodeOffset, appCodeSize, appDataOffset, appDataSize;
    LwU16 tag;
    LwU32 idx_minion;

    // Find first valid MINION instance
    for (idx_minion = 0; idx_minion < LWSWITCH_ENG_COUNT(device, MINION, ); idx_minion++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, MINION, idx_minion))
        {
            break;
        }
    }
    if (idx_minion >= LWSWITCH_ENG_COUNT(device, MINION, ))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: No MINIONs instantiated.  Skipping MINION ucode load\n",
            __FUNCTION__);
        goto _lwswitch_minion_copy_ucode_bc_exit;
    }

    dataSize = sizeof(minion_ucode_data[0]);

    // Initialize address of IMEM to 0x0 and set auto-increment on write
    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_IMEMC, _AINCW, _TRUE, data);
    LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMC(0), data);

    //
    // Initialize IMEM tag to 0 explicitly even though power-on value is 0.
    // Writes to IMEM don't work if we don't do this
    //
    tag = 0;
    LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMT(0), tag);

    // Copy over IMEM part of the ucode and tag along the way
    for (i = 0; i < (pUcodeHeader->osCodeSize / dataSize) ; i++)
    {
        // Increment tag for after every block (256 bytes)
        if (i && ((i % (FALCON_IMEM_BLK_SIZE_BYTES_LR10 / dataSize)) == 0))
        {
            tag++;
            LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMT(0), (LwU32) tag);
        }

        // Copy IMEM DWORD by DWORD
        data = minion_ucode_data[(pUcodeHeader->osCodeOffset / dataSize) + i];
        LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMD(0), data);
    }

    // Initialize address of DMEM to 0x0 and set auto-increment on write
    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_DMEMC, _AINCW, _TRUE, data);
    LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_DMEMC(0), data);

    // Copy over DMEM part of the ucode
    for (i = 0; i < (pUcodeHeader->osDataSize / dataSize) ; i++)
    {
        // Copy DMEM DWORD by DWORD
        data = minion_ucode_data[(pUcodeHeader->osDataOffset / dataSize) + i];
        LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_DMEMD(0), data);
    }

    // Copy over any apps in the ucode with the appropriate tags
    if (pUcodeHeader->numApps)
    {
        for (app = 0; app < pUcodeHeader->numApps ; app++)
        {
            // Index into the app code info
            appCodeOffset = pHeader[FALCON_CODE_HDR_APP_CODE_START_LR10 + 2*app];
            appCodeSize   = pHeader[FALCON_CODE_HDR_APP_CODE_START_LR10 + 2*app + 1];

            // Index into the app data info using appCodeStart offset as a base
            appDataOffset = pHeader[FALCON_CODE_HDR_APP_CODE_START_LR10 +
                2*pUcodeHeader->numApps + 2*app];
            appDataSize   = pHeader[FALCON_CODE_HDR_APP_CODE_START_LR10 +
                2*pUcodeHeader->numApps + 2*app + 1];

            // Mark the following IMEM blocks as secure
            data = LWSWITCH_MINION_RD32_LR10(device, idx_minion, _CMINION, _FALCON_IMEMC(0));
            data = FLD_SET_DRF_NUM(_CMINION, _FALCON_IMEMC, _SELWRE, 0x1, data);
            LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMC(0), data);

            // Copy to IMEM and tag along the way
            tag = (LwU16)(appCodeOffset >> 8);
            LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMT(0), tag);

            // Copy app code to IMEM picking up where last load left off
            for (i = 0; i < (appCodeSize / dataSize); i++)
            {
                if (i && ((i % (FALCON_IMEM_BLK_SIZE_BYTES_LR10 / dataSize)) == 0))
                {
                    tag++;
                    LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMT(0), tag);
                }

                data = minion_ucode_data[(appCodeOffset / dataSize) + i];
                LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_IMEMD(0), data);
            }

            // Copy app data to DMEM picking up where last load left off
            for (i = 0; i < (appDataSize / dataSize); i++)
            {
                data = minion_ucode_data[appDataOffset + i];
                LWSWITCH_MINION_WR32_BCAST_LR10(device, _CMINION, _FALCON_DMEMD(0), data);
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
    LWSWITCH_MINION_WR32_LR10(device, instance, _CMINION, _FALCON_IMEMC(0), data);

    LWSWITCH_PRINT(device, SETUP, "MINION IMEMD = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LR10(device, instance, _CMINION, _FALCON_IMEMD(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);

    LWSWITCH_PRINT(device, SETUP, "MINION IMEMC = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LR10(device, instance, _CMINION, _FALCON_IMEMC(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);

    data = 0;
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _OFFS, 0x0, data);
    data = FLD_SET_DRF_NUM(_CMINION, _FALCON_DMEMC, _BLK, 0x0, data);
    data = FLD_SET_DRF(_CMINION, _FALCON_DMEMC, _AINCR, _TRUE, data);
    LWSWITCH_MINION_WR32_LR10(device, instance, _CMINION, _FALCON_DMEMC(0), data);

    LWSWITCH_PRINT(device, SETUP, "MINION DMEMD = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LR10(device, instance, _CMINION, _FALCON_DMEMD(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);

    LWSWITCH_PRINT(device, SETUP, "MINION DMEMC = \n");
    for (i = 0; i < 8 ; i++)
    {
        buf[i] = LWSWITCH_MINION_RD32_LR10(device, instance, _CMINION, _FALCON_DMEMC(0));
    }
    LWSWITCH_PRINT(device, SETUP, " 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                   buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
#endif  //defined(DEBUG) || defined(DEVELOP) || defined(LW_MODS)
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
    localLinkNumber = linkNumber % LWSWITCH_LINKS_PER_MINION;

    if (lwswitch_minion_send_command(device, linkNumber,
        LW_MINION_LWLINK_DL_CMD_COMMAND_SWINTR, 0) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: SWINTR DL CMD failed for link %d.\n",
            __FUNCTION__, linkNumber);
        return LW_FALSE;
    }

    interrupts = LWSWITCH_MINION_LINK_RD32_LR10(device, linkNumber,
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
        LWSWITCH_MINION_LINK_WR32_LR10(device, linkNumber, _MINION,
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

    falcon_os = LWSWITCH_MINION_RD32_LR10(device, id, _CMINION, _FALCON_OS);
    falcon_mailbox = LWSWITCH_MINION_RD32_LR10(device, id, _CMINION, _FALCON_MAILBOX1);
    falcon_sctl = LWSWITCH_MINION_RD32_LR10(device, id, _CMINION, _FALCON_SCTL);

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
    LwU32            data, i, link_num;
    LwU64            link_mask;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwlStatus        status = LWL_SUCCESS;

    for (i = 0; i < LWSWITCH_ENG_COUNT(device, MINION, ) ; i++)
    {
        if (!LWSWITCH_ENG_IS_VALID(device, MINION, i))
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
        _lwswitch_minion_print_ucode(device, i);

        // Write boot vector to 0x0
        data = LWSWITCH_MINION_RD32_LR10(device, i, _CMINION, _FALCON_BOOTVEC);
        data = FLD_SET_DRF_NUM(_CMINION, _FALCON_BOOTVEC, _VEC, 0x0, data);
        LWSWITCH_MINION_WR32_LR10(device, i, _CMINION, _FALCON_BOOTVEC, data);

        //
        // Start the Falcon
        // If a falcon is managed (and thus supports secure mode), we need to
        // write LW_PFALCON_FALCON_CPUCTL_ALIAS_STARTCPU = _TRUE.
        // Below write is a nop in secure mode.
        //
        data = LWSWITCH_MINION_RD32_LR10(device, i, _CMINION, _FALCON_CPUCTL);
        data = FLD_SET_DRF(_CMINION, _FALCON_CPUCTL, _STARTCPU, _TRUE, data);
        LWSWITCH_MINION_WR32_LR10(device, i, _CMINION, _FALCON_CPUCTL, data);

        if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
        {
            lwswitch_timeout_create(10*LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
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

            data = LWSWITCH_MINION_RD32_LR10(device, i, _MINION, _MINION_STATUS);

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
            lwswitch_minion_service_falcon_interrupts_lr10(device, i);

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
        link_mask = LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(i);
        FOR_EACH_INDEX_IN_MASK(64, link_num, link_mask)
        {
            // Pick a valid lick in this LWLipt
            if (lwswitch_is_link_valid(device, link_num))
            {
                break;
            }
        }
        FOR_EACH_INDEX_IN_MASK_END;

        if (!_lwswitch_minion_test_dlcmd(device, link_num))
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
 * @Brief : Send MINION DL CMD for a particular link
 *
 * @param[in] device     Send command to MINION on this device
 * @param[in] linkNumber DLCMD will be sent on this link number
 *
 * @return           Returns true if the DLCMD passed
 */
LwlStatus
lwswitch_minion_send_command_lr10
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

    localLinkNumber = linkNumber % LWSWITCH_LINKS_PER_MINION;

    if (!lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, linkNumber, MINION)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION %d is not initialized for link %08x.\n",
            __FUNCTION__, LWSWITCH_GET_LINK_ENG_INST(device, linkNumber, MINION),
            linkNumber);
        return LWL_SUCCESS;
    }

    data = LWSWITCH_MINION_LINK_RD32_LR10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber));
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
            LWSWITCH_MINION_WR32_LR10(device,
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

            ingressEccRegVal = LWSWITCH_NPORT_RD32_LR10(device, linkNumber, _INGRESS, _ERR_ECC_CTRL);
            LWSWITCH_NPORT_WR32_LR10(device, linkNumber, _INGRESS, _ERR_ECC_CTRL,
                FLD_SET_DRF(_INGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, _DISABLE, ingressEccRegVal));

            egressEccRegVal = LWSWITCH_NPORT_RD32_LR10(device, linkNumber, _EGRESS, _ERR_ECC_CTRL);
            LWSWITCH_NPORT_WR32_LR10(device, linkNumber, _EGRESS, _ERR_ECC_CTRL,
                FLD_SET_DRF(_EGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, _DISABLE, egressEccRegVal));
            break;
        default:
            break;
    }

    data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _COMMAND, command, data);
    data = FLD_SET_DRF_NUM(_MINION, _LWLINK_DL_CMD, _FAULT,   1,   data);
    LWSWITCH_MINION_LINK_WR32_LR10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber), data);

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

        data = LWSWITCH_MINION_LINK_RD32_LR10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber));
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
                LWSWITCH_MINION_LINK_WR32_LR10(device, linkNumber, _MINION, _LWLINK_DL_CMD(localLinkNumber), data);
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

        if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
        {
            lwswitch_os_sleep(1);
        }
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
        LWSWITCH_NPORT_WR32_LR10(device, linkNumber, _INGRESS, _ERR_ECC_CTRL, ingressEccRegVal);
        LWSWITCH_NPORT_WR32_LR10(device, linkNumber, _EGRESS,  _ERR_ECC_CTRL, egressEccRegVal);
    }

    return LWL_SUCCESS;
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
    data = LWSWITCH_MINION_RD32_LR10(device, 0, _CMINION, _SCP_CTL_STAT);
    bDebugMode = FLD_TEST_DRF(_CMINION, _SCP_CTL_STAT, _DEBUG_MODE, _DISABLED, data) ?
                 (LW_FALSE) : (LW_TRUE);

    //
    // If ucode load fails via regkey fallback to the default ucode.
    // Copy the ucode to IMEM and DMEM by using backdoor PMB access
    //
    if (bDebugMode)
    {
        status = _lwswitch_minion_copy_ucode_bc(device, minion_ucode_data_lr10_dbg, minion_ucode_header_lr10_dbg);
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
        status = _lwswitch_minion_copy_ucode_bc(device, minion_ucode_data_lr10_prod, minion_ucode_header_lr10_prod);
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
lwswitch_init_minion_lr10
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

LwlStatus
lwswitch_minion_get_dl_status_lr10
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
    localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION;

    if (!lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: MINION %d is not initialized for link %08x.\n",
            __FUNCTION__, LWSWITCH_GET_LINK_ENG_INST(device, linkId, MINION),
            linkId);
        return -LWL_ERR_ILWALID_STATE;
    }

    // Query the DL status interface to get the data
    LWSWITCH_MINION_LINK_WR32_LR10(device, linkId, _MINION, _LWLINK_DL_STAT(localLinkNumber),
            DRF_NUM(_MINION, _LWLINK_DL_STAT, _ARGS, statusArgs) |
            DRF_NUM(_MINION, _LWLINK_DL_STAT, _STATUSIDX, statusIdx));

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(20 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    }

    // Poll for READY bit to be set
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        regData = LWSWITCH_MINION_LINK_RD32_LR10(device, linkId, _MINION, _LWLINK_DL_STAT(localLinkNumber));
        if (FLD_TEST_DRF_NUM(_MINION, _LWLINK_DL_STAT, _READY, 1, regData))
        {
            *statusData = LWSWITCH_MINION_LINK_RD32_LR10(device, linkId, _MINION, _LWLINK_DL_STATDATA(localLinkNumber));
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

LwlStatus
lwswitch_minion_get_initoptimize_status_lr10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwU32            statData;
    LwlStatus        status;

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(100 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(20 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }

    // Poll for READY bit to be set
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        // Poll for INITOPTIMIZE status on MINION DL STAT interface
        status = lwswitch_minion_get_dl_status(device, linkId, LW_LWLSTAT_UC01, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }

        if (FLD_TEST_DRF_NUM(_LWLSTAT, _UC01, _TRAINING_GOOD, 0x1, statData))
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: INITOPTIMIZE TRAINING_GOOD on link: %d\n",
                __FUNCTION__, linkId);
            return LWL_SUCCESS;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    LWSWITCH_PRINT(device, ERROR,
        "%s: Timeout waiting for INITOPTIMIZE TRAINING_GOOD on link: %d\n",
        __FUNCTION__, linkId);
    return -LWL_ERR_ILWALID_STATE;
}

LwlStatus
lwswitch_minion_get_initnegotiate_status_lr10
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
        lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(2 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }

    // Poll for READY bit to be set
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        // Check INITNEGOTIATE status on MINION DL STAT interface
        status = lwswitch_minion_get_dl_status(device, linkId, LW_LWLSTAT_UC01, 0, &statData);
        if (status != LWL_SUCCESS)
        {
            return status;
        }

        if (FLD_TEST_DRF(_LWLSTAT, _UC01, _CONFIG_GOOD, _SUCCESS, statData))
        {
            LWSWITCH_PRINT(device, INFO,
                    "%s: INITNEGOTIATE CONFIG_GOOD on link: %d\n",
                    __FUNCTION__, linkId);

            return LWL_SUCCESS;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    LWSWITCH_PRINT(device, ERROR,
        "%s: Timeout waiting for INITNEGOTIATE CONFIG_GOOD on link: %d\n",
        __FUNCTION__, linkId);

    return -LWL_ERR_ILWALID_STATE;
}

LwlStatus
lwswitch_minion_get_rxdet_status_lr10
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
        lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
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
                LWSWITCH_REVERSE_BITMASK_32(LWSWITCH_NUM_LANES_LR10,
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
                LWSWITCH_REVERSE_BITMASK_32(LWSWITCH_NUM_LANES_LR10,
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

LwlStatus
lwswitch_minion_set_rx_term_lr10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    if (lwswitch_minion_send_command(device, linkId,
        LW_MINION_LWLINK_DL_CMD_COMMAND_INITRXTERM, 0) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: INITRXTERM DL CMD failed for link %d.\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_ILWALID_STATE;
    }

    return LWL_SUCCESS;
}

/*
* @brief: This function will restore seed data back into MINION for training
* @params[in] device        reference to current lwswitch device
* @params[in] linkId        link we want to save seed data for
* @params[in] seedData      referene to a buffer to read
*/
LwlStatus 
lwswitch_minion_restore_seed_data_lr10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *seedData
)
{
    LwlStatus status = LWL_SUCCESS;
    LwU32 localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION;
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
    LWSWITCH_MINION_LINK_WR32_LR10(device, linkId, _MINION_LWLINK_DL_CMD,
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

        LWSWITCH_MINION_LINK_WR32_LR10(device, linkId, _MINION_LWLINK_DL_CMD,
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
lwswitch_minion_save_seed_data_lr10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *seedData
)
{
    LwlStatus status = LW_OK;
    LwU32 localLinkNumber = linkId % LWSWITCH_LINKS_PER_MINION;
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
    size = LWSWITCH_MINION_LINK_RD32_LR10(device, linkId, _MINION_LWLINK_DL_CMD,
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

        seedData[i] = LWSWITCH_MINION_LINK_RD32_LR10(device, linkId, _MINION_LWLINK_DL_CMD,
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

LwU32
lwswitch_minion_get_line_rate_Mbps_lr10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwU32     statData;
    LwlStatus status;

    status = lwswitch_minion_get_dl_status(device, linkId, LW_LWLSTAT_LNK3, 0, &statData);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to retrieve LINERATE from MINION DLSTAT for link %d.\n",
            __FUNCTION__, linkId);
    }

    return DRF_NUM(_LWLSTAT, _LNK3, _LINERATE, statData);
}

LwU32
lwswitch_minion_get_data_rate_KiBps_lr10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwU32     statData;
    LwlStatus status;

    status = lwswitch_minion_get_dl_status(device, linkId, LW_LWLSTAT_LNK5, 0, &statData);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to retrieve DATARATE from MINION DLSTAT for link %d.\n",
            __FUNCTION__, linkId);
    }

    return DRF_NUM(_LWLSTAT, _LNK5, _DATARATE, statData);
}

LwlStatus
lwswitch_set_minion_initialized_lr10
(
    lwswitch_device *device,
    LwU32 idx_minion,
    LwBool initialized
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    if (!LWSWITCH_ENG_VALID_LR10(device, MINION, idx_minion))
    {
        return -LWL_BAD_ARGS;
    }

    chip_device->engMINION[idx_minion].initialized = initialized;
    return LWL_SUCCESS;
}

LwBool
lwswitch_is_minion_initialized_lr10
(
    lwswitch_device *device,
    LwU32 idx_minion
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    if (!LWSWITCH_ENG_VALID_LR10(device, MINION, idx_minion))
    {
        return LW_FALSE;
    }
    return (chip_device->engMINION[idx_minion].initialized != 0);
}

LwlStatus
lwswitch_minion_clear_dl_error_counters_lr10
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
        LWSWITCH_PRINT(device, ERROR, "%s : Failed to clear error count to MINION for (%s):(%d).\n",
            __FUNCTION__, device->name, linkId);
    }
    return status;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus
lwswitch_ctrl_config_eom_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_CONFIG_EOM *p
)
{
    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, p->link))
    {
        return -LWL_BAD_ARGS;
    }

    return lwswitch_minion_send_command(device, p->link, LW_MINION_LWLINK_DL_CMD_COMMAND_CONFIGEOM, p->params);
}

LwlStatus
lwswitch_ctrl_read_uphy_pad_lane_reg_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG *p
)
{
    LwlStatus status = LWL_SUCCESS;
    LwU32 statData;
    LwU32 statArgs;

    statArgs = DRF_NUM(_MINION, _LWLINK_DL_STAT, _ARGS_LANEID, p->lane) |
                   DRF_NUM(_MINION, _LWLINK_DL_STAT, _ARGS_ADDRS, p->addr);

    status = lwswitch_minion_get_dl_status(device, p->link, LW_LWLSTAT_DB10, statArgs, &statData);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO,"%s : Failed to read uphy pad lane for (%s):(%d)\n",
                        __FUNCTION__, device->name, p->link);
    }
    else
    {
        p->phy_config_data = statData;
    }

    return status;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
