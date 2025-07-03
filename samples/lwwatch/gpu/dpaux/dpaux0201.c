/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "kepler/gk104/dev_pmgr.h"
#include "displayport.h"
#include "hal.h"
#include "g_dpaux_private.h"     // (rmconfig)  implementation prototypes

BOOL
dpauxGetHpdStatus_v02_01
(
    LwU32 port
)
{
    LwU32 reg;

    reg = GPU_REG_RD32(LW_PMGR_DP_AUXSTAT(port));
    return (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _HPD_STATUS, _PLUGGED, reg));
}

BOOL
dpauxIsPadPowerUpForPort_v02_01
(
    LwU32 port
)
{
    LwU32   reg;

    reg = GPU_REG_RD32(LW_PMGR_HYBRID_SPARE(port));
    return (FLD_TEST_DRF(_PMGR, _HYBRID_SPARE, _PAD_PWR, _POWERUP, reg));
}

BOOL
dpauxHybridAuxInDpMode_v02_01
(
    LwU32 port
)
{
    LwU32   reg;

    reg = GPU_REG_RD32(LW_PMGR_HYBRID_PADCTL(port));
    return (FLD_TEST_DRF(_PMGR, _HYBRID_PADCTL, _MODE, _AUX, reg));
}

LwS8 dpauxChRead_v02_01(LwU32 port, LwU32 addr)
{
    LwU8 data;

    // 1 byte to read
    if (dpauxChReadMulti_v02_01(port, addr, &data, 1) == 1)
        return data;
    else
        return -1;
}

void dpauxChWrite_v02_01(LwU32 port, LwU32 addr, LwU8 auxdata)
{
    // 1 byte to write
    dpauxChWriteMulti_v02_01(port, addr, &auxdata, 1);
}

LwU32 dpauxChReadMulti_v02_01
(
    LwU32 port,
    LwU32 addr,
    LwU8 *data,
    LwU32 reqBytes
)
{
    LwU32 auxCtl, auxStat, hybridMode, size, timeOutRetries, maxDeferRetries;
    LwU32 finishedBytes, auxData[DP_AUX_CHANNEL_MAX_BYTES / sizeof(LwU32)];
    BOOL  isDone;

    if (port > LW_PMGR_HYBRID_PADCTL__SIZE_1)
    {
        dprintf ("lw: Incorrect port number (%d), max port number allowed is %d\n",
            port, LW_PMGR_HYBRID_PADCTL__SIZE_1);
        return 0;
    }

    // Check if the hybrid pad is in AUX mode
    hybridMode = GPU_REG_RD32(LW_PMGR_HYBRID_PADCTL(port));
    if (FLD_TEST_DRF(_PMGR, _HYBRID_PADCTL, _MODE, _I2C, hybridMode))
    {
        dprintf ("lw: Hybrid DpAux/I2c pads are in I2C mode, switch to dpaux mode before performing aux reads\n");
        return 0;
    }

    auxStat = GPU_REG_RD32(LW_PMGR_DP_AUXSTAT(port));
    if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _HPD_STATUS, _UNPLUG, auxStat))
    {
        dprintf("ERROR: %s: DP not plugged in. Bailing out early\n",
            __FUNCTION__);
        return 0;
    }

    isDone = FALSE;
    timeOutRetries = DP_AUX_CHANNEL_TIMEOUT_MAX_TRIES;
    maxDeferRetries = DP_AUX_CHANNEL_DEFAULT_DEFER_MAX_TRIES;
    finishedBytes = 0;
    do
    {
        // set the dpAuxCh addr.
        GPU_REG_WR32(LW_PMGR_DP_AUXADDR(port), addr);

        auxCtl = GPU_REG_RD32(LW_PMGR_DP_AUXCTL(port));
        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _CMD, _AUXRD, auxCtl);

        if (reqBytes - finishedBytes > DP_AUX_CHANNEL_MAX_BYTES)
            size = DP_AUX_CHANNEL_MAX_BYTES - 1;
        else
            size = reqBytes - 1;

        auxCtl = FLD_SET_DRF_NUM(_PMGR, _DP_AUXCTL, _CMDLEN, size, auxCtl);

        // reset aux before initiating the transaction.
        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _ASSERT, auxCtl);
        GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _DEASSERT, auxCtl);
        GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

        osPerfDelay(400);

        // this initiates the transaction.
        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _TRANSACTREQ, _TRIGGER, auxCtl);
        GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

        do
        {
            auxCtl  = GPU_REG_RD32(LW_PMGR_DP_AUXCTL(port));
        } while (FLD_TEST_DRF(_PMGR, _DP_AUXCTL, _TRANSACTREQ, _PENDING, auxCtl));

        auxStat = GPU_REG_RD32(LW_PMGR_DP_AUXSTAT(port));
        if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _TIMEOUT, _PENDING, auxStat) ||
            FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _RX_ERROR, _PENDING, auxStat) ||
            FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _SINKSTAT_ERROR, _PENDING, auxStat) ||
            FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _NO_STOP_ERROR, _PENDING, auxStat))
        {
            if (timeOutRetries)
            {
                timeOutRetries--;
            }
            else
            {
                dprintf("ERROR: %s: running out of timeout retries\n",
                    __FUNCTION__);
                break;
            }
            continue;
        }
        else if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _REPLYTYPE, _DEFER, auxStat))
        {
            if (maxDeferRetries)
            {
                maxDeferRetries--;
            }
            else
            {
                dprintf("ERROR: %s: running out of defer retries\n",
                    __FUNCTION__);
                break;
            }
            continue;
        }
        else if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _REPLYTYPE, _ACK, auxStat))
        {
            // Received bytes is 1-based.
            LwU32 receivedBytes = DRF_VAL(_PMGR, _DP_AUXSTAT, _REPLY_M, auxStat);

            auxData[0] = GPU_REG_RD32(LW_PMGR_DP_AUXDATA_READ_W0(port));
            if (receivedBytes > sizeof(LwU32))
                auxData[1] = GPU_REG_RD32(LW_PMGR_DP_AUXDATA_READ_W1(port));
            if (receivedBytes > sizeof(LwU32) * 2)
                auxData[2] = GPU_REG_RD32(LW_PMGR_DP_AUXDATA_READ_W2(port));
            if (receivedBytes > sizeof(LwU32) * 3)
                auxData[3] = GPU_REG_RD32(LW_PMGR_DP_AUXDATA_READ_W3(port));

            memcpy(data, (LwU8*)auxData, receivedBytes);
            finishedBytes += receivedBytes;
            if (finishedBytes == reqBytes)
            {
                isDone = TRUE;
            }
            else if (finishedBytes < reqBytes)
            {
                data += receivedBytes;
                addr += receivedBytes;
            }
            else
            {
                dprintf("ERROR: %s: Too many bytes received\n", __FUNCTION__);
                break;
            }
        }
        else
        {
            dprintf("ERROR: %s: got invalid reply\n", __FUNCTION__);
            break;
        }
    } while (isDone == FALSE);

    // reset after the transaction completes.
    auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _ASSERT, auxCtl);
    GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

    auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _DEASSERT, auxCtl);
    GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

    return finishedBytes;
}

LwU32 dpauxChWriteMulti_v02_01
(
    LwU32 port,
    LwU32 addr,
    LwU8 *data,
    LwU32 reqBytes
)
{
    LwU32 auxCtl, auxStat, hybridMode, size, maxDeferRetries, finishedBytes;
    LwU32 auxData[DP_AUX_CHANNEL_MAX_BYTES / sizeof(LwU32)];
    BOOL  isDone;

    if (port > LW_PMGR_HYBRID_PADCTL__SIZE_1)
    {
        dprintf("lw: Incorrect port number (%d), max port number allowed is %d\n",
            port, LW_PMGR_HYBRID_PADCTL__SIZE_1);
        return 0;
    }

    // Check if the hybrid pad is in AUX mode
    hybridMode = GPU_REG_RD32(LW_PMGR_HYBRID_PADCTL(port));
    if (FLD_TEST_DRF(_PMGR, _HYBRID_PADCTL, _MODE, _I2C, hybridMode))
    {
        dprintf("lw: Hybrid DpAux/I2c pads are in I2C mode, switch to dpaux mode before performing aux reads\n");
        return 0;
    }

    auxStat = GPU_REG_RD32(LW_PMGR_DP_AUXSTAT(port));
    if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _HPD_STATUS, _UNPLUG, auxStat))
    {
        dprintf("ERROR: %s: DP not plugged in. Bailing out early\n",
            __FUNCTION__);
        return 0;
    }

    isDone = FALSE;
    maxDeferRetries = DP_AUX_CHANNEL_DEFAULT_DEFER_MAX_TRIES;
    finishedBytes = 0;
    do
    {
        // set the dpAuxCh addr
        GPU_REG_WR32(LW_PMGR_DP_AUXADDR(port), addr);

        // size is 0-based
        if (reqBytes - finishedBytes > DP_AUX_CHANNEL_MAX_BYTES)
            size = DP_AUX_CHANNEL_MAX_BYTES - 1;
        else
            size = reqBytes - 1;

        memcpy((LwU8*)auxData, data, size + 1);
        GPU_REG_WR32(LW_PMGR_DP_AUXDATA_WRITE_W0(port), auxData[0]);
        if (size / sizeof(LwU32))
            GPU_REG_WR32(LW_PMGR_DP_AUXDATA_WRITE_W1(port), auxData[1]);
        if (size / sizeof(LwU32) > 1)
            GPU_REG_WR32(LW_PMGR_DP_AUXDATA_WRITE_W2(port), auxData[2]);
        if (size / sizeof(LwU32) > 2)
            GPU_REG_WR32(LW_PMGR_DP_AUXDATA_WRITE_W3(port), auxData[3]);

        auxCtl = GPU_REG_RD32(LW_PMGR_DP_AUXCTL(port));
        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _CMD, _AUXWR, auxCtl);
        auxCtl = FLD_SET_DRF_NUM(_PMGR, _DP_AUXCTL, _CMDLEN, size, auxCtl);

        // reset aux before initiating the transaction.
        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _ASSERT, auxCtl);
        GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _DEASSERT, auxCtl);
        GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

        osPerfDelay(400);

        // this initiates the transaction.
        auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _TRANSACTREQ, _TRIGGER, auxCtl);
        GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

        do
        {
            auxCtl  = GPU_REG_RD32(LW_PMGR_DP_AUXCTL(port));
        } while (FLD_TEST_DRF(_PMGR, _DP_AUXCTL, _TRANSACTREQ, _PENDING, auxCtl));

        auxStat = GPU_REG_RD32(LW_PMGR_DP_AUXSTAT(port));
        if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _TIMEOUT, _PENDING, auxStat) ||
            FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _RX_ERROR, _PENDING, auxStat) ||
            FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _SINKSTAT_ERROR, _PENDING, auxStat) ||
            FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _NO_STOP_ERROR, _PENDING, auxStat))
        {
            dprintf("ERROR: %s: aux error: %x\n", __FUNCTION__, auxStat);
            break;
        }

        if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _REPLYTYPE, _DEFER, auxStat))
        {
            if (maxDeferRetries)
            {
                maxDeferRetries--;
            }
            else
            {
                dprintf("ERROR: %s: running out of defer retries\n",
                    __FUNCTION__);
                break;
            }
            continue;
        }

        if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _REPLYTYPE, _ACK, auxStat))
        {
            // ACK for aux write means success & REPLY_M = 0
            LwU32 sentBytes = size + 1;
            finishedBytes += sentBytes;
            if (finishedBytes == reqBytes)
            {
                isDone = TRUE;
            }
            else
            {
                addr += sentBytes;
                data += sentBytes;
            }
        }
        else if (FLD_TEST_DRF(_PMGR, _DP_AUXSTAT, _REPLYTYPE, _NACK, auxStat))
        {
            // Some bytes could be written with NACK in 1-based.
            LwU32 sentBytes = DRF_VAL(_PMGR, _DP_AUXSTAT, _REPLY_M, auxStat);
            finishedBytes += sentBytes;
            if (sentBytes && finishedBytes < reqBytes)
            {
                addr += sentBytes;
                data += sentBytes;
            }
            else
            {
                dprintf("ERROR: %s: got invalid reply\n", __FUNCTION__);
                break;
            }
        }
        else
        {
            dprintf("ERROR: %s: got invalid reply\n", __FUNCTION__);
            break;
        }
    } while (!isDone);

    // reset after the transaction completes.
    auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _ASSERT, auxCtl);
    GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

    auxCtl = FLD_SET_DRF(_PMGR, _DP_AUXCTL, _RESET, _DEASSERT, auxCtl);
    GPU_REG_WR32(LW_PMGR_DP_AUXCTL(port), auxCtl);

    return finishedBytes;
}
