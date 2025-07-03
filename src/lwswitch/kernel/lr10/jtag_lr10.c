/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/jtag_lr10.h"
#include "lr10/fuse_lr10.h"

#include "lwswitch/lr10/dev_host.h"
#include "lwswitch/lr10/dev_trim.h"
#include "lwswitch/lr10/dev_fuse.h"

//
// JTAG functions (adapted from RM GP100 implementation drivers\resman\kernel\gpu\pascal\gpugp100.c)
//

LwlStatus
lwswitch_jtag_read_seq_lr10
(
    lwswitch_device *device,
    LwU32 chainLen,
    LwU32 chipletSel,
    LwU32 instrId,
    LwU32 *data,
    LwU32 dataArrayLen      // in dwords
)
{
    LwU32 jtagCtrl, jtagConfig, jtagData;
    LwU32 dwordEn, maxDwordEn;
    LwlStatus retval = LWL_SUCCESS;
    LWSWITCH_TIMEOUT timeout;
    LwU32 clusterSel = chipletSel;
    LwU32 jtag_clock_en;

    // Sanity check
    LWSWITCH_ASSERT(data != NULL);
    LWSWITCH_ASSERT(dataArrayLen == chainLen/32 + !!(chainLen%32));

    maxDwordEn  = dataArrayLen;
    dwordEn     = 0;
    jtagCtrl    = 0;
    jtagConfig  = 0;
    jtagData    = 0;

    // Make sure JTAG clock is enabled
    jtag_clock_en = LWSWITCH_REG_RD32(device, _PCLOCK_LWSW, _JTAGINTFC);
    LWSWITCH_REG_WR32(device, _PCLOCK_LWSW, _JTAGINTFC,
        FLD_SET_DRF(_PCLOCK_LWSW, _JTAGINTFC, _JTAGTM_INTFC_CLK_EN, _ON, jtag_clock_en));

    // Make sure to turn off host2jtag request - this should be redundant
    LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, 0);

    while (dwordEn < maxDwordEn)
    {
        // Write DWORD_EN, REG_LENGTH, BURST to ACCESS_CONFIG
        jtagConfig =
            FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CONFIG, _REG_LENGTH, chainLen>>11, jtagConfig);
        jtagConfig =
            FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CONFIG, _DWORD_EN, dwordEn>>6, jtagConfig);
        jtagConfig =
            FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CONFIG, _BURST, 1, jtagConfig);

        LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CONFIG, jtagConfig);

        // Update DWORD_EN
        jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _REG_LENGTH, chainLen, jtagCtrl);
        jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _INSTR_ID, instrId, jtagCtrl);
        jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _CLUSTER_SEL, clusterSel, jtagCtrl);
        jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _DWORD_EN, dwordEn, jtagCtrl);
        jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _REQ_CTRL, 1, jtagCtrl);
        LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, jtagCtrl);

        //
        // It is possible that decode traps have disallowed us write access to
        // host2jtag registers and in that case we would not have been able to
        // write to LW_PJTAG_ACCESS_CTRL. Check if that was the case, and if
        // so, return an error
        //
        jtagCtrl = LWSWITCH_OFF_RD32(device, LW_PJTAG_ACCESS_CTRL);
        if (jtagCtrl == 0)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: No JTAG write access\n", __FUNCTION__);

            //
            // We can just return early without needing to turn off host2jtag
            // since we didn't manage to turn it on anyway
            //
            return -LWL_ERR_ILWALID_STATE;
        }

        // Poll until CTRL_STATUS == 1
        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
        while (!FLD_TEST_DRF_NUM(_PJTAG, _ACCESS_CTRL, _CTRL_STATUS, 1, jtagCtrl))
        {
            if (lwswitch_timeout_check(&timeout))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s - timeout error waiting for _CTRL_STATUS = 1\n",
                    __FUNCTION__);

                // Turn off host2jtag request
                LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, 0);
                return -LWL_ERR_GENERIC;
            }
            jtagCtrl = LWSWITCH_OFF_RD32(device, LW_PJTAG_ACCESS_CTRL);
        }

        // Read ACCESS_DATA
        jtagData = LWSWITCH_OFF_RD32(device, LW_PJTAG_ACCESS_DATA);

        if ((dwordEn == (maxDwordEn-1)) && ((chainLen%32) != 0))
        {
            jtagData = jtagData >> (32-((chainLen)%32));
        }
        *(data+dwordEn) = jtagData;

        // Increment
        dwordEn++;
    }

    // Turn off host2jtag request
    LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, 0);

    // Restore JTAG clock enable
    LWSWITCH_REG_WR32(device, _PCLOCK_LWSW, _JTAGINTFC, jtag_clock_en);

    return retval;
}

LwlStatus
lwswitch_jtag_write_seq_lr10
(
    lwswitch_device *device,
    LwU32 chainLen,
    LwU32 chipletSel,
    LwU32 instrId,
    LwU32 *data,
    LwU32 dataArrayLen      // in dwords
)
{
    LwU32 jtagCtrl, jtagConfig;
    LwU32 dwordEn, maxDwordEn;
    LwlStatus retval = LWL_SUCCESS;
    LWSWITCH_TIMEOUT timeout;
    LwU32 clusterSel = chipletSel;
    LwU32 jtag_clock_en;

    // Sanity check
    LWSWITCH_ASSERT(data != NULL);
    LWSWITCH_ASSERT(dataArrayLen == chainLen/32 + !!(chainLen%32));

    maxDwordEn  = dataArrayLen;
    dwordEn     = 0;
    jtagCtrl    = 0;
    jtagConfig  = 0;

    // Make sure JTAG clock is enabled
    jtag_clock_en = LWSWITCH_REG_RD32(device, _PCLOCK_LWSW, _JTAGINTFC);
    LWSWITCH_REG_WR32(device, _PCLOCK_LWSW, _JTAGINTFC,
        FLD_SET_DRF(_PCLOCK_LWSW, _JTAGINTFC, _JTAGTM_INTFC_CLK_EN, _ON, jtag_clock_en));

    // Turn off host2jtag request
    LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, 0);

    // Write DWORD_EN and REG_LENGTH to ACCESS_CONFIG
    jtagConfig =
        FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CONFIG, _REG_LENGTH, chainLen>>11, jtagConfig);
    LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CONFIG, jtagConfig);

    jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _REG_LENGTH, chainLen, jtagCtrl);
    jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _INSTR_ID, instrId, jtagCtrl);
    jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _CLUSTER_SEL, clusterSel, jtagCtrl);
    jtagCtrl = FLD_SET_DRF_NUM(_PJTAG, _ACCESS_CTRL, _REQ_CTRL, 1, jtagCtrl);
    LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, jtagCtrl);

    //
    // It is possible that decode traps have disallowed us write access to
    // host2jtag registers and in that case we would not have been able to
    // write to LW_PJTAG_ACCESS_CTRL. Check if that was the case, and if so,
    // return an error
    //
    jtagCtrl = LWSWITCH_OFF_RD32(device, LW_PJTAG_ACCESS_CTRL);
    if (jtagCtrl == 0)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: No JTAG write access\n", __FUNCTION__);

        //
        // We can just return early without needing to turn off host2jtag
        // since we didn't manage to turn it on anyway
        //
        return -LWL_ERR_ILWALID_STATE;
    }

    // Poll until CTRL_STATUS == 1
    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    while (!FLD_TEST_DRF_NUM(_PJTAG, _ACCESS_CTRL, _CTRL_STATUS, 1, jtagCtrl))
    {
        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s - timeout error waiting for _CTRL_STATUS = 1\n",
                __FUNCTION__);

            // Turn off host2jtag request
            LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, 0);
            return -LWL_ERR_GENERIC;
        }
        jtagCtrl = LWSWITCH_OFF_RD32(device, LW_PJTAG_ACCESS_CTRL);
    }

    // Start writing data
    while (dwordEn < maxDwordEn)
    {
        LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_DATA, *(data+dwordEn));
        dwordEn++;
    }

    // Turn off host2jtag request
    LWSWITCH_OFF_WR32(device, LW_PJTAG_ACCESS_CTRL, 0);

    // Restore JTAG clock enable
    LWSWITCH_REG_WR32(device, _PCLOCK_LWSW, _JTAGINTFC, jtag_clock_en);

    return retval;
}
