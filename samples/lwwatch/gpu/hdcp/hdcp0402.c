/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/****************************** LwWatch ***********************************\
*                                                                          *
*                      HDCP V04_02 routines                                *
*                                                                          *
\***************************************************************************/

#include <stdint.h>
#include "inc/disp.h"
#include "t23x/t234/address_map_new.h"
#include "t23x/t234/arclk_rst.h"
#include "t23x/t234/dev_fuse.h"
#include "t23x/t234/dev_arfuse.h"

#include "g_disp_private.h"

#define MAX_TIMEOUT_MS 4

//
// TODO: This needs to be included once HW generates it bug 200747926, as of now hardcoding the register address
// #include "arfuse_kfuse.h"
// #include "arfuse_kfuse.h"
//
#define LW_PFUSE_KFUSE_STATE                          0x00018000      /* RW-4R */
#define LW_PFUSE_KFUSE_STATE_LWRBLOCK                        5:0      /* R--VF */
#define LW_PFUSE_KFUSE_STATE_ERRBLOCK                       13:8      /* R--VF */
#define LW_PFUSE_KFUSE_STATE_DONE                          16:16      /* R--VF */
#define LW_PFUSE_KFUSE_STATE_CRCPASS                       17:17      /* R--VF */
#define LW_PFUSE_KFUSE_STATE_RESTART                       24:24      /* RW-VF */
#define LW_PFUSE_KFUSE_STATE_STOP                          25:25      /* RW-VF */
#define LW_PFUSE_KFUSE_STATE_RESET                         31:31      /* RW-VF */
#define LW_PFUSE_KFUSE_ERRCOUNT                       0x00018004      /* R--4R */
#define LW_PFUSE_KFUSE_ERRCOUNT_ERR_1                        6:0      /* R--VF */
#define LW_PFUSE_KFUSE_ERRCOUNT_ERR_2                       14:8      /* R--VF */
#define LW_PFUSE_KFUSE_ERRCOUNT_ERR_3                      22:16      /* R--VF */
#define LW_PFUSE_KFUSE_ERRCOUNT_ERR_FATAL                  30:24      /* R--VF */
#define LW_PFUSE_KFUSE_KEYADDR                        0x00018008      /* RW-4R */
#define LW_PFUSE_KFUSE_KEYADDR_ADDR                          7:0      /* RWIVF */
#define LW_PFUSE_KFUSE_KEYADDR_ADDR_INIT                     0x0      /* RWI-V */
#define LW_PFUSE_KFUSE_KEYADDR_AUTOINC                     16:16      /* RWIVF */
#define LW_PFUSE_KFUSE_KEYADDR_AUTOINC_INIT                  0x1      /* RWI-V */
#define LW_PFUSE_KFUSE_KEYS                           0x0001800c      /* R--4R */
#define LW_PFUSE_KFUSE_KEYS_DATA                            31:0      /* R-XVF */
#define LW_PFUSE_OPT_HDCP_EN_OPT_HDCP_EN_YES          0x00000001      /* RW--V */

/**
* @brief Reads the specified clk registers
*
* @param[in] reg Offset of the register that needs to be read
*/
#define TEGRA_CLK_REG_RD32(reg) SYSMEM_RD32(LW_ADDRESS_MAP_CAR_BASE + reg)

/**
* @brief Writes value into the clk registers
*
* @param[in] reg Offset of the register that needs to be written
* @param[in] val Value that needs to be written into the register
*/
#define TEGRA_CLK_REG_WR32(reg, val) SYSMEM_WR32((LW_ADDRESS_MAP_CAR_BASE + reg), (val)) 

/**
* @brief Reads the specified fuse registers
*
* @param[in] reg Offset of the register that needs to be read
*/
#define TEGRA_FUSE_REG_RD32(reg) SYSMEM_RD32(LW_ADDRESS_MAP_FUSE_CONTROLLER_BASE + reg)

/**
* @brief Writes value into the fuse registers
*
* @param[in] reg Offset of the register that needs to be written
* @param[in] val Value that needs to be written into the register
*/
#define TEGRA_FUSE_REG_WR32(reg, val) SYSMEM_WR32((LW_ADDRESS_MAP_FUSE_CONTROLLER_BASE + reg), (val)) 

LW_STATUS
dispEnableFuseClocks_v04_02
(
)
{
    TEGRA_CLK_REG_WR32(CLK_RST_CONTROLLER_CLK_OUT_ENB_FUSE_0, 0x01);
	if (!TEGRA_CLK_REG_RD32(CLK_RST_CONTROLLER_CLK_OUT_ENB_FUSE_0))
	{
		dprintf("Failed to enable fuse clocks!");
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

LW_STATUS
dispDisableFuseClocks_v04_02
(
)
{
    TEGRA_CLK_REG_WR32(CLK_RST_CONTROLLER_CLK_OUT_ENB_FUSE_0, 0x00);
	if (TEGRA_CLK_REG_RD32(CLK_RST_CONTROLLER_CLK_OUT_ENB_FUSE_0))
	{
		dprintf("Failed to disable fuse clocks!");
        return LW_ERR_GENERIC;
	}
    return LW_OK;
}

LW_STATUS dispFuseCrcStatus_v04_02
(
)
{
    LwU32 data32 = TEGRA_FUSE_REG_RD32(LW_PFUSE_CRC_STATUS);
    if (!((FLD_TEST_DRF(_PFUSE_CRC, _STATUS, _CRC_STATUS_H4, _CRC_PRESENT_NO_ERRORS, data32)) ||
        (FLD_TEST_DRF(_PFUSE_CRC, _STATUS, _CRC_STATUS_H4, _CRC_PRESENT_CORRECTED_ERRORS, data32))))
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

LW_STATUS dispFuseOptHdcpEnStatus_v04_02
(
)
{
    LwU32 fuseOptHdcpElwal = TEGRA_FUSE_REG_RD32(LW_PFUSE_OPT_HDCP_EN);
    if(!FLD_TEST_DRF(_PFUSE, _OPT_HDCP_EN, _OPT_HDCP_EN, _YES, fuseOptHdcpElwal))
    {
        dprintf("\nHDCP not supported\n");
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

void dispPrintFuseCrcStatus_v04_02
(
)
{
    LwU32 data32 = TEGRA_FUSE_REG_RD32(LW_PFUSE_CRC_STATUS);
    dprintf("\n\n%-35s %-35s", "LW_PFUSE_CRC_STATUS", "_CRC_STATUS_H4");
    switch (DRF_VAL(_PFUSE, _CRC_STATUS,  _CRC_STATUS_H4, data32))
    {
        case LW_PFUSE_CRC_STATUS_CRC_STATUS_H4_CRC_NOT_PARSED_YET:
            dprintf("%-35s", "CRC_NOT_PARSED_YET");
            break;
    
        case LW_PFUSE_CRC_STATUS_CRC_STATUS_H4_CRC_NOT_PRESENT:
            dprintf("%-35s", "CRC_NOT_PRESENT");
            break;

        case LW_PFUSE_CRC_STATUS_CRC_STATUS_H4_CRC_PRESENT_NO_ERRORS:
            dprintf("%-35s", "CRC_PRESENT_NO_ERRORS");
            break;

        case LW_PFUSE_CRC_STATUS_CRC_STATUS_H4_CRC_PRESENT_CORRECTED_ERRORS:
            dprintf("%-35s", "CRC_PRESENT_CORRECTED_ERRORS");
            break;

        case LW_PFUSE_CRC_STATUS_CRC_STATUS_H4_CRC_PRESENT_UNCORRECTED_ERRORS:
            dprintf("%-35s", "CRC_PRESENT_UNCORRECTED_ERRORS");
            break;
            
        default:
            dprintf("%-35s", "Unknown State");
            break;
    }
}

LW_STATUS dispHdcpIsKfuseReady_v04_02
(
)
{
    LwU32 kfuseStateVal;
    LwU32 kfuseErrorCountVal;
	LwU32 timer = 0;

    do
    {
        if (++timer == MAX_TIMEOUT_MS)
        {
            LW_PRINTF(LEVEL_ERROR,
                      "ERROR: Timeout while waiting for KFuse to complete\n");
            return LW_ERR_TIMEOUT;
        }

        kfuseStateVal =  TEGRA_FUSE_REG_RD32(LW_PFUSE_KFUSE_STATE);
		osPerfDelay(1000); // 1ms
    }
    while (!(kfuseStateVal & DRF_NUM(_PFUSE, _KFUSE_STATE, _DONE, 1)));

    kfuseStateVal      =  TEGRA_FUSE_REG_RD32(LW_PFUSE_KFUSE_STATE);
    kfuseErrorCountVal =  TEGRA_FUSE_REG_RD32(LW_PFUSE_KFUSE_ERRCOUNT);
    if ( (! ((kfuseStateVal & DRF_NUM(_PFUSE, _KFUSE_STATE, _DONE, 1)) &&
             (kfuseStateVal & DRF_NUM(_PFUSE, _KFUSE_STATE, _CRCPASS, 1))) ) ||
         (kfuseErrorCountVal & DRF_NUM(_PFUSE, _KFUSE_ERRCOUNT, _ERR_FATAL,  0xffffffff)) )
    {
        LW_PRINTF(LEVEL_ERROR,
                  "KFUSE reported not complete or in error state: LW_KFUSE_STATE: 0x%x; "
                  "LW_KFUSE_ERRCOUNT: 0x%x\n", kfuseStateVal, kfuseErrorCountVal);
        return LW_ERR_GENERIC;
    }

    if (pDisp[indexGpu].dispFuseCrcStatus() != LW_OK)
    {
        dprintf("\nCRC errors exist\n");
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}