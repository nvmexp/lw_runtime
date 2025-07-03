/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEBIF_H_
#define _SOEBIF_H_

/*!
 * @file   soebif.h
 * @brief  SOE BIF Command Queue
 *
 *         The BIF unit ID will be used for sending and recieving
 *         Command Messages between driver and BIF unit of SOE
 */

/*!
 * Commands offered by the SOE Bus Interface.
 */
enum
{
    /*!
     * Update the UPHY EOM(Eye Opening Measurement) parameters.
     */
    RM_SOE_BIF_CMD_UPDATE_EOM,

    /*!
     * This command sends UPHY register's address and lane from the client
     * to the SOE and gets the register value.
     */
    RM_SOE_BIF_CMD_GET_UPHY_DLN_CFG_SPACE,

    /*!
     * Set PCIE link speed
     */
    RM_SOE_BIF_CMD_SET_PCIE_LINK_SPEED,

    /*!
     * Get UPHY EOM(Eye Opening Measurement) status.
     */
    RM_SOE_BIF_CMD_GET_EOM_STATUS,

    /*!
     * Signal Lane Margining
     */
    RM_SOE_BIF_CMD_SIGNAL_LANE_MARGINING,

    /*!
     * Handle Margining interrupt
     */
    RM_SOE_BIF_CMD_SERVICE_MARGINING_INTERRUPTS,
};

/*!
 * BIF queue command payload
 */
typedef struct
{
    LwU8 cmdType;
    LwU8 mode;
    LwU8 nerrs;
    LwU8 nblks;
    LwU8 berEyeSel;
} RM_SOE_BIF_CMD_EOM;

typedef struct
{
    LwU8 cmdType;
    LwU8 mode;
    LwU8 nerrs;
    LwU8 nblks;
    LwU8 berEyeSel;
    LwU32 laneMask;
    RM_FLCN_U64 dmaHandle;
} RM_SOE_BIF_CMD_EOM_STATUS;

typedef struct
{
    LwU8  cmdType;
    LwU32 regAddress;
    LwU32 laneSelectMask;
} RM_SOE_BIF_CMD_UPHY_DLN_CFG_SPACE;

typedef struct
{
    LwU8 cmdType;
    LwU8 laneNum;
} RM_SOE_BIF_CMD_LANE_MARGINING;

#define RM_SOE_BIF_LINK_SPEED_ILWALID      (0x00)
#define RM_SOE_BIF_LINK_SPEED_GEN1PCIE     (0x01)
#define RM_SOE_BIF_LINK_SPEED_GEN2PCIE     (0x02)
#define RM_SOE_BIF_LINK_SPEED_GEN3PCIE     (0x03)
#define RM_SOE_BIF_LINK_SPEED_GEN4PCIE     (0x04)
#define RM_SOE_BIF_LINK_SPEED_GEN5PCIE     (0x05)

#define RM_SOE_BIF_LINK_WIDTH_ILWALID  (0x00)
#define RM_SOE_BIF_LINK_WIDTH_X1       (0x01)
#define RM_SOE_BIF_LINK_WIDTH_X2       (0x02)
#define RM_SOE_BIF_LINK_WIDTH_X4       (0x03)
#define RM_SOE_BIF_LINK_WIDTH_X8       (0x04)
#define RM_SOE_BIF_LINK_WIDTH_X16      (0x05)

// Maximum time to wait for LTSSM to go idle, in ns
#define BIF_LTSSM_IDLE_TIMEOUT_NS          (200 * SOE_INTERVAL_1USEC_IN_NS)
// Maximum time to wait for LTSSM to declare link ready, in ns
#define BIF_LTSSM_LINK_READY_TIMEOUT_NS    (20 * SOE_INTERVAL_1MSEC_IN_NS)
// Maximum time to keep trying to change link speed, in ns
#define BIF_LINK_CHANGE_TIMEOUT_NS         (10 * SOE_INTERVAL_5MSEC_IN_NS)
// Maximum PCIe lanes supported per link is 16 as of PCIe spec 4.0r1.0
#define BIF_MAX_PCIE_LANES   16U

typedef struct
{
    LwU8 cmdType;
    LwU32 linkSpeed;
} RM_SOE_BIF_CMD_PCIE_LINK_SPEED;

typedef union
{
    LwU8 cmdType;
    RM_SOE_BIF_CMD_EOM eomctl;
    RM_SOE_BIF_CMD_UPHY_DLN_CFG_SPACE cfgctl;
    RM_SOE_BIF_CMD_PCIE_LINK_SPEED speedctl;
    RM_SOE_BIF_CMD_EOM_STATUS eomStatus;
    RM_SOE_BIF_CMD_LANE_MARGINING laneMargining;
} RM_SOE_BIF_CMD;

#endif  // _SOEBIF_H_
