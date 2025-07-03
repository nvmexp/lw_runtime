/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEIFSMBPBI_H_
#define _SOEIFSMBPBI_H_

/*!
 * @file   soeifsmbpbi.h
 * @brief  SOE SMBPBI Command Queue
 *
 *         The SMBPBI unit ID will be used for sending and recieving
 *         Command Messages between driver and SMBPBI unit of SOE
 */

/*!
 * Test command/event type
 */
enum
{
    RM_SOE_SMBPBI_EVT_INTR = 0x0,
    RM_SOE_SMBPBI_CMD_ID_INIT,
    RM_SOE_SMBPBI_CMD_ID_UNLOAD,
    RM_SOE_SMBPBI_CMD_ID_SET_LINK_ERROR_INFO,
};

/*!
 * SMBPBI queue init command payload
 */
typedef struct
{
    LwU8        cmdType;
    LwU32       driverPollingPeriodUs;
    RM_FLCN_U64 dmaHandle;
} RM_SOE_SMBPBI_CMD_INIT, *PRM_SOE_SMBPBI_CMD_INIT;

/*!
 * SMBPBI queue Msgbox command payload
 */
typedef struct
{
    LwU8    cmdType;
    LwU32   msgboxCmd;
} RM_PMU_SMBPBI_CMD_MSGBOX, *PRM_PMU_SMBPBI_CMD_MSGBOX;

/*!
 * SMBPBI queue unload command payload
 */
typedef struct
{
    LwU8    cmdType;
} RM_SOE_SMBPBI_CMD_UNLOAD, *PRM_SOE_SMBPBI_CMD_UNLOAD;

/*!
 * Training error info bitmasks
 */
typedef struct
{
    LwBool      isValid;
    RM_FLCN_U64 attemptedTrainingMask0;
    RM_FLCN_U64 trainingErrorMask0;
} RM_SOE_SMBPBI_TRAINING_ERROR_INFO,
*PRM_SOE_SMBPBI_TRAINING_ERROR_INFO;

/*!
 * Runtime error link bitmask
 */
typedef struct
{
    LwBool      isValid;
    RM_FLCN_U64 mask0;
} RM_SOE_SMBPBI_RUNTIME_ERROR_INFO,
*PRM_SOE_SMBPBI_RUNTIME_ERROR_INFO;

/*!
 * SMBPBI queue set training error command payload
 */
typedef struct
{
    LwU8                              cmdType;
    RM_SOE_SMBPBI_TRAINING_ERROR_INFO trainingErrorInfo;
    RM_SOE_SMBPBI_RUNTIME_ERROR_INFO  runtimeErrorInfo;
} RM_SOE_SMBPBI_CMD_SET_LINK_ERROR_INFO,
*PRM_SOE_SMBPBI_CMD_SET_LINK_ERROR_INFO;

/*!
 * SMBPBI queue command payload
 */
typedef union
{
    LwU8                                   cmdType;
    RM_PMU_SMBPBI_CMD_MSGBOX               msgbox;
    RM_SOE_SMBPBI_CMD_INIT                 init;
    RM_SOE_SMBPBI_CMD_UNLOAD               unload;
    RM_SOE_SMBPBI_CMD_SET_LINK_ERROR_INFO  linkErrorInfo;
} RM_SOE_SMBPBI_CMD;

#endif  // _SOEIFSMBPBI_H_
