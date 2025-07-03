/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEIFCHNMGMT_H_
#define _SOEIFCHNMGMT_H_

/*!
 * @file   soeifchnmgmt.h
 * @brief  SOE Command/Message Interfaces - SOE chnmgmt
 */

/*!
 * @brief Defines for CHNMGMT commands
 */
enum
{
    RM_SOE_CHNMGMT_CMD_ID_ENGINE_RC_RECOVERY,
    RM_SOE_CHNMGMT_CMD_ID_FINISH_RC_RECOVERY,
};

/*!
 * @brief CHNMGMT engine RC command
 */
typedef struct
{
    LwU8 cmdType;  //<! Type of command, MUST be first.
    LwU8 pad[3];
} RM_SOE_CHNMGMT_CMD_ENGINE_RC_RECOVERY;

/*!
 * @brief CHNMGMT finish RC command
 */
typedef struct
{
    LwU8 cmdType;  //<! Type of command, MUST be first.
    LwU8 pad[3];
} RM_SOE_CHNMGMT_CMD_FINISH_RC_RECOVERY;

/*!
 * A simple union of all the chnmgmt commands. Use the 'cmdType' variable to
 * determine the actual type of the command.
 */
typedef union
{
    LwU8                                    cmdType;
    RM_SOE_CHNMGMT_CMD_ENGINE_RC_RECOVERY  engineRcCmd;
    RM_SOE_CHNMGMT_CMD_FINISH_RC_RECOVERY  finishRcCmd;
} RM_SOE_CHNMGMT_CMD;


/*!
 * @brief Defines for CHNMGMT messages
 */
enum
{
    RM_SOE_CHNMGMT_MSG_ID_TRIGGER_RC_RECOVERY,
    RM_SOE_CHNMGMT_MSG_ID_UNBLOCK_RC_RECOVERY,
    RM_SOE_CHNMGMT_MSG_ID_ENGINE_RC_RECOVERY,
    RM_SOE_CHNMGMT_MSG_ID_FINISH_RC_RECOVERY,
};

/*!
 * @brief CHNMGMT trigger RC recovery msg
 */
typedef struct
{
    LwU8 msgType; //<! Tag indicating the message type. MUST be first
    LwU8 pad[3];
} RM_SOE_CHNMGMT_MSG_TRIGGER_RC_RECOVERY;

/*!
 * @brief CHNMGMT unblock RC recovery msg
 */
typedef struct
{
    LwU8 msgType; //<! Tag indicating the message type. MUST be first
    LwU8 pad[3];
} RM_SOE_CHNMGMT_MSG_UNBLOCK_RC_RECOVERY;

/*!
 * @brief CHNMGMT engine RC recovery msg
 */
typedef struct
{
    LwU8 msgType; //<! Tag indicating the message type. MUST be first
    LwU8 pad[3];
} RM_SOE_CHNMGMT_MSG_ENGINE_RC_RECOVERY;

/*!
 * @brief CHNMGMT finish RC recovery msg
 */
typedef struct
{
    LwU8 msgType; //<! Tag indicating the message type. MUST be first
    LwU8 pad[3];
} RM_SOE_CHNMGMT_MSG_FINISH_RC_RECOVERY;

/*!
 * A simple union of all the Test messages. Use the 'msgType' variable to
 * determine the actual type of the message. This will be the cmdType of the
 * command for which the message is being sent.
 */
typedef union
{
    LwU8                                     msgType;
    RM_SOE_CHNMGMT_MSG_TRIGGER_RC_RECOVERY  triggerRC;
    RM_SOE_CHNMGMT_MSG_UNBLOCK_RC_RECOVERY  unblockRC;
    RM_SOE_CHNMGMT_MSG_ENGINE_RC_RECOVERY   engineRC;
    RM_SOE_CHNMGMT_MSG_FINISH_RC_RECOVERY   finishRC;
} RM_SOE_CHNMGMT_MSG;

#endif  // _SOEIFCHNMGMT_H_


