
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEIFTEST_H_
#define _SOEIFTEST_H_

/*!
 * @file   soeiftest.h
 * @brief  SOE Command/Message Interfaces - SOE test
 *         The Test unit ID will be used for dummy test jobs.
 *         Lwrrently we have just the RT timer test job. This will be
 *         run only if the SOE is is debug mode.
 */

/*!
 * Test command type
 */
enum
{
    RM_SOE_TEST_CMD_ID_RTTIMER_TEST          = 0x0,
};


/*!
 * @brief Test command to test RT timer
 */
typedef struct
{
    LwU8                cmdType;     //<! Type of command, MUST be first.
    LwBool              bCheckTime;  //<! Verify that the RT timer is accurate
    LwU8                pad[2];
    LwU32               count;       //<! Count value to program the RT Timer
} RM_SOE_TEST_CMD_RTTIMER_TEST;

enum
{
    FAKEIDLE_UNMASK_INTR  = 0x1,
    FAKEIDLE_MASK_INTR    = 0x2,
    FAKEIDLE_PROGRAM_IDLE = 0x3,
    FAKEIDLE_PROGRAM_BUSY = 0x4,
    FAKEIDLE_CTXSW_DETECT = 0x5,
    FAKEIDLE_MTHD_DETECT  = 0x6,
    FAKEIDLE_COMPLETE     = 0x7,
};

/*!
 * A simple union of all the test commands. Use the 'cmdType' variable to
 * determine the actual type of the command.
 */
typedef union
{
    LwU8                                   cmdType;
    RM_SOE_TEST_CMD_RTTIMER_TEST           rttimer;
} RM_SOE_TEST_CMD;

/*!
 * List of statuses that the Test task/job can send to the RM.
 *
 * RM_SOE_TEST_MSG_STATUS_OK:               Command was successful
 * RM_SOE_TEST_MSG_STATUS_FAIL:             Command failed
 */
#define RM_SOE_TEST_MSG_STATUS_OK              (0x0)
#define RM_SOE_TEST_MSG_STATUS_FAIL            (0x1)

/*!
 * @brief Test msg to test RT timer
 */
typedef struct
{
    LwU8                   msgType; //<! Type of message, MUST be first.
    LwU8                   status;
    LwU8                   pad[2];
    LwU32                  oneShotNs;
    LwU32                  continuousNs;
} RM_SOE_TEST_MSG_RTTIMER_TEST;


/*!
 * A simple union of all the Test messages. Use the 'msgType' variable to
 * determine the actual type of the message. This will be the cmdType of the
 * command for which the message is being sent.
 */
typedef union
{
    LwU8                                  msgType;
    RM_SOE_TEST_MSG_RTTIMER_TEST          rttimer;

} RM_SOE_TEST_MSG;

#endif  // _SOEIFTEST_H_
