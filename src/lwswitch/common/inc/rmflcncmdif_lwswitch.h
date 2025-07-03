/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RMFLCNCMDIF_LWSWITCH_H_
#define _RMFLCNCMDIF_LWSWITCH_H_

/*!
 * @file   rmflcncmdif_lwswitch.h
 * @brief  Top-level header-file that defines the generic command/message
 *         interfaces that may be used to communicate with the falcon (e.g. SOE)
 */

#include "flcnifcmn.h"
#include "rmsoecmdif.h"

/*!
 * Generic command struct which can be used in generic falcon code
 */
typedef union RM_FLCN_CMD
{
    RM_FLCN_CMD_GEN     cmdGen;

    RM_FLCN_CMD_SOE     cmdSoe;
} RM_FLCN_CMD, *PRM_FLCN_CMD;

/*!
 * Falcon Message structure
 */
typedef union RM_FLCN_MSG
{
    RM_FLCN_MSG_GEN     msgGen;     // Generic Message

    RM_FLCN_MSG_SOE     msgSoe;     // SOE message
} RM_FLCN_MSG, *PRM_FLCN_MSG;

#endif // _RMFLCNCMDIF_LWSWITCH_H_
