/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "FMErrorCodesInternal.h"
#include "lw_fm_agent.h"
#include "prbdec.h"
#include "g_fmInternalLib_pb.h"
#include "FmSocketMessageHdr.h"

#define FM_INTERNAL_MSG_RECV_RETRY   5 // 5 times
#define FM_INTERNAL_MSG_BUF_SIZE     1024

void fmInternalApiConnHandlerInit(void);

FMIntReturn_t connectToFMInstance(unsigned int connTimeoutMs, unsigned int msgTimeoutMs);

void disconnectFromFMInstance(void);

bool isConnectedToFMInstance(void);

uint32_t getNextRequestId(void);

void *getConnectionHandle(void);

FMIntReturn_t exchangeMsgBlocking(PRB_ENCODER *pFmlibEncodeMsg,
                                  PRB_MSG *pFmlibDecodeMsg);

