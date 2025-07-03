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

#include "FMErrorCodesInternal.h"
#include "FmClientConnection.h"
#include "lw_fm_agent.h"
#include "FmConnection.h"
#include "fmlib.pb.h"

class fmLibClientConnHandler
{
public:

    // initializes client handler
    fmLibClientConnHandler();
    virtual ~fmLibClientConnHandler();

    // establish connection to the running FM instance
    FMIntReturn_t openConnToRunningFMInstance(char* addressInfo, fmHandle_t* pConnHandle,
                                              unsigned int timeoutMs, bool addressIsUnixSocket);

    // close connection with the running FM instance
    FMIntReturn_t closeConnToRunningFMInstance(fmHandle_t connHandle);

    // exchange protobuf encoded commands with the FM instance in req/resp (blocking) manner
    FMIntReturn_t exchangeMsgBlocking(fmHandle_t connHandle, fmlib::Msg *mpFmlibEncodeMsg, fmlib::Msg *mpFmlibDecodeMsg, 
                                      fmlib::Command **pRecvdCmd, unsigned int timeout);

private:

    FMIntReturn_t tryConnectingToFMInstance(char* addressIdentifier,
                                            unsigned int portNumber,
                                            fmHandle_t* connHandle,
                                            bool addressIsUnixSocket,
                                            int connectionTimeoutMs);

    FmClientListener* mClientBase;
    FmConnectionHandler* mConnectionHandler;
};
