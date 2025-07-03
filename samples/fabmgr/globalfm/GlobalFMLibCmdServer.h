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

#include "fabricmanager.pb.h"
#include "FmServerConnection.h"
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"
#include "FMCommCtrl.h"
#include "FMConnectionBase.h"
#include "GlobalFabricManager.h"
#include "fmlib.pb.h"
#include "GlobalFmApiHandler.h"

class GlobalFMLibCmdServer: public FmSocket
{

public:

    GlobalFMLibCmdServer( GlobalFabricManager *gfm, GlobalFmApiHandler *pApiHandler,
                          int portNumber, char *sockpath, int isTCP );
    virtual ~GlobalFMLibCmdServer();

    /*****************************************************************************
    This method handles the message received on the socket.
    *****************************************************************************/
   virtual int OnRequest(fm_request_id_t requestId, FmServerConnection *pConnection);

protected:

   void processMessage(fmlib::Command *pCmd);
   int sendReplyToClient(fmlib::Msg *fmlibProtoMsg, FmServerConnection *pConnection, 
                         fm_request_id_t requestId, unsigned int msgType);
   GlobalFabricManager *mpGfm;
   GlobalFmApiHandler  *mpApiHandler;
};

