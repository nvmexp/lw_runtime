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

#include "LocalFMLWLinkReqBase.h"

/*****************************************************************************/
/*  Fabric Manager Link discovery related requests                           */
/*****************************************************************************/

/*
 * This class represents LWLink device discovery related requests in LFM context.
 * All the device recovery related requests are specific to a node and there is 
 * no master/slave peer LFM sync. This class mainly handles the following
 * device discovery related GPB messages from GFM and call the specified 
 * corresponding LWLink driver ioctl. The final status along with discovery 
 * token information is then returned to GFM.
 *
 * FM_LWLINK_DISCOVER_INTRANODE_CONNS ==> IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS
 * FM_LWLINK_WRITE_DISCOVERY_TOKENS   ==> IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS
 * FM_LWLINK_READ_DISCOVERY_TOKENS    ==> IOCTL_LWLINK_READ_DISCOVERY_TOKENS
 *
 */

 class LocalFMLWLinkReqDiscovery : public LocalFMLWLinkReqBase
{
public:

    LocalFMLWLinkReqDiscovery(lwswitch::fmMessage *pFmMessage,
                              FMConnInterface *ctrlConnIntf,
                              LocalFMLWLinkDrvIntf *linkDrvIntf,
                              LocalFMLWLinkDevRepo *linkDevRepo);

    virtual ~LocalFMLWLinkReqDiscovery();

    // implementation of framework for handling various request types and events.
    // Declared as pure virtual function in LocalFMLWLinkReqBase
    virtual bool processNewMasterRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processReqTimeOut();

    // debug functions
    void dumpInfo(std::ostream *os);

private:
    virtual void sendRequestCompletion(void);

    // helper functions to ilwoke corresponding ioctl calls
    void doDiscoverIntraNodeConns(void);
    void doWriteDiscoveryToken(void);
    void doReadDiscoveryToken(void);
    void doReadSid(void);

    // helper functions to generate response GPB message to GFM
    void genDiscoverIntraNodeConnsResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genWriteDiscoveryTokenResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genReadDiscoveryTokenResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genReadSidResp(lwswitch::lwlinkResponseMsg *rspMsg);

    // holds the write discovery token result from LWLink Driver
    FMLWLinkDiscoveryTokenList mWriteTokenList;
    // holds the read discovery token result from LWLink Driver    
    FMLWLinkDiscoveryTokenList mReadTokenList;
    // holds the read discovery sid result from LWLink Driver    
    FMLWLinkSidList mReadSidList;
};
