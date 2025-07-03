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
/*  Fabric Manager Link connection managment related requests                */
/*****************************************************************************/

/*
 * This class represents LWLink connection mgmt related requests in LFM context.
 * All the connection related requests are specific to a node and there is 
 * no master/slave peer LFM sync. This class mainly handles the following
 * connection managment related GPB messages from GFM and call the specified 
 * corresponding LWLink driver ioctl. The final status along with connection
 * information is then returned to GFM.
 *
 * FM_LWLINK_ADD_INTERNODE_CONN ==> IOCTL_LWLINK_ADD_INTERNODE_CONN
 * FM_LWLINK_GET_INTRANODE_CONNS ==> IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS
 *
 */

class LocalFMLWLinkReqConn : public LocalFMLWLinkReqBase
{
public:

    LocalFMLWLinkReqConn(lwswitch::fmMessage *pFmMessage,
                         FMConnInterface *ctrlConnIntf,
                         LocalFMLWLinkDrvIntf *linkDrvIntf,
                         LocalFMLWLinkDevRepo *linkDevRepo);

    virtual ~LocalFMLWLinkReqConn();

    // implementation of framework for handling various request types and events.
    // Declared as pure virtual function in LocalFMLWLinkReqBase
    virtual bool processNewMasterRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processReqTimeOut();

    // debug functions
    void dumpInfo(std::ostream *os);

private:

    virtual void sendRequestCompletion(void);

    // helper functions to ilwoke corresponding ioctl calls
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void doAddInterNodeConn(void);
#endif
    void doGetIntraNodeConns(void);

    void genGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void genAddInterNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void parseAddInterNodeConnParams(lwswitch::lwlinkRequestMsg &reqMsg);
#endif

    bool isDuplicateConnection(FMLWLinkConnInfo &conn);

    // holds LWLink connection information to be passed to LWLink driver or GFM
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    lwlink_add_internode_conn mInterNodeAddConnParam;
#endif
    FMLWLinkConnList mIntraNodeGetConnList;
};
