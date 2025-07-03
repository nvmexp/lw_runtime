
#pragma once

#include "DcgmFMLWLinkReqBase.h"

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

class DcgmFMLWLinkReqConn : public DcgmFMLWLinkReqBase
{
public:

    DcgmFMLWLinkReqConn(lwswitch::fmMessage *pFmMessage,
                        FMConnInterface *ctrlConnIntf,
                        DcgmFMLWLinkDrvIntf *linkDrvIntf,
                        DcgmLFMLWLinkDevRepo *linkDevRepo);

    virtual ~DcgmFMLWLinkReqConn();

    // implementation of framework for handling various request types and events.
    // Declared as pure virtual function in DcgmFMLWLinkReqBase
    virtual bool processNewMasterRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processReqTimeOut();

    // debug functions
    void dumpInfo(std::ostream *os);

private:

    virtual void sendRequestCompletion(void);

    // helper functions to ilwoke corresponding ioctl calls
    void doAddInterNodeConn(void);
    void doGetIntraNodeConns(void);

    void genAddInterNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void parseAddInterNodeConnParams(lwswitch::lwlinkRequestMsg &reqMsg);

    bool isDuplicateConnection(DcgmLWLinkConnInfo &conn);

    // holds LWLink connection information to be passed to LWLink driver or GFM
    lwlink_add_internode_conn mInterNodeAddConnParam;
    DcgmLWLinkConnList mIntraNodeGetConnList;
};
