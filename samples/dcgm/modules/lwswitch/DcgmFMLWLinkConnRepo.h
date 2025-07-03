#pragma once

#include <iostream>
#include <time.h>
#include <map>

#include "DcgmFMCommon.h"
#include "DcgmFMLWLinkTypes.h"

/*****************************************************************************/
/* Maintain the list of inter and intra node LWLink connections in GFM       */
/*****************************************************************************/
class DcgmGFMLWLinkDevRepo;

class DcgmFMLWLinkDetailedConnInfo
{
public:

    DcgmFMLWLinkDetailedConnInfo(DcgmLWLinkEndPointInfo masterEnd,
                                 DcgmLWLinkEndPointInfo slaveEnd);
    ~DcgmFMLWLinkDetailedConnInfo();

    DcgmLWLinkEndPointInfo getMasterEndPointInfo() { return mMasterEnd; }
    DcgmLWLinkEndPointInfo getSlaveEndPointInfo() { return mSlaveEnd; }    
    DcgmLWLinkStateInfo getMasterLinkStateInfo() { return mMasterLinkState; }
    DcgmLWLinkStateInfo getSlaveLinkStateInfo() { return mSlaveLinkState; }    

    void setLinkStateInfo(DcgmLWLinkStateInfo masterInfo,
                          DcgmLWLinkStateInfo slaveInfo);

    bool isConnTrainedToActive(void);

    void dumpConnInfo(std::ostream *os, DcgmGFMLWLinkDevRepo &linkDevRepo);
    void dumpConnAndStateInfo(std::ostream *os, DcgmGFMLWLinkDevRepo &linkDevRepo);

    bool operator==(const DcgmFMLWLinkDetailedConnInfo& rhs)
    {
        if ( (mMasterEnd == rhs.mMasterEnd) && (mSlaveEnd == rhs.mSlaveEnd) ) {
            return true;
        }

        // do reverse comparison as well
        if ( (mMasterEnd == rhs.mSlaveEnd) && (mSlaveEnd == rhs.mMasterEnd) ) {
            return true;
        }

        return false;
    }    

private:
    bool linkStateActive(void);
    bool txSubLinkStateActive(void);
    bool rxSubLinkStateActive(void);

    DcgmLWLinkEndPointInfo mMasterEnd;
    DcgmLWLinkEndPointInfo mSlaveEnd;
    DcgmLWLinkStateInfo mMasterLinkState;
    DcgmLWLinkStateInfo mSlaveLinkState;
};

typedef std::list<DcgmFMLWLinkDetailedConnInfo*> DcgmLWLinkDetailedConnList;
// Intra-node connections, maintained as a list of connections per node id
typedef std::map <uint32, DcgmLWLinkDetailedConnList> LWLinkIntraConnMap;
//Inter node connections, maintained as a list of connections
typedef DcgmLWLinkDetailedConnList LWLinkInterNodeConns;

class DcgmFMLWLinkConnRepo
{
public:
    DcgmFMLWLinkConnRepo();
    ~DcgmFMLWLinkConnRepo();

    // interface to populate the connections
    bool addIntraConnections(uint32 nodeId, DcgmLWLinkConnList connList);
    bool addInterConnections(DcgmLWLinkConnInfo connInfo);

    // interface to query all the connections
    LWLinkIntraConnMap& getIntraConnections(void) { return mIntraConnMap; }
    LWLinkInterNodeConns& getInterConnections(void) { return mInterConnMap; }

    // interface to get a specific connection
    DcgmFMLWLinkDetailedConnInfo* getConnectionInfo(DcgmLWLinkEndPointInfo &endInfo);

    void dumpAllConnAndStateInfo(DcgmGFMLWLinkDevRepo &linkDevRepo);

private:
    LWLinkIntraConnMap mIntraConnMap;
    LWLinkInterNodeConns mInterConnMap;
};
