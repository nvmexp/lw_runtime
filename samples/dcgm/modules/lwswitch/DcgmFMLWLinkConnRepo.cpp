
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <sstream>
#include <string.h>

#include "logging.h"
#include "DcgmFMLWLinkState.h"
#include "DcgmFMLWLinkDeviceRepo.h"
#include "DcgmFMLWLinkConnRepo.h"

DcgmFMLWLinkDetailedConnInfo::DcgmFMLWLinkDetailedConnInfo(DcgmLWLinkEndPointInfo masterEnd,
                                                           DcgmLWLinkEndPointInfo slaveEnd)
{
    mMasterEnd = masterEnd;
    mSlaveEnd = slaveEnd;
    memset( &mMasterLinkState, -1, sizeof(mMasterLinkState) );
    memset( &mSlaveLinkState, -1, sizeof(mSlaveLinkState) );
}

DcgmFMLWLinkDetailedConnInfo::~DcgmFMLWLinkDetailedConnInfo()
{
    // nothing specific
}

void
DcgmFMLWLinkDetailedConnInfo::setLinkStateInfo(DcgmLWLinkStateInfo masterInfo,
                                               DcgmLWLinkStateInfo slaveInfo)
{
    mMasterLinkState = masterInfo;
    mSlaveLinkState = slaveInfo;
}

bool
DcgmFMLWLinkDetailedConnInfo::isConnTrainedToActive(void)
{
    if ( linkStateActive() &&
         txSubLinkStateActive() &&
         rxSubLinkStateActive() ) {
        return true;
    }
    return false;
}

void
DcgmFMLWLinkDetailedConnInfo::dumpConnInfo(std::ostream *os, DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    DcgmFMLWLinkDevInfo masterDevInfo;
    DcgmFMLWLinkDevInfo slaveDevInfo;

    linkDevRepo.getDeviceInfo( mMasterEnd.nodeId, mMasterEnd.gpuOrSwitchId, masterDevInfo );
    linkDevRepo.getDeviceInfo( mSlaveEnd.nodeId, mSlaveEnd.gpuOrSwitchId, slaveDevInfo );

    // Note: uncomment NodeID when multi-node system is released
    //*os << "NodeID: " << mMasterEnd.nodeId;
    *os << "DeviceName:" << masterDevInfo.getDeviceName();
    *os << " LinkIndex:" << mMasterEnd.linkIndex;

    *os << " <=======> ";
    // Note: uncomment NodeID when multi-node system is released
    //*os << "NodeID:" << mSlaveEnd.nodeId;
    *os << "DeviceName:" << slaveDevInfo.getDeviceName();
    *os << " LinkIndex:" << mSlaveEnd.linkIndex << std::endl;
}

void
DcgmFMLWLinkDetailedConnInfo::dumpConnAndStateInfo(std::ostream *os, DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    DcgmFMLWLinkDevInfo masterDevInfo;
    DcgmFMLWLinkDevInfo slaveDevInfo;

    linkDevRepo.getDeviceInfo( mMasterEnd.nodeId, mMasterEnd.gpuOrSwitchId, masterDevInfo );
    linkDevRepo.getDeviceInfo( mSlaveEnd.nodeId, mSlaveEnd.gpuOrSwitchId, slaveDevInfo );

    // Note: uncomment NodeID when multi-node system is released
    //*os << "NodeID: " << mMasterEnd.nodeId;
    *os << "DeviceName:" << masterDevInfo.getDeviceName();
    *os << " LinkIndex:" << mMasterEnd.linkIndex;

    *os << " <=======> ";
    // Note: uncomment NodeID when multi-node system is released
    //*os << NodeID:" << mSlaveEnd.nodeId;
    *os << "DeviceName:" << slaveDevInfo.getDeviceName();
    *os << " LinkIndex:" << mSlaveEnd.linkIndex << std::endl;

    // dump link state
    *os << "Main_State:" << DcgmFMLWLinkState::getMainLinkState( mMasterLinkState.linkMode );
    *os << " Tx_State:" << DcgmFMLWLinkState::getTxSubLinkState( mMasterLinkState.txSubLinkMode );
    *os << " Rx_State: " << DcgmFMLWLinkState::getRxSubLinkState( mMasterLinkState.rxSubLinkMode );
    *os << " <=======> ";
    *os << "Main_State:" << DcgmFMLWLinkState::getMainLinkState( mSlaveLinkState.linkMode );
    *os << " Tx_State:" << DcgmFMLWLinkState::getTxSubLinkState( mSlaveLinkState.txSubLinkMode );
    *os << " Rx_State: " << DcgmFMLWLinkState::getRxSubLinkState( mSlaveLinkState.rxSubLinkMode ) << std::endl << std::endl;
}

bool
DcgmFMLWLinkDetailedConnInfo::linkStateActive(void)
{
    if ( (mMasterLinkState.linkMode == lwlink_link_mode_active) &&
         (mSlaveLinkState.linkMode == lwlink_link_mode_active) ) {
        return true;
    }

    return false;
}

bool
DcgmFMLWLinkDetailedConnInfo::txSubLinkStateActive(void)
{
    if ( ((mMasterLinkState.txSubLinkMode == lwlink_tx_sublink_mode_hs) ||
          (mMasterLinkState.txSubLinkMode == lwlink_tx_sublink_mode_single_lane)) &&
         ((mSlaveLinkState.txSubLinkMode == lwlink_tx_sublink_mode_hs) ||
          (mSlaveLinkState.txSubLinkMode == lwlink_tx_sublink_mode_single_lane)) ) {
        return true;
    }

    return false;
}

bool
DcgmFMLWLinkDetailedConnInfo::rxSubLinkStateActive(void)
{
    if ( ((mMasterLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_hs) ||
          (mMasterLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_single_lane)) &&
         ((mSlaveLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_hs) ||
          (mSlaveLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_single_lane)) ) {
        return true;
    }

    return false;
}

DcgmFMLWLinkConnRepo::DcgmFMLWLinkConnRepo()
{
    mIntraConnMap.clear();
    mInterConnMap.clear();
}

DcgmFMLWLinkConnRepo::~DcgmFMLWLinkConnRepo()
{
    // delete all the intra-node connections
    LWLinkIntraConnMap::iterator it = mIntraConnMap.begin();
    while ( it != mIntraConnMap.end() ) {
        DcgmLWLinkDetailedConnList &connList = it->second;
        DcgmLWLinkDetailedConnList::iterator jit = connList.begin();
        while ( jit != connList.end() ) {
            DcgmFMLWLinkDetailedConnInfo *conn = (*jit);
            connList.erase( jit++ );
            delete conn;
        }
        mIntraConnMap.erase( it++ );
    }

    // delete all the inter-node connections
    LWLinkInterNodeConns::iterator jit =  mInterConnMap.begin();
    while ( jit != mInterConnMap.end() ) {
        DcgmFMLWLinkDetailedConnInfo *conn = (*jit);
        mInterConnMap.erase( jit++ );
        delete conn;
    }
}

bool
DcgmFMLWLinkConnRepo::addIntraConnections(uint32 nodeId, DcgmLWLinkConnList connList)
{
    DcgmLWLinkDetailedConnList detailedConnList;
    DcgmFMLWLinkDetailedConnInfo* detailedConnInfo;

    // bail if we already have connection information for the specified node
    LWLinkIntraConnMap::iterator it = mIntraConnMap.find( nodeId );
    if ( it != mIntraConnMap.end() ) {
        return false;
    }

    DcgmLWLinkConnList::iterator jit;
    for ( jit = connList.begin(); jit != connList.end(); jit++ ) {
        DcgmLWLinkConnInfo connInfo = (*jit);
        // create our dcgm detailed connection information
        detailedConnInfo = new DcgmFMLWLinkDetailedConnInfo( connInfo.masterEnd, connInfo.slaveEnd );
        detailedConnList.push_back( detailedConnInfo );
    }

    // add node's device information locally
    mIntraConnMap.insert( std::make_pair(nodeId, detailedConnList) );
    return true;
}

bool
DcgmFMLWLinkConnRepo::addInterConnections(DcgmLWLinkConnInfo connInfo)
{
    DcgmFMLWLinkDetailedConnInfo *detailedConnInfo;
    detailedConnInfo = new DcgmFMLWLinkDetailedConnInfo( connInfo.masterEnd, connInfo.slaveEnd );

    // before inserting, check for existing connection
    LWLinkInterNodeConns::iterator it;
    for ( it = mInterConnMap.begin(); it != mInterConnMap.end(); it++ ) {
        DcgmFMLWLinkDetailedConnInfo *tempConnInfo = (*it);
        if ( tempConnInfo == detailedConnInfo ) {
            // found the connection, with same endPoint info
            delete detailedConnInfo;
            return false;
        }
    }

    // no such connection exists for us, add it
    mInterConnMap.push_back( detailedConnInfo );
    return true;
}

DcgmFMLWLinkDetailedConnInfo*
DcgmFMLWLinkConnRepo::getConnectionInfo(DcgmLWLinkEndPointInfo &endInfo)
{
    // first check in intra connection list,
    LWLinkIntraConnMap::iterator it = mIntraConnMap.find( endInfo.nodeId );
    if ( it != mIntraConnMap.end() ) {
        DcgmLWLinkDetailedConnList::iterator jit;
        DcgmLWLinkDetailedConnList connList = it->second;
        for ( jit = connList.begin(); jit != connList.end(); jit++ ) {
            DcgmFMLWLinkDetailedConnInfo *tempConnInfo = (*jit);
            // check whether source or dest endPoint of the connection match to
            // the given endPoint for look-up
            if  ( (tempConnInfo->getMasterEndPointInfo() == endInfo) ||
                  (tempConnInfo->getSlaveEndPointInfo() == endInfo) ) {
                // found the connection, return it
                return tempConnInfo;
            }
        }
    }

    // check in inter connection list as well.
    LWLinkInterNodeConns::iterator jit;
    for ( jit = mInterConnMap.begin(); jit != mInterConnMap.end(); jit++ ) {
        DcgmFMLWLinkDetailedConnInfo *tempConnInfo = (*jit);
        // check whether source or dest endPoint of the connection match to
        // the given endPoint for look-up
        if  ( (tempConnInfo->getMasterEndPointInfo() == endInfo) ||
              (tempConnInfo->getSlaveEndPointInfo() == endInfo) ) {
            // found the connection, return it
            return tempConnInfo;
        }
    }

    // not found in both intra and inter connection list
    return NULL;
}

void
DcgmFMLWLinkConnRepo::dumpAllConnAndStateInfo(DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    PRINT_INFO( "", "Dumping all Intra-Node connections" );

    LWLinkIntraConnMap::iterator it;
    for ( it = mIntraConnMap.begin(); it != mIntraConnMap.end(); it++ ) {
        DcgmLWLinkDetailedConnList connList = it->second;
        DcgmLWLinkDetailedConnList::iterator jit;
        PRINT_INFO( "%d", "Intra-Node connections for Node Index:%d",  it->first );
        PRINT_INFO( "%ld", "Number of connections:%ld", connList.size() );
        // dump each connection information
        for (jit = connList.begin(); jit != connList.end(); jit++ ) {
            std::stringstream outStr;
            DcgmFMLWLinkDetailedConnInfo *connInfo = (*jit);
            connInfo->dumpConnAndStateInfo( &outStr, linkDevRepo );
            PRINT_INFO( "%s", "\n%s", outStr.str().c_str() );
        }
    }

    PRINT_INFO("", "Dumping all Inter-Node connections");
    LWLinkInterNodeConns::iterator jit;
    for (jit = mInterConnMap.begin(); jit != mInterConnMap.end(); jit++ ) {
        std::stringstream outStr;
        DcgmFMLWLinkDetailedConnInfo *connInfo = (*jit);
        connInfo->dumpConnAndStateInfo( &outStr, linkDevRepo );
        PRINT_INFO( "%s", "\n%s", outStr.str().c_str() );
    }
}
