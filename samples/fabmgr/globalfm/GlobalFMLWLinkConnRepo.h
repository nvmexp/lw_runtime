/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include <iostream>
#include <time.h>
#include <map>
#include <string.h>
#include <vector>
#include <iomanip>

#include "FMCommonTypes.h"
#include "FMLWLinkTypes.h"
#include "GlobalFmFabricParser.h"

/*****************************************************************************/
/* Maintain the list of inter and intra node LWLink connections in GFM       */
/*****************************************************************************/
class GlobalFMLWLinkDevRepo;

typedef std::map<std::vector<int>, std::vector<int>> ConnectionPortInfoMap;

class FMLWLinkDetailedConnInfo
{
public:

    FMLWLinkDetailedConnInfo(FMLWLinkEndPointInfo masterEnd,
                             FMLWLinkEndPointInfo slaveEnd);
    ~FMLWLinkDetailedConnInfo();

    FMLWLinkEndPointInfo getMasterEndPointInfo() { return mMasterEnd; }
    FMLWLinkEndPointInfo getSlaveEndPointInfo() { return mSlaveEnd; }    
    FMLWLinkStateInfo getMasterLinkStateInfo() { return mMasterLinkState; }
    FMLWLinkStateInfo getSlaveLinkStateInfo() { return mSlaveLinkState; }    

    uint8 getMasterLinkEomLow() { return mMasterLinkQualityInfo.eomLow; }
    uint8 getSlaveLinkEomLow() { return mSlaveLinkLinkQualityInfo.eomLow; }

    void setLinkStateInfo(FMLWLinkStateInfo masterInfo,
                          FMLWLinkStateInfo slaveInfo);

    void setLinkStateInfoMaster(FMLWLinkStateInfo masterInfo);

    void setLinkStateInfoSlave(FMLWLinkStateInfo slaveInfo);

    void setMasterLinkQualityInfo(FMLWLinkQualityInfo &linkQualityInfo);

    void setSlaveLinkQualityInfo(FMLWLinkQualityInfo &linkQualityInfo);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void setMasterLinkFomValues(FMLWLinkFomValues &fomValues) { mMasterLinkFomValues = fomValues; }

    void setSlaveLinkFomValues(FMLWLinkFomValues &fomValues) { mSlaveLinkFomValues = fomValues; }

    void setMasterLinkGradingValues(FMLWLinkGradingValues &gradingValues) { mMasterLinkGradingValues = gradingValues; }

    void setSlaveLinkGradingValues(FMLWLinkGradingValues &gradingValues) { mSlaveLinkGradingValues = gradingValues; }
#endif

    bool isConnTrainedToActive(void);
    bool isConnInContainState(void);

    void dumpConnInfo(std::ostream *os, GlobalFabricManager *gfm, GlobalFMLWLinkDevRepo &linkDevRepo);
    void dumpConnAndStateInfo(std::ostream *os, GlobalFabricManager *gfm, GlobalFMLWLinkDevRepo &linkDevRepo);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void printFomAndGradingValues(GlobalFabricManager *gfm, GlobalFMLWLinkDevRepo &linkDevRepo);
    void dumpDiscoveredConnectionPortInfo(GlobalFabricManager *gfm,
                                          ConnectionPortInfoMap &discoveredConnPortInfoMap,
                                          GlobalFMLWLinkDevRepo &linkDevRepo);
#endif

    bool operator==(const FMLWLinkDetailedConnInfo& rhs)
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
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    bool connectionPortInfoExists(ConnectionPortInfoMap &discoveredConnPortInfoMap,
                                  std::vector<int> portInfo, std::vector<int> farPortInfo);

    bool checkConnectionInconsistency(ConnectionPortInfoMap &discoveredConnPortInfoMap,
                                      std::vector<int> portInfo, std::vector<int> farPortInfo);
#endif
    FMLWLinkEndPointInfo mMasterEnd;
    FMLWLinkEndPointInfo mSlaveEnd;
    FMLWLinkStateInfo mMasterLinkState;
    FMLWLinkStateInfo mSlaveLinkState;
    FMLWLinkQualityInfo mMasterLinkQualityInfo;
    FMLWLinkQualityInfo mSlaveLinkLinkQualityInfo;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FMLWLinkFomValues mMasterLinkFomValues;
    FMLWLinkFomValues mSlaveLinkFomValues;
    FMLWLinkGradingValues mMasterLinkGradingValues;
    FMLWLinkGradingValues mSlaveLinkGradingValues;
#endif
    enum ConnectionPortInfo {
        NODEID = 1,
        SLOTNO = 2,
        PORTNUM = 3
    };
};

typedef std::list<FMLWLinkDetailedConnInfo*> FMLWLinkDetailedConnInfoList;
// Intra-node connections, maintained as a list of connections per node id
typedef std::map <uint32, FMLWLinkDetailedConnInfoList> LWLinkIntraConnMap;
//Inter node connections, maintained as a list of connections
typedef FMLWLinkDetailedConnInfoList LWLinkInterNodeConns;

class GlobalFMLWLinkConnRepo
{
public:
    GlobalFMLWLinkConnRepo();
    ~GlobalFMLWLinkConnRepo();

    // interface to populate the connections
    bool addIntraConnections(uint32 nodeId, FMLWLinkConnList &connList);
    bool addInterConnections(FMLWLinkConnInfo connInfo);
    bool mergeIntraConnections(uint32 nodeId, FMLWLinkDetailedConnInfoList connList);

    // interface to query all the connections
    LWLinkIntraConnMap& getIntraConnections(void) { return mIntraConnMap; }
    LWLinkInterNodeConns& getInterConnections(void) { return mInterConnMap; }

    // interface to get a specific connection
    FMLWLinkDetailedConnInfo* getConnectionInfo(FMLWLinkEndPointInfo &endInfo);

    void dumpAllConnAndStateInfo(GlobalFabricManager *gfm, GlobalFMLWLinkDevRepo &linkDevRepo);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void dumpAllDiscoveredConnectionPortInfo(GlobalFabricManager *gfm,
                                             GlobalFMLWLinkDevRepo &linkDevRepo);
#endif
    void clearConns();

private:
    LWLinkIntraConnMap mIntraConnMap;
    LWLinkInterNodeConns mInterConnMap;
};
