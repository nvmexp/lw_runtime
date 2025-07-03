/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
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
#include <fstream>
#include "fabricmanager.pb.h"
#include "FMCommonTypes.h"
#include "FMLWLinkTypes.h"
#include "LocalFMLWLinkDrvIntf.h"

class GlobalFabricManager;
/*****************************************************************************/
/*  Fabric Manager LWLink Device list/repository                             */
/*****************************************************************************/

/*
 * FMLWLinkDevInfo:
 *  This class holds detailed information about an LWLink device. It also creates
 *  a unique device ID for addressing each device externally. This device ID is
 *  lwrrently unique only within the node.
 *
 * LocalFMLWLinkDevRepo
 *  Holds the list of LWLink devices in the local node which is queried from the
 *  LWLink driver. The LWLink GPB message handlers will query the list of LWLink
 *  devices to repeat the LWLink driver IOCLTs for all the devices. (as
 *  few IOCTLs are per LWLink device)
 *
 * GlobalFMLWLinkDevRepo
 *  Holds the list of LWLink devices received from all the nodes in GFM context.
 *  This device information later used to uniquely address each LWLink device
 *  for GFM <-> LFM communication. The device information will be used for 
 *  logging and topology verification as well.
 */

class FMLWLinkDevInfo
{
public:

    // the default constructor is for assignment
    FMLWLinkDevInfo();

    // this constructor will be used by LFM to create device information
    FMLWLinkDevInfo(uint32 nodeId, lwlink_detailed_dev_info devInfo);

    // this constrlwtor will be used by GFM to create device information
    // received from other peer LFMs
    FMLWLinkDevInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoMsg &devInfoMsg);

    ~FMLWLinkDevInfo();

    // getters for all the member/device information
    uint64 getDeviceId() { return mDeviceId; }
    uint16 getNumLinks() { return mNumLinks; }
    uint64 getDeviceType() { return mDevType; }
    std::string getDeviceName() { return mDeviceName; }
    uint8* getDeviceUuid() { return mDevUuid; }
    lwlink_pci_dev_info getDevicePCIInfo() { return mPciInfo ;}
    char* getDevicePciBusId() { return mPciBusId; }
    uint64 getEnabledLinkMask(void) { return mEnabledLinkMask; }
    void getInitFailedLinksIndex(std::list<uint32> &initFailedLinks);
    void getNonActiveLinksIndex(std::list<uint32> &nonActiveLinks, std::list<uint32> &missingConnLinkIndex);
    void getFailedNonContainLinksIndex(std::list<uint32> &failedLinks, std::list<uint32> &missingConnLinkIndex);
    bool isAllLinksInContain();
    void setLinkInitStatus(FMLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN]);
    bool isAllLinksInitialized(void);
    bool getLinkState(uint32 linkIndex, FMLWLinkStateInfo &linkState);
    bool setLinkState(uint32 linkIndex, FMLWLinkStateInfo linkState);
    bool getNumActiveLinks(uint32_t &numActiveLinks);
    void dumpInfo(std::ostream *os);
    bool isLinkActive(uint32_t linkIndex);
    FMPciInfo_t getPciInfo();
private:

    void populatePciBusIdInformation();

    uint64 mDeviceId;
    lwlink_pci_dev_info mPciInfo;
    char mPciBusId[FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //the tuple domain:bus:device PCI identifier (&amp; NULL terminator)    
    std::string mDeviceName;
    uint8 mDevUuid[LWLINK_UUID_LEN];
    uint16 mNumLinks;
    uint64 mDevType;
    uint64 mEnabledLinkMask;
    uint32 mNodeId;
    
    typedef std::vector<FMLinkInitInfo> FmPerLinkInitStatus;
    FmPerLinkInitStatus mLinkInitStatus;

    typedef std::map <uint32, FMLWLinkStateInfo> FmPerLinkStateInfo;    
    FmPerLinkStateInfo mLinkStateInfo;
};

// LFM maintains a list of devices for easy iterating of all the devices.
typedef std::list <FMLWLinkDevInfo> FMLWLinkDevInfoList;
class LocalFMLWLinkDevRepo
{
public:

    LocalFMLWLinkDevRepo(LocalFMLWLinkDrvIntf *linkDrvIntf);
    virtual ~LocalFMLWLinkDevRepo();

    lwlink_pci_dev_info getDevicePCIInfo(uint64 deviceId);

    uint64 getDeviceId (lwlink_pci_dev_info pciInfo);

    void setLocalNodeId(uint32 nodeId);

    // query LWLink driver for all the registered LWLink devices
    void populateLWLinkDeviceList(void);

    void setDevLinkInitStatus(uint64 deviceId, 
                              FMLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN]);

    uint32 getLocalNodeId(void) { return mLocalNodeId ; }

    FMLWLinkDevInfoList& getDeviceList(void) { return mLWLinkDevList; }

    void dumpInfo(std::ostream *os);

private:

    FMLWLinkDevInfoList mLWLinkDevList;
    uint32 mLocalNodeId;
    LocalFMLWLinkDrvIntf *mLWLinkDrvIntf;
};

// On GFM side, the device look-up is a combination of NodeID and DeviceId
// because DeviceId is only unique within a node.
// So maintained it as Map of Map (per node, per DeviceId)
typedef std::map <uint64, FMLWLinkDevInfo> FMLWLinkDevPerId;
typedef std::map <uint32, FMLWLinkDevPerId> FMLWLinkDevInfoPerNode;
class GlobalFMLWLinkDevRepo
{
public:

    GlobalFMLWLinkDevRepo();
    virtual ~GlobalFMLWLinkDevRepo();

    void addDeviceInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoRsp &devInfoRsp);

    bool getDeviceInfo(uint32 nodeId, uint64 devId, FMLWLinkDevInfo &devInfo);

    bool getDeviceInfo(uint32 nodeId, FMPciInfo_t pciInfo, FMLWLinkDevInfo &devInfo);

    bool setDeviceLinkInitStatus(uint32 nodeId, FMLinkInitStatusInfoList &statusInfo);

    bool setDeviceLinkState(uint32 nodeId, uint64 devId, uint32 linkIndex,
                            FMLWLinkStateInfo linkState);

    bool getDeviceList(uint32 nodeId, FMLWLinkDevInfoList &devList);

    bool mergeDeviceInfoForNode(uint32 nodeId, GlobalFMLWLinkDevRepo &srclwLinkDevRepo);

    void dumpInfo(std::ostream *os);
private:

    FMLWLinkDevInfoPerNode mLWLinkDevPerNode;
};

