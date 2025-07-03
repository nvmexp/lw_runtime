
#pragma once

#include <iostream>
#include <fstream>
#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmFMLWLinkTypes.h"
#include "DcgmFMLWLinkDrvIntf.h"

class DcgmGlobalFabricManager;
/*****************************************************************************/
/*  Fabric Manager LWLink Device list/repository                             */
/*****************************************************************************/

/*
 * DcgmFMLWLinkDevInfo:
 *  This class holds detailed information about an LWLink device. It also creates
 *  a unique device ID for addressing each device externally. This device ID is
 *  lwrrently unique only within the node.
 *
 * DcgmLFMLWLinkDevRepo
 *  Holds the list of LWLink devices in the local node which is queried from the
 *  LWLink driver. The LWLink GPB message handlers will query the list of LWLink
 *  devices to repeat the LWLink driver IOCLTs for all the devices. (as
 *  few IOCTLs are per LWLink device)
 *
 * DcgmGFMLWLinkDevRepo
 *  Holds the list of LWLink devices received from all the nodes in GFM context.
 *  This device information later used to uniquely address each LWLink device
 *  for GFM <-> LFM communication. The device information will be used for 
 *  logging and topology verification as well.
 */

class DcgmFMLWLinkDevInfo
{
public:

    // the default constructor is for assignment
    DcgmFMLWLinkDevInfo();

    // this constructor will be used by LFM to create device information
    DcgmFMLWLinkDevInfo(uint32 nodeId, lwlink_detailed_dev_info devInfo);

    // this constrlwtor will be used by GFM to create device information
    // received from other peer LFMs
    DcgmFMLWLinkDevInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoMsg &devInfoMsg);

    ~DcgmFMLWLinkDevInfo();

    // getters for all the member/device information
    uint64 getDeviceId() { return mDeviceId; }
    uint16 getNumLinks() { return mNumLinks; }
    uint64 getDeviceType() { return mDevType; }
    std::string getDeviceName() { return mDeviceName; }
    uint8* getDeviceUuid() { return mDevUuid; }
    lwlink_pci_dev_info getDevicePCIInfo() { return mPciInfo ;}
    uint64 getEnabledLinkMask(void) { return mEnabledLinkMask; }
    void getInitFailedLinksIndex(std::list<uint32> &initFailedLinks);
    void setLinkInitStatus(DcgmLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN]);
    bool isAllLinksInitialized(void);
    bool setLinkState(uint32 linkIndex, DcgmLWLinkStateInfo linkState);
    bool publishLinkStateToCacheManager(DcgmGlobalFabricManager *pGfm);
    void dumpInfo(std::ostream *os);
private:
    bool publishGpuLinkStateToCacheManager(DcgmGlobalFabricManager *pGfm);
    bool publishLWSwitchLinkStateToCacheManager(DcgmGlobalFabricManager *pGfm);
    bool getGpuEnumIndex(DcgmGlobalFabricManager *pGfm, uint32_t &gpuEnumIdx);
    bool getLWSwitchPhysicalId(DcgmGlobalFabricManager *pGfm, uint32_t &physicalId);

    uint64 mDeviceId;
    lwlink_pci_dev_info mPciInfo;
    std::string mDeviceName;
    uint8 mDevUuid[LWLINK_UUID_LEN];
    uint16 mNumLinks;
    uint64 mDevType;
    uint64 mEnabledLinkMask;
    uint32 mNodeId;

    typedef std::vector<DcgmLinkInitInfo> DcgmPerLinkInitStatus;
    DcgmPerLinkInitStatus mLinkInitStatus;

    typedef std::map <uint32, DcgmLWLinkStateInfo> DcgmPerLinkStateInfo;    
    DcgmPerLinkStateInfo mLinkStateInfo;
};

// LFM maintains a list of devices for easy iterating of all the devices.
typedef std::list <DcgmFMLWLinkDevInfo> DcgmFMLWLinkDevInfoList;
class DcgmLFMLWLinkDevRepo
{
public:

    DcgmLFMLWLinkDevRepo(DcgmFMLWLinkDrvIntf *linkDrvIntf);
    virtual ~DcgmLFMLWLinkDevRepo();

    lwlink_pci_dev_info getDevicePCIInfo(uint64 deviceId);

    uint64 getDeviceId (lwlink_pci_dev_info pciInfo);

    void setLocalNodeId(uint32 nodeId);

    // query LWLink driver for all the registered LWLink devices
    void populateLWLinkDeviceList(void);

    void setDevLinkInitStatus(uint64 deviceId, 
                              DcgmLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN]);

    uint32 getLocalNodeId(void) { return mLocalNodeId ; }

    DcgmFMLWLinkDevInfoList& getDeviceList(void) { return mLWLinkDevList; }

    void dumpInfo(std::ostream *os);

private:

    DcgmFMLWLinkDevInfoList mLWLinkDevList;
    uint32 mLocalNodeId;
    DcgmFMLWLinkDrvIntf *mLWLinkDrvIntf;
};

// On GFM side, the device look-up is a combination of NodeID and DeviceId
// because DeviceId is only unique within a node.
// So maintained it as Map of Map (per node, per DeviceId)
typedef std::map <uint64, DcgmFMLWLinkDevInfo> DcgmFMLWLinkDevPerId;
typedef std::map <uint32, DcgmFMLWLinkDevPerId> DcgmFMLWLinkDevInfoPerNode;
class DcgmGFMLWLinkDevRepo
{
public:

    DcgmGFMLWLinkDevRepo();
    virtual ~DcgmGFMLWLinkDevRepo();

    void addDeviceInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoRsp &devInfoRsp);

    bool getDeviceInfo(uint32 nodeId, uint64 devId, DcgmFMLWLinkDevInfo &devInfo);

    bool getDeviceInfo(uint32 nodeId, DcgmFMPciInfo pciInfo, DcgmFMLWLinkDevInfo &devInfo);

    bool setDeviceLinkInitStatus(uint32 nodeId, DcgmLinkInitStatusInfoList &statusInfo);

    bool setDeviceLinkState(uint32 nodeId, uint64 devId, uint32 linkIndex,
                            DcgmLWLinkStateInfo linkState);

    bool publishNodeLinkStateToCacheManager(uint32 nodeId, DcgmGlobalFabricManager *pGfm);

    bool getDeviceList(uint32 nodeId, DcgmFMLWLinkDevInfoList &devList);

    void dumpInfo(std::ostream *os);
private:

    DcgmFMLWLinkDevInfoPerNode mLWLinkDevPerNode;
};

