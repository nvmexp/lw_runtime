#pragma once

#include <iostream>
#include <map>
#include <list>

#include "DcgmFMCommon.h"
#include "lwlink_lib_ioctl.h"

/*****************************************************************************/
/*  Fabric Manager LWLink common data types                                  */
/*****************************************************************************/

/*
 * This file defines all the LWLink related data structures used by Fabric Manager
 * Most of the structure are typedef from linux/lwlink driver style to FM style.
 * Some of them are redefined, so that we can have appropriate equal to operator
 * and copy constructor
 * Driver structures are defined in drivers\lwlink\interface\lwlink_lib_ctrl.h
 * The idea is to reuse same structures used by the driver so that, if the 
 * driver changes, FM will get a compile time error.
 */

// keep the link, tx and rx sublink state information 
typedef lwlink_link_state DcgmLWLinkStateInfo;

// lwlink tran types, like off_to_safe, safe_to_high etc.
typedef enum DcgmLWLinkTrainType
{
    LWLINK_TRAIN_OFF_TO_SAFE = 0,
    LWLINK_TRAIN_SAFE_TO_HIGH,
    LWLINK_TRAIN_TO_OFF,
    LWLINK_TRAIN_HIGH_TO_SAFE,
    LWLINK_TRAIN_SAFE_TO_OFF
} DcgmLWLinkTrainType;

// LWLink Endpoint and connection information
typedef struct DcgmLWLinkEndPointInfo
{
    uint32 nodeId;
    uint32 linkIndex;
    uint64 gpuOrSwitchId;

    bool operator==(const DcgmLWLinkEndPointInfo& rhs)
    {
        if ( (nodeId == rhs.nodeId) &&
             (linkIndex == rhs.linkIndex) &&
             (gpuOrSwitchId == rhs.gpuOrSwitchId) ) {
            return true;
        } else {
            return false;
        }
    }
} DcgmLWLinkEndPointInfo;

typedef struct DcgmLWLinkConnInfo
{
    DcgmLWLinkEndPointInfo masterEnd;
    DcgmLWLinkEndPointInfo slaveEnd;
    bool operator==(const DcgmLWLinkConnInfo& rhs)
    {
        if ( (masterEnd == rhs.masterEnd) &&
             (slaveEnd == rhs.slaveEnd) ) {
            return true;
        } else {
            return false;
        }
    }    
} DcgmLWLinkConnInfo;

typedef std::list<DcgmLWLinkConnInfo> DcgmLWLinkConnList;

// discovery token information returned by lwlink driver
typedef struct DcgmLinkDiscoveryTokenInfo
{
    uint32 nodeId;
    uint64 gpuOrSwitchId;
    uint32 linkIndex;
    uint64 tokelwalue;
} DcgmLinkDiscoveryTokenInfo;

typedef std::list<DcgmLinkDiscoveryTokenInfo> DcgmLWLinkDiscoveryTokenList;

typedef lwlink_link_init_status DcgmLinkInitInfo;

// Link initialization status information for all the devices on the node
typedef struct DcgmLinkInitStatusInfo
{
    uint32 nodeId;
    uint64 gpuOrSwitchId;
    DcgmLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN];
} DcgmLinkInitStatusInfo;

typedef std::list<DcgmLinkInitStatusInfo> DcgmLinkInitStatusInfoList;

typedef struct DcgmLWLinkRemoteEndPointInfo
{
    uint32 nodeId;
    uint32 linkIndex;
    uint16 pciDomain;
    uint8  pciBus;
    uint8  pciDevice;
    uint8  pciFunction;
    uint64 devType;
    uint8  uuid[LWLINK_UUID_LEN];
} DcgmLWLinkRemoteEndPointInfo;
