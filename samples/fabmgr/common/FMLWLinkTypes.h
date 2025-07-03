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

#include <iostream>
#include <map>
#include <list>

#include "FMCommonTypes.h"
#include "lwlink_lib_ioctl.h"
#include "ctrl_dev_lwswitch.h"
#include "ioctl_common_lwswitch.h"

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
typedef lwlink_link_state FMLWLinkStateInfo;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
typedef struct FMLWLinkFomValues
{
    uint8 numLanes;
    uint16 fomValues[LWSWITCH_LWLINK_MAX_LANES];
} FMLWLinkFomValues;

typedef struct FMLWLinkGradingValues
{
    uint8 laneMask;
    uint8 txInit[LWSWITCH_CCI_XVCR_LANES];
    uint8 rxInit[LWSWITCH_CCI_XVCR_LANES];
    uint8 txMaint[LWSWITCH_CCI_XVCR_LANES];
    uint8 rxMaint[LWSWITCH_CCI_XVCR_LANES];
} FMLWLinkGradingValues;
#endif

typedef struct FMLWLinkQualityInfo
{
    uint8 eomLow;
} FMLWLinkQualityInfo;
// lwlink tran types, like off_to_safe, safe_to_high etc.
typedef enum FMLWLinkTrainType
{
    LWLINK_TRAIN_OFF_TO_SAFE = 0,
    LWLINK_TRAIN_SAFE_TO_HIGH,
    LWLINK_TRAIN_TO_OFF,
    LWLINK_TRAIN_HIGH_TO_SAFE,
    LWLINK_TRAIN_SAFE_TO_OFF,
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    LWLINK_TRAIN_SAFE_TO_INITOPTIMIZE,
    LWLINK_TRAIN_POST_INITOPTIMIZE,
    LWLINK_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH,
    LWLINK_TRAIN_INTERNODE_PARALLEL_TO_OFF,
    LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE,
    LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX,
    LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX,
    LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE,
    LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ,
    LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ,
    LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS,
    LWLINK_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE,
    LWLINK_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES,
#endif
	LWLINK_TRAIN_SAFE_TO_HIGH_SUBLINK,
    LWLINK_TRAIN_SAFE_TO_HIGH_MAINLINK,
    LWLINK_TRAIN_HIGH_TO_SAFE_SUBLINK, 
    LWLINK_TRAIN_HIGH_TO_SAFE_MAINLINK,
    LWLINK_TRAIN_OFF_TO_SAFE_SUBLINK, 
    LWLINK_TRAIN_OFF_TO_SAFE_MAINLINK
} FMLWLinkTrainType;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
#define FM_LWLINK_OPTICAL_FORCE_EQ_RETRY_CNT 3
#endif

// LWLink Endpoint and connection information
typedef struct FMLWLinkEndPointInfo
{
    uint32 nodeId;
    uint32 linkIndex;
    uint64 gpuOrSwitchId;

    bool operator==(const FMLWLinkEndPointInfo& rhs)
    {
        if ( (nodeId == rhs.nodeId) &&
             (linkIndex == rhs.linkIndex) &&
             (gpuOrSwitchId == rhs.gpuOrSwitchId) ) {
            return true;
        } else {
            return false;
        }
    }
} FMLWLinkEndPointInfo;

typedef struct FMLWLinkConnInfo
{
    FMLWLinkEndPointInfo masterEnd;
    FMLWLinkEndPointInfo slaveEnd;
    bool operator==(const FMLWLinkConnInfo& rhs)
    {
        if ( (masterEnd == rhs.masterEnd) &&
             (slaveEnd == rhs.slaveEnd) ) {
            return true;
        } else {
            return false;
        }
    }    
} FMLWLinkConnInfo;

typedef std::list<FMLWLinkConnInfo> FMLWLinkConnList;

// discovery token information returned by lwlink driver
typedef struct FMLinkDiscoveryTokenInfo
{
    uint32 nodeId;
    uint64 gpuOrSwitchId;
    uint32 linkIndex;
    uint64 tokelwalue;
} FMLinkDiscoveryTokenInfo;

typedef std::list<FMLinkDiscoveryTokenInfo> FMLWLinkDiscoveryTokenList;

// link SID information returned by lwlink driver
typedef struct FMLinkSidInfo
{
    uint32 nodeId;
    uint64 gpuOrSwitchId;
    uint64 nearSid;
    uint32 nearLinkIndex;
    uint64 farSid;
    uint32 farLinkIndex;
} FMLinkSidInfo;

typedef std::list<FMLinkSidInfo> FMLWLinkSidList;

typedef lwlink_link_init_status FMLinkInitInfo;

// Link initialization status information for all the devices on the node
typedef struct FMLinkInitStatusInfo
{
    uint32 nodeId;
    uint64 gpuOrSwitchId;
    FMLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN];
} FMLinkInitStatusInfo;

typedef std::list<FMLinkInitStatusInfo> FMLinkInitStatusInfoList;

typedef struct FMLWLinkRemoteEndPointInfo
{
    uint32 nodeId;
    uint32 linkIndex;
    uint16 pciDomain;
    uint8  pciBus;
    uint8  pciDevice;
    uint8  pciFunction;
    uint64 devType;
    uint8  uuid[LWLINK_UUID_LEN];
} FMLWLinkRemoteEndPointInfo;
