/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LWLINK_H_
#define _LWLINK_H_

#include "os.h"
#include "hal.h"
#include "g_lwlink_private.h"

#define DEVICE_INFO_TYPE_ILWALID    0xffffffff
#define DEVICE_ID_ILWALID           0xffffffff

//
// Used to mask the PLWLTL_INTEN register as only the bottom 14 bits are used
// to represent interrupts.
//
#define LW_PLWLTL_INTEN_REG_MASK    0x00003FFF

#define LWLINK_UPHY_REG_POLL_TIMEOUT    1000

//defined offsets between addresses of base lwlink devices
#define LWLINK_LINK_STRIDE 0x8000
#define LWLINK_IOCTRL_STRIDE 0x40000

#define LWLINK_MAX_LINKS_PER_IOCTRL_SW 6
#define LWLINK_MAX_IOCTRL_UNITS 3
#define LWLINK_MAX_LINKS 18

typedef enum
{
    LWL_OFF,
    LWL_NORMAL,
    LWL_S_SAFE,
    LWL_ERROR,
    LWL_SLEEP
} LWL_STATUS;

typedef enum
{
    LWL_P2P,
    LWL_SYSMEM,
    LWL_UNKNOWN,
    LWL_T_NA
} LWL_TYPE;

typedef enum
{
    LWL_FULL,
    LWL_POFF,
    LWL_L2
} LWL_POWER;

typedef enum
{
    LWL_SAFE,
    LWL_FAST,
    LWL_NA,
    LWL_M_OFF
} LWL_MODE;

typedef enum
{
    LWL_ILWALID,
    LWL_IOCTRL,
    LWL_LWLTL,
    LWL_LWLINK
} LWL_IOCTRL_DEVICE;

// Basic structure of the lwlink status node.
typedef struct lwlStatusNode
{
    LwU32       linkNum;
    LWL_STATUS  status;
    LWL_TYPE    type;
    LwU32       peer;
    LWL_POWER   power;
    LWL_MODE    rxMode;
    LWL_MODE    txMode;
    LwU32       tlPriBase;
    LwU32       dlPriBase;
} lwlStatus;

typedef struct lwlDiscoverNode
{
    LwU32 ID;
    LwU32 addr;
    LWL_IOCTRL_DEVICE deviceType;
    struct lwlDiscoverNode *next;
} lwlDiscover;


// LWLink helper function for lwwatch commands
void lwlinkPrintHelp(void);

// Utility functions
const char *lwlinkStatusText(LWL_STATUS status);
const char *lwlinkTypeText(LWL_TYPE type);
const char *lwlinkPowerText(LWL_POWER power);
const char *lwlinkModeText(LWL_MODE mode);

// IOCTRL and lwlink devices discovery functions
lwlDiscover *lwlinkDeviceInstanceInfoLookUp(lwlDiscover *pLwlDiscoverList, LwU32 linkId, LwU32 deviceType);
LW_STATUS    initLwlStatusArray(LwU32 *statusArraySize, LwU32 linksDiscovered, lwlDiscover *pLwlDiscoverList);
LwU32        lwlinkIoctrlDiscovery(void);
LwU32        lwlinkLinkDiscovery(void **pLwlVoid);
void         printLwlStatusArray(LwU32 statusArraySize);
void         freeLwlStatusArray(void);
void         freeLwlDiscoveryList(lwlDiscover *pLwlDiscoverList);

// Print LWLink status
void lwlinkPrintStatus(LwBool bComplete);
void lwlinkPrintVerbose(BOOL bPretty);

// Print traffic type (sysmem/peer) for the given link
void lwlinkPrintLinkTrafficType(LwU32 linkId);

// Print status of units inside IOCTRL
void lwlinkPrintLinkDlplState(LwU32 addr);
void lwlinkPrintLinkTlcState(LwU32 addr);
void lwlinkPrintLinkMifState(LwU32 addr);

// Print link and sublink states
LW_STATUS lwlinkGetStatus(LwU32 statusArraySize, void *pLwlVoid);
LW_STATUS lwlinkGetRxTxMode(LwU32 statusArraySize, void *pLwlVoid);

// Print power state of links
LW_STATUS lwlinkGetPowerStatus(LwU32 statusArraySize, void *pLwlVoid);

// LWLTL related functions
void lwlinkProgramTlCounters(LwS32 linkId);
void lwlinkResetTlCounters(LwS32 linkId);
void lwlinkReadTlCounters(LwS32 linkId);

// HSHUB related functions
LW_STATUS lwlinkGetHshubStatus(LwU32 statusArraySize, void *pLwlVoid);
void lwlinkPrintHshubConfig(void);
void lwlinkPrintHshubIdleStatus(void);
void lwlinkEnableHshubLogging(void);
void lwlinkLogHshubErrors(void);
void lwlinkPrintHshubReqTimeoutInfo(void);
void lwlinkPrintHshubReadTimeoutInfo(void);
void lwlinkPrintHshubConnectionCfg(void);
void lwlinkPrintHshubMuxConfig(void);

// UPHY related functions (Leagcy - only supported on Pascal, not supported Volta+)
LwU16 lwlinkReadUPhyPLLCfg(LwU32 link, LwU32 addr);
LwU16 lwlinkReadUPhyLaneCfg(LwU32 link, LwU32 lane, LwU32 addr);
void  lwlinkWriteUPhyPLLCfg(LwU32 link, LwU32 addr, LwU16 data);
void  lwlinkWriteUPhyLaneCfg(LwU32 link, LwU32 lane, LwU32 addr, LwU16 data);
void  lwlinkDumpUPhys(void);

#endif // _LWLINK_H_
