/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * This file defines CTRL calls that are device specifics.
 *
 * This is a platform agnostic file and lists the CTRL calls used by all the
 * clients, Fabric Manager, MODS or LWSwitch GTEST etc.
 *
 * As Fabric Manager relies on driver ABI compatibility the CTRL calls listed in
 * this file contribute to the driver ABI version.
 *
 * Note: ctrl_dev_lwswitch.h and ctrl_dev_internal_lwswitch.h do not share any
 * data. This helps to keep the driver ABI stable.
 */

#ifndef _CTRL_DEVICE_LWSWITCH_H_
#define _CTRL_DEVICE_LWSWITCH_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "g_lwconfig.h"
#include "lwtypes.h"
#include "lwfixedtypes.h"
#include "lwmisc.h"
#include "ioctl_common_lwswitch.h"

/*
 * CTRL_LWSWITCH_GET_INFO
 *
 * Control for querying miscellaneous device information.
 *
 * This provides a single API to query for multiple pieces of miscellaneous
 * information via a single call.
 *
 * Parameters:
 *   count [IN]
 *      Count of queries. Max supported queries per-call are
 *      LWSWITCH_GET_INFO_COUNT_MAX
 *   index [IN]
 *      One of the LWSWITCH_GET_INFO_INDEX type value.
 *
 *   info [OUT]
 *      Data pertaining to the provided LWSWITCH_GET_INFO_INDEX type value.
 */

#define LWSWITCH_GET_INFO_COUNT_MAX 32

typedef enum lwswitch_get_info_index
{
    LWSWITCH_GET_INFO_INDEX_ARCH = 0x0,
    LWSWITCH_GET_INFO_INDEX_IMPL,
    LWSWITCH_GET_INFO_INDEX_CHIPID,
    LWSWITCH_GET_INFO_INDEX_REVISION_MAJOR,
    LWSWITCH_GET_INFO_INDEX_REVISION_MINOR,
    LWSWITCH_GET_INFO_INDEX_REVISION_MINOR_EXT,
    LWSWITCH_GET_INFO_INDEX_FOUNDRY,
    LWSWITCH_GET_INFO_INDEX_FAB,
    LWSWITCH_GET_INFO_INDEX_LOT_CODE_0,
    LWSWITCH_GET_INFO_INDEX_LOT_CODE_1,
    LWSWITCH_GET_INFO_INDEX_WAFER,
    LWSWITCH_GET_INFO_INDEX_XCOORD,
    LWSWITCH_GET_INFO_INDEX_YCOORD,
    LWSWITCH_GET_INFO_INDEX_SPEEDO_REV,
    LWSWITCH_GET_INFO_INDEX_SPEEDO0,
    LWSWITCH_GET_INFO_INDEX_SPEEDO1,
    LWSWITCH_GET_INFO_INDEX_SPEEDO2,
    LWSWITCH_GET_INFO_INDEX_IDDQ,
    LWSWITCH_GET_INFO_INDEX_IDDQ_REV,
    LWSWITCH_GET_INFO_INDEX_IDDQ_DVDD,
    LWSWITCH_GET_INFO_INDEX_ATE_REV,
    LWSWITCH_GET_INFO_INDEX_VENDOR_CODE,
    LWSWITCH_GET_INFO_INDEX_OPS_RESERVED,
    LWSWITCH_GET_INFO_INDEX_PLATFORM,
    LWSWITCH_GET_INFO_INDEX_DEVICE_ID,

    LWSWITCH_GET_INFO_INDEX_NUM_PORTS = 0x100,
    LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0,
    LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32,
    LWSWITCH_GET_INFO_INDEX_NUM_VCS,
    LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_TABLE_SIZE,
    LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTA_TABLE_SIZE,
    LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTB_TABLE_SIZE,
    LWSWITCH_GET_INFO_INDEX_ROUTING_ID_TABLE_SIZE,
    LWSWITCH_GET_INFO_INDEX_ROUTING_LAN_TABLE_SIZE,
    LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_MULTICAST_TABLE_SIZE,

    LWSWITCH_GET_INFO_INDEX_FREQ_KHZ = 0x200,
    LWSWITCH_GET_INFO_INDEX_VCOFREQ_KHZ,
    LWSWITCH_GET_INFO_INDEX_VOLTAGE_MVOLT,
    LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID,

    LWSWITCH_GET_INFO_INDEX_PCI_DOMAIN = 0x300,
    LWSWITCH_GET_INFO_INDEX_PCI_BUS,
    LWSWITCH_GET_INFO_INDEX_PCI_DEVICE,
    LWSWITCH_GET_INFO_INDEX_PCI_FUNCTION,

    LWSWITCH_GET_INFO_INDEX_INFOROM_LWL_SUPPORTED = 0x400,
    LWSWITCH_GET_INFO_INDEX_INFOROM_BBX_SUPPORTED
    /* See enum modification guidelines at the top of this file */
} LWSWITCH_GET_INFO_INDEX;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
#define LWSWITCH_GET_INFO_INDEX_ARCH_SV10     0x01
#define LWSWITCH_GET_INFO_INDEX_IMPL_SV10     0x01
#endif

#define LWSWITCH_GET_INFO_INDEX_ARCH_LR10     0x02
#define LWSWITCH_GET_INFO_INDEX_IMPL_LR10     0x01

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#define LWSWITCH_GET_INFO_INDEX_ARCH_LS10     0x03
#define LWSWITCH_GET_INFO_INDEX_IMPL_LS10     0x01
#endif
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWSWITCH_GET_INFO_INDEX_IMPL_S000     0x02
#endif

#define LWSWITCH_GET_INFO_INDEX_PLATFORM_UNKNOWN    0x00
#define LWSWITCH_GET_INFO_INDEX_PLATFORM_RTLSIM     0x01
#define LWSWITCH_GET_INFO_INDEX_PLATFORM_FMODEL     0x02
#define LWSWITCH_GET_INFO_INDEX_PLATFORM_EMULATION  0x03
#define LWSWITCH_GET_INFO_INDEX_PLATFORM_SILICON    0x04

typedef struct lwswitch_get_info
{
    LwU32 count;
    LwU32 index[LWSWITCH_GET_INFO_COUNT_MAX];
    LwU32 info[LWSWITCH_GET_INFO_COUNT_MAX];

} LWSWITCH_GET_INFO;

/*
 * CTRL_LWSWITCH_SET_INGRESS_REQUEST_TABLE
 *
 * Control for programming ingress request tables.
 * This interface is only supported on SV10 architecture.  All others will
 * return an error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the ingress request table from which table entries
 *      should be programmed.
 *   numEntries [IN]
 *      Number of entries to be programmed. Lwrrently, the call supports
 *      programming LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX at a time.
 *   entries [IN]
 *      The entries (entry format is architecture dependent).
 */

#define LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX 256

/* TODO: document the entry format in detail */
typedef struct lwswitch_ingress_request_entry
{
    LwU32  vcModeValid7_0;
    LwU32  vcModeValid15_8;
    LwU32  vcModeValid17_16;
    LwU32  mappedAddress;
    LwU32  routePolicy;
    LwBool entryValid;

} LWSWITCH_INGRESS_REQUEST_ENTRY;

typedef struct lwswitch_set_ingress_request_table
{
    LwU32                          portNum;
    LwU32                          firstIndex;
    LwU32                          numEntries;
    LWSWITCH_INGRESS_REQUEST_ENTRY entries[LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX];

} LWSWITCH_SET_INGRESS_REQUEST_TABLE;

/*
 * CTRL_LWSWITCH_GET_INGRESS_REQUEST_TABLE
 *
 * Control for reading ingress request tables. A sparse list of nonzero entries
 * and their table indices is returned.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the ingress request table from which table entries
 *      should be read.
 *   nextIndex [OUT]
 *      The table index of the next entry to read. Set to INGRESS_MAP_TABLE_SIZE
 *      when the end of the table has been reached.
 *   numEntries [OUT]
 *      Number of entries returned. Lwrrently, the call supports returning up to
 *      LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX entries at a time.
 *   entries [OUT]
 *      Ingress request entries along with their table indices.
 *      Entry format is architecture dependent.
 */

typedef struct lwswitch_ingress_request_idx_entry
{
    LwU32                          idx;
    LWSWITCH_INGRESS_REQUEST_ENTRY entry;

} LWSWITCH_INGRESS_REQUEST_IDX_ENTRY;

typedef struct lwswitch_get_ingress_request_table_params
{
    LwU32                               portNum;
    LwU32                               firstIndex;
    LwU32                               nextIndex;
    LwU32                               numEntries;
    LWSWITCH_INGRESS_REQUEST_IDX_ENTRY  entries[LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX];

} LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS;

/*
 * CTRL_LWSWITCH_SET_INGRESS_REQUEST_VALID
 *
 * Control for toggling the existing ingress request table entries' validity.
 * This interface is only supported on SV10 architecture.  All others will
 * return an error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the ingress request table from which table entries
 *      should be programmed.
 *   numEntries [IN]
 *      Number of entries to be programmed. Lwrrently, the call supports
 *      programming LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX at a time.
 *   entryValid [IN]
 *      If true, an existing entry is marked valid, else will be marked invalid.
 */

typedef struct lwswitch_set_ingress_request_valid
{
    LwU32  portNum;
    LwU32  firstIndex;
    LwU32  numEntries;
    LwBool entryValid[LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX];

} LWSWITCH_SET_INGRESS_REQUEST_VALID;

/*
 * CTRL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE
 *
 * Control for programming ingress response tables.
 * This interface is only supported on SV10 architecture.  All others will
 * return an error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the ingress request table from which table entries
 *      should be programmed.
 *   numEntries [IN]
 *      Number of entries to be programmed. Lwrrently, the call supports
 *      programming LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX at a time.
 *   entries [IN]
 *      The entries (entry format is architecture dependent).
 */

#define LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX 256

/* TODO: document the entry format in detail */
typedef struct lwswitch_ingress_response_entry
{
    LwU32  vcModeValid7_0;
    LwU32  vcModeValid15_8;
    LwU32  vcModeValid17_16;
    LwU32  routePolicy;
    LwBool entryValid;

} LWSWITCH_INGRESS_RESPONSE_ENTRY;

typedef struct lwswitch_set_ingress_response_table
{
    LwU32                           portNum;
    LwU32                           firstIndex;
    LwU32                           numEntries;
    LWSWITCH_INGRESS_RESPONSE_ENTRY entries[LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX];

} LWSWITCH_SET_INGRESS_RESPONSE_TABLE;

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY
 *
 * Control to load remap policy table
 * This interface is not supported on SV10 architecture.  SV10 will return an
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   tableSelect [IN]
 *      Remap table selector
 *   firstIndex [IN]
 *      A starting index of the remap table from which table entries
 *      should be programmed.  Valid range should be queried using
 *      LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_TABLE_SIZE.
 *   numEntries [IN]
 *      Number of entries to be programmed. Lwrrently, the call supports
 *      programming LWSWITCH_REMAP_POLICY_ENTRIES_MAX at a time.
 *   remapPolicy [IN]
 *      The entries (see LWSWITCH_REMAP_POLICY_ENTRY).
 */

#define LWSWITCH_REMAP_POLICY_ENTRIES_MAX 64

#define LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR      LWBIT(0)
#define LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK   LWBIT(1)
#define LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE LWBIT(2)
#define LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE        LWBIT(4)
#define LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET      LWBIT(5)    /* Apply address offset */
#define LWSWITCH_REMAP_POLICY_FLAGS_REFLECTIVE      LWBIT(30)   /* Reflective mapping */
#define LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE       LWBIT(31)   /* Enforce address type checking */

typedef struct lwswitch_remap_policy_entry
{
    LwBool entryValid;
    LwU32  targetId;                            /* Unique endpoint ID */

    LwU32  irlSelect;                           /* Injection rate limiter (0=none/1=IRL1/2=IRL2) */

    LwU32  flags;                               /* See LWSWITCH_REMAP_POLICY_FLAGS_* */

    LW_DECLARE_ALIGNED(LwU64 address, 8);       /* 47-bit remap address. Bits 46:36 are used. */

                                                /* reqContext fields are used when */
                                                /* routing function _REQCTXT_CHECK or _REPLACE */
                                                /* is set. */
    LwU32  reqCtxMask;                          /* Used to mask packet request ctxt before */
                                                /* checking. */

    LwU32  reqCtxChk;                           /* Post-mask packet request ctxt check value. */
                                                /* Packets that fail compare are colwerted to */
                                                /* UR response and looped back. */

    LwU32  reqCtxRep;                           /* Replaces packet request context when */
                                                /* _REQCTXT_REPLACE is set. */

    LW_DECLARE_ALIGNED(LwU64 addressOffset, 8); /* offset - base is added to packet address if */
                                                /* routing function _ADR_OFFSET & _ADR_BASE are */
                                                /* set. 64GB offset 1MB aligned on LR10. */

    LW_DECLARE_ALIGNED(LwU64 addressBase,  8);  /* If routing function _ADR_BASE is set, limits */
    LW_DECLARE_ALIGNED(LwU64 addressLimit, 8);  /* application of _ADR_OFFSET to packet */
                                                /* addresses that pass base/limit bounds check. */
                                                /* Maximum 64GB size 1MB aligned on LR10. */


} LWSWITCH_REMAP_POLICY_ENTRY;

typedef enum lwswitch_table_select_remap
{
    LWSWITCH_TABLE_SELECT_REMAP_PRIMARY = 0,
    LWSWITCH_TABLE_SELECT_REMAP_EXTA,
    LWSWITCH_TABLE_SELECT_REMAP_EXTB,
    LWSWITCH_TABLE_SELECT_REMAP_MULTICAST
} LWSWITCH_TABLE_SELECT_REMAP;

typedef struct lwswitch_set_remap_policy
{
    LwU32                       portNum;
    LWSWITCH_TABLE_SELECT_REMAP tableSelect;
    LwU32                       firstIndex;
    LwU32                       numEntries;
    LWSWITCH_REMAP_POLICY_ENTRY remapPolicy[LWSWITCH_REMAP_POLICY_ENTRIES_MAX];

} LWSWITCH_SET_REMAP_POLICY;

/*
 * CTRL_LWSWITCH_GET_REMAP_POLICY
 *
 * Control to get remap policy table
 * This interface is not supported on SV10 architecture. SV10 will return unsupported
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   tableSelect [IN]
 *      Remap table selector
 *   firstIndex [IN]
 *      A starting index of the remap policy table from which table entries
 *      should be read.
 *   numEntries [OUT]
 *      Number of entries returned. This call returns
 *      LWSWITCH_REMAP_POLICY_ENTRIES_MAX entries at a time.
 *   nextIndex [OUT]
 *      The table index of the next entry to read. Set to INGRESS_REMAPTAB_SIZE
 *      when the end of the table has been reached.
 *   entries [OUT]
 *      The entries (see LWSWITCH_REMAP_POLICY_ENTRY).
 */


typedef struct lwswitch_get_remap_policy_params
{
    LwU32                             portNum;
    LWSWITCH_TABLE_SELECT_REMAP       tableSelect;
    LwU32                             firstIndex;
    LwU32                             numEntries;
    LwU32                             nextIndex;
    LWSWITCH_REMAP_POLICY_ENTRY       entry[LWSWITCH_REMAP_POLICY_ENTRIES_MAX];

} LWSWITCH_GET_REMAP_POLICY_PARAMS;

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY_VALID
 *
 * Control to set remap policy tables valid/invalid
 * This interface is not supported on SV10 architecture.  SV10 will return unsupported
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   tableSelect [IN]
 *      Remap table selector
 *   firstIndex [IN]
 *      A starting index of the remap policy table from which table entries
 *      should be programmed.
 *   numEntries [IN]
 *      Number of entries to be programmed. The call supports
 *      programming of maximum LWSWITCH_REMAP_POLICY_ENTRIES_MAX at a time.
 *   entryValid [IN]
 *      If true, an existing entry is marked valid, else will be marked invalid.
 */

typedef struct lwswitch_set_remap_policy_valid
{
    LwU32                      portNum;
    LWSWITCH_TABLE_SELECT_REMAP tableSelect;
    LwU32                      firstIndex;
    LwU32                      numEntries;
    LwBool                     entryValid[LWSWITCH_REMAP_POLICY_ENTRIES_MAX];

} LWSWITCH_SET_REMAP_POLICY_VALID;

/*
 * CTRL_LWSWITCH_SET_ROUTING_ID
 *
 * Control to load Routing ID table
 * The routing ID table configures the VC and routing policy as well as the
 * valid set if ganged link routes.
 * This interface is not supported on SV10 architecture.  SV10 will return an
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the routing ID table from which table entries
 *      should be programmed.  Valid range should be queried using
 *      LWSWITCH_GET_INFO_INDEX_ROUTING_ID_TABLE_SIZE.
 *   numEntries [IN]
 *      Number of entries to be programmed. Lwrrently, the call supports programming
 *      maximum of LWSWITCH_ROUTING_ID_ENTRIES_MAX entries at a time.
 *   routingId [IN]
 *      The entries (see LWSWITCH_ROUTING_ID_ENTRY).
 */

#define LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX  16
#define LWSWITCH_ROUTING_ID_VC_MODE_MAX          4
#define LWSWITCH_ROUTING_ID_ENTRIES_MAX         64

typedef enum lwswitch_routing_id_vcmap
{
    LWSWITCH_ROUTING_ID_VCMAP_SAME = 0x0,
    LWSWITCH_ROUTING_ID_VCMAP_ILWERT,
    LWSWITCH_ROUTING_ID_VCMAP_ZERO,
    LWSWITCH_ROUTING_ID_VCMAP_ONE
    /* See enum modification guidelines at the top of this file */
} LWSWITCH_ROUTING_ID_VCMAP;

typedef struct lwswitch_routing_id_dest_port_list
{
    LwU32 vcMap;      /* LWSWITCH_ROUTING_ID_VCMAP_* */
    LwU32 destPortNum;

} LWSWITCH_ROUTING_ID_DEST_PORT_LIST;

typedef struct lwswitch_routing_id_entry
{
    LwBool                              entryValid;
    LwBool                              useRoutingLan;
    LwBool                              enableIrlErrResponse;
    LwU32                               numEntries;
    LWSWITCH_ROUTING_ID_DEST_PORT_LIST  portList[LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX];

} LWSWITCH_ROUTING_ID_ENTRY;

typedef struct lwswitch_set_routing_id
{
    LwU32                       portNum;
    LwU32                       firstIndex;
    LwU32                       numEntries;
    LWSWITCH_ROUTING_ID_ENTRY   routingId[LWSWITCH_ROUTING_ID_ENTRIES_MAX];

} LWSWITCH_SET_ROUTING_ID;

/*
 * CTRL_LWSWITCH_GET_ROUTING_ID
 *
 * Control to get routing ID table
 * This interface is not supported on SV10 architecture. SV10 will return unsupported
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the routing id table from which table entries
 *      should be read.
 *   numEntries [OUT]
 *      Number of entries returned. The call returns only
 *      LWSWITCH_ROUTING_ID_ENTRIES_MAX entries at a time.
 *   nextIndex [OUT]
 *      The table index of the next entry to read. Set to INGRESS_RIDTAB_SIZE
 *      when the end of the table has been reached.
 *   entries [OUT]
 *      The entries (see LWSWITCH_ROUTING_ID_IDX_ENTRY).
 */

typedef struct lwswitch_routing_id_idx_entry
{
    LwU32                               idx;
    LWSWITCH_ROUTING_ID_ENTRY          entry;

} LWSWITCH_ROUTING_ID_IDX_ENTRY;

typedef struct lwswitch_get_routing_id_params
{
    LwU32                             portNum;
    LwU32                             firstIndex;
    LwU32                             numEntries;
    LwU32                             nextIndex;
    LWSWITCH_ROUTING_ID_IDX_ENTRY     entries[LWSWITCH_ROUTING_ID_ENTRIES_MAX];

} LWSWITCH_GET_ROUTING_ID_PARAMS;

/*
 * CTRL_LWSWITCH_SET_ROUTING_ID_VALID
 *
 * Control to set routing ID tables valid/invalid
 * This interface is not supported on SV10 architecture.  SV10 will return unsupported
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the routing lan table from which table entries
 *      should be programmed.
 *   numEntries [IN]
 *      Number of entries to be programmed. This call supports programming
 *      maximum entries of LWSWITCH_ROUTING_ID_ENTRIES_MAX at a time.
 *   entryValid [IN]
 *      If true, an existing entry is marked valid, else will be marked invalid.
 */

typedef struct lwswitch_set_routing_id_valid
{
    LwU32                      portNum;
    LwU32                      firstIndex;
    LwU32                      numEntries;
    LwBool                     entryValid[LWSWITCH_ROUTING_ID_ENTRIES_MAX];

} LWSWITCH_SET_ROUTING_ID_VALID;

/*
 * CTRL_LWSWITCH_SET_ROUTING_LAN
 *
 * Control to load routing LAN table
 * This interface is not supported on SV10 architecture.  SV10 will return an
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the ingress request table from which table entries
 *      should be programmed.  Valid range should be queried using
 *      LWSWITCH_GET_INFO_INDEX_ROUTING_LAN_TABLE_SIZE.
 *   numEntries [IN]
 *      Number of entries to be programmed. Lwrrently, the call supports
 *      programming LWSWITCH_ROUTING_LAN_ENTRIES_MAX at a time.
 *   routingLan [IN]
 *      The entries (see LWSWITCH_ROUTING_LAN_ENTRY).
 */

#define LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX  16
#define LWSWITCH_ROUTING_LAN_GROUP_SIZE_MAX 16
#define LWSWITCH_ROUTING_LAN_ENTRIES_MAX    64

typedef struct lwswitch_routing_lan_port_select
{
    LwU32  groupSelect;                 /* Port list group selector */
    LwU32  groupSize;                   /* Valid range: 1..16 */

} LWSWITCH_ROUTING_LAN_PORT_SELECT;

typedef struct lwswitch_routing_lan_entry
{
    LwBool                              entryValid;
    LwU32                               numEntries;
    LWSWITCH_ROUTING_LAN_PORT_SELECT    portList[LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX];

} LWSWITCH_ROUTING_LAN_ENTRY;

typedef struct lwswitch_set_routing_lan
{
    LwU32                      portNum;
    LwU32                      firstIndex;
    LwU32                      numEntries;
    LWSWITCH_ROUTING_LAN_ENTRY routingLan[LWSWITCH_ROUTING_LAN_ENTRIES_MAX];

} LWSWITCH_SET_ROUTING_LAN;

/*
 * CTRL_LWSWITCH_GET_ROUTING_LAN
 *
 * Control to get routing LAN table
 * This interface is not supported on SV10 architecture. SV10 will return unsupported
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the routing lan table from which table entries
 *      should be read.
 *   numEntries [OUT]
 *      Number of entries returned. Lwrrently, the call supports
 *      LWSWITCH_ROUTING_LAN_ENTRIES_MAX at a time.
 *   nextIndex [OUT]
 *      The table index of the next entry to read. Set to INGRESS_RLANTAB_SIZE
 *      when the end of the table has been reached.
 *   entries [OUT]
 *      The entries (see LWSWITCH_ROUTING_LAN_IDX_ENTRY).
 */

typedef struct lwswitch_routing_lan_idx_entry
{
    LwU32                               idx;
    LWSWITCH_ROUTING_LAN_ENTRY          entry;

} LWSWITCH_ROUTING_LAN_IDX_ENTRY;

typedef struct lwswitch_get_routing_lan_params
{
    LwU32                             portNum;
    LwU32                             firstIndex;
    LwU32                             numEntries;
    LwU32                             nextIndex;
    LWSWITCH_ROUTING_LAN_IDX_ENTRY    entries[LWSWITCH_ROUTING_LAN_ENTRIES_MAX];

} LWSWITCH_GET_ROUTING_LAN_PARAMS;

/*
 * CTRL_LWSWITCH_SET_ROUTING_LAN_VALID
 *
 * Control to set routing LAN tables valid/invalid
 * This interface is not supported on SV10 architecture.  SV10 will return unsupported
 * error. Architecture can be queried using _GET_INFO_INDEX_ARCH. 
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the routing lan table from which table entries
 *      should be programmed.
 *   numEntries [IN]
 *      Number of entries to be programmed. Lwrrently, the call supports
 *      programming LWSWITCH_ROUTING_LAN_ENTRIES_MAX at a time.
 *   entryValid [IN]
 *      If true, an existing entry is marked valid, else will be marked invalid.
 */

typedef struct lwswitch_set_routing_lan_valid
{
    LwU32                      portNum;
    LwU32                      firstIndex;
    LwU32                      numEntries;
    LwBool                     entryValid[LWSWITCH_ROUTING_LAN_ENTRIES_MAX];

} LWSWITCH_SET_ROUTING_LAN_VALID;

/*
 * CTRL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE
 *
 * Control for reading ingress response tables. A sparse list of nonzero entries
 * and their table indices is returned.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   firstIndex [IN]
 *      A starting index of the ingress response table from which table entries
 *      should be read.
 *   nextIndex [OUT]
 *      The table index of the next entry to read. Set to INGRESS_MAP_TABLE_SIZE
 *      when the end of the table has been reached.
 *   numEntries [OUT]
 *      Number of entries returned. Lwrrently, the call supports returning up to
 *      LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX entries at a time.
 *   entries [OUT]
 *      Ingress response entries along with their table indices.
 *      Entry format is architecture dependent.
 */

typedef struct lwswitch_ingress_response_idx_entry
{
    LwU32                               idx;
    LWSWITCH_INGRESS_RESPONSE_ENTRY     entry;

} LWSWITCH_INGRESS_RESPONSE_IDX_ENTRY;

typedef struct lwswitch_get_ingress_response_table_params
{
    LwU32                               portNum;
    LwU32                               firstIndex;
    LwU32                               nextIndex;
    LwU32                               numEntries;
    LWSWITCH_INGRESS_RESPONSE_IDX_ENTRY entries[LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX];

} LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS;

/*
 * CTRL_LWSWITCH_GET_ERRORS
 *
 * Control to query error information.
 *
 * Parameters:
 *   errorType [IN]
 *      Allows to query specific class of errors. See LWSWITCH_ERROR_SEVERITY_xxx.
 *
 *   errorIndex [IN/OUT]
 *      On input: The index of the first error of the specified 'errorType' at which to start
 *                reading out of the driver.
 *
 *      On output: The index of the first error that wasn't reported through the 'error' array
 *                 in this call to CTRL_LWSWITCH_GET_ERRORS. Specific to the specified 'errorType'.
 *
 *   nextErrorIndex[OUT]
 *      The index that will be assigned to the next error to occur for the specified 'errorType'.
 *      Users of the GET_ERRORS control call may set 'errorIndex' to this field on initialization
 *      to bypass errors that have already oclwrred without making multiple control calls.
 *
 *   errorCount [OUT]
 *      Number of errors returned by the call. Lwrrently, errorCount is limited
 *      by LWSWITCH_ERROR_COUNT_SIZE. In order to query all the errors, a
 *      client needs to keep calling the control till errorCount is zero.
 *   error [OUT]
 *      The error entires.
 */

typedef enum lwswitch_error_severity_type
{
    LWSWITCH_ERROR_SEVERITY_NONFATAL = 0,
    LWSWITCH_ERROR_SEVERITY_FATAL,
    LWSWITCH_ERROR_SEVERITY_MAX
    /* See enum modification guidelines at the top of this file */
} LWSWITCH_ERROR_SEVERITY_TYPE;

typedef enum lwswitch_error_src_type
{
    LWSWITCH_ERROR_SRC_NONE = 0,
    LWSWITCH_ERROR_SRC_HW
    /* See enum modification guidelines at the top of this file */
} LWSWITCH_ERROR_SRC_TYPE;

typedef enum lwswitch_err_type
{
    LWSWITCH_ERR_NO_ERROR                                                = 0x0,

    /*
     * These error enumerations are derived from the error bits defined in each
     * hardware manual.
     *
     * LWSwitch errors values should start from 10000 (decimal) to be
     * distinguishable from GPU errors.
     */

    /* HOST */
    LWSWITCH_ERR_HW_HOST                                               = 10000,
    LWSWITCH_ERR_HW_HOST_PRIV_ERROR                                    = 10001,
    LWSWITCH_ERR_HW_HOST_PRIV_TIMEOUT                                  = 10002,
    LWSWITCH_ERR_HW_HOST_UNHANDLED_INTERRUPT                           = 10003,
    LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_START                           = 10004,
    LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_END                             = 10005,
    LWSWITCH_ERR_HW_HOST_THERMAL_SHUTDOWN                              = 10006,
    LWSWITCH_ERR_HW_HOST_IO_FAILURE                                    = 10007,
    LWSWITCH_ERR_HW_HOST_LAST,


    /* NPORT: Ingress errors */
    LWSWITCH_ERR_HW_NPORT_INGRESS                                      = 11000,
    LWSWITCH_ERR_HW_NPORT_INGRESS_CMDDECODEERR                         = 11001,
    LWSWITCH_ERR_HW_NPORT_INGRESS_BDFMISMATCHERR                       = 11002,
    LWSWITCH_ERR_HW_NPORT_INGRESS_BUBBLEDETECT                         = 11003,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ACLFAIL                              = 11004,
    LWSWITCH_ERR_HW_NPORT_INGRESS_PKTPOISONSET                         = 11005,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ECCSOFTLIMITERR                      = 11006,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ECCHDRDOUBLEBITERR                   = 11007,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ILWALIDCMD                           = 11008,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ILWALIDVCSET                         = 11009,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ERRORINFO                            = 11010,
    LWSWITCH_ERR_HW_NPORT_INGRESS_REQCONTEXTMISMATCHERR                = 11011,
    LWSWITCH_ERR_HW_NPORT_INGRESS_NCISOC_HDR_ECC_LIMIT_ERR             = 11012,
    LWSWITCH_ERR_HW_NPORT_INGRESS_NCISOC_HDR_ECC_DBE_ERR               = 11013,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ADDRBOUNDSERR                        = 11014,
    LWSWITCH_ERR_HW_NPORT_INGRESS_RIDTABCFGERR                         = 11015,
    LWSWITCH_ERR_HW_NPORT_INGRESS_RLANTABCFGERR                        = 11016,
    LWSWITCH_ERR_HW_NPORT_INGRESS_REMAPTAB_ECC_DBE_ERR                 = 11017,
    LWSWITCH_ERR_HW_NPORT_INGRESS_RIDTAB_ECC_DBE_ERR                   = 11018,
    LWSWITCH_ERR_HW_NPORT_INGRESS_RLANTAB_ECC_DBE_ERR                  = 11019,
    LWSWITCH_ERR_HW_NPORT_INGRESS_NCISOC_PARITY_ERR                    = 11020,
    LWSWITCH_ERR_HW_NPORT_INGRESS_REMAPTAB_ECC_LIMIT_ERR               = 11021,
    LWSWITCH_ERR_HW_NPORT_INGRESS_RIDTAB_ECC_LIMIT_ERR                 = 11022,
    LWSWITCH_ERR_HW_NPORT_INGRESS_RLANTAB_ECC_LIMIT_ERR                = 11023,
    LWSWITCH_ERR_HW_NPORT_INGRESS_ADDRTYPEERR                          = 11024,
    LWSWITCH_ERR_HW_NPORT_INGRESS_LAST, /* NOTE: Must be last */

    /* NPORT: Egress errors */
    LWSWITCH_ERR_HW_NPORT_EGRESS                                       = 12000,
    LWSWITCH_ERR_HW_NPORT_EGRESS_EGRESSBUFERR                          = 12001,
    LWSWITCH_ERR_HW_NPORT_EGRESS_PKTROUTEERR                           = 12002,
    LWSWITCH_ERR_HW_NPORT_EGRESS_ECCSINGLEBITLIMITERR0                 = 12003,
    LWSWITCH_ERR_HW_NPORT_EGRESS_ECCHDRDOUBLEBITERR0                   = 12004,
    LWSWITCH_ERR_HW_NPORT_EGRESS_ECCDATADOUBLEBITERR0                  = 12005,
    LWSWITCH_ERR_HW_NPORT_EGRESS_ECCSINGLEBITLIMITERR1                 = 12006,
    LWSWITCH_ERR_HW_NPORT_EGRESS_ECCHDRDOUBLEBITERR1                   = 12007,
    LWSWITCH_ERR_HW_NPORT_EGRESS_ECCDATADOUBLEBITERR1                  = 12008,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NCISOCHDRCREDITOVFL                   = 12009,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NCISOCDATACREDITOVFL                  = 12010,
    LWSWITCH_ERR_HW_NPORT_EGRESS_ADDRMATCHERR                          = 12011,
    LWSWITCH_ERR_HW_NPORT_EGRESS_TAGCOUNTERR                           = 12012,
    LWSWITCH_ERR_HW_NPORT_EGRESS_FLUSHRSPERR                           = 12013,
    LWSWITCH_ERR_HW_NPORT_EGRESS_DROPNPURRSPERR                        = 12014,
    LWSWITCH_ERR_HW_NPORT_EGRESS_POISONERR                             = 12015,
    LWSWITCH_ERR_HW_NPORT_EGRESS_PACKET_HEADER                         = 12016,
    LWSWITCH_ERR_HW_NPORT_EGRESS_BUFFER_DATA                           = 12017,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NCISOC_CREDITS                        = 12018,
    LWSWITCH_ERR_HW_NPORT_EGRESS_TAG_DATA                              = 12019,
    LWSWITCH_ERR_HW_NPORT_EGRESS_SEQIDERR                              = 12020,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NXBAR_HDR_ECC_LIMIT_ERR               = 12021,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NXBAR_HDR_ECC_DBE_ERR                 = 12022,
    LWSWITCH_ERR_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_LIMIT_ERR             = 12023,
    LWSWITCH_ERR_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_DBE_ERR               = 12024,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NCISOCCREDITOVFL                      = 12025,
    LWSWITCH_ERR_HW_NPORT_EGRESS_REQTGTIDMISMATCHERR                   = 12026,
    LWSWITCH_ERR_HW_NPORT_EGRESS_RSPREQIDMISMATCHERR                   = 12027,
    LWSWITCH_ERR_HW_NPORT_EGRESS_PRIVRSPERR                            = 12028,
    LWSWITCH_ERR_HW_NPORT_EGRESS_HWRSPERR                              = 12029,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NXBAR_HDR_PARITY_ERR                  = 12030,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NCISOC_CREDIT_PARITY_ERR              = 12031,
    LWSWITCH_ERR_HW_NPORT_EGRESS_NXBAR_FLITTYPE_MISMATCH_ERR           = 12032,
    LWSWITCH_ERR_HW_NPORT_EGRESS_CREDIT_TIME_OUT_ERR                   = 12033,
    LWSWITCH_ERR_HW_NPORT_EGRESS_TIMESTAMP_LOG                         = 12034,
    LWSWITCH_ERR_HW_NPORT_EGRESS_MISC_LOG                              = 12035,
    LWSWITCH_ERR_HW_NPORT_EGRESS_HEADER_LOG                            = 12036,
    LWSWITCH_ERR_HW_NPORT_EGRESS_LAST, /* NOTE: Must be last */

    /* NPORT: Fstate errors */
    LWSWITCH_ERR_HW_NPORT_FSTATE                                       = 13000,
    LWSWITCH_ERR_HW_NPORT_FSTATE_TAGPOOLBUFERR                         = 13001,
    LWSWITCH_ERR_HW_NPORT_FSTATE_CRUMBSTOREBUFERR                      = 13002,
    LWSWITCH_ERR_HW_NPORT_FSTATE_SINGLEBITECCLIMITERR_CRUMBSTORE       = 13003,
    LWSWITCH_ERR_HW_NPORT_FSTATE_UNCORRECTABLEECCERR_CRUMBSTORE        = 13004,
    LWSWITCH_ERR_HW_NPORT_FSTATE_SINGLEBITECCLIMITERR_TAGSTORE         = 13005,
    LWSWITCH_ERR_HW_NPORT_FSTATE_UNCORRECTABLEECCERR_TAGSTORE          = 13006,
    LWSWITCH_ERR_HW_NPORT_FSTATE_SINGLEBITECCLIMITERR_FLUSHREQSTORE    = 13007,
    LWSWITCH_ERR_HW_NPORT_FSTATE_UNCORRECTABLEECCERR_FLUSHREQSTORE     = 13008,
    LWSWITCH_ERR_HW_NPORT_FSTATE_LAST, /* NOTE: Must be last */

    /* NPORT: Tstate errors */
    LWSWITCH_ERR_HW_NPORT_TSTATE                                       = 14000,
    LWSWITCH_ERR_HW_NPORT_TSTATE_TAGPOOLBUFERR                         = 14001,
    LWSWITCH_ERR_HW_NPORT_TSTATE_CRUMBSTOREBUFERR                      = 14002,
    LWSWITCH_ERR_HW_NPORT_TSTATE_SINGLEBITECCLIMITERR_CRUMBSTORE       = 14003,
    LWSWITCH_ERR_HW_NPORT_TSTATE_UNCORRECTABLEECCERR_CRUMBSTORE        = 14004,
    LWSWITCH_ERR_HW_NPORT_TSTATE_SINGLEBITECCLIMITERR_TAGSTORE         = 14005,
    LWSWITCH_ERR_HW_NPORT_TSTATE_UNCORRECTABLEECCERR_TAGSTORE          = 14006,
    LWSWITCH_ERR_HW_NPORT_TSTATE_TAGPOOL_ECC_LIMIT_ERR                 = 14007,
    LWSWITCH_ERR_HW_NPORT_TSTATE_TAGPOOL_ECC_DBE_ERR                   = 14008,
    LWSWITCH_ERR_HW_NPORT_TSTATE_CRUMBSTORE_ECC_LIMIT_ERR              = 14009,
    LWSWITCH_ERR_HW_NPORT_TSTATE_CRUMBSTORE_ECC_DBE_ERR                = 14010,
    LWSWITCH_ERR_HW_NPORT_TSTATE_COL_CRUMBSTOREBUFERR                  = 14011,
    LWSWITCH_ERR_HW_NPORT_TSTATE_COL_CRUMBSTORE_ECC_LIMIT_ERR          = 14012,
    LWSWITCH_ERR_HW_NPORT_TSTATE_COL_CRUMBSTORE_ECC_DBE_ERR            = 14013,
    LWSWITCH_ERR_HW_NPORT_TSTATE_TD_TID_RAMBUFERR                      = 14014,
    LWSWITCH_ERR_HW_NPORT_TSTATE_TD_TID_RAM_ECC_LIMIT_ERR              = 14015,
    LWSWITCH_ERR_HW_NPORT_TSTATE_TD_TID_RAM_ECC_DBE_ERR                = 14016,
    LWSWITCH_ERR_HW_NPORT_TSTATE_ATO_ERR                               = 14017,
    LWSWITCH_ERR_HW_NPORT_TSTATE_CAMRSP_ERR                            = 14018,
    LWSWITCH_ERR_HW_NPORT_TSTATE_LAST, /* NOTE: Must be last */

    /* NPORT: Route errors */
    LWSWITCH_ERR_HW_NPORT_ROUTE                                        = 15000,
    LWSWITCH_ERR_HW_NPORT_ROUTE_ROUTEBUFERR                            = 15001,
    LWSWITCH_ERR_HW_NPORT_ROUTE_NOPORTDEFINEDERR                       = 15002,
    LWSWITCH_ERR_HW_NPORT_ROUTE_ILWALIDROUTEPOLICYERR                  = 15003,
    LWSWITCH_ERR_HW_NPORT_ROUTE_ECCLIMITERR                            = 15004,
    LWSWITCH_ERR_HW_NPORT_ROUTE_UNCORRECTABLEECCERR                    = 15005,
    LWSWITCH_ERR_HW_NPORT_ROUTE_TRANSDONERESVERR                       = 15006,
    LWSWITCH_ERR_HW_NPORT_ROUTE_PACKET_HEADER                          = 15007,
    LWSWITCH_ERR_HW_NPORT_ROUTE_GLT_ECC_LIMIT_ERR                      = 15008,
    LWSWITCH_ERR_HW_NPORT_ROUTE_GLT_ECC_DBE_ERR                        = 15009,
    LWSWITCH_ERR_HW_NPORT_ROUTE_PDCTRLPARERR                           = 15010,
    LWSWITCH_ERR_HW_NPORT_ROUTE_LWS_ECC_LIMIT_ERR                      = 15011,
    LWSWITCH_ERR_HW_NPORT_ROUTE_LWS_ECC_DBE_ERR                        = 15012,
    LWSWITCH_ERR_HW_NPORT_ROUTE_CDTPARERR                              = 15013,
    LWSWITCH_ERR_HW_NPORT_ROUTE_LAST, /* NOTE: Must be last */

    /* NPORT: Nport errors */
    LWSWITCH_ERR_HW_NPORT                                              = 16000,
    LWSWITCH_ERR_HW_NPORT_DATAPOISONED                                 = 16001,
    LWSWITCH_ERR_HW_NPORT_UCINTERNAL                                   = 16002,
    LWSWITCH_ERR_HW_NPORT_CINTERNAL                                    = 16003,
    LWSWITCH_ERR_HW_NPORT_LAST, /* NOTE: Must be last */

    /* LWLCTRL: LWCTRL errors */
    LWSWITCH_ERR_HW_LWLCTRL                                            = 17000,
    LWSWITCH_ERR_HW_LWLCTRL_INGRESSECCSOFTLIMITERR                     = 17001,
    LWSWITCH_ERR_HW_LWLCTRL_INGRESSECCHDRDOUBLEBITERR                  = 17002,
    LWSWITCH_ERR_HW_LWLCTRL_INGRESSECCDATADOUBLEBITERR                 = 17003,
    LWSWITCH_ERR_HW_LWLCTRL_INGRESSBUFFERERR                           = 17004,
    LWSWITCH_ERR_HW_LWLCTRL_EGRESSECCSOFTLIMITERR                      = 17005,
    LWSWITCH_ERR_HW_LWLCTRL_EGRESSECCHDRDOUBLEBITERR                   = 17006,
    LWSWITCH_ERR_HW_LWLCTRL_EGRESSECCDATADOUBLEBITERR                  = 17007,
    LWSWITCH_ERR_HW_LWLCTRL_EGRESSBUFFERERR                            = 17008,
    LWSWITCH_ERR_HW_LWLCTRL_LAST, /* NOTE: Must be last */

    /* Nport: Lwlipt errors */
    LWSWITCH_ERR_HW_LWLIPT                                             = 18000,
    LWSWITCH_ERR_HW_LWLIPT_DLPROTOCOL                                  = 18001,
    LWSWITCH_ERR_HW_LWLIPT_DATAPOISONED                                = 18002,
    LWSWITCH_ERR_HW_LWLIPT_FLOWCONTROL                                 = 18003,
    LWSWITCH_ERR_HW_LWLIPT_RESPONSETIMEOUT                             = 18004,
    LWSWITCH_ERR_HW_LWLIPT_TARGETERROR                                 = 18005,
    LWSWITCH_ERR_HW_LWLIPT_UNEXPECTEDRESPONSE                          = 18006,
    LWSWITCH_ERR_HW_LWLIPT_RECEIVEROVERFLOW                            = 18007,
    LWSWITCH_ERR_HW_LWLIPT_MALFORMEDPACKET                             = 18008,
    LWSWITCH_ERR_HW_LWLIPT_STOMPEDPACKETRECEIVED                       = 18009,
    LWSWITCH_ERR_HW_LWLIPT_UNSUPPORTEDREQUEST                          = 18010,
    LWSWITCH_ERR_HW_LWLIPT_UCINTERNAL                                  = 18011,
    LWSWITCH_ERR_HW_LWLIPT_PHYRECEIVER                                 = 18012,
    LWSWITCH_ERR_HW_LWLIPT_BADAN0PKT                                   = 18013,
    LWSWITCH_ERR_HW_LWLIPT_REPLAYTIMEOUT                               = 18014,
    LWSWITCH_ERR_HW_LWLIPT_ADVISORYERROR                               = 18015,
    LWSWITCH_ERR_HW_LWLIPT_CINTERNAL                                   = 18016,
    LWSWITCH_ERR_HW_LWLIPT_HEADEROVERFLOW                              = 18017,
    LWSWITCH_ERR_HW_LWLIPT_RSTSEQ_PHYARB_TIMEOUT                       = 18018,
    LWSWITCH_ERR_HW_LWLIPT_RSTSEQ_PLL_TIMEOUT                          = 18019,
    LWSWITCH_ERR_HW_LWLIPT_CLKCTL_ILLEGAL_REQUEST                      = 18020,
    LWSWITCH_ERR_HW_LWLIPT_LAST, /* NOTE: Must be last */

    /* Nport: Lwltlc TX/RX errors */
    LWSWITCH_ERR_HW_LWLTLC                                             = 19000,
    LWSWITCH_ERR_HW_LWLTLC_TXHDRCREDITOVFERR                           = 19001,
    LWSWITCH_ERR_HW_LWLTLC_TXDATACREDITOVFERR                          = 19002,
    LWSWITCH_ERR_HW_LWLTLC_TXDLCREDITOVFERR                            = 19003,
    LWSWITCH_ERR_HW_LWLTLC_TXDLCREDITPARITYERR                         = 19004,
    LWSWITCH_ERR_HW_LWLTLC_TXRAMHDRPARITYERR                           = 19005,
    LWSWITCH_ERR_HW_LWLTLC_TXRAMDATAPARITYERR                          = 19006,
    LWSWITCH_ERR_HW_LWLTLC_TXUNSUPVCOVFERR                             = 19007,
    LWSWITCH_ERR_HW_LWLTLC_TXSTOMPDET                                  = 19008,
    LWSWITCH_ERR_HW_LWLTLC_TXPOISONDET                                 = 19009,
    LWSWITCH_ERR_HW_LWLTLC_TARGETERR                                   = 19010,
    LWSWITCH_ERR_HW_LWLTLC_TX_PACKET_HEADER                            = 19011,
    LWSWITCH_ERR_HW_LWLTLC_UNSUPPORTEDREQUESTERR                       = 19012,
    LWSWITCH_ERR_HW_LWLTLC_RXDLHDRPARITYERR                            = 19013,
    LWSWITCH_ERR_HW_LWLTLC_RXDLDATAPARITYERR                           = 19014,
    LWSWITCH_ERR_HW_LWLTLC_RXDLCTRLPARITYERR                           = 19015,
    LWSWITCH_ERR_HW_LWLTLC_RXRAMDATAPARITYERR                          = 19016,
    LWSWITCH_ERR_HW_LWLTLC_RXRAMHDRPARITYERR                           = 19017,
    LWSWITCH_ERR_HW_LWLTLC_RXILWALIDAEERR                              = 19018,
    LWSWITCH_ERR_HW_LWLTLC_RXILWALIDBEERR                              = 19019,
    LWSWITCH_ERR_HW_LWLTLC_RXILWALIDADDRALIGNERR                       = 19020,
    LWSWITCH_ERR_HW_LWLTLC_RXPKTLENERR                                 = 19021,
    LWSWITCH_ERR_HW_LWLTLC_RSVCMDENCERR                                = 19022,
    LWSWITCH_ERR_HW_LWLTLC_RSVDATLENENCERR                             = 19023,
    LWSWITCH_ERR_HW_LWLTLC_RSVADDRTYPEERR                              = 19024,
    LWSWITCH_ERR_HW_LWLTLC_RSVRSPSTATUSERR                             = 19025,
    LWSWITCH_ERR_HW_LWLTLC_RSVPKTSTATUSERR                             = 19026,
    LWSWITCH_ERR_HW_LWLTLC_RSVCACHEATTRPROBEREQERR                     = 19027,
    LWSWITCH_ERR_HW_LWLTLC_RSVCACHEATTRPROBERSPERR                     = 19028,
    LWSWITCH_ERR_HW_LWLTLC_DATLENGTATOMICREQMAXERR                     = 19029,
    LWSWITCH_ERR_HW_LWLTLC_DATLENGTRMWREQMAXERR                        = 19030,
    LWSWITCH_ERR_HW_LWLTLC_DATLENLTATRRSPMINERR                        = 19031,
    LWSWITCH_ERR_HW_LWLTLC_ILWALIDCACHEATTRPOERR                       = 19032,
    LWSWITCH_ERR_HW_LWLTLC_ILWALIDCRERR                                = 19033,
    LWSWITCH_ERR_HW_LWLTLC_RXRESPSTATUSTARGETERR                       = 19034,
    LWSWITCH_ERR_HW_LWLTLC_RXRESPSTATUSUNSUPPORTEDREQUESTERR           = 19035,
    LWSWITCH_ERR_HW_LWLTLC_RXHDROVFERR                                 = 19036,
    LWSWITCH_ERR_HW_LWLTLC_RXDATAOVFERR                                = 19037,
    LWSWITCH_ERR_HW_LWLTLC_STOMPDETERR                                 = 19038,
    LWSWITCH_ERR_HW_LWLTLC_RXPOISONERR                                 = 19039,
    LWSWITCH_ERR_HW_LWLTLC_CORRECTABLEINTERNALERR                      = 19040,
    LWSWITCH_ERR_HW_LWLTLC_RXUNSUPVCOVFERR                             = 19041,
    LWSWITCH_ERR_HW_LWLTLC_RXUNSUPLWLINKCREDITRELERR                   = 19042,
    LWSWITCH_ERR_HW_LWLTLC_RXUNSUPNCISOCCREDITRELERR                   = 19043,
    LWSWITCH_ERR_HW_LWLTLC_RX_PACKET_HEADER                            = 19044,
    LWSWITCH_ERR_HW_LWLTLC_RX_ERR_HEADER                               = 19045,
    LWSWITCH_ERR_HW_LWLTLC_TX_SYS_NCISOC_PARITY_ERR                    = 19046,
    LWSWITCH_ERR_HW_LWLTLC_TX_SYS_NCISOC_HDR_ECC_DBE_ERR               = 19047,
    LWSWITCH_ERR_HW_LWLTLC_TX_SYS_NCISOC_DAT_ECC_DBE_ERR               = 19048,
    LWSWITCH_ERR_HW_LWLTLC_TX_SYS_NCISOC_ECC_LIMIT_ERR                 = 19049,
    LWSWITCH_ERR_HW_LWLTLC_TX_SYS_TXRSPSTATUS_HW_ERR                   = 19050,
    LWSWITCH_ERR_HW_LWLTLC_TX_SYS_TXRSPSTATUS_UR_ERR                   = 19051,
    LWSWITCH_ERR_HW_LWLTLC_TX_SYS_TXRSPSTATUS_PRIV_ERR                 = 19052,
    LWSWITCH_ERR_HW_LWLTLC_RX_SYS_NCISOC_PARITY_ERR                    = 19053,
    LWSWITCH_ERR_HW_LWLTLC_RX_SYS_HDR_RAM_ECC_DBE_ERR                  = 19054,
    LWSWITCH_ERR_HW_LWLTLC_RX_SYS_HDR_RAM_ECC_LIMIT_ERR                = 19055,
    LWSWITCH_ERR_HW_LWLTLC_RX_SYS_DAT0_RAM_ECC_DBE_ERR                 = 19056,
    LWSWITCH_ERR_HW_LWLTLC_RX_SYS_DAT0_RAM_ECC_LIMIT_ERR               = 19057,
    LWSWITCH_ERR_HW_LWLTLC_RX_SYS_DAT1_RAM_ECC_DBE_ERR                 = 19058,
    LWSWITCH_ERR_HW_LWLTLC_RX_SYS_DAT1_RAM_ECC_LIMIT_ERR               = 19059,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_CREQ_RAM_HDR_ECC_DBE_ERR             = 19060,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_CREQ_RAM_DAT_ECC_DBE_ERR             = 19061,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_CREQ_RAM_ECC_LIMIT_ERR               = 19062,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_RSP_RAM_HDR_ECC_DBE_ERR              = 19063,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_RSP_RAM_DAT_ECC_DBE_ERR              = 19064,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_RSP_RAM_ECC_LIMIT_ERR                = 19065,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_COM_RAM_HDR_ECC_DBE_ERR              = 19066,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_COM_RAM_DAT_ECC_DBE_ERR              = 19067,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_COM_RAM_ECC_LIMIT_ERR                = 19068,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_RSP1_RAM_HDR_ECC_DBE_ERR             = 19069,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_RSP1_RAM_DAT_ECC_DBE_ERR             = 19070,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_RSP1_RAM_ECC_LIMIT_ERR               = 19071,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC0                      = 19072,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC1                      = 19073,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC2                      = 19074,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC3                      = 19075,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC4                      = 19076,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC5                      = 19077,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC6                      = 19078,
    LWSWITCH_ERR_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC7                      = 19079,
    LWSWITCH_ERR_HW_LWLTLC_RX_LNK_RXRSPSTATUS_HW_ERR                   = 19080,
    LWSWITCH_ERR_HW_LWLTLC_RX_LNK_RXRSPSTATUS_UR_ERR                   = 19081,
    LWSWITCH_ERR_HW_LWLTLC_RX_LNK_RXRSPSTATUS_PRIV_ERR                 = 19082,
    LWSWITCH_ERR_HW_LWLTLC_RX_LNK_ILWALID_COLLAPSED_RESPONSE_ERR       = 19083,
    LWSWITCH_ERR_HW_LWLTLC_RX_LNK_AN1_HEARTBEAT_TIMEOUT_ERR            = 19084,
    LWSWITCH_ERR_HW_LWLTLC_LAST, /* NOTE: Must be last */

    /* DLPL: errors ( SL1 errors too) */
    LWSWITCH_ERR_HW_DLPL                                               = 20000,
    LWSWITCH_ERR_HW_DLPL_TX_REPLAY                                     = 20001,
    LWSWITCH_ERR_HW_DLPL_TX_RECOVERY_SHORT                             = 20002,
    LWSWITCH_ERR_HW_DLPL_TX_RECOVERY_LONG                              = 20003,
    LWSWITCH_ERR_HW_DLPL_TX_FAULT_RAM                                  = 20004,
    LWSWITCH_ERR_HW_DLPL_TX_FAULT_INTERFACE                            = 20005,
    LWSWITCH_ERR_HW_DLPL_TX_FAULT_SUBLINK_CHANGE                       = 20006,
    LWSWITCH_ERR_HW_DLPL_RX_FAULT_SUBLINK_CHANGE                       = 20007,
    LWSWITCH_ERR_HW_DLPL_RX_FAULT_DL_PROTOCOL                          = 20008,
    LWSWITCH_ERR_HW_DLPL_RX_SHORT_ERROR_RATE                           = 20009,
    LWSWITCH_ERR_HW_DLPL_RX_LONG_ERROR_RATE                            = 20010,
    LWSWITCH_ERR_HW_DLPL_RX_ILA_TRIGGER                                = 20011,
    LWSWITCH_ERR_HW_DLPL_RX_CRC_COUNTER                                = 20012,
    LWSWITCH_ERR_HW_DLPL_LTSSM_FAULT                                   = 20013,
    LWSWITCH_ERR_HW_DLPL_LTSSM_PROTOCOL                                = 20014,
    LWSWITCH_ERR_HW_DLPL_MINION_REQUEST                                = 20015,
    LWSWITCH_ERR_HW_DLPL_FIFO_DRAIN_ERR                                = 20016,
    LWSWITCH_ERR_HW_DLPL_CONST_DET_ERR                                 = 20017,
    LWSWITCH_ERR_HW_DLPL_OFF2SAFE_LINK_DET_ERR                         = 20018,
    LWSWITCH_ERR_HW_DLPL_SAFE2NO_LINK_DET_ERR                          = 20019,
    LWSWITCH_ERR_HW_DLPL_SCRAM_LOCK_ERR                                = 20020,
    LWSWITCH_ERR_HW_DLPL_SYM_LOCK_ERR                                  = 20021,
    LWSWITCH_ERR_HW_DLPL_SYM_ALIGN_END_ERR                             = 20022,
    LWSWITCH_ERR_HW_DLPL_FIFO_SKEW_ERR                                 = 20023,
    LWSWITCH_ERR_HW_DLPL_TRAIN2SAFE_LINK_DET_ERR                       = 20024,
    LWSWITCH_ERR_HW_DLPL_HS2SAFE_LINK_DET_ERR                          = 20025,
    LWSWITCH_ERR_HW_DLPL_FENCE_ERR                                     = 20026,
    LWSWITCH_ERR_HW_DLPL_SAFE_NO_LD_ERR                                = 20027,
    LWSWITCH_ERR_HW_DLPL_E2SAFE_LD_ERR                                 = 20028,
    LWSWITCH_ERR_HW_DLPL_RC_RXPWR_ERR                                  = 20029,
    LWSWITCH_ERR_HW_DLPL_RC_TXPWR_ERR                                  = 20030,
    LWSWITCH_ERR_HW_DLPL_RC_DEADLINE_ERR                               = 20031,
    LWSWITCH_ERR_HW_DLPL_TX_HS2LP_ERR                                  = 20032,
    LWSWITCH_ERR_HW_DLPL_RX_HS2LP_ERR                                  = 20033,
    LWSWITCH_ERR_HW_DLPL_LTSSM_FAULT_UP                                = 20034,
    LWSWITCH_ERR_HW_DLPL_LTSSM_FAULT_DOWN                              = 20035,
    LWSWITCH_ERR_HW_DLPL_PHY_A                                         = 20036,
    LWSWITCH_ERR_HW_DLPL_TX_PL_ERROR                                   = 20037,
    LWSWITCH_ERR_HW_DLPL_RX_PL_ERROR                                   = 20038,
    LWSWITCH_ERR_HW_DLPL_LAST, /* NOTE: Must be last */

    /* AFS: errors */
    LWSWITCH_ERR_HW_AFS                                                = 21000,
    LWSWITCH_ERR_HW_AFS_UC_INGRESS_CREDIT_OVERFLOW                     = 21001,
    LWSWITCH_ERR_HW_AFS_UC_INGRESS_CREDIT_UNDERFLOW                    = 21002,
    LWSWITCH_ERR_HW_AFS_UC_EGRESS_CREDIT_OVERFLOW                      = 21003,
    LWSWITCH_ERR_HW_AFS_UC_EGRESS_CREDIT_UNDERFLOW                     = 21004,
    LWSWITCH_ERR_HW_AFS_UC_INGRESS_NON_BURSTY_PKT_DETECTED             = 21005,
    LWSWITCH_ERR_HW_AFS_UC_INGRESS_NON_STICKY_PKT_DETECTED             = 21006,
    LWSWITCH_ERR_HW_AFS_UC_INGRESS_BURST_GT_17_DATA_VC_DETECTED        = 21007,
    LWSWITCH_ERR_HW_AFS_UC_INGRESS_BURST_GT_1_NONDATA_VC_DETECTED      = 21008,
    LWSWITCH_ERR_HW_AFS_UC_ILWALID_DST                                 = 21009,
    LWSWITCH_ERR_HW_AFS_UC_PKT_MISROUTE                                = 21010,
    LWSWITCH_ERR_HW_AFS_LAST, /* NOTE: Must be last */

    /* MINION: errors */
    LWSWITCH_ERR_HW_MINION                                             = 22000,
    LWSWITCH_ERR_HW_MINION_UCODE_IMEM                                  = 22001,
    LWSWITCH_ERR_HW_MINION_UCODE_DMEM                                  = 22002,
    LWSWITCH_ERR_HW_MINION_HALT                                        = 22003,
    LWSWITCH_ERR_HW_MINION_BOOT_ERROR                                  = 22004,
    LWSWITCH_ERR_HW_MINION_TIMEOUT                                     = 22005,
    LWSWITCH_ERR_HW_MINION_DLCMD_FAULT                                 = 22006,
    LWSWITCH_ERR_HW_MINION_DLCMD_TIMEOUT                               = 22007,
    LWSWITCH_ERR_HW_MINION_DLCMD_FAIL                                  = 22008,
    LWSWITCH_ERR_HW_MINION_FATAL_INTR                                  = 22009,
    LWSWITCH_ERR_HW_MINION_WATCHDOG                                    = 22010,
    LWSWITCH_ERR_HW_MINION_EXTERR                                      = 22011,
    LWSWITCH_ERR_HW_MINION_FATAL_LINK_INTR                             = 22012,
    LWSWITCH_ERR_HW_MINION_NONFATAL                                    = 22013,
    LWSWITCH_ERR_HW_MINION_LAST, /* NOTE: Must be last */

    /* NXBAR errors */
    LWSWITCH_ERR_HW_NXBAR                                              = 23000,
    LWSWITCH_ERR_HW_NXBAR_TILE_INGRESS_BUFFER_OVERFLOW                 = 23001,
    LWSWITCH_ERR_HW_NXBAR_TILE_INGRESS_BUFFER_UNDERFLOW                = 23002,
    LWSWITCH_ERR_HW_NXBAR_TILE_EGRESS_CREDIT_OVERFLOW                  = 23003,
    LWSWITCH_ERR_HW_NXBAR_TILE_EGRESS_CREDIT_UNDERFLOW                 = 23004,
    LWSWITCH_ERR_HW_NXBAR_TILE_INGRESS_NON_BURSTY_PKT                  = 23005,
    LWSWITCH_ERR_HW_NXBAR_TILE_INGRESS_NON_STICKY_PKT                  = 23006,
    LWSWITCH_ERR_HW_NXBAR_TILE_INGRESS_BURST_GT_9_DATA_VC              = 23007,
    LWSWITCH_ERR_HW_NXBAR_TILE_INGRESS_PKT_ILWALID_DST                 = 23008,
    LWSWITCH_ERR_HW_NXBAR_TILE_INGRESS_PKT_PARITY_ERROR                = 23009,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_INGRESS_BUFFER_OVERFLOW              = 23010,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_INGRESS_BUFFER_UNDERFLOW             = 23011,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_EGRESS_CREDIT_OVERFLOW               = 23012,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_EGRESS_CREDIT_UNDERFLOW              = 23013,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_INGRESS_NON_BURSTY_PKT               = 23014,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_INGRESS_NON_STICKY_PKT               = 23015,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_INGRESS_BURST_GT_9_DATA_VC           = 23016,
    LWSWITCH_ERR_HW_NXBAR_TILEOUT_EGRESS_CDT_PARITY_ERROR              = 23017,
    LWSWITCH_ERR_HW_NXBAR_LAST, /* NOTE: Must be last */

    /* NPORT: SOURCETRACK errors */
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK                                         = 24000,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR     = 24001,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN0_TD_CRUMBSTORE_ECC_LIMIT_ERR  = 24002,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR     = 24003,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR       = 24004,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN0_TD_CRUMBSTORE_ECC_DBE_ERR    = 24005,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR       = 24006,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_SOURCETRACK_TIME_OUT_ERR                = 24007,
    LWSWITCH_ERR_HW_NPORT_SOURCETRACK_LAST, /* NOTE: Must be last */

    /* LWLIPT_LNK errors */
    LWSWITCH_ERR_HW_LWLIPT_LNK                                         = 25000,
    LWSWITCH_ERR_HW_LWLIPT_LNK_ILLEGALLINKSTATEREQUEST                 = 25001,
    LWSWITCH_ERR_HW_LWLIPT_LNK_FAILEDMINIONREQUEST                     = 25002,
    LWSWITCH_ERR_HW_LWLIPT_LNK_RESERVEDREQUESTVALUE                    = 25003,
    LWSWITCH_ERR_HW_LWLIPT_LNK_LINKSTATEWRITEWHILEBUSY                 = 25004,
    LWSWITCH_ERR_HW_LWLIPT_LNK_LINK_STATE_REQUEST_TIMEOUT              = 25005,
    LWSWITCH_ERR_HW_LWLIPT_LNK_WRITE_TO_LOCKED_SYSTEM_REG_ERR          = 25006,
    LWSWITCH_ERR_HW_LWLIPT_LNK_SLEEPWHILEACTIVELINK                    = 25007,
    LWSWITCH_ERR_HW_LWLIPT_LNK_RSTSEQ_PHYCTL_TIMEOUT                   = 25008,
    LWSWITCH_ERR_HW_LWLIPT_LNK_RSTSEQ_CLKCTL_TIMEOUT                   = 25009,
    LWSWITCH_ERR_HW_LWLIPT_LNK_LAST, /* Note: Must be last */

    /* SOE errors */
    LWSWITCH_ERR_HW_SOE                                                = 26000,
    LWSWITCH_ERR_HW_SOE_RESET                                          = 26001,
    LWSWITCH_ERR_HW_SOE_BOOTSTRAP                                      = 26002,
    LWSWITCH_ERR_HW_SOE_COMMAND_QUEUE                                  = 26003,
    LWSWITCH_ERR_HW_SOE_TIMEOUT                                        = 26004,
    LWSWITCH_ERR_HW_SOE_SHUTDOWN                                       = 26005,
    LWSWITCH_ERR_HW_SOE_HALT                                           = 26006,
    LWSWITCH_ERR_HW_SOE_EXTERR                                         = 26007,
    LWSWITCH_ERR_HW_SOE_WATCHDOG                                       = 26008,
    LWSWITCH_ERR_HW_SOE_LAST, /* Note: Must be last */

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    /* CCI errors */
    LWSWITCH_ERR_HW_CCI                                                = 27000,
    LWSWITCH_ERR_HW_CCI_RESET                                          = 27001,
    LWSWITCH_ERR_HW_CCI_INIT                                           = 27002,
    LWSWITCH_ERR_HW_CCI_TIMEOUT                                        = 27003,
    LWSWITCH_ERR_HW_CCI_SHUTDOWN                                       = 27004,
    LWSWITCH_ERR_HW_CCI_LAST, /* Note: Must be last */
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    /* Please update lwswitch_translate_hw_errors with a newly added error class. */
    LWSWITCH_ERR_LAST
    /* See enum modification guidelines at the top of this file */
} LWSWITCH_ERR_TYPE;

typedef enum lwswitch_pri_error_instance
{
    LWSWITCH_PBUS_PRI_SQUASH = 0, 
    LWSWITCH_PBUS_PRI_FECSERR,
    LWSWITCH_PBUS_PRI_TIMEOUT,
    LWSWITCH_PPRIV_WRITE_SYS,
    LWSWITCH_PPRIV_WRITE_PRT
} LWSWITCH_PRI_ERROR_INSTANCE;

typedef struct lwswitch_error
{
    LwU32  error_value;                 /* LWSWITCH_ERR_* */
    LwU32  error_src;                   /* LWSWITCH_ERROR_SRC_* */
    LwU32  instance;                    /* Used for link# or subengine instance */
    LwU32  subinstance;                 /* Used for lane# or similar */
    LW_DECLARE_ALIGNED(LwU64 time, 8);  /* Platform time (nsec) */
    LwBool error_resolved;              /* If an error is correctable, set to true. */
} LWSWITCH_ERROR;

#define LWSWITCH_ERROR_COUNT_SIZE 64

typedef struct lwswitch_get_errors
{
    LwU32          errorType;
    LwU64          errorIndex;
    LwU64          nextErrorIndex;
    LwU32          errorCount;
    LWSWITCH_ERROR error[LWSWITCH_ERROR_COUNT_SIZE];
} LWSWITCH_GET_ERRORS_PARAMS;

/*
 * CTRL_LWSWITCH_GET_INTERNAL_LATENCY
 *
 * Control for querying latency bins.
 *
 * Parameters:
 *   vc_selector [IN]
 *      A valid VC number returned by LWSWITCH_GET_INFO.
 *
 *   elapsed_time_msec [OUT]
 *      Elapsed time since the latency bins were queried.
 *   egressHistogram [OUT]
 *      Latency bin data/histogram format. The data will be available for the
 *      enabled/supported ports returned by LWSWITCH_GET_INFO.
 */

#define LWSWITCH_MAX_PORTS 64

/* TODO: describe the format */
typedef struct lwswitch_internal_latency_bins
{
    LW_DECLARE_ALIGNED(LwU64 low,    8);
    LW_DECLARE_ALIGNED(LwU64 medium, 8);
    LW_DECLARE_ALIGNED(LwU64 high,   8);
    LW_DECLARE_ALIGNED(LwU64 panic,  8);
    LW_DECLARE_ALIGNED(LwU64 count,  8);
} LWSWITCH_INTERNAL_LATENCY_BINS;

typedef struct lwswitch_get_internal_latency
{
    LwU32                          vc_selector;
    LW_DECLARE_ALIGNED(LwU64 elapsed_time_msec, 8);
    LWSWITCH_INTERNAL_LATENCY_BINS egressHistogram[LWSWITCH_MAX_PORTS];
} LWSWITCH_GET_INTERNAL_LATENCY;

/*
 * CTRL_LWSWITCH_SET_LATENCY_BINS
 *
 * Control for setting latency bins.
 *
 * Parameters:
 *   LWSWITCH_LATENCY_BIN [IN]
 *     Latency bin thresholds. The thresholds would be only applied to the
 *     enabled ports and the supported VCs by those ports.
 *     LWSWITCH_GET_INFO can be used to query enabled ports and supported VCs.
 */

#define LWSWITCH_MAX_VCS 8

/* TODO: describe the format */
typedef struct lwswitch_latency_bin
{
    LwU32   lowThreshold;       /* in nsec */
    LwU32   medThreshold;       /* in nsec */
    LwU32   hiThreshold;        /* in nsec */

} LWSWITCH_LATENCY_BIN;

typedef struct lwswitch_set_latency_bins
{
    LWSWITCH_LATENCY_BIN bin[LWSWITCH_MAX_VCS];

} LWSWITCH_SET_LATENCY_BINS;

/*
 * CTRL_LWSWITCH_SET_SWITCH_PORT_CONFIG
 *
 * Control for setting device port configurations.
 *
 * Parameters:
 *    portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO.
 *   type [IN]
 *      A connection type. See LWSWITCH_CONNECTION_TYPE.
 *   requesterLinkID [IN]
 *      An unique port ID in the fabric.
 *   requesterLan [IN]
 *      A Lan Id.
 *   count [IN]
 *      Endpoint Count
 *   acCoupled [IN]
 *      Set true, if the port is AC coupled.
 *   enableVC1 [IN]
 *      Set true, if VC1 should be enabled for the port.
 */

typedef enum lwswitch_connection_type
{
    CONNECT_ACCESS_GPU = 0,
    CONNECT_ACCESS_CPU,
    CONNECT_TRUNK_SWITCH,
    CONNECT_ACCESS_SWITCH
    /* See enum modification guidelines at the top of this file */
} LWSWITCH_CONNECTION_TYPE;

typedef enum lwswitch_connection_count
{
    CONNECT_COUNT_512 = 0,
    CONNECT_COUNT_1024,
    CONNECT_COUNT_2048
    /* See enum modification guidelines at the top of this file */
} LWSWITCH_CONNECTION_COUNT;

typedef struct lwswitch_set_switch_port_config
{
    LwU32  portNum;
    LwU32  type;
    LwU32  requesterLinkID;
    LwU32  requesterLanID;
    LwU32  count;
    LwBool acCoupled;
    LwBool enableVC1;

} LWSWITCH_SET_SWITCH_PORT_CONFIG;

/*
 * CTRL_LWSWITCH_SET_GANGED_LINK_TABLE
 *
 * Control for setting ganged link tables.
 * This interface is only supported on architectures that report
 * _GET_INFO_INDEX_ARCH == SV10.  All others will return an error.
 *
 * Parameters:
 *    linkMask [IN]
 *      A valid link/port mask returned by the port masks returned by
 *      LWSWITCH_GET_INFO.
 *   entries [IN]
 *      The Ganged link entires. (TODO: Describe format)
 */

#define LWSWITCH_GANGED_LINK_TABLE_ENTRIES_MAX 256

typedef struct lwswitch_set_ganged_link_table
{
    LwU32 link_mask;
    LwU32 entries[LWSWITCH_GANGED_LINK_TABLE_ENTRIES_MAX];

} LWSWITCH_SET_GANGED_LINK_TABLE;

/*
 * CTRL_LWSWITCH_GET_LWLIPT_COUNTER
 *
 * Control for querying LWLIPT counters.
 *
 * Parameters:
 *    liptCounter [OUT]
 *      Port's TX/RX traffic data. The data will be available for the
 *      enabled/supported ports returned by LWSWITCH_GET_INFO.
 */

typedef struct lwswitch_lwlipt_counter
{
    LW_DECLARE_ALIGNED(LwU64 txCounter0, 8);
    LW_DECLARE_ALIGNED(LwU64 txCounter1, 8);
    LW_DECLARE_ALIGNED(LwU64 rxCounter0, 8);
    LW_DECLARE_ALIGNED(LwU64 rxCounter1, 8);

} LWSWITCH_LWLIPT_COUNTER;

typedef struct lwswitch_get_lwlipt_counters
{
    LWSWITCH_LWLIPT_COUNTER liptCounter[LWSWITCH_MAX_PORTS];

} LWSWITCH_GET_LWLIPT_COUNTERS;

/*
 * CTRL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG
 *
 * Control to set LWLIPT counter configuration.
 *
 * Parameters:
 *    linkMask [IN]
 *      A valid link/port mask returned by the port masks returned by
 *      LWSWITCH_GET_INFO.
 *    tx0/tx1/rx0/rx1 [IN]
 *      TX/RX link configurations.
 */

/* TODO: describe format */
typedef struct lwlipt_counter_config
{
    LwU32 ctrl_0;
    LwU32 ctrl_1;
    LwU32 req_filter;
    LwU32 rsp_filter;
    LwU32 misc_filter;
    LW_DECLARE_ALIGNED(LwU64 addr_filter, 8);
    LW_DECLARE_ALIGNED(LwU64 addr_mask,   8);

} LWLIPT_COUNTER_CONFIG;

typedef struct lwswitch_set_lwlipt_counter_config
{
    LW_DECLARE_ALIGNED(LwU64 link_mask, 8);
    LWLIPT_COUNTER_CONFIG tx0;
    LWLIPT_COUNTER_CONFIG tx1;
    LWLIPT_COUNTER_CONFIG rx0;
    LWLIPT_COUNTER_CONFIG rx1;

} LWSWITCH_SET_LWLIPT_COUNTER_CONFIG;

/*
 * CTRL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG
 *
 * Control to query LWLIPT counter configuration.
 *
 * Parameters:
 *    link [IN]
 *      A valid link/port returned by the port masks returned by
 *      LWSWITCH_GET_INFO.
 *
 *    tx0/tx1/rx0/rx1 [OUT]
 *      TX/RX link configurations for the provide port.
 */

typedef struct lwswitch_get_lwlipt_counter_config
{
    LwU32                 link;
    LWLIPT_COUNTER_CONFIG tx0;
    LWLIPT_COUNTER_CONFIG tx1;
    LWLIPT_COUNTER_CONFIG rx0;
    LWLIPT_COUNTER_CONFIG rx1;

} LWSWITCH_GET_LWLIPT_COUNTER_CONFIG;

/*
 * CTRL_LWSWITCH_GET_INGRESS_REQLINKID
 *
 * Control to query the ingress requestor link id.
 *
 * Parameters:
 *    portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *
 *    requesterLinkID [OUT]
 *      Ingress requestor link id for the provided port.
 */

typedef struct lwswitch_get_ingress_reqlinkid_params
{
    LwU32       portNum;
    LwU32       requesterLinkID;

} LWSWITCH_GET_INGRESS_REQLINKID_PARAMS;

/*
 * CTRL_LWSWITCH_UNREGISTER_LINK
 *
 * Control to unregister the request link (port). This ensures that the black-
 * listed link will not be initialized or trained by the driver.
 *
 * Parameters:
 *    portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 */

typedef struct lwswitch_unregister_link_params
{
    LwU32       portNum;

} LWSWITCH_UNREGISTER_LINK_PARAMS;

/*
 * CTRL_RESET_AND_DRAIN_LINKS
 *
 * Control to reset and drain the links. Resets LWLinks and ensures to drain
 * backed up traffic.
 *
 * Parameters:
 *    linkMask [IN]
 *      A mask of link(s) to be reset.
 *      For SV10, the linkMask must contain at least a link-pair (even-odd links).
 *
 * Returns:
 *     LWL_SUCCESS if there were no errors
 *    -LWL_BAD_PARAMS if input parameters are wrong.
 *    -LWL_ERR_ILWALID_STATE if other errors are present and a full-chip reset is required.
 *    -LWL_INITIALIZATION_TOTAL_FAILURE if NPORT initialization failed and a retry is required.
 */

typedef struct lwswitch_reset_and_drain_links_params
{
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);

} LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS;

/*
 * CTRL_LWSWITCH_GET_LWLINK_STATUS
 *
 *   enabledLinkMask
 *     This field specifies the mask of available links on this subdevice.
 *   linkInfo
 *     This structure stores the per-link status of different LWLink
 *     parameters. The link is identified using an index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

/*
 * LWSWITCH_LWLINK_DEVICE_INFO
 *
 * This structure stores information about the device to which this link is
 * associated
 *
 *   deviceIdFlags
 *      Bitmask that specifies which IDs are valid for the device
 *      Refer LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_* for possible values
 *      If LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_PCI is set, PCI
 *      information is valid
 *      If LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_UUID is set, UUID is
 *      valid
 *   domain, bus, device, function, pciDeviceId
 *      PCI information for the device
 *   deviceType
 *      Type of the device
 *      See LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_* for possible values
 *   deviceUUID
 *      This field specifies the device UUID of the device. Useful for
 *      identifying the device (or version)
 */

typedef struct
{
    // ID Flags
    LwU32  deviceIdFlags;

    // PCI Information
    LwU32  domain;
    LwU16  bus;
    LwU16  device;
    LwU16  function;
    LwU32  pciDeviceId;

    // Device Type
    LW_DECLARE_ALIGNED(LwU64 deviceType, 8);

    // Device UUID
    LwU8   deviceUUID[16];
} LWSWITCH_LWLINK_DEVICE_INFO;

#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS        31:0
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_NONE   (0x00000000)
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_PCI    (0x00000001)
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_UUID   (0x00000002)

#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_EBRIDGE    (0x00000000)
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_NPU        (0x00000001)
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_GPU        (0x00000002)
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_SWITCH     (0x00000003)
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_TEGRA      (0x00000004)
#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_NONE       (0x000000FF)

#define LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_UUID_ILWALID    (0xFFFFFFFF)

/*
 * LWSWITCH_LWLINK_LINK_STATUS_INFO
 *
 * This structure stores the per-link status of different LWLink parameters.
 *
 *   capsTbl
 *     This is bit field for getting different global caps. The individual
 *     bitfields are specified by LWSWITCH_LWLINK_CAPS_*
 *   phyType
 *     This field specifies the type of PHY (LWHS or GRS) being used for this
 *     link.
 *   subLinkWidth
 *     This field specifies the no. of lanes per sublink.
 *   linkState
 *     This field specifies the current state of the link. See
 *     LWSWITCH_GET_LWLINK_STATUS_LINK_STATE_* for possible values.
 *   linkPowerState
 *     This field specifies the current power state of the link. See
 *     LWSWITCH_LWLINK_STATUS_LINK_POWER_STATE_* for possible values.
 *   rxSublinkStatus
 *     This field specifies the current state of RX sublink. See
 *     LWSWITCH_GET_LWLINK_STATUS_SUBLINK_RX_STATE_* for possible values.
 *   txSublinkStatus
 *     This field specifies the current state of TX sublink. See
 *     LWSWITCH_GET_LWLINK_STATUS_SUBLINK_TX_STATE_* for possible values.
 *   lwlinkVersion
 *     This field specifies the LWLink version supported by the link.
 *   nciVersion
 *     This field specifies the NCI version supported by the link.
 *   phyVersion
 *     This field specifies the version of PHY being used by the link.
 *   lwlinkCommonClockSpeed
 *     This field gives the value of lwlink common clock.
 *   lwlinkRefClkSpeed
 *     This field gives the value of lwlink refclk clock.
 *   lwlinkRefClkType
 *     This field specifies whether refclk is taken from LWHS reflck or PEX
 *     refclk for the current GPU. See LWSWITCH_LWLINK_REFCLK_TYPE_ILWALID*
 *     for possible values.
 *   lwlinkLinkClock
 *     This field gives the actual clock/speed at which links is running.
 *   connected
 *     This field specifies if any device is connected on the other end of the
 *     link
 *   loopProperty
 *     This field specifies if the link is a loopback/loopout link. See
 *     LWSWITCH_LWLINK_STATUS_LOOP_PROPERTY_* for possible values.
 *   laneRxdetStatusMask
 *     This field reports the per-lane RX Detect status provided by MINION.
 *   remoteDeviceLinkNumber
 *     This field specifies the link number on the remote end of the link
 *   remoteDeviceInfo
 *     This field stores the device information for the remote end of the link
 *
 */

typedef struct
{
    // Top level capablilites
    LwU32   capsTbl;

    LwU8    phyType;
    LwU8    subLinkWidth;

    // Link and sublink states
    LwU32   linkState;
    LwU32   linkPowerState;
    LwU8    rxSublinkStatus;
    LwU8    txSublinkStatus;

    // Indicates that lane reveral is in effect on this link.
    LwBool  bLaneReversal;

    LwU8    lwlinkVersion;
    LwU8    nciVersion;
    LwU8    phyVersion;

    // Clock information

    // These are being deprecated, please use HW Consistent terminology below
    LwU32   lwlinkLinkClockKHz;
    LwU32   lwlinkCommonClockSpeedKHz;
    LwU32   lwlinkRefClkSpeedKHz;
    LwU32   lwlinkCommonClockSpeedMhz;

    // HW consistent terminology
    LwU32   lwlinkLineRateMbps;
    LwU32   lwlinkLinkDataRateKiBps;
    LwU32   lwlinkLinkClockMhz;
    LwU32   lwlinkRefClkSpeedMhz;
    LwU8    lwlinkRefClkType;

    // Connection information
    LwBool  connected;
    LwU8    loopProperty;
    LwU8    remoteDeviceLinkNumber;
    LwU8    localDeviceLinkNumber;

    //
    // Added as part of LwLink 3.0
    // Note: SID has link info appended to it when provided by minion
    //
    LW_DECLARE_ALIGNED(LwU64 remoteLinkSid, 8);
    LW_DECLARE_ALIGNED(LwU64 localLinkSid,  8);

    // LR10+ only
    LwU32   laneRxdetStatusMask;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // LS10+ only
    LwBool  bIsRepeaterMode;
#endif

    LWSWITCH_LWLINK_DEVICE_INFO remoteDeviceInfo;
    LWSWITCH_LWLINK_DEVICE_INFO localDeviceInfo;
} LWSWITCH_LWLINK_LINK_STATUS_INFO;

/* LWLink link states */
#define LWSWITCH_LWLINK_STATUS_LINK_STATE_INIT               (0x00000000)
#define LWSWITCH_LWLINK_STATUS_LINK_STATE_HWCFG              (0x00000001)
#define LWSWITCH_LWLINK_STATUS_LINK_STATE_SWCFG              (0x00000002)
#define LWSWITCH_LWLINK_STATUS_LINK_STATE_ACTIVE             (0x00000003)
#define LWSWITCH_LWLINK_STATUS_LINK_STATE_FAULT              (0x00000004)
#define LWSWITCH_LWLINK_STATUS_LINK_STATE_RECOVERY           (0x00000006)
#define LWSWITCH_LWLINK_STATUS_LINK_STATE_ILWALID            (0xFFFFFFFF)

/* LWLink link power states */
#define LWSWITCH_LWLINK_STATUS_LINK_POWER_STATE_L0           (0x00000000)
#define LWSWITCH_LWLINK_STATUS_LINK_POWER_STATE_L1           (0x00000001)
#define LWSWITCH_LWLINK_STATUS_LINK_POWER_STATE_ILWALID      (0xFFFFFFFF)

/* LWLink Tx sublink states */
#define LWSWITCH_LWLINK_STATUS_SUBLINK_RX_STATE_HIGH_SPEED_1 (0x00000000)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_RX_STATE_SINGLE_LANE  (0x00000004)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_RX_STATE_TRAINING     (0x00000005)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_RX_STATE_SAFE_MODE    (0x00000006)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_RX_STATE_OFF          (0x00000007)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_RX_STATE_ILWALID      (0x000000FF)

/* LWLink Rx sublink states */
#define LWSWITCH_LWLINK_STATUS_SUBLINK_TX_STATE_HIGH_SPEED_1 (0x00000000)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_TX_STATE_SINGLE_LANE  (0x00000004)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_TX_STATE_TRAINING     (0x00000005)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_TX_STATE_SAFE_MODE    (0x00000006)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_TX_STATE_OFF          (0x00000007)
#define LWSWITCH_LWLINK_STATUS_SUBLINK_TX_STATE_ILWALID      (0x000000FF)

#define LWSWITCH_LWLINK_STATUS_PHY_LWHS                      (0x00000001)
#define LWSWITCH_LWLINK_STATUS_PHY_GRS                       (0x00000002)
#define LWSWITCH_LWLINK_STATUS_PHY_ILWALID                   (0x000000FF)

/* Version information */
#define LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_1_0            (0x00000001)
#define LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_2_0            (0x00000002)
#define LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_2_2            (0x00000004)
#define LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_3_0            (0x00000005)
#define LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_3_1            (0x00000006)
#define LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_4_0            (0x00000007)
#define LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_ILWALID        (0x000000FF)

#define LWSWITCH_LWLINK_STATUS_NCI_VERSION_1_0               (0x00000001)
#define LWSWITCH_LWLINK_STATUS_NCI_VERSION_2_0               (0x00000002)
#define LWSWITCH_LWLINK_STATUS_NCI_VERSION_2_2               (0x00000004)
#define LWSWITCH_LWLINK_STATUS_NCI_VERSION_3_0               (0x00000005)
#define LWSWITCH_LWLINK_STATUS_NCI_VERSION_3_1               (0x00000006)
#define LWSWITCH_LWLINK_STATUS_NCI_VERSION_4_0               (0x00000007)
#define LWSWITCH_LWLINK_STATUS_NCI_VERSION_ILWALID           (0x000000FF)

#define LWSWITCH_LWLINK_STATUS_LWHS_VERSION_1_0              (0x00000001)
#define LWSWITCH_LWLINK_STATUS_LWHS_VERSION_ILWALID          (0x000000FF)

#define LWSWITCH_LWLINK_STATUS_GRS_VERSION_1_0               (0x00000001)
#define LWSWITCH_LWLINK_STATUS_GRS_VERSION_ILWALID           (0x000000FF)

/* Connection properties */
#define LWSWITCH_LWLINK_STATUS_CONNECTED_TRUE                (0x00000001)
#define LWSWITCH_LWLINK_STATUS_CONNECTED_FALSE               (0x00000000)

#define LWSWITCH_LWLINK_STATUS_LOOP_PROPERTY_LOOPBACK        (0x00000001)
#define LWSWITCH_LWLINK_STATUS_LOOP_PROPERTY_LOOPOUT         (0x00000002)
#define LWSWITCH_LWLINK_STATUS_LOOP_PROPERTY_NONE            (0x00000000)

#define LWSWITCH_LWLINK_STATUS_REMOTE_LINK_NUMBER_ILWALID    (0x000000FF)

#define LWSWITCH_LWLINK_MAX_LINKS                            64

/* LWLink REFCLK types */
#define LWSWITCH_LWLINK_REFCLK_TYPE_ILWALID                  (0x00)
#define LWSWITCH_LWLINK_REFCLK_TYPE_LWHS                     (0x01)
#define LWSWITCH_LWLINK_REFCLK_TYPE_PEX                      (0x02)

typedef struct
{
    LW_DECLARE_ALIGNED(LwU64 enabledLinkMask, 8);
    LWSWITCH_LWLINK_LINK_STATUS_INFO linkInfo[LWSWITCH_LWLINK_MAX_LINKS];
} LWSWITCH_GET_LWLINK_STATUS_PARAMS;

/* List of supported capability type */
#define LWSWITCH_CAP_FABRIC_MANAGEMENT 0

/*
 * Max supported capabilities count.
 */
#define LWSWITCH_CAP_COUNT 1

/*
 * CTRL_LWSWITCH_ACQUIRE_CAPABILITY
 *
 * Upon success, user mode would acquire the requested capability
 * to perform privilege operations. This IOCTL will acquire one
 * capability at a time.
 *
 * Parameters:
 *   capDescriptor [IN]
 *      The OS file descriptor or handle representing the capability.
 *   cap [IN]
 *      The requested capability. One of the LWSWITCH_CAP_*.
 */
typedef struct
{
    /* input parameters */
    LW_DECLARE_ALIGNED(LwU64 capDescriptor, 8);
    LwU32 cap;


} LWSWITCH_ACQUIRE_CAPABILITY_PARAMS;

/*
 * CTRL_LWSWITCH_GET_TEMPERATURE
 *
 * Control to query temperature of Lwswitch sensors.
 *
 * The Temperatures are returned in FXP 24.8(LwTemp) format.
 *
 * Parameters:
 *   channelMask [IN]
 *      Mask of all the thermal channels queried.
 *   temperature [OUT]
 *     Temperature of the channel.
 *   status [OUT]
 *     Return status of the channel.
 */

#define  LWSWITCH_NUM_MAX_CHANNELS  16

typedef struct
{
    LwU32  channelMask;
    LwTemp temperature[LWSWITCH_NUM_MAX_CHANNELS];
    LwS32  status[LWSWITCH_NUM_MAX_CHANNELS];
} LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS;

#define LWSWITCH_CTRL_THERMAL_EVENT_ID_WARN 0
#define LWSWITCH_CTRL_THERMAL_EVENT_ID_OVERT 1

typedef struct
{
    LwU32  thermalEventId;
    LwTemp temperatureLimit;
} LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
/*
 * Limerock thermal channels (Locations)
 */
#define LWSWITCH_THERM_LOCATION_SV10_CENTER  0x0
#define LWSWITCH_NUM_CHANNELS_SV10             1
#endif

/*
 * Limerock thermal channels
 */
#define LWSWITCH_THERM_CHANNEL_LR10_TSENSE_MAX         0x00
#define LWSWITCH_THERM_CHANNEL_LR10_TSENSE_OFFSET_MAX  0x01
#define LWSWITCH_THERM_CHANNEL_LR10_TDIODE             0x02
#define LWSWITCH_THERM_CHANNEL_LR10_TDIODE_OFFSET      0x03
#define LWSWITCH_NUM_CHANNELS_LR10                        4

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
/*
 * Laguna Seca thermal channels
 */
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_MAX         0x00
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_OFFSET_MAX  0x01
#define LWSWITCH_THERM_CHANNEL_LS10_TDIODE             0x02
#define LWSWITCH_THERM_CHANNEL_LS10_TDIODE_OFFSET      0x03
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_0           0x04
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_1           0x05
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_2           0x06
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_3           0x07
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_4           0x08
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_5           0x09
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_6           0x0A
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_7           0x0B
#define LWSWITCH_THERM_CHANNEL_LS10_TSENSE_8           0x0C
#define LWSWITCH_NUM_CHANNELS_LS10                       13
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

/*
 * CTRL_LWSWITCH_GET_THROUGHPUT_COUNTERS
 *
 * Control for querying LWLINK throughput counters.
 *
 * Parameters:
 *    counterMask [IN]
 *      A mask of counter types.
 *      One of the LWSWITCH_THROUGHPUT_COUNTERS_TYPE_* macros
 *    linkMask [IN]
 *      A mask of desired link(s)
 *    counters [OUT]
 *      Fetched counter values
 */

/* LWLink throughput counter types */

/* Lwlink throughput counters reading data flits in TX */
#define LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_TX       (0x00000001)

/* Lwlink throughput counters reading data flits in RX */
#define LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_RX       (0x00000002)

/* Lwlink throughput counters reading all flits in TX */
#define LWSWITCH_THROUGHPUT_COUNTERS_TYPE_RAW_TX        (0x00000004)

/* Lwlink throughput counters reading all flits in RX */
#define LWSWITCH_THROUGHPUT_COUNTERS_TYPE_RAW_RX        (0x00000008)

#define LWSWITCH_THROUGHPUT_COUNTERS_TYPE_MAX           4

typedef struct lwswitch_throughput_values
{
    LwU64 values[LWSWITCH_THROUGHPUT_COUNTERS_TYPE_MAX];

} LWSWITCH_THROUGHPUT_COUNTER_VALUES;

typedef struct lwswitch_get_throughput_counters
{
    LwU16 counterMask;
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
    LWSWITCH_THROUGHPUT_COUNTER_VALUES counters[LWSWITCH_MAX_PORTS];

} LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS;

/*
 * CTRL_LWSWITCH_GET_BIOS_INFO
 *
 * Control call to get VBIOS information.
 *
 * Parameters:
 *     version [OUT]
 *       Vbios version in hex value.
 */
typedef struct lwswitch_get_bios_info
{
    LwU64 version;
} LWSWITCH_GET_BIOS_INFO_PARAMS;

/*
 * CTRL_LWSWITCH_BLACKLIST_DEVICE
 *
 * Control to Blacklist a device.  A blacklisted device will have
 * interrupts disabled, and opens/ioctls will fail.  If a device is 
 * blacklisted OOB then the setting is persistent.  If a device is 
 * blacklisted by the OS (such as module parameter) then the setting 
 * persists for the OS until the config file is changed and the driver 
 * reloaded. If a device is blacklisted by ioctl then the setting does
 * not persist across driver unload/reload.
 *
 * See BLACKLIST_REASON enum definition in interface/ioctl_common_lwswitch.h
 *
 * Parameters:
 *    deviceReason [IN]
 *      The reason the device is blacklisted
 */
typedef struct lwswitch_blacklist_device
{
    LWSWITCH_DEVICE_BLACKLIST_REASON deviceReason;
} LWSWITCH_BLACKLIST_DEVICE_PARAMS;

/*
 * CTRL_LWSWITCH_SET_FM_DRIVER_STATE
 *
 * Control to set the FM driver state for a device (heartbeat).
 *
 * Driver Fabric State is intended to reflect the state of the driver and
 * fabric manager.  Once FM sets the Driver State to CONFIGURED, it is
 * expected the FM will send heartbeat updates.  If the heartbeat is not
 * received before the session timeout, then the driver reports status
 * as MANAGER_TIMEOUT.  See also control device ioctl CTRL_LWSWITCH_GET_DEVICES_V2.
 *
 * See DRIVER_FABRIC_STATE enum definition in interface/ioctl_common_lwswitch.h
 *
 * Parameters:
 *    driverState [IN]
 *      The driver state for the device
 */
typedef struct lwswitch_set_fm_driver_state
{
    LWSWITCH_DRIVER_FABRIC_STATE driverState;
} LWSWITCH_SET_FM_DRIVER_STATE_PARAMS;

/*
 * CTRL_LWSWITCH_SET_DEVICE_FABRIC_STATE
 *
 * Control to set the device fabric state
 *
 * Device Fabric State reflects the fabric state of the lwswitch device.
 * FM sets the Device Fabric State to CONFIGURED once FM is managing the
 * device.
 *
 * See DEVICE_FABRIC_STATE enum definition in interface/ioctl_common_lwswitch.h
 *
 * Parameters:
 *    deviceState [IN]
 *      The device fabric state
 */
typedef struct lwswitch_set_device_fabric_state
{
    LWSWITCH_DEVICE_FABRIC_STATE deviceState;
} LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS;

/*
 * CTRL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT
 *
 * Control to set the FM session heartbeat timeout for a device
 *
 * If a device is managed by FM, and if a heartbeat is not received
 * by the FM_HEARTBEAT_TIMEOUT, then the driver reports Driver
 * Fabric State as MANAGER_TIMEOUT.
 *
 * LWSWITCH_DEFAULT_FM_HEARTBEAT_TIMEOUT_MSEC is the default timeout
 *
 * Parameters:
 *    fmTimeout [IN]
 *      The FM timeout value for the device, in milliseconds
 */
typedef struct lwswitch_set_fm_heartbeat_timeout
{
    LwU32 fmTimeout;
} LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT_PARAMS;
#define LWSWITCH_DEFAULT_FM_HEARTBEAT_TIMEOUT_MSEC (10*1000)

/*
 * CTRL_LWSWITCH_SET_LINK_ERROR_STATE_INFO
 *
 * Control to set bitmask info of the
 * link training error
 *
 * Parameters:
 *    attemptedTrainingMask0 [IN]
 *      Bitmask of links that have been
 *      attempted to train.
 *    trainingErrorMask0     [IN]
 *      Bitmaks of links that have an error
 *      during training.
 */
typedef struct lwswitch_set_training_error_info
{
    LwU64 attemptedTrainingMask0;
    LwU64 trainingErrorMask0;
} LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS;

#define LWSWITCH_DEVICE_EVENT_FATAL           0
#define LWSWITCH_DEVICE_EVENT_NONFATAL        1
#define LWSWITCH_DEVICE_EVENT_PORT_UP         2
#define LWSWITCH_DEVICE_EVENT_PORT_DOWN       3
#define LWSWITCH_DEVICE_EVENT_INBAND_DATA     4
#define LWSWITCH_DEVICE_EVENT_COUNT           5
#define LWSWITCH_REGISTER_EVENTS_MAX_EVENT_IDS (500)

/*
 * CTRL_LWSWITCH_REGISTER_EVENTS
 *
 * Control to register event IDs with an OS descriptor
 *
 * This control allows for clients to register one or more event IDs
 * with an OS descriptor. After registering event IDs, clients may poll
 * the OS descriptor for the registered event.
 *
 * Subsequent calls to register_event will overwrite lwrrently registered
 * event IDs. This allows the client to switch event polling as and when required.
 * Explicit unregister_events control call isn't necessary when the
 * client wishes to change the event types lwrrently being monitored.
 *
 * On Linux, only a single event ID can be registered to each
 * OS descriptor at a time. Calling this control with
 * numEvents > 1 on Linux will cause an error to be returned.
 *
 * On Windows, the osDescriptor field should be a valid
 * Windows EVENT handle.
 *
 * osDescriptor is unused on other operating systems.
 *
 * Parameters:
 *    eventIds [IN]
 *      A buffer of event IDs to register for
 *    numEvents [IN]
 *      Number of valid elements in eventIds
 *    osDescriptor [IN]
 *      OS event handle (Windows only)
 */
typedef struct lwswitch_register_events
{
    LwU32 eventIds[LWSWITCH_REGISTER_EVENTS_MAX_EVENT_IDS];
    LwU32 numEvents;
    void *osDescriptor;
} LWSWITCH_REGISTER_EVENTS_PARAMS;

/*
 * CTRL_LWSWITCH_UNREGISTER_EVENTS
 *
 * Control to unregister all event IDs from an OS descriptor
 *
 * This control unregisters all registered event IDs associated
 * with an OS descriptor.
 *
 * On Windows, the osDescriptor field should be a valid
 * Windows EVENT handle.
 *
 * osDescriptor is unused on other operating systems.
 *
 * Parameters:
 *    osDescriptor [IN]
 *      OS event handle (Windows only)
 */
typedef struct lwswitch_unregister_events
{
    void *osDescriptor;
} LWSWITCH_UNREGISTER_EVENTS_PARAMS;

/*
 * CTRL_LWSWITCH_GET_FATAL_ERROR_SCOPE
 *
 * Control to query if a fatal error is oclwrred on a port or device
 *
 * Parameters:
 *    device [OUT]
 *      Set to LW_TRUE if the lwswitch device encountered a fatal error
 *    port [OUT]
 *      An array of booleans indicating which ports
 *      encountered a fatal error
 */
typedef struct lwswitch_get_fatal_error_scope_params
{
    LwBool device;
    LwBool port[LWSWITCH_MAX_PORTS];
} LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/*
 * CTRL_LWSWITCH_SET_MC_RID_TABLE
 *
 * Control for programming an ingress multicast RID table entry.
 * This interface is only supported on LS10 architecture.  All others will
 * return an error. Architecture can be queried using _GET_INFO_INDEX_ARCH.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   index [IN]
 *      Index within the multicast RID table to be programmed. This is
 *      equivalent to MCID.
 *   extendedTable [IN]
 *      boolean: Set the requested entry in the extended table
 *      else set the requested entry in the main table
 *   ports [IN]
 *      The list of ports. For each multicast request, the address hash
 *      selects the multicast port string, and hardware multicasts to ports
 *      in that string.
 *   vcHop [IN]
 *      Array of VC hop values for each port.
 *   mcSize [IN]
 *      Number of ports in the multicast group (must be a nonzero value).
 *      Must be the number of ports in the main table, plus the extended table
 *      if that is used.
 *      Must be the same for all spray groups.
 *      Caller is responsible for ensuring the above conditions, as the driver
 *      provides only minimal range checking.
 *   numSprayGroups [IN]
 *      Number of groups to spray over. This must be a nonzero value.
 *   portsPerSprayGroup [IN]
 *      Array, number of ports contained in each spray group.
 *      Note these must all be the same size unless an extended entry
 *      is used,
 *      _and_ numSprayGroups is the same for both the main entry and extended
 *      entry,
 *      _and_ the sum of ports in the main and extended groups equals
 *      mcSize for each spray group.
 *      FM is responsible for providing the correct value. Driver provides only
 *      minimal range checking.
 *   replicaOffset [IN]
 *      Array, offsets within each spray group to the primary replica port for the group.
 *      The caller should specify mcSize primaryReplicas.
 *   replicaValid [IN]
 *      boolean:  Array, set the primary replica according to the replicaOffset array.
 *      else let hardware choose a default primary replica port
 *   extendedPtr [IN]
 *      pointer to the extended table to append to the multicast table entry
 *      can only be valid in the main table entries
 *   extendedValid [IN]
 *      boolean: Use the extended index to append to the main table string.
 *      else the main string specifies the complete operation for its MCID
 *   noDynRsp [IN]
 *      boolean: no dynamic alt selection on MC responses. This field has no meaning in
 *      the extended table
 *   entryValid
 *      boolean: flag this entry in the MC RID table as valid
 */

#define LWSWITCH_MC_MAX_PORTS           64
#define LWSWITCH_MC_MAX_SPRAYGROUPS     16

#define LWSWITCH_MC_VCHOP_PASS          0
#define LWSWITCH_MC_VCHOP_ILWERT        1
#define LWSWITCH_MC_VCHOP_FORCE0        2
#define LWSWITCH_MC_VCHOP_FORCE1        3

typedef struct lwswitch_set_mc_rid_table_params
{
    LwU32                           portNum;
    LwU32                           index;
    LwBool                          extendedTable;
    LwU32                           ports[LWSWITCH_MC_MAX_PORTS];
    LwU8                            vcHop[LWSWITCH_MC_MAX_PORTS];
    LwU32                           mcSize;
    LwU32                           numSprayGroups;
    LwU32                           portsPerSprayGroup[LWSWITCH_MC_MAX_SPRAYGROUPS];
    LwU32                           replicaOffset[LWSWITCH_MC_MAX_SPRAYGROUPS];
    LwBool                          replicaValid[LWSWITCH_MC_MAX_SPRAYGROUPS];
    LwU32                           extendedPtr;
    LwBool                          extendedValid;
    LwBool                          noDynRsp;
    LwBool                          entryValid;
} LWSWITCH_SET_MC_RID_TABLE_PARAMS;

/*
 * CTRL_LWSWITCH_GET_MC_RID_TABLE
 *
 * Control for reading an ingress multicast RID table entry.
 * This interface is only supported on LS10 architecture.  All others will
 * return an error. Architecture can be queried using _GET_INFO_INDEX_ARCH.
 *
 * Parameters:
 *   portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO
 *   index [IN]
 *      Index within the multicast RID table to be retrieved. This is
 *      equivalent to MCID.
 *   extendedTable [IN]
 *      boolean: Get the requested entry from the extended table.
 *      Else get the requested entry from the main table.
 *   ports [OUT]
 *      The list of ports. Port order within spray groups is not guaranteed
 *      to be preserved.
 *   vcHop [OUT]
 *      Array containing VC hop values for each entry in the ports array.
 *   mcSize [OUT]
 *      Number of ports in the multicast group.
 *   numSprayGroups [OUT]
 *      Number of groups to spray over.
 *   portsPerSprayGroup [OUT]
 *      Array, each element contains the number of ports within each corresponding
 *      spray group.
 *   replicaOffset [OUT]
 *      Array, offsets within each spray group to the primary replica port
 *      for the group.
 *   replicaValid [OUT]
 *      boolean:  Array, specifies whether each entry in the replicaOffset
 *      array is valid.
 *   extendedPtr [OUT]
 *      Pointer to the extended table appended to the main table entry.
 *      Only valid for main table entries.
 *   extendedValid [OUT]
 *      boolean: Whether the extendedPtr is valid.
 *   noDynRsp [IN]
 *      boolean: no dynamic alt selection on MC responses.
 *      This field has no meaning in the extended table.
 *   entryValid
 *      boolean: Whether this entry in the MC RID table is valid
 */

typedef struct lwswitch_get_mc_rid_table_params
{
    LwU32                           portNum;
    LwU32                           index;
    LwBool                          extendedTable;
    LwU32                           ports[LWSWITCH_MC_MAX_PORTS];
    LwU8                            vcHop[LWSWITCH_MC_MAX_PORTS];
    LwU32                           mcSize;
    LwU32                           numSprayGroups;
    LwU32                           portsPerSprayGroup[LWSWITCH_MC_MAX_SPRAYGROUPS];
    LwU32                           replicaOffset[LWSWITCH_MC_MAX_SPRAYGROUPS];
    LwBool                          replicaValid[LWSWITCH_MC_MAX_SPRAYGROUPS];
    LwU32                           extendedPtr;
    LwBool                          extendedValid;
    LwBool                          noDynRsp;
    LwBool                          entryValid;
} LWSWITCH_GET_MC_RID_TABLE_PARAMS;
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#define LWSWITCH_I2C_SMBUS_CMD_QUICK      0
#define LWSWITCH_I2C_SMBUS_CMD_BYTE       1
#define LWSWITCH_I2C_SMBUS_CMD_BYTE_DATA  2
#define LWSWITCH_I2C_SMBUS_CMD_WORD_DATA  3

/*
 * LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_BYTE_RW
 *
 * This structure provides data for the SMBUS Byte command.
 *
 * message [IN/OUT]
 *    8 Bit data message to read or write.
 */
typedef struct
{
    LwU8   message;
} LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_BYTE_RW;

/*
 * LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_BYTE_DATA_RW
 *
 * This structure provides data for the SMBUS Byte Data command.
 *
 * cmd [IN]
 *   SMBUS input command.
 * message [IN/OUT]
 *    8 Bit data message to read or write.
 */
typedef struct
{
    LwU8   cmd;
    LwU8   message;
} LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_BYTE_DATA_RW;

/*
 * LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_WORD_DATA_RW
 *
 * This structure provides data for the SMBUS Word Data command.
 *
 * cmd [IN]
 *   SMBUS input command.
 * message [IN/OUT]
 *    16 Bit data message to read or write.
 */
typedef struct
{
    LwU8   cmd;
    LwU16  message;
} LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_WORD_DATA_RW;

typedef union
{
    LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_BYTE_RW smbusByte;
    LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_BYTE_DATA_RW smbusByteData;
    LWSWITCH_I2C_TRANSACTION_DATA_SMBUS_WORD_DATA_RW smbusWordData;
} LWSWITCH_I2C_TRANSACTION_DATA;

/*
 * CTRL_LWSWITCH_I2C_SMBUS_COMMAND
 *
 * Control to issue SMBUS I2C transaction to an I2C device
 *
 * Parameters:
 *    deviceAddr [IN]
 *       The I2C Slave address to issue a transaction to. This is the unshifted,
 *       normal 7-bit address. For example, the input would be address 0x50 for
 *       device 0xA0.
 *    port [IN]
 *       The logical port/bus in which the I2C transaction is requested.
 *    cmdType [IN]
 *       The desired SMBUS command type. See LWSWITCH_I2C_SMBUS_CMD_*.
 *    bRead [IN]
 *       This field must be specified to indicate whether the
 *       command is a write (FALSE) or a read (TRUE).
 *    transactionData [IN/OUT]
 *       The LWSWITCH_I2C_TRANSACTION_DATA union to be filled out/read back
 *       depending on the SMBUS command type.
 */
typedef struct lwswitch_i2c_smbus_command_params
{
    LwU16  deviceAddr;
    LwU32  port;
    LwU8   cmdType;
    LwBool bRead;
    LWSWITCH_I2C_TRANSACTION_DATA transactionData;
} LWSWITCH_I2C_SMBUS_COMMAND_PARAMS;

/*
 * APIs for getting LWLink counters
 */

// These are the bitmask definitions for different counter types

#define LWSWITCH_LWLINK_COUNTER_ILWALID                      0x00000000

#define LWSWITCH_LWLINK_COUNTER_TL_TX0                       0x00000001
#define LWSWITCH_LWLINK_COUNTER_TL_TX1                       0x00000002
#define LWSWITCH_LWLINK_COUNTER_TL_RX0                       0x00000004
#define LWSWITCH_LWLINK_COUNTER_TL_RX1                       0x00000008

#define LWSWITCH_LWLINK_LP_COUNTERS_DL                       0x00000010

#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT           0x00010000

#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(i)      (1 << (i + 17))
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE     8
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0        0x00020000
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1        0x00040000
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2        0x00080000
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3        0x00100000
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4        0x00200000
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5        0x00400000
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6        0x00800000
#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7        0x01000000

#define LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY             0x02000000
#define LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY           0x04000000

#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY             0x08000000

#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_MASKED         0x10000000

#define LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_ECC_COUNTS         0x20000000

#define LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS             0x40000000
#define LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL             0x80000000

/*
 * Note that COUNTER_MAX_TYPES will need to be updated each time adding
 * a new counter type exceeds the existing value.
 *
 */
#define LWSWITCH_LWLINK_COUNTER_MAX_TYPES                    32

/*
 * CTRL_LWSWITCH_GET_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * [in] linkId
 *  This parameter specifies the TL link id/no for which we want to get
 *  counters for.
 *
 * [in]  counterMask
 *  This parameter specifies the input mask for desired counter types.
 *
 * [out] bTx0TlCounterOverflow
 *  This boolean is set to LW_TRUE if TX Counter 0 has rolled over.
 *
 * [out] bTx1TlCounterOverflow
 *  This boolean is set to LW_TRUE if TX Counter 1 has rolled over.
 *
 * [out] bRx0TlCounterOverflow
 *  This boolean is set to LW_TRUE if RX Counter 0 has rolled over.
 *
 * [out] bRx1TlCounterOverflow
 *  This boolean is set to LW_TRUE if RX Counter 1 has rolled over.
 *
 * [out] lwlinkCounters
 *  This array contains the error counts for each error type as requested from
 *  the counterMask. The array indexes correspond to the mask bits one-to-one.
 */

typedef struct
{
    LwU8   linkId;
    LwU32  counterMask;
    LwBool bTx0TlCounterOverflow;
    LwBool bTx1TlCounterOverflow;
    LwBool bRx0TlCounterOverflow;
    LwBool bRx1TlCounterOverflow;
    LW_DECLARE_ALIGNED(LwU64 lwlinkCounters[LWSWITCH_LWLINK_COUNTER_MAX_TYPES], 8);
} LWSWITCH_LWLINK_GET_COUNTERS_PARAMS;

/*
 * Structure to store the ECC error data.
 * valid
 *     Is the lane valid or not
 * eccErrorValue
 *     Value of the Error.
 * overflowed
 *     If the error overflowed or not
 */
typedef struct
{
    LwBool valid;
    LwU32  eccErrorValue;
    LwBool overflowed;
} LWSWITCH_LANE_ERROR;

/*
 * Structure to store ECC error data for Links
 * errorLane array index corresponds to the lane number.
 *
 * errorLane[]
 *    Stores the ECC error data per lane.
 */
typedef struct
{
    LWSWITCH_LANE_ERROR       errorLane[LWSWITCH_LWLINK_MAX_LANES];
    LwU32                     eccDecFailed;
    LwBool                    eccDecFailedOverflowed;
} LWSWITCH_LINK_ECC_ERROR;

/*
 * CTRL_GET_LWLINK_ECC_ERRORS
 *
 * Control to get the values of ECC ERRORS
 *
 * Parameters:
 *    linkMask [IN]
 *      Links on which the ECC error data requested
 *      A valid link/port mask returned by the port masks returned by
 *      LWSWITCH_GET_INFO
 *    errorLink[] [OUT]
 *      Stores the ECC error related information for each link.
 *      errorLink array index corresponds to the link Number.
 */

typedef struct lwswitch_get_lwlink_ecc_errors
{
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
    LWSWITCH_LINK_ECC_ERROR   errorLink[LWSWITCH_LWLINK_MAX_LINKS];
} LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/*
 * CTRL_LWSWITCH_CCI_CMIS_PRESENCE
 *
 * Control to get module presence bitmasks
 *
 * Parameters:
 *    cagesMask [OUT]
 *      Bitmask representing the CMIS module cages present 
 *      (associated with) the selected ASIC device
 *    modulesMask [OUT]  
 *      Bitmask representing the CMIS modules lwrrently present 
 *      (plugged in) on the selected ASIC device
 */

typedef struct lwswitch_cci_cmis_presence_params
{
      LwU32 cagesMask;
      LwU32 modulesMask;
} LWSWITCH_CCI_CMIS_PRESENCE_PARAMS;

#define LWSWITCH_CCI_CMIS_LWLINK_MAPPING_ENCODED_VALUE(i)           (7 + (i<<3)):(i<<3)
#define LWSWITCH_CCI_CMIS_LWLINK_MAPPING_ENCODED_VALUE_LINK_ID      5:0
#define LWSWITCH_CCI_CMIS_LWLINK_MAPPING_GET_OSFP_LANE_MASK(laneMask, linkId, eVal)                   \
    do {                                                                                              \
        LwU8 _byte, _lane;                                                                            \
                                                                                                      \
        laneMask = 0;                                                                                 \
        for (_lane = 0; _lane < 8; _lane++)                                                           \
        {                                                                                             \
            _byte = REF_VAL64(LWSWITCH_CCI_CMIS_LWLINK_MAPPING_ENCODED_VALUE(_lane), eVal);           \
            if (REF_VAL64(LWSWITCH_CCI_CMIS_LWLINK_MAPPING_ENCODED_VALUE_LINK_ID, _byte) == linkId)   \
            {                                                                                         \
                laneMask |= LWBIT(_lane);                                                             \
            }                                                                                         \
        }                                                                                             \
    } while (0);

/*
 * CTRL_LWSWITCH_CCI_CMIS_LWLINK_MAPPING
 *
 * Control to get cage to LWLink link mappings
 *
 * Provided macros should be used to extract fields from
 * encoded value.
 *
 * Parameters:
 *    cageIndex [IN]
 *      Target cage index (>=0 and <= 31) on the 
 *      selected ASIC device.
 *    linkMask [OUT]
 *      Mask of Lwlinks mapped to the given cage
 *    encodedValue [OUT]  
 *      Value that encodes the following:
 *      -Link Ids to OSFP lane number
 */
 
typedef struct lwswitch_cci_cmis_lwlink_mapping_params
{
    LwU8 cageIndex;
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);    
    LW_DECLARE_ALIGNED(LwU64 encodedValue, 8);
} LWSWITCH_CCI_CMIS_LWLINK_MAPPING_PARAMS;

#define LWSWITCH_CCI_CMIS_MEMORY_ACCESS_BUF_SIZE (128)

/*
 * CTRL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ
 *
 * Control for direct memory accesses to cages
 *
 * Parameters:
 *    cageIndex [IN]
 *      Target cage index (>=0 and <= 31) on the 
 *      selected ASIC device
 *    bank [IN]
 *      Target bank in module (if address >= 0x80)
 *    page [IN]
 *      Target page in module (if address >= 0x80)
 *    address [IN]
 *      Target byte address (offset) in module
 *    count [IN]
 *      Number of bytes to read (>=0 and <= 0x7F)
 *    data[] [OUT]  
 *      128 byte data buffer
 */

typedef struct lwswitch_cci_cmis_memory_access_read_params
{
      LwU8 cageIndex;
      LwU8 bank;
      LwU8 page;
      LwU8 address;
      LwU8 count;
      LwU8 data[LWSWITCH_CCI_CMIS_MEMORY_ACCESS_BUF_SIZE];    
} LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ_PARAMS;

/*
 * CTRL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE
 *
 * Control for direct memory accesses to cages
 *
 * Parameters:
 *    cageIndex [IN]
 *      Target cage index (>=0 and <= 31) on the 
 *      selected ASIC device
 *    bank [IN]
 *      Target bank in module (if address >= 0x80)
 *    page [IN]
 *      Target page in module (if address >= 0x80)
 *    address [IN]
 *      Target byte address (offset) in module
 *    count [IN]
 *      Number of bytes to write (>=0 and <= 0x7F)
 *    data[] [IN]  
 *      128 byte data buffer
 */

typedef struct lwswitch_cci_cmis_memory_access_write_params
{
      LwU8 cageIndex;
      LwU8 bank;
      LwU8 page;
      LwU8 address;
      LwU8 count;
      LwU8 data[LWSWITCH_CCI_CMIS_MEMORY_ACCESS_BUF_SIZE];    
} LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE_PARAMS;

#define LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING_LEN    31

/*
 * CTRL_LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING
 *
 * Control to get bezel information for a cage.
 *
 * Parameters:
 *    cageIndex [IN]
 *      Target cage index (>=0 and <= 31) on the 
 *      selected ASIC device.
 *    bezelMarking [OUT]
 *              
 */

typedef struct lwswitch_cci_cmis_cage_bezel_marking_params
{
    LwU8 cageIndex;
    char bezelMarking[LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING_LEN + 1];
} LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING_PARAMS;

#define LWSWITCH_CCI_XVCR_LANES     0x8

/*
 *
 * Structure to store cci grading values
 *
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   tx_init
 *     TX-Input Initial Tuning grading.
 *   rx_init
 *     RX-Input Initial Tuning grading.
 *   tx_maint
 *     TX-Input Maintenance grading.
 *   rx_maint
 *     RX-Input Maintenance grading.
 */
typedef struct lwswitch_cci_grading_values
{
    LwU8  tx_init[LWSWITCH_CCI_XVCR_LANES];
    LwU8  rx_init[LWSWITCH_CCI_XVCR_LANES];
    LwU8  tx_maint[LWSWITCH_CCI_XVCR_LANES];
    LwU8  rx_maint[LWSWITCH_CCI_XVCR_LANES];
} LWSWITCH_CCI_GRADING_VALUES;

/*
 * CTRL_LWSWITCH_CCI_GET_GRADING_VALUES
 *
 * Control to get cci xvcr grading values
 *
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   link [IN]
 *     Link number
 *   laneMask [OUT]
 *     Lane mask of valid indexes in the grading data
 *   grading [OUT]
 *     xvcr grading values
 */
typedef struct lwswitch_cci_get_grading_values_params
{
    LwU32 linkId;
    LwU8  laneMask;
    LWSWITCH_CCI_GRADING_VALUES grading;
} LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#define LWSWITCH_LWLINK_MAX_CORRECTABLE_ERROR_DAYS      5
#define LWSWITCH_LWLINK_MAX_CORRECTABLE_ERROR_MONTHS    3

typedef struct
{
    LwU32 lastUpdated;
    LwU32 flitCrcErrorsPerMinute;
    LwU32 laneCrcErrorsPerMinute[LWSWITCH_LWLINK_MAX_LANES];
} LWSWITCH_LWLINK_CORRECTABLE_ERROR_RATES;

#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_LWHS      0
#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_RESERVED  1
#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_OTHER     2
#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_ENTER    3
#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_NUM_TX_LP_EXIT     4
#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_COUNT_TX_SLEEP     5
#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_MAX_COUNTERS       6
/*
 * CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS
 *
 * Reads LWLINK low power counters for given linkId
 *
 * Parameters:
 *    linkId [IN]
 *      ID of the link to be queried
 *    counterValidMask [IN,OUT]
 *      Mask of valid counters
 *    counterValues [OUT]
 *      Low power counter values returned
 */
typedef struct lwswitch_get_lwlink_lp_counters_params
{
      LwU32 linkId;
      LwU32 counterValidMask;
      LwU32 counterValues[CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS_MAX_COUNTERS];
} LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS;

/*
 * CTRL_LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES
 *
 * This command queries recent correctable error rates for the given link.
 *
 * The error rates specify the maximum number of errors per minute recorded
 * for the given link within a 24-hour period for daily maximums or a 30-day
 * period for monthly maximums.
 *
 * Parameters:
 *    linkId [in]
 *      LWLink link ID
 *    dailyMaxCorrectableErrorRates[] [OUT]
 *      LWLink daily max correctable error rate array
 *    monthlyMaxCorrectableErrorRates[] [OUT]
 *      LWLink monthly max correctable error rate array
 */
 
typedef struct
{
    LwU8   linkId;
    LWSWITCH_LWLINK_CORRECTABLE_ERROR_RATES dailyMaxCorrectableErrorRates[LWSWITCH_LWLINK_MAX_CORRECTABLE_ERROR_DAYS];
    LWSWITCH_LWLINK_CORRECTABLE_ERROR_RATES monthlyMaxCorrectableErrorRates[LWSWITCH_LWLINK_MAX_CORRECTABLE_ERROR_MONTHS];
} LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS;

#define LWSWITCH_LWLINK_ERROR_READ_SIZE            128 //could not read the maximum size (721) of entries in one call

typedef enum
{
    LWSWITCH_LWLINK_NO_ERROR                                    = 0,

    //DL RX Fatal Counts
    LWSWITCH_LWLINK_ERR_DL_RX_FAULT_DL_PROTOCOL_FATAL           = 1000,
    LWSWITCH_LWLINK_ERR_DL_RX_FAULT_SUBLINK_CHANGE_FATAL,

    //DL RX Correctable Aclwmulated Counts
    LWSWITCH_LWLINK_ERR_DL_RX_FLIT_CRC_CORR, 
    LWSWITCH_LWLINK_ERR_DL_RX_LANE0_CRC_CORR,
    LWSWITCH_LWLINK_ERR_DL_RX_LANE1_CRC_CORR,
    LWSWITCH_LWLINK_ERR_DL_RX_LANE2_CRC_CORR,
    LWSWITCH_LWLINK_ERR_DL_RX_LANE3_CRC_CORR,
    LWSWITCH_LWLINK_ERR_DL_RX_LINK_REPLAY_EVENTS_CORR,

    //DL TX Fatal Counts
    LWSWITCH_LWLINK_ERR_DL_TX_FAULT_RAM_FATAL,
    LWSWITCH_LWLINK_ERR_DL_TX_FAULT_INTERFACE_FATAL,
    LWSWITCH_LWLINK_ERR_DL_TX_FAULT_SUBLINK_CHANGE_FATAL,

    //DL TX Correctable Aclwmulated Counts
    LWSWITCH_LWLINK_ERR_DL_TX_LINK_REPLAY_EVENTS_CORR,

    //DL NA Fatal Counts
    LWSWITCH_LWLINK_ERR_DL_LTSSM_FAULT_UP_FATAL,
    LWSWITCH_LWLINK_ERR_DL_LTSSM_FAULT_DOWN_FATAL,

    //DL NA Correctable Aclwmulated Counts
    LWSWITCH_LWLINK_ERR_DL_LINK_RECOVERY_EVENTS_CORR,

    //TLC RX Fatal Counts
    LWSWITCH_LWLINK_ERR_TLC_RX_DL_HDR_PARITY_ERR_FATAL          = 2000,
    LWSWITCH_LWLINK_ERR_TLC_RX_DL_DATA_PARITY_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_DL_CTRL_PARITY_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_AE_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_BE_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_ADDR_ALIGN_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_PKTLEN_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_PACKET_STATUS_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_CACHE_ATTR_PROBE_REQ_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_CACHE_ATTR_PROBE_RSP_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_DATLEN_GT_RMW_REQ_MAX_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_DATLEN_LT_ATR_RSP_MIN_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_CR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_COLLAPSED_RESPONSE_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_HDR_OVERFLOW_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_DATA_OVERFLOW_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_STOMP_DETECTED_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_CMD_ENC_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_DAT_LEN_ENC_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_PO_FOR_CACHE_ATTR_FATAL,

    //TLC RX Non-Fatal Counts
    LWSWITCH_LWLINK_ERR_TLC_RX_RSP_STATUS_HW_ERR_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_RSP_STATUS_UR_ERR_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_RSP_STATUS_PRIV_ERR_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_POISON_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_AN1_HEARTBEAT_TIMEOUT_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_ILLEGAL_PRI_WRITE_NONFATAL,
    
    //TLC RX Fatal Counts addendum
    LWSWITCH_LWLINK_ERR_TLC_RX_HDR_RAM_ECC_DBE_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_DAT0_RAM_ECC_DBE_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_RX_DAT1_RAM_ECC_DBE_FATAL,

    //TLC TX Fatal Counts
    LWSWITCH_LWLINK_ERR_TLC_TX_DL_CREDIT_PARITY_ERR_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_NCISOC_HDR_ECC_DBE_FATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_NCISOC_PARITY_ERR_FATAL,

    //TLC TX Non-Fatal Counts
    LWSWITCH_LWLINK_ERR_TLC_TX_ILLEGAL_PRI_WRITE_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC0_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC1_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC2_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC3_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC4_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC5_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC6_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC7_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_POISON_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_RSP_STATUS_HW_ERR_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_RSP_STATUS_UR_ERR_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_RSP_STATUS_PRIV_ERR_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_CREQ_DAT_RAM_ECC_DBE_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_RSP_DAT_RAM_ECC_DBE_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_COM_DAT_RAM_ECC_DBE_NONFATAL,
    LWSWITCH_LWLINK_ERR_TLC_TX_RSP1_DAT_RAM_ECC_DBE_FATAL,
    
    //LWLIPT Fatal Counts
    LWSWITCH_LWLINK_ERR_LWLIPT_SLEEP_WHILE_ACTIVE_LINK_FATAL    = 3000,
    LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_PHYCTL_TIMEOUT_FATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_CLKCTL_TIMEOUT_FATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_CLKCTL_ILLEGAL_REQUEST_FATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_PLL_TIMEOUT_FATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_PHYARB_TIMEOUT_FATAL,

    //LWLIPT Non-Fatal Counts
    LWSWITCH_LWLINK_ERR_LWLIPT_ILLEGAL_LINK_STATE_REQUEST_NONFATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_FAILED_MINION_REQUEST_NONFATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_RESERVED_REQUEST_VALUE_NONFATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_LINK_STATE_WRITE_WHILE_BUSY_NONFATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_WRITE_TO_LOCKED_SYSTEM_REG_NONFATAL,
    LWSWITCH_LWLINK_ERR_LWLIPT_LINK_STATE_REQUEST_TIMEOUT_NONFATAL,
} LWSWITCH_LWLINK_ERROR_TYPE;

typedef struct
{
    LwU8  instance;
    LwU32 error; //LWSWITCH_LWLINK_ERROR_TYPE
    LwU32 timeStamp;
    LwU64 count;
} LWSWITCH_LWLINK_ERROR_ENTRY;

/*
 * CTRL_LWSWITCH_GET_LWLINK_ERROR_COUNTS
 *
 * Control to get the LWLINK errors from inforom cache 
 *
 * Parameters:
 *    errorIndex [IN/OUT]
 *      On input: The index of the first LWLink error to retrieve from inforom cache
 *      On output: The index of the first error to retrieve after the previous call.
 *    errorCount [OUT]
 *      Number of errors returned by the call. Lwrrently, errorCount is limited
 *      by LWSWITCH_LWLINK_ERROR_READ_SIZE. In order to query all the errors, a
 *      client needs to keep calling the control till errorCount is zero.
 *    errorLog[] [OUT]
 *      LWLINK error array
 */

typedef struct
{
    LwU32 errorIndex;
    LwU32 errorCount;
    LWSWITCH_LWLINK_ERROR_ENTRY errorLog[LWSWITCH_LWLINK_ERROR_READ_SIZE];
} LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS;

#define LWSWITCH_ECC_ERRORS_MAX_READ_COUNT    128

typedef struct
{
    LwU32  sxid;
    LwU8   linkId;
    LwU32  lastErrorTimestamp;
    LwBool bAddressValid;
    LwU32  address;
    LwU32  correctedCount;
    LwU32  uncorrectedCount;
} LWSWITCH_ECC_ERROR_ENTRY;

/*
 * CTRL_LWSWITCH_GET_ECC_ERROR_COUNTS
 *
 * Control to get the ECC error counts and logs from inforom 
 *
 * Parameters:
 *    uncorrectedTotal [out]
 *      uncorrected ECC errors count
 *    correctedTotal [out]
 *      corrected ECC errors count
 *    errorCount [out]
 *      recorded error log count in the array
 *    errorLog[] [OUT]
 *      ECC errors array
 */

typedef struct
{
    LwU64 uncorrectedTotal;
    LwU64 correctedTotal;
    LwU32 errorCount;
    LWSWITCH_ECC_ERROR_ENTRY errorLog[LWSWITCH_ECC_ERRORS_MAX_READ_COUNT];
} LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS;

#define LWSWITCH_SXID_ENTRIES_NUM    10

typedef struct
{
    LwU32 sxid;
    LwU32 timestamp;
} LWSWITCH_SXID_ENTRY;

/*
 * CTRL_LWSWITCH_GET_SXIDS
 *
 * Control to get the LWSwitch SXID errors from inforom cache 
 *
 * Parameters:
 *    sxidCount [OUT]
 *      The total SXID error number
 *    sxidFirst [OUT]
 *      The array of the first LWSWITCH_SXID_ENTRIES_NUM (10) SXIDs
 *    sxidLast [OUT]
 *      The array of the last LWSWITCH_SXID_ENTRIES_NUM (10) SXIDs
 */

typedef struct
{
    LwU32 sxidCount;
    LWSWITCH_SXID_ENTRY sxidFirst[LWSWITCH_SXID_ENTRIES_NUM];
    LWSWITCH_SXID_ENTRY sxidLast[LWSWITCH_SXID_ENTRIES_NUM];
} LWSWITCH_GET_SXIDS_PARAMS;

/*
 * CTRL_LWSWITCH_GET_FOM_VALUES
 *   This command gives the FOM values to MODS
 *
 *  [in] linkId
 *    Link number on which the FOM values are requested
 *  [out] numLanes
 *    This field specifies the no. of lanes per link
 *  [out] figureOfMetritValues
 *    This field contains the FOM values per lane
 */

typedef struct lwswitch_get_fom_values_params
{
    LwU32 linkId;
    LwU8  numLanes;
    LwU16 figureOfMeritValues[LWSWITCH_LWLINK_MAX_LANES];
} LWSWITCH_GET_FOM_VALUES_PARAMS;

/*
 * CTRL_LWSWITCH_SET_RESIDENCY_BINS
 *
 * Control for setting residency bins.
 *
 * Parameters:
 *  [in] table_select
 *      Which table to return.
 *  [in] LWSWITCH_RESIDENCY_BIN
 *     Residency thresholds. The thresholds would be only applied to the
 *     enabled ports.
 *     LWSWITCH_GET_INFO can be used to query enabled ports.
 */

typedef struct lwswitch_residency_bin
{
    LwU32   lowThreshold;       /* in nsec */
    LwU32   hiThreshold;        /* in nsec */

} LWSWITCH_RESIDENCY_THRESHOLDS;

#define LWSWITCH_TABLE_SELECT_MULTICAST     0
#define LWSWITCH_TABLE_SELECT_REDUCTION     1

typedef struct lwswitch_set_residency_bins
{
    LwU32 table_select;     // LWSWITCH_TABLE_SELECT_MULTICAST/_REDUCTION
    LWSWITCH_RESIDENCY_THRESHOLDS bin;

} LWSWITCH_SET_RESIDENCY_BINS;

/*
 * CTRL_LWSWITCH_GET_RESIDENCY_BINS
 *
 * Control for querying multicast & reduction residency histogram.
 *
 * Parameters:
 *  [in] linkId
 *    Link number on which the residency histogram is requested
 *  [in] table_select
 *      Which table to return.
 *
 *  [in] bin
 *     Residency thresholds.
 *  [out] residency
 *      Residency data/histogram format. The data will be available for the
 *      enabled/supported ports returned by LWSWITCH_GET_INFO.
 */

typedef struct lwswitch_residency_bins
{
    LW_DECLARE_ALIGNED(LwU64 low,    8);
    LW_DECLARE_ALIGNED(LwU64 medium, 8);
    LW_DECLARE_ALIGNED(LwU64 high,   8);
} LWSWITCH_RESIDENCY_BINS;

#define LWSWITCH_RESIDENCY_SIZE     128

typedef struct lwswitch_get_residency_bins
{
    LwU32 link;
    LwU32 table_select;     // LWSWITCH_TABLE_SELECT_MULTICAST/_REDUCTION
    LWSWITCH_RESIDENCY_THRESHOLDS bin;
    LWSWITCH_RESIDENCY_BINS residency[LWSWITCH_RESIDENCY_SIZE];
} LWSWITCH_GET_RESIDENCY_BINS;

/*
 * CTRL_LWSWITCH_GET_RB_STALL_BUSY
 *
 * Control for querying reduction buffer stall/busy counters.
 *
 * Parameters:
 *  [in] linkId
 *    Link number on which the stall/busy counters are requested
 *  [in] table_select
 *      Which table to return.
 *
 *  [out] stall_busy
 *      Reduction buffer stall/busy counters. The data will be available for the
 *      enabled/supported ports returned by LWSWITCH_GET_INFO.
 */

typedef struct lwswitch_stall_busy
{
    LW_DECLARE_ALIGNED(LwU64 time,  8); // in ns
    LW_DECLARE_ALIGNED(LwU64 stall, 8);
    LW_DECLARE_ALIGNED(LwU64 busy,  8);
} LWSWITCH_STALL_BUSY;

typedef struct lwswitch_get_rd_stall_busy
{
    LwU32 link;
    LwU32 table_select;         // LWSWITCH_TABLE_SELECT_MULTICAST/_REDUCTION
    LWSWITCH_STALL_BUSY vc0;
    LWSWITCH_STALL_BUSY vc1;
} LWSWITCH_GET_RB_STALL_BUSY;

// TODO: To remove Macro once the "lwlink_inband_msg.h" is available.
#define LWSWITCH_INBAND_DATA_SIZE 4096

/*
 * CTRL_LWSWITCH_INBAND_SEND_DATA
 * 
 * Control call used for sending data over inband.
 *
 * Parameters:
 *
 *    dataSize[IN]
 *      Valid data in the buffer
 *
 *    linkId[IN]
 *      Link number on which the data needs to be sent
 *
 *    buffer[IN]
 *      Data which needs to be sent on the other side
 *
 *    dataSent [OUT]
 *      Bytes of data which were sent to the other side
 *
 *    status [OUT]
 *      status of the data send
 */
typedef struct lwswitch_inband_send_data_params
{
    /* input parameters */
    LwU32 dataSize;
    LwU32 linkId;
    LwU8  buffer[LWSWITCH_INBAND_DATA_SIZE];

    /* output parameters */
    LwU32 dataSent;
    LwU32 status;
} LWSWITCH_INBAND_SEND_DATA_PARAMS;

/*
 * CTRL_LWSWITCH_INBAND_READ_DATA
 * 
 * Control call used for reading data received over inband
 *
 * Parameters:
 *
 *    linkId[IN]
 *      Link number on which the data needs to be read.
 *
 *    dataSize[OUT]
 *      Valid data in the buffer
 *
 *    status [OUT]
 *      status of the data
 *
 *    buffer[OUT]
 *      Data which needs to be read from the other side
 */
typedef struct lwswitch_inband_read_data_params
{
    /* input parameters */
    LwU32 linkId;

    /* output parameters */
    LwU32 dataSize;
    LwU32 status;
    LwU8  buffer[LWSWITCH_INBAND_DATA_SIZE];
} LWSWITCH_INBAND_READ_DATA_PARAMS;

/*
 * CTRL_LWSWTICH_INBAND_FLUSH_DATA
 * 
 * Flushing all the pending data for the corresponding link.
 * Messages would be stored in a queue. If flush is send all the
 * pending messages which are there for that linkId will be deleted.
 *
 * Parameters:
 *
 *    linkMask[IN]
 *      Mask of Links on which the data needs to be flushed.
 *
 *    status [OUT]
 *      status of the flush
 */
typedef struct lwswitch_inband_flush_data_params
{
    /* input parameters */
    LwU64 linkMask;

    /* output parameters */
    LwU32 status;
} LWSWITCH_INBAND_FLUSH_DATA_PARAMS;

/*
 * CTRL_LWSWITCH_INBAND_PENDING_DATA_STATS
 * 
 * Control call to check which links have pending data
 *
 * Parameters:
 *
 *    linkMask[OUT]
 *      Mask of the links which has data on it.
 *
 *    status [OUT]
 *      status in providing the mask
 */
typedef struct lwswitch_inband_pending_data_stats_params
{
    /* output parameters */
    LwU64 linkMask;
    LwU32 status;
} LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS;

/*
 * CTRL call command list.
 *
 * Linux driver supports only 8-bit commands.
 *
 * See struct control call command  modification guidelines at the top
 * of this file.
 */
#define CTRL_LWSWITCH_GET_INFO                              0x01
#define CTRL_LWSWITCH_SET_SWITCH_PORT_CONFIG                0x02
#define CTRL_LWSWITCH_SET_INGRESS_REQUEST_TABLE             0x03
#define CTRL_LWSWITCH_SET_INGRESS_REQUEST_VALID             0x04
#define CTRL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE            0x05
#define CTRL_LWSWITCH_SET_GANGED_LINK_TABLE                 0x06
#define CTRL_LWSWITCH_GET_INTERNAL_LATENCY                  0x07
#define CTRL_LWSWITCH_SET_LATENCY_BINS                      0x08
#define CTRL_LWSWITCH_GET_LWLIPT_COUNTERS                   0x09
#define CTRL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG             0x0A
#define CTRL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG             0x0B
#define CTRL_LWSWITCH_GET_ERRORS                            0x0C
#define CTRL_LWSWITCH_SET_REMAP_POLICY                      0x0D
#define CTRL_LWSWITCH_SET_ROUTING_ID                        0x0E
#define CTRL_LWSWITCH_SET_ROUTING_LAN                       0x0F
#define CTRL_LWSWITCH_GET_INGRESS_REQUEST_TABLE             0x10
#define CTRL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE            0x11
#define CTRL_LWSWITCH_GET_INGRESS_REQLINKID                 0x12
#define CTRL_LWSWITCH_UNREGISTER_LINK                       0x13
#define CTRL_LWSWITCH_RESET_AND_DRAIN_LINKS                 0x14
#define CTRL_LWSWITCH_GET_ROUTING_LAN                       0x15
#define CTRL_LWSWITCH_SET_ROUTING_LAN_VALID                 0x16
#define CTRL_LWSWITCH_GET_LWLINK_STATUS                     0x17
#define CTRL_LWSWITCH_ACQUIRE_CAPABILITY                    0x18
#define CTRL_LWSWITCH_GET_ROUTING_ID                        0x19
#define CTRL_LWSWITCH_SET_ROUTING_ID_VALID                  0x1A
#define CTRL_LWSWITCH_GET_TEMPERATURE                       0x1B
#define CTRL_LWSWITCH_GET_REMAP_POLICY                      0x1C
#define CTRL_LWSWITCH_SET_REMAP_POLICY_VALID                0x1D
#define CTRL_LWSWITCH_GET_THROUGHPUT_COUNTERS               0x1E
#define CTRL_LWSWITCH_GET_BIOS_INFO                         0x1F
#define CTRL_LWSWITCH_BLACKLIST_DEVICE                      0x20
#define CTRL_LWSWITCH_SET_FM_DRIVER_STATE                   0x21
#define CTRL_LWSWITCH_SET_DEVICE_FABRIC_STATE               0x22
#define CTRL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT              0x23
#define CTRL_LWSWITCH_REGISTER_EVENTS                       0x24
#define CTRL_LWSWITCH_UNREGISTER_EVENTS                     0x25
#define CTRL_LWSWITCH_SET_TRAINING_ERROR_INFO               0x26
#define CTRL_LWSWITCH_GET_FATAL_ERROR_SCOPE                 0x27
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define CTRL_LWSWITCH_SET_MC_RID_TABLE                      0x28
#define CTRL_LWSWITCH_GET_MC_RID_TABLE                      0x29
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define CTRL_LWSWITCH_GET_COUNTERS                          0x2A
#define CTRL_LWSWITCH_GET_LWLINK_ECC_ERRORS                 0x2B
#define CTRL_LWSWITCH_I2C_SMBUS_COMMAND                     0x2C
#define CTRL_LWSWITCH_GET_TEMPERATURE_LIMIT                 0x2D
#define CTRL_LWSWITCH_GET_LWLINK_MAX_ERROR_RATES            0x2E
#define CTRL_LWSWITCH_GET_LWLINK_ERROR_COUNTS               0x2F
#define CTRL_LWSWITCH_GET_ECC_ERROR_COUNTS                  0x30
#define CTRL_LWSWITCH_GET_SXIDS                             0x31
#define CTRL_LWSWITCH_GET_FOM_VALUES                        0x32
#define CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS                0x33
#define CTRL_LWSWITCH_SET_RESIDENCY_BINS                    0x34
#define CTRL_LWSWITCH_GET_RESIDENCY_BINS                    0x35
#define CTRL_LWSWITCH_GET_RB_STALL_BUSY                     0x36
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define CTRL_LWSWITCH_CCI_CMIS_PRESENCE                     0x37
#define CTRL_LWSWITCH_CCI_CMIS_LWLINK_MAPPING               0x38
#define CTRL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ           0x39
#define CTRL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE          0x3A
#define CTRL_LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING           0x3B
#define CTRL_LWSWITCH_CCI_GET_GRADING_VALUES                0x3C
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define CTRL_LWSWITCH_INBAND_SEND_DATA                      0x43
#define CTRL_LWSWITCH_INBAND_READ_DATA                      0x44
#define CTRL_LWSWTICH_INBAND_FLUSH_DATA                     0x45
#define CTRL_LWSWITCH_INBAND_PENDING_DATA_STATS             0x46
/*
 * DO NOT ADD CODE AFTER THIS LINE. If the command hits 0xA0, see
 * ctrl_dev_internal_lwswitch.h to adjust the internal range.
 */

#ifdef __cplusplus
}
#endif

#endif // _CTRL_DEVICE_LWSWITCH_H_
