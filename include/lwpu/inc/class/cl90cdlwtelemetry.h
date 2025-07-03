/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl90cdLwTemelemty_h_
#define _cl90cdLwTemelemty_h_

#define __SCHEMA_UPDATE_1__ 1

/* This file defines parameters for LwTelemetry events*/

#define LW90CD_EVENT_BUFFER_LWTELEMETRY_VERSION 1

/*
* These are the types of LwTelemetry switch events
* This field gets added to LW90CD_EVENT_BUFFER_LWTELEMETRY_RECORD to specify the sub type of LwTelemetry event
*/
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_EVENT                       0x1000

#define LW90CD_EVENT_BUFFER_LWTELEMETRY_CLOCK_CONFIG_EVENT          0x0000  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_DEBUG_MODE_EVENT            0x0001  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_FAN_CONFIG_EVENT            0x0002  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_GC6_FAILURE_EVENT           0x0004  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_GSYNC_EVENT                 0x0005  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_HOTPLUG_EVENT               0x0006  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_ILWALID_VBIOS_SIG_EVENT     0x0007  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_NULL_EVENT                  0x0008  // LW2080 event -- used for watchdog
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_SECONDARY_BUS_RESET_EVENT   0x0009  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_TDR_EVENT                   0x000a  // LW0000 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_VOLT_CONFIG_EVENT           0x000b  // LW2080 event
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_NUMBER_OF_EVENTS            0x000c  // LW2080 event

/*
** event structures
**
** the following structures defines the data for each event.
**
*/
/*
* event structure versioning --
* each structure has a version associated with it.
* The version should be incremented when a change is made that makes the
* structure incompatable with the previous version.
* If a field is added, it shoiuld be added to the end of the structure
* (preserving the layout of the previous fields to maintain compatability)
*
* If a field needs to be deleted, the size of a field (type) changed or the order of the
* fields changed, then a new version should be created.
* when creating a new version, the old version should be preserved to allow
* backward compatability for the consumer, a newly versioned structe created,
* and the current info updated so that the driver will use the latest version.
*
* It is STRONGLY recommended that the driver not reference specific versions of
* the structures, rather, it refers to the current versions.  This way the only
* driver changes required when adding a new version will be to update the
* refernces to the changed fields within the structure.
*
* when interpreting the structure the length of the data provided should be
* checked against the offest of the field + it's size to dermine if the field
* is actually present prior to reading the field.
*/

/*=============================================================================
* Telemetry event header & version number.
*    header to define each event type, & hold common data.
*    note -- each event contains a counter. that counter is for each individual event,
*       I.E. the number of LW90CD_EVENT_BUFFER_LWT_CLOCK_CONFIG_EVENTs reported.
*       which is why I considered it unique to each event, and not p[art of the common
*       data
* 
*    drivrVersion -- cersion of driver providing the data
*
*    eventType -- the event type. used to create the correct event & interprete
*       the data.
*
*    eventVersion -- the event version. used to interprete the data.
*
*    gpuId -- gpu associated with the event
*
*    counter -- counter of the number of events of the given event type has been reported.
*
*    timeStamp -- the time the event was recorded
*
*    eventDataSize -- the event data size. used to interprete the data.
*/
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_EVENT_VERSION_1                     0x0001
//-----------------------------------------------------------------------------
typedef struct
{
    LwU16           driverVersion;
    LwU16           eventType;
    LwU16           eventVersion;
#ifndef __SCHEMA_UPDATE_1__
    LwU32           gpuId;
#else
    LwU64           gpuEcid[2];
#endif
    LwU32           counter;
    LwU64           timeStamp;
    LwU32           eventDataSize;
}   LW90CD_EVENT_BUFFER_LWTELEMETRY_EVENT_HEADER_V1;
//-----------------------------------------------------------------------------
#define LW90CD_LWRR_EVENT_BUFFER_LWTELEMETRY_EVENT_VERSION                  LW90CD_EVENT_BUFFER_LWTELEMETRY_EVENT_VERSION_1
typedef LW90CD_EVENT_BUFFER_LWTELEMETRY_EVENT_HEADER_V1 LW90CD_LWRR_EVENT_BUFFER_LWTELEMETRY_EVENT_HEADER;


/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_CLOCK_CONFIG_EVENT.
*   event record indicating a clock configuration call has been made.
*   this event has no varData.
*
*/

// V0  ------------------------------------------------------------------------
// there is lwrrently no event data associated with this beyond the header.
#define  LW90CD_EVENT_BUFFER_LWT_CLOCK_CONFIG_EVENT_V0                      0x0000

// current version  -----------------------------------------------------------
#define LW90CD_LWRR_EVENT_BUFFER_LWTELEMETRY_CLOCK_CONFIG_EVENT_VERSION     LW90CD_EVENT_BUFFER_LWT_CLOCK_CONFIG_EVENT_V0

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWTELEMETRY_DEBUG_MODE_EVENT.
*   event indicating we system is entering/exiting debug mode
*   this event has no varData.
*
*    enable -- indicates if debug mode is enabled.
*
*    source -- indicates where the in the driver the debug mode was changed.
*/

// source defines
#define LW90CD_EVENT_BUFFER_LWT_DEBUG_MODE_SRC_UNKNOWN              0
#define LW90CD_EVENT_BUFFER_LWT_DEBUG_MODE_SRC_DEBUG_MODE           1
#define LW90CD_EVENT_BUFFER_LWT_DEBUG_MODE_SRC_DEFAULT_MODE_P2X     2
#define LW90CD_EVENT_BUFFER_LWT_DEBUG_MODE_SRC_DEFAULT_MODE_P3X     3

// V1  ------------------------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWTELEMETRY_DEBUG_MODE_EVENT_VERSION_V1         0x0001
typedef struct
{
    LwBool  enable;
    LwU16   source;
}   LW90CD_EVENT_BUFFER_LWT_DEBUG_MODE_EVENT_BUFFER_V1;

// current version  -----------------------------------------------------------
typedef LW90CD_EVENT_BUFFER_LWT_DEBUG_MODE_EVENT_BUFFER_V1          LW90CD_EVENT_BUFFER_LWT_LWRR_DEBUG_MODE_EVENT_BUFFER;
#define LW90CD_EVENT_BUFFER_LWT_DEBUG_MODE_EVENT_LWRR_VERSION       LW90CD_EVENT_BUFFER_LWTELEMETRY_DEBUG_MODE_EVENT_VERSION_V1

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_FAN_CONFIG_EVENT.
*   event record indicating a fan configuration call has been made.
*   this event has no varData.
*
*/

// V0  ------------------------------------------------------------------------
// there is lwrrently no event data associated with this beyond the header.
#define LW90CD_EVENT_BUFFER_LWT_FAN_CONFIG_EVENT_VERSION_V0         0x0000

// current version  -----------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_FAN_CONFIG_EVENT_LWRR_VERSION       LW90CD_EVENT_BUFFER_LWT_FAN_CONFIG_EVENT_VERSION_V0

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_GC6_FAILURE_EVENT.
*   event record indicating the gpu has failed to enter GC6.
*   this event has no varData.
*
*/

// V0  ------------------------------------------------------------------------
// there is lwrrently no event data associated with this beyond the header.
#define LW90CD_EVENT_BUFFER_LWT_GC6_FAILURE_EVENT_VERSION_V0        0x0000

// current version  -----------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_GC6_FAILURE_EVENT_LWRR_VERSION      LW90CD_EVENT_BUFFER_LWT_GC6_FAILURE_EVENT_VERSION_V0

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWTELEMETRY_GSYNC_EVENT.
*   event record indicating the specified head has entered/exited gsync.
*   this event has no varData.
*
*    counter -- number of gsync events reported
*
*    head -- head that whose gsync state is being reported
*
*    enabled -- indicates if GSync is enabled on the specified head.
*
*    tearing -- indicates if tearing mode enabled on the specified head.
*
*/

// V1  ------------------------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_GSYNC_EVENT_VERSION_V1              0x0001
typedef struct
{
    LwU32   counter;
    LwU32   head;
    LwU32   enabled;
    LwU32   tearing;
}   LW90CD_EVENT_BUFFER_LWT_GSYNC_EVENT_BUFFER_V1;

// current version  -----------------------------------------------------------
typedef LW90CD_EVENT_BUFFER_LWT_GSYNC_EVENT_BUFFER_V1               LW90CD_EVENT_BUFFER_LWT_LWRR_GSYNC_EVENT_BUFFER;
#define LW90CD_EVENT_BUFFER_LWT_GSYNC_EVENT_LWRR_VERSION            LW90CD_EVENT_BUFFER_LWT_GSYNC_EVENT_VERSION_V1   

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_HOTPLUG_EVENT.
*   event record indicating the gpu has processed a hot (un)plug event.
*   this event has no varData.
*
*    hotPluginChangeDeviceMap -- bitmap indicating which heads have been attached.
*
*    hotUnplugChangeDeviceMap -- bitmap indicating which heads have been dettached.
*
*/

// V1  ------------------------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_HOTPLUG_EVENT_VERSION_V1            0x0001
typedef struct
{
    LwU32   hotPluginChangeDeviceMap;
    LwU32   hotUnplugChangeDeviceMap;
}   LW90CD_EVENT_BUFFER_LWT_HOTPLUG_EVENT_BUFFER_V1;

// current version  -----------------------------------------------------------
typedef LW90CD_EVENT_BUFFER_LWT_HOTPLUG_EVENT_BUFFER_V1             LW90CD_EVENT_BUFFER_LWT_LWRR_HOTPLUG_EVENT_BUFFER;
#define LW90CD_EVENT_BUFFER_LWT_HOTPLUG_EVENT_LWRR_VERSION          LW90CD_EVENT_BUFFER_LWT_HOTPLUG_EVENT_VERSION_V1

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_ILWALID_VBIOS_SIG_EVENT.
*   event record indicating the specified gpu has a VBIOS error.
*   this event has no varData.
*
*   vbiosStatus -- status code for the vbios.  note that not all statuses are
*       reported. only status codes that are content related.
*
*/

// V1  ------------------------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_ILWALID_VBIOS_SIG_EVENT_VERSION_V1  0x0001
typedef struct
{
    LwU32   vbiosStatus;
}   LW90CD_EVENT_BUFFER_LWT_ILWALID_VBIOS_SIG_EVENT_BUFFER_V1;
// current version  -----------------------------------------------------------
typedef LW90CD_EVENT_BUFFER_LWT_ILWALID_VBIOS_SIG_EVENT_BUFFER_V1       LW90CD_EVENT_BUFFER_LWT_LWRR_ILWALID_VBIOS_SIG_EVENT_BUFFER;
#define LW90CD_EVENT_BUFFER_LWT_ILWALID_VBIOS_SIG_EVENT_LWRR_VERSION    LW90CD_EVENT_BUFFER_LWT_ILWALID_VBIOS_SIG_EVENT_VERSION_V1

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_VOLT_CONFIG_EVENT.
*   event record indicating a volt config call has been processed.
*   this event has no varData.
*
*    ovDelta-- the overvolting delta.
*
*/

// V1  ------------------------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_VOLT_CONFIG_EVENT_V1                0x0001
typedef struct
{
    LwU32   ovDelta;
}   LW90CD_EVENT_BUFFER_LWT_VOLT_CONFIG_EVENT_BUFFER_V1;

// current version  -----------------------------------------------------------
typedef LW90CD_EVENT_BUFFER_LWT_VOLT_CONFIG_EVENT_BUFFER_V1                 LW90CD_EVENT_BUFFER_LWT_LWRR_VOLT_CONFIG_EVENT_BUFFER;
#define LW90CD_LWRR_EVENT_BUFFER_LWTELEMETRY_VOLT_CONFIG_EVENT_VERSION      LW90CD_EVENT_BUFFER_LWT_VOLT_CONFIG_EVENT_V1

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_SECONDARY_BUS_RESET_EVENT.
*   event record indicating the specified gpu performed a secondary bus reset.
*   this event has no varData.
*
*/

// V0  ------------------------------------------------------------------------
// there is lwrrently no event data associated with this beyond the header.
#define LW90CD_EVENT_BUFFER_LWT_SECONDARY_BUS_RESET_EVENT_VERSION_V0  0x0000

// current version  -----------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_SECONDARY_BUS_RESET_EVENT_LWRR_VERSION      LW90CD_EVENT_BUFFER_LWT_SECONDARY_BUS_RESET_EVENT_VERSION_V0

/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_TDR_EVENT.
*   event record indicating the specified gpu has TDRd.
*   this event has varData. it uses the LW90CD_EVENT_BUFFER_LWT_PS3X_OVERCLOCK_INFO_BUFFER
*   structure to report the overclocking state at the time of the TDR.
*
*   ocEntryCount -- count of ocEntries
*
*   ocEntries -- the oc setting for each domain.
*
*/

// V1  ------------------------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_TDR_EVENT_VERSION_V1                0x0001

typedef struct
{
    LwU32   varDataVersion;
}   LW90CD_EVENT_BUFFER_LWT_TDR_EVENT_BUFFER_V1;

// current version  -----------------------------------------------------------
typedef LW90CD_EVENT_BUFFER_LWT_TDR_EVENT_BUFFER_V1         LW90CD_EVENT_BUFFER_LWT_LWRR_TDR_EVENT_BUFFER;
#define LW90CD_EVENT_BUFFER_LWT_TDR_EVENT_LWRR_VERSION      LW90CD_EVENT_BUFFER_LWT_TDR_EVENT_VERSION_V1


/*=============================================================================
*  LW90CD_EVENT_BUFFER_LWT_PS3X_OVERCLOCK_INFO.
*   varData structure used to report the overclocking state
*
*
*    version -- version of this structure
*
*   pstateVer -- version of pstate running.
*
*    count -- number of data entries listed
*
*    data[].domain -- clock domain.
*
*    data[].delta[0] -- min delta
*
*    data[].delta[1] -- avg delta
*
*    data[].delta[2] -- max delta
*
*/
#define LW90CD_EVENT_BUFFER_LWT_MAX_OC_ENTRIES  256

// V1  ------------------------------------------------------------------------
#define LW90CD_EVENT_BUFFER_LWT_PSX_OVERCLOCK_INFO_V1      1
typedef struct
{
    LwU32   domain;
    LwU32   delta[3];
}   LW90CD_EVENT_BUFFER_LWT_PSX_OVERCLOCK_ENTRY_V1;

typedef struct
{
    LwU32   pstateVer;
    LwU32   count;
}   LW90CD_EVENT_BUFFER_LWT_PSX_OVERCLOCK_BUFFER_HEADER_V1;


// current version  -----------------------------------------------------------
typedef LW90CD_EVENT_BUFFER_LWT_PSX_OVERCLOCK_BUFFER_HEADER_V1     LW90CD_EVENT_BUFFER_LWT_LWRR_PSX_OVERCLOCK_BUFFER_HEADER;
typedef LW90CD_EVENT_BUFFER_LWT_PSX_OVERCLOCK_ENTRY_V1             LW90CD_EVENT_BUFFER_LWT_LWRR_PSX_OVERCLOCK_ENTRY;
#define LW90CD_LWRR_EVENT_BUFFER_LWT_PSX_OVERCLOCK_INFO_VERSION    LW90CD_EVENT_BUFFER_LWT_PSX_OVERCLOCK_INFO_V1


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef union
{
    LW90CD_EVENT_BUFFER_LWT_LWRR_DEBUG_MODE_EVENT_BUFFER            debugModeEvent;
    LW90CD_EVENT_BUFFER_LWT_LWRR_GSYNC_EVENT_BUFFER                 gsyncEvent;
    LW90CD_EVENT_BUFFER_LWT_LWRR_HOTPLUG_EVENT_BUFFER               hotplugEvent;
    LW90CD_EVENT_BUFFER_LWT_LWRR_ILWALID_VBIOS_SIG_EVENT_BUFFER     ilwalidVbiosSigEvent;
    LW90CD_EVENT_BUFFER_LWT_LWRR_TDR_EVENT_BUFFER                   tdrEvent;
    LW90CD_EVENT_BUFFER_LWT_LWRR_VOLT_CONFIG_EVENT_BUFFER           voltConfigEvent;
}   LW90CD_EVENT_BUFFER_LWTELEMETRY_PAYLOAD;

// telemetry event as submitted to the osEventNotification call
typedef struct
{
    LwU16                                               version;
    LwU16                                               dataOffset;
    LW90CD_LWRR_EVENT_BUFFER_LWTELEMETRY_EVENT_HEADER   header;
    LW90CD_EVENT_BUFFER_LWTELEMETRY_PAYLOAD             data;
} LW90CD_EVENT_BUFFER_LWTELEMETRY_DATA;

// telemetry event as presented in the buffer
typedef struct
{
    LW_EVENT_BUFFER_RECORD_HEADER                       recordHeader;
    LW90CD_EVENT_BUFFER_LWTELEMETRY_DATA                eventRecord;
} LW90CD_EVENT_BUFFER_LWTELEMETRY_RECORD;

#define LWT_ETW_CP_BLK_RM_SETUP_ADAPTER                 0x0001
#define LWT_ETW_CP_BLK_RM_INIT_DEVICE                   0x0002
#define LWT_ETW_CP_BLK_VBIOS_INIT                       0x0003

// generic checkpoint codes
#define LWT_ETW_CP_START                                0x0001
#define LWT_ETW_CP_SUCCESS                              0x0002


// failure checkpoint flag.

#define LWT_ETW_CP_FALIURE_FLAG                         0x10000000
#define LWT_ETW_FALIURE_FLAG                            0x20000000

// RmSetupLwAdapter specific
#define LWT_ETW_CP_GPU_AVAIL                            (0x0100)
#define LWT_ETW_CP_FIND_ADAPTER_FAILED                  (LWT_ETW_CP_FALIURE_FLAG | 0x101)
#define LWT_ETW_CP_REGISTER_GPU_ID_FAILED               (LWT_ETW_CP_FALIURE_FLAG | 0x102)
#define LWT_ETW_CP_ALLOC_GPU_FAILED                     (LWT_ETW_CP_FALIURE_FLAG | 0x103)
#define LWT_ETW_CP_GPU_LOCK_ALLOC_FAILED                (LWT_ETW_CP_FALIURE_FLAG | 0x104)
#define LWT_ETW_CP_CREATE_DEVICE_FAILED                 (LWT_ETW_CP_FALIURE_FLAG | 0x105)
#define LWT_ETW_CP_GPU_LOCK_HIDE_FAILED                 (LWT_ETW_CP_FALIURE_FLAG | 0x106)
#define LWT_ETW_CP_GPU_ATTACH_FAILED                    (LWT_ETW_CP_FALIURE_FLAG | 0x107)
#define LWT_ETW_CP_NULL_FB_MAP                          (LWT_ETW_CP_FALIURE_FLAG | 0x108)
#define LWT_ETW_CP_SCALABILITY_FAILED                   (LWT_ETW_CP_FALIURE_FLAG | 0x109)
#define LWT_ETW_CP_HAL_INIT_FAILED                      (LWT_ETW_CP_FALIURE_FLAG | 0x10a)
#define LWT_ETW_CP_SWAP_DB_INIT_FAILED                  (LWT_ETW_CP_FALIURE_FLAG | 0x10b)

// _rmInitLwDevice specific
#define LWT_ETW_CP_AQUIRE_SEMA_FAILED                   (LWT_ETW_CP_FALIURE_FLAG | 0x121)
#define LWT_ETW_CP_AQUIRE_LOCK_FAILED                   (LWT_ETW_CP_FALIURE_FLAG | 0x122)
#define LWT_ETW_CP_GPU_HIDE_LOCK_FAILED                 (LWT_ETW_CP_FALIURE_FLAG | 0x123)
#define LWT_ETW_CP_GPUMGR_QUADRO_DETECT_FAILED          (LWT_ETW_CP_FALIURE_FLAG | 0x124)
#define LWT_ETW_CP_GPUMGR_GEFORCE_SMB_DETECT_FAILED     (LWT_ETW_CP_FALIURE_FLAG | 0x125)
#define LWT_ETW_CP_GPUMGR_TESLA_DETECT_FAILED           (LWT_ETW_CP_FALIURE_FLAG | 0x126)
#define LWT_ETW_CP_GPUMGR_VGX_DETECT_FAILED             (LWT_ETW_CP_FALIURE_FLAG | 0x127)
#define LWT_ETW_CP_GPUMGR_GRID_DETECT_FAILED            (LWT_ETW_CP_FALIURE_FLAG | 0x128)
#define LWT_ETW_CP_GPUMGR_GRID_SYSTEM_DETECT_FAILED     (LWT_ETW_CP_FALIURE_FLAG | 0x129)
#define LWT_ETW_CP_GPUMGR_TITAN_DETECT_FAILED           (LWT_ETW_CP_FALIURE_FLAG | 0x12a)
#define LWT_ETW_CP_GPU_STATE_PREINIT_FAILED             (LWT_ETW_CP_FALIURE_FLAG | 0x12b)
#define LWT_ETW_CP_GPU_STATE_INIT_FAILED                (LWT_ETW_CP_FALIURE_FLAG | 0x12c)
#define LWT_ETW_CP_STATE_LOAD_FAILED                    (LWT_ETW_CP_FALIURE_FLAG | 0x12d)
#define LWT_ETW_CP_GR_CAPS_INIT_FAILED                  (LWT_ETW_CP_FALIURE_FLAG | 0x12e)
#define LWT_ETW_CP_HAL_VALIDATION_FAILED                (LWT_ETW_CP_FALIURE_FLAG | 0x12f)

// descriete failures
// vbiosInit
#define LWT_ETW_DRIVER_FAIL_VBIOS_HEADER_FAILED         (LWT_ETW_FALIURE_FLAG | 0x1001)
#define LWT_ETW_DRIVER_FAIL_VBIOS_BITADDR_FAILED        (LWT_ETW_FALIURE_FLAG | 0x1002)
#define LWT_ETW_DRIVER_FAIL_VBIOS_NETLIST_FAILED        (LWT_ETW_FALIURE_FLAG | 0x1003)
#define LWT_ETW_DRIVER_FAIL_VBIOS_REVLOCK_FAILED        (LWT_ETW_FALIURE_FLAG | 0x1004)
#define LWT_ETW_DRIVER_FAIL_VBIOS_PMU_PD_INIT_FAILED    (LWT_ETW_FALIURE_FLAG | 0x1005)
#define LWT_ETW_DRIVER_FAIL_VBIOS_TEST                  (LWT_ETW_FALIURE_FLAG | 0x1006)

// the strings below are now no longer used int he driver as of this CL. they will be removed once the client has the corrispoinding update.
//
// following defines are for strings that  will be used for reporting events via the event log.
// They are defined here so they are shared between the driver (fopr creating the message) 
// & plug-in (for parsing the message).
#define LWPMRPT_HDR_STR                                 "LwT"
#define LWPMRPT_FLD_SEP_STR                             ":"
                                                                     //                      type                      driver version           subtype                  type format 
#define LWPMRPT_FMT_STR                                  LWPMRPT_HDR_STR LWPMRPT_FLD_SEP_STR  "%s" LWPMRPT_FLD_SEP_STR "%s" LWPMRPT_FLD_SEP_STR "%s" LWPMRPT_FLD_SEP_STR "%s"
#define LWPMRPT_ILWALID_STR                             "***"
#define LWPMRPT_ILWALID_FMT_STR                         ""

#define LWPMRPT_TYPE_DRIVER_FAILURE_STR                 "DIE"
#define LWPMRPT_TYPE_DRIVER_FAILURE_FMT_STR             ""
#define LWPMRPT_TYPE_DRIVER_CP_STR                      "DPCP"
#define LWPMRPT_TYPE_DRIVER_CP_FMT_STR                  "%x:%x:%x:"

// report subtypes

// Driver Failure subtypes
#define LWPMRPT_SUBTYPE_VBIOS_HEADER_STR                "VBH"
#define LWPMRPT_SUBTYPE_VBIOS_BITADDR_STR               "VBBA"
#define LWPMRPT_SUBTYPE_VBIOS_NETLIST_STR               "VBNR"
#define LWPMRPT_SUBTYPE_VBIOS_REVLOCK_STR               "VBRL"
#define LWPMRPT_SUBTYPE_VBIOS_PMU_PD_INIT_STR           "VBPDI"
#define LWPMRPT_SUBTYPE_VBIOS_TEST_STR                  "Test"


// Check point subtypes
#define LWPMRPT_SUBTYPE_DRIVER_BLOCK_START_STR          "BS"
#define LWPMRPT_SUBTYPE_DRIVER_BLOCK_CHECKPOINT_STR     "CP"
#define LWPMRPT_SUBTYPE_DRIVER_BLOCK_END_STR            "BE"

#endif /* _cl90cdLwTelemetry_h_ */

