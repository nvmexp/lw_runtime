/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _SMBPBI_FABRIC_LWSWITCH_H_
#define _SMBPBI_FABRIC_LWSWITCH_H_

/*
 * TODO: Find a better place for shared definitions.
 *
 * Shared Fabric state enums
 *
 * Definitions:
 *  Driver Fabric State is intended to reflect the state of the driver and
 *  fabric manager.  Once FM sets the Driver State to CONFIGURED, it is
 *  expected the FM will send heartbeat updates.  If the heartbeat is not
 *  received before the session timeout, then the driver reports status
 *  as MANAGER_TIMEOUT.
 *
 *  Device Fabric State reflects the state of the lwswitch device.
 *  FM sets the Device Fabric State to CONFIGURED once FM is managing the
 *  device.  If the Device Fabric State is BLACKLISTED then the device is
 *  not available for use; opens fail for a blacklisted device, and interrupts
 *  are disabled.
 *
 *  Blacklist Reason provides additional detail of why a device is blacklisted.
 */
typedef enum lwswitch_driver_fabric_state
{
    LWSWITCH_DRIVER_FABRIC_STATE_OFFLINE = 0,      // offline (No driver loaded)
    LWSWITCH_DRIVER_FABRIC_STATE_STANDBY,          // driver up, no FM
    LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED,       // driver up, FM up
    LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_TIMEOUT,  // driver up, FM timed out
    LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_ERROR     // driver up, FM in error state
} LWSWITCH_DRIVER_FABRIC_STATE;

typedef enum lwswitch_device_fabric_state
{
    LWSWITCH_DEVICE_FABRIC_STATE_OFFLINE = 0,      // offline: No driver, no FM
    LWSWITCH_DEVICE_FABRIC_STATE_STANDBY,          // driver up, no FM, not blacklisted
    LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED,       // driver up, FM up, not blacklisted
    LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED       // device is blacklisted
} LWSWITCH_DEVICE_FABRIC_STATE;

typedef enum lwswitch_device_blacklist_mode
{
    LWSWITCH_DEVICE_BLACKLIST_REASON_NONE = 0,                  // device is not blacklisted
    LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_OUT_OF_BAND,        // manually blacklisted by out-of-band client
    LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_IN_BAND,            // manually blacklisted by in-band OS config
    LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_PEER,               // FM indicates blacklisted due to peer manual blacklisted
    LWSWITCH_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE,        // FM indicates blacklisted due to trunk link failure
    LWSWITCH_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE_PEER,   // FM indicates blacklisted due to trunk link failure of peer
    LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE,       // FM indicates blacklisted due to access link failure
    LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE_PEER,  // FM indicates blacklisted due to access link failure of peer
    LWSWITCH_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE,     // FM indicates blacklisted due to unspecified device failure
    LWSWITCH_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE_PEER // FM indicates blacklisted due to unspec device failure of peer
} LWSWITCH_DEVICE_BLACKLIST_REASON;

#endif // _SMBPBI_FABRIC_STATE_H_
