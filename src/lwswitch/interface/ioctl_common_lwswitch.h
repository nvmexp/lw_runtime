/*******************************************************************************
    Copyright (c) 2019 LWpu Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef _IOCTL_COMMON_LWSWITCH_H_
#define _IOCTL_COMMON_LWSWITCH_H_

#ifdef __cplusplus
extern "C"
{
#endif

#define LWSWITCH_DEV_IO_TYPE 'd'
#define LWSWITCH_CTL_IO_TYPE 'c'

/*
 * Defines for IOCTL Hints
 *
 * LWSWITCH_IO_READ_ONLY :
 *      Only reads parameters from the kernel and does not pass any to it
 *
 * LWSWITCH_IO_WRITE_ONLY :
 *      Only writes parameters to the kernel, but does not want anything back.
 *
 * LWSWITCH_IO_WRITE_READ :
 *      Writes data to the kernel and wants information back
 *
 * LWSWITCH_IO_DEFAULT :
 *      Don't copy anything into the kernel, nor copy anything back.
 */
#define LWSWITCH_IO_READ_ONLY             0x0
#define LWSWITCH_IO_WRITE_ONLY            0x1
#define LWSWITCH_IO_WRITE_READ            0x2
#define LWSWITCH_IO_DEFAULT               0x3

#if (defined(_WIN32) || defined(_WIN64))
/*
 * Values of less than 0x800 are reserved for Microsoft.
 * Values of 0x800 and higher can be used by vendors.
 */
#define IOCTL_START_INDEX             0x800

/*
 * Macro for defining new IOCTLs in a platform independent way.
 */
#define LWSWITCH_IOCTL_CODE(ioType, ctrl, paramType, direction)                \
    CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_START_INDEX + ctrl, METHOD_BUFFERED,   \
            (FILE_READ_DATA | FILE_WRITE_DATA))
#else

/*
 * Macro for defining new IOCTLs in a platform independent way.
 *
 * Select Linux specific IOCTL defining macro (_IO, _IOR, _IOW, _IOWR)
 * based on IOCTL direction.
 */
#define LWSWITCH_IOCTL_CODE(ioType, ctrl, paramType, direction)               \
    ((direction == LWSWITCH_IO_READ_ONLY)  ? _IOR(ioType, ctrl, paramType)  : \
     (direction == LWSWITCH_IO_WRITE_ONLY) ? _IOW(ioType, ctrl, paramType)  : \
     (direction == LWSWITCH_IO_WRITE_READ) ? _IOWR(ioType, ctrl, paramType) : \
     _IO(ioType, ctrl))

#endif // (defined(_WIN32) || defined(_WIN64))

/*
 * LWSWITCH_LWLINK_MAX_LANES is used by both internal and exteranl IOCTLs.
 */
#define LWSWITCH_LWLINK_MAX_LANES  4

/*
 * Common Fabric State enums
 *
 * Definitions:
 *    Driver Fabric State is intended to reflect the state of the driver and
 *    fabric manager.  Once FM sets the Driver State to CONFIGURED, it is
 *    expected the FM will send heartbeat updates.  If the heartbeat is not
 *    received before the session timeout, then the driver reports status
 *    as MANAGER_TIMEOUT.
 *
 *    Device Fabric State reflects the state of the lwswitch device.
 *    FM sets the Device Fabric State to CONFIGURED once FM is managing the
 *    device.  If the Device Fabric State is BLACKLISTED then the device is
 *    not available for use; opens fail for a blacklisted device, and interrupts
 *    are disabled.
 *
 *    Blacklist Reason provides additional detail of why a device is blacklisted.
 */
typedef enum lwswitch_driver_fabric_state
{
    LWSWITCH_DRIVER_FABRIC_STATE_OFFLINE = 0,      // offline (No driver loaded)
    LWSWITCH_DRIVER_FABRIC_STATE_STANDBY,          // driver up, no FM
    LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED,       // driver up, FM up
    LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_TIMEOUT,  // driver up, FM timed out
    LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_ERROR,    // driver up, FM in error state
    LWSWITCH_DRIVER_FABRIC_STATE_COUNT
} LWSWITCH_DRIVER_FABRIC_STATE;

typedef enum lwswitch_device_fabric_state
{
    LWSWITCH_DEVICE_FABRIC_STATE_OFFLINE = 0,      // offline: No driver, no FM
    LWSWITCH_DEVICE_FABRIC_STATE_STANDBY,          // driver up, no FM, not blacklisted
    LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED,       // driver up, FM up, not blacklisted
    LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED,      // device is blacklisted
    LWSWITCH_DEVICE_FABRIC_STATE_COUNT
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

#ifdef __cplusplus
}
#endif // __cplusplus

#endif //_IOCTL_COMMON_LWSWITCH_H_
