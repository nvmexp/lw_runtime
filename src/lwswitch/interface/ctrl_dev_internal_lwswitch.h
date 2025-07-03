/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * This file defines CTRL calls that are device specifics.
 *
 * This is a platform agnostic file and lists the internal CTRL calls used by
 * MODS or LWSwitch GTEST.
 *
 * The CTRL calls listed in this file do not contribute to the driver ABI
 * version.
 *
 * Note: ctrl_dev_lwswitch.h and ctrl_dev_internal_lwswitch.h do not share any
 * data. This helps to keep the driver ABI stable.
 */

#ifndef _CTRL_DEVICE_INTERNAL_LWSWITCH_H_
#define _CTRL_DEVICE_INTERNAL_LWSWITCH_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "lwtypes.h"
#include "lwfixedtypes.h"
#include "ioctl_common_lwswitch.h"

/*
 * CTRL_LWSWITCH_SET_PORT_TEST_MODE
 *
 * Control for setting device port debug & test configurations.
 *
 * Parameters:
 *    portNum [IN]
 *      A valid port number present in the port masks returned by
 *      LWSWITCH_GET_INFO.
 *   nea [IN]
 *      Set true if port should be configured near end analog loopback.
 *   ned [IN]
 *      Set true if port should be configured near end digital loopback.
 *      NEA and NED can not both be enabled simultaneously.
 */

typedef struct lwswitch_set_port_test_mode
{
    LwU32  portNum;
    LwBool nea;
    LwBool ned;
} LWSWITCH_SET_PORT_TEST_MODE;

/*
 * CTRL_LWSWITCH_INJECT_LINK_ERROR
 */

typedef struct
{
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
    LwBool bFatalError;
} LWSWITCH_INJECT_LINK_ERROR;

/*
 * CTRL_LWSWITCH_REGISTER_READ/WRITE
 *
 * This provides direct access to the MMIO space for trusted clients like
 * MODS. This can be also used on debug/develop driver builds.
 * This API should not be exposed to unselwre clients.
 */

typedef struct
{
    LwU32   engine;     // REGISTER_RW_ENGINE_*
    LwU32   instance;   // engine instance
    LwU32   offset;     // Register offset within device/instance
    LwU32   val;        // out: register value read
} LWSWITCH_REGISTER_READ;

typedef struct
{
    LwU32   engine;     // REGISTER_RW_ENGINE_*
    LwU32   instance;   // engine instance
    LwBool  bcast;      // Unicast or broadcast
    LwU32   offset;     // Register offset within engine/instance
    LwU32   val;        // in: register value to write
} LWSWITCH_REGISTER_WRITE;

#define REGISTER_RW_ENGINE_RAW                       0x00

#define REGISTER_RW_ENGINE_CLKS                      0x10
#define REGISTER_RW_ENGINE_FUSE                      0x11
#define REGISTER_RW_ENGINE_JTAG                      0x12
#define REGISTER_RW_ENGINE_PMGR                      0x13
#define REGISTER_RW_ENGINE_SAW                       0x14
#define REGISTER_RW_ENGINE_XP3G                      0x15
#define REGISTER_RW_ENGINE_XVE                       0x16
#define REGISTER_RW_ENGINE_SOE                       0x17
#define REGISTER_RW_ENGINE_SMR                       0x18
#define REGISTER_RW_ENGINE_SE                        0x19
#define REGISTER_RW_ENGINE_CLKS_SYS                  0x1A
#define REGISTER_RW_ENGINE_CLKS_SYSB                 0x1B
#define REGISTER_RW_ENGINE_CLKS_P0                   0x1C
#define REGISTER_RW_ENGINE_XPL                       0x1D
#define REGISTER_RW_ENGINE_XTL                       0x1E

#define REGISTER_RW_ENGINE_SIOCTRL                   0x20
#define REGISTER_RW_ENGINE_MINION                    0x21
#define REGISTER_RW_ENGINE_LWLIPT                    0x22
#define REGISTER_RW_ENGINE_LWLTLC                    0x23
#define REGISTER_RW_ENGINE_LWLTLC_MULTICAST          0x24
#define REGISTER_RW_ENGINE_DLPL                      0x25
#define REGISTER_RW_ENGINE_TX_PERFMON                0x26
#define REGISTER_RW_ENGINE_TX_PERFMON_MULTICAST      0x27
#define REGISTER_RW_ENGINE_RX_PERFMON                0x28
#define REGISTER_RW_ENGINE_RX_PERFMON_MULTICAST      0x29
#define REGISTER_RW_ENGINE_LWLW                      0x2a
#define REGISTER_RW_ENGINE_LWLIPT_LNK                0x2b
#define REGISTER_RW_ENGINE_LWLIPT_LNK_MULTICAST      0x2c
#define REGISTER_RW_ENGINE_LWLDL                     0x2d
#define REGISTER_RW_ENGINE_LWLDL_MULTICAST           0x2e
#define REGISTER_RW_ENGINE_PLL                       0x2f

#define REGISTER_RW_ENGINE_NPG                       0x30
#define REGISTER_RW_ENGINE_NPORT                     0x31
#define REGISTER_RW_ENGINE_NPORT_MULTICAST           0x32
#define REGISTER_RW_ENGINE_NPG_PERFMON               0x33
#define REGISTER_RW_ENGINE_NPORT_PERFMON             0x34
#define REGISTER_RW_ENGINE_NPORT_PERFMON_MULTICAST   0x35

#define REGISTER_RW_ENGINE_SWX                       0x40
#define REGISTER_RW_ENGINE_SWX_PERFMON               0x41
#define REGISTER_RW_ENGINE_AFS                       0x42
#define REGISTER_RW_ENGINE_AFS_PERFMON               0x43
#define REGISTER_RW_ENGINE_NXBAR                     0x44
#define REGISTER_RW_ENGINE_NXBAR_PERFMON             0x45
#define REGISTER_RW_ENGINE_TILE                      0x46
#define REGISTER_RW_ENGINE_TILE_MULTICAST            0x47
#define REGISTER_RW_ENGINE_TILE_PERFMON              0x48
#define REGISTER_RW_ENGINE_TILE_PERFMON_MULTICAST    0x49

#define REGISTER_RW_ENGINE_LWLW_PERFMON              0x50

/*
 * CTRL_LWSWITCH_READ/WRITE_JTAG_CHAIN
 *
 * Used to read/write values from/to JTAG interface
 */

typedef struct
{
    LwU32 chainLen;
    LwU32 chipletSel;
    LwU32 instrId;
    LwU32 dataArrayLen;         // in dwords
    LwU32 *data;
} LWSWITCH_JTAG_CHAIN_PARAMS;

/*
 * PEX
 */

/*
 * Note that MAX_COUNTER_TYPES will need to be updated each time
 * a new counter type gets added to the list below. The value
 * depends on the bits set for the last valid define. Look
 * at pexCounters[] comments above for details.
 *
 */
#define LWSWITCH_PEX_COUNTER_TYPE                           0x00000000
#define LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS                0x00000001
#define LWSWITCH_PEX_COUNTER_REPLAY_COUNT                   0x00000002
#define LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT          0x00000004
#define LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT                 0x00000008
#define LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT                  0x00000010
#define LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT             0x00000020
#define LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT       0x00000040
#define LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT              0x00000080
#define LWSWITCH_PEX_COUNTER_FAILED_L0S_EXITS_COUNT         0x00000100
#define LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT                0x00000200
#define LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT                0x00000400
#define LWSWITCH_PEX_COUNTER_LANE_ERRORS                    0x00000800
#define LWSWITCH_PEX_COUNTER_L1_TO_RECOVERY_COUNT           0x00001000
#define LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT           0x00002000
#define LWSWITCH_PEX_COUNTER_RECOVERY_COUNT                 0x00004000
#define LWSWITCH_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT   0x00008000
#define LWSWITCH_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT       0x00010000
#define LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT                 0x00020000
#define LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT                0x00040000
#define LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT            0x00080000
#define LWSWITCH_PEX_COUNTER_ASLM_COUNT                     0x00100000
#define LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT         0x00200000
#define LWSWITCH_PEX_COUNTER_CORR_ERROR_COUNT               0x00400000
#define LWSWITCH_PEX_COUNTER_NON_FATAL_ERROR_COUNT          0x00800000
#define LWSWITCH_PEX_COUNTER_FATAL_ERROR_COUNT              0x01000000
#define LWSWITCH_PEX_COUNTER_UNSUPP_REQ_COUNT               0x02000000
#define LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT               0x04000000
#define LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT               0x08000000
#define LWSWITCH_PEX_COUNTER_L1_2_ABORT_COUNT               0x10000000
#define LWSWITCH_PEX_COUNTER_L1SS_TO_DEEP_L1_TIMEOUT_COUNT  0x20000000
#define LWSWITCH_PEX_COUNTER_L1_SHORT_DURATION_COUNT        0x40000000

/*
 * CTRL_LWSWITCH_PEX_GET_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * pexCounterMask
 *  This parameter specifies the input mask for desired counter types.
 *
 * pexTotalCorrectableErrors
 *  This parameter gives the total correctable errors which includes
 *  LW_XVE_ERROR_COUNTER1 plus LCRC Errors, 8B10B Errors, NAKS and Failed L0s
 *
 * pexCorrectableErrors
 *  This parameter only includes LW_XVE_ERROR_COUNTER1 value.
 *
 * pexTotalNonFatalErrors
 *  This parameter returns total Non-Fatal Errors which may or may not
 *  include Correctable Errors.
 *
 * pexTotalFatalErrors
 *  This parameter returns Total Fatal Errors
 *
 * pexTotalUnsupportedReqs
 *  This parameter returns Total Unsupported Requests
 *
 * pexErrors
 *  This array contains the error counts for each error type as requested from
 *  the pexCounterMask. The array indexes correspond to the mask bits one-to-one.
 */

#define LWSWITCH_PEX_MAX_COUNTER_TYPES             31

typedef struct
{
    LwU32  pexCounterMask;
    LwU32  pexTotalCorrectableErrors;
    LwU16  pexCorrectableErrors;
    LwU8   pexTotalNonFatalErrors;
    LwU8   pexTotalFatalErrors;
    LwU8   pexTotalUnsupportedReqs;
    LwU16  pexCounters[LWSWITCH_PEX_MAX_COUNTER_TYPES];
} LWSWITCH_PEX_GET_COUNTERS_PARAMS;

/*
 * CTRL_LWSWITCH_PEX_CLEAR_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * pexCounterMask
 *  This parameter specifies the input mask for desired counters to be
 *  cleared. Note that all counters cannot be cleared.
 */

typedef struct
{
    LwU32  pexCounterMask;
} LWSWITCH_PEX_CLEAR_COUNTERS_PARAMS;

/*
 * LWSWITCH_PEX_GET_LANE_COUNTERS
 *  This command gets the per Lane Counters and the type of errors.
 *
 * pexLaneErrorStatus
 *  This mask specifies the type of error detected on any of the Lanes.
 *
 * pexLaneCounter
 *  This array gives the counters per Lane. Each index corresponds to Lane
 *  index + 1
 */

#define LWSWITCH_PEX_MAX_LANES                      16

typedef struct
{
    LwU16 pexLaneErrorStatus;
    LwU8  pexLaneCounter[LWSWITCH_PEX_MAX_LANES];
} LWSWITCH_PEX_GET_LANE_COUNTERS_PARAMS;


/*
 * Read/write I2C interface
 */

/* This field specifies the maximum regular port identifier allowed. */
#define LWSWITCH_CTRL_NUM_I2C_PORTS                                           10

/*
 * LWSWITCH_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED
 *   The port exists on this hardware.
 */

#define LWSWITCH_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED                          0:0
#define LWSWITCH_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED_NO                      0x00
#define LWSWITCH_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED_YES                     0x01

/*
 * CTRL_LWSWITCH_I2C_GET_PORT_INFO
 *
 * Returns information for the I2C ports.
 *
 *   info
 *     This parameter is an output from the command and is ignored as an
 *     input.  The port numbers here are 0-indexed as opposed to 1-indexed.
 *
 */
typedef struct
{
    LwU32 info[LWSWITCH_CTRL_NUM_I2C_PORTS];

} LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS;

/*
 * LWSWITCH_CTRL_I2C_DEVICE_INFO
 *
 * This structure describes the basic I2C Device information.
 *
 *   type
 *     This field return the type of device LWSWITCH_I2C_DEVICE_<xyz>
 *   i2cAddress
 *     This field contains the 7 bit/10 bit address of the I2C device.
 *   i2cLogicalPort
 *     This field contains the Logical port of the I2C device.
 */

typedef enum
{
    LWSWITCH_I2C_PORT_I2CA      = 0,
    LWSWITCH_I2C_PORT_I2CB,
    LWSWITCH_I2C_PORT_I2CC,
    LWSWITCH_I2C_PORT_I2CD
} LWSWITCH_I2C_PORT_TYPE;

typedef enum
{
    LWSWITCH_I2C_DEVICE_UNKNOWN             = 0,

    // THERMAL CHIPS
    LWSWITCH_I2C_DEVICE_ADT7473             = 0x0A,
    LWSWITCH_I2C_DEVICE_ADT7461             = 0x0D,
    LWSWITCH_I2C_DEVICE_TMP451              = 0x0E,

    // POWER CONTROLLERS: SMBUS
    LWSWITCH_I2C_DEVICE_NCT3933U            = 0x4B,

    // POWER SENSORS
    LWSWITCH_I2C_DEVICE_INA3221             = 0x4E,

    // GENERAL PURPOSE GPIO CONTROLLERS
    LWSWITCH_I2C_DEVICE_TCA6408             = 0x61,

    // GPIO controller used on the optical carrier boards
    LWSWITCH_I2C_DEVICE_PCAL9538            = 0x62,

    // Data ROMs
    LWSWITCH_I2C_DEVICE_AT24C02C            = 0xA0,
    LWSWITCH_I2C_DEVICE_AT24CM02,
    LWSWITCH_I2C_DEVICE_AT88SC25616C,
    LWSWITCH_I2C_DEVICE_AT24C02D,

    // OSFP Devices
    LWSWITCH_I2C_DEVICE_CMIS4_MODULE       = 0xB0,

    // LED Drivers
    LWSWITCH_I2C_DEVICE_PCA9685BS          = 0xC0,
    LWSWITCH_I2C_DEVICE_TCA6507            = 0xC1,

    // ROM Devices
    LWSWITCH_I2C_DEVICE_CCI_PARTITION_0_IPMI_FRU  = 0xA2,
    LWSWITCH_I2C_DEVICE_CCI_PARTITION_1_IPMI_FRU  = 0xA3,

    // CDFP Devices (for LWSwitch)
    LWSWITCH_I2C_DEVICE_CDFP_LOWER_PADDLE   = 0xD2,
    LWSWITCH_I2C_DEVICE_CDFP_UPPER_PADDLE   = 0xD3,

    LWSWITCH_I2C_DEVICE_SKIP                = 0xFF

} LWSWITCH_I2C_DEVICE_TYPE;

typedef struct
{
    LWSWITCH_I2C_DEVICE_TYPE  type;
    LwU32  i2cAddress;
    LWSWITCH_I2C_PORT_TYPE  i2cPortLogical;
} LWSWITCH_CTRL_I2C_DEVICE_INFO;

/* Maximum number of I2C devices in DCB */
#define LWSWITCH_CTRL_I2C_MAX_DEVICES             32

/*
 * CTRL_LWSWITCH_I2C_TABLE_GET_DEV_INFO
 *
 * RM Control to get I2C device info from the DCB I2C Devices Table.
 *
 *   i2cDevCount
 *     The value of this parameter will give the number of valid
 *     I2C devices returned in structure.
 *
 *   i2cDevInfo[]
 *     For each device the control call will report the device info
 *
 */
typedef struct
{
    LwU8   i2cDevCount;
    LWSWITCH_CTRL_I2C_DEVICE_INFO i2cDevInfo[LWSWITCH_CTRL_I2C_MAX_DEVICES];
} LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS;

//! Maximum size of index.
#define LWSWITCH_CTRL_I2C_INDEX_LENGTH_MAX                      4

/*! Set if the command should begin with a START.  For a transactional
 *  interface (highly recommended), this should always be _SEND.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_START                          0:0
#define LWSWITCH_CTRL_I2C_FLAGS_START_NONE                       0
#define LWSWITCH_CTRL_I2C_FLAGS_START_SEND                       1

/*!
 *  Indicate whether to send a repeated start between the index and
 *  message phrases.
 *
 *  This flag will send a restart between each index and message.  This should
 *  be set for reads, but rarely (if ever) for writes.
 *
 *  A RESTART is required when switching directions; this is called a combined
 *  format.  These are typically used in indexed read commands, where an index
 *  is written to the device to indicate what register(s) to read, and then
 *  the register is read.  Almost always, indexed writes do not require a
 *  restart, though some devices will accept them.  However, this flag should
 *  be used for writes in the rare case where a restart should be sent between
 *  the last index and the message.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_RESTART                        1:1
#define LWSWITCH_CTRL_I2C_FLAGS_RESTART_NONE                     0
#define LWSWITCH_CTRL_I2C_FLAGS_RESTART_SEND                     1

/*! Set if the command should conclude with a STOP.  For a transactional
 *  interface (highly recommended), this should always be _SEND.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_STOP                           2:2
#define LWSWITCH_CTRL_I2C_FLAGS_STOP_NONE                        0
#define LWSWITCH_CTRL_I2C_FLAGS_STOP_SEND                        1

/*! The slave addressing mode: 7-bit (most common) or 10-bit.  It is possible
 *  but not recommended to send no address at all using _NONE.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE                   4:3
#define LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_NO_ADDRESS          0
#define LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT                1
#define LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT               2

//! The length of the index.  If length is 0, no index will be sent.
#define LWSWITCH_CTRL_I2C_FLAGS_INDEX_LENGTH                   7:5
#define LWSWITCH_CTRL_I2C_FLAGS_INDEX_LENGTH_ZERO                0
#define LWSWITCH_CTRL_I2C_FLAGS_INDEX_LENGTH_ONE                 1
#define LWSWITCH_CTRL_I2C_FLAGS_INDEX_LENGTH_TWO                 2
#define LWSWITCH_CTRL_I2C_FLAGS_INDEX_LENGTH_THREE               3
#define LWSWITCH_CTRL_I2C_FLAGS_INDEX_LENGTH_MAXIMUM             LWSWITCH_CTRL_I2C_INDEX_LENGTH_MAX

/*! The flavor to use: software bit-bang or hardware controller.  The hardware
 *  controller is faster, but is not necessarily available or capable.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_FLAVOR                         8:8
#define LWSWITCH_CTRL_I2C_FLAGS_FLAVOR_HW                        0
#define LWSWITCH_CTRL_I2C_FLAGS_FLAVOR_SW                        1

/*! The target speed at which to drive the transaction at.
 *
 *  Note: The lib reserves the right to lower the speed mode if the I2C master
 *  implementation cannot handle the speed given.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE                    11:9
#define LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_DEFAULT      0x00000000
#define LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_100KHZ       0x00000003
#define LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_200KHZ       0x00000004
#define LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_300KHZ       0x00000005
#define LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_400KHZ       0x00000006
#define LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_1000KHZ      0x00000007

/*
 * LWSWITCH_CTRL_I2C_FLAGS_TRANSACTION_MODE
 *   A client uses this field to specify a transaction mode.
 *   Possible values are:
 *     LWSWITCH_CTRL_I2C_FLAGS_TRANSACTION_MODE_NORMAL
 *       The default, this value indicates to use the normal I2C transaction
 *       mode which will involve read/write operations depending on client's
 *       needs.
 *     LWSWITCH_CTRL_I2C_FLAGS_TRANSACTION_MODE_PING
 *       This value specifies that the device only needs to be pinged. No need
 *       of performing a complete read/write transaction. This will address
 *       the device to be pinged but not send any data. On receiving an ACK,
 *       we will get a confirmation on the device's availability.
 *       PING requires that:
 *          _START   = _SEND
 *          _RESTART = _NONE
 *          _STOP    = _SEND
 *          _ADDRESS_MODE != _NO_ADDRESS
 *          _INDEX_LENGTH = _ZERO
 *          messageLength = 0
 */
#define LWSWITCH_CTRL_I2C_FLAGS_TRANSACTION_MODE                          12:12
#define LWSWITCH_CTRL_I2C_FLAGS_TRANSACTION_MODE_NORMAL             (0x00000000)
#define LWSWITCH_CTRL_I2C_FLAGS_TRANSACTION_MODE_PING               (0x00000001)

/*!
 * Block Reads/Writes: There are two different protocols for reading/writing >2
 * byte sets of data to/from a slave device.  The SMBus specification section
 * 5.5.7 defines "Block Reads/Writes" in which the first byte of the payload
 * specifies the size of the data to be read/written s.t. payload_size =
 * data_size + 1.  However, many other devices depend on the master to already
 * know the size of the data being accessed (i.e. SW written with knowledge of
 * the device's I2C register spec) and skip this overhead.  This second behavior
 * is actually the default behavior of all the lib's I2C interfaces.
 *
 * Setting this bit will enable the block protocol for reads and writes for size
 * >2.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_BLOCK_PROTOCOL               17:17
#define LWSWITCH_CTRL_I2C_FLAGS_BLOCK_PROTOCOL_DISABLED 0x00000000
#define LWSWITCH_CTRL_I2C_FLAGS_BLOCK_PROTOCOL_ENABLED  0x00000001

/*!
 * LWSWITCH_CTRL_I2C_FLAGS_RESERVED
 *   A client must leave this field as 0, as it is reserved for future use.
 */
#define LWSWITCH_CTRL_I2C_FLAGS_RESERVED                    31:18

#define LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX                256

/*
 * CTRL_LWSWITCH_I2C_INDEXED
 *
 * Perform a basic I2C transaction synchronously.
 *
 *   portId
 *     This field must be specified by the client to indicate the logical
 *     port/bus for which the transaction is requested.
 *
 *   bIsRead
 *     This field must be specified by the client to indicate whether the
 *     command is a write (FALSE) or a read (TRUE).
 *
 *   flags
 *     This parameter specifies optional flags used to control certain modal
 *     features such as target speed and addressing mode.  The lwrrently
 *     defined fields are described previously; see LWSWITCH_CTRL_I2C_FLAGS_*.
 *
 *   acquirer
 *     The ID of the client that is trying to take control of the I2C module.
 *
 *   address
 *     The address of the I2C slave.  The address should be shifted left by
 *     one.  For example, the I2C address 0x50, often used for reading EDIDs,
 *     would be stored here as 0xA0.  This matches the position within the
 *     byte sent by the master, as the last bit is reserved to specify the
 *     read or write direction.
 *
 *   index
 *     This parameter, required of the client if index is one or more,
 *     specifies the index to be written.  The buffer should be arranged such
 *     that index[0] will be the first byte sent.
 *
 *   messageLength
 *     This parameter, required of the client, specifies the number of bytes to
 *     read or write from the slave after the index is written.
 *
 *   message
 *     This parameter, required of the client, specifies the data to be written
 *     to the slave.  The buffer should be arranged such that message[0] will
 *     be the first byte read or written.  If the transaction is a read, then
 *     it will follow the combined format described in the I2C specification.
 *     If the transaction is a write, the message will immediately follow the
 *     index without a restart.
 *
 */
typedef struct
{
    LwU8  port;
    LwU8  bIsRead;
    LwU16 address;
    LwU32 flags;
    LwU32 acquirer;

    LwU8 index[LWSWITCH_CTRL_I2C_INDEX_LENGTH_MAX];

    LwU32 messageLength;
    LwU8  message[LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX];
} LWSWITCH_CTRL_I2C_INDEXED_PARAMS;

/*
 * CTRL_LWSWITCH_GET_VOLTAGE
 *
 * Zero(0) indicates that a measurement is not available on the current platform.
 *
 */

typedef struct
{
    LwU32 vdd_mv;
    LwU32 dvdd_mv;
    LwU32 hvdd_mv;
} LWSWITCH_CTRL_GET_VOLTAGE_PARAMS;

/*
 * CTRL_LWSWITCH_CONFIG_EOM
 *
 * This command issues a DLCMD to minion by writing to LW_PMINION_SCRATCH_SWRW_0
 * then calls an EOM DLCMD on minion for the desired link. Only one DLCMD at a
 * time can be issued to any given link.
 *
 * Params Packing is as follows:
 * TODO
 */

typedef struct
{
    LwU32 link;
    LwU32 params;
} LWSWITCH_CTRL_CONFIG_EOM;

/*
 * LW2080_CTRL_CMD_LWLINK_READ_UPHY_PAD_LANE_REG
 *
 * This command packs the lane and addr values into LW_PMINION_MISC_0_SCRATCH_SWRW_0
 * and then issues a READPADLANEREG DLCMD to minion for the desired link. Only one DLCMD 
 * at a time can be issued to any given link.
 * 
 * After this command completes it is necessary to read the appropriate
 * LW_PLWL_BR0_PAD_CTL_7_CFG_RDATA register to retrieve the results of the read
 *
 * [in] linkId
 *     Link whose pad lane register is being read
 * [in] lane
 *     Lane whose pad lane register is being read
 * [in] addr
 *     Address of the pad lane register to read
 * [out] phy_config_data
 *     Value of phy config register
 */
typedef struct
{
    LwU8  link;
    LwU8  lane;
    LwU16 addr;
    LwU32 phy_config_data;
} LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG;

/*
 * CTRL_LWSWITCH_GET_LWLINK_ERROR_DATA
 */

#define LWSWITCH_LWLINK_ARCH_ERROR_NONE             0
#define LWSWITCH_LWLINK_ARCH_ERROR_GENERIC          1
#define LWSWITCH_LWLINK_ARCH_ERROR_HW_FATAL         2
#define LWSWITCH_LWLINK_ARCH_ERROR_HW_CORRECTABLE   3
#define LWSWITCH_LWLINK_ARCH_ERROR_HW_UNCORRECTABLE 4

#define LWSWITCH_LWLINK_HW_ERROR_NONE               0x0
#define LWSWITCH_LWLINK_HW_GENERIC                  0x1
#define LWSWITCH_LWLINK_HW_INGRESS                  0x2
#define LWSWITCH_LWLINK_HW_EGRESS                   0x3
#define LWSWITCH_LWLINK_HW_FSTATE                   0x4
#define LWSWITCH_LWLINK_HW_TSTATE                   0x5
#define LWSWITCH_LWLINK_HW_ROUTE                    0x6
#define LWSWITCH_LWLINK_HW_NPORT                    0x7
#define LWSWITCH_LWLINK_HW_LWLCTRL                  0x8
#define LWSWITCH_LWLINK_HW_LWLIPT                   0x9
#define LWSWITCH_LWLINK_HW_LWLTLC                   0xA
#define LWSWITCH_LWLINK_HW_DLPL                     0xB
#define LWSWITCH_LWLINK_HW_AFS                      0xC
#define LWSWITCH_LWLINK_HW_MINION                   0xD
#define LWSWITCH_LWLINK_HW_HOST                     0xE
#define LWSWITCH_LWLINK_HW_NXBAR                    0XF
#define LWSWITCH_LWLINK_HW_SOURCETRACK              0x10

typedef LwU32 LWSWITCH_LWLINK_ARCH_ERROR;
typedef LwU32 LWSWITCH_LWLINK_HW_ERROR;

#define LWSWITCH_LWLINK_ERROR_COUNT_SIZE   64

typedef struct
{
    LwU32   linkNumber;
    LwU32   laneNumber;
    LWSWITCH_LWLINK_ARCH_ERROR  enumArchError;
    LWSWITCH_LWLINK_HW_ERROR    enumHWError;
} LWSWITCH_LWLINK_ERROR;

typedef struct
{
    LwU32                   errorCount;         /* out */
    LWSWITCH_LWLINK_ERROR   error[LWSWITCH_LWLINK_ERROR_COUNT_SIZE];
} LWSWITCH_GET_LWLINK_ERROR_DATA;

/*
 * CTRL_LWSWITCH_GET_LWLINK_CAPS
 *
 * This command returns the LWLink capabilities supported by the subdevice.
 *
 *   capsTbl
 *     This is bit field for getting different global caps. The individual
 *     bitfields are specified by LWSWITCH_LWLINK_CAPS_* 
 *   lowestLwlinkVersion
 *     This field specifies the lowest supported LWLink version for this
 *     subdevice.
 *   highestLwlinkVersion
 *     This field specifies the highest supported LWLink version for this
 *     subdevice.
 *   lowestNciVersion
 *     This field specifies the lowest supported NCI version for this
 *     subdevice.
 *   highestNciVersion
 *     This field specifies the highest supported NCI version for this
 *     subdevice.
 *   enabledLinkMask
 *     This field provides a bitfield mask of LWLink links enabled on this
 *     subdevice.
 *   activeRepeaterMask
 *     This field provides a bitfield mask of LWLink links enabled on this
 *     subdevice.
 *
 */
typedef struct
{
    LwU32   capsTbl;

    LwU8    lowestLwlinkVersion;
    LwU8    highestLwlinkVersion;
    LwU8    lowestNciVersion;
    LwU8    highestNciVersion;

    LW_DECLARE_ALIGNED(LwU64 enabledLinkMask, 8);
    LW_DECLARE_ALIGNED(LwU64 activeRepeaterMask, 8);
} LWSWITCH_GET_LWLINK_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LWSWITCH_LWLINK_GET_CAP(tbl,c)              (((LwU8)tbl[(1?c)]) & (0?c))

/*
 * LWSWITCH_LWLINK_CAPS_*
 *
 *   SUPPORTED
 *     Set if LWLink is present and supported on this subdevice, LW_FALSE
 *     otherwise. This field is used for *global* caps only and NOT for
 *     per-link caps
 *   P2P_SUPPORTED
 *     Set if P2P over LWLink is supported on this subdevice, LW_FALSE
 *     otherwise.
 *   SYSMEM_ACCESS
 *     Set if sysmem can be accessed over LWLink on this subdevice, LW_FALSE
 *     otherwise.
 *   PEER_ATOMICS
 *     Set if P2P atomics are supported over LWLink on this subdevice, LW_FALSE
 *     otherwise.
 *   SYSMEM_ATOMICS
 *     Set if sysmem atomic transcations are supported over LWLink on this
 *     subdevice, LW_FALSE otherwise.
 *   PEX_TUNNELING
 *     Set if PEX tunneling over LWLink is supported on this subdevice,
 *     LW_FALSE otherwise.
 *   SLI_BRIDGE
 *     GLOBAL: Set if SLI over LWLink is supported on this subdevice, LW_FALSE
 *     otherwise.
 *     LINK:   Set if SLI over LWLink is supported on a link, LW_FALSE
 *     otherwise.
 *   SLI_BRIDGE_SENSABLE
 *     GLOBAL: Set if the subdevice is capable of sensing SLI bridges, LW_FALSE
 *     otherwise.
 *     LINK:   Set if the link is capable of sensing an SLI bridge, LW_FALSE
 *     otherwise.
 *   POWER_STATE_L0
 *     Set if L0 is a supported power state on this subdevice/link, LW_FALSE
 *     otherwise.
 *   POWER_STATE_L1
 *     Set if L1 is a supported power state on this subdevice/link, LW_FALSE
 *     otherwise.
 *   POWER_STATE_L2
 *     Set if L2 is a supported power state on this subdevice/link, LW_FALSE
 *     otherwise.
 *   POWER_STATE_L3
 *     Set if L3 is a supported power state on this subdevice/link, LW_FALSE
 *     otherwise.
 *   VALID
 *     Set if this link is supported on this subdevice, LW_FALSE otherwise.
 *     This field is used for *per-link* caps only and NOT for global caps.
 *
 */

/* caps format is byte_index:bit_mask */
#define LWSWITCH_LWLINK_CAPS_SUPPORTED                          0:0x01
#define LWSWITCH_LWLINK_CAPS_P2P_SUPPORTED                      0:0x02
#define LWSWITCH_LWLINK_CAPS_SYSMEM_ACCESS                      0:0x04
#define LWSWITCH_LWLINK_CAPS_P2P_ATOMICS                        0:0x08
#define LWSWITCH_LWLINK_CAPS_SYSMEM_ATOMICS                     0:0x10
#define LWSWITCH_LWLINK_CAPS_PEX_TUNNELING                      0:0x20
#define LWSWITCH_LWLINK_CAPS_SLI_BRIDGE                         0:0x40
#define LWSWITCH_LWLINK_CAPS_SLI_BRIDGE_SENSABLE                0:0x80
#define LWSWITCH_LWLINK_CAPS_POWER_STATE_L0                     1:0x01
#define LWSWITCH_LWLINK_CAPS_POWER_STATE_L1                     1:0x02
#define LWSWITCH_LWLINK_CAPS_POWER_STATE_L2                     1:0x04
#define LWSWITCH_LWLINK_CAPS_POWER_STATE_L3                     1:0x08
#define LWSWITCH_LWLINK_CAPS_VALID                              1:0x10

/*
 * Size in bytes of lwlink caps table.  This value should be one greater
 * than the largest byte_index value above.
 */
#define LWSWITCH_LWLINK_CAPS_TBL_SIZE                           2

#define LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_ILWALID      (0x00000000)
#define LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_1_0          (0x00000001)
#define LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_2_0          (0x00000002)
#define LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_2_2          (0x00000004)
#define LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_3_0          (0x00000005)
#define LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_3_1          (0x00000006)
#define LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_4_0          (0x00000007)

#define LWSWITCH_LWLINK_CAPS_NCI_VERSION_ILWALID         (0x00000000)
#define LWSWITCH_LWLINK_CAPS_NCI_VERSION_1_0             (0x00000001)
#define LWSWITCH_LWLINK_CAPS_NCI_VERSION_2_0             (0x00000002)
#define LWSWITCH_LWLINK_CAPS_NCI_VERSION_2_2             (0x00000004)
#define LWSWITCH_LWLINK_CAPS_NCI_VERSION_3_0             (0x00000005)
#define LWSWITCH_LWLINK_CAPS_NCI_VERSION_3_1             (0x00000006)
#define LWSWITCH_LWLINK_CAPS_NCI_VERSION_4_0             (0x00000007)

/*
 * CTRL_LWSWITCH_GET_ERR_INFO
 *     This command is used to query the LWLINK error information
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

/*
 * LWSWITCH_LWLINK_ERR_INFO
 *   Error information per link
 *
 *   TLErrlog
 *     Returns the error mask for LWLINK TL errors
 *     Used in Pascal
 *
 *   TLIntrEn
 *     Returns the intr enable mask for LWLINK TL errors
 *     Used in Pascal
 *
 *   TLCTxErrStatus0
 *     Returns the TLC Tx Error Mask 0
 *     Used in Volta
 *
 *   TLCRxErrStatus0
 *     Returns the TLC Rx Error Mask 0
 *     Used in Volta
 *
 *   TLCRxErrStatus1
 *     Returns the TLC Rx Error Mask 1
 *     Used in Volta
 *
 *   TLCTxErrLogEn0
 *     Returns the TLC Tx Error Log En 0
 *     Used in Volta
 *
 *   TLCRxErrLogEn0
 *     Returns the TLC Rx Error Log En 0
 *     Used in Volta
 *
 *   TLCRxErrLogEn1
 *     Returns the TLC Rx Error Log En 1
 *     Used in Volta
 *
 *   MIFTxErrStatus0
 *     Returns the MIF Rx Error Mask 0
 *     Used in Volta
 *
 *   MIFRxErrStatus0
 *     Returns the MIF Tx Error Mask 0
 *     Used in Volta
 *
 *   DLSpeedStatusTx
 *     Returns the LWLINK DL speed status for sublink Tx
 *
 *   DLSpeedStatusRx
 *     Returns the LWLINK DL speed status for sublink Rx
 *
 *   bExcessErrorDL
 *     Returns true for excessive error rate interrupt from DL
 */
typedef struct
{
    LwU32   TLErrlog;
    LwU32   TLIntrEn;
    LwU32   TLCTxErrStatus0;
    LwU32   TLCRxErrStatus0;
    LwU32   TLCRxErrStatus1;
    LwU32   TLCTxErrLogEn0;
    LwU32   TLCRxErrLogEn0;
    LwU32   TLCRxErrLogEn1;
    LwU32   MIFTxErrStatus0;
    LwU32   MIFRxErrStatus0;
    LwU32   DLSpeedStatusTx;
    LwU32   DLSpeedStatusRx;
    LwBool  bExcessErrorDL;
} LWSWITCH_LWLINK_ERR_INFO;

/* Extract the error status bit for a given TL error index i */
#define LWSWITCH_LWLINK_GET_TL_ERRLOG_BIT(intr, i)       ((LWBIT(i) & (intr)) >> i)

/* Extract the intr enable bit for a given TL error index i */
#define LWSWITCH_LWLINK_GET_TL_INTEN_BIT(intr, i)        LWSWITCH_LWLINK_GET_TL_ERRLOG_BIT(intr, i)

/* Error status values for a given LWLINK TL error */
#define LWSWITCH_LWLINK_TL_ERRLOG_TRUE                  (0x00000001)
#define LWSWITCH_LWLINK_TL_ERRLOG_FALSE                 (0x00000000)

/* Intr enable/disable for a given LWLINK TL error */
#define LWSWITCH_LWLINK_TL_INTEN_TRUE                   (0x00000001)
#define LWSWITCH_LWLINK_TL_INTEN_FALSE                  (0x00000000)

/* LWLINK TL interrupt enable fields for errors */
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXDLDATAPARITYEN                 0
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXDLCTRLPARITYEN                 1
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXPROTOCOLEN                     2
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXOVERFLOWEN                     3
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXRAMDATAPARITYEN                4
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXRAMHDRPARITYEN                 5
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXRESPEN                         6
#define LWSWITCH_LWLINK_TL_INTEN_IDX_RXPOISONEN                       7
#define LWSWITCH_LWLINK_TL_INTEN_IDX_TXRAMDATAPARITYEN                8
#define LWSWITCH_LWLINK_TL_INTEN_IDX_TXRAMHDRPARITYEN                 9
#define LWSWITCH_LWLINK_TL_INTEN_IDX_DLFLOWPARITYEN                  10
#define LWSWITCH_LWLINK_TL_INTEN_IDX_DLHDRPARITYEN                   12
#define LWSWITCH_LWLINK_TL_INTEN_IDX_TXCREDITEN                      13
#define LWSWITCH_LWLINK_TL_INTEN_IDX_MAX                             14

/* LWLINK TL error fields */
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXDLDATAPARITYERR               0
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXDLCTRLPARITYERR               1
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXPROTOCOLERR                   2
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXOVERFLOWERR                   3
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXRAMDATAPARITYERR              4
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXRAMHDRPARITYERR               5
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXRESPERR                       6
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_RXPOISONERR                     7
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_TXRAMDATAPARITYERR              8
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_TXRAMHDRPARITYERR               9
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_DLFLOWPARITYERR                10
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_DLHDRPARITYERR                 12
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_TXCREDITERR                    13
#define LWSWITCH_LWLINK_TL_ERRLOG_IDX_MAX                            14

/* LWLINK DL speed status for sublink Tx*/
#define LWSWITCH_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_HS          (0x00000000)
#define LWSWITCH_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SINGLE_LANE (0x00000004)
#define LWSWITCH_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_TRAIN       (0x00000005)
#define LWSWITCH_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SAFE        (0x00000006)
#define LWSWITCH_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_OFF         (0x00000007)

/* LWLINK DL speed status for sublink Rx*/
#define LWSWITCH_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_HS          (0x00000000)
#define LWSWITCH_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SINGLE_LANE (0x00000004)
#define LWSWITCH_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_TRAIN       (0x00000005)
#define LWSWITCH_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SAFE        (0x00000006)
#define LWSWITCH_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_OFF         (0x00000007)

#define LWSWITCH_LWLINK_MAX_LINKS                                    64

/*
 *   LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS
 *
 *   linkMask
 *     Returns the mask of links enabled
 *
 *   linkErrInfo
 *     Returns the error information for all the links
 */
typedef struct
{
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
    LWSWITCH_LWLINK_ERR_INFO linkErrInfo[LWSWITCH_LWLINK_MAX_LINKS];
} LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS;

/* Maximum number of entries in maskInfoList[] */
#define LWSWITCH_IRQ_INFO_LIST_SIZE 16
/*
 * Structure to store the IRQ data.
 * irqPendingOffset
 *     Register to read IRQ pending status
 * irqEnabledOffset
 *     Register to read the enabled interrupts
 * irqEnableOffset
 *     Register to write to enable interrupts
 * irqDisableOffset
 *     Register to write to disable interrupts
 */
typedef struct
{
    LwU32 irqPendingOffset;
    LwU32 irqEnabledOffset;
    LwU32 irqEnableOffset; 
    LwU32 irqDisableOffset;
} LWSWITCH_IRQ_INFO;

/*
 * CTRL_LWSWITCH_GET_IRQ_INFO
 *   This command gets the IRQ information to support interrupt handling for clients like MODS
 * 
 * [out] maskInfoCount
 *  Tells the number of valid entires in maskInfoList[] from entry 0.
 * [out] maskInfoList[]
 *  Stores the IRQ related data.
 */
typedef struct
{
    LwU32             maskInfoCount;
    LWSWITCH_IRQ_INFO maskInfoList[LWSWITCH_IRQ_INFO_LIST_SIZE]; 
} LWSWITCH_GET_IRQ_INFO_PARAMS;

/*
 * CTRL_LWSWITCH_CLEAR_COUNTERS
 *  This command clears/resets the counters for the specified types.
 *
 * [in] linkMask
 *  This parameter specifies for which links we want to clear the 
 *  counters.
 *
 * [in] counterMask
 *  This parameter specifies the input mask for desired counters to be
 *  cleared. Note that all counters cannot be cleared.
 *
 *  NOTE: Bug# 2098529: On Turing all DL errors and LP counters are cleared
 *        together. They cannot be cleared individually per error type. RM
 *        would possibly move to a new API on Ampere and beyond
 */  

typedef struct
{
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
    LwU32  counterMask; 
} LWSWITCH_LWLINK_CLEAR_COUNTERS_PARAMS;

/*
 * CTRL_LWSWITCH_PEX_SET_EOM
 *
 * UPHY EOM(Eye Opening Measurements) parameters on LwSwitch.
 * These are used to control the Bit Error Rate(BER) target for the EOM process in UPHY.
 *
 * Parameters:
 *    mode [IN]
 *      Mode of EOM.
 *    nblks [IN]
 *      Number of blocks.
 *   nerrs [IN]
 *      Number of Errors.
 */

typedef struct lwswitch_pex_ctrl_eom
{
    LwU8 mode;
    LwU8 nblks;
    LwU8 nerrs;
    LwU8 berEyeSel;
} LWSWITCH_PEX_CTRL_EOM;

/*
 * CTRL_LWSWITCH_PEX_GET_EOM_STATUS
 *
 * This command runs EOM sequence in the SOE and returns EOM status.
 */
#define LWSWITCH_CTRL_PEX_MAX_NUM_LANES  32

typedef struct
{
    LwU8   mode;
    LwU8   nblks;
    LwU8   nerrs;
    LwU8   berEyeSel;
    LwU32  laneMask;
    LwU16  eomStatus[LWSWITCH_CTRL_PEX_MAX_NUM_LANES];
} LWSWITCH_PEX_GET_EOM_STATUS_PARAMS;

/*
 * CTRL_LWSWITCH_PEX_GET_UPHY_DLN_CFG_SPACE
 *
 * This call sends UPHY register's address and lane from the client
 * to the SOE and returns back the register value.
 *
 * Parameters:
 *    regAddress [IN]
 *      Register address whose value is to be retrieved.
 *   laneSelectMask [IN]
 *      Mask of lanes to read from.
 *   regValue [OUT]
 *      Value of register address.
 */

typedef struct lwswitch_get_pex_uphy_dln_cfg_space_params
{
    LwU32 regAddress;
    LwU32 laneSelectMask;
    LwU16 regValue;
} LWSWITCH_GET_PEX_UPHY_DLN_CFG_SPACE_PARAMS;

/*
 * CTRL_LWSWITCH_SET_THERMAL_SLOWDOWN
 *
 * Control to force and revert slowdown on the links.
 *
 * This API is not supported on SV10.
 * On Limerock, the SOE sets all the links to Single Lane Mode(SLM).
 *
 * Parameters:
 *   slowdown [IN]
 *     control to force or revert slowdown.
 *   periodUs [IN]
 *     slowdown time.
 */
typedef struct lwswitch_ctrl_set_thermal_slowdown
{
    LwBool slowdown;
    LwU32  periodUs;
} LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN;

/*
 * CTRL_LWSWITCH_SET_PCIE_LINK_SPEED
 *
 * This call sends SOE request to set PCIE link speed.
 *
 * Parameters:
 *    linkSpeed [IN]
 *      Pcie link speed
 */
typedef struct lwswitch_set_pcie_link_speed_params
{
    LwU32 linkSpeed;
} LWSWITCH_SET_PCIE_LINK_SPEED_PARAMS;

/* 
 * PCIE link speeds
 */
#define LWSWITCH_BIF_LINK_SPEED_ILWALID      (0x00)
#define LWSWITCH_BIF_LINK_SPEED_GEN1PCIE     (0x01)
#define LWSWITCH_BIF_LINK_SPEED_GEN2PCIE     (0x02)
#define LWSWITCH_BIF_LINK_SPEED_GEN3PCIE     (0x03)
#define LWSWITCH_BIF_LINK_SPEED_GEN4PCIE     (0x04)
#define LWSWITCH_BIF_LINK_SPEED_GEN5PCIE     (0x05)

/*
 * Structure to store CCI capabilities
 *
 * Parameters:
 *   identifier
 *     Type of serial module of SFF-8024.
 *   rev_compliance
 *     CMIS revision - 0x01 indicates version 0.1, 0x21 indicates version 2.1.
 *   flat_mem
 *     Is upper memory flat or paged.
 *   twi_max_speed_khz
 *     Maximum two-wire serial speed supported by the module.
 *   host_interface_id
 *     Host Electrical Interface ID
 *   host_lane_count
 *     Host Lane Count
 *   module_interface_id
 *     Module Media Interface ID
 *   module_lane_count
 *     Media Lane Count
 */
typedef struct lwswitch_cci_capabilities
{
    LwU8 identifier;
    LwU8 rev_compliance;
    LwBool flat_mem;
    LwU16 twi_max_speed_khz;
    LwU8 host_interface_id;
    LwU8 host_lane_count;
    LwU8 module_interface_id;
    LwU8 module_lane_count;
} LWSWITCH_CCI_CAPABILITIES;

/*
 * CTRL_LWSWITCH_CCI_GET_CAPABILITIES
 *
 * Control to get cci capabilities of the transreciever.
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   link [IN]
 *     Link number
 *   capabilities [OUT]
 *     Stores the CCI capabilities
 */
typedef struct lwswitch_cci_get_capabilities_params
{
    LwU32 linkId;
    LWSWITCH_CCI_CAPABILITIES capabilities;
} LWSWITCH_CCI_GET_CAPABILITIES_PARAMS;

/*
 * CTRL_LWSWITCH_CCI_GET_TEMPERATURE
 *
 * Control to get cci capabilities of the transreciever.
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   link [IN]
 *    Link number
 *   temperature [OUT]
 *     Temperature of the transreviever module.
 */
typedef struct lwswitch_cci_get_temperature
{
    LwU32 linkId;
    LwTemp temperature;
}LWSWITCH_CCI_GET_TEMPERATURE;

#define LWSWITCH_CCI_FW_FLAGS_PRESENT        0:0
#define LWSWITCH_CCI_FW_FLAGS_PRESENT_NO       0
#define LWSWITCH_CCI_FW_FLAGS_PRESENT_YES      1
#define LWSWITCH_CCI_FW_FLAGS_ACTIVE         1:1
#define LWSWITCH_CCI_FW_FLAGS_ACTIVE_NO        0
#define LWSWITCH_CCI_FW_FLAGS_ACTIVE_YES       1
#define LWSWITCH_CCI_FW_FLAGS_COMMITED       2:2
#define LWSWITCH_CCI_FW_FLAGS_COMMITED_NO      0
#define LWSWITCH_CCI_FW_FLAGS_COMMITED_YES     1
#define LWSWITCH_CCI_FW_FLAGS_EMPTY          3:3
#define LWSWITCH_CCI_FW_FLAGS_EMPTY_NO         0
#define LWSWITCH_CCI_FW_FLAGS_EMPTY_YES        1

#define LWSWITCH_CCI_FW_IMAGE_A         0x0
#define LWSWITCH_CCI_FW_IMAGE_B         0x1
#define LWSWITCH_CCI_FW_IMAGE_FACTORY   0x2
#define LWSWITCH_CCI_FW_IMAGE_COUNT     0x3

/*
 * Structure to store FW revision parameters
 *
 * Parameters:
 *   status
 *     FW status flags
 *   image
 *     Firmware Image A/B/Factory.
 *   major
 *     FW major revision.
 *   minor
 *     FW minor revision.
 *   build
 *     FW build number.
 */
typedef struct lwswitch_cci_get_fw_revisions
{
    LwU8 flags;
    LwU8 major;
    LwU8 minor;
    LwU16 build;
} LWSWITCH_CCI_GET_FW_REVISIONS;

/*
 * CTRL_LWSWITCH_CCI_GET_FW_REVISIONS
 *
 * Control to get cci firmware revisions of the transreciever.
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   link [IN]
 *     Link number
 *   revisions [OUT]
 *     Stores the CCI FW revision params
 */
typedef struct lwswitch_cci_get_fw_revision_params
{
    LwU32 linkId;
    LWSWITCH_CCI_GET_FW_REVISIONS revisions[LWSWITCH_CCI_FW_IMAGE_COUNT];
} LWSWITCH_CCI_GET_FW_REVISION_PARAMS;

/*
 * Structure to store cci module State
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   bLReserved
 *          indicates if the module is in reserved state
 *   bLModuleLowPwrState
 *          indicates if the module is in low power state
 *   bLModulePwrUpState
 *          indicates if the module is in power up state
 *   bLModuleReadyState
 *          indicates if the module is ready
 *   bLModulePwrDnState
 *          indicates if the module is in power down state
 *   bLFaultState
 *          indicates if the module is in fault state
 */
typedef struct lwswitch_cci_module_state
{
    LwBool bLReserved;
    LwBool bLModuleLowPwrState;
    LwBool bLModulePwrUpState;
    LwBool bLModuleReadyState;
    LwBool bLModulePwrDnState;
    LwBool bLFaultState;
} LWSWITCH_CCI_MODULE_STATE;

/*
 * CTRL_LWSWITCH_CCI_GET_MODULE_STATE
 * 
 * Control to get cci module state information
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   link [IN]
 *     Link number
 *   info [OUT]
 *     module info
 */
typedef struct lwswitch_cci_get_module_state
{
    LwU32 linkId;
    LWSWITCH_CCI_MODULE_STATE info;
} LWSWITCH_CCI_GET_MODULE_STATE;

/*
 * Structure to store cci module Flags 
 *
 * This API is not supported on SV10.
 *
 */
typedef struct lwswitch_cci_module_flags
{
    LwBool bLCDBBlock2Complete;
    LwBool bLCDBBlock1Complete;
    
    LwBool bLModuleStateChange;
    LwBool bLModuleFirmwareFault;
    LwBool bDatapathFirmwareFault;

    LwBool bLTempHighAlarm;
    LwBool bLTempLowAlarm;
    LwBool bLTempHighWarn;
    LwBool bLTempLowWarn;

    LwBool bLVccHighAlarm;
    LwBool bLVccLowAlarm;
    LwBool bLVccHighWarn;
    LwBool bLVccLowWarn;

    LwBool bLAux1HighAlarm;
    LwBool bLAux1LowAlarm;
    LwBool bLAux1HighWarn;
    LwBool bLAux1LowWarn;

    LwBool bLAux2HighAlarm;
    LwBool bLAux2LowAlarm;
    LwBool bLAux2HighWarn;
    LwBool bLAux2LowWarn;

    LwBool bLAux3HighAlarm;
    LwBool bLAux3LowAlarm;
    LwBool bLAux3HighWarn;
    LwBool bLAux3LowWarn;

    LwBool bLVendorHighAlarm;
    LwBool bLVendorLowAlarm;
    LwBool bLVendorHighWarn;
    LwBool bLVendorLowWarn;
} LWSWITCH_CCI_MODULE_FLAGS;

/*
 * CTRL_LWSWITCH_CCI_GET_MODULE_FLAGS
 *
 * Structure to store cci module Flags 
 *
 * This API is not supported on SV10.
 * Parameters:
 *   link [IN]
 *     Link number
 *   flags [OUT]
 *     module flags
 *
 */
typedef struct lwswitch_cci_get_module_flags
{
    LwU32 linkId;
    LWSWITCH_CCI_MODULE_FLAGS flags;
} LWSWITCH_CCI_GET_MODULE_FLAGS;

/*
 * Structure to store cci module Voltage 
 *
 * This API is not supported on SV10.
 *
 */
typedef struct lwswitch_cci_voltage
{
    LwU16 voltage_mV;
} LWSWITCH_CCI_VOLTAGE;


/*
 * CTRL_LWSWITCH_CCI_GET_VOLTAGE
 *
 * Control to get cci module information
 *
 * This API is not supported on SV10.
 *
 * Parameters:
 *   link [IN]
 *     Link number
 *   Voltage [OUT]
 *     module voltage
 */
typedef struct lwswitch_cci_get_voltage
{
    LwU32 linkId;
    LWSWITCH_CCI_VOLTAGE voltage;
} LWSWITCH_CCI_GET_VOLTAGE;


/*
 * Internal CTRL call command list (all internal calls should start at 0xA0)
 *
 * Linux driver supports only 8-bit commands. MODS supports up to 32-bits.
 * Hence, if an IOCTL is MODS-only, use 0x100+ command.
 */

/* LWSwitch control utils (MODS and debug/develop drivers) */
#define CTRL_LWSWITCH_REGISTER_READ                0xA0
#define CTRL_LWSWITCH_REGISTER_WRITE               0xA1

/* LWSwitch control call interfaces (MODS only) */
#define CTRL_LWSWITCH_INJECT_LINK_ERROR            0x100
#define CTRL_LWSWITCH_READ_JTAG_CHAIN              0x101
#define CTRL_LWSWITCH_WRITE_JTAG_CHAIN             0x102
#define CTRL_LWSWITCH_PEX_GET_COUNTERS             0x103
#define CTRL_LWSWITCH_PEX_CLEAR_COUNTERS           0x104
#define CTRL_LWSWITCH_PEX_GET_LANE_COUNTERS        0x105
#define CTRL_LWSWITCH_I2C_GET_PORT_INFO            0x106
#define CTRL_LWSWITCH_I2C_GET_DEV_INFO             0x107
#define CTRL_LWSWITCH_I2C_INDEXED                  0x108
#define CTRL_LWSWITCH_GET_VOLTAGE                  0x109
#define CTRL_LWSWITCH_CONFIG_EOM                   0x10A
#define CTRL_LWSWITCH_GET_LWLINK_CAPS              0x10B
#define CTRL_LWSWITCH_CLEAR_COUNTERS               0x10C
#define CTRL_LWSWITCH_GET_ERR_INFO                 0x10D
#define CTRL_LWSWITCH_SET_PORT_TEST_MODE           0x10E
#define CTRL_LWSWITCH_GET_IRQ_INFO                 0x10F
#define CTRL_LWSWITCH_READ_UPHY_PAD_LANE_REG       0x110
#define CTRL_LWSWITCH_PEX_SET_EOM                  0x111
#define CTRL_LWSWITCH_PEX_GET_UPHY_DLN_CFG_SPACE   0x112
#define CTRL_LWSWITCH_SET_THERMAL_SLOWDOWN         0x113
#define CTRL_LWSWITCH_SET_PCIE_LINK_SPEED          0x114
#define CTRL_LWSWITCH_PEX_GET_EOM_STATUS           0x115
#define CTRL_LWSWITCH_CCI_GET_CAPABILITIES         0x116
#define CTRL_LWSWITCH_CCI_GET_TEMPERATURE          0x117
#define CTRL_LWSWITCH_CCI_GET_FW_REVISIONS         0x118
#define CTRL_LWSWITCH_CCI_GET_MODULE_STATE         0x119
#define CTRL_LWSWITCH_CCI_GET_MODULE_FLAGS         0x11A
#define CTRL_LWSWITCH_CCI_GET_VOLTAGE              0x11B
/* DO NOT ADD CODE AFTER THIS LINE */

#ifdef __cplusplus
}
#endif

#endif // _CTRL_DEVICE_INTERNAL_LWSWITCH_H_
