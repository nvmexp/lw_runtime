/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _PMGR_LWSWITCH_H_
#define _PMGR_LWSWITCH_H_

#define LWSWITCH_BITS_PER_BYTE       8

#define LWSWITCH_HIGH LW_TRUE
#define LWSWITCH_LOW  LW_FALSE

/*! Extract the first byte of a 10-bit address. */
#define LWSWITCH_GET_ADDRESS_10BIT_FIRST(a) ((LwU8)((((a) >> 8) & 0x6) | 0xF0))

/*! Extract the second byte of a 10-bit address. */
#define LWSWITCH_GET_ADDRESS_10BIT_SECOND(a) ((LwU8)(((a) >> 1) & 0xFF))

/*! Attaching read to read application interface */
#define LWSWITCH_I2C_READ(a,b) _lwswitch_i2c_i2cRead(device, a, b)

#define LWSWITCH_I2C_DELAY(a)    LWSWITCH_NSEC_DELAY(a)

#define LWSWITCH_MAX_I2C_PORTS       4

/*! bit 0 of address set indicates read cycle to follow */
#define LWSWITCH_I2C_READCYCLE                                              ((LwU8)0x01)

/*! Determine if an address is valid in the 7-bit address space. */
#define LWSWITCH_I2C_IS_7BIT_I2C_ADDRESS(a)                                ((a) <= 0xFF)

/*! Determine if an address is valid in the 10-bit address space. */
#define LWSWITCH_I2C_IS_10BIT_I2C_ADDRESS(a)                              ((a) <= 0x7FF)

// by-the-spec delay defaults (yields 100KHz)
#define LWSWITCH_I2C_PROFILE_STANDARD_tF              300
#define LWSWITCH_I2C_PROFILE_STANDARD_tR             1000
#define LWSWITCH_I2C_PROFILE_STANDARD_tSUDAT         1800    // actually, spec calls for (min) 250, but we've borrowed from tHDDAT
#define LWSWITCH_I2C_PROFILE_STANDARD_tHDDAT         1900    // actually, spec calls for (max) 3450, but we've loaned time to tSUDAT
#define LWSWITCH_I2C_PROFILE_STANDARD_tHIGH          4000
#define LWSWITCH_I2C_PROFILE_STANDARD_tSUSTO         4000
#define LWSWITCH_I2C_PROFILE_STANDARD_tHDSTA         4000
#define LWSWITCH_I2C_PROFILE_STANDARD_tSUSTA         4700
#define LWSWITCH_I2C_PROFILE_STANDARD_tBUF           4700
#define LWSWITCH_I2C_PROFILE_STANDARD_tLOW           4700    // LWSWITCH_I2C_PROFILE_STANDARD_tSUDAT + LWSWITCH_I2C_PROFILE_STANDARD_tR + LWSWITCH_I2C_PROFILE_STANDARD_tHDDAT
#define LWSWITCH_I2C_PROFILE_STANDARD_CYCLEPERIOD   10000    // LWSWITCH_I2C_PROFILE_STANDARD_tF + LWSWITCH_I2C_PROFILE_STANDARD_tLOW + LWSWITCH_I2C_PROFILE_STANDARD_tR + LWSWITCH_I2C_PROFILE_STANDARD_tHIGH

// by-the-spec delay defaults (yields 400KHz)
#define LWSWITCH_I2C_PROFILE_FAST_tF                  300
#define LWSWITCH_I2C_PROFILE_FAST_tR                  300
#define LWSWITCH_I2C_PROFILE_FAST_tSUDAT              200    // actually, spec calls for (min) 100, but we've borrowed from tHDDAT
#define LWSWITCH_I2C_PROFILE_FAST_tHDDAT              800    // actually, spec calls for (max) 900, but we've loaned time to tSUDAT
#define LWSWITCH_I2C_PROFILE_FAST_tHIGH               600
#define LWSWITCH_I2C_PROFILE_FAST_tSUSTO              600
#define LWSWITCH_I2C_PROFILE_FAST_tHDSTA              600
#define LWSWITCH_I2C_PROFILE_FAST_tSUSTA              600
#define LWSWITCH_I2C_PROFILE_FAST_tBUF               1300
#define LWSWITCH_I2C_PROFILE_FAST_tLOW               1300    // LWSWITCH_I2C_PROFILE_STANDARD_tSUDAT + LWSWITCH_I2C_PROFILE_STANDARD_tR + LWSWITCH_I2C_PROFILE_STANDARD_tHDDAT
#define LWSWITCH_I2C_PROFILE_FAST_CYCLEPERIOD        2500    // LWSWITCH_I2C_PROFILE_STANDARD_tF + LWSWITCH_I2C_PROFILE_STANDARD_tLOW + LWSWITCH_I2C_PROFILE_STANDARD_tR + LWSWITCH_I2C_PROFILE_STANDARD_tHIGH

/*!
 * The I2C specification does not specify any timeout conditions for clock
 * stretching, i.e. any device can hold down SCL as long as it likes so this
 * value needs to be adjusted on case by case basis.
 */
#define LWSWITCH_I2C_SCL_CLK_TIMEOUT_1200US  1200
#define LWSWITCH_I2C_SCL_CLK_TIMEOUT_1000KHZ    (LWSWITCH_I2C_SCL_CLK_TIMEOUT_100KHZ * 4)
#define LWSWITCH_I2C_SCL_CLK_TIMEOUT_400KHZ     (LWSWITCH_I2C_SCL_CLK_TIMEOUT_100KHZ * 4)
#define LWSWITCH_I2C_SCL_CLK_TIMEOUT_300KHZ     (LWSWITCH_I2C_SCL_CLK_TIMEOUT_100KHZ * 3)
#define LWSWITCH_I2C_SCL_CLK_TIMEOUT_200KHZ     (LWSWITCH_I2C_SCL_CLK_TIMEOUT_100KHZ * 2)
#define LWSWITCH_I2C_SCL_CLK_TIMEOUT_100KHZ     (LWSWITCH_I2C_SCL_CLK_TIMEOUT_1200US / 10)

/* A reasonable SCL timeout is five cycles at 20 KHz.  Full use should be rare
 * in devices, oclwrring when in the middle of a real-time task. That comes to
 * 25 clock cycles at 100 KHz, or 250 us. */
#define LWSWITCH_I2C_SCL_CLK_TIMEOUT_250US 250

/* We don't want I2C to deal with traffic slower than 20 KHz (50 us cycle).
 */
#define LWSWITCH_I2C_MAX_CYCLE_US 50

/* The longest HW I2C transaction: S BYTE*2 S BYTE*4 P, at 1 each for S/P, and
 * 9 for each byte (+ack). */
#define LWSWITCH_I2C_HW_MAX_CYCLES ((1 * 3) + (9 * 6))

/* We determine the HW operational timeout as the longest operation, plus two
 * long SCL clock stretches. */
#define I2C_HW_IDLE_TIMEOUT_NS (1000 * \
    ((LWSWITCH_I2C_MAX_CYCLE_US * LWSWITCH_I2C_HW_MAX_CYCLES) + (LWSWITCH_I2C_SCL_CLK_TIMEOUT_1200US * 2)))

//
// PMGR board configuration information
//

#define LWSWITCH_DESCRIBE_I2C_DEVICE(_port, _addr, _type, _rdWrAccessMask)        \
    {LWSWITCH_I2C_PORT ## _port, _addr, LWSWITCH_I2C_DEVICE ## _type, _rdWrAccessMask}

#define LWSWITCH_DESCRIBE_GPIO_PIN(_pin, _func, _hw_select, _misc_io)    \
    {_pin, LWSWITCH_GPIO_ENTRY_FUNCTION ## _func, _hw_select,   \
        LWSWITCH_GPIO_ENTRY_MISC_IO_ ## _misc_io}

/*! Structure containing a description of the I2C bus as needed by the software
 *  bit-banging implementation.
 */
typedef struct
{
    LwU32 sclOut;      // Bit number for SCL Output
    LwU32 sdaOut;      // Bit number for SDA Output

    LwU32 sclIn;       // Bit number for SCL Input
    LwU32 sdaIn;       // Bit number for SDA Input

    LwU32 port;        // Port number of the driving lines
    LwU32 lwrLine;     // Required for isLineHighFunction

    LwU32 regCache;    // Keeps the cache value of registers.
    //
    // The following timings are used as stand-ins for I2C spec timings, so
    // that different speed modes may share the same code.
    //
    LwU16 tF;
    LwU16 tR;
    LwU16 tSuDat;
    LwU16 tHdDat;
    LwU16 tHigh;
    LwU16 tSuSto;
    LwU16 tHdSta;
    LwU16 tSuSta;
    LwU16 tBuf;
    LwU16 tLow;
} LWSWITCH_I2C_SW_BUS;

/*! @brief Internal Command structure for HW I2C to perform I2C transaction */
typedef struct
{
    LwU32   port;
    LwU32   bRead;
    LwU32   cntl;
    LwU32   data;
    LwU32   bytesRemaining;
    LwS32   status;
    LwU8   *pMessage;
    LwBool  bBlockProtocol;
} LWSWITCH_I2C_HW_CMD, *PLWSWITCH_I2C_HW_CMD;


typedef enum
{
    LWSWITCH_I2C_ACQUIRER_NONE = 0,
    LWSWITCH_I2C_ACQUIRER_UNKNOWN,
    LWSWITCH_I2C_ACQUIRER_IOCTL,          // e.g. MODS                  
    LWSWITCH_I2C_ACQUIRER_EXTERNAL,       // e.g. Linux Direct
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LWSWITCH_I2C_ACQUIRER_CCI_INITIALIZE, // CCI Init/Startup
    LWSWITCH_I2C_ACQUIRER_CCI_TRAIN,      // Cable training
    LWSWITCH_I2C_ACQUIRER_CCI_UX,         // User interface e.g. LEDs
    LWSWITCH_I2C_ACQUIRER_CCI_SERVICE,    // e.g. ISR
    LWSWITCH_I2C_ACQUIRER_CCI_SMBPBI,     // OOB path
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

} LWSWITCH_I2C_ACQUIRER;

typedef enum {
    i2cProfile_Standard,
    i2cProfile_Fast,
    i2cProfile_End
} LWSWITCH_I2CPROFILE;

typedef enum
{
    pmgrReg_i2cAddr,
    pmgrReg_i2cCntl,
    pmgrReg_i2cTiming,
    pmgrReg_i2cOverride,
    pmgrReg_i2cPoll,
    pmgrReg_i2cData,
    pmgrReg_unsupported
} LWSWITCH_PMGRREG_TYPE;


// I2C Speed limits
#define LWSWITCH_I2C_SPEED_LIMIT_NONE                 LW_U16_MAX  //Close enough to not having a speed limit.
#define LWSWITCH_I2C_SPEED_1000KHZ                    1000
#define LWSWITCH_I2C_SPEED_400KHZ                     400
#define LWSWITCH_I2C_SPEED_300KHZ                     300
#define LWSWITCH_I2C_SPEED_200KHZ                     200
#define LWSWITCH_I2C_SPEED_100KHZ                     100

enum
{
    i2cSpeedLimit_dcb = 0,
    i2cSpeedLimit_ctrl,

    // Always leave as last element!
    LWSWITCH_I2C_SPEED_LIMIT_MAX_DEVICES
};


// Timing for I2C cycles (allows for possibility of tweaking timing)
typedef struct __LWSWITCH_LWSWITCH_I2CTIMING
{
    LwU32 tR;            // at 100KHz, normally 1000ns
    LwU32 tF;            // at 100KHz, normally  300ns
    LwU32 tHIGH;         // at 100KHz, normally 4000ns
    LwU32 tSUDAT;        // at 100KHz, normally  250ns (min), but we borrow time from tHDDAT to improve clock phase
    LwU32 tHDDAT;        // at 100KHz, normally 3450ns (max), but we loan time to tSUDAT to improve clock phase
    LwU32 tSUSTO;        // at 100KHz, normally 4000ns
    LwU32 tHDSTA;        // at 100KHz, normally 4000ns
    LwU32 tBUF;          // at 100KHz, normally 4700ns

    LwU32 tLOW;          // computed to be:  tSUDAT + tR + tHDDAT

    LwU32 speed;         // Port speed

} LWSWITCH_I2CTIMING;

#define LW_LWSWITCH_I2C_DEVICE_WRITE_ACCESS_LEVEL              2:0
#define LW_LWSWITCH_I2C_DEVICE_READ_ACCESS_LEVEL               5:3
#define LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_PUBLIC             0x00000000
#define LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_PRIVILEGED         0x00000001
#define LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_INTERNAL           0x00000002
#define LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_INACCESSIBLE       0x00000003
#define LW_LWSWITCH_I2C_DEVICE_READ_ACCESS_LEVEL_PUBLIC        LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_PUBLIC
#define LW_LWSWITCH_I2C_DEVICE_READ_ACCESS_LEVEL_PRIVILEGED    LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_PRIVILEGED
#define LW_LWSWITCH_I2C_DEVICE_READ_ACCESS_LEVEL_INTERNAL      LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_INTERNAL
#define LW_LWSWITCH_I2C_DEVICE_READ_ACCESS_LEVEL_INACCESSIBLE  LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_INACCESSIBLE
#define LW_LWSWITCH_I2C_DEVICE_WRITE_ACCESS_LEVEL_PUBLIC       LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_PUBLIC
#define LW_LWSWITCH_I2C_DEVICE_WRITE_ACCESS_LEVEL_PRIVILEGED   LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_PRIVILEGED
#define LW_LWSWITCH_I2C_DEVICE_WRITE_ACCESS_LEVEL_INTERNAL     LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_INTERNAL
#define LW_LWSIWTCH_I2C_DEVICE_WRITE_ACCESS_LEVEL_INACCESSIVLE LW_LWSWITCH_I2C_DEVICE_ACCESS_LEVEL_INACCESSIBLE

typedef struct LWSWITCH_I2C_DEVICE_DESCRIPTOR
{
    LWSWITCH_I2C_PORT_TYPE      i2cPortLogical;     //<! Logical I2C port where the device sits
    LwU32                       i2cAddress;         //<! I2C slave address
    LWSWITCH_I2C_DEVICE_TYPE    i2cDeviceType;
    LwU8                        i2cRdWrAccessMask;
} LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE;


typedef struct LWSWITCH_OBJI2C   *PLWSWITCH_OBJI2C;

#define LWSWITCH_I2C_SPEED_MODE_100KHZ  0
#define LWSWITCH_I2C_SPEED_MODE_200KHZ  1
#define LWSWITCH_I2C_SPEED_MODE_300KHZ  2
#define LWSWITCH_I2C_SPEED_MODE_400KHZ  3
#define LWSWITCH_I2C_SPEED_MODE_1000KHZ 4

typedef struct _lwswitch_tag_i2c_port
{
    // Timing for I2C cycles (allows for possibility of tweaking timing)
    LWSWITCH_I2CTIMING Timing;

    LWSWITCH_I2C_HW_CMD  hwCmd;

    LwU32 defaultSpeedMode;
} LWSWITCH_I2CPORT, *PLWSWITCH_I2CPORT;


struct LWSWITCH_OBJI2C
{
    //
    // Addresses of I2C ports
    //
    // Note: The index of array is logical port number NOT physical
    //
    LWSWITCH_I2CPORT Ports[LWSWITCH_MAX_I2C_PORTS];

    //
    // Private data
    //

    // I2C Mutex/Synchronization state
    LwU32         I2CAcquired;

    LwU32 PortInfo[LWSWITCH_MAX_I2C_PORTS];
#define LW_I2C_PORTINFO_DEFINED                             0:0
#define LW_I2C_PORTINFO_DEFINED_ABSENT               0x00000000
#define LW_I2C_PORTINFO_DEFINED_PRESENT              0x00000001
#define LW_I2C_PORTINFO_ACCESS_ALLOWED                      1:1
#define LW_I2C_PORTINFO_ACCESS_ALLOWED_FALSE         0x00000000
#define LW_I2C_PORTINFO_ACCESS_ALLOWED_TRUE          0x00000001

    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE           *device_list;
    LwU32                                     device_list_size;

    // I2C device allow list
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE        *i2c_allow_list;
    LwU32                                  i2c_allow_list_size;
};

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// Board ID
//

typedef enum 
{
    LWSWITCH_BOARD_ID_UNKNOWN       = 0,
    LWSWITCH_BOARD_ID_E3600_A01,
    LWSWITCH_BOARD_ID_E3600_A02,
    LWSWITCH_BOARD_ID_VANGUARD,
    LWSWITCH_BOARD_ID_EXPLORER,
    LWSWITCH_BOARD_ID_PIONEER,
    LWSWITCH_BOARD_ID_E4700_A02,
    LWSWITCH_BOARD_ID_E4760_A00,
    LWSWITCH_BOARD_ID_E4761_A00,
    LWSWITCH_BOARD_ID_DELTA,
    LWSWITCH_BOARD_ID_WOLF,
    LWSWITCH_BOARD_ID_E4840,
    LWSWITCH_BOARD_ID_LAST,
} LWSWITCH_BOARD_ID_TYPE;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

//
// Thermal
//

#define     LWSWITCH_THERM_METHOD_UNKNOWN   0x00
#define     LWSWITCH_THERM_METHOD_I2C       0x01
#define     LWSWITCH_THERM_METHOD_MLW       0x02

typedef struct lwswitch_tdiode_info_type
{
    LwU32   method;
    struct LWSWITCH_I2C_DEVICE_DESCRIPTOR *method_i2c_info;

    LwS32   A;
    LwS32   B;
    LwS32   offset;
} LWSWITCH_TDIODE_INFO_TYPE;

void lwswitch_i2c_destroy(lwswitch_device *device);
void lwswitch_i2c_init(lwswitch_device *device);

#endif //_PMGR_LWSWITCH_H_
