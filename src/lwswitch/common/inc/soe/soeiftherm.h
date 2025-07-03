/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEIFTHERM_H_
#define _SOEIFTHERM_H_

#include "lwfixedtypes.h"

/*!
 * @file   soeiftherm.h
 * @brief  SOE Thermal Command Queue
 *          
 *         The Therm unit ID will be used for sending and recieving
 *         Command Messages between driver and Thermal unt of SOE
 */

/* ------------------------ Defines ---------------------------------*/

// Macros for FXP9.5 colwersion

#define LW_TSENSE_FXP_9_5_INTEGER            13:4
#define LW_TSENSE_FXP_9_5_FRACTIONAL         4:0

// Colwert 32 bit Signed integer or Floating Point value to FXP9.5
#define LW_TSENSE_COLWERT_TO_FXP_9_5(val)  \
       (LwU32) (val *(1 << DRF_SIZE(LW_TSENSE_FXP_9_5_FRACTIONAL)))

// Colwert FXP 9.5 to Celsius (Works only for temperatures >= 0)
#define LW_TSENSE_FXP_9_5_TO_CELSIUS(fxp)    \
       (LwU32) (fxp /(1 << DRF_SIZE(LW_TSENSE_FXP_9_5_FRACTIONAL)))

// Colwert FXP 9.5 to LwTemp
#define LW_TSENSE_FXP_9_5_SIGN(fxp)  \
    DRF_VAL(_TYPES, _SFXP, _INTEGER_SIGN(9,5), fxp)

#define LW_TSENSE_FXP_9_5_TO_24_8(fxp)                  \
    (LwTemp) ((LW_TSENSE_FXP_9_5_SIGN(fxp) ==           \
             LW_TYPES_SFXP_INTEGER_SIGN_NEGATIVE ?      \
             DRF_SHIFTMASK(31:17) : 0x0) | (fxp << 3))

/*!
 * Macros for LwType <-> Celsius temperature colwersion.
 */
#define RM_SOE_CELSIUS_TO_LW_TEMP(cel)                                      \
                                LW_TYPES_S32_TO_SFXP_X_Y(24,8,(cel))
#define RM_SOE_LW_TEMP_TO_CELSIUS_TRUNCED(lwt)                              \
                                LW_TYPES_SFXP_X_Y_TO_S32(24,8,(lwt))
#define RM_SOE_LW_TEMP_TO_CELSIUS_ROUNDED(lwt)                              \
                                LW_TYPES_SFXP_X_Y_TO_S32_ROUNDED(24,8,(lwt))

/*!
 * Thermal Message IDs
 */
enum
{
    RM_SOE_THERM_MSG_ID_SLOWDOWN_STATUS,
    RM_SOE_THERM_MSG_ID_SHUTDOWN_STATUS,
};

/*!
 * @brief message for thermal shutdown
 */
typedef struct
{
    LwU8   msgType;
    LwTemp maxTemperature;
    LwTemp overtThreshold;

    struct
    {
        LwBool bTsense;
        LwBool bPmgr;    
    }source;
} RM_SOE_THERM_MSG_SHUTDOWN_STATUS;

/*!
 * @brief message for thermal slowdown
 */
typedef struct
{
    LwU8   msgType;
    LwBool bSlowdown;
    LwTemp maxTemperature;
    LwTemp warnThreshold;

    struct
    {
        LwBool bTsense;
        LwBool bPmgr;    
    }source;
} RM_SOE_THERM_MSG_SLOWDOWN_STATUS;

/*!
 * A simple union of all the Thermal messages.
 * Use the 'msgType' variable to determine the actual type of the message.
 */
typedef union
{
    LwU8  msgType;
    // The following structs are expected to include cmdType as the first member
    RM_SOE_THERM_MSG_SLOWDOWN_STATUS  slowdown;
    RM_SOE_THERM_MSG_SHUTDOWN_STATUS  shutdown;
}RM_SOE_THERM_MSG;

/*!
 * Thermal Command types
 */
enum
{
    RM_SOE_THERM_FORCE_SLOWDOWN,
    RM_SOE_THERM_SEND_MSG_TO_DRIVER,
};

/*!
 * @brief Force Thermal slowdown
 */
typedef struct
{
    LwU8   cmdType;
    LwBool slowdown;
    LwU32  periodUs;
} RM_SOE_THERM_CMD_FORCE_SLOWDOWN;

/*!
 * @brief Send aysncronous message about thermal events.
 */
typedef struct
{
    LwU8   cmdType;
    union
    {
        LwU8 msgType;
        RM_SOE_THERM_MSG_SLOWDOWN_STATUS  slowdown;
        RM_SOE_THERM_MSG_SHUTDOWN_STATUS  shutdown;
    } status;
} RM_SOE_THERM_CMD_SEND_ASYNC_MSG;

/*!
 * A simple union of all the therm commands. Use the 'cmdType' variable to
 * determine the actual type of the command.
 */
typedef union
{
    LwU8  cmdType;
    // The following structs are expected to include cmdType as the first member
    RM_SOE_THERM_CMD_FORCE_SLOWDOWN  slowdown;
    RM_SOE_THERM_CMD_SEND_ASYNC_MSG  msg;
}RM_SOE_THERM_CMD;

#endif  // _SOEIFTHERM_H_

