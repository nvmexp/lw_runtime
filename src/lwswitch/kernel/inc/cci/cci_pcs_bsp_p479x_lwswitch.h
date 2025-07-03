/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _CCI_PCS_BSP_P479X_LWSWITCH_H_
#define _CCI_PCS_BSP_P479X_LWSWITCH_H_

#include "lwtypes.h"

// P479X BSP state
struct CCI_PCS_BSP_P479X
{
    // The I2C port used to communicate with the PCS
    struct LWSWITCH_I2C_DEVICE_DESCRIPTOR *pPcs_i2c_info;

    // The GPIO struct index describing the GPIO details used for 
    // global PCS reset
    LwU8   pcsResetGpioDcbIndex;
    
    // The GPIO struct index describing the GPIO details used for
    // global PCS interrupts
    LwU8   pcsInterruptGpioDcbIndex;

    // There is a FRU EEPROM this board (ATC24C02)
    LwU8   fruEepromI2CAddress;

    // There are 4 OSFP modules on this board (CMIS)
    LwU8   cmisModuleI2CAddress[4];

    // There are two LED drivers on this board (TCA6507)
    LwU8   ledDriverI2CAddress[2];

    // There is a GPIO expander the contains the interrupt and presence state
    // of all modules cages on the PCS board (PCAL9538).
    LwU8   moduleStateGpioExpanderI2CAddress;

    // There is a GPIO expander that controls the reset of all devices
    // on the PCS board (PCAL9538).
    LwU8   deviceResetGpioExpanderI2CAddress;
};

#endif //_CCI_PCS_BSP_P479X_LWSWITCH_H_
