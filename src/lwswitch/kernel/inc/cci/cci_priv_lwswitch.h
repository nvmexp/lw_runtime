/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _CCI_PRIV_LWSWITCH_H_
#define _CCI_PRIV_LWSWITCH_H_

#include "lwtypes.h"
#include "cci/cci_lwswitch.h"
#include "cci/cci_pcs_lwswitch.h"

//
// CCI is the top-level management state for all cable controllers on a device.
// The management tasks related to cable controllers is encapsulated by a PCS
// or Platform Cable System, for which CCI is largely a container.
//

#define LWSWITCH_CCI_LINK_NUM_MAX      64
#define LWSWITCH_CCI_OSFP_NUM_MAX      32
#define LWSWITCH_CCI_LED_DRV_NUM_MAX    4
#define LWSWITCH_CCI_ROM_NUM_MAX        2

// Cable Controller Interface
struct CCI
{
    // Links that are supported by CCI. The value here is defined in the BIOS
    // and is a static property of the system.  See repeater bit in LWLink.
    LwU64  linkMask;

    // ===========================================================================
    // === State below this line would be tracked in the SOE if CCI is in SOE. ===
    // ===========================================================================

    // PCSs (Platform Cable Systems) supported by this device.
    // This value is defined in the BIOS and is a static property of the
    // system.
    LwU64  pcsMask;

    // A PCS can be determined to not be present despite being in the BIOS.
    // This value starts as the pcsMask and PCS instances are removed if
    // they are not detected by PCS support functions.
    LwU64  pcsPresent;

    // The set of known PCS states is allocated as a monolithic indexable array.
    // After checking that a link is supported above, it checks with each
    // present PCS to see if it handles the link in question, and if so, that
    // is used for calls related to CCI support.
    struct PCS *pPcs;

    // ================================================================
    // === State below this line has been moved and can be deleted. ===
    // ================================================================

    // Other member variables specific to CCI go here
    LwBool bDiscovered;
    LwBool bSupported;
    LwBool bInitialized;
    LwU32  boardId;
    LwU32  boardPartitionType;
    LwU32  osfpMaskAll;     // All the possible module positions
    LwU32  osfpMaskPresent; // Lwrrently present modules
    LwU32  cagesMask;       // All the possible module cage positions(set by reading FRUs)
    LwU32  modulesMask;     // Lwrrently present modules(lwrrently mirrors osfpMaskPresent)
    LwU32  numLinks;
    LwBool preTrainingFailed;
    LwBool preTrainingComplete;
    LWSWITCH_CCI_MODULE_LINK_LANE_MAP *osfp_map;
    struct LWSWITCH_I2C_DEVICE_DESCRIPTOR *osfp_i2c_info;
    struct LWSWITCH_I2C_DEVICE_DESCRIPTOR *led_drv_i2c_info[LWSWITCH_CCI_LED_DRV_NUM_MAX];
    struct LWSWITCH_I2C_DEVICE_DESCRIPTOR *rom_i2c_info[LWSWITCH_CCI_ROM_NUM_MAX];
    LwU8   *romCache[LWSWITCH_CCI_ROM_NUM_MAX];
    LwU32  osfp_map_size;
    LwU32  osfp_num;
    LwU32  led_drv_num;
    LwU32  rom_num;

    struct {
        void (*functionPtr)(struct lwswitch_device*);
        LwU32 interval;
    } callbackList[LWSWITCH_CCI_CALLBACK_NUM_MAX];
    LwU32  callbackCounter;
    LwU8   xcvrLwrrentLedState[LWSWITCH_CCI_OSFP_NUM_MAX];
    LwU8   xcvrNextLedState[LWSWITCH_CCI_OSFP_NUM_MAX];
    LwU64  tpCounterPreviousSum[LWSWITCH_CCI_LINK_NUM_MAX];
};

#endif //_CCI_PRIV_LWSWITCH_H_
