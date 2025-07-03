/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _CCI_PCS_BSP_LWSWITCH_H_
#define _CCI_PCS_BSP_LWSWITCH_H_

#include "lwtypes.h"
#include "cci/cci_pcs_bsp_e476x_lwswitch.h"
#include "cci/cci_pcs_bsp_p479x_lwswitch.h"

//
// BSPs are designed around the notion of a module. Modules may or may not be
// an actual physcial module, for example, an OSFP module.
// Modules do have a property of being present or not present, so unplugged.
// Communication with devices on the PCS are all done through the BSP hals
// and there is no understanding of how that happens in this file.
// 

// CCI PCS Board Support Package state
struct CCI_PCS_BSP
{
    // A link can be mapped to any of the modules in the PCS
    LwU8   link2moduleMap[64];

    // The lane start base within a module that a link is assigned to
    LwU8   link2moduleLaneBaseMap[64];

    // In theory, there can be up to 64 modules (1 module per link).
    LwU64  moduleSupportedMask;

    // Modules can be unplugged or simply not present, so we need to track 
    LwU64  modulePresentMask;

    // The Board Support Package type used for this PCS, defined in the BIOS.
    LwU32  bspType;

    // BSPs have different hals or support functions, which is what makes
    // up the functionality of the PCS, and it uses the BSP state
    // union below.
//    bsp_hal *pHal;

    // Union of all of the BSP state types.
    union 
    {
        struct CCI_PCS_BSP_E476X e476x;
        struct CCI_PCS_BSP_P479X p479x;
    } bsp_state;
};

#endif //_CCI_PCS_BSP_LWSWITCH_H_
