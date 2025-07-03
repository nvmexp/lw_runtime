/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _CCI_PCS_LWSWITCH_H_
#define _CCI_PCS_LWSWITCH_H_

#include "lwtypes.h"
#include "cci/cci_pcs_bsp_lwswitch.h"

//
// PCS tracks a particular system or subsystem that contains the optical
// support implementation.  There may be multiple PCS's per CCI state.
// Individual PCS's are called out in the PCS BIOS tables.
//

// CCI Platform Cable System state
struct CCI_PCS
{
    // The links handled by this PCS as defined in the bios.
    LwU64  assignedLinksMask;

    // The links that are actually present and usable on this PCS.
    LwU64  presentLinksMask;

    struct CCI_PCS_BSP bsp;
};

#endif //_CCI_PCS_LWSWITCH_H_
