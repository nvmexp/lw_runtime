/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOE_PRIV_LWSWITCH_H_
#define _SOE_PRIV_LWSWITCH_H_

#include "soe/haldefs_soe_lwswitch.h"
#include "soe/soeifcmn.h"

#include "flcn/flcnqueue_lwswitch.h"
#include "flcn/flcnable_lwswitch.h"

#define SOE_DMEM_ALIGNMENT (4)

struct SOE
{
    // needs to be the first thing in this struct so that a PSOE can be
    // re-interpreted as a PFLCNABLE and vise-versa. While it is possible
    // to remove this restriction by using (&pSoe->parent) instead of a cast,
    // 1) the reverse (getting a PSOE from a PFLCNABLE) would be diffilwlt and
    // spooky 2) that would force anybody wanting to do the colwersion
    // to know the layout of an SOE object (not a big deal, but still annoying)
    union {
        // pointer to our function table - should always be the first thing in any object (including parent)
        soe_hal *pHal;
        FLCNABLE parent;
    } base;

    // Other member variables specific to SOE go here

    /*!
     * Structure tracking all information for active and inactive SEC2 sequences.
     */
    FLCN_QMGR_SEQ_INFO      seqInfo[RM_SOE_MAX_NUM_SEQUENCES];

    /*! The event descriptor for the Thermal event handler */
    LwU32                   thermEvtDesc;
};

#endif //_SOE_PRIV_LWSWITCH_H_
