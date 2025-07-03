
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbgk208.c
//
//*****************************************************

//
// includes
//
#include "hal.h"
#include "kepler/gk208/hwproject.h"

/*!
 * @brief Gets the LTS per LTC count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The LTS per LTC count.
 */
LwU32
fbGetLTSPerLTCCountLwW_GK208( void )
{
    return LW_SCAL_LITTER_NUM_LTC_SLICES;
}
