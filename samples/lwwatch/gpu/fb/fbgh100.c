/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbgh100.c
//
//*****************************************************

//
// includes
//
#include "fb.h"
#include "g_fb_private.h"
#include "hopper/gh100/pri_lw_xal_ep.h"

LwU32 fbGetBAR0WindowRegAddress_GH100( void )
{
     return LW_XAL_EP_BAR0_WINDOW;
}

void fbSetBAR0WindowBase_GH100(LwU32 baseOffset)
{
    GPU_REG_WR32(LW_XAL_EP_BAR0_WINDOW,
                 DRF_NUM(_XAL_EP, _BAR0_WINDOW, _BASE, baseOffset));
}
