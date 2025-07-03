/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cl507d_sw_h_
#define _cl507d_sw_h_

/* This file is *not* auto-generated. */

#define LW507D_HEAD_SET_MISC_CONFIG1(a)                                         (0x00000BA4 + (a)*0x00000400)
#define LW507D_HEAD_SET_MISC_CONFIG1_VPLL_REF                                   1:0
#define LW507D_HEAD_SET_MISC_CONFIG1_VPLL_REF_NO_PREF                           (0x00000000)
#define LW507D_HEAD_SET_MISC_CONFIG1_VPLL_REF_GSYNC                             (0x00000001)
// We'll keep *_DISPLAYMASK method until clients are completely shifted to *_DISPLAY_ID method
#define LW507D_HEAD_SET_DISPLAYMASK(a)                                          (0x00000BA8 + (a)*0x00000400)
#define LW507D_HEAD_SET_DISPLAY_ID(a,b)                                         (0x00000BA8 + (a)*0x00000400 + (b)*0x00000004)

#endif // _cl507d_sw_h

