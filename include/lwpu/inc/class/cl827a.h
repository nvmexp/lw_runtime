// WARNING!!! THIS HEADER INCLUDES SOFTWARE METHODS!!!
// ********** DO NOT USE IN HW TREE.  ********** 
/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2004 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cl827a_h_
#define _cl827a_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW827A_LWRSOR_CHANNEL_PIO                                               (0x0000827A)

typedef volatile struct _cl827a_tag0 {
    LwV32 Reserved00[0x2];
    LwV32 Free;                                                                 // 0x00000008 - 0x0000000B
    LwV32 Reserved01[0x1D];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetLwrsorHotSpotPointOut;                                             // 0x00000084 - 0x00000087
    LwV32 Reserved02[0x3DE];
} G82DispLwrsorControlPio;

#define LW827A_FREE                                                             (0x00000008)
#define LW827A_FREE_COUNT                                                       5:0
#define LW827A_UPDATE                                                           (0x00000080)
#define LW827A_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW827A_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW827A_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW827A_SET_LWRSOR_HOT_SPOT_POINT_OUT                                    (0x00000084)
#define LW827A_SET_LWRSOR_HOT_SPOT_POINT_OUT_X                                  15:0
#define LW827A_SET_LWRSOR_HOT_SPOT_POINT_OUT_Y                                  31:16

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl827a_h

