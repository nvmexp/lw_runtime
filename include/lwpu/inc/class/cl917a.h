/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2010 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cl917a_h_
#define _cl917a_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW917A_LWRSOR_CHANNEL_PIO                                               (0x0000917A)

typedef volatile struct _cl917a_tag0 {
    LwV32 Reserved00[0x2];
    LwV32 Free;                                                                 // 0x00000008 - 0x0000000B
    LwV32 Reserved01[0x1D];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetLwrsorHotSpotPointsOut[2];                                         // 0x00000084 - 0x0000008B
    LwV32 Reserved02[0x3DD];
} GK104DispLwrsorControlPio;

#define LW917A_FREE                                                             (0x00000008)
#define LW917A_FREE_COUNT                                                       5:0
#define LW917A_UPDATE                                                           (0x00000080)
#define LW917A_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW917A_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW917A_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW917A_SET_LWRSOR_HOT_SPOT_POINTS_OUT(b)                                (0x00000084 + (b)*0x00000004)
#define LW917A_SET_LWRSOR_HOT_SPOT_POINTS_OUT_X                                 15:0
#define LW917A_SET_LWRSOR_HOT_SPOT_POINTS_OUT_Y                                 31:16

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl917a_h

