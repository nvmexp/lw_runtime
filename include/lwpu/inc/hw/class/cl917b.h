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


#ifndef _cl917b_h_
#define _cl917b_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW917B_OVERLAY_IMM_CHANNEL_PIO                                          (0x0000917B)

typedef volatile struct _cl917b_tag0 {
    LwV32 Reserved00[0x2];
    LwV32 Free;                                                                 // 0x00000008 - 0x0000000B
    LwV32 Reserved01[0x1D];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetPointsOut[2];                                                      // 0x00000084 - 0x0000008B
    LwV32 Reserved02[0x1];
    LwV32 AwakenOnceFlippedTo;                                                  // 0x00000090 - 0x00000093
    LwV32 Reserved03[0x3DB];
} GK104DispOverlayImmControlPio;

#define LW917B_FREE                                                             (0x00000008)
#define LW917B_FREE_COUNT                                                       5:0
#define LW917B_UPDATE                                                           (0x00000080)
#define LW917B_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW917B_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW917B_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW917B_SET_POINTS_OUT(b)                                                (0x00000084 + (b)*0x00000004)
#define LW917B_SET_POINTS_OUT_X                                                 15:0
#define LW917B_SET_POINTS_OUT_Y                                                 31:16
#define LW917B_AWAKEN_ONCE_FLIPPED_TO                                           (0x00000090)
#define LW917B_AWAKEN_ONCE_FLIPPED_TO_AWAKEN_COUNT                              11:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl917b_h

