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


#ifndef _cl507b_h_
#define _cl507b_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW507B_OVERLAY_IMM_CHANNEL_PIO                                          (0x0000507B)

typedef volatile struct _cl507b_tag0 {
    LwV32 Reserved00[0x2];
    LwV32 Free;                                                                 // 0x00000008 - 0x0000000B
    LwV32 Reserved01[0x1D];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetPointOut;                                                          // 0x00000084 - 0x00000087
    LwV32 AwakenOnceFlippedTo;                                                  // 0x00000088 - 0x0000008B
    LwV32 Reserved02[0x3DD];
} Lw50DispOverlayImmControlPio;

#define LW507B_FREE                                                             (0x00000008)
#define LW507B_FREE_COUNT                                                       5:0
#define LW507B_UPDATE                                                           (0x00000080)
#define LW507B_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW507B_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW507B_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW507B_SET_POINT_OUT                                                    (0x00000084)
#define LW507B_SET_POINT_OUT_X                                                  15:0
#define LW507B_SET_POINT_OUT_Y                                                  31:16
#define LW507B_AWAKEN_ONCE_FLIPPED_TO                                           (0x00000088)
#define LW507B_AWAKEN_ONCE_FLIPPED_TO_AWAKEN_COUNT                              11:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl507b_h

