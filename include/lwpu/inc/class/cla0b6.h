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


#include "lwtypes.h"

#ifndef _cla0b6_h_
#define _cla0b6_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWA0B6_VIDEO_COMPOSITOR                                             (0x0000A0B6)

typedef volatile struct _cla0b6_tag0 {
    LwV32 Reserved00[0x40];
    LwV32 Nop;                                                                  // 0x00000100 - 0x00000103
    LwV32 Reserved01[0xF];
    LwV32 PmTrigger;                                                            // 0x00000140 - 0x00000143
    LwV32 Reserved02[0x2F];
    LwV32 SetApplicationID;                                                     // 0x00000200 - 0x00000203
    LwV32 SetWatchdogTimer;                                                     // 0x00000204 - 0x00000207
    LwV32 Reserved03[0xE];
    LwV32 SemaphoreA;                                                           // 0x00000240 - 0x00000243
    LwV32 SemaphoreB;                                                           // 0x00000244 - 0x00000247
    LwV32 SemaphoreC;                                                           // 0x00000248 - 0x0000024B
    LwV32 CtxSaveArea;                                                          // 0x0000024C - 0x0000024F
    LwV32 CtxSwitch;                                                            // 0x00000250 - 0x00000253
    LwV32 Reserved04[0x2B];
    LwV32 Execute;                                                              // 0x00000300 - 0x00000303
    LwV32 SemaphoreD;                                                           // 0x00000304 - 0x00000307
    LwV32 Reserved05[0x3E];
    LwV32 SetSurface0Slot0LumaOffset;                                           // 0x00000400 - 0x00000403
    LwV32 SetSurface0Slot0ChromaU_Offset;                                       // 0x00000404 - 0x00000407
    LwV32 SetSurface0Slot0ChromaV_Offset;                                       // 0x00000408 - 0x0000040B
    LwV32 SetSurface1Slot0LumaOffset;                                           // 0x0000040C - 0x0000040F
    LwV32 SetSurface1Slot0ChromaU_Offset;                                       // 0x00000410 - 0x00000413
    LwV32 SetSurface1Slot0ChromaV_Offset;                                       // 0x00000414 - 0x00000417
    LwV32 SetSurface2Slot0LumaOffset;                                           // 0x00000418 - 0x0000041B
    LwV32 SetSurface2Slot0ChromaU_Offset;                                       // 0x0000041C - 0x0000041F
    LwV32 SetSurface2Slot0ChromaV_Offset;                                       // 0x00000420 - 0x00000423
    LwV32 SetSurface3Slot0LumaOffset;                                           // 0x00000424 - 0x00000427
    LwV32 SetSurface3Slot0ChromaU_Offset;                                       // 0x00000428 - 0x0000042B
    LwV32 SetSurface3Slot0ChromaV_Offset;                                       // 0x0000042C - 0x0000042F
    LwV32 SetSurface4Slot0LumaOffset;                                           // 0x00000430 - 0x00000433
    LwV32 SetSurface4Slot0ChromaU_Offset;                                       // 0x00000434 - 0x00000437
    LwV32 SetSurface4Slot0ChromaV_Offset;                                       // 0x00000438 - 0x0000043B
    LwV32 SetSurface5Slot0LumaOffset;                                           // 0x0000043C - 0x0000043F
    LwV32 SetSurface5Slot0ChromaU_Offset;                                       // 0x00000440 - 0x00000443
    LwV32 SetSurface5Slot0ChromaV_Offset;                                       // 0x00000444 - 0x00000447
    LwV32 SetSurface6Slot0LumaOffset;                                           // 0x00000448 - 0x0000044B
    LwV32 SetSurface6Slot0ChromaU_Offset;                                       // 0x0000044C - 0x0000044F
    LwV32 SetSurface6Slot0ChromaV_Offset;                                       // 0x00000450 - 0x00000453
    LwV32 SetSurface7Slot0LumaOffset;                                           // 0x00000454 - 0x00000457
    LwV32 SetSurface7Slot0ChromaU_Offset;                                       // 0x00000458 - 0x0000045B
    LwV32 SetSurface7Slot0ChromaV_Offset;                                       // 0x0000045C - 0x0000045F
    LwV32 SetSurface0Slot1LumaOffset;                                           // 0x00000460 - 0x00000463
    LwV32 SetSurface0Slot1ChromaU_Offset;                                       // 0x00000464 - 0x00000467
    LwV32 SetSurface0Slot1ChromaV_Offset;                                       // 0x00000468 - 0x0000046B
    LwV32 SetSurface1Slot1LumaOffset;                                           // 0x0000046C - 0x0000046F
    LwV32 SetSurface1Slot1ChromaU_Offset;                                       // 0x00000470 - 0x00000473
    LwV32 SetSurface1Slot1ChromaV_Offset;                                       // 0x00000474 - 0x00000477
    LwV32 SetSurface2Slot1LumaOffset;                                           // 0x00000478 - 0x0000047B
    LwV32 SetSurface2Slot1ChromaU_Offset;                                       // 0x0000047C - 0x0000047F
    LwV32 SetSurface2Slot1ChromaV_Offset;                                       // 0x00000480 - 0x00000483
    LwV32 SetSurface3Slot1LumaOffset;                                           // 0x00000484 - 0x00000487
    LwV32 SetSurface3Slot1ChromaU_Offset;                                       // 0x00000488 - 0x0000048B
    LwV32 SetSurface3Slot1ChromaV_Offset;                                       // 0x0000048C - 0x0000048F
    LwV32 SetSurface4Slot1LumaOffset;                                           // 0x00000490 - 0x00000493
    LwV32 SetSurface4Slot1ChromaU_Offset;                                       // 0x00000494 - 0x00000497
    LwV32 SetSurface4Slot1ChromaV_Offset;                                       // 0x00000498 - 0x0000049B
    LwV32 SetSurface5Slot1LumaOffset;                                           // 0x0000049C - 0x0000049F
    LwV32 SetSurface5Slot1ChromaU_Offset;                                       // 0x000004A0 - 0x000004A3
    LwV32 SetSurface5Slot1ChromaV_Offset;                                       // 0x000004A4 - 0x000004A7
    LwV32 SetSurface6Slot1LumaOffset;                                           // 0x000004A8 - 0x000004AB
    LwV32 SetSurface6Slot1ChromaU_Offset;                                       // 0x000004AC - 0x000004AF
    LwV32 SetSurface6Slot1ChromaV_Offset;                                       // 0x000004B0 - 0x000004B3
    LwV32 SetSurface7Slot1LumaOffset;                                           // 0x000004B4 - 0x000004B7
    LwV32 SetSurface7Slot1ChromaU_Offset;                                       // 0x000004B8 - 0x000004BB
    LwV32 SetSurface7Slot1ChromaV_Offset;                                       // 0x000004BC - 0x000004BF
    LwV32 SetSurface0Slot2LumaOffset;                                           // 0x000004C0 - 0x000004C3
    LwV32 SetSurface0Slot2ChromaU_Offset;                                       // 0x000004C4 - 0x000004C7
    LwV32 SetSurface0Slot2ChromaV_Offset;                                       // 0x000004C8 - 0x000004CB
    LwV32 SetSurface1Slot2LumaOffset;                                           // 0x000004CC - 0x000004CF
    LwV32 SetSurface1Slot2ChromaU_Offset;                                       // 0x000004D0 - 0x000004D3
    LwV32 SetSurface1Slot2ChromaV_Offset;                                       // 0x000004D4 - 0x000004D7
    LwV32 SetSurface2Slot2LumaOffset;                                           // 0x000004D8 - 0x000004DB
    LwV32 SetSurface2Slot2ChromaU_Offset;                                       // 0x000004DC - 0x000004DF
    LwV32 SetSurface2Slot2ChromaV_Offset;                                       // 0x000004E0 - 0x000004E3
    LwV32 SetSurface3Slot2LumaOffset;                                           // 0x000004E4 - 0x000004E7
    LwV32 SetSurface3Slot2ChromaU_Offset;                                       // 0x000004E8 - 0x000004EB
    LwV32 SetSurface3Slot2ChromaV_Offset;                                       // 0x000004EC - 0x000004EF
    LwV32 SetSurface4Slot2LumaOffset;                                           // 0x000004F0 - 0x000004F3
    LwV32 SetSurface4Slot2ChromaU_Offset;                                       // 0x000004F4 - 0x000004F7
    LwV32 SetSurface4Slot2ChromaV_Offset;                                       // 0x000004F8 - 0x000004FB
    LwV32 SetSurface5Slot2LumaOffset;                                           // 0x000004FC - 0x000004FF
    LwV32 SetSurface5Slot2ChromaU_Offset;                                       // 0x00000500 - 0x00000503
    LwV32 SetSurface5Slot2ChromaV_Offset;                                       // 0x00000504 - 0x00000507
    LwV32 SetSurface6Slot2LumaOffset;                                           // 0x00000508 - 0x0000050B
    LwV32 SetSurface6Slot2ChromaU_Offset;                                       // 0x0000050C - 0x0000050F
    LwV32 SetSurface6Slot2ChromaV_Offset;                                       // 0x00000510 - 0x00000513
    LwV32 SetSurface7Slot2LumaOffset;                                           // 0x00000514 - 0x00000517
    LwV32 SetSurface7Slot2ChromaU_Offset;                                       // 0x00000518 - 0x0000051B
    LwV32 SetSurface7Slot2ChromaV_Offset;                                       // 0x0000051C - 0x0000051F
    LwV32 SetSurface0Slot3LumaOffset;                                           // 0x00000520 - 0x00000523
    LwV32 SetSurface0Slot3ChromaU_Offset;                                       // 0x00000524 - 0x00000527
    LwV32 SetSurface0Slot3ChromaV_Offset;                                       // 0x00000528 - 0x0000052B
    LwV32 SetSurface1Slot3LumaOffset;                                           // 0x0000052C - 0x0000052F
    LwV32 SetSurface1Slot3ChromaU_Offset;                                       // 0x00000530 - 0x00000533
    LwV32 SetSurface1Slot3ChromaV_Offset;                                       // 0x00000534 - 0x00000537
    LwV32 SetSurface2Slot3LumaOffset;                                           // 0x00000538 - 0x0000053B
    LwV32 SetSurface2Slot3ChromaU_Offset;                                       // 0x0000053C - 0x0000053F
    LwV32 SetSurface2Slot3ChromaV_Offset;                                       // 0x00000540 - 0x00000543
    LwV32 SetSurface3Slot3LumaOffset;                                           // 0x00000544 - 0x00000547
    LwV32 SetSurface3Slot3ChromaU_Offset;                                       // 0x00000548 - 0x0000054B
    LwV32 SetSurface3Slot3ChromaV_Offset;                                       // 0x0000054C - 0x0000054F
    LwV32 SetSurface4Slot3LumaOffset;                                           // 0x00000550 - 0x00000553
    LwV32 SetSurface4Slot3ChromaU_Offset;                                       // 0x00000554 - 0x00000557
    LwV32 SetSurface4Slot3ChromaV_Offset;                                       // 0x00000558 - 0x0000055B
    LwV32 SetSurface5Slot3LumaOffset;                                           // 0x0000055C - 0x0000055F
    LwV32 SetSurface5Slot3ChromaU_Offset;                                       // 0x00000560 - 0x00000563
    LwV32 SetSurface5Slot3ChromaV_Offset;                                       // 0x00000564 - 0x00000567
    LwV32 SetSurface6Slot3LumaOffset;                                           // 0x00000568 - 0x0000056B
    LwV32 SetSurface6Slot3ChromaU_Offset;                                       // 0x0000056C - 0x0000056F
    LwV32 SetSurface6Slot3ChromaV_Offset;                                       // 0x00000570 - 0x00000573
    LwV32 SetSurface7Slot3LumaOffset;                                           // 0x00000574 - 0x00000577
    LwV32 SetSurface7Slot3ChromaU_Offset;                                       // 0x00000578 - 0x0000057B
    LwV32 SetSurface7Slot3ChromaV_Offset;                                       // 0x0000057C - 0x0000057F
    LwV32 SetSurface0Slot4LumaOffset;                                           // 0x00000580 - 0x00000583
    LwV32 SetSurface0Slot4ChromaU_Offset;                                       // 0x00000584 - 0x00000587
    LwV32 SetSurface0Slot4ChromaV_Offset;                                       // 0x00000588 - 0x0000058B
    LwV32 SetSurface1Slot4LumaOffset;                                           // 0x0000058C - 0x0000058F
    LwV32 SetSurface1Slot4ChromaU_Offset;                                       // 0x00000590 - 0x00000593
    LwV32 SetSurface1Slot4ChromaV_Offset;                                       // 0x00000594 - 0x00000597
    LwV32 SetSurface2Slot4LumaOffset;                                           // 0x00000598 - 0x0000059B
    LwV32 SetSurface2Slot4ChromaU_Offset;                                       // 0x0000059C - 0x0000059F
    LwV32 SetSurface2Slot4ChromaV_Offset;                                       // 0x000005A0 - 0x000005A3
    LwV32 SetSurface3Slot4LumaOffset;                                           // 0x000005A4 - 0x000005A7
    LwV32 SetSurface3Slot4ChromaU_Offset;                                       // 0x000005A8 - 0x000005AB
    LwV32 SetSurface3Slot4ChromaV_Offset;                                       // 0x000005AC - 0x000005AF
    LwV32 SetSurface4Slot4LumaOffset;                                           // 0x000005B0 - 0x000005B3
    LwV32 SetSurface4Slot4ChromaU_Offset;                                       // 0x000005B4 - 0x000005B7
    LwV32 SetSurface4Slot4ChromaV_Offset;                                       // 0x000005B8 - 0x000005BB
    LwV32 SetSurface5Slot4LumaOffset;                                           // 0x000005BC - 0x000005BF
    LwV32 SetSurface5Slot4ChromaU_Offset;                                       // 0x000005C0 - 0x000005C3
    LwV32 SetSurface5Slot4ChromaV_Offset;                                       // 0x000005C4 - 0x000005C7
    LwV32 SetSurface6Slot4LumaOffset;                                           // 0x000005C8 - 0x000005CB
    LwV32 SetSurface6Slot4ChromaU_Offset;                                       // 0x000005CC - 0x000005CF
    LwV32 SetSurface6Slot4ChromaV_Offset;                                       // 0x000005D0 - 0x000005D3
    LwV32 SetSurface7Slot4LumaOffset;                                           // 0x000005D4 - 0x000005D7
    LwV32 SetSurface7Slot4ChromaU_Offset;                                       // 0x000005D8 - 0x000005DB
    LwV32 SetSurface7Slot4ChromaV_Offset;                                       // 0x000005DC - 0x000005DF
    LwV32 Reserved06[0x48];
    LwV32 SetControlParams;                                                     // 0x00000700 - 0x00000703
    LwV32 SetContextId;                                                         // 0x00000704 - 0x00000707
    LwV32 SetContextIdForSlot0;                                                 // 0x00000708 - 0x0000070B
    LwV32 SetContextIdForSlot1;                                                 // 0x0000070C - 0x0000070F
    LwV32 SetContextIdForSlot2;                                                 // 0x00000710 - 0x00000713
    LwV32 SetContextIdForSlot3;                                                 // 0x00000714 - 0x00000717
    LwV32 SetContextIdForSlot4;                                                 // 0x00000718 - 0x0000071B
    LwV32 SetFceUcodeSize;                                                      // 0x0000071C - 0x0000071F
    LwV32 SetConfigStructOffset;                                                // 0x00000720 - 0x00000723
    LwV32 SetPaletteOffset;                                                     // 0x00000724 - 0x00000727
    LwV32 SetHistOffset;                                                        // 0x00000728 - 0x0000072B
    LwV32 SetFceUcodeOffset;                                                    // 0x0000072C - 0x0000072F
    LwV32 SetOutputSurfaceLumaOffset;                                           // 0x00000730 - 0x00000733
    LwV32 SetOutputSurfaceChromaU_Offset;                                       // 0x00000734 - 0x00000737
    LwV32 SetOutputSurfaceChromaV_Offset;                                       // 0x00000738 - 0x0000073B
    LwV32 SetPictureIndex;                                                      // 0x0000073C - 0x0000073F
    LwV32 Reserved07[0x275];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved08[0x3BA];
} T40VICControlPio;

#define LWA0B6_VIDEO_COMPOSITOR_NOP                                                              (0x00000100)
#define LWA0B6_VIDEO_COMPOSITOR_NOP_PARAMETER                                                    31:0
#define LWA0B6_VIDEO_COMPOSITOR_PM_TRIGGER                                                       (0x00000140)
#define LWA0B6_VIDEO_COMPOSITOR_PM_TRIGGER_V                                                     31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID                                               (0x00000200)
#define LWA0B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID                                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID_COMPOSITOR                                 (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWA0B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_A                                                      (0x00000240)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_A_UPPER                                                7:0
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_B                                                      (0x00000244)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_B_LOWER                                                31:0
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_C                                                      (0x00000248)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA                                                    (0x0000024C)
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA_OFFSET                                             27:0
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA_CTX_VALID                                          31:28
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH                                                       (0x00000250)
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE                                               0:0
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE_FALSE                                         (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE_TRUE                                          (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY                                            1:1
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY_FALSE                                      (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY_TRUE                                       (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESERVED                                              7:2
#define LWA0B6_VIDEO_COMPOSITOR_CTX_SWITCH_ASID                                                  23:8
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE                                                          (0x00000300)
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY                                                   0:0
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON                                                1:1
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN                                                   8:8
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D                                                      (0x00000304)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION                                            17:16
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWA0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_LUMA_OFFSET                                   (0x00000400)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_U_OFFSET                               (0x00000404)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_V_OFFSET                               (0x00000408)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_LUMA_OFFSET                                   (0x0000040C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_U_OFFSET                               (0x00000410)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_V_OFFSET                               (0x00000414)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_LUMA_OFFSET                                   (0x00000418)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_U_OFFSET                               (0x0000041C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_V_OFFSET                               (0x00000420)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_LUMA_OFFSET                                   (0x00000424)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_U_OFFSET                               (0x00000428)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_V_OFFSET                               (0x0000042C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_LUMA_OFFSET                                   (0x00000430)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_U_OFFSET                               (0x00000434)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_V_OFFSET                               (0x00000438)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_LUMA_OFFSET                                   (0x0000043C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_U_OFFSET                               (0x00000440)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_V_OFFSET                               (0x00000444)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_LUMA_OFFSET                                   (0x00000448)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_U_OFFSET                               (0x0000044C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_V_OFFSET                               (0x00000450)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_LUMA_OFFSET                                   (0x00000454)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_U_OFFSET                               (0x00000458)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_V_OFFSET                               (0x0000045C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_LUMA_OFFSET                                   (0x00000460)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_U_OFFSET                               (0x00000464)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_V_OFFSET                               (0x00000468)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_LUMA_OFFSET                                   (0x0000046C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_U_OFFSET                               (0x00000470)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_V_OFFSET                               (0x00000474)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_LUMA_OFFSET                                   (0x00000478)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_U_OFFSET                               (0x0000047C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_V_OFFSET                               (0x00000480)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_LUMA_OFFSET                                   (0x00000484)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_U_OFFSET                               (0x00000488)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_V_OFFSET                               (0x0000048C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_LUMA_OFFSET                                   (0x00000490)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_U_OFFSET                               (0x00000494)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_V_OFFSET                               (0x00000498)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_LUMA_OFFSET                                   (0x0000049C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_U_OFFSET                               (0x000004A0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_V_OFFSET                               (0x000004A4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_LUMA_OFFSET                                   (0x000004A8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_U_OFFSET                               (0x000004AC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_V_OFFSET                               (0x000004B0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_LUMA_OFFSET                                   (0x000004B4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_U_OFFSET                               (0x000004B8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_V_OFFSET                               (0x000004BC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_LUMA_OFFSET                                   (0x000004C0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_U_OFFSET                               (0x000004C4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_V_OFFSET                               (0x000004C8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_LUMA_OFFSET                                   (0x000004CC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_U_OFFSET                               (0x000004D0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_V_OFFSET                               (0x000004D4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_LUMA_OFFSET                                   (0x000004D8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_U_OFFSET                               (0x000004DC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_V_OFFSET                               (0x000004E0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_LUMA_OFFSET                                   (0x000004E4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_U_OFFSET                               (0x000004E8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_V_OFFSET                               (0x000004EC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_LUMA_OFFSET                                   (0x000004F0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_U_OFFSET                               (0x000004F4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_V_OFFSET                               (0x000004F8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_LUMA_OFFSET                                   (0x000004FC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_U_OFFSET                               (0x00000500)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_V_OFFSET                               (0x00000504)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_LUMA_OFFSET                                   (0x00000508)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_U_OFFSET                               (0x0000050C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_V_OFFSET                               (0x00000510)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_LUMA_OFFSET                                   (0x00000514)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_U_OFFSET                               (0x00000518)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_V_OFFSET                               (0x0000051C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_LUMA_OFFSET                                   (0x00000520)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_U_OFFSET                               (0x00000524)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_V_OFFSET                               (0x00000528)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_LUMA_OFFSET                                   (0x0000052C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_U_OFFSET                               (0x00000530)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_V_OFFSET                               (0x00000534)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_LUMA_OFFSET                                   (0x00000538)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_U_OFFSET                               (0x0000053C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_V_OFFSET                               (0x00000540)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_LUMA_OFFSET                                   (0x00000544)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_U_OFFSET                               (0x00000548)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_V_OFFSET                               (0x0000054C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_LUMA_OFFSET                                   (0x00000550)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_U_OFFSET                               (0x00000554)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_V_OFFSET                               (0x00000558)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_LUMA_OFFSET                                   (0x0000055C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_U_OFFSET                               (0x00000560)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_V_OFFSET                               (0x00000564)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_LUMA_OFFSET                                   (0x00000568)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_U_OFFSET                               (0x0000056C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_V_OFFSET                               (0x00000570)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_LUMA_OFFSET                                   (0x00000574)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_U_OFFSET                               (0x00000578)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_V_OFFSET                               (0x0000057C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_LUMA_OFFSET                                   (0x00000580)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_U_OFFSET                               (0x00000584)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_V_OFFSET                               (0x00000588)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_LUMA_OFFSET                                   (0x0000058C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_U_OFFSET                               (0x00000590)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_V_OFFSET                               (0x00000594)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_LUMA_OFFSET                                   (0x00000598)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_U_OFFSET                               (0x0000059C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_V_OFFSET                               (0x000005A0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_LUMA_OFFSET                                   (0x000005A4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_U_OFFSET                               (0x000005A8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_V_OFFSET                               (0x000005AC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_LUMA_OFFSET                                   (0x000005B0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_U_OFFSET                               (0x000005B4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_V_OFFSET                               (0x000005B8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_LUMA_OFFSET                                   (0x000005BC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_U_OFFSET                               (0x000005C0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_V_OFFSET                               (0x000005C4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_LUMA_OFFSET                                   (0x000005C8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_U_OFFSET                               (0x000005CC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_V_OFFSET                               (0x000005D0)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_LUMA_OFFSET                                   (0x000005D4)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_U_OFFSET                               (0x000005D8)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_V_OFFSET                               (0x000005DC)
#define LWA0B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS                                               (0x00000700)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_GPTIMER_ON                                    0:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_DEBUG_MODE                                    4:4
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_FALCON_CONTROL                                8:8
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_CONFIG_STRUCT_SIZE                            31:16
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID                                                   (0x00000704)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FCE_UCODE                                         3:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_CONFIG                                            7:4
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_PALETTE                                           11:8
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_OUTPUT                                            15:12
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_HIST                                              19:16
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0                                         (0x00000708)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC0                             3:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC1                             7:4
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC2                             11:8
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC3                             15:12
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC4                             19:16
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC5                             23:20
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC6                             27:24
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC7                             31:28
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1                                         (0x0000070C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC0                             3:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC1                             7:4
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC2                             11:8
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC3                             15:12
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC4                             19:16
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC5                             23:20
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC6                             27:24
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC7                             31:28
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2                                         (0x00000710)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC0                             3:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC1                             7:4
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC2                             11:8
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC3                             15:12
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC4                             19:16
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC5                             23:20
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC6                             27:24
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC7                             31:28
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3                                         (0x00000714)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC0                             3:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC1                             7:4
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC2                             11:8
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC3                             15:12
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC4                             19:16
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC5                             23:20
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC6                             27:24
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC7                             31:28
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4                                         (0x00000718)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC0                             3:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC1                             7:4
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC2                             11:8
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC3                             15:12
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC4                             19:16
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC5                             23:20
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC6                             27:24
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC7                             31:28
#define LWA0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE                                               (0x0000071C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE_FCE_SZ                                        15:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET                                         (0x00000720)
#define LWA0B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET_OFFSET                                  31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET                                               (0x00000724)
#define LWA0B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET_OFFSET                                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET                                                  (0x00000728)
#define LWA0B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET_OFFSET                                           31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET                                             (0x0000072C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET_OFFSET                                      31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET                                   (0x00000730)
#define LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET_OFFSET                            31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_U_OFFSET                               (0x00000734)
#define LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_V_OFFSET                               (0x00000738)
#define LWA0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWA0B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX                                                (0x0000073C)
#define LWA0B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWA0B6_VIDEO_COMPOSITOR_PM_TRIGGER_END                                                   (0x00001114)
#define LWA0B6_VIDEO_COMPOSITOR_PM_TRIGGER_END_V                                                 31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cla0b6_h

