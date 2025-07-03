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

#ifndef _cl86b6_h_
#define _cl86b6_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW86B6_VIDEO_COMPOSITOR                                             (0x000086B6)

typedef volatile struct _cl86b6_tag0 {
    LwV32 Reserved00[0x40];
    LwV32 Nop;                                                                  // 0x00000100 - 0x00000103
    LwV32 Reserved01[0xF];
    LwV32 PmTrigger;                                                            // 0x00000140 - 0x00000143
    LwV32 Reserved02[0xF];
    LwV32 SetCtxDma[7];                                                         // 0x00000180 - 0x0000019B
    LwV32 Reserved03[0x19];
    LwV32 SetApplicationID;                                                     // 0x00000200 - 0x00000203
    LwV32 SetWatchdogTimer;                                                     // 0x00000204 - 0x00000207
    LwV32 Reserved04[0xE];
    LwV32 SetSemaphoreA;                                                        // 0x00000240 - 0x00000243
    LwV32 SetSemaphoreB;                                                        // 0x00000244 - 0x00000247
    LwV32 SetSemaphorePayload;                                                  // 0x00000248 - 0x0000024B
    LwV32 Reserved05[0x2D];
    LwV32 Execute;                                                              // 0x00000300 - 0x00000303
    LwV32 BackendSemaphore;                                                     // 0x00000304 - 0x00000307
    LwV32 Reserved06[0x3E];
    LwV32 SetSurface0Slot0LumaOffset;                                           // 0x00000400 - 0x00000403
    LwV32 SetSurface0Slot0ChromaOffset;                                         // 0x00000404 - 0x00000407
    LwV32 SetSurface1Slot0LumaOffset;                                           // 0x00000408 - 0x0000040B
    LwV32 SetSurface1Slot0ChromaOffset;                                         // 0x0000040C - 0x0000040F
    LwV32 SetSurface2Slot0LumaOffset;                                           // 0x00000410 - 0x00000413
    LwV32 SetSurface2Slot0ChromaOffset;                                         // 0x00000414 - 0x00000417
    LwV32 SetSurface3Slot0LumaOffset;                                           // 0x00000418 - 0x0000041B
    LwV32 SetSurface3Slot0ChromaOffset;                                         // 0x0000041C - 0x0000041F
    LwV32 SetSurface4Slot0LumaOffset;                                           // 0x00000420 - 0x00000423
    LwV32 SetSurface4Slot0ChromaOffset;                                         // 0x00000424 - 0x00000427
    LwV32 SetSurface5Slot0LumaOffset;                                           // 0x00000428 - 0x0000042B
    LwV32 SetSurface5Slot0ChromaOffset;                                         // 0x0000042C - 0x0000042F
    LwV32 SetSurface6Slot0LumaOffset;                                           // 0x00000430 - 0x00000433
    LwV32 SetSurface6Slot0ChromaOffset;                                         // 0x00000434 - 0x00000437
    LwV32 SetSurface7Slot0LumaOffset;                                           // 0x00000438 - 0x0000043B
    LwV32 SetSurface7Slot0ChromaOffset;                                         // 0x0000043C - 0x0000043F
    LwV32 SetSurface0Slot1LumaOffset;                                           // 0x00000440 - 0x00000443
    LwV32 SetSurface0Slot1ChromaOffset;                                         // 0x00000444 - 0x00000447
    LwV32 SetSurface1Slot1LumaOffset;                                           // 0x00000448 - 0x0000044B
    LwV32 SetSurface1Slot1ChromaOffset;                                         // 0x0000044C - 0x0000044F
    LwV32 SetSurface2Slot1LumaOffset;                                           // 0x00000450 - 0x00000453
    LwV32 SetSurface2Slot1ChromaOffset;                                         // 0x00000454 - 0x00000457
    LwV32 SetSurface3Slot1LumaOffset;                                           // 0x00000458 - 0x0000045B
    LwV32 SetSurface3Slot1ChromaOffset;                                         // 0x0000045C - 0x0000045F
    LwV32 SetSurface4Slot1LumaOffset;                                           // 0x00000460 - 0x00000463
    LwV32 SetSurface4Slot1ChromaOffset;                                         // 0x00000464 - 0x00000467
    LwV32 SetSurface5Slot1LumaOffset;                                           // 0x00000468 - 0x0000046B
    LwV32 SetSurface5Slot1ChromaOffset;                                         // 0x0000046C - 0x0000046F
    LwV32 SetSurface6Slot1LumaOffset;                                           // 0x00000470 - 0x00000473
    LwV32 SetSurface6Slot1ChromaOffset;                                         // 0x00000474 - 0x00000477
    LwV32 SetSurface7Slot1LumaOffset;                                           // 0x00000478 - 0x0000047B
    LwV32 SetSurface7Slot1ChromaOffset;                                         // 0x0000047C - 0x0000047F
    LwV32 SetSurface0Slot2LumaOffset;                                           // 0x00000480 - 0x00000483
    LwV32 SetSurface0Slot2ChromaOffset;                                         // 0x00000484 - 0x00000487
    LwV32 SetSurface1Slot2LumaOffset;                                           // 0x00000488 - 0x0000048B
    LwV32 SetSurface1Slot2ChromaOffset;                                         // 0x0000048C - 0x0000048F
    LwV32 SetSurface2Slot2LumaOffset;                                           // 0x00000490 - 0x00000493
    LwV32 SetSurface2Slot2ChromaOffset;                                         // 0x00000494 - 0x00000497
    LwV32 SetSurface3Slot2LumaOffset;                                           // 0x00000498 - 0x0000049B
    LwV32 SetSurface3Slot2ChromaOffset;                                         // 0x0000049C - 0x0000049F
    LwV32 SetSurface4Slot2LumaOffset;                                           // 0x000004A0 - 0x000004A3
    LwV32 SetSurface4Slot2ChromaOffset;                                         // 0x000004A4 - 0x000004A7
    LwV32 SetSurface5Slot2LumaOffset;                                           // 0x000004A8 - 0x000004AB
    LwV32 SetSurface5Slot2ChromaOffset;                                         // 0x000004AC - 0x000004AF
    LwV32 SetSurface6Slot2LumaOffset;                                           // 0x000004B0 - 0x000004B3
    LwV32 SetSurface6Slot2ChromaOffset;                                         // 0x000004B4 - 0x000004B7
    LwV32 SetSurface7Slot2LumaOffset;                                           // 0x000004B8 - 0x000004BB
    LwV32 SetSurface7Slot2ChromaOffset;                                         // 0x000004BC - 0x000004BF
    LwV32 SetSurface0Slot3LumaOffset;                                           // 0x000004C0 - 0x000004C3
    LwV32 SetSurface0Slot3ChromaOffset;                                         // 0x000004C4 - 0x000004C7
    LwV32 SetSurface1Slot3LumaOffset;                                           // 0x000004C8 - 0x000004CB
    LwV32 SetSurface1Slot3ChromaOffset;                                         // 0x000004CC - 0x000004CF
    LwV32 SetSurface2Slot3LumaOffset;                                           // 0x000004D0 - 0x000004D3
    LwV32 SetSurface2Slot3ChromaOffset;                                         // 0x000004D4 - 0x000004D7
    LwV32 SetSurface3Slot3LumaOffset;                                           // 0x000004D8 - 0x000004DB
    LwV32 SetSurface3Slot3ChromaOffset;                                         // 0x000004DC - 0x000004DF
    LwV32 SetSurface4Slot3LumaOffset;                                           // 0x000004E0 - 0x000004E3
    LwV32 SetSurface4Slot3ChromaOffset;                                         // 0x000004E4 - 0x000004E7
    LwV32 SetSurface5Slot3LumaOffset;                                           // 0x000004E8 - 0x000004EB
    LwV32 SetSurface5Slot3ChromaOffset;                                         // 0x000004EC - 0x000004EF
    LwV32 SetSurface6Slot3LumaOffset;                                           // 0x000004F0 - 0x000004F3
    LwV32 SetSurface6Slot3ChromaOffset;                                         // 0x000004F4 - 0x000004F7
    LwV32 SetSurface7Slot3LumaOffset;                                           // 0x000004F8 - 0x000004FB
    LwV32 SetSurface7Slot3ChromaOffset;                                         // 0x000004FC - 0x000004FF
    LwV32 SetSurface0Slot4LumaOffset;                                           // 0x00000500 - 0x00000503
    LwV32 SetSurface0Slot4ChromaOffset;                                         // 0x00000504 - 0x00000507
    LwV32 SetSurface1Slot4LumaOffset;                                           // 0x00000508 - 0x0000050B
    LwV32 SetSurface1Slot4ChromaOffset;                                         // 0x0000050C - 0x0000050F
    LwV32 SetSurface2Slot4LumaOffset;                                           // 0x00000510 - 0x00000513
    LwV32 SetSurface2Slot4ChromaOffset;                                         // 0x00000514 - 0x00000517
    LwV32 SetSurface3Slot4LumaOffset;                                           // 0x00000518 - 0x0000051B
    LwV32 SetSurface3Slot4ChromaOffset;                                         // 0x0000051C - 0x0000051F
    LwV32 SetSurface4Slot4LumaOffset;                                           // 0x00000520 - 0x00000523
    LwV32 SetSurface4Slot4ChromaOffset;                                         // 0x00000524 - 0x00000527
    LwV32 SetSurface5Slot4LumaOffset;                                           // 0x00000528 - 0x0000052B
    LwV32 SetSurface5Slot4ChromaOffset;                                         // 0x0000052C - 0x0000052F
    LwV32 SetSurface6Slot4LumaOffset;                                           // 0x00000530 - 0x00000533
    LwV32 SetSurface6Slot4ChromaOffset;                                         // 0x00000534 - 0x00000537
    LwV32 SetSurface7Slot4LumaOffset;                                           // 0x00000538 - 0x0000053B
    LwV32 SetSurface7Slot4ChromaOffset;                                         // 0x0000053C - 0x0000053F
    LwV32 Reserved07[0x70];
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
    LwV32 SetOutputSurfaceChromaOffset;                                         // 0x00000734 - 0x00000737
    LwV32 Reserved08[0x1];
    LwV32 SetPictureIndex;                                                      // 0x0000073C - 0x0000073F
    LwV32 Reserved09[0x275];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved10[0x3BA];
} IGT21AVICControlPio;

#define LW86B6_VIDEO_COMPOSITOR_NOP                                                              (0x00000100)
#define LW86B6_VIDEO_COMPOSITOR_NOP_PARAMETER                                                    31:0
#define LW86B6_VIDEO_COMPOSITOR_PM_TRIGGER                                                       (0x00000140)
#define LW86B6_VIDEO_COMPOSITOR_PM_TRIGGER_V                                                     31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CTX_DMA(b)                                                   (0x00000180 + (b)*0x00000004)
#define LW86B6_VIDEO_COMPOSITOR_SET_CTX_DMA_HANDLE                                               31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID                                               (0x00000200)
#define LW86B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID                                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID_COMPOSITOR                                 (0x00000000)
#define LW86B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW86B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SEMAPHORE_A                                                  (0x00000240)
#define LW86B6_VIDEO_COMPOSITOR_SET_SEMAPHORE_A_UPPER                                            7:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SEMAPHORE_A_CTX_DMA                                          31:28
#define LW86B6_VIDEO_COMPOSITOR_SET_SEMAPHORE_B                                                  (0x00000244)
#define LW86B6_VIDEO_COMPOSITOR_SET_SEMAPHORE_B_LOWER                                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SEMAPHORE_PAYLOAD                                            (0x00000248)
#define LW86B6_VIDEO_COMPOSITOR_SET_SEMAPHORE_PAYLOAD_PAYLOAD                                    31:0
#define LW86B6_VIDEO_COMPOSITOR_EXELWTE                                                          (0x00000300)
#define LW86B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY                                                   0:0
#define LW86B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LW86B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LW86B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN                                                   8:8
#define LW86B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LW86B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LW86B6_VIDEO_COMPOSITOR_BACKEND_SEMAPHORE                                                (0x00000304)
#define LW86B6_VIDEO_COMPOSITOR_BACKEND_SEMAPHORE_STRUCT_SIZE                                    0:0
#define LW86B6_VIDEO_COMPOSITOR_BACKEND_SEMAPHORE_STRUCT_SIZE_ONE                                (0x00000000)
#define LW86B6_VIDEO_COMPOSITOR_BACKEND_SEMAPHORE_STRUCT_SIZE_FOUR                               (0x00000001)
#define LW86B6_VIDEO_COMPOSITOR_BACKEND_SEMAPHORE_AWAKEN                                         8:8
#define LW86B6_VIDEO_COMPOSITOR_BACKEND_SEMAPHORE_AWAKEN_DISABLE                                 (0x00000000)
#define LW86B6_VIDEO_COMPOSITOR_BACKEND_SEMAPHORE_AWAKEN_ENABLE                                  (0x00000001)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_LUMA_OFFSET                                   (0x00000400)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_OFFSET                                 (0x00000404)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_LUMA_OFFSET                                   (0x00000408)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_OFFSET                                 (0x0000040C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_LUMA_OFFSET                                   (0x00000410)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_OFFSET                                 (0x00000414)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_LUMA_OFFSET                                   (0x00000418)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_OFFSET                                 (0x0000041C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_LUMA_OFFSET                                   (0x00000420)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_OFFSET                                 (0x00000424)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_LUMA_OFFSET                                   (0x00000428)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_OFFSET                                 (0x0000042C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_LUMA_OFFSET                                   (0x00000430)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_OFFSET                                 (0x00000434)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_LUMA_OFFSET                                   (0x00000438)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_OFFSET                                 (0x0000043C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT0_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_LUMA_OFFSET                                   (0x00000440)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_OFFSET                                 (0x00000444)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_LUMA_OFFSET                                   (0x00000448)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_OFFSET                                 (0x0000044C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_LUMA_OFFSET                                   (0x00000450)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_OFFSET                                 (0x00000454)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_LUMA_OFFSET                                   (0x00000458)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_OFFSET                                 (0x0000045C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_LUMA_OFFSET                                   (0x00000460)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_OFFSET                                 (0x00000464)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_LUMA_OFFSET                                   (0x00000468)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_OFFSET                                 (0x0000046C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_LUMA_OFFSET                                   (0x00000470)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_OFFSET                                 (0x00000474)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_LUMA_OFFSET                                   (0x00000478)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_OFFSET                                 (0x0000047C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT1_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_LUMA_OFFSET                                   (0x00000480)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_OFFSET                                 (0x00000484)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_LUMA_OFFSET                                   (0x00000488)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_OFFSET                                 (0x0000048C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_LUMA_OFFSET                                   (0x00000490)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_OFFSET                                 (0x00000494)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_LUMA_OFFSET                                   (0x00000498)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_OFFSET                                 (0x0000049C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_LUMA_OFFSET                                   (0x000004A0)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_OFFSET                                 (0x000004A4)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_LUMA_OFFSET                                   (0x000004A8)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_OFFSET                                 (0x000004AC)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_LUMA_OFFSET                                   (0x000004B0)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_OFFSET                                 (0x000004B4)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_LUMA_OFFSET                                   (0x000004B8)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_OFFSET                                 (0x000004BC)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT2_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_LUMA_OFFSET                                   (0x000004C0)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_OFFSET                                 (0x000004C4)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_LUMA_OFFSET                                   (0x000004C8)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_OFFSET                                 (0x000004CC)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_LUMA_OFFSET                                   (0x000004D0)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_OFFSET                                 (0x000004D4)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_LUMA_OFFSET                                   (0x000004D8)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_OFFSET                                 (0x000004DC)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_LUMA_OFFSET                                   (0x000004E0)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_OFFSET                                 (0x000004E4)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_LUMA_OFFSET                                   (0x000004E8)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_OFFSET                                 (0x000004EC)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_LUMA_OFFSET                                   (0x000004F0)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_OFFSET                                 (0x000004F4)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_LUMA_OFFSET                                   (0x000004F8)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_OFFSET                                 (0x000004FC)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT3_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_LUMA_OFFSET                                   (0x00000500)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_OFFSET                                 (0x00000504)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE0_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_LUMA_OFFSET                                   (0x00000508)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_OFFSET                                 (0x0000050C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE1_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_LUMA_OFFSET                                   (0x00000510)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_OFFSET                                 (0x00000514)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE2_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_LUMA_OFFSET                                   (0x00000518)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_OFFSET                                 (0x0000051C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE3_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_LUMA_OFFSET                                   (0x00000520)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_OFFSET                                 (0x00000524)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE4_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_LUMA_OFFSET                                   (0x00000528)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_OFFSET                                 (0x0000052C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE5_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_LUMA_OFFSET                                   (0x00000530)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_OFFSET                                 (0x00000534)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE6_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_LUMA_OFFSET                                   (0x00000538)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_OFFSET                                 (0x0000053C)
#define LW86B6_VIDEO_COMPOSITOR_SET_SURFACE7_SLOT4_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS                                               (0x00000700)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_GPTIMER_ON                                    0:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_DEBUG_MODE                                    4:4
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_FALCON_CONTROL                                8:8
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_CONFIG_STRUCT_SIZE                            31:16
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID                                                   (0x00000704)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FCE_UCODE                                         3:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_CONFIG                                            7:4
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_PALETTE                                           11:8
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_OUTPUT                                            15:12
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_HIST                                              19:16
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0                                         (0x00000708)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC0                             3:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC1                             7:4
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC2                             11:8
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC3                             15:12
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC4                             19:16
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC5                             23:20
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC6                             27:24
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT0_CTX_ID_SFC7                             31:28
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1                                         (0x0000070C)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC0                             3:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC1                             7:4
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC2                             11:8
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC3                             15:12
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC4                             19:16
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC5                             23:20
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC6                             27:24
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT1_CTX_ID_SFC7                             31:28
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2                                         (0x00000710)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC0                             3:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC1                             7:4
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC2                             11:8
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC3                             15:12
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC4                             19:16
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC5                             23:20
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC6                             27:24
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT2_CTX_ID_SFC7                             31:28
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3                                         (0x00000714)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC0                             3:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC1                             7:4
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC2                             11:8
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC3                             15:12
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC4                             19:16
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC5                             23:20
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC6                             27:24
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT3_CTX_ID_SFC7                             31:28
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4                                         (0x00000718)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC0                             3:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC1                             7:4
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC2                             11:8
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC3                             15:12
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC4                             19:16
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC5                             23:20
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC6                             27:24
#define LW86B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FOR_SLOT4_CTX_ID_SFC7                             31:28
#define LW86B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE                                               (0x0000071C)
#define LW86B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE_FCE_SZ                                        15:0
#define LW86B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET                                         (0x00000720)
#define LW86B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET_OFFSET                                  31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET                                               (0x00000724)
#define LW86B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET_OFFSET                                        31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET                                                  (0x00000728)
#define LW86B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET_OFFSET                                           31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET                                             (0x0000072C)
#define LW86B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET_OFFSET                                      31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET                                   (0x00000730)
#define LW86B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET_OFFSET                            31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_OFFSET                                 (0x00000734)
#define LW86B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_OFFSET_OFFSET                          31:0
#define LW86B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX                                                (0x0000073C)
#define LW86B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX_INDEX                                          31:0
#define LW86B6_VIDEO_COMPOSITOR_PM_TRIGGER_END                                                   (0x00001114)
#define LW86B6_VIDEO_COMPOSITOR_PM_TRIGGER_END_V                                                 31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl86b6_h

