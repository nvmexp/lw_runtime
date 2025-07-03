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

#ifndef _clb0b6_h_
#define _clb0b6_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWB0B6_VIDEO_COMPOSITOR                                             (0x0000B0B6)

typedef volatile struct _clb0b6_tag0 {
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
    LwV32 SetSurface0LumaOffset[8];                                             // 0x00000400 - 0x0000041F
    LwV32 Reserved06[0x7];
    LwV32 SetSurface0ChromaU_Offset[8];                                         // 0x00000404 - 0x00000423
    LwV32 Reserved07[0x7];
    LwV32 SetSurface0ChromaV_Offset[8];                                         // 0x00000408 - 0x00000427
    LwV32 Reserved08[0x7];
    LwV32 SetSurface1LumaOffset[8];                                             // 0x0000040C - 0x0000042B
    LwV32 Reserved09[0x7];
    LwV32 SetSurface1ChromaU_Offset[8];                                         // 0x00000410 - 0x0000042F
    LwV32 Reserved10[0x7];
    LwV32 SetSurface1ChromaV_Offset[8];                                         // 0x00000414 - 0x00000433
    LwV32 Reserved11[0x7];
    LwV32 SetSurface2LumaOffset[8];                                             // 0x00000418 - 0x00000437
    LwV32 Reserved12[0x7];
    LwV32 SetSurface2ChromaU_Offset[8];                                         // 0x0000041C - 0x0000043B
    LwV32 Reserved13[0x7];
    LwV32 SetSurface2ChromaV_Offset[8];                                         // 0x00000420 - 0x0000043F
    LwV32 Reserved14[0x7];
    LwV32 SetSurface3LumaOffset[8];                                             // 0x00000424 - 0x00000443
    LwV32 Reserved15[0x7];
    LwV32 SetSurface3ChromaU_Offset[8];                                         // 0x00000428 - 0x00000447
    LwV32 Reserved16[0x7];
    LwV32 SetSurface3ChromaV_Offset[8];                                         // 0x0000042C - 0x0000044B
    LwV32 Reserved17[0x7];
    LwV32 SetSurface4LumaOffset[8];                                             // 0x00000430 - 0x0000044F
    LwV32 Reserved18[0x7];
    LwV32 SetSurface4ChromaU_Offset[8];                                         // 0x00000434 - 0x00000453
    LwV32 Reserved19[0x7];
    LwV32 SetSurface4ChromaV_Offset[8];                                         // 0x00000438 - 0x00000457
    LwV32 Reserved20[0x7];
    LwV32 SetSurface5LumaOffset[8];                                             // 0x0000043C - 0x0000045B
    LwV32 Reserved21[0x7];
    LwV32 SetSurface5ChromaU_Offset[8];                                         // 0x00000440 - 0x0000045F
    LwV32 Reserved22[0x7];
    LwV32 SetSurface5ChromaV_Offset[8];                                         // 0x00000444 - 0x00000463
    LwV32 Reserved23[0x7];
    LwV32 SetSurface6LumaOffset[8];                                             // 0x00000448 - 0x00000467
    LwV32 Reserved24[0x7];
    LwV32 SetSurface6ChromaU_Offset[8];                                         // 0x0000044C - 0x0000046B
    LwV32 Reserved25[0x7];
    LwV32 SetSurface6ChromaV_Offset[8];                                         // 0x00000450 - 0x0000046F
    LwV32 Reserved26[0x7];
    LwV32 SetSurface7LumaOffset[8];                                             // 0x00000454 - 0x00000473
    LwV32 Reserved27[0x7];
    LwV32 SetSurface7ChromaU_Offset[8];                                         // 0x00000458 - 0x00000477
    LwV32 Reserved28[0x7];
    LwV32 SetSurface7ChromaV_Offset[8];                                         // 0x0000045C - 0x0000047B
    LwV32 Reserved29[0xA1];
    LwV32 SetPictureIndex;                                                      // 0x00000700 - 0x00000703
    LwV32 SetControlParams;                                                     // 0x00000704 - 0x00000707
    LwV32 SetConfigStructOffset;                                                // 0x00000708 - 0x0000070B
    LwV32 SetFilterStructOffset;                                                // 0x0000070C - 0x0000070F
    LwV32 SetPaletteOffset;                                                     // 0x00000710 - 0x00000713
    LwV32 SetHistOffset;                                                        // 0x00000714 - 0x00000717
    LwV32 SetContextId;                                                         // 0x00000718 - 0x0000071B
    LwV32 SetFceUcodeSize;                                                      // 0x0000071C - 0x0000071F
    LwV32 SetOutputSurfaceLumaOffset;                                           // 0x00000720 - 0x00000723
    LwV32 SetOutputSurfaceChromaU_Offset;                                       // 0x00000724 - 0x00000727
    LwV32 SetOutputSurfaceChromaV_Offset;                                       // 0x00000728 - 0x0000072B
    LwV32 SetFceUcodeOffset;                                                    // 0x0000072C - 0x0000072F
    LwV32 SetCrcStructOffset;                                                   // 0x00000730 - 0x00000733
    LwV32 SetCrcMode;                                                           // 0x00000734 - 0x00000737
    LwV32 Reserved30[0x2];
    LwV32 SetSlotContextId[8];                                                  // 0x00000740 - 0x0000075F
    LwV32 SetCompTagBuffer_Offset[8];                                           // 0x00000760 - 0x0000077F
    LwV32 SetHistoryBufferOffset[8];                                            // 0x00000780 - 0x0000079F
    LwV32 Reserved31[0x25D];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved32[0x3BA];
} T40VICControlPio;

#define LWB0B6_VIDEO_COMPOSITOR_NOP                                                              (0x00000100)
#define LWB0B6_VIDEO_COMPOSITOR_NOP_PARAMETER                                                    31:0
#define LWB0B6_VIDEO_COMPOSITOR_PM_TRIGGER                                                       (0x00000140)
#define LWB0B6_VIDEO_COMPOSITOR_PM_TRIGGER_V                                                     31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID                                               (0x00000200)
#define LWB0B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID                                            31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID_COMPOSITOR                                 (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWB0B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_A                                                      (0x00000240)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_A_UPPER                                                7:0
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_B                                                      (0x00000244)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_B_LOWER                                                31:0
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_C                                                      (0x00000248)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA                                                    (0x0000024C)
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA_OFFSET                                             27:0
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA_CTX_VALID                                          31:28
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH                                                       (0x00000250)
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE                                               0:0
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE_FALSE                                         (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE_TRUE                                          (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY                                            1:1
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY_FALSE                                      (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY_TRUE                                       (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESERVED                                              7:2
#define LWB0B6_VIDEO_COMPOSITOR_CTX_SWITCH_ASID                                                  23:8
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE                                                          (0x00000300)
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY                                                   0:0
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON                                                1:1
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN                                                   8:8
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D                                                      (0x00000304)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION                                            17:16
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWB0B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE0_LUMA_OFFSET(b)                                      (0x00000400 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE0_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_U_OFFSET(b)                                  (0x00000404 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_V_OFFSET(b)                                  (0x00000408 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE1_LUMA_OFFSET(b)                                      (0x0000040C + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE1_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_U_OFFSET(b)                                  (0x00000410 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_V_OFFSET(b)                                  (0x00000414 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE2_LUMA_OFFSET(b)                                      (0x00000418 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE2_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_U_OFFSET(b)                                  (0x0000041C + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_V_OFFSET(b)                                  (0x00000420 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE3_LUMA_OFFSET(b)                                      (0x00000424 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE3_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_U_OFFSET(b)                                  (0x00000428 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_V_OFFSET(b)                                  (0x0000042C + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE4_LUMA_OFFSET(b)                                      (0x00000430 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE4_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_U_OFFSET(b)                                  (0x00000434 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_V_OFFSET(b)                                  (0x00000438 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE5_LUMA_OFFSET(b)                                      (0x0000043C + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE5_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_U_OFFSET(b)                                  (0x00000440 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_V_OFFSET(b)                                  (0x00000444 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE6_LUMA_OFFSET(b)                                      (0x00000448 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE6_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_U_OFFSET(b)                                  (0x0000044C + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_V_OFFSET(b)                                  (0x00000450 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE7_LUMA_OFFSET(b)                                      (0x00000454 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE7_LUMA_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_U_OFFSET(b)                                  (0x00000458 + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_V_OFFSET(b)                                  (0x0000045C + (b)*0x00000060)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX                                                (0x00000700)
#define LWB0B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS                                               (0x00000704)
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_GPTIMER_ON                                    0:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_DEBUG_MODE                                    4:4
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_FALCON_CONTROL                                8:8
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_CONFIG_STRUCT_SIZE                            31:16
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET                                         (0x00000708)
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_FILTER_STRUCT_OFFSET                                         (0x0000070C)
#define LWB0B6_VIDEO_COMPOSITOR_SET_FILTER_STRUCT_OFFSET_OFFSET                                  31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET                                               (0x00000710)
#define LWB0B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET_OFFSET                                        31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET                                                  (0x00000714)
#define LWB0B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET_OFFSET                                           31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID                                                   (0x00000718)
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FCE_UCODE                                         3:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_CONFIG                                            7:4
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_PALETTE                                           11:8
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_OUTPUT                                            15:12
#define LWB0B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_HIST                                              19:16
#define LWB0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE                                               (0x0000071C)
#define LWB0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE_FCE_SZ                                        15:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET                                   (0x00000720)
#define LWB0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET_OFFSET                            31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_U_OFFSET                               (0x00000724)
#define LWB0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_V_OFFSET                               (0x00000728)
#define LWB0B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET                                             (0x0000072C)
#define LWB0B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET_OFFSET                                      31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_STRUCT_OFFSET                                            (0x00000730)
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_STRUCT_OFFSET_OFFSET                                     31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_MODE                                                     (0x00000734)
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_ASEL                                      3:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_BSEL                                      7:4
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_CSEL                                      11:8
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_DSEL                                      15:12
#define LWB0B6_VIDEO_COMPOSITOR_SET_CRC_MODE_CRC_MODE                                            16:16
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID(b)                                           (0x00000740 + (b)*0x00000004)
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC0                                  3:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC1                                  7:4
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC2                                  11:8
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC3                                  15:12
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC4                                  19:16
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC5                                  23:20
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC6                                  27:24
#define LWB0B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC7                                  31:28
#define LWB0B6_VIDEO_COMPOSITOR_SET_COMP_TAG_BUFFER_OFFSET(b)                                    (0x00000760 + (b)*0x00000004)
#define LWB0B6_VIDEO_COMPOSITOR_SET_COMP_TAG_BUFFER_OFFSET_OFFSET                                31:0
#define LWB0B6_VIDEO_COMPOSITOR_SET_HISTORY_BUFFER_OFFSET(b)                                     (0x00000780 + (b)*0x00000004)
#define LWB0B6_VIDEO_COMPOSITOR_SET_HISTORY_BUFFER_OFFSET_OFFSET                                 31:0
#define LWB0B6_VIDEO_COMPOSITOR_PM_TRIGGER_END                                                   (0x00001114)
#define LWB0B6_VIDEO_COMPOSITOR_PM_TRIGGER_END_V                                                 31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clb0b6_h

