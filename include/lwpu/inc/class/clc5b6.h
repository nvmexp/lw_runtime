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

#ifndef _clc5b6_h_
#define _clc5b6_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWC5B6_VIDEO_COMPOSITOR                                             (0x0000C5B6)

typedef volatile struct _clc5b6_tag0 {
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
    LwV32 Reserved05[0xFE];
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
    LwV32 SetStatusOffset;                                                      // 0x00000738 - 0x0000073B
    LwV32 Reserved06[0x1];
    LwV32 SetSlotContextId[16];                                                 // 0x00000740 - 0x0000077F
    LwV32 SetHistoryBufferOffset[16];                                           // 0x00000780 - 0x000007BF
    LwV32 SetCompTagBuffer_Offset[16];                                          // 0x000007C0 - 0x000007FF
    LwV32 SetSparseWarpMap_Offset;                                              // 0x00000800 - 0x00000803
    LwV32 SetMaskBitMap_Offset;                                                 // 0x00000804 - 0x00000807
    LwV32 SetXsobelSurface_Offset;                                              // 0x00000808 - 0x0000080B
    LwV32 SetXsobelDsSurface_Offset;                                            // 0x0000080C - 0x0000080F
    LwV32 SetXSobelNeighborBuffer_Offset;                                       // 0x00000810 - 0x00000813
    LwV32 SetTNR3PrevFrmSurfaceLumaOffset;                                      // 0x00000814 - 0x00000817
    LwV32 SetTNR3PrevFrmSurfaceChromaU_Offset;                                  // 0x00000818 - 0x0000081B
    LwV32 SetTNR3PrevFrmSurfaceChromaV_Offset;                                  // 0x0000081C - 0x0000081F
    LwV32 SetTNR3LwrAlphaSurfaceOffset;                                         // 0x00000820 - 0x00000823
    LwV32 SetTNR3PrevAlphaSurfaceOffset;                                        // 0x00000824 - 0x00000827
    LwV32 SetTNR3NeighborBufferOffset;                                          // 0x00000828 - 0x0000082B
    LwV32 SetStatusNotifierInputOffset;                                         // 0x0000082C - 0x0000082F
    LwV32 SetStatusNotifierOutputOffset;                                        // 0x00000830 - 0x00000833
    LwV32 Reserved07[0x238];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved08[0x3A];
    LwV32 SetSurface0LumaOffset[16];                                            // 0x00001200 - 0x0000123F
    LwV32 Reserved09[0xF];
    LwV32 SetSurface0ChromaU_Offset[16];                                        // 0x00001204 - 0x00001243
    LwV32 Reserved10[0xF];
    LwV32 SetSurface0ChromaV_Offset[16];                                        // 0x00001208 - 0x00001247
    LwV32 Reserved11[0xF];
    LwV32 SetSurface1LumaOffset[16];                                            // 0x0000120C - 0x0000124B
    LwV32 Reserved12[0xF];
    LwV32 SetSurface1ChromaU_Offset[16];                                        // 0x00001210 - 0x0000124F
    LwV32 Reserved13[0xF];
    LwV32 SetSurface1ChromaV_Offset[16];                                        // 0x00001214 - 0x00001253
    LwV32 Reserved14[0xF];
    LwV32 SetSurface2LumaOffset[16];                                            // 0x00001218 - 0x00001257
    LwV32 Reserved15[0xF];
    LwV32 SetSurface2ChromaU_Offset[16];                                        // 0x0000121C - 0x0000125B
    LwV32 Reserved16[0xF];
    LwV32 SetSurface2ChromaV_Offset[16];                                        // 0x00001220 - 0x0000125F
    LwV32 Reserved17[0xF];
    LwV32 SetSurface3LumaOffset[16];                                            // 0x00001224 - 0x00001263
    LwV32 Reserved18[0xF];
    LwV32 SetSurface3ChromaU_Offset[16];                                        // 0x00001228 - 0x00001267
    LwV32 Reserved19[0xF];
    LwV32 SetSurface3ChromaV_Offset[16];                                        // 0x0000122C - 0x0000126B
    LwV32 Reserved20[0xF];
    LwV32 SetSurface4LumaOffset[16];                                            // 0x00001230 - 0x0000126F
    LwV32 Reserved21[0xF];
    LwV32 SetSurface4ChromaU_Offset[16];                                        // 0x00001234 - 0x00001273
    LwV32 Reserved22[0xF];
    LwV32 SetSurface4ChromaV_Offset[16];                                        // 0x00001238 - 0x00001277
    LwV32 Reserved23[0xF];
    LwV32 SetSurface5LumaOffset[16];                                            // 0x0000123C - 0x0000127B
    LwV32 Reserved24[0xF];
    LwV32 SetSurface5ChromaU_Offset[16];                                        // 0x00001240 - 0x0000127F
    LwV32 Reserved25[0xF];
    LwV32 SetSurface5ChromaV_Offset[16];                                        // 0x00001244 - 0x00001283
    LwV32 Reserved26[0xF];
    LwV32 SetSurface6LumaOffset[16];                                            // 0x00001248 - 0x00001287
    LwV32 Reserved27[0xF];
    LwV32 SetSurface6ChromaU_Offset[16];                                        // 0x0000124C - 0x0000128B
    LwV32 Reserved28[0xF];
    LwV32 SetSurface6ChromaV_Offset[16];                                        // 0x00001250 - 0x0000128F
    LwV32 Reserved29[0xF];
    LwV32 SetSurface7LumaOffset[16];                                            // 0x00001254 - 0x00001293
    LwV32 Reserved30[0xF];
    LwV32 SetSurface7ChromaU_Offset[16];                                        // 0x00001258 - 0x00001297
    LwV32 Reserved31[0xF];
    LwV32 SetSurface7ChromaV_Offset[16];                                        // 0x0000125C - 0x0000129B
    LwV32 Reserved32[0x359];
} C5B6VICControlPio;

#define LWC5B6_VIDEO_COMPOSITOR_NOP                                                              (0x00000100)
#define LWC5B6_VIDEO_COMPOSITOR_NOP_PARAMETER                                                    31:0
#define LWC5B6_VIDEO_COMPOSITOR_PM_TRIGGER                                                       (0x00000140)
#define LWC5B6_VIDEO_COMPOSITOR_PM_TRIGGER_V                                                     31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID                                               (0x00000200)
#define LWC5B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID                                            31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_APPLICATION_ID_ID_COMPOSITOR                                 (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWC5B6_VIDEO_COMPOSITOR_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_A                                                      (0x00000240)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_A_UPPER                                                7:0
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_B                                                      (0x00000244)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_B_LOWER                                                31:0
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_C                                                      (0x00000248)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA                                                    (0x0000024C)
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA_OFFSET                                             27:0
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SAVE_AREA_CTX_VALID                                          31:28
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH                                                       (0x00000250)
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE                                               0:0
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE_FALSE                                         (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESTORE_TRUE                                          (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY                                            1:1
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY_FALSE                                      (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_RST_NOTIFY_TRUE                                       (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_RESERVED                                              7:2
#define LWC5B6_VIDEO_COMPOSITOR_CTX_SWITCH_ASID                                                  23:8
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE                                                          (0x00000300)
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY                                                   0:0
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON                                                1:1
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN                                                   8:8
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D                                                      (0x00000304)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION                                            17:16
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWC5B6_VIDEO_COMPOSITOR_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWC5B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX                                                (0x00000700)
#define LWC5B6_VIDEO_COMPOSITOR_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS                                               (0x00000704)
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_GPTIMER_ON                                    0:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_DEBUG_MODE                                    4:4
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_BUF_STAT_NOTIFIER                             5:5
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_FALCON_CONTROL                                8:8
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTROL_PARAMS_CONFIG_STRUCT_SIZE                            31:16
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET                                         (0x00000708)
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONFIG_STRUCT_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_FILTER_STRUCT_OFFSET                                         (0x0000070C)
#define LWC5B6_VIDEO_COMPOSITOR_SET_FILTER_STRUCT_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET                                               (0x00000710)
#define LWC5B6_VIDEO_COMPOSITOR_SET_PALETTE_OFFSET_OFFSET                                        31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET                                                  (0x00000714)
#define LWC5B6_VIDEO_COMPOSITOR_SET_HIST_OFFSET_OFFSET                                           31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID                                                   (0x00000718)
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_FCE_UCODE                                         3:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_CONFIG                                            7:4
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_PALETTE                                           11:8
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_OUTPUT                                            15:12
#define LWC5B6_VIDEO_COMPOSITOR_SET_CONTEXT_ID_HIST                                              19:16
#define LWC5B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE                                               (0x0000071C)
#define LWC5B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_SIZE_FCE_SZ                                        15:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET                                   (0x00000720)
#define LWC5B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_LUMA_OFFSET_OFFSET                            31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_U_OFFSET                               (0x00000724)
#define LWC5B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_U_OFFSET_OFFSET                        31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_V_OFFSET                               (0x00000728)
#define LWC5B6_VIDEO_COMPOSITOR_SET_OUTPUT_SURFACE_CHROMA_V_OFFSET_OFFSET                        31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET                                             (0x0000072C)
#define LWC5B6_VIDEO_COMPOSITOR_SET_FCE_UCODE_OFFSET_OFFSET                                      31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_STRUCT_OFFSET                                            (0x00000730)
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_STRUCT_OFFSET_OFFSET                                     31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_MODE                                                     (0x00000734)
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_ASEL                                      3:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_BSEL                                      7:4
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_CSEL                                      11:8
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_MODE_INTF_PART_DSEL                                      15:12
#define LWC5B6_VIDEO_COMPOSITOR_SET_CRC_MODE_CRC_MODE                                            16:16
#define LWC5B6_VIDEO_COMPOSITOR_SET_STATUS_OFFSET                                                (0x00000738)
#define LWC5B6_VIDEO_COMPOSITOR_SET_STATUS_OFFSET_OFFSET                                         31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID(b)                                           (0x00000740 + (b)*0x00000004)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC0                                  3:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC1                                  7:4
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC2                                  11:8
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC3                                  15:12
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC4                                  19:16
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC5                                  23:20
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC6                                  27:24
#define LWC5B6_VIDEO_COMPOSITOR_SET_SLOT_CONTEXT_ID_CTX_ID_SFC7                                  31:28
#define LWC5B6_VIDEO_COMPOSITOR_SET_HISTORY_BUFFER_OFFSET(b)                                     (0x00000780 + (b)*0x00000004)
#define LWC5B6_VIDEO_COMPOSITOR_SET_HISTORY_BUFFER_OFFSET_OFFSET                                 31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_COMP_TAG_BUFFER_OFFSET(b)                                    (0x000007C0 + (b)*0x00000004)
#define LWC5B6_VIDEO_COMPOSITOR_SET_COMP_TAG_BUFFER_OFFSET_OFFSET                                31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SPARSE_WARP_MAP_OFFSET                                       (0x00000800)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SPARSE_WARP_MAP_OFFSET_OFFSET                                31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_MASK_BIT_MAP_OFFSET                                          (0x00000804)
#define LWC5B6_VIDEO_COMPOSITOR_SET_MASK_BIT_MAP_OFFSET_OFFSET                                   31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_XSOBEL_SURFACE_OFFSET                                        (0x00000808)
#define LWC5B6_VIDEO_COMPOSITOR_SET_XSOBEL_SURFACE_OFFSET_OFFSET                                 31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_XSOBEL_DS_SURFACE_OFFSET                                     (0x0000080C)
#define LWC5B6_VIDEO_COMPOSITOR_SET_XSOBEL_DS_SURFACE_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_XSOBEL_NEIGHBOR_BUFFER_OFFSET                                (0x00000810)
#define LWC5B6_VIDEO_COMPOSITOR_SET_XSOBEL_NEIGHBOR_BUFFER_OFFSET_OFFSET                         31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_FRM_SURFACE_LUMA_OFFSET                            (0x00000814)
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_FRM_SURFACE_LUMA_OFFSET_OFFSET                     31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_FRM_SURFACE_CHROMA_U_OFFSET                        (0x00000818)
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_FRM_SURFACE_CHROMA_U_OFFSET_OFFSET                 31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_FRM_SURFACE_CHROMA_V_OFFSET                        (0x0000081C)
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_FRM_SURFACE_CHROMA_V_OFFSET_OFFSET                 31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_LWR_ALPHA_SURFACE_OFFSET                                (0x00000820)
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_LWR_ALPHA_SURFACE_OFFSET_OFFSET                         31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_ALPHA_SURFACE_OFFSET                               (0x00000824)
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_PREV_ALPHA_SURFACE_OFFSET_OFFSET                        31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_NEIGHBOR_BUFFER_OFFSET                                  (0x00000828)
#define LWC5B6_VIDEO_COMPOSITOR_SET_TNR3_NEIGHBOR_BUFFER_OFFSET_OFFSET                           31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_STATUS_NOTIFIER_INPUT_OFFSET                                 (0x0000082C)
#define LWC5B6_VIDEO_COMPOSITOR_SET_STATUS_NOTIFIER_INPUT_OFFSET_OFFSET                          31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_STATUS_NOTIFIER_OUTPUT_OFFSET                                (0x00000830)
#define LWC5B6_VIDEO_COMPOSITOR_SET_STATUS_NOTIFIER_OUTPUT_OFFSET_OFFSET                         31:0
#define LWC5B6_VIDEO_COMPOSITOR_PM_TRIGGER_END                                                   (0x00001114)
#define LWC5B6_VIDEO_COMPOSITOR_PM_TRIGGER_END_V                                                 31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE0_LUMA_OFFSET(b)                                      (0x00001200 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE0_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_U_OFFSET(b)                                  (0x00001204 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_V_OFFSET(b)                                  (0x00001208 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE0_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE1_LUMA_OFFSET(b)                                      (0x0000120C + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE1_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_U_OFFSET(b)                                  (0x00001210 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_V_OFFSET(b)                                  (0x00001214 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE1_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE2_LUMA_OFFSET(b)                                      (0x00001218 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE2_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_U_OFFSET(b)                                  (0x0000121C + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_V_OFFSET(b)                                  (0x00001220 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE2_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE3_LUMA_OFFSET(b)                                      (0x00001224 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE3_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_U_OFFSET(b)                                  (0x00001228 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_V_OFFSET(b)                                  (0x0000122C + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE3_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE4_LUMA_OFFSET(b)                                      (0x00001230 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE4_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_U_OFFSET(b)                                  (0x00001234 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_V_OFFSET(b)                                  (0x00001238 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE4_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE5_LUMA_OFFSET(b)                                      (0x0000123C + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE5_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_U_OFFSET(b)                                  (0x00001240 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_V_OFFSET(b)                                  (0x00001244 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE5_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE6_LUMA_OFFSET(b)                                      (0x00001248 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE6_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_U_OFFSET(b)                                  (0x0000124C + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_V_OFFSET(b)                                  (0x00001250 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE6_CHROMA_V_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE7_LUMA_OFFSET(b)                                      (0x00001254 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE7_LUMA_OFFSET_OFFSET                                  31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_U_OFFSET(b)                                  (0x00001258 + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_U_OFFSET_OFFSET                              31:0
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_V_OFFSET(b)                                  (0x0000125C + (b)*0x00000060)
#define LWC5B6_VIDEO_COMPOSITOR_SET_SURFACE7_CHROMA_V_OFFSET_OFFSET                              31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clc5b6_h

