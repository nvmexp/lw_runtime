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

#ifndef _clb0b5_h_
#define _clb0b5_h_

#ifdef __cplusplus
extern "C" {
#endif

#define MAXWELL_DMA_COPY_A                                                            (0x0000B0B5)

typedef volatile struct _clb0b5_tag0 {
    LwV32 Reserved00[0x40];
    LwV32 Nop;                                                                  // 0x00000100 - 0x00000103
    LwV32 Reserved01[0xF];
    LwV32 PmTrigger;                                                            // 0x00000140 - 0x00000143
    LwV32 Reserved02[0x3F];
    LwV32 SetSemaphoreA;                                                        // 0x00000240 - 0x00000243
    LwV32 SetSemaphoreB;                                                        // 0x00000244 - 0x00000247
    LwV32 SetSemaphorePayload;                                                  // 0x00000248 - 0x0000024B
    LwV32 Reserved03[0x2];
    LwV32 SetRenderEnableA;                                                     // 0x00000254 - 0x00000257
    LwV32 SetRenderEnableB;                                                     // 0x00000258 - 0x0000025B
    LwV32 SetRenderEnableC;                                                     // 0x0000025C - 0x0000025F
    LwV32 SetSrcPhysMode;                                                       // 0x00000260 - 0x00000263
    LwV32 SetDstPhysMode;                                                       // 0x00000264 - 0x00000267
    LwV32 Reserved04[0x26];
    LwV32 LaunchDma;                                                            // 0x00000300 - 0x00000303
    LwV32 Reserved05[0x3F];
    LwV32 OffsetInUpper;                                                        // 0x00000400 - 0x00000403
    LwV32 OffsetInLower;                                                        // 0x00000404 - 0x00000407
    LwV32 OffsetOutUpper;                                                       // 0x00000408 - 0x0000040B
    LwV32 OffsetOutLower;                                                       // 0x0000040C - 0x0000040F
    LwV32 PitchIn;                                                              // 0x00000410 - 0x00000413
    LwV32 PitchOut;                                                             // 0x00000414 - 0x00000417
    LwV32 LineLengthIn;                                                         // 0x00000418 - 0x0000041B
    LwV32 LineCount;                                                            // 0x0000041C - 0x0000041F
    LwV32 Reserved06[0xB8];
    LwV32 SetRemapConstA;                                                       // 0x00000700 - 0x00000703
    LwV32 SetRemapConstB;                                                       // 0x00000704 - 0x00000707
    LwV32 SetRemapComponents;                                                   // 0x00000708 - 0x0000070B
    LwV32 SetDstBlockSize;                                                      // 0x0000070C - 0x0000070F
    LwV32 SetDstWidth;                                                          // 0x00000710 - 0x00000713
    LwV32 SetDstHeight;                                                         // 0x00000714 - 0x00000717
    LwV32 SetDstDepth;                                                          // 0x00000718 - 0x0000071B
    LwV32 SetDstLayer;                                                          // 0x0000071C - 0x0000071F
    LwV32 SetDstOrigin;                                                         // 0x00000720 - 0x00000723
    LwV32 Reserved07[0x1];
    LwV32 SetSrcBlockSize;                                                      // 0x00000728 - 0x0000072B
    LwV32 SetSrcWidth;                                                          // 0x0000072C - 0x0000072F
    LwV32 SetSrcHeight;                                                         // 0x00000730 - 0x00000733
    LwV32 SetSrcDepth;                                                          // 0x00000734 - 0x00000737
    LwV32 SetSrcLayer;                                                          // 0x00000738 - 0x0000073B
    LwV32 SetSrcOrigin;                                                         // 0x0000073C - 0x0000073F
    LwV32 Reserved08[0x275];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved09[0x3BA];
} maxwell_dma_copy_aControlPio;

#define LWB0B5_NOP                                                              (0x00000100)
#define LWB0B5_NOP_PARAMETER                                                    31:0
#define LWB0B5_PM_TRIGGER                                                       (0x00000140)
#define LWB0B5_PM_TRIGGER_V                                                     31:0
#define LWB0B5_SET_SEMAPHORE_A                                                  (0x00000240)
#define LWB0B5_SET_SEMAPHORE_A_UPPER                                            7:0
#define LWB0B5_SET_SEMAPHORE_B                                                  (0x00000244)
#define LWB0B5_SET_SEMAPHORE_B_LOWER                                            31:0
#define LWB0B5_SET_SEMAPHORE_PAYLOAD                                            (0x00000248)
#define LWB0B5_SET_SEMAPHORE_PAYLOAD_PAYLOAD                                    31:0
#define LWB0B5_SET_RENDER_ENABLE_A                                              (0x00000254)
#define LWB0B5_SET_RENDER_ENABLE_A_UPPER                                        7:0
#define LWB0B5_SET_RENDER_ENABLE_B                                              (0x00000258)
#define LWB0B5_SET_RENDER_ENABLE_B_LOWER                                        31:0
#define LWB0B5_SET_RENDER_ENABLE_C                                              (0x0000025C)
#define LWB0B5_SET_RENDER_ENABLE_C_MODE                                         2:0
#define LWB0B5_SET_RENDER_ENABLE_C_MODE_FALSE                                   (0x00000000)
#define LWB0B5_SET_RENDER_ENABLE_C_MODE_TRUE                                    (0x00000001)
#define LWB0B5_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                             (0x00000002)
#define LWB0B5_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                         (0x00000003)
#define LWB0B5_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                     (0x00000004)
#define LWB0B5_SET_SRC_PHYS_MODE                                                (0x00000260)
#define LWB0B5_SET_SRC_PHYS_MODE_TARGET                                         1:0
#define LWB0B5_SET_SRC_PHYS_MODE_TARGET_LOCAL_FB                                (0x00000000)
#define LWB0B5_SET_SRC_PHYS_MODE_TARGET_COHERENT_SYSMEM                         (0x00000001)
#define LWB0B5_SET_SRC_PHYS_MODE_TARGET_NONCOHERENT_SYSMEM                      (0x00000002)
#define LWB0B5_SET_DST_PHYS_MODE                                                (0x00000264)
#define LWB0B5_SET_DST_PHYS_MODE_TARGET                                         1:0
#define LWB0B5_SET_DST_PHYS_MODE_TARGET_LOCAL_FB                                (0x00000000)
#define LWB0B5_SET_DST_PHYS_MODE_TARGET_COHERENT_SYSMEM                         (0x00000001)
#define LWB0B5_SET_DST_PHYS_MODE_TARGET_NONCOHERENT_SYSMEM                      (0x00000002)
#define LWB0B5_LAUNCH_DMA                                                       (0x00000300)
#define LWB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE                                    1:0
#define LWB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NONE                               (0x00000000)
#define LWB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_PIPELINED                          (0x00000001)
#define LWB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED                      (0x00000002)
#define LWB0B5_LAUNCH_DMA_FLUSH_ENABLE                                          2:2
#define LWB0B5_LAUNCH_DMA_FLUSH_ENABLE_FALSE                                    (0x00000000)
#define LWB0B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE                                     (0x00000001)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_TYPE                                        4:3
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_TYPE_NONE                                   (0x00000000)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE             (0x00000001)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE            (0x00000002)
#define LWB0B5_LAUNCH_DMA_INTERRUPT_TYPE                                        6:5
#define LWB0B5_LAUNCH_DMA_INTERRUPT_TYPE_NONE                                   (0x00000000)
#define LWB0B5_LAUNCH_DMA_INTERRUPT_TYPE_BLOCKING                               (0x00000001)
#define LWB0B5_LAUNCH_DMA_INTERRUPT_TYPE_NON_BLOCKING                           (0x00000002)
#define LWB0B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT                                     7:7
#define LWB0B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_BLOCKLINEAR                         (0x00000000)
#define LWB0B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH                               (0x00000001)
#define LWB0B5_LAUNCH_DMA_DST_MEMORY_LAYOUT                                     8:8
#define LWB0B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR                         (0x00000000)
#define LWB0B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH                               (0x00000001)
#define LWB0B5_LAUNCH_DMA_MULTI_LINE_ENABLE                                     9:9
#define LWB0B5_LAUNCH_DMA_MULTI_LINE_ENABLE_FALSE                               (0x00000000)
#define LWB0B5_LAUNCH_DMA_MULTI_LINE_ENABLE_TRUE                                (0x00000001)
#define LWB0B5_LAUNCH_DMA_REMAP_ENABLE                                          10:10
#define LWB0B5_LAUNCH_DMA_REMAP_ENABLE_FALSE                                    (0x00000000)
#define LWB0B5_LAUNCH_DMA_REMAP_ENABLE_TRUE                                     (0x00000001)
#define LWB0B5_LAUNCH_DMA_FORCE_RMWDISABLE                                      11:11
#define LWB0B5_LAUNCH_DMA_FORCE_RMWDISABLE_FALSE                                (0x00000000)
#define LWB0B5_LAUNCH_DMA_FORCE_RMWDISABLE_TRUE                                 (0x00000001)
#define LWB0B5_LAUNCH_DMA_SRC_TYPE                                              12:12
#define LWB0B5_LAUNCH_DMA_SRC_TYPE_VIRTUAL                                      (0x00000000)
#define LWB0B5_LAUNCH_DMA_SRC_TYPE_PHYSICAL                                     (0x00000001)
#define LWB0B5_LAUNCH_DMA_DST_TYPE                                              13:13
#define LWB0B5_LAUNCH_DMA_DST_TYPE_VIRTUAL                                      (0x00000000)
#define LWB0B5_LAUNCH_DMA_DST_TYPE_PHYSICAL                                     (0x00000001)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION                                   17:14
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_IMIN                              (0x00000000)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_IMAX                              (0x00000001)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_IXOR                              (0x00000002)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_IAND                              (0x00000003)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_IOR                               (0x00000004)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_IADD                              (0x00000005)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_INC                               (0x00000006)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_DEC                               (0x00000007)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_FADD                              (0x0000000A)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN                              18:18
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN_SIGNED                       (0x00000000)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_SIGN_UNSIGNED                     (0x00000001)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE                            19:19
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE_FALSE                      (0x00000000)
#define LWB0B5_LAUNCH_DMA_SEMAPHORE_REDUCTION_ENABLE_TRUE                       (0x00000001)
#define LWB0B5_LAUNCH_DMA_BYPASS_L2                                             20:20
#define LWB0B5_LAUNCH_DMA_BYPASS_L2_USE_PTE_SETTING                             (0x00000000)
#define LWB0B5_LAUNCH_DMA_BYPASS_L2_FORCE_VOLATILE                              (0x00000001)
#define LWB0B5_OFFSET_IN_UPPER                                                  (0x00000400)
#define LWB0B5_OFFSET_IN_UPPER_UPPER                                            7:0
#define LWB0B5_OFFSET_IN_LOWER                                                  (0x00000404)
#define LWB0B5_OFFSET_IN_LOWER_VALUE                                            31:0
#define LWB0B5_OFFSET_OUT_UPPER                                                 (0x00000408)
#define LWB0B5_OFFSET_OUT_UPPER_UPPER                                           7:0
#define LWB0B5_OFFSET_OUT_LOWER                                                 (0x0000040C)
#define LWB0B5_OFFSET_OUT_LOWER_VALUE                                           31:0
#define LWB0B5_PITCH_IN                                                         (0x00000410)
#define LWB0B5_PITCH_IN_VALUE                                                   31:0
#define LWB0B5_PITCH_OUT                                                        (0x00000414)
#define LWB0B5_PITCH_OUT_VALUE                                                  31:0
#define LWB0B5_LINE_LENGTH_IN                                                   (0x00000418)
#define LWB0B5_LINE_LENGTH_IN_VALUE                                             31:0
#define LWB0B5_LINE_COUNT                                                       (0x0000041C)
#define LWB0B5_LINE_COUNT_VALUE                                                 31:0
#define LWB0B5_SET_REMAP_CONST_A                                                (0x00000700)
#define LWB0B5_SET_REMAP_CONST_A_V                                              31:0
#define LWB0B5_SET_REMAP_CONST_B                                                (0x00000704)
#define LWB0B5_SET_REMAP_CONST_B_V                                              31:0
#define LWB0B5_SET_REMAP_COMPONENTS                                             (0x00000708)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X                                       2:0
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X_SRC_X                                 (0x00000000)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X_SRC_Y                                 (0x00000001)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X_SRC_Z                                 (0x00000002)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X_SRC_W                                 (0x00000003)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X_CONST_A                               (0x00000004)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X_CONST_B                               (0x00000005)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_X_NO_WRITE                              (0x00000006)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y                                       6:4
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y_SRC_X                                 (0x00000000)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y_SRC_Y                                 (0x00000001)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y_SRC_Z                                 (0x00000002)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y_SRC_W                                 (0x00000003)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y_CONST_A                               (0x00000004)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y_CONST_B                               (0x00000005)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Y_NO_WRITE                              (0x00000006)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z                                       10:8
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z_SRC_X                                 (0x00000000)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z_SRC_Y                                 (0x00000001)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z_SRC_Z                                 (0x00000002)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z_SRC_W                                 (0x00000003)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z_CONST_A                               (0x00000004)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z_CONST_B                               (0x00000005)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_Z_NO_WRITE                              (0x00000006)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W                                       14:12
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W_SRC_X                                 (0x00000000)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W_SRC_Y                                 (0x00000001)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W_SRC_Z                                 (0x00000002)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W_SRC_W                                 (0x00000003)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W_CONST_A                               (0x00000004)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W_CONST_B                               (0x00000005)
#define LWB0B5_SET_REMAP_COMPONENTS_DST_W_NO_WRITE                              (0x00000006)
#define LWB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE                              17:16
#define LWB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_ONE                          (0x00000000)
#define LWB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_TWO                          (0x00000001)
#define LWB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_THREE                        (0x00000002)
#define LWB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_FOUR                         (0x00000003)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS                          21:20
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS_ONE                      (0x00000000)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS_TWO                      (0x00000001)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS_THREE                    (0x00000002)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS_FOUR                     (0x00000003)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS                          25:24
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS_ONE                      (0x00000000)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS_TWO                      (0x00000001)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS_THREE                    (0x00000002)
#define LWB0B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS_FOUR                     (0x00000003)
#define LWB0B5_SET_DST_BLOCK_SIZE                                               (0x0000070C)
#define LWB0B5_SET_DST_BLOCK_SIZE_WIDTH                                         3:0
#define LWB0B5_SET_DST_BLOCK_SIZE_WIDTH_QUARTER_GOB                             (0x0000000E)
#define LWB0B5_SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB                                 (0x00000000)
#define LWB0B5_SET_DST_BLOCK_SIZE_HEIGHT                                        7:4
#define LWB0B5_SET_DST_BLOCK_SIZE_HEIGHT_ONE_GOB                                (0x00000000)
#define LWB0B5_SET_DST_BLOCK_SIZE_HEIGHT_TWO_GOBS                               (0x00000001)
#define LWB0B5_SET_DST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                              (0x00000002)
#define LWB0B5_SET_DST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                             (0x00000003)
#define LWB0B5_SET_DST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                           (0x00000004)
#define LWB0B5_SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                         (0x00000005)
#define LWB0B5_SET_DST_BLOCK_SIZE_DEPTH                                         11:8
#define LWB0B5_SET_DST_BLOCK_SIZE_DEPTH_ONE_GOB                                 (0x00000000)
#define LWB0B5_SET_DST_BLOCK_SIZE_DEPTH_TWO_GOBS                                (0x00000001)
#define LWB0B5_SET_DST_BLOCK_SIZE_DEPTH_FOUR_GOBS                               (0x00000002)
#define LWB0B5_SET_DST_BLOCK_SIZE_DEPTH_EIGHT_GOBS                              (0x00000003)
#define LWB0B5_SET_DST_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                            (0x00000004)
#define LWB0B5_SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                          (0x00000005)
#define LWB0B5_SET_DST_BLOCK_SIZE_GOB_HEIGHT                                    15:12
#define LWB0B5_SET_DST_BLOCK_SIZE_GOB_HEIGHT_GOB_HEIGHT_TESLA_4                 (0x00000000)
#define LWB0B5_SET_DST_BLOCK_SIZE_GOB_HEIGHT_GOB_HEIGHT_FERMI_8                 (0x00000001)
#define LWB0B5_SET_DST_WIDTH                                                    (0x00000710)
#define LWB0B5_SET_DST_WIDTH_V                                                  31:0
#define LWB0B5_SET_DST_HEIGHT                                                   (0x00000714)
#define LWB0B5_SET_DST_HEIGHT_V                                                 31:0
#define LWB0B5_SET_DST_DEPTH                                                    (0x00000718)
#define LWB0B5_SET_DST_DEPTH_V                                                  31:0
#define LWB0B5_SET_DST_LAYER                                                    (0x0000071C)
#define LWB0B5_SET_DST_LAYER_V                                                  31:0
#define LWB0B5_SET_DST_ORIGIN                                                   (0x00000720)
#define LWB0B5_SET_DST_ORIGIN_X                                                 15:0
#define LWB0B5_SET_DST_ORIGIN_Y                                                 31:16
#define LWB0B5_SET_SRC_BLOCK_SIZE                                               (0x00000728)
#define LWB0B5_SET_SRC_BLOCK_SIZE_WIDTH                                         3:0
#define LWB0B5_SET_SRC_BLOCK_SIZE_WIDTH_QUARTER_GOB                             (0x0000000E)
#define LWB0B5_SET_SRC_BLOCK_SIZE_WIDTH_ONE_GOB                                 (0x00000000)
#define LWB0B5_SET_SRC_BLOCK_SIZE_HEIGHT                                        7:4
#define LWB0B5_SET_SRC_BLOCK_SIZE_HEIGHT_ONE_GOB                                (0x00000000)
#define LWB0B5_SET_SRC_BLOCK_SIZE_HEIGHT_TWO_GOBS                               (0x00000001)
#define LWB0B5_SET_SRC_BLOCK_SIZE_HEIGHT_FOUR_GOBS                              (0x00000002)
#define LWB0B5_SET_SRC_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                             (0x00000003)
#define LWB0B5_SET_SRC_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                           (0x00000004)
#define LWB0B5_SET_SRC_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                         (0x00000005)
#define LWB0B5_SET_SRC_BLOCK_SIZE_DEPTH                                         11:8
#define LWB0B5_SET_SRC_BLOCK_SIZE_DEPTH_ONE_GOB                                 (0x00000000)
#define LWB0B5_SET_SRC_BLOCK_SIZE_DEPTH_TWO_GOBS                                (0x00000001)
#define LWB0B5_SET_SRC_BLOCK_SIZE_DEPTH_FOUR_GOBS                               (0x00000002)
#define LWB0B5_SET_SRC_BLOCK_SIZE_DEPTH_EIGHT_GOBS                              (0x00000003)
#define LWB0B5_SET_SRC_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                            (0x00000004)
#define LWB0B5_SET_SRC_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                          (0x00000005)
#define LWB0B5_SET_SRC_BLOCK_SIZE_GOB_HEIGHT                                    15:12
#define LWB0B5_SET_SRC_BLOCK_SIZE_GOB_HEIGHT_GOB_HEIGHT_TESLA_4                 (0x00000000)
#define LWB0B5_SET_SRC_BLOCK_SIZE_GOB_HEIGHT_GOB_HEIGHT_FERMI_8                 (0x00000001)
#define LWB0B5_SET_SRC_WIDTH                                                    (0x0000072C)
#define LWB0B5_SET_SRC_WIDTH_V                                                  31:0
#define LWB0B5_SET_SRC_HEIGHT                                                   (0x00000730)
#define LWB0B5_SET_SRC_HEIGHT_V                                                 31:0
#define LWB0B5_SET_SRC_DEPTH                                                    (0x00000734)
#define LWB0B5_SET_SRC_DEPTH_V                                                  31:0
#define LWB0B5_SET_SRC_LAYER                                                    (0x00000738)
#define LWB0B5_SET_SRC_LAYER_V                                                  31:0
#define LWB0B5_SET_SRC_ORIGIN                                                   (0x0000073C)
#define LWB0B5_SET_SRC_ORIGIN_X                                                 15:0
#define LWB0B5_SET_SRC_ORIGIN_Y                                                 31:16
#define LWB0B5_PM_TRIGGER_END                                                   (0x00001114)
#define LWB0B5_PM_TRIGGER_END_V                                                 31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clb0b5_h

