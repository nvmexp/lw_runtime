/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2003-2004 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_lw50_twod_h_
#define _cl_lw50_twod_h_

/* This file is generated - do not edit. */

#include "lwtypes.h"

#define LW50_TWOD    0x502D

typedef volatile struct _cl502d_tag0 {
    LwU32 SetObject;
    LwU32 Reserved_0x04[0x3F];
    LwU32 NoOperation;
    LwU32 Notify;
    LwU32 Reserved_0x108[0x2];
    LwU32 WaitForIdle;
    LwU32 Reserved_0x114[0xB];
    LwU32 PmTrigger;
    LwU32 Reserved_0x144[0xF];
    LwU32 SetContextDmaNotify;
    LwU32 SetDstContextDma;
    LwU32 SetSrcContextDma;
    LwU32 SetSemaphoreContextDma;
    LwU32 Reserved_0x190[0x1C];
    LwU32 SetDstFormat;
    LwU32 SetDstMemoryLayout;
    LwU32 SetDstBlockSize;
    LwU32 SetDstDepth;
    LwU32 SetDstLayer;
    LwU32 SetDstPitch;
    LwU32 SetDstWidth;
    LwU32 SetDstHeight;
    LwU32 SetDstOffsetUpper;
    LwU32 SetDstOffsetLower;
    LwU32 SetPixelsFromCpuIndexWrap;
    LwU32 SetKind2dCheckEnable;
    LwU32 SetSrcFormat;
    LwU32 SetSrcMemoryLayout;
    LwU32 SetSrcBlockSize;
    LwU32 SetSrcDepth;
    LwU32 SetSrcLayer;
    LwU32 SetSrcPitch;
    LwU32 SetSrcWidth;
    LwU32 SetSrcHeight;
    LwU32 SetSrcOffsetUpper;
    LwU32 SetSrcOffsetLower;
    LwU32 SetPixelsFromMemorySectorPromotion;
    LwU32 Reserved_0x25C[0x1];
    LwU32 SetNumTpcs;
    LwU32 SetRenderEnableA;
    LwU32 SetRenderEnableB;
    LwU32 SetRenderEnableC;
    LwU32 Reserved_0x270[0x4];
    LwU32 SetClipX0;
    LwU32 SetClipY0;
    LwU32 SetClipWidth;
    LwU32 SetClipHeight;
    LwU32 SetClipEnable;
    LwU32 SetColorKeyFormat;
    LwU32 SetColorKey;
    LwU32 SetColorKeyEnable;
    LwU32 SetRop;
    LwU32 SetBeta1;
    LwU32 SetBeta4;
    LwU32 SetOperation;
    LwU32 SetPatternOffset;
    LwU32 SetPatternSelect;
    LwU32 Reserved_0x2B8[0xC];
    LwU32 SetMonochromePatternColorFormat;
    LwU32 SetMonochromePatternFormat;
    LwU32 SetMonochromePatternColor0;
    LwU32 SetMonochromePatternColor1;
    LwU32 SetMonochromePattern0;
    LwU32 SetMonochromePattern1;
    LwU32 ColorPatternX8R8G8B8[0x40];
    LwU32 ColorPatternR5G6B5[0x20];
    LwU32 ColorPatternX1R5G5B5[0x20];
    LwU32 ColorPatternY8[0x10];
    LwU32 Reserved_0x540[0x10];
    LwU32 RenderSolidPrimMode;
    LwU32 SetRenderSolidPrimColorFormat;
    LwU32 SetRenderSolidPrimColor;
    LwU32 SetRenderSolidLineTieBreakBits;
    LwU32 Reserved_0x590[0x14];
    LwU32 RenderSolidPrimPointXY;
    LwU32 Reserved_0x5E4[0x7];
    struct {
        LwU32 SetX;
        LwU32 Y;
    } RenderSolidPrimPoint[0x40];
    LwU32 SetPixelsFromCpuDataType;
    LwU32 SetPixelsFromCpuColorFormat;
    LwU32 SetPixelsFromCpuIndexFormat;
    LwU32 SetPixelsFromCpuMonoFormat;
    LwU32 SetPixelsFromCpuWrap;
    LwU32 SetPixelsFromCpuColor0;
    LwU32 SetPixelsFromCpuColor1;
    LwU32 SetPixelsFromCpuMonoOpacity;
    LwU32 Reserved_0x820[0x6];
    LwU32 SetPixelsFromCpuSrcWidth;
    LwU32 SetPixelsFromCpuSrcHeight;
    LwU32 SetPixelsFromCpuDxDuFrac;
    LwU32 SetPixelsFromCpuDxDuInt;
    LwU32 SetPixelsFromCpuDyDvFrac;
    LwU32 SetPixelsFromCpuDyDvInt;
    LwU32 SetPixelsFromCpuDstX0Frac;
    LwU32 SetPixelsFromCpuDstX0Int;
    LwU32 SetPixelsFromCpuDstY0Frac;
    LwU32 SetPixelsFromCpuDstY0Int;
    LwU32 PixelsFromCpuData;
    LwU32 Reserved_0x864[0x3];
    LwU32 SetBigEndianControl;
    LwU32 Reserved_0x874[0x3];
    LwU32 SetPixelsFromMemoryBlockShape;
    LwU32 SetPixelsFromMemoryCorralSize;
    LwU32 SetPixelsFromMemorySafeOverlap;
    LwU32 SetPixelsFromMemorySampleMode;
    LwU32 Reserved_0x890[0x8];
    LwU32 SetPixelsFromMemoryDstX0;
    LwU32 SetPixelsFromMemoryDstY0;
    LwU32 SetPixelsFromMemoryDstWidth;
    LwU32 SetPixelsFromMemoryDstHeight;
    LwU32 SetPixelsFromMemoryDuDxFrac;
    LwU32 SetPixelsFromMemoryDuDxInt;
    LwU32 SetPixelsFromMemoryDvDyFrac;
    LwU32 SetPixelsFromMemoryDvDyInt;
    LwU32 SetPixelsFromMemorySrcX0Frac;
    LwU32 SetPixelsFromMemorySrcX0Int;
    LwU32 SetPixelsFromMemorySrcY0Frac;
    LwU32 PixelsFromMemorySrcY0Int;
} lw50_twod_t;


#define LW502D_SET_OBJECT                                                                                  0x0000
#define LW502D_SET_OBJECT_POINTER                                                                            15:0

#define LW502D_NO_OPERATION                                                                                0x0100
#define LW502D_NO_OPERATION_V                                                                                31:0

#define LW502D_NOTIFY                                                                                      0x0104
#define LW502D_NOTIFY_TYPE                                                                                   31:0
#define LW502D_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define LW502D_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define LW502D_WAIT_FOR_IDLE                                                                               0x0110
#define LW502D_WAIT_FOR_IDLE_V                                                                               31:0

#define LW502D_PM_TRIGGER                                                                                  0x0140
#define LW502D_PM_TRIGGER_V                                                                                  31:0

#define LW502D_SET_CONTEXT_DMA_NOTIFY                                                                      0x0180
#define LW502D_SET_CONTEXT_DMA_NOTIFY_HANDLE                                                                 31:0

#define LW502D_SET_DST_CONTEXT_DMA                                                                         0x0184
#define LW502D_SET_DST_CONTEXT_DMA_HANDLE                                                                    31:0

#define LW502D_SET_SRC_CONTEXT_DMA                                                                         0x0188
#define LW502D_SET_SRC_CONTEXT_DMA_HANDLE                                                                    31:0

#define LW502D_SET_SEMAPHORE_CONTEXT_DMA                                                                   0x018c
#define LW502D_SET_SEMAPHORE_CONTEXT_DMA_HANDLE                                                              31:0

#define LW502D_SET_DST_FORMAT                                                                              0x0200
#define LW502D_SET_DST_FORMAT_V                                                                               7:0
#define LW502D_SET_DST_FORMAT_V_A8R8G8B8                                                               0x000000CF
#define LW502D_SET_DST_FORMAT_V_A8RL8GL8BL8                                                            0x000000D0
#define LW502D_SET_DST_FORMAT_V_A2R10G10B10                                                            0x000000DF
#define LW502D_SET_DST_FORMAT_V_A8B8G8R8                                                               0x000000D5
#define LW502D_SET_DST_FORMAT_V_A8BL8GL8RL8                                                            0x000000D6
#define LW502D_SET_DST_FORMAT_V_A2B10G10R10                                                            0x000000D1
#define LW502D_SET_DST_FORMAT_V_X8R8G8B8                                                               0x000000E6
#define LW502D_SET_DST_FORMAT_V_X8RL8GL8BL8                                                            0x000000E7
#define LW502D_SET_DST_FORMAT_V_X8B8G8R8                                                               0x000000F9
#define LW502D_SET_DST_FORMAT_V_X8BL8GL8RL8                                                            0x000000FA
#define LW502D_SET_DST_FORMAT_V_R5G6B5                                                                 0x000000E8
#define LW502D_SET_DST_FORMAT_V_A1R5G5B5                                                               0x000000E9
#define LW502D_SET_DST_FORMAT_V_X1R5G5B5                                                               0x000000F8
#define LW502D_SET_DST_FORMAT_V_Y8                                                                     0x000000F3
#define LW502D_SET_DST_FORMAT_V_Y16                                                                    0x000000EE
#define LW502D_SET_DST_FORMAT_V_Y32                                                                    0x000000FF
#define LW502D_SET_DST_FORMAT_V_Z1R5G5B5                                                               0x000000FB
#define LW502D_SET_DST_FORMAT_V_O1R5G5B5                                                               0x000000FC
#define LW502D_SET_DST_FORMAT_V_Z8R8G8B8                                                               0x000000FD
#define LW502D_SET_DST_FORMAT_V_O8R8G8B8                                                               0x000000FE
#define LW502D_SET_DST_FORMAT_V_Y1_8X8                                                                 0x0000001C
#define LW502D_SET_DST_FORMAT_V_RF16                                                                   0x000000F2
#define LW502D_SET_DST_FORMAT_V_RF32                                                                   0x000000E5
#define LW502D_SET_DST_FORMAT_V_RF32_GF32                                                              0x000000CB
#define LW502D_SET_DST_FORMAT_V_RF16_GF16_BF16_AF16                                                    0x000000CA
#define LW502D_SET_DST_FORMAT_V_RF16_GF16_BF16_X16                                                     0x000000CE
#define LW502D_SET_DST_FORMAT_V_RF32_GF32_BF32_AF32                                                    0x000000C0
#define LW502D_SET_DST_FORMAT_V_RF32_GF32_BF32_X32                                                     0x000000C3

#define LW502D_SET_DST_MEMORY_LAYOUT                                                                       0x0204
#define LW502D_SET_DST_MEMORY_LAYOUT_V                                                                        0:0
#define LW502D_SET_DST_MEMORY_LAYOUT_V_BLOCKLINEAR                                                     0x00000000
#define LW502D_SET_DST_MEMORY_LAYOUT_V_PITCH                                                           0x00000001

#define LW502D_SET_DST_BLOCK_SIZE                                                                          0x0208
#define LW502D_SET_DST_BLOCK_SIZE_WIDTH                                                                       3:0
#define LW502D_SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB                                                        0x00000000
#define LW502D_SET_DST_BLOCK_SIZE_HEIGHT                                                                      7:4
#define LW502D_SET_DST_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LW502D_SET_DST_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LW502D_SET_DST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LW502D_SET_DST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LW502D_SET_DST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LW502D_SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LW502D_SET_DST_BLOCK_SIZE_DEPTH                                                                      11:8
#define LW502D_SET_DST_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LW502D_SET_DST_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LW502D_SET_DST_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LW502D_SET_DST_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LW502D_SET_DST_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LW502D_SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LW502D_SET_DST_DEPTH                                                                               0x020c
#define LW502D_SET_DST_DEPTH_V                                                                               31:0

#define LW502D_SET_DST_LAYER                                                                               0x0210
#define LW502D_SET_DST_LAYER_V                                                                               31:0

#define LW502D_SET_DST_PITCH                                                                               0x0214
#define LW502D_SET_DST_PITCH_V                                                                               31:0

#define LW502D_SET_DST_WIDTH                                                                               0x0218
#define LW502D_SET_DST_WIDTH_V                                                                               31:0

#define LW502D_SET_DST_HEIGHT                                                                              0x021c
#define LW502D_SET_DST_HEIGHT_V                                                                              31:0

#define LW502D_SET_DST_OFFSET_UPPER                                                                        0x0220
#define LW502D_SET_DST_OFFSET_UPPER_V                                                                         7:0

#define LW502D_SET_DST_OFFSET_LOWER                                                                        0x0224
#define LW502D_SET_DST_OFFSET_LOWER_V                                                                        31:0

#define LW502D_SET_PIXELS_FROM_CPU_INDEX_WRAP                                                              0x0228
#define LW502D_SET_PIXELS_FROM_CPU_INDEX_WRAP_V                                                               0:0
#define LW502D_SET_PIXELS_FROM_CPU_INDEX_WRAP_V_WRAP                                                   0x00000000
#define LW502D_SET_PIXELS_FROM_CPU_INDEX_WRAP_V_NO_WRAP                                                0x00000001

#define LW502D_SET_KIND2D_CHECK_ENABLE                                                                     0x022c
#define LW502D_SET_KIND2D_CHECK_ENABLE_V                                                                      0:0
#define LW502D_SET_KIND2D_CHECK_ENABLE_V_FALSE                                                         0x00000000
#define LW502D_SET_KIND2D_CHECK_ENABLE_V_TRUE                                                          0x00000001

#define LW502D_SET_SRC_FORMAT                                                                              0x0230
#define LW502D_SET_SRC_FORMAT_V                                                                               7:0
#define LW502D_SET_SRC_FORMAT_V_A8R8G8B8                                                               0x000000CF
#define LW502D_SET_SRC_FORMAT_V_A8RL8GL8BL8                                                            0x000000D0
#define LW502D_SET_SRC_FORMAT_V_A2R10G10B10                                                            0x000000DF
#define LW502D_SET_SRC_FORMAT_V_A8B8G8R8                                                               0x000000D5
#define LW502D_SET_SRC_FORMAT_V_A8BL8GL8RL8                                                            0x000000D6
#define LW502D_SET_SRC_FORMAT_V_A2B10G10R10                                                            0x000000D1
#define LW502D_SET_SRC_FORMAT_V_X8R8G8B8                                                               0x000000E6
#define LW502D_SET_SRC_FORMAT_V_X8RL8GL8BL8                                                            0x000000E7
#define LW502D_SET_SRC_FORMAT_V_X8B8G8R8                                                               0x000000F9
#define LW502D_SET_SRC_FORMAT_V_X8BL8GL8RL8                                                            0x000000FA
#define LW502D_SET_SRC_FORMAT_V_R5G6B5                                                                 0x000000E8
#define LW502D_SET_SRC_FORMAT_V_A1R5G5B5                                                               0x000000E9
#define LW502D_SET_SRC_FORMAT_V_X1R5G5B5                                                               0x000000F8
#define LW502D_SET_SRC_FORMAT_V_Y8                                                                     0x000000F3
#define LW502D_SET_SRC_FORMAT_V_AY8                                                                    0x0000001D
#define LW502D_SET_SRC_FORMAT_V_Y16                                                                    0x000000EE
#define LW502D_SET_SRC_FORMAT_V_Y32                                                                    0x000000FF
#define LW502D_SET_SRC_FORMAT_V_Z1R5G5B5                                                               0x000000FB
#define LW502D_SET_SRC_FORMAT_V_O1R5G5B5                                                               0x000000FC
#define LW502D_SET_SRC_FORMAT_V_Z8R8G8B8                                                               0x000000FD
#define LW502D_SET_SRC_FORMAT_V_O8R8G8B8                                                               0x000000FE
#define LW502D_SET_SRC_FORMAT_V_Y1_8X8                                                                 0x0000001C
#define LW502D_SET_SRC_FORMAT_V_RF16                                                                   0x000000F2
#define LW502D_SET_SRC_FORMAT_V_RF32                                                                   0x000000E5
#define LW502D_SET_SRC_FORMAT_V_RF32_GF32                                                              0x000000CB
#define LW502D_SET_SRC_FORMAT_V_RF16_GF16_BF16_AF16                                                    0x000000CA
#define LW502D_SET_SRC_FORMAT_V_RF16_GF16_BF16_X16                                                     0x000000CE
#define LW502D_SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32                                                    0x000000C0
#define LW502D_SET_SRC_FORMAT_V_RF32_GF32_BF32_X32                                                     0x000000C3

#define LW502D_SET_SRC_MEMORY_LAYOUT                                                                       0x0234
#define LW502D_SET_SRC_MEMORY_LAYOUT_V                                                                        0:0
#define LW502D_SET_SRC_MEMORY_LAYOUT_V_BLOCKLINEAR                                                     0x00000000
#define LW502D_SET_SRC_MEMORY_LAYOUT_V_PITCH                                                           0x00000001

#define LW502D_SET_SRC_BLOCK_SIZE                                                                          0x0238
#define LW502D_SET_SRC_BLOCK_SIZE_WIDTH                                                                       3:0
#define LW502D_SET_SRC_BLOCK_SIZE_WIDTH_ONE_GOB                                                        0x00000000
#define LW502D_SET_SRC_BLOCK_SIZE_HEIGHT                                                                      7:4
#define LW502D_SET_SRC_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LW502D_SET_SRC_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LW502D_SET_SRC_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LW502D_SET_SRC_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LW502D_SET_SRC_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LW502D_SET_SRC_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LW502D_SET_SRC_BLOCK_SIZE_DEPTH                                                                      11:8
#define LW502D_SET_SRC_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LW502D_SET_SRC_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LW502D_SET_SRC_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LW502D_SET_SRC_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LW502D_SET_SRC_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LW502D_SET_SRC_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LW502D_SET_SRC_DEPTH                                                                               0x023c
#define LW502D_SET_SRC_DEPTH_V                                                                               31:0

#define LW502D_SET_SRC_LAYER                                                                               0x0240
#define LW502D_SET_SRC_LAYER_V                                                                               31:0

#define LW502D_SET_SRC_PITCH                                                                               0x0244
#define LW502D_SET_SRC_PITCH_V                                                                               31:0

#define LW502D_SET_SRC_WIDTH                                                                               0x0248
#define LW502D_SET_SRC_WIDTH_V                                                                               31:0

#define LW502D_SET_SRC_HEIGHT                                                                              0x024c
#define LW502D_SET_SRC_HEIGHT_V                                                                              31:0

#define LW502D_SET_SRC_OFFSET_UPPER                                                                        0x0250
#define LW502D_SET_SRC_OFFSET_UPPER_V                                                                         7:0

#define LW502D_SET_SRC_OFFSET_LOWER                                                                        0x0254
#define LW502D_SET_SRC_OFFSET_LOWER_V                                                                        31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION                                                     0x0258
#define LW502D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V                                                      1:0
#define LW502D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_NO_PROMOTION                                  0x00000000
#define LW502D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_PROMOTE_TO_2_V                                0x00000001
#define LW502D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_PROMOTE_TO_2_H                                0x00000002
#define LW502D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_PROMOTE_TO_4                                  0x00000003

#define LW502D_SET_NUM_TPCS                                                                                0x0260
#define LW502D_SET_NUM_TPCS_V                                                                                 0:0
#define LW502D_SET_NUM_TPCS_V_ALL                                                                      0x00000000
#define LW502D_SET_NUM_TPCS_V_ONE                                                                      0x00000001

#define LW502D_SET_RENDER_ENABLE_A                                                                         0x0264
#define LW502D_SET_RENDER_ENABLE_A_OFFSET_UPPER                                                               7:0

#define LW502D_SET_RENDER_ENABLE_B                                                                         0x0268
#define LW502D_SET_RENDER_ENABLE_B_OFFSET_LOWER                                                              31:0

#define LW502D_SET_RENDER_ENABLE_C                                                                         0x026c
#define LW502D_SET_RENDER_ENABLE_C_MODE                                                                       2:0
#define LW502D_SET_RENDER_ENABLE_C_MODE_FALSE                                                          0x00000000
#define LW502D_SET_RENDER_ENABLE_C_MODE_TRUE                                                           0x00000001
#define LW502D_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                                                    0x00000002
#define LW502D_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                                0x00000003
#define LW502D_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                            0x00000004

#define LW502D_SET_CLIP_X0                                                                                 0x0280
#define LW502D_SET_CLIP_X0_V                                                                                 31:0

#define LW502D_SET_CLIP_Y0                                                                                 0x0284
#define LW502D_SET_CLIP_Y0_V                                                                                 31:0

#define LW502D_SET_CLIP_WIDTH                                                                              0x0288
#define LW502D_SET_CLIP_WIDTH_V                                                                              31:0

#define LW502D_SET_CLIP_HEIGHT                                                                             0x028c
#define LW502D_SET_CLIP_HEIGHT_V                                                                             31:0

#define LW502D_SET_CLIP_ENABLE                                                                             0x0290
#define LW502D_SET_CLIP_ENABLE_V                                                                              0:0
#define LW502D_SET_CLIP_ENABLE_V_FALSE                                                                 0x00000000
#define LW502D_SET_CLIP_ENABLE_V_TRUE                                                                  0x00000001

#define LW502D_SET_COLOR_KEY_FORMAT                                                                        0x0294
#define LW502D_SET_COLOR_KEY_FORMAT_V                                                                         2:0
#define LW502D_SET_COLOR_KEY_FORMAT_V_A16R5G6B5                                                        0x00000000
#define LW502D_SET_COLOR_KEY_FORMAT_V_A1R5G5B5                                                         0x00000001
#define LW502D_SET_COLOR_KEY_FORMAT_V_A8R8G8B8                                                         0x00000002
#define LW502D_SET_COLOR_KEY_FORMAT_V_A2R10G10B10                                                      0x00000003
#define LW502D_SET_COLOR_KEY_FORMAT_V_Y8                                                               0x00000004
#define LW502D_SET_COLOR_KEY_FORMAT_V_Y16                                                              0x00000005
#define LW502D_SET_COLOR_KEY_FORMAT_V_Y32                                                              0x00000006

#define LW502D_SET_COLOR_KEY                                                                               0x0298
#define LW502D_SET_COLOR_KEY_V                                                                               31:0

#define LW502D_SET_COLOR_KEY_ENABLE                                                                        0x029c
#define LW502D_SET_COLOR_KEY_ENABLE_V                                                                         0:0
#define LW502D_SET_COLOR_KEY_ENABLE_V_FALSE                                                            0x00000000
#define LW502D_SET_COLOR_KEY_ENABLE_V_TRUE                                                             0x00000001

#define LW502D_SET_ROP                                                                                     0x02a0
#define LW502D_SET_ROP_V                                                                                      7:0

#define LW502D_SET_BETA1                                                                                   0x02a4
#define LW502D_SET_BETA1_V                                                                                   31:0

#define LW502D_SET_BETA4                                                                                   0x02a8
#define LW502D_SET_BETA4_B                                                                                    7:0
#define LW502D_SET_BETA4_G                                                                                   15:8
#define LW502D_SET_BETA4_R                                                                                  23:16
#define LW502D_SET_BETA4_A                                                                                  31:24

#define LW502D_SET_OPERATION                                                                               0x02ac
#define LW502D_SET_OPERATION_V                                                                                2:0
#define LW502D_SET_OPERATION_V_SRCCOPY_AND                                                             0x00000000
#define LW502D_SET_OPERATION_V_ROP_AND                                                                 0x00000001
#define LW502D_SET_OPERATION_V_BLEND_AND                                                               0x00000002
#define LW502D_SET_OPERATION_V_SRCCOPY                                                                 0x00000003
#define LW502D_SET_OPERATION_V_ROP                                                                     0x00000004
#define LW502D_SET_OPERATION_V_SRCCOPY_PREMULT                                                         0x00000005
#define LW502D_SET_OPERATION_V_BLEND_PREMULT                                                           0x00000006

#define LW502D_SET_PATTERN_OFFSET                                                                          0x02b0
#define LW502D_SET_PATTERN_OFFSET_X                                                                           5:0
#define LW502D_SET_PATTERN_OFFSET_Y                                                                          13:8

#define LW502D_SET_PATTERN_SELECT                                                                          0x02b4
#define LW502D_SET_PATTERN_SELECT_V                                                                           1:0
#define LW502D_SET_PATTERN_SELECT_V_MONOCHROME_8x8                                                     0x00000000
#define LW502D_SET_PATTERN_SELECT_V_MONOCHROME_64x1                                                    0x00000001
#define LW502D_SET_PATTERN_SELECT_V_MONOCHROME_1x64                                                    0x00000002
#define LW502D_SET_PATTERN_SELECT_V_COLOR                                                              0x00000003

#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT                                                         0x02e8
#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V                                                          2:0
#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8X8R5G6B5                                        0x00000000
#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A1R5G5B5                                          0x00000001
#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8R8G8B8                                          0x00000002
#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8Y8                                              0x00000003
#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8X8Y16                                           0x00000004
#define LW502D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_Y32                                               0x00000005

#define LW502D_SET_MONOCHROME_PATTERN_FORMAT                                                               0x02ec
#define LW502D_SET_MONOCHROME_PATTERN_FORMAT_V                                                                0:0
#define LW502D_SET_MONOCHROME_PATTERN_FORMAT_V_CGA6_M1                                                 0x00000000
#define LW502D_SET_MONOCHROME_PATTERN_FORMAT_V_LE_M1                                                   0x00000001

#define LW502D_SET_MONOCHROME_PATTERN_COLOR0                                                               0x02f0
#define LW502D_SET_MONOCHROME_PATTERN_COLOR0_V                                                               31:0

#define LW502D_SET_MONOCHROME_PATTERN_COLOR1                                                               0x02f4
#define LW502D_SET_MONOCHROME_PATTERN_COLOR1_V                                                               31:0

#define LW502D_SET_MONOCHROME_PATTERN0                                                                     0x02f8
#define LW502D_SET_MONOCHROME_PATTERN0_V                                                                     31:0

#define LW502D_SET_MONOCHROME_PATTERN1                                                                     0x02fc
#define LW502D_SET_MONOCHROME_PATTERN1_V                                                                     31:0

#define LW502D_COLOR_PATTERN_X8R8G8B8(i)                                                           (0x0300+(i)*4)
#define LW502D_COLOR_PATTERN_X8R8G8B8_B0                                                                      7:0
#define LW502D_COLOR_PATTERN_X8R8G8B8_G0                                                                     15:8
#define LW502D_COLOR_PATTERN_X8R8G8B8_R0                                                                    23:16
#define LW502D_COLOR_PATTERN_X8R8G8B8_IGNORE0                                                               31:24

#define LW502D_COLOR_PATTERN_R5G6B5(i)                                                             (0x0400+(i)*4)
#define LW502D_COLOR_PATTERN_R5G6B5_B0                                                                        4:0
#define LW502D_COLOR_PATTERN_R5G6B5_G0                                                                       10:5
#define LW502D_COLOR_PATTERN_R5G6B5_R0                                                                      15:11
#define LW502D_COLOR_PATTERN_R5G6B5_B1                                                                      20:16
#define LW502D_COLOR_PATTERN_R5G6B5_G1                                                                      26:21
#define LW502D_COLOR_PATTERN_R5G6B5_R1                                                                      31:27

#define LW502D_COLOR_PATTERN_X1R5G5B5(i)                                                           (0x0480+(i)*4)
#define LW502D_COLOR_PATTERN_X1R5G5B5_B0                                                                      4:0
#define LW502D_COLOR_PATTERN_X1R5G5B5_G0                                                                      9:5
#define LW502D_COLOR_PATTERN_X1R5G5B5_R0                                                                    14:10
#define LW502D_COLOR_PATTERN_X1R5G5B5_IGNORE0                                                               15:15
#define LW502D_COLOR_PATTERN_X1R5G5B5_B1                                                                    20:16
#define LW502D_COLOR_PATTERN_X1R5G5B5_G1                                                                    25:21
#define LW502D_COLOR_PATTERN_X1R5G5B5_R1                                                                    30:26
#define LW502D_COLOR_PATTERN_X1R5G5B5_IGNORE1                                                               31:31

#define LW502D_COLOR_PATTERN_Y8(i)                                                                 (0x0500+(i)*4)
#define LW502D_COLOR_PATTERN_Y8_Y0                                                                            7:0
#define LW502D_COLOR_PATTERN_Y8_Y1                                                                           15:8
#define LW502D_COLOR_PATTERN_Y8_Y2                                                                          23:16
#define LW502D_COLOR_PATTERN_Y8_Y3                                                                          31:24

#define LW502D_RENDER_SOLID_PRIM_MODE                                                                      0x0580
#define LW502D_RENDER_SOLID_PRIM_MODE_V                                                                       2:0
#define LW502D_RENDER_SOLID_PRIM_MODE_V_POINTS                                                         0x00000000
#define LW502D_RENDER_SOLID_PRIM_MODE_V_LINES                                                          0x00000001
#define LW502D_RENDER_SOLID_PRIM_MODE_V_POLYLINE                                                       0x00000002
#define LW502D_RENDER_SOLID_PRIM_MODE_V_TRIANGLES                                                      0x00000003
#define LW502D_RENDER_SOLID_PRIM_MODE_V_RECTS                                                          0x00000004

#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT                                                          0x0584
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V                                                           7:0
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A8R8G8B8                                           0x000000CF
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A2R10G10B10                                        0x000000DF
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A8B8G8R8                                           0x000000D5
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A2B10G10R10                                        0x000000D1
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_X8R8G8B8                                           0x000000E6
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_X8B8G8R8                                           0x000000F9
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_R5G6B5                                             0x000000E8
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A1R5G5B5                                           0x000000E9
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_X1R5G5B5                                           0x000000F8
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Y8                                                 0x000000F3
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Y16                                                0x000000EE
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Y32                                                0x000000FF
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Z1R5G5B5                                           0x000000FB
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_O1R5G5B5                                           0x000000FC
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Z8R8G8B8                                           0x000000FD
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_O8R8G8B8                                           0x000000FE

#define LW502D_SET_RENDER_SOLID_PRIM_COLOR                                                                 0x0588
#define LW502D_SET_RENDER_SOLID_PRIM_COLOR_V                                                                 31:0

#define LW502D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS                                                        0x058c
#define LW502D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_XMAJ__XINC__YINC                                          0:0
#define LW502D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_XMAJ__XDEC__YINC                                          4:4
#define LW502D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_YMAJ__XINC__YINC                                          8:8
#define LW502D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_YMAJ__XDEC__YINC                                        12:12

#define LW502D_RENDER_SOLID_PRIM_POINT_X_Y                                                                 0x05e0
#define LW502D_RENDER_SOLID_PRIM_POINT_X_Y_X                                                                 15:0
#define LW502D_RENDER_SOLID_PRIM_POINT_X_Y_Y                                                                31:16

#define LW502D_RENDER_SOLID_PRIM_POINT_SET_X(j)                                                    (0x0600+(j)*8)
#define LW502D_RENDER_SOLID_PRIM_POINT_SET_X_V                                                               31:0

#define LW502D_RENDER_SOLID_PRIM_POINT_Y(j)                                                        (0x0604+(j)*8)
#define LW502D_RENDER_SOLID_PRIM_POINT_Y_V                                                                   31:0

#define LW502D_SET_PIXELS_FROM_CPU_DATA_TYPE                                                               0x0800
#define LW502D_SET_PIXELS_FROM_CPU_DATA_TYPE_V                                                                0:0
#define LW502D_SET_PIXELS_FROM_CPU_DATA_TYPE_V_COLOR                                                   0x00000000
#define LW502D_SET_PIXELS_FROM_CPU_DATA_TYPE_V_INDEX                                                   0x00000001

#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT                                                            0x0804
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V                                                             7:0
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A8R8G8B8                                             0x000000CF
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A2R10G10B10                                          0x000000DF
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A8B8G8R8                                             0x000000D5
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A2B10G10R10                                          0x000000D1
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_X8R8G8B8                                             0x000000E6
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_X8B8G8R8                                             0x000000F9
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_R5G6B5                                               0x000000E8
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A1R5G5B5                                             0x000000E9
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_X1R5G5B5                                             0x000000F8
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Y8                                                   0x000000F3
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Y16                                                  0x000000EE
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Y32                                                  0x000000FF
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Z1R5G5B5                                             0x000000FB
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_O1R5G5B5                                             0x000000FC
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Z8R8G8B8                                             0x000000FD
#define LW502D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_O8R8G8B8                                             0x000000FE

#define LW502D_SET_PIXELS_FROM_CPU_INDEX_FORMAT                                                            0x0808
#define LW502D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V                                                             1:0
#define LW502D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V_I1                                                   0x00000000
#define LW502D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V_I4                                                   0x00000001
#define LW502D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V_I8                                                   0x00000002

#define LW502D_SET_PIXELS_FROM_CPU_MONO_FORMAT                                                             0x080c
#define LW502D_SET_PIXELS_FROM_CPU_MONO_FORMAT_V                                                              0:0
#define LW502D_SET_PIXELS_FROM_CPU_MONO_FORMAT_V_CGA6_M1                                               0x00000000
#define LW502D_SET_PIXELS_FROM_CPU_MONO_FORMAT_V_LE_M1                                                 0x00000001

#define LW502D_SET_PIXELS_FROM_CPU_WRAP                                                                    0x0810
#define LW502D_SET_PIXELS_FROM_CPU_WRAP_V                                                                     1:0
#define LW502D_SET_PIXELS_FROM_CPU_WRAP_V_WRAP_PIXEL                                                   0x00000000
#define LW502D_SET_PIXELS_FROM_CPU_WRAP_V_WRAP_BYTE                                                    0x00000001
#define LW502D_SET_PIXELS_FROM_CPU_WRAP_V_WRAP_DWORD                                                   0x00000002

#define LW502D_SET_PIXELS_FROM_CPU_COLOR0                                                                  0x0814
#define LW502D_SET_PIXELS_FROM_CPU_COLOR0_V                                                                  31:0

#define LW502D_SET_PIXELS_FROM_CPU_COLOR1                                                                  0x0818
#define LW502D_SET_PIXELS_FROM_CPU_COLOR1_V                                                                  31:0

#define LW502D_SET_PIXELS_FROM_CPU_MONO_OPACITY                                                            0x081c
#define LW502D_SET_PIXELS_FROM_CPU_MONO_OPACITY_V                                                             0:0
#define LW502D_SET_PIXELS_FROM_CPU_MONO_OPACITY_V_TRANSPARENT                                          0x00000000
#define LW502D_SET_PIXELS_FROM_CPU_MONO_OPACITY_V_OPAQUE                                               0x00000001

#define LW502D_SET_PIXELS_FROM_CPU_SRC_WIDTH                                                               0x0838
#define LW502D_SET_PIXELS_FROM_CPU_SRC_WIDTH_V                                                               31:0

#define LW502D_SET_PIXELS_FROM_CPU_SRC_HEIGHT                                                              0x083c
#define LW502D_SET_PIXELS_FROM_CPU_SRC_HEIGHT_V                                                              31:0

#define LW502D_SET_PIXELS_FROM_CPU_DX_DU_FRAC                                                              0x0840
#define LW502D_SET_PIXELS_FROM_CPU_DX_DU_FRAC_V                                                              31:0

#define LW502D_SET_PIXELS_FROM_CPU_DX_DU_INT                                                               0x0844
#define LW502D_SET_PIXELS_FROM_CPU_DX_DU_INT_V                                                               31:0

#define LW502D_SET_PIXELS_FROM_CPU_DY_DV_FRAC                                                              0x0848
#define LW502D_SET_PIXELS_FROM_CPU_DY_DV_FRAC_V                                                              31:0

#define LW502D_SET_PIXELS_FROM_CPU_DY_DV_INT                                                               0x084c
#define LW502D_SET_PIXELS_FROM_CPU_DY_DV_INT_V                                                               31:0

#define LW502D_SET_PIXELS_FROM_CPU_DST_X0_FRAC                                                             0x0850
#define LW502D_SET_PIXELS_FROM_CPU_DST_X0_FRAC_V                                                             31:0

#define LW502D_SET_PIXELS_FROM_CPU_DST_X0_INT                                                              0x0854
#define LW502D_SET_PIXELS_FROM_CPU_DST_X0_INT_V                                                              31:0

#define LW502D_SET_PIXELS_FROM_CPU_DST_Y0_FRAC                                                             0x0858
#define LW502D_SET_PIXELS_FROM_CPU_DST_Y0_FRAC_V                                                             31:0

#define LW502D_SET_PIXELS_FROM_CPU_DST_Y0_INT                                                              0x085c
#define LW502D_SET_PIXELS_FROM_CPU_DST_Y0_INT_V                                                              31:0

#define LW502D_PIXELS_FROM_CPU_DATA                                                                        0x0860
#define LW502D_PIXELS_FROM_CPU_DATA_V                                                                        31:0

#define LW502D_SET_BIG_ENDIAN_CONTROL                                                                      0x0870
#define LW502D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_1                                                              0:0
#define LW502D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_4                                                              1:1
#define LW502D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_8                                                              2:2
#define LW502D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_16                                                             3:3
#define LW502D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_1                                                              4:4
#define LW502D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_4                                                              5:5
#define LW502D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_8                                                              6:6
#define LW502D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_16                                                             7:7
#define LW502D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_1                                                               8:8
#define LW502D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_4                                                               9:9
#define LW502D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_8                                                             10:10
#define LW502D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_16                                                            11:11
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_1                                                     12:12
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_4                                                     13:13
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_8                                                     14:14
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_16                                                    15:15
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_1                                                       16:16
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_4                                                       17:17
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_8                                                       18:18
#define LW502D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_16                                                      19:19
#define LW502D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_1                                                             20:20
#define LW502D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_4                                                             21:21
#define LW502D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_8                                                             22:22
#define LW502D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_16                                                            23:23
#define LW502D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_1                                                             24:24
#define LW502D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_4                                                             25:25
#define LW502D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_8                                                             26:26
#define LW502D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_16                                                            27:27
#define LW502D_SET_BIG_ENDIAN_CONTROL_OVERRIDE                                                              28:28

#define LW502D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE                                                          0x0880
#define LW502D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V                                                           2:0
#define LW502D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V_AUTO                                               0x00000000
#define LW502D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V_SHAPE_8X4                                          0x00000001
#define LW502D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V_SHAPE_16X2                                         0x00000002

#define LW502D_SET_PIXELS_FROM_MEMORY_CORRAL_SIZE                                                          0x0884
#define LW502D_SET_PIXELS_FROM_MEMORY_CORRAL_SIZE_V                                                           5:0

#define LW502D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP                                                         0x0888
#define LW502D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP_V                                                          0:0
#define LW502D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP_V_FALSE                                             0x00000000
#define LW502D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP_V_TRUE                                              0x00000001

#define LW502D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE                                                          0x088c
#define LW502D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN                                                      0:0
#define LW502D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN_CENTER                                        0x00000000
#define LW502D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN_CORNER                                        0x00000001
#define LW502D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER                                                      4:4
#define LW502D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER_POINT                                         0x00000000
#define LW502D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER_BILINEAR                                      0x00000001

#define LW502D_SET_PIXELS_FROM_MEMORY_DST_X0                                                               0x08b0
#define LW502D_SET_PIXELS_FROM_MEMORY_DST_X0_V                                                               31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_DST_Y0                                                               0x08b4
#define LW502D_SET_PIXELS_FROM_MEMORY_DST_Y0_V                                                               31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_DST_WIDTH                                                            0x08b8
#define LW502D_SET_PIXELS_FROM_MEMORY_DST_WIDTH_V                                                            31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_DST_HEIGHT                                                           0x08bc
#define LW502D_SET_PIXELS_FROM_MEMORY_DST_HEIGHT_V                                                           31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_DU_DX_FRAC                                                           0x08c0
#define LW502D_SET_PIXELS_FROM_MEMORY_DU_DX_FRAC_V                                                           31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_DU_DX_INT                                                            0x08c4
#define LW502D_SET_PIXELS_FROM_MEMORY_DU_DX_INT_V                                                            31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_DV_DY_FRAC                                                           0x08c8
#define LW502D_SET_PIXELS_FROM_MEMORY_DV_DY_FRAC_V                                                           31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_DV_DY_INT                                                            0x08cc
#define LW502D_SET_PIXELS_FROM_MEMORY_DV_DY_INT_V                                                            31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC                                                          0x08d0
#define LW502D_SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC_V                                                          31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_SRC_X0_INT                                                           0x08d4
#define LW502D_SET_PIXELS_FROM_MEMORY_SRC_X0_INT_V                                                           31:0

#define LW502D_SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC                                                          0x08d8
#define LW502D_SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC_V                                                          31:0

#define LW502D_PIXELS_FROM_MEMORY_SRC_Y0_INT                                                               0x08dc
#define LW502D_PIXELS_FROM_MEMORY_SRC_Y0_INT_V                                                               31:0

#endif /* _cl_lw50_twod_h_ */
