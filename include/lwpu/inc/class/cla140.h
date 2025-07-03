/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef HWFE_KEPLER_INLINE_TO_MEMORY_B_SW_H
#define HWFE_KEPLER_INLINE_TO_MEMORY_B_SW_H

/* AUTO GENERATED FILE -- DO NOT EDIT */
/* Command: ../../../../class/bin/sw_header.pl kepler_inline_to_memory_b */

#include "lwtypes.h"

#define KEPLER_INLINE_TO_MEMORY_B    0xA140

typedef volatile struct kepler_inline_to_memory_b_struct {
    LwU32 SetObject;
    LwU32 Reserved_0x04[0x3F];
    LwU32 NoOperation;
    LwU32 SetNotifyA;
    LwU32 SetNotifyB;
    LwU32 Notify;
    LwU32 WaitForIdle;
    LwU32 Reserved_0x114[0x7];
    LwU32 SetGlobalRenderEnableA;
    LwU32 SetGlobalRenderEnableB;
    LwU32 SetGlobalRenderEnableC;
    LwU32 SendGoIdle;
    LwU32 PmTrigger;
    LwU32 PmTriggerWfi;
    LwU32 Reserved_0x148[0x2];
    LwU32 SetInstrumentationMethodHeader;
    LwU32 SetInstrumentationMethodData;
    LwU32 Reserved_0x158[0xA];
    LwU32 LineLengthIn;
    LwU32 LineCount;
    LwU32 OffsetOutUpper;
    LwU32 OffsetOut;
    LwU32 PitchOut;
    LwU32 SetDstBlockSize;
    LwU32 SetDstWidth;
    LwU32 SetDstHeight;
    LwU32 SetDstDepth;
    LwU32 SetDstLayer;
    LwU32 SetDstOriginBytesX;
    LwU32 SetDstOriginSamplesY;
    LwU32 LaunchDma;
    LwU32 LoadInlineData;
    LwU32 Reserved_0x1B8[0x9];
    LwU32 SetI2mSemaphoreA;
    LwU32 SetI2mSemaphoreB;
    LwU32 SetI2mSemaphoreC;
    LwU32 Reserved_0x1E8[0x2];
    LwU32 SetI2mSpareNoop00;
    LwU32 SetI2mSpareNoop01;
    LwU32 SetI2mSpareNoop02;
    LwU32 SetI2mSpareNoop03;
    LwU32 SetFalcon00;
    LwU32 SetFalcon01;
    LwU32 SetFalcon02;
    LwU32 SetFalcon03;
    LwU32 SetFalcon04;
    LwU32 SetFalcon05;
    LwU32 SetFalcon06;
    LwU32 SetFalcon07;
    LwU32 SetFalcon08;
    LwU32 SetFalcon09;
    LwU32 SetFalcon10;
    LwU32 SetFalcon11;
    LwU32 SetFalcon12;
    LwU32 SetFalcon13;
    LwU32 SetFalcon14;
    LwU32 SetFalcon15;
    LwU32 SetFalcon16;
    LwU32 SetFalcon17;
    LwU32 SetFalcon18;
    LwU32 SetFalcon19;
    LwU32 SetFalcon20;
    LwU32 SetFalcon21;
    LwU32 SetFalcon22;
    LwU32 SetFalcon23;
    LwU32 SetFalcon24;
    LwU32 SetFalcon25;
    LwU32 SetFalcon26;
    LwU32 SetFalcon27;
    LwU32 SetFalcon28;
    LwU32 SetFalcon29;
    LwU32 SetFalcon30;
    LwU32 SetFalcon31;
    LwU32 Reserved_0x280[0x4B4];
    LwU32 SetRenderEnableA;
    LwU32 SetRenderEnableB;
    LwU32 SetRenderEnableC;
    LwU32 Reserved_0x155C[0xFA];
    LwU32 SetRenderEnableOverride;
    LwU32 Reserved_0x1948[0x6AE];
    LwU32 SetMmeShadowScratch[0x8];
} kepler_inline_to_memory_b_t;


#define LWA140_SET_OBJECT                                                                                  0x0000
#define LWA140_SET_OBJECT_CLASS_ID                                                                           15:0
#define LWA140_SET_OBJECT_ENGINE_ID                                                                         20:16

#define LWA140_NO_OPERATION                                                                                0x0100
#define LWA140_NO_OPERATION_V                                                                                31:0

#define LWA140_SET_NOTIFY_A                                                                                0x0104
#define LWA140_SET_NOTIFY_A_ADDRESS_UPPER                                                                    24:0

#define LWA140_SET_NOTIFY_B                                                                                0x0108
#define LWA140_SET_NOTIFY_B_ADDRESS_LOWER                                                                    31:0

#define LWA140_NOTIFY                                                                                      0x010c
#define LWA140_NOTIFY_TYPE                                                                                   31:0
#define LWA140_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define LWA140_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define LWA140_WAIT_FOR_IDLE                                                                               0x0110
#define LWA140_WAIT_FOR_IDLE_V                                                                               31:0

#define LWA140_SET_GLOBAL_RENDER_ENABLE_A                                                                  0x0130
#define LWA140_SET_GLOBAL_RENDER_ENABLE_A_OFFSET_UPPER                                                        7:0

#define LWA140_SET_GLOBAL_RENDER_ENABLE_B                                                                  0x0134
#define LWA140_SET_GLOBAL_RENDER_ENABLE_B_OFFSET_LOWER                                                       31:0

#define LWA140_SET_GLOBAL_RENDER_ENABLE_C                                                                  0x0138
#define LWA140_SET_GLOBAL_RENDER_ENABLE_C_MODE                                                                2:0
#define LWA140_SET_GLOBAL_RENDER_ENABLE_C_MODE_FALSE                                                   0x00000000
#define LWA140_SET_GLOBAL_RENDER_ENABLE_C_MODE_TRUE                                                    0x00000001
#define LWA140_SET_GLOBAL_RENDER_ENABLE_C_MODE_CONDITIONAL                                             0x00000002
#define LWA140_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                         0x00000003
#define LWA140_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                     0x00000004

#define LWA140_SEND_GO_IDLE                                                                                0x013c
#define LWA140_SEND_GO_IDLE_V                                                                                31:0

#define LWA140_PM_TRIGGER                                                                                  0x0140
#define LWA140_PM_TRIGGER_V                                                                                  31:0

#define LWA140_PM_TRIGGER_WFI                                                                              0x0144
#define LWA140_PM_TRIGGER_WFI_V                                                                              31:0

#define LWA140_SET_INSTRUMENTATION_METHOD_HEADER                                                           0x0150
#define LWA140_SET_INSTRUMENTATION_METHOD_HEADER_V                                                           31:0

#define LWA140_SET_INSTRUMENTATION_METHOD_DATA                                                             0x0154
#define LWA140_SET_INSTRUMENTATION_METHOD_DATA_V                                                             31:0

#define LWA140_LINE_LENGTH_IN                                                                              0x0180
#define LWA140_LINE_LENGTH_IN_VALUE                                                                          31:0

#define LWA140_LINE_COUNT                                                                                  0x0184
#define LWA140_LINE_COUNT_VALUE                                                                              31:0

#define LWA140_OFFSET_OUT_UPPER                                                                            0x0188
#define LWA140_OFFSET_OUT_UPPER_VALUE                                                                        24:0

#define LWA140_OFFSET_OUT                                                                                  0x018c
#define LWA140_OFFSET_OUT_VALUE                                                                              31:0

#define LWA140_PITCH_OUT                                                                                   0x0190
#define LWA140_PITCH_OUT_VALUE                                                                               31:0

#define LWA140_SET_DST_BLOCK_SIZE                                                                          0x0194
#define LWA140_SET_DST_BLOCK_SIZE_WIDTH                                                                       3:0
#define LWA140_SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB                                                        0x00000000
#define LWA140_SET_DST_BLOCK_SIZE_HEIGHT                                                                      7:4
#define LWA140_SET_DST_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LWA140_SET_DST_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LWA140_SET_DST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LWA140_SET_DST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LWA140_SET_DST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LWA140_SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LWA140_SET_DST_BLOCK_SIZE_DEPTH                                                                      11:8
#define LWA140_SET_DST_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LWA140_SET_DST_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LWA140_SET_DST_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LWA140_SET_DST_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LWA140_SET_DST_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LWA140_SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LWA140_SET_DST_WIDTH                                                                               0x0198
#define LWA140_SET_DST_WIDTH_V                                                                               31:0

#define LWA140_SET_DST_HEIGHT                                                                              0x019c
#define LWA140_SET_DST_HEIGHT_V                                                                              31:0

#define LWA140_SET_DST_DEPTH                                                                               0x01a0
#define LWA140_SET_DST_DEPTH_V                                                                               31:0

#define LWA140_SET_DST_LAYER                                                                               0x01a4
#define LWA140_SET_DST_LAYER_V                                                                               31:0

#define LWA140_SET_DST_ORIGIN_BYTES_X                                                                      0x01a8
#define LWA140_SET_DST_ORIGIN_BYTES_X_V                                                                      20:0

#define LWA140_SET_DST_ORIGIN_SAMPLES_Y                                                                    0x01ac
#define LWA140_SET_DST_ORIGIN_SAMPLES_Y_V                                                                    16:0

#define LWA140_LAUNCH_DMA                                                                                  0x01b0
#define LWA140_LAUNCH_DMA_DST_MEMORY_LAYOUT                                                                   0:0
#define LWA140_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR                                                0x00000000
#define LWA140_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH                                                      0x00000001
#define LWA140_LAUNCH_DMA_COMPLETION_TYPE                                                                     5:4
#define LWA140_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_DISABLE                                                0x00000000
#define LWA140_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY                                                   0x00000001
#define LWA140_LAUNCH_DMA_COMPLETION_TYPE_RELEASE_SEMAPHORE                                            0x00000002
#define LWA140_LAUNCH_DMA_INTERRUPT_TYPE                                                                      9:8
#define LWA140_LAUNCH_DMA_INTERRUPT_TYPE_NONE                                                          0x00000000
#define LWA140_LAUNCH_DMA_INTERRUPT_TYPE_INTERRUPT                                                     0x00000001
#define LWA140_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE                                                             12:12
#define LWA140_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_FOUR_WORDS                                             0x00000000
#define LWA140_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_ONE_WORD                                               0x00000001
#define LWA140_LAUNCH_DMA_REDUCTION_ENABLE                                                                    1:1
#define LWA140_LAUNCH_DMA_REDUCTION_ENABLE_FALSE                                                       0x00000000
#define LWA140_LAUNCH_DMA_REDUCTION_ENABLE_TRUE                                                        0x00000001
#define LWA140_LAUNCH_DMA_REDUCTION_OP                                                                      15:13
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_ADD                                                         0x00000000
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_MIN                                                         0x00000001
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_MAX                                                         0x00000002
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_INC                                                         0x00000003
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_DEC                                                         0x00000004
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_AND                                                         0x00000005
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_OR                                                          0x00000006
#define LWA140_LAUNCH_DMA_REDUCTION_OP_RED_XOR                                                         0x00000007
#define LWA140_LAUNCH_DMA_REDUCTION_FORMAT                                                                    3:2
#define LWA140_LAUNCH_DMA_REDUCTION_FORMAT_UNSIGNED_32                                                 0x00000000
#define LWA140_LAUNCH_DMA_REDUCTION_FORMAT_SIGNED_32                                                   0x00000001
#define LWA140_LAUNCH_DMA_SYSMEMBAR_DISABLE                                                                   6:6
#define LWA140_LAUNCH_DMA_SYSMEMBAR_DISABLE_FALSE                                                      0x00000000
#define LWA140_LAUNCH_DMA_SYSMEMBAR_DISABLE_TRUE                                                       0x00000001

#define LWA140_LOAD_INLINE_DATA                                                                            0x01b4
#define LWA140_LOAD_INLINE_DATA_V                                                                            31:0

#define LWA140_SET_I2M_SEMAPHORE_A                                                                         0x01dc
#define LWA140_SET_I2M_SEMAPHORE_A_OFFSET_UPPER                                                              24:0

#define LWA140_SET_I2M_SEMAPHORE_B                                                                         0x01e0
#define LWA140_SET_I2M_SEMAPHORE_B_OFFSET_LOWER                                                              31:0

#define LWA140_SET_I2M_SEMAPHORE_C                                                                         0x01e4
#define LWA140_SET_I2M_SEMAPHORE_C_PAYLOAD                                                                   31:0

#define LWA140_SET_I2M_SPARE_NOOP00                                                                        0x01f0
#define LWA140_SET_I2M_SPARE_NOOP00_V                                                                        31:0

#define LWA140_SET_I2M_SPARE_NOOP01                                                                        0x01f4
#define LWA140_SET_I2M_SPARE_NOOP01_V                                                                        31:0

#define LWA140_SET_I2M_SPARE_NOOP02                                                                        0x01f8
#define LWA140_SET_I2M_SPARE_NOOP02_V                                                                        31:0

#define LWA140_SET_I2M_SPARE_NOOP03                                                                        0x01fc
#define LWA140_SET_I2M_SPARE_NOOP03_V                                                                        31:0

#define LWA140_SET_FALCON00                                                                                0x0200
#define LWA140_SET_FALCON00_V                                                                                31:0

#define LWA140_SET_FALCON01                                                                                0x0204
#define LWA140_SET_FALCON01_V                                                                                31:0

#define LWA140_SET_FALCON02                                                                                0x0208
#define LWA140_SET_FALCON02_V                                                                                31:0

#define LWA140_SET_FALCON03                                                                                0x020c
#define LWA140_SET_FALCON03_V                                                                                31:0

#define LWA140_SET_FALCON04                                                                                0x0210
#define LWA140_SET_FALCON04_V                                                                                31:0

#define LWA140_SET_FALCON05                                                                                0x0214
#define LWA140_SET_FALCON05_V                                                                                31:0

#define LWA140_SET_FALCON06                                                                                0x0218
#define LWA140_SET_FALCON06_V                                                                                31:0

#define LWA140_SET_FALCON07                                                                                0x021c
#define LWA140_SET_FALCON07_V                                                                                31:0

#define LWA140_SET_FALCON08                                                                                0x0220
#define LWA140_SET_FALCON08_V                                                                                31:0

#define LWA140_SET_FALCON09                                                                                0x0224
#define LWA140_SET_FALCON09_V                                                                                31:0

#define LWA140_SET_FALCON10                                                                                0x0228
#define LWA140_SET_FALCON10_V                                                                                31:0

#define LWA140_SET_FALCON11                                                                                0x022c
#define LWA140_SET_FALCON11_V                                                                                31:0

#define LWA140_SET_FALCON12                                                                                0x0230
#define LWA140_SET_FALCON12_V                                                                                31:0

#define LWA140_SET_FALCON13                                                                                0x0234
#define LWA140_SET_FALCON13_V                                                                                31:0

#define LWA140_SET_FALCON14                                                                                0x0238
#define LWA140_SET_FALCON14_V                                                                                31:0

#define LWA140_SET_FALCON15                                                                                0x023c
#define LWA140_SET_FALCON15_V                                                                                31:0

#define LWA140_SET_FALCON16                                                                                0x0240
#define LWA140_SET_FALCON16_V                                                                                31:0

#define LWA140_SET_FALCON17                                                                                0x0244
#define LWA140_SET_FALCON17_V                                                                                31:0

#define LWA140_SET_FALCON18                                                                                0x0248
#define LWA140_SET_FALCON18_V                                                                                31:0

#define LWA140_SET_FALCON19                                                                                0x024c
#define LWA140_SET_FALCON19_V                                                                                31:0

#define LWA140_SET_FALCON20                                                                                0x0250
#define LWA140_SET_FALCON20_V                                                                                31:0

#define LWA140_SET_FALCON21                                                                                0x0254
#define LWA140_SET_FALCON21_V                                                                                31:0

#define LWA140_SET_FALCON22                                                                                0x0258
#define LWA140_SET_FALCON22_V                                                                                31:0

#define LWA140_SET_FALCON23                                                                                0x025c
#define LWA140_SET_FALCON23_V                                                                                31:0

#define LWA140_SET_FALCON24                                                                                0x0260
#define LWA140_SET_FALCON24_V                                                                                31:0

#define LWA140_SET_FALCON25                                                                                0x0264
#define LWA140_SET_FALCON25_V                                                                                31:0

#define LWA140_SET_FALCON26                                                                                0x0268
#define LWA140_SET_FALCON26_V                                                                                31:0

#define LWA140_SET_FALCON27                                                                                0x026c
#define LWA140_SET_FALCON27_V                                                                                31:0

#define LWA140_SET_FALCON28                                                                                0x0270
#define LWA140_SET_FALCON28_V                                                                                31:0

#define LWA140_SET_FALCON29                                                                                0x0274
#define LWA140_SET_FALCON29_V                                                                                31:0

#define LWA140_SET_FALCON30                                                                                0x0278
#define LWA140_SET_FALCON30_V                                                                                31:0

#define LWA140_SET_FALCON31                                                                                0x027c
#define LWA140_SET_FALCON31_V                                                                                31:0

#define LWA140_SET_RENDER_ENABLE_A                                                                         0x1550
#define LWA140_SET_RENDER_ENABLE_A_OFFSET_UPPER                                                               7:0

#define LWA140_SET_RENDER_ENABLE_B                                                                         0x1554
#define LWA140_SET_RENDER_ENABLE_B_OFFSET_LOWER                                                              31:0

#define LWA140_SET_RENDER_ENABLE_C                                                                         0x1558
#define LWA140_SET_RENDER_ENABLE_C_MODE                                                                       2:0
#define LWA140_SET_RENDER_ENABLE_C_MODE_FALSE                                                          0x00000000
#define LWA140_SET_RENDER_ENABLE_C_MODE_TRUE                                                           0x00000001
#define LWA140_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                                                    0x00000002
#define LWA140_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                                0x00000003
#define LWA140_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                            0x00000004

#define LWA140_SET_RENDER_ENABLE_OVERRIDE                                                                  0x1944
#define LWA140_SET_RENDER_ENABLE_OVERRIDE_MODE                                                                1:0
#define LWA140_SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE                                       0x00000000
#define LWA140_SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER                                           0x00000001
#define LWA140_SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER                                            0x00000002

#define LWA140_SET_MME_SHADOW_SCRATCH(i)                                                           (0x3400+(i)*4)
#define LWA140_SET_MME_SHADOW_SCRATCH_V                                                                      31:0

#endif /* HWFE_KEPLER_INLINE_TO_MEMORY_B_SW_H */
