/*
 * SPDX-FileCopyrightText: Copyright (c) 2003-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _cl_fermi_twod_a_h_
#define _cl_fermi_twod_a_h_

#define FERMI_TWOD_A    0x902D

typedef volatile struct fermi_twod_a_struct {
    LwU32 SetObject;
    LwU32 Reserved_0x04[0x3F];
    LwU32 NoOperation;
    LwU32 SetNotifyA;
    LwU32 SetNotifyB;
    LwU32 Notify;
    LwU32 WaitForIdle;
    LwU32 LoadMmeInstructionRamPointer;
    LwU32 LoadMmeInstructionRam;
    LwU32 LoadMmeStartAddressRamPointer;
    LwU32 LoadMmeStartAddressRam;
    LwU32 SetMmeShadowRamControl;
    LwU32 Reserved_0x128[0x2];
    LwU32 SetGlobalRenderEnableA;
    LwU32 SetGlobalRenderEnableB;
    LwU32 SetGlobalRenderEnableC;
    LwU32 SendGoIdle;
    LwU32 PmTrigger;
    LwU32 Reserved_0x144[0x3];
    LwU32 SetInstrumentationMethodHeader;
    LwU32 SetInstrumentationMethodData;
    LwU32 Reserved_0x158[0x25];
    LwU32 SetMmeSwitchState;
    LwU32 Reserved_0x1F0[0x4];
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
    LwU32 FlushAndIlwalidateRopMiniCache;
    LwU32 SetSpareNoop06;
    LwU32 SetSrcFormat;
    LwU32 SetSrcMemoryLayout;
    LwU32 SetSrcBlockSize;
    LwU32 SetSrcDepth;
    LwU32 TwodIlwalidateTextureDataCache;
    LwU32 SetSrcPitch;
    LwU32 SetSrcWidth;
    LwU32 SetSrcHeight;
    LwU32 SetSrcOffsetUpper;
    LwU32 SetSrcOffsetLower;
    LwU32 SetPixelsFromMemorySectorPromotion;
    LwU32 SetSpareNoop12;
    LwU32 SetNumProcessingClusters;
    LwU32 SetRenderEnableA;
    LwU32 SetRenderEnableB;
    LwU32 SetRenderEnableC;
    LwU32 SetSpareNoop08;
    LwU32 SetSpareNoop01;
    LwU32 SetSpareNoop11;
    LwU32 SetSpareNoop07;
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
    LwU32 SetDstColorRenderToZetaSurface;
    LwU32 SetSpareNoop04;
    LwU32 SetSpareNoop15;
    LwU32 SetSpareNoop13;
    LwU32 SetSpareNoop03;
    LwU32 SetSpareNoop14;
    LwU32 SetSpareNoop02;
    LwU32 SetCompression;
    LwU32 SetSpareNoop09;
    LwU32 SetRenderEnableOverride;
    LwU32 SetPixelsFromMemoryDirection;
    LwU32 SetSpareNoop10;
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
    LwU32 SetRenderSolidPrimColor0;
    LwU32 SetRenderSolidPrimColor1;
    LwU32 SetRenderSolidPrimColor2;
    LwU32 SetRenderSolidPrimColor3;
    LwU32 SetMmeMemAddressA;
    LwU32 SetMmeMemAddressB;
    LwU32 SetMmeDataRamAddress;
    LwU32 MmeDmaRead;
    LwU32 MmeDmaReadFifoed;
    LwU32 MmeDmaWrite;
    LwU32 MmeDmaReduction;
    LwU32 MmeDmaSysmembar;
    LwU32 MmeDmaSync;
    LwU32 SetMmeDataFifoConfig;
    LwU32 Reserved_0x578[0x2];
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
    LwU32 Reserved_0x960[0x123];
    LwU32 MmeDmaWriteMethodBarrier;
    LwU32 Reserved_0xDF0[0x984];
    LwU32 SetMmeShadowScratch[0x100];
    struct {
        LwU32 Macro;
        LwU32 Data;
    } CallMme[0xE0];
} fermi_twod_a_t;


#define LW902D_SET_OBJECT                                                                                  0x0000
#define LW902D_SET_OBJECT_CLASS_ID                                                                           15:0
#define LW902D_SET_OBJECT_ENGINE_ID                                                                         20:16

#define LW902D_NO_OPERATION                                                                                0x0100
#define LW902D_NO_OPERATION_V                                                                                31:0

#define LW902D_SET_NOTIFY_A                                                                                0x0104
#define LW902D_SET_NOTIFY_A_ADDRESS_UPPER                                                                    24:0

#define LW902D_SET_NOTIFY_B                                                                                0x0108
#define LW902D_SET_NOTIFY_B_ADDRESS_LOWER                                                                    31:0

#define LW902D_NOTIFY                                                                                      0x010c
#define LW902D_NOTIFY_TYPE                                                                                   31:0
#define LW902D_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define LW902D_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define LW902D_WAIT_FOR_IDLE                                                                               0x0110
#define LW902D_WAIT_FOR_IDLE_V                                                                               31:0

#define LW902D_LOAD_MME_INSTRUCTION_RAM_POINTER                                                            0x0114
#define LW902D_LOAD_MME_INSTRUCTION_RAM_POINTER_V                                                            31:0

#define LW902D_LOAD_MME_INSTRUCTION_RAM                                                                    0x0118
#define LW902D_LOAD_MME_INSTRUCTION_RAM_V                                                                    31:0

#define LW902D_LOAD_MME_START_ADDRESS_RAM_POINTER                                                          0x011c
#define LW902D_LOAD_MME_START_ADDRESS_RAM_POINTER_V                                                          31:0

#define LW902D_LOAD_MME_START_ADDRESS_RAM                                                                  0x0120
#define LW902D_LOAD_MME_START_ADDRESS_RAM_V                                                                  31:0

#define LW902D_SET_MME_SHADOW_RAM_CONTROL                                                                  0x0124
#define LW902D_SET_MME_SHADOW_RAM_CONTROL_MODE                                                                1:0
#define LW902D_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK                                            0x00000000
#define LW902D_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK_WITH_FILTER                                0x00000001
#define LW902D_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_PASSTHROUGH                                      0x00000002
#define LW902D_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_REPLAY                                           0x00000003

#define LW902D_SET_GLOBAL_RENDER_ENABLE_A                                                                  0x0130
#define LW902D_SET_GLOBAL_RENDER_ENABLE_A_OFFSET_UPPER                                                        7:0

#define LW902D_SET_GLOBAL_RENDER_ENABLE_B                                                                  0x0134
#define LW902D_SET_GLOBAL_RENDER_ENABLE_B_OFFSET_LOWER                                                       31:0

#define LW902D_SET_GLOBAL_RENDER_ENABLE_C                                                                  0x0138
#define LW902D_SET_GLOBAL_RENDER_ENABLE_C_MODE                                                                2:0
#define LW902D_SET_GLOBAL_RENDER_ENABLE_C_MODE_FALSE                                                   0x00000000
#define LW902D_SET_GLOBAL_RENDER_ENABLE_C_MODE_TRUE                                                    0x00000001
#define LW902D_SET_GLOBAL_RENDER_ENABLE_C_MODE_CONDITIONAL                                             0x00000002
#define LW902D_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                         0x00000003
#define LW902D_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                     0x00000004

#define LW902D_SEND_GO_IDLE                                                                                0x013c
#define LW902D_SEND_GO_IDLE_V                                                                                31:0

#define LW902D_PM_TRIGGER                                                                                  0x0140
#define LW902D_PM_TRIGGER_V                                                                                  31:0

#define LW902D_SET_INSTRUMENTATION_METHOD_HEADER                                                           0x0150
#define LW902D_SET_INSTRUMENTATION_METHOD_HEADER_V                                                           31:0

#define LW902D_SET_INSTRUMENTATION_METHOD_DATA                                                             0x0154
#define LW902D_SET_INSTRUMENTATION_METHOD_DATA_V                                                             31:0

#define LW902D_SET_MME_SWITCH_STATE                                                                        0x01ec
#define LW902D_SET_MME_SWITCH_STATE_VALID                                                                     0:0
#define LW902D_SET_MME_SWITCH_STATE_VALID_FALSE                                                        0x00000000
#define LW902D_SET_MME_SWITCH_STATE_VALID_TRUE                                                         0x00000001
#define LW902D_SET_MME_SWITCH_STATE_SAVE_MACRO                                                               11:4
#define LW902D_SET_MME_SWITCH_STATE_RESTORE_MACRO                                                           19:12

#define LW902D_SET_DST_FORMAT                                                                              0x0200
#define LW902D_SET_DST_FORMAT_V                                                                               7:0
#define LW902D_SET_DST_FORMAT_V_A8R8G8B8                                                               0x000000CF
#define LW902D_SET_DST_FORMAT_V_A8RL8GL8BL8                                                            0x000000D0
#define LW902D_SET_DST_FORMAT_V_A2R10G10B10                                                            0x000000DF
#define LW902D_SET_DST_FORMAT_V_A8B8G8R8                                                               0x000000D5
#define LW902D_SET_DST_FORMAT_V_A8BL8GL8RL8                                                            0x000000D6
#define LW902D_SET_DST_FORMAT_V_A2B10G10R10                                                            0x000000D1
#define LW902D_SET_DST_FORMAT_V_X8R8G8B8                                                               0x000000E6
#define LW902D_SET_DST_FORMAT_V_X8RL8GL8BL8                                                            0x000000E7
#define LW902D_SET_DST_FORMAT_V_X8B8G8R8                                                               0x000000F9
#define LW902D_SET_DST_FORMAT_V_X8BL8GL8RL8                                                            0x000000FA
#define LW902D_SET_DST_FORMAT_V_R5G6B5                                                                 0x000000E8
#define LW902D_SET_DST_FORMAT_V_A1R5G5B5                                                               0x000000E9
#define LW902D_SET_DST_FORMAT_V_X1R5G5B5                                                               0x000000F8
#define LW902D_SET_DST_FORMAT_V_Y8                                                                     0x000000F3
#define LW902D_SET_DST_FORMAT_V_Y16                                                                    0x000000EE
#define LW902D_SET_DST_FORMAT_V_Y32                                                                    0x000000FF
#define LW902D_SET_DST_FORMAT_V_Z1R5G5B5                                                               0x000000FB
#define LW902D_SET_DST_FORMAT_V_O1R5G5B5                                                               0x000000FC
#define LW902D_SET_DST_FORMAT_V_Z8R8G8B8                                                               0x000000FD
#define LW902D_SET_DST_FORMAT_V_O8R8G8B8                                                               0x000000FE
#define LW902D_SET_DST_FORMAT_V_Y1_8X8                                                                 0x0000001C
#define LW902D_SET_DST_FORMAT_V_RF16                                                                   0x000000F2
#define LW902D_SET_DST_FORMAT_V_RF32                                                                   0x000000E5
#define LW902D_SET_DST_FORMAT_V_RF32_GF32                                                              0x000000CB
#define LW902D_SET_DST_FORMAT_V_RF16_GF16_BF16_AF16                                                    0x000000CA
#define LW902D_SET_DST_FORMAT_V_RF16_GF16_BF16_X16                                                     0x000000CE
#define LW902D_SET_DST_FORMAT_V_RF32_GF32_BF32_AF32                                                    0x000000C0
#define LW902D_SET_DST_FORMAT_V_RF32_GF32_BF32_X32                                                     0x000000C3
#define LW902D_SET_DST_FORMAT_V_R16_G16_B16_A16                                                        0x000000C6
#define LW902D_SET_DST_FORMAT_V_RN16_GN16_BN16_AN16                                                    0x000000C7
#define LW902D_SET_DST_FORMAT_V_BF10GF11RF11                                                           0x000000E0
#define LW902D_SET_DST_FORMAT_V_AN8BN8GN8RN8                                                           0x000000D7
#define LW902D_SET_DST_FORMAT_V_RF16_GF16                                                              0x000000DE
#define LW902D_SET_DST_FORMAT_V_R16_G16                                                                0x000000DA
#define LW902D_SET_DST_FORMAT_V_RN16_GN16                                                              0x000000DB
#define LW902D_SET_DST_FORMAT_V_G8R8                                                                   0x000000EA
#define LW902D_SET_DST_FORMAT_V_GN8RN8                                                                 0x000000EB
#define LW902D_SET_DST_FORMAT_V_RN16                                                                   0x000000EF
#define LW902D_SET_DST_FORMAT_V_RN8                                                                    0x000000F4
#define LW902D_SET_DST_FORMAT_V_A8                                                                     0x000000F7

#define LW902D_SET_DST_MEMORY_LAYOUT                                                                       0x0204
#define LW902D_SET_DST_MEMORY_LAYOUT_V                                                                        0:0
#define LW902D_SET_DST_MEMORY_LAYOUT_V_BLOCKLINEAR                                                     0x00000000
#define LW902D_SET_DST_MEMORY_LAYOUT_V_PITCH                                                           0x00000001

#define LW902D_SET_DST_BLOCK_SIZE                                                                          0x0208
#define LW902D_SET_DST_BLOCK_SIZE_HEIGHT                                                                      6:4
#define LW902D_SET_DST_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LW902D_SET_DST_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LW902D_SET_DST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LW902D_SET_DST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LW902D_SET_DST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LW902D_SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LW902D_SET_DST_BLOCK_SIZE_DEPTH                                                                      10:8
#define LW902D_SET_DST_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LW902D_SET_DST_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LW902D_SET_DST_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LW902D_SET_DST_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LW902D_SET_DST_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LW902D_SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LW902D_SET_DST_DEPTH                                                                               0x020c
#define LW902D_SET_DST_DEPTH_V                                                                               31:0

#define LW902D_SET_DST_LAYER                                                                               0x0210
#define LW902D_SET_DST_LAYER_V                                                                               31:0

#define LW902D_SET_DST_PITCH                                                                               0x0214
#define LW902D_SET_DST_PITCH_V                                                                               31:0

#define LW902D_SET_DST_WIDTH                                                                               0x0218
#define LW902D_SET_DST_WIDTH_V                                                                               31:0

#define LW902D_SET_DST_HEIGHT                                                                              0x021c
#define LW902D_SET_DST_HEIGHT_V                                                                              31:0

#define LW902D_SET_DST_OFFSET_UPPER                                                                        0x0220
#define LW902D_SET_DST_OFFSET_UPPER_V                                                                         7:0

#define LW902D_SET_DST_OFFSET_LOWER                                                                        0x0224
#define LW902D_SET_DST_OFFSET_LOWER_V                                                                        31:0

#define LW902D_FLUSH_AND_ILWALIDATE_ROP_MINI_CACHE                                                         0x0228
#define LW902D_FLUSH_AND_ILWALIDATE_ROP_MINI_CACHE_V                                                          0:0

#define LW902D_SET_SPARE_NOOP06                                                                            0x022c
#define LW902D_SET_SPARE_NOOP06_V                                                                            31:0

#define LW902D_SET_SRC_FORMAT                                                                              0x0230
#define LW902D_SET_SRC_FORMAT_V                                                                               7:0
#define LW902D_SET_SRC_FORMAT_V_A8R8G8B8                                                               0x000000CF
#define LW902D_SET_SRC_FORMAT_V_A8RL8GL8BL8                                                            0x000000D0
#define LW902D_SET_SRC_FORMAT_V_A2R10G10B10                                                            0x000000DF
#define LW902D_SET_SRC_FORMAT_V_A8B8G8R8                                                               0x000000D5
#define LW902D_SET_SRC_FORMAT_V_A8BL8GL8RL8                                                            0x000000D6
#define LW902D_SET_SRC_FORMAT_V_A2B10G10R10                                                            0x000000D1
#define LW902D_SET_SRC_FORMAT_V_X8R8G8B8                                                               0x000000E6
#define LW902D_SET_SRC_FORMAT_V_X8RL8GL8BL8                                                            0x000000E7
#define LW902D_SET_SRC_FORMAT_V_X8B8G8R8                                                               0x000000F9
#define LW902D_SET_SRC_FORMAT_V_X8BL8GL8RL8                                                            0x000000FA
#define LW902D_SET_SRC_FORMAT_V_R5G6B5                                                                 0x000000E8
#define LW902D_SET_SRC_FORMAT_V_A1R5G5B5                                                               0x000000E9
#define LW902D_SET_SRC_FORMAT_V_X1R5G5B5                                                               0x000000F8
#define LW902D_SET_SRC_FORMAT_V_Y8                                                                     0x000000F3
#define LW902D_SET_SRC_FORMAT_V_AY8                                                                    0x0000001D
#define LW902D_SET_SRC_FORMAT_V_Y16                                                                    0x000000EE
#define LW902D_SET_SRC_FORMAT_V_Y32                                                                    0x000000FF
#define LW902D_SET_SRC_FORMAT_V_Z1R5G5B5                                                               0x000000FB
#define LW902D_SET_SRC_FORMAT_V_O1R5G5B5                                                               0x000000FC
#define LW902D_SET_SRC_FORMAT_V_Z8R8G8B8                                                               0x000000FD
#define LW902D_SET_SRC_FORMAT_V_O8R8G8B8                                                               0x000000FE
#define LW902D_SET_SRC_FORMAT_V_Y1_8X8                                                                 0x0000001C
#define LW902D_SET_SRC_FORMAT_V_RF16                                                                   0x000000F2
#define LW902D_SET_SRC_FORMAT_V_RF32                                                                   0x000000E5
#define LW902D_SET_SRC_FORMAT_V_RF32_GF32                                                              0x000000CB
#define LW902D_SET_SRC_FORMAT_V_RF16_GF16_BF16_AF16                                                    0x000000CA
#define LW902D_SET_SRC_FORMAT_V_RF16_GF16_BF16_X16                                                     0x000000CE
#define LW902D_SET_SRC_FORMAT_V_RF32_GF32_BF32_AF32                                                    0x000000C0
#define LW902D_SET_SRC_FORMAT_V_RF32_GF32_BF32_X32                                                     0x000000C3
#define LW902D_SET_SRC_FORMAT_V_R16_G16_B16_A16                                                        0x000000C6
#define LW902D_SET_SRC_FORMAT_V_RN16_GN16_BN16_AN16                                                    0x000000C7
#define LW902D_SET_SRC_FORMAT_V_BF10GF11RF11                                                           0x000000E0
#define LW902D_SET_SRC_FORMAT_V_AN8BN8GN8RN8                                                           0x000000D7
#define LW902D_SET_SRC_FORMAT_V_RF16_GF16                                                              0x000000DE
#define LW902D_SET_SRC_FORMAT_V_R16_G16                                                                0x000000DA
#define LW902D_SET_SRC_FORMAT_V_RN16_GN16                                                              0x000000DB
#define LW902D_SET_SRC_FORMAT_V_G8R8                                                                   0x000000EA
#define LW902D_SET_SRC_FORMAT_V_GN8RN8                                                                 0x000000EB
#define LW902D_SET_SRC_FORMAT_V_RN16                                                                   0x000000EF
#define LW902D_SET_SRC_FORMAT_V_RN8                                                                    0x000000F4
#define LW902D_SET_SRC_FORMAT_V_A8                                                                     0x000000F7

#define LW902D_SET_SRC_MEMORY_LAYOUT                                                                       0x0234
#define LW902D_SET_SRC_MEMORY_LAYOUT_V                                                                        0:0
#define LW902D_SET_SRC_MEMORY_LAYOUT_V_BLOCKLINEAR                                                     0x00000000
#define LW902D_SET_SRC_MEMORY_LAYOUT_V_PITCH                                                           0x00000001

#define LW902D_SET_SRC_BLOCK_SIZE                                                                          0x0238
#define LW902D_SET_SRC_BLOCK_SIZE_HEIGHT                                                                      6:4
#define LW902D_SET_SRC_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LW902D_SET_SRC_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LW902D_SET_SRC_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LW902D_SET_SRC_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LW902D_SET_SRC_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LW902D_SET_SRC_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LW902D_SET_SRC_BLOCK_SIZE_DEPTH                                                                      10:8
#define LW902D_SET_SRC_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LW902D_SET_SRC_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LW902D_SET_SRC_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LW902D_SET_SRC_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LW902D_SET_SRC_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LW902D_SET_SRC_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LW902D_SET_SRC_DEPTH                                                                               0x023c
#define LW902D_SET_SRC_DEPTH_V                                                                               31:0

#define LW902D_TWOD_ILWALIDATE_TEXTURE_DATA_CACHE                                                          0x0240
#define LW902D_TWOD_ILWALIDATE_TEXTURE_DATA_CACHE_V                                                           1:0
#define LW902D_TWOD_ILWALIDATE_TEXTURE_DATA_CACHE_V_L1_ONLY                                            0x00000000
#define LW902D_TWOD_ILWALIDATE_TEXTURE_DATA_CACHE_V_L2_ONLY                                            0x00000001
#define LW902D_TWOD_ILWALIDATE_TEXTURE_DATA_CACHE_V_L1_AND_L2                                          0x00000002

#define LW902D_SET_SRC_PITCH                                                                               0x0244
#define LW902D_SET_SRC_PITCH_V                                                                               31:0

#define LW902D_SET_SRC_WIDTH                                                                               0x0248
#define LW902D_SET_SRC_WIDTH_V                                                                               31:0

#define LW902D_SET_SRC_HEIGHT                                                                              0x024c
#define LW902D_SET_SRC_HEIGHT_V                                                                              31:0

#define LW902D_SET_SRC_OFFSET_UPPER                                                                        0x0250
#define LW902D_SET_SRC_OFFSET_UPPER_V                                                                         7:0

#define LW902D_SET_SRC_OFFSET_LOWER                                                                        0x0254
#define LW902D_SET_SRC_OFFSET_LOWER_V                                                                        31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION                                                     0x0258
#define LW902D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V                                                      1:0
#define LW902D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_NO_PROMOTION                                  0x00000000
#define LW902D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_PROMOTE_TO_2_V                                0x00000001
#define LW902D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_PROMOTE_TO_2_H                                0x00000002
#define LW902D_SET_PIXELS_FROM_MEMORY_SECTOR_PROMOTION_V_PROMOTE_TO_4                                  0x00000003

#define LW902D_SET_SPARE_NOOP12                                                                            0x025c
#define LW902D_SET_SPARE_NOOP12_V                                                                            31:0

#define LW902D_SET_NUM_PROCESSING_CLUSTERS                                                                 0x0260
#define LW902D_SET_NUM_PROCESSING_CLUSTERS_V                                                                  0:0
#define LW902D_SET_NUM_PROCESSING_CLUSTERS_V_ALL                                                       0x00000000
#define LW902D_SET_NUM_PROCESSING_CLUSTERS_V_ONE                                                       0x00000001

#define LW902D_SET_RENDER_ENABLE_A                                                                         0x0264
#define LW902D_SET_RENDER_ENABLE_A_OFFSET_UPPER                                                               7:0

#define LW902D_SET_RENDER_ENABLE_B                                                                         0x0268
#define LW902D_SET_RENDER_ENABLE_B_OFFSET_LOWER                                                              31:0

#define LW902D_SET_RENDER_ENABLE_C                                                                         0x026c
#define LW902D_SET_RENDER_ENABLE_C_MODE                                                                       2:0
#define LW902D_SET_RENDER_ENABLE_C_MODE_FALSE                                                          0x00000000
#define LW902D_SET_RENDER_ENABLE_C_MODE_TRUE                                                           0x00000001
#define LW902D_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                                                    0x00000002
#define LW902D_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                                0x00000003
#define LW902D_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                            0x00000004

#define LW902D_SET_SPARE_NOOP08                                                                            0x0270
#define LW902D_SET_SPARE_NOOP08_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP01                                                                            0x0274
#define LW902D_SET_SPARE_NOOP01_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP11                                                                            0x0278
#define LW902D_SET_SPARE_NOOP11_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP07                                                                            0x027c
#define LW902D_SET_SPARE_NOOP07_V                                                                            31:0

#define LW902D_SET_CLIP_X0                                                                                 0x0280
#define LW902D_SET_CLIP_X0_V                                                                                 31:0

#define LW902D_SET_CLIP_Y0                                                                                 0x0284
#define LW902D_SET_CLIP_Y0_V                                                                                 31:0

#define LW902D_SET_CLIP_WIDTH                                                                              0x0288
#define LW902D_SET_CLIP_WIDTH_V                                                                              31:0

#define LW902D_SET_CLIP_HEIGHT                                                                             0x028c
#define LW902D_SET_CLIP_HEIGHT_V                                                                             31:0

#define LW902D_SET_CLIP_ENABLE                                                                             0x0290
#define LW902D_SET_CLIP_ENABLE_V                                                                              0:0
#define LW902D_SET_CLIP_ENABLE_V_FALSE                                                                 0x00000000
#define LW902D_SET_CLIP_ENABLE_V_TRUE                                                                  0x00000001

#define LW902D_SET_COLOR_KEY_FORMAT                                                                        0x0294
#define LW902D_SET_COLOR_KEY_FORMAT_V                                                                         2:0
#define LW902D_SET_COLOR_KEY_FORMAT_V_A16R5G6B5                                                        0x00000000
#define LW902D_SET_COLOR_KEY_FORMAT_V_A1R5G5B5                                                         0x00000001
#define LW902D_SET_COLOR_KEY_FORMAT_V_A8R8G8B8                                                         0x00000002
#define LW902D_SET_COLOR_KEY_FORMAT_V_A2R10G10B10                                                      0x00000003
#define LW902D_SET_COLOR_KEY_FORMAT_V_Y8                                                               0x00000004
#define LW902D_SET_COLOR_KEY_FORMAT_V_Y16                                                              0x00000005
#define LW902D_SET_COLOR_KEY_FORMAT_V_Y32                                                              0x00000006

#define LW902D_SET_COLOR_KEY                                                                               0x0298
#define LW902D_SET_COLOR_KEY_V                                                                               31:0

#define LW902D_SET_COLOR_KEY_ENABLE                                                                        0x029c
#define LW902D_SET_COLOR_KEY_ENABLE_V                                                                         0:0
#define LW902D_SET_COLOR_KEY_ENABLE_V_FALSE                                                            0x00000000
#define LW902D_SET_COLOR_KEY_ENABLE_V_TRUE                                                             0x00000001

#define LW902D_SET_ROP                                                                                     0x02a0
#define LW902D_SET_ROP_V                                                                                      7:0

#define LW902D_SET_BETA1                                                                                   0x02a4
#define LW902D_SET_BETA1_V                                                                                   31:0

#define LW902D_SET_BETA4                                                                                   0x02a8
#define LW902D_SET_BETA4_B                                                                                    7:0
#define LW902D_SET_BETA4_G                                                                                   15:8
#define LW902D_SET_BETA4_R                                                                                  23:16
#define LW902D_SET_BETA4_A                                                                                  31:24

#define LW902D_SET_OPERATION                                                                               0x02ac
#define LW902D_SET_OPERATION_V                                                                                2:0
#define LW902D_SET_OPERATION_V_SRCCOPY_AND                                                             0x00000000
#define LW902D_SET_OPERATION_V_ROP_AND                                                                 0x00000001
#define LW902D_SET_OPERATION_V_BLEND_AND                                                               0x00000002
#define LW902D_SET_OPERATION_V_SRCCOPY                                                                 0x00000003
#define LW902D_SET_OPERATION_V_ROP                                                                     0x00000004
#define LW902D_SET_OPERATION_V_SRCCOPY_PREMULT                                                         0x00000005
#define LW902D_SET_OPERATION_V_BLEND_PREMULT                                                           0x00000006

#define LW902D_SET_PATTERN_OFFSET                                                                          0x02b0
#define LW902D_SET_PATTERN_OFFSET_X                                                                           5:0
#define LW902D_SET_PATTERN_OFFSET_Y                                                                          13:8

#define LW902D_SET_PATTERN_SELECT                                                                          0x02b4
#define LW902D_SET_PATTERN_SELECT_V                                                                           1:0
#define LW902D_SET_PATTERN_SELECT_V_MONOCHROME_8x8                                                     0x00000000
#define LW902D_SET_PATTERN_SELECT_V_MONOCHROME_64x1                                                    0x00000001
#define LW902D_SET_PATTERN_SELECT_V_MONOCHROME_1x64                                                    0x00000002
#define LW902D_SET_PATTERN_SELECT_V_COLOR                                                              0x00000003

#define LW902D_SET_DST_COLOR_RENDER_TO_ZETA_SURFACE                                                        0x02b8
#define LW902D_SET_DST_COLOR_RENDER_TO_ZETA_SURFACE_V                                                         0:0
#define LW902D_SET_DST_COLOR_RENDER_TO_ZETA_SURFACE_V_FALSE                                            0x00000000
#define LW902D_SET_DST_COLOR_RENDER_TO_ZETA_SURFACE_V_TRUE                                             0x00000001

#define LW902D_SET_SPARE_NOOP04                                                                            0x02bc
#define LW902D_SET_SPARE_NOOP04_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP15                                                                            0x02c0
#define LW902D_SET_SPARE_NOOP15_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP13                                                                            0x02c4
#define LW902D_SET_SPARE_NOOP13_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP03                                                                            0x02c8
#define LW902D_SET_SPARE_NOOP03_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP14                                                                            0x02cc
#define LW902D_SET_SPARE_NOOP14_V                                                                            31:0

#define LW902D_SET_SPARE_NOOP02                                                                            0x02d0
#define LW902D_SET_SPARE_NOOP02_V                                                                            31:0

#define LW902D_SET_COMPRESSION                                                                             0x02d4
#define LW902D_SET_COMPRESSION_ENABLE                                                                         0:0
#define LW902D_SET_COMPRESSION_ENABLE_FALSE                                                            0x00000000
#define LW902D_SET_COMPRESSION_ENABLE_TRUE                                                             0x00000001

#define LW902D_SET_SPARE_NOOP09                                                                            0x02d8
#define LW902D_SET_SPARE_NOOP09_V                                                                            31:0

#define LW902D_SET_RENDER_ENABLE_OVERRIDE                                                                  0x02dc
#define LW902D_SET_RENDER_ENABLE_OVERRIDE_MODE                                                                1:0
#define LW902D_SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE                                       0x00000000
#define LW902D_SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER                                           0x00000001
#define LW902D_SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER                                            0x00000002

#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION                                                            0x02e0
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_HORIZONTAL                                                    1:0
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_HORIZONTAL_HW_DECIDES                                  0x00000000
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_HORIZONTAL_LEFT_TO_RIGHT                               0x00000001
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_HORIZONTAL_RIGHT_TO_LEFT                               0x00000002
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_VERTICAL                                                      5:4
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_VERTICAL_HW_DECIDES                                    0x00000000
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_VERTICAL_TOP_TO_BOTTOM                                 0x00000001
#define LW902D_SET_PIXELS_FROM_MEMORY_DIRECTION_VERTICAL_BOTTOM_TO_TOP                                 0x00000002

#define LW902D_SET_SPARE_NOOP10                                                                            0x02e4
#define LW902D_SET_SPARE_NOOP10_V                                                                            31:0

#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT                                                         0x02e8
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V                                                          2:0
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8X8R5G6B5                                        0x00000000
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A1R5G5B5                                          0x00000001
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8R8G8B8                                          0x00000002
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8Y8                                              0x00000003
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_A8X8Y16                                           0x00000004
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_Y32                                               0x00000005
#define LW902D_SET_MONOCHROME_PATTERN_COLOR_FORMAT_V_BYTE_EXPAND                                       0x00000006

#define LW902D_SET_MONOCHROME_PATTERN_FORMAT                                                               0x02ec
#define LW902D_SET_MONOCHROME_PATTERN_FORMAT_V                                                                0:0
#define LW902D_SET_MONOCHROME_PATTERN_FORMAT_V_CGA6_M1                                                 0x00000000
#define LW902D_SET_MONOCHROME_PATTERN_FORMAT_V_LE_M1                                                   0x00000001

#define LW902D_SET_MONOCHROME_PATTERN_COLOR0                                                               0x02f0
#define LW902D_SET_MONOCHROME_PATTERN_COLOR0_V                                                               31:0

#define LW902D_SET_MONOCHROME_PATTERN_COLOR1                                                               0x02f4
#define LW902D_SET_MONOCHROME_PATTERN_COLOR1_V                                                               31:0

#define LW902D_SET_MONOCHROME_PATTERN0                                                                     0x02f8
#define LW902D_SET_MONOCHROME_PATTERN0_V                                                                     31:0

#define LW902D_SET_MONOCHROME_PATTERN1                                                                     0x02fc
#define LW902D_SET_MONOCHROME_PATTERN1_V                                                                     31:0

#define LW902D_COLOR_PATTERN_X8R8G8B8(i)                                                           (0x0300+(i)*4)
#define LW902D_COLOR_PATTERN_X8R8G8B8_B0                                                                      7:0
#define LW902D_COLOR_PATTERN_X8R8G8B8_G0                                                                     15:8
#define LW902D_COLOR_PATTERN_X8R8G8B8_R0                                                                    23:16
#define LW902D_COLOR_PATTERN_X8R8G8B8_IGNORE0                                                               31:24

#define LW902D_COLOR_PATTERN_R5G6B5(i)                                                             (0x0400+(i)*4)
#define LW902D_COLOR_PATTERN_R5G6B5_B0                                                                        4:0
#define LW902D_COLOR_PATTERN_R5G6B5_G0                                                                       10:5
#define LW902D_COLOR_PATTERN_R5G6B5_R0                                                                      15:11
#define LW902D_COLOR_PATTERN_R5G6B5_B1                                                                      20:16
#define LW902D_COLOR_PATTERN_R5G6B5_G1                                                                      26:21
#define LW902D_COLOR_PATTERN_R5G6B5_R1                                                                      31:27

#define LW902D_COLOR_PATTERN_X1R5G5B5(i)                                                           (0x0480+(i)*4)
#define LW902D_COLOR_PATTERN_X1R5G5B5_B0                                                                      4:0
#define LW902D_COLOR_PATTERN_X1R5G5B5_G0                                                                      9:5
#define LW902D_COLOR_PATTERN_X1R5G5B5_R0                                                                    14:10
#define LW902D_COLOR_PATTERN_X1R5G5B5_IGNORE0                                                               15:15
#define LW902D_COLOR_PATTERN_X1R5G5B5_B1                                                                    20:16
#define LW902D_COLOR_PATTERN_X1R5G5B5_G1                                                                    25:21
#define LW902D_COLOR_PATTERN_X1R5G5B5_R1                                                                    30:26
#define LW902D_COLOR_PATTERN_X1R5G5B5_IGNORE1                                                               31:31

#define LW902D_COLOR_PATTERN_Y8(i)                                                                 (0x0500+(i)*4)
#define LW902D_COLOR_PATTERN_Y8_Y0                                                                            7:0
#define LW902D_COLOR_PATTERN_Y8_Y1                                                                           15:8
#define LW902D_COLOR_PATTERN_Y8_Y2                                                                          23:16
#define LW902D_COLOR_PATTERN_Y8_Y3                                                                          31:24

#define LW902D_SET_RENDER_SOLID_PRIM_COLOR0                                                                0x0540
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR0_V                                                                31:0

#define LW902D_SET_RENDER_SOLID_PRIM_COLOR1                                                                0x0544
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR1_V                                                                31:0

#define LW902D_SET_RENDER_SOLID_PRIM_COLOR2                                                                0x0548
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR2_V                                                                31:0

#define LW902D_SET_RENDER_SOLID_PRIM_COLOR3                                                                0x054c
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR3_V                                                                31:0

#define LW902D_SET_MME_MEM_ADDRESS_A                                                                       0x0550
#define LW902D_SET_MME_MEM_ADDRESS_A_UPPER                                                                   24:0

#define LW902D_SET_MME_MEM_ADDRESS_B                                                                       0x0554
#define LW902D_SET_MME_MEM_ADDRESS_B_LOWER                                                                   31:0

#define LW902D_SET_MME_DATA_RAM_ADDRESS                                                                    0x0558
#define LW902D_SET_MME_DATA_RAM_ADDRESS_WORD                                                                 31:0

#define LW902D_MME_DMA_READ                                                                                0x055c
#define LW902D_MME_DMA_READ_LENGTH                                                                           31:0

#define LW902D_MME_DMA_READ_FIFOED                                                                         0x0560
#define LW902D_MME_DMA_READ_FIFOED_LENGTH                                                                    31:0

#define LW902D_MME_DMA_WRITE                                                                               0x0564
#define LW902D_MME_DMA_WRITE_LENGTH                                                                          31:0

#define LW902D_MME_DMA_REDUCTION                                                                           0x0568
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP                                                                 2:0
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_ADD                                                  0x00000000
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_MIN                                                  0x00000001
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_MAX                                                  0x00000002
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_INC                                                  0x00000003
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_DEC                                                  0x00000004
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_AND                                                  0x00000005
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_OR                                                   0x00000006
#define LW902D_MME_DMA_REDUCTION_REDUCTION_OP_RED_XOR                                                  0x00000007
#define LW902D_MME_DMA_REDUCTION_REDUCTION_FORMAT                                                             5:4
#define LW902D_MME_DMA_REDUCTION_REDUCTION_FORMAT_UNSIGNED                                             0x00000000
#define LW902D_MME_DMA_REDUCTION_REDUCTION_FORMAT_SIGNED                                               0x00000001
#define LW902D_MME_DMA_REDUCTION_REDUCTION_SIZE                                                               8:8
#define LW902D_MME_DMA_REDUCTION_REDUCTION_SIZE_FOUR_BYTES                                             0x00000000
#define LW902D_MME_DMA_REDUCTION_REDUCTION_SIZE_EIGHT_BYTES                                            0x00000001

#define LW902D_MME_DMA_SYSMEMBAR                                                                           0x056c
#define LW902D_MME_DMA_SYSMEMBAR_V                                                                            0:0

#define LW902D_MME_DMA_SYNC                                                                                0x0570
#define LW902D_MME_DMA_SYNC_VALUE                                                                            31:0

#define LW902D_SET_MME_DATA_FIFO_CONFIG                                                                    0x0574
#define LW902D_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE                                                             2:0
#define LW902D_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_0KB                                             0x00000000
#define LW902D_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_4KB                                             0x00000001
#define LW902D_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_8KB                                             0x00000002
#define LW902D_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_12KB                                            0x00000003
#define LW902D_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_16KB                                            0x00000004

#define LW902D_RENDER_SOLID_PRIM_MODE                                                                      0x0580
#define LW902D_RENDER_SOLID_PRIM_MODE_V                                                                       2:0
#define LW902D_RENDER_SOLID_PRIM_MODE_V_POINTS                                                         0x00000000
#define LW902D_RENDER_SOLID_PRIM_MODE_V_LINES                                                          0x00000001
#define LW902D_RENDER_SOLID_PRIM_MODE_V_POLYLINE                                                       0x00000002
#define LW902D_RENDER_SOLID_PRIM_MODE_V_TRIANGLES                                                      0x00000003
#define LW902D_RENDER_SOLID_PRIM_MODE_V_RECTS                                                          0x00000004

#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT                                                          0x0584
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V                                                           7:0
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_RF32_GF32_BF32_AF32                                0x000000C0
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_RF16_GF16_BF16_AF16                                0x000000CA
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_RF32_GF32                                          0x000000CB
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A8R8G8B8                                           0x000000CF
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A2R10G10B10                                        0x000000DF
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A8B8G8R8                                           0x000000D5
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A2B10G10R10                                        0x000000D1
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_X8R8G8B8                                           0x000000E6
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_X8B8G8R8                                           0x000000F9
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_R5G6B5                                             0x000000E8
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_A1R5G5B5                                           0x000000E9
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_X1R5G5B5                                           0x000000F8
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Y8                                                 0x000000F3
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Y16                                                0x000000EE
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Y32                                                0x000000FF
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Z1R5G5B5                                           0x000000FB
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_O1R5G5B5                                           0x000000FC
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_Z8R8G8B8                                           0x000000FD
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_FORMAT_V_O8R8G8B8                                           0x000000FE

#define LW902D_SET_RENDER_SOLID_PRIM_COLOR                                                                 0x0588
#define LW902D_SET_RENDER_SOLID_PRIM_COLOR_V                                                                 31:0

#define LW902D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS                                                        0x058c
#define LW902D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_XMAJ__XINC__YINC                                          0:0
#define LW902D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_XMAJ__XDEC__YINC                                          4:4
#define LW902D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_YMAJ__XINC__YINC                                          8:8
#define LW902D_SET_RENDER_SOLID_LINE_TIE_BREAK_BITS_YMAJ__XDEC__YINC                                        12:12

#define LW902D_RENDER_SOLID_PRIM_POINT_X_Y                                                                 0x05e0
#define LW902D_RENDER_SOLID_PRIM_POINT_X_Y_X                                                                 15:0
#define LW902D_RENDER_SOLID_PRIM_POINT_X_Y_Y                                                                31:16

#define LW902D_RENDER_SOLID_PRIM_POINT_SET_X(j)                                                    (0x0600+(j)*8)
#define LW902D_RENDER_SOLID_PRIM_POINT_SET_X_V                                                               31:0

#define LW902D_RENDER_SOLID_PRIM_POINT_Y(j)                                                        (0x0604+(j)*8)
#define LW902D_RENDER_SOLID_PRIM_POINT_Y_V                                                                   31:0

#define LW902D_SET_PIXELS_FROM_CPU_DATA_TYPE                                                               0x0800
#define LW902D_SET_PIXELS_FROM_CPU_DATA_TYPE_V                                                                0:0
#define LW902D_SET_PIXELS_FROM_CPU_DATA_TYPE_V_COLOR                                                   0x00000000
#define LW902D_SET_PIXELS_FROM_CPU_DATA_TYPE_V_INDEX                                                   0x00000001

#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT                                                            0x0804
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V                                                             7:0
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A8R8G8B8                                             0x000000CF
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A2R10G10B10                                          0x000000DF
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A8B8G8R8                                             0x000000D5
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A2B10G10R10                                          0x000000D1
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_X8R8G8B8                                             0x000000E6
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_X8B8G8R8                                             0x000000F9
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_R5G6B5                                               0x000000E8
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_A1R5G5B5                                             0x000000E9
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_X1R5G5B5                                             0x000000F8
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Y8                                                   0x000000F3
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Y16                                                  0x000000EE
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Y32                                                  0x000000FF
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Z1R5G5B5                                             0x000000FB
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_O1R5G5B5                                             0x000000FC
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_Z8R8G8B8                                             0x000000FD
#define LW902D_SET_PIXELS_FROM_CPU_COLOR_FORMAT_V_O8R8G8B8                                             0x000000FE

#define LW902D_SET_PIXELS_FROM_CPU_INDEX_FORMAT                                                            0x0808
#define LW902D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V                                                             1:0
#define LW902D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V_I1                                                   0x00000000
#define LW902D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V_I4                                                   0x00000001
#define LW902D_SET_PIXELS_FROM_CPU_INDEX_FORMAT_V_I8                                                   0x00000002

#define LW902D_SET_PIXELS_FROM_CPU_MONO_FORMAT                                                             0x080c
#define LW902D_SET_PIXELS_FROM_CPU_MONO_FORMAT_V                                                              0:0
#define LW902D_SET_PIXELS_FROM_CPU_MONO_FORMAT_V_CGA6_M1                                               0x00000000
#define LW902D_SET_PIXELS_FROM_CPU_MONO_FORMAT_V_LE_M1                                                 0x00000001

#define LW902D_SET_PIXELS_FROM_CPU_WRAP                                                                    0x0810
#define LW902D_SET_PIXELS_FROM_CPU_WRAP_V                                                                     1:0
#define LW902D_SET_PIXELS_FROM_CPU_WRAP_V_WRAP_PIXEL                                                   0x00000000
#define LW902D_SET_PIXELS_FROM_CPU_WRAP_V_WRAP_BYTE                                                    0x00000001
#define LW902D_SET_PIXELS_FROM_CPU_WRAP_V_WRAP_DWORD                                                   0x00000002

#define LW902D_SET_PIXELS_FROM_CPU_COLOR0                                                                  0x0814
#define LW902D_SET_PIXELS_FROM_CPU_COLOR0_V                                                                  31:0

#define LW902D_SET_PIXELS_FROM_CPU_COLOR1                                                                  0x0818
#define LW902D_SET_PIXELS_FROM_CPU_COLOR1_V                                                                  31:0

#define LW902D_SET_PIXELS_FROM_CPU_MONO_OPACITY                                                            0x081c
#define LW902D_SET_PIXELS_FROM_CPU_MONO_OPACITY_V                                                             0:0
#define LW902D_SET_PIXELS_FROM_CPU_MONO_OPACITY_V_TRANSPARENT                                          0x00000000
#define LW902D_SET_PIXELS_FROM_CPU_MONO_OPACITY_V_OPAQUE                                               0x00000001

#define LW902D_SET_PIXELS_FROM_CPU_SRC_WIDTH                                                               0x0838
#define LW902D_SET_PIXELS_FROM_CPU_SRC_WIDTH_V                                                               31:0

#define LW902D_SET_PIXELS_FROM_CPU_SRC_HEIGHT                                                              0x083c
#define LW902D_SET_PIXELS_FROM_CPU_SRC_HEIGHT_V                                                              31:0

#define LW902D_SET_PIXELS_FROM_CPU_DX_DU_FRAC                                                              0x0840
#define LW902D_SET_PIXELS_FROM_CPU_DX_DU_FRAC_V                                                              31:0

#define LW902D_SET_PIXELS_FROM_CPU_DX_DU_INT                                                               0x0844
#define LW902D_SET_PIXELS_FROM_CPU_DX_DU_INT_V                                                               31:0

#define LW902D_SET_PIXELS_FROM_CPU_DY_DV_FRAC                                                              0x0848
#define LW902D_SET_PIXELS_FROM_CPU_DY_DV_FRAC_V                                                              31:0

#define LW902D_SET_PIXELS_FROM_CPU_DY_DV_INT                                                               0x084c
#define LW902D_SET_PIXELS_FROM_CPU_DY_DV_INT_V                                                               31:0

#define LW902D_SET_PIXELS_FROM_CPU_DST_X0_FRAC                                                             0x0850
#define LW902D_SET_PIXELS_FROM_CPU_DST_X0_FRAC_V                                                             31:0

#define LW902D_SET_PIXELS_FROM_CPU_DST_X0_INT                                                              0x0854
#define LW902D_SET_PIXELS_FROM_CPU_DST_X0_INT_V                                                              31:0

#define LW902D_SET_PIXELS_FROM_CPU_DST_Y0_FRAC                                                             0x0858
#define LW902D_SET_PIXELS_FROM_CPU_DST_Y0_FRAC_V                                                             31:0

#define LW902D_SET_PIXELS_FROM_CPU_DST_Y0_INT                                                              0x085c
#define LW902D_SET_PIXELS_FROM_CPU_DST_Y0_INT_V                                                              31:0

#define LW902D_PIXELS_FROM_CPU_DATA                                                                        0x0860
#define LW902D_PIXELS_FROM_CPU_DATA_V                                                                        31:0

#define LW902D_SET_BIG_ENDIAN_CONTROL                                                                      0x0870
#define LW902D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_1                                                              0:0
#define LW902D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_4                                                              1:1
#define LW902D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_8                                                              2:2
#define LW902D_SET_BIG_ENDIAN_CONTROL_X32_SWAP_16                                                             3:3
#define LW902D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_1                                                              4:4
#define LW902D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_4                                                              5:5
#define LW902D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_8                                                              6:6
#define LW902D_SET_BIG_ENDIAN_CONTROL_X16_SWAP_16                                                             7:7
#define LW902D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_1                                                               8:8
#define LW902D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_4                                                               9:9
#define LW902D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_8                                                             10:10
#define LW902D_SET_BIG_ENDIAN_CONTROL_X8_SWAP_16                                                            11:11
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_1                                                     12:12
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_4                                                     13:13
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_8                                                     14:14
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_CGA6_SWAP_16                                                    15:15
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_1                                                       16:16
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_4                                                       17:17
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_8                                                       18:18
#define LW902D_SET_BIG_ENDIAN_CONTROL_I1_X8_LE_SWAP_16                                                      19:19
#define LW902D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_1                                                             20:20
#define LW902D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_4                                                             21:21
#define LW902D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_8                                                             22:22
#define LW902D_SET_BIG_ENDIAN_CONTROL_I4_SWAP_16                                                            23:23
#define LW902D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_1                                                             24:24
#define LW902D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_4                                                             25:25
#define LW902D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_8                                                             26:26
#define LW902D_SET_BIG_ENDIAN_CONTROL_I8_SWAP_16                                                            27:27
#define LW902D_SET_BIG_ENDIAN_CONTROL_OVERRIDE                                                              28:28

#define LW902D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE                                                          0x0880
#define LW902D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V                                                           2:0
#define LW902D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V_AUTO                                               0x00000000
#define LW902D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V_SHAPE_8X8                                          0x00000001
#define LW902D_SET_PIXELS_FROM_MEMORY_BLOCK_SHAPE_V_SHAPE_16X4                                         0x00000002

#define LW902D_SET_PIXELS_FROM_MEMORY_CORRAL_SIZE                                                          0x0884
#define LW902D_SET_PIXELS_FROM_MEMORY_CORRAL_SIZE_V                                                           9:0

#define LW902D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP                                                         0x0888
#define LW902D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP_V                                                          0:0
#define LW902D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP_V_FALSE                                             0x00000000
#define LW902D_SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP_V_TRUE                                              0x00000001

#define LW902D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE                                                          0x088c
#define LW902D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN                                                      0:0
#define LW902D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN_CENTER                                        0x00000000
#define LW902D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_ORIGIN_CORNER                                        0x00000001
#define LW902D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER                                                      4:4
#define LW902D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER_POINT                                         0x00000000
#define LW902D_SET_PIXELS_FROM_MEMORY_SAMPLE_MODE_FILTER_BILINEAR                                      0x00000001

#define LW902D_SET_PIXELS_FROM_MEMORY_DST_X0                                                               0x08b0
#define LW902D_SET_PIXELS_FROM_MEMORY_DST_X0_V                                                               31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_DST_Y0                                                               0x08b4
#define LW902D_SET_PIXELS_FROM_MEMORY_DST_Y0_V                                                               31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_DST_WIDTH                                                            0x08b8
#define LW902D_SET_PIXELS_FROM_MEMORY_DST_WIDTH_V                                                            31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_DST_HEIGHT                                                           0x08bc
#define LW902D_SET_PIXELS_FROM_MEMORY_DST_HEIGHT_V                                                           31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_DU_DX_FRAC                                                           0x08c0
#define LW902D_SET_PIXELS_FROM_MEMORY_DU_DX_FRAC_V                                                           31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_DU_DX_INT                                                            0x08c4
#define LW902D_SET_PIXELS_FROM_MEMORY_DU_DX_INT_V                                                            31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_DV_DY_FRAC                                                           0x08c8
#define LW902D_SET_PIXELS_FROM_MEMORY_DV_DY_FRAC_V                                                           31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_DV_DY_INT                                                            0x08cc
#define LW902D_SET_PIXELS_FROM_MEMORY_DV_DY_INT_V                                                            31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC                                                          0x08d0
#define LW902D_SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC_V                                                          31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_SRC_X0_INT                                                           0x08d4
#define LW902D_SET_PIXELS_FROM_MEMORY_SRC_X0_INT_V                                                           31:0

#define LW902D_SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC                                                          0x08d8
#define LW902D_SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC_V                                                          31:0

#define LW902D_PIXELS_FROM_MEMORY_SRC_Y0_INT                                                               0x08dc
#define LW902D_PIXELS_FROM_MEMORY_SRC_Y0_INT_V                                                               31:0

#define LW902D_SET_FALCON00                                                                                0x08e0
#define LW902D_SET_FALCON00_V                                                                                31:0

#define LW902D_SET_FALCON01                                                                                0x08e4
#define LW902D_SET_FALCON01_V                                                                                31:0

#define LW902D_SET_FALCON02                                                                                0x08e8
#define LW902D_SET_FALCON02_V                                                                                31:0

#define LW902D_SET_FALCON03                                                                                0x08ec
#define LW902D_SET_FALCON03_V                                                                                31:0

#define LW902D_SET_FALCON04                                                                                0x08f0
#define LW902D_SET_FALCON04_V                                                                                31:0

#define LW902D_SET_FALCON05                                                                                0x08f4
#define LW902D_SET_FALCON05_V                                                                                31:0

#define LW902D_SET_FALCON06                                                                                0x08f8
#define LW902D_SET_FALCON06_V                                                                                31:0

#define LW902D_SET_FALCON07                                                                                0x08fc
#define LW902D_SET_FALCON07_V                                                                                31:0

#define LW902D_SET_FALCON08                                                                                0x0900
#define LW902D_SET_FALCON08_V                                                                                31:0

#define LW902D_SET_FALCON09                                                                                0x0904
#define LW902D_SET_FALCON09_V                                                                                31:0

#define LW902D_SET_FALCON10                                                                                0x0908
#define LW902D_SET_FALCON10_V                                                                                31:0

#define LW902D_SET_FALCON11                                                                                0x090c
#define LW902D_SET_FALCON11_V                                                                                31:0

#define LW902D_SET_FALCON12                                                                                0x0910
#define LW902D_SET_FALCON12_V                                                                                31:0

#define LW902D_SET_FALCON13                                                                                0x0914
#define LW902D_SET_FALCON13_V                                                                                31:0

#define LW902D_SET_FALCON14                                                                                0x0918
#define LW902D_SET_FALCON14_V                                                                                31:0

#define LW902D_SET_FALCON15                                                                                0x091c
#define LW902D_SET_FALCON15_V                                                                                31:0

#define LW902D_SET_FALCON16                                                                                0x0920
#define LW902D_SET_FALCON16_V                                                                                31:0

#define LW902D_SET_FALCON17                                                                                0x0924
#define LW902D_SET_FALCON17_V                                                                                31:0

#define LW902D_SET_FALCON18                                                                                0x0928
#define LW902D_SET_FALCON18_V                                                                                31:0

#define LW902D_SET_FALCON19                                                                                0x092c
#define LW902D_SET_FALCON19_V                                                                                31:0

#define LW902D_SET_FALCON20                                                                                0x0930
#define LW902D_SET_FALCON20_V                                                                                31:0

#define LW902D_SET_FALCON21                                                                                0x0934
#define LW902D_SET_FALCON21_V                                                                                31:0

#define LW902D_SET_FALCON22                                                                                0x0938
#define LW902D_SET_FALCON22_V                                                                                31:0

#define LW902D_SET_FALCON23                                                                                0x093c
#define LW902D_SET_FALCON23_V                                                                                31:0

#define LW902D_SET_FALCON24                                                                                0x0940
#define LW902D_SET_FALCON24_V                                                                                31:0

#define LW902D_SET_FALCON25                                                                                0x0944
#define LW902D_SET_FALCON25_V                                                                                31:0

#define LW902D_SET_FALCON26                                                                                0x0948
#define LW902D_SET_FALCON26_V                                                                                31:0

#define LW902D_SET_FALCON27                                                                                0x094c
#define LW902D_SET_FALCON27_V                                                                                31:0

#define LW902D_SET_FALCON28                                                                                0x0950
#define LW902D_SET_FALCON28_V                                                                                31:0

#define LW902D_SET_FALCON29                                                                                0x0954
#define LW902D_SET_FALCON29_V                                                                                31:0

#define LW902D_SET_FALCON30                                                                                0x0958
#define LW902D_SET_FALCON30_V                                                                                31:0

#define LW902D_SET_FALCON31                                                                                0x095c
#define LW902D_SET_FALCON31_V                                                                                31:0

#define LW902D_MME_DMA_WRITE_METHOD_BARRIER                                                                0x0dec
#define LW902D_MME_DMA_WRITE_METHOD_BARRIER_V                                                                 0:0

#define LW902D_SET_MME_SHADOW_SCRATCH(i)                                                           (0x3400+(i)*4)
#define LW902D_SET_MME_SHADOW_SCRATCH_V                                                                      31:0

#define LW902D_CALL_MME_MACRO(j)                                                                   (0x3800+(j)*8)
#define LW902D_CALL_MME_MACRO_V                                                                              31:0

#define LW902D_CALL_MME_DATA(j)                                                                    (0x3804+(j)*8)
#define LW902D_CALL_MME_DATA_V                                                                               31:0

#endif /* _cl_fermi_twod_a_h_ */
