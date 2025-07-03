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

#ifndef _cl_volta_compute_b_h_
#define _cl_volta_compute_b_h_

/* AUTO GENERATED FILE -- DO NOT EDIT */
/* Command: ../../../../class/bin/sw_header.pl volta_compute_b */

#include "lwtypes.h"

#define VOLTA_COMPUTE_B    0xC4C0

typedef volatile struct volta_compute_b_struct {
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
    LwU32 FeAtomicSequenceBegin;
    LwU32 FeAtomicSequenceEnd;
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
    LwU32 SetSmScgControl;
    LwU32 Reserved_0x1EC[0x1];
    LwU32 SetI2mSpareNoop00;
    LwU32 SetI2mSpareNoop01;
    LwU32 SetI2mSpareNoop02;
    LwU32 SetI2mSpareNoop03;
    LwU32 SetValidSpanOverflowAreaA;
    LwU32 SetValidSpanOverflowAreaB;
    LwU32 SetValidSpanOverflowAreaC;
    LwU32 Reserved_0x20C[0x1];
    LwU32 PerfmonTransfer;
    LwU32 SetQmdVirtualizationBaseA;
    LwU32 SetQmdVirtualizationBaseB;
    LwU32 IlwalidateShaderCaches;
    LwU32 SetReservedSwMethod00;
    LwU32 SetReservedSwMethod01;
    LwU32 SetReservedSwMethod02;
    LwU32 SetReservedSwMethod03;
    LwU32 SetReservedSwMethod04;
    LwU32 SetReservedSwMethod05;
    LwU32 SetReservedSwMethod06;
    LwU32 SetReservedSwMethod07;
    LwU32 Reserved_0x240[0x1];
    LwU32 IlwalidateTextureHeaderCacheNoWfi;
    LwU32 SetCwdRefCounter;
    LwU32 SetReservedSwMethod08;
    LwU32 SetReservedSwMethod09;
    LwU32 SetReservedSwMethod10;
    LwU32 SetReservedSwMethod11;
    LwU32 SetReservedSwMethod12;
    LwU32 SetReservedSwMethod13;
    LwU32 SetReservedSwMethod14;
    LwU32 SetReservedSwMethod15;
    LwU32 Reserved_0x26C[0x1];
    LwU32 SetScgControl;
    LwU32 Reserved_0x274[0x3];
    LwU32 SetComputeClassVersion;
    LwU32 CheckComputeClassVersion;
    LwU32 SetQmdVersion;
    LwU32 Reserved_0x28C[0x1];
    LwU32 CheckQmdVersion;
    LwU32 Reserved_0x294[0x1];
    LwU32 IlwalidateSkedCaches;
    LwU32 SetQmdVirtualizationControl;
    LwU32 SetShaderSharedMemoryWindowA;
    LwU32 SetShaderSharedMemoryWindowB;
    LwU32 ScgHysteresisControl;
    LwU32 Reserved_0x2AC[0x1];
    LwU32 SetCwdSlotCount;
    LwU32 SendPcasA;
    LwU32 SendPcasB;
    LwU32 SendSignalingPcasB;
    LwU32 Reserved_0x2C0[0x3];
    LwU32 SetSkedCacheControl;
    LwU32 Reserved_0x2D0[0x5];
    LwU32 SetShaderLocalMemoryNonThrottledA;
    LwU32 SetShaderLocalMemoryNonThrottledB;
    LwU32 SetShaderLocalMemoryNonThrottledC;
    LwU32 Reserved_0x2F0[0x8];
    LwU32 SetSpaVersion;
    LwU32 Reserved_0x314[0x1];
    LwU32 SetInlineQmdAddressA;
    LwU32 SetInlineQmdAddressB;
    LwU32 LoadInlineQmdData[0x40];
    LwU32 Reserved_0x420[0x38];
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
    LwU32 Reserved_0x580[0x84];
    LwU32 SetShaderLocalMemoryA;
    LwU32 SetShaderLocalMemoryB;
    LwU32 Reserved_0x798[0x6];
    LwU32 SetShaderLocalMemoryWindowA;
    LwU32 SetShaderLocalMemoryWindowB;
    LwU32 Reserved_0x7B8[0x177];
    LwU32 SetShaderCacheControl;
    LwU32 Reserved_0xD98[0x13];
    LwU32 SetSmTimeoutInterval;
    LwU32 Reserved_0xDE8[0x109];
    LwU32 IlwalidateSamplerCacheAll;
    LwU32 IlwalidateTextureHeaderCacheAll;
    LwU32 Reserved_0x1214[0x1D];
    LwU32 IlwalidateTextureDataCacheNoWfi;
    LwU32 Reserved_0x128C[0x7];
    LwU32 ActivatePerfSettingsForComputeContext;
    LwU32 Reserved_0x12AC[0x21];
    LwU32 IlwalidateSamplerCache;
    LwU32 IlwalidateTextureHeaderCache;
    LwU32 IlwalidateTextureDataCache;
    LwU32 Reserved_0x133C[0x3A];
    LwU32 IlwalidateSamplerCacheNoWfi;
    LwU32 Reserved_0x1428[0x40];
    LwU32 SetShaderExceptions;
    LwU32 Reserved_0x152C[0x9];
    LwU32 SetRenderEnableA;
    LwU32 SetRenderEnableB;
    LwU32 SetRenderEnableC;
    LwU32 SetTexSamplerPoolA;
    LwU32 SetTexSamplerPoolB;
    LwU32 SetTexSamplerPoolC;
    LwU32 Reserved_0x1568[0x3];
    LwU32 SetTexHeaderPoolA;
    LwU32 SetTexHeaderPoolB;
    LwU32 SetTexHeaderPoolC;
    LwU32 Reserved_0x1580[0x46];
    LwU32 IlwalidateShaderCachesNoWfi;
    LwU32 Reserved_0x169C[0xAA];
    LwU32 SetRenderEnableOverride;
    LwU32 Reserved_0x1948[0x37];
    LwU32 TestForQuadro;
    LwU32 Reserved_0x1A28[0x1];
    LwU32 PipeNop;
    LwU32 SetSpare00;
    LwU32 SetSpare01;
    LwU32 SetSpare02;
    LwU32 SetSpare03;
    LwU32 Reserved_0x1A40[0x30];
    LwU32 SetReportSemaphoreA;
    LwU32 SetReportSemaphoreB;
    LwU32 SetReportSemaphoreC;
    LwU32 SetReportSemaphoreD;
    LwU32 Reserved_0x1B10[0x2BA];
    LwU32 SetTrapHandlerA;
    LwU32 SetTrapHandlerB;
    LwU32 Reserved_0x2600[0x2];
    LwU32 SetBindlessTexture;
    LwU32 Reserved_0x260C[0x33A];
    LwU32 SetShaderPerformanceSnapshotCounterValue[0x8];
    LwU32 SetShaderPerformanceSnapshotCounterValueUpper[0x8];
    LwU32 EnableShaderPerformanceSnapshotCounter;
    LwU32 DisableShaderPerformanceSnapshotCounter;
    LwU32 SetShaderPerformanceCounterValueUpper[0x8];
    LwU32 SetShaderPerformanceCounterValue[0x8];
    LwU32 SetShaderPerformanceCounterEvent[0x8];
    LwU32 SetShaderPerformanceCounterControlA[0x8];
    LwU32 SetShaderPerformanceCounterControlB[0x8];
    LwU32 SetShaderPerformanceCounterTrapControl;
    LwU32 StartShaderPerformanceCounter;
    LwU32 StopShaderPerformanceCounter;
    LwU32 SetShaderPerformanceCounterSctlFilter;
    LwU32 SetShaderPerformanceCounterCoreMioFilter;
    LwU32 Reserved_0x33F0[0x4];
    LwU32 SetMmeShadowScratch[0x8];
} volta_compute_b_t;


#define LWC4C0_SET_OBJECT                                                                                  0x0000
#define LWC4C0_SET_OBJECT_CLASS_ID                                                                           15:0
#define LWC4C0_SET_OBJECT_ENGINE_ID                                                                         20:16

#define LWC4C0_NO_OPERATION                                                                                0x0100
#define LWC4C0_NO_OPERATION_V                                                                                31:0

#define LWC4C0_SET_NOTIFY_A                                                                                0x0104
#define LWC4C0_SET_NOTIFY_A_ADDRESS_UPPER                                                                     7:0

#define LWC4C0_SET_NOTIFY_B                                                                                0x0108
#define LWC4C0_SET_NOTIFY_B_ADDRESS_LOWER                                                                    31:0

#define LWC4C0_NOTIFY                                                                                      0x010c
#define LWC4C0_NOTIFY_TYPE                                                                                   31:0
#define LWC4C0_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define LWC4C0_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define LWC4C0_WAIT_FOR_IDLE                                                                               0x0110
#define LWC4C0_WAIT_FOR_IDLE_V                                                                               31:0

#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_A                                                                  0x0130
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_A_OFFSET_UPPER                                                        7:0

#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_B                                                                  0x0134
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_B_OFFSET_LOWER                                                       31:0

#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_C                                                                  0x0138
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_C_MODE                                                                2:0
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_C_MODE_FALSE                                                   0x00000000
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_C_MODE_TRUE                                                    0x00000001
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_C_MODE_CONDITIONAL                                             0x00000002
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                         0x00000003
#define LWC4C0_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                     0x00000004

#define LWC4C0_SEND_GO_IDLE                                                                                0x013c
#define LWC4C0_SEND_GO_IDLE_V                                                                                31:0

#define LWC4C0_PM_TRIGGER                                                                                  0x0140
#define LWC4C0_PM_TRIGGER_V                                                                                  31:0

#define LWC4C0_PM_TRIGGER_WFI                                                                              0x0144
#define LWC4C0_PM_TRIGGER_WFI_V                                                                              31:0

#define LWC4C0_FE_ATOMIC_SEQUENCE_BEGIN                                                                    0x0148
#define LWC4C0_FE_ATOMIC_SEQUENCE_BEGIN_V                                                                    31:0

#define LWC4C0_FE_ATOMIC_SEQUENCE_END                                                                      0x014c
#define LWC4C0_FE_ATOMIC_SEQUENCE_END_V                                                                      31:0

#define LWC4C0_SET_INSTRUMENTATION_METHOD_HEADER                                                           0x0150
#define LWC4C0_SET_INSTRUMENTATION_METHOD_HEADER_V                                                           31:0

#define LWC4C0_SET_INSTRUMENTATION_METHOD_DATA                                                             0x0154
#define LWC4C0_SET_INSTRUMENTATION_METHOD_DATA_V                                                             31:0

#define LWC4C0_LINE_LENGTH_IN                                                                              0x0180
#define LWC4C0_LINE_LENGTH_IN_VALUE                                                                          31:0

#define LWC4C0_LINE_COUNT                                                                                  0x0184
#define LWC4C0_LINE_COUNT_VALUE                                                                              31:0

#define LWC4C0_OFFSET_OUT_UPPER                                                                            0x0188
#define LWC4C0_OFFSET_OUT_UPPER_VALUE                                                                        16:0

#define LWC4C0_OFFSET_OUT                                                                                  0x018c
#define LWC4C0_OFFSET_OUT_VALUE                                                                              31:0

#define LWC4C0_PITCH_OUT                                                                                   0x0190
#define LWC4C0_PITCH_OUT_VALUE                                                                               31:0

#define LWC4C0_SET_DST_BLOCK_SIZE                                                                          0x0194
#define LWC4C0_SET_DST_BLOCK_SIZE_WIDTH                                                                       3:0
#define LWC4C0_SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB                                                        0x00000000
#define LWC4C0_SET_DST_BLOCK_SIZE_HEIGHT                                                                      7:4
#define LWC4C0_SET_DST_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LWC4C0_SET_DST_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LWC4C0_SET_DST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LWC4C0_SET_DST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LWC4C0_SET_DST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LWC4C0_SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LWC4C0_SET_DST_BLOCK_SIZE_DEPTH                                                                      11:8
#define LWC4C0_SET_DST_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LWC4C0_SET_DST_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LWC4C0_SET_DST_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LWC4C0_SET_DST_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LWC4C0_SET_DST_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LWC4C0_SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LWC4C0_SET_DST_WIDTH                                                                               0x0198
#define LWC4C0_SET_DST_WIDTH_V                                                                               31:0

#define LWC4C0_SET_DST_HEIGHT                                                                              0x019c
#define LWC4C0_SET_DST_HEIGHT_V                                                                              31:0

#define LWC4C0_SET_DST_DEPTH                                                                               0x01a0
#define LWC4C0_SET_DST_DEPTH_V                                                                               31:0

#define LWC4C0_SET_DST_LAYER                                                                               0x01a4
#define LWC4C0_SET_DST_LAYER_V                                                                               31:0

#define LWC4C0_SET_DST_ORIGIN_BYTES_X                                                                      0x01a8
#define LWC4C0_SET_DST_ORIGIN_BYTES_X_V                                                                      20:0

#define LWC4C0_SET_DST_ORIGIN_SAMPLES_Y                                                                    0x01ac
#define LWC4C0_SET_DST_ORIGIN_SAMPLES_Y_V                                                                    16:0

#define LWC4C0_LAUNCH_DMA                                                                                  0x01b0
#define LWC4C0_LAUNCH_DMA_DST_MEMORY_LAYOUT                                                                   0:0
#define LWC4C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR                                                0x00000000
#define LWC4C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH                                                      0x00000001
#define LWC4C0_LAUNCH_DMA_COMPLETION_TYPE                                                                     5:4
#define LWC4C0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_DISABLE                                                0x00000000
#define LWC4C0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY                                                   0x00000001
#define LWC4C0_LAUNCH_DMA_COMPLETION_TYPE_RELEASE_SEMAPHORE                                            0x00000002
#define LWC4C0_LAUNCH_DMA_INTERRUPT_TYPE                                                                      9:8
#define LWC4C0_LAUNCH_DMA_INTERRUPT_TYPE_NONE                                                          0x00000000
#define LWC4C0_LAUNCH_DMA_INTERRUPT_TYPE_INTERRUPT                                                     0x00000001
#define LWC4C0_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE                                                             12:12
#define LWC4C0_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_FOUR_WORDS                                             0x00000000
#define LWC4C0_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_ONE_WORD                                               0x00000001
#define LWC4C0_LAUNCH_DMA_REDUCTION_ENABLE                                                                    1:1
#define LWC4C0_LAUNCH_DMA_REDUCTION_ENABLE_FALSE                                                       0x00000000
#define LWC4C0_LAUNCH_DMA_REDUCTION_ENABLE_TRUE                                                        0x00000001
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP                                                                      15:13
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_ADD                                                         0x00000000
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_MIN                                                         0x00000001
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_MAX                                                         0x00000002
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_INC                                                         0x00000003
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_DEC                                                         0x00000004
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_AND                                                         0x00000005
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_OR                                                          0x00000006
#define LWC4C0_LAUNCH_DMA_REDUCTION_OP_RED_XOR                                                         0x00000007
#define LWC4C0_LAUNCH_DMA_REDUCTION_FORMAT                                                                    3:2
#define LWC4C0_LAUNCH_DMA_REDUCTION_FORMAT_UNSIGNED_32                                                 0x00000000
#define LWC4C0_LAUNCH_DMA_REDUCTION_FORMAT_SIGNED_32                                                   0x00000001
#define LWC4C0_LAUNCH_DMA_SYSMEMBAR_DISABLE                                                                   6:6
#define LWC4C0_LAUNCH_DMA_SYSMEMBAR_DISABLE_FALSE                                                      0x00000000
#define LWC4C0_LAUNCH_DMA_SYSMEMBAR_DISABLE_TRUE                                                       0x00000001

#define LWC4C0_LOAD_INLINE_DATA                                                                            0x01b4
#define LWC4C0_LOAD_INLINE_DATA_V                                                                            31:0

#define LWC4C0_SET_I2M_SEMAPHORE_A                                                                         0x01dc
#define LWC4C0_SET_I2M_SEMAPHORE_A_OFFSET_UPPER                                                               7:0

#define LWC4C0_SET_I2M_SEMAPHORE_B                                                                         0x01e0
#define LWC4C0_SET_I2M_SEMAPHORE_B_OFFSET_LOWER                                                              31:0

#define LWC4C0_SET_I2M_SEMAPHORE_C                                                                         0x01e4
#define LWC4C0_SET_I2M_SEMAPHORE_C_PAYLOAD                                                                   31:0

#define LWC4C0_SET_SM_SCG_CONTROL                                                                          0x01e8
#define LWC4C0_SET_SM_SCG_CONTROL_COMPUTE_IN_GRAPHICS                                                         0:0
#define LWC4C0_SET_SM_SCG_CONTROL_COMPUTE_IN_GRAPHICS_FALSE                                            0x00000000
#define LWC4C0_SET_SM_SCG_CONTROL_COMPUTE_IN_GRAPHICS_TRUE                                             0x00000001

#define LWC4C0_SET_I2M_SPARE_NOOP00                                                                        0x01f0
#define LWC4C0_SET_I2M_SPARE_NOOP00_V                                                                        31:0

#define LWC4C0_SET_I2M_SPARE_NOOP01                                                                        0x01f4
#define LWC4C0_SET_I2M_SPARE_NOOP01_V                                                                        31:0

#define LWC4C0_SET_I2M_SPARE_NOOP02                                                                        0x01f8
#define LWC4C0_SET_I2M_SPARE_NOOP02_V                                                                        31:0

#define LWC4C0_SET_I2M_SPARE_NOOP03                                                                        0x01fc
#define LWC4C0_SET_I2M_SPARE_NOOP03_V                                                                        31:0

#define LWC4C0_SET_VALID_SPAN_OVERFLOW_AREA_A                                                              0x0200
#define LWC4C0_SET_VALID_SPAN_OVERFLOW_AREA_A_ADDRESS_UPPER                                                   7:0

#define LWC4C0_SET_VALID_SPAN_OVERFLOW_AREA_B                                                              0x0204
#define LWC4C0_SET_VALID_SPAN_OVERFLOW_AREA_B_ADDRESS_LOWER                                                  31:0

#define LWC4C0_SET_VALID_SPAN_OVERFLOW_AREA_C                                                              0x0208
#define LWC4C0_SET_VALID_SPAN_OVERFLOW_AREA_C_SIZE                                                           31:0

#define LWC4C0_PERFMON_TRANSFER                                                                            0x0210
#define LWC4C0_PERFMON_TRANSFER_V                                                                            31:0

#define LWC4C0_SET_QMD_VIRTUALIZATION_BASE_A                                                               0x0214
#define LWC4C0_SET_QMD_VIRTUALIZATION_BASE_A_ADDRESS_UPPER                                                    7:0

#define LWC4C0_SET_QMD_VIRTUALIZATION_BASE_B                                                               0x0218
#define LWC4C0_SET_QMD_VIRTUALIZATION_BASE_B_ADDRESS_LOWER                                                   31:0

#define LWC4C0_ILWALIDATE_SHADER_CACHES                                                                    0x021c
#define LWC4C0_ILWALIDATE_SHADER_CACHES_INSTRUCTION                                                           0:0
#define LWC4C0_ILWALIDATE_SHADER_CACHES_INSTRUCTION_FALSE                                              0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_INSTRUCTION_TRUE                                               0x00000001
#define LWC4C0_ILWALIDATE_SHADER_CACHES_DATA                                                                  4:4
#define LWC4C0_ILWALIDATE_SHADER_CACHES_DATA_FALSE                                                     0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_DATA_TRUE                                                      0x00000001
#define LWC4C0_ILWALIDATE_SHADER_CACHES_CONSTANT                                                            12:12
#define LWC4C0_ILWALIDATE_SHADER_CACHES_CONSTANT_FALSE                                                 0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_CONSTANT_TRUE                                                  0x00000001
#define LWC4C0_ILWALIDATE_SHADER_CACHES_LOCKS                                                                 1:1
#define LWC4C0_ILWALIDATE_SHADER_CACHES_LOCKS_FALSE                                                    0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_LOCKS_TRUE                                                     0x00000001
#define LWC4C0_ILWALIDATE_SHADER_CACHES_FLUSH_DATA                                                            2:2
#define LWC4C0_ILWALIDATE_SHADER_CACHES_FLUSH_DATA_FALSE                                               0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_FLUSH_DATA_TRUE                                                0x00000001

#define LWC4C0_SET_RESERVED_SW_METHOD00                                                                    0x0220
#define LWC4C0_SET_RESERVED_SW_METHOD00_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD01                                                                    0x0224
#define LWC4C0_SET_RESERVED_SW_METHOD01_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD02                                                                    0x0228
#define LWC4C0_SET_RESERVED_SW_METHOD02_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD03                                                                    0x022c
#define LWC4C0_SET_RESERVED_SW_METHOD03_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD04                                                                    0x0230
#define LWC4C0_SET_RESERVED_SW_METHOD04_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD05                                                                    0x0234
#define LWC4C0_SET_RESERVED_SW_METHOD05_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD06                                                                    0x0238
#define LWC4C0_SET_RESERVED_SW_METHOD06_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD07                                                                    0x023c
#define LWC4C0_SET_RESERVED_SW_METHOD07_V                                                                    31:0

#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI                                                      0x0244
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES                                                   0:0
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ALL                                        0x00000000
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ONE                                        0x00000001
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_TAG                                                    25:4

#define LWC4C0_SET_CWD_REF_COUNTER                                                                         0x0248
#define LWC4C0_SET_CWD_REF_COUNTER_SELECT                                                                     5:0
#define LWC4C0_SET_CWD_REF_COUNTER_VALUE                                                                     23:8

#define LWC4C0_SET_RESERVED_SW_METHOD08                                                                    0x024c
#define LWC4C0_SET_RESERVED_SW_METHOD08_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD09                                                                    0x0250
#define LWC4C0_SET_RESERVED_SW_METHOD09_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD10                                                                    0x0254
#define LWC4C0_SET_RESERVED_SW_METHOD10_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD11                                                                    0x0258
#define LWC4C0_SET_RESERVED_SW_METHOD11_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD12                                                                    0x025c
#define LWC4C0_SET_RESERVED_SW_METHOD12_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD13                                                                    0x0260
#define LWC4C0_SET_RESERVED_SW_METHOD13_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD14                                                                    0x0264
#define LWC4C0_SET_RESERVED_SW_METHOD14_V                                                                    31:0

#define LWC4C0_SET_RESERVED_SW_METHOD15                                                                    0x0268
#define LWC4C0_SET_RESERVED_SW_METHOD15_V                                                                    31:0

#define LWC4C0_SET_SCG_CONTROL                                                                             0x0270
#define LWC4C0_SET_SCG_CONTROL_COMPUTE1_MAX_SM_COUNT                                                          8:0
#define LWC4C0_SET_SCG_CONTROL_COMPUTE1_MIN_SM_COUNT                                                        20:12
#define LWC4C0_SET_SCG_CONTROL_DISABLE_COMPUTE1_LIMIT_IN_ALL_COMPUTE                                        24:24
#define LWC4C0_SET_SCG_CONTROL_DISABLE_COMPUTE1_LIMIT_IN_ALL_COMPUTE_FALSE                             0x00000000
#define LWC4C0_SET_SCG_CONTROL_DISABLE_COMPUTE1_LIMIT_IN_ALL_COMPUTE_TRUE                              0x00000001

#define LWC4C0_SET_COMPUTE_CLASS_VERSION                                                                   0x0280
#define LWC4C0_SET_COMPUTE_CLASS_VERSION_LWRRENT                                                             15:0
#define LWC4C0_SET_COMPUTE_CLASS_VERSION_OLDEST_SUPPORTED                                                   31:16

#define LWC4C0_CHECK_COMPUTE_CLASS_VERSION                                                                 0x0284
#define LWC4C0_CHECK_COMPUTE_CLASS_VERSION_LWRRENT                                                           15:0
#define LWC4C0_CHECK_COMPUTE_CLASS_VERSION_OLDEST_SUPPORTED                                                 31:16

#define LWC4C0_SET_QMD_VERSION                                                                             0x0288
#define LWC4C0_SET_QMD_VERSION_LWRRENT                                                                       15:0
#define LWC4C0_SET_QMD_VERSION_OLDEST_SUPPORTED                                                             31:16

#define LWC4C0_CHECK_QMD_VERSION                                                                           0x0290
#define LWC4C0_CHECK_QMD_VERSION_LWRRENT                                                                     15:0
#define LWC4C0_CHECK_QMD_VERSION_OLDEST_SUPPORTED                                                           31:16

#define LWC4C0_ILWALIDATE_SKED_CACHES                                                                      0x0298
#define LWC4C0_ILWALIDATE_SKED_CACHES_V                                                                       0:0

#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL                                                              0x029c
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_CONSTANT_BUFFER_MASK                                            7:0
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_I2M_ADDR_ENABLE                                                 8:8
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_I2M_ADDR_ENABLE_FALSE                                    0x00000000
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_I2M_ADDR_ENABLE_TRUE                                     0x00000001
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_I2M_CONSTANT_BUFFER_ENABLE                                    12:12
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_I2M_CONSTANT_BUFFER_ENABLE_FALSE                         0x00000000
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_I2M_CONSTANT_BUFFER_ENABLE_TRUE                          0x00000001
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_IQ2M_ADDR_ENABLE                                              16:16
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_IQ2M_ADDR_ENABLE_FALSE                                   0x00000000
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_IQ2M_ADDR_ENABLE_TRUE                                    0x00000001
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_IQ2M_CONSTANT_BUFFER_ENABLE                                   20:20
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_IQ2M_CONSTANT_BUFFER_ENABLE_FALSE                        0x00000000
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_IQ2M_CONSTANT_BUFFER_ENABLE_TRUE                         0x00000001
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE                                              24:24
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE_FALSE                                   0x00000000
#define LWC4C0_SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE_TRUE                                    0x00000001

#define LWC4C0_SET_SHADER_SHARED_MEMORY_WINDOW_A                                                           0x02a0
#define LWC4C0_SET_SHADER_SHARED_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER                                          16:0

#define LWC4C0_SET_SHADER_SHARED_MEMORY_WINDOW_B                                                           0x02a4
#define LWC4C0_SET_SHADER_SHARED_MEMORY_WINDOW_B_BASE_ADDRESS                                                31:0

#define LWC4C0_SCG_HYSTERESIS_CONTROL                                                                      0x02a8
#define LWC4C0_SCG_HYSTERESIS_CONTROL_USE_TIMEOUT_ONCE                                                        0:0
#define LWC4C0_SCG_HYSTERESIS_CONTROL_USE_TIMEOUT_ONCE_FALSE                                           0x00000000
#define LWC4C0_SCG_HYSTERESIS_CONTROL_USE_TIMEOUT_ONCE_TRUE                                            0x00000001
#define LWC4C0_SCG_HYSTERESIS_CONTROL_USE_NULL_TIMEOUT_ONCE                                                   1:1
#define LWC4C0_SCG_HYSTERESIS_CONTROL_USE_NULL_TIMEOUT_ONCE_FALSE                                      0x00000000
#define LWC4C0_SCG_HYSTERESIS_CONTROL_USE_NULL_TIMEOUT_ONCE_TRUE                                       0x00000001

#define LWC4C0_SET_CWD_SLOT_COUNT                                                                          0x02b0
#define LWC4C0_SET_CWD_SLOT_COUNT_V                                                                           7:0

#define LWC4C0_SEND_PCAS_A                                                                                 0x02b4
#define LWC4C0_SEND_PCAS_A_QMD_ADDRESS_SHIFTED8                                                              31:0

#define LWC4C0_SEND_PCAS_B                                                                                 0x02b8
#define LWC4C0_SEND_PCAS_B_FROM                                                                              23:0
#define LWC4C0_SEND_PCAS_B_DELTA                                                                            31:24

#define LWC4C0_SEND_SIGNALING_PCAS_B                                                                       0x02bc
#define LWC4C0_SEND_SIGNALING_PCAS_B_ILWALIDATE                                                               0:0
#define LWC4C0_SEND_SIGNALING_PCAS_B_ILWALIDATE_FALSE                                                  0x00000000
#define LWC4C0_SEND_SIGNALING_PCAS_B_ILWALIDATE_TRUE                                                   0x00000001
#define LWC4C0_SEND_SIGNALING_PCAS_B_SCHEDULE                                                                 1:1
#define LWC4C0_SEND_SIGNALING_PCAS_B_SCHEDULE_FALSE                                                    0x00000000
#define LWC4C0_SEND_SIGNALING_PCAS_B_SCHEDULE_TRUE                                                     0x00000001

#define LWC4C0_SET_SKED_CACHE_CONTROL                                                                      0x02cc
#define LWC4C0_SET_SKED_CACHE_CONTROL_IGNORE_VEID                                                             0:0
#define LWC4C0_SET_SKED_CACHE_CONTROL_IGNORE_VEID_FALSE                                                0x00000000
#define LWC4C0_SET_SKED_CACHE_CONTROL_IGNORE_VEID_TRUE                                                 0x00000001

#define LWC4C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A                                                     0x02e4
#define LWC4C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A_SIZE_UPPER                                             7:0

#define LWC4C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B                                                     0x02e8
#define LWC4C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B_SIZE_LOWER                                            31:0

#define LWC4C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C                                                     0x02ec
#define LWC4C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C_MAX_SM_COUNT                                           8:0

#define LWC4C0_SET_SPA_VERSION                                                                             0x0310
#define LWC4C0_SET_SPA_VERSION_MINOR                                                                          7:0
#define LWC4C0_SET_SPA_VERSION_MAJOR                                                                         15:8

#define LWC4C0_SET_INLINE_QMD_ADDRESS_A                                                                    0x0318
#define LWC4C0_SET_INLINE_QMD_ADDRESS_A_QMD_ADDRESS_SHIFTED8_UPPER                                           31:0

#define LWC4C0_SET_INLINE_QMD_ADDRESS_B                                                                    0x031c
#define LWC4C0_SET_INLINE_QMD_ADDRESS_B_QMD_ADDRESS_SHIFTED8_LOWER                                           31:0

#define LWC4C0_LOAD_INLINE_QMD_DATA(i)                                                             (0x0320+(i)*4)
#define LWC4C0_LOAD_INLINE_QMD_DATA_V                                                                        31:0

#define LWC4C0_SET_FALCON00                                                                                0x0500
#define LWC4C0_SET_FALCON00_V                                                                                31:0

#define LWC4C0_SET_FALCON01                                                                                0x0504
#define LWC4C0_SET_FALCON01_V                                                                                31:0

#define LWC4C0_SET_FALCON02                                                                                0x0508
#define LWC4C0_SET_FALCON02_V                                                                                31:0

#define LWC4C0_SET_FALCON03                                                                                0x050c
#define LWC4C0_SET_FALCON03_V                                                                                31:0

#define LWC4C0_SET_FALCON04                                                                                0x0510
#define LWC4C0_SET_FALCON04_V                                                                                31:0

#define LWC4C0_SET_FALCON05                                                                                0x0514
#define LWC4C0_SET_FALCON05_V                                                                                31:0

#define LWC4C0_SET_FALCON06                                                                                0x0518
#define LWC4C0_SET_FALCON06_V                                                                                31:0

#define LWC4C0_SET_FALCON07                                                                                0x051c
#define LWC4C0_SET_FALCON07_V                                                                                31:0

#define LWC4C0_SET_FALCON08                                                                                0x0520
#define LWC4C0_SET_FALCON08_V                                                                                31:0

#define LWC4C0_SET_FALCON09                                                                                0x0524
#define LWC4C0_SET_FALCON09_V                                                                                31:0

#define LWC4C0_SET_FALCON10                                                                                0x0528
#define LWC4C0_SET_FALCON10_V                                                                                31:0

#define LWC4C0_SET_FALCON11                                                                                0x052c
#define LWC4C0_SET_FALCON11_V                                                                                31:0

#define LWC4C0_SET_FALCON12                                                                                0x0530
#define LWC4C0_SET_FALCON12_V                                                                                31:0

#define LWC4C0_SET_FALCON13                                                                                0x0534
#define LWC4C0_SET_FALCON13_V                                                                                31:0

#define LWC4C0_SET_FALCON14                                                                                0x0538
#define LWC4C0_SET_FALCON14_V                                                                                31:0

#define LWC4C0_SET_FALCON15                                                                                0x053c
#define LWC4C0_SET_FALCON15_V                                                                                31:0

#define LWC4C0_SET_FALCON16                                                                                0x0540
#define LWC4C0_SET_FALCON16_V                                                                                31:0

#define LWC4C0_SET_FALCON17                                                                                0x0544
#define LWC4C0_SET_FALCON17_V                                                                                31:0

#define LWC4C0_SET_FALCON18                                                                                0x0548
#define LWC4C0_SET_FALCON18_V                                                                                31:0

#define LWC4C0_SET_FALCON19                                                                                0x054c
#define LWC4C0_SET_FALCON19_V                                                                                31:0

#define LWC4C0_SET_FALCON20                                                                                0x0550
#define LWC4C0_SET_FALCON20_V                                                                                31:0

#define LWC4C0_SET_FALCON21                                                                                0x0554
#define LWC4C0_SET_FALCON21_V                                                                                31:0

#define LWC4C0_SET_FALCON22                                                                                0x0558
#define LWC4C0_SET_FALCON22_V                                                                                31:0

#define LWC4C0_SET_FALCON23                                                                                0x055c
#define LWC4C0_SET_FALCON23_V                                                                                31:0

#define LWC4C0_SET_FALCON24                                                                                0x0560
#define LWC4C0_SET_FALCON24_V                                                                                31:0

#define LWC4C0_SET_FALCON25                                                                                0x0564
#define LWC4C0_SET_FALCON25_V                                                                                31:0

#define LWC4C0_SET_FALCON26                                                                                0x0568
#define LWC4C0_SET_FALCON26_V                                                                                31:0

#define LWC4C0_SET_FALCON27                                                                                0x056c
#define LWC4C0_SET_FALCON27_V                                                                                31:0

#define LWC4C0_SET_FALCON28                                                                                0x0570
#define LWC4C0_SET_FALCON28_V                                                                                31:0

#define LWC4C0_SET_FALCON29                                                                                0x0574
#define LWC4C0_SET_FALCON29_V                                                                                31:0

#define LWC4C0_SET_FALCON30                                                                                0x0578
#define LWC4C0_SET_FALCON30_V                                                                                31:0

#define LWC4C0_SET_FALCON31                                                                                0x057c
#define LWC4C0_SET_FALCON31_V                                                                                31:0

#define LWC4C0_SET_SHADER_LOCAL_MEMORY_A                                                                   0x0790
#define LWC4C0_SET_SHADER_LOCAL_MEMORY_A_ADDRESS_UPPER                                                       16:0

#define LWC4C0_SET_SHADER_LOCAL_MEMORY_B                                                                   0x0794
#define LWC4C0_SET_SHADER_LOCAL_MEMORY_B_ADDRESS_LOWER                                                       31:0

#define LWC4C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A                                                            0x07b0
#define LWC4C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER                                           16:0

#define LWC4C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B                                                            0x07b4
#define LWC4C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B_BASE_ADDRESS                                                 31:0

#define LWC4C0_SET_SHADER_CACHE_CONTROL                                                                    0x0d94
#define LWC4C0_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE                                                0:0
#define LWC4C0_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_FALSE                                   0x00000000
#define LWC4C0_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_TRUE                                    0x00000001

#define LWC4C0_SET_SM_TIMEOUT_INTERVAL                                                                     0x0de4
#define LWC4C0_SET_SM_TIMEOUT_INTERVAL_COUNTER_BIT                                                            5:0

#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_ALL                                                                0x120c
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_ALL_V                                                                 0:0

#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_ALL                                                         0x1210
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_ALL_V                                                          0:0

#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI                                                        0x1288
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES                                                     0:0
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES_ALL                                          0x00000000
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES_ONE                                          0x00000001
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_NO_WFI_TAG                                                      25:4

#define LWC4C0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT                                                  0x12a8
#define LWC4C0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT_ALL                                                 0:0
#define LWC4C0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT_ALL_FALSE                                    0x00000000
#define LWC4C0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT_ALL_TRUE                                     0x00000001

#define LWC4C0_ILWALIDATE_SAMPLER_CACHE                                                                    0x1330
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_LINES                                                                 0:0
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_LINES_ALL                                                      0x00000000
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_LINES_ONE                                                      0x00000001
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_TAG                                                                  25:4

#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE                                                             0x1334
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES                                                          0:0
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES_ALL                                               0x00000000
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_LINES_ONE                                               0x00000001
#define LWC4C0_ILWALIDATE_TEXTURE_HEADER_CACHE_TAG                                                           25:4

#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE                                                               0x1338
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_LINES                                                            0:0
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_LINES_ALL                                                 0x00000000
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_LINES_ONE                                                 0x00000001
#define LWC4C0_ILWALIDATE_TEXTURE_DATA_CACHE_TAG                                                             25:4

#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_NO_WFI                                                             0x1424
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_NO_WFI_LINES                                                          0:0
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ALL                                               0x00000000
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ONE                                               0x00000001
#define LWC4C0_ILWALIDATE_SAMPLER_CACHE_NO_WFI_TAG                                                           25:4

#define LWC4C0_SET_SHADER_EXCEPTIONS                                                                       0x1528
#define LWC4C0_SET_SHADER_EXCEPTIONS_ENABLE                                                                   0:0
#define LWC4C0_SET_SHADER_EXCEPTIONS_ENABLE_FALSE                                                      0x00000000
#define LWC4C0_SET_SHADER_EXCEPTIONS_ENABLE_TRUE                                                       0x00000001

#define LWC4C0_SET_RENDER_ENABLE_A                                                                         0x1550
#define LWC4C0_SET_RENDER_ENABLE_A_OFFSET_UPPER                                                               7:0

#define LWC4C0_SET_RENDER_ENABLE_B                                                                         0x1554
#define LWC4C0_SET_RENDER_ENABLE_B_OFFSET_LOWER                                                              31:0

#define LWC4C0_SET_RENDER_ENABLE_C                                                                         0x1558
#define LWC4C0_SET_RENDER_ENABLE_C_MODE                                                                       2:0
#define LWC4C0_SET_RENDER_ENABLE_C_MODE_FALSE                                                          0x00000000
#define LWC4C0_SET_RENDER_ENABLE_C_MODE_TRUE                                                           0x00000001
#define LWC4C0_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                                                    0x00000002
#define LWC4C0_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                                0x00000003
#define LWC4C0_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                            0x00000004

#define LWC4C0_SET_TEX_SAMPLER_POOL_A                                                                      0x155c
#define LWC4C0_SET_TEX_SAMPLER_POOL_A_OFFSET_UPPER                                                           16:0

#define LWC4C0_SET_TEX_SAMPLER_POOL_B                                                                      0x1560
#define LWC4C0_SET_TEX_SAMPLER_POOL_B_OFFSET_LOWER                                                           31:0

#define LWC4C0_SET_TEX_SAMPLER_POOL_C                                                                      0x1564
#define LWC4C0_SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX                                                          19:0

#define LWC4C0_SET_TEX_HEADER_POOL_A                                                                       0x1574
#define LWC4C0_SET_TEX_HEADER_POOL_A_OFFSET_UPPER                                                            16:0

#define LWC4C0_SET_TEX_HEADER_POOL_B                                                                       0x1578
#define LWC4C0_SET_TEX_HEADER_POOL_B_OFFSET_LOWER                                                            31:0

#define LWC4C0_SET_TEX_HEADER_POOL_C                                                                       0x157c
#define LWC4C0_SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX                                                           21:0

#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI                                                             0x1698
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION                                                    0:0
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_FALSE                                       0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE                                        0x00000001
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA                                                    4:4
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_FALSE                                       0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE                                        0x00000001
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_CONSTANT                                                     12:12
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_FALSE                                          0x00000000
#define LWC4C0_ILWALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE                                           0x00000001

#define LWC4C0_SET_RENDER_ENABLE_OVERRIDE                                                                  0x1944
#define LWC4C0_SET_RENDER_ENABLE_OVERRIDE_MODE                                                                1:0
#define LWC4C0_SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE                                       0x00000000
#define LWC4C0_SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER                                           0x00000001
#define LWC4C0_SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER                                            0x00000002

#define LWC4C0_TEST_FOR_QUADRO                                                                             0x1a24
#define LWC4C0_TEST_FOR_QUADRO_V                                                                             31:0

#define LWC4C0_PIPE_NOP                                                                                    0x1a2c
#define LWC4C0_PIPE_NOP_V                                                                                    31:0

#define LWC4C0_SET_SPARE00                                                                                 0x1a30
#define LWC4C0_SET_SPARE00_V                                                                                 31:0

#define LWC4C0_SET_SPARE01                                                                                 0x1a34
#define LWC4C0_SET_SPARE01_V                                                                                 31:0

#define LWC4C0_SET_SPARE02                                                                                 0x1a38
#define LWC4C0_SET_SPARE02_V                                                                                 31:0

#define LWC4C0_SET_SPARE03                                                                                 0x1a3c
#define LWC4C0_SET_SPARE03_V                                                                                 31:0

#define LWC4C0_SET_REPORT_SEMAPHORE_A                                                                      0x1b00
#define LWC4C0_SET_REPORT_SEMAPHORE_A_OFFSET_UPPER                                                            7:0

#define LWC4C0_SET_REPORT_SEMAPHORE_B                                                                      0x1b04
#define LWC4C0_SET_REPORT_SEMAPHORE_B_OFFSET_LOWER                                                           31:0

#define LWC4C0_SET_REPORT_SEMAPHORE_C                                                                      0x1b08
#define LWC4C0_SET_REPORT_SEMAPHORE_C_PAYLOAD                                                                31:0

#define LWC4C0_SET_REPORT_SEMAPHORE_D                                                                      0x1b0c
#define LWC4C0_SET_REPORT_SEMAPHORE_D_OPERATION                                                               1:0
#define LWC4C0_SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE                                                0x00000000
#define LWC4C0_SET_REPORT_SEMAPHORE_D_OPERATION_TRAP                                                   0x00000003
#define LWC4C0_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE                                                         20:20
#define LWC4C0_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                              0x00000000
#define LWC4C0_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                               0x00000001
#define LWC4C0_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE                                                        28:28
#define LWC4C0_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS                                        0x00000000
#define LWC4C0_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD                                          0x00000001
#define LWC4C0_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE                                                           2:2
#define LWC4C0_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                              0x00000000
#define LWC4C0_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                               0x00000001
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE                                                        3:3
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE_FALSE                                           0x00000000
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE_TRUE                                            0x00000001
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP                                                           11:9
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_ADD                                             0x00000000
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_MIN                                             0x00000001
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_MAX                                             0x00000002
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_INC                                             0x00000003
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_DEC                                             0x00000004
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_AND                                             0x00000005
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_OR                                              0x00000006
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_XOR                                             0x00000007
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT                                                      18:17
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT_UNSIGNED_32                                     0x00000000
#define LWC4C0_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT_SIGNED_32                                       0x00000001

#define LWC4C0_SET_TRAP_HANDLER_A                                                                          0x25f8
#define LWC4C0_SET_TRAP_HANDLER_A_ADDRESS_UPPER                                                              16:0

#define LWC4C0_SET_TRAP_HANDLER_B                                                                          0x25fc
#define LWC4C0_SET_TRAP_HANDLER_B_ADDRESS_LOWER                                                              31:0

#define LWC4C0_SET_BINDLESS_TEXTURE                                                                        0x2608
#define LWC4C0_SET_BINDLESS_TEXTURE_CONSTANT_BUFFER_SLOT_SELECT                                               2:0

#define LWC4C0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE(i)                                    (0x32f4+(i)*4)
#define LWC4C0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_V                                               31:0

#define LWC4C0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_UPPER(i)                              (0x3314+(i)*4)
#define LWC4C0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_UPPER_V                                         31:0

#define LWC4C0_ENABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER                                                  0x3334
#define LWC4C0_ENABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_V                                                   0:0

#define LWC4C0_DISABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER                                                 0x3338
#define LWC4C0_DISABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_V                                                  0:0

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_VALUE_UPPER(i)                                       (0x333c+(i)*4)
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_VALUE_UPPER_V                                                  31:0

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_VALUE(i)                                             (0x335c+(i)*4)
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_VALUE_V                                                        31:0

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_EVENT(i)                                             (0x337c+(i)*4)
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_EVENT_EVENT                                                     7:0

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A(i)                                         (0x339c+(i)*4)
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT0                                                1:0
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT0                                           4:2
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT1                                                6:5
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT1                                           9:7
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT2                                              11:10
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT2                                         14:12
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT3                                              16:15
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT3                                         19:17
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT4                                              21:20
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT4                                         24:22
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT5                                              26:25
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT5                                         29:27
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_SPARE                                               31:30

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B(i)                                         (0x33bc+(i)*4)
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_EDGE                                                  0:0
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_MODE                                                  2:1
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_WINDOWED                                              3:3
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_FUNC                                                 19:4

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL                                                 0x33dc
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL_MASK                                               7:0

#define LWC4C0_START_SHADER_PERFORMANCE_COUNTER                                                            0x33e0
#define LWC4C0_START_SHADER_PERFORMANCE_COUNTER_COUNTER_MASK                                                  7:0

#define LWC4C0_STOP_SHADER_PERFORMANCE_COUNTER                                                             0x33e4
#define LWC4C0_STOP_SHADER_PERFORMANCE_COUNTER_COUNTER_MASK                                                   7:0

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_SCTL_FILTER                                                  0x33e8
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_SCTL_FILTER_V                                                  31:0

#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CORE_MIO_FILTER                                              0x33ec
#define LWC4C0_SET_SHADER_PERFORMANCE_COUNTER_CORE_MIO_FILTER_V                                              31:0

#define LWC4C0_SET_MME_SHADOW_SCRATCH(i)                                                           (0x3400+(i)*4)
#define LWC4C0_SET_MME_SHADOW_SCRATCH_V                                                                      31:0

#endif /* _cl_volta_compute_b_h_ */
