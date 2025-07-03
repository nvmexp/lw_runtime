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

#ifndef _cl90b3_h_
#define _cl90b3_h_

#ifdef __cplusplus
extern "C" {
#endif

#define GF100_MSPPP                                                               (0x000090B3)

typedef volatile struct _cl90b3_tag0 {
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
    LwV32 Reserved04[0x2D];
    LwV32 Execute;                                                              // 0x00000300 - 0x00000303
    LwV32 SemaphoreD;                                                           // 0x00000304 - 0x00000307
    LwV32 SetPredicationOffsetUpper;                                            // 0x00000308 - 0x0000030B
    LwV32 SetPredicationOffsetLower;                                            // 0x0000030C - 0x0000030F
    LwV32 Reserved05[0x3C];
    LwV32 H264SetFgtSeiMsgAddr;                                                 // 0x00000400 - 0x00000403
    LwV32 H264SetRngSeedY;                                                      // 0x00000404 - 0x00000407
    LwV32 H264SetRngSeedCb;                                                     // 0x00000408 - 0x0000040B
    LwV32 H264SetRngSeedCr;                                                     // 0x0000040C - 0x0000040F
    LwV32 H264SetFgtDb;                                                         // 0x00000410 - 0x00000413
    LwV32 H264SetBboxWeightSlope;                                               // 0x00000414 - 0x00000417
    LwV32 H264SetBboxOutHOffset;                                                // 0x00000418 - 0x0000041B
    LwV32 H264SetBboxOutVOffset;                                                // 0x0000041C - 0x0000041F
    LwV32 Reserved06[0x38];
    LwV32 Vc1SetControl;                                                        // 0x00000500 - 0x00000503
    LwV32 Vc1SetBboxWeightSlope;                                                // 0x00000504 - 0x00000507
    LwV32 Vc1SetBboxOutHOffset;                                                 // 0x00000508 - 0x0000050B
    LwV32 Vc1SetBboxOutVOffset;                                                 // 0x0000050C - 0x0000050F
    LwV32 Reserved07[0x7C];
    LwV32 SetControlParams1;                                                    // 0x00000700 - 0x00000703
    LwV32 SetPicParams;                                                         // 0x00000704 - 0x00000707
    LwV32 SetSrcPicTopFieldY;                                                   // 0x00000708 - 0x0000070B
    LwV32 SetSrcPicBotFieldY;                                                   // 0x0000070C - 0x0000070F
    LwV32 SetSrcPicTopFieldC;                                                   // 0x00000710 - 0x00000713
    LwV32 SetSrcPicBotFieldC;                                                   // 0x00000714 - 0x00000717
    LwV32 SetDstPicTopFieldY;                                                   // 0x00000718 - 0x0000071B
    LwV32 SetDstPicBotFieldY;                                                   // 0x0000071C - 0x0000071F
    LwV32 SetDstPicTopFieldC;                                                   // 0x00000720 - 0x00000723
    LwV32 SetDstPicBotFieldC;                                                   // 0x00000724 - 0x00000727
    LwV32 SetHcWinStart;                                                        // 0x00000728 - 0x0000072B
    LwV32 SetHcWinEnd;                                                          // 0x0000072C - 0x0000072F
    LwV32 SetHcResult;                                                          // 0x00000730 - 0x00000733
    LwV32 SetPictureIndex;                                                      // 0x00000734 - 0x00000737
    LwV32 SetControlParams2;                                                    // 0x00000738 - 0x0000073B
    LwV32 SetFlowCtrlOffset;                                                    // 0x0000073C - 0x0000073F
    LwV32 Reserved08[0x275];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved09[0x3BA];
} GF100MsdecMSPPPControlPio;

#define LW90B3_NOP                                                              (0x00000100)
#define LW90B3_NOP_PARAMETER                                                    31:0
#define LW90B3_PM_TRIGGER                                                       (0x00000140)
#define LW90B3_PM_TRIGGER_V                                                     31:0
#define LW90B3_SET_APPLICATION_ID                                               (0x00000200)
#define LW90B3_SET_APPLICATION_ID_ID                                            31:0
#define LW90B3_SET_APPLICATION_ID_ID_MPEG12_H264_BYPASS_HCONLY_BBOXONLY         (0x00000003)
#define LW90B3_SET_APPLICATION_ID_ID_VC1_COPYENGINE                             (0x00000002)
#define LW90B3_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW90B3_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LW90B3_SEMAPHORE_A                                                      (0x00000240)
#define LW90B3_SEMAPHORE_A_UPPER                                                7:0
#define LW90B3_SEMAPHORE_B                                                      (0x00000244)
#define LW90B3_SEMAPHORE_B_LOWER                                                31:0
#define LW90B3_SEMAPHORE_C                                                      (0x00000248)
#define LW90B3_SEMAPHORE_C_PAYLOAD                                              31:0
#define LW90B3_EXELWTE                                                          (0x00000300)
#define LW90B3_EXELWTE_NOTIFY                                                   0:0
#define LW90B3_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LW90B3_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LW90B3_EXELWTE_NOTIFY_ON                                                1:1
#define LW90B3_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LW90B3_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LW90B3_EXELWTE_PREDICATION                                              2:2
#define LW90B3_EXELWTE_PREDICATION_DISABLE                                      (0x00000000)
#define LW90B3_EXELWTE_PREDICATION_ENABLE                                       (0x00000001)
#define LW90B3_EXELWTE_PREDICATION_OP                                           3:3
#define LW90B3_EXELWTE_PREDICATION_OP_EQUAL_ZERO                                (0x00000000)
#define LW90B3_EXELWTE_PREDICATION_OP_NOT_EQUAL_ZERO                            (0x00000001)
#define LW90B3_EXELWTE_AWAKEN                                                   8:8
#define LW90B3_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LW90B3_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LW90B3_SEMAPHORE_D                                                      (0x00000304)
#define LW90B3_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LW90B3_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LW90B3_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LW90B3_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LW90B3_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LW90B3_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LW90B3_SEMAPHORE_D_OPERATION                                            17:16
#define LW90B3_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LW90B3_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LW90B3_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LW90B3_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LW90B3_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LW90B3_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LW90B3_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LW90B3_SET_PREDICATION_OFFSET_UPPER                                     (0x00000308)
#define LW90B3_SET_PREDICATION_OFFSET_UPPER_OFFSET                              7:0
#define LW90B3_SET_PREDICATION_OFFSET_LOWER                                     (0x0000030C)
#define LW90B3_SET_PREDICATION_OFFSET_LOWER_OFFSET                              31:0
#define LW90B3_H264_SET_FGT_SEI_MSG_ADDR                                        (0x00000400)
#define LW90B3_H264_SET_FGT_SEI_MSG_ADDR_ADDR                                   31:0
#define LW90B3_H264_SET_RNG_SEED_Y                                              (0x00000404)
#define LW90B3_H264_SET_RNG_SEED_Y_SEED                                         31:0
#define LW90B3_H264_SET_RNG_SEED_CB                                             (0x00000408)
#define LW90B3_H264_SET_RNG_SEED_CB_SEED                                        31:0
#define LW90B3_H264_SET_RNG_SEED_CR                                             (0x0000040C)
#define LW90B3_H264_SET_RNG_SEED_CR_SEED                                        31:0
#define LW90B3_H264_SET_FGT_DB                                                  (0x00000410)
#define LW90B3_H264_SET_FGT_DB_ADDR                                             31:0
#define LW90B3_H264_SET_BBOX_WEIGHT_SLOPE                                       (0x00000414)
#define LW90B3_H264_SET_BBOX_WEIGHT_SLOPE_SLOPE_X                               15:0
#define LW90B3_H264_SET_BBOX_WEIGHT_SLOPE_SLOPE_Y                               31:16
#define LW90B3_H264_SET_BBOX_OUT_HOFFSET                                        (0x00000418)
#define LW90B3_H264_SET_BBOX_OUT_HOFFSET_OFFSET                                 31:0
#define LW90B3_H264_SET_BBOX_OUT_VOFFSET                                        (0x0000041C)
#define LW90B3_H264_SET_BBOX_OUT_VOFFSET_OFFSET                                 31:0
#define LW90B3_VC1_SET_CONTROL                                                  (0x00000500)
#define LW90B3_VC1_SET_CONTROL_INTERLACED                                       0:0
#define LW90B3_VC1_SET_CONTROL_INTERLACED_FALSE                                 (0x00000000)
#define LW90B3_VC1_SET_CONTROL_INTERLACED_TRUE                                  (0x00000001)
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPUV_FLAG                             1:1
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPUV_FLAG_FALSE                       (0x00000000)
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPUV_FLAG_TRUE                        (0x00000001)
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPY_FLAG                              2:2
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPY_FLAG_FALSE                        (0x00000000)
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPY_FLAG_TRUE                         (0x00000001)
#define LW90B3_VC1_SET_CONTROL_DEBLOCK_FLAG                                     3:3
#define LW90B3_VC1_SET_CONTROL_DEBLOCK_FLAG_FALSE                               (0x00000000)
#define LW90B3_VC1_SET_CONTROL_DEBLOCK_FLAG_TRUE                                (0x00000001)
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPUV                                  6:4
#define LW90B3_VC1_SET_CONTROL_RANGE_EXT_MAPY                                   10:8
#define LW90B3_VC1_SET_CONTROL_DEBLOCK_PQUANT                                   15:11
#define LW90B3_VC1_SET_CONTROL_RESERVED0                                        7:7
#define LW90B3_VC1_SET_CONTROL_RESERVED1                                        31:16
#define LW90B3_VC1_SET_BBOX_WEIGHT_SLOPE                                        (0x00000504)
#define LW90B3_VC1_SET_BBOX_WEIGHT_SLOPE_SLOPE_X                                15:0
#define LW90B3_VC1_SET_BBOX_WEIGHT_SLOPE_SLOPE_Y                                31:16
#define LW90B3_VC1_SET_BBOX_OUT_HOFFSET                                         (0x00000508)
#define LW90B3_VC1_SET_BBOX_OUT_HOFFSET_OFFSET                                  31:0
#define LW90B3_VC1_SET_BBOX_OUT_VOFFSET                                         (0x0000050C)
#define LW90B3_VC1_SET_BBOX_OUT_VOFFSET_OFFSET                                  31:0
#define LW90B3_SET_CONTROL_PARAMS1                                              (0x00000700)
#define LW90B3_SET_CONTROL_PARAMS1_CODEC_TYPE                                   3:0
#define LW90B3_SET_CONTROL_PARAMS1_CODEC_TYPE_MPEG1                             (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS1_CODEC_TYPE_MPEG2                             (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS1_CODEC_TYPE_VC1                               (0x00000002)
#define LW90B3_SET_CONTROL_PARAMS1_CODEC_TYPE_H264                              (0x00000003)
#define LW90B3_SET_CONTROL_PARAMS1_CODEC_TYPE_MPEG4                             (0x00000004)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_PIC_STRUCT                               7:4
#define LW90B3_SET_CONTROL_PARAMS1_OUT_PIC_STRUCT_FRAME                         (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_PIC_STRUCT_FIELD                         (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_TILE_MODE                                11:8
#define LW90B3_SET_CONTROL_PARAMS1_OUT_TILE_MODE_1D                             (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_TILE_MODE_PITCH_LINEAR_2D                (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_TILE_MODE_16X16                          (0x00000002)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_TILE_MODE_BLOCK_LINEAR_TEXTURE           (0x00000003)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_TILE_MODE_BLOCK_LINEAR_NAIVE             (0x00000004)
#define LW90B3_SET_CONTROL_PARAMS1_BYPASS                                       12:12
#define LW90B3_SET_CONTROL_PARAMS1_BYPASS_FALSE                                 (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS1_BYPASS_TRUE                                  (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS1_HC                                           13:13
#define LW90B3_SET_CONTROL_PARAMS1_HC_FALSE                                     (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS1_HC_TRUE                                      (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS1_COPY                                         14:14
#define LW90B3_SET_CONTROL_PARAMS1_COPY_FALSE                                   (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS1_COPY_TRUE                                    (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS1_FLOW_CTRL                                    15:15
#define LW90B3_SET_CONTROL_PARAMS1_FLOW_CTRL_DISABLE                            (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS1_FLOW_CTRL_ENABLE                             (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS1_OUT_STRIDE_LUMA                              23:16
#define LW90B3_SET_CONTROL_PARAMS1_OUT_STRIDE_CHROMA                            31:24
#define LW90B3_SET_PIC_PARAMS                                                   (0x00000704)
#define LW90B3_SET_PIC_PARAMS_WIDTH                                             7:0
#define LW90B3_SET_PIC_PARAMS_HEIGHT                                            15:8
#define LW90B3_SET_PIC_PARAMS_STRIDE_LUMA                                       23:16
#define LW90B3_SET_PIC_PARAMS_STRIDE_CHROMA                                     31:24
#define LW90B3_SET_SRC_PIC_TOP_FIELD_Y                                          (0x00000708)
#define LW90B3_SET_SRC_PIC_TOP_FIELD_Y_ADDR                                     31:0
#define LW90B3_SET_SRC_PIC_BOT_FIELD_Y                                          (0x0000070C)
#define LW90B3_SET_SRC_PIC_BOT_FIELD_Y_ADDR                                     31:0
#define LW90B3_SET_SRC_PIC_TOP_FIELD_C                                          (0x00000710)
#define LW90B3_SET_SRC_PIC_TOP_FIELD_C_ADDR                                     31:0
#define LW90B3_SET_SRC_PIC_BOT_FIELD_C                                          (0x00000714)
#define LW90B3_SET_SRC_PIC_BOT_FIELD_C_ADDR                                     31:0
#define LW90B3_SET_DST_PIC_TOP_FIELD_Y                                          (0x00000718)
#define LW90B3_SET_DST_PIC_TOP_FIELD_Y_ADDR                                     31:0
#define LW90B3_SET_DST_PIC_BOT_FIELD_Y                                          (0x0000071C)
#define LW90B3_SET_DST_PIC_BOT_FIELD_Y_ADDR                                     31:0
#define LW90B3_SET_DST_PIC_TOP_FIELD_C                                          (0x00000720)
#define LW90B3_SET_DST_PIC_TOP_FIELD_C_ADDR                                     31:0
#define LW90B3_SET_DST_PIC_BOT_FIELD_C                                          (0x00000724)
#define LW90B3_SET_DST_PIC_BOT_FIELD_C_ADDR                                     31:0
#define LW90B3_SET_HC_WIN_START                                                 (0x00000728)
#define LW90B3_SET_HC_WIN_START_X0                                              11:0
#define LW90B3_SET_HC_WIN_START_Y0                                              27:16
#define LW90B3_SET_HC_WIN_START_RESERVED0                                       15:12
#define LW90B3_SET_HC_WIN_START_RESERVED1                                       31:28
#define LW90B3_SET_HC_WIN_END                                                   (0x0000072C)
#define LW90B3_SET_HC_WIN_END_X1                                                11:0
#define LW90B3_SET_HC_WIN_END_Y1                                                27:16
#define LW90B3_SET_HC_WIN_END_RESERVED0                                         15:12
#define LW90B3_SET_HC_WIN_END_RESERVED1                                         31:28
#define LW90B3_SET_HC_RESULT                                                    (0x00000730)
#define LW90B3_SET_HC_RESULT_ADDR                                               31:0
#define LW90B3_SET_PICTURE_INDEX                                                (0x00000734)
#define LW90B3_SET_PICTURE_INDEX_INDEX                                          31:0
#define LW90B3_SET_CONTROL_PARAMS2                                              (0x00000738)
#define LW90B3_SET_CONTROL_PARAMS2_RESERVED0                                    3:0
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_GPTIMER_ON                              4:4
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_GPTIMER_ON_FALSE                        (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_GPTIMER_ON_TRUE                         (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_RET_ERROR                               5:5
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_RET_ERROR_FALSE                         (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_RET_ERROR_TRUE                          (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_FC_DBG                                  6:6
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_FC_DBG_FALSE                            (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_FC_DBG_TRUE                             (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_BBOX                                    7:7
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_BBOX_FALSE                              (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_BBOX_TRUE                               (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS2_RESERVED1                                    11:8
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_BBOX_SQRT                               12:12
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_BBOX_SQRT_FALSE                         (0x00000000)
#define LW90B3_SET_CONTROL_PARAMS2_CTRL_BBOX_SQRT_TRUE                          (0x00000001)
#define LW90B3_SET_CONTROL_PARAMS2_RESERVED2                                    31:13
#define LW90B3_SET_FLOW_CTRL_OFFSET                                             (0x0000073C)
#define LW90B3_SET_FLOW_CTRL_OFFSET_OFFSET                                      31:0
#define LW90B3_PM_TRIGGER_END                                                   (0x00001114)
#define LW90B3_PM_TRIGGER_END_V                                                 31:0

#define LW90B3_ERROR_NONE                                                       (0x00000000)
#define LW90B3_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LW90B3_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LW90B3_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LW90B3_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LW90B3_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LW90B3_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LW90B3_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LW90B3_OS_ERROR_APPLICATION                                             (0x00000008)
#define LW90B3_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LW90B3_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LW90B3_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LW90B3_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LW90B3_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LW90B3_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LW90B3_FGT_ERROR_HWERR                                                  (0x00004100)
#define LW90B3_FGT_ERROR_PROGERR                                                (0x00004200)
#define LW90B3_FGT_ERROR_TIMEOUT                                                (0x00004400)
#define LW90B3_FGT_ERROR_IDLEWATCHDOG                                           (0x00004800)
#define LW90B3_VC1_ERROR_HWERR                                                  (0x00005100)
#define LW90B3_VC1_ERROR_PROGERR                                                (0x00005200)
#define LW90B3_VC1_ERROR_TIMEOUT                                                (0x00005400)
#define LW90B3_VC1_ERROR_IDLEWATCHDOG                                           (0x00005800)
#define LW90B3_TF_ERROR_HWERR                                                   (0x00007001)
#define LW90B3_TF_ERROR_PROGERR                                                 (0x00007002)
#define LW90B3_TF_ERROR_TIMEOUT                                                 (0x00007004)
#define LW90B3_TF_ERROR_IDLEWATCHDOG                                            (0x00007008)
#define LW90B3_CRB_ERROR_HWERR                                                  (0x00007010)
#define LW90B3_CRB_ERROR_PROGERR                                                (0x00007020)
#define LW90B3_CRB_ERROR_TIMEOUT                                                (0x00007040)
#define LW90B3_CRB_ERROR_IDLEWATCHDOG                                           (0x00007080)
#define LW90B3_APP_TIMER_EXPIRED                                                (0x00007100)
#define LW90B3_ERROR_ILWALID_OUT_STRIDE                                         (0x00007200)
#define LW90B3_ERROR_ILWALID_IN_WIDTH_HEIGHT                                    (0x00007201)
#define LW90B3_ERROR_ILWALID_IN_STRIDE                                          (0x00007202)
#define LW90B3_ERROR_ILWALID_CTXID_FC                                           (0x00007204)
#define LW90B3_ERROR_ILWALID_CTXID_HC                                           (0x00007208)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl90b3_h

