/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2010 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/



#ifndef _cl927c_h_
#define _cl927c_h_


#ifdef __cplusplus
extern "C" {
#endif

#define LW927C_BASE_CHANNEL_DMA                                                 (0x0000927C)

typedef volatile struct _cl927c_tag0 {
    LwV32 Put;                                                                  // 0x00000000 - 0x00000003
    LwV32 Get;                                                                  // 0x00000004 - 0x00000007
    LwV32 Reserved00[0x2];
    LwV32 GetScanline;                                                          // 0x00000010 - 0x00000013
    LwV32 Reserved01[0x1B];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetPresentControl;                                                    // 0x00000084 - 0x00000087
    LwV32 SetSemaphoreControl;                                                  // 0x00000088 - 0x0000008B
    LwV32 SetSemaphoreAcquire;                                                  // 0x0000008C - 0x0000008F
    LwV32 SetSemaphoreRelease;                                                  // 0x00000090 - 0x00000093
    LwV32 SetContextDmaSemaphore;                                               // 0x00000094 - 0x00000097
    LwV32 Reserved02[0x2];
    LwV32 SetNotifierControl;                                                   // 0x000000A0 - 0x000000A3
    LwV32 SetContextDmaNotifier;                                                // 0x000000A4 - 0x000000A7
    LwV32 Reserved03[0x6];
    LwV32 SetContextDmasIso[4];                                                 // 0x000000C0 - 0x000000CF
    LwV32 Reserved04[0x4];
    LwV32 SetBaseLutLo;                                                         // 0x000000E0 - 0x000000E3
    LwV32 SetBaseLutHi;                                                         // 0x000000E4 - 0x000000E7
    LwV32 SetOutputLutLo;                                                       // 0x000000E8 - 0x000000EB
    LwV32 SetOutputLutHi;                                                       // 0x000000EC - 0x000000EF
    LwV32 Reserved05[0x3];
    LwV32 SetContextDmaLut;                                                     // 0x000000FC - 0x000000FF
    LwV32 SetDistRenderControl;                                                 // 0x00000100 - 0x00000103
    LwV32 SetDistRenderExtendControl;                                           // 0x00000104 - 0x00000107
    LwV32 SetDistRenderInhibitFlipRegion;                                       // 0x00000108 - 0x0000010B
    LwV32 SetDistRenderConfig;                                                  // 0x0000010C - 0x0000010F
    LwV32 SetProcessing;                                                        // 0x00000110 - 0x00000113
    LwV32 SetColwersionRed;                                                     // 0x00000114 - 0x00000117
    LwV32 SetColwersionGrn;                                                     // 0x00000118 - 0x0000011B
    LwV32 SetColwersionBlu;                                                     // 0x0000011C - 0x0000011F
    LwV32 Reserved06[0x4];
    LwV32 SetTimestampOriginLo;                                                 // 0x00000130 - 0x00000133
    LwV32 SetTimestampOriginHi;                                                 // 0x00000134 - 0x00000137
    LwV32 SetUpdateTimestampLo;                                                 // 0x00000138 - 0x0000013B
    LwV32 SetUpdateTimestampHi;                                                 // 0x0000013C - 0x0000013F
    LwV32 SetCscRed2Red;                                                        // 0x00000140 - 0x00000143
    LwV32 SetCscGrn2Red;                                                        // 0x00000144 - 0x00000147
    LwV32 SetCscBlu2Red;                                                        // 0x00000148 - 0x0000014B
    LwV32 SetCscConstant2Red;                                                   // 0x0000014C - 0x0000014F
    LwV32 SetCscRed2Grn;                                                        // 0x00000150 - 0x00000153
    LwV32 SetCscGrn2Grn;                                                        // 0x00000154 - 0x00000157
    LwV32 SetCscBlu2Grn;                                                        // 0x00000158 - 0x0000015B
    LwV32 SetCscConstant2Grn;                                                   // 0x0000015C - 0x0000015F
    LwV32 SetCscRed2Blu;                                                        // 0x00000160 - 0x00000163
    LwV32 SetCscGrn2Blu;                                                        // 0x00000164 - 0x00000167
    LwV32 SetCscBlu2Blu;                                                        // 0x00000168 - 0x0000016B
    LwV32 SetCscConstant2Blu;                                                   // 0x0000016C - 0x0000016F
    LwV32 Reserved07[0x93];
    LwV32 SetSpare;                                                             // 0x000003BC - 0x000003BF
    LwV32 SetSpareNoop[4];                                                      // 0x000003C0 - 0x000003CF
    LwV32 Reserved08[0xC];
    struct {
        LwV32 SetOffset[2];                                                     // 0x00000400 - 0x00000407
        LwV32 SetSize;                                                          // 0x00000408 - 0x0000040B
        LwV32 SetStorage;                                                       // 0x0000040C - 0x0000040F
        LwV32 SetParams;                                                        // 0x00000410 - 0x00000413
        LwV32 Reserved09[0x3];
    } Surface[2];
    LwV32 Reserved10[0x2F0];
} LW927CDispControlDma;


#define LW_DISP_BASE_NOTIFIER_1                                                      0x00000000
#define LW_DISP_BASE_NOTIFIER_1_SIZEOF                                               0x00000004
#define LW_DISP_BASE_NOTIFIER_1__0                                                   0x00000000
#define LW_DISP_BASE_NOTIFIER_1__0_PRESENTATION_COUNT                                15:0
#define LW_DISP_BASE_NOTIFIER_1__0_TIMESTAMP                                         29:16
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS                                            31:30
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS_NOT_BEGUN                                  0x00000000
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS_BEGUN                                      0x00000001
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS_FINISHED                                   0x00000002


#define LW_DISP_NOTIFICATION_2                                                       0x00000000
#define LW_DISP_NOTIFICATION_2_SIZEOF                                                0x00000010
#define LW_DISP_NOTIFICATION_2_TIME_STAMP_0                                          0x00000000
#define LW_DISP_NOTIFICATION_2_TIME_STAMP_0_NANOSECONDS0                             31:0
#define LW_DISP_NOTIFICATION_2_TIME_STAMP_1                                          0x00000001
#define LW_DISP_NOTIFICATION_2_TIME_STAMP_1_NANOSECONDS1                             31:0
#define LW_DISP_NOTIFICATION_2_INFO32_2                                              0x00000002
#define LW_DISP_NOTIFICATION_2_INFO32_2_R0                                           31:0
#define LW_DISP_NOTIFICATION_2_INFO16_3                                              0x00000003
#define LW_DISP_NOTIFICATION_2_INFO16_3_PRESENT_COUNT                                7:0
#define LW_DISP_NOTIFICATION_2_INFO16_3_FIELD                                        8:8
#define LW_DISP_NOTIFICATION_2_INFO16_3_R1                                           15:9
#define LW_DISP_NOTIFICATION_2__3_STATUS                                             31:16
#define LW_DISP_NOTIFICATION_2__3_STATUS_NOT_BEGUN                                   0x00008000
#define LW_DISP_NOTIFICATION_2__3_STATUS_BEGUN                                       0x0000FFFF
#define LW_DISP_NOTIFICATION_2__3_STATUS_FINISHED                                    0x00000000


#define LW_DISP_NOTIFICATION_INFO16                                                  0x00000000
#define LW_DISP_NOTIFICATION_INFO16_SIZEOF                                           0x00000002
#define LW_DISP_NOTIFICATION_INFO16__0                                               0x00000000
#define LW_DISP_NOTIFICATION_INFO16__0_PRESENT_COUNT                                 7:0
#define LW_DISP_NOTIFICATION_INFO16__0_FIELD                                         8:8
#define LW_DISP_NOTIFICATION_INFO16__0_R1                                            15:9


#define LW_DISP_NOTIFICATION_STATUS                                                  0x00000000
#define LW_DISP_NOTIFICATION_STATUS_SIZEOF                                           0x00000002
#define LW_DISP_NOTIFICATION_STATUS__0                                               0x00000000
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS                                        15:0
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS_NOT_BEGUN                              0x00008000
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS_BEGUN                                  0x0000FFFF
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS_FINISHED                               0x00000000


// dma opcode instructions
#define LW927C_DMA                                                         0x00000000 
#define LW927C_DMA_OPCODE                                                       31:29 
#define LW927C_DMA_OPCODE_METHOD                                           0x00000000 
#define LW927C_DMA_OPCODE_JUMP                                             0x00000001 
#define LW927C_DMA_OPCODE_NONINC_METHOD                                    0x00000002 
#define LW927C_DMA_OPCODE_SET_SUBDEVICE_MASK                               0x00000003 
#define LW927C_DMA_OPCODE                                                       31:29 
#define LW927C_DMA_OPCODE_METHOD                                           0x00000000 
#define LW927C_DMA_OPCODE_NONINC_METHOD                                    0x00000002 
#define LW927C_DMA_METHOD_COUNT                                                 27:18 
#define LW927C_DMA_METHOD_OFFSET                                                 11:2 
#define LW927C_DMA_DATA                                                          31:0 
#define LW927C_DMA_DATA_NOP                                                0x00000000 
#define LW927C_DMA_OPCODE                                                       31:29 
#define LW927C_DMA_OPCODE_JUMP                                             0x00000001 
#define LW927C_DMA_JUMP_OFFSET                                                   11:2 
#define LW927C_DMA_OPCODE                                                       31:29 
#define LW927C_DMA_OPCODE_SET_SUBDEVICE_MASK                               0x00000003 
#define LW927C_DMA_SET_SUBDEVICE_MASK_VALUE                                      11:0 

// class methods
#define LW927C_PUT                                                              (0x00000000)
#define LW927C_PUT_PTR                                                          11:2
#define LW927C_GET                                                              (0x00000004)
#define LW927C_GET_PTR                                                          11:2
#define LW927C_GET_SCANLINE                                                     (0x00000010)
#define LW927C_GET_SCANLINE_LINE                                                15:0
#define LW927C_UPDATE                                                           (0x00000080)
#define LW927C_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW927C_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW927C_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW927C_UPDATE_RESET_SPEC_FLIP_SEQ                                       12:12
#define LW927C_UPDATE_RESET_SPEC_FLIP_SEQ_DISABLE                               (0x00000000)
#define LW927C_UPDATE_RESET_SPEC_FLIP_SEQ_ENABLE                                (0x00000001)
#define LW927C_UPDATE_SPECIAL_HANDLING                                          25:24
#define LW927C_UPDATE_SPECIAL_HANDLING_NONE                                     (0x00000000)
#define LW927C_UPDATE_SPECIAL_HANDLING_INTERRUPT_RM                             (0x00000001)
#define LW927C_UPDATE_SPECIAL_HANDLING_MODE_SWITCH                              (0x00000002)
#define LW927C_UPDATE_SPECIAL_HANDLING_REASON                                   23:16
#define LW927C_SET_PRESENT_CONTROL                                              (0x00000084)
#define LW927C_SET_PRESENT_CONTROL_BEGIN_MODE                                   9:8
#define LW927C_SET_PRESENT_CONTROL_BEGIN_MODE_NON_TEARING                       (0x00000000)
#define LW927C_SET_PRESENT_CONTROL_BEGIN_MODE_IMMEDIATE                         (0x00000001)
#define LW927C_SET_PRESENT_CONTROL_BEGIN_MODE_ON_LINE                           (0x00000002)
#define LW927C_SET_PRESENT_CONTROL_STEREO_FLIP_MODE                             3:3
#define LW927C_SET_PRESENT_CONTROL_STEREO_FLIP_MODE_PAIR_FLIP                   (0x00000000)
#define LW927C_SET_PRESENT_CONTROL_STEREO_FLIP_MODE_AT_ANY_FRAME                (0x00000001)
#define LW927C_SET_PRESENT_CONTROL_TIMESTAMP_MODE                               2:2
#define LW927C_SET_PRESENT_CONTROL_TIMESTAMP_MODE_DISABLE                       (0x00000000)
#define LW927C_SET_PRESENT_CONTROL_TIMESTAMP_MODE_ENABLE                        (0x00000001)
#define LW927C_SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL                         7:4
#define LW927C_SET_PRESENT_CONTROL_BEGIN_LINE                                   30:16
#define LW927C_SET_PRESENT_CONTROL_ON_LINE_MARGIN                               15:10
#define LW927C_SET_PRESENT_CONTROL_MODE                                         1:0
#define LW927C_SET_PRESENT_CONTROL_MODE_MONO                                    (0x00000000)
#define LW927C_SET_PRESENT_CONTROL_MODE_STEREO                                  (0x00000001)
#define LW927C_SET_PRESENT_CONTROL_MODE_SPEC_FLIP                               (0x00000002)
#define LW927C_SET_SEMAPHORE_CONTROL                                            (0x00000088)
#define LW927C_SET_SEMAPHORE_CONTROL_OFFSET                                     11:2
#define LW927C_SET_SEMAPHORE_CONTROL_DELAY                                      26:26
#define LW927C_SET_SEMAPHORE_CONTROL_DELAY_DISABLE                              (0x00000000)
#define LW927C_SET_SEMAPHORE_CONTROL_DELAY_ENABLE                               (0x00000001)
#define LW927C_SET_SEMAPHORE_CONTROL_FORMAT                                     28:28
#define LW927C_SET_SEMAPHORE_CONTROL_FORMAT_LEGACY                              (0x00000000)
#define LW927C_SET_SEMAPHORE_CONTROL_FORMAT_FOUR_WORD                           (0x00000001)
#define LW927C_SET_SEMAPHORE_ACQUIRE                                            (0x0000008C)
#define LW927C_SET_SEMAPHORE_ACQUIRE_VALUE                                      31:0
#define LW927C_SET_SEMAPHORE_RELEASE                                            (0x00000090)
#define LW927C_SET_SEMAPHORE_RELEASE_VALUE                                      31:0
#define LW927C_SET_CONTEXT_DMA_SEMAPHORE                                        (0x00000094)
#define LW927C_SET_CONTEXT_DMA_SEMAPHORE_HANDLE                                 31:0
#define LW927C_SET_NOTIFIER_CONTROL                                             (0x000000A0)
#define LW927C_SET_NOTIFIER_CONTROL_MODE                                        30:30
#define LW927C_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LW927C_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LW927C_SET_NOTIFIER_CONTROL_OFFSET                                      11:2
#define LW927C_SET_NOTIFIER_CONTROL_DELAY                                       26:26
#define LW927C_SET_NOTIFIER_CONTROL_DELAY_DISABLE                               (0x00000000)
#define LW927C_SET_NOTIFIER_CONTROL_DELAY_ENABLE                                (0x00000001)
#define LW927C_SET_NOTIFIER_CONTROL_FORMAT                                      28:28
#define LW927C_SET_NOTIFIER_CONTROL_FORMAT_LEGACY                               (0x00000000)
#define LW927C_SET_NOTIFIER_CONTROL_FORMAT_FOUR_WORD                            (0x00000001)
#define LW927C_SET_CONTEXT_DMA_NOTIFIER                                         (0x000000A4)
#define LW927C_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LW927C_SET_CONTEXT_DMAS_ISO(b)                                          (0x000000C0 + (b)*0x00000004)
#define LW927C_SET_CONTEXT_DMAS_ISO_HANDLE                                      31:0
#define LW927C_SET_BASE_LUT_LO                                                  (0x000000E0)
#define LW927C_SET_BASE_LUT_LO_ENABLE                                           31:30
#define LW927C_SET_BASE_LUT_LO_ENABLE_DISABLE                                   (0x00000000)
#define LW927C_SET_BASE_LUT_LO_ENABLE_USE_CORE_LUT                              (0x00000001)
#define LW927C_SET_BASE_LUT_LO_ENABLE_ENABLE                                    (0x00000002)
#define LW927C_SET_BASE_LUT_LO_MODE                                             27:24
#define LW927C_SET_BASE_LUT_LO_MODE_LORES                                       (0x00000000)
#define LW927C_SET_BASE_LUT_LO_MODE_HIRES                                       (0x00000001)
#define LW927C_SET_BASE_LUT_LO_MODE_INDEX_1025_UNITY_RANGE                      (0x00000003)
#define LW927C_SET_BASE_LUT_LO_MODE_INTERPOLATE_1025_UNITY_RANGE                (0x00000004)
#define LW927C_SET_BASE_LUT_LO_MODE_INTERPOLATE_1025_XRBIAS_RANGE               (0x00000005)
#define LW927C_SET_BASE_LUT_LO_MODE_INTERPOLATE_1025_XVYCC_RANGE                (0x00000006)
#define LW927C_SET_BASE_LUT_LO_MODE_INTERPOLATE_257_UNITY_RANGE                 (0x00000007)
#define LW927C_SET_BASE_LUT_LO_MODE_INTERPOLATE_257_LEGACY_RANGE                (0x00000008)
#define LW927C_SET_BASE_LUT_HI                                                  (0x000000E4)
#define LW927C_SET_BASE_LUT_HI_ORIGIN                                           31:0
#define LW927C_SET_OUTPUT_LUT_LO                                                (0x000000E8)
#define LW927C_SET_OUTPUT_LUT_LO_ENABLE                                         31:30
#define LW927C_SET_OUTPUT_LUT_LO_ENABLE_DISABLE                                 (0x00000000)
#define LW927C_SET_OUTPUT_LUT_LO_ENABLE_USE_CORE_LUT                            (0x00000001)
#define LW927C_SET_OUTPUT_LUT_LO_ENABLE_ENABLE                                  (0x00000002)
#define LW927C_SET_OUTPUT_LUT_LO_MODE                                           27:24
#define LW927C_SET_OUTPUT_LUT_LO_MODE_LORES                                     (0x00000000)
#define LW927C_SET_OUTPUT_LUT_LO_MODE_HIRES                                     (0x00000001)
#define LW927C_SET_OUTPUT_LUT_LO_MODE_INDEX_1025_UNITY_RANGE                    (0x00000003)
#define LW927C_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_UNITY_RANGE              (0x00000004)
#define LW927C_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_XRBIAS_RANGE             (0x00000005)
#define LW927C_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_XVYCC_RANGE              (0x00000006)
#define LW927C_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_257_UNITY_RANGE               (0x00000007)
#define LW927C_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_257_LEGACY_RANGE              (0x00000008)
#define LW927C_SET_OUTPUT_LUT_HI                                                (0x000000EC)
#define LW927C_SET_OUTPUT_LUT_HI_ORIGIN                                         31:0
#define LW927C_SET_CONTEXT_DMA_LUT                                              (0x000000FC)
#define LW927C_SET_CONTEXT_DMA_LUT_HANDLE                                       31:0
#define LW927C_SET_DIST_RENDER_CONTROL                                          (0x00000100)
#define LW927C_SET_DIST_RENDER_CONTROL_MODE                                     1:0
#define LW927C_SET_DIST_RENDER_CONTROL_MODE_OFF                                 (0x00000000)
#define LW927C_SET_DIST_RENDER_CONTROL_MODE_STRIP_INTERLEAVE                    (0x00000001)
#define LW927C_SET_DIST_RENDER_CONTROL_MODE_FRAME_INTERLEAVE                    (0x00000002)
#define LW927C_SET_DIST_RENDER_CONTROL_MODE_FRAME_AND_STRIP_INTERLEAVE          (0x00000003)
#define LW927C_SET_DIST_RENDER_CONTROL_RENDER_START                             16:2
#define LW927C_SET_DIST_RENDER_CONTROL_RENDER_STOP                              31:17
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL                                   (0x00000104)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_EXTENSION_ENABLE                  31:31
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_EXTENSION_ENABLE_DISABLE          (0x00000000)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_EXTENSION_ENABLE_ENABLE           (0x00000001)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_MODE                              3:0
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_OFF                          (0x00000000)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_FOS_2CHIP                    (0x00000001)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_FOS_4CHIP                    (0x00000002)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_FRAME_INTERLEAVE_FOS_4CHIP   (0x00000008)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT                9:8
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT_X1             (0x00000000)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT_X2             (0x00000001)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT_X4             (0x00000002)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT               13:12
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT_D1            (0x00000000)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT_D2            (0x00000001)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT_D4            (0x00000002)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE                   17:16
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE_LINEAR            (0x00000000)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE_GAMMA             (0x00000001)
#define LW927C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE_NONE              (0x00000003)
#define LW927C_SET_DIST_RENDER_INHIBIT_FLIP_REGION                              (0x00000108)
#define LW927C_SET_DIST_RENDER_INHIBIT_FLIP_REGION_LINES_BEFORE_RENDER_START    7:0
#define LW927C_SET_DIST_RENDER_INHIBIT_FLIP_REGION_LINES_AFTER_RENDER_START     15:8
#define LW927C_SET_DIST_RENDER_INHIBIT_FLIP_REGION_LINES_BEFORE_RENDER_STOP     23:16
#define LW927C_SET_DIST_RENDER_INHIBIT_FLIP_REGION_LINES_AFTER_RENDER_STOP      31:24
#define LW927C_SET_DIST_RENDER_CONFIG                                           (0x0000010C)
#define LW927C_SET_DIST_RENDER_CONFIG_GROUP_SIZE                                27:24
#define LW927C_SET_DIST_RENDER_CONFIG_GROUP_POSITION                            31:28
#define LW927C_SET_DIST_RENDER_CONFIG_METERED_FLIPPING                          20:20
#define LW927C_SET_DIST_RENDER_CONFIG_METERED_FLIPPING_DISABLE                  (0x00000000)
#define LW927C_SET_DIST_RENDER_CONFIG_METERED_FLIPPING_ENABLE                   (0x00000001)
#define LW927C_SET_DIST_RENDER_CONFIG_MIN_TIME_FACTOR                           18:16
#define LW927C_SET_PROCESSING                                                   (0x00000110)
#define LW927C_SET_PROCESSING_USE_GAIN_OFS                                      0:0
#define LW927C_SET_PROCESSING_USE_GAIN_OFS_DISABLE                              (0x00000000)
#define LW927C_SET_PROCESSING_USE_GAIN_OFS_ENABLE                               (0x00000001)
#define LW927C_SET_COLWERSION_RED                                               (0x00000114)
#define LW927C_SET_COLWERSION_RED_GAIN                                          15:0
#define LW927C_SET_COLWERSION_RED_OFS                                           31:16
#define LW927C_SET_COLWERSION_GRN                                               (0x00000118)
#define LW927C_SET_COLWERSION_GRN_GAIN                                          15:0
#define LW927C_SET_COLWERSION_GRN_OFS                                           31:16
#define LW927C_SET_COLWERSION_BLU                                               (0x0000011C)
#define LW927C_SET_COLWERSION_BLU_GAIN                                          15:0
#define LW927C_SET_COLWERSION_BLU_OFS                                           31:16
#define LW927C_SET_TIMESTAMP_ORIGIN_LO                                          (0x00000130)
#define LW927C_SET_TIMESTAMP_ORIGIN_LO_TIMESTAMP_LO                             31:0
#define LW927C_SET_TIMESTAMP_ORIGIN_HI                                          (0x00000134)
#define LW927C_SET_TIMESTAMP_ORIGIN_HI_TIMESTAMP_HI                             31:0
#define LW927C_SET_UPDATE_TIMESTAMP_LO                                          (0x00000138)
#define LW927C_SET_UPDATE_TIMESTAMP_LO_TIMESTAMP_LO                             31:0
#define LW927C_SET_UPDATE_TIMESTAMP_HI                                          (0x0000013C)
#define LW927C_SET_UPDATE_TIMESTAMP_HI_TIMESTAMP_HI                             31:0
#define LW927C_SET_CSC_RED2RED                                                  (0x00000140)
#define LW927C_SET_CSC_RED2RED_OWNER                                            31:31
#define LW927C_SET_CSC_RED2RED_OWNER_CORE                                       (0x00000000)
#define LW927C_SET_CSC_RED2RED_OWNER_BASE                                       (0x00000001)
#define LW927C_SET_CSC_RED2RED_COEFF                                            18:0
#define LW927C_SET_CSC_GRN2RED                                                  (0x00000144)
#define LW927C_SET_CSC_GRN2RED_COEFF                                            18:0
#define LW927C_SET_CSC_BLU2RED                                                  (0x00000148)
#define LW927C_SET_CSC_BLU2RED_COEFF                                            18:0
#define LW927C_SET_CSC_CONSTANT2RED                                             (0x0000014C)
#define LW927C_SET_CSC_CONSTANT2RED_COEFF                                       18:0
#define LW927C_SET_CSC_RED2GRN                                                  (0x00000150)
#define LW927C_SET_CSC_RED2GRN_COEFF                                            18:0
#define LW927C_SET_CSC_GRN2GRN                                                  (0x00000154)
#define LW927C_SET_CSC_GRN2GRN_COEFF                                            18:0
#define LW927C_SET_CSC_BLU2GRN                                                  (0x00000158)
#define LW927C_SET_CSC_BLU2GRN_COEFF                                            18:0
#define LW927C_SET_CSC_CONSTANT2GRN                                             (0x0000015C)
#define LW927C_SET_CSC_CONSTANT2GRN_COEFF                                       18:0
#define LW927C_SET_CSC_RED2BLU                                                  (0x00000160)
#define LW927C_SET_CSC_RED2BLU_COEFF                                            18:0
#define LW927C_SET_CSC_GRN2BLU                                                  (0x00000164)
#define LW927C_SET_CSC_GRN2BLU_COEFF                                            18:0
#define LW927C_SET_CSC_BLU2BLU                                                  (0x00000168)
#define LW927C_SET_CSC_BLU2BLU_COEFF                                            18:0
#define LW927C_SET_CSC_CONSTANT2BLU                                             (0x0000016C)
#define LW927C_SET_CSC_CONSTANT2BLU_COEFF                                       18:0
#define LW927C_SET_SPARE                                                        (0x000003BC)
#define LW927C_SET_SPARE_UNUSED                                                 31:0
#define LW927C_SET_SPARE_NOOP(b)                                                (0x000003C0 + (b)*0x00000004)
#define LW927C_SET_SPARE_NOOP_UNUSED                                            31:0

#define LW927C_SURFACE_SET_OFFSET(a,b)                                          (0x00000400 + (a)*0x00000020 + (b)*0x00000004)
#define LW927C_SURFACE_SET_OFFSET_ORIGIN                                        31:0
#define LW927C_SURFACE_SET_SIZE(a)                                              (0x00000408 + (a)*0x00000020)
#define LW927C_SURFACE_SET_SIZE_WIDTH                                           15:0
#define LW927C_SURFACE_SET_SIZE_HEIGHT                                          31:16
#define LW927C_SURFACE_SET_STORAGE(a)                                           (0x0000040C + (a)*0x00000020)
#define LW927C_SURFACE_SET_STORAGE_BLOCK_HEIGHT                                 3:0
#define LW927C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_ONE_GOB                         (0x00000000)
#define LW927C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_TWO_GOBS                        (0x00000001)
#define LW927C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_FOUR_GOBS                       (0x00000002)
#define LW927C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_EIGHT_GOBS                      (0x00000003)
#define LW927C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_SIXTEEN_GOBS                    (0x00000004)
#define LW927C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_THIRTYTWO_GOBS                  (0x00000005)
#define LW927C_SURFACE_SET_STORAGE_PITCH                                        20:8
#define LW927C_SURFACE_SET_STORAGE_MEMORY_LAYOUT                                24:24
#define LW927C_SURFACE_SET_STORAGE_MEMORY_LAYOUT_BLOCKLINEAR                    (0x00000000)
#define LW927C_SURFACE_SET_STORAGE_MEMORY_LAYOUT_PITCH                          (0x00000001)
#define LW927C_SURFACE_SET_PARAMS(a)                                            (0x00000410 + (a)*0x00000020)
#define LW927C_SURFACE_SET_PARAMS_FORMAT                                        15:8
#define LW927C_SURFACE_SET_PARAMS_FORMAT_I8                                     (0x0000001E)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_VOID16                                 (0x0000001F)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_VOID32                                 (0x0000002E)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_RF16_GF16_BF16_AF16                    (0x000000CA)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_A8R8G8B8                               (0x000000CF)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_A2B10G10R10                            (0x000000D1)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_X2BL10GL10RL10_XRBIAS                  (0x00000022)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_X2BL10GL10RL10_XVYCC                   (0x00000024)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_A8B8G8R8                               (0x000000D5)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_R5G6B5                                 (0x000000E8)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_A1R5G5B5                               (0x000000E9)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_R16_G16_B16_A16                        (0x000000C6)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_R16_G16_B16_A16_LWBIAS                 (0x00000023)
#define LW927C_SURFACE_SET_PARAMS_FORMAT_A2R10G10B10                            (0x000000DF)
#define LW927C_SURFACE_SET_PARAMS_SUPER_SAMPLE                                  1:0
#define LW927C_SURFACE_SET_PARAMS_SUPER_SAMPLE_X1_AA                            (0x00000000)
#define LW927C_SURFACE_SET_PARAMS_SUPER_SAMPLE_X4_AA                            (0x00000002)
#define LW927C_SURFACE_SET_PARAMS_GAMMA                                         2:2
#define LW927C_SURFACE_SET_PARAMS_GAMMA_LINEAR                                  (0x00000000)
#define LW927C_SURFACE_SET_PARAMS_GAMMA_SRGB                                    (0x00000001)
#define LW927C_SURFACE_SET_PARAMS_LAYOUT                                        5:4
#define LW927C_SURFACE_SET_PARAMS_LAYOUT_FRM                                    (0x00000000)
#define LW927C_SURFACE_SET_PARAMS_LAYOUT_FLD1                                   (0x00000001)
#define LW927C_SURFACE_SET_PARAMS_LAYOUT_FLD2                                   (0x00000002)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl927c_h
