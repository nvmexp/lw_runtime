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


#ifndef _cl857c_h_
#define _cl857c_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW857C_BASE_CHANNEL_DMA                                                 (0x0000857C)

typedef volatile struct _cl857c_tag0 {
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
    LwV32 Reserved06[0x2];
    LwV32 SetProcessing;                                                        // 0x00000110 - 0x00000113
    LwV32 SetColwersion;                                                        // 0x00000114 - 0x00000117
    LwV32 Reserved07[0x1A9];
    LwV32 SetSpare;                                                             // 0x000007BC - 0x000007BF
    LwV32 SetSpareNoop[16];                                                     // 0x000007C0 - 0x000007FF
    struct {
        LwV32 SetOffset[2];                                                     // 0x00000800 - 0x00000807
        LwV32 SetSize;                                                          // 0x00000808 - 0x0000080B
        LwV32 SetStorage;                                                       // 0x0000080C - 0x0000080F
        LwV32 SetParams;                                                        // 0x00000810 - 0x00000813
        LwV32 Reserved08[0x3];
    } Surface[2];
    LwV32 Reserved09[0x1F0];
} GT214DispBaseControlDma;


#define LW_DISP_BASE_NOTIFIER_1                                                      0x00000000
#define LW_DISP_BASE_NOTIFIER_1_SIZEOF                                               0x00000004
#define LW_DISP_BASE_NOTIFIER_1__0                                                   0x00000000
#define LW_DISP_BASE_NOTIFIER_1__0_PRESENTATION_COUNT                                15:0
#define LW_DISP_BASE_NOTIFIER_1__0_TIMESTAMP                                         29:16
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS                                            31:30
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS_NOT_BEGUN                                  0x00000000
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS_BEGUN                                      0x00000001
#define LW_DISP_BASE_NOTIFIER_1__0_STATUS_FINISHED                                   0x00000002


// dma opcode instructions
#define LW857C_DMA                                                                              0x00000000 
#define LW857C_DMA_OPCODE                                                                            31:29 
#define LW857C_DMA_OPCODE_METHOD                                                                0x00000000 
#define LW857C_DMA_OPCODE_JUMP                                                                  0x00000001 
#define LW857C_DMA_OPCODE_NONINC_METHOD                                                         0x00000002 
#define LW857C_DMA_OPCODE_SET_SUBDEVICE_MASK                                                    0x00000003 
#define LW857C_DMA_OPCODE                                                                            31:29 
#define LW857C_DMA_OPCODE_METHOD                                                                0x00000000 
#define LW857C_DMA_OPCODE_NONINC_METHOD                                                         0x00000002 
#define LW857C_DMA_METHOD_COUNT                                                                      27:18 
#define LW857C_DMA_METHOD_OFFSET                                                                      11:2 
#define LW857C_DMA_DATA                                                                               31:0 
#define LW857C_DMA_DATA_NOP                                                                     0x00000000 
#define LW857C_DMA_OPCODE                                                                            31:29 
#define LW857C_DMA_OPCODE_JUMP                                                                  0x00000001 
#define LW857C_DMA_JUMP_OFFSET                                                                        11:2 
#define LW857C_DMA_OPCODE                                                                            31:29 
#define LW857C_DMA_OPCODE_SET_SUBDEVICE_MASK                                                    0x00000003 
#define LW857C_DMA_SET_SUBDEVICE_MASK_VALUE                                                           11:0 

// class methods
#define LW857C_PUT                                                              (0x00000000)
#define LW857C_PUT_PTR                                                          11:2
#define LW857C_GET                                                              (0x00000004)
#define LW857C_GET_PTR                                                          11:2
#define LW857C_GET_SCANLINE                                                     (0x00000010)
#define LW857C_GET_SCANLINE_LINE                                                15:0
#define LW857C_UPDATE                                                           (0x00000080)
#define LW857C_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW857C_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW857C_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW857C_UPDATE_RESET_SPEC_FLIP_SEQ                                       12:12
#define LW857C_UPDATE_RESET_SPEC_FLIP_SEQ_DISABLE                               (0x00000000)
#define LW857C_UPDATE_RESET_SPEC_FLIP_SEQ_ENABLE                                (0x00000001)
#define LW857C_SET_PRESENT_CONTROL                                              (0x00000084)
#define LW857C_SET_PRESENT_CONTROL_BEGIN_MODE                                   9:8
#define LW857C_SET_PRESENT_CONTROL_BEGIN_MODE_NON_TEARING                       (0x00000000)
#define LW857C_SET_PRESENT_CONTROL_BEGIN_MODE_IMMEDIATE                         (0x00000001)
#define LW857C_SET_PRESENT_CONTROL_BEGIN_MODE_ON_LINE                           (0x00000002)
#define LW857C_SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL                         7:4
#define LW857C_SET_PRESENT_CONTROL_BEGIN_LINE                                   30:16
#define LW857C_SET_PRESENT_CONTROL_ON_LINE_MARGIN                               15:10
#define LW857C_SET_PRESENT_CONTROL_MODE                                         1:0
#define LW857C_SET_PRESENT_CONTROL_MODE_MONO                                    (0x00000000)
#define LW857C_SET_PRESENT_CONTROL_MODE_STEREO                                  (0x00000001)
#define LW857C_SET_PRESENT_CONTROL_MODE_SPEC_FLIP                               (0x00000002)
#define LW857C_SET_SEMAPHORE_CONTROL                                            (0x00000088)
#define LW857C_SET_SEMAPHORE_CONTROL_OFFSET                                     11:2
#define LW857C_SET_SEMAPHORE_CONTROL_DELAY                                      26:26
#define LW857C_SET_SEMAPHORE_CONTROL_DELAY_DISABLE                              (0x00000000)
#define LW857C_SET_SEMAPHORE_CONTROL_DELAY_ENABLE                               (0x00000001)
#define LW857C_SET_SEMAPHORE_ACQUIRE                                            (0x0000008C)
#define LW857C_SET_SEMAPHORE_ACQUIRE_VALUE                                      31:0
#define LW857C_SET_SEMAPHORE_RELEASE                                            (0x00000090)
#define LW857C_SET_SEMAPHORE_RELEASE_VALUE                                      31:0
#define LW857C_SET_CONTEXT_DMA_SEMAPHORE                                        (0x00000094)
#define LW857C_SET_CONTEXT_DMA_SEMAPHORE_HANDLE                                 31:0
#define LW857C_SET_NOTIFIER_CONTROL                                             (0x000000A0)
#define LW857C_SET_NOTIFIER_CONTROL_MODE                                        30:30
#define LW857C_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LW857C_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LW857C_SET_NOTIFIER_CONTROL_OFFSET                                      11:2
#define LW857C_SET_NOTIFIER_CONTROL_DELAY                                       26:26
#define LW857C_SET_NOTIFIER_CONTROL_DELAY_DISABLE                               (0x00000000)
#define LW857C_SET_NOTIFIER_CONTROL_DELAY_ENABLE                                (0x00000001)
#define LW857C_SET_CONTEXT_DMA_NOTIFIER                                         (0x000000A4)
#define LW857C_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LW857C_SET_CONTEXT_DMAS_ISO(b)                                          (0x000000C0 + (b)*0x00000004)
#define LW857C_SET_CONTEXT_DMAS_ISO_HANDLE                                      31:0
#define LW857C_SET_BASE_LUT_LO                                                  (0x000000E0)
#define LW857C_SET_BASE_LUT_LO_ENABLE                                           31:30
#define LW857C_SET_BASE_LUT_LO_ENABLE_DISABLE                                   (0x00000000)
#define LW857C_SET_BASE_LUT_LO_ENABLE_USE_CORE_LUT                              (0x00000001)
#define LW857C_SET_BASE_LUT_LO_ENABLE_ENABLE                                    (0x00000003)
#define LW857C_SET_BASE_LUT_LO_MODE                                             29:29
#define LW857C_SET_BASE_LUT_LO_MODE_LORES                                       (0x00000000)
#define LW857C_SET_BASE_LUT_LO_MODE_HIRES                                       (0x00000001)
#define LW857C_SET_BASE_LUT_LO_ORIGIN                                           7:2
#define LW857C_SET_BASE_LUT_HI                                                  (0x000000E4)
#define LW857C_SET_BASE_LUT_HI_ORIGIN                                           31:0
#define LW857C_SET_OUTPUT_LUT_LO                                                (0x000000E8)
#define LW857C_SET_OUTPUT_LUT_LO_ENABLE                                         31:31
#define LW857C_SET_OUTPUT_LUT_LO_ENABLE_DISABLE                                 (0x00000000)
#define LW857C_SET_OUTPUT_LUT_LO_ENABLE_ENABLE                                  (0x00000001)
#define LW857C_SET_OUTPUT_LUT_LO_MODE                                           30:30
#define LW857C_SET_OUTPUT_LUT_LO_MODE_LORES                                     (0x00000000)
#define LW857C_SET_OUTPUT_LUT_LO_MODE_HIRES                                     (0x00000001)
#define LW857C_SET_OUTPUT_LUT_LO_ORIGIN                                         7:2
#define LW857C_SET_OUTPUT_LUT_HI                                                (0x000000EC)
#define LW857C_SET_OUTPUT_LUT_HI_ORIGIN                                         31:0
#define LW857C_SET_CONTEXT_DMA_LUT                                              (0x000000FC)
#define LW857C_SET_CONTEXT_DMA_LUT_HANDLE                                       31:0
#define LW857C_SET_DIST_RENDER_CONTROL                                          (0x00000100)
#define LW857C_SET_DIST_RENDER_CONTROL_MODE                                     1:0
#define LW857C_SET_DIST_RENDER_CONTROL_MODE_OFF                                 (0x00000000)
#define LW857C_SET_DIST_RENDER_CONTROL_MODE_STRIP_INTERLEAVE                    (0x00000001)
#define LW857C_SET_DIST_RENDER_CONTROL_MODE_FRAME_INTERLEAVE                    (0x00000002)
#define LW857C_SET_DIST_RENDER_CONTROL_MODE_FRAME_AND_STRIP_INTERLEAVE          (0x00000003)
#define LW857C_SET_DIST_RENDER_CONTROL_RENDER_START                             16:2
#define LW857C_SET_DIST_RENDER_CONTROL_RENDER_STOP                              31:17
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL                                   (0x00000104)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_EXTENSION_ENABLE                  31:31
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_EXTENSION_ENABLE_DISABLE          (0x00000000)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_EXTENSION_ENABLE_ENABLE           (0x00000001)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_MODE                              3:0
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_OFF                          (0x00000000)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_FOS_2CHIP                    (0x00000001)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_FOS_4CHIP                    (0x00000002)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_MODE_FRAME_INTERLEAVE_FOS_4CHIP   (0x00000008)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT                9:8
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT_X1             (0x00000000)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT_X2             (0x00000001)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_INPUT_PIXEL_WEIGHT_X4             (0x00000002)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT               13:12
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT_D1            (0x00000000)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT_D2            (0x00000001)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_OUTPUT_PIXEL_WEIGHT_D4            (0x00000002)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE                   17:16
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE_LINEAR            (0x00000000)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE_GAMMA             (0x00000001)
#define LW857C_SET_DIST_RENDER_EXTEND_CONTROL_ARITHMETIC_MODE_NONE              (0x00000003)
#define LW857C_SET_PROCESSING                                                   (0x00000110)
#define LW857C_SET_PROCESSING_USE_GAIN_OFS                                      0:0
#define LW857C_SET_PROCESSING_USE_GAIN_OFS_DISABLE                              (0x00000000)
#define LW857C_SET_PROCESSING_USE_GAIN_OFS_ENABLE                               (0x00000001)
#define LW857C_SET_COLWERSION                                                   (0x00000114)
#define LW857C_SET_COLWERSION_GAIN                                              15:0
#define LW857C_SET_COLWERSION_OFS                                               31:16
#define LW857C_SET_SPARE                                                        (0x000007BC)
#define LW857C_SET_SPARE_UNUSED                                                 31:0
#define LW857C_SET_SPARE_NOOP(b)                                                (0x000007C0 + (b)*0x00000004)
#define LW857C_SET_SPARE_NOOP_UNUSED                                            31:0

#define LW857C_SURFACE_SET_OFFSET(a,b)                                          (0x00000800 + (a)*0x00000020 + (b)*0x00000004)
#define LW857C_SURFACE_SET_OFFSET_ORIGIN                                        31:0
#define LW857C_SURFACE_SET_SIZE(a)                                              (0x00000808 + (a)*0x00000020)
#define LW857C_SURFACE_SET_SIZE_WIDTH                                           14:0
#define LW857C_SURFACE_SET_SIZE_HEIGHT                                          30:16
#define LW857C_SURFACE_SET_STORAGE(a)                                           (0x0000080C + (a)*0x00000020)
#define LW857C_SURFACE_SET_STORAGE_BLOCK_HEIGHT                                 3:0
#define LW857C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_ONE_GOB                         (0x00000000)
#define LW857C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_TWO_GOBS                        (0x00000001)
#define LW857C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_FOUR_GOBS                       (0x00000002)
#define LW857C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_EIGHT_GOBS                      (0x00000003)
#define LW857C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_SIXTEEN_GOBS                    (0x00000004)
#define LW857C_SURFACE_SET_STORAGE_BLOCK_HEIGHT_THIRTYTWO_GOBS                  (0x00000005)
#define LW857C_SURFACE_SET_STORAGE_PITCH                                        19:8
#define LW857C_SURFACE_SET_STORAGE_MEMORY_LAYOUT                                20:20
#define LW857C_SURFACE_SET_STORAGE_MEMORY_LAYOUT_BLOCKLINEAR                    (0x00000000)
#define LW857C_SURFACE_SET_STORAGE_MEMORY_LAYOUT_PITCH                          (0x00000001)
#define LW857C_SURFACE_SET_PARAMS(a)                                            (0x00000810 + (a)*0x00000020)
#define LW857C_SURFACE_SET_PARAMS_FORMAT                                        15:8
#define LW857C_SURFACE_SET_PARAMS_FORMAT_I8                                     (0x0000001E)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_VOID16                                 (0x0000001F)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_VOID32                                 (0x0000002E)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_RF16_GF16_BF16_AF16                    (0x000000CA)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_A8R8G8B8                               (0x000000CF)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_A2B10G10R10                            (0x000000D1)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_A8B8G8R8                               (0x000000D5)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_R5G6B5                                 (0x000000E8)
#define LW857C_SURFACE_SET_PARAMS_FORMAT_A1R5G5B5                               (0x000000E9)
#define LW857C_SURFACE_SET_PARAMS_SUPER_SAMPLE                                  1:0
#define LW857C_SURFACE_SET_PARAMS_SUPER_SAMPLE_X1_AA                            (0x00000000)
#define LW857C_SURFACE_SET_PARAMS_SUPER_SAMPLE_X4_AA                            (0x00000002)
#define LW857C_SURFACE_SET_PARAMS_SUPER_SAMPLE_X8_AA                            (0x00000003)
#define LW857C_SURFACE_SET_PARAMS_GAMMA                                         2:2
#define LW857C_SURFACE_SET_PARAMS_GAMMA_LINEAR                                  (0x00000000)
#define LW857C_SURFACE_SET_PARAMS_GAMMA_SRGB                                    (0x00000001)
#define LW857C_SURFACE_SET_PARAMS_LAYOUT                                        5:4
#define LW857C_SURFACE_SET_PARAMS_LAYOUT_FRM                                    (0x00000000)
#define LW857C_SURFACE_SET_PARAMS_LAYOUT_FLD1                                   (0x00000001)
#define LW857C_SURFACE_SET_PARAMS_LAYOUT_FLD2                                   (0x00000002)
#define LW857C_SURFACE_SET_PARAMS_RESERVED0                                     22:16
#define LW857C_SURFACE_SET_PARAMS_RESERVED1                                     24:24

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl857c_h

