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


#ifndef _cl857e_h_
#define _cl857e_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW857E_OVERLAY_CHANNEL_DMA                                              (0x0000857E)

typedef volatile struct _cl857e_tag0 {
    LwV32 Put;                                                                  // 0x00000000 - 0x00000003
    LwV32 Get;                                                                  // 0x00000004 - 0x00000007
    LwV32 Reserved00[0x1E];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetPresentControl;                                                    // 0x00000084 - 0x00000087
    LwV32 SetSemaphoreAcquire;                                                  // 0x00000088 - 0x0000008B
    LwV32 SetSemaphoreRelease;                                                  // 0x0000008C - 0x0000008F
    LwV32 SetSemaphoreControl;                                                  // 0x00000090 - 0x00000093
    LwV32 SetContextDmaSemaphore;                                               // 0x00000094 - 0x00000097
    LwV32 Reserved01[0x2];
    LwV32 SetNotifierControl;                                                   // 0x000000A0 - 0x000000A3
    LwV32 SetContextDmaNotifier;                                                // 0x000000A4 - 0x000000A7
    LwV32 Reserved02[0x2];
    LwV32 SetContextDmaLut;                                                     // 0x000000B0 - 0x000000B3
    LwV32 SetOverlayLutLo;                                                      // 0x000000B4 - 0x000000B7
    LwV32 SetOverlayLutHi;                                                      // 0x000000B8 - 0x000000BB
    LwV32 Reserved03[0x1];
    LwV32 SetContextDmaIso;                                                     // 0x000000C0 - 0x000000C3
    LwV32 Reserved04[0x7];
    LwV32 SetPointIn;                                                           // 0x000000E0 - 0x000000E3
    LwV32 SetSizeIn;                                                            // 0x000000E4 - 0x000000E7
    LwV32 SetSizeOut;                                                           // 0x000000E8 - 0x000000EB
    LwV32 Reserved05[0x5];
    LwV32 SetCompositionControl;                                                // 0x00000100 - 0x00000103
    LwV32 SetKeyColor;                                                          // 0x00000104 - 0x00000107
    LwV32 SetKeyMask;                                                           // 0x00000108 - 0x0000010B
    LwV32 Reserved06[0x9];
    LwV32 SetTimestampOriginLo;                                                 // 0x00000130 - 0x00000133
    LwV32 SetTimestampOriginHi;                                                 // 0x00000134 - 0x00000137
    LwV32 SetUpdateTimestampLo;                                                 // 0x00000138 - 0x0000013B
    LwV32 SetUpdateTimestampHi;                                                 // 0x0000013C - 0x0000013F
    LwV32 Reserved07[0x19F];
    LwV32 SetSpare;                                                             // 0x000007BC - 0x000007BF
    LwV32 SetSpareNoop[16];                                                     // 0x000007C0 - 0x000007FF
    struct {
        LwV32 SetOffset;                                                        // 0x00000800 - 0x00000803
        LwV32 Reserved08[0x1];
        LwV32 SetSize;                                                          // 0x00000808 - 0x0000080B
        LwV32 SetStorage;                                                       // 0x0000080C - 0x0000080F
        LwV32 SetParams;                                                        // 0x00000810 - 0x00000813
        LwV32 Reserved09[0x3];
    } Surface[1];
    LwV32 Reserved10[0x1F8];
} GT214DispOverlayControlDma;


#define LW_DISP_NOTIFICATION_1                                                       0x00000000
#define LW_DISP_NOTIFICATION_1_SIZEOF                                                0x00000010
#define LW_DISP_NOTIFICATION_1_TIME_STAMP_0                                          0x00000000
#define LW_DISP_NOTIFICATION_1_TIME_STAMP_0_NANOSECONDS0                             31:0
#define LW_DISP_NOTIFICATION_1_TIME_STAMP_1                                          0x00000001
#define LW_DISP_NOTIFICATION_1_TIME_STAMP_1_NANOSECONDS1                             31:0
#define LW_DISP_NOTIFICATION_1__2                                                    0x00000002
#define LW_DISP_NOTIFICATION_1__2_AUDIT_TIMESTAMP                                    31:0
#define LW_DISP_NOTIFICATION_1__3                                                    0x00000003
#define LW_DISP_NOTIFICATION_1__3_PRESENT_COUNT                                      7:0
#define LW_DISP_NOTIFICATION_1__3_R0                                                 15:8
#define LW_DISP_NOTIFICATION_1__3_STATUS                                             31:16
#define LW_DISP_NOTIFICATION_1__3_STATUS_NOT_BEGUN                                   0x00008000
#define LW_DISP_NOTIFICATION_1__3_STATUS_BEGUN                                       0x0000FFFF
#define LW_DISP_NOTIFICATION_1__3_STATUS_FINISHED                                    0x00000000


#define LW_DISP_NOTIFICATION_INFO16                                                  0x00000000
#define LW_DISP_NOTIFICATION_INFO16_SIZEOF                                           0x00000002
#define LW_DISP_NOTIFICATION_INFO16__0                                               0x00000000
#define LW_DISP_NOTIFICATION_INFO16__0_PRESENT_COUNT                                 7:0
#define LW_DISP_NOTIFICATION_INFO16__0_R0                                            15:8


#define LW_DISP_NOTIFICATION_STATUS                                                  0x00000000
#define LW_DISP_NOTIFICATION_STATUS_SIZEOF                                           0x00000002
#define LW_DISP_NOTIFICATION_STATUS__0                                               0x00000000
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS                                        15:0
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS_NOT_BEGUN                              0x00008000
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS_BEGUN                                  0x0000FFFF
#define LW_DISP_NOTIFICATION_STATUS__0_STATUS_FINISHED                               0x00000000


// dma opcode instructions
#define LW857E_DMA                                                                              0x00000000 
#define LW857E_DMA_OPCODE                                                                            31:29 
#define LW857E_DMA_OPCODE_METHOD                                                                0x00000000 
#define LW857E_DMA_OPCODE_JUMP                                                                  0x00000001 
#define LW857E_DMA_OPCODE_NONINC_METHOD                                                         0x00000002 
#define LW857E_DMA_OPCODE_SET_SUBDEVICE_MASK                                                    0x00000003 
#define LW857E_DMA_OPCODE                                                                            31:29 
#define LW857E_DMA_OPCODE_METHOD                                                                0x00000000 
#define LW857E_DMA_OPCODE_NONINC_METHOD                                                         0x00000002 
#define LW857E_DMA_METHOD_COUNT                                                                      27:18 
#define LW857E_DMA_METHOD_OFFSET                                                                      11:2 
#define LW857E_DMA_DATA                                                                               31:0 
#define LW857E_DMA_DATA_NOP                                                                     0x00000000 
#define LW857E_DMA_OPCODE                                                                            31:29 
#define LW857E_DMA_OPCODE_JUMP                                                                  0x00000001 
#define LW857E_DMA_JUMP_OFFSET                                                                        11:2 
#define LW857E_DMA_OPCODE                                                                            31:29 
#define LW857E_DMA_OPCODE_SET_SUBDEVICE_MASK                                                    0x00000003 
#define LW857E_DMA_SET_SUBDEVICE_MASK_VALUE                                                           11:0 

// class methods
#define LW857E_PUT                                                              (0x00000000)
#define LW857E_PUT_PTR                                                          11:2
#define LW857E_GET                                                              (0x00000004)
#define LW857E_GET_PTR                                                          11:2
#define LW857E_UPDATE                                                           (0x00000080)
#define LW857E_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW857E_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW857E_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW857E_SET_PRESENT_CONTROL                                              (0x00000084)
#define LW857E_SET_PRESENT_CONTROL_BEGIN_MODE                                   1:0
#define LW857E_SET_PRESENT_CONTROL_BEGIN_MODE_ASAP                              (0x00000000)
#define LW857E_SET_PRESENT_CONTROL_BEGIN_MODE_TIMESTAMP                         (0x00000003)
#define LW857E_SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL                         7:4
#define LW857E_SET_SEMAPHORE_ACQUIRE                                            (0x00000088)
#define LW857E_SET_SEMAPHORE_ACQUIRE_VALUE                                      31:0
#define LW857E_SET_SEMAPHORE_RELEASE                                            (0x0000008C)
#define LW857E_SET_SEMAPHORE_RELEASE_VALUE                                      31:0
#define LW857E_SET_SEMAPHORE_CONTROL                                            (0x00000090)
#define LW857E_SET_SEMAPHORE_CONTROL_OFFSET                                     11:2
#define LW857E_SET_CONTEXT_DMA_SEMAPHORE                                        (0x00000094)
#define LW857E_SET_CONTEXT_DMA_SEMAPHORE_HANDLE                                 31:0
#define LW857E_SET_NOTIFIER_CONTROL                                             (0x000000A0)
#define LW857E_SET_NOTIFIER_CONTROL_MODE                                        30:30
#define LW857E_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LW857E_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LW857E_SET_NOTIFIER_CONTROL_OFFSET                                      11:2
#define LW857E_SET_CONTEXT_DMA_NOTIFIER                                         (0x000000A4)
#define LW857E_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LW857E_SET_CONTEXT_DMA_LUT                                              (0x000000B0)
#define LW857E_SET_CONTEXT_DMA_LUT_HANDLE                                       31:0
#define LW857E_SET_OVERLAY_LUT_LO                                               (0x000000B4)
#define LW857E_SET_OVERLAY_LUT_LO_ENABLE                                        30:30
#define LW857E_SET_OVERLAY_LUT_LO_ENABLE_DISABLE                                (0x00000000)
#define LW857E_SET_OVERLAY_LUT_LO_ENABLE_ENABLE                                 (0x00000001)
#define LW857E_SET_OVERLAY_LUT_LO_MODE                                          29:29
#define LW857E_SET_OVERLAY_LUT_LO_MODE_LORES                                    (0x00000000)
#define LW857E_SET_OVERLAY_LUT_LO_MODE_HIRES                                    (0x00000001)
#define LW857E_SET_OVERLAY_LUT_LO_ORIGIN                                        7:2
#define LW857E_SET_OVERLAY_LUT_HI                                               (0x000000B8)
#define LW857E_SET_OVERLAY_LUT_HI_ORIGIN                                        31:0
#define LW857E_SET_CONTEXT_DMA_ISO                                              (0x000000C0)
#define LW857E_SET_CONTEXT_DMA_ISO_HANDLE                                       31:0
#define LW857E_SET_POINT_IN                                                     (0x000000E0)
#define LW857E_SET_POINT_IN_X                                                   14:0
#define LW857E_SET_POINT_IN_Y                                                   30:16
#define LW857E_SET_SIZE_IN                                                      (0x000000E4)
#define LW857E_SET_SIZE_IN_WIDTH                                                14:0
#define LW857E_SET_SIZE_IN_HEIGHT                                               30:16
#define LW857E_SET_SIZE_OUT                                                     (0x000000E8)
#define LW857E_SET_SIZE_OUT_WIDTH                                               14:0
#define LW857E_SET_COMPOSITION_CONTROL                                          (0x00000100)
#define LW857E_SET_COMPOSITION_CONTROL_MODE                                     3:0
#define LW857E_SET_COMPOSITION_CONTROL_MODE_SOURCE_COLOR_VALUE_KEYING           (0x00000000)
#define LW857E_SET_COMPOSITION_CONTROL_MODE_DESTINATION_COLOR_VALUE_KEYING      (0x00000001)
#define LW857E_SET_COMPOSITION_CONTROL_MODE_OPAQUE_SUSPEND_BASE                 (0x00000002)
#define LW857E_SET_KEY_COLOR                                                    (0x00000104)
#define LW857E_SET_KEY_COLOR_COLOR                                              31:0
#define LW857E_SET_KEY_MASK                                                     (0x00000108)
#define LW857E_SET_KEY_MASK_MASK                                                31:0
#define LW857E_SET_TIMESTAMP_ORIGIN_LO                                          (0x00000130)
#define LW857E_SET_TIMESTAMP_ORIGIN_LO_TIMESTAMP_LO                             31:0
#define LW857E_SET_TIMESTAMP_ORIGIN_HI                                          (0x00000134)
#define LW857E_SET_TIMESTAMP_ORIGIN_HI_TIMESTAMP_HI                             31:0
#define LW857E_SET_UPDATE_TIMESTAMP_LO                                          (0x00000138)
#define LW857E_SET_UPDATE_TIMESTAMP_LO_TIMESTAMP_LO                             31:0
#define LW857E_SET_UPDATE_TIMESTAMP_HI                                          (0x0000013C)
#define LW857E_SET_UPDATE_TIMESTAMP_HI_TIMESTAMP_HI                             31:0
#define LW857E_SET_SPARE                                                        (0x000007BC)
#define LW857E_SET_SPARE_UNUSED                                                 31:0
#define LW857E_SET_SPARE_NOOP(b)                                                (0x000007C0 + (b)*0x00000004)
#define LW857E_SET_SPARE_NOOP_UNUSED                                            31:0

#define LW857E_SURFACE_SET_OFFSET                                               (0x00000800)
#define LW857E_SURFACE_SET_OFFSET_ORIGIN                                        31:0
#define LW857E_SURFACE_SET_SIZE                                                 (0x00000808)
#define LW857E_SURFACE_SET_SIZE_WIDTH                                           14:0
#define LW857E_SURFACE_SET_SIZE_HEIGHT                                          30:16
#define LW857E_SURFACE_SET_STORAGE                                              (0x0000080C)
#define LW857E_SURFACE_SET_STORAGE_BLOCK_HEIGHT                                 3:0
#define LW857E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_ONE_GOB                         (0x00000000)
#define LW857E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_TWO_GOBS                        (0x00000001)
#define LW857E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_FOUR_GOBS                       (0x00000002)
#define LW857E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_EIGHT_GOBS                      (0x00000003)
#define LW857E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_SIXTEEN_GOBS                    (0x00000004)
#define LW857E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_THIRTYTWO_GOBS                  (0x00000005)
#define LW857E_SURFACE_SET_STORAGE_PITCH                                        19:8
#define LW857E_SURFACE_SET_STORAGE_MEMORY_LAYOUT                                20:20
#define LW857E_SURFACE_SET_STORAGE_MEMORY_LAYOUT_BLOCKLINEAR                    (0x00000000)
#define LW857E_SURFACE_SET_STORAGE_MEMORY_LAYOUT_PITCH                          (0x00000001)
#define LW857E_SURFACE_SET_PARAMS                                               (0x00000810)
#define LW857E_SURFACE_SET_PARAMS_FORMAT                                        15:8
#define LW857E_SURFACE_SET_PARAMS_FORMAT_VE8YO8UE8YE8                           (0x00000028)
#define LW857E_SURFACE_SET_PARAMS_FORMAT_YO8VE8YE8UE8                           (0x00000029)
#define LW857E_SURFACE_SET_PARAMS_FORMAT_A2B10G10R10                            (0x000000D1)
#define LW857E_SURFACE_SET_PARAMS_FORMAT_A8R8G8B8                               (0x000000CF)
#define LW857E_SURFACE_SET_PARAMS_FORMAT_A1R5G5B5                               (0x000000E9)
#define LW857E_SURFACE_SET_PARAMS_COLOR_SPACE                                   1:0
#define LW857E_SURFACE_SET_PARAMS_COLOR_SPACE_RGB                               (0x00000000)
#define LW857E_SURFACE_SET_PARAMS_COLOR_SPACE_YUV_601                           (0x00000001)
#define LW857E_SURFACE_SET_PARAMS_COLOR_SPACE_YUV_709                           (0x00000002)
#define LW857E_SURFACE_SET_PARAMS_RESERVED0                                     22:16
#define LW857E_SURFACE_SET_PARAMS_RESERVED1                                     24:24

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl857e_h

