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



#ifndef _cl9878_h_
#define _cl9878_h_


#ifdef __cplusplus
extern "C" {
#endif

#define LW9878_WRITEBACK_CHANNEL_DMA                                            (0x00009878)

typedef volatile struct _cl9878_tag0 {
    LwV32 Put;                                                                  // 0x00000000 - 0x00000003
    LwV32 Get;                                                                  // 0x00000004 - 0x00000007
    LwV32 Reserved00[0x1E];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetPresentControl;                                                    // 0x00000084 - 0x00000087
    LwV32 SetSemaphoreAcquire;                                                  // 0x00000088 - 0x0000008B
    LwV32 SetSemaphoreRelease;                                                  // 0x0000008C - 0x0000008F
    LwV32 SetSemaphoreControl;                                                  // 0x00000090 - 0x00000093
    LwV32 SetContextDmaSemaphore;                                               // 0x00000094 - 0x00000097
    LwV32 SetDescriptorControl;                                                 // 0x00000098 - 0x0000009B
    LwV32 SetContextDmaDescriptor;                                              // 0x0000009C - 0x0000009F
    LwV32 SetNotifierControl;                                                   // 0x000000A0 - 0x000000A3
    LwV32 SetContextDmaNotifier;                                                // 0x000000A4 - 0x000000A7
    LwV32 SetFramePeriod;                                                       // 0x000000A8 - 0x000000AB
    LwV32 SetProtection;                                                        // 0x000000AC - 0x000000AF
    LwV32 Reserved01[0x1];
    LwV32 SetContextDmaSurface[3];                                              // 0x000000B4 - 0x000000BF
    LwV32 Reserved02[0xD0];
    struct {
        LwV32 SetOffset[2];                                                     // 0x00000400 - 0x00000407
        LwV32 Reserved03[0x1];
        LwV32 SetStorage;                                                       // 0x0000040C - 0x0000040F
        LwV32 SetParams;                                                        // 0x00000410 - 0x00000413
        LwV32 Reserved04[0x3];
    } Surface[3];
    LwV32 Reserved05[0x2E8];
} LW9878DispControlDma;


#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1                                        0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_SIZEOF                                 0x00000030
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0                              0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_BLOCK_HEIGHT                 3:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_BLOCK_HEIGHT_ONE_GOB         0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_BLOCK_HEIGHT_TWO_GOBS        0x00000001
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_BLOCK_HEIGHT_FOUR_GOBS       0x00000002
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_BLOCK_HEIGHT_EIGHT_GOBS      0x00000003
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_BLOCK_HEIGHT_SIXTEEN_GOBS    0x00000004
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_BLOCK_HEIGHT_THIRTYTWO_GOBS  0x00000005
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_MEMORY_LAYOUT                4:4
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_MEMORY_LAYOUT_BLOCKLINEAR    0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_MEMORY_LAYOUT_PITCH          0x00000001
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_R0                           7:5
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STORAGE_0_PITCH                        20:8
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__1                                     0x00000001
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__1_WIDTH                               15:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__1_HEIGHT                              31:16
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2                                     0x00000002
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT                        3:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y8___U8V8_N420         0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y10___U10V10_N420      0x00000001
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y12___U12V12_N420      0x00000002
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y8___U8V8_N444         0x00000003
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y10___U10V10_N444      0x00000004
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y12___U12V12_N444      0x00000005
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y8___U8___V8_N444      0x00000008
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y10___U10___V10_N444   0x00000009
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_Y12___U12___V12_N444   0x0000000A
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_A2R10G10B10            0x0000000E
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_COLOR_FORMAT_A8R8G8B8               0x0000000F
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_WBOR_INDEX                          5:4
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_R0                                  7:6
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_STEREO_STATUS                       10:8
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_STEREO_STATUS_MONO                  0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_STEREO_STATUS_STEREO_LEFT           0x00000001
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_STEREO_STATUS_STEREO_RIGHT          0x00000002
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_STEREO_STATUS_STEREO_PAIR           0x00000003
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_STEREO_STATUS_STEREO_PACKED_TB      0x00000004
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__2_R1                                  31:11
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__3                                     0x00000003
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__3_R2                                  31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__4                                     0x00000004
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__4_R3                                  31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__5                                     0x00000005
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__5_R4                                  31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_BEGIN_TS_6                             0x00000006
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_BEGIN_TS_6_NANOSECONDS0                31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_BEGIN_TS_7                             0x00000007
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_BEGIN_TS_7_NANOSECONDS1                31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_DONE_TS_8                              0x00000008
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_DONE_TS_8_NANOSECONDS0                 31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_DONE_TS_9                              0x00000009
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_DONE_TS_9_NANOSECONDS1                 31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__10                                    0x0000000A
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1__10_R5                                 31:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11                              0x0000000B
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_LINE_INDEX_L                 12:0
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_LINE_INDEX_R                 25:13
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_TYPE                         28:26
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_TYPE_NON_STARTED             0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_TYPE_IN_PROGRESS             0x00000001
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_TYPE_DONE                    0x00000002
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_TYPE_SKIPPED                 0x00000003
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_TYPE_REPEATED                0x00000004
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_TYPE_MEMORY_EXCEPTION        0x00000005
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_VPR                          29:29
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_VPR_FALSE                    0x00000000
#define LW_DISP_WRITE_BACK_FRAME_DESCRIPTOR_1_STATUS_11_VPR_TRUE                     0x00000001


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
#define LW9878_DMA                                                         0x00000000 
#define LW9878_DMA_OPCODE                                                       31:29 
#define LW9878_DMA_OPCODE_METHOD                                           0x00000000 
#define LW9878_DMA_OPCODE_JUMP                                             0x00000001 
#define LW9878_DMA_OPCODE_NONINC_METHOD                                    0x00000002 
#define LW9878_DMA_OPCODE_SET_SUBDEVICE_MASK                               0x00000003 
#define LW9878_DMA_METHOD_COUNT                                                 27:18 
#define LW9878_DMA_METHOD_OFFSET                                                 11:2 
#define LW9878_DMA_DATA                                                          31:0 
#define LW9878_DMA_DATA_NOP                                                0x00000000 
#define LW9878_DMA_JUMP_OFFSET                                                   11:2 
#define LW9878_DMA_SET_SUBDEVICE_MASK_VALUE                                      11:0 

// class methods
#define LW9878_PUT                                                              (0x00000000)
#define LW9878_PUT_PTR                                                          11:2
#define LW9878_GET                                                              (0x00000004)
#define LW9878_GET_PTR                                                          11:2
#define LW9878_UPDATE                                                           (0x00000080)
#define LW9878_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW9878_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW9878_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW9878_UPDATE_SPECIAL_HANDLING                                          25:24
#define LW9878_UPDATE_SPECIAL_HANDLING_NONE                                     (0x00000000)
#define LW9878_UPDATE_SPECIAL_HANDLING_INTERRUPT_RM                             (0x00000001)
#define LW9878_UPDATE_SPECIAL_HANDLING_MODE_SWITCH                              (0x00000002)
#define LW9878_UPDATE_SPECIAL_HANDLING_REASON                                   23:16
#define LW9878_UPDATE_GET_FRAME                                                 1:1
#define LW9878_UPDATE_GET_FRAME_DISABLE                                         (0x00000000)
#define LW9878_UPDATE_GET_FRAME_ENABLE                                          (0x00000001)
#define LW9878_SET_PRESENT_CONTROL                                              (0x00000084)
#define LW9878_SET_PRESENT_CONTROL_BEGIN_MODE                                   1:0
#define LW9878_SET_PRESENT_CONTROL_BEGIN_MODE_ASAP                              (0x00000000)
#define LW9878_SET_PRESENT_CONTROL_BEGIN_MODE_FRAME_BOUNDARY                    (0x00000001)
#define LW9878_SET_PRESENT_CONTROL_MIN_GRAB_INTERVAL                            7:4
#define LW9878_SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL                         11:8
#define LW9878_SET_PRESENT_CONTROL_REPORT_MODE                                  3:2
#define LW9878_SET_PRESENT_CONTROL_REPORT_MODE_BEGIN                            (0x00000000)
#define LW9878_SET_PRESENT_CONTROL_REPORT_MODE_FINISH                           (0x00000001)
#define LW9878_SET_SEMAPHORE_ACQUIRE                                            (0x00000088)
#define LW9878_SET_SEMAPHORE_ACQUIRE_VALUE                                      31:0
#define LW9878_SET_SEMAPHORE_RELEASE                                            (0x0000008C)
#define LW9878_SET_SEMAPHORE_RELEASE_VALUE                                      31:0
#define LW9878_SET_SEMAPHORE_CONTROL                                            (0x00000090)
#define LW9878_SET_SEMAPHORE_CONTROL_OFFSET                                     11:2
#define LW9878_SET_SEMAPHORE_CONTROL_FORMAT                                     0:0
#define LW9878_SET_SEMAPHORE_CONTROL_FORMAT_LEGACY                              (0x00000000)
#define LW9878_SET_SEMAPHORE_CONTROL_FORMAT_FOUR_WORD                           (0x00000001)
#define LW9878_SET_CONTEXT_DMA_SEMAPHORE                                        (0x00000094)
#define LW9878_SET_CONTEXT_DMA_SEMAPHORE_HANDLE                                 31:0
#define LW9878_SET_DESCRIPTOR_CONTROL                                           (0x00000098)
#define LW9878_SET_DESCRIPTOR_CONTROL_OFFSET                                    11:2
#define LW9878_SET_CONTEXT_DMA_DESCRIPTOR                                       (0x0000009C)
#define LW9878_SET_CONTEXT_DMA_DESCRIPTOR_HANDLE                                31:0
#define LW9878_SET_NOTIFIER_CONTROL                                             (0x000000A0)
#define LW9878_SET_NOTIFIER_CONTROL_MODE                                        30:30
#define LW9878_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LW9878_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LW9878_SET_NOTIFIER_CONTROL_OFFSET                                      11:2
#define LW9878_SET_NOTIFIER_CONTROL_FORMAT                                      0:0
#define LW9878_SET_NOTIFIER_CONTROL_FORMAT_LEGACY                               (0x00000000)
#define LW9878_SET_NOTIFIER_CONTROL_FORMAT_FOUR_WORD                            (0x00000001)
#define LW9878_SET_CONTEXT_DMA_NOTIFIER                                         (0x000000A4)
#define LW9878_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LW9878_SET_FRAME_PERIOD                                                 (0x000000A8)
#define LW9878_SET_FRAME_PERIOD_VALUE                                           31:0
#define LW9878_SET_PROTECTION                                                   (0x000000AC)
#define LW9878_SET_PROTECTION_FORCE                                             0:0
#define LW9878_SET_PROTECTION_FORCE_FALSE                                       (0x00000000)
#define LW9878_SET_PROTECTION_FORCE_TRUE                                        (0x00000001)
#define LW9878_SET_CONTEXT_DMA_SURFACE(b)                                       (0x000000B4 + (b)*0x00000004)
#define LW9878_SET_CONTEXT_DMA_SURFACE_HANDLE                                   31:0

#define LW9878_SURFACE_SET_OFFSET(a,b)                                          (0x00000400 + (a)*0x00000020 + (b)*0x00000004)
#define LW9878_SURFACE_SET_OFFSET_ORIGIN                                        31:0
#define LW9878_SURFACE_SET_STORAGE(a)                                           (0x0000040C + (a)*0x00000020)
#define LW9878_SURFACE_SET_STORAGE_BLOCK_HEIGHT                                 3:0
#define LW9878_SURFACE_SET_STORAGE_BLOCK_HEIGHT_ONE_GOB                         (0x00000000)
#define LW9878_SURFACE_SET_STORAGE_BLOCK_HEIGHT_TWO_GOBS                        (0x00000001)
#define LW9878_SURFACE_SET_STORAGE_BLOCK_HEIGHT_FOUR_GOBS                       (0x00000002)
#define LW9878_SURFACE_SET_STORAGE_BLOCK_HEIGHT_EIGHT_GOBS                      (0x00000003)
#define LW9878_SURFACE_SET_STORAGE_BLOCK_HEIGHT_SIXTEEN_GOBS                    (0x00000004)
#define LW9878_SURFACE_SET_STORAGE_BLOCK_HEIGHT_THIRTYTWO_GOBS                  (0x00000005)
#define LW9878_SURFACE_SET_STORAGE_PITCH                                        20:8
#define LW9878_SURFACE_SET_STORAGE_MEMORY_LAYOUT                                24:24
#define LW9878_SURFACE_SET_STORAGE_MEMORY_LAYOUT_BLOCKLINEAR                    (0x00000000)
#define LW9878_SURFACE_SET_STORAGE_MEMORY_LAYOUT_PITCH                          (0x00000001)
#define LW9878_SURFACE_SET_PARAMS(a)                                            (0x00000410 + (a)*0x00000020)
#define LW9878_SURFACE_SET_PARAMS_FORMAT                                        11:8
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y8___U8V8_N420                         (0x00000000)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y10___U10V10_N420                      (0x00000001)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y12___U12V12_N420                      (0x00000002)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y8___U8V8_N444                         (0x00000003)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y10___U10V10_N444                      (0x00000004)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y12___U12V12_N444                      (0x00000005)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y8___U8___V8_N444                      (0x00000008)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y10___U10___V10_N444                   (0x00000009)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_Y12___U12___V12_N444                   (0x0000000A)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_A2R10G10B10                            (0x0000000E)
#define LW9878_SURFACE_SET_PARAMS_FORMAT_A8R8G8B8                               (0x0000000F)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl9878_h
