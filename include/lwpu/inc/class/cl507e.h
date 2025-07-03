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


#ifndef _cl507e_h_
#define _cl507e_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW507E_OVERLAY_CHANNEL_DMA                                              (0x0000507E)

typedef volatile struct _cl507e_tag0 {
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
    LwV32 Reserved02[0x6];
    LwV32 SetContextDmaIso;                                                     // 0x000000C0 - 0x000000C3
    LwV32 Reserved03[0x7];
    LwV32 SetPointIn;                                                           // 0x000000E0 - 0x000000E3
    LwV32 SetSizeIn;                                                            // 0x000000E4 - 0x000000E7
    LwV32 SetSizeOut;                                                           // 0x000000E8 - 0x000000EB
    LwV32 Reserved04[0x5];
    LwV32 SetCompositionControl;                                                // 0x00000100 - 0x00000103
    LwV32 SetKeyColor;                                                          // 0x00000104 - 0x00000107
    LwV32 SetKeyMask;                                                           // 0x00000108 - 0x0000010B
    LwV32 Reserved05[0x5];
    LwV32 SetTimestampValue;                                                    // 0x00000120 - 0x00000123
    LwV32 SetUpdateTimestamp;                                                   // 0x00000124 - 0x00000127
    LwV32 Reserved06[0x1A5];
    LwV32 SetSpare;                                                             // 0x000007BC - 0x000007BF
    LwV32 SetSpareNoop[16];                                                     // 0x000007C0 - 0x000007FF
    struct {
        LwV32 SetOffset;                                                        // 0x00000800 - 0x00000803
        LwV32 Reserved07[0x1];
        LwV32 SetSize;                                                          // 0x00000808 - 0x0000080B
        LwV32 SetStorage;                                                       // 0x0000080C - 0x0000080F
        LwV32 SetParams;                                                        // 0x00000810 - 0x00000813
        LwV32 Reserved08[0x3];
    } Surface[1];
    LwV32 Reserved09[0x1F8];
} Lw50DispOverlayControlDma;


#define LW_DISP_OVERLAY_NOTIFIER_1                                                   0x00000000
#define LW_DISP_OVERLAY_NOTIFIER_1_SIZEOF                                            0x00000008
#define LW_DISP_OVERLAY_NOTIFIER_1__0                                                0x00000000
#define LW_DISP_OVERLAY_NOTIFIER_1__0_PRESENT_COUNT                                  15:0
#define LW_DISP_OVERLAY_NOTIFIER_1__0_TIMESTAMP                                      29:16
#define LW_DISP_OVERLAY_NOTIFIER_1__0_STATUS                                         31:30
#define LW_DISP_OVERLAY_NOTIFIER_1__0_STATUS_NOT_BEGUN                               0x00000000
#define LW_DISP_OVERLAY_NOTIFIER_1__0_STATUS_BEGUN                                   0x00000001
#define LW_DISP_OVERLAY_NOTIFIER_1__0_STATUS_FINISHED                                0x00000002
#define LW_DISP_OVERLAY_NOTIFIER_1__1                                                0x00000001
#define LW_DISP_OVERLAY_NOTIFIER_1__1_PRESENT_START_TIME                             31:0


// dma opcode instructions
#define LW507E_DMA                                     0x00000000 
#define LW507E_DMA_OPCODE                                   31:29 
#define LW507E_DMA_OPCODE_METHOD                       0x00000000 
#define LW507E_DMA_OPCODE_JUMP                         0x00000001 
#define LW507E_DMA_OPCODE_NONINC_METHOD                0x00000002 
#define LW507E_DMA_OPCODE_SET_SUBDEVICE_MASK           0x00000003 
#define LW507E_DMA_OPCODE                                   31:29 
#define LW507E_DMA_OPCODE_METHOD                       0x00000000 
#define LW507E_DMA_OPCODE_NONINC_METHOD                0x00000002 
#define LW507E_DMA_METHOD_COUNT                             27:18 
#define LW507E_DMA_METHOD_OFFSET                             11:2 
#define LW507E_DMA_DATA                                      31:0 
#define LW507E_DMA_NOP                                 0x00000000 
#define LW507E_DMA_OPCODE                                   31:29 
#define LW507E_DMA_OPCODE_JUMP                         0x00000001 
#define LW507E_DMA_JUMP_OFFSET                               11:2 
#define LW507E_DMA_OPCODE                                   31:29 
#define LW507E_DMA_OPCODE_SET_SUBDEVICE_MASK           0x00000003 
#define LW507E_DMA_SET_SUBDEVICE_MASK_VALUE                  11:0 

// class methods
#define LW507E_PUT                                                              (0x00000000)
#define LW507E_PUT_PTR                                                          11:2
#define LW507E_GET                                                              (0x00000004)
#define LW507E_GET_PTR                                                          11:2
#define LW507E_UPDATE                                                           (0x00000080)
#define LW507E_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW507E_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW507E_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW507E_SET_PRESENT_CONTROL                                              (0x00000084)
#define LW507E_SET_PRESENT_CONTROL_BEGIN_MODE                                   1:0
#define LW507E_SET_PRESENT_CONTROL_BEGIN_MODE_ASAP                              (0x00000000)
#define LW507E_SET_PRESENT_CONTROL_BEGIN_MODE_TIMESTAMP                         (0x00000003)
#define LW507E_SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL                         7:4
#define LW507E_SET_SEMAPHORE_ACQUIRE                                            (0x00000088)
#define LW507E_SET_SEMAPHORE_ACQUIRE_VALUE                                      31:0
#define LW507E_SET_SEMAPHORE_RELEASE                                            (0x0000008C)
#define LW507E_SET_SEMAPHORE_RELEASE_VALUE                                      31:0
#define LW507E_SET_SEMAPHORE_CONTROL                                            (0x00000090)
#define LW507E_SET_SEMAPHORE_CONTROL_OFFSET                                     11:2
#define LW507E_SET_CONTEXT_DMA_SEMAPHORE                                        (0x00000094)
#define LW507E_SET_CONTEXT_DMA_SEMAPHORE_HANDLE                                 31:0
#define LW507E_SET_NOTIFIER_CONTROL                                             (0x000000A0)
#define LW507E_SET_NOTIFIER_CONTROL_MODE                                        30:30
#define LW507E_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LW507E_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LW507E_SET_NOTIFIER_CONTROL_OFFSET                                      11:2
#define LW507E_SET_CONTEXT_DMA_NOTIFIER                                         (0x000000A4)
#define LW507E_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LW507E_SET_CONTEXT_DMA_ISO                                              (0x000000C0)
#define LW507E_SET_CONTEXT_DMA_ISO_HANDLE                                       31:0
#define LW507E_SET_POINT_IN                                                     (0x000000E0)
#define LW507E_SET_POINT_IN_X                                                   14:0
#define LW507E_SET_POINT_IN_Y                                                   30:16
#define LW507E_SET_SIZE_IN                                                      (0x000000E4)
#define LW507E_SET_SIZE_IN_WIDTH                                                14:0
#define LW507E_SET_SIZE_IN_HEIGHT                                               30:16
#define LW507E_SET_SIZE_OUT                                                     (0x000000E8)
#define LW507E_SET_SIZE_OUT_WIDTH                                               14:0
#define LW507E_SET_COMPOSITION_CONTROL                                          (0x00000100)
#define LW507E_SET_COMPOSITION_CONTROL_MODE                                     3:0
#define LW507E_SET_COMPOSITION_CONTROL_MODE_SOURCE_COLOR_VALUE_KEYING           (0x00000000)
#define LW507E_SET_COMPOSITION_CONTROL_MODE_DESTINATION_COLOR_VALUE_KEYING      (0x00000001)
#define LW507E_SET_COMPOSITION_CONTROL_MODE_OPAQUE_SUSPEND_BASE                 (0x00000002)
#define LW507E_SET_KEY_COLOR                                                    (0x00000104)
#define LW507E_SET_KEY_COLOR_COLOR                                              31:0
#define LW507E_SET_KEY_MASK                                                     (0x00000108)
#define LW507E_SET_KEY_MASK_MASK                                                31:0
#define LW507E_SET_TIMESTAMP_VALUE                                              (0x00000120)
#define LW507E_SET_TIMESTAMP_VALUE_TIMESTAMP                                    31:0
#define LW507E_SET_UPDATE_TIMESTAMP                                             (0x00000124)
#define LW507E_SET_UPDATE_TIMESTAMP_TIMESTAMP                                   31:0
#define LW507E_SET_SPARE                                                        (0x000007BC)
#define LW507E_SET_SPARE_UNUSED                                                 31:0
#define LW507E_SET_SPARE_NOOP(b)                                                (0x000007C0 + (b)*0x00000004)
#define LW507E_SET_SPARE_NOOP_UNUSED                                            31:0

#define LW507E_SURFACE_SET_OFFSET                                               (0x00000800)
#define LW507E_SURFACE_SET_OFFSET_ORIGIN                                        31:0
#define LW507E_SURFACE_SET_SIZE                                                 (0x00000808)
#define LW507E_SURFACE_SET_SIZE_WIDTH                                           14:0
#define LW507E_SURFACE_SET_SIZE_HEIGHT                                          30:16
#define LW507E_SURFACE_SET_STORAGE                                              (0x0000080C)
#define LW507E_SURFACE_SET_STORAGE_BLOCK_HEIGHT                                 3:0
#define LW507E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_ONE_GOB                         (0x00000000)
#define LW507E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_TWO_GOBS                        (0x00000001)
#define LW507E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_FOUR_GOBS                       (0x00000002)
#define LW507E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_EIGHT_GOBS                      (0x00000003)
#define LW507E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_SIXTEEN_GOBS                    (0x00000004)
#define LW507E_SURFACE_SET_STORAGE_BLOCK_HEIGHT_THIRTYTWO_GOBS                  (0x00000005)
#define LW507E_SURFACE_SET_STORAGE_PITCH                                        17:8
#define LW507E_SURFACE_SET_STORAGE_MEMORY_LAYOUT                                20:20
#define LW507E_SURFACE_SET_STORAGE_MEMORY_LAYOUT_BLOCKLINEAR                    (0x00000000)
#define LW507E_SURFACE_SET_STORAGE_MEMORY_LAYOUT_PITCH                          (0x00000001)
#define LW507E_SURFACE_SET_PARAMS                                               (0x00000810)
#define LW507E_SURFACE_SET_PARAMS_FORMAT                                        15:8
#define LW507E_SURFACE_SET_PARAMS_FORMAT_VE8YO8UE8YE8                           (0x00000028)
#define LW507E_SURFACE_SET_PARAMS_FORMAT_YO8VE8YE8UE8                           (0x00000029)
#define LW507E_SURFACE_SET_PARAMS_FORMAT_A8R8G8B8                               (0x000000CF)
#define LW507E_SURFACE_SET_PARAMS_FORMAT_A1R5G5B5                               (0x000000E9)
#define LW507E_SURFACE_SET_PARAMS_COLOR_SPACE                                   1:0
#define LW507E_SURFACE_SET_PARAMS_COLOR_SPACE_RGB                               (0x00000000)
#define LW507E_SURFACE_SET_PARAMS_COLOR_SPACE_YUV_601                           (0x00000001)
#define LW507E_SURFACE_SET_PARAMS_COLOR_SPACE_YUV_709                           (0x00000002)
#define LW507E_SURFACE_SET_PARAMS_KIND                                          22:16
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_PITCH                               (0x00000000)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_GENERIC_8BX2                        (0x00000070)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_GENERIC_8BX2_BANKSWIZ               (0x00000072)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_GENERIC_16BX1                       (0x00000074)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_GENERIC_16BX1_BANKSWIZ              (0x00000076)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_C32_MS4                             (0x00000078)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_C32_MS8                             (0x00000079)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_C32_MS4_BANKSWIZ                    (0x0000007A)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_C32_MS8_BANKSWIZ                    (0x0000007B)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_C64_MS4                             (0x0000007C)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_C64_MS8                             (0x0000007D)
#define LW507E_SURFACE_SET_PARAMS_KIND_KIND_C128_MS4                            (0x0000007E)
#define LW507E_SURFACE_SET_PARAMS_KIND_FROM_PTE                                 (0x0000007F)
#define LW507E_SURFACE_SET_PARAMS_PART_STRIDE                                   24:24
#define LW507E_SURFACE_SET_PARAMS_PART_STRIDE_PARTSTRIDE_256                    (0x00000000)
#define LW507E_SURFACE_SET_PARAMS_PART_STRIDE_PARTSTRIDE_1024                   (0x00000001)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl507e_h

