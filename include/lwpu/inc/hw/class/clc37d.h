/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2015 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/



#ifndef _clC37d_h_
#define _clC37d_h_


#ifdef __cplusplus
extern "C" {
#endif

#define LWC37D_CORE_CHANNEL_DMA                                                 (0x0000C37D)

typedef volatile struct _clc37d_tag0 {
    LwV32 Put;                                                                  // 0x00000000 - 0x00000003
    LwV32 Get;                                                                  // 0x00000004 - 0x00000007
    LwV32 Reserved00[0x7E];
    LwV32 Update;                                                               // 0x00000200 - 0x00000203
    LwV32 Reserved01[0x1];
    LwV32 SetContextDmaNotifier;                                                // 0x00000208 - 0x0000020B
    LwV32 SetNotifierControl;                                                   // 0x0000020C - 0x0000020F
    LwV32 SetControl;                                                           // 0x00000210 - 0x00000213
    LwV32 SetSyncPointControl;                                                  // 0x00000214 - 0x00000217
    LwV32 SetInterlockFlags;                                                    // 0x00000218 - 0x0000021B
    LwV32 SetWindowInterlockFlags;                                              // 0x0000021C - 0x0000021F
    LwV32 GetRgScanLine[8];                                                     // 0x00000220 - 0x0000023F
    LwV32 SetGetBlankingCtrl[8];                                                // 0x00000240 - 0x0000025F
    LwV32 Reserved02[0x8];
    struct {
        LwV32 SetControl;                                                       // 0x00000280 - 0x00000283
        LwV32 SetLwstomReason;                                                  // 0x00000284 - 0x00000287
        LwV32 SetSwSpareA;                                                      // 0x00000288 - 0x0000028B
        LwV32 SetSwSpareB;                                                      // 0x0000028C - 0x0000028F
        LwV32 Reserved03[0x4];
    } Pior[4];
    struct {
        LwV32 SetControl;                                                       // 0x00000300 - 0x00000303
        LwV32 SetLwstomReason;                                                  // 0x00000304 - 0x00000307
        LwV32 SetSwSpareA;                                                      // 0x00000308 - 0x0000030B
        LwV32 SetSwSpareB;                                                      // 0x0000030C - 0x0000030F
        LwV32 Reserved04[0x4];
    } Sor[8];
    struct {
        LwV32 SetControl;                                                       // 0x00000400 - 0x00000403
        LwV32 SetLwstomReason;                                                  // 0x00000404 - 0x00000407
        LwV32 SetSwSpareA;                                                      // 0x00000408 - 0x0000040B
        LwV32 SetSwSpareB;                                                      // 0x0000040C - 0x0000040F
        LwV32 Reserved05[0x4];
    } Wbor[8];
    LwV32 Reserved06[0x2C0];
    struct {
        LwV32 SetControl;                                                       // 0x00001000 - 0x00001003
        LwV32 SetWindowFormatUsageBounds;                                       // 0x00001004 - 0x00001007
        LwV32 SetWindowRotatedFormatUsageBounds;                                // 0x00001008 - 0x0000100B
        LwV32 SetMaxInputScaleFactor;                                           // 0x0000100C - 0x0000100F
        LwV32 SetWindowUsageBounds;                                             // 0x00001010 - 0x00001013
        LwV32 Reserved07[0x1B];
    } Window[32];
    struct {
        LwV32 SetProcamp;                                                       // 0x00002000 - 0x00002003
        LwV32 SetControlOutputResource;                                         // 0x00002004 - 0x00002007
        LwV32 SetControl;                                                       // 0x00002008 - 0x0000200B
        LwV32 SetPixelClockFrequency;                                           // 0x0000200C - 0x0000200F
        LwV32 SetPixelReorderControl;                                           // 0x00002010 - 0x00002013
        LwV32 SetControlOutputScaler;                                           // 0x00002014 - 0x00002017
        LwV32 SetDitherControl;                                                 // 0x00002018 - 0x0000201B
        LwV32 SetPixelClockConfiguration;                                       // 0x0000201C - 0x0000201F
        LwV32 SetDisplayId[2];                                                  // 0x00002020 - 0x00002027
        LwV32 SetPixelClockFrequencyMax;                                        // 0x00002028 - 0x0000202B
        LwV32 SetMaxOutputScaleFactor;                                          // 0x0000202C - 0x0000202F
        LwV32 SetHeadUsageBounds;                                               // 0x00002030 - 0x00002033
        LwV32 SetStallLock;                                                     // 0x00002034 - 0x00002037
        LwV32 SetHdmiAudioControl;                                              // 0x00002038 - 0x0000203B
        LwV32 SetDpAudioControl;                                                // 0x0000203C - 0x0000203F
        LwV32 SetLockOffset;                                                    // 0x00002040 - 0x00002043
        LwV32 SetLockChain;                                                     // 0x00002044 - 0x00002047
        LwV32 SetViewportPointIn;                                               // 0x00002048 - 0x0000204B
        LwV32 SetViewportSizeIn;                                                // 0x0000204C - 0x0000204F
        LwV32 SetViewportValidSizeIn;                                           // 0x00002050 - 0x00002053
        LwV32 SetViewportValidPointIn;                                          // 0x00002054 - 0x00002057
        LwV32 SetViewportSizeOut;                                               // 0x00002058 - 0x0000205B
        LwV32 SetViewportPointOutAdjust;                                        // 0x0000205C - 0x0000205F
        LwV32 SetDesktopColor;                                                  // 0x00002060 - 0x00002063
        LwV32 SetRasterSize;                                                    // 0x00002064 - 0x00002067
        LwV32 SetRasterSyncEnd;                                                 // 0x00002068 - 0x0000206B
        LwV32 SetRasterBlankEnd;                                                // 0x0000206C - 0x0000206F
        LwV32 SetRasterBlankStart;                                              // 0x00002070 - 0x00002073
        LwV32 SetRasterVertBlank2;                                              // 0x00002074 - 0x00002077
        LwV32 SetOverscanColor;                                                 // 0x00002078 - 0x0000207B
        LwV32 SetFramePackedVactiveColor;                                       // 0x0000207C - 0x0000207F
        LwV32 SetHdmiCtrl;                                                      // 0x00002080 - 0x00002083
        LwV32 Reserved08[0x1];
        LwV32 SetContextDmaLwrsor[2];                                           // 0x00002088 - 0x0000208F
        LwV32 SetOffsetLwrsor[2];                                               // 0x00002090 - 0x00002097
        LwV32 SetPresentControlLwrsor;                                          // 0x00002098 - 0x0000209B
        LwV32 SetControlLwrsor;                                                 // 0x0000209C - 0x0000209F
        LwV32 SetControlLwrsorComposition;                                      // 0x000020A0 - 0x000020A3
        LwV32 SetControlOutputLut;                                              // 0x000020A4 - 0x000020A7
        LwV32 SetOffsetOutputLut;                                               // 0x000020A8 - 0x000020AB
        LwV32 SetContextDmaOutputLut;                                           // 0x000020AC - 0x000020AF
        LwV32 SetRegionCrcControl;                                              // 0x000020B0 - 0x000020B3
        LwV32 Reserved09[0x3];
        LwV32 SetRegionCrcPointIn[9];                                           // 0x000020C0 - 0x000020E3
        LwV32 Reserved10[0x7];
        LwV32 SetRegionCrcSize[9];                                              // 0x00002100 - 0x00002123
        LwV32 Reserved11[0x7];
        LwV32 SetRegionGoldenCrc[9];                                            // 0x00002140 - 0x00002163
        LwV32 Reserved12[0x7];
        LwV32 SetContextDmaCrc;                                                 // 0x00002180 - 0x00002183
        LwV32 SetCrcControl;                                                    // 0x00002184 - 0x00002187
        LwV32 SetCrlwserData;                                                   // 0x00002188 - 0x0000218B
        LwV32 SetPresentControl;                                                // 0x0000218C - 0x0000218F
        LwV32 SetVgaCrcControl;                                                 // 0x00002190 - 0x00002193
        LwV32 SetSwSpareA;                                                      // 0x00002194 - 0x00002197
        LwV32 SetSwSpareB;                                                      // 0x00002198 - 0x0000219B
        LwV32 SetSwSpareC;                                                      // 0x0000219C - 0x0000219F
        LwV32 SetSwSpareD;                                                      // 0x000021A0 - 0x000021A3
        LwV32 SyncPointRelease;                                                 // 0x000021A4 - 0x000021A7
        LwV32 SetDisplayRate;                                                   // 0x000021A8 - 0x000021AB
        LwV32 SetDscTopCtl;                                                     // 0x000021AC - 0x000021AF
        LwV32 SetDscDelay;                                                      // 0x000021B0 - 0x000021B3
        LwV32 SetDscCommonCtl;                                                  // 0x000021B4 - 0x000021B7
        LwV32 SetDscSliceInfo;                                                  // 0x000021B8 - 0x000021BB
        LwV32 SetDscRcDelayInfo;                                                // 0x000021BC - 0x000021BF
        LwV32 SetDscRcScaleInfo;                                                // 0x000021C0 - 0x000021C3
        LwV32 SetDscRcScaleInfo2;                                               // 0x000021C4 - 0x000021C7
        LwV32 SetDscRcBpgoffInfo;                                               // 0x000021C8 - 0x000021CB
        LwV32 SetDscRcOffsetInfo;                                               // 0x000021CC - 0x000021CF
        LwV32 SetDscRcFlatnessInfo;                                             // 0x000021D0 - 0x000021D3
        LwV32 SetDscRcParamSet;                                                 // 0x000021D4 - 0x000021D7
        LwV32 SetDscRcBufThresh0;                                               // 0x000021D8 - 0x000021DB
        LwV32 SetDscRcBufThresh1;                                               // 0x000021DC - 0x000021DF
        LwV32 SetDscRcBufThresh2;                                               // 0x000021E0 - 0x000021E3
        LwV32 SetDscRcBufThresh3;                                               // 0x000021E4 - 0x000021E7
        LwV32 SetDscRcRangeCfg0;                                                // 0x000021E8 - 0x000021EB
        LwV32 SetDscRcRangeCfg1;                                                // 0x000021EC - 0x000021EF
        LwV32 SetDscRcRangeCfg2;                                                // 0x000021F0 - 0x000021F3
        LwV32 SetDscRcRangeCfg3;                                                // 0x000021F4 - 0x000021F7
        LwV32 SetDscRcRangeCfg4;                                                // 0x000021F8 - 0x000021FB
        LwV32 SetDscRcRangeCfg5;                                                // 0x000021FC - 0x000021FF
        LwV32 SetDscRcRangeCfg6;                                                // 0x00002200 - 0x00002203
        LwV32 SetDscRcRangeCfg7;                                                // 0x00002204 - 0x00002207
        LwV32 SetDslwnitSet;                                                    // 0x00002208 - 0x0000220B
        LwV32 SetRsb;                                                           // 0x0000220C - 0x0000220F
        LwV32 SetStreamId;                                                      // 0x00002210 - 0x00002213
        LwV32 SetOutputScalerCoeffValue;                                        // 0x00002214 - 0x00002217
        LwV32 SetMinFrameIdle;                                                  // 0x00002218 - 0x0000221B
        LwV32 Reserved13[0x79];
    } Head[8];
} LWC37DDispControlDma;


#define LW_DISP_NOTIFIER                                                             0x00000000
#define LW_DISP_NOTIFIER_SIZEOF                                                      0x00000010
#define LW_DISP_NOTIFIER__0                                                          0x00000000
#define LW_DISP_NOTIFIER__0_PRESENT_COUNT                                            7:0
#define LW_DISP_NOTIFIER__0_FIELD                                                    8:8
#define LW_DISP_NOTIFIER__0_FLIP_TYPE                                                9:9
#define LW_DISP_NOTIFIER__0_FLIP_TYPE_NON_TEARING                                    0x00000000
#define LW_DISP_NOTIFIER__0_FLIP_TYPE_IMMEDIATE                                      0x00000001
#define LW_DISP_NOTIFIER__0_R1                                                       15:10
#define LW_DISP_NOTIFIER__0_R2                                                       23:16
#define LW_DISP_NOTIFIER__0_R3                                                       29:24
#define LW_DISP_NOTIFIER__0_STATUS                                                   31:30
#define LW_DISP_NOTIFIER__0_STATUS_NOT_BEGUN                                         0x00000000
#define LW_DISP_NOTIFIER__0_STATUS_BEGUN                                             0x00000001
#define LW_DISP_NOTIFIER__0_STATUS_FINISHED                                          0x00000002
#define LW_DISP_NOTIFIER__1                                                          0x00000001
#define LW_DISP_NOTIFIER__1_R4                                                       31:0
#define LW_DISP_NOTIFIER__2                                                          0x00000002
#define LW_DISP_NOTIFIER__2_TIMESTAMP_LO                                             31:0
#define LW_DISP_NOTIFIER__3                                                          0x00000003
#define LW_DISP_NOTIFIER__3_TIMESTAMP_HI                                             31:0


// dma opcode instructions
#define LWC37D_DMA                                                                     
#define LWC37D_DMA_OPCODE                                                        31:29 
#define LWC37D_DMA_OPCODE_METHOD                                            0x00000000 
#define LWC37D_DMA_OPCODE_JUMP                                              0x00000001 
#define LWC37D_DMA_OPCODE_NONINC_METHOD                                     0x00000002 
#define LWC37D_DMA_OPCODE_SET_SUBDEVICE_MASK                                0x00000003 
#define LWC37D_DMA_METHOD_COUNT                                                  27:18 
#define LWC37D_DMA_METHOD_OFFSET                                                  13:2 
#define LWC37D_DMA_DATA                                                           31:0 
#define LWC37D_DMA_DATA_NOP                                                 0x00000000 
#define LWC37D_DMA_JUMP_OFFSET                                                    11:2 
#define LWC37D_DMA_SET_SUBDEVICE_MASK_VALUE                                       11:0 

// if cap SUPPORT_FLEXIBLE_WIN_MAPPING is FALSE, this define can be used to obtain which head a window is mapped to
#define LWC37D_WINDOW_MAPPED_TO_HEAD(w) ((w)>>1)
#define LWC37D_GET_VALID_WINDOWMASK_FOR_HEAD(h) ((1<<((h)*2)) | (1<<((h)*2+1)))

// class methods
#define LWC37D_PUT                                                              (0x00000000)
#define LWC37D_PUT_PTR                                                          9:0
#define LWC37D_GET                                                              (0x00000004)
#define LWC37D_GET_PTR                                                          9:0
#define LWC37D_UPDATE                                                           (0x00000200)
#define LWC37D_UPDATE_SPECIAL_HANDLING                                          21:20
#define LWC37D_UPDATE_SPECIAL_HANDLING_NONE                                     (0x00000000)
#define LWC37D_UPDATE_SPECIAL_HANDLING_INTERRUPT_RM                             (0x00000001)
#define LWC37D_UPDATE_SPECIAL_HANDLING_MODE_SWITCH                              (0x00000002)
#define LWC37D_UPDATE_SPECIAL_HANDLING_REASON                                   19:12
#define LWC37D_UPDATE_INHIBIT_INTERRUPTS                                        24:24
#define LWC37D_UPDATE_INHIBIT_INTERRUPTS_FALSE                                  (0x00000000)
#define LWC37D_UPDATE_INHIBIT_INTERRUPTS_TRUE                                   (0x00000001)
#define LWC37D_UPDATE_RELEASE_ELV                                               0:0
#define LWC37D_UPDATE_RELEASE_ELV_FALSE                                         (0x00000000)
#define LWC37D_UPDATE_RELEASE_ELV_TRUE                                          (0x00000001)
#define LWC37D_UPDATE_FLIP_LOCK_PIN                                             8:4
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_NONE                               (0x00000000)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN(i)                                 (0x00000001 +(i))
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN__SIZE_1                            16
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_0                                  (0x00000001)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_1                                  (0x00000002)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_2                                  (0x00000003)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_3                                  (0x00000004)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_4                                  (0x00000005)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_5                                  (0x00000006)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_6                                  (0x00000007)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_7                                  (0x00000008)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_8                                  (0x00000009)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_9                                  (0x0000000A)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_A                                  (0x0000000B)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_B                                  (0x0000000C)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_C                                  (0x0000000D)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_D                                  (0x0000000E)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_E                                  (0x0000000F)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_F                                  (0x00000010)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_0                        (0x00000014)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_1                        (0x00000015)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_2                        (0x00000016)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_3                        (0x00000017)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK(i)                       (0x00000018 +(i))
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1                  8
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_0                        (0x00000018)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_1                        (0x00000019)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_2                        (0x0000001A)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_3                        (0x0000001B)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_4                        (0x0000001C)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_5                        (0x0000001D)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_6                        (0x0000001E)
#define LWC37D_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_7                        (0x0000001F)
#define LWC37D_SET_CONTEXT_DMA_NOTIFIER                                         (0x00000208)
#define LWC37D_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LWC37D_SET_NOTIFIER_CONTROL                                             (0x0000020C)
#define LWC37D_SET_NOTIFIER_CONTROL_MODE                                        0:0
#define LWC37D_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LWC37D_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LWC37D_SET_NOTIFIER_CONTROL_OFFSET                                      11:4
#define LWC37D_SET_NOTIFIER_CONTROL_NOTIFY                                      12:12
#define LWC37D_SET_NOTIFIER_CONTROL_NOTIFY_DISABLE                              (0x00000000)
#define LWC37D_SET_NOTIFIER_CONTROL_NOTIFY_ENABLE                               (0x00000001)
#define LWC37D_SET_CONTROL                                                      (0x00000210)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN(i)                                     ((i)+0):((i)+0)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN__SIZE_1                                4
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN_DISABLE                                (0x00000000)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN_ENABLE                                 (0x00000001)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN0                                       0:0
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN0_DISABLE                               (0x00000000)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN0_ENABLE                                (0x00000001)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN1                                       1:1
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN1_DISABLE                               (0x00000000)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN1_ENABLE                                (0x00000001)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN2                                       2:2
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN2_DISABLE                               (0x00000000)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN2_ENABLE                                (0x00000001)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN3                                       3:3
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN3_DISABLE                               (0x00000000)
#define LWC37D_SET_CONTROL_FLIP_LOCK_PIN3_ENABLE                                (0x00000001)
#define LWC37D_SET_SYNC_POINT_CONTROL                                           (0x00000214)
#define LWC37D_SET_SYNC_POINT_CONTROL_ENABLE                                    0:0
#define LWC37D_SET_SYNC_POINT_CONTROL_ENABLE_DISABLE                            (0x00000000)
#define LWC37D_SET_SYNC_POINT_CONTROL_ENABLE_ENABLE                             (0x00000001)
#define LWC37D_SET_SYNC_POINT_CONTROL_INDEX                                     8:1
#define LWC37D_SET_INTERLOCK_FLAGS                                              (0x00000218)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR(i)                     ((i)+0):((i)+0)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR__SIZE_1                8
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR_DISABLE                (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR_ENABLE                 (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR0                       0:0
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR0_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR0_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR1                       1:1
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR1_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR1_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR2                       2:2
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR2_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR2_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR3                       3:3
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR3_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR3_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR4                       4:4
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR4_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR4_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR5                       5:5
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR5_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR5_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR6                       6:6
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR6_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR6_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR7                       7:7
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR7_DISABLE               (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR7_ENABLE                (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK(i)                  ((i)+8):((i)+8)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK__SIZE_1             8
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK_DISABLE             (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK_ENABLE              (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK0                    8:8
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK0_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK0_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK1                    9:9
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK1_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK1_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK2                    10:10
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK2_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK2_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK3                    11:11
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK3_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK3_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK4                    12:12
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK4_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK4_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK5                    13:13
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK5_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK5_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK6                    14:14
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK6_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK6_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK7                    15:15
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK7_DISABLE            (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK7_ENABLE             (0x00000001)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_CORE                          16:16
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_CORE_DISABLE                  (0x00000000)
#define LWC37D_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_CORE_ENABLE                   (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS                                       (0x0000021C)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW(i)              ((i)+0):((i)+0)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW__SIZE_1         32
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW_DISABLE         (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW_ENABLE          (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW0                0:0
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW0_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW0_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW1                1:1
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW1_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW1_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW2                2:2
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW2_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW2_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW3                3:3
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW3_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW3_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW4                4:4
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW4_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW4_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW5                5:5
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW5_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW5_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW6                6:6
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW6_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW6_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW7                7:7
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW7_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW7_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW8                8:8
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW8_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW8_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW9                9:9
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW9_DISABLE        (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW9_ENABLE         (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW10               10:10
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW10_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW10_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW11               11:11
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW11_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW11_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW12               12:12
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW12_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW12_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW13               13:13
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW13_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW13_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW14               14:14
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW14_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW14_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW15               15:15
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW15_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW15_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW16               16:16
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW16_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW16_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW17               17:17
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW17_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW17_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW18               18:18
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW18_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW18_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW19               19:19
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW19_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW19_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW20               20:20
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW20_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW20_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW21               21:21
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW21_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW21_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW22               22:22
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW22_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW22_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW23               23:23
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW23_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW23_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW24               24:24
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW24_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW24_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW25               25:25
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW25_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW25_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW26               26:26
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW26_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW26_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW27               27:27
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW27_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW27_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW28               28:28
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW28_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW28_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW29               29:29
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW29_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW29_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW30               30:30
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW30_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW30_ENABLE        (0x00000001)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW31               31:31
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW31_DISABLE       (0x00000000)
#define LWC37D_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW31_ENABLE        (0x00000001)
#define LWC37D_GET_RG_SCAN_LINE(b)                                              (0x00000220 + (b)*0x00000004)
#define LWC37D_GET_RG_SCAN_LINE_LINE                                            15:0
#define LWC37D_GET_RG_SCAN_LINE_VBLANK                                          16:16
#define LWC37D_GET_RG_SCAN_LINE_VBLANK_FALSE                                    (0x00000000)
#define LWC37D_GET_RG_SCAN_LINE_VBLANK_TRUE                                     (0x00000001)
#define LWC37D_SET_GET_BLANKING_CTRL(b)                                         (0x00000240 + (b)*0x00000004)
#define LWC37D_SET_GET_BLANKING_CTRL_BLANK                                      0:0
#define LWC37D_SET_GET_BLANKING_CTRL_BLANK_NO_CHANGE                            (0x00000000)
#define LWC37D_SET_GET_BLANKING_CTRL_BLANK_ENABLE                               (0x00000001)
#define LWC37D_SET_GET_BLANKING_CTRL_UNBLANK                                    1:1
#define LWC37D_SET_GET_BLANKING_CTRL_UNBLANK_NO_CHANGE                          (0x00000000)
#define LWC37D_SET_GET_BLANKING_CTRL_UNBLANK_ENABLE                             (0x00000001)

#define LWC37D_PIOR_SET_CONTROL(a)                                              (0x00000280 + (a)*0x00000020)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK                                      7:0
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_NONE                                 (0x00000000)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD0                                (0x00000001)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD1                                (0x00000002)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD2                                (0x00000004)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD3                                (0x00000008)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD4                                (0x00000010)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD5                                (0x00000020)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD6                                (0x00000040)
#define LWC37D_PIOR_SET_CONTROL_OWNER_MASK_HEAD7                                (0x00000080)
#define LWC37D_PIOR_SET_CONTROL_PROTOCOL                                        11:8
#define LWC37D_PIOR_SET_CONTROL_PROTOCOL_EXT_TMDS_ENC                           (0x00000000)
#define LWC37D_PIOR_SET_CONTROL_PROTOCOL_EXT_SDI_SD_ENC                         (0x00000001)
#define LWC37D_PIOR_SET_CONTROL_PROTOCOL_EXT_SDI_HD_ENC                         (0x00000002)
#define LWC37D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_OUT                        (0x00000004)
#define LWC37D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_IN                         (0x00000005)
#define LWC37D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_INOUT                      (0x00000006)
#define LWC37D_PIOR_SET_CONTROL_DE_SYNC_POLARITY                                16:16
#define LWC37D_PIOR_SET_CONTROL_DE_SYNC_POLARITY_POSITIVE_TRUE                  (0x00000000)
#define LWC37D_PIOR_SET_CONTROL_DE_SYNC_POLARITY_NEGATIVE_TRUE                  (0x00000001)
#define LWC37D_PIOR_SET_LWSTOM_REASON(a)                                        (0x00000284 + (a)*0x00000020)
#define LWC37D_PIOR_SET_LWSTOM_REASON_CODE                                      31:0
#define LWC37D_PIOR_SET_SW_SPARE_A(a)                                           (0x00000288 + (a)*0x00000020)
#define LWC37D_PIOR_SET_SW_SPARE_A_CODE                                         31:0
#define LWC37D_PIOR_SET_SW_SPARE_B(a)                                           (0x0000028C + (a)*0x00000020)
#define LWC37D_PIOR_SET_SW_SPARE_B_CODE                                         31:0

#define LWC37D_SOR_SET_CONTROL(a)                                               (0x00000300 + (a)*0x00000020)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK                                       7:0
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_NONE                                  (0x00000000)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD0                                 (0x00000001)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD1                                 (0x00000002)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD2                                 (0x00000004)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD3                                 (0x00000008)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD4                                 (0x00000010)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD5                                 (0x00000020)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD6                                 (0x00000040)
#define LWC37D_SOR_SET_CONTROL_OWNER_MASK_HEAD7                                 (0x00000080)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL                                         11:8
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_LVDS_LWSTOM                             (0x00000000)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_A                           (0x00000001)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_B                           (0x00000002)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_DUAL_TMDS                               (0x00000005)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_DP_A                                    (0x00000008)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_DP_B                                    (0x00000009)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_DSI                                     (0x0000000A)
#define LWC37D_SOR_SET_CONTROL_PROTOCOL_LWSTOM                                  (0x0000000F)
#define LWC37D_SOR_SET_CONTROL_DE_SYNC_POLARITY                                 16:16
#define LWC37D_SOR_SET_CONTROL_DE_SYNC_POLARITY_POSITIVE_TRUE                   (0x00000000)
#define LWC37D_SOR_SET_CONTROL_DE_SYNC_POLARITY_NEGATIVE_TRUE                   (0x00000001)
#define LWC37D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE                             21:20
#define LWC37D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE_OFF                         (0x00000000)
#define LWC37D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE_X2                          (0x00000001)
#define LWC37D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE_X4                          (0x00000002)
#define LWC37D_SOR_SET_LWSTOM_REASON(a)                                         (0x00000304 + (a)*0x00000020)
#define LWC37D_SOR_SET_LWSTOM_REASON_CODE                                       31:0
#define LWC37D_SOR_SET_SW_SPARE_A(a)                                            (0x00000308 + (a)*0x00000020)
#define LWC37D_SOR_SET_SW_SPARE_A_CODE                                          31:0
#define LWC37D_SOR_SET_SW_SPARE_B(a)                                            (0x0000030C + (a)*0x00000020)
#define LWC37D_SOR_SET_SW_SPARE_B_CODE                                          31:0

#define LWC37D_WBOR_SET_CONTROL(a)                                              (0x00000400 + (a)*0x00000020)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK                                      7:0
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_NONE                                 (0x00000000)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD0                                (0x00000001)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD1                                (0x00000002)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD2                                (0x00000004)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD3                                (0x00000008)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD4                                (0x00000010)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD5                                (0x00000020)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD6                                (0x00000040)
#define LWC37D_WBOR_SET_CONTROL_OWNER_MASK_HEAD7                                (0x00000080)
#define LWC37D_WBOR_SET_CONTROL_WRITEBACK                                       8:8
#define LWC37D_WBOR_SET_CONTROL_WRITEBACK_DISABLE                               (0x00000000)
#define LWC37D_WBOR_SET_CONTROL_WRITEBACK_ENABLE                                (0x00000001)
#define LWC37D_WBOR_SET_LWSTOM_REASON(a)                                        (0x00000404 + (a)*0x00000020)
#define LWC37D_WBOR_SET_LWSTOM_REASON_CODE                                      31:0
#define LWC37D_WBOR_SET_SW_SPARE_A(a)                                           (0x00000408 + (a)*0x00000020)
#define LWC37D_WBOR_SET_SW_SPARE_A_CODE                                         31:0
#define LWC37D_WBOR_SET_SW_SPARE_B(a)                                           (0x0000040C + (a)*0x00000020)
#define LWC37D_WBOR_SET_SW_SPARE_B_CODE                                         31:0

#define LWC37D_WINDOW_SET_CONTROL(a)                                            (0x00001000 + (a)*0x00000080)
#define LWC37D_WINDOW_SET_CONTROL_OWNER                                         3:0
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD(i)                                 (0x00000000 +(i))
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD__SIZE_1                            8
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD0                                   (0x00000000)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD1                                   (0x00000001)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD2                                   (0x00000002)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD3                                   (0x00000003)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD4                                   (0x00000004)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD5                                   (0x00000005)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD6                                   (0x00000006)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_HEAD7                                   (0x00000007)
#define LWC37D_WINDOW_SET_CONTROL_OWNER_NONE                                    (0x0000000F)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS(a)                         (0x00001004 + (a)*0x00000080)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED1BPP             0:0
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED1BPP_FALSE       (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED1BPP_TRUE        (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED2BPP             1:1
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED2BPP_FALSE       (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED2BPP_TRUE        (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED4BPP             2:2
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED4BPP_FALSE       (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED4BPP_TRUE        (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED8BPP             3:3
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED8BPP_FALSE       (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED8BPP_TRUE        (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PACKED422              4:4
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PACKED422_FALSE        (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PACKED422_TRUE         (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PLANAR420              5:5
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PLANAR420_FALSE        (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PLANAR420_TRUE         (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PLANAR444              6:6
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PLANAR444_FALSE        (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_PLANAR444_TRUE         (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR420         7:7
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR420_FALSE   (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR420_TRUE    (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422         8:8
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422_FALSE   (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422_TRUE    (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422R        9:9
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422R_FALSE  (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422R_TRUE   (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR444         10:10
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR444_FALSE   (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR444_TRUE    (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR420          11:11
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR420_FALSE    (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR420_TRUE     (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR444          12:12
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR444_FALSE    (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR444_TRUE     (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR420     13:13
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR420_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR420_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422     14:14
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422R    15:15
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422R_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422R_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR444     16:16
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR444_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR444_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS(a)                 (0x00001008 + (a)*0x00000080)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED1BPP     0:0
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED1BPP_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED1BPP_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED2BPP     1:1
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED2BPP_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED2BPP_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED4BPP     2:2
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED4BPP_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED4BPP_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED8BPP     3:3
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED8BPP_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_RGB_PACKED8BPP_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PACKED422      4:4
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PACKED422_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PACKED422_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PLANAR420      5:5
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PLANAR420_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PLANAR420_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PLANAR444      6:6
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PLANAR444_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_PLANAR444_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR420 7:7
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR420_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR420_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422 8:8
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422R 9:9
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422R_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR422R_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR444 10:10
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR444_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_YUV_SEMI_PLANAR444_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR420  11:11
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR420_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR420_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR444  12:12
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR444_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_PLANAR444_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR420 13:13
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR420_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR420_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422 14:14
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422R 15:15
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422R_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR422R_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR444 16:16
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR444_FALSE (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS_EXT_YUV_SEMI_PLANAR444_TRUE (0x00000001)
#define LWC37D_WINDOW_SET_MAX_INPUT_SCALE_FACTOR(a)                             (0x0000100C + (a)*0x00000080)
#define LWC37D_WINDOW_SET_MAX_INPUT_SCALE_FACTOR_HORIZONTAL                     15:0
#define LWC37D_WINDOW_SET_MAX_INPUT_SCALE_FACTOR_VERTICAL                       31:16
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS(a)                                (0x00001010 + (a)*0x00000080)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_MAX_PIXELS_FETCHED_PER_LINE       14:0
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_DIST_RENDER_USABLE                18:18
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_DIST_RENDER_USABLE_FALSE          (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_DIST_RENDER_USABLE_TRUE           (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_LUT                         17:16
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_LUT_USAGE_NONE              (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_LUT_USAGE_257               (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_LUT_USAGE_1025              (0x00000002)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_SCALER_TAPS                 22:20
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_SCALER_TAPS_TAPS_2          (0x00000001)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_SCALER_TAPS_TAPS_5          (0x00000004)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_UPSCALING_ALLOWED                 24:24
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_UPSCALING_ALLOWED_FALSE           (0x00000000)
#define LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS_UPSCALING_ALLOWED_TRUE            (0x00000001)

#define LWC37D_HEAD_SET_PROCAMP(a)                                              (0x00002000 + (a)*0x00000400)
#define LWC37D_HEAD_SET_PROCAMP_COLOR_SPACE                                     1:0
#define LWC37D_HEAD_SET_PROCAMP_COLOR_SPACE_RGB                                 (0x00000000)
#define LWC37D_HEAD_SET_PROCAMP_COLOR_SPACE_YUV_601                             (0x00000001)
#define LWC37D_HEAD_SET_PROCAMP_COLOR_SPACE_YUV_709                             (0x00000002)
#define LWC37D_HEAD_SET_PROCAMP_COLOR_SPACE_YUV_2020                            (0x00000003)
#define LWC37D_HEAD_SET_PROCAMP_CHROMA_LPF                                      3:3
#define LWC37D_HEAD_SET_PROCAMP_CHROMA_LPF_DISABLE                              (0x00000000)
#define LWC37D_HEAD_SET_PROCAMP_CHROMA_LPF_ENABLE                               (0x00000001)
#define LWC37D_HEAD_SET_PROCAMP_SAT_COS                                         15:4
#define LWC37D_HEAD_SET_PROCAMP_SAT_SINE                                        27:16
#define LWC37D_HEAD_SET_PROCAMP_DYNAMIC_RANGE                                   28:28
#define LWC37D_HEAD_SET_PROCAMP_DYNAMIC_RANGE_VESA                              (0x00000000)
#define LWC37D_HEAD_SET_PROCAMP_DYNAMIC_RANGE_CEA                               (0x00000001)
#define LWC37D_HEAD_SET_PROCAMP_RANGE_COMPRESSION                               29:29
#define LWC37D_HEAD_SET_PROCAMP_RANGE_COMPRESSION_DISABLE                       (0x00000000)
#define LWC37D_HEAD_SET_PROCAMP_RANGE_COMPRESSION_ENABLE                        (0x00000001)
#define LWC37D_HEAD_SET_PROCAMP_BLACK_LEVEL                                     31:30
#define LWC37D_HEAD_SET_PROCAMP_BLACK_LEVEL_AUTO                                (0x00000000)
#define LWC37D_HEAD_SET_PROCAMP_BLACK_LEVEL_VIDEO                               (0x00000001)
#define LWC37D_HEAD_SET_PROCAMP_BLACK_LEVEL_GRAPHICS                            (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE(a)                              (0x00002004 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE                        1:0
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE_ACTIVE_RASTER          (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE_COMPLETE_RASTER        (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE_NON_ACTIVE_RASTER      (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_HSYNC_POLARITY                  2:2
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_HSYNC_POLARITY_POSITIVE_TRUE    (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_HSYNC_POLARITY_NEGATIVE_TRUE    (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_VSYNC_POLARITY                  3:3
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_VSYNC_POLARITY_POSITIVE_TRUE    (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_VSYNC_POLARITY_NEGATIVE_TRUE    (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH                     7:4
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_16_422          (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_18_444          (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_20_422          (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_422          (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_444          (0x00000004)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_30_444          (0x00000005)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_32_422          (0x00000006)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_36_444          (0x00000007)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_48_444          (0x00000008)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_OVERRIDE            24:24
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_OVERRIDE_DISABLE    (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_OVERRIDE_ENABLE     (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_FLAG                23:12
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_MSA_STEREO_OVERRIDE             25:25
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_MSA_STEREO_OVERRIDE_DISABLE     (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_MSA_STEREO_OVERRIDE_ENABLE      (0x00000001)
#define LWC37D_HEAD_SET_CONTROL(a)                                              (0x00002008 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTROL_STRUCTURE                                       1:0
#define LWC37D_HEAD_SET_CONTROL_STRUCTURE_PROGRESSIVE                           (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_STRUCTURE_INTERLACED                            (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_STRUCTURE_INTERLACED_INOUT                      (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_STEREO3D_STRUCTURE                              2:2
#define LWC37D_HEAD_SET_CONTROL_STEREO3D_STRUCTURE_NORMAL                       (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_STEREO3D_STRUCTURE_FRAME_PACKED                 (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE                                 11:10
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_NO_LOCK                         (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_FRAME_LOCK                      (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_RASTER_LOCK                     (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN                                  8:4
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_NONE                    (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN(i)                      (0x00000001 +(i))
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN__SIZE_1                 16
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_0                       (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_1                       (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_2                       (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_3                       (0x00000004)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_4                       (0x00000005)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_5                       (0x00000006)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_6                       (0x00000007)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_7                       (0x00000008)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_8                       (0x00000009)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_9                       (0x0000000A)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_A                       (0x0000000B)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_B                       (0x0000000C)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_C                       (0x0000000D)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_D                       (0x0000000E)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_E                       (0x0000000F)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_F                       (0x00000010)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK_0             (0x00000014)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK_1             (0x00000015)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK_2             (0x00000016)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK_3             (0x00000017)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK(i)            (0x00000018 +(i))
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1       8
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_0             (0x00000018)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_1             (0x00000019)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_2             (0x0000001A)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_3             (0x0000001B)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_4             (0x0000001C)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_5             (0x0000001D)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_6             (0x0000001E)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_7             (0x0000001F)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_LOCKOUT_WINDOW                            15:12
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_MODE                                23:22
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_MODE_NO_LOCK                        (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_MODE_FRAME_LOCK                     (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_MODE_RASTER_LOCK                    (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN                                 20:16
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_NONE                   (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN(i)                     (0x00000001 +(i))
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN__SIZE_1                16
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_0                      (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_1                      (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_2                      (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_3                      (0x00000004)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_4                      (0x00000005)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_5                      (0x00000006)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_6                      (0x00000007)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_7                      (0x00000008)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_8                      (0x00000009)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_9                      (0x0000000A)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_A                      (0x0000000B)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_B                      (0x0000000C)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_C                      (0x0000000D)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_D                      (0x0000000E)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_E                      (0x0000000F)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_F                      (0x00000010)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK_0            (0x00000014)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK_1            (0x00000015)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK_2            (0x00000016)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK_3            (0x00000017)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK(i)           (0x00000018 +(i))
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1      8
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_0            (0x00000018)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_1            (0x00000019)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_2            (0x0000001A)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_3            (0x0000001B)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_4            (0x0000001C)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_5            (0x0000001D)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_6            (0x0000001E)
#define LWC37D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_7            (0x0000001F)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN                                      28:24
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_NONE                        (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN(i)                          (0x00000001 +(i))
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN__SIZE_1                     16
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_0                           (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_1                           (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_2                           (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_3                           (0x00000004)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_4                           (0x00000005)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_5                           (0x00000006)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_6                           (0x00000007)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_7                           (0x00000008)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_8                           (0x00000009)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_9                           (0x0000000A)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_A                           (0x0000000B)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_B                           (0x0000000C)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_C                           (0x0000000D)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_D                           (0x0000000E)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_E                           (0x0000000F)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_F                           (0x00000010)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK_0                 (0x00000014)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK_1                 (0x00000015)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK_2                 (0x00000016)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK_3                 (0x00000017)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK(i)                (0x00000018 +(i))
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK__SIZE_1           8
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_0                 (0x00000018)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_1                 (0x00000019)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_2                 (0x0000001A)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_3                 (0x0000001B)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_4                 (0x0000001C)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_5                 (0x0000001D)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_6                 (0x0000001E)
#define LWC37D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_7                 (0x0000001F)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_STEREO_LOCK_MODE                          30:30
#define LWC37D_HEAD_SET_CONTROL_SLAVE_STEREO_LOCK_MODE_DISABLE                  (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_SLAVE_STEREO_LOCK_MODE_ENABLE                   (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_MASTER_STEREO_LOCK_MODE                         31:31
#define LWC37D_HEAD_SET_CONTROL_MASTER_STEREO_LOCK_MODE_DISABLE                 (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_MASTER_STEREO_LOCK_MODE_ENABLE                  (0x00000001)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY(a)                                (0x0000200C + (a)*0x00000400)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_HERTZ                             30:0
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_ADJ1000DIV1001                    31:31
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_ADJ1000DIV1001_FALSE              (0x00000000)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_ADJ1000DIV1001_TRUE               (0x00000001)
#define LWC37D_HEAD_SET_PIXEL_REORDER_CONTROL(a)                                (0x00002010 + (a)*0x00000400)
#define LWC37D_HEAD_SET_PIXEL_REORDER_CONTROL_BANK_WIDTH                        13:0
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER(a)                                (0x00002014 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS                     2:0
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS_TAPS_2              (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS_TAPS_5              (0x00000004)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER_HORIZONTAL_TAPS                   6:4
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER_HORIZONTAL_TAPS_TAPS_2            (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER_HORIZONTAL_TAPS_TAPS_5            (0x00000004)
#define LWC37D_HEAD_SET_DITHER_CONTROL(a)                                       (0x00002018 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DITHER_CONTROL_ENABLE                                   0:0
#define LWC37D_HEAD_SET_DITHER_CONTROL_ENABLE_DISABLE                           (0x00000000)
#define LWC37D_HEAD_SET_DITHER_CONTROL_ENABLE_ENABLE                            (0x00000001)
#define LWC37D_HEAD_SET_DITHER_CONTROL_BITS                                     5:4
#define LWC37D_HEAD_SET_DITHER_CONTROL_BITS_TO_6_BITS                           (0x00000000)
#define LWC37D_HEAD_SET_DITHER_CONTROL_BITS_TO_8_BITS                           (0x00000001)
#define LWC37D_HEAD_SET_DITHER_CONTROL_BITS_TO_10_BITS                          (0x00000002)
#define LWC37D_HEAD_SET_DITHER_CONTROL_BITS_TO_12_BITS                          (0x00000003)
#define LWC37D_HEAD_SET_DITHER_CONTROL_OFFSET_ENABLE                            2:2
#define LWC37D_HEAD_SET_DITHER_CONTROL_OFFSET_ENABLE_DISABLE                    (0x00000000)
#define LWC37D_HEAD_SET_DITHER_CONTROL_OFFSET_ENABLE_ENABLE                     (0x00000001)
#define LWC37D_HEAD_SET_DITHER_CONTROL_MODE                                     10:8
#define LWC37D_HEAD_SET_DITHER_CONTROL_MODE_DYNAMIC_ERR_ACC                     (0x00000000)
#define LWC37D_HEAD_SET_DITHER_CONTROL_MODE_STATIC_ERR_ACC                      (0x00000001)
#define LWC37D_HEAD_SET_DITHER_CONTROL_MODE_DYNAMIC_2X2                         (0x00000002)
#define LWC37D_HEAD_SET_DITHER_CONTROL_MODE_STATIC_2X2                          (0x00000003)
#define LWC37D_HEAD_SET_DITHER_CONTROL_MODE_TEMPORAL                            (0x00000004)
#define LWC37D_HEAD_SET_DITHER_CONTROL_PHASE                                    13:12
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION(a)                            (0x0000201C + (a)*0x00000400)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_NOT_DRIVER                    0:0
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_NOT_DRIVER_FALSE              (0x00000000)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_NOT_DRIVER_TRUE               (0x00000001)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING                       4:4
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_DISABLE               (0x00000000)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_ENABLE                (0x00000001)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_MODE                  9:8
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_MODE_VBLANK           (0x00000000)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_MODE_HBLANK           (0x00000001)
#define LWC37D_HEAD_SET_DISPLAY_ID(a,b)                                         (0x00002020 + (a)*0x00000400 + (b)*0x00000004)
#define LWC37D_HEAD_SET_DISPLAY_ID_CODE                                         31:0
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX(a)                            (0x00002028 + (a)*0x00000400)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_HERTZ                         30:0
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_ADJ1000DIV1001                31:31
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_ADJ1000DIV1001_FALSE          (0x00000000)
#define LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_ADJ1000DIV1001_TRUE           (0x00000001)
#define LWC37D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR(a)                              (0x0000202C + (a)*0x00000400)
#define LWC37D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR_HORIZONTAL                      15:0
#define LWC37D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR_VERTICAL                        31:16
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS(a)                                    (0x00002030 + (a)*0x00000400)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_LWRSOR                                2:0
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_LWRSOR_USAGE_NONE                     (0x00000000)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_LWRSOR_USAGE_W32_H32                  (0x00000001)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_LWRSOR_USAGE_W64_H64                  (0x00000002)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_LWRSOR_USAGE_W128_H128                (0x00000003)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_LWRSOR_USAGE_W256_H256                (0x00000004)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_OUTPUT_LUT                            5:4
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_OUTPUT_LUT_USAGE_NONE                 (0x00000000)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_OUTPUT_LUT_USAGE_257                  (0x00000001)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_OUTPUT_LUT_USAGE_1025                 (0x00000002)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_UPSCALING_ALLOWED                     8:8
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_UPSCALING_ALLOWED_FALSE               (0x00000000)
#define LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS_UPSCALING_ALLOWED_TRUE                (0x00000001)
#define LWC37D_HEAD_SET_STALL_LOCK(a)                                           (0x00002034 + (a)*0x00000400)
#define LWC37D_HEAD_SET_STALL_LOCK_ENABLE                                       0:0
#define LWC37D_HEAD_SET_STALL_LOCK_ENABLE_FALSE                                 (0x00000000)
#define LWC37D_HEAD_SET_STALL_LOCK_ENABLE_TRUE                                  (0x00000001)
#define LWC37D_HEAD_SET_STALL_LOCK_MODE                                         2:2
#define LWC37D_HEAD_SET_STALL_LOCK_MODE_CONTINUOUS                              (0x00000000)
#define LWC37D_HEAD_SET_STALL_LOCK_MODE_ONE_SHOT                                (0x00000001)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN                                     8:4
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_NONE                       (0x00000000)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN(i)                         (0x00000001 +(i))
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN__SIZE_1                    16
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_0                          (0x00000001)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_1                          (0x00000002)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_2                          (0x00000003)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_3                          (0x00000004)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_4                          (0x00000005)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_5                          (0x00000006)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_6                          (0x00000007)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_7                          (0x00000008)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_8                          (0x00000009)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_9                          (0x0000000A)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_A                          (0x0000000B)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_B                          (0x0000000C)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_C                          (0x0000000D)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_D                          (0x0000000E)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_E                          (0x0000000F)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_F                          (0x00000010)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK_0                (0x00000014)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK_1                (0x00000015)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK_2                (0x00000016)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK_3                (0x00000017)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK(i)               (0x00000018 +(i))
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1          8
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_0                (0x00000018)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_1                (0x00000019)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_2                (0x0000001A)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_3                (0x0000001B)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_4                (0x0000001C)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_5                (0x0000001D)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_6                (0x0000001E)
#define LWC37D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_7                (0x0000001F)
#define LWC37D_HEAD_SET_STALL_LOCK_UNSTALL_MODE                                 12:12
#define LWC37D_HEAD_SET_STALL_LOCK_UNSTALL_MODE_CRASH_LOCK                      (0x00000000)
#define LWC37D_HEAD_SET_STALL_LOCK_UNSTALL_MODE_LINE_LOCK                       (0x00000001)
#define LWC37D_HEAD_SET_STALL_LOCK_TEPOLARITY                                   14:14
#define LWC37D_HEAD_SET_STALL_LOCK_TEPOLARITY_POSITIVE_TRUE                     (0x00000000)
#define LWC37D_HEAD_SET_STALL_LOCK_TEPOLARITY_NEGATIVE_TRUE                     (0x00000001)
#define LWC37D_HEAD_SET_HDMI_AUDIO_CONTROL(a)                                   (0x00002038 + (a)*0x00000400)
#define LWC37D_HEAD_SET_HDMI_AUDIO_CONTROL_ENABLE                               0:0
#define LWC37D_HEAD_SET_HDMI_AUDIO_CONTROL_ENABLE_DISABLE                       (0x00000000)
#define LWC37D_HEAD_SET_HDMI_AUDIO_CONTROL_ENABLE_ENABLE                        (0x00000001)
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL(a)                                     (0x0000203C + (a)*0x00000400)
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL_ENABLE                                 0:0
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL_ENABLE_DISABLE                         (0x00000000)
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL_ENABLE_ENABLE                          (0x00000001)
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL_MUTE                                   2:1
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL_MUTE_DISABLE                           (0x00000000)
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL_MUTE_ENABLE                            (0x00000001)
#define LWC37D_HEAD_SET_DP_AUDIO_CONTROL_MUTE_AUTO                              (0x00000002)
#define LWC37D_HEAD_SET_LOCK_OFFSET(a)                                          (0x00002040 + (a)*0x00000400)
#define LWC37D_HEAD_SET_LOCK_OFFSET_X                                           14:0
#define LWC37D_HEAD_SET_LOCK_OFFSET_Y                                           30:16
#define LWC37D_HEAD_SET_LOCK_CHAIN(a)                                           (0x00002044 + (a)*0x00000400)
#define LWC37D_HEAD_SET_LOCK_CHAIN_POSITION                                     3:0
#define LWC37D_HEAD_SET_VIEWPORT_POINT_IN(a)                                    (0x00002048 + (a)*0x00000400)
#define LWC37D_HEAD_SET_VIEWPORT_POINT_IN_X                                     14:0
#define LWC37D_HEAD_SET_VIEWPORT_POINT_IN_Y                                     30:16
#define LWC37D_HEAD_SET_VIEWPORT_SIZE_IN(a)                                     (0x0000204C + (a)*0x00000400)
#define LWC37D_HEAD_SET_VIEWPORT_SIZE_IN_WIDTH                                  14:0
#define LWC37D_HEAD_SET_VIEWPORT_SIZE_IN_HEIGHT                                 30:16
#define LWC37D_HEAD_SET_VIEWPORT_VALID_SIZE_IN(a)                               (0x00002050 + (a)*0x00000400)
#define LWC37D_HEAD_SET_VIEWPORT_VALID_SIZE_IN_WIDTH                            14:0
#define LWC37D_HEAD_SET_VIEWPORT_VALID_SIZE_IN_HEIGHT                           30:16
#define LWC37D_HEAD_SET_VIEWPORT_VALID_POINT_IN(a)                              (0x00002054 + (a)*0x00000400)
#define LWC37D_HEAD_SET_VIEWPORT_VALID_POINT_IN_X                               14:0
#define LWC37D_HEAD_SET_VIEWPORT_VALID_POINT_IN_Y                               30:16
#define LWC37D_HEAD_SET_VIEWPORT_SIZE_OUT(a)                                    (0x00002058 + (a)*0x00000400)
#define LWC37D_HEAD_SET_VIEWPORT_SIZE_OUT_WIDTH                                 14:0
#define LWC37D_HEAD_SET_VIEWPORT_SIZE_OUT_HEIGHT                                30:16
#define LWC37D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST(a)                            (0x0000205C + (a)*0x00000400)
#define LWC37D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST_X                             15:0
#define LWC37D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST_Y                             31:16
#define LWC37D_HEAD_SET_DESKTOP_COLOR(a)                                        (0x00002060 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DESKTOP_COLOR_ALPHA                                     7:0
#define LWC37D_HEAD_SET_DESKTOP_COLOR_RED                                       15:8
#define LWC37D_HEAD_SET_DESKTOP_COLOR_GREEN                                     23:16
#define LWC37D_HEAD_SET_DESKTOP_COLOR_BLUE                                      31:24
#define LWC37D_HEAD_SET_RASTER_SIZE(a)                                          (0x00002064 + (a)*0x00000400)
#define LWC37D_HEAD_SET_RASTER_SIZE_WIDTH                                       14:0
#define LWC37D_HEAD_SET_RASTER_SIZE_HEIGHT                                      30:16
#define LWC37D_HEAD_SET_RASTER_SYNC_END(a)                                      (0x00002068 + (a)*0x00000400)
#define LWC37D_HEAD_SET_RASTER_SYNC_END_X                                       14:0
#define LWC37D_HEAD_SET_RASTER_SYNC_END_Y                                       30:16
#define LWC37D_HEAD_SET_RASTER_BLANK_END(a)                                     (0x0000206C + (a)*0x00000400)
#define LWC37D_HEAD_SET_RASTER_BLANK_END_X                                      14:0
#define LWC37D_HEAD_SET_RASTER_BLANK_END_Y                                      30:16
#define LWC37D_HEAD_SET_RASTER_BLANK_START(a)                                   (0x00002070 + (a)*0x00000400)
#define LWC37D_HEAD_SET_RASTER_BLANK_START_X                                    14:0
#define LWC37D_HEAD_SET_RASTER_BLANK_START_Y                                    30:16
#define LWC37D_HEAD_SET_RASTER_VERT_BLANK2(a)                                   (0x00002074 + (a)*0x00000400)
#define LWC37D_HEAD_SET_RASTER_VERT_BLANK2_YSTART                               14:0
#define LWC37D_HEAD_SET_RASTER_VERT_BLANK2_YEND                                 30:16
#define LWC37D_HEAD_SET_OVERSCAN_COLOR(a)                                       (0x00002078 + (a)*0x00000400)
#define LWC37D_HEAD_SET_OVERSCAN_COLOR_RED_CR                                   9:0
#define LWC37D_HEAD_SET_OVERSCAN_COLOR_GREEN_Y                                  19:10
#define LWC37D_HEAD_SET_OVERSCAN_COLOR_BLUE_CB                                  29:20
#define LWC37D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR(a)                           (0x0000207C + (a)*0x00000400)
#define LWC37D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR_RED_CR                       9:0
#define LWC37D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR_GREEN_Y                      19:10
#define LWC37D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR_BLUE_CB                      29:20
#define LWC37D_HEAD_SET_HDMI_CTRL(a)                                            (0x00002080 + (a)*0x00000400)
#define LWC37D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT                                  2:0
#define LWC37D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT_NORMAL                           (0x00000000)
#define LWC37D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT_EXTENDED                         (0x00000001)
#define LWC37D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT_STEREO3D                         (0x00000002)
#define LWC37D_HEAD_SET_HDMI_CTRL_HDMI_VIC                                      11:4
#define LWC37D_HEAD_SET_CONTEXT_DMA_LWRSOR(a,b)                                 (0x00002088 + (a)*0x00000400 + (b)*0x00000004)
#define LWC37D_HEAD_SET_CONTEXT_DMA_LWRSOR_HANDLE                               31:0
#define LWC37D_HEAD_SET_OFFSET_LWRSOR(a,b)                                      (0x00002090 + (a)*0x00000400 + (b)*0x00000004)
#define LWC37D_HEAD_SET_OFFSET_LWRSOR_ORIGIN                                    31:0
#define LWC37D_HEAD_SET_PRESENT_CONTROL_LWRSOR(a)                               (0x00002098 + (a)*0x00000400)
#define LWC37D_HEAD_SET_PRESENT_CONTROL_LWRSOR_MODE                             0:0
#define LWC37D_HEAD_SET_PRESENT_CONTROL_LWRSOR_MODE_MONO                        (0x00000000)
#define LWC37D_HEAD_SET_PRESENT_CONTROL_LWRSOR_MODE_STEREO                      (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR(a)                                       (0x0000209C + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_ENABLE                                   31:31
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_ENABLE_DISABLE                           (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_ENABLE_ENABLE                            (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_FORMAT                                   7:0
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_FORMAT_A1R5G5B5                          (0x000000E9)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_FORMAT_A8R8G8B8                          (0x000000CF)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_SIZE                                     9:8
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_SIZE_W32_H32                             (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_SIZE_W64_H64                             (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_SIZE_W128_H128                           (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_SIZE_W256_H256                           (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_HOT_SPOT_X                               19:12
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_HOT_SPOT_Y                               27:20
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_DE_GAMMA                                 29:28
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_DE_GAMMA_NONE                            (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_DE_GAMMA_SRGB                            (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_DE_GAMMA_YUV8_10                         (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_DE_GAMMA_YUV12                           (0x00000003)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION(a)                           (0x000020A0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_K1                           7:0
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_LWRSOR_COLOR_FACTOR_SELECT   11:8
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_LWRSOR_COLOR_FACTOR_SELECT_K1 (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_LWRSOR_COLOR_FACTOR_SELECT_K1_TIMES_SRC (0x00000005)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_VIEWPORT_COLOR_FACTOR_SELECT 15:12
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_VIEWPORT_COLOR_FACTOR_SELECT_ZERO (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_VIEWPORT_COLOR_FACTOR_SELECT_K1 (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_VIEWPORT_COLOR_FACTOR_SELECT_NEG_K1_TIMES_SRC (0x00000007)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_MODE                         16:16
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_MODE_BLEND                   (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_MODE_XOR                     (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT(a)                                   (0x000020A4 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_SIZE                                 1:0
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_SIZE_SIZE_257                        (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_SIZE_SIZE_1025                       (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_RANGE                                5:4
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_RANGE_UNITY                          (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_RANGE_XRBIAS                         (0x00000001)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_RANGE_XVYCC                          (0x00000002)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_OUTPUT_MODE                          9:8
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_OUTPUT_MODE_INDEX                    (0x00000000)
#define LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT_OUTPUT_MODE_INTERPOLATE              (0x00000001)
#define LWC37D_HEAD_SET_OFFSET_OUTPUT_LUT(a)                                    (0x000020A8 + (a)*0x00000400)
#define LWC37D_HEAD_SET_OFFSET_OUTPUT_LUT_ORIGIN                                31:0
#define LWC37D_HEAD_SET_CONTEXT_DMA_OUTPUT_LUT(a)                               (0x000020AC + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTEXT_DMA_OUTPUT_LUT_HANDLE                           31:0
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL(a)                                   (0x000020B0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION(i)                            ((i)+0):((i)+0)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION__SIZE_1                       9
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION_DISABLE                       (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION_ENABLE                        (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION0                              0:0
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION0_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION0_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION1                              1:1
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION1_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION1_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION2                              2:2
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION2_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION2_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION3                              3:3
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION3_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION3_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION4                              4:4
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION4_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION4_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION5                              5:5
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION5_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION5_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION6                              6:6
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION6_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION6_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION7                              7:7
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION7_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION7_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION8                              8:8
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION8_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION8_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION_CRC_READBACK_LOCATION         31:31
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION_CRC_READBACK_LOCATION_CRC_READBACK_HW (0x00000000)
#define LWC37D_HEAD_SET_REGION_CRC_CONTROL_REGION_CRC_READBACK_LOCATION_CRC_READBACK_GOLDEN (0x00000001)
#define LWC37D_HEAD_SET_REGION_CRC_POINT_IN(a,b)                                (0x000020C0 + (a)*0x00000400 + (b)*0x00000004)
#define LWC37D_HEAD_SET_REGION_CRC_POINT_IN_X                                   14:0
#define LWC37D_HEAD_SET_REGION_CRC_POINT_IN_Y                                   30:16
#define LWC37D_HEAD_SET_REGION_CRC_SIZE(a,b)                                    (0x00002100 + (a)*0x00000400 + (b)*0x00000004)
#define LWC37D_HEAD_SET_REGION_CRC_SIZE_WIDTH                                   14:0
#define LWC37D_HEAD_SET_REGION_CRC_SIZE_HEIGHT                                  30:16
#define LWC37D_HEAD_SET_REGION_GOLDEN_CRC(a,b)                                  (0x00002140 + (a)*0x00000400 + (b)*0x00000004)
#define LWC37D_HEAD_SET_REGION_GOLDEN_CRC_GOLDEN_CRC                            31:0
#define LWC37D_HEAD_SET_CONTEXT_DMA_CRC(a)                                      (0x00002180 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CONTEXT_DMA_CRC_HANDLE                                  31:0
#define LWC37D_HEAD_SET_CRC_CONTROL(a)                                          (0x00002184 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CRC_CONTROL_CONTROLLING_CHANNEL                         4:0
#define LWC37D_HEAD_SET_CRC_CONTROL_EXPECT_BUFFER_COLLAPSE                      8:8
#define LWC37D_HEAD_SET_CRC_CONTROL_EXPECT_BUFFER_COLLAPSE_FALSE                (0x00000000)
#define LWC37D_HEAD_SET_CRC_CONTROL_EXPECT_BUFFER_COLLAPSE_TRUE                 (0x00000001)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC                                 19:12
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_NONE                            (0x00000000)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SF                              (0x00000030)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR(i)                          (0x00000050 +(i))
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR__SIZE_1                     8
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR0                            (0x00000050)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR1                            (0x00000051)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR2                            (0x00000052)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR3                            (0x00000053)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR4                            (0x00000054)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR5                            (0x00000055)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR6                            (0x00000056)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_SOR7                            (0x00000057)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_PIOR(i)                         (0x00000060 +(i))
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_PIOR__SIZE_1                    4
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_PIOR0                           (0x00000060)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_PIOR1                           (0x00000061)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_PIOR2                           (0x00000062)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_PIOR3                           (0x00000063)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR(i)                         (0x00000070 +(i))
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR__SIZE_1                    8
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR0                           (0x00000070)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR1                           (0x00000071)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR2                           (0x00000072)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR3                           (0x00000073)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR4                           (0x00000074)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR5                           (0x00000075)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR6                           (0x00000076)
#define LWC37D_HEAD_SET_CRC_CONTROL_PRIMARY_CRC_WBOR7                           (0x00000077)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC                               27:20
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_NONE                          (0x00000000)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SF                            (0x00000030)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR(i)                        (0x00000050 +(i))
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR__SIZE_1                   8
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR0                          (0x00000050)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR1                          (0x00000051)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR2                          (0x00000052)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR3                          (0x00000053)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR4                          (0x00000054)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR5                          (0x00000055)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR6                          (0x00000056)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_SOR7                          (0x00000057)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_PIOR(i)                       (0x00000060 +(i))
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_PIOR__SIZE_1                  4
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_PIOR0                         (0x00000060)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_PIOR1                         (0x00000061)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_PIOR2                         (0x00000062)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_PIOR3                         (0x00000063)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR(i)                       (0x00000070 +(i))
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR__SIZE_1                  8
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR0                         (0x00000070)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR1                         (0x00000071)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR2                         (0x00000072)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR3                         (0x00000073)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR4                         (0x00000074)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR5                         (0x00000075)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR6                         (0x00000076)
#define LWC37D_HEAD_SET_CRC_CONTROL_SECONDARY_CRC_WBOR7                         (0x00000077)
#define LWC37D_HEAD_SET_CRC_CONTROL_CRC_DURING_SNOOZE                           9:9
#define LWC37D_HEAD_SET_CRC_CONTROL_CRC_DURING_SNOOZE_DISABLE                   (0x00000000)
#define LWC37D_HEAD_SET_CRC_CONTROL_CRC_DURING_SNOOZE_ENABLE                    (0x00000001)
#define LWC37D_HEAD_SET_CRC_CONTROL_COLOR_DEPTH_AGNOSTIC                        10:10
#define LWC37D_HEAD_SET_CRC_CONTROL_COLOR_DEPTH_AGNOSTIC_DISABLE                (0x00000000)
#define LWC37D_HEAD_SET_CRC_CONTROL_COLOR_DEPTH_AGNOSTIC_ENABLE                 (0x00000001)
#define LWC37D_HEAD_SET_CRC_USER_DATA(a)                                        (0x00002188 + (a)*0x00000400)
#define LWC37D_HEAD_SET_CRC_USER_DATA_DATA                                      31:0
#define LWC37D_HEAD_SET_PRESENT_CONTROL(a)                                      (0x0000218C + (a)*0x00000400)
#define LWC37D_HEAD_SET_PRESENT_CONTROL_USE_BEGIN_FIELD                         0:0
#define LWC37D_HEAD_SET_PRESENT_CONTROL_USE_BEGIN_FIELD_DISABLE                 (0x00000000)
#define LWC37D_HEAD_SET_PRESENT_CONTROL_USE_BEGIN_FIELD_ENABLE                  (0x00000001)
#define LWC37D_HEAD_SET_PRESENT_CONTROL_BEGIN_FIELD                             6:4
#define LWC37D_HEAD_SET_VGA_CRC_CONTROL(a)                                      (0x00002190 + (a)*0x00000400)
#define LWC37D_HEAD_SET_VGA_CRC_CONTROL_COMPUTE                                 0:0
#define LWC37D_HEAD_SET_VGA_CRC_CONTROL_COMPUTE_DISABLE                         (0x00000000)
#define LWC37D_HEAD_SET_VGA_CRC_CONTROL_COMPUTE_ENABLE                          (0x00000001)
#define LWC37D_HEAD_SET_SW_SPARE_A(a)                                           (0x00002194 + (a)*0x00000400)
#define LWC37D_HEAD_SET_SW_SPARE_A_CODE                                         31:0
#define LWC37D_HEAD_SET_SW_SPARE_B(a)                                           (0x00002198 + (a)*0x00000400)
#define LWC37D_HEAD_SET_SW_SPARE_B_CODE                                         31:0
#define LWC37D_HEAD_SET_SW_SPARE_C(a)                                           (0x0000219C + (a)*0x00000400)
#define LWC37D_HEAD_SET_SW_SPARE_C_CODE                                         31:0
#define LWC37D_HEAD_SET_SW_SPARE_D(a)                                           (0x000021A0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_SW_SPARE_D_CODE                                         31:0
#define LWC37D_HEAD_SYNC_POINT_RELEASE(a)                                       (0x000021A4 + (a)*0x00000400)
#define LWC37D_HEAD_SYNC_POINT_RELEASE_INDEX                                    7:0
#define LWC37D_HEAD_SYNC_POINT_RELEASE_EVENT                                    11:8
#define LWC37D_HEAD_SYNC_POINT_RELEASE_EVENT_NONE                               (0x00000000)
#define LWC37D_HEAD_SYNC_POINT_RELEASE_EVENT_FRAME_START                        (0x00000002)
#define LWC37D_HEAD_SYNC_POINT_RELEASE_EVENT_FRAME_END                          (0x00000003)
#define LWC37D_HEAD_SET_DISPLAY_RATE(a)                                         (0x000021A8 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DISPLAY_RATE_RUN_MODE                                   0:0
#define LWC37D_HEAD_SET_DISPLAY_RATE_RUN_MODE_CONTINUOUS                        (0x00000000)
#define LWC37D_HEAD_SET_DISPLAY_RATE_RUN_MODE_ONE_SHOT                          (0x00000001)
#define LWC37D_HEAD_SET_DISPLAY_RATE_MIN_REFRESH_INTERVAL                       25:4
#define LWC37D_HEAD_SET_DISPLAY_RATE_MIN_REFRESH                                2:2
#define LWC37D_HEAD_SET_DISPLAY_RATE_MIN_REFRESH_DISABLE                        (0x00000000)
#define LWC37D_HEAD_SET_DISPLAY_RATE_MIN_REFRESH_ENABLE                         (0x00000001)
#define LWC37D_HEAD_SET_DSC_TOP_CTL(a)                                          (0x000021AC + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_ENABLE                                  0:0
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_ENABLE_DISABLE                          (0x00000000)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_ENABLE_ENABLE                           (0x00000001)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_AUTO_RESET                              1:1
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_AUTO_RESET_DISABLE                      (0x00000000)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_AUTO_RESET_ENABLE                       (0x00000001)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_DUAL_ENABLE                             2:2
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_DUAL_ENABLE_DISABLE                     (0x00000000)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_DUAL_ENABLE_ENABLE                      (0x00000001)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_FORCE_ICH_EOL_RESET                     3:3
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_FORCE_ICH_EOL_RESET_DISABLE             (0x00000000)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_FORCE_ICH_EOL_RESET_ENABLE              (0x00000001)
#define LWC37D_HEAD_SET_DSC_TOP_CTL_DSC_TIMEOUT_COUNTER                         19:4
#define LWC37D_HEAD_SET_DSC_DELAY(a)                                            (0x000021B0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_DELAY_DSC_CORE_OUTPUT_DELAY                         15:0
#define LWC37D_HEAD_SET_DSC_DELAY_DSC_WRAP_OUTPUT_DELAY                         31:16
#define LWC37D_HEAD_SET_DSC_COMMON_CTL(a)                                       (0x000021B4 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_COMMON_CTL_DSC_BITS_PER_PIXEL                       9:0
#define LWC37D_HEAD_SET_DSC_COMMON_CTL_DSC_BLOCK_PRED_ENABLE                    12:12
#define LWC37D_HEAD_SET_DSC_COMMON_CTL_DSC_BLOCK_PRED_ENABLE_DISABLE            (0x00000000)
#define LWC37D_HEAD_SET_DSC_COMMON_CTL_DSC_BLOCK_PRED_ENABLE_ENABLE             (0x00000001)
#define LWC37D_HEAD_SET_DSC_COMMON_CTL_DSC_CHUNK_SIZE                           31:16
#define LWC37D_HEAD_SET_DSC_SLICE_INFO(a)                                       (0x000021B8 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_SLICE_INFO_DSC_SLICE_WIDTH                          15:0
#define LWC37D_HEAD_SET_DSC_SLICE_INFO_DSC_SLICE_HEIGHT                         31:16
#define LWC37D_HEAD_SET_DSC_RC_DELAY_INFO(a)                                    (0x000021BC + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_DELAY_INFO_DSC_INITIAL_XMIT_DELAY                9:0
#define LWC37D_HEAD_SET_DSC_RC_DELAY_INFO_DSC_INITIAL_DEC_DELAY                 27:12
#define LWC37D_HEAD_SET_DSC_RC_SCALE_INFO(a)                                    (0x000021C0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_SCALE_INFO_DSC_INITIAL_SCALE_VALUE               5:0
#define LWC37D_HEAD_SET_DSC_RC_SCALE_INFO_DSC_SCALE_DECR_INTERVAL               23:8
#define LWC37D_HEAD_SET_DSC_RC_SCALE_INFO2(a)                                   (0x000021C4 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_SCALE_INFO2_DSC_SCALE_INCR_INTERVAL              15:0
#define LWC37D_HEAD_SET_DSC_RC_BPGOFF_INFO(a)                                   (0x000021C8 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_BPGOFF_INFO_DSC_NFL_BPG_OFFSET                   15:0
#define LWC37D_HEAD_SET_DSC_RC_BPGOFF_INFO_DSC_SLICE_BPG_OFFSET                 31:16
#define LWC37D_HEAD_SET_DSC_RC_OFFSET_INFO(a)                                   (0x000021CC + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_OFFSET_INFO_DSC_INITIAL_OFFSET                   15:0
#define LWC37D_HEAD_SET_DSC_RC_OFFSET_INFO_DSC_FINAL_OFFSET                     31:16
#define LWC37D_HEAD_SET_DSC_RC_FLATNESS_INFO(a)                                 (0x000021D0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_FLATNESS_INFO_DSC_FLATNESS_MIN_QP                4:0
#define LWC37D_HEAD_SET_DSC_RC_FLATNESS_INFO_DSC_FLATNESS_MAX_QP                12:8
#define LWC37D_HEAD_SET_DSC_RC_FLATNESS_INFO_DSC_FIRST_LINE_BPG_OFFS            20:16
#define LWC37D_HEAD_SET_DSC_RC_PARAM_SET(a)                                     (0x000021D4 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_PARAM_SET_DSC_RC_EDGE_FACTOR                     3:0
#define LWC37D_HEAD_SET_DSC_RC_PARAM_SET_DSC_RC_QUANT_INCR_LIMIT0               8:4
#define LWC37D_HEAD_SET_DSC_RC_PARAM_SET_DSC_RC_QUANT_INCR_LIMIT1               16:12
#define LWC37D_HEAD_SET_DSC_RC_PARAM_SET_DSC_RC_TGT_OFFSET_HI                   23:20
#define LWC37D_HEAD_SET_DSC_RC_PARAM_SET_DSC_RC_TGT_OFFSET_LO                   27:24
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH0(a)                                   (0x000021D8 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH0_DSC_RC_MODEL_SIZE                    15:0
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH0_DSC_RC_BUF_THRESH0                   23:16
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH0_DSC_RC_BUF_THRESH1                   31:24
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH1(a)                                   (0x000021DC + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH1_DSC_RC_BUF_THRESH2                   7:0
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH1_DSC_RC_BUF_THRESH3                   15:8
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH1_DSC_RC_BUF_THRESH4                   23:16
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH1_DSC_RC_BUF_THRESH5                   31:24
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH2(a)                                   (0x000021E0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH2_DSC_RC_BUF_THRESH6                   7:0
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH2_DSC_RC_BUF_THRESH7                   15:8
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH2_DSC_RC_BUF_THRESH8                   23:16
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH2_DSC_RC_BUF_THRESH9                   31:24
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH3(a)                                   (0x000021E4 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH3_DSC_RC_BUF_THRESH10                  7:0
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH3_DSC_RC_BUF_THRESH11                  15:8
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH3_DSC_RC_BUF_THRESH12                  23:16
#define LWC37D_HEAD_SET_DSC_RC_BUF_THRESH3_DSC_RC_BUF_THRESH13                  31:24
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG0(a)                                    (0x000021E8 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG0_DSC_RC_RANGE_PARAM0MIN_QP             4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG0_DSC_RC_RANGE_PARAM0MAX_QP             9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG0_DSC_RC_RANGE_PARAM0BPG_OFFSET         15:10
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG0_DSC_RC_RANGE_PARAM1MIN_QP             20:16
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG0_DSC_RC_RANGE_PARAM1MAX_QP             25:21
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG0_DSC_RC_RANGE_PARAM1BPG_OFFSET         31:26
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG1(a)                                    (0x000021EC + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG1_DSC_RC_RANGE_PARAM2MIN_QP             4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG1_DSC_RC_RANGE_PARAM2MAX_QP             9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG1_DSC_RC_RANGE_PARAM2BPG_OFFSET         15:10
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG1_DSC_RC_RANGE_PARAM3MIN_QP             20:16
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG1_DSC_RC_RANGE_PARAM3MAX_QP             25:21
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG1_DSC_RC_RANGE_PARAM3BPG_OFFSET         31:26
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG2(a)                                    (0x000021F0 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG2_DSC_RC_RANGE_PARAM4MIN_QP             4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG2_DSC_RC_RANGE_PARAM4MAX_QP             9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG2_DSC_RC_RANGE_PARAM4BPG_OFFSET         15:10
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG2_DSC_RC_RANGE_PARAM5MIN_QP             20:16
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG2_DSC_RC_RANGE_PARAM5MAX_QP             25:21
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG2_DSC_RC_RANGE_PARAM5BPG_OFFSET         31:26
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG3(a)                                    (0x000021F4 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG3_DSC_RC_RANGE_PARAM6MIN_QP             4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG3_DSC_RC_RANGE_PARAM6MAX_QP             9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG3_DSC_RC_RANGE_PARAM6BPG_OFFSET         15:10
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG3_DSC_RC_RANGE_PARAM7MIN_QP             20:16
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG3_DSC_RC_RANGE_PARAM7MAX_QP             25:21
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG3_DSC_RC_RANGE_PARAM7BPG_OFFSET         31:26
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG4(a)                                    (0x000021F8 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG4_DSC_RC_RANGE_PARAM8MIN_QP             4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG4_DSC_RC_RANGE_PARAM8MAX_QP             9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG4_DSC_RC_RANGE_PARAM8BPG_OFFSET         15:10
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG4_DSC_RC_RANGE_PARAM9MIN_QP             20:16
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG4_DSC_RC_RANGE_PARAM9MAX_QP             25:21
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG4_DSC_RC_RANGE_PARAM9BPG_OFFSET         31:26
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG5(a)                                    (0x000021FC + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG5_DSC_RC_RANGE_PARAM10MIN_QP            4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG5_DSC_RC_RANGE_PARAM10MAX_QP            9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG5_DSC_RC_RANGE_PARAM10BPG_OFFSET        15:10
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG5_DSC_RC_RANGE_PARAM11MIN_QP            20:16
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG5_DSC_RC_RANGE_PARAM11MAX_QP            25:21
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG5_DSC_RC_RANGE_PARAM11BPG_OFFSET        31:26
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG6(a)                                    (0x00002200 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG6_DSC_RC_RANGE_PARAM12MIN_QP            4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG6_DSC_RC_RANGE_PARAM12MAX_QP            9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG6_DSC_RC_RANGE_PARAM12BPG_OFFSET        15:10
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG6_DSC_RC_RANGE_PARAM13MIN_QP            20:16
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG6_DSC_RC_RANGE_PARAM13MAX_QP            25:21
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG6_DSC_RC_RANGE_PARAM13BPG_OFFSET        31:26
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG7(a)                                    (0x00002204 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG7_DSC_RC_RANGE_PARAM14MIN_QP            4:0
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG7_DSC_RC_RANGE_PARAM14MAX_QP            9:5
#define LWC37D_HEAD_SET_DSC_RC_RANGE_CFG7_DSC_RC_RANGE_PARAM14BPG_OFFSET        15:10
#define LWC37D_HEAD_SET_DSC_UNIT_SET(a)                                         (0x00002208 + (a)*0x00000400)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_SLICE_NUM_MINUS1IN_LINE                1:0
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_LINEBUF_DEPTH                          4:4
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_LINEBUF_DEPTH_BITS_9                   (0x00000000)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_LINEBUF_DEPTH_BITS_8                   (0x00000001)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_RC_SOLUTION_MODE                       9:8
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_RC_SOLUTION_MODE_DISABLE               (0x00000000)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_RC_SOLUTION_MODE_FULL_FIX              (0x00000001)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_RC_SOLUTION_MODE_MENTAL_FRIENDLY_FIX   (0x00000002)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_CHECK_FLATNESS2                        10:10
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_CHECK_FLATNESS2_FALSE                  (0x00000000)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_CHECK_FLATNESS2_TRUE                   (0x00000001)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_FLATNESS_FIX_EN                        11:11
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_FLATNESS_FIX_EN_DISABLE                (0x00000000)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_FLATNESS_FIX_EN_ENABLE                 (0x00000001)
#define LWC37D_HEAD_SET_DSC_UNIT_SET_DSC_RC_OVERFLOW_THRESH                     21:12
#define LWC37D_HEAD_SET_RSB(a)                                                  (0x0000220C + (a)*0x00000400)
#define LWC37D_HEAD_SET_RSB_NON_SELWRE                                          0:0
#define LWC37D_HEAD_SET_STREAM_ID(a)                                            (0x00002210 + (a)*0x00000400)
#define LWC37D_HEAD_SET_STREAM_ID_ID                                            9:0
#define LWC37D_HEAD_SET_OUTPUT_SCALER_COEFF_VALUE(a)                            (0x00002214 + (a)*0x00000400)
#define LWC37D_HEAD_SET_OUTPUT_SCALER_COEFF_VALUE_DATA                          9:0
#define LWC37D_HEAD_SET_OUTPUT_SCALER_COEFF_VALUE_INDEX                         19:12
#define LWC37D_HEAD_SET_MIN_FRAME_IDLE(a)                                       (0x00002218 + (a)*0x00000400)
#define LWC37D_HEAD_SET_MIN_FRAME_IDLE_LEADING_RASTER_LINES                     14:0
#define LWC37D_HEAD_SET_MIN_FRAME_IDLE_TRAILING_RASTER_LINES                    30:16

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clC37d_h
