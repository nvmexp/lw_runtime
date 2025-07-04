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



#ifndef _cl977d_h_
#define _cl977d_h_


#ifdef __cplusplus
extern "C" {
#endif

#define LW977D_CORE_CHANNEL_DMA                                                 (0x0000977D)

typedef volatile struct _cl977d_tag0 {
    LwV32 Put;                                                                  // 0x00000000 - 0x00000003
    LwV32 Get;                                                                  // 0x00000004 - 0x00000007
    LwV32 Reserved00[0x1E];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetNotifierControl;                                                   // 0x00000084 - 0x00000087
    LwV32 SetContextDmaNotifier;                                                // 0x00000088 - 0x0000008B
    LwV32 GetCapabilities;                                                      // 0x0000008C - 0x0000008F
    LwV32 Reserved01[0x37];
    LwV32 SetSpare;                                                             // 0x0000016C - 0x0000016F
    LwV32 SetSpareNoop[4];                                                      // 0x00000170 - 0x0000017F
    struct {
        LwV32 SetControl;                                                       // 0x00000180 - 0x00000183
        LwV32 SetSwSpareA;                                                      // 0x00000184 - 0x00000187
        LwV32 SetSwSpareB;                                                      // 0x00000188 - 0x0000018B
        LwV32 Reserved02[0x1];
        LwV32 SetLwstomReason;                                                  // 0x00000190 - 0x00000193
        LwV32 Reserved03[0x3];
    } Dac[4];
    struct {
        LwV32 SetControl;                                                       // 0x00000200 - 0x00000203
        LwV32 SetSwSpareA;                                                      // 0x00000204 - 0x00000207
        LwV32 SetSwSpareB;                                                      // 0x00000208 - 0x0000020B
        LwV32 Reserved04[0x1];
        LwV32 SetLwstomReason;                                                  // 0x00000210 - 0x00000213
        LwV32 Reserved05[0x3];
    } Sor[8];
    struct {
        LwV32 SetControl;                                                       // 0x00000300 - 0x00000303
        LwV32 SetSwSpareA;                                                      // 0x00000304 - 0x00000307
        LwV32 SetSwSpareB;                                                      // 0x00000308 - 0x0000030B
        LwV32 Reserved06[0x1];
        LwV32 SetLwstomReason;                                                  // 0x00000310 - 0x00000313
        LwV32 Reserved07[0x3];
    } Pior[4];
    LwV32 Reserved08[0x20];
    struct {
        LwV32 SetPresentControl;                                                // 0x00000400 - 0x00000403
        LwV32 SetControlOutputResource;                                         // 0x00000404 - 0x00000407
        LwV32 SetControl;                                                       // 0x00000408 - 0x0000040B
        LwV32 SetLockOffset;                                                    // 0x0000040C - 0x0000040F
        LwV32 SetOverscanColor;                                                 // 0x00000410 - 0x00000413
        LwV32 SetRasterSize;                                                    // 0x00000414 - 0x00000417
        LwV32 SetRasterSyncEnd;                                                 // 0x00000418 - 0x0000041B
        LwV32 SetRasterBlankEnd;                                                // 0x0000041C - 0x0000041F
        LwV32 SetRasterBlankStart;                                              // 0x00000420 - 0x00000423
        LwV32 SetRasterVertBlank2;                                              // 0x00000424 - 0x00000427
        LwV32 SetLockChain;                                                     // 0x00000428 - 0x0000042B
        LwV32 SetDefaultBaseColor;                                              // 0x0000042C - 0x0000042F
        LwV32 SetCrcControl;                                                    // 0x00000430 - 0x00000433
        LwV32 SetLegacyCrcControl;                                              // 0x00000434 - 0x00000437
        LwV32 SetContextDmaCrc;                                                 // 0x00000438 - 0x0000043B
        LwV32 Reserved09[0x1];
        LwV32 SetBaseLutLo;                                                     // 0x00000440 - 0x00000443
        LwV32 SetBaseLutHi;                                                     // 0x00000444 - 0x00000447
        LwV32 SetOutputLutLo;                                                   // 0x00000448 - 0x0000044B
        LwV32 SetOutputLutHi;                                                   // 0x0000044C - 0x0000044F
        LwV32 SetPixelClockFrequency;                                           // 0x00000450 - 0x00000453
        LwV32 SetPixelClockConfiguration;                                       // 0x00000454 - 0x00000457
        LwV32 SetPixelClockFrequencyMax;                                        // 0x00000458 - 0x0000045B
        LwV32 SetContextDmaLut;                                                 // 0x0000045C - 0x0000045F
        LwV32 SetOffset;                                                        // 0x00000460 - 0x00000463
        LwV32 Reserved10[0x1];
        LwV32 SetSize;                                                          // 0x00000468 - 0x0000046B
        LwV32 SetStorage;                                                       // 0x0000046C - 0x0000046F
        LwV32 SetParams;                                                        // 0x00000470 - 0x00000473
        LwV32 SetContextDmasIso;                                                // 0x00000474 - 0x00000477
        LwV32 Reserved11[0x1];
        LwV32 SetPresentControlLwrsor;                                          // 0x0000047C - 0x0000047F
        LwV32 SetControlLwrsor;                                                 // 0x00000480 - 0x00000483
        LwV32 SetOffsetsLwrsor[2];                                              // 0x00000484 - 0x0000048B
        LwV32 SetContextDmasLwrsor[2];                                          // 0x0000048C - 0x00000493
        LwV32 SetControlOutputScaler;                                           // 0x00000494 - 0x00000497
        LwV32 SetProcamp;                                                       // 0x00000498 - 0x0000049B
        LwV32 Reserved12[0x1];
        LwV32 SetDitherControl;                                                 // 0x000004A0 - 0x000004A3
        LwV32 Reserved13[0x3];
        LwV32 SetViewportPointIn;                                               // 0x000004B0 - 0x000004B3
        LwV32 Reserved14[0x1];
        LwV32 SetViewportSizeIn;                                                // 0x000004B8 - 0x000004BB
        LwV32 SetViewportPointOutAdjust;                                        // 0x000004BC - 0x000004BF
        LwV32 SetViewportSizeOut;                                               // 0x000004C0 - 0x000004C3
        LwV32 SetViewportSizeOutMin;                                            // 0x000004C4 - 0x000004C7
        LwV32 SetViewportSizeOutMax;                                            // 0x000004C8 - 0x000004CB
        LwV32 Reserved15[0x1];
        LwV32 SetBaseChannelUsageBounds;                                        // 0x000004D0 - 0x000004D3
        LwV32 SetOverlayUsageBounds;                                            // 0x000004D4 - 0x000004D7
        LwV32 Reserved16[0x2];
        LwV32 SetProcessing;                                                    // 0x000004E0 - 0x000004E3
        LwV32 SetColwersionRed;                                                 // 0x000004E4 - 0x000004E7
        LwV32 SetColwersionGrn;                                                 // 0x000004E8 - 0x000004EB
        LwV32 SetColwersionBlu;                                                 // 0x000004EC - 0x000004EF
        LwV32 SetCscRed2Red;                                                    // 0x000004F0 - 0x000004F3
        LwV32 SetCscGrn2Red;                                                    // 0x000004F4 - 0x000004F7
        LwV32 SetCscBlu2Red;                                                    // 0x000004F8 - 0x000004FB
        LwV32 SetCscConstant2Red;                                               // 0x000004FC - 0x000004FF
        LwV32 SetCscRed2Grn;                                                    // 0x00000500 - 0x00000503
        LwV32 SetCscGrn2Grn;                                                    // 0x00000504 - 0x00000507
        LwV32 SetCscBlu2Grn;                                                    // 0x00000508 - 0x0000050B
        LwV32 SetCscConstant2Grn;                                               // 0x0000050C - 0x0000050F
        LwV32 SetCscRed2Blu;                                                    // 0x00000510 - 0x00000513
        LwV32 SetCscGrn2Blu;                                                    // 0x00000514 - 0x00000517
        LwV32 SetCscBlu2Blu;                                                    // 0x00000518 - 0x0000051B
        LwV32 SetCscConstant2Blu;                                               // 0x0000051C - 0x0000051F
        LwV32 SetHdmiCtrl;                                                      // 0x00000520 - 0x00000523
        LwV32 SetVactiveSpaceColor;                                             // 0x00000524 - 0x00000527
        LwV32 SetPixelReorderControl;                                           // 0x00000528 - 0x0000052B
        LwV32 SetDisplayId[2];                                                  // 0x0000052C - 0x00000533
        LwV32 Reserved17[0x6];
        LwV32 SetSwSpareA;                                                      // 0x0000054C - 0x0000054F
        LwV32 SetSwSpareB;                                                      // 0x00000550 - 0x00000553
        LwV32 SetSwSpareC;                                                      // 0x00000554 - 0x00000557
        LwV32 SetSwSpareD;                                                      // 0x00000558 - 0x0000055B
        LwV32 SetGetBlankingCtrl;                                               // 0x0000055C - 0x0000055F
        LwV32 SetControlCompression;                                            // 0x00000560 - 0x00000563
        LwV32 SetControlCompressionLA;                                          // 0x00000564 - 0x00000567
        LwV32 SetStallLock;                                                     // 0x00000568 - 0x0000056B
        LwV32 Reserved18[0x59];
        LwV32 SetSwMethodPlaceholderA;                                          // 0x000006D0 - 0x000006D3
        LwV32 SetSwMethodPlaceholderB;                                          // 0x000006D4 - 0x000006D7
        LwV32 SetSwMethodPlaceholderC;                                          // 0x000006D8 - 0x000006DB
        LwV32 SetSwMethodPlaceholderD;                                          // 0x000006DC - 0x000006DF
        LwV32 Reserved19[0x3];
        LwV32 SetSpare;                                                         // 0x000006EC - 0x000006EF
        LwV32 SetSpareNoop[4];                                                  // 0x000006F0 - 0x000006FF
    } Head[4];
} LW977DDispControlDma;


#define LW977D_CORE_NOTIFIER_3                                                      0x00000000
#define LW977D_CORE_NOTIFIER_3_SIZEOF                                               0x00000150
#define LW977D_CORE_NOTIFIER_3_COMPLETION_0                                         0x00000000
#define LW977D_CORE_NOTIFIER_3_COMPLETION_0_DONE                                    0:0
#define LW977D_CORE_NOTIFIER_3_COMPLETION_0_DONE_FALSE                              0x00000000
#define LW977D_CORE_NOTIFIER_3_COMPLETION_0_DONE_TRUE                               0x00000001
#define LW977D_CORE_NOTIFIER_3_COMPLETION_0_R0                                      15:1
#define LW977D_CORE_NOTIFIER_3_COMPLETION_0_TIMESTAMP                               29:16
#define LW977D_CORE_NOTIFIER_3__1                                                   0x00000001
#define LW977D_CORE_NOTIFIER_3__1_R1                                                31:0
#define LW977D_CORE_NOTIFIER_3__2                                                   0x00000002
#define LW977D_CORE_NOTIFIER_3__2_R2                                                31:0
#define LW977D_CORE_NOTIFIER_3__3                                                   0x00000003
#define LW977D_CORE_NOTIFIER_3__3_R3                                                31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4                                       0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_DONE                                  0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_DONE_FALSE                            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_DONE_TRUE                             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_VM_USABLE4ISO                         1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_VM_USABLE4ISO_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_VM_USABLE4ISO_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_LWM_USABLE4ISO                        2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_LWM_USABLE4ISO_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_LWM_USABLE4ISO_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_R0                                    19:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_FOS_FETCH_X4AA                        20:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_FOS_FETCH_X4AA_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_FOS_FETCH_X4AA_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_R1                                    29:21
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_INTERNAL_FLIP_LOCK_PIN0USABLE         30:30
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_INTERNAL_FLIP_LOCK_PIN0USABLE_FALSE   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_INTERNAL_FLIP_LOCK_PIN0USABLE_TRUE    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_INTERNAL_FLIP_LOCK_PIN1USABLE         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_INTERNAL_FLIP_LOCK_PIN1USABLE_FALSE   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_4_INTERNAL_FLIP_LOCK_PIN1USABLE_TRUE    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5                                       0x00000005
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN0USAGE                        3:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN0USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN0USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN0USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN0USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN1USAGE                        7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN1USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN1USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN1USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN1USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN2USAGE                        11:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN2USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN2USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN2USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN2USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN3USAGE                        15:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN3USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN3USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN3USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN3USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN4USAGE                        19:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN4USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN4USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN4USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN4USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN5USAGE                        23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN5USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN5USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN5USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN5USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN6USAGE                        27:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN6USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN6USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN6USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN6USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN7USAGE                        31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN7USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN7USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN7USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_5_LOCK_PIN7USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6                                       0x00000006
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN8USAGE                        3:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN8USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN8USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN8USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN8USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN9USAGE                        7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN9USAGE_UNAVAILABLE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN9USAGE_SCAN_LOCK              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN9USAGE_FLIP_LOCK              0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN9USAGE_STEREO                 0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_AUSAGE                       11:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_AUSAGE_UNAVAILABLE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_AUSAGE_SCAN_LOCK             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_AUSAGE_FLIP_LOCK             0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_AUSAGE_STEREO                0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_BUSAGE                       15:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_BUSAGE_UNAVAILABLE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_BUSAGE_SCAN_LOCK             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_BUSAGE_FLIP_LOCK             0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_BUSAGE_STEREO                0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_LWSAGE                       19:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_LWSAGE_UNAVAILABLE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_LWSAGE_SCAN_LOCK             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_LWSAGE_FLIP_LOCK             0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_LWSAGE_STEREO                0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_DUSAGE                       23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_DUSAGE_UNAVAILABLE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_DUSAGE_SCAN_LOCK             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_DUSAGE_FLIP_LOCK             0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_DUSAGE_STEREO                0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_EUSAGE                       27:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_EUSAGE_UNAVAILABLE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_EUSAGE_SCAN_LOCK             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_EUSAGE_FLIP_LOCK             0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_EUSAGE_STEREO                0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_FUSAGE                       31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_FUSAGE_UNAVAILABLE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_FUSAGE_SCAN_LOCK             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_FUSAGE_FLIP_LOCK             0x00000002
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_6_LOCK_PIN_FUSAGE_STEREO                0x00000004
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_7                                       0x00000007
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_7_DISPCLK_MAX                           7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_7_R4                                    31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_8                                       0x00000008
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_8_R5                                    31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_9                                       0x00000009
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_9_R6                                    31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_10                                      0x0000000A
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_10_R7                                   31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_11                                      0x0000000B
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_11_R8                                   31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12                             0x0000000C
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_RGB_USABLE                  0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_RGB_USABLE_FALSE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_RGB_USABLE_TRUE             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_TV_USABLE                   1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_TV_USABLE_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_TV_USABLE_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_TV_MACROVISION_USABLE       2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_TV_MACROVISION_USABLE_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_TV_MACROVISION_USABLE_TRUE  0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_SCART_USABLE                3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_SCART_USABLE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_SCART_USABLE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_12_R0                          31:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_13                             0x0000000D
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_13_CRT_CLK_MAX                 7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC0_13_R1                          31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14                             0x0000000E
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_RGB_USABLE                  0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_RGB_USABLE_FALSE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_RGB_USABLE_TRUE             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_TV_USABLE                   1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_TV_USABLE_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_TV_USABLE_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_TV_MACROVISION_USABLE       2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_TV_MACROVISION_USABLE_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_TV_MACROVISION_USABLE_TRUE  0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_SCART_USABLE                3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_SCART_USABLE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_SCART_USABLE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_14_R0                          31:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_15                             0x0000000F
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_15_CRT_CLK_MAX                 7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC1_15_R1                          31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16                             0x00000010
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_RGB_USABLE                  0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_RGB_USABLE_FALSE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_RGB_USABLE_TRUE             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_TV_USABLE                   1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_TV_USABLE_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_TV_USABLE_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_TV_MACROVISION_USABLE       2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_TV_MACROVISION_USABLE_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_TV_MACROVISION_USABLE_TRUE  0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_SCART_USABLE                3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_SCART_USABLE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_SCART_USABLE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_16_R0                          31:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_17                             0x00000011
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_17_CRT_CLK_MAX                 7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC2_17_R1                          31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18                             0x00000012
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_RGB_USABLE                  0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_RGB_USABLE_FALSE            0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_RGB_USABLE_TRUE             0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_TV_USABLE                   1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_TV_USABLE_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_TV_USABLE_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_TV_MACROVISION_USABLE       2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_TV_MACROVISION_USABLE_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_TV_MACROVISION_USABLE_TRUE  0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_SCART_USABLE                3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_SCART_USABLE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_SCART_USABLE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_18_R0                          31:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_19                             0x00000013
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_19_CRT_CLK_MAX                 7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_DAC3_19_R1                          31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20                             0x00000014
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_20_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_21                             0x00000015
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_21_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_21_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_21_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR0_21_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22                             0x00000016
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_22_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_23                             0x00000017
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_23_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_23_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_23_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR1_23_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24                             0x00000018
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_24_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_25                             0x00000019
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_25_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_25_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_25_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR2_25_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26                             0x0000001A
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_26_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_27                             0x0000001B
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_27_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_27_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_27_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR3_27_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28                             0x0000001C
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_28_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_29                             0x0000001D
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_29_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_29_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_29_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR4_29_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30                             0x0000001E
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_30_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_31                             0x0000001F
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_31_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_31_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_31_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR5_31_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32                             0x00000020
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_32_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_33                             0x00000021
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_33_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_33_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_33_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR6_33_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34                             0x00000022
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_LVDS18               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_LVDS18_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_LVDS18_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_LVDS24               1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_LVDS24_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_LVDS24_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_LVDS18                 2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_LVDS18_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_LVDS18_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_LVDS24                 3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_LVDS24_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_LVDS24_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_R0                          7:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_TMDS_A               8:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_TMDS_A_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_TMDS_A_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_TMDS_B               9:9
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_TMDS_B_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SINGLE_TMDS_B_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_R1                          10:10
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_TMDS                   11:11
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_TMDS_FALSE             0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DUAL_TMDS_TRUE              0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_R2                          12:12
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DISPLAY_OVER_PCIE           13:13
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DISPLAY_OVER_PCIE_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DISPLAY_OVER_PCIE_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_R3                          15:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SDI                         16:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SDI_FALSE                   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_SDI_TRUE                    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_R4                          19:17
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_R5                          23:20
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_A                        24:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_A_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_A_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_B                        25:25
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_B_FALSE                  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_B_TRUE                   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_INTERLACE                26:26
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_INTERLACE_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP_INTERLACE_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP8LANES                    27:27
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP8LANES_FALSE              0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_DP8LANES_TRUE               0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_34_R6                          31:28
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_35                             0x00000023
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_35_DP_CLK_MAX                  7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_35_R7                          15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_35_TMDS_CLK_MAX                23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_SOR7_35_LVDS_CLK_MAX                31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36                            0x00000024
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TMDS_ENC               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TMDS_ENC_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TMDS_ENC_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TV_ENC                 1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TV_ENC_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TV_ENC_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_SDI_ENC                2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_SDI_ENC_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_SDI_ENC_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_OUT            3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_OUT_FALSE      0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_OUT_TRUE       0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_IN             4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_IN_FALSE       0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_IN_TRUE        0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_IN_OUT         5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_IN_OUT_FALSE   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_DIST_RENDER_IN_OUT_TRUE    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TMDS10BPC_ALLOWED      6:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TMDS10BPC_ALLOWED_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_EXT_TMDS10BPC_ALLOWED_TRUE 0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_36_R0                         31:7
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_37                            0x00000025
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_37_EXT_ENC_CLK_MAX            7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_37_R1                         15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_37_DIST_RENDER_CLK_MAX        23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR0_37_R2                         31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38                            0x00000026
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TMDS_ENC               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TMDS_ENC_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TMDS_ENC_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TV_ENC                 1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TV_ENC_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TV_ENC_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_SDI_ENC                2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_SDI_ENC_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_SDI_ENC_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_OUT            3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_OUT_FALSE      0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_OUT_TRUE       0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_IN             4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_IN_FALSE       0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_IN_TRUE        0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_IN_OUT         5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_IN_OUT_FALSE   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_DIST_RENDER_IN_OUT_TRUE    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TMDS10BPC_ALLOWED      6:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TMDS10BPC_ALLOWED_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_EXT_TMDS10BPC_ALLOWED_TRUE 0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_38_R0                         31:7
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_39                            0x00000027
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_39_EXT_ENC_CLK_MAX            7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_39_R1                         15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_39_DIST_RENDER_CLK_MAX        23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR1_39_R2                         31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40                            0x00000028
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TMDS_ENC               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TMDS_ENC_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TMDS_ENC_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TV_ENC                 1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TV_ENC_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TV_ENC_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_SDI_ENC                2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_SDI_ENC_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_SDI_ENC_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_OUT            3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_OUT_FALSE      0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_OUT_TRUE       0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_IN             4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_IN_FALSE       0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_IN_TRUE        0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_IN_OUT         5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_IN_OUT_FALSE   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_DIST_RENDER_IN_OUT_TRUE    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TMDS10BPC_ALLOWED      6:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TMDS10BPC_ALLOWED_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_EXT_TMDS10BPC_ALLOWED_TRUE 0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_40_R0                         31:7
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_41                            0x00000029
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_41_EXT_ENC_CLK_MAX            7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_41_R1                         15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_41_DIST_RENDER_CLK_MAX        23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR2_41_R2                         31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42                            0x0000002A
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TMDS_ENC               0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TMDS_ENC_FALSE         0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TMDS_ENC_TRUE          0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TV_ENC                 1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TV_ENC_FALSE           0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TV_ENC_TRUE            0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_SDI_ENC                2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_SDI_ENC_FALSE          0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_SDI_ENC_TRUE           0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_OUT            3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_OUT_FALSE      0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_OUT_TRUE       0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_IN             4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_IN_FALSE       0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_IN_TRUE        0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_IN_OUT         5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_IN_OUT_FALSE   0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_DIST_RENDER_IN_OUT_TRUE    0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TMDS10BPC_ALLOWED      6:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TMDS10BPC_ALLOWED_FALSE 0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_EXT_TMDS10BPC_ALLOWED_TRUE 0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_42_R0                         31:7
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_43                            0x0000002B
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_43_EXT_ENC_CLK_MAX            7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_43_R1                         15:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_43_DIST_RENDER_CLK_MAX        23:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_PIOR3_43_R2                         31:24
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44                            0x0000002C
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_FRAME_SEQ           0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_FRAME_SEQ_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_FRAME_SEQ_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_TOP_BOTTOM          1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_TOP_BOTTOM_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_TOP_BOTTOM_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_SIDE_BY_SIDE        2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_SIDE_BY_SIDE_FALSE  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_SIDE_BY_SIDE_TRUE   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_FRAME_PACK          3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_FRAME_PACK_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_STEREO_FRAME_PACK_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_OUTPUT_YUV420              4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_OUTPUT_YUV420_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_OUTPUT_YUV420_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_OUTPUT_YUV444              5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_OUTPUT_YUV444_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_OUTPUT_YUV444_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_R0                         15:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_44_MAX_PIXELS_PER_LINE        31:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_45                            0x0000002D
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR0_45_R2                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46                            0x0000002E
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_FRAME_SEQ           0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_FRAME_SEQ_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_FRAME_SEQ_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_TOP_BOTTOM          1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_TOP_BOTTOM_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_TOP_BOTTOM_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_SIDE_BY_SIDE        2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_SIDE_BY_SIDE_FALSE  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_SIDE_BY_SIDE_TRUE   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_FRAME_PACK          3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_FRAME_PACK_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_STEREO_FRAME_PACK_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_OUTPUT_YUV420              4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_OUTPUT_YUV420_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_OUTPUT_YUV420_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_OUTPUT_YUV444              5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_OUTPUT_YUV444_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_OUTPUT_YUV444_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_R0                         15:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_46_MAX_PIXELS_PER_LINE        31:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_47                            0x0000002F
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR1_47_R2                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48                            0x00000030
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_FRAME_SEQ           0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_FRAME_SEQ_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_FRAME_SEQ_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_TOP_BOTTOM          1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_TOP_BOTTOM_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_TOP_BOTTOM_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_SIDE_BY_SIDE        2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_SIDE_BY_SIDE_FALSE  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_SIDE_BY_SIDE_TRUE   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_FRAME_PACK          3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_FRAME_PACK_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_STEREO_FRAME_PACK_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_OUTPUT_YUV420              4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_OUTPUT_YUV420_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_OUTPUT_YUV420_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_OUTPUT_YUV444              5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_OUTPUT_YUV444_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_OUTPUT_YUV444_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_R0                         15:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_48_MAX_PIXELS_PER_LINE        31:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_49                            0x00000031
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR2_49_R2                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50                            0x00000032
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_FRAME_SEQ           0:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_FRAME_SEQ_FALSE     0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_FRAME_SEQ_TRUE      0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_TOP_BOTTOM          1:1
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_TOP_BOTTOM_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_TOP_BOTTOM_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_SIDE_BY_SIDE        2:2
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_SIDE_BY_SIDE_FALSE  0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_SIDE_BY_SIDE_TRUE   0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_FRAME_PACK          3:3
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_FRAME_PACK_FALSE    0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_STEREO_FRAME_PACK_TRUE     0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_OUTPUT_YUV420              4:4
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_OUTPUT_YUV420_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_OUTPUT_YUV420_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_OUTPUT_YUV444              5:5
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_OUTPUT_YUV444_FALSE        0x00000000
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_OUTPUT_YUV444_TRUE         0x00000001
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_R0                         15:6
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_50_MAX_PIXELS_PER_LINE        31:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_51                            0x00000033
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_WBOR3_51_R2                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_52                            0x00000034
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_52_REORDER_BANK_WIDTH_SIZE_MAX 13:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_52_R0                         31:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_53                            0x00000035
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_53_MAX_PIXELS5TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_53_R1                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_53_MAX_PIXELS5TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_53_R2                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_54                            0x00000036
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_54_MAX_PIXELS3TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_54_R3                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_54_MAX_PIXELS3TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_54_R4                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_55                            0x00000037
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_55_MAX_PIXELS2TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_55_R5                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_55_MAX_PIXELS2TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_55_R6                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_56                            0x00000038
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_56_PCLK_MAX                   7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_56_R7                         31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_57                            0x00000039
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_57_R8                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_58                            0x0000003A
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_58_R9                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_59                            0x0000003B
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD0_59_R10                        31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_60                            0x0000003C
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_60_REORDER_BANK_WIDTH_SIZE_MAX 13:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_60_R0                         31:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_61                            0x0000003D
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_61_MAX_PIXELS5TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_61_R1                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_61_MAX_PIXELS5TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_61_R2                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_62                            0x0000003E
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_62_MAX_PIXELS3TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_62_R3                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_62_MAX_PIXELS3TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_62_R4                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_63                            0x0000003F
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_63_MAX_PIXELS2TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_63_R5                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_63_MAX_PIXELS2TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_63_R6                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_64                            0x00000040
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_64_PCLK_MAX                   7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_64_R7                         31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_65                            0x00000041
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_65_R8                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_66                            0x00000042
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_66_R9                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_67                            0x00000043
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD1_67_R10                        31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_68                            0x00000044
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_68_REORDER_BANK_WIDTH_SIZE_MAX 13:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_68_R0                         31:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_69                            0x00000045
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_69_MAX_PIXELS5TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_69_R1                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_69_MAX_PIXELS5TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_69_R2                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_70                            0x00000046
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_70_MAX_PIXELS3TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_70_R3                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_70_MAX_PIXELS3TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_70_R4                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_71                            0x00000047
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_71_MAX_PIXELS2TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_71_R5                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_71_MAX_PIXELS2TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_71_R6                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_72                            0x00000048
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_72_PCLK_MAX                   7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_72_R7                         31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_73                            0x00000049
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_73_R8                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_74                            0x0000004A
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_74_R9                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_75                            0x0000004B
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD2_75_R10                        31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_76                            0x0000004C
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_76_REORDER_BANK_WIDTH_SIZE_MAX 13:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_76_R0                         31:14
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_77                            0x0000004D
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_77_MAX_PIXELS5TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_77_R1                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_77_MAX_PIXELS5TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_77_R2                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_78                            0x0000004E
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_78_MAX_PIXELS3TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_78_R3                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_78_MAX_PIXELS3TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_78_R4                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_79                            0x0000004F
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_79_MAX_PIXELS2TAP444          14:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_79_R5                         15:15
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_79_MAX_PIXELS2TAP422          30:16
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_79_R6                         31:31
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_80                            0x00000050
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_80_PCLK_MAX                   7:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_80_R7                         31:8
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_81                            0x00000051
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_81_R8                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_82                            0x00000052
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_82_R9                         31:0
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_83                            0x00000053
#define LW977D_CORE_NOTIFIER_3_CAPABILITIES_CAP_HEAD3_83_R10                        31:0


// dma opcode instructions
#define LW977D_DMA                                                         0x00000000 
#define LW977D_DMA_OPCODE                                                       31:29 
#define LW977D_DMA_OPCODE_METHOD                                           0x00000000 
#define LW977D_DMA_OPCODE_JUMP                                             0x00000001 
#define LW977D_DMA_OPCODE_NONINC_METHOD                                    0x00000002 
#define LW977D_DMA_OPCODE_SET_SUBDEVICE_MASK                               0x00000003 
#define LW977D_DMA_METHOD_COUNT                                                 27:18 
#define LW977D_DMA_METHOD_OFFSET                                                 11:2 
#define LW977D_DMA_DATA                                                          31:0 
#define LW977D_DMA_DATA_NOP                                                0x00000000 
#define LW977D_DMA_JUMP_OFFSET                                                   11:2 
#define LW977D_DMA_SET_SUBDEVICE_MASK_VALUE                                      11:0 

// class methods
#define LW977D_PUT                                                              (0x00000000)
#define LW977D_PUT_PTR                                                          11:2
#define LW977D_GET                                                              (0x00000004)
#define LW977D_GET_PTR                                                          11:2
#define LW977D_UPDATE                                                           (0x00000080)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR(i)                                  (0 +(i)*4):(0 +(i)*4)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR__SIZE_1                             4
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR_DISABLE                             (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR_ENABLE                              (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR0                                    0:0
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR0_DISABLE                            (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR0_ENABLE                             (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR1                                    4:4
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR1_DISABLE                            (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR1_ENABLE                             (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR2                                    8:8
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR2_DISABLE                            (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR2_ENABLE                             (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR3                                    12:12
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR3_DISABLE                            (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_LWRSOR3_ENABLE                             (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE(i)                                    (1 +(i)*4):(1 +(i)*4)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE__SIZE_1                               4
#define LW977D_UPDATE_INTERLOCK_WITH_BASE_DISABLE                               (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE_ENABLE                                (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE0                                      1:1
#define LW977D_UPDATE_INTERLOCK_WITH_BASE0_DISABLE                              (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE0_ENABLE                               (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE1                                      5:5
#define LW977D_UPDATE_INTERLOCK_WITH_BASE1_DISABLE                              (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE1_ENABLE                               (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE2                                      9:9
#define LW977D_UPDATE_INTERLOCK_WITH_BASE2_DISABLE                              (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE2_ENABLE                               (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE3                                      13:13
#define LW977D_UPDATE_INTERLOCK_WITH_BASE3_DISABLE                              (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_BASE3_ENABLE                               (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY(i)                                 (2 +(i)*4):(2 +(i)*4)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY__SIZE_1                            4
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_DISABLE                            (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_ENABLE                             (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY0                                   2:2
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY0_DISABLE                           (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY0_ENABLE                            (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY1                                   6:6
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY1_DISABLE                           (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY1_ENABLE                            (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY2                                   10:10
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY2_DISABLE                           (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY2_ENABLE                            (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY3                                   14:14
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY3_DISABLE                           (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY3_ENABLE                            (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM(i)                             (3 +(i)*4):(3 +(i)*4)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM__SIZE_1                        4
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM_DISABLE                        (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM_ENABLE                         (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM0                               3:3
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM0_DISABLE                       (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM0_ENABLE                        (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM1                               7:7
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM1_DISABLE                       (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM1_ENABLE                        (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM2                               11:11
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM2_DISABLE                       (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM2_ENABLE                        (0x00000001)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM3                               15:15
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM3_DISABLE                       (0x00000000)
#define LW977D_UPDATE_INTERLOCK_WITH_OVERLAY_IMM3_ENABLE                        (0x00000001)
#define LW977D_UPDATE_SPECIAL_HANDLING                                          25:24
#define LW977D_UPDATE_SPECIAL_HANDLING_NONE                                     (0x00000000)
#define LW977D_UPDATE_SPECIAL_HANDLING_INTERRUPT_RM                             (0x00000001)
#define LW977D_UPDATE_SPECIAL_HANDLING_MODE_SWITCH                              (0x00000002)
#define LW977D_UPDATE_SPECIAL_HANDLING_REASON                                   23:16
#define LW977D_UPDATE_NOT_DRIVER_FRIENDLY                                       31:31
#define LW977D_UPDATE_NOT_DRIVER_FRIENDLY_FALSE                                 (0x00000000)
#define LW977D_UPDATE_NOT_DRIVER_FRIENDLY_TRUE                                  (0x00000001)
#define LW977D_UPDATE_NOT_DRIVER_UNFRIENDLY                                     30:30
#define LW977D_UPDATE_NOT_DRIVER_UNFRIENDLY_FALSE                               (0x00000000)
#define LW977D_UPDATE_NOT_DRIVER_UNFRIENDLY_TRUE                                (0x00000001)
#define LW977D_UPDATE_INHIBIT_INTERRUPTS                                        29:29
#define LW977D_UPDATE_INHIBIT_INTERRUPTS_FALSE                                  (0x00000000)
#define LW977D_UPDATE_INHIBIT_INTERRUPTS_TRUE                                   (0x00000001)
#define LW977D_SET_NOTIFIER_CONTROL                                             (0x00000084)
#define LW977D_SET_NOTIFIER_CONTROL_MODE                                        30:30
#define LW977D_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LW977D_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LW977D_SET_NOTIFIER_CONTROL_OFFSET                                      11:2
#define LW977D_SET_NOTIFIER_CONTROL_NOTIFY                                      31:31
#define LW977D_SET_NOTIFIER_CONTROL_NOTIFY_DISABLE                              (0x00000000)
#define LW977D_SET_NOTIFIER_CONTROL_NOTIFY_ENABLE                               (0x00000001)
#define LW977D_SET_NOTIFIER_CONTROL_FORMAT                                      28:28
#define LW977D_SET_NOTIFIER_CONTROL_FORMAT_LEGACY                               (0x00000000)
#define LW977D_SET_NOTIFIER_CONTROL_FORMAT_FOUR_WORD                            (0x00000001)
#define LW977D_SET_CONTEXT_DMA_NOTIFIER                                         (0x00000088)
#define LW977D_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LW977D_GET_CAPABILITIES                                                 (0x0000008C)
#define LW977D_GET_CAPABILITIES_DUMMY                                           31:0
#define LW977D_SET_SPARE                                                        (0x0000016C)
#define LW977D_SET_SPARE_UNUSED                                                 31:0
#define LW977D_SET_SPARE_NOOP(b)                                                (0x00000170 + (b)*0x00000004)
#define LW977D_SET_SPARE_NOOP_UNUSED                                            31:0

#define LW977D_DAC_SET_CONTROL(a)                                               (0x00000180 + (a)*0x00000020)
#define LW977D_DAC_SET_CONTROL_OWNER_MASK                                       3:0
#define LW977D_DAC_SET_CONTROL_OWNER_MASK_NONE                                  (0x00000000)
#define LW977D_DAC_SET_CONTROL_OWNER_MASK_HEAD0                                 (0x00000001)
#define LW977D_DAC_SET_CONTROL_OWNER_MASK_HEAD1                                 (0x00000002)
#define LW977D_DAC_SET_CONTROL_OWNER_MASK_HEAD2                                 (0x00000004)
#define LW977D_DAC_SET_CONTROL_OWNER_MASK_HEAD3                                 (0x00000008)
#define LW977D_DAC_SET_CONTROL_PROTOCOL                                         12:8
#define LW977D_DAC_SET_CONTROL_PROTOCOL_RGB_CRT                                 (0x00000000)
#define LW977D_DAC_SET_CONTROL_PROTOCOL_YUV_CRT                                 (0x00000013)
#define LW977D_DAC_SET_SW_SPARE_A(a)                                            (0x00000184 + (a)*0x00000020)
#define LW977D_DAC_SET_SW_SPARE_A_CODE                                          31:0
#define LW977D_DAC_SET_SW_SPARE_B(a)                                            (0x00000188 + (a)*0x00000020)
#define LW977D_DAC_SET_SW_SPARE_B_CODE                                          31:0
#define LW977D_DAC_SET_LWSTOM_REASON(a)                                         (0x00000190 + (a)*0x00000020)
#define LW977D_DAC_SET_LWSTOM_REASON_CODE                                       31:0

#define LW977D_SOR_SET_CONTROL(a)                                               (0x00000200 + (a)*0x00000020)
#define LW977D_SOR_SET_CONTROL_OWNER_MASK                                       3:0
#define LW977D_SOR_SET_CONTROL_OWNER_MASK_NONE                                  (0x00000000)
#define LW977D_SOR_SET_CONTROL_OWNER_MASK_HEAD0                                 (0x00000001)
#define LW977D_SOR_SET_CONTROL_OWNER_MASK_HEAD1                                 (0x00000002)
#define LW977D_SOR_SET_CONTROL_OWNER_MASK_HEAD2                                 (0x00000004)
#define LW977D_SOR_SET_CONTROL_OWNER_MASK_HEAD3                                 (0x00000008)
#define LW977D_SOR_SET_CONTROL_PROTOCOL                                         11:8
#define LW977D_SOR_SET_CONTROL_PROTOCOL_LVDS_LWSTOM                             (0x00000000)
#define LW977D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_A                           (0x00000001)
#define LW977D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_B                           (0x00000002)
#define LW977D_SOR_SET_CONTROL_PROTOCOL_DUAL_TMDS                               (0x00000005)
#define LW977D_SOR_SET_CONTROL_PROTOCOL_DP_A                                    (0x00000008)
#define LW977D_SOR_SET_CONTROL_PROTOCOL_DP_B                                    (0x00000009)
#define LW977D_SOR_SET_CONTROL_PROTOCOL_LWSTOM                                  (0x0000000F)
#define LW977D_SOR_SET_CONTROL_DE_SYNC_POLARITY                                 14:14
#define LW977D_SOR_SET_CONTROL_DE_SYNC_POLARITY_POSITIVE_TRUE                   (0x00000000)
#define LW977D_SOR_SET_CONTROL_DE_SYNC_POLARITY_NEGATIVE_TRUE                   (0x00000001)
#define LW977D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE                             21:20
#define LW977D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE_OFF                         (0x00000000)
#define LW977D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE_X2                          (0x00000001)
#define LW977D_SOR_SET_CONTROL_PIXEL_REPLICATE_MODE_X4                          (0x00000002)
#define LW977D_SOR_SET_SW_SPARE_A(a)                                            (0x00000204 + (a)*0x00000020)
#define LW977D_SOR_SET_SW_SPARE_A_CODE                                          31:0
#define LW977D_SOR_SET_SW_SPARE_B(a)                                            (0x00000208 + (a)*0x00000020)
#define LW977D_SOR_SET_SW_SPARE_B_CODE                                          31:0
#define LW977D_SOR_SET_LWSTOM_REASON(a)                                         (0x00000210 + (a)*0x00000020)
#define LW977D_SOR_SET_LWSTOM_REASON_CODE                                       31:0

#define LW977D_PIOR_SET_CONTROL(a)                                              (0x00000300 + (a)*0x00000020)
#define LW977D_PIOR_SET_CONTROL_OWNER_MASK                                      3:0
#define LW977D_PIOR_SET_CONTROL_OWNER_MASK_NONE                                 (0x00000000)
#define LW977D_PIOR_SET_CONTROL_OWNER_MASK_HEAD0                                (0x00000001)
#define LW977D_PIOR_SET_CONTROL_OWNER_MASK_HEAD1                                (0x00000002)
#define LW977D_PIOR_SET_CONTROL_OWNER_MASK_HEAD2                                (0x00000004)
#define LW977D_PIOR_SET_CONTROL_OWNER_MASK_HEAD3                                (0x00000008)
#define LW977D_PIOR_SET_CONTROL_PROTOCOL                                        11:8
#define LW977D_PIOR_SET_CONTROL_PROTOCOL_EXT_TMDS_ENC                           (0x00000000)
#define LW977D_PIOR_SET_CONTROL_PROTOCOL_EXT_TV_ENC                             (0x00000001)
#define LW977D_PIOR_SET_CONTROL_PROTOCOL_EXT_SDI_SD_ENC                         (0x00000002)
#define LW977D_PIOR_SET_CONTROL_PROTOCOL_EXT_SDI_HD_ENC                         (0x00000003)
#define LW977D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_OUT                        (0x00000004)
#define LW977D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_IN                         (0x00000005)
#define LW977D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_INOUT                      (0x00000006)
#define LW977D_PIOR_SET_CONTROL_DE_SYNC_POLARITY                                14:14
#define LW977D_PIOR_SET_CONTROL_DE_SYNC_POLARITY_POSITIVE_TRUE                  (0x00000000)
#define LW977D_PIOR_SET_CONTROL_DE_SYNC_POLARITY_NEGATIVE_TRUE                  (0x00000001)
#define LW977D_PIOR_SET_SW_SPARE_A(a)                                           (0x00000304 + (a)*0x00000020)
#define LW977D_PIOR_SET_SW_SPARE_A_CODE                                         31:0
#define LW977D_PIOR_SET_SW_SPARE_B(a)                                           (0x00000308 + (a)*0x00000020)
#define LW977D_PIOR_SET_SW_SPARE_B_CODE                                         31:0
#define LW977D_PIOR_SET_LWSTOM_REASON(a)                                        (0x00000310 + (a)*0x00000020)
#define LW977D_PIOR_SET_LWSTOM_REASON_CODE                                      31:0

#define LW977D_HEAD_SET_PRESENT_CONTROL(a)                                      (0x00000400 + (a)*0x00000300)
#define LW977D_HEAD_SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL                    3:0
#define LW977D_HEAD_SET_PRESENT_CONTROL_USE_BEGIN_FIELD                         8:8
#define LW977D_HEAD_SET_PRESENT_CONTROL_USE_BEGIN_FIELD_DISABLE                 (0x00000000)
#define LW977D_HEAD_SET_PRESENT_CONTROL_USE_BEGIN_FIELD_ENABLE                  (0x00000001)
#define LW977D_HEAD_SET_PRESENT_CONTROL_BEGIN_FIELD                             6:4
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE(a)                              (0x00000404 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE                        1:0
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE_ACTIVE_RASTER          (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE_COMPLETE_RASTER        (0x00000001)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_CRC_MODE_NON_ACTIVE_RASTER      (0x00000002)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_HSYNC_POLARITY                  3:3
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_HSYNC_POLARITY_POSITIVE_TRUE    (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_HSYNC_POLARITY_NEGATIVE_TRUE    (0x00000001)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_VSYNC_POLARITY                  4:4
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_VSYNC_POLARITY_POSITIVE_TRUE    (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_VSYNC_POLARITY_NEGATIVE_TRUE    (0x00000001)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH                     9:6
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_DEFAULT             (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_16_422          (0x00000001)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_18_444          (0x00000002)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_20_422          (0x00000003)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_422          (0x00000004)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_444          (0x00000005)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_30_444          (0x00000006)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_32_422          (0x00000007)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_36_444          (0x00000008)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_48_444          (0x00000009)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_OVERRIDE            12:12
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_OVERRIDE_DISABLE    (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_OVERRIDE_ENABLE     (0x00000001)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE_COLOR_SPACE_FLAG                24:13
#define LW977D_HEAD_SET_CONTROL(a)                                              (0x00000408 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTROL_STRUCTURE                                       0:0
#define LW977D_HEAD_SET_CONTROL_STRUCTURE_PROGRESSIVE                           (0x00000000)
#define LW977D_HEAD_SET_CONTROL_STRUCTURE_INTERLACED                            (0x00000001)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE                                 3:2
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_NO_LOCK                         (0x00000000)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_FRAME_LOCK                      (0x00000001)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_RASTER_LOCK                     (0x00000003)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN                                  19:15
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN(i)                      (0x00000000 +(i))
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN__SIZE_1                 16
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_0                       (0x00000000)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_1                       (0x00000001)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_2                       (0x00000002)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_3                       (0x00000003)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_4                       (0x00000004)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_5                       (0x00000005)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_6                       (0x00000006)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_7                       (0x00000007)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_8                       (0x00000008)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_9                       (0x00000009)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_A                       (0x0000000A)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_B                       (0x0000000B)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_C                       (0x0000000C)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_D                       (0x0000000D)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_E                       (0x0000000E)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_F                       (0x0000000F)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_UNSPECIFIED                      (0x00000010)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK(i)            (0x00000018 +(i))
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1       4
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_0             (0x00000018)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_1             (0x00000019)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_2             (0x0000001A)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_SCAN_LOCK_3             (0x0000001B)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK(i)            (0x0000001E +(i))
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK__SIZE_1       2
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK_0             (0x0000001E)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_INTERNAL_FLIP_LOCK_1             (0x0000001F)
#define LW977D_HEAD_SET_CONTROL_SLAVE_LOCKOUT_WINDOW                            7:4
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_MODE                                9:8
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_MODE_NO_LOCK                        (0x00000000)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_MODE_FRAME_LOCK                     (0x00000001)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_MODE_RASTER_LOCK                    (0x00000003)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN                                 14:10
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN(i)                     (0x00000000 +(i))
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN__SIZE_1                16
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_0                      (0x00000000)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_1                      (0x00000001)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_2                      (0x00000002)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_3                      (0x00000003)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_4                      (0x00000004)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_5                      (0x00000005)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_6                      (0x00000006)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_7                      (0x00000007)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_8                      (0x00000008)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_9                      (0x00000009)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_A                      (0x0000000A)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_B                      (0x0000000B)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_C                      (0x0000000C)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_D                      (0x0000000D)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_E                      (0x0000000E)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_LOCK_PIN_F                      (0x0000000F)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_UNSPECIFIED                     (0x00000010)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK(i)           (0x00000018 +(i))
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1      4
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_0            (0x00000018)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_1            (0x00000019)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_2            (0x0000001A)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_SCAN_LOCK_3            (0x0000001B)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK(i)           (0x0000001E +(i))
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK__SIZE_1      2
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK_0            (0x0000001E)
#define LW977D_HEAD_SET_CONTROL_MASTER_LOCK_PIN_INTERNAL_FLIP_LOCK_1            (0x0000001F)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK                                       1:1
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_DISABLE                               (0x00000000)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_ENABLE                                (0x00000001)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN                                   24:20
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN(i)                       (0x00000000 +(i))
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN__SIZE_1                  16
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_0                        (0x00000000)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_1                        (0x00000001)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_2                        (0x00000002)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_3                        (0x00000003)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_4                        (0x00000004)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_5                        (0x00000005)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_6                        (0x00000006)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_7                        (0x00000007)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_8                        (0x00000008)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_9                        (0x00000009)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_A                        (0x0000000A)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_B                        (0x0000000B)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_C                        (0x0000000C)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_D                        (0x0000000D)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_E                        (0x0000000E)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_LOCK_PIN_F                        (0x0000000F)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_UNSPECIFIED                       (0x00000010)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK(i)             (0x00000018 +(i))
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1        4
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_0              (0x00000018)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_1              (0x00000019)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_2              (0x0000001A)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_3              (0x0000001B)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK(i)             (0x0000001E +(i))
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK__SIZE_1        2
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_0              (0x0000001E)
#define LW977D_HEAD_SET_CONTROL_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_1              (0x0000001F)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN                                      29:25
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN(i)                          (0x00000000 +(i))
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN__SIZE_1                     16
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_0                           (0x00000000)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_1                           (0x00000001)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_2                           (0x00000002)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_3                           (0x00000003)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_4                           (0x00000004)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_5                           (0x00000005)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_6                           (0x00000006)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_7                           (0x00000007)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_8                           (0x00000008)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_9                           (0x00000009)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_A                           (0x0000000A)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_B                           (0x0000000B)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_C                           (0x0000000C)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_D                           (0x0000000D)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_E                           (0x0000000E)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_LOCK_PIN_F                           (0x0000000F)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_UNSPECIFIED                          (0x00000010)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK(i)                (0x00000018 +(i))
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK__SIZE_1           4
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_0                 (0x00000018)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_1                 (0x00000019)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_2                 (0x0000001A)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_SCAN_LOCK_3                 (0x0000001B)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK(i)                (0x0000001E +(i))
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK__SIZE_1           2
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK_0                 (0x0000001E)
#define LW977D_HEAD_SET_CONTROL_STEREO_PIN_INTERNAL_FLIP_LOCK_1                 (0x0000001F)
#define LW977D_HEAD_SET_CONTROL_SLAVE_STEREO_LOCK_MODE                          30:30
#define LW977D_HEAD_SET_CONTROL_SLAVE_STEREO_LOCK_MODE_DISABLE                  (0x00000000)
#define LW977D_HEAD_SET_CONTROL_SLAVE_STEREO_LOCK_MODE_ENABLE                   (0x00000001)
#define LW977D_HEAD_SET_CONTROL_MASTER_STEREO_LOCK_MODE                         31:31
#define LW977D_HEAD_SET_CONTROL_MASTER_STEREO_LOCK_MODE_DISABLE                 (0x00000000)
#define LW977D_HEAD_SET_CONTROL_MASTER_STEREO_LOCK_MODE_ENABLE                  (0x00000001)
#define LW977D_HEAD_SET_LOCK_OFFSET(a)                                          (0x0000040C + (a)*0x00000300)
#define LW977D_HEAD_SET_LOCK_OFFSET_X                                           14:0
#define LW977D_HEAD_SET_LOCK_OFFSET_Y                                           30:16
#define LW977D_HEAD_SET_OVERSCAN_COLOR(a)                                       (0x00000410 + (a)*0x00000300)
#define LW977D_HEAD_SET_OVERSCAN_COLOR_RED                                      9:0
#define LW977D_HEAD_SET_OVERSCAN_COLOR_GRN                                      19:10
#define LW977D_HEAD_SET_OVERSCAN_COLOR_BLU                                      29:20
#define LW977D_HEAD_SET_RASTER_SIZE(a)                                          (0x00000414 + (a)*0x00000300)
#define LW977D_HEAD_SET_RASTER_SIZE_WIDTH                                       14:0
#define LW977D_HEAD_SET_RASTER_SIZE_HEIGHT                                      30:16
#define LW977D_HEAD_SET_RASTER_SYNC_END(a)                                      (0x00000418 + (a)*0x00000300)
#define LW977D_HEAD_SET_RASTER_SYNC_END_X                                       14:0
#define LW977D_HEAD_SET_RASTER_SYNC_END_Y                                       30:16
#define LW977D_HEAD_SET_RASTER_BLANK_END(a)                                     (0x0000041C + (a)*0x00000300)
#define LW977D_HEAD_SET_RASTER_BLANK_END_X                                      14:0
#define LW977D_HEAD_SET_RASTER_BLANK_END_Y                                      30:16
#define LW977D_HEAD_SET_RASTER_BLANK_START(a)                                   (0x00000420 + (a)*0x00000300)
#define LW977D_HEAD_SET_RASTER_BLANK_START_X                                    14:0
#define LW977D_HEAD_SET_RASTER_BLANK_START_Y                                    30:16
#define LW977D_HEAD_SET_RASTER_VERT_BLANK2(a)                                   (0x00000424 + (a)*0x00000300)
#define LW977D_HEAD_SET_RASTER_VERT_BLANK2_YSTART                               14:0
#define LW977D_HEAD_SET_RASTER_VERT_BLANK2_YEND                                 30:16
#define LW977D_HEAD_SET_LOCK_CHAIN(a)                                           (0x00000428 + (a)*0x00000300)
#define LW977D_HEAD_SET_LOCK_CHAIN_POSITION                                     27:24
#define LW977D_HEAD_SET_DEFAULT_BASE_COLOR(a)                                   (0x0000042C + (a)*0x00000300)
#define LW977D_HEAD_SET_DEFAULT_BASE_COLOR_RED                                  9:0
#define LW977D_HEAD_SET_DEFAULT_BASE_COLOR_GREEN                                19:10
#define LW977D_HEAD_SET_DEFAULT_BASE_COLOR_BLUE                                 29:20
#define LW977D_HEAD_SET_CRC_CONTROL(a)                                          (0x00000430 + (a)*0x00000300)
#define LW977D_HEAD_SET_CRC_CONTROL_CONTROLLING_CHANNEL                         1:0
#define LW977D_HEAD_SET_CRC_CONTROL_CONTROLLING_CHANNEL_CORE                    (0x00000000)
#define LW977D_HEAD_SET_CRC_CONTROL_CONTROLLING_CHANNEL_BASE                    (0x00000001)
#define LW977D_HEAD_SET_CRC_CONTROL_CONTROLLING_CHANNEL_OVERLAY                 (0x00000002)
#define LW977D_HEAD_SET_CRC_CONTROL_EXPECT_BUFFER_COLLAPSE                      2:2
#define LW977D_HEAD_SET_CRC_CONTROL_EXPECT_BUFFER_COLLAPSE_FALSE                (0x00000000)
#define LW977D_HEAD_SET_CRC_CONTROL_EXPECT_BUFFER_COLLAPSE_TRUE                 (0x00000001)
#define LW977D_HEAD_SET_CRC_CONTROL_TIMESTAMP_MODE                              3:3
#define LW977D_HEAD_SET_CRC_CONTROL_TIMESTAMP_MODE_FALSE                        (0x00000000)
#define LW977D_HEAD_SET_CRC_CONTROL_TIMESTAMP_MODE_TRUE                         (0x00000001)
#define LW977D_HEAD_SET_CRC_CONTROL_FLIPLOCK_MODE                               4:4
#define LW977D_HEAD_SET_CRC_CONTROL_FLIPLOCK_MODE_FALSE                         (0x00000000)
#define LW977D_HEAD_SET_CRC_CONTROL_FLIPLOCK_MODE_TRUE                          (0x00000001)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT                              19:8
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_DAC(i)                       (0x00000FF0 +(i))
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_DAC__SIZE_1                  4
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_DAC0                         (0x00000FF0)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_DAC1                         (0x00000FF1)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_DAC2                         (0x00000FF2)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_DAC3                         (0x00000FF3)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_RG(i)                        (0x00000FF8 +(i))
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_RG__SIZE_1                   4
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_RG0                          (0x00000FF8)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_RG1                          (0x00000FF9)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_RG2                          (0x00000FFA)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_RG3                          (0x00000FFB)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR(i)                       (0x00000F0F +(i)*16)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR__SIZE_1                  8
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR0                         (0x00000F0F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR1                         (0x00000F1F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR2                         (0x00000F2F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR3                         (0x00000F3F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR4                         (0x00000F4F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR5                         (0x00000F5F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR6                         (0x00000F6F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SOR7                         (0x00000F7F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SF(i)                        (0x00000F8F +(i)*16)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SF__SIZE_1                   4
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SF0                          (0x00000F8F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SF1                          (0x00000F9F)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SF2                          (0x00000FAF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_SF3                          (0x00000FBF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR(i)                      (0x000000FF +(i)*256)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR__SIZE_1                 8
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR0                        (0x000000FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR1                        (0x000001FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR2                        (0x000002FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR3                        (0x000003FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR4                        (0x000004FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR5                        (0x000005FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR6                        (0x000006FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_PIOR7                        (0x000007FF)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_WBOR0                        (0x00000FC0)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_WBOR1                        (0x00000FC1)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_WBOR2                        (0x00000FC2)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_WBOR3                        (0x00000FC3)
#define LW977D_HEAD_SET_CRC_CONTROL_PRIMARY_OUTPUT_NONE                         (0x00000FFF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT                            31:20
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_DAC(i)                     (0x00000FF0 +(i))
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_DAC__SIZE_1                4
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_DAC0                       (0x00000FF0)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_DAC1                       (0x00000FF1)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_DAC2                       (0x00000FF2)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_DAC3                       (0x00000FF3)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_RG(i)                      (0x00000FF8 +(i))
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_RG__SIZE_1                 4
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_RG0                        (0x00000FF8)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_RG1                        (0x00000FF9)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_RG2                        (0x00000FFA)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_RG3                        (0x00000FFB)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR(i)                     (0x00000F0F +(i)*16)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR__SIZE_1                8
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR0                       (0x00000F0F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR1                       (0x00000F1F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR2                       (0x00000F2F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR3                       (0x00000F3F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR4                       (0x00000F4F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR5                       (0x00000F5F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR6                       (0x00000F6F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SOR7                       (0x00000F7F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SF(i)                      (0x00000F8F +(i)*16)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SF__SIZE_1                 4
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SF0                        (0x00000F8F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SF1                        (0x00000F9F)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SF2                        (0x00000FAF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_SF3                        (0x00000FBF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR(i)                    (0x000000FF +(i)*256)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR__SIZE_1               8
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR0                      (0x000000FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR1                      (0x000001FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR2                      (0x000002FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR3                      (0x000003FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR4                      (0x000004FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR5                      (0x000005FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR6                      (0x000006FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_PIOR7                      (0x000007FF)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_WBOR0                      (0x00000FC0)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_WBOR1                      (0x00000FC1)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_WBOR2                      (0x00000FC2)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_WBOR3                      (0x00000FC3)
#define LW977D_HEAD_SET_CRC_CONTROL_SECONDARY_OUTPUT_NONE                       (0x00000FFF)
#define LW977D_HEAD_SET_CRC_CONTROL_CRC_DURING_SNOOZE                           5:5
#define LW977D_HEAD_SET_CRC_CONTROL_CRC_DURING_SNOOZE_DISABLE                   (0x00000000)
#define LW977D_HEAD_SET_CRC_CONTROL_CRC_DURING_SNOOZE_ENABLE                    (0x00000001)
#define LW977D_HEAD_SET_CRC_CONTROL_WIDE_PIPE_CRC                               6:6
#define LW977D_HEAD_SET_CRC_CONTROL_WIDE_PIPE_CRC_DISABLE                       (0x00000000)
#define LW977D_HEAD_SET_CRC_CONTROL_WIDE_PIPE_CRC_ENABLE                        (0x00000001)
#define LW977D_HEAD_SET_LEGACY_CRC_CONTROL(a)                                   (0x00000434 + (a)*0x00000300)
#define LW977D_HEAD_SET_LEGACY_CRC_CONTROL_COMPUTE                              0:0
#define LW977D_HEAD_SET_LEGACY_CRC_CONTROL_COMPUTE_DISABLE                      (0x00000000)
#define LW977D_HEAD_SET_LEGACY_CRC_CONTROL_COMPUTE_ENABLE                       (0x00000001)
#define LW977D_HEAD_SET_CONTEXT_DMA_CRC(a)                                      (0x00000438 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTEXT_DMA_CRC_HANDLE                                  31:0
#define LW977D_HEAD_SET_BASE_LUT_LO(a)                                          (0x00000440 + (a)*0x00000300)
#define LW977D_HEAD_SET_BASE_LUT_LO_ENABLE                                      31:31
#define LW977D_HEAD_SET_BASE_LUT_LO_ENABLE_DISABLE                              (0x00000000)
#define LW977D_HEAD_SET_BASE_LUT_LO_ENABLE_ENABLE                               (0x00000001)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE                                        27:24
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_LORES                                  (0x00000000)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_HIRES                                  (0x00000001)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_INDEX_1025_UNITY_RANGE                 (0x00000003)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_INTERPOLATE_1025_UNITY_RANGE           (0x00000004)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_INTERPOLATE_1025_XRBIAS_RANGE          (0x00000005)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_INTERPOLATE_1025_XVYCC_RANGE           (0x00000006)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_INTERPOLATE_257_UNITY_RANGE            (0x00000007)
#define LW977D_HEAD_SET_BASE_LUT_LO_MODE_INTERPOLATE_257_LEGACY_RANGE           (0x00000008)
#define LW977D_HEAD_SET_BASE_LUT_LO_NEVER_YIELD_TO_BASE                         20:20
#define LW977D_HEAD_SET_BASE_LUT_LO_NEVER_YIELD_TO_BASE_DISABLE                 (0x00000000)
#define LW977D_HEAD_SET_BASE_LUT_LO_NEVER_YIELD_TO_BASE_ENABLE                  (0x00000001)
#define LW977D_HEAD_SET_BASE_LUT_HI(a)                                          (0x00000444 + (a)*0x00000300)
#define LW977D_HEAD_SET_BASE_LUT_HI_ORIGIN                                      31:0
#define LW977D_HEAD_SET_OUTPUT_LUT_LO(a)                                        (0x00000448 + (a)*0x00000300)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_ENABLE                                    31:31
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_ENABLE_DISABLE                            (0x00000000)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_ENABLE_ENABLE                             (0x00000001)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE                                      27:24
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_LORES                                (0x00000000)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_HIRES                                (0x00000001)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_INDEX_1025_UNITY_RANGE               (0x00000003)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_UNITY_RANGE         (0x00000004)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_XRBIAS_RANGE        (0x00000005)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_XVYCC_RANGE         (0x00000006)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_257_UNITY_RANGE          (0x00000007)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_MODE_INTERPOLATE_257_LEGACY_RANGE         (0x00000008)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_NEVER_YIELD_TO_BASE                       20:20
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_NEVER_YIELD_TO_BASE_DISABLE               (0x00000000)
#define LW977D_HEAD_SET_OUTPUT_LUT_LO_NEVER_YIELD_TO_BASE_ENABLE                (0x00000001)
#define LW977D_HEAD_SET_OUTPUT_LUT_HI(a)                                        (0x0000044C + (a)*0x00000300)
#define LW977D_HEAD_SET_OUTPUT_LUT_HI_ORIGIN                                    31:0
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY(a)                                (0x00000450 + (a)*0x00000300)
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_HERTZ                             30:0
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_ADJ1000DIV1001                    31:31
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_ADJ1000DIV1001_FALSE              (0x00000000)
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_ADJ1000DIV1001_TRUE               (0x00000001)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION(a)                            (0x00000454 + (a)*0x00000300)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_MODE                          21:20
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_MODE_CLK_25                   (0x00000000)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_MODE_CLK_28                   (0x00000001)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_MODE_CLK_LWSTOM               (0x00000002)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_NOT_DRIVER                    24:24
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_NOT_DRIVER_FALSE              (0x00000000)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_NOT_DRIVER_TRUE               (0x00000001)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_ENABLE_HOPPING                25:25
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_ENABLE_HOPPING_FALSE          (0x00000000)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_ENABLE_HOPPING_TRUE           (0x00000001)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_MODE                  26:26
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_MODE_VBLANK           (0x00000000)
#define LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION_HOPPING_MODE_HBLANK           (0x00000001)
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX(a)                            (0x00000458 + (a)*0x00000300)
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_HERTZ                         30:0
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_ADJ1000DIV1001                31:31
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_ADJ1000DIV1001_FALSE          (0x00000000)
#define LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX_ADJ1000DIV1001_TRUE           (0x00000001)
#define LW977D_HEAD_SET_CONTEXT_DMA_LUT(a)                                      (0x0000045C + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTEXT_DMA_LUT_HANDLE                                  31:0
#define LW977D_HEAD_SET_OFFSET(a)                                               (0x00000460 + (a)*0x00000300)
#define LW977D_HEAD_SET_OFFSET_ORIGIN                                           31:0
#define LW977D_HEAD_SET_SIZE(a)                                                 (0x00000468 + (a)*0x00000300)
#define LW977D_HEAD_SET_SIZE_WIDTH                                              15:0
#define LW977D_HEAD_SET_SIZE_HEIGHT                                             31:16
#define LW977D_HEAD_SET_STORAGE(a)                                              (0x0000046C + (a)*0x00000300)
#define LW977D_HEAD_SET_STORAGE_BLOCK_HEIGHT                                    3:0
#define LW977D_HEAD_SET_STORAGE_BLOCK_HEIGHT_ONE_GOB                            (0x00000000)
#define LW977D_HEAD_SET_STORAGE_BLOCK_HEIGHT_TWO_GOBS                           (0x00000001)
#define LW977D_HEAD_SET_STORAGE_BLOCK_HEIGHT_FOUR_GOBS                          (0x00000002)
#define LW977D_HEAD_SET_STORAGE_BLOCK_HEIGHT_EIGHT_GOBS                         (0x00000003)
#define LW977D_HEAD_SET_STORAGE_BLOCK_HEIGHT_SIXTEEN_GOBS                       (0x00000004)
#define LW977D_HEAD_SET_STORAGE_BLOCK_HEIGHT_THIRTYTWO_GOBS                     (0x00000005)
#define LW977D_HEAD_SET_STORAGE_PITCH                                           20:8
#define LW977D_HEAD_SET_STORAGE_MEMORY_LAYOUT                                   24:24
#define LW977D_HEAD_SET_STORAGE_MEMORY_LAYOUT_BLOCKLINEAR                       (0x00000000)
#define LW977D_HEAD_SET_STORAGE_MEMORY_LAYOUT_PITCH                             (0x00000001)
#define LW977D_HEAD_SET_PARAMS(a)                                               (0x00000470 + (a)*0x00000300)
#define LW977D_HEAD_SET_PARAMS_FORMAT                                           15:8
#define LW977D_HEAD_SET_PARAMS_FORMAT_I8                                        (0x0000001E)
#define LW977D_HEAD_SET_PARAMS_FORMAT_VOID16                                    (0x0000001F)
#define LW977D_HEAD_SET_PARAMS_FORMAT_VOID32                                    (0x0000002E)
#define LW977D_HEAD_SET_PARAMS_FORMAT_RF16_GF16_BF16_AF16                       (0x000000CA)
#define LW977D_HEAD_SET_PARAMS_FORMAT_A8R8G8B8                                  (0x000000CF)
#define LW977D_HEAD_SET_PARAMS_FORMAT_A2B10G10R10                               (0x000000D1)
#define LW977D_HEAD_SET_PARAMS_FORMAT_X2BL10GL10RL10_XRBIAS                     (0x00000022)
#define LW977D_HEAD_SET_PARAMS_FORMAT_X2BL10GL10RL10_XVYCC                      (0x00000024)
#define LW977D_HEAD_SET_PARAMS_FORMAT_A8B8G8R8                                  (0x000000D5)
#define LW977D_HEAD_SET_PARAMS_FORMAT_R5G6B5                                    (0x000000E8)
#define LW977D_HEAD_SET_PARAMS_FORMAT_A1R5G5B5                                  (0x000000E9)
#define LW977D_HEAD_SET_PARAMS_FORMAT_R16_G16_B16_A16                           (0x000000C6)
#define LW977D_HEAD_SET_PARAMS_FORMAT_R16_G16_B16_A16_LWBIAS                    (0x00000023)
#define LW977D_HEAD_SET_PARAMS_FORMAT_A2R10G10B10                               (0x000000DF)
#define LW977D_HEAD_SET_PARAMS_SUPER_SAMPLE                                     1:0
#define LW977D_HEAD_SET_PARAMS_SUPER_SAMPLE_X1_AA                               (0x00000000)
#define LW977D_HEAD_SET_PARAMS_SUPER_SAMPLE_X4_AA                               (0x00000002)
#define LW977D_HEAD_SET_PARAMS_GAMMA                                            2:2
#define LW977D_HEAD_SET_PARAMS_GAMMA_LINEAR                                     (0x00000000)
#define LW977D_HEAD_SET_PARAMS_GAMMA_SRGB                                       (0x00000001)
#define LW977D_HEAD_SET_CONTEXT_DMAS_ISO(a)                                     (0x00000474 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTEXT_DMAS_ISO_HANDLE                                 31:0
#define LW977D_HEAD_SET_PRESENT_CONTROL_LWRSOR(a)                               (0x0000047C + (a)*0x00000300)
#define LW977D_HEAD_SET_PRESENT_CONTROL_LWRSOR_MODE                             1:0
#define LW977D_HEAD_SET_PRESENT_CONTROL_LWRSOR_MODE_MONO                        (0x00000000)
#define LW977D_HEAD_SET_PRESENT_CONTROL_LWRSOR_MODE_STEREO                      (0x00000001)
#define LW977D_HEAD_SET_PRESENT_CONTROL_LWRSOR_MODE_SPEC_FLIP                   (0x00000002)
#define LW977D_HEAD_SET_CONTROL_LWRSOR(a)                                       (0x00000480 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_ENABLE                                   31:31
#define LW977D_HEAD_SET_CONTROL_LWRSOR_ENABLE_DISABLE                           (0x00000000)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_ENABLE_ENABLE                            (0x00000001)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_FORMAT                                   25:24
#define LW977D_HEAD_SET_CONTROL_LWRSOR_FORMAT_A1R5G5B5                          (0x00000000)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_FORMAT_A8R8G8B8                          (0x00000001)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_SIZE                                     27:26
#define LW977D_HEAD_SET_CONTROL_LWRSOR_SIZE_W32_H32                             (0x00000000)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_SIZE_W64_H64                             (0x00000001)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_SIZE_W128_H128                           (0x00000002)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_SIZE_W256_H256                           (0x00000003)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_HOT_SPOT_X                               15:8
#define LW977D_HEAD_SET_CONTROL_LWRSOR_HOT_SPOT_Y                               23:16
#define LW977D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION                              29:28
#define LW977D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_ALPHA_BLEND                  (0x00000000)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_PREMULT_ALPHA_BLEND          (0x00000001)
#define LW977D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION_XOR                          (0x00000002)
#define LW977D_HEAD_SET_OFFSETS_LWRSOR(a,b)                                     (0x00000484 + (a)*0x00000300 + (b)*0x00000004)
#define LW977D_HEAD_SET_OFFSETS_LWRSOR_ORIGIN                                   31:0
#define LW977D_HEAD_SET_CONTEXT_DMAS_LWRSOR(a,b)                                (0x0000048C + (a)*0x00000300 + (b)*0x00000004)
#define LW977D_HEAD_SET_CONTEXT_DMAS_LWRSOR_HANDLE                              31:0
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER(a)                                (0x00000494 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS                     2:0
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS_TAPS_1              (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS_TAPS_2              (0x00000001)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS_TAPS_3              (0x00000002)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS_TAPS_3_ADAPTIVE     (0x00000003)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_VERTICAL_TAPS_TAPS_5              (0x00000004)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_HORIZONTAL_TAPS                   4:3
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_HORIZONTAL_TAPS_TAPS_1            (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_HORIZONTAL_TAPS_TAPS_2            (0x00000001)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_HORIZONTAL_TAPS_TAPS_8            (0x00000002)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_HRESPONSE_BIAS                    23:16
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_VRESPONSE_BIAS                    31:24
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_FORCE422                          8:8
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_FORCE422_DISABLE                  (0x00000000)
#define LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER_FORCE422_ENABLE                   (0x00000001)
#define LW977D_HEAD_SET_PROCAMP(a)                                              (0x00000498 + (a)*0x00000300)
#define LW977D_HEAD_SET_PROCAMP_COLOR_SPACE                                     1:0
#define LW977D_HEAD_SET_PROCAMP_COLOR_SPACE_RGB                                 (0x00000000)
#define LW977D_HEAD_SET_PROCAMP_COLOR_SPACE_YUV_601                             (0x00000001)
#define LW977D_HEAD_SET_PROCAMP_COLOR_SPACE_YUV_709                             (0x00000002)
#define LW977D_HEAD_SET_PROCAMP_COLOR_SPACE_YUV_2020                            (0x00000003)
#define LW977D_HEAD_SET_PROCAMP_CHROMA_LPF                                      2:2
#define LW977D_HEAD_SET_PROCAMP_CHROMA_LPF_AUTO                                 (0x00000000)
#define LW977D_HEAD_SET_PROCAMP_CHROMA_LPF_ON                                   (0x00000001)
#define LW977D_HEAD_SET_PROCAMP_SAT_COS                                         19:8
#define LW977D_HEAD_SET_PROCAMP_SAT_SINE                                        31:20
#define LW977D_HEAD_SET_PROCAMP_DYNAMIC_RANGE                                   5:5
#define LW977D_HEAD_SET_PROCAMP_DYNAMIC_RANGE_VESA                              (0x00000000)
#define LW977D_HEAD_SET_PROCAMP_DYNAMIC_RANGE_CEA                               (0x00000001)
#define LW977D_HEAD_SET_PROCAMP_RANGE_COMPRESSION                               6:6
#define LW977D_HEAD_SET_PROCAMP_RANGE_COMPRESSION_DISABLE                       (0x00000000)
#define LW977D_HEAD_SET_PROCAMP_RANGE_COMPRESSION_ENABLE                        (0x00000001)
#define LW977D_HEAD_SET_DITHER_CONTROL(a)                                       (0x000004A0 + (a)*0x00000300)
#define LW977D_HEAD_SET_DITHER_CONTROL_ENABLE                                   0:0
#define LW977D_HEAD_SET_DITHER_CONTROL_ENABLE_DISABLE                           (0x00000000)
#define LW977D_HEAD_SET_DITHER_CONTROL_ENABLE_ENABLE                            (0x00000001)
#define LW977D_HEAD_SET_DITHER_CONTROL_BITS                                     2:1
#define LW977D_HEAD_SET_DITHER_CONTROL_BITS_DITHER_TO_6_BITS                    (0x00000000)
#define LW977D_HEAD_SET_DITHER_CONTROL_BITS_DITHER_TO_8_BITS                    (0x00000001)
#define LW977D_HEAD_SET_DITHER_CONTROL_BITS_DITHER_TO_10_BITS                   (0x00000002)
#define LW977D_HEAD_SET_DITHER_CONTROL_MODE                                     6:3
#define LW977D_HEAD_SET_DITHER_CONTROL_MODE_DYNAMIC_ERR_ACC                     (0x00000000)
#define LW977D_HEAD_SET_DITHER_CONTROL_MODE_STATIC_ERR_ACC                      (0x00000001)
#define LW977D_HEAD_SET_DITHER_CONTROL_MODE_DYNAMIC_2X2                         (0x00000002)
#define LW977D_HEAD_SET_DITHER_CONTROL_MODE_STATIC_2X2                          (0x00000003)
#define LW977D_HEAD_SET_DITHER_CONTROL_MODE_TEMPORAL                            (0x00000004)
#define LW977D_HEAD_SET_DITHER_CONTROL_PHASE                                    8:7
#define LW977D_HEAD_SET_VIEWPORT_POINT_IN(a)                                    (0x000004B0 + (a)*0x00000300)
#define LW977D_HEAD_SET_VIEWPORT_POINT_IN_X                                     14:0
#define LW977D_HEAD_SET_VIEWPORT_POINT_IN_Y                                     30:16
#define LW977D_HEAD_SET_VIEWPORT_SIZE_IN(a)                                     (0x000004B8 + (a)*0x00000300)
#define LW977D_HEAD_SET_VIEWPORT_SIZE_IN_WIDTH                                  14:0
#define LW977D_HEAD_SET_VIEWPORT_SIZE_IN_HEIGHT                                 30:16
#define LW977D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST(a)                            (0x000004BC + (a)*0x00000300)
#define LW977D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST_X                             15:0
#define LW977D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST_Y                             31:16
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT(a)                                    (0x000004C0 + (a)*0x00000300)
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_WIDTH                                 14:0
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_HEIGHT                                30:16
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MIN(a)                                (0x000004C4 + (a)*0x00000300)
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MIN_WIDTH                             14:0
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MIN_HEIGHT                            30:16
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MAX(a)                                (0x000004C8 + (a)*0x00000300)
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MAX_WIDTH                             14:0
#define LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MAX_HEIGHT                            30:16
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS(a)                            (0x000004D0 + (a)*0x00000300)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_USABLE                        0:0
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_USABLE_FALSE                  (0x00000000)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_USABLE_TRUE                   (0x00000001)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_DIST_RENDER_USABLE            4:4
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_DIST_RENDER_USABLE_FALSE      (0x00000000)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_DIST_RENDER_USABLE_TRUE       (0x00000001)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_PIXEL_DEPTH                   11:8
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_PIXEL_DEPTH_BPP_8             (0x00000000)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_PIXEL_DEPTH_BPP_16            (0x00000001)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_PIXEL_DEPTH_BPP_32            (0x00000003)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_PIXEL_DEPTH_BPP_64            (0x00000005)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_SUPER_SAMPLE                  13:12
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_SUPER_SAMPLE_X1_AA            (0x00000000)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_SUPER_SAMPLE_X4_AA            (0x00000002)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_BASE_LUT                      17:16
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_BASE_LUT_USAGE_NONE           (0x00000000)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_BASE_LUT_USAGE_257            (0x00000001)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_BASE_LUT_USAGE_1025           (0x00000002)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_OUTPUT_LUT                    21:20
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_OUTPUT_LUT_USAGE_NONE         (0x00000000)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_OUTPUT_LUT_USAGE_257          (0x00000001)
#define LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS_OUTPUT_LUT_USAGE_1025         (0x00000002)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS(a)                                 (0x000004D4 + (a)*0x00000300)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_USABLE                             0:0
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_USABLE_FALSE                       (0x00000000)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_USABLE_TRUE                        (0x00000001)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH                        11:8
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH_BPP_16                 (0x00000001)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH_BPP_32                 (0x00000003)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH_BPP_64                 (0x00000005)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_OVERLAY_LUT                        13:12
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_OVERLAY_LUT_USAGE_NONE             (0x00000000)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_OVERLAY_LUT_USAGE_257              (0x00000001)
#define LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS_OVERLAY_LUT_USAGE_1025             (0x00000002)
#define LW977D_HEAD_SET_PROCESSING(a)                                           (0x000004E0 + (a)*0x00000300)
#define LW977D_HEAD_SET_PROCESSING_USE_GAIN_OFS                                 0:0
#define LW977D_HEAD_SET_PROCESSING_USE_GAIN_OFS_DISABLE                         (0x00000000)
#define LW977D_HEAD_SET_PROCESSING_USE_GAIN_OFS_ENABLE                          (0x00000001)
#define LW977D_HEAD_SET_COLWERSION_RED(a)                                       (0x000004E4 + (a)*0x00000300)
#define LW977D_HEAD_SET_COLWERSION_RED_GAIN                                     15:0
#define LW977D_HEAD_SET_COLWERSION_RED_OFS                                      31:16
#define LW977D_HEAD_SET_COLWERSION_GRN(a)                                       (0x000004E8 + (a)*0x00000300)
#define LW977D_HEAD_SET_COLWERSION_GRN_GAIN                                     15:0
#define LW977D_HEAD_SET_COLWERSION_GRN_OFS                                      31:16
#define LW977D_HEAD_SET_COLWERSION_BLU(a)                                       (0x000004EC + (a)*0x00000300)
#define LW977D_HEAD_SET_COLWERSION_BLU_GAIN                                     15:0
#define LW977D_HEAD_SET_COLWERSION_BLU_OFS                                      31:16
#define LW977D_HEAD_SET_CSC_RED2RED(a)                                          (0x000004F0 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_RED2RED_NEVER_YIELD_TO_BASE                         31:31
#define LW977D_HEAD_SET_CSC_RED2RED_NEVER_YIELD_TO_BASE_DISABLE                 (0x00000000)
#define LW977D_HEAD_SET_CSC_RED2RED_NEVER_YIELD_TO_BASE_ENABLE                  (0x00000001)
#define LW977D_HEAD_SET_CSC_RED2RED_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_GRN2RED(a)                                          (0x000004F4 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_GRN2RED_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_BLU2RED(a)                                          (0x000004F8 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_BLU2RED_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_CONSTANT2RED(a)                                     (0x000004FC + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_CONSTANT2RED_COEFF                                  18:0
#define LW977D_HEAD_SET_CSC_RED2GRN(a)                                          (0x00000500 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_RED2GRN_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_GRN2GRN(a)                                          (0x00000504 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_GRN2GRN_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_BLU2GRN(a)                                          (0x00000508 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_BLU2GRN_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_CONSTANT2GRN(a)                                     (0x0000050C + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_CONSTANT2GRN_COEFF                                  18:0
#define LW977D_HEAD_SET_CSC_RED2BLU(a)                                          (0x00000510 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_RED2BLU_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_GRN2BLU(a)                                          (0x00000514 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_GRN2BLU_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_BLU2BLU(a)                                          (0x00000518 + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_BLU2BLU_COEFF                                       18:0
#define LW977D_HEAD_SET_CSC_CONSTANT2BLU(a)                                     (0x0000051C + (a)*0x00000300)
#define LW977D_HEAD_SET_CSC_CONSTANT2BLU_COEFF                                  18:0
#define LW977D_HEAD_SET_HDMI_CTRL(a)                                            (0x00000520 + (a)*0x00000300)
#define LW977D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT                                  2:0
#define LW977D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT_NORMAL                           (0x00000000)
#define LW977D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT_EXTENDED                         (0x00000001)
#define LW977D_HEAD_SET_HDMI_CTRL_VIDEO_FORMAT_STEREO3D                         (0x00000002)
#define LW977D_HEAD_SET_HDMI_CTRL_HDMI_VIC                                      11:4
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE                            15:12
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_FRAME_PACKED               (0x00000000)
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_FIELD_ALTERNATIVE          (0x00000001)
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_LINE_ALTERNATIVE           (0x00000002)
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_SIDE_BY_SIDE_FULL          (0x00000003)
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_L_DEPTH                    (0x00000004)
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_L_DEPTH_GRAPHICS           (0x00000005)
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_TOP_AND_BOTTOM             (0x00000006)
#define LW977D_HEAD_SET_HDMI_CTRL_STEREO3D_STRUCTURE_SIDE_BY_SIDE_HALF          (0x00000008)
#define LW977D_HEAD_SET_VACTIVE_SPACE_COLOR(a)                                  (0x00000524 + (a)*0x00000300)
#define LW977D_HEAD_SET_VACTIVE_SPACE_COLOR_RED_CR                              9:0
#define LW977D_HEAD_SET_VACTIVE_SPACE_COLOR_GRN_Y                               19:10
#define LW977D_HEAD_SET_VACTIVE_SPACE_COLOR_BLU_CB                              29:20
#define LW977D_HEAD_SET_PIXEL_REORDER_CONTROL(a)                                (0x00000528 + (a)*0x00000300)
#define LW977D_HEAD_SET_PIXEL_REORDER_CONTROL_BANK_WIDTH                        13:0
#define LW977D_HEAD_SET_DISPLAY_ID(a,b)                                         (0x0000052C + (a)*0x00000300 + (b)*0x00000004)
#define LW977D_HEAD_SET_DISPLAY_ID_CODE                                         31:0
#define LW977D_HEAD_SET_SW_SPARE_A(a)                                           (0x0000054C + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_SPARE_A_CODE                                         31:0
#define LW977D_HEAD_SET_SW_SPARE_B(a)                                           (0x00000550 + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_SPARE_B_CODE                                         31:0
#define LW977D_HEAD_SET_SW_SPARE_C(a)                                           (0x00000554 + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_SPARE_C_CODE                                         31:0
#define LW977D_HEAD_SET_SW_SPARE_D(a)                                           (0x00000558 + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_SPARE_D_CODE                                         31:0
#define LW977D_HEAD_SET_GET_BLANKING_CTRL(a)                                    (0x0000055C + (a)*0x00000300)
#define LW977D_HEAD_SET_GET_BLANKING_CTRL_BLANK                                 0:0
#define LW977D_HEAD_SET_GET_BLANKING_CTRL_BLANK_NO_CHANGE                       (0x00000000)
#define LW977D_HEAD_SET_GET_BLANKING_CTRL_BLANK_ENABLE                          (0x00000001)
#define LW977D_HEAD_SET_GET_BLANKING_CTRL_UNBLANK                               1:1
#define LW977D_HEAD_SET_GET_BLANKING_CTRL_UNBLANK_NO_CHANGE                     (0x00000000)
#define LW977D_HEAD_SET_GET_BLANKING_CTRL_UNBLANK_ENABLE                        (0x00000001)
#define LW977D_HEAD_SET_CONTROL_COMPRESSION(a)                                  (0x00000560 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_ENABLE                              0:0
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_ENABLE_DISABLE                      (0x00000000)
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_ENABLE_ENABLE                       (0x00000001)
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_CHUNK_BANDWIDTH                     12:1
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_LAST_BANDWIDTH                      24:13
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_LA(a)                               (0x00000564 + (a)*0x00000300)
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_LA_LOSSY1                           7:4
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_LA_LOSSY2                           11:8
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_LA_LOSSY3                           15:12
#define LW977D_HEAD_SET_CONTROL_COMPRESSION_LA_CHUNK_SIZE                       23:16
#define LW977D_HEAD_SET_STALL_LOCK(a)                                           (0x00000568 + (a)*0x00000300)
#define LW977D_HEAD_SET_STALL_LOCK_ENABLE                                       0:0
#define LW977D_HEAD_SET_STALL_LOCK_ENABLE_FALSE                                 (0x00000000)
#define LW977D_HEAD_SET_STALL_LOCK_ENABLE_TRUE                                  (0x00000001)
#define LW977D_HEAD_SET_STALL_LOCK_MODE                                         1:1
#define LW977D_HEAD_SET_STALL_LOCK_MODE_CONTINUOUS                              (0x00000000)
#define LW977D_HEAD_SET_STALL_LOCK_MODE_ONE_SHOT                                (0x00000001)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN                                     6:2
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN(i)                         (0x00000000 +(i))
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN__SIZE_1                    16
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_0                          (0x00000000)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_1                          (0x00000001)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_2                          (0x00000002)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_3                          (0x00000003)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_4                          (0x00000004)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_5                          (0x00000005)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_6                          (0x00000006)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_7                          (0x00000007)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_8                          (0x00000008)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_9                          (0x00000009)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_A                          (0x0000000A)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_B                          (0x0000000B)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_C                          (0x0000000C)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_D                          (0x0000000D)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_E                          (0x0000000E)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_LOCK_PIN_F                          (0x0000000F)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_UNSPECIFIED                         (0x00000010)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK(i)               (0x00000018 +(i))
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1          4
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_0                (0x00000018)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_1                (0x00000019)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_2                (0x0000001A)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_SCAN_LOCK_3                (0x0000001B)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK(i)               (0x0000001E +(i))
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK__SIZE_1          2
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK_0                (0x0000001E)
#define LW977D_HEAD_SET_STALL_LOCK_LOCK_PIN_INTERNAL_FLIP_LOCK_1                (0x0000001F)
#define LW977D_HEAD_SET_STALL_LOCK_UNSTALL_MODE                                 7:7
#define LW977D_HEAD_SET_STALL_LOCK_UNSTALL_MODE_CRASH_LOCK                      (0x00000000)
#define LW977D_HEAD_SET_STALL_LOCK_UNSTALL_MODE_LINE_LOCK                       (0x00000001)
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_A(a)                              (0x000006D0 + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_A_UNUSED                          31:0
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_B(a)                              (0x000006D4 + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_B_UNUSED                          31:0
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_C(a)                              (0x000006D8 + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_C_UNUSED                          31:0
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_D(a)                              (0x000006DC + (a)*0x00000300)
#define LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_D_UNUSED                          31:0
#define LW977D_HEAD_SET_SPARE(a)                                                (0x000006EC + (a)*0x00000300)
#define LW977D_HEAD_SET_SPARE_UNUSED                                            31:0
#define LW977D_HEAD_SET_SPARE_NOOP(a,b)                                         (0x000006F0 + (a)*0x00000300 + (b)*0x00000004)
#define LW977D_HEAD_SET_SPARE_NOOP_UNUSED                                       31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl977d_h
