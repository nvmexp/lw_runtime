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



#ifndef _clC67e_h_
#define _clC67e_h_


#ifdef __cplusplus
extern "C" {
#endif

#define LWC67E_WINDOW_CHANNEL_DMA                                               (0x0000C67E)

typedef volatile struct _clc67e_tag0 {
    LwV32 Put;                                                                  // 0x00000000 - 0x00000003
    LwV32 Get;                                                                  // 0x00000004 - 0x00000007
    LwV32 Reserved00[0x7E];
    LwV32 Update;                                                               // 0x00000200 - 0x00000203
    LwV32 SetSemaphoreAcquireHi;                                                // 0x00000204 - 0x00000207
    LwV32 GetLine;                                                              // 0x00000208 - 0x0000020B
    LwV32 SetSemaphoreControl;                                                  // 0x0000020C - 0x0000020F
    LwV32 SetSemaphoreAcquire;                                                  // 0x00000210 - 0x00000213
    LwV32 SetSemaphoreRelease;                                                  // 0x00000214 - 0x00000217
    LwV32 SetContextDmaSemaphore;                                               // 0x00000218 - 0x0000021B
    LwV32 SetContextDmaNotifier;                                                // 0x0000021C - 0x0000021F
    LwV32 SetNotifierControl;                                                   // 0x00000220 - 0x00000223
    LwV32 SetSize;                                                              // 0x00000224 - 0x00000227
    LwV32 SetStorage;                                                           // 0x00000228 - 0x0000022B
    LwV32 SetParams;                                                            // 0x0000022C - 0x0000022F
    LwV32 SetPlanarStorage[3];                                                  // 0x00000230 - 0x0000023B
    LwV32 SetSemaphoreReleaseHi;                                                // 0x0000023C - 0x0000023F
    LwV32 SetContextDmaIso[6];                                                  // 0x00000240 - 0x00000257
    LwV32 Reserved01[0x2];
    LwV32 SetOffset[6];                                                         // 0x00000260 - 0x00000277
    LwV32 Reserved02[0x6];
    LwV32 SetPointIn[2];                                                        // 0x00000290 - 0x00000297
    LwV32 SetSizeIn;                                                            // 0x00000298 - 0x0000029B
    LwV32 SetValidPointIn;                                                      // 0x0000029C - 0x0000029F
    LwV32 SetValidSizeIn;                                                       // 0x000002A0 - 0x000002A3
    LwV32 SetSizeOut;                                                           // 0x000002A4 - 0x000002A7
    LwV32 SetControlInputScaler;                                                // 0x000002A8 - 0x000002AB
    LwV32 SetInputScalerCoeffValue;                                             // 0x000002AC - 0x000002AF
    LwV32 Reserved03[0xF];
    LwV32 SetCompositionControl;                                                // 0x000002EC - 0x000002EF
    LwV32 SetCompositionConstantAlpha;                                          // 0x000002F0 - 0x000002F3
    LwV32 SetCompositionFactorSelect;                                           // 0x000002F4 - 0x000002F7
    LwV32 SetKeyAlpha;                                                          // 0x000002F8 - 0x000002FB
    LwV32 SetKeyRed_Cr;                                                         // 0x000002FC - 0x000002FF
    LwV32 SetKeyGreen_Y;                                                        // 0x00000300 - 0x00000303
    LwV32 SetKeyBlue_Cb;                                                        // 0x00000304 - 0x00000307
    LwV32 SetPresentControl;                                                    // 0x00000308 - 0x0000030B
    LwV32 SetAcqSemaphoreValueHi;                                               // 0x0000030C - 0x0000030F
    LwV32 SetOpaquePointIn[4];                                                  // 0x00000310 - 0x0000031F
    LwV32 SetOpaqueSizeIn[4];                                                   // 0x00000320 - 0x0000032F
    LwV32 SetAcqSemaphoreControl;                                               // 0x00000330 - 0x00000333
    LwV32 SetAcqSemaphoreValue;                                                 // 0x00000334 - 0x00000337
    LwV32 SetContextDmaAcqSemaphore;                                            // 0x00000338 - 0x0000033B
    LwV32 SetScanDirection;                                                     // 0x0000033C - 0x0000033F
    LwV32 SetTimestampOriginLo;                                                 // 0x00000340 - 0x00000343
    LwV32 SetTimestampOriginHi;                                                 // 0x00000344 - 0x00000347
    LwV32 SetUpdateTimestampLo;                                                 // 0x00000348 - 0x0000034B
    LwV32 SetUpdateTimestampHi;                                                 // 0x0000034C - 0x0000034F
    LwV32 SyncPointRelease;                                                     // 0x00000350 - 0x00000353
    LwV32 SetSyncPointControl;                                                  // 0x00000354 - 0x00000357
    LwV32 SetSyncPointAcquire;                                                  // 0x00000358 - 0x0000035B
    LwV32 Reserved04[0x2];
    LwV32 CdeControl;                                                           // 0x00000364 - 0x00000367
    LwV32 CdeCtbEntry;                                                          // 0x00000368 - 0x0000036B
    LwV32 CdeZbcColor;                                                          // 0x0000036C - 0x0000036F
    LwV32 SetInterlockFlags;                                                    // 0x00000370 - 0x00000373
    LwV32 SetWindowInterlockFlags;                                              // 0x00000374 - 0x00000377
    LwV32 Reserved05[0x2];
    LwV32 CdeComptagBase[6];                                                    // 0x00000380 - 0x00000397
    LwV32 SetExtPacketControl;                                                  // 0x00000398 - 0x0000039B
    LwV32 SetExtPacketData;                                                     // 0x0000039C - 0x0000039F
    LwV32 Reserved06[0x18];
    LwV32 SetFmtCoefficientC00;                                                 // 0x00000400 - 0x00000403
    LwV32 SetFmtCoefficientC01;                                                 // 0x00000404 - 0x00000407
    LwV32 SetFmtCoefficientC02;                                                 // 0x00000408 - 0x0000040B
    LwV32 SetFmtCoefficientC03;                                                 // 0x0000040C - 0x0000040F
    LwV32 SetFmtCoefficientC10;                                                 // 0x00000410 - 0x00000413
    LwV32 SetFmtCoefficientC11;                                                 // 0x00000414 - 0x00000417
    LwV32 SetFmtCoefficientC12;                                                 // 0x00000418 - 0x0000041B
    LwV32 SetFmtCoefficientC13;                                                 // 0x0000041C - 0x0000041F
    LwV32 SetFmtCoefficientC20;                                                 // 0x00000420 - 0x00000423
    LwV32 SetFmtCoefficientC21;                                                 // 0x00000424 - 0x00000427
    LwV32 SetFmtCoefficientC22;                                                 // 0x00000428 - 0x0000042B
    LwV32 SetFmtCoefficientC23;                                                 // 0x0000042C - 0x0000042F
    LwV32 Reserved07[0x4];
    LwV32 SetILutControl;                                                       // 0x00000440 - 0x00000443
    LwV32 SetContextDmaILut;                                                    // 0x00000444 - 0x00000447
    LwV32 SetOffsetILut;                                                        // 0x00000448 - 0x0000044B
    LwV32 Reserved08[0x4];
    LwV32 SetCsc00Control;                                                      // 0x0000045C - 0x0000045F
    LwV32 SetCsc00CoefficientC00;                                               // 0x00000460 - 0x00000463
    LwV32 SetCsc00CoefficientC01;                                               // 0x00000464 - 0x00000467
    LwV32 SetCsc00CoefficientC02;                                               // 0x00000468 - 0x0000046B
    LwV32 SetCsc00CoefficientC03;                                               // 0x0000046C - 0x0000046F
    LwV32 SetCsc00CoefficientC10;                                               // 0x00000470 - 0x00000473
    LwV32 SetCsc00CoefficientC11;                                               // 0x00000474 - 0x00000477
    LwV32 SetCsc00CoefficientC12;                                               // 0x00000478 - 0x0000047B
    LwV32 SetCsc00CoefficientC13;                                               // 0x0000047C - 0x0000047F
    LwV32 SetCsc00CoefficientC20;                                               // 0x00000480 - 0x00000483
    LwV32 SetCsc00CoefficientC21;                                               // 0x00000484 - 0x00000487
    LwV32 SetCsc00CoefficientC22;                                               // 0x00000488 - 0x0000048B
    LwV32 SetCsc00CoefficientC23;                                               // 0x0000048C - 0x0000048F
    LwV32 Reserved09[0x4];
    LwV32 SetCsc0LutControl;                                                    // 0x000004A0 - 0x000004A3
    LwV32 SetCsc0LutSegmentSize;                                                // 0x000004A4 - 0x000004A7
    LwV32 SetCsc0LutEntry;                                                      // 0x000004A8 - 0x000004AB
    LwV32 Reserved10[0x4];
    LwV32 SetCsc01Control;                                                      // 0x000004BC - 0x000004BF
    LwV32 SetCsc01CoefficientC00;                                               // 0x000004C0 - 0x000004C3
    LwV32 SetCsc01CoefficientC01;                                               // 0x000004C4 - 0x000004C7
    LwV32 SetCsc01CoefficientC02;                                               // 0x000004C8 - 0x000004CB
    LwV32 SetCsc01CoefficientC03;                                               // 0x000004CC - 0x000004CF
    LwV32 SetCsc01CoefficientC10;                                               // 0x000004D0 - 0x000004D3
    LwV32 SetCsc01CoefficientC11;                                               // 0x000004D4 - 0x000004D7
    LwV32 SetCsc01CoefficientC12;                                               // 0x000004D8 - 0x000004DB
    LwV32 SetCsc01CoefficientC13;                                               // 0x000004DC - 0x000004DF
    LwV32 SetCsc01CoefficientC20;                                               // 0x000004E0 - 0x000004E3
    LwV32 SetCsc01CoefficientC21;                                               // 0x000004E4 - 0x000004E7
    LwV32 SetCsc01CoefficientC22;                                               // 0x000004E8 - 0x000004EB
    LwV32 SetCsc01CoefficientC23;                                               // 0x000004EC - 0x000004EF
    LwV32 Reserved11[0x4];
    LwV32 SetTmoControl;                                                        // 0x00000500 - 0x00000503
    LwV32 Reserved12[0x1];
    LwV32 SetTmoLowIntensityZone;                                               // 0x00000508 - 0x0000050B
    LwV32 SetTmoLowIntensityValue;                                              // 0x0000050C - 0x0000050F
    LwV32 SetTmoMediumIntensityZone;                                            // 0x00000510 - 0x00000513
    LwV32 SetTmoMediumIntensityValue;                                           // 0x00000514 - 0x00000517
    LwV32 SetTmoHighIntensityZone;                                              // 0x00000518 - 0x0000051B
    LwV32 SetTmoHighIntensityValue;                                             // 0x0000051C - 0x0000051F
    LwV32 Reserved13[0x2];
    LwV32 SetContextDmaTmoLut;                                                  // 0x00000528 - 0x0000052B
    LwV32 SetOffsetTmoLut;                                                      // 0x0000052C - 0x0000052F
    LwV32 Reserved14[0x3];
    LwV32 SetCsc10Control;                                                      // 0x0000053C - 0x0000053F
    LwV32 SetCsc10CoefficientC00;                                               // 0x00000540 - 0x00000543
    LwV32 SetCsc10CoefficientC01;                                               // 0x00000544 - 0x00000547
    LwV32 SetCsc10CoefficientC02;                                               // 0x00000548 - 0x0000054B
    LwV32 SetCsc10CoefficientC03;                                               // 0x0000054C - 0x0000054F
    LwV32 SetCsc10CoefficientC10;                                               // 0x00000550 - 0x00000553
    LwV32 SetCsc10CoefficientC11;                                               // 0x00000554 - 0x00000557
    LwV32 SetCsc10CoefficientC12;                                               // 0x00000558 - 0x0000055B
    LwV32 SetCsc10CoefficientC13;                                               // 0x0000055C - 0x0000055F
    LwV32 SetCsc10CoefficientC20;                                               // 0x00000560 - 0x00000563
    LwV32 SetCsc10CoefficientC21;                                               // 0x00000564 - 0x00000567
    LwV32 SetCsc10CoefficientC22;                                               // 0x00000568 - 0x0000056B
    LwV32 SetCsc10CoefficientC23;                                               // 0x0000056C - 0x0000056F
    LwV32 Reserved15[0x4];
    LwV32 SetCsc1LutControl;                                                    // 0x00000580 - 0x00000583
    LwV32 SetCsc1LutSegmentSize;                                                // 0x00000584 - 0x00000587
    LwV32 SetCsc1LutEntry;                                                      // 0x00000588 - 0x0000058B
    LwV32 Reserved16[0x4];
    LwV32 SetCsc11Control;                                                      // 0x0000059C - 0x0000059F
    LwV32 SetCsc11CoefficientC00;                                               // 0x000005A0 - 0x000005A3
    LwV32 SetCsc11CoefficientC01;                                               // 0x000005A4 - 0x000005A7
    LwV32 SetCsc11CoefficientC02;                                               // 0x000005A8 - 0x000005AB
    LwV32 SetCsc11CoefficientC03;                                               // 0x000005AC - 0x000005AF
    LwV32 SetCsc11CoefficientC10;                                               // 0x000005B0 - 0x000005B3
    LwV32 SetCsc11CoefficientC11;                                               // 0x000005B4 - 0x000005B7
    LwV32 SetCsc11CoefficientC12;                                               // 0x000005B8 - 0x000005BB
    LwV32 SetCsc11CoefficientC13;                                               // 0x000005BC - 0x000005BF
    LwV32 SetCsc11CoefficientC20;                                               // 0x000005C0 - 0x000005C3
    LwV32 SetCsc11CoefficientC21;                                               // 0x000005C4 - 0x000005C7
    LwV32 SetCsc11CoefficientC22;                                               // 0x000005C8 - 0x000005CB
    LwV32 SetCsc11CoefficientC23;                                               // 0x000005CC - 0x000005CF
    LwV32 SetClampRange;                                                        // 0x000005D0 - 0x000005D3
    LwV32 SwReserved[4];                                                        // 0x000005D4 - 0x000005E3
    LwV32 Reserved17[0x287];
} LWC67EDispControlDma;


// dma opcode instructions
#define LWC67E_DMA                                                                     
#define LWC67E_DMA_OPCODE                                                        31:29 
#define LWC67E_DMA_OPCODE_METHOD                                            0x00000000 
#define LWC67E_DMA_OPCODE_JUMP                                              0x00000001 
#define LWC67E_DMA_OPCODE_NONINC_METHOD                                     0x00000002 
#define LWC67E_DMA_OPCODE_SET_SUBDEVICE_MASK                                0x00000003 
#define LWC67E_DMA_METHOD_COUNT                                                  27:18 
#define LWC67E_DMA_METHOD_OFFSET                                                  13:2 
#define LWC67E_DMA_DATA                                                           31:0 
#define LWC67E_DMA_DATA_NOP                                                 0x00000000 
#define LWC67E_DMA_JUMP_OFFSET                                                    11:2 
#define LWC67E_DMA_SET_SUBDEVICE_MASK_VALUE                                       11:0 

// class methods
#define LWC67E_PUT                                                              (0x00000000)
#define LWC67E_PUT_PTR                                                          9:0
#define LWC67E_GET                                                              (0x00000004)
#define LWC67E_GET_PTR                                                          9:0
#define LWC67E_UPDATE                                                           (0x00000200)
#define LWC67E_UPDATE_RELEASE_ELV                                               0:0
#define LWC67E_UPDATE_RELEASE_ELV_FALSE                                         (0x00000000)
#define LWC67E_UPDATE_RELEASE_ELV_TRUE                                          (0x00000001)
#define LWC67E_UPDATE_FLIP_LOCK_PIN                                             8:4
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_NONE                               (0x00000000)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN(i)                                 (0x00000001 +(i))
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN__SIZE_1                            16
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_0                                  (0x00000001)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_1                                  (0x00000002)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_2                                  (0x00000003)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_3                                  (0x00000004)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_4                                  (0x00000005)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_5                                  (0x00000006)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_6                                  (0x00000007)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_7                                  (0x00000008)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_8                                  (0x00000009)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_9                                  (0x0000000A)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_A                                  (0x0000000B)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_B                                  (0x0000000C)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_C                                  (0x0000000D)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_D                                  (0x0000000E)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_E                                  (0x0000000F)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_LOCK_PIN_F                                  (0x00000010)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_0                        (0x00000014)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_1                        (0x00000015)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_2                        (0x00000016)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_FLIP_LOCK_3                        (0x00000017)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK(i)                       (0x00000018 +(i))
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK__SIZE_1                  8
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_0                        (0x00000018)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_1                        (0x00000019)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_2                        (0x0000001A)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_3                        (0x0000001B)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_4                        (0x0000001C)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_5                        (0x0000001D)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_6                        (0x0000001E)
#define LWC67E_UPDATE_FLIP_LOCK_PIN_INTERNAL_SCAN_LOCK_7                        (0x0000001F)
#define LWC67E_UPDATE_INTERLOCK_WITH_WIN_IMM                                    12:12
#define LWC67E_UPDATE_INTERLOCK_WITH_WIN_IMM_DISABLE                            (0x00000000)
#define LWC67E_UPDATE_INTERLOCK_WITH_WIN_IMM_ENABLE                             (0x00000001)
#define LWC67E_SET_SEMAPHORE_ACQUIRE_HI                                         (0x00000204)
#define LWC67E_SET_SEMAPHORE_ACQUIRE_HI_VALUE                                   31:0
#define LWC67E_GET_LINE                                                         (0x00000208)
#define LWC67E_GET_LINE_LINE                                                    15:0
#define LWC67E_SET_SEMAPHORE_CONTROL                                            (0x0000020C)
#define LWC67E_SET_SEMAPHORE_CONTROL_OFFSET                                     7:0
#define LWC67E_SET_SEMAPHORE_CONTROL_SKIP_ACQ                                   11:11
#define LWC67E_SET_SEMAPHORE_CONTROL_SKIP_ACQ_FALSE                             (0x00000000)
#define LWC67E_SET_SEMAPHORE_CONTROL_SKIP_ACQ_TRUE                              (0x00000001)
#define LWC67E_SET_SEMAPHORE_CONTROL_PAYLOAD_SIZE                               15:15
#define LWC67E_SET_SEMAPHORE_CONTROL_PAYLOAD_SIZE_PAYLOAD_32BIT                 (0x00000000)
#define LWC67E_SET_SEMAPHORE_CONTROL_PAYLOAD_SIZE_PAYLOAD_64BIT                 (0x00000001)
#define LWC67E_SET_SEMAPHORE_CONTROL_ACQ_MODE                                   13:12
#define LWC67E_SET_SEMAPHORE_CONTROL_ACQ_MODE_EQ                                (0x00000000)
#define LWC67E_SET_SEMAPHORE_CONTROL_ACQ_MODE_CGEQ                              (0x00000001)
#define LWC67E_SET_SEMAPHORE_CONTROL_ACQ_MODE_STRICT_GEQ                        (0x00000002)
#define LWC67E_SET_SEMAPHORE_CONTROL_REL_MODE                                   14:14
#define LWC67E_SET_SEMAPHORE_CONTROL_REL_MODE_WRITE                             (0x00000000)
#define LWC67E_SET_SEMAPHORE_CONTROL_REL_MODE_WRITE_AWAKEN                      (0x00000001)
#define LWC67E_SET_SEMAPHORE_ACQUIRE                                            (0x00000210)
#define LWC67E_SET_SEMAPHORE_ACQUIRE_VALUE                                      31:0
#define LWC67E_SET_SEMAPHORE_RELEASE                                            (0x00000214)
#define LWC67E_SET_SEMAPHORE_RELEASE_VALUE                                      31:0
#define LWC67E_SET_CONTEXT_DMA_SEMAPHORE                                        (0x00000218)
#define LWC67E_SET_CONTEXT_DMA_SEMAPHORE_HANDLE                                 31:0
#define LWC67E_SET_CONTEXT_DMA_NOTIFIER                                         (0x0000021C)
#define LWC67E_SET_CONTEXT_DMA_NOTIFIER_HANDLE                                  31:0
#define LWC67E_SET_NOTIFIER_CONTROL                                             (0x00000220)
#define LWC67E_SET_NOTIFIER_CONTROL_MODE                                        0:0
#define LWC67E_SET_NOTIFIER_CONTROL_MODE_WRITE                                  (0x00000000)
#define LWC67E_SET_NOTIFIER_CONTROL_MODE_WRITE_AWAKEN                           (0x00000001)
#define LWC67E_SET_NOTIFIER_CONTROL_OFFSET                                      11:4
#define LWC67E_SET_SIZE                                                         (0x00000224)
#define LWC67E_SET_SIZE_WIDTH                                                   15:0
#define LWC67E_SET_SIZE_HEIGHT                                                  31:16
#define LWC67E_SET_STORAGE                                                      (0x00000228)
#define LWC67E_SET_STORAGE_BLOCK_HEIGHT                                         3:0
#define LWC67E_SET_STORAGE_BLOCK_HEIGHT_LWD_BLOCK_HEIGHT_ONE_GOB                (0x00000000)
#define LWC67E_SET_STORAGE_BLOCK_HEIGHT_LWD_BLOCK_HEIGHT_TWO_GOBS               (0x00000001)
#define LWC67E_SET_STORAGE_BLOCK_HEIGHT_LWD_BLOCK_HEIGHT_FOUR_GOBS              (0x00000002)
#define LWC67E_SET_STORAGE_BLOCK_HEIGHT_LWD_BLOCK_HEIGHT_EIGHT_GOBS             (0x00000003)
#define LWC67E_SET_STORAGE_BLOCK_HEIGHT_LWD_BLOCK_HEIGHT_SIXTEEN_GOBS           (0x00000004)
#define LWC67E_SET_STORAGE_BLOCK_HEIGHT_LWD_BLOCK_HEIGHT_THIRTYTWO_GOBS         (0x00000005)
#define LWC67E_SET_PARAMS                                                       (0x0000022C)
#define LWC67E_SET_PARAMS_FORMAT                                                7:0
#define LWC67E_SET_PARAMS_FORMAT_I8                                             (0x0000001E)
#define LWC67E_SET_PARAMS_FORMAT_R4G4B4A4                                       (0x0000002F)
#define LWC67E_SET_PARAMS_FORMAT_R5G6B5                                         (0x000000E8)
#define LWC67E_SET_PARAMS_FORMAT_A1R5G5B5                                       (0x000000E9)
#define LWC67E_SET_PARAMS_FORMAT_R5G5B5A1                                       (0x0000002E)
#define LWC67E_SET_PARAMS_FORMAT_A8R8G8B8                                       (0x000000CF)
#define LWC67E_SET_PARAMS_FORMAT_X8R8G8B8                                       (0x000000E6)
#define LWC67E_SET_PARAMS_FORMAT_A8B8G8R8                                       (0x000000D5)
#define LWC67E_SET_PARAMS_FORMAT_X8B8G8R8                                       (0x000000F9)
#define LWC67E_SET_PARAMS_FORMAT_A2R10G10B10                                    (0x000000DF)
#define LWC67E_SET_PARAMS_FORMAT_A2B10G10R10                                    (0x000000D1)
#define LWC67E_SET_PARAMS_FORMAT_R16_G16_B16_A16_LWBIAS                         (0x00000023)
#define LWC67E_SET_PARAMS_FORMAT_R16_G16_B16_A16                                (0x000000C6)
#define LWC67E_SET_PARAMS_FORMAT_RF16_GF16_BF16_AF16                            (0x000000CA)
#define LWC67E_SET_PARAMS_FORMAT_Y8_U8__Y8_V8_N422                              (0x00000028)
#define LWC67E_SET_PARAMS_FORMAT_U8_Y8__V8_Y8_N422                              (0x00000029)
#define LWC67E_SET_PARAMS_FORMAT_Y8___U8V8_N444                                 (0x00000035)
#define LWC67E_SET_PARAMS_FORMAT_Y8___U8V8_N422                                 (0x00000036)
#define LWC67E_SET_PARAMS_FORMAT_Y8___V8U8_N420                                 (0x00000038)
#define LWC67E_SET_PARAMS_FORMAT_Y8___U8___V8_N444                              (0x0000003A)
#define LWC67E_SET_PARAMS_FORMAT_Y8___U8___V8_N420                              (0x0000003B)
#define LWC67E_SET_PARAMS_FORMAT_Y10___U10V10_N444                              (0x00000055)
#define LWC67E_SET_PARAMS_FORMAT_Y10___U10V10_N422                              (0x00000056)
#define LWC67E_SET_PARAMS_FORMAT_Y10___V10U10_N420                              (0x00000058)
#define LWC67E_SET_PARAMS_FORMAT_Y12___U12V12_N444                              (0x00000075)
#define LWC67E_SET_PARAMS_FORMAT_Y12___U12V12_N422                              (0x00000076)
#define LWC67E_SET_PARAMS_FORMAT_Y12___V12U12_N420                              (0x00000078)
#define LWC67E_SET_PARAMS_CLAMP_BEFORE_BLEND                                    18:18
#define LWC67E_SET_PARAMS_CLAMP_BEFORE_BLEND_DISABLE                            (0x00000000)
#define LWC67E_SET_PARAMS_CLAMP_BEFORE_BLEND_ENABLE                             (0x00000001)
#define LWC67E_SET_PARAMS_SWAP_UV                                               19:19
#define LWC67E_SET_PARAMS_SWAP_UV_DISABLE                                       (0x00000000)
#define LWC67E_SET_PARAMS_SWAP_UV_ENABLE                                        (0x00000001)
#define LWC67E_SET_PARAMS_FMT_ROUNDING_MODE                                     22:22
#define LWC67E_SET_PARAMS_FMT_ROUNDING_MODE_ROUND_TO_NEAREST                    (0x00000000)
#define LWC67E_SET_PARAMS_FMT_ROUNDING_MODE_ROUND_DOWN                          (0x00000001)
#define LWC67E_SET_PLANAR_STORAGE(b)                                            (0x00000230 + (b)*0x00000004)
#define LWC67E_SET_PLANAR_STORAGE_PITCH                                         12:0
#define LWC67E_SET_SEMAPHORE_RELEASE_HI                                         (0x0000023C)
#define LWC67E_SET_SEMAPHORE_RELEASE_HI_VALUE                                   31:0
#define LWC67E_SET_CONTEXT_DMA_ISO(b)                                           (0x00000240 + (b)*0x00000004)
#define LWC67E_SET_CONTEXT_DMA_ISO_HANDLE                                       31:0
#define LWC67E_SET_OFFSET(b)                                                    (0x00000260 + (b)*0x00000004)
#define LWC67E_SET_OFFSET_ORIGIN                                                31:0
#define LWC67E_SET_POINT_IN(b)                                                  (0x00000290 + (b)*0x00000004)
#define LWC67E_SET_POINT_IN_X                                                   15:0
#define LWC67E_SET_POINT_IN_Y                                                   31:16
#define LWC67E_SET_SIZE_IN                                                      (0x00000298)
#define LWC67E_SET_SIZE_IN_WIDTH                                                15:0
#define LWC67E_SET_SIZE_IN_HEIGHT                                               31:16
#define LWC67E_SET_VALID_POINT_IN                                               (0x0000029C)
#define LWC67E_SET_VALID_POINT_IN_X                                             15:0
#define LWC67E_SET_VALID_POINT_IN_Y                                             31:16
#define LWC67E_SET_VALID_SIZE_IN                                                (0x000002A0)
#define LWC67E_SET_VALID_SIZE_IN_WIDTH                                          15:0
#define LWC67E_SET_VALID_SIZE_IN_HEIGHT                                         31:16
#define LWC67E_SET_SIZE_OUT                                                     (0x000002A4)
#define LWC67E_SET_SIZE_OUT_WIDTH                                               15:0
#define LWC67E_SET_SIZE_OUT_HEIGHT                                              31:16
#define LWC67E_SET_CONTROL_INPUT_SCALER                                         (0x000002A8)
#define LWC67E_SET_CONTROL_INPUT_SCALER_VERTICAL_TAPS                           2:0
#define LWC67E_SET_CONTROL_INPUT_SCALER_VERTICAL_TAPS_TAPS_2                    (0x00000001)
#define LWC67E_SET_CONTROL_INPUT_SCALER_VERTICAL_TAPS_TAPS_5                    (0x00000004)
#define LWC67E_SET_CONTROL_INPUT_SCALER_HORIZONTAL_TAPS                         6:4
#define LWC67E_SET_CONTROL_INPUT_SCALER_HORIZONTAL_TAPS_TAPS_2                  (0x00000001)
#define LWC67E_SET_CONTROL_INPUT_SCALER_HORIZONTAL_TAPS_TAPS_5                  (0x00000004)
#define LWC67E_SET_INPUT_SCALER_COEFF_VALUE                                     (0x000002AC)
#define LWC67E_SET_INPUT_SCALER_COEFF_VALUE_DATA                                9:0
#define LWC67E_SET_INPUT_SCALER_COEFF_VALUE_INDEX                               19:12
#define LWC67E_SET_COMPOSITION_CONTROL                                          (0x000002EC)
#define LWC67E_SET_COMPOSITION_CONTROL_COLOR_KEY_SELECT                         1:0
#define LWC67E_SET_COMPOSITION_CONTROL_COLOR_KEY_SELECT_DISABLE                 (0x00000000)
#define LWC67E_SET_COMPOSITION_CONTROL_COLOR_KEY_SELECT_SRC                     (0x00000001)
#define LWC67E_SET_COMPOSITION_CONTROL_COLOR_KEY_SELECT_DST                     (0x00000002)
#define LWC67E_SET_COMPOSITION_CONTROL_DEPTH                                    11:4
#define LWC67E_SET_COMPOSITION_CONTROL_BYPASS                                   16:16
#define LWC67E_SET_COMPOSITION_CONTROL_BYPASS_DISABLE                           (0x00000000)
#define LWC67E_SET_COMPOSITION_CONTROL_BYPASS_ENABLE                            (0x00000001)
#define LWC67E_SET_COMPOSITION_CONSTANT_ALPHA                                   (0x000002F0)
#define LWC67E_SET_COMPOSITION_CONSTANT_ALPHA_K1                                7:0
#define LWC67E_SET_COMPOSITION_CONSTANT_ALPHA_K2                                15:8
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT                                    (0x000002F4)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT      3:0
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT_ONE  (0x00000001)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT_K1   (0x00000002)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT_K1_TIMES_SRC (0x00000005)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT_K1_TIMES_DST (0x00000006)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT_NEG_K1_TIMES_DST (0x00000008)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT   7:4
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT_ONE (0x00000001)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT_K1 (0x00000002)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT_K1_TIMES_SRC (0x00000005)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT_K1_TIMES_DST (0x00000006)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT_NEG_K1_TIMES_DST (0x00000008)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT      11:8
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_ONE  (0x00000001)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_K1   (0x00000002)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_K2   (0x00000003)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_NEG_K1 (0x00000004)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_K1_TIMES_DST (0x00000006)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_NEG_K1_TIMES_SRC (0x00000007)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT_NEG_K1_TIMES_DST (0x00000008)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT   15:12
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_ONE (0x00000001)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_K1 (0x00000002)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_K2 (0x00000003)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_NEG_K1 (0x00000004)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_K1_TIMES_DST (0x00000006)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_NEG_K1_TIMES_SRC (0x00000007)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT_NEG_K1_TIMES_DST (0x00000008)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_MATCH_SELECT      19:16
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_MATCH_SELECT_K1   (0x00000002)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_MATCH_SELECT_K2   (0x00000003)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_MATCH_SELECT_NEG_K1_TIMES_DST (0x00000008)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_NO_MATCH_SELECT   23:20
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_NO_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_NO_MATCH_SELECT_K1 (0x00000002)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_NO_MATCH_SELECT_K2 (0x00000003)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_SRC_ALPHA_FACTOR_NO_MATCH_SELECT_NEG_K1_TIMES_DST (0x00000008)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_MATCH_SELECT      27:24
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_MATCH_SELECT_ONE  (0x00000001)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_MATCH_SELECT_K2   (0x00000003)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_MATCH_SELECT_NEG_K1_TIMES_SRC (0x00000007)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_NO_MATCH_SELECT   31:28
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_NO_MATCH_SELECT_ZERO (0x00000000)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_NO_MATCH_SELECT_ONE (0x00000001)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_NO_MATCH_SELECT_K2 (0x00000003)
#define LWC67E_SET_COMPOSITION_FACTOR_SELECT_DST_ALPHA_FACTOR_NO_MATCH_SELECT_NEG_K1_TIMES_SRC (0x00000007)
#define LWC67E_SET_KEY_ALPHA                                                    (0x000002F8)
#define LWC67E_SET_KEY_ALPHA_MIN                                                15:0
#define LWC67E_SET_KEY_ALPHA_MAX                                                31:16
#define LWC67E_SET_KEY_RED_CR                                                   (0x000002FC)
#define LWC67E_SET_KEY_RED_CR_MIN                                               15:0
#define LWC67E_SET_KEY_RED_CR_MAX                                               31:16
#define LWC67E_SET_KEY_GREEN_Y                                                  (0x00000300)
#define LWC67E_SET_KEY_GREEN_Y_MIN                                              15:0
#define LWC67E_SET_KEY_GREEN_Y_MAX                                              31:16
#define LWC67E_SET_KEY_BLUE_CB                                                  (0x00000304)
#define LWC67E_SET_KEY_BLUE_CB_MIN                                              15:0
#define LWC67E_SET_KEY_BLUE_CB_MAX                                              31:16
#define LWC67E_SET_PRESENT_CONTROL                                              (0x00000308)
#define LWC67E_SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL                         3:0
#define LWC67E_SET_PRESENT_CONTROL_BEGIN_MODE                                   6:4
#define LWC67E_SET_PRESENT_CONTROL_BEGIN_MODE_NON_TEARING                       (0x00000000)
#define LWC67E_SET_PRESENT_CONTROL_BEGIN_MODE_IMMEDIATE                         (0x00000001)
#define LWC67E_SET_PRESENT_CONTROL_TIMESTAMP_MODE                               8:8
#define LWC67E_SET_PRESENT_CONTROL_TIMESTAMP_MODE_DISABLE                       (0x00000000)
#define LWC67E_SET_PRESENT_CONTROL_TIMESTAMP_MODE_ENABLE                        (0x00000001)
#define LWC67E_SET_PRESENT_CONTROL_STEREO_MODE                                  13:12
#define LWC67E_SET_PRESENT_CONTROL_STEREO_MODE_MONO                             (0x00000000)
#define LWC67E_SET_PRESENT_CONTROL_STEREO_MODE_PAIR_FLIP                        (0x00000001)
#define LWC67E_SET_PRESENT_CONTROL_STEREO_MODE_AT_ANY_FRAME                     (0x00000002)
#define LWC67E_SET_ACQ_SEMAPHORE_VALUE_HI                                       (0x0000030C)
#define LWC67E_SET_ACQ_SEMAPHORE_VALUE_HI_VALUE                                 31:0
#define LWC67E_SET_OPAQUE_POINT_IN(b)                                           (0x00000310 + (b)*0x00000004)
#define LWC67E_SET_OPAQUE_POINT_IN_X                                            14:0
#define LWC67E_SET_OPAQUE_POINT_IN_Y                                            30:16
#define LWC67E_SET_OPAQUE_SIZE_IN(b)                                            (0x00000320 + (b)*0x00000004)
#define LWC67E_SET_OPAQUE_SIZE_IN_WIDTH                                         14:0
#define LWC67E_SET_OPAQUE_SIZE_IN_HEIGHT                                        30:16
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL                                        (0x00000330)
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_OFFSET                                 7:0
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_PAYLOAD_SIZE                           15:15
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_PAYLOAD_SIZE_PAYLOAD_32BIT             (0x00000000)
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_PAYLOAD_SIZE_PAYLOAD_64BIT             (0x00000001)
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_ACQ_MODE                               13:12
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_ACQ_MODE_EQ                            (0x00000000)
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_ACQ_MODE_CGEQ                          (0x00000001)
#define LWC67E_SET_ACQ_SEMAPHORE_CONTROL_ACQ_MODE_STRICT_GEQ                    (0x00000002)
#define LWC67E_SET_ACQ_SEMAPHORE_VALUE                                          (0x00000334)
#define LWC67E_SET_ACQ_SEMAPHORE_VALUE_VALUE                                    31:0
#define LWC67E_SET_CONTEXT_DMA_ACQ_SEMAPHORE                                    (0x00000338)
#define LWC67E_SET_CONTEXT_DMA_ACQ_SEMAPHORE_HANDLE                             31:0
#define LWC67E_SET_SCAN_DIRECTION                                               (0x0000033C)
#define LWC67E_SET_SCAN_DIRECTION_HORIZONTAL_DIRECTION                          0:0
#define LWC67E_SET_SCAN_DIRECTION_HORIZONTAL_DIRECTION_FROM_LEFT                (0x00000000)
#define LWC67E_SET_SCAN_DIRECTION_HORIZONTAL_DIRECTION_FROM_RIGHT               (0x00000001)
#define LWC67E_SET_SCAN_DIRECTION_VERTICAL_DIRECTION                            1:1
#define LWC67E_SET_SCAN_DIRECTION_VERTICAL_DIRECTION_FROM_TOP                   (0x00000000)
#define LWC67E_SET_SCAN_DIRECTION_VERTICAL_DIRECTION_FROM_BOTTOM                (0x00000001)
#define LWC67E_SET_SCAN_DIRECTION_COLUMN_ORDER                                  2:2
#define LWC67E_SET_SCAN_DIRECTION_COLUMN_ORDER_FALSE                            (0x00000000)
#define LWC67E_SET_SCAN_DIRECTION_COLUMN_ORDER_TRUE                             (0x00000001)
#define LWC67E_SET_TIMESTAMP_ORIGIN_LO                                          (0x00000340)
#define LWC67E_SET_TIMESTAMP_ORIGIN_LO_TIMESTAMP_LO                             31:0
#define LWC67E_SET_TIMESTAMP_ORIGIN_HI                                          (0x00000344)
#define LWC67E_SET_TIMESTAMP_ORIGIN_HI_TIMESTAMP_HI                             31:0
#define LWC67E_SET_UPDATE_TIMESTAMP_LO                                          (0x00000348)
#define LWC67E_SET_UPDATE_TIMESTAMP_LO_TIMESTAMP_LO                             31:0
#define LWC67E_SET_UPDATE_TIMESTAMP_HI                                          (0x0000034C)
#define LWC67E_SET_UPDATE_TIMESTAMP_HI_TIMESTAMP_HI                             31:0
#define LWC67E_SYNC_POINT_RELEASE                                               (0x00000350)
#define LWC67E_SYNC_POINT_RELEASE_INDEX                                         7:0
#define LWC67E_SYNC_POINT_RELEASE_EVENT                                         11:8
#define LWC67E_SYNC_POINT_RELEASE_EVENT_NONE                                    (0x00000000)
#define LWC67E_SYNC_POINT_RELEASE_EVENT_BUFFER_DONE                             (0x00000001)
#define LWC67E_SET_SYNC_POINT_CONTROL                                           (0x00000354)
#define LWC67E_SET_SYNC_POINT_CONTROL_ENABLE                                    0:0
#define LWC67E_SET_SYNC_POINT_CONTROL_ENABLE_DISABLE                            (0x00000000)
#define LWC67E_SET_SYNC_POINT_CONTROL_ENABLE_ENABLE                             (0x00000001)
#define LWC67E_SET_SYNC_POINT_CONTROL_INDEX                                     8:1
#define LWC67E_SET_SYNC_POINT_ACQUIRE                                           (0x00000358)
#define LWC67E_SET_SYNC_POINT_ACQUIRE_VALUE                                     31:0
#define LWC67E_CDE_CONTROL                                                      (0x00000364)
#define LWC67E_CDE_CONTROL_ENABLE_SURFACE                                       0:0
#define LWC67E_CDE_CONTROL_ENABLE_SURFACE_DISABLE                               (0x00000000)
#define LWC67E_CDE_CONTROL_ENABLE_SURFACE_ENABLE                                (0x00000001)
#define LWC67E_CDE_CONTROL_KIND                                                 7:4
#define LWC67E_CDE_CONTROL_KIND_CRA                                             (0x00000000)
#define LWC67E_CDE_CONTROL_KIND_BRA                                             (0x00000001)
#define LWC67E_CDE_CONTROL_KIND_YUV_8B_1C                                       (0x00000002)
#define LWC67E_CDE_CONTROL_KIND_YUV_8B_2C                                       (0x00000003)
#define LWC67E_CDE_CONTROL_KIND_YUV_10B_1C                                      (0x00000004)
#define LWC67E_CDE_CONTROL_KIND_YUV_10B_2C                                      (0x00000005)
#define LWC67E_CDE_CONTROL_KIND_YUV_12B_1C                                      (0x00000006)
#define LWC67E_CDE_CONTROL_KIND_YUV_12B_2C                                      (0x00000007)
#define LWC67E_CDE_CONTROL_TRAVERSAL_PATTERN                                    1:1
#define LWC67E_CDE_CONTROL_TRAVERSAL_PATTERN_FIXED                              (0x00000000)
#define LWC67E_CDE_CONTROL_TRAVERSAL_PATTERN_RANDOM                             (0x00000001)
#define LWC67E_CDE_CONTROL_CTB_PITCH                                            9:8
#define LWC67E_CDE_CONTROL_CTB_PITCH_S_32B                                      (0x00000000)
#define LWC67E_CDE_CONTROL_CTB_PITCH_S_64B                                      (0x00000001)
#define LWC67E_CDE_CONTROL_CTB_PITCH_S_128B                                     (0x00000002)
#define LWC67E_CDE_CONTROL_CTB_PITCH_S_256B                                     (0x00000003)
#define LWC67E_CDE_CONTROL_INIT_MODE                                            12:12
#define LWC67E_CDE_CONTROL_INIT_MODE_CLEAR                                      (0x00000000)
#define LWC67E_CDE_CONTROL_INIT_MODE_MEMORY                                     (0x00000001)
#define LWC67E_CDE_CTB_ENTRY                                                    (0x00000368)
#define LWC67E_CDE_CTB_ENTRY_MAX_ENTRY_COUNT                                    31:0
#define LWC67E_CDE_ZBC_COLOR                                                    (0x0000036C)
#define LWC67E_CDE_ZBC_COLOR_PIXEL_COLOR                                        31:0
#define LWC67E_SET_INTERLOCK_FLAGS                                              (0x00000370)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_CORE                          0:0
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_CORE_DISABLE                  (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_CORE_ENABLE                   (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR(i)                     ((i)+1):((i)+1)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR__SIZE_1                8
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR_DISABLE                (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR_ENABLE                 (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR0                       1:1
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR0_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR0_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR1                       2:2
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR1_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR1_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR2                       3:3
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR2_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR2_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR3                       4:4
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR3_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR3_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR4                       5:5
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR4_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR4_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR5                       6:6
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR5_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR5_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR6                       7:7
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR6_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR6_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR7                       8:8
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR7_DISABLE               (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_LWRSOR7_ENABLE                (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK(i)                  ((i)+9):((i)+9)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK__SIZE_1             8
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK_DISABLE             (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK_ENABLE              (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK0                    9:9
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK0_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK0_ENABLE             (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK1                    10:10
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK1_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK1_ENABLE             (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK2                    11:11
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK2_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK2_ENABLE             (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK3                    12:12
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK3_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK3_ENABLE             (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK4                    13:13
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK4_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK4_ENABLE             (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK5                    14:14
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK5_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK5_ENABLE             (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK6                    15:15
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK6_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK6_ENABLE             (0x00000001)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK7                    16:16
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK7_DISABLE            (0x00000000)
#define LWC67E_SET_INTERLOCK_FLAGS_INTERLOCK_WITH_WRITEBACK7_ENABLE             (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS                                       (0x00000374)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW(i)              ((i)+0):((i)+0)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW__SIZE_1         32
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW_DISABLE         (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW_ENABLE          (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW0                0:0
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW0_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW0_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW1                1:1
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW1_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW1_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW2                2:2
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW2_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW2_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW3                3:3
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW3_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW3_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW4                4:4
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW4_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW4_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW5                5:5
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW5_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW5_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW6                6:6
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW6_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW6_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW7                7:7
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW7_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW7_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW8                8:8
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW8_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW8_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW9                9:9
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW9_DISABLE        (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW9_ENABLE         (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW10               10:10
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW10_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW10_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW11               11:11
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW11_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW11_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW12               12:12
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW12_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW12_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW13               13:13
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW13_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW13_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW14               14:14
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW14_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW14_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW15               15:15
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW15_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW15_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW16               16:16
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW16_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW16_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW17               17:17
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW17_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW17_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW18               18:18
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW18_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW18_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW19               19:19
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW19_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW19_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW20               20:20
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW20_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW20_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW21               21:21
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW21_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW21_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW22               22:22
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW22_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW22_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW23               23:23
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW23_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW23_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW24               24:24
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW24_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW24_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW25               25:25
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW25_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW25_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW26               26:26
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW26_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW26_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW27               27:27
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW27_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW27_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW28               28:28
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW28_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW28_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW29               29:29
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW29_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW29_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW30               30:30
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW30_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW30_ENABLE        (0x00000001)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW31               31:31
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW31_DISABLE       (0x00000000)
#define LWC67E_SET_WINDOW_INTERLOCK_FLAGS_INTERLOCK_WITH_WINDOW31_ENABLE        (0x00000001)
#define LWC67E_CDE_COMPTAG_BASE(b)                                              (0x00000380 + (b)*0x00000004)
#define LWC67E_CDE_COMPTAG_BASE_ADDRESS                                         31:0
#define LWC67E_SET_EXT_PACKET_CONTROL                                           (0x00000398)
#define LWC67E_SET_EXT_PACKET_CONTROL_ENABLE                                    0:0
#define LWC67E_SET_EXT_PACKET_CONTROL_ENABLE_DISABLE                            (0x00000000)
#define LWC67E_SET_EXT_PACKET_CONTROL_ENABLE_ENABLE                             (0x00000001)
#define LWC67E_SET_EXT_PACKET_CONTROL_LOCATION                                  4:4
#define LWC67E_SET_EXT_PACKET_CONTROL_LOCATION_VSYNC                            (0x00000000)
#define LWC67E_SET_EXT_PACKET_CONTROL_LOCATION_VBLANK                           (0x00000001)
#define LWC67E_SET_EXT_PACKET_CONTROL_FREQUENCY                                 8:8
#define LWC67E_SET_EXT_PACKET_CONTROL_FREQUENCY_EVERY_FRAME                     (0x00000000)
#define LWC67E_SET_EXT_PACKET_CONTROL_FREQUENCY_ONCE                            (0x00000001)
#define LWC67E_SET_EXT_PACKET_CONTROL_HEADER_OVERRIDE                           12:12
#define LWC67E_SET_EXT_PACKET_CONTROL_HEADER_OVERRIDE_DISABLE                   (0x00000000)
#define LWC67E_SET_EXT_PACKET_CONTROL_HEADER_OVERRIDE_ENABLE                    (0x00000001)
#define LWC67E_SET_EXT_PACKET_CONTROL_SIZE                                      27:16
#define LWC67E_SET_EXT_PACKET_DATA                                              (0x0000039C)
#define LWC67E_SET_EXT_PACKET_DATA_DB0                                          7:0
#define LWC67E_SET_EXT_PACKET_DATA_DB1                                          15:8
#define LWC67E_SET_EXT_PACKET_DATA_DB2                                          23:16
#define LWC67E_SET_EXT_PACKET_DATA_DB3                                          31:24
#define LWC67E_SET_FMT_COEFFICIENT_C00                                          (0x00000400)
#define LWC67E_SET_FMT_COEFFICIENT_C00_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C01                                          (0x00000404)
#define LWC67E_SET_FMT_COEFFICIENT_C01_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C02                                          (0x00000408)
#define LWC67E_SET_FMT_COEFFICIENT_C02_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C03                                          (0x0000040C)
#define LWC67E_SET_FMT_COEFFICIENT_C03_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C10                                          (0x00000410)
#define LWC67E_SET_FMT_COEFFICIENT_C10_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C11                                          (0x00000414)
#define LWC67E_SET_FMT_COEFFICIENT_C11_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C12                                          (0x00000418)
#define LWC67E_SET_FMT_COEFFICIENT_C12_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C13                                          (0x0000041C)
#define LWC67E_SET_FMT_COEFFICIENT_C13_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C20                                          (0x00000420)
#define LWC67E_SET_FMT_COEFFICIENT_C20_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C21                                          (0x00000424)
#define LWC67E_SET_FMT_COEFFICIENT_C21_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C22                                          (0x00000428)
#define LWC67E_SET_FMT_COEFFICIENT_C22_VALUE                                    20:0
#define LWC67E_SET_FMT_COEFFICIENT_C23                                          (0x0000042C)
#define LWC67E_SET_FMT_COEFFICIENT_C23_VALUE                                    20:0
#define LWC67E_SET_ILUT_CONTROL                                                 (0x00000440)
#define LWC67E_SET_ILUT_CONTROL_INTERPOLATE                                     0:0
#define LWC67E_SET_ILUT_CONTROL_INTERPOLATE_DISABLE                             (0x00000000)
#define LWC67E_SET_ILUT_CONTROL_INTERPOLATE_ENABLE                              (0x00000001)
#define LWC67E_SET_ILUT_CONTROL_MIRROR                                          1:1
#define LWC67E_SET_ILUT_CONTROL_MIRROR_DISABLE                                  (0x00000000)
#define LWC67E_SET_ILUT_CONTROL_MIRROR_ENABLE                                   (0x00000001)
#define LWC67E_SET_ILUT_CONTROL_MODE                                            3:2
#define LWC67E_SET_ILUT_CONTROL_MODE_SEGMENTED                                  (0x00000000)
#define LWC67E_SET_ILUT_CONTROL_MODE_DIRECT8                                    (0x00000001)
#define LWC67E_SET_ILUT_CONTROL_MODE_DIRECT10                                   (0x00000002)
#define LWC67E_SET_ILUT_CONTROL_SIZE                                            18:8
#define LWC67E_SET_CONTEXT_DMA_ILUT                                             (0x00000444)
#define LWC67E_SET_CONTEXT_DMA_ILUT_HANDLE                                      31:0
#define LWC67E_SET_OFFSET_ILUT                                                  (0x00000448)
#define LWC67E_SET_OFFSET_ILUT_ORIGIN                                           31:0
#define LWC67E_SET_CSC00CONTROL                                                 (0x0000045C)
#define LWC67E_SET_CSC00CONTROL_ENABLE                                          0:0
#define LWC67E_SET_CSC00CONTROL_ENABLE_DISABLE                                  (0x00000000)
#define LWC67E_SET_CSC00CONTROL_ENABLE_ENABLE                                   (0x00000001)
#define LWC67E_SET_CSC00COEFFICIENT_C00                                         (0x00000460)
#define LWC67E_SET_CSC00COEFFICIENT_C00_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C01                                         (0x00000464)
#define LWC67E_SET_CSC00COEFFICIENT_C01_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C02                                         (0x00000468)
#define LWC67E_SET_CSC00COEFFICIENT_C02_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C03                                         (0x0000046C)
#define LWC67E_SET_CSC00COEFFICIENT_C03_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C10                                         (0x00000470)
#define LWC67E_SET_CSC00COEFFICIENT_C10_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C11                                         (0x00000474)
#define LWC67E_SET_CSC00COEFFICIENT_C11_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C12                                         (0x00000478)
#define LWC67E_SET_CSC00COEFFICIENT_C12_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C13                                         (0x0000047C)
#define LWC67E_SET_CSC00COEFFICIENT_C13_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C20                                         (0x00000480)
#define LWC67E_SET_CSC00COEFFICIENT_C20_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C21                                         (0x00000484)
#define LWC67E_SET_CSC00COEFFICIENT_C21_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C22                                         (0x00000488)
#define LWC67E_SET_CSC00COEFFICIENT_C22_VALUE                                   20:0
#define LWC67E_SET_CSC00COEFFICIENT_C23                                         (0x0000048C)
#define LWC67E_SET_CSC00COEFFICIENT_C23_VALUE                                   20:0
#define LWC67E_SET_CSC0LUT_CONTROL                                              (0x000004A0)
#define LWC67E_SET_CSC0LUT_CONTROL_INTERPOLATE                                  0:0
#define LWC67E_SET_CSC0LUT_CONTROL_INTERPOLATE_DISABLE                          (0x00000000)
#define LWC67E_SET_CSC0LUT_CONTROL_INTERPOLATE_ENABLE                           (0x00000001)
#define LWC67E_SET_CSC0LUT_CONTROL_MIRROR                                       1:1
#define LWC67E_SET_CSC0LUT_CONTROL_MIRROR_DISABLE                               (0x00000000)
#define LWC67E_SET_CSC0LUT_CONTROL_MIRROR_ENABLE                                (0x00000001)
#define LWC67E_SET_CSC0LUT_CONTROL_ENABLE                                       4:4
#define LWC67E_SET_CSC0LUT_CONTROL_ENABLE_DISABLE                               (0x00000000)
#define LWC67E_SET_CSC0LUT_CONTROL_ENABLE_ENABLE                                (0x00000001)
#define LWC67E_SET_CSC0LUT_SEGMENT_SIZE                                         (0x000004A4)
#define LWC67E_SET_CSC0LUT_SEGMENT_SIZE_IDX                                     5:0
#define LWC67E_SET_CSC0LUT_SEGMENT_SIZE_VALUE                                   18:16
#define LWC67E_SET_CSC0LUT_ENTRY                                                (0x000004A8)
#define LWC67E_SET_CSC0LUT_ENTRY_IDX                                            10:0
#define LWC67E_SET_CSC0LUT_ENTRY_VALUE                                          31:16
#define LWC67E_SET_CSC01CONTROL                                                 (0x000004BC)
#define LWC67E_SET_CSC01CONTROL_ENABLE                                          0:0
#define LWC67E_SET_CSC01CONTROL_ENABLE_DISABLE                                  (0x00000000)
#define LWC67E_SET_CSC01CONTROL_ENABLE_ENABLE                                   (0x00000001)
#define LWC67E_SET_CSC01COEFFICIENT_C00                                         (0x000004C0)
#define LWC67E_SET_CSC01COEFFICIENT_C00_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C01                                         (0x000004C4)
#define LWC67E_SET_CSC01COEFFICIENT_C01_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C02                                         (0x000004C8)
#define LWC67E_SET_CSC01COEFFICIENT_C02_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C03                                         (0x000004CC)
#define LWC67E_SET_CSC01COEFFICIENT_C03_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C10                                         (0x000004D0)
#define LWC67E_SET_CSC01COEFFICIENT_C10_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C11                                         (0x000004D4)
#define LWC67E_SET_CSC01COEFFICIENT_C11_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C12                                         (0x000004D8)
#define LWC67E_SET_CSC01COEFFICIENT_C12_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C13                                         (0x000004DC)
#define LWC67E_SET_CSC01COEFFICIENT_C13_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C20                                         (0x000004E0)
#define LWC67E_SET_CSC01COEFFICIENT_C20_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C21                                         (0x000004E4)
#define LWC67E_SET_CSC01COEFFICIENT_C21_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C22                                         (0x000004E8)
#define LWC67E_SET_CSC01COEFFICIENT_C22_VALUE                                   20:0
#define LWC67E_SET_CSC01COEFFICIENT_C23                                         (0x000004EC)
#define LWC67E_SET_CSC01COEFFICIENT_C23_VALUE                                   20:0
#define LWC67E_SET_TMO_CONTROL                                                  (0x00000500)
#define LWC67E_SET_TMO_CONTROL_INTERPOLATE                                      0:0
#define LWC67E_SET_TMO_CONTROL_INTERPOLATE_DISABLE                              (0x00000000)
#define LWC67E_SET_TMO_CONTROL_INTERPOLATE_ENABLE                               (0x00000001)
#define LWC67E_SET_TMO_CONTROL_SAT_MODE                                         3:2
#define LWC67E_SET_TMO_CONTROL_SIZE                                             18:8
#define LWC67E_SET_TMO_LOW_INTENSITY_ZONE                                       (0x00000508)
#define LWC67E_SET_TMO_LOW_INTENSITY_ZONE_END                                   29:16
#define LWC67E_SET_TMO_LOW_INTENSITY_VALUE                                      (0x0000050C)
#define LWC67E_SET_TMO_LOW_INTENSITY_VALUE_LIN_WEIGHT                           8:0
#define LWC67E_SET_TMO_LOW_INTENSITY_VALUE_NON_LIN_WEIGHT                       20:12
#define LWC67E_SET_TMO_LOW_INTENSITY_VALUE_THRESHOLD                            31:24
#define LWC67E_SET_TMO_MEDIUM_INTENSITY_ZONE                                    (0x00000510)
#define LWC67E_SET_TMO_MEDIUM_INTENSITY_ZONE_START                              13:0
#define LWC67E_SET_TMO_MEDIUM_INTENSITY_ZONE_END                                29:16
#define LWC67E_SET_TMO_MEDIUM_INTENSITY_VALUE                                   (0x00000514)
#define LWC67E_SET_TMO_MEDIUM_INTENSITY_VALUE_LIN_WEIGHT                        8:0
#define LWC67E_SET_TMO_MEDIUM_INTENSITY_VALUE_NON_LIN_WEIGHT                    20:12
#define LWC67E_SET_TMO_MEDIUM_INTENSITY_VALUE_THRESHOLD                         31:24
#define LWC67E_SET_TMO_HIGH_INTENSITY_ZONE                                      (0x00000518)
#define LWC67E_SET_TMO_HIGH_INTENSITY_ZONE_START                                13:0
#define LWC67E_SET_TMO_HIGH_INTENSITY_VALUE                                     (0x0000051C)
#define LWC67E_SET_TMO_HIGH_INTENSITY_VALUE_LIN_WEIGHT                          8:0
#define LWC67E_SET_TMO_HIGH_INTENSITY_VALUE_NON_LIN_WEIGHT                      20:12
#define LWC67E_SET_TMO_HIGH_INTENSITY_VALUE_THRESHOLD                           31:24
#define LWC67E_SET_CONTEXT_DMA_TMO_LUT                                          (0x00000528)
#define LWC67E_SET_CONTEXT_DMA_TMO_LUT_HANDLE                                   31:0
#define LWC67E_SET_OFFSET_TMO_LUT                                               (0x0000052C)
#define LWC67E_SET_OFFSET_TMO_LUT_ORIGIN                                        31:0
#define LWC67E_SET_CSC10CONTROL                                                 (0x0000053C)
#define LWC67E_SET_CSC10CONTROL_ENABLE                                          0:0
#define LWC67E_SET_CSC10CONTROL_ENABLE_DISABLE                                  (0x00000000)
#define LWC67E_SET_CSC10CONTROL_ENABLE_ENABLE                                   (0x00000001)
#define LWC67E_SET_CSC10COEFFICIENT_C00                                         (0x00000540)
#define LWC67E_SET_CSC10COEFFICIENT_C00_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C01                                         (0x00000544)
#define LWC67E_SET_CSC10COEFFICIENT_C01_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C02                                         (0x00000548)
#define LWC67E_SET_CSC10COEFFICIENT_C02_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C03                                         (0x0000054C)
#define LWC67E_SET_CSC10COEFFICIENT_C03_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C10                                         (0x00000550)
#define LWC67E_SET_CSC10COEFFICIENT_C10_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C11                                         (0x00000554)
#define LWC67E_SET_CSC10COEFFICIENT_C11_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C12                                         (0x00000558)
#define LWC67E_SET_CSC10COEFFICIENT_C12_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C13                                         (0x0000055C)
#define LWC67E_SET_CSC10COEFFICIENT_C13_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C20                                         (0x00000560)
#define LWC67E_SET_CSC10COEFFICIENT_C20_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C21                                         (0x00000564)
#define LWC67E_SET_CSC10COEFFICIENT_C21_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C22                                         (0x00000568)
#define LWC67E_SET_CSC10COEFFICIENT_C22_VALUE                                   20:0
#define LWC67E_SET_CSC10COEFFICIENT_C23                                         (0x0000056C)
#define LWC67E_SET_CSC10COEFFICIENT_C23_VALUE                                   20:0
#define LWC67E_SET_CSC1LUT_CONTROL                                              (0x00000580)
#define LWC67E_SET_CSC1LUT_CONTROL_INTERPOLATE                                  0:0
#define LWC67E_SET_CSC1LUT_CONTROL_INTERPOLATE_DISABLE                          (0x00000000)
#define LWC67E_SET_CSC1LUT_CONTROL_INTERPOLATE_ENABLE                           (0x00000001)
#define LWC67E_SET_CSC1LUT_CONTROL_MIRROR                                       1:1
#define LWC67E_SET_CSC1LUT_CONTROL_MIRROR_DISABLE                               (0x00000000)
#define LWC67E_SET_CSC1LUT_CONTROL_MIRROR_ENABLE                                (0x00000001)
#define LWC67E_SET_CSC1LUT_CONTROL_ENABLE                                       4:4
#define LWC67E_SET_CSC1LUT_CONTROL_ENABLE_DISABLE                               (0x00000000)
#define LWC67E_SET_CSC1LUT_CONTROL_ENABLE_ENABLE                                (0x00000001)
#define LWC67E_SET_CSC1LUT_SEGMENT_SIZE                                         (0x00000584)
#define LWC67E_SET_CSC1LUT_SEGMENT_SIZE_IDX                                     5:0
#define LWC67E_SET_CSC1LUT_SEGMENT_SIZE_VALUE                                   18:16
#define LWC67E_SET_CSC1LUT_ENTRY                                                (0x00000588)
#define LWC67E_SET_CSC1LUT_ENTRY_IDX                                            10:0
#define LWC67E_SET_CSC1LUT_ENTRY_VALUE                                          31:16
#define LWC67E_SET_CSC11CONTROL                                                 (0x0000059C)
#define LWC67E_SET_CSC11CONTROL_ENABLE                                          0:0
#define LWC67E_SET_CSC11CONTROL_ENABLE_DISABLE                                  (0x00000000)
#define LWC67E_SET_CSC11CONTROL_ENABLE_ENABLE                                   (0x00000001)
#define LWC67E_SET_CSC11COEFFICIENT_C00                                         (0x000005A0)
#define LWC67E_SET_CSC11COEFFICIENT_C00_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C01                                         (0x000005A4)
#define LWC67E_SET_CSC11COEFFICIENT_C01_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C02                                         (0x000005A8)
#define LWC67E_SET_CSC11COEFFICIENT_C02_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C03                                         (0x000005AC)
#define LWC67E_SET_CSC11COEFFICIENT_C03_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C10                                         (0x000005B0)
#define LWC67E_SET_CSC11COEFFICIENT_C10_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C11                                         (0x000005B4)
#define LWC67E_SET_CSC11COEFFICIENT_C11_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C12                                         (0x000005B8)
#define LWC67E_SET_CSC11COEFFICIENT_C12_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C13                                         (0x000005BC)
#define LWC67E_SET_CSC11COEFFICIENT_C13_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C20                                         (0x000005C0)
#define LWC67E_SET_CSC11COEFFICIENT_C20_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C21                                         (0x000005C4)
#define LWC67E_SET_CSC11COEFFICIENT_C21_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C22                                         (0x000005C8)
#define LWC67E_SET_CSC11COEFFICIENT_C22_VALUE                                   20:0
#define LWC67E_SET_CSC11COEFFICIENT_C23                                         (0x000005CC)
#define LWC67E_SET_CSC11COEFFICIENT_C23_VALUE                                   20:0
#define LWC67E_SET_CLAMP_RANGE                                                  (0x000005D0)
#define LWC67E_SET_CLAMP_RANGE_LOW                                              15:0
#define LWC67E_SET_CLAMP_RANGE_HIGH                                             31:16
#define LWC67E_SW_RESERVED(b)                                                   (0x000005D4 + (b)*0x00000004)
#define LWC67E_SW_RESERVED_VALUE                                                31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clC67e_h
