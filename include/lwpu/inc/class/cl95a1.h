// WARNING!!! THIS HEADER INCLUDES SOFTWARE METHODS!!!
// ********** DO NOT USE IN HW TREE.  ********** 
/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#include "lwtypes.h"

#ifndef _cl95a1_h_
#define _cl95a1_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW95A1_TSEC                                                                (0x000095A1)

typedef volatile struct _cl95a1_tag0 {
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
    LwV32 SetSemaphorePayloadLower;                                             // 0x00000308 - 0x0000030B
    LwV32 SetSemaphorePayloadUpper;                                             // 0x0000030C - 0x0000030F
    LwV32 Reserved05[0x7C];
    LwV32 HdcpInit;                                                             // 0x00000500 - 0x00000503
    LwV32 HdcpCreateSession;                                                    // 0x00000504 - 0x00000507
    LwV32 HdcpVerifyCertRx;                                                     // 0x00000508 - 0x0000050B
    LwV32 HdcpGenerateEKm;                                                      // 0x0000050C - 0x0000050F
    LwV32 HdcpRevocationCheck;                                                  // 0x00000510 - 0x00000513
    LwV32 HdcpVerifyHprime;                                                     // 0x00000514 - 0x00000517
    LwV32 HdcpEncryptPairingInfo;                                               // 0x00000518 - 0x0000051B
    LwV32 HdcpDecryptPairingInfo;                                               // 0x0000051C - 0x0000051F
    LwV32 HdcpUpdateSession;                                                    // 0x00000520 - 0x00000523
    LwV32 HdcpGenerateLcInit;                                                   // 0x00000524 - 0x00000527
    LwV32 HdcpVerifyLprime;                                                     // 0x00000528 - 0x0000052B
    LwV32 HdcpGenerateSkeInit;                                                  // 0x0000052C - 0x0000052F
    LwV32 HdcpVerifyVprime;                                                     // 0x00000530 - 0x00000533
    LwV32 HdcpEncryptionRunCtrl;                                                // 0x00000534 - 0x00000537
    LwV32 HdcpSessionCtrl;                                                      // 0x00000538 - 0x0000053B
    LwV32 HdcpComputeSprime;                                                    // 0x0000053C - 0x0000053F
    LwV32 HdcpGetCertRx;                                                        // 0x00000540 - 0x00000543
    LwV32 HdcpExchangeInfo;                                                     // 0x00000544 - 0x00000547
    LwV32 HdcpDecryptKm;                                                        // 0x00000548 - 0x0000054B
    LwV32 HdcpGetHprime;                                                        // 0x0000054C - 0x0000054F
    LwV32 HdcpGenerateEKhKm;                                                    // 0x00000550 - 0x00000553
    LwV32 HdcpVerifyRttChallenge;                                               // 0x00000554 - 0x00000557
    LwV32 HdcpGetLprime;                                                        // 0x00000558 - 0x0000055B
    LwV32 HdcpDecryptKs;                                                        // 0x0000055C - 0x0000055F
    LwV32 HdcpDecrypt;                                                          // 0x00000560 - 0x00000563
    LwV32 HdcpGetRrx;                                                           // 0x00000564 - 0x00000567
    LwV32 HdcpDecryptReEncrypt;                                                 // 0x00000568 - 0x0000056B
    LwV32 HdcpAesEcbCrypt;                                                      // 0x0000056C - 0x0000056F
    LwV32 Reserved06[0x24];
    LwV32 LwsiSwapAreaOffset;                                                   // 0x00000600 - 0x00000603
    LwV32 LwsiSwapAreaSize;                                                     // 0x00000604 - 0x00000607
    LwV32 LwsiShaderCodeOffset;                                                 // 0x00000608 - 0x0000060B
    LwV32 LwsiShaderCodeSize;                                                   // 0x0000060C - 0x0000060F
    LwV32 LwsiLoader2CodeSize;                                                  // 0x00000610 - 0x00000613
    LwV32 LwsiShaderDataOffset;                                                 // 0x00000614 - 0x00000617
    LwV32 LwsiShaderDataSize;                                                   // 0x00000618 - 0x0000061B
    LwV32 LwsiShaderVolDataSize;                                                // 0x0000061C - 0x0000061F
    LwV32 LwsiSharedSegOffset;                                                  // 0x00000620 - 0x00000623
    LwV32 LwsiSharedSegSize;                                                    // 0x00000624 - 0x00000627
    LwV32 LwsiMiscDataOffset;                                                   // 0x00000628 - 0x0000062B
    LwV32 LwsiMiscDataSize;                                                     // 0x0000062C - 0x0000062F
    LwV32 Reserved07[0x34];
    LwV32 HdcpValidateSrm;                                                      // 0x00000700 - 0x00000703
    LwV32 HdcpValidateStream;                                                   // 0x00000704 - 0x00000707
    LwV32 HdcpTestSelwreStatus;                                                 // 0x00000708 - 0x0000070B
    LwV32 HdcpSetDcpKpub;                                                       // 0x0000070C - 0x0000070F
    LwV32 HdcpSetRxKpub;                                                        // 0x00000710 - 0x00000713
    LwV32 HdcpSetCertRx;                                                        // 0x00000714 - 0x00000717
    LwV32 HdcpSetScratchBuffer;                                                 // 0x00000718 - 0x0000071B
    LwV32 HdcpSetSrm;                                                           // 0x0000071C - 0x0000071F
    LwV32 HdcpSetReceiverIdList;                                                // 0x00000720 - 0x00000723
    LwV32 HdcpSetSprime;                                                        // 0x00000724 - 0x00000727
    LwV32 HdcpSetEncInputBuffer;                                                // 0x00000728 - 0x0000072B
    LwV32 HdcpSetEncOutputBuffer;                                               // 0x0000072C - 0x0000072F
    LwV32 HdcpGetRttChallenge;                                                  // 0x00000730 - 0x00000733
    LwV32 HdcpStreamManage;                                                     // 0x00000734 - 0x00000737
    LwV32 HdcpReadCaps;                                                         // 0x00000738 - 0x0000073B
    LwV32 HdcpEncrypt;                                                          // 0x0000073C - 0x0000073F
    LwV32 Reserved08[0x130];
    LwV32 SetContentInitialVector[4];                                           // 0x00000C00 - 0x00000C0F
    LwV32 SetCtlCount;                                                          // 0x00000C10 - 0x00000C13
    LwV32 SetMdecH2MKey;                                                        // 0x00000C14 - 0x00000C17
    LwV32 SetMdecM2HKey;                                                        // 0x00000C18 - 0x00000C1B
    LwV32 SetMdecFrameKey;                                                      // 0x00000C1C - 0x00000C1F
    LwV32 SetUpperSrc;                                                          // 0x00000C20 - 0x00000C23
    LwV32 SetLowerSrc;                                                          // 0x00000C24 - 0x00000C27
    LwV32 SetUpperDst;                                                          // 0x00000C28 - 0x00000C2B
    LwV32 SetLowerDst;                                                          // 0x00000C2C - 0x00000C2F
    LwV32 SetUpperCtl;                                                          // 0x00000C30 - 0x00000C33
    LwV32 SetLowerCtl;                                                          // 0x00000C34 - 0x00000C37
    LwV32 SetBlockCount;                                                        // 0x00000C38 - 0x00000C3B
    LwV32 SetStretchMask;                                                       // 0x00000C3C - 0x00000C3F
    LwV32 Reserved09[0x30];
    LwV32 SetUpperFlowCtrlInselwre;                                             // 0x00000D00 - 0x00000D03
    LwV32 SetLowerFlowCtrlInselwre;                                             // 0x00000D04 - 0x00000D07
    LwV32 Reserved10[0x2];
    LwV32 SetUcodeLoaderParams;                                                 // 0x00000D10 - 0x00000D13
    LwV32 Reserved11[0x1];
    LwV32 SetUpperFlowCtrlSelwre;                                               // 0x00000D18 - 0x00000D1B
    LwV32 SetLowerFlowCtrlSelwre;                                               // 0x00000D1C - 0x00000D1F
    LwV32 Reserved12[0x5];
    LwV32 SetUcodeLoaderOffset;                                                 // 0x00000D34 - 0x00000D37
    LwV32 Reserved13[0x72];
    LwV32 SetSessionKey[4];                                                     // 0x00000F00 - 0x00000F0F
    LwV32 SetContentKey[4];                                                     // 0x00000F10 - 0x00000F1F
    LwV32 Reserved14[0x7D];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved15[0x3BA];
} LW95A1_TSECControlPio;

#define LW95A1_NOP                                                              (0x00000100)
#define LW95A1_NOP_PARAMETER                                                    31:0
#define LW95A1_PM_TRIGGER                                                       (0x00000140)
#define LW95A1_PM_TRIGGER_V                                                     31:0
#define LW95A1_SET_APPLICATION_ID                                               (0x00000200)
#define LW95A1_SET_APPLICATION_ID_ID                                            31:0
#define LW95A1_SET_APPLICATION_ID_ID_HDCP                                       (0x00000001)
#define LW95A1_SET_APPLICATION_ID_ID_LWSI                                       (0x00000002)
#define LW95A1_SET_APPLICATION_ID_ID_GFE                                        (0x00000003)
#define LW95A1_SET_APPLICATION_ID_ID_VPR                                        (0x00000004)
#define LW95A1_SET_APPLICATION_ID_ID_CTR64                                      (0x00000005)
#define LW95A1_SET_APPLICATION_ID_ID_PR                                         (0x00000005)
#define LW95A1_SET_APPLICATION_ID_ID_STRETCH_CTR64                              (0x00000006)
#define LW95A1_SET_APPLICATION_ID_ID_MDEC_LEGACY                                (0x00000007)
#define LW95A1_SET_APPLICATION_ID_ID_UCODE_LOADER                               (0x00000008)
#define LW95A1_SET_APPLICATION_ID_ID_HWV                                        (0x00000009)
#define LW95A1_SET_APPLICATION_ID_ID_APM                                        (0x0000000A)
#define LW95A1_SET_APPLICATION_ID_ID_LWSR                                       (0x00000007)
#define LW95A1_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW95A1_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LW95A1_SEMAPHORE_A                                                      (0x00000240)
#define LW95A1_SEMAPHORE_A_UPPER                                                7:0
#define LW95A1_SEMAPHORE_B                                                      (0x00000244)
#define LW95A1_SEMAPHORE_B_LOWER                                                31:0
#define LW95A1_SEMAPHORE_C                                                      (0x00000248)
#define LW95A1_SEMAPHORE_C_PAYLOAD                                              31:0
#define LW95A1_EXELWTE                                                          (0x00000300)
#define LW95A1_EXELWTE_NOTIFY                                                   0:0
#define LW95A1_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LW95A1_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LW95A1_EXELWTE_NOTIFY_ON                                                1:1
#define LW95A1_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LW95A1_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LW95A1_EXELWTE_AWAKEN                                                   8:8
#define LW95A1_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LW95A1_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LW95A1_EXELWTE_FLUSH                                                    9:9
#define LW95A1_EXELWTE_FLUSH_DISABLE                                            (0x00000000)
#define LW95A1_EXELWTE_FLUSH_ENABLE                                             (0x00000001)
#define LW95A1_EXELWTE_STRUCTURE_SIZE                                           11:10
#define LW95A1_EXELWTE_STRUCTURE_SIZE_ONE                                       (0x00000000)
#define LW95A1_EXELWTE_STRUCTURE_SIZE_FOUR                                      (0x00000001)  // Keeping FOUR at value 1 for backwards compatiblity
#define LW95A1_EXELWTE_STRUCTURE_SIZE_TWO                                       (0x00000002)
#define LW95A1_EXELWTE_PAYLOAD_SIZE                                             16:16
#define LW95A1_EXELWTE_PAYLOAD_SIZE_32_BIT                                      (0x00000000)
#define LW95A1_EXELWTE_PAYLOAD_SIZE_64_BIT                                      (0x00000001)
#define LW95A1_SEMAPHORE_D                                                      (0x00000304)
#define LW95A1_SEMAPHORE_D_STRUCTURE_SIZE                                       1:0
#define LW95A1_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LW95A1_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)  // Keeping FOUR at value 1 for backwards compatiblity
#define LW95A1_SEMAPHORE_D_STRUCTURE_SIZE_TWO                                   (0x00000002)
#define LW95A1_SEMAPHORE_D_PAYLOAD_SIZE                                         4:4
#define LW95A1_SEMAPHORE_D_PAYLOAD_SIZE_32_BIT                                  (0x00000000)
#define LW95A1_SEMAPHORE_D_PAYLOAD_SIZE_64_BIT                                  (0x00000001)
#define LW95A1_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LW95A1_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LW95A1_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LW95A1_SEMAPHORE_D_OPERATION                                            17:16
#define LW95A1_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LW95A1_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LW95A1_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LW95A1_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LW95A1_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LW95A1_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LW95A1_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LW95A1_SET_SEMAPHORE_PAYLOAD_UPPER                                      (0x00000308)
#define LW95A1_SET_SEMAPHORE_PAYLOAD_UPPER_PAYLOAD                              31:0
#define LW95A1_SET_SEMAPHORE_PAYLOAD_LOWER                                      (0x0000030C)
#define LW95A1_SET_SEMAPHORE_PAYLOAD_LOWER_PAYLOAD                              31:0
#define LW95A1_PR_FUNCTION_ID                                                   (0x00000400)
#define LW95A1_PR_REQUEST_MESSAGE                                               (0x00000404)
#define LW95A1_PR_REQUEST_MESSAGE_OFFSET                                        31:0
#define LW95A1_PR_REQUEST_MESSAGE_SIZE                                          (0x00000408)
#define LW95A1_PR_REQUEST_MESSAGE_SIZE_VALUE                                    31:0
#define LW95A1_PR_RESPONSE_MESSAGE                                              (0x0000040C)
#define LW95A1_PR_RESPONSE_MESSAGE_OFFSET                                       31:0
#define LW95A1_PR_RESPONSE_MESSAGE_SIZE                                         (0x00000410)
#define LW95A1_PR_RESPONSE_MESSAGE_SIZE_VALUE                                   31:0
#define LW95A1_PR_STAT                                                          (0x00000414)
#define LW95A1_PR_STAT_OFFSET                                                   31:0
#define LW95A1_HDCP_INIT                                                        (0x00000500)
#define LW95A1_HDCP_INIT_PARAM_OFFSET                                           31:0
#define LW95A1_HDCP_CREATE_SESSION                                              (0x00000504)
#define LW95A1_HDCP_CREATE_SESSION_PARAM_OFFSET                                 31:0
#define LW95A1_HDCP_VERIFY_CERT_RX                                              (0x00000508)
#define LW95A1_HDCP_VERIFY_CERT_RX_PARAM_OFFSET                                 31:0
#define LW95A1_HDCP_GENERATE_EKM                                                (0x0000050C)
#define LW95A1_HDCP_GENERATE_EKM_PARAM_OFFSET                                   31:0
#define LW95A1_HDCP_REVOCATION_CHECK                                            (0x00000510)
#define LW95A1_HDCP_REVOCATION_CHECK_PARAM_OFFSET                               31:0
#define LW95A1_HDCP_VERIFY_HPRIME                                               (0x00000514)
#define LW95A1_HDCP_VERIFY_HPRIME_PARAM_OFFSET                                  31:0
#define LW95A1_HDCP_ENCRYPT_PAIRING_INFO                                        (0x00000518)
#define LW95A1_HDCP_ENCRYPT_PAIRING_INFO_PARAM_OFFSET                           31:0
#define LW95A1_HDCP_DECRYPT_PAIRING_INFO                                        (0x0000051C)
#define LW95A1_HDCP_DECRYPT_PAIRING_INFO_PARAM_OFFSET                           31:0
#define LW95A1_HDCP_UPDATE_SESSION                                              (0x00000520)
#define LW95A1_HDCP_UPDATE_SESSION_PARAM_OFFSET                                 31:0
#define LW95A1_HDCP_GENERATE_LC_INIT                                            (0x00000524)
#define LW95A1_HDCP_GENERATE_LC_INIT_PARAM_OFFSET                               31:0
#define LW95A1_HDCP_VERIFY_LPRIME                                               (0x00000528)
#define LW95A1_HDCP_VERIFY_LPRIME_PARAM_OFFSET                                  31:0
#define LW95A1_HDCP_GENERATE_SKE_INIT                                           (0x0000052C)
#define LW95A1_HDCP_GENERATE_SKE_INIT_PARAM_OFFSET                              31:0
#define LW95A1_HDCP_VERIFY_VPRIME                                               (0x00000530)
#define LW95A1_HDCP_VERIFY_VPRIME_PARAM_OFFSET                                  31:0
#define LW95A1_HDCP_ENCRYPTION_RUN_CTRL                                         (0x00000534)
#define LW95A1_HDCP_ENCRYPTION_RUN_CTRL_PARAM_OFFSET                            31:0
#define LW95A1_HDCP_SESSION_CTRL                                                (0x00000538)
#define LW95A1_HDCP_SESSION_CTRL_PARAM_OFFSET                                   31:0
#define LW95A1_HDCP_COMPUTE_SPRIME                                              (0x0000053C)
#define LW95A1_HDCP_COMPUTE_SPRIME_PARAM_OFFSET                                 31:0
#define LW95A1_HDCP_GET_CERT_RX                                                 (0x00000540)
#define LW95A1_HDCP_GET_CERT_RX_PARAM_OFFSET                                    31:0
#define LW95A1_HDCP_EXCHANGE_INFO                                               (0x00000544)
#define LW95A1_HDCP_EXCHANGE_INFO_PARAM_OFFSET                                  31:0
#define LW95A1_HDCP_DECRYPT_KM                                                  (0x00000548)
#define LW95A1_HDCP_DECRYPT_KM_PARAM_OFFSET                                     31:0
#define LW95A1_HDCP_GET_HPRIME                                                  (0x0000054C)
#define LW95A1_HDCP_GET_HPRIME_PARAM_OFFSET                                     31:0
#define LW95A1_HDCP_GENERATE_EKH_KM                                             (0x00000550)
#define LW95A1_HDCP_GENERATE_EKH_KM_PARAM_OFFSET                                31:0
#define LW95A1_HDCP_VERIFY_RTT_CHALLENGE                                        (0x00000554)
#define LW95A1_HDCP_VERIFY_RTT_CHALLENGE_PARAM_OFFSET                           31:0
#define LW95A1_HDCP_GET_LPRIME                                                  (0x00000558)
#define LW95A1_HDCP_GET_LPRIME_PARAM_OFFSET                                     31:0
#define LW95A1_HDCP_DECRYPT_KS                                                  (0x0000055C)
#define LW95A1_HDCP_DECRYPT_KS_PARAM_OFFSET                                     31:0
#define LW95A1_HDCP_DECRYPT                                                     (0x00000560)
#define LW95A1_HDCP_DECRYPT_PARAM_OFFSET                                        31:0
#define LW95A1_HDCP_GET_RRX                                                     (0x00000564)
#define LW95A1_HDCP_GET_RRX_PARAM_OFFSET                                        31:0
#define LW95A1_HDCP_DECRYPT_REENCRYPT                                           (0x00000568)
#define LW95A1_HDCP_DECRYPT_REENCRYPT_PARAM_OFFSET                              31:0
#define LW95A1_HDCP_AES_ECB_CRYPT                                               (0x0000056C)
#define LW95A1_HDCP_AES_ECB_CRYPT_PARAM_OFFSET                                  31:0
#define LW95A1_LWSI_SWAP_AREA_OFFSET                                            (0x00000600)
#define LW95A1_LWSI_SWAP_AREA_OFFSET_OFFSET                                     31:0
#define LW95A1_LWSI_SWAP_AREA_SIZE                                              (0x00000604)
#define LW95A1_LWSI_SWAP_AREA_SIZE_VALUE                                        31:0
#define LW95A1_LWSI_SHADER_CODE_OFFSET                                          (0x00000608)
#define LW95A1_LWSI_SHADER_CODE_OFFSET_OFFSET                                   31:0
#define LW95A1_LWSI_SHADER_CODE_SIZE                                            (0x0000060C)
#define LW95A1_LWSI_SHADER_CODE_SIZE_VALUE                                      31:0
#define LW95A1_LWSI_LOADER2_CODE_SIZE                                           (0x00000610)
#define LW95A1_LWSI_LOADER2_CODE_SIZE_VALUE                                     31:0
#define LW95A1_LWSI_SHADER_DATA_OFFSET                                          (0x00000614)
#define LW95A1_LWSI_SHADER_DATA_OFFSET_OFFSET                                   31:0
#define LW95A1_LWSI_SHADER_DATA_SIZE                                            (0x00000618)
#define LW95A1_LWSI_SHADER_DATA_SIZE_VALUE                                      31:0
#define LW95A1_LWSI_SHADER_VOL_DATA_SIZE                                        (0x0000061C)
#define LW95A1_LWSI_SHADER_VOL_DATA_SIZE_VALUE                                  31:0
#define LW95A1_LWSI_SHARED_SEG_OFFSET                                           (0x00000620)
#define LW95A1_LWSI_SHARED_SEG_OFFSET_OFFSET                                    31:0
#define LW95A1_LWSI_SHARED_SEG_SIZE                                             (0x00000624)
#define LW95A1_LWSI_SHARED_SEG_SIZE_VALUE                                       31:0
#define LW95A1_LWSI_MISC_DATA_OFFSET                                            (0x00000628)
#define LW95A1_LWSI_MISC_DATA_OFFSET_OFFSET                                     31:0
#define LW95A1_LWSI_MISC_DATA_SIZE                                              (0x0000062C)
#define LW95A1_LWSI_MISC_DATA_SIZE_VALUE                                        31:0
#define LW95A1_HDCP_VALIDATE_SRM                                                (0x00000700)
#define LW95A1_HDCP_VALIDATE_SRM_PARAM_OFFSET                                   31:0
#define LW95A1_HDCP_VALIDATE_STREAM                                             (0x00000704)
#define LW95A1_HDCP_VALIDATE_STREAM_PARAM_OFFSET                                31:0
#define LW95A1_HDCP_TEST_SELWRE_STATUS                                          (0x00000708)
#define LW95A1_HDCP_TEST_SELWRE_STATUS_PARAM_OFFSET                             31:0
#define LW95A1_HDCP_SET_DCP_KPUB                                                (0x0000070C)
#define LW95A1_HDCP_SET_DCP_KPUB_OFFSET                                         31:0
#define LW95A1_HDCP_SET_RX_KPUB                                                 (0x00000710)
#define LW95A1_HDCP_SET_RX_KPUB_OFFSET                                          31:0
#define LW95A1_HDCP_SET_CERT_RX                                                 (0x00000714)
#define LW95A1_HDCP_SET_CERT_RX_OFFSET                                          31:0
#define LW95A1_HDCP_SET_SCRATCH_BUFFER                                          (0x00000718)
#define LW95A1_HDCP_SET_SCRATCH_BUFFER_OFFSET                                   31:0
#define LW95A1_HDCP_SET_SRM                                                     (0x0000071C)
#define LW95A1_HDCP_SET_SRM_OFFSET                                              31:0
#define LW95A1_HDCP_SET_RECEIVER_ID_LIST                                        (0x00000720)
#define LW95A1_HDCP_SET_RECEIVER_ID_LIST_OFFSET                                 31:0
#define LW95A1_HDCP_SET_SPRIME                                                  (0x00000724)
#define LW95A1_HDCP_SET_SPRIME_OFFSET                                           31:0
#define LW95A1_HDCP_SET_ENC_INPUT_BUFFER                                        (0x00000728)
#define LW95A1_HDCP_SET_ENC_INPUT_BUFFER_OFFSET                                 31:0
#define LW95A1_HDCP_SET_ENC_OUTPUT_BUFFER                                       (0x0000072C)
#define LW95A1_HDCP_SET_ENC_OUTPUT_BUFFER_OFFSET                                31:0
#define LW95A1_HDCP_GET_RTT_CHALLENGE                                           (0x00000730)
#define LW95A1_HDCP_GET_RTT_CHALLENGE_PARAM_OFFSET                              31:0
#define LW95A1_HDCP_STREAM_MANAGE                                               (0x00000734)
#define LW95A1_HDCP_STREAM_MANAGE_PARAM_OFFSET                                  31:0
#define LW95A1_HDCP_READ_CAPS                                                   (0x00000738)
#define LW95A1_HDCP_READ_CAPS_PARAM_OFFSET                                      31:0
#define LW95A1_HDCP_ENCRYPT                                                     (0x0000073C)
#define LW95A1_HDCP_ENCRYPT_PARAM_OFFSET                                        31:0
#define LW95A1_GFE_READ_ECID                                                    (0x00000500)
#define LW95A1_GFE_READ_ECID_PARAM_OFFSET                                       31:0
#define LW95A1_GFE_SET_ECID_SIGN_BUF                                            (0x00000504)
#define LW95A1_GFE_SET_ECID_SIGN_BUF_PARAM_OFFSET                               31:0
#define LW95A1_GFE_SET_PRIV_KEY_BUF                                             (0x00000508)
#define LW95A1_GFE_SET_PRIV_KEY_BUF_PARAM_OFFSET                                31:0
#define LW95A1_GFE_SET_SCRATCH_BUF                                              (0x0000050C)
#define LW95A1_GFE_SET_SCRATCH_BUF_OFFSET                                       31:0
#define LW95A1_LWSR_MUTEX_ACQUIRE                                               (0x00000500)
#define LW95A1_LWSR_MUTEX_ACQUIRE_ALGORITHM                                     31:0
#define LW95A1_LWSR_MUTEX_ACQUIRE_ALGORITHM_HMACSHA1                            (0x00000000)
#define LW95A1_LWSR_MUTEX_ACQUIRE_ALGORITHM_FEISTELCIPHER                       (0x00000001)
#define LW95A1_LWSR_MUTEX_ACQUIRE_KEYBUF                                        (0x00000504)
#define LW95A1_LWSR_MUTEX_ACQUIRE_KEYBUF_OFFSET                                 31:0
#define LW95A1_HWV_DO_COPY                                                      (0x00000500)
#define LW95A1_HWV_DO_COPY_SIZE                                                 31:0
#define LW95A1_HWV_SET_DO_COPY_IO_BUFFER                                        (0x00000504)
#define LW95A1_HWV_PERF_EVAL                                                    (0x00000508)
#define LW95A1_HWV_SET_PERF_EVAL_PARAMETER_BUFFER                               (0x0000050C)
#define LW95A1_HWV_SET_PERF_EVAL_INPUT_BUFFER                                   (0x00000510)
#define LW95A1_HWV_SET_PERF_EVAL_OUTPUT_BUFFER                                  (0x00000514)
#define LW95A1_VPR_FIRST_METHOD                                                 (0x00000500)
#define LW95A1_VPR_PROGRAM_REGION                                               (0x00000500)
#define LW95A1_VPR_PROGRAM_REGION_OFFSET                                        31:0
#define LW95A1_VPR_PROGRAM_REGION_SIZE                                          (0x00000504)
#define LW95A1_VPR_PROGRAM_REGION_SIZE_V                                        31:0
#define LW95A1_APM_GET_ACK                                                      (0x00000500)
#define LW95A1_APM_GET_ACK_ADDR                                                 31:0
#define LW95A1_APM_COPY                                                         (0x00000504)
#define LW95A1_APM_COPY_TYPE                                                    0:0
#define LW95A1_APM_COPY_TYPE_ENCRYPT                                            (0x00000000)
#define LW95A1_APM_COPY_TYPE_DECRYPT                                            (0x00000001)
#define LW95A1_APM_COPY_SRC_ADDR_HI                                             (0x00000508)
#define LW95A1_APM_COPY_SRC_ADDR_HI_DATA                                        24:0
#define LW95A1_APM_COPY_SRC_ADDR_LO                                             (0x0000050C)
#define LW95A1_APM_COPY_SRC_ADDR_LO_DATA                                        31:4
#define LW95A1_APM_COPY_DST_ADDR_HI                                             (0x00000510)
#define LW95A1_APM_COPY_DST_ADDR_HI_DATA                                        24:0
#define LW95A1_APM_COPY_DST_ADDR_LO                                             (0x00000514)
#define LW95A1_APM_COPY_DST_ADDR_LO_DATA                                        31:4
#define LW95A1_APM_COPY_SIZE_BYTES                                              (0x00000518)
#define LW95A1_APM_COPY_SIZE_BYTES_DATA                                         31:2
#define LW95A1_APM_ENCRYPT_IV_ADDR_HI                                           (0x0000051C)
#define LW95A1_APM_ENCRYPT_IV_ADDR_HI_DATA                                      24:0
#define LW95A1_APM_ENCRYPT_IV_ADDR_LO                                           (0x00000520)
#define LW95A1_APM_ENCRYPT_IV_ADDR_LO_DATA                                      31:4
#define LW95A1_APM_DIGEST_ADDR_HI                                               (0x00000524)
#define LW95A1_APM_DIGEST_ADDR_HI_DATA                                          24:0
#define LW95A1_APM_DIGEST_ADDR_LO                                               (0x00000528)
#define LW95A1_APM_DIGEST_ADDR_LO_DATA                                          31:4
#define LW95A1_SET_CONTENT_INITIAL_VECTOR(b)                                    (0x00000C00 + (b)*0x00000004)
#define LW95A1_SET_CONTENT_INITIAL_VECTOR_VALUE                                 31:0
#define LW95A1_SET_CTL_COUNT                                                    (0x00000C10)
#define LW95A1_SET_CTL_COUNT_VALUE                                              31:0
#define LW95A1_SET_MDEC_H2_MKEY                                                 (0x00000C14)
#define LW95A1_SET_MDEC_H2_MKEY_HOST_SKEY                                       15:0
#define LW95A1_SET_MDEC_H2_MKEY_HOST_KEY_HASH                                   23:16
#define LW95A1_SET_MDEC_H2_MKEY_DEC_ID                                          31:24
#define LW95A1_SET_MDEC_M2_HKEY                                                 (0x00000C18)
#define LW95A1_SET_MDEC_M2_HKEY_MPEG_SKEY                                       15:0
#define LW95A1_SET_MDEC_M2_HKEY_SELECTOR                                        23:16
#define LW95A1_SET_MDEC_M2_HKEY_MPEG_KEY_HASH                                   31:24
#define LW95A1_SET_MDEC_FRAME_KEY                                               (0x00000C1C)
#define LW95A1_SET_MDEC_FRAME_KEY_VALUE                                         15:0
#define LW95A1_SET_UPPER_SRC                                                    (0x00000C20)
#define LW95A1_SET_UPPER_SRC_OFFSET                                             7:0
#define LW95A1_SET_LOWER_SRC                                                    (0x00000C24)
#define LW95A1_SET_LOWER_SRC_OFFSET                                             31:0
#define LW95A1_SET_UPPER_DST                                                    (0x00000C28)
#define LW95A1_SET_UPPER_DST_OFFSET                                             7:0
#define LW95A1_SET_LOWER_DST                                                    (0x00000C2C)
#define LW95A1_SET_LOWER_DST_OFFSET                                             31:0
#define LW95A1_SET_UPPER_CTL                                                    (0x00000C30)
#define LW95A1_SET_UPPER_CTL_OFFSET                                             7:0
#define LW95A1_SET_LOWER_CTL                                                    (0x00000C34)
#define LW95A1_SET_LOWER_CTL_OFFSET                                             31:0
#define LW95A1_SET_BLOCK_COUNT                                                  (0x00000C38)
#define LW95A1_SET_BLOCK_COUNT_VALUE                                            31:0
#define LW95A1_SET_STRETCH_MASK                                                 (0x00000C3C)
#define LW95A1_SET_STRETCH_MASK_VALUE                                           31:0
#define LW95A1_SET_UPPER_FLOW_CTRL_INSELWRE                                     (0x00000D00)
#define LW95A1_SET_UPPER_FLOW_CTRL_INSELWRE_OFFSET                              7:0
#define LW95A1_SET_LOWER_FLOW_CTRL_INSELWRE                                     (0x00000D04)
#define LW95A1_SET_LOWER_FLOW_CTRL_INSELWRE_OFFSET                              31:0
#define LW95A1_SET_UCODE_LOADER_PARAMS                                          (0x00000D10)
#define LW95A1_SET_UCODE_LOADER_PARAMS_BLOCK_COUNT                              7:0
#define LW95A1_SET_UCODE_LOADER_PARAMS_SELWRITY_PARAM                           15:8
#define LW95A1_SET_UPPER_FLOW_CTRL_SELWRE                                       (0x00000D18)
#define LW95A1_SET_UPPER_FLOW_CTRL_SELWRE_OFFSET                                7:0
#define LW95A1_SET_LOWER_FLOW_CTRL_SELWRE                                       (0x00000D1C)
#define LW95A1_SET_LOWER_FLOW_CTRL_SELWRE_OFFSET                                31:0
#define LW95A1_SET_UCODE_LOADER_OFFSET                                          (0x00000D34)
#define LW95A1_SET_UCODE_LOADER_OFFSET_OFFSET                                   31:0
#define LW95A1_SET_SESSION_KEY(b)                                               (0x00000F00 + (b)*0x00000004)
#define LW95A1_SET_SESSION_KEY_VALUE                                            31:0
#define LW95A1_SET_CONTENT_KEY(b)                                               (0x00000F10 + (b)*0x00000004)
#define LW95A1_SET_CONTENT_KEY_VALUE                                            31:0
#define LW95A1_PM_TRIGGER_END                                                   (0x00001114)
#define LW95A1_PM_TRIGGER_END_V                                                 31:0

#define LW95A1_ERROR_NONE                                                       (0x00000000)
#define LW95A1_ERROR_DMA_NACK                                                   (0xFFFFFFFF)
#define LW95A1_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LW95A1_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LW95A1_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LW95A1_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LW95A1_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LW95A1_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LW95A1_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LW95A1_OS_ERROR_APPLICATION                                             (0x00000008)
#define LW95A1_OS_ERROR_WDTIMER                                                 (0x00000009)
#define LW95A1_OS_ERROR_REGISTER_ACCESS                                         (0x0000000A)
#define LW95A1_OS_ERROR_SHA_FAILURE                                             (0x0000000B)
#define LW95A1_OS_ERROR_AES_CTR_FAILURE                                         (0x0000000C)
#define LW95A1_OS_ERROR_DMA_CONFIG                                              (0x0000000D)
#define LW95A1_ERROR_APM_COPY_MISALIGNED_SIZE                                   (0x00000050)
#define LW95A1_ERROR_APM_SRC_ADDR_MISALIGNED_POINTER                            (0x00000051)
#define LW95A1_ERROR_APM_DEST_ADDR_MISALIGNED_POINTER                           (0x00000052)
#define LW95A1_ERROR_APM_COPY_DMA_NACK                                          (0x00000053)
#define LW95A1_ERROR_APM_DIGEST_ADDR_MISALIGNED_POINTER                         (0x00000054)
#define LW95A1_ERROR_APM_DIGEST_ADDR_DMA_NACK                                   (0x00000055)
#define LW95A1_ERROR_APM_DIGEST_CHECK_FAILURE                                   (0x00000056)
#define LW95A1_ERROR_APM_ILWALID_METHOD                                         (0x00000057)
#define LW95A1_ERROR_APM_ILWALID_KMB                                            (0x00000058)
#define LW95A1_ERROR_APM_IV_ADDR_MISALIGNED_POINTER                             (0x00000059)
#define LW95A1_ERROR_APM_IV_OVERFLOW_ENCRYPT                                    (0x0000005A)
#define LW95A1_ERROR_APM_IV_OVERFLOW_DECRYPT                                    (0x0000005B)
#define LW95A1_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LW95A1_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LW95A1_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl95a1_h

