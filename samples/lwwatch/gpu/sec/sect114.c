/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//-----------------------------------------------------
//
// sec0100.c - SEC routines
// 
//-----------------------------------------------------

#include "hal.h"
#include "tegrasys.h"
#include "t11x/t114/dev_sec_pri.h"
#include "t11x/t114/dev_falcon_v5.h"

#include "sec.h"

#include "g_cipher_private.h"     // (rmconfig)  implementation prototypes

// TSEC Device specific register access macros
#define SEC_REG_RD32(reg)           (DEV_REG_RD32((reg - DRF_BASE(LW_PSEC)), "SEC", 0))
#define SEC_REG_WR32(reg,val)       (DEV_REG_WR32((reg - DRF_BASE(LW_PSEC)), val, "SEC", 0))
#define SEC_REG_RD_DRF(d,r,f)       (((SEC_REG_RD32(LW ## d ## r))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f)) 

#define LW_95A1_NOP                                                              (0x00000100)
#define LW_95A1_PM_TRIGGER                                                       (0x00000140)
#define LW_95A1_SET_APPLICATION_ID                                               (0x00000200)
#define LW_95A1_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW_95A1_SEMAPHORE_A                                                      (0x00000240)
#define LW_95A1_SEMAPHORE_B                                                      (0x00000244)
#define LW_95A1_SEMAPHORE_C                                                      (0x00000248)
#define LW_95A1_EXELWTE                                                          (0x00000300)
#define LW_95A1_SEMAPHORE_D                                                      (0x00000304)
#define LW_95A1_HDCP_INIT                                                        (0x00000500)
#define LW_95A1_HDCP_CREATE_SESSION                                              (0x00000504)
#define LW_95A1_HDCP_VERIFY_CERT_RX                                              (0x00000508)
#define LW_95A1_HDCP_GENERATE_EKM                                                (0x0000050C)
#define LW_95A1_HDCP_REVOCATION_CHECK                                            (0x00000510)
#define LW_95A1_HDCP_VERIFY_HPRIME                                               (0x00000514)
#define LW_95A1_HDCP_ENCRYPT_PAIRING_INFO                                        (0x00000518)
#define LW_95A1_HDCP_DECRYPT_PAIRING_INFO                                        (0x0000051C)
#define LW_95A1_HDCP_UPDATE_SESSION                                              (0x00000520)
#define LW_95A1_HDCP_GENERATE_LC_INIT                                            (0x00000524)
#define LW_95A1_HDCP_VERIFY_LPRIME                                               (0x00000528)
#define LW_95A1_HDCP_GENERATE_SKE_INIT                                           (0x0000052C)
#define LW_95A1_HDCP_VERIFY_VPRIME                                               (0x00000530)
#define LW_95A1_HDCP_ENCRYPTION_RUN_CTRL                                         (0x00000534)
#define LW_95A1_HDCP_SESSION_CTRL                                                (0x00000538)
#define LW_95A1_HDCP_COMPUTE_SPRIME                                              (0x0000053C)
#define LW_95A1_LWSI_SWAP_AREA_OFFSET                                            (0x00000600)
#define LW_95A1_LWSI_SWAP_AREA_SIZE                                              (0x00000604)
#define LW_95A1_LWSI_SHADER_CODE_OFFSET                                          (0x00000608)
#define LW_95A1_LWSI_SHADER_CODE_SIZE                                            (0x0000060C)
#define LW_95A1_LWSI_LOADER2_CODE_SIZE                                           (0x00000610)
#define LW_95A1_LWSI_SHADER_DATA_OFFSET                                          (0x00000614)
#define LW_95A1_LWSI_SHADER_DATA_SIZE                                            (0x00000618)
#define LW_95A1_LWSI_SHADER_VOL_DATA_SIZE                                        (0x0000061C)
#define LW_95A1_LWSI_SHARED_SEG_OFFSET                                           (0x00000620)
#define LW_95A1_LWSI_SHARED_SEG_SIZE                                             (0x00000624)
#define LW_95A1_LWSI_MISC_DATA_OFFSET                                            (0x00000628)
#define LW_95A1_LWSI_MISC_DATA_SIZE                                              (0x0000062C)
#define LW_95A1_HDCP_VALIDATE_SRM                                                (0x00000700)
#define LW_95A1_HDCP_VALIDATE_STREAM                                             (0x00000704)
#define LW_95A1_HDCP_TEST_SELWRE_STATUS                                          (0x00000708)
#define LW_95A1_HDCP_SET_DCP_KPUB                                                (0x0000070C)
#define LW_95A1_HDCP_SET_RX_KPUB                                                 (0x00000710)
#define LW_95A1_HDCP_SET_CERT_RX                                                 (0x00000714)
#define LW_95A1_HDCP_SET_SCRATCH_BUFFER                                          (0x00000718)
#define LW_95A1_HDCP_SET_SRM                                                     (0x0000071C)
#define LW_95A1_HDCP_SET_RECEIVER_ID_LIST                                        (0x00000720)
#define LW_95A1_HDCP_SET_SPRIME                                                  (0x00000724)
#define LW_95A1_HDCP_SET_ENC_INPUT_BUFFER                                        (0x00000728)
#define LW_95A1_HDCP_SET_ENC_OUTPUT_BUFFER                                       (0x0000072C)
#define LW_95A1_HDCP_GET_RTT_CHALLENGE                                           (0x00000730)
#define LW_95A1_HDCP_STREAM_MANAGE                                               (0x00000734)
#define LW_95A1_HDCP_READ_CAPS                                                   (0x00000738)
#define LW_95A1_HDCP_ENCRYPT                                                     (0x0000073C)
#define LW_95A1_SET_CONTENT_INITIAL_VECTOR(b)                                    (0x00000C00 + (b)*0x00000004)
#define LW_95A1_SET_CTL_COUNT                                                    (0x00000C10)
#define LW_95A1_SET_MDEC_H2_MKEY                                                 (0x00000C14)
#define LW_95A1_SET_MDEC_M2_HKEY                                                 (0x00000C18)
#define LW_95A1_SET_MDEC_FRAME_KEY                                               (0x00000C1C)
#define LW_95A1_SET_UPPER_SRC                                                    (0x00000C20)
#define LW_95A1_SET_LOWER_SRC                                                    (0x00000C24)
#define LW_95A1_SET_UPPER_DST                                                    (0x00000C28)
#define LW_95A1_SET_LOWER_DST                                                    (0x00000C2C)
#define LW_95A1_SET_UPPER_CTL                                                    (0x00000C30)
#define LW_95A1_SET_LOWER_CTL                                                    (0x00000C34)
#define LW_95A1_SET_BLOCK_COUNT                                                  (0x00000C38)
#define LW_95A1_SET_STRETCH_MASK                                                 (0x00000C3C)
#define LW_95A1_SET_UPPER_FLOW_CTRL_INSELWRE                                     (0x00000D00)
#define LW_95A1_SET_LOWER_FLOW_CTRL_INSELWRE                                     (0x00000D04)
#define LW_95A1_SET_UCODE_LOADER_PARAMS                                          (0x00000D10)
#define LW_95A1_SET_UPPER_FLOW_CTRL_SELWRE                                       (0x00000D18)
#define LW_95A1_SET_LOWER_FLOW_CTRL_SELWRE                                       (0x00000D1C)
#define LW_95A1_SET_UCODE_LOADER_OFFSET                                          (0x00000D34)
#define LW_95A1_SET_SESSION_KEY(b)                                               (0x00000F00 + (b)*0x00000004)
#define LW_95A1_SET_CONTENT_KEY(b)                                               (0x00000F10 + (b)*0x00000004)
#define LW_95A1_PM_TRIGGER_END                                                   (0x00001114)

dbg_sec_t114 secMethodTable_t114[] =
{
    privInfo_sec_t114(LW_95A1_NOP),
    privInfo_sec_t114(LW_95A1_PM_TRIGGER),
    privInfo_sec_t114(LW_95A1_SET_APPLICATION_ID),
    privInfo_sec_t114(LW_95A1_SET_WATCHDOG_TIMER),
    privInfo_sec_t114(LW_95A1_SEMAPHORE_A),
    privInfo_sec_t114(LW_95A1_SEMAPHORE_B),
    privInfo_sec_t114(LW_95A1_SEMAPHORE_C),
    privInfo_sec_t114(LW_95A1_EXELWTE),
    privInfo_sec_t114(LW_95A1_SEMAPHORE_D),
    privInfo_sec_t114(LW_95A1_HDCP_INIT),
    privInfo_sec_t114(LW_95A1_HDCP_CREATE_SESSION),
    privInfo_sec_t114(LW_95A1_HDCP_VERIFY_CERT_RX),
    privInfo_sec_t114(LW_95A1_HDCP_GENERATE_EKM),
    privInfo_sec_t114(LW_95A1_HDCP_REVOCATION_CHECK),
    privInfo_sec_t114(LW_95A1_HDCP_VERIFY_HPRIME),
    privInfo_sec_t114(LW_95A1_HDCP_ENCRYPT_PAIRING_INFO),
    privInfo_sec_t114(LW_95A1_HDCP_DECRYPT_PAIRING_INFO),
    privInfo_sec_t114(LW_95A1_HDCP_UPDATE_SESSION),
    privInfo_sec_t114(LW_95A1_HDCP_GENERATE_LC_INIT),
    privInfo_sec_t114(LW_95A1_HDCP_VERIFY_LPRIME),
    privInfo_sec_t114(LW_95A1_HDCP_GENERATE_SKE_INIT),
    privInfo_sec_t114(LW_95A1_HDCP_VERIFY_VPRIME),
    privInfo_sec_t114(LW_95A1_HDCP_ENCRYPTION_RUN_CTRL),
    privInfo_sec_t114(LW_95A1_HDCP_SESSION_CTRL),
    privInfo_sec_t114(LW_95A1_HDCP_COMPUTE_SPRIME),
    privInfo_sec_t114(LW_95A1_LWSI_SWAP_AREA_OFFSET),
    privInfo_sec_t114(LW_95A1_LWSI_SWAP_AREA_SIZE),
    privInfo_sec_t114(LW_95A1_LWSI_SHADER_CODE_OFFSET),
    privInfo_sec_t114(LW_95A1_LWSI_SHADER_CODE_SIZE),
    privInfo_sec_t114(LW_95A1_LWSI_LOADER2_CODE_SIZE),
    privInfo_sec_t114(LW_95A1_LWSI_SHADER_DATA_OFFSET),
    privInfo_sec_t114(LW_95A1_LWSI_SHADER_DATA_SIZE),
    privInfo_sec_t114(LW_95A1_LWSI_SHADER_VOL_DATA_SIZE),
    privInfo_sec_t114(LW_95A1_LWSI_SHARED_SEG_OFFSET),
    privInfo_sec_t114(LW_95A1_LWSI_SHARED_SEG_SIZE),
    privInfo_sec_t114(LW_95A1_LWSI_MISC_DATA_OFFSET),
    privInfo_sec_t114(LW_95A1_LWSI_MISC_DATA_SIZE),
    privInfo_sec_t114(LW_95A1_HDCP_VALIDATE_SRM),
    privInfo_sec_t114(LW_95A1_HDCP_VALIDATE_STREAM),
    privInfo_sec_t114(LW_95A1_HDCP_TEST_SELWRE_STATUS),
    privInfo_sec_t114(LW_95A1_HDCP_SET_DCP_KPUB),
    privInfo_sec_t114(LW_95A1_HDCP_SET_RX_KPUB),
    privInfo_sec_t114(LW_95A1_HDCP_SET_CERT_RX),
    privInfo_sec_t114(LW_95A1_HDCP_SET_SCRATCH_BUFFER),
    privInfo_sec_t114(LW_95A1_HDCP_SET_SRM),
    privInfo_sec_t114(LW_95A1_HDCP_SET_RECEIVER_ID_LIST),
    privInfo_sec_t114(LW_95A1_HDCP_SET_SPRIME),
    privInfo_sec_t114(LW_95A1_HDCP_SET_ENC_INPUT_BUFFER),
    privInfo_sec_t114(LW_95A1_HDCP_SET_ENC_OUTPUT_BUFFER),
    privInfo_sec_t114(LW_95A1_HDCP_GET_RTT_CHALLENGE),
    privInfo_sec_t114(LW_95A1_HDCP_STREAM_MANAGE),
    privInfo_sec_t114(LW_95A1_HDCP_READ_CAPS),
    privInfo_sec_t114(LW_95A1_HDCP_ENCRYPT),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_INITIAL_VECTOR(0)),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_INITIAL_VECTOR(1)),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_INITIAL_VECTOR(2)),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_INITIAL_VECTOR(3)),
    privInfo_sec_t114(LW_95A1_SET_CTL_COUNT),
    privInfo_sec_t114(LW_95A1_SET_MDEC_H2_MKEY),
    privInfo_sec_t114(LW_95A1_SET_MDEC_M2_HKEY),
    privInfo_sec_t114(LW_95A1_SET_MDEC_FRAME_KEY),
    privInfo_sec_t114(LW_95A1_SET_UPPER_SRC),
    privInfo_sec_t114(LW_95A1_SET_LOWER_SRC),
    privInfo_sec_t114(LW_95A1_SET_UPPER_DST),
    privInfo_sec_t114(LW_95A1_SET_LOWER_DST),
    privInfo_sec_t114(LW_95A1_SET_UPPER_CTL),
    privInfo_sec_t114(LW_95A1_SET_LOWER_CTL),
    privInfo_sec_t114(LW_95A1_SET_BLOCK_COUNT),
    privInfo_sec_t114(LW_95A1_SET_STRETCH_MASK),
    privInfo_sec_t114(LW_95A1_SET_UPPER_FLOW_CTRL_INSELWRE),
    privInfo_sec_t114(LW_95A1_SET_LOWER_FLOW_CTRL_INSELWRE),
    privInfo_sec_t114(LW_95A1_SET_UCODE_LOADER_PARAMS),
    privInfo_sec_t114(LW_95A1_SET_UPPER_FLOW_CTRL_SELWRE),
    privInfo_sec_t114(LW_95A1_SET_LOWER_FLOW_CTRL_SELWRE),
    privInfo_sec_t114(LW_95A1_SET_UCODE_LOADER_OFFSET),
    privInfo_sec_t114(LW_95A1_SET_SESSION_KEY(0)),
    privInfo_sec_t114(LW_95A1_SET_SESSION_KEY(1)),
    privInfo_sec_t114(LW_95A1_SET_SESSION_KEY(2)),
    privInfo_sec_t114(LW_95A1_SET_SESSION_KEY(3)),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_KEY(0)),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_KEY(1)),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_KEY(2)),
    privInfo_sec_t114(LW_95A1_SET_CONTENT_KEY(3)),
    privInfo_sec_t114(LW_95A1_PM_TRIGGER_END),
    privInfo_sec_t114(0),
};

dbg_sec_t114 secPrivReg_t114[] =
{
    privInfo_sec_t114(LW_PSEC_FALCON_IRQMODE),
    privInfo_sec_t114(LW_PSEC_FALCON_IRQDEST),
    privInfo_sec_t114(LW_PSEC_FALCON_GPTMRINT),
    privInfo_sec_t114(LW_PSEC_FALCON_GPTMRVAL),
    privInfo_sec_t114(LW_PSEC_FALCON_GPTMRCTL),
    privInfo_sec_t114(LW_PSEC_FALCON_WDTMRVAL),
    privInfo_sec_t114(LW_PSEC_FALCON_WDTMRCTL),
    privInfo_sec_t114(LW_PSEC_FALCON_MTHDID),
    privInfo_sec_t114(LW_PSEC_FALCON_MTHDWDAT),
    privInfo_sec_t114(LW_PSEC_FALCON_MTHDRAMSZ),
    privInfo_sec_t114(LW_PSEC_FALCON_LWRCTX),
    privInfo_sec_t114(LW_PSEC_FALCON_NXTCTX),
    privInfo_sec_t114(LW_PSEC_FALCON_MAILBOX0),
    privInfo_sec_t114(LW_PSEC_FALCON_MAILBOX1),
    privInfo_sec_t114(LW_PSEC_FALCON_ITFEN),
    privInfo_sec_t114(LW_PSEC_FALCON_PRIVSTATE),
    privInfo_sec_t114(LW_PSEC_FALCON_SFTRESET),
    privInfo_sec_t114(LW_PSEC_FALCON_OS),
    privInfo_sec_t114(LW_PSEC_FALCON_RM),
    privInfo_sec_t114(LW_PSEC_FALCON_SOFT_PM),
    privInfo_sec_t114(LW_PSEC_FALCON_SOFT_MODE),
    privInfo_sec_t114(LW_PSEC_FALCON_DEBUG1),
    privInfo_sec_t114(LW_PSEC_FALCON_DEBUGINFO),
    privInfo_sec_t114(LW_PSEC_FALCON_IBRKPT1),
    privInfo_sec_t114(LW_PSEC_FALCON_IBRKPT2),
    privInfo_sec_t114(LW_PSEC_FALCON_CGCTL),
    privInfo_sec_t114(LW_PSEC_FALCON_ENGCTL),
    privInfo_sec_t114(LW_PSEC_FALCON_PMM),
    privInfo_sec_t114(LW_PSEC_FALCON_ADDR),
    privInfo_sec_t114(LW_PSEC_FALCON_CPUCTL),
    privInfo_sec_t114(LW_PSEC_FALCON_BOOTVEC),
    privInfo_sec_t114(LW_PSEC_FALCON_DMACTL),
    privInfo_sec_t114(LW_PSEC_FALCON_DMATRFBASE),
    privInfo_sec_t114(LW_PSEC_FALCON_DMATRFMOFFS),
    privInfo_sec_t114(LW_PSEC_FALCON_DMATRFCMD),
    privInfo_sec_t114(LW_PSEC_FALCON_DMATRFFBOFFS),
    privInfo_sec_t114(LW_PSEC_FALCON_DMAPOLL_FB),
    privInfo_sec_t114(LW_PSEC_FALCON_DMAPOLL_CP),
    privInfo_sec_t114(LW_PSEC_FALCON_IMCTL),
    privInfo_sec_t114(LW_PSEC_FALCON_TRACEIDX),
    privInfo_sec_t114(LW_PSEC_FALCON_CG1_SLCG),
    privInfo_sec_t114(LW_PSEC_FALCON_ICD_CMD),
    privInfo_sec_t114(LW_PSEC_FALCON_ICD_ADDR),
    privInfo_sec_t114(LW_PSEC_FALCON_ICD_WDATA),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMC(0)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMC(1)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMC(2)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMC(3)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMD(0)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMD(1)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMD(2)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMD(3)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMT(0)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMT(1)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMT(2)),
    privInfo_sec_t114(LW_PSEC_FALCON_IMEMT(3)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(0)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(1)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(2)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(3)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(4)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(5)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(6)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMC(7)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(0)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(1)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(2)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(3)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(4)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(5)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(6)),
    privInfo_sec_t114(LW_PSEC_FALCON_DMEMD(7)),
    privInfo_sec_t114(LW_PSEC_THI_INCR_SYNCPT),
    privInfo_sec_t114(LW_PSEC_THI_INCR_SYNCPT_CTRL),
    privInfo_sec_t114(LW_PSEC_THI_INCR_SYNCPT_ERR),
    privInfo_sec_t114(LW_PSEC_THI_CTXSW_INCR_SYNCPT),
    privInfo_sec_t114(LW_PSEC_THI_CTXSW),
    privInfo_sec_t114(LW_PSEC_THI_CONT_SYNCPT_EOF),
    privInfo_sec_t114(LW_PSEC_THI_METHOD0),
    privInfo_sec_t114(LW_PSEC_THI_METHOD1),
    privInfo_sec_t114(LW_PSEC_THI_CONTEXT_SWITCH),
    privInfo_sec_t114(LW_PSEC_THI_INT_STATUS),
    privInfo_sec_t114(LW_PSEC_THI_INT_MASK),
    privInfo_sec_t114(LW_PSEC_THI_CONFIG0),
    privInfo_sec_t114(LW_PSEC_THI_DBG_MISC),
    privInfo_sec_t114(LW_PSEC_THI_SLCG_OVERRIDE_HIGH_A),
    privInfo_sec_t114(LW_PSEC_THI_SLCG_OVERRIDE_LOW_A),
    privInfo_sec_t114(LW_PSEC_THI_CLK_OVERRIDE),
    privInfo_sec_t114(LW_PSEC_BAR0_CSR),
    privInfo_sec_t114(LW_PSEC_BAR0_ADDR),
    privInfo_sec_t114(LW_PSEC_BAR0_DATA),
    privInfo_sec_t114(LW_PSEC_BAR0_TMOUT),
    privInfo_sec_t114(LW_PSEC_VERSION),
    privInfo_sec_t114(LW_PSEC_CAP_REG0),
    privInfo_sec_t114(LW_PSEC_CAP_REG1),
    privInfo_sec_t114(LW_PSEC_CAP_REG2),
    privInfo_sec_t114(LW_PSEC_CAP_REG3),
    privInfo_sec_t114(LW_PSEC_SCRATCH0),
    privInfo_sec_t114(LW_PSEC_SCRATCH1),
    privInfo_sec_t114(LW_PSEC_SCRATCH2),
    privInfo_sec_t114(LW_PSEC_SCRATCH3),
    privInfo_sec_t114(LW_PSEC_GPTMRINT),
    privInfo_sec_t114(LW_PSEC_GPTMRVAL),
    privInfo_sec_t114(LW_PSEC_GPTMRCTL),
    privInfo_sec_t114(LW_PSEC_MISC_INTEN),
    privInfo_sec_t114(LW_PSEC_MISC_INTSTAT),
    privInfo_sec_t114(LW_PSEC_TEGRA_CTL),
    privInfo_sec_t114(LW_PSEC_TFBIF_CTL),
    privInfo_sec_t114(LW_PSEC_TFBIF_MCCIF_FIFOCTRL),
    privInfo_sec_t114(LW_PSEC_TFBIF_THROTTLE),
    privInfo_sec_t114(LW_PSEC_SCP_CTL0),
    privInfo_sec_t114(LW_PSEC_SCP_CTL1),
    privInfo_sec_t114(LW_PSEC_SCP_CTL_CFG),
    privInfo_sec_t114(LW_PSEC_SCP_CFG0),
    privInfo_sec_t114(LW_PSEC_SCP_CTL_PKEY),
    privInfo_sec_t114(LW_PSEC_SCP_CTL_DEBUG),
    privInfo_sec_t114(LW_PSEC_SCP_DEBUG0),
    privInfo_sec_t114(LW_PSEC_SCP_DEBUG1),
    privInfo_sec_t114(LW_PSEC_SCP_DEBUG_CMD),
    privInfo_sec_t114(LW_PSEC_SCP_RNG_STAT0),
    privInfo_sec_t114(LW_PSEC_SCP_INTR),
    privInfo_sec_t114(LW_PSEC_SCP_INTR_EN),
    privInfo_sec_t114(LW_PSEC_SCP_ACL_VIO),
    privInfo_sec_t114(LW_PSEC_SCP_CMD_ERROR),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL0),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL1),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL2),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL3),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL4),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL5),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL6),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL7),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL8),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL9),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL10),
    privInfo_sec_t114(LW_PSEC_SCP_RNDCTL11),
    privInfo_sec_t114(0),
};

//-----------------------------------------------------
// secIsSupported_T194
//-----------------------------------------------------
BOOL secIsSupported_T194( LwU32 indexGpu )
{
    pSecPrivReg = secPrivReg_t114;
    pSecMethodTable = secMethodTable_t114;
    return TRUE;
}

//-----------------------------------------------------
// secDumpImem_T194 - Dumps SEC instruction memory
//-----------------------------------------------------
LW_STATUS secDumpImem_T194( LwU32 indexGpu , LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addressImem = LW_PSEC_FALCON_IMEMD(0);
    LwU32 address2Imem = LW_PSEC_FALCON_IMEMC(0);
    LwU32 address2Imemt = LW_PSEC_FALCON_IMEMT(0);
    LwU32 u;
    LwU32 blk=0;
    imemSizeMax = (SEC_REG_RD_DRF(_PSEC_FALCON, _HWCFG, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u SEC IMEM -- \n", indexGpu);    
    dprintf("lw: -- Gpu %u SEC IMEM SIZE =  0x%08x-- \n", indexGpu,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0) {
            SEC_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PSEC_FALCON_IMEMC_OFFS));
        SEC_REG_WR32(address2Imem,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  SEC_REG_RD32(addressImem));
    }
    return status;  
}

//-----------------------------------------------------
// secDumpDmem_T194 - Dumps SEC data memory
//-----------------------------------------------------
LW_STATUS secDumpDmem_T194( LwU32 indexGpu , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 address, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZE] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZE] = {0};
    LwU32 methodIdx;

    dmemSizeMax = (SEC_REG_RD_DRF(_PSEC_FALCON, _HWCFG, _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    address     = LW_PSEC_FALCON_DMEMD(0);
    address2    = LW_PSEC_FALCON_DMEMC(0);
    classNum    = 0xA0B7;

    dprintf("\n");
    dprintf("lw: -- Gpu %u SEC DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u SEC DMEM SIZE =  0x%08x-- \n", indexGpu,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    

    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PSEC_FALCON_IMEMC_OFFS));
        SEC_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  SEC_REG_RD32(address));
    }

    // get methods offset are in the DWORD#3 in dmem
    u = (3<<(0?LW_PSEC_FALCON_IMEMC_OFFS));
    SEC_REG_WR32(address2,u);
    comMthdOffs = (SEC_REG_RD32(address)) >> 2;
    appMthdOffs = comMthdOffs + 16;

    for(u=0; u<CMNMETHODARRAYSIZE;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PSEC_FALCON_IMEMC_OFFS));
        SEC_REG_WR32(address2,i);
        comMthd[u] = SEC_REG_RD32(address);
        i = ((u+appMthdOffs)<<(0?LW_PSEC_FALCON_IMEMC_OFFS));
        SEC_REG_WR32(address2,i);
        appMthd[u] = SEC_REG_RD32(address);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZE; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        CMNMETHODBASE+4*u, comMthd[u], CMNMETHODBASE+4*(u+1), comMthd[u+1], 
        CMNMETHODBASE+4*(u+2), comMthd[u+2], CMNMETHODBASE+4*(u+3), comMthd[u+3]);
    }
    dprintf("\n");
    dprintf("\n[APP METHODS]\n");
    for (u=0; u<APPMETHODARRAYSIZE; u+=4)
    {

        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        APPMETHODBASE+4*u, appMthd[u], APPMETHODBASE+4*(u+1), appMthd[u+1],
        APPMETHODBASE+4*(u+2), appMthd[u+2], APPMETHODBASE+4*(u+3), appMthd[u+3]);
    }

    dprintf("\n[COMMON METHODS]\n");
    for(u=0;u<16;u++)
    {
        for(methodIdx=0;;methodIdx++)
        {
            if(pSecMethodTable[methodIdx].m_id == (CMNMETHODBASE+4*u))
            {
                secPrintMethodData_t114(40,pSecMethodTable[methodIdx].m_tag, pSecMethodTable[methodIdx].m_id, comMthd[u]);
                break;
            }
            else if (pSecMethodTable[methodIdx].m_id == 0)
            {
                break;
            }
        }
    }
    dprintf("\n");
    // app methods
    dprintf("\n[APP METHODS]\n");
    for(u=0;u<16;u++)
    {
        for(methodIdx=0;;methodIdx++)
        {
            if(pSecMethodTable[methodIdx].m_id == (APPMETHODBASE+4*u))
            {
                secPrintMethodData_t114(40,pSecMethodTable[methodIdx].m_tag, pSecMethodTable[methodIdx].m_id, appMthd[u]);
                break;
            }
            else if (pSecMethodTable[methodIdx].m_id == 0)
            {
                break;
            }
        }
    }
    dprintf("\n");
    return status;  
}

//-----------------------------------------------------
// secTestState_T194 - Test basic sec state
//-----------------------------------------------------
LW_STATUS secTestState_T194( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;
    LwU32   secBaseAddress;
    PDEVICE_RELOCATION pDev = NULL;

    pDev     = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], "TSEC", 0);
    assert(pDev);
    secBaseAddress = (LwU32)pDev->start;
    
    //check falcon interrupts
    regIntr = SEC_REG_RD32(LW_PSEC_FALCON_IRQSTAT);
    regIntrEn = SEC_REG_RD32(LW_PSEC_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PSEC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PSEC_FALCON_IRQMASK_SWGEN1 disabled\n");

   
    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t SEC interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PSEC_FALCON_GPTMRINT:    0x%08x\n", 
            SEC_REG_RD32(LW_PSEC_FALCON_GPTMRINT) );
        dprintf("lw: LW_PSEC_FALCON_GPTMRVAL:    0x%08x\n", 
            SEC_REG_RD32(LW_PSEC_FALCON_GPTMRVAL) );
        
    }
    
    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PSEC_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            SEC_REG_RD32(LW_PSEC_FALCON_MTHDDATA) );
        
        data32 = SEC_REG_RD32(LW_PSEC_FALCON_MTHDID);
        dprintf("lw: LW_PSEC_FALCON_MTHDID_ID:    0x%08x\n", 
           DRF_VAL( _PSEC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PSEC_FALCON_MTHDID_SUBCH:    0x%08x\n", 
           DRF_VAL( _PSEC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PSEC_FALCON_MTHDID_PRIV:    0x%08x\n", 
           DRF_VAL( _PSEC,_FALCON_MTHDID, _PRIV, data32)  );
    }
    
    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_CTXSW pending\n");
    }
    
    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_HALT pending\n");
    }
    
    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_EXTERR pending\n");
    }
    
    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(secBaseAddress);
    }

    if ( DRF_VAL( _PSEC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PSEC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = SEC_REG_RD32(LW_PSEC_FALCON_IDLESTATE);

    if ( DRF_VAL( _PSEC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PSEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = SEC_REG_RD32(LW_PSEC_FALCON_FHSTATE);
 
    if ( DRF_VAL( _PSEC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PSEC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PSEC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PSEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PSEC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PSEC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = SEC_REG_RD32(LW_PSEC_FALCON_ENGCTL);
    
    if ( DRF_VAL( _PSEC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PSEC_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PSEC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PSEC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = SEC_REG_RD32(LW_PSEC_FALCON_CPUCTL);

    if ( DRF_VAL( _PSEC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PSEC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PSEC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PSEC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PSEC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PSEC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = SEC_REG_RD32(LW_PSEC_FALCON_ITFEN);

    if (DRF_VAL( _PSEC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(secBaseAddress, "PSEC") == LW_ERR_GENERIC)
        {
            dprintf("lw: Current ctx state invalid\n");
            addUnitErr("\t Current ctx state invalid\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: Current ctx state valid\n");
        }
    }
    else
    {
        dprintf("lw: + LW_PSEC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PSEC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PSEC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PSEC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(secBaseAddress, "PSEC") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;  
}

//-----------------------------------------------------
// secPrintPriv_T194
//-----------------------------------------------------
void secPrintPriv_T194(LwU32 clmn, char *tag, LwU32 id)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n",id,SEC_REG_RD32(id));
}

//-----------------------------------------------------
// secDumpPriv_T194 - Dumps SEC priv reg space
//-----------------------------------------------------
LW_STATUS secDumpPriv_T194(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u SEC priv registers -- \n", indexGpu);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(pSecPrivReg[u].m_id==0)
        {
            break;
        }
        
        pCipher[indexGpu].secPrintPriv(40,pSecPrivReg[u].m_tag,pSecPrivReg[u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// secDisplayHwcfg_T194 - Display SEC HW config
//--------------------------------------------------------
LW_STATUS secDisplayHwcfg_T194(LwU32 indexGpu)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u SEC HWCFG -- \n", indexGpu);
    dprintf("lw:\n");

    hwcfg  = SEC_REG_RD32(LW_PSEC_FALCON_HWCFG);
    dprintf("lw: LW_PSEC_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PSEC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
            DRF_VAL(_PSEC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PSEC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg), 
            DRF_VAL(_PSEC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PSEC, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg)); 
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PSEC, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg)); 

    dprintf("lw:\n");

    hwcfg1 = SEC_REG_RD32(LW_PSEC_FALCON_HWCFG1);
    dprintf("lw: LW_PSEC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_PSEC, _FALCON_HWCFG1, _CORE_REV, hwcfg1)); 
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PSEC, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1)); 
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PSEC, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1)); 
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PSEC, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1)); 
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PSEC, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1)); 

    return LW_OK;  
}

 /*
0   IV0
1   IV1
3   EV
4   SP
5   PC
6   IMB
7   DMB
8   CSW
*/
// indx taken from Falcon 4.0 arch Table 3
LW_STATUS  secDisplayFlcnSPR_T194(LwU32 indexGpu)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u SEC Special Purpose Registers -- \n", indexGpu);
    dprintf("lw:\n");

    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1008);
    dprintf("lw: SEC IV0 :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1108);
    dprintf("lw: SEC IV1 :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1308);
    dprintf("lw: SEC EV  :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1408);
    dprintf("lw: SEC SP  :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1508);
    dprintf("lw: SEC PC  :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1608);
    dprintf("lw: SEC IMB :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1708);
    dprintf("lw: SEC DMB :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    SEC_REG_WR32(LW_PSEC_FALCON_ICD_CMD, 0x1808);
    dprintf("lw: SEC CSW :    0x%08x\n", SEC_REG_RD32(LW_PSEC_FALCON_ICD_RDATA)); 
    dprintf("lw:\n\n");

    return LW_OK; 
}

//-----------------------------------------------------
// secPrintMethodData_t114
//-----------------------------------------------------
void secPrintMethodData_t114(LwU32 clmn, char *tag, LwU32 method, LwU32 data)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n",method,data);
}

