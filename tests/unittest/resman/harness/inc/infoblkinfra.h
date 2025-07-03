/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   infoblkinfra.h
 * @brief  declaration of structure to store the size of various engines
 *         declaration of functions to UtApiSet the size of various engines
 */

#ifndef _INFOBLKINFRA_H_
#define _INFOBLKINFRA_H_

#include "lwtypes.h"

//
//structure containig the sizes of hal info block
//for each engine, used for construction for infoblock
//
typedef struct
{
    LwU32 dpuInfoBlkSize;
    LwU32 resInfoBlkSize;
    LwU32 pgengInfoBlkSize;
    LwU32 pgctrlInfoBlkSize;
    LwU32 pgInfoBlkSize;
    LwU32 inforomInfoBlkSize;
    LwU32 msencInfoBlkSize;
    LwU32 vicInfoBlkSize;
    LwU32 spbInfoBlkSize;
    LwU32 pmuInfoBlkSize;
    LwU32 ceInfoBlkSize;
    LwU32 isohubInfoBlkSize;
    LwU32 cveInfoBlkSize;
    LwU32 cipherInfoBlkSize;
    LwU32 hdmiInfoBlkSize;
    LwU32 hdcpInfoBlkSize;
    LwU32 hdtvInfoBlkSize;
    LwU32 vpInfoBlkSize;
    LwU32 videoInfoBlkSize;
    LwU32 mpInfoBlkSize;
    LwU32 mpegInfoBlkSize;
    LwU32 bspInfoBlkSize;
    LwU32 smuInfoBlkSize;
    LwU32 sorInfoBlkSize;
    LwU32 piorInfoBlkSize;
    LwU32 orInfoBlkSize;
    LwU32 thermInfoBlkSize;
    LwU32 voltInfoBlkSize;
    LwU32 fuseInfoBlkSize;
    LwU32 fanInfoBlkSize;
    LwU32 gpioInfoBlkSize;
    LwU32 i2cInfoBlkSize;
    LwU32 gpuInfoBlkSize;
    LwU32 swInfoBlkSize;
    LwU32 rcInfoBlkSize;
    LwU32 vbiosInfoBlkSize;
    LwU32 vgaInfoBlkSize;
    LwU32 pppInfoBlkSize;
    LwU32 seqInfoBlkSize;
    LwU32 tmrInfoBlkSize;
    LwU32 stereoInfoBlkSize;
    LwU32 ssInfoBlkSize;
    LwU32 perfInfoBlkSize;
    LwU32 mcInfoBlkSize;
    LwU32 intrInfoBlkSize;
    LwU32 instInfoBlkSize;
    LwU32 headInfoBlkSize;
    LwU32 hbloatInfoBlkSize;
    LwU32 grInfoBlkSize;
    LwU32 flcnInfoBlkSize;
    LwU32 fifoInfoBlkSize;
    LwU32 fbsrInfoBlkSize;
    LwU32 fbInfoBlkSize;
    LwU32 fbflcnInfoBlkSize;
    LwU32 dplinkInfoBlkSize;
    LwU32 dpauxInfoBlkSize;
    LwU32 dmaInfoBlkSize;
    LwU32 dispInfoBlkSize;
    LwU32 dacInfoBlkSize;
    LwU32 olddispInfoBlkSize;
    LwU32 clkInfoBlkSize;
    LwU32 busInfoBlkSize;
    LwU32 bifInfoBlkSize;
    LwU32 lwjpgInfoBlkSize;
    LwU32 ofaInfoBlkSize;
} engInfoBlockSize;

extern engInfoBlockSize unitAllEngineInfoBlkSize;

// set DPU hal Info Block Size
void utApiSetDpuInfoBlkSize(LwU32 size);
#define UTAPI_DPU_INFO_BLK_SIZE(size) utApiSetDpuInfoBlkSize(size);

// set RES hal Info Block Size
void utApiSetResInfoBlkSize(LwU32 size);
#define UTAPI_RES_INFO_BLK_SIZE(size) utApiSetResInfoBlkSize(size);

// set PGENG hal Info Block Size
void utApiSetPgengInfoBlkSize(LwU32 size);
#define UTAPI_PGENG_INFO_BLK_SIZE(size) utApiSetPgengInfoBlkSize(size);

// set PGCTRL hal Info Block Size
void utApiSetPgctrlInfoBlkSize(LwU32 size);
#define UTAPI_PGCTRL_INFO_BLK_SIZE(size) utApiSetPgctrlInfoBlkSize(size);

// set PG hal Info Block Size
void utApiSetPgInfoBlkSize(LwU32 size);
#define UTAPI_PG_INFO_BLK_SIZE(size) utApiSetPgInfoBlkSize(size);

// set INFOROM hal Info Block Size
void utApiSetInforomInfoBlkSize(LwU32 size);
#define UTAPI_INFOROM_INFO_BLK_SIZE(size) utApiSetInforomInfoBlkSize(size);

// set MSENC hal Info Block Size
void utApiSetMsencInfoBlkSize(LwU32 size);
#define UTAPI_MSENC_INFO_BLK_SIZE(size) utApiSetMsencInfoBlkSize(size);

// set VIC hal Info Block Size
void utApiSetVicInfoBlkSize(LwU32 size);
#define UTAPI_VIC_INFO_BLK_SIZE(size) utApiSetVicInfoBlkSize(size);

// set SPB hal Info Block Size
void utApiSetSpbInfoBlkSize(LwU32 size);
#define UTAPI_SPB_INFO_BLK_SIZE(size) utApiSetSpbInfoBlkSize(size);

// set PMU hal Info Block Size
void utApiSetPmuInfoBlkSize(LwU32 size);
#define UTAPI_PMU_INFO_BLK_SIZE(size) utApiSetPmuInfoBlkSize(size);

// set CE hal Info Block Size
void utApiSetCeInfoBlkSize(LwU32 size);
#define UTAPI_CE_INFO_BLK_SIZE(size) utApiSetCeInfoBlkSize(size);

// set ISOHUB hal Info Block Size
void utApiSetIsohubInfoBlkSize(LwU32 size);
#define UTAPI_ISOHUB_INFO_BLK_SIZE(size) utApiSetIsohubInfoBlkSize(size);

// set CVE hal Info Block Size
void utApiSetCveInfoBlkSize(LwU32 size);
#define UTAPI_CVE_INFO_BLK_SIZE(size) utApiSetCveInfoBlkSize(size);

// set CIPHER hal Info Block Size
void utApiSetCipherInfoBlkSize(LwU32 size);
#define UTAPI_CIPHER_INFO_BLK_SIZE(size) utApiSetCipherInfoBlkSize(size);

// set HDMI hal Info Block Size
void utApiSetHdmiInfoBlkSize(LwU32 size);
#define UTAPI_HDMI_INFO_BLK_SIZE(size) utApiSetHdmiInfoBlkSize(size);

// set HDCP hal Info Block Size
void utApiSetHdcpInfoBlkSize(LwU32 size);
#define UTAPI_HDCP_INFO_BLK_SIZE(size) utApiSetHdcpInfoBlkSize(size);

// set HDTV hal Info Block Size
void utApiSetHdtvInfoBlkSize(LwU32 size);
#define UTAPI_HDTV_INFO_BLK_SIZE(size) utApiSetHdtvInfoBlkSize(size);

// set VP hal Info Block Size
void utApiSetVpInfoBlkSize(LwU32 size);
#define UTAPI_VP_INFO_BLK_SIZE(size) utApiSetVpInfoBlkSize(size);

// set VIDEO hal Info Block Size
void utApiSetVideoInfoBlkSize(LwU32 size);
#define UTAPI_VIDEO_INFO_BLK_SIZE(size) utApiSetVideoInfoBlkSize(size);

// set MP hal Info Block Size
void utApiSetMpInfoBlkSize(LwU32 size);
#define UTAPI_MP_INFO_BLK_SIZE(size) utApiSetMpInfoBlkSize(size);

// set MPEG hal Info Block Size
void utApiSetMpegInfoBlkSize(LwU32 size);
#define UTAPI_MPEG_INFO_BLK_SIZE(size) utApiSetMpegInfoBlkSize(size);

// set BSP hal Info Block Size
void utApiSetBspInfoBlkSize(LwU32 size);
#define UTAPI_BSP_INFO_BLK_SIZE(size) utApiSetBspInfoBlkSize(size);

// set SMU hal Info Block Size
void utApiSetSmuInfoBlkSize(LwU32 size);
#define UTAPI_SMU_INFO_BLK_SIZE(size) utApiSetSmuInfoBlkSize(size);

// set SOR hal Info Block Size
void utApiSetSorInfoBlkSize(LwU32 size);
#define UTAPI_SOR_INFO_BLK_SIZE(size) utApiSetSorInfoBlkSize(size);

// set PIOR hal Info Block Size
void utApiSetPiorInfoBlkSize(LwU32 size);
#define UTAPI_PIOR_INFO_BLK_SIZE(size) utApiSetPiorInfoBlkSize(size);

// set OR hal Info Block Size
void utApiSetOrInfoBlkSize(LwU32 size);
#define UTAPI_OR_INFO_BLK_SIZE(size) utApiSetOrInfoBlkSize(size);

// set THERM hal Info Block Size
void utApiSetThermInfoBlkSize(LwU32 size);
#define UTAPI_THERM_INFO_BLK_SIZE(size) utApiSetThermInfoBlkSize(size);

// set VOLT hal Info Block Size
void utApiSetVoltInfoBlkSize(LwU32 size);
#define UTAPI_VOLT_INFO_BLK_SIZE(size) utApiSetVoltInfoBlkSize(size);

// set FUSE hal Info Block Size
void utApiSetFuseInfoBlkSize(LwU32 size);
#define UTAPI_FUSE_INFO_BLK_SIZE(size) utApiSetFuseInfoBlkSize(size);

// set FAN hal Info Block Size
void utApiSetFanInfoBlkSize(LwU32 size);
#define UTAPI_FAN_INFO_BLK_SIZE(size) utApiSetFanInfoBlkSize(size);

// set GPIO hal Info Block Size
void utApiSetGpioInfoBlkSize(LwU32 size);
#define UTAPI_GPIO_INFO_BLK_SIZE(size) utApiSetGpioInfoBlkSize(size);

// set I2C hal Info Block Size
void utApiSetI2cInfoBlkSize(LwU32 size);
#define UTAPI_I2C_INFO_BLK_SIZE(size) utApiSetI2cInfoBlkSize(size);

// set GPU hal Info Block Size
void utApiSetGpuInfoBlkSize(LwU32 size);
#define UTAPI_GPU_INFO_BLK_SIZE(size) utApiSetGpuInfoBlkSize(size);

// set SW hal Info Block Size
void utApiSetSwInfoBlkSize(LwU32 size);
#define UTAPI_SW_INFO_BLK_SIZE(size) utApiSetSwInfoBlkSize(size);

// set RC hal Info Block Size
void utApiSetRcInfoBlkSize(LwU32 size);
#define UTAPI_RC_INFO_BLK_SIZE(size) utApiSetRcInfoBlkSize(size);

// set VBIOS hal Info Block Size
void utApiSetVbiosInfoBlkSize(LwU32 size);
#define UTAPI_VBIOS_INFO_BLK_SIZE(size) utApiSetVbiosInfoBlkSize(size);

// set VGA hal Info Block Size
void utApiSetVgaInfoBlkSize(LwU32 size);
#define UTAPI_VGA_INFO_BLK_SIZE(size) utApiSetVgaInfoBlkSize(size);

// set PPP hal Info Block Size
void utApiSetPppInfoBlkSize(LwU32 size);
#define UTAPI_PPP_INFO_BLK_SIZE(size) utApiSetPppInfoBlkSize(size);

// set SEQ hal Info Block Size
void utApiSetSeqInfoBlkSize(LwU32 size);
#define UTAPI_SEQ_INFO_BLK_SIZE(size) utApiSetSeqInfoBlkSize(size);

// set TMR hal Info Block Size
void utApiSetTmrInfoBlkSize(LwU32 size);
#define UTAPI_TMR_INFO_BLK_SIZE(size) utApiSetTmrInfoBlkSize(size);

// set STEREO hal Info Block Size
void utApiSetStereoInfoBlkSize(LwU32 size);
#define UTAPI_STEREO_INFO_BLK_SIZE(size) utApiSetStereoInfoBlkSize(size);

// set SS hal Info Block Size
void utApiSetSsInfoBlkSize(LwU32 size);
#define UTAPI_SS_INFO_BLK_SIZE(size) utApiSetSsInfoBlkSize(size);

// set PERF hal Info Block Size
void utApiSetPerfInfoBlkSize(LwU32 size);
#define UTAPI_PERF_INFO_BLK_SIZE(size) utApiSetPerfInfoBlkSize(size);

// set MC hal Info Block Size
void utApiSetMcInfoBlkSize(LwU32 size);
#define UTAPI_MC_INFO_BLK_SIZE(size) utApiSetMcInfoBlkSize(size);

// set INTR hal Info Block Size
void utApiSetIntrInfoBlkSize(LwU32 size);
#define UTAPI_INTR_INFO_BLK_SIZE(size) utApiSetIntrInfoBlkSize(size);

// set INST hal Info Block Size
void utApiSetInstInfoBlkSize(LwU32 size);
#define UTAPI_INST_INFO_BLK_SIZE(size) utApiSetInstInfoBlkSize(size);

// set HEAD hal Info Block Size
void utApiSetHeadInfoBlkSize(LwU32 size);
#define UTAPI_HEAD_INFO_BLK_SIZE(size) utApiSetHeadInfoBlkSize(size);

// set HBLOAT hal Info Block Size
void utApiSetHbloatInfoBlkSize(LwU32 size);
#define UTAPI_HBLOAT_INFO_BLK_SIZE(size) utApiSetHbloatInfoBlkSize(size);

// set GR hal Info Block Size
void utApiSetGrInfoBlkSize(LwU32 size);
#define UTAPI_GR_INFO_BLK_SIZE(size) utApiSetGrInfoBlkSize(size);

// set FLCN hal Info Block Size
void utApiSetFlcnInfoBlkSize(LwU32 size);
#define UTAPI_FLCN_INFO_BLK_SIZE(size) utApiSetFlcnInfoBlkSize(size);

// set FIFO hal Info Block Size
void utApiSetFifoInfoBlkSize(LwU32 size);
#define UTAPI_FIFO_INFO_BLK_SIZE(size) utApiSetFifoInfoBlkSize(size);

// set FBSR hal Info Block Size
void utApiSetFbsrInfoBlkSize(LwU32 size);
#define UTAPI_FBSR_INFO_BLK_SIZE(size) utApiSetFbsrInfoBlkSize(size);

// set FB hal Info Block Size
void utApiSetFbInfoBlkSize(LwU32 size);
#define UTAPI_FB_INFO_BLK_SIZE(size) utApiSetFbInfoBlkSize(size);

// set FB falcon hal Info Block Size
void utApiSetFbflcnInfoBlkSize(LwU32 size);
#define UTAPI_FBFLCN_INFO_BLK_SIZE(size) utApiSetFbflcnInfoBlkSize(size);

// set DPLINK hal Info Block Size
void utApiSetDplinkInfoBlkSize(LwU32 size);
#define UTAPI_DPLINK_INFO_BLK_SIZE(size) utApiSetDplinkInfoBlkSize(size);

// set DPAUX hal Info Block Size
void utApiSetDpauxInfoBlkSize(LwU32 size);
#define UTAPI_DPAUX_INFO_BLK_SIZE(size) utApiSetDpauxInfoBlkSize(size);

// set DMA hal Info Block Size
void utApiSetDmaInfoBlkSize(LwU32 size);
#define UTAPI_DMA_INFO_BLK_SIZE(size) utApiSetDmaInfoBlkSize(size);

// set DISP hal Info Block Size
void utApiSetDispInfoBlkSize(LwU32 size);
#define UTAPI_DISP_INFO_BLK_SIZE(size) utApiSetDispInfoBlkSize(size);

// set DAC hal Info Block Size
void utApiSetDacInfoBlkSize(LwU32 size);
#define UTAPI_DAC_INFO_BLK_SIZE(size) utApiSetDacInfoBlkSize(size);

// set OLDDISP hal Info Block Size
void utApiSetOlddispInfoBlkSize(LwU32 size);
#define UTAPI_OLDDISP_INFO_BLK_SIZE(size) utApiSetOlddispInfoBlkSize(size);

// set CLK hal Info Block Size
void utApiSetClkInfoBlkSize(LwU32 size);
#define UTAPI_CLK_INFO_BLK_SIZE(size) utApiSetClkInfoBlkSize(size);

// set BUS hal Info Block Size
void utApiSetBusInfoBlkSize(LwU32 size);
#define UTAPI_BUS_INFO_BLK_SIZE(size) utApiSetBusInfoBlkSize(size);

// set BIF hal Info Block Size
void utApiSetBifInfoBlkSize(LwU32 size);
#define UTAPI_BIF_INFO_BLK_SIZE(size) utApiSetBifInfoBlkSize(size);

// set LWJPG hal Info Block Size
void utApiSetLwjpgInfoBlkSize(LwU32 size);
#define UTAPI_LWJPG_INFO_BLK_SIZE(size) utApiSetLwjpgInfoBlkSize(size);

// set OFA hal Info Block Size
void utApiSetOfaInfoBlkSize(LwU32 size);
#define UTAPI_OFA_INFO_BLK_SIZE(size) utApiSetOfaInfoBlkSize(size);

#endif // _INFOBLKINFRA_H_
