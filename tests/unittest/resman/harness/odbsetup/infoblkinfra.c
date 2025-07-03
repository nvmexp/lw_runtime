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
 * @file   infoblkinfra.c
 * @brief  functions to UtApiSet the hal infoblock size of any sw gpu engine
 */

#include "infoblkinfra.h"

//
//shared instance between infra and test to
//specify the size of the infoblock of any
//given engine
//
engInfoBlockSize unitAllEngineInfoBlkSize = {0};

/*!
 *@brief set DPU hal Info Block Size
 *
 *@param[in]      size    size of DPUHALINFO block
 */
void utApiSetDpuInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.dpuInfoBlkSize) = size;
}

/*!
 *@brief set RES hal Info Block Size
 *
 *@param[in]      size    size of RESHALINFO block
 */
void utApiSetResInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.resInfoBlkSize) = size;
}

/*!
 *@brief set PGENG hal Info Block Size
 *
 *@param[in]      size    size of PGENGHALINFO block
 */
void utApiSetPgengInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.pgengInfoBlkSize) = size;
}

/*!
 *@brief set PGCTRL hal Info Block Size
 *
 *@param[in]      size    size of PGCTRLHALINFO block
 */
void utApiSetPgctrlInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.pgctrlInfoBlkSize) = size;
}

/*!
 *@brief set PG hal Info Block Size
 *
 *@param[in]      size    size of PGHALINFO block
 */
void utApiSetPgInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.pgInfoBlkSize) = size;
}

/*!
 *@brief set INFOROM hal Info Block Size
 *
 *@param[in]      size    size of INFOROMHALINFO block
 */
void utApiSetInforomInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.inforomInfoBlkSize) = size;
}

/*!
 *@brief set MSENC hal Info Block Size
 *
 *@param[in]      size    size of MSENCHALINFO block
 */
void utApiSetMsencInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.msencInfoBlkSize) = size;
}

/*!
 *@brief set VIC hal Info Block Size
 *
 *@param[in]      size    size of VICHALINFO block
 */
void utApiSetVicInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.vicInfoBlkSize) = size;
}

/*!
 *@brief set SPB hal Info Block Size
 *
 *@param[in]      size    size of SPBHALINFO block
 */
void utApiSetSpbInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.spbInfoBlkSize) = size;
}

/*!
 *@brief set PMU hal Info Block Size
 *
 *@param[in]      size    size of PMUHALINFO block
 */
void utApiSetPmuInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.pmuInfoBlkSize) = size;
}

/*!
 *@brief set CE hal Info Block Size
 *
 *@param[in]      size    size of CEHALINFO block
 */
void utApiSetCeInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.ceInfoBlkSize) = size;
}

/*!
 *@brief set ISOHUB hal Info Block Size
 *
 *@param[in]      size    size of ISOHUBHALINFO block
 */
void utApiSetIsohubInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.isohubInfoBlkSize) = size;
}

/*!
 *@brief set CVE hal Info Block Size
 *
 *@param[in]      size    size of CVEHALINFO block
 */
void utApiSetCveInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.cveInfoBlkSize) = size;
}

/*!
 *@brief set CIPHER hal Info Block Size
 *
 *@param[in]      size    size of CIPHERHALINFO block
 */
void utApiSetCipherInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.cipherInfoBlkSize) = size;
}

/*!
 *@brief set HDMI hal Info Block Size
 *
 *@param[in]      size    size of HDMIHALINFO block
 */
void utApiSetHdmiInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.hdmiInfoBlkSize) = size;
}

/*!
 *@brief set HDCP hal Info Block Size
 *
 *@param[in]      size    size of HDCPHALINFO block
 */
void utApiSetHdcpInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.hdcpInfoBlkSize) = size;
}

/*!
 *@brief set HDTV hal Info Block Size
 *
 *@param[in]      size    size of HDTVHALINFO block
 */
void utApiSetHdtvInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.hdtvInfoBlkSize) = size;
}

/*!
 *@brief set VP hal Info Block Size
 *
 *@param[in]      size    size of VPHALINFO block
 */
void utApiSetVpInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.vpInfoBlkSize) = size;
}

/*!
 *@brief set VIDEO hal Info Block Size
 *
 *@param[in]      size    size of VIDEOHALINFO block
 */
void utApiSetVideoInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.videoInfoBlkSize) = size;
}

/*!
 *@brief set MP hal Info Block Size
 *
 *@param[in]      size    size of MPHALINFO block
 */
void utApiSetMpInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.mpInfoBlkSize) = size;
}

/*!
 *@brief set MPEG hal Info Block Size
 *
 *@param[in]      size    size of MPEGHALINFO block
 */
void utApiSetMpegInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.mpegInfoBlkSize) = size;
}

/*!
 *@brief set BSP hal Info Block Size
 *
 *@param[in]      size    size of BSPHALINFO block
 */
void utApiSetBspInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.bspInfoBlkSize) = size;
}

/*!
 *@brief set SMU hal Info Block Size
 *
 *@param[in]      size    size of SMUHALINFO block
 */
void utApiSetSmuInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.smuInfoBlkSize) = size;
}

/*!
 *@brief set SOR hal Info Block Size
 *
 *@param[in]      size    size of SORHALINFO block
 */
void utApiSetSorInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.sorInfoBlkSize) = size;
}

/*!
 *@brief set PIOR hal Info Block Size
 *
 *@param[in]      size    size of PIORHALINFO block
 */
void utApiSetPiorInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.piorInfoBlkSize) = size;
}

/*!
 *@brief set OR hal Info Block Size
 *
 *@param[in]      size    size of ORHALINFO block
 */
void utApiSetOrInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.orInfoBlkSize) = size;
}

/*!
 *@brief set THERM hal Info Block Size
 *
 *@param[in]      size    size of THERMHALINFO block
 */
void utApiSetThermInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.thermInfoBlkSize) = size;
}

/*!
 *@brief set VOLT hal Info Block Size
 *
 *@param[in]      size    size of VOLTHALINFO block
 */
void utApiSetVoltInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.voltInfoBlkSize) = size;
}

/*!
 *@brief set FUSE hal Info Block Size
 *
 *@param[in]      size    size of FUSEHALINFO block
 */
void utApiSetFuseInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.fuseInfoBlkSize) = size;
}

/*!
 *@brief set FAN hal Info Block Size
 *
 *@param[in]      size    size of FANHALINFO block
 */
void utApiSetFanInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.fanInfoBlkSize) = size;
}

/*!
 *@brief set GPIO hal Info Block Size
 *
 *@param[in]      size    size of GPIOHALINFO block
 */
void utApiSetGpioInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.gpioInfoBlkSize) = size;
}

/*!
 *@brief set I2C hal Info Block Size
 *
 *@param[in]      size    size of I2CHALINFO block
 */
void utApiSetI2cInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.i2cInfoBlkSize) = size;
}

/*!
 *@brief set GPU hal Info Block Size
 *
 *@param[in]      size    size of GPUHALINFO block
 */
void utApiSetGpuInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.gpuInfoBlkSize) = size;
}

/*!
 *@brief set SW hal Info Block Size
 *
 *@param[in]      size    size of SWHALINFO block
 */
void utApiSetSwInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.swInfoBlkSize) = size;
}

/*!
 *@brief set RC hal Info Block Size
 *
 *@param[in]      size    size of RCHALINFO block
 */
void utApiSetRcInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.rcInfoBlkSize) = size;
}

/*!
 *@brief set VBIOS hal Info Block Size
 *
 *@param[in]      size    size of VBIOSHALINFO block
 */
void utApiSetVbiosInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.vbiosInfoBlkSize) = size;
}

/*!
 *@brief set VGA hal Info Block Size
 *
 *@param[in]      size    size of VGAHALINFO block
 */
void utApiSetVgaInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.vgaInfoBlkSize) = size;
}

/*!
 *@brief set PPP hal Info Block Size
 *
 *@param[in]      size    size of PPPHALINFO block
 */
void utApiSetPppInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.pppInfoBlkSize) = size;
}

/*!
 *@brief set SEQ hal Info Block Size
 *
 *@param[in]      size    size of SEQHALINFO block
 */
void utApiSetSeqInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.seqInfoBlkSize) = size;
}

/*!
 *@brief set TMR hal Info Block Size
 *
 *@param[in]      size    size of TMRHALINFO block
 */
void utApiSetTmrInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.tmrInfoBlkSize) = size;
}

/*!
 *@brief set STEREO hal Info Block Size
 *
 *@param[in]      size    size of STEREOHALINFO block
 */
void utApiSetStereoInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.stereoInfoBlkSize) = size;
}

/*!
 *@brief set SS hal Info Block Size
 *
 *@param[in]      size    size of SSHALINFO block
 */
void utApiSetSsInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.ssInfoBlkSize) = size;
}

/*!
 *@brief set PERF hal Info Block Size
 *
 *@param[in]      size    size of PERFHALINFO block
 */
void utApiSetPerfInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.perfInfoBlkSize) = size;
}

/*!
 *@brief set MC hal Info Block Size
 *
 *@param[in]      size    size of MCHALINFO block
 */
void utApiSetMcInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.mcInfoBlkSize) = size;
}

/*!
 *@brief set INTR hal Info Block Size
 *
 *@param[in]      size    size of INTRHALINFO block
 */
void utApiSetIntrInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.intrInfoBlkSize) = size;
}

/*!
 *@brief set INST hal Info Block Size
 *
 *@param[in]      size    size of INSTHALINFO block
 */
void utApiSetInstInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.instInfoBlkSize) = size;
}

/*!
 *@brief set HEAD hal Info Block Size
 *
 *@param[in]      size    size of HEADHALINFO block
 */
void utApiSetHeadInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.headInfoBlkSize) = size;
}

/*!
 *@brief set HBLOAT hal Info Block Size
 *
 *@param[in]      size    size of HBLOATHALINFO block
 */
void utApiSetHbloatInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.hbloatInfoBlkSize) = size;
}

/*!
 *@brief set GR hal Info Block Size
 *
 *@param[in]      size    size of GRHALINFO block
 */
void utApiSetGrInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.grInfoBlkSize) = size;
}

/*!
 *@brief set FLCN hal Info Block Size
 *
 *@param[in]      size    size of FLCNHALINFO block
 */
void utApiSetFlcnInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.flcnInfoBlkSize) = size;
}

/*!
 *@brief set FIFO hal Info Block Size
 *
 *@param[in]      size    size of FIFOHALINFO block
 */
void utApiSetFifoInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.fifoInfoBlkSize) = size;
}

/*!
 *@brief set FBSR hal Info Block Size
 *
 *@param[in]      size    size of FBSRHALINFO block
 */
void utApiSetFbsrInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.fbsrInfoBlkSize) = size;
}

/*!
 *@brief set FB hal Info Block Size
 *
 *@param[in]      size    size of FBHALINFO block
 */
void utApiSetFbInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.fbInfoBlkSize) = size;
}

/*!
 *@brief set FB falcon hal Info Block Size
 *
 *@param[in]      size    size of FBFLCNHALINFO block
 */
void utApiSetFbflcnInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.fbflcnInfoBlkSize) = size;
}

/*!
 *@brief set DPLINK hal Info Block Size
 *
 *@param[in]      size    size of DPLINKHALINFO block
 */
void utApiSetDplinkInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.dplinkInfoBlkSize) = size;
}

/*!
 *@brief set DPAUX hal Info Block Size
 *
 *@param[in]      size    size of DPAUXHALINFO block
 */
void utApiSetDpauxInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.dpauxInfoBlkSize) = size;
}

/*!
 *@brief set DMA hal Info Block Size
 *
 *@param[in]      size    size of DMAHALINFO block
 */
void utApiSetDmaInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.dmaInfoBlkSize) = size;
}

/*!
 *@brief set DISP hal Info Block Size
 *
 *@param[in]      size    size of DISPHALINFO block
 */
void utApiSetDispInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.dispInfoBlkSize) = size;
}

/*!
 *@brief set DAC hal Info Block Size
 *
 *@param[in]      size    size of DACHALINFO block
 */
void utApiSetdacInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.dacInfoBlkSize) = size;
}

/*!
 *@brief set OLDDISP hal Info Block Size
 *
 *@param[in]      size    size of OLDDISPHALINFO block
 */
void utApiSetOlddispInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.olddispInfoBlkSize) = size;
}

/*!
 *@brief set CLK hal Info Block Size
 *
 *@param[in]      size    size of CLKHALINFO block
 */
void utApiSetClkInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.clkInfoBlkSize) = size;
}

/*!
 *@brief set BUS hal Info Block Size
 *
 *@param[in]      size    size of BUSHALINFO block
 */
void utApiSetBusInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.busInfoBlkSize) = size;
}

/*!
 *@brief set BIF hal Info Block Size
 *
 *@param[in]      size    size of BIFHALINFO block
 */
void utApiSetBifInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.bifInfoBlkSize) = size;
}

/*!
 *@brief set LWJPG hal Info Block Size
 *
 *@param[in]      size    size of LWJPGHALINFO block
 */
void utApiSetLwjpgInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.lwjpgInfoBlkSize) = size;
}

/*!
 *@brief set OFA hal Info Block Size
 *
 *@param[in]      size    size of OFAHALINFO block
 */
void utApiSetOfaInfoBlkSize(LwU32 size)
{
    (unitAllEngineInfoBlkSize.ofaInfoBlkSize) = size;
}

