/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file
 * @brief  functions to create & init any given engine object
 */
#include "odbinfra.h"
#include "unitodb.h"
#include "regops.h"
#include "utility.h"

// staic var or hold chip
static CHIP unitTestChip;

//
// redirect the gpu read/write register
// operations to infra implementaion
//
static void redirectGpuRegisterOperations(POBJGPU pGpu);

// Globals object to be shared between infra and test
rmObject unitTestRmObject = {0};
rmInfoBlock unitTestRmInfoBlock = {0};

//
// object to store the list of missing engines
//
// here, DATA_ID_FREE_OBJ is the last element
// in the enum list. So it should ideally be the
// size of this array
//
static LwBool unitMissingEngineBlock[DATA_ID_FREE_OBJ] = {0};

/*!
 * @brief reset above var(unitMissingEngineBlock) to NULL
 *
 */
void resetMissingEngineBlock()
{
    memset(unitMissingEngineBlock, 0, sizeof(LwBool)*DATA_ID_FREE_OBJ);
}

//
// static var to check wheteher chip specific
// PDB properties have to be initialized or not
//
LwBool pdbInit = FALSE;

/*!
 * @brief set above var to true/false
 *
 * @param[in]      val    value to set
 */
void setPdbInit(LwBool val)
{
    pdbInit = val;
}

/*!
 * @brief set chip version
 *
 * @param[in]      chip    enum of the chip name
 */
void setUnitTestChip(CHIP chip)
{
        unitTestChip = chip;
}

/*!
 * @brief Update the module descriptor on the basis of specified chip
 *
 * @param[in]      pMod    pointer to moduloe descriptor
 */
void useChip(PMODULEDESCRIPTOR pMod)
{
    switch (unitTestChip)
    {
#if RMCFG_CHIP_ENABLED(G84)
        case G84:
            pMod->pHalSetIfaces             = &halIface_G84;
            break;
#endif

#if RMCFG_CHIP_ENABLED(GF100)
        case GF100:
            pMod->pHalSetIfaces             = &halIface_GF100;
            break;
#endif

#if RMCFG_CHIP_ENABLED(GK100)
        case GK100:
            pMod->pHalSetIfaces             = &halIface_GK100;
            break;
#endif

        default:
            UNIT_ASSERT(0);
            break;
    }
}

/*!
 * @brief create the generic infoblock
 *
 * @param[in]      pHPI    address of pointer to infoblock
 *
 * @param[in]      size    size of the infoblock
 */
void createInfoBlock(void **pDHPI, LwU32 size)
{
    if (size > 0)
    {
        // allocate memory of size "size"
        *pDHPI = (void*)unitMalloc(size, UNIT_CLASS_INFOBLK);
        memset(*pDHPI, 0, size);
    }
}

/*!
 * @brief redirected function for returning the hal info block
 *
 * @param[in]      head    pointer head of the eng info link node
 *
 * @param[in]      dataId  enum to diiferentiate between different
                           engine objects
 *
 * @return         pointer to the info block of requested engine
 */
void *
getInfloblockStub(PENG_INFO_LINK_NODE head, LwU32 dataId)
{
    static void *pDpuHPI = NULL;
    static void *pResHPI = NULL;
    static void *pPgengHPI = NULL;
    static void *pPgctrlHPI = NULL;
    static void *pPgHPI = NULL;
    static void *pInforomHPI = NULL;
    static void *pMsencHPI = NULL;
    static void *pVicHPI = NULL;
    static void *pSpbHPI = NULL;
    static void *pPmuHPI = NULL;
    static void *pCeHPI = NULL;
    static void *pIsohubHPI = NULL;
    static void *pCveHPI = NULL;
    static void *pCipherHPI = NULL;
    static void *pHdmiHPI = NULL;
    static void *pHdcpHPI = NULL;
    static void *pHdtvHPI = NULL;
    static void *pVpHPI = NULL;
    static void *pVideoHPI = NULL;
    static void *pMpHPI = NULL;
    static void *pMpegHPI = NULL;
    static void *pBspHPI = NULL;
    static void *pSmuHPI = NULL;
    static void *pSorHPI = NULL;
    static void *pPiorHPI = NULL;
    static void *pOrHPI = NULL;
    static void *pThermHPI = NULL;
    static void *pVoltHPI = NULL;
    static void *pFuseHPI = NULL;
    static void *pFanHPI = NULL;
    static void *pGpioHPI = NULL;
    static void *pI2cHPI = NULL;
    static void *pGpuHPI = NULL;
    static void *pSwHPI = NULL;
    static void *pRcHPI = NULL;
    static void *pVbiosHPI = NULL;
    static void *pVgaHPI = NULL;
    static void *pPppHPI = NULL;
    static void *pSeqHPI = NULL;
    static void *pTmrHPI = NULL;
    static void *pStereoHPI = NULL;
    static void *pPerfHPI = NULL;
    static void *pMcHPI = NULL;
    static void *pIntrHPI = NULL;
    static void *pInstHPI = NULL;
    static void *pHeadHPI = NULL;
    static void *pGrHPI = NULL;
    static void *pFlcnHPI = NULL;
    static void *pFifoHPI = NULL;
    static void *pFbsrHPI = NULL;
    static void *pFbHPI = NULL;
    static void *pFbflcnHPI = NULL;
    static void *pDplinkHPI = NULL;
    static void *pDpauxHPI = NULL;
    static void *pDmaHPI = NULL;
    static void *pDispHPI = NULL;
    static void *pDacHPI = NULL;
    static void *pClkHPI = NULL;
    static void *pBusHPI = NULL;
    static void *pBifHPI = NULL;
    static void *pLwjpgHPI = NULL;
    static void *pOfaHPI = NULL;

    if (head)
    {
        switch(head->dataId)
        {
        case DATA_ID_DPU:

            if (!pDpuHPI)
            {

                // allocate memory for dpu hal info block
                createInfoBlock(&pDpuHPI, unitAllEngineInfoBlkSize.dpuInfoBlkSize);

                if (pDpuHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillDpuHalInfoBlock)
                        unitTestRmInfoBlock.fillDpuHalInfoBlock(pDpuHPI);
                }
            }

            return (void *)pDpuHPI;
            break;

        case DATA_ID_RES:

            if (!pResHPI)
            {

                // allocate memory for res hal info block
                createInfoBlock(&pResHPI, unitAllEngineInfoBlkSize.resInfoBlkSize);

                if (pResHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillResHalInfoBlock)
                        unitTestRmInfoBlock.fillResHalInfoBlock(pResHPI);
                }
            }

            return (void *)pResHPI;
            break;

        case DATA_ID_PGENG:

            if (!pPgengHPI)
            {

                // allocate memory for pgeng hal info block
                createInfoBlock(&pPgengHPI, unitAllEngineInfoBlkSize.pgengInfoBlkSize);

                if (pPgengHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillPgengHalInfoBlock)
                        unitTestRmInfoBlock.fillPgengHalInfoBlock(pPgengHPI);
                }
            }

            return (void *)pPgengHPI;
            break;

        case DATA_ID_PGCTRL:

            if (!pPgctrlHPI)
            {

                // allocate memory for pgctrl hal info block
                createInfoBlock(&pPgctrlHPI, unitAllEngineInfoBlkSize.pgctrlInfoBlkSize);

                if (pPgctrlHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillPgctrlHalInfoBlock)
                        unitTestRmInfoBlock.fillPgctrlHalInfoBlock(pPgctrlHPI);
                }
            }

            return (void *)pPgctrlHPI;
            break;

        case DATA_ID_PG:

            if (!pPgHPI)
            {

                // allocate memory for pg hal info block
                createInfoBlock(&pPgHPI, unitAllEngineInfoBlkSize.pgInfoBlkSize);

                if (pPgHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillPgHalInfoBlock)
                        unitTestRmInfoBlock.fillPgHalInfoBlock(pPgHPI);
                }
            }

            return (void *)pPgHPI;
            break;

        case DATA_ID_INFOROM:

            if (!pInforomHPI)
            {

                // allocate memory for inforom hal info block
                createInfoBlock(&pInforomHPI, unitAllEngineInfoBlkSize.inforomInfoBlkSize);

                if (pInforomHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillInforomHalInfoBlock)
                        unitTestRmInfoBlock.fillInforomHalInfoBlock(pInforomHPI);
                }
            }

            return (void *)pInforomHPI;
            break;

        case DATA_ID_MSENC:

            if (!pMsencHPI)
            {

                // allocate memory for msenc hal info block
                createInfoBlock(&pMsencHPI, unitAllEngineInfoBlkSize.msencInfoBlkSize);

                if (pMsencHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillMsencHalInfoBlock)
                        unitTestRmInfoBlock.fillMsencHalInfoBlock(pMsencHPI);
                }
            }

            return (void *)pMsencHPI;
            break;

        case DATA_ID_VIC:

            if (!pVicHPI)
            {

                // allocate memory for vic hal info block
                createInfoBlock(&pVicHPI, unitAllEngineInfoBlkSize.vicInfoBlkSize);

                if (pVicHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillVicHalInfoBlock)
                        unitTestRmInfoBlock.fillVicHalInfoBlock(pVicHPI);
                }
            }

            return (void *)pVicHPI;
            break;

        case DATA_ID_SPB:

            if (!pSpbHPI)
            {

                // allocate memory for spb hal info block
                createInfoBlock(&pSpbHPI, unitAllEngineInfoBlkSize.spbInfoBlkSize);

                if (pSpbHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillSpbHalInfoBlock)
                        unitTestRmInfoBlock.fillSpbHalInfoBlock(pSpbHPI);
                }
            }

            return (void *)pSpbHPI;
            break;

        case DATA_ID_PMU:

            if (!pPmuHPI)
            {

                // allocate memory for pmu hal info block
                createInfoBlock(&pPmuHPI, unitAllEngineInfoBlkSize.pmuInfoBlkSize);

                if (pPmuHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillPmuHalInfoBlock)
                        unitTestRmInfoBlock.fillPmuHalInfoBlock(pPmuHPI);
                }
            }

            return (void *)pPmuHPI;
            break;

        case DATA_ID_CE:

            if (!pCeHPI)
            {

                // allocate memory for ce1 hal info block
                createInfoBlock(&pCeHPI, unitAllEngineInfoBlkSize.ceInfoBlkSize);

                if (pCeHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillCeHalInfoBlock)
                        unitTestRmInfoBlock.fillCeHalInfoBlock(pCeHPI);
                }
            }

            return (void *)pCeHPI;
            break;

        case DATA_ID_ISOHUB:

            if (!pIsohubHPI)
            {

                // allocate memory for isohub hal info block
                createInfoBlock(&pIsohubHPI, unitAllEngineInfoBlkSize.isohubInfoBlkSize);

                if (pIsohubHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillIsohubHalInfoBlock)
                        unitTestRmInfoBlock.fillIsohubHalInfoBlock(pIsohubHPI);
                }
            }

            return (void *)pIsohubHPI;
            break;

        case DATA_ID_CVE:

            if (!pCveHPI)
            {

                // allocate memory for cve hal info block
                createInfoBlock(&pCveHPI, unitAllEngineInfoBlkSize.cveInfoBlkSize);

                if (pCveHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillCveHalInfoBlock)
                        unitTestRmInfoBlock.fillCveHalInfoBlock(pCveHPI);
                }
            }

            return (void *)pCveHPI;
            break;

        case DATA_ID_CIPHER:

            if (!pCipherHPI)
            {

                // allocate memory for cipher hal info block
                createInfoBlock(&pCipherHPI, unitAllEngineInfoBlkSize.cipherInfoBlkSize);

                if (pCipherHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillCipherHalInfoBlock)
                        unitTestRmInfoBlock.fillCipherHalInfoBlock(pCipherHPI);
                }
            }

            return (void *)pCipherHPI;
            break;

        case DATA_ID_HDMI:

            if (!pHdmiHPI)
            {

                // allocate memory for hdmi hal info block
                createInfoBlock(&pHdmiHPI, unitAllEngineInfoBlkSize.hdmiInfoBlkSize);

                if (pHdmiHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillHdmiHalInfoBlock)
                        unitTestRmInfoBlock.fillHdmiHalInfoBlock(pHdmiHPI);
                }
            }

            return (void *)pHdmiHPI;
            break;

        case DATA_ID_HDCP:

            if (!pHdcpHPI)
            {

                // allocate memory for hdcp hal info block
                createInfoBlock(&pHdcpHPI, unitAllEngineInfoBlkSize.hdcpInfoBlkSize);

                if (pHdcpHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillHdcpHalInfoBlock)
                        unitTestRmInfoBlock.fillHdcpHalInfoBlock(pHdcpHPI);
                }
            }

            return (void *)pHdcpHPI;
            break;

        case DATA_ID_HDTV:

            if (!pHdtvHPI)
            {

                // allocate memory for hdtv hal info block
                createInfoBlock(&pHdtvHPI, unitAllEngineInfoBlkSize.hdtvInfoBlkSize);

                if (pHdtvHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillHdtvHalInfoBlock)
                        unitTestRmInfoBlock.fillHdtvHalInfoBlock(pHdtvHPI);
                }
            }

            return (void *)pHdtvHPI;
            break;

        case DATA_ID_VP:

            if (!pVpHPI)
            {

                // allocate memory for vp hal info block
                createInfoBlock(&pVpHPI, unitAllEngineInfoBlkSize.vpInfoBlkSize);

                if (pVpHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillVpHalInfoBlock)
                        unitTestRmInfoBlock.fillVpHalInfoBlock(pVpHPI);
                }
            }

            return (void *)pVpHPI;
            break;

        case DATA_ID_VIDEO:

            if (!pVideoHPI)
            {

                // allocate memory for video hal info block
                createInfoBlock(&pVideoHPI, unitAllEngineInfoBlkSize.videoInfoBlkSize);

                if (pVideoHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillVideoHalInfoBlock)
                        unitTestRmInfoBlock.fillVideoHalInfoBlock(pVideoHPI);
                }
            }

            return (void *)pVideoHPI;
            break;

        case DATA_ID_MP:

            if (!pMpHPI)
            {

                // allocate memory for mp hal info block
                createInfoBlock(&pMpHPI, unitAllEngineInfoBlkSize.mpInfoBlkSize);

                if (pMpHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillMpHalInfoBlock)
                        unitTestRmInfoBlock.fillMpHalInfoBlock(pMpHPI);
                }
            }

            return (void *)pMpHPI;
            break;

        case DATA_ID_MPEG:

            if (!pMpegHPI)
            {

                // allocate memory for mpeg hal info block
                createInfoBlock(&pMpegHPI, unitAllEngineInfoBlkSize.mpegInfoBlkSize);

                if (pMpegHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillMpegHalInfoBlock)
                        unitTestRmInfoBlock.fillMpegHalInfoBlock(pMpegHPI);
                }
            }

            return (void *)pMpegHPI;
            break;

        case DATA_ID_BSP:

            if (!pBspHPI)
            {

                // allocate memory for bsp hal info block
                createInfoBlock(&pBspHPI, unitAllEngineInfoBlkSize.bspInfoBlkSize);

                if (pBspHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillBspHalInfoBlock)
                        unitTestRmInfoBlock.fillBspHalInfoBlock(pBspHPI);
                }
            }

            return (void *)pBspHPI;
            break;

        case DATA_ID_SMU:

            if (!pSmuHPI)
            {

                // allocate memory for smu hal info block
                createInfoBlock(&pSmuHPI, unitAllEngineInfoBlkSize.smuInfoBlkSize);

                if (pSmuHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillSmuHalInfoBlock)
                        unitTestRmInfoBlock.fillSmuHalInfoBlock(pSmuHPI);
                }
            }

            return (void *)pSmuHPI;
            break;

        case DATA_ID_SOR:

            if (!pSorHPI)
            {

                // allocate memory for sor hal info block
                createInfoBlock(&pSorHPI, unitAllEngineInfoBlkSize.sorInfoBlkSize);

                if (pSorHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillSorHalInfoBlock)
                        unitTestRmInfoBlock.fillSorHalInfoBlock(pSorHPI);
                }
            }

            return (void *)pSorHPI;
            break;

        case DATA_ID_PIOR:

            if (!pPiorHPI)
            {

                // allocate memory for pior hal info block
                createInfoBlock(&pPiorHPI, unitAllEngineInfoBlkSize.piorInfoBlkSize);

                if (pPiorHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillPiorHalInfoBlock)
                        unitTestRmInfoBlock.fillPiorHalInfoBlock(pPiorHPI);
                }
            }

            return (void *)pPiorHPI;
            break;

        case DATA_ID_OR:

            if (!pOrHPI)
            {

                // allocate memory for or hal info block
                createInfoBlock(&pOrHPI, unitAllEngineInfoBlkSize.orInfoBlkSize);

                if (pOrHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillOrHalInfoBlock)
                        unitTestRmInfoBlock.fillOrHalInfoBlock(pOrHPI);
                }
            }

            return (void *)pOrHPI;
            break;

        case DATA_ID_THERM:

            if (!pThermHPI)
            {

                // allocate memory for therm hal info block
                createInfoBlock(&pThermHPI, unitAllEngineInfoBlkSize.thermInfoBlkSize);

                if (pThermHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillThermHalInfoBlock)
                        unitTestRmInfoBlock.fillThermHalInfoBlock(pThermHPI);
                }
            }

            return (void *)pThermHPI;
            break;

        case DATA_ID_VOLT:

            if (!pVoltHPI)
            {

                // allocate memory for volt hal info block
                createInfoBlock(&pVoltHPI, unitAllEngineInfoBlkSize.voltInfoBlkSize);

                if (pVoltHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillVoltHalInfoBlock)
                        unitTestRmInfoBlock.fillVoltHalInfoBlock(pVoltHPI);
                }
            }

            return (void *)pVoltHPI;
            break;

        case DATA_ID_FUSE:

            if (!pFuseHPI)
            {

                // allocate memory for fuse hal info block
                createInfoBlock(&pFuseHPI, unitAllEngineInfoBlkSize.fuseInfoBlkSize);

                if (pFuseHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillFuseHalInfoBlock)
                        unitTestRmInfoBlock.fillFuseHalInfoBlock(pFuseHPI);
                }
            }

            return (void *)pFuseHPI;
            break;

        case DATA_ID_FAN:

            if (!pFanHPI)
            {

                // allocate memory for fan hal info block
                createInfoBlock(&pFanHPI, unitAllEngineInfoBlkSize.fanInfoBlkSize);

                if (pFanHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillFanHalInfoBlock)
                        unitTestRmInfoBlock.fillFanHalInfoBlock(pFanHPI);
                }
            }

            return (void *)pFanHPI;
            break;

        case DATA_ID_GPIO:

            if (!pGpioHPI)
            {

                // allocate memory for gpio hal info block
                createInfoBlock(&pGpioHPI, unitAllEngineInfoBlkSize.gpioInfoBlkSize);

                if (pGpioHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillGpioHalInfoBlock)
                        unitTestRmInfoBlock.fillGpioHalInfoBlock(pGpioHPI);
                }
            }

            return (void *)pGpioHPI;
            break;

        case DATA_ID_I2C:

            if (!pI2cHPI)
            {

                // allocate memory for i2c hal info block
                createInfoBlock(&pI2cHPI, unitAllEngineInfoBlkSize.i2cInfoBlkSize);

                if (pI2cHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillI2cHalInfoBlock)
                        unitTestRmInfoBlock.fillI2cHalInfoBlock(pI2cHPI);
                }
            }

            return (void *)pI2cHPI;
            break;

        case DATA_ID_GPU:

            if (!pGpuHPI)
            {

                // allocate memory for gpu hal info block
                createInfoBlock(&pGpuHPI, unitAllEngineInfoBlkSize.gpuInfoBlkSize);

                if (pGpuHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillGpuHalInfoBlock)
                        unitTestRmInfoBlock.fillGpuHalInfoBlock(pGpuHPI);
                }
            }

            return (void *)pGpuHPI;
            break;

        case DATA_ID_RC:

            if (!pRcHPI)
            {

                // allocate memory for rc hal info block
                createInfoBlock(&pRcHPI, unitAllEngineInfoBlkSize.rcInfoBlkSize);

                if (pRcHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillRcHalInfoBlock)
                        unitTestRmInfoBlock.fillRcHalInfoBlock(pRcHPI);
                }
            }

            return (void *)pRcHPI;
            break;

        case DATA_ID_VBIOS:

            if (!pVbiosHPI)
            {

                // allocate memory for vbios hal info block
                createInfoBlock(&pVbiosHPI, unitAllEngineInfoBlkSize.vbiosInfoBlkSize);

                if (pVbiosHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillVbiosHalInfoBlock)
                        unitTestRmInfoBlock.fillVbiosHalInfoBlock(pVbiosHPI);
                }
            }

            return (void *)pVbiosHPI;
            break;

        case DATA_ID_VGA:

            if (!pVgaHPI)
            {

                // allocate memory for vga hal info block
                createInfoBlock(&pVgaHPI, unitAllEngineInfoBlkSize.vgaInfoBlkSize);

                if (pVgaHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillVgaHalInfoBlock)
                        unitTestRmInfoBlock.fillVgaHalInfoBlock(pVgaHPI);
                }
            }

            return (void *)pVgaHPI;
            break;

        case DATA_ID_PPP:

            if (!pPppHPI)
            {

                // allocate memory for ppp hal info block
                createInfoBlock(&pPppHPI, unitAllEngineInfoBlkSize.pppInfoBlkSize);

                if (pPppHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillPppHalInfoBlock)
                        unitTestRmInfoBlock.fillPppHalInfoBlock(pPppHPI);
                }
            }

            return (void *)pPppHPI;
            break;

        case DATA_ID_SEQ:

            if (!pSeqHPI)
            {

                // allocate memory for seq hal info block
                createInfoBlock(&pSeqHPI, unitAllEngineInfoBlkSize.seqInfoBlkSize);

                if (pSeqHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillSeqHalInfoBlock)
                        unitTestRmInfoBlock.fillSeqHalInfoBlock(pSeqHPI);
                }
            }

            return (void *)pSeqHPI;
            break;

        case DATA_ID_TMR:

            if (!pTmrHPI)
            {

                // allocate memory for tmr hal info block
                createInfoBlock(&pTmrHPI, unitAllEngineInfoBlkSize.tmrInfoBlkSize);

                if (pTmrHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillTmrHalInfoBlock)
                        unitTestRmInfoBlock.fillTmrHalInfoBlock(pTmrHPI);
                }
            }

            return (void *)pTmrHPI;
            break;

        case DATA_ID_STEREO:

            if (!pStereoHPI)
            {

                // allocate memory for stereo hal info block
                createInfoBlock(&pStereoHPI, unitAllEngineInfoBlkSize.stereoInfoBlkSize);

                if (pStereoHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillStereoHalInfoBlock)
                        unitTestRmInfoBlock.fillStereoHalInfoBlock(pStereoHPI);
                }
            }

            return (void *)pStereoHPI;
            break;

        case DATA_ID_PERF:

            if (!pPerfHPI)
            {

                // allocate memory for perf hal info block
                createInfoBlock(&pPerfHPI, unitAllEngineInfoBlkSize.perfInfoBlkSize);

                if (pPerfHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillPerfHalInfoBlock)
                        unitTestRmInfoBlock.fillPerfHalInfoBlock(pPerfHPI);
                }
            }

            return (void *)pPerfHPI;
            break;

        case DATA_ID_MC:

            if (!pMcHPI)
            {

                // allocate memory for mc hal info block
                createInfoBlock(&pMcHPI, unitAllEngineInfoBlkSize.mcInfoBlkSize);

                if (pMcHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillMcHalInfoBlock)
                        unitTestRmInfoBlock.fillMcHalInfoBlock(pMcHPI);
                }
            }

            return (void *)pMcHPI;
            break;

        case DATA_ID_INTR:

            if (!pIntrHPI)
            {

                // allocate memory for intr hal info block
                createInfoBlock(&pIntrHPI, unitAllEngineInfoBlkSize.intrInfoBlkSize);

                if (pIntrHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillIntrHalInfoBlock)
                        unitTestRmInfoBlock.fillIntrHalInfoBlock(pIntrHPI);
                }
            }

            return (void *)pIntrHPI;
            break;

        case DATA_ID_INST:

            if (!pInstHPI)
            {

                // allocate memory for inst hal info block
                createInfoBlock(&pInstHPI, unitAllEngineInfoBlkSize.instInfoBlkSize);

                if (pInstHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillInstHalInfoBlock)
                        unitTestRmInfoBlock.fillInstHalInfoBlock(pInstHPI);
                }
            }

            return (void *)pInstHPI;
            break;

        case DATA_ID_HEAD:

            if (!pHeadHPI)
            {

                // allocate memory for head hal info block
                createInfoBlock(&pHeadHPI, unitAllEngineInfoBlkSize.headInfoBlkSize);

                if (pHeadHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillHeadHalInfoBlock)
                        unitTestRmInfoBlock.fillHeadHalInfoBlock(pHeadHPI);
                }
            }

            return (void *)pHeadHPI;
            break;

        case DATA_ID_GR:

            if (!pGrHPI)
            {

                // allocate memory for gr hal info block
                createInfoBlock(&pGrHPI, unitAllEngineInfoBlkSize.grInfoBlkSize);

                if (pGrHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillGrHalInfoBlock)
                        unitTestRmInfoBlock.fillGrHalInfoBlock(pGrHPI);
                }
            }

            return (void *)pGrHPI;
            break;

        case DATA_ID_FLCN:

            if (!pFlcnHPI)
            {

                // allocate memory for flcn hal info block
                createInfoBlock(&pFlcnHPI, unitAllEngineInfoBlkSize.flcnInfoBlkSize);

                if (pFlcnHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillFlcnHalInfoBlock)
                        unitTestRmInfoBlock.fillFlcnHalInfoBlock(pFlcnHPI);
                }
            }

            return (void *)pFlcnHPI;
            break;

        case DATA_ID_FIFO:

            if (!pFifoHPI)
            {

                // allocate memory for fifo hal info block
                createInfoBlock(&pFifoHPI, unitAllEngineInfoBlkSize.fifoInfoBlkSize);

                if (pFifoHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillFifoHalInfoBlock)
                        unitTestRmInfoBlock.fillFifoHalInfoBlock(pFifoHPI);
                }
            }

            return (void *)pFifoHPI;
            break;

        case DATA_ID_FBSR:

            if (!pFbsrHPI)
            {

                // allocate memory for fbsr hal info block
                createInfoBlock(&pFbsrHPI, unitAllEngineInfoBlkSize.fbsrInfoBlkSize);

                if (pFbsrHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillFbsrHalInfoBlock)
                        unitTestRmInfoBlock.fillFbsrHalInfoBlock(pFbsrHPI);
                }
            }

            return (void *)pFbsrHPI;
            break;

        case DATA_ID_FB:

            if (!pFbHPI)
            {

                // allocate memory for fb hal info block
                createInfoBlock(&pFbHPI, unitAllEngineInfoBlkSize.fbInfoBlkSize);

                if (pFbHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillFbHalInfoBlock)
                        unitTestRmInfoBlock.fillFbHalInfoBlock(pFbHPI);
                }
            }

            return (void *)pFbHPI;
            break;

        case DATA_ID_FBFLCN:

            if (!pFbflcnHPI)
            {

                // allocate memory for FB falcon hal info block
                createInfoBlock(&pFbflcnHPI, unitAllEngineInfoBlkSize.fbflcnInfoBlkSize);

                if (pFbflcnHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillFbflcnHalInfoBlock)
                        unitTestRmInfoBlock.fillFbflcnHalInfoBlock(pFbflcnHPI);
                }
            }

            return (void *)pFbflcnHPI;
            break;

        case DATA_ID_DPLINK:

            if (!pDplinkHPI)
            {

                // allocate memory for dplink hal info block
                createInfoBlock(&pDplinkHPI, unitAllEngineInfoBlkSize.dplinkInfoBlkSize);

                if (pDplinkHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillDplinkHalInfoBlock)
                        unitTestRmInfoBlock.fillDplinkHalInfoBlock(pDplinkHPI);
                }
            }

            return (void *)pDplinkHPI;
            break;

        case DATA_ID_DPAUX:

            if (!pDpauxHPI)
            {

                // allocate memory for dpaux hal info block
                createInfoBlock(&pDpauxHPI, unitAllEngineInfoBlkSize.dpauxInfoBlkSize);

                if (pDpauxHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillDpauxHalInfoBlock)
                        unitTestRmInfoBlock.fillDpauxHalInfoBlock(pDpauxHPI);
                }
            }

            return (void *)pDpauxHPI;
            break;

        case DATA_ID_DMA:

            if (!pDmaHPI)
            {

                // allocate memory for dma hal info block
                createInfoBlock(&pDmaHPI, unitAllEngineInfoBlkSize.dmaInfoBlkSize);

                if (pDmaHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillDmaHalInfoBlock)
                        unitTestRmInfoBlock.fillDmaHalInfoBlock(pDmaHPI);
                }
            }

            return (void *)pDmaHPI;
            break;

        case DATA_ID_DISP:

            if (!pDispHPI)
            {

                // allocate memory for disp hal info block
                createInfoBlock(&pDispHPI, unitAllEngineInfoBlkSize.dispInfoBlkSize);

                if (pDispHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillDispHalInfoBlock)
                        unitTestRmInfoBlock.fillDispHalInfoBlock(pDispHPI);
                }
            }

            return (void *)pDispHPI;
            break;

        case DATA_ID_DAC:

            if (!pDacHPI)
            {

                // allocate memory for dac hal info block
                createInfoBlock(&pDacHPI, unitAllEngineInfoBlkSize.dacInfoBlkSize);

                if (pDacHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillDacHalInfoBlock)
                        unitTestRmInfoBlock.fillDacHalInfoBlock(pDacHPI);
                }
            }

            return (void *)pDacHPI;
            break;

        case DATA_ID_CLK:

            if (!pClkHPI)
            {

                // allocate memory for clk hal info block
                createInfoBlock(&pClkHPI, unitAllEngineInfoBlkSize.clkInfoBlkSize);

                if (pClkHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillClkHalInfoBlock)
                        unitTestRmInfoBlock.fillClkHalInfoBlock(pClkHPI);
                }
            }
            return (void *)pClkHPI;
            break;

        case DATA_ID_BUS:

            if (!pBusHPI)
            {

                // allocate memory for bus hal info block
                createInfoBlock(&pBusHPI, unitAllEngineInfoBlkSize.busInfoBlkSize);

                if (pBusHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillBusHalInfoBlock)
                        unitTestRmInfoBlock.fillBusHalInfoBlock(pBusHPI);
                }
            }

            return (void *)pBusHPI;
            break;

        case DATA_ID_BIF:

            if (!pBifHPI)
            {

                // allocate memory for bif hal info block
                createInfoBlock(&pBifHPI, unitAllEngineInfoBlkSize.bifInfoBlkSize);

                if (pBifHPI)
                {

                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillBifHalInfoBlock)
                        unitTestRmInfoBlock.fillBifHalInfoBlock(pBifHPI);
                }
            }

            return (void *)pBifHPI;
            break;

        case DATA_ID_LWJPG:

            if (!pLwjpgHPI)
            {
                // allocate memory for lwjpg hal info block
                createInfoBlock(&pLwjpgHPI, unitAllEngineInfoBlkSize.lwjpgInfoBlkSize);

                if (pLwjpgHPI)
                {
                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillLwjpgHalInfoBlock)
                        unitTestRmInfoBlock.fillLwjpgHalInfoBlock(pLwjpgHPI);
                }
            }

            return (void *)pLwjpgHPI;
            break;

        case DATA_ID_OFA:

            if (!pOfaHPI)
            {
                // allocate memory for ofa hal info block
                createInfoBlock(&pOfaHPI, unitAllEngineInfoBlkSize.ofaInfoBlkSize);

                if (pOfaHPI)
                {
                    // fill the infoblock, if user has specified
                    if (unitTestRmInfoBlock.fillOfaHalInfoBlock)
                        unitTestRmInfoBlock.fillOfaHalInfoBlock(pOfaHPI);
                }
            }

            return (void *)pOfaHPI;
            break;

        default:
                return NULL;
        }
    }
    else
    {
        if (dataId == DATA_ID_FREE_OBJ)
        {
            pDpuHPI = NULL;
            pResHPI = NULL;
            pPgengHPI = NULL;
            pPgctrlHPI = NULL;
            pPgHPI = NULL;
            pInforomHPI = NULL;
            pMsencHPI = NULL;
            pVicHPI = NULL;
            pSpbHPI = NULL;
            pPmuHPI = NULL;
            pCeHPI = NULL;
            pIsohubHPI = NULL;
            pCveHPI = NULL;
            pCipherHPI = NULL;
            pHdmiHPI = NULL;
            pHdcpHPI = NULL;
            pHdtvHPI = NULL;
            pVpHPI = NULL;
            pVideoHPI = NULL;
            pMpHPI = NULL;
            pMpegHPI = NULL;
            pBspHPI = NULL;
            pSmuHPI = NULL;
            pSorHPI = NULL;
            pPiorHPI = NULL;
            pOrHPI = NULL;
            pThermHPI = NULL;
            pVoltHPI = NULL;
            pFuseHPI = NULL;
            pFanHPI = NULL;
            pGpioHPI = NULL;
            pI2cHPI = NULL;
            pGpuHPI = NULL;
            pSwHPI = NULL;
            pRcHPI = NULL;
            pVbiosHPI = NULL;
            pVgaHPI = NULL;
            pPppHPI = NULL;
            pSeqHPI = NULL;
            pTmrHPI = NULL;
            pStereoHPI = NULL;
            pPerfHPI = NULL;
            pMcHPI = NULL;
            pIntrHPI = NULL;
            pInstHPI = NULL;
            pHeadHPI = NULL;
            pGrHPI = NULL;
            pFlcnHPI = NULL;
            pFifoHPI = NULL;
            pFbsrHPI = NULL;
            pFbHPI = NULL;
            pDplinkHPI = NULL;
            pDpauxHPI = NULL;
            pDmaHPI = NULL;
            pDispHPI = NULL;
            pDacHPI = NULL;
            pClkHPI = NULL;
            pBusHPI = NULL;
            pBifHPI = NULL;
            pLwjpgHPI = NULL;
            pOfaHPI = NULL;

            return NULL;
        }

    }
    return NULL;
}

/*!
 * @brief create DAC object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pDac    pointer to DAC object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createDac(POBJGPU pGpu, POBJDAC pDac, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pDac->objOr.odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->dacHalIfacesSetupFn(pGpu, &pDac->hal);

    // Initialize non-hal ptrs
    //dacObjIfacesSetup(pGpu, pDac);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillDacObject)
        unitTestRmObject.fillDacObject(pDac);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pDac->dacGetInfoBlock = getInfloblockStub;
    pDac->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pDac->infoList->dataId = DATA_ID_DAC;
}

/*!
 * @brief create DPU object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pDpu    pointer to DPU object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createDpu(POBJGPU pGpu, POBJDPU pDpu, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pDpu->odbCommon);

    // Initialize property range
    pDpu->initProperties(pDpu, PDB_PROP_DPU_BEGIN, PDB_PROP_DPU_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->dpuHalIfacesSetupFn(pGpu, &pDpu->hal);

    // Initialize non-hal ptrs
    dpuObjIfacesSetup(pGpu, pDpu);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pDpu->hal.dpuSetPropertiesList(pGpu, pDpu);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillDpuObject)
        unitTestRmObject.fillDpuObject(pDpu);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pDpu->odbCommon.odbGetObject = getObjectMock;

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pDpu->dpuGetInfoBlock = getInfloblockStub;
    pDpu->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pDpu->infoList->dataId = DATA_ID_DPU;
}

/*!
 * @brief create RES object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pRes    pointer to RES object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createRes(POBJGPU pGpu, POBJRES pRes, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pRes->odbCommon);

    // Initialize non-hal ptrs
    resObjIfacesSetup(pRes);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillResObject)
        unitTestRmObject.fillResObject(pRes);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pRes->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create PGENG object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pPgeng    pointer to PGENG object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createPgeng(POBJGPU pGpu, POBJPGENG pPgeng, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pPgeng->odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->pgengHalIfacesSetupFn(pGpu, &pPgeng->hal);

    // Initialize non-hal ptrs
    //pgengObjIfacesSetup(pGpu, pPgeng);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPgengObject)
        unitTestRmObject.fillPgengObject(pPgeng);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pPgeng->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create PGCTRL object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pPgctrl    pointer to PGCTRL object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createPgctrl(POBJGPU pGpu, POBJPGCTRL pPgctrl, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pPgctrl->odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->pgctrlHalIfacesSetupFn(pGpu, &pPgctrl->hal);

    // Initialize non-hal ptrs
    //pgctrlObjIfacesSetup(pGpu, pPgctrl);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPgctrlObject)
        unitTestRmObject.fillPgctrlObject(pPgctrl);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pPgctrl->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create LPWR object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pLpwr   pointer to LPWR object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createLpwr(POBJGPU pGpu, POBJLPWR pLpwr, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pLpwr->odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->pgHalIfacesSetupFn(pGpu, &pLpwr->hal);

    // Initialize non-hal ptrs
    pgObjIfacesSetup(pGpu, pLpwr);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPgObject)
        unitTestRmObject.fillPgObject(pLpwr);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pLpwr->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create INFOROM object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pInforom    pointer to INFOROM object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createInforom(POBJGPU pGpu, POBJINFOROM pInforom, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pInforom->odbCommon);

    // Initialize property range
    pInforom->initProperties(pInforom, PDB_PROP_INFOROM_BEGIN, PDB_PROP_INFOROM_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->inforomHalIfacesSetupFn(pGpu, &pInforom->hal);

    // Initialize non-hal ptrs
    inforomObjIfacesSetup(pGpu, pInforom);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pInforom->hal.inforomSetPropertiesList(pGpu, pInforom);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillInforomObject)
        unitTestRmObject.fillInforomObject(pInforom);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pInforom->odbCommon.odbGetObject = getObjectMock;

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pInforom->inforomGetInfoBlock = getInfloblockStub;
    pInforom->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pInforom->infoList->dataId = DATA_ID_INFOROM;
}

/*!
 * @brief create MSENC object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pMsenc    pointer to MSENC object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createMsenc(POBJGPU pGpu, POBJMSENC pMsenc, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pMsenc->odbCommon);

    // Initialize property range
    pMsenc->initProperties(pMsenc, PDB_PROP_MSENC_BEGIN, PDB_PROP_MSENC_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->msencHalIfacesSetupFn(pGpu, &pMsenc->hal);

    // Initialize non-hal ptrs
    msencObjIfacesSetup(pGpu, pMsenc);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pMsenc->hal.msencSetPropertiesList(pGpu, pMsenc);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillMsencObject)
        unitTestRmObject.fillMsencObject(pMsenc);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pMsenc->odbCommon.odbGetObject = getObjectMock;

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pMsenc->msencGetInfoBlock = getInfloblockStub;
    pMsenc->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pMsenc->infoList->dataId = DATA_ID_MSENC;
}

/*!
 * @brief create VIC object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pVic    pointer to VIC object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createVic(POBJGPU pGpu, POBJVIC pVic, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pVic->odbCommon);

    // Initialize property range
    pVic->initProperties(pVic, PDB_PROP_VIC_BEGIN, PDB_PROP_VIC_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->vicHalIfacesSetupFn(pGpu, &pVic->hal);

    // Initialize non-hal ptrs
    vicObjIfacesSetup(pGpu, pVic);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pVic->hal.vicSetPropertiesList(pGpu, pVic);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillVicObject)
        unitTestRmObject.fillVicObject(pVic);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pVic->odbCommon.odbGetObject = getObjectMock;

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pVic->vicGetInfoBlock = getInfloblockStub;
    pVic->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pVic->infoList->dataId = DATA_ID_VIC;
}

/*!
 * @brief create SPB object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pSpb    pointer to SPB object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createSpb(POBJGPU pGpu, POBJSPB pSpb, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pSpb->odbCommon);

    // Initialize property range
    pSpb->initProperties(pSpb, PDB_PROP_SPB_BEGIN, PDB_PROP_SPB_END);

    // Initialize non-hal ptrs
    spbObjIfacesSetup(pSpb);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillSpbObject)
        unitTestRmObject.fillSpbObject(pSpb);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pSpb->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create PMU object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pPmu    pointer to PMU object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createPmu(POBJGPU pGpu, POBJPMU pPmu, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pPmu->odbCommon);

    // Initialize property range
    pPmu->initProperties(pPmu, PDB_PROP_PMU_BEGIN, PDB_PROP_PMU_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->pmuHalIfacesSetupFn(pGpu, &pPmu->hal);

    // Initialize non-hal ptrs
    pmuObjIfacesSetup(pGpu, pPmu);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pPmu->hal.pmuSetPropertiesList(pGpu, pPmu);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPmuObject)
        unitTestRmObject.fillPmuObject(pPmu);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pPmu->odbCommon.odbGetObject = getObjectMock;

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pPmu->pmuGetInfoBlock = getInfloblockStub;
    pPmu->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pPmu->infoList->dataId = DATA_ID_PMU;
}

/*!
 * @brief create Ce object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pCe    pointer to Ce object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createCE(POBJGPU pGpu, POBJCE pCe, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pCe->odbCommon);

    // Initialize property range
    pCe->initProperties(pCe, PDB_PROP_CE_BEGIN, PDB_PROP_CE_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->ceHalIfacesSetupFn(pGpu, &pCe->hal);

    // Initialize non-hal ptrs
    //CeObjIfacesSetup(pGpu, pCe);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pCe->hal.ceSetPropertiesList(pGpu, pCe);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillCeObject)
        unitTestRmObject.fillCeObject(pCe);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pCe->odbCommon.odbGetObject = getObjectMock;

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pCe->ceGetInfoBlock = getInfloblockStub;
    pCe->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pCe->infoList->dataId = DATA_ID_CE;
}

/*!
 * @brief create ISOHUB object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pIsohub    pointer to ISOHUB object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createIsohub(POBJGPU pGpu, POBJISOHUB pIsohub, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pIsohub->odbCommon);

    // Initialize property range
    pIsohub->initProperties(pIsohub, PDB_PROP_ISOHUB_BEGIN, PDB_PROP_ISOHUB_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->isohubHalIfacesSetupFn(pGpu, &pIsohub->hal);

    // Initialize non-hal ptrs
    //isohubObjIfacesSetup(pGpu, pIsohub);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pIsohub->hal.isohubSetPropertiesList(pGpu, pIsohub);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillIsohubObject)
        unitTestRmObject.fillIsohubObject(pIsohub);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pIsohub->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create CVE object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pCve    pointer to CVE object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createCve(POBJGPU pGpu, POBJCVE pCve, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pCve->odbCommon);

    // Initialize property range
    pCve->initProperties(pCve, PDB_PROP_CVE_BEGIN, PDB_PROP_CVE_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->cveHalIfacesSetupFn(pGpu, &pCve->hal);

    // Initialize non-hal ptrs
    cveObjIfacesSetup(pGpu, pCve);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pCve->hal.cveSetPropertiesList(pGpu, pCve);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillCveObject)
        unitTestRmObject.fillCveObject(pCve);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pCve->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create CIPHER object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pCipher    pointer to CIPHER object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createCipher(POBJGPU pGpu, POBJCIPHER pCipher, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pCipher->odbCommon);

    // Initialize property range
    pCipher->initProperties(pCipher, PDB_PROP_CIPHER_BEGIN, PDB_PROP_CIPHER_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->cipherHalIfacesSetupFn(pGpu, &pCipher->hal);

    // Initialize non-hal ptrs
    cipherObjIfacesSetup(pGpu, pCipher);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pCipher->hal.cipherSetPropertiesList(pGpu, pCipher);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillCipherObject)
        unitTestRmObject.fillCipherObject(pCipher);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pCipher->cipherGetInfoBlock = getInfloblockStub;
    pCipher->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pCipher->infoList->dataId = DATA_ID_CIPHER;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pCipher->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create HDMI object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pHdmi    pointer to HDMI object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createHdmi(POBJGPU pGpu, POBJHDMI pHdmi, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pHdmi->odbCommon);

    // Initialize property range
    pHdmi->initProperties(pHdmi, PDB_PROP_HDMI_BEGIN, PDB_PROP_HDMI_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->hdmiHalIfacesSetupFn(pGpu, &pHdmi->hal);

    // Initialize non-hal ptrs
    hdmiObjIfacesSetup(pGpu, pHdmi);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pHdmi->hal.hdmiSetPropertiesList(pGpu, pHdmi);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillHdmiObject)
        unitTestRmObject.fillHdmiObject(pHdmi);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pHdmi->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create HDCP object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pHdcp    pointer to HDCP object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createHdcp(POBJGPU pGpu, POBJHDCP pHdcp, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pHdcp->odbCommon);

    // Initialize property range
    pHdcp->initProperties(pHdcp, PDB_PROP_HDCP_BEGIN, PDB_PROP_HDCP_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->hdcpHalIfacesSetupFn(pGpu, &pHdcp->hal);

    // Initialize non-hal ptrs
    hdcpObjIfacesSetup(pGpu, pHdcp);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pHdcp->hal.hdcpSetPropertiesList(pGpu, pHdcp);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillHdcpObject)
        unitTestRmObject.fillHdcpObject(pHdcp);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pHdcp->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create HDTV object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pHdtv    pointer to HDTV object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createHdtv(POBJGPU pGpu, POBJHDTV pHdtv, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pHdtv->odbCommon);

    // Initialize property range
    pHdtv->initProperties(pHdtv, PDB_PROP_HDTV_BEGIN, PDB_PROP_HDTV_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->hdtvHalIfacesSetupFn(pGpu, &pHdtv->hal);

    // Initialize non-hal ptrs
    //hdtvObjIfacesSetup(pGpu, pHdtv);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pHdtv->hal.hdtvSetPropertiesList(pGpu, pHdtv);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillHdtvObject)
        unitTestRmObject.fillHdtvObject(pHdtv);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pHdtv->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create VP object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pVp    pointer to VP object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createVp(POBJGPU pGpu, POBJVP pVp, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pVp->odbCommon);

    // Initialize property range
    pVp->initProperties(pVp, PDB_PROP_VP_BEGIN, PDB_PROP_VP_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->vpHalIfacesSetupFn(pGpu, &pVp->hal);

    // Initialize non-hal ptrs
    vpObjIfacesSetup(pGpu, pVp);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pVp->hal.vpSetPropertiesList(pGpu, pVp);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillVpObject)
        unitTestRmObject.fillVpObject(pVp);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pVp->vpGetInfoBlock = getInfloblockStub;
    pVp->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pVp->infoList->dataId = DATA_ID_VP;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pVp->odbCommon.odbGetObject = getObjectMock;

}

/*!
 * @brief create MP object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pMp    pointer to MP object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createMp(POBJGPU pGpu, POBJMP pMp, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pMp->odbCommon);

    // Initialize property range
    pMp->initProperties(pMp, PDB_PROP_MEDIAPORT_BEGIN, PDB_PROP_MEDIAPORT_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->mpHalIfacesSetupFn(pGpu, &pMp->hal);

    // Initialize non-hal ptrs
    mpObjIfacesSetup(pGpu, pMp);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pMp->hal.mpSetPropertiesList(pGpu, pMp);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillMpObject)
        unitTestRmObject.fillMpObject(pMp);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pMp->mpGetInfoBlock = getInfloblockStub;
    pMp->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pMp->infoList->dataId = DATA_ID_MP;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pMp->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create MPEG object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pMpeg    pointer to MPEG object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createMpeg(POBJGPU pGpu, POBJMPEG pMpeg, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pMpeg->odbCommon);

    // Initialize property range
    pMpeg->initProperties(pMpeg, PDB_PROP_MPEG_BEGIN, PDB_PROP_MPEG_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->mpegHalIfacesSetupFn(pGpu, &pMpeg->hal);

    // Initialize non-hal ptrs
    //mpegObjIfacesSetup(pGpu, pMpeg);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pMpeg->hal.mpegSetPropertiesList(pGpu, pMpeg);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillMpegObject)
        unitTestRmObject.fillMpegObject(pMpeg);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pMpeg->mpegGetInfoBlock = getInfloblockStub;
    pMpeg->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pMpeg->infoList->dataId = DATA_ID_MPEG;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pMpeg->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create BSP object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pBsp    pointer to BSP object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createBsp(POBJGPU pGpu, POBJBSP pBsp, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pBsp->odbCommon);

    // Initialize property range
    pBsp->initProperties(pBsp, PDB_PROP_BSP_BEGIN, PDB_PROP_BSP_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->bspHalIfacesSetupFn(pGpu, &pBsp->hal);

    // Initialize non-hal ptrs
    bspObjIfacesSetup(pGpu, pBsp);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pBsp->hal.bspSetPropertiesList(pGpu, pBsp);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillBspObject)
        unitTestRmObject.fillBspObject(pBsp);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pBsp->bspGetInfoBlock = getInfloblockStub;
    pBsp->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pBsp->infoList->dataId = DATA_ID_BSP;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pBsp->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create SMU object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pSmu    pointer to SMU object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createSmu(POBJGPU pGpu, POBJSMU pSmu, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pSmu->odbCommon);

    // Initialize property range
    pSmu->initProperties(pSmu, PDB_PROP_SMU_BEGIN, PDB_PROP_SMU_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->smuHalIfacesSetupFn(pGpu, &pSmu->hal);

    // Initialize non-hal ptrs
    //smuObjIfacesSetup(pGpu, pSmu);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pSmu->hal.smuSetPropertiesList(pGpu, pSmu);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillSmuObject)
        unitTestRmObject.fillSmuObject(pSmu);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pSmu->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create SOR object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pSor    pointer to SOR object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createSor(POBJGPU pGpu, POBJSOR pSor, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pSor->objOr.odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->sorHalIfacesSetupFn(pGpu, &pSor->hal);

    // Initialize non-hal ptrs
    sorObjIfacesSetup(pGpu, pSor);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillSorObject)
        unitTestRmObject.fillSorObject(pSor);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pSor->objOr.odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create PIOR object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pPior    pointer to PIOR object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createPior(POBJGPU pGpu, POBJPIOR pPior, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pPior->objOr.odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->piorHalIfacesSetupFn(pGpu, &pPior->hal);

    // Initialize non-hal ptrs
    piorObjIfacesSetup(pGpu, pPior);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPiorObject)
        unitTestRmObject.fillPiorObject(pPior);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pPior->objOr.odbCommon.odbGetObject = getObjectMock;

}

/*!
 * @brief create THERM object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pTherm    pointer to THERM object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createTherm(POBJGPU pGpu, POBJTHERM pTherm, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pTherm->odbCommon);

    // Initialize property range
    pTherm->initProperties(pTherm, PDB_PROP_THERM_BEGIN, PDB_PROP_THERM_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->thermHalIfacesSetupFn(pGpu, &pTherm->hal);

    // Initialize non-hal ptrs
    thermObjIfacesSetup(pGpu, pTherm);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pTherm->hal.thermSetPropertiesList(pGpu, pTherm);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillThermObject)
        unitTestRmObject.fillThermObject(pTherm);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pTherm->thermGetInfoBlock = getInfloblockStub;
    pTherm->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pTherm->infoList->dataId = DATA_ID_THERM;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pTherm->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create VOLT object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pVolt    pointer to VOLT object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createVolt(POBJGPU pGpu, POBJVOLT pVolt, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pVolt->odbCommon);

    // Initialize property range
    pVolt->initProperties(pVolt, PDB_PROP_VOLT_BEGIN, PDB_PROP_VOLT_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->voltHalIfacesSetupFn(pGpu, &pVolt->hal);

    // Initialize non-hal ptrs
    voltObjIfacesSetup(pGpu, pVolt);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pVolt->hal.voltSetPropertiesList(pGpu, pVolt);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillVoltObject)
        unitTestRmObject.fillVoltObject(pVolt);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pVolt->voltGetInfoBlock = getInfloblockStub;
    pVolt->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pVolt->infoList->dataId = DATA_ID_VOLT;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pVolt->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create FUSE object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pFuse    pointer to FUSE object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createFuse(POBJGPU pGpu, POBJFUSE pFuse, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pFuse->odbCommon);

    // Initialize property range
    pFuse->initProperties(pFuse, PDB_PROP_FUSE_BEGIN, PDB_PROP_FUSE_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->fuseHalIfacesSetupFn(pGpu, &pFuse->hal);

    // Initialize non-hal ptrs
    //fuseObjIfacesSetup(pGpu, pFuse);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pFuse->hal.fuseSetPropertiesList(pGpu, pFuse);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillFuseObject)
        unitTestRmObject.fillFuseObject(pFuse);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pFuse->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create FAN object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pFan    pointer to FAN object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createFan(POBJGPU pGpu, POBJFAN pFan, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pFan->odbCommon);

    // Initialize property range
    pFan->initProperties(pFan, PDB_PROP_FAN_BEGIN, PDB_PROP_FAN_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->fanHalIfacesSetupFn(pGpu, &pFan->hal);

    // Initialize non-hal ptrs
    //fanObjIfacesSetup(pGpu, pFan);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pFan->hal.fanSetPropertiesList(pGpu, pFan);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillFanObject)
        unitTestRmObject.fillFanObject(pFan);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pFan->fanGetInfoBlock = getInfloblockStub;
    pFan->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pFan->infoList->dataId = DATA_ID_FAN;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pFan->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create GPIO object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pGpio    pointer to GPIO object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createGpio(POBJGPU pGpu, POBJGPIO pGpio, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pGpio->odbCommon);

    // Initialize property range
    pGpio->initProperties(pGpio, PDB_PROP_GPIO_BEGIN, PDB_PROP_GPIO_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->gpioHalIfacesSetupFn(pGpu, &pGpio->hal[0].iHal);

    // Initialize non-hal ptrs
    //gpioObjIfacesSetup(pGpu, pGpio);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pGpio->hal[0].iHal.gpioSetPropertiesList(pGpu, pGpio);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillGpioObject)
        unitTestRmObject.fillGpioObject(pGpio);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pGpio->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create I2C object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pI2c    pointer to I2C object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createI2c(POBJGPU pGpu, OBJI2C *pI2c, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pI2c->odbCommon);

    // Initialize property range
    pI2c->initProperties(pI2c, PDB_PROP_I2C_BEGIN, PDB_PROP_I2C_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->i2cHalIfacesSetupFn(pGpu, &pI2c->hal);

    // Initialize non-hal ptrs
    //i2cObjIfacesSetup(pGpu, pI2c);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pI2c->hal.i2cSetPropertiesList(pGpu, pI2c);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillI2cObject)
        unitTestRmObject.fillI2cObject(pI2c);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pI2c->i2cGetInfoBlock = getInfloblockStub;
    pI2c->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pI2c->infoList->dataId = DATA_ID_I2C;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pI2c->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create RC object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pRc    pointer to RC object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createRc(POBJGPU pGpu, POBJRC pRc, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pRc->odbCommon);

    // Initialize non-hal ptrs
    rcObjIfacesSetup(pGpu, pRc);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillRcObject)
        unitTestRmObject.fillRcObject(pRc);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pRc->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create VBIOS object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pVbios    pointer to VBIOS object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createVbios(POBJGPU pGpu, POBJVBIOS pVbios, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pVbios->odbCommon);

    // Initialize property range
    pVbios->initProperties(pVbios, PDB_PROP_VBIOS_BEGIN, PDB_PROP_VBIOS_END);

    // Initialize non-hal ptrs
    vbiosObjIfacesSetup(pGpu, pVbios);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillVbiosObject)
        unitTestRmObject.fillVbiosObject(pVbios);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pVbios->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create VGA object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pVga    pointer to VGA object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createVga(POBJGPU pGpu, POBJVGA pVga, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pVga->odbCommon);

    // Initialize property range
    pVga->initProperties(pVga, PDB_PROP_VGA_BEGIN, PDB_PROP_VGA_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->vgaHalIfacesSetupFn(pGpu, &pVga->hal);

    // Initialize non-hal ptrs
    //vgaObjIfacesSetup(pGpu, pVga);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pVga->hal.vgaSetPropertiesList(pGpu, pVga);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillVgaObject)
        unitTestRmObject.fillVgaObject(pVga);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pVga->odbCommon.odbGetObject = getObjectMock;

}

/*!
 * @brief create PPP object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pMsppp  pointer to PPP object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createPpp(POBJGPU pGpu, POBJMSPPP pMsppp, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pMsppp->odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->pppHalIfacesSetupFn(pGpu, &pMsppp->hal);

    // Initialize non-hal ptrs
    pppObjIfacesSetup(pGpu, pMsppp);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPppObject)
        unitTestRmObject.fillPppObject(pMsppp);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pMsppp->mspppGetInfoBlock = getInfloblockStub;
    pMsppp->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pMsppp->infoList->dataId = DATA_ID_PPP;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pMsppp->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create SEQ object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pSeq    pointer to SEQ object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createSeq(POBJGPU pGpu, POBJSEQ pSeq, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pSeq->odbCommon);

    // Initialize property range
    pSeq->initProperties(pSeq, PDB_PROP_SEQ_BEGIN, PDB_PROP_SEQ_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->seqHalIfacesSetupFn(pGpu, &pSeq->hal);

    // Initialize non-hal ptrs
    seqObjIfacesSetup(pGpu, pSeq);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pSeq->hal.seqSetPropertiesList(pGpu, pSeq);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillSeqObject)
        unitTestRmObject.fillSeqObject(pSeq);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pSeq->seqGetInfoBlock = getInfloblockStub;
    pSeq->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pSeq->infoList->dataId = DATA_ID_SEQ;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pSeq->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create TMR object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pTmr    pointer to TMR object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createTmr(POBJGPU pGpu, POBJTMR pTmr, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pTmr->odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->tmrHalIfacesSetupFn(pGpu, &pTmr->hal);

    // Initialize non-hal ptrs
    tmrObjIfacesSetup(pGpu, pTmr);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillTmrObject)
        unitTestRmObject.fillTmrObject(pTmr);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pTmr->tmrGetInfoBlock = getInfloblockStub;
    pTmr->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pTmr->infoList->dataId = DATA_ID_TMR;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pTmr->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create STEREO object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pStereo    pointer to STEREO object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createStereo(POBJGPU pGpu, POBJSTEREO pStereo, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pStereo->odbCommon);

    // Initialize property range
    pStereo->initProperties(pStereo, PDB_PROP_STEREO_BEGIN, PDB_PROP_STEREO_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->stereoHalIfacesSetupFn(pGpu, &pStereo->hal);

    // Initialize non-hal ptrs
    stereoObjIfacesSetup(pGpu, pStereo);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pStereo->hal.stereoSetPropertiesList(pGpu, pStereo);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillStereoObject)
        unitTestRmObject.fillStereoObject(pStereo);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pStereo->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create PERF object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pPerf    pointer to PERF object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createPerf(POBJGPU pGpu, POBJPERF pPerf, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pPerf->odbCommon);

    // Initialize property range
    pPerf->initProperties(pPerf, PDB_PROP_PERF_BEGIN, PDB_PROP_PERF_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->perfHalIfacesSetupFn(pGpu, &pPerf->hal);

    // Initialize non-hal ptrs
    perfObjIfacesSetup(pGpu, pPerf);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pPerf->hal.perfSetPropertiesList(pGpu, pPerf);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPerfObject)
        unitTestRmObject.fillPerfObject(pPerf);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pPerf->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create MC object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pMc    pointer to MC object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createMc(POBJGPU pGpu, POBJMC pMc, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pMc->odbCommon);

    // Initialize property range
    pMc->initProperties(pMc, PDB_PROP_MC_BEGIN, PDB_PROP_MC_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->mcHalIfacesSetupFn(pGpu, &pMc->hal);

    // Initialize non-hal ptrs
    mcObjIfacesSetup(pGpu, pMc);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pMc->hal.mcSetPropertiesList(pGpu, pMc);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillMcObject)
        unitTestRmObject.fillMcObject(pMc);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pMc->mcGetInfoBlock = getInfloblockStub;
    pMc->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pMc->infoList->dataId = DATA_ID_MC;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pMc->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create INTR object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pIntr    pointer to INTR object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createIntr(POBJGPU pGpu, POBJINTR pIntr, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pIntr->odbCommon);

    // Initialize property range
    pIntr->initProperties(pIntr, PDB_PROP_INTR_BEGIN, PDB_PROP_INTR_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->intrHalIfacesSetupFn(pGpu, &pIntr->hal);

    // Initialize non-hal ptrs
    intrObjIfacesSetup(pGpu, pIntr);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pIntr->hal.intrSetPropertiesList(pGpu, pIntr);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillIntrObject)
        unitTestRmObject.fillIntrObject(pIntr);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pIntr->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create INST object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pInst    pointer to INST object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createInst(POBJGPU pGpu, POBJINST pInst, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pInst->odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->instHalIfacesSetupFn(pGpu, &pInst->hal);

    // Initialize non-hal ptrs
    instObjIfacesSetup(pGpu, pInst);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillInstObject)
        unitTestRmObject.fillInstObject(pInst);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pInst->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create HEAD object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pHead    pointer to HEAD object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createHead(POBJGPU pGpu, POBJHEAD pHead, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pHead->odbCommon);

    // Initialize property range
    pHead->initProperties(pHead, PDB_PROP_HEAD_BEGIN, PDB_PROP_HEAD_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->headHalIfacesSetupFn(pGpu, &pHead->hal);

    // Initialize non-hal ptrs
    //headObjIfacesSetup(pGpu, pHead);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pHead->hal.headSetPropertiesList(pGpu, pHead);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillHeadObject)
        unitTestRmObject.fillHeadObject(pHead);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pHead->headGetInfoBlock = getInfloblockStub;
    pHead->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pHead->infoList->dataId = DATA_ID_HEAD;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pHead->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create GR object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pGr    pointer to GR object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createGr(POBJGPU pGpu, POBJGR pGr, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pGr->odbCommon);

    // Initialize property range
    pGr->initProperties(pGr, PDB_PROP_GRAPHICS_BEGIN, PDB_PROP_GRAPHICS_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->grHalIfacesSetupFn(pGpu, &pGr->hal);

    // Initialize non-hal ptrs
    grObjIfacesSetup(pGpu, pGr);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pGr->hal.grSetPropertiesList(pGpu, pGr);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillGrObject)
        unitTestRmObject.fillGrObject(pGr);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pGr->grGetInfoBlock = getInfloblockStub;
    pGr->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pGr->infoList->dataId = DATA_ID_GR;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pGr->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create FIFO object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pFifo    pointer to FIFO object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createFifo(POBJGPU pGpu, POBJFIFO pFifo, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pFifo->odbCommon);

    // Initialize property range
    pFifo->initProperties(pFifo, PDB_PROP_FIFO_BEGIN, PDB_PROP_FIFO_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->fifoHalIfacesSetupFn(pGpu, &pFifo->hal);

    // Initialize non-hal ptrs
    fifoObjIfacesSetup(pGpu, pFifo);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pFifo->hal.fifoSetPropertiesList(pGpu, pFifo);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillFifoObject)
        unitTestRmObject.fillFifoObject(pFifo);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pFifo->fifoGetInfoBlock = getInfloblockStub;
    pFifo->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pFifo->infoList->dataId = DATA_ID_FIFO;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pFifo->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create FBSR object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pFbsr    pointer to FBSR object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createFbsr(POBJGPU pGpu, POBJFBSR pFbsr, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pFbsr->odbCommon);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->fbsrHalIfacesSetupFn(pGpu, &pFbsr->hal);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillFbsrObject)
        unitTestRmObject.fillFbsrObject(pFbsr);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pFbsr->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create FB object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pFb    pointer to FB object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createFb(POBJGPU pGpu, POBJFB pFb, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pFb->odbCommon);

    // Initialize property range
    pFb->initProperties(pFb, PDB_PROP_FB_BEGIN, PDB_PROP_FB_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->fbHalIfacesSetupFn(pGpu, &pFb->hal);

    // Initialize non-hal ptrs
    fbObjIfacesSetup(pGpu, pFb);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pFb->hal.fbSetPropertiesList(pGpu, pFb);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillFbObject)
        unitTestRmObject.fillFbObject(pFb);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pFb->fbGetInfoBlock = getInfloblockStub;
    pFb->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pFb->infoList->dataId = DATA_ID_FB;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pFb->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create FB FALCON object as specified by the user
 *
 * @param[in]      pGpu         pointer to GPU object
 *
 * @param[in]      pFbflcn      pointer to FB FALCON object
 *
 * @param[in]      pMod         pointer to module descriptor, to init
                                HAL ptrs
 *
 */
void createFbflcn(POBJGPU pGpu, POBJFBFLCN pFbflcn, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pFbflcn->odbCommon);

    // Initialize property range
    pFbflcn->initProperties(pFbflcn, PDB_PROP_FBFLCN_BEGIN, PDB_PROP_FBFLCN_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->fbflcnHalIfacesSetupFn(pGpu, &pFbflcn->hal);

    // Initialize non-hal ptrs
    fbflcnObjIfacesSetup(pGpu, pFbflcn);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pFbflcn->hal.fbflcnSetPropertiesList(pGpu, pFbflcn);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillFbflcnObject)
        unitTestRmObject.fillFbflcnObject(pFbflcn);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pFbflcn->fbflcnGetInfoBlock = getInfloblockStub;
    pFbflcn->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pFbflcn->infoList->dataId = DATA_ID_FBFLCN;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pFbflcn->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create DPLINK object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pDplink    pointer to DPLINK object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createDplink(POBJGPU pGpu, POBJDPLINK pDplink, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pDplink->odbCommon);

    // Initialize property range
    pDplink->initProperties(pDplink, PDB_PROP_DPLINK_BEGIN, PDB_PROP_DPLINK_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->dplinkHalIfacesSetupFn(pGpu, &pDplink->hal);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pDplink->hal.dpLinkSetPropertiesList(pGpu, pDplink);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillDplinkObject)
        unitTestRmObject.fillDplinkObject(pDplink);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pDplink->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create DPAUX object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pDpaux    pointer to DPAUX object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createDpaux(POBJGPU pGpu, POBJDPAUX pDpaux, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pDpaux->odbCommon);

    // Initialize property range
    pDpaux->initProperties(pDpaux, PDB_PROP_DPAUX_BEGIN, PDB_PROP_DPAUX_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->dpauxHalIfacesSetupFn(pGpu, &pDpaux->hal);

    // Initialize non-hal ptrs
    dpauxObjIfacesSetup(pGpu, pDpaux);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pDpaux->hal.dpAuxSetPropertiesList(pGpu, pDpaux);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillDpauxObject)
        unitTestRmObject.fillDpauxObject(pDpaux);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pDpaux->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create DMA object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pDma    pointer to DMA object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createDma(POBJGPU pGpu, POBJDMA pDma, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pDma->odbCommon);

    // Initialize property range
    pDma->initProperties(pDma, PDB_PROP_DMA_BEGIN, PDB_PROP_DMA_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->dmaHalIfacesSetupFn(pGpu, &pDma->hal);

    // Initialize non-hal ptrs
    dmaObjIfacesSetup(pGpu, pDma);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pDma->hal.dmaSetPropertiesList(pGpu, pDma);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillDmaObject)
        unitTestRmObject.fillDmaObject(pDma);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pDma->dmaGetInfoBlock = getInfloblockStub;
    pDma->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pDma->infoList->dataId = DATA_ID_DMA;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pDma->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create DISP object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pDisp    pointer to DISP object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createDisp(POBJGPU pGpu, POBJDISP pDisp, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pDisp->odbCommon);

    // Initialize property range
    pDisp->initProperties(pDisp, PDB_PROP_DISPLAY_BEGIN, PDB_PROP_DISPLAY_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->dispHalIfacesSetupFn(pGpu, &pDisp->hal);

    // Initialize non-hal ptrs
    //dispObjIfacesSetup(pGpu, pDisp);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pDisp->hal.dispSetPropertiesList(pGpu, pDisp);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillDispObject)
        unitTestRmObject.fillDispObject(pDisp);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pDisp->dispGetInfoBlock = getInfloblockStub;
    pDisp->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pDisp->infoList->dataId = DATA_ID_DISP;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pDisp->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create CLK object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pClk    pointer to CLK object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createClk(POBJGPU pGpu, POBJCLK pClk, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pClk->odbCommon);

    // Initialize property range
    pClk->initProperties(pClk, PDB_PROP_CLK_BEGIN, PDB_PROP_CLK_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->clkHalIfacesSetupFn(pGpu, &pClk->hal);

    // Initialize non-hal ptrs
    clkObjIfacesSetup(pGpu, pClk);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pClk->hal.clkSetPropertiesList(pGpu, pClk);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillClkObject)
        unitTestRmObject.fillClkObject(pClk);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pClk->clkGetInfoBlock = getInfloblockStub;
    pClk->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pClk->infoList->dataId = DATA_ID_CLK;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pClk->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create BUS object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pBus    pointer to BUS object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createBus(POBJGPU pGpu, POBJBUS pBus, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pBus->odbCommon);

    // Initialize property range
    pBus->initProperties(pBus, PDB_PROP_BUS_BEGIN, PDB_PROP_BUS_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->busHalIfacesSetupFn(pGpu, &pBus->hal);

    // Initialize non-hal ptrs
    busObjIfacesSetup(pGpu, pBus);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pBus->hal.busSetPropertiesList(pGpu, pBus);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillBusObject)
        unitTestRmObject.fillBusObject(pBus);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pBus->busGetInfoBlock = getInfloblockStub;
    pBus->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pBus->infoList->dataId = DATA_ID_BUS;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pBus->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create BIF object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pBif    pointer to BIF object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createBif(POBJGPU pGpu, POBJBIF pBif, PMODULEDESCRIPTOR pMod)
{

    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pBif->odbCommon);

    // Initialize property range
    pBif->initProperties(pBif, PDB_PROP_BIF_BEGIN, PDB_PROP_BIF_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->bifHalIfacesSetupFn(pGpu, &pBif->hal);

    // Initialize non-hal ptrs
    bifObjIfacesSetup(pGpu, pBif);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pBif->hal.bifSetPropertiesList(pGpu, pBif);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillBifObject)
        unitTestRmObject.fillBifObject(pBif);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pBif->bifGetInfoBlock = getInfloblockStub;
    pBif->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pBif->infoList->dataId = DATA_ID_BIF;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pBif->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create GPU object as specified the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createGpu(POBJGPU pGpu, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pGpu->odbCommon);

    // Initialize property range
    pGpu->initProperties(pGpu, PDB_PROP_GPU_BEGIN, PDB_PROP_GPU_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->gpuHalIfacesSetupFn(pGpu, &pGpu->hal);
    // Initialize non-hal ptrs
    gpuObjIfacesSetup(pGpu);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pGpu->hal.gpuSetPropertiesList(pGpu);
    }

    if(unitTestRmObject.fillGpuObject)
        unitTestRmObject.fillGpuObject(pGpu);

    //
    // redirect odbGetObject to infra function
    // so the GPU_GET_XXX passes through infra
    //
    pGpu->odbCommon.odbGetObject = getObjectMock;

      // redirect the xxxgetInfoBlock fun ptr to infra function
    pGpu->gpuGetInfoBlock = getInfloblockStub;
    pGpu->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pGpu->infoList->dataId = DATA_ID_GPU; //  for Gpu
      // redirect the gpu read/write register operations to infra implementaion
    redirectGpuRegisterOperations(pGpu);
}

/*!
 * @brief create FLCN object as specified the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pFlcn    pointer to FLCN object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createFlcn(POBJGPU pGpu, POBJFLCN pFlcn, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pFlcn->odbCommon);

    // Initialize property range
    pFlcn->initProperties(pFlcn, PDB_PROP_FLCN_BEGIN, PDB_PROP_FLCN_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->flcnHalIfacesSetupFn(pGpu, &pFlcn->hal);

    // Initialize non-hal ptrs
    flcnObjIfacesSetup(pGpu, pFlcn);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pFlcn->hal.flcnSetPropertiesList(pGpu, pFlcn);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillFlcnObject)
        unitTestRmObject.fillFlcnObject(pFlcn);

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pFlcn->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create SYSTEM object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pSystem    pointer to SYSTEM object
 *
 */
void createSystem(POBJSYS pSystem)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pSystem->odbCommon);

    // Initialize property range
    pSystem->initProperties(pSystem, PDB_PROP_SYS_BEGIN, PDB_PROP_SYS_END);

    // Initialize non-hal ptrs
    //sysObjIfacesSetup(pSystem);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillSystemObject)
        unitTestRmObject.fillSystemObject(pSystem);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pSystem->odbCommon.odbGetObject = getObjectMock;

    g_pSys = pSystem;

}

/*!
 * @brief create SYSCON object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pSyscon    pointer to SYSCON object
 *
 */
void createSyscon(POBJSYS pSystem, POBJSYSCON pSyscon)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pSyscon->odbCommon);

    // Initialize property range
    pSyscon->initProperties(pSyscon, PDB_PROP_SYSCON_BEGIN, PDB_PROP_SYSCON_END);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillSysconObject)
        unitTestRmObject.fillSysconObject(pSyscon);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pSyscon->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pSysCon = pSyscon;
}

/*!
 * @brief create CORELOGIC object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pCorelogic    pointer to CORELOGIC object
 *
 */
void createCorelogic(POBJSYS pSystem, POBJCL pCorelogic)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pCorelogic->odbCommon);

    // Initialize property range
    pCorelogic->initProperties(pCorelogic, PDB_PROP_CL_BEGIN, PDB_PROP_CL_END);

    // Initialize non-hal ptrs
    //corelogicObjIfacesSetup(pCorelogic);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillCorelogicObject)
        unitTestRmObject.fillCorelogicObject(pCorelogic);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pCorelogic->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pCl = pCorelogic;
}

/*!
 * @brief create OS object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pOs    pointer to OS object
 *
 */
void createOs(POBJSYS pSystem, POBJOS pOs)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pOs->odbCommon);

    // Initialize property range
    pOs->initProperties(pOs, PDB_PROP_OS_BEGIN, PDB_PROP_OS_END);

    // initialize os public interfaces
    initOSFunctionPointers(pOs);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillOsObject)
        unitTestRmObject.fillOsObject(pOs);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pOs->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pOS = pOs;
}

/*!
 * @brief create PFM object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pPfm    pointer to PFM object
 *
 */
void createPfm(POBJSYS pSystem, POBJPFM pPfm)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pPfm->odbCommon);

    // Initialize property range
    pPfm->initProperties(pPfm, PDB_PROP_PLATFORM_BEGIN, PDB_PROP_PLATFORM_END);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillPfmObject)
        unitTestRmObject.fillPfmObject(pPfm);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pPfm->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pPfm = pPfm;
}

/*!
 * @brief create GPUMGR object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pGpumgr    pointer to GPUMGR object
 *
 */
void createGpumgr(POBJSYS pSystem, POBJGPUMGR pGpumgr)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pGpumgr->odbCommon);

    // Initialize property range
    pGpumgr->initProperties(pGpumgr, PDB_PROP_GPUMGR_BEGIN, PDB_PROP_GPUMGR_END);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillGpumgrObject)
        unitTestRmObject.fillGpumgrObject(pGpumgr);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pGpumgr->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pGpuMgr = pGpumgr;
}

/*!
 * @brief create GVOMGR object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pGvomgr    pointer to GVOMGR object
 *
 */
void createGvomgr(POBJSYS pSystem, POBJGVOMGR pGvomgr)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pGvomgr->odbCommon);

    // Initialize property range
    //pGvomgr->initProperties(pGvomgr, PDB_PROP_EXTDEV_GVO_BEGIN, PDB_PROP_EXTDEV_GVO_END);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillGvomgrObject)
        unitTestRmObject.fillGvomgrObject(pGvomgr);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pGvomgr->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pGvoMgr = pGvomgr;
}

/*!
 * @brief create GVIMGR object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pGvimgr    pointer to GVIMGR object
 *
 */
void createGvimgr(POBJSYS pSystem, POBJGVIMGR pGvimgr)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pGvimgr->odbCommon);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillGvimgrObject)
        unitTestRmObject.fillGvimgrObject(pGvimgr);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pGvimgr->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pGviMgr = pGvimgr;
}

/*!
 * @brief create GSYNCMGR object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pGsyncmgr  pointer to GSYNCMGR object
 *
 */
void createGsyncmgr(POBJSYS pSystem, POBJGSYNCMGR pGsyncmgr)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pGsyncmgr->odbCommon);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillGsyncmgrObject)
        unitTestRmObject.fillGsyncmgrObject(pGsyncmgr);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pGsyncmgr->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pGsyncMgr = pGsyncmgr;
}

/*!
 * @brief create SWINSTR object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pSwinstr    pointer to SWINSTR object
 *
 */
void createSwinstr(POBJSYS pSystem, POBJSWINSTR pSwinstr)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pSwinstr->odbCommon);

    // Initialize property range
    pSwinstr->initProperties(pSwinstr, PDB_PROP_SWINSTR_BEGIN, PDB_PROP_SWINSTR_END);

    // Initialize non-hal ptrs
    //swinstrObjIfacesSetup(pSwinstr);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillSwinstrObject)
        unitTestRmObject.fillSwinstrObject(pSwinstr);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pSwinstr->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pSwInstr = pSwinstr;
}

/*!
 * @brief create RCDB object as specified by the user
 *
 * @param[in]      pSystem    pointer to system root object
 *
 * @param[in]      pRcdb    pointer to RCDB object
 *
 */
void createRcdb(POBJSYS pSystem, POBJRCDB pRcdb)
{

    // Initialize ODNCOMMON function ptrs
    unitOdbInitCommon(&pRcdb->odbCommon);

    // Initialize property range
    pRcdb->initProperties(pRcdb, PDB_PROP_RCDB_BEGIN, PDB_PROP_RCDB_END);

    // Initialize non-hal ptrs
    //rcdbObjIfacesSetup(pRcdb);

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillRcdbObject)
        unitTestRmObject.fillRcdbObject(pRcdb);

    //redirect odbGetObject ao that OBJ_GET_XXX is intercepted by getObjectMock
    pRcdb->odbCommon.odbGetObject = getObjectMock;

    if (g_pSys == NULL)
        createSystem(pSystem);

    pSystem->pRcDB = pRcdb;
}

/*!
 * @brief create LWJPG object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pLwjpg  pointer to LWJPG object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createLwjpg(POBJGPU pGpu, POBJLWJPG pLwjpg, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pLwjpg->odbCommon);

    // Initialize property range
    pLwjpg->initProperties(pLwjpg, PDB_PROP_LWJPG_BEGIN, PDB_PROP_LWJPG_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->lwjpgHalIfacesSetupFn(pGpu, &pLwjpg->hal);

    // Initialize non-hal ptrs
    lwjpgObjIfacesSetup(pGpu, pLwjpg);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pLwjpg->hal.lwjpgSetPropertiesList(pGpu, pLwjpg);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillLwjpgObject)
        unitTestRmObject.fillLwjpgObject(pLwjpg);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pLwjpg->lwjpgGetInfoBlock = getInfloblockStub;
    pLwjpg->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pLwjpg->infoList->dataId = DATA_ID_LWJPG;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pLwjpg->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief create OFA object as specified by the user
 *
 * @param[in]      pGpu    pointer to GPU object
 *
 * @param[in]      pOfa  pointer to OFA object
 *
 * @param[in]      pMod    pointer to module descriptor, to init
                           HAL ptrs
 *
 */
void createOfa(POBJGPU pGpu, POBJOFA pOfa, PMODULEDESCRIPTOR pMod)
{
    // Initialize ODBCOMMON function ptrs
    unitOdbInitCommon(&pOfa->odbCommon);

    // Initialize property range
    pOfa->initProperties(pOfa, PDB_PROP_OFA_BEGIN, PDB_PROP_OFA_END);

    // Initialize hal ptrs
    pMod->pHalSetIfaces->ofaHalIfacesSetupFn(pGpu, &pOfa->hal);

    // Initialize non-hal ptrs
    ofaObjIfacesSetup(pGpu, pOfa);

    if (pdbInit)
    {
        // Inititalize PDB Properties
        pOfa->hal.ofaSetPropertiesList(pGpu, pOfa);
    }

    // fill the object as specified by the test writer
    if(unitTestRmObject.fillOfaObject)
        unitTestRmObject.fillOfaObject(pOfa);

    // redirect the xxxgetInfoBlock fun ptr to infra function
    pOfa->ofaGetInfoBlock = getInfloblockStub;
    pOfa->infoList = (PENG_INFO_LINK_NODE)unitMalloc(sizeof(ENG_INFO_LINK_NODE), UNIT_CLASS_MISC);
    pOfa->infoList->dataId = DATA_ID_OFA;

    // redirect odbGetObject ao that ENG_GET_XXX is intercepted by getObjectMock
    pOfa->odbCommon.odbGetObject = getObjectMock;
}

/*!
 * @brief Disable a particular engine/object
 *
 * @param[in]      id    Any legal value from set of enums in DATA_ID
 *                       corresponding to particular engine/object
 *
 */
void enableEngineMissing(DATA_ID id)
{

    UNIT_ASSERT(0 <= id && DATA_ID_FREE_OBJ > id);

    unitMissingEngineBlock[id] = LW_TRUE;

}

/*!
 * @brief check if the engine corresponding to given id,
 *              is disabled
 *
 * @param[in]      id    Any legal value from set of enums in DATA_ID
 *                       corresponding to particular engine/object
 *
 * @return         LW_TRUE, if engine disabled LW_FALSE otherwise
 */
LwBool isEngineMissing(DATA_ID id)
{

    UNIT_ASSERT(0 <= id && DATA_ID_FREE_OBJ > id);

    return unitMissingEngineBlock[id];
}

/*!
 * @brief redirected function, to create user
 *        specified chip specific OBJXXX
 *
 * @param[in]      pGpuCommon           pointer to GPU ODBCOMMON
 *
 * @param[in]      ODB_CLASS            enum to specify class/engine
 *
 * @param[in]      requestedPublicID    not used, only for the protype matching
 *
 * @param[in]      odbObjFlags          not used, only for the protype matching
 *
 * @return         Pointer to ODBCOMMON of requested object
 *                 If the object doesnt exist then it's created
 */
PODBCOMMON
getObjectMock
(
    PODBCOMMON pGpuCommon,
    ODB_CLASS requestedClass,
    LwU32 requestedPublicID,
    LwU32 odbObjFlags
)
{
    static void *pDpu = NULL;
    static void *pRes = NULL;
    static void *pPgeng = NULL;
    static void *pPgctrl = NULL;
    static void *pLpwr = NULL;
    static void *pInforom = NULL;
    static void *pMsenc = NULL;
    static void *pVic = NULL;
    static void *pSpb = NULL;
    static void *pPmu = NULL;
    static void *pCE = NULL;
    static void *pIsohub = NULL;
    static void *pCve = NULL;
    static void *pCipher = NULL;
    static void *pHdmi = NULL;
    static void *pHdcp = NULL;
    static void *pHdtv = NULL;
    static void *pVp = NULL;
    static void *pMp = NULL;
    static void *pMpeg = NULL;
    static void *pBsp = NULL;
    static void *pSmu = NULL;
    static void *pSor = NULL;
    static void *pPior = NULL;
    static void *pOr = NULL;
    static void *pTherm = NULL;
    static void *pVolt = NULL;
    static void *pFuse = NULL;
    static void *pFan = NULL;
    static void *pGpio = NULL;
    static void *pI2c = NULL;
    static void *pGpu = NULL;
    static void *pRc = NULL;
    static void *pVbios = NULL;
    static void *pVga = NULL;
    static void *pMsppp = NULL;
    static void *pSeq = NULL;
    static void *pTmr = NULL;
    static void *pStereo = NULL;
    static void *pPerf = NULL;
    static void *pMc = NULL;
    static void *pIntr = NULL;
    static void *pInst = NULL;
    static void *pHead = NULL;
    static void *pGr = NULL;
    static void *pFlcn = NULL;
    static void *pFifo = NULL;
    static void *pFbsr = NULL;
    static void *pFb = NULL;
    static void *pFbflcn = NULL;
    static void *pDplink = NULL;
    static void *pDpaux = NULL;
    static void *pDma = NULL;
    static void *pDisp = NULL;
    static void *pDac = NULL;
    static void *pClk = NULL;
    static void *pBus = NULL;
    static void *pBif = NULL;
    static void *pLwjpg = NULL:
    static void *pOfa = NULL:

    static void *pSystem = NULL;
    static void *pSyscon = NULL;
    static void *pCorelogic = NULL;
    static void *pOs = NULL;
    static void *pPfm = NULL;
    static void *pGpumgr = NULL;
    static void *pGvomgr = NULL;
    static void *pGvimgr = NULL;
    static void *pGsyncmgr = NULL;
    static void *pSwinstr = NULL;
    static void *pRcdb = NULL;

    static PMODULEDESCRIPTOR pMod = NULL;
    if (!pMod)
    {
        pMod = (PMODULEDESCRIPTOR)unitMalloc(sizeof(MODULEDESCRIPTOR), UNIT_CLASS_MISC);
        memset(pMod, 0, sizeof(MODULEDESCRIPTOR));
        useChip(pMod);
    }

    switch (requestedClass)
    {
        case ODB_CLASS_GPU:
            if (isEngineMissing(DATA_ID_GPU))
                return NULL;

            if (!pGpu)
            {
                pGpu = (POBJGPU)unitMalloc(sizeof(OBJGPU), UNIT_CLASS_RM_OBJECT);
                memset(pGpu, 0, sizeof(OBJGPU));
                createGpu(pGpu, pMod);
            }

            return (PODBCOMMON)pGpu;
            break;

        case ODB_CLASS_DPU:

            if (isEngineMissing(DATA_ID_DPU))
                return NULL;

            if (!pDpu)
            {
                pDpu = (POBJDPU)unitMalloc(sizeof(OBJDPU), UNIT_CLASS_RM_OBJECT);
                memset(pDpu, 0, sizeof(OBJDPU));
                createDpu(pGpu, pDpu, pMod);
            }

            return (PODBCOMMON)pDpu;
            break;

        case ODB_CLASS_RES:

            if (isEngineMissing(DATA_ID_RES))
                return NULL;

            if (!pRes)
            {
                pRes = (POBJRES)unitMalloc(sizeof(OBJRES), UNIT_CLASS_RM_OBJECT);
                memset(pRes, 0, sizeof(OBJRES));
                createRes(pGpu, pRes, pMod);
            }

            return (PODBCOMMON)pRes;
            break;

        case ODB_CLASS_PGENG:

            if (isEngineMissing(DATA_ID_PGENG))
                return NULL;

            if (!pPgeng)
            {
                pPgeng = (POBJPGENG)unitMalloc(sizeof(OBJPGENG), UNIT_CLASS_RM_OBJECT);
                memset(pPgeng, 0, sizeof(OBJPGENG));
                createPgeng(pGpu, pPgeng, pMod);
            }

            return (PODBCOMMON)pPgeng;
            break;

        case ODB_CLASS_PGCTRL:

            if (isEngineMissing(DATA_ID_PGCTRL))
                return NULL;

            if (!pPgctrl)
            {
                pPgctrl = (POBJPGCTRL)unitMalloc(sizeof(OBJPGCTRL), UNIT_CLASS_RM_OBJECT);
                memset(pPgctrl, 0, sizeof(OBJPGCTRL));
                createPgctrl(pGpu, pPgctrl, pMod);
            }

            return (PODBCOMMON)pPgctrl;
            break;

        case ODB_CLASS_PG:

            if (isEngineMissing(DATA_ID_PG))
                return NULL;

            if (!pLpwr)
            {
                pLpwr = (POBJLPWR)unitMalloc(sizeof(OBJLPWR), UNIT_CLASS_RM_OBJECT);
                memset(pLpwr, 0, sizeof(OBJLPWR));
                createLpwr(pGpu, pLpwr, pMod);
            }

            return (PODBCOMMON)pLpwr;
            break;

        case ODB_CLASS_INFOROM:

            if (isEngineMissing(DATA_ID_INFOROM))
                return NULL;

            if (!pInforom)
            {
                pInforom = (POBJINFOROM)unitMalloc(sizeof(OBJINFOROM), UNIT_CLASS_RM_OBJECT);
                memset(pInforom, 0, sizeof(OBJINFOROM));
                createInforom(pGpu, pInforom, pMod);
            }

            return (PODBCOMMON)pInforom;
            break;

        case ODB_CLASS_MSENC:

            if (isEngineMissing(DATA_ID_MSENC))
                return NULL;

            if (!pMsenc)
            {
                pMsenc = (POBJMSENC)unitMalloc(sizeof(OBJMSENC), UNIT_CLASS_RM_OBJECT);
                memset(pMsenc, 0, sizeof(OBJMSENC));
                createMsenc(pGpu, pMsenc, pMod);
            }

            return (PODBCOMMON)pMsenc;
            break;

        case ODB_CLASS_VIC:

            if (isEngineMissing(DATA_ID_VIC))
                return NULL;

            if (!pVic)
            {
                pVic = (POBJVIC)unitMalloc(sizeof(OBJVIC), UNIT_CLASS_RM_OBJECT);
                memset(pVic, 0, sizeof(OBJVIC));
                createVic(pGpu, pVic, pMod);
            }

            return (PODBCOMMON)pVic;
            break;

        case ODB_CLASS_SPB:

            if (isEngineMissing(DATA_ID_SPB))
                return NULL;

            if (!pSpb)
            {
                pSpb = (POBJSPB)unitMalloc(sizeof(OBJSPB), UNIT_CLASS_RM_OBJECT);
                memset(pSpb, 0, sizeof(OBJSPB));
                createSpb(pGpu, pSpb, pMod);
            }

            return (PODBCOMMON)pSpb;
            break;

        case ODB_CLASS_PMU:

            if (isEngineMissing(DATA_ID_PMU))
                return NULL;

            if (!pPmu)
            {
                pPmu = (POBJPMU)unitMalloc(sizeof(OBJPMU), UNIT_CLASS_RM_OBJECT);
                memset(pPmu, 0, sizeof(OBJPMU));
                createPmu(pGpu, pPmu, pMod);
            }

            return (PODBCOMMON)pPmu;
            break;

        case ODB_CLASS_CE:

            if (isEngineMissing(DATA_ID_CE))
                return NULL;

            if (!pCE)
            {
                pCE = (POBJCE)unitMalloc(sizeof(OBJCE), UNIT_CLASS_RM_OBJECT);
                memset(pCE, 0, sizeof(OBJCE));
                createCE(pGpu, pCE, pMod);
            }

            return (PODBCOMMON)pCE;
            break;

        case ODB_CLASS_ISOHUB:

            if (isEngineMissing(DATA_ID_ISOHUB))
                return NULL;

            if (!pIsohub)
            {
                pIsohub = (POBJISOHUB)unitMalloc(sizeof(OBJISOHUB), UNIT_CLASS_RM_OBJECT);
                memset(pIsohub, 0, sizeof(OBJISOHUB));
                createIsohub(pGpu, pIsohub, pMod);
            }

            return (PODBCOMMON)pIsohub;
            break;

        case ODB_CLASS_CVE:

            if (isEngineMissing(DATA_ID_CVE))
                return NULL;

            if (!pCve)
            {
                pCve = (POBJCVE)unitMalloc(sizeof(OBJCVE), UNIT_CLASS_RM_OBJECT);
                memset(pCve, 0, sizeof(OBJCVE));
                createCve(pGpu, pCve, pMod);
            }

            return (PODBCOMMON)pCve;
            break;

        case ODB_CLASS_CIPHER:

            if (isEngineMissing(DATA_ID_CIPHER))
                return NULL;

            if (!pCipher)
            {
                pCipher = (POBJCIPHER)unitMalloc(sizeof(OBJCIPHER), UNIT_CLASS_RM_OBJECT);
                memset(pCipher, 0, sizeof(OBJCIPHER));
                createCipher(pGpu, pCipher, pMod);
            }

            return (PODBCOMMON)pCipher;
            break;

        case ODB_CLASS_HDMI:

            if (isEngineMissing(DATA_ID_HDMI))
                return NULL;

            if (!pHdmi)
            {
                pHdmi = (POBJHDMI)unitMalloc(sizeof(OBJHDMI), UNIT_CLASS_RM_OBJECT);
                memset(pHdmi, 0, sizeof(OBJHDMI));
                createHdmi(pGpu, pHdmi, pMod);
            }

            return (PODBCOMMON)pHdmi;
            break;

        case ODB_CLASS_HDCP:

            if (isEngineMissing(DATA_ID_HDCP))
                return NULL;

            if (!pHdcp)
            {
                pHdcp = (POBJHDCP)unitMalloc(sizeof(OBJHDCP), UNIT_CLASS_RM_OBJECT);
                memset(pHdcp, 0, sizeof(OBJHDCP));
                createHdcp(pGpu, pHdcp, pMod);
            }

            return (PODBCOMMON)pHdcp;
            break;

        case ODB_CLASS_HDTV:

            if (isEngineMissing(DATA_ID_HDTV))
                return NULL;

            if (!pHdtv)
            {
                pHdtv = (POBJHDTV)unitMalloc(sizeof(OBJHDTV), UNIT_CLASS_RM_OBJECT);
                memset(pHdtv, 0, sizeof(OBJHDTV));
                createHdtv(pGpu, pHdtv, pMod);
            }

            return (PODBCOMMON)pHdtv;
            break;

        case ODB_CLASS_VIDEO_PROCESSOR:

            if (isEngineMissing(DATA_ID_VP))
                return NULL;

            if (!pVp)
            {
                pVp = (POBJVP)unitMalloc(sizeof(OBJVP), UNIT_CLASS_RM_OBJECT);
                memset(pVp, 0, sizeof(OBJVP));
                createVp(pGpu, pVp, pMod);
            }

            return (PODBCOMMON)pVp;
            break;

        case ODB_CLASS_MEDIAPORT:

            if (isEngineMissing(DATA_ID_MP))
                return NULL;

            if (!pMp)
            {
                pMp = (POBJMP)unitMalloc(sizeof(OBJMP), UNIT_CLASS_RM_OBJECT);
                memset(pMp, 0, sizeof(OBJMP));
                createMp(pGpu, pMp, pMod);
            }

            return (PODBCOMMON)pMp;
            break;

        case ODB_CLASS_MPEG:

            if (isEngineMissing(DATA_ID_MPEG))
                return NULL;

            if (!pMpeg)
            {
                pMpeg = (POBJMPEG)unitMalloc(sizeof(OBJMPEG), UNIT_CLASS_RM_OBJECT);
                memset(pMpeg, 0, sizeof(OBJMPEG));
                createMpeg(pGpu, pMpeg, pMod);
            }

            return (PODBCOMMON)pMpeg;
            break;

        case ODB_CLASS_BSP:

            if (isEngineMissing(DATA_ID_BSP))
                return NULL;

            if (!pBsp)
            {
                pBsp = (POBJBSP)unitMalloc(sizeof(OBJBSP), UNIT_CLASS_RM_OBJECT);
                memset(pBsp, 0, sizeof(OBJBSP));
                createBsp(pGpu, pBsp, pMod);
            }

            return (PODBCOMMON)pBsp;
            break;

        case ODB_CLASS_SMU:

            if (isEngineMissing(DATA_ID_SMU))
                return NULL;

            if (!pSmu)
            {
                pSmu = (POBJSMU)unitMalloc(sizeof(OBJSMU), UNIT_CLASS_RM_OBJECT);
                memset(pSmu, 0, sizeof(OBJSMU));
                createSmu(pGpu, pSmu, pMod);
            }

            return (PODBCOMMON)pSmu;
            break;

        case ODB_CLASS_SOR:

            if (isEngineMissing(DATA_ID_SOR))
                return NULL;

            if (!pSor)
            {
                pSor = (POBJSOR)unitMalloc(sizeof(OBJSOR), UNIT_CLASS_RM_OBJECT);
                memset(pSor, 0, sizeof(OBJSOR));
                createSor(pGpu, pSor, pMod);
            }

            return (PODBCOMMON)pSor;
            break;

        case ODB_CLASS_PIOR:

            if (isEngineMissing(DATA_ID_PIOR))
                return NULL;

            if (!pPior)
            {
                pPior = (POBJPIOR)unitMalloc(sizeof(OBJPIOR), UNIT_CLASS_RM_OBJECT);
                memset(pPior, 0, sizeof(OBJPIOR));
                createPior(pGpu, pPior, pMod);
            }

            return (PODBCOMMON)pPior;
            break;

        case ODB_CLASS_THERM:

            if (isEngineMissing(DATA_ID_THERM))
                return NULL;

            if (!pTherm)
            {
                pTherm = (POBJTHERM)unitMalloc(sizeof(OBJTHERM), UNIT_CLASS_RM_OBJECT);
                memset(pTherm, 0, sizeof(OBJTHERM));
                createTherm(pGpu, pTherm, pMod);
            }

            return (PODBCOMMON)pTherm;
            break;

        case ODB_CLASS_VOLT:

            if (isEngineMissing(DATA_ID_VOLT))
                return NULL;

            if (!pVolt)
            {
                pVolt = (POBJVOLT)unitMalloc(sizeof(OBJVOLT), UNIT_CLASS_RM_OBJECT);
                memset(pVolt, 0, sizeof(OBJVOLT));
                createVolt(pGpu, pVolt, pMod);
            }

            return (PODBCOMMON)pVolt;
            break;

        case ODB_CLASS_FUSE:

            if (isEngineMissing(DATA_ID_FUSE))
                return NULL;

            if (!pFuse)
            {
                pFuse = (POBJFUSE)unitMalloc(sizeof(OBJFUSE), UNIT_CLASS_RM_OBJECT);
                memset(pFuse, 0, sizeof(OBJFUSE));
                createFuse(pGpu, pFuse, pMod);
            }

            return (PODBCOMMON)pFuse;
            break;

        case ODB_CLASS_FAN:

            if (isEngineMissing(DATA_ID_FAN))
                return NULL;

            if (!pFan)
            {
                pFan = (POBJFAN)unitMalloc(sizeof(OBJFAN), UNIT_CLASS_RM_OBJECT);
                memset(pFan, 0, sizeof(OBJFAN));
                createFan(pGpu, pFan, pMod);
            }

            return (PODBCOMMON)pFan;
            break;

        case ODB_CLASS_GPIO:

            if (isEngineMissing(DATA_ID_GPIO))
                return NULL;

            if (!pGpio)
            {
                pGpio = (POBJGPIO)unitMalloc(sizeof(OBJGPIO), UNIT_CLASS_RM_OBJECT);
                memset(pGpio, 0, sizeof(OBJGPIO));
                createGpio(pGpu, pGpio, pMod);
            }

            return (PODBCOMMON)pGpio;
            break;

        case ODB_CLASS_I2C:

            if (isEngineMissing(DATA_ID_I2C))
                return NULL;

            if (!pI2c)
            {
                pI2c = (OBJI2C *)unitMalloc(sizeof(OBJI2C), UNIT_CLASS_RM_OBJECT);
                memset(pI2c, 0, sizeof(OBJI2C));
                createI2c(pGpu, pI2c, pMod);
            }

            return (PODBCOMMON)pI2c;
            break;

        case ODB_CLASS_RC:

            if (isEngineMissing(DATA_ID_RC))
                return NULL;

            if (!pRc)
            {
                pRc = (POBJRC)unitMalloc(sizeof(OBJRC), UNIT_CLASS_RM_OBJECT);
                memset(pRc, 0, sizeof(OBJRC));
                createRc(pGpu, pRc, pMod);
            }

            return (PODBCOMMON)pRc;
            break;

        case ODB_CLASS_VBIOS:

            if (isEngineMissing(DATA_ID_VBIOS))
                return NULL;

            if (!pVbios)
            {
                pVbios = (POBJVBIOS)unitMalloc(sizeof(OBJVBIOS), UNIT_CLASS_RM_OBJECT);
                memset(pVbios, 0, sizeof(OBJVBIOS));
                createVbios(pGpu, pVbios, pMod);
            }

            return (PODBCOMMON)pVbios;
            break;

        case ODB_CLASS_VGA:

            if (isEngineMissing(DATA_ID_VGA))
                return NULL;

            if (!pVga)
            {
                pVga = (POBJVGA)unitMalloc(sizeof(OBJVGA), UNIT_CLASS_RM_OBJECT);
                memset(pVga, 0, sizeof(OBJVGA));
                createVga(pGpu, pVga, pMod);
            }

            return (PODBCOMMON)pVga;
            break;

        case ODB_CLASS_PPP:

            if (isEngineMissing(DATA_ID_PPP))
                return NULL;

            if (!pMsppp)
            {
                pMsppp = (POBJMSPPP)unitMalloc(sizeof(OBJMSPPP), UNIT_CLASS_RM_OBJECT);
                memset(pMsppp, 0, sizeof(OBJMSPPP));
                createPpp(pGpu, pMsppp, pMod);
            }

            return (PODBCOMMON)pMsppp;
            break;

        case ODB_CLASS_SEQ:

            if (isEngineMissing(DATA_ID_SEQ))
                return NULL;

            if (!pSeq)
            {
                pSeq = (POBJSEQ)unitMalloc(sizeof(OBJSEQ), UNIT_CLASS_RM_OBJECT);
                memset(pSeq, 0, sizeof(OBJSEQ));
                createSeq(pGpu, pSeq, pMod);
            }

            return (PODBCOMMON)pSeq;
            break;

        case ODB_CLASS_TIMER:

            if (isEngineMissing(DATA_ID_TMR))
                return NULL;

            if (!pTmr)
            {
                pTmr = (POBJTMR)unitMalloc(sizeof(OBJTMR), UNIT_CLASS_RM_OBJECT);
                memset(pTmr, 0, sizeof(OBJTMR));
                createTmr(pGpu, pTmr, pMod);
            }

            return (PODBCOMMON)pTmr;
            break;

        case ODB_CLASS_STEREO:

            if (isEngineMissing(DATA_ID_STEREO))
                return NULL;

            if (!pStereo)
            {
                pStereo = (POBJSTEREO)unitMalloc(sizeof(OBJSTEREO), UNIT_CLASS_RM_OBJECT);
                memset(pStereo, 0, sizeof(OBJSTEREO));
                createStereo(pGpu, pStereo, pMod);
            }

            return (PODBCOMMON)pStereo;
            break;

        case ODB_CLASS_PERF:

            if (isEngineMissing(DATA_ID_PERF))
                return NULL;

            if (!pPerf)
            {
                pPerf = (POBJPERF)unitMalloc(sizeof(OBJPERF), UNIT_CLASS_RM_OBJECT);
                memset(pPerf, 0, sizeof(OBJPERF));
                createPerf(pGpu, pPerf, pMod);
            }

            return (PODBCOMMON)pPerf;
            break;

        case ODB_CLASS_MC:

            if (isEngineMissing(DATA_ID_MC))
                return NULL;

            if (!pMc)
            {
                pMc = (POBJMC)unitMalloc(sizeof(OBJMC), UNIT_CLASS_RM_OBJECT);
                memset(pMc, 0, sizeof(OBJMC));
                createMc(pGpu, pMc, pMod);
            }

            return (PODBCOMMON)pMc;
            break;

        case ODB_CLASS_INTR:

            if (isEngineMissing(DATA_ID_INTR))
                return NULL;

            if (!pIntr)
            {
                pIntr = (POBJINTR)unitMalloc(sizeof(OBJINTR), UNIT_CLASS_RM_OBJECT);
                memset(pIntr, 0, sizeof(OBJINTR));
                createIntr(pGpu, pIntr, pMod);
            }

            return (PODBCOMMON)pIntr;
            break;

        case ODB_CLASS_INSTMEM:

            if (isEngineMissing(DATA_ID_INST))
                return NULL;

            if (!pInst)
            {
                pInst = (POBJINST)unitMalloc(sizeof(OBJINST), UNIT_CLASS_RM_OBJECT);
                memset(pInst, 0, sizeof(OBJINST));
                createInst(pGpu, pInst, pMod);
            }

            return (PODBCOMMON)pInst;
            break;

        case ODB_CLASS_HEAD:

            if (isEngineMissing(DATA_ID_HEAD))
                return NULL;

            if (!pHead)
            {
                pHead = (POBJHEAD)unitMalloc(sizeof(OBJHEAD), UNIT_CLASS_RM_OBJECT);
                memset(pHead, 0, sizeof(OBJHEAD));
                createHead(pGpu, pHead, pMod);
            }

            return (PODBCOMMON)pHead;
            break;

        case ODB_CLASS_GRAPHICS:

            if (isEngineMissing(DATA_ID_GR))
                return NULL;

            if (!pGr)
            {
                pGr = (POBJGR)unitMalloc(sizeof(OBJGR), UNIT_CLASS_RM_OBJECT);
                memset(pGr, 0, sizeof(OBJGR));
                createGr(pGpu, pGr, pMod);
            }

            return (PODBCOMMON)pGr;
            break;

        case ODB_CLASS_FLCN:

            if (isEngineMissing(DATA_ID_FLCN))
                return NULL;

            if (!pFlcn)
            {
                pFlcn = (POBJFLCN)unitMalloc(sizeof(OBJFLCN), UNIT_CLASS_RM_OBJECT);
                memset(pFlcn, 0, sizeof(OBJFLCN));
                createFlcn(pGpu, pFlcn, pMod);
            }

            return (PODBCOMMON)pFlcn;
            break;

        case ODB_CLASS_FIFO:

            if (isEngineMissing(DATA_ID_FIFO))
                return NULL;

            if (!pFifo)
            {
                pFifo = (POBJFIFO)unitMalloc(sizeof(OBJFIFO), UNIT_CLASS_RM_OBJECT);
                memset(pFifo, 0, sizeof(OBJFIFO));
                createFifo(pGpu, pFifo, pMod);
            }

            return (PODBCOMMON)pFifo;
            break;

        case ODB_CLASS_FBSR:

            if (isEngineMissing(DATA_ID_FBSR))
                return NULL;

            if (!pFbsr)
            {
                pFbsr = (POBJFBSR)unitMalloc(sizeof(OBJFBSR), UNIT_CLASS_RM_OBJECT);
                memset(pFbsr, 0, sizeof(OBJFBSR));
                createFbsr(pGpu, pFbsr, pMod);
            }

            return (PODBCOMMON)pFbsr;
            break;

        case ODB_CLASS_FB:

            if (isEngineMissing(DATA_ID_FB))
                return NULL;

            if (!pFb)
            {
                pFb = (POBJFB)unitMalloc(sizeof(OBJFB), UNIT_CLASS_RM_OBJECT);
                memset(pFb, 0, sizeof(OBJFB));
                createFb(pGpu, pFb, pMod);
            }

            return (PODBCOMMON)pFb;
            break;

        case ODB_CLASS_FBFLCN:

            if (isEngineMissing(DATA_ID_FBFLCN))
                return NULL;

            if (!pFbflcn)
            {
                pFbflcn = (POBJFBFLCN)unitMalloc(sizeof(OBJFBFLCN), UNIT_CLASS_RM_OBJECT);
                memset(pFbflcn, 0, sizeof(OBJFBFLCN));
                createFbflcn(pGpu, pFbflcn, pMod);
            }

            return (PODBCOMMON)pFbflcn;
            break;

        case ODB_CLASS_DPLINK:

            if (isEngineMissing(DATA_ID_DPLINK))
                return NULL;

            if (!pDplink)
            {
                pDplink = (POBJDPLINK)unitMalloc(sizeof(OBJDPLINK), UNIT_CLASS_RM_OBJECT);
                memset(pDplink, 0, sizeof(OBJDPLINK));
                createDplink(pGpu, pDplink, pMod);
            }

            return (PODBCOMMON)pDplink;
            break;

        case ODB_CLASS_DPAUX:

            if (isEngineMissing(DATA_ID_DPAUX))
                return NULL;

            if (!pDpaux)
            {
                pDpaux = (POBJDPAUX)unitMalloc(sizeof(OBJDPAUX), UNIT_CLASS_RM_OBJECT);
                memset(pDpaux, 0, sizeof(OBJDPAUX));
                createDpaux(pGpu, pDpaux, pMod);
            }

            return (PODBCOMMON)pDpaux;
            break;

        case ODB_CLASS_DMA:

            if (isEngineMissing(DATA_ID_DMA))
                return NULL;

            if (!pDma)
            {
                pDma = (POBJDMA)unitMalloc(sizeof(OBJDMA), UNIT_CLASS_RM_OBJECT);
                memset(pDma, 0, sizeof(OBJDMA));
                createDma(pGpu, pDma, pMod);
            }

            return (PODBCOMMON)pDma;
            break;

        case ODB_CLASS_DISPLAY:

            if (isEngineMissing(DATA_ID_DISP))
                return NULL;

            if (!pDisp)
            {
                pDisp = (POBJDISP)unitMalloc(sizeof(OBJDISP), UNIT_CLASS_RM_OBJECT);
                memset(pDisp, 0, sizeof(OBJDISP));
                createDisp(pGpu, pDisp, pMod);
            }

            return (PODBCOMMON)pDisp;
            break;

        case ODB_CLASS_DAC:

            if (isEngineMissing(DATA_ID_DAC))
                return NULL;

            if (!pDac)
            {
                pDac = (POBJDAC)unitMalloc(sizeof(OBJDAC), UNIT_CLASS_RM_OBJECT);
                memset(pDac, 0, sizeof(OBJDAC));
                createDac(pGpu, pDac, pMod);
            }

            return (PODBCOMMON)pDac;
            break;

        case ODB_CLASS_CLK:

            if (isEngineMissing(DATA_ID_CLK))
                return NULL;

            if (!pClk)
            {
                pClk = (POBJCLK)unitMalloc(sizeof(OBJCLK), UNIT_CLASS_RM_OBJECT);
                memset(pClk, 0, sizeof(OBJCLK));
                createClk(pGpu, pClk, pMod);
            }

            return (PODBCOMMON)pClk;
            break;

        case ODB_CLASS_BUS:

            if (isEngineMissing(DATA_ID_BUS))
                return NULL;

            if (!pBus)
            {
                pBus = (POBJBUS)unitMalloc(sizeof(OBJBUS), UNIT_CLASS_RM_OBJECT);
                memset(pBus, 0, sizeof(OBJBUS));
                createBus(pGpu, pBus, pMod);
            }

            return (PODBCOMMON)pBus;
            break;

        case ODB_CLASS_BIF:

            if (isEngineMissing(DATA_ID_BIF))
                return NULL;

            if (!pBif)
            {
                pBif = (POBJBIF)unitMalloc(sizeof(OBJBIF), UNIT_CLASS_RM_OBJECT);
                memset(pBif, 0, sizeof(OBJBIF));
                createBif(pGpu, pBif, pMod);
            }

            return (PODBCOMMON)pBif;
            break;

        case ODB_CLASS_LWJPG:

            if (isEngineMissing(DATA_ID_LWJPG))
                return NULL;

            if (!pLwjpg)
            {
                pLwjpg = (POBJLWJPG)unitMalloc(sizeof(OBJLWJPG), UNIT_CLASS_RM_OBJECT);
                memset(pLwjpg, 0, sizeof(OBJLWJPG));
                createLwjpg(pGpu, pLwjpg, pMod);
            }

            return (PODBCOMMON)pLwjpg;
            break;

        case ODB_CLASS_OFA:

            if (isEngineMissing(DATA_ID_OFA))
                return NULL;

            if (!pOfa)
            {
                pOfa = (POBJOFA)unitMalloc(sizeof(OBJOFA), UNIT_CLASS_RM_OBJECT);
                memset(pOfa, 0, sizeof(OBJOFA));
                createOfa(pGpu, pOfa, pMod);
            }

            return (PODBCOMMON)pOfa;
            break;

        case ODB_CLASS_SYS:

            if (isEngineMissing(DATA_ID_SYSTEM))
                return NULL;

            if (!pSystem)
            {
                pSystem = (POBJSYS)unitMalloc(sizeof(OBJSYS), UNIT_CLASS_RM_OBJECT);
                memset(pSystem, 0, sizeof(OBJSYS));
                createSystem(pSystem);
            }

            return (PODBCOMMON)pSystem;
            break;

        case ODB_CLASS_SYSCON:

            if (isEngineMissing(DATA_ID_SYSCON))
                return NULL;

            if (!pSyscon)
            {
                pSyscon = (POBJSYSCON)unitMalloc(sizeof(OBJSYSCON), UNIT_CLASS_RM_OBJECT);
                memset(pSyscon, 0, sizeof(OBJSYSCON));
                createSyscon(pSystem, pSyscon);
            }

            return (PODBCOMMON)pSyscon;
            break;

        case ODB_CLASS_CL:

            if (isEngineMissing(DATA_ID_CORELOGIC))
                return NULL;

            if (!pCorelogic)
            {
                pCorelogic = (POBJCL)unitMalloc(sizeof(OBJCL), UNIT_CLASS_RM_OBJECT);
                memset(pCorelogic, 0, sizeof(OBJCL));
                createCorelogic(pSystem, pCorelogic);
            }

            return (PODBCOMMON)pCorelogic;
            break;

        case ODB_CLASS_OS:

            if (isEngineMissing(DATA_ID_OS))
                return NULL;

            if (!pOs)
            {
                pOs = (POBJOS)unitMalloc(sizeof(OBJOS), UNIT_CLASS_RM_OBJECT);
                memset(pOs, 0, sizeof(OBJOS));
                createOs(pSystem, pOs);
            }

            return (PODBCOMMON)pOs;
            break;

        case ODB_CLASS_PFM:

            if (isEngineMissing(DATA_ID_PFM))
                return NULL;

            if (!pPfm)
            {
                pPfm = (POBJPFM)unitMalloc(sizeof(OBJPFM), UNIT_CLASS_RM_OBJECT);
                memset(pPfm, 0, sizeof(OBJPFM));
                createPfm(pSystem, pPfm);
            }

            return (PODBCOMMON)pPfm;
            break;

        case ODB_CLASS_GPUMGR:

            if (isEngineMissing(DATA_ID_GPUMGR))
                return NULL;

            if (!pGpumgr)
            {
                pGpumgr = (POBJGPUMGR)unitMalloc(sizeof(OBJGPUMGR), UNIT_CLASS_RM_OBJECT);
                memset(pGpumgr, 0, sizeof(OBJGPUMGR));
                createGpumgr(pSystem, pGpumgr);
            }

            return (PODBCOMMON)pGpumgr;
            break;

        case ODB_CLASS_GVOMGR:

            if (isEngineMissing(DATA_ID_GVOMGR))
                return NULL;

            if (!pGvomgr)
            {
                pGvomgr = (POBJGVOMGR)unitMalloc(sizeof(OBJGVOMGR), UNIT_CLASS_RM_OBJECT);
                memset(pGvomgr, 0, sizeof(OBJGVOMGR));
                createGvomgr(pSystem, pGvomgr);
            }

            return (PODBCOMMON)pGvomgr;
            break;

        case ODB_CLASS_GVIMGR:

            if (isEngineMissing(DATA_ID_GVIMGR))
                return NULL;

            if (!pGvimgr)
            {
                pGvimgr = (POBJGVIMGR)unitMalloc(sizeof(OBJGVIMGR), UNIT_CLASS_RM_OBJECT);
                memset(pGvimgr, 0, sizeof(OBJGVIMGR));
                createGvimgr(pSystem, pGvimgr);
            }

            return (PODBCOMMON)pGvimgr;
            break;

        case ODB_CLASS_GSYNCMGR:

            if (isEngineMissing(DATA_ID_GSYNCMGR))
                return NULL;

            if (!pGsyncmgr)
            {
                pGsyncmgr = (POBJGSYNCMGR)unitMalloc(sizeof(OBJGSYNCMGR), UNIT_CLASS_RM_OBJECT);
                memset(pGsyncmgr, 0, sizeof(OBJGSYNCMGR));
                createGsyncmgr(pSystem, pGsyncmgr);
            }

            return (PODBCOMMON)pGsyncmgr;
            break;

        case ODB_CLASS_SWINSTR:

            if (isEngineMissing(DATA_ID_SWINSTR))
                return NULL;

            if (!pSwinstr)
            {
                pSwinstr = (POBJSWINSTR)unitMalloc(sizeof(OBJSWINSTR), UNIT_CLASS_RM_OBJECT);
                memset(pSwinstr, 0, sizeof(OBJSWINSTR));
                createSwinstr(pSystem, pSwinstr);
            }

            return (PODBCOMMON)pSwinstr;
            break;

        case ODB_CLASS_RCDB:

            if (isEngineMissing(DATA_ID_RCDB))
                return NULL;

            if (!pRcdb)
            {
                pRcdb = (POBJRCDB)unitMalloc(sizeof(OBJRCDB), UNIT_CLASS_RM_OBJECT);
                memset(pRcdb, 0, sizeof(OBJRCDB));
                createRcdb(pSystem, pRcdb);
            }

            return (PODBCOMMON)pRcdb;
            break;

        case ODB_CLASS_UNKNOWN:

            if (odbObjFlags == 0xDEADBEEF && requestedPublicID == 0xDEADBEEF)
            {
                // Assign NULL to all obj pointers
                pDpu = NULL;
                pRes = NULL;
                pPgeng = NULL;
                pPgctrl = NULL;
                pPg = NULL;
                pInforom = NULL;
                pMsenc = NULL;
                pVic = NULL;
                pSpb = NULL;
                pPmu = NULL;
                pCE = NULL;
                pIsohub = NULL;
                pCve = NULL;
                pCipher = NULL;
                pHdmi = NULL;
                pHdcp = NULL;
                pHdtv = NULL;
                pVp = NULL;
                pMp = NULL;
                pMpeg = NULL;
                pBsp = NULL;
                pSmu = NULL;
                pSor = NULL;
                pPior = NULL;
                pOr = NULL;
                pTherm = NULL;
                pVolt = NULL;
                pFuse = NULL;
                pFan = NULL;
                pGpio = NULL;
                pI2c = NULL;
                pGpu = NULL;
                pRc = NULL;
                pVbios = NULL;
                pVga = NULL;
                pMsppp = NULL;
                pSeq = NULL;
                pTmr = NULL;
                pStereo = NULL;
                pPerf = NULL;
                pMc = NULL;
                pIntr = NULL;
                pInst = NULL;
                pHead = NULL;
                pGr = NULL;
                pFlcn = NULL;
                pFifo = NULL;
                pFbsr = NULL;
                pFb = NULL;
                pDplink = NULL;
                pDpaux = NULL;
                pDma = NULL;
                pDisp = NULL;
                pDac = NULL;
                pClk = NULL;
                pBus = NULL;
                pBif = NULL;
                pLwjpg = NULL;
                pOfa = NULL;

                pSystem = NULL;
                pSyscon = NULL;
                pCorelogic = NULL;
                pOs = NULL;
                pPfm = NULL;
                pGpumgr = NULL;
                pGvomgr = NULL;
                pGvimgr = NULL;
                pGsyncmgr = NULL;
                pSwinstr = NULL;
                pRcdb = NULL;

                //make null module descriptor
                pMod = NULL;
            }

            return NULL;
            break;

        default:
            return NULL;
    }
}

/*!
 * @brief redirect the gpu read/write register
 *        operations to infra implementaion
 *
 * @param[in]      pGpu    pointer to GPU object
 */
static void redirectGpuRegisterOperations(POBJGPU pGpu)
{
    //
    // Removed replacing register access functions with unitGpuRead|WriteRegister0##
    // Do we get rid of these functions now, or is somebody still using them?
    //
}
