/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch PMU helper.  
// pmugm107.c
//
//*****************************************************

//
// includes
//
#include "pmu.h"
#include "maxwell/gm107/dev_pwr_pri.h"
#include "maxwell/gm107/dev_master.h"
#include "fuseUcode/gm107/fuseappcode_GM107.h"

extern LwU32 fuse_ucode_data_GM107[];
extern LwU32 fuse_ucode_header_GM107[];
#define FUSE_UCODE_HEADER_OS_CODE_OFFSET      0
#define FUSE_UCODE_HEADER_OS_CODE_SIZE        1
#define FUSE_UCODE_HEADER_APP_CODE_OFFSET     5
#define FUSE_UCODE_HEADER_APP_CODE_SIZE       6

#define FUSE_UCODE_HEADER_OS_DATA_OFFSET      2
#define FUSE_UCODE_HEADER_OS_DATA_SIZE        3
#define FUSE_UCODE_HEADER_APP_DATA_OFFSET     7
#define FUSE_UCODE_HEADER_APP_DATA_SIZE       8

#define FUSE_CODE_SELWRE                      1
#define PMU_RESET_TIMEOUT                     0x1000 //us
#define PMU_HALT_TIMEOUT                      0x4000 //us

#define FUSE_DUMMY_VALUE                      0x12345678

//MAILBOX1 codes
#define FUSE_BINARY_SELWRE_STARTING 0x1234AAAA
#define FUSE_BINARY_STARTING        0x5678BBBB
#define FUSE_BINARY_ENDING          0x5678CCCC

//MAILBOX0 codes
#define FUSE_LEVEL_SUCCESS          0xCAFE10DE 
#define FUSE_VERSION_ERROR          0xBAD10000
#define FUSE_CHIP_MISMATCH          0xBAD20000
#define FUSE_UNKNOWN_ERROR          0xBAD30000


static LwU32
pmuCallwlateImemSizeReq(LwU32 *pUcodeHeader)
{
    LwU32 codeSize = 0;
    LwU32 index    = 0;

    // pUcodeHeader[1] is the OS size
    codeSize += pUcodeHeader[1];
    
    // Loop through app
    for (index = 0; index < pUcodeHeader[4]; index++)
    {
        codeSize += (pUcodeHeader[4 + (index*4) + 2]);
    }
    
    return codeSize;
}

static LW_STATUS
pmuLoadCode(LwU32 verbose, LwU32 *pFreeBlk, LwU32 *pOriUCode, LwU32 offsetInBytes, LwU32 sizeInBytes, LwBool bIsSelwre)
{
    LwU32 iport      = 0;
    LwU32 imemcOrig  = 0;
    LwU32 data       = 0;
    LwU32 index      = 0;
    LwU32 *pUcode    = &(pOriUCode[offsetInBytes/4]);
    LwU32 tag        = offsetInBytes >> 8;

    // offset and size needs to be 256B aligned
    if ((offsetInBytes & 0xFF) || (sizeInBytes & 0xFF))
    {
        PMU_LOG(VB0, "FUSE: ERROR - Offset/Size not in 256B alignment\n");
        return LW_ERR_GENERIC;
    }

    imemcOrig = GPU_REG_RD32(LW_PPWR_FALCON_IMEMC(iport));

    data = FLD_SET_DRF_NUM(_PPWR_FALCON, _IMEMC, _OFFS, 0, data);
    data = FLD_SET_DRF_NUM(_PPWR_FALCON, _IMEMC, _BLK, (*pFreeBlk), data);
    data = FLD_SET_DRF(_PPWR_FALCON, _IMEMC, _AINCW, _TRUE, data);

    if (bIsSelwre == LW_TRUE)
    {
        data = FLD_SET_DRF_NUM(_PPWR_FALCON, _IMEMC, _SELWRE, 0x1, data);
    }

    GPU_REG_WR32(LW_PPWR_FALCON_IMEMC(iport), data);

    for (index = 0; index < (sizeInBytes/4); index++)
    {
        if ((index % 64) == 0)
        {
            GPU_REG_WR32(LW_PPWR_FALCON_IMEMT(iport), FLD_SET_DRF_NUM(_PPWR_FALCON, _IMEMT, _TAG, tag, 0));
            tag++;
        }
        GPU_REG_WR32(LW_PPWR_FALCON_IMEMD(iport), FLD_SET_DRF_NUM(_PPWR_FALCON, _IMEMD, _DATA, pUcode[index], 0));
    }

    *pFreeBlk += ((sizeInBytes/256) + ((sizeInBytes % 256)?1:0));

    GPU_REG_WR32(LW_PPWR_FALCON_IMEMC(iport), imemcOrig);

    return LW_OK;
}

static LW_STATUS
pmuLoadData(LwU32 verbose, LwU32 *pFreeBlk, LwU32 dataImg[], LwU32 offsetInBytes, LwU32 sizeInBytes)
{
    LwU32 dport      = 0;
    LwU32 dmemcOrig  = 0;
    LwU32 data       = 0;
    LwU32 index      = 0;
    LwU32 *pData     = &(dataImg[offsetInBytes/4]);

    // offset and size needs to be 256B aligned
    if ((offsetInBytes & 0xFF) || (sizeInBytes & 0xFF))
    {
        PMU_LOG(VB0, "FUSE: ERROR - Offset/Size not in 256B alignment\n");
        return LW_ERR_GENERIC;
    }

    dmemcOrig = GPU_REG_RD32(LW_PPWR_FALCON_DMEMC(dport));

    data = FLD_SET_DRF_NUM(_PPWR_FALCON, _DMEMC, _OFFS, 0, data);
    data = FLD_SET_DRF_NUM(_PPWR_FALCON, _DMEMC, _BLK, (*pFreeBlk), data);
    data = FLD_SET_DRF(_PPWR_FALCON, _DMEMC, _AINCW, _TRUE, data);

    GPU_REG_WR32(LW_PPWR_FALCON_DMEMC(dport), data);


    for (index = 0; index < (sizeInBytes/4); index++)
    {
        GPU_REG_WR32(LW_PPWR_FALCON_DMEMD(dport), FLD_SET_DRF_NUM(_PPWR_FALCON, _DMEMD, _DATA, pData[index], 0));
    }

    *pFreeBlk = (sizeInBytes/256) + ((sizeInBytes % 256)?1:0);
    GPU_REG_WR32(LW_PPWR_FALCON_DMEMC(dport), dmemcOrig);

    return LW_OK;
}

LW_STATUS
pmuGetFuseBinary_GM107
(
    PmuFuseBinaryDesc *pDesc
)
{
    pDesc->pUcodeData   = fuse_ucode_data_GM107;
    pDesc->pUcodeHeader = fuse_ucode_header_GM107;
    return LW_OK;
}    

/*!
 *  Reset the PMU 
 */
LW_STATUS pmuMasterReset_GM107()
{   
    LwU32 reg32;
    LwS32 timeoutUs = PMU_RESET_TIMEOUT;

    reg32 = GPU_REG_RD32(LW_PMC_ENABLE);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PWR, _DISABLED, reg32);
    GPU_REG_WR32(LW_PMC_ENABLE, reg32);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PWR, _ENABLED, reg32);
    GPU_REG_WR32(LW_PMC_ENABLE, reg32);

    // Wait for SCRUBBING to complete
    while (timeoutUs > 0)
    {
        reg32 = GPU_REG_RD32(LW_PPWR_FALCON_DMACTL);

        if (FLD_TEST_DRF(_PPWR, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, reg32) &&
            FLD_TEST_DRF(_PPWR, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, reg32))
        {
            break;
        }
        osPerfDelay(20);
        timeoutUs -= 20;
    }

    if (timeoutUs <= 0)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}


/*!
 * @brief Checks if PMU DEBUG fuse is blown or not
 *
 * @param[in] pGpu    OBJGPU pointer
 * @param[in] pPmu    OBJPMU pointer
 */
LwBool
pmuIsDebugMode_GM107()
{
    LwU32 ctlStat =  GPU_REG_RD32(LW_PPWR_PMU_SCP_CTL_STAT);

    return !FLD_TEST_DRF(_PPWR_PMU, _SCP_CTL_STAT, _DEBUG_MODE, _DISABLED, ctlStat);
}


LW_STATUS
pmuVerifyFuse_GM107(LwU32 indexGpu)
{
    LwU32             blk         = 0;
    LwU32             dmaCtrl     = 0;
    LwU32             imemSize    = pPmu[indexGpu].pmuImemGetSize();
    LwS32             timeoutUs   = PMU_RESET_TIMEOUT;
    LwU32             verbose     = VB1;
    PmuFuseBinaryDesc ucodeDesc;

    if (!(pPmu[indexGpu].pmuIsDebugMode()))
    {
        PMU_LOG(VB0, "FUSE: ERROR - FUSE binary cannot be verified on PROD chips\n");
        return LW_ERR_GENERIC; 
    }

    // Lets get the data first
    pPmu[indexGpu].pmuGetFuseBinary(&ucodeDesc);

    if (imemSize < pmuCallwlateImemSizeReq(ucodeDesc.pUcodeHeader))
    {
        PMU_LOG(VB0, "FUSE: ERROR - IMEM Size less than Ucode size\n");
        return LW_ERR_GENERIC;
    }

    // Reset the PMU first
    if (pPmu[indexGpu].pmuMasterReset() != LW_OK)
    {
        PMU_LOG(VB0, "FUSE: Resetting PMU FAILED. Could be scrubbing timeout.\n"); 
        return LW_ERR_GENERIC;
    }

    //Set REQUIRE_CTX to FALSE
    dmaCtrl = GPU_REG_RD32(LW_PPWR_FALCON_DMACTL);
    GPU_REG_WR32(LW_PPWR_FALCON_DMACTL, FLD_SET_DRF(_PPWR_FALCON, _DMACTL, _REQUIRE_CTX, _FALSE, dmaCtrl));

    // Load data into DMEM
    pmuLoadData(verbose, &blk, ucodeDesc.pUcodeData, ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_OS_DATA_OFFSET], 
                                           ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_OS_DATA_SIZE]);
    pmuLoadData(verbose, &blk, ucodeDesc.pUcodeData, ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_APP_DATA_OFFSET], 
                                           ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_APP_DATA_SIZE]);

    // Load code into IMEM
    blk = 0;
    pmuLoadCode(verbose, &blk, ucodeDesc.pUcodeData, ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_OS_CODE_OFFSET], 
                              ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_OS_CODE_SIZE], !FUSE_CODE_SELWRE);
    pmuLoadCode(verbose, &blk, ucodeDesc.pUcodeData, ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_APP_CODE_OFFSET], 
                              ucodeDesc.pUcodeHeader[FUSE_UCODE_HEADER_APP_CODE_SIZE], FUSE_CODE_SELWRE);

    // Set BOOTVEC to 0
    GPU_REG_WR32(LW_PPWR_FALCON_BOOTVEC, 0);

    // Start the PMU now
    PMU_LOG(VB0, "FUSE: Starting PMU\n");
    GPU_REG_WR32(LW_PPWR_FALCON_MAILBOX0, FUSE_DUMMY_VALUE);
    GPU_REG_WR32(LW_PPWR_FALCON_CPUCTL, FLD_SET_DRF(_PPWR_FALCON, _CPUCTL, _STARTCPU, _TRUE, 0));

    PMU_LOG(VB0, "FUSE: Waiting for PMU to halt\n");
    timeoutUs = PMU_HALT_TIMEOUT;
    while (timeoutUs > 0)
    {
        dmaCtrl = GPU_REG_RD32(LW_PPWR_FALCON_CPUCTL);
        if (FLD_TEST_DRF(_PPWR, _FALCON_CPUCTL, _HALTED, _TRUE, dmaCtrl)) 
            break;
        osPerfDelay(20);
        timeoutUs -= 20;
    }

    if (timeoutUs <= 0)
    {
        PMU_LOG(VB0, "FUSE: ERROR - PMU not halting!\n");
        return LW_ERR_GENERIC;
    }

    dmaCtrl = GPU_REG_RD32(LW_PPWR_FALCON_MAILBOX0);
    if (dmaCtrl != FUSE_LEVEL_SUCCESS)
    {
        PMU_LOG(VB0, "FUSE: ERROR - PMU FUSE verification failed - MAILBOX0 returns [0x%08x] !\n", dmaCtrl);
        return LW_ERR_GENERIC;
    }

    PMU_LOG(VB0, "FUSE: PMU FUSE verification SUCCESS - MAILBOX0 returns [0x%08x] !\n", dmaCtrl); 
    return LW_OK;
}

const char *
pmuUcodeName_GM107()
{
    return "g_c85b6_gm10x";
}
