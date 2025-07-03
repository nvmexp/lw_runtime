/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "psdl.h"
#include "rmpsdl.h"
#include "os.h"

#include "chip.h"
#include "disp.h"
#include "pmu.h"
#include "sig.h"
#include "fb.h"
#include "fifo.h"
#include "inst.h"
#include "clk.h"
#include "smbpbi.h"
#include "acr.h"
#include "falcphys.h"

#include "g_psdl_private.h"          // (rmconfig) hal/obj setup
#include "g_pmu_hal.h"
#include "g_sec2_hal.h"
#include "g_lwdec_hal.h"

#include "maxwell/gm200/dev_pwr_pri.h"
#include "maxwell/gm200/dev_lwdec_pri.h"
#include "maxwell/gm200/dev_falcon_v4.h"
#include "maxwell/gm200/dev_sec_pri.h"
#include "maxwell/gm200/dev_master.h"
#include "maxwell/gm200/dev_fuse.h"
#include "rmpsdl.h"

#include "psdl/bin/gm20x/g_psdluc_sec2_lww_gm20x.h"
#include "psdl/bin/gm20x/g_psdluc_sec2_lww_gm20x_gm204_sig.h"
#include "psdl/bin/gm20x/g_psdluc_lwdec_lww_gm20x.h"
#include "psdl/bin/gm20x/g_psdluc_lwdec_lww_gm20x_gm204_sig.h"
#include "psdl/bin/gm20x/g_psdluc_pmu_lww_gm20x.h"
#include "psdl/bin/gm20x/g_psdluc_pmu_lww_gm20x_gm204_sig.h"

#define PSDL_HALT_TIMEOUT 0x7FFFFFFF

//-----------------------------------------------------
// psdlIsSupported_GM20X
//-----------------------------------------------------
BOOL psdlIsSupported_GM20X()
{
    return TRUE;
}


/*!
 * @brief Patch signatures into ucode image
 *
 * @param[in] pImg        Ucode image
 * @param[in] pProdSig    Production signatures
 * @param[in] pDbgSig     Debug signatures
 * @param[in] pPatchLoc   Patch locations
 * @param[in] pPatchInd   Signature indexes to patch
 */
static LW_STATUS
_psdlUcodePatchSignature
(
    LwU32    engineBase,
    LwU32    *pImg,
    LwU32    *pProdSig,
    LwU32    *pDbgSig,
    LwU32    *pPatchLoc,
    LwU32    *pPatchInd,
    LwBool   bIsDebugMode
)
{
    LwU32     i;
    LwU32     *pSig;

    if (!bIsDebugMode)
    {
        pSig = pProdSig;
    }
    else
    {
        pSig = pDbgSig;
    }

    // Patching logic:
    for (i=0; i < sizeof(*pPatchLoc)>>2 ; i++)
    {
        pImg[(pPatchLoc[i]>>2)]   = pSig[(pPatchInd[i]<<2)] ;
        pImg[(pPatchLoc[i]>>2)+1] = pSig[(pPatchInd[i]<<2)+1] ;
        pImg[(pPatchLoc[i]>>2)+2] = pSig[(pPatchInd[i]<<2)+2] ;
        pImg[(pPatchLoc[i]>>2)+3] = pSig[(pPatchInd[i]<<2)+3] ;
    }
    return LW_OK ;
}

//-----------------------------------------------------
// Function to parse the return code from PSDL ucode
//-----------------------------------------------------
static
LW_STATUS _psdlParseReturnCode( LwU32 returnCode)
{
    LW_STATUS rc = LW_ERR_GENERIC;

    switch (returnCode)
    {
        case RM_PSDL_RC_OK:
            rc = LW_OK;
            dprintf("lw:\tPSDL -> PASS\n");
            break;
        case RM_PSDL_RC_ERROR_NO_SIG_CHECK:
            dprintf("lw:\tPSDL -> Ucode Sig check FAILED\n");
            break;
        case RM_PSDL_RC_CERT_EXT_NOT_FOUND:
            dprintf("lw:\tPSDL -> No PSDL extension (8) found - FAIL\n");
            break;
        case RM_PSDL_RC_KA_EXT_NOT_FOUND:
            dprintf("lw:\tPSDL -> No KA extension (9) found - FAIL\n");
            break;
        case RM_PSDL_RC_CERT_NO_EXT_FLAG:
            dprintf("lw:\tPSDL -> No extension found - FAILED\n");
            break;
        case RM_PSDL_RC_CERT_SIG_FAIL:
            dprintf("lw:\tPSDL -> License certificate verification FAILED\n");
            break;
        case RM_PSDL_RC_PSDL_ILWALID_HEADER:
            dprintf("lw:\tPSDL -> Invalid PSDL header - FAIL\n");
            break;
        case RM_PSDL_RC_PSDL_ECID_NOT_FOUND:
            dprintf("lw:\tPSDL -> ECID Not found - FAIL\n");
            break;
        case RM_PSDL_RC_PSDL_NO_PRIV:
            dprintf("lw:\tPSDL -> No PRIV registers specified - FAIL\n");
            break;
        default:
            dprintf("lw:\tPSDL -> Unknown error 0x%0x - FAIL\n", returnCode);
            break;
    }
    return rc;

}

//-----------------------------------------------------
// A common function to help with PSDL
//-----------------------------------------------------
static LW_STATUS
_psdlExelwte( LwU32 engineBase, LwU8 *pCert, LwU32 certSizeB, psdl_engine_config *pConfig, LwBool bIsDebugMode)
{
    LwU32 imemPA         = 0;
    LwU32 dmemPA         = 0;
    LwU32 inselwreOffset = pConfig->pUcodeHeader[0];
    LwU32 *pImemIS       = &(pConfig->pUcodeData[(inselwreOffset / 4)]);
    LwU32 selwreOffset   = pConfig->pUcodeHeader[5];
    LwU32 *pImemHS       = &(pConfig->pUcodeData[(selwreOffset / 4)]);
    LwU32 dataOffset     = pConfig->pUcodeHeader[2];
    LwU32 *pDmem         = &(pConfig->pUcodeData[(dataOffset / 4)]);
    LwU32 inselwreSizeB  = pConfig->pUcodeHeader[1];
    LwU32 selwreSizeB    = pConfig->pUcodeHeader[6];
    LwU32 dataSizeB      = pConfig->pUcodeHeader[3];
    LwU32 port           = 0;
    LwU32 bytesWritten   = 0;


    // Patch Ucode first
    _psdlUcodePatchSignature(engineBase, pConfig->pUcodeData, pConfig->pSigProd,
                       pConfig->pSigDbg, pConfig->pSigPatchLoc, pConfig->pSigPatchSig, bIsDebugMode);

    dprintf("lw:\tPSDL -> Loading Ucode ....\n");

    // Write inselwre IMEM first
    if ((bytesWritten = pConfig->pFCIF->flcnImemWriteBuf(engineBase, imemPA, inselwreOffset,
                     pImemIS, inselwreSizeB, port, LW_FALSE)) != inselwreSizeB)
    {
        dprintf("lw:\tPSDL -> Ucode loading FAILED - Bytes written 0x%0x\n",
                                 bytesWritten);
        return LW_ERR_GENERIC;
    }

    // Write secure IMEM
    imemPA += inselwreSizeB;
    if ((bytesWritten = pConfig->pFCIF->flcnImemWriteBuf(engineBase, imemPA , selwreOffset,
                            pImemHS, selwreSizeB, port, LW_TRUE)) != selwreSizeB)
    {
        dprintf("lw:\tPSDL -> Ucode loading FAILED - Bytes written 0x%0x\n",
                                 bytesWritten);
        return LW_ERR_GENERIC;
    }


    // Write DMEM
    if ((bytesWritten = pConfig->pFCIF->flcnDmemWriteBuf(engineBase, dmemPA,
                     pDmem, dataSizeB, port)) != dataSizeB)
    {
        dprintf("lw:\tPSDL -> Ucode Data loading FAILED - Bytes written 0x%0x\n",
                                 bytesWritten);
        return LW_ERR_GENERIC;
    }

    dprintf("lw:\tPSDL -> Loading cert ....\n");
    // Write CERT
    dmemPA = dmemPA + dataSizeB;
    if ((bytesWritten = pConfig->pFCIF->flcnDmemWriteBuf(engineBase, dmemPA,
                     (LwU32*)pCert, certSizeB, port)) != certSizeB)
    {
        dprintf("lw:\tPSDL -> CERT loading FAILED - Bytes written 0x%0x\n",
                                 bytesWritten);
        return LW_ERR_GENERIC;
    }

    // Write a known patter into MAILBOX0
    FLCN_REG_WR32(LW_PFALCON_FALCON_MAILBOX0, 0xDEADBEEF);

    dprintf("lw:\tPSDL -> Starting the falcon ....\n");

    // Bootstrap FALCON
    pConfig->pFCIF->flcnBootstrap(engineBase, 0x0);

    // Wait for falcon to HALT
    if (pConfig->pFCIF->flcnWaitForHalt(engineBase, PSDL_HALT_TIMEOUT) != LW_OK)
    {
        dprintf("lw:\tPSDL -> Error while waiting for falcon to Halt\n");
        return LW_ERR_GENERIC;
    }

    //Read mailbox now and verify it
    return _psdlParseReturnCode(FLCN_REG_RD32(LW_PFALCON_FALCON_MAILBOX0));
}

//-----------------------------------------------------
// psdlGetSec2Config_GM204
//-----------------------------------------------------
LW_STATUS psdlGetSec2Config_GM204( LwU32 indexGpu, void *pvConfig)
{
    psdl_engine_config *pConfig = (psdl_engine_config *)pvConfig;
    pConfig->pFCIF        = (const FLCN_CORE_IFACES *) &(flcnCoreIfaces_v05_01);
    pConfig->pSigDbg      = psdl_sec2_sig_dbg;
    pConfig->pSigProd     = psdl_sec2_sig_prod;
    pConfig->pSigPatchLoc = psdl_sec2_sig_patch_location;
    pConfig->pSigPatchSig = psdl_sec2_sig_patch_signature;

    return LW_OK;
}

//-----------------------------------------------------
// psdlGetPmuConfig_GM204
//-----------------------------------------------------
LW_STATUS psdlGetPmuConfig_GM204( LwU32 indexGpu, void *pvConfig)
{
    psdl_engine_config *pConfig = (psdl_engine_config *)pvConfig;
    pConfig->pFCIF        = (const FLCN_CORE_IFACES *) pPmu[indexGpu].pmuGetFalconCoreIFace();
    pConfig->pFEIF        = (const FLCN_ENGINE_IFACES *) pPmu[indexGpu].pmuGetFalconEngineIFace();
    pConfig->pSigDbg      = psdl_pmu_lww_sig_dbg;
    pConfig->pSigProd     = psdl_pmu_lww_sig_prod;
    pConfig->pSigPatchLoc = psdl_pmu_lww_sig_patch_location;
    pConfig->pSigPatchSig = psdl_pmu_lww_sig_patch_signature;

    return LW_OK;
}

//-----------------------------------------------------
// psdlGetPmuConfig_GM204
//-----------------------------------------------------
LW_STATUS psdlGetLwdecConfig_GM204( LwU32 indexGpu, void *pvConfig)
{
    psdl_engine_config *pConfig = (psdl_engine_config *)pvConfig;
    pConfig->pFCIF        = (const FLCN_CORE_IFACES *) &(flcnCoreIfaces_v05_01);
    pConfig->pSigDbg      = psdl_lwdec_sig_dbg;
    pConfig->pSigProd     = psdl_lwdec_sig_prod;
    pConfig->pSigPatchLoc = psdl_lwdec_sig_patch_location;
    pConfig->pSigPatchSig = psdl_lwdec_sig_patch_signature;

    return LW_OK;
}

//-----------------------------------------------------
// psdlUseSec2_GM20X
//-----------------------------------------------------
LW_STATUS psdlUseSec2_GM20X( LwU32 indexGpu, LwU8 *pCert, LwU32 size )
{
    LwU32              engineBase = DEVICE_BASE(LW_PSEC);
    LwBool             bIsDebugMode = pSec2[indexGpu].sec2IsDebugMode();
    psdl_engine_config peconfig;

    dprintf("lw:\tPSDL -> Resetting SEC2 .....\n");

    //
    // Reset SEC2 first
    //
    if (pSec2[indexGpu].sec2MasterReset() != LW_OK)
    {
        dprintf("lw:\tPSDL -> Reset SEC2 FAILED\n");
    }

    // Populate the config
    pPsdl[indexGpu].psdlGetSec2Config(indexGpu, &peconfig);
    peconfig.pUcodeData   = psdl_ucode_data_sec2_lww_gm20x;
    peconfig.pUcodeHeader = psdl_ucode_header_sec2_lww_gm20x;
    peconfig.ucodeSize    = psdl_ucode_data_size_sec2_lww_gm20x;

    _psdlExelwte(engineBase, pCert, size, &peconfig, bIsDebugMode);

    return LW_OK;
}

//-----------------------------------------------------
// psdlUsePmu_GM20X
// Use PMU to verify and process PSDL certificate
//-----------------------------------------------------
LW_STATUS psdlUsePmu_GM20X( LwU32 indexGpu, LwU8 *pCert, LwU32 size )
{
    LwU32              engineBase   = 0;
    LwBool             bIsDebugMode = pPmu[indexGpu].pmuIsDebugMode();
    psdl_engine_config peconfig;

    dprintf("lw:\tPSDL -> Resetting PMU .....\n");

    // Reset PMU first
    if (pPmu[indexGpu].pmuMasterReset() != LW_OK)
    {
        dprintf("lw:\tPSDL -> Reset PMU FAILED\n");
    }

    // Populate the config
    pPsdl[indexGpu].psdlGetPmuConfig(indexGpu, &peconfig);
    peconfig.pUcodeData   = psdl_ucode_data_pmu_lww_gm20x;
    peconfig.pUcodeHeader = psdl_ucode_header_pmu_lww_gm20x;
    peconfig.ucodeSize    = psdl_ucode_data_size_pmu_lww_gm20x;

    engineBase = peconfig.pFEIF->flcnEngGetFalconBase();
    _psdlExelwte(engineBase, pCert, size, &peconfig, bIsDebugMode);

    return LW_OK;
}

//-----------------------------------------------------
// psdlUseLwdec_GM20X
//-----------------------------------------------------
LW_STATUS psdlUseLwdec_GM20X( LwU32 indexGpu, LwU8 *pCert, LwU32 size )
{
    LwU32              engineBase   = DEVICE_BASE(LW_PLWDEC);
    LwBool             bIsDebugMode = pLwdec[indexGpu].lwdecIsDebugMode(0);
    psdl_engine_config peconfig;

    dprintf("lw:\tPSDL -> Resetting LWDEC .....\n");

    //
    // Reset LWDEC first
    //
    if (pLwdec[indexGpu].lwdecMasterReset(0) != LW_OK)
    {
        dprintf("lw:\tPSDL -> Reset LWDEC FAILED\n");
    }

    // Populate the config
    pPsdl[indexGpu].psdlGetLwdecConfig(indexGpu, &peconfig);
    peconfig.pUcodeData   = psdl_ucode_data_lwdec_lww_gm20x;
    peconfig.pUcodeHeader = psdl_ucode_header_lwdec_lww_gm20x;
    peconfig.ucodeSize    = psdl_ucode_data_size_lwdec_lww_gm20x;

    _psdlExelwte(engineBase, pCert, size, &peconfig, bIsDebugMode);

    return LW_OK;
}

//-----------------------------------------------------
// psdlPrintEcid
//-----------------------------------------------------
LW_STATUS psdlPrintEcid_GM20X(LwU32 indexGpu)
{
    LwU64   serialNum[2];
    LwU32   offset     = 0;
    LwU32   lotCode0   = 0;
    LwU32   lotCode1   = 0;
    LwU32   fabCode    = 0;
    LwU32   xCoord     = 0;
    LwU32   yCoord     = 0;
    LwU32   waferId    = 0;
    LwU32   vendorCode = 0;
    LwS32   si         = 0;


    serialNum[0] = serialNum[1] = 0;

    lotCode0   = DRF_VAL(_FUSE, _OPT_LOT_CODE_0, _DATA, GPU_REG_RD32(LW_FUSE_OPT_LOT_CODE_0));
    lotCode1   = DRF_VAL(_FUSE, _OPT_LOT_CODE_1, _DATA, GPU_REG_RD32(LW_FUSE_OPT_LOT_CODE_1));
    fabCode    = DRF_VAL(_FUSE, _OPT_FAB_CODE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FAB_CODE));
    xCoord     = DRF_VAL(_FUSE, _OPT_X_COORDINATE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_X_COORDINATE));
    yCoord     = DRF_VAL(_FUSE, _OPT_Y_COORDINATE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_Y_COORDINATE));
    waferId    = DRF_VAL(_FUSE, _OPT_WAFER_ID, _DATA, GPU_REG_RD32(LW_FUSE_OPT_WAFER_ID));
    vendorCode = DRF_VAL(_FUSE, _OPT_VENDOR_CODE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_VENDOR_CODE));

    if ((!lotCode0) && (!lotCode1) && (!fabCode) && (!xCoord) && (!yCoord) && (!waferId) && (!vendorCode))
    {
        //
        // We are running either in fmodel or emulation. Assign known values
        // Final hex string: 7205bc37 afae36bf 83962205 00000000
        //
        lotCode0   = 0xABCDEF01;
        lotCode1   = 0xEFCDAB;
        fabCode    = 0x17;
        xCoord     = 0x29;
        yCoord     = 0x29;
        waferId    = 0x1A;
        vendorCode = 0x2;
        dprintf("lw:\t*************** Note: Using FAKE ECID in simulation *****************\n");
    }
    else
    {
        dprintf("lw:\t                              ECID                                    \n");
    }
    dprintf("lw:\t======================================================================\n\n");
    dprintf("lw:\t\t\tVendor Code   :\t0x%08x\n", vendorCode);
    dprintf("lw:\t\t\tFab Code      :\t0x%08x\n", fabCode);
    dprintf("lw:\t\t\tLot Code0     :\t0x%08x\n", lotCode0);
    dprintf("lw:\t\t\tLot Code1     :\t0x%08x\n", lotCode1);
    dprintf("lw:\t\t\tWafer ID      :\t0x%08x\n", waferId);
    dprintf("lw:\t\t\tX coordinate  :\t0x%08x\n", xCoord);
    dprintf("lw:\t\t\tY coordinate  :\t0x%08x\n", yCoord);

    serialNum[0] = (LwU64) (vendorCode);
    offset = DRF_SIZE(LW_FUSE_OPT_VENDOR_CODE_DATA);        // +4 = 4

    serialNum[0] |= (LwU64) (fabCode) << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_FAB_CODE_DATA);          // +6 = 10

    serialNum[0] |= (LwU64) (lotCode0) << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_LOT_CODE_0_DATA);        // +32 = 42

    serialNum[0] |= (LwU64) (lotCode1) << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_LOT_CODE_1_DATA);        // +28 = 70!
    offset -= 64;                                           // 6

    serialNum[1] = (LwU64) (lotCode1) >> (DRF_SIZE(LW_FUSE_OPT_LOT_CODE_1_DATA) - offset);

    dprintf("lw:\t\t\tECID Pattern  :\t");
    for (si = 15; si >= 0; --si)
    {
        dprintf("%02x", ((LwU8*)serialNum)[si]);
    }
    dprintf("\n");

    serialNum[1] |= (LwU64) (waferId) << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_WAFER_ID_DATA);          // +6 = 12

    serialNum[1] |= (LwU64) (xCoord) << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_X_COORDINATE_DATA);      // +9 = 21

    serialNum[1] |= (LwU64) (yCoord) << offset;
    offset += DRF_SIZE(LW_FUSE_OPT_Y_COORDINATE_DATA);      // +9 = 30

    offset = LW_ALIGN_UP((offset + 64), 8); //make it byte aligned

    dprintf("lw:\t\t\tFull ECID     :\t");
    for (si = 15; si >= 0; --si)
    {
        dprintf("%02x", ((LwU8*)serialNum)[si]);
    }
    dprintf("\n");

    return LW_OK;
}
