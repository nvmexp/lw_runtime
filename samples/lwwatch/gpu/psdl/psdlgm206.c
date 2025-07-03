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

#include "maxwell/gm200/dev_pwr_pri.h"
#include "maxwell/gm200/dev_lwdec_pri.h"
#include "maxwell/gm200/dev_falcon_v4.h"
#include "maxwell/gm200/dev_sec_pri.h"
#include "maxwell/gm200/dev_master.h"
#include "maxwell/gm200/dev_fuse.h"
#include "rmpsdl.h"

#include "psdl/bin/gm20x/g_psdluc_sec2_lww_gm20x_gm206_sig.h"
#include "psdl/bin/gm20x/g_psdluc_lwdec_lww_gm20x_gm206_sig.h"
#include "psdl/bin/gm20x/g_psdluc_pmu_lww_gm20x_gm206_sig.h"


//-----------------------------------------------------
// psdlGetSec2Config_GM206
//-----------------------------------------------------
LW_STATUS psdlGetSec2Config_GM206( LwU32 indexGpu, void *pvConfig)
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
// psdlGetPmuConfig_GM206
//-----------------------------------------------------
LW_STATUS psdlGetPmuConfig_GM206( LwU32 indexGpu, void *pvConfig)
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
// psdlGetPmuConfig_GM206
//-----------------------------------------------------
LW_STATUS psdlGetLwdecConfig_GM206( LwU32 indexGpu, void *pvConfig)
{
    psdl_engine_config *pConfig = (psdl_engine_config *)pvConfig;
    pConfig->pFCIF        = (const FLCN_CORE_IFACES *) &(flcnCoreIfaces_v05_01);
    pConfig->pSigDbg      = psdl_lwdec_sig_dbg;
    pConfig->pSigProd     = psdl_lwdec_sig_prod;
    pConfig->pSigPatchLoc = psdl_lwdec_sig_patch_location;
    pConfig->pSigPatchSig = psdl_lwdec_sig_patch_signature;

    return LW_OK;
}
