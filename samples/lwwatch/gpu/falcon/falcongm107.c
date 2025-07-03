/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


//
// includes
//

#include "gpuanalyze.h"
#include "maxwell/gm107/dev_falcon_v4.h"

#include "falcon.h"
#include "g_falcon_private.h"     // (rmconfig)  implementation prototypes


//defs from msdecos.h
#define MSDECOS_ERROR_NONE                                         (0x00000000) // default return code for app
#define MSDECOS_ERROR_EXELWTE_INSUFFICIENT_DATA                    (0x00000001) // to be returned by app to OS
#define MSDECOS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                  (0x00000002) // insufficient semaphore methods received
#define MSDECOS_ERROR_ILWALID_METHOD                               (0x00000003) // unsupported method
#define MSDECOS_ERROR_ILWALID_DMA_PAGE                             (0x00000004) // not used lwrrently
#define MSDECOS_ERROR_UNHANDLED_INTERRUPT                          (0x00000005) // either app has no interrupt handler, or an unhandled os error
#define MSDECOS_ERROR_EXCEPTION                                    (0x00000006) // exception raised by falcon
#define MSDECOS_ERROR_ILWALID_CTXSW_REQUEST                        (0x00000007) // invalid ctxsw request to OS
#define MSDECOS_ERROR_APPLICATION                                  (0x00000008) // application returned nonzero error code
#define MSDECOS_ERROR_SWBREAKPT                                    (0x00000009) // exception raised to dump registers in debug mode
#define MSDECOS_INTERRUPT_EXELWTE_AWAKEN                           (0x00000100) // execute awaken enabled
#define MSDECOS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                 (0x00000200) // backend semaphore awaken enabled (os assumes that bck_awkn addr = 2*exe_awkn addr)
#define MSDECOS_INTERRUPT_CTX_ERROR_FBIF                           (0x00000300) // ctx error from fbif
#define MSDECOS_INTERRUPT_LIMIT_VIOLATION                          (0x00000400) // limit violation
#define MSDECOS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                 (0x00000500) // limit violation and fbif ctx error (if both happen together)
#define MSDECOS_INTERRUPT_HALT_ENGINE                              (0x00000600) // wait for dma transfers and halt engine in response to interrupt from RM


//-----------------------------------------------------
//falconTestPC_GM107
// test of falcon pc: check if it is valid, if valid,
// check if it is changing
// @param[in] engineBase - base of engine in register space
// @param[in] engName - name of the engine, e.g. "LWDEC"
// @param[out] LW_OK/LW_ERR_GENERIC on success/failure
//-----------------------------------------------------
LW_STATUS falconTestPC_GM107(LwU32 engineBase, char *engName)
{
    LW_STATUS    status = LW_OK;
    LwU32   pc[3];

    // Read 1st time
    GPU_REG_WR32(engineBase + LW_PFALCON_FALCON_ICD_CMD,
             LW_PFALCON_FALCON_ICD_CMD_IDX_PC << 8 | LW_PFALCON_FALCON_ICD_CMD_OPC_RREG);
    pc[0] = GPU_REG_RD32(engineBase + LW_PFALCON_FALCON_ICD_RDATA);
    // Read 2nd time
    GPU_REG_WR32(engineBase + LW_PFALCON_FALCON_ICD_CMD,
             LW_PFALCON_FALCON_ICD_CMD_IDX_PC << 8 | LW_PFALCON_FALCON_ICD_CMD_OPC_RREG);
    pc[1] = GPU_REG_RD32(engineBase + LW_PFALCON_FALCON_ICD_RDATA);
    // Read 3rd time
    GPU_REG_WR32(engineBase + LW_PFALCON_FALCON_ICD_CMD,
             LW_PFALCON_FALCON_ICD_CMD_IDX_PC << 8 | LW_PFALCON_FALCON_ICD_CMD_OPC_RREG);
    pc[2] = GPU_REG_RD32(engineBase + LW_PFALCON_FALCON_ICD_RDATA);

    // check if stuck
    if ((pc[0] == pc[1]) && (pc[0] == pc[2]))
    {
        dprintf("lw: LW_%s_PRI_PC appears to be stuck at:    0x%04x\n",
                engName, pc[0]);
        addUnitErr("\t LW_%s_PRI_PC appears to be stuck at:    0x%04x\n",
                engName, pc[0]);

        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("lw: LW_%s_PRI_PC is changing. Here's a few iterations...\n", engName);
        dprintf("\t0x%04x\t", pc[0]);
        dprintf("0x%04x\t", pc[1]);
        dprintf("0x%04x\t", pc[2]);
    }

    return status;
}

/*!
 *  Based on _GM107
 *  test of falcon context state: check if it is valid,
 *  fetch chid and see if it is active in host,
 *  check get/put pointers
 *  @param[in] engineBase - base of engine in register space
 *  @param[in] engName - name of the engine, e.g. "PMSPDEC"
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
LW_STATUS falconTestCtxState_GM107(LwU32 engineBase, char* engName)
{    
    PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0();
}
