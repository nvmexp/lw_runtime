/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
#include "smbpbi.h"
#include "kepler/gk104/dev_therm.h"

#include "g_smbpbi_private.h"          // (rmconfig) implementation prototypes

LW_STATUS
smbpbiGetContext_GK104
(
    SMBPBI_CONTEXT *pContext
)
{
    pContext->cmd     = GPU_REG_RD32(LW_THERM_MSGBOX_COMMAND);
    pContext->dataIn  = GPU_REG_RD32(LW_THERM_MSGBOX_DATA_IN);
    pContext->dataOut = GPU_REG_RD32(LW_THERM_MSGBOX_DATA_OUT);
    return LW_OK;
}

LW_STATUS
smbpbiSetContext_GK104
(
    SMBPBI_CONTEXT *pContext
)
{
    // write data-in and data-out before writing the command
    GPU_REG_WR32(LW_THERM_MSGBOX_DATA_IN , pContext->dataIn);
    GPU_REG_WR32(LW_THERM_MSGBOX_DATA_OUT, pContext->dataOut);
    GPU_REG_WR32(LW_THERM_MSGBOX_COMMAND , pContext->cmd);
    return LW_OK;
}

