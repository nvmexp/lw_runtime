/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "maxwell/gm107/dev_fuse.h"
#include "hal.h"

//-----------------------------------------------------
// ceIsValid_GM107
//-----------------------------------------------------
BOOL ceIsValid_GM107( LwU32 indexCe )
{
    return (indexCe <= (LW_FUSE_CTRL_OPT_CE_IDX__SIZE_1 - 1));
}

//-----------------------------------------------------
// ceGetCeFsStatus_GM107
//-----------------------------------------------------
BOOL ceIsPresent_GM107( LwU32 indexCe )
{
    LwU32 fsstatus = GPU_REG_RD32(LW_FUSE_STATUS_OPT_CE);
    return (DRF_IDX_VAL(_FUSE, _STATUS, _OPT_CE_IDX, indexCe, fsstatus) ==
                         LW_FUSE_STATUS_OPT_CE_IDX_ENABLE);
}
