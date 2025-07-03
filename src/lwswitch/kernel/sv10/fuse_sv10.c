/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "common_lwswitch.h"
#include "sv10/sv10.h"

#include "lwswitch/svnp01/dev_fuse.h"

/*
 * Read a LW_FUSE_OPT_* register
 */

static LwU32
_lwswitch_fuse_read
(
    lwswitch_device *device,
    LwU32   fuse_reg
)
{
    LwU32   fuse_data = 0x0;

    fuse_data = LWSWITCH_OFF_RD32(device, fuse_reg);

    LWSWITCH_PRINT(device, INFO,
        "%s: Fuse 0x%x = 0x%x\n",
        __FUNCTION__, fuse_reg, fuse_data);

    return fuse_data;
}

#define FUSE_OPT_READ(_d, _fuse, _data)                             \
    case LW_FUSE_OPT ## _fuse:                                      \
    _data = DRF_VAL(_FUSE_OPT, _fuse, _DATA, _lwswitch_fuse_read(_d, LW_FUSE_OPT ## _fuse));  \
        LWSWITCH_PRINT(_d, INFO,                                    \
            "%s: %s (%x)=%x\n",                                     \
            __FUNCTION__,                                           \
            #_fuse, LW_FUSE_OPT ## _fuse, _data);                   \
    break;


LwU32
lwswitch_fuse_opt_read_sv10
(
    lwswitch_device *device,
    LwU32   fuse_opt
)
{
    LwU32   fuse_data = 0x0;

    switch(fuse_opt)
    {
        // Fuse definitions lwrrently used by driver
        FUSE_OPT_READ(device, _SELWRE_PJTAG_ACCESS_WR_SELWRE, fuse_data)
        FUSE_OPT_READ(device, _VENDOR_CODE, fuse_data)
        FUSE_OPT_READ(device, _FAB_CODE, fuse_data)
        FUSE_OPT_READ(device, _LOT_CODE_0, fuse_data)
        FUSE_OPT_READ(device, _LOT_CODE_1, fuse_data)
        FUSE_OPT_READ(device, _WAFER_ID, fuse_data)
        FUSE_OPT_READ(device, _X_COORDINATE, fuse_data)
        FUSE_OPT_READ(device, _Y_COORDINATE, fuse_data)
        FUSE_OPT_READ(device, _OPS_RESERVED, fuse_data)
        FUSE_OPT_READ(device, _IDDQ, fuse_data)
        FUSE_OPT_READ(device, _IDDQ_REV, fuse_data)
        FUSE_OPT_READ(device, _SPEEDO_REV, fuse_data)
        FUSE_OPT_READ(device, _SPEEDO0, fuse_data)
        FUSE_OPT_READ(device, _SPEEDO1, fuse_data)
        FUSE_OPT_READ(device, _SPEEDO2, fuse_data)
        FUSE_OPT_READ(device, _LWST_ATE_REV, fuse_data)
        FUSE_OPT_READ(device, _RAM_SVOP_PDP, fuse_data)         // _TDIODE_CENTER
        FUSE_OPT_READ(device, _PFG, fuse_data)                  // _TDIODE_WEST
        FUSE_OPT_READ(device, _SPARE_FS, fuse_data)             // _TDIODE_EAST

        // Fuse definitions potentially used by driver
        FUSE_OPT_READ(device, _MASK_REVISION_ID, fuse_data)
        FUSE_OPT_READ(device, _MINOR_EXT_REVISION_ID, fuse_data)
        FUSE_OPT_READ(device, _FAB_ID, fuse_data)
        FUSE_OPT_READ(device, _INTERNAL_SKU, fuse_data)
        FUSE_OPT_READ(device, _PRIV_SEC_EN, fuse_data)
        FUSE_OPT_READ(device, _MAJOR_REVISION_ID, fuse_data)
        FUSE_OPT_READ(device, _MINOR_REVISION_ID, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_DECODE_TRAP_WR_SELWRE, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_RING_PRIV_LEVEL_WR_SELWRE, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_FUSE_SPEEDOINFO_RD_SELWRE, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_JTAG_SELWREID_VALID, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_PMGR_GPIO_OUT_WR_SELWRE, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_PMGR_GPIO_INPUT_CNTL_WR_SELWRE, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_MINION_DEBUG_DIS, fuse_data)
        FUSE_OPT_READ(device, _FUSE_UCODE_MINION_REV, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_LWLINK_MASK_WR_SELWRE, fuse_data)
        FUSE_OPT_READ(device, _SELWRE_LWLINK_BUFFER_READY_SELWRE, fuse_data)

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unknown FUSE_OPT 0x%x!\n",
                __FUNCTION__,
                fuse_opt);
        break;

    }
    return fuse_data;
}
