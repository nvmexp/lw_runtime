/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "common_lwswitch.h"
#include "lr10/lr10.h"

#include "lwswitch/lr10/dev_fuse.h"

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

    // Fuses not instantiated on fmodel and RTL
    if (!(IS_FMODEL(device) || IS_RTLSIM(device)))
    {
        fuse_data = LWSWITCH_OFF_RD32(device, fuse_reg);
    }

    LWSWITCH_PRINT(device, INFO,
        "%s: Fuse 0x%x = 0x%x\n",
        __FUNCTION__, fuse_reg, fuse_data);

    return fuse_data;
}

#define FUSE_OPT_READ(_d, _fuse, _data)                             \
    case LW_FUSE_OPT ## _fuse:                                      \
    {                                                               \
        LwU32 plm;                                                  \
        plm = _lwswitch_fuse_read(_d, LW_FUSE_OPT ## _fuse ## __PRIV_LEVEL_MASK);                   \
        if (FLD_TEST_DRF(_FUSE_OPT, _PRIV_LEVEL_MASK, _READ_PROTECTION_LEVEL0, _ENABLE, plm))       \
        {                                                                                           \
            _data = DRF_VAL(_FUSE_OPT, _fuse, _DATA, _lwswitch_fuse_read(_d, LW_FUSE_OPT ## _fuse)); \
        }                                                           \
        else                                                        \
        {                                                           \
            _data = 0;                                              \
        }                                                           \
        LWSWITCH_PRINT(_d, INFO,                                    \
            "%s: %s (%x)=%x\n",                                     \
            __FUNCTION__,                                           \
            #_fuse, LW_FUSE_OPT ## _fuse, _data);                   \
    }                                                               \
    break;


LwU32
lwswitch_fuse_opt_read_lr10
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
        FUSE_OPT_READ(device, _CP2_TDIODE_OFFSET, fuse_data)

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unknown FUSE_OPT 0x%x!\n",
                __FUNCTION__,
                fuse_opt);
        break;

    }
    return fuse_data;
}

