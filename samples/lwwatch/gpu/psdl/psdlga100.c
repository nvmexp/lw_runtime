/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "psdl.h"
#include "rmpsdl.h"
#include "os.h"

#include "g_psdl_private.h"          // (rmconfig) hal/obj setup

#include "ampere/ga100/dev_fuse.h"
#include "rmpsdl.h"

//-----------------------------------------------------
// psdlPrintEcid
//-----------------------------------------------------
LW_STATUS psdlPrintEcid_GA100(LwU32 indexGpu)
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
