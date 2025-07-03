/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "os.h"

#include "hopper/gh100/lw_gsp_riscv_address_map.h"

#include "riscv_man.h"
#include "riscv_man_gh100_gsp.h"

LwU64 getRiscvDef_GH100_GSP(const char *macro)
{
    GETRISCVDEF_ADDDEF(LW_RISCV_AMAP_IMEM_START);
    GETRISCVDEF_ADDDEF(LW_RISCV_AMAP_DMEM_START);
    GETRISCVDEF_ADDDEF(LW_RISCV_AMAP_EMEM_START);
    GETRISCVDEF_ADDDEF(LW_RISCV_AMAP_FBGPA_START);
    GETRISCVDEF_ADDDEF(LW_RISCV_AMAP_FBGPA_SIZE);
    GETRISCVDEF_ADDDEF(LW_RISCV_AMAP_PRIV_START);
    GETRISCVDEF_ADDDEF(LW_RISCV_AMAP_PRIV_SIZE);
    dprintf("%s: unsupported macro: %s", __FUNCTION__, macro);
    return 0;
}
