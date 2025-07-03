/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stddef.h>

#include "os.h"

#include "riscv_man.h"
#include "riscv_man_gh100_fsp.h"
#include "riscv_man_gh100_gsp.h"
#include "riscv_man_gh100_minion.h"
#include "riscv_man_gh100_lwdec.h"
#include "riscv_man_gh100_pmu.h"
#include "riscv_man_gh100_sec2.h"
#include "riscv_man_ls10_soe.h"
#include "riscv_man_t234_lwdec.h"
#include "riscv_man_t234_pmu.h"
#include "riscv_man_t234_tsec.h"

typedef LwU64 RiscvDefGetter(const char *);

// These arrays are indexed by the RiscvInstanceType enum.
static RiscvDefGetter *riscvDefGetters_GH100[] = {
    getRiscvDef_GH100_GSP,
    getRiscvDef_GH100_SEC2,
    getRiscvDef_GH100_PMU,
    getRiscvDef_GH100_MINION,
    NULL,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_LWDEC,
    getRiscvDef_GH100_FSP,
    getRiscvDef_LS10_SOE,
};

static RiscvDefGetter *riscvDefGetters_T234[] = {
    NULL,
    NULL,
    getRiscvDef_T234_PMU,
    NULL,
    getRiscvDef_T234_TSEC,
    getRiscvDef_T234_LWDEC,
    NULL,
    NULL,
};

static LwU64 getRiscvDef_GENERIC
(
    RiscvInstanceType engine,
    const char *macro,
    RiscvDefGetter *riscvDefGetters[]
)
{
    if (engine >= RISCV_INSTANCE_END || riscvDefGetters[engine] == NULL)
    {
        dprintf("%s: unsupported engine: %d\n", __FUNCTION__, engine);
        return 0;
    }
    return riscvDefGetters[engine](macro);
}

LwU64 getRiscvDef_GH100
(
    RiscvInstanceType engine,
    const char *macro
)
{
    return getRiscvDef_GENERIC(engine, macro, riscvDefGetters_GH100);
}

LwU64 getRiscvDef_T234
(
    RiscvInstanceType engine,
    const char *macro
)
{
    return getRiscvDef_GENERIC(engine, macro, riscvDefGetters_T234);
}
