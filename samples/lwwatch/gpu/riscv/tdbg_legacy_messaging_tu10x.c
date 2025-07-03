/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "turing/tu102/dev_gsp.h"
#include "turing/tu102/dev_sec_pri.h"
#include "riscv_taskdbg.h"
#include "tdbg_legacy_messaging.h"
#include "riscv_prv.h"

#include "g_riscv_private.h"

LwU32 _queueHead_TU10X()
{
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return LW_PGSP_QUEUE_HEAD(DEBUGGER_QUEUE);
    case RISCV_INSTANCE_SEC2:
        return LW_PSEC_QUEUE_HEAD(DEBUGGER_QUEUE);
    default:
        return (LwU32) -1;
    }
}

LwU32 _queueTail_TU10X()
{
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return LW_PGSP_QUEUE_TAIL(DEBUGGER_QUEUE);
    case RISCV_INSTANCE_SEC2:
        return LW_PSEC_QUEUE_TAIL(DEBUGGER_QUEUE);
    default:
        return (LwU32) -1;
    }
}

LwU32 _msgqHead_TU10X()
{
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return LW_PGSP_MSGQ_HEAD(DEBUGGER_QUEUE);
    case RISCV_INSTANCE_SEC2:
        return LW_PSEC_MSGQ_HEAD(DEBUGGER_QUEUE);
    default:
        return (LwU32) -1;
    }
}

LwU32 _msgqTail_TU10X()
{
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return LW_PGSP_MSGQ_TAIL(DEBUGGER_QUEUE);
    case RISCV_INSTANCE_SEC2:
        return LW_PSEC_MSGQ_TAIL(DEBUGGER_QUEUE);
    default:
        return (LwU32) -1;
    }
}
