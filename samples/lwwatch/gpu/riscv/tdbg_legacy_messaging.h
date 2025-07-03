/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_TDBG_EMEM_MESSAGING_H_
#define _RISCV_TDBG_EMEM_MESSAGING_H_

#include "riscv_prv.h"
#include "riscv_taskdbg.h"

#define RISCV_LEGACY_MESSAGING_TIMEOUT_MS 9001

// 0: No debug info
// 1: Minimal debug info
// 2: Full debug info
#define DEBUG_LEGACY_MESSAGING 2

void riscvEmemEmergencyReset();
LW_STATUS riscvRecvMessage(Session *session, void *pBuffer);
LW_STATUS riscvSendCommand(Session *session, TaskDebuggerPacket *pTaskDbgCmd);

#if DEBUG_LEGACY_MESSAGING > 1
static void debugQueues(LwU32 addr, LwU32 size, LwU32 head, LwU32 tail);
#endif

#endif //_RISCV_TDBG_EMEM_MESSAGING_H_
