/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_TASKDBG_COMMUNICATION_H_
#define _RISCV_TASKDBG_COMMUNICATION_H_

#include "riscv_prv.h"

// SafeRTOS task debugger interface
#include "safertos_task_debugger.h"

LW_STATUS _tdbgTaskGetSignal(Session *pSession, LwBool *bIsHalted, LwU64 *cause);

LW_STATUS _tdbgTaskMemRead(Session *pSession, LwU8 *pBuffer, LwU64 addr, unsigned *pSize);
LW_STATUS _tdbgTaskMemWritePacket(Session *pSession, LwU8 *pBuffer, LwU64 addr, unsigned *pSize);
LW_STATUS _tdbgTaskSetBreakpoint(Session *pSession, const char *pRequest);
LW_STATUS _tdbgTaskClearBreakpoint(Session *pSession, const char *pRequest);
LW_STATUS _tdbgTaskHalt(Session *pSession);
LwBool _tdbgTaskWaitForHalt(Session *pSession, LwU64 timeoutMs, LwS8 correctSignal, LwU64 *receivedSignal);
LW_STATUS _tdbgTaskGo(Session *pSession);
LW_STATUS _tdbgTaskAttach(Session *pSession, LwU64 *cause);
LW_STATUS _tdbgTaskDetach(Session *pSession, LwU64 flags);
LW_STATUS _tdbgTaskClearStep(Session *pSession);
LW_STATUS _tdbgTaskStep(Session *pSession, const char *pRequest, LwU64 *cause);
LW_STATUS _tdbgTaskRegsRead(Session *pSession, LwU64 *xCtx);
LW_STATUS _tdbgTaskRegsWrite(Session *pSession, LwU64 *xCtx, LwBool bSendPc);
LW_STATUS _tdbgCmdVerifyAck(Session *pSession, TaskDebuggerPacket *cmd);

#define TDBG_SIGNAL_NONE (-1)
#define TDBG_SIGNAL_ANY (-2)
#define rvSIGTRAP (5)
#define rvSIGINT (2)

//
// Implementation specific hack below.
// Works: TU10x, GA100
// Breaks: GA10x
//
#define RISCV_GPR_COUNT (32)

// RTOS xCtx is uxGpr + uxPc + uxMstatus + uxMie + uxFl
// RISCV context size covers uxGpr + uxPc:
#define RTOS_RISCV_CONTEXT_SIZE (RISCV_GPR_COUNT*sizeof(LwU64))
// GDB context is x0 + uxGpr + uxPc
#define GDB_RISCV_CONTEXT_SIZE (((RISCV_GPR_COUNT)+1)*sizeof(LwU64))

// SafeRTOS configuration. Wait 20 ticks max for task to step.
#define RISCV_RTOS_STEP_TICKS (20)
// And if RTOS times out, poll every 1000 ms.
#define RISCV_RTOS_TIMEOUT_POLL_MS (1000)

#endif //_RISCV_TASKDBG_COMMUNICATION_H_
