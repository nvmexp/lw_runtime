/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_TASKDBG_H_
#define _RISCV_TASKDBG_H_

// SafeRTOS task debugger interface
#include "safertos_task_debugger.h"
#include "riscv_gdbstub.h"

#define DEBUGGER_QUEUE 1
#define LW_RTOS_PTRACE_MAX_TASKS 8

LW_STATUS tdbgResetDebugger(Session *pSession);
LW_STATUS tdbgListTasks(RiscVInstance *pInstance);
LW_STATUS tdbgListSessions();

LwBool taskDebuggerCheckInstanceSwitch(const RiscVInstance *pInstance);
LW_STATUS taskDebuggerStub(const RiscVInstance *pInstance, LwU64 xTCB, LwBool bWaitForHalt, LwU64 xWaitForHaltMaxPollingInterval);

#define TASKSTATE_ILWALID   0
#define TASKSTATE_RUNNING   1
#define TASKSTATE_HALTED    2
#define TASKSTATE_CRASHED   3

#endif //_RISCV_TASKDBG_H_
