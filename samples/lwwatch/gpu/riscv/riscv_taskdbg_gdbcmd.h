/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_TASKDBG_GDBCMD_H_
#define _RISCV_TASKDBG_GDBCMD_H_

#include "riscv_prv.h"

LW_STATUS _taskDebuggerStubInstance(const RiscVInstance *pInstance, Session *pSession, LwU64 instanceId);

#endif //_RISCV_TASKDBG_GDBCMD_H_
