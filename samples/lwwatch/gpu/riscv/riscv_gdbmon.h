/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_GDBMON_H
#define _RISCV_GDBMON_H

#include "riscv_config.h"

extern char monitorReplyBuf[MONITOR_BUF_SIZE];
extern char *pMonitorSavedPtr; // for tokenizer

#endif // _RISCV_GDBMON_H
