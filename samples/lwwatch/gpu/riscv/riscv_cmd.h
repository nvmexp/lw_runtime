/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef RISCV_CMD_H
#define RISCV_CMD_H

#include "riscv.h"

typedef struct
{
    const char *command;
    const char *help;
    LW_STATUS (*handler)(const char *);
    LwBool requiresLock;
} Command;

extern RiscVInstance instances[RISCV_INSTANCE_END];

extern RiscVInstance *pRiscvInstance;

#endif // RISCV_CMD_H
