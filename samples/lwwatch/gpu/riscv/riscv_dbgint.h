/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef RISCV_DBGINT_H
#define RISCV_DBGINT_H
#include "lwtypes.h"
#include "riscv_config.h"

/*
 * For now only debug dmesg-like buffer is specified here.
 * Buffer rules:
 * - Header is placed at the end of DMEM
 * - Size is defined by RTOS
 * - write_offset is updated by RTOS, read_offset by reader
 * - magic is fixed and must be checked
 * - Buffer is located before header (0x10000 - sizeof(HDR) - buffer_size)
 */

typedef struct {
    LwU32 read_offset;
    LwU32 write_offset;
    LwU32 buffer_size;
    LwU32 magic;
} RiscvDbgDmesgHdr;

#define RISCV_DMESG_MAGIC    0xF007BA11U

#endif
