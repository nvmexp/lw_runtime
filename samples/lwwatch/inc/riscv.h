/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef RISCV_H
#define RISCV_H

#include "lwstatus.h"

#define U64_IMM_(X) (X ## ULL)
#define U64_IMM(X) U64_IMM_(X)

typedef struct
{
    LwU64 gpr[32];
    LwF32 fpgpr[32];
    LwU64 pc;
} RiscvRegs;

typedef enum  {
    RISCV_INSTANCE_GSP=0,
    RISCV_INSTANCE_SEC2,
    RISCV_INSTANCE_PMU,
    RISCV_INSTANCE_MINION,
    RISCV_INSTANCE_TSEC,
    RISCV_INSTANCE_LWDEC0,
    RISCV_INSTANCE_LWDEC1,
    RISCV_INSTANCE_LWDEC2,
    RISCV_INSTANCE_LWDEC3,
    RISCV_INSTANCE_LWDEC4,
    RISCV_INSTANCE_LWDEC5,
    RISCV_INSTANCE_LWDEC6,
    RISCV_INSTANCE_LWDEC7,
    RISCV_INSTANCE_LWDEC_LAST=RISCV_INSTANCE_LWDEC7,
    RISCV_INSTANCE_FSP,
    RISCV_INSTANCE_SOE,
    RISCV_INSTANCE_END,
} RiscvInstanceType;

// Indexed by RiscvInstanceType
extern const char * const riscvInstanceNames[RISCV_INSTANCE_END];

/*!
 * @brief    Riscv instance core address spaces info.
 * @details  Populated by @ref riscvPopulateCoreInfo.
 */
typedef struct
{
    //
    // Name of the RISCV instance.
    // Remain NULL if not supported.
    //
    const char *name;
    RiscvInstanceType instance_no;
    LwU64 bar0Base; // Sourced from dev_falcon_v(1|4).h
    LwU64 riscvBase; // Sourced from dev_riscv_pri.h
    LwU16 defaultPort;
    LwU64 riscv_dmesg_hdr_addr;
    LwU64 riscv_imem_start;
    LwU64 riscv_dmem_start;
    LwU64 riscv_imem_size;
    LwU64 riscv_dmem_size;
    LwU64 riscv_fb_start;
    LwU64 riscv_fb_size;
    LwU64 riscv_priv_start;
    LwU64 riscv_priv_size;
    LwU64 riscv_emem_start;
    LwU32 riscv_emem_size;
} RiscVInstance;

//
// ICD interface
//

typedef enum
{
    ICD_WIDTH_8 = 0,
    ICD_WIDTH_16,
    ICD_WIDTH_32,
    ICD_WIDTH_64
} ICD_ACCESS_WIDTH;

typedef enum
{
    ICD_RSTAT0 = 0,
    ICD_RSTAT1,
    ICD_RSTAT2,
    ICD_RSTAT3,
    ICD_RSTAT4,
    ICD_RSTAT5,
    ICD_RSTAT_END = ICD_RSTAT5
} ICD_RSTAT;

typedef enum
{
    ICD_CMD = 0,
    ICD_ADDR,
    ICD_WDATA,
    ICD_RDATA,
    _ICD_END = ICD_RDATA
} ICD_REGS;

typedef enum
{
    CMD_STOP = 0,
    CMD_RUN = 1,
    CMD_STEP = 5,
    CMD_JUMP = 6,
    CMD_EMASK = 7,
    CMD_RREG,
    CMD_WREG,
    CMD_RDM,
    CMD_WDM,
    CMD_RCM,
    CMD_WCM,
    CMD_RSTAT,
    CMD_SBU,
    CMD_RCSR,
    CMD_WCSR,
    CMD_RPC,
    CMD_RFREG,
    CMD_WFREG,
} ICD_CMDS;

//
// Breakpoint interface
//

typedef enum
{
    TRIGGER_UNUSED = 0,
    TRIGGER_ON_LOAD = 0x1,
    TRIGGER_ON_STORE = 0x2,
    TRIGGER_ON_EXEC = 0x4,
    TRIGGER_MASK = 0x7,
    TRIGGER_ALL = TRIGGER_MASK
} TRIGGER_EVENT;

#define TRIGGERS_MAX 64
extern LwU64 triggerAddrs[TRIGGERS_MAX];
extern TRIGGER_EVENT triggerEvents[TRIGGERS_MAX];

//
// Trace buffer
//

typedef enum
{
    TRACE_FULL = 0,
    TRACE_REDUCED,
    TRACE_STACK
} TRACE_MODE;

/* ------------------------ Non-HAL Function - riscv_main.c  ---------------- */
#define RISCV_GDBSTUB_INIT_TIMEOUT_MS 15000
LW_STATUS riscvMain(const char *pArgs);
LW_STATUS riscvGdbMain(const char *pArgs);

#endif // RISCV_H
