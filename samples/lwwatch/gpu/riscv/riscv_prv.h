/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _RISCV_PRV_H_
#define _RISCV_PRV_H_

#include <lwstatus.h>
#include <lwtypes.h>
#include <print.h>

#include "riscv.h"
#include "riscv_config.h"

#include "hal.h"
#include "g_riscv_hal.h"

#include "lwsync_porting.h"

extern RiscVInstance *pRiscvInstance;

extern lwmutex_t icdLock;

LW_STATUS riscvInit();

// tokenizer (@misc)
// return - first non-token char
// start == first char of token
// tok_len == length of token
const char * riscvGetToken(const char *arg, const char **tok_start, int *tok_len);
void riscvDumpHex(const char *data, unsigned size, LwU64 offs);

extern const char reg_name[][4];
extern const char fpreg_name[][5];

//
// Mem interface
//

#define BACKDOOR_NO             0
#define MEM_FORCE_ICD_ACCESS    LW_TRUE
#define MEM_SMART_ACCESS        LW_FALSE
LW_STATUS riscvMemRead(LwU64 addr, unsigned len, void *pBuffer, LwBool bForceIcd);
LW_STATUS riscvMemWrite(LwU64 addr, unsigned len, void *pBuffer, LwBool bForceIcd);

//
// I/O interface
//
LwU32 bar0Read(LwUPtr offset);
void bar0Write(LwUPtr offset, LwU32 value);
LwU32 bar0ReadLegacy(LwUPtr offset);
void bar0WriteLegacy(LwUPtr offset, LwU32 value);
void riscvDumpState(void);
// Starts the core (if it was stopped)
#define BOOT_IN_ICD      LW_TRUE
#define BOOT_WITHOUT_ICD LW_FALSE
LW_STATUS riscvBoot(LwU64 startAddr, LwBool bStartInIcd);
// Reset the core, waits until it is in reset
// If Hammer is set - does engine reset
#define RESET_ENGINE LW_TRUE
#define RESET_CORE   LW_FALSE
LwBool riscvIsInactive(void);
LwBool riscvIsActive(void);
LwBool riscvIsRunning(void);

//
// ICD interface
//

#define ICD_ACCESS_WIDTH_TO_BYTES(X) ((LwU8)(1 << (X)))


#define CMD_WIDTH(width) DRF_NUM(_PRISCV_RISCV, _ICD_CMD, _SZ, (width))
#define CMD_REG(regIdx)  DRF_NUM(_PRISCV_RISCV, _ICD_CMD, _IDX, (regIdx))
#define CMD_PARM(value)  DRF_NUM(_PRISCV_RISCV, _ICD_CMD, _PARM, (value))
#define CMD_PARM_DM_ACCESS_PA   0
#define CMD_PARM_DM_ACCESS_VA   1

LW_STATUS riscvIcdStop(void);
LW_STATUS riscvIcdRun(void);
LW_STATUS riscvIcdJump(LwU64 addr);
LW_STATUS riscvIcdStep(void);
LW_STATUS riscvIcdSetEmask(LwU64 mask);
LW_STATUS riscvIcdRReg(unsigned reg, LwU64 *pValue);
LW_STATUS riscvIcdWReg(unsigned reg, LwU64 value);
LW_STATUS riscvIcdRdm(LwU64 address, void *pValue, ICD_ACCESS_WIDTH width);
LW_STATUS riscvIcdWdm(LwU64 address, LwU64 value, ICD_ACCESS_WIDTH width);
// Fast versions of RDM/WDM use cache and assume core is in ICD
LW_STATUS riscvIcdRdmFast(LwU64 address, void *pValue, ICD_ACCESS_WIDTH width);
LW_STATUS riscvIcdWdmFast(LwU64 address, LwU64 value, ICD_ACCESS_WIDTH width);

LW_STATUS riscvIcdRcm(LwU64 address, LwU64 *pValue);
LW_STATUS riscvIcdWcm(LwU64 address, LwU64 value);
LW_STATUS riscvIcdCmdSbu(void);
LW_STATUS riscvIcdRcsr(LwU16 address, LwU64 *pValue);
LW_STATUS riscvIcdWcsr(LwU16 address, LwU64 value);
LW_STATUS riscvIcdRPc(LwU64 *pValue);
LW_STATUS riscvIcdReadRstat(ICD_RSTAT no, LwU64 *pValue);
LW_STATUS riscvIcdRRegs(RiscvRegs *pRegs);
void riscvIcdDumpRegs(void);

void _icdDelay(void);
LW_STATUS _icdRead32(ICD_REGS reg, LwU32 *value);
LW_STATUS _icdWrite64(ICD_REGS reg, LwU64 value);
LW_STATUS _icdRead64(ICD_REGS reg, LwU64 *pValue);
LW_STATUS _icdWriteCommand(LwU32 cmd);
LW_STATUS _icdWriteAddress(LwU64 address, LwBool bCached);
LW_STATUS _icdWriteWdata(LwU64 value, ICD_ACCESS_WIDTH width, LwBool bCached);

#define CHECK_TARGET_IS_HALTED_OR_RETURN \
do\
{\
    if (!pRiscv[indexGpu].riscvIsInIcd())\
    { \
        dprintf("%s: Target not in ICD.\n", __FUNCTION__); \
        return LW_ERR_ILWALID_STATE; \
    } \
} while(0)

//
// Breakpoint interface
//
LW_STATUS riscvTriggerSetAt(LwU64 addr, TRIGGER_EVENT event);
LW_STATUS riscvTriggerClearAt(LwU64 addr, TRIGGER_EVENT event);
LW_STATUS riscvTriggerClearAll(void);
LwBool riscvCheckBreakpointFlagIsEnabled(LwU64 bp_flags);

//
// Utilities
//
void riscvDumpCsr(void);

//
// Advanced commands (mix of simple interfaces above)
//
LW_STATUS riscvWaitForHalt(unsigned timeoutUs);
LW_STATUS riscvGdbMonitor(char *pCmd, const char **ppReply);
// Translates symbolic CSR name to register address. Return -1 if not found.
struct NamedCsr
{
    const char *name;
    LwU16 address;
};
LwS16 riscvDecodeCsr(const char *name, size_t nameLen);
// Dumps MPU decode
LW_STATUS riscvDumpMpu(int regions);
LW_STATUS riscvGetLsInfo(void);

// Dumps breakpoints
LW_STATUS riscvDumpBreakpoint(int regions);
LW_STATUS riscvSetBreakpoint(int index, LwU64 addr, LwU64 flags);
LW_STATUS riscvClearAllBreakpoints(void);
LW_STATUS riscvClearBreakpoint(int index);

// Dump dmesg buffer
#define DMESG_VIA_ICD LW_TRUE
#define DMESG_VIA_PMB LW_FALSE
LW_STATUS riscvDumpDmesg(LwBool bFlush, LwBool bIcd);

//
// Debugger
//

enum GDB_MODE
{
    GDB_MODE_INTERACTIVE = 0,
    GDB_MODE_ONELINE,
    GDB_MODE_SCRIPT,
};

void stubSetDebugPrints(LwBool enabled);
LW_STATUS gdbStub(const RiscVInstance *pInstance, int connectTimeout, LwBool installSignalHandler);
LW_STATUS gdbScriptRun(const char *script, enum GDB_MODE mode);
LW_STATUS gdbStubKill();
LwBool stubCheckInstanceSwitch(RiscVInstance *pInstance);
LW_STATUS gdbProcessElwVars();

extern char gdbPath[512];
extern char gdbElf[512];
extern char gdbCmds[512];

//
// Trace buffer
//

LW_STATUS riscvTraceEnable(TRACE_MODE mode);
LW_STATUS riscvTraceDisable(void);
LW_STATUS riscvTraceFlush(void);
LW_STATUS riscvTraceDump(void);

// tmp interface
void riscvDelay(unsigned msec);
#define TGT_DEBUG(...)

#define CHECK_SUCCESS_OR_RETURN(X) do { \
    LW_STATUS _ret = (X); \
    if (_ret != LW_OK) { \
        dprintf("%s returned error: %x\n", #X, _ret); \
        return _ret; \
    }} while(0)

#define CHECK_SUCCESS_OR_GOTO(X, L) do { \
    LW_STATUS _ret = (X); \
    if (_ret != LW_OK) { \
        dprintf("%s returned error: %x\n", #X, _ret); \
        goto L; \
    }} while (0)

LwBool riscvCoreIsSec2(void);
LwBool riscvCoreIsGsp(void);
LwBool riscvCoreIsPmu(void);
LwBool riscvCoreIsMinion(void);
LwBool riscvCoreIsTsec(void);
LwBool riscvCoreIsLwdec(LwU32 instId);
LwBool riscvCoreIsFsp(void);
LwBool riscvCoreIsSoe(void);

enum MEM_TYPE
{
    MEM_DMEM = 0,
    MEM_IMEM,
    MEM_IMEMS,
    MEM_VIRT,
    MEM_FB,
    MEM_EMEM
};

#endif // RISCV_PRV_H
