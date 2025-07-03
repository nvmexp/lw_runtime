/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// os.h
//
//*****************************************************

#ifndef _OS_H_
#define _OS_H_

#ifdef __cplusplus
extern "C" {
#endif

//
// Memory types
//
typedef enum _MEM_TYPE
{
    SYSTEM_PHYS,
    SYSTEM_VIRT,
    REGISTER,
    FRAMEBUFFER,
    INSTANCE
} MEM_TYPE;

#include "lwwatch.h"
#include <lwstatus.h>

#if !defined(USERMODE) && !defined(CLIENT_SIDE_RESMAN)
#ifdef WIN32
#include "osWin.h"
#include "hwref/lwutil.h"
#include "osMods.h"
#endif // WIN32
#endif // !USERMODE && !CLIENT_SIDE_RESMAN

#if LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#include "osWin.h"
#endif

#ifdef USERMODE
#include "lwtypes.h"
#include "lwgputypes.h"
#include "hwref/lwutil.h"
#include <stdio.h>
#if !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#include "usermode.h"
#endif
#include <sys/types.h>
#ifndef WIN32
#include <dirent.h>// can't find this on windows
#endif
#include <stdlib.h>         // malloc, getelw, etc.
#include <string.h>         // everybody loves strcpy()
#endif // USERMODE

#ifdef WIN32
#define snprintf _snprintf
#endif

#include <lwtypes.h>
#include <lwgputypes.h>
#include "hwref/lwutil.h"

#if defined(CLIENT_SIDE_RESMAN)
#include "simLinux.h"
#include <stdio.h>
#endif

#if LWWATCHCFG_IS_PLATFORM(UNIX)
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include "unix.h"
#endif

#ifdef LW_QNX
#  include "osQNX.h"
#endif

//
// lw include
//
#include <lwtypes.h>
#include <lwgputypes.h>
#include "lwwatchErrors.h"

//
// defines
//
#define MAX_ICB_ENTRIES     16

#ifdef WIN32
#define DIR_SLASH               "\\"
#define DIR_SLASH_CHAR          '\\'
#define DIR_EXAMPLE_BASE        "c:\\sw\\dev\\gpu_drv\\chips_a"
#else
#define DIR_SLASH               "/"
#define DIR_SLASH_CHAR          '/'
#define DIR_EXAMPLE_BASE        "/src/sw/dev/gpu_drv/chips_a"
#endif
#define CLASS_DIR_FRAGMENT      DIR_SLASH "sdk" DIR_SLASH "lwpu" DIR_SLASH "inc" DIR_SLASH "class"
#define CLASS_DIR_EXAMPLE       DIR_EXAMPLE_BASE CLASS_DIR_FRAGMENT
#define INC_DIR_EXAMPLE         DIR_SLASH DIR_SLASH "sw" DIR_SLASH "resman" DIR_SLASH "manuals"

typedef enum
{
    MODE_NONE = 0,
    MODE_LIVE = BIT(0),
    MODE_DUMP = BIT(1),
    MODE_ANY  = MODE_LIVE | MODE_DUMP,
} LW_MODE;

#define CHECK(call)                         \
    status = (call);                        \
    if (status != LW_OK)                    \
        return status;

#define CHECK_EXIT(call)                    \
    status = (call);                        \
    if (status != LW_OK)                    \
        goto exit;

#if   defined(USERMODE)
#define CHECK_INIT(mode_mask)
#else
#define CHECK_INIT(mode_mask)                                               \
    if (!usingMods && (lwBar0 == 0) && !(lwMode & (mode_mask)))             \
    {                                                                       \
        dprintf("lw: Call !lw.init <lwBar0> or !lw.modsinit "               \
                "or !lw.dumpinit to begin.\n");                             \
        return;                                                             \
    }
#endif

//
// globals - for MGPU_LOOP_XXX
//
#define MAX_GPUS        64

extern LwU64 multiGpuBar0[MAX_GPUS];
extern LwU64 multiGpuBar1[MAX_GPUS];
extern LwU32 multiPmcBoot0[MAX_GPUS];

#define MGPU_LOOP_START                                     \
{                                                           \
    LwU32 orig_verboseLevel = verboseLevel;                 \
    PhysAddr orig_lwBar0    = lwBar0;                       \
    PhysAddr orig_lwBar1    = lwBar1;                       \
    LwU32 orig_pmc_boot0    = pmc_boot0;                    \
    LwU32 m_Count = 0;                                      \
    do                                                      \
    {                                                       \
        if (multiGpuBar0[m_Count] && multiGpuBar1[m_Count]) \
        {                                                   \
            lwBar0 = (PhysAddr) multiGpuBar0[m_Count];      \
            lwBar1 = (PhysAddr) multiGpuBar1[m_Count];      \
            pmc_boot0 = multiPmcBoot0[m_Count];             \
            verboseLevel = 0;                               \
            osInitHal();                                    \
            verboseLevel = orig_verboseLevel;               \
            dprintf("lw: device %u, lwBar0: " PhysAddr_FMT, \
                    m_Count, lwBar0);                       \
            dprintf(" lwBar1: " PhysAddr_FMT " ", lwBar1);  \
            dprintf(" LW_PMC_BOOT_0: 0x%08x\n", pmc_boot0); \
            m_Count++;                                      \
        }

#define MGPU_LOOP_END                                       \
    } while (m_Count < MAX_GPUS &&                          \
             multiGpuBar0[m_Count] &&                       \
             multiGpuBar1[m_Count]);                        \
    lwBar0    = orig_lwBar0;                                \
    lwBar1    = orig_lwBar1;                                \
    pmc_boot0 = orig_pmc_boot0;                             \
    verboseLevel = 0;                                       \
    if (lwBar0 && m_Count > 1)                              \
        osInitHal();                                        \
    verboseLevel = orig_verboseLevel;                       \
}

//
// Variable accesses should really be cleaned up, cos not every platform
// runs WinDBG.
//
#ifndef LW_VAR
#define LW_VAR(NAME)    "@@(" NAME ")"
#endif

#ifndef LW_ARRAY_ELEMENTS
#define LW_ARRAY_ELEMENTS(x)   ((sizeof(x)/sizeof((x)[0])))
#endif

#ifndef LwU64_FMT
#define LwU64_FMT       "0x%016llx"
#define LwU40_FMT       "0x%010llx"
#endif

//
// data structs
//
typedef struct _def_chipinfo_struct
{
    LwU32 Implementation;
    LwU32 Revision;
    LwU32 Architecture;
    LwU32 MaskRevision;
} LWWATCHCHIPINFOSTRUCT, *PLWWATCHCHIPINFOSTRUCT;

#ifdef WIN32
#pragma pack (push, _store_)
#pragma pack (1)
#endif // #ifdef WIN32

typedef union
{
    LwU32 data32;
    struct
    {
        LwU8 WriteIdx;
        LwU8 ReadIdx;
        LwU8 Rsvd;
        LwU8 Access; // Always 0 for CRTC
    } CR;
    struct
    {
        LwU8 WriteIO;
        LwU8 ReadIO;
        LwU8 HiIO;
        LwU8 Access; // Always 1 for Direct IO
    } DI;
    struct
    {
        LwU32 Bus       : 3;
        LwU32 Device    : 5;
        LwU32 Function  : 3;
        LwU32 Bar       : 3;
        LwU32 Type      : 2;
        LwU32 Offset    : 8;
        LwU32 Access    : 8;
    } PI;
} ICBENTRY;

typedef LwU8 MemProtFlags;
#define MEM_PROT_NONE   0x0
#define MEM_PROT_EXEC   0x1
#define MEM_PROT_READ   0x2
#define MEM_PROT_WRITE  0x4

typedef LwU64 PhysAddr;
#define PhysAddr_FMT LwU64_FMT

#ifdef WIN32
#pragma pack (pop, _store_)
#endif // WIN32

//
// globals - remove eventually and place in halObj
//
extern PhysAddr   lwBar1;
extern PhysAddr   lwBar0;
extern PhysAddr   lwClassCodeRevId;

extern LwU32   verboseLevel;
extern LwU32   pmc_boot0;
extern LW_MODE lwMode;

//
// globals - for lwwatchMods
//
extern LwU32        usingMods;
extern int          debugOCMFlag;
extern const char * const LWWATCH_OC_DEBUG_FILE;


//
// Callbacks
//
typedef void (*osPciCallback)(LwU16, LwU8, LwU8, LwU8);

//
// OS routines
//
LW_STATUS   osInit(void);
LW_STATUS   osInitHal(void);
void        osDestroyHal(void);
void        osInitBar1(PhysAddr *bar1);
LwU32       osRegRd32(PhysAddr reg);
void        osRegWr32(PhysAddr reg, LwU32 data);
LwU8        osRegRd08(PhysAddr reg);
void        osRegWr08(PhysAddr reg, LwU8 data);
LwU8        REG_RDCR(LwU8 crReg, LwU32 crtcOffset);
void        REG_WRCR(LwU8 crReg, LwU8 crVal, LwU32 crtcOffset);
LwU32       FB_RD32(LwU32 reg);
LwU32       FB_RD32_64(LwU64 reg);
void        FB_WR32(LwU32 reg, LwU32 data);
LwU8        SYSMEM_RD08(LwU64 physAddr);
LwU16       SYSMEM_RD16(LwU64 physAddr);
LwU32       SYSMEM_RD32(LwU64 physAddr);
void        SYSMEM_WR32(LwU64 physAddr, LwU32 data);
LwU32       RD_PHYS32(PhysAddr physAddr);
void        WR_PHYS32(PhysAddr physAddr, LwU32 data);
LwU32       RD_VIRT32(LwU32 virtAddr);
LwU32       TMDS_RD(LwU32 Link, LwU32 index);
void        TMDS_WR(LwU32 Link, LwU32 index, LwU32 data);
LwU32       DEV_REG_RD32(PhysAddr reg, const char * const devName, LwU32 devIndex);
void        DEV_REG_WR32(PhysAddr reg, LwU32 data, const char * const devName, LwU32 devIndex);
LwU32       GPU_REG_RD32(PhysAddr reg);
LwU32       GPU_REG_RD32_DIRECT(PhysAddr reg);
void        GPU_REG_WR32(PhysAddr reg, LwU32 data);
void        GPU_REG_WR32_DIRECT(PhysAddr reg, LwU32 data);
LwU8        GPU_REG_RD08(PhysAddr reg);
void        GPU_REG_WR08(PhysAddr reg, LwU8 data);

LW_STATUS   osReadMemByType(PhysAddr address, void* buf, LwU64 size, LwU64 *pSizer, MEM_TYPE memoryType);
LW_STATUS   readSystem(LwU64 pa, void* buffer, LwU32 length);
LW_STATUS   writeSystem(LwU64 pa, void* buffer, LwU32 length);

LW_STATUS   osPciRead32(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU32* Buffer, LwU32 Offset);
LW_STATUS   osPciRead16(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU16* Buffer, LwU32 Offset);
LW_STATUS   osPciRead08(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU8* Buffer, LwU32 Offset);
LW_STATUS   osPciWrite32(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU32 Data, LwU32 Offset);
LW_STATUS   osPciFindDevices(LwU16 DeviceId, LwU16 VendorId, osPciCallback callback);
LW_STATUS   osPciFindDevicesByClass(LwU32 classCode, osPciCallback callback);
LW_STATUS   osPciGetBarInfo(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU8 BarIndex,  LwU64 *BaseAddr, LwU64 *BarSize);

LW_STATUS   osMapDeviceMemory(LwU64 BaseAddr, LwU64 Size, MemProtFlags prot, void **ppBar);
LW_STATUS   osUnMapDeviceMemory(void *pBar, LwU64 BarSize);

LwU32       osCheckControlC(void);
void        osPerfDelay(LwU32 MicroSeconds);
LW_STATUS   osGetInputLine(LwU8 *prompt, LwU8 *buffer, LwU32 size);
BOOL        IsAndroid(void);

// offline dump mode
LW_STATUS   dumpModeInit(const char *zipFilename, const char *innerFilename);
LW_STATUS   dumpModePrint(char *prbFieldName);
LW_STATUS   dumpModeFeedback(const char *filename);
LW_STATUS   dumpModeReadFb(LwU64 offset, LwU32 length, LwU8 size);
LwU32       REG_RD32_DUMP(PhysAddr reg);
LwU32       FB_RD32_DUMP(LwU32 reg);
LW_STATUS   fbReadWrite_DUMP(LwU64 offset, void* buffer, LwU32 length, LwU32 is_write);
void        osCmdProlog();
void        osCmdEpilog();
extern      LwBool bRegisterMissing;
LW_STATUS   getRiscvCoreDumpFromProtobuf(LwU8* buffer, LwU32 bufferSize, LwU32 *actualSize);

#ifdef __cplusplus
}
#endif

#endif // _OS_H_

