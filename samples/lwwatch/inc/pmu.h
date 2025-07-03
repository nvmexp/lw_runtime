/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _PMU_H_
#define _PMU_H_

#include "hal.h"
#include "falcon.h"

// RM header for definition of structure PMU_TCB_PVT
#include "lw_rtos_extension.h"

// PMU registers RD/WR in GPU space
#define PMU_REG_RD32(a)     DEV_REG_RD32((a), "GPU", 0)
#define PMU_REG_WR32(a, b)  DEV_REG_WR32((a), (b), "GPU", 0)

#define LW_PMU_DMEMC_DEFAULT_PORT       0x03   /** User port 3 for DMEM by default */
#define LW_PPWR_FALCON_IMBLK_VALID      24:24  /** As defined by Falcon Architecture (core 3) */
#define LW_PPWR_FALCON_IMBLK_PENDING    25:25
#define LW_PPWR_FALCON_IMBLK_SELWRE     26:26
#define LW_PPWR_FALCON_IMTAG_VALID      LW_PPWR_FALCON_IMBLK_VALID
#define LW_PPWR_FALCON_IMTAG_PENDING    LW_PPWR_FALCON_IMBLK_PENDING
#define LW_PPWR_FALCON_IMTAG_SELWRE     LW_PPWR_FALCON_IMBLK_SELWRE
#define LW_PPWR_FALCON_IMTAG_MULTI_HIT  30:30
#define LW_PPWR_FALCON_IMTAG_MISS       31:31

#define PMU_IMEM_BLOCK_SIZE_BYTES       256
#define PMU_IMEM_BLOCK_SIZE_WORDS       64

#define PMU_DMEM_BLOCK_SIZE_BYTES       256
#define PMU_DMEM_BLOCK_SIZE_WORDS       64

#define PMU_ILWALID_MUTEX_OWNER_ID      0xFF

// sanity test macros
#ifdef CLIENT_SIDE_RESMAN
    void PMU_LOG(int lvl, const char *fmt, ...);
#else
    #define PMU_LOG(l,f,...) PMU_LOGGING(verbose, l, f, ##__VA_ARGS__)
#endif

#define VB0  0
#define VB1  1
#define VB2  2

/*!
 *  PMU sanity test logging wrapper.
 */
#ifndef CLIENT_SIDE_RESMAN
#define PMU_LOGGING(v,l,f,...) \
    do\
    {\
        if (v > l)\
        {\
            int lvl = l;\
            if (lvl) dprintf(" ");\
            while (lvl--) dprintf(">");\
            dprintf (f, ## __VA_ARGS__); \
        }\
    } while(0)
#endif

/*!
 * PMU sanity test function flags
 *
 * AUTO          - include in the autorun mode
 * DESTRUCTIVE   - changes the current state of PMU
 * REQUIRE_ARGS  - requires an extra arugument to be passed.
 * OPTIONAL_ARGS - takes an argument, but not required.
 * PROD_UCODE    - requires a production ucode.
 * VERIF_UCOD    - requires a verfication ucode.
 */
#define PMU_TEST_FLAGS_CODE    "ADXOPV"

#define PMU_TEST_AUTO          BIT(0)
#define PMU_TEST_DESTRUCTIVE   BIT(1)
#define PMU_TEST_REQUIRE_ARGS  BIT(2)
#define PMU_TEST_OPTIONAL_ARGS BIT(3)
#define PMU_TEST_PROD_UCODE    BIT(4)
#define PMU_TEST_VERIF_UCODE   BIT(5)

#define PMU_DEREF_DMEM_PTR(ptr, port, pVal) \
    ((pPmu[indexGpu].pmuDmemRead(ptr, LW_TRUE, 1, port, pVal)) == 1 ? TRUE : FALSE)

#define PMU_DEREF_DMEM_PTR_64(ptr, port, pVal) \
    ((pPmu[indexGpu].pmuDmemRead(ptr, LW_TRUE, 2, port, (LwU32*)pVal)) == 2 ? TRUE : FALSE)

#define PMU_PRINT_SDK_MESSAGE()                                                                    \
    dprintf("lw: %s - Please set LWW_MANUAL_SDK to a HW manual directory.\n", __FUNCTION__);       \
    dprintf("lw: For example, c:\\sw\\dev\\gpu_drv\\chips_a\\drivers\\resman\\kernel\\inc.\n");    \
    dprintf("lw: \n");                                                                             \
    dprintf("lw: Be sure to use the same *software* branch from which the ucode was built. It\n"); \
    dprintf("lw: is from this path that the ucode binary path will be derived.\n");

// Falcon Register index (these need to moved into a shared manual)
#ifndef LW_FALCON_REG_R0
#define LW_FALCON_REG_R0                       (0)
#define LW_FALCON_REG_R1                       (1)
#define LW_FALCON_REG_R2                       (2)
#define LW_FALCON_REG_R3                       (3)
#define LW_FALCON_REG_R4                       (4)
#define LW_FALCON_REG_R5                       (5)
#define LW_FALCON_REG_R6                       (6)
#define LW_FALCON_REG_R7                       (7)
#define LW_FALCON_REG_R8                       (8)
#define LW_FALCON_REG_R9                       (9)
#define LW_FALCON_REG_R10                      (10)
#define LW_FALCON_REG_R11                      (11)
#define LW_FALCON_REG_R12                      (12)
#define LW_FALCON_REG_R13                      (13)
#define LW_FALCON_REG_R14                      (14)
#define LW_FALCON_REG_R15                      (15)
#define LW_FALCON_REG_IV0                      (16)
#define LW_FALCON_REG_IV1                      (17)
#define LW_FALCON_REG_UNDEFINED                (18)
#define LW_FALCON_REG_EV                       (19)
#define LW_FALCON_REG_SP                       (20)
#define LW_FALCON_REG_PC                       (21)
#define LW_FALCON_REG_IMB                      (22)
#define LW_FALCON_REG_DMB                      (23)
#define LW_FALCON_REG_CSW                      (24)
#define LW_FALCON_REG_CCR                      (25)
#define LW_FALCON_REG_SEC                      (26)
#define LW_FALCON_REG_CTX                      (27)
#define LW_FALCON_REG_EXCI                     (28)
#define LW_FALCON_REG_RSVD0                    (29)
#define LW_FALCON_REG_RSVD1                    (30)
#define LW_FALCON_REG_RSVD2                    (31)
#define LW_FALCON_REG_SIZE                     (32)

#define FALC_REG(x)                            LW_FALCON_REG_##x
#endif

/** @typedef typedef struct PmuBlock PmuBlock
 *  @see struct PmuBlock
 **/
typedef struct PmuBlock PmuBlock;

/** @typedef typedef struct PmuTagBlock PmuTagBlock
 *  @see struct PmuTagBlock
 **/
typedef struct PmuTagBlock PmuTagBlock;

/** @typedef enum PMU_TAG_MAPPING PMU_TAG_MAPPING
 *  @see enum PMU_TAG_MAPPING
 **/
typedef enum PMU_TAG_MAPPING PMU_TAG_MAPPING;


/** @typedef struct PmuSanityTestEntry PmuSanityTestEntry
 *  @see struct PmuSanityTestEntry
 **/
typedef struct PmuSanityTestEntry PmuSanityTestEntry;

/** @typedef LwU32 (*PmuTestFunc) (LwU32, char *)
 *  function pointer to each test case
 *  1st arg - verbose
 *  2nd arg - extra argument
 **/
typedef LwU32 (*PmuTestFunc) (LwU32, char *);

/** @enum PMU_TAG_MAPPING
 *
 *  Possible IMEM tag to block mapping states.
 **/
enum PMU_TAG_MAPPING
{
    PMU_TAG_MAPPED       = 0,   /**< Tag is mapped to exactly 1 block */
    PMU_TAG_UNMAPPED     = 1,   /**< Tag is not mapped to any block */
    PMU_TAG_MULTI_MAPPED = 2    /**< Tag is mapped to multiple blocks */
};

/** @struct PmuBlock
 *
 *  PMU IMEM block information.
 **/
struct PmuBlock
{
    LwU32 tag;                   /**< IMEM tag of this block */
    LwU32 blockIndex;            /**< Index of block */
    LwU32 bValid   :1;           /**< Is the block valid? */
    LwU32 bPending :1;           /**< Is the block pending? */
    LwU32 bSelwre  :1;           /**< Is the block secure? */
};

/** @struct PmuTagBlock
 *
 *  Describes the IMEM tag to block mapping for a specific tag. The mapType
 *  specifies whether or not blockInfo is valid. PMU_TAG_UNMAPPED implies that
 *  the blockInfo data is not set or is invalid.
 **/
struct PmuTagBlock
{
    PMU_TAG_MAPPING mapType;    /**< Type of mapping from tag to block */
    PmuBlock        blockInfo;  /**< Information of mapped block */
};

typedef struct
{
    LwU32               itemValue;
    LwU32               next;
    LwU32               prev;
    LwU32               owner;
    LwU32               container;
} PMU_XLIST_ITEM;

typedef struct
{
    LwU32               itemValue;
    LwU32               next;
    LwU32               prev;
} PMU_XMINI_LIST_ITEM;

typedef struct
{
    LwU32               numItems;
    LwU32               pIndex;
    PMU_XMINI_LIST_ITEM listEnd;
} PMU_XLIST;

/*
 * Version defines for the max supported overlays.
 * The latest version should match with what is defined
 * in the RTOS code.
 */
#define PMU_MAX_ATTACHED_OVLS_IMEM_VER_0 16
#define PMU_MAX_ATTACHED_OVLS_IMEM_VER_1 32
#define PMU_MAX_ATTACHED_OVLS_IMEM_VER_2 PMU_MAX_ATTACHED_OVLS_IMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_IMEM_VER_3 PMU_MAX_ATTACHED_OVLS_IMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_IMEM_VER_4 PMU_MAX_ATTACHED_OVLS_IMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_IMEM_VER_5 PMU_MAX_ATTACHED_OVLS_IMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_IMEM_VER_6 PMU_MAX_ATTACHED_OVLS_IMEM_VER_1

#define PMU_MAX_ATTACHED_OVLS_DMEM_VER_0 16
#define PMU_MAX_ATTACHED_OVLS_DMEM_VER_1 32
#define PMU_MAX_ATTACHED_OVLS_DMEM_VER_2 PMU_MAX_ATTACHED_OVLS_DMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_DMEM_VER_3 PMU_MAX_ATTACHED_OVLS_DMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_DMEM_VER_4 PMU_MAX_ATTACHED_OVLS_DMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_DMEM_VER_5 PMU_MAX_ATTACHED_OVLS_DMEM_VER_1
#define PMU_MAX_ATTACHED_OVLS_DMEM_VER_6 PMU_MAX_ATTACHED_OVLS_DMEM_VER_1

typedef enum
{
    PMU_TCB_VER_0 = 0,
    PMU_TCB_VER_1 = 1,
    PMU_TCB_VER_2 = 2,
    PMU_TCB_VER_3 = 3,
    PMU_TCB_VER_4 = 4,
    PMU_TCB_VER_5 = 5
} PMU_TCB_VER;

typedef enum
{
    PMU_TCB_PVT_VER_0       = 0, // version snapped 12/04/2015
    PMU_TCB_PVT_VER_1       = 1, // version snapped 12/05/2015
    PMU_TCB_PVT_VER_2       = 2, // version snapped 03/2016
    PMU_TCB_PVT_VER_3       = 3, // version snapped 07/2016
    PMU_TCB_PVT_VER_4       = 4, // version snapped 11/2016
    PMU_TCB_PVT_VER_5       = 5, // version snapped 02/2018
    PMU_TCB_PVT_VER_6       = 6, // version snapped 10/2018
    PMU_TCB_PVT_VER_COUNT   = 7
} PMU_TCB_PVT_VER;

/*
 * This structure contains a union of all the PVT TCB versions
 * that have existed within RM. The latest PVT TCB version must
 * match the definition inside lw_rtos_extension.h.
 */
typedef union
{
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   pData;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwU8    privilegeLevel;
        LwU8    ovlCnt;
        LwU8    ovlList[0];
    } pmuTcbPvt0;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwU8    privilegeLevel;
        LwU8    ovlCntImem;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } pmuTcbPvt1;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImem;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } pmuTcbPvt2;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } pmuTcbPvt3;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   spMin;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } pmuTcbPvt4;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   spMin;
        LwU32   pRunTimeStats;
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } pmuTcbPvt5;
    struct
    {
        LwU32   pPrivTcbNext;
        LwU32   pStack;
        LwU32   spMin;
        LwU32   pRunTimeStats;
        LwU32   stackCanary;    // New field in version 6
        LwU16   stackSize;
        LwU16   usedHeap;
        LwU8    taskID;
        LwBool  bReload;
        LwBool  bPagingOnDemandImem;
        LwU8    privilegeLevel;
        LwU8    ovlIdxStack;
        LwU8    ovlCntImemLS;
        LwU8    ovlCntImemHS;
        LwU8    ovlCntDmem;
        LwU8    ovlList[0];
    } pmuTcbPvt6;
} PMU_TCB_PVT_INT;

typedef struct
{
    LwU32           tcbPvtAddr;
    PMU_TCB_PVT_VER tcbPvtVer;
    PMU_TCB_PVT_INT pmuTcbPvt;
} PMU_TCB_PVT;

typedef struct
{
    PMU_TCB_VER tcbVer;
    LwU32       tcbAddr;

    union
    {
        // The old tcb structure before we added private tcb field to it
        struct
        {
            LwU32                 pTopOfStack;
            LwU32                 priority;
            LwU32                 pStack;
            LwU32                 tcbNumber;
            char                  taskName[8];
            LwU16                 stackDepth;
            LwU32                 address;
        } pmuTcb0;

        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            PMU_XLIST_ITEM        xGenericListItem;
            PMU_XLIST_ITEM        xEventListItem;
            LwU32                 uxPriority;
            LwU8                  ucTaskID;
        } pmuTcb1;

        // SafeRTOSv5.10.1-lw1.1 Falcon
        // SafeRTOSv5.16.0-lw1.2 Falcon
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 uxTopOfStackMirror;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU32                 uxNotifiedValue;
            LwU32                 xNotifyState;
            LwU8                  ucPriority;
        } pmuTcb2;

        // SafeRTOSv5.16.0-lw1.2 Falcon after adding pcStackBaseAddress (pStack in RM_RTOS_TCB_PVT)
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 pcStackBaseAddress;
            LwU32                 uxTopOfStackMirror;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU32                 uxNotifiedValue;
            LwU32                 xNotifyState;
            LwU8                  ucPriority;
        } pmuTcb3;

        // OpenRTOS Falcon after adding pcStackBaseAddress (pStack in RM_RTOS_TCB_PVT)
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 pcStackBaseAddress;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU32                 uxPriority;
            LwU8                  ucTaskID;
        } pmuTcb4;

        // SafeRTOSv5.16.0-lw1.2 Falcon after removing task notifications.
        // Same for SafeRTOSv5.16.0-lw1.3.
        struct
        {
            LwU32                 pxTopOfStack;
            LwU32                 pvTcbPvt;
            LwU32                 pcStackBaseAddress;
            LwU32                 uxTopOfStackMirror;
            FLCN_RTOS_XLIST_ITEM  xGenericListItem;
            FLCN_RTOS_XLIST_ITEM  xEventListItem;
            LwU8                  ucPriority;
        } pmuTcb5;
    } pmuTcb;
} PMU_TCB;

typedef struct PMU_XQUEUE
{
    LwU32     head;
    LwU32     tail;
    LwU32     writeTo;
    LwU32     readFrom;
    PMU_XLIST xTasksWaitingToSend;
    PMU_XLIST xTasksWaitingToReceive;
    LwU32     messagesWaiting;
    LwU32     length;
    LwU32     itemSize;
    LwS32     rxLock;
    LwS32     txLock;
    LwU32     next;
} PMU_XQUEUE;

/** @struct PmuSanityTestEntry
 *
 *  Declares each PMU sanity test case. Consists of a function pointer and
 *  description.
 **/
struct PmuSanityTestEntry
{
    PmuTestFunc       fnPtr;
    LwU32             flags;
    const char* const fnInfo;
};

typedef struct PmuFuseBinaryDesc
{
    LwU32  *pUcodeData;
    LwU32  *pUcodeHeader;
} PmuFuseBinaryDesc;

typedef struct PMU_SYM   PMU_SYM;
typedef struct PMU_SYM *PPMU_SYM;

#define PMU_SYM_NAME_SIZE 128
struct PMU_SYM
{
    LwU32      addr;
    LwU32      size;
    char       section;
    char       name[PMU_SYM_NAME_SIZE];
    char       nameUpper[PMU_SYM_NAME_SIZE];
    BOOL       bData;
    BOOL       bSizeEstimated;
    PMU_SYM   *pNext;
    PMU_SYM   *pTemp;
};

//
// Common Non-HAL Functions
//
void        pmuStrToUpper             (char *pDst, const char *pSrc);
void        pmuStrLwtRegion           (char **ppStr, LwU32 offs, LwU32 len);
void        pmuStrTrim                (char **ppStr);
void        pmuDmemDump               (LwU32, LwU32, LwU8, LwU8);
BOOL        pmuTcbGetLwrrent          (PMU_TCB *, LwU32);
LwBool      pmuTcbFetchAll            (LwU32, PMU_TCB **, LwU32 *);
BOOL        pmuTcbGetPriv             (PMU_TCB_PVT **, LwU32, LwU32);
LwU32       pmuTcbValidate            (PMU_TCB *pTcb);
void        pmuTcbDump                (PMU_TCB *, BOOL, LwU32, LwU8);
void        pmuTcbDumpAll             (BOOL);
void        pmuGetTaskNameFromTcb     (PMU_TCB *, char *, LwU32, LwU32);
void        pmuSimpleBootstrap        (const char *);
void        pmuSchedDump              (BOOL);
LwBool      pmuQueueFetchAll          (PMU_XQUEUE **, LwU32 **, char **, LwU32 *, LwU32 *);
void        pmuEventQueueDumpAll      (LwBool);
void        pmuEventQueueDumpByAddr   (LwU32);
void        pmuEventQueueDumpBySymbol (const char *);
void        pmuImemMapDump            (LwU32, LwU32, BOOL);

BOOL        pmustLoad                 (const char *, LwU32, BOOL);
BOOL        pmustUnload               (void);
void        pmustPrintStacktrace      (LwU32, LwU32, LwU32);
void        pmustStacktraceForTasks   (LwU8 *, LwU32, LwU32, LwS32, LwU32, LwU32);
void        pmuswakExec               (const char *, LwBool);
const char *pmuGetTaskName            (LwU32);

void        pmuSymDump                (const char *, BOOL);
BOOL        pmuSymLwsymFileLoad       (const char *, LwU8 **, LwU32 *, BOOL);
PMU_SYM    *pmuSymResolve             (LwU32);
void        pmuSymLoad                (const char *, LwU32);
void        pmuSymUnload              (void);
void        pmuSymPrintBrief          (PMU_SYM *, LwU32);
PMU_SYM    *pmuSymFind                (const char *, BOOL, BOOL *, LwU32 *);
BOOL        pmuSymCheckIfLoaded       (void);
BOOL        pmuSymCheckAutoLoad       (void);

void        pmuExec                   (char *);

POBJFLCN    pmuGetFalconObject        (void);

const char* pmuGetSymFilePath         (void);
const char* pmuGetEngineName          (void);
LwU32       pmuGetDmemAccessPort      (void);

LW_STATUS pmuAcquireMutex(LwU32 mutexIndex, LwU16  retryCount, LwU32 *pToken);
LW_STATUS pmuReleaseMutex(LwU32 mutexIndex, LwU32 *pToken);

//
// PMU SANITY TEST HEADERS
//

// GK104
LW_STATUS pmuSanityTest_CheckImage_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_Reset_GK104        (LwU32, char *);
LW_STATUS pmuSanityTest_Latency_GK104      (LwU32, char *);
LW_STATUS pmuSanityTest_MutexIdGen_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_PBI_GK104          (LwU32, char *);
LW_STATUS pmuSanityTest_Bar0Master_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_Bar0FECS_GK104     (LwU32, char *);
LW_STATUS pmuSanityTest_PbiInterface_GK104 (LwU32, char *);
LW_STATUS pmuSanityTest_GPTMR_GK104        (LwU32, char *);
LW_STATUS pmuSanityTest_Vblank_GK104       (LwU32, char *);
LW_STATUS pmuSanityTest_ScanlineIO_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_ScanlineIntr_GK104 (LwU32, char *);

// GP100
LW_STATUS pmuSanityTest_Latency_GP100      (LwU32, char *);

#include "g_pmu_hal.h"     // (rmconfig)  public interfaces


#endif // _PMU_H_
