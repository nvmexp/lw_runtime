/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _PMU_H_
#define _PMU_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
//#include "mmu.h"

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
    ((pPmu[indexGpu].pmuDmemRead(ptr, 1, port, pVal)) == 1 ? TRUE : FALSE)

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
    U032 tag;                   /**< IMEM tag of this block */
    U032 blockIndex;            /**< Index of block */
    U032 bValid   :1;           /**< Is the block valid? */
    U032 bPending :1;           /**< Is the block pending? */
    U032 bSelwre  :1;           /**< Is the block secure? */
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
    U032                 pTopOfStack;
    U032                 priority;
    U032                 pStack;
    U032                 tcbNumber;
    char                 taskName[8];
    U016                 stackDepth;
    U032                 address;
} PMU_TCB;

typedef struct
{
    U032                 itemValue;
    U032                 next;
    U032                 prev;
    U032                 owner;
    U032                 container;
} PMU_XLIST_ITEM;

typedef struct
{
    U032                 itemValue;
    U032                 next;
    U032                 prev;
} PMU_XMINI_LIST_ITEM;

typedef struct
{
    U032                 numItems;
    U032                 pIndex;
    PMU_XMINI_LIST_ITEM  listEnd;
} PMU_XLIST;

typedef struct PMU_XQUEUE
{
    U032      head;
    U032      tail;
    U032      writeTo;
    U032      readFrom;
    PMU_XLIST xTasksWaitingToSend;
    PMU_XLIST xTasksWaitingToReceive;
    U032      messagesWaiting;
    U032      length;
    U032      itemSize;
    S032      rxLock;
    S032      txLock;
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

struct PMU_SYM
{
    U032       addr;
    U032       size;
    char       section;
    char       name[41];
    char       nameUpper[41];
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
VOID        pmuDmemDump               (U032, U032, U008, U008);
BOOL        pmuTcbGetLwrrent          (PMU_TCB *, U032);
VOID        pmuTcbDump                (PMU_TCB *, BOOL, U032, U008);
VOID        pmuTcbDumpAll             (BOOL);
VOID        pmuSimpleBootstrap        (const char *);
VOID        pmuSchedDump              (BOOL);
VOID        pmuEventQueueDumpAll      (BOOL);
VOID        pmuEventQueueDumpByAddr   (U032);
VOID        pmuEventQueueDumpBySymbol (const char *);
VOID        pmuImemMapDump            (U032, U032, BOOL);

VOID        pmuSymDump                (const char *, BOOL);
PMU_SYM    *pmuSymResolve             (U032);
VOID        pmuSymLoad                (const char *, LwU32);
VOID        pmuSymUnload              (void);
VOID        pmuSymPrintBrief          (PMU_SYM *, U032);
PMU_SYM    *pmuSymFind                (const char *, BOOL, BOOL *, U032 *);
BOOL        pmuSymCheckIfLoaded       (void);
BOOL        pmuSymCheckAutoLoad       (void);

VOID        pmuExec                   (char *);


const char* pmuGetSymFilePath         (void);
const char* pmuGetEngineName          (void);

LW_STATUS pmuAcquireMutex(LwU32 mutexIndex, LwU16  retryCount, LwU32 *pToken);
LW_STATUS pmuReleaseMutex(LwU32 mutexIndex, LwU32 *pToken);

//
// PMU SANITY TEST HEADERS
//

// GT215
LwU32 pmuSanityTest_CheckImage_GT215   (LwU32, char *);

// GF100
LwU32 pmuSanityTest_Reset_GF100        (LwU32, char *);
LwU32 pmuSanityTest_Latency_GF100      (LwU32, char *);
LwU32 pmuSanityTest_MutexIdGen_GF100   (LwU32, char *);
LwU32 pmuSanityTest_PBI_GF100          (LwU32, char *);
LwU32 pmuSanityTest_Bar0Master_GF100   (LwU32, char *);
LwU32 pmuSanityTest_PbiInterface_GF100 (LwU32, char *);
LwU32 pmuSanityTest_GPTMR_GF100        (LwU32, char *);
LwU32 pmuSanityTest_Vblank_GF100       (LwU32, char *);
LwU32 pmuSanityTest_ScanlineIO_GF100   (LwU32, char *);
LwU32 pmuSanityTest_ScanlineIntr_GF100 (LwU32, char *);

// GF119
LwU32 pmuSanityTest_Bar0FECS_GF119     (LwU32, char *);

#include "g_pmu_hal.h"     // (rmconfig)  public interfaces


#endif // _PMU_H_
