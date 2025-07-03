/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FALCON_H_
#define _FALCON_H_

/* ------------------------ Types definitions ------------------------------ */
typedef struct _objectFlcn          OBJFLCN, *POBJFLCN;
typedef struct _flcnSymbol          FLCN_SYM, *PFLCN_SYM;
typedef struct _flcnQueue           FLCN_QUEUE, *PFLCN_QUEUE;
typedef struct _flcnCoreIFaces      FLCN_CORE_IFACES;
typedef struct _flcnEngineIFaces    FLCN_ENGINE_IFACES;
typedef struct _flcnBlock           FLCN_BLOCK, *PFLCN_BLOCK;
typedef enum   _flcnTagMapping      FLCN_TAG_MAPPING;
typedef struct _flcnTag             FLCN_TAG, *PFLCN_TAG;
typedef struct _flcnFunc            FLCN_FUNC;


/* ------------------------ Includes --------------------------------------- */

#include "flcngdb.h"

#include "g_falcon_hal.h"     // (rmconfig) public interface

#include "flcnrtos.h"

#include "chip.h"

/* ------------------------ Common Defines --------------------------------- */
void flcnRegWr32(LwU32 addr, LwU32 value, LwU32 engineBase);
LwU32 flcnRegRd32(LwU32 addr, LwU32 engineBase);

#define FLCN_REG_WR32(addr, value)  flcnRegWr32((addr), (value), engineBase)
#define FLCN_REG_RD32(addr)         flcnRegRd32((addr), engineBase)

#ifndef LW_FLCN_REG_R0
#define LW_FLCN_REG_R0                       (0)
#define LW_FLCN_REG_R1                       (1)
#define LW_FLCN_REG_R2                       (2)
#define LW_FLCN_REG_R3                       (3)
#define LW_FLCN_REG_R4                       (4)
#define LW_FLCN_REG_R5                       (5)
#define LW_FLCN_REG_R6                       (6)
#define LW_FLCN_REG_R7                       (7)
#define LW_FLCN_REG_R8                       (8)
#define LW_FLCN_REG_R9                       (9)
#define LW_FLCN_REG_R10                      (10)
#define LW_FLCN_REG_R11                      (11)
#define LW_FLCN_REG_R12                      (12)
#define LW_FLCN_REG_R13                      (13)
#define LW_FLCN_REG_R14                      (14)
#define LW_FLCN_REG_R15                      (15)
#define LW_FLCN_REG_IV0                      (16)
#define LW_FLCN_REG_IV1                      (17)
#define LW_FLCN_REG_UNDEFINED                (18)
#define LW_FLCN_REG_EV                       (19)
#define LW_FLCN_REG_SP                       (20)
#define LW_FLCN_REG_PC                       (21)
#define LW_FLCN_REG_IMB                      (22)
#define LW_FLCN_REG_DMB                      (23)
#define LW_FLCN_REG_CSW                      (24)
#define LW_FLCN_REG_CCR                      (25)
#define LW_FLCN_REG_SEC                      (26)
#define LW_FLCN_REG_CTX                      (27)
#define LW_FLCN_REG_EXCI                     (28)
#define LW_FLCN_REG_RSVD0                    (29)
#define LW_FLCN_REG_RSVD1                    (30)
#define LW_FLCN_REG_RSVD2                    (31)
#define LW_FLCN_REG_SIZE                     (32)
#endif  // LW_FLCN_REG_R0

// Falcon IMSTAT bit field
#define LW_PFALCON_FALCON_IMBLK_VALID      24:24  /** As defined by Falcon Architecture (core 3) */
#define LW_PFALCON_FALCON_IMBLK_PENDING    25:25
#define LW_PFALCON_FALCON_IMBLK_SELWRE     26:26
#define LW_PFALCON_FALCON_IMTAG_VALID      LW_PFALCON_FALCON_IMBLK_VALID
#define LW_PFALCON_FALCON_IMTAG_PENDING    LW_PFALCON_FALCON_IMBLK_PENDING
#define LW_PFALCON_FALCON_IMTAG_SELWRE     LW_PFALCON_FALCON_IMBLK_SELWRE
#define LW_PFALCON_FALCON_IMTAG_MULTI_HIT  30:30
#define LW_PFALCON_FALCON_IMTAG_MISS       31:31


// Falcon DMSTAT bit field
#define LW_PFALCON_FALCON_DMTAG_MISS       20:20  /** As defined by Falcon Architecture (core 6) */
#define LW_PFALCON_FALCON_DMTAG_MULTI_HIT  21:21
#define LW_PFALCON_FALCON_DMBLK_SELWRE     26:24
#define LW_PFALCON_FALCON_DMBLK_VALID      28:28
#define LW_PFALCON_FALCON_DMBLK_PENDING    29:29
#define LW_PFALCON_FALCON_DMTAG_VALID      LW_PFALCON_FALCON_DMBLK_VALID
#define LW_PFALCON_FALCON_DMTAG_PENDING    LW_PFALCON_FALCON_DMBLK_PENDING
#define LW_PFALCON_FALCON_DMTAG_SELWRE     LW_PFALCON_FALCON_DMBLK_SELWRE
#define LW_PFALCON_FALCON_DMTAG_DIRTY      30:30


/*!
 * Defines the Falcon DMEM block-size (as a power-of-2).
 */
#define FALCON_DMEM_BLKSIZE                (8)

/*!
 * Defines the alignment/granularity of falcon memory blocks
 */
#define FLCN_BLK_ALIGNMENT (1 << FALCON_DMEM_BLKSIZE)

/*!
 * Obtain the DMEM block from the DMEM address.
 */
#define FLCN_DMEM_ADDR_TO_BLK(addr)        (addr >> FALCON_DMEM_BLKSIZE)

/*!
 * Obtain the DMEM address from the DMEM block.
 */
#define FLCN_DMEM_BLK_TO_ADDR(blk)         (blk << FALCON_DMEM_BLKSIZE)

/*!
 * Define to indicate that DMEM_PRIV_RANGE0/1 is invalid.
 */
#define FALCON_DMEM_PRIV_RANGE_ILWALID (0xFFFFFFFF)

/*!
 * Usage message for FLCN engine commands
 */
#define FLCN_PRINT_USAGE_MESSAGE()                                      \
        dprintf("lw: Avaiable commands:\n");                            \
        dprintf("lw: dd        dw        db        r       import\n");  \
        dprintf("lw: sympath   load      unload    x       imemrd\n");  \
        dprintf("lw: imemwr    queues    imtag     imblk   st\n");      \
        dprintf("lw: tcb       sched     evtq      dmemwr  ememrd\n");  \
        dprintf("lw: ememwr    dmemrd    dmtag     dmblk   dmemovl\n"); \
        dprintf("lw: tracepc   ldobjdump\n");

/* ---------------- Prototypes - Falcon Core HAL Interface ------------------ */
/*
 *  Prototypes in "g_falcon_hal.h"
 */
FalconDmemGetSize          flcnDmemGetSize_v04_00;
FalconDmemGetNumPorts      flcnDmemGetNumPorts_v04_00;
FalconDmemRead             flcnDmemRead_v04_00;
FalconDmemRead             flcnDmemRead_v06_00;
FalconDmemWrite            flcnDmemWrite_v04_00;
FalconDmemWrite            flcnDmemWrite_v06_00;
FalconImemGetNumBlocks     flcnImemGetNumBlocks_v04_00;
FalconImemGetSize          flcnImemGetSize_v04_00;
FalconImemGetNumPorts      flcnImemGetNumPorts_v04_00;
FalconImemRead             flcnImemRead_v04_00;
FalconImemRead             flcnImemRead_v05_01; 
FalconImemWrite            flcnImemWrite_v04_00;
FalconImemWrite            flcnImemWrite_v05_01;
FalconImemWriteBuf         flcnImemWriteBuf_v04_00;
FalconWaitForHalt          flcnWaitForHalt_v04_00;
FalconDmemWriteBuf         flcnDmemWriteBuf_v04_00;
FalconImemGetTagWidth      flcnImemGetTagWidth_v04_00;
FalconDmemGetTagWidth      flcnDmemGetTagWidth_v06_00;
FalconDmemGetTagWidth      flcnDmemGetTagWidth_STUB;
FalconImemBlk              flcnImemBlk_v04_00;
FalconImemBlk              flcnImemBlk_v05_01;
FalconImemTag              flcnImemTag_v04_00;
FalconImemTag              flcnImemTag_v05_01;
FalconImemSetTag           flcnImemSetTag_v04_00;
FalconGetRegister          flcnGetRegister_v04_00;
FalconUcodeGetVersion      flcnUcodeGetVersion_v04_00;
FalconBootstrap            flcnBootstrap_v04_00;
FalconIsDmemAccessAllowed  flcnIsDmemAccessAllowed_STUB;
FalconIsDmemAccessAllowed  flcnIsDmemAccessAllowed_v05_01;
FalconGetTasknameFromId    flcnGetTasknameFromId_STUB;
FalconDmemBlk              flcnDmemBlk_v06_00;
FalconDmemTag              flcnDmemTag_v06_00;
FalconDmemBlk              flcnDmemBlk_STUB;
FalconDmemTag              flcnDmemTag_STUB;
FalconDmemVaBoundaryGet    flcnDmemVaBoundaryGet_v06_00;
FalconDmemVaBoundaryGet    flcnDmemVaBoundaryGet_STUB;

/* ---------------- Prototypes - Falcon Engine HAL Interface ---------------- */
typedef const FLCN_CORE_IFACES *  FlcnEngGetCoreIFace(void);
typedef LwU32                     FlcnEngGetFalconBase(void);
typedef LwU32                     FlcnEngQueueGetNum(void);
typedef LwBool                    FlcnEngQueueRead(LwU32 queueId, PFLCN_QUEUE pQueue);
typedef const char *              FlcnEngGetSymFilePath(void);
typedef const char *              FlcnEngUcodeName(void);
typedef const char *              FlcnEngGetEngineName(void);
typedef LwU32                     FlcnEngGetDmemAccessPort(void);
typedef LwBool                    FlcnEngIsDmemRangeAccessible(LwU32 blkLo, LwU32 blkHi);
typedef const char *              FlcnEngGetTasknameFromId(LwU32 taskId);
typedef LwU32                     FlcnEngEmemGetSize(void);
typedef LwU32                     FlcnEngEmemGetOffsetInDmemVaSpace(void);
typedef LwU32                     FlcnEngEmemGetNumPorts(void);
typedef LwU32                     FlcnEngEmemRead(LwU32 offset, LwU32 length, LwU32 port, LwU32 *pBuf);
typedef LwU32                     FlcnEngEmemWrite(LwU32 offset, LwU32 val, LwU32 width, LwU32 length, LwU32 port);

/* ------------------------ Hal Interface  --------------------------------- */

/*
 *  The Falcon Interface
 *
 *    FLCN_CORE_IFACES
 *        Falcon Core Interface is the collection of falcon HAL functions.
 *    Those functions are defined in falcon.def and included in this table. 
 *    Eash Falcon engine has the knowledge about which core version it uses
 *    and provides this information via Engine specific function 
 *    FlcnEngGetCoreIFace().  
 *
 *
 *    FLCN_ENGINE_IFACES
 *        Falcon Engine Interface is the collection of falcon Engine specific
 *    functions.  Each Falcon engine should implement its own Engine Specific
 *    functions.  It could be Hal functions defined in xxx.def (Ex. pmu.def) 
 *    or non-Hal object functions.
 *
 *
 *    Non-Hal Falcon common functions are declared in falcon.h and implemented
 *    in falcon.c. 
 * 
 */


/* 
 *  FALCON Core Interfaces
 *  
 *  Those are Falcon Hal functions that should be defined in falcon.def. 
 */
struct _flcnCoreIFaces {
    FalconDmemGetSize          *flcnDmemGetSize;
    FalconDmemGetNumPorts      *flcnDmemGetNumPorts;
    FalconDmemRead             *flcnDmemRead;
    FalconDmemWrite            *flcnDmemWrite;
    FalconImemGetNumBlocks     *flcnImemGetNumBlocks;
    FalconImemGetSize          *flcnImemGetSize;
    FalconImemGetNumPorts      *flcnImemGetNumPorts;
    FalconImemRead             *flcnImemRead;    
    FalconImemWrite            *flcnImemWrite;
    FalconImemWriteBuf         *flcnImemWriteBuf;
    FalconWaitForHalt          *flcnWaitForHalt;
    FalconDmemWriteBuf         *flcnDmemWriteBuf;
    FalconImemGetTagWidth      *flcnImemGetTagWidth;
    FalconDmemGetTagWidth      *flcnDmemGetTagWidth;
    FalconImemBlk              *flcnImemBlk;
    FalconImemTag              *flcnImemTag;
    FalconImemSetTag           *flcnImemSetTag;
    FalconGetRegister          *flcnGetRegister;
    FalconUcodeGetVersion      *flcnUcodeGetVersion;
    FalconBootstrap            *flcnBootstrap;
    FalconIsDmemAccessAllowed  *flcnIsDmemAccessAllowed;
    FalconGetTasknameFromId    *flcnGetTasknameFromId;
    FalconDmemBlk              *flcnDmemBlk;
    FalconDmemTag              *flcnDmemTag;
    FalconDmemVaBoundaryGet    *flcnDmemVaBoundaryGet;
}; 


/* 
 *  FALCON Engine Interfaces
 */
struct _flcnEngineIFaces {
    FlcnEngGetCoreIFace               *flcnEngGetCoreIFace;
    FlcnEngGetFalconBase              *flcnEngGetFalconBase;
    FlcnEngGetEngineName              *flcnEngGetEngineName;
    FlcnEngUcodeName                  *flcnEngUcodeName;
    FlcnEngGetSymFilePath             *flcnEngGetSymFilePath;
    FlcnEngQueueGetNum                *flcnEngQueueGetNum;
    FlcnEngQueueRead                  *flcnEngQueueRead;
    FlcnEngGetDmemAccessPort          *flcnEngGetDmemAccessPort;
    FlcnEngIsDmemRangeAccessible      *flcnEngIsDmemRangeAccessible;
    FlcnEngEmemGetSize                *flcnEngEmemGetSize;
    FlcnEngEmemGetOffsetInDmemVaSpace *flcnEngEmemGetOffsetInDmemVaSpace;
    FlcnEngEmemGetNumPorts            *flcnEngEmemGetNumPorts;
    FlcnEngEmemRead                   *flcnEngEmemRead;
    FlcnEngEmemWrite                  *flcnEngEmemWrite;
};

/* ------------------------ Static variables ------------------------------- */

static const FLCN_CORE_IFACES flcnCoreIfaces_v04_00 =
{
    flcnDmemGetSize_v04_00,                  // flcnDmemGetSize
    flcnDmemGetNumPorts_v04_00,              // flcnDmemGetNumPorts
    flcnDmemRead_v04_00,                     // flcnDmemRead
    flcnDmemWrite_v04_00,                    // flcnDmemWrite
    flcnImemGetNumBlocks_v04_00,             // flcnImemGetNumBlocks
    flcnImemGetSize_v04_00,                  // flcnImemGetSize
    flcnImemGetNumPorts_v04_00,              // flcnImemGetNumPorts
    flcnImemRead_v04_00,                     // flcnImemRead
    flcnImemWrite_v04_00,                    // flcnImemWrite
    flcnImemWriteBuf_v04_00,                 // flcnImemWriteBuf
    flcnWaitForHalt_v04_00,                  // flcnWaitForHalt
    flcnDmemWriteBuf_v04_00,                 // flcnDmemWriteBuf
    flcnImemGetTagWidth_v04_00,              // flcnImemGetTagWidth
    flcnDmemGetTagWidth_STUB,                // flcnDmemGetTagWidth
    flcnImemBlk_v04_00,                      // flcnImemBlk
    flcnImemTag_v04_00,                      // flcnImemTag
    flcnImemSetTag_v04_00,                   // flcnImemSetTag
    flcnGetRegister_v04_00,                  // flcnGetRegister
    flcnUcodeGetVersion_v04_00,              // flcnUcodeGetVersion
    flcnBootstrap_v04_00,                    // flcnBootstrap
    flcnIsDmemAccessAllowed_STUB,            // flcnIsDmemAccessAllowed
    flcnGetTasknameFromId_STUB,              // flcnGetTasknameFromId
    flcnDmemBlk_STUB,                        // flcnDmemBlk
    flcnDmemTag_STUB,                        // flcnDmemTag
    flcnDmemVaBoundaryGet_STUB,                 // flcnDmemVaBoundaryGet
};  // flcnCoreIfaces_v04_00

static const FLCN_CORE_IFACES flcnCoreIfaces_v05_01 =
{
    flcnDmemGetSize_v04_00,                  // flcnDmemGetSize
    flcnDmemGetNumPorts_v04_00,              // flcnDmemGetNumPorts
    flcnDmemRead_v04_00,                     // flcnDmemRead
    flcnDmemWrite_v04_00,                    // flcnDmemWrite
    flcnImemGetNumBlocks_v04_00,             // flcnImemGetNumBlocks
    flcnImemGetSize_v04_00,                  // flcnImemGetSize
    flcnImemGetNumPorts_v04_00,              // flcnImemGetNumPorts
    flcnImemRead_v05_01,                     // flcnImemRead
    flcnImemWrite_v05_01,                    // flcnImemWrite
    flcnImemWriteBuf_v04_00,                 // flcnImemWriteBuf
    flcnWaitForHalt_v04_00,                  // flcnWaitForHalt
    flcnDmemWriteBuf_v04_00,                 // flcnDmemWriteBuf
    flcnImemGetTagWidth_v04_00,              // flcnImemGetTagWidth
    flcnDmemGetTagWidth_STUB,                // flcnDmemGetTagWidth
    flcnImemBlk_v05_01,                      // flcnImemBlk
    flcnImemTag_v05_01,                      // flcnImemTag
    flcnImemSetTag_v04_00,                   // flcnImemSetTag
    flcnGetRegister_v04_00,                  // flcnGetRegister
    flcnUcodeGetVersion_v04_00,              // flcnUcodeGetVersion
    flcnBootstrap_v04_00,                    // flcnBootstrap
    flcnIsDmemAccessAllowed_v05_01,          // flcnIsDmemAccessAllowed
    flcnGetTasknameFromId_STUB,              // flcnGetTasknameFromId
    flcnDmemBlk_STUB,                        // flcnDmemBlk
    flcnDmemTag_STUB,                        // flcnDmemTag
    flcnDmemVaBoundaryGet_STUB,                 // flcnDmemVaBoundaryGet
};  // flcnCoreIfaces_v05_01

static const FLCN_CORE_IFACES flcnCoreIfaces_v06_00 =
{
    flcnDmemGetSize_v04_00,                  // flcnDmemGetSize
    flcnDmemGetNumPorts_v04_00,              // flcnDmemGetNumPorts
    flcnDmemRead_v06_00,                     // flcnDmemRead
    flcnDmemWrite_v06_00,                    // flcnDmemWrite
    flcnImemGetNumBlocks_v04_00,             // flcnImemGetNumBlocks
    flcnImemGetSize_v04_00,                  // flcnImemGetSize
    flcnImemGetNumPorts_v04_00,              // flcnImemGetNumPorts
    flcnImemRead_v05_01,                     // flcnImemRead
    flcnImemWrite_v05_01,                    // flcnImemWrite
    flcnImemWriteBuf_v04_00,                 // flcnImemWriteBuf
    flcnWaitForHalt_v04_00,                  // flcnWaitForHalt
    flcnDmemWriteBuf_v04_00,                 // flcnDmemWriteBuf
    flcnImemGetTagWidth_v04_00,              // flcnImemGetTagWidth
    flcnDmemGetTagWidth_v06_00,              // flcnDmemGetTagWidth
    flcnImemBlk_v05_01,                      // flcnImemBlk
    flcnImemTag_v05_01,                      // flcnImemTag
    flcnImemSetTag_v04_00,                   // flcnImemSetTag
    flcnGetRegister_v04_00,                  // flcnGetRegister
    flcnUcodeGetVersion_v04_00,              // flcnUcodeGetVersion
    flcnBootstrap_v04_00,                    // flcnBootstrap
    flcnIsDmemAccessAllowed_v05_01,          // flcnIsDmemAccessAllowed
    flcnGetTasknameFromId_STUB,              // flcnGetTasknameFromId
    flcnDmemBlk_v06_00,                      // flcnDmemBlk
    flcnDmemTag_v06_00,                      // flcnDmemTag
    flcnDmemVaBoundaryGet_v06_00,               // flcnDmemVaBoundaryGet
};  // flcnCoreIfaces_v06_00

typedef struct flcnRegInfo
{
    LwU32  regIdx;                        // Index of Falcon internal register, LW_FLCN_REG_*
    char  *pName;                         // String of the Register Name
}FLCN_REG_INFO, *PFLCN_REG_INFO;

static const FLCN_REG_INFO FLCN_REG_TABLE[] =
{
    {LW_FLCN_REG_R0   , "r0"  },
    {LW_FLCN_REG_R1   , "r1"  },
    {LW_FLCN_REG_R2   , "r2"  },
    {LW_FLCN_REG_R3   , "r3"  },
    {LW_FLCN_REG_R4   , "r4"  },
    {LW_FLCN_REG_R5   , "r5"  },
    {LW_FLCN_REG_R6   , "r6"  },
    {LW_FLCN_REG_R7   , "r7"  },
    {LW_FLCN_REG_R8   , "r8"  },
    {LW_FLCN_REG_R9   , "r9"  },
    {LW_FLCN_REG_R10  , "r10" },
    {LW_FLCN_REG_R11  , "r11" },
    {LW_FLCN_REG_R12  , "r12" },
    {LW_FLCN_REG_R13  , "r13" },
    {LW_FLCN_REG_R14  , "r14" },
    {LW_FLCN_REG_R15  , "r15" },
    {LW_FLCN_REG_IV0  , "iv0" },
    {LW_FLCN_REG_IV1  , "iv1" },
    {LW_FLCN_REG_EV   , "ev"  },
    {LW_FLCN_REG_IMB  , "imb" },
    {LW_FLCN_REG_DMB  , "dmb" },
    {LW_FLCN_REG_CSW  , "csw" },
    {LW_FLCN_REG_CCR  , "ccr" },
    {LW_FLCN_REG_SEC  , "sec" },
    {LW_FLCN_REG_CTX  , "ctx" },
    {LW_FLCN_REG_EXCI , "exci"},
};

/* ------------------------ Non-HAL Function - falcon.c  -------------------- */
void        flcnQueueDump             (LwBool, PFLCN_QUEUE, const char*);
void        flcnDmemDump              (const FLCN_ENGINE_IFACES*, LwU32, LwU32, LwU8, LwU8);
void        flcnDmemWrWrapper         (const FLCN_ENGINE_IFACES*, LwU32, LwU32, LwU32, LwU32, LwU8);
void        flcnImemMapDump           (LwU32, LwU32, BOOL);
void        flcnSimpleBootstrap       (const FLCN_ENGINE_IFACES*, const char*);
void        flcnImemDump              (const FLCN_ENGINE_IFACES*, LwU32, LwU32, LwU8);
void        flcnTrpcClear             (LwBool, LwBool);

/* ------------------------ Non-HAL Function - flcndbg.c  ------------------- */
void        flcnExec                    (char *, POBJFLCN);
void        flcngdbMenu                 (char *sessionID, char *pSymbPath);

/* ------------------------ Non-HAL Function - flcnsym.c  ------------------- */
void        flcnSymDump                 (const char *, LwBool);
PFLCN_SYM   flcnSymResolve              (LwU32);
void        flcnSymLoad                 (const char *, LwU32);
void        flcnSymUnload               (void);
void        flcnSymPrintBrief           (PFLCN_SYM, LwU32);
PFLCN_SYM   flcnSymFind                 (const char *, LwBool, LwBool *, LwU32 *);
LwBool      flcnSymCheckIfLoaded        (void);
LwBool      flcnSymCheckAutoLoad        (void);
BOOL        flcnstLoad                  (const char *, LwU32, BOOL);
BOOL        flcnstUnload                (void);
BOOL        flcnTcbGetPriv              (FLCN_TCB_PVT **, LwU32, LwU32);
void        flcnstPrintStacktrace       (LwU32, LwU32, LwU32);

/* ------------------------ Function Specific Defines ---------------------- */
#define LW_FLCN_MAX_QUEUE_SIZE          0x20    // 128 bytes (0x20 4-byte words)

#define LW_FLCN_DMEM_VA_BOUND_NONE      0xFFFFFF    // 24bit DMEM upper limit

/*
 *  The Falcon object
 */
struct _objectFlcn
{
    const FLCN_CORE_IFACES *pFCIF;
    const FLCN_ENGINE_IFACES *pFEIF;
    const char*           engineName;
    LwU32                 engineBase;

    LwBool                bSympathSet;
    LwBool                bSympathAutoDerived;
    LwBool                bSymLoaded;
    char                  symPath[256];
    PFLCN_SYM             symTable;
    FLCN_FUNC           **ppFlcnFuncTable;
    LwU32                 numFlcnFuncs;
    BOOL                  bFlcnFuncLoaded;

    LwU8                 *pObjdumpBuffer;
    LwU32                 objdumpFileSize;
    LwBool                bObjdumpFileLoaded;
    
    char                **ppObjdumpFileFunc;
    LwU32                 objdumpFileFuncN;

    char                **pExtTracepcBuffer;
    LwU32                 extTracepcNum;
};

/*
 *  The Falcon Symbol Table
 */
struct _flcnSymbol
{
    LwU32      addr;
    LwU32      size;
    char       section;
    char       name[41];
    char       nameUpper[41];
    LwBool     bData;
    LwBool     bSizeEstimated;
    PFLCN_SYM  pNext;
    PFLCN_SYM  pTemp;
};

/*  
 *  The Falcon Queue Information
 *
 *  Queue tail "chases" the head. However, it is a cirlwal buffer in memory. 
 *  The data of the queue is reported in a sequential buffer. The beginning of
 *  the data buffer starts at the tail, and moves towards the head. Note this
 *  data is not always sequentially increasing in memory since the head may be
 *  in lower memory than the tail in wrap-around cases.
 *
 *  Message and Command queues are treated equally. They are the same in all
 *  aspects that are concerned here.
 *
 *  Not all Falcon engines have command queues, right now only PMU and DPU use
 *  them. 
 *
 *  This structure is used in function falconQueueDump and flcnEngQueueRead.
 */
struct _flcnQueue
{
    LwU32 data[LW_FLCN_MAX_QUEUE_SIZE];  // Byte contents between tail and head 
    LwU32 length;                        // Length of data in words 
    LwU32 head;                          // DMEM offset of head
    LwU32 tail;                          // DMEM offset of tail
    LwU32 id;                            // ID of the queue
};



/** @struct _flcnBlock
 *  
 *  Falcon IMEM block information.
 **/
struct _flcnBlock
{
    LwU32 tag;                   /**< IMEM tag of this block */
    LwU32 blockIndex;            /**< Index of block */
    LwU32 bValid   :1;           /**< Is the block valid? */
    LwU32 bPending :1;           /**< Is the block pending? */
    LwU32 bSelwre  :3;           /**< Is the block secure? */
};



/** @enum _flcnTagMapping
 *  
 *  Possible IMEM tag to block mapping states.
 **/
enum _flcnTagMapping
{
    FALCON_TAG_MAPPED       = 0,   /**< Tag is mapped to exactly 1 block */
    FALCON_TAG_UNMAPPED     = 1,   /**< Tag is not mapped to any block */
    FALCON_TAG_MULTI_MAPPED = 2    /**< Tag is mapped to multiple blocks */
};



/** @struct _flcnTag
 *  
 *  Describes the IMEM tag to block mapping for a specific tag. The mapType 
 *  specifies whether or not blockInfo is valid.
 *  FALCON_TAG_UNMAPPED implies that the blockInfo data is not
 *  set or is invalid.
 **/
struct _flcnTag
{
    FLCN_TAG_MAPPING    mapType;    /**< Type of mapping from tag to block */
    FLCN_BLOCK          blockInfo;  /**< Information of mapped block */
};


#endif // _FALCON_H_

