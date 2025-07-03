/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FALCON_H_
#define _FALCON_H_

/* ------------------------ Types definitions ------------------------------ */
typedef struct _flcnEngineIFaces    FLCN_ENGINE_IFACES;



/* ------------------------ Includes --------------------------------------- */

#include "flcngdb.h"
//#include "g_falcon_hal.h"     // (rmconfig) public interface


/* ------------------------ Common Defines --------------------------------- */
#define FLCN_REG_WR32(addr, value)          GPU_REG_WR32((addr)+engineBase, (value))
#define FLCN_REG_RD32(addr)                 GPU_REG_RD32((addr)+engineBase)

// macro to get appropriate IBRKPT register, i = 0 is for BRKPT1 and so on till IBRKPT5
#define IBRKPT_REG_GET(i)  ((i == 0 || i == 1)?(LW_PFALCON_FALCON_IBRKPT1 + 0x00000004 * (i)) : (LW_PFALCON_FALCON_IBRKPT1 + 0x00000010 + 0x00000004 * (i)))

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

/* ---------------- Prototypes - Falcon Core HAL Interface ------------------ */
/*
 *  Prototypes in "g_falcon_hal.h"
 */

/* ---------------- Prototypes - Falcon Engine HAL Interface ---------------- */
typedef LwU32                     FlcnEngGetFalconBase(void);


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

/* 
 *  FALCON Engine Interfaces
 */
struct _flcnEngineIFaces {
    FlcnEngGetFalconBase        *flcnEngGetFalconBase;
};

/* ------------------------ Static variables ------------------------------- */

#endif // _FALCON_H_

