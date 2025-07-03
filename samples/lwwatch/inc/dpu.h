/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// Non-Hal public functions for DPU
// dpu.h
//
//*****************************************************

#ifndef _DPU_H_
#define _DPU_H_

/* ------------------------ Includes --------------------------------------- */
#include "os.h"
#include "hal.h"
#include "falcon.h"

/* ------------------------ Function Prototypes ---------------------------- */
const char*             dpuGetSymFilePath(void);
const char*             dpuGetEngineName(void);
LwU32                   dpuGetDmemAccessPort(void);
POBJFLCN                dpuGetFalconObject(void);

/* ------------------------ Common Defines --------------------------------- */
// sanity test macros
#ifdef CLIENT_SIDE_RESMAN
    void DPU_LOG(int lvl, const char *fmt, ...);
#else
    #define DPU_LOG(l,f,...) DPU_LOGGING(verbose, l, f, ##__VA_ARGS__)
#endif

#define VB0  0
#define VB1  1
#define VB2  2

/*!
 *  DPU sanity test logging wrapper.
 */
#ifndef CLIENT_SIDE_RESMAN
#define DPU_LOGGING(v,l,f,...) \
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
 * DPU sanity test function flags
 *
 * AUTO          - include in the autorun mode
 * DESTRUCTIVE   - changes the current state of DPU
 * REQUIRE_ARGS  - requires an extra arugument to be passed.
 * OPTIONAL_ARGS - takes an argument, but not required.
 * PROD_UCODE    - requires a production ucode.
 * VERIF_UCOD    - requires a verfication ucode.
 */
#define DPU_TEST_FLAGS_CODE    "ADXOPV"

#define DPU_TEST_AUTO          BIT(0)
#define DPU_TEST_DESTRUCTIVE   BIT(1)
#define DPU_TEST_REQUIRE_ARGS  BIT(2)
#define DPU_TEST_OPTIONAL_ARGS BIT(3)
#define DPU_TEST_PROD_UCODE    BIT(4)
#define DPU_TEST_VERIF_UCODE   BIT(5)

/* ------------------------ Types definitions ------------------------------ */
/** @typedef struct DpuSanityTestEntry DpuSanityTestEntry
 *  @see struct DpuSanityTestEntry
 **/
typedef struct DpuSanityTestEntry DpuSanityTestEntry;

/** @typedef LwU32 (*DpuTestFunc) (LwU32, char *)
 *  function pointer to each test case
 *  1st arg - verbose
 *  2nd arg - extra argument
 **/
typedef LwU32 (*DpuTestFunc) (LwU32, char *);

/** @struct DpuSanityTestEntry
 *
 *  Declares each DPU sanity test case. Consists of a function pointer and
 *  description.
 **/
struct DpuSanityTestEntry
{
    DpuTestFunc       fnPtr;
    LwU32             flags;
    const char* const fnInfo;
};

/* ------------------------ Static variables ------------------------------- */

//
// DPU SANITY TEST HEADERS
//

// v02_01
LW_STATUS dpuSanityTest_Reset_v02_01        (LwU32, char *);
LW_STATUS dpuSanityTest_Latency_v02_01      (LwU32, char *);
LW_STATUS dpuSanityTest_GPTMR_v02_01        (LwU32, char *);

#include "g_dpu_hal.h"     // (rmconfig)  public interfaces

#endif // _DPU_H_
