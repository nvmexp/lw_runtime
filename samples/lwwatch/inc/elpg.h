/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************************************
// elpg.h
//*****************************************************************************

#ifndef __ELPG_H__
#define __ELPG_H__

#include "os.h"
#include "hal.h"
#include "chip.h"

//*****************************************************************************
// Macros
//*****************************************************************************


// ELPG registers RD/WR in GPU space
#define ELPG_REG_RD32(r)     GPU_REG_RD32(r)
#define ELPG_REG_WR32(r, v)  GPU_REG_WR32((r), (v))

//
// Prints a message conditionally based on whether the given DRF (d,r,f) is
// non-zero or zero.
//
#define PRINT_DRF_CONDITIONALLY(d, r, f, v, trueOutput, falseOutput) do {   \
    if (DRF_VAL(d, r, f, v))                                                \
    {                                                                       \
        dprintf("lw:  +\t%-30s\t: %s\n", #f, trueOutput);                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        dprintf("lw:  +\t%-30s\t: %s\n", #f, falseOutput);                  \
    }                                                                       \
} while(0)

//
// Prints a message conditionally based on whether the given DRF index
// (d,r,f,i) is non-zero or zero.
//
#define PRINT_DRF_IDX_CONDITIONALLY(d, r, f, i, v, trueOutput, falseOutput) do { \
    if (DRF_IDX_VAL(d, r, f, i, v))                                         \
    {                                                                       \
        dprintf("lw:  +\t%-30s\t: %s\n", #f, trueOutput);                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        dprintf("lw:  +\t%-30s\t: %s\n", #f, falseOutput);                  \
    }                                                                       \
} while(0)

// Prints a message if the given DRF (d,r,f) is set to the given value (c).
#define PRINT_DRF_IF_SET_TO(d, r, f, c, v, output) do {                     \
    if (FLD_TEST_DRF(d, r, f, c, v))                                        \
    {                                                                       \
        dprintf("lw:  +\t%-30s\t: %s\n", #f, output);                       \
    }                                                                       \
} while(0)

//
// Prints a message if the given DRF index (d,r,f, i) is set to the given
// value (c).
//
#define PRINT_DRF_IDX_IF_SET_TO(d, r, f, i, c, v, output) do {              \
    if (FLD_IDX_TEST_DRF(d, r, f, i, c, v))                                 \
    {                                                                       \
        dprintf("lw:  +\t%-30s\t: %s\n", #f, output);                       \
    }                                                                       \
} while(0)

//
// Prints register and value.
//
#define PRINT_REG_AND_VALUE(reg, v) do {                                    \
    dprintf("lw:  %-40s\t: 0x%08x\n", reg, v);                              \
} while(0)

//
// Prints register, engine id and value for index based register.
//
#define PRINT_REG_IDX_AND_VALUE(reg,engid, v) do {                          \
    dprintf("lw:  "#reg"\t: 0x%08x\n", engid, v);                     \
} while(0)

//
// Prints register, engine name and value for index based register.
//
#define PRINT_REG_ENG_AND_VALUE   PRINT_REG_IDX_AND_VALUE

#define PRINT_REG_AND_STRING(reg, s) do {                                   \
    dprintf("lw:  %-40s\t: 0x%08x\n", reg, v);                              \
} while(0)

#define PRINT_FIELD_AND_VALUE(field, v) do {                                \
    dprintf("lw:  +\t%-30s\t: 0x%08x\n", field, v);                         \
} while(0)

#define PRINT_FIELD_AND_STRING(field, s) do {                               \
    dprintf("lw:  +\t%-30s\t: %s\n", field, s);                             \
} while(0)

#define PRINT_STRING(s) do {                                                \
    dprintf("lw:  %s\n", s);                                                \
} while(0)

//
// Prints string and message values.
//
#define PRINT_STRING_AND_FIELD_VALUE(field,...) do {                        \
    dprintf("lw:  "#field" \n", ##__VA_ARGS__);                             \
} while(0)

#define PRINT_INDENTED_STRING(s) do {                                       \
    dprintf("lw:  +\t%s\n", s);                                             \
} while(0)

#define PRINT_INDENTED_ERROR(s) do {                                        \
    dprintf("lw:  !\tERROR: %s\n", s);                                      \
} while(0)

#define PRINT_NEWLINE do {                                                  \
    dprintf("lw:  \n");                                                     \
} while(0)

#define ELPG_ENGINE_ID_GRAPHICS       0
#define ELPG_ENGINE_ID_VIDEO          1
#define ELPG_ENGINE_ID_GR_PASSIVE     ELPG_ENGINE_ID_VIDEO
#define ELPG_ENGINE_ID_VIC            2
#define ELPG_ENGINE_ID_GR_RG          ELPG_ENGINE_ID_VIC
#define ELPG_ENGINE_ID_DI             3
#define ELPG_ENGINE_ID_EI             ELPG_ENGINE_ID_DI
#define ELPG_ENGINE_ID_MS             4
#define ELPG_ENGINE_ID_MS_LTC         4
#define ELPG_ENGINE_ID_MS_PASSIVE     4
#define ELPG_ENGINE_ID_EI_PASSIVE     3
#define ELPG_ENGINE_ID_DIFR_PREFETCH  5
#define ELPG_ENGINE_ID_DIFR_SW_ASR    6
#define ELPG_ENGINE_ID_DIFR_CG        7
#define ELPG_ENGINE_ID_ILWALID        ELPG_ENGINE_ID_DIFR_CG + 1
// Maximum supported ELPG engines.
#define ELPG_MAX_SUPPORTED_ENGINES   ELPG_ENGINE_ID_ILWALID

//*****************************************************************************
// External defines
//*****************************************************************************
extern char *LpwrEng[ELPG_MAX_SUPPORTED_ENGINES];

//*****************************************************************************
// Prototypes
//*****************************************************************************

void  elpgDisplayHelp(void);
void  elpgDumpPgLog(void);
LW_STATUS lpwrGetFsmState(void);
LW_STATUS elpgGetStatus(void);
LW_STATUS lpwrGetStatus(void);

#include "g_elpg_hal.h"    // (rmconfig) public interface


#endif /* __ELPG_H__ */
