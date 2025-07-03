/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
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
// clk.h
//
//*****************************************************

#ifndef _CLK_H_
#define _CLK_H_

#include "clk/fs/clkfreqsrc.h"
#include "os.h"
#include "hal.h"

//
// Clock sources can be PLLs or other clocks.
//
typedef enum
{
   clkSrcWhich_Ilwalid           = -1,
   clkSrcWhich_Default           = 0,
   clkSrcWhich_XTAL              = 1,
   clkSrcWhich_XTALS             = 2,
   clkSrcWhich_XTAL4X            = 3,
   clkSrcWhich_EXTREF            = 4,
   clkSrcWhich_QUALEXTREF        = 5,
   clkSrcWhich_EXTSPREAD         = 6,
   // SPPLL0 and 1 enums should be conselwtive.
   clkSrcWhich_SPPLL0            = 7,
   clkSrcWhich_SPPLL1            = 8,
   clkSrcWhich_XCLK              = 9,    
   clkSrcWhich_XCLK3XDIV2        = 10,
   clkSrcWhich_MPLL              = 11,
   clkSrcWhich_HOSTCLK           = 12,
   clkSrcWhich_PEXREFCLK         = 13,
   // VPLL0-3 enums should be conselwtive.
   clkSrcWhich_VPLL0             = 14, 
   clkSrcWhich_VPLL1             = 15,
   clkSrcWhich_VPLL2             = 16,
   clkSrcWhich_VPLL3             = 17,
   clkSrcWhich_PWRCLK            = 18,
   clkSrcWhich_HDACLK            = 19,
   // Fermi specific new clock sources.
   clkSrcWhich_GPCPLL            = 20,
   clkSrcWhich_LTCPLL            = 21,
   clkSrcWhich_XBARPLL           = 22,
   clkSrcWhich_SYSPLL            = 23,
   clkSrcWhich_REFMPLL           = 24,
   clkSrcWhich_XCLK500           = 25,
   clkSrcWhich_XCLKGEN3          = 26,
   clkSrcWhich_Supported         = 27
} CLKSRCWHICH;

typedef struct
{
    LwU32       TargetFreq;  // In 10s of KHz.  This is what we are trying to hit
    LwU32       ActualFreq;  // In 10s of KHz.  This is what we are trying to hit

} CLKSETUPPARAMS;

typedef struct
{
    LwU32       TargetFreq; // In 10s of KHz.  This is what we are trying to hit
    LwU32       ActualFreq; // In 10s of KHz.  Actual freq after the pulse eater
    LwU32       PDivSlow;   // Lwrrently implemented only for MCLK
    LwU32       F;          // FDIV. Exists only for SPPLLs. Think about privatizing.
    union
    {
        struct
        {
            LwU32 M, Mb, N, Nb, P, PL;
            BOOL FD;
        } OneStage;

        struct {
            LwU32 Ma, Mb, Na, Nb, P, PL;
            BOOL FDa, FDb;
        } TwoStage;

        struct {
            LwU32 Ma, Mb, Na, Nb, P, PVPLL2;
        } SysPll;

        struct {
            BOOL    bSDM, bAlt;
            LwU32   M, N, P, PL, SDM, FP;
        } OneSrc;


    } Arch;

} PLLSETUPPARAMS;

// used by clock counter code
typedef enum 
{
    clkWhich_DISP = 0,
    clkWhich_NCDISP,
    clkWhich_NCSCC,
    clkWhich_LWT,
    clkWhich_ONESRC_ALT,
    clkWhich_ONESRC_REF,
    clkWhich_ONESRC_ROOT,
    clkWhich_PEX
} CLKWHICH;

typedef struct 
{
    CLKWHICH clk;
    LwU32 regCfgAddr;
    LwU32 regCntAddr;
} CLK_CNTR;

#define MAX_CLK_CNTR     13 // Max number of clk counters available in HW
#define MAX_CLK_CNTR_SRC 15 // Max num of clk sources in each HW clk counter
typedef struct
{
    LwU32 clkCntrCfgReg;
    LwU32 clkCntrCntReg;
    LwU32 clkCntrSrcNum;
    struct {
        LwU32 srcIdx;
        char* srcName;
    } srcInfo[MAX_CLK_CNTR_SRC];
} CLK_COUNTER_INFO;

typedef struct
{
    LwU32 cfgReg;
    LwU32 srcReg;       // unused from TU102 onwards
    LwU32 cntRegLsb;
    LwU32 cntRegMsb;
} CLK_FR_COUNTER_REG_INFO;

#define CLK_DOMAIN_NAME_STR_LEN 6

typedef struct
{
    char  clkDomainName[CLK_DOMAIN_NAME_STR_LEN];
    LwU32 clkDomain;                                // LW2080_CTRL_CLK_DOMAIN_ index
    LwU32 srcNum;
    struct {
        LwU32 srcIdx;
        char* srcName;
    } srcInfo[MAX_CLK_CNTR_SRC];
} CLK_FR_COUNTER_SRC_INFO;

#define MAX_TIMEOUT_US 100
#define MAX_STABLE_COUNT 5
#define CLK_IP_XTAL_CYCLES 1080
#define XTAL_CLK_PERIOD_NS 37


typedef union
{
    LwU64 val;
    struct
    {
        LwU32 lo;
        LwU32 hi;
    }parts;
} LwU64_PARTS;


#define LwU64_PARTS_SUB(result, x, y)                                          \
do {                                                                           \
    (result).parts.lo = (x).parts.lo - (y).parts.lo;                           \
    (result).parts.hi = (x).parts.hi - (y).parts.hi -                          \
        (((x).parts.lo < (y).parts.lo) ? 1 : 0);                               \
} while (0)

#define LwU64_PARTS_ADD(result, x, y)                                          \
do {                                                                           \
    (result).parts.lo = (x).parts.lo + (y).parts.lo;                           \
    (result).parts.hi = (x).parts.hi + (y).parts.hi +                          \
        (((result).parts.lo < (x).parts.lo) ? 1 : 0);                          \
} while (0)

#define LwU64_PARTS_COMPARE(result, x, y)                                     \
do {                                                                          \
    if ((x).parts.hi > (y).parts.hi)                                          \
        result = 1;                                                           \
    else if (((x).parts.hi == (y).parts.hi) && ((x).parts.lo > (y).parts.lo)) \
        result = 1;                                                           \
    else if (((x).parts.hi == (y).parts.hi) && ((x).parts.lo == (y).parts.lo))\
        result = 0;                                                           \
    else                                                                      \
        result = -1;                                                          \
} while (0)


// AVFS defines

/*!
 * Valid global NAFLL ID values
 */
#define CLK_NAFLL_ID_SYS        (0x00000000)
#define CLK_NAFLL_ID_LTC        (0x00000001)  // GP100 only
#define CLK_NAFLL_ID_XBAR       (0x00000002)
#define CLK_NAFLL_ID_GPC0       (0x00000003)
#define CLK_NAFLL_ID_GPC1       (0x00000004)
#define CLK_NAFLL_ID_GPC2       (0x00000005)
#define CLK_NAFLL_ID_GPC3       (0x00000006)
#define CLK_NAFLL_ID_GPC4       (0x00000007)
#define CLK_NAFLL_ID_GPC5       (0x00000008)
#define CLK_NAFLL_ID_GPC6       (0x00000009)  // Ampere and later
#define CLK_NAFLL_ID_GPC7       (0x0000000A)  // Ampere and later
#define CLK_NAFLL_ID_GPCS       (0x0000000B)
#define CLK_NAFLL_ID_LWD        (0x0000000C)  // Volta and later
#define CLK_NAFLL_ID_HOST       (0x0000000D)  // Volta and later
#define CLK_NAFLL_ID_GPC8       (0x0000000E)  // Ada-only
#define CLK_NAFLL_ID_GPC9       (0x0000000F)  // Ada-only
#define CLK_NAFLL_ID_GPC10      (0x00000010)  // Ada-only
#define CLK_NAFLL_ID_GPC11      (0x00000011)  // Ada-only
#define CLK_NAFLL_ID_UNDEFINED  (0x000000FF)

/*!
 * Enumeration of register types used to control NAFLL-s.
 */
#define CLK_NAFLL_REG_TYPE_LUT_READ_ADDR            0
#define CLK_NAFLL_REG_TYPE_LUT_READ_DATA            1
#define CLK_NAFLL_REG_TYPE_LUT_CFG                  2
#define CLK_NAFLL_REG_TYPE_LUT_DEBUG2               3
#define CLK_NAFLL_REG_TYPE_LUT_STATUS               3  // Alias for the _DEBUG2 register
#define CLK_NAFLL_REG_TYPE_NAFLL_COEFF              4
#define CLK_NAFLL_REG_TYPE_LUT_ACK                  5
#define CLK_NAFLL_REG_TYPE_LUT_READ_OFFSET_DATA     6
#define CLK_NAFLL_REG_TYPE__COUNT                   7

/*!
 * LUT temperature index defines.
 * There is space for 5 tables in the LUT, so the max is set to 4.
 */
#define CLK_LUT_TEMP_IDX_0                                          (0x00000000)
#define CLK_LUT_TEMP_IDX_1                                          (0x00000001)
#define CLK_LUT_TEMP_IDX_2                                          (0x00000002)
#define CLK_LUT_TEMP_IDX_3                                          (0x00000003)
#define CLK_LUT_TEMP_IDX_4                                          (0x00000004)
#define CLK_LUT_TEMP_IDX_MAX                                        CLK_LUT_TEMP_IDX_4
#define CLK_LUT_TEMP_IDX_ILWALID                                    (0x000000ff)
/*!
 * The number of entries per LUT stride
 */
#define CLK_LUT_ENTRIES_PER_STRIDE          (0x00000002)

/*!
 * Structure to hold the map of NAFLL ID and its register addresses. All
 * these registers should be accessed via FECS bus to be efficient.
 */
typedef struct
{
    /*!
     * The global ID @ref CLK_NAFLL_ID_<xyz> of this NAFLL device.
     */
    LwU32     id;

    /*!
     * An array of register addresses indexed by register type enum.
     */
    LwU32   regAddr[CLK_NAFLL_REG_TYPE__COUNT];
} CLK_NAFLL_ADDRESS_MAP,
*PCLK_NAFLL_ADDRESS_MAP;

/*!
 * INVALID index into the NAFLL address map table
 */
#define CLK_NAFLL_ADDRESS_TABLE_IDX_ILWALID  0xffffffff

// AVFS defines - end

/*!
 * Undefined register address in a address map table
 */
#define CLK_REGISTER_ADDR_UNDEFINED         0xFFFFFFFF

//
// prototypes
//
const char* getClkSrcName(CLKSRCWHICH);
LwU32 configFreqCounter(LwU32 tgtClkCntCfgReg, LwU32 tgtClkSrcDef, LwU32 tgtClkCntReg, LwU32 clockInput);

#include "g_clk_hal.h"        // (rmconfig) public interface


#endif // _CLK_H_
