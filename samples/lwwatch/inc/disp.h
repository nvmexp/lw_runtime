/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// qpark@lwpu.com - May 16, 2006 
// disp.h - display routines
// 
//*****************************************************

#ifndef _DISP_H_
#define _DISP_H_

#include "hal.h"

// CAP related
#define DISP_STATE (0x0001)
#define DISP_DEBUG (0x0002)
#define DISP_SPVSR (0x0004)
#define DISP_EXCPT (0x0008)
#define DISP_CORE  (0x0010) // SPECIAL
#define DISP_DEFLT (DISP_STATE | DISP_EXCPT)

// iso ctx dma index
#define     LW_CORE_SURFACE     2
#define     LW_BASE_SURFACE     4


//Temp defines. Has to use the defns from fermi/gf11x/dev_disp.h
#define     LW_PDISP_MAX_HEAD_T   4
#define     LW_PDISP_SORS_T       8
#define     LW_PDISP_DACS_T       4
#define     LW_PDISP_PIORS_T      4

// Defines for DSLI
#define OR_OWNER_UNDEFINED      0xDEADBABE
#define LW_PDISP_MAX_PINS_T     0xF

// Defines for dsorpadlinkconn
#define     LW_MAX_SUBLINK        2

typedef enum
{
    CHNTYPE_CORE,
    CHNTYPE_BASE,
    CHNTYPE_OVLY,
    CHNTYPE_OVIM,
    CHNTYPE_LWRS
} ChnType;

typedef enum
{
    EXT_REFCLK_SRCA,
    EXT_REFCLK_SRCB
}ExtRefClkSrc;

typedef struct
{
    char *name;
    LwU8  headNum;
    LwU8  numHeads;
    LwU32 highbit;  // bits to read
    LwU32 lowbit;     
    LwU8  numstate;     
    LwU32 base;
    LwU32 cap; 
    ChnType id;
} ChanDesc_t ;

typedef enum
{
    LWDISPLAY_CHNTYPE_CORE,
    LWDISPLAY_CHNTYPE_WIN,
    LWDISPLAY_CHNTYPE_WINIM,
    LWDISPLAY_CHNTYPE_LWRS
} ChnType_Lwdisplay;

typedef struct
{
    char        *name;
    LwU8        headNum;
    LwU8        numHeads;
    LwU32       shift;  // bits to read
    LwU32       mask;
    LwU8        numInstance;
    LwU8        numstate;     
    LwU8        scIndex;
    LwU32       base;
    LwU32       cap; 
    ChnType_Lwdisplay id;
} ChanDesc_t_Lwdisplay;

typedef enum
{
    LW_OR_SOR,
    LW_OR_PIOR,
    LW_OR_DAC
} LWOR;

typedef enum
{
    HEAD_UNSPECIFIED,
    HEAD0,
    HEAD1,
    HEAD2,
    HEAD3
} HEAD;

typedef enum
{
    sorProtocol_LvdsLwstom,
    sorProtocol_SingleTmdsA,
    sorProtocol_SingleTmdsB,
    sorProtocol_SingleTmdsAB,
    sorProtocol_DualSingleTmds,
    sorProtocol_DualTmds,
    sorProtocol_SdiOut,
    sorProtocol_DdiOut,
    sorProtocol_DpA,
    sorProtocol_DpB,
    sorProtocol_HdmiFrl = 12,
    sorProtocol_Lwstom = 15,

    piorProtocol_ExtTmdsEnc = 0,
    piorProtocol_ExtTvEnc,
    piorProtocol_ExtSdiSdEnc,
    piorProtocol_ExtSdiHdEnc,
    piorProtocol_DistRenderOut,
    piorProtocol_DistRenderIn,
    piorProtocol_DistRenderInout,

    dacProtocol_RgbCrt = 0,
    dacProtocol_CpstNtscM,
    dacProtocol_CpstNtscJ,
    dacProtocol_CpstPalBdghi,
    dacProtocol_CpstPalM,
    dacProtocol_CpstPalN,
    dacProtocol_CpstPalCn,
    dacProtocol_CompNtscM,
    dacProtocol_CompNtscJ,
    dacProtocol_CompPalBdghi,
    dacProtocol_CompPalM,
    dacProtocol_CompPalN,
    dacProtocol_CompPalCn,
    dacProtocol_Comp480p60,
    dacProtocol_Comp576p50,
    dacProtocol_Comp720p50,
    dacProtocol_Comp720p60,
    dacProtocol_Comp1080i50,
    dacProtocol_Comp1080i60,
    dacProtocol_YuvCrt,
    dacProtocol_Lwstom = 63,

    protocolError = -1
} ORPROTOCOL;

typedef enum
{
    PADLINK_A = 0,
    PADLINK_B,
    PADLINK_C,
    PADLINK_D,
    PADLINK_E,
    PADLINK_F,
    PADLINK_G,

    PADLINK_MAX,
    PADLINK_NONE = -1
} PADLINK;

typedef enum
{
    PRIMARY = 0,
    SECONDARY
}SOR_SUBLINK;

typedef struct
{
    LwU32  handle;
    LwU32  data;
} HASH_TABLE_ENTRY;

typedef struct
{
    LwU32  classNum;
    LwU32  limitLo;
    LwU32  adjust;
    LwU32  limitHi;
    LwU32  tags;
    LwU32  partStride;
} DMAOBJECT;

#define MAX_PBCTL_REGS_PER_CHANNEL      4

typedef struct
{
    LwU32 PbCtlOffset[MAX_PBCTL_REGS_PER_CHANNEL];
}PBCTLOFFSET;

char *dispGetStringForOrProtocol(LwU32 orType, ORPROTOCOL orProtocol);
const char* dispGetPadLinkString(PADLINK padLink);
void printDpAuxlog(char *buffer, LwU32 entries);

// Initialization function
void initializeDisp_v02_01(char *chipName);
void initializeDisp_v03_00(char *chipName);
char* dispGetORString(LwU32 orType);
void dispHeadORConnectionAscii(void);

#define HEAD_IDX(owner)           (owner-HEAD0)
#define HEAD(hwHead)              ((HEAD)(HEAD0 + hwHead))

//*****************************************************
//
// Display Constant Query - see Bug 517296 and 610404
//
//*****************************************************

// This macro is used once to enumerate the fields and then
// once in each query HAL function implementation.
// The fields are wrapped in macros which are defined
// differently for each of the two usages.

#define DISP_DECLARE_CONSTANTS(devClassC, devClassD, devClassE) \
    DECLARE_CONSTANT(, _PDISP_SORS, ) \
    DECLARE_CONSTANT(, _PDISP_PIORS, ) \
    DECLARE_CONSTANT(, _PDISP_DACS, ) \
    DECLARE_CONSTANT(devClassD, _SC_DAC_SET_CONTROL_ARM, _OWNER_NONE) \
    DECLARE_CONSTANT(devClassD, _SC_DAC_SET_CONTROL_ARM, _OWNER_HEAD0) \
    DECLARE_CONSTANT(devClassD, _SC_DAC_SET_CONTROL_ARM, _OWNER_HEAD1) \
    DECLARE_CONSTANT(devClassD, _SC_SOR_SET_CONTROL_ARM, _OWNER_NONE) \
    DECLARE_CONSTANT(devClassD, _SC_SOR_SET_CONTROL_ARM, _OWNER_HEAD0) \
    DECLARE_CONSTANT(devClassD, _SC_SOR_SET_CONTROL_ARM, _OWNER_HEAD1) \
    DECLARE_CONSTANT(devClassD, _SC_PIOR_SET_CONTROL_ARM, _OWNER_NONE) \
    DECLARE_CONSTANT(devClassD, _SC_PIOR_SET_CONTROL_ARM, _OWNER_HEAD0) \
    DECLARE_CONSTANT(devClassD, _SC_PIOR_SET_CONTROL_ARM, _OWNER_HEAD1) \
    DECLARE_INDEXED(devClassD, _SC_DAC_SET_CONTROL_ARM, ) \
    DECLARE_INDEXED(devClassD, _SC_SOR_SET_CONTROL_ARM, ) \
    DECLARE_INDEXED(devClassD, _SC_PIOR_SET_CONTROL_ARM, ) \
    DECLARE_FIELD(devClassD, _SC_DAC_SET_CONTROL_ARM, _OWNER) \
    DECLARE_FIELD(devClassD, _SC_DAC_SET_CONTROL_ARM, _PROTOCOL) \
    DECLARE_FIELD(devClassD, _SC_SOR_SET_CONTROL_ARM, _OWNER) \
    DECLARE_FIELD(devClassD, _SC_SOR_SET_CONTROL_ARM, _PROTOCOL) \
    DECLARE_FIELD(devClassD, _SC_PIOR_SET_CONTROL_ARM, _OWNER) \
    DECLARE_FIELD(devClassD, _SC_PIOR_SET_CONTROL_ARM, _PROTOCOL)

// These definitions are used to enumerate the fields of the global state cache
// structure. Warning! They are undefined and redefined later in this file to
// automate the state cache read functions as well.

#define DECLARE_CONSTANT(d, r, f) DISP_CONSTANT_##r##f,
#define DECLARE_INDEXED(d, r, f) DISP_INDEXED_##r##f,
#define DECLARE_FIELD(d, r, f) DISP_FIELD_##r##f,

typedef enum
{
    DISP_DECLARE_CONSTANTS(,,)
} DISP_CONSTANT;

// Now the declaration macros can be redefined for read usage.

#undef DECLARE_CONSTANT
#undef DECLARE_INDEXED
#undef DECLARE_FIELD

#define DECLARE_CONSTANT(d, r, f) \
    case DISP_CONSTANT_##r##f: \
        return LW##d##r##f;

#define DECLARE_INDEXED(d, r, f) \
    case DISP_INDEXED_##r##f: \
        return LW##d##r##f(param);

#define DECLARE_FIELD(d, r, f) \
    case DISP_FIELD_##r##f: \
        return DRF_VAL(d, r, f, param);

// These macros help read values from the display state cache.

#define DISP_CONST(name) \
    pDisp[indexGpu].dispGetConstant(DISP_CONSTANT_##name, 0)
#define DISP_INDEX(name, index) \
    pDisp[indexGpu].dispGetConstant(DISP_INDEXED_##name, index)
#define DISP_FIELD(name, value) \
    pDisp[indexGpu].dispGetConstant(DISP_FIELD_##name, value)



//Struct to gather data to support dhdorconn -ascii option
typedef struct
{
    LwU32 headDisplayIds[LW_PDISP_MAX_HEAD_T];
    LwU32 sorOwnerMasks[LW_PDISP_SORS_T];
    LwU32 dacOwnerMasks[LW_PDISP_DACS_T];
    LwU32 piorOwnerMasks[LW_PDISP_PIORS_T];
}asciiOrConnData;


#define PRINTCONFIGHEAD(regName, dataStruct, val) \
{ \
                                dprintf("%-40s |", regName); \
                                for (head=0; head < numHead; ++head) \
                                { \
                                    dprintf("0x%-10x |", (dataStruct+head)->val); \
                                } \
                                dprintf("\n"); \
}

#define PRINTCONFIGPIOR(regName) \
{ \
                                dprintf("%-40s |", regName); \
                                for (pior=0; pior < numPior; ++pior) \
                                { \
                                    dprintf("0x%-10x |", pDsliPiorData->DsliPiorDro[pior]); \
                                } \
                                dprintf("\n"); \
}

#define PRINTCONFIG(regName, val) \
{ \
                                dprintf("%-40s |", regName); \
                                dprintf("0x%-10x |", val); \
                                dprintf("\n"); \
}

#define PRINTPINTABLE(lock, value) dprintf("%16s%11s%x\n", lock, "0x", value);

#define PRINTHEADTABLE(parameter, dataStruct1, dataStruct2, val, verbose, verboseVal) \
{ \
                                dprintf("%-40s", parameter); \
                                for (head = 0; head < numHead; ++head) \
                                { \
                                    if (verbose) \
                                    { \
                                        dprintf("0x%x%s", (dataStruct1+head)->verboseVal, "<===>"); \
                                    } \
                                    dprintf("%-21s", (dataStruct2+head)->val); \
                                } \
                                dprintf("\n"); \
}

#define PRINTLOCKTABLE(parameter, dataStruct1, dataStruct2, val, val1, verbose, verboseVal) \
{ \
                                dprintf("%-40s", parameter); \
                                for (head = 0; head < numHead; ++head) \
                                { \
                                    char tmp[21]; \
                                    if (verbose) \
                                    { \
                                        dprintf("0x%x%s", (dataStruct1+head)->verboseVal, "<===>"); \
                                    } \
                                    snprintf(tmp, 21, "%s%s", (dataStruct2+head)->val, (dataStruct2+head)->val1); \
                                    dprintf("%-21s", tmp); \
                                } \
                                dprintf("\n"); \
}

#define PRINTSYNCVAL(parameter, dataStruct1, dataStruct2, val, verbose, verboseVal) \
{ \
                                dprintf("%-40s", parameter); \
                                for (head = 0; head < numHead; ++head) \
                                { \
                                    if (verbose) \
                                    { \
                                        dprintf("0x%x%s", (dataStruct1+head)->verboseVal, "<===>"); \
                                    } \
                                    dprintf("0x%-19x", (dataStruct2+head)->val); \
                                } \
                                dprintf("\n"); \
}

enum intrTargets
{
    INTR_TGT_NONE = 0,
    INTR_TGT_RM,
    INTR_TGT_PMU,
    INTR_TGT_DPU
};

#define DIN_POPU(x)                   eve=GPU_REG_RD32(LW_PDISP_DSI_EVENT##x);                                        \
                                      en0=GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_EN0##x);                                  \
                                      rmk=GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_MSK##x);                                  \
                                      dmk=GPU_REG_RD32(LW_PDISP_DSI_DPU_INTR_MSK##x);                                 \
                                      pmk=GPU_REG_RD32(LW_PDISP_DSI_PMU_INTR_MSK##x);                                 \
                                      rip=GPU_REG_RD32(LW_PDISP_DSI_RM_INTR##x);                                      \
                                      pip=GPU_REG_RD32(LW_PDISP_DSI_PMU_INTR##x);                                     \
                                      dip=GPU_REG_RD32(LW_PDISP_DSI_DPU_INTR##x);

#define DIN_LOOP_START(ind,size,r )   DIN_POPU(r);ind=0;while(ind < size){                                        
#define DIN_LOOP_END(ind)             ++ind;}
#define DIN_GET_TGT_NAME(tgt)         (tgt==INTR_TGT_NONE)?"NONE":                                                \
                                      ((tgt == INTR_TGT_RM)?"RM":((tgt == INTR_TGT_PMU)?"PMU":"DPU"))             

#define DIN_PRINT_IDX(n, i, e, p, t)  dprintf("%s%-*d   %-10s   %-10s   %-10s  ",                                 \
                                         n, (int)(40-strlen(n)),i, p?"YES":"NO", e?"YES":"NO", DIN_GET_TGT_NAME(t)); 

#define DIN_PRINT(n, e, p, t)         dprintf("%-40s   %-10s   %-10s   %-10s  ",                                  \
                                             n, p?"YES":"NO", e?"YES":"NO", DIN_GET_TGT_NAME(t));                 

#define DIN_GET_TGT(a,rm,dp,pm)       if(DRF_VAL(_PDISP, _DSI_RM_INTR_MSK, a, rm))                                \
                                          tgt = INTR_TGT_RM;                                                      \
                                      else if(DRF_VAL(_PDISP, _DSI_PMU_INTR_MSK, a, pm))                          \
                                          tgt = INTR_TGT_PMU;                                                     \
                                      else if(DRF_VAL(_PDISP, _DSI_DPU_INTR_MSK, a, dp))                          \
                                          tgt = INTR_TGT_DPU;                                                     \
                                      else                                                                        \
                                          tgt = INTR_TGT_NONE;

#define DIN_SANITY(pe,tg,rpe,ppe,dpe) pe?((tg==INTR_TGT_NONE)?"FAIL: Target None when intr pending":              \
                                      ((!(rpe||ppe||dpe))?"FAIL: No intr in RM|PMU|DPU":                          \
                                      (((rpe|(ppe<<1)|(dpe<<2))!=tg)?"FAIL: Target not matching intr":"PASS"))):  \
                                      ((rpe||ppe||dpe)?"FAIL: Intr found when event is none":"PASS")
                                         

#define DIN_ANL_IDX(r,f,p,s)          DIN_LOOP_START(ind, LW_PDISP_DSI_EVENT##f##__SIZE_1, r)                     \
                                      pen=(DRF_VAL(_PDISP, _DSI_EVENT, f(ind), eve) ==                            \
                                                                                 LW_PDISP_DSI_EVENT##f##_PENDING);\
                                      rpe=(DRF_VAL(_PDISP,_DSI_RM_INTR,f(ind),rip)==                              \
                                                                               LW_PDISP_DSI_RM_INTR##f##_PENDING);\
                                      ppe=(DRF_VAL(_PDISP,_DSI_PMU_INTR,f(ind),pip)==                             \
                                                                             LW_PDISP_DSI_PMU_INTR##f##_PENDING); \
                                      dpe=(DRF_VAL(_PDISP,_DSI_DPU_INTR,f(ind),dip)==                             \
                                                                             LW_PDISP_DSI_DPU_INTR##f##_PENDING); \
                                      ebe=(DRF_VAL(_PDISP,_DSI_RM_INTR_EN0,f(ind),en0)==                          \
                                                                          LW_PDISP_DSI_RM_INTR_EN0##f##_ENABLE);  \
                                      DIN_GET_TGT(f(ind),rmk,dmk,pmk);                                            \
                                      DIN_PRINT_IDX(p,ind+s,ebe,pen,tgt);                                         \
                                      dprintf("%s\n",DIN_SANITY(pen, tgt, rpe, ppe, dpe));                        \
                                      DIN_LOOP_END(ind)

#define DIN_ANL(r,f,p)                DIN_POPU(r);                                                                \
                                      pen=(DRF_VAL(_PDISP,_DSI_EVENT,f,eve)==LW_PDISP_DSI_EVENT##f##_PENDING);    \
                                      rpe=(DRF_VAL(_PDISP,_DSI_RM_INTR,f,rip)==                                   \
                                                                          LW_PDISP_DSI_RM_INTR##f##_PENDING);     \
                                      ppe=(DRF_VAL(_PDISP,_DSI_PMU_INTR,f,pip)==                                  \
                                                                          LW_PDISP_DSI_PMU_INTR##f##_PENDING);    \
                                      dpe=(DRF_VAL(_PDISP,_DSI_DPU_INTR,f,dip)==                                  \
                                                                      LW_PDISP_DSI_DPU_INTR##f##_PENDING);        \
                                      ebe=(DRF_VAL(_PDISP,_DSI_RM_INTR_EN0,f,en0)==\
                                                                          LW_PDISP_DSI_RM_INTR_EN0##f##_ENABLE);  \
                                      DIN_GET_TGT(f,rmk,dmk,pmk);                                                 \
                                      DIN_PRINT(p,ebe,pen,tgt);                                                   \
                                      dprintf("%s\n",DIN_SANITY(pen, tgt, rpe, ppe, dpe));

// Structure to get data from SLI - SLIConfig
typedef struct 
{
    LwU32 DsliRgDistRndr;            //LW_PDISP_RG_DIST_RNDR  (unused on >=v03_00)
                                     //Sets distributed rendering configuration controls for the rg.
    LwU32 DsliRgDistRndrSyncAdv;     //LW_PDISP_RG_DIST_RNDR_SYNC_ADVANCE - Must be non-zero for DistRndr 

    // DsliRgFliplock varies based on architecture:
    // - disp0201:            LW_PDISP_RG_FLIPLOCK
    // - disp0300:            LW_PDISP_RG_SWAP_LOCKOUT
    // It is used only if hardware fliplock is in effect.
    LwU32 DsliRgFlipLock;

    LwU32 DsliRgStatus;              //LW_PDISP_RG_STATUS
    LwU32 DsliRgStatusLocked;        //LW_PDISP_RG_STATUS_LOCKED

    // DsliRgStatusFlipLocked varies based on architecture:
    // - disp0201:            LW_PDISP_RG_STATUS_FLIPLOCKED
    // - disp0300:            LWC37D_SET_CONTROL_FLIP_LOCK_PIN0
    LwU32 DsliRgStatusFlipLocked;

    // DsliClkRemVpllExtRef varies based on architecture:
    // - disp0201:            LW_PDISP_CLK_REM_VPLL_EXT_REF_CONFIG
    // - disp0300:            LW_PVTRIM_SYS_VPLL_MISC
    LwU32 DsliClkRemVpllExtRef;

    BOOL DsliHeadActive;             //Keep track of whether head is using PIOR SLI or not

    // DsliHeadSetCntrl varies based on architecture:
    // - disp0201:  LW907D_HEAD_SET_CONTROL
    // - disp0300:  LWC37D_HEAD_SET_CONTROL
    LwU32 DsliHeadSetCntrl;

    LwU32 DsliHeadSetSlaveLockMode;  //HEAD_SET_CONTROL_SLAVE_LOCK_MODE
    LwU32 DsliHeadSetMasterLockMode; //HEAD_SET_CONTROL_MASTER_LOCK_MODE
    LwU32 DsliHeadSetSlaveLockPin;   //HEAD_SET_CONTROL_SLAVE_LOCK_PIN
    LwU32 DsliHeadSetMasterLockPin;  //HEAD_SET_CONTROL_MASTER_LOCK_PIN

    LwU32 DsliPvTrimSysVClkRefSwitch;//LW_PVTRIM_SYS_VCLK_REF_SWITCH
    LwU32 DsliVclkRefSwitchFinalSel; //LW_PVTRIM_SYS_VCLK0_REF_SWITCH_FINALSEL
    LwU32 DsliClkDriver;             //EXT_REF or XTAL_IN or PCIREF
    LwU32 DsliSlowClk;               //LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK
    LwU32 DsliMisCclk;               //LW_PVTRIM_SYS_VCLK_REF_SWITCH_MISCLK

    LwU32 DsliClkDriverSrc;          //If External then SRCA, SRCB, or (>=v04_00) FL_REFCLK_IN
    LwU32 DsliQualStatus;            //LW_PDISP_CLK_REM_VPLL_EXT_REF_CONFIG_QUAL_STATUS  
}DSLI_DATA;

typedef struct
{
    LwU32 DsliPiorDro[LW_PDISP_PIORS_T];                 //LW_PDISP_PIOR_DRO

    LwU32 DsliVgaPiorCntrlOwner[LW_PDISP_PIORS_T];       //LW_PDISP_VGA_PIOR_SET_CONTROL_OWNER
    char *DsliVgaPiorCntrlProtocol[LW_PDISP_PIORS_T];   //LW_PDISP_VGA_PIOR_SET_CONTROL_PROTOCOL

    // DsliCap varies based on architecture:
    // - disp0201:  LW_PDISP_DSI_CAPA
    // - disp0300:  LW_PDISP_FE_SW_LOCK_PIN_CAP
    LwU32 DsliCap;
    // DsliCap varies based on architecture:
    // - disp0201:  LW_PDISP_DSI_CAPA_LOCK_PIN[0/1/2/3]_USAGE_[FLIP_LOCK/SCAN_LOCK/STEREO]
    // - disp0300:  (unused)
    LwU32 DsliCapLockPinUsage[LW_PDISP_MAX_PINS_T];
}DSLI_PIOR_DATA;

typedef struct
{
    char    *headStatus;
    char    *slaveLock;
    char    *slaveLockPin;
    char    *masterLock;
    char    *masterLockPin;
    char    *scanLockStatus;
    char    *flipLock;
    char    *flipLockStatus;
    LwU32   syncAdvance;
    char    *refClkForVpll;
}DSLI_PRINT_PARAM;

// Structure for dpinfo
typedef struct
{
    BOOL attached;
    BOOL bDpEnabled;
    BOOL bFlushEnabled;
    BOOL bMstEnabled;
    BOOL bSingleHeadMst;
    LwU32 headMask;
    LwU32 timeSlotStart;
    LwU32 timeSlotLength;
    LwU32 pbn;
    LwU32 timeSlotStart2nd;
    LwU32 timeSlotLength2nd;
    LwU32 pbn2nd;
    LwU32 displayId;
}DPINFO_SF;

typedef struct
{
    BOOL bExist;
    BOOL bDpActive[LW_MAX_SUBLINK];
    LwU32 link[LW_MAX_SUBLINK];
    LwU32 auxPort[LW_MAX_SUBLINK];
    LwU8 headMask;
    LwU8 protocol;
}DPINFO_SOR;

// Function to Initialize SLI Print structure
void dispInitializeSliData(DSLI_PRINT_PARAM *pDsliPrintData);

// Function to print SLI config data, used by DSLI
void dispPrintSliData(LwU32 numHead, LwU32 numPior, LwU32 numPin, DSLI_DATA *pDsliData, DSLI_PIOR_DATA *pDsliPiorData,
                      DSLI_PRINT_PARAM *pDsliPrintData, LwU32 verbose); 
 
// Function to print Data on screen
void dispPrintSliStatus (LwU32 numHead, DSLI_DATA *pDsliData, DSLI_PRINT_PARAM *pDsliPrintData, LwU32 verbose);

// To find the current display class
LwU32 findDisplayClass(char *classNames, char separator, char *chan_template, int n, char *lwr_class);

// To populate the classHeaderNum array..
LwU32 dispFindHeaderNum(LwU32 *header, char *classNames, char separator, char *chan_template, int n);

// To initialize the classHeaderNum array from which the appropriate class header is found (used for parsing display push buffer)
void initializeClassHeaderNum(char *classNames, LwU32 classHeaderNum[]);

// Function to print DP Rx's information
void dispPrintDpRxInfo(LwU32 port);

// Function to print DP Rx's enumeration.
void dispPrintDpRxEnum(void);

//Declaration of helper function for dispAnalyzeInterrupts
char *ynfunc(LwU32 val);
const char *santest(LwU32 evtPen, LwU32 rm_intrPen, LwU32 pmu_intrPen, LwU32 gsp_intrPen,
               LwU32 ie, LwU32 rm_im, LwU32 pmu_im, LwU32 gsp_im); //Sanity Test

// Display the HDCP releated help info
void hdcpDisplayHelp(void);

#include "g_disp_hal.h"    // (rmconfig) public interface


#endif

