/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2014 by LWPU Corporation.  All rights reserved.  All
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
// gr.h
//
//*****************************************************

#ifndef _GR_H_
#define _GR_H_

#include "compat.h"
#include "utils/lw_enum.h"
#include "os.h"
#include "ioaccess/ioaccess.h"
#include "hal.h"
#include "priv.h"

extern LwU32 grEngineId;

// For pre-fermi only... For fermi and later, use LW_TPC_IN_GPC_STRIDE
#define TPC_STRIDE    (0x1000)

// For SMC
#define ILWALID_SWIZZID    0xFFFFFFFF
#define ILWALID_GR_IDX     0xFFFFFFFF
#define MAX_GR_IDX         8
#define MAX_SWIZZID        15

// bit fields should be defined in manuals
#define LW_PGRAPH_PRI_PROP_STATE0_WATCHDOG_ERROR             0:0
#define LW_PGRAPH_PRI_PROP_STATE0_MRT_PASS                   3:1
#define LW_PGRAPH_PRI_PROP_STATE0_STALL_C                    4:4
#define LW_PGRAPH_PRI_PROP_STATE0_C_STATE                    5:5
#define LW_PGRAPH_PRI_PROP_STATE0_STALL_Z                    6:6
#define LW_PGRAPH_PRI_PROP_STATE0_Z_STATE                    7:7
#define LW_PGRAPH_PRI_PROP_STATE0_MRT_STATE                  9:8
#define LW_PGRAPH_PRI_PROP_STATE0_MRT_STATE_ERROR           10:10
#define LW_PGRAPH_PRI_PROP_STATE0_MRF_OVERFLOW              11:11
#define LW_PGRAPH_PRI_PROP_STATE0_ZD_STATE                  12:12
#define LW_PGRAPH_PRI_PROP_STATE0_VARG_STATE                13:13
#define LW_PGRAPH_PRI_PROP_STATE0_VARG_VALIDS               15:14
#define LW_PGRAPH_PRI_PROP_STATE0_VARG_HAS_Z                16:16
#define LW_PGRAPH_PRI_PROP_STATE0_VARG_HAS_C                17:17
#define LW_PGRAPH_PRI_PROP_STATE0_VAG_HAS_Z                 18:18
#define LW_PGRAPH_PRI_PROP_STATE0_VAG_HAS_C                 19:19
#define LW_PGRAPH_PRI_PROP_STATE0_MA_HAS_Z                  20:20
#define LW_PGRAPH_PRI_PROP_STATE0_MA_HAS_C                  21:21
#define LW_PGRAPH_PRI_PROP_STATE0_MA_VALIDS                 24:22
#define LW_PGRAPH_PRI_PROP_STATE0_ZTLF_SYNC_STALL           25:25
#define LW_PGRAPH_PRI_PROP_STATE0_CSB_LWMCOVG_WAIT          26:26
#define LW_PGRAPH_PRI_PROP_STATE0_ZTLF_LWMCOVG_WAIT         27:27
#define LW_PGRAPH_PRI_PROP_STATE0_TPC2_EARLYZ_MODE          28:28
#define LW_PGRAPH_PRI_PROP_STATE0_TPC_EARLYZ_MODE           29:29
#define LW_PGRAPH_PRI_PROP_STATE0_TPZ_EARLYZ_MODE           30:30

#define LW_PGRAPH_PRI_PROP_STATE1_QSSM_STATE                 4:0
#define LW_PGRAPH_PRI_PROP_STATE1_ZRECSB_EARLYZ_ACTIVE       5:5
#define LW_PGRAPH_PRI_PROP_STATE1_COMPSB_EARLYZ_ACTIVE       6:6
#define LW_PGRAPH_PRI_PROP_STATE1_CDP2SMLF_2D_SHUNT          8:8
#define LW_PGRAPH_PRI_PROP_STATE1_SMLF2CDP_BLIT_CORRAL       9:9
#define LW_PGRAPH_PRI_PROP_STATE1_BLT_CORRAL_ACTIVE         10:10
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_SMST_PVLD2         12:12
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_SMST_PVLD          13:13
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_SMLD_PVLD          14:14
#define LW_PGRAPH_PRI_PROP_STATE1_CTLFSMC_PMASK_ERROR       15:15
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_EZ_PIX_PVLD        16:16
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_PIX_PVLD           17:17
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_EZ_PS_ILW_PVLD     18:18
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_ILW_PVLD           19:19
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_EZ_COH_TICK_PVLD   20:20
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_COH_TICK_PVLD      21:21
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_EZ_SB_PVLD         22:22
#define LW_PGRAPH_PRI_PROP_STATE1_S1_CSB_SB_PVLD            23:23
#define LW_PGRAPH_PRI_PROP_STATE1_S4_CDP_PD_ID              27:24
#define LW_PGRAPH_PRI_PROP_STATE1_S4_CDP_DATA_IS_VALID      28:28
#define LW_PGRAPH_PRI_PROP_STATE1_S4_CDP_BUNDLE_IS_VALID    29:29
#define LW_PGRAPH_PRI_PROP_STATE1_S0_ZSB_EARLYZ_ACTIVE      30:30
#define LW_PGRAPH_PRI_PROP_STATE1_S0_CSB_EARLYZ_ACTIVE      31:31

#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_SB_SYNCS_LTE        0:0
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_SB_SYNCS_AT_MAX     1:1
#define LW_PGRAPH_PRI_PROP_STATE2_S1_CSB_SB_SYNCS_LTE        2:2
#define LW_PGRAPH_PRI_PROP_STATE2_S1_CSB_SB_SYNCS_AT_MAX     3:3
#define LW_PGRAPH_PRI_PROP_STATE2_S1_CSB_BS_GT_ZERO          4:4
#define LW_PGRAPH_PRI_PROP_STATE2_BLTSYNC_AT_MAX             5:5
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_REP_IN_FLIGHT       8:6
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZTLF_COH_TICK_IS_VALID  9:9
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZTLF_TILE_IS_VALID     10:10
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZTLF_BUNDLE_VALID      11:11
#define LW_PGRAPH_PRI_PROP_STATE2_S0_ZSB_ZMODE_ZS_ONLY      12:12
#define LW_PGRAPH_PRI_PROP_STATE2_S0_ZSB_ZCLEAR_ACTIVE      13:13
#define LW_PGRAPH_PRI_PROP_STATE2_S0_ZSB_ZTLF_SELECT        14:14
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_SB_PVLD            15:15
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_COH_TICK_PVLD      16:16
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_ZPIX_PVLD          17:17
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_TILE_NOZ_PVLD      18:18
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_Z1_NOTILE_PVLD     19:19
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_TILE_PVLD          20:20
#define LW_PGRAPH_PRI_PROP_STATE2_S1_ZSB_Z0_TILE_PVLD       21:21
#define LW_PGRAPH_PRI_PROP_STATE2_ZSBSM_STATE               24:22
#define LW_PGRAPH_PRI_PROP_STATE2_SEMA_STATE                26:25
#define LW_PGRAPH_PRI_PROP_STATE2_CSBSM_STATE               29:27
#define LW_PGRAPH_PRI_PROP_STATE2_SMLF_EMPTY                30:30
#define LW_PGRAPH_PRI_PROP_STATE2_SMLF_FULL                 31:31

#define LW_PGRAPH_PRI_PROP_STATE3_SCTR_LO                   29:0

#define LW_PGRAPH_PRI_PROP_STATE4_SCTR_HI                    9:0
#define LW_PGRAPH_PRI_PROP_STATE4_MRT_FIRST_PASS            16:16
#define LW_PGRAPH_PRI_PROP_STATE4_MRT_ACTIVE                17:17
#define LW_PGRAPH_PRI_PROP_STATE4_ZPASS_RPLAYFF_OVFL        18:18
#define LW_PGRAPH_PRI_PROP_STATE4_ZPARTFIFO_OVFL            19:19
#define LW_PGRAPH_PRI_PROP_STATE4_TFIFO_OVFL                20:20

#define LW_PGRAPH_PRI_PROP_STATE5_PROP2CBAR_CREDIT_CNT       5:0
#define LW_PGRAPH_PRI_PROP_STATE5_PROP2ZBAR_CREDIT_CNT      12:8
#define LW_PGRAPH_PRI_PROP_STATE5_C2ZLAT_CREDIT_CNT         23:16
#define LW_PGRAPH_PRI_PROP_STATE5_ZPMTFIFO_EMPTY            31:24

/*!
 * IO Aperture types for GR.
 */
typedef enum
{
    GR_UNIT_TYPE_GPU_DEVICE = 0,
    GR_UNIT_TYPE_GR,
    GR_UNIT_TYPE_GPC,
    GR_UNIT_TYPE_EGPC,
    GR_UNIT_TYPE_ETPC,
    GR_UNIT_TYPE_TPC,
    GR_UNIT_TYPE_ROP,
    GR_UNIT_TYPE_PPC,

    GR_UNIT_TYPE_COUNT
} GR_UNIT_TYPE;

/// Value used as an index to fetch Broadcast aperture.
#define GR_BROADCAST 0xFFFFFFFF

/*!
 * GR_IO_APERTURE describes one type of aperture for zero based register addressing.
 * For example: The entire GR engine will have one GR_IO_APERTURE.
 * GPC is a unit within GR, which can have multiple sub-units each with their own apertures.
 *
 * For Example In Hopper architecture
 * The GPC apertures will be captured as part of aperture array pointed by
 * grApertures->pChildren[GR_UNIT_TYPE_GPC].
 * Similarly, ROP is another unit within GPC, which will be described by
 * gpcAperture->pChildren[GR_UNIT_TYPE_ROP] and PPC apertures can be at
 * gpcAperture->pChildren[GR_UNIT_TYPE_PPC_IN_TPC]...
 *
 * Check out the SW confluence page for "Hopper-144 PRI space restructure" for more details.
 */
typedef struct GR_IO_APERTURE
{
    ///< IO Aperture used for register access.
    IO_APERTURE aperture;

    ///< index within the parent aperture.
    LwU32 unitIndex;

    ///< flag indicating if aperture is for a broadcast unit.
    LwBool bIsBroadcast;

    struct GR_IO_APERTURE *pParent;

    ///
    /// Child units under this unit.
    /// Array of pointers, each pointer points to a block of n+1 instances of that type.
    /// The extra aperture being for the broadcast registers.
    /// eg: grApertures.pChildren[UNIT_TYPE_GPC] --> pGpcApertures[GPC_COUNT+1]
    ///
    struct GR_IO_APERTURE *pChildren[GR_UNIT_TYPE_COUNT];

    ///
    /// Stores the number of each type of units, including broadcast aperture if the unit
    /// supports broadcast
    ///
    LwU32 unitCounts[GR_UNIT_TYPE_COUNT];

    ///
    /// Index of aperture for broadcast registers within the aperture array.
    /// This also serves as the number of instances for each aperture type.
    ///
    LwU32 sharedIndices[GR_UNIT_TYPE_COUNT];
} GR_IO_APERTURE;

extern GR_IO_APERTURE grApertures[MAX_GPUS];

//-----------------------------------------------------
// Decode the 3 bit unit status (ACTIVITY) fields
//-----------------------------------------------------
typedef enum
{
    EMPTY,
    ACTIVE,
    PAUSED,
    QUIESCENT,
    UNKNOWN,
    STALLED,
    FAULTED,
    HALTED,
    GPU_UNIT_STATUS_MAXIMUM,
} GPU_UNIT_STATUS;
typedef enum
{
    PREEMPTED =4,
} GPU_UNIT_STATUS_PREEMPTED;

//-----------------------------------------------------
// GR register dumping
//-----------------------------------------------------
#define GR_REG_NAME_BUFFER_LEN 100
#define QUOTE_ME(x) #x

LW_STATUS _grGetAperture(GR_IO_APERTURE *pApertureIn, GR_IO_APERTURE **ppApertureOut, LwU32 count, ...);

/*!
 * Macro used to fetch an Aperture.
 * This colwerts type index pairs into an array and their count for getGetIoAperture.
 *
 * for eg.,
 * GR_GET_APERTURE(&pGrAperture->ioAperture, &pRopAp, GR_UNIT_TYPE_GPC, gpcCounter, GR_UNIT_TYPE_ROP, ropCounter);
 * becomes
 * grGetAperture(&pGrAperture->ioAperture, &pRopAp, {GR_UNIT_TYPE_GPC, gpcCounter, GR_UNIT_TYPE_ROP, ropCounter}, 4);
 */
#ifdef _WIN32
    // TODO: Use latest Windows compiler version to unify #define GR_GET_APERTURE to match RM implementation. Bug: 3110134
    #define GR_GET_APERTURE(pApertureIn, ppApertureOut, ...) \
        _grGetAperture((pApertureIn), (ppApertureOut), LW_NUM_ARGS(__VA_ARGS__), __VA_ARGS__)
#else
    #define GR_GET_APERTURE(pApertureIn, ppApertureOut, ...) \
        pGr[indexGpu].grGetAperture((pApertureIn), (ppApertureOut), ((LwU32[]){__VA_ARGS__}), LW_NUM_ARGS(__VA_ARGS__))
#endif

/*!
 * Form a GPC/TPC register address.  Strides defined in hwproject.h
 */
#define GPC_REG_ADDR( strReg, nGpc )                            \
    (LW_PGRAPH_PRI_GPC0_##strReg + nGpc * LW_GPC_PRI_STRIDE)
#define TPC_REG_ADDR( strReg, nGpc, nTpc )                              \
    (LW_PGRAPH_PRI_GPC0_TPC0_##strReg + nGpc * LW_GPC_PRI_STRIDE +      \
     nTpc * LW_TPC_IN_GPC_STRIDE )
#define SM_REG_ADDR( strReg, nGpc, nTpc, nSm )                          \
    (LW_PGRAPH_PRI_GPC0_TPC0_SM0_##strReg + nGpc * LW_GPC_PRI_STRIDE +  \
     nTpc * LW_TPC_IN_GPC_STRIDE + nSm * LW_SM_PRI_STRIDE )
#define PPC_REG_ADDR( strReg, nGpc, nPpc )                              \
    (LW_PGRAPH_PRI_GPC0_PPC0_##strReg + nGpc * LW_GPC_PRI_STRIDE +      \
     nPpc * LW_PPC_IN_GPC_STRIDE )
#define BE_REG_ADDR( strReg, nFbp )                                     \
    (LW_PGRAPH_PRI_BE0_##strReg + nFbp * LW_ROP_PRI_STRIDE)

/*!
 * Dumps a PGRAPH register without decoding it.
 */
#define DUMP_REG(strReg)        do{                                     \
    sprintf( buffer, "LW_PGRAPH_%s", #strReg );                     \
    DPRINTF_REG(buffer, LW_PGRAPH_##strReg, GPU_REG_RD32(LW_PGRAPH_ ## strReg) ); \
}while(0)

/*!
 * Dumps a GPC register without decoding it.
 */
#define DUMP_GPC_REG(strReg, nGpc)        do{                                   \
    sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_%s", (LwU32)nGpc, #strReg );            \
    DPRINTF_REG(buffer, GPC_REG_ADDR(strReg, nGpc),                 \
                    GPU_REG_RD32(GPC_REG_ADDR(strReg, nGpc)));              \
}while(0)

#define DUMP_GPC_REG_Z(strReg, nGpc)        do{                 \
    LwU32 val = GPU_REG_RD32(GPC_REG_ADDR(strReg, nGpc));            \
    if(val)                                                         \
    {                                                                   \
        sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_%s", (LwU32)nGpc, #strReg ); \
        DPRINTF_REG(buffer, GPC_REG_ADDR(strReg, nGpc), val);           \
    }                                                                   \
}while(0)

/*!
 * Dumps a TPC register without decoding it.
 */
#define DUMP_TPC_REG(strReg, nGpc, nTpc)         do{                            \
    sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_%s", (LwU32)nGpc, (LwU32)nTpc, #strReg); \
    DPRINTF_REG(buffer, TPC_REG_ADDR(strReg, nGpc, nTpc),           \
                    GPU_REG_RD32(TPC_REG_ADDR(strReg, nGpc, nTpc)));      \
}while(0)

#define DUMP_TPC_REG_Z(strReg, nGpc, nTpc)         do{                            \
    LwU32 val = GPU_REG_RD32(TPC_REG_ADDR(strReg, nGpc, nTpc));              \
    if(val)                                                             \
    {                                                                   \
        sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_%s", (LwU32)nGpc, (LwU32)nTpc, #strReg); \
        DPRINTF_REG(buffer, TPC_REG_ADDR(strReg, nGpc, nTpc), val);     \
    }                                                                   \
}while(0)

/*!
 * Dumps a SM register without decoding it.
 */
#define DUMP_SM_REG(strReg, nGpc, nTpc, nSm)         do{                    \
    sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_%s", (LwU32)nGpc, (LwU32)nTpc, (LwU32)nSm, #strReg); \
    DPRINTF_REG(buffer, SM_REG_ADDR(strReg, nGpc, nTpc, nSm),               \
                GPU_REG_RD32(SM_REG_ADDR(strReg, nGpc, nTpc, nSm)));        \
}while(0)

#define DUMP_SM_REG_Z(strReg, nGpc, nTpc, nSm)         do{                  \
    LwU32 val = GPU_REG_RD32(SM_REG_ADDR(strReg, nGpc, nTpc, nSm));         \
    if(val)                                                                 \
    {                                                                       \
        sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_%s", (LwU32)nGpc, (LwU32)nTpc, (LwU32)nSm, #strReg); \
        DPRINTF_REG(buffer, SM_REG_ADDR(strReg, nGpc, nTpc, nSm), val);     \
    }                                                                       \
}while(0)

/*!
 * Dumps a PPC register without decoding it.
 */
#define DUMP_PPC_REG(strReg, nGpc, nPpc)         do{                            \
    sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_PPC%d_%s", (LwU32)nGpc, (LwU32)nPpc, #strReg); \
    DPRINTF_REG(buffer, PPC_REG_ADDR(strReg, nGpc, nPpc),           \
                    GPU_REG_RD32((PPC_REG_ADDR(strReg, nGpc, nPpc))));      \
}while(0)

/*!
 * Dumps a BE (FBP) register without decoding it.
 */
#define DUMP_BE_REG(strReg, nFbp)         do{                       \
    sprintf(buffer, "LW_PGRAPH_PRI_BE%d_%s", (LwU32)nFbp, #strReg);  \
    DPRINTF_REG(buffer, BE_REG_ADDR(strReg, nFbp),                  \
                GPU_REG_RD32((BE_REG_ADDR(strReg, nFbp))));             \
}while(0)

#define DUMP_BE_REG_Z(strReg, nFbp)         do{     \
    LwU32 val = GPU_REG_RD32(BE_REG_ADDR(strReg,nFbp));  \
    if(val)                                                             \
    {                                                                   \
        sprintf(buffer, "LW_PGRAPH_PRI_BE%d_%s", (LwU32)nFbp, #strReg);  \
        DPRINTF_REG(buffer, BE_REG_ADDR(strReg, nFbp), val);            \
    }                                                                   \
}while(0)

/*!
 * Decodes and prints a register using priv_dump.
 */
#define PRINT_REG_PD(d, r) do{          \
    priv_dump(QUOTE_ME(LW ## d ## r));  \
}while(0)

#define PRINT_REG_PD_Z(d, r) do{                                 \
    priv_dump_register(QUOTE_ME(LW ## d ## r),                  \
                       PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES);   \
}while(0)

/*!
 * Decodes and prints a unit's register using priv_dump_register.
 * @param unit Can be BE or GPC.
 */
#define PRINT_UNIT_REG_PD1(unit, strReg, nUnit, zeroes) do{                     \
    sprintf( buffer, "LW_PGRAPH_PRI_%s%d_%s", #unit, (LwU32)nUnit, #strReg );   \
    priv_dump_register(buffer, zeroes);                                         \
}while(0)

/*!
 * Decodes and prints a unit's register using priv_dump_register.
 * @param unit Only GPC is used here.
 * @param subunit Can be TPC or PPC.
 */
#define PRINT_UNIT_REG_PD2(unit, subunit, strReg, nUnit, nSubunit, zeroes) do{  \
    sprintf( buffer, "LW_PGRAPH_PRI_%s%d_%s%d_%s",                              \
             #unit, (LwU32)nUnit, #subunit, (LwU32)nSubunit, #strReg );         \
    priv_dump_register(buffer, zeroes);                                         \
}while(0)

/*!
 * Decodes and prints a unit's register using priv_dump_register.
 * @param unit Only GPC is used here.
 * @param subunit Can be TPC or PPC.
 * @param subunit2 Can be SM.
 */
#define PRINT_UNIT_REG_PD3(unit, subunit, subunit2, strReg, nUnit, nSubunit, nSubunit2, zeroes) do{ \
    sprintf( buffer, "LW_PGRAPH_PRI_%s%d_%s%d_%s%d_%s",                                             \
             #unit, (LwU32)nUnit, #subunit, (LwU32)nSubunit, #subunit2, (LwU32)nSubunit2, #strReg );\
    priv_dump_register(buffer, zeroes);                                         \
}while(0)

/*!
 * Macro specifically added for reading CBMGR registers starting from Pascal. See http://lwbugs/1559723.
 * It defines specific decodings for CBM_STATUS_APERTURE register indices 0 through 4.
 */
#define GR_DUMP_APERTURE_FIELDS(regVal, apInd) do{                                                                      \
    LwU32 val;                                                                                                          \
    val = DRF_VAL ( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _VAL, regVal );                                        \
    DPRINTF_FIELD ( "VAL", NULL, val );                                                                                 \
    switch ( apInd ) {                                                                                                  \
        case 0:                                                                                                         \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _0_ALPHA_FSM, regVal );                         \
            DPRINTF_FIELD ( "ALPHA_FSM", NULL, val  );                                                                  \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _0_BETA_FSM, regVal );                          \
            DPRINTF_FIELD ( "BETA_FSM", NULL, val  );                                                                   \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _0_WWDX2CBMGR, regVal );                        \
            DPRINTF_FIELD ( "WWDX2CBMGR", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _0_AB_MERGE, regVal );                          \
            DPRINTF_FIELD ( "AB_MERGE", NULL, val  );                                                                   \
            break;                                                                                                      \
        case 1:                                                                                                         \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _1_ALPHA_COUNT, regVal );                       \
            DPRINTF_FIELD ( "ALPHA_COUNT", NULL, val  );                                                                \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _1_BETA_COUNT, regVal );                        \
            DPRINTF_FIELD ( "BETA_COUNT", NULL, val  );                                                                 \
            break;                                                                                                      \
        case 2:                                                                                                         \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_ALPHA_ALLOC, regVal );                       \
            DPRINTF_FIELD ( "ALPHA_ALLOC", NULL, val  );                                                                \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_ALPHA_ALLOC_ALLOC, regVal );                 \
            DPRINTF_FIELD ( "ALPHA_ALLOC_ALLOC", NULL, val  );                                                          \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_BETA_ALLOC, regVal );                        \
            DPRINTF_FIELD ( "BETA_ALLOC", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_BETA_ALLOC_ALLOC, regVal );                  \
            DPRINTF_FIELD ( "BETA_ALLOC_ALLOC", NULL, val  );                                                           \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_PE2CBMGR_VSC, regVal );                      \
            DPRINTF_FIELD ( "PE2CBMGR_VSC", NULL, val  );                                                               \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_CBMGR2PE_VSC, regVal );                      \
            DPRINTF_FIELD ( "CBMGR2PE_VSC", NULL, val  );                                                               \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_CBMGR2PE_PIN, regVal );                      \
            DPRINTF_FIELD ( "CBMGR2PE_PIN", NULL, val  );                                                               \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_PE2CBMGR_ILWACK, regVal );                   \
            DPRINTF_FIELD ( "PE2CBMGR_ILWACK", NULL, val  );                                                            \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_ALPHA_SIZE_GT_MAX_HANG, regVal );            \
            DPRINTF_FIELD ( "ALPHA_SIZE_GT_MAX_HANG", NULL, val  );                                                     \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _2_BETA_SIZE_GT_MAX_HANG, regVal );             \
            DPRINTF_FIELD ( "BETA_SIZE_GT_MAX_HANG", NULL, val  );                                                      \
            break;                                                                                                      \
        case 3:                                                                                                         \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_MPC2CBMGR0, regVal );                        \
            DPRINTF_FIELD ( "MPC2CBMGR0", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_CBMGR2MPC0, regVal );                        \
            DPRINTF_FIELD ( "CBMGR2MPC0", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_MPC2CBMGR1, regVal );                        \
            DPRINTF_FIELD ( "MPC2CBMGR1", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_CBMGR2MPC1, regVal );                        \
            DPRINTF_FIELD ( "CBMGR2MPC1", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_MPC2CBMGR2, regVal );                        \
            DPRINTF_FIELD ( "MPC2CBMGR2", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_CBMGR2MPC2, regVal );                        \
            DPRINTF_FIELD ( "CBMGR2MPC2", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_MPC2CBMGR3, regVal );                        \
            DPRINTF_FIELD ( "MPC2CBMGR3", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_CBMGR2MPC3, regVal );                        \
            DPRINTF_FIELD ( "CBMGR2MPC3", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_MPC2CBMGR4, regVal );                        \
            DPRINTF_FIELD ( "MPC2CBMGR4", NULL, val  );                                                                 \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _3_CBMGR2MPC4, regVal );                        \
            DPRINTF_FIELD ( "CBMGR2MPC4", NULL, val  );                                                                 \
            break;                                                                                                      \
        case 4:                                                                                                         \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _4_ALPHA_ALLOC_SPACE, regVal );                 \
            DPRINTF_FIELD ( "ALPHA_ALLOC_SPACE", NULL, val  );                                                          \
            val = DRF_VAL( _PGRAPH, _PRI_GPC0_PPC0_CBM_STATUS_APERTURE, _4_BETA_ALLOC_SPACE, regVal );                  \
            DPRINTF_FIELD ( "BETA_ALLOC_SPACE", NULL, val  );                                                           \
            break;                                                                                                      \
        default:                                                                                                        \
            dprintf ("Field decodings corresponding to aperture index %u are not defined in inc/gr.h!\n", apInd);       \
    }                                                                                                                   \
}while(0)                                                                                                               \

/*!
 * Decodes and prints a GPC register using priv_dump_register.
 */
#define PRINT_GPC_REG_PD(strReg, nGpc)        do{                       \
    PRINT_UNIT_REG_PD1(GPC, strReg, nGpc,                               \
                       PRIV_DUMP_REGISTER_FLAGS_DEFAULT );              \
}while(0)

#define PRINT_GPC_REG_PD_Z(strReg, nGpc)        do{                      \
    PRINT_UNIT_REG_PD1(GPC, strReg, nGpc,                               \
                       PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES );          \
}while(0)

/*!
 * Decodes and prints a BE register using priv_dump.
 */
#define PRINT_BE_REG_PD(strReg, nFbp)        do{                        \
    PRINT_UNIT_REG_PD1(BE, strReg, nFbp,                                \
                       PRIV_DUMP_REGISTER_FLAGS_DEFAULT );              \
}while(0)

#define PRINT_BE_REG_PD_Z(strReg, nFbp)        do{                       \
    PRINT_UNIT_REG_PD1(BE, strReg, nFbp,                                \
                       PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES );          \
}while(0)

/*!
 * Decodes and prints a TPC register using priv_dump.  2nd
 * form skips zero values - used in consolidated printouts
 */
#define PRINT_TPC_REG_PD(strReg, nGpc, nTpc)         do{                \
    PRINT_UNIT_REG_PD2(GPC, TPC, strReg, nGpc, nTpc,                    \
                       PRIV_DUMP_REGISTER_FLAGS_DEFAULT );              \
}while(0)

#define PRINT_TPC_REG_PD_Z(strReg, nGpc, nTpc)         do{              \
    PRINT_UNIT_REG_PD2(GPC, TPC, strReg, nGpc, nTpc,                    \
                       PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES );          \
}while(0)

/*!
 * Decodes and prints a SM register using priv_dump.  2nd
 * form skips zero values - used in consolidated printouts
 */
#define PRINT_SM_REG_PD(strReg, nGpc, nTpc, nSm)         do{            \
        PRINT_UNIT_REG_PD3(GPC, TPC, SM, strReg, nGpc, nTpc, nSm,       \
                       PRIV_DUMP_REGISTER_FLAGS_DEFAULT );              \
}while(0)

#define PRINT_SM_REG_PD_Z(strReg, nGpc, nTpc, nSm)         do{          \
        PRINT_UNIT_REG_PD3(GPC, TPC, SM, strReg, nGpc, nTpc, nSm,       \
                       PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES );          \
}while(0)

/*!
 * Decodes and prints a ROP register using priv_dump.
 */
#define PRINT_ROP_REG_PD(strReg, nGpc, nRop)         do{                \
    PRINT_UNIT_REG_PD2(GPC, ROP, strReg, nGpc, nRop,                    \
                       PRIV_DUMP_REGISTER_FLAGS_DEFAULT );              \
}while(0)

#define PRINT_ROP_REG_PD_Z(strReg, nGpc, nRop)         do{              \
    PRINT_UNIT_REG_PD2(GPC, ROP, strReg, nGpc, nRop,                    \
                       PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES );          \
}while(0)

/*!
 * Decodes and prints a PPC register using priv_dump.
 */
#define PRINT_PPC_REG_PD(strReg, nGpc, nPpc)         do{                \
    PRINT_UNIT_REG_PD2(GPC, PPC, strReg, nGpc, nPpc,                    \
                       PRIV_DUMP_REGISTER_FLAGS_DEFAULT );              \
}while(0)

//-----------------------------------------------------
// Hardcoded register dump macros used by the default grstatus report,
// which must work without manuals.  "grstatus -a" reports can use
// priv_dump which requires manuals.
//-----------------------------------------------------

// _PGRAPH, _STATUS
#define GR_REG_FIELDS_PGRAPH_STATUS_GF100(d, r) \
    PRINT_GR_Z( d, r, _STATE, "BUSY");                          \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_METHOD_LOWER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_FUNNEL, "BUSY");                      \
    PRINT_GR_Z( d, r, _FE_NOTIFY, "BUSY");                      \
    PRINT_GR_Z( d, r, _SEMAPHORE, "BUSY");                      \
    PRINT_GR_Z( d, r, _MEMFMT, "BUSY");                         \
    PRINT_GR_Z( d, r, _CONTEXT_SWITCH, "BUSY");                 \
    PRINT_GR_Z( d, r, _PD, "BUSY");                             \
    PRINT_GR_Z( d, r, _PDB, "BUSY");                            \
    PRINT_GR_Z( d, r, _SCC, "BUSY");                            \
    PRINT_GR_Z( d, r, _SSYNC, "BUSY");                          \
    PRINT_GR_Z( d, r, _CWD, "BUSY");                            \
    PRINT_GR_Z( d, r, _RASTWOD, "BUSY");                        \
    PRINT_GR_Z( d, r, _PMA, "BUSY");                            \
    PRINT_GR_Z( d, r, _PMMSYS, "BUSY");                         \
    PRINT_GR_Z( d, r, _FB, "BUSY");                             \
    PRINT_GR_Z( d, r, _GPC, "BUSY");                            \
    PRINT_GR_Z( d, r, _BE, "BUSY");

// _PGRAPH, _STATUS
#define GR_REG_FIELDS_PGRAPH_STATUS_GK104(d, r)                 \
    PRINT_GR_Z( d, r, _STATE, "BUSY");                          \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_METHOD_LOWER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_FUNNEL, "BUSY");                      \
    PRINT_GR_Z( d, r, _FE_NOTIFY, "BUSY");                      \
    PRINT_GR_Z( d, r, _SEMAPHORE, "BUSY");                      \
    PRINT_GR_Z( d, r, _MEMFMT, "BUSY");                         \
    PRINT_GR_Z( d, r, _CONTEXT_SWITCH, "BUSY");                 \
    PRINT_GR_Z( d, r, _PD, "BUSY");                             \
    PRINT_GR_Z( d, r, _PDB, "BUSY");                            \
    PRINT_GR_Z( d, r, _SCC, "BUSY");                            \
    PRINT_GR_Z( d, r, _SSYNC, "BUSY");                          \
    PRINT_GR_Z( d, r, _CWD, "BUSY");                            \
    PRINT_GR_Z( d, r, _RASTWOD, "BUSY");                        \
    PRINT_GR_Z( d, r, _PMA, "BUSY");                            \
    PRINT_GR_Z( d, r, _PMMSYS, "BUSY");                         \
    PRINT_GR_Z( d, r, _FB, "BUSY");                             \
    PRINT_GR_Z( d, r, _SKED, "BUSY");                           \
    PRINT_GR_Z( d, r, _FE_CONST, "BUSY");                       \
    PRINT_GR_Z( d, r, _GPC, "BUSY");                            \
    PRINT_GR_Z( d, r, _BE, "BUSY");

// _PGRAPH, _STATUS
#define GR_REG_FIELDS_PGRAPH_STATUS_GM200(d, r) \
    PRINT_GR_Z( d, r, _STATE, "BUSY");                          \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_METHOD_LOWER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_FUNNEL, "BUSY");                      \
    PRINT_GR_Z( d, r, _FE_NOTIFY, "BUSY");                      \
    PRINT_GR_Z( d, r, _SEMAPHORE, "BUSY");                      \
    PRINT_GR_Z( d, r, _MEMFMT, "BUSY");                         \
    PRINT_GR_Z( d, r, _CONTEXT_SWITCH, "BUSY");                 \
    PRINT_GR_Z( d, r, _PD, "BUSY");                             \
    PRINT_GR_Z( d, r, _PDB, "BUSY");                            \
    PRINT_GR_Z( d, r, _SCC, "BUSY");                            \
    PRINT_GR_Z( d, r, _SSYNC, "BUSY");                          \
    PRINT_GR_Z( d, r, _CWD, "BUSY");                            \
    PRINT_GR_Z( d, r, _RASTWOD, "BUSY");                        \
    PRINT_GR_Z( d, r, _FB, "BUSY");                             \
    PRINT_GR_Z( d, r, _SKED, "BUSY");                           \
    PRINT_GR_Z( d, r, _FE_CONST, "BUSY");                       \
    PRINT_GR_Z( d, r, _FE_GI, "BUSY");                          \
    PRINT_GR_Z( d, r, _TPC_MGR, "BUSY");                        \
    PRINT_GR_Z( d, r, _GPC, "BUSY");                            \
    PRINT_GR_Z( d, r, _BE, "BUSY");

// _PGRAPH, _STATUS
#define GR_REG_FIELDS_PGRAPH_STATUS_TU102(d, r) \
    PRINT_GR_Z( d, r, _STATE, "BUSY");                          \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_METHOD_LOWER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_FUNNEL, "BUSY");                      \
    PRINT_GR_Z( d, r, _FE_NOTIFY, "BUSY");                      \
    PRINT_GR_Z( d, r, _SEMAPHORE, "BUSY");                      \
    PRINT_GR_Z( d, r, _MEMFMT, "BUSY");                         \
    PRINT_GR_Z( d, r, _CONTEXT_SWITCH, "BUSY");                 \
    PRINT_GR_Z( d, r, _PD, "BUSY");                             \
    PRINT_GR_Z( d, r, _PDB, "BUSY");                            \
    PRINT_GR_Z( d, r, _SCC, "BUSY");                            \
    PRINT_GR_Z( d, r, _SSYNC, "BUSY");                          \
    PRINT_GR_Z( d, r, _CWD, "BUSY");                            \
    PRINT_GR_Z( d, r, _RASTWOD, "BUSY");                        \
    PRINT_GR_Z( d, r, _FB, "BUSY");                             \
    PRINT_GR_Z( d, r, _SKED, "BUSY");                           \
    PRINT_GR_Z( d, r, _FE_CONST, "BUSY");                       \
    PRINT_GR_Z( d, r, _FE_GI, "BUSY");                          \
    PRINT_GR_Z( d, r, _TPC_MGR, "BUSY");                        \
    PRINT_GR_Z( d, r, _GPC, "BUSY");                            \
    PRINT_GR_Z( d, r, _BE, "BUSY");                             \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER_FE1, "BUSY");            \
    PRINT_GR_Z( d, r, _FE_NOTIFY_FE1, "BUSY");

// _PGRAPH, _STATUS
#define GR_REG_FIELDS_PGRAPH_STATUS_GA100(d, r) \
    PRINT_GR_Z( d, r, _STATE, "BUSY");                          \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_METHOD_LOWER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_FUNNEL, "BUSY");                      \
    PRINT_GR_Z( d, r, _FE_NOTIFY, "BUSY");                      \
    PRINT_GR_Z( d, r, _SEMAPHORE, "BUSY");                      \
    PRINT_GR_Z( d, r, _MEMFMT, "BUSY");                         \
    PRINT_GR_Z( d, r, _CONTEXT_SWITCH, "BUSY");                 \
    PRINT_GR_Z( d, r, _PD, "BUSY");                             \
    PRINT_GR_Z( d, r, _PDB, "BUSY");                            \
    PRINT_GR_Z( d, r, _SCC, "BUSY");                            \
    PRINT_GR_Z( d, r, _SSYNC, "BUSY");                          \
    PRINT_GR_Z( d, r, _CWD, "BUSY");                            \
    PRINT_GR_Z( d, r, _RASTWOD, "BUSY");                        \
    PRINT_GR_Z( d, r, _SMCARB, "BUSY");                        \
    PRINT_GR_Z( d, r, _FB, "BUSY");                             \
    PRINT_GR_Z( d, r, _SKED, "BUSY");                           \
    PRINT_GR_Z( d, r, _FE_CONST, "BUSY");                       \
    PRINT_GR_Z( d, r, _FE_GI, "BUSY");                          \
    PRINT_GR_Z( d, r, _TPC_MGR, "BUSY");                        \
    PRINT_GR_Z( d, r, _GPC, "BUSY");                            \
    PRINT_GR_Z( d, r, _BE, "BUSY");                             \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER_FE1, "BUSY");            \
    PRINT_GR_Z( d, r, _FE_NOTIFY_FE1, "BUSY");

// _PGRAPH, _STATUS
#define GR_REG_FIELDS_PGRAPH_STATUS_GA102(d, r) \
    PRINT_GR_Z( d, r, _STATE, "BUSY");                          \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_METHOD_LOWER, "BUSY");                \
    PRINT_GR_Z( d, r, _FE_FUNNEL, "BUSY");                      \
    PRINT_GR_Z( d, r, _FE_NOTIFY, "BUSY");                      \
    PRINT_GR_Z( d, r, _SEMAPHORE, "BUSY");                      \
    PRINT_GR_Z( d, r, _MEMFMT, "BUSY");                         \
    PRINT_GR_Z( d, r, _CONTEXT_SWITCH, "BUSY");                 \
    PRINT_GR_Z( d, r, _PD, "BUSY");                             \
    PRINT_GR_Z( d, r, _PDB, "BUSY");                            \
    PRINT_GR_Z( d, r, _SCC, "BUSY");                            \
    PRINT_GR_Z( d, r, _SSYNC, "BUSY");                          \
    PRINT_GR_Z( d, r, _CWD, "BUSY");                            \
    PRINT_GR_Z( d, r, _RASTWOD, "BUSY");                        \
    PRINT_GR_Z( d, r, _SMCARB, "BUSY");                        \
    PRINT_GR_Z( d, r, _FB, "BUSY");                             \
    PRINT_GR_Z( d, r, _SKED, "BUSY");                           \
    PRINT_GR_Z( d, r, _FE_CONST, "BUSY");                       \
    PRINT_GR_Z( d, r, _FE_GI, "BUSY");                          \
    PRINT_GR_Z( d, r, _TPC_MGR, "BUSY");                        \
    PRINT_GR_Z( d, r, _GPC, "BUSY");                            \
    PRINT_GR_Z( d, r, _FE_METHOD_UPPER_FE1, "BUSY");            \
    PRINT_GR_Z( d, r, _FE_NOTIFY_FE1, "BUSY");

// _PGRAPH, _INTR
#define GR_REG_FIELDS_PGRAPH_INTR_GF100(d, r) \
    PRINT_GR_Z( d, r, _NOTIFY,           "PENDING" ); \
    PRINT_GR_Z( d, r, _SEMAPHORE,        "PENDING" ); \
    PRINT_GR_Z( d, r, _SEMAPHORE_TIMEOUT,"PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_METHOD,   "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_CLASS,    "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_NOTIFY,   "PENDING" ); \
    PRINT_GR_Z( d, r, _DEBUG_METHOD,     "PENDING" ); \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD,  "PENDING" ); \
    PRINT_GR_Z( d, r, _BUFFER_NOTIFY,    "PENDING" ); \
    PRINT_GR_Z( d, r, _FECS_ERROR,       "PENDING" ); \
    PRINT_GR_Z( d, r, _CLASS_ERROR,      "PENDING" ); \
    PRINT_GR_Z( d, r, _EXCEPTION,        "PENDING" );

// _PGRAPH, _INTR
#define GR_REG_FIELDS_PGRAPH_INTR_GM107(d, r) \
    PRINT_GR_Z( d, r, _NOTIFY,           "PENDING" ); \
    PRINT_GR_Z( d, r, _SEMAPHORE,        "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_METHOD,   "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_CLASS,    "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_NOTIFY,   "PENDING" ); \
    PRINT_GR_Z( d, r, _DEBUG_METHOD,     "PENDING" ); \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD,  "PENDING" ); \
    PRINT_GR_Z( d, r, _BUFFER_NOTIFY,    "PENDING" ); \
    PRINT_GR_Z( d, r, _FECS_ERROR,       "PENDING" ); \
    PRINT_GR_Z( d, r, _CLASS_ERROR,      "PENDING" ); \
    PRINT_GR_Z( d, r, _EXCEPTION,        "PENDING" );

// _PGRAPH, _INTR
#define GR_REG_FIELDS_PGRAPH_INTR_GV100(d, r) \
    PRINT_GR_Z( d, r, _NOTIFY,           "PENDING" ); \
    PRINT_GR_Z( d, r, _SEMAPHORE,        "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_METHOD,   "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_CLASS,    "PENDING" ); \
    PRINT_GR_Z( d, r, _ILLEGAL_NOTIFY,   "PENDING" ); \
    PRINT_GR_Z( d, r, _DEBUG_METHOD,     "PENDING" ); \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD,  "PENDING" ); \
    PRINT_GR_Z( d, r, _BUFFER_NOTIFY,    "PENDING" ); \
    PRINT_GR_Z( d, r, _FECS_ERROR,       "PENDING" ); \
    PRINT_GR_Z( d, r, _CLASS_ERROR,      "PENDING" ); \
    PRINT_GR_Z( d, r, _EXCEPTION,        "PENDING" );

// _PGRAPH, _GRFIFO_STATUS
#define GR_REG_FIELDS_PGRAPH_GRFIFO_STATUS(d, r) \
    PRINT_GR_Z( d, r, _EMPTY,            NULL );         \
    PRINT_GR_Z( d, r, _FULL,             NULL );         \
    PRINT_GR_Z( d, r, _COUNT,            NULL );      \
    PRINT_GR_Z( d, r, _READ_PTR,         NULL );      \
    PRINT_GR_Z( d, r, _WRITE_PTR,        NULL );

// _PGRAPH, _FECS_INTR
#define GR_REG_FIELDS_PGRAPH_FECS_INTR(d, r) \
    PRINT_GR_Z( d, r, _ILLEGAL_METHOD,   "PENDING" ); \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD,  "PENDING" );

// _PGRAPH, _PRI_FECS_HOST_INT_STATUS
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_HOST_INT_STATUS(d, r) \
    PRINT_GR( d, r, _CTXSW_INTR ); \
    PRINT_GR_Z( d, r, _FAULT_DURING_CTXSW,    "ACTIVE" ); \
    PRINT_GR_Z( d, r, _UMIMP_FIRMWARE_METHOD, "ACTIVE" ); \
    PRINT_GR_Z( d, r, _UMIMP_ILLEGAL_METHOD,  "ACTIVE" ); \
    PRINT_GR_Z( d, r, _WATCHDOG,              "ACTIVE" ); \

// _PGRAPH, _PRI_FECS_HOST_INT_STATUS
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_HOST_INT_STATUS_TU102(d, r) \
    PRINT_GR( d, r, _CTXSW_INTR ); \
    PRINT_GR_Z( d, r, _FAULT_DURING_CTXSW,    "ACTIVE" ); \
    PRINT_GR_Z( d, r, _UMIMP_FIRMWARE_METHOD, "ACTIVE" ); \
    PRINT_GR_Z( d, r, _UMIMP_ILLEGAL_METHOD,  "ACTIVE" ); \
    PRINT_GR_Z( d, r, _WATCHDOG,              "ACTIVE" ); \
    PRINT_GR_Z( d, r, _FLUSH_WHEN_BUSY,       "ACTIVE" ); \
    PRINT_GR_Z( d, r, _ECC_CORRECTED,         "ACTIVE" ); \
    PRINT_GR_Z( d, r, _ECC_UNCORRECTED,       "ACTIVE" ); \

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_FE_0_GF100(d, r) \
    PRINT_GR_Z( d, r, _CTXSW_REQ, "REQ" );                      \
    PRINT_GR_Z( d, r, _CTXSW_VLD, "VALID" );                    \
    PRINT_GR_Z( d, r, _CTXSW_ACK, "ACK" );                      \
    PRINT_GR_Z( d, r, _STALL_REQ, "REQ" );                      \
    PRINT_GR_Z( d, r, _STALL_ACK, "ACK" );                      \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_REQ, "REQ" );        \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_ACK, "ACK" );        \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_REQ, "REQ" );       \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_ACK, "ACK" );       \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD_INTERRUPT, "REQ" );      \
    PRINT_GR_Z( d, r, _UNIMPLEMENTED_METHOD_INTERRUPT, "REQ" ); \
    PRINT_GR_Z( d, r, _FE_WFI_REQ, "REQ" );                     \
    PRINT_GR_Z( d, r, _FE_WFI_ACK, "ACK" );                     \
    PRINT_GR_Z( d, r, _FE_HALT_REQ, "REQ" );                    \
    PRINT_GR_Z( d, r, _FE_HALT_ACK, "ACK" );                    \
    PRINT_GR_Z( d, r, _GR_INTR, "REQ" );                        \
    PRINT_GR_Z( d, r, _DFD_INTR, "REQ" );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_FE_0_GK104(d, r) \
    PRINT_GR_Z( d, r, _CTXSW_REQ, "REQ" );                      \
    PRINT_GR_Z( d, r, _CTXSW_VLD, "VALID" );                    \
    PRINT_GR_Z( d, r, _CTXSW_ACK, "ACK" );                      \
    PRINT_GR_Z( d, r, _CTXSW_INT, "REQ" );                      \
    PRINT_GR_Z( d, r, _STALL_REQ, "REQ" );                      \
    PRINT_GR_Z( d, r, _STALL_ACK, "ACK" );                      \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_REQ, "REQ" );        \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_ACK, "ACK" );        \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_REQ, "REQ" );       \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_ACK, "ACK" );       \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD_INTERRUPT, "REQ" );      \
    PRINT_GR_Z( d, r, _UNIMPLEMENTED_METHOD_INTERRUPT, "REQ" ); \
    PRINT_GR_Z( d, r, _FE_WFI_REQ, "REQ" );                     \
    PRINT_GR_Z( d, r, _FE_WFI_ACK, "ACK" );                     \
    PRINT_GR_Z( d, r, _FE_HALT_REQ, "REQ" );                    \
    PRINT_GR_Z( d, r, _FE_HALT_ACK, "ACK" );                    \
    PRINT_GR_Z( d, r, _GR_INTR, "REQ" );                        \
    PRINT_GR_Z( d, r, _DFD_INTR, "REQ" );                       \
    PRINT_GR_Z( d, r, _FE_WFI_PREEMPT, "ACTIVE" );              \
    PRINT_GR_Z( d, r, _RING_MANAGE_STATUS, "SUCCESS" );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_FE_0_GV100(d, r) \
    PRINT_GR_Z( d, r, _CTXSW_REQ, "REQ" );                          \
    PRINT_GR_Z( d, r, _CTXSW_VLD, "VALID" );                        \
    PRINT_GR_Z( d, r, _CTXSW_ACK, "ACK" );                          \
    PRINT_GR_Z( d, r, _CTXSW_INT, "REQ" );                          \
    PRINT_GR_Z( d, r, _STALL_REQ, "REQ" );                          \
    PRINT_GR_Z( d, r, _STALL_ACK, "ACK" );                          \
    PRINT_GR_Z( d, r, _CONTEXT_PREEMPTED, "TRUE" );                 \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_REQ, "REQ" );            \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_ACK, "ACK" );            \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_REQ, "REQ" );           \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_ACK, "ACK" );           \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD_INTERRUPT, "REQ" );          \
    PRINT_GR_Z( d, r, _UNIMPLEMENTED_METHOD_INTERRUPT, "REQ" );     \
    PRINT_GR_Z( d, r, _FE_WFI_REQ, "REQ" );                         \
    PRINT_GR_Z( d, r, _FE_WFI_ACK, "ACK" );                         \
    PRINT_GR_Z( d, r, _FE_HALT_REQ, "REQ" );                        \
    PRINT_GR_Z( d, r, _FE_HALT_ACK, "ACK" );                        \
    PRINT_GR_Z( d, r, _GR_INTR, "REQ" );                            \
    PRINT_GR_Z( d, r, _DFD_INTR, "REQ" );                           \
    PRINT_GR_Z( d, r, _RING_MANAGE_STATUS, "SUCCESS" );             \
    PRINT_GR_Z( d, r, _TBCS_ABORT_CTXSW, "TRUE" );                  \
    PRINT_GR_Z( d, r, _TBCS_NACK_RECEIVED, "TRUE" );                \
    PRINT_GR_Z( d, r, _TBCS_NACK_ACTION , "TRUE" );                 \
    PRINT_GR_Z( d, r, _TBCS_DUT_FAULTED , "TRUE" );                 \
    PRINT_GR_Z( d, r, _GFX_PREEMPTION_TYPE_ISSUED_BY_FE , "GFXP" ); \
    PRINT_GR_Z( d, r, _COMPUTE_PREEMPTION_TYPE_ISSUED_BY_FE, "BUSY" );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_FE_0_TU102(d, r) \
    PRINT_GR_Z( d, r, _CTXSW_REQ, "REQ" );                          \
    PRINT_GR_Z( d, r, _CTXSW_VLD, "VALID" );                        \
    PRINT_GR_Z( d, r, _CTXSW_ACK, "ACK" );                          \
    PRINT_GR_Z( d, r, _CTXSW_INT, "REQ" );                          \
    PRINT_GR_Z( d, r, _STALL_REQ, "REQ" );                          \
    PRINT_GR_Z( d, r, _STALL_ACK, "ACK" );                          \
    PRINT_GR_Z( d, r, _CONTEXT_PREEMPTED, "TRUE" );                 \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_REQ, "REQ" );            \
    PRINT_GR_Z( d, r, _PMU_SLAVE_INTERRUPT_ACK, "ACK" );            \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_REQ, "REQ" );           \
    PRINT_GR_Z( d, r, _PMU_MASTER_INTERRUPT_ACK, "ACK" );           \
    PRINT_GR_Z( d, r, _FIRMWARE_METHOD_INTERRUPT, "REQ" );          \
    PRINT_GR_Z( d, r, _UNIMPLEMENTED_METHOD_INTERRUPT, "REQ" );     \
    PRINT_GR_Z( d, r, _FE_WFI_REQ, "REQ" );                         \
    PRINT_GR_Z( d, r, _FE_WFI_ACK, "ACK" );                         \
    PRINT_GR_Z( d, r, _FE_HALT_REQ, "REQ" );                        \
    PRINT_GR_Z( d, r, _FE_HALT_ACK, "ACK" );                        \
    PRINT_GR_Z( d, r, _GR_INTR, "REQ" );                            \
    PRINT_GR_Z( d, r, _DFD_INTR, "REQ" );                           \
    PRINT_GR_Z( d, r, _FE_INTERVAL_TIMER_INTR, "REQ" );             \
    PRINT_GR_Z( d, r, _RING_MANAGE_STATUS, "SUCCESS" );             \
    PRINT_GR_Z( d, r, _TBCS_ABORT_CTXSW, "TRUE" );                  \
    PRINT_GR_Z( d, r, _TBCS_NACK_RECEIVED, "TRUE" );                \
    PRINT_GR_Z( d, r, _TBCS_NACK_ACTION , "TRUE" );                 \
    PRINT_GR_Z( d, r, _TBCS_DUT_FAULTED , "TRUE" );                 \
    PRINT_GR_Z( d, r, _GFX_PREEMPTION_TYPE_ISSUED_BY_FE , "GFXP" ); \
    PRINT_GR_Z( d, r, _COMPUTE_PREEMPTION_TYPE_ISSUED_BY_FE, "BUSY" );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_1
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_1_GF100(d, r) \
    PRINT_GR_Z( d, r, _CONSTANT_0, NULL );                       \
    PRINT_GR_Z( d, r, _LOCAL_PRIV_ERROR, NULL );                 \
    PRINT_GR_Z( d, r, _RC_LANES, NULL );                         \
    PRINT_GR_Z( d, r, _CONTEXT_RESET, "DISABLED" );              \
    PRINT_GR_Z( d, r, _FE_PRIV, NULL );                          \
    PRINT_GR_Z( d, r, _PRIV_SEQUENCER, "BUSY" );                 \
    PRINT_GR_Z( d, r, _READ, "DONE" );                           \
    PRINT_GR_Z( d, r, _WRACK, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP0, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP1, "DONE" );                          \
    PRINT_GR_Z( d, r, _DFD_TRIGGER, "ACTIVE" );                  \
    PRINT_GR_Z( d, r, _WATCHDOG_TRIGGER, "ACTIVE" );             \
    PRINT_GR_Z( d, r, _ARB_BUSY, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_NACK, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_HALTED, NULL );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_1
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_1_GK104(d, r) \
    PRINT_GR_Z( d, r, _CONSTANT_0, NULL );                       \
    PRINT_GR_Z( d, r, _LOCAL_PRIV_ERROR, NULL );                 \
    PRINT_GR_Z( d, r, _RC_LANES, "BUSY" );                       \
    PRINT_GR_Z( d, r, _CONTEXT_RESET, "DISABLED" );              \
    PRINT_GR_Z( d, r, _FE_PRIV, NULL );                          \
    PRINT_GR_Z( d, r, _PRIV_SEQUENCER, "BUSY" );                 \
    PRINT_GR_Z( d, r, _READ, "DONE" );                           \
    PRINT_GR_Z( d, r, _WRACK, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP0, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP1, "DONE" );                          \
    PRINT_GR_Z( d, r, _DFD_TRIGGER, "ACTIVE" );                  \
    PRINT_GR_Z( d, r, _WATCHDOG_TRIGGER, "ACTIVE" );             \
    PRINT_GR_Z( d, r, _ARB_BUSY, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_NACK, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_HALTED, NULL );                       \
    PRINT_GR_Z( d, r, _ARB_FLUSH_WHEN_BUSY, NULL );              \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP, NULL );                 \
    PRINT_GR_Z( d, r, _PRIV_UCODE_TRAP, NULL );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_1
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_1_GM107(d, r) \
    PRINT_GR_Z( d, r, _CONSTANT_0, NULL );                       \
    PRINT_GR_Z( d, r, _LOCAL_PRIV_ERROR, NULL );                 \
    PRINT_GR_Z( d, r, _RC_LANES, "BUSY" );                       \
    PRINT_GR_Z( d, r, _CONTEXT_RESET, "DISABLED" );              \
    PRINT_GR_Z( d, r, _FE_PRIV, NULL );                          \
    PRINT_GR_Z( d, r, _PRIV_SEQUENCER, "BUSY" );                 \
    PRINT_GR_Z( d, r, _READ, "DONE" );                           \
    PRINT_GR_Z( d, r, _WRACK, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP0, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP1, "DONE" );                          \
    PRINT_GR_Z( d, r, _DFD_TRIGGER, "ACTIVE" );                  \
    PRINT_GR_Z( d, r, _WATCHDOG_TRIGGER, "ACTIVE" );             \
    PRINT_GR_Z( d, r, _ARB_BUSY, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_NACK, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_HALTED, NULL );                       \
    PRINT_GR_Z( d, r, _ARB_FLUSH_WHEN_BUSY, NULL );              \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP_PL, "VIOLATION" );       \
    PRINT_GR_Z( d, r, _SEQ_PRIV_ERROR, NULL );                   \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP, "WRITE_DROPPED" );      \
    PRINT_GR_Z( d, r, _PRIV_UCODE_TRAP, "PENDING" );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_1
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_1_GM200(d, r) \
    PRINT_GR_Z( d, r, _CONSTANT_0, NULL );                       \
    PRINT_GR_Z( d, r, _LOCAL_PRIV_ERROR, NULL );                 \
    PRINT_GR_Z( d, r, _RC_LANES, "BUSY" );                       \
    PRINT_GR_Z( d, r, _CONTEXT_RESET, "DISABLED" );              \
    PRINT_GR_Z( d, r, _PRIV_SEQUENCER, "BUSY" );                 \
    PRINT_GR_Z( d, r, _READ, "DONE" );                           \
    PRINT_GR_Z( d, r, _WRACK, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP0, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP1, "DONE" );                          \
    PRINT_GR_Z( d, r, _DFD_TRIGGER, "ACTIVE" );                  \
    PRINT_GR_Z( d, r, _WATCHDOG_TRIGGER, "ACTIVE" );             \
    PRINT_GR_Z( d, r, _ARB_BUSY, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_NACK, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_HALTED, NULL );                       \
    PRINT_GR_Z( d, r, _ARB_FLUSH_WHEN_BUSY, NULL );              \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP_PL, "VIOLATION" );       \
    PRINT_GR_Z( d, r, _SEQ_PRIV_ERROR, NULL );                   \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP, "WRITE_DROPPED" );      \
    PRINT_GR_Z( d, r, _PRIV_UCODE_TRAP, "PENDING" );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_1
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_1_GV100(d, r) \
    PRINT_GR_Z( d, r, _CONSTANT_0, NULL );                       \
    PRINT_GR_Z( d, r, _LOCAL_PRIV_ERROR, NULL );                 \
    PRINT_GR_Z( d, r, _RC_LANES, "BUSY" );                       \
    PRINT_GR_Z( d, r, _CONTEXT_RESET, "DISABLED" );              \
    PRINT_GR_Z( d, r, _PRIV_SEQUENCER, "BUSY" );                 \
    PRINT_GR_Z( d, r, _READ, "DONE" );                           \
    PRINT_GR_Z( d, r, _WRACK, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP0, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP1, "DONE" );                          \
    PRINT_GR_Z( d, r, _DFD_TRIGGER, "ACTIVE" );                  \
    PRINT_GR_Z( d, r, _WATCHDOG_TRIGGER, "ACTIVE" );             \
    PRINT_GR_Z( d, r, _ARB_BUSY, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_NACK, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_HALTED, NULL );                       \
    PRINT_GR_Z( d, r, _ARB_FLUSH_WHEN_BUSY, NULL );              \
    PRINT_GR_Z( d, r, _CTXSW_CHECKSUM_MISMATCH, NULL );          \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP_PL, "VIOLATION" );       \
    PRINT_GR_Z( d, r, _SEQ_PRIV_ERROR, NULL );                   \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP, "WRITE_DROPPED" );      \
    PRINT_GR_Z( d, r, _PRIV_UCODE_TRAP, "PENDING" );

// _PGRAPH, _PRI_FECS_CTXSW_STATUS_1
#define GR_REG_FIELDS_PGRAPH_PRI_FECS_CTXSW_STATUS_1_GA100(d, r) \
    PRINT_GR_Z( d, r, _CONSTANT_0, NULL );                       \
    PRINT_GR_Z( d, r, _LOCAL_PRIV_ERROR, NULL );                 \
    PRINT_GR_Z( d, r, _RC_LANES, "BUSY" );                       \
    PRINT_GR_Z( d, r, _CONTEXT_RESET, "DISABLED" );              \
    PRINT_GR_Z( d, r, _PRIV_SEQUENCER, "BUSY" );                 \
    PRINT_GR_Z( d, r, _READ, "DONE" );                           \
    PRINT_GR_Z( d, r, _WRACK, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP0, "DONE" );                          \
    PRINT_GR_Z( d, r, _COMP1, "DONE" );                          \
    PRINT_GR_Z( d, r, _DFD_TRIGGER, "ACTIVE" );                  \
    PRINT_GR_Z( d, r, _WATCHDOG_TRIGGER, "ACTIVE" );             \
    PRINT_GR_Z( d, r, _ARB_BUSY, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_NACK, NULL );                         \
    PRINT_GR_Z( d, r, _ARB_HALTED, NULL );                       \
    PRINT_GR_Z( d, r, _ARB_FLUSH_WHEN_BUSY, NULL );              \
    PRINT_GR_Z( d, r, _CTXSW_CHECKSUM_MISMATCH, NULL );          \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP_PL, "VIOLATION" );       \
    PRINT_GR_Z( d, r, _PRIV_ERROR, NULL );                   \
    PRINT_GR_Z( d, r, _PRIV_DECODE_TRAP, "WRITE_DROPPED" );      \
    PRINT_GR_Z( d, r, _PRIV_UCODE_TRAP, "PENDING" );

// _PGRAPH, _EXCEPTION
#define GR_REG_FIELDS_PGRAPH_EXCEPTION_GF100(d, r) \
    PRINT_GR_Z( d, r, _FE,      "PENDING" ); \
    PRINT_GR_Z( d, r, _MEMFMT,  "PENDING" ); \
    PRINT_GR_Z( d, r, _PD,      "PENDING" ); \
    PRINT_GR_Z( d, r, _SCC,     "PENDING" ); \
    PRINT_GR_Z( d, r, _DS,      "PENDING" ); \
    PRINT_GR_Z( d, r, _SSYNC,   "PENDING" ); \
    PRINT_GR_Z( d, r, _MME,     "PENDING" ); \
    PRINT_GR_Z( d, r, _GPC,     "PENDING" ); \
    PRINT_GR_Z( d, r, _BE,      "PENDING" ); \

#define GR_REG_FIELDS_PGRAPH_EXCEPTION_GK104(d, r) \
    PRINT_GR_Z( d, r, _FE,      "PENDING" ); \
    PRINT_GR_Z( d, r, _MEMFMT,  "PENDING" ); \
    PRINT_GR_Z( d, r, _PD,      "PENDING" ); \
    PRINT_GR_Z( d, r, _SCC,     "PENDING" ); \
    PRINT_GR_Z( d, r, _DS,      "PENDING" ); \
    PRINT_GR_Z( d, r, _SSYNC,   "PENDING" ); \
    PRINT_GR_Z( d, r, _MME,     "PENDING" ); \
    PRINT_GR_Z( d, r, _SKED,    "PENDING" ); \
    PRINT_GR_Z( d, r, _GPC,     "PENDING" ); \
    PRINT_GR_Z( d, r, _BE,      "PENDING" ); \

#define GR_REG_FIELDS_PGRAPH_EXCEPTION_GV100(d, r) \
    PRINT_GR_Z( d, r, _FE,      "PENDING" ); \
    PRINT_GR_Z( d, r, _MEMFMT,  "PENDING" ); \
    PRINT_GR_Z( d, r, _PD,      "PENDING" ); \
    PRINT_GR_Z( d, r, _SCC,     "PENDING" ); \
    PRINT_GR_Z( d, r, _DS,      "PENDING" ); \
    PRINT_GR_Z( d, r, _SSYNC,   "PENDING" ); \
    PRINT_GR_Z( d, r, _MME,     "PENDING" ); \
    PRINT_GR_Z( d, r, _SKED,    "PENDING" ); \
    PRINT_GR_Z( d, r, _GPC,     "PENDING" ); \
    PRINT_GR_Z( d, r, _BE,      "PENDING" ); \

/*
 * Registers in instanced units (BE/FBP, GPC, TPC) use helper macros
 * with a different API
 */
#define GR_REG_FIELDS_BE_BECS_BE_ACTIVITY0_GF100(  ) \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GF100( BECS_BE_ACTIVITY0, CROP );     \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GF100( BECS_BE_ACTIVITY0, ZROP );               \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GF100( BECS_BE_ACTIVITY0, RDM );                \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GF100( BECS_BE_ACTIVITY0, PMMFBP );             \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GF100( BECS_BE_ACTIVITY0, BECS );

#define GR_REG_FIELDS_BE_BECS_BE_ACTIVITY0_GV100(  ) \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GM200( BECS_BE_ACTIVITY0, CROP );               \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GM200( BECS_BE_ACTIVITY0, ZROP );               \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GM200( BECS_BE_ACTIVITY0, RDM );                \
    PRINT_BECS_BE_UNIT_ACTIVITY_Z_GM200( BECS_BE_ACTIVITY0, BECS );


#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_GPC_0_GF100( )               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_GPC_0, HWW, NULL );      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_GPC_0, CTXSW_HALT, NULL ); \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_GPC_0, STOP_SM, NULL );

#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_GPC_0_GV100( )                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_GPC_0, HWW, NULL );                   \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_GPC_0, CTXSW_HALT, NULL );            \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_GPC_0, STOP_SM, NULL );

#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_1_GF100( )                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONSTANT_0, NULL );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, LOCAL_PRIV_ERROR, NULL );         \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, RC_LANES, "BUSY" );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONTEXT_RESET, "DISABLED" );        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, FE_PRIV, NULL );                  \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_SEQUENCER, "BUSY" );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, READ, "DONE" );                     \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WRACK, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP0, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP1, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, DFD_TRIGGER, "ACTIVE" );            \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WATCHDOG_TRIGGER, "ACTIVE" );       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_BUSY, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_NACK, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_HALTED, NULL );

#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_1_GK104(  )                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONSTANT_0, NULL );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, LOCAL_PRIV_ERROR, NULL );         \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, RC_LANES, "BUSY" );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONTEXT_RESET, "DISABLED" );        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, FE_PRIV, NULL );                  \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_SEQUENCER, "BUSY" );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, READ, "DONE" );                     \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WRACK, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP0, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP1, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, DFD_TRIGGER, "ACTIVE" );            \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WATCHDOG_TRIGGER, "ACTIVE" );       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_BUSY, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_NACK, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_HALTED, NULL );   \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_FLUSH_WHEN_BUSY, NULL );

#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_1_GM107(  )                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONSTANT_0, NULL );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, LOCAL_PRIV_ERROR, NULL );         \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, RC_LANES, "BUSY" );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONTEXT_RESET, "DISABLED" );        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, FE_PRIV, NULL );                  \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_SEQUENCER, "BUSY" );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, READ, "DONE" );                     \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WRACK, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP0, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP1, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, DFD_TRIGGER, "ACTIVE" );            \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WATCHDOG_TRIGGER, "ACTIVE" );       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_BUSY, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_NACK, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_HALTED, NULL );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_FLUSH_WHEN_BUSY, NULL );      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP_PL, NULL );      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, SEQ_PRIV_ERROR, NULL );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP, NULL );         \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_UCODE_TRAP, NULL );

#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_1_GM200(  )                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONSTANT_0, NULL );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, LOCAL_PRIV_ERROR, NULL );         \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, RC_LANES, "BUSY" );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONTEXT_RESET, "DISABLED" );        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_SEQUENCER, "BUSY" );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, READ, "DONE" );                     \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WRACK, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP0, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP1, "DONE" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, DFD_TRIGGER, "ACTIVE" );            \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WATCHDOG_TRIGGER, "ACTIVE" );       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_BUSY, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_NACK, NULL );                 \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_HALTED, NULL );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_FLUSH_WHEN_BUSY, NULL );      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP_PL, NULL );      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, SEQ_PRIV_ERROR, NULL );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP, NULL );         \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_UCODE_TRAP, NULL );

#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_1_GV100(  )                        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONSTANT_0, NULL );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, LOCAL_PRIV_ERROR, NULL );              \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, RC_LANES, "BUSY" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONTEXT_RESET, "DISABLED" );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_SEQUENCER, "BUSY" );              \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, READ, "DONE" );                        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WRACK, "DONE" );                       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP0, "DONE" );                       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP1, "DONE" );                       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, DFD_TRIGGER, "ACTIVE" );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WATCHDOG_TRIGGER, "ACTIVE" );          \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_BUSY, NULL );                      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_NACK, NULL );                      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_HALTED, NULL );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_FLUSH_WHEN_BUSY, NULL );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CTXSW_CHECKSUM_MISMATCH, NULL );       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP_PL, "VIOLATION" );    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, SEQ_PRIV_ERROR, NULL );                \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP, "WRITE_DROPPED" );   \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_UCODE_TRAP, "PENDING" );

#define GR_REG_FIELDS_GPC_GPCCS_CTXSW_STATUS_1_GA102(  )                        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONSTANT_0, NULL );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, LOCAL_PRIV_ERROR, NULL );              \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, RC_LANES, "BUSY" );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CONTEXT_RESET, "DISABLED" );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_SEQUENCER, "BUSY" );              \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, READ, "DONE" );                        \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WRACK, "DONE" );                       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP0, "DONE" );                       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, COMP1, "DONE" );                       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, DFD_TRIGGER, "ACTIVE" );               \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, WATCHDOG_TRIGGER, "ACTIVE" );          \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_BUSY, NULL );                      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_NACK, NULL );                      \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_HALTED, NULL );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, ARB_FLUSH_WHEN_BUSY, NULL );           \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, CTXSW_CHECKSUM_MISMATCH, NULL );       \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP_PL, "VIOLATION" );    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_ERROR, NULL );                    \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_DECODE_TRAP, "WRITE_DROPPED" );   \
    PRINT_GPC_F_Z( GPCCS_CTXSW_STATUS_1, PRIV_UCODE_TRAP, "PENDING" );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY0_GF100(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, SETUP );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, CRSTR );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, ZLWLL );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, WIDCLIP );               \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, FRSTR );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, TC );                    \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, RASTERARB );             \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY0, PROP );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY0_GV100(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, SETUP );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, CRSTR );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, ZLWLL );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, WIDCLIP );               \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, FRSTR );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, TC );                    \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, RASTERARB );             \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, PROP );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, GNIC );             \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, GPCCS );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY0_GA102(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, SETUP );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, CRSTR );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, ZLWLL );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, WIDCLIP );               \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, FRSTR );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, TC );                    \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, RASTERARB );             \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, PROP );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY0, GPCCS );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY1_GF100(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, GPMPD );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, GPMSD );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, GPMREPORT );            \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, SWDX );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, WDXPS );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, GCC );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, GPCMMU );               \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY1, PMMGPC );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY1_GV100(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPMPD );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPMSD );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPMREPORT );            \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, SWDX );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, WDXPS );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GCC );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPCMMU );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY1_GA102(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPMPD );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPMSD );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPMREPORT );            \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, SWDX );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, WDXPS );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GCC );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GPCMMU0 );              \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY1, GNIC );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY2_GF100(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, GNIC );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, GPCCS );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY2_GK104(  )            \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, GNIC );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, GPCCS );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, PES0 );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, CBMGR0 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, WWDX0 );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY2_GM107(  )            \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, GNIC );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, GPCCS );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, PES0 );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, PES1 );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, CBMGR0 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, CBMGR1 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, WWDX0 );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( GPCCS_GPC_ACTIVITY2, WWDX1 );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY2_GV100(  )            \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, PES0 );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, PES1 );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, PES2 );                  \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, CBMGR0 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, CBMGR1 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, CBMGR2 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, WWDX0 );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, WWDX1 );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY2, WWDX2 );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY3_GF100(  )          \
    PRINT_GPCCS_GPC_TPC_ACTIVITY_Z_GF100( );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY3_GV100(  )          \
    PRINT_GPCCS_GPC_TPC_ACTIVITY_Z_GM200( );

#define GR_REG_FIELDS_GPC_GPCCS_GPC_ACTIVITY4_GA102(  )         \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY4, CROP0 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY4, CROP1 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY4, ZROP0 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY4, ZROP1 );                \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY4, RRH0 );                 \
    PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( GPCCS_GPC_ACTIVITY4, RRH1 );

#define GR_REG_FIELDS_TPC_TPCCS_TPC_ACTIVITY0_GF100(  )          \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, PE );   \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, TEX );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, MPC );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, CBMGR ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, WWDX ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, L1C );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, SM );

#define GR_REG_FIELDS_TPC_TPCCS_TPC_ACTIVITY0_GK104(  )          \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, PE );   \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, TEX );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, MPC );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, L1C );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, SM );

#define GR_REG_FIELDS_TPC_TPCCS_TPC_ACTIVITY0_GM107(  )          \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, PE );   \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, TEX );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, MPC );  \
    PRINT_TPC_UNIT_STATUS_Z_GF100( TPCCS_TPC_ACTIVITY0, SM );

#define GR_REG_FIELDS_TPC_TPCCS_TPC_ACTIVITY0_GV100(  )         \
    PRINT_TPC_UNIT_STATUS_Z_GM200( TPCCS_TPC_ACTIVITY0, PE );         \
    PRINT_TPC_UNIT_STATUS_Z_GM200( TPCCS_TPC_ACTIVITY0, MPC );        \
    PRINT_TPC_UNIT_STATUS_Z_GM200( TPCCS_TPC_ACTIVITY0, SM );

#define GR_REG_FIELDS_TPC_MPC_STATUS_GF100(  )                    \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_STATUS, TOP_STATUS );    \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_STATUS, VTG_STATUS );    \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_STATUS, PIX_STATUS );    \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_STATUS, WLU_STATUS );

#define GR_REG_FIELDS_TPC_MPC_STATUS_GV100(  )                    \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_STATUS, TOP_STATUS );    \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_STATUS, VTG_STATUS );    \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_STATUS, PIX_STATUS );    \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_STATUS, WLU_STATUS );    \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_STATUS, COMP_STATUS );

#define GR_REG_FIELDS_TPC_MPC_VTG_STATUS_GF100(  )          \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, ALPHA_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, BETA_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, ISBMGR_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, SMIO_STATUS ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, MODE, "BETA" );        \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAFA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAFB_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, ISBCP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, ISBTI_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSC_ALPHA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAF_BETA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TGBFILL_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, GSPILL_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSC_BETA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSA_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSB_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TI_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TS_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, GS_WARP_OUTSTANDING, NULL );

#define GR_REG_FIELDS_TPC_MPC_VTG_STATUS_GM107(  )          \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, ALPHA_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, BETA_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, ISBMGR_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_VTG_STATUS, SMIO_STATUS ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, MODE, "BETA" );        \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAFA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAFB_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSC_ALPHA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAF_BETA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TGBFILL_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, GSPILL_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSC_BETA_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSA_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSB_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TI_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TS_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, GS_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, CBMGR_OUTSTANDING, NULL );

#define GR_REG_FIELDS_TPC_MPC_VTG_STATUS_GV100(  )                  \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_VTG_STATUS, ALPHA_STATUS );        \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_VTG_STATUS, BETA_STATUS );         \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_VTG_STATUS, ISBMGR0_STATUS );      \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_VTG_STATUS, ISBMGR1_STATUS );      \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_VTG_STATUS, SMIO_STATUS );         \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, MODE, "BETA" );                  \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAFA_OUTSTANDING, NULL );        \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAFB_OUTSTANDING, NULL );        \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSC_ALPHA_OUTSTANDING, NULL );   \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VAF_BETA_OUTSTANDING, NULL );    \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TGBFILL_OUTSTANDING, NULL );     \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, GSPILL_OUTSTANDING, NULL );      \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSC_BETA_OUTSTANDING, NULL );    \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSA_WARP_OUTSTANDING, NULL );    \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, VSB_WARP_OUTSTANDING, NULL );    \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TI_WARP_OUTSTANDING, NULL );     \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, TS_WARP_OUTSTANDING, NULL );     \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, GS_WARP_OUTSTANDING, NULL );     \
    PRINT_TPC_F_Z( MPC_VTG_STATUS, CBMGR_OUTSTANDING, NULL );


#define GR_REG_FIELDS_TPC_MPC_PIX_STATUS_GF100(  )        \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_PIX_STATUS, TRAM_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_PIX_STATUS, INPUT_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_PIX_STATUS, LAUNCH_STATUS ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( MPC_PIX_STATUS, OUTPUT_STATUS ); \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, TRAMFILL_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, START_XY_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, PS_WARP_OUTSTANDING, NULL ); \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, DQUAD_UNLOAD_OUTSTANDING, NULL );

#define GR_REG_FIELDS_TPC_MPC_PIX_STATUS_GV100(  )        \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_PIX_STATUS, TRAM_STATUS );                 \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_PIX_STATUS, INPUT_STATUS );                \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_PIX_STATUS, LAUNCH_STATUS );               \
    PRINT_TPC_UNIT_STATUS_Z_GM200( MPC_PIX_STATUS, OUTPUT_STATUS );               \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, UNEXPECTED_TRAM_DEALLOC, NULL );         \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, TRAMFILL_OUTSTANDING, NULL );            \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, START_XY_OUTSTANDING, NULL );            \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, PS_WARP_OUTSTANDING, NULL );             \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, DQUAD_UNLOAD_OUTSTANDING, NULL );        \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, TRAM_STATE, NULL );                      \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, INPUT_STATE, NULL );                     \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, 2D_PIXOUT_CMD_DONE_OUTSTANDING, NULL );  \
    PRINT_TPC_F_Z( MPC_PIX_STATUS, RCVD_3D_AFTER_CONTEXT_SWITCH, NULL );

#define GR_REG_FIELDS_TPC_PE_STATUS_GF100(  )        \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, PE ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, PIF ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, PIN); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, VAF); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, ACACHE); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, STRI); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, TGA); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, GSPILL); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, GPULL); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS, VSC);

#define GR_REG_FIELDS_TPC_PE_STATUS_EXT_GF100(  )       \
    PRINT_TPC_UNIT_STATUS_Z_GF100( PE_STATUS_EXT, TGB ); 

#define GR_REG_FIELDS_TPC_PE_STATUS_GM200(  )       \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, PE ); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, PIF ); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, PIN); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, VAF); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, ACACHE); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, STRI); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, TGA); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, GSPILL); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, GPULL); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( PE_STATUS, VSC);

#define GR_REG_FIELDS_TPC_SM_INFO_SUBUNIT_STATUS_GF100(  )       \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, ICC ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, IMC0); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, IMC1); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, IDC); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, GIF); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, PIC); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, RFA); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, DSM); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, SLTP); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, SLTT); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, DATAPATH); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, AGUALL); 

#define GR_REG_FIELDS_TPC_SM_INFO_SUBUNIT_STATUS_GK104(  )       \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, ICC ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, RFA); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, GIF); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, HALF0); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, HALF1); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, SLTP); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, SLTT0); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, SLTT1);

#define GR_REG_FIELDS_TPC_SM_INFO_SUBUNIT_STATUS_GM107(  )       \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, ICC ); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, RFA); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, GIF); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, HALF0); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, HALF1); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, SLTP); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, MIOP0); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, MIOP1); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, MIO); \
    PRINT_TPC_UNIT_STATUS_Z_GF100( SM_INFO_SUBUNIT_STATUS, MIOS); 

#define GR_REG_FIELDS_TPC_SM_INFO_SUBUNIT_STATUS_GM200(  )       \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, ICC ); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, RFA); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, GIF); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, HALF0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, HALF1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, SLTP); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, MIOP0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, MIOP1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, MIO); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_INFO_SUBUNIT_STATUS, MIOS); 

#define GR_REG_FIELDS_TPC_SM_STATUS_GV100(  )        \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, SFE ); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, SCTL0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, SCTL1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, L1C0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, L1C1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXAM0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXAM1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXDF0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXDF1);

#define GR_REG_FIELDS_TPC_SM_STATUS_TU102(  )        \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, SFE ); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, SCTL0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, SCTL1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, L1C0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, L1C1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXAM0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXAM1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXDF0); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TEXDF1); \
    PRINT_TPC_UNIT_STATUS_Z_GM200( SM_STATUS, TTU);


//-----------------------------------------------------
// Hardcoded GR status printing macros.
//-----------------------------------------------------
/*!
 * Print a register field
 * @param b buffer containing the field name
 * @param sv symbolic value for the field
 * @param v numeric value for the field
 */
#define DPRINTF_FIELD( b, sv, v ) do {                                   \
        dprintf("                %-33s = <%s> [0x%04x]\n", b, sv ? sv : "", v ); \
} while(0)

/*!
 * Print a register
 * @param b buffer containing the register name
 * @param a register address
 * @param v register value
 */
#define DPRINTF_REG( b, a, v ) do {                             \
    dprintf("%-25s @(0x%08x) = 0x%08x\n", b, a, v );       \
} while(0)


#define PRINT_GR(d, r, f) do{ \
    val = DRF_VAL(d, r, f, grStatus);           \
    DPRINTF_FIELD( #f, NULL, val );             \
}while(0)

/*!
 * Print a message if a field is zero,
 * or a different message if the field is not zero.
 * Changed this to only print the non-zero case to make output more
 * concise.  Printing zero values just clutters the report with
 * no added value for debug.
 * @param output The message to print if the field is zero.
 * @param else_output The message to print if the field is not zero.
 */
#define PRINT_GR_CONDITIONALLY(d, r, f, output, else_output) do{    \
    val = DRF_VAL(d, r, f, grStatus);                               \
    if( FALSE == val )                                                  \
    {                                                                   \
        DPRINTF_FIELD( #f, output, val );                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        DPRINTF_FIELD( #f, else_output, val );                          \
    }                                                                   \
}while(0)

#define PRINT_GR_Z(d, r, f, output) do{                                 \
    val = DRF_VAL(d, r, f, grStatus);                                   \
    if( FALSE != val )                                                  \
    {                                                                   \
        DPRINTF_FIELD( #f, output, val );                               \
    }                                                                   \
}while(0)

/*
 * Print a TPC register field if not zero (FALSE)
 * @param nGpc GPC ID
 * @param nTpc TPC ID
 * @param r register within the TPC
 * @param f field in the register
 * @param output string to print if the field value != 0
 */

#define PRINT_TPC_F_Z( r, f, output )  do{                  \
    val = DRF_VAL( _PGRAPH, _PRI_GPC0_TPC0_ ## r, _ ## f, grStatus );   \
    if( FALSE != val )                                                  \
    {                                                                   \
        DPRINTF_FIELD( #f, output, val );                               \
    }                                                                   \
}while(0)

/*
 * Print a GPC register field if not zero (FALSE)
 * @param nGpc GPC ID
 * @param r register within the TPC
 * @param f field in the register
 * @param output string to print if the field value != 0
 */

#define PRINT_GPC_F_Z( r, f, output )  do{                   \
    val = DRF_VAL( _PGRAPH, _PRI_GPC0_ ## r, _ ## f, grStatus );        \
    if( FALSE != val )                                                  \
    {                                                                   \
        DPRINTF_FIELD( #f, output, val );                               \
    }                                                                   \
}while(0)


/*!
 * Print a message if a given state has been reached.
 */
#define PRINT_GR_BASED_ON_DEF(d, r, f, state ) do{      \
    if ( grStatus & DRF_DEF(d, r, f, state ) )          \
    {                                                                 \
        dprintf("lw: %s\n", QUOTE_ME( LW ## d ## r ## f ## state ) ); \
    }                                                                 \
}while(0)

// Just look for non-EMPTY TPC's.  Floorswept TPC's will show as empty

#define PRINT_GPCCS_GPC_TPC_ACTIVITY_Z_GF100( ) do { \
    LwU32 tpcActivity = grStatus;                                        \
    LwU32 tpcId;                                                         \
    char tpcName[GR_REG_NAME_BUFFER_LEN];                               \
    for(tpcId = 0; tpcId < numActiveTpc ; tpcId++)                            \
    {                                                                   \
        if(!FLD_TEST_DRF(_PGRAPH,_PRI_GPC0_GPCCS_GPC_ACTIVITY3,_TPC0,_EMPTY,tpcActivity)) \
        {                                                               \
            sprintf(tpcName,"TPC%d",tpcId);                             \
            grPrintGpuUnitStatus_GF100(DRF_VAL(_PGRAPH,_PRI_GPC0_GPCCS_GPC_ACTIVITY3,_TPC0,tpcActivity), tpcName); \
        }                                                               \
        tpcActivity >>= DRF_SIZE(LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY3_TPC0); \
    }                                                                   \
}while(0)

#define PRINT_GPCCS_GPC_TPC_ACTIVITY_Z_GM200( ) do { \
    LwU32 tpcActivity = grStatus;                                        \
    LwU32 tpcId;                                                         \
    char tpcName[GR_REG_NAME_BUFFER_LEN];                               \
    for(tpcId = 0; tpcId < numActiveTpc ; tpcId++)                            \
    {                                                                   \
        if(!FLD_TEST_DRF(_PGRAPH,_PRI_GPC0_GPCCS_GPC_ACTIVITY3,_TPC0,_EMPTY,tpcActivity)) \
        {                                                               \
            sprintf(tpcName,"TPC%d",tpcId);                             \
            grPrintGpuUnitStatus_GM200(DRF_VAL(_PGRAPH,_PRI_GPC0_GPCCS_GPC_ACTIVITY3,_TPC0,tpcActivity), tpcName); \
        }                                                               \
        tpcActivity >>= DRF_SIZE(LW_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY3_TPC0); \
    }                                                                   \
}while(0)

#define PRINT_GPCCS_UNIT_ACTIVITY_Z_GF100( r, f ) do {                          \
    val = DRF_VAL(_PGRAPH, _PRI_GPC0_ ## r, _  ## f, grStatus); \
    if(EMPTY != val)                                                  \
    {                                                                   \
        grPrintGpuUnitStatus_GF100(val, #f);                            \
    }                                                                   \
}while(0)

#define PRINT_GPCCS_UNIT_ACTIVITY_Z_GM200( r, f ) do {                          \
    val = DRF_VAL(_PGRAPH, _PRI_GPC0_ ## r, _  ## f, grStatus); \
    if(EMPTY != val)                                                  \
    {                                                                   \
        grPrintGpuUnitStatus_GM200(val, #f);                            \
    }                                                                   \
}while(0)

#define PRINT_TPC_UNIT_STATUS_Z_GF100( r, f ) do {                  \
    val = DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_ ## r, _ ## f, grStatus);   \
    if(EMPTY != val)                                                  \
    {                                                                   \
        grPrintGpuUnitStatus_GF100(val, #f);                                  \
    }                                                                   \
}while(0)

#define PRINT_TPC_UNIT_STATUS_Z_GM200( r, f ) do {                  \
    val = DRF_VAL(_PGRAPH, _PRI_GPC0_TPC0_ ## r, _ ## f, grStatus);   \
    if(EMPTY != val)                                                  \
    {                                                                   \
        grPrintGpuUnitStatus_GM200(val, #f);                                  \
    }                                                                   \
}while(0)

#define PRINT_BECS_BE_UNIT_ACTIVITY_Z_GF100( r, f ) do {          \
    val = DRF_VAL(_PGRAPH, _PRI_BE0_ ## r, _ ## f, grStatus);           \
    if(EMPTY != val)                                                    \
    {                                                                   \
        grPrintGpuUnitStatus_GF100(val, #f);                                    \
    }                                                                   \
}while(0)

#define PRINT_BECS_BE_UNIT_ACTIVITY_Z_GM200( r, f ) do {          \
    val = DRF_VAL(_PGRAPH, _PRI_BE0_ ## r, _ ## f, grStatus);           \
    if(EMPTY != val)                                                    \
    {                                                                   \
        grPrintGpuUnitStatus_GM200(val, #f);                                    \
    }                                                                   \
}while(0)

/*!
 * Print out the fields of a register.
 * @param FIELDS A macro listing the register's fields.
 */
#define PRINT_REG_HELPER(FIELDS, d, r, z) do{                           \
    grStatus = GPU_REG_RD32(LW ## d ## r);                                  \
    if((z!=PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES) || (grStatus))         \
    {                                                                   \
        DPRINTF_REG(QUOTE_ME(LW ## d ## r),                         \
                        LW ## d ## r, grStatus);                        \
        FIELDS(d, r);                                                   \
    }                                                                   \
}while(0)

/*!
 * Simplified API for printing a GR register.
 */
#define PRINT_REG(d,r) PRINT_REG_HELPER( GR_REG_FIELDS ## d ## r, d, r, PRIV_DUMP_REGISTER_FLAGS_DEFAULT )
#define PRINT_REG_Z(d,r) PRINT_REG_HELPER( GR_REG_FIELDS ## d ## r, d, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

/*!
 * Simplified API for printing a GR register.
 * @param chip The chip name that will be appended to the lookup list.
 *  This is used if two chips have different field definitions for the same reg.
 */
#define PRINT_REG2(d,r,chip) \
    PRINT_REG_HELPER( GR_REG_FIELDS ## d ## r ## _ ## chip, d, r, PRIV_DUMP_REGISTER_FLAGS_DEFAULT )
#define PRINT_REG2_Z(d,r,chip) \
    PRINT_REG_HELPER( GR_REG_FIELDS ## d ## r ## _ ## chip, d, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

// BE (FBP), GPC/TPC registers are instanced, necessitating a different macro
#define PRINT_BE_REG_Z( fbpId, r, chip ) \
    PRINT_BE_REG_HELPER( GR_REG_FIELDS ## _BE ## _ ## r ## _ ## chip, fbpId, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

#define PRINT_BE_REG_HELPER( FIELDS, fbpId, r, z ) do {       \
    grStatus = GPU_REG_RD32(BE_REG_ADDR( r, fbpId ));                      \
    if((z!=PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES) || (grStatus))         \
    {                                                                   \
        sprintf( buffer, "LW_PGRAPH_PRI_BE%d_%s", fbpId, #r );          \
        DPRINTF_REG( buffer, BE_REG_ADDR(r, fbpId), grStatus );         \
        FIELDS( );                                                \
    }                                                                   \
}while(0)
// BE (FBP), GPC/TPC registers are instanced, necessitating a different macro
#define PRINT_GPC_REG_Z( gpcId, r, chip ) \
    PRINT_GPC_REG_HELPER( GR_REG_FIELDS ## _GPC ## _ ## r ## _ ## chip, gpcId, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

#define PRINT_GPC_REG_HELPER( FIELDS, gpcId, r, z ) do {       \
    grStatus = GPU_REG_RD32(GPC_REG_ADDR( r, gpcId ));                      \
    if((z!=PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES) || (grStatus))         \
    {                                                                   \
        sprintf( buffer, "LW_PGRAPH_PRI_GPC%d_%s", gpcId, #r );         \
        DPRINTF_REG( buffer, GPC_REG_ADDR(r, gpcId), grStatus );        \
        FIELDS( );                                                \
    }                                                                   \
}while(0)

#define PRINT_TPC_REG_Z( gpcId, tpcId, r, chip )                         \
    PRINT_TPC_REG_HELPER( GR_REG_FIELDS ## _TPC ## _ ## r ## _ ## chip, gpcId, tpcId, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

#define PRINT_TPC_REG_HELPER( FIELDS, gpcIdid, tpcId, r, z ) do {        \
    grStatus = GPU_REG_RD32(TPC_REG_ADDR( r, gpcId, tpcId ));                 \
    if((z!=PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES) || (grStatus))         \
    {                                                                   \
        sprintf( buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_%s", gpcId, tpcId, #r ); \
        DPRINTF_REG( buffer, TPC_REG_ADDR(r, gpcId, tpcId), grStatus ); \
        FIELDS( );                                         \
    }                                                                   \
}while(0)


// Aperture-based macros

/*!
 * Dumps a PGRAPH register without decoding it.
 */
#define DUMP_APERTURE_REG(a, strReg)        do{                      \
    sprintf( buffer, "LW_PGRAPH_%s", #strReg );                     \
    DPRINTF_REG(buffer, LW_PGRAPH_##strReg, REG_RD32(&a->aperture,LW_PGRAPH_ ## strReg) ); \
}while(0)

/*!
 * Dumps a GPC register without decoding it.
 */
#define DUMP_GPC_APERTURE_REG(a, strReg)        do{                \
    sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_%s", a->unitIndex, #strReg );            \
    DPRINTF_REG(buffer, REG_GET_ADDR(&a->aperture,LW_PGPC_PRI_ ## strReg),    \
        REG_RD32(&a->aperture,LW_PGPC_PRI_ ## strReg));  \
}while(0)

#define DUMP_GPC_APERTURE_REG_Z(a, strReg)        do{           \
    LwU32 val = REG_RD32(&a->aperture,LW_PGPC_PRI_ ## strReg);       \
    if(val)                                                         \
    {                                                                   \
        sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_%s", a->unitIndex, #strReg ); \
        DPRINTF_REG(buffer, REG_GET_ADDR(&a->aperture,LW_PGPC_PRI_ ## strReg), val); \
    }                                                                   \
}while(0)

/*!
 * Dumps a TPC register without decoding it.
 */
#define DUMP_TPC_APERTURE_REG(a, strReg)         do{                              \
    sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_%s", a->pParent->unitIndex, a->unitIndex, #strReg); \
    DPRINTF_REG(buffer, REG_GET_ADDR(&a->aperture, LW_PTPC_PRI_ ## strReg),  \
                REG_RD32(&a->aperture,LW_PTPC_PRI_ ## strReg));         \
}while(0)

#define DUMP_TPC_APERTURE_REG_Z(a, strReg)         do{                \
    LwU32 val = REG_RD32(&a->aperture,LW_PTPC_PRI_ ## strReg); \
    if(val)                                                             \
    {                                                                   \
        sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_%s", a->pParent->unitIndex, a->unitIndex, #strReg); \
        DPRINTF_REG(buffer, REG_GET_ADDR(&a->aperture, LW_PTPC_PRI_ ## strReg), val); \
    }                                                                   \
}while(0)

/*!
 * Dumps a PPC register without decoding it.
 */
#define DUMP_PPC_APERTURE_REG(a, strReg)         do{         \
    sprintf(buffer, "LW_PGRAPH_PRI_GPC%d_PPC%d_%s", a->pParent->unitIndex, a->unitIndex, #strReg); \
    DPRINTF_REG(buffer, REG_GET_ADDR(a, LW_PPPC_PRI_ ## strReg),             \
                REG_RD32(&a->aperture,LW_PPPC_PRI_ ## strReg));                               \
}while(0)


/*!
 * Print out the fields of a register.
 * @param FIELDS A macro listing the register's fields.
 */
#define PRINT_APERTURE_REG_HELPER(a, FIELDS, d, r, z) do{                \
    grStatus = REG_RD32(&a->aperture,LW ## d ## r);                      \
    if((z!=PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES) || (grStatus))         \
    {                                                                   \
        DPRINTF_REG(QUOTE_ME(LW ## d ## r),                         \
                        LW ## d ## r, grStatus);                        \
        FIELDS(d, r);                                                   \
    }                                                                   \
}while(0)

/*!
 * Simplified API for printing a GR register.
 */
#define PRINT_APERTURE_REG(a,d,r) PRINT_APERTURE_REG_HELPER( a, GR_REG_FIELDS ## d ## r, d, r, PRIV_DUMP_REGISTER_FLAGS_DEFAULT )
#define PRINT_APERTURE_REG_Z(a,d,r) PRINT_APERTURE_REG_HELPER( a, GR_REG_FIELDS ## d ## r, d, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

/*!
 * Simplified API for printing a GR register.
 * @param chip The chip name that will be appended to the lookup list.
 *  This is used if two chips have different field definitions for the same reg.
 */
#define PRINT_APERTURE_REG2(a,d,r,chip)                                 \
    PRINT_APERTURE_REG_HELPER( a, GR_REG_FIELDS ## d ## r ## _ ## chip, d, r, PRIV_DUMP_REGISTER_FLAGS_DEFAULT )
#define PRINT_APERTURE_REG2_Z(a,d,r,chip)                               \
    PRINT_APERTURE_REG_HELPER( a, GR_REG_FIELDS ## d ## r ## _ ## chip, d, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )


#define PRINT_GPC_APERTURE_REG_Z( a, r, chip )              \
    PRINT_GPC_APERTURE_REG_HELPER( a, r, GR_REG_FIELDS ## _GPC ## _ ## r ## _ ## chip, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

#define PRINT_GPC_APERTURE_REG_HELPER( a, r, FIELDS, z ) do { \
    grStatus = REG_RD32( &a->aperture, LW_PGPC_PRI_ ## r );           \
    if((z!=PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES) || (grStatus))         \
    {                                                                   \
        sprintf( buffer, "LW_PGRAPH_PRI_GPC%d_%s", a->unitIndex, #r );         \
        DPRINTF_REG( buffer, REG_GET_ADDR(&a->aperture ,LW_PGPC_PRI_ ## r), grStatus ); \
        FIELDS( );                                                \
    }                                                                   \
}while(0)

#define PRINT_TPC_APERTURE_REG_Z( a, r, chip )                           \
  PRINT_TPC_APERTURE_REG_HELPER( a, GR_REG_FIELDS ## _TPC ## _ ## r ## _ ## chip, r, PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES )

#define PRINT_TPC_APERTURE_REG_HELPER( a, FIELDS, r, z ) do {   \
    grStatus = REG_RD32( &a->aperture, LW_PTPC_PRI_ ## r);    \
    if((z!=PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES) || (grStatus))         \
    {                                                                   \
      sprintf( buffer, "LW_PGRAPH_PRI_GPC%d_TPC%d_%s", a->pParent->unitIndex, a->unitIndex, #r ); \
        DPRINTF_REG( buffer, REG_GET_ADDR(&a->aperture, LW_PTPC_PRI_ ## r), grStatus ); \
        FIELDS( );                                         \
    }                                                                   \
}while(0)





#include "g_gr_hal.h"                    // (rmconfig) public interface

#endif

