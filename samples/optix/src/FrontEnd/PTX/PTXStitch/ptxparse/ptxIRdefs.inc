/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2010-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

/*
 *   **WARNING**: Do NOT include this file directly into sources. 
 *                Instead include ptxObfuscatedIRdefs.h.
 *                
 *                If a single modifier value requires to be marked with __obfuscate,
 *                then it needs to be handled as a special case in get*AsString functions.
 *
 *               '__obfuscate' mechanism does not work for modifiers whose strings
 *                are decleared using 'PREFIX'.
 *                Refer note above 'PTX_TYPES_TABLE' for details.
 *
 */

#define GET_CLASS(CLASS, ENUM, NAME) CLASS
#define GET_ENUM(CLASS, ENUM, NAME) ENUM
#define GET_NAME(CLASS, ENUM, NAME) NAME
#define GET_ALL(CLASS, ENUM, NAME) (CLASS ENUM NAME)

#define PTX_KEEPREFMOD_TABLE(PICKATTR)                                         \
    PICKATTR("KEEPREF", ptxNOKEEPREF_MOD,   ""),                               \
    PICKATTR("KEEPREF", ptxKEEPREF_MOD,     __obfuscate(".keeprefcount"))

#define PTX_NOATEXITMOD_TABLE(PICKATTR)                                        \
    PICKATTR("NOATEXIT", ptxNONOATEXIT_MOD, ""),                               \
    PICKATTR("NOATEXIT", ptxNOATEXIT_MOD,   __obfuscate(".no_atexit"))

#define PTX_COMPARE_TABLE(PICKATTR) \
    PICKATTR("COMPARE",    ptxNO_Comparison,      ""),                         \
    PICKATTR("COMPARE",    ptxEQ_Comparison,      ".eq"),                      \
    PICKATTR("COMPARE",    ptxNE_Comparison,      ".ne"),                      \
    PICKATTR("COMPARE",    ptxLT_Comparison,      ".lt"),                      \
    PICKATTR("COMPARE",    ptxLE_Comparison,      ".le"),                      \
    PICKATTR("COMPARE",    ptxGT_Comparison,      ".gt"),                      \
    PICKATTR("COMPARE",    ptxGE_Comparison,      ".ge"),                      \
    PICKATTR("COMPARE",    ptxLO_Comparison,      ".lo"),                      \
    PICKATTR("COMPARE",    ptxLS_Comparison,      ".ls"),                      \
    PICKATTR("COMPARE",    ptxHI_Comparison,      ".hi"),                      \
    PICKATTR("COMPARE",    ptxHS_Comparison,      ".hs"),                      \
    PICKATTR("COMPARE",    ptxNUM_Comparison,     ".num"),                     \
    PICKATTR("COMPARE",    ptxNAN_Comparison,     ".nan"),                     \
    PICKATTR("COMPARE",    ptxEQU_Comparison,     ".equ"),                     \
    PICKATTR("COMPARE",    ptxNEU_Comparison,     ".neu"),                     \
    PICKATTR("COMPARE",    ptxLTU_Comparison,     ".ltu"),                     \
    PICKATTR("COMPARE",    ptxLEU_Comparison,     ".leu"),                     \
    PICKATTR("COMPARE",    ptxGTU_Comparison,     ".gtu"),                     \
    PICKATTR("COMPARE",    ptxGEU_Comparison,     ".geu")

#define PTX_OPERATOR_TABLE(PICKATTR) \
    PICKATTR("POSTOP",     ptxNOP,                ""),                         \
    PICKATTR("POSTOP",     ptxLTOp,               "<"),                        \
    PICKATTR("POSTOP",     ptxLTEQOp,             "<="),                       \
    PICKATTR("POSTOP",     ptxGTOp,               ">"),                        \
    PICKATTR("POSTOP",     ptxGTEQOp,             ">="),                       \
    PICKATTR("POSTOP",     ptxEQOp,               "=="),                       \
    PICKATTR("POSTOP",     ptxNEQOp,              "!="),                       \
    PICKATTR("POSTOP",     ptxOROp,               ".or"),                      \
    PICKATTR("POSTOP",     ptxOROROp,             "||"),                       \
    PICKATTR("POSTOP",     ptxANDOp,              ".and"),                     \
    PICKATTR("POSTOP",     ptxANDANDOp,           "&&"),                       \
    PICKATTR("POSTOP",     ptxXOROp,              ".xor"),                     \
    PICKATTR("POSTOP",     ptxADDOp,              ".add"),                     \
    PICKATTR("POSTOP",     ptxSUBOp,              ".sub"),                     \
    PICKATTR("POSTOP",     ptxSHLOp,              "<<"),                       \
    PICKATTR("POSTOP",     ptxSHROp,              ">>"),                       \
    PICKATTR("POSTOP",     ptxMULOp,              "*"),                        \
    PICKATTR("POSTOP",     ptxDIVOp,              "/"),                        \
    PICKATTR("POSTOP",     ptxREMOp,              "%"),                        \
    PICKATTR("POSTOP",     ptxNOTOp,              "!"),                        \
    PICKATTR("POSTOP",     ptxILWOp,              "~"),                        \
    PICKATTR("POSTOP",     ptxMINOp,              ".min"),                     \
    PICKATTR("POSTOP",     ptxMAXOp,              ".max"),                     \
    PICKATTR("POSTOP",     ptxMAXABSOp,           ".maxabs"),                  \
    PICKATTR("POSTOP",     ptxINCOp,              ".inc"),                     \
    PICKATTR("POSTOP",     ptxDECOp,              ".dec"),                     \
    PICKATTR("POSTOP",     ptxPOPCOp,             ".popc"),                    \
    PICKATTR("POSTOP",     ptxCASOp,              ".cas"),                     \
    PICKATTR("POSTOP",     ptxEXCHOp,             ".exch"),                    \
    PICKATTR("POSTOP",     ptxSAFEADDOp,          ".safeadd")

#define PTX_NOINCMOD_TABLE(PICKATTR)                                           \
    PICKATTR("NOINC",      ptxNONOINC_MOD,        ""),                         \
    PICKATTR("NOINC",      ptxNOINC_MOD,          ".noinc")

/* NOTE: PTX type string is used in two formats : ".TYPE" and "_TYPE". But current 'PREFIX'
 *       based mechanism does not work with '__obfuscate' due to sequence of invocation
 *       of perl script and preprocessor.
 *       Therefore, All type strings are declared here in form ".TYPE" and as per requirement
 *       they are manually colwerted into "_TYPE"
 */
#define PTX_TYPES_TABLE(PICKATTR)                           \
    PICKATTR("TYPES",      ptxNOTYPE,             ""),      \
    PICKATTR("TYPES",      ptxTypeB1,            ".b1"),    \
    PICKATTR("TYPES",      ptxTypeB2,            ".b2"),    \
    PICKATTR("TYPES",      ptxTypeB4,            ".b4"),    \
    PICKATTR("TYPES",      ptxTypeB8,            ".b8"),    \
    PICKATTR("TYPES",      ptxTypeB16,           ".b16"),   \
    PICKATTR("TYPES",      ptxTypeB32,           ".b32"),   \
    PICKATTR("TYPES",      ptxTypeB64,           ".b64"),   \
    PICKATTR("TYPES",      ptxTypeB128,           __obfuscate(".b128")),  \
    PICKATTR("TYPES",      ptxTypeU2,            ".u2"),    \
    PICKATTR("TYPES",      ptxTypeU4,            ".u4"),    \
    PICKATTR("TYPES",      ptxTypeU8,            ".u8"),    \
    PICKATTR("TYPES",      ptxTypeU16,           ".u16"),   \
    PICKATTR("TYPES",      ptxTypeU32,           ".u32"),   \
    PICKATTR("TYPES",      ptxTypeU64,           ".u64"),   \
    PICKATTR("TYPES",      ptxTypeS2,            ".s2"),    \
    PICKATTR("TYPES",      ptxTypeS4,            ".s4"),    \
    PICKATTR("TYPES",      ptxTypeS8,            ".s8"),    \
    PICKATTR("TYPES",      ptxTypeS16,           ".s16"),   \
    PICKATTR("TYPES",      ptxTypeS32,           ".s32"),   \
    PICKATTR("TYPES",      ptxTypeS64,           ".s64"),   \
    PICKATTR("TYPES",      ptxTypeE4M3,          __obfuscate(".e4m3")),    \
    PICKATTR("TYPES",      ptxTypeE5M2,          __obfuscate(".e5m2")),    \
    PICKATTR("TYPES",      ptxTypeE4M3x2,        __obfuscate(".e4m3x2")),  \
    PICKATTR("TYPES",      ptxTypeE5M2x2,        __obfuscate(".e5m2x2")),  \
    PICKATTR("TYPES",      ptxTypeF16,           ".f16"),   \
    PICKATTR("TYPES",      ptxTypeBF16,          ".bf16"),  \
    PICKATTR("TYPES",      ptxTypeBF16x2,        ".bf16x2"),\
    PICKATTR("TYPES",      ptxTypeTF32,          ".tf32"),  \
    PICKATTR("TYPES",      ptxTypeF16x2,         ".f16x2"), \
    PICKATTR("TYPES",      ptxTypeF32,           ".f32"),   \
    PICKATTR("TYPES",      ptxTypeF64,           ".f64"),   \
    PICKATTR("TYPES",      ptxTypePred,          ".pred")


#define PTX_APRXMOD_TABLE(PICKATTR) \
    PICKATTR("APRX",       ptxNOAPRX_MOD,         ""),                         \
    PICKATTR("APRX",       ptxAPRX_MOD,           ".approx")

#define PTX_RELUMOD_TABLE(PICKATTR) \
    PICKATTR("RELU",        ptxNORELU_MOD,        ""),                         \
    PICKATTR("RELU",        ptxRELU_MOD,          ".relu")

#define PTX_NANMODEMOD_TABLE(PICKATTR) \
    PICKATTR("NANMODE",     ptxNONANMODE_MOD,     ""),                         \
    PICKATTR("NANMODE",     ptxNANMODE_MOD,       ".NaN")
    
#define PTX_XORSIGNMOD_TABLE(PICKATTR) \
    PICKATTR("XORSIGN",     ptxNOXORSIGN_MOD,     ""),                         \
    PICKATTR("XORSIGN",     ptxXORSIGN_MOD,       ".xorsign")

#define PTX_TRANSAMOD_TABLE(PICKATTR) \
    PICKATTR("TRANSA",      ptxNOTRANSA_MOD,      ""),                         \
    PICKATTR("TRANSA",      ptxTRANSA_MOD,        __obfuscate(".transA"))

#define PTX_NEGAMOD_TABLE(PICKATTR) \
    PICKATTR("NEGA",        ptxNONEGA_MOD,        ""),                         \
    PICKATTR("NEGA",        ptxNEGA_MOD,          __obfuscate(".negA"))

#define PTX_TRANSBMOD_TABLE(PICKATTR) \
    PICKATTR("TRANSB",      ptxNOTRANSB_MOD,      ""),                         \
    PICKATTR("TRANSB",      ptxTRANSB_MOD,        __obfuscate(".transB"))

#define PTX_NEGBMOD_TABLE(PICKATTR) \
    PICKATTR("NEGB",        ptxNONEGB_MOD,        ""),                         \
    PICKATTR("NEGB",        ptxNEGB_MOD,          __obfuscate(".negB"))

#define PTX_IGNORECMOD_TABLE(PICKATTR) \
    PICKATTR("IGNOREC",     ptxNOIGNOREC_MOD,     ""),                         \
    PICKATTR("IGNOREC",     ptxIGNOREC_MOD,       __obfuscate(".ignoreC"))

#define PTX_ADDRTYPEMOD_TABLE(PICKATTR) \
    PICKATTR("ADDRTYPE",     ptxNOADDRTYPE_MOD,     ""),                       \
    PICKATTR("ADDRTYPE",     ptxADDRTYPE_FREL_MOD,  __obfuscate(".frel"))

#define PTX_OOBMOD_TABLE(PICKATTR) \
    PICKATTR("OOB",        ptxNOOOB_MOD,          ""),                         \
    PICKATTR("OOB",        ptxOOB_MOD,            __obfuscate(".oob"))

#define PTX_FTZMOD_TABLE(PICKATTR) \
    PICKATTR("FTZ",        ptxNOFTZ_MOD,          ""),                         \
    PICKATTR("FTZ",        ptxFTZ_MOD,            ".ftz")

#define PTX_NOFTZMOD_TABLE(PICKATTR) \
    PICKATTR("NOFTZ",      ptxNO_EXPLICIT_NOFTZ_MOD,   ""),                    \
    PICKATTR("NOFTZ",      ptxEXPLICIT_NOFTZ_MOD,      ".noftz")

#define PTX_SATMOD_TABLE_RAW(PICKATTR, PREFIX)                                 \
    PICKATTR("SAT",        ptxNOSAT_MOD,          ""),                         \
    PICKATTR("SAT",        ptxSAT_MOD,            PREFIX"sat")

#define PTX_SATMOD_TABLE(PICKATTR) PTX_SATMOD_TABLE_RAW(PICKATTR, ".")

#define PTX_SATFMOD_TABLE_RAW(PICKATTR, PREFIX)                                \
    PICKATTR("SATF",        ptxNOSATF_MOD,          ""),                       \
    PICKATTR("SATF",        ptxSATFINITE_MOD,       PREFIX"satfinite")

#define PTX_SATFMOD_TABLE(PICKATTR) PTX_SATFMOD_TABLE_RAW(PICKATTR, ".")

#define PTX_CCMOD_TABLE(PICKATTR) \
    PICKATTR("CC",         ptxNOCC_MOD,           ""),                         \
    PICKATTR("CC",         ptxCC_MOD,             ".cc")

#define PTX_SHAMTMOD_TABLE(PICKATTR) \
    PICKATTR("SHAMT",      ptxNOSHAMT_MOD,        ""),                         \
    PICKATTR("SHAMT",      ptxSHAMT_MOD,          ".shiftamt")

#define PTX_SCOPEMOD_TABLE(PICKATTR) \
    PICKATTR("SCOPE",        ptxNOSCOPE_MOD,        ""),                       \
    PICKATTR("SCOPE",        ptxCTA_MOD,            ".cta"),                   \
    PICKATTR("SCOPE",        ptxGL_MOD,             ".gl"),                    \
    PICKATTR("SCOPE",        ptxGPU_MOD,            ".gpu"),                   \
    PICKATTR("SCOPE",        ptxSYS_MOD,            ".sys")

/*********************************************************/
/*           Need to check TADDR string value            */
/*********************************************************/
#define PTX_TADDRMOD_TABLE(PICKATTR) \
    PICKATTR("TADDR",      ptxNOTADDR_MOD,        ""),                         \
    PICKATTR("TADDR",      ptxTADDR_MOD,          "")

#define PTX_ORDERMOD_TABLE(PICKATTR) \
    PICKATTR("ORDER",   ptxNOORDER_MOD,        ""),                      \
    PICKATTR("ORDER",   ptxWEAK_MOD,           ".weak"),                 \
    PICKATTR("ORDER",   ptxRELAXED_MOD,        ".relaxed"),              \
    PICKATTR("ORDER",   ptxRELEASE_MOD,        ".release"),              \
    PICKATTR("ORDER",   ptxACQUIRE_MOD,        ".acquire"),              \
    PICKATTR("ORDER",   ptxACQREL_MOD,         ".acq_rel"),              \
    PICKATTR("ORDER",   ptxSC_MOD,             ".sc"),                   \
    PICKATTR("ORDER",   ptxMMIO_MOD,           __obfuscate(".mmio")),    \
    PICKATTR("ORDER",   ptxVOLATILE_MOD,       ".volatile")

#define PTX_NCMOD_TABLE(PICKATTR) \
    PICKATTR("NC",        ptxNONC_MOD,           ""),                          \
    PICKATTR("NC",        ptxNC_MOD,             ".nc")

#define PTX_ROUNDMOD_TABLE(PICKATTR) \
    PICKATTR("ROUND",      ptxNOROUND_MOD,        ""),                         \
    PICKATTR("ROUND",      ptxRN_MOD,             ".rn"),                      \
    PICKATTR("ROUND",      ptxRNA_MOD,            ".rna"),                     \
    PICKATTR("ROUND",      ptxRM_MOD,             ".rm"),                      \
    PICKATTR("ROUND",      ptxRP_MOD,             ".rp"),                      \
    PICKATTR("ROUND",      ptxRZ_MOD,             ".rz"),                      \
    PICKATTR("ROUND",      ptxRNI_MOD,            ".rni"),                     \
    PICKATTR("ROUND",      ptxRMI_MOD,            ".rmi"),                     \
    PICKATTR("ROUND",      ptxRPI_MOD,            ".rpi"),                     \
    PICKATTR("ROUND",      ptxRZI_MOD,            ".rzi")

#define PTX_TESTPMOD_TABLE(PICKATTR) \
    PICKATTR("TESTP",      ptxNOTESTP_MOD,        ""),                         \
    PICKATTR("TESTP",      ptxFINITE_MOD,         ".finite"),                  \
    PICKATTR("TESTP",      ptxINFINITE_MOD,       ".infinite"),                \
    PICKATTR("TESTP",      ptxNUMBER_MOD,         ".number"),                  \
    PICKATTR("TESTP",      ptxNOTANUMBER_MOD,     ".notanumber"),              \
    PICKATTR("TESTP",      ptxNORMAL_MOD,         ".normal"),                  \
    PICKATTR("TESTP",      ptxSUBNORMAL_MOD,      ".subnormal")

#define PTX_CACHEOPMOD_TABLE(PICKATTR) \
    PICKATTR("CACHEOP",    ptxNOCACHEOP_MOD,      ""),                         \
    PICKATTR("CACHEOP",    ptxCA_MOD,             ".ca"),                      \
    PICKATTR("CACHEOP",    ptxCG_MOD,             ".cg"),                      \
    PICKATTR("CACHEOP",    ptxCS_MOD,             ".cs"),                      \
    PICKATTR("CACHEOP",    ptxLU_MOD,             ".lu"),                      \
    PICKATTR("CACHEOP",    ptxCV_MOD,             ".cv"),                      \
    PICKATTR("CACHEOP",    ptxWB_MOD,             ".wb"),                      \
    PICKATTR("CACHEOP",    ptxWT_MOD,             ".wt"),                      \
    PICKATTR("CACHEOP",    ptxILW_MOD,            __obfuscate(".ilw")),        \
    PICKATTR("CACHEOP",    ptxILWALL_MOD,         __obfuscate(".ilwall"))

#define PTX_LEVELMOD_TABLE(PICKATTR) \
    PICKATTR("LEVEL",      ptxNOLEVEL_MOD,        ""),                         \
    PICKATTR("LEVEL",      ptxL1_MOD,             ".L1"),                      \
    PICKATTR("LEVEL",      ptxL2_MOD,             ".L2")

#define PTX_EVICTPRIORITYMOD_TABLE(PICKATTR) \
    PICKATTR("EVICTPRIORITY",  ptxNOEVICTPRIORITY_MOD, ""),                    \
    PICKATTR("EVICTPRIORITY",  ptxEVICTFIRST_MOD,      ".evict_first"),        \
    PICKATTR("EVICTPRIORITY",  ptxEVICTNORMAL_MOD,     ".evict_normal"),       \
    PICKATTR("EVICTPRIORITY",  ptxEVICTLAST_MOD,       ".evict_last"),         \
    PICKATTR("EVICTPRIORITY",  ptxEVICTUNCHANGED_MOD,  ".evict_unchanged"),    \
    PICKATTR("EVICTPRIORITY",  ptxEVICTNOALLOC_MOD,    ".no_allocate")

#define PTX_EXCLUSIVEMOD_TABLE(PICKATTR) \
    PICKATTR("EXCLUSIVE",  ptxNOEXCLUSIVE_MOD,     ""),                        \
    PICKATTR("EXCLUSIVE",  ptxEXCLUSIVE_MOD,       __obfuscate(".exclusive"))

#define PTX_FLOWMOD_TABLE(PICKATTR) \
    PICKATTR("FLOW",       ptxNOFLOW_MOD,         ""),                         \
    PICKATTR("FLOW",       ptxUNI_MOD,            ".uni")

#define PTX_BRANCHMOD_TABLE(PICKATTR) \
    PICKATTR("BRANCH",      ptxNOBRANCH_MOD,      ""),                         \
    PICKATTR("BRANCH",      ptxUNANI_MOD,         __obfuscate(".unanimous")),  \
    PICKATTR("BRANCH",      ptxCOLW_MOD,          __obfuscate(".colw")),       \
    PICKATTR("BRANCH",      ptxDIV_MOD,           __obfuscate(".div"))         \

#define PTX_VECTORMOD_TABLE(PICKATTR) \
    PICKATTR("VECTOR",     ptxNOVECTOR_MOD,       ""),                         \
    PICKATTR("VECTOR",     ptxV2_MOD,             ".v2"),                      \
    PICKATTR("VECTOR",     ptxV4_MOD,             ".v4"),                      \
    PICKATTR("VECTOR",     ptxV8_MOD,             ".v8")

    /* ALWAYS Ensure TEXTUREMOD and DIMMOD has consistent dimensionality
     * sequence. DIMMOD gets specialized to TEXMOD or TENSORDIM */

#define PTX_TEXTUREMOD_TABLE(PICKATTR) \
    PICKATTR("TEXTURE",    ptxNOTEXTURE_MOD,      ""),                         \
    PICKATTR("TEXTURE",    ptx1D_MOD,             ".1d"),                      \
    PICKATTR("TEXTURE",    ptx2D_MOD,             ".2d"),                      \
    PICKATTR("TEXTURE",    ptx3D_MOD,             ".3d"),                      \
    PICKATTR("TEXTURE",    ptx1D_BUFFER_MOD,      ".1d_buffer"),               \
    PICKATTR("TEXTURE",    ptxA1D_MOD,            ".a1d"),                     \
    PICKATTR("TEXTURE",    ptxA2D_MOD,            ".a2d"),                     \
    PICKATTR("TEXTURE",    ptxLWBE_MOD,           ".lwbe"),                    \
    PICKATTR("TEXTURE",    ptxALWBE_MOD,          ".alwbe"),                   \
    PICKATTR("TEXTURE",    ptx2DMS_MOD,           ".2dms"),                    \
    PICKATTR("TEXTURE",    ptxA2DMS_MOD,          ".a2dms")

#define PTX_DIMMOD_TABLE(PICKATTR) \
    PICKATTR("DIM",        ptxNODIM_MOD,          ""),                         \
    PICKATTR("DIM",        ptxDIM1D_MOD,          ".1d"),                      \
    PICKATTR("DIM",        ptxDIM2D_MOD,          ".2d"),                      \
    PICKATTR("DIM",        ptxDIM3D_MOD,          ".3d"),                      \
    PICKATTR("DIM",        ptxDIM4D_MOD,          ".4d"),                      \
    PICKATTR("DIM",        ptxDIM5D_MOD,          ".5d")

#define PTX_IM2COLMOD_TABLE(PICKATTR) \
    PICKATTR("IM2COL",     ptxNOIM2COL_MOD,       ""),                         \
    PICKATTR("IM2COL",     ptxIM2COL_MOD,         __obfuscate(".im2col"))

#define PTX_PACKEDOFFMOD_TABLE(PICKATTR) \
    PICKATTR("PACKEDOFF",  ptxNOPACKEDOFF_MOD,    ""),                         \
    PICKATTR("PACKEDOFF",  ptxPACKEDOFF_MOD,      __obfuscate(".packed_offsets"))

#define PTX_MULTICASTMOD_TABLE(PICKATTR) \
    PICKATTR("MULTICAST",  ptxNOMULTICAST_MOD,    ""),                         \
    PICKATTR("MULTICAST",  ptxMULTICAST_MOD,      __obfuscate(".multicast"))

#define PTX_MBARRIERMOD_TABLE(PICKATTR) \
    PICKATTR("MBARRIER",   ptxNOMBARRIER_MOD,     ""),                         \
    PICKATTR("MBARRIER",   ptxMBARRIER_MOD,       ".mbarrier")

#define PTX_FOOTPRINTMOD_TABLE(PICKATTR) \
    PICKATTR("FOOTPRINT",  ptxNOFOOTPRINT_MOD,    ""),                         \
    PICKATTR("FOOTPRINT",  ptxFOOTPRINT_MOD,      __obfuscate(".footprint"))

#define PTX_COARSEMOD_TABLE(PICKATTR) \
    PICKATTR("COARSE",     ptxNOCOARSE_MOD,       ""),                         \
    PICKATTR("COARSE",     ptxCOARSE_MOD,         __obfuscate(".coarse"))

#define PTX_COMPONENTMOD_TABLE(PICKATTR) \
    PICKATTR("COMPONENT",  ptxNOCOMP_MOD,         ""),                         \
    PICKATTR("COMPONENT",  ptxCOMP_R_MOD,         ".r"),                       \
    PICKATTR("COMPONENT",  ptxCOMP_G_MOD,         ".g"),                       \
    PICKATTR("COMPONENT",  ptxCOMP_B_MOD,         ".b"),                       \
    PICKATTR("COMPONENT",  ptxCOMP_A_MOD,         ".a")

#define PTX_VIDEOSELECTOR_TABLE(PICKATTR) \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_NONE,     ""),                       \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_H0,       ".h0"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_H1,       ".h1"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_H2,       ".h2"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_H3,       ".h3"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B0,       ".b0"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B1,       ".b1"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B2,       ".b2"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B3,       ".b3"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B4,       ".b4"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B5,       ".b5"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B6,       ".b6"),                    \
    PICKATTR("VIDEOSELECTOR",   ptxCOMP_B7,       ".b7")

#define PTX_BYTESELECTOR_TABLE(PICKATTR) \
    PICKATTR("BYTESELECTOR",   ptxBYTE_NONE,     ""),                       \
    PICKATTR("BYTESELECTOR",   ptxBYTE_B0,       ".b0"),                    \
    PICKATTR("BYTESELECTOR",   ptxBYTE_B1,       ".b1"),                    \
    PICKATTR("BYTESELECTOR",   ptxBYTE_B2,       ".b2"),                    \
    PICKATTR("BYTESELECTOR",   ptxBYTE_B3,       ".b3"),                    \
    PICKATTR("BYTESELECTOR",   ptxBYTE_B01,      ".b01"),                   \
    PICKATTR("BYTESELECTOR",   ptxBYTE_B12,      ".b12"),                   \
    PICKATTR("BYTESELECTOR",   ptxBYTE_B23,      ".b23"),                   \
    PICKATTR("BYTESELECTOR",   ptxBYTE_H0,       ".h0"),                    \
    PICKATTR("BYTESELECTOR",   ptxBYTE_H1,       ".h1")

#define PTX_QUERYMOD_TABLE(PICKATTR) \
    PICKATTR("QUERY",      ptxNOQUERY_MOD,        ""),                         \
    PICKATTR("QUERY",      ptxQUERY_WIDTH_MOD,    ".width"),                   \
    PICKATTR("QUERY",      ptxQUERY_HEIGHT_MOD,   ".height"),                  \
    PICKATTR("QUERY",      ptxQUERY_DEPTH_MOD,    ".depth"),                   \
    PICKATTR("QUERY",      ptxQUERY_CHTYPE_MOD,   ".channel_data_type"),       \
    PICKATTR("QUERY",      ptxQUERY_CHORDER_MOD,  ".channel_order"),           \
    PICKATTR("QUERY",      ptxQUERY_NORM_MOD,     ".normalized_coords"),       \
    PICKATTR("QUERY",      ptxQUERY_FILTER_MOD,   ".filter_mode"),             \
    PICKATTR("QUERY",      ptxQUERY_ADDR0_MOD,    ".addr_mode_0"),             \
    PICKATTR("QUERY",      ptxQUERY_ADDR1_MOD,    ".addr_mode_1"),             \
    PICKATTR("QUERY",      ptxQUERY_ADDR2_MOD,    ".addr_mode_2"),             \
    PICKATTR("QUERY",      ptxQUERY_UNNORM_MOD,   ".force_unnormalized_coords"), \
    PICKATTR("QUERY",      ptxQUERY_ARRSIZE_MOD,  ".array_size"),              \
    PICKATTR("QUERY",      ptxQUERY_MIPLEVEL_MOD, ".num_mipmap_levels"),       \
    PICKATTR("QUERY",      ptxQUERY_SAMPLES_MOD,  ".num_samples"),             \
    PICKATTR("QUERY",      ptxQUERY_LAYOUT_MOD,   ".memory_layout")

#define PTX_VOTEMOD_TABLE_RAW(PICKATTR, PREFIX)                                \
    PICKATTR("VOTE",       ptxNOVOTE_MOD,         ""),                         \
    PICKATTR("VOTE",       ptxVALL_MOD,           PREFIX"all"),                \
    PICKATTR("VOTE",       ptxVANY_MOD,           PREFIX"any"),                \
    PICKATTR("VOTE",       ptxBALLOT_MOD,         PREFIX"ballot"),             \
    PICKATTR("VOTE",       ptxVUNI_MOD,           PREFIX"uni")

#define PTX_VOTEMOD_TABLE(PICKATTR)  PTX_VOTEMOD_TABLE_RAW(PICKATTR, ".")

#define PTX_CLAMPMOD_TABLE(PICKATTR) \
    PICKATTR("CLAMP",      ptxNOCLAMP_MOD,        ""),                         \
    PICKATTR("CLAMP",      ptxCLAMP_MOD,          ".clamp"),                   \
    PICKATTR("CLAMP",      ptxWRAP_MOD,           ".wrap"),                    \
    PICKATTR("CLAMP",      ptxTRAP_MOD,           ".trap"),                    \
    PICKATTR("CLAMP",      ptxZERO_MOD,           ".zero")

#define PTX_SHRMOD_TABLE(PICKATTR) \
    PICKATTR("SHR",        ptxNOSHR_MOD,          ""),                         \
    PICKATTR("SHR",        ptxSHR7_MOD,           ".shr7"),                    \
    PICKATTR("SHR",        ptxSHR15_MOD,          ".shr15")

#define PTX_VMADMOD_TABLE(PICKATTR) \
    PICKATTR("VMAD",       ptxNOVMAD_MOD,         ""),                         \
    PICKATTR("VMAD",       ptxVMAD_NEG_AB_MOD,    ".negAB"),                   \
    PICKATTR("VMAD",       ptxVMAD_NEG_C_MOD,     ".negB"),                    \
    PICKATTR("VMAD",       ptxVMAD_PO_MOD,        ".po")

#define PTX_PRMTMOD_TABLE(PICKATTR) \
    PICKATTR("PRMT",       ptxNOPRMT_MOD,         ""),                         \
    PICKATTR("PRMT",       ptxPRMT_F4E_MOD,       ".f4e"),                     \
    PICKATTR("PRMT",       ptxPRMT_B4E_MOD,       ".b4e"),                     \
    PICKATTR("PRMT",       ptxPRMT_RC8_MOD,       ".rc8"),                     \
    PICKATTR("PRMT",       ptxPRMT_ECL_MOD,       ".ecl"),                     \
    PICKATTR("PRMT",       ptxPRMT_ECR_MOD,       ".erc"),                     \
    PICKATTR("PRMT",       ptxPRMT_RC16_MOD,      ".rc16")

#define PTX_SHFLMOD_TABLE_RAW(PICKATTR, PREFIX)                                \
    PICKATTR("SHFL",       ptxNOSHFL_MOD,         ""),                         \
    PICKATTR("SHFL",       ptxSHFL_UP_MOD,        PREFIX"up"),                 \
    PICKATTR("SHFL",       ptxSHFL_DOWN_MOD,      PREFIX"down"),               \
    PICKATTR("SHFL",       ptxSHFL_BFLY_MOD,      PREFIX"bfly"),               \
    PICKATTR("SHFL",       ptxSHFL_IDX_MOD,       PREFIX"idx")

#define PTX_SHFLMOD_TABLE(PICKATTR) PTX_SHFLMOD_TABLE_RAW(PICKATTR, ".")

#define PTX_ENDISMOD_TABLE(PICKATTR) \
    PICKATTR("ENDIS",      ptxNOENDIS_MOD,        ""),                         \
    PICKATTR("ENDIS",      ptxENDIS_EN_MOD,       __obfuscate(".en")),         \
    PICKATTR("ENDIS",      ptxENDIS_DIS_MOD,      __obfuscate(".dis"))

#define PTX_UNIFORMMOD_TABLE(PICKATTR) \
    PICKATTR("UNIFORM",    ptxNOUNIFORM_MOD,      ""),                         \
    PICKATTR("UNIFORM",    ptxUNIFORM_MOD,        ".uni")

#define PTX_SYNCMOD_TABLE(PICKATTR) \
    PICKATTR("SYNC",       ptxNOSYNC_MOD,         ""),                         \
    PICKATTR("SYNC",       ptxSYNC_MOD,           ".sync")

#define PTX_NOCOMPLETEMOD_TABLE(PICKATTR) \
    PICKATTR("NOCOMPLETE", ptxNONOCOMPLETE_MOD,   ""),                         \
    PICKATTR("NOCOMPLETE", ptxNOCOMPLETE_MOD,     ".noComplete")

#define PTX_NOSLEEPMOD_TABLE(PICKATTR) \
    PICKATTR("NOSLEEP",    ptxNONOSLEEP_MOD,      ""),                         \
    PICKATTR("NOSLEEP",    ptxNOSLEEP_MOD,        __obfuscate(".nosleep"))

#define PTX_BARMOD_TABLE(PICKATTR) \
    PICKATTR("BAR",        ptxNOBAR_MOD,          ""),                         \
    PICKATTR("BAR",        ptxBALL_MOD,           ".all"),                     \
    PICKATTR("BAR",        ptxBANY_MOD,           ".any")

#define PTX_THREADSMOD_TABLE(PICKATTR) \
    PICKATTR("THREADS",    ptxNOTHREADS_MOD,      ""),                         \
    PICKATTR("THREADS",    ptxTALL_MOD,           ".all"),                     \
    PICKATTR("THREADS",    ptxTANY_MOD,           ".any")

#define PTX_THREADGROUPMOD_TABLE(PICKATTR) \
    PICKATTR("THREADGROUP",   ptxNOTG_MOD,        ""),                         \
    PICKATTR("THREADGROUP",   ptxTGTHREAD_MOD,    __obfuscate(".thread")),     \
    PICKATTR("THREADGROUP",   ptxTGPAIR_MOD,      __obfuscate(".pair")),       \
    PICKATTR("THREADGROUP",   ptxTGQUAD_MOD,      __obfuscate(".quad"))

#define PTX_ALIGNMOD_TABLE(PICKATTR) \
    PICKATTR("ALIGN",      ptxNOALIGNED_MOD,      ""),                         \
    PICKATTR("ALIGN",      ptxALIGNED_MOD,        ".aligned")

#define PTX_RANDMOD_TABLE(PICKATTR) \
    PICKATTR("RAND",       ptxNORAND_MOD,         ""),                         \
    PICKATTR("RAND",       ptxRAND_MOD,           __obfuscate(".rand"))

#define PTX_TTUSLOTMOD_TABLE(PICKATTR) \
    PICKATTR("TTUSLOT",    ptxNOTTUSLOT_MOD,      ""),                         \
    PICKATTR("TTUSLOT",    ptxTTUSLOT_DUAL_MOD,   __obfuscate(".dual"))

#define PTX_SHAREDSCOPEMOD_TABLE(PICKATTR) \
    PICKATTR("SHAREDSCOPE",      ptxNOSHAREDSCOPE_MOD,        ""),                           \
    PICKATTR("SHAREDSCOPE",      ptxSHAREDSCOPE_SELF_MOD,            __obfuscate(".self")),  \
    PICKATTR("SHAREDSCOPE",      ptxSHAREDSCOPE_CLUSTER_SHARED_MOD,  __obfuscate(".clustershared"))

#define PTX_TTUMOD_TABLE(PICKATTR) \
    PICKATTR("TTU",        ptxNOTTU_MOD,          ""),                         \
    PICKATTR("TTU",        ptxTTU_CLOSE_MOD,      __obfuscate(".close"))

#define PTX_TRANSMOD_TABLE(PICKATTR) \
    PICKATTR("TRANS",      ptxNOTRANS_MOD,        ""),                         \
    PICKATTR("TRANS",      ptxTRANS_MOD,          ".trans")

#define PTX_EXPANDMOD_TABLE(PICKATTR) \
    PICKATTR("EXPAND",     ptxNOEXPAND_MOD,       ""),                         \
    PICKATTR("EXPAND",     ptxEXPAND_MOD,         __obfuscate(".expand"))

#define PTX_SEQMOD_TABLE(PICKATTR) \
    PICKATTR("SEQ",        ptxNOSEQ_MOD,          ""),                         \
    PICKATTR("SEQ",        ptxSEQ_MOD,            __obfuscate(".seq"))

#define PTX_NUMMOD_TABLE(PICKATTR) \
    PICKATTR("NUM",        ptxNONUM_MOD,          ""),                         \
    PICKATTR("NUM",        ptxNUM_x1_MOD,         ".x1"),                      \
    PICKATTR("NUM",        ptxNUM_x2_MOD,         ".x2"),                      \
    PICKATTR("NUM",        ptxNUM_x4_MOD,         ".x4")

#define PTX_GROUPMOD_TABLE(PICKATTR) \
    PICKATTR("GROUP",        ptxNOGROUP_MOD,      ""),                         \
    PICKATTR("GROUP",        ptxGROUP_g1_MOD,     __obfuscate(".1g")),         \
    PICKATTR("GROUP",        ptxGROUP_g2_MOD,     __obfuscate(".2g")),         \
    PICKATTR("GROUP",        ptxGROUP_g4_MOD,     __obfuscate(".4g"))

#define PTX_ABSMOD_TABLE(PICKATTR) \
    PICKATTR("ABS",        ptxNOABS_MOD,          ""),                         \
    PICKATTR("ABS",        ptxABS_MOD,            ".abs")

#define PTX_CACHEHINTMOD_TABLE(PICKATTR) \
    PICKATTR("CACHEHINT",        ptxNOCACHEHINT_MOD,          ""),             \
    PICKATTR("CACHEHINT",        ptxCACHEHINT_MOD,            ".L2::cache_hint")

#define PTX_LEVELEVICTPRIORITYMOD_TABLE(PICKATTR) \
    PICKATTR("LEVELEVICTPRIORITY",  ptxNOLEVELEVICTPRIORITY_MOD, ""),                        \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL2EVICTFIRST_MOD,         ".L2::evict_first"),        \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL2EVICTNORMAL_MOD,        ".L2::evict_normal"),       \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL2EVICTLAST_MOD,          ".L2::evict_last"),         \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL2EVICTUNCHANGED_MOD,     ".L2::evict_unchanged"),    \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL2EVICTNOALLOC_MOD,       ".L2::no_allocate"),        \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL1EVICTFIRST_MOD,         ".L1::evict_first"),        \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL1EVICTNORMAL_MOD,        ".L1::evict_normal"),       \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL1EVICTLAST_MOD,          ".L1::evict_last"),         \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL1EVICTUNCHANGED_MOD,     ".L1::evict_unchanged"),    \
    PICKATTR("LEVELEVICTPRIORITY",  ptxL1EVICTNOALLOC_MOD,       ".L1::no_allocate")

#define PTX_PROXYKINDMOD_TABLE(PICKATTR) \
    PICKATTR("PROXYKIND",        ptxNOPROXYKIND_MOD,                ""),                     \
    PICKATTR("PROXYKIND",        ptxPROXYKIND_ALIAS_MOD,            ".alias")


// note: any addition of a new type in this table might require updating
// the number of bits for modifiers:
// ATYPE, BTYPE, COMPRESSEDFORMAT, EXPANDEDFORMAT in ptxIR.h.
#define PTX_TYPEMOD_TABLE(PICKATTR) \
    PICKATTR("TYPE",        ptxNOTYPE_MOD,         ""),                        \
    PICKATTR("TYPE",        ptxTYPE_b1_MOD,        ".b1"),                     \
    PICKATTR("TYPE",        ptxTYPE_b2_MOD,        ".b2"),                     \
    PICKATTR("TYPE",        ptxTYPE_b4_MOD,        __obfuscate(".b4")),        \
    PICKATTR("TYPE",        ptxTYPE_b8_MOD,        ".b8"),                     \
    PICKATTR("TYPE",        ptxTYPE_b16_MOD,       ".b16"),                    \
    PICKATTR("TYPE",        ptxTYPE_b32_MOD,       ".b32"),                    \
    PICKATTR("TYPE",        ptxTYPE_b64_MOD,       ".b64"),                    \
    PICKATTR("TYPE",        ptxTYPE_b128_MOD,      __obfuscate(".b128")),      \
    PICKATTR("TYPE",        ptxTYPE_u2_MOD,        ".u2"),                     \
    PICKATTR("TYPE",        ptxTYPE_u4_MOD,        ".u4"),                     \
    PICKATTR("TYPE",        ptxTYPE_u8_MOD,        ".u8"),                     \
    PICKATTR("TYPE",        ptxTYPE_u16_MOD,       ".u16"),                    \
    PICKATTR("TYPE",        ptxTYPE_u32_MOD,       ".u32"),                    \
    PICKATTR("TYPE",        ptxTYPE_u64_MOD,       ".u64"),                    \
    PICKATTR("TYPE",        ptxTYPE_s2_MOD,        ".s2"),                     \
    PICKATTR("TYPE",        ptxTYPE_s4_MOD,        ".s4"),                     \
    PICKATTR("TYPE",        ptxTYPE_s8_MOD,        ".s8"),                     \
    PICKATTR("TYPE",        ptxTYPE_s16_MOD,       ".s16"),                    \
    PICKATTR("TYPE",        ptxTYPE_s32_MOD,       ".s32"),                    \
    PICKATTR("TYPE",        ptxTYPE_s64_MOD,       ".s64"),                    \
    PICKATTR("TYPE",        ptxTYPE_e4m3_MOD,      __obfuscate(".e4m3")),      \
    PICKATTR("TYPE",        ptxTYPE_e5m2_MOD,      __obfuscate(".e5m2")),      \
    PICKATTR("TYPE",        ptxTYPE_f16_MOD,       ".f16"),                    \
    PICKATTR("TYPE",        ptxTYPE_f32_MOD,       ".f32"),                    \
    PICKATTR("TYPE",        ptxTYPE_f16x2_MOD,     ".f16x2"),                  \
    PICKATTR("TYPE",        ptxTYPE_f64_MOD,       ".f64"),                    \
    PICKATTR("TYPE",        ptxTYPE_TF32_MOD,      ".tf32"),                   \
    PICKATTR("TYPE",        ptxTYPE_BF16_MOD,      ".bf16"),                   \
    PICKATTR("TYPE",        ptxTYPE_BF16x2_MOD,    ".bf16x2")


#define PTX_SHAPEMOD_TABLE_RAW(PICKATTR)                                              \
    PICKATTR("SHAPE",      ptxNOSHAPE_MOD,           ""),                             \
    PICKATTR("SHAPE",      ptxSHAPE_080804_MOD,      __obfuscate(".m8n8k4")),         \
    PICKATTR("SHAPE",      ptxSHAPE_080816_MOD,      __obfuscate(".m8n8k16")),        \
    PICKATTR("SHAPE",      ptxSHAPE_080832_MOD,      __obfuscate(".m8n8k32")),        \
    PICKATTR("SHAPE",      ptxSHAPE_161608_MOD,      __obfuscate(".m16n16k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_161616_MOD,      __obfuscate(".m16n16k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_320816_MOD,      __obfuscate(".m32n8k16")),       \
    PICKATTR("SHAPE",      ptxSHAPE_083216_MOD,      __obfuscate(".m8n32k16")),       \
    PICKATTR("SHAPE",      ptxSHAPE_160808_MOD,      __obfuscate(".m16n8k8")),        \
    PICKATTR("SHAPE",      ptxSHAPE_160816_MOD,      __obfuscate(".m16n8k16")),       \
    PICKATTR("SHAPE",      ptxSHAPE_160804_MOD,      __obfuscate(".m16n8k4")),        \
    PICKATTR("SHAPE",      ptxSHAPE_0808128_MOD,     __obfuscate(".m8n8k128")),       \
    PICKATTR("SHAPE",      ptxSHAPE_080800_MOD,      __obfuscate(".m8n8")),           \
    PICKATTR("SHAPE",      ptxSHAPE_081600_MOD,      __obfuscate(".m8n16")),          \
    PICKATTR("SHAPE",      ptxSHAPE_083200_MOD,      __obfuscate(".m8n32")),          \
    PICKATTR("SHAPE",      ptxSHAPE_086400_MOD,      __obfuscate(".m8n64")),          \
    PICKATTR("SHAPE",      ptxSHAPE_160832_MOD,      __obfuscate(".m16n8k32")),       \
    PICKATTR("SHAPE",      ptxSHAPE_080864_MOD,      __obfuscate(".m8n8k64")),        \
    PICKATTR("SHAPE",      ptxSHAPE_160864_MOD,      __obfuscate(".m16n8k64")),       \
    PICKATTR("SHAPE",      ptxSHAPE_1608128_MOD,     __obfuscate(".m16n8k128")),      \
    PICKATTR("SHAPE",      ptxSHAPE_1608256_MOD,     __obfuscate(".m16n8k256")),      \
    PICKATTR("SHAPE",      ptxSHAPE_640832_MOD,      __obfuscate(".m64n8k32")),       \
    PICKATTR("SHAPE",      ptxSHAPE_641632_MOD,      __obfuscate(".m64n16k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_642432_MOD,      __obfuscate(".m64n24k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_643232_MOD,      __obfuscate(".m64n32k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_644032_MOD,      __obfuscate(".m64n40k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_644832_MOD,      __obfuscate(".m64n48k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_645632_MOD,      __obfuscate(".m64n56k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_646432_MOD,      __obfuscate(".m64n64k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_647232_MOD,      __obfuscate(".m64n72k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_648032_MOD,      __obfuscate(".m64n80k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_648832_MOD,      __obfuscate(".m64n88k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_649632_MOD,      __obfuscate(".m64n96k32")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6410432_MOD,     __obfuscate(".m64n104k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6411232_MOD,     __obfuscate(".m64n112k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6412032_MOD,     __obfuscate(".m64n120k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6412832_MOD,     __obfuscate(".m64n128k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6413632_MOD,     __obfuscate(".m64n136k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6414432_MOD,     __obfuscate(".m64n144k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6415232_MOD,     __obfuscate(".m64n152k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6416032_MOD,     __obfuscate(".m64n160k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6416832_MOD,     __obfuscate(".m64n168k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6417632_MOD,     __obfuscate(".m64n176k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6418432_MOD,     __obfuscate(".m64n184k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6419232_MOD,     __obfuscate(".m64n192k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6420032_MOD,     __obfuscate(".m64n200k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6420832_MOD,     __obfuscate(".m64n208k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6421632_MOD,     __obfuscate(".m64n216k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6422432_MOD,     __obfuscate(".m64n224k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6423232_MOD,     __obfuscate(".m64n232k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6424032_MOD,     __obfuscate(".m64n240k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6424832_MOD,     __obfuscate(".m64n248k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6425632_MOD,     __obfuscate(".m64n256k32")),     \
    PICKATTR("SHAPE",      ptxSHAPE_640816_MOD,      __obfuscate(".m64n8k16")),       \
    PICKATTR("SHAPE",      ptxSHAPE_641616_MOD,      __obfuscate(".m64n16k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_642416_MOD,      __obfuscate(".m64n24k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_643216_MOD,      __obfuscate(".m64n32k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_644016_MOD,      __obfuscate(".m64n40k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_644816_MOD,      __obfuscate(".m64n48k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_645616_MOD,      __obfuscate(".m64n56k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_646416_MOD,      __obfuscate(".m64n64k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_647216_MOD,      __obfuscate(".m64n72k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_648016_MOD,      __obfuscate(".m64n80k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_648816_MOD,      __obfuscate(".m64n88k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_649616_MOD,      __obfuscate(".m64n96k16")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6410416_MOD,     __obfuscate(".m64n104k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6411216_MOD,     __obfuscate(".m64n112k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6412016_MOD,     __obfuscate(".m64n120k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6412816_MOD,     __obfuscate(".m64n128k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6413616_MOD,     __obfuscate(".m64n136k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6414416_MOD,     __obfuscate(".m64n144k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6415216_MOD,     __obfuscate(".m64n152k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6416016_MOD,     __obfuscate(".m64n160k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6416816_MOD,     __obfuscate(".m64n168k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6417616_MOD,     __obfuscate(".m64n176k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6418416_MOD,     __obfuscate(".m64n184k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6419216_MOD,     __obfuscate(".m64n192k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6420016_MOD,     __obfuscate(".m64n200k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6420816_MOD,     __obfuscate(".m64n208k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6421616_MOD,     __obfuscate(".m64n216k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6422416_MOD,     __obfuscate(".m64n224k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6423216_MOD,     __obfuscate(".m64n232k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6424016_MOD,     __obfuscate(".m64n240k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6424816_MOD,     __obfuscate(".m64n248k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6425616_MOD,     __obfuscate(".m64n256k16")),     \
    PICKATTR("SHAPE",      ptxSHAPE_640808_MOD,      __obfuscate(".m64n8k8")),        \
    PICKATTR("SHAPE",      ptxSHAPE_641608_MOD,      __obfuscate(".m64n16k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_642408_MOD,      __obfuscate(".m64n24k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_643208_MOD,      __obfuscate(".m64n32k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_644008_MOD,      __obfuscate(".m64n40k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_644808_MOD,      __obfuscate(".m64n48k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_645608_MOD,      __obfuscate(".m64n56k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_646408_MOD,      __obfuscate(".m64n64k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_647208_MOD,      __obfuscate(".m64n72k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_648008_MOD,      __obfuscate(".m64n80k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_648808_MOD,      __obfuscate(".m64n88k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_649608_MOD,      __obfuscate(".m64n96k8")),       \
    PICKATTR("SHAPE",      ptxSHAPE_6410408_MOD,     __obfuscate(".m64n104k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6411208_MOD,     __obfuscate(".m64n112k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6412008_MOD,     __obfuscate(".m64n120k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6412808_MOD,     __obfuscate(".m64n128k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6413608_MOD,     __obfuscate(".m64n136k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6414408_MOD,     __obfuscate(".m64n144k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6415208_MOD,     __obfuscate(".m64n152k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6416008_MOD,     __obfuscate(".m64n160k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6416808_MOD,     __obfuscate(".m64n168k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6417608_MOD,     __obfuscate(".m64n176k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6418408_MOD,     __obfuscate(".m64n184k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6419208_MOD,     __obfuscate(".m64n192k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6420008_MOD,     __obfuscate(".m64n200k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6420808_MOD,     __obfuscate(".m64n208k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6421608_MOD,     __obfuscate(".m64n216k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6422408_MOD,     __obfuscate(".m64n224k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6423208_MOD,     __obfuscate(".m64n232k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6424008_MOD,     __obfuscate(".m64n240k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6424808_MOD,     __obfuscate(".m64n248k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6425608_MOD,     __obfuscate(".m64n256k8")),      \
    PICKATTR("SHAPE",      ptxSHAPE_640864_MOD,      __obfuscate(".m64n8k64")),       \
    PICKATTR("SHAPE",      ptxSHAPE_641664_MOD,      __obfuscate(".m64n16k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_642464_MOD,      __obfuscate(".m64n24k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_643264_MOD,      __obfuscate(".m64n32k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_644064_MOD,      __obfuscate(".m64n40k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_644864_MOD,      __obfuscate(".m64n48k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_645664_MOD,      __obfuscate(".m64n56k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_646464_MOD,      __obfuscate(".m64n64k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_647264_MOD,      __obfuscate(".m64n72k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_648064_MOD,      __obfuscate(".m64n80k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_648864_MOD,      __obfuscate(".m64n88k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_649664_MOD,      __obfuscate(".m64n96k64")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6410464_MOD,     __obfuscate(".m64n104k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6411264_MOD,     __obfuscate(".m64n112k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6412064_MOD,     __obfuscate(".m64n120k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6412864_MOD,     __obfuscate(".m64n128k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6413664_MOD,     __obfuscate(".m64n136k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6414464_MOD,     __obfuscate(".m64n144k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6415264_MOD,     __obfuscate(".m64n152k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6416064_MOD,     __obfuscate(".m64n160k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6416864_MOD,     __obfuscate(".m64n168k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6417664_MOD,     __obfuscate(".m64n176k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6418464_MOD,     __obfuscate(".m64n184k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6419264_MOD,     __obfuscate(".m64n192k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6420064_MOD,     __obfuscate(".m64n200k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6420864_MOD,     __obfuscate(".m64n208k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6421664_MOD,     __obfuscate(".m64n216k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6422464_MOD,     __obfuscate(".m64n224k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6423264_MOD,     __obfuscate(".m64n232k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6424064_MOD,     __obfuscate(".m64n240k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6424864_MOD,     __obfuscate(".m64n248k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6425664_MOD,     __obfuscate(".m64n256k64")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6408256_MOD,     __obfuscate(".m64n8k256")),      \
    PICKATTR("SHAPE",      ptxSHAPE_6416256_MOD,     __obfuscate(".m64n16k256")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6424256_MOD,     __obfuscate(".m64n24k256")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6432256_MOD,     __obfuscate(".m64n32k256")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6448256_MOD,     __obfuscate(".m64n48k256")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6464256_MOD,     __obfuscate(".m64n64k256")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6480256_MOD,     __obfuscate(".m64n80k256")),     \
    PICKATTR("SHAPE",      ptxSHAPE_6496256_MOD,     __obfuscate(".m64n96k256")),     \
    PICKATTR("SHAPE",      ptxSHAPE_64112256_MOD,    __obfuscate(".m64n112k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64128256_MOD,    __obfuscate(".m64n128k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64144256_MOD,    __obfuscate(".m64n144k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64160256_MOD,    __obfuscate(".m64n160k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64176256_MOD,    __obfuscate(".m64n176k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64192256_MOD,    __obfuscate(".m64n192k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64208256_MOD,    __obfuscate(".m64n208k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64224256_MOD,    __obfuscate(".m64n224k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64240256_MOD,    __obfuscate(".m64n240k256")),    \
    PICKATTR("SHAPE",      ptxSHAPE_64256256_MOD,    __obfuscate(".m64n256k256"))
    


#define PTX_SHAPEMOD_TABLE(PICKATTR) PTX_SHAPEMOD_TABLE_RAW(PICKATTR)

#define PTX_LAYOUTMOD_TABLE_RAW(PICKATTR, PREFIX)                              \
    PICKATTR("LAYOUT",     ptxNOLAYOUT_MOD,         ""),                       \
    PICKATTR("LAYOUT",     ptxLAYOUT_ROW_MOD,       PREFIX"row"),              \
    PICKATTR("LAYOUT",     ptxLAYOUT_COL_MOD,       PREFIX"col")

#define PTX_LAYOUTMOD_TABLE(PICKATTR) PTX_LAYOUTMOD_TABLE_RAW(PICKATTR, ".")

#define PTX_DESCMOD_TABLE(PICKATTR)                                            \
    PICKATTR("DESC",       ptxNODESC_MOD,           ""),                       \
    PICKATTR("DESC",       ptxDESC_MOD,             __obfuscate(".desc"))

#define PTX_SPARSITYMOD_TABLE(PICKATTR) \
    PICKATTR("SPARSITY",   ptxNOSPARSITY_MOD,         ""),                     \
    PICKATTR("SPARSITY",   ptxSPARSITY_MOD,           ".sp")
     
#define PTX_SPFORMATMOD_TABLE(PICKATTR) \
    PICKATTR("SPFORMAT",   ptxNOSPFORMAT_MOD,         ""),                     \
    PICKATTR("SPFORMAT",   ptxSPFORMAT_THREAD_MOD,    __obfuscate(".thread")), \
    PICKATTR("SPFORMAT",   ptxSPFORMAT_REGOFFSET_MOD, __obfuscate(".regoffset"))

#define PTX_MMA_OPMOD_TABLE(PICKATTR) \
    PICKATTR("MMA_OP",     ptxNOMMA_MOD,                 ""),                  \
    PICKATTR("MMA_OP",     ptxMMA_AND_MOD,               ".and"),              \
    PICKATTR("MMA_OP",     ptxMMA_XOR_MOD,               ".xor"),              \
    PICKATTR("MMA_OP",     ptxMMA_POPC_MOD,              ".popc")

#define PTX_PREFETCHSIZEMOD_TABLE(PICKATTR) \
    PICKATTR("PREFETCHSIZE",    ptxNOPREFETCHSIZE_MOD,    ""),                  \
    PICKATTR("PREFETCHSIZE",    ptxPREFETCHSIZE_64B_MOD,  __obfuscate(".64B")), \
    PICKATTR("PREFETCHSIZE",    ptxPREFETCHSIZE_128B_MOD, __obfuscate(".128B")),\
    PICKATTR("PREFETCHSIZE",    ptxPREFETCHSIZE_256B_MOD, __obfuscate(".256B"))

#define PTX_CACHEPREFETCHMOD_TABLE(PICKATTR) \
    PICKATTR("CACHEPREFETCH",    ptxNOCACHEPREFETCH_MOD,       ""),             \
    PICKATTR("CACHEPREFETCH",    ptxCACHEPREFETCH_L2_64B_MOD,  __obfuscate(".L2::64B")),     \
    PICKATTR("CACHEPREFETCH",    ptxCACHEPREFETCH_L2_128B_MOD, __obfuscate(".L2::128B")),    \
    PICKATTR("CACHEPREFETCH",    ptxCACHEPREFETCH_L2_256B_MOD, __obfuscate(".L2::256B"))

#define ALL_TABLES(PICKATTR)         \
    PTX_COMPARE_TABLE(PICKATTR)      \
    PTX_KEEPREFMOD_TABLE(PICKATTR)   \
    PTX_NOATEXIT_TABLE(PICKATTR)     \
    PTX_OPERATOR_TABLE(PICKATTR)     \
    PTX_TYPES_TABLE(PICKATTR)        \
    PTX_APRXMOD_TABLE(PICKATTR)      \
    PTX_FTZMOD_TABLE(PICKATTR)       \
    PTX_XORSIGNMOD_TABLE(PICKATTR)   \
    PTX_NOFTZMOD_TABLE(PICKATTR)     \
    PTX_SATMOD_TABLE(PICKATTR)       \
    PTX_SATFMOD_TABLE(PICKATTR)      \
    PTX_CCMOD_TABLE(PICKATTR)        \
    PTX_SHAMTMOD_TABLE(PICKATTR)     \
    PTX_SCOPEMOD_TABLE(PICKATTR)     \
    PTX_TADDRMOD_TABLE(PICKATTR)     \
    PTX_ORDERMOD_TABLE(PICKATTR)     \
    PTX_NCMOD_TABLE(PICKATTR)        \
    PTX_ROUNDMOD_TABLE(PICKATTR)     \
    PTX_TESTPMOD_TABLE(PICKATTR)     \
    PTX_CACHEOPMOD_TABLE(PICKATTR)   \
    PTX_LEVELMOD_TABLE(PICKATTR)     \
    PTX_EVICTPRIORITYMOD_TABLE(PICKATTR) \
    PTX_LEVELEVICTPRIORITYMOD_TABLE(PICKATTR) \
    PTX_FLOWMOD_TABLE(PICKATTR)      \
    PTX_BRANCHMOD_TABLE(PICKATTR)    \
    PTX_EXCLUSIVEMOD_TABLE(PICKATTR) \
    PTX_VECTORMOD_TABLE(PICKATTR)    \
    PTX_TEXTUREMOD_TABLE(PICKATTR)   \
    PTX_DIMMOD_TABLE(PICKATTR)       \
    PTX_IM2COLMOD_TABLE(PICKATTR)    \
    PTX_PACKEDOFFMOD_TABLE(PICKATTR) \
    PTX_MULTICASTMOD_TABLE(PICKATTR) \
    PTX_MBARRIERMOD_TABLE(PICKATTR)  \
    PTX_FOOTPRINTMOD_TABLE(PICKATTR) \
    PTX_COARSEMOD_TABLE(PICKATTR)    \
    PTX_COMPONENTMOD_TABLE(PICKATTR) \
    PTX_VIDEOSELECTOR_TABLE(PICKATTR)\
    PTX_QUERYMOD_TABLE(PICKATTR)     \
    PTX_VOTEMOD_TABLE(PICKATTR)      \
    PTX_CLAMPMOD_TABLE(PICKATTR)     \
    PTX_SHRMOD_TABLE(PICKATTR)       \
    PTX_VMADMOD_TABLE(PICKATTR)      \
    PTX_PRMTMOD_TABLE(PICKATTR)      \
    PTX_SHFLMOD_TABLE(PICKATTR)      \
    PTX_ENDISMOD_TABLE(PICKATTR)     \
    PTX_UNIFORMMOD_TABLE(PICKATTR)   \
    PTX_RANDMOD_TABLE(PICKATTR)      \
    PTX_SHAPEMOD_TABLE(PICKATTR)     \
    PTX_LAYOUTMOD_TABLE(PICKATTR)    \
    PTX_SYNCMOD_TABLE(PICKATTR)      \
    PTX_NOINCMOD_TABLE(PICKATTR)     \
    PTX_NOCOMPLETEMOD_TABLE(PICKATTR)\
    PTX_NOSLEEPMOD_TABLE(PICKATTR)   \
    PTX_BARMOD_TABLE(PICKATTR)       \
    PTX_THREADSMOD_TABLE(PICKATTR)   \
    PTX_ALIGNMOD_TABLE(PICKATTR)     \
    PTX_TTUSLOTMOD_TABLE(PICKATTR)   \
    PTX_SHAREDSCOPEMOD_TABLE(PICKATTR) \
    PTX_PREFETCHSIZE_TABLE(PICKATTR) \
    PTX_CACHEPREFETCH_TABLE(PICKATTR) \
    PTX_TTUMOD_TABLE(PICKATTR)       \
    PTX_TRANSMOD_TABLE(PICKATTR)     \
    PTX_EXPANDMOD_TABLE(PICKATTR)    \
    PTX_SEQMOD_TABLE(PICKATTR)       \
    PTX_NUMMOD_TABLE(PICKATTR)       \
    PTX_GROUPMOD_TABLE(PICKATTR)     \
    PTX_TYPEMOD_TABLE(PICKATTR)      \
    PTX_MMA_OPMOD_TABLE(PICKATTR)    \
    PTX_THREADGROUP_TABLE(PICKATTR)  \
    PTX_SPARSITYMOD_TABLE(PICKATTR)  \
    PTX_SPFORMATMOD_TABLE(PICKATTR)  \
    PTX_DESCMOD_TABLE(PICKATTR)      \
    PTX_ADDRTYPEMOD_TABLE(PICKATTR)  \
    PTX_ABSMOD_TABLE(PICKATTR)       \
    PTX_CACHEHINTMOD_TABLE(PICKATTR) \
    PTX_OOBMOD_TABLE(PICKATTR)       \
    PTX_PROXYKINDMOD_TABLE(PICKATTR)
