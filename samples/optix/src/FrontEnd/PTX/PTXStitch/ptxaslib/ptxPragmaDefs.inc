/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2020-2021, LWPU CORPORATION.  All rights reserved.
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
 *                Instead include ptxObfuscatedPragmaDefs.h.
 *                
 *                If a single pragma value requires to be marked with __obfuscate,
 *                then entire pragma class (ALL values) needs to be marked with
 *                __obfuscate. This avoids special casing in get*PrgAsString functions.
 */

#define GET_CLASS(CLASS, ENUM, NAME) CLASS
#define GET_ENUM(CLASS, ENUM, NAME) ENUM
#define GET_NAME(CLASS, ENUM, NAME) NAME
#define GET_ALL(CLASS, ENUM, NAME) (CLASS ENUM NAME)

#define PTX_NOUNROLL_TABLE(PICKATTR)                                           \
    PICKATTR("NOUNROLL", ptxUnroll_PRG,         ""),                           \
    PICKATTR("NOUNROLL", ptxNoUnroll_PRG,       "nounroll")

#define PTX_NOPTRTRUNC_TABLE(PICKATTR)                                         \
    PICKATTR("NOPTRTRUNC", ptxPointerTrunc_PRG,     ""),                       \
    PICKATTR("NOPTRTRUNC", ptxNoPointerTrunc_PRG,   __obfuscate("noPointerTrunc"))

#define PTX_JETFIRE_TABLE(PICKATTR)                                                 \
    PICKATTR("JETFIRE", ptxNoJetfire_PRG,       ""),                                \
    PICKATTR("JETFIRE", ptxSetKnob_PRG,         __obfuscate("set knob")),           \
    PICKATTR("JETFIRE", ptxResetKnob_PRG,       __obfuscate("reset knob")),         \
    PICKATTR("JETFIRE", ptxNextKnob_PRG,        __obfuscate("next knob")),          \
    PICKATTR("JETFIRE", ptxGlobalKnob_PRG,      __obfuscate("global knob")),        \
    PICKATTR("JETFIRE", ptxLwopt_PRG,           __obfuscate("lwopt")),              \
    PICKATTR("JETFIRE", ptxDisPrgKnobs_PRG,     __obfuscate("DISABLEPRAGMAKNOBS"))

#define PTX_FREQUENCY_TABLE(PICKATTR)                                          \
    PICKATTR("FREQUENCY", ptxNoFrequency_PRG,   ""),                           \
    PICKATTR("FREQUENCY", ptxFrequency_PRG,     __obfuscate("frequency"))

#define PTX_LWSTOMABI_TABLE(PICKATTR)                                                                              \
    PICKATTR("LWSTOMABI", ptxNoLwstomAbi_PRG,                   ""),                                               \
    PICKATTR("LWSTOMABI", ptxAbiParamReg_PRG,                   __obfuscate("abi_param_reg")),                     \
    PICKATTR("LWSTOMABI", ptxCallAbiParamReg_PRG,               __obfuscate("call_abi_param_reg")),                \
    PICKATTR("LWSTOMABI", ptxRetAddrRegBeforeParamReg_PRG,      __obfuscate("retaddr_reg_before_param_reg")),      \
    PICKATTR("LWSTOMABI", ptxCallRetAddrRegBeforeParamReg_PRG,  __obfuscate("call_retaddr_reg_before_param_reg")), \
    PICKATTR("LWSTOMABI", ptxRetAddrReg_PRG,                    __obfuscate("retaddr_reg")),                       \
    PICKATTR("LWSTOMABI", ptxRetAddrRegU_PRG,                   __obfuscate("retaddr_reg<U>")),                    \
    PICKATTR("LWSTOMABI", ptxRetAddrRegR_PRG,                   __obfuscate("retaddr_reg<R>")),                    \
    PICKATTR("LWSTOMABI", ptxCallRetAddrReg_PRG,                __obfuscate("call_retaddr_reg")),                  \
    PICKATTR("LWSTOMABI", ptxScratchRegsR_PRG,                  __obfuscate("scratch_regs<R>")),                   \
    PICKATTR("LWSTOMABI", ptxScratchRegsB_PRG,                  __obfuscate("scratch_regs<B>")),                   \
    PICKATTR("LWSTOMABI", ptxCallScratchRegsR_PRG,              __obfuscate("call_scratch_regs<R>")),              \
    PICKATTR("LWSTOMABI", ptxCallScratchRegsB_PRG,              __obfuscate("call_scratch_regs<B>")),              \
    PICKATTR("LWSTOMABI", ptxCallRetAddrRegU_PRG,               __obfuscate("call_retaddr_reg<U>")),               \
    PICKATTR("LWSTOMABI", ptxCallRetAddrRegR_PRG,               __obfuscate("call_retaddr_reg<R>")),               \
    PICKATTR("LWSTOMABI", ptxLocalMaxNReg_PRG,                  __obfuscate("local_maxnreg")),                     \
    PICKATTR("LWSTOMABI", ptxRetAddrRegRRel32_PRG,              __obfuscate("retaddr_reg<R:rel32>")),              \
    PICKATTR("LWSTOMABI", ptxCallRetAddrRegRRel32_PRG,          __obfuscate("call_retaddr_reg<R:rel32>"))

#define PTX_COROUTINE_TABLE(PICKATTR)                                                                              \
    PICKATTR("COROUTINE", ptxNoCoroutine_PRG,                   ""),                                               \
    PICKATTR("COROUTINE", ptxCoroutine_PRG,                   __obfuscate("coroutine")),                           \
    PICKATTR("COROUTINE", ptxCallCoroutine_PRG,               __obfuscate("call_coroutine"))

#define PTX_SYNC_TABLE(PICKATTR)                                                                                   \
    PICKATTR("SYNC", ptxConfirm_PRG,                            __obfuscate("confirm")),                           \
    PICKATTR("SYNC", ptxPredict_PRG,                            __obfuscate("predict"))

// TODO: Move the below table at suitable place.

#define PTX_PREPROCMACROFLAG_TABLE(PICKATTR)                                                                     \
    PICKATTR("PREPROCMACROFLAG", ptxNoPreProcMacroFlag_FLG,       ""),                                           \
    PICKATTR("PREPROCMACROFLAG", ptxSupportFastDiv_FLG,           __obfuscate("SUPPORT_FAST_DIVISION")),         \
    PICKATTR("PREPROCMACROFLAG", ptxNeedVideoEmulation_FLG,       __obfuscate("NEED_VIDEO_EMULATION")),          \
    PICKATTR("PREPROCMACROFLAG", ptxForceAlignedSyncInstr_FLG,    __obfuscate("FORCE_ALIGNED_SYNC_INSTRS")),     \
    PICKATTR("PREPROCMACROFLAG", ptxLegacyBarWarpWide_FLG,        __obfuscate("LEGACY_BAR_WARP_WIDE_BEHAVIOR")), \
    PICKATTR("PREPROCMACROFLAG", ptxForceOutlinedWmma_FLG,        __obfuscate("FORCE_OUTLINED_WMMA")),           \
    PICKATTR("PREPROCMACROFLAG", ptxEnableUnalignedWmmaInstr_FLG, __obfuscate("ENABLE_UNALIGNED_WMMA_INSTRS")),  \
    PICKATTR("PREPROCMACROFLAG", ptxDisableSuperHmma_FLG,         __obfuscate("DISABLE_SUPER_HMMA")),            \
    PICKATTR("PREPROCMACROFLAG", ptxUseMMA884Emulation_FLG,       __obfuscate("USE_MMA884_EMULATION")),          \
    PICKATTR("PREPROCMACROFLAG", ptxIsMerlwry_FLG,                __obfuscate("IS_MERLWRY")),                    \
    PICKATTR("PREPROCMACROFLAG", ptxExpandTexIntoSyscall_FLG,     __obfuscate("EXPAND_TEX_INTO_SYSCALL")),       \
    PICKATTR("PREPROCMACROFLAG", ptxExpandSyncInstLate_FLG,       __obfuscate("EXPAND_SYNC_INST_LATE"))

// TODO: Move the below table at suitable place.

#define PTX_ERRMSGSTR_TABLE(PICKATTR)                                                                \
    PICKATTR("ERRMSGSTR", ptxNoErrMsg_STR,          ""),                                             \
    PICKATTR("ERRMSGSTR", ptxSparse_STR,            __obfuscate("Sparse ")),                         \
    PICKATTR("ERRMSGSTR", ptxSparsity_STR,          __obfuscate("sparsity")),                        \
    PICKATTR("ERRMSGSTR", ptxSpformat_STR,          __obfuscate("sparse format")),                   \
    PICKATTR("ERRMSGSTR", ptxSparseMMA_STR,         __obfuscate("Sparse MMA")),                      \
    PICKATTR("ERRMSGSTR", ptxSparseMMAINT_STR,      __obfuscate("Sparse MMA with integer types")),   \
    PICKATTR("ERRMSGSTR", ptxSparseMMAINTSHAPE_STR, __obfuscate("Sparse Integer MMA with shape ")),  \
    PICKATTR("ERRMSGSTR", ptxIdxSize_STR,           __obfuscate("idxSize = ")),                      \
    PICKATTR("ERRMSGSTR", ptxSeqAndIdxSize_STR,     __obfuscate(".seq modifier and idxSize = ")),    \
    PICKATTR("ERRMSGSTR", ptxEleFormat_STR,         __obfuscate("eleFormat = ")),                    \
    PICKATTR("ERRMSGSTR", ptxElSize_STR,            __obfuscate("elSize = ")),                       \
    PICKATTR("ERRMSGSTR", ptxMMA884_STR,            __obfuscate("_mma.m8n8k4")),                     \
    PICKATTR("ERRMSGSTR", ptxCopyAsyncGroupMod_STR, __obfuscate("cp.async group")),                  \
    PICKATTR("ERRMSGSTR", ptxMemDesc_STR,           __obfuscate("Modifier .desc")),                  \
    PICKATTR("ERRMSGSTR", ptxCopyInstr_STR,         __obfuscate("Instruction 'cp'")),                \
    PICKATTR("ERRMSGSTR", ptxMBarrierInstr_STR,     __obfuscate("Instruction 'mbarrier'")),          \
    PICKATTR("ERRMSGSTR", ptxReduxInstr_STR,        __obfuscate("Instruction 'redux'")),             \
    PICKATTR("ERRMSGSTR", ptxQGMMA_STR,             __obfuscate("GMMA with FP8 types")),             \
    PICKATTR("ERRMSGSTR", ptxFPGMMA_STR,            __obfuscate("GMMA with floating point types")),  \
    PICKATTR("ERRMSGSTR", ptxIGMMA_STR,             __obfuscate("GMMA with integer types")),  \
    PICKATTR("ERRMSGSTR", ptxBGMMA_STR,             __obfuscate("GMMA with boolean type")),  \
    PICKATTR("ERRMSGSTR", ptxWARPGROUP_STR,         __obfuscate("_warpgroup instruction"))

// TODO: Move the below table at suitable place.

#define PTX_OPTIONSTR_TABLE(PICKATTR)                                                                \
    PICKATTR("OPTIONSTR", ptxNoOption_STR,          ""),                                             \
    PICKATTR("OPTIONSTR", ptxExtDescFile_STR,       "ext-desc-file"),                                \
    PICKATTR("OPTIONSTR", ptxExtDescString_STR,     "ext-desc-string")

#define PTX_SRSTR_TABLE(PICKATTR)                                                                    \
    PICKATTR("SR_STR", ptxNoSR_STR,                 ""),                                             \
    PICKATTR("SR_STR", ptxStackinit_entry_STR,      __obfuscate("%stackinit_entry")),                \
    PICKATTR("SR_STR", ptxStackend_STR,             __obfuscate("%stackend"))

#define PTX_BUILTINS_TABLE(PICKATTR)                                                                 \
    PICKATTR("BUILTINS", ptxNoBuiltins_STR,         ""),                                             \
    PICKATTR("BUILTINS", ptxBuiltin_STR,            __obfuscate("__lw_ptx_builtin_")),               \
    PICKATTR("BUILTINS", ptxSuspendBuiltin_STR,     __obfuscate("__lw_ptx_builtin_suspend")),        \
    PICKATTR("BUILTINS", ptxOcgBuiltin_STR,         __obfuscate("__lw_ptx_builtin_ocg_")),            \
    PICKATTR("BUILTINS", ptxOcgBuiltinGenomics_STR, __obfuscate("__lw_ptx_builtin_ocg_vi"))

#define ALL_PRAGMA_TABLES(PICKATTR)          \
PTX_NOUNROLL_TABLE(PICKATTR)                 \
PTX_NOPTRTRUNC_TABLE(PICKATTR)               \
PTX_JETFIRE_TABLE(PICKATTR)                  \
PTX_FREQUENCY_TABLE(PICKATTR)                \
PTX_LWSTOMABI_TABLE(PICKATTR)                \
PTX_SYNC_TABLE(PICKATTR)                     \
PTX_COROUTINE_TABLE(PICKATTR)

#define MISC_SENSITIVE_STR_TABLES(PICKATTR) \
PTX_PREPROCMACROFLAG_TABLE(PICKATTR)        \
PTX_ERRMSGSTR_TABLE(PICKATTR)               \
PTX_OPTIONSTR_TABLE(PICKATTR)               \
PTX_SRSTR_TABLE(PICKATTR)                   \
PTX_BUILTINS_TABLE(PICKATTR)
