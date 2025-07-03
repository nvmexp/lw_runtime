

#ifndef __PARSEASMDECLS_H_
#define __PARSEASMDECLS_H_

typedef enum {
    PARSE_LWVP_TEXT,
    PARSE_LWGP_TEXT,
    PARSE_LWFP_TEXT,
    PARSE_LWHP_TEXT,    // OGL: tessellation control
    PARSE_LWTP_TEXT,    // OGL: tessellation evaluation
    PARSE_LWCP_TEXT,    // OGL: compute
    PARSE_LWMP_TEXT,    // OGL: mesh
    PARSE_LWMTP_TEXT,   // OGL: task
    PARSE_ARBVP_TEXT,
    PARSE_ARBFP_TEXT,
    PARSE_D3D_VS1X,
    PARSE_D3D_VS2X,
    PARSE_D3D_VS3X,
    PARSE_D3D_VSBIN,
    PARSE_D3D_PS1X,
    PARSE_D3D_PS14,
    PARSE_D3D_PS2X,
    PARSE_D3D_PS3X,
    PARSE_D3D_PSBIN,
    PARSE_D3D10_SM4X,
    PARSE_D3D10_SM4X_BIN,
    PARSE_D3D_GS40,
    PARSE_LWINST_TEXT,
    PARSE_LWIR_TEXT,
    PARSE_SHD,
    PARSE_UNKNOWN
} ParseMethods;

#ifdef __cplusplus
extern "C" {
#endif

struct Parseasm_ProfileOption_Rec;

extern int o_verbosityLevel;
extern int o_validateLevel;
extern int o_ValidateAssert;
extern int o_evaluatorNumSets;
extern int o_showDebugID;
extern int o_ProfilerLevel;
extern int o_DumpStatsMemAllocBTree;
extern int o_DumpStatsAtEnd;
extern int o_NoCommandLine;
extern int o_d3dOpt;
extern int o_oglOpt;
extern int o_codeGenRuns;
extern int o_eProf;
extern int o_lwinst;
extern int o_ConstantComputationExtraction;
extern int o_ControlPS;
extern int o_ControlVS;
extern int o_NumCombiners;
extern int o_SupportSignedRemap;
extern int o_SingleIssueTex;
extern int o_TextureType;
extern int o_PackFloatInput;
extern int o_TextureRange;
extern int o_CanPromoteFixed;
extern int o_CanPromoteHalf;
extern int o_CanReorderFixedWithCheck;
extern int o_CanReorderHalf;
extern int o_CanReorderFloat;
extern int o_CanIgnoreNan;
extern int o_CanIgnoreInf;
extern int o_CanIgnoreSignedZero;
extern int o_CanDemoteNonFP32Targets;
extern int o_CanUseNrmhAlways;
extern int o_Const31InRange;
extern int o_FloatRT;
extern int o_Srgb;
extern int o_PartialTexLoad;
extern int o_TexShadowMap;
extern int o_SrcAlpha;
extern int o_SrcRGB;
extern int o_TextureRemap[16];
extern int o_TexSRGB;
extern int o_UseConstTexAlpha;
extern int o_AddFog;
extern int o_Binary;
extern char *o_BoolsString;
extern unsigned int o_ConstBoolValues;
extern int o_OnlyDumpInst;
extern int o_dumpTokenStream;

extern int o_RandomSched;
extern int o_SchedTargetRegSize;

#include "lwInst.h"

// input args
extern char *o_InputFileName;
extern char *o_InputListFileName;
extern char *o_InputFormat;
extern char *o_InputPhaseVectorFileName;
extern char *o_d3dArgs;
// output args
extern char *o_OutputFileName;
extern char *o_BaseOutputDir;
// codegen args
extern const char *o_Profile;
extern int o_MerlwryShortCirlwitLevel;
extern int o_Quadro;
    
extern int EnablePrinting;

extern int o_OptLevel;
extern char o_HasOptLevel;

// ------------------------------ State ------------------------------- //
// Defined in parseasmlib.c
extern int o_OriControl;
extern int o_debugOutputMask;
extern int o_Help;
extern int o_DumpText;
extern int o_DumpBinaryUCode;
extern int o_PostFileOptions;
extern int o_DebugBreakAlloc;
extern int o_DebugTrap;
extern int o_Hack;
extern int o_Hack2;
extern int o_ForceDX10SAT;
extern int o_ForceDX10AddressInRange;
extern int o_StartInst;
extern char *o_InfoFileName;
extern int o_SchedTargetRegSize;
extern int o_RandomSched;
extern int o_RandomSeed;
extern int o_DXVer;
extern char o_HasUseEarlyPhase1;
extern int o_EnableIOPacking;
extern int o_ArchVariant;
extern char o_HasArchVariant;
extern unsigned int o_MrtMask;
extern char o_OverrideMrtMask;
extern char o_UseEarlyPhase1;
extern unsigned int o_DeadOutputMask[DEAD_OUTPUT_MASK_ARR_LEN];
extern char o_HasDeadOutputMask;
extern char o_HasBindlessTextureBank;
extern int o_BindlessTextureBank;
extern int o_OptimizerConstantBank;
extern int o_SSAAttribReg;
extern unsigned int o_FlatUpperAttributes;
extern unsigned int o_UpperAttributeMask;
extern unsigned int o_maxRRegCount;
extern unsigned int o_schedRegTarget;
extern int o_hintControl;
extern int o_CompileAsFunction;
extern char *o_BoolsString;
extern int o_Hash;
extern int o_oglInfo;
extern unsigned int o_FmtFlags;
extern int o_SchedulerFlags;
extern char o_HasTBat;
extern char o_HasTBatMaxT2TClk;
extern char o_HasLat;
extern char o_HasI2ILat;
extern char o_AutoBatch;
extern int o_TBat;
extern int o_TBatMaxT2TClk;
extern int o_Lat;
extern int o_I2ILat;
extern char o_UsePhaseWindows;
extern char o_HasUsePhaseWindows;
extern int o_UseVLIW;
extern int o_Direct2IR;
extern int o_RegStressFlags;
extern int o_ShowInstFreq;
extern const char *o_VLIWfileName;
extern const char *o_mdesFileName;
extern const char *o_Tasks;
extern int o_EmitLwir;
extern int o_HasEmitLwir;
extern int o_ExperimentalA;
extern int o_ExperimentalB;
extern int o_ExperimentalC;
extern int o_ExperimentalD;
extern LWuCode **o_ppUCode; 
extern char o_HasUserPhaseCount;
extern int o_PhaseCount;
extern int o_EarlyPhase;
extern int o_RRegBankSize;
extern int o_DoUniformAtomExpand;
extern int o_DoFastUniformAtomExpand;
extern int o_DoUniformAtomExpandUsingSHFL;
extern int o_DoVectorAtomExpand;
extern int o_EnableZeroCoverageKill;
extern int o_CanUseGlobalScratch;
extern int o_SharedMemorySize;
extern int o_SMemScratchPerWarp;
extern int o_GMemScratchBase;
extern int o_GMemScratchConstBank;
extern int o_GMemScratchConstOffset;
extern int o_NoFloatMAD;
extern int o_ForceFMZ;
extern int o_DisableFTZ;
extern char o_IsWPOSOne;
extern char o_HasDoUniformAtomExpand;
extern char o_HasDoFastUniformAtomExpand;
extern char o_HasDoUniformAtomExpandUsingSHFL;
extern char o_HasDoVectorAtomExpand;
extern char o_HasEnableZeroCoverageKill;
extern char o_HasCanUseGlobalScratch;
extern char o_HasSharedMemorySize;
extern char o_HasSMemScratchPerWarp;
extern char o_HasGMemScratchBase;
extern char o_HasGMemScratchConstBank;
extern char o_HasGMemScratchConstOffset;
extern char o_HasNoFloatMAD;
extern char o_HasForceFMZ;
extern char o_HasDisableFTZ;
extern char o_HasSFUFloatMUL;
extern char o_HasTxdBatchSize;
extern int o_SFUFloatMUL;
extern int o_TxdBatchSize;
extern int o_vtxA;
extern int o_vtxB;
extern int o_ColwertFastGs;
extern int o_FastGs;
extern int o_FastGsWar;
extern char o_HasAddDummyCRead;
extern char o_AddDummyCRead;
extern int o_MaxInstInBasicBlock;
extern int o_TextureReduction;
extern int o_HasTextureReduction;
extern int o_MaxWarpsPerTile;
extern int o_HasMaxWarpsPerTile;
extern int o_RandomizeOrder;
extern const char *o_KnobsFileName;
extern char *o_KnobsString;
extern int o_SpelwlativelyHoistTex;
extern int o_DumpPerfStats;
extern int o_DumpPgmLatInfo;
extern int o_unknownLoopIttr;
extern int o_numberOfSM;
extern int o_numWarpsLaunched;
extern int o_vulkan;
extern int o_glinternal;
extern int o_usesTaskShader;

extern struct Parseasm_ProfileOption_Rec *o_ProfileStruct;
extern ParseMethods parseMethod;
extern char *commandLine;
extern char *chipVersion;

#ifdef __cplusplus
}
#endif

#endif // __PARSEASMDECLS_H_
