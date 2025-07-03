/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2009-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

// File: parseasmlib.cpp

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h> /* Added for time measurement testing */
#include <assert.h>
#ifdef _MSC_VER
#include <crtdbg.h>
#include <windows.h>
#endif
#include <ctype.h>

#include "lwTypes.h"

#include "copi_inglobals.h"
#include "cop_parseasm.h"

#if defined(ALLOW_D3D_BUILD)
#include "parseD3D.h"
    //this is required for the moment as we transition the 4X code to the unified
    //vertex format
#endif // ALLOW_D3D_BUILD

#if defined(NEED_PARSESHD)
#include "parseSHD.h"
#endif // NEED_PARSESHD

#if defined(LW_LDDM_DDK_BUILD)
#include "parseD3D10.h"
#endif

#include "parseasmdecls.h"
#include "parseasmfun.h"
#include "parseasm_profileoption.h"

// Used for version info
#include "lwBldVer.h"
#include "lwver.h"
#include "parseogl.h"

#ifndef ALEN
#define ALEN(X) ((int) (sizeof(X) / sizeof(X[0])))
#endif

// --------------------------------------------------------------------------------------------

/*
 * lCallExit() - Wrapper to call "exit()".  Allows use of a single breakpoint to catch multiple
 *         exit paths and improves your debugging experience.
 */

void lCallExit(int returlwalue)
{
    exit(returlwalue);
} // lCallExit

int o_Help = 0;
// input args
char *o_InputFileName = NULL;
char *o_InputListFileName = NULL;
char *o_InputFormat = NULL;
char *o_InputPhaseVectorFileName = NULL;
char *o_d3dArgs = NULL;
// output args
char *o_OutputFileName = NULL;
char *o_BaseOutputDir = NULL;
// codegen args
const char *o_Profile = NULL;
int o_MerlwryShortCirlwitLevel = 0;
int o_Quadro = 0;
int o_eProf = 0;
int o_lwinst = 0;
int o_Binary = 0;
int o_DumpText = 0;
int o_DumpBinaryUCode = 0;
int o_d3dOpt = 0;
int o_oglOpt = 0;
int o_PostFileOptions = 0;
// debug args
int o_verbosityLevel = 0;
int o_validateLevel = 0;
int o_ValidateAssert = 0;
int o_evaluatorNumSets = 20;
int o_showDebugID = 0;
int o_ProfilerLevel = -1;         // -1 means use provided by lwinst
int o_DumpStatsMemAllocBTree = 0;
int o_DumpStatsAtEnd = 0;
int o_NoCommandLine = 0;
int o_DebugBreakAlloc = -1;
int o_DebugTrap = 0;
// Scratch flags useful for debugging things
int o_Hack = -1;
int o_Hack2 = -1;
// optimizer flag args
int o_CanDemoteNonFP32Targets = -1; // -1 means use platform default
int o_CanIgnoreNan = -1;            // -1 means use platform default
int o_CanIgnoreInf = -1;            // -1 means use platform default
int o_CanIgnoreSignedZero = -1;     // -1 means use platform default
int o_CanPromoteFixed = -1;         // -1 means use platform default
int o_CanPromoteHalf = -1;          // -1 means use platform default
int o_CanReorderFixedWithCheck = -1;// -1 means use platform default
int o_CanReorderHalf = -1;          // -1 means use platform default
int o_CanReorderFloat = -1;         // -1 means use platform default
int o_CanUseNrmhAlways = -1;        // -1 means use platform default
int o_Const31InRange;               // const 31 in 0..1 range (for Far Cry fog hoist)
int o_ControlPS = ~0;               // default run all fragment optimize
int o_ControlVS = ~0;               // default run all vertex optimize
int o_FloatRT = 0;
int o_AddFog = 0;                  // Add Fog instructions to shader: default: don't set
char *o_InfoFileName = NULL;
int o_NumCombiners = 0;             // how many combiners run after the program (used for
                                    // cyclecount estimate)
int o_PartialTexLoad = 0;           // Partial load of textures: default: don't set
int o_TexShadowMap = 0;             // Texture shadow map: default: don't set
int o_SchedTargetRegSize = 0;       // scheduling register target
int o_RandomSched = 0;              // introduce randomness in scheduler
int o_RandomSeed = -1;              // random seed to use
int o_SupportSignedRemap = 0;       // default don't allow colwersion of all textures
int o_SrcAlpha = 1;                 // Does output mask include alpha ? 0 = no, 1 = yes, default = 1
int o_SrcRGB = 1;                   // Does output mask include RGB ? 0 = no, 1 = yes, default = 1
int o_Srgb = 0;                     // Add SRGB to D3D shaders: default: don't set
int o_TextureRange = 0;             // what are the ranges of values in each texture
int o_TextureRemap[16] = {-1, -1, -1, -1,
                          -1, -1, -1, -1,
                          -1, -1, -1, -1,
                          -1, -1, -1, -1};
int o_TextureType = 0;              // types of textures. 2 bits per texture
int o_TexSRGB = 0;
int o_UseConstTexAlpha = 0;
int o_PackFloatInput = 0;           // packed float textures, 2 bits per texture
int o_ForceDX10SAT = 0;             // Enforce SAT(NaN) = 0.0
int o_ForceDX10AddressInRange = 0;  // Enforce DX10 requirement of base & offset in range 
int o_StartInst = 0;                // The starting instruction number of the input program

// Set the maximum number of instructions in a basic block
int o_MaxInstInBasicBlock = MAX_INSTRUCTIONS_IN_BASIC_BLOCK;

// d3d flags
int o_DXVer = 9;
int o_ConstantComputationExtraction = 0;
unsigned int o_ConstBoolValues = 0;
char *o_BoolsString = NULL;
int o_dumpTokenStream = 0;

// Misc flags
int o_codeGenRuns = 0;
int o_Hash = 0;
int o_oglInfo = 0;
// profile flags
// lw50:
unsigned int o_DeadOutputMask[DEAD_OUTPUT_MASK_ARR_LEN] = { 0, };
char o_HasDeadOutputMask = 0;
unsigned int o_MrtMask = 0;
char o_OverrideMrtMask = 0;
unsigned int o_FmtFlags = FMT_FLAG_NONE;
int o_SchedulerFlags = 0;
char o_HasTBat = 0;
char o_HasTBatMaxT2TClk = 0;
char o_HasLat = 0;
char o_HasI2ILat = 0;
char o_AutoBatch = 0;
int o_TBat = 0;
int o_TBatMaxT2TClk = 0;
int o_Lat = 0;
int o_I2ILat = 0;
char o_UsePhaseWindows = 0;
char o_HasUsePhaseWindows = 0;
char o_UseEarlyPhase1 = 0;
char o_HasUseEarlyPhase1 = 0;
int o_UseVLIW = 0;
int o_Direct2IR = 0;
int o_RegStressFlags = 0;
int o_ShowInstFreq = 0;
const char *o_VLIWfileName = NULL;
const char *o_mdesFileName = NULL;
const char *o_Tasks = NULL;
int o_TextureReduction = 0;
int o_HasTextureReduction = 0;
int o_MaxWarpsPerTile = 0;
int o_HasMaxWarpsPerTile = 0;
int o_EmitLwir = 0;
int o_HasEmitLwir = 0;
int o_ExperimentalA = 0;
int o_ExperimentalB = 0;
int o_ExperimentalC = 0;
int o_ExperimentalD = 0;
char o_HasBindlessTextureBank = 0;
int o_BindlessTextureBank = 0;
int o_OptimizerConstantBank = 4;
int o_SSAAttribReg = 4*aiFace;
unsigned int o_FlatUpperAttributes = 0;
unsigned int o_UpperAttributeMask = 0;
char o_HasUserPhaseCount = 0;
int o_PhaseCount = 0;
int o_EarlyPhase = 0;
int o_RRegBankSize = 0;
unsigned int o_maxRRegCount = 0;
unsigned int o_schedRegTarget = 0;
int o_hintControl = 0;
int o_CompileAsFunction = 0;
int o_OnlyDumpInst = 0;
int o_EnableIOPacking = 1;
int o_ArchVariant = 0;
char o_HasArchVariant = 0;
int o_CanUseGlobalScratch = 0;
int o_DoUniformAtomExpand = 0;
int o_DoFastUniformAtomExpand = 0;
int o_DoUniformAtomExpandUsingSHFL = 0;
int o_DoVectorAtomExpand = 0;
int o_EnableZeroCoverageKill= 0;
int o_SharedMemorySize = 0;
int o_SMemScratchPerWarp = 0;
int o_GMemScratchBase = 0;
int o_GMemScratchConstBank = 4;
int o_GMemScratchConstOffset = 0;
int o_NoFloatMAD = 0;
int o_ForceFMZ = 0;
int o_DisableFTZ = 0;
int o_SFUFloatMUL = 0;
char o_HasCanUseGlobalScratch = 0;
char o_IsWPOSOne = 0;
char o_HasDoFastUniformAtomExpand = 0;
char o_HasDoUniformAtomExpandUsingSHFL = 0;
char o_HasDoUniformAtomExpand = 0;
char o_HasDoVectorAtomExpand = 0;
char o_HasEnableZeroCoverageKill = 0;
char o_HasSharedMemorySize = 0;
char o_HasSMemScratchPerWarp = 0;
char o_HasGMemScratchBase = 0;
char o_HasGMemScratchConstBank = 0;
char o_HasGMemScratchConstOffset = 0;
char o_HasNoFloatMAD = 0;
char o_HasForceFMZ = 0;
char o_HasDisableFTZ = 0;
char o_HasSFUFloatMUL = 0;
char o_HasTxdBatchSize = 0;
int o_TxdBatchSize = 0;
int o_vtxA = 0;
int o_vtxB = 0;
int o_ColwertFastGs = 0;
int o_FastGs = 0;
int o_FastGsWar = 0;
char o_HasAddDummyCRead = 0;
char o_AddDummyCRead = 0;
int o_ExportConstants = 0; // Specify if constants should be exported or evaluated
int o_D3DModShaderAll16 = 0; // Specifies that the D3D shader should be set to all 16bit operations/registers
int o_D3DModShaderAll32 = 0; // Specifies that the D3D shader should be set to all 32bit operations/registers
const char *o_KnobsFileName = NULL;
 // String containing all knobs concatenated together, separated by whitespace
char *o_KnobsString = NULL; 
int o_SpelwlativelyHoistTex = -1;
int o_DumpPerfStats = 0;
int o_DumpPgmLatInfo = 0;
int o_unknownLoopIttr = 0;
int o_numberOfSM = 0;
int o_numWarpsLaunched = 0;
int o_vulkan = 0;
int o_glinternal = 0;
int o_usesTaskShader = -1;

Parseasm_ProfileOption *o_ProfileStruct = NULL;
ParseMethods parseMethod = PARSE_UNKNOWN;
char *commandLine = NULL;
char *chipVersion = NULL;

// To enble export of these values...
char *o_BuildBranchVersion = LW_BUILD_BRANCH_VERSION;
char *o_DisplayDriverTitle = LW_DISPLAY_DRIVER_TITLE;
unsigned long o_BuildChangelistNum = LW_BUILD_CHANGELIST_NUM;
unsigned long o_LastOfficialChangelistNum = LW_LAST_OFFICIAL_CHANGELIST_NUM;
char *o_DriverFormalVersion = LW_VERSION_STRING;

// LWTRACE varaibles
int lwrrentDebugLevel = 50;
int EnablePrinting = 0;
int o_WriteToCompiledShaderData = 0;

int o_OptLevel = 0;                 // default: unspecified optimization level
char o_HasOptLevel;                 // Has optLevl been specified ?
int o_OriControl;
int o_debugOutputMask;

// Needed for D3D vertex path to lw50 - should go away some time.
// With the latest changes to D3D this is now also needed for pixel programs!  Things just keep on getting worse!
LWuCode **o_ppUCode; 

#if defined(__cplusplus)
extern "C" {
#endif

static ParamsForCOP lGlobalParamsForCOP;
ParamsForCOP *GlobalParamsForCOP = &lGlobalParamsForCOP;

void InitParamsForCOP(ParamsForCOP *fParams);
void InitCOPBaseArgs(COPBaseArgs *fargs, ParamsForCOP *fParams);

extern int ParseShader(COPBaseArgs *args, char *source,
    int(*lookupConstFnPtrIn)(char *name, void*), void* lookupArg);

extern void *pCompiledShaderData;
extern unsigned long compiledShaderSize;

#if defined(__cplusplus)
}
#endif


extern void ProcessFileOptions(char *fString, unsigned fSize);


/*
 * ParseShaderAsm()
 *
 */

extern "C"
static int ParseShaderAsm(COPBaseArgs *args, char *source)
{
    int ret = ParseShader(args, source, 0, 0);

    if (ret == 0) {
        if (args->eOutputMode == -1)
            fpFindOutputInfo(args->inInsts, args);
    }
    return ret;
} // ParseShaderAsm

/*
 * lGetFreq() : Duplicate of COPProfiler::GetFreq()
 *
 */

static double lGetFreq() 
{
    double f;

#ifdef _MSC_VER
    LARGE_INTEGER freq;

    if(QueryPerformanceFrequency(&freq)) {
        f = (double) freq.QuadPart;
    } else {
        f = (double) CLOCKS_PER_SEC;
    }
#else
    f = (double) CLOCKS_PER_SEC;
#endif // _MSC_VER
    return f;
} // lGetFreq

/*
 * lGetTick() : Duplicate of COPProfiler::GetTick()
 *
 */

static LwU64 lGetTick() 
{
    LwU64 t;

#ifdef _MSC_VER
    LARGE_INTEGER tick;

    if (QueryPerformanceCounter(&tick)) {
        t = (LwU64) tick.QuadPart;
    } else {
        t = (LwU64) clock();
    }
#else    
    t = (LwU64) clock();
#endif // _MSC_VER
    return t;
} // lGetTick


/*
 * compileOnceMemoryBuffer()
 *
 */

int compileOnceMemoryBuffer(unsigned char *lRawInput, int lSize, FILE *outFile,
                            LWuCode **ppUCode, void **ppCgBinaryData, 
                            unsigned int *pCgBinarySize, int fBinary)
{
    int retVal = 0;
    COPVSArgs vsArgs;
    COPPSArgs psArgs;
    COPGSArgs gsArgs;
    COPCSArgs csArgs;
    COPHSArgs hsArgs;
    COPTSArgs tsArgs;
    COPMSArgs msArgs;
    COPMTSArgs mtsArgs;
    COPBaseArgs *pBaseArgs;
    lwInst *outInsts = NULL;
    lwInst *outStart = NULL;
    int ii, lCompileCount = 1;
    double startTime = 0;
    double stopTime  = 0;
    static double MAX_PROF_TIME, FREQUENCY;
    unsigned int len;

    FREQUENCY = lGetFreq();
    MAX_PROF_TIME = 10.0 * FREQUENCY;

    if (o_PostFileOptions) {
        ProcessFileOptions((char *)lRawInput, lSize);
    }

    if (o_BoolsString) {
        if (!_stricmp(o_BoolsString, "random")) {
            // Randomize the input...
            unsigned int unFlag = 1;
            int ii;
            for (ii = 0; ii < 32; ii++) {
                if (rand() % 2)
                    o_ConstBoolValues |= unFlag;
                unFlag <<= 1;
            }
        } else {
            sscanf(o_BoolsString, "0x%x", &o_ConstBoolValues);
        }
    }

    if (o_RandomSched) {
        unsigned seed;
        if (o_RandomSeed != -1) {
            seed = o_RandomSeed;
        } else {
            // use random seed based on time
            seed = (unsigned)time(0);
        }
        printf("seed srand(%d)\n",seed);
        srand(seed);
    }

    if (o_codeGenRuns > 0) {
        lCompileCount = o_codeGenRuns;
        startTime = (double) lGetTick();
    }

    if (o_eProf & prVertexPrograms) {
        pBaseArgs = &vsArgs.baseArgs;
        InitCOPVSArgs(&vsArgs, GlobalParamsForCOP);
    } else if (o_eProf & prPixelPrograms) {
        pBaseArgs = &psArgs.baseArgs;
        InitCOPPSArgs(&psArgs, GlobalParamsForCOP);
    } else if (o_eProf & prGeometryPrograms) {
        if (parseMethod == PARSE_D3D_GS40) {
            // HACK!!!!! D3D lwrrently uses COPVSArgs for Geometry programs
            pBaseArgs = &vsArgs.baseArgs;
            InitCOPVSArgs(&vsArgs, GlobalParamsForCOP);
        }
        else {
            pBaseArgs = &gsArgs.baseArgs;
            InitCOPGSArgs(&gsArgs, GlobalParamsForCOP);
        }
    } else if (o_eProf & prComputePrograms) {
        pBaseArgs = &csArgs.baseArgs;
        InitCOPCSArgs(&csArgs, GlobalParamsForCOP);
    } else if (o_eProf & prHullPrograms) {
        pBaseArgs = &hsArgs.baseArgs;
        InitCOPHSArgs(&hsArgs, GlobalParamsForCOP);
    } else if (o_eProf & prTessellationPrograms) {
        pBaseArgs = &tsArgs.baseArgs;
        InitCOPTSArgs(&tsArgs, GlobalParamsForCOP);
    } else if (o_eProf & prMeshPrograms) {
        pBaseArgs = &msArgs.baseArgs;
        InitCOPMSArgs(&msArgs, GlobalParamsForCOP);
    } else if (o_eProf & prTaskPrograms) {
        pBaseArgs = &mtsArgs.baseArgs;
        InitCOPMTSArgs(&mtsArgs, GlobalParamsForCOP);
    } else {
        assert(!"unexpected program type");
    }


    for (ii = 0; ii < lCompileCount; ii++) {

        InitParamsForCOP(GlobalParamsForCOP);

        if (ppUCode) {
            *ppUCode = NULL;
            o_ppUCode = ppUCode;
        }

        GlobalParamsForCOP->outFile = outFile;
        GlobalParamsForCOP->UnknownTexturesAre2D = TRUE;
        GlobalParamsForCOP->commandLine = o_NoCommandLine ? NULL : commandLine;
        GlobalParamsForCOP->infoFileName = o_InfoFileName;
        if (o_eProf == prTeslaFP || o_eProf == prTeslaVP || o_eProf == prTeslaGP || 
            o_eProf == prTeslaCP || o_eProf == prFermiFP || o_eProf == prFermiVP || 
            o_eProf == prFermiGP || o_eProf == prFermiCP || o_eProf == prFermiHP || 
            o_eProf == prFermiTP || o_eProf == prTuringMP || o_eProf == prTuringMTP)
        {
            if (o_lwinst) {
                GlobalParamsForCOP->UseTextFileOutput = 0;
            } else {
                GlobalParamsForCOP->UseTextFileOutput = 1;
            }
            GlobalParamsForCOP->optimizerConstBank = o_OptimizerConstantBank;
        } else {
            GlobalParamsForCOP->UseTextFileOutput = 0;
        }
        if (o_HasBindlessTextureBank) {
            GlobalParamsForCOP->OverrideBindlessTextureBank = 1;
            GlobalParamsForCOP->parseasmBindlessTextureBank = o_BindlessTextureBank;
        }
        GlobalParamsForCOP->fmtFlags = o_FmtFlags;
#if 111
        GlobalParamsForCOP->schedulerFlags = o_SchedulerFlags;
#endif
        if (o_HasTBat) {
            GlobalParamsForCOP->OverrideDriverTBat = 1;
            GlobalParamsForCOP->parseasmTBat = o_TBat;
            GlobalParamsForCOP->ParseasmAutobatch = o_AutoBatch;
        }
        if (o_HasTBatMaxT2TClk) {
            GlobalParamsForCOP->OverrideDriverTBatMaxT2TCycles = 1;
            GlobalParamsForCOP->parseasmTBatMaxT2TCycles = o_TBatMaxT2TClk;
        }
        if (o_HasLat) {
            GlobalParamsForCOP->OverrideDriverLat = 1;
            GlobalParamsForCOP->parseasmLatency = o_Lat;
        }
        if (o_HasI2ILat) {
            GlobalParamsForCOP->OverrideI2ILatency = 1;
            GlobalParamsForCOP->parseasmI2iIssueLatency = o_I2ILat;
        }
        if (o_HasUserPhaseCount) {
            GlobalParamsForCOP->OverrideDriverMinPhase = 1;
            GlobalParamsForCOP->parseasmMinPhase = o_PhaseCount;
        }
        if (o_HasUsePhaseWindows) {
            GlobalParamsForCOP->OverridePhaseWindows = 1;
            GlobalParamsForCOP->parseasmPhaseWindows = o_UsePhaseWindows;
        }
        if (o_HasUseEarlyPhase1) {
            GlobalParamsForCOP->OverrideUseEarlyPhase1 = 1;
            GlobalParamsForCOP->parseasmUseEarlyPhase1 = o_UseEarlyPhase1;
        }
        if (o_UseVLIW) {
            GlobalParamsForCOP->ParseasmUseVLIW = o_UseVLIW;
        }
        GlobalParamsForCOP->ParseasmDirect2IR = o_Direct2IR;
        GlobalParamsForCOP->regStressFlags = o_RegStressFlags;

#if HAS_VLIW_FILENAME > 0
        if (o_VLIWfileName != NULL) {
            GlobalParamsForCOP->ParseasmVLIWfileName = o_VLIWfileName;
        }
#endif
        if (o_mdesFileName != NULL) {
            GlobalParamsForCOP->ParseasmMdesFileName = o_mdesFileName;
        }
        if (o_HasEmitLwir) {
            GlobalParamsForCOP->OverrideEmitLwir = o_HasEmitLwir;
            GlobalParamsForCOP->parseasmEmitLwir = o_EmitLwir;
        }
#if (HAS_TASKS > 0)
        if (o_Tasks != NULL) {
            GlobalParamsForCOP->tasks = o_Tasks;
        }
        GlobalParamsForCOP->parseasmTextureReduction = o_TextureReduction;
        GlobalParamsForCOP->OverrideTextureReduction = o_HasTextureReduction;
#endif
        GlobalParamsForCOP->parseasmMaxWarpsPerTile = o_MaxWarpsPerTile;
        GlobalParamsForCOP->OverrideMaxWarpsPerTile = o_HasMaxWarpsPerTile;
        GlobalParamsForCOP->ParseasmExperimentalA = o_ExperimentalA;
        GlobalParamsForCOP->ParseasmExperimentalB = o_ExperimentalB;
        GlobalParamsForCOP->ParseasmExperimentalC = o_ExperimentalC;
        GlobalParamsForCOP->ParseasmExperimentalD = o_ExperimentalD;
        if (o_ShowInstFreq) {
            GlobalParamsForCOP->ParseasmShowInstFreq = o_ShowInstFreq;
        }
        if (o_HasArchVariant) {
            GlobalParamsForCOP->OverrideArchVariant = 1;
            GlobalParamsForCOP->parseasmArchVariant = o_ArchVariant;
        }
        if (o_IsWPOSOne) {
            GlobalParamsForCOP->OverrideIsWPOSOne = 1;
            GlobalParamsForCOP->parseasmIsWPOSOne = o_IsWPOSOne;
        }
        if (o_HasDoUniformAtomExpand) {
            GlobalParamsForCOP->OverrideDoUniformAtomExpand = 1;
            GlobalParamsForCOP->parseasmDoUniformAtomExpand = o_DoUniformAtomExpand;
        }
        if (o_HasDoFastUniformAtomExpand) {
            GlobalParamsForCOP->OverrideDoFastUniformAtomExpand = 1;
            GlobalParamsForCOP->parseasmDoFastUniformAtomExpand = o_DoFastUniformAtomExpand;
        }
        if (o_HasDoUniformAtomExpandUsingSHFL) {
            GlobalParamsForCOP->OverrideDoUniformAtomExpandUsingSHFL = 1;
            GlobalParamsForCOP->parseasmDoUniformAtomExpandUsingSHFL = o_DoUniformAtomExpandUsingSHFL;
        }
        if (o_HasDoVectorAtomExpand) {
            GlobalParamsForCOP->OverrideDoVectorAtomExpand = 1;
            GlobalParamsForCOP->parseasmDoVectorAtomExpand = o_DoVectorAtomExpand;
        }
        if (o_HasEnableZeroCoverageKill) {
            GlobalParamsForCOP->OverrideEnableZeroCoverageKill = 1;
            GlobalParamsForCOP->parseasmEnableZeroCoverageKill= o_EnableZeroCoverageKill;
        }
        if (o_HasCanUseGlobalScratch) {
            GlobalParamsForCOP->OverrideCanUseGlobalScratch = 1;
            GlobalParamsForCOP->parseasmCanUseGlobalScratch = o_CanUseGlobalScratch;
        }
        if (o_HasSharedMemorySize) {
            GlobalParamsForCOP->OverrideSharedMemorySize = 1;
            GlobalParamsForCOP->parseasmSharedMemorySize = o_SharedMemorySize;
        }
        if (o_HasSMemScratchPerWarp) {
            GlobalParamsForCOP->OverrideSMemScratchPerWarp = 1;
            GlobalParamsForCOP->parseasmSMemScratchPerWarp = o_SMemScratchPerWarp;
        }
        if (o_HasGMemScratchBase) {
            GlobalParamsForCOP->OverrideGMemScratchBase = 1;
            GlobalParamsForCOP->parseasmGMemScratchBase = o_GMemScratchBase;
        }
        if (o_HasGMemScratchConstBank) {
            GlobalParamsForCOP->OverrideGMemScratchConstBank = 1;
            GlobalParamsForCOP->parseasmGMemScratchConstBank = o_GMemScratchConstBank;
        }
        if (o_HasGMemScratchConstOffset) {
            GlobalParamsForCOP->OverrideGMemScratchConstOffset = 1;
            GlobalParamsForCOP->parseasmGMemScratchConstOffset = o_GMemScratchConstOffset;
        }
        if (o_HasNoFloatMAD) {
            GlobalParamsForCOP->OverrideNoFloatMAD = 1;
            GlobalParamsForCOP->parseasmNoFloatMAD = o_NoFloatMAD;
        }
        if (o_HasForceFMZ) {
            GlobalParamsForCOP->OverrideForceFMZ = 1;
            GlobalParamsForCOP->parseasmForceFMZ = o_ForceFMZ;
        }
        if (o_HasDisableFTZ) {
            GlobalParamsForCOP->OverrideDisableFTZ = 1;
            GlobalParamsForCOP->parseasmDisableFTZ = o_DisableFTZ;
        }
        if (o_HasSFUFloatMUL) {
            GlobalParamsForCOP->OverrideSFUFloatMUL = 1;
            GlobalParamsForCOP->parseasmSFUFloatMUL = o_SFUFloatMUL;
        }
        if (o_HasTxdBatchSize) {
            GlobalParamsForCOP->OverrideTxdBatchSize = 1;
            GlobalParamsForCOP->parseasmTxdBatchSize = o_TxdBatchSize;
        }
        if (o_HasAddDummyCRead) {
            GlobalParamsForCOP->OverrideAddDummyCRead = 1;
            GlobalParamsForCOP->AddDummyCRead = o_AddDummyCRead;
        }
        if (o_CanIgnoreNan != -1) {
            GlobalParamsForCOP->OverrideCanIgnoreNan = 1;
            GlobalParamsForCOP->parseasmCanIgnoreNan = o_CanIgnoreNan;
        }
        if (o_CanIgnoreInf != -1) {
            GlobalParamsForCOP->OverrideCanIgnoreInf = 1;
            GlobalParamsForCOP->parseasmCanIgnoreInf = o_CanIgnoreInf;
        }
        if (o_CanIgnoreSignedZero != -1) {
            GlobalParamsForCOP->OverrideCanIgnoreSignedZero = 1;
            GlobalParamsForCOP->parseasmCanIgnoreSignedZero = o_CanIgnoreSignedZero;
        }
        if (o_CanReorderFloat != -1) {
            GlobalParamsForCOP->OverrideCanReorderFloat = 1;
            GlobalParamsForCOP->parseasmCanReorderFloat = o_CanReorderFloat;
        }
        if (o_CanReorderHalf != -1) {
            GlobalParamsForCOP->OverrideCanReorderHalf = 1;
            GlobalParamsForCOP->parseasmCanReorderHalf = o_CanReorderHalf;
        }
        if (o_SpelwlativelyHoistTex != -1) {
            GlobalParamsForCOP->OverrideSpelwlativelyHoistTex = 1;
            GlobalParamsForCOP->parseasmSpelwlativelyHoistTex = o_SpelwlativelyHoistTex;
        }
        if (o_usesTaskShader != -1) {
            GlobalParamsForCOP->OverrideUsesTaskShader = 1;
            GlobalParamsForCOP->parseasmUsesTaskShader = o_usesTaskShader;
        }
        if (o_ProfilerLevel != -1) {
            GlobalParamsForCOP->OverrideProfilerLevel = 1;
            GlobalParamsForCOP->parseasmProfilerLevel = o_ProfilerLevel;
        }
        GlobalParamsForCOP->parseasmRRegBankSize = o_RRegBankSize;
        if (o_HasDeadOutputMask) {
            GlobalParamsForCOP->OverrideDeadOutputMask = 1;
            memcpy(GlobalParamsForCOP->parseasmDeadOutputMask, o_DeadOutputMask, sizeof(o_DeadOutputMask));
        }
        GlobalParamsForCOP->FlatUpperAttributes = o_FlatUpperAttributes;
        GlobalParamsForCOP->upperAttributeMask = o_UpperAttributeMask;
        GlobalParamsForCOP->mrtMask = o_MrtMask;
        GlobalParamsForCOP->OverrideMrtMask = o_OverrideMrtMask;
        GlobalParamsForCOP->ForceDX10SAT = o_ForceDX10SAT;
        GlobalParamsForCOP->ForceDX10AddressInRange = o_ForceDX10AddressInRange;
        GlobalParamsForCOP->SSAAttribReg = o_SSAAttribReg;
        GlobalParamsForCOP->maxRRegsAllowed = o_maxRRegCount;
        GlobalParamsForCOP->parseasmMaxRRegsAllowed = o_maxRRegCount;
        GlobalParamsForCOP->parseasmSchedRegTarget = o_schedRegTarget;
        GlobalParamsForCOP->parseasmHintControl = o_hintControl;
        GlobalParamsForCOP->CompileAsFunction = o_CompileAsFunction;
        GlobalParamsForCOP->OverrideMaxInstInBasicBlock = o_MaxInstInBasicBlock;
        GlobalParamsForCOP->OverrideOptLevel = o_HasOptLevel;
        GlobalParamsForCOP->parseasmOptLevel = o_OptLevel;
        GlobalParamsForCOP->vtxA = o_vtxA;
        GlobalParamsForCOP->OverrideVtxA = o_vtxA;
        GlobalParamsForCOP->vtxB = o_vtxB;
        GlobalParamsForCOP->OverrideVtxB = o_vtxB;
        GlobalParamsForCOP->parseasmKnobsString = o_KnobsString;      
        GlobalParamsForCOP->parseasmKnobsFileName = o_KnobsFileName;
        GlobalParamsForCOP->parseasmOriControl = o_OriControl;
        GlobalParamsForCOP->parseasmDebugOutputMask = o_debugOutputMask;
        GlobalParamsForCOP->ppCgBinaryData = ppCgBinaryData;
        GlobalParamsForCOP->pCgBinarySize = pCgBinarySize;
        GlobalParamsForCOP->parseasmDumpPerfStats = o_DumpPerfStats;
        GlobalParamsForCOP->parseasmDumpPgmLatInfo = o_DumpPgmLatInfo;
        GlobalParamsForCOP->parseasmUnknownLoopIttr = o_unknownLoopIttr;
        GlobalParamsForCOP->parseasmNumberOfSM = o_numberOfSM;
        GlobalParamsForCOP->parseasmNumWarpsLaunched = o_numWarpsLaunched;
#if defined(LW_PARSEASM)
        GlobalParamsForCOP->profileStruct = o_ProfileStruct;
        if (o_ProfileStruct) {
            GlobalParamsForCOP->FillParamsStructure = o_ProfileStruct->FillParamsStructure;
        } else {
            GlobalParamsForCOP->FillParamsStructure = NULL;
        }
#endif // LW_PARSEASM

        SetCOPBaseArgs(pBaseArgs, &outInsts, &outStart, GlobalParamsForCOP);

        // Determine the size
        switch (parseMethod) {
        case PARSE_D3D10_SM4X_BIN:
        case PARSE_D3D_VSBIN:
        case PARSE_D3D_PSBIN:
            // This is a binary file
            len = lSize;
          break;
          
        default:
            // All others are strings
            len = (unsigned)strlen((char *) lRawInput);
           break;
        }

        switch (parseMethod) {
        case PARSE_LWVP_TEXT:
        case PARSE_LWGP_TEXT:
        case PARSE_LWFP_TEXT:
        case PARSE_LWHP_TEXT:
        case PARSE_LWTP_TEXT:
        case PARSE_LWCP_TEXT:
        case PARSE_LWMP_TEXT:
        case PARSE_LWMTP_TEXT:
        case PARSE_ARBVP_TEXT:
        case PARSE_ARBFP_TEXT:
        {
            char *chipVersionGL = chipVersion;
            int glTarget;
            if (o_Quadro) {
                chipVersionGL = (char*) malloc(strlen(chipVersion) + 3);
                strcpy(chipVersionGL, chipVersion);
                strcat(chipVersionGL, "gl");
            }
            switch (parseMethod) {
            case PARSE_LWVP_TEXT:   glTarget = GL_VERTEX_PROGRAM_LW; break;
            case PARSE_LWGP_TEXT:   glTarget = GL_GEOMETRY_PROGRAM_LW; break;
            case PARSE_LWFP_TEXT:   glTarget = GL_FRAGMENT_PROGRAM_LW; break;
            case PARSE_LWHP_TEXT:   glTarget = GL_TESS_CONTROL_PROGRAM_LW; break;
            case PARSE_LWTP_TEXT:   glTarget = GL_TESS_EVALUATION_PROGRAM_LW; break;
            case PARSE_LWCP_TEXT:   glTarget = GL_COMPUTE_PROGRAM_LW; break;
            case PARSE_LWMP_TEXT:   glTarget = GL_MESH_PROGRAM_LW; break;
            case PARSE_LWMTP_TEXT:  glTarget = GL_TASK_PROGRAM_LW; break;
            case PARSE_ARBVP_TEXT:  glTarget = GL_VERTEX_PROGRAM_ARB; break;
            case PARSE_ARBFP_TEXT:  glTarget = GL_FRAGMENT_PROGRAM_ARB; break;
            default:
                assert(0);
                break;
            }
            GlobalParamsForCOP->IsOGL = 1;
            LwU32 dwFlags = 0;
            if (o_vulkan) {
                dwFlags |= OGLPFLAG_VULKAN;
            }
            if (o_glinternal) {
                dwFlags |= OGLPFLAG_GLINTERNAL;
            }
            retVal = oglParseasm_Parse(pBaseArgs, lRawInput, len, chipVersionGL, glTarget,
                                       outFile, NULL, dwFlags);
            if (o_Quadro) {
                free(chipVersionGL);
            }
            break;
        }
        case PARSE_LWINST_TEXT:
            retVal = ParseShaderAsm(pBaseArgs, (char *) lRawInput);
            break;
#if defined(NEED_PARSESHD)
        case PARSE_SHD:
            retVal = ParseSHD(pBaseArgs, inpFileName);
            break;
#endif // NEED_PARSESHD
#if defined(ALLOW_D3D_BUILD)
        case PARSE_D3D_PS1X:
        case PARSE_D3D_PS14:
        case PARSE_D3D_PS2X:
        case PARSE_D3D_PS3X:
        case PARSE_D3D_PSBIN:
            {
                int flags = D3DCPFLAG_OPTIMIZE | D3DCPFLAG_PIXEL;
                
                if (o_ExportConstants)
                    flags |= D3DCPFLAG_EXPORTCONSTANTS;
                else
                    flags |= D3DCPFLAG_EVALCONSTANTS;
                    
                if(o_D3DModShaderAll16)
                  flags |= D3DCPFLAG_DEMOTE_FP16;
                else if(o_D3DModShaderAll32)
                  flags |= D3DCPFLAG_PROMOTE_FP32;

                if (o_Binary)
                    flags |= D3DCPFLAG_BINARY_OUT;
                if (GlobalParamsForCOP->UseTextFileOutput || o_codeGenRuns > 0)
                    flags |= D3DCPFLAG_QUIET;
                if (o_Hash)
                    flags |= D3DCPFLAG_PRINTHASH;
                if (parseMethod == PARSE_D3D_PSBIN)
                    flags |= D3DCPFLAG_PREASSEMBLED;
                if (o_dumpTokenStream)
                    flags |= D3DCPFLAG_DUMPTOKENSTREAM;

                switch (o_DXVer) {
                case 9:
                    flags |= D3DCPFLAG_VERSION_DX9;
                    break;

                case 8:
                default:
                    flags |= D3DCPFLAG_VERSION_DX8;
                    break;
                }

                D3DCompileProgram(chipVersion, lRawInput, len, flags, outFile);
            }
            break;
        case PARSE_D3D_VS1X:
        case PARSE_D3D_VS2X:
        case PARSE_D3D_VS3X:
        case PARSE_D3D_VSBIN:
            {
                int flags = D3DCPFLAG_OPTIMIZE | D3DCPFLAG_VERTEX;
                
                if (o_Binary)
                    flags |= D3DCPFLAG_BINARY_OUT;
                if (GlobalParamsForCOP->UseTextFileOutput || o_codeGenRuns > 0)
                    flags |= D3DCPFLAG_QUIET;
                if (o_Hash)
                    flags |= D3DCPFLAG_PRINTHASH;
                if (parseMethod == PARSE_D3D_VSBIN)
                    flags |= D3DCPFLAG_PREASSEMBLED;
                if (o_dumpTokenStream)
                    flags |= D3DCPFLAG_DUMPTOKENSTREAM;

                switch (o_DXVer) {
                case 9:
                    flags |= D3DCPFLAG_VERSION_DX9;
                    break;

                case 8:
                default:
                    flags |= D3DCPFLAG_VERSION_DX8;
                    break;
                }

                // There's no way to return this to parseasm on this path!
                o_ppUCode = ppUCode;
                D3DCompileVertexProgram(chipVersion, lRawInput, len, flags, outFile);
            }
            break;
#endif // ALLOW_D3D_BUILD

        case PARSE_D3D10_SM4X_BIN:
        case PARSE_D3D10_SM4X:
#if defined(LW_LDDM_DDK_BUILD)
            compileD3D10ShaderEx(pBaseArgs, lRawInput, lSize, (parseMethod == PARSE_D3D10_SM4X_BIN), chipVersion, ppUCode, o_d3dArgs);
#else // defined(LW_LDDM_DDK_BUILD)
            assert(0 && "No support for SM4x compiled!"); /////
#endif
            break;
        }

        if (retVal == 0) {
            if (pBaseArgs->inInsts != NULL) {
                if (o_verbosityLevel) {
                    printf("Input was parsed successfully and colwerted to internal structures..\n");
                    fpPrintLwInst(pBaseArgs->inInsts, stdout,
                                pBaseArgs->eOutputMode != omH0 && pBaseArgs->eOutputMode != omCombiners,
                                pBaseArgs->eProf);
                }

                // Fix startinst if the corresponding option is specified to parseasm
                if (pBaseArgs->inInsts == pBaseArgs->inStart && o_StartInst != 0) {
                    // Assume that the start inst was not set correctly.
                    lwInst *lStart = pBaseArgs->inInsts;
                    for (ii = 0; ii < o_StartInst && lStart; ++ii) {
                        lStart = lStart->next;
                    }
                    assert(ii >= o_StartInst || (!"Invalid starting instruction number"));
                    pBaseArgs->inStart = lStart;
                }

                // optimize program in pBaseArgs->inInsts, leaving result in pBaseArgs->outInsts
                retVal = optimizeProgram(pBaseArgs, outFile, ppUCode);
                if (GlobalParamsForCOP->IsOGL && !(o_eProf & prAR20)) {
                    // Print out addition OpenGL program information, if requested on the
                    // command line.
                    if (o_oglInfo) {
                        oglParseasm_PrintOGLInfo(outFile);
                    }

                    if (o_WriteToCompiledShaderData || (o_Binary && outFile)) {
                        oglParseasm_MakeGLprogramFromLWInstructions(*(pBaseArgs->outInsts), GL_TRUE);
                        
                        if(o_WriteToCompiledShaderData) {
                            unsigned char *pucGPUCode = NULL;
                            int nGPUCodeSize;
                              
                            oglParseasm_GetUCode(&pucGPUCode, &nGPUCodeSize);
                            
                            if (pucGPUCode) {
                                compiledShaderSize = nGPUCodeSize;
                                pCompiledShaderData = malloc(sizeof(unsigned char) * compiledShaderSize);
                                memset(pCompiledShaderData, 0, compiledShaderSize);
                                memcpy(pCompiledShaderData, pucGPUCode, compiledShaderSize);
                            }
                        }

                        if (o_Binary && outFile) {
                            oglParseasm_FwriteProgram(outFile);
                        }
                    }
                }
            } else {
                // error
            }

            // Free the optimized program.
            FreeProgram(*(pBaseArgs->outInsts), &pBaseArgs->memParams);
        }
        if (o_codeGenRuns > 0) {
            stopTime = (double) lGetTick(); 
            if ((stopTime - startTime) > MAX_PROF_TIME) {
                // Inc ii to indicate number of compiles.
                ii++;
                break;
            }
        }
    }

    if (o_codeGenRuns > 0) {
        double timePerIteration;
        timePerIteration = (double) (stopTime - startTime) / FREQUENCY;
        timePerIteration = timePerIteration / (double)ii;
        printf("\nTime per iter for %d iterations: %2.5f sec\n", ii, timePerIteration);
    }
    
    
    // Free the input program.
    if (pBaseArgs->inInsts)
        FreeProgram(pBaseArgs->inInsts, &pBaseArgs->memParams);
    return retVal;
} // compileOnceMemoryBuffer

/*
 * SetDefaultProfile() - If no profile is specified, infer one from the file extension:
 *
 */

void SetDefaultProfile(const char *fname)
{
    const char *ext;

    if (fname == NULL)
        return;
    if (o_Profile != NULL)
        return;
    ext = &fname[lwStrLen(fname)];
    while (--ext > fname) {
        if (*ext == '.') {
                   if (!lwStrCmp(ext, ".vp1") ||
                       !lwStrCmp(ext, ".vp2") ||
                       !lwStrCmp(ext, ".vp4") ||
                       !lwStrCmp(ext, ".avp1") ||
                       !lwStrCmp(ext, ".vs1") ||
                       !lwStrCmp(ext, ".vs2") ||
                       !lwStrCmp(ext, ".vs3") ||
                       !lwStrCmp(ext, ".vs4") ||
                       !lwStrCmp(ext, ".vs5") ||
                       !lwStrCmp(ext, ".vp4t"))
            {
                o_eProf = prFermiVP;
                chipVersion = "gf100";
                o_Profile = "vpf";
            } else if (!lwStrCmp(ext, ".fp1") ||
                       !lwStrCmp(ext, ".fp4") ||
                       !lwStrCmp(ext, ".fp5") ||
                       !lwStrCmp(ext, ".afp1") ||
                       !lwStrCmp(ext, ".ps1") ||
                       !lwStrCmp(ext, ".ps2") ||
                       !lwStrCmp(ext, ".ps3") ||
                       !lwStrCmp(ext, ".ps4") ||
                       !lwStrCmp(ext, ".ps5") ||
                       !lwStrCmp(ext, ".fp4t"))
            {
                o_eProf = prFermiFP;
                chipVersion = "gf100";
                o_Profile = "fpf";
            } else if (!lwStrCmp(ext, ".gp4") ||
                       !lwStrCmp(ext, ".gs5"))
            {
                o_eProf = prFermiGP;
                chipVersion = "gf100";
                o_Profile = "gpf";
            } else if (!lwStrCmp(ext, ".hp5") ||
                       !lwStrCmp(ext, ".hs5"))
            {
                o_eProf = prFermiHP;
                chipVersion = "gf100";
                o_Profile = "hpf";
            } else if (!lwStrCmp(ext, ".tp5")) {
                o_eProf = prFermiTP;
                chipVersion = "gf100";
                o_Profile = "tpf";
            } else if (!lwStrCmp(ext, ".cs5")) {
                o_eProf = prFermiCP;
                chipVersion = "gf100";
                o_Profile = "cpf";
            }
            break;
        }
    }
} // SetDefaultProfile

/*
 * SetDefaults() - If no input format is supplied, infer one from the file extension:
 *
 * Extension:  Profile:  Format:
 *   .vp1        vpf      vp1x
 *   .vp2        vpf      vp2x
 *   .vp4        vpf      vp4x
 *   .avp1       vpf      arbvp1
 *   .vs1        vpf      vs1x
 *   .vs2        vpf      vs2x
 *   .vs3        vpf      vs3x
 *   .vs4        vpf      sm4x
 *   .vs5        vpf      sm4x
 *   .vp4t       vpf      fp40text
 *   .fp1        fpf      fp1x
 *   .fp4        fpf      fp4x
 *   .fp5        fpf      fp4x
 *   .afp1       fpf      arbfp1
 *   .ps1        fpf      ps1x
 *   .ps2        fpf      ps2x
 *   .ps3        fpf      ps3x
 *   .ps4        fpf      sm4x
 *   .ps5        fpf      sm4x
 *   .fp4t       fpf      fp40text
 *   .gp4        gpf      gp4x
 *   .gs5        gpf      sm4x
 *   .hp5        hpf      hp5x
 *   .hs5        hpf      sm4x
 *   .tp5        tpf      tp5x
 *   .cs5        cpf      sm4x
 */

void SetDefaults(void)
{
    const char *ext;

    if (o_InputFileName == NULL)
        return;
    if (o_InputFormat != NULL)
        return;
    ext = &o_InputFileName[lwStrLen(o_InputFileName)];
    while (--ext > o_InputFileName) {
        if (*ext == '.') {
            if (!lwStrCmp(ext, ".vp1")) {
                o_InputFormat = "vp1x";
            } else if (!lwStrCmp(ext, ".vp2")) {
                o_InputFormat = "vp2x";
            } else if (!lwStrCmp(ext, ".vp4")) {
                o_InputFormat = "vp4x";
            } else if (!lwStrCmp(ext, ".avp1")) {
                o_InputFormat = "arbvp1";
            } else if (!lwStrCmp(ext, ".vs1")) {
                o_InputFormat = "vs1x";
            } else if (!lwStrCmp(ext, ".vs2")) {
                o_InputFormat = "vs2x";
            } else if (!lwStrCmp(ext, ".vs3")) {
                o_InputFormat = "vs3x";
            } else if (!lwStrCmp(ext, ".vs4")) {
                o_InputFormat = "sm4x";
            } else if (!lwStrCmp(ext, ".vs5")) {
                o_InputFormat = "sm4x";
            } else if (!lwStrCmp(ext, ".fp1")) {
                o_InputFormat = "fp1x";
            } else if (!lwStrCmp(ext, ".fp4")) {
                o_InputFormat = "fp4x";
            } else if (!lwStrCmp(ext, ".fp5")) {
                o_InputFormat = "fp4x";
            } else if (!lwStrCmp(ext, ".afp1")) {
                o_InputFormat = "arbfp1";
            } else if (!lwStrCmp(ext, ".ps1")) {
                o_InputFormat = "ps1x";
            } else if (!lwStrCmp(ext, ".ps2")) {
                o_InputFormat = "ps2x";
            } else if (!lwStrCmp(ext, ".ps3")) {
                o_InputFormat = "ps3x";
            } else if (!lwStrCmp(ext, ".ps4")) {
                o_InputFormat = "sm4x";
            } else if (!lwStrCmp(ext, ".ps5")) {
                o_InputFormat = "sm4x";
            } else if (!lwStrCmp(ext, ".fp4t")) {
                o_InputFormat = "fp40text";
            } else if (!lwStrCmp(ext, ".vp4t")) {
                o_InputFormat = "fp40text";
            } else if (!lwStrCmp(ext, ".gp4")) {
                o_InputFormat = "gp4x";
            } else if (!lwStrCmp(ext, ".cs5")) {
                o_InputFormat = "sm4x";
            } else if (!lwStrCmp(ext, ".gs5")) {
                o_InputFormat = "sm4x";
            } else if (!lwStrCmp(ext, ".hp5")) {
                o_InputFormat = "hp5x";
            } else if (!lwStrCmp(ext, ".tp5")) {
                o_InputFormat = "tp5x";
            } else if (!lwStrCmp(ext, ".hs5")) {
                o_InputFormat = "sm4x";
            } else if (!lwStrCmp(ext, ".cs5")) {
                o_InputFormat = "sm4x";
            }
            break;
        }
    }
} // SetDefaults

/*
 * ProcessProfile()
 *
 */

int ProcessProfile(void)
{
    if (!_stricmp(o_Profile, "vpf")) {
        o_eProf = prFermiVP;
        chipVersion = "gf100";
    } else if (!_stricmp(o_Profile, "fpf")) {
        o_eProf = prFermiFP;
        chipVersion = "gf100";
    } else if (!_stricmp(o_Profile, "gpf")) {
        o_eProf = prFermiGP;
        chipVersion = "gf100";
    } else if (!_stricmp(o_Profile, "cpf")) {
        o_eProf = prFermiCP;
        chipVersion = "gf100";
    } else if (!_stricmp(o_Profile, "hpf")) {
        o_eProf = prFermiHP;
        chipVersion = "gf100";
    } else if (!_stricmp(o_Profile, "tpf")) {
        o_eProf = prFermiTP;
        chipVersion = "gf100";
    } else if (!_stricmp(o_Profile, "mpf")) {
        o_eProf = prTuringMP;
        chipVersion = "gf100";
    } else if (!_stricmp(o_Profile, "mtpf")) {
        o_eProf = prTuringMTP;
        chipVersion = "gf100";
    } else {
        return(1);
    }
    if (o_HasArchVariant) {
        lwiOptimizationProfile neededProfile;
        switch (o_ArchVariant) {
        case COP_ARCH_VARIANT_FERMI_SM2_0:
            neededProfile = prFermi;
            chipVersion = "gf100";
            break;
        case COP_ARCH_VARIANT_FERMI_SM2_1:
            neededProfile = prFermi;
            chipVersion = "gf104";
            break;
        case COP_ARCH_VARIANT_KEPLER_SM3_0:
            neededProfile = prFermi;    // Kepler uses Fermi profiles
            chipVersion = "gk104";
            break;
        case COP_ARCH_VARIANT_KEPLER_SM3_2:
            neededProfile = prFermi;    // Kepler uses Fermi profiles
            chipVersion = "gk20a";
            break;
        case COP_ARCH_VARIANT_KEPLER_SM4_0:
            neededProfile = prFermi;    // Kepler uses Fermi profiles
            chipVersion = "gk110";
            break;
        case COP_ARCH_VARIANT_MAXWELL_SM5_0:
            neededProfile = prFermi;    // Maxwell uses Fermi profiles
            chipVersion = "gm100";
            break;
        case COP_ARCH_VARIANT_MAXWELL_SM5_2:
            neededProfile = prFermi;    // Maxwell uses Fermi profiles
            chipVersion = "gm200";
            break;
        case COP_ARCH_VARIANT_MAXWELL_SM5_3:
            neededProfile = prFermi;    // Maxwell uses Fermi profiles
            chipVersion = "gm20y";
            break;
        case COP_ARCH_VARIANT_PASCAL_SM6_0:
            neededProfile = prFermi;    // Pascal uses Fermi profiles
            chipVersion = "gp100";      
            break;
        case COP_ARCH_VARIANT_PASCAL_SM6_1:
            neededProfile = prFermi;    // Pascal uses Fermi profiles
            chipVersion = "gp107";      // sm61 -> gp107/8 
            break;
        case COP_ARCH_VARIANT_PASCAL_SM6_2:
            neededProfile = prFermi;    // Pascal uses Fermi profiles
            chipVersion = "gp10b";      // sm62 -> gp10b (cheetah) 
            break;
        case COP_ARCH_VARIANT_VOLTA_SM7_0:
            neededProfile = prFermi;    // Volta uses Fermi profiles
            chipVersion = "gv100";
            break;
        case COP_ARCH_VARIANT_VOLTA_SM7_2:
            neededProfile = prFermi;    // Volta uses Fermi profiles
            chipVersion = "gv11b";
            break;
        case COP_ARCH_VARIANT_TURING_SM7_3:
            neededProfile = prFermi;    // Turing uses Fermi profiles
            chipVersion = "tu106";      // sm73 -> tu106/7
            break;
        case COP_ARCH_VARIANT_TURING_SM7_5:
            neededProfile = prFermi;    // Turing uses Fermi profiles
            chipVersion = "tu102";      // sm75 -> tu101/2/4/5
            break;
#if LWCFG(GLOBAL_ARCH_AMPERE)
        case COP_ARCH_VARIANT_AMPERE_SM8_2:
            neededProfile = prFermi;    // Ampere uses Fermi profiles
            chipVersion = "ga100";      // sm82 -> ga100
            break;
        case COP_ARCH_VARIANT_AMPERE_SM8_6:
            neededProfile = prFermi;    // Ampere uses Fermi profiles
            chipVersion = "ga102";      // sm86 -> ga102/3/4/6/7
            break;
#if LWCFG(GLOBAL_GPU_IMPL_GA10B)
        case COP_ARCH_VARIANT_AMPERE_SM8_7:
            neededProfile = prFermi;    // Ampere uses Fermi profiles
            chipVersion = "ga10b";      // sm87 -> ga10b
            break;
#endif // LWCFG(GLOBAL_GPU_IMPL_GA10B)
#if LWCFG(GLOBAL_GPU_IMPL_GA10F)
        case COP_ARCH_VARIANT_AMPERE_SM8_8:
            neededProfile = prFermi;    // Ampere uses Fermi profiles
            chipVersion = "ga10d";      // sm88 -> ga10d
            break;
#endif // LWCFG(GLOBAL_GPU_IMPL_GA10F)
#if LWCFG(GLOBAL_ARCH_ADA)
        case COP_ARCH_VARIANT_ADA_SM8_9:
            neededProfile = prFermi;    // Ada uses Fermi profiles
            chipVersion = "ad102";      // sm89 -> ad102
            break;
#endif // LWCFG(GLOBAL_ARCH_ADA)
#if LWCFG(GLOBAL_ARCH_HOPPER)
        case COP_ARCH_VARIANT_HOPPER_SM9_0:
            neededProfile = prFermi;    // Hopper uses Fermi profiles
            chipVersion = "gh100";      // sm90 -> gh100
            break;
#endif // LWCFG(GLOBAL_ARCH_HOPPER)
#endif // LWCFG(GLOBAL_ARCH_AMPERE)
        default:
            fprintf(stderr, "Error:  Unknown architecture variant.\n");
            lCallExit(1);
            break;
        }
        if (0 == (o_eProf & neededProfile)) {
            fprintf(stderr, "Error:  Architecture variant not compatible "
                    "with profile.\n");
            lCallExit(1);
        }
    }
        
    return(0);
} // ProcessProfile

/*
 * ProcessInputFormat()
 *
 */

int ProcessInputFormat() 
{
    if (o_InputFormat == NULL) {
        return 1;
    } else if (!strcmp(o_InputFormat, "vp1x") ||
        !strcmp(o_InputFormat, "vp2x") ||
        !strcmp(o_InputFormat, "vp3x"))
    {
        parseMethod = PARSE_LWVP_TEXT;
    } else if (!strcmp(o_InputFormat, "gp4x")) { // !!LWgp4.0 or newer
        parseMethod = PARSE_LWGP_TEXT;
    } else if (!strcmp(o_InputFormat, "fp1_0") ||
               !strcmp(o_InputFormat, "fp1x")) {
        parseMethod = PARSE_LWFP_TEXT;
    } else if (!strcmp(o_InputFormat, "hp5x") || !strcmp(o_InputFormat, "tcp5x")) { // !!LWtcp5.0 or newer
        parseMethod = PARSE_LWHP_TEXT;
    } else if (!strcmp(o_InputFormat, "tp5x") || !strcmp(o_InputFormat, "tep5x")) { // !!LWtep5.0 or newer
        parseMethod = PARSE_LWTP_TEXT;
    } else if (!strcmp(o_InputFormat, "mp5x")) { // !!LWmp5.0 or newer
        parseMethod = PARSE_LWMP_TEXT;
    } else if (!strcmp(o_InputFormat, "mtp5x")) { // !!LWmtp5.0 or newer
        parseMethod = PARSE_LWMTP_TEXT;
    } else if (!strcmp(o_InputFormat, "cp1x")) {
        parseMethod = PARSE_LWCP_TEXT;
    } else if (!strcmp(o_InputFormat, "arbvp1") ||
               !strcmp(o_InputFormat, "vp4x")) { // !!LWvp4.0 uses ARB funcs
        parseMethod = PARSE_ARBVP_TEXT;
    } else if (!strcmp(o_InputFormat, "arbfp1") ||
               !strcmp(o_InputFormat, "fp4x")) { // !!LWfp4.0 uses ARB funcs
        parseMethod = PARSE_ARBFP_TEXT;
    } else if (!strcmp(o_InputFormat, "vs1x")) {
        parseMethod = PARSE_D3D_VS1X;
    } else if (!strcmp(o_InputFormat, "vs2x")) {
        parseMethod = PARSE_D3D_VS2X;
    } else if (!strcmp(o_InputFormat, "vs3x")) {
        parseMethod = PARSE_D3D_VS3X;
    } else if (!strcmp(o_InputFormat, "vsbin")) {
        parseMethod = PARSE_D3D_VSBIN;
    } else if (!strcmp(o_InputFormat, "ps1x")) {
        parseMethod = PARSE_D3D_PS1X;
    } else if (!strcmp(o_InputFormat, "ps14")) {
        parseMethod = PARSE_D3D_PS14;
    } else if (!strcmp(o_InputFormat, "ps2x")) {
        parseMethod = PARSE_D3D_PS2X;
    } else if (!strcmp(o_InputFormat, "ps3x")) {
        parseMethod = PARSE_D3D_PS3X;
    } else if (!strcmp(o_InputFormat, "psbin")) {
        parseMethod = PARSE_D3D_PSBIN;
#if defined(LW_LDDM_DDK_BUILD)
    } else if (!strcmp(o_InputFormat, "sm4x_bin")) {
        parseMethod = PARSE_D3D10_SM4X_BIN;
    } else if (!strcmp(o_InputFormat, "sm4x")) {
        parseMethod = PARSE_D3D10_SM4X;
#endif
    } else if (!strcmp(o_InputFormat, "gs40")) {
        parseMethod = PARSE_D3D_GS40;
    } else if (!strcmp(o_InputFormat, "fp40text") || !strcmp(o_InputFormat, "lwinst")) {
        parseMethod = PARSE_LWINST_TEXT;
    } else if (!strcmp(o_InputFormat, "lwir")) {
        parseMethod = PARSE_LWIR_TEXT;
    }
#if defined(NEED_PARSESHD)
    else if (!strcmp(o_InputFormat, "shd")) {
        parseMethod = PARSE_SHD;
    }
#endif // NEED_PARSESHD
    else {
      return(1);
    }

    return(0);
} // ProcessInputFormat

/*
 * SetupParseasmParams()
 *
 */

void SetupParseasmParams(char *pcProfile, char *pcInputFormat)
{
    o_Profile = pcProfile;
    o_InputFormat = pcInputFormat;
    
    o_lwinst = 1;
    
    ProcessProfile();
    ProcessInputFormat();
} // SetupParseasmParams

/*
 * ExtractUCodeSegment()
 *
 */

int ExtractUCodeSegment(LWuCode *pUCode, unsigned char **ppucBuffer, unsigned long *pulSize)
{
    LwU16 ii;
  
    for (ii = 0; ii < pUCode->header.numSections; ii++) {
        unsigned long ulBufferSize = pUCode->sHeader[ii].genericHeader.size;
        if (ulBufferSize > 0 &&
            (pUCode->sHeader[ii].genericHeader.kind == LWUC_SECTION_UCODE))
        {
            *ppucBuffer =
                ((unsigned char *) pUCode) + pUCode->sHeader[ii].genericHeader.offset.offset;
            *pulSize = ulBufferSize;
        return(1);
        }
    }
  return 0;
} // ExtractUCodeSegment

/*
 * GetLWuCodeLength()
 *
 */

int GetLWuCodeLength(LWuCode *pUCode)
{
  if(pUCode)
    return(pUCode->header.size);
  else
    return(0);
} // GetLWuCodeLength

#if defined(ENABLE_TRACE_CODE)
extern "C" void relprintf(const char *format, ...)
{
    va_list args;
    va_start(args, format);

    vprintf(format, args);

    va_end(args);
}

extern "C" int tprintf(const char *format, ...)
{
    int ret = 0;

    va_list args;
    va_start(args, format);

    ret = vprintf(format, args);

    va_end(args);

    return ret;
}

extern "C" int vtprintf(const char *format, va_list args)
{
    return vprintf(format, args);
}

extern "C" void tprintString(const char *str)
{
    tprintf("%s", str);
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// End of file parseasmlib.cpp /////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
