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

//
// parseasm.cpp
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h> /* Added for time measurement testing */
#include <assert.h>
#ifdef _MSC_VER
#include <crtdbg.h>
#include <windows.h>
#endif // _MSC_VER
#include <ctype.h>

#include "lwTypes.h"

#include "copi_inglobals.h"
#include "cop_parseasm.h"

#if defined(ALLOW_D3D_BUILD)
#include "parseD3D.h"
#if defined(LW_LDDM_DDK_BUILD)
#include "parseD3D10.h"
#endif
    //this is required for the moment as we transition the 4X code to the unified
    //vertex format
#endif // ALLOW_D3D_BUILD

#if defined(NEED_PARSESHD)
#include "parseSHD.h"
#endif // NEED_PARSESHD

#include "parseasmdecls.h"
#include "parseasmfun.h"

#include "parseasm_profileoption.h"

#if defined(AR20)
#include "copi_tegra_interface.h"
#include "parseasm_profileoption_ar20.h"
#endif // AR20

// Used for version info
#include "lwBldVer.h"
#include "lwver.h"
#if defined(TESLA)
#include "parseogl.h"
#endif

#ifndef ALEN
#define ALEN(X) ((int) (sizeof(X) / sizeof(X[0])))
#endif

extern char *commandLine;
extern char *chipVersion;

// Extern functions
extern void lCallExit(int returlwalue);
extern int compileOnceMemoryBuffer(unsigned char *lRawInput, int lSize, FILE *outFile, 
                                   LWuCode **ppUCode, void **ppCgBinaryData, 
                                   unsigned int *pCgBinarySize, int fBinary);
extern int ProcessProfile();
extern int ProcessInputFormat();
extern void SetDefaultProfile(const char *fName);
extern void SetDefaults(void);


// Forward Decls
static int lParseTexRemap(int argc, const char *argv[], int argi);
static int lParseOptLevel(int argc, const char *argv[], int argi);
static int lParseOptLevelN(int argc, const char *argv[], int argi);
static int lParseProfileOpt(int argc, const char *argv[], int argi);
static int lParseKnobCommand(int argc, const char *argv[], int argi);
static int lParseKnobsFileName(int argc, const char *argv[], int argi);

// ------------------------------ Utility functions ------------------------------- //

/*
 * fpEmitString() - Wrapper function to add strings to the output.
 *
 */

void fpEmitString(void *fArg, const char *fString)
{
    fprintf(fArg == NULL ? stderr : (FILE *) fArg, "%s", fString);
    // %s is intensional as fString can have format specifier e.g. %ctid
} // fpEmitString


// --------------------------------------------------------------------------------------------
/*
 * lGetFileSize() - Get File Size..
 *
 */

#ifdef WIN32
#define STATFUN _stat
#define STATSTRUCT struct _stat
#else
#define STATFUN stat
#define STATSTRUCT struct stat
#define stricmp strcasecmp
#endif


static int lGetFileSize(const char  *fPath)
{
    STATSTRUCT lStat;
    int result;

    result = STATFUN(fPath, &lStat);
    if (result != 0) {
        fprintf(stderr, "Error getting input file info.: %s\n", fPath);
        lCallExit(1);
    }
    return lStat.st_size;
} // lGetFileSize


/*
 * lReadFile() - Read a File in string..
 * Load a file into a string that's malloc'd
 * If filename is NULL, read from stdin
 */

static unsigned char *lReadFile(const char *fPath, int isBinary, unsigned int *fSize)
{
    //
    // Open file
    //
    FILE *lInpFile;
    unsigned int lSize;
    unsigned char *progstring;
    size_t proglen;

    if (!fPath)
        return NULL;

    lSize = lGetFileSize(fPath) + (isBinary ? 0 : 1); // add a '\n' at the end of the buffer

    lInpFile = fopen(fPath, "rb");
    if (!lInpFile) {
        fprintf(stderr, "Error opening input file: %s -- %s\n", fPath, strerror( errno) );
        lCallExit(1);
    }

    //
    // Allocate buffer
    //

    progstring = (unsigned char *) malloc(lSize+2);
    if (!progstring) {
        fprintf(stderr, "malloc() error for file buffer.  Size=%i\n", lSize + 1);
        lCallExit(1);
    }
    proglen = fread(progstring, 1, lSize, lInpFile);
    if (proglen < 1) {
        fprintf(stderr, "Error reading input file, or it was empty\n");
        lCallExit(1);
    }
    if (!isBinary) {
        progstring[proglen] = '\n';
        progstring[proglen+1] = '\0';
    }
    if (fclose(lInpFile) != 0) {
        fprintf(stderr, "Error closing input file: %s -- %s\n", fPath, strerror( errno) );
        lCallExit(1);
    }

    *fSize = lSize;
    return progstring;
} // lReadFile


// ------------------------------ Command line options ------------------------------- //

typedef enum {
    OT_UNDEF,
    OT_INT,
    OT_FLOAT,
    OT_BOOL,    //  0 / 1
    OT_HEX,
    OT_NOOPT,
    OT_STRING,
    OT_FUN,     // complex option parsed by function pointer arg
} OptionType;

typedef struct {
    char       *name;
    OptionType  type;
    const void *val;
    char       *helpMessage;
} OptionStruct;

typedef int (*ProcessOptionFun)(int argc, const char *argv[], int argi);

#define OPTNL "\n        "
static OptionStruct optionArr[] = {
    {"h", OT_NOOPT, &o_Help, "This message."},
    {"help", OT_NOOPT, &o_Help, "This message."},
    // input args
    {"i", OT_STRING, &o_InputFileName, "Name of input file."},
    {"il", OT_STRING, &o_InputListFileName, "Name of file that contains list of input files."},
    {"inputlist", OT_STRING, &o_InputListFileName, "Name of file that contains list of input files."},
    {"if", OT_STRING, &o_InputFormat, "Input format."},
    {"informat", OT_STRING, &o_InputFormat, "Input format. Supported input formats are:"
     OPTNL "cp1x - OpenGL LW_compute_program_1 assembly format"
     OPTNL "vp1x - OpenGL LW_vertex_program_1 assembly format"
     OPTNL "vp2x - OpenGL LW_vertex_program_2 assembly format"
     OPTNL "vp3x - OpenGL LW_vertex_program_3 assembly format"
     OPTNL "hp5x - OpenGL LW_tessellation_control_program_5 (!!LWtcp5.0) assembly format"
     OPTNL "tp5x - OpenGL LW_tessellation_evaluation_program_5 (!!LWtep5.0) assembly format"
    OPTNL "mp5x - OpenGL LW_mesh_shader mesh program (!!LWmp5.0) assembly format"
    OPTNL "mtp5x - OpenGL LW_mesh_shader task program (!!LWmtp5.0) assembly format"
     OPTNL "fp1x - (Default) OpenGL LW_fragment_program_1"
     OPTNL "arbfp1 - OpenGL ARB_fragment_program_1: !!ARBFP1.0"
     OPTNL "arbvp1 - OpenGL ARB_vertex_program_1: !!ARBVP1.0"
     OPTNL "vs1x - DirectX 8 vertex shader: vs_1_0, vs_1_1, vs_1_x"
     OPTNL "vs2x - DirectX 9 vertex shader: vs_2_0, vs_2_x"
     OPTNL "vs3x - DirectX 9 vertex shader: vs_3_0"
     OPTNL "ps1x - DirectX 8 pixel shader: ps_1_1, ps_1_3 "
     OPTNL "ps14 - DirectX 8 pixel shader: ps_1_4"
     OPTNL "ps2x - DirectX 9 pixel shader: ps_2_0, ps_2_x"
     OPTNL "ps3x - DirectX 9 pixel shader"
#if defined(LW_LDDM_DDK_BUILD)
     OPTNL "sm4x - All DX10 Shader Model 4.x targets as text: vs_4_0, gs_4_0, ps_4_0"
     OPTNL "sm4x_bin - All DX10 Shader Model 4.x targets as binary byte code"
#endif
     OPTNL "fp30bin: LW3X binary ucode as input"
     OPTNL "fp40bin: LW4X binary ucode as input"
     OPTNL "lwinst - uses parseasm assembler for driver lwInst trace output"
     OPTNL "shd - uses pscomp"
     OPTNL "lwir - LWIR Trace point file"},
    // output args
    {"o", OT_STRING, &o_OutputFileName, "Output filename."},
    {"od", OT_STRING, &o_BaseOutputDir, "Directory in which to place outputs when using -il."},
    // codegen args
    {"profile", OT_STRING, &o_Profile, "Target profile.  Supported profiles are:"
     OPTNL "fp30 - Text or binary form of microcode for LW30 fragment processor"
     OPTNL "fp34 - Text or binary form of microcode for LW34 fragment processor"
     OPTNL "fp35 - Text or binary form of microcode for LW35 fragment processor (Default)"
     OPTNL "fp40 - Text or binary form of microcode for LW40 fragment processor"
     OPTNL "fp47 - Text or binary form of microcode for LW47 fragment processor"
     OPTNL "vp30 - Text or binary form of microcode for LW30 vertex processor"
     OPTNL "vp40 - Text or binary form of microcode for LW40 vertex processor"
     OPTNL "fpf - Text or binary form of microcode for Fermi fragment processor"
     OPTNL "gpf - Text or binary form of microcode for Fermi geometry processor"
     OPTNL "vpf - Text or binary form of microcode for Fermi vertex processor"
     OPTNL "cpf - Text or binary form of microcode for Fermi compute processor"
     OPTNL "hpf - Text or binary form of microcode for Fermi hull processor"
     OPTNL "tpf - Text or binary form of microcode for Fermi tessellation processor"
     OPTNL "mpf - Text or binary form of microcode for Fermi mesh processor"
     OPTNL "mtpf - Text or binary form of microcode for Fermi task processor"
    },
#if LWCFG(GLOBAL_ARCH_HOPPER)
    {"merlwry-colwerter", OT_INT, &o_MerlwryShortCirlwitLevel, "Input to enable Merlwry short circuit compilation"
     OPTNL "0: No Merlwry IR generated"
     OPTNL "1: ORI-IR to MercIR and SASS generation"
     OPTNL "2: ORI-IR to MercIR, finalizer expansion and SASS generation"},
#endif // HOPPER
    {"lwdqro", OT_NOOPT, &o_Quadro, "Generate code for Lwdqro device."},
    {"lwinst", OT_NOOPT, &o_lwinst, "Use lwInst code path."},
    {"psinst", OT_NOOPT, &o_lwinst, "Use lwInst code path."},
    {"binary", OT_NOOPT, &o_Binary, "Produce Binary Output."},
    {"dumpTokenStream", OT_NOOPT, &o_dumpTokenStream, "Dump (API) binary token stream to stdout and exit"},
    {"text", OT_NOOPT, &o_DumpText, "Attempt to force use of text profiles by code generator."},
    {"ucode", OT_NOOPT, &o_DumpBinaryUCode, "Output microcode to file ucode.bin (LW50 only)"},
    {"d3dOpt", OT_NOOPT, &o_d3dOpt, "Use D3D optimization path."},
    {"oglOpt", OT_NOOPT, &o_oglOpt, "Use OGL optimization path."},
    {"fileOpts", OT_NOOPT, &o_PostFileOptions, "Parse the file looking for extra parseasm options."
     OPTNL "flag is: ;#PASM_OPTS:"},
    // debug args
    {"l", OT_INT, &o_verbosityLevel, "Verbosity level."},
    {"vl", OT_INT, &o_validateLevel, "Validation level."},
    {"VA", OT_NOOPT, &o_ValidateAssert, "Assert on validation errors"},    
    {"etv", OT_INT, &o_evaluatorNumSets, "Number of random input sets for evaluator."},
    {"dbgid", OT_NOOPT, &o_showDebugID, "Show debug ID's in debug dumps."},
    {"pl", OT_INT, &o_ProfilerLevel, "COP Profiler level."},
    {"dumpStatsMemAllocBTree", OT_NOOPT, &o_DumpStatsMemAllocBTree, "Dump MemAllocBTree stats. Works only if used with -pl" },
    {"dumpStatsAtEnd", OT_NOOPT, &o_DumpStatsAtEnd, "Dump Compile Time and Memory Footprint stats at end. Works only if used with -pl" },
    {"nocl", OT_NOOPT, &o_NoCommandLine, "Do not add command line in generated output."},
#ifdef _DEBUG
#ifdef _MSC_VER
    {"setbreakalloc", OT_INT, &o_DebugBreakAlloc, "Set a breakpoint on nth memory allocation"
     OPTNL "used to break on specified memory allocation, useful finding leaks"},
    {"trap", OT_NOOPT, &o_DebugTrap, "Trap asserts (for debugging)."},
#endif // _MSC_VER
    {"hack", OT_INT, &o_Hack},
    {"hack2", OT_INT, &o_Hack2},
#endif // _DEBUG
    // optimizer flag args
    {"CanDemoteNonFP32Targets", OT_BOOL, &o_CanDemoteNonFP32Targets, "Can Demote Non FP32 targets."},
    {"CanIgnoreNan", OT_BOOL, &o_CanIgnoreNan, "Opt: Can ignore NAN."},
    {"CanIgnoreInf", OT_BOOL, &o_CanIgnoreInf, "Opt: Can ignore INF."},
    {"CanIgnoreSignedZero", OT_BOOL, &o_CanIgnoreSignedZero, "Opt: Can ignore signed zero."},
    {"CanPromoteFixed", OT_BOOL, &o_CanPromoteFixed, "Opt: Can promote fixed to FP16/FP32."},
    {"CanPromoteHalf", OT_BOOL, &o_CanPromoteHalf, "Opt: Can promote FP16 to FP32."},
    {"CanReorderFixedWithCheck", OT_BOOL, &o_CanReorderFixedWithCheck, "Opt: Can reassociate fixed ops."},
    {"CanReorderHalf", OT_BOOL, &o_CanReorderHalf, "Opt: Can reassociate FP16 ops."},
    {"CanReorderFloat", OT_BOOL, &o_CanReorderFloat, "Opt: Can reassociate FP32 ops."},
    {"CanUseNrmhAlways", OT_BOOL, &o_CanUseNrmhAlways, "Opt: Can use fp4x NRMH for all normalize operations."},
    {"Const31InRange", OT_BOOL, &o_Const31InRange, "Const 31 in 0..1 range (for Far Cry fog hoist)."},
    {"DX10SAT", OT_NOOPT, &o_ForceDX10SAT, "Opt: SAT(NaN) = 0.0."},
    {"DX10AddressInRange", OT_NOOPT, &o_ForceDX10AddressInRange, "Opt: As required by DX10 do not generate out of bound reads."},
    {"StartI", OT_INT, &o_StartInst, "Starting instruction number of the program"},
    {"MaxInstInBasicBlock", OT_INT, &o_MaxInstInBasicBlock, "Maximum number of instructions in a basic block"},
    {"miib", OT_INT, &o_MaxInstInBasicBlock, "Maximum number of instructions in a basic block"},
    {"constantComputationExtraction", OT_BOOL, &o_ConstantComputationExtraction, "Enable constant computation extraction"},
    {"controlPS", OT_HEX, &o_ControlPS, "Opt: PS Optimization setting bitmask."
     OPTNL "0: HW WAR/legalization only"
     OPTNL "1: COP optimization"},
    {"controlVS", OT_HEX, &o_ControlVS, "Opt: VS Optimization setting bitmask."
     OPTNL "0: HW WAR/legalization only"
     OPTNL "1: COP optimization"},
    {"debugMask", OT_HEX, &o_debugOutputMask, "specifies debug sections to output"
     OPTNL "1: source line number table"
     OPTNL "2: register map table"
     OPTNL "4: block map"
     OPTNL "8: all input/output entries (even unreferenced ones)"},
    {"ol", OT_FUN, lParseOptLevel, "Optimization level: (min, low, default, high, max)" }, 
    {"O0", OT_FUN, lParseOptLevelN, "No optimization."},
    {"O1", OT_FUN, lParseOptLevelN, "Low optimization."},
    {"O2", OT_FUN, lParseOptLevelN, "High optimization."},
    {"O3", OT_FUN, lParseOptLevelN, "Max optimization."},
    {"floatrt", OT_BOOL, &o_FloatRT, "1 means the render target is float." },
    {"fog", OT_INT, &o_AddFog, "Add fog instructions: "
     OPTNL "0: No Fog"
     OPTNL "1: EXP Fog"
     OPTNL "2: EXP2 Fog"
     OPTNL "3: Linear Fog"},
    {"infoFile", OT_STRING, &o_InfoFileName, "Read optimizer flags from file."},
    {"numcombiners", OT_INT, &o_NumCombiners, "Number of combiner stages used (for lw3x cyclecount)."},
    {"partialtexld", OT_HEX, &o_PartialTexLoad, "Bitfield of 2 bits per texture,"
     OPTNL "0: Textutre does not need to be split"
     OPTNL "1: semi-fat texture split needed"
     OPTNL "2: fat texture split needed"},
    {"reglimit", OT_INT, &o_SchedTargetRegSize, "Scheduling target register limit (in reg components)."},
    {"randomSched", OT_INT, &o_RandomSched, "at random interval from 1..<n>, select random instruction to schedule."},
    {"randomSeed", OT_INT, &o_RandomSeed, "random seed to use.  Will use time(0) if not specified."},
    {"signtex", OT_HEX, &o_SupportSignedRemap, "Bitfield of 1 bit per texture"
     OPTNL "0: Does not suppport signed remap,"
     OPTNL "1: supports signed remap"},
    {"srcalpha", OT_BOOL, &o_SrcAlpha, "Sets the source alpha mask."},
    {"srcrgb", OT_BOOL, &o_SrcRGB, "Sets the source RGB mask."},
    {"srgb", OT_BOOL, &o_Srgb, "1 means add srgb instructions."},
    {"texrange", OT_HEX, &o_TextureRange, "Bitfield of 2 bits per texture,"
     OPTNL "0: Texture range FP32"
     OPTNL "1: Texture range FP16"
     OPTNL "2: Texture range (-1, 1) FX9"
     OPTNL "3: Texture range (0, 1) FX8"},
    {"texRemap", OT_FUN, &lParseTexRemap, "<texID> <hexVal> - Sets the texture remap mode for a texunit (one byte for each component 0=x, 1=y, etc.)"
     OPTNL "00000000: - XXXX smear"
     OPTNL "03020100: - no smearing"},
    {"TexShadowMap", OT_HEX, &o_TexShadowMap, "Bitfield of 1 bit per texture,"
     OPTNL "0: Texture is not shadowmap"
     OPTNL "1: Texture is shadowmap"},
    {"texsrgb", OT_HEX, &o_TexSRGB, "Bitfield of 1 bit per texture,"
     OPTNL "0: Texture is not SRGB"
     OPTNL "1: Texture is SRGB"},
    {"textype", OT_HEX, &o_TextureType, "Bitfield of 2 bits per texture,"
     OPTNL "0: Texture type UNKNOWN"
     OPTNL "1: Texture type 2D"
     OPTNL "2: Texture type 3D"
     OPTNL "3: Texture type LWBE"},
    {"useconsttexalpha", OT_HEX, &o_UseConstTexAlpha, "Bitfield of 1 bit per texture,"
     OPTNL "0: Texture doesn't have constant alpha value"
     OPTNL "1: Texture does have constant alpha value"},
    // d3d flags
    {"dxver", OT_INT, &o_DXVer, "Specify DXVersion:"
     OPTNL "8: DirectX8"
     OPTNL "9: DirectX9 (default)"},
    {"bools", OT_STRING, &o_BoolsString, "Specify value for D3D bool constantants:"
     OPTNL "0xXX: use hex bitfield to initialize"
     OPTNL "random: use random values to initialize"},
    {"packFloatInput", OT_HEX, &o_PackFloatInput, "Bitfield of 2 bits per texture,"
     OPTNL "0: not packed (default)"
     OPTNL "1: float stored as 4 packed unsigned bytes"
     OPTNL "2: 2 halfs stored as 4 packed unsigned bytes"
     OPTNL "3: 2 floats stored as 4 packed unsigned shorts"},

    // misc flags
    {"measureTime", OT_INT, &o_codeGenRuns, "number of codegen runs for measuring time."},
    {"mt", OT_INT, &o_codeGenRuns, "number of codegen runs for measuring time."},
    {"hash", OT_NOOPT, &o_Hash, "Print hash code with output"},
    {"oglInfo", OT_NOOPT, &o_oglInfo, "Print additional info maintained primarily in the"
     OPTNL "driver for OpenGL programs."},
    {"vulkan", OT_NOOPT, &o_vulkan, "Compile GLASM as Vulkan"},
    {"glinternal", OT_NOOPT, &o_glinternal, "Compile GLASM as 'internal' code (for GLSL-generated source)." },
    {"dumpi", OT_NOOPT, &o_OnlyDumpInst, "Just dump input lwInsts and exit" },
    {"d3dArgs", OT_STRING, &o_d3dArgs, "List of options for DX10 driver to control DXASM to LwInst translation:" },
    // profile specifc flags
    {"po", OT_FUN, &lParseProfileOpt, "Profile-specific options:"
     OPTNL "debug:"
     OPTNL "col - show color information in debug output"
     OPTNL "reg - show register allocation information in debug output"
     OPTNL "sch - show scheduling information in debug output"
     OPTNL "num - numeric interpolants and outputs (Default)"
     OPTNL "sym - symbolic"
     OPTNL "ld - show live-dead register mask in assembly output"
     OPTNL "ln - show line/file number in debug output"
     OPTNL "rgb - only compute the rgb components (fragment programs)"
     OPTNL "lat2, lat3 - use instruction latency of 2, 3"
     OPTNL "lat3 - use instruction latency of 3"
     OPTNL "i2ilat n - number of cycles between instruction to instruction issue"
     OPTNL "tbat0, tbat1, tbat2, ..., tbat7 - batch texture instructions by 0...7"
     OPTNL "tbat n - batch texture instructions by 'n' (current internal limit is 16)"
     OPTNL "tbatauto - automatically pick texture batch size (pixel shaders)"
     OPTNL "tbatmaxt2tclk=n - maximum cycles bewteen two texures to be in the same batch phase"
     OPTNL "minphase n - don't use phases unless n or more TEX inst,n=0..7,0=never"
     OPTNL "minphase n - default for n = tbat count + 1, or 0 if no tbat"
     OPTNL "no32 - don't generate any 32-bit instructions"
     OPTNL "cimm - use immediate values for literal constants"
     OPTNL "creg - assign literal constants to c[] registers (Default)"
     OPTNL "obank n - Use bank number n for optimizer generated constants"
     OPTNL "flat mask - Hex mask to mark fragment intropolants flat, 1b per 4-vec."
     OPTNL "clamped mask - Hex mask to mark fragment interpolants to clamp to 0..1, 1b per 4-vec."
     OPTNL "dontmapibufs - Don't map IBUF indexes into compressed set.  Only for FakeGL use!"
     OPTNL "nofloatmad 0/1 - don't generate float mad."
     OPTNL "douniformatomexpand 0/1 - enable/disable expansion of uniform atomics."
     OPTNL "dofastuniformatomexpand 0/1 - enable/disable fast-path expansion of uniform atomics."
     OPTNL "douniformatomexpandusingshfl 0/1 - enable/disable expansion of uniform atomics using shfl."
     OPTNL "dovectoratomexpand 0/1 - enable/disable expansion of vector atomics."
     OPTNL "enablezerocoveragekill 0/1 - insert KIL if coverage output is written and 0"
     OPTNL "canuseglobalscratch 0/1 - use immediate global pointer instead of via const bank"
     OPTNL "sharedmemorysize n - size of shared memory used by application also offset into shared memory for OCG scratch"
     OPTNL "smemscratchperwarp n - num bytes of shared scratch available to OCG per warp"
     OPTNL "gmemscratchbase - global memory pointer to global OCG scratch space"
     OPTNL "gmemscratchconstbank - const bank where global pointer to OCG scratch is"
     OPTNL "gmemscratchconstoffset - const bank offset where global pointer to OCG scratch is"
     OPTNL "forcefmz - enforce 0*x=0 rule applied to all f32 mad and mul ops"
     OPTNL "rbank n - Schedule as if n registers available in register bank."
     OPTNL "          Note, value used as given and not scaled like driver supplied"
     OPTNL "          value.  E.g. 240 == driver value of 256."
     OPTNL "ssareg n - Use register n for SSA attribute reg (default value 4*aiFace)"
     OPTNL "sm_off - Compute instruction statistics using sm11 performance semantics"
     OPTNL "smXY - Target SMX.Y, 2 <= X <= 7  (sm50 -> Maxwell, sm60 -> Pascal, sm70 -> Volta etc.)"
     OPTNL "maxrregcount - Limit usage of r regs to maxrregcount"
     OPTNL "schedregtarget <n> - scheduling register target"
     OPTNL "mrt <mask> - for specifying mrt output"
     OPTNL "pw - Use phase windows"
     OPTNL "tepid - Show tepid clock latency in sass output"
     OPTNL "packio - use driver packing of user input/output attribute registers in OGL"
     OPTNL "dontpackio - preserve user input/output attribute registers in OGL"
     OPTNL "hexfp - print fp numbers in hex format"
     OPTNL "txdbat[1,2,4] - select how TEX insts are grouped when expanding TXD"
     OPTNL "vtxA - limit output to position; ignore unused inputs (vertex profiles only)"
     OPTNL "vtxB - limit output to non-position; ignore unused inputs (vertex profiles only)"
     OPTNL "vtxAB - append SM lwll instructions at the end (vertex profiles only)"
     OPTNL "colwertfastgs - if a geometry shader (version 5_0 or higher) is passed colwert the regular geometry shader to its fast geometry shader form"
     OPTNL "fastgs - the geometry shader (version 5_0 or higher) that is passed is already in fast geometry shader form"
     OPTNL "fastgswar - if a fast gs is generated with fastgs or colwertfastgs then add the WAR for bug 1514369. This flag is also valid on any regular tessellation-domain shader (version 5_0 or higher) to match the fast gs input."
     OPTNL "nostats - don't print any stats comments text/ucode"
     OPTNL "oriControl - control ori optimizer (bitmask):"
     OPTNL "    0: do default"
     OPTNL "    1: enable"
     OPTNL "    2: disable"
     OPTNL "ori - same as 'oriControl 1'"
     OPTNL "noori - same as 'oriControl 2'"
     OPTNL "task=xxx - list of compiler tasks"
     OPTNL "lwir - return Lwir in ucode (-ucode must also be specified)"
     OPTNL "texred [=1 or 0] - Enable Fermi texture reduction (remove .T for conselwtive tex to same SID/TID"
     OPTNL "maxwarpspertile [=n] - Set the max warps per tile"
     OPTNL "surfbank=n - Use bank n for bindless surface references"
     OPTNL "texbank=n - Use bank n for bindless texture references"
     OPTNL "mdes=filepath - Machine description file"
     OPTNL "SpelwlativelyHoistTex - Enable multiblock loop unrolling and texture hoisting for better batching"
     OPTNL "DumpPerfStats - Dump performance statistics."
     OPTNL "DumpPgmLatInfo=n - Dump shader exelwtion latency estimation information"
     OPTNL "    0: do not estimate."
     OPTNL "    1: estimate worst case latency."
     OPTNL "    2: estimate average case latency"
     OPTNL "unknownLoopIttr=n - Iteration counts for loops with unknown iteration count."
     OPTNL "numberOfSM=n - number of SMs"
     OPTNL "numOfWarpsLaunched=n - number of warps launched in a drawcall"
     OPTNL "usesTaskShader=n - compile mesh shaders to be used with(1) or without(0) a task shader"
    },
    {"knobsfile", OT_FUN, &lParseKnobsFileName, "Name of a file containing knobs."},
    {"knob", OT_FUN, &lParseKnobCommand, "Internal Knobs"
     OPTNL "<ident>"
     OPTNL "<ident>=<int>"
     OPTNL "<ident>=<0Xhex>"
    }
#undef OPTNL
};

// -texRemap <texid> <remapmode>

static int lParseTexRemap(int argc, const char *argv[], int argi)
{
    int texid;
    assert(argi + 2 < argc);
    sscanf(argv[++argi], "%d", &texid);
    if (texid >= 0 && texid < 16) {
        sscanf(argv[++argi], "%X", &o_TextureRemap[texid]);
    }
    else {
        printf("Warning: index (%d) out of range\n", texid);
    }
    return 2;
} // lParseTexRemap

// -ol <low/min/default/high/max>

static int lParseOptLevel(int argc, const char *argv[], int argi)
{
    const char *optl;

    assert(argi + 1 < argc);
    optl = argv[++argi];
    if (!_stricmp(optl, "min")) {
        o_OptLevel = OPT_LEVEL_MIN;
    } else if (!_stricmp(optl, "low")) {
        o_OptLevel = OPT_LEVEL_LOW;
    } else if (!_stricmp(optl, "default")) {
        o_OptLevel = OPT_LEVEL_DEFAULT;
    } else if (!_stricmp(optl, "high")) {
        o_OptLevel = OPT_LEVEL_HIGH;
    } else if (!_stricmp(optl, "max")) {
        o_OptLevel = OPT_LEVEL_MAX;
    } else {
        printf("Warning: unrecognized -ol option %s\n", argv[argi]);
        o_OptLevel = OPT_LEVEL_UNSPEC;
    }
    o_HasOptLevel = 1;
    return 1;
} // lParseOptLevel

// -O0, -O1, -O2, -O3

static int lParseOptLevelN(int argc, const char *argv[], int argi)
{
    const char *optl;

    optl = argv[argi];
    if (!_stricmp(optl, "-O0")) {
        o_OptLevel = OPT_LEVEL_MIN;
    } else if (!_stricmp(optl, "-O1")) {
        o_OptLevel = OPT_LEVEL_LOW;
    } else if (!_stricmp(optl, "-O2")) {
        o_OptLevel = OPT_LEVEL_HIGH;
    } else if (!_stricmp(optl, "-O3")) {
        o_OptLevel = OPT_LEVEL_MAX;
    } else {
        o_OptLevel = OPT_LEVEL_DEFAULT;
    }
    o_HasOptLevel = 1;
    return 0;
}

/*
 * lParseProfileOpt()
 * 
 * -po id [ = ( <int> | <hex> | <float> ) ]
 *
 */

static int lParseProfileOpt(int argc, const char *argv[], int argi)
{
    char ch;
    int i, idx, index, val;
    const char *pstr, *pBase, *pString;
    char base[4000], *lStr;
    int HasNumber, number, len;

    i = argi;
    if (++i < argc) {

        // Find args ending in "=999":
        pBase = argv[i];
        HasNumber = 0;
        number = 1;
        len = (int) strlen(argv[i]);
        if (len < 4000) {
            strcpy(base, argv[i]);
            lStr = base;
            pString = "";
            ch = *lStr++;
            while (ch != '\0') {
                if (ch == '=') {
                    pString = lStr;
                    ch = lStr[0];
                    if (ch >= '0' && ch <= '9') {
                        number = ch - '0';
                        ch = lStr[1];
                        if (ch == 'x') {
                            sscanf(lStr, "%x", &number);
                        } else {
                            sscanf(lStr, "%d", &number);
                        }
                        HasNumber = 1;
                    }
                    *--lStr = '\0';
                    pBase = &base[0];
                    break;
                }
                ch = *lStr++;
            }
        }

        // Separate out -po smXX from rest of options
        if (argv[i][0] == 's' && argv[i][1] == 'm') {
            if (!strcmp(argv[i], "sm_off")) {
                o_FmtFlags |= FMT_SM_VERSION_11;
            } else if (!strcmp(argv[i], "sm20")) {
                o_ArchVariant = COP_ARCH_VARIANT_FERMI_SM2_0;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm21")) {
                o_ArchVariant = COP_ARCH_VARIANT_FERMI_SM2_1;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm30")) {
                o_ArchVariant = COP_ARCH_VARIANT_KEPLER_SM3_0;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm32")) {
                o_ArchVariant = COP_ARCH_VARIANT_KEPLER_SM3_2;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm35")) {
                o_ArchVariant = COP_ARCH_VARIANT_KEPLER_SM4_0;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm40")) {
                o_ArchVariant = COP_ARCH_VARIANT_KEPLER_SM4_0;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm50")) {
                o_ArchVariant = COP_ARCH_VARIANT_MAXWELL_SM5_0;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm52")) {
                o_ArchVariant = COP_ARCH_VARIANT_MAXWELL_SM5_2;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm53")) {
                o_ArchVariant = COP_ARCH_VARIANT_MAXWELL_SM5_3;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm60")) {
                o_ArchVariant = COP_ARCH_VARIANT_PASCAL_SM6_0;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm61")) {
                o_ArchVariant = COP_ARCH_VARIANT_PASCAL_SM6_1;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm62")) {
                o_ArchVariant = COP_ARCH_VARIANT_PASCAL_SM6_2;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm70")) {
                o_ArchVariant = COP_ARCH_VARIANT_VOLTA_SM7_0;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm72")) {
                o_ArchVariant = COP_ARCH_VARIANT_VOLTA_SM7_2;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm73")) {
                o_ArchVariant = COP_ARCH_VARIANT_TURING_SM7_3;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm75")) {
                o_ArchVariant = COP_ARCH_VARIANT_TURING_SM7_5;
                o_HasArchVariant = 1;
#if LWCFG(GLOBAL_ARCH_AMPERE)
            } else if (!strcmp(argv[i], "sm82")) {
                o_ArchVariant = COP_ARCH_VARIANT_AMPERE_SM8_2;
                o_HasArchVariant = 1;
            } else if (!strcmp(argv[i], "sm86")) {
                o_ArchVariant = COP_ARCH_VARIANT_AMPERE_SM8_6;
                o_HasArchVariant = 1;
#if LWCFG(GLOBAL_GPU_IMPL_GA10B)
            } else if (!strcmp(argv[i], "sm87")) {
                o_ArchVariant = COP_ARCH_VARIANT_AMPERE_SM8_7;
                o_HasArchVariant = 1;
#endif // LWCFG(GLOBAL_GPU_IMPL_GA10B)
#if LWCFG(GLOBAL_GPU_IMPL_GA10F)
            } else if (!strcmp(argv[i], "sm88")) {
                o_ArchVariant = COP_ARCH_VARIANT_AMPERE_SM8_8;
                o_HasArchVariant = 1;
#endif // LWCFG(GLOBAL_GPU_IMPL_GA10F)
#endif // LWCFG(GLOBAL_ARCH_AMPERE)
#if LWCFG(GLOBAL_ARCH_ADA)
            } else if (!strcmp(argv[i], "sm89")) {
                o_ArchVariant = COP_ARCH_VARIANT_ADA_SM8_9;
                o_HasArchVariant = 1;
#endif // LWCFG(GLOBAL_ARCH_ADA)
#if LWCFG(GLOBAL_ARCH_HOPPER)
            } else if (!strcmp(argv[i], "sm90")) {
                o_ArchVariant = COP_ARCH_VARIANT_HOPPER_SM9_0;
                o_HasArchVariant = 1;
#endif // LWCFG(GLOBAL_ARCH_HOPPER)
            } else {
                fprintf(stderr, "Unknown sm specified\n");
                lCallExit(1);
            }
        }

        // Parse rest of options
        if (!strcmp(argv[i], "num")) {
            o_FmtFlags &= ~FMT_FLAG_SYMBOLIC;
        } else if (!strcmp(argv[i], "sym")) {
            o_FmtFlags |= FMT_FLAG_SYMBOLIC;
        } else if (!strcmp(argv[i], "col")) {
            o_FmtFlags |= FMT_FLAG_SHOW_COLORS;
        } else if (!strcmp(argv[i], "ld")) {
            o_FmtFlags |= FMT_FLAG_SHOW_LIVEDEAD;
        } else if (!strcmp(argv[i], "ln")) {
            o_FmtFlags |= FMT_LINE_NUMBERS;
        } else if (!strcmp(pBase, "sch")) {
            if ((number & 1) != 0) {
                o_FmtFlags |= FMT_FLAG_TRACE_SCHEDULING;
            } else {
                o_FmtFlags &= ~FMT_FLAG_TRACE_SCHEDULING;
            }
            o_SchedulerFlags = number;
        } else if (!strcmp(argv[i], "reg")) {
            o_FmtFlags |= FMT_FLAG_TRACE_REG_ALLOC;
        } else if (!strcmp(argv[i], "rgb")) {
            o_FmtFlags |= FMT_FLAG_RGB_ONLY_OUTPUT;
        } else if (!strcmp(argv[i], "cimm")) {
            o_FmtFlags |= FMT_CONST_MODE_IMMEDIATE;
        } else if (!strcmp(argv[i], "creg")) {
            o_FmtFlags &= ~FMT_CONST_MODE_IMMEDIATE;
        } else if (!strcmp(argv[i], "tepid")) {
            o_FmtFlags |= FMT_TEPID_LATENCY;
        } else if (!strcmp(argv[i], "lat0")) {
            o_FmtFlags &= ~FMT_LATENCY_MASK;
            o_FmtFlags |= FMT_LATENCY_0;
            o_HasLat = 1;
            o_Lat = 0;
        } else if (!strcmp(argv[i], "lat1")) {
            o_FmtFlags &= ~FMT_LATENCY_MASK;
            o_FmtFlags |= FMT_LATENCY_1;
            o_HasLat = 1;
            o_Lat = 1;
        } else if (!strcmp(argv[i], "lat2")) {
            o_FmtFlags &= ~FMT_LATENCY_MASK;
            o_FmtFlags |= FMT_LATENCY_2;
            o_HasLat = 1;
            o_Lat = 2;
        } else if (!strcmp(argv[i], "lat3")) {
            o_FmtFlags &= ~FMT_LATENCY_MASK;
            o_FmtFlags |= FMT_LATENCY_3;
            o_HasLat = 1;
            o_Lat = 3;
        } else if (!strcmp(pBase, "i2ilat")) {
            o_HasI2ILat = 1;
            o_I2ILat = number;
        } else if (!strcmp(argv[i], "tbat0")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_HasTBat = 1;
            o_TBat = 0;
            o_AutoBatch = 0;
        } else if (!strcmp(argv[i], "tbat1")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_FmtFlags |= FMT_TEX_BATCH_SIZE_1;
            o_HasTBat = 1;
            o_TBat = 1;
            o_AutoBatch = 0;
            if (!o_HasUserPhaseCount)
                o_PhaseCount = 2;
        } else if (!strcmp(argv[i], "tbat2")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_FmtFlags |= FMT_TEX_BATCH_SIZE_2;
            o_HasTBat = 1;
            o_TBat = 2;
            o_AutoBatch = 0;
            if (!o_HasUserPhaseCount)
                o_PhaseCount = 3;
        } else if (!strcmp(argv[i], "tbat3")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_FmtFlags |= FMT_TEX_BATCH_SIZE_3;
            o_HasTBat = 1;
            o_TBat = 3;
            o_AutoBatch = 0;
            if (!o_HasUserPhaseCount)
                o_PhaseCount = 4;
        } else if (!strcmp(argv[i], "tbat4")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_FmtFlags |= FMT_TEX_BATCH_SIZE_4;
            o_HasTBat = 1;
            o_TBat = 4;
            o_AutoBatch = 0;
            if (!o_HasUserPhaseCount)
                o_PhaseCount = 5;
        } else if (!strcmp(argv[i], "tbat5")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_FmtFlags |= FMT_TEX_BATCH_SIZE_5;
            o_HasTBat = 1;
            o_TBat = 5;
            o_AutoBatch = 0;
            if (!o_HasUserPhaseCount)
                o_PhaseCount = 6;
        } else if (!strcmp(argv[i], "tbat6")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_FmtFlags |= FMT_TEX_BATCH_SIZE_6;
            o_HasTBat = 1;
            o_TBat = 6;
            o_AutoBatch = 0;
            if (!o_HasUserPhaseCount)
                o_PhaseCount = 7;
        } else if (!strcmp(argv[i], "tbat7")) {
            o_FmtFlags &= ~FMT_TEX_BATCH_MASK;
            o_FmtFlags |= FMT_TEX_BATCH_SIZE_7;
            o_HasTBat = 1;
            o_TBat = 7;
            o_AutoBatch = 0;
            if (!o_HasUserPhaseCount)
                o_PhaseCount = 7; // SB: 8 but no room.
        } else if (!strcmp(argv[i], "tbat")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &o_TBat);
                o_HasTBat = 1;
                o_AutoBatch = 0;
            }
        } else if (!strcmp(pBase, "tbat")) {
            o_HasTBat = 1;
            o_AutoBatch = 0;
            o_TBat = number;
        } else if (!strcmp(argv[i], "tbatauto")) {
            o_HasTBat = 1;
            o_AutoBatch = 1;
        } else if (!strcmp(pBase, "tbatmaxt2tclk")) {
            o_TBatMaxT2TClk = number;
            o_HasTBatMaxT2TClk = 1;
        } else if (!strcmp(pBase, "vliw")) {
            o_UseVLIW = number;
        } else if (!strcmp(pBase, "direct2ir")) {
            if (++i < argc) {
                if (!strcmp(argv[i], "default")) {
                    o_Direct2IR = COP_UNKNOWN_DEFAULT;
                } else if (!strcmp(argv[i], "on")) {
                    o_Direct2IR = COP_TRUE_ENABLE;
                } else if (!strcmp(argv[i], "off")) {
                    o_Direct2IR = COP_FALSE_DISABLE;
                } else {
                    fprintf(stderr, "direct2ir option needs default/on/off as value, %s specified\n", argv[i]);
                    lCallExit(1);
                }
            } else {
                fprintf(stderr, "direct2ir option needs default/on/off as value, %s specified\n", argv[i]);
                lCallExit(1);
            }
        } else if (!strcmp(pBase, "stress-noglobalregalloc")) {
            o_RegStressFlags |= COP_STRESS_REG_NO_GLOBAL_REG_ALLOC;
        } else if (!strcmp(pBase, "stress-no-crp")) {
            o_RegStressFlags |= COP_STRESS_REG_ALLOC_NO_CRP;
        } else if (!strcmp(pBase, "stress-maxrregcount")) {
            // TODO: pass reg number
            o_RegStressFlags |= COP_STRESS_REG_ALLOC_FORCE_MAXREG;                
        } else if (!strcmp(pBase, "vliwfile")) {
            char *lStr = (char *) malloc(strlen(pString) + 1);
            strcpy(lStr, pString);
            o_VLIWfileName = lStr;
        } else if (!strcmp(pBase, "mdes")) {
            char *lStr = (char *) malloc(strlen(pString) + 1);
            strcpy(lStr, pString);
            o_mdesFileName = lStr;
        } else if (!strcmp(pBase, "task")) {
            char *lStr = (char *) malloc(strlen(pString) + 1);
            strcpy(lStr, pString);
            o_Tasks = lStr;
        } else if (!strcmp(pBase, "lwir")) {
            o_EmitLwir = number;
            o_HasEmitLwir = 1;
        } else if (!strcmp(pBase, "texred")) {
            o_TextureReduction = number;
            o_HasTextureReduction = 1;
        } else if (!strcmp(pBase, "maxwarpspertile")) {
            o_MaxWarpsPerTile = number;
            o_HasMaxWarpsPerTile = 1;
        } else if (!strcmp(pBase, "exA")) {
            o_ExperimentalA = number;
        } else if (!strcmp(pBase, "exB")) {
            o_ExperimentalB = number;
        } else if (!strcmp(pBase, "exC")) {
            o_ExperimentalC = number;
        } else if (!strcmp(pBase, "exD")) {
            o_ExperimentalD = number;
        } else if (!strcmp(pBase, "freq")) {
            o_ShowInstFreq = number;
        } else if (!strcmp(argv[i], "no32")) {
            o_FmtFlags |= FMT_NO_32_BIT_INST;
        } else if (!strcmp(argv[i], "altsch")) {
            o_FmtFlags |= FMT_ALT_SCHEDULER;
        } else if (!strcmp(argv[i], "dontmapibufs")) {
            o_FmtFlags |= FMT_DONT_PACK_IBUF_NOS;
        } else if (!strcmp(argv[i], "pw")) {
            o_UsePhaseWindows = 1;
            o_HasUsePhaseWindows = 1;
        } else if (!strcmp(argv[i], "earlyp1")) {
            o_UseEarlyPhase1 = 1;
            o_HasUseEarlyPhase1 = 1;
        } else if (!strcmp(argv[i], "minphase")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &idx);
                if (idx >= 0 && idx < 8) {
                    o_PhaseCount = idx;
                    o_HasUserPhaseCount = 1;
                }
            }
        } else if (!strcmp(argv[i], "rbank")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &o_RRegBankSize);
            }
        } else if (!strcmp(argv[i], "packio")) {
            o_EnableIOPacking = 1;
        } else if (!strcmp(argv[i], "dontpackio")) {
            o_EnableIOPacking = 0;
        } else if (!strcmp(argv[i], "mrt")) {
            if (++i < argc) {
                sscanf(argv[i], "%x", &o_MrtMask);
                o_OverrideMrtMask = 1;
            }
        } else if (!strcmp(argv[i], "oriControl")) {
            if (++i < argc) {
                sscanf(argv[i], "%x", &o_OriControl);
            }
        } else if (!strcmp(argv[i], "ori")) {
            o_OriControl |= ORI_ENABLE;
        } else if (!strcmp(argv[i], "noori")) {
            o_OriControl |= ORI_DISABLE;
        } else if (!strcmp(argv[i], "earlyphase") || !strcmp(argv[i], "ep")) {
            o_EarlyPhase = 1;
        } else if (!strcmp(argv[i], "hexfp")) {
            o_FmtFlags |= FMT_HEX_FLOAT_CONSTANTS;
        } else if (!strcmp(argv[i], "deadout")) {
            if (++i < argc) {
                index = (int)strlen(argv[i]) - 1;
                pstr = argv[i];
                while (*pstr != '\0') {
                    ch = *pstr++;
                    if ((index >> 3) < DEAD_OUTPUT_MASK_ARR_LEN) {
                        if (ch >= '0' && ch <= '9') {
                            val = ch - '0';
                        } else if (ch >= 'A' && ch <= 'F') {
                            val = (ch - 'A') + 10;
                        } else if (ch >= 'a' && ch <= 'f') {
                            val = (ch - 'a') + 10;
                        } else {
                            fprintf(stderr, "Error - malformed hex mask \"%s\"\n",
                                    argv[i]);
                            lCallExit(1);
                        }
                        o_DeadOutputMask[index >> 3] |= val << ((index & 7)*4);
                        o_HasDeadOutputMask = 1;
                    }
                    index--;
                }
            }
        } else if (!strcmp(argv[i], "flat")) {
            if (++i < argc) {
                sscanf(argv[i], "%X", &o_UpperAttributeMask);
                o_FlatUpperAttributes = 1;
            }
        } else if (!strcmp(argv[i], "obank")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &idx);
                if (idx >= 0 && idx < 16) {
                    o_OptimizerConstantBank = idx;
                }
            }
        } else if (!strcmp(argv[i], "ssareg")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &idx);
                if (idx >= 0 && idx < 255) {
                    o_SSAAttribReg = idx;
                }
            }
        } else if (!strcmp(pBase, "texbank")) {
            o_HasBindlessTextureBank = 1;
            o_BindlessTextureBank = number;
        } else if (!strcmp(argv[i], "maxrregcount")) {
            if (++i < argc)
                sscanf(argv[i], "%d", &o_maxRRegCount);
        } else if (!strcmp(pBase, "maxrregcount")) {
            o_maxRRegCount = number;
        } else if (!strcmp(argv[i], "schedregtarget")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &o_schedRegTarget);
            }
        } else if (!strcmp(pBase, "hint")) {
            o_hintControl = number;
        } else if (!strcmp(argv[i], "nofloatmad")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &o_NoFloatMAD);
                o_HasNoFloatMAD = 1;
            }
        } else if (!strcmp(pBase, "forcefmz")) {
            sscanf(argv[i], "%d", &number);
            o_ForceFMZ = number;
            o_HasForceFMZ = 1;
        } else if (!strcmp(pBase, "disableftz")) {
            sscanf(argv[i], "%d", &number);
            o_DisableFTZ = number;
            o_HasDisableFTZ = 1;
        } else if (!strcmp(pBase, "IsWPOSOne")) {
            sscanf(argv[i], "%d", &number);
            o_IsWPOSOne = number;
        } else if (!strcmp(pBase, "douniformatomexpand")) {
            if (o_HasDoFastUniformAtomExpand || o_HasDoUniformAtomExpandUsingSHFL) {
                fprintf(stderr, "specify douniformatomexpand or dofastuniformatomexpand or douniformatomexpandusingshfl\n");
                lCallExit(1);
            }

            sscanf(argv[i], "%d", &number);
            o_DoUniformAtomExpand = number;
            o_HasDoUniformAtomExpand = 1;
        } else if (!strcmp(pBase, "dofastuniformatomexpand")) {
            if (o_HasDoUniformAtomExpand || o_HasDoUniformAtomExpandUsingSHFL) {
                fprintf(stderr, "specify douniformatomexpand or dofastuniformatomexpand or douniformatomexpandusingshfl\n");
                lCallExit(1);
            }

            sscanf(argv[i], "%d", &number);
            o_DoFastUniformAtomExpand = number;
            o_HasDoFastUniformAtomExpand = 1;
        } else if (!strcmp(pBase, "douniformatomexpandusingshfl")) {
            if (o_HasDoUniformAtomExpand || o_HasDoFastUniformAtomExpand) {
                fprintf(stderr, "specify douniformatomexpand or dofastuniformatomexpand or douniformatomexpandusingshfl\n");
                lCallExit(1);
            }

            sscanf(argv[i], "%d", &number);
            o_DoUniformAtomExpandUsingSHFL = number;
            o_HasDoUniformAtomExpandUsingSHFL = 1;
        } else if (!strcmp(pBase, "dovectoratomexpand")) {
            sscanf(argv[i], "%d", &number);
            o_DoVectorAtomExpand = number;
            o_HasDoVectorAtomExpand = 1;
        } else if (!strcmp(pBase, "DumpPerfStats")) {
            o_DumpPerfStats = 1;
        } else if (!strcmp(pBase, "DumpPgmLatInfo")) {
            sscanf(argv[i], "%d", &number);
            o_DumpPgmLatInfo = number;
        } else if (!strcmp(pBase, "unknownLoopIttr")) {
            sscanf(argv[i], "%d", &number);
            o_unknownLoopIttr = number;
        } else if (!strcmp(pBase, "numberOfSM")) {
            sscanf(argv[i], "%d", &number);
            o_numberOfSM = number;
        } else if (!strcmp(pBase, "numWarpsLaunched")) {
            sscanf(argv[i], "%d", &number);
            o_numWarpsLaunched = number;
        } else if (!strcmp(pBase, "enablezerocoveragekill")) {
            sscanf(argv[i], "%d", &number);
            o_EnableZeroCoverageKill = number;
            o_HasEnableZeroCoverageKill = 1;
        } else if (!strcmp(pBase, "canuseglobalscratch")) {
            sscanf(argv[i], "%d", &number);
            o_CanUseGlobalScratch = number;
            o_HasCanUseGlobalScratch = 1;
        } else if (!strcmp(pBase, "sharedmemorysize")) {
            sscanf(argv[i], "%d", &number);
            o_SharedMemorySize = number;
            o_HasSharedMemorySize = 1;
         } else if (!strcmp(pBase, "smemscratchperwarp")) {
            sscanf(argv[i], "%d", &number);
            o_SMemScratchPerWarp = number;
            o_HasSMemScratchPerWarp = 1;
        } else if (!strcmp(pBase, "gmemscratchbase")) {
            sscanf(argv[i], "%d", &number);
            o_GMemScratchBase = number;
            o_HasGMemScratchBase = 1;
        } else if (!strcmp(pBase, "gmemscratchconstbank")) {
            sscanf(argv[i], "%d", &number);
            o_GMemScratchConstBank = number;
            o_HasGMemScratchConstBank = 1;
        } else if (!strcmp(pBase, "gmemscratchconstoffset")) {
            sscanf(argv[i], "%d", &number);
            o_GMemScratchConstOffset = number;
            o_HasGMemScratchConstOffset = 1;
        } else if (!strcmp(argv[i], "sfufloatmul")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &o_SFUFloatMUL);
                o_HasSFUFloatMUL = 1;
            }
        } else if (!strcmp(argv[i], "txdbat1")) {
            o_HasTxdBatchSize = 1;
            o_TxdBatchSize = 1;
        } else if (!strcmp(argv[i], "txdbat2")) {
            o_HasTxdBatchSize = 1;
            o_TxdBatchSize = 2;
        } else if (!strcmp(argv[i], "txdbat4")) {
            o_HasTxdBatchSize = 1;
            o_TxdBatchSize = 4;
        } else if (!strcmp(argv[i], "vtxA")) {
            o_vtxA = 1;
            o_vtxB = 0;
        } else if (!strcmp(argv[i], "vtxB")) {
            o_vtxB = 1;
            o_vtxA = 0;
        } else if (!strcmp(argv[i], "vtxAB")) {
            o_vtxA = 1;
            o_vtxB = 1;
        } else if (!strcmp(argv[i], "colwertfastgs")) {
            o_ColwertFastGs = 1;
        } else if (!strcmp(argv[i], "fastgs")) {
            o_FastGs = 1;
        } else if (!strcmp(argv[i], "fastgswar")) {
            o_FastGsWar = 1;
        } else if (!strcmp(argv[i], "dummycread")) {
            o_HasAddDummyCRead = 1;
            o_AddDummyCRead = 1;
        } else if (!strcmp(argv[i], "asfunc")) {
            o_CompileAsFunction = 1;
        } else if (!strcmp(argv[i], "nostats")) {
            o_FmtFlags |= FMT_NO_STATS;
        } else if (!strcmp(pBase, "knobsFile") || !strcmp(pBase, "knobsfile")) {
            char *lStr = (char *) malloc(strlen(pString) + 1);
            strcpy(lStr, pString);
            o_KnobsFileName = lStr;
        } else if (!strcmp(pBase, "SpelwlativelyHoistTex")) {
            o_SpelwlativelyHoistTex = 1;
        } else if (!strcmp(pBase, "usesTaskShader")) {
            if (++i < argc) {
                sscanf(argv[i], "%d", &o_usesTaskShader);
            }
        } else {
            if (o_ProfileStruct)
                o_ProfileStruct->ProcessProfileOptions(pBase, pString, HasNumber, number);
        }
    }
    return i - argi;
} // lParseProfileOpt


/*
 * lParseKnobCommand() - Append the knob at the end of o_KnobsString
 *
 */

static int lParseKnobCommand(int argc, const char *argv[], int argi)
{
    int newArgi;

    newArgi = argi;
    if (++newArgi < argc) {
        const char *knobStr = argv[newArgi];
        int knobLen = (int) strlen(knobStr);

        if (o_KnobsString == NULL) {
            o_KnobsString = (char *)malloc(knobLen + 1);
            strcpy(o_KnobsString, knobStr);
        } else {
            // realloc for every new knob option is inefficient,
            // but is OK since knobs are not for production code.
            int oldLen = (int) strlen(o_KnobsString);
            int newLen = oldLen + knobLen + 2;
            char *str = (char *)realloc(o_KnobsString, newLen);
            o_KnobsString = str;
            // insert a space between two knobs
            str += oldLen;
            *str++ = ' ';
            strcpy(str, knobStr); 
        }
    }
    return newArgi - argi;
} // lParseKnobCommand

/*
 * lParseKnobsFileName() -
 *
 */

static int lParseKnobsFileName(int argc, const char *argv[], int argi)
{
    if (argi < argc) {
        o_KnobsFileName = argv[argi + 1];
        return 1;
    } else {
        return 0;
    }
} // lParseKnobsFileName

/*
 * PrintOptionsHelp() -
 *
 */

static void PrintOptionsHelp(void)
{
    int ii;

    for (ii = 0; ii < sizeof(optionArr)/sizeof(optionArr[0]); ii++) {
        printf("    -%-24s ",optionArr[ii].name);
        switch (optionArr[ii].type) {
        case OT_INT:
            printf("<int> - %s", optionArr[ii].helpMessage);
            break;
        case OT_FLOAT:
            printf("<float> - %s", optionArr[ii].helpMessage);
            break;
        case OT_BOOL:
            printf("<0/1> - %s", optionArr[ii].helpMessage);
            break;
        case OT_HEX:
            printf("<Hex> - %s", optionArr[ii].helpMessage);
            break;
        case OT_NOOPT:
        case OT_FUN:
            printf("%s", optionArr[ii].helpMessage);
            break;
        case OT_STRING:
            printf("<string> - %s", optionArr[ii].helpMessage);
            break;
        default:
            printf("???\n");
            break;
        }
#if defined(LW_LDDM_DDK_BUILD)
        if (strcmp(optionArr[ii].name, "d3dArgs") == 0)
        {
            printf("\n");
            printf(g_d3dArg_HelpString);
        }
#endif
        printf("\n");
    }
    
#if defined(AR20)
    PrintHelp_ar20fp();
#endif //AR20

} // PrintOptionsHelp

/*
 * ProcessOption() -
 *
 */

static int ProcessOption(int argc, const char* argv[], int argi, int *extraInc)
{
    int ii;
    const char *str;

    str = argv[argi];
    if (*str != '-')
        return 0;
    str++;

    for (ii = 0; ii < sizeof(optionArr)/sizeof(optionArr[0]); ii++) {
        if (_stricmp(str, optionArr[ii].name) == 0) {
            if (optionArr[ii].type != OT_NOOPT && optionArr[ii].type != OT_FUN) {
                if (argv[argi + 1] == NULL) {
                    printf("Error: '%s' option expects a parameter\n",str);
                    return FALSE;
                }
            }
            switch (optionArr[ii].type) {
            case OT_INT:
                *(int *) optionArr[ii].val = atoi(argv[argi + 1]);
                *extraInc = 1;
                return TRUE;
            case OT_FLOAT:
                *(float *) optionArr[ii].val = (float) atof(argv[argi + 1]);
                *extraInc = 1;
                return TRUE;
            case OT_BOOL:
                *(int *) optionArr[ii].val = atoi(argv[argi + 1]);
                if (*(int *) optionArr[ii].val != 0 &&
                    *(int *) optionArr[ii].val != 1)
                {
                    printf("Error: %s option expects 0/1\n",str);
                    return FALSE;
                }
                *extraInc = 1;
                return TRUE;
            case OT_HEX:
                sscanf(argv[argi + 1], "%X", (int *) optionArr[ii].val);
                *extraInc = 1;
                return TRUE;
            case OT_NOOPT:
                *(int *) optionArr[ii].val = 1;
                *extraInc = 0;
                return TRUE;
            case OT_STRING:
                *(const char **) optionArr[ii].val = argv[argi + 1];
                *extraInc = 1;
                return TRUE;
            case OT_FUN:
                *extraInc = ((ProcessOptionFun) optionArr[ii].val)(argc, argv, argi);
                return TRUE;
            default:
                assert("unknown option type" == 0);
                return FALSE;
            }
        }
    }
    return FALSE;
} // ProcessOption


/*
 * ProcessFileOptions() -
 *
 */

void ProcessFileOptions(char *fString, unsigned fSize)
{
    char optBuffer[20][20];
    const char *optv[ALEN(optBuffer)];
    char *buf;
    int ii, jj, kk, extraInc;

    for(buf = fString; buf < fString+fSize; buf++) {
        if (strncmp(buf, ";#PASM_OPTS:", 12) == 0) {
            buf += 12;
            for(jj = 0; jj < ALEN(optBuffer); jj++) {
                // skip spaces
                for(;;) {
                    if (!isspace(*buf))
                        break;
                    buf++;
                }
                for(kk = 0; kk < ALEN(optBuffer[0]); kk++) {
                    if (isspace(*buf) || iscntrl(*buf))
                        break;
                    optBuffer[jj][kk] = *buf;
                    buf++;
                }
                if (kk == 0)
                    break;
                if (kk == ALEN(optBuffer[0])) {
                    fprintf(stderr, "Warning: file option %.20s... too long.\n", optBuffer[jj]);
                } else {
                    optBuffer[jj][kk] = 0;
                    optv[jj] = optBuffer[jj];
                }
                if (iscntrl(*buf))
                    break;
            }
            if (jj == ALEN(optBuffer)) {
                fprintf(stderr, "Warning: maximum number of file options exceeded\n");
                jj--;
            }
            // process the options
            for(ii = 0; ii <= jj; ii ++) {
                if (ProcessOption(jj+1, optv, ii, &extraInc)) {
                    ii += extraInc;
                } else {
                    fprintf(stderr, "Warning: unrecognized option in file - %s\n", optBuffer[ii]);
                }
            }
        }
    }
} // ProcessFileOptions

int compileOnce(char *inpFileName, FILE *outFile, LWuCode **ppUCode, void **ppCgBinaryData, 
                unsigned int *pCgBinarySize, int fBinary)
{
    unsigned char *lRawInput = NULL;
    int lSize, nRetVal;

    if (!o_OnlyDumpInst) {
        printf("Compiling %s for %s\n", inpFileName, o_Profile);
    }

    lRawInput = lReadFile(inpFileName,
                          (parseMethod == PARSE_D3D10_SM4X_BIN) || (parseMethod == PARSE_D3D_VSBIN) || (parseMethod == PARSE_D3D_PSBIN),
                          (unsigned int *) &lSize);
    if (!lRawInput) {
        fprintf(stderr, " Could not read inp file\n");
        lCallExit(1);
    }

    nRetVal = compileOnceMemoryBuffer(lRawInput, lSize, outFile, ppUCode, ppCgBinaryData, 
                                      pCgBinarySize, fBinary);
    
    if (lRawInput)
      free(lRawInput);
    
    return(nRetVal);
} // compileOnce


int __cdecl main(int argc, const char* argv[])
{
    int ii, retVal = 0, extraInc;
    FILE *lOutFile = stdout, *lInListFile;
    size_t len = -1, commandLineLen;
    LWuCode *pUCode;
    void *pCgBinary;
    unsigned int lCgBinarySize = 0;
    char baseOutputDir[512] = "";

#if defined(_MSC_VER)
    // Make C runtime check errors go to stderr instead of popping up a window.
    // Pass -trap if you want a window/breakpoint in the debugger.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);

    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    SetErrorMode(3);
#endif

    // Set stdout and stderr to not be buffered so we get all of our
    // debug output.
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    // Set Defults:

    // o_InputFormat = "fp1x";
    // o_Profile = "fp35";

    // Parse command line

    commandLineLen = 0;
    for (ii = 1; ii < argc; ii++)
        commandLineLen += strlen(argv[ii]) + 1;
    commandLine = (char *) malloc(commandLineLen + 1);
    commandLine[0] = '\0';
    for (ii = 1; ii < argc; ii++) {
        if (ii > 1)
            strcat(commandLine, " ");
        strcat(commandLine, argv[ii]);
    }
    o_PhaseCount = 0;
    o_HasUserPhaseCount = 0;
    o_RRegBankSize = 0;    
    // first identify the profile so that the profile struct mechanism can work
    for (ii = 1; ii < argc; ii++) {
        if (!strcmp(argv[ii], "-profile")) {
            if (++ii < argc) {
                o_Profile = argv[ii];
                if (ProcessProfile()) {
                    printf("Unsupported Input Profile: %s\n", argv[ii]);
                    printf("Use: -h for help.\n");
                    lCallExit(0);
                }
            }
        }
    }
    // If the profile isn't present, find the file name and set an appropriate default:
    if (o_Profile == NULL) {
        for (ii = 1; ii < argc; ii++) {
            if (!strcmp(argv[ii], "-i")) {
                if (++ii < argc) {
                    SetDefaultProfile(argv[ii]);
                    break;
                }
            }
        }
    }
    InitializeProfileStruct();
    for (ii = 1; ii < argc; ii++) {
        if (ProcessOption(argc, argv, ii, &extraInc)) {
            ii += extraInc;
        } else {
            printf("Unknown Input Option: %s\n", argv[ii]);
            printf("Use: -h for help.\n");
            lCallExit(0);
        }
    }

    if (o_Help) {
        PrintOptionsHelp();
        lCallExit(0);
    }

    if (o_BaseOutputDir) {
        strcpy(baseOutputDir, o_BaseOutputDir);
        len = strlen(baseOutputDir);
        if (baseOutputDir[len-1] != '/' && baseOutputDir[len-1] != '\\') {
            baseOutputDir[len] = '/';
            baseOutputDir[len+1] = '\0';
        }
    }

    SetDefaults();

    if (ProcessInputFormat() != 0) {
        fprintf(stderr, "Error - unknown input format \"%s\"\n", o_InputFormat);
        printf("Use: -h for help.\n");
        lCallExit(1);
    }
    
    // This will set up o_eProf and chipVersion
    if (ProcessProfile() != 0) {
        fprintf(stderr, "Error - unknown profile \"%s\"\n", o_Profile);
        printf("Use: -h for help.\n");
        lCallExit(1);
    }

    if (o_DumpBinaryUCode)
        o_lwinst = 1;
    if (o_DumpText)
        o_FmtFlags |= FMT_FORCE_TEXT_PROFILE;

#ifdef ENABLE_TRACE_CODE
    if (o_verbosityLevel) {
        EnablePrinting = 1;
        TR_TURN_MASK_ON(TR_COP);
    }
    if (o_verbosityLevel >= 4) {
        lwTraceLevel = 100;
    } else if (o_verbosityLevel >= 3) {
        lwTraceLevel = 70;
    } else if (o_verbosityLevel >= 2) {
        lwTraceLevel = 60;
    } else if (o_verbosityLevel >= 1) {
        lwTraceLevel = 50;
    }
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
    if (o_DebugTrap) {
        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_WNDW);
        _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_WNDW);
    }
    if (o_DebugBreakAlloc > 0)
        _CrtSetBreakAlloc(o_DebugBreakAlloc);
#endif

    o_FmtFlags &= ~FMT_PHASE_COUNT_MASK;
    o_FmtFlags |= (o_PhaseCount << FMT_PHASE_COUNT_SHIFT) & FMT_PHASE_COUNT_MASK;

    if (o_EarlyPhase)
        o_FmtFlags |= FMT__EARLY_PHASE_MASK;

    if (o_Binary && 
        _stricmp(o_Profile, "fp30") &&
        _stricmp(o_Profile, "fp31") &&
        _stricmp(o_Profile, "fp34") &&
        _stricmp(o_Profile, "fp35") &&
        _stricmp(o_Profile, "fp38") &&
        _stricmp(o_Profile, "fp36") &&
        _stricmp(o_Profile, "fp40") &&
        _stricmp(o_Profile, "vp40") &&
        _stricmp(o_Profile, "fp47") &&
        _stricmp(o_Profile, "ar20vp") &&
        _stricmp(o_Profile, "ar20fp"))
    {
        fprintf(stderr, "Error - can't specify both -binary and -profile %s\n", o_Profile);
        lCallExit(1);
    }

    if (!o_Binary && !_stricmp(o_Profile, "ar20fp")) {
        fprintf(stderr, "Error - the profile %s requires -binary\n", o_Profile);
        lCallExit(1);
    }

    if (o_oglOpt && o_d3dOpt) {
        fprintf(stderr, "Error - can't specify both -oglOpt and -d3dOpt\n");
        lCallExit(1);
    }
    if (!o_InputFileName && !o_InputListFileName) {
        printf("Missing input file(s)\n");
        printf("Use -h for list of options.\n");
        lCallExit(0);
    }
    if (o_InputFileName && o_InputListFileName) {
        fprintf(stderr, "Error - can't specify both -i and -il\n");
        lCallExit(1);
    }
    if (o_InputListFileName && o_OutputFileName) {
        fprintf(stderr, "Error - can't specify both -il and -o\n");
        fprintf(stderr, "        Output files are determined from input file name\n");
        lCallExit(1);
    }
    if (o_OutputFileName) {
        if (!o_DumpBinaryUCode) {
            if (o_Binary) {
                lOutFile = fopen(o_OutputFileName, "wb");
            } else {
                lOutFile = fopen(o_OutputFileName, "w");
            }
            if (!lOutFile) {
                fprintf(stderr, "Error opening output file: %s\n", o_OutputFileName);
                lCallExit(1);
            }
        }
    }

    if (o_InputFileName) {
        retVal |= compileOnce(o_InputFileName, lOutFile, &pUCode, &pCgBinary, &lCgBinarySize,
                              o_DumpBinaryUCode);
        if (fclose(lOutFile) != 0) {
            fprintf(stderr, "Error closing output file: %s -- %s\n", o_OutputFileName, strerror(errno));
            lCallExit(1);
        }
    } else {
        int ii;
        char *ptr;
        char lListEntry[512], lOutName[512];

        lInListFile = fopen(o_InputListFileName, "rt");
        if (!lInListFile) {
            fprintf(stderr, "Error opening input list file: %s\n", o_InputListFileName);
            lCallExit(1);
        }
        while (fgets(lListEntry, sizeof(lListEntry), lInListFile)) {
            ptr = strchr(lListEntry, '\r');
            if (ptr)
                *ptr = 0;
            ptr = strchr(lListEntry, '\n');
            if (ptr)
                *ptr = 0;
            // leading spaces not allowed, but chop off anything after first space
            ptr = strchr(lListEntry, ' ');
            if (ptr)
                *ptr = 0;
            ptr = strchr(lListEntry, '\t');
            if (ptr)
                *ptr = 0;

            // Skip blank or comment lines.
            if (lListEntry[0] == 0 || lListEntry[0] == '#')
                continue;

            // Get the basename of the input file
            for (ii = (int)strlen(lListEntry); ii >= 0; ii--) {
                if (lListEntry[ii] == '/' || lListEntry[ii] == '\\')
                    break;
            }
            strcpy(lOutName, baseOutputDir);
            strcat(lOutName, &lListEntry[ii + 1]);
            strcat(lOutName, ".opt");

            if (o_Binary) {
                lOutFile = fopen(lOutName, "wb");
            } else {
                lOutFile = fopen(lOutName, "w");
            }
            if (!lOutFile) {
                fprintf(stderr, "Error opening output file: %s -- %s\n", lOutName, strerror(errno));
                lCallExit(1);
            }

            // Catch exceptions when we're compiling multiple inputs
            // so we can get a full set of results.
#if defined(_MSC_VER)
            __try {
#endif
                retVal |= compileOnce(lListEntry, lOutFile, &pUCode, &pCgBinary, 
                                      &lCgBinarySize, o_DumpBinaryUCode);
#if defined(_MSC_VER)
            } __except(!o_DebugTrap) {
                fprintf(stderr, "*** exception during compilation ***\n");
                retVal = 1;
            }
#endif
            if (fclose(lOutFile) != 0) {
                fprintf(stderr, "Error closing output file: %s -- %s\n", lOutName, strerror(errno));
                lCallExit(1);
            }
                
        }
        fclose(lInListFile);
    }
    if (!retVal && o_DumpBinaryUCode) {
        RelocateLWuCodeStruct(pUCode, 0);
        if (o_OutputFileName) {
            DumpLWuCodeBinaryFile(o_OutputFileName, pUCode);
        } else {
            DumpLWuCodeBinaryFile("ucode.bin", pUCode);
        }
    }

    if (commandLine)
        free(commandLine);

    FinalizeProfileStruct();

#if defined(_MSC_VER) && 0
    _CrtDumpMemoryLeaks();
#endif
    return retVal;
} // main

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// End of file parseasm.cpp ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

