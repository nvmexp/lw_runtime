/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to LWPU ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 *
 *  Module name              : gpuCodeGen.c
 *
 *  Last update              :
 *
 *  Description              :
 * 
 */

/*--------------------------------- Includes ---------------------------------*/

#include <stdMessages.h>
#include <stdLocal.h>
#include <stdProcess.h>
#include <stdList.h>
#include <stdCmdOpt.h>
#include <stdFileNames.h>
#include <stdString.h>
#include <stdMap.h>
#include <stdToolInstallation.h>
#include <gpuInfoMessageDefs.h>

//#include <../cglang/frontend/cgclib.h>
//#include <../interface/copi_binding.h>
//#include <../interface/copi_atom_interface.h>
//#include <../interface/copi_base_regs.h>
//#include <../interface/copi_dag_interface.h>
//#include <../interface/copi_lw50_interface.h>

#include "ptxAs.h"
#include "gpuInfo.h"
#include "ptxOptEnums.h"
#include "ptxDCI.h"

#include "g_lwconfig.h"
#if LWCFG(GLOBAL_ARCH_FERMI)
//#include <../interface/copi_fermi_interface.h>
#endif


/*-------------------------------- Defines -----------------------------------*/

#define STRESS_MAXREG_OPTION "stress-maxrregcount"
#define STRESS_NO_CRP_OPTION "stress-no-crp"
#define STRESS_NO_GLOBAL_REGALLOC "stress-noglobalregalloc"
#define LEGACY_CVT_F64_OPTION "legacy-cvtf64"

// FIXME: This information is available in GPU.spec as 'MAX_WARPS' field. Use that
// and get rid of these #defines
#define GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_G8X    24 
#define GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_GT2XX  32
#define GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_GF1XX  48

#define GPU_CODE_GEN_NUM_REGS_PER_LANE_SM_GT2XX  256

/*------------------------------ Local functions -----------------------------*/

/*
 * External function, so that we can replace it with a dummy 
 * when building in release mode:
 */
extern String gpuDisassemble( Byte *image, uInt size, Address vbase, Int architecture );

/*------------------------------ Local functions -----------------------------*/
#if USE_OLD_DISASSEMBLER == 1
#ifdef ASSEMBLER_BUILD
   /*
    * This means that the assembler framework 
    * in //compiler/gpgpu is enabled, which is
    * partilwlarly not the case for driver builds.
    * For compiler tools builds, we can simply disable
    * disassembly support by not releasing the md files:
    */
    #include <asPrint.h>
    #include <asCodeGen.h>
    #include <archMDKeys.h>

    #include <archTeslaInstructionFormat.h>
    static Bool                            TeslaInited;
    static asInstructionFormatDriver       TeslaIF;
    static mdCpuDef                        TeslaMD;

#if LWCFG(GLOBAL_ARCH_FERMI)
    #include <archFermiInstructionFormat.h>
    static Bool                            FermiInited;
    static asInstructionFormatDriver       FermiIF;
    static mdCpuDef                        FermiMD;
#else
    #define FermiIF  NULL
    #define FermiMD  NULL
#endif

#if LWCFG(GLOBAL_ARCH_KEPLER)
    #include <archKeplerInstructionFormat.h>
    static Bool                            KeplerInited;
    static asInstructionFormatDriver       KeplerIF;
    static mdCpuDef                        KeplerMD;
    #include <archKepler2InstructionFormat.h>
    static Bool                            SM4Inited;
    static asInstructionFormatDriver       SM4IF;
    static mdCpuDef                        SM4MD;
#else
    #define KeplerIF  NULL
    #define KeplerMD  NULL
    #define SM4IF     NULL
    #define SM4MD     NULL
#endif
#else 
    #define mdCpuDef                   Pointer
    #define asInstructionFormatDriver  Pointer
    #define SM4IF    NULL
    #define SM4MD    NULL
    #define KeplerIF NULL
    #define KeplerMD NULL
    #define FermiIF  NULL
    #define FermiMD  NULL
    #define TeslaIF  NULL
    #define TeslaMD  NULL
#endif
#else 
    #define mdCpuDef                   Pointer
    #define asInstructionFormatDriver  Pointer
    #define SM4IF    NULL
    #define SM4MD    NULL
    #define KeplerIF NULL
    #define KeplerMD NULL
    #define FermiIF  NULL
    #define FermiMD  NULL
    #define TeslaIF  NULL
    #define TeslaMD  NULL
#endif



    
static Bool initializeAssembler( String profileName )
{
#if USE_OLD_DISASSEMBLER == 1 
#ifdef ASSEMBLER_BUILD
    gpuFeaturesProfile profile = gpuGetFeaturesProfile( profileName );

    if (profile && profile->isaClass) {
        Bool                        *inited = NULL;
        mdCpuDef                    *md     = NULL;
        asInstructionFormatDriver   *ift    = NULL;
        uInt32                       mdKey  = 0;
        
        if (stdEQSTRING(profile->isaClass,"Tesla")) {
            inited = &TeslaInited;
            md     = &TeslaMD;
            ift    = &TeslaIF;
            mdKey  = archTeslaKey;
        } else
#if LWCFG(GLOBAL_ARCH_FERMI)
        if (stdEQSTRING(profile->isaClass,"Fermi")) {
            inited = &FermiInited;
            md     = &FermiMD;
            ift    = &FermiIF;
            mdKey  = archFermiKey;
        } else
#endif
#if LWCFG(GLOBAL_ARCH_KEPLER)
        if (stdEQSTRING(profile->isaClass,"Kepler")&& stdEQSTRING(profile->internalName,"sm_30")) {
            inited = &KeplerInited;
            md     = &KeplerMD;
            ift    = &KeplerIF;
            mdKey  = archKeplerKey;
            md     = &FermiMD; /* for now */
            ift    = &FermiIF; /* for now */
        } else 
#endif
#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
        if (stdEQSTRING(profile->isaClass,"Kepler") && stdEQSTRING(profile->internalName,"sm_32")) {
            inited = &SM4Inited;
            md     = &SM4MD;
            ift    = &SM4IF;
            mdKey  = archKepler2Key;
            md     = &FermiMD; /* for now */
            ift    = &FermiIF; /* for now */
        } else
#endif
#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
        if (stdEQSTRING(profile->isaClass,"Kepler") && stdEQSTRING(profile->internalName,"sm_35")) {
            inited = &SM4Inited;
            md     = &SM4MD;
            ift    = &SM4IF;
            mdKey  = archKepler2Key;
            md     = &FermiMD; /* for now */
            ift    = &FermiIF; /* for now */
        } else 
#endif
#if LWCFG(GLOBAL_GPU_IMPL_GK110C)
        if (stdEQSTRING(profile->isaClass,"Kepler") && stdEQSTRING(profile->internalName,"sm_37")) {
            inited = &SM4Inited;
            md     = &SM4MD;
            ift    = &SM4IF;
            mdKey  = archKepler2Key;
            md     = &FermiMD; /* for now */
            ift    = &FermiIF; /* for now */
        } else 
#endif
#if LWCFG(GLOBAL_ARCH_MAXWELL)
        /* FIXME: Need to update below instructions, once we have Maxwell setup ready */
        if (stdEQSTRING(profile->isaClass,"Maxwell") && stdEQSTRING(profile->internalName,"sm_50")) {
            inited = &SM4Inited; /* for now */
            md     = &SM4MD; /* for now */
            ift    = &SM4IF; /* for now */
            mdKey  = archKepler2Key; /* for now */
            md     = &FermiMD; /* for now */
            ift    = &FermiIF; /* for now */
        } else 
#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
        /* FIXME: Need to update below instructions, once we have Maxwell SM20X ready */
        if (stdEQSTRING(profile->isaClass,"Maxwell") && stdEQSTRING(profile->internalName,"sm_52")) {
            inited = &SM4Inited; /* for now */
            md     = &SM4MD; /* for now */
            ift    = &SM4IF; /* for now */
            mdKey  = archKepler2Key; /* for now */
            md     = &FermiMD; /* for now */
            ift    = &FermiIF; /* for now */
        } else if (stdEQSTRING(profile->isaClass,"Maxwell") && stdEQSTRING(profile->internalName,"sm_53")) {
            inited = &SM4Inited; /* for now */
            md     = &SM4MD; /* for now */
            ift    = &SM4IF; /* for now */
            mdKey  = archKepler2Key; /* for now */
            md     = &FermiMD; /* for now */
            ift    = &FermiIF; /* for now */
        } else 
#endif
#endif
        {
            stdASSERT(False,("Unknown ISA: '%s'", profile->isaClass));
        }
        
        
        if (*inited) { 
            return (*md) != NULL; 
        } else {
            String exec, sdkInstallPath, mdFileName;
            
            exec       = procLwrrentExelwtableName();
            fnamDecomposePath(exec,&sdkInstallPath,NULL,NULL);
#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
            if (stdEQSTRING(profile->internalName,"sm_32")) {
                mdFileName = fnamComposePath(sdkInstallPath, "Kepler2", "md");
            } else
#endif
#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
            if (stdEQSTRING(profile->internalName,"sm_35")) {
                mdFileName = fnamComposePath(sdkInstallPath, "Kepler2", "md");
            } else 
#endif
#if LWCFG(GLOBAL_ARCH_MAXWELL)
            /* FIXME: Need to update below instruction, once we have Maxwell setup ready */
            if (stdEQSTRING(profile->internalName,"sm_50")) {
                mdFileName = fnamComposePath(sdkInstallPath, "Kepler2", "md"); /* for now */
            } else 
#endif
            {
                mdFileName = fnamComposePath(sdkInstallPath, profile->isaClass, "md");
            }
           *inited = True;

           *md = mdParse( mdFileName, mdKey, False, True );
            if (!*md) { return False; }

            if (stdEQSTRING(profile->isaClass,"Tesla")) {
                TeslaIF= archTeslaIF(*md);
            } else
#if LWCFG(GLOBAL_ARCH_FERMI)
            if (stdEQSTRING(profile->isaClass,"Fermi")) {
                FermiIF= archFermiIF(*md);
            } else 
#endif
#if LWCFG(GLOBAL_ARCH_KEPLER)
            if (stdEQSTRING(profile->isaClass,"Kepler")&& stdEQSTRING(profile->internalName,"sm_30")) {
               KeplerIF= archKeplerIF(*md);
               FermiIF= KeplerIF; /* for now */
            } else 
#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
            if (stdEQSTRING(profile->isaClass,"Kepler") && stdEQSTRING(profile->internalName,"sm_32")) {
                SM4IF= archSM4IF(*md);
                FermiIF= SM4IF; /* for now */
            } else
#endif
#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
            if (stdEQSTRING(profile->isaClass,"Kepler") && stdEQSTRING(profile->internalName,"sm_35")) {
                SM4IF= archSM4IF(*md);
                FermiIF= SM4IF; /* for now */
            } else 
#endif
#endif
#if LWCFG(GLOBAL_ARCH_MAXWELL)
            /* FIXME: Need to update below instructions, once we have Maxwell setup ready */
            if (stdEQSTRING(profile->isaClass,"Maxwell") && stdEQSTRING(profile->internalName,"sm_50")) {
                SM4IF= archSM4IF(*md); /* for now */
                FermiIF= SM4IF; /* for now */
            } else 
#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
            /* FIXME: Need to update below instructions, once we have Maxwell setup ready */
            if (stdEQSTRING(profile->isaClass,"Maxwell") && stdEQSTRING(profile->internalName,"sm_52")) {
                SM4IF= archSM4IF(*md); /* for now */
                FermiIF= SM4IF; /* for now */
            } else if (stdEQSTRING(profile->isaClass,"Maxwell") && stdEQSTRING(profile->internalName,"sm_53")) {
                SM4IF= archSM4IF(*md); /* for now */
                FermiIF= SM4IF; /* for now */
            } else 
#endif // LWCFG(GLOBAL_GPU_FAMILY_GM20X)
#endif // LWCFG(GLOBAL_ARCH_MAXWELL)
            {
                stdASSERT(False,("Unknown ISA: '%s'", profile->isaClass));
            }

          (*ift)->init(*md,(*ift)->data);
        }
          
        return True;
    }
#endif
#endif
    return False;
}
    
static String disassemble( Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded,
                           Int sassArch, String isaName, mdCpuDef md, asInstructionFormatDriver ift )
{
#ifndef ASSEMBLER_BUILD
    return NULL;
#elif USE_OLD_DISASSEMBLER != 1
    return NULL;
#else
    stdString_t       result  = stringNEW();
    uInt              dummy;
    asPrintParameters parms;

    stdMEMCLEAR(&parms);
    parms.operationSeparator = "\t";
    parms.outFile            = wtrCreateFileWriter(stdout);
    parms.printAddress       = True;
    parms.image              = image;

    if (!decoded) { decoded= &dummy; }

    if (oneShot) {
        struct asInstructionRec    instruction;
        asOperation op;
        
        stdMEMCLEAR(&instruction);
        
        instruction.header.address = vbase;
       *decoded                    = ift->decode(md,image,size,&instruction,ift->data);
    
        op = instruction.operations[0];
        
        if (op) {
            asPrintOperation( op, &parms );
        
            stdFREE(op->arguments);
            stdFREE(op);
        }
        
    } else {
        asObject obj= asDecodeCode(md, ift, vbase, image, size);
    
       *decoded = 0;

        asPrintObject ( obj, &parms );
        asDeleteObject( obj          );
    }
    
    wtrDelete( parms.outFile );

    return stringStripToBuf( result );
#endif
}

static Bool lParseBool(String fOption, String fValue)
{
    if (stdEQSTRING(fValue, "0") || stdEQSTRING(fValue, "false")) {
        return False;
    } else if (stdEQSTRING(fValue, "1") || stdEQSTRING(fValue, "true")) {
        return True;
    } else {
        stdCHECK(0, (gpuinfMsgIlwalidOptiolwalue, fValue, fOption));
        return False;
    }
}

static uInt lParseInt(String fOption, String fValue, uInt fLo, uInt fHi)
{
    uInt lVal;
    if (sscanf(fValue, "%u", &lVal) != 1 || lVal < fLo || lVal > fHi) {
        lVal = 0;
        stdCHECK(0, (gpuinfMsgIlwalidOptiolwalue, fValue, fOption));
    }
    return lVal;
}

/*-------------------------------- G80 Driver --------------------------------*/

static void sm1x_getFatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, 
                                  unsigned int *addrRegFatpoint, 
                                  unsigned int *genRegFatpoint)
{
    LdParams_lw50* ld = (LdParams_lw50 *)Ld;

    if(ld->perfStatsMask & I_LW50_PERFSTAT_REGFATPOINT)  {
        *ccRegFatpoint = ld->stats.outFatPointCount[I_LW50_REGCLASS_CC];
        *addrRegFatpoint = ld->stats.outFatPointCount[I_LW50_REGCLASS_A]; 
        *genRegFatpoint = ld->stats.outFatPointCount[I_LW50_REGCLASS_R];
    } 
}

static void sm1x_getStatsMask(String str, unsigned int *num) 
{
    if(str && !strcmp(str,"reg-fatpoint")) {
        *num = *num | I_LW50_PERFSTAT_REGFATPOINT;
    } else {
        *num = *num | I_LW50_PERFSTAT_NOSTATS;
    }
}

static void sm1x_setCallInfo(Pointer Ld, Pointer fCallInfo)
{
    LdParams_lw50* ld = (LdParams_lw50 *)Ld;
    ld->pCallInfo = (IPCallInfo *) fCallInfo;
}

typedef struct {
    LdParams *ld;
    gpuFeaturesProfile gpuProfile;
    CmdLineOptions *cl_options;
}  ProfileOptions_Param;

static void sm_1xProfileOptions(String fOption, String fValue, ProfileOptions_Param *fParam)
{
    static Bool StressWarned = False;

    if (stdEQSTRING(fOption, STRESS_MAXREG_OPTION) || stdEQSTRING(fOption, STRESS_NO_CRP_OPTION)) {
        // Ignore stress reg alloc options for sm_1x profiles
        stdCHECK(!fParam->cl_options->verboseMode || StressWarned, 
                 (gpuinfMsgIgnoredOption, fOption, fParam->cl_options->gpuName));
        StressWarned = True;
    } else if (stdEQSTRING(fOption, LEGACY_CVT_F64_OPTION)) {
        if (lParseInt(fOption, fValue, 0, 1)) {
            fParam->cl_options->legacyCvtF64 = True;
        }
    } else {
        // no profile options supported for sm_1x lwrrently
        stdCHECK(0, (gpuinfMsgUnknownOption, fOption));
    }
}

static LdParams *sm_1xLdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    LdParams_lw50cp *ld;
    IMemPool *lPool = (IMemPool *)fPool;
    CmdLineOptions *lOptions = (CmdLineOptions *) fOptions;
    ProfileOptions_Param lParam;
    ptxDCIHandle dciHandle;

    ld = (LdParams_lw50cp *) NewLdParams_lw50cp(lPool);
    if (lOptions->RetAtEnd) {
        ld->base.RetAtEnd = 1;
    }
    
    dciHandle = ptxGetDCIHandle();
    sm1x_getStatsMask(lOptions->perfStats, &ld->base.perfStatsMask);
    ld->FirstUserGRFAllocation = 16;
    ld->base.base.hwFlags = LW50CP_TESLA;
    ld->base.base.optimizerConstBank = dciHandle->GetOCGConstBank();
    if (lOptions->profileOptions) {
        lParam.ld = (LdParams *)ld;
        lParam.gpuProfile = profile;
        lParam.cl_options = lOptions;
        mapTraverse (lOptions->profileOptions, (stdPairFun) sm_1xProfileOptions, &lParam);
    }
    return &(ld->base.base);
}

/*
 * sm_1xComputeMaxRReg() : Computes the maxregcount value as follows:
 *        We have
 *            maxThreadsPerCta = ntid[0] * ntid[1] * ntid[2]
 *            maxRegsPerSM     = max available registers per sm 
 *                              (e.g. 8K on tesla for arch lower than GT200)
 *        Compute
 *            maxRegsPerCta = maxRegsPerSM / minnctapersm
 *        (i.e. max regs that can be allocated to a cta)
 *
 *        Refine this value by rounding it down based on the reg allocation unit.
 *        This is required because at runtime, the number of ctas that can run on an SM
 *        are determined by regs per cta rounded off to a multiple of reg unit.
 *        (e.g. reg unit is 512 for arch > GT200)
 *        Hence
 *            maxRegsPerCta = regAllolwnit * (regsPerCta / regAllolwnit)
 *        Then
 *            maxrregcount = maxRegsPerCta / maxThreadsPerCta
 *        Finally take into account the register alignment (maxrregcount must be a 
 *        multiple of the regalignment value) by rounding maxrregcoutn down
 *            maxrregcount = regalign * (maxrregcount / regalign)
 */

static int sm_1xComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM) 
{
    uInt maxRegsPerCta, newMaxRReg, newMaxThreadsPerCta, numWarpsNeeded;                                     
    int maxRegsPerSM;                     // arch-specific limits
    int regAllolwnit, regAlignment;       // arch-specific values

    // regfilesize is in KB. regsize is 4 bytes.
    maxRegsPerSM = gpuInfo->regFileSize / 4;
    regAllolwnit = gpuInfo->regAllolwnit;
    regAlignment = gpuInfo->regAlignment;
    maxRegsPerCta = maxRegsPerSM / minNCtaPerSM;
    maxRegsPerCta = regAllolwnit * (maxRegsPerCta / regAllolwnit);       
    // reg file is 64regs wide and  maxRegsPerSM/64 tall.
    newMaxThreadsPerCta = stdROUNDUP(maxThreadsPerCta, 64);
    numWarpsNeeded = (newMaxThreadsPerCta / gpuInfo->warpSize) * minNCtaPerSM;
    if (numWarpsNeeded > gpuInfo->maxWarps)
        return 0; // return 0 for invalid configuration.

    newMaxRReg = maxRegsPerCta / newMaxThreadsPerCta;
    newMaxRReg = regAlignment * (newMaxRReg / regAlignment);
    return newMaxRReg;
}

static Bool sm_10CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cp50_optx");
    return GenerateCodeNew_lw50cp_ucode(ld);
}

static Bool sm_10TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;
    
    ld->profileName      = AddIAtom(ld->atable, "cp50");
    return GenerateCodeNew_lw50cp(ld);
}

static void* sm_10LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    LdParams *ld;

    ld = sm_1xLdParamsGen(profile, fOptions, fPool);
    ld->archVariant      = COP_ARCH_VARIANT_TESLA_SM1_0;
    ld->numThreadIdPerSM = GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_G8X;
    return ld;
}

static PtxArch sm_10ptxArch = PTX_ARCH_SM_10;

static String sm_10Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return disassemble(image,size,vbase,oneShot,decoded,COP_ARCH_VARIANT_TESLA_SM1_0,"Tesla",TeslaMD,TeslaIF);
}

static void sm_10MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    ptxDCIHandle dciHandle;

    dciHandle = ptxGetDCIHandle();
    *lmemMax = dciHandle->GetMaxLocalMemory();
    *smemMax = dciHandle->GetMaxSharedMemory();
    *cmemMax = dciHandle->GetConstantBankSize();
}

static void sm_10FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, 
                               unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm1x_getFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static int sm_10ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_1xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

/*-------------------------------- G8x Driver --------------------------------*/

static Bool sm_11CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cp50_optx");
    return GenerateCodeNew_lw50cp_ucode(ld);
}

static Bool sm_11TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;
    
    ld->profileName      = AddIAtom(ld->atable, "cp50");
    return GenerateCodeNew_lw50cp(ld);
}

static void* sm_11LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    LdParams *ld;

    ld = sm_1xLdParamsGen(profile, fOptions, fPool);
    ld->archVariant      = COP_ARCH_VARIANT_TESLA_SM1_1;
    ld->numThreadIdPerSM = GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_G8X;
    return ld;
}

static PtxArch sm_11ptxArch = PTX_ARCH_SM_11;

static String sm_11Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return disassemble(image,size,vbase,oneShot,decoded,COP_ARCH_VARIANT_TESLA_SM1_1,"Tesla",TeslaMD,TeslaIF);
}


static void sm_11MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    ptxDCIHandle dciHandle;

    dciHandle = ptxGetDCIHandle();
    *lmemMax = dciHandle->GetMaxLocalMemory();
    *smemMax = dciHandle->GetMaxSharedMemory();
    *cmemMax = dciHandle->GetConstantBankSize();
}

static void sm_11FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, 
                               unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm1x_getFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static int sm_11ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_1xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

/*------------------------------- GT216/218 Driver ---------------------------*/

static Bool sm_12CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName  = AddIAtom(ld->atable, "cp50_optx");
    return GenerateCodeNew_lw50cp_ucode(ld);
}

static Bool sm_12TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;
    
    ld->profileName  = AddIAtom(ld->atable, "cp50");
    return GenerateCodeNew_lw50cp(ld);
}

static void* sm_12LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    LdParams *ld;

    ld = sm_1xLdParamsGen(profile, fOptions, fPool);
    ld->archVariant      = COP_ARCH_VARIANT_TESLA_SM1_5;
    ld->numThreadIdPerSM = GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_GT2XX;
    ld->rRegBankSize     = GPU_CODE_GEN_NUM_REGS_PER_LANE_SM_GT2XX;

    return ld;
}

static PtxArch sm_12ptxArch = PTX_ARCH_SM_12;

static String sm_12Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return disassemble(image,size,vbase,oneShot,decoded,COP_ARCH_VARIANT_TESLA_SM1_5,"Tesla",TeslaMD,TeslaIF);
}

static void sm_12MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    ptxDCIHandle dciHandle;

    dciHandle = ptxGetDCIHandle();
    *lmemMax = dciHandle->GetMaxLocalMemory();
    *smemMax = dciHandle->GetMaxSharedMemory();
    *cmemMax = dciHandle->GetConstantBankSize();
}


static void sm_12FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, 
                               unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm1x_getFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static int sm_12ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_1xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

/*------------------------------- GT200/212/214 Driver -----------------------*/

static Bool sm_13CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer  *fOptions )
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cp50_optx");
    return GenerateCodeNew_lw50cp_ucode(ld);
}

static Bool sm_13TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer  *fOptions )
{
    LdParams *ld= (LdParams *)ldParms;
    
    ld->profileName      = AddIAtom(ld->atable, "cp50");
    return GenerateCodeNew_lw50cp(ld);
}

static void* sm_13LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    LdParams *ld;

    ld = sm_1xLdParamsGen(profile, fOptions, fPool);
    ld->archVariant      = COP_ARCH_VARIANT_TESLA_SM1_3;
    ld->numThreadIdPerSM = GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_GT2XX;
    ld->rRegBankSize     = GPU_CODE_GEN_NUM_REGS_PER_LANE_SM_GT2XX;
    return ld;
}
static PtxArch sm_13ptxArch = PTX_ARCH_SM_13;

static String sm_13Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return disassemble(image,size,vbase,oneShot,decoded,COP_ARCH_VARIANT_TESLA_SM1_3,"Tesla",TeslaMD,TeslaIF);
}


static void sm_13MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    ptxDCIHandle dciHandle;

    dciHandle = ptxGetDCIHandle();
    *lmemMax = dciHandle->GetMaxLocalMemory();
    *smemMax = dciHandle->GetMaxSharedMemory();
    *cmemMax = dciHandle->GetConstantBankSize();
}

static void sm_13FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, 
                               unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm1x_getFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static int sm_13ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_1xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

/*------------------------------- Fermi Driver -------------------------------*/

#if LWCFG(GLOBAL_ARCH_FERMI)

// Generic fermi codegen functions
static Bool sm_2xCodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cpf_optx");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 1))
            return 1;
    }
    return GenerateCode_Fermi_cp(ld, 1);
}

static Bool sm_2xTextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions )
{
    LdParams *ld= (LdParams *)ldParms;
    
    ld->profileName      = AddIAtom(ld->atable, "cpf");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 0))
            return 1;
    }
    return GenerateCode_Fermi_cp(ld, 0);
}

static void sm_2xFatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    msgReport(gpuinfMsgUnsupportedOption, "sm_20", "--perf-stats");
}

static void sm_2xProfileOptions(String fOption, String fValue, ProfileOptions_Param *fParam)
{
    LdParams *fLd = fParam->ld;
    LdParams_Fermi *lLdParams_Fermi;

    if (stdEQSTRING(fOption, "lds128colwert")) {
        lLdParams_Fermi = (LdParams_Fermi *) fLd;
        if (stdEQSTRING(fValue, "always")) {
            lLdParams_Fermi->lds128OptiMode = FERMI_LDS128_OPTI_ALWAYS;
        } else if (stdEQSTRING(fValue, "nonconst")) {
            lLdParams_Fermi->lds128OptiMode = FERMI_LDS128_OPTI_NONCONST;
        } else if (stdEQSTRING(fValue, "never")) {
            lLdParams_Fermi->lds128OptiMode = FERMI_LDS128_OPTI_NEVER;
        } else {
            stdCHECK(0, (gpuinfMsgIlwalidOptiolwalue, fValue, fOption));
        }
    } else if (stdEQSTRING(fOption, STRESS_NO_CRP_OPTION)) {
        if (lParseBool(fOption, fValue)) {
            fLd->stressRegAlloc |= STRESS_REG_ALLOC_NO_CRP;
        }
    } else if (stdEQSTRING(fOption, STRESS_MAXREG_OPTION)) {
        fLd->stressRegAlloc |= STRESS_REG_ALLOC_FORCE_MAXREG;
        fLd->maxRRegsAllowed = lParseInt(fOption, fValue, 1, fParam->gpuProfile->maxRegsPerThread);
    } else if (stdEQSTRING(fOption, STRESS_NO_GLOBAL_REGALLOC)) {
        if (lParseBool(fOption, fValue)) {
            fLd->stressRegAlloc |= STRESS_REG_NO_GLOBAL_REG_ALLOC;
        }
    } else if (stdEQSTRING(fOption, LEGACY_CVT_F64_OPTION)) {
        // Ignore legacy-cvtf64 option for sm_2x profiles
        stdCHECK(!fParam->cl_options->verboseMode || fParam->cl_options->legacyCvtF64, 
                 (gpuinfMsgIgnoredOption, fOption, fParam->cl_options->gpuName));
        fParam->cl_options->legacyCvtF64 = True;
#ifndef RELEASE
    } else if (stdEQSTRING(fOption, "tepid")) {
        if (lParseBool(fOption, fValue))
            fLd->fmtFlags |= FMT_TEPID_LATENCY;
    } else if (stdEQSTRING(fOption, "lat")) {
        fLd->fmtFlags &= FMT_LATENCY_MASK;
        switch (lParseInt(fOption, fValue, 0, 3)) {
        default:
        case 0: fLd->fmtFlags |= FMT_LATENCY_0; break;
        case 1: fLd->fmtFlags |= FMT_LATENCY_1; break;
        case 2: fLd->fmtFlags |= FMT_LATENCY_2; break;
        case 3: fLd->fmtFlags |= FMT_LATENCY_3; break;
        }
    } else if (stdEQSTRING(fOption, "tbat")) {
        fLd->fmtFlags &= FMT_TEX_BATCH_MASK;
        switch (lParseInt(fOption, fValue, 0, 7)) {
        default:
        case 0: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_0; break;
        case 1: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_1; break;
        case 2: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_2; break;
        case 3: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_3; break;
        case 4: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_4; break;
        case 5: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_5; break;
        case 6: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_6; break;
        case 7: fLd->fmtFlags |= FMT_TEX_BATCH_SIZE_7; break;
        }
    } else if (stdEQSTRING(fOption, "ld")) {
        if (lParseBool(fOption, fValue))
            fLd->fmtFlags |= FMT_FLAG_SHOW_LIVEDEAD;
    } else if (stdEQSTRING(fOption, "freq")){
        if (lParseBool(fOption, fValue))
            fLd->printInstFrequency = 1;
    } else if (stdEQSTRING(fOption, "tasks")) {
        fLd->tasks = fValue;
    } else if (stdEQSTRING(fOption, "nopred")) {
        if (lParseBool(fOption, fValue)) {
            fLd->optFlags.CanIfCvt = 0;
        } else {
            fLd->optFlags.CanIfCvt = 1;
        }
    } else if (stdEQSTRING(fOption, "mdes")) {
        fLd->mdesFileName = fValue;
    } else if (stdEQSTRING(fOption, "dpddlk_war")) {
        lLdParams_Fermi = (LdParams_Fermi *) fLd;
        lLdParams_Fermi->DoDPDeadlockWAR_SW609198 = lParseBool(fOption, fValue) ? 1 : 0;
    } else if (stdEQSTRING(fOption, "HW665749_war")) {
        lLdParams_Fermi = (LdParams_Fermi *) fLd;
        if (stdEQSTRING(fValue, "all")) {
            lLdParams_Fermi->HW665749_War_Mode = HW665749_WAR_ALL;
        } else if (stdEQSTRING(fValue, "branched")) {
            lLdParams_Fermi->HW665749_War_Mode = HW665749_WAR_BRANCHED;
        } else {
            stdCHECK(0, (gpuinfMsgIlwalidOptiolwalue, fValue, fOption));
        }
    } else if (stdEQSTRING(fOption, "SW712753_war")) {
        lLdParams_Fermi = (LdParams_Fermi *) fLd;
        lLdParams_Fermi->Apply_SW712753_War = lParseBool(fOption, fValue);
#endif
    } else {
        stdCHECK(0, (gpuinfMsgUnknownOption, fOption));
    }
}

static void* sm_2xLdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool, int archVariant)
{
    LdParams *ld;
    LdParams_Fermi_cp *lLdParams_Fermi;
    IMemPool *lPool = (IMemPool *)fPool;
    CmdLineOptions *lOptions = (CmdLineOptions *) fOptions;
    ProfileOptions_Param lParam;
    ptxDCIHandle dciHandle = ptxGetDCIHandle();

    ld = NewLdParams_Fermi_cp(lPool);
    ld->hwFlags           = GF100CP_FERMI;
    ld->archVariant       = archVariant;
    ld->numThreadIdPerSM  = GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_GF1XX;
    ld->FlushDenormToZero = 0;
    if (lOptions->CompileAsSyscall) {
        ld->optimizerConstBank = -1;
    } else {
        ld->optimizerConstBank = dciHandle->GetOCGConstBank();
    }
    lLdParams_Fermi = (LdParams_Fermi_cp *) ld;
    lLdParams_Fermi->base.PreciseShortIntMath = 1;
    lLdParams_Fermi->UseDCI = 1;
    if (lOptions->profileOptions) {
        lParam.ld = ld;
        lParam.gpuProfile = profile;
        lParam.cl_options = lOptions;
        mapTraverse (lOptions->profileOptions, (stdPairFun) sm_2xProfileOptions, &lParam);
    }

    if (lOptions->noFastRegAlloc) {
        lLdParams_Fermi->base.EnableFastRegAlloc = 0;
    }

    return ld;
}

static String sm_2xDisassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded, 
                                int archVariant)
{
    return disassemble(image,size,vbase,oneShot,decoded,archVariant,"Fermi",FermiMD,FermiIF);
}

static void sm_2xMemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    ptxDCIHandle dciHandle;

    dciHandle = ptxGetDCIHandle();
    *lmemMax = dciHandle->GetMaxLocalMemory();
    *smemMax = dciHandle->GetMaxSharedMemory();
    *cmemMax = dciHandle->GetConstantBankSize();
}

/*
 * sm_2xcomputeMaxRReg() : Computes the maxregcount value as follows:
 *        We have
 *            maxThreadsPerCta = ntid[0] * ntid[1] * ntid[2]
 *            maxRegsPerSM     = max available registers per sm 
 *                              (e.g. 8K on tesla for arch lower than GT200)
 *        Compute
 *            maxRegsPerCta = maxRegsPerSM / minnctapersm
 *        (i.e. max regs that can be allocated to a cta)
 *
 *        Refine this value by rounding it down based on the reg allocation unit.
 *        This is required because at runtime, the number of ctas that can run on an SM
 *        are determined by regs per cta rounded off to a multiple of reg unit.
 *        (e.g. reg unit is 512 for arch > GT200)
 *        Hence
 *            maxRegsPerCta = regAllolwnit * (regsPerCta / regAllolwnit)
 * 
 *        Round up the thread count so that number of warps is aligned to the required alignment.
 *
 *          warpSize = info->gpuInfo->warpSize; // number of threads in each warp
 *          numWarps = (maxThreadsPerCta + (warpSize - 1)) / warpSize;
 *          numWarps = ROUNDUP(numWarps, warpAlign);
 *          maxThreadsPerCta = numWarps * warpSize;
 *
 *        Then
 *            maxrregcount = maxRegsPerCta / maxThreadsPerCta
 *        Then take into account the register alignment (maxrregcount must be a 
 *        multiple of the regalignment value, by rounding this value down
 *            maxrregcount = regalign * (maxrregcount / regalign)
 *
 *        Finally, there are some special checks for Fermi as dolwmented in bug 607054. These 
 *        checks were suggested by Gentaro Hirota.
 */
static int sm_2xComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM) 
{
    uInt newMaxRReg;
    uInt maxWarpsPerCta, numWarpPairs;
    uInt maxRegsPerCta, maxRegsPerSM, numRegPairs;        // arch-specific limits
    uInt regAllolwnit, regAlignment, warpSize, warpAlign;        // arch-specific values

    warpSize = gpuInfo->warpSize; // number of threads in each warp
    warpAlign = gpuInfo->warpAlign;
    regAllolwnit = gpuInfo->regAllolwnit;
    regAlignment = gpuInfo->regAlignment;

    // regfilesize is in KB. regsize is 4 bytes.
    maxRegsPerSM = gpuInfo->regFileSize / 4;
    maxRegsPerCta = stdROUNDDOWN(maxRegsPerSM / minNCtaPerSM, regAllolwnit);

    // Assume each CTA can only execute even number of warps
    maxWarpsPerCta = (maxThreadsPerCta + (warpSize - 1)) / warpSize;
    maxWarpsPerCta = stdROUNDUP(maxWarpsPerCta, warpAlign);
    maxThreadsPerCta = maxWarpsPerCta * warpSize;

    if (maxWarpsPerCta * minNCtaPerSM > gpuInfo->maxWarps) 
    {
        return 0;    // Return 0 for invalid configuration
    }

    newMaxRReg = stdROUNDDOWN(maxRegsPerCta / maxThreadsPerCta, regAlignment);

    numWarpPairs = maxWarpsPerCta / 2;
    numRegPairs = (newMaxRReg + 1) / 2;
    if ((numRegPairs == 11 && numWarpPairs > 22) || (numRegPairs == 15 && numWarpPairs > 16)) {
        newMaxRReg = (numRegPairs - 1) * 2;
    }

    // bug:869447 - avoid allocating 21/22/29/30/37/38/45/46 registers, only checking for even values
    // as odd values are rounded-off
    switch (newMaxRReg) {
    case 22:
    case 30:
    case 38:
    case 46:
        newMaxRReg = stdROUNDDOWN(newMaxRReg, 4);
        break;
    default:
        break;
    }

    return newMaxRReg;
}

//generic kepler functions

/*
 * sm_3xcomputeMaxRReg() : Computes the maxregcount value as follows:
 *        We have
 *            maxThreadsPerCta = ntid[0] * ntid[1] * ntid[2]
 *            maxRegsPerSM     = max available registers per sm 
 *                              (e.g. 8K on tesla for arch lower than GT200)
 *            maxRegsPerQuad   = max available registers per quadrant, 
 *            maxWarpRegsPerQuad = max available warp-registers per quad (kepler has 4 quadrants), 
 *                                  A warp-register is 32 threads worth of 32-bit scalar registers. 
 *                                  i.e. 128B
 *
 *        Warps can be aligned per SM for Kepler, bug:850939
 *            numWarpsPerCta = (maxThreadsPerCta + (warpSize - 1)) / warpSize;
 *            numWarpsPerSM = maxWarpsPerCta * minNCtaPerSM;
 *
 *          Round up the warp/quad count
 *            maxWarpsPerQuad = (maxWarpsPerSM + 3) / 4;
 *
 *          Then taking into account the register alignment (maxrregcount must be a 
 *          multiple of the regalignment value), by rounding this value down:
 *            maxrregcount  = stdROUNDDOWN(maxWarpRegsPerQuad / numWarpsPerQuad, regAlignment);
 *
 */
static int sm_3xComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM) 
{
    uInt newMaxRReg;
    uInt maxWarpsPerCta, numWarpsPerSM, numWarpsPerQuad;
    uInt maxRegsPerSM, maxWarpRegsPerQuad; // arch-specific limits
    uInt regAlignment, warpSize; // arch-specific values
    const uInt numQuads = 4;

    warpSize = gpuInfo->warpSize; // number of threads in each warp
    regAlignment = gpuInfo->regAlignment;

    // regfilesize is in KB. regsize is 4 bytes.
    maxRegsPerSM = gpuInfo->regFileSize / 4;
    maxWarpRegsPerQuad = maxRegsPerSM / (numQuads * warpSize);

    // Assume each CTA can only execute even number of warps
    maxWarpsPerCta = (maxThreadsPerCta + (warpSize - 1)) / warpSize;
    numWarpsPerSM = maxWarpsPerCta * minNCtaPerSM;
    numWarpsPerQuad = stdROUNDUP(numWarpsPerSM, numQuads) / numQuads;
    
    if (numWarpsPerQuad * numQuads > gpuInfo->maxWarps) 
    {
        return 0;    // Return 0 for invalid configuration
    }

    newMaxRReg = stdROUNDDOWN(maxWarpRegsPerQuad / numWarpsPerQuad, regAlignment);
    return newMaxRReg;
}

/* --------------------------------  SM20  ---------------------------------------- */

static PtxArch sm_20ptxArch = PTX_ARCH_SM_20;

static Bool sm_20CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xCodeGen(info, ldParms, fOptions);
}

static Bool sm_20TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xTextGen(info, ldParms, fOptions);
}

static void sm_20FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_20LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_FERMI_SM2_0);
}

static String sm_20Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_FERMI_SM2_0);
}

static void sm_20MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_20ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_2xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

/* --------------------------------  SM21  ---------------------------------------- */

static PtxArch sm_21ptxArch = PTX_ARCH_SM_21;

static Bool sm_21CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xCodeGen(info, ldParms, fOptions);
}

static Bool sm_21TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xTextGen(info, ldParms, fOptions);
}

static void sm_21FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_21LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_FERMI_SM2_1);
}

static String sm_21Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_FERMI_SM2_1);
}

static void sm_21MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_21ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_2xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

#endif // #if LWCFG(GLOBAL_ARCH_FERMI)

#if LWCFG(GLOBAL_ARCH_KEPLER)
/* --------------------------------  SM30  ---------------------------------------- */

static PtxArch sm_30ptxArch = PTX_ARCH_SM_30;

static Bool sm_30CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xCodeGen(info, ldParms, fOptions);
}

static Bool sm_30TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xTextGen(info, ldParms, fOptions);
}

static void sm_30FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_30LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_KEPLER_SM3_0);
}

static String sm_30Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_KEPLER_SM3_0);
}


static void sm_30MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_30ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_3xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}
#endif // #if LWCFG(GLOBAL_ARCH_KEPLER)

#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 

/* --------------------------------  SM32  ---------------------------------------- */

static PtxArch sm_32ptxArch = PTX_ARCH_SM_32;

static Bool sm_32CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xCodeGen(info, ldParms, fOptions);
}

static Bool sm_32TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xTextGen(info, ldParms, fOptions);
}

static void sm_32FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_32LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_KEPLER_SM3_2);
}

static String sm_32Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_KEPLER_SM3_2);
}


static void sm_32MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_32ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_3xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

#endif // #if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 

#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)

/* --------------------------------  SM35  ---------------------------------------- */

static PtxArch sm_35ptxArch = PTX_ARCH_SM_35;

static Bool sm_35CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xCodeGen(info, ldParms, fOptions);
}

static Bool sm_35TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xTextGen(info, ldParms, fOptions);
}

static void sm_35FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_35LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_KEPLER_SM4_0);
}

static String sm_35Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_KEPLER_SM4_0);
}

static void sm_35MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_35ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_3xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

#endif // #if LWCFG(GLOBAL_GPU_FAMILY_GK11X)

#if LWCFG(GLOBAL_GPU_IMPL_GK110C)

/* --------------------------------  SM37  ---------------------------------------- */

static PtxArch sm_37ptxArch = PTX_ARCH_SM_37;

static Bool sm_37CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xCodeGen(info, ldParms, fOptions);
}

static Bool sm_37TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    return sm_2xTextGen(info, ldParms, fOptions);
}

static void sm_37FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_37LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_KEPLER_SM4_0);
}

static String sm_37Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_KEPLER_SM4_0);
}

static void sm_37MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_37ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_3xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

#endif // #if LWCFG(GLOBAL_GPU_IMPL_GK110C)


#if LWCFG(GLOBAL_ARCH_MAXWELL)


/* --------------------------------  SM50  ---------------------------------------- */

static PtxArch sm_50ptxArch = PTX_ARCH_SM_50;

static Bool sm_50CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cpf_optx");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 1))
            return 1;
    }
    // FIXME : No need to guard once all SM_5x functions are guarded properly.
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    return GenerateCode_Maxwell_cp(ld, 1);
#else
    return GenerateCode_Fermi_cp(ld, 1);
#endif
}

static Bool sm_50TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cpf_optx");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 0))
            return 1;
    }
    // FIXME : No need to guard once all SM_5x functions are guarded properly.
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    return GenerateCode_Maxwell_cp(ld, 0);
#else
    return GenerateCode_Fermi_cp(ld, 0);
#endif
}

static void sm_50FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_50LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_MAXWELL_SM5_0);
}

static String sm_50Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_MAXWELL_SM5_0);
}


static void sm_50MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_50ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_3xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

/* --------------------------------  SM52  ---------------------------------------- */

static PtxArch sm_52ptxArch = PTX_ARCH_SM_52;

static Bool sm_52CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cpf_optx");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 1))
            return 1;
    }
    // FIXME : No need to guard once all SM_5x functions are guarded properly.
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    return GenerateCode_Maxwell_cp(ld, 1);
#else
    return GenerateCode_Fermi_cp(ld, 1);
#endif
}

static Bool sm_52TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cpf_optx");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 0))
            return 1;
    }
    // FIXME : No need to guard once all SM_5x functions are guarded properly.
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    return GenerateCode_Maxwell_cp(ld, 0);
#else
    return GenerateCode_Fermi_cp(ld, 0);
#endif
}

static void sm_52FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_52LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_MAXWELL_SM5_2);
}

static String sm_52Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_MAXWELL_SM5_2);
}

static void sm_52MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_52ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_3xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

/* --------------------------------  SM53  ---------------------------------------- */

static PtxArch sm_53ptxArch = PTX_ARCH_SM_53;

static Bool sm_53CodeGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cpf_optx");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 1))
            return 1;
    }
    // FIXME : No need to guard once all SM_5x functions are guarded properly.
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    return GenerateCode_Maxwell_cp(ld, 1);
#else
    return GenerateCode_Fermi_cp(ld, 1);
#endif
}

static Bool sm_53TextGen( gpuFeaturesProfile info, Pointer ldParms,  Pointer fOptions)
{
    LdParams *ld= (LdParams *)ldParms;

    ld->profileName      = AddIAtom(ld->atable, "cpf_optx");
    if (ld->oriControl == ORI_ENABLE) {
        if (OriGenerateCode_Fermi_ptx(ld, 0))
            return 1;
    }
    // FIXME : No need to guard once all SM_5x functions are guarded properly.
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    return GenerateCode_Maxwell_cp(ld, 0);
#else
    return GenerateCode_Fermi_cp(ld, 0);
#endif
}

static void sm_53FatPointCount(Pointer Ld, unsigned int *ccRegFatpoint, unsigned int *addrRegFatpoint, 
                               unsigned int *genRegFatpoint)
{
    sm_2xFatPointCount(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

static void *sm_53LdParamsGen(gpuFeaturesProfile profile, Pointer fOptions, Pointer fPool)
{
    return sm_2xLdParamsGen(profile, fOptions, fPool, COP_ARCH_VARIANT_MAXWELL_SM5_3);
}

static String sm_53Disassembler(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded)
{
    return sm_2xDisassembler(image, size, vbase, oneShot, decoded, COP_ARCH_VARIANT_MAXWELL_SM5_3);
}


static void sm_53MemoryLimits(Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    sm_2xMemoryLimits(lmemMax, smemMax, cmemMax);
}

static int sm_53ComputeMaxRReg(gpuFeaturesProfile gpuInfo, int maxThreadsPerCta, int minNCtaPerSM)
{
    return sm_3xComputeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

#endif //#if LWCFG(GLOBAL_ARCH_MAXWELL)

/*----------------------------------------------------------------------------*/

static Bool initialized = False;

static stdMap_t textGenerators;
static stdMap_t codeGenerators;
static stdMap_t ldparamsGenerators;
static stdMap_t ptxArchEnums;
static stdMap_t disassemblers;
static stdMap_t memoryLimitFuncts;
static stdMap_t getFatPointFuncts;
static stdMap_t computeMaxRRegFuncts;

void deleteAllMaps(void)
{
    if (initialized) {
        initialized = False;
        mapDelete(textGenerators);
        mapDelete(codeGenerators);
        mapDelete(ldparamsGenerators);
        mapDelete(ptxArchEnums);
        mapDelete(disassemblers);
        mapDelete(memoryLimitFuncts);
        mapDelete(getFatPointFuncts);
        mapDelete(computeMaxRRegFuncts);
    }
}

static void initialize(void)
{
    if (!initialized) {
        stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );

        textGenerators          = mapNEW(String,8);
        codeGenerators          = mapNEW(String,8);
        disassemblers           = mapNEW(String,8);
        ldparamsGenerators      = mapNEW(String,8);
        ptxArchEnums            = mapNEW(String,8);
        memoryLimitFuncts       = mapNEW(String,8);
        getFatPointFuncts       = mapNEW(String,8);
        computeMaxRRegFuncts    = mapNEW(String,8);
        {
            #define  DO_CODEGENERATORS 1
            #include "GPUSpec.inc"
        }
        initialized= True;
        
        stdSwapMemSpace(savedSpace);
    }
}

/*
 * Function        : Obtain a code generator functioni for the specified 
 *                   compilation profile.
 * Parameters      : profileName (I) Name of gpu profile to obtain 
 *                                   code generator for.
 *                   text        (I) Return a text assembly generator
 *                                   if this value is equal to True.
 *                                   otherwise return a binary code
 *                                   generator.
 * Function Result : Requested code generator, or NULL if not defined.
 */
gpuCodeGenFun gpuGetCodeGenerator( String profileName, Bool text )
{
    initialize();
    
    if (text) {
        return mapApply(textGenerators,profileName);
    } else {
        return mapApply(codeGenerators,profileName);
    }
}

/*
 * Function        : Obtain a disassembler function for the specified 
 *                   compilation profile.
 * Parameters      : profileName (I) Name of gpu profile to obtain 
 *                                   disassembler for.
 * Function Result : Requested disassembler, or NULL if not defined.
 */
gpuDisassembleFun gpuGetDisassembler( String profileName )
{
    initialize();
    
    if (initializeAssembler(profileName)) {
        return mapApply(disassemblers,profileName);
    } else {
        return NULL;
    }
}


/*
 * Function        : Free assembly obtained via gpuGetDisassembler.
 * Parameters      : disassembly (I) Disassembly generated by a disassembler
 *                                   function obtained from gpuGetDisassembler
 */
void gpuInfoFreeDisassembly(String disassembly)
{
    stdFREE(disassembly);
}


/*
 * Function        : Obtain a LdParams generator function for the 
 *                   specified compilation profile.
 * Parameters      : profileName (I) Name of gpu profile to obtain 
 *                                   LdParams generator for.
 *
 * Function Result : Requested LdParams generator, or NULL if not defined.
 */
ldParamsGenFun gpuGetLdParamsGenerator( String profileName)
{
    initialize();
    
    return mapApply(ldparamsGenerators,profileName);
}

/*
 * Function        : obtain the PtxArch enum for the specified compilation profile.
 * Parameters      : profileName (I) Name of the gpu profile
 * Function Result : Requested ptxArch enum or -1 if not defined.
 */
PtxArch gpuGetPtxArch(String profileName)
{
    PtxArch *arch;

    initialize();

    arch = mapApply(ptxArchEnums, profileName);
    return ((arch) ? *arch : -1);
}



/* 
 * Function        : obtain the limits on size of local/shared/constant memory segments imposed by the
 *                   specified compilation profile
 * Parameters      : profileName (I) Name of the gpu profile
 * Function Result : None
 */
void gpuGetMemoryLimits(String profileName, Int *lmemMax, Int *smemMax, Int *cmemMax)
{
    void (*memLimitFunct)(Int *, Int *, Int *);

    initialize();

    memLimitFunct = mapApply(memoryLimitFuncts,profileName);
    memLimitFunct(lmemMax, smemMax, cmemMax);
}

void gpuGetFatPointCount(String profileName, Pointer Ld, unsigned int *ccRegFatpoint, 
                         unsigned int *addrRegFatpoint, unsigned int *genRegFatpoint)
{
    void (*getFatPointCountFun)(Pointer, unsigned int *, unsigned int *, unsigned int *);

    initialize();
   
    getFatPointCountFun = mapApply(getFatPointFuncts, profileName);
    getFatPointCountFun(Ld, ccRegFatpoint, addrRegFatpoint, genRegFatpoint);
}

void gpuSetCallInfo(Pointer Ld, Pointer fCallInfo, String profileName)
{
    if (IS_TESLA(gpuGetPtxArch(profileName)))
        sm1x_setCallInfo(Ld, fCallInfo);
    else 
        stdASSERT(0,("Tesla profile expected"));
}

/*
 * Function    : compute max R Reg count to be targeted for code generation for the given
 *               compilation profile, and the user specified max threads/CTA and CTA/SM
 *
 */

int gpuComputeMaxRReg(String profileName, gpuFeaturesProfile gpuInfo, 
                      int maxThreadsPerCta, int minNCtaPerSM)
{
    int (*computeMaxRReg)(Pointer, int, int);

    initialize();
    computeMaxRReg = mapApply(computeMaxRRegFuncts,profileName);
    return computeMaxRReg(gpuInfo, maxThreadsPerCta, minNCtaPerSM);
}

#undef GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_G8X
#undef GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_GT2XX
#undef GPU_CODE_GEN_NUM_THREAD_IDS_PER_SM_GF1XX

/* 
 * Function : search and set the minimal real profile 
 */

static void searchMinimalReal(gpuFeaturesProfile lwrrProfile, gpuFeaturesProfile *minReal)
{
    if (lwrrProfile->isVirtual)
        return;

    if ((*minReal == NULL) ||
        (gpuGetPtxArch(lwrrProfile->profileName) < gpuGetPtxArch((*minReal)->profileName))) {
        *minReal = lwrrProfile; 
    }
    return;
}

/*
 * Function        : Identify the minimal real profile that implements a virtual profile
 * 
 * Parameters      : virtualProfile (I) Virtual Profile
 * Parameters      : realProfile    (O) Minimal Real Profile
 * Function Result :
 */
gpuFeaturesProfile gpuGetMinimalRealProfile(gpuFeaturesProfile virtualProfile)
{
    gpuFeaturesProfile realProfile = NULL;

    if (!virtualProfile->isVirtual)
        return virtualProfile;

    setTraverse(virtualProfile->implementsProfile, (stdEltFun) searchMinimalReal, &realProfile);

    stdCHECK((realProfile && !realProfile->isVirtual), (gpuinfMsgNoRealProfile, virtualProfile->profileName));

    return realProfile;
}
