/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2021, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : gpuInfo.h
 *
 *  Description              :
 *            API to GPU description module compiled from GPU.spec by featureTool
 */

#ifndef stdGpuInfo_INCLUDED
#define stdGpuInfo_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include <stdTypes.h>
#include <stdSet.h>
#include <g_lwconfig.h>
#include "ptxDCI.h"
#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct _gpuFeaturesProfile *gpuFeaturesProfile;
typedef struct _gpuFeatureHandle   *gpuFeaturesProfileHandle;

typedef void    (*gpuProfileFun     )(gpuFeaturesProfile profile, Pointer data);
typedef Bool    (*gpuCodeGenFun     )(gpuFeaturesProfile profile, Pointer ldParms, Pointer fOptions );
typedef Pointer (*ldParamsGenFun    )(gpuFeaturesProfile profile, Pointer fOptions, Pointer fMemPool, ptxDCIHandle dciHandle);
typedef String  (*gpuDisassembleFun )(Byte *image, uInt size, Address vbase, Bool oneShot, uInt *decoded);

// External world just needs to have this handle 
struct _gpuFeatureHandle {
    int  smValue;
    Bool isVirtual;  // compute or sm
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    Bool isMerlwry;
#endif
    Bool isUnavailable;
};

struct _gpuFeaturesProfile {
    Bool                isVirtual;
    Bool                isIntermediate;
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    Bool                isMerlwry;
    Bool                noMerlwrySupport;  /* some architectures may not support Mercruy
                                              even previous version support it (e.g. Ada) */
#endif
    String              profileName;
    String              internalName;
    String              isaClass;
    String              archRepresentative;
    String              ccopts;
    
    stdSet_t            runsProfile;          /* gpuFeaturesProfile */
    stdSet_t            linksProfile;         /* gpuFeaturesProfile */
    stdSet_t            implementsProfile;    /* gpuFeaturesProfile */
    gpuFeaturesProfile  minimalVirtual;

    /* arch-specific limits */
    uInt                regFileSize;
    uInt                regFileSizePerCTA;
    uInt                regAllolwnit;
    uInt                regAlignment;
    uInt                maxRegsPerThread;
    uInt                minRegsPerThread;
    uInt                maxCtaPerSM;
    uInt                maxWarps;
    uInt                warpSize;
    uInt                warpAlign;
};

// As this is mostly static, avoiding overheads of adding it in the above struct
#define GPU_INFO_MAX_USER_BARRIER_REGS          11
#define GPU_INFO_MAX_USER_RREGS_VOLTA           253
#define GPU_INFO_MAX_USER_UREGS_TURING          63
#define GPU_INFO_FIRST_ALLOWED_PARAM_REGISTER   4

typedef enum {
    gpuRealProfile,
    gpuAnyProfile,
} gpuProfileKind;

/*-------------------------------- Chip Driver -------------------------------*/

/*
 * Function        : Obtain a compilation info handle for the specified gpu/variant.
 * Parameters      : gpuName      (I) Name of gpu to obtain info for.
 * Function Result : Requested info, or NULL if the specified gpu is not defined.
 */
gpuFeaturesProfile gpuGetFeaturesProfile( String gpuName );

/*
 * Function        : Obtain the feature profile handle from the string representing the gpu
 *
 * Parameters      : gpuName (I) Name of the gpu profile
 * Function Result :             GPU feature profile handle of the profile named 'gpuName'
 *
 * Note             : Handles can be created only for valid profiles
 *                    "Valid profile" include supported and deprecated architectures
 */
gpuFeaturesProfileHandle gpuGetFeaturesProfileHandle( String gpuName );

/*
 * Function        : Obtain a comma separated list of the names of all gpus 
 *                   lwrrently described by this module.
 * Parameters      : kind  (I) Indicates the required profile kind that 
 *                              must be included in the result.
 * Function Result : String containing gpu names.
 */
String gpuSupportedArchitectures( gpuProfileKind kind );

/*
 * Function        : Obtain a comma separated list of the names of all gpus
 *                   lwrrently described by this module, plus "native" and "all".
 * Parameters      : None.
 * Function Result : String containing gpu names, plus "native" and "all".
 */
String gpuSupportedArchitecturesPlusNativeAndAll();

/*
 * Function        : Obtain the name of the default profile to be used by 
 *                   lwcc and ptxas, or NULL if not speciefied.
 * Function Result : String containing gpu features names.
 */
String gpuDefaultProfile(void);


/*
 * Function        : Test if code compiled for gpu1 will execute on gpu2.
 * Parameters      : gpu1 (I) Name of code gpu
 *                   gpu2 (I) Name of exelwtion gpu 
 * Function Result : True iff. g1 is compatible with g2 in this sense
 */
Bool gpuInfoRunsOn(String gpu1, String gpu2);
/* similar to RunsOn, but checks whether link compatible */
Bool gpuInfoLinksOn(String gpu1, String gpu2);

/*
 * Function        : Apply specified function to all profiles.
 *                   with specified generic data element as additional parameter.
 * Parameters      : doVirtual  (I) include virtual profiles in traversal iff. this parameter has value True.
 *                   doPhysical (I) include physical profiles in traversal iff. this parameter has value True.
 *                   traverse   (I) Function to apply to traversed profiles.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result :
 */
void  gpuTraverseProfile( Bool doVirtual, Bool doPhysical, gpuProfileFun traverse, Pointer data );



/*
 * Function        : Construct qualified name of gpu code, reflecting ISA plus 
 *                   virtual architecture features for which the code was compiled.
 * Parameters      : code          (I) ISA architecture
 *                 : architecture  (I) Virtual features architecture
 * Function Result : Constructed name
 */
String gpuComposeCodeName( String code, String architecture );


/*
 * Function        : Splitup qualified name of gpu code, into the name of the ISA plus 
 *                   the name of the virtual architecture features for which the code was compiled.
 * Parameters      : qualified     (I) Virtual features architecture
 * Parameters      : code          (O) ISA architecture
 *                 : architecture  (O) Virtual features architecture
 * Function Result :
 */
void gpuDecomposeCodeName( String qualified, String *code, String *architecture );

/*
 * Function        : Identify the minimal real profile that implements a virtual profile
 * 
 * Parameters      : virtualProfile (I) Virtual Profile
 * Parameters      : realProfile    (O) Minimal Real Profile
 * Function Result :
 */
gpuFeaturesProfile gpuGetMinimalRealProfile(gpuFeaturesProfile virtualProfile);

/*
 * Function        : Get the minimal virtual profile that implements the input profile
 * 
 * Parameters      :  profile (I) Profile whose minimal virtual is to be queried
 * Function Result :              Minimal virtual profile of the inpute profile
 *
 * Note            :  For deprecated GPU profile, sm_XY is just colwerted to compute_XY
 */
gpuFeaturesProfileHandle gpuGetMinimalVirtualProfile(gpuFeaturesProfileHandle profile);

/*
 * Function        :  Check if a valid profile is a virtual profile or a real profile
 * 
 * Parameters      :  profile (I) Profile whose virtual-ness is to be determined
 * Function Result :              True if the input profile is virual
                                  False, otherwise
 */
Bool gpuIsVirtual(gpuFeaturesProfileHandle profile);

/*
 * Function        : Check whether the profile is not available.
 *
 *                   A profile is "unavailable" if it corresponds to a
 *                   known GPU, but support of that GPU was excluded
 *                   from this driver build.
 *
 *                   E.g.: a deprecated profile, or a CheetAh profile
 *                   specified on a non-CheetAh driver.
 *
 * Parameters      :  profile (I) Profile whose validity is to checked
 * Function Result :             True,  if the 'profile' is unavailable
 */
Bool gpuIsUnavailableProfile(gpuFeaturesProfileHandle profile);

/*
 * Function        : Checks if a profile gpu1 has a higher feature weight than profile gpu2  
 * 
 * Parameters      :  gpu1 (I) Name of the first  gpu profile which is to be compared
 * Parameters      :  gpu2 (I) Name of the second gpu profile which is to be compared
 * Function Result :           True,  if gpu1 has greater feature weight than gpu2
 *                             False, otherwise 
 *
 * Note            : Feature weight of deprecated GPU is assumed to be 0
 */
Bool gpuIsBetterThan(gpuFeaturesProfileHandle gpu1, gpuFeaturesProfileHandle gpu2);

/*
 * Function        : Checks if a profile named gpu1 is 'equal to' the profile named gpu2 
 *                    where atleast one of the profiles is deprecatred
 * 
 * Parameters      :  gpu1 (I) Name of the first  gpu profile which is to be compared
 * Parameters      :  gpu2 (I) Name of the second gpu profile which is to be compared
 * Function Result :           True,  if gpu1 is value-wise equal to gpu2
 *                             False, otherwise
 *
 * Note            : Feature weight of deprecated GPU is assumed to be 0
 *                   If both 'gpu1' and 'gpu2' are deprecated then their SM value is used for comparision
 */
Bool gpuIsEqualTo(gpuFeaturesProfileHandle gpu1, gpuFeaturesProfileHandle gpu2);
    
/* Check if 'candidateProfile' implements 'targetProfile' */
Bool gpuImplementsProfile(gpuFeaturesProfile candidateProfile, gpuFeaturesProfile targetProfile);

/*
 * Function        :  Check if the 'candidateProfile' profile can be run on 'targetProfile'
 * 
 * Parameters      :  candidateProfile (I) profile which needs to be run 
 * Parameters      :  targetProfile    (I) profile on which 'candidateProfile' needs to be run on 
 * Function Result :                       True,  if 'candidateProfile' profile can be run on 'targetProfile'
 *                                         False, otherwise
 */
Bool gpuRunsProfile(gpuFeaturesProfileHandle candidateProfile, gpuFeaturesProfileHandle targetProfile);
/* similar to gpuRunsProfile but only True if link compatible */
Bool gpuLinksProfile(gpuFeaturesProfileHandle candidateProfile, gpuFeaturesProfileHandle targetProfile);

/*
 * Function        :  Query the profile name of a profile
 * 
 * Parameters      :  profile (I) profile whose profile name is to be queries
 * Function Result :              profile name of 'profile' as a string
 *
 * Note            :  'profile' can be invalid profile which is still being tested for its validity
 */
String gpuGetProfileName(gpuFeaturesProfileHandle profile);

/*
 * Function        :  Query the Arch Representative of a profile
 * 
 * Parameters      :  profile (I) profile whose arch representative is to be queries
 * Function Result :              String which has the arch representative of the input param 'profile'
 *
 */
String gpuGetArchRepresentative(gpuFeaturesProfileHandle profile);

#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
/*
 * Function   : Determine if the specified arch is virtual.
 *
 * Parameters : str  (I) String of the form "sm_xx" , "sass_xx", "lto_xx" or "compute_xx"
 *                   (O) True if the string is "sm_xx" and xx >= first merlwry arch, else False
 */
Bool gpuIsMerlwryArch(String str);

/*
 * Function        :  Obtain the SM number of first internally supported Merlwry gpu architecture
 *
 * Parameters      :  none
 * Function Result :              SM number of first internally supported Merlwry gpu architecture

 */
int gpuFirstInternalMerlwrySMNumber();

/*
 * Function        :  Obtain the SM number of first offically supported Merlwry gpu architecture
 *
 * Parameters      :  none
 * Function Result :              SM number of first offically supported Merlwry gpu architecture

 */
int gpuFirstMerlwrySMNumber();

/*
 * Function        :  Set the first internally supported Merlwry gpu SM number
 *
 * Parameters      :  sm number
 * Function Result :              none

 */
void gpuSetFirstInternalMerlwrySMNumber(int sm);

/*
 * Function        :  Set the first officially supported Merlwry gpu SM number
 *
 * Parameters      :  sm number
 * Function Result :              none

 */
void gpuSetFirstMerlwrySMNumber(int sm);
#endif

void deleteAllMaps(void);

#ifdef __cplusplus
}
#endif

#endif
