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
 *  Module name              : gpuInfo.c
 *
 *  Description              :
 * 
 */

/*--------------------------------- Includes ---------------------------------*/

#include <stdMessages.h>
#include <stdLocal.h>
#include <stdProcess.h>
#include <stdList.h>
#include <stdString.h>
#include <stdFileNames.h>
#include <stdString.h>
#include <stdMap.h>
#include <gpuInfo.h>
#include <ctArch.h>
#include "g_lwconfig.h"
#include "interfaceUtils.h"

/*-------------------------- Known PTX Profiles SMs --------------------------*/

// This list is consulted when a PTX object is presented that does not
// correspond to any profile available in gpuInfo. Typical examples
// are sm_1x, and also CheetAh SMs on desktop builds.
static const int knownSMValues[] = { 10, 11, 12, 13,
                               20, 21,
                               30, 32, 35, 37,
                               50, 52, 53,
                               60, 61, 62, 69,
                               70 };

static stdString_t realProfileNames;
static stdString_t allProfileNames;

/*------------------------------- Chip Driver --------------------------------*/

static stdMap_t infoTable = NULL;
static Bool initialized = False;

// Because featureTool is released separately from gpuInfo, 
// we cannot change interface between them;
// instead add a new version of createProfile. 
// TODO: no longer use featureTool so could simplify this.
static gpuFeaturesProfile createProfile_v2 (
    Bool isVirtual, 
    Bool isIntermediate, 
    String profileName, 
    String internalName, 
    String isaClass, 
    String ccopts, 
    String archRepresentative)
{
    gpuFeaturesProfile result;
    
    stdNEW(result);
    
    result->isVirtual                       = isVirtual;
    result->isIntermediate                  = isIntermediate;
    result->profileName                     = profileName;
    result->internalName                    = internalName;
    result->isaClass                        = isaClass;
    result->ccopts                          = ccopts;

    result->runsProfile                     = setNEW(Pointer,8);
    result->linksProfile                    = setNEW(Pointer,8);
    result->implementsProfile               = setNEW(Pointer,8);
    
    result->archRepresentative              = archRepresentative;
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    result->isMerlwry                       = False;
    result->noMerlwrySupport                = False;
#endif
    
    // create comma-separated list of names
    if (! stringIsEmpty(allProfileNames)) {
        stringAddChar(allProfileNames, ',');
    }
    stringAddBuf(allProfileNames, profileName);
    if (! isVirtual) {
        if (! stringIsEmpty(realProfileNames)) {
            stringAddChar(realProfileNames, ',');
        }
        stringAddBuf(realProfileNames, profileName);
    }
    return result;
}

static gpuFeaturesProfile createProfile( Bool isVirtual, 
    String profileName, String internalName, String isaClass, 
    String ccopts, String archRepresentative)
{
    return createProfile_v2 (isVirtual, False, 
                             profileName, internalName, isaClass, 
                             ccopts, archRepresentative);
}


/*
 * Function        : Release and delete all previously obtained compilation information.
 */
    static void deleteProfile( gpuFeaturesProfile info, Pointer dummy )
    {
        mapUndefine(infoTable, info->profileName);
        setDelete(info->runsProfile);
        setDelete(info->linksProfile);
        setDelete(info->implementsProfile);
        stdFREE(info);
    }

static void gpuDeleteAllFeaturesProfiles(void)
{
    if (initialized) {
        initialized = False;
        mapRangeTraverse(infoTable, (stdEltFun)deleteProfile, NULL);
        mapDelete(infoTable);
        infoTable = NULL;
        stringDelete(allProfileNames);
        stringDelete(realProfileNames);
    }
}

/*
 * Function        : Obtain a compilation info handle for the specified gpu/variant.
 * Parameters      : gpuName      (I) Name of gpu to obtain info for.
 * Function Result : Requested info, or NULL if the specified gpu is not defined.
 */
    static void initialize(void)
    {
        if (initialized) {
            return;
        }

        interface_mutex_enter(PTX_GPUINFO_LOCK);
        msgTry(True) {
            // We need to check again for 'initialized' to ensure that initialization
            // will be done by single thread even if multiple threads observe
            // initialized as False
            if (!initialized) {
                stdMemSpace_t savedSpace = stdSwapMemSpace( memspNativeMemSpace );
                infoTable= mapNEW(String,8);
                allProfileNames = stringNEW();
                realProfileNames = stringNEW();
                {
                    #include "gpuSPEC.inc"
                }
                stdSetCleanupHandler((stdDataFun)gpuDeleteAllFeaturesProfiles,NULL);
                stdSwapMemSpace( savedSpace );
                initialized = True;
            }
        } msgOtherwise {
        } msgEndTry;
        interface_mutex_exit(PTX_GPUINFO_LOCK);
    }

gpuFeaturesProfile gpuGetFeaturesProfile( String gpuName )
{
    initialize();
    return mapApply(infoTable,gpuName);
}

/*
 * Function        : Obtain a comma separated list of the names of all gpus 
 *                   lwrrently described by this module.
 * Parameters      : kind  (I) Indicates the required profile kind that 
 *                              must be included in the result.
 * Function Result : String containing gpu names.
 */
String gpuSupportedArchitectures( gpuProfileKind kind )
{
    initialize(); // so names are set
    switch (kind) {
    case gpuRealProfile        : return stringToBuf(realProfileNames);
    case gpuAnyProfile         : return stringToBuf(allProfileNames);
    default                    : stdASSERT( False, ("Case label out of bounds") );
                                 return NULL;
    }
}

/*
 * Function        : Obtain a comma separated list of the names of all gpus
 *                   lwrrently described by this module, plus "native" and
 *                   "all", "all-major".
 * Parameters      : None.
 * Function Result : String containing gpu names, plus "native" and "all",
 *                   "all-major".
 */
String gpuSupportedArchitecturesPlusNativeAndAll()
{
    stdString_t str;
    initialize(); // so names are set
    str = stringCopy(allProfileNames);
    stringAddBuf(str, ",native,all,all-major");
    String s = stringToBuf(str);
    stringDelete(str);
    return s;
}

/*
 * Function        : Obtain the name of the default profile to be used by 
 *                   lwcc and ptxas, or NULL if not speciefied.
 * Function Result : String containing gpu features names.
 */
String gpuDefaultProfile()
{
    return DEFAULT_PROFILE;
}

/*
 * Function        : Test if code compiled for gpu1 will execute on gpu2.
 * Parameters      : gpu1 (I) Name of code gpu
 *                   gpu2 (I) Name of exelwtion gpu 
 * Function Result : True iff. g1 is compatible with g2 in this sense
 */
Bool gpuInfoRunsOn(String gpu1, String gpu2)
{
    gpuFeaturesProfileHandle fp1, fp2;
    Bool ret = False;

    fp1= gpuGetFeaturesProfileHandle(gpu1);
    fp2= gpuGetFeaturesProfileHandle(gpu2);

    if (fp1 && fp2) {
        ret = gpuRunsProfile(fp1, fp2);
    }
    if (fp1) {
        stdFREE(fp1);
    }
    if (fp2) {
        stdFREE(fp2);
    }
    return ret;
}

/* similar to RunsOn but checks if link compatible */
Bool gpuInfoLinksOn(String gpu1, String gpu2)
{
    gpuFeaturesProfileHandle fp1, fp2;
    Bool ret = False;

    fp1= gpuGetFeaturesProfileHandle(gpu1);
    fp2= gpuGetFeaturesProfileHandle(gpu2);

    if (fp1 && fp2) {
        ret = gpuLinksProfile(fp1, fp2);
    }
    if (fp1) {
        stdFREE(fp1);
    }
    if (fp2) {
        stdFREE(fp2);
    }
    return ret;
}

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
    typedef struct {
        Bool           doVirtual;
        Bool           doPhysical;
        gpuProfileFun  traverse;
        Pointer        data;
    } ProfileRec;

    static void traverseProfile( gpuFeaturesProfile profile, ProfileRec *rec )
    {
        if ( rec->doVirtual &&  profile->isVirtual) { rec->traverse(profile,rec->data); }
        if (!rec->doVirtual && !profile->isVirtual) { rec->traverse(profile,rec->data); }
    } 

void  gpuTraverseProfile( Bool doVirtual, Bool doPhysical, gpuProfileFun traverse, Pointer data )
{
    ProfileRec rec;
    
    rec.doVirtual  = doVirtual;
    rec.doPhysical = doPhysical;
    rec.traverse   = traverse;
    rec.data       = data;
    
    mapRangeTraverse( infoTable, (stdEltFun)traverseProfile, &rec );
}




/*
 * Function        : Construct qualified name of gpu code, reflecting ISA plus 
 *                   virtual architecture features for which the code was compiled.
 * Parameters      : code          (I) ISA architecture
 *                 : architecture  (I) Virtual features architecture
 * Function Result : Constructed name
 */
String gpuComposeCodeName( String code, String architecture )
{
    if (!architecture) {
        return stdCOPYSTRING(code);
    } else {
        stdString_t result= stringNEW();

        stringAddFormat(result,"%s@%s",code,architecture);

        return stringStripToBuf(result);
    }
}


/*
 * Function        : Splitup qualified name of gpu code, into the name of the ISA plus 
 *                   the name of the virtual architecture features for which the code was compiled.
 * Parameters      : qualified     (I) Virtual features architecture
 * Parameters      : code          (O) ISA architecture
 *                 : architecture  (O) Virtual features architecture
 * Function Result :
 */
void gpuDecomposeCodeName( String qualified, String *code, String *architecture )
{
    String qucpy = stdCOPYSTRING(qualified);
    Char  *sep   = strchr(qucpy,'@');
    
   *code = qucpy;
   
    if (!sep) {
        *architecture = NULL;
    } else {
        *sep          = 0;
        *architecture = stdCOPYSTRING(sep+1);
    }
}

static gpuFeaturesProfile gpuGetFeatureProfileFromHandle(gpuFeaturesProfileHandle profile)
{
    gpuFeaturesProfile retVal;
    String searchStr = gpuGetProfileName(profile);
    retVal = gpuGetFeaturesProfile(searchStr);
    stdFREE(searchStr);
    return retVal;
}

static Bool gpuIsKnownSMValue(int sm)
{
    int i;
    for(i = 0; i < sizeof(knownSMValues)/sizeof(knownSMValues[0]); i++) {
        if (sm == knownSMValues[i]) {
            return True;
        }
    }
    return False;
}

/*
 * Function        : Obtain the feature profile handle from the string representing the gpu
 *
 * Parameters      : gpuName (I) Name of the gpu profile
 * Function Result :             GPU feature profile handle of the profile named 'gpuName'
 *
 * Note             : Handles can be created only for "Known SM Values"
 */
gpuFeaturesProfileHandle gpuGetFeaturesProfileHandle(String gpuName)
{
    gpuFeaturesProfileHandle res;
    int sm;
    Bool isUnavailable = False;

    if (gpuName == NULL) {
        stdASSERT(False, ("Null name"));
        return NULL;
    }

    sm = ctParseArchVersion(gpuName);

    // All known SMs have a profile, even if the driver does not
    // support them. This is useful in PTX, where the PTX will be JIT
    // compiled to a newer available profile.
    if (gpuGetFeaturesProfile(gpuName) == NULL) {
        isUnavailable = True;
        if (!gpuIsKnownSMValue(sm))
            return NULL;
    }

    stdNEW(res);
    res->smValue = sm;
    res->isVirtual = ctIsVirtualArch(gpuName);
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    res->isMerlwry = gpuIsMerlwryArch(gpuName);
#endif
    res->isUnavailable = isUnavailable;
    return res;
}

#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
Bool gpuIsMerlwryArch(String str)
{
    unsigned int sm_version = ctParseArchVersion(str);
    if (ctIsVirtualArch(str) || sm_version < gpuFirstMerlwrySMNumber() || strncmp(str, "sass_", 5) == 0)
        return False;
    return True;
}
#endif
static gpuFeaturesProfileHandle gpuGetFeatureProfileHandleFromProfile(gpuFeaturesProfile profile)
{
    gpuFeaturesProfileHandle pHandle;

    if (profile == NULL) {
        stdASSERT(False, ("Null profile"));
        return NULL;
    }

    stdNEW(pHandle);

    pHandle->isVirtual = profile->isVirtual;
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    pHandle->isMerlwry = gpuIsMerlwryArch(profile->profileName);
#endif
    pHandle->smValue   = ctParseArchVersion(profile->profileName);

    return pHandle;
}

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
Bool gpuIsUnavailableProfile(gpuFeaturesProfileHandle profile)
{
    if (profile == NULL) {
        stdASSERT(False, ("Null profile"));
        return False;
    }
    return profile->isUnavailable;
}


/*
 * Function        : Get the minimal virtual profile that implements the input profile
 * 
 * Parameters      :  profile (I) Profile whose minimal virtual is to be queried
 * Function Result :              Minimal virtual profile of the inpute profile
 *
 * Note            :  For unavailable GPU profile, sm_XY is just colwerted to compute_XY
 */
gpuFeaturesProfileHandle gpuGetMinimalVirtualProfile(gpuFeaturesProfileHandle pHandle)
{
    gpuFeaturesProfile profile;
    String computeVal;
    gpuFeaturesProfileHandle lHandle;

    if (pHandle == NULL) {
        stdASSERT(False, ("Null profile"));
        return NULL;
    } else if (!gpuIsUnavailableProfile(pHandle)) {
         profile = gpuGetFeatureProfileFromHandle(pHandle)->minimalVirtual;
         return gpuGetFeatureProfileHandleFromProfile(profile);
    } else {
        computeVal = stdMALLOC(12);
        sprintf(computeVal, "compute_%2d", pHandle->smValue);
        lHandle = gpuGetFeaturesProfileHandle(computeVal);
        stdFREE(computeVal);
        return lHandle;
    }
}

/*
 * Function        :  Check if a valid profile is a virtual profile or a real profile
 * 
 * Parameters      :  profile (I) Profile whose virtual-ness is to be determined
 * Function Result :              True if the input profile is virual
                                  False, otherwise
 */
Bool gpuIsVirtual(gpuFeaturesProfileHandle profile)
{
    if (profile == NULL) {
        stdASSERT(False, ("Null profile"));
        return False;
    } 
    return profile->isVirtual;
}

/*
 * Function        : Checks if a profile gpu1 has a higher feature weight than profile gpu2  
 * 
 * Parameters      :  gpu1 (I) Name of the first  gpu profile which is to be compared
 * Parameters      :  gpu2 (I) Name of the second gpu profile which is to be compared
 * Function Result :           True,  if gpu1 has greater feature weight than gpu2
 *                             False, otherwise 
 *
 * Note            : Feature weight of unavailable GPU is assumed to be 0
 *                   If both 'gpu1' and 'gpu2' are unavailable then their SM value is used for comparision
 */
Bool gpuIsBetterThan(gpuFeaturesProfileHandle gpu1, gpuFeaturesProfileHandle gpu2)
{
    if (gpu1 == NULL || gpu2 == NULL) {
        stdASSERT(False, ("Null profile"));
        return False;
    } else {
        // assumption is that higher sm is always better
        return gpu1->smValue > gpu2->smValue;
    }
}

/*
 * Function        : Checks if a profile named gpu1 is 'equal to' the profile named gpu2 
 *                    where atleast one of the profiles is deprecatred
 * 
 * Parameters      :  gpu1 (I) Name of the first  gpu profile which is to be compared
 * Parameters      :  gpu2 (I) Name of the second gpu profile which is to be compared
 * Function Result :           True,  if gpu1 is value-wise equal to gpu2
 *                             False, otherwise
 *
 * Note            : Feature weight of unavailable GPU is assumed to be 0
 *                   If both 'gpu1' and 'gpu2' are unavailable then their SM value is used for comparision
 */
Bool gpuIsEqualTo(gpuFeaturesProfileHandle gpu1, gpuFeaturesProfileHandle gpu2)
{
    if (gpu1 == NULL || gpu2 == NULL) {
        stdASSERT(False, ("Null profile"));
        return False;
    } else {
        return gpu1->smValue == gpu2->smValue;
    }
}

Bool gpuImplementsProfile(gpuFeaturesProfile candidateProfile, gpuFeaturesProfile targetProfile)
{
    gpuFeaturesProfileHandle candidateHandle = gpuGetFeatureProfileHandleFromProfile(candidateProfile);
    gpuFeaturesProfileHandle targetHandle = gpuGetFeatureProfileHandleFromProfile(targetProfile);

    if (candidateProfile == NULL || targetProfile == NULL) {
        stdASSERT(False, ("Null profile"));
        return False;
    }
    if (gpuIsUnavailableProfile(candidateHandle) ||
        gpuIsUnavailableProfile(targetHandle))
    {
        return False;
    }
    return candidateHandle->smValue >= targetHandle->smValue;
}

/*
 * Function        :  Check if the 'candidateProfile' profile can be run directly (binary compatible) 
 *                    or implemented indirectly (JIT compiled) on 'targetProfile'
 * 
 * Parameters      :  candidateProfile (I) profile which needs to be run/implemented
 * Parameters      :  targetProfile    (I) profile on which 'candidateProfile' needs to be run/implemented on 
 * Function Result :                       True,  if 'candidateProfile' profile can be run directly or implemented indirectly on 'targetProfile'
 *                                         False, otherwise
 */
Bool gpuRunsProfile(gpuFeaturesProfileHandle candidateProfile, gpuFeaturesProfileHandle targetProfile)
{
    Bool candidateUnavailable, targetUnavailable;
   
    if (candidateProfile == NULL || targetProfile == NULL) {
        stdASSERT(False, ("Null profile"));
        return False;
    }
    
    candidateUnavailable = gpuIsUnavailableProfile(candidateProfile);
    targetUnavailable    = gpuIsUnavailableProfile(targetProfile);

    if (targetUnavailable) {
        stdASSERT(False, ("Checking for compatibility on an unavailable GPU. Driver should not be installable on this GPU"));
        return False;
    }
       
    if (!candidateUnavailable) {
        gpuFeaturesProfile candProf = gpuGetFeatureProfileFromHandle(candidateProfile);
        gpuFeaturesProfile targProf = gpuGetFeatureProfileFromHandle(targetProfile);

        if (candidateProfile->isVirtual) {
            /* can JIT to any later profile */
            return candidateProfile->smValue <= targetProfile->smValue;
        } else {
            return (setElement(candProf->runsProfile, targProf) != NULL);
        }
    }

    // TODO: Handle partial deprecation of a family on a non-virtual
    // candidateProfile.
    //
    // Consider the situation when sm_30 is deprecated and becomes
    // unavailable. An sm_30 ELF should still be allowed on sm_35,
    // since it is binary compatible. But sm_32 is a CheetAh profile and
    // unavailble on the desktop. An sm_32 ELF should not be allowed
    // on sm_35 because it is not compatible.
    //
    // Both situations are conservatively disallowed for now.
    if (!candidateProfile->isVirtual) return False;

    // For a virtual profile (PTX), a lower unavailable candidate can
    // run on a higher available target.
    return candidateProfile->smValue < targetProfile->smValue;
}

/* Similar to gpuRunsProfile but only True if can also link with profile */
Bool gpuLinksProfile(gpuFeaturesProfileHandle candidateProfile, gpuFeaturesProfileHandle targetProfile)
{
    Bool candidateUnavailable, targetUnavailable;
   
    if (candidateProfile == NULL || targetProfile == NULL) {
        stdASSERT(False, ("Null profile"));
        return False;
    }
    
    candidateUnavailable = gpuIsUnavailableProfile(candidateProfile);
    targetUnavailable    = gpuIsUnavailableProfile(targetProfile);

    if (targetUnavailable) {
        stdASSERT(False, ("Checking for compatibility on an unavailable GPU. Driver should not be installable on this GPU"));
        return False;
    }
       
    if (!candidateUnavailable) {
        gpuFeaturesProfile candProf = gpuGetFeatureProfileFromHandle(candidateProfile);
        gpuFeaturesProfile targProf = gpuGetFeatureProfileFromHandle(targetProfile);
        if (candidateProfile->isVirtual) {
            /* can JIT to any later profile */
            return candidateProfile->smValue <= targetProfile->smValue;
        } else {
            return (setElement(candProf->linksProfile, targProf) != NULL);
        }
    }

    // TODO: Handle partial deprecation of a family on a non-virtual
    // candidateProfile.
    //
    // Consider the situation when sm_30 is deprecated and becomes
    // unavailable. An sm_30 ELF should still be allowed on sm_35,
    // since it is binary compatible. But sm_32 is a CheetAh profile and
    // unavailable on the desktop. An sm_32 ELF should not be allowed
    // on sm_35 because it is not compatible.
    //
    // Both situations are conservatively disallowed for now.
    return False;
}

/*
 * Function        :  Query the profile name of a profile
 * 
 * Parameters      :  profile (I) profile whose profile name is to be queries
 * Function Result :              profile name of 'profile' as a string
 *
 */
String gpuGetProfileName(gpuFeaturesProfileHandle profile)
{
    String computeVal;

    if (profile == NULL) {
        stdASSERT(False, ("Null profile"));
        return stdCOPYSTRING("");
    }
    
    computeVal = stdMALLOC(12);
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    sprintf(computeVal, profile->isVirtual ? "compute_%2d" : (profile->smValue >= gpuFirstMerlwrySMNumber() && !profile->isMerlwry) ? "sass_%2d" : "sm_%2d", profile->smValue);
#else
    sprintf(computeVal, profile->isVirtual ? "compute_%2d" : "sm_%2d", profile->smValue);
#endif
    if (!gpuIsUnavailableProfile(profile)) {
        gpuFeaturesProfile gpu = gpuGetFeaturesProfile(computeVal);
        stdFREE(computeVal);
        return stdCOPYSTRING(gpu->profileName);
    } else {
        return computeVal;
    }
}

/*
 * Function        :  Query the Arch Representative of a profile
 * 
 * Parameters      :  profile (I) profile whose arch representative is to be queries
 * Function Result :              String which has the arch representative of the input param 'profile'
 *
 */
String gpuGetArchRepresentative(gpuFeaturesProfileHandle profile)
{
    if (profile == NULL || gpuIsUnavailableProfile(profile)) {
        stdASSERT(False, ("Invalid profile"));
        return stdCOPYSTRING("");
    } 
    return stdCOPYSTRING(gpuGetFeatureProfileFromHandle(profile)->archRepresentative);
}

static int theFirstInternalMerlwrySMNumber;
static int theFirstMerlwrySMNumber;
/*
 * Function        :  Obtain the SM number of first internally supported Merlwry gpu architecture
 *
 * Parameters      :  none
 * Function Result :              SM number of first internally supported Merlwry gpu architecture

 */
int gpuFirstInternalMerlwrySMNumber()
{
    return theFirstInternalMerlwrySMNumber;
}

/*
 * Function        :  Obtain the SM number of first offically supported Merlwry gpu architecture
 *
 * Parameters      :  none
 * Function Result :              SM number of first offically supported Merlwry gpu architecture

 */
int gpuFirstMerlwrySMNumber()
{
    return theFirstMerlwrySMNumber;;
}

/*
 * Function        :  Set the first internally supported Merlwry gpu SM number
 *
 * Parameters      :  sm number
 * Function Result :              none

 */
void gpuSetFirstInternalMerlwrySMNumber(int sm)
{
    theFirstInternalMerlwrySMNumber = sm;
}

/*
 * Function        :  Set the first officially supported Merlwry gpu SM number
 *
 * Parameters      :  sm number
 * Function Result :              none

 */
void gpuSetFirstMerlwrySMNumber(int sm)
{
    theFirstMerlwrySMNumber = sm;
}
