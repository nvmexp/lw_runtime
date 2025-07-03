/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2011-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef ptxDCI_INCLUDED
#define ptxDCI_INCLUDED

#ifndef stdTypes_INCLUDED
typedef unsigned char Bool;
#endif

#include "API.h"
#include "lwelf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENABLE_MERLWRY_DCI 0

// Use this flag to temporary disable merlwry related code
#define ENABLE_MERLWRY_IMAGE_DCI 0

// Use this flag to enable new constant bank sections in ELF
#define ENABLE_MERLWRY_CONST_ELF_DCI 0

typedef enum {
    PTX_SPACE_NONE,

    PTX_SPACE_LOCAL,

    PTX_SPACE_SHARED,

    PTX_SPACE_CONST_MIN,
    PTX_SPACE_CONST0 = PTX_SPACE_CONST_MIN,
    PTX_SPACE_CONST1,
    PTX_SPACE_CONST2,
    PTX_SPACE_CONST3,
    PTX_SPACE_CONST4,
    PTX_SPACE_CONST5,
    PTX_SPACE_CONST6,
    PTX_SPACE_CONST7,
    PTX_SPACE_CONST8,
    PTX_SPACE_CONST9,
    PTX_SPACE_CONST10,
    PTX_SPACE_CONST11,
    PTX_SPACE_CONST12,
    PTX_SPACE_CONST13,
    PTX_SPACE_CONST14,
    PTX_SPACE_CONST15,
    PTX_SPACE_CONST16,
    PTX_SPACE_CONST17,
    PTX_SPACE_CONST_MAX = PTX_SPACE_CONST17,

    PTX_SPACE_GLOBAL_MIN,
    PTX_SPACE_GLOBAL0 = PTX_SPACE_GLOBAL_MIN,
    PTX_SPACE_GLOBAL1,
    PTX_SPACE_GLOBAL2,
    PTX_SPACE_GLOBAL3,
    PTX_SPACE_GLOBAL4,
    PTX_SPACE_GLOBAL5,
    PTX_SPACE_GLOBAL6,
    PTX_SPACE_GLOBAL7,
    PTX_SPACE_GLOBAL8,
    PTX_SPACE_GLOBAL9,
    PTX_SPACE_GLOBAL10,
    PTX_SPACE_GLOBAL11,
    PTX_SPACE_GLOBAL12,
    PTX_SPACE_GLOBAL13,
    PTX_SPACE_GLOBAL14,
    PTX_SPACE_GLOBAL15,
    PTX_SPACE_GLOBAL_MAX = PTX_SPACE_GLOBAL15,

    PTX_SPACE_GENERIC,

    PTX_SPACE_TEX,
    PTX_SPACE_SURF,
    PTX_SPACE_TEXSAMPLER,

    PTX_SPACE_PARAM,
    PTX_SPACE_IPARAM,  // Space for input parameters to non-entry functions
    PTX_SPACE_OPARAM,  // Space for output parameters to non-entry functions
    PTX_SPACE_UNIFIED_CONST, // Space to represent bankless const objects
    PTX_SPACE_MAX
} ptxMemSpace;

typedef enum ConstPointerKind_enum {
    KIND_OFFSET,
    KIND_FAT_ADDRESS,
    KIND_GENERIC
} ConstPointerKind;

typedef enum {
    PTX_SR_ILWALID,
    PTX_SR_LANEID,
    PTX_SR_WARPID,
    PTX_SR_SMID,
    PTX_SR_CTAID,
    PTX_SR_PM0,
    PTX_SR_PM1,
    PTX_SR_PM2,
    PTX_SR_PM3,
    PTX_SR_PM4,
    PTX_SR_PM5,
    PTX_SR_PM6,
    PTX_SR_PM7,
    PTX_SR_PM0_64,
    PTX_SR_PM1_64,
    PTX_SR_PM2_64,
    PTX_SR_PM3_64,
    PTX_SR_PM4_64,
    PTX_SR_PM5_64,
    PTX_SR_PM6_64,
    PTX_SR_PM7_64,
    PTX_SR_NSMID,
    PTX_SR_GRIDID,
    PTX_SR_EQMASK,
    PTX_SR_LTMASK,
    PTX_SR_LEMASK,
    PTX_SR_GTMASK,
    PTX_SR_GEMASK,
    PTX_SR_CLOCKLO,
    PTX_SR_CLOCKHI,
    PTX_SR_CLOCK64,
    PTX_SR_NWARPID,
    PTX_SR_AFFINITY,
    PTX_SR_ARRAYID,
    PTX_SR_GLOBALTIMERLO,
    PTX_SR_GLOBALTIMERHI,
    PTX_SR_GLOBALTIMER,
    PTX_SR_CQENTRYID,
    PTX_SR_CQENTRYADDR,
    PTX_SR_CQINCRMINUS1,
    PTX_SR_ISQUEUECTA,
    PTX_SR_ELWREG0,
    PTX_SR_ELWREG1,
    PTX_SR_ELWREG2,
    PTX_SR_ELWREG3,
    PTX_SR_ELWREG4,
    PTX_SR_ELWREG5,
    PTX_SR_ELWREG6,
    PTX_SR_ELWREG7,
    PTX_SR_ELWREG8,
    PTX_SR_ELWREG9,
    PTX_SR_ELWREG10,
    PTX_SR_ELWREG11,
    PTX_SR_ELWREG12,
    PTX_SR_ELWREG13,
    PTX_SR_ELWREG14,
    PTX_SR_ELWREG15,
    PTX_SR_ELWREG16,
    PTX_SR_ELWREG17,
    PTX_SR_ELWREG18,
    PTX_SR_ELWREG19,
    PTX_SR_ELWREG20,
    PTX_SR_ELWREG21,
    PTX_SR_ELWREG22,
    PTX_SR_ELWREG23,
    PTX_SR_ELWREG24,
    PTX_SR_ELWREG25,
    PTX_SR_ELWREG26,
    PTX_SR_ELWREG27,
    PTX_SR_ELWREG28,
    PTX_SR_ELWREG29,
    PTX_SR_ELWREG30,
    PTX_SR_ELWREG31,
    PTX_SR_BAR0,
    PTX_SR_BAR1,
    PTX_SR_BAR2,
    PTX_SR_BAR3,
    PTX_SR_BAR4,
    PTX_SR_BAR5,
    PTX_SR_BAR6,
    PTX_SR_BAR7,
    PTX_SR_BAR8,
    PTX_SR_BAR9,
    PTX_SR_BAR10,
    PTX_SR_BAR11,
    PTX_SR_BAR12,
    PTX_SR_BAR13,
    PTX_SR_BAR14,
    PTX_SR_BAR15,
    PTX_SR_BARWARP,
    PTX_SR_BARWARPRES,
    PTX_SR_BARWARPRESP,
    PTX_SR_SMEMSIZE,
    PTX_SR_DYNAMIC_SMEMSIZE,
    PTX_SR_TEX_QUERY_DESC_TABLE,
    PTX_SR_SAMP_QUERY_DESC_TABLE,
    PTX_SR_SURF_QUERY_DESC_TABLE,
    PTX_SR_USR_CONST_GENERIC_BASE,
    PTX_SR_DRIVER_CONST_GENERIC_BASE,
    PTX_SR_KPARAM_CONST_GENERIC_BASE,
    PTX_SR_RESERVED_SMEM_BEGIN,
    PTX_SR_RESERVED_SMEM_END,
    PTX_SR_RESERVED_SMEM_CAP,
    PTX_SR_RESERVED_SMEM_OFFSET_0,
    PTX_SR_RESERVED_SMEM_OFFSET_1,
    PTX_SR_VIRTUAL_ENGINEID,
    PTX_SR_HWTASKID,
    PTX_SR_NLATC,
    PTX_SR_PM0_SNAP_64,
    PTX_SR_PM1_SNAP_64,
    PTX_SR_PM2_SNAP_64,
    PTX_SR_PM3_SNAP_64,
    PTX_SR_PM4_SNAP_64,
    PTX_SR_PM5_SNAP_64,
    PTX_SR_PM6_SNAP_64,
    PTX_SR_PM7_SNAP_64,
    PTX_SR_STACKEND,
    PTX_SR_STACKINIT_ENTRY,
    PTX_SR_IS_CLUSTER_CTA,
    PTX_SR_CLUSTER_CTARANK,
    PTX_SR_CLUSTER_NCTARANK,
    PTX_SR_CLUSTERID_GPC,
} PtxSRegs;

typedef struct ptxDCI* ptxDCIHandle;

struct ptxDCI {
    /* APIs for querying Memory Limits */
    int (*GetMaxSharedMemoryPerCta)(void);
    int (*GetMaxSharedMemoryPerSm)(int SMArch);
    int (*GetSharedMemoryAllocSize)(void);
    int (*GetMaxLocalMemory)(void);
    int (*GetConstantBankSize)(void);
    int (*GetMaxTexturesPerEntry)(void);
    int (*GetMaxSamplersPerEntry)(void);
    int (*GetMaxSurfacesPerEntry)(void);
    int (*GetPCSize)(void);

    /* APIs for querying kernel parameters related information */
    elfWord (*GetParamConstBank)(void);
    int (*GetParamConstBase)(void);
    int (*GetMaxParamConstSize)(void);
    int (*GetParamSharedBase)(void);
    int (*GetMaxParamSharedSize)(void);

    /* APIs for querying constant bank mappings, 
     * banks are returned as elf SHT_ values; 
     * returns SHT_NULL if not applicable */
    // TODO : Do we need mappings for special register banks etc? Lwrrently this part of
    // DCI is handled completely in OCG and PTXAS, Linker doesn't need this information.
    elfWord (*GetOCGConstBank)(void);
    elfWord (*GetDriverConstBank)(void);
    elfWord (*GetDevtoolsConstBank)(void);
    elfWord (*GetGlobalRelocConstBank)(void);
    elfWord (*GetConstBackingStoreBank)(void);
    elfWord (*GetSRegConstBank)(PtxSRegs sreg);
    elfWord (*GetFirstUserConstBank)(void);

    Bool (*GetIsUserConstBank)(elfWord hwBank);
    Bool (*GetIsGlobalConstBank)(elfWord hwBank);
    Bool (*GetIsLocalConstBank)(elfWord hwBank);

    /* APIs for querying fixed offsets in DCI, return -1 if not applicable */
    int (*GetConstBackingStoreBase)(elfWord hwBank);
    int (*GetConstBackingStoreLogAlignment)(void);
    int (*GetSRegCBankBase)(PtxSRegs sreg);
    int (*GetInternallyReservedSMemCBankOffset)();
    int (*GetFastAliasFenceCBankOffset)();

    /* APIs related to texture, sampler, surface DCI */
    Bool (*GetUsesBindlessFormat)(void);
    elfWord (*GetBindlessTextureBank)(void);
    elfWord (*GetBindlessSurfaceBank)(void);
    Bool (*GetUsesUnifiedTextureSurfaceDescriptors)(void);

    Bool (*GetIsQueryDescConstBank)(elfWord bank);
    int (*GetTextureQueryDescriptorSize)(Bool TexModeUnified);
    int (*GetSamplerQueryDescriptorSize)(void);
    int (*GetSurfaceQueryDescriptorSize)(void);
    int (*GetTextureQueryDescriptorNumAttributes)(Bool TexModeUnified);
    int (*GetSamplerQueryDescriptorNumAttributes)(void);
    int (*GetSurfaceQueryDescriptorNumAttributes)(void);
    ptxMemSpace (*GetTextureQueryDescriptorStorage)(Bool IsLocal);
    ptxMemSpace (*GetSurfaceQueryDescriptorStorage)(Bool IsLocal);

    // For query descriptors stored in global memory, returns bank holding
    // address of descriptor table. Returns -1 if not applicable.
    elfWord (*GetGlobalQueryDescTableBank)(void);
    // For query descriptors stored in global memory, returns offset in bank holding
    // address of descriptor table. Returns -1 if not applicable.
    int (*GetGlobalTextureQueryDescTableBase)(void);
    int (*GetGlobalSamplerQueryDescTableBase)(void);
    int (*GetGlobalSurfaceQueryDescTableBase)(void);

    // For bindless textures, returns max offset from BindlessTextureBank which can be used as
    // immediate in TEX instruction. Bindless Offsets greater than this needs to be accessed with
    // Tex.B instruction. Returns -1 if not applicable.
    int (*GetBindlessTextureImmediateOffset)(void);

    int (*GetBindlessTextureHeaderIndexSize)(void);
    int (*GetBindlessSurfaceHeaderSize)(void);

    /* MISC APIs */
    
    // Returns True if global variables are accessed with indirection from GlobalRelocConstBank
    Bool (*GetUsesIndirectionForGlobals)(void);

    // Returns global memory segment to use for ptxGlobals. This is only applicable for Tesla, returns
    // 0 for others.
    int (*GetGlobalMemorySegment)(void);

    // Returns True if ptxas owns allocation of constant pointers to constant bank/global memory.
    Bool (*GetConstPointersAllocatedByPtxas)(void);

    // Returns constant pointer kind when constant pointers are used as kernel parameters
    ConstPointerKind (*GetConstPointerKind)(int NumConstPtrsUsed);

    // Returns HW storage used for ptx constant bank
    ptxMemSpace (*GetHWStorageForPtxCbank)(int ptxBank, Bool UseDriverConstBank);
    // Returns PTX constant bank corresponding to HW constant bank
    int (*GetPTXCbankForHWCbank)(elfWord hwBank);

    // Returns True if architecture supports 64bit Gridid
    Bool (*GetUses64bitGridid)(void);

    // Returns True if architecture supports CNP
    Bool (*GetSupportsCNP)(int SMArch);

    // Returns alignment required for code sections.
    int (*GetCodeSectionAlignment)(Bool isMerlwry);
    
    // Query if Launch bound computation depends on Caching turned ON/OFF
    Bool (*NeedCTASteeringForCaching)(int SMArch);
   
    // Check if RF/CTA limit is to be used 
    Bool (*GetUsesRFPerCTALimit)(int SMArch);
    
    // Hardware cache will be turned on based on cache request
    Bool (*NeedsCacheRequestFlagToEnableCache)(int SMArch);
    
    // Returns true if the special register gets mapped to the constant bank
    Bool (*IsSRegMapsToCBank)(PtxSRegs sreg);

    // Returns true if on the given SMArch, some portion of total SMEM is reserved for internal purpose.
    Bool (*IsSMemInternallyReserved)(int SMArch);
};

// Given SM arch, returns a handle which can be used to query DCI information
ptxDCIHandle InitDCIHandle(int SMArch);

#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
// like InitDCIHandle but returns DCI for Merlwry
ptxDCIHandle InitMerlwryDCIHandle(int SMArch);
#endif

// Frees the DCI Handle
void terminateDCIHandle(ptxDCIHandle *fDCIHandle);

#if     defined(__cplusplus)
}
#endif 

#endif // ptxDCI_INCLUDED

