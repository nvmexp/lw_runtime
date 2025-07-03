/*
 *  Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 *  NOTICE TO USER: The source code, and related code and software
 *  ("Code"), is copyrighted under U.S. and international laws.
 *
 *  LWPU Corporation owns the copyright and any patents issued or
 *  pending for the Code.
 *
 *  LWPU CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY
 *  OF THIS CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS-IS" WITHOUT EXPRESS
 *  OR IMPLIED WARRANTY OF ANY KIND.  LWPU CORPORATION DISCLAIMS ALL
 *  WARRANTIES WITH REGARD TO THE CODE, INCLUDING NON-INFRINGEMENT, AND
 *  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE.  IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 *  WHATSOEVER ARISING OUT OF OR IN ANY WAY RELATED TO THE USE OR
 *  PERFORMANCE OF THE CODE, INCLUDING, BUT NOT LIMITED TO, INFRINGEMENT,
 *  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 *  NEGLIGENCE OR OTHER TORTIOUS ACTION, AND WHETHER OR NOT THE
 *  POSSIBILITY OF SUCH DAMAGES WERE KNOWN OR MADE KNOWN TO LWPU
 *  CORPORATION.
 *
 */

#include "g_lwconfig.h"
#include "stdLocal.h"
#include "ptxDCI.h"

static ptxDCIHandle dciHandle;

static PtxArch GetPtxArchFromSM(int SMArch)
{
    switch (SMArch) {
    case 10: return PTX_ARCH_SM_10;
    case 11: return PTX_ARCH_SM_11;
    case 12: return PTX_ARCH_SM_12;
    case 13: return PTX_ARCH_SM_13;
    case 20: return PTX_ARCH_SM_20;
    case 21: return PTX_ARCH_SM_21;
    case 30: return PTX_ARCH_SM_30;
    case 35: return PTX_ARCH_SM_35;
#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
    case 32: return PTX_ARCH_SM_32;
#endif
#if LWCFG(GLOBAL_GPU_IMPL_GK110C)
    case 37: return PTX_ARCH_SM_37;
#endif
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    case 50: return PTX_ARCH_SM_50;
#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
    case 52: return PTX_ARCH_SM_52;
    case 53: return PTX_ARCH_SM_53;
#endif // GLOBAL_GPU_FAMILY_GM20X
#endif // GLOBAL_ARCH_MAXWELL
    default:
             stdASSERT(0, ("Unknown Arch"));
             return PTX_ARCH_TESLA_MIN;
    }
}

static void InitTeslaDCI(ptxDCIHandle teslaDCI);
static void InitFermiDCI(ptxDCIHandle fermiDCI);
static void InitKeplerDCI(ptxDCIHandle keplerDCI);
#if LWCFG(GLOBAL_ARCH_MAXWELL)
static void InitMaxwellDCI(ptxDCIHandle maxwellDCI);
#endif

// Given SM arch, returns a handle which can be used to query DCI information
ptxDCIHandle InitDCIHandle(int SMArch)
{
    PtxArch arch;

    arch = GetPtxArchFromSM(SMArch);
    dciHandle = (ptxDCIHandle) stdMALLOC(sizeof(struct ptxDCI));

    if (IS_TESLA(arch)) {
        InitTeslaDCI(dciHandle);
    } else if (IS_FERMI(arch)) {
        InitFermiDCI(dciHandle);
    } else if (IS_KEPLER(arch)) {
        InitKeplerDCI(dciHandle);
#if LWCFG(GLOBAL_ARCH_MAXWELL)
    } else if (IS_MAXWELL(arch)) {
        InitMaxwellDCI(dciHandle);
#endif
    } else {
        stdASSERT(0, ("Unknown arch"));
        InitTeslaDCI(dciHandle);
    }
    return dciHandle;
}

// Frees the DCI Handle
void terminateDCIHandle(ptxDCIHandle *fDCIHandle)
{
    stdFREE(*fDCIHandle);
    *fDCIHandle = NULL;
}

static int GetTextureQueryDescriptorSize_common(Bool TexModeUnified)
{
    if (TexModeUnified) {
        return getUnifiedTexrefAttributeOffset(LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX) +
            getUnifiedTexrefAttributeSize(LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX);
    } else {
        return getIndependentTexrefAttributeOffset(LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX) +
            getIndependentTexrefAttributeSize(LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX);
    }
}

static int GetSamplerQueryDescriptorSize_common(void)
{
    return getSamplerrefAttributeOffset(LWBIN_SAMPLERREF_ATTRIBUTE_MAX) +
        getSamplerrefAttributeSize(LWBIN_SAMPLERREF_ATTRIBUTE_MAX);
}

static int GetSurfaceQueryDescriptorSize_common(void)
{
    return getSurfrefAttributeOffset(LWBIN_SURFREF_ATTRIBUTE_MAX) +
        getSurfrefAttributeSize(LWBIN_SURFREF_ATTRIBUTE_MAX);
}

static int GetTextureQueryDescriptorNumAttributes_common(Bool TexModeUnified)
{
    if (TexModeUnified) {
        return LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX + 1;
    } else {
        return LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX + 1;
    }
}

static int GetSamplerQueryDescriptorNumAttributes_common(void)
{
    return LWBIN_SAMPLERREF_ATTRIBUTE_MAX + 1;
}

static int GetSurfaceQueryDescriptorNumAttributes_common(void)
{
    return LWBIN_SURFREF_ATTRIBUTE_MAX + 1;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////TESLA DCI///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

#define TESLA_MAX_SHARED_MEMORY                 (16 * 1024) // 16K
#define TESLA_MAX_LOCAL_MEMORY                  (16 * 1024) // 16K
#define TESLA_CONST_BANK_SIZE                   (64 * 1024) // 64K
#define TESLA_NUM_TEXTURES_PER_ENTRY            128
#define TESLA_NUM_SAMPLERS_PER_ENTRY            16
#define TESLA_NUM_SURFACES_PER_ENTRY            8

#define TESLA_USER_CONST_BANK_MIN               0
#define TESLA_USER_CONST_BANK_MAX               11
#define TESLA_GLOBAL_RELOC_CONST_BANK           14
#define TESLA_PARAM_CONST_BANK                  13
#define TESLA_GLOBAL_REF_DESCRIPTOR_CONST_BANK  14
#define TESLA_LOCAL_REF_DESCRIPTOR_CONST_BANK   15
#define TESLA_OCG_CONST_BANK                    1
#define TESLA_ELWREGS_CONST_BANK                12
#define TESLA_GLOBAL_MEMORY_SEGMENT             14

#define TESLA_PARAM_CBANK_BASE                  16
#define TESLA_PARAM_CBANK_SIZE                  (4352 - TESLA_PARAM_CBANK_BASE)
#define TESLA_PARAM_SMEM_BASE                   16
#define TESLA_PARAM_SMEM_SIZE                   256
#define TESLA_LAST_SMEM_LOC_FOR_PARAMS          (TESLA_PARAM_SMEM_BASE + TESLA_PARAM_SMEM_SIZE)
#define TESLA_ELWREGS_CBANK_BASE                0

#define TESLA_CODE_ALIGN                        4

/* APIs for querying Memory Limits */
static int GetMaxSharedMemory_tesla(void) { return TESLA_MAX_SHARED_MEMORY; }
static int GetMaxLocalMemory_tesla(void) { return TESLA_MAX_LOCAL_MEMORY; }
static int GetConstantBankSize_tesla(void) { return TESLA_CONST_BANK_SIZE; }
static int GetMaxTexturesPerEntry_tesla(void) { return TESLA_NUM_TEXTURES_PER_ENTRY; }
static int GetMaxSamplersPerEntry_tesla(void) { return TESLA_NUM_SAMPLERS_PER_ENTRY; }
static int GetMaxSurfacesPerEntry_tesla(void) { return TESLA_NUM_SURFACES_PER_ENTRY; }

/* APIs for querying kernel parameters related information */
static int GetParamConstBank_tesla(void) { return TESLA_PARAM_CONST_BANK; }
static int GetParamConstBase_tesla(void) { return TESLA_PARAM_CBANK_BASE; }
static int GetMaxParamConstSize_tesla(void) { return TESLA_PARAM_CBANK_SIZE; }
static int GetParamSharedBase_tesla(void) { return TESLA_PARAM_SMEM_BASE; }
static int GetMaxParamSharedSize_tesla(void) { return TESLA_PARAM_SMEM_SIZE; }

/* APIs for querying constant bank mappings, returns -1 if not applicable */
static int GetElwregConstBank_tesla(void) { return TESLA_ELWREGS_CONST_BANK; }
static int GetOCGConstBank_tesla(void) { return TESLA_OCG_CONST_BANK; }
static int GetDriverConstantBank_tesla(void) { return -1; }
static int GetGlobalRelocConstBank_tesla(void) { return TESLA_GLOBAL_RELOC_CONST_BANK; }
static int GetConstBackingStoreBank_tesla(void) { return -1; }

static Bool GetIsUserConstBank_tesla(int hwBank)
{
    return (hwBank >= TESLA_USER_CONST_BANK_MIN && hwBank <= TESLA_USER_CONST_BANK_MAX &&
            hwBank != TESLA_OCG_CONST_BANK);
}

static Bool GetIsGlobalConstBank_tesla(int hwBank)
{
    return (GetIsUserConstBank_tesla(hwBank) || (hwBank == TESLA_GLOBAL_RELOC_CONST_BANK));
}

static Bool GetIsLocalConstBank_tesla(int hwBank)
{
    return !GetIsGlobalConstBank_tesla(hwBank);
}

/* APIs for querying fixed offsets in DCI, return -1 if not applicable */
static int GetElwregConstBase_tesla(void) { return TESLA_ELWREGS_CBANK_BASE; }
static int GetConstBackingStoreBase_tesla(int hwBank) { return -1; }

/* APIs related to texture, sampler, surface DCI */
static Bool GetUsesBindlessFormat_tesla(void) { return False; }
static Bool GetUsesUnifiedTextureSurfaceDescriptors_tesla(void) { return False; }
static int GetBindlessTextureBank_tesla(void) { return -1; }
static int GetBindlessSurfaceBank_tesla(void) { return -1; }
static int GetBindlessTextureHeaderIndexSize_tesla(void ) { return -1; }
static int GetBindlessSurfaceHeaderSize_tesla(void) { return -1; }
static Bool GetUsesLayeredSurfaceEmulation_tesla(void) { return False; }

static Bool GetIsQueryDescBank_tesla(int hwBank)
{
    return ((hwBank == TESLA_GLOBAL_REF_DESCRIPTOR_CONST_BANK) ||
            (hwBank == TESLA_LOCAL_REF_DESCRIPTOR_CONST_BANK));
}

static ptxMemSpace GetQueryDescriptorStorage_tesla(Bool IsLocal)
{
    if (IsLocal) {
        return (ptxMemSpace)(PTX_SPACE_CONST0 + TESLA_LOCAL_REF_DESCRIPTOR_CONST_BANK);
    }
    return (ptxMemSpace)(PTX_SPACE_CONST0 + TESLA_GLOBAL_REF_DESCRIPTOR_CONST_BANK);
}

static ptxMemSpace GetTextureQueryDescriptorStorage_tesla(Bool IsLocal)
{
    return GetQueryDescriptorStorage_tesla(IsLocal);
}

static ptxMemSpace GetSurfaceQueryDescriptorStorage_tesla(Bool IsLocal)
{
    return GetQueryDescriptorStorage_tesla(IsLocal);
}

static int GetGlobalQueryDescTableBank_tesla(void) { return -1; }
static int GetGlobalTextureQueryDescTableBase_tesla(void) { return -1; }
static int GetGlobalSamplerQueryDescTableBase_tesla(void) { return -1; }
static int GetGlobalSurfaceQueryDescTableBase_tesla(void) { return -1; }
static int GetBindlessTextureImmediateOffset_tesla(void) { return -1; }

/* MISC APIs */
static Bool GetUsesIndirectionForGlobals_tesla(void) { return True; }

static int GetGlobalMemorySegment_tesla(void) { return TESLA_GLOBAL_MEMORY_SEGMENT; }

static Bool GetConstPointersAllocatedByPtxas_tesla(void) { return True; }

static ConstPointerKind GetConstPointerKind_tesla(int NumConstPtrsUsed)
{
    if (NumConstPtrsUsed) {
        return KIND_FAT_ADDRESS;
    } else {
        return KIND_OFFSET;
    }
}

static ptxMemSpace GetHWStorageForPtxCbank_tesla(int ptxBank, Bool UseDriverConstBank)
{
    if (UseDriverConstBank) {
        stdASSERT(0, ("Driver constant bank not supported"));
        return PTX_SPACE_NONE;
    } else {
        // PTX bank 0 => HW bank 0
        // PTX bank 1-10 => HW bank 2-11
        if (ptxBank == 0) {
            return PTX_SPACE_CONST0;
        } else {
            return (ptxMemSpace)(PTX_SPACE_CONST0 + (ptxBank + 1));
        }
    }
}

static int GetPTXCbankForHWCbank_tesla(int hwBank)
{
    if (hwBank == 0) {
        return 0;
    } else if (hwBank >= 2 && hwBank <= TESLA_USER_CONST_BANK_MAX) {
        return hwBank - 1;
    } else {
        return -1;
    }
}

static Bool GetUses64bitGridid_tesla(void) { return False; }
static Bool GetSupportsCNP_tesla(int SMArch) { return False; }
static int GetCodeSectionAlignment_tesla(void) { return TESLA_CODE_ALIGN; }

static void InitTeslaDCI(ptxDCIHandle teslaDCI)
{
    teslaDCI->GetMaxSharedMemory = GetMaxSharedMemory_tesla;
    teslaDCI->GetMaxLocalMemory = GetMaxLocalMemory_tesla;
    teslaDCI->GetConstantBankSize = GetConstantBankSize_tesla;
    teslaDCI->GetMaxTexturesPerEntry = GetMaxTexturesPerEntry_tesla;
    teslaDCI->GetMaxSamplersPerEntry = GetMaxSamplersPerEntry_tesla;
    teslaDCI->GetMaxSurfacesPerEntry = GetMaxSurfacesPerEntry_tesla;

    teslaDCI->GetParamConstBank = GetParamConstBank_tesla;
    teslaDCI->GetParamConstBase = GetParamConstBase_tesla;
    teslaDCI->GetMaxParamConstSize = GetMaxParamConstSize_tesla;
    teslaDCI->GetParamSharedBase = GetParamSharedBase_tesla;
    teslaDCI->GetMaxParamSharedSize = GetMaxParamSharedSize_tesla;

    teslaDCI->GetElwregConstBank = GetElwregConstBank_tesla;
    teslaDCI->GetOCGConstBank = GetOCGConstBank_tesla;
    teslaDCI->GetDriverConstBank = GetDriverConstantBank_tesla;
    teslaDCI->GetGlobalRelocConstBank = GetGlobalRelocConstBank_tesla;
    teslaDCI->GetConstBackingStoreBank = GetConstBackingStoreBank_tesla;

    teslaDCI->GetIsUserConstBank = GetIsUserConstBank_tesla;
    teslaDCI->GetIsGlobalConstBank = GetIsGlobalConstBank_tesla;
    teslaDCI->GetIsLocalConstBank = GetIsLocalConstBank_tesla;

    teslaDCI->GetElwregConstBase = GetElwregConstBase_tesla;
    teslaDCI->GetConstBackingStoreBase = GetConstBackingStoreBase_tesla;

    teslaDCI->GetUsesBindlessFormat = GetUsesBindlessFormat_tesla;
    teslaDCI->GetUsesUnifiedTextureSurfaceDescriptors = GetUsesUnifiedTextureSurfaceDescriptors_tesla;
    teslaDCI->GetBindlessTextureBank = GetBindlessTextureBank_tesla;
    teslaDCI->GetBindlessSurfaceBank = GetBindlessSurfaceBank_tesla;
    teslaDCI->GetIsQueryDescConstBank = GetIsQueryDescBank_tesla;
    teslaDCI->GetUsesLayeredSurfaceEmulation = GetUsesLayeredSurfaceEmulation_tesla;
    teslaDCI->GetTextureQueryDescriptorSize = GetTextureQueryDescriptorSize_common;
    teslaDCI->GetSamplerQueryDescriptorSize = GetSamplerQueryDescriptorSize_common;
    teslaDCI->GetSurfaceQueryDescriptorSize = GetSurfaceQueryDescriptorSize_common;
    teslaDCI->GetTextureQueryDescriptorNumAttributes = GetTextureQueryDescriptorNumAttributes_common;
    teslaDCI->GetSamplerQueryDescriptorNumAttributes = GetSamplerQueryDescriptorNumAttributes_common;
    teslaDCI->GetSurfaceQueryDescriptorNumAttributes = GetSurfaceQueryDescriptorNumAttributes_common;
    teslaDCI->GetTextureQueryDescriptorStorage = GetTextureQueryDescriptorStorage_tesla;
    teslaDCI->GetSurfaceQueryDescriptorStorage = GetSurfaceQueryDescriptorStorage_tesla;
    teslaDCI->GetGlobalQueryDescTableBank = GetGlobalQueryDescTableBank_tesla;
    teslaDCI->GetGlobalTextureQueryDescTableBase = GetGlobalTextureQueryDescTableBase_tesla;
    teslaDCI->GetGlobalSamplerQueryDescTableBase = GetGlobalSamplerQueryDescTableBase_tesla;
    teslaDCI->GetGlobalSurfaceQueryDescTableBase = GetGlobalSurfaceQueryDescTableBase_tesla;
    teslaDCI->GetBindlessTextureImmediateOffset = GetBindlessTextureImmediateOffset_tesla;
    teslaDCI->GetBindlessTextureHeaderIndexSize = GetBindlessTextureHeaderIndexSize_tesla;
    teslaDCI->GetBindlessSurfaceHeaderSize = GetBindlessSurfaceHeaderSize_tesla;

    teslaDCI->GetUsesIndirectionForGlobals = GetUsesIndirectionForGlobals_tesla;
    teslaDCI->GetGlobalMemorySegment = GetGlobalMemorySegment_tesla;
    teslaDCI->GetConstPointersAllocatedByPtxas = GetConstPointersAllocatedByPtxas_tesla;
    teslaDCI->GetConstPointerKind = GetConstPointerKind_tesla;
    teslaDCI->GetHWStorageForPtxCbank = GetHWStorageForPtxCbank_tesla;
    teslaDCI->GetPTXCbankForHWCbank = GetPTXCbankForHWCbank_tesla;
    teslaDCI->GetUses64bitGridid = GetUses64bitGridid_tesla;
    teslaDCI->GetSupportsCNP = GetSupportsCNP_tesla;
    teslaDCI->GetCodeSectionAlignment = GetCodeSectionAlignment_tesla;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////FERMI DCI///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

#define FERMI_MAX_SHARED_MEMORY                 (48 * 1024) // 48K
#define FERMI_MAX_LOCAL_MEMORY                  (16 * 1024 * 1024) // 16M
#define FERMI_CONST_BANK_SIZE                   (64 * 1024) // 64K
#define FERMI_NUM_TEXTURES_PER_ENTRY            128
#define FERMI_NUM_SAMPLERS_PER_ENTRY            16
#define FERMI_NUM_SURFACES_PER_ENTRY            8

#define FERMI_USER_CONST_BANK_MIN               2
#define FERMI_USER_CONST_BANK_MAX               12
#define FERMI_GLOBAL_RELOC_CONST_BANK           14
#define FERMI_PARAM_CONST_BANK                  0
#define FERMI_GLOBAL_REF_DESCRIPTOR_CONST_BANK  14
#define FERMI_LOCAL_REF_DESCRIPTOR_CONST_BANK   15
#define FERMI_OCG_CONST_BANK                    16
#define FERMI_DRIVER_CONST_BANK                 1
#define FERMI_ELWREGS_CONST_BANK                1
#define FERMI_NUM_PTX_BANKS                     11
#define FERMI_GLOBAL_MEMORY_SEGMENT             0

#define FERMI_PARAM_CBANK_BASE                  32
#define FERMI_PARAM_CBANK_SIZE                  (4096 + 256)
#define FERMI_PARAM_SMEM_BASE                   0
#define FERMI_PARAM_SMEM_SIZE                   0
#define FERMI_ELWREGS_CBANK_BASE                0
#define FERMI_CONST_BACKING_STORE_CBANK         1
#define FERMI_CONST_BACKING_STORE_BASE          128
#define FERMI_PARAM_BANK_BACKING_STORE_BASE     216
#define FERMI_DRIVER_BANK_BACKING_STORE_BASE    232

#define FERMI_CODE_ALIGN                        4

/* APIs for querying Memory Limits */
static int GetMaxSharedMemory_fermi(void) { return FERMI_MAX_SHARED_MEMORY; }
static int GetMaxLocalMemory_fermi(void) { return FERMI_MAX_LOCAL_MEMORY; }
static int GetConstantBankSize_fermi(void) { return FERMI_CONST_BANK_SIZE; }
static int GetMaxTexturesPerEntry_fermi(void) { return FERMI_NUM_TEXTURES_PER_ENTRY; }
static int GetMaxSamplersPerEntry_fermi(void) { return FERMI_NUM_SAMPLERS_PER_ENTRY; }
static int GetMaxSurfacesPerEntry_fermi(void) { return FERMI_NUM_SURFACES_PER_ENTRY; }

/* APIs for querying kernel parameters related information */
static int GetParamConstBank_fermi(void) { return FERMI_PARAM_CONST_BANK; }
static int GetParamConstBase_fermi(void) { return FERMI_PARAM_CBANK_BASE; }
static int GetMaxParamConstSize_fermi(void) { return FERMI_PARAM_CBANK_SIZE; }
static int GetParamSharedBase_fermi(void) { return FERMI_PARAM_SMEM_BASE; }
static int GetMaxParamSharedSize_fermi(void) { return FERMI_PARAM_SMEM_SIZE; }

/* APIs for querying constant bank mappings, returns -1 if not applicable */
static int GetElwregConstBank_fermi(void) { return FERMI_ELWREGS_CONST_BANK; }
static int GetOCGConstBank_fermi(void) { return FERMI_OCG_CONST_BANK; }
static int GetDriverConstantBank_fermi(void) { return FERMI_DRIVER_CONST_BANK; }
static int GetGlobalRelocConstBank_fermi(void) { return FERMI_GLOBAL_RELOC_CONST_BANK; }
static int GetConstBackingStoreBank_fermi(void) { return FERMI_CONST_BACKING_STORE_CBANK; }

static Bool GetIsUserConstBank_fermi(int hwBank)
{
    return ((hwBank >= FERMI_USER_CONST_BANK_MIN && hwBank <= FERMI_USER_CONST_BANK_MAX) ||
            hwBank == FERMI_DRIVER_CONST_BANK);
}

static Bool GetIsGlobalConstBank_fermi(int hwBank)
{
    return (GetIsUserConstBank_fermi(hwBank) || (hwBank == FERMI_GLOBAL_RELOC_CONST_BANK));
}

static Bool GetIsLocalConstBank_fermi(int hwBank)
{
    return !GetIsGlobalConstBank_fermi(hwBank);
}

/* APIs for querying fixed offsets in DCI, return -1 if not applicable */
static int GetElwregConstBase_fermi(void) { return FERMI_ELWREGS_CBANK_BASE; }
static int GetConstBackingStoreBase_fermi(int hwBank)
{ 
    if (hwBank == FERMI_PARAM_CONST_BANK) {
        return FERMI_PARAM_BANK_BACKING_STORE_BASE;
    } else if (hwBank == FERMI_DRIVER_CONST_BANK) {
        return FERMI_DRIVER_BANK_BACKING_STORE_BASE;
    } else {
        stdASSERT(hwBank >= FERMI_USER_CONST_BANK_MIN && hwBank <= FERMI_USER_CONST_BANK_MAX, ("Unexpected bank"));
        // Backing store address is 64bit
        return FERMI_CONST_BACKING_STORE_BASE + (hwBank - FERMI_USER_CONST_BANK_MIN) * 8;
    }
}

/* APIs related to texture, sampler, surface DCI */
static Bool GetUsesBindlessFormat_fermi(void) { return False; }
static Bool GetUsesUnifiedTextureSurfaceDescriptors_fermi(void) { return False; }
static int GetBindlessTextureBank_fermi(void) { return -1; }
static int GetBindlessSurfaceBank_fermi(void) { return -1; }
static Bool GetUsesLayeredSurfaceEmulation_fermi(void) { return True; }

static Bool GetIsQueryDescBank_fermi(int hwBank)
{
    return ((hwBank == FERMI_GLOBAL_REF_DESCRIPTOR_CONST_BANK) ||
            (hwBank == FERMI_LOCAL_REF_DESCRIPTOR_CONST_BANK));
}

static ptxMemSpace GetQueryDescriptorStorage_fermi(Bool IsLocal)
{
    if (IsLocal) {
        return (ptxMemSpace)(PTX_SPACE_CONST0 + FERMI_LOCAL_REF_DESCRIPTOR_CONST_BANK);
    }
    return (ptxMemSpace)(PTX_SPACE_CONST0 + FERMI_GLOBAL_REF_DESCRIPTOR_CONST_BANK);
}

static ptxMemSpace GetTextureQueryDescriptorStorage_fermi(Bool IsLocal)
{
    return GetQueryDescriptorStorage_fermi(IsLocal);
}

static ptxMemSpace GetSurfaceQueryDescriptorStorage_fermi(Bool IsLocal)
{
    return GetQueryDescriptorStorage_fermi(IsLocal);
}

static int GetGlobalQueryDescTableBank_fermi(void) { return -1; }
static int GetGlobalTextureQueryDescTableBase_fermi(void) { return -1; }
static int GetGlobalSamplerQueryDescTableBase_fermi(void) { return -1; }
static int GetGlobalSurfaceQueryDescTableBase_fermi(void) { return -1; }
static int GetBindlessTextureImmediateOffset_fermi(void) { return -1; }
static int GetBindlessTextureHeaderIndexSize_fermi(void ) { return -1; }
static int GetBindlessSurfaceHeaderSize_fermi(void ) { return -1; }

/* MISC APIs */
static Bool GetUsesIndirectionForGlobals_fermi(void) { return True; }

static int GetGlobalMemorySegment_fermi(void) { return FERMI_GLOBAL_MEMORY_SEGMENT; }

static Bool GetConstPointersAllocatedByPtxas_fermi(void) { return False; }

static ConstPointerKind GetConstPointerKind_fermi(int NumConstPtrsUsed)
{
    if (NumConstPtrsUsed) {
        return KIND_FAT_ADDRESS;
    } else {
        return KIND_OFFSET;
    }
}

static ptxMemSpace GetHWStorageForPtxCbank_fermi(int ptxBank, Bool UseDriverConstBank)
{
    if (UseDriverConstBank) {
        stdASSERT(ptxBank == 0, ("Bank 0 expected"));
        return (ptxMemSpace)(PTX_SPACE_CONST0 + GetDriverConstantBank_fermi());
    } else {
        return (ptxMemSpace)(PTX_SPACE_CONST0 + (FERMI_USER_CONST_BANK_MIN + ptxBank));
    }
}

static int GetPTXCbankForHWCbank_fermi(int hwBank)
{
    if (hwBank >= FERMI_USER_CONST_BANK_MIN && hwBank <= FERMI_USER_CONST_BANK_MAX) {
        return hwBank - FERMI_USER_CONST_BANK_MIN;
    } else if (hwBank == FERMI_DRIVER_CONST_BANK) {
        return 0;
    } else {
        return -1;
    }
}

static Bool GetUses64bitGridid_fermi(void) { return False; }
static Bool GetSupportsCNP_fermi(int SMArch) { return False; }
static int GetCodeSectionAlignment_fermi(void) { return FERMI_CODE_ALIGN; }

static void InitFermiDCI(ptxDCIHandle fermiDCI)
{
    fermiDCI->GetMaxSharedMemory = GetMaxSharedMemory_fermi;
    fermiDCI->GetMaxLocalMemory = GetMaxLocalMemory_fermi;
    fermiDCI->GetConstantBankSize = GetConstantBankSize_fermi;
    fermiDCI->GetMaxTexturesPerEntry = GetMaxTexturesPerEntry_fermi;
    fermiDCI->GetMaxSamplersPerEntry = GetMaxSamplersPerEntry_fermi;
    fermiDCI->GetMaxSurfacesPerEntry = GetMaxSurfacesPerEntry_fermi;

    fermiDCI->GetParamConstBank = GetParamConstBank_fermi;
    fermiDCI->GetParamConstBase = GetParamConstBase_fermi;
    fermiDCI->GetMaxParamConstSize = GetMaxParamConstSize_fermi;
    fermiDCI->GetParamSharedBase = GetParamSharedBase_fermi;
    fermiDCI->GetMaxParamSharedSize = GetMaxParamSharedSize_fermi;

    fermiDCI->GetElwregConstBank = GetElwregConstBank_fermi;
    fermiDCI->GetOCGConstBank = GetOCGConstBank_fermi;
    fermiDCI->GetDriverConstBank = GetDriverConstantBank_fermi;
    fermiDCI->GetGlobalRelocConstBank = GetGlobalRelocConstBank_fermi;
    fermiDCI->GetConstBackingStoreBank = GetConstBackingStoreBank_fermi;

    fermiDCI->GetIsUserConstBank = GetIsUserConstBank_fermi;
    fermiDCI->GetIsGlobalConstBank = GetIsGlobalConstBank_fermi;
    fermiDCI->GetIsLocalConstBank = GetIsLocalConstBank_fermi;

    fermiDCI->GetElwregConstBase = GetElwregConstBase_fermi;
    fermiDCI->GetConstBackingStoreBase = GetConstBackingStoreBase_fermi;

    fermiDCI->GetUsesBindlessFormat = GetUsesBindlessFormat_fermi;
    fermiDCI->GetUsesUnifiedTextureSurfaceDescriptors = GetUsesUnifiedTextureSurfaceDescriptors_fermi;
    fermiDCI->GetBindlessTextureBank = GetBindlessTextureBank_fermi;
    fermiDCI->GetBindlessSurfaceBank = GetBindlessSurfaceBank_fermi;
    fermiDCI->GetIsQueryDescConstBank = GetIsQueryDescBank_fermi;
    fermiDCI->GetUsesLayeredSurfaceEmulation = GetUsesLayeredSurfaceEmulation_fermi;
    fermiDCI->GetTextureQueryDescriptorSize = GetTextureQueryDescriptorSize_common;
    fermiDCI->GetSamplerQueryDescriptorSize = GetSamplerQueryDescriptorSize_common;
    fermiDCI->GetSurfaceQueryDescriptorSize = GetSurfaceQueryDescriptorSize_common;
    fermiDCI->GetTextureQueryDescriptorNumAttributes = GetTextureQueryDescriptorNumAttributes_common;
    fermiDCI->GetSamplerQueryDescriptorNumAttributes = GetSamplerQueryDescriptorNumAttributes_common;
    fermiDCI->GetSurfaceQueryDescriptorNumAttributes = GetSurfaceQueryDescriptorNumAttributes_common;
    fermiDCI->GetTextureQueryDescriptorStorage = GetTextureQueryDescriptorStorage_fermi;
    fermiDCI->GetSurfaceQueryDescriptorStorage = GetSurfaceQueryDescriptorStorage_fermi;
    fermiDCI->GetGlobalQueryDescTableBank = GetGlobalQueryDescTableBank_fermi;
    fermiDCI->GetGlobalTextureQueryDescTableBase = GetGlobalTextureQueryDescTableBase_fermi;
    fermiDCI->GetGlobalSamplerQueryDescTableBase = GetGlobalSamplerQueryDescTableBase_fermi;
    fermiDCI->GetGlobalSurfaceQueryDescTableBase = GetGlobalSurfaceQueryDescTableBase_fermi;
    fermiDCI->GetBindlessTextureImmediateOffset = GetBindlessTextureImmediateOffset_fermi;
    fermiDCI->GetBindlessTextureHeaderIndexSize = GetBindlessTextureHeaderIndexSize_fermi;
    fermiDCI->GetBindlessSurfaceHeaderSize = GetBindlessSurfaceHeaderSize_fermi;

    fermiDCI->GetUsesIndirectionForGlobals = GetUsesIndirectionForGlobals_fermi;
    fermiDCI->GetGlobalMemorySegment = GetGlobalMemorySegment_fermi;
    fermiDCI->GetConstPointersAllocatedByPtxas = GetConstPointersAllocatedByPtxas_fermi;
    fermiDCI->GetConstPointerKind = GetConstPointerKind_fermi;
    fermiDCI->GetHWStorageForPtxCbank = GetHWStorageForPtxCbank_fermi;
    fermiDCI->GetPTXCbankForHWCbank = GetPTXCbankForHWCbank_fermi;
    fermiDCI->GetUses64bitGridid = GetUses64bitGridid_fermi;
    fermiDCI->GetSupportsCNP = GetSupportsCNP_fermi;
    fermiDCI->GetCodeSectionAlignment = GetCodeSectionAlignment_fermi;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////KEPLER DCI//////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

#define KEPLER_MAX_SHARED_MEMORY                (48 * 1024) // 48K
#define KEPLER_MAX_LOCAL_MEMORY                 (16 * 1024 * 1024) // 16M
#define KEPLER_CONST_BANK_SIZE                  (64 * 1024) // 64K
#define KEPLER_NUM_TEXTURES_PER_ENTRY           256
#define KEPLER_NUM_SAMPLERS_PER_ENTRY           32
#define KEPLER_NUM_SURFACES_PER_ENTRY           16

#define KEPLER_USER_CONST_BANK_MIN              3
#define KEPLER_USER_CONST_BANK_MAX              5
#define KEPLER_PARAM_CONST_BANK                 0
#define KEPLER_OCG_CONST_BANK                   2
#define KEPLER_DRIVER_CONST_BANK                1
#define KEPLER_ELWREGS_CONST_BANK               0
#define KEPLER_BINDLESS_TEX_BANK                0
#define KEPLER_BINDLESS_SURF_BANK               0
#define KEPLER_NUM_PTX_BANKS                    3
#define KEPLER_BINDLESS_SURF_HEADER_SIZE        (8 * 4)
#define KEPLER_BINDLESS_TEX_HEADER_INDEX_SIZE   4
#define KEPLER_BINDLESS_IMM_TEX_OFFSET          (8192 * 4)
#define KEPLER_GLOBAL_MEMORY_SEGMENT            0

#define KEPLER_PARAM_CBANK_BASE                 320
#define KEPLER_PARAM_CBANK_SIZE                 (4096 + 256)
#define KEPLER_PARAM_SMEM_BASE                  0
#define KEPLER_PARAM_SMEM_SIZE                  0
#define KEPLER_ELWREGS_CBANK_BASE               76
#define KEPLER_SW_DESC_TABLE_BANK               0
#define KEPLER_TEX_SW_DESC_TABLE_OFFSET         204
#define KEPLER_SAMP_SW_DESC_TABLE_OFFSET        212
#define KEPLER_CONST_BACKING_STORE_CBANK        0
#define KEPLER_CONST_BACKING_STORE_BASE         240
#define KEPLER_PARAM_BANK_BACKING_STORE_BASE    232
#define KEPLER_DRIVER_BANK_BACKING_STORE_BASE   280

#define KEPLER_CODE_ALIGN                       64

/* APIs for querying Memory Limits */
static int GetMaxSharedMemory_kepler(void) { return KEPLER_MAX_SHARED_MEMORY; }
static int GetMaxLocalMemory_kepler(void) { return KEPLER_MAX_LOCAL_MEMORY; }
static int GetConstantBankSize_kepler(void) { return KEPLER_CONST_BANK_SIZE; }
static int GetMaxTexturesPerEntry_kepler(void) { return KEPLER_NUM_TEXTURES_PER_ENTRY; }
static int GetMaxSamplersPerEntry_kepler(void) { return KEPLER_NUM_SAMPLERS_PER_ENTRY; }
static int GetMaxSurfacesPerEntry_kepler(void) { return KEPLER_NUM_SURFACES_PER_ENTRY; }

/* APIs for querying kernel parameters related information */
static int GetParamConstBank_kepler(void) { return KEPLER_PARAM_CONST_BANK; }
static int GetParamConstBase_kepler(void) { return KEPLER_PARAM_CBANK_BASE; }
static int GetMaxParamConstSize_kepler(void) { return KEPLER_PARAM_CBANK_SIZE; }
static int GetParamSharedBase_kepler(void) { return KEPLER_PARAM_SMEM_BASE; }
static int GetMaxParamSharedSize_kepler(void) { return KEPLER_PARAM_SMEM_SIZE; }

/* APIs for querying constant bank mappings, returns -1 if not applicable */
static int GetElwregConstBank_kepler(void) { return KEPLER_ELWREGS_CONST_BANK; }
static int GetOCGConstBank_kepler(void) { return KEPLER_OCG_CONST_BANK; }
static int GetDriverConstantBank_kepler(void) { return KEPLER_DRIVER_CONST_BANK; }
static int GetGlobalRelocConstBank_kepler(void) { return -1; }
static int GetConstBackingStoreBank_kepler(void) { return KEPLER_CONST_BACKING_STORE_CBANK; }

static Bool GetIsUserConstBank_kepler(int hwBank)
{
    return ((hwBank >= KEPLER_USER_CONST_BANK_MIN && hwBank <= KEPLER_USER_CONST_BANK_MAX) ||
            hwBank == KEPLER_DRIVER_CONST_BANK);
}

static Bool GetIsGlobalConstBank_kepler(int hwBank)
{
    return GetIsUserConstBank_kepler(hwBank);
}

static Bool GetIsLocalConstBank_kepler(int hwBank)
{
    return !GetIsGlobalConstBank_kepler(hwBank);
}

/* APIs for querying fixed offsets in DCI, return -1 if not applicable */
static int GetElwregConstBase_kepler(void) { return KEPLER_ELWREGS_CBANK_BASE; }
static int GetConstBackingStoreBase_kepler(int hwBank) 
{ 
    if (hwBank == KEPLER_PARAM_CONST_BANK) {
        return KEPLER_PARAM_BANK_BACKING_STORE_BASE;
    } else if (hwBank == KEPLER_DRIVER_CONST_BANK) {
        return KEPLER_DRIVER_BANK_BACKING_STORE_BASE;
    } else {
        // DCI has backing store address of hwBank 6 but that bank is lwrrently reserved.
        stdASSERT((hwBank >= KEPLER_USER_CONST_BANK_MIN && hwBank <= KEPLER_USER_CONST_BANK_MAX), ("Unexpected bank"));
        // backing store address is 64bit
        return KEPLER_CONST_BACKING_STORE_BASE + (hwBank - KEPLER_USER_CONST_BANK_MIN) * 8;
    }
}

/* APIs related to texture, sampler, surface DCI */
static Bool GetUsesBindlessFormat_kepler(void) { return True; }
static Bool GetUsesUnifiedTextureSurfaceDescriptors_kepler(void) { return False; }
static int GetBindlessTextureBank_kepler(void) { return KEPLER_BINDLESS_TEX_BANK; }
static int GetBindlessSurfaceBank_kepler(void) { return KEPLER_BINDLESS_SURF_BANK; }
static Bool GetUsesLayeredSurfaceEmulation_kepler(void) { return False; }

static Bool GetIsQueryDescBank_kepler(int hwBank)
{
    return hwBank == KEPLER_BINDLESS_TEX_BANK;
}

static ptxMemSpace GetTextureQueryDescriptorStorage_kepler(Bool IsLocal)
{
    return PTX_SPACE_GLOBAL0;
}

static ptxMemSpace GetSurfaceQueryDescriptorStorage_kepler(Bool IsLocal)
{
    return (ptxMemSpace)(PTX_SPACE_CONST0 + GetBindlessSurfaceBank_kepler());
}

static int GetGlobalQueryDescTableBank_kepler(void) { return KEPLER_SW_DESC_TABLE_BANK; }
static int GetGlobalTextureQueryDescTableBase_kepler(void) { return KEPLER_TEX_SW_DESC_TABLE_OFFSET; }
static int GetGlobalSamplerQueryDescTableBase_kepler(void) { return KEPLER_SAMP_SW_DESC_TABLE_OFFSET; }
static int GetGlobalSurfaceQueryDescTableBase_kepler(void) { return -1; }
static int GetBindlessTextureImmediateOffset_kepler(void) { return KEPLER_BINDLESS_IMM_TEX_OFFSET; }
static int GetBindlessSurfaceHeaderSize_kepler(void ) { return KEPLER_BINDLESS_SURF_HEADER_SIZE; }
static int GetBindlessTextureHeaderIndexSize_kepler(void ) { return KEPLER_BINDLESS_TEX_HEADER_INDEX_SIZE; }

/* MISC APIs */
static Bool GetUsesIndirectionForGlobals_kepler(void) { return False; }

static int GetGlobalMemorySegment_kepler(void) { return KEPLER_GLOBAL_MEMORY_SEGMENT; }

static Bool GetConstPointersAllocatedByPtxas_kepler(void) { return True; }

static ConstPointerKind GetConstPointerKind_kepler(int NumConstPtrsUsed)
{
    if (NumConstPtrsUsed == 0) {
        return KIND_OFFSET;
    } else if (NumConstPtrsUsed <= 2) {
        return KIND_FAT_ADDRESS;
    } else {
        // Promote constant pointers to global memory if can't be allocated in const banks
        return KIND_GENERIC;
    }
}

static ptxMemSpace GetHWStorageForPtxCbank_kepler(int ptxBank, Bool UseDriverConstBank)
{
    if (UseDriverConstBank) {
        stdASSERT(ptxBank == 0, ("Bank 0 expected"));
        return (ptxMemSpace)(PTX_SPACE_CONST0 + GetDriverConstantBank_kepler());
    } else {
        if (ptxBank >= 3) {
            // Promote C[3] to C[10] to global memory
            return PTX_SPACE_GLOBAL0;
        } else {
            return (ptxMemSpace) (PTX_SPACE_CONST0 + (ptxBank + KEPLER_USER_CONST_BANK_MIN));
        }
    }
}

static int GetPTXCbankForHWCbank_kepler(int hwBank)
{
    if (hwBank >= KEPLER_USER_CONST_BANK_MIN && hwBank <= KEPLER_USER_CONST_BANK_MAX) {
        return hwBank - KEPLER_USER_CONST_BANK_MIN;
    } else if (hwBank == KEPLER_DRIVER_CONST_BANK) {
        return 0;
    } else {
        return -1;
    }
}

static Bool GetUses64bitGridid_kepler(void) { return True; }
static Bool GetSupportsCNP_kepler(int SMArch) { return SMArch == 35 ? True : False; }
static int GetCodeSectionAlignment_kepler(void) { return KEPLER_CODE_ALIGN; }

static void InitKeplerDCI(ptxDCIHandle keplerDCI)
{
    keplerDCI->GetMaxSharedMemory = GetMaxSharedMemory_kepler;
    keplerDCI->GetMaxLocalMemory = GetMaxLocalMemory_kepler;
    keplerDCI->GetConstantBankSize = GetConstantBankSize_kepler;
    keplerDCI->GetMaxTexturesPerEntry = GetMaxTexturesPerEntry_kepler;
    keplerDCI->GetMaxSamplersPerEntry = GetMaxSamplersPerEntry_kepler;
    keplerDCI->GetMaxSurfacesPerEntry = GetMaxSurfacesPerEntry_kepler;

    keplerDCI->GetParamConstBank = GetParamConstBank_kepler;
    keplerDCI->GetParamConstBase = GetParamConstBase_kepler;
    keplerDCI->GetMaxParamConstSize = GetMaxParamConstSize_kepler;
    keplerDCI->GetParamSharedBase = GetParamSharedBase_kepler;
    keplerDCI->GetMaxParamSharedSize = GetMaxParamSharedSize_kepler;

    keplerDCI->GetElwregConstBank = GetElwregConstBank_kepler;
    keplerDCI->GetOCGConstBank = GetOCGConstBank_kepler;
    keplerDCI->GetDriverConstBank = GetDriverConstantBank_kepler;
    keplerDCI->GetGlobalRelocConstBank = GetGlobalRelocConstBank_kepler;
    keplerDCI->GetConstBackingStoreBank = GetConstBackingStoreBank_kepler;

    keplerDCI->GetIsUserConstBank = GetIsUserConstBank_kepler;
    keplerDCI->GetIsGlobalConstBank = GetIsGlobalConstBank_kepler;
    keplerDCI->GetIsLocalConstBank = GetIsLocalConstBank_kepler;

    keplerDCI->GetElwregConstBase = GetElwregConstBase_kepler;
    keplerDCI->GetConstBackingStoreBase = GetConstBackingStoreBase_kepler;

    keplerDCI->GetUsesBindlessFormat = GetUsesBindlessFormat_kepler;
    keplerDCI->GetUsesUnifiedTextureSurfaceDescriptors = GetUsesUnifiedTextureSurfaceDescriptors_kepler;
    keplerDCI->GetBindlessTextureBank = GetBindlessTextureBank_kepler;
    keplerDCI->GetBindlessSurfaceBank = GetBindlessSurfaceBank_kepler;
    keplerDCI->GetIsQueryDescConstBank = GetIsQueryDescBank_kepler;
    keplerDCI->GetUsesLayeredSurfaceEmulation = GetUsesLayeredSurfaceEmulation_kepler;
    keplerDCI->GetTextureQueryDescriptorSize = GetTextureQueryDescriptorSize_common;
    keplerDCI->GetSamplerQueryDescriptorSize = GetSamplerQueryDescriptorSize_common;
    keplerDCI->GetSurfaceQueryDescriptorSize = GetSurfaceQueryDescriptorSize_common;
    keplerDCI->GetTextureQueryDescriptorNumAttributes = GetTextureQueryDescriptorNumAttributes_common;
    keplerDCI->GetSamplerQueryDescriptorNumAttributes = GetSamplerQueryDescriptorNumAttributes_common;
    keplerDCI->GetSurfaceQueryDescriptorNumAttributes = GetSurfaceQueryDescriptorNumAttributes_common;
    keplerDCI->GetTextureQueryDescriptorStorage = GetTextureQueryDescriptorStorage_kepler;
    keplerDCI->GetSurfaceQueryDescriptorStorage = GetSurfaceQueryDescriptorStorage_kepler;
    keplerDCI->GetGlobalQueryDescTableBank = GetGlobalQueryDescTableBank_kepler;
    keplerDCI->GetGlobalTextureQueryDescTableBase = GetGlobalTextureQueryDescTableBase_kepler;
    keplerDCI->GetGlobalSamplerQueryDescTableBase = GetGlobalSamplerQueryDescTableBase_kepler;
    keplerDCI->GetGlobalSurfaceQueryDescTableBase = GetGlobalSurfaceQueryDescTableBase_kepler;
    keplerDCI->GetBindlessTextureImmediateOffset = GetBindlessTextureImmediateOffset_kepler;
    keplerDCI->GetBindlessTextureHeaderIndexSize = GetBindlessTextureHeaderIndexSize_kepler;
    keplerDCI->GetBindlessSurfaceHeaderSize = GetBindlessSurfaceHeaderSize_kepler;

    keplerDCI->GetUsesIndirectionForGlobals = GetUsesIndirectionForGlobals_kepler;
    keplerDCI->GetGlobalMemorySegment = GetGlobalMemorySegment_kepler;
    keplerDCI->GetConstPointersAllocatedByPtxas = GetConstPointersAllocatedByPtxas_kepler;
    keplerDCI->GetConstPointerKind = GetConstPointerKind_kepler;
    keplerDCI->GetHWStorageForPtxCbank = GetHWStorageForPtxCbank_kepler;
    keplerDCI->GetPTXCbankForHWCbank = GetPTXCbankForHWCbank_kepler;
    keplerDCI->GetUses64bitGridid = GetUses64bitGridid_kepler;
    keplerDCI->GetSupportsCNP = GetSupportsCNP_kepler;
    keplerDCI->GetCodeSectionAlignment = GetCodeSectionAlignment_kepler;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////MAXWELL DCI//////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

#if LWCFG(GLOBAL_ARCH_MAXWELL)

#define MAXWELL_MAX_SHARED_MEMORY                (64 * 1024) // 64K
#define MAXWELL_MAX_LOCAL_MEMORY                 (16 * 1024 * 1024) // 16M
#define MAXWELL_CONST_BANK_SIZE                  (64 * 1024) // 64K
#define MAXWELL_NUM_TEXTURES_PER_ENTRY           256
#define MAXWELL_NUM_SAMPLERS_PER_ENTRY           32
#define MAXWELL_NUM_SURFACES_PER_ENTRY           16

#define MAXWELL_USER_CONST_BANK_MIN              3
#define MAXWELL_USER_CONST_BANK_MAX              5
#define MAXWELL_PARAM_CONST_BANK                 0
#define MAXWELL_OCG_CONST_BANK                   2
#define MAXWELL_DRIVER_CONST_BANK                1
#define MAXWELL_ELWREGS_CONST_BANK               0
#define MAXWELL_BINDLESS_TEX_BANK                0
#define MAXWELL_BINDLESS_SURF_BANK               0
#define MAXWELL_NUM_PTX_BANKS                    3
#define MAXWELL_BINDLESS_SURF_HEADER_SIZE        4
#define MAXWELL_BINDLESS_TEX_HEADER_INDEX_SIZE   4
#define MAXWELL_BINDLESS_IMM_TEX_OFFSET          (8192 * 4)
#define MAXWELL_GLOBAL_MEMORY_SEGMENT            0

#define MAXWELL_PARAM_CBANK_BASE                 320
#define MAXWELL_PARAM_CBANK_SIZE                 (4096 + 256)
#define MAXWELL_PARAM_SMEM_BASE                  0
#define MAXWELL_PARAM_SMEM_SIZE                  0
#define MAXWELL_ELWREGS_CBANK_BASE               48
#define MAXWELL_SW_DESC_TABLE_BANK               0
#define MAXWELL_TEX_SW_DESC_TABLE_OFFSET         176
#define MAXWELL_SAMP_SW_DESC_TABLE_OFFSET        184
#define MAXWELL_SURF_SW_DESC_TABLE_OFFSET        192
#define MAXWELL_CONST_BACKING_STORE_CBANK        0
#define MAXWELL_CONST_BACKING_STORE_BASE         208
#define MAXWELL_PARAM_BANK_BACKING_STORE_BASE    200
#define MAXWELL_DRIVER_BANK_BACKING_STORE_BASE   240

#define MAXWELL_CODE_ALIGN                       32

/* APIs for querying Memory Limits */
static int GetMaxSharedMemory_maxwell(void) { return MAXWELL_MAX_SHARED_MEMORY; }
static int GetMaxLocalMemory_maxwell(void) { return MAXWELL_MAX_LOCAL_MEMORY; }
static int GetConstantBankSize_maxwell(void) { return MAXWELL_CONST_BANK_SIZE; }
static int GetMaxTexturesPerEntry_maxwell(void) { return MAXWELL_NUM_TEXTURES_PER_ENTRY; }
static int GetMaxSamplersPerEntry_maxwell(void) { return MAXWELL_NUM_SAMPLERS_PER_ENTRY; }
static int GetMaxSurfacesPerEntry_maxwell(void) { return MAXWELL_NUM_SURFACES_PER_ENTRY; }

/* APIs for querying kernel parameters related information */
static int GetParamConstBank_maxwell(void) { return MAXWELL_PARAM_CONST_BANK; }
static int GetParamConstBase_maxwell(void) { return MAXWELL_PARAM_CBANK_BASE; }
static int GetMaxParamConstSize_maxwell(void) { return MAXWELL_PARAM_CBANK_SIZE; }
static int GetParamSharedBase_maxwell(void) { return MAXWELL_PARAM_SMEM_BASE; }
static int GetMaxParamSharedSize_maxwell(void) { return MAXWELL_PARAM_SMEM_SIZE; }

/* APIs for querying constant bank mappings, returns -1 if not applicable */
static int GetElwregConstBank_maxwell(void) { return MAXWELL_ELWREGS_CONST_BANK; }
static int GetOCGConstBank_maxwell(void) { return MAXWELL_OCG_CONST_BANK; }
static int GetDriverConstantBank_maxwell(void) { return MAXWELL_DRIVER_CONST_BANK; }
static int GetGlobalRelocConstBank_maxwell(void) { return -1; }
static int GetConstBackingStoreBank_maxwell(void) { return MAXWELL_CONST_BACKING_STORE_CBANK; }

static Bool GetIsUserConstBank_maxwell(int hwBank)
{
    return ((hwBank >= MAXWELL_USER_CONST_BANK_MIN && hwBank <= MAXWELL_USER_CONST_BANK_MAX) ||
            hwBank == MAXWELL_DRIVER_CONST_BANK);
}

static Bool GetIsGlobalConstBank_maxwell(int hwBank)
{
    return GetIsUserConstBank_maxwell(hwBank);
}

static Bool GetIsLocalConstBank_maxwell(int hwBank)
{
    return !GetIsGlobalConstBank_maxwell(hwBank);
}

/* APIs for querying fixed offsets in DCI, return -1 if not applicable */
static int GetElwregConstBase_maxwell(void) { return MAXWELL_ELWREGS_CBANK_BASE; }
static int GetConstBackingStoreBase_maxwell(int hwBank) 
{ 
    if (hwBank == MAXWELL_PARAM_CONST_BANK) {
        return MAXWELL_PARAM_BANK_BACKING_STORE_BASE;
    } else if (hwBank == MAXWELL_DRIVER_CONST_BANK) {
        return MAXWELL_DRIVER_BANK_BACKING_STORE_BASE;
    } else {
        // DCI has backing store address of hwBank 6 but that bank is lwrrently reserved.
        stdASSERT((hwBank >= MAXWELL_USER_CONST_BANK_MIN && hwBank <= MAXWELL_USER_CONST_BANK_MAX), ("Unexpected bank"));
        // backing store address is 64bit
        return MAXWELL_CONST_BACKING_STORE_BASE + (hwBank - MAXWELL_USER_CONST_BANK_MIN) * 8;
    }
}

/* APIs related to texture, sampler, surface DCI */
static Bool GetUsesBindlessFormat_maxwell(void) { return True; }
static Bool GetUsesUnifiedTextureSurfaceDescriptors_maxwell(void) { return True; }
static int GetBindlessTextureBank_maxwell(void) { return MAXWELL_BINDLESS_TEX_BANK; }
static int GetBindlessSurfaceBank_maxwell(void) { return MAXWELL_BINDLESS_SURF_BANK; }
static Bool GetUsesLayeredSurfaceEmulation_maxwell(void) { return False; }

static Bool GetIsQueryDescBank_maxwell(int hwBank)
{
    return False;
}

static ptxMemSpace GetTextureQueryDescriptorStorage_maxwell(Bool IsLocal)
{
    return PTX_SPACE_GLOBAL0;
}

static ptxMemSpace GetSurfaceQueryDescriptorStorage_maxwell(Bool IsLocal)
{
    return PTX_SPACE_GLOBAL0;
}

static int GetGlobalQueryDescTableBank_maxwell(void) { return MAXWELL_SW_DESC_TABLE_BANK; }
static int GetGlobalTextureQueryDescTableBase_maxwell(void) { return MAXWELL_TEX_SW_DESC_TABLE_OFFSET; }
static int GetGlobalSamplerQueryDescTableBase_maxwell(void) { return MAXWELL_SAMP_SW_DESC_TABLE_OFFSET; }
static int GetGlobalSurfaceQueryDescTableBase_maxwell(void) { return MAXWELL_SURF_SW_DESC_TABLE_OFFSET; }
static int GetBindlessTextureImmediateOffset_maxwell(void) { return MAXWELL_BINDLESS_IMM_TEX_OFFSET; }
static int GetBindlessSurfaceHeaderSize_maxwell(void ) { return MAXWELL_BINDLESS_SURF_HEADER_SIZE; }
static int GetBindlessTextureHeaderIndexSize_maxwell(void ) { return MAXWELL_BINDLESS_TEX_HEADER_INDEX_SIZE; }

/* MISC APIs */
static Bool GetUsesIndirectionForGlobals_maxwell(void) { return False; }

static int GetGlobalMemorySegment_maxwell(void) { return MAXWELL_GLOBAL_MEMORY_SEGMENT; }

static Bool GetConstPointersAllocatedByPtxas_maxwell(void) { return True; }

static ConstPointerKind GetConstPointerKind_maxwell(int NumConstPtrsUsed)
{
    if (NumConstPtrsUsed == 0) {
        return KIND_OFFSET;
    } else if (NumConstPtrsUsed <= 2) {
        return KIND_FAT_ADDRESS;
    } else {
        // Promote constant pointers to global memory if can't be allocated in const banks
        return KIND_GENERIC;
    }
}

static ptxMemSpace GetHWStorageForPtxCbank_maxwell(int ptxBank, Bool UseDriverConstBank)
{
    if (UseDriverConstBank) {
        stdASSERT(ptxBank == 0, ("Bank 0 expected"));
        return (ptxMemSpace)(PTX_SPACE_CONST0 + GetDriverConstantBank_maxwell());
    } else {
        if (ptxBank >= 3) {
            // Promote C[3] to C[10] to global memory
            return PTX_SPACE_GLOBAL0;
        } else {
            return (ptxMemSpace) (PTX_SPACE_CONST0 + (ptxBank + MAXWELL_USER_CONST_BANK_MIN));
        }
    }
}

static int GetPTXCbankForHWCbank_maxwell(int hwBank)
{
    if (hwBank >= MAXWELL_USER_CONST_BANK_MIN && hwBank <= MAXWELL_USER_CONST_BANK_MAX) {
        return hwBank - MAXWELL_USER_CONST_BANK_MIN;
    } else if (hwBank == MAXWELL_DRIVER_CONST_BANK) {
        return 0;
    } else {
        return -1;
    }
}

static Bool GetUses64bitGridid_maxwell(void) { return True; }
static Bool GetSupportsCNP_maxwell(int SMArch) { return True; }
static int GetCodeSectionAlignment_maxwell(void) { return MAXWELL_CODE_ALIGN; }

static void InitMaxwellDCI(ptxDCIHandle maxwellDCI)
{
    maxwellDCI->GetMaxSharedMemory = GetMaxSharedMemory_maxwell;
    maxwellDCI->GetMaxLocalMemory = GetMaxLocalMemory_maxwell;
    maxwellDCI->GetConstantBankSize = GetConstantBankSize_maxwell;
    maxwellDCI->GetMaxTexturesPerEntry = GetMaxTexturesPerEntry_maxwell;
    maxwellDCI->GetMaxSamplersPerEntry = GetMaxSamplersPerEntry_maxwell;
    maxwellDCI->GetMaxSurfacesPerEntry = GetMaxSurfacesPerEntry_maxwell;

    maxwellDCI->GetParamConstBank = GetParamConstBank_maxwell;
    maxwellDCI->GetParamConstBase = GetParamConstBase_maxwell;
    maxwellDCI->GetMaxParamConstSize = GetMaxParamConstSize_maxwell;
    maxwellDCI->GetParamSharedBase = GetParamSharedBase_maxwell;
    maxwellDCI->GetMaxParamSharedSize = GetMaxParamSharedSize_maxwell;

    maxwellDCI->GetElwregConstBank = GetElwregConstBank_maxwell;
    maxwellDCI->GetOCGConstBank = GetOCGConstBank_maxwell;
    maxwellDCI->GetDriverConstBank = GetDriverConstantBank_maxwell;
    maxwellDCI->GetGlobalRelocConstBank = GetGlobalRelocConstBank_maxwell;
    maxwellDCI->GetConstBackingStoreBank = GetConstBackingStoreBank_maxwell;

    maxwellDCI->GetIsUserConstBank = GetIsUserConstBank_maxwell;
    maxwellDCI->GetIsGlobalConstBank = GetIsGlobalConstBank_maxwell;
    maxwellDCI->GetIsLocalConstBank = GetIsLocalConstBank_maxwell;

    maxwellDCI->GetElwregConstBase = GetElwregConstBase_maxwell;
    maxwellDCI->GetConstBackingStoreBase = GetConstBackingStoreBase_maxwell;

    maxwellDCI->GetUsesBindlessFormat = GetUsesBindlessFormat_maxwell;
    maxwellDCI->GetUsesUnifiedTextureSurfaceDescriptors = GetUsesUnifiedTextureSurfaceDescriptors_maxwell;
    maxwellDCI->GetBindlessTextureBank = GetBindlessTextureBank_maxwell;
    maxwellDCI->GetBindlessSurfaceBank = GetBindlessSurfaceBank_maxwell;
    maxwellDCI->GetIsQueryDescConstBank = GetIsQueryDescBank_maxwell;
    maxwellDCI->GetUsesLayeredSurfaceEmulation = GetUsesLayeredSurfaceEmulation_maxwell;
    maxwellDCI->GetTextureQueryDescriptorSize = GetTextureQueryDescriptorSize_common;
    maxwellDCI->GetSamplerQueryDescriptorSize = GetSamplerQueryDescriptorSize_common;
    maxwellDCI->GetSurfaceQueryDescriptorSize = GetSurfaceQueryDescriptorSize_common;
    maxwellDCI->GetTextureQueryDescriptorNumAttributes = GetTextureQueryDescriptorNumAttributes_common;
    maxwellDCI->GetSamplerQueryDescriptorNumAttributes = GetSamplerQueryDescriptorNumAttributes_common;
    maxwellDCI->GetSurfaceQueryDescriptorNumAttributes = GetSurfaceQueryDescriptorNumAttributes_common;
    maxwellDCI->GetTextureQueryDescriptorStorage = GetTextureQueryDescriptorStorage_maxwell;
    maxwellDCI->GetSurfaceQueryDescriptorStorage = GetSurfaceQueryDescriptorStorage_maxwell;
    maxwellDCI->GetGlobalQueryDescTableBank = GetGlobalQueryDescTableBank_maxwell;
    maxwellDCI->GetGlobalTextureQueryDescTableBase = GetGlobalTextureQueryDescTableBase_maxwell;
    maxwellDCI->GetGlobalSamplerQueryDescTableBase = GetGlobalSamplerQueryDescTableBase_maxwell;
    maxwellDCI->GetGlobalSurfaceQueryDescTableBase = GetGlobalSurfaceQueryDescTableBase_maxwell;
    maxwellDCI->GetBindlessTextureImmediateOffset = GetBindlessTextureImmediateOffset_maxwell;
    maxwellDCI->GetBindlessTextureHeaderIndexSize = GetBindlessTextureHeaderIndexSize_maxwell;
    maxwellDCI->GetBindlessSurfaceHeaderSize = GetBindlessSurfaceHeaderSize_maxwell;

    maxwellDCI->GetUsesIndirectionForGlobals = GetUsesIndirectionForGlobals_maxwell;
    maxwellDCI->GetGlobalMemorySegment = GetGlobalMemorySegment_maxwell;
    maxwellDCI->GetConstPointersAllocatedByPtxas = GetConstPointersAllocatedByPtxas_maxwell;
    maxwellDCI->GetConstPointerKind = GetConstPointerKind_maxwell;
    maxwellDCI->GetHWStorageForPtxCbank = GetHWStorageForPtxCbank_maxwell;
    maxwellDCI->GetPTXCbankForHWCbank = GetPTXCbankForHWCbank_maxwell;
    maxwellDCI->GetUses64bitGridid = GetUses64bitGridid_maxwell;
    maxwellDCI->GetSupportsCNP = GetSupportsCNP_maxwell;
    maxwellDCI->GetCodeSectionAlignment = GetCodeSectionAlignment_maxwell;
}
#endif // LWCFG(GLOBAL_ARCH_MAXWELL)

