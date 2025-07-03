/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2007-2021, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ptxOptEnums.h
 *
 */

#ifndef ptxOptEnums_INCLUDED
#define ptxOptEnums_INCLUDED

#include <stdTypes.h>
#include "ptxDCI.h"

#ifdef __cplusplus
extern "C" {
#endif

/*------------------------------ Definitions ---------------------------------*/

#define COARSE_GRAINED_DESCRIPTOR 1
#define NUM_CONST_BANKS (REG_CONST_MAX - REG_CONST_MIN + 1)

#define NUM_ELW_REGISTERS 32
#define NUM_BAR_REGISTERS 16
#define INDIRECT_FUNCTION_TABLE_ENTRY_SIZE 8

typedef enum {
    PTX_NO_ABI_COMPILE,
    PTX_ABI_COMPILE,
} PtxCompileKind;

typedef enum {
    PTX_ARCH_MIN        = 0,

    PTX_ARCH_KEPLER_MIN = 7,
    PTX_ARCH_SM_30      = 7,
    PTX_ARCH_SM_32      = 8,
    PTX_ARCH_SM_35      = 9,
    PTX_ARCH_SM_37      = 10,
    PTX_ARCH_KEPLER_MAX = 10,

    PTX_ARCH_MAXWELL_MIN = 11,
    PTX_ARCH_SM_50       = 11,
    PTX_ARCH_SM_52       = 12,
    PTX_ARCH_SM_53       = 13,
    PTX_ARCH_MAXWELL_MAX = 13,

    PTX_ARCH_PASCAL_MIN  = 14,
    PTX_ARCH_SM_60       = 14,    
    PTX_ARCH_SM_61       = 15,    
    PTX_ARCH_SM_62       = 16,    
    PTX_ARCH_PASCAL_MAX  = 16,

    PTX_ARCH_VOLTA_MIN   = 17,
    PTX_ARCH_SM_70       = 17,
    PTX_ARCH_SM_72       = 18,
    PTX_ARCH_VOLTA_MAX   = 18,

    PTX_ARCH_TURING_MIN  = 19,
    PTX_ARCH_SM_73       = 19,
    PTX_ARCH_SM_75       = 20,
    PTX_ARCH_TURING_MAX  = 20,

    PTX_ARCH_AMPERE_MIN  = 21,
    PTX_ARCH_SM_80       = 21,
    PTX_ARCH_SM_86       = 23,
    PTX_ARCH_SM_87       = 24,
    PTX_ARCH_SM_88       = 25,
    PTX_ARCH_AMPERE_MAX  = 25,

    PTX_ARCH_ADA_MIN     = 26,
    PTX_ARCH_SM_89       = 26,
    PTX_ARCH_ADA_MAX     = 26,

    PTX_ARCH_HOPPER_MIN  = 27,
    PTX_ARCH_SM_90       = 27,
    PTX_ARCH_HOPPER_MAX  = 27,

    PTX_ARCH_MAX = 28
} PtxArch;

#define IS_ARCH_ATLEAST(arch, minArch) ((arch) >= (minArch))
#define IS_ARCH_FAMILY_ATLEAST(arch, family) ((arch) >= PTX_ARCH_##family##_MIN)

#define IS_KEPLER(arch) ((arch) >= PTX_ARCH_KEPLER_MIN && (arch) <= PTX_ARCH_KEPLER_MAX)
#define IS_MAXWELL(arch) ((arch) >= PTX_ARCH_MAXWELL_MIN && (arch) <= PTX_ARCH_MAXWELL_MAX)
#define IS_PASCAL(arch) ((arch) >= PTX_ARCH_PASCAL_MIN && (arch) <= PTX_ARCH_PASCAL_MAX)
#define IS_VOLTA(arch)  ((arch) >= PTX_ARCH_VOLTA_MIN  && (arch) <= PTX_ARCH_VOLTA_MAX)
#define IS_TURING(arch) ((arch) >= PTX_ARCH_TURING_MIN  && (arch) <= PTX_ARCH_TURING_MAX)
#define IS_AMPERE(arch) ((arch) >= PTX_ARCH_AMPERE_MIN  && (arch) <= PTX_ARCH_AMPERE_MAX)
#define IS_ADA(arch)    ((arch) >= PTX_ARCH_ADA_MIN  && (arch) <= PTX_ARCH_ADA_MAX)
#define IS_HOPPER(arch) ((arch) >= PTX_ARCH_HOPPER_MIN  && (arch) <= PTX_ARCH_HOPPER_MAX)

// PtxArch -> SM number mapping
int ptxDCIGetSMFromPtxArch(PtxArch arch);


#if     defined(__cplusplus)
}
#endif 

#endif // ptxOptEnums_INCLUDED
