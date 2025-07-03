/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

/******************************************************************************
*
*   Module: API.h
*
*   Description:
*       Implementation of compiler API for the driver.
*
******************************************************************************/

#ifndef API_INCLUDED
#define API_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

typedef enum LwbinCBankParams_enum {
    CBANK_PARAM_CTAID_LW50 = 0,
    CBANK_PARAM_NCTAID_LW50 = 1,
    // Add any *non-user* params here
    CBANK_PARAM_NON_USER_MAX,

    CBANK_PARAM_USER_LW50 = CBANK_PARAM_NON_USER_MAX,
    CBANK_PARAM_USER_FERMI,

    CBANK_PARAM_MAX
} LwbinCBankParams;


/*
 * New attribute fields must be added to end of enums
 */

typedef enum LwbinUnifiedTexrefAttributeKind_enum {
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MIN = 0,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_WIDTH  = LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MIN,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_HEIGHT,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_DEPTH,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_ADDR0,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_ADDR1,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_ADDR2,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_FILTER,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_NORM,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_CHTYPE,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_CHORDER,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_ARRAY_SIZE,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_NUM_MIPMAP_LEVELS,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_NUM_SAMPLES,
    LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX = LWBIN_UNIFIED_TEXREF_ATTRIBUTE_NUM_SAMPLES
} LwbinUnifiedTexRefArttributeKind;

typedef enum LwbinIndendentTexrefAttributeKind_enum {
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MIN = 0,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_WIDTH  = LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MIN,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_HEIGHT,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_DEPTH,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_NORM,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_CHTYPE,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_CHORDER,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_ARRAY_SIZE,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_NUM_MIPMAP_LEVELS,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_NUM_SAMPLES,
    LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX = LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_NUM_SAMPLES
} LwbinIndependentTexRefArttributeKind;

typedef enum LwbinSamplerrefAttributeKind_enum {
    LWBIN_SAMPLERREF_ATTRIBUTE_MIN = 0,
    LWBIN_SAMPLERREF_ATTRIBUTE_ADDR0 = LWBIN_SAMPLERREF_ATTRIBUTE_MIN,
    LWBIN_SAMPLERREF_ATTRIBUTE_ADDR1,
    LWBIN_SAMPLERREF_ATTRIBUTE_ADDR2,
    LWBIN_SAMPLERREF_ATTRIBUTE_FILTER,
    LWBIN_SAMPLERREF_ATTRIBUTE_UNNORM,
    LWBIN_SAMPLERREF_ATTRIBUTE_MAX = LWBIN_SAMPLERREF_ATTRIBUTE_UNNORM
} LwbinSamplerrefAttributeKind;

typedef enum LwbinSurfrefAttributeKind_enum {
    LWBIN_SURFREF_ATTRIBUTE_MIN = 0, 
    LWBIN_SURFREF_ATTRIBUTE_WIDTH = LWBIN_SURFREF_ATTRIBUTE_MIN,
    LWBIN_SURFREF_ATTRIBUTE_HEIGHT,
    LWBIN_SURFREF_ATTRIBUTE_DEPTH,
    LWBIN_SURFREF_ATTRIBUTE_CHTYPE,
    LWBIN_SURFREF_ATTRIBUTE_CHORDER,
    LWBIN_SURFREF_ATTRIBUTE_LAYOUT_IN_MEMORY,
    LWBIN_SURFREF_ATTRIBUTE_PITCH,
    LWBIN_SURFREF_ATTRIBUTE_ARRAY_SIZE,
    LWBIN_SURFREF_ATTRIBUTE_MAX = LWBIN_SURFREF_ATTRIBUTE_ARRAY_SIZE
} LwbinSurfrefAttributeKind;

int getCBankParamOffset(int paramIdx);
int getCBankParamSize(int paramIdx);
int getUnifiedTexrefAttributeOffset(int attribute);
int getUnifiedTexrefAttributeSize(int attribute);
int getIndependentTexrefAttributeOffset(int attribute);
int getIndependentTexrefAttributeSize(int attribute);
int getSamplerrefAttributeOffset(int attribute);
int getSamplerrefAttributeSize(int attribute);
int getSurfrefAttributeOffset(int attribute);
int getSurfrefAttributeSize(int attribute);

#define PTX_KERNEL_PARAM_LIMIT_ISA14  (256)
#define PTX_KERNEL_PARAM_LIMIT_ISA15  (256+4096)

#ifdef __cplusplus
}
#endif

#endif
