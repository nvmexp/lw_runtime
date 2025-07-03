/* 
* Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*/

/******************************************************************************
*
*   Module: API.c
*  
*   Description:
*       Implementation of compiler API for the driver.
*
******************************************************************************/

#include "API.h"
#include "stdLocal.h"

#define ALEN(x) sizeof(x)/sizeof(x[0])

/*
 * getCBankParamOffset() - Return the offset of the cbank param
 *
 */

int getCBankParamOffset(int paramIdx)
{
    static const unsigned int cBankParamOffsets[] = {0, 2, 16, 0}; // Adjust this array if "LwbinCBankParams" changes

    stdASSERT(paramIdx < CBANK_PARAM_MAX, ("invalid param index"));
    return cBankParamOffsets[paramIdx];
} 

/*
 * getCBankParamSize() - Return the size of the cbank param 
 * 
 */

int getCBankParamSize(int paramIdx) 
{
    static const unsigned int cBankParamSizes[] = {2, 2}; // Adjust this array if "LwbinCBankParams" changes

    stdASSERT(paramIdx < CBANK_PARAM_NON_USER_MAX, ("invalid param index"));
    return cBankParamSizes[paramIdx];
}

/*
 *  getUnifiedTexrefAttributeOffset() - Return the attribute offset for unified texturing mode
 *
 */

int getUnifiedTexrefAttributeOffset(int attribute)
{
    static const unsigned int lwbinUniTexrefAttributeOffset[] = { 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48 };

    stdASSERT(ALEN(lwbinUniTexrefAttributeOffset) == (LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MIN && 
        attribute <= LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX, ("Invalid Texref attribute"));
    return lwbinUniTexrefAttributeOffset[attribute];
}

/*
 *  getUnifiedTexrefAttributeSize() - Return the attribute size for unified texturing mode
 *
 */

int getUnifiedTexrefAttributeSize(int attribute)
{
    static const unsigned int lwbinUniTexrefAttributeSize[] = { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };

    stdASSERT(ALEN(lwbinUniTexrefAttributeSize) == (LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MIN && 
        attribute <= LWBIN_UNIFIED_TEXREF_ATTRIBUTE_MAX, ("Invalid Texref attribute"));
    return lwbinUniTexrefAttributeSize[attribute];
}

/*
 *  getUnifiedTexrefAttributeOffset() - Return the attribute offset for Indepedent texturing mode
 *
 */

int getIndependentTexrefAttributeOffset(int attribute)
{
    static const unsigned int lwbinIndTexrefAttributeOffset[] = { 0, 4, 8, 12, 16, 20, 24, 28, 32 };

    stdASSERT(ALEN(lwbinIndTexrefAttributeOffset) == (LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MIN && 
        attribute <= LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX, ("Invalid Texref attribute"));
    return lwbinIndTexrefAttributeOffset[attribute];
}

/*
 *  getUnifiedTexrefAttributeSize() - Return the attribute size for Indepedent texturing mode
 *
 */

int getIndependentTexrefAttributeSize(int attribute)
{
    static const unsigned int lwbinIndTexrefAttributeSize[] = { 4, 4, 4, 4, 4, 4, 4, 4, 4 };

    stdASSERT(ALEN(lwbinIndTexrefAttributeSize) == (LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MIN && 
        attribute <= LWBIN_INDEPENDENT_TEXREF_ATTRIBUTE_MAX, ("Invalid Texref attribute"));
    return lwbinIndTexrefAttributeSize[attribute];
}

/*
 *  getSamplerrefAttributeOffset() - Return the attribute offset
 *
 */

int getSamplerrefAttributeOffset(int attribute)
{
    static const unsigned int lwbinSamplerrefAttributeOffset[] = { 0, 4, 8, 12, 16 } ;

    stdASSERT(ALEN(lwbinSamplerrefAttributeOffset) == (LWBIN_SAMPLERREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_SAMPLERREF_ATTRIBUTE_MIN && attribute <= LWBIN_SAMPLERREF_ATTRIBUTE_MAX,
                ("Invalid Samplerref attribute"));
    return lwbinSamplerrefAttributeOffset[attribute];
}

/*
 *  getSamplerrefAttributeSize() - Return the attribute size
 *
 */

int getSamplerrefAttributeSize(int attribute)
{
    static const unsigned int lwbinSamplerrefAttributeSize[] = { 4, 4, 4, 4, 4 };

    stdASSERT(ALEN(lwbinSamplerrefAttributeSize) == (LWBIN_SAMPLERREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_SAMPLERREF_ATTRIBUTE_MIN && attribute <= LWBIN_SAMPLERREF_ATTRIBUTE_MAX,
                ("Invalid Samplerref attribute"));
    return lwbinSamplerrefAttributeSize[attribute];
}

/*
 *  getSurfrefAttributeOffset() - Return the attribute offset
 *
 */

int getSurfrefAttributeOffset(int attribute)
{
    static const unsigned int lwbinSurfrefAttributeOffset[] = { 0, 4, 8, 12, 16, 20, 24 };

    stdASSERT(ALEN(lwbinSurfrefAttributeOffset) == (LWBIN_SURFREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_SURFREF_ATTRIBUTE_MIN && attribute <= LWBIN_SURFREF_ATTRIBUTE_MAX,
                ("Invalid Surfref attribute"));
    return lwbinSurfrefAttributeOffset[attribute];
}

/*
 *  getSurfrefAttributeSize() - Return the attribute size
 *
 */

int getSurfrefAttributeSize(int attribute)
{
    static const unsigned int lwbinSurfrefAttributeSize[] = { 4, 4, 4, 4, 4, 4, 4 };

    stdASSERT(ALEN(lwbinSurfrefAttributeSize) == (LWBIN_SURFREF_ATTRIBUTE_MAX + 1), 
        ("Number of attributes don't match with number of offsets"));
    stdASSERT(attribute >= LWBIN_SURFREF_ATTRIBUTE_MIN && attribute <= LWBIN_SURFREF_ATTRIBUTE_MAX,
                ("Invalid Surfref attribute"));
    return lwbinSurfrefAttributeSize[attribute];
}
