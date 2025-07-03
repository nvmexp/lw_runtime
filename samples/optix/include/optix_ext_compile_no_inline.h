/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/**
 * @file   optix_ext_compile_no_inline.h
 * @author LWPU Corporation
 * @brief  OptiX public API header
 */

#ifndef __optix_optix_ext_compile_no_inline_h__
#define __optix_optix_ext_compile_no_inline_h__

#include "optix_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// This will enable or disable the compilation of non-inlined functions as seen by OptiX in any known modules.
// The property cannot be changed after the compilation of the first module.
OptixResult optixExtCompileNoInlineSetEnabled( OptixDeviceContext contextAPI, bool enabled );
// Query the current value, i.e., whether or not the compilation of non-inlined functions is enabled.
bool optixExtCompileNoInlineIsEnabled( OptixDeviceContext contextAPI );


#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
// When changing the ABI version make sure you know exactly what you are doing. See
// apps/optix/exp/functionTable/functionTable.cpp for instructions. See
// https://confluence.lwpu.com/display/RAV/ABI+Versions+in+the+Wild for released ABI versions.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
#define OPTIX_EXT_COMPILE_NO_INLINE_ABI_VERSION 3001

typedef struct OptixExtCompileNoInlineFunctionTable
{
    OptixResult ( *optixExtCompileNoInlineSetEnabled )( OptixDeviceContext contextAPI, bool enabled );
    bool ( *optixExtCompileNoInlineIsEnabled )( OptixDeviceContext contextAPI );

} OptixExtCompileNoInlineFunctionTable;


#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_ext_compile_no_inline_h__ */
