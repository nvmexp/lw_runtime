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
 * @file   optix_ext_compute_instance_aabbs_function_table_definition.h
 * @author LWPU Corporation
 * @brief  OptiX public API header
 *
 */

#ifndef __optix_optix_ext_compute_instance_aabbs_function_table_definition_h__
#define __optix_optix_ext_compute_instance_aabbs_function_table_definition_h__

#include "optix_ext_compute_instance_aabbs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* If the stubs in optix_ext_compute_instance_aabbs_stubs.h are used, then the function table needs to be
   defined in exactly one translation unit. This can be achieved by including this header file in
   that translation unit. */
OptixExtComputeInstanceAabbsFunctionTable g_optixExtComputeInstanceAabbsFunctionTable;

#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_ext_compute_instance_aabbs_function_table_definition_h__ */
