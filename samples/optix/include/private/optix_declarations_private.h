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


/******************************************************************************\
 *
 * Contains declarations used internally.  This file is never to be released.
 *
\******************************************************************************/

#ifndef __optix_optix_declarations_private_h__
#define __optix_optix_declarations_private_h__

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RT_BUFFER_PARTITIONED_INTERNAL     = 1u << 16,
    RT_BUFFER_PINNED_INTERNAL          = 1u << 17,
    RT_BUFFER_WRITECOMBINED_INTERNAL   = 1u << 18,
    RT_BUFFER_DEVICE_ONLY_INTERNAL     = 1u << 19,
    RT_BUFFER_FORCE_ZERO_COPY          = 1u << 20,
    RT_BUFFER_LAYERED_RESERVED         = 1u << 21, /* reserved here, declared in optix_declarations.h */
    RT_BUFFER_LWBEMAP_RESERVED         = 1u << 22, /* reserved here, declared in optix_declarations.h */
    RT_BUFFER_INTERNAL_PREFER_TEX_HEAP = 1u << 23, /* For buffers wanting to use texture heap */
    RT_BUFFER_HINT_STATIC              = 1u << 24, /* Tell OptiX that the buffer will be filled very rarely (maybe just once). Allows e.g. deallocation of host memory on unmap. Valid only with INPUT buffers and MAP_WRITE_DISCARD. */
} RTbufferflag_internal;

namespace optix {
enum ObjectStorageType
{
    OBJECT_STORAGE_CONSTANT,
    OBJECT_STORAGE_SHARED,
    OBJECT_STORAGE_GLOBAL,
    OBJECT_STORAGE_LINEAR_TEXTURE,
    OBJECT_STORAGE_BLOCKED_TEXTURE
};
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* __optix_optix_declarations_private_h__ */
