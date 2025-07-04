
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

#ifndef __optixu_optixu_vector_functions_h__
#define __optixu_optixu_vector_functions_h__

#include "optixu_vector_types.h"


#if defined(__cplusplus)

/*
 * We need to check to see if <vector_functions.h> has already been included before we've
 * had a chance to include it here.  If, so that means that the contents will be in the
 * global namespace, and we will add them to the optix namespace below in order to have
 * overloaded function in the optix namespace function correctly
 */

#  if defined(__VECTOR_FUNCTIONS_H__)
#    define RT_PULL_IN_VECTOR_FUNCTIONS
#  endif

namespace optix {

#endif /* #if defined (__cplusplus) */



#include <vector_functions.h>



#if defined(__cplusplus)
} /* end namespace optix */
#endif

/* Pull the global namespace LWCA types in LWCA C into the optix namespace. */
#if defined(RT_PULL_IN_VECTOR_FUNCTIONS)
#define RT_DEFINE_HELPER(type) \
  using ::make_##type##1; \
  using ::make_##type##2; \
  using ::make_##type##3; \
  using ::make_##type##4;

namespace optix {
RT_DEFINE_HELPER(char)
RT_DEFINE_HELPER(uchar)
RT_DEFINE_HELPER(short)
RT_DEFINE_HELPER(ushort)
RT_DEFINE_HELPER(int)
RT_DEFINE_HELPER(uint)
RT_DEFINE_HELPER(long)
RT_DEFINE_HELPER(ulong)
RT_DEFINE_HELPER(float)
RT_DEFINE_HELPER(longlong)
RT_DEFINE_HELPER(ulonglong)
RT_DEFINE_HELPER(double)
} /* end namespace optix */

#undef RT_DEFINE_HELPER

#endif

#endif /* #ifndef __optixu_optixu_vector_functions_h__ */

