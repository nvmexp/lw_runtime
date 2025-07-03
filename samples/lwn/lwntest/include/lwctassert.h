/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __LW_CTASSERT_H
#define __LW_CTASSERT_H

#include <assert.h>

#if defined(_MSC_VER) || (defined(__GNUC__) && defined(__cplusplus))
#define _Static_assert static_assert
#endif

#define ct_assert_i(x, str, file, line) _Static_assert(x, "ct_assert(" str ") failed at " file ":" #line)
#define ct_assert(x) ct_assert_i(x, #x, __FILE__, __LINE__)

#endif // __LW_CTASSERT_H
