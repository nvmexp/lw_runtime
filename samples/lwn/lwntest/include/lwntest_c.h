/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// This file serves as the primary header used for lwntest code using the pure
// C interface for LWN, based on the header "lwn/lwn.h".  Files using the
// interface should generally not also use the C++ interface.

#ifndef __lwntest_c_h__
#define __lwntest_c_h__

// Allow using all deprecated LWN functions from lwogtest
#define LWN_PRE_DEPRECATED
#define LWN_POST_DEPRECATED

// Disallow the use of deprecated types.
#define LWN_NO_DEPRECATED_BASIC_TYPES

// For the pure C interface, we include the pure C API headers and the C
// function pointer interface (via inline functions).
#include "lwn/lwn.h"
#include "lwn/lwn_FuncPtrInline.h"

// Set up common lwntest definitions.
#include "cppogtest.h"

#endif // #ifndef __lwntest_c_h__
