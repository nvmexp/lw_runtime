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
// C++ interface for LWN, based on the header "lwn/lwn_Cpp.h".  Files using
// the interface should generally not also use the C interface.

#ifndef __lwntest_cpp_h__
#define __lwntest_cpp_h__

// Allow for use of all deprecated LWN functions from lwntest.
#define LWN_PRE_DEPRECATED
#define LWN_POST_DEPRECATED

// Disallow the use of deprecated types.
#define LWN_NO_DEPRECATED_BASIC_TYPES

// Allow for user-defined overloads of LWN C++ classes by dropping them into
// lwn::objects.
#define LWN_OVERLOAD_CPP_OBJECTS

// For the C++ interface, we need to include the pure C++ API headers and the
// C++ class method definitions.  We don't include the C API headers at all
// here to prevent mixing of C and C++ constructs.
#include "lwn/lwn_Cpp.h"
#include "lwn/lwn_CppMethods.h"

// Or at least, we'd like to. However, the lwnUtil utility code includes C
// function calls and defines in their headers, so our C++ test code still
// needs to know how to call C functions.
#include "lwn/lwn.h"
#include "lwn/lwn_FuncPtrInline.h"

// Set up common lwntest definitions.
#include "cppogtest.h"

// Set up LWN utility code to use the native C++ interface.
#define LWNUTIL_USE_CPP_INTERFACE

#endif // #ifndef __lwntest_cpp_h__
