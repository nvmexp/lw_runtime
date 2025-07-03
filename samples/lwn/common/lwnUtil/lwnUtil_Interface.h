/*
 * Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
 *
 * THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
 * LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
 * IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
 *
 *
 */
#ifndef __lwnUtil_Interface_h__
#define __lwnUtil_Interface_h__

//
// LWN UTILITY INTERFACE SELECTION
//
// This module provides defines to select the LWN API interface supported by
// LWN utility code modules.  This file will set up a #define
// "LWNUTIL_INTERFACE_TYPE" indicating which interface (C or C++) should be
// provided.  The value is one of the two "LWNUTIL_INTERFACE_TYPE_XXX"
// #defines below.
//
// By default, the lwnUtil modules only support the native C interface.
// However, if an application defines LWNUTIL_USE_CPP_INTERFACE before
// including this file, the C++ interface will be supported.
//

// #defines identifying which of the interfaces is being used;
// LWNUTIL_INTERFACE_TYPE will be set to one of these values.
#define LWNUTIL_INTERFACE_TYPE_C         0
#define LWNUTIL_INTERFACE_TYPE_CPP       1

#if defined(LWNUTIL_INTERFACE_TYPE)
#error "Don't set LWNUTIL_INTERFACE_TYPE before including lwnUtil_Interface.h."
#endif

#if defined(LWNUTIL_USE_CPP_INTERFACE)
#define LWNUTIL_INTERFACE_TYPE      LWNUTIL_INTERFACE_TYPE_CPP
#else
#define LWNUTIL_INTERFACE_TYPE      LWNUTIL_INTERFACE_TYPE_C
#endif

#endif // #ifndef __lwnUtil_Interface_h__
