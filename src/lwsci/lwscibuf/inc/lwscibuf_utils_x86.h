/*
 * lwscibuf_utils_x86.h
 *
 * Header file for constraint library S/W Unit
 *
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscierror.h"
#include "lwRmShim/lwRmShimError.h"

#ifndef INCLUDED_LWSCIBUF_UTILS_X86_H
#define INCLUDED_LWSCIBUF_UTILS_X86_H

LwSciError LwRmShimErrorToLwSciError(
    LwRmShimError shimErr);

#endif  /* INCLUDED_LWSCIBUF_UTILS_H */
