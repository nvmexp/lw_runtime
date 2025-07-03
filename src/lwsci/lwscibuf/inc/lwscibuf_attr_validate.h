/*
 * Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_VALIDATE_H
#define INCLUDED_LWSCIBUF_ATTR_VALIDATE_H

#if !defined(__x86_64__)
#include "lwscibuf_attr_validate_platform_tegra.h"
#else
#include "lwscibuf_attr_validate_platform_x86.h"
#endif //!defined(__x86_64__)

#endif /* INCLUDED_LWSCIBUF_ATTR_VALIDATE_H */
