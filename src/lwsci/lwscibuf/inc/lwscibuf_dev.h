/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciBuf Device Interfaces and Data Structures Definitions</b>
 *
 * @b Description: This file contains LwSciBuf device interfaces and
 *                 data structures definitions.
 */

#ifndef INCLUDED_LWSCIBUF_DEV_H
#define INCLUDED_LWSCIBUF_DEV_H

#if defined(__x86_64__)
#include "lwscibuf_dev_platform_x86.h"
#else
#include "lwscibuf_dev_platform_tegra.h"
#endif

#endif /* INCLUDED_LWSCIBUF_DEV_H */
