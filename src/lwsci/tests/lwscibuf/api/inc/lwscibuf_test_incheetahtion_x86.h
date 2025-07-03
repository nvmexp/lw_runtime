/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_TEST_INTEGRATION_X86_H
#define INCLUDED_LWSCIBUF_TEST_INTEGRATION_X86_H

#include <string.h>
#include "ctrl/ctrl0041.h"
#include "lwRmApi.h"
#include "ctrl/ctrl0000.h"
#include "ctrl/ctrl2080/ctrl2080mc.h"
#include "class/cl0080.h"
#include "class/cl2080.h"

#if defined(LW_LINUX)
#define CEIL_TO_LEVEL(x, lvl) 4096
#elif defined(LW_QNX)
#define CEIL_TO_LEVEL(x, lvl) 32768
#endif

const char *lwstatusToString(LwU32 lwStatusIn);

#endif
