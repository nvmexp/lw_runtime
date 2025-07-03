/*
 * lwscibuf_tu_constraints.h
 *
 * Header file to define gpu TU arch's constraint extraction APIs.
 *
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_TU_CONSTRAINTS_H
#define INCLUDED_LWSCIBUF_TU_CONSTRAINTS_H

#include "lwscierror.h"
#include "lwscibuf_constraint_lib.h"
#include "ctrl/ctrl2080/ctrl2080mc.h"
#include "class/cl2080.h"

LwSciError LwSciBufGetTUImageConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImageConstraints* imgConstraints);

LwSciError LwSciBufGetTUArrayConstraints(
    LwSciBufHwEngine engine,
    LwSciBufArrayConstraints* arrConstraints);

LwSciError LwSciBufGetTUImagePyramidConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints);

#endif /* INCLUDED_LWSCIBUF_TU_CONSTRAINTS_H */
