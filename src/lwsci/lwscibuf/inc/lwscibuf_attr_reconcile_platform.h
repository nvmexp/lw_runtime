/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_RECONCILE_PLATFORM_H
#define INCLUDED_LWSCIBUF_ATTR_RECONCILE_PLATFORM_H

#include "lwscibuf_attr_priv.h"

LwSciError LwSciBufValidateGpuType(
    LwSciBufAttrList attrList,
    LwSciRmGpuId gpuId,
    LwSciBufGpuType gpuType,
    bool* match
);


#endif /* INCLUDED_LWSCIBUF_ATTR_RECONCILE_PLATFORM_H */
