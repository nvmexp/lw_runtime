/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_KEY_DEP_PLATFORM_H
#define INCLUDED_LWSCIBUF_ATTR_KEY_DEP_PLATFORM_H

#include "lwscibuf_attr_mgmt.h"
#include "lwscibuf_constraint_lib.h"

/**
 * \brief Sets LwSciBufPrivateAttrKey_HeapType.
 *
 * \param[out] attrList reconciled LwSciBufAttrList.
 *
 * \return LwSciError
 */
LwSciError LwSciBufAttrListPlatformHeapDependency(
    LwSciBufAttrList attrList,
    LwSciBufHeapType* heapType);

LwSciError LwSciBufAttrListPlatformGpuIdDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListPlatformGetDefaultGpuCacheability(
    LwSciBufAttrList attrList,
    LwSciRmGpuId gpuId,
    bool* defaultCacheability);

LwSciError LwSciBufAttrListPlatformGpuCompressionDependency(
    LwSciBufAttrList attrList,
    LwSciRmGpuId gpuId,
    bool isBlockLinear,
    bool* isCompressible);

LwSciError LwSciBufAttrListPlatformVidmemDependency(
    LwSciBufAttrList attrList);

#endif /* INCLUDED_LWSCIBUF_ATTR_KEY_DEP_PLATFORM_H */
