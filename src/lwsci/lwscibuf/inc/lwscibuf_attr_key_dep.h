/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_KEY_DEP_H
#define INCLUDED_LWSCIBUF_ATTR_KEY_DEP_H

#include "lwscibuf_attr_key_dep_platform.h"
#include "lwscibuf_constraint_lib.h"

#define NUM_TENSOR_DIMS 4

typedef LwSciError (*LwSciBufAttrListKeyDependency)(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListImageTensorKeyDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListTensorKeyDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListGpuCompressionDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint);

LwSciError LwSciBufAttrListGpuSwCacheCoherDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint);

LwSciError LwSciBufAttrListGpuCacheEnableDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListGpuIdDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListCpuKeysDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint);

LwSciError LwSciBufAttrListHeapTypeDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListMemDomainDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListSetDefaultCpuAccess(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListSetDefaultRequiredPerm(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListActualPermDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListVidmemDependency(
    LwSciBufAttrList attrList);

LwSciError LwSciBufAttrListSetGeneralKeyDependency(
    LwSciBufAttrList attrList,
    bool localPeer,
    LwSciIpcEndpoint ipcEndpoint);

#endif /* INCLUDED_LWSCIBUF_ATTR_KEY_DEP_H */
