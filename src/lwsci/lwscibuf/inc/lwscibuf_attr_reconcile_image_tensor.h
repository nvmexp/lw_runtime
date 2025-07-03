/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_RECONCILE_IMAGE_TENSOR_H
#define INCLUDED_LWSCIBUF_ATTR_RECONCILE_IMAGE_TENSOR_H

#include "lwscibuf_attr_reconcile_datatypes.h"

LwSciError LwSciBufAttrListGetImageTensorRecKeyPair(
    const LwSciBufAttrListRecKeyPair** recKeyPair,
    size_t* numElements);

LwSciError LwSciBufAttrListLwstomCompareImageTensor(
    LwSciBufAttrList attrList);

#endif /* INCLUDED_LWSCIBUF_ATTR_RECONCILE_IMAGE_TENSOR_H */
