/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_CONSTRAINT_PRIV_H
#define INCLUDED_LWSCIBUF_ATTR_CONSTRAINT_PRIV_H

#include "lwscibuf_attr_constraint.h"
#include "lwscibuf_constraint_lib.h"

typedef LwSciError (*LwSciBufApplyTypeConstraint)(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints);

static LwSciError LwSciBufRawConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints);

static LwSciError LwSciBufImageConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints);

static LwSciError LwSciBufArrayConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints);

static LwSciError LwSciBufPyramidConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints);

static LwSciError LwSciBufTensorConstraint(
    LwSciBufAttrList attrList,
    const LwSciBufHwConstraints* constraints);

#endif /* INCLUDED_LWSCIBUF_ATTR_CONSTRAINT_PRIV_H */
