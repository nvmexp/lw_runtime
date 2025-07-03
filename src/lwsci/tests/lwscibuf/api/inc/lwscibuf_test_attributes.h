/*
 * lwscibuf_test_attributes.h
 *
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCIBUF_TEST_ATTRIBUTES_H
#define INCLUDED_LWSCIBUF_TEST_ATTRIBUTES_H

#include "lwscibuf_test_integration.h"

#include <utility>

#define ATTR_NAME(key) LwSciBufAttrKeyToString((key))
#define INTERNAL_ATTR_NAME(key) LwSciBufInternalAttrKeyToString((key))

const char* LwSciBufAttrKeyToString(LwSciBufAttrKey key);
const char* LwSciBufInternalAttrKeyToString(LwSciBufInternalAttrKey key);

#endif // INCLUDED_LWSCIBUF_TEST_ATTRIBUTES_H
