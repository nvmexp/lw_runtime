/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_X86_H
#define INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_X86_H

#include "lwscibuf_attr_validate_platform_common.h"

RANGE_CHECK(MemDomain, LwSciBufMemDomain_Sysmem, LwSciBufMemDomain_Vidmem,
    LwSciBufMemDomain)
RANGE_CHECK(HeapType, LwSciBufHeapType_Resman, LwSciBufHeapType_Resman,
    LwSciBufHeapType)

static inline LwSciError LwSciBufValidateHwEngine(
    LwSciBufAttrList attrList,
    const void *val)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufHwEngine engine = *(const LwSciBufHwEngine *)val;

    (void)attrList;

    /** TODO : We need to also validate if the engine is actually present on the system or not.
      *  For this we will need API from deivce unit to give us GPU arch & engines inside GPU arch
      */

    if (LwSciBufHwEngine_ResmanNamespaceId != engine.engNamespace) {
        LWSCI_ERR_STR("Invalid HW engine value.");
        sciErr = LwSciError_BadParameter;
    }

    return sciErr;
}

#endif /* INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_X86_H */
