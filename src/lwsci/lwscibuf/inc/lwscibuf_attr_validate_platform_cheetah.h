/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_TEGRA_H
#define INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_TEGRA_H

#include "lwscibuf_attr_validate_platform_common.h"

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
#if (LW_IS_SAFETY == 0)
RANGE_CHECK(MemDomain, LwSciBufMemDomain_Cvsram, LwSciBufMemDomain_Vidmem,
    LwSciBufMemDomain)
#else
RANGE_CHECK(MemDomain, LwSciBufMemDomain_Sysmem, LwSciBufMemDomain_Sysmem,
    LwSciBufMemDomain)
#endif //(LW_IS_SAFETY == 0)
RANGE_CHECK(HeapType, LwSciBufHeapType_IOMMU, LwSciBufHeapType_CvsRam,
    LwSciBufHeapType)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))

static inline LwSciError LwSciBufValidateHwEngine(
    LwSciBufAttrList attrList,
    const void *val)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufHwEngine engine = *(const LwSciBufHwEngine *)val;
    LwSciBufHwEngName engineName = {0};

    (void)attrList;

    /** TODO : We need to also validate if the engine is actually present on the system or not.
      *  For this we will need API from constraint_lib to get list of all engine versions for owner platform
      */

    if (LwSciBufHwEngine_TegraNamespaceId != engine.engNamespace) {
        LWSCI_ERR_STR("Engine namespace doesnt not match LwSciBufHwEngine_TegraNamespaceId on cheetah platform.");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    sciErr = LwSciBufHwEngGetNameFromId(engine.rmModuleID, &engineName);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("Invalid HW engine value.");
    }

ret:
    return sciErr;
}

#endif /* INCLUDED_LWSCIBUF_ATTR_VALIDATE_PLATFORM_TEGRA_H */
