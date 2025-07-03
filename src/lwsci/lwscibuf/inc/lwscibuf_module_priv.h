/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciBuf Module Structures</b>
 *
 * @b Description: This file contains LwSciBuf Module private structures.
 */

#ifndef INCLUDED_LWSCIBUF_MODULE_PRIV_H
#define INCLUDED_LWSCIBUF_MODULE_PRIV_H

#include "lwscibuf_module.h"

#define LW_SCI_BUF_MODULE_MAGIC 0x1A2B3C4DU

/**
 * @brief Structure that LwSciBufModule points to.
 * This structure is allocated using LwSciCommon functionality.
 * An LwSciBufModuleRec holds a reference to a LwSciBufModuleObjPriv structure
 * which contains the actual module resource data. Multiple LwSciBufModuleRecs
 * can reference particular LwSciBufModuleObjPriv. This structure is
 * deallocated using LwSciCommon functionality.
 */
typedef struct LwSciBufModuleRec {
    /** Referencing header used for refcounting and locking the reference */
    LwSciRef refHeader;
} LwSciBufModuleRefPriv;

static inline LwSciBufModuleRefPriv*
    LwSciCastRefToBufModuleRefPriv(LwSciRef* arg)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    return (LwSciBufModuleRefPriv*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciBufModuleRefPriv, refHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}

/**
 * @brief Actual structure that contains data corresponding to a module
 * resource. This structure is allocated using LwSciCommon functionality as
 * part of allocating LwSciBufModuleRec. This structure is deallocated using
 * LwSciCommon functionality when all the LwSciBufModuleRecs holding reference
 * to it are deallocated.
 */
typedef struct {
    /** Object referencing header used for refcounting and locking the object
     */
    LwSciObj objHeader;
    /** Magic ID to detect if this LwSciBufModuleObjPriv is valid. This member
     * must be initialized to a particular constant value when the
     * LwSciBufModuleObjPriv is allocated. The constant value chosen to
     * initialize this member must be non-zero. It must be changed to a
     * different value when this LwSciBufModuleObjPriv is deallocated.
     * This member must NOT be modified in between allocation and deallocation
     * of the LwSciBufModuleObjPriv. Whenever an LwSciBufModuleObjPriv is
     * retrieved using LwSciCommon from an LwSciBufModuleRec received
     * (via an LwSciBufModule) from outside the module unit (including,
     * but not limited to, LwSciBufModuleValidate()), the module unit must
     * validate the Magic ID.
     *
     * Conlwrrency:
     *  - This is not protected by locking the LwSciObj since the value of this
     *    does not change during the lifecycle of the LwSciBufModuleObjPriv. If
     *    it does, this indicates corruption. As such, there is no
     *    data-dependency and no locking is necessary.
     */
    uint32_t magic;
    /**
     * Device context per module. This member must NOT be modified in between
     * allocation and deallocation of the LwSciBufModuleObjPriv.
     *
     * Conlwrrency:
     *  - This is not protected by locking the LwSciObj
     *  - Once returned from an element-level API, this value does not change.
     *    This means conlwrrent reads will always read a constant value once
     *    the LwSciBufModule is returned to the caller of any element-level
     *    API. As such, there is no data-dependency and no locking is
     *    necessary.
     */
    LwSciBufDev dev;
    /**
     * Interface open context for every allocation interface. This member must
     * NOT be modified in between allocation and deallocation of the
     * LwSciBufModuleObjPriv.
     *
     * Conlwrrency:
     *  - This is not protected by locking the LwSciObj
     *  - Once returned from an element-level API, this value does not change.
     *    This means conlwrrent reads will always read a constant value once
     *    the LwSciBufModule is returned to the caller of any element-level
     *    API. As such, there is no data-dependency and no locking is
     *    necessary.
     */
    void* iFaceOpenContext[LwSciBufAllocIfaceType_Max];
} LwSciBufModuleObjPriv;

static inline LwSciBufModuleObjPriv*
    LwSciCastObjToBufModuleObjPriv(LwSciObj* arg)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    return (LwSciBufModuleObjPriv*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciBufModuleObjPriv, objHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}

#endif /* INCLUDED_LWSCIBUF_MODULE_PRIV_H */
