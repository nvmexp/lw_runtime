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
 * \brief <b>LwSciSync Module Management Core Structures Definitions</b>
 */

#ifndef INCLUDED_LWSCISYNC_MODULE_PRIV_H
#define INCLUDED_LWSCISYNC_MODULE_PRIV_H

#include "lwscibuf.h"
#include "lwscicommon_objref.h"
#include "lwscicommon_covanalysis.h"
#include "lwscisync_backend.h"

/**
 * \brief Core structures declaration.
 */

/**
 * \brief LwSciSync core module structure.
 */
typedef struct {
    LwSciObj objModule;
    /** Magic ID to ensure this is valid module. This member must NOT be
     * modified in between allocation and deallocation of the
     * LwSciSyncCoreModule.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. If it does, this indicates corruption. As such, there is
     *        no data-dependency and no locking is necessary.
     */
    uint64_t header;
    /** Counter used to create unique ID for sync object
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is protected by LwSciObj lock
     *      - Any conlwrrent access must be serialized by holding the LwSciObj
     *        lock
     */
    uint64_t moduleCounter;
    /** LwSciBufModule needed for buffer allocation */
    LwSciBufModule bufModule;
    /**
     * Backend of RM resources. This member must NOT be modified in between
     * allocation and deallocation of the LwSciSyncCoreModule.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. This means conlwrrent reads will always read a constant
     *        value once the LwSciSyncModule is returned to the caller of any
     *        element-level API. As such, there is no data-dependency and no
     *        locking is necessary.
     */
    LwSciSyncCoreRmBackEnd backEnd;
} LwSciSyncCoreModule;

static inline LwSciSyncCoreModule*
    LwSciCastObjToSyncCoreModule(LwSciObj* arg)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),"LwSciSync-ADV-MISRAC2012-013")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciSync-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    return (LwSciSyncCoreModule*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciSyncCoreModule, objModule));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
}

/**
 * \brief LwSciSync module wrapper structure.
 */
struct LwSciSyncModuleRec {
    /** Reference to the core module */
    LwSciRef refModule;
};
#endif
