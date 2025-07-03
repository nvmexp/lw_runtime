/*
 * Copyright (c) 2018 - 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_PRIV_H
#define INCLUDED_LWSCIBUF_ALLOC_PRIV_H

#include <limits.h>

#include "lwscibuf_attr_reconcile.h"
#include "lwscibuf_alloc_interface.h"
#include "lwscibuf_obj_mgmt.h"

/**
 * @brief macro definitions
 */
#define LW_SCI_BUF_OBJ_MAGIC            0xDEADBEEFU

#define LW_SCI_BUF_LWMEDIA_FLAG_COUNT   (sizeof(uint32_t) * (uint32_t)(CHAR_BIT))

#define LWSCIBUF_OBJ_MAX_STR_SIZE       50

/*
 * Lwrrently, LwSciBuf Supports allocation value of single heap.
 * When multiple heaps are needed, attribute management layer also
 * has to change. Hence, Setting Max allowed heaps to 1.
 */
#define LWSCIBUF_MAX_ALLOWED_HEAPS 1

/**
 * @brief structure to store allocation parameters of buffer
 */
typedef struct {
    uint64_t size;
    uint64_t alignment;
    bool coherency;
    LwSciBufHeapType heap[LWSCIBUF_MAX_ALLOWED_HEAPS];
    uint32_t numHeaps;
    bool cpuMapping;
} LwSciBufObjAllocVal;

/**
 * @brief private representation of LwSciBuf object
 */
typedef struct {
    /** Object referencing header used for refcounting and locking the object */
    LwSciObj objHeader;
    /**
     * Magic ID for sanity check of object. This member must NOT be modified in
     * between allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. If it does, this indicates corruption. As such, there is
     *        no data-dependency and no locking is necessary.
     */
    uint32_t magic;
    /**
     * Reference to LwSciBufAttrList. This member must NOT be modified in
     * between allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Modification of this value during the lifecycle of the
     *        LwSciBufObjPriv must be serialized
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary to obtain the associated LwSciBufAttrList.
     */
    LwSciBufAttrList attrList;
    /** parent LwSciBufObj */
    LwSciBufObj parentObj;
    /**
     * Allocation interface type. This member must NOT be modified in between
     * allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    LwSciBufAllocIfaceType allocType;
    /**
     * LwSciBuf allocation interface open context. This member must NOT be
     * modified in between allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    void* openContext;
    /**
     * LwSciBuf allocation interface context. This member must NOT be modified
     * in between allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    void* allocContext;
    /**
     * RM handle. This member must NOT be modified in between allocation and
     * deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    LwSciBufRmHandle rmHandle;
    /**
     * offset of the memory represented by object. This member must NOT be
     * modified in between allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    uint64_t offset;
    /**
     * CPU pointer. This member must NOT be modified in between allocation and
     * deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    void* ptr;
    /* Whether CPU access needed for buffer. This member must NOT be modified
     * in between allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     * */
    bool needCpuAccess;
    /* access permissions for the buffer. This member must NOT be modified in
     * between allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    LwSciBufAttrValAccessPerm accessPerm;
    /* allocation parameters. This member must NOT be modified in between
     * allocation and deallocation of the LwSciBufObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. As such, there is no data-dependency and no locking is
     *        necessary.
     */
    LwSciBufObjAllocVal allocVal;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    /* Interface specific target handle for C2c case. */
    LwSciC2cInterfaceTargetHandle c2cInterfaceTargetHandle;
    /* LwSciC2c copy functions */
    LwSciC2cCopyFuncs c2cCopyFuncs;
    /* boolean flags indicating if LwSciBufObj is local or remote.
     * It is set to TRUE in C2c case when remote LwSciBufObj is imported,
     * false otherwise.
     */
    bool isRemoteObject;
#endif
} LwSciBufObjPriv;

static inline LwSciBufObjPriv*
    LwSciCastObjToBufObjPriv(LwSciObj* arg)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    return (LwSciBufObjPriv*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciBufObjPriv, objHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}

typedef struct LwSciBufObjRefRec {
    /** Referencing header used for refcounting and locking the reference */
    LwSciRef refHeader;
    /**
     * This field is used to get and set LwMedia flags for an LwSciBuf object.
     * LwMedia can use each bit of this field for each LwMedia datatype to
     * figure out if LwMedia datatype object is associated with LwSciBuf object.
     *
     * Note-1: all the bits in the field are set to 0 when LwSciBuf object is
     * created or duplicated using LwSciBufObjDup() call.
     *
     * Note-2: In future, if datatype of 'lwMediaFlag' is changed then
     * macro 'LW_SCI_BUF_LWMEDIA_FLAG_COUNT' which gives number of bits in
     * 'lwMediaFlag' should also be changed
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This member should be protected by locking the LwsciRef using
     *        LwSciCommonRefLock() before reads/writes
     */
    uint32_t lwMediaFlag;
} LwSciBufObjRefPriv;

static inline LwSciBufObjRefPriv*
    LwSciCastRefToBufObjRefPriv(LwSciRef* arg)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    return (LwSciBufObjRefPriv*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciBufObjRefPriv, refHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}

/**
 * @brief structure keeping track of parameters needed to figure out allocation
 * type of buffer
 */
typedef struct {
    /* Note: In future, allocation type can be figured out from parameters apart
     * from memory domain. This structure can be used to extend list of
     * parameters that are needed to figure out allocation type
     */
    LwSciBufMemDomain memDomain;
} LwSciBufObjAllocTypeParams;

/**
 * @brief mapping of heaps from LwSciBufObj --> LwSciBufAllocIface layer
 */
typedef struct {
    const char heapName[LWSCIBUF_OBJ_MAX_STR_SIZE];
    LwSciBufAllocIfaceHeapType allocIfaceHeap;
} LwSciBufObjToallocIfaceHeapMap;

#endif /* INCLUDED_LWSCIBUF_ALLOC_PRIV_H */
