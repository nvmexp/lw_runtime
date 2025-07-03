/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_RECONCILE_PRIV_H
#define INCLUDED_LWSCIBUF_ATTR_RECONCILE_PRIV_H

#include "lwscibuf_attr_reconcile.h"

typedef struct {
    LwSciBufAttrList appendList;
    LwSciBufAttrList reconciledList;
} LwSciBufReconcilePolicyCookie;

typedef LwSciError (*LwSciBufPolicySameKey)(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

typedef LwSciError (*LwSciBufPolicyDifferentKey)(
    uint32_t key1,
    uint32_t key2,
    LwSciBufAttrList reconciledList);

typedef LwSciError (*LwSciBufMergeBufferType)(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[]);

typedef LwSciError (*LwSciBufCheckBufferType)(
    LwSciBufAttrList reconciledList);

static LwSciError GpuCacheAndPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError GpuCompressionMatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError MatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError OrPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError MaxPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError ArrayUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError ArrayIntersectionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError ListUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyMatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyOrPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyMaxPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyArrayUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyArrayIntersectionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyListUnionPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyGpuCacheAndPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError VerifyGpuCompressionMatchPolicySameKey(
    uint32_t key,
    const void* ipAddr,
    const LwSciBufAttrStatus* ipStatus,
    const uint64_t* ipSetSize,
    void* recAddr,
    LwSciBufAttrStatus* recStatus,
    uint64_t* recSetSize,
    const LwSciBufReconcilePolicyCookie* cookie);

static LwSciError CompareDifferentKeysForMatch(
    uint32_t key1,
    uint32_t key2,
    LwSciBufAttrList reconciledList);

static LwSciError LwSciBufAttrListMergeRawAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[]);

static LwSciError LwSciBufAttrListMergeImgAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[]);

static LwSciError LwSciBufAttrListMergeTensorAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[]);

static LwSciError LwSciBufAttrListMergeArrAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[]);

static LwSciError LwSciBufAttrListMergePyrAttr(
    LwSciBufAttrList appendList,
    LwSciBufAttrList reconciledList,
    const LwSciBufPolicySameKey policyMap[]);

#endif /* INCLUDED_LWSCIBUF_ATTR_RECONCILE_PRIV_H */
