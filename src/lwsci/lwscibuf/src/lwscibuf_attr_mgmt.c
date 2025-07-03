/*
 * Copyright (c) 2018-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdlib.h>

#include "lwscibuf_attr_mgmt.h"
#include "lwscibuf_utils.h"
#include "lwscicommon_os.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscibuf_constraint_lib.h"

#define LW_SCI_BUF_ATTR_MAGIC 0xA1D2C0DEU

/**
 * @brief This definition is used during definition of descriptor list
 */
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 20_10), "LwSciBuf-ADV-MISRAC2012-021")
#define LW_SCI_BUF_ATTR_KEY_DEF(_key, _datatype, _dem, _perm, _recpolicy,\
    _externalValidateFn, _internalValidateFn, _importQualifier,\
    _reconcileQualifier, _localPeerIpcAffinity, _remotePeerIpcAffinity) \
    [LW_SCI_BUF_DECODE_ATTRKEY(_key)] = { \
                .name = #_key, \
                .dataSize = (uint32_t)sizeof(_datatype), \
                .dataIndex = (uint32_t)(_key), \
                .dataOffset = (uintptr_t)&(((PARENT *)0)->_key[0]), \
                .sizeOffset = (uintptr_t)&(((PARENT *)0)->_key##_Size), \
                .statusOffset = (uintptr_t)&(((PARENT *)0)->_key##_Status), \
                .dataMaxInstance = (uint32_t)(_dem), \
                .keyAccess = (LwSciBufKeyAccess)(_perm), \
                .recpolicy = (_recpolicy), \
                .externalValidateFn = (_externalValidateFn), \
                .internalValidateFn = (_internalValidateFn), \
                .importQualifier = (_importQualifier), \
                .reconcileQualifier = (_reconcileQualifier), \
                .localPeerIpcAffinity = _localPeerIpcAffinity, \
                .remotePeerIpcAffinity = _remotePeerIpcAffinity, \
                },

#define KEY_HAS_DESC_ENTRY(keyType, dataType, key) \
    ((attrKeyDescList[keyType][dataType][key]).dataIndex != 0U)

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_4), "LwSciBuf-ADV-MISRAC2012-013")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_5), "LwSciBuf-ADV-MISRAC2012-020")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_1), "LwSciBuf-REQ-MISRAC2012-003")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 9_3), "LwSciBuf-REQ-MISRAC2012-002")
static const LwSciBufAttrKeyDescPriv
    attrKeyDescList[LwSciBufAttrKeyType_MaxValid]
                    [LwSciBufType_MaxValid]
                    [LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE] = {
        [LwSciBufAttrKeyType_Public] = {
            [LwSciBufType_General] = {
                #define PARENT LwSciBufGeneralAttrObjPriv
                LW_SCI_BUF_PUB_GENERAL_ATTR
                #undef PARENT
            },
            [LwSciBufType_RawBuffer] = {
                #define PARENT LwSciBufRawBufferAttrObjPriv
                LW_SCI_BUF_PUB_RAWBUFFER_ATTR
                #undef PARENT
            },
            [LwSciBufType_Image] = {
                #define PARENT LwSciBufImageAttrObjPriv
                LW_SCI_BUF_PUB_IMAGE_ATTR
                #undef PARENT
            },
            [LwSciBufType_Tensor] = {
                #define PARENT LwSciBufTensorAttrObjPriv
                LW_SCI_BUF_PUB_TENSOR_ATTR
                #undef PARENT
            },
            [LwSciBufType_Array] = {
                #define PARENT LwSciBufArrayAttrObjPriv
                LW_SCI_BUF_PUB_ARRAY_ATTR
                #undef PARENT
            },
            [LwSciBufType_Pyramid] = {
                #define PARENT LwSciBufPyramidAttrObjPriv
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
                LW_SCI_BUF_PUB_PYRAMID_ATTR
                #undef PARENT
            }
        },
        [LwSciBufAttrKeyType_Internal] = {
            [LwSciBufType_General] = {
                #define PARENT LwSciBufGeneralAttrObjPriv
                LW_SCI_BUF_INT_GENERAL_ATTR
                #undef PARENT
            },
            [LwSciBufType_RawBuffer] = {
                {0}
            },
            [LwSciBufType_Image] = {
                #define PARENT LwSciBufImageAttrObjPriv
                LW_SCI_BUF_INT_IMAGE_ATTR
                #undef PARENT
            },
            [LwSciBufType_Tensor] = {
                {0}
            },
            [LwSciBufType_Array] = {
                {0}
            },
            [LwSciBufType_Pyramid] = {
                {0}
            }
        },
        [LwSciBufAttrKeyType_Private] = {
            [LwSciBufType_General] = {
                #define PARENT LwSciBufPrivateAttrObjPriv
                LW_SCI_BUF_PRIVATE_ATTR
                #undef PARENT
            },
            [LwSciBufType_RawBuffer] = {
                {0}
            },
            [LwSciBufType_Image] = {
                {0}
            },
            [LwSciBufType_Tensor] = {
                {0}
            },
            [LwSciBufType_Array] = {
                {0}
            },
            [LwSciBufType_Pyramid] = {
                {0}
            }
        },
        [LwSciBufAttrKeyType_Transport] = {
            [LwSciBufType_General] = {
                {0}
            },
            [LwSciBufType_RawBuffer] = {
                {0}
            },
            [LwSciBufType_Image] = {
                {0}
            },
            [LwSciBufType_Tensor] = {
                {0}
            },
            [LwSciBufType_Array] = {
                {0}
            },
            [LwSciBufType_Pyramid] = {
                {0}
            }
        },
        [LwSciBufAttrKeyType_UMDPrivate] = {
            [LwSciBufType_General] = {
                #define PARENT LwSciBufUmdAttrObjPriv
                LW_SCI_BUF_INT_UMD_ATTR
                #undef PARENT
            },
            [LwSciBufType_RawBuffer] = {
                {0}
            },
            [LwSciBufType_Image] = {
                {0}
            },
            [LwSciBufType_Tensor] = {
                {0}
            },
            [LwSciBufType_Array] = {
                {0}
            },
            [LwSciBufType_Pyramid] = {
                {0}
            }
        },
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 9_3))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_1))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_5))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_4))

static inline bool LwSciBufIsAttrKeyMemberof(
    uint32_t key,
    LwSciBufAttrKeyType keytype,
    uint32_t datatype)
{
    uint32_t decodedDataType = LW_SCI_BUF_DECODE_DATATYPE(key);
    uint32_t decodedKeyType = LW_SCI_BUF_DECODE_KEYTYPE(key);
    return ((decodedDataType == datatype) && (decodedKeyType == (uint32_t)keytype));
}

static bool LwSciBufAttrListIsWritable(
    LwSciBufAttrList attrList)
{
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;
    LwSciBufAttrListState state = LwSciBufAttrListState_UpperBound;
    bool isAttrListWritable = false;

    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);
    state = attrListObj->state;

    isAttrListWritable = (state == LwSciBufAttrListState_Unreconciled);

    return isAttrListWritable;
}

static LwSciError LwSciBufValidateLwSciBufType(
    uint32_t bufType)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (bufType >= (uint32_t)LwSciBufType_MaxValid) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufValidateAttrKeyType(
    uint32_t keyType)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((uint32_t)LwSciBufAttrKeyType_MaxValid <= keyType) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrKeyValidate(
    uint32_t key)
{
    LwSciError err = LwSciError_Success;

    uint32_t decodedKey = LW_SCI_BUF_DECODE_ATTRKEY(key);
    uint32_t decodedDataType = LW_SCI_BUF_DECODE_DATATYPE(key);
    uint32_t decodedKeyType = LW_SCI_BUF_DECODE_KEYTYPE(key);

    LWSCI_FNENTRY("");

    if ((uint32_t)LwSciBufAttrKeyType_UMDPrivate == decodedKeyType) {
        /* We assume all UMD-specific keys are valid */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE <= decodedKey) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufValidateLwSciBufType(decodedDataType);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufValidateAttrKeyType(decodedKeyType);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* If this isn't a UMD-specific key, the integer is considered valid if and
     * only if there exists a valid Attribute Key Descriptor associated. */
     LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if (!KEY_HAS_DESC_ENTRY(decodedKeyType, decodedDataType, decodedKey)) {
        LWSCI_ERR_STR("Invalid key descriptor entry");
        LWSCI_ERR_UINT("decodedKeyType: ", decodedKeyType);
        LWSCI_ERR_UINT("decodedDataType: ", decodedDataType);
        LWSCI_ERR_UINT("decodedKey: ", decodedKey);

        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");

    return err;
}

static LwSciError LwSciBufAttrKeyIsOfKeyType(
    uint32_t key,
    LwSciBufAttrKeyType keyType)
{
    LwSciError err = LwSciError_Success;

    uint32_t decodedKeyType = 0U;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrKeyValidate(key);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufValidateAttrKeyType((uint32_t)keyType);
    if (LwSciError_Success != err) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LwSciCommonPanic();
    }

    decodedKeyType = LW_SCI_BUF_DECODE_KEYTYPE(key);
    if (decodedKeyType != (uint32_t)keyType) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static void LwSciBufFreeIpcRouteCb(
    void* valPtr)
{
    LwSciBufIpcRoute* const * ipcRoute = (LwSciBufIpcRoute* const *)valPtr;
    LwSciBufFreeIpcRoute(ipcRoute);
}

static LwSciError LwSciBufIpcRouteCloneCb(
    const void* srcIpcRouteAddr,
    void* dstIpcRouteAddr)
{
    LwSciBufIpcRoute* const * srcIpcRoutePtr = (LwSciBufIpcRoute* const *)srcIpcRouteAddr;
    const LwSciBufIpcRoute* srcIpcRoute = *srcIpcRoutePtr;
    LwSciBufIpcRoute** dstIpcRoute = (LwSciBufIpcRoute**)dstIpcRouteAddr;

    return LwSciBufIpcRouteClone(&srcIpcRoute, dstIpcRoute);
}

static void LwSciBufFreeIpcTableCb(
    void* valPtr)
{
    LwSciBufIpcTable* const * ipcTable = (LwSciBufIpcTable**)valPtr;
    LwSciBufFreeIpcTable(ipcTable);
}

static LwSciError LwSciBufIpcTableCloneCb(
    const void* srcIpcTableAddr,
    void* dstIpcTableAddr)
{
    LwSciBufIpcTable* const * srcIpcTablePtr = (LwSciBufIpcTable* const *)srcIpcTableAddr;
    const LwSciBufIpcTable* srcIpcTable = *srcIpcTablePtr;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
    LwSciBufIpcTable** dstIpcTable = (LwSciBufIpcTable**)dstIpcTableAddr;

    return LwSciBufIpcTableClone(&srcIpcTable, dstIpcTable);
}

static const uint64_t dataTypeSizeMap[LwSciBufType_MaxValid] = {
    [LwSciBufType_General]      = sizeof(LwSciBufGeneralAttrObjPriv),
    [LwSciBufType_RawBuffer]    = sizeof(LwSciBufRawBufferAttrObjPriv),
    [LwSciBufType_Image]        = sizeof(LwSciBufImageAttrObjPriv),
    [LwSciBufType_Tensor]       = sizeof(LwSciBufTensorAttrObjPriv),
    [LwSciBufType_Array]        = sizeof(LwSciBufArrayAttrObjPriv),
    [LwSciBufType_Pyramid]      = sizeof(LwSciBufPyramidAttrObjPriv),
};

static inline void* PtrAddOffset(
    void* ptr,
    uintptr_t offset)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    return (void*)&(((uint8_t*)ptr)[offset]);
}

static void LwSciBufAttrGetDesc(
    uint32_t key,
    const LwSciBufAttrKeyDescPriv** desc)
{
    LwSciError err = LwSciError_Success;
    uint32_t decodedKey = 0U;
    uint32_t decodedDataType = 0U;
    uint32_t decodedKeyType = 0U;

    LWSCI_FNENTRY("");

    /* Sanitize output */
    *desc = NULL;

    /* Decode Key */
    err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType,
            &decodedKeyType);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to decode key\n");
        LWSCI_ERR_UINT("key: \n", key);
        LwSciCommonPanic();
    }

    /* Get descriptor entry and validate */
    *desc = &attrKeyDescList[decodedKeyType][decodedDataType][decodedKey];
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if (!KEY_HAS_DESC_ENTRY(decodedKeyType, decodedDataType, decodedKey)) {
        LWSCI_ERR_STR("Invalid key descriptor entry\n");
        LWSCI_ERR_UINT("decodedKeyType: ", decodedKeyType);
        LWSCI_ERR_UINT(", decodedDataType: ", decodedDataType);
        LWSCI_ERR_UINT(", decodedKey: \n", decodedKey);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
int32_t LwSciBufAttrKeyCompare(
    const void* elem1,
    const void* elem2)
{
    int32_t ret = 0;
    uint32_t attrKey1 = 0U;
    uint32_t attrKey2 = 0U;

    if ((NULL == elem1) || (NULL == elem2)) {
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    attrKey1 = *(const uint32_t*)elem1;
    attrKey2 = *(const uint32_t*)elem2;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LWSCI_FNENTRY("");

    if (attrKey1 > attrKey2) {
        ret = 1;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (attrKey1 < attrKey2) {
        ret = -1;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return ret;
}

/**
 * @brief Static function definition
 */
static uint64_t LwSciBufAttrListTotalSlotCount(
    const LwSciBufAttrList inputAttrListArray[],
    uint64_t attrListCount)
{
    uint64_t totalSlotCount = 0U;
    uint64_t i = 0U;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;
    uint8_t status = OP_FAIL;

    LWSCI_FNENTRY("");

    for (i=0; i<attrListCount; i++) {
        /* get object from reference */
        LwSciCommonGetObjFromRef(&inputAttrListArray[i]->refHeader,
            &attrListObjParam);

        attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);;

        //totalSlotCount += attrListObj->slotCount
        u64Add(totalSlotCount, attrListObj->slotCount, &totalSlotCount, &status);

        if (OP_SUCCESS != status) {
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto count_zero;
        }
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

count_zero:
    totalSlotCount = 0U;
ret:
    LWSCI_INFO("Output: total SlotCount %lu", totalSlotCount);
    LWSCI_FNEXIT("");
    return totalSlotCount;
}

static void LwSciBufAttrListResetStatus(
    LwSciBufAttrList dst,
    size_t dstSlotIndex)
{
    uint32_t key = 0U;
    void* baseAddr = NULL;
    LwSciBufAttrStatus* status = NULL;
    uint64_t* setLen = NULL;
    LwSciBufAttrKeyIterator iter;
    bool keyEnd = false;
    bool datatypeEnd = false;
    bool keytypeEnd = false;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: dst: %p\n", dst);

    /* Initialize iterator to public keys */
    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_General, 1U, &iter);

    while (true) {
        LwSciBufAttrKeyIterNext(&iter, &keytypeEnd, &datatypeEnd,
            &keyEnd, &key);

        /* end of iterator */
        if ((true == datatypeEnd) || (true == keytypeEnd)) {
            break;
        }

        /* end of one datatype, just continue to next */
        if (true == keyEnd) {
            continue;
        }

        /* get key details and reset the status */
        LwSciBufAttrGetKeyDetail(dst, dstSlotIndex, key, &baseAddr, &status, &setLen);
        if ((NULL != status) && (LwSciBufAttrStatus_SetLocked == *status)) {
            *status = LwSciBufAttrStatus_SetUnlocked;
        }
    }

    /* Initialize iterator for internal keys */
    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Internal,
        (uint32_t)LwSciBufType_General, 1, &iter);

    while (true) {
        (void)LwSciBufAttrKeyIterNext(&iter, &keytypeEnd, &datatypeEnd,
                &keyEnd, &key);

        /* end of iterator */
        if ((true == datatypeEnd) || (true == keytypeEnd)) {
            break;
        }

        /* end of one datatype, just continue to next */
        if (true == keyEnd) {
            continue;
        }

        /* get key details and reset the status */
        LwSciBufAttrGetKeyDetail(dst, dstSlotIndex, key, &baseAddr, &status, &setLen);
        if ((NULL != status) && (LwSciBufAttrStatus_SetLocked == *status)) {
            *status = LwSciBufAttrStatus_SetUnlocked;
        }
    }

    LWSCI_FNEXIT("");
    return;
}

static LwSciError LwSciBufAttrIsWritable(
    LwSciBufAttrList attrList,
    uint32_t key,
    bool *isWritable)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrListState state = LwSciBufAttrListState_UpperBound;
    const LwSciBufAttrKeyDescPriv* desc = NULL;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LwSciBufAttrGetDesc(key, &desc);
    if (NULL == desc) {
        LWSCI_ERR_STR("Failed to check if key is writable.\n");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);
    state = attrListObj->state;

    /* Previously, we checked if (state == LwSciBufAttrListState_Unreconciled)
     * to allow only unreconciled LwSciBufAttrList to be written. We have
     * changed it to (state != LwSciBufAttrListState_Reconciled) which allows
     * appended LwSciBufAttrList to be written as well. Note that, we have
     * added check LwSciBufAttrListIsWritable() which allows only unreconciled
     * LwSciBufAttrList to be writable. This check MUST be added in any
     * element level API (such as LwSciBufAttrListSetAttrs/SetInternalAttrs).
     * This ensures that user of LwSciBuf can only write to unreconciled
     * LwSciBufAttrList while LwSciBuf can write to appended LwSciBufAttrList
     * too even if 'override' flag is passed as false to
     * LwSciBufAttrListCommonSetAttrs().
     * In other words, if we are here and state = LwSciBufAttrListState_Appended
     * then we should be here only in cases where LwSciBuf internally calls
     * LwSciBufAttrListCommonSetAttrs() with override flag as false. We must
     * never come here from element level set APIs if
     * state = LwSciBufAttrListState_Appended.
     */
    *isWritable = (state != LwSciBufAttrListState_Reconciled) &&
                    (desc->keyAccess != LwSciBufKeyAccess_Output);

ret:
    return err;
}

static LwSciError LwSciBufAttrIsReadable(
    LwSciBufAttrList attrList,
    uint32_t key,
    bool* isReadable)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrListState state = LwSciBufAttrListState_UpperBound;
    const LwSciBufAttrKeyDescPriv* desc = NULL;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;
    bool isReconciled = false;

    LwSciBufAttrGetDesc(key, &desc);
    if (NULL == desc) {
        LWSCI_ERR_STR("Failed to check if key is writable.\n");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);
    state = attrListObj->state;

    isReconciled = (state == LwSciBufAttrListState_Reconciled);

    *isReadable = (!isReconciled && (desc->keyAccess != LwSciBufKeyAccess_Output)) ||
                (isReconciled && (desc->keyAccess != LwSciBufKeyAccess_Input));

ret:
    return err;
}

static LwSciError LwSciBufValidateAttrMetaData(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    uint64_t len,
    const void* value)
{
    LwSciError err = LwSciError_Success;
    void* baseAddr = NULL;
    LwSciBufAttrStatus* status = NULL;
    uint64_t* setLen = NULL;
    uint64_t maxExpectedSize = 0UL;
    uint8_t mulStatus = OP_FAIL;
    const LwSciBufAttrKeyDescPriv* desc = NULL;

    LWSCI_FNENTRY("");

    /* value has to be checked here since the callers of this function are
     * not doing it. */
    if ((NULL == value) || (0U == len)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufValidateAttrMetaData");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Get key details */
    LwSciBufAttrGetKeyDetail(attrList, slotIndex, key, &baseAddr,
        &status, &setLen);
    LwSciBufAttrGetDesc(key, &desc);

    /* If base addr == NULL means this is wrong datatype key used to set
     *  wrong attribute key
     */
    if ((NULL == baseAddr) || (NULL == desc) || (NULL == status)) {
        LWSCI_ERR_STR("Bad key details in LwSciBufValidateAttrMetaData");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    u64Mul((uint64_t)desc->dataMaxInstance, (uint64_t)desc->dataSize,
            &maxExpectedSize, &mulStatus);
    if (OP_FAIL == mulStatus) {
        LwSciCommonPanic();
    }

    /* Check if input length is multiple of key one element size */
    if ((0UL != (len % desc->dataSize)) || (len > maxExpectedSize)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Invalid length of key data");
        LWSCI_ERR_ULONG("inputsize: ", len);
        LWSCI_ERR_ULONG("size: ", desc->dataSize);
        LWSCI_ERR_UINT("totalsize: ", desc->dataMaxInstance);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufValidateAttrVal(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    uint64_t len,
    const void* value,
    bool override,
    bool skipValidation)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufAttrKeyDescPriv* desc = NULL;
    void* baseAddr = NULL;
    LwSciBufAttrStatus* status = NULL;
    uint64_t* setLen = NULL;
    bool isWritable = false;
    LwSciBufValidateAttrFn validateFn = NULL;

    /* Get key details */
    LwSciBufAttrGetKeyDetail(attrList, slotIndex, key, &baseAddr,
        &status, &setLen);
    LwSciBufAttrGetDesc(key, &desc);

    if (false == override) {
        err = LwSciBufAttrIsWritable(attrList, key, &isWritable);
        if (LwSciError_Success != err) {
            LWSCI_ERR_INT("Unable to write to key ", (int32_t)key);
            err = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        if (false == isWritable) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_STR("Invalid operation on key");
            LWSCI_ERR_STR("isWritable: false");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (LwSciBufAttrStatus_SetLocked == *status) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_STR("Invalid operation on key");
            LWSCI_ERR_INT("accessibility: ", (int)desc->keyAccess);
            LWSCI_ERR_UINT("status: ", (uint32_t)*status);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

    }

    if (false == skipValidation) {
        if (false == override) {
            /* external user is setting attributes. Use validation function which
             * validates values allowed to be set by users of LwSciBuf.
             * Note that this validation path must also be used during importing
             * unreconciled/appended LwSciBufAttrList(s).
             */
            validateFn = desc->externalValidateFn;
        } else {
            /* LwSciBuf is setting attributes. Use validation function which
             * validates values allowed to be set by LwSciBuf.
             * Note that this validation path must also be used during importing
             * reconciled LwSciBufAttrList(s).
             */
            validateFn = desc->internalValidateFn;
        }
    }

    if (NULL != validateFn) {
        size_t numKeyValues = len / desc->dataSize;
        size_t i = 0UL;
        uint64_t tmpMul = 0UL;
        uint8_t mulStatus = OP_FAIL;

        for (i = 0; i < numKeyValues; ++i) {
            u64Mul(desc->dataSize, i, &tmpMul, &mulStatus);
            if (OP_FAIL == mulStatus) {
                LwSciCommonPanic();
            }
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            err = validateFn(attrList,
                    (const uint8_t*)value + tmpMul);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            if (LwSciError_Success != err) {
                LWSCI_ERR_HEXUINT("Attribute value validation failed for key ",
                    key);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListValidateKeyValue(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    uint64_t len,
    const void* value,
    bool override,
    bool skipValidation)
{
    LwSciError err = LwSciError_Success;

    err = LwSciBufValidateAttrMetaData(attrList, slotIndex, key, len, value);
    if (LwSciError_Success != err) {
        LWSCI_ERR_HEXUINT("LwSciBufValidateAttrMetaData failed for key ",
                          key);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufValidateAttrVal(attrList, slotIndex, key, len, value,
            override, skipValidation);
    if (LwSciError_Success != err) {
        LWSCI_ERR_HEXUINT("LwSciBufValidateAttrVal failed for key ",
                          key);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    return err;
}

/**
 * Set key attribute value in a list.
 * Note: this function must be called after LwSciBufAttrListValidateKeyValue()
 */
static void LwSciBufAttrListCommonSetAttr(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    uint64_t len,
    const void* value)
{
    const LwSciBufAttrKeyDescPriv* desc = NULL;
    void* baseAddr = NULL;
    LwSciBufAttrStatus* status = NULL;
    uint64_t* setLen = NULL;
    uint64_t maxExpectedSize = 0UL;
    uint8_t mulStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: obj %p, key: %u, len: %lu, value: %p", attrList,
        key, len, value);

    /* Get key details */
    LwSciBufAttrGetKeyDetail(attrList, slotIndex, key, &baseAddr,
        &status, &setLen);
    LwSciBufAttrGetDesc(key, &desc);

    u64Mul((uint64_t)desc->dataMaxInstance, (uint64_t)desc->dataSize,
            &maxExpectedSize, &mulStatus);
    if (OP_FAIL == mulStatus) {
        LwSciCommonPanic();
    }

    /* copy data*/
    LwSciCommonMemcpyS(baseAddr, maxExpectedSize, value, len);
    *setLen = len;
    *status = LwSciBufAttrStatus_SetLocked;

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufAttrListCommonGetAttr(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    uint64_t* len,
    void** value,
    bool override)
{
    LwSciError err = LwSciError_Success;
    void* baseAddr = NULL;
    LwSciBufAttrStatus* status = NULL;
    uint64_t* setLen = NULL;
    bool isReadable = false;

    LWSCI_FNENTRY("");

    /* print inputs */
    LWSCI_INFO("Input: slot %p, key: %u, len: %p, value: %p\n", attrList,
        key, len, value);

    err = LwSciBufAttrIsReadable(attrList, key, &isReadable);
    if (LwSciError_Success != err) {
        LWSCI_ERR_HEXUINT("Unable to read key  \n", key);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((false == override) && (false == isReadable)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Invalid operation on key");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get key details */
    LwSciBufAttrGetKeyDetail(attrList, slotIndex, key, &baseAddr, &status,
        &setLen);

    /* set output */
    if ((NULL == baseAddr) || (NULL == setLen) || (NULL == status)) {
        *value = NULL;
        *len = 0U;
    } else {
        *value = baseAddr;
        *len = *setLen;
    }

    /* print output values */
    LWSCI_INFO("Output: slot %p, key: %u, len: %lu, value: %p\n", attrList,
        key, *len, *value);

    err =  LwSciError_Success;

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCloneUmdAttr(
    LwSciBufAttrList srcAttrList,
    size_t srcSlotIndex,
    LwSciBufAttrList dstAttrList,
    size_t dstSlotIndex)
{
    LwSciError err  = LwSciError_Success;
    uint64_t len = 0U;
    LWListRec* dstNodeHeadAddr = NULL;
    LWListRec* srcNodeHeadAddr = NULL;
    const LwSciBufUmdAttrValData* iterator = NULL;
    LwSciBufUmdAttrValData* dstNode = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: srcAttrList: %p dstAttrList: %p\n", srcAttrList, dstAttrList);

    err = LwSciBufAttrListCommonGetAttr(srcAttrList, srcSlotIndex,
       (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst,
        &len, (void**)&srcNodeHeadAddr, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to to get attribute from source list\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListCommonGetAttr(dstAttrList, dstSlotIndex,
       (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst,
        &len, (void**)&dstNodeHeadAddr, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to to get attribute from destination list\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    lwListInit(dstNodeHeadAddr);

    /* copy umd attrs */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    lwListForEachEntry(iterator, srcNodeHeadAddr, listEntry) {
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

        err = LwSciBufCreatePrivateKeyNode(iterator->key, iterator->len,
                iterator->value, &dstNode);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to copy private key\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        /* change status to unlocked */
        dstNode->privateAttrStatus = LwSciBufAttrStatus_SetUnlocked;
        lwListAppend(&dstNode->listEntry, dstNodeHeadAddr);
    }

ret:
    LWSCI_FNEXIT("");
    return err;

}

static void LwSciBufAttrKeyGetCallbackDesc(
    uint32_t key,
    LwSciBufAttrKeyCallbackDesc* cbDesc)
{
    uint32_t decodedKey;
    uint32_t decodedDataType;
    uint32_t decodedKeyType;
    LwSciError err = LwSciError_Success;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 9_3), "LwSciBuf-REQ-MISRAC2012-002")
    static const LwSciBufAttrKeyCallbackDesc
        attrKeyCallbackList[LwSciBufAttrKeyType_MaxValid]
                        [LwSciBufType_MaxValid]
                        [LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE] = {
        [LW_SCI_BUF_DECODE_KEYTYPE(LwSciBufPrivateAttrKey_SciIpcRoute)]
        [LW_SCI_BUF_DECODE_DATATYPE(LwSciBufPrivateAttrKey_SciIpcRoute)]
        [LW_SCI_BUF_DECODE_ATTRKEY(LwSciBufPrivateAttrKey_SciIpcRoute)] = {
            .freeCallback = LwSciBufFreeIpcRouteCb,
            .cloneCallback = LwSciBufIpcRouteCloneCb,
        },
        [LW_SCI_BUF_DECODE_KEYTYPE(LwSciBufPrivateAttrKey_IPCTable)]
        [LW_SCI_BUF_DECODE_DATATYPE(LwSciBufPrivateAttrKey_IPCTable)]
        [LW_SCI_BUF_DECODE_ATTRKEY(LwSciBufPrivateAttrKey_IPCTable)] = {
            .freeCallback = LwSciBufFreeIpcTableCb,
            .cloneCallback = LwSciBufIpcTableCloneCb,
        },
    };
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 9_3))

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs - CallbackDesc ptr %p AttrKey: %"PRIu32"\n",
                cbDesc, key);

    err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType,
            &decodedKeyType);
    if (LwSciError_Success != err) {
       LwSciCommonPanic();
    }

    *cbDesc = attrKeyCallbackList[decodedKeyType]
                            [decodedDataType]
                            [decodedKey];

    LWSCI_FNEXIT("");
    return;

}

static inline void IncrementHelper(
    uint32_t* key,
    uint32_t* keyType,
    uint32_t* encAttrKey)
{
    LwSciError err = LwSciError_Success;

    uint32_t tmpKey = *key;
    uint32_t tmpKeyType = *keyType;
    LWSCI_FNENTRY("");

    // Check key and keyType are within allowed range.
    if (LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE <= tmpKey) {
        LWSCI_ERR_STR("key out of range for defined keys per type");
        LwSciCommonPanic();
    }

    err = LwSciBufValidateAttrKeyType(tmpKeyType);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("keyType out of range");
        LwSciCommonPanic();
    }

    tmpKey = (tmpKey + 1U) % ((uint32_t)LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE);
    if (0U == tmpKey) {
        tmpKeyType++;
    }

    // Skip cloning LwSciBufAttrKeyType_UMDPrivate key
    if ((uint32_t)LwSciBufAttrKeyType_UMDPrivate == tmpKeyType) {
        tmpKeyType++;
    }

    *encAttrKey = LW_SCI_BUF_ATTRKEY_ENCODE(tmpKeyType, LwSciBufType_General, tmpKey);
    *key = tmpKey;
    *keyType = tmpKeyType;
    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufAttrListCloneSlot(
    LwSciBufAttrList srcAttrList,
    size_t srcSlotIndex,
    LwSciBufAttrList dstAttrList,
    size_t dstSlotIndex)
{
    LwSciError err  = LwSciError_Success;
    uint8_t dataTypeNum = 0U;
    const LwSciBufAttrListObjPriv* srcAttrListObj = NULL;
    LwSciObj* srcAttrListObjParam = NULL;
    const LwSciBufAttrListObjPriv* dstAttrListObj = NULL;
    LwSciObj* dstAttrListObjParam = NULL;
    const LwSciBufPerSlotAttrList* src = NULL;
    LwSciBufPerSlotAttrList* dst = NULL;
    uint64_t len = 0U;
    uint32_t key = 0U;
    uint32_t encAttrKey = 0U;
    uint32_t keyType = 0U;
    void* srcBaseAddr  = NULL;
    void* dstBaseAddr  = NULL;
    LwSciBufAttrKeyCallbackDesc cbDesc = {0};

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: src: %p dst: %p\n", src, dst);

    /* Get object from reference */
    LwSciCommonGetObjFromRef(&srcAttrList->refHeader, &srcAttrListObjParam);
    srcAttrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(srcAttrListObjParam);

    LwSciCommonGetObjFromRef(&dstAttrList->refHeader, &dstAttrListObjParam);
    dstAttrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(dstAttrListObjParam);

    src = &srcAttrListObj->slotAttrList[srcSlotIndex];
    dst = &dstAttrListObj->slotAttrList[dstSlotIndex];

    /* copy general attributes */
    LwSciCommonMemcpyS((void*)&dst->genAttr, sizeof(dst->genAttr),
                        (const void*)&src->genAttr, sizeof(src->genAttr));

    /* copy private attributes */
    LwSciCommonMemcpyS((void*)&dst->privAttr, sizeof(dst->privAttr),
                        (const void*)&src->privAttr, sizeof(src->privAttr));

    err = LwSciBufAttrListMallocBufferType(dstAttrList, dstSlotIndex);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to allocate memory for destination dataType\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* copy dayatype attributes */
    for (dataTypeNum = 0U; dataTypeNum < (uint8_t)LwSciBufType_MaxValid;
            dataTypeNum++) {

        if (NULL != src->dataTypeAttr[dataTypeNum]) {
            LwSciCommonMemcpyS((void*)dst->dataTypeAttr[dataTypeNum],
                                dataTypeSizeMap[dataTypeNum],
                                (void*)src->dataTypeAttr[dataTypeNum],
                                dataTypeSizeMap[dataTypeNum]);
        }
    }

    keyType = (uint32_t)LwSciBufAttrKeyType_Public;
    key = 1U;
    encAttrKey = LW_SCI_BUF_ATTRKEY_ENCODE(keyType, LwSciBufType_General, key);

    while((uint32_t)LwSciBufAttrKeyType_Transport > keyType) {

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        if (!KEY_HAS_DESC_ENTRY(keyType, LwSciBufType_General, key)) {
            IncrementHelper(&key, &keyType, &encAttrKey);
            continue;
        }

        err = LwSciBufAttrListCommonGetAttr(srcAttrList, srcSlotIndex, encAttrKey,
                &len, &srcBaseAddr, true);

        if ((0U == len) || (NULL == srcBaseAddr)) {
            IncrementHelper(&key, &keyType, &encAttrKey);
            continue;
        }

        err = LwSciBufAttrListCommonGetAttr(dstAttrList, dstSlotIndex, encAttrKey,
                &len, &dstBaseAddr, true);

        LwSciBufAttrKeyGetCallbackDesc(encAttrKey, &cbDesc);

        if (NULL == cbDesc.cloneCallback) {
            IncrementHelper(&key, &keyType, &encAttrKey);
            continue;
        }

        err = cbDesc.cloneCallback(srcBaseAddr, dstBaseAddr);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to clone AttrKey \n", encAttrKey);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        IncrementHelper(&key, &keyType, &encAttrKey);
    }

    /* reset status of all keys to unlocked */
    LwSciBufAttrListResetStatus(dstAttrList, dstSlotIndex);

    err = LwSciBufAttrListCloneUmdAttr(srcAttrList, srcSlotIndex, dstAttrList,
            dstSlotIndex);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to clone Umd attr\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static void LwSciBufAttrGetKeyDetailPriv(
    LwSciBufPerSlotAttrList* slotAttrList,
    uint32_t key,
    void** baseAddr,
    LwSciBufAttrStatus** status,
    uint64_t** setLen)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufAttrKeyDescPriv* desc = NULL;
    void* startAddr = NULL;
    uint32_t decodedKey = 0U;
    uint32_t decodedDataType = 0U;
    uint32_t decodedKeyType = 0U;

    LWSCI_FNENTRY("");

    /* Sanitize output */
    *baseAddr = NULL;
    *status = NULL;
    *setLen = NULL;

    /* Decode Key */
    err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType,
            &decodedKeyType);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to decode key\n");
        LWSCI_ERR_UINT("key: \n", key);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Get descriptor entry and validate */
    desc = &attrKeyDescList[decodedKeyType][decodedDataType][decodedKey];
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if (!KEY_HAS_DESC_ENTRY(decodedKeyType, decodedDataType, decodedKey)) {
        LWSCI_ERR_STR("Invalid key descriptor entry\n");
        LWSCI_ERR_UINT("decodedKeyType: , ", decodedKeyType);
        LWSCI_ERR_UINT("decodedDataType: , ", decodedDataType);
        LWSCI_ERR_UINT("decodedKey: \n", decodedKey);
        LwSciCommonPanic();
    }

    if ((uint32_t)LwSciBufAttrKeyType_UMDPrivate == decodedKeyType) {
        startAddr = &slotAttrList->umdAttr;
    } else if ((uint32_t)LwSciBufAttrKeyType_Private == decodedKeyType) {
        startAddr = &slotAttrList->privAttr;
    } else if ((uint32_t)LwSciBufType_General == decodedDataType) {
        startAddr = (void*)&slotAttrList->genAttr;
    } else {
        startAddr = slotAttrList->dataTypeAttr[decodedDataType];
    }

    if (NULL != startAddr) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        *baseAddr = PtrAddOffset(startAddr, desc->dataOffset);
        *status = PtrAddOffset(startAddr, desc->statusOffset);
        *setLen = PtrAddOffset(startAddr, desc->sizeOffset);
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    }

    LWSCI_INFO("Output: baseAddr %p, status %d, setLen %lu\n", *baseAddr, *status, *setLen);

ret:
    LWSCI_FNEXIT("");
    return;
}

static void LwSciBufAttrListCommonGetAttrPriv(
    LwSciBufPerSlotAttrList* slotAttrList,
    uint32_t key,
    uint64_t* len,
    void** value)
{
    void* baseAddr = NULL;
    LwSciBufAttrStatus* status = NULL;
    uint64_t* setLen = NULL;

    LWSCI_FNENTRY("");

    /* print inputs */
    LWSCI_INFO("Input: slot %p, key: %u, len: %p, value: %p\n", slotAttrList,
        key, len, value);

    /* get key details */
    LwSciBufAttrGetKeyDetailPriv(slotAttrList, key, &baseAddr, &status,
        &setLen);

    *value = baseAddr;
    *len = *setLen;

    /* print output values */
    LWSCI_INFO("Output: slot %p, key: %u, len: %lu, value: %p\n", slotAttrList,
        key, *len, *value);

    LWSCI_FNEXIT("");
}

static LwSciError LwSciBufSetSlotBufferType(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    const LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount,
    bool override,
    bool* dataTypeIsSet,
    bool (*lwrrentDataTypes)[LwSciBufType_MaxValid],
    bool skipValidation)
{
    LwSciError err = LwSciError_Success;

    uint64_t pairIndex = 0U;
    *dataTypeIsSet = false;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: %p, pairArray: %p, pairCount %lu", pairArray, pairCount);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = (uint64_t)pairArray[pairIndex].len;
        const void* value = pairArray[pairIndex].value;

        const LwSciBufType* bufPtr = NULL;
        size_t i = 0U;

        if ((uint32_t)LwSciBufGeneralAttrKey_Types == key) {
            err = LwSciBufAttrListValidateKeyValue(attrList, slotIndex, key,
                                                   len, value, override,
                                                    skipValidation);
            if (LwSciError_Success != err) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
            LwSciBufAttrListCommonSetAttr(attrList, slotIndex, key, len, value);
            *dataTypeIsSet = true;

            // Memorize which data types have been requested by the caller
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            bufPtr = (const LwSciBufType*) value;
            for (i = 0; i < (len / sizeof(LwSciBufType)); i++) {
                uint32_t bufType = (uint32_t)bufPtr[i];
                (*lwrrentDataTypes)[bufType] = true;
                /*
                 * We mimic behaviour of LwSciBufAttrListMallocBufferType:
                 * if datatype is pyramid we also need to allocate memory for
                 * image
                 */
                if ((uint32_t)LwSciBufType_Pyramid == bufType) {
                    (*lwrrentDataTypes)[LwSciBufType_Image] = true;
                }
            }

            err = LwSciBufAttrListMallocBufferType(attrList, slotIndex);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("Failed to Allocate memory to Datatype");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListCommonGetUmdKey(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    uint64_t* len,
    const void** value)
{
    LwSciError err = LwSciError_Success;
    LWListRec* nodeHeadAddr = NULL;
    const LwSciBufUmdAttrValData* iterator = NULL;
    uint64_t keyLen = 0U;

    /* Search for that node containing key */
    LWSCI_INFO("Searching for %d\n", key);

    err = LwSciBufAttrListCommonGetAttr(attrList, slotIndex,
            (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst,
            &keyLen, (void**)&nodeHeadAddr, true);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    lwListForEachEntry(iterator, nodeHeadAddr, listEntry) {
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        LWSCI_INFO("\tLooping %d\n", iterator->key);
        /** we have found the key lets exit */
        if (key == iterator->key) {
            LWSCI_INFO("\t\tKey Found\n");
            *len = iterator->len;
            *value = iterator->value;
            err =  LwSciError_Success;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    LWSCI_INFO("\tNot found\n");
    /* Key was not found return error */
    *len = 0U;
    *value = NULL;

ret:
    return err;
}

static LwSciError LwSciBufAttrListCommonSetUmdKey(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    uint64_t len,
    const void* value)
{
    LwSciError err = LwSciError_Success;
    LWListRec* nodeHeadAddr = NULL;
    LwSciBufUmdAttrValData* privateKeyNode = NULL;
    LwSciBufUmdAttrValData* iterator = NULL;
    LwSciBufUmdAttrValData* iteratorTemp = NULL;
    uint64_t keyLen = 0U;

    err = LwSciBufCreatePrivateKeyNode(key, len, value, &privateKeyNode);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Cannot create Umd key node\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Setting key %d\n", key);

    err = LwSciBufAttrListCommonGetAttr(attrList, slotIndex,
            (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst,
            &keyLen, (void**)&nodeHeadAddr, true);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    lwListForEachEntry_safe(iterator, iteratorTemp, nodeHeadAddr, listEntry) {
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        /** we have found the key lets exit */
        LWSCI_INFO("\tLooping key %d\n", iterator->key);
        if (key == iterator->key) {

            LWSCI_INFO("\t\tFound key %d\n", key);

            if (LwSciBufAttrStatus_SetUnlocked == iterator->privateAttrStatus) {

                LWSCI_INFO("\t\tKey is unlocked %d\n", key);
                lwListDel(&iterator->listEntry);
                LwSciCommonFree(iterator->value);
                LwSciCommonFree(iterator);
                break;

            } else {

                LWSCI_INFO("\t\tKey is locked %d\n", key);
                err = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_privatekeynode;

            }
        } else if (iterator->key < key) {

            LWSCI_INFO("\t\tadding after key %d\n", iterator->key);
            lwListAppend(&privateKeyNode->listEntry, &iterator->listEntry);
            err =  LwSciError_Success;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;

        } else {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
            lwListForEachEntryEnd_safe(iterator, iteratorTemp, listEntry);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        }
    }

    LWSCI_INFO("\tadding end\n");
    lwListAppend(&privateKeyNode->listEntry, nodeHeadAddr);

    err =  LwSciError_Success;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_privatekeynode:
    LwSciCommonFree(privateKeyNode->value);
    LwSciCommonFree(privateKeyNode);
ret:
    return err;
}

static LwSciError LwSciBufAttrListInit(
    LwSciBufModule module,
    uint64_t slotCount,
    LwSciBufAttrList attrList,
    bool emptyIpcRoute)
{
    LwSciError err = LwSciError_Success;
    LwSciObj* attrListObjParam = NULL;
    LwSciBufAttrListObjPriv* attrListObj = NULL;
    uint64_t i = 0U;
    uint64_t len = 0U;
    void* nodeHeadAddr = NULL;

    LWSCI_FNENTRY("");

    /* Get Object from reference */
    LwSciCommonGetObjFromRef(&(attrList->refHeader), &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    attrListObj->slotCount = slotCount;
    attrListObj->magic = LW_SCI_BUF_ATTR_MAGIC;

    err = LwSciBufModuleDupRef(module, &attrListObj->module);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to duplicate module reference\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Allocate memory for all slots */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    attrListObj->slotAttrList = LwSciCommonCalloc(slotCount,
                                sizeof(LwSciBufPerSlotAttrList));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if (NULL == attrListObj->slotAttrList) {
        err = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Failed to allocate memory for LwSciBufPerSlotAttrList\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    for (i = 0; i < slotCount; i++) {
        err = LwSciBufAttrListCommonGetAttr(attrList, i,
                (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst,
                &len, (void**)&nodeHeadAddr, true);
        if (LwSciError_Success != err) {
            LwSciCommonPanic();
        }
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP37_C), "LwSciBuf-ADV-CERTC-001")
        lwListInit(nodeHeadAddr);
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

        if (true == emptyIpcRoute) {
            const LwSciBufIpcRoute* tmpIpcRoute = NULL;
            LwSciBufPrivateAttrKeyValuePair keyValPair = {0};
            (void)memset(&keyValPair, 0x0, sizeof(keyValPair));

            /* For each slot, create the empty IPC route */
            keyValPair.key = LwSciBufPrivateAttrKey_SciIpcRoute;
            keyValPair.value = &tmpIpcRoute;
            keyValPair.len = sizeof(tmpIpcRoute);

            err = LwSciBufAttrListCommonSetAttrs(attrList, i, &keyValPair, 1,
                    LwSciBufAttrKeyType_Private, true, false);
            if (LwSciError_Success != err) {
                /* This should not happen */
                LwSciCommonPanic();
            }
        }
    }

    /* print outgoing paramters */
    LWSCI_INFO("Input: slotCount:%lu, module: %p, newAttrListObj: %p\n",
            attrListObj->slotCount, attrListObj->module, attrListObj);

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListSlotGetInternalAttrsPriv(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    LwSciBufInternalAttrKeyValuePair* pairArray,
    size_t pairCount,
    bool override,
    bool acquireLock)
{
    LwSciError err = LwSciError_Success;
    uint64_t pairIndex = 0U;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p, slotIndex: %lu, pairArray: %p, "
        "pairCount %lu\n", attrList, slotIndex, pairArray, pairCount);

    if ((NULL == pairArray) || (0U == pairCount)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListSlotGetInternalAttrs\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListSlotGetInternalAttrs\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader,
                &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    if (slotIndex >= attrListObj->slotCount) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Slot count provided is than actual slot count.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == acquireLock) {
        LwSciCommonObjLock(&attrList->refHeader);
    }

    for (pairIndex = 0; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = 0U;
        const void* value = NULL;

        LwSciError errIsInternal = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_Internal);
        LwSciError errIsUmd = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_UMDPrivate);

        if ((LwSciError_Success != errIsUmd) && (LwSciError_Success != errIsInternal) ) {
            LWSCI_ERR_STR("Invalid Attr Key");
            LWSCI_ERR_UINT("Key is not a valid Internal/UMD key: \n", key);

            err = LwSciError_BadParameter;
        } else if (LwSciError_Success == errIsInternal) {
            void* tmpValue = NULL;
            err = LwSciBufAttrListCommonGetAttr(attrList, slotIndex, key, &len,
                    &tmpValue, override);
            value = tmpValue;
        } else {
            err = LwSciBufAttrListCommonGetUmdKey(attrList, slotIndex, key, &len,
                    &value);
        }

        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }

        pairArray[pairIndex].len = (size_t)len;
        pairArray[pairIndex].value = value;
    }

set_failure:
    if (true == acquireLock) {
        LwSciCommonObjUnlock(&attrList->refHeader);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListSlotSetPrivateAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    const LwSciBufPrivateAttrKeyValuePair* pairArray,
    size_t pairCount,
    bool override,
    bool skipValidation)
{
    LwSciError err = LwSciError_Success;

    uint64_t pairIndex = 0U;

    // Track assignments for private keys
    uint32_t* keysCopy = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p, slotIndex: %lu, pairArray: %p, "
        "pairCount %lu\n", attrList, slotIndex, pairArray, pairCount);

    LwSciCommonObjLock(&attrList->refHeader);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = (uint64_t)pairArray[pairIndex].len;
        const void* value = pairArray[pairIndex].value;

        err = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_Private);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Invalid Attr Key");
            LWSCI_ERR_HEXUINT("Key is not valid private key:", key);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }

        err = LwSciBufAttrListValidateKeyValue(attrList, slotIndex, key, len,
                                               value, override, skipValidation);
        if (LwSciError_Success != err) {
            LWSCI_ERR_HEXUINT("LwSciBufAttrListValidateKeyValue failed for key: \n", key);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }
    }

    // Check for duplicates in the array
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    keysCopy = (uint32_t*)LwSciCommonCalloc(pairCount, sizeof(uint32_t));
    if (NULL == keysCopy) {
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto set_failure;
    }

    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        keysCopy[pairIndex] = (uint32_t)pairArray[pairIndex].key;
    }
    LwSciCommonSort(keysCopy, pairCount, sizeof(uint32_t), LwSciBufAttrKeyCompare);
    for (pairIndex = 0U; pairIndex < (pairCount - 1U); pairIndex++) {
        uint64_t pairIndexNext = pairIndex + 1U;

        if (keysCopy[pairIndex] == keysCopy[pairIndexNext]) {
            err = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_keysCopy;
        }
    }

    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = (uint64_t)pairArray[pairIndex].len;
        const void* value = pairArray[pairIndex].value;

        LwSciBufAttrListCommonSetAttr(attrList, slotIndex, key, len, value);
    }

free_keysCopy:
    LwSciCommonFree(keysCopy);

set_failure:
    LwSciCommonObjUnlock(&attrList->refHeader);

    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListSlotGetAttrsPriv(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount,
    bool override,
    bool acquireLock)
{
    LwSciError err = LwSciError_Success;
    uint64_t pairIndex = 0U;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p, slotIndex: %lu, pairArray: %p, "
        "pairCount %lu\n", attrList, slotIndex, pairArray, pairCount);

    if ((NULL == pairArray) || (0U == pairCount)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListSlotGetAttrs\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListSlotGetAttrs\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    if (slotIndex >= attrListObj->slotCount) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Slot count provided is than actual slot count.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == acquireLock) {
        LwSciCommonObjLock(&attrList->refHeader);
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = 0U;
        void* value = NULL;

        err = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_Public);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Invalid Attr Key\n");
            LWSCI_ERR_UINT("Key is not valid key: \n", key);

            err = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }

        err = LwSciBufAttrListCommonGetAttr(attrList, slotIndex, key, &len,
                &value, override);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }
        pairArray[pairIndex].len = (size_t)len;
        pairArray[pairIndex].value = value;
    }

set_failure:
    if (true == acquireLock) {
        LwSciCommonObjUnlock(&attrList->refHeader);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListSlotGetPrivateAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    LwSciBufPrivateAttrKeyValuePair* pairArray,
    size_t pairCount,
    bool override,
    bool acquireLock)
{
    LwSciError err = LwSciError_Success;
    uint64_t pairIndex = 0U;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p, slotIndex: %lu, pairArray: %p, "
        "pairCount %lu\n", attrList, slotIndex, pairArray, pairCount);

    if ((NULL == pairArray) || (0U == pairCount)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListSlotGetPrivateAttrs\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListSlotGetPrivateAttrs\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    if (slotIndex >= attrListObj->slotCount) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Slot count provided is than actual slot count.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == acquireLock) {
        LwSciCommonObjLock(&attrList->refHeader);
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = 0U;
        void* value = NULL;

        err = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_Private);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Invalid Attr Key");
            LWSCI_ERR_UINT("Key is not valid: ", key);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }

        err = LwSciBufAttrListCommonGetAttr(attrList, slotIndex, key, &len,
                &value, override);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }

        pairArray[pairIndex].len = (size_t)len;
        pairArray[pairIndex].value = value;
    }

set_failure:
    if (true == acquireLock) {
        LwSciCommonObjUnlock(&attrList->refHeader);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static void LwSciBufAttrListSlotDataTypeFree(
    const LwSciBufAttrListObjPriv* attrListObj,
    size_t slotIndex,
    const bool* dataTypeSetPtr)
{
    for (uint8_t dataType = 0U; dataType < (uint8_t)LwSciBufType_MaxValid; dataType++) {
        bool dataTypeIsFree;
        dataTypeIsFree = (false == dataTypeSetPtr[dataType]);
        dataTypeIsFree = dataTypeIsFree &&
            (attrListObj->slotAttrList[slotIndex].dataTypeAttr[dataType] != NULL);
        /* Free buffer data types */
        if (true == dataTypeIsFree) {
            LwSciCommonFree(attrListObj->slotAttrList[slotIndex].dataTypeAttr[dataType]);
            attrListObj->slotAttrList[slotIndex].dataTypeAttr[dataType] = NULL;
        }
    }
}

/**
 * @bried Private interface definition
 */

static LwSciError LwSciBufAttrListSlotSetAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    const LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount,
    bool override,
    bool skipValidation)
{
    LwSciError err;

    uint64_t pairIndex = 0U;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;
    uint8_t dataType = 0U;
    // Current datatypes to be set in LwSciBufSetSlotBufferType
    bool lwrrentDataTypes[LwSciBufType_MaxValid] = {false};
    // Previously set data types
    bool prevDataTypes[LwSciBufType_MaxValid] = {false};
    bool dataTypeIsSet = false;
    void* bufTypeBaseAddrPtr = NULL;
    LwSciBufAttrStatus* bufTypeStatusPtr = NULL;
    uint64_t* bufTypeSetLenPtr = NULL;
    LwSciBufAttrStatus bufTypeStatus = LwSciBufAttrStatus_Empty;
    uint64_t bufTypeSetLen = 0U;
    LwSciBufType bufTypeValue[LwSciBufType_MaxValid] = { };

    // Track assignments for public keys
    uint32_t* keysCopy = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p, slotIndex: %lu, pairArray: %p, "
               "pairCount %lu\n",
               attrList, slotIndex, pairArray, pairCount);

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    LwSciCommonObjLock(&attrList->refHeader);

    /*
     * Cache status, setLen and value for LwSciBufGeneralAttrKey_Types to
     * restore later if whole operation fails
     */
    LwSciBufAttrGetKeyDetail(attrList, slotIndex,
        (uint32_t)LwSciBufGeneralAttrKey_Types, &bufTypeBaseAddrPtr,
        &bufTypeStatusPtr, &bufTypeSetLenPtr);
    bufTypeStatus = *bufTypeStatusPtr;
    bufTypeSetLen = *bufTypeSetLenPtr;
    if (sizeof(bufTypeValue) < bufTypeSetLen) {
        LWSCI_ERR_STR("Too many buffer types were already set\n");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(bufTypeValue, sizeof(bufTypeValue), bufTypeBaseAddrPtr,
                       bufTypeSetLen);

    /* Memorize previously allocated buffer data types */
    for (dataType = 0U; dataType < (uint8_t)LwSciBufType_MaxValid; dataType++) {
        prevDataTypes[dataType] =
            (attrListObj->slotAttrList[slotIndex].dataTypeAttr[dataType] !=
             NULL);
    }

    err =
        LwSciBufSetSlotBufferType(attrList, slotIndex, pairArray, pairCount,
                                  override, &dataTypeIsSet, &lwrrentDataTypes,
                                    skipValidation);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set Buffer type\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attrobj;
    }

    /* Validate attributes before list modification */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = (uint64_t)pairArray[pairIndex].len;
        const void* value = pairArray[pairIndex].value;

        err = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_Public);
        if (LwSciError_Success != err) {
            LWSCI_ERR_HEXUINT("Key is not valid public key: ", key);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto unlock_attrobj;
        }

        if ((uint32_t)LwSciBufGeneralAttrKey_Types == key) {
            /*
             * since we have already set this key in LwSciBufSetSlotBufferType
             * - skip it
             */
            continue;
        }

        err = LwSciBufAttrListValidateKeyValue(attrList, slotIndex, key, len,
                                               value, override, skipValidation);

        if (LwSciError_Success != err) {
            LWSCI_ERR_HEXUINT(
                "LwSciBufAttrListValidateKeyValue failed for key ", key);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto unlock_attrobj;
        }
    }
    // Check for duplicates in the array
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    keysCopy = (uint32_t*)LwSciCommonCalloc(pairCount, sizeof(uint32_t));
    if (NULL == keysCopy) {
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attrobj;
    }

    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        keysCopy[pairIndex] = (uint32_t)pairArray[pairIndex].key;
    }
    LwSciCommonSort(keysCopy, pairCount, sizeof(uint32_t), LwSciBufAttrKeyCompare);
    for (pairIndex = 0U; pairIndex < (pairCount - 1U); pairIndex++) {
        uint64_t pairIndexNext = pairIndex + 1U;

        if (keysCopy[pairIndex] == keysCopy[pairIndexNext]) {
            err = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_keysCopy;
        }
    }

    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = (uint64_t)pairArray[pairIndex].len;
        const void* value = pairArray[pairIndex].value;

        if ((uint32_t)LwSciBufGeneralAttrKey_Types == key) {
            /*
             * since we have already set this key in LwSciBufSetSlotBufferType
             *  - skip it
             */
            continue;
        }

        LwSciBufAttrListCommonSetAttr(attrList, slotIndex, key, len, value);
    }

    err = LwSciError_Success;

free_keysCopy:
    LwSciCommonFree(keysCopy);

unlock_attrobj:
    if ((LwSciError_Success != err) && (true == dataTypeIsSet)) {
        /*
         * Revert LwSciBufGeneralAttrKey_Types newly allocated buffer data
         * types, setLen, status and value
         */
        void* baseAddr = NULL;
        LwSciBufAttrStatus* status = NULL;
        uint64_t* setLen = NULL;
        LwSciBufAttrGetKeyDetail(attrList, slotIndex,
                (uint32_t)LwSciBufGeneralAttrKey_Types, &baseAddr, &status, &setLen);
        *setLen = bufTypeSetLen;
        *status = bufTypeStatus;
        LwSciCommonMemcpyS(bufTypeBaseAddrPtr, bufTypeSetLen, bufTypeValue,
                           bufTypeSetLen);
        /* Free only newly allocated buffer data types */
        LwSciBufAttrListSlotDataTypeFree(attrListObj, slotIndex, prevDataTypes);
    } else if (true == dataTypeIsSet) {
        /*
         * Free unused data types, this should happen only in the case
         * of a cloned attribute list, when LwSciBufGeneralAttrKey_Types
         * is unlocked and the caller wants to set different types.
         */
        LwSciBufAttrListSlotDataTypeFree(attrListObj, slotIndex, lwrrentDataTypes);
    } else {
        // Empty else clause to satisfy MISRA 15.7 rule
    }

    LwSciCommonObjUnlock(&attrList->refHeader);

    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListSlotSetInternalAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    const LwSciBufInternalAttrKeyValuePair* pairArray,
    size_t pairCount,
    bool override,
    bool skipValidation)
{
    LwSciError err = LwSciError_Success;

    uint64_t pairIndex = 0U;

    // Track assignments for internal keys
    uint32_t* keysCopy = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p, slotIndex: %lu, pairArray: %p, "
        "pairCount %lu\n", attrList, slotIndex, pairArray, pairCount);

    LwSciCommonObjLock(&attrList->refHeader);

    // Validate attribute key value pairs first
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = (uint64_t)pairArray[pairIndex].len;
        const void* value = pairArray[pairIndex].value;

        LwSciError errIsInternal = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_Internal);
        LwSciError errIsUmd = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_UMDPrivate);

        if ((LwSciError_Success != errIsUmd) && (LwSciError_Success != errIsInternal)) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_HEXUINT("Key is not valid key: ", key);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto set_failure;
        }

        if (LwSciError_Success == errIsInternal) {
            err = LwSciBufAttrListValidateKeyValue(attrList, slotIndex, key,
                                                   len, value, override,
                                                    skipValidation);
            if (LwSciError_Success != err) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto set_failure;
            }
        }
    }

    // Check for duplicates in the array
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    keysCopy = (uint32_t*)LwSciCommonCalloc(pairCount, sizeof(uint32_t));
    if (NULL == keysCopy) {
        err = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto set_failure;
    }

    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        keysCopy[pairIndex] = (uint32_t)pairArray[pairIndex].key;
    }
    LwSciCommonSort(keysCopy, pairCount, sizeof(uint32_t), LwSciBufAttrKeyCompare);
    for (pairIndex = 0U; pairIndex < (pairCount - 1U); pairIndex++) {
        uint64_t pairIndexNext = pairIndex + 1U;

        if (keysCopy[pairIndex] == keysCopy[pairIndexNext]) {
            err = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_keysCopy;
        }
    }

    for (pairIndex = 0U; pairIndex < pairCount; pairIndex++) {
        uint32_t key = (uint32_t)pairArray[pairIndex].key;
        uint64_t len = (uint64_t)pairArray[pairIndex].len;
        const void* value = pairArray[pairIndex].value;

        LwSciError errIsInternal = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_Internal);
        LwSciError errIsUmd = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_UMDPrivate);

        if ((LwSciError_Success != errIsUmd) && (LwSciError_Success != errIsInternal)) {
            /* We just validated that these are actual Internal/UMD keys. */
            LwSciCommonPanic();
        } else if (LwSciError_Success == errIsInternal) {
            LwSciBufAttrListCommonSetAttr(attrList, slotIndex, key, len, value);
        } else {
            err = LwSciBufAttrListCommonSetUmdKey(attrList, slotIndex, key, len,
                    value);
        }

        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_keysCopy;
        }
    }

free_keysCopy:
    LwSciCommonFree(keysCopy);

set_failure:
    LwSciCommonObjUnlock(&attrList->refHeader);

    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufAttrListSlotGetInternalAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    LwSciBufInternalAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");

    err = LwSciBufAttrListSlotGetInternalAttrsPriv(attrList, slotIndex,
            pairArray, pairCount, false, true);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static bool ContainsBufferType(
    const LwSciBufType *arr,
    size_t len,
    LwSciBufType bufType)
{
    bool match = false;
    size_t i = 0U;

    for (i = 0; i < len; ++i) {
        if (arr[i] == bufType) {
            match = true;
            break;
        }
    }

    return match;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
int32_t LwSciBufAttrListCompare(
    const void* elem1,
    const void* elem2)
{
    int32_t ret = 0;
    LwSciError err = LwSciError_Success;

    uint64_t attrListAddr1 = 0U;
    uint64_t attrListAddr2 = 0U;

    LwSciBufAttrList attrList1 = NULL;
    LwSciBufAttrList attrList2 = NULL;

    LWSCI_FNENTRY("");

    if ((NULL == elem1) || (NULL == elem2)) {
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    attrList1 = *(const LwSciBufAttrList*)elem1;
    attrList2 = *(const LwSciBufAttrList*)elem2;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    err = LwSciBufAttrListValidate(attrList1);
    if (LwSciError_Success != err) {
        LwSciCommonPanic();
    }
    err = LwSciBufAttrListValidate(attrList2);
    if (LwSciError_Success != err) {
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_4), "LwSciBuf-ADV-MISRAC2012-013")
    attrListAddr1 = (uint64_t)attrList1;
    attrListAddr2 = (uint64_t)attrList2;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_4))

    if (attrListAddr1 > attrListAddr2) {
        ret = 1;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (attrListAddr1 < attrListAddr2) {
        ret = -1;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return ret;
}

LwSciError LwSciBufAttrListSetState(
    LwSciBufAttrList attrList,
    LwSciBufAttrListState state)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    if (LwSciBufAttrListState_UpperBound <= state) {
        LWSCI_ERR_INT(" Ilwaild BufAttrList state: %d\n", (int32_t)state);
        LwSciCommonPanic();
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate LwSciBufAttrList reference\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    attrListObj->state = state;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
void LwSciBufAttrCleanupCallback(
    LwSciObj* attrListPtr)
{
    uint64_t i = 0U;
    uint64_t len = 0U;
    uint8_t types = 0U;
    uint32_t key = 0;
    bool keyTypeEnd = false;
    bool keyEnd = false;
    bool dataTypeEnd = false;
    void* baseAddr = NULL;
    uint64_t* setLen = NULL;
    LwSciBufPerSlotAttrList* slotAttrList = NULL;
    LWListRec* nodeHeadAddr = NULL;
    LwSciBufUmdAttrValData* iterator,* tempIter;
    LwSciBufAttrKeyIterator iter;
    LwSciBufAttrStatus* status = NULL;
    LwSciBufAttrKeyCallbackDesc cbDesc = {0};
    const LwSciBufAttrListObjPriv* attrListObj = NULL;

    LWSCI_FNENTRY("");

    /* Print input */
    LWSCI_INFO("Input: attrListPtr: %p\n", attrListPtr);

    if (NULL == attrListPtr) {
        LwSciCommonPanic();
    }

    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListPtr);

    /* validate Magic ID */
    if (LW_SCI_BUF_ATTR_MAGIC != attrListObj->magic) {
        LWSCI_ERR_STR("Failed to validate attribute list");
        LWSCI_ERR_HEXUINT("magic: 0x ", attrListObj->magic);
        LwSciCommonPanic();
    }

    /* Close Module */
    LwSciBufModuleClose(attrListObj->module);

    /* Check for per slot Attr List */
    if (NULL == attrListObj->slotAttrList) {
        LWSCI_ERR_STR("Trying to free NULL per slot attr list\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Do free for all slots */
    for (i = 0U; i < attrListObj->slotCount; i++) {
        slotAttrList = &attrListObj->slotAttrList[i];

        /* Get UMD link list head*/
        LwSciBufAttrListCommonGetAttrPriv(slotAttrList,
                (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst,
                &len, (void**)&nodeHeadAddr);

        /* Free UMD Link List */
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        lwListForEachEntry_safe(iterator, tempIter, nodeHeadAddr, listEntry) {
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            lwListDel(&iterator->listEntry);
            LwSciCommonFree(iterator->value);
            LwSciCommonFree(iterator);

            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
            lwListForEachEntryEnd_safe(iterator, tempIter, listEntry);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        }

        /*
         * This is an optimization here.
         * Today, we know the cleanup callback is required only for private
         * general attrs. Hence, running the iterator only for those attributes.
         * Modify when cleanup callback is required for other attributes also.
         */
        LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Private,
            (uint32_t)LwSciBufType_General, 0U, &iter);

        for ( ; ; ) {
            LwSciBufAttrKeyIterNext(&iter, &keyTypeEnd, &dataTypeEnd,
                &keyEnd, &key);

            if (true == keyEnd) {
                break;
            }

            LwSciBufAttrGetKeyDetailPriv(slotAttrList, key, &baseAddr,
                                     &status, &setLen);
            if ((NULL == setLen) || (0U == *setLen) || (NULL == baseAddr)) {
                continue;
            }

            LwSciBufAttrKeyGetCallbackDesc(key, &cbDesc);

            if (NULL != cbDesc.freeCallback) {
                cbDesc.freeCallback(baseAddr);
            }
        }

        /* For all Data types free */
        for (types = 0U;
            types < (uint8_t)LwSciBufType_MaxValid; types++) {
            LwSciCommonFree(slotAttrList->dataTypeAttr[types]);
        }
    }

    LwSciCommonFree(attrListObj->slotAttrList);

ret:
    LWSCI_FNEXIT("");
    return;
}

LwSciError LwSciBufAttrListsLock(
    const LwSciBufAttrList inputAttrListArr[],
    size_t attrListCount)
{
    LwSciError error = LwSciError_Success;

    LwSciBufAttrList* inputAttrListArrCopy = NULL;
    size_t lwrrAttrList = 0U;
    size_t attrListArrSize = 0U;
    uint8_t overflow = OP_FAIL;

    LWSCI_FNENTRY("");

    if ((NULL == inputAttrListArr) || (0U == attrListCount)) {
        LWSCI_ERR_STR("Invalid input parameters\n");
        LwSciCommonPanic();
    }

    error = LwSciBufValidateAttrListArray(inputAttrListArr, attrListCount);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to validate Attribute List array\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    sizeMul(sizeof(LwSciBufAttrList), attrListCount, &attrListArrSize, &overflow);
    if (OP_FAIL == overflow) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    inputAttrListArrCopy = (LwSciBufAttrList*)LwSciCommonCalloc(
            attrListCount, sizeof(LwSciBufAttrList));
    if (NULL == inputAttrListArrCopy) {
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    LwSciCommonMemcpyS(inputAttrListArrCopy, attrListArrSize,
                        inputAttrListArr, attrListArrSize);

    /*
     * Sort the Attribute Lists to enforce a Resource Ordering, so we don't run
     * into a situation where we can deadlock.
     */
    LwSciCommonSort(inputAttrListArrCopy, attrListCount,
            sizeof(LwSciBufAttrList), LwSciBufAttrListCompare);

    for (lwrrAttrList = 0U; lwrrAttrList < attrListCount; ++lwrrAttrList) {
        LwSciBufAttrList attrList = inputAttrListArrCopy[lwrrAttrList];

        LwSciCommonObjLock(&attrList->refHeader);
    }

    LwSciCommonFree(inputAttrListArrCopy);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciBufAttrListsUnlock(
    const LwSciBufAttrList inputAttrListArr[],
    size_t attrListCount)
{
    LwSciError error = LwSciError_Success;

    size_t lwrrAttrList = 0U;

    LWSCI_FNENTRY("");

    if ((NULL == inputAttrListArr) || (0U == attrListCount)) {
        LWSCI_ERR_STR("Invalid input parameters\n");
        LwSciCommonPanic();
    }

    error = LwSciBufValidateAttrListArray(inputAttrListArr, attrListCount);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to validate Attribute List array\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    for (lwrrAttrList = 0U; lwrrAttrList < attrListCount; ++lwrrAttrList) {
        LwSciBufAttrList attrList = inputAttrListArr[lwrrAttrList];
        LwSciCommonObjUnlock(&attrList->refHeader);
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciBufAttrKeyDecode(
    uint32_t key,
    uint32_t* decodedKey,
    uint32_t* decodedDataType,
    uint32_t* decodedKeyType)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* Check for input */
    if ((NULL == decodedKey) || (NULL == decodedDataType) ||
        (NULL == decodedKeyType)) {
        LWSCI_ERR_STR("Bad input arguments provided");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Print input */
    LWSCI_INFO("Input: key: %u, decodedKey: %p, decodedDataType: %p, decodedKeyType: %p",
        key, decodedKey, decodedDataType, decodedKeyType);

    err = LwSciBufAttrKeyValidate(key);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Decode keys */
    *decodedKey = LW_SCI_BUF_DECODE_ATTRKEY(key);
    *decodedDataType = LW_SCI_BUF_DECODE_DATATYPE(key);
    *decodedKeyType = LW_SCI_BUF_DECODE_KEYTYPE(key);

    /* Print output */
    LWSCI_INFO("Output: decodedKey: %u, decodedDataType: %u, decodedKeyType: %u",
        *decodedKey, *decodedDataType, *decodedKeyType);

ret:
    LWSCI_FNEXIT("");
    return err;
}

void LwSciBufAttrGetDataDetails(
    uint32_t key,
    size_t* dataSize,
    uint32_t* dataMaxInstance)
{
    LwSciError err = LwSciError_Success;

    const LwSciBufAttrKeyDescPriv* desc = NULL;
    LWSCI_FNENTRY("");

    err = LwSciBufAttrKeyValidate(key);
    if ((LwSciError_Success != err) || (NULL == dataSize) || (NULL == dataMaxInstance)) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LWSCI_ERR_STR("Invalid arguments");
        LwSciCommonPanic();
    }
    if (((uint32_t)LwSciBufAttrKeyType_UMDPrivate == LW_SCI_BUF_DECODE_KEYTYPE(key)) &&
            ((uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst != key)) {
        /* If this is an UMD key, then we need to also panic for attribute keys
         * other than LwSciBufInternalAttrKey_LwMediaPrivateFirst, since they
         * have no attribute key descriptor */
        LwSciCommonPanic();
    }

    LwSciBufAttrGetDesc(key, &desc);

    *dataSize = desc->dataSize;

    *dataMaxInstance = desc->dataMaxInstance;

    LWSCI_FNEXIT("");
    return;
}

void LwSciBufAttrGetKeyAccessDetails(
     uint32_t key,
     LwSciBufKeyAccess* keyAccess)
{
    LwSciError err = LwSciError_Success;

    const LwSciBufAttrKeyDescPriv* desc = NULL;
    LWSCI_FNENTRY("");

    err = LwSciBufAttrKeyValidate(key);
    if ((LwSciError_Success != err) || (NULL == keyAccess)) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LWSCI_ERR_STR("Invalid argument");
        LwSciCommonPanic();
    }
    if (((uint32_t)LwSciBufAttrKeyType_UMDPrivate == LW_SCI_BUF_DECODE_KEYTYPE(key)) &&
            ((uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst != key)) {
        /* If this is an UMD key, then we need to also panic for attribute keys
         * other than LwSciBufInternalAttrKey_LwMediaPrivateFirst, since they
         * have no attribute key descriptor */
        LwSciCommonPanic();
    }

     LwSciBufAttrGetDesc(key, &desc);

     *keyAccess = desc->keyAccess;

     LWSCI_FNEXIT("");
     return;
}

void LwSciBufAttrGetKeyDetail(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    void** baseAddr,
    LwSciBufAttrStatus** status,
    uint64_t** setLen)
{
    LwSciError err = LwSciError_Success;

    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;
    LwSciBufPerSlotAttrList* slotAttrList = NULL;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrKeyValidate(key);
    if ((LwSciError_Success != err) || (NULL == baseAddr) || (NULL == status) || (NULL == setLen)) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LWSCI_ERR_STR("Invalid arguments");
        LwSciCommonPanic();
    }
    if (((uint32_t)LwSciBufAttrKeyType_UMDPrivate == LW_SCI_BUF_DECODE_KEYTYPE(key)) &&
            ((uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst != key)) {
        /* If this is an UMD key, then we need to also panic for attribute keys
         * other than LwSciBufInternalAttrKey_LwMediaPrivateFirst, since they
         * have no attribute key descriptor */
        LwSciCommonPanic();
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate LwSciBufAttrList");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    if (slotIndex >= attrListObj->slotCount) {
        /* We can't panic here since this is potentially directly deserialized
         * without any validation. Any caller must check setLen. */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    slotAttrList = &attrListObj->slotAttrList[slotIndex];

    LwSciBufAttrGetKeyDetailPriv(slotAttrList, key, baseAddr, status, setLen);

ret:
    LWSCI_FNEXIT("");
    return;
}

LwSciError LwSciBufAttrListMallocBufferType(
    LwSciBufAttrList attrList,
    size_t slotIndex)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;
    LwSciBufPerSlotAttrList* slotAttrList = NULL;
    uint32_t key = 0U;
    uint64_t len = 0U, count = 0U, index = 0U;
    LwSciBufType bufType, *bufPtr = NULL;
    uint32_t tmp = 0U;

    LWSCI_FNENTRY("");

    if (NULL == attrList) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter to function\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate LwSciBufAttrList reference\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);


    if (slotIndex >= attrListObj->slotCount) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LwSciCommonPanic();
    }

    slotAttrList = &attrListObj->slotAttrList[slotIndex];

    key = (uint32_t)LwSciBufGeneralAttrKey_Types;
    err = LwSciBufAttrListCommonGetAttr(attrList, slotIndex, key, &len,
            (void**)&bufPtr, true);
    if (LwSciError_Success != err) {
        LwSciCommonPanic();
    }

    if ((NULL == bufPtr) || (0U == len)) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttr failed for LwSciBufGeneralAttrKey_Types\n");
        LWSCI_ERR_ULONG("len \n", len);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    count = (len / sizeof(LwSciBufType));

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (index = 0; index < count; index++) {
        bufType = bufPtr[index];
        tmp = (uint32_t)bufType;

        err = LwSciBufValidateLwSciBufType((uint32_t)bufType);
        if ((LwSciError_Success != err) || (LwSciBufType_General == bufType)) {
            LWSCI_ERR_STR("Invalid LwSciBufType");
            LWSCI_ERR_ULONG("LwSciBufType at index: ", index);
            LWSCI_ERR_UINT(" is: \n", (uint32_t)bufType);
            err = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (NULL == slotAttrList->dataTypeAttr[tmp]) {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            slotAttrList->dataTypeAttr[tmp] =
                LwSciCommonCalloc(1, dataTypeSizeMap[tmp]);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            if (NULL == slotAttrList->dataTypeAttr[tmp]) {
                LWSCI_ERR_UINT(
                    "Failed to allocate memory for attributes of LwSciBufType:\n",
                    bufType);
                err = LwSciError_InsufficientMemory;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }

        /* if datatype is pyramid we also need to allocate memory for image
         *  datatype
         */
        if ((LwSciBufType_Pyramid == bufType) &&
            (NULL == slotAttrList->dataTypeAttr[LwSciBufType_Image])) {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            slotAttrList->dataTypeAttr[LwSciBufType_Image] =
                LwSciCommonCalloc(1, dataTypeSizeMap[LwSciBufType_Image]);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            if (NULL == slotAttrList->dataTypeAttr[LwSciBufType_Image]) {
                LWSCI_ERR_STR("Failed to allocate memory for attributes of image LwSciBufType\n");
                err = LwSciError_InsufficientMemory;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufCreatePrivateKeyNode(
    uint32_t key,
    uint64_t len,
    const void* value,
    LwSciBufUmdAttrValData** privateKeyNode)

{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((0U == len) || (NULL == value) || (NULL == privateKeyNode))
    {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Invalid input parameters\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    err = LwSciBufAttrKeyIsOfKeyType(key, LwSciBufAttrKeyType_UMDPrivate);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Allocate memory for data node */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    *privateKeyNode = LwSciCommonCalloc(1, sizeof(LwSciBufUmdAttrValData));
    if (NULL == *privateKeyNode) {
        err = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Could not allocate memory for App data\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* fill the data in node */
    (*privateKeyNode)->len = len;
    (*privateKeyNode)->key = key;
    (*privateKeyNode)->privateAttrStatus = LwSciBufAttrStatus_SetLocked;
    lwListInit(&(*privateKeyNode)->listEntry);

    (*privateKeyNode)->value = LwSciCommonCalloc(1, (*privateKeyNode)->len);
    if (NULL == (*privateKeyNode)->value) {
        err = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Could not allocate memory for App data\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_privatekeynode;
    }

    LwSciCommonMemcpyS((*privateKeyNode)->value, (*privateKeyNode)->len,
                        value, len);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_privatekeynode:
    LwSciCommonFree(*privateKeyNode);
ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufValidateAttrListArray(
    const LwSciBufAttrList inputArray[],
    size_t inputCount)
{
    LwSciError err = LwSciError_Success;
    uint64_t listNum = 0U;
    bool isEqual = false;
    LwSciBufModule module1 = NULL, module2 = NULL;
    const LwSciBufAttrListObjPriv* attrListObj1 = NULL;
    const LwSciBufAttrListObjPriv* attrListObj2 = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    for (listNum = 0U; listNum < inputCount; listNum++) {
        err = LwSciBufAttrListValidate(inputArray[listNum]);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Invalid attrList supplied\n");
            LWSCI_ERR_ULONG("Index of invalid attrList: \n", listNum);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&inputArray[0]->refHeader, &attrListObjParam);
    attrListObj1 = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    module1 = attrListObj1->module;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (listNum = 0U; listNum < inputCount; listNum++) {
        LwSciCommonGetObjFromRef(&inputArray[listNum]->refHeader,
            &attrListObjParam);
        attrListObj2 = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

        if (LwSciBufAttrListState_Reconciled == attrListObj2->state) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_STR("Invalid attr list to LwSciBufValidateAttrListArray\n");
            LWSCI_ERR_ULONG("AttrList is reconciled list: \n", listNum);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        module2 = attrListObj2->module;

        err = LwSciBufModuleIsEqual(module1, module2, &isEqual);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufModuleIsEqual failed\n");
            LwSciCommonPanic();
        }
        if (false == isEqual) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_STR("AttrList are from different modules\n");
            LWSCI_ERR_ULONG("attrList: \n", listNum);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListDupRef(
    LwSciBufAttrList oldAttrList,
    LwSciBufAttrList* newAttrList)
{
    LwSciRef* newAttrListParam = NULL;
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: oldmodule: %p, newmoduleaddr: %p\n", oldAttrList,
        newAttrList);
    /* verify input parameters */
    if ((NULL == oldAttrList) || (NULL == newAttrList)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListDupRef\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Validate attribute list */
    sciErr = LwSciBufAttrListValidate(oldAttrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to validate LwSciBufAttrList reference.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Duplicate reference */
    sciErr = LwSciCommonDuplicateRef(&oldAttrList->refHeader, &newAttrListParam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to duplicate module instance\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *newAttrList = LwSciCastRefToLwSciBufAttrListRefPriv(newAttrListParam);

    /* print output variables */
    LWSCI_INFO("Output: oldAttrList %p, newAttrList: %p\n",
        oldAttrList, *newAttrList);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufAttrListGetModule(
    LwSciBufAttrList attrList,
    LwSciBufModule* module)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: attrList: %p, module: %p\n", attrList, module);
    /* verify input parameters */
    if ((NULL == attrList) || (NULL == module)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListGetModule\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate LwSciBufAttrList reference.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    *module = attrListObj->module;

    LWSCI_INFO("Output: module: %p\n", *module);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufUmdAttrKeyIterInit(
    LwSciBufAttrList attrList,
    uint64_t slotNum,
    LwSciBufUmdAttrKeyIterator* iter)
{
    LwSciError err = LwSciError_Success;

    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    uint64_t len = 0U;
    void* value = NULL;
    uint32_t key = 0U;

    LwSciBufUmdAttrKeyIterator* iterLocal = iter;

    LWSCI_FNENTRY("");

    if ((NULL == iter) || (NULL == attrList)) {
        LWSCI_ERR_STR("Invalid input arguments to iterate for Private Keys.");
        LwSciCommonPanic();
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);
    if (slotNum >= attrListObj->slotCount) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    key = (uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst;
    err = LwSciBufAttrListCommonGetAttr(attrList, slotNum, key, &len, &value,
            true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to to get attribute from attribute list");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    iterLocal->headAddr = value;
    iterLocal->iterAddr = iterLocal->headAddr->next;

ret:
    LWSCI_FNEXIT("");
    return err;
}

void LwSciBufUmdAttrKeyIterNext(
    LwSciBufUmdAttrKeyIterator* iter,
    bool* keyEnd,
    uint32_t* key)
{
    const LwSciBufUmdAttrValData* umdNodeAddr = NULL;

    LWSCI_FNENTRY("");

    if ((NULL == iter) || (NULL == keyEnd) || (NULL == key)) {
        LWSCI_ERR_STR("Invalid paramter to function LwSciBufUmdAttrKeyIterNext");
        LwSciCommonPanic();
    }

    if (iter->iterAddr == iter->headAddr) {
        *keyEnd = true;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    umdNodeAddr = (LwSciBufUmdAttrValData*)(void*)((char*)(void*)iter->iterAddr
        - LW_OFFSETOF(LwSciBufUmdAttrValData, listEntry));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    iter->iterAddr = iter->iterAddr->next;
    *key = umdNodeAddr->key;
    LWSCI_INFO("Output: keyEnd: %d, key: %u\n", *keyEnd, *key);

    LWSCI_FNEXIT("");
}

void LwSciBufAttrKeyIterInit(
    uint32_t keyTypeOffset,
    uint32_t dataTypeOffset,
    uint32_t keyOffset,
    LwSciBufAttrKeyIterator* iter)
{
    LwSciError err = LwSciError_Success;

    uint64_t tmp = 0UL;

    LWSCI_FNENTRY("");

    if (NULL == iter) {
        LWSCI_ERR_STR("Output argument cannot be Null\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: keyTypeOffset: %u, dataTypeOffset: %u, keyOffset %u, "
        "iter %p\n", keyTypeOffset, dataTypeOffset, keyOffset, iter);

    err = LwSciBufValidateAttrKeyType(keyTypeOffset);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Input arguments invalid");
        LwSciCommonPanic();
    }

    if ((uint32_t)LwSciBufAttrKeyType_UMDPrivate == keyTypeOffset) {
        /* LwSciBufAttrKeyIterNext() lwrrently doesn't support UMD keys, so if
         * we start on an UMD key bad things happen. */
        LWSCI_ERR_STR("Input arguments invalid");
        LwSciCommonPanic();
    }

    if ((uint32_t)LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE <= keyOffset) {
        LWSCI_ERR_STR("Input arguments invalid");
        LwSciCommonPanic();
    }

    err = LwSciBufValidateLwSciBufType(dataTypeOffset);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Input arguments invalid");
        LwSciCommonPanic();
    }

    iter->keyType = 0U;
    tmp = sizeof(attrKeyDescList)/sizeof(attrKeyDescList[0]);
    if (tmp > UINT32_MAX) {
        LWSCI_ERR_STR("keyTypeMax is out of range.\n");
        LwSciCommonPanic();
    } else {
        iter->keyTypeMax = (uint32_t)tmp;
    }

    iter->dataType = 0U;
    tmp = sizeof(attrKeyDescList[0])/sizeof(attrKeyDescList[0][0]);
    if (tmp > UINT32_MAX) {
        LWSCI_ERR_STR("dataTypeMax is out of range.\n");
        LwSciCommonPanic();
    } else {
        iter->dataTypeMax = (uint32_t)tmp;
    }

    iter->key = 0U;
    tmp = sizeof(attrKeyDescList[0][0])/sizeof(attrKeyDescList[0][0][0]);
    if (tmp > UINT32_MAX) {
        LWSCI_ERR_STR("keyMax is out of range.\n");
        LwSciCommonPanic();
    } else {
        iter->keyMax = (uint32_t)tmp;
    }

    iter->keyType = keyTypeOffset;
    iter->dataType = dataTypeOffset;
    iter->key = keyOffset;

    LWSCI_INFO("Output: keyType: %u, dataType: %u, key %u\n"
        "keyTypeMax: %u, dataTypeMax: %u, keyMax: %u\n",
        iter->keyType, iter->dataType, iter->key,
        iter->keyTypeMax, iter->dataTypeMax, iter->keyMax);

    LWSCI_FNEXIT("");
}

void LwSciBufAttrKeyIterNext(
    LwSciBufAttrKeyIterator* iter,
    bool* keyTypeEnd,
    bool* dataTypeEnd,
    bool* keyEnd,
    uint32_t* key)
{
    const LwSciBufAttrKeyDescPriv* desc = NULL;
    bool isKeyValid = false;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: iter %p, keyTypeEnd %p, dataTypeEnd %p, keyEnd %p, "
        "key %p\n", iter, keyTypeEnd, dataTypeEnd, keyEnd, key);

    if ((NULL == iter) || (NULL == keyTypeEnd) || (NULL == dataTypeEnd)
            || (NULL == keyEnd) || (NULL == key)) {
        LWSCI_ERR_STR("Invalid paramter to function LwSciBufAttrKeyIterNext\n");
        LwSciCommonPanic();
    }

    /* clear user datail, to remove false positive */
    *keyEnd = false;
    *dataTypeEnd = false;
    *keyTypeEnd = false;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    while(true) {

        /* save the output first */
        *key = LW_SCI_BUF_ATTRKEY_ENCODE(iter->keyType,
                                         iter->dataType, iter->key);
        desc = &attrKeyDescList[iter->keyType][iter->dataType][iter->key];

        isKeyValid = desc->name != NULL;

        if (iter->key == iter->keyMax) {
            iter->key = 0;
            /* increament parent index, reset my index */
            if ((UINT32_MAX - 1U) < iter->dataType) {
                LWSCI_ERR_STR("iter->dataType is too large.\n");
                LwSciCommonPanic();
            } else {
                iter->dataType++;
            }
            *keyEnd = true;
            break;
        }

        if (iter->dataType == iter->dataTypeMax) {
            iter->key = 0;
            iter->dataType = 0;
            *keyEnd = true;
            *dataTypeEnd = true;
            /* increament parent index, reset my index */
            if ((UINT32_MAX - 1U) < iter->keyType) {
                LWSCI_ERR_STR("iter->dataType is too large.\n");
                LwSciCommonPanic();
            } else {
                iter->keyType++;
            }
            //TODO: call Umd iterator
            if ((uint32_t)LwSciBufAttrKeyType_UMDPrivate == iter->keyType) {
                iter->keyType++;
            }
            break;
        }

        if (iter->keyType == iter->keyTypeMax) {
            *keyEnd = true;
            *dataTypeEnd = true;
            *keyTypeEnd = true;
            break;
        }

        iter->key++;

        if (true == isKeyValid) {
            break;
        }

    }



    LWSCI_INFO("Output: keyTypeEnd %u, dataTypeEnd %u, keyEnd %u, "
        "key %u\n", *keyTypeEnd, *dataTypeEnd, *keyEnd, *key);

    LWSCI_FNEXIT("");
}

LwSciError LwSciBufAttrListValidate(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    /* print incoming paramters */
    LWSCI_INFO("Input: attrList: %p\n", attrList);
    if (NULL == attrList) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListValidate\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get attrList object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    /* validate Magic ID */
    if (LW_SCI_BUF_ATTR_MAGIC != attrListObj->magic) {
        LWSCI_ERR_STR("Failed to validate attribute list\n");
        LWSCI_ERR_HEXUINT("magic: 0x \n", attrListObj->magic);
        LwSciCommonPanic();
    }

    /* print output going variables */
    LWSCI_INFO("Output: reference: %p, object: %p\n", attrList, attrListObj);

ret:
    LWSCI_FNEXIT("");
    return err;
}



/**
 * @brief Public interface definition
 */
void LwSciBufAttrKeyGetIpcRouteAffinity(
    uint32_t key,
    bool localPeer,
    LwSciBufIpcRouteAffinity* routeAffinity)
{
    LwSciError err = LwSciError_Success;
    uint32_t decodedKey = 0U;
    uint32_t decodedDataType = 0U;
    uint32_t decodedKeyType = 0U;
    LwSciBufIpcRouteAffinity tmpRouteAffinity = LwSciBufIpcRoute_Max;

    LWSCI_FNENTRY("");

    if (NULL == routeAffinity) {
        LWSCI_ERR_STR("routeAffinity is NULL!");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs - routeAffinity ptr %p AttrKey: %"PRIu32,
                routeAffinity, key);

    err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType,
            &decodedKeyType);
    if (LwSciError_Success != err) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LWSCI_ERR_STR("Failed to decode key");
        LwSciCommonPanic();
    }

    if (true == localPeer) {
        tmpRouteAffinity = attrKeyDescList[decodedKeyType]
                                [decodedDataType]
                                [decodedKey].localPeerIpcAffinity;
    } else {
        tmpRouteAffinity = attrKeyDescList[decodedKeyType]
                                [decodedDataType]
                                [decodedKey].remotePeerIpcAffinity;
    }

    if (LwSciBufIpcRoute_Max <= tmpRouteAffinity) {
        LwSciCommonPanic();
    }

    *routeAffinity = tmpRouteAffinity;

    LWSCI_INFO("Outputs - routeAffinity %"PRIu32"", *routeAffinity);

    LWSCI_FNEXIT("");
}

void LwSciBufAttrKeyGetPolicy(
    uint32_t key,
    LwSciBuf_ReconcilePolicy* policy)
{
    LwSciError err = LwSciError_Success;
    uint32_t decodedKey;
    uint32_t decodedDataType;
    uint32_t decodedKeyType;

    LWSCI_FNENTRY("");

    if (NULL == policy) {
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs - Policy ptr %p AttrKey: %"PRIu32"\n",
                policy, key);

    err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType,
            &decodedKeyType);
    if (LwSciError_Success != err) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LWSCI_ERR_STR("Failed to decode key");
        LwSciCommonPanic();
    }
    if (((uint32_t)LwSciBufAttrKeyType_UMDPrivate == decodedKeyType) &&
            ((uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst != key)) {
        /* If this is an UMD key, then we need to also panic for attribute keys
         * other than LwSciBufInternalAttrKey_LwMediaPrivateFirst, since they
         * have no attribute key descriptor */
        LwSciCommonPanic();
    }

    *policy = attrKeyDescList[decodedKeyType]
                            [decodedDataType]
                            [decodedKey].recpolicy;

    LWSCI_INFO("Outputs - Policy %"PRIu32"\n", *policy);

    LWSCI_FNEXIT("");
}

/**
 * @brief Public interface definition
 */

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListCreate(
    LwSciBufModule module,
    LwSciBufAttrList* newAttrList)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListCreateMultiSlot(module, 1U, newAttrList, true);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListCreateMultiSlot(
    LwSciBufModule module,
    size_t slotCount,
    LwSciBufAttrList* newAttrList,
    bool emptyIpcRoute)
{
    LwSciError err = LwSciError_Success;
    LwSciObj* attrListObjParam = NULL;
    LwSciBufAttrListRefPriv* attrListRef = NULL;
    LwSciRef* attrListRefParam = NULL;

    LWSCI_FNENTRY("");

    /* Input sanity check */
    if ((NULL == newAttrList) || (0U == slotCount) || (NULL == module)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListCreateMultiSlot\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *newAttrList = NULL;

    /* print incoming parameters */
    LWSCI_INFO("Input: slotCount:%lu, module: %p newAttrList: %p\n",
        slotCount, module, newAttrList);

    /* validate module structure */
    err = LwSciBufModuleValidate(module);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate input module\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* create attrList reference and object */
    err = LwSciCommonAllocObjWithRef(sizeof(LwSciBufAttrListObjPriv),
        sizeof(LwSciBufAttrListRefPriv), &attrListObjParam,
        &attrListRefParam);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed allocate memory for LwSciBufAttrList\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    attrListRef = LwSciCastRefToLwSciBufAttrListRefPriv(attrListRefParam);

    /* initialize data inside attrList */
    err = LwSciBufAttrListInit(module, slotCount, attrListRef, emptyIpcRoute);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not Initialize attrList\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_attrList;
    }

    err = LwSciBufAttrListSetState(attrListRef,
            LwSciBufAttrListState_Unreconciled);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not set attrList state\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_attrList;
    }

    /* save in output variable */
    *newAttrList = attrListRef;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_attrList:
    LwSciBufAttrListFree(attrListRef);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListCompareReconcileStatus(
    LwSciBufAttrList attrList,
    bool isReconciled)
{
    LwSciError err = LwSciError_Success;
    bool reconciledStatus = false;

    LWSCI_FNENTRY("");

    if (NULL == attrList) {
        LWSCI_ERR_STR("Invalid paramter to function\n");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListIsReconciled(attrList, &reconciledStatus);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListIsReconciled failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (reconciledStatus != isReconciled) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 10_5), "LwSciBuf-ADV-MISRAC2012-009")
        LWSCI_ERR_INT("AttrList reconciled status value ", (int32_t)reconciledStatus);
        LWSCI_ERR_INT(" expected value: \n", (int32_t)isReconciled);
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 10_5))
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListIsReconciled(
    LwSciBufAttrList attrList,
    bool* isReconciled)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p, isReconiled addr: %p\n", attrList,
        isReconciled);

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListIsReconciled\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (NULL == isReconciled) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListIsReconciled\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    if (LwSciBufAttrListState_Reconciled == attrListObj->state) {
        *isReconciled = true;
    } else {
        *isReconciled = false;
    }

    LWSCI_INFO("Output: isReconiled: %d\n", *isReconciled);

ret:
    LWSCI_FNEXIT("");
    return err;
}

size_t LwSciBufAttrListGetSlotCount(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    uint64_t slotCount = 0U;
    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    /* print incoming variables */
    LWSCI_INFO("Input: attrList: %p\n", attrList);

    /* validate attrList*/
    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListGetSlotCount\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    /* get slotCount from object */
    slotCount = attrListObj->slotCount;
    /* print output variables */
    LWSCI_INFO("Output: reference: %p, object: %p, slotCount: %lu\n", attrList,
        attrListObj, slotCount);

ret:
    LWSCI_FNEXIT("");
    return slotCount;
}

void LwSciBufAttrListFree(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* No-op on NULL attribute list */
    if (NULL == attrList) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Input sanity check */
    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* free reference and object */
    LwSciCommonFreeObjAndRef(&attrList->refHeader,
        LwSciBufAttrCleanupCallback, NULL);

    LWSCI_INFO("Output: return status: %d\n", err);
ret:
    LWSCI_FNEXIT("");
}

LwSciError LwSciBufAttrListClone(
    LwSciBufAttrList origAttrList,
    LwSciBufAttrList* newAttrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrList ogAttrList = origAttrList;
    const LwSciBufAttrListObjPriv* ogAttrListObj = NULL;
    LwSciObj* ogAttrListObjParam = NULL;
    LwSciBufAttrListObjPriv* newAttrListObj = NULL;
    LwSciObj* newAttrListObjParam = NULL;
    uint64_t slotCount = 0U;
    uint64_t slotNum = 0U;

    LWSCI_FNENTRY("");

    /* Validate input attribute list */
    err = LwSciBufAttrListValidate(ogAttrList);
    if (LwSciError_Success != err) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad attribute list to LwSciBufAttrListCloneUnreconciled\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /*validate input arguments */
    if (NULL == newAttrList) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad attribute list to LwSciBufAttrListCloneUnreconciled\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: ogAttrList: %p, newattrListAddr: %p\n",
        origAttrList, newAttrList);

    /* get object from reference */
    LwSciCommonGetObjFromRef(&ogAttrList->refHeader, &ogAttrListObjParam);
    ogAttrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(ogAttrListObjParam);

    slotCount = ogAttrListObj->slotCount;

    /* lock orignal attr list */
    LwSciCommonObjLock(&ogAttrList->refHeader);

    /* allocate memory for new attribute list */
    err = LwSciBufAttrListCreateMultiSlot(ogAttrListObj->module,
            slotCount, newAttrList, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to create newUnreconciledAttrList\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_lock;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&(*newAttrList)->refHeader, &newAttrListObjParam);
    newAttrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(newAttrListObjParam);

    /* copy slot-by-slot*/
    for (slotNum=0U; slotNum<slotCount; slotNum++) {
        err = LwSciBufAttrListCloneSlot(
                ogAttrList, slotNum, *newAttrList, slotNum);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_newAttrList;
        }
    }

    newAttrListObj->state = ogAttrListObj->state;

    LWSCI_INFO("Output: newAttrList: %p\n", *newAttrList);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto free_lock;

free_newAttrList:
    LwSciBufAttrListFree(*newAttrList);
    *newAttrList = NULL;

free_lock:
    LwSciCommonObjUnlock(&ogAttrList->refHeader);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
LwSciError LwSciBufAttrListCommonSetAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    const void* pairArray,
    size_t pairCount,
    LwSciBufAttrKeyType keyType,
    bool override,
    bool skipValidation)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    const LwSciBufAttrListObjPriv* attrListObj = NULL;
    LwSciObj* attrListObjParam = NULL;

    LWSCI_FNENTRY("");

    if ((NULL == attrList) || (NULL == pairArray) || (0U == pairCount)) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid attrList to LwSciBufAttrListSlotSetAttrs\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&attrList->refHeader, &attrListObjParam);
    attrListObj = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    if (slotIndex >= attrListObj->slotCount) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Slot count provided is than actual slot count.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LwSciBufAttrKeyType_Public == keyType) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        const LwSciBufAttrKeyValuePair* keyValPair = (const LwSciBufAttrKeyValuePair*)pairArray;

        err = LwSciBufAttrListSlotSetAttrs(attrList, slotIndex, keyValPair,
                pairCount, override, skipValidation);
    } else if ((LwSciBufAttrKeyType_Internal == keyType) ||
        (LwSciBufAttrKeyType_UMDPrivate == keyType)) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        const LwSciBufInternalAttrKeyValuePair* intKeyValPair = (const LwSciBufInternalAttrKeyValuePair*)pairArray;

        err = LwSciBufAttrListSlotSetInternalAttrs(attrList, slotIndex,
                intKeyValPair, pairCount, override, skipValidation);
    } else if (LwSciBufAttrKeyType_Private == keyType) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        const LwSciBufPrivateAttrKeyValuePair* pvtKeyValPair = (const LwSciBufPrivateAttrKeyValuePair*)pairArray;

        err = LwSciBufAttrListSlotSetPrivateAttrs(attrList, slotIndex,
                pvtKeyValPair, pairCount, override, skipValidation);
    } else {
        LWSCI_WARN("Invalid KeyType %u to LwSciBufAttrListCommonSetAttrs\n", keyType);
        err = LwSciError_BadParameter;
    }

    if (LwSciError_Success != err) {
        LWSCI_ERR_UINT("Failed to set keyType attributes: \n", (uint32_t)keyType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListSetAttrs(
    LwSciBufAttrList attrList,
    LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");

    if ((NULL == attrList) || (NULL == pairArray) || (0U == pairCount)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters supplied to LwSciBufAttrListSetAttrs().");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid LwSciBufAttrList supplied to LwSciBufAttrListSetAttrs().");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (!LwSciBufAttrListIsWritable(attrList)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Cannot set LwSciBufAttrKey(s) in the provided LwSciBufAttrList.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0U, pairArray, pairCount,
            LwSciBufAttrKeyType_Public, false, false);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListSetInternalAttrs(
    LwSciBufAttrList attrList,
    LwSciBufInternalAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");

    if ((NULL == attrList) || (NULL == pairArray) || (0U == pairCount)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters supplied to LwSciBufAttrListSetInternalAttrs().");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid LwSciBufAttrList supplied to LwSciBufAttrListSetInternalAttrs().");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (!LwSciBufAttrListIsWritable(attrList)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Cannot set LwSciBufInternalAttrKey(s) in the provided LwSciBufAttrList.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListCommonSetAttrs(attrList, 0, pairArray,
            pairCount, LwSciBufAttrKeyType_Internal, false, false);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
LwSciError LwSciBufAttrListCommonGetAttrsWithLock(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    void* pairArray,
    size_t pairCount,
    LwSciBufAttrKeyType keyType,
    bool override,
    bool acquireLock)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");

    if (LwSciBufAttrKeyType_Public == keyType) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LwSciBufAttrKeyValuePair* keyValPair =
                            (LwSciBufAttrKeyValuePair*)pairArray;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        err = LwSciBufAttrListSlotGetAttrsPriv(attrList, slotIndex, keyValPair,
                pairCount, override, acquireLock);
    } else if ((LwSciBufAttrKeyType_Internal == keyType) ||
                (LwSciBufAttrKeyType_UMDPrivate == keyType)) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LwSciBufInternalAttrKeyValuePair* intKeyValPair =
                            (LwSciBufInternalAttrKeyValuePair*)pairArray;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        err = LwSciBufAttrListSlotGetInternalAttrsPriv(attrList, slotIndex,
                intKeyValPair, pairCount, override, acquireLock);
    } else if (LwSciBufAttrKeyType_Private == keyType) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LwSciBufPrivateAttrKeyValuePair* pvtKeyValPair =
                            (LwSciBufPrivateAttrKeyValuePair*)pairArray;
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        err = LwSciBufAttrListSlotGetPrivateAttrs(attrList, slotIndex,
                pvtKeyValPair, pairCount, override, acquireLock);
    } else {
        /* Do nothing */
        LWSCI_WARN("Invalid KeyType to LwSciBufAttrListCommonGetAttrs\n");
        err = LwSciError_BadParameter;
    }

    if (LwSciError_Success != err) {
        LWSCI_ERR_UINT("Failed to get keyType attributes: \n", (uint32_t)keyType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
LwSciError LwSciBufAttrListCommonGetAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    void* pairArray,
    size_t pairCount,
    LwSciBufAttrKeyType keyType,
    bool override)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))

    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListCommonGetAttrsWithLock(attrList, slotIndex, pairArray,
            pairCount, keyType, override, true/* Take lock by default */);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrsWithLock failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListGetAttrs(
    LwSciBufAttrList attrList,
    LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");

    err = LwSciBufAttrListSlotGetAttrs(attrList, 0, pairArray, pairCount);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListSlotGetAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListSlotGetAttrsPriv(attrList, slotIndex, pairArray,
            pairCount, false, true);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListGetInternalAttrs(
    LwSciBufAttrList attrList,
    LwSciBufInternalAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");

    err = LwSciBufAttrListSlotGetInternalAttrs(attrList, 0, pairArray,
            pairCount);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListAppendWithLocksUnreconciled(
    const LwSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool acquireLocks,
    LwSciBufAttrList* newUnreconciledAttrList)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufAttrListObjPriv* attrListObj2 = NULL;
    LwSciObj* attrListObjParam = NULL;
    LwSciBufAttrList newAttrList = NULL;
    uint64_t totalSlotNum = 0U;
    uint64_t totalSlotCount = 0U;
    uint64_t listNum = 0U, listSlotCount = 0U, slotNum = 0U;

    LWSCI_FNENTRY("");

    LWSCI_INFO("inputAttrListArr: %p, size: %lu, outputarr: %p\n",
        inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
        newUnreconciledAttrList);

    if ((NULL == inputUnreconciledAttrListArray) ||
        (0U == inputUnreconciledAttrListCount) ||
        (NULL == newUnreconciledAttrList)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters to LwSciBufAttrListAppendUnreconciled\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufValidateAttrListArray(inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate attribute list array\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (acquireLocks) {
        err = LwSciBufAttrListsLock(inputUnreconciledAttrListArray,
                inputUnreconciledAttrListCount);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* Get total slot count in all input attrlist array */
    totalSlotCount = LwSciBufAttrListTotalSlotCount(
        inputUnreconciledAttrListArray, inputUnreconciledAttrListCount);

    /* get object from reference */
    LwSciCommonGetObjFromRef(
            &inputUnreconciledAttrListArray[0]->refHeader,
            &attrListObjParam);
    attrListObj2 = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

    /* Create attrlist with total number of slots */
    err = LwSciBufAttrListCreateMultiSlot(attrListObj2->module, totalSlotCount,
            &newAttrList, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to create newUnreconciledAttrList\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attr_lists;
    }

    /* get object from reference */
    LwSciCommonGetObjFromRef(&newAttrList->refHeader, &attrListObjParam);

    for (listNum = 0U; listNum < inputUnreconciledAttrListCount; listNum++) {
        /* get object from reference */
        LwSciCommonGetObjFromRef(
            &inputUnreconciledAttrListArray[listNum]->refHeader,
            &attrListObjParam);

        attrListObj2 = LwSciCastObjToLwSciBufAttrListObjPriv(attrListObjParam);

        listSlotCount = attrListObj2->slotCount;

        for (slotNum = 0U; slotNum < listSlotCount; slotNum++) {
            err = LwSciBufAttrListCloneSlot(
                    inputUnreconciledAttrListArray[listNum], slotNum,
                    newAttrList, totalSlotNum);
            if (LwSciError_Success != err) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_newAttrList;
            }
            totalSlotNum++;
        }
    }

    *newUnreconciledAttrList = newAttrList;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto unlock_attr_lists;

free_newAttrList:
    LwSciBufAttrListFree(newAttrList);

unlock_attr_lists:
    if (acquireLocks) {
        LwSciError error = LwSciError_Success;
        error = LwSciBufAttrListsUnlock(inputUnreconciledAttrListArray,
                inputUnreconciledAttrListCount);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Could not unlock Attribute Lists\n");
            LwSciCommonPanic();
        }
    }
ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListAppendUnreconciled(
    const LwSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciBufAttrList* newUnreconciledAttrList)
{
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");

    err = LwSciBufAttrListAppendWithLocksUnreconciled(
            inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
            true, newUnreconciledAttrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not append attribute list\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListSetState(*newUnreconciledAttrList,
            LwSciBufAttrListState_Appended);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not set attribute list state\n");
        LwSciCommonPanic();
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListGetDataTypes(
    LwSciBufAttrList attrList,
    const LwSciBufType** bufType,
    size_t* numDataTypes)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair = {0};

    LWSCI_FNENTRY("");

    if ((NULL == attrList) || (NULL == bufType) || (NULL == numDataTypes)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufAttrListGetDataTypes\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: attrList ptr: %p, bufType ptr: %p, numDataTypes ptr: %p\n"
        ,attrList, bufType, numDataTypes);

    *bufType = NULL;
    *numDataTypes = 0;

    /* get LwSciBufType[] from attribute list */
    keyValPair.key = LwSciBufGeneralAttrKey_Types;
    err = LwSciBufAttrListGetAttrs(attrList, &keyValPair, 1);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetAttrs failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *numDataTypes = keyValPair.len/sizeof(LwSciBufType);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    *bufType = (const LwSciBufType*)keyValPair.value;

    LWSCI_INFO("Output: bufType ptr: %p, numElements: %zu\n", *bufType,
        *numDataTypes);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufGetUMDPrivateKeyWithOffset(
    LwSciBufInternalAttrKey key,
    uint32_t offset,
    LwSciBufInternalAttrKey* offsettedKey)
{
    LwSciError sciErr = LwSciError_Success;
    uint32_t offsettedKeyUint = 0U;
    bool isMemberOf = false;

    LWSCI_FNENTRY("");

    if ((NULL == offsettedKey) || ((uint32_t)LW_SCI_BUF_MAX_KEY_COUNT < offset) ||
        ((uint32_t)LwSciBufAttrKeyType_UMDPrivate !=
            LW_SCI_BUF_DECODE_KEYTYPE(key))) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufGetUMDPrivateKeyWithOffset\n");
        LWSCI_ERR_UINT("offset: \n", offset);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        LWSCI_ERR_UINT("Invalid key type: \n", LW_SCI_BUF_DECODE_KEYTYPE(key));
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: key: %"PRIu32", offset: %"PRIu32", offsetted ptr: %p\n",
        key, offset, offsettedKey);

    offsettedKeyUint = (uint32_t)key + offset;

    /* this will check if we have overflown into
     *  a) other key type
     *  b) other UMDPrivateKeys
     */
    isMemberOf = LwSciBufIsAttrKeyMemberof(offsettedKeyUint,
                                            LwSciBufAttrKeyType_UMDPrivate,
                                            LW_SCI_BUF_DECODE_DATATYPE(key));
    if (false == isMemberOf) {
        LWSCI_ERR_STR("Offsetted key has overflown\n");
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        LWSCI_ERR_UINT("input key KeyType:\n", LW_SCI_BUF_DECODE_KEYTYPE(key));
        LWSCI_ERR_UINT("input key DataType\n", LW_SCI_BUF_DECODE_DATATYPE(key));
        LWSCI_ERR_UINT("offsetted key KeyType:\n", LW_SCI_BUF_DECODE_KEYTYPE(offsettedKeyUint));
        LWSCI_ERR_UINT("offsetted key DataType\n", LW_SCI_BUF_DECODE_DATATYPE(offsettedKeyUint));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
        sciErr = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(offsettedKey, sizeof(*offsettedKey),
               &offsettedKeyUint, sizeof(offsettedKeyUint));

    LWSCI_INFO("Output: offsetted key: %"PRIu32"\n", *offsettedKey);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufImportCheckingNeeded(
    LwSciBufAttrList reconciledList,
    uint32_t key,
    bool *result)
{
    LwSciError err = LwSciError_Success;

    uint32_t decodedKey = 0U;
    uint32_t decodedDataType = 0U;
    uint32_t decodedKeyType = 0U;

    bool needsCheck = true;

    const LwSciBufAttrKeyDescPriv* desc = NULL;

    LWSCI_FNENTRY("");

    if (NULL == result) {
        LwSciCommonPanic();
    }

    err = LwSciBufAttrListValidate(reconciledList);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType, &decodedKeyType);
    if (LwSciError_Success != err) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LwSciCommonPanic();
    }
    if (((uint32_t)LwSciBufAttrKeyType_UMDPrivate == decodedKeyType) &&
            ((uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst != key)) {
        /* If this is an UMD key, then we need to also panic for attribute keys
         * other than LwSciBufInternalAttrKey_LwMediaPrivateFirst, since they
         * have no attribute key descriptor */
        LwSciCommonPanic();
    }

    LwSciBufAttrGetDesc(key, &desc);

    if (LwSciBufKeyAccess_Input == desc->keyAccess) {
        /* The keys are not readable in reconciled attribute list*/
        needsCheck = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (LwSciBufKeyImportQualifier_Optional == desc->importQualifier) {
        needsCheck = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (LwSciBufKeyImportQualifier_Mandatory == desc->importQualifier) {
        needsCheck = true;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (LwSciBufKeyImportQualifier_Conditional == desc->importQualifier) {
        LwSciError error = LwSciError_Success;

        uint32_t bufTypeKey = (uint32_t)LwSciBufGeneralAttrKey_Types;
        const LwSciBufType *bufType = NULL;
        uint64_t len = 0U;
        uint64_t numBufTypes = 0U;

        error = LwSciBufAttrListCommonGetAttr(reconciledList, 0U,
                bufTypeKey, &len, (void*)&bufType, false);
        if (LwSciError_Success != error) {
            /* This is impossible since LwSciBufGeneralAttrKey_Types will
             * always be readable. */
            LwSciCommonPanic();
        }
        numBufTypes = len / sizeof(LwSciBufType);

        switch (key) {
            case ((uint32_t)LwSciBufImageAttrKey_ImageCount):
            {
                /* This key is:
                 *   - mandatory when performing Image/Tensor reconciliation
                 *   - optional otherwise */
                needsCheck = ((numBufTypes == 2U)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Image)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Tensor));

                break;
            }
            case ((uint32_t)LwSciBufTensorAttrKey_AlignmentPerDim):
            {
                /* This key is:
                 *   - mandatory when performing Tensor/Tensor reconciliation
                 *   - optional otherwise */
                needsCheck = ((numBufTypes == 1U)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Tensor));
                break;
            }
            case ((uint32_t)LwSciBufTensorAttrKey_PixelFormat):
            {
                /* This key is:
                 *   - mandatory when performing Image/Tensor reconciliation
                 *   - optional otherwise */
                needsCheck = ((numBufTypes == 2U)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Image)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Tensor));

                break;
            }
            default:
            {
                needsCheck = true;
                break;
            }
        }
    }

fn_exit:
    *result = needsCheck;

    LWSCI_FNEXIT("");

    return err;
}

LwSciError LwSciBufReconcileCheckingNeeded(
    LwSciBufAttrList reconciledList,
    uint32_t key,
    bool *result)
{
    LwSciError err = LwSciError_Success;

    uint32_t decodedKey = 0U;
    uint32_t decodedDataType = 0U;
    uint32_t decodedKeyType = 0U;

    bool needsCheck = true;

    const LwSciBufAttrKeyDescPriv* desc = NULL;

    LWSCI_FNENTRY("");

    if (NULL == result) {
        LwSciCommonPanic();
    }

    err = LwSciBufAttrListValidate(reconciledList);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    err = LwSciBufAttrKeyDecode(key, &decodedKey, &decodedDataType, &decodedKeyType);
    if (LwSciError_Success != err) {
        /* These variables are controlled by LwSci APIs and are not exposed to the
         * caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LwSciCommonPanic();
    }
    if (((uint32_t)LwSciBufAttrKeyType_UMDPrivate == decodedKeyType) &&
            ((uint32_t)LwSciBufInternalAttrKey_LwMediaPrivateFirst != key)) {
        /* If this is an UMD key, then we need to also panic for attribute keys
         * other than LwSciBufInternalAttrKey_LwMediaPrivateFirst, since they
         * have no attribute key descriptor */
        LwSciCommonPanic();
    }

    LwSciBufAttrGetDesc(key, &desc);

    if (LwSciBufKeyAccess_Output == desc->keyAccess) {
        /* The keys are not readable in unreconciled attribute list since these
         * are set by LwSciBuf */
        needsCheck = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (LwSciBufKeyReconcileQualifier_Optional == desc->reconcileQualifier) {
        needsCheck = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (LwSciBufKeyReconcileQualifier_Mandatory == desc->reconcileQualifier) {
        needsCheck = true;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

    if (LwSciBufKeyReconcileQualifier_Conditional == desc->reconcileQualifier) {
        LwSciError error = LwSciError_Success;

        uint32_t bufTypeKey = (uint32_t)LwSciBufGeneralAttrKey_Types;
        const LwSciBufType *bufType = NULL;
        uint64_t len = 0U;
        uint64_t numBufTypes = 0U;

        error = LwSciBufAttrListCommonGetAttr(reconciledList, 0U,
                bufTypeKey, &len, (void*)&bufType, false);
        if (LwSciError_Success != error) {
            /* This is impossible since LwSciBufGeneralAttrKey_Types will
             * always be readable. */
            LwSciCommonPanic();
        }
        numBufTypes = len / sizeof(LwSciBufType);

        switch (key) {
            case ((uint32_t)LwSciBufImageAttrKey_ImageCount):
            {
                /* This key is:
                 *   - mandatory when performing Image/Tensor reconciliation
                 *   - optional otherwise */
                needsCheck = ((numBufTypes == 2U)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Image)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Tensor));

                break;
            }
            case ((uint32_t)LwSciBufTensorAttrKey_AlignmentPerDim):
            {
                /* This key is:
                 *   - mandatory when performing Tensor/Tensor reconciliation
                 *   - optional otherwise */
                needsCheck = ((numBufTypes == 1U)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Tensor));

                break;
            }
            case ((uint32_t)LwSciBufTensorAttrKey_PixelFormat):
            {
                /* This key is:
                 *   - mandatory when performing Image/Tensor reconciliation
                 *   - optional otherwise */
                needsCheck = ((numBufTypes == 2U)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Image)
                        && ContainsBufferType(bufType, numBufTypes, LwSciBufType_Tensor));

                break;
            }
            default:
            {
                needsCheck = true;
                break;
            }
        }
    }

fn_exit:
    *result = needsCheck;

    LWSCI_FNEXIT("");

    return err;
}

LwSciError LwSciBufAttrListIsIsoEngine(
    LwSciBufAttrList attrList,
    bool* isIsoEngine)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufInternalAttrKeyValuePair intKeyValPair;
    uint64_t len = 0U, arrayCount = 0U;
    const LwSciBufHwEngine* engineArray = NULL;

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListValidate(attrList);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListValidate() failed.");
        goto ret;
    }

    if (isIsoEngine == NULL) {
        LwSciCommonPanic();
    }
    (void)memset(&intKeyValPair, 0x0, sizeof(intKeyValPair));

    LWSCI_INFO("Input: attrList: %p, isIsoEngine: %p", attrList,
        isIsoEngine);

    /* initialize output*/
    *isIsoEngine = false;

    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;
    sciErr = LwSciBufAttrListCommonGetAttrs(attrList, 0, &intKeyValPair, 1,
                LwSciBufAttrKeyType_Internal, true);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttr failed.");
        goto ret;
    }

    len = (uint64_t)intKeyValPair.len;
    engineArray = (const LwSciBufHwEngine *)intKeyValPair.value;

    arrayCount = len/sizeof(LwSciBufHwEngine);

    sciErr = LwSciBufIsIsoEngine(engineArray, arrayCount, isIsoEngine);
    if (sciErr != LwSciError_Success) {
        goto ret;
    }

    LWSCI_INFO("Output: isIsoEngine: %s", *isIsoEngine ? "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}
