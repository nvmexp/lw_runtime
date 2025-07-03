/*
 * lwscibuf_constraint_lib_common.c
 *
 * Constraint Library to get hardware constraints
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <string.h>

#include "lwscibuf_constraint_lib_priv.h"
#include "lwscicommon_os.h"
#include "lwscilog.h"

const LwSciBufArrayConstraints arrayDefaultConstraints =
{
    .startAddrAlign = 1U,
    .dataAlign = 1U,
};

const LwSciBufImagePyramidConstraints imgPyramidDefaultConstraints =
{
    .scaleFactor = 0.1f,
    .levelCount = LW_SCI_BUF_PYRAMID_MAX_LEVELS,
};

static void LwSciBufHwEngCreateIdHelper(
    LwSciBufHwEngName engName,
    uint32_t instance,
    int64_t* engId)
{
    uint32_t id = (((uint32_t)engName & LW_SCI_BUF_ENG_NAME_BIT_MASK)
            << LW_SCI_BUF_ENG_NAME_BIT_START)
        | ((instance & LW_SCI_BUF_ENG_INSTANCE_BIT_MASK)
            << LW_SCI_BUF_ENG_INSTANCE_BIT_START);
    *engId = (int64_t) id;
}

static inline bool LwSciBufHwEngNameValidate(
    LwSciBufHwEngName engName)
{
    /* Since this enum is non-contiguous, we need to validate this via
     * enumerating each enum value.
     * TODO: Remove this once we make this enum contiguous. */
    bool isValid = false;

    switch (engName)
    {
        case LwSciBufHwEngName_Display:
        case LwSciBufHwEngName_Isp:
        case LwSciBufHwEngName_Vi:
        case LwSciBufHwEngName_Csi:
        case LwSciBufHwEngName_Vic:
        case LwSciBufHwEngName_Gpu:
        case LwSciBufHwEngName_MSENC:
        case LwSciBufHwEngName_LWDEC:
        case LwSciBufHwEngName_LWJPG:
        case LwSciBufHwEngName_PVA:
        case LwSciBufHwEngName_DLA:
        case LwSciBufHwEngName_PCIe:
        case LwSciBufHwEngName_OFA:
        {
            isValid = true;
            break;
        }
        default:
        {
            isValid = false;
            break;
        }
    }
    return isValid;
}

void LwSciBufGetDefaultImageConstraints(
    LwSciBufImageConstraints* constraints)
{
    /* Image Default constraints are considered to work by default
     *
     * Note: Designated initializers are not used for the nested structs due to
     * Restriction #64 in the QNX Safety Manual:
     *
     *  > Designated initializers with nested structs should be not be used to
     *  > avoid a known problem in the compiler that results in incorrect values
     *  > used for some fields.
     */
    if (NULL == constraints) {
        LwSciCommonPanic();
    }

/*
 * TODO lwbugs/200671636 [LwSciBuf] Fix default image constraints in constraint library
 */
#if !defined(__x86_64__)
    constraints->plConstraints.startAddrAlign = 128U;
    constraints->plConstraints.pitchAlign = 128U;
    constraints->plConstraints.heightAlign = 1U;
    constraints->plConstraints.sizeAlign = 1U;

    constraints->blConstraints.startAddrAlign = 8192U;
    constraints->blConstraints.pitchAlign = 64U;
    constraints->blConstraints.heightAlign = 128U;
    constraints->blConstraints.sizeAlign = 1U;

    constraints->blSpecificConstraints.log2GobSize = 9U;
    constraints->blSpecificConstraints.log2GobsperBlockX = 0U;
    constraints->blSpecificConstraints.log2GobsperBlockY = 4U;
    constraints->blSpecificConstraints.log2GobsperBlockZ = 0U;
#else // !defined(__x86_64__)
    constraints->plConstraints.startAddrAlign = 1U;
    constraints->plConstraints.pitchAlign = 1U;
    constraints->plConstraints.heightAlign = 1U;
    constraints->plConstraints.sizeAlign = 1U;

    constraints->blConstraints.startAddrAlign = 1U;
    constraints->blConstraints.pitchAlign = 1U;
    constraints->blConstraints.heightAlign = 1U;
    constraints->blConstraints.sizeAlign = 1U;

    constraints->blSpecificConstraints.log2GobSize = 0U;
    constraints->blSpecificConstraints.log2GobsperBlockX = 0U;
    constraints->blSpecificConstraints.log2GobsperBlockY = 0U;
    constraints->blSpecificConstraints.log2GobsperBlockZ = 0U;
#endif // !defined(__x86_64__)
}

LwSciError LwSciBufReconcileOutputImgConstraints(
    LwSciBufHwConstraints *dest,
    const LwSciBufImageCommonConstraints *src)
{
    LwSciError sciErr = LwSciError_Success;
    LWSCI_FNENTRY("");

    if ((NULL == dest) || (NULL == src)) {
        LWSCI_ERR_STR("Either of the inputs is NULL\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs: Dest: StartAlign: %d PitchAlign: %d HeightAlign: %d "
        "sizeAlign: %d\n", dest->startAddrAlign, dest->pitchAlign,
        dest->heightAlign, dest->sizeAlign);
    LWSCI_INFO("Inputs: Src: StartAlign: %d PitchAlign: %d HeightAlign: %d "
        "sizeAlign: %d\n", src->startAddrAlign, src->pitchAlign,
        src->heightAlign, src->sizeAlign);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LW_SCI_BUF_RECONCILE_MAX(dest->startAddrAlign, src->startAddrAlign);
    LW_SCI_BUF_RECONCILE_MAX(dest->pitchAlign, src->pitchAlign);
    LW_SCI_BUF_RECONCILE_MAX(dest->heightAlign, src->heightAlign);
    LW_SCI_BUF_RECONCILE_MAX(dest->sizeAlign, src->sizeAlign);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    LWSCI_INFO("Outputs: Dest: StartAlign: %d PitchAlign: %d HeightAlign: %d, "
        "sizeAlign: %d\n",
               dest->startAddrAlign, dest->pitchAlign, dest->heightAlign,
               dest->sizeAlign);

    LWSCI_FNEXIT("");
    return sciErr;

}

LwSciError LwSciBufReconcileOutputBLConstraints(
    LwSciBufHwConstraints *dest,
    const LwSciBufImageBLConstraints *src)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((NULL == dest) || (NULL == src)) {
        LWSCI_ERR_STR("Either of the inputs is NULL\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs: Dest: ");
    LWSCI_INFO("log2GobSize: %d log2GobsX: %d log2GobsY: %d log2GobsZ: %d\n",
               dest->log2GobSize, dest->log2GobsperBlockX,
               dest->log2GobsperBlockY, dest->log2GobsperBlockZ);
    LWSCI_INFO("Src:log2GobSize: %d log2GobsX %d log2GobsY %d log2GobsZ %d\n",
               src->log2GobSize, src->log2GobsperBlockX,
               src->log2GobsperBlockY, src->log2GobsperBlockZ);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobSize, src->log2GobSize);
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobsperBlockX, src->log2GobsperBlockX);
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobsperBlockY, src->log2GobsperBlockY);
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobsperBlockZ, src->log2GobsperBlockZ);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    LWSCI_INFO("Outputs: Dest: \n");
    LWSCI_INFO("log2GobSize: %d log2GobsX: %d log2GobsY: %d log2GobsZ: %d\n",
               dest->log2GobSize, dest->log2GobsperBlockX,
               dest->log2GobsperBlockY, dest->log2GobsperBlockZ);

    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufReconcileIntPyramidConstraints(
    uint8_t* dest,
    const uint8_t* src)
{
    LwSciError sciErr = LwSciError_Success;

    uint8_t* destLevelCount = NULL;
    const uint8_t* srcLevelCount = NULL;

    uint32_t tmpOutputLevelCount = 0U;
    uint32_t tmpInputLevelCount = 0U;

    LWSCI_FNENTRY("");

    if ((NULL == dest) || (NULL == src)) {
        LWSCI_ERR_STR("Either of the inputs is NULL\n");
        LwSciCommonPanic();
    }

    destLevelCount = dest;
    srcLevelCount = src;

    tmpOutputLevelCount = *destLevelCount;
    tmpInputLevelCount = *srcLevelCount;

    LWSCI_INFO("Inputs: Dest: levelCount: %d ", destLevelCount);
    LWSCI_INFO("Src: levelCount: %f\n", srcLevelCount);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LW_SCI_BUF_RECONCILE_NONZERO_MIN_U(tmpOutputLevelCount, tmpInputLevelCount);
    if (tmpOutputLevelCount > (uint32_t)UINT8_MAX) {
        LWSCI_ERR_STR("Type colwersion error\n");
        sciErr = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *destLevelCount = (uint8_t) tmpOutputLevelCount;

    LWSCI_INFO("Outputs: Dest: levelCount: %d\n", destLevelCount);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufReconcileFloatPyramidConstraints(
    float* dest,
    const float* src)
{
    LwSciError sciErr = LwSciError_Success;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    float *destScaleFactor = NULL;
    const float *srcScaleFactor = NULL;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_6))

    LWSCI_FNENTRY("");

    if ((NULL == dest) || (NULL == src)) {
        LWSCI_ERR_STR("Either of the inputs is NULL\n");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    destScaleFactor = dest;
    srcScaleFactor = src;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_6))

    LWSCI_INFO("Inputs: Dest: scaleFactor: %f ", *destScaleFactor);
    LWSCI_INFO("Src: scaleFactor: %f\n", *srcScaleFactor);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LW_SCI_BUF_RECONCILE_NONZERO_MIN_F(*destScaleFactor, *srcScaleFactor);

    LWSCI_INFO("Outputs: Dest: scaleFactor: %f\n", *destScaleFactor);

    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufReconcileImgCommonConstraints(
    LwSciBufImageCommonConstraints *dest,
    const LwSciBufImageCommonConstraints *src)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((NULL == dest) || (NULL == src)) {
        LWSCI_ERR_STR("Either of the inputs is NULL\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs: Dest: StartAlign: %d PitchAlign: %d HeightAlign: %d "
        "sizeAlign: %d\n", dest->startAddrAlign, dest->pitchAlign,
        dest->heightAlign, dest->sizeAlign);
    LWSCI_INFO("Inputs: Src: StartAlign: %d PitchAlign: %d HeightAlign: %d "
        "sizeAlign: %d\n", src->startAddrAlign, src->pitchAlign,
        src->heightAlign, src->sizeAlign);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LW_SCI_BUF_RECONCILE_MAX(dest->startAddrAlign, src->startAddrAlign);
    LW_SCI_BUF_RECONCILE_MAX(dest->pitchAlign, src->pitchAlign);
    LW_SCI_BUF_RECONCILE_MAX(dest->heightAlign, src->heightAlign);
    LW_SCI_BUF_RECONCILE_MAX(dest->sizeAlign, src->sizeAlign);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    LWSCI_INFO("Outputs: Dest: StartAlign: %d PitchAlign: %d HeightAlign: %d, "
        "sizeAlign: %d\n",
               dest->startAddrAlign, dest->pitchAlign, dest->heightAlign,
               dest->sizeAlign);

    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufReconcileImageBLConstraints(
    LwSciBufImageBLConstraints *dest,
    const LwSciBufImageBLConstraints *src)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((NULL == dest) || (NULL == src)) {
        LWSCI_ERR_STR("Either of the inputs is NULL\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs: Dest: ");
    LWSCI_INFO("log2GobSize: %d log2GobsX: %d log2GobsY: %d log2GobsZ: %d\n",
               dest->log2GobSize, dest->log2GobsperBlockX,
               dest->log2GobsperBlockY, dest->log2GobsperBlockZ);
    LWSCI_INFO("Src:log2GobSize: %d log2GobsX %d log2GobsY %d log2GobsZ %d\n",
               src->log2GobSize, src->log2GobsperBlockX,
               src->log2GobsperBlockY, src->log2GobsperBlockZ);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobSize, src->log2GobSize);
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobsperBlockX, src->log2GobsperBlockX);
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobsperBlockY, src->log2GobsperBlockY);
    LW_SCI_BUF_RECONCILE_MAX(dest->log2GobsperBlockZ, src->log2GobsperBlockZ);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    LWSCI_INFO("Outputs: Dest: \n");
    LWSCI_INFO("log2GobSize: %d log2GobsX: %d log2GobsY: %d log2GobsZ: %d\n",
               dest->log2GobSize, dest->log2GobsperBlockX,
               dest->log2GobsperBlockY, dest->log2GobsperBlockZ);

    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufGetConstraints(
    LwSciBufType bufType,
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((LwSciBufType_MaxValid <= bufType) ||
        (NULL == constraints) || ((LwSciBufType_Image == bufType) && (NULL == data)))
    {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufGetConstraints\n");
        LWSCI_ERR_UINT("bufType: \n", (uint32_t)bufType);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: bufType: %u, constraints ptr: %p engineArray ptr %p, engineCount %u\n",
        bufType, engineArray, engineCount, constraints);

    if (NULL != perDataTypeConstraints[(uint32_t)bufType]) {
        err = perDataTypeConstraints[(uint32_t)bufType](chipId, engineArray, engineCount,
            constraints, data);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to get engine constraints for datatype \n",
                bufType);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufHwEngCreateIdWithInstance(
    LwSciBufHwEngName engName,
    uint32_t instance,
    int64_t* engId)
{
    LwSciError err = LwSciError_Success;
    bool isValidEngineName = false;

    LWSCI_FNENTRY("");

    isValidEngineName = LwSciBufHwEngNameValidate(engName);
    if (false == isValidEngineName) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_UINT("Input engine name invalid ", (uint32_t)engName);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (NULL == engId) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufHwEngCreateIdWithInstance\n");
        LWSCI_ERR_UINT("engName; \n", (uint32_t)engName);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: engName: %u, instance: %u, engId: %p\n", engName,
        instance, engId);

    LwSciBufHwEngCreateIdHelper(engName, instance, engId);

    LWSCI_INFO("output: engId: 0x%x\n", *engId);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufHwEngCreateIdWithoutInstance(
    LwSciBufHwEngName engName,
    int64_t* engId)
{
    LwSciError err = LwSciError_Success;
    bool isValidEngineName = false;

    LWSCI_FNENTRY("");

    isValidEngineName = LwSciBufHwEngNameValidate(engName);
    if (false == isValidEngineName) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_UINT("Input engine name invalid ", (uint32_t)engName);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (NULL == engId) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufHwEngCreateIdWithoutInstance\n");
        LWSCI_ERR_UINT("engName: \n", (uint32_t)engName);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: engName: %u, engId: %p\n", engName, engId);

    LwSciBufHwEngCreateIdHelper(engName, 0, engId);

    LWSCI_INFO("output: engId: 0x%x\n", *engId);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufHwEngGetNameFromId(
    int64_t engId,
    LwSciBufHwEngName* engName)
{
    LwSciError err = LwSciError_Success;
    bool isValidEngineName = false;
    uint32_t tmpEngName;
    LwSciBufHwEngName name = LwSciBufHwEngName_Ilwalid;

    LWSCI_FNENTRY("");

    if (NULL == engName) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufHwEngGetNameFromId\n");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: endId: 0x%x, engName: %p\n", engId, engName);

    if (engId < 0L) {
        LWSCI_ERR_INT("engId id negative: \n", (int32_t)engId);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    if (engId > INT32_MAX) {
        /* We only use the lower 32 bits of the engine id */
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    tmpEngName = (((uint32_t)engId >> LW_SCI_BUF_ENG_NAME_BIT_START) &
                    LW_SCI_BUF_ENG_NAME_BIT_MASK);

    LwSciCommonMemcpyS(&name, sizeof(name), &tmpEngName, sizeof(tmpEngName));

    isValidEngineName = LwSciBufHwEngNameValidate(name);
    if (false == isValidEngineName) {
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *engName = name;

    LWSCI_INFO("Output: engName: %u\n", *engName);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufHwEngGetInstanceFromId(
    int64_t engId,
    uint32_t* instance)
{
    LwSciError err = LwSciError_Success;

    LwSciBufHwEngName engName = LwSciBufHwEngName_Ilwalid;

    LWSCI_FNENTRY("");

    if (NULL == instance) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufHwEngGetInstanceFromId\n");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: endId: 0x%x, instance: %p\n", engId, instance);

    /* Validate that the engId is valid. Since instances aren't validated, we
     * check that we can extract an LwSciBufHwEngName. */
    err = LwSciBufHwEngGetNameFromId(engId, &engName);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *instance = (((uint32_t)engId >> LW_SCI_BUF_ENG_INSTANCE_BIT_START) &
                    LW_SCI_BUF_ENG_INSTANCE_BIT_MASK);

    LWSCI_INFO("Output: engine instance: %u\n", *instance);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufHasEngine(
    const LwSciBufHwEngine engineArray[],
    const uint64_t engineCount,
    const LwSciBufHwEngName queryEngineName,
    bool* hasEngine)
{
    LwSciError sciErr = LwSciError_Success;
    uint64_t arrayIndex = 0UL;
    bool isValidEngineName = false;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engineArray: %p, engineCount: %lu, queryEngineName: %x,"
        " hasEngine: %p", engineArray, engineCount, queryEngineName, hasEngine);

    isValidEngineName = LwSciBufHwEngNameValidate(queryEngineName);
    if ((NULL == hasEngine) || (false == isValidEngineName)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufHasEngine");
        LwSciCommonPanic();
    }

    *hasEngine = false;

    if (NULL == engineArray) {
        LWSCI_INFO("No engines set\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (arrayIndex = 0; arrayIndex < engineCount; arrayIndex++) {
        LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;

        sciErr = LwSciBufHwEngGetNameFromId(engineArray[arrayIndex].rmModuleID, &engineName);
        if (LwSciError_Success != sciErr) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (queryEngineName == engineName) {
            *hasEngine = true;
            break;
        }
    }

    LWSCI_INFO("Output: hasEngine: %s\n", *hasEngine ? "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}
