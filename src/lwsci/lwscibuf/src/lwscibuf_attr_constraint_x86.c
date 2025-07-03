/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdbool.h>
#include <string.h>

#include "lwscibuf_constraint_lib.h"
#include "lwscibuf_attr_constraint_x86.h"
#include "lwscibuf_utils.h"
#include "lwscicommon_os.h"

static LwSciError ValidateWithEngineNotSupported(
    LwSciBufAttrList reconcileList __attribute__((unused)),
    const LwSciBufOutputAttrKeyValuePair* outputAttributes __attribute__((unused)),
    const LwSciBufHwConstraints* constraints __attribute__((unused)))
{
    LwSciError err = LwSciError_NotSupported;

    LWSCI_FNENTRY("");


    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ImageValidateInternalAttrWithEngine(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    uint32_t key = 0U;
    const void* value = NULL;
    uint32_t planeCount = 0U;
    uint32_t planeNum = 0U;
    size_t i = 0UL;
    LwSciBufAttrKeyValuePair keyValPair = {0};

    LWSCI_FNENTRY("");

    keyValPair.key = LwSciBufImageAttrKey_PlaneCount;
    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &keyValPair,
                1, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not get value of key LwSciBufImageAttrKey_PlaneCount from attrlist");
        goto ret;
    }

    if (0U != keyValPair.len) {
        planeCount = *(const uint32_t*)keyValPair.value;
    } else {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR(" key not set for image datatype");
        goto ret;
    }

    LwSciBufAttrValImageLayoutType layoutType;

    keyValPair.key = LwSciBufImageAttrKey_Layout;
    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0,
                            &keyValPair, 1,
                            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not get value of key LwSciBufImageAttrKey_Layout from attrlist");
        goto ret;
    }

    layoutType =
        *(const LwSciBufAttrValImageLayoutType*)keyValPair.value;


    for (i = 0UL; i < outputAttributes->internalAttrsCount; i++) {
        key = (uint32_t)outputAttributes->internalAttrs[i].key;
        value = outputAttributes->internalAttrs[i].value;

        switch (key) {
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneGobSize):
            {
                uint32_t gobSizeConstraint = constraints->log2GobSize;
                const uint32_t* log2GobSize = (const uint32_t*)value;

                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if ((LwSciBufImage_BlockLinearType == layoutType) &&
                        (log2GobSize[planeNum] < gobSizeConstraint)) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneGobSize value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX):
            {
                uint32_t gobSizeConstraint = constraints->log2GobsperBlockX;
                const uint32_t* log2GobsperBlockX = (const uint32_t*)value;

                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if ((LwSciBufImage_BlockLinearType == layoutType) &&
                        (log2GobsperBlockX[planeNum] < gobSizeConstraint)) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY):
            {
                uint32_t gobSizeConstraint = constraints->log2GobsperBlockY;
                const uint32_t* log2GobsperBlockY = (const uint32_t*)value;

                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if ((LwSciBufImage_BlockLinearType == layoutType) &&
                        (log2GobsperBlockY[planeNum] < gobSizeConstraint)) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ):
            {
                uint32_t gobSizeConstraint = constraints->log2GobsperBlockZ;
                const uint32_t* log2GobsperBlockZ = (const uint32_t*)value;

                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if ((LwSciBufImage_BlockLinearType == layoutType) &&
                        (log2GobsperBlockZ[planeNum] < gobSizeConstraint)) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }
            default:
            {
                break;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}


static LwSciError ImageValidatePublicAttrWithEngine(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;
    uint32_t key = 0U;
    const void* value = NULL;
    uint32_t planeCount = 0U;
    uint32_t planeNum = 0U;
    size_t i = 0UL;
    LwSciBufAttrKeyValuePair keyValPair = {0};

    LWSCI_FNENTRY("");

    keyValPair.key = LwSciBufImageAttrKey_PlaneCount;
    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &keyValPair,
                1, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not get value of key LwSciBufImageAttrKey_PlaneCount from attrlist");
        goto ret;
    }

    if (0U != keyValPair.len) {
        planeCount = *(const uint32_t*)keyValPair.value;
    } else {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR(" key not set for image datatype");
        goto ret;
    }

    for (i = 0UL; i < outputAttributes->publicAttrsCount; i++) {
        key = (uint32_t)outputAttributes->publicAttrs[i].key;
        value = outputAttributes->publicAttrs[i].value;

        switch (key) {
            case ((uint32_t)LwSciBufImageAttrKey_Size):
            {
                uint64_t sizeValue = *(const uint64_t*)value;
                if (0 !=
                    sizeValue % LWSCIBUF_PREFERRED_PLANE_ALIGNMENT_CONSTRAINT) {
                    err = LwSciError_BadParameter;
                    LWSCI_ERR_STR("LwSciBufImageAttrKey_Size value in outputAttr does not satisfy engine constraint value");
                    goto ret;
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_Alignment):
            {
                uint64_t alignmentValue = *(const uint64_t*)value;
                if (0 != alignmentValue % constraints->startAddrAlign) {
                    err = LwSciError_BadParameter;
                    LWSCI_ERR_STR("LwSciBufImageAttrKey_Alignment value in outputAttr does not satisfy engine constraint value");
                    goto ret;
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneAlignedHeight):
            {
                const uint32_t* heightValue = (const uint32_t*)value;
                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if (0 != heightValue[planeNum] % constraints->heightAlign) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneAlignedHeight value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }

            case ((uint32_t)LwSciBufImageAttrKey_PlaneAlignedSize):
            {
                const uint64_t* planeSizeValue = (const uint64_t*)value;
                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if (0 != (planeSizeValue[planeNum] %
                        LWSCIBUF_PREFERRED_PLANE_ALIGNMENT_CONSTRAINT)) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneAlignedSize value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlanePitch):
            {
                const uint32_t* pitchValue = (const uint32_t*)value;
                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if (0 != pitchValue[planeNum] % constraints->pitchAlign) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlanePitch value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneOffset):
            {
                const uint64_t* planeOffsetValue = (const uint64_t*)value;
                for (planeNum = 0; planeNum < planeCount; planeNum++) {
                    if (0 != (planeOffsetValue[planeNum] %
                        LWSCIBUF_PREFERRED_PLANE_ALIGNMENT_CONSTRAINT)) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneOffset value in outputAttr does not satisfy engine constraint value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneDatatype):
            case ((uint32_t)LwSciBufImageAttrKey_PlaneChannelCount):
            case ((uint32_t)LwSciBufImageAttrKey_PlaneBitsPerPixel):
            case ((uint32_t)LwSciBufImageAttrKey_PlaneSecondFieldOffset):
            {
                break;
            }
            default:
            {
                break;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ImageValidateWithEngine(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    const LwSciBufHwConstraints* constraints)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = ImageValidatePublicAttrWithEngine(reconcileList, outputAttributes,
            constraints);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate public attributes in outputAttributes against engine constraints");
        goto ret;
    }

    err = ImageValidateInternalAttrWithEngine(reconcileList, outputAttributes,
            constraints);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate internal attributes in outputAttributes against engine constraints");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

typedef LwSciError (*TypeValidateOutputAttrWithEngine)(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes,
    const LwSciBufHwConstraints* constraints);

static const TypeValidateOutputAttrWithEngine
    typeValidateWithEngineMap[LwSciBufType_MaxValid] = {
    [LwSciBufType_General] = NULL,
    [LwSciBufType_RawBuffer] = ValidateWithEngineNotSupported,
    [LwSciBufType_Image] = ImageValidateWithEngine,
    [LwSciBufType_Tensor] = ValidateWithEngineNotSupported,
    [LwSciBufType_Array] = ValidateWithEngineNotSupported,
    [LwSciBufType_Pyramid] = ValidateWithEngineNotSupported,
};

static LwSciError ValidateOutputAttrWithEngine(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;
    size_t numBufTypes = 0U, index = 0U, engineCount = 0U;
    LwSciBufType bufType = LwSciBufType_MaxValid;
    const LwSciBufType* bufTypePtr = NULL;
    TypeValidateOutputAttrWithEngine typeValidation = NULL;
    LwSciBufHwConstraints constraints = {0};
    LwSciBufAttrKeyValuePair keyValPair = {0};
    LwSciBufInternalAttrKeyValuePair internalKeyValPair = {0};
    const LwSciBufHwEngine* engineArray = NULL;
    LwSciBufAttrValImageLayoutType imageLayout = LwSciBufImage_PitchLinearType;

    LWSCI_FNENTRY("");

    /* get buffer type */
    err = LwSciBufAttrListGetDataTypes(reconcileList, &bufTypePtr,
            &numBufTypes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed");
        goto ret;
    }

    /* get HW engines operating on the buffer */
    internalKeyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;
    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &internalKeyValPair,
            1, LwSciBufAttrKeyType_Internal, true);
    if (LwSciError_Success != err) {
         LWSCI_ERR_STR("LwSciBufAttrListGetInternalAttrs failed");
        goto ret;
    }

    engineArray = (const LwSciBufHwEngine*)internalKeyValPair.value;
    engineCount = internalKeyValPair.len / sizeof(LwSciBufHwEngine);

    /* get consolidated HW constraints for engines corresponding to all the
     * specified datatypes
     */
    for (index = 0UL; index < numBufTypes; index++) {

        if ((LwSciBufType_Image == bufTypePtr[index]) ||
            (LwSciBufType_Pyramid == bufTypePtr[index])) {
            /* we need to pass image layout to LwSciBufGetConstraints function
             * in order to get either pitchlinear of blocklinear constraints
             */
            keyValPair.key = LwSciBufImageAttrKey_Layout;
            err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &keyValPair,
                    1, LwSciBufAttrKeyType_Public, true);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("Could not get value of key LwSciBufImageAttrKey_Layout from attrlist");
                goto ret;
            }

            if (0U != keyValPair.len) {
                imageLayout =
                    *(const LwSciBufAttrValImageLayoutType*)keyValPair.value;
            } else {
                err = LwSciError_BadParameter;
                LWSCI_ERR_STR("LwSciBufImageAttrKey_Layout key not set for image datatype");
                goto ret;
            }
        }

        err = LwSciBufGetConstraints(bufTypePtr[index], LWRM_T194_ID,
                engineArray, (uint32_t)engineCount,
                &constraints, (void *)&imageLayout);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufGetConstraints failed");
            goto ret;
        }
    }

    /* Verify outputAttrs against each buffer type combined Engine constraints
     */
    for (index = 0; index < numBufTypes; index++) {

        bufType = bufTypePtr[index];
        typeValidation = typeValidateWithEngineMap[bufType];
        if (NULL == typeValidation) {
            LWSCI_ERR_UINT("No validation mapping for buffer type", bufType);
            LwSciCommonPanic();
        }

        err = typeValidation(reconcileList, outputAttributes, &constraints);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to apply constraints for buffer type",
            bufType);
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ValidateWithReconcileListNotSupported(
    LwSciBufAttrList reconcileList __attribute__((unused)),
    const LwSciBufOutputAttrKeyValuePair* outputAttributes __attribute__((unused)))
{
    LwSciError err = LwSciError_NotSupported;

    LWSCI_FNENTRY("");


    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ImageValidateInternalAttrWithReconcileList(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair pubKeyValPair[2] = {0};
    LwSciBufInternalAttrKeyValuePair intKeyValPair[4] = {0};
    size_t pubAttrCount = 0U;
    size_t intAttrCount = 0U;
    const uint32_t* log2GobSize;
    const uint32_t* log2GobsperBlockX;
    const uint32_t* log2GobsperBlockY;
    const uint32_t* log2GobsperBlockZ;
    const uint32_t* planeCount = NULL;
    const LwSciBufAttrValImageLayoutType* layoutType;
    uint32_t key = 0U;
    const void* value = NULL;
    size_t i = 0UL;

    LWSCI_FNENTRY("");

    pubKeyValPair[pubAttrCount++].key = LwSciBufImageAttrKey_PlaneCount;
    pubKeyValPair[pubAttrCount++].key = LwSciBufImageAttrKey_Layout;

    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &pubKeyValPair,
                pubAttrCount, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not get value of keys from reconciled attrlist");
        goto ret;
    }

    for (i = 0U; i < pubAttrCount; i++) {
        if (0U == pubKeyValPair[i].len) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_UINT(" key not set ", pubKeyValPair[i].key);
            goto ret;
        }
    }

    pubAttrCount = 0;
    planeCount = (const uint32_t*)pubKeyValPair[pubAttrCount++].value;
    layoutType = (const LwSciBufAttrValImageLayoutType*)
                    pubKeyValPair[pubAttrCount++].value;


    intKeyValPair[intAttrCount++].key =
                        LwSciBufInternalImageAttrKey_PlaneGobSize;
    intKeyValPair[intAttrCount++].key =
                        LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX;
    intKeyValPair[intAttrCount++].key =
                        LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY;
    intKeyValPair[intAttrCount++].key =
                        LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ;

    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &intKeyValPair,
                intAttrCount, LwSciBufAttrKeyType_Internal, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not get value of keys from reconciled attrlist");
        goto ret;
    }

    for (i = 0U; i < intAttrCount; i++) {
        if (0U == intKeyValPair[i].len) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_UINT(" key not set ", intKeyValPair[i].key);
            goto ret;
        }
    }

    intAttrCount = 0;
    log2GobSize = (const uint32_t*)intKeyValPair[intAttrCount++].value;
    log2GobsperBlockX = (const uint32_t*)intKeyValPair[intAttrCount++].value;
    log2GobsperBlockY = (const uint32_t*)intKeyValPair[intAttrCount++].value;
    log2GobsperBlockZ = (const uint32_t*)intKeyValPair[intAttrCount++].value;

    for (i = 0UL; i < outputAttributes->internalAttrsCount; i++) {
        key = (uint32_t)outputAttributes->internalAttrs[i].key;
        value = outputAttributes->internalAttrs[i].value;

        switch (key) {
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneGobSize):
            {
                const uint32_t* outputGobSize = (const uint32_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if ((LwSciBufImage_BlockLinearType == *layoutType) &&
                        (outputGobSize[idx] < log2GobSize[idx])) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneGobSize value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX):
            {
                const uint32_t* outputGobBlockX = (const uint32_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if ((LwSciBufImage_BlockLinearType == *layoutType) &&
                        (outputGobBlockX[idx] < log2GobsperBlockX[idx])) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY):
            {
                const uint32_t* outputGobBlockY = (const uint32_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if ((LwSciBufImage_BlockLinearType == *layoutType) &&
                        (outputGobBlockY[idx] < log2GobsperBlockY[idx])) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ):
            {
                const uint32_t* outputGobBlockZ = (const uint32_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if ((LwSciBufImage_BlockLinearType == *layoutType) &&
                        (outputGobBlockZ[idx] < log2GobsperBlockZ[idx])) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            default:
            {
                break;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ImageValidatePublicAttrWithReconcileList(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair[10] = {0};
    size_t attrCount = 0U;
    const uint64_t* size = NULL;
    const uint64_t* alignment = NULL;
    const uint32_t* bpp = NULL;
    const uint64_t* planeOffset = NULL;
    const uint64_t* planeSize = NULL;
    const uint32_t* planePitch = NULL;
    const uint32_t* planeHeight = NULL;
    const LwSciBufAttrValDataType* dataType = NULL;
    const uint8_t* channelCount = NULL;
    const uint32_t* planeCount = NULL;
    uint32_t key = 0U;
    const void* value = NULL;
    size_t i = 0UL;

    LWSCI_FNENTRY("");

    keyValPair[attrCount++].key = LwSciBufImageAttrKey_Size;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_Alignment;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneBitsPerPixel;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneOffset;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneDatatype;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneChannelCount;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlanePitch;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneAlignedHeight;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneCount;
    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneAlignedSize;

    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &keyValPair,
                attrCount, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not get value of keys from reconciled attrlist");
        goto ret;
    }

    for (i = 0U; i < attrCount; i++) {
        if (0U == keyValPair[i].len) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_UINT(" key not set ", keyValPair[i].key);
            goto ret;
        }
    }

    attrCount = 0;
    size = (const uint64_t*)keyValPair[attrCount++].value;
    alignment = (const uint64_t*)keyValPair[attrCount++].value;
    bpp = (const uint32_t*)keyValPair[attrCount++].value;
    planeOffset = (const uint64_t*)keyValPair[attrCount++].value;
    dataType = (const LwSciBufAttrValDataType*)keyValPair[attrCount++].value;
    channelCount = (const uint8_t*)keyValPair[attrCount++].value;
    planePitch = (const uint32_t*)keyValPair[attrCount++].value;
    planeHeight = (const uint32_t*)keyValPair[attrCount++].value;
    planeCount = (const uint32_t*)keyValPair[attrCount++].value;
    planeSize = (const uint64_t*)keyValPair[attrCount++].value;

    for (i = 0UL; i < outputAttributes->publicAttrsCount; i++) {
        key = (uint32_t)outputAttributes->publicAttrs[i].key;
        value = outputAttributes->publicAttrs[i].value;

        switch (key) {
            case ((uint32_t)LwSciBufImageAttrKey_Size):
            {
                const uint64_t* outputAttrSize = (const uint64_t*)value;
                if (*outputAttrSize < *size) {
                    err = LwSciError_BadParameter;
                    LWSCI_ERR_STR("LwSciBufImageAttrKey_Size value in outputAttr does not satisfy reconciled list value");
                    goto ret;
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_Alignment):
            {
                const uint64_t* outputAttrAlign = (const uint64_t*)value;
                if (*outputAttrAlign < *alignment) {
                    err = LwSciError_BadParameter;
                    LWSCI_ERR_STR("LwSciBufImageAttrKey_Alignment value in outputAttr does not satisfy reconciled list value");
                    goto ret;
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneBitsPerPixel):
            {
                const uint32_t* outputAttrBpp = (const uint32_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if (outputAttrBpp[idx] != bpp[idx]) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneBitsPerPixel value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneDatatype):
            {
                const LwSciBufAttrValDataType* outputAttrDataType = (const LwSciBufAttrValDataType*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if (outputAttrDataType[idx] != dataType[idx]) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneDatatype value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneChannelCount):
            {
                const uint8_t* outputAttrChannelCount = (const uint8_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if (outputAttrChannelCount[idx] != channelCount[idx]) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneChannelCount value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlanePitch):
            {
                const uint32_t* outputAttrPlanePitch = (const uint32_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if (outputAttrPlanePitch[idx] < planePitch[idx]) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlanePitch value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneAlignedHeight):
            {
                const uint32_t* outputAttrPlaneHeight = (const uint32_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if (outputAttrPlaneHeight[idx] < planeHeight[idx]) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneAlignedHeight value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneOffset):
            {
                const uint64_t* outputAttrPlaneOffset = (const uint64_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if (outputAttrPlaneOffset[idx] < planeOffset[idx]) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneOffset value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            case ((uint32_t)LwSciBufImageAttrKey_PlaneAlignedSize):
            {
                const uint64_t* outputAttrPlaneSize = (const uint64_t*)value;
                for (uint32_t idx = 0U; idx < *planeCount; idx++) {
                    if (outputAttrPlaneSize[idx] < planeSize[idx]) {
                        err = LwSciError_BadParameter;
                        LWSCI_ERR_STR("LwSciBufImageAttrKey_PlaneAlignedSize value in outputAttr does not satisfy reconciled list value");
                        goto ret;
                    }
                }
                break;
            }
            default:
            {
                break;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ImageValidateWithReconcileList(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = ImageValidatePublicAttrWithReconcileList(reconcileList,
                                outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate public attributes in outputAttributes against reconciled attribute list");
        goto ret;
    }

    err = ImageValidateInternalAttrWithReconcileList(reconcileList,
                                outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate internal attributes in outputAttributes against reconciled attribute list");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

typedef LwSciError (*TypeValidateOutputerAttrWithReconcileList)(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes);

static const TypeValidateOutputerAttrWithReconcileList
    typeValidateWithReconcileListMap[LwSciBufType_MaxValid] = {
    [LwSciBufType_General] = NULL,
    [LwSciBufType_RawBuffer] = ValidateWithReconcileListNotSupported,
    [LwSciBufType_Image] = ImageValidateWithReconcileList,
    [LwSciBufType_Tensor] = ValidateWithReconcileListNotSupported,
    [LwSciBufType_Array] = ValidateWithReconcileListNotSupported,
    [LwSciBufType_Pyramid] = ValidateWithReconcileListNotSupported,
};


static LwSciError ValidateOutputAttrWithReconciledList(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;
    size_t numBufTypes = 0U, index = 0U;
    const LwSciBufType* bufTypePtr = NULL;
    TypeValidateOutputerAttrWithReconcileList attrValidation = NULL;

    LWSCI_FNENTRY("");

    /* get buffer type */
    err = LwSciBufAttrListGetDataTypes(reconcileList, &bufTypePtr,
            &numBufTypes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed");
        goto ret;
    }

    for (index = 0UL; index < numBufTypes; index++) {
        attrValidation = typeValidateWithReconcileListMap[bufTypePtr[index]];
        if (NULL == attrValidation) {
            LWSCI_ERR_UINT("No validation mapping for buffer type", bufTypePtr[index]);
            LwSciCommonPanic();
        }

        err = attrValidation(reconcileList, outputAttributes);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to validate outputAttr against reconciled attr list for buffer type",
            bufTypePtr[index]);
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ImageValidateWithComputation(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair[2] = {0};
    size_t attrCount = 0U;
    const uint64_t* size = NULL;
    uint64_t imageSize = 0UL;
    const uint64_t* planeSize = NULL;
    const uint32_t* planeCount = NULL;
    const uint32_t* planePitch = NULL;
    const uint32_t* planeHeight = NULL;
    uint32_t key = 0U;
    const void* value = NULL;
    size_t i = 0UL;

    LWSCI_FNENTRY("");

    keyValPair[attrCount++].key = LwSciBufImageAttrKey_PlaneCount;

    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0, &keyValPair,
                attrCount, LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not get value of keys from reconciled attrlist");
        goto ret;
    }

    for (i = 0U; i < attrCount; i++) {
        if (0U == keyValPair[i].len) {
            err = LwSciError_BadParameter;
            LWSCI_ERR_UINT(" key not set ", keyValPair[i].key);
            goto ret;
        }
    }

    attrCount = 0U;
    planeCount = (const uint32_t*)keyValPair[attrCount++].value;

    for (i = 0UL; i < outputAttributes->publicAttrsCount; i++) {
        key = (uint32_t)outputAttributes->publicAttrs[i].key;
        value = outputAttributes->publicAttrs[i].value;

        switch (key) {
            case (LwSciBufImageAttrKey_Size):
            {
                size = (const uint64_t*)value;
                break;
            }
            case (LwSciBufImageAttrKey_PlaneAlignedSize):
            {
                planeSize = (const uint64_t*)value;
                break;
            }
            case (LwSciBufImageAttrKey_PlanePitch):
            {
                planePitch = (const uint32_t*)value;
                break;
            }
            case (LwSciBufImageAttrKey_PlaneAlignedHeight):
            {
                planeHeight = (const uint32_t*)value;
                break;
            }
            default:
            {
                break;
            }
        }
    }

    for (uint32_t idx = 0U; idx < *planeCount; idx++) {
        if (planeSize[idx] < planePitch[idx]*planeHeight[idx]) {
            LWSCI_ERR_STR("plane size doesn't match pitch and height in outputAttrs");
            err = LwSciError_BadParameter;
            goto ret;
        }

        imageSize += planeSize[idx];
    }

    if (*size < imageSize) {
        LWSCI_ERR_STR("image size doesn't match total plane size in outputAttrs");
        err = LwSciError_BadParameter;
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ValidateWithComputationNotSupported(
    LwSciBufAttrList reconcileList __attribute__((unused)),
    const LwSciBufOutputAttrKeyValuePair* outputAttributes __attribute__((unused)))
{
    LwSciError err = LwSciError_NotSupported;

    LWSCI_FNENTRY("");


    LWSCI_FNEXIT("");
    return err;
}

typedef LwSciError (*TypeValidateOutputerAttrWithComputation)(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes);

static const TypeValidateOutputerAttrWithReconcileList
    typeValidateWithComputationMap[LwSciBufType_MaxValid] = {
    [LwSciBufType_General] = NULL,
    [LwSciBufType_RawBuffer] = ValidateWithComputationNotSupported,
    [LwSciBufType_Image] = ImageValidateWithComputation,
    [LwSciBufType_Tensor] = ValidateWithComputationNotSupported,
    [LwSciBufType_Array] = ValidateWithComputationNotSupported,
    [LwSciBufType_Pyramid] = ValidateWithComputationNotSupported,
};

static LwSciError ValidateOutputAttrWithComputation(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;
    size_t numBufTypes = 0U, index = 0U;
    const LwSciBufType* bufTypePtr = NULL;
    TypeValidateOutputerAttrWithComputation attrValidation = NULL;

    LWSCI_FNENTRY("");

    /* get buffer type */
    err = LwSciBufAttrListGetDataTypes(reconcileList, &bufTypePtr,
            &numBufTypes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed");
        goto ret;
    }

    for (index = 0UL; index < numBufTypes; index++) {
        attrValidation = typeValidateWithComputationMap[bufTypePtr[index]];
        if (NULL == attrValidation) {
            LWSCI_ERR_UINT("No validation mapping for buffer type", bufTypePtr[index]);
            LwSciCommonPanic();
        }

        err = attrValidation(reconcileList, outputAttributes);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to validate outputAttr against reconciled attr list for buffer type",
            bufTypePtr[index]);
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ImageSetPrivateAttr(
    LwSciBufAttrList reconcileList)
{
    LwSciError err = LwSciError_Success;
    size_t attrCount = 0U;
    LwSciBufAttrKeyValuePair ipImagePublicKeyValPair[2];
    LwSciBufPrivateAttrKeyValuePair opImagePrivateKeyValPair[2];

    LWSCI_FNENTRY("");

    ipImagePublicKeyValPair[attrCount++].key = LwSciBufImageAttrKey_Size;
    ipImagePublicKeyValPair[attrCount++].key = LwSciBufImageAttrKey_Alignment;

    err = LwSciBufAttrListCommonGetAttrs(reconcileList, 0,
            &ipImagePublicKeyValPair, attrCount, LwSciBufAttrKeyType_Public,
            true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get image size.\n");
        goto ret;
    }

    attrCount = 0UL;
    opImagePrivateKeyValPair[attrCount].key = LwSciBufPrivateAttrKey_Size;
    opImagePrivateKeyValPair[attrCount].value = ipImagePublicKeyValPair[0].value;
    opImagePrivateKeyValPair[attrCount].len = sizeof(uint64_t);
    attrCount++;

    opImagePrivateKeyValPair[attrCount].key = LwSciBufPrivateAttrKey_Alignment;
    opImagePrivateKeyValPair[attrCount].value = ipImagePublicKeyValPair[1].value;
    opImagePrivateKeyValPair[attrCount].len = sizeof(uint64_t);

    err = LwSciBufAttrListCommonSetAttrs(reconcileList, 0,
            &opImagePrivateKeyValPair, attrCount, LwSciBufAttrKeyType_Private,
            true, false);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to set private keys.\n");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;

}

static LwSciError SetPrivateAttrNotSupported(
    LwSciBufAttrList reconcileList __attribute__((unused)))
{
    LwSciError err = LwSciError_NotSupported;

    LWSCI_FNENTRY("");


    LWSCI_FNEXIT("");
    return err;
}

typedef LwSciError (*TypeSetPrivateAttr)(
    LwSciBufAttrList reconcileList);

static const TypeSetPrivateAttr
    typeSetPrivateAttrMap[LwSciBufType_MaxValid] = {
    [LwSciBufType_General] = NULL,
    [LwSciBufType_RawBuffer] = SetPrivateAttrNotSupported,
    [LwSciBufType_Image] = ImageSetPrivateAttr,
    [LwSciBufType_Tensor] = SetPrivateAttrNotSupported,
    [LwSciBufType_Array] = SetPrivateAttrNotSupported,
    [LwSciBufType_Pyramid] = SetPrivateAttrNotSupported,
};

static LwSciError SetOutputAttr(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err;
    size_t numBufTypes = 0U, index = 0U;
    const LwSciBufType* bufTypePtr = NULL;
    TypeSetPrivateAttr setPrivateAttr = NULL;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListCommonSetAttrs(reconcileList, 0,
            outputAttributes->publicAttrs, outputAttributes->publicAttrsCount,
            LwSciBufAttrKeyType_Public, true, false);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to set public attributes");
        goto ret;
    }

    err = LwSciBufAttrListCommonSetAttrs(reconcileList, 0,
            outputAttributes->internalAttrs,
            outputAttributes->internalAttrsCount,
            LwSciBufAttrKeyType_Internal, true, false);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to set internal attributes");
        goto ret;
    }

    /* get buffer type */
    err = LwSciBufAttrListGetDataTypes(reconcileList, &bufTypePtr,
            &numBufTypes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed");
        goto ret;
    }

    for (index = 0UL; index < numBufTypes; index++) {
        setPrivateAttr = typeSetPrivateAttrMap[bufTypePtr[index]];
        if (NULL == setPrivateAttr) {
            LWSCI_ERR_UINT("No  mapping for buffer type", bufTypePtr[index]);
            LwSciCommonPanic();
        }

        err = setPrivateAttr(reconcileList);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to set private attr for buffer type",
            bufTypePtr[index]);
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufValidateOutputAttrData(
    LwSciBufAttrList reconcileList,
    const LwSciBufOutputAttrKeyValuePair* outputAttributes)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: reconcileList %p", reconcileList);

    /* Validate attribute list */
    err = LwSciBufAttrListValidate(reconcileList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate reconcileList.");
        goto ret;
    }

    err = ValidateOutputAttrWithEngine(reconcileList, outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate output attrs against engine constraints.");
        err = LwSciError_BadParameter;
        goto ret;
    }

    err = ValidateOutputAttrWithReconciledList(reconcileList, outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate output attrs against reconciled list.");
        err = LwSciError_BadParameter;
        goto ret;
    }

    err = ValidateOutputAttrWithComputation(reconcileList, outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate output attrs against itself.");
        err = LwSciError_BadParameter;
        goto ret;
    }

    err = SetOutputAttr(reconcileList, outputAttributes);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set output attrs in reconciled list.");
        err = LwSciError_BadParameter;
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
