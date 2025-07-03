/*
 * lwscibuf_constraint_lib_x86.c
 *
 * Constraint Library to get hardware constraints
 *
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include <string.h>

#include "lwscibuf_constraint_lib_priv.h"

#include "lwscibuf_attr_mgmt.h"
/* Pascal Series
 * - GP104 (GTX 1080, Lwdqro P4000)
 */
#include "lwscibuf_gp_constraints.h"
/* Volta Series
 * - GV100 (Tesla V100)
 */
#include "lwscibuf_gv_constraints.h"
/* Turing Series
 * - TU102 (RTX 6000, Tesla T40)
 * - TU104 (PG189-600, VdChip RTX 2080, Lwdqro RTX 4000)
 */
#include "lwscibuf_tu_constraints.h"
/* Amphere Series
 * TODO Add data from POR
 */
#include "lwscibuf_ga_constraints.h"
#include "lwscilog.h"

static LwSciError LwSciBufGetImageConstraints(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data);

static LwSciError LwSciBufGetArrayConstraints(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data);

static LwSciError LwSciBufGetImagePyramidConstraints(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data);

static const LwSciBufConstraintFvt gpuFvtTable[] =  {
    {
        { .gpuArch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 },
        LwSciBufGetGPImageConstraints,
        LwSciBufGetGPArrayConstraints,
        LwSciBufGetGPImagePyramidConstraints,
    }, {
        { .gpuArch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100 },
        LwSciBufGetGVImageConstraints,
        LwSciBufGetGVArrayConstraints,
        LwSciBufGetGVImagePyramidConstraints,
    }, {
        { .gpuArch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_TU100 },
        LwSciBufGetTUImageConstraints,
        LwSciBufGetTUArrayConstraints,
        LwSciBufGetTUImagePyramidConstraints,
    }, {
        { .gpuArch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 },
        LwSciBufGetGAImageConstraints,
        LwSciBufGetGAArrayConstraints,
        LwSciBufGetGAImagePyramidConstraints,
    },
};

const LwSciBufDataTypeConstraints
        perDataTypeConstraints[LwSciBufType_MaxValid] = {
        [LwSciBufType_Image]    = LwSciBufGetImageConstraints,
        [LwSciBufType_Array]    = LwSciBufGetArrayConstraints,
        [LwSciBufType_Pyramid]  = LwSciBufGetImagePyramidConstraints,
};

static LwSciError LwSciBufGetGpuFvt(
    uint32_t gpuArch,
    const LwSciBufConstraintFvt** fvt)
{
    LwSciError sciErr = LwSciError_Success;
    uint64_t chipcount = sizeof(gpuFvtTable)/
                        sizeof(LwSciBufConstraintFvt);
    uint64_t var = 0;

    LWSCI_FNENTRY("");
    LWSCI_INFO("Inputs: fvt: %p \n", fvt);

    if (fvt == NULL) {
        LWSCI_ERR_STR("Fvt pointer is NULL\n");
        LwSciCommonPanic();
    }
    *fvt = NULL;

    for (var = 0; var < chipcount; var++) {
        if (gpuFvtTable[var].hwid.gpuArch == gpuArch) {
            *fvt = &gpuFvtTable[var];
            goto ret;
         }
    }

    LWSCI_ERR_STR("Chipid not found in the GPU FVT\n");
    sciErr = LwSciError_NotSupported;

ret:
    LWSCI_INFO("Outputs: *fvt: %p sciErr: %d \n", *fvt, sciErr);
    LWSCI_FNEXIT("");
    return (sciErr);
}

static void LwSciBufCopyDefaultImageConstraints(
    LwSciBufImageConstraints* imageConstraints,
    size_t imageConstraintsLen)
{
    LwSciBufImageConstraints imageDefaultConstraints = { 0 };

    LwSciBufGetDefaultImageConstraints(&imageDefaultConstraints);

    LwSciCommonMemcpyS(imageConstraints, imageConstraintsLen,
                        &imageDefaultConstraints,
                        sizeof(imageDefaultConstraints));
}

static LwSciError LwSciBufGetImageConstraints(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data)
{
    LwSciError sciErr = LwSciError_Success;

    const LwSciBufConstraintFvt* gpufvt = NULL;
    LwSciBufImageConstraints imageConstraints = {0};
    LwSciBufImageConstraints engineConstraints = {0};
    LwSciBufAttrValImageLayoutType imageLayout = LwSciBufImage_PitchLinearType;
    uint32_t id = 0;

    (void)chipId;

    LWSCI_FNENTRY("");

    if (constraints == NULL || data == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufGetImageConstraints\n");
        LWSCI_ERR("constraints ptr: %p, data %p\n", constraints, data);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Inputs: chipId: %d engineCount: %d, constraints ptr: %p, data %p\n",
                chipId, engineCount, constraints, data);

    imageLayout = *(const LwSciBufAttrValImageLayoutType*)data;

    LwSciBufCopyDefaultImageConstraints(&imageConstraints, sizeof(imageConstraints));

    for (id = 0; id < engineCount; id++) {
        if(engineArray[id].engNamespace != LwSciBufHwEngine_ResmanNamespaceId) {
            LWSCI_ERR("User specified LwSciBufHwEngine_TegraNamespaceId on x86! Failed to get fvt.\n");
            sciErr = LwSciError_BadParameter;
            goto ret;
        } else {
            sciErr = LwSciBufGetGpuFvt(engineArray[id].rev.gpu.arch, &gpufvt);
            if (sciErr != LwSciError_Success) {
                LWSCI_ERR("Failed to get GPU FVT\n");
                goto ret;
            }

            if (gpufvt->getImageConstraints == NULL) {
                LWSCI_ERR("getImageConstraints fvt entry is NULL\n");
                sciErr = LwSciError_NotSupported;
                goto ret;
            }
            sciErr = gpufvt->getImageConstraints(engineArray[id],
                        &engineConstraints);
            if (sciErr != LwSciError_Success) {
                LWSCI_ERR("Failed to get GPU image constraints\n");
                goto ret;
            }
        }

        if (imageLayout == LwSciBufImage_PitchLinearType) {
            LwSciBufReconcileImgCommonConstraints(
                &imageConstraints.plConstraints,
                &engineConstraints.plConstraints);
        } else if (imageLayout == LwSciBufImage_BlockLinearType) {
            LwSciBufReconcileImgCommonConstraints(
                &imageConstraints.blConstraints,
                &engineConstraints.blConstraints);
            LwSciBufReconcileImageBLConstraints(
            &imageConstraints.blSpecificConstraints,
            &engineConstraints.blSpecificConstraints);
        } else {
            sciErr = LwSciError_NotSupported;
            LWSCI_ERR("image layout type %u not supported\n", imageLayout);
            goto ret;
        }
    }

    /* Reconcile image specific constraints with output constraints */

    if (imageLayout == LwSciBufImage_PitchLinearType) {
        /* Reconcile common constraints for pitchlinear */
        LW_SCI_BUF_RECONCILE_MAX(constraints->startAddrAlign,
            imageConstraints.plConstraints.startAddrAlign);
        LW_SCI_BUF_RECONCILE_MAX(constraints->pitchAlign,
            imageConstraints.plConstraints.pitchAlign);
        LW_SCI_BUF_RECONCILE_MAX(constraints->heightAlign,
            imageConstraints.plConstraints.heightAlign);
        LW_SCI_BUF_RECONCILE_MAX(constraints->sizeAlign,
            imageConstraints.plConstraints.sizeAlign);
    } else if (imageLayout == LwSciBufImage_BlockLinearType) {
        /* Reconcile common constraints for blocklinear */
        LW_SCI_BUF_RECONCILE_MAX(constraints->startAddrAlign,
            imageConstraints.blConstraints.startAddrAlign);
        LW_SCI_BUF_RECONCILE_MAX(constraints->pitchAlign,
            imageConstraints.blConstraints.pitchAlign);
        LW_SCI_BUF_RECONCILE_MAX(constraints->heightAlign,
            imageConstraints.blConstraints.heightAlign);
        LW_SCI_BUF_RECONCILE_MAX(constraints->sizeAlign,
            imageConstraints.blConstraints.sizeAlign);

        /* Reconcile blocklinear specific constraints */
        LW_SCI_BUF_RECONCILE_MAX(constraints->log2GobSize,
            imageConstraints.blSpecificConstraints.log2GobSize);
        LW_SCI_BUF_RECONCILE_MAX(constraints->log2GobsperBlockX,
            imageConstraints.blSpecificConstraints.log2GobsperBlockX);
        LW_SCI_BUF_RECONCILE_MAX(constraints->log2GobsperBlockY,
            imageConstraints.blSpecificConstraints.log2GobsperBlockY);
        LW_SCI_BUF_RECONCILE_MAX(constraints->log2GobsperBlockZ,
            imageConstraints.blSpecificConstraints.log2GobsperBlockZ);
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR("image layout type %u not supported\n", imageLayout);
        goto ret;
    }

    LWSCI_INFO("Output: constraints: %p\n", constraints);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufGetArrayConstraints(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data)
{
    LwSciError sciErr = LwSciError_Success;

    const LwSciBufConstraintFvt* gpufvt = NULL;
    LwSciBufArrayConstraints arrayConstraints = {0};
    LwSciBufArrayConstraints engineConstraints = {0};
    uint32_t id = 0;

    (void)chipId;
    (void)data;

    LWSCI_FNENTRY("");

    if (constraints == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufGetArrayConstraints\n");
        LWSCI_ERR("constraints ptr: %p", constraints);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Inputs: chipId: %d engineCount: %d, constraints ptr: %p\n",
                chipId, engineCount, constraints);

    LwSciCommonMemcpyS(&arrayConstraints, sizeof(arrayConstraints),
                        &arrayDefaultConstraints,
                        sizeof(arrayDefaultConstraints));

    for (id = 0U; id < engineCount; id++) {
        if(engineArray[id].engNamespace != LwSciBufHwEngine_ResmanNamespaceId) {
            LWSCI_ERR("User specified LwSciBufHwEngine_TegraNamespaceId on x86! Failed to get fvt.\n");
            goto ret;
        } else {
            sciErr = LwSciBufGetGpuFvt(engineArray[id].rev.gpu.arch, &gpufvt);
            if (sciErr != LwSciError_Success) {
                LWSCI_ERR("Failed to get GPU fvt\n");
                goto ret;
            }

            if (gpufvt->getArrayConstraints == NULL) {
                LWSCI_ERR("getArrayConstraints fvt entry is NULL\n");
                sciErr = LwSciError_NotSupported;
                goto ret;
            }
            sciErr = gpufvt->getArrayConstraints(engineArray[id],
                                                &engineConstraints);
            if (sciErr != LwSciError_Success) {
                LWSCI_ERR("Failed to get GPU array constraints\n");
                goto ret;
            }
        }

        LW_SCI_BUF_RECONCILE_MAX(arrayConstraints.startAddrAlign,
            engineConstraints.startAddrAlign);
        LW_SCI_BUF_RECONCILE_MAX(arrayConstraints.dataAlign,
            engineConstraints.dataAlign);
    }

    /* Reconcile array constraints with output constraints */
    LW_SCI_BUF_RECONCILE_MAX(constraints->startAddrAlign,
        arrayConstraints.startAddrAlign);
    LW_SCI_BUF_RECONCILE_MAX(constraints->dataAlign,
        arrayConstraints.dataAlign);

    LWSCI_INFO("Output: constraints: %p\n", constraints);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufGetImagePyramidConstraints(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data)
{
    const LwSciBufConstraintFvt* gpufvt = NULL;
    LwSciError sciErr = LwSciError_Success;
    LwSciBufImagePyramidConstraints imagePyramidConstraints = {.0F, 0U};
    LwSciBufImagePyramidConstraints engineConstraints = {.0F, 0U};
    uint32_t id = 0;

    LWSCI_FNENTRY("");

    if (constraints == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufGetImagePyramidConstraints\n");
        LWSCI_ERR("constraints ptr: %p\n", constraints);
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    LWSCI_INFO("Inputs: chipId: %d engineCount: %d,constraint ptr: %p\n",
        chipId, engineCount, constraints);

    /* Buffer type Pyramid needs Image constraints as well.
     * In this step, we 1st compute Hw Image constraints and
     *  store in output variable.
     * After ImageConstraints, we get Pyramid constraints and store in same
     *  output variable without modifying ImageConstraints
     */
    sciErr = LwSciBufGetImageConstraints(chipId, engineArray, engineCount,
                constraints, data);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR("Failed to get image contraints for buffer type pyramid\n");
        goto ret;
    }

    LwSciCommonMemcpyS(&imagePyramidConstraints, sizeof(imagePyramidConstraints),
                        &imgPyramidDefaultConstraints,
                        sizeof(imgPyramidDefaultConstraints));

    for (id = 0; id < engineCount; id++) {
        if(engineArray[id].engNamespace != LwSciBufHwEngine_ResmanNamespaceId) {
            LWSCI_ERR("User specified LwSciBufHwEngine_TegraNamespaceId on x86! Failed to get fvt.\n");
            goto ret;
        } else {
            sciErr = LwSciBufGetGpuFvt(engineArray[id].rev.gpu.arch, &gpufvt);
            if (sciErr != LwSciError_Success) {
                LWSCI_ERR("Failed to get pyramid constraints\n");
                goto ret;
            }

            if (gpufvt->getImagePyramidConstraints == NULL) {
                LWSCI_ERR("getImagePyramidConstraints fvt entry is NULL\n");
                sciErr = LwSciError_NotSupported;
                goto ret;
            }

            sciErr = gpufvt->getImagePyramidConstraints(engineArray[id],
                &engineConstraints);
            if (sciErr != LwSciError_Success) {
                LWSCI_ERR("Failed to get pyramid constraints\n");
                goto ret;
            }
        }

        LW_SCI_BUF_RECONCILE_NONZERO_MIN_F(imagePyramidConstraints.scaleFactor,
                         engineConstraints.scaleFactor);
        LW_SCI_BUF_RECONCILE_NONZERO_MIN_U(imagePyramidConstraints.levelCount,
                         engineConstraints.levelCount);
    }

    /* Reconcile imagepyramid constraints with output constraints */
    LW_SCI_BUF_RECONCILE_NONZERO_MIN_F(constraints->scaleFactor,
                     imagePyramidConstraints.scaleFactor);
    LW_SCI_BUF_RECONCILE_NONZERO_MIN_U(constraints->levelCount,
                     imagePyramidConstraints.levelCount);

    LWSCI_INFO("Output: constraints: %p\n", constraints);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}
LwSciError LwSciBufIsIsoEngine(
    const LwSciBufHwEngine engineArray[],
    uint64_t engineCount,
    bool* isIsoEngine)
{
    LwSciError err = LwSciError_Success;

    (void)engineArray;
    (void)engineCount;

    LWSCI_FNENTRY("");

    if (isIsoEngine == NULL) {
        LWSCI_ERR("Bad parameter supplied to LwSciBufIsIsoEngine\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: engineArray: %p, engineCount: %" PRIu64 ", isIsoEngine: %p\n",
        engineArray, engineCount, isIsoEngine);

    *isIsoEngine = false;

    LWSCI_INFO("Output: isIsoEngine: %s\n", *isIsoEngine ? "true" : "false");

    LWSCI_FNEXIT("");

    return err;
}

LwSciError LwSciBufHasDlaEngine(
    const LwSciBufHwEngine engineArray[],
    uint64_t engineCount,
    bool* isDlaEngine)
{
    LwSciError sciErr = LwSciError_Success;

    (void)engineArray;
    (void)engineCount;

    LWSCI_FNENTRY("");

    if (isDlaEngine == NULL) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR("Bad parameter supplied to LwSciBufIsIsoEngine\n");
        LWSCI_ERR("isDlaEngine: %p\n", isDlaEngine);
        goto ret;
    }

    LWSCI_INFO("Input: engineArray: %p, engineCount: %lu, isDlaEngine: %p\n",
        engineArray, engineCount, isDlaEngine);

    /* initialize output */
    *isDlaEngine = false;

    LWSCI_INFO("Output: isDlaEngine: %s\n", *isDlaEngine ? "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

