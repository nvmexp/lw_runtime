/*
 * lwscibuf_constraint_lib_tegra.c
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
#include "lwscibuf_t194_constraints.h"
#include "lwscicommon_os.h"
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

const LwSciBufDataTypeConstraints
        perDataTypeConstraints[LwSciBufType_MaxValid] = {
        [LwSciBufType_Image]    = LwSciBufGetImageConstraints,
        [LwSciBufType_Array]    = LwSciBufGetArrayConstraints,
        [LwSciBufType_Pyramid]  = LwSciBufGetImagePyramidConstraints,
};

#if (LW_IS_SAFETY == 0)
static const LwSciBufConstraintFvt gpuFvtTable[] =  {
    {
        { .gpuArch = 0x150 },
        NULL,
        NULL,
        NULL,
    },
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

    if (NULL == fvt) {
        LWSCI_ERR_STR("Fvt pointer is NULL\n");
        LwSciCommonPanic();
    }
    *fvt = NULL;

    for (var = 0; var < chipcount; var++) {
        if (gpuArch == gpuFvtTable[var].hwid.gpuArch) {
            *fvt = &gpuFvtTable[var];
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
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
#endif

static LwSciError LwSciBufGetTegraFvt(
    uint32_t chipid,
    const LwSciBufConstraintFvt** fvt)
{
    LwSciError sciErr = LwSciError_Success;
    static const LwSciBufConstraintFvt tegraFvtTable[] =  {
        {
            { .chipId = LWRM_T194_ID },
            LwSciBufGetT194ImageConstraints,
            LwSciBufGetT194ArrayConstraints,
            LwSciBufGetT194ImagePyramidConstraints,
        }
    };

    uint64_t chipcount = sizeof(tegraFvtTable)/
                        sizeof(LwSciBufConstraintFvt);
    uint64_t var = 0U;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs: chipid: %d fvt: %p \n", chipid, fvt);

    *fvt = NULL;

    for (var = 0; var < chipcount; var++) {
        if (chipid == tegraFvtTable[var].hwid.chipId) {
            *fvt = &tegraFvtTable[var];
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    LWSCI_ERR_STR("Chipid not found in the CheetAh FVT\n");
    sciErr = LwSciError_NotSupported;

ret:
    LWSCI_INFO("Outputs: *fvt: %p sciErr: %d\n", *fvt, sciErr);
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError GetEngineConstraints(
    const LwSciBufHwEngine engine,
    const LwSciBufConstraintFvt* tegrafvt,
    LwSciBufImageConstraints *engineConstraints)
{
    LwSciError sciErr = LwSciError_Success;

    LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;
#if (LW_IS_SAFETY == 0)
    const LwSciBufConstraintFvt* gpufvt = NULL;
#endif
    LWSCI_FNENTRY("");

    sciErr = LwSciBufHwEngGetNameFromId(engine.rmModuleID, &engineName);
    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LwSciBufHwEngName_Gpu != engineName) {
        if (NULL != tegrafvt->getImageConstraints) {
            sciErr = tegrafvt->getImageConstraints(engine, engineConstraints);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Failed to get image constraints\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }
#if (LW_IS_SAFETY == 0)
    else {
        sciErr = LwSciBufGetGpuFvt(engine.rev.gpu.arch, &gpufvt);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to get GPU FVT\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (NULL != gpufvt->getImageConstraints) {
            sciErr = gpufvt->getImageConstraints(engine, engineConstraints);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Failed to get GPU image constraints\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }
#endif

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError ReconcileConstraints(
    LwSciBufAttrValImageLayoutType imageLayout,
    LwSciBufImageConstraints *imageConstraints,
    const LwSciBufImageConstraints *engineConstraints)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (LwSciBufImage_PitchLinearType == imageLayout) {
        sciErr = LwSciBufReconcileImgCommonConstraints(
            &imageConstraints->plConstraints,
            &engineConstraints->plConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("errCode = .\n", (uint32_t)sciErr);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else if (LwSciBufImage_BlockLinearType == imageLayout) {
        sciErr = LwSciBufReconcileImgCommonConstraints(
            &imageConstraints->blConstraints,
            &engineConstraints->blConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("errCode = .\n", (uint32_t)sciErr);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        sciErr = LwSciBufReconcileImageBLConstraints(
            &imageConstraints->blSpecificConstraints,
            &engineConstraints->blSpecificConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_UINT("errCode = .\n", (uint32_t)sciErr);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_UINT("image layout type not supported: \n", (uint32_t)imageLayout);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufGetEngineImageConstraints(
    const LwSciBufHwEngine engineArray[],
    const LwSciBufConstraintFvt* tegrafvt,
    LwSciBufAttrValImageLayoutType imageLayout,
    LwSciBufImageConstraints *imageConstraints,
    LwSciBufImageConstraints *engineConstraints,
    uint32_t engineCount)
{
    LwSciError sciErr = LwSciError_Success;
    uint32_t id = 0U;

    LWSCI_FNENTRY("");

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (id = 0U; id < engineCount; id++) {
        sciErr = GetEngineConstraints(engineArray[id],
                    tegrafvt, engineConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to get engine constraints.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        sciErr = ReconcileConstraints(imageLayout, imageConstraints,
                    engineConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to reconcile constraints.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

// Helper to hide the default LwSciBufImageConstraints to work around Safety
// Manual Restriction #64.
static void LwSciBufCopyDefaultImageConstraints(
    LwSciBufImageConstraints *imageConstraints,
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
    const LwSciBufConstraintFvt* tegrafvt = NULL;
    LwSciError sciErr = LwSciError_Success;
    LwSciBufImageConstraints imageConstraints = {0};
    LwSciBufImageConstraints engineConstraints = {0};
    LwSciBufAttrValImageLayoutType imageLayout = LwSciBufImage_PitchLinearType;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs: chipId: %d engineCount: %d, constraints ptr: %p, data %p\n",
                chipId, engineCount, constraints, data);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    imageLayout = *(const LwSciBufAttrValImageLayoutType*)data;

    LwSciBufCopyDefaultImageConstraints(&imageConstraints, sizeof(imageConstraints));

    sciErr = LwSciBufGetTegraFvt(chipId, &tegrafvt);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get cheetah fvt\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufGetEngineImageConstraints(engineArray,
                                               tegrafvt,
                                               imageLayout,
                                               &imageConstraints,
                                               &engineConstraints,
                                               engineCount);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get Engine Image constraints\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Reconcile image specific constraints with output constraints */

    if (LwSciBufImage_PitchLinearType == imageLayout) {
        /* Reconcile common constraints for pitchlinear */
        sciErr = LwSciBufReconcileOutputImgConstraints(constraints,
                                                       &imageConstraints.plConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to reconciled output image constraints\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else if (LwSciBufImage_BlockLinearType == imageLayout) {
        /* Reconcile common constraints for blocklinear */
        sciErr = LwSciBufReconcileOutputImgConstraints(constraints,
                                                       &imageConstraints.blConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to reconcile output image constraints\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        /* Reconcile blocklinear specific constraints */
        sciErr = LwSciBufReconcileOutputBLConstraints(constraints,
                                                      &imageConstraints.blSpecificConstraints);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to reconcile the output BL constraints\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_UINT("image layout type not supported: \n", (uint32_t)imageLayout);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
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
    const LwSciBufConstraintFvt* tegrafvt = NULL;
    LwSciError sciErr = LwSciError_Success;
    LwSciBufArrayConstraints arrayConstraints = {0};
    LwSciBufArrayConstraints engineConstraints = {0};
    uint32_t id = 0;

    (void)data;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs: chipId: %d engineCount: %d, constraints ptr: %p\n",
                chipId, engineCount, constraints);

    LwSciCommonMemcpyS(&arrayConstraints, sizeof(arrayConstraints),
                        &arrayDefaultConstraints,
                        sizeof(arrayDefaultConstraints));

    sciErr = LwSciBufGetTegraFvt(chipId, &tegrafvt);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get cheetah fvt\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (id = 0U; id < engineCount; id++) {
        LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;

        sciErr = LwSciBufHwEngGetNameFromId(engineArray[id].rmModuleID, &engineName);
        if (LwSciError_Success != sciErr) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (LwSciBufHwEngName_Gpu != engineName) {
            if (NULL != tegrafvt->getArrayConstraints) {
                sciErr = tegrafvt->getArrayConstraints(engineArray[id],
                    &engineConstraints);
                if (LwSciError_Success != sciErr) {
                    LWSCI_ERR_STR("Failed to get array constraints\n");
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto ret;
                }
            }
        }
#if (LW_IS_SAFETY == 0)
        else {
            const LwSciBufConstraintFvt* gpufvt = NULL;

            sciErr = LwSciBufGetGpuFvt(engineArray[id].rev.gpu.arch, &gpufvt);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Failed to get GPU fvt\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }

            if (NULL != gpufvt->getArrayConstraints) {
                sciErr = gpufvt->getArrayConstraints(engineArray[id],
                                                &engineConstraints);
                if (LwSciError_Success != sciErr) {
                    LWSCI_ERR_STR("Failed to get GPU array constraints\n");
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto ret;
                }
            }
        }
#endif

        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        LW_SCI_BUF_RECONCILE_MAX(arrayConstraints.startAddrAlign,
            engineConstraints.startAddrAlign);
        LW_SCI_BUF_RECONCILE_MAX(arrayConstraints.dataAlign,
            engineConstraints.dataAlign);
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
    }

    /* Reconcile array constraints with output constraints */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    LW_SCI_BUF_RECONCILE_MAX(constraints->startAddrAlign,
        arrayConstraints.startAddrAlign);
    LW_SCI_BUF_RECONCILE_MAX(constraints->dataAlign,
        arrayConstraints.dataAlign);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    LWSCI_INFO("Output: arrConstraints: %p\n", arrayConstraints);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufGetEnginePyramidConstraints(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    LwSciBufImagePyramidConstraints *imagePyramidConstraints,
    LwSciBufImagePyramidConstraints *engineConstraints,
    uint32_t id)
{
    LwSciError sciErr = LwSciError_Success;

    LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;
    const LwSciBufConstraintFvt* tegrafvt = NULL;

    LWSCI_FNENTRY("");

    sciErr = LwSciBufGetTegraFvt(chipId, &tegrafvt);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get cheetah fvt\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufHwEngGetNameFromId(engineArray[id].rmModuleID, &engineName);
    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LwSciBufHwEngName_Gpu != engineName) {
        if (NULL != tegrafvt->getImagePyramidConstraints) {
            sciErr = tegrafvt->getImagePyramidConstraints(engineArray[id],
                engineConstraints);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Failed to get pyramid constraints\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }
#if (LW_IS_SAFETY == 0)
    else {
        const LwSciBufConstraintFvt* gpufvt = NULL;

        sciErr = LwSciBufGetGpuFvt(engineArray[id].rev.gpu.arch, &gpufvt);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to get pyramid constraints\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (NULL != gpufvt->getImagePyramidConstraints) {
            sciErr = gpufvt->getImagePyramidConstraints(engineArray[id],
                engineConstraints);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Failed to get pyramid constraints\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }
#endif
    sciErr = LwSciBufReconcileFloatPyramidConstraints(&imagePyramidConstraints->scaleFactor,
                                                      &engineConstraints->scaleFactor);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to reconcile pyramid constraints\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufReconcileIntPyramidConstraints(&imagePyramidConstraints->levelCount,
                                                    &engineConstraints->levelCount);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to reconcile pyramid constraints\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

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
    LwSciError sciErr = LwSciError_Success;
    LwSciBufImagePyramidConstraints imagePyramidConstraints = {.0F, 0U};
    LwSciBufImagePyramidConstraints engineConstraints = {.0F, 0U};
    uint32_t id = 0;

    LWSCI_FNENTRY("");

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
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get image contraints for buffer type pyramid\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(&imagePyramidConstraints, sizeof(imagePyramidConstraints),
                        &imgPyramidDefaultConstraints,
                        sizeof(imgPyramidDefaultConstraints));

    for (id = 0U; id < engineCount; id++) {
        sciErr = LwSciBufGetEnginePyramidConstraints(chipId,
                                                     engineArray,
                                                     &imagePyramidConstraints,
                                                     &engineConstraints,
                                                     id);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to get engine pyramid constraints\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* Reconcile imagepyramid constraints with output constraints */
    sciErr = LwSciBufReconcileFloatPyramidConstraints(&constraints->scaleFactor,
                                        &imagePyramidConstraints.scaleFactor);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to reconcile pyramid constraints\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    sciErr = LwSciBufReconcileIntPyramidConstraints(&constraints->levelCount,
                                        &imagePyramidConstraints.levelCount);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to reconcile pyramid constraints\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

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

    uint64_t arrayIndex = 0U;

    LWSCI_FNENTRY("");

    if (NULL == isIsoEngine) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufIsIsoEngine\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: engineArray: %p, engineCount: %" PRIu64 ", isIsoEngine: %p\n",
        engineArray, engineCount, isIsoEngine);

    /* initialize output */
    *isIsoEngine = false;

    if (NULL == engineArray) {
        LWSCI_INFO("No engines set\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (arrayIndex = 0; arrayIndex < engineCount; arrayIndex++) {
        LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;

        err = LwSciBufHwEngGetNameFromId(engineArray[arrayIndex].rmModuleID, &engineName);
        if (LwSciError_Success != err) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if ((LwSciBufHwEngName_Display == engineName) || (LwSciBufHwEngName_Vi == engineName)) {
            *isIsoEngine = true;
            break;
        }
    }

    LWSCI_INFO("Output: isIsoEngine: %s\n", *isIsoEngine ? "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufHasDlaEngine(
    const LwSciBufHwEngine engineArray[],
    uint64_t engineCount,
    bool* isDlaEngine)
{
    LwSciError sciErr = LwSciError_Success;
    uint64_t arrayIndex = 0U;

    LWSCI_FNENTRY("");

    if (NULL == isDlaEngine) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufIsIsoEngine\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Input: engineArray: %p, engineCount: %lu, isDlaEngine: %p\n",
        engineArray, engineCount, isDlaEngine);

    /* initialize output */
    *isDlaEngine = false;

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

        if (LwSciBufHwEngName_DLA == engineName) {
            *isDlaEngine = true;
            break;
        }
    }

    LWSCI_INFO("Output: isDlaEngine: %s\n", *isDlaEngine ? "true" : "false");

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}
