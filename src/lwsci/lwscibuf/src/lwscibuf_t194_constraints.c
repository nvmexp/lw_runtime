/*
 * lwscibuf_t194_constraints.c
 *
 * T194 Constraint Library
 *
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdbool.h>

#include "lwscibuf_t194_constraints.h"
#include "lwscicommon_os.h"
#include "lwscilog.h"

typedef enum {
    LwSciBufEngine_T194_Unknown = 0,
    LwSciBufEngine_T194_Vic,
    LwSciBufEngine_T194_LwEnc,
    LwSciBufEngine_T194_LwDec,
    LwSciBufEngine_T194_Disp,
  /* Note: PVA and DLA engines are not added to LwSciBufHwEngName*/
    LwSciBufEngine_T194_PVA,
    LwSciBufEngine_T194_DLA,
    LwSciBufEngine_T194_LwJpg,
    LwSciBufEngine_T194_Vi,
    LwSciBufEngine_T194_Isp,
    LwSciBufEngine_T194_Csi,
    LwSciBufEngine_T194_PCIe,
    LwSciBufEngine_T194_OFA,
    /* Upper bound */
    LwSciBufEngine_T194_UpperBound,
} LwSciBufT194Engine;

typedef struct {
    bool isValid;
    LwSciBufT194Engine lwEngine;
/* FIXME: version will be used only when lwrm_chipid.h is available */
    uint32_t version;
} LwSciBufT194EngineMap;

static void LwSciBufGetT194ImgConstraints(
    LwSciBufT194Engine engine,
    LwSciBufImageConstraints* t194ImgConstraints)
{
    /*
     * The constraints in this array are dolwmented in hardware doc.
     * Refer to Section 3.2.4 and 3.4.1 of Perforce path for T194 Constraints:
     *
     *  /hw/ar/doc/t19x/sysarch/system/global_functions/pixel_formats/arch/T19X Pixel Format GFD.docx
     *
     * Note: Designated initializers are not used for the nested structs due to
     * Restriction #64 in the QNX Safety Manual:
     *
     *  > Designated initializers with nested structs should be not be used to
     *  > avoid a known problem in the compiler that results in incorrect values
     *  > used for some fields.
     */
    if (NULL == t194ImgConstraints) {
        LwSciCommonPanic();
    }

    if ((LwSciBufEngine_T194_Unknown >= engine) ||
        (LwSciBufEngine_T194_UpperBound <= engine)) {
        LwSciCommonPanic();
    }

    switch (engine) {
        case LwSciBufEngine_T194_Vic:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 256U;
            t194ImgConstraints->plConstraints.pitchAlign = 256U;
            t194ImgConstraints->plConstraints.heightAlign = 4U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 8192U;
            t194ImgConstraints->blConstraints.pitchAlign = 64U;
            t194ImgConstraints->blConstraints.heightAlign = 128U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 9U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 4U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_LwEnc:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 256U;
            t194ImgConstraints->plConstraints.pitchAlign = 256U;
            t194ImgConstraints->plConstraints.heightAlign = 4U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 8192U;
            t194ImgConstraints->blConstraints.pitchAlign = 64U;
            t194ImgConstraints->blConstraints.heightAlign = 128U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 9U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 4U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_Vi:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 64U;
            t194ImgConstraints->plConstraints.pitchAlign = 64U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 8192U;
            t194ImgConstraints->blConstraints.pitchAlign = 64U;
            t194ImgConstraints->blConstraints.heightAlign = 128U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 9U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 4U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_Isp:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 64U;
            t194ImgConstraints->plConstraints.pitchAlign = 64U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 8192U;
            t194ImgConstraints->blConstraints.pitchAlign = 64U;
            t194ImgConstraints->blConstraints.heightAlign = 128U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 9U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 4U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_Csi:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 64U;
            t194ImgConstraints->plConstraints.pitchAlign = 64U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 8192U;
            t194ImgConstraints->blConstraints.pitchAlign = 64U;
            t194ImgConstraints->blConstraints.heightAlign = 128U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 9U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 4U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_Disp:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 65536U;
            t194ImgConstraints->plConstraints.pitchAlign = 64U;
            t194ImgConstraints->plConstraints.heightAlign = 4U;
            t194ImgConstraints->plConstraints.sizeAlign = 65536U;

            t194ImgConstraints->blConstraints.startAddrAlign = 65536U;
            t194ImgConstraints->blConstraints.pitchAlign = 64U;
            t194ImgConstraints->blConstraints.heightAlign = 128U;
            t194ImgConstraints->blConstraints.sizeAlign = 65536U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        /* The following engines are not applicable for 5.1 safety */
        /* and the constraints need to be fixed yet. */
        case LwSciBufEngine_T194_PVA:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 1024U;
            t194ImgConstraints->plConstraints.pitchAlign = 32U;
            t194ImgConstraints->plConstraints.heightAlign = 4U;
            t194ImgConstraints->plConstraints.sizeAlign = 128U * 1024U;

            t194ImgConstraints->blConstraints.startAddrAlign = 8192U;
            t194ImgConstraints->blConstraints.pitchAlign = 1U;
            t194ImgConstraints->blConstraints.heightAlign = 128U;
            t194ImgConstraints->blConstraints.sizeAlign = 128U * 1024U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 9U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 32U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_DLA:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 32U;
            t194ImgConstraints->plConstraints.pitchAlign = 32U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 1U;
            t194ImgConstraints->blConstraints.pitchAlign = 1U;
            t194ImgConstraints->blConstraints.heightAlign = 1U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_PCIe:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 4U;
            t194ImgConstraints->plConstraints.pitchAlign = 1U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 4U;

            t194ImgConstraints->blConstraints.startAddrAlign = 4U;
            t194ImgConstraints->blConstraints.pitchAlign = 1U;
            t194ImgConstraints->blConstraints.heightAlign = 1U;
            t194ImgConstraints->blConstraints.sizeAlign = 4U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_OFA:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 256U;
            t194ImgConstraints->plConstraints.pitchAlign = 256U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 512U;
            t194ImgConstraints->blConstraints.pitchAlign = 1U;
            t194ImgConstraints->blConstraints.heightAlign = 1U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufEngine_T194_LwDec:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 64U;
            t194ImgConstraints->plConstraints.pitchAlign = 64U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 512U;
            t194ImgConstraints->blConstraints.pitchAlign = 1U;
            t194ImgConstraints->blConstraints.heightAlign = 1U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        default:
        {
            t194ImgConstraints->plConstraints.startAddrAlign = 1U;
            t194ImgConstraints->plConstraints.pitchAlign = 1U;
            t194ImgConstraints->plConstraints.heightAlign = 1U;
            t194ImgConstraints->plConstraints.sizeAlign = 1U;

            t194ImgConstraints->blConstraints.startAddrAlign = 1U;
            t194ImgConstraints->blConstraints.pitchAlign = 1U;
            t194ImgConstraints->blConstraints.heightAlign = 1U;
            t194ImgConstraints->blConstraints.sizeAlign = 1U;

            t194ImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            t194ImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
    }
}

static void LwSciBufGetT194ArrConstraints(
    LwSciBufT194Engine engine,
    LwSciBufArrayConstraints* t194ArrConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufEngine_T194_LwEnc:
        {
            t194ArrConstraints->startAddrAlign = 256U;
            t194ArrConstraints->dataAlign = 256U;

            break;
        }
        default:
        {
            t194ArrConstraints->startAddrAlign = 1U;
            t194ArrConstraints->dataAlign = 1U;

            break;
        }
    }
}

static void LwSciBufGetT194ImgPyramidConstraints(
    LwSciBufT194Engine engine,
    LwSciBufImagePyramidConstraints* t194ImgPyramidConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufEngine_T194_LwEnc:
        {
            t194ImgPyramidConstraints->scaleFactor = 0.0F;
            t194ImgPyramidConstraints->levelCount = 0U;

            break;
        }
        default:
        {
            (void)memset(t194ImgPyramidConstraints, 0x0, sizeof(*t194ImgPyramidConstraints));

            break;
        }
    }
};

static inline bool LwSciBufGetTegraEngine(
    int64_t module,
    LwSciBufT194Engine *engine)
{
    static const LwSciBufT194EngineMap LwSciBufHwEnginetoTegraEngineMap[LwSciBufHwEngName_Num] = {
        [LwSciBufHwEngName_Vic] = {true, LwSciBufEngine_T194_Vic, 0},
        [LwSciBufHwEngName_MSENC] = {true, LwSciBufEngine_T194_LwEnc, 0},
        [LwSciBufHwEngName_LWDEC] = {true, LwSciBufEngine_T194_LwDec, 0},
        [LwSciBufHwEngName_Display] = {true, LwSciBufEngine_T194_Disp, 0},
        [LwSciBufHwEngName_LWJPG] = {true, LwSciBufEngine_T194_LwJpg, 0},
        [LwSciBufHwEngName_Vi] = {true, LwSciBufEngine_T194_Vi, 0},
        [LwSciBufHwEngName_Isp] = {true, LwSciBufEngine_T194_Isp, 0},
        [LwSciBufHwEngName_Csi] = {true, LwSciBufEngine_T194_Csi, 0},
        [LwSciBufHwEngName_DLA] = {true, LwSciBufEngine_T194_DLA, 0},
        [LwSciBufHwEngName_PVA] = {true, LwSciBufEngine_T194_PVA, 0},
        [LwSciBufHwEngName_PCIe] = {true, LwSciBufEngine_T194_PCIe, 0},
        [LwSciBufHwEngName_OFA] = {true, LwSciBufEngine_T194_OFA, 0},
    };

    bool mapexist = false;
    LwSciBufHwEngName moduleWithoutInst;
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufHwEngGetNameFromId(module, &moduleWithoutInst);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufHwEngGetNameFromId failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((LwSciBufHwEngName_Ilwalid >= moduleWithoutInst) ||
            (LwSciBufHwEngName_Num <= moduleWithoutInst))
    {
        LWSCI_ERR_STR("Wrong Module ID or engine is NULL\n" );
        LWSCI_ERR_SLONG("module_id: \n", module);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs: module_id: %ld engine: %p\n", module, engine);
    if (true == LwSciBufHwEnginetoTegraEngineMap[moduleWithoutInst].isValid) {
        *engine = LwSciBufHwEnginetoTegraEngineMap[moduleWithoutInst].lwEngine;
        mapexist = true;
    } else {
        *engine = LwSciBufEngine_T194_Unknown;
    }

    LWSCI_INFO("output engine: %d\n", *engine);
ret:
    LWSCI_FNEXIT("");
    return (mapexist);
}

// Helper to hide the default LwSciBufImageConstraints to work around Safety
// Manual Restriction #64.
static void LwSciBufGetT194EngineImageConstraints(
    LwSciBufT194Engine t194Engine,
    LwSciBufImageConstraints* imageConstraints,
    size_t imageConstraintsLen)
{
    LwSciBufImageConstraints t194EngineImgConstraints = { 0 };

    LwSciBufGetT194ImgConstraints(t194Engine, &t194EngineImgConstraints);

    LwSciCommonMemcpyS(imageConstraints, imageConstraintsLen,
                        &t194EngineImgConstraints,
                        sizeof(t194EngineImgConstraints));
}

static void LwSciBufGetT194EngineArrayConstraints(
    LwSciBufT194Engine t194Engine,
    LwSciBufArrayConstraints* arrConstraints,
    size_t arrConstraintsLen)
{
    LwSciBufArrayConstraints t194EngineArrConstraints = { 0 };

    LwSciBufGetT194ArrConstraints(t194Engine, &t194EngineArrConstraints);

    LwSciCommonMemcpyS(arrConstraints, arrConstraintsLen,
                        &t194EngineArrConstraints,
                        sizeof(t194EngineArrConstraints));
}

static void LwSciBufGetT194EngineImagePyramidConstraints(
    LwSciBufT194Engine t194Engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints,
    size_t imgPyramidConstraintsLen)
{
    LwSciBufImagePyramidConstraints t194EngineImgPyramidConstraints = { 0.0f, 0U };

    LwSciBufGetT194ImgPyramidConstraints(t194Engine, &t194EngineImgPyramidConstraints);

    LwSciCommonMemcpyS(imgPyramidConstraints, imgPyramidConstraintsLen,
                        &t194EngineImgPyramidConstraints,
                        sizeof(t194EngineImgPyramidConstraints));
}

LwSciError LwSciBufGetT194ImageConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImageConstraints* imgConstraints)
{
    LwSciError sciErr = LwSciError_Success;

    LwSciBufT194Engine t194Engine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", imgConstraints: %p\n",
        &engine, (uint64_t)engine.rmModuleID, imgConstraints);

    if (NULL == imgConstraints) {
        LWSCI_ERR_STR("Image constraints is NULL\n");
        LwSciCommonPanic();
    }

    (void)memset(imgConstraints, 0, sizeof(*imgConstraints));
    if (LwSciBufGetTegraEngine(engine.rmModuleID, &t194Engine)) {
        LwSciBufGetT194EngineImageConstraints(t194Engine, imgConstraints, sizeof(*imgConstraints));
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_STR("Engine not supported\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: \n");
    LWSCI_INFO("Pitch-linear image constraints:\n");
    LWSCI_INFO("startAddrAlign: %zu, pitchAlign: %zu, heightAlign: %zu,"
        " sizeAlign: %zu\n", imgConstraints->plConstraints.startAddrAlign,
        imgConstraints->plConstraints.pitchAlign,
        imgConstraints->plConstraints.heightAlign,
        imgConstraints->plConstraints.sizeAlign);
    LWSCI_INFO("Block-linear image constraints:\n");
    LWSCI_INFO("startAddrAlign: %zu, pitchAlign: %zu, heightAlign: %zu,"
        " sizeAlign: %zu, log2GobSize: %u, log2GobsperBlockX: %" PRIu32 ","
        " log2GobsperBlockY: %" PRIu32 ", log2GobsperBlockZ: %" PRIu32 "\n",
        imgConstraints->blConstraints.startAddrAlign,
        imgConstraints->blConstraints.pitchAlign,
        imgConstraints->blConstraints.heightAlign,
        imgConstraints->blConstraints.sizeAlign,
        imgConstraints->blSpecificConstraints.log2GobSize,
        imgConstraints->blSpecificConstraints.log2GobsperBlockX,
        imgConstraints->blSpecificConstraints.log2GobsperBlockY,
        imgConstraints->blSpecificConstraints.log2GobsperBlockZ);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufGetT194ArrayConstraints(
    LwSciBufHwEngine engine,
    LwSciBufArrayConstraints* arrConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufT194Engine t194Engine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", arrConstraints: %p\n",
        &engine, (uint64_t)engine.rmModuleID, arrConstraints);

    if (NULL == arrConstraints) {
        LWSCI_ERR_STR("Array constraints is NULL\n");
        LwSciCommonPanic();
    }

    (void)memset(arrConstraints, 0, sizeof(*arrConstraints));
    if (LwSciBufGetTegraEngine(engine.rmModuleID, &t194Engine)) {
        LwSciBufGetT194EngineArrayConstraints(t194Engine, arrConstraints, sizeof(*arrConstraints));
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_STR("Engine not supported\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: \n");
    LWSCI_INFO("Array constraints:\n");
    LWSCI_INFO("startAddrAlign: %zu, dataAlign: %zu\n",
        arrConstraints->startAddrAlign, arrConstraints->dataAlign);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufGetT194ImagePyramidConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufT194Engine t194Engine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", "
        "imgPyramidConstraints: %p\n", &engine, (uint64_t)engine.rmModuleID,
        imgPyramidConstraints);

    if (NULL == imgPyramidConstraints) {
        LWSCI_ERR_STR("Pyramid constraints is NULL\n");
        LwSciCommonPanic();
    }

    (void)memset(imgPyramidConstraints, 0,
                    sizeof(LwSciBufImagePyramidConstraints));

    if (LwSciBufGetTegraEngine(engine.rmModuleID, &t194Engine)) {
        LwSciBufGetT194EngineImagePyramidConstraints(t194Engine, imgPyramidConstraints, sizeof(*imgPyramidConstraints));
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_STR("Engine not supported\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: \n");
    LWSCI_INFO("Image pyramid constraints:\n");
    LWSCI_INFO("scaleFactor: %.1f, levelCount: %" PRIu8 "\n",
        imgPyramidConstraints->scaleFactor,
        imgPyramidConstraints->levelCount);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}
