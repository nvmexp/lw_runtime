/*
 * lwscibuf_ga_constraints.c
 *
 * GPU GA Arch Constraint Library
 *
 * Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdbool.h>
#include "lwscibuf_ga_constraints.h"
#include "lwscilog.h"

typedef enum {
    LwSciBufGA_Engine_Unknown = 0,
    LwSciBufGA_Engine_Graphics,
    LwSciBufGA_Engine_Copy,
    LwSciBufGA_Engine_LwEnc,
    LwSciBufGA_Engine_LwDec,
    LwSciBufGA_Engine_Mpeg,
    LwSciBufGA_Engine_Vic,
    LwSciBufGA_Engine_UpperBound,
} LwSciBufGAEngine;

#define LWSCIBUFGA_ENGINE_NUM \
    (LwSciBufGA_Engine_UpperBound - LwSciBufGA_Engine_Unknown)

#define LW2080_ENGINE_TYPE_NUM \
    LW2080_ENGINE_TYPE_LAST - LW2080_ENGINE_TYPE_NULL

typedef struct {
    bool isValid;
    LwSciBufGAEngine lwEngine;
} LwSciBufGAEngineMap;

static const LwSciBufGAEngineMap subEngineIdtoGAEngineMap[LW2080_ENGINE_TYPE_NUM] = {
    [LW2080_ENGINE_TYPE_GRAPHICS] = {true, LwSciBufGA_Engine_Graphics},
    [LW2080_ENGINE_TYPE_COPY0] = {true, LwSciBufGA_Engine_Copy},
    [LW2080_ENGINE_TYPE_LWDEC0] = {true, LwSciBufGA_Engine_LwDec},
    [LW2080_ENGINE_TYPE_MPEG] = {true, LwSciBufGA_Engine_Mpeg},
    [LW2080_ENGINE_TYPE_VIC] = {true, LwSciBufGA_Engine_Vic},
    [LW2080_ENGINE_TYPE_LWENC0] = {true, LwSciBufGA_Engine_LwEnc},
};

static void LwSciBufGetGaImgConstraints(
    LwSciBufGAEngine engine,
    LwSciBufImageConstraints* gaImgConstraints)
{
    /* Note: This is used instead of designated initializers to be consistent
     * with QNX, which has a restriction in the Safety Manual when using
     * designated initializers on nested structs. */
    switch (engine) {
        case LwSciBufGA_Engine_Graphics:
        {
            gaImgConstraints->plConstraints.startAddrAlign = 128U;
            gaImgConstraints->plConstraints.pitchAlign = 128U;
            gaImgConstraints->plConstraints.heightAlign = 1U;
            gaImgConstraints->plConstraints.sizeAlign = 256U;

            gaImgConstraints->blConstraints.startAddrAlign = 256U;
            gaImgConstraints->blConstraints.pitchAlign = 64U;
            gaImgConstraints->blConstraints.heightAlign = 1U;
            gaImgConstraints->blConstraints.sizeAlign = 256U;

            gaImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGA_Engine_Mpeg:
        {
            gaImgConstraints->plConstraints.startAddrAlign = 1U;
            gaImgConstraints->plConstraints.pitchAlign = 1U;
            gaImgConstraints->plConstraints.heightAlign = 1U;
            gaImgConstraints->plConstraints.sizeAlign = 1U;

            gaImgConstraints->blConstraints.startAddrAlign = 1U;
            gaImgConstraints->blConstraints.pitchAlign = 1U;
            gaImgConstraints->blConstraints.heightAlign = 1U;
            gaImgConstraints->blConstraints.sizeAlign = 1U;

            gaImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGA_Engine_Vic:
        {
            gaImgConstraints->plConstraints.startAddrAlign = 256U;
            gaImgConstraints->plConstraints.pitchAlign = 256U;
            gaImgConstraints->plConstraints.heightAlign = 1U;
            gaImgConstraints->plConstraints.sizeAlign = 256U;

            gaImgConstraints->blConstraints.startAddrAlign = 256U;
            gaImgConstraints->blConstraints.pitchAlign = 64U;
            gaImgConstraints->blConstraints.heightAlign = 1U;
            gaImgConstraints->blConstraints.sizeAlign = 256U;

            gaImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGA_Engine_LwEnc:
        case LwSciBufGA_Engine_LwDec:
        {
            gaImgConstraints->plConstraints.startAddrAlign = 256U;
            gaImgConstraints->plConstraints.pitchAlign = 256U;
            gaImgConstraints->plConstraints.heightAlign = 1U;
            gaImgConstraints->plConstraints.sizeAlign = 256U;

            gaImgConstraints->blConstraints.startAddrAlign = 256U;
            gaImgConstraints->blConstraints.pitchAlign = 64U;
            gaImgConstraints->blConstraints.heightAlign = 1U;
            gaImgConstraints->blConstraints.sizeAlign = 256U;

            gaImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        default:
        {
            gaImgConstraints->plConstraints.startAddrAlign = 1U;
            gaImgConstraints->plConstraints.pitchAlign = 1U;
            gaImgConstraints->plConstraints.heightAlign = 1U;
            gaImgConstraints->plConstraints.sizeAlign = 1U;

            gaImgConstraints->blConstraints.startAddrAlign = 1U;
            gaImgConstraints->blConstraints.pitchAlign = 1U;
            gaImgConstraints->blConstraints.heightAlign = 1U;
            gaImgConstraints->blConstraints.sizeAlign = 1U;

            gaImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            gaImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
    }
}

static void LwSciBufGetGaArrConstraints(
    LwSciBufGAEngine engine,
    LwSciBufArrayConstraints* gaArrConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufGA_Engine_LwEnc:
        {
            gaArrConstraints->startAddrAlign = 1U;
            gaArrConstraints->dataAlign = 1U;

            break;
        }
        default:
        {
            gaArrConstraints->startAddrAlign = 1U;
            gaArrConstraints->dataAlign = 1U;

            break;
        }
    }
}

static void LwSciBufGetGaImgPyramidConstraints(
    LwSciBufGAEngine engine,
    LwSciBufImagePyramidConstraints* gaImgPyramidConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufGA_Engine_LwEnc:
        {
            gaImgPyramidConstraints->scaleFactor = 0.1F;
            gaImgPyramidConstraints->levelCount = LW_SCI_BUF_PYRAMID_MAX_LEVELS;

            break;
        }
        default:
        {
            memset(gaImgPyramidConstraints, 0x0, sizeof(*gaImgPyramidConstraints));
            break;
        }
    }
}

static void LwSciBufGetGaEngineImageConstraints(
    LwSciBufGAEngine engine,
    LwSciBufImageConstraints* imgConstraints,
    size_t imageConstraintsLen)
{
    LwSciBufImageConstraints gaImgConstraints = { 0 };

    LwSciBufGetGaImgConstraints(engine, &gaImgConstraints);

    LwSciCommonMemcpyS(imgConstraints, imageConstraintsLen,
                        &gaImgConstraints, sizeof(gaImgConstraints));
}

static void LwSciBufGetGaEngineArrayConstraints(
    LwSciBufGAEngine engine,
    LwSciBufArrayConstraints* arrConstraints,
    size_t arrConstraintsLen)
{
    LwSciBufArrayConstraints gaArrConstraints = { 0 };

    LwSciBufGetGaArrConstraints(engine, &gaArrConstraints);

    LwSciCommonMemcpyS(arrConstraints, arrConstraintsLen,
                        &gaArrConstraints, sizeof(gaArrConstraints));
}

static void LwSciBufGetGaEngineImagePyramidConstraints(
    LwSciBufGAEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints,
    size_t imgPyramidConstraintsLen)
{
    LwSciBufImagePyramidConstraints gaImgPyramidConstraints = { 0 };

    LwSciBufGetGaImgPyramidConstraints(engine, &gaImgPyramidConstraints);

    LwSciCommonMemcpyS(imgPyramidConstraints, imgPyramidConstraintsLen,
                        &gaImgPyramidConstraints, sizeof(gaImgPyramidConstraints));
}

static inline bool LwSciBufGetGAEngine(
    uint32_t subEngineID,
    LwSciBufGAEngine *engine)
{
    bool mapexist = false;

    LWSCI_FNENTRY("");

    if (subEngineID > (uint32_t) LW2080_ENGINE_TYPE_LAST || engine == NULL) {
        LWSCI_ERR_STR("Wrong subEngineID provided for GA Arch or engine is NULL\n" );
        goto ret;
    }

    LWSCI_INFO("Inputs: subEngineId: %d engine: %p\n", subEngineID, engine);
    if (subEngineIdtoGAEngineMap[subEngineID].isValid == true) {
        *engine = subEngineIdtoGAEngineMap[subEngineID].lwEngine;
        mapexist = true;
    } else {
        *engine = LwSciBufGA_Engine_UpperBound;
    }

    LWSCI_INFO("output engine: %d\n", *engine);
ret:
    LWSCI_FNEXIT("");
    return (mapexist);
}

LwSciError LwSciBufGetGAImageConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImageConstraints* imgConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGAEngine gaEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", imgConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, imgConstraints);

    if (imgConstraints == NULL) {
        LWSCI_ERR_STR("Image constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(imgConstraints, 0, sizeof(*imgConstraints));
    if (LwSciBufGetGAEngine(engine.subEngineID, &gaEngine)) {
        LwSciBufGetGaEngineImageConstraints(gaEngine, imgConstraints, sizeof(*imgConstraints));
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_STR("Engine not supported\n");
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

LwSciError LwSciBufGetGAArrayConstraints(
    LwSciBufHwEngine engine,
    LwSciBufArrayConstraints* arrConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGAEngine gaEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", arrConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, arrConstraints);

    if (arrConstraints == NULL) {
        LWSCI_ERR_STR("Array constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(arrConstraints, 0, sizeof(*arrConstraints));
    if (LwSciBufGetGAEngine(engine.subEngineID, &gaEngine)) {
        LwSciBufGetGaEngineArrayConstraints(gaEngine, arrConstraints, sizeof(*arrConstraints));
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_STR("Engine not supported\n");
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

LwSciError LwSciBufGetGAImagePyramidConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGAEngine gaEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", "
        "imgPyramidConstraints: %p\n", engine, (uint64_t)engine.subEngineID,
        imgPyramidConstraints);

    if (imgPyramidConstraints == NULL) {
        LWSCI_ERR_STR("Pyramid constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(imgPyramidConstraints, 0,
                sizeof(LwSciBufImagePyramidConstraints));
    if (LwSciBufGetGAEngine(engine.subEngineID, &gaEngine)) {
        LwSciBufGetGaEngineImagePyramidConstraints(gaEngine,
                imgPyramidConstraints, sizeof(*imgPyramidConstraints));
    } else {
        sciErr = LwSciError_NotSupported;
        LWSCI_ERR_STR("Engine not supported\n");
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
