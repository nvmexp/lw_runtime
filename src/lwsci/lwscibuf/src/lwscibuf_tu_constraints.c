/*
 * lwscibuf_tu_constraints.c
 *
 * GPU TU Arch Constraint Library
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
#include "lwscibuf_tu_constraints.h"
#include "lwscilog.h"

typedef enum {
    LwSciBufTU_Engine_Unknown = 0,
    LwSciBufTU_Engine_Graphics,
    LwSciBufTU_Engine_Copy,
    LwSciBufTU_Engine_LwEnc,
    LwSciBufTU_Engine_LwDec,
    LwSciBufTU_Engine_Mpeg,
    LwSciBufTU_Engine_Vic,
    LwSciBufTU_Engine_UpperBound,
} LwSciBufTUEngine;

#define LWSCIBUFTU_ENGINE_NUM \
    (LwSciBufTU_Engine_UpperBound - LwSciBufTU_Engine_Unknown)

#define LW2080_ENGINE_TYPE_NUM \
    LW2080_ENGINE_TYPE_LAST - LW2080_ENGINE_TYPE_NULL

typedef struct {
    bool isValid;
    LwSciBufTUEngine lwEngine;
} LwSciBufTUEngineMap;

static const LwSciBufTUEngineMap subEngineIdtoTUEngineMap[LW2080_ENGINE_TYPE_NUM] = {
    [LW2080_ENGINE_TYPE_GRAPHICS] = {true, LwSciBufTU_Engine_Graphics},
    [LW2080_ENGINE_TYPE_COPY0] = {true, LwSciBufTU_Engine_Copy},
    [LW2080_ENGINE_TYPE_LWDEC0] = {true, LwSciBufTU_Engine_LwDec},
    [LW2080_ENGINE_TYPE_MPEG] = {true, LwSciBufTU_Engine_Mpeg},
    [LW2080_ENGINE_TYPE_VIC] = {true, LwSciBufTU_Engine_Vic},
    [LW2080_ENGINE_TYPE_LWENC0] = {true, LwSciBufTU_Engine_LwEnc},
};

static void LwSciBufGetTuImgConstraints(
    LwSciBufTUEngine engine,
    LwSciBufImageConstraints* tuImgConstraints)
{
    /* Note: This is used instead of designated initializers to be consistent
     * with QNX, which has a restriction in the Safety Manual when using
     * designated initializers on nested structs. */
    switch (engine) {
        case LwSciBufTU_Engine_Graphics:
        {
            tuImgConstraints->plConstraints.startAddrAlign = 128U;
            tuImgConstraints->plConstraints.pitchAlign = 128U;
            tuImgConstraints->plConstraints.heightAlign = 1U;
            tuImgConstraints->plConstraints.sizeAlign = 256U;

            tuImgConstraints->blConstraints.startAddrAlign = 256U;
            tuImgConstraints->blConstraints.pitchAlign = 64U;
            tuImgConstraints->blConstraints.heightAlign = 1U;
            tuImgConstraints->blConstraints.sizeAlign = 256U;

            tuImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufTU_Engine_Mpeg:
        {
            tuImgConstraints->plConstraints.startAddrAlign = 1U;
            tuImgConstraints->plConstraints.pitchAlign = 1U;
            tuImgConstraints->plConstraints.heightAlign = 1U;
            tuImgConstraints->plConstraints.sizeAlign = 1U;

            tuImgConstraints->blConstraints.startAddrAlign = 1U;
            tuImgConstraints->blConstraints.pitchAlign = 1U;
            tuImgConstraints->blConstraints.heightAlign = 1U;
            tuImgConstraints->blConstraints.sizeAlign = 1U;

            tuImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufTU_Engine_Vic:
        {
            tuImgConstraints->plConstraints.startAddrAlign = 256U;
            tuImgConstraints->plConstraints.pitchAlign = 64U;
            tuImgConstraints->plConstraints.heightAlign = 1U;
            tuImgConstraints->plConstraints.sizeAlign = 256U;

            tuImgConstraints->blConstraints.startAddrAlign = 256U;
            tuImgConstraints->blConstraints.pitchAlign = 64U;
            tuImgConstraints->blConstraints.heightAlign = 1U;
            tuImgConstraints->blConstraints.sizeAlign = 256U;

            tuImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufTU_Engine_LwEnc:
        case LwSciBufTU_Engine_LwDec:
        {
            tuImgConstraints->plConstraints.startAddrAlign = 256U;
            tuImgConstraints->plConstraints.pitchAlign = 256U;
            tuImgConstraints->plConstraints.heightAlign = 1U;
            tuImgConstraints->plConstraints.sizeAlign = 256U;

            tuImgConstraints->blConstraints.startAddrAlign = 256U;
            tuImgConstraints->blConstraints.pitchAlign = 64U;
            tuImgConstraints->blConstraints.heightAlign = 1U;
            tuImgConstraints->blConstraints.sizeAlign = 256U;

            tuImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        default:
        {
            tuImgConstraints->plConstraints.startAddrAlign = 1U;
            tuImgConstraints->plConstraints.pitchAlign = 1U;
            tuImgConstraints->plConstraints.heightAlign = 1U;
            tuImgConstraints->plConstraints.sizeAlign = 1U;

            tuImgConstraints->blConstraints.startAddrAlign = 1U;
            tuImgConstraints->blConstraints.pitchAlign = 1U;
            tuImgConstraints->blConstraints.heightAlign = 1U;
            tuImgConstraints->blConstraints.sizeAlign = 1U;

            tuImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            tuImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
    }
}

static void LwSciBufGetTuArrConstraints(
    LwSciBufTUEngine engine,
    LwSciBufArrayConstraints* tuArrConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufTU_Engine_LwEnc:
        {
            tuArrConstraints->startAddrAlign = 1U;
            tuArrConstraints->dataAlign = 1U;

            break;
        }
        default:
        {
            tuArrConstraints->startAddrAlign = 1U;
            tuArrConstraints->dataAlign = 1U;

            break;
        }
    }
}

static void LwSciBufGetTuImgPyramidConstraints(
    LwSciBufTUEngine engine,
    LwSciBufImagePyramidConstraints* tuImgPyramidConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufTU_Engine_LwEnc:
        {
            tuImgPyramidConstraints->scaleFactor = 0.1F;
            tuImgPyramidConstraints->levelCount = LW_SCI_BUF_PYRAMID_MAX_LEVELS;

            break;
        }
        default:
        {
            memset(tuImgPyramidConstraints, 0x0, sizeof(*tuImgPyramidConstraints));
            break;
        }
    }
}

static void LwSciBufGetTuEngineImageConstraints(
    LwSciBufTUEngine engine,
    LwSciBufImageConstraints* imgConstraints,
    size_t imageConstraintsLen)
{
    LwSciBufImageConstraints tuImgConstraints = { 0 };

    LwSciBufGetTuImgConstraints(engine, &tuImgConstraints);

    LwSciCommonMemcpyS(imgConstraints, imageConstraintsLen,
                        &tuImgConstraints, sizeof(tuImgConstraints));
}

static void LwSciBufGetTuEngineArrayConstraints(
    LwSciBufTUEngine engine,
    LwSciBufArrayConstraints* arrConstraints,
    size_t arrConstraintsLen)
{
    LwSciBufArrayConstraints tuArrConstraints = { 0 };

    LwSciBufGetTuArrConstraints(engine, &tuArrConstraints);

    LwSciCommonMemcpyS(arrConstraints, arrConstraintsLen,
                        &tuArrConstraints, sizeof(tuArrConstraints));
}

static void LwSciBufGetTuEngineImagePyramidConstraints(
    LwSciBufTUEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints,
    size_t imgPyramidConstraintsLen)
{
    LwSciBufImagePyramidConstraints tuImgPyramidConstraints = { 0 };

    LwSciBufGetTuImgPyramidConstraints(engine, &tuImgPyramidConstraints);

    LwSciCommonMemcpyS(imgPyramidConstraints, imgPyramidConstraintsLen,
                        &tuImgPyramidConstraints, sizeof(tuImgPyramidConstraints));
}

static inline bool LwSciBufGetTUEngine(
    uint32_t subEngineID,
    LwSciBufTUEngine *engine)
{
    bool mapexist = false;

    LWSCI_FNENTRY("");

    if (subEngineID > (uint32_t) LW2080_ENGINE_TYPE_LAST || engine == NULL) {
        LWSCI_ERR_STR("Wrong subEngineID provided for TU Arch or engine is NULL\n" );
        goto ret;
    }

    LWSCI_INFO("Inputs: subEngineId: %d engine: %p\n", subEngineID, engine);
    if (subEngineIdtoTUEngineMap[subEngineID].isValid == true) {
        *engine = subEngineIdtoTUEngineMap[subEngineID].lwEngine;
        mapexist = true;
    } else {
        *engine = LwSciBufTU_Engine_UpperBound;
    }

    LWSCI_INFO("output engine: %d\n", *engine);
ret:
    LWSCI_FNEXIT("");
    return (mapexist);
}

LwSciError LwSciBufGetTUImageConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImageConstraints* imgConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufTUEngine tuEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", imgConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, imgConstraints);

    if (imgConstraints == NULL) {
        LWSCI_ERR_STR("Image constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(imgConstraints, 0, sizeof(*imgConstraints));
    if (LwSciBufGetTUEngine(engine.subEngineID, &tuEngine)) {
        LwSciBufGetTuEngineImageConstraints(tuEngine, imgConstraints, sizeof(*imgConstraints));
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

LwSciError LwSciBufGetTUArrayConstraints(
    LwSciBufHwEngine engine,
    LwSciBufArrayConstraints* arrConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufTUEngine tuEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", arrConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, arrConstraints);

    if (arrConstraints == NULL) {
        LWSCI_ERR_STR("Array constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(arrConstraints, 0, sizeof(*arrConstraints));
    if (LwSciBufGetTUEngine(engine.subEngineID, &tuEngine)) {
        LwSciBufGetTuEngineArrayConstraints(tuEngine, arrConstraints, sizeof(*arrConstraints));
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

LwSciError LwSciBufGetTUImagePyramidConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufTUEngine tuEngine;

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
    if (LwSciBufGetTUEngine(engine.subEngineID, &tuEngine)) {
        LwSciBufGetTuEngineImagePyramidConstraints(tuEngine,
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
