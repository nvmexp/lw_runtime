/*
 * lwscibuf_gv_constraints.c
 *
 * GPU GV Arch Constraint Library
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
#include "lwscibuf_gv_constraints.h"
#include "lwscilog.h"

typedef enum {
    LwSciBufGV_Engine_Unknown = 0,
    LwSciBufGV_Engine_Graphics,
    LwSciBufGV_Engine_Copy,
    LwSciBufGV_Engine_LwEnc,
    LwSciBufGV_Engine_LwDec,
    LwSciBufGV_Engine_Mpeg,
    LwSciBufGV_Engine_Vic,
    LwSciBufGV_Engine_UpperBound,
} LwSciBufGVEngine;

#define LWSCIBUFGV_ENGINE_NUM \
    (LwSciBufGV_Engine_UpperBound - LwSciBufGV_Engine_Unknown)

#define LW2080_ENGINE_TYPE_NUM \
    LW2080_ENGINE_TYPE_LAST - LW2080_ENGINE_TYPE_NULL

typedef struct {
    bool isValid;
    LwSciBufGVEngine lwEngine;
} LwSciBufGVEngineMap;

static const LwSciBufGVEngineMap subEngineIdtoGVEngineMap[LW2080_ENGINE_TYPE_NUM] = {
    [LW2080_ENGINE_TYPE_GRAPHICS] = {true, LwSciBufGV_Engine_Graphics},
    [LW2080_ENGINE_TYPE_COPY0] = {true, LwSciBufGV_Engine_Copy},
    [LW2080_ENGINE_TYPE_LWDEC0] = {true, LwSciBufGV_Engine_LwDec},
    [LW2080_ENGINE_TYPE_MPEG] = {true, LwSciBufGV_Engine_Mpeg},
    [LW2080_ENGINE_TYPE_VIC] = {true, LwSciBufGV_Engine_Vic},
    [LW2080_ENGINE_TYPE_LWENC0] = {true, LwSciBufGV_Engine_LwEnc},
};

static void LwSciBufGetGvImgConstraints(
    LwSciBufGVEngine engine,
    LwSciBufImageConstraints* gvImgConstraints)
{
    /* Note: This is used instead of designated initializers to be consistent
     * with QNX, which has a restriction in the Safety Manual when using
     * designated initializers on nested structs. */
    switch (engine) {
        case LwSciBufGV_Engine_Graphics:
        {
            gvImgConstraints->plConstraints.startAddrAlign = 256U;
            gvImgConstraints->plConstraints.pitchAlign = 128U;
            gvImgConstraints->plConstraints.heightAlign = 1U;
            gvImgConstraints->plConstraints.sizeAlign = 128U;

            gvImgConstraints->blConstraints.startAddrAlign = 512U;
            gvImgConstraints->blConstraints.pitchAlign = 64U;
            gvImgConstraints->blConstraints.heightAlign = 1U;
            gvImgConstraints->blConstraints.sizeAlign = 128U;

            gvImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGV_Engine_Mpeg:
        {
            gvImgConstraints->plConstraints.startAddrAlign = 1U;
            gvImgConstraints->plConstraints.pitchAlign = 1U;
            gvImgConstraints->plConstraints.heightAlign = 1U;
            gvImgConstraints->plConstraints.sizeAlign = 1U;

            gvImgConstraints->blConstraints.startAddrAlign = 1U;
            gvImgConstraints->blConstraints.pitchAlign = 1U;
            gvImgConstraints->blConstraints.heightAlign = 1U;
            gvImgConstraints->blConstraints.sizeAlign = 1U;

            gvImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGV_Engine_Vic:
        {
            gvImgConstraints->plConstraints.startAddrAlign = 256U;
            gvImgConstraints->plConstraints.pitchAlign = 256U;
            gvImgConstraints->plConstraints.heightAlign = 8U;
            gvImgConstraints->plConstraints.sizeAlign = 256U;

            gvImgConstraints->blConstraints.startAddrAlign = 256U;
            gvImgConstraints->blConstraints.pitchAlign = 64U;
            gvImgConstraints->blConstraints.heightAlign = 1U;
            gvImgConstraints->blConstraints.sizeAlign = 256U;

            gvImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGV_Engine_LwEnc:
        case LwSciBufGV_Engine_LwDec:
        {
            gvImgConstraints->plConstraints.startAddrAlign = 256U;
            gvImgConstraints->plConstraints.pitchAlign = 64U;
            gvImgConstraints->plConstraints.heightAlign = 8U;
            gvImgConstraints->plConstraints.sizeAlign = 128U;

            gvImgConstraints->blConstraints.startAddrAlign = 512U;
            gvImgConstraints->blConstraints.pitchAlign = 64U;
            gvImgConstraints->blConstraints.heightAlign = 1U;
            gvImgConstraints->blConstraints.sizeAlign = 256U;

            gvImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        default:
        {
            gvImgConstraints->plConstraints.startAddrAlign = 1U;
            gvImgConstraints->plConstraints.pitchAlign = 1U;
            gvImgConstraints->plConstraints.heightAlign = 1U;
            gvImgConstraints->plConstraints.sizeAlign = 1U;

            gvImgConstraints->blConstraints.startAddrAlign = 1U;
            gvImgConstraints->blConstraints.pitchAlign = 1U;
            gvImgConstraints->blConstraints.heightAlign = 1U;
            gvImgConstraints->blConstraints.sizeAlign = 1U;

            gvImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            gvImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
    }
}

static void LwSciBufGetGvArrConstraints(
    LwSciBufGVEngine engine,
    LwSciBufArrayConstraints* gvArrConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufGV_Engine_LwEnc:
        {
            gvArrConstraints->startAddrAlign = 1U;
            gvArrConstraints->dataAlign = 1U;

            break;
        }
        default:
        {
            gvArrConstraints->startAddrAlign = 1U;
            gvArrConstraints->dataAlign = 1U;

            break;
        }
    }
}

static void LwSciBufGetGvImgPyramidConstraints(
    LwSciBufGVEngine engine,
    LwSciBufImagePyramidConstraints* gvImgPyramidConstraints)
{
    if (gvImgPyramidConstraints == NULL) {
        LwSciCommonPanic();
    }

    if (engine <= LwSciBufGV_Engine_Unknown || engine >= LwSciBufGV_Engine_UpperBound) {
        LwSciCommonPanic();
    }

    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufGV_Engine_LwEnc:
        {
            gvImgPyramidConstraints->scaleFactor = 0.1F;
            gvImgPyramidConstraints->levelCount = LW_SCI_BUF_PYRAMID_MAX_LEVELS;

            break;
        }
        default:
        {
            memset(gvImgPyramidConstraints, 0x0, sizeof(*gvImgPyramidConstraints));
            break;
        }
    }
}

static void LwSciBufGetGvEngineImageConstraints(
    LwSciBufGVEngine engine,
    LwSciBufImageConstraints* imgConstraints,
    size_t imageConstraintsLen)
{
    LwSciBufImageConstraints gvImgConstraints = { 0 };

    LwSciBufGetGvImgConstraints(engine, &gvImgConstraints);

    LwSciCommonMemcpyS(imgConstraints, imageConstraintsLen,
                        &gvImgConstraints, sizeof(gvImgConstraints));
}

static void LwSciBufGetGvEngineArrayConstraints(
    LwSciBufGVEngine engine,
    LwSciBufArrayConstraints* arrConstraints,
    size_t arrConstraintsLen)
{
    LwSciBufArrayConstraints gvArrConstraints = { 0 };

    LwSciBufGetGvArrConstraints(engine, &gvArrConstraints);

    LwSciCommonMemcpyS(arrConstraints, arrConstraintsLen,
                        &gvArrConstraints, sizeof(gvArrConstraints));
}

static void LwSciBufGetGvEngineImagePyramidConstraints(
    LwSciBufGVEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints,
    size_t imgPyramidConstraintsLen)
{
    LwSciBufImagePyramidConstraints gvImgPyramidConstraints = { 0 };

    LwSciBufGetGvImgPyramidConstraints(engine, &gvImgPyramidConstraints);

    LwSciCommonMemcpyS(imgPyramidConstraints, imgPyramidConstraintsLen,
                        &gvImgPyramidConstraints, sizeof(gvImgPyramidConstraints));
}

static inline bool LwSciBufGetGVEngine(
    uint32_t subEngineID,
    LwSciBufGVEngine *engine)
{
    bool mapexist = false;

    LWSCI_FNENTRY("");

    if (subEngineID > (uint32_t) LW2080_ENGINE_TYPE_LAST || engine == NULL) {
        LWSCI_ERR_STR("Wrong subEngineID provided for GV Arch or engine is NULL\n" );
        goto ret;
    }

    LWSCI_INFO("Inputs: subEngineId: %d engine: %p\n", subEngineID, engine);
    if (subEngineIdtoGVEngineMap[subEngineID].isValid == true) {
        *engine = subEngineIdtoGVEngineMap[subEngineID].lwEngine;
        mapexist = true;
    } else {
        *engine = LwSciBufGV_Engine_UpperBound;
    }

    LWSCI_INFO("output engine: %d\n", *engine);
ret:
    LWSCI_FNEXIT("");
    return (mapexist);
}

LwSciError LwSciBufGetGVImageConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImageConstraints* imgConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGVEngine gvEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", imgConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, imgConstraints);

    if (imgConstraints == NULL) {
        LWSCI_ERR_STR("Image constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(imgConstraints, 0, sizeof(*imgConstraints));
    if (LwSciBufGetGVEngine(engine.subEngineID, &gvEngine)) {
        LwSciBufGetGvEngineImageConstraints(gvEngine, imgConstraints, sizeof(*imgConstraints));
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

LwSciError LwSciBufGetGVArrayConstraints(
    LwSciBufHwEngine engine,
    LwSciBufArrayConstraints* arrConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGVEngine gvEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", arrConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, arrConstraints);

    if (arrConstraints == NULL) {
        LWSCI_ERR_STR("Array constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(arrConstraints, 0, sizeof(*arrConstraints));
    if (LwSciBufGetGVEngine(engine.subEngineID, &gvEngine)) {
        LwSciBufGetGvEngineArrayConstraints(gvEngine, arrConstraints, sizeof(*arrConstraints));
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

LwSciError LwSciBufGetGVImagePyramidConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGVEngine gvEngine;

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
    if (LwSciBufGetGVEngine(engine.subEngineID, &gvEngine)) {
        LwSciBufGetGvEngineImagePyramidConstraints(gvEngine,
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
