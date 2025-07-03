/*
 * lwscibuf_gp_constraints.c
 *
 * GPU GP Arch Constraint Library
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
#include "lwscibuf_gp_constraints.h"
#include "lwscilog.h"

typedef enum {
    LwSciBufGP_Engine_Unknown = 0,
    LwSciBufGP_Engine_Graphics,
    LwSciBufGP_Engine_Copy,
    LwSciBufGP_Engine_LwEnc,
    LwSciBufGP_Engine_LwDec,
    LwSciBufGP_Engine_Mpeg,
    LwSciBufGP_Engine_Vic,
    LwSciBufGP_Engine_UpperBound,
} LwSciBufGPEngine;

#define LWSCIBUFGP_ENGINE_NUM \
    (LwSciBufGP_Engine_UpperBound - LwSciBufGP_Engine_Unknown)

#define LW2080_ENGINE_TYPE_NUM \
    LW2080_ENGINE_TYPE_LAST - LW2080_ENGINE_TYPE_NULL

typedef struct {
    bool isValid;
    LwSciBufGPEngine lwEngine;
} LwSciBufGPEngineMap;

static const LwSciBufGPEngineMap subEngineIdtoGPEngineMap[LW2080_ENGINE_TYPE_NUM] = {
    [LW2080_ENGINE_TYPE_GRAPHICS] = {true, LwSciBufGP_Engine_Graphics},
    [LW2080_ENGINE_TYPE_COPY0] = {true, LwSciBufGP_Engine_Copy},
    [LW2080_ENGINE_TYPE_LWDEC0] = {true, LwSciBufGP_Engine_LwDec},
    [LW2080_ENGINE_TYPE_MPEG] = {true, LwSciBufGP_Engine_Mpeg},
    [LW2080_ENGINE_TYPE_VIC] = {true, LwSciBufGP_Engine_Vic},
    [LW2080_ENGINE_TYPE_LWENC0] = {true, LwSciBufGP_Engine_LwEnc},
};

static void LwSciBufGetGpImgConstraints(
    LwSciBufGPEngine engine,
    LwSciBufImageConstraints* gpImgConstraints)
{
    /* Note: This is used instead of designated initializers to be consistent
     * with QNX, which has a restriction in the Safety Manual when using
     * designated initializers on nested structs. */
    switch (engine) {
        case LwSciBufGP_Engine_Graphics:
        {
            gpImgConstraints->plConstraints.startAddrAlign = 256U;
            gpImgConstraints->plConstraints.pitchAlign = 128U;
            gpImgConstraints->plConstraints.heightAlign = 1U;
            gpImgConstraints->plConstraints.sizeAlign = 128U;

            gpImgConstraints->blConstraints.startAddrAlign = 512U;
            gpImgConstraints->blConstraints.pitchAlign = 64U;
            gpImgConstraints->blConstraints.heightAlign = 1U;
            gpImgConstraints->blConstraints.sizeAlign = 128U;

            gpImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGP_Engine_Mpeg:
        {
            gpImgConstraints->plConstraints.startAddrAlign = 1U;
            gpImgConstraints->plConstraints.pitchAlign = 1U;
            gpImgConstraints->plConstraints.heightAlign = 1U;
            gpImgConstraints->plConstraints.sizeAlign = 1U;

            gpImgConstraints->blConstraints.startAddrAlign = 1U;
            gpImgConstraints->blConstraints.pitchAlign = 1U;
            gpImgConstraints->blConstraints.heightAlign = 1U;
            gpImgConstraints->blConstraints.sizeAlign = 1U;

            gpImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGP_Engine_Vic:
        {
            gpImgConstraints->plConstraints.startAddrAlign = 256U;
            gpImgConstraints->plConstraints.pitchAlign = 256U;
            gpImgConstraints->plConstraints.heightAlign = 8U;
            gpImgConstraints->plConstraints.sizeAlign = 256U;

            gpImgConstraints->blConstraints.startAddrAlign = 1024U;
            gpImgConstraints->blConstraints.pitchAlign = 64U;
            gpImgConstraints->blConstraints.heightAlign = 1U;
            gpImgConstraints->blConstraints.sizeAlign = 256U;

            gpImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        case LwSciBufGP_Engine_LwEnc:
        case LwSciBufGP_Engine_LwDec:
        {
            gpImgConstraints->plConstraints.startAddrAlign = 256U;
            gpImgConstraints->plConstraints.pitchAlign = 64U;
            gpImgConstraints->plConstraints.heightAlign = 8U;
            gpImgConstraints->plConstraints.sizeAlign = 128U;

            gpImgConstraints->blConstraints.startAddrAlign = 512U;
            gpImgConstraints->blConstraints.pitchAlign = 64U;
            gpImgConstraints->blConstraints.heightAlign = 1U;
            gpImgConstraints->blConstraints.sizeAlign = 256U;

            gpImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockY = 1U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
        default:
        {
            gpImgConstraints->plConstraints.startAddrAlign = 1U;
            gpImgConstraints->plConstraints.pitchAlign = 1U;
            gpImgConstraints->plConstraints.heightAlign = 1U;
            gpImgConstraints->plConstraints.sizeAlign = 1U;

            gpImgConstraints->blConstraints.startAddrAlign = 1U;
            gpImgConstraints->blConstraints.pitchAlign = 1U;
            gpImgConstraints->blConstraints.heightAlign = 1U;
            gpImgConstraints->blConstraints.sizeAlign = 1U;

            gpImgConstraints->blSpecificConstraints.log2GobSize = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockX = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockY = 0U;
            gpImgConstraints->blSpecificConstraints.log2GobsperBlockZ = 0U;

            break;
        }
    }
}

static void LwSciBufGetGpArrConstraints(
    LwSciBufGPEngine engine,
    LwSciBufArrayConstraints* gpArrConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufGP_Engine_LwEnc:
        {
            gpArrConstraints->startAddrAlign = 1U,
            gpArrConstraints->dataAlign = 1U;

            break;
        }
        default:
        {
            gpArrConstraints->startAddrAlign = 1U,
            gpArrConstraints->dataAlign = 1U;

            break;
        }
    }
}

static void LwSciBufGetGpImgPyramidConstraints(
    LwSciBufGPEngine engine,
    LwSciBufImagePyramidConstraints* gpImgPyramidConstraints)
{
    /* Note: This pattern is used to be consistent with the pattern used to
     * access LwSciBufImageConstraints */
    switch (engine) {
        case LwSciBufGP_Engine_LwEnc:
        {
            gpImgPyramidConstraints->scaleFactor = 0.1F;
            gpImgPyramidConstraints->levelCount = LW_SCI_BUF_PYRAMID_MAX_LEVELS;

            break;
        }
        default:
        {
            memset(gpImgPyramidConstraints, 0x0, sizeof(*gpImgPyramidConstraints));
            break;
        }
    }
}

static void LwSciBufGetGpEngineImageConstraints(
    LwSciBufGPEngine engine,
    LwSciBufImageConstraints* imgConstraints,
    size_t imageConstraintsLen)
{
    LwSciBufImageConstraints gpImgConstraints = { 0 };

    LwSciBufGetGpImgConstraints(engine, &gpImgConstraints);

    LwSciCommonMemcpyS(imgConstraints, imageConstraintsLen,
                        &gpImgConstraints, sizeof(gpImgConstraints));
}

static void LwSciBufGetGpEngineArrayConstraints(
    LwSciBufGPEngine engine,
    LwSciBufArrayConstraints* arrConstraints,
    size_t arrConstraintsLen)
{
    LwSciBufArrayConstraints gpArrConstraints = { 0 };

    LwSciBufGetGpArrConstraints(engine, &gpArrConstraints);

    LwSciCommonMemcpyS(arrConstraints, arrConstraintsLen,
                        &gpArrConstraints, sizeof(gpArrConstraints));
}

static void LwSciBufGetGpEngineImagePyramidConstraints(
    LwSciBufGPEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints,
    size_t imgPyramidConstraintsLen)
{
    LwSciBufImagePyramidConstraints gpImgPyramidConstraints = { 0 };

    LwSciBufGetGpImgPyramidConstraints(engine, &gpImgPyramidConstraints);

    LwSciCommonMemcpyS(imgPyramidConstraints, imgPyramidConstraintsLen,
                        &gpImgPyramidConstraints, sizeof(gpImgPyramidConstraints));
}

static inline bool LwSciBufGetGPEngine(
    uint32_t subEngineID,
    LwSciBufGPEngine *engine)
{
    bool mapexist = false;

    LWSCI_FNENTRY("");

    if (subEngineID > (uint32_t) LW2080_ENGINE_TYPE_LAST || engine == NULL) {
        LWSCI_ERR_STR("Wrong subEngineID provided for GP Arch or engine is NULL\n" );
        goto ret;
    }

    LWSCI_INFO("Inputs: subEngineId: %d engine: %p\n", subEngineID, engine);
    if (subEngineIdtoGPEngineMap[subEngineID].isValid == true) {
        *engine = subEngineIdtoGPEngineMap[subEngineID].lwEngine;
        mapexist = true;
    } else {
        *engine = LwSciBufGP_Engine_UpperBound;
    }

    LWSCI_INFO("output engine: %d\n", *engine);
ret:
    LWSCI_FNEXIT("");
    return (mapexist);
}

LwSciError LwSciBufGetGPImageConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImageConstraints* imgConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGPEngine gpEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", imgConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, imgConstraints);

    if (imgConstraints == NULL) {
        LWSCI_ERR_STR("Image constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(imgConstraints, 0, sizeof(*imgConstraints));
    if (LwSciBufGetGPEngine(engine.subEngineID, &gpEngine)) {
        LwSciBufGetGpEngineImageConstraints(gpEngine, imgConstraints, sizeof(*imgConstraints));
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

LwSciError LwSciBufGetGPArrayConstraints(
    LwSciBufHwEngine engine,
    LwSciBufArrayConstraints* arrConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGPEngine gpEngine;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Input: engine: %p, enginID: %" PRIu64 ", arrConstraints: %p\n",
        engine, (uint64_t)engine.subEngineID, arrConstraints);

    if (arrConstraints == NULL) {
        LWSCI_ERR_STR("Array constraints is NULL\n");
        sciErr = LwSciError_BadParameter;
        goto ret;
    }

    (void)memset(arrConstraints, 0, sizeof(*arrConstraints));
    if (LwSciBufGetGPEngine(engine.subEngineID, &gpEngine)) {
        LwSciBufGetGpEngineArrayConstraints(gpEngine, arrConstraints, sizeof(*arrConstraints));
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

LwSciError LwSciBufGetGPImagePyramidConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufGPEngine gpEngine;

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
    if (LwSciBufGetGPEngine(engine.subEngineID, &gpEngine)) {
        LwSciBufGetGpEngineImagePyramidConstraints(gpEngine,
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
