/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2011-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef __LWDWARF_H
#define __LWDWARF_H
// OPTIX_HAND_EDIT
#if 0
#include "ptxOptimize.h"
#include "DebugInfo.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpuDebugImageRec	*gpuDebugImage;
struct gpuDebugImageRec {
    Byte * image;
    uInt32 size;
};

typedef struct lwDwarfInfoRec *lwDwarfInfo;
extern lwDwarfInfo lwInitDwarf(int is_64bit);
// OPTIX_HAND_EDIT
#if 0
extern void lwObjDwarfCreate(elfw_t objElf, DebugInfo *debugHandle, lwDwarfInfo dwarfInfo,
                             ptxParsingState state, stdList_t ptxInputs, Bool compileOnly,
                             Bool deviceDebug, Bool lineInfoOnly, Bool forceDebugFrame,
                             Bool suppressDebugInfo);
void ptxDwarfExtractLiveRanges(ptxParsingState, lwDwarfInfo dwarfInfo);
#endif
// OPTIX_HAND_EDIT
char* ptxDwarfCreateByteStream( ptxDwarfSection section, stdVector_t stringVect, stdMap_t dwarfLabelMap );

#ifdef __cplusplus
}
#endif

#endif /* __LWDWARF_H */
