/*
 * lwscibuf_t194_constraints.h
 *
 * Header file to define T194 constraint extraction APIs.
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_T194_CONSTRAINTS_H
#define INCLUDED_LWSCIBUF_T194_CONSTRAINTS_H

#include "lwscierror.h"
#include "lwscibuf_constraint_lib.h"

/**
 * \brief Returns LwSciBufImageConstraints for specified LwSciBufHwEngine.
 * T194 specific constraints per engine corresponding to LwSciBufHwEngine are
 * obtained from /hw/ar/doc/t19x/sysarch/system/global_functions/pixel_formats/
 * arch/T19X Pixel Format GFD.docx. Please see section 3.2.4 and section 3.4.1.
 * Additionally, for LwSciBufHwEngine_Display, software constraints need to be
 * followed such that, overall size and base address of the buffer is aligned to
 * 64KB.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] engine LwSciBufHwEngine
 * valid value: LwSciBufHwEngine is valid if rmModuleID member of
 * LwSciBufHwEngine is obtained by successfully calling
 * LwSciBufHwEngCreateIdWithoutInstance() or
 * LwSciBufHwEngCreateIdWithInstance().
 * \param[out] imgConstraints: Struct to return image constraints
 *              for T194 HW engine. Returns start address, pitch,
 *              size and height information
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_NotSupported if @a engine.rmModuleId is not supported
 * - Panics if @a imgConstraints is NULL
 *
 * \implements{21598184}
 */
LwSciError LwSciBufGetT194ImageConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImageConstraints* imgConstraints);

/**
 * \brief Get Array constraints for CheetAh HW engine
 * Returns start address and data information
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] engine: T194 HW engine
 * \param[out] arrConstraints: Struct to return array constraints
 *              for T194 HW engine. Returns start address and
 *              data information
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_NotSupported if @a engine.rmModuleId is not supported
 * - Panics if @a arrConstraints is NULL
 *
 * \implements{22062157}
 */
LwSciError LwSciBufGetT194ArrayConstraints(
    LwSciBufHwEngine engine,
    LwSciBufArrayConstraints* arrConstraints);

/**
 * \brief Get Image Pyramid constraints for CheetAh HW engine
 * Returns scale factor and level count
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] engine: T194 HW engine
 * \param[out] imgPyramidConstraints: Struct to return image pyramid constraints
 *              for T194 HW engine. Returns scale factor and level count
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_NotSupported if @a engine.rmModuleId is not supported
 * - Panics if @a imgPyramidConstraints is NULL
 *
 * \implements{22062160}
 */
LwSciError LwSciBufGetT194ImagePyramidConstraints(
    LwSciBufHwEngine engine,
    LwSciBufImagePyramidConstraints* imgPyramidConstraints);

#endif /* INCLUDED_LWSCIBUF_T194_CONSTRAINTS_H */
