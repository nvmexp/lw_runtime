/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCICOMMON_COLORCOLWERSION_H
#define INCLUDED_LWSCICOMMON_COLORCOLWERSION_H

#include "lwtypes.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscibuf_internal.h"
#include "lwcolor.h"

#define LWSCI_IS_VALID_COLOR(x) (\
        ((uint32_t)(x) > (uint32_t)LwSciColor_LowerBound) && \
        ((uint32_t)(x) < (uint32_t)LwSciColor_UpperBound) )


/**
 * \page lwscibuf_page_blanket_statements LwSciBuf blanket statements
 * \section lwscibuf_element_dependency Dependency on other elements
 *  LwSciBuf refers to below liblwrm_surface interfaces and datatypes:
 *  - LwColorDataType, which defines the data type of color components.
 *  - LwColorFormat, which represents a way of laying out pixels in memory.
 *  - LwColorGetBPP(), extract the number of bits per pixel for an
 *    LwColorFormat.
 *  - LwColorGetDataType(), extract color data type of an LwColorFormat.
 *
 * \implements{18842583}
 */

/**
 * @defgroup lwscibuf_helper_api LwSciBuf Helper APIs
 * List of helper function APIs
 * @{
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 *
 * \implements{18842979}
 *
 * \fn LwSciError LwSciColorToLwColor(LwSciBufAttrValColorFmt lwSciColorFmt,
 *                                    LwColorFormat* lwColorFmt);
 */


/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \implements{18842982}
 *
 * \fn LwSciError LwColorToLwSciColor(LwColorFormat lwColorFmt,
 *                                    LwSciBufAttrValColorFmt* lwSciColorFmt);
 */

/**
 * @}
 */

/**
 * @brief Get the number of color components in LwSciBufAttrValColorFmt
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @param[in] lwSciColorFmt LwSciBufAttrValColorFmt whose component count is
 *            requested. The valid data range is LwSciColor_LowerBound <
 *            lwSciColorFmt < LwSciColor_UpperBound.
 * @param[out] componentCount: number of color components in lwSciColorFmt.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a componentCount is NULL
 *      - @a lwSciColorFmt is not a supported by LwSciBuf
 *
 * \implements{18842994}
 */

LwSciError LwSciColorGetComponentCount(
    LwSciBufAttrValColorFmt lwSciColorFmt,
    uint8_t* componentCount);

#endif /* INCLUDED_LWSCICOMMON_COLORCOLWERSION_H */
