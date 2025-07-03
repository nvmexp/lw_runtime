/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_CONSTRAINT_H
#define INCLUDED_LWSCIBUF_ATTR_CONSTRAINT_H

#include "lwscibuf_attr_mgmt.h"

/**
 * \brief Computes consolidated buffer constraints to be considered
 * for buffer allocation.
 *
 * Gets consolidated buffer constraints by calling Common Constraints interface
 * and computes buffer size for every LwSciBufType specified in the given
 * LwSciBufAttrList using them. Verifies that size of the buffer callwlated for
 * every LwSciBufType matches with that callwlated for the other LwSciBufTypes
 * specified in the LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization:
 *      - The user must ensure that the same LwSciBufAttrList is not used by
 *        multiple threads at the same time
 *
 * \param[in,out] reconcileList The LwSciBufAttrList.
 *
 * \return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_Overflow if computing operation
 *   (including add, subtract, multiply) exceed data type MAX.
 * - LwSciError_NotSupported if any of the following oclwrs:
 *       - the build type does not support allocating buffer from GPU.
 * - LwSciError_IlwalidState if callwlated buffer size is not the same
 *   for every LwSciBufType.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a reconcileList is NULL
 *       - @a Failed to obtain attributes from given @a reconcileList
 * - Panic if any of the following oclwrs:
 *       - @a There is no constraint mapping for specific LwSciBufType
 *       - @a reconcileList is invalid
 *       - @a LwSciBufAttrValDataType obtained from @a reconcileList is equal
 *       or larger than LwSciDataType_UpperBound
 *
 * \implements{17827419}
 */
LwSciError LwSciBufApplyConstraints(LwSciBufAttrList reconcileList);

#endif /* INCLUDED_LWSCIBUF_ATTR_CONSTRAINT_H */
