/*
 * lwscibuf_constraint_lib.h
 *
 * Header file for constraint library S/W Unit
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_CONSTRAINT_LIB_PRIV_H
#define INCLUDED_LWSCIBUF_CONSTRAINT_LIB_PRIV_H

#include "lwscibuf_constraint_lib.h"

#define LW_SCI_BUF_RECONCILE_NONZERO_MIN(dest, src, zero)   \
            ((dest) = ((((dest) == (zero)) || ((src) == (zero))) ?  \
                    LW_SCI_BUF_MAX_NUM((dest), (src)): \
                    LW_SCI_BUF_MIN_NUM((dest), (src))))

#define LW_SCI_BUF_RECONCILE_NONZERO_MIN_U(dest, src) \
        (LW_SCI_BUF_RECONCILE_NONZERO_MIN((dest), (src), 0U))
#define LW_SCI_BUF_RECONCILE_NONZERO_MIN_F(dest, src) \
        (LW_SCI_BUF_RECONCILE_NONZERO_MIN((dest), (src), 0.0))

#define LW_SCI_BUF_ENG_NAME_BIT_COUNT   16U
#define LW_SCI_BUF_ENG_NAME_BIT_START   0U

/*  (uint32_t)1U is an obviously redundant typecast, however it inhibits
    following MISRA violation:

    Event misra_c_2012_rule_12_2_violation:
    In expression "1U << 16U", shifting more than 7 bits,
    the number of bits in the essential type of the
    left expression, "1U", is not allowed. The shift amount is 8.
*/
#define LW_SCI_BUF_ENG_NAME_BIT_MASK \
        (((uint32_t)1U << LW_SCI_BUF_ENG_NAME_BIT_COUNT) - 1U)

#define LW_SCI_BUF_ENG_INSTANCE_BIT_COUNT   8U
#define LW_SCI_BUF_ENG_INSTANCE_BIT_START   LW_SCI_BUF_ENG_NAME_BIT_COUNT

/*  (uint32_t)1U is an obviously redundant typecast, however it inhibits
    following MISRA violation:

    Event misra_c_2012_rule_12_2_violation:
    In expression "1U << 8U", shifting more than 7 bits,
    the number of bits in the essential type of the
    left expression, "1U", is not allowed. The shift amount is 8.
*/
#define LW_SCI_BUF_ENG_INSTANCE_BIT_MASK \
        (((uint32_t)1U << LW_SCI_BUF_ENG_INSTANCE_BIT_COUNT) - 1U)

/**
 * \brief Structure specifying the cheetah/GPU architecture
 *        identification details.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842139}
 */
typedef struct {
    /** Identifies cheetah architecture. */
    uint32_t chipId;

    /** Identifies GPU architecture. */
    uint32_t gpuArch;
} LwSciBufHwId;

/**
 * \brief Structure specifying pointers to functions corresponding to
 *        LwSciBufTypes which provide architecture specific buffer
 *        constraints for specified LwSciBufHwEngine which operates
 *        on buffer allocated for that LwSciBufType.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842142}
 */
typedef struct {
    /** cheetah/GPU architecture identification details */
    LwSciBufHwId  hwid;

    /** Pointer to function which gets architecture specific
     * LwSciBufImageConstraints specified by LwSciBufHwId for given
     * LwSciBufHwEngine for LwSciBufType_Image.
     */
    LwSciError (*getImageConstraints)(LwSciBufHwEngine engine,
        LwSciBufImageConstraints* imgConstraints);

    /** Pointer to function which gets architecture specific
     * LwSciBufArrayConstraints specified by LwSciBufHwId for given
     * LwSciBufHwEngine for LwSciBufType_Array.
     */
    LwSciError (*getArrayConstraints)(LwSciBufHwEngine engine,
        LwSciBufArrayConstraints* arrConstraints);

    /** Pointer to function which gets architecture specific
     * LwSciBufImagePyramidConstraints specified by LwSciBufHwId for
     * given LwSciBufHwEngine for LwSciBufType_Pyramid.
     */
    LwSciError (*getImagePyramidConstraints)(LwSciBufHwEngine engine,
        LwSciBufImagePyramidConstraints* imgPyramidConstraints);
} LwSciBufConstraintFvt;

/**
 *\brief Function pointer which points to the function that returns
 *       consolidated LwSciBufHwConstraints specific to a LwSciBufType
 *       for specified LwSciBufHwEngine(s).
 *
 * \implements{18842157}
 */
typedef LwSciError (*LwSciBufDataTypeConstraints)(
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data);

/**
 * @defgroup lwscibuf_hw_engine_api LwSciBuf APIs to get/set HW engine ID
 * List of APIs exposed internally to get/set LwSciBuf HW engine IDs
 * @{
 */

/**
 * The generated hardware engine ID must represent the combination of
 * LwSciBufHwEngName and instance of LwSciBufHwEngName. For this interface,
 * assume that the instance is 0.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @implements{18842748}
 *
 * @fn LwSciError LwSciBufHwEngCreateIdWithoutInstance(
 * LwSciBufHwEngName engName,int64_t* engId)
 */

/**
 * The generated hardware engine ID must represent the combination of
 * LwSciBufHwEngName and instance of LwSciBufHwEngName.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @implements{18842751}
 *
 * @fn LwSciError LwSciBufHwEngCreateIdWithInstance(LwSciBufHwEngName engName,
 * uint32_t instance, int64_t* engId)
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @implements{18842754}
 *
 * @fn LwSciError LwSciBufHwEngGetNameFromId(int64_t engId,
 * LwSciBufHwEngName* engName)
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @implements{18842757}
 *
 * @fn LwSciError LwSciBufHwEngGetInstanceFromId(int64_t engId,
 * uint32_t* instance)
 */

/**
 * @}
 */

/**
 * \brief Get the Default LwSciBufImageConstraints used in reconciliation.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[out] constraints: Pointer to the default LwSciBufImageConstraints.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *        - @a constraints is NULL
 *
 * \implements{19808982}
 */
void LwSciBufGetDefaultImageConstraints(
    LwSciBufImageConstraints* constraints);

/**
 * \brief Reconciles each of the members having the same constraint type in the
 * given LwSciBufHwConstraints and LwSciBufImageCommonConstraints, and updates
 * the result in the output LwSciBufHwConstraints. Reconciliation is done by
 * taking the maximum of each corresponding member of the same constraint type
 * between src and dest.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output dest parameter being
 *        reconciled is not used by multiple threads at the same time
 *
 * \param[in,out] dest: Input LwSciBufHwConstraints to be reconciled with
 * LwSciBufImageCommonConstraints. Reconciled constraints are output to
 * LwSciBufHwConstraints. Valid value: dest is valid input if it is not NULL.
 * \param[in] src: LwSciBufImageCommonConstraints to reconcile.
 *  Valid value: src is valid input if it is not NULL.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - panics if any of the following oclwrs:
 *        - @a dest is NULL
 *        - @a src is NULL
 *
 * \implements{18842766}
 */
LwSciError LwSciBufReconcileOutputImgConstraints(
    LwSciBufHwConstraints *dest,
    const LwSciBufImageCommonConstraints *src);

/**
 * \brief Reconciles each of the members having the same constraint type in the
 * given LwSciBufHwConstraints and LwSciBufImageBLConstraints, and updates the
 * result in the output LwSciBufHwConstraints. Reconciliation is done by taking
 * the maximum of each corresponding member of the same constraint type between
 * src and dest.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output dest parameter being
 *        reconciled is not used by multiple threads at the same time
 *
 * \param[in,out] dest: Input LwSciBufHwConstraints to be reconciled with
 * LwSciBufImageBLConstraints. Reconciled constraints are output to
 * LwSciBufHwConstraints. Valid value: dest is valid input if it is not NULL.
 * \param[in] src: LwSciBufImageBLConstraints to reconcile.
 *  Valid value: src is valid input if it is not NULL.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - panics if any of the following oclwrs:
 *        - @a dest is NULL
 *        - @a src is NULL
 *
 * \implements{18842769}
 */
LwSciError LwSciBufReconcileOutputBLConstraints(
    LwSciBufHwConstraints *dest,
    const LwSciBufImageBLConstraints *src);

/**
 * \brief Reconciles each of the members having the same constraint type in the
 * given input LwSciBufImageCommonConstraints structures, and updates the result
 * in the output LwSciBufImageCommonConstraints. Reconciliation is done by
 * taking the maximum of each corresponding member of the same constraint type
 * between src and dest.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output dest parameter being
 *        reconciled is not used by multiple threads at the same time
 *
 * \param[in,out] dest: Input LwSciBufImageCommonConstraints to be reconciled with
 * LwSciBufImageCommonConstraints. Reconciled constraints are output to this
 * LwSciBufImageCommonConstraints. Valid value: dest is valid input if it is not NULL.
 * \param[in] src: LwSciBufImageCommonConstraints to reconcile.
 *  Valid value: src is valid input if it is not NULL.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - panics if any of the following oclwrs:
 *        - @a dest is NULL
 *        - @a src is NULL
 *
 * \implements{18842772}
 */
LwSciError LwSciBufReconcileImgCommonConstraints(
    LwSciBufImageCommonConstraints *dest,
    const LwSciBufImageCommonConstraints *src);

/**
 * \brief Reconciles each of the members having the same constraint type in the
 * given LwSciBufImageBLConstraints and LwSciBufImageBLConstraints, and updates
 * the result in the output LwSciBufImageBLConstraints. Reconciliation is done
 * by taking the maximum of each corresponding member of the same constraint type
 * between src and dest.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output dest parameter being
 *        reconciled is not used by multiple threads at the same time
 *
 * \param[in,out] dest: Input LwSciBufImageBLConstraints to be reconciled with
 * LwSciBufImageBLConstraints. Reconciled constraints are output to this
 * LwSciBufImageBLConstraints. Valid value: dest is valid input if it is not NULL.
 * \param[in] src: LwSciBufImageBLConstraints to reconcile.
 *  Valid value: src is valid input if it is not NULL.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - panics if any of the following oclwrs:
 *        - @a dest is NULL
 *        - @a src is NULL
 *
 * \implements{18842775}
 */
LwSciError LwSciBufReconcileImageBLConstraints(
    LwSciBufImageBLConstraints *dest,
    const LwSciBufImageBLConstraints *src);

/**
 * \brief Reconciles destination and source integer LwSciBufImagePyramidConstraints
 *        and overwrites the result to destination. Reconciliation is done by choosing
 *        max between source and destination if either of source or destination is zero,
 *        if both source and destination are non-zero then min between source and
 *        destination is chosen.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output dest parameter being
 *        reconciled is not used by multiple threads at the same time
 *
 * \param[in,out] dest: Pointer to integer member of LwSciBufImagePyramidConstraints
 *  to reconcile and to which the result will be overwritten. Valid value: dest is valid
 *  input if it is not NULL.
 * \param[in] src: Pointer to integer member of LwSciBufImagePyramidConstraints
 *  to reconcile. Valid value: src is valid input if it is not NULL.
 *
 * \return LwSciError, the completion code of the operation
 * - LwSciError_Success if successful
 * - LwSciError_Overflow if destination integer value after reconciliation is
 *   greater than UINT8_MAX.
 * - panics if any of the following oclwrs:
 *        - @a dest is NULL
 *        - @a src is NULL
 *
 * \implements{22062151}
 */
LwSciError LwSciBufReconcileIntPyramidConstraints(
    uint8_t* dest,
    const uint8_t* src);

/**
 * \brief Reconciles destination and source float LwSciBufImagePyramidConstraints
 *        and overwrites the result to destination. Reconciliation is done by choosing
 *        max between source and destination if either of source or destination is zero,
 *        if both source and destination are non-zero then min between source and
 *        destination is chosen.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input/output dest parameter being
 *        reconciled is not used by multiple threads at the same time
 *
 * \param[in,out] dest: Pointer to float member of LwSciBufImagePyramidConstraints
 *  to reconcile and to which the result will be overwritten. Valid value: dest is valid
 *  input if it is not NULL.
 * \param[in] src: Pointer to float member of LwSciBufImagePyramidConstraints
 *  to reconcile. Valid value: src is valid input if it is not NULL.
 *
 * \return LwSciError, the completion code of the operation
 * - LwSciError_Success if successful
 * - panics if any of the following oclwrs:
 *        - @a dest is NULL
 *        - @a src is NULL
 *
 * \implements{22062154}
 */
LwSciError LwSciBufReconcileFloatPyramidConstraints(
    float* dest,
    const float* src);

extern const LwSciBufArrayConstraints arrayDefaultConstraints;
extern const LwSciBufImagePyramidConstraints imgPyramidDefaultConstraints;

/**
 * \brief An array of LwSciBufDataTypeConstraints function pointers
 * specifying functions corresponding to each LwSciBufType retrieving
 * LwSciBufHwConstraints for specified LwSciBufHwEngines operating on
 * the buffer allocated for the LwSciBufType.
 *
 * \implements{18842163}
 */
extern const LwSciBufDataTypeConstraints
                perDataTypeConstraints[LwSciBufType_MaxValid];

#endif  /* INCLUDED_LWSCIBUF_CONSTRAINT_LIB_PRIV_H */
