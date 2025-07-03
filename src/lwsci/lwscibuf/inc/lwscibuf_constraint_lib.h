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

#ifndef INCLUDED_LWSCIBUF_CONSTRAINT_LIB_H
#define INCLUDED_LWSCIBUF_CONSTRAINT_LIB_H

#include <stddef.h>

#include "lwscierror.h"
#include "lwscicommon_libc.h"
#include "lwscibuf_dev.h"
#include "lwscibuf_internal.h"
#include "lwscibuf_utils.h"

/* FIXME: This macro to be removed after the mirroring of lwrm_chipid.h */
#define LWRM_T194_ID 0x19

/*
 * TODO we need to move this alignment constraint to constraint library.
 * The question for which engines should we add this constraint.
 * LwRmSurface applies this constraint to all allocations.
 * refer to file lwrm_surface_fusa.c for bug ids
 * lwbugs/200671636 [LwSciBuf] Fix default image constraints in constraint library
 */
#if defined(__x86_64__)
    #define LWSCIBUF_PREFERRED_PLANE_ALIGNMENT_CONSTRAINT 1
#else // defined(__x86_64__)
    #define LWSCIBUF_PREFERRED_PLANE_ALIGNMENT_CONSTRAINT 128U*1024U
#endif // defined(__x86_64__)


/*
 * Consolidated constraints of all datatypes
 */
/**
 * \brief Structure specifying consolidated hardware constraints obtained by
 *        taking into account the constraints of every LwSciBufHwEngine applicable
 *        for every LwSciBufType in reconciled LwSciBufAttrList.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842145}
 */
typedef struct {
    /** Consolidated start address alignment of the buffer to be allocated.
     *  It is callwlated by choosing maximum value of start address alignment
     *  applicable for any LwSciBufHwEngine for any LwSciBufType specified in
     *  reconciled LwSciBufAttrList.
     */
    uint32_t startAddrAlign;
    /** Consolidated pitch alignment of the buffer to be allocated. It is
     *  callwlated by choosing maximum value of pitch alignment applicable
     *  for any LwSciBufHwEngine for any LwSciBufType specified in reconciled
     *  LwSciBufAttrList.
     */
    uint32_t pitchAlign;
    /** Consolidated height alignment of the buffer to be allocated. It is
     *  callwlated by choosing maximum value of height alignment applicable
     *  for any LwSciBufHwEngine for any LwSciBufType specified in reconciled
     *  LwSciBufAttrList.
     */
    uint32_t heightAlign;
    /** Consolidated buffer size alignment (in bytes) of the buffer to be
     *  allocated. It is callwlated by choosing maximum value of buffer size
     *  alignment applicable for any LwSciBufHwEngine for any LwSciBufType
     *  specified in reconciled LwSciBufAttrList.
     */
    uint32_t sizeAlign;

    /** Consolidated data size alignment (in bytes) of the buffer to be allocated
     *  which is applicable only for LwSciBufType_Array. It is callwlated by
     *  choosing maximum value of data size alignment applicable for any
     *  LwSciBufHwEngine for LwSciBufType_Array specified in reconciled
     *  LwSciBufAttrList.
     */
    size_t dataAlign;

    /** Consolidated Log Base 2 value of GobSize (Group of Block Size) of the
     *  buffer to be allocated which is applicable only for block linear images.
     *  It is callwlated by choosing maximum value of GobSize applicable for any
     *  LwSciBufHwEngine for any LwSciBufType specified in reconciled LwSciBufAttrList.
     */
    uint32_t log2GobSize;

    /** Consolidated Log Base 2 value of Gobs per Block X of the buffer to be
     *  allocated which is applicable only for block linear images. It is callwlated
     *  by choosing maximum value of Gobs per Block X applicable for any LwSciBufHwEngine
     *  for any LwSciBufType specified in reconciled LwSciBufAttrList.
     */
    uint32_t log2GobsperBlockX;

    /** Consolidated Log Base 2 value of Gobs per Block Y of the buffer to be
     *  allocated which is applicable only for block linear images. It is callwlated
     *  by choosing maximum value of Gobs per Block Y applicable for any LwSciBufHwEngine
     *  for any LwSciBufType specified in reconciled LwSciBufAttrList.
     */
    uint32_t log2GobsperBlockY;

    /** Consolidated Log Base 2 value of Gobs per Block Z of the buffer to be
     *  allocated which is applicable only for block linear images. It is callwlated
     *  by choosing maximum value of Gobs per Block Z applicable for any LwSciBufHwEngine
     *  for any LwSciBufType specified in reconciled LwSciBufAttrList.
     */
    uint32_t log2GobsperBlockZ;

    /** Consolidated scale factor of the Hardware constraints */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    float scaleFactor;
    /** Consolidated  number of mipmap levels */
    uint8_t levelCount;
} LwSciBufHwConstraints;

/**
 * \brief Structure specifying buffer constraints for LwSciBufType_Image which
 * are applicable for both image layouts LwSciBufImage_BlockLinearType and
 * LwSciBufImage_PitchLinearType.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842148}
 */
typedef struct {
    /** Start address alignment of the buffer to be allocated */
    uint32_t  startAddrAlign;
    /** Pitch alignment of the buffer to be allocated */
    uint32_t  pitchAlign;
    /** Height alignment of the buffer to be allocated */
    uint32_t  heightAlign;
    /** Buffer size alignment (in bytes) of the buffer to be allocated */
    uint32_t  sizeAlign;
} LwSciBufImageCommonConstraints;

/**
 * \brief Structure specifying buffer constraints for LwSciBufType_Image which
 * are applicable for image layout LwSciBufImage_BlockLinearType only.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842151}
 */
typedef struct {
    /** Log Base 2 value of GobSize (Group of Block Size) of the buffer to be allocated */
    uint32_t  log2GobSize;
    /** Log Base 2 value of Number of Gobs per Block X of the buffer to be allocated */
    uint32_t  log2GobsperBlockX;
    /** Log Base 2 value of Number of Gobs per Block Y of the buffer to be allocated */
    uint32_t  log2GobsperBlockY;
    /** Log Base 2 value of Number of Gobs per Block Z of the buffer to be allocated */
    uint32_t  log2GobsperBlockZ;
} LwSciBufImageBLConstraints;

/**
 * \brief Structure specifying buffer constraints for LwSciBufType_Image.
 * This structure specifies consolidated constraints for both image layouts
 * LwSciBufImage_BlockLinearType and LwSciBufImage_PitchLinearType.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{18842154}
 */
typedef struct {
    /** Buffer constraints for LwSciBufType_Image for
     * LwSciBufImage_PitchLinearType layout */
    LwSciBufImageCommonConstraints  plConstraints;

    LwSciBufImageCommonConstraints  blConstraints;

    /** Buffer constraints for LwSciBufType_Image for
     * LwSciBufImage_BlockLinearType layout */
    LwSciBufImageBLConstraints  blSpecificConstraints;
} LwSciBufImageConstraints;

/**
 * \brief Structure specifying the LwSciBufType_Array constraints.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{22034049}
 */
typedef struct {
    /** Start address alignment of the buffer to be allocated */
    uint32_t  startAddrAlign;

    /** Data size alignment (in bytes) of the buffer to be allocated */
    size_t  dataAlign;
} LwSciBufArrayConstraints;

/**
 * \brief Structure specifying the LwSciBufType_Pyramid constraints.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * \implements{22034053}
 */
typedef struct {
    /** Required scale factor for pyramid image constraints */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    float scaleFactor;
    /** Required number of mipmap levels */
    uint8_t levelCount;
} LwSciBufImagePyramidConstraints;

/**
 * \brief Gets LwSciBufHwConstraints for specified LwSciBufHwEngine
 *        array based on the LwSciBufType. Calls function corresponding to
 *        specified LwSciBufType from an array of LwSciBufDataTypeConstraints
 *        function pointers represented by perDatatypeConstraints.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same output constraint parameter is not
 *        used by multiple threads at the same time
 *
 * \param[in] bufType: LwSciBufType for which LwSciBufHwConstraints need
 *  to be obtained.
 *  Valid value: LwSciBufType enum value > LwSciBufType_General and <
 *  LwSciBufType_UpperBound.
 *
 * \param[in] chipId: Chip type.
 *  Valid value: LWRM_T194_ID for cheetah usecases.
 *
 * \param[in] engineArray: Array of LwSciBufHwEngine.
 *  Valid value: NULL is a valid value. If engineArray is non-NULL,the array is
 *  valid if rmModuleID member of every LwSciBufHwEngine in an array is obtained
 *  by successfully calling LwSciBufHwEngCreateIdWithoutInstance() or
 *  LwSciBufHwEngCreateIdWithInstance().
 *
 * \param[in] engineCount: Number of LwSciBufHwEngine that
 *  engineArray argument holds. Valid value: 0 if engineArray is NULL. non-zero
 *  if engineArray is not NULL.
 *
 * \param[in] data: LwSciBufType specific private data required for obtaining
 *  LwSciBufHwConstraints for the specified LwSciBufType.
 *  Valid value: Non-NULL and *data = LwSciBufImage_PitchLinearType or
 *  LwSciBufImage_BlockLinearType, if LwSciBufType = LwSciBufType_Image.
 *  Don't care if LwSciBufType != LwSciBufType_Image.
 *
 * \param[out] constraints: Reconciled LwSciBufHwConstraints for specified
 *  LwSciBufHwEngine(s) for specified LwSciBufType.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_NotSupported if @a chipId is not found.
 * - LwSciError_NotSupported if data is not one of the below
 *   if bufType = LwSciBufType_Image
 *    - LwSciBufImage_PitchLinearType
 *    - LwSciBufImage_BlocklinearType
 * - panics if:
 *     - @a bufType >= LwSciBufType_MaxValid
 *     - @a constraints is NULL
 *     - @a when bufType is LwSciBufType_Image, data is NULL
 *
 * \implements{18842763}
 */
LwSciError LwSciBufGetConstraints(
    LwSciBufType bufType,
    uint32_t chipId,
    const LwSciBufHwEngine engineArray[],
    uint32_t engineCount,
    LwSciBufHwConstraints* constraints,
    const void* data);

/**
 * \brief Checks if any of the given LwSciBufHwEngine is
 *        an ISO (isochronous) engine. An isochronous engine is an engine
 *        which either streams data in, streams data out or does both.
 *        In the context of an ASIC engine, it means that there is a requirement
 *        on the engine to either deliver or to receive data at a continuous and
 *        sustained rate from or to a source that is "streaming", meaning that it
 *        is not stallable.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same output isIsoEngine parameter is
 *        not used by multiple threads at the same time
 *
 * \param[in] engineArray: Array of LwSciBufHwEngine.
 *  Valid value: NULL is a valid value. If engineArray is non-NULL,the array is
 *  valid if rmModuleID member of every LwSciBufHwEngine in an array is obtained
 *  by successfully calling LwSciBufHwEngCreateIdWithoutInstance() or
 *  LwSciBufHwEngCreateIdWithInstance().
 * \param[in] engineCount: Number of LwSciBufHwEngine that
 *  engineArray argument holds. Valid value: 0 if engineArray is NULL. non-zero
 *  if engineArray is not NULL.
 * \param[out] isIsoEngine: True if any of LwSciBufHwEngine in specified array
 *  is isochronous. False otherwise.
 *
 * \returns LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if the given engineArray is invalid
 * - panics if isIsoEngine is NULL
 *
 * \implements{18842778}
 */
LwSciError LwSciBufIsIsoEngine(
    const LwSciBufHwEngine engineArray[],
    uint64_t engineCount,
    bool* isIsoEngine);

/**
 * \brief Checks if any of the given LwSciBufHwEngine is
 *        an DLA engine. If rmModuleID of any of the given LwSciBufHwEngine
 *        is LwSciBufHwEngName_DLA, it returns true.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same output isDlaEngine parameter is
 *        not used by multiple threads at the same time
 *
 * \param[in] engineArray: Array of LwSciBufHwEngine.
 *  Valid value: NULL is a valid value. If engineArray is non-NULL,the array is
 *  valid if rmModuleID member of every LwSciBufHwEngine in an array is obtained
 *  by successfully calling LwSciBufHwEngCreateIdWithoutInstance() or
 *  LwSciBufHwEngCreateIdWithInstance().
 * \param[in] engineCount: Number of LwSciBufHwEngine that
 *  engineArray argument holds. Valid value: 0 if engineArray is NULL. non-zero
 *  if engineArray is not NULL.
 * \param[out] isDlaEngine: True if rmModuleID of any of LwSciBufHwEngine in
 *  specified array is LwSciBufHwEngName_DLA, false otherwise.
 *
 * \returns LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if the given engineArray is invalid
 * - panics if @a isDlaEngine is NULL
 *
 * \implements{21555101}
 */
LwSciError LwSciBufHasDlaEngine(
    const LwSciBufHwEngine engineArray[],
    uint64_t engineCount,
    bool* isDlaEngine);

/**
 * \brief Checks if the query LwSciBufHwEngineName is present in the input
 *        LwSciBufHwEngine list. Returns true, if the queried
 *        LwSciBufHwEngineName is present in the input LwSciBufHwEngine list.
 *
 * \param[in] engineArray: Array of LwSciBufHwEngine.
 *  Valid value: NULL is a valid value. If engineArray is non-NULL,the array is
 *  valid if rmModuleID member of every LwSciBufHwEngine in an array is obtained
 *  by successfully calling LwSciBufHwEngCreateIdWithoutInstance() or
 *  LwSciBufHwEngCreateIdWithInstance().
 * \param[in] engineCount: Number of LwSciBufHwEngine that
 *  engineArray argument holds. Valid value: 0 if engineArray is NULL. non-zero
 *  if engineArray is not NULL.
 * \param[in] queryEngineName: LwSciBufHwEngName to be searched in input @a engineArray
 * \param[out] hasEngine: True if LwSciBufHwEngine name of any of LwSciBufHwEngine in
 *  specified array is queryEngineName, false otherwise.
 *
 * \returns LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a engineArray is invalid
 *      - @a queryEngineName is not a valid LwSciBufHwEngName
 * - panics if @a hasEngine is NULL
 *
 * \implements{}
 */
LwSciError LwSciBufHasEngine(
    const LwSciBufHwEngine engineArray[],
    const uint64_t engineCount,
    const LwSciBufHwEngName queryEngineName,
    bool* hasEngine);


#endif  /* INCLUDED_LWSCIBUF_CONSTRAINT_LIB_H */
