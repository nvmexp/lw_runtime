/*
 * Header file for LwSciBuf APIs
 *
 * Copyright (c) 2018-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
/**
 * @file
 *
 * @brief <b> LWPU Software Communications Interface (SCI) : LwSciBuf </b>
 *
 * Allows applications to allocate and exchange buffers in memory.
 */
#ifndef INCLUDED_LWSCIBUF_H
#define INCLUDED_LWSCIBUF_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "lwscierror.h"
#include <lwsciipc.h>

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @defgroup lwscibuf_blanket_statements LwSciBuf blanket statements.
 * Generic statements applicable for LwSciBuf interfaces.
 * @ingroup lwsci_buf
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements LwSciBuf blanket statements
 * \section lwscibuf_in_params Input parameters
 * - LwSciBufModule passed as input parameter to an API is valid input if it is
 * returned from a successful call to LwSciBufModuleOpen() and has not yet been
 * deallocated using LwSciBufModuleClose().
 * - LwSciIpcEndpoint passed as input parameter to an API is valid if it is
 * obtained from successful call to LwSciIpcOpenEndpoint() and has not yet been
 * freed using LwSciIpcCloseEndpoint().
 * - LwSciBufObj is valid if it is obtained from a successful call to
 * LwSciBufObjAlloc() or if it is obtained from a successful call to
 * LwSciBufAttrListReconcileAndObjAlloc() or if it is obtained from a
 * successful call to LwSciBufObjIpcImport() or if it is obtained from a
 * successful call to LwSciBufObjDup() or if it is obtained from a successful
 * call to LwSciBufObjDupWithReducePerm() and has not been deallocated
 * using LwSciBufObjFree().
 * - Unreconciled LwSciBufAttrList is valid if it is obtained from successful
 * call to LwSciBufAttrListCreate() or if it is obtained from successful call to
 * LwSciBufAttrListClone() where input to LwSciBufAttrListClone() is valid
 * unreconciled LwSciBufAttrList or if it is obtained from successful call to
 * LwSciBufAttrListIpcImportUnreconciled() and has not been deallocated using
 * LwSciBufAttrListFree().
 * - Reconciled LwSciBufAttrList is valid if it is obtained from successful call
 * to LwSciBufAttrListReconcile() or if it is obtained from successful call to
 * LwSciBufAttrListClone() where input to LwSciBufAttrListClone() is valid
 * reconciled LwSciBufAttrList or if it is obtained from successful call to
 * LwSciBufAttrListIpcImportReconciled() and has not been deallocated using
 * LwSciBufAttrListFree().
 * - If the valid range for the input parameter is not explicitly mentioned in
 * the API specification or in the blanket statements then it is considered that
 * the input parameter takes any value from the entire range corresponding to
 * its datatype as the valid value. Please note that this also applies to the
 * members of a structure if the structure is taken as an input parameter.
 *
 * \section lwscibuf_out_params Output parameters
 * - In general, output parameters are passed by reference through pointers.
 * Also, since a null pointer cannot be used to convey an output parameter, API
 * functions typically return an error code if a null pointer is supplied for a
 * required output parameter unless otherwise stated explicitly. Output
 * parameter is valid only if error code returned by an API is
 * LwSciError_Success unless otherwise stated explicitly.
 *
 * \section lwscibuf_conlwrrency Conlwrrency
 * - Every individual function can be called conlwrrently with itself without
 * any side-effects unless otherwise stated explicitly in the interface
 * specifications.
 * - The conditions for combinations of functions that cannot be called
 * conlwrrently or calling them conlwrrently leads to side effects are
 * explicitly stated in the interface specifications.
 */

/**
 * @}
 */

/**
 * @defgroup lwsci_buf Buffer Allocation APIs
 *
 * The LwSciBuf library contains the APIs for applications to allocate
 * and exchange buffers in memory.
 *
 * @ingroup lwsci_group_stream
 * @{
 */
/**
 * @defgroup lwscibuf_datatype LwSciBuf Datatype Definitions
 * Contains a list of all LwSciBuf datatypes.
 * @{
 */

/**
 * @brief Enum definitions of LwSciBuf datatypes.
 *
 * @implements{17824095}
 */
typedef enum {
    /** Reserved for General keys.
     * Shouldn't be used as valid value for  LwSciBufGeneralAttrKey_Types.
     */
    LwSciBufType_General = 0U,
    LwSciBufType_RawBuffer = 1U,
    LwSciBufType_Image = 2U,
    LwSciBufType_Tensor = 3U,
    LwSciBufType_Array = 4U,
    LwSciBufType_Pyramid = 5U,
    LwSciBufType_MaxValid = 6U,
    LwSciBufType_UpperBound = 6U,
} LwSciBufType;

/**
 * @}
 */

/**
 * @defgroup lwscibuf_constants LwSciBuf Global Constants
 * Definitions of all LwSciBuf Global Constants/Macros
 *
 * @{
 */
/**
 * @brief LwSciBuf API Major version number.
 *
 * @implements{18840105}
 */
static const uint32_t LwSciBufMajorVersion = 2U;

/**
 * @brief LwSciBuf API Minor version number.
 *
 * @implements{18840108}
 */
static const uint32_t LwSciBufMinorVersion = 4U;

#if defined(__cplusplus)

/**
 * @brief Maximum number of dimensions supported by tensor datatype.
 */
static const int LW_SCI_BUF_TENSOR_MAX_DIMS = 8;

/**
 * @brief Maximum number of planes supported by image datatype.
 */
static const int LW_SCI_BUF_IMAGE_MAX_PLANES = 3;

/**
 * @brief Maximum number of levels supported by pyramid datatype.
 */
static const int LW_SCI_BUF_PYRAMID_MAX_LEVELS = 10;

/**
 * @brief Indicates the size of export descriptor.
 */
static const int LWSCIBUF_EXPORT_DESC_SIZE = 32;

/**
 * @brief Indicates number of bits used for defining an attribute key.
 * Note: Maximum 16K attribute Keys per datatype.
 */
static const int LW_SCI_BUF_ATTRKEY_BIT_COUNT = 16;

/**
 * @brief Indicates number of bits used for defining an datatype of a key.
 * Note: Maximum 1K datatypes.
 */
static const int LW_SCI_BUF_DATATYPE_BIT_COUNT = 10;

/**
 * @brief Indicates the attribute key is a public key type.
 */
static const int LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC = 0;

/*
 * @brief Global constant to specify the start-bit of attribute Keytype.
 */
static const int LW_SCI_BUF_KEYTYPE_BIT_START =
        (LW_SCI_BUF_DATATYPE_BIT_COUNT + LW_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates starting value of General attribute keys.
 */
static const int LW_SCI_BUF_GENERAL_ATTR_KEY_START =
           (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) |
           (LwSciBufType_General << LW_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of Raw-buffer Datatype keys.
 */
static const int LW_SCI_BUF_RAW_BUF_ATTR_KEY_START =
           (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) |
           (LwSciBufType_RawBuffer << LW_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of Image Datatype keys.
 */
static const int LW_SCI_BUF_IMAGE_ATTR_KEY_START =
           (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) |
           (LwSciBufType_Image << LW_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of ImagePyramid Datatype keys.
 */
static const int LW_SCI_BUF_PYRAMID_ATTR_KEY_START =
           (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) |
           (LwSciBufType_Pyramid << LW_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of LwSciBuf Array Datatype keys.
 */
static const int LW_SCI_BUF_ARRAY_ATTR_KEY_START =
           (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) |
           (LwSciBufType_Array << LW_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of Tensor Datatype keys.
 */
static const int LW_SCI_BUF_TENSOR_ATTR_KEY_START =
           (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) |
           (LwSciBufType_Tensor << LW_SCI_BUF_ATTRKEY_BIT_COUNT);

#else

/**
 * @brief Maximum number of dimensions supported by LwSciBufType_Tensor.
 *
 * @implements{18840096}
 */
#define LW_SCI_BUF_TENSOR_MAX_DIMS  8u

/**
 * @brief Maximum number of planes supported by LwSciBufType_Image.
 *
 * @implements{18840099}
 */
#define LW_SCI_BUF_IMAGE_MAX_PLANES 3u

/**
 * @brief Maximum number of levels supported by LwSciBufType_Pyramid.
 */
#define LW_SCI_BUF_PYRAMID_MAX_LEVELS 10u

/**
 * @brief Indicates the size of export descriptor.
 */
#define LWSCIBUF_EXPORT_DESC_SIZE   32u

/**
 * @brief Global constant to indicate number of bits used for
 * defining an attribute key. Note: Maximum 16K attribute keys
 * per LwSciBufType.
 */
#define LW_SCI_BUF_ATTRKEY_BIT_COUNT  16u

/**
 * @brief Global constant to indicate number of bits used for
 * defining LwSciBufType of an attribute key. Note: Maximum 1K
 * LwSciBufType(s).
 */
#define LW_SCI_BUF_DATATYPE_BIT_COUNT  10u

/**
 * @brief Global constant to indicate the attribute key type is public.
 */
#define LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC 0u

/**
 * @brief Global constant to specify the start-bit of attribute key type.
 */
#define LW_SCI_BUF_KEYTYPE_BIT_START \
        (LW_SCI_BUF_DATATYPE_BIT_COUNT + LW_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates starting value of LwSciBufAttrKey for LwSciBufType_General.
 */
#define LW_SCI_BUF_GENERAL_ATTR_KEY_START \
        (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) | \
        (LwSciBufType_General << LW_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates starting value of LwSciBufAttrKey for LwSciBufType_RawBuffer.
 */
#define LW_SCI_BUF_RAW_BUF_ATTR_KEY_START \
          (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) | \
          (LwSciBufType_RawBuffer << LW_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates the starting value of LwSciBufAttrKey for LwSciBufType_Image.
 */
#define LW_SCI_BUF_IMAGE_ATTR_KEY_START \
          (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) | \
          (LwSciBufType_Image << LW_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates the starting value of LwSciBufAttrKey for LwSciBufType_Pyramid.
 */
#define LW_SCI_BUF_PYRAMID_ATTR_KEY_START \
          (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) | \
          (LwSciBufType_Pyramid << LW_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates the starting value of LwSciBufAttrKey for LwSciBufType_Array.
 */
#define LW_SCI_BUF_ARRAY_ATTR_KEY_START \
          (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) | \
          (LwSciBufType_Array << LW_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates the starting value of LwSciBufAttrKey for LwSciBufType_Tensor.
 */
#define LW_SCI_BUF_TENSOR_ATTR_KEY_START \
          (LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << LW_SCI_BUF_KEYTYPE_BIT_START) | \
          (LwSciBufType_Tensor << LW_SCI_BUF_ATTRKEY_BIT_COUNT)

#endif

/**
 * @}
 */

/**
 * @defgroup lwscibuf_attr_key LwSciBuf Enumerations for Attribute Keys
 * List of all LwSciBuf enumerations for attribute keys.
 * @{
 */

/**
 * @brief Describes the LwSciBuf public attribute keys holding corresponding
 * values specifying buffer constraints.
 * The accessibility property of an attribute refers to whether the value of an
 * attribute is accessible in an LwSciBufAttrList. Input attribute keys specify
 * desired buffer constraints from client and can be set/retrieved by client
 * to/from unreconciled LwSciBufAttrList using
 * LwSciBufAttrListSetAttrs()/LwSciBufAttrListGetAttrs() respectively.
 * Output attribute keys specify actual buffer constraints computed by LwSciBuf
 * if reconciliation succeeds. Output attributes can be retrieved from
 * reconciled LwSciBufAttrList using LwSciBufAttrListGetAttrs().
 * The presence property of an attribute refers to whether the value of an
 * attribute having accessibility as input needs to be present in at least one
 * of the unreconciled attribute lists for reconciliation.
 * The presence property of an attribute can have one of the three values:
 * Mandatory/Optional/Conditional.
 * Mandatory implies that it is mandatory that the value of an attribute be set
 * in at least one of the unreconciled LwSciBufAttrLists ilwolved in
 * reconciliation. Failing to set mandatory input attribute in at least one of
 * the input unreconciled LwSciBufAttrLists results in reconciliation failure.
 * Optional implies that it is not mandatory that value of an attribute be set
 * in at least of the unreconciled LwSciBufAttrLists ilwolved in reconciliation.
 * If the optional input attribute is not set in any of the input unreconciled
 * LwSciBufAttrLists, LwSciBuf uses default value of such attribute to
 * callwlate/reconcile output attributes dependent on such input attribute.
 * Conditional implies that the presence of an attribute is mandatory if
 * condition associated with its presence is satisfied, otherwise its optional.
 *
 * @implements{17824098}
 */
typedef enum {
    /**
     * Specifies the lower bound value to check for a valid LwSciBuf attribute
     * key type.
     */
    LwSciBufAttrKey_LowerBound =         LW_SCI_BUF_GENERAL_ATTR_KEY_START,

    /** An array of all types that the buffer is expected to have. For each type
     * the buffer has, the associated attributes are valid. In order to set
     * @a LwSciBufAttrKeys corresponding to the LwSciBufType, LwSciBufType must
     * be set first using this key.
     * NOTE: A single buffer may have multiple types. For example, a buffer may
     * simultaneously be a LwSciBufType_Image (for integration with LwMedia), a
     * LwSciBufType_Tensor (for integration with TensorRT or LwMedia), and a
     * LwSciBufType_RawBuffer (for integration with LWCA kernels that will
     * directly access it).
     *
     * During reconciliation, if all the LwSciBufTypes
     * specified by all the unreconciled LwSciBufAttrLists are same, this
     * key outputs the specified LwSciBufType. If all LwSciBufTypes are
     * not same, reconciliation succeeds only if the set of LwSciBufTypes
     * contains LwSciBufType_Image and LwSciBufType_Tensor only otherwise
     * reconciliation fails.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if value of this
     * attribute set in any of the input unreconciled LwSciBufAttrList(s) is
     * not present in the set of values of this attribute in the provided
     * reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @ref LwSciBufType[]
     *
     * valid input value: All values defined by LwSciBufType for safety build
     * such that LwSciBufType_General < value < LwSciBufType_MaxValid
     */
    LwSciBufGeneralAttrKey_Types,

    /** Specifies if CPU access is required for the buffer. If this attribute is
     * set to @c true, then the CPU will be able to obtain a pointer to the
     * buffer from LwSciBufObjGetConstCpuPtr() if at least read permissions are
     * granted or from LwSciBufObjGetCpuPtr() if read/write permissions are
     * granted.
     *
     * During reconciliation, reconciler sets value of this key to true in the
     * reconciled LwSciBufAttrList if any of the unreconciled
     * LwSciBufAttrList(s) ilwolved in reconciliation that is owned by the
     * reconciler has this key set to true, otherwise it is set to false in
     * reconciled LwSciBufAttrList.
     *
     * When importing the reconciled LwSciBufAttrList, for every peer owning the
     * unreconciled LwSciBufAttrList(s) ilwolved in reconciliation, if any of
     * the unreconciled LwSciBufAttrList(s) owned by the peer set the key to
     * true then value of this key is true in the reconciled LwSciBufAttrList
     * imported by the peer otherwise its false.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if value of this
     * attribute set in any of the unreconciled LwSciBufAttrList(s) belonging
     * to LwSciIpc channel owner is true and value of the same attribute in
     * reconciled LwSciBufAttrList is false.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: @c bool
     */
    LwSciBufGeneralAttrKey_NeedCpuAccess,

    /** Specifies buffer access permissions.
     * If reconciliation succeeds, granted buffer permissions are reflected in
     * LwSciBufGeneralAttrKey_ActualPerm. If
     * LwSciBufGeneralAttrKey_NeedCpuAccess is true and write permission
     * are granted, then LwSciBufObjGetCpuPtr() can be used to obtain a
     * non-const pointer to the buffer.
     * NOTE: Whether this key is present in reconciled attribute lists is
     * unspecified, as is its value if it is present.
     *
     * Accessibility: Input attribute
     * Presence: Optional
     *
     * Value: @ref LwSciBufAttrValAccessPerm
     *
     * valid input value: LwSciBufAccessPerm_Readonly or
     * LwSciBufAccessPerm_ReadWrite
     */
    LwSciBufGeneralAttrKey_RequiredPerm,

    /** Specifies whether to enable/disable CPU caching.
     * If set to @c true:
     *
     * The CPU must perform write-back caching of the buffer to the greatest
     * extent possible considering all the CPUs that are sharing the buffer.
     *  Coherency is guaranteed with:
     *         - Other CPU accessors.
     *         - All I/O-Coherent accessors that do not have CPU-invisible
     *           caches.
     *
     * If set to @c false:
     *
     * The CPU must not access the caches at all on read or write accesses
     * to the buffer from applications.
     *  Coherency is guaranteed with:
     *         - Other CPU accessors.
     *         - All I/O accessors (whether I/O-coherent or not) that do not
     *              have CPU-invisible caches.
     *
     * During reconciliation, this key is set to true in reconciled
     * LwSciBufAttrList if any of the unreconciled LwSciBufAttrList owned by any
     * peer set it to true, otherwise it is set to false.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if value of this
     * attribute set in any of the unreconciled LwSciBufAttrList(s) is true and
     * value of the same attribute in reconciled LwSciBufAttrList is false.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: @c bool
     */
    LwSciBufGeneralAttrKey_EnableCpuCache,

    /** GpuIDs of the GPUs in the system that will access the buffer.
     * In a multi GPU System, if multiple GPUs are supposed to access
     * the buffer, then provide the GPU IDs of all the GPUs that
     * need to access the buffer. The GPU which is not specified in the
     * list of GPUIDs may not be able to access the buffer.
     *
     * During reconciliation, the value of this attribute in reconciled
     * LwSciBufAttrList is equivalent to the aggregate of all the values
     * specified by all the unreconciled LwSciBufAttrLists ilwolved in
     * reconciliation that have this attribute set. The value of this attribute
     * is set to implementation chosen default value if none of the unreconciled
     * LwSciBufAttrLists specify this attribute. Note that the default value
     * chosen by the implementation must be an invalid LwSciRmGpuId value.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if value of this
     * attribute set in any of the input unreconciled LwSciBufAttrList(s) is
     * not present in the set of values of this attribute in the provided
     * reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: @ref LwSciRmGpuId[]
     *
     * valid input value: Valid LwSciRmGpuId of the GPU(s) present in the
     * system.
     */
    LwSciBufGeneralAttrKey_GpuId,

    /** Indicates whether the CPU is required to flush before reads and
     * after writes. This can be accomplished using
     * LwSciBufObjFlushCpuCacheRange(), or (if the application prefers) with
     * OS-specific flushing functions. It is set to true in reconciled
     * LwSciBufAttrList if both LwSciBufGeneralAttrKey_EnableCpuCache and
     * LwSciBufGeneralAttrKey_NeedCpuAccess are requested by setting them
     * to true in any of the unreconciled LwSciBufAttrList(s) from which
     * reconciled LwSciBufAttrList is obtained and any of the ISO engines would
     * operate on the buffer.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if value of this
     * attribute reconciled from the input unreconciled LwSciBufAttrList(s) is
     * true and value of the same attribute in the provided reconciled
     * LwSciBufAttrList is false.
     *
     * Accessibility: Output attribute
     *
     * Value: @c bool
     */
    LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency,

    /** Specifies the buffer access permissions to the LwSciBufObj.
     * This key is only valid in reconciled LwSciBufAttrList
     * and undefined in unreconciled LwSciBufAttrList.
     *
     * During reconciliation, this attribute is set to the maximum value of the
     * requested permission set in LwSciBufGeneralAttrKey_RequiredPerm of all
     * the unreconciled LwSciBufAttrLists are ilwolved in the reconciliation.
     * This attribute is set to default value of LwSciBufAccessPerm_Readonly if
     * none of the unreconciled LwSciBufAttrLists specify value of the
     * LwSciBufGeneralAttrKey_RequiredPerm attribute.
     *
     * If LwSciBufObj is obtained by calling LwSciBufObjAlloc(),
     * LwSciBufGeneralAttrKey_ActualPerm is set to LwSciBufAccessPerm_ReadWrite
     * in the reconciled LwSciBufAttrList corresponding to it since allocated
     * LwSciBufObj gets read-write permissions by default.
     *
     * For any peer importing the reconciled LwSciBufAttrList, this key is set
     * to maximum value of the requested permission set in
     * LwSciBufGeneralAttrKey_RequiredPerm of all the unreconciled
     * LwSciBufAttrLists that were exported by the peer for reconciliation.
     * The key is set by the reconciler when exporting the reconciled
     * LwSciBufAttrList.
     *
     * For any peer importing the LwSciBufObj, this key is set in the reconciled
     * LwSciBufAttrList to the permissions associated with
     * LwSciBufObjIpcExportDescriptor.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if value of this
     * attribute reconciled from the input unreconciled LwSciBufAttrList(s) is
     * greater than the value of the same attribute in the provided reconciled
     * LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     *
     * Value: LwSciBufAttrValAccessPerm
     */
    LwSciBufGeneralAttrKey_ActualPerm,

    /** GPU ID of dGPU from which vidmem allocation should come when multiple
     * GPUs are sharing buffer. This key should be empty if multiple GPUs
     * access shared buffer from sysmem.
     *
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match. This attribute is set to implementation chosen default
     * value if none of the unreconciled LwSciBufAttrLists ilwolved in
     * reconciliation specify this attribute. Note that the default value
     * chosen by the implementation must be an invalid LwSciRmGpuId value.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: LwSciRmGpuId
     *
     * valid input value: Valid LwSciRmGpuId of the dGPU present in the system.
     */
    LwSciBufGeneralAttrKey_VidMem_GpuId,

    /**
     * TODO: Revisit the description of this attribute from SWAD/SWUD standpoint.
     * Add reconciliation validation description.
     *
     * An array of LwSciBufAttrValGpuCache[] specifying GPU cacheability
     * requirements.
     *
     * Lwrrently, LwSciBuf supports cacheability control for single iGPU and
     * thus if user decides to request cacheability control via this attribute
     * then an array, LwSciBufAttrValGpuCache[] shall provide a single value
     * where GPU ID specified in it is of type iGPU and is part of GPU IDs
     * specified in LwSciBufGeneralAttrKey_GpuId. Not satisfying any of the
     * above conditions results in reconciliation failure.
     *
     * During reconciliation, for all the unreconciled LwSciBufAttrLists
     * ilwolved in reconciliation, the input values of this attribute for a
     * particular GPU ID are taken from
     * a) Value specified by unreconciled LwSciBufAttrLists
     * b) Default value based on table specified below if particular
     * unreconciled LwSciBufAttrList does not specify it.
     * The set of input values are then reconciled using AND policy.
     * The policy specified above is applied for ALL the GPU IDs specified in
     * LwSciBufGeneralAttrKey_GpuId.
     *
     * |----------|---------------|-----------|---------------|
     * | GPU TYPE | MEMORY DOMAIN | PLATFORM  | DEFAULT VALUE |
     * |----------|---------------|-----------|---------------|
     * | iGPU     | Sysmem        | CheetAh     | TRUE          |
     * |----------|---------------|-----------|---------------|
     * | dGPU     | Sysmem        | CheetAh/X86 | FALSE         |
     * |----------|---------------|-----------|---------------|
     * | dGPU     | Vidmem        | CheetAh/X86 | TRUE          |
     * |----------|---------------|-----------|---------------|
     *
     * Type: Input/Output attribute
     * Presence: Optional
     *
     * Datatype: LwSciBufAttrValGpuCache[]
     */
    LwSciBufGeneralAttrKey_EnableGpuCache,

    /**
     * TODO: Revisit the description of this attribute from SWAD/SWUD standpoint.
     * Add reconciliation validation description.
     *
     * An attribute indicating whether application needs to perform GPU cache
     * maintenance before read and after writes. The value of this attribute is
     * set in reconciled LwSciBufAttrList as follows:
     * The value in LwSciBufAttrValGpuCache is set to TRUE for a particular
     * GPU ID in the same struct if,
     * 1) Memory domain is Sysmem AND that particular GPU ID in the
     *     LwSciBufGeneralAttrKey_EnableGpuCache has cacheability value set to
     *     TRUE AND
     *     a) At least one of the GPU IDs in the
     *        LwSciBufGeneralAttrKey_EnableGpuCache has cacheability set to
     *        FALSE. OR
     *     b) At least one of the unreconciled LwSciBufAttrList has requested
     *        CPU access via LwSciBufGeneralAttrKey_NeedCpuAccess OR
     *     c) LwSciBufInternalGeneralAttrKey_EngineArray attribute has at least
     *        one LwSciBufHwEngine set.
     * 2) Memory domain is Vidmem AND that particular GPU ID in the
     *     LwSciBufGeneralAttrKey_EnableGpuCache has cacheability value set to
     *     TRUE AND
     *     a) Any of the engines accessing the buffer are not cache coherent
     *        with Vidmem
     * It is set to FALSE otherwise.
     *
     * Type: Output attribute
     * Datatype: LwSciBufAttrValGpuCache[]
     */
    LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency,

    /**
     * Specifies whether to enable/disable GPU compression for the particular
     * GPU.
     * User can specify the value of this attribute in terms of an
     * array of @a LwSciBufAttrValGpuCompression.
     *
     * During reconciliation, if any of the following conditions are satisfied,
     * the reconciliation fails:
     * 1. The GPU ID specified as the member of @a LwSciBufAttrValGpuCompression
     * does not match with any of the GPU ID values specified as an array in
     * LwSciBufGeneralAttrKey_GpuId attribute.
     * 2. For the particular GPU ID specified in the
     * @a LwSciBufAttrValGpuCompression, the value of @a LwSciBufCompressionType
     * is not the same for that particular GPU ID in all of the unreconciled
     * LwSciBufAttrLists that have specified it.
     *
     * If none of the conditions mentioned above for reconciliation failure are
     * met then this attribute is reconciled as follows:
     * 1. If multiple GPUs request compression via
     * LwSciBufGeneralAttrKey_EnableGpuCompression, reconciliation fills
     * LwSciBufCompressionType_None (aka compression is not enabled) for all
     * GPUs specified in LwSciBufGeneralAttrKey_GpuId.
     * 2. If UMDs set any of the non-GPU HW engines in the unreconciled
     * LwSciBufAttrLists implying that at least one non-GPU engine is going to
     * access the buffer represented by LwSciBufObj,
     * reconciliation fills LwSciBufCompressionType_None (aka compression is not
     * enabled) for all GPUs specified in LwSciBufGeneralAttrKey_GpuId.
     * 3. If LwSciBufGeneralAttrKey_NeedCpuAccess attribute is set in at least
     * one of the unreconciled LwSciBufAttrLists implying that CPU access to the
     * buffer represented by LwSciBufObj is needed, reconciliation fills
     * LwSciBufCompressionType_None (aka compression is not enabled) for all
     * GPUs specified in LwSciBufGeneralAttrKey_GpuId.
     * 4. If none of the above conditions are satisfied then the value of
     * LwSciBufCompressionType for that particular GPU ID is set as the matching
     * value specified by all the unreconciled LwSciBufAttrLists that have set
     * it. LwSciBuf then queries lower level LWPU driver stack to check if
     * reconciled LwSciBufCompressionType is allowed for the particular GPU.
     * LwSciBuf keeps the reconciled value of LwSciBufCompressionType if this
     * compression type is supported, otherwise LwSciBuf falls back to
     * LwSciBufCompressionType_None.
     * 5. For a particular GPU ID specified in LwSciBufGeneralAttrKey_GpuId,
     * if none of the unreconciled LwSciBufAttrLists specify the compression
     * type needed for that GPU ID via this attribute then LwSciBuf fills
     * the default value of LwSciBufCompressionType_None for that GPU ID in the
     * reconciled LwSciBufAttrList.
     *
     * The number of elements in the array value
     * @a LwSciBufAttrValGpuCompression[] of
     * LwSciBufGeneralAttrKey_EnableGpuCompression attribute in the reconciled
     * LwSciBufAttrList is equal to the number of GPU IDs specified in the
     * LwSciBufGeneralAttrKey_GpuId attribute.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * input unreconciled LwSciBufAttrLists yield
     * @a LwSciBufAttrValGpuCompression value of LwSciBufCompressionType_None
     * for the particular GPU ID while value of the same GPU ID in the
     * reconciled LwSciBufAttrList is other than LwSciBufCompressionType_None.
     *
     * Type: Input/Output attribute
     * Presence: Optional
     *
     * Datatype: LwSciBufAttrValGpuCompression[]
     */
    LwSciBufGeneralAttrKey_EnableGpuCompression,

    /** Specifies the size of the buffer to be allocated for
     * LwSciBufType_RawBuffer. Input size specified in unreconciled
     * LwSciBufAttrList should be greater than 0.
     *
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) yields LwSciBufType_RawBuffer and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     * Unit: byte
     *
     * Value: @c uint64_t
     */
    LwSciBufRawBufferAttrKey_Size   =  LW_SCI_BUF_RAW_BUF_ATTR_KEY_START,

    /** Specifies the alignment requirement of LwSciBufType_RawBuffer. Input
     * alignment should be power of 2. If more than one unreconciled
     * LwSciBufAttrLists specify this input attribute, value in the reconciled
     * LwSciBufAttrList corresponds to maximum of the values specified in all of
     * the unreconciled LwSciBufAttrLists. The value of this attribute is set to
     * default alignment with which buffer is allocated if none of the
     * unreconciled LwSciBufAttrList(s) ilwolved in reconciliation specify this
     * attribute.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) yields LwSciBufType_RawBuffer and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     * Unit: byte
     *
     * Value: @c uint64_t
     *
     * valid input value: value is power of 2.
     */
    LwSciBufRawBufferAttrKey_Align,

    /** Specifies the layout of LwSciBufType_Image: Block-linear or
     * Pitch-linear. If more than one unreconciled LwSciBufAttrLists specify
     * this input attribute, reconciliation is successful if all the input
     * attributes of this type match.
     *
     * Only pitch-linear layout is supported for image-tensor buffer type
     * reconciliation.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @ref LwSciBufAttrValImageLayoutType
     *
     * valid input value: Any value defined by LwSciBufAttrValImageLayoutType
     * enum.
     */
    LwSciBufImageAttrKey_Layout   =    LW_SCI_BUF_IMAGE_ATTR_KEY_START,

    /** Specifies the top padding for the LwSciBufType_Image. If more than one
     * unreconciled LwSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match. This attribute is set to default value of 0 if none of the
     * unreconciled LwSciBufAttrList(s) ilwolved in reconciliation specify the
     * attribute.
     *
     * Padding is not allowed to be specified for image-tensor reconciliation.
     * It is allowed for image only buffer reconciliation.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     * Unit: pixel
     *
     * Value: @c uint64_t[]
     */
    LwSciBufImageAttrKey_TopPadding,

    /** Specifies the bottom padding for the LwSciBufType_Image. If more than
     * one unreconciled LwSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match. This attribute is set to default value of 0 if none of the
     * unreconciled LwSciBufAttrList(s) ilwolved in reconciliation specify the
     * attribute.
     *
     * Padding is not allowed to be specified for image-tensor reconciliation.
     * It is allowed for image only buffer reconciliation.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     * Unit: pixel
     *
     * Value: uint64_t[]
     */
    LwSciBufImageAttrKey_BottomPadding,

    /** Specifies the left padding for the LwSciBufType_Image. If more than one
     * unreconciled LwSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match. This attribute is set to default value of 0 if none of the
     * unreconciled LwSciBufAttrList(s) ilwolved in reconciliation specify the
     * attribute.
     *
     * Padding is not allowed to be specified for image-tensor reconciliation.
     * It is allowed for image only buffer reconciliation.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     * Unit: pixel
     *
     * Value: @c uint64_t[]
     */
    LwSciBufImageAttrKey_LeftPadding,

    /** Specifies the right padding for the LwSciBufType_Image. If more than one
     * unreconciled LwSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match. This attribute is set to default value of 0 if none of the
     * unreconciled LwSciBufAttrList(s) ilwolved in reconciliation specify the
     * attribute.
     *
     * Padding is not allowed to be specified for image-tensor reconciliation.
     * It is allowed for image only buffer reconciliation.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     * Unit: pixel
     *
     * Value: @c uint64_t[]
     */
    LwSciBufImageAttrKey_RightPadding,

    /** Specifies the VPR flag for the LwSciBufType_Image.
     *
     * During reconciliation, this key is set to true in reconciled
     * LwSciBufAttrList if any of the unreconciled LwSciBufAttrList owned by any
     * peer set it to true, otherwise it is set to false.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is true and value of same attribute in the
     * provided reconciled LwSciBufAttrList is false.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: @c bool
     */
    LwSciBufImageAttrKey_VprFlag,

    /** Output size of the LwSciBufType_Image after successful reconciliation.
     * The output size for this key is computed by aggregating size of all the
     * planes in the output key LwSciBufImageAttrKey_PlaneAlignedSize.
     *
     * The size is callwlated the following way:
     * LwSciBufImageAttrKey_Size = sum of LwSciBufImageAttrKey_PlaneAlignedSize
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t
     */
    LwSciBufImageAttrKey_Size,

    /** Output alignment of the LwSciBufType_Image after successful
     * reconciliation.
     * The output value of this key is same as alignment value of the first
     * plane in the key LwSciBufImageAttrKey_PlaneBaseAddrAlign after
     * reconciliation.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t
     */
    LwSciBufImageAttrKey_Alignment,

    /** Specifies the number of planes for LwSciBufType_Image. If more than one
     * unreconciled LwSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match. If LwSciBufType_Image and LwSciBufType_Tensor are ilwolved in
     * reconciliation and if this attribute is set in any of unreconciled
     * LwSciBufAttrList(s) to be reconciled, the value of this attribute should
     * be 1.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. This is Mandatory when surface-based image
     * attributes are not used, and should not be specified otherwise.
     *
     * Value: @c uint32_t
     *
     * valid input value: 1 <= value <= LW_SCI_BUF_IMAGE_MAX_PLANES
     */
    LwSciBufImageAttrKey_PlaneCount,

    /** Specifies the LwSciBufAttrValColorFmt of the LwSciBufType_Image plane.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. This is Mandatory when surface-based image
     * attributes are not used, and should not be specified otherwise.
     *
     * Value: @ref LwSciBufAttrValColorFmt[]
     *
     * valid input value: LwSciColor_LowerBound < value < LwSciColor_UpperBound
     */
    LwSciBufImageAttrKey_PlaneColorFormat,

    /** Specifies a set of plane color standards. If more than
     * one unreconciled LwSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match. This attribute is set to implementation chosen default value
     * if none of the unreconciled LwSciBufAttrList(s) ilwolved in
     * reconciliation specify this attribute.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: @ref LwSciBufAttrValColorStd[]
     *
     * valid input value: Any value defined by LwSciBufAttrValColorStd enum.
     */
    LwSciBufImageAttrKey_PlaneColorStd,

    /** Specifies the LwSciBufType_Image plane base address alignment for every
     * plane in terms of an array. Input alignment must be power of 2.
     * If more than one unreconciled LwSciBufAttrLists specify this attribute,
     * reconciled LwSciBufAttrList has maximum alignment value per array index
     * of the values specified in unreconciled LwSciBufAttrLists for the same
     * array index. On top of that, for all the HW engines for which buffer is
     * being allocated, if the maximum start address alignment constraint of all
     * the engines taken together is greater than the reconciled alignment value
     * at any index, it is replaced with start address alignment value. In other
     * words,
     * reconciled alignment per array index =
     * MAX(MAX(alignments in unreconciled list at the same index),
     * MAX(start address alignment constraint of all the engines))
     * The value of this attribute is set to default alignment with which buffer
     * is allocated if none of the unreconciled LwSciBufAttrList(s) ilwolved in
     * reconciliation specify this attribute.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     * Unit: byte
     *
     * Value: @c uint32_t[]
     *
     * valid input value: value is power of 2.
     */
    LwSciBufImageAttrKey_PlaneBaseAddrAlign,

    /** Specifies the LwSciBufType_Image plane width in pixels. If more than
     * one unreconciled LwSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. This is Mandatory when surface-based image
     * attributes are not used, and should not be specified otherwise.
     * Unit: pixel
     *
     * Value: @c uint32_t[]
     */
    LwSciBufImageAttrKey_PlaneWidth,

    /** Specifies the LwSciBufType_Image plane height in number of pixels. If
     * more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. This is Mandatory when surface-based image
     * attributes are not used, and should not be specified otherwise.
     * Unit: pixel
     *
     * Value: @c uint32_t[]
     */
    LwSciBufImageAttrKey_PlaneHeight,

    /** Specifies the LwSciBufType_Image scan type: Progressive or Interlaced.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes
     * of this type match.
     *
     * @note LwSciBufImageAttrKey_PlaneScanType is deprecated and may be
     * removed in some future release. Use LwSciBufImageAttrKey_ScanType
     * wherever possible.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @ref LwSciBufAttrValImageScanType
     *
     * valid input value: Any value defined by LwSciBufAttrValImageScanType
     * enum.
     */
    LwSciBufImageAttrKey_PlaneScanType = 0x2000e,
    LwSciBufImageAttrKey_ScanType = LwSciBufImageAttrKey_PlaneScanType,

    /** Outputs number of bits per pixel corresponding to the
     * LwSciBufAttrValColorFmt for each plane specified in
     * LwSciBufImageAttrKey_PlaneColorFormat.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     *
     * Value: @c uint32_t[]
     */
    LwSciBufImageAttrKey_PlaneBitsPerPixel,

    /** Indicates the starting offset of the LwSciBufType_Image plane from the
     * first plane.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t[]
     */
    LwSciBufImageAttrKey_PlaneOffset,

    /** Outputs the LwSciBufAttrValDataType of each plane based on the
     * LwSciBufAttrValColorFmt provided in
     * LwSciBufImageAttrKey_PlaneColorFormat for every plane.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     *
     * Value: @ref LwSciBufAttrValDataType[]
     */
    LwSciBufImageAttrKey_PlaneDatatype,

    /** Outputs number of channels per plane.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     *
     * Value: @c uint8_t[]
     */
    LwSciBufImageAttrKey_PlaneChannelCount,

    /** Indicates the offset of the start of the second field, 0 for progressive
     * valid for interlaced.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t[]
     */
    LwSciBufImageAttrKey_PlaneSecondFieldOffset,

    /** Outputs the pitch (aka width in bytes) for every plane. The pitch is
     * aligned to the maximum of the pitch alignment constraint value of all the
     * HW engines that are going to operate on the buffer.
     *
     * The pitch is callwalted the following way:
     * LwSciBufImageAttrKey_PlanePitch =
     *     (LwSciBufImageAttrKey_PlaneWidth * (Bits per pixel for
     *      LwSciBufImageAttrKey_PlaneColorFormat)) / 8
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint32_t[]
     */
    LwSciBufImageAttrKey_PlanePitch,

    /** Outputs the aligned height of evey plane in terms of number of pixels.
     * This height is callwlated by aligning value for every plane provided in
     * LwSciBufImageAttrKey_PlaneHeight with maximum of the height alignment
     * constraints of all the engines that are going to operate on the buffer.
     *
     * The height is callwlated the following way:
     * If (LwSciBufImageAttrKey_ScanType == LwSciBufScan_InterlaceType)
     *     LwSciBufImageAttrKey_PlaneAlignedHeight =
     *     (LwSciBufImageAttrKey_PlaneHeight / 2)
     *     This value is aligned to highest height HW constraints among all
     *     the HW engines in LwSciBufInternalGeneralAttrKey_EngineArray
     * Else
     *     LwSciBufImageAttrKey_PlaneAlignedHeight =
     *     LwSciBufImageAttrKey_PlaneHeight
     *     This value is aligned to highest height HW constraints among all
     *     the HW engines in LwSciBufInternalGeneralAttrKey_EngineArray
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: pixel
     *
     * Value: @c uint32_t[]
     */
    LwSciBufImageAttrKey_PlaneAlignedHeight,

    /** Indicates the aligned size of every plane. The size is callwlated from
     * the value of LwSciBufImageAttrKey_PlanePitch and
     * LwSciBufImageAttrKey_PlaneAlignedHeight.
     *
     * The size is callwlated the following way:
     * If (LwSciBufImageAttrKey_ScanType == LwSciBufScan_InterlaceType)
     *     LwSciBufImageAttrKey_PlaneAlignedSize =
     *         LwSciBufImageAttrKey_PlanePitch *
     *         LwSciBufImageAttrKey_PlaneAlignedHeight * 2
     * Else
     *     LwSciBufImageAttrKey_PlaneAlignedSize =
     *         LwSciBufImageAttrKey_PlanePitch *
     *         LwSciBufImageAttrKey_PlaneAlignedHeight
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t[]
     */
    LwSciBufImageAttrKey_PlaneAlignedSize,

    /** Attribute to specify number of LwSciBufType_Image(s) for which buffer
     * should be allocated. If more than one unreconciled LwSciBufAttrLists
     * specify this input attribute, reconciliation is successful if all the
     * input attributes of this type match. This attribute is set to default
     * value of 1 if none of the unreconciled LwSciBufAttrList(s) specify this
     * attribute and the condition for the optional presence of this attribute
     * is satisfied.
     * LwSciBuf supports allocating buffer for single image only and thus, this
     * attribute should be set to 1. A single buffer cannot be allocated for
     * multiple images. Allocating 'N' buffers corresponding to 'N' images is
     * allowed.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory for Image/Tensor reconciliation,
     * Optional otherwise.
     *
     * Value: @c uint64_t
     *
     * valid input value: 1
     */
    LwSciBufImageAttrKey_ImageCount,

    /**
     * Specifies the LwSciBufSurfType. If more than one unreconciled
     * LwSciBufAttrList specifies this input attribute, reconciliation is
     * successful if all the input attributes of this type match. This value is
     * set on the reconciled LwSciBufAttrList. This attribute is unset in the
     * reconciled LwSciBufAttrList if no surface-based image attributes were
     * requested in the unreconciled LwSciBufAttrList(s).
     *
     * During validation of a reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList
     *   when it is set in at least one of the unreconciled LwSciBufAttrList(s)
     *   that are being verified against this reconciled LwSciBufAttrList
     * - The value of this attribute set in any of the input unreconciled
     *   LwSciBufAttrList(s) is not equal to the value of same attribute in the
     *   provided reconciled LwSciBufAttrList.
     *
     * @note This is a colwenience attribute key. Such surface-based attribute
     * keys are mtually exclusive with the plane-based attribute keys. If both
     * types of attribute keys are specified then reconciliation will fail.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory when surface-based image attributes are
     * used, and should not be specified otherwise.
     *
     * Value: @ref LwSciBufSurfType
     */
    LwSciBufImageAttrKey_SurfType,

    /**
     * Species the LwSciBufSurfMemLayout. If more than one unreconciled
     * LwSciBufAttrList specifies this input attribute, reconciliation is
     * successful if all the input attributes of this type match. This value is
     * set on the reconciled LwSciBufAttrList. This attribute is unset in the
     * reconciled LwSciBufAttrList if no surface-based image attributes were
     * requested in the unreconciled LwSciBufAttrList(s).
     *
     * During validation of a reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList
     *   when it is set in at least one of the unreconciled LwSciBufAttrList(s)
     *   that are being verified against this reconciled LwSciBufAttrList
     * - The value of this attribute set in any of the input unreconciled
     *   LwSciBufAttrList(s) is not equal to the value of same attribute in the
     *   provided reconciled LwSciBufAttrList.
     *
     * @note This is a colwenience attribute key. Such surface-based attribute
     * keys are mtually exclusive with the plane-based attribute keys. If both
     * types of attribute keys are specified then reconciliation will fail.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory when surface-based image attributes are
     * used, and should not be specified otherwise.
     *
     * Value: @ref LwSciBufSurfMemLayout
     */
    LwSciBufImageAttrKey_SurfMemLayout,

    /**
     * Specifies the LwSciBufSurfSampleType. If more than one unreconciled
     * LwSciBufAttrList specifies this input attribute, reconciliation is
     * successful if all the input attributes of this type match. This value is
     * set on the reconciled LwSciBufAttrList. This attribute is unset in the
     * reconciled LwSciBufAttrList if no surface-based image attributes were
     * requested in the unreconciled LwSciBufAttrList(s).
     *
     * During validation of a reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList
     *   when it is set in at least one of the unreconciled LwSciBufAttrList(s)
     *   that are being verified against this reconciled LwSciBufAttrList
     * - The value of this attribute set in any of the input unreconciled
     *   LwSciBufAttrList(s) is not equal to the value of same attribute in the
     *   provided reconciled LwSciBufAttrList.
     *
     * @note This is a colwenience attribute key. Such surface-based attribute
     * keys are mtually exclusive with the plane-based attribute keys. If both
     * types of attribute keys are specified then reconciliation will fail.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory when surface-based image attributes are
     * used, and should not be specified otherwise.
     *
     * Value: @ref LwSciBufSurfSampleType
     *
     * valid input value:
     */
    LwSciBufImageAttrKey_SurfSampleType,

    /**
     * Specifies the LwSciBufSurfBPC. If more than one unreconciled
     * LwSciBufAttrList specifies this input attribute, reconciliation is
     * successful if all the input attributes of this type match. This value is
     * set on the reconciled LwSciBufAttrList. This attribute is unset in the
     * reconciled LwSciBufAttrList if no surface-based image attributes were
     * requested in the unreconciled LwSciBufAttrList(s).
     *
     * During validation of a reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList
     *   when it is set in at least one of the unreconciled LwSciBufAttrList(s)
     *   that are being verified against this reconciled LwSciBufAttrList
     * - The value of this attribute set in any of the input unreconciled
     *   LwSciBufAttrList(s) is not equal to the value of same attribute in the
     *   provided reconciled LwSciBufAttrList.
     *
     * @note This is a colwenience attribute key. Such surface-based attribute
     * keys are mtually exclusive with the plane-based attribute keys. If both
     * types of attribute keys are specified then reconciliation will fail.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory when surface-based image attributes are
     * used, and should not be specified otherwise.
     *
     * Value: @ref LwSciBufSurfBPC
     */
    LwSciBufImageAttrKey_SurfBPC,

    /**
     * Specifies the LwSciSurfComponentOrder. If more than one unreconciled
     * LwSciBufAttrList specifies this input attribute, reconciliation is
     * successful if all the input attributes of this type match. This value is
     * set on the reconciled LwSciBufAttrList. This attribute is unset in the
     * reconciled LwSciBufAttrList if no surface-based image attributes were
     * requested in the unreconciled LwSciBufAttrList(s).
     *
     * During validation of a reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList
     *   when it is set in at least one of the unreconciled LwSciBufAttrList(s)
     *   that are being verified against this reconciled LwSciBufAttrList
     * - The value of this attribute set in any of the input unreconciled
     *   LwSciBufAttrList(s) is not equal to the value of same attribute in the
     *   provided reconciled LwSciBufAttrList.
     *
     * @note This is a colwenience attribute key. Such surface-based attribute
     * keys are mtually exclusive with the plane-based attribute keys. If both
     * types of attribute keys are specified then reconciliation will fail.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory when surface-based image attributes are
     * used, and should not be specified otherwise.
     *
     * Value: @ref LwSciSurfComponentOrder
     */
    LwSciBufImageAttrKey_SurfComponentOrder,

    /**
     * Specifies the surface base width. If more than one unreconciled
     * LwSciBufAttrList specifies this input attribute, reconciliation is
     * successful if all the input attributes of this type match. This value is
     * set on the reconciled LwSciBufAttrList. This attribute is unset in the
     * reconciled LwSciBufAttrList if no surface-based image attributes were
     * requested in the unreconciled LwSciBufAttrList(s).
     *
     * During validation of a reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList
     *   when it is set in at least one of the unreconciled LwSciBufAttrList(s)
     *   that are being verified against this reconciled LwSciBufAttrList
     * - The value of this attribute set in any of the input unreconciled
     *   LwSciBufAttrList(s) is not equal to the value of same attribute in the
     *   provided reconciled LwSciBufAttrList.
     *
     * @note This is a colwenience attribute key. Such surface-based attribute
     * keys are mtually exclusive with the plane-based attribute keys. If both
     * types of attribute keys are specified then reconciliation will fail.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory when surface-based image attributes are
     * used, and should not be specified otherwise.
     * Unit: pixel
     *
     * Value: @c uint32_t
     */
    LwSciBufImageAttrKey_SurfWidthBase,

    /**
     * Specifies the Surface base height. If more than one unreconciled
     * LwSciBufAttrList specifies this input attribute, reconciliation is
     * successful if all the input attributes of this type match. This value is
     * set on the reconciled LwSciBufAttrList. This attribute is unset in the
     * reconciled LwSciBufAttrList if no surface-based image attributes were
     * requested in the unreconciled LwSciBufAttrList(s).
     *
     * During validation of a reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Image and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList
     *   when it is set in at least one of the unreconciled LwSciBufAttrList(s)
     *   that are being verified against this reconciled LwSciBufAttrList
     * - The value of this attribute set in any of the input unreconciled
     *   LwSciBufAttrList(s) is not equal to the value of same attribute in the
     *   provided reconciled LwSciBufAttrList.
     *
     * @note This is a colwenience attribute key. Such surface-based attribute
     * keys are mtually exclusive with the plane-based attribute keys. If both
     * types of attribute keys are specified then reconciliation will fail.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory when surface-based image attributes are
     * used, and should not be specified otherwise.
     * Unit: pixel
     *
     * Value: @c uint32_t
     */
    LwSciBufImageAttrKey_SurfHeightBase,

    /** Specifies the tensor data type.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @ref LwSciBufAttrValDataType
     *
     * valid input value: LwSciDataType_Int4 <= value <= LwSciDataType_Float32
     */
    LwSciBufTensorAttrKey_DataType  =  LW_SCI_BUF_TENSOR_ATTR_KEY_START,

    /** Specifies the number of tensor dimensions. A maximum of 8 dimensions are
     * allowed. If more than one unreconciled LwSciBufAttrLists specify this
     * input attribute, reconciliation is successful if all the input attributes
     * of this type match.
     * If LwSciBufType_Image and LwSciBufType_Tensor LwSciBufTypes are used
     * in reconciliation, reconciliation succeeds only if this key is set
     * to 4, since LwSciBuf only supports reconciliation of LwSciBufType_Tensor
     * of NHWC type with LwSciBufType_Image.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @c uint32_t
     *
     * valid input value: 1 <= value <= LW_SCI_BUF_TENSOR_MAX_DIMS
     */
    LwSciBufTensorAttrKey_NumDims,

    /** Specifies the size of each tensor dimension.
     * This attribute takes size value in terms of an array.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match. Number of elements in value array of this attribute
     * should not be less than value specified by LwSciBufTensorAttrKey_NumDims
     * attribute.
     * @note Array indices are not tied to the semantics of the
     * dimension if LwSciBufType_Tensor is the only LwSciBufType ilwolved
     * in reconciliation. If LwSciBufType_Tensor and LwSciBufType_Image
     * are ilwolved in reconciliation, LwSciBuf only supports
     * reconciliation of LwSciBufType_Image with NHWC LwSciBufType_Tensor
     * where N=1 and thus reconciliation succeeds only if value of
     * dimension 0 is 1 and it matches with value of
     * LwSciBufImageAttrKey_ImageCount, value of dimension 1 matches with value
     * of LwSciBufImageAttrKey_PlaneHeight, value of dimension 2 matches with
     * value of LwSciBufImageAttrKey_PlaneWidth and dimension 3 specifies
     * channel count for LwSciBufAttrValColorFmt specified in
     * LwSciBufTensorAttrKey_PixelFormat key
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @c uint64_t[]
     */
    LwSciBufTensorAttrKey_SizePerDim,

    /** Specifies the alignment constraints per tensor dimension.
     * Number of elements in value array of this attribute should not be less
     * than value specified by LwSciBufTensorAttrKey_NumDims attribute. Value of
     * every element in the value array should be power of two. If more than one
     * unreconciled LwSciBufAttrLists specify this input attribute, value in the
     * reconciled LwSciBufAttrList corresponds to maximum of the values
     * specified in all of the unreconciled LwSciBufAttrLists that have set this
     * attribute. The value of this attribute is set to default alignment with
     * which buffer is allocated if none of the unreconciled LwSciBufAttrList(s)
     * ilwolved in reconciliation specify this attribute and condition for the
     * optional presence of this attribute is satisfied.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory for Tensor only reconciliation, optional
     * otherwise.
     * Unit: byte
     *
     * Value: @c uint32_t[]
     *
     * valid input value: value is power of 2.
     */
    LwSciBufTensorAttrKey_AlignmentPerDim,

    /** Returns the stride value (in bytes) for each tensor dimension.
     * @note The returned array contains stride values in decreasing order.
     * In other words, the index @em 0 of the array will have the largest
     * stride while [@em number-of-dims - 1] index will have the smallest stride.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t[]
     */
    LwSciBufTensorAttrKey_StridesPerDim,

    /** Attribute providing pixel format of the tensor. This key needs to be
     * set only if LwSciBufType_Image and LwSciBufType_Tensor are ilwolved in
     * reconciliation.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match. Additionally, reconciliation succeeds only if value of
     * this attribute matches with the value of
     * LwSciBufImageAttrKey_PlaneColorFormat in all the input unreconciled
     * LwSciBufAttrList(s) that have set it.
     * Image/Tensor reconciliation only supports LwSciColor_A8B8G8R8 and
     * LwSciColor_Float_A16B16G16R16 color formats as of now. This attribute is
     * set to default value if none of the unreconciled LwSciBufAttrList(s)
     * ilwolved in reconciliation specify this attribute and condition for the
     * optional presence of this attribute is satisfied.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is not equal to the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Conditional. Mandatory for Image/Tensor reconciliation,
     * optional otherwise.
     *
     * Value: @ref LwSciBufAttrValColorFmt
     *
     * valid input value: LwSciColor_LowerBound < value < LwSciColor_UpperBound
     */
    LwSciBufTensorAttrKey_PixelFormat,

    /** Attribute providing base address alignment requirements for tensor.
     * Input value provided for this attribute must be power of two. Output
     * value of this attribute is always power of two. If more than one
     * unreconciled LwSciBufAttrLists specify this input attribute, value in the
     * reconciled LwSciBufAttrList corresponds to maximum of the values
     * specified in all of the unreconciled LwSciBufAttrLists that have set this
     * attribute.
     * The value of this attribute is set to default alignment with
     * which buffer is allocated if none of the unreconciled LwSciBufAttrList(s)
     * ilwolved in reconciliation specify this attribute and condition for the
     * optional presence of this attribute is satisfied.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute set in any of the input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     * Unit: byte
     *
     * Value: @c uint64_t
     *
     * valid input value: value is power of 2.
     */
    LwSciBufTensorAttrKey_BaseAddrAlign,

    /** Size of buffer allocated for 'N' tensors.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if reconciliation of
     * LwSciBufGeneralAttrKey_Types attribute from the input unreconciled
     * LwSciBufAttrList(s) contains LwSciBufType_Tensor and any of the
     * following conditions is satisfied:
     * - This attribute is not set in the provided reconciled LwSciBufAttrList.
     * - Value of this attribute reconciled from input unreconciled
     * LwSciBufAttrList(s) is greater than the value of same attribute in the
     * provided reconciled LwSciBufAttrList.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t
     */
    LwSciBufTensorAttrKey_Size,

    /** Specifies the data type of a LwSciBufType_Array.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @ref LwSciBufAttrValDataType
     *
     * valid input value: LwSciDataType_Int4 <= value <= LwSciDataType_Bool
     */
    LwSciBufArrayAttrKey_DataType   =  LW_SCI_BUF_ARRAY_ATTR_KEY_START,

    /** Specifies the stride of each element in the LwSciBufType_Array.
     * Stride must be greater than or equal to size of datatype specified by
     * LwSciBufArrayAttrKey_DataType.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     * Unit: byte
     *
     * Value: @c uint64_t
     */
    LwSciBufArrayAttrKey_Stride,

    /** Specifies the LwSciBufType_Array capacity.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @c uint64_t
     */
    LwSciBufArrayAttrKey_Capacity,

    /** Indicates the total size of a LwSciBufType_Array.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t
     */
    LwSciBufArrayAttrKey_Size,

    /** Indicates the base alignment of a LwSciBufType_Array.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t
     */
    LwSciBufArrayAttrKey_Alignment,

    /** Specifies the number of levels of images in a pyramid.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @c uint32_t
     *
     * valid input value: 1 <= value <= LW_SCI_BUF_PYRAMID_MAX_LEVELS
     */
    LwSciBufPyramidAttrKey_NumLevels  =  LW_SCI_BUF_PYRAMID_ATTR_KEY_START,

    /** Specifies the scaling factor by which each successive image in a
     * pyramid must be scaled.
     * If more than one unreconciled LwSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Accessibility: Input/Output attribute
     * Presence: Mandatory
     *
     * Value: @c float
     *
     * valid input value: 0.0f < value <= 1.0f
     */
    LwSciBufPyramidAttrKey_Scale,

    /** LwSciBuf allocates the total buffer size considering all levels in an
     * image pyramid and returns an array of buffer offsets for each level.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t[]
     */
    LwSciBufPyramidAttrKey_LevelOffset,

    /** LwSciBuf allocates the total buffer size considering all levels in an
     * image pyramid and returns an array of buffer sizes for each level.
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: @c uint64_t[]
     */
    LwSciBufPyramidAttrKey_LevelSize,

    /** Alignment attribute of pyramid
     *
     * Accessibility: Output attribute
     * Unit: byte
     *
     * Value: uint64_t
     */
    LwSciBufPyramidAttrKey_Alignment,

    /** Specifies the maximum number of LwSciBuf attribute keys.
     * The total space for keys is 32K.
     *
     * Value: None
     */
    LwSciBufAttrKey_UpperBound = 0x3ffffffU,

} LwSciBufAttrKey;

/**
 * @}
 */

/**
 * @addtogroup lwscibuf_datatype
 * @{
 */

/**
 * @brief Defines buffer access permissions for LwSciBufObj.
 *
 * @implements{18840072}
 */
typedef enum {
    LwSciBufAccessPerm_Readonly = 1,
    LwSciBufAccessPerm_ReadWrite = 3,
    /** Usage of Auto permissions is restricted only for export, import APIs
     * and shouldn't be used to set value for
     * LwSciBufGeneralAttrKey_RequiredPerm Attribute */
    LwSciBufAccessPerm_Auto,
    LwSciBufAccessPerm_Ilwalid,
} LwSciBufAttrValAccessPerm;

/**
 * @brief Defines the image layout type for LwSciBufType_Image.
 *
 * @implements{18840075}
 */
typedef enum {
    LwSciBufImage_BlockLinearType,
    LwSciBufImage_PitchLinearType,
} LwSciBufAttrValImageLayoutType;

/**
 * @brief Defines the image scan type for LwSciBufType_Image.
 *
 * @implements{18840078}
 */
typedef enum {
    LwSciBufScan_ProgressiveType,
    LwSciBufScan_InterlaceType,
} LwSciBufAttrValImageScanType;

/**
 * @brief Defines the image color formats for LwSciBufType_Image.
 *
 * @implements{18840081}
 */
typedef enum {
    LwSciColor_LowerBound,
    /* RAW PACKED */
    LwSciColor_Bayer8RGGB,
    LwSciColor_Bayer8CCCC,
    LwSciColor_Bayer8BGGR,
    LwSciColor_Bayer8GBRG,
    LwSciColor_Bayer8GRBG,
    LwSciColor_Bayer16BGGR,
    LwSciColor_Bayer16CCCC,
    LwSciColor_Bayer16GBRG,
    LwSciColor_Bayer16GRBG,
    LwSciColor_Bayer16RGGB,
    LwSciColor_Bayer16RCCB,
    LwSciColor_Bayer16BCCR,
    LwSciColor_Bayer16CRBC,
    LwSciColor_Bayer16CBRC,
    LwSciColor_Bayer16RCCC,
    LwSciColor_Bayer16CCCR,
    LwSciColor_Bayer16CRCC,
    LwSciColor_Bayer16CCRC,
    LwSciColor_X2Bayer14GBRG,
    LwSciColor_X4Bayer12GBRG,
    LwSciColor_X6Bayer10GBRG,
    LwSciColor_X2Bayer14GRBG,
    LwSciColor_X4Bayer12GRBG,
    LwSciColor_X6Bayer10GRBG,
    LwSciColor_X2Bayer14BGGR,
    LwSciColor_X4Bayer12BGGR,
    LwSciColor_X6Bayer10BGGR,
    LwSciColor_X2Bayer14RGGB,
    LwSciColor_X4Bayer12RGGB,
    LwSciColor_X6Bayer10RGGB,
    LwSciColor_X2Bayer14CCCC,
    LwSciColor_X4Bayer12CCCC,
    LwSciColor_X6Bayer10CCCC,
    LwSciColor_X4Bayer12RCCB,
    LwSciColor_X4Bayer12BCCR,
    LwSciColor_X4Bayer12CRBC,
    LwSciColor_X4Bayer12CBRC,
    LwSciColor_X4Bayer12RCCC,
    LwSciColor_X4Bayer12CCCR,
    LwSciColor_X4Bayer12CRCC,
    LwSciColor_X4Bayer12CCRC,
    LwSciColor_Signed_X2Bayer14CCCC,
    LwSciColor_Signed_X4Bayer12CCCC,
    LwSciColor_Signed_X6Bayer10CCCC,
    LwSciColor_Signed_Bayer16CCCC,
    LwSciColor_FloatISP_Bayer16CCCC,
    LwSciColor_FloatISP_Bayer16RGGB,
    LwSciColor_FloatISP_Bayer16BGGR,
    LwSciColor_FloatISP_Bayer16GRBG,
    LwSciColor_FloatISP_Bayer16GBRG,
    LwSciColor_FloatISP_Bayer16RCCB,
    LwSciColor_FloatISP_Bayer16BCCR,
    LwSciColor_FloatISP_Bayer16CRBC,
    LwSciColor_FloatISP_Bayer16CBRC,
    LwSciColor_FloatISP_Bayer16RCCC,
    LwSciColor_FloatISP_Bayer16CCCR,
    LwSciColor_FloatISP_Bayer16CRCC,
    LwSciColor_FloatISP_Bayer16CCRC,
    LwSciColor_X12Bayer20CCCC,
    LwSciColor_X12Bayer20BGGR,
    LwSciColor_X12Bayer20RGGB,
    LwSciColor_X12Bayer20GRBG,
    LwSciColor_X12Bayer20GBRG,
    LwSciColor_X12Bayer20RCCB,
    LwSciColor_X12Bayer20BCCR,
    LwSciColor_X12Bayer20CRBC,
    LwSciColor_X12Bayer20CBRC,
    LwSciColor_X12Bayer20RCCC,
    LwSciColor_X12Bayer20CCCR,
    LwSciColor_X12Bayer20CRCC,
    LwSciColor_X12Bayer20CCRC,
    LwSciColor_Signed_X12Bayer20CCCC,
    LwSciColor_Signed_X12Bayer20GBRG,

    /* Semiplanar formats */
    LwSciColor_U8V8,
    LwSciColor_U8_V8,
    LwSciColor_V8U8,
    LwSciColor_V8_U8,
    LwSciColor_U10V10,
    LwSciColor_V10U10,
    LwSciColor_U12V12,
    LwSciColor_V12U12,
    LwSciColor_U16V16,
    LwSciColor_V16U16,

    /* PLANAR formats */
    LwSciColor_Y8,
    LwSciColor_Y10,
    LwSciColor_Y12,
    LwSciColor_Y16,
    LwSciColor_U8,
    LwSciColor_V8,
    LwSciColor_U10,
    LwSciColor_V10,
    LwSciColor_U12,
    LwSciColor_V12,
    LwSciColor_U16,
    LwSciColor_V16,

    /* Packed YUV formats */
    LwSciColor_A8Y8U8V8,
    LwSciColor_Y8U8Y8V8,
    LwSciColor_Y8V8Y8U8,
    LwSciColor_U8Y8V8Y8,
    LwSciColor_V8Y8U8Y8,
    LwSciColor_A16Y16U16V16,

    /* RGBA PACKED */
    LwSciColor_A8,
    LwSciColor_Signed_A8,
    LwSciColor_B8G8R8A8,
    LwSciColor_A8R8G8B8,
    LwSciColor_A8B8G8R8,
    LwSciColor_A2R10G10B10,
    LwSciColor_A16,
    LwSciColor_Signed_A16,
    LwSciColor_Signed_R16G16,
    LwSciColor_A16B16G16R16,
    LwSciColor_Signed_A16B16G16R16,
    LwSciColor_Float_A16B16G16R16,
    LwSciColor_A32,
    LwSciColor_Signed_A32,
    LwSciColor_Float_A16,

    /* 10-bit 4x4 RGB-IR Bayer formats */
    LwSciColor_X6Bayer10BGGI_RGGI,
    LwSciColor_X6Bayer10GBIG_GRIG,
    LwSciColor_X6Bayer10GIBG_GIRG,
    LwSciColor_X6Bayer10IGGB_IGGR,
    LwSciColor_X6Bayer10RGGI_BGGI,
    LwSciColor_X6Bayer10GRIG_GBIG,
    LwSciColor_X6Bayer10GIRG_GIBG,
    LwSciColor_X6Bayer10IGGR_IGGB,

    /* 12-bit 4x4 RGB-IR Bayer formats */
    LwSciColor_X4Bayer12BGGI_RGGI,
    LwSciColor_X4Bayer12GBIG_GRIG,
    LwSciColor_X4Bayer12GIBG_GIRG,
    LwSciColor_X4Bayer12IGGB_IGGR,
    LwSciColor_X4Bayer12RGGI_BGGI,
    LwSciColor_X4Bayer12GRIG_GBIG,
    LwSciColor_X4Bayer12GIRG_GIBG,
    LwSciColor_X4Bayer12IGGR_IGGB,

    /* 14-bit 4x4 RGB-IR Bayer formats */
    LwSciColor_X2Bayer14BGGI_RGGI,
    LwSciColor_X2Bayer14GBIG_GRIG,
    LwSciColor_X2Bayer14GIBG_GIRG,
    LwSciColor_X2Bayer14IGGB_IGGR,
    LwSciColor_X2Bayer14RGGI_BGGI,
    LwSciColor_X2Bayer14GRIG_GBIG,
    LwSciColor_X2Bayer14GIRG_GIBG,
    LwSciColor_X2Bayer14IGGR_IGGB,

    /* 16-bit 4x4 RGB-IR Bayer formats */
    LwSciColor_Bayer16BGGI_RGGI,
    LwSciColor_Bayer16GBIG_GRIG,
    LwSciColor_Bayer16GIBG_GIRG,
    LwSciColor_Bayer16IGGB_IGGR,
    LwSciColor_Bayer16RGGI_BGGI,
    LwSciColor_Bayer16GRIG_GBIG,
    LwSciColor_Bayer16GIRG_GIBG,
    LwSciColor_Bayer16IGGR_IGGB,

    LwSciColor_UpperBound
} LwSciBufAttrValColorFmt;

/**
 * @brief Defines the image color standard for LwSciBufType_Image.
 *
 * @implements{18840084}
 */
typedef enum {
    LwSciColorStd_SRGB,
    LwSciColorStd_REC601_SR,
    LwSciColorStd_REC601_ER,
    LwSciColorStd_REC709_SR,
    LwSciColorStd_REC709_ER,
    LwSciColorStd_REC2020_RGB,
    LwSciColorStd_REC2020_SR,
    LwSciColorStd_REC2020_ER,
    LwSciColorStd_YcCbcCrc_SR,
    LwSciColorStd_YcCbcCrc_ER,
    LwSciColorStd_SENSOR_RGBA,
    LwSciColorStd_REQ2020PQ_ER,
} LwSciBufAttrValColorStd;

/**
 * @brief Surface types
 *
 * @implements{}
 */
typedef enum {
    /** YUV surface */
    LwSciSurfType_YUV,
    /**
     * RGBA surface
     *
     * Note: This is lwrrently not supported, and setting this attribute key
     * will fail.
     */
    LwSciSurfType_RGBA,
    /**
     * RAW surface
     *
     * Note: This is lwrrently not supported, and setting this attribute key
     * will fail.
     */
    LwSciSurfType_RAW,
    LwSciSurfType_MaxValid,
} LwSciBufSurfType;

/**
 * @brief Memory type
 *
 * @implements{}
 */
typedef enum {
    /**
     * Packed format
     *
     * Note: This is lwrrently not supported, and setting this attribute key
     * will fail.
     */
    LwSciSurfMemLayout_Packed,
    /**
     * Semi-planar format
     */
    LwSciSurfMemLayout_SemiPlanar,
    /**
     * Planar format
     */
    LwSciSurfMemLayout_Planar,
    LwSciSurfMemLayout_MaxValid,
} LwSciBufSurfMemLayout;

/**
 * @brief Subsampling type
 *
 * @implements{}
 */
typedef enum {
    /** 4:2:0 subsampling */
    LwSciSurfSampleType_420,
    /** 4:2:2 subsampling */
    LwSciSurfSampleType_422,
    /** 4:4:4 subsampling */
    LwSciSurfSampleType_444,
    /** 4:2:2 (transposed) subsampling */
    LwSciSurfSampleType_422R,
    LwSciSurfSampleType_MaxValid,
} LwSciBufSurfSampleType;

/**
 * @brief Bits Per Component
 *
 * @implements{}
 */
typedef enum {
    /** 16:8:8 bits per component layout */
    LwSciSurfBPC_Layout_16_8_8,
    /** 10:8:8 bits per component layout */
    LwSciSurfBPC_Layout_10_8_8,
    LwSciSurfBPC_MaxValid,
} LwSciBufSurfBPC;

/**
 * @brief Component ordering
 *
 * @implements{}
 */
typedef enum {
    /** YUV component order */
    LwSciSurfComponentOrder_YUV,
    /** YVU component order */
    LwSciSurfComponentOrder_YVU,
    LwSciSurfComponentOrder_MaxValid,
} LwSciBufSurfComponentOrder;

/**
 * @brief Defines various numeric datatypes for LwSciBuf.
 *
 * @implements{18840087}
 */
typedef enum {
    LwSciDataType_Int4,
    LwSciDataType_Uint4,
    LwSciDataType_Int8,
    LwSciDataType_Uint8,
    LwSciDataType_Int16,
    LwSciDataType_Uint16,
    LwSciDataType_Int32,
    LwSciDataType_Uint32,
    LwSciDataType_Float16,
    LwSciDataType_Float32,
    LwSciDataType_FloatISP,
    LwSciDataType_Bool,
    LwSciDataType_UpperBound
} LwSciBufAttrValDataType;

/**
 * @brief an enum spcifying various GPU compression values supported by LwSciBuf
 */
typedef enum {
    /**
     * Default value spcifying that GPU compression defaults to incompressible
     * kind. LwSciBuf fills this value in the reconciled LwSciBufAttrList if
     * the GPU compression is not granted for the particular GPU.
     * If compression is not needed, user does not have to explicitly
     * specify this value in the unreconciled LwSciBufAttrList. LwSciBuf does
     * not allow setting this value in the unreconciled LwSciBufAttrList.
     * Attempting to do so results in LwSciBufAttrListSetAttrs() returning an
     * error.
     */
    LwSciBufCompressionType_None,

    /**
     * Enum to request all possible GPU compression including enabling PLC (Post
     * L-2 Compression).
     * LWCA can read/write the GPU compressible memory with PLC enabled.
     * Vulkan can also read/write the GPU compressible memory with PLC
     * enabled.
     * This compression can be requested in LWCA to LWCA, LWCA to Vulkan and
     * Vulkan to Vulkan interop use-cases.
     */
    LwSciBufCompressionType_GenericCompressible,
} LwSciBufCompressionType;

/**
 * @brief Defines GPU ID structure. This structure is used to
 * set the value for LwSciBufGeneralAttrKey_GpuId attribute.
 *
 * @implements{18840093}
 */
typedef struct {
    /** GPU ID. This member is initialized by the successful
     * call to lwDeviceGetUuid() for LWCA usecases */
    uint8_t bytes[16];
} LwSciRmGpuId;

/**
 * Datatype specifying GPU cacheability preference for a particular GPU ID.
 *
 * TODO: Add implements tag for SWAD/SWUD
 * @implements{}
 */
typedef struct {
    /**
     * GPU ID for which cache preference need to be specified
     */
    LwSciRmGpuId gpuId;

    /**
     * boolean value specifying cacheability preference. TRUE implies caching
     * needs to be enabled, FALSE indicates otherwise.
     */
    bool cacheability;
} LwSciBufAttrValGpuCache;

/**
 * @brief Datatype specifying compression type needed for a particular GPU ID.
 */
typedef struct {
    /**
     * GPU ID for which compression needs to be specified
     */
    LwSciRmGpuId gpuId;

    /**
     * Type of compression
     */
    LwSciBufCompressionType compressionType;
} LwSciBufAttrValGpuCompression;


/**
 * @}
 */

/**
 * @defgroup lwscibuf_attr_datastructures LwSciBuf Data Structures
 * Specifies LwSciBuf data structures.
 * @{
 */

/**
 * @brief top-level container for the following set of
 *        resources: LwSciBufAttrLists, memory objects, and LwSciBufObjs.
 * Any @ref LwSciBufAttrList created or imported using a particular @ref LwSciBufModule
 * is bound to it, along with any @ref LwSciBufObj created or
 * imported using those LwSciBufAttrList(s).
 *
 * @note For any LwSciBuf API call that has more than one input of type
 * LwSciBufModule, LwSciBufAttrList, and/or LwSciBufObj, all such inputs
 * must agree on the LwSciBufModule instance.
 */
typedef struct LwSciBufModuleRec* LwSciBufModule;

/**
 * @brief This structure defines a key/value pair used to get or set
 * the LwSciBufAttrKey(s) and their corresponding values from or to
 * LwSciBufAttrList.
 *
 * @note An array of this structure need to be set in order to
 * allocate a buffer.
 *
 * @implements{18840090}
 */
typedef struct {
     /** LwSciBufAttrKey for which value needs to be set/retrieved. This member
      * is initialized to any one of the LwSciBufAttrKey other than
      * LwSciBufAttrKey_LowerBound and LwSciBufAttrKey_UpperBound */
      LwSciBufAttrKey key;

      /** Pointer to the value corresponding to the attribute.
       * If the value is an array, the pointer points to the first element. */
      const void* value;

      /** Length of the value in bytes */
      size_t len;
} LwSciBufAttrKeyValuePair;

/**
 * A memory object is a container holding the reconciled LwSciBufAttrList
 * defining constraints of the buffer, the handle of the allocated buffer
 * enforcing the buffer access permissions represented by
 * LwSciBufGeneralAttrKey_ActualPerm key in reconciled LwSciBufAttrList
 * and the buffer properties.
 */

/**
 * @brief A reference to a particular Memory object.
 *
 * @note Every @ref LwSciBufObj that has been created but not freed
 * holds a reference to the @ref LwSciBufModule, preventing it
 * from being de-initialized.
 */
typedef struct LwSciBufObjRefRec* LwSciBufObj;

/**
 * @brief A reference, that is not modifiable, to a particular Memory Object.
 */
typedef const struct LwSciBufObjRefRec* LwSciBufObjConst;


/**
 * @brief A container constituting an attribute list which contains
 * - set of LwSciBufAttrKey attributes defining buffer constraints
 * - slotcount defining number of slots in an attribute list
 * - flag specifying if attribute list is reconciled or unreconciled
 *
 * @note Every @ref LwSciBufAttrList that has been created but not freed
 * holds a reference to the @ref LwSciBufModule, preventing it
 * from being de-initialized.
 */
typedef struct LwSciBufAttrListRec* LwSciBufAttrList;

/**
 * @brief Defines the exported form of LwSciBufObj intended to be
 * shared across an LwSciIpc channel. On successful exelwtion of the
 * LwSciBufObjIpcExport(), the permission requested via this API is stored
 * in the LwSciBufObjIpcExportDescriptor to be granted to the LwSciBufObj
 * on import provided the permission requested via the API is not
 * LwSciBufAccessPerm_Auto. If the LwSciBufAccessPerm_Auto permission is
 * requested via the API then the permission stored in the
 * LwSciBufObjIpcExportDescriptor is equal to the maximum value of the
 * permissions requested via LwSciBufGeneralAttrKey_RequiredPerm attribute in
 * all of the unreconciled LwSciBufAttrLists that were exported by the peer to
 * which the LwSciBufObjIpcExportDescriptor is being exported.
 *
 * @implements{18840114}
 */
typedef struct {
      /** Exported data (blob) for LwSciBufObj */
      uint64_t data[LWSCIBUF_EXPORT_DESC_SIZE];
} __attribute__((packed)) LwSciBufObjIpcExportDescriptor;

/**
 * @}
 */

/**
 * @defgroup lwscibuf_attr_list_api LwSciBuf Attribute List APIs
 * Methods to perform operations on LwSciBuf attribute lists.
 * @{
 */

/**
 * @brief Creates a new, single slot, unreconciled LwSciBufAttrList associated
 * with the input LwSciBufModule with empty LwSciBufAttrKeys.
 *
 * @param[in] module LwSciBufModule to associate with the newly
 * created LwSciBufAttrList.
 * @param[out] newAttrList The new LwSciBufAttrList.
 *
 * @return ::LwSciError, the completion status of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a module is NULL
 *      - @a newAttrList is NULL
 * - ::LwSciError_InsufficientMemory if insufficient system memory to
 *   create a LwSciBufAttrList.
 * - ::LwSciError_IlwalidState if a new LwSciBufAttrList cannot be associated
 *   with the given LwSciBufModule.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a module is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListCreate(
    LwSciBufModule module,
    LwSciBufAttrList* newAttrList);

/**
 * @brief Frees the LwSciBufAttrList and removes its association with the
 * LwSciBufModule with which it was created.
 *
 * @note Every owner of the LwSciBufAttrList shall call LwSciBufAttrListFree()
 * only after all the functions ilwoked by the owner with LwSciBufAttrList as
 * an input are completed.
 *
 * @param[in] attrList The LwSciBufAttrList to be freed.
 *
 * @return void
 * - panics if LwSciBufAttrList is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the input
 *        LwSciBufAttrList @a attrList
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciBufAttrListFree(
    LwSciBufAttrList attrList);

/**
 * @brief Sets the values for LwSciBufAttrKey(s) in the LwSciBufAttrList.
 * It only reads values from LwSciBufAttrKeyValuePair array and
 * saves copies during this call.
 *
 * @note All combinations of LwSciBufAttrListSetAttrs(),
 * LwSciBufAttrListGetAttrs(), LwSciBufAttrListAppendUnreconciled()
 * and LwSciBufAttrListReconcile() can be called conlwrrently,
 * however, function completion order is not guaranteed by LwSciBuf
 * and thus outcome of calling these functions conlwrrently is
 * undefined.
 *
 * @param[in] attrList Unreconciled LwSciBufAttrList.
 * @param[in] pairArray Array of LwSciBufAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and
 * key member of every LwSciBufAttrKeyValuePair in the array is a valid enumeration
 * value defined by the LwSciBufAttrKey enum and value member of every
 * LwSciBufAttrKeyValuePair in the array is not NULL.
 * @param[in] pairCount Number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * @return ::LwSciError, the completion status of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a attrList is reconciled
 *      - @a attrList is an unreconciled LwSciBufAttrList obtained from
 *        LwSciBufAttrListAppendUnreconciled or
 *        LwSciBufAttrListIpcImportUnreconciled
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - any of the LwSciBufAttrKey(s) specified in the LwSciBufAttrKeyValuePair
 *        array is output only
 *      - any of the LwSciBufAttrKey(s) specified in the LwSciBufAttrKeyValuePair
 *        array has already been set
 *      - the LwSciBufGeneralAttrKey_Types key set (or lwrrently being set) on
 *        @a attrList does not contain the LwSciBufType of the datatype
 *        LwSciBufAttrKey(s)
 *      - any of the LwSciBufAttrKey(s) specified in the LwSciBufAttrKeyValuePair
 *        array oclwrs more than once
 *      - any of the LwSciBufAttrKey(s) specified in @a pairArray is not a
 *        valid enumeration value defined by the LwSciBufAttrKey enum
 *      - length(s) set for LwSciBufAttrKey(s) in @a pairArray are invalid
 *      - value(s) set for LwSciBufAttrKey(s) in @a pairArray are invalid
 * - ::LwSciError_InsufficientMemory if not enough system memory.
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListSetAttrs(
    LwSciBufAttrList attrList,
    LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * @brief Returns the slot count per LwSciBufAttrKey in a LwSciBufAttrList.
 *
 * @param[in] attrList The LwSciBufAttrList to retrieve the slot count from.
 *
 * @return size_t
 * - Number of slots in the LwSciBufAttrList
 * - panics if @a attrList is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
size_t LwSciBufAttrListGetSlotCount(
    LwSciBufAttrList attrList);

/**
 * @brief Returns an array of LwSciBufAttrKeyValuePair for a given set of LwSciBufAttrKey(s).
 * This function accepts a set of LwSciBufAttrKey(s) passed in the @ref LwSciBufAttrKeyValuePair
 * structure. The return values, stored back into @ref LwSciBufAttrKeyValuePair, consist of
 * @c const @c void* pointers to the attribute values from the @ref LwSciBufAttrList.
 * The application must not write to this data.
 *
 * @param[in] attrList LwSciBufAttrList to fetch the LwSciBufAttrKeyValuePair(s) from.
 * @param[in,out] pairArray Array of LwSciBufAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every LwSciBufAttrKeyValuePair in the array is a valid enumeration value
 * defined by the LwSciBufAttrKey enum.
 * @param[in] pairCount Number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - any of the LwSciBufAttrKey(s) in @a pairArray is not a valid
 *        enumeration value defined by the LwSciBufAttrKey enum
 *      - @a attrList is reconciled and any of the LwSciBufAttrKey(s) specified
 *        in LwSciBufAttrKeyValuePair is input only
 *      - @a attrList is unreconciled and any of the LwSciBufAttrKey(s)
 *        specified in LwSciBufAttrKeyValuePair is output only
 * - Panics if any of the following oclwrs:
 *      - @a attrList is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListGetAttrs(
    LwSciBufAttrList attrList,
    LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * @brief Returns an array of LwSciBufAttrKeyValuePair(s) from input
 * LwSciBufAttrList at the given slot index. The return values, stored in @ref
 * LwSciBufAttrKeyValuePair, consist of @c const @c void* pointers to the attribute values
 * from the LwSciBufAttrList. The application must not write to this data.
 *
 * @note When exporting an array containing multiple unreconciled LwSciBufAttrList(s),
 * the importing endpoint still imports just one unreconciled LwSciBufAttrList.
 * This unreconciled LwSciBufAttrList is referred to as a multi-slot
 * LwSciBufAttrList. It logically represents an array of LwSciBufAttrList(s), where
 * each key has an array of values, one per slot.
 *
 * @param[in] attrList LwSciBufAttrList to fetch the LwSciBufAttrKeyValuePair(s) from.
 * @param[in] slotIndex Index in the LwSciBufAttrList.
 * Valid value: 0 to slot count of LwSciBufAttrList - 1.
 * @param[in,out] pairArray Array of LwSciBufAttrKeyValuePair. Holds the LwSciBufAttrKey(s)
 * passed into the function and returns an array of LwSciBufAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every LwSciBufAttrKeyValuePair is a valid enumeration value defined by the
 * LwSciBufAttrKey enum
 * @param[in] pairCount Number of elements/entries in pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a pairArray is NULL
 *      - any of the LwSciBufAttrKey(s) in @a pairArray is not a valid
 *        enumeration value defined by the LwSciBufAttrKey enum
 *      - @a pairCount is 0
 *      - @a slotIndex >= slot count of LwSciBufAttrList
 *      - LwSciBufAttrKey specified in @a pairArray is invalid.
 *      - @a attrList is reconciled and any of the LwSciBufAttrKey(s) specified
 *        in LwSciBufAttrKeyValuePair is input only
 *      - @a attrList is unreconciled and any of the LwSciBufAttrKey(s)
 *        specified in LwSciBufAttrKeyValuePair is output only
 * - Panics if any of the following oclwrs:
 *      - @a attrList is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListSlotGetAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    LwSciBufAttrKeyValuePair* pairArray,
    size_t pairCount);

#if (LW_IS_SAFETY == 0)
/**
 * @brief Allocates a buffer and then dumps the contents of the specified
 * attribute list into the buffer.
 *
 * @param[in] attrList Attribute list to fetch contents from.
 * @param[out] buf A pointer to the buffer allocated for the debug dump.
 * @param[out] len The length of the buffer allocated for the debug dump.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a attrList is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListDebugDump(
    LwSciBufAttrList attrList,
    void** buf,
    size_t* len);
#endif

/**
 * @brief Reconciles the given unreconciled LwSciBufAttrList(s) into a new
 * reconciled LwSciBufAttrList.
 * On success, this API call returns reconciled LwSciBufAttrList, which has to
 * be freed by the caller using LwSciBufAttrListFree().
 *
 * @param[in] inputArray Array containing unreconciled LwSciBufAttrList(s) to be
 *            reconciled. @a inputArray is valid if it is non-NULL.
 * @param[in] inputCount The number of unreconciled LwSciBufAttrList(s) in
 *            @a inputArray. This value must be non-zero. For a single
 *            LwSciBufAttrList, the count must be set 1.
 * @param[out] newReconciledAttrList Reconciled LwSciBufAttrList. This field
 *             is populated only if the reconciliation succeeded.
 */
#if (LW_IS_SAFETY == 0)
/**
 * @param[out] newConflictList Unreconciled LwSciBufAttrList consisting of the
 * key/value pairs which caused the reconciliation failure. This field is
 * populated only if the reconciliation failed.
 */
#else
/**
 * @param[out] newConflictList unused.
 */
#endif
/**
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a inputArray[] is NULL
 *      - @a inputCount is 0
 *      - @a newReconciledAttrList is NULL
 *      - any of the LwSciBufAttrList in @a inputArray is reconciled.
 *      - not all the LwSciBufAttrLists in @a inputArray are bound to the
 *        same LwSciBufModule.
 *      - an attribute key necessary for reconciling against the given data
 *        type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation is
 *        unset
 *      - an attribute key is set to an unsupported value considering the data
 *        type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation
 */
#if (LW_IS_SAFETY == 0)
/**      - @a newConflictList is NULL
 */
#endif
/**
 * - ::LwSciError_InsufficientMemory if not enough system memory.
 * - ::LwSciError_IlwalidState if a new LwSciBufAttrList cannot be associated
 *   with the LwSciBufModule associated with the LwSciBufAttrList(s) in the
 *   given @a inputArray to create a new reconciled LwSciBufAttrList
 * - ::LwSciError_NotSupported if an attribute key is set resulting in a
 *   combination of given constraints that are not supported
 * - ::LwSciError_Overflow if internal integer overflow is detected.
 * - ::LwSciError_ReconciliationFailed if reconciliation failed.
 * - ::LwSciError_ResourceError if,
 *      - System lacks resource other than memory.
 *      - LWPU driver stack failed during this operation.
 * - Panic if:
 *      - @a unreconciled LwSciBufAttrList(s) in inputArray is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListReconcile(
    const LwSciBufAttrList inputArray[],
    size_t inputCount,
    LwSciBufAttrList* newReconciledAttrList,
    LwSciBufAttrList* newConflictList);

/**
 * @brief Clones an unreconciled/reconciled LwSciBufAttrList. The resulting
 * LwSciBufAttrList contains all the values of the input LwSciBufAttrList.
 * If the input LwSciBufAttrList is an unreconciled LwSciBufAttrList, then
 * modification to the output LwSciBufAttrList will be allowed using
 * LwSciBufAttrListSetAttrs().
 *
 * @param[in] origAttrList LwSciBufAttrList to be cloned.
 *
 * @param[out] newAttrList The new LwSciBufAttrList.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a origAttrList is NULL
 *      - @a newAttrList is NULL
 *      - the LwSciBufGeneralAttrKey_Types key is not set on @a origAttrList
 * - ::LwSciError_InsufficientMemory if there is insufficient system memory
 *   to create a new LwSciBufAttrList.
 * - ::LwSciError_IlwalidState if a new LwSciBufAttrList cannot be associated
 *   with the LwSciBufModule of @a origAttrList to create the new
 *   LwSciBufAttrList.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a origAttrList is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListClone(
    LwSciBufAttrList origAttrList,
    LwSciBufAttrList* newAttrList);

/**
 * @brief Appends multiple unreconciled LwSciBufAttrList(s) together, forming a
 *  single new unreconciled LwSciBufAttrList with a slot count equal to the
 *  sum of all the slot counts of LwSciBufAttrList(s) in the input array and
 *  containing the contents of all the LwSciBufAttrList(s) in the input array.
 *
 * @param[in] inputUnreconciledAttrListArray[] Array containing the
 *  unreconciled LwSciBufAttrList(s) to be appended together.
 *  Valid value: Array of valid LwSciBufAttrList(s) where the array
 *  size is at least 1.
 * @param[in] inputUnreconciledAttrListCount Number of unreconciled
 * LwSciBufAttrList(s) in @a inputUnreconciledAttrListArray.
 * Valid value: inputUnreconciledAttrListCount is valid input if it
 * is non-zero.
 *
 * @param[out] newUnreconciledAttrList Appended LwSciBufAttrList.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a inputUnreconciledAttrListArray is NULL
 *      - @a inputUnreconciledAttrListCount is 0
 *      - @a newUnreconciledAttrList is NULL
 *      - any of the LwSciBufAttrList(s) in @a inputUnreconciledAttrListArray
 *        is reconciled
 *      - not all the LwSciBufAttrLists in @a inputUnreconciledAttrListArray
 *        are bound to the same LwSciBufModule instance.
 *      - the LwSciBufGeneralAttrKey_Types key is not set on any of the
 *        LwSciBufAttrList(s) in @a inputUnreconciledAttrListArray
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if a new LwSciBufAttrList cannot be associated
 *   with the LwSciBufModule associated with the LwSciBufAttrList(s) in the
 *   given @a inputUnreconciledAttrListArray to create the new LwSciBufAttrList.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a any LwSciBufAttrList in the @a
 *   inputUnreconciledAttrListArray is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListAppendUnreconciled(
    const LwSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciBufAttrList* newUnreconciledAttrList);

/**
 * @brief Checks if the LwSciBufAttrList is reconciled.
 *
 * @param[in] attrList LwSciBufAttrList to check.
 * @param[out] isReconciled boolean value indicating whether the
 * @a attrList is reconciled or not.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a isReconciled is NULL
 * - panics if @a attrList is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListIsReconciled(
    LwSciBufAttrList attrList,
    bool* isReconciled);

/**
 * @brief Validates a reconciled LwSciBufAttrList against a set of
 *        unreconciled LwSciBufAttrList(s).
 *
 * @param[in] reconciledAttrList Reconciled LwSciBufAttrList list to be
 *            validated.
 * @param[in] unreconciledAttrListArray Set of unreconciled LwSciBufAttrList(s)
 *            that need to be used for validation. @a unreconciledAttrListArray
 *            is valid if it is non-NULL.
 * @param[in] unreconciledAttrListCount Number of unreconciled
 *            LwSciBufAttrList(s). This value must be non-zero.
 *            For a single LwSciBufAttrList, the count must be set to 1.
 * @param[out] isReconcileListValid Flag indicating if the reconciled
 *             LwSciBufAttrList satisfies the constraints of set of
 *             unreconciled LwSciBufAttrList(s).
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a reconciledAttrList is NULL or
 *         - @a unreconciledAttrListArray[] is NULL or
 *         - @a unreconciledAttrListCount is zero or
 *         - @a isReconcileListValid is NULL
 *         - any of the LwSciBufAttrList in @a unreconciledAttrListArray is
 *           reconciled.
 *         - not all the LwSciBufAttrLists in @a unreconciledAttrListArray are
 *           bound to the same LwSciBufModule.
 * - ::LwSciError_ReconciliationFailed if validation of reconciled
 *   LwSciBufAttrList failed against input unreconciled LwSciBufAttrList(s).
 * - ::LwSciError_InsufficientMemory if internal memory allocation failed.
 * - ::LwSciError_Overflow if internal integer overflow oclwrs.
 * - Panics if:
 *         - @a unreconciled LwSciBufAttrList(s) in unreconciledAttrListArray
 *           is invalid.
 *         - @a reconciledAttrList is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListValidateReconciled(
    LwSciBufAttrList reconciledAttrList,
    const LwSciBufAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    bool* isReconcileListValid);

/**
 * @}
 */

/**
 * @defgroup lwscibuf_obj_api LwSciBuf Object APIs
 * List of APIs to create/operate on LwSciBufObj.
 * @{
 */

/**
 * @brief Creates a new LwSciBufObj holding reference to the same
 * Memory object to which input LwSciBufObj holds the reference.
 *
 * @note The new LwSciBufObj created with LwSciBufObjDup() has same
 * LwSciBufAttrValAccessPerm as the input LwSciBufObj.
 *
 * @param[in] bufObj LwSciBufObj from which new LwSciBufObj needs
 * to be created.
 * @param[out] dupObj The new LwSciBufObj.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL
 *      - @a dupObj is NULL
 * - ::LwSciError_InsufficientMemory if memory allocation is failed.
 * - ::LwSciError_IlwalidState if the total number of LwSciBufObjs referencing
 *   the memory object is INT32_MAX and the caller tries to take one more
 *   reference using this API.
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 * - Panics if @a bufObj is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjDup(
    LwSciBufObj bufObj,
    LwSciBufObj* dupObj);

/**
 * @brief Reconciles the input unreconciled LwSciBufAttrList(s) into a new
 * reconciled LwSciBufAttrList and allocates LwSciBufObj that meets all the
 * constraints in the reconciled LwSciBufAttrList.
 *
 * @note This interface just combines LwSciBufAttrListReconcile() and
 * LwSciBufObjAlloc() interfaces together.
 *
 * @param[in] attrListArray Array containing unreconciled LwSciBufAttrList(s) to
 * reconcile. Valid value: Array of valid unreconciled LwSciBufAttrList(s) where
 * array size is at least 1.
 * @param[in] attrListCount The number of unreconciled LwSciBufAttrList(s) in
 * @c attrListArray. Valid value: 1 to SIZE_MAX.
 *
 * @param[out] bufObj The new LwSciBufObj.
 */
#if (LW_IS_SAFETY == 0)
/**
 * @param[out] newConflictList Unreconciled LwSciBufAttrList consisting of the
 * key/value pairs which caused the reconciliation failure. This field is
 * populated only if the reconciliation failed.
 */
#else
/**
 * @param[out] newConflictList unused.
 */
#endif
/**
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrListCount is 0
 *      - @a attrListArray is NULL
 *      - @a bufObj is NULL
 *      - any of the LwSciBufAttrList in @a attrListArray is reconciled.
 *      - not all the LwSciBufAttrLists in @a attrListArray are bound to
 *        the same LwSciBufModule.
 *      - an attribute key necessary for reconciling against the given data
 *      type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation is
 *      unset
 *      - an attribute key is set to an unsupported value considering the data
 *      type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation
 */
#if (LW_IS_SAFETY == 0)
/**
 *      - @a newConflictList is NULL
 */
#endif
/**
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if a new LwSciBufAttrList cannot be associated
 *   with the LwSciBufModule associated with the LwSciBufAttrList(s) in the
 *   given @a attrListArray to create the new LwSciBufAttrList.
 * - ::LwSciError_NotSupported if an attribute key is set specifying a
 *   combination of constraints that are not supported
 * - ::LwSciError_Overflow if internal integer overflow is detected.
 * - ::LwSciError_ReconciliationFailed if reconciliation failed.
 * - ::LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed during buffer allocation
 *      - system lacks resource other than memory
 * - Panics if any of the unreconciled LwSciBufAttrLists is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListReconcileAndObjAlloc(
    const LwSciBufAttrList attrListArray[],
    size_t attrListCount,
    LwSciBufObj* bufObj,
    LwSciBufAttrList* newConflictList);

/**
 * @brief Removes reference to the Memory object by destroying the LwSciBufObj.
 *
 * @note Every owner of the LwSciBufObj shall call LwSciBufObjFree()
 * only after all the functions ilwoked by the owner with LwSciBufObj
 * as an input are completed.
 *
 * \param[in] bufObj The LwSciBufObj to deallocate.
 *
 * @return void
 * - Panics if @a bufObj is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the LwSciBufAttrList
 *        obtained from LwSciBufObjGetAttrList() to be freed, since the
 *        lifetime of that reconciled LwSciBufAttrList is tied to the
 *        associated LwSciBufObj
 *      - Provided there is no active operation ilwolving the LwSciBufObj to be
 *        freed
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciBufObjFree(
    LwSciBufObj bufObj);

/**
 * @brief Retrieves the reconciled LwSciBufAttrList whose attributes define
 * the constraints of the allocated buffer from the LwSciBufObj.
 *
 * @note The retrieved LwSciBufAttrList from an LwSciBufObj is read-only,
 * and the attribute values in the list cannot be modified using
 * set attribute APIs. In addition, the retrieved LwSciBufAttrList must
 * not be freed with LwSciBufAttrListFree.
 *
 * @param[in] bufObj The LwSciBufObj to retrieve the LwSciBufAttrList from.
 * @param[out] bufAttrList The retrieved reconciled LwSciBufAttrList.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL.
 *      - @a bufAttrList is NULL.
 * - Panics if @a bufObj is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjGetAttrList(
    LwSciBufObj bufObj,
    LwSciBufAttrList* bufAttrList);

/**
 * @brief Gets the CPU virtual address (VA) of the read/write buffer
 * referenced by the LwSciBufObj.
 *
 * @note This interface can be called successfully only if LwSciBufObj
 * was obtained from successful call to LwSciBufObjAlloc() or
 * LwSciBufObj was obtained from successful call to LwSciBufObjIpcImport()/
 * LwSciBufIpcImportAttrListAndObj() where LwSciBufAccessPerm_ReadWrite
 * permissions are granted to the imported LwSciBufObj (The permissions
 * of the LwSciBufObj are indicated by LwSciBufGeneralAttrKey_ActualPerm
 * key in the reconciled LwSciBufAttrList associated with it) and CPU
 * access is requested by setting LwSciBufGeneralAttrKey_NeedCpuAccess
 * to true.
 *
 * @param[in] bufObj The LwSciBufObj.
 *
 * @param[out] ptr The CPU virtual address (VA).
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL.
 *      - @a ptr is NULL.
 * - ::LwSciError_BadParameter LwSciBufObj either did not request
 *   for CPU access by setting LwSciBufGeneralAttrKey_NeedCpuAccess
 *   to true OR does not have LwSciBufAccessPerm_ReadWrite to the
 *   buffer.
 * - Panics if @a bufObj is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjGetCpuPtr(
    LwSciBufObj bufObj,
    void**  ptr);

/**
 * @brief Gets the CPU virtual address (VA) of the read-only buffer
 * referenced by the LwSciBufObj.
 *
 * @note This interface can be called successfully only if LwSciBufObj
 * was obtained from successful call to LwSciBufObjAlloc() or
 * LwSciBufObj was obtained from successful call to LwSciBufObjIpcImport()/
 * LwSciBufIpcImportAttrListAndObj() where at least LwSciBufAccessPerm_Readonly
 * permissions are granted to the imported LwSciBufObj (The permissions of the
 * LwSciBufObj are indicated by LwSciBufGeneralAttrKey_ActualPerm key in the
 * reconciled LwSciBufAttrList associated with it) and CPU access is
 * requested by setting LwSciBufGeneralAttrKey_NeedCpuAccess to true.
 *
 * @param[in] bufObj The LwSciBufObj.
 *
 * @param[out] ptr the CPU virtual address (VA).
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL.
 *      - @a ptr is NULL.
 * - ::LwSciError_BadParameter LwSciBufObj either did not request
 *   for CPU access by setting LwSciBufGeneralAttrKey_NeedCpuAccess
 *   to true OR does not have at least LwSciBufAccessPerm_ReadOnly
 *   permissions to the buffer.
 * - Panics if @a bufObj is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjGetConstCpuPtr(
    LwSciBufObj bufObj,
    const void**  ptr);

/**
 * @brief Flushes the given @c len bytes at starting @c offset in the
 * buffer referenced by the LwSciBufObj. Flushing is done only when
 * LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency key is set in
 * reconciled LwSciBufAttrList to true.
 *
 * @param[in] bufObj The LwSciBufObj.
 * @param[in] offset The starting offset in memory of the LwSciBufObj.
 * Valid value: 0 to buffer size - 1.
 * @param[in] len The length (in bytes) to flush.
 * Valid value: 1 to buffer size - offset.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL
 *      - @a len is zero
 *      - @a offset + @a len > buffer size.
 * - ::LwSciError_NotPermitted if buffer referenced by @a bufObj is
 *   not mapped to CPU.
 * - ::LwSciError_NotSupported if LwSciBufAllocIfaceType associated with the
 *   LwSciBufObj is not supported.
 * - ::LwSciError_Overflow if @a offset + @a len exceeds UINT64_MAX
 * - ::LwSciError_ResourceError if LWPU driver stack could not flush the
 *   CPU cache range.
 * - Panics if @a bufObj is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciBufObjFlushCpuCacheRange(
    LwSciBufObj bufObj,
    uint64_t offset,
    uint64_t len);

/**
 * @brief Allocates a buffer that satisfies all the constraints defined by
 * the attributes of the specified reconciled LwSciBufAttrList, and outputs
 * a new LwSciBufObj referencing the Memory object containing the allocated
 * buffer properties.
 *
 * @note It is not guaranteed that the input reconciled LwSciBufAttrList in
 * this API is the same LwSciBufAttrList that is ultimately associated with the
 * allocated LwSciBufObj. If the user needs to query attributes from an
 * LwSciBufAttrList associated with an LwSciBufObj after allocation, they must
 * first obtain the reconciled LwSciBufAttrList from the LwSciBufObj using
 * LwSciBufObjGetAttrList().
 *
 * @param[in] reconciledAttrList The reconciled LwSciBufAttrList.
 *
 * @param[out] bufObj The new LwSciBufObj.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a reconciledAttrList is NULL
 *      - @a reconciledAttrList is not a reconciled LwSciBufAttrList
 *      - @a bufObj is NULL
 * - ::LwSciError_InsufficientMemory if there is insufficient memory
 *   to complete the operation.
 * - ::LwSciError_IlwalidState if a new LwSciBufObj cannot be associated
 *   with the LwSciBufModule with which @a reconciledAttrList is associated to
 *   create the new LwSciBufObj.
 * - ::LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed during buffer allocation
 *      - system lacks resource other than memory
 * - Panics if @a reconciledAttrList is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjAlloc(
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObj* bufObj);

/**
 * @brief Creates a new memory object containing a buffer handle representing
 * the new LwSciBufAttrValAccessPerm to the same buffer for the buffer
 * handle contained in the input memory object referenced by the input
 * LwSciBufObj and creates a new LwSciBufObj referencing it provided
 * LwSciBufAttrValAccessPerm are less than permissions represented by buffer
 * handle in the memory object referenced by input LwSciBufObj. When this is the
 * case, the new memory object will contains a new LwSciBufAttrList which is
 * cloned from the original LwSciBufAttrList associated with the input
 * LwSciBufObj, but with the requested LwSciBufAttrValAccessPerm.
 *
 * This interface has same effect as calling LwSciBufObjDup() if
 * LwSciBufAttrValAccessPerm are the same as the permissions represented by
 * the buffer handle in the memory object referenced by the input LwSciBufObj.
 *
 * @param[in] bufObj LwSciBufObj.
 * @param[in] reducedPerm Reduced access permissions that need to be imposed on
 * the new LwSciBufObj (see @ref LwSciBufAttrValAccessPerm).
 * Valid value: LwSciBufAccessPerm_Readonly or LwSciBufAccessPerm_ReadWrite,
 * which is <= LwSciBufAttrValAccessPerm represented by the value of the
 * LwSciBufGeneralAttrKey_ActualPerm key in the reconciled LwSciBufAttrList
 * associated with the input LwSciBufObj.
 * \param[out] newBufObj The new LwSciBufObj with new permissions.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL
 *      - @a newBufObj is NULL
 *      - @a reducedPerm is not LwSciBufAccessPerm_Readonly or
 *        LwSciBufAccessPerm_ReadWrite
 *      - @a reducedPerm is greater than the permissions specified in the value
 *        of the LwSciBufGeneralAttrKey_ActualPerm key
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if any of the following oclwrs:
 *      - the total number of LwSciBufObj(s) referencing the memory object is
 *        INT32_MAX and the caller tries to take one more reference using this
 *        API.
 *      - a new LwSciBufObj cannot be associated with the LwSciBufModule with
 *        which @a bufObj is associated to create the new LwSciBufAttrList
 *        when the requested access permissions are less than the permissions
 *        represented by the input LwSciBufObj
 * - ::LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed while assigning new permission to the buffer handle
 *      - system lacks resource other than memory
 */
#if (LW_IS_SAFETY == 0)
/**
 *  - ::LwSciError_NotSupported if this API is called for LwSciBufObj imported
 *      from the remote Soc.
 */
#endif
/**
 * - Panics of @a bufObj is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjDupWithReducePerm(
    LwSciBufObj bufObj,
    LwSciBufAttrValAccessPerm reducedPerm,
    LwSciBufObj* newBufObj);

/**
 * @}
 */

/**
 * @defgroup lwscibuf_transport_api LwSciBuf APIs
 * List of APIs to transport LwSciBuf buffers and attribute list objects across
 * various communication boundaries that interact using LwSciIpc.
 * @{
 */


/**
 * @brief Exports LwSciBufAttrList and LwSciBufObj into an
 * LwSciIpc-transferable object export descriptor. The blob can be
 * transferred to the other processes to create a matching LwSciBufObj.
 *
 * @param[in] bufObj LwSciBufObj to export.
 * @param[in] permissions Flag indicating the expected access permission
 *            (see @ref LwSciBufAttrValAccessPerm). The valid value is either
 *            of LwSciBufAccessPerm_Readonly or LwSciBufAccessPerm_ReadWrite
 *            such that the value of LwSciBufGeneralAttrKey_ActualPerm set in
 *            the reconciled LwSciBufAttrList exported to the peer to which
 *            LwSciBufObj is being exported is less than or equal to
 *            @a permissions and @a permissions is less than or equal to
 *            underlying LwSciBufObj permission. Additionally,
 *            LwSciBufAccessPerm_Auto value is unconditionally valid.
 * @param[in] ipcEndpoint LwSciIpcEndpoint to identify the peer process.
 * \param[out] attrListAndObjDesc LwSciBuf allocates and fills in the
 *             exportable form of LwSciBufObj and its corresponding
 *             LwSciBufAttrList to be shared across an LwSciIpc channel.
 * \param[out] attrListAndObjDescSize Size of the exported blob.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrListAndObjDesc is NULL
 *      - @a attrListAndObjDescSize is NULL
 *      - @a bufObj is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a permissions takes value other than LwSciBufAccessPerm_Readonly,
 *        LwSciBufAccessPerm_ReadWrite or LwSciBufAccessPerm_Auto.
 * - ::LwSciError_InsufficientMemory if memory allocation failed
 * - ::LwSciError_IlwalidOperation if reconciled LwSciBufAttrList of @a bufObj
 *   has greater permissions for the @a ipcEndpoint peer than the
 *   @a permissions
 * - ::LwSciError_Overflow if an arithmetic overflow oclwrs due to an invalid
 *     export descriptor
 * - ::LwSciError_NotPermitted if LwSciBufObj and LwSciBufAttrList associated
 *   with it are not being exported in the reverse direction of IPC path in
 *   which unreconciled LwSciBufAttrLists ilwolved in reconciliation of
 *   LwSciBufAttrList associated with the input LwScibufObj were exported.
 * - ::LwSciError_ResourceError if the LWPU driver stack failed.
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panic if @a bufObj is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufIpcExportAttrListAndObj(
    LwSciBufObj bufObj,
    LwSciBufAttrValAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** attrListAndObjDesc,
    size_t* attrListAndObjDescSize);

/**
 * @brief This API is ilwoked by the importing process after it receives the
 * object export descriptor sent by the other process who has created
 * descriptor.
 * The importing process will create its own LwSciBufObj and return as
 * output.
 *
 * @param[in] module LwSciBufModule to be used for importing LwSciBufObj.
 * @param[in] ipcEndpoint LwSciIpcEndpoint to identify the peer process.
 * @param[in] attrListAndObjDesc The exported form of LwSciBufAttrList and
 *            LwSciBufObj. The valid value must be non NULL.
 * @param[in] attrListAndObjDescSize Size of the imported blob. This value must
 *            be non-zero.
 * @param[in] attrList[] Receiver side array of LwSciBufAttrList(s) against
 *            which the imported LwSciBufAttrList has to be validated. NULL is
 *            valid value here if the validation of the received
 *            LwSciBufAttrList needs to be skipped.
 * @param[in] count Number of LwSciBufAttrList objects in the array. This value
 *            must be non-zero, provided @a attrList is non-NULL.
 * @param[in] minPermissions Minimum permissions of the LwSciBufObj that the
 *            process is expecting to import it with (see @ref
 *            LwSciBufAttrValAccessPerm). The valid value is either of
 *            LwSciBufAccessPerm_Readonly or LwSciBufAccessPerm_ReadWrite such
 *            that the value is less than or equal to LwSciBufAttrValAccessPerm
 *            with which LwSciBufObj was exported. Additionally,
 *            LwSciBufAccessPerm_Auto value is unconditionally valid.
 * @param[in] timeoutUs Maximum delay (in microseconds) before an LwSciBufObj
 *            times out. The value of the variable is ignored lwrrently.
 * \param[out] bufObj LwSciBufObj duplicated and exported during the
 *             importing process. This LwSciBufObj is associated with the
 *             reconciled LwSciBufAttrList imported from the attrListAndObjDesc.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a attrListAndObjDesc is NULL
 *      - @a attrListAndObjDesc represents an LwSciBufAttrList with invalid
 *        attribute key values set
 *      - @a attrListAndObjDesc represents an LwSciBufAttrList which is
 *        unreconciled.
 *      - @a attrListAndObjDesc is invalid
 *      - @a attrListAndObjDescSize is 0
 *      - @a count is 0, provided @a attrList is non-NULL
 *      - @a minPermissions are invalid.
 *      - @a bufObj is NULL
 * - ::LwSciError_NotSupported if any of the following oclwrs:
 *      - @a attrListAndObjDesc is incompatible
 *      - Internal attribute of the imported LwSciBufAttrList represents
 *        memory domain which is not supported.
 * - ::LwSciError_AccessDenied if @a minPermissions are greater than permissions
 *     with which LwSciBufObj was exported
 * - ::LwSciError_AttrListValidationFailed if input unreconciled
 *    LwSciBufAttrList(s)' contraints are not satisfied by attributes
 *    associated with the imported LwSciBufObj
 * - ::LwSciError_InsufficientMemory if memory allocation failed
 * - ::LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed
 *      - system lacks resource other than memory
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - ::LwSciError_IlwalidState if any of the following oclwrs:
 *      - Imported LwSciBufAttrList cannot be associated with @a module.
 *      - Imported LwSciBufObj cannot be associated with @a module.
 * - Panic if:
 *    - @a any of the unreconciled LwSciBufAttrList(s) are not valid
 *    - @a module is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufIpcImportAttrListAndObj(
    LwSciBufModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* attrListAndObjDesc,
    size_t attrListAndObjDescSize,
    const LwSciBufAttrList attrList[],
    size_t count,
    LwSciBufAttrValAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciBufObj* bufObj);

/**
 * @brief Frees the descriptor used for exporting both LwSciBufAttrList and
 * LwSciBufObj together.
 *
 * @param[in] attrListAndObjDescBuf Descriptor to be freed. The valid value is
 *            the one returned by successful call to
 *            LwSciBufIpcExportAttrListAndObj().
 *
 * @return void
 * - Panics if:
 *     - @a attrListAndObjDescBuf is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the
 *        @a attrListAndObjDescBuf to be freed
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciBufAttrListAndObjFreeDesc(
    void* attrListAndObjDescBuf);

/**
 * @brief Exports the LwSciBufObj into an LwSciIpc-transferable object
 * export descriptor.
 * Descriptor can be transferred to other end of IPC where matching
 * LwSciBufObj can be created from the descriptor.
 *
 * @param[in] bufObj LwSciBufObj to export.
 * @param[in] accPerm Flag indicating the expected access permission
 *            (see @ref LwSciBufAttrValAccessPerm). The valid value is either
 *            of LwSciBufAccessPerm_Readonly or LwSciBufAccessPerm_ReadWrite
 *            such that the value of LwSciBufGeneralAttrKey_ActualPerm set in
 *            the reconciled LwSciBufAttrList exported to the peer to which
 *            LwSciBufObj is being exported is less than or equal to @a accPerm
 *            and @a accPerm is less than or equal to underlying LwSciBufObj
 *            permission. Additionally, LwSciBufAccessPerm_Auto value is
 *            unconditionally valid.
 * @param[in] ipcEndpoint LwSciIpcEndpoint.
 * \param[out] exportData LwSciBuf populates the return value with exportable
 *             form of LwSciBufObj shared across an LwSciIpc channel.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL
 *      - @a accPerm takes value other than LwSciBufAccessPerm_Readonly,
 *        LwSciBufAccessPerm_ReadWrite or LwSciBufAccessPerm_Auto.
 *      - @a ipcEndpoint is invalid
 *      - @a exportData is NULL
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidOperation if reconciled LwSciBufAttrList of @a bufObj
 *   has greater permissions for the @a ipcEndpoint peer than the
 *   @a accPerm
 * - ::LwSciError_NotPermitted if LwSciBufObj is not being exported in the
 *   reverse direction of IPC path in which unreconciled LwSciBufAttrLists
 *   ilwolved in reconciliation of LwSciBufAttrList associated with the input
 *   LwScibufObj were exported.
 * - ::LwSciError_ResourceError if the LWPU driver stack failed.
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panic if @a bufObj is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjIpcExport(
    LwSciBufObj bufObj,
    LwSciBufAttrValAccessPerm accPerm,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufObjIpcExportDescriptor* exportData);

/**
 * @brief Creates the LwSciBufObj based on supplied object export descriptor
 * and returns the LwSciBufObj bound to the reconciled LwSciBufAttrList.
 *
 * @note It is not guaranteed that the input reconciled LwSciBufAttrList in
 * this API is the same LwSciBufAttrList that is ultimately associated with the
 * allocated LwSciBufObj. If the user needs to query attributes from an
 * LwSciBufAttrList associated with an LwSciBufObj after allocation, they must
 * first obtain the reconciled LwSciBufAttrList from the LwSciBufObj using
 * LwSciBufObjGetAttrList().
 *
 * @param[in] ipcEndpoint LwSciIpcEndpoint.
 * @param[in] desc A pointer to an LwSciBufObjIpcExportDescriptor. The valid
 *            value is non-NULL that points to descriptor received on LwSciIpc
 *            channel.
 * @param[in] reconciledAttrList Reconciled LwSciBufAttrList returned by
 *            LwSciBufAttrListIpcImportReconciled().
 * @param[in] minPermissions Minimum permissions of the LwSciBufObj that the
 *            process is expecting to import it with (see @ref
 *            LwSciBufAttrValAccessPerm). The valid value is either of
 *            LwSciBufAccessPerm_Readonly or LwSciBufAccessPerm_ReadWrite such
 *            that the value is less than or equal to LwSciBufAttrValAccessPerm
 *            with which LwSciBufObj was exported. Additionally,
 *            LwSciBufAccessPerm_Auto value is unconditionally valid.
 * @param[in] timeoutUs Maximum delay (in microseconds) before an LwSciBufObj
              times out. The value of the variable is ignored lwrrently.
 * @param[out] bufObj Imported LwSciBufObj created from the descriptor.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_AccessDenied if minPermissions are greater than permissions
 *     with which object was exported
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a ipcEndpoint is invalid
 *      - @a desc is NULL or invalid
 *      - @a reconciledAttrList is NULL
 *      - @a reconciledAttrList is unreconciled.
 *      - @a minPermissions are invalid.
 *      - @a bufObj is NULL
 * - ::LwSciError_InsufficientMemory if there is insufficient system memory.
 * - ::LwSciError_Overflow if an arithmetic overflow oclwrs due to an invalid
 *     export descriptor
 * - ::LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed
 *      - system lacks resource other than memory
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 */
#if (BACKEND_RESMAN)
/**
 * - ::LwSciError_IlwalidOperation if the LwSciBufObj has already been freed in
 *     the exporting peer
 */
#endif
/**
 * - ::LwSciError_IlwalidState if imported LwSciBufObj cannot be associated with
 *     LwSciBufModule with which @a reconciledAttrList is associated.
 * - ::LwSciError_NotSupported if the internal attribute of
 *     @a reconciledAttrList represents memory domain which is not supported.
 * - Panic if @a reconciledAttrList is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufObjIpcImport(
    LwSciIpcEndpoint ipcEndpoint,
    const LwSciBufObjIpcExportDescriptor* desc,
    LwSciBufAttrList reconciledAttrList,
    LwSciBufAttrValAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciBufObj* bufObj);

/**
 * @brief Transforms the input unreconciled LwSciBufAttrList(s) to an exportable
 * unreconciled LwSciBufAttrList descriptor that can be transported by the
 * application to any remote process as a serialized set of bytes over an
 * LwSciIpc channel.
 *
 * @param[in] unreconciledAttrListArray The unreconciled LwSciBufAttrList(s) to
 *            be exported. The valid value is non NULL.
 * @param[in] unreconciledAttrListCount Number of unreconciled
 *            LwSciBufAttrList(s) in @a unreconciledAttrListArray. This value
 *            must be non-zero. For a single list, the count must be set 1.
 * @param[in] ipcEndpoint The LwSciIpcEndpoint.
 * @param[out] descBuf A pointer to the new unreconciled LwSciBufAttrList
 *             descriptor, which the caller can deallocate later using
 *             LwSciBufAttrListFreeDesc().
 * @param[out] descLen The size of the new unreconciled LwSciBufAttrList
 *             descriptor.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a unreconciledAttrListArray is NULL
 *      - any of the LwSciBufAttrLists in the @a unreconciledAttrListArray is
 *        reconciled.
 *      - not all the LwSciBufAttrLists in the @a unreconciledAttrListArray are
 *        bound to the same LwSciBufModule.
 *      - @a unreconciledAttrListCount is 0
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descLen is NULL
 * - ::LwSciError_InsufficientResource if any of the following oclwrs:
 *      - the API is unable to implicitly append an additional attribute key
 *        when needed
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - Panic if @a any of the LwSciBufAttrList(s) in @a unreconciledAttrListArray
 *   is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListIpcExportUnreconciled(
    const LwSciBufAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * @brief Transforms the reconciled LwSciBufAttrList to an exportable reconciled
 * LwSciBufAttrList descriptor that can be transported by the application to any
 * remote process as a serialized set of bytes over an LwSciIpc channel.
 *
 * @param[in] reconciledAttrList The reconciled LwSciBufAttrList to be exported.
 * @param[in] ipcEndpoint LwSciIpcEndpoint.
 * @param[out] descBuf A pointer to the new reconciled LwSciBufAttrList
 *             descriptor, which the caller can deallocate later using
 *             LwSciBufAttrListFreeDesc().
 * @param[out] descLen The size of the new reconciled LwSciBufAttrList
 *             descriptor.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a reconciledAttrList is NULL
 *      - @a reconciledAttrList is unreconciled.
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descLen is NULL
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_NotPermitted if reconciled LwSciBufAttrList is not being
 *   exported in the reverse direction of IPC path in which unreconciled
 *   LwSciBufAttrLists ilwolved in reconciliation of input LwSciBufAttrList were
 *   exported.
 * - Panic if @a reconciledAttrList is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListIpcExportReconciled(
    LwSciBufAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * @brief Translates an exported unreconciled LwSciBufAttrList descriptor
 * (potentially received from any process) into an unreconciled LwSciBufAttrList.
 *
 * @param[in] module LwScibufModule with which to associate the
 *            imported LwSciBufAttrList.
 * @param[in] ipcEndpoint LwSciIpcEndpoint.
 * @param[in] descBuf The unreconciled LwSciBufAttrList descriptor to be
 *            translated into an unreconciled LwSciBufAttrList.  The valid value
 *            is non-NULL that points to descriptor received on LwSciIpc
 *            channel.
 * @param[in] descLen The size of the unreconciled LwSciBufAttrList descriptor.
 *            This value must be non-zero.
 * @param[out] importedUnreconciledAttrList The imported unreconciled
 *             LwSciBufAttrList.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descBuf represents an LwSciBufAttrList with invalid attribute key
 *        values set
 *      - @a descBuf represents an LwSciBufAttrList which is reconciled.
 *      - @a descBuf is invalid
 *      - @a descLen is 0
 *      - @a importedUnreconciledAttrList is NULL
 * - ::LwSciError_NotSupported if @a descBuf represents an LwSciBufAttrList with
 *     same key multiple times.
 * - ::LwSciError_InsufficientMemory if insufficient system memory.
 * - ::LwSciError_IlwalidState if imported LwSciBufAttrList cannot be
 *     associated with @a module.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - Panic if @a module is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListIpcImportUnreconciled(
    LwSciBufModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    LwSciBufAttrList* importedUnreconciledAttrList);

/**
 * @brief Translates an exported reconciled LwSciBufAttrList descriptor
 * (potentially received from any process) into a reconciled LwSciBufAttrList.
 *
 * It also validates that the reconciled LwSciBufAttrList to be imported will
 * be a reconciled LwSciBufAttrList that is consistent with the constraints in
 * an array of input unreconciled LwSciBufAttrList(s). This is recommended
 * while importing what is expected to be a reconciled LwSciBufAttrList to
 * cause LwSciBuf to validate the reconciled LwSciBufAttrList against the input
 * un-reconciled LwSciBufAttrList(s), so that the importing process can be sure
 * that an LwSciBufObj will satisfy the input constraints.
 *
 * @param[in] module LwScibufModule with which to associate the
 *            imported LwSciBufAttrList.
 * @param[in] ipcEndpoint LwSciIpcEndpoint.
 * @param[in] descBuf The reconciled LwSciBufAttrList descriptor to be
 *            translated into a reconciled LwSciBufAttrList.  The valid value is
 *            non-NULL that points to descriptor received on LwSciIpc channel.
 * @param[in] descLen The size of the reconciled LwSciBufAttrList descriptor.
 *            This value must be non-zero.
 * @param[in] inputUnreconciledAttrListArray The array of unreconciled
 *            LwSciBufAttrList against which the new reconciled
 *            LwSciBufAttrList is to be validated. NULL pointer is acceptable
 *            as a parameter if the validation needs to be skipped.
 * @param[in] inputUnreconciledAttrListCount The number of unreconciled
 *            LwSciBufAttrList(s) in @a inputUnreconciledAttrListArray. If
 *            @a inputUnreconciledAttrListCount is non-zero, then this operation
 *            will fail with an error unless all the constraints of all the
 *            unreconciled LwSciBufAttrList(s) in inputUnreconciledAttrListArray
 *            are met by the imported reconciled LwSciBufAttrList.
 * @param[out] importedReconciledAttrList Imported reconciled LwSciBufAttrList.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descBuf represents an LwSciBufAttrList with invalid attribute key
 *        values set
 *      - @a descBuf represents an LwSciBufAttrList which is unreconciled.
 *      - @a descBuf is invalid
 *      - @a descLen is 0
 *      - @a importedReconciledAttrList is NULL
 *      - @a inputUnreconciledAttrListCount is 0 provided
 *        @a inputUnreconciledAttrListArray is non-NULL
 * - ::LwSciError_NotSupported if any of the following oclwrs:
 *      - @a descBuf is incompatible
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_AttrListValidationFailed if input unreconciled
 *     LwSciBufAttrList(s)' attribute constraints are not satisfied by
 *     attributes associated with the imported importedReconciledAttrList.
 * - ::LwSciError_IlwalidState if imported LwSciBufAttrList cannot be
 *     associated with @a module.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - Panic if:
 *      - @a any of the LwSciBufAttrList in
 *        inputUnreconciledAttrListArray is invalid
 *      - @a module is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufAttrListIpcImportReconciled(
    LwSciBufModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const LwSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciBufAttrList* importedReconciledAttrList);


/**
 * @brief Frees the LwSciBuf exported LwSciBufAttrList descriptor.
 *
 * @param[in] descBuf LwSciBufAttrList descriptor to be freed.  The valid value
 * is non-NULL.
 *
 * @return void
 * - Panics if:
 *     - @a descBuf is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the @a descBuf to be
 *        freed
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciBufAttrListFreeDesc(
    void* descBuf);

/**
 * @}
 */

/**
 * @defgroup lwscibuf_init_api LwSciBuf Initialization APIs
 * List of APIs to initialize/de-initialize LwSciBuf module.
 * @{
 */

/**
 * @brief Initializes and returns a new LwSciBufModule with no
 * LwSciBufAttrLists, buffers, or LwSciBufObjs bound to it.
 * @note A process may call this function multiple times.
 * Each successful invocation will yield a new LwSciBufModule.
 *
 * @param[out] newModule The new LwSciBufModule.
 *
 * @return ::LwSciError, the completion code of this operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a newModule is NULL.
 * - ::LwSciError_InsufficientMemory if memory is not available.
 * - ::LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed
 *      - system lacks resource other than memory
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufModuleOpen(
    LwSciBufModule* newModule);

/**
 * @brief Releases the LwSciBufModule obtained through
 * an earlier call to LwSciBufModuleOpen(). Once the LwSciBufModule is closed
 * and all LwSciBufAttrLists and LwSciBufObjs bound to it
 * are freed, the LwSciBufModule will be de-initialized in
 * the calling process.
 *
 * @note Every owner of the LwSciBufModule shall call LwSciBufModuleClose()
 * only after all the functions ilwoked by the owner with LwSciBufModule as
 * an input are completed.
 *
 * @param[in] module The LwSciBufModule to close.
 *
 * @return void
 * - Panic if @a module is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the input
 *        LwSciBufModule @a module
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciBufModuleClose(
    LwSciBufModule module);

/**
 * @brief Checks if loaded LwSciBuf library version is compatible with
 * LwSciBuf library version with which elements dependent on LwSciBuf
 * were built.
 * This function checks loaded LwSciBuf library version with input LwSciBuf
 * library version and sets output variable true provided major version of the
 * loaded library is same as @a majorVer and minor version of the
 * loaded library is not less than @a minorVer.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Additionally, this function also checks the  versions of libraries that
 * LwSciBuf depends on and sets the output variable to true if all libraries are
 * compatible, else sets output to false.
 */
#endif
/**
 *
 * @param[in] majorVer build major version.
 * @param[in] minorVer build minor version.
 * @param[out] isCompatible boolean value stating if loaded LwSciBuf library is
 * compatible or not.
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a isCompatible is NULL
 */
#if (LW_IS_SAFETY == 0)
/**
 *      - failed to check dependent library versions.
 */
#endif
/**
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciBufCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible);

/**
 * @}
 */

/** @} */

#if defined(__cplusplus)
}
#endif // __cplusplus

#endif /* INCLUDED_LWSCIBUF_H */
