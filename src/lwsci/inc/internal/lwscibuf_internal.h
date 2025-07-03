/*
 * Copyright (c) 2018-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_INTERNAL_H
#define INCLUDED_LWSCIBUF_INTERNAL_H

#include "lwscibuf.h"
#include "lwscievent.h"

#ifndef LWSCI_ARCHITECTURE_DESKTOP
#if defined(__x86_64__)
#define LWSCI_ARCHITECTURE_DESKTOP 1
#else
#define LWSCI_ARCHITECTURE_DESKTOP 0
#endif
#endif

#if LWSCI_ARCHITECTURE_DESKTOP
#include "lwscibuf_internal_x86.h"
#else
#include "lwscibuf_internal_tegra.h"
#endif
#include "lwcolor.h"

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @defgroup lwscibuf_blanket_statements LwSciBuf blanket statements.
 * Generic statements applicable for LwSciBuf interfaces.
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements LwSciBuf blanket statements
 * \section lwscibuf_in_params Input parameters
 * - LwSciBufObj is valid if it is obtained from successful call to
 * LwSciBufObjCreateFromMemHandle() and has not been deallocated using
 * LwSciBufObjFree().
 */

/**
 * @}
 */

/**
 * @defgroup lwscibuf_int_consts LwSciBuf Internal - Global Constants
 * List of all LwSciBuf internal global constants.
 * @{
 */

#if defined(__cplusplus)

/**
 * @brief Global Constant to define the count of lwmedia flags.
 */
static const int LW_SCI_BUF_LW_MEDIA_FLAG_COUNT = 32;

/**
 * @brief Maximum number of HW Engines supported
 */
static const int LW_SCI_BUF_HW_ENGINE_MAX_NUMBER = 128;

/*
 * @brief Global constant that defines number of bit used for encoding key-type.
 */
static const int LW_SCI_BUF_KEYTYPE_BIT_COUNT = 3;

/*
 * @brief Global constant to specify the key-type of internal attributes.
 */
static const int LW_SCI_BUF_ATTR_KEY_TYPE_INTERNAL = 1;

/*
 * @brief Global constant to specify the key-type of internal attributes.
 */
static const int LW_SCI_BUF_ATTR_KEY_TYPE_INTERNALAPP_PRIVATE = 2;

/**
 * @brief Global Constant to specify starting value of
 *        General internal attribute keys.
 */
static const int  LW_SCI_BUF_GENERAL_ATTR_INTERNAL_KEY_START =
      (LW_SCI_BUF_ATTR_KEY_TYPE_INTERNAL << LW_SCI_BUF_KEYTYPE_BIT_START);

/**
 * @brief Global Constant to specify the start of Image internal Datatype keys.
 */
static const int  LW_SCI_BUF_IMAGE_ATTR_INTERNAL_KEY_START  =
      ((LW_SCI_BUF_ATTR_KEY_TYPE_INTERNAL << LW_SCI_BUF_KEYTYPE_BIT_START) |
       (LwSciBufType_Image << LW_SCI_BUF_ATTRKEY_BIT_COUNT));

/**
 * @brief Global Constant to specify the start of Internal App Private Keys.
 */
static const int LW_SCI_BUF_INTERNAL_APP_PRIVATE_KEY_START  =
      (LW_SCI_BUF_ATTR_KEY_TYPE_INTERNALAPP_PRIVATE <<
       LW_SCI_BUF_KEYTYPE_BIT_START);

/**
 * @brief Global Constant to define Maximum number of attribute keys per type.
 */
static const int LW_SCI_BUF_MAX_KEY_COUNT =
              ((1 << LW_SCI_BUF_ATTRKEY_BIT_COUNT) - 1);

#else

/**
 * @brief Global Constant to define the count of lwmedia flags.
 *
 * @implements{18840144}
 */
#define LW_SCI_BUF_LW_MEDIA_FLAG_COUNT 32u
/**
 * @brief Global Constant to define maximum number of LwSciBufHwEngine
 * supported.
 *
 * @implements{18840147}
 */
#define LW_SCI_BUF_HW_ENGINE_MAX_NUMBER  128u

/*
 * @brief Global constant that defines number of bits used
 *        for encoding attribute key type.
 */
#define LW_SCI_BUF_KEYTYPE_BIT_COUNT  3u

/**
 * @brief Global constant to indicate the attribute key
 *        type is internal.
 */
#define LW_SCI_BUF_ATTR_KEY_TYPE_INTERNAL  1u

/*
 * @brief Global constant to specify the key type of UMD specific
 *        LwSciBufInternalAttrKey(s).
 */
#define LW_SCI_BUF_ATTR_KEY_TYPE_INTERNALAPP_PRIVATE 2u

/**
 * @brief Global Constant to specify the starting value of
 *        LwSciBufInternalAttrKey for LwSciBufType_General.
 */
#define LW_SCI_BUF_GENERAL_ATTR_INTERNAL_KEY_START  \
     (LW_SCI_BUF_ATTR_KEY_TYPE_INTERNAL << LW_SCI_BUF_KEYTYPE_BIT_START)

/**
 * @brief Global Constant to specify the starting value of
 *        LwSciBufInternalAttrKey for LwSciBufType_Image.
 */
#define LW_SCI_BUF_IMAGE_ATTR_INTERNAL_KEY_START  \
      ((LW_SCI_BUF_ATTR_KEY_TYPE_INTERNAL << LW_SCI_BUF_KEYTYPE_BIT_START) | \
       (LwSciBufType_Image << LW_SCI_BUF_ATTRKEY_BIT_COUNT))

/**
 * @brief Global Constant to specify the starting value of
 *        UMD specific LwSciBufInternalAttrKey.
 */
#define LW_SCI_BUF_INTERNAL_APP_PRIVATE_KEY_START  \
        (LW_SCI_BUF_ATTR_KEY_TYPE_INTERNALAPP_PRIVATE <<  \
         LW_SCI_BUF_KEYTYPE_BIT_START)

/**
 * @brief Global Constant to define Maximum number of
 *        LwSciBufInternalAttrKey(s) per type.
 */
#define LW_SCI_BUF_MAX_KEY_COUNT \
            (((uint32_t)1 << LW_SCI_BUF_ATTRKEY_BIT_COUNT) - (uint32_t)1)

#endif
/**
 * @}
 */


/**
 * @defgroup attr_key_int LwSciBuf internal enums for attribute keys
 * List of all LwSciBuf Internal Enumerations for attribute keys
 * @{
 */

/**
 * @brief Describes the LwSciBuf internal attribute keys holding corresponding
 * values specifying buffer constraints.
 * The accessibility property of an attribute refers to whether the value of an
 * attribute is accessible in an LwSciBufAttrList. Input attribute keys specify
 * desired buffer constraints from client and can be set/retrieved by client
 * to/from unreconciled LwSciBufAttrList using
 * LwSciBufAttrListSetInternalAttrs()/LwSciBufAttrListGetInternalAttrs()
 * respectively.
 * Output attribute keys specify actual buffer constraints computed by LwSciBuf
 * if reconciliation succeeds. Output attributes can be retrieved from
 * reconciled LwSciBufAttrList using LwSciBufAttrListGetInternalAttrs().
 * The presence property of an attribute refers to whether the value of an
 * attribute having accessibility as input needs to be present in at least one
 * of the unreconciled attribute lists for reconciliation.
 * The presence property of an attribute can have one of the three values:
 * Mandatory/Optional/Conditional.
 * Mandatory implies that it is mandatory that the value of an attribute be set
 * in at least one of the unreconciled LwSciBufAttrLists ilwolved in
 * reconciliation.
 * Optional implies that it is not mandatory that value of an attribute be set
 * in at least of the unreconciled LwSciBufAttrLists ilwolved in reconciliation.
 * Conditional implies that the presence of an attribute is mandatory if
 * condition associated with its presence is satisfied otherwise its optional.
 *
 * @implements{17824164}
 */
typedef enum {
    /** Invalid key. Needed for lower bound check on LwSciBuf internal attribute
     * keys.
     * NOTE: external attribute keys occupy space of 0 - 32k. Thus,
     * internal keys should start from 32K
     *
     * Value: None
     */
    LwSciBufInternalAttrKey_LowerBound =
                            LW_SCI_BUF_GENERAL_ATTR_INTERNAL_KEY_START,

    /** List of engines that the buffer should be accessible to, and whose
     * constraints are to be considered for allocation. During reconciliation,
     * value of this attribute is set to aggregate of all the values for the
     * same attribute specified by all the unreconciled LwSciBufAttrLists.
     * The value of this attribute is set to default if none of the unreconciled
     * LwSciBufAttrList(s) ilwolved in reconciliation set this attribute.
     *
     * During reconciliation, buffer constraints corresponding to every
     * LwSciBufHwEngine in reconciled LwSciBufAttList applicable for
     * every LwSciBufType in the same LwSciBufAttrList are reconciled and
     * considered as consolidated constraints for buffer allocation.
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
     * Value: LwSciBufHwEngine[]
     *
     * valid input value: valid LwSciBufHwEngine.
     */
    LwSciBufInternalGeneralAttrKey_EngineArray,


    /** An array of memory domains that the allocation can come from.
     * During reconciliation, the value of this attribute is set to the set of
     * values that are common among all the unreconciled LwSciBufAttrLists that
     * have this attribute set.
     * This attribute is set to default value of LwSciBufMemDomain_Sysmem if
     * none of the unreconciled LwSciBufAttrList(s) ilwolved in reconciliation
     * set this attribute.
     * In safety, only LwSciBufMemDomain_Sysmem is supported as valid memory
     * domain.
     */
#if (LW_IS_SAFETY == 0)
    /**
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if any of the values
     * in the set of values of this attribute in the provided reconciled
     * LwSciBufAttrList is not present in the set of values of any of the input
     * unreconciled LwSciBufAttrList(s) that have this attribute set.
     */
#else
    /**
     * Validation of this attribute will always pass in safety since in safety,
     * only LwSciBufMemDomain_Sysmem is supported.
     *
     */
#endif
    /**
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: LwSciBufMemDomain[]
     */
    LwSciBufInternalGeneralAttrKey_MemDomainArray,

    /** this is an output attribute
     * Per plane GOB size.
     * The value of this attribute is always reconciled to 0 for each plane
     * in the reconciled LwSciBufAttrList.
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
     *
     * Value: uint32_t[]
     */
    LwSciBufInternalImageAttrKey_PlaneGobSize =
                  LW_SCI_BUF_IMAGE_ATTR_INTERNAL_KEY_START,

    /** this is an output attribute
     * The value of this attribute is always reconciled to 0 for each plane
     * in the reconciled LwSciBufAttrList.
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
     *
     * Value: uint32_t[]
     */
    LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,

    /** this is an output attribute
     * The value of this attribute is always reconciled to 1 for each plane
     * in the reconciled LwSciBufAttrList.
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
     *
     * Value: uint32_t[]
     */
    LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,

    /** this is an output attribute
     * The value of this attribute is always reconciled to 0 for each plane
     * in the reconciled LwSciBufAttrList.
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
     *
     * Value: uint32_t[]
     */
    LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,

    /** 64K keys for private UMD data.
     * UMDs are advised to prefer fine-grained attributes over structures so
     * that reconciliation can be done for a subset of attributes, where some
     * attributes are specified by one end and some by the other end.
     * However, it is also possible for UMDs to record all their private
     * attributes in a single value structure for a single key. If more than one
     * unreconciled LwSciBufAttrLists set these keys, LwSciBuf compares
     * values of same keys. Reconciliation succeeds only if the values match.
     *
     * During validation of reconciled LwSciBufAttrList against input
     * unreconciled LwSciBufAttrList(s), validation fails if value of this
     * attribute set in any of the input unreconciled LwSciBufAttrList(s) is
     * not equal to the value of this attribute in the provided reconciled
     * LwSciBufAttrList.
     *
     * Accessibility: Input/Output attribute
     * Presence: Optional
     *
     * Value: User specified custom datatype
     */
    LwSciBufInternalAttrKey_LwMediaPrivateFirst =
                   LW_SCI_BUF_INTERNAL_APP_PRIVATE_KEY_START,
    LwSciBufInternalAttrKey_LwMediaPrivateLast =
        (LwSciBufInternalAttrKey_LwMediaPrivateFirst +
         LW_SCI_BUF_MAX_KEY_COUNT),

    LwSciBufInternalAttrKey_DriveworksPrivateFirst,
    LwSciBufInternalAttrKey_DriveworksPrivateLast =
        (LwSciBufInternalAttrKey_DriveworksPrivateFirst +
         LW_SCI_BUF_MAX_KEY_COUNT),

} LwSciBufInternalAttrKey;


/**
 * @}
 */

/**
 * @defgroup attr_val_int LwSciBuf internal Datastructures
 * List of all LwSciBuf internal datastructures.
 * @{
 */

/**
 * @brief Enum to identify hardware engines from cheetah
 *        or resman world
 *
 * @implements{18840126}
 */
typedef enum {
    LwSciBufHwEngine_TegraNamespaceId,
    LwSciBufHwEngine_ResmanNamespaceId
} LwSciBufHwEngNamespace;

/**
 * @brief Enum to identify hardware engines
 *
 * @implements{18840123}
 */
typedef enum {
    /* NOTE: we are removing LwSciBuf dependency on lwrm_module.h by defining
     * these enums. In order to not break UMD code, we are keeping enum values
     * here consistent with enum values in lwrm_module.h.
     * Once, UMDs move to using lwscibuf engine names instead of lwrm module-ids
     * , we can assign enum values sequentially.
     */
    LwSciBufHwEngName_Ilwalid   = 0,
    LwSciBufHwEngName_Display   = 4,
    LwSciBufHwEngName_Isp       = 11,
    LwSciBufHwEngName_Vi        = 12,
    LwSciBufHwEngName_Csi       = 30,
    LwSciBufHwEngName_Vic       = 106,
    LwSciBufHwEngName_Gpu       = 107,
    LwSciBufHwEngName_MSENC     = 109,
    LwSciBufHwEngName_LWDEC     = 117,
    LwSciBufHwEngName_LWJPG     = 118,
    LwSciBufHwEngName_PVA       = 121,
    LwSciBufHwEngName_DLA       = 122,
    LwSciBufHwEngName_PCIe      = 123,
    LwSciBufHwEngName_OFA       = 124,
    LwSciBufHwEngName_Num       = 125
} LwSciBufHwEngName;

/**
 * @brief Enum to identify memory domain
 *
 * @implements{18840141}
 */
typedef enum {
#if (LW_IS_SAFETY == 0)
    LwSciBufMemDomain_Cvsram = 0,
#endif
    LwSciBufMemDomain_Sysmem = 1,
#if (LW_IS_SAFETY == 0)
    LwSciBufMemDomain_Vidmem = 2,
#endif
    LwSciBufMemDomain_UpperBound = 3
} LwSciBufMemDomain;

/**
 * @brief Union specifying revision of the hardware engine
 *  accessing LwSciBuf.
 */
typedef union {
    /** Revision of the non-GPU hardware engine. This should not be filled by user,
     *  LwSciBuf fills it.
     */
    int32_t engine_rev;

    /** Revision of the GPU hardware engine */
    struct {
        /** GPU HW architecture. It should be initialized as 0x150
         *  by UMDs (example: LwMedia) for cheetah usecases.
         */
        uint32_t arch;

        /** Implementation version. It should be initialized to zero by UMDs
         *  (example: LwMedia) for cheetah usecases since it is not being used
         *  for them.
         */
        uint32_t impl;

        /** GPU HW revision. It should be initialized to zero by UMDs (example: LwMedia)
         *  for cheetah usecases since it is not being used for them.
         */
        uint32_t rev;
    }  __attribute__((packed)) gpu;    // For GPUs.
}  __attribute__((packed)) LwSciBufHwEngineRevId;

/**
 * @brief Structure identifying information about the hardware engine accessing
 *  LwSciBuf. An attribute key, LwSciBufInternalGeneralAttrKey_EngineArray,
 *  set to array of this structure specifies all the hardware engines whose
 *  constraints should be taken into account while allocating the buffer.
 */
typedef struct {
    /** Specifies the hardware engine is from cheetah or resman
     *  world. It is initialized to LwSciBufHwEngine_TegraNamespaceId
     *  for cheetah usecases.
     */
    LwSciBufHwEngNamespace engNamespace;
    /**
     * Hardware engine ID specifying LwSciBufHwEngName for which buffer
     * constraints need to be applied. It should be initialized by calling
     * LwSciBufHwEngCreateIdWithoutInstance() for engines without instance
     * and is initialized by calling LwSciBufHwEngCreateIdWithInstance()
     * for engines having multiple instances. This member is applicable
     * only for cheetah usecases. Note that GPUs accessing the buffer are
     * specified by attribute LwSciBufGeneralAttrKey_GpuId and thus should
     * not be specified using LwSciBufHwEngine.
     */
    int64_t rmModuleID;
    /**
     * This field is unused. If user fills it, it will be ignored.
     */
    uint32_t subEngineID;

    /**
     * Specifies the revision of the hardware engine.
     */
    LwSciBufHwEngineRevId   rev;
} __attribute__((packed)) LwSciBufHwEngine;

/**
 * @}
 */

/**
 * @defgroup lwscibuf_attr_datatype_int LwSciBuf internal attribute datatypes
 * List of all LwSciBuf internal datatypes for attributes
 * @{
 */

/**
 * @brief This structure defines a key/value pair used to get or set
 * the LwSciBufInternalAttrKey(s) and their corresponding values from or
 * to LwSciBufAttrList.
 *
 * @note An array of this structure need to be set in order to
 * allocate a buffer.
 *
 * @implements{18840135}
 */
typedef struct LwSciBufInternalAttrKeyValuePairRec {
    /** LwSciBufInternalAttrKey value for which value needs to be
     * set/retrieved. This member is initialized to any one of
     * the LwSciBufInternalAttrKey other than
     * LwSciBufInternalAttrKey_LowerBound */
    LwSciBufInternalAttrKey key;

    /** Pointer to the value corresponding to the attribute.
     * If the value is an array, the pointer points to the first element. */
    const void* value;

    /** Length of the value in bytes */
    size_t len;
} LwSciBufInternalAttrKeyValuePair;

/**
 * @}
 */

/**
 * @defgroup lwscibuf_attr_list_api_int LwSciBuf internal Attribute list APIs
 * Attribute list APIs exposed internally
 * @{
 */

/**
 * @brief Sets the value of LwSciBufInternalAttrKey(s) in the LwSciBufAttrList.
 * It only reads values from LwSciBufInternalAttrKeyValuePair array and saves
 * copies during this call.
 *
 * @note The LwSciBufAttrListSetInternalAttrs() and
 * LwSciBufAttrListGetInternalAttrs() APIs can be called conlwrrently,
 * however, function completion order is not guaranteed by LwSciBuf
 * and thus outcome of calling these functions conlwrrently is
 * undefined.
 *
 * @param[in] attrList Unreconciled LwSciBufAttrList.
 * @param[in] pairArray Array of LwSciBufInternalAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and key member of
 * every LwSciBufInternalAttrKeyValuePair in the array is an input or
 * input/output attribute and it is a valid enumeration value defined by the
 * LwSciBufInternalAttrKey enum or a UMD-specific LwSciBufInternalAttrKey
 * and the value member of every LwSciBufInternalAttrKeyValuePair in the array
 * is not NULL.
 * @param[in] pairCount number of elements/entries in pairArray
 * Valid value: pairCount is valid input if it is non-zero.
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a attrList is reconciled
 *      - @a attrList is an unreconciled LwSciBufAttrList obtained from
 *        LwSciBufAttrListAppendUnreconciled or
 *        LwSciBufAttrListIpcImportUnreconciled
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - any of the LwSciBufInternalAttrKey(s) specified in the
 *        LwSciBufInternalAttrKeyValuePair array is output only
 *      - any of the LwSciBufInternalAttrKey(s) specified in the
 *        LwSciBufInternalAttrKeyValuePair array has already been set
 *      - any of the LwSciBufInternalAttrKey(s) specified in the
 *        LwSciBufInternalAttrKeyValuePair array oclwrs more than once
 *      - the LwSciBufGeneralAttrKey_Types key is not set on @a attrList
 *        before attempting to set datatype-specific LwSciBufInternalAttrKey(s)
 *      - any of the LwSciBufInternalAttrKey(s) specified in @a pairArray is not
 *        a valid enumeration value defined in the LwSciBufInternalAttrKey enum
 *        or is not a UMD-specific LwSciBufInternalAttrKey
 *      - length(s) set for LwSciBufInternalAttrKey(s) in @a pairArray are
 *        invalid
 *      - value(s) set for LwSciBufInternalAttrKey(s) in @a pairArray are invalid
 * - LwSciError_InsufficientMemory if not enough system memory
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid.
 */
LwSciError LwSciBufAttrListSetInternalAttrs(
    LwSciBufAttrList attrList,
    LwSciBufInternalAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * @brief Returns an array of LwSciBufInternalAttrKeyValuePair from
 * LwSciBufAttrList for a given set of LwSciBufInternalAttrKey(s).
 * This function accepts a set of LwSciBufInternalAttrKey(s) passed in
 * the @ref LwSciBufInternalAttrKeyValuePair structure. The return values,
 * stored back into @ref LwSciBufInternalAttrKeyValuePair, consist of
 * @c const @c void* pointers to the attribute values from the @ref
 * LwSciBufAttrList. The application must not write to this data.
 *
 * @param[in] attrList LwSciBufAttrList to fetch the
 * LwSciBufInternalAttrKeyValuePair(s) from.
 * @param[in,out] pairArray Array of LwSciBufInternalAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and key member of every
 * LwSciBufInternalAttrKeyValuePair in the array is a valid enumeration value
 * defined by the LwSciBufInternalAttrKey enum.
 * @param[out] pairCount Number of elements/entries in pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a attrList is NULL
 *    - @a pairArray is NULL
 *    - @a pairCount is 0
 *    - @a attrList is reconciled and any of the LwSciBufInternalAttrKey(s)
 *      specified in LwSciBufInternalAttrKeyValuePair array is input only.
 *    - @a attrList is unreconciled and any of the LwSciBufInternalAttrKey(s)
 *      specified in LwSciBufInternalAttrKeyValuePair array is output only.
 *    - any of the LwSciBufInternalAttrKey(s) specified in @a pairArray is not
 *      a valid enumeration value defined in the LwSciBufInternalAttrKey enum
 *      or is not a UMD-specific LwSciBufInternalAttrKey
 * - Panics if any of the following oclwrs:
 *    - @a attrList is not valid
 */
LwSciError LwSciBufAttrListGetInternalAttrs(
    LwSciBufAttrList attrList,
    LwSciBufInternalAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * @brief Adds @a offset within given UMD specific LwSciBufInternalAttrKey,
 * returning new LwSciBufInternalAttrKey from given offset.
 *
 * @param[in] key LwSciBufInternalAttrKey.
 * Valid value: value within range of defined UMD private keys.
 * @param[in] offset offset that needs to be added to @a key.
 * Valid value: offset value which satisfies @a key + @a offset <= 64k
 * @param[out] offsettedKey offsetted LwSciBufInternalAttrKey.
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a offsettedKey is NULL
 *    - @a key is not a UMD specific LwSciBufInternalAttrKey.
 * - LwSciError_Overflow if the output after adding offset is not a
 *   UMD specific LwSciBufInternalAttrKey.
 * - LwSciError_BadParameter if offset > 64k.
 */
LwSciError LwSciBufGetUMDPrivateKeyWithOffset(
    LwSciBufInternalAttrKey key,
    uint32_t offset,
    LwSciBufInternalAttrKey* offsettedKey);

/**
 * @}
 */

/**
 * @defgroup lwscibuf_obj_api_int LwSciBuf internal object APIs
 * List of internal APIs to operate on LwSciBuf object
 * @{
 */

#if (LW_IS_SAFETY == 0)
/**
 * @brief Creates sub-buffer out of valid buffer.
 *
 * @param[in] parentObj LwSciBuf object from which sub-buffer needs to be
 * created
 * @param[in] offset offset into parent buffer
 * @param[in] len length of sub-buffer
 * @param[in] reconciledAttrList reconciled attribute list for sub-buffer
 *
 * @param[out] childObj LwSciBuf object for sub-buffer
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a parentObj is invalid
 *      - @a reconciledAttrList is invalid
 *      - @a childObj is NULL
 * - ::LwSciError_IlwalidState if any of the following oclwrs:
 *      - the newly created child LwSciBufObj cannot be associated with the
 *        LwSciBufModule with which @a reconciledAttrList is associated
 *      - number of references to "memory object" of parent LwSciBufObj is
 *        INT32_MAX and the child LwSciBufObj tries to take one more reference
 *        using this API
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 */
LwSciError LwSciBufObjCreateSubBuf(
    LwSciBufObj parentObj,
    size_t offset,
    size_t len,
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObj* childObj);
#endif

/**
 * @brief Retrieves LwSciBufRmHandle, offset and length of the buffer
 * represented by LwSciBufObj.
 * @note LwRmMemHandle returned as part of LwSciBufRmHandle is owned by LwSciBuf
 * and thus must not be freed by the caller.
 *
 * @param[in] bufObj LwSciBufObj.
 *
 * @param[out] memHandle LwSciBufRmHandle associated with LwSciBufObj.
 * @param[out] offset Offset of the buffer represented by LwSciBufObj within
 * the buffer represented by LwSciBufRmHandle.
 * @param[out] len Length of the buffer represented by LwSciBufObj. The length
 * of the buffer represented by LwSciBufRmHandle is greater than or equal to
 * @a offset + @a len.
 *
 * @return LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a bufObj is NULL.
 *    - @a memHandle is NULL.
 *    - @a offset is NULL.
 *    - @a len is NULL.
 * - Panics if @a bufObj is invalid.
 */
LwSciError LwSciBufObjGetMemHandle(
    LwSciBufObj bufObj,
    LwSciBufRmHandle* memHandle,
    uint64_t* offset,
    uint64_t* len);

/**
 * @brief Creates a new memory object containing specified buffer properties,
 * offset and len, specified reconciled LwSciBufAttrList and buffer handle,
 * LwSciBufRmHandle and references it by creating new LwSciBufObj.
 *
 * @param[in] memHandle LwSciBufRmHandle.
 * Valid value: memHandle is valid input if the RM memory handle represented
 * by it is received from a successful call to LwRmMemHandleAllocAttr() and
 * has not been deallocated by using LwRmMemHandleFree().
 * @param[in] offset The offset within the buffer represented by
 * LwSciBufRmHandle to be represented by the new LwSciBufObj.
 * Valid value: 0 to size of the buffer represented by LwSciBufRmHandle - 1.
 * @param[in] len The length of the buffer to be represented by the new
 * LwSciBufObj. The size of the buffer represented by LwSciBufRmHandle must
 * be at least @a offset + @a len.
 * Valid value: 1 to size of the buffer represented by LwSciBufRmHandle -
 * @a offset.
 * @param[in] reconciledAttrList A reconciled LwSciBufAttrList specifying the
 * buffer constraints of the buffer represented by the new LwSciBufObj.
 * @param[out] bufObj new LwSciBufObj.
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a bufObj is NULL.
 *    - @a reconciledAttrList is NULL.
 *    - @a reconciledAttrList is unreconciled.
 *    - @a len is 0.
 *    - @a offset + @a len > buffer size represented by LwSciBufRmHandle.
 *    - @a len > buffer size represented by output attributes for respective
 *      LwSciBufType in @a reconciledAttrList.
 *    - buffer size represented by output attributes for respective
 *      LwSciBufType in @a reconciledAttrList > buffer size represented by
 *      LwSciBufRmHandle.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_IlwalidState if a new LwSciBufObj cannot be associated with the
 *   LwSciBufModule with which @a reconciledAttrList is associated to create
 *   the new LwSciBufObj.
 * - LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed
 *      - system lacks resource other than memory
 * - LwSciError_Overflow if @a len + @a offset exceeds UINT64_MAX.
 * - LwSciError_NotSupported if the memory domain on the LwSciBufAttrList is
 *   not supported.
 * - Panics if any of the following oclwrs:
 *      - @a reconciledAttrList is invalid
 */
LwSciError LwSciBufObjCreateFromMemHandle(
    const LwSciBufRmHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrList reconciledAttrList,
    LwSciBufObj* bufObj);

/**
 * @brief This API is used to get and set LwMedia flags for an LwSciBufObj,
 * each of the 32 bits represent a flag. LwMedia can use each of 32 bits for
 * each of LwMedia datatypes. When LwSciBufObj is created or duplicated using
 * duplicate APIs, all the 32 bits are set to 0.
 *
 * @param[in] bufObj LwSciBufObj.
 * @param[in] flagIndex Index of bit to change in the flag.
 * Valid value: 0 to 31.
 * @param[in] newValue New value to set. Valid value: true or false.
 *
 * @return bool
 * - false if value of bit indexed by @a flagIndex was 0 before
 *   @a newValue is set.
 * - true if value of bit indexed by @a flagIndex was 1 before @a newValue
 *   is set.
 * - Panics if @a bufObj is invalid.
 */
bool LwSciBufObjAtomicGetAndSetLwMediaFlag(
    LwSciBufObj bufObj,
    uint32_t flagIndex,
    bool newValue);

/**
 * @brief Increments a refcount on LwSciBufObj.
 *
 * @note Incrementing refcount using this API implies taking multiple
 * references to the memory object via same LwSciBufObj.
 *
 * @param[in] bufObj LwSciBufObj for which reference needs to be incremented
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a bufObj is NULL
 * - LwSciError_IlwalidState if total number of references to memory
 *   object are INT32_MAX and caller tries to take one more reference
 *   using this API.
 * - Panics if @a bufObj is invalid.
 */
LwSciError LwSciBufObjRef(
    LwSciBufObj bufObj);

/**
 * @}
 */

/**
 * @defgroup lwscibuf_hw_engine_api LwSciBuf APIs to get/set HW engine ID
 * List of APIs exposed internally to get/set LwSciBuf HW engine IDs
 * @{
 */

/**
 * @brief Generates LwSciBuf hardware engine ID from the given LwSciBufHwEngName
 *        without hardware engine instance.
 *
 * @param[in] engName: LwSciBufHwEngName from which the ID is obtained.
 *  Valid value: LwSciBufHwEngName enum value > LwSciBufHwEngName_Ilwalid and <
 *  LwSciBufHwEngName_Num.
 * @param[out] engId: LwSciBuf hardware engine ID generated from LwSciBufHwEngName.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a engName is invalid
 *      - @a engId is NULL
 */
LwSciError LwSciBufHwEngCreateIdWithoutInstance(
    LwSciBufHwEngName engName,
    int64_t* engId);

/**
 * @brief Generates LwSciBuf hardware engine ID from the given LwSciBufHwEngName
 *        and given hardware engine instance.
 *
 * @param[in] engName: LwSciBufHwEngName from which the ID is obtained.
 *  Valid value: LwSciBufHwEngName enum value > LwSciBufHwEngName_Ilwalid and <
 *  LwSciBufHwEngName_Num.
 * @param[in] instance: hardware engine instance of LwSciBufHwEngName.
 *  Valid value: 0 to UINT32_MAX.
 * @param[out] engId: LwSciBuf hardware engine ID generated from LwSciBufHwEngName
 *  and its corresponding hardware engine instance.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a engName is invalid
 *      - @a engId is NULL.
 */
LwSciError LwSciBufHwEngCreateIdWithInstance(
    LwSciBufHwEngName engName,
    uint32_t instance,
    int64_t* engId);

/**
 * @brief Retrieves LwSciBufHwEngName from LwSciBuf hardware engine ID.
 *
 * @param[in] engId: LwSciBuf hardware engine ID.
 *  Valid value: engine ID obtained from successful call to
 *  LwSciBufHwEngCreateIdWithoutInstance() or
 *  LwSciBufHwEngCreateIdWithInstance().
 * @param[out] engName: LwSciBufHwEngName retrieved from engine ID.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a engId is invalid
 *       - @a engName is NULL.
 */
LwSciError LwSciBufHwEngGetNameFromId(
    int64_t engId,
    LwSciBufHwEngName* engName);

/**
 * @brief Retrieves hardware engine instance from LwSciBuf
 *        hardware engine ID.
 *
 * @param[in] engId: LwSciBuf hardware engine ID.
 *  Valid value: engine ID obtained from successful call to
 *  LwSciBufHwEngCreateIdWithoutInstance() or
 *  LwSciBufHwEngCreateIdWithInstance().
 * @param[out] instance: Hardware engine instance retrieved from
 *  engine ID.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a instance is NULL.
 *       - @a engId is invalid.
 */
LwSciError LwSciBufHwEngGetInstanceFromId(
    int64_t engId,
    uint32_t* instance);

/**
 * @}
 */

/**
 * @defgroup lwscibuf_helper_api LwSciBuf Helper APIs
 * List of helper function APIs
 * @{
 */

/**
 * @brief This API is used to retrieve LwColorFormat for a given
 * LwSciBufAttrValColorFmt
 *
 * @param[in] lwSciColorFmt LwSciBufAttrValColorFmt that needs to be
 *            transformed. The valid data range is LwSciColor_LowerBound <
 *            lwSciColorFmt < LwSciColor_UpperBound.
 * @param[out] lwColorFmt LwColorFormat.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a lwColorFmt is NULL
 *      - @a lwSciColorFmt is not a supported by LwSciBuf
 *
 */
LwSciError LwSciColorToLwColor(
    LwSciBufAttrValColorFmt lwSciColorFmt,
    LwColorFormat* lwColorFmt);

/**
 * @brief This API is used to retrieve LwSciBufAttrValColorFmt for a given
 * LwColorFormat
 *
 * @param[in] lwColorFmt LwColorFormat that needs to be transformed. The
 *            valid data range is LwColorFormat_Unspecified < lwColorFmt
 *            < LwColorFormat_Force64.
 * @param[out] lwSciColorFmt LwSciBufAttrValColorFmt.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a lwSciColorFmt is NULL
 *      - @a lwColorFmt does not have corresponding LwSciBufAttrValColorFmt
 *
 */
LwSciError LwColorToLwSciColor(
    LwColorFormat lwColorFmt,
    LwSciBufAttrValColorFmt* lwSciColorFmt);
/**
 * @}
 */

#if defined(__cplusplus)
}
#endif // __cplusplus

#endif /* INCLUDED_LWSCIBUF_INTERNAL_H */
