/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_PRIV_H
#define INCLUDED_LWSCIBUF_PRIV_H

#include "lwscibuf_internal.h"
#include "lwscicommon_os.h"

/**
 * @brief Enum denotes the type of an attribute key.
 *
 * @implements{18842205}
 */
typedef enum {
    LwSciBufAttrKeyType_Public = LW_SCI_BUF_ATTR_KEY_TYPE_PUBLIC,
    LwSciBufAttrKeyType_Internal = LW_SCI_BUF_ATTR_KEY_TYPE_INTERNAL,
    LwSciBufAttrKeyType_UMDPrivate =
        LW_SCI_BUF_ATTR_KEY_TYPE_INTERNALAPP_PRIVATE,
    LwSciBufAttrKeyType_Private,
    LwSciBufAttrKeyType_Transport,
    LwSciBufAttrKeyType_MaxValid,
} LwSciBufAttrKeyType;

#define LW_SCI_BUF_DATATYPE_BIT_START LW_SCI_BUF_ATTRKEY_BIT_COUNT

#define LW_SCI_BUF_KEYTYPE_BIT_MASK  \
        (((uint32_t)1U << LW_SCI_BUF_KEYTYPE_BIT_COUNT) - 1U)

#define LW_SCI_BUF_DATATYPE_BIT_MASK \
        (((uint32_t)1U << LW_SCI_BUF_DATATYPE_BIT_COUNT) - 1U)

#define LW_SCI_BUF_ATTRKEY_BIT_MASK \
        (((uint32_t)1U << LW_SCI_BUF_ATTRKEY_BIT_COUNT) - 1U)

/**
 * @brief Global Constant to decode LwSciBufAttrKeyType from given key.
 */
#define LW_SCI_BUF_DECODE_KEYTYPE(key) \
        (((uint32_t)(key) >> LW_SCI_BUF_KEYTYPE_BIT_START) & LW_SCI_BUF_KEYTYPE_BIT_MASK)

/**
 * @brief Global Constant to decode LwSciBufType from given key.
 */
#define LW_SCI_BUF_DECODE_DATATYPE(key) \
        (((uint32_t)(key) >> LW_SCI_BUF_DATATYPE_BIT_START) & LW_SCI_BUF_DATATYPE_BIT_MASK)

/**
 * @brief Global Constant to decode attribute key value from given key.
 */
#define LW_SCI_BUF_DECODE_ATTRKEY(key) ((uint32_t)(key) & LW_SCI_BUF_ATTRKEY_BIT_MASK)

#define LW_SCI_BUF_ATTRKEY_ENCODE(keytype, datatype, key) \
        (((uint32_t)(keytype) << LW_SCI_BUF_KEYTYPE_BIT_START) | \
          ((uint32_t)(datatype) << LW_SCI_BUF_ATTRKEY_BIT_COUNT) | (key))

/**
 * @brief Global Constant to specify the starting value of
 *        LwSciBufPrivateAttrKey.
 */
#define LW_SCI_BUF_PRIVATE_KEY_START \
         (LwSciBufAttrKeyType_Private << LW_SCI_BUF_KEYTYPE_BIT_START)

/**
 * @brief Enum which describes the LwSciBuf private attribute keys specifying
 * values computed after reconciliation which are used privately by LwSciBuf.
 *
 * @implements{18842211}
 */
typedef enum {
    /** Invalid key. Needed for lower bound check on LwSciBuf private attribute
     * keys.
     * NOTE: internal attribute keys occupy space of (128k - 1). Thus,
     * private keys should start from 128K
     *
     * Value: None
     */
    LwSciBufPrivateAttrKey_LowerBound =  LW_SCI_BUF_PRIVATE_KEY_START,

    /**
     * Total buffer size callwlated after reconciliation of attributes. The
     * value of this key is copied from any one of the output attribute keys
     * specifying buffer size corresponding to LwSciBufTypes ilwolved in
     * reconciliation.
     *
     * Type: Output attribute
     * Value: uint64_t
     */
    LwSciBufPrivateAttrKey_Size,

    /**
     * Alignment of buffer callwlated after reconciliation of attributes
     * Type: Output attribute. The value of this key is copied from any one of
     * the output attribute keys specifying buffer alignment corresponding to
     * LwSciBufTypes ilwolved in reconciliation. The value is always power
     * of 2
     *
     * Type: Output attribute
     * Value: uint64_t
     */
    LwSciBufPrivateAttrKey_Alignment,

    /**
     * Heap type where buffer needs to be allocated.
     */
#if (LW_IS_SAFETY == 0)
    /**
     * If value of LwSciBufPrivateAttrKey_MemDomain is LwSciBufMemDomain_Cvsram,
     * this attribute is set to LwSciBufHeapType_CvsRam.
     */
#endif
    /**
     * If value of LwSciBufPrivateAttrKey_MemDomain is LwSciBufMemDomain_Sysmem,
     */
#if (LW_IS_SAFETY == 0)
    /**
     * Set this attribute to LwSciBufHeapType_ExternalCarveout if any of the
     * ISO engines (LwSciBufHwEngine_VI, LwSciBufHwEngine_Display) are specified
     * in the LwSciBufInternalGeneralAttrKey_EngineArray.
     */
#else
    /**
     * Set this attribute to LwSciBufHeapType_IVC if LwSciBufHwEngine_VI is
     * specified in the LwSciBufInternalGeneralAttrKey_EngineArray.
     * Set this attribute to LwSciBufHeapType_ExternalCarveout if
     * LwSciBufInternalGeneralAttrKey_EngineArray contains
     * LwSciBufHwEngine_Display and doesn't contain LwSciBufHwEngine_VI.
     */
#endif
    /**
     * Set this attribute to LwSciBufHeapType_IOMMU in all other cases.
     *
     * Type: Output attribute
     * Value: LwSciBufHeapType
     */
    LwSciBufPrivateAttrKey_HeapType,

    /**
     * Memory domain from which buffer needs to be allocated
     * Type: Output attribute
     * Value: LwSciBufMemDomain
     */
    LwSciBufPrivateAttrKey_MemDomain,

    /** Valid for unreconciled lists only. An LwSciBufAttrValTopoId specifying
     * which VM in which SoC created the containing attribute list slot.
     * This key is set by LwSciBufAttrListCreate() and is intended never to be
     * modified thereafter.
     *
     * Value: LwSciBufAttrValTopoId
     */
    LwSciBufPrivateAttrKey_OriginTopoId,

    /** An array of LwSciBufAttrValTopoId structures specifying all the VMs
     * (possibly excluding the VM containing this LwSciBufAttrList) that need
     * to be able to import a buffer allocated using this LwSciBufAttrList.
     * LwSciBufExportAttrList() appends the topology ID of the caller to the
     * array of LwSciBufAttrValTopoId structures residing in slot index zero of
     * the exported attribute list descriptor (adding the key if it is not
     * already present). LwSciBufReconcileAttrLists() ensures that the
     * reconciled list’s array of LwSciBufAttrValTopoId structures contains all
     * the topology IDs in all the arrays of LwSciBufAttrValTopoId structures
     * in the slots of the unreconciled lists.
     *
     * Value: LwSciBufAttrValTopoId[]
     */
    LwSciBufPrivateAttrKey_OtherTopoIdArray,

    /**
     * ipc route defined by LwSciIpcEndPoints.
     *
     * Type: Output attribute
     * Value: LwSciBufIpcRoute*
     */
    LwSciBufPrivateAttrKey_SciIpcRoute,

    /**
     * ipcEndpoint table, it stores information about access permission,
     *  need CPU access
     * Type: Output
     * Value: LwSciBufIpcTable*
     */
    LwSciBufPrivateAttrKey_IPCTable,

    /**
     * This key represents conflict key number.
     * Type: Output
     */
    LwSciBufPrivateAttrKey_ConflictKey,
} LwSciBufPrivateAttrKey;

/**
 * @brief Enum denotes the heap types supported by LwSciBuf.
 *
 * @implements{18842202}
 */
typedef enum {
    LwSciBufHeapType_IOMMU,
    LwSciBufHeapType_ExternalCarveout,
    LwSciBufHeapType_IVC,
    LwSciBufHeapType_VidMem,
    LwSciBufHeapType_CvsRam,
    LwSciBufHeapType_Resman, // not a actual heaptype, placeholder for resman backend
    LwSciBufHeapType_Ilwalid,
} LwSciBufHeapType;

/**
 * @brief Enum denoting type of GPUs.
 *
 * @implements{}
 */
typedef enum {
    LwSciBufGpuType_iGPU = 0U,
    LwSciBufGpuType_dGPU = 1U,
} LwSciBufGpuType;

/**
 * @brief This structure contains SoC ID and VM ID which describes which all
 * entities are consumer of the buffer so we can relay back the buffer info.
 *
 * @implements{18842199}
 */
typedef struct {
    /** SoC ID information */
    uint32_t SocId;
    /** VM ID information */
    uint32_t VmId;
} LwSciBufAttrValTopoId;

/**
 * @brief This structure defines a key/value pair used to get or set the
 * LwSciBufPrivateAttrKey(s) and their corresponding values from or to
 * LwSciBufAttrList.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * @implements{18842208}
 */
typedef struct {
    /** LwSciBufPrivateAttrKey for which value needs to be set/retrieved.
     * This member is initialized to any one of the LwSciBufPrivateAttrKey(s)
     * other than LwSciBufPrivateAttrKey_LowerBound */
    LwSciBufPrivateAttrKey key;

    /** Length of the value in bytes */
    size_t len;

    /** Pointer to the value corresponding to the attribute.
     * If the value is an array, the pointer points to the first element. */
    const void* value;
} LwSciBufPrivateAttrKeyValuePair;

#endif /* INCLUDED_LWSCIBUF_PRIV_H */
