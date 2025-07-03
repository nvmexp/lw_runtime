/*
 * lwscibuf_transport_priv.h
 *
 * Transport Layer header file for LwSciBuf
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_TRANSPORT_H
#define INCLUDED_LWSCIBUF_ATTR_TRANSPORT_H

#if defined(__x86_64__)
#include "lwscibuf_transport_priv_x86.h"
#else
#include "lwscibuf_transport_priv_tegra.h"
#endif

/**
 * Macro defining starting encoded value of keys having transport keytype
 */
#define LW_SCI_BUF_TRANSPORT_KEY_START \
         (LwSciBufAttrKeyType_Transport << LW_SCI_BUF_KEYTYPE_BIT_START)

/**
 * Magic ID used to validate export descriptor received over IPC
 */
#define LW_SCI_BUF_TRANSPORT_MAGIC  0xF00DCAFEU

/**
 * Number of transport keys in the serialized attribute lists' export descriptor
 */
#define LW_SCI_BUF_NUM_ATTRLIST_HEADER_TRANSPORT_KEYS  2U

/**
 * Macro to get LwSciBuf major version number from LwSciBuf version
 *
 * \param[in] version LwSciBuf version
 */
#define LW_SCI_BUF_VERSION_MAJOR(version) ((uint32_t)((version) >> 32))

/**
 * Macro to get LwSciBuf version composed of major and minor number
 */
#define LW_SCI_BUF_VERSION \
       (((uint64_t)LwSciBufMajorVersion << 32) | LwSciBufMinorVersion)

/**
 * Index of the transport key relative to the first transport
 * key (LwSciBufTransportAttrKey_LowerBound)
 *
 * \param[in] key transport key for which index relative to the first transport
 *            key needs to be callwlated
 */
#define LW_SCI_BUF_TRANSKEY_IDX(key) \
        ((uint32_t)(key) - (uint32_t)LwSciBufTransportAttrKey_LowerBound)

/**
 * check validity of transport keys
 */
#define LW_SCI_BUF_VALID_TRANSPORT_KEYS(key) \
    ((((uint32_t)(key)) < (uint32_t)LwSciBufTransportAttrKey_UpperBound) && \
     (((uint32_t)(key)) > (uint32_t)LwSciBufTransportAttrKey_LowerBound))

/**
 * Total number of valid transport keys
 */
#define LW_SCI_BUF_NUM_TRANSPORT_KEYS \
        ((uint32_t)LwSciBufTransportAttrKey_UpperBound - \
         (uint32_t)LwSciBufTransportAttrKey_LowerBound)

/**
 * enum defining special attribute keys added for LwSciBuf transport layer
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 2_3), "LwSciBuf-ADV-MISRAC2012-002")
typedef enum {
/**
 * Key defining starting encoded value of transport keytype
 * Datatype: None
 */
    LwSciBufTransportAttrKey_LowerBound =  LW_SCI_BUF_TRANSPORT_KEY_START,

/**
 * Number of slots of the attribute list to be exported
 * Datatype: uint64_t (alternate for size_t for transport).
 */
    LwSciBufTransportAttrKey_AttrListSlotCount,

/**
 * Transport Key to specify if the exported attribute list is reconciled or not.
 * Datatype: uint8_t
 */
    LwSciBufTransportAttrKey_AttrListReconciledFlag,

/**
 * slot number from which the attributes need to be read.
 * Datatype: uint64_t (alternate for size_t for transport).
 */
    LwSciBufTransportAttrKey_AttrListSlotIndex,

/**
 * LwSciBufObj related data needed to recreate the object from export
 * descriptor on importing side.
 * Datatype: LwSciBufObjExportDescPriv
 */
    LwSciBufTransportAttrKey_ObjDesc,

/**
 * Number of valid transport keys
 */
    LwSciBufTransportAttrKey_UpperBound,
} LwSciBufTransportAttrKey;
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 2_3))

/**
 * Function callback for general attribute keys when colwerting attribute
 * list(s) into export descriptor while exporting
 */
typedef LwSciError (*LwSciBufAttrKeyExportCb)(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm exportAPIperms,
    void** exportDesc,
    uint64_t* len);

/**
 * Function callback for general attribute keys when colwerting export
 * descriptor into attribute list(s) while importing
 */
typedef LwSciError (*LwSciBufAttrKeyImportCb)(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    const void* exportDesc,
    uint64_t len);

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * Abstraction for PCIe vs NPM token handles. We will add NPM token handle
 * when it is supported. Lwrrently, only PCIe is supported.
 */
typedef union {
   LwSciC2cPcieAuthToken pcieAuthToken;
} LwSciC2cInterfaceAuthToken;
#endif

/**
 * LwSciBufObj related data needed to recreate the object from export
 * descriptor on importing side.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 */
typedef struct {
    /**
     * The offset within the memory represented by memHandle of the memory
     * represented by the LwSciBufObj
     */
    uint64_t offset;

    /** The length of the memory represented by the LwSciBufObj */
    uint64_t bufLen;

    /** access permissions with which LwSciBufObj is exported */
    LwSciBufAttrValAccessPerm perms;

    /** platform specific data associated with LwSciBufObj */
    LwSciBufObjExportPlatformDescPriv platformDesc;

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    /** C2c token if the communication boundary is Soc. */
    LwSciC2cInterfaceAuthToken c2cToken;
#endif
} __attribute__((packed)) LwSciBufObjExportDescPriv;

/**
 * Descriptor defining various properties of transport keys
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 */
typedef struct {
    /** size of the key value */
    size_t keysize;
} LwSciBufTransportKeyDesc;

/**
 * Structure defining export/import callbacks for general attribute keys
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 */
typedef struct {
    /**
     * Export callback for general attribute key
     * Used when exporting attribute list(s)
     */
    LwSciBufAttrKeyExportCb exportCallback;

    /**
     * Import callback for general attribute key
     * Used when importing attribute list(s)
     */
    LwSciBufAttrKeyImportCb importCallback;
} LwSciBufAttrKeyTransportDesc;

/* Structure representing data being passed into a transport FSM
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 */
typedef struct {
    uint32_t key;
    size_t length;
    const void* value;
} LwSciBufSerializedKeyValPair;

/*
 * Structure representing FSM context data
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 */
typedef struct {
    /* LwSciBufModule to associate with the LwSciBufAttrList being imported.
     *
     * LwSciBufAttrList descriptor: Used
     * LwSciBufObj descriptor: Ununsed
     * LwSciBufAttrList + LwSciBufObj descriptor: Used
     */
    LwSciBufModule module;

    /* When importing LwSciBufAttrList descriptors, outputAttrList corresponds
     * to the LwSciBufAttrList being imported.
     *
     * When importing LwSciBufObj descriptors, outputAttrList corresponds to
     * the reconciled LwSciBufAttrList to associate with the LwSciBufObj.
     *
     * LwSciBufAttrList descriptor: Used
     * LwSciBufObj descriptor: Used
     * LwSciBufAttrList + LwSciBufObj descriptor: Used
     */
    LwSciBufAttrList* outputAttrList;

    /* The LwSciIpcEndpoint we are importing through
     *
     * LwSciBufAttrList descriptor: Used
     * LwSciBufObj descriptor: Used
     * LwSciBufAttrList + LwSciBufObj descriptor: Used
     */
    LwSciIpcEndpoint ipcEndpoint;

    /* Set by the FSM to determine whether a reconciled or unreconciled
     * LwSciBufAttrList is being imported.
     *
     * LwSciBufAttrList descriptor: Used
     * LwSciBufObj descriptor: Unused
     * LwSciBufAttrList + LwSciBufObj descriptor: Used
     */
    bool importingReconciledAttr;

    /* The expected current slot index for the LwSciBufAttrList being
     * deserialized.
     *
     * LwSciBufAttrList descriptor: Used
     * LwSciBufObj descriptor: Unused
     * LwSciBufAttrList + LwSciBufObj descriptor: Used
     */
    size_t slotIndex;

    /* The expected permissions associated.
     *
     * LwSciBufAttrList descriptor: Unused
     * LwSciBufObj descriptor: Used
     * LwSciBufAttrList + LwSciBufObj descriptor: Used
     */
    LwSciBufAttrValAccessPerm perms;

    /* The output LwSciBufObj. Only used when importing LwSciBufObj descriptors.
     *
     * LwSciBufAttrList descriptor: Unused
     * LwSciBufObj descriptor: Used
     * LwSciBufAttrList + LwSciBufObj descriptor: Used
     */
    LwSciBufObj* bufObj;
} LwSciBufTransportFsmContext;

/**
 * @defgroup lwscibuf_transport_api LwSciBuf APIs
 * List of APIs to transport LwSciBuf buffers and attribute list objects across
 * various communication boundaries that interact using LwSciIpc.
 * @{
 */

/**
 * 1) Retrieves the LwSciBufAttrList from the bufObj using
 *    LwSciBufObjGetAttrList().
 *
 * 2) Gets the slot count for that LwSciBufAttrList through
 * LwSciBufAttrListGetSlotCount().
 *
 * 3) Iterates through the LwSciBufAttrList to identify the total size of the
 * attribute values and number of attribute keys.
 *
 * 4) Allocates the memory for the transport buffers of type
 * LwSciCommonTransportBuf* based on the memory requirements of attributes and
 * size of LwSciBufObjExportPlatformDescPriv using
 * LwSciCommonTransportAllocTxBufferForKeys().
 *
 * 5) Colwerts bufObj to LwSciBufRmHandle type using LwSciBufObjGetMemHandle
 * and gets the LwSciBufObjExportPlatformDescPriv for it using
 * LwSciBufTransportGetPlatformDesc().
 *
 * 6) Copies all the attribute keys and values, along with slot index and
 * additional header information if required and LwSciBufObjExportDescPriv
 * to LwSciCommonTransportBuf*.
 *
 * 7) Finally export the LwSciCommonTransportBuf* to attrListAndObjDesc using
 * LwSciCommonTransportPrepareBufferForTx().
 *
 * The attrListAndObjDesc can be freed using
 * LwSciBufAttrListAndObjFreeDesc.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the only modification to
 *        the LwSciBufAttrList after reconciilation is during LwSciBufObj
 *        allocation. In addition, LwSciBufGeneralAttrKey_ActualPerm is the
 *        only such attribute key, and the exported value is not read from the
 *        LwSciBufAttrList, but rather from the IPC table to compute the
 *        importing peer's list permission (so there is no data-dependency)
 *      - Reads only occur from immutable data since the only modification to
 *        the LwSciBufObj after allocation is to set LwMedia flags on the
 *        reference instead of the object which are not exported (so there is
 *        no data-dependency).
 *
 * \implements{18843171}
 *
 * \fn LwSciError LwSciBufIpcExportAttrListAndObj(
 *    LwSciBufObj bufObj,
 *    LwSciBufAttrValAccessPerm permissions,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    void** attrListAndObjDesc,
 *    size_t* attrListAndObjDescSize);
 */

/**
 * Gets the LwSciCommonTransportBuf* from the attrListAndObjDesc using
 * LwSciCommonTransportGetRxBufferAndParams. Iterates through the received
 * LwSciCommonTransportBuf* and extracts the attribute's key-values,
 * LwSciBufObjExportPlatformDescPriv and header information.
 *
 * RM mem handle is obtained from the LwSciBufObjExportPlatformDescPriv using
 * LwSciBufTransportGetMemHandle. The LwSciBufObj is then obtained from the RM
 * handle using LwSciBufTransportCreateObjFromMemHandle.
 *
 * The received reconciled LwSciBufAttrList is also validated against
 * un-reconciled attrList[] by LwSciBufAttrListValidateReconciled API.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the input LwSciBufModule is provided via
 *        LwSciBufAttrListCreateMultiSlot()
 *      - Conlwrrent access to the LwSciBufModule associated with the imported
 *        LwSciBufAttrList is provided via LwSciBufAttrListClone() if the input
 *        minPermissions parameter is less than the exported permissions
 *        associated with the descriptor
 *      - Conlwrrent access to the unreconciled LwSciBufAttrList(s) is handled
 *        via LwSciBufAttrListValidateReconciled()
 *
 * \implements{17827290}
 *
 * \fn LwSciError LwSciBufIpcImportAttrListAndObj(
 *    LwSciBufModule module,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    const void* attrListAndObjDesc,
 *    size_t attrListAndObjDescSize,
 *    const LwSciBufAttrList attrList[],
 *    size_t count,
 *    LwSciBufAttrValAccessPerm minPermissions,
 *    int64_t timeoutUs,
 *   LwSciBufObj* bufObj);
 */

/**
 * Uses LwSciCommonFree() to free the attrListAndObjDescBuf.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same input attrListAndObjDescBuf is not
 *        freed by multiple threads at the same time
 *      - The user must ensure that the same input attrListAndObjDescBuf is not
 *        being used when freeing
 *
 * \implements{18843195}
 *
 * \fn void LwSciBufAttrListAndObjFreeDesc(
 *    void* attrListAndObjDescBuf);
 */

/**
 * Colwerts bufObj to LwSciBufRmHandle type using LwSciBufObjGetMemHandle
 * and gets the LwSciBufObjExportPlatformDescPriv for it using
 * LwSciBufTransportGetPlatformDesc.
 *
 * Allocates the memory for the transport buffers of type
 * LwSciCommonTransportBuf* and size of LwSciBufObjExportDescPriv using
 * LwSciCommonTransportAllocTxBufferForKeys and copies
 * LwSciBufObjExportDescPriv to it.
 *
 * Finally export the LwSciCommonTransportBuf* using
 * LwSciCommonTransportPrepareBufferForTx.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the only modification to
 *        the LwSciBufObj after allocation is to set LwMedia flags on the
 *        reference instead of the object which are not exported (so there is
 *        no data-dependency).
 *
 * \implements{18843177}
 *
 * \fn LwSciError LwSciBufObjIpcExport(
 *    LwSciBufObj bufObj,
 *    LwSciBufAttrValAccessPerm accPerm,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    LwSciBufObjIpcExportDescriptor* exportData);
 */

/**
 * Gets the LwSciCommonTransportBuf* from the desc using
 * LwSciCommonTransportGetRxBufferAndParams. Extracts the
 * LwSciBufObjExportPlatformDescPriv data from LwSciCommonTransportBuf*.
 *
 * LwSciBufRmHandle is obtained from the LwSciBufObjExportPlatformDescPriv
 * using LwSciBufTransportGetMemHandle. The LwSciBufObj is then obtained from
 * the LwSciBufRmHandle using LwSciBufTransportCreateObjFromMemHandle.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The LwSciBufGeneralAttrKey_ActualPerm key is only ever modified after
 *        reconciliation in LwSciBufObjAlloc(). But LwSciBufObjAlloc() is not
 *        thread-safe if multiple APIs are using the same reconciled
 *        LwSciBufAttrList, so conlwrrent modification and reads leading to a
 *        non-thread-safe behavior is not possible.
 *      - Conlwrrent access to the LwSciBufModule associated with the input
 *        reconciledAttrList parameter is provided via LwSciBufAttrListClone()
 *        if the input minPermissions parameter is less than the exported
 *        permissions associated with the descriptor
 *
 * \implements{18843180}
 *
 * \fn LwSciError LwSciBufObjIpcImport(
 *    LwSciIpcEndpoint ipcEndpoint,
 *    const LwSciBufObjIpcExportDescriptor* desc,
 *    LwSciBufAttrList reconciledAttrList,
 *    LwSciBufAttrValAccessPerm minPermissions,
 *    int64_t timeoutUs,
 *    LwSciBufObj* bufObj);
 */

/**
 * Iterates through each of the LwSciBufAttrList in
 * unreconciledAttrListArray[] to identify the total size of the attribute
 * values and key-value pairs. Allocates the memory for the transport buffers
 * of type LwSciCommonTransportBuf* based on the total size identified, along
 * with size of headerinfo if any, using
 * LwSciCommonTransportAllocTxBufferForKeys and copies all
 * the attribute keys and values to it along with any header info.
 *
 * Finally export the LwSciCommonTransportBuf* using
 * LwSciCommonTransportPrepareBufferForTx.
 *
 * The descBuf can be freed using LwSciBufAttrListFreeDesc.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization:
 *      - Locks are taken on each LwSciBufAttrList in unreconciledAttrListArray
 *        to serialize reads using LwSciBufAttrListsLock()
 *      - Locks are held for the duration of any reads from all of the
 *        LwSciBufAttrList(s) in unreconciledAttrListArray
 *      - Locks are released when all operations on the LwSciBufAttrList(s) in
 *        unreconciledAttrListArray are complete using
 *        LwSciBufAttrListsUnlock()
 *
 * \implements{18843183}
 *
 * \fn LwSciError LwSciBufAttrListIpcExportUnreconciled(
 *    const LwSciBufAttrList unreconciledAttrListArray[],
 *    size_t unreconciledAttrListCount,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    void** descBuf,
 *    size_t* descLen);
 */

/**
 * Iterates through the reconciled LwSciBufAttrList to
 * identify the total size of the attribute values and key-value pairs.
 * Allocates the memory for the transport buffers of type
 * LwSciCommonTransportBuf* based on the total size identified, along with size
 * of headerinfo, using LwSciCommonTransportAllocTxBufferForKeys and copies all
 * the attribute keys and values to it along with any header info.
 *
 * Finally export the LwSciCommonTransportBuf* using
 * LwSciCommonTransportPrepareBufferForTx.
 *
 * The descBuf can be freed using LwSciBufAttrListFreeDesc.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The LwSciBufGeneralAttrKey_ActualPerm key is only ever modified after
 *        reconciliation in LwSciBufObjAlloc(). But LwSciBufObjAlloc() is not
 *        thread-safe if multiple APIs are using the same reconciled
 *        LwSciBufAttrList, so conlwrrent modification and reads leading to a
 *        non-thread-safe behavior is not possible.
 *
 * \implements{18843189}
 *
 * \fn LwSciError LwSciBufAttrListIpcExportReconciled(
 *    LwSciBufAttrList reconciledAttrList,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    void** descBuf,
 *    size_t* descLen);
 */


/**
 * Gets the LwSciCommonTransportBuf* from the descBuf using
 * LwSciCommonTransportGetRxBufferAndParams. Iterates through the received
 * LwSciCommonTransportBuf* and extracts the attribute's key-values and header
 * information.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the input LwSciBufModule is provided via
 *        LwSciBufAttrListCreateMultiSlot()
 *
 * \implements{17827302}
 *
 * \fn LwSciError LwSciBufAttrListIpcImportUnreconciled(
 *    LwSciBufModule module,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    const void* descBuf,
 *    size_t descLen,
 *    LwSciBufAttrList* importedUnreconciledAttrList);
 */

/**
 * Gets the LwSciCommonTransportBuf* from the descBuf using
 * LwSciCommonTransportGetRxBufferAndParams. Iterates through the received
 * LwSciCommonTransportBuf* and extracts the attribute's key-values and header
 * information.
 *
 * The received reconciled LwSciBufAttrList is also validated against
 * un-reconciled inputUnreconciledAttrListArray[] by
 * LwSciBufAttrListValidateReconciled API.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the input LwSciBufModule is provided via
 *        LwSciBufAttrListCreateMultiSlot()
 *      - Conlwrrent access to the input LwSciBufAttrList(s) in
 *        inputUnreconciledAttrListArray is provided via
 *        LwSciBufAttrListValidateReconciled()
 *
 * \implements{17827308}
 *
 * \fn LwSciError LwSciBufAttrListIpcImportReconciled(
 *    LwSciBufModule module,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    const void* descBuf,
 *    size_t descLen,
 *    const LwSciBufAttrList inputUnreconciledAttrListArray[],
 *    size_t inputUnreconciledAttrListCount,
 *    LwSciBufAttrList* importedReconciledAttrList);
 */


/**
 * Uses LwSciCommonFree() to free the descBuf.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same input descBuf is not freed by
 *        multiple threads at the same time
 *      - The user must ensure that the same input descBuf is not being used
 *        when freeing
 *
 * \implements{18843198}
 *
 * \fn void LwSciBufAttrListFreeDesc(
 *    void* descBuf);
 */

/**
  * @}
  */

#endif /* INCLUDED_LWSCIBUF_ATTR_TRANSPORT_H */
