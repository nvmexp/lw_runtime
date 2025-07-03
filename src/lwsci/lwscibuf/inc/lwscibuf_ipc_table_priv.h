/*
 * lwscibuf_ipc_table_priv.h
 *
 * Private Header for LwSciBuf IPC Table Unit
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_IPC_TABLE_PRIV_H
#define INCLUDED_LWSCIBUF_IPC_TABLE_PRIV_H

/**
 * \brief Defines an invalid index into IPC table.
 * This macro is used for initializing IPC table iterator.
 */
#define LWSCIBUF_ILWALID_IPCTABLE_IDX   0xFFFFFFFFU

/******************************************************
 *    All Structure Definitions                       *
 ******************************************************/

/**
 * \brief structure defining topology of the LwSciIpcEndpoint.
 */
typedef struct {
    /**
     * LwSciIpcTopoId defining SocId and VmId of the LwSciIpcEndpoint.
     */
    LwSciIpcTopoId topoId;
    /**
     * LwSciIpcEndpointVuid defining VM wide unique Id of the LwSciIpcEndpoint.
     */
    LwSciIpcEndpointVuid vuId;
} __attribute__((packed)) LwSciBufIpcTopoId;

/**
 * \brief Structure that defines an IPC Route.
 * An IPC Route contains two things:
 *   - List of LwSciIpc Endpoints that an attribute list has traversed.
 *   - Number of LwSciIpc Endpoints in the list.
 * LwSciBuf uses IPC Route to keep track of traversal path of the unreconciled
 * attribute lists. For example, if an unreconciled attribute list is
 * is transported from procA->procB->procC, where all these processes
 * have LwSciIpc channels created between them for IPC communication,
 * an IPC route contains the information to indicate that the attribute list is
 * originated from procA, transported to procB and then to procC.
 * IPCRoute will be used during reconciliation as well as export/import of some
 * of the keys eg: ActualPerm attribute Key which is modified during export/
 * import of a reconciled attribute list according to the permissions computed
 * for the IPCRoute. This way, it can be guranteed that a process will be given
 * appropriate permissions according to permissions that it requested.
 */
typedef struct LwSciBufIpcRouteRec {
    /**
     * Pointer to the list of LwSciBufIpcTopoId. List is represented as an
     * array.
     *  Valid values:
     *     - NULL indicates NULL IPC route. endpointCount should be 0 when
     *     this is NULL.
     *     - any valid non-NULL pointer, otherwise.
     */
    LwSciBufIpcTopoId* ipcEndpointList;
    /**
     * Indicates the number of endpoints stored in the ipcEndpointList.
     * This value will be greater than 0 only when ipcEndpointList is non-NULL.
     * Valid values: [0, SIZE_MAX].
     */
    size_t endpointCount;
} LwSciBufIpcRouteRecPriv;

/**
 * \brief Structure definition for attribute data of an IPC table entry.
 * Attribute data in an IPC table entry is managed as linked list of lwList
 * nodes.
 */
typedef struct {
    /**
     * Attribute key to be stored in this node.
     * Valid values:
     *    - All values of LwSciBufAttrKey type except lower and upper bounds.
     *    - All values of LwSciBufInternalAttrKey type except lower bound.
     */
    uint32_t key;
    /**
     * Length of the attribute value.
     * valid values: 1 to SIZE_MAX.
     */
    uint64_t len;
    /**
     * Pointer to attribute value
     * valid values: any non-NULL pointer.
     */
    void* value;
    /**
     * Pointer to next attribute data node in the linked list.
     * valid values: any non-NULL pointer returned by lwList APIs.
     */
    LWListRec listEntry;
} LwSciBufIpcTableAttrData;

/**
 * \brief Structure definition for an entry in the IPC table.
 * (Note: Refer to IPC table structure definition for info of the IPC table.)
 * An IPC table entry contains the following fields:
 * 1) IPC route
 * 2) Linked list head to attribute data.
 * 3) Size required to export the entry. This information is used later to
 * compute the size required to export a table.
 */
typedef struct {
    /**
     * IPC Route which acts as search key for the table.
     * Valid values: any non-NULL pointer returned as output parameter by
     * the IPC Route APIs.
     */
    LwSciBufIpcRoute* ipcRoute;
    /**
     * Linked list head to the attribute data.
     * Valid values: any non-NULL pointer returned by lwListInit function.
     */
    LWListRec ipcAttrEntryHead;
    /**
     * Size of the entry required for exporting the entry
     * Valid value: [8, SIZE_MAX]
     */
    size_t entryExportSize;
} LwSciBufIpcTableEntry;

/**
 * \brief Structure that defines an IPC Table.
 * An IPC Table is created during reconciliation and is stored as an attribute
 * in a reconciled attribute list. This table is created is to keep track of
 * some of the attributes on per IPC route basis from every unreconciled
 * attribute list participating in reconciliation. These attributes in the
 * IPC table are later used for computing the attribute value for that
 * process/thread, when the reconciled list is exported to a process/thread
 * that is in the path of any of the IPC routes in the IPC table.
 *
 * An IPC table contains a collection of entries. Each entry contains the
 * following basic information. (Note: Refer to Table entry structure for
 * detailed information).
 *     1) IPC route. This is the key to search an entry in the table.
 *     2) Linked list of attribute data. Each node of the list contains
 *     attribute key & value. Pointer to head node is stored in the table
 *     entry.
 */
struct LwSciBufIpcTableRec {
    /**
     * Pointer to collection of IPC table entries represented as an array.
     * Valid values: any valid non-NULL pointer.
     */
    LwSciBufIpcTableEntry* ipcTableEntryArr;
    /**
     * Indicates the maximum number of entries which the IPC Table can
     * hold. In case of dynamically allocated ipcTableEntryArr, this
     * indicates the number of entries for which the table is allocated for.
     * valid values: 1 to SIZE_MAX
     */
    size_t allocEntryCount;
    /**
     * Indicates the number of entries that contain valid entries.
     * valid values: 0 to SIZE_MAX and <= allocEntryCount
     */
    size_t validEntryCount;
};

/**
 * \brief Structure definition of the IPC table iterator.
 * An IPC table iterator is used to iterate through the IPC table for entries
 * that match the user specified critieria. Once the iterator indexes to an
 * entry in the table, that matches the criteria, user can access all the
 * information related to that entry using the IPC table management unit APIs.
 * The user can specify the following to define the criteria for iterating
 * through the IPC table:
 * 1) LwSciIpc endpoint that should match outer endpoint of the IPC route
 * of an entry.
 * 2) Flag indicating whether IPC endpoint should partly/fully match the IPC
 * route.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 */
struct LwSciBufIpcTableIterRec {
    /**
     * IPC table on which the Iterator has to iterate thru
     * Valid values: any non-NULL value returned by IPC table APIs.
     */
    const LwSciBufIpcTable* ipcTable;

    LwSciBufIpcRoute* ipcRoute;
    /**
     * Index of the current entry of the IPC table that matches
     * the search criteria. This is intialized to LWSCIBUF_ILWALID_IPCTABLE_IDX
     * when iterator is created and the value changes after every call to
     * LwSciBufIpcTableIterNext API call when an new entry that matches
     * the specified critieria is satisfied.
     * Valid values: [0, SIZE_MAX] and <= validEntryCount of the ipcTable.
     */
    size_t lwrrMatchEntryIdx;
};

/*
 * \brief Structure definition of header for an IPC table entry in the
 * export descriptor of IPC Table. Refer to IPC table export header
 * for more details of the usage of this header.
 *
 * Synchronization: Access to an instance of this datatype must be
 * externally synchronized
 */
typedef struct {
    /**
     * Size of the entry in bytes.
     * valid values: 0 to SIZE_MAX
     */
    uint64_t entrySize;
    /**
     * Number of keys of this entr that are exported in the descriptor
     * valid values: 0 to SIZE_MAX
     */
    uint64_t keyCount;
    /**
     * Indicates the end of the header. Doesn't carry any value.
     * Sized due to MISRA violation fix for rule 18.7
     * Do not rely on the size of desc
     */
    uint8_t desc[1];
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
} __attribute__((packed)) LwSciBufIpcTableEntryExportHeader;

/**
 * \brief Structure defintion of header added in the export descriptor of the
 * IPC table. When an IPC table is exported, all the information in the IPC
 * table is serialized as a set of bytes into the export descriptor and the
 * header contains a summary of the key information useful for de-serializing
 * the descriptor and recreating an IPC table that was exported by the exporting
 * application.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 */
typedef struct {
    /**
     * Indicates the number of IPC table entries that are contained in this
     * descriptor.
     * valid values: 1 to SIZE_MAX
     */
    uint64_t entryCount;
    /**
     * Indicates the size of the IPC endpoint in the exporting application.
     * This is useful to verify if the software versions of the exporting
     * and importing applications are compatible and the LwSciIpcEndpoint size
     * is same on the exporting and importing applications.
     * valid values: 0 to SIZE_MAX
     */
    uint64_t ipcEndpointSize;
    /**
     * Size of the descriptor computed by the exporting application.
     * This is useful to compare whether the descriptor received by the
     * importing is complete or incomplete.
     * valid values: 0 to SIZE_MAX
     */
    uint64_t totalSize;
    /**
     * Indicates the end of the header. Doesn't contain any value.
     * Sized due to MISRA violation fix for rule 18.7
     * Do not rely on the size of entryStart
     */
    LwSciBufIpcTableEntryExportHeader entryStart[1];
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
} __attribute__((packed)) LwSciBufIpcTableExportHeader;

#endif     //INCLUDED_LWSCIBUF_IPC_TABLE_PRIV_H
