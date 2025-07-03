//! \file
//! \brief LwSciStream SafeConnection base class.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef SAFECONNECTION_H
#define SAFECONNECTION_H
#include <cstdint>
#include <utility>
#include <cassert>
#include <atomic>
#include <cmath>
#include <vector>
#include <functional>
#include <array>
#include <unordered_map>
#include <memory>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "endinfo.h"
#include "elements.h"
#include "packet.h"

namespace LwSciStream {
/**
 * @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
 * Generic statements applicable for LwSciStream interfaces.
 * @{
 */

/**
 * \page lwscistream_page_blanket_statements LwSciStream blanket statements
 * \section lwscistream_downstream_interaction Interaction with destination block
 * - When an instance of any block type which supports downstream connections needs to
 *   interact with one of its destination block instances, it creates a SafeConnection::Access
 *   object from the corresponding destination SafeConnection by calling Block::getDst()
 *   interface, then it calls the required DstBlockInterface of the destination
 *   block by calling the corresponding BlockWrap interface of the SafeConnection::Access
 *   object. In this case, the SafeConnection::Access object is used in a transitory
 *   fashion. Callers create it, access the interface(s) they need and destroy it,
 *   No allocations are performed in this process. If the smart pointer in BlockWrap
 *   is NULL, either because the destination SafeConnection is not yet completed by
 *   calling SafeConnection::connComplete() interface or cancelled by calling
 *   SafeConnection::connCancel() interface or disconnected by calling
 *   SafeConnection::disconnect() interface, call to any BlockWrap interface
 *   (except the reset() interfaces) of the SafeConnection::Access object created
 *   from the destination SafeConnection will be a no operation.
 *
 * \section lwscistream_upstream_interaction Interaction with source block
 * - When an instance of any block type which supports upstream connection needs to
 *   interact with its source block instance, it creates a SafeConnection::Access
 *   object from the corresponding source SafeConnection by calling Block::getSrc()
 *   interface, then it calls the required SrcBlockInterface of the source
 *   block by calling the corresponding BlockWrap interface of the SafeConnection::Access
 *   object. In this case, the SafeConnection::Access object is used in a transitory
 *   fashion. Callers create it, access the interface(s) they need and destroy it,
 *   No allocations are performed in this process. If the smart pointer in BlockWrap
 *   is NULL, either because the source SafeConnection is not yet completed by calling
 *   SafeConnection::connComplete() interface or cancelled by calling
 *   SafeConnection::connCancel() interface or disconnected by calling
 *   SafeConnection::disconnect() interface, call to any BlockWrap interface
 *   (except the reset() interfaces) of the SafeConnection::Access object created
 *   from the source SafeConnection will be a no operation.
 */

/**
 * @}
 */

// We could perhaps do without the template specialization here, and use
// a generic connection that provided all block interfaces. Specialization
// safeguards against accidental use of a src connection as a dst, and vice
// versa. If we rely on the caller to use things properly, then we could
// simplify these classes.

//! \brief Wrapper for SrcBlockInterface or DstBlockInterface smart pointers
//!  of the connected block along with an associated connection index used to
//!  avoid pointer and connection index tracking in the caller. The generic
//!  version of this template class is forbidden to be used by the missing
//!  constructors/destructors. Instead it is specialized by SrcBlockInterface
//!  and DstBlockInterface versions which support separate interfaces.
//!
//! \tparam T: SrcBlockInterface or DstBlockInterface
//!
//! \implements{18723255}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A14_1_1), "LwSciStream-ADV-AUTOSARC++14-002")
template<typename T>
class BlockWrap
{
protected:
    //! \brief Move constructor is needed by subclass.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    BlockWrap(BlockWrap<T>&& ref) noexcept = default;
private:
    //! \brief This generic template cannot be used
    BlockWrap(void) noexcept                                    = delete;
    BlockWrap(const BlockWrap<T>& ref) noexcept                 = delete;
    ~BlockWrap(void) noexcept                                   = delete;
    BlockWrap<T>& operator=(const BlockWrap<T>& ref) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    BlockWrap<T>& operator=(BlockWrap<T>&& ref) & noexcept      = delete;
};

//! \brief Specialized version of LwSciStream::BlockWrap< T > for SrcBlockInterface.
//!  It is a wrapper for SrcBlockInterface smart pointer with an associated connection
//!  index. It exposes interfaces which checks the smart pointer to SrcBlockInterface
//!  instance is not NULL and calls the appropriate interface of the
//!  SrcBlockInterface instance along with the connection index.
//!
//! \implements{18723276}
template<>
class BlockWrap<SrcBlockInterface>
{
public:
    //! Allows for deletion of objects through BlockWrap interface
    //! The destructor is made public only because VectorCast needs
    //! to access it when instrumenting the code.
    virtual ~BlockWrap(void) noexcept = default;
protected:
    //! The default empty constructor can be used, but the
    //! class cannot be directly instantiated.
    BlockWrap(void) noexcept = default;

    //! \brief Move constructor is needed by subclass.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    BlockWrap(BlockWrap<SrcBlockInterface>&& ref) noexcept = default;
private:
    //! Other constructors are not allowed
    BlockWrap(const BlockWrap<SrcBlockInterface>& ref) noexcept   = delete;
    BlockWrap<SrcBlockInterface>&
    operator=(const BlockWrap<SrcBlockInterface>& ref) & noexcept = delete;
    BlockWrap<SrcBlockInterface>&
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    operator=(BlockWrap<SrcBlockInterface>&& ref) & noexcept      = delete;

protected:
    // Child classes can fill in or clear the fields

    //! \brief Resets the smart pointer to SrcBlockInterface instance
    //!  and resets the connection index as zero.
    //!
    //! \return void
    //!
    //! \implements{18724548}
    void reset(void) noexcept
    {
        ptr.reset();
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        connIndex = 0U;
    };

    //! \brief Initializes the smart pointer to SrcBlockInterface instance
    //!  and connection index with the given arguments.
    //!
    //! \param [in] paramPtr: smart pointer to SrcBlockInterface instance.
    //! \param [in] paramConnIndex: Connection index.
    //!
    //! \return void
    //!
    //! \implements{18724554}
    void reset(
        const std::shared_ptr<SrcBlockInterface>& paramPtr,
        uint32_t const paramConnIndex) noexcept
    {
        ptr = paramPtr;
        connIndex = paramConnIndex;
    };

public:
    // Wrappers for all src interfaces

    //! \brief Wrapper interface which calls the dstRecvConsInfo()
    //!  interface of the source block with the connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvConsInfo.
    //!
    //! \return void
    void dstRecvConsInfo(
        EndInfoVector const& info) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvConsInfo(connIndex, info);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvSupportedElements()
    //!  interface of the source block with the given arguments
    //!  and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvSupportedElements.
    //!
    //! \return void
    void dstRecvSupportedElements(Elements const& inElements) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvSupportedElements(connIndex, inElements);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvAllocatedElements()
    //!  interface of the source block with the given arguments
    //!  and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvAllocatedElements.
    //!
    //! \return void
    void dstRecvAllocatedElements(Elements const& inElements) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvAllocatedElements(connIndex, inElements);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvPacketCreate()
    //!  interface of the source block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvPacketCreate.
    //!
    //! \return void
    void dstRecvPacketCreate(
        Packet const& packet) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvPacketCreate(connIndex, packet);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvPacketDelete()
    //!  interface of the source block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvPacketDelete.
    //!
    //! \return void
    void dstRecvPacketDelete(
        LwSciStreamPacket const handle) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvPacketDelete(connIndex, handle);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvPacketStatus()
    //!  interface of the source block with the given arguments
    //!  and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvPacketStatus.
    //!
    //! \return void
    void dstRecvPacketStatus(
        Packet const& origPacket) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvPacketStatus(connIndex, origPacket);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvPacketsComplete()
    //!  interface of the source block with the and connection index.
    //!
    //! \return void
    void dstRecvPacketsComplete(void) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvPacketsComplete(connIndex);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvSyncWaiter()
    //!  interface of the source block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvSyncWaiter.
    //!
    //! \return void
    void dstRecvSyncWaiter(
        Waiters const& syncWaiter) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvSyncWaiter(connIndex, syncWaiter);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvSyncSignal()
    //!  interface of the source block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvSyncSignal.
    //!
    //! \return void
    void dstRecvSyncSignal(
        Signals const& syncSignal) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvSyncSignal(connIndex, syncSignal);
        }
    };

    //! \brief Wrapper interface which calls the dstRecvPayload()
    //!  interface of the source block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::dstRecvPayload
    //!
    //! \return void
    void dstRecvPayload(
        Packet const& consPayload) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvPayload(connIndex, consPayload);
        }
    };

    //! \brief Wrapper interface which calls the dstDequeuePayload()
    //!  interface of the source block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters and return values</b>
    //!  - As specified for SrcBlockInterface::dstDequeuePayload.
    //!
    //! \return
    //! * PacketPtr returned by SrcBlockInterface::dstDequeuePayload.
    //!
    //! \implements{18724533}
    PacketPtr dstDequeuePayload(void) noexcept
    {
        PacketPtr rv {};
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            rv = ptr->dstDequeuePayload(connIndex);
        }

        return rv;
    };

    //! \brief Wrapper interface which calls the dstRecvPhaseReady()
    //!  interface of the source block with the connection index.
    //!
    //! \return void
    void dstRecvPhaseReady(void) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstRecvPhaseReady(connIndex);
        }
    };

public:
    //! \brief Notifies the source block about disconnect by calling
    //!  dstDisconnect() interface of the source block with the given
    //!  @a connIndex.
    //!
    //! \param [in] ptr: Smart pointer to SrcBlockInterface instance.
    //! \param [in] connIndex: Index within source block's list of destination
    //!  connections.
    //!
    //! \return void
    //!
    //! \implements{18724542}
    static void disconnectBlock(
        std::shared_ptr<SrcBlockInterface> const& ptr,
        uint32_t const connIndex) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->dstDisconnect(connIndex);
        }
    };

private:
    //! \brief Smart pointer to SrcBlockInterface instance. It is
    //!  initialized when a SafeConnection::Access object is created, it
    //!  is reset when a SafeConnection::Access object is destroyed.
    std::shared_ptr<SrcBlockInterface> ptr;

    //! \brief Index within source block's list of destination connections.
    //!  It is initialized to a valid connection index when a
    //!  SafeConnection::Access object is created. It is reset to
    //!  zero when the SafeConnection::Access object is destroyed.
    uint32_t connIndex {0U};
};

//! \brief Specialized version of LwSciStream::BlockWrap< T > for DstBlockInterface.
//!  It is a wrapper for DstBlockInterface smart pointer with an associated connection
//!  index. It exposes interfaces which checks the smart pointer to DstBlockInterface
//!  instance is not NULL and calls the appropriate interface of the DstBlockInterface
//!  instance along with the connection index.
//!
//! \implements{18723291}
template<>
class BlockWrap<DstBlockInterface>
{
public:
    //! Allows for deletion of objects through BlockWrap interface
    //! The destructor is made public only because VectorCast needs
    //! to access it when instrumenting the code.
    virtual ~BlockWrap(void) noexcept = default;
protected:
    //! The default empty constructor can be used, but the
    //! class cannot be directly instantiated.
    BlockWrap(void) noexcept = default;

    //! \brief Move constructor is needed by subclass.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    BlockWrap(BlockWrap<DstBlockInterface>&& ref) noexcept = default;
private:
    //! Other constructors are not allowed
    BlockWrap(const BlockWrap<DstBlockInterface>& ref) noexcept   = delete;
    BlockWrap<DstBlockInterface>&
    operator=(const BlockWrap<DstBlockInterface>& ref) & noexcept = delete;
    BlockWrap<DstBlockInterface>&
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    operator=(BlockWrap<DstBlockInterface>&& ref) & noexcept      = delete;

protected:
    // Child classes can fill in or clear the fields

    //! \brief Resets the smart pointer to DstBlockInterface instance
    //!  and resets the connection index as zero.
    //!
    //! \return void
    //!
    //! \implements{18724611}
    void reset(void) noexcept
    {
        ptr.reset();
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        connIndex = 0UL;
    };

    //! \brief Initializes the smart pointer to DstBlockInterface instance and
    //!  connection index with the given arguments.
    //!
    //! \param [in] paramPtr: smart pointer to DstBlockInterface instance.
    //! \param [in] paramConnIndex: Connection index.
    //!
    //! \return void
    //!
    //! \implements{18724617}
    void reset(
        const std::shared_ptr<DstBlockInterface>& paramPtr,
        uint32_t const paramConnIndex) noexcept
    {
        ptr = paramPtr;
        connIndex = paramConnIndex;
    };

public:

    // Wrappers for all src interfaces

    //! \brief Wrapper interface which calls the srcRecvProdInfo()
    //!  interface of the destination block with the
    //!  connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvProdInfo.
    //!
    //! \return void
    void srcRecvProdInfo(
        EndInfoVector const& info) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvProdInfo(connIndex, info);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvSupportedElements()
    //!  interface of the destination block with the given arguments
    //!  and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvSupportedElements.
    //!
    //! \return void
    void srcRecvSupportedElements(Elements const& inElements) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvSupportedElements(connIndex, inElements);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvAllocatedElements()
    //!  interface of the destination block with the given arguments
    //!  and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvAllocatedElements.
    //!
    //! \return void
    void srcRecvAllocatedElements(Elements const& inElements) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvAllocatedElements(connIndex, inElements);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvPacketCreate()
    //!  interface of the destination block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvPacketCreate.
    //!
    //! \return void
    void srcRecvPacketCreate(
        Packet const& packet) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvPacketCreate(connIndex, packet);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvPacketDelete()
    //!  interface of the destination block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvPacketDelete.
    //!
    //! \return void
    void srcRecvPacketDelete(
        LwSciStreamPacket const handle) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvPacketDelete(connIndex, handle);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvSyncWaiter()
    //!  interface of the destination block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvSyncWaiter.
    //!
    //! \return void
    //!
    //! \implements{18724566}
    void srcRecvSyncWaiter(
        Waiters const& syncWaiter) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvSyncWaiter(connIndex, syncWaiter);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvSyncSignal()
    //!  interface of the destination block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvSyncSignal.
    //!
    //! \return void
    void srcRecvSyncSignal(
        Signals const& syncSignal) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvSyncSignal(connIndex, syncSignal);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvPacketStatus()
    //!  interface of the destination block with the given arguments
    //!  and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for SrcBlockInterface::srcRecvPacketStatus.
    //!
    //! \return void
    void srcRecvPacketStatus(
        Packet const& origPacket) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvPacketStatus(connIndex, origPacket);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvPacketsComplete()
    //!  interface of the destination block with the and connection index.
    //!
    //! \return void
    void srcRecvPacketsComplete(void) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvPacketsComplete(connIndex);
        }
    };

    //! \brief Wrapper interface which calls the srcRecvPayload()
    //!  interface of the destination block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters</b>
    //!  - As specified for DstBlockInterface::srcRecvPayload.
    //!
    //! \return void
    void srcRecvPayload(
        Packet const& prodPayload) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvPayload(connIndex, prodPayload);
        }
    };

    //! \brief Wrapper interface which calls the srcDequeuePayload()
    //!  interface of the destination block with the given
    //!  arguments and connection index.
    //!
    //! <b>Parameters and return values</b>
    //!  - As specified for DstBlockInterface::srcDequeuePayload.
    //!
    //! \return
    //! * PacketPtr returned by DstBlockInterface::srcDequeuePayload.
    //!
    //! \implements{18724533}
    PacketPtr srcDequeuePayload(void) noexcept
    {
        PacketPtr rv {};
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            rv = ptr->srcDequeuePayload(connIndex);
        }

        return rv;
    };

    //! \brief Wrapper interface which calls the srcRecvPhaseChange()
    //!  interface of the source block with the connection index.
    //!
    //! \return void
    void srcRecvPhaseChange(void) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcRecvPhaseChange(connIndex);
        }
    };

public:
    //! \brief Notifies the destination block about disconnect by calling
    //!  srcDisconnect() interface of the destination block with the
    //!  given @a connIndex.
    //!
    //! \param [in] ptr: Smart pointer to DstBlockInterface instance.
    //! \param [in] connIndex: Index within destination block's list of source
    //!  connections.
    //!
    //! \return void
    //!
    //! \implements{18724605}
    static void disconnectBlock(
        std::shared_ptr<DstBlockInterface> const& ptr,
        uint32_t const connIndex) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != ptr) {
            ptr->srcDisconnect(connIndex);
        }
    };

private:
    //! \brief Smart pointer to DstBlockInterface instance. It is
    //!  initialized when a SafeConnection::Access object is
    //!  created, it is reset when a SafeConnection::Access
    //!  object is destroyed.
    std::shared_ptr<DstBlockInterface> ptr;

    //! \brief Index within destination block's list of source connections.
    //!  It is initialized to a valid connection index when a
    //!  SafeConnection::Access object is created. It is reset to
    //!  zero when the SafeConnection::Access object is destroyed.
    uint32_t connIndex {0U};
};

//! \brief SafeConnection is a template class which manages the connection
//!  between blocks and teardown of connected-blocks' smart pointers
//!  in a thread-safe manner.
//!
//!  * SafeConnection tracks a counter of callers that are lwrrently accessing
//!    functions of connected block's smart pointer. When a caller triggers
//!    disconnect, if no callers are accessing it, disconnect oclwrs
//!    immediately. Otherwise it is immediately ilwalidated, and no further
//!    attempts to use it are allowed, but disconnect is not done until any
//!    operations already in progress are completed. Once the last access is gone,
//!    disconnect is triggered automatically.
//!  * Tracking the in progress accesses is done through a specialized smart
//!    pointer object (SafeConnection::Access object)that safeguards the connection.
//!    This object is expected to be used in a transitory fashion. Callers
//!    create it, access the function(s) they need, and destroy it. (No allocations
//!    are performed in this process).
//!  * The reference counting makes use of several atomically modified fields
//!    that allow this to work without requiring any locks.
//!
//! \tparam T: SrcBlockInterface or DstBlockInterface
//!
//! \if TIER4_SWAD
//! \implements{18723294}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{20715159}
//! \endif
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A14_1_1), "LwSciStream-ADV-AUTOSARC++14-002")
template<typename T>
class SafeConnection final
{
public:
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
    //! \brief Access objects are the specialized smart pointers that
    //!  provide access to the connected block's interfaces.
    //!   - It inherits the BlockWrap< T > class through which the
    //!     SrcBlockInterface or DstBlockInterface of the connected
    //!     block can be accessed.
    //!
    //! \tparam T: SrcBlockInterface or DstBlockInterface.
    //!
    //! \implements{18723297}
    class Access : public BlockWrap<T>
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M2_10_1))
    {
    public:
        //! \todo  We really want to restrict use of these to temporary
        //!        objects on the stack. Callers should not save these
        //!        accesses long term. Any way to fully enforce this?


        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        //! \brief Parameterized constructor which initializes an Access instance
        //!  with the given parameters.
        //!
        //! \tparam T: SrcBlockInterface or DstBlockInterface.
        //!
        //! \param [in] paramConn: pointer to parent SafeConnection.
        //! \param [in] paramPtr: smart pointer of the connected block.
        //! \param [in] paramConnIndex: connection index within the
        //!  connected block.
        //!
        //! \implements{18724662}
        Access(LwSciStream::SafeConnection<T>* const paramConn,
               const std::shared_ptr<T>& paramPtr,
               std::uint32_t const paramConnIndex) noexcept :
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            BlockWrap<T>()
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
        {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            if (nullptr != paramConn) {
                if (paramConn->reserve()) {
                    if (paramConn->isConnected()) {
                        conn = paramConn;
                        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
                        BlockWrap<T>::reset(paramPtr, paramConnIndex);
                    } else {
                        paramConn->release();
                    }
                }
            }
        };
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M2_10_1))

        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M0_1_10), "Bug 3266014")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        //! \brief Move constructor which moves the parent
        //!  SafeConnection instance and the content of
        //!  BlockWrap instance from input Access object to
        //!  the new Access object and resets the input Access
        //!  object.
        //!
        //! \param [in] ref: reference to Access object.
        //!
        //! \implements{18724665}
        Access(Access&& ref) noexcept :
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
            BlockWrap<T> {std::move(ref)},
            conn {ref.conn}
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M0_1_10))
        {
            ref.reset();
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            ref.conn = nullptr;
        };

        // No empty or copy construction
        Access(void) noexcept = delete;
        Access(const Access& ref) noexcept = delete;

        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
        //! \brief Destructor which releases the parent SafeConnection.
        //!
        //! \implements{18724668}
        ~Access(void) noexcept final
        {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            if (nullptr != conn) {
                conn->release();
            }
        };
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

        // No copy operators
        Access& operator=(const Access& ref) & noexcept = delete;
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
        Access& operator=(Access&& ref) & noexcept = delete;

    private:
        //! \brief Pointer to the parent SafeConnection. It is initialized to instance
        //!  of parent SafeConnection and reserved by calling SafeConnection::reserve()
        //!  interface when a new Access object is created. It is released by calling
        //!  SafeConnection::release() interface when the Access object is destroyed.
        LwSciStream::SafeConnection<T>* conn {nullptr};
    };

private:
    //! \brief Enum representing internal state of SafeConnection
    //!
    //! \implements{18723690}
    enum class ConnState : uint8_t {
        //! \brief SafeConnection is available
        //!        for establishing new connection.
        Available,
        //! \brief SafeConnection is claimed by a block
        //!        for establishing a connection.
        Claim,
        //! \brief SafeConnection is in Connected state.
        Connect,
        //! \brief SafeConnection is in Disconnect state.
        Disconnecting,
        //! \brief SafeConnection is marked to be Destroyed.
        Destroyed
    };

public:
    //! \brief Default empty constructor and default destructor are allowed
    SafeConnection(void) noexcept  = default;
    ~SafeConnection(void) noexcept = default;
    //! \brief Copy operations and other constructors are not allowed
    SafeConnection(SafeConnection<T> const&) noexcept= delete;
    auto operator=(SafeConnection<T> const&) & noexcept -> SafeConnection& = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    SafeConnection(SafeConnection<T>&&) noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    auto operator=(SafeConnection<T>&&) & noexcept -> SafeConnection& = delete;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_11), "Bug 2738197")
    //! \brief Initializes the SafeConnection if it is not already done.
    //!  \if TIER4_SWUD
    //!   This interface proceeds only if the current state is
    //!   ConnState::Available and updates the state as ConnState::Claim
    //!   after initializing the SafeConnection.
    //!  \endif
    //!
    //! \tparam T: SrcBlockInterface or DstBlockInterface
    //!
    //! \param [in] paramBlockPtr: Smart pointer to SrcBlockInterface
    //!  instance if the template type is SrcBlockInterface. Smart pointer
    //!  to DstBlockInterface instance if the template type is
    //!  DstBlockInterface. Valid value: paramBlockPtr is valid
    //!  if it is not NULL.
    //!
    //! \return bool
    //! * true if the initialization is successful.
    //! * false if SafeConnection is not available for establishing
    //!   a new connection or paramBlockPtr is NULL.
    //!
    //! \if TIER4_SWAD
    //! \implements{18724626}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18723636}
    //! \endif
    bool connInitiate(const std::shared_ptr<T>& paramBlockPtr) noexcept
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_11))
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
        bool result {false};
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (nullptr != paramBlockPtr) {
            ConnState expected {ConnState::Available};
            if (state.compare_exchange_strong(expected, ConnState::Claim)) {
                blkPtr = paramBlockPtr;
                LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
                result = true;
            }
        }
        return result;
    };

    //! \brief Completes the SafeConnection with the
    //!  @a paramConnIndex if not already done.
    //!  \if TIER4_SWUD
    //!   This interface proceeds only if the current state is
    //!   ConnState::Claim and updates the state as ConnState::Connect
    //!   after completing the SafeConnection.
    //!  \endif
    //!
    //!
    //! <b>Preconditions</b>
    //! - The smart pointer to SrcBlockInterface or
    //!   DstBlockInterface instance should have been
    //!   already initialized by calling connInitiate()
    //!   interface.
    //!
    //! \param [in] paramConnIndex: The connection index within
    //!  the list of connected block's list of SafeConnections.
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{18724629}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18724446}
    //! \endif
    void connComplete(uint32_t const paramConnIndex) noexcept
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
        assert(ConnState::Claim == state);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A13_5_3))
        connIndex = paramConnIndex;
        state = ConnState::Connect;
    }

    //! \brief Cancels the SafeConnection by resetting the smart pointer
    //!  to SrcBlockInterface or DstBlockInterface instance of the connected
    //!  block if it was already initialized and makes the SafeConnection as
    //!  available for establishing a new connection.
    //!  \if TIER4_SWUD
    //!   This interface proceeds only if the current state is
    //!   ConnState::Claim and updates the state as ConnState::Available
    //!   after cancelling the SafeConnection.
    //!  \endif
    //!
    //!
    //! <b>Preconditions</b>
    //! - The smart pointer to SrcBlockInterface or
    //!   DstBlockInterface instance should have been
    //!   already initialized by calling connInitiate()
    //!   interface.
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{18724632}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18724452}
    //! \endif
    void connCancel(void) noexcept
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
        assert(ConnState::Claim == state);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A13_5_3))
        blkPtr.reset();
        state = ConnState::Available;
    }

    //! \brief Disconnects the SafeConnection if not already done.
    //!  \if TIER4_SWUD
    //!   - This interface proceeds only if the current state is
    //!     ConnState::Connect.
    //!   - Updates the current state as ConnState::Disconnecting.
    //!   - If refCount of the SafeConnection is zero, it notifies the
    //!     connected block about disconnect by calling disconnectBlock()
    //!     interface of the BlockWrap instance and resets the smart pointer
    //!     to SrcBlockInterface or DstBlockInterface instance of the
    //!     connected block using reset() interface of the BlockWrap instance.
    //!   - If the refCount is non zero, it means the
    //!     active SafeConnection::Access objects are present which
    //!     reserves the SafeConnection, so it will be disconnected
    //!     when the last SafeConnection::Access object is gone and
    //!     updates the state from ConnState::Disconnecting to
    //!     ConnState::Destroyed.
    //!  \endif
    //!
    //!
    //! <b>Preconditions</b>
    //! - The smart pointer to SrcBlockInterface or DstBlockInterface
    //!   instance and connection index should have been already
    //!   initialized by calling connInitiate() and connComplete()
    //!   interfaces respectively.
    //!
    //! \note Only the first thread calling this function is allowed to
    //!  proceed, multiple disconnects are allowed but ignored.
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{18724635}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18724455}
    //! \endif
    void disconnect(void) noexcept
    {
        // Atomically flag the state as disconnected.
        ConnState expected {ConnState::Connect};
        if (state.compare_exchange_strong(expected, ConnState::Disconnecting)){
            // Atomically subtract maxThreads from the counter.
            // If no other thread is using the connection, disconnect can be
            // done immediately. Otherwise it is left to the other thread when
            // it releases the reservation.
            if (counter.fetch_sub(maxThreads) == zeroCount) {
                finish();
            }
        }
    };

    //! \brief Checks whether the SafeConnection is connected to
    //!  a block instance or not.
    //!  \if TIER4_SWUD
    //!   Returns true if the current state is ConnState::Connect,
    //!   false otherwise.
    //!  \endif
    //!
    //! \return bool
    //! * true if connected.
    //! * false if not connected.
    //!
    //! \if TIER4_SWAD
    //! \implements{18724638}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18724458}
    //! \endif
    bool isConnected(void) const noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A13_5_3), "LwSciStream-ADV-AUTOSARC++14-001")
        return (ConnState::Connect == state);
    };

    //! \brief Reserves the SafeConnection.
    //! \if TIER4_SWUD
    //!  It atomically increments the refCount of the SafeConnection
    //!  if disconnect() interface has not yet been called. Otherwise,
    //!  releases the SafeConnection using SafeConnection::release() interface.
    //! \endif
    //!
    //! \return bool
    //!  * true if SafeConnection is reserved successfully.
    //!  * false if disconnect() interface has already been called.
    //!
    //! \if TIER4_SWAD
    //! \implements{18724641}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18724461}
    //! \endif
    //TODO:  Any way to restrict this so only the Access object and internal
    //       functions can call this?
    bool reserve(void) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
        bool result {false};
        // Atomically increments the counter.
        if (counter.fetch_add(incVal) >= zeroCount) {
            // If its non-negative, disconnect hasn't oclwrred yet and its safe
            // to proceed with using the pointer.
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            result = true;
        } else {
            // Otherwise, disconnect is initiated, cancel the reservation.
            release();
        }
        return result;
    };

    //! \brief Releases the SafeConnection.
    //! \if TIER4_SWUD
    //!  It atomically decrements the refCount of SafeConnection.
    //!  If it is the last pending reservation of the SafeConnection to be
    //!  released and disconnect() interface has been already called, it
    //!  notifies the connected block about disconnect by calling
    //!  disconnectBlock() interface of the BlockWrap instance and
    //!  resets the smart pointer to SrcBlockInterface or
    //!  DstBlockInterface instance of the connected block
    //!  and updates the state to ConnState::Destroyed if
    //!  the current state is ConnState::Disconnecting.
    //! \endif
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{18724644}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18724464}
    //! \endif
    //TODO:  Any way to restrict this so only the Access object and internal
    //       functions can call this?
    void release(void) noexcept
    {
        // Atomically decrements the counter.
        // If it reaches the lowest value, then this thread is responsible for
        // handling the actual disconnect.
        if ((static_cast<int64_t>(counter.fetch_sub(incVal)) - incVal) == -maxThreads) {
            finish();
        }
    };

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A7_1_5), "Bug 2751189")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M9_3_3), "Bug 2751379")
    //! \brief Creates an instance of SafeConnection::Access
    //! \if TIER4_SWUD
    //!  It calls the parameterized constructor of
    //!  SafeConnection::Access with the instance of this
    //!  SafeConnection, smart pointer to SrcBlockInterface
    //!  or DstBlockInterface instance of the connected block
    //!  and connection index passed as arguments.
    //!  Then it returns the created Access object. This object is
    //!  expected to be used in a transitory fashion. Callers create
    //!  it, access the BlockWrap interface(s) they need, and destroy it.
    //! \endif
    //!
    //! \return new SafeConnection::Access object.
    //!
    //! \if TIER4_SWAD
    //! \implements{18724647}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18723639}
    //! \endif
    auto getAccess(void) noexcept -> Access
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M9_3_3))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A7_1_5))
    {
        return Access(this, blkPtr, connIndex);
    };

private:
    //! \brief Informs the connected source or destination
    //!  block about the disconnection.
    //!
    //! \return void
    void informDisconnect(void) const noexcept
    {
        // Appropriate disconnect interface will be ilwoked based on the
        // specialization of BlockWrap class.
        BlockWrap<T>::disconnectBlock(blkPtr, connIndex);
    };

    //! \brief Handles the actual disconnect operation, informs the connected
    //!  source or destination block about the disconnection and reset the
    //!  referenced source or destination block's smart pointer.
    //!
    //! \return void
    void finish(void) noexcept
    {
        ConnState expected {ConnState::Disconnecting};
        if (state.compare_exchange_strong(expected, ConnState::Destroyed)) {
            informDisconnect();
            blkPtr.reset();
        }
    };

private:
    //! \cond TIER4_SWAD
    //! \brief Smart pointer to SrcBlockInterface or DstBlockInterface instance of
    //!  the connected block. It is initialized by calling connInitiate() interface. It is
    //!  reset during disconnect() call if no active SafeConnection::Access objects
    //!  are present for the connection or once the last SafeConnection::Access
    //!  object is gone.
    std::shared_ptr<T> blkPtr;

    //! \brief The connection index within the list of connected block's list
    //!  of SafeConnections. It is initialized by calling connComplete() interface.
    uint32_t connIndex {0U};
    //! \endcond

    //! \cond TIER4_SWUD
    //! \brief Enum variable indicates the state of the connection. By default,
    //! it is initialized as ConnState::Available. Updates are atomic,
    //! ensuring only one thread is allowed to perform the operations
    //! associated with advancing to the next state.
    std::atomic<ConnState> state {ConnState::Available};

    //! \brief Refcount used for tracking the number of SafeConnection::Access
    //!  objects present for the connection and triggering disconnect. By default,
    //!  it is initialized to zero.
    std::atomic<int32_t> counter {0};
    //! \endcond

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_5_2), "Bug 2738296")
    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.

    //! \cond
    // Large constant number used with the counter (needs to be bigger than the number
    // of threads that might access a block at once). By default, it is initialized to 0x4000.
    constexpr static int16_t maxThreads {0x4000};

    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.

    // Constant value for 1 used within SafeConnection.
    constexpr static int32_t incVal {1};
    // Constant value for 0 used within SafeConnection.
    constexpr static int32_t zeroCount {0};
    //! \endcond
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_5_2))
};

//! \brief Shorthand for source SafeConnection.
//!
//! \implements{18723303}
using SrcConnection = SafeConnection<SrcBlockInterface>;

//! \brief Shorthand for destination SafeConnection.
//!
//! \implements{18723309}
using DstConnection = SafeConnection<DstBlockInterface>;

//! \brief Shorthand for access to source SafeConnection.
//!
//! \implements{18723312}
using SrcAccess     = SafeConnection<SrcBlockInterface>::Access;

//! \brief Shorthand for access to destination SafeConnection.
//!
//! \implements{18723315}
using DstAccess     = SafeConnection<DstBlockInterface>::Access;

} // namespace LwSciStream

#endif // SAFECONNECTION_H
