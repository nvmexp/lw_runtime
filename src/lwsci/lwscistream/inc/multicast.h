//! \file
//! \brief LwSciStream MultiCast declaration.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef MULTICAST_H
#define MULTICAST_H
#include <cstdint>
#include <utility>
#include <atomic>
#include <map>
#include <unordered_map>
#include "covanalysis.h"
#include "block.h"
#include "elements.h"
#include "syncwait.h"
#include "syncsignal.h"
#include "lwscistream_common.h"

/**
 * @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
 * Generic statements applicable for LwSciStream interfaces.
 * @{
 */

/**
 * \page lwscistream_page_blanket_statements LwSciStream blanket statements
 *
 * \section lwscistream_dependency Dependency on other sub-elements
 *   LwSciStream calls the following LwSciSync interfaces:
 *    - LwSciSyncAttrListAppendUnreconciled() for merging unreconciled LwSciSyncAttrLists.
 * \section lwscistream_dependency Dependency on other sub-elements
 *   LwSciStream calls the following LwSciBuf interfaces:
 *    - LwSciBufAttrListAppendUnreconciled() for merging unreconciled LwSciBufAttrLists.
 */

/**
 * @}
 */

namespace LwSciStream
{

//! \brief MultiCast class implements the functionality of the multicast block.
//!   When a stream has more than one consumer, a multicast block is used to
//!   connect the separate pipelines. A multicast block has one source
//!   connection and multiple destination connections. It distributes the
//!   producer's resources and actions to all the consumers, who are unaware
//!   that each other exist. It combines all the consumers' resources and
//!   actions, making them appear as a single virtual consumer to the producer.
//!
//! * It inherits from the Block class which provides common functionalities
//!   for all blocks.
//! * It overrides the SrcBlockInterface interfaces, which are called by the
//!   destination blocks.
//! * It overrides the DstBlockInterface interfaces, which are called by the
//!   source block.
//!
//! \if TIER4_SWAD
//! \implements{19781388}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{20263161}
//! \endif
class MultiCast : public Block
{
public:
    //! \brief Constructs an instance of the MultiCast class and initializes
    //!   its data members. It calls the constructor of the base Block class
    //!   with BlockType::MULTICAST, one source connection and the given
    //!   number of destination connections.
    //!
    //! \param [in] dstTotal: The number of destination connections.
    //!
    //! \return None
    //!
    //! \if TIER4_SWAD
    //! \implements{19781307}
    //! \endif
    explicit MultiCast(
        uint32_t const dstTotal) noexcept;

    MultiCast(const MultiCast&) noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    MultiCast(MultiCast&&) noexcept = delete;

    MultiCast& operator=(const MultiCast&) & noexcept = delete;

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    MultiCast& operator=(MultiCast&&) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    ~MultiCast(void) noexcept final = default;

    //! \brief Disconnects the source block and all destination blocks.
    //!
    //! \copydetails LwSciStream::APIBlockInterface::disconnect
    //!
    //! \if TIER4_SWAD
    //! \implements{19781310}
    //! \endif
    LwSciError disconnect(void) noexcept final;

    //! \brief Receives consumer endpoint information from a connection,
    //!   updates the banch map, and copies the information into the
    //!   aclwmulated list.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvConsInfo
    //!
    //! \return void
    void dstRecvConsInfo(
        uint32_t const dstIndex,
        EndInfoVector const& info) noexcept final;

    //! \brief Receives supported element information from consumer(s) on
    //!   one downstream connection and merges into consolicated list.
    //!   When the last output provides its support, forwards the merged
    //!   list upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSupportedElements
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_AlreadyDone: Elements have already been received from
    //!   the output.
    //! - Any error returned by Elements::mapMerge().
    //! - Any error returned by Elements::mapDone().
    //! - Any error returned by Elements::dataSend().
    void dstRecvSupportedElements(
        uint32_t const dstIndex,
        Elements const& inElements) noexcept final;

    //! \brief Initializes sync information with the number of elements
    //!   and then ilwokes the base function to pass the element information
    //!   downstream.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvAllocatedElements
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by Waiters::sizeInit().
    //! - Any error set by Block::srcRecvAllocatedElements()
    void srcRecvAllocatedElements(
        uint32_t const srcIndex,
        Elements const& inElements) noexcept final;

    //! \brief Receives acceptance or rejection of a packet from the
    //!   consumer(s) and merges into consolidated list. When the last
    //!   output provides its status for the packet, forwards the merged
    //!   list upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPacketStatus
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet handle not found in map.
    //! - Any error returned by Packet::statusConsMerge().
    void dstRecvPacketStatus(
        uint32_t const dstIndex,
        Packet const& origPacket) noexcept final;

    //! \brief Receives consumer LwSciSync waiter information from the output
    //!   referenced by @a dstIndex and merges it into the consolidated
    //!   list. When all outputs have provided their information, sends
    //!   the full list upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncWaiter
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by BranchTrack::set().
    //! - Any error returned by Waiters::merge().
    //! - Any error returned by Waiters::setDone().
    void dstRecvSyncWaiter(
        uint32_t const dstIndex,
        Waiters const& syncWaiter) noexcept final;

    //! \brief Receives consumer LwSciSync signal information from the output
    //!   referenced by @a dstIndex and merges it into the consolidated
    //!   list. When all outputs have provided their information, sends
    //!   the full list upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSyncSignal
    //!
    //! \return void, Triggers the following error events:
    //! - Any error returned by BranchTrack::set().
    //! - Any error returned by Signals::collate().
    //! - Any error returned by Signals::setDone().
    void dstRecvSyncSignal(
        uint32_t const dstIndex,
        Signals const& syncSignal) noexcept final;

    //! \brief Receives returned packet, marks it as no longer in use by the
    //!   output, and adds its fences to the aclwmulated list for the packet.
    //!   If all outputs have returned the packet, forwards it upstream.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPayload
    //!
    //! \return void, Triggers the following error events:
    //! - LwSciError_StreamBadPacket: Packet is not valid.
    //! - LwSciError_StreamPacketInaccessible: Packet is not downstream.
    //! - Any error returned by Packet::fenceConsCollate().
    //!
    //! \if TIER4_SWAD
    //! \implements{19781334}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept final;

    //! \brief Disconnects the source block and disconnects destination
    //!   block at the given @a dstIndex, prepares the
    //!   LwSciStreamEventType_Disconnected event if all the destination
    //!   blocks are disconnected.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstDisconnect
    //!
    //! \return void
    //!  - Triggers the following error events:
    //!     - LwSciError_IlwalidState: If the disconnect notification from
    //!       the destination block with the @a dstIndex has already been received.
    //!
    //! \if TIER4_SWAD
    //! \implements{19781337}
    //! \endif
    void dstDisconnect(
        uint32_t const dstIndex) noexcept final;

    //! \brief Disconnects source and destination blocks and prepares
    //!   LwSciStreamEventType_Disconnected event.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcDisconnect
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19781367}
    //! \endif
    void srcDisconnect(
        uint32_t const srcIndex) noexcept final;

private:
    //! \brief Number of output branches.
    //!   Initialized during construction with user input.
    size_t                      branchCount;

    //! \brief Track range of consumers associated with each connection.
    //!   Initialized at construction with connection count and then filled
    //!   with consumer information as it arrives from each connection.
    BranchMap                   branchMap;

    //! \brief Aclwmulated list of consumer info before passing to base block.
    //!   Initialized at construction to empty and filled as consumers connect.
    EndInfoVector               aclwmulatedConsInfo;

    //! \brief Track which destination blocks have provided Elements.
    //!   Initialized at construction with branch count.
    BranchTrack                 consumerElementsBranch;

    //! \brief Aclwmulated list of elements from consumers.
    Elements                    consumerElements;

    //! \brief Track which destination blocks have provided Waiters.
    //!   Initialized at construction with branch count.
    BranchTrack                 consSyncWaiterBranch;

    //! \brief Consolidated list of sync waiter information from all consumers.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Waiters                     consSyncWaiter;

    //! \brief Track which destination blocks have provided Signals.
    //!   Initialized at construction with branch count.
    BranchTrack                 consSyncSignalBranch;

    //! \brief Consolidated list of sync signal information from all consumers.
    //!   It is initialized to empty at construction, and then its array
    //!   is initialized when the allocated element list arrives.
    Signals                     consSyncSignal;

    //! \brief Track which destination blocks have disconnected.
    //!   Initialized at construction with branch count.
    BranchTrack                 disconnectBranch;
};

}

#endif // MULTICAST_H
