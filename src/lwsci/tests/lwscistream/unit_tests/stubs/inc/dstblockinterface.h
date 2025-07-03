//! \file
//! \brief LwSciStream downstream block interface.
//!
//! \copyright
//! Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef DSTBLOCKINTERFACE_H
#define DSTBLOCKINTERFACE_H
#include <cstdint>
#include <utility>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "endinfo.h"
#include "elements.h"
#include "packet.h"
#include "syncwait.h"
#include "syncsignal.h"

namespace LwSciStream {

/**
 * @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
 * Generic statements applicable for LwSciStream interfaces.
 * @{
 */

/**
 * \page lwscistream_page_blanket_statements LwSciStream blanket statements
 * \section lwscistream_return_values Return values
 * - When any of the DstBlockInterface receives invalid source connection index,
 *   it triggers an error event with LwSciError_StreamBadSrcIndex.
 * - When any of the DstBlockInterface other than srcDisconnect() is called when
 *   producer block in the stream is not connected to every consumer block in the
 *   stream, it triggers an error event with LwSciError_StreamNotConnected unless
 *   otherwise stated explicitly.
 *
 * \section lwscistream_input_parameters Input parameters
 * - Source connection index passed as an input parameter to a
 *   DstBlockInterface is valid if it is less than the possible number
 *   of source connections of that block.
 * - LwSciStreamPacket passed as input parameter to a DstBlockInterface
 *   of any block other than pool block is valid if the packet creation
 *   event for the LwSciStreamPacket was received earlier from pool block
 *   through srcRecvPacketCreate() or dstRecvPacketCreate() call and packet
 *   deletion event for the same LwSciStreamPacket from pool block is not
 *   yet received through srcRecvPacketDelete() or dstRecvPacketDelete() call.
 *   In other words, they become valid when received by a srcRecvPacketCreate()
 *   or dstRecvPacketCreate() call, and remain valid until received in a
 *   srcRecvPacketDelete() or dstRecvPacketDelete() call.
 * - LwSciStreamPacket passed as input parameter to a DstBlockInterface
 *   of pool block is valid if it is returned from a successful call
 *   to apiPacketCreate() interface of the pool block and has not yet been
 *   deleted by using apiPacketDelete() interface of the pool block unless
 *   otherwise stated explicitly.
 */

/**
 * @}
 */

//! \brief Set of block interfaces which are declared as pure virtual functions.
//!  These interfaces are overridden by derived classes as required. They are
//!  called by a connected source block through SafeConnection, ilwoking the
//!  actual implementation. In general, these interfaces receive information
//!  or requests from the source block and act accordingly.
//!
//! \implements{18699882}
class DstBlockInterface
{
public:

    //
    // Connection definition functions
    //

    //! \brief Receives producer information from upstream connection.
    //!   This indicates there is a complete path to the producer
    //!   accessible through the indexed connection. The information is
    //!   stored and passed on as appropriate, and triggers an
    //!   LwSciStreamEventType_Connected event.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] info: Info from producer
    virtual void srcRecvProdInfo(
                        uint32_t const srcIndex,
                        EndInfoVector const& info) noexcept = 0;

    //
    // Element definition functions
    //

    //! \brief Receives supported element information from producer endpoint(s)
    //!   and processes appropriately for the block type.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] inElements: Supported element information.
    virtual void srcRecvSupportedElements(
                        uint32_t const srcIndex,
                        Elements const& inElements) noexcept = 0;

    //! \brief Receives allocated element information from pool
    //!   and processes appropriately for the block type.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] inElements: Allocated element information.
    virtual void srcRecvAllocatedElements(
                        uint32_t const srcIndex,
                        Elements const& inElements) noexcept = 0;

    //
    // Packet definition functions
    //

    //! \brief Process new packet from the pool.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] origPacket: Reference to the incoming packet information.
    virtual void srcRecvPacketCreate(
                        uint32_t const srcIndex,
                        Packet const& origPacket) noexcept = 0;

    //! \brief Processes the deletion information of the packet referenced
    //!  by the given @a handle from Pool.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] handle: LwSciStreamPacket.
    virtual void srcRecvPacketDelete(
                        uint32_t const srcIndex,
                        LwSciStreamPacket const handle) noexcept = 0;

    //! \brief Processes acceptance or rejection by producer of a packet.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] origPacket: Packet instance containing status information.
    virtual void srcRecvPacketStatus(
                        uint32_t const srcIndex,
                        Packet const& origPacket) noexcept = 0;

    //! \brief Inform block that the pool has finished exporting packets.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    virtual void srcRecvPacketsComplete(
                        uint32_t const srcIndex) noexcept = 0;

    //! \brief Processes the LwSciSync waiter information from the producer.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] syncWaiter: Incoming waiter information.
    virtual void srcRecvSyncWaiter(
                        uint32_t const srcIndex,
                        Waiters const& syncWaiter) noexcept = 0;

    //! \brief Processes the LwSciSyncObjs signalled by the producer.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] syncSignal: Incoming signaller information.
    virtual void srcRecvSyncSignal(
                        uint32_t const srcIndex,
                        Signals const& syncSignal) noexcept = 0;

    //
    // Payload functions
    //

    //! \brief Processes a payload coming downstream from the producer.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    //! \param [in] packet: Packet information.
    virtual void srcRecvPayload(
                        uint32_t const srcIndex,
                        Packet const& prodPayload) noexcept = 0;

    //! \brief Retrieve an available payload.
    //!   Only used to get payloads from Pool blocks.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    virtual PacketPtr srcDequeuePayload(
                        uint32_t const srcIndex) noexcept = 0;

    //! \brief Process message indicating upstream has entered runtime.
    //!
    //! Note: This could be a non-virtual function in the base block but
    //!       we need to make it accessible through SafeConnection.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    virtual void srcRecvPhaseChange(
                        uint32_t const srcIndex) noexcept = 0;

    //! \brief Processes disconnect notification from source block. It passes on
    //!   the notification to any connected source and destination blocks, and
    //!   schedules a pending LwSciStreamEventType_Disconnected event.
    //!
    //! \param [in] srcIndex: Index of connection to source block.
    virtual void srcDisconnect(
                        uint32_t const srcIndex) noexcept = 0;

    //! \brief Default destructor of DstBlockInterface
    virtual ~DstBlockInterface(void) noexcept = default;

protected:
    //! \brief Declared constructor of abstract class protected.
    DstBlockInterface(void) noexcept = default;
    DstBlockInterface(DstBlockInterface const&) noexcept = default;
    DstBlockInterface& operator=(DstBlockInterface const&) & noexcept = default;
    DstBlockInterface(DstBlockInterface&&) noexcept = default;
    DstBlockInterface& operator=(DstBlockInterface&&) & noexcept = default;
};

} // namespace LwSciStream

#endif // DSTBLOCKINTERFACE_H
