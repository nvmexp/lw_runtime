//! \file
//! \brief LwSciStream upstream block interface.
//!
//! \copyright
//! Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef SRCBLOCKINTERFACE_H
#define SRCBLOCKINTERFACE_H
#include <cstdint>
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
 * - When any of the SrcBlockInterface receives an invalid destination
 *   connection index, it triggers an error event with LwSciError_StreamBadDstIndex.
 * - When any of the SrcBlockInterface other than dstDisconnect(),
 *   dstRecvPayload() and dstDequeuePayload() is called when
 *   producer block in the stream is not connected to every consumer block in the
 *   stream, it triggers an error event with LwSciError_StreamNotConnected unless
 *   otherwise stated explicitly.
 *
 * \section lwscistream_input_parameters Input parameters
 * - Destination connection index passed as an input parameter to a
 *   SrcBlockInterface is valid if it is less than the possible number
 *   of destinations connections of that block.
 * - LwSciStreamPacket passed as input parameter to a SrcBlockInterface
 *   of any block other than pool block is valid if the packet creation
 *   event for the LwSciStreamPacket was received earlier from pool block
 *   through srcRecvPacketCreate() or dstRecvPacketCreate() call and packet
 *   deletion event for the same LwSciStreamPacket from pool block is not
 *   yet received through srcRecvPacketDelete() or dstRecvPacketDelete() call.
 *   In other words, they become valid when received by a srcRecvPacketCreate()
 *   or dstRecvPacketCreate() call, and remain valid until received in a
 *   srcRecvPacketDelete() or dstRecvPacketDelete() call.
 * - LwSciStreamPacket passed as input parameter to a SrcBlockInterface
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
//!  called by a connected destination block through SafeConnection, ilwoking
//!  the actual implementation. In general, these interfaces receive information
//!  or requests from the destination block and act accordingly.
//!
//! \implements{18699873}
class SrcBlockInterface
{
public:

    //
    // Connection definition functions
    //

    //! \brief Receives consumer information from downstream connection.
    //!   This indicates there is a complete path to all consumers
    //!   accessible through the indexed connection. The information is
    //!   stored and passed on as appropriate.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] info: Info from consumer(s)
    virtual void dstRecvConsInfo(
                        uint32_t const dstIndex,
                        EndInfoVector const& info) noexcept = 0;

    //
    // Element definition functions
    //

    //! \brief Receives supported element information from consumer endpoint(s)
    //!   and processes appropriately for the block type.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] inElements: Supported element information.
    virtual void dstRecvSupportedElements(
                        uint32_t const dstIndex,
                        Elements const& inElements) noexcept = 0;

    //! \brief Receives allocated element information from pool
    //!   and processes appropriately for the block type.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] inElements: Allocated element information.
    virtual void dstRecvAllocatedElements(
                        uint32_t const dstIndex,
                        Elements const& inElements) noexcept = 0;

    //
    // Packet definition functions
    //

    //! \brief Process new packet from the pool.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] origPacket: Reference to the incoming packet information.
    virtual void dstRecvPacketCreate(
                        uint32_t const dstIndex,
                        Packet const& origPacket) noexcept = 0;

    //! \brief Processes the deletion information of the packet referenced
    //!  by the given @a handle from Pool.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] handle: LwSciStreamPacket.
    virtual void dstRecvPacketDelete(
                        uint32_t const dstIndex,
                        LwSciStreamPacket const handle) noexcept = 0;

    //! \brief Processes acceptance or rejection by consumer(s) of a packet.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] origPacket: Packet instance containing status information.
    virtual void dstRecvPacketStatus(
                        uint32_t const dstIndex,
                        Packet const& origPacket) noexcept = 0;

    //! \brief Inform block that the pool has finished exporting packets.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    virtual void dstRecvPacketsComplete(
                        uint32_t const dstIndex) noexcept = 0;

    //! \brief Processes the LwSciSync waiter information from the consumer(s).
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] syncWaiter: Incoming waiter information.
    virtual void dstRecvSyncWaiter(
                        uint32_t const dstIndex,
                        Waiters const& syncWaiter) noexcept = 0;

    //! \brief Processes the LwSciSyncObjs signalled by the consumer(s).
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] syncSignal: Incoming signaller information.
    virtual void dstRecvSyncSignal(
                        uint32_t const dstIndex,
                        Signals const& syncSignal) noexcept = 0;

    //
    // Payload functions
    //

    //! \brief Processes a payload returning upstream for reuse.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    //! \param [in] packet: Packet information.
    virtual void dstRecvPayload(
                        uint32_t const dstIndex,
                        Packet const& consPayload) noexcept = 0;

    //! \brief Retrieve an available payload.
    //!   Only used to get payloads from Queue blocks.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    virtual PacketPtr dstDequeuePayload(
                        uint32_t const dstIndex) noexcept = 0;


    //! \brief Process message indicating downstream is ready for runtime.
    //!
    //! Note: This could be a non-virtual function in the base block but
    //!       we need to make it accessible through SafeConnection.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    virtual void dstRecvPhaseReady(
                        uint32_t const dstIndex) noexcept = 0;

    //! \brief Processes disconnect notification from destination block. It
    //!   passes on the notification to any connected source and destination
    //!   blocks, and schedules a pending LwSciStreamEventType_Disconnected event.
    //!
    //! \param [in] dstIndex: Index of connection to destination block.
    virtual void dstDisconnect(
                        uint32_t const dstIndex) noexcept = 0;

    //! \brief Default destructor of SrcBlockInterface
    virtual ~SrcBlockInterface(void) noexcept = default;

protected:
    //! \brief Declared constructor of abstract class protected.
    SrcBlockInterface(void) noexcept = default;
    SrcBlockInterface(SrcBlockInterface const&) noexcept = default;
    SrcBlockInterface& operator=(SrcBlockInterface const&) & noexcept = default;
    SrcBlockInterface(SrcBlockInterface&&) noexcept = default;
    SrcBlockInterface& operator=(SrcBlockInterface&&) & noexcept = default;
};

} // namespace LwSciStream

#endif // SRCBLOCKINTERFACE_H
