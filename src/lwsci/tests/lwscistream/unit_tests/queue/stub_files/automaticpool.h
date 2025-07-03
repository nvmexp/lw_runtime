//! \file
//! \brief LwSciStream automatic pool class declaration.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef AUTOMATICPOOL_H
#define AUTOMATICPOOL_H
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <array>
#include "covanalysis.h"
#include "pool.h"

namespace LwSciStream {

//! \brief The AutomaticPool is as a default secondary pool connected to the
//!   C2CDst block if the application doesn't attach a pool instance when
//!   creating the C2CDst block.
//!
//!   AutomaticPool
//!   - Inherits from the Pool class, which provides the base pool functions.
//!   - Spawns a thread that will monitor the pool for events.
//!   - Reads out the producer and consumer attributes as they arrive. Takes
//!     the producer specification as the final packet layout.
//!   - Creates three packets and allocates buffers. Sends packets upstream
//!     to the C2CSrc block and downstream to the consumer block.
//!   - Handles the returning packet status events.
class AutomaticPool :
    public Pool
{
public:
    //! \brief Constructs pool with three packets
    AutomaticPool(void) noexcept;

    //! \brief Destroys the AutomaticPool block instance.
    ~AutomaticPool(void) noexcept final;

    AutomaticPool(const AutomaticPool&) noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    AutomaticPool(AutomaticPool&&) noexcept = delete;
    AutomaticPool& operator=(const AutomaticPool&) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    AutomaticPool& operator=(AutomaticPool&&) & noexcept = delete;

private:
    //! \brief Waits for and handles pool events.
    void handlePoolEvents(void) noexcept;

    // Helper functions to handle pool events

    //! \brief Handles LwSciStreamEventType_Elements event.
    //!
    //! \return bool, true if the operation completed successfully.
    bool handlePacketCreate(void) noexcept;

    //! \brief Handles LwSciStreamEventType_PacketStatus event.
    //!
    //! \return bool, true if the operation completed successfully.
    bool handlePacketsStatus(void) noexcept;

private:
    //! \brief Default number of packets
    static constexpr uint32_t   numPackets{ 3U };

    //! \brief Dispatched thread for processing pool events.
    //!   This thread launches when a new AutomaticPool instance is
    //!   created and exelwtes handlePoolEvents().
    std::thread                 dispatchThread;

    //! \brief vector of handles referencing to packets created by this pool.
    std::array<LwSciStreamPacket, numPackets> newPackets;
};

} // namespace LwSciStream

#endif // AUTOMATICPOOL_H
