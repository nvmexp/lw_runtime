//! \file
//! \brief LwSciStream ipc common class declaration.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef IPC_COMM_COMMON_H
#define IPC_COMM_COMMON_H
#include <cstdint>
#include <atomic>
#include <cassert>
#include <utility>
#include <cstddef>
#include <iostream>
#include <array>
#include "covanalysis.h"
#include "lwscistream_common.h"

namespace LwSciStream {

//! \brief Flags used to monitor events available for
//!        processing over the IPC channel.
//!        This is a return structure used by IpcComm
//!        to communicate event and error state with
//!        the owning IPC block.
//!
//! \implements{19700556}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A9_6_1), "Proposed TID-599")
struct IpcQueryFlags {
    //! \brief Flag to indicate whether ipc read frame is available.
    bool readReady;

    //! \brief Flag to indicate whether ipc write frame is available while
    //!  there is at least one write request pending.
    bool writeReady;

    //! \brief Error encountered while fetching the event.
    //!  Set to LwSciError_Success if no error in fetching events, set
    //!  to LwSciError encountered while fetching events.
    LwSciError err;
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A9_6_1))

} // namespace LwSciStream
#endif // IPC_COMM_COMMON_H
