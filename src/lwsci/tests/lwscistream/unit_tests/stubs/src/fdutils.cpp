//! \file
//! \brief Implementation of simple wrappers for fd_set macros
//!
//! The macros used to manipulate fd_sets contain a number of autosar
//!   violations which we can do nothing about since it isn't our code.
//!   We therefore wrap them in simple functions and isolate them in a
//!   separate file which we can omit from our scan reports. (Possibly
//!   we can colwert the functions to pure C code and scan with misra
//!   rather than autosar, but we'll probably still have violations.)
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstdlib>
#include <sys/select.h>
#include "lwscistream_common.h"
#include "fdutils.h"
#include "covanalysis.h"

namespace LwSciStream {

//! \brief Utility to safely zero fds and consolidate associated violations
void
fdZEROWrap(
    fd_set& fds) noexcept
{
    // The FD_ZERO macro has the following inherent violations:
    //   A3-9-1 (use of int rather than int32_t)
    //   A5-0-2 (using an int as a boolean)
    //   A7-1-1 (temporary constant variable not defined as const)
    //   A8-5-2 (temporary constant variable not initialized with braces)
    //   M0-1-2 (dead code after expansion)
    //   M5-0-4 (implicit sign colwersion)
    //   M6-3-1 (non-compound for loop body)
    FD_ZERO(&fds);
}

//! \brief Utility to safely set fds and consolidate associated violations
int32_t
fdSETWrap(
    int32_t const fd,
    fd_set& fds,
    int32_t const oldMax) noexcept
{
    // The FD_SET macro has the following inherent violations:
    //   A3-9-1 (use of int rather than int32_t)
    //   A4-7-1 (colwerting 64-bit to 32-bit)
    //   A4-7-1 (colwerting 8-bit to 64-bit [may be bug])
    //   A5-2-2 (C-style casts)
    //   M5-0-8 (colwersion increases size)
    //   M5-0-9 (colwersion changes sign)
    //   M5-0-10 (implicit cast)
    //   M5-0-21 (bitwise operation on signed type)
    //   M6-2-1 (assignment used in sub-expression)
    // CERT requires checking for negative FD, even though we've made sure
    //   elsewhere that we have valid values. CERT is stupid, and ugly, and
    //   it's mother dresses it funny.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (fd >= 0) {
        FD_SET(fd, &fds);
    }
    return std::max(fd, oldMax);
}

//! \brief Utility to safely check fds and consolidate associated violations
bool
fdISSETWrap(
    int32_t const fd,
    fd_set const& fds) noexcept
{
    // The FD_ISSET macro has the following inherent violations:
    //   A3-9-1 (use of int rather than int32_t)
    //   A4-7-1 (colwerting 64-bit to 32-bit)
    //   A4-7-1 (colwerting 8-bit to 64-bit [may be bug])
    //   A5-2-2 (C-style casts)
    //   M5-0-8 (colwersion increases size)
    //   M5-0-9 (colwersion changes sign)
    //   M5-0-10 (implicit cast)
    //   M5-0-21 (bitwise operation on signed type)
    // CERT requires checking for negative FD, even though we've made sure
    //   elsewhere that we have valid values. CERT is stupid, and ugly, and
    //   it's mother dresses it funny.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    return ((fd >= 0) && FD_ISSET(fd, &fds));
}

} // namespace LwSciStream
