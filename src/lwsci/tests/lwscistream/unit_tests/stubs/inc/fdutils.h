//! \file
//! \brief Declaration of simple wrappers for fd_set macros
//!
//! The macros used to manipulate fd_sets contain a number of autosar
//!   violations which we can do nothing about since it isn't our code.
//!   We therefore wrap them in simple functions and isolate them in a
//!   separate file which we can omit from our scan reports. (Possibly
//!   we can colwert the functions to pure C code and scan with misra
//!   rather than autosar, but we'll probably still have violations.)
//!
//! \copyright
//! Copyright (c) 2019 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef FDUTILS_H
#define FDUTILS_H

#include <cstdlib>
#include <sys/select.h>

namespace LwSciStream {

static constexpr int32_t ILWALID_FD { -1 };

void
fdZEROWrap(
    fd_set& fds) noexcept;

int32_t
fdSETWrap(
    int32_t const fd,
    fd_set& fds,
    int32_t const oldMax) noexcept;

bool
fdISSETWrap(
    int32_t const fd,
    fd_set const& fds) noexcept;

} // namespace LwSciStream

#endif // FDUTILS_H
