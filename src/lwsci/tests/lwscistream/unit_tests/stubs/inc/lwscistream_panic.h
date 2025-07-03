//! \file
//! \brief LwSciStream safety/non-safety panic declaration
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef LWSCISTREAM_PANIC_H
#define LWSCISTREAM_PANIC_H

#include "lwscicommon_os.h"

namespace LwSciStream {

//! \brief Function to panic for impossible errors
//!
//! \if TIER4_SWAD
//! \implements{20546958}
//! \endif
void lwscistreamPanic(void) noexcept;

} // namespace LwSciStream
#endif // LWSCISTREAM_PANIC_H
