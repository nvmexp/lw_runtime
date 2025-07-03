//! \file
//! \brief LwSciStream panic implementation for non-safety builds
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "covanalysis.h"
#include "lwscistream_panic.h"

namespace LwSciStream {

//! \brief Function to panic for impossible errors
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M0_1_8), "Non-safety scan only")
void lwscistreamPanic(void) noexcept
{
   /* NOP */
}

} // namespace LwSciStream
