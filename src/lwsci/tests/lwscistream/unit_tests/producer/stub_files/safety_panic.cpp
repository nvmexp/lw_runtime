//! \file
//! \brief LwSciStream panic implementation for safety builds
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include "lwscicommon_os.h"
#include "lwscistream_panic.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!    - It calls LwSciCommonPanic() interface to terminate the program
//!      exelwtion.
//!
//! \implements{20546970}
void lwscistreamPanic(void) noexcept
{
   LwSciCommonPanic();
}

} // namespace LwSciStream
