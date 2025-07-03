//! \file
//! \brief LwSciStream unit testing.
//!
//! \copyright
//! Copyright (c) 2020 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistream_panic.h"

struct lwscistreamPanicMock
{
    lwscistreamPanicMock();
    MOCK_METHOD(void, lwscistreamPanic_m, ());
    ~lwscistreamPanicMock();
};

static std::function<void(void)> _lwscistreamPanicMock;

lwscistreamPanicMock::lwscistreamPanicMock()
{
    assert(!_lwscistreamPanicMock);
    _lwscistreamPanicMock = [this]() { return lwscistreamPanic_m(); };
}

lwscistreamPanicMock::~lwscistreamPanicMock()
{
    _lwscistreamPanicMock = {};
}

namespace LwSciStream {

    //! \brief Function to panic for impossible errors
    void lwscistreamPanic(void) noexcept
    {
           _lwscistreamPanicMock();
    }

} // namespace LwSciStream
