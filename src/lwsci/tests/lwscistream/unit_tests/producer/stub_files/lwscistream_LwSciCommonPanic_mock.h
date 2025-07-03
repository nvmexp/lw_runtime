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
#include <functional>
#include "gmock/gmock.h"

struct lwscicommonPanicMock
{
    lwscicommonPanicMock();
    MOCK_METHOD(void, lwscicommonPanic_m, ());
    ~lwscicommonPanicMock();
};

static std::function<void(void)> _lwscicommonPanicMock;

lwscicommonPanicMock::lwscicommonPanicMock()
{
    assert(!_lwscicommonPanicMock);
    _lwscicommonPanicMock = [this]() { return lwscicommonPanic_m(); };
}

lwscicommonPanicMock::~lwscicommonPanicMock()
{
    _lwscicommonPanicMock = {};
}

//! \brief Function to panic for impossible errors
void LwSciCommonPanic()
{
 _lwscicommonPanicMock();
}

