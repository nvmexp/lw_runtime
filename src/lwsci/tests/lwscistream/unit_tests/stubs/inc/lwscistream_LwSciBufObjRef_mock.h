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
#include "sciwrap.h"

struct LwSciBufObjRefMock
{
    LwSciBufObjRefMock();
    MOCK_METHOD(LwSciError, LwSciBufObjRef, (LwSciBufObj));
    ~LwSciBufObjRefMock();
};

static std::function<LwSciError(LwSciBufObj)> _LwSciBufObjRefMock;

LwSciBufObjRefMock::LwSciBufObjRefMock()
{
    assert(!_LwSciBufObjRefMock);
    //LwSciBufObj obj;
    _LwSciBufObjRefMock = [this](LwSciBufObj obj) { return LwSciBufObjRef(obj); };
}

LwSciBufObjRefMock::~LwSciBufObjRefMock()
{
    _LwSciBufObjRefMock = {};
}



    //! \brief Function to be mocked
LwSciError  LwSciBufObjRef(LwSciBufObj obj1)
{
    return _LwSciBufObjRefMock(obj1);
}
