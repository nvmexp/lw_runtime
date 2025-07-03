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

struct LwSciSyncObjRefMock
{
    LwSciSyncObjRefMock();
    MOCK_METHOD(LwSciError, LwSciSyncObjRef, (LwSciSyncObj));
    ~LwSciSyncObjRefMock();
};

static std::function<LwSciError(LwSciSyncObj)> _LwSciSyncObjRefMock;

LwSciSyncObjRefMock::LwSciSyncObjRefMock()
{
    assert(!_LwSciSyncObjRefMock);
    //LwSciSyncObj obj;
    _LwSciSyncObjRefMock = [this](LwSciSyncObj obj)  { return LwSciSyncObjRef(obj); };
}

LwSciSyncObjRefMock::~LwSciSyncObjRefMock()
{
    _LwSciSyncObjRefMock = {};
}



    //! \brief Function to be mocked
LwSciError  LwSciSyncObjRef(LwSciSyncObj     obj1)
{
           _LwSciSyncObjRefMock(obj1);
}
