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

struct LwSciSyncFenceDupMock
{
    LwSciSyncFenceDupMock();
    MOCK_METHOD(LwSciError, LwSciSyncFenceDup, (const LwSciSyncFence*,LwSciSyncFence*));
    ~LwSciSyncFenceDupMock();
};

static std::function<LwSciError(LwSciSyncFence const*,LwSciSyncFence*)> _LwSciSyncFenceDupMock;

LwSciSyncFenceDupMock::LwSciSyncFenceDupMock()
{
    assert(!_LwSciSyncFenceDupMock);
    _LwSciSyncFenceDupMock = [this](LwSciSyncFence const*obj1,LwSciSyncFence *obj2)
    { return LwSciSyncFenceDup(obj1,obj2); };
}

LwSciSyncFenceDupMock::~LwSciSyncFenceDupMock()
{
    _LwSciSyncFenceDupMock = {};
}



    //! \brief Function to be mocked
LwSciError  LwSciSyncFenceDup(LwSciSyncFence   const  *obj1,LwSciSyncFence *obj2)
{
  memcpy(obj2,obj1,sizeof(LwSciSyncFence));
  _LwSciSyncFenceDupMock(obj1,obj2);
}
