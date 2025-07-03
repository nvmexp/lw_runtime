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

struct LwSciSyncAttrListCloneMock
{
    LwSciSyncAttrListCloneMock();
    MOCK_METHOD(LwSciError, LwSciSyncAttrListClone, (LwSciSyncAttrList,LwSciSyncAttrList*));
    ~LwSciSyncAttrListCloneMock();
};

static std::function<LwSciError(LwSciSyncAttrList,LwSciSyncAttrList*)> _LwSciSyncAttrListCloneMock;

LwSciSyncAttrListCloneMock::LwSciSyncAttrListCloneMock()
{
    assert(!_LwSciSyncAttrListCloneMock);
    //
    _LwSciSyncAttrListCloneMock = [this](LwSciSyncAttrList attr, LwSciSyncAttrList *newAttr )
    { return LwSciSyncAttrListClone(attr,newAttr); };
}

LwSciSyncAttrListCloneMock::~LwSciSyncAttrListCloneMock()
{
    _LwSciSyncAttrListCloneMock = {};
}



    //! \brief Function to be mocked
LwSciError  LwSciSyncAttrListClone(LwSciSyncAttrList attr, LwSciSyncAttrList *newAttr)
{          *newAttr=attr;
           _LwSciSyncAttrListCloneMock(attr,newAttr);
}
