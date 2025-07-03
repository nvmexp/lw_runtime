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

struct LwSciBufAttrListCloneMock
{
    LwSciBufAttrListCloneMock();
    MOCK_METHOD(LwSciError, LwSciBufAttrListClone, (LwSciBufAttrList,LwSciBufAttrList*));
    ~LwSciBufAttrListCloneMock();
};

static std::function<LwSciError(LwSciBufAttrList,LwSciBufAttrList*)> _LwSciBufAttrListCloneMock;

LwSciBufAttrListCloneMock::LwSciBufAttrListCloneMock()
{
    assert(!_LwSciBufAttrListCloneMock);
    //
    _LwSciBufAttrListCloneMock = [this](LwSciBufAttrList attr, LwSciBufAttrList *newAttr )
    { return LwSciBufAttrListClone(attr,newAttr); };
}

LwSciBufAttrListCloneMock::~LwSciBufAttrListCloneMock()
{
    _LwSciBufAttrListCloneMock = {};
}



    //! \brief Function to be mocked
LwSciError  LwSciBufAttrListClone(LwSciBufAttrList attr, LwSciBufAttrList *newAttr)
{           *newAttr=attr;
           _LwSciBufAttrListCloneMock(attr,newAttr);
}
