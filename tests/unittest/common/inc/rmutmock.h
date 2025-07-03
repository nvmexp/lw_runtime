 /* Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure withoUTAPI_ the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

//! \file rm_ut_mock.h
//! \brief: This file contain all the mock function declaration exposed
//!         for test writes

#ifndef _RMUTMOCK_H_
#define _RMUTMOCK_H_

#include "lwmock.h"

#define UTAPI_MockReturn(fn)                 _LwMockReturn           (fn)
#define UTAPI_MockWillReturn(fn, v)          _LwMockWillReturn       (tc, fn, (const MOCK_RETURN_TYPE)v, 1)
#define UTAPI_MockWillReturn_Count(fn, v, c) _LwMockWillReturn       (tc, fn, (const MOCK_RETURN_TYPE)v, c)
#define UTAPI_MockWillReturnDefault(fn, v)   _LwMockWillReturnDefault(tc, fn, (const MOCK_RETURN_TYPE)v)
#define UTAPI_MockReturnParam(fn, p)         (MOCK_PTR_TYPE)_LwMockReturnParam      (fn, p)

#define UTAPI_MockParamWillReturn(fn,arg,v)  _LwMockWillReturnParam  (tc, fn, arg, (const MOCK_RETURN_TYPE)v)
#define UTAPI_IsParamMocked(fn, p)           _LwIsParamMocked        (fn, p)

#define UTAPI_MockInstallCallBack(fn, v)     _LwMockInstallCallBack  (tc, fn, (FUNCTION_POINTER)v)
#define UTAPI_GetMockedFunction(fn, v)       _LwGetMockedFunction    (fn, v)
#define UTAPI_MockGetFuncRefCount(fn)        _LwMockGetFuncRefCount  (fn)
#define UTAPI_IsMocked(fn)                   _LwIsMocked             (fn)

#endif // _RMUTMOCK_H_

