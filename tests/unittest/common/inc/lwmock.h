/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation. Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//! \file lwmock.h
//! \brief This file contains all the declration of functions and data structure
//!        required for MOCKing.

#ifndef _LWMOCK_H_
#define _LWMOCK_H_
#define _VA_LIST_DEFINED
#include "lwtest.h"
#include "utility.h"

#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE 0
#endif

typedef void (*FUNCTION_POINTER)(void);

// MOCK structures
typedef struct MOCK_NODE     MOCK_NODE;
typedef struct MOCK          MOCK;
typedef struct MOCK_ELEMENT  MOCK_ELEMENT;
typedef struct MOCK_FUNCTION  MOCK_FUNCTION;

struct MOCK_NODE
{
    unsigned int  hash;
    MOCK         *pMock;
    MOCK_NODE    *pNext;
};

struct MOCK
{
    char          *pName;

    //
    // pParentFnName is NULL in case of mocking a function.
    // It will come into picture if function arg is mocked
    // because in that case mocked arg's parent will be related function and
    // in that scenario we say that function is mocked from API _LwIsMocked(fn).
    //
    char          *pParentFnName;
    int           isMocked;
    MOCK_RETURN_TYPE         pDefaultData;
    MOCK_FUNCTION *pFuncPointerData;
    MOCK_ELEMENT  *pElementHead;
    MOCK_ELEMENT  *pElementTail;
};

struct MOCK_ELEMENT
{
    MOCK_RETURN_TYPE         pData;
    signed int    refCount;
    MOCK_ELEMENT *pNext;
};

struct MOCK_FUNCTION
{
    FUNCTION_POINTER pFuncPointer;
    unsigned int     count;
};

enum MOCK_STATUS
{
    MOCK_INSTALL_SUCCESSFULL,
    MOCK_CALLBACK_ALREADY_INSTALLED,
    MOCK_ALREADY_INSTALLED
};

typedef enum MOCK_STATUS MOCK_STATUS;

void             _LwMockSetup(LwTest *tc);
MOCK_RETURN_TYPE _LwMockReturn(const char *const pName);
void             _LwMockTeardown(LwTest *tc);
int              _LwMockGetFuncRefCount(const char *const pName);
int              _LwIsMocked(const char *const pName);
int              _LwIsParamMocked(const char   *const pFuncName,
                                  const char   *const pArg);
MOCK_RETURN_TYPE _LwMockReturnParam(const char   *const pFuncName,
                                    const char   *const pArg);
MOCK_STATUS      _LwMockWillReturnParam(LwTest *tc, const char *const pFnName,
                                        const char *const pArgPos,
                                        const MOCK_RETURN_TYPE pData);
MOCK_STATUS      _LwMockWillReturn(LwTest *tc, const char *const pName,
                                   const MOCK_RETURN_TYPE pData,
                                   const int count);
MOCK_STATUS      _LwMockInstallCallBack(LwTest *tc, const char *const pName,
                                        FUNCTION_POINTER pData);
MOCK_STATUS      _LwMockWillReturnDefault(LwTest *tc, const char *const pName,
                                          const MOCK_RETURN_TYPE pData);
FUNCTION_POINTER _LwGetMockedFunction(const char *const pName,
                                      int bIncRefCount);

#endif // _LWMOCK_H

