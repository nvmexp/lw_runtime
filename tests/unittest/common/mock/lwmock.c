/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//! \file lwmock.c
//! \brief This file contains all the definations of functions required
//!        for MOCKing.

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lwmock.h"

static MOCK_NODE *pHeadNode;

static MOCK         *_mockCreate       (LwTest *tc, const char *const pName);
static MOCK         *_mockGet          (LwTest *tc, const char *const pName,
                                        char bCreate, char bCheckParamMocked);
static unsigned int  _mockHash         (const char *const str);
static void          _mockAddByHash    (LwTest *tc, MOCK *pMock,
                                        unsigned int hash);
static void          _mockInsertElement(LwTest *tc, MOCK *pMock,
                                        const MOCK_RETURN_TYPE pData,
                                        const signed int count);

/*
 * @brief Initialise the headnode pointer
 *
 * @param[in] tc    Pointer to testCase
 */
void
_LwMockSetup
(
    LwTest *tc
)
{
    pHeadNode = NULL;
}

/*
 * @brief Free the allocated resources
 *
 * @param[in] tc    Pointer to testcase
 */
void
_LwMockTeardown
(
    LwTest *tc
)
{
    MOCK_NODE    *pNode;
    MOCK_ELEMENT *pElement;
    while (pHeadNode != NULL)
    {
        pNode     = pHeadNode;
        pHeadNode = pHeadNode->pNext;

        // deallocate the element list
        while (pNode->pMock->pElementHead != NULL)
        {
            pElement = pNode->pMock->pElementHead;
            pNode->pMock->pElementHead = pElement->pNext;

            pElement->pNext = NULL;
            free(pElement);
        }
        pNode->pMock->pElementTail     = NULL;
        pNode->pMock->pDefaultData     = 0;

        if (pNode->pMock->pFuncPointerData != NULL)
        {
            pNode->pMock->pFuncPointerData->pFuncPointer = NULL;
            free(pNode->pMock->pFuncPointerData);
        }
        pNode->pMock->pFuncPointerData = NULL;

        // deallocate the mock name
        free(pNode->pMock->pName);
        if (pNode->pMock->pParentFnName)
        {
            free(pNode->pMock->pParentFnName);
        }

        // deallocate the mock object
        free(pNode->pMock);
        pNode->pMock = NULL;
        pNode->pNext = NULL;
        free(pNode);
    }
}

/*
 * @brief Redirect function call
 *
 * @param[in] tc     Pointer to test case
 * @param[in] pName  Function name which needs redirection
 * @param[in] pData  pointer to redirected function
 *
 * @return MOCK_INSTALL_SUCCESSFULL is pass else other MOCK_STATUS value
 */
MOCK_STATUS
_LwMockInstallCallBack
(
    LwTest *tc,
    const char *const pName,
    FUNCTION_POINTER  pData
)
{

    MOCK *pMock;

    //
    // Get MOck node for specified function name.
    // If not already exists , create new
    //
    pMock = _mockGet(tc, pName, 1, 0);

    //
    // If any type of MOCK is already installed for this function,
    // this call should fail
    //
    if (pMock->isMocked)
    {
        return MOCK_ALREADY_INSTALLED;
    }

    if (pMock->pFuncPointerData == NULL)
    {
        pMock->pFuncPointerData = LW_ALLOC(MOCK_FUNCTION);
        pMock->pFuncPointerData->pFuncPointer = NULL;
        pMock->pFuncPointerData->count = 0;
    }

    pMock->isMocked = 1;
    pMock->pFuncPointerData->pFuncPointer = pData;
    return MOCK_INSTALL_SUCCESSFULL;
}

/*
 * @brief Return pointer to redirected function, if any
 *
 * @param[in] pName        Function name of the mocked function
 * @param[in] bIncRefCount If caller want to call mocked function,
 *                         specify True, else if it is just for query
 *                         specify false.
 *
 * @return Function pointer of redirected function
 */
FUNCTION_POINTER
_LwGetMockedFunction
(
    const char *const pName,
    int bIncRefCount
)
{
    MOCK *pMock;

    pMock = _mockGet(NULL, pName, 0, 0);

    if ((pMock == NULL) || (pMock->pFuncPointerData == NULL))
    {
        return NULL;
    }
    else
    {
        if (bIncRefCount)
        {
            pMock->pFuncPointerData->count++;
        }
        return pMock->pFuncPointerData->pFuncPointer;
    }

    return NULL;
}

/*
 * @brief Return number of times mocked function
 *                                 referenced from SUT
 *
 * @param[in] pName  Function name whoes reference count is to return
 *
 * @return nunmber of tims function referenced
 */
int
_LwMockGetFuncRefCount
(
    const char *const pName
)
{
    MOCK *pMock;

    pMock = _mockGet(NULL, pName, 0, 0);

    if ((pMock == NULL) || (pMock->pFuncPointerData == NULL))
    {
        return 0;
    }

    return pMock->pFuncPointerData->count;
}

/*
 * @brief Check if mock installed on specified function
 *
 * @param[in] pName  Function Name for which mock installation need to check
 *
 * \return TRUE if mock installed else FALSE
 */
int
_LwIsMocked
(
    const char *const pName
)
{
    MOCK *pMock;

    pMock = _mockGet(NULL, pName, 0, 1);

    if (pMock != NULL)
    {
        return pMock->isMocked;
    }
    return FALSE;
}

/*
 * @brief Setup the return value of the next 'count' returns from an object.
 *        The object's name is used to uniquely identify the name of the
 *        object to mock.
 *
 * @param[in] tc     The test-case to assert against upon error
 * @param[in] pName  The name of the object to mock
 * @param[in] pData  The data to return
 * @param[in] count  The number of times the data should be returned. A count
 *                   of -1 indicates that the value should be retured forever.
 *
 * @return MOCK_INSTALL_SUCCESSFULL is pass else MOCK_CALLBACK_ALREADY_INSTALLED
 *         if callback applied for this
 */
MOCK_STATUS
_LwMockWillReturn
(
    LwTest *tc,
    const char   *const pName,
    const MOCK_RETURN_TYPE  pData,
    const int    count
)
{
    MOCK *pMock;

    pMock = _mockGet(tc, pName, 1, 0);

    if (pMock->pFuncPointerData != NULL)
    {
        return MOCK_CALLBACK_ALREADY_INSTALLED;
    }

    _mockInsertElement(
        tc,
        pMock,
        pData,
        count);

    return MOCK_INSTALL_SUCCESSFULL;
}

MOCK_STATUS
_LwMockWillReturnParam
(
    LwTest *tc,
    const char *const pFnName,
    const char *const pArgPos,
    const MOCK_RETURN_TYPE pData
)
{
    MOCK *pMock;

    LwString *str = LwStringNew();
    MOCK_STATUS status;
    LwStringInsert(str, pFnName, 0);
    LwStringAppend(str, pArgPos);
    status = _LwMockWillReturn(tc, (const char *)str->buffer, pData, 1);

    if (status == MOCK_INSTALL_SUCCESSFULL)
    {
        pMock = _mockGet(tc, (const char *)str->buffer, 0, 0);
        pMock->pParentFnName = (char *)malloc((strlen(pFnName)+1)*sizeof(char));
        UNIT_ASSERT(pMock->pParentFnName != NULL);

        // Populated
        strcpy(pMock->pParentFnName, pFnName);
    }

    LwStringFree(str);
    return status;
}
/*
 * @brief Returns the next value that has been prepared to be returned for the
 *        object. If no more values are available, the default is returned if
 *        a default was specified. NULL is returned if a mock does not exist
 *        for the object.
 *
 * @param[in] pName  The name of the object to retrieve the return value for.
 *
 * @return The next return value that is ready for the object if available,
 *         else, the default return value if a default was specified, else
 *         NULL if no mock-object was found.
 */
MOCK_RETURN_TYPE
_LwMockReturn
(
    const char *const pName
)
{
    MOCK         *pMock;
    MOCK_ELEMENT *pElement;
    MOCK_RETURN_TYPE        pData = 0;

    // find the mock object for the caller (if one exists)
    pMock = _mockGet(NULL, pName, 0, 0);
    if (pMock == NULL)
    {
        return 0;
    }

    //
    // If an explict value exists for the object, return it.  Otherwise,
    // return the default mock value.
    //
    if (pMock->pElementHead != NULL)
    {
        pElement = pMock->pElementHead;
        pData    = pElement->pData;

        //
        // Subtract from the reference count if the value should not be
        // returned forever. If the reference count reaches zero, unlink and
        // free the element.
        //
        if (pElement->refCount != -1)
        {
            pElement->refCount--;
            if (pElement->refCount == 0)
            {
                pMock->pElementHead = pElement->pNext;
                pElement->pData     = 0;
                free(pElement);
                if (pMock->pElementHead == NULL)
                {
                    pMock->pElementTail = NULL;
                }
            }
        }
    }
    else
    {
        pData = pMock->pDefaultData;
    }
    return pData;
}

/*
 * @brief Setup the default return value for all returns from an object.
 *        The default is used when no explicit return mock-values have been
 *        specified.
 *
 * @param[in] tc     The test-case to assert against upon error
 * @param[in] pName  The name of the object to mock
 * @param[in] pData  The data to return
 *
 * @return MOCK_INSTALL_SUCCESSFULL is pass or MOCK_CALLBACK_ALREADY_INSTALLED
 *         call back functon already installed
 */
MOCK_STATUS
_LwMockWillReturnDefault
(
    LwTest *tc,
    const char   *const pName,
    const MOCK_RETURN_TYPE pData
)
{
    MOCK *pMock         = _mockGet(tc, pName, 1, 0);

    if (pMock->pFuncPointerData != NULL)
    {
        return MOCK_CALLBACK_ALREADY_INSTALLED;
    }

    pMock->pDefaultData = pData;
    return MOCK_INSTALL_SUCCESSFULL;
}

/*
 * @brief return the value which is mocked for specified argument
 *
 * @param[in] pFuncName  The name of the function whose argument is mocked
 * @param[in] pArg       The argemnt number which is mocked
 *
 * @return mocked value for specified argument
 *
 */
MOCK_RETURN_TYPE
_LwMockReturnParam
(
    const char   *const pFuncName,
    const char   *const pArg
)
{
    unsigned int funcLen = strlen(pFuncName);
    unsigned int argLen = strlen(pArg);
    char *str = (char *)malloc(sizeof(char) * (funcLen + argLen + 1));
    MOCK_RETURN_TYPE data = 0;
    strcpy(str, pFuncName);
    strcat(str, pArg);
    data = _LwMockReturn(str);
    free(str);
    return data;
}

/*
 * @brief Check if specified argument is mocked
 *
 * @param[in] pFuncName  The name of the function whose argument need to check
 * @param[in] pArg       The argument number which need to check
 *
 * @return 1 is mocked elser 0
 *
 */
int
_LwIsParamMocked
(
    const char   *const pFuncName,
    const char   *const pArg
)
{
    unsigned int funcLen = strlen(pFuncName);
    unsigned int argLen = strlen(pArg);
    char *str = (char *)malloc(sizeof(char) * (funcLen + argLen + 1));
    int data = 0;
    strcpy(str, pFuncName);
    strcat(str, pArg);
    data = _LwIsMocked(str);
    free(str);
    return data;
}

/*
 * @brief  Find the mock object corresponding to the object
 *         represented by the specified name. If a mock object
 *         does not exist, a new mock object will be created
 *         and inserted into the mock-object collection.
 *
 * @param[in] tc       The test-case to assert against upon error.
 * @param[in] pName    The name of the object to find the mock for.
 * @param[in] bCreate  Non-zero to create the mock when it does not exist
 *                     zero otherwise.
 *
 * @return A pointer to the mock for the specified object.
 */
static MOCK *
_mockGet
(
    LwTest *      tc,
    const char   *const pName,
    char          bCreate,
    char          bCheckParamMocked
)
{
    MOCK      *pMock = NULL;
    MOCK_NODE *pNode;
    unsigned int exHash;

    // search for the mock using the hash-value of the name
    exHash = _mockHash(pName);
    pNode  = pHeadNode;
    while (pNode != NULL)
    {
        if (pNode->hash == exHash || bCheckParamMocked)
        {
            if ((strcmp(pNode->pMock->pName, pName) == 0) ||
                (bCheckParamMocked && pNode->pMock->pParentFnName && (strcmp(pNode->pMock->pParentFnName, pName) == 0)))
            {
                break;
            }
        }
        pNode = pNode->pNext;
    }

    // create a mock if one does not already exist
    if (pNode == NULL)
    {
        if (bCreate != 0)
        {
            pMock = _mockCreate(tc, pName);
            _mockAddByHash(tc, pMock, exHash);
        }
    }
    else
    {
        pMock = pNode->pMock;
    }
    return pMock;
}

/*
 * @brief Create a mock to the object represented by the specific
 *        name.
 *
 * @param[in] tc     The test-case to assert against upon error.
 * @param[in] pName  The name of the object the mock represents.
 *
 * @return A pointer to the created mock object. This value may NEVER by NULL.
 *         An assertion will be raised against the test-case upon any failure.
 */
static MOCK *
_mockCreate
(
    LwTest *      tc,
    const char   *const pName
)
{
    MOCK *pMock;

    // allocate
    pMock = LW_ALLOC(MOCK);
    UNIT_ASSERT((pMock != NULL));

    pMock->pName = (char *)malloc((strlen(pName)+1)*sizeof(char));
    UNIT_ASSERT((pMock->pName != NULL));

    // populate
    strcpy(pMock->pName, pName);
    pMock->isMocked         = 0;
    pMock->pParentFnName    = NULL;
    pMock->pElementHead     = NULL;
    pMock->pElementTail     = NULL;
    pMock->pDefaultData     = 0;
    pMock->pFuncPointerData = NULL;

    return pMock;
}

/*
 * @brief Integer-based string hashing function (taken from Java).
 *        All characters in the string will be entered into the
 *        following equation to produce a 32-bit integer hash:
 *
 *        str[0]*(n-0) + str[1]*(n-1) + ... + str[n-1]*(n - (n - 1))
 *
 *        where:
 *        - str[i] is the ith character of the string
 *        - n is the length of the string
 *
 * @param[in] str : The string to hash
 *
 * @return unsigned integer hash equivalent of the string.
 */
static unsigned int
_mockHash
(
    const char *const str
)
{
    unsigned int val = 0;
    unsigned int n;
    unsigned int i;

    n = strlen(str);
    for (i = 0; i < n; i++)
    {
        val += (100 + str[i]) * (n-i);
    }
    return val;
}

/*
 * @brief Insert the specified mock-object into the
 *        mock-object collection.
 *
 * @param[in] tc     The test-case to assert against upon error.
 * @param[in] pMock  A pointer to the mock-object to insert.
 * @param[in] hash   The hash-value for the mock-object
 */
static void
_mockAddByHash
(
    LwTest      *tc,
    MOCK        *pMock,
    unsigned int hash
)
{
    MOCK_NODE *pNode;

    // allocate
    pNode = LW_ALLOC(MOCK_NODE);
    UNIT_ASSERT((pNode != NULL));

    // populate
    pNode->pMock = pMock;
    pNode->hash  = hash;

    // attach
    pNode->pNext = pHeadNode;
    pHeadNode = pNode;
}

/*
 * @brief Creates a new mock element for the data and
 *        appends to the end of the mock-object's
 *        element-list.
 *
 * @param[in] tc     The test-case to assert against upon error
 * @param[in] pMock  The mock-object at insert the element into
 * @param[in] pData  The data to represent in the mock-element
 * @param[in] count  The reference count for the element.  This determines the
 *                   duration in which the element will exist in the mock when
 *                   it is being accessed.
 */
static void
_mockInsertElement
(
    LwTest      *tc,
    MOCK        *pMock,
    const MOCK_RETURN_TYPE pData,
    const signed int   count
)
{
    MOCK_ELEMENT *pElement;

    // allocate
    pElement = LW_ALLOC(MOCK_ELEMENT);
    UNIT_ASSERT((pElement != NULL));

    // populate
    pElement->pData    = pData;
    pElement->refCount = count;
    pElement->pNext    = NULL;
    pMock->isMocked = 1;

    // attach
    if (pMock->pElementTail != NULL)
    {
        pMock->pElementTail->pNext = pElement;
        pMock->pElementTail = pElement;
    }
    else
    {
        pMock->pElementHead = pElement;
        pMock->pElementTail = pElement;
    }
}
