/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   fixtureutil.c
 * @brief  utility functions for allocated objects linked list management
 */

#include "fixtureutil.h"
#include "odbinfra.h"
#include "utility.h"

//  var pointing to head & tail of Book-keeping structures
rmInfoBlockList *rmInfoBlockListHead = NULL;
rmObjectList *rmObjectListHead = NULL;

rmInfoBlockList *rmInfoBlockListTail = NULL;
rmObjectList *rmObjectListTail = NULL;

rmMiscNodeList *rmMiscNodeListHead = NULL;
rmMiscNodeList *rmMiscNodeListTail = NULL;

/*!
 * @brief Adds the object viz,OBJXXX to the Object list
 *
 * @param[in]      pData    pointer to the rm object
 */
void addToRmOjectList(void *pData)
{
    rmObjectList *pNode = (rmObjectList *)malloc(sizeof(rmObjectList));
    pNode->pData = pData;
    pNode->pNext = NULL;

    // if head null then create head
    if (rmObjectListHead == NULL)
    {
        rmObjectListHead = pNode;
        rmObjectListTail = pNode;
    }
    // append to the tail
    else
    {
        rmObjectListTail->pNext = pNode;
        rmObjectListTail = pNode;
    }
}

/*!
 * @brief Adds the allocated infoblock to the list
 *
 * @param[in]      pData    pointer to the eng info block
 */
void addToRmInfoBlockList(void *pData)
{
    rmInfoBlockList *pNode = (rmInfoBlockList *)malloc(sizeof(rmInfoBlockList));
    pNode->pData = pData;
    pNode->pNext = NULL;

    // if head null then create head
    if (rmInfoBlockListHead == NULL)
    {
        rmInfoBlockListHead = pNode;
        rmInfoBlockListTail = pNode;
    }
    // append to the tail
    else
    {
        rmInfoBlockListTail->pNext = pNode;
        rmInfoBlockListTail = pNode;
    }
}

/*!
 * @brief Adds all other allocated structures to this list
 *
 * @param[in]      pData    pointer to the allocated memory
 */
void addToRmMiscNodeList(void *pData)
{
    rmMiscNodeList *pNode = (rmMiscNodeList *)malloc(sizeof(rmMiscNodeList));
    pNode->pData = pData;
    pNode->pNext = NULL;

    // if head null then create head
    if (rmMiscNodeListHead == NULL)
    {
        rmMiscNodeListHead = pNode;
        rmMiscNodeListTail = pNode;
    }
    // append to the tail
    else
    {
        rmMiscNodeListTail->pNext = pNode;
        rmMiscNodeListTail = pNode;
    }
}

/*!
 * @brief free the rm object list
 */
static void destroyRmOjectList()
{
    rmObjectList *pNode = rmObjectListHead;

    while (rmObjectListHead != NULL)
    {
        rmObjectListHead = pNode->pNext;
        free(pNode->pData);
        free(pNode);
        pNode = rmObjectListHead;
    }
}

/*!
 * @brief free the infoblock list
 */
static void destroyRmInfoBlockList()
{
    rmInfoBlockList *pNode = rmInfoBlockListHead;

    while (rmInfoBlockListHead != NULL)
    {
        rmInfoBlockListHead = pNode->pNext;
        free(pNode->pData);
        free(pNode);
        pNode = rmInfoBlockListHead;
    }
}

/*!
 * @brief free the other structures list
 */
static void destroyRmMiscNodeList()
{
    rmMiscNodeList *pNode = rmMiscNodeListHead;

    while (rmMiscNodeListHead != NULL)
    {
        rmMiscNodeListHead = pNode->pNext;
        free(pNode->pData);
        free(pNode);
        pNode = rmMiscNodeListHead;
    }
}

/*!
 * @brief destroy all lists
 */
void destroyAllList()
{
    destroyRmOjectList();
    destroyRmInfoBlockList();
    destroyRmMiscNodeList();

    memset(&unitTestRmObject, 0, sizeof(unitTestRmObject));
    memset(&unitTestRmInfoBlock, 0, sizeof(unitTestRmInfoBlock));
    resetMissingEngineBlock();

    // make NULL all rm objrct ptrs
    UNIT_ASSERT(getObjectMock(NULL, ODB_CLASS_UNKNOWN, 0xDEADBEEF, 0xDEADBEEF) == NULL);

    //make NULL all ptr to rm infoblks
    UNIT_ASSERT(getInfloblockStub(NULL, DATA_ID_FREE_OBJ) == NULL);

    // clears rmAssertNode head and tail
    clearRmAssertList();
}

/*!
 * @brief Unit Test Infra specific memeory allocation
 *
 * @param[in]      size           size of memory to be allocated
 *
 * @param[in]      uClass         type of object ;rm Object, infoblock or misc
 *
 * @return         Pointer to allocated memory
 */
void * unitMalloc(LwU32 size, UNIT_CLASS uClass)
{
    void *pMem = (void *)malloc(size);

    switch(uClass)
    {
    case UNIT_CLASS_RM_OBJECT: addToRmOjectList(pMem);
                               break;

    case UNIT_CLASS_INFOBLK  : addToRmInfoBlockList(pMem);
                               break;

    default                  : addToRmMiscNodeList(pMem);
                               break;
    }

    return pMem;
}
