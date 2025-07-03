/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwtypes.h"
#include "fixtureutil.h"
#include "utility.h"
#include "lwtest.h"

//
// list to keep track of
// RM_ASSERT being hit & when
//
struct unitRmAssertNode
{
    LwBool hitStatus;
    struct unitRmAssertNode *pNext;
};

typedef struct unitRmAssertNode rmAssertNode;

// pointer to keep track of head and tail of above list
static rmAssertNode *pRmAssertHead = NULL;
static rmAssertNode *pRmAssertTail = NULL;

//
// flag to determine whether RM_ASSERT verfication needs to be done
// will save exelwtion time for other tests which dont care about
// RM_ASSERT hitting
//
static LwBool verifEnabled = LW_FALSE;

/*
 * @brief sets the value of above variable to true
 *
 * @param[in] tc       Pointer to test case object
 *
 * @param[in] verify   Verify Function which would be Exelwted
 *                     if RM_AASERTS fails and we move out of
 *                     Test Case. In that case there would be
 *                     no time to execute the verify section of
 *                     the test case.
 *                     In Such a case, verfiy function would be
 *                     exelwted after the test case which eould hold
 *                     the verif logic
 *
 */
void utApiEnableRmAssertVerification(LwTest *tc, TestFunction verify)
{
    // update the verify function
    tc->verify = verify;

    verifEnabled = LW_TRUE;
}

/*
 * @brief clears rmAssertNode head and tail
 *
 */
void clearRmAssertList()
{
    pRmAssertHead = NULL;
    pRmAssertTail = NULL;
    verifEnabled = LW_FALSE;

}

/*
 * @brief log the condition which was to be asserted
 *
 * @param[in] hitStatus  The condition value as evaluated in assert
 *
 * @param[in] file       file name where assert was hit
 *
 * @param[in] line       Line on whihc Assert was hit
 */
void logRmAssert(LwBool hitStatus, char *file, LwU32 line)
{

    rmAssertNode *pNode = NULL;

    if (!verifEnabled)
    {
        if (!hitStatus)
            failDueToRmAssert(file, line);
        return;
    }

    pNode = (rmAssertNode *)unitMalloc(sizeof(rmAssertNode), UNIT_CLASS_MISC);

    pNode->hitStatus = hitStatus;

    // list already present
    if (pRmAssertHead)
    {
        pRmAssertTail->pNext = pNode;
        pRmAssertTail = pNode;
    }

    // create the list as in, init the head & tail
    else
    {
        pRmAssertHead = pNode;
        pRmAssertTail = pNode;
    }

    if (! hitStatus)
        returnDueToExpectedRmAssert();
}

/*
 * @brief verify whether RM_ASSERT was hit or not
 *
 * @param[in] status  evaluateed value LW_TRUE/LW_FALSE
 *
 * @param[in] count   Nth RM_ASSERT oclwrence in the SUT
 */
LwBool verifyRmAssertHit(LwBool status, LwU32 count)
{
    rmAssertNode *pNode = pRmAssertHead;
    LwU32 verifCount = 0;

    // count cant be zero
    UNIT_ASSERT( count != 0);

    // there is no list present, fail
    if (!pRmAssertHead)
    {
        return FALSE;
    }

    while (pNode)
    {
        verifCount += 1;

        if (verifCount == count)
        {
            return ( pNode->hitStatus == status );
        }
        else
        {
            pNode = pNode->pNext;
        }
    }

    // node cant be found, fail
    return FALSE;
}

/*
 * @brief that's a no-op
 *        kept only for the sake of backward compatibility
 *        for the tests already using it
 *
 * @param[in] count   Nth RM_ASSERT oclwrence in the SUT
 */
void utApiMockRmAssert(LwU32 count)
{
    // assuming tests, already using it, will have "count" >= 0
    verifEnabled = !!count;
    return ;
}

