/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _RMASSERT_H_
#define _RMASSERT_H_

//this how the RM_ASSERT would look like
//#define RM_ASSERT(cond) logRmAssert(cond)

// log the condition which was to be asserted
void logRmAssert(LwBool hitStatus, char *file, LwU32 line);

//verify whether RM_ASSRT was hit or not
LwBool verifyRmAssertHit(LwBool status, LwU32 count);

#define UNIT_VERIFY_RM_ASSERT(status , count) UNIT_ASSERT(verifyRmAssertHit(status, count))

//
// that's a no-op
// kept only for the sake of backward compatibility
// for the tests already using it
//
void utApiMockRmAssert(LwU32 count);

// clears rmAssertNode head and tail
void clearRmAssertList();

#endif //_RMASSERT_H_
