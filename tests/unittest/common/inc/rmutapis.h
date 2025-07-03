/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RMUTAPIS_H
#define _RMUTAPIS_H

#include "rmutmock.h"

#define TC_EXIT 1
#define TC_NO_EXIT 0

//
// Suite specific Public APIS
//

// Create a new suite
#define UTAPI_NEW_SUITE(setup, teardown)                    LwSuiteNew(setup, teardown, __FUNCTION__)

// Add a child suite to its parent
#define UTAPI_AddSuite(parent, child)                       LwSuiteAddSuite(parent, child)

// Spew detail info of the suite
#define UTAPI_SuiteDetailInfo(suite, buffer)                LwSuiteDetails(suite, buffer)

// Exute the suite , means execute all suite, tests inside the suite
#define UTAPI_ExelwteSuite(suite, bLogVerbose)                           LwSuiteRun(suite, bLogVerbose)

// Add test case to the suite
#define UTAPI_ADD_TC(SUITE, SETUPFN, TEST, TEARDOWNFN)      LwSuiteAddTest(SUITE, LwTestNew(#TEST, (TestFunction)SETUPFN, (TestFunction)TEST, (TestFunction)TEARDOWNFN))

// Skip all tests under the suite
#define UTAPI_SkipAllTests(suite)                           SkipAllTests(suite)

// Destroy the suite, in a way do the cleanup.
#define UTAPI_DestroySuite(suite)                           DestroySuite(suite)

// Assert specific Public APIS, it exit and mark test case as "failed" if assert fail
#define UTAPI_FAIL(tc, ms)                              LwFail((tc), __FILE__, __LINE__, NULL, (ms), TC_EXIT)
#define UTAPI_ASSERT(tc, ms, cond)                      LwAssert((tc), __FILE__, __LINE__, (ms), (cond), TC_EXIT)
#define UTAPI_ASSERT_TRUE(tc, cond)                     LwAssert((tc), __FILE__, __LINE__, "assert failed", (cond), TC_EXIT)
#define UTAPI_ASSERT_STR_EQUALS(tc,ex,ac)               LwAssertStrEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac), TC_EXIT)
#define UTAPI_ASSERT_STR_EQUALS_MSG(tc,ms,ex,ac)        LwAssertStrEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac), TC_EXIT)
#define UTAPI_ASSERT_INT_EQUALS(tc,ex,ac)               LwAssertIntEquals((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(ex),(MOCK_RETURN_TYPE)(ac), TC_EXIT)
#define UTAPI_ASSERT_INT_EQUALS_MSG(tc,ms,ex,ac)        LwAssertIntEquals((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(ex),(MOCK_RETURN_TYPE)(ac), TC_EXIT)
#define UTAPI_ASSERT_DBL_EQUALS(tc,ex,ac,dl)            LwAssertDblEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac),(dl), TC_EXIT)
#define UTAPI_ASSERT_DBL_EQUALS_MSG(tc,ms,ex,ac,dl)     LwAssertDblEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac),(dl), TC_EXIT)
#define UTAPI_ASSERT_PTR_EQUALS(tc,ex,ac)               LwAssertPtrEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac), TC_EXIT)
#define UTAPI_ASSERT_PTR_EQUALS_MSG(tc,ms,ex,ac)        LwAssertPtrEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac), TC_EXIT)
#define UTAPI_ASSERT_PTR_NOT_NULL(tc,p)                 LwAssert((tc),__FILE__,__LINE__,"null pointer unexpected",(p != NULL), TC_EXIT)
#define UTAPI_ASSERT_PTR_NOT_NULL_MSG(tc,msg,p)         LwAssert((tc),__FILE__,__LINE__,(msg),(p != NULL), TC_EXIT)
#define UTAPI_ASSERT_GREATER(tc,v1,v2)                  LwAssertGreater((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_GREATER_MSG(tc,ms,v1,v2)           LwAssertGreater((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_GREATER_OR_EQUALS(tc,v1,v2)        LwAssertGreaterOrEquals((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_GREATER_OR_EQUALS_MSG(tc,ms,v1,v2) LwAssertGreaterOrEquals((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_LESS(tc,v1,v2)                     LwAssertLess((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_LESS_MSG(tc,ms,v1,v2)              LwAssertLess((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_LESS_OR_EQUALS(tc,v1,v2)           LwAssertLessOrEquals((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_LESS_OR_EQUALS_MSG(tc,ms,v1,v2)    LwAssertLessOrEquals((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_EXIT)
#define UTAPI_ASSERT_CALL_COUNT(tc,ex,ac)               LwAssertIntEquals((tc),__FILE__,__LINE__,"unexpected number of calls to function",(MOCK_RETURN_TYPE)(ex),(MOCK_RETURN_TYPE)(ac), TC_EXIT)
#define UTAPI_ASSERT_ARRAY_EQUALS(tc,ex,ac,c)           LwAssertArrayEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac),(c), TC_EXIT)
#define UTAPI_ASSERT_ARRAY_EQUALS_MSG(tc,ms,ex,ac,c)    LwAssertArrayEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac),(c), TC_EXIT)
#define UTAPI_ASSERT_PTR_NULL(tc,p)                     LwAssert((tc),__FILE__,__LINE__,"null pointer expected",(p == NULL), TC_EXIT)
#define UTAPI_ASSERT_PTR_NULL_MSG(tc,msg,p)             LwAssert((tc),__FILE__,__LINE__,(msg),(p == NULL), TC_EXIT)
#define UTAPI_ASSERT_RM_ASSERT_MSG(tc,msg,status,count) LwVerifRmAssert((tc),__FILE__,__LINE__,(msg),status, count, TC_EXIT)
#define UTAPI_ASSERT_RM_ASSERT(tc,status,count)         LwVerifRmAssert((tc),__FILE__,__LINE__,NULL,status, count, TC_EXIT)
#define UTAPI_LWSTOM_ASSERT(tc, func, arg)              LwLwstomAssert((tc), __FILE__, __LINE__, (LwstomAssertFunc)func, (void *)arg, NULL, TC_EXIT);
#define UTAPI_LWSTOM_ASSERT_MSG(tc, func, arg, msg)     LwLwstomAssert((tc), __FILE__, __LINE__, (LwstomAssertFunc)func, (void *)arg, msg, TC_EXIT);

// Verif Specific Public APIS, it just mark test as "failed" but not exit.
#define UTAPI_VERIF(tc, ms, cond)                       LwAssert((tc), __FILE__, __LINE__, (ms), (cond), TC_NO_EXIT)
#define UTAPI_VERIF_TRUE(tc, cond)                      LwAssert((tc), __FILE__, __LINE__, "verif failed", (cond), TC_NO_EXIT)
#define UTAPI_VERIF_STR_EQUALS(tc,ex,ac)                LwAssertStrEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac), TC_NO_EXIT)
#define UTAPI_VERIF_STR_EQUALS_MSG(tc,ms,ex,ac)         LwAssertStrEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac), TC_NO_EXIT)
#define UTAPI_VERIF_INT_EQUALS(tc,ex,ac)                LwAssertIntEquals((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(ex),(MOCK_RETURN_TYPE)(ac), TC_NO_EXIT)
#define UTAPI_VERIF_INT_EQUALS_MSG(tc,ms,ex,ac)         LwAssertIntEquals((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(ex),(MOCK_RETURN_TYPE)(ac), TC_NO_EXIT)
#define UTAPI_VERIF_DBL_EQUALS(tc,ex,ac,dl)             LwAssertDblEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac),(dl), TC_NO_EXIT)
#define UTAPI_VERIF_DBL_EQUALS_MSG(tc,ms,ex,ac,dl)      LwAssertDblEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac),(dl), TC_NO_EXIT)
#define UTAPI_VERIF_PTR_EQUALS(tc,ex,ac)                LwAssertPtrEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac), TC_NO_EXIT)
#define UTAPI_VERIF_PTR_EQUALS_MSG(tc,ms,ex,ac)         LwAssertPtrEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac), TC_NO_EXIT)
#define UTAPI_VERIF_PTR_NOT_NULL(tc,p)                  LwAssert((tc),__FILE__,__LINE__,"null pointer unexpected",(p != NULL), TC_NO_EXIT)
#define UTAPI_VERIF_PTR_NOT_NULL_MSG(tc,msg,p)          LwAssert((tc),__FILE__,__LINE__,(msg),(p != NULL), TC_NO_EXIT)
#define UTAPI_VERIF_GREATER(tc,v1,v2)                   LwAssertGreater((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_GREATER_MSG(tc,ms,v1,v2)            LwAssertGreater((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_GREATER_OR_EQUALS(tc,v1,v2)         LwAssertGreaterOrEquals((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_GREATER_OR_EQUALS_MSG(tc,ms,v1,v2)  LwAssertGreaterOrEquals((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_LESS(tc,v1,v2)                      LwAssertLess((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_LESS_MSG(tc,ms,v1,v2)               LwAssertLess((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_LESS_OR_EQUALS(tc,v1,v2)            LwAssertLessOrEquals((tc),__FILE__,__LINE__,NULL,(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_LESS_OR_EQUALS_MSG(tc,ms,v1,v2)     LwAssertLessOrEquals((tc),__FILE__,__LINE__,(ms),(MOCK_RETURN_TYPE)(v1),(MOCK_RETURN_TYPE)(v2), TC_NO_EXIT)
#define UTAPI_VERIF_CALL_COUNT(tc,ex,ac)                LwAssertIntEquals((tc),__FILE__,__LINE__,"unexpected number of calls to function",(MOCK_RETURN_TYPE)(ex),(MOCK_RETURN_TYPE)(ac), TC_NO_EXIT)
#define UTAPI_VERIF_ARRAY_EQUALS(tc,ex,ac,c)            LwAssertArrayEquals((tc),__FILE__,__LINE__,NULL,(ex),(ac),(c), TC_NO_EXIT)
#define UTAPI_VERIF_ARRAY_EQUALS_MSG(tc,ms,ex,ac,c)     LwAssertArrayEquals((tc),__FILE__,__LINE__,(ms),(ex),(ac),(c), TC_NO_EXIT)
#define UTAPI_VERIF_PTR_NULL(tc,p)                      LwAssert((tc),__FILE__,__LINE__,"null pointer expected",(p == NULL), TC_NO_EXIT)
#define UTAPI_VERIF_PTR_NULL_MSG(tc,msg,p)              LwAssert((tc),__FILE__,__LINE__,(msg),(p == NULL), TC_NO_EXIT)
#define UTAPI_VERIFY_RM_ASSERT(tc,status,count)         LwVerifRmAssert((tc),__FILE__,__LINE__, NULL,status, count, TC_NO_EXIT)
#define UTAPI_VERIFY_RM_ASSERT_MSG(tc,msg,status,count) LwVerifRmAssert((tc),__FILE__,__LINE__,(msg),status, count, TC_NO_EXIT)
#define UTAPI_LWSTOM_VERIF(tc, func, arg)               LwLwstomAssert((tc), __FILE__, __LINE__, (LwstomAssertFunc)func, (void *)arg, NULL, TC_NO_EXIT)
#define UTAPI_LWSTOM_VERIF_MSG(tc, func, arg, msg)      LwLwstomAssert((tc), __FILE__, __LINE__, (LwstomAssertFunc)func, (void *)arg, msg, TC_NO_EXIT)

// String specific Public APIS
#define UTAPI_NewString                              LwStringNew
#define UTAPI_InitString(str)                        LwStringInit(str)

#endif // _RMUTAPIS_H

