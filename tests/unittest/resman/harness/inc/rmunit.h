/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright by LWPU Corporation.  All rights reserved. All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   rmunit.h
 * @brief  wrapper for headers required for resman unit testing
 */

#ifndef _RMUNIT_H_
#define _RMUNIT_H_

#include <stdarg.h>
#include "rmutapis.h"
#include "odbinfra.h"
#include "unitodb.h"
#include "regops.h"
#include "rmassert.h"

//
// Enable verification of RM_ASSERTS being hit
// also provide a verif function whihc would be called after
// when assert hits and we move out of test method
//
void utApiEnableRmAssertVerification(LwTest *tc, TestFunction verify);
#endif // _RMUNIT_H_

