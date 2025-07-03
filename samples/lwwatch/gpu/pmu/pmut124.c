/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// lwwatch WinDbg Extension for PMU
// pmut124.c
//
//*****************************************************

//
// Includes
//
#include "pmu.h"

#include "g_pmu_private.h"     // (rmconfig)  implementation prototypes

//////////////////////////////////////////////////////////////////////////////
// PMU Sanity Tests
//////////////////////////////////////////////////////////////////////////////

// Prototypes

static PmuSanityTestEntry PmuSanityTests_T124[] =
{
    // Reset Test
    {   
        pmuSanityTest_Reset_GK104, 
        PMU_TEST_DESTRUCTIVE,
        "Reset Test"
    },
    // Mutex and ID Generator Test
    {
        pmuSanityTest_MutexIdGen_GK104,
        PMU_TEST_AUTO,
        "Mutex and ID Generator Test"
    }, 
    // BAR0 Master Test
    {
        pmuSanityTest_Bar0Master_GK104,
        PMU_TEST_AUTO,
        "Bar0 Master Test"
    },
    // BAR0 FECS Test
    {
        pmuSanityTest_Bar0FECS_GK104,
        PMU_TEST_AUTO,
        "Bar0 FECS Test"
    },
};

#define PMU_SANITY_TEST_NUM (sizeof(PmuSanityTests_T124) / sizeof(PmuSanityTestEntry))

#define BAILOUT(cond, label) \
    do {status = cond; goto label;} while(0)

//
// PMU Sanity Test Cases
// 

/*!
 *  @returns the number of PMU sanity tests available.
 */
LwU32 
pmuSanityTestGetNum_T124
(
    void
)
{
    return PMU_SANITY_TEST_NUM;
}


/*!
 *  @returns test table 
 */
void *
pmuSanityTestGetEntryTable_T124()
{
    return (void *) PmuSanityTests_T124;
}
