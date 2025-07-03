/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// Non-Hal public functions for PMGR
// pmgr.h
//
//*****************************************************

#ifndef _PMGR_H_
#define _PMGR_H_

/* ------------------------ Defines ---------------------------------------- */
#define PMGR_MUTEX_TIMEOUT_US (0x2000)

/* ------------------------ Includes --------------------------------------- */
#include "os.h"
#include "hal.h"

#include "g_pmgr_private.h"     // (rmconfig)  public interfaces

/* ------------------------ Static Variables ------------------------------- */

/* ------------------------ Function Prototypes ---------------------------- */
const char* pmgrGetEngineName    (void);

enum pmgrMutexPhysicalToLogical_GP100
{
    PMGR_MUTEX_ID_I2C_0,                          // 0
    PMGR_MUTEX_ID_I2C_1,                          // 1
    PMGR_MUTEX_ID_I2C_2,                          // 2
    PMGR_MUTEX_ID_I2C_3,                          // 3
    PMGR_MUTEX_ID_I2C_4,                          // 4
    PMGR_MUTEX_ID_I2C_5,                          // 5
    PMGR_MUTEX_ID_I2C_6,                          // 6
    PMGR_MUTEX_ID_I2C_7,                          // 7
    PMGR_MUTEX_ID_I2C_8,                          // 8
    PMGR_MUTEX_ID_I2C_9,                          // 9
    PMGR_MUTEX_ID_I2C_A,                          // 10
    PMGR_MUTEX_ID_I2C_B,                          // 11
    PMGR_MUTEX_ID_I2C_C,                          // 12
    PMGR_MUTEX_ID_I2C_D,                          // 13
    PMGR_MUTEX_ID_I2C_E,                          // 14
    PMGR_MUTEX_ID_I2C_F,                          // 15
    PMGR_MUTEX_ID_DISP_SCRATCH,                   // 16
    PMGR_MUTEX_ID_SEC2_EMEM_ACCESS,               // 17
    PMGR_MUTEX_ID_ILWALID,                        // 18
    PMGR_MUTEX_ID_ILWALID1,                       // 19
    PMGR_MUTEX_ID_ILWALID2,                       // 20
    PMGR_MUTEX_ID_ILWALID3,                       // 21
    PMGR_MUTEX_ID_ILWALID5,                       // 22
    PMGR_MUTEX_ID_ILWALID6,                       // 23
    PMGR_MUTEX_ID_ILWALID7,                       // 24
    PMGR_MUTEX_ID_ILWALID8,                       // 25
    PMGR_MUTEX_ID_ILWALID9,                       // 26
    PMGR_MUTEX_ID_ILWALID10,                      // 27
    PMGR_MUTEX_ID_ILWALID11,                      // 28
    PMGR_MUTEX_ID_ILWALID12,                      // 29
    PMGR_MUTEX_ID_ILWALID13,                      // 30
    PMGR_MUTEX_ID_ILWALID14,                      // 31
};                                             

#endif // _PMGR_H_
