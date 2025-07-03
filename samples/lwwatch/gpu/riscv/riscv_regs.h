/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _RISCV_REGS_H
#define _RISCV_REGS_H

/*
 * This file contains selection of lwwatch-related registers that are not
 * lwrrently contained in the hardware reference manuals.
 */

#define LW_FALCON_SEC_BASE                            0x00840000

#define LW_FALCON_IDLESTATE_FBIF_BUSY                 1:1
#define LW_FALCON_IDLESTATE_FBIF_BUSY_TRUE            0x00000001
#define LW_FALCON_IDLESTATE_FBIF_BUSY_FALSE           0x00000000

#define LW_RISCV_CSR_MSTATUS_VM_MPU                   0x1e

//
// @todo  Wait for dev_riscv_pri.ref to include this.
//        It's falcon2 base (0x400) on top of minion falcon base (0xa04000)
//        bug 200603404 tracking
//
#define LW_FALCON2_MINION_BASE 0xA04400

#endif // _RISCV_REGS_H
