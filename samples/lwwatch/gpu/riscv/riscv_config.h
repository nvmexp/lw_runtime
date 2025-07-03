/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _RISCV_CONFIG_H_
#define _RISCV_CONFIG_H_

#define FALCON_PAGE_SIZE        0x100
#define FALCON_PAGE_MASK        0xFF

#define RISCV_MAX_UPLOAD            0x10000
#define RISCV_DEFAULT_COREDUMP_SIZE 8192

#define TARGET_DEFAULT_TIMEOUT_MS   5000
#define NETWORK_BUFFER_SIZE         4096U
#define MONITOR_BUF_SIZE            10000U

#define RISCV_ICD_TIMEOUT_MS        100
#define RISCV_ICD_POLL_MAX_MS       250
#define RISCV_SCRUBBING_TIMEOUT_MS  5000
#define RISCV_AUTO_STEP_MAX         32

typedef struct
{
    int bPrintMemTransactions; // Dump memory read/write requests
    int bPrintGdbCommunication; // Dump GDB communication
    int bPrintTargetCommunication; // Dump ICD/IBRKPT operations
} RISCV_CONFIG;

extern RISCV_CONFIG config;

#endif
