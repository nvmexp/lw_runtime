/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _RISCV_IO_DIO_H_
#define _RISCV_IO_DIO_H_

#include <lwstatus.h>
#include <lwtypes.h>
#include <print.h>

#include "riscv.h"
// #include "riscv_config.h"

#include "hal.h"
#include "g_riscv_hal.h"

// #include "lwsync_porting.h"

typedef enum
{
    DIO_TYPE_SE,
    DIO_TYPE_EXTRA
} DIO_TYPE;

#define DIO_TYPE_SNIC DIO_TYPE_EXTRA
#define DIO_TYPE_SNIC_PORT_IDX 0

static const char* DIO_TYPE_STR[] =
{
    "DIO_TYPE_SE",
    "DIO_TYPE_EXTRA"
};

typedef struct
{
    DIO_TYPE dioType;
    LwU32 portIdx;
} DIO_PORT;

typedef enum
{
    DIO_OPERATION_RD,
    DIO_OPERATION_WR
} DIO_OPERATION;

LW_STATUS riscvDioReadWrite(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 addr, LwU32 *pData);

#endif // _RISCV_IO_DIO_H_