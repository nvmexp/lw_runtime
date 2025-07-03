/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "riscv_io_dio.h"

LW_STATUS riscvDioReadWrite(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 addr, LwU32 *pData)
{
    return pRiscv[indexGpu].riscvDioReadWrite(&dioPort, &dioOp, addr, pData);
}
