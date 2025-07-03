/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEIFSPI_H_
#define _SOEIFSPI_H_

enum
{
    RM_SOE_SPI_INIT,
};


typedef union
{
    LwU8    cmdType;
}RM_SOE_SPI_CMD;

#endif // _SOEIFSPI_H_
