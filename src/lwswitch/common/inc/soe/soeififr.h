/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEIFIFR_H_
#define _SOEIFIFR_H_

#include "flcnifcmn.h"

#define INFOROM_FS_FILE_NAME_SIZE       3

enum
{
	RM_SOE_IFR_READ,
	RM_SOE_IFR_WRITE,
};

typedef struct
{
    LwU8        cmdType;
    RM_FLCN_U64 dmaHandle;
    LwU32       offset;
    LwU32       sizeInBytes;
    char        fileName[INFOROM_FS_FILE_NAME_SIZE];
} RM_SOE_IFR_CMD_PARAMS;

typedef union
{
	LwU8	cmdType;
	RM_SOE_IFR_CMD_PARAMS params;
} RM_SOE_IFR_CMD;

#endif // _SOEIFIFR_H_
