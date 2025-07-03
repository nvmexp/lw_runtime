/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "os.h"

void ScanLWTopology(LwU32 base_address)
{
    fprintf(stderr, "STUB: %s()\n", __FUNCTION__);
}

LW_STATUS readVirtMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *read)
{
    if (read != NULL)
        *read = 0;
    fprintf(stderr, "STUB: %s()\n", __FUNCTION__);
    return LW_ERR_GENERIC;
}

BOOL isValidClassHeader(LwU32 classNum)
{
    fprintf(stderr, "STUB: %s()\n", __FUNCTION__);
    return FALSE;
}

BOOL parseClassHeader(LwU32 classnum, LwU32 method, LwU32 data)
{
    fprintf(stderr, "STUB: %s()\n", __FUNCTION__);
    return FALSE;
}
