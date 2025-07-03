/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwwatch.h"
#include "lwlink.h"

// Search cached discovery info for specific device instance
lwlDiscover *lwlinkDeviceInstanceInfoLookUp(lwlDiscover *pLwlDiscoverList, LwU32 linkId, LwU32 deviceType)
{
    lwlDiscover *pLwrr = pLwlDiscoverList;
    while (pLwrr != NULL)
    {
        if (pLwrr->ID == linkId)
        {
            if (pLwrr->deviceType == deviceType)
            {
                // Found requested device instance (linId/DeviceId combo)
                return pLwrr;
            }
            pLwrr = pLwrr->next;
        }
        else
        {
            pLwrr = pLwrr->next;
        }
    }

    // Not able to find requested device instance (linId/DeviceId combo)
    return 0;
}

void freeLwlDiscoveryList(lwlDiscover *pLwlDiscoverList)
{
    lwlDiscover *pTemp = pLwlDiscoverList;
    while (pTemp != NULL)
    {
        lwlDiscover *pDel = pTemp;
        pTemp = pTemp->next;
        free(pDel);
    }
}
