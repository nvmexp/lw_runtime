/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include <queue>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

extern "C"
{
    #include "lwlink_user_api.h"
}


/*****************************************************************************/
/*  Fabric Manager : Abstract the LWLink Driver interface/ioctl              */
/*****************************************************************************/

class LocalFMLWLinkDrvIntf
{
public:
    LocalFMLWLinkDrvIntf();
    ~LocalFMLWLinkDrvIntf();

    int doIoctl(int ioctlCmd, void *ioctlParam, int paramSize);

private:
    int isIoctlCmdSupported(int ioctlCmd);
    void logIoctlError(int ioctlCmd, void *ioctlParam);
    void acquireFabricManagementCapability(void);

    lwlink_session *mpLWLinkDrvSession;
};

