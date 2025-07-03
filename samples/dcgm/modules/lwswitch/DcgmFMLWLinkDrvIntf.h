#pragma once

#include <queue>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

/*****************************************************************************/
/*  Fabric Manager : Abstract the LWLink Driver interface/ioctl              */
/*****************************************************************************/

class DcgmFMLWLinkDrvIntf
{
public:
    DcgmFMLWLinkDrvIntf();
    ~DcgmFMLWLinkDrvIntf();

    int doIoctl(int ioctlCmd, void *ioctlParam);

private:
    int isIoctlCmdSupported(int ioctlCmd);
    void logIoctlError(int ioctlCmd, void *ioctlParam);
    const char* getIoctlCmdString(int ioctlCmd);
    int mDrvHandle;
};

