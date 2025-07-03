/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/types.h>
#include "mods.h"

using namespace std;

int main(int argc, char **argv)
{
    // Get the min. supported version from cmdline
    if (argc != 3)
    {
        cout<<"Usage: ./version_checker <major> <minor>"<<endl;
        return 2;
    }

    __u64 min_supported_version_major = strtol(argv[1], NULL, 16);
    __u64 min_supported_version_minor = strtol(argv[2], NULL, 16);
    __u64 min_supported_version = (min_supported_version_major << 8) | min_supported_version_minor;

    const char s_DevMods[] = "/dev/mods";

    // Open the driver
    int krnFd = open(s_DevMods, O_RDWR);
    if (krnFd == -1)
    {
        cout<<"Error: Unable to open "<<s_DevMods<<". Another instance of MODS may be running"<<endl;
        return 2;
    }

    // Get the installed MODS kernel driver version
    MODS_GET_VERSION modsGetVersion = {0};
    const int ret = ioctl(krnFd, MODS_ESC_GET_API_VERSION, &modsGetVersion);
    if (ret)
    {
        cout<<"Error: Can't get MODS kernel module API version"<<endl;
        return 2;
    }

    __u64 version = modsGetVersion.version;

#ifdef DEBUG
    cout<<"Current version: 0x"<<hex<<version<<endl;
#endif

    if (close(krnFd) == -1)
    {
        cout<<"Error: Unable to close "<<s_DevMods<<endl;
        return 2;
    }

    // Compare with version from driver.tgz
#ifdef DEBUG
    cout<<"Min. supported version: 0x"<<hex<<min_supported_version<<endl;
#endif
    
    if (version < min_supported_version)
    {
        return 1;
    }

    return 0;
}
