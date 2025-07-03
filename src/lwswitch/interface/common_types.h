/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * Common defines, types and declarations shared between kernel mode and user mode.
 *
 */

#ifndef _COMMON_TYPES_H_
#define _COMMON_TYPES_H_

#ifdef __cplusplus
extern "C"
{
#endif

// LwSwitch Control device name and symbolic name for IOCTL calls on Windows.
#define LWSWITCH_CONTROL_DEVICE_NAME                L"\\Device\\LwidiaLwSwitchControl"
#define LWSWITCH_CONTROL_DEVICE_SYMBOLIC_NAME       L"\\DosDevices\\LwidiaLwSwitchControl"
#define LWSWITCH_CONTROL_DEVICE_FILE_NAME            "\\\\.\\LwidiaLwSwitchControl"


// LwSwitch device name and symbolic name for IOCTL calls on Windows.
#define LWSWITCH_DEVICE_NAME_N                      L"\\Device\\LwidiaLwSwitchDevice%u"
#define LWSWITCH_DEVICE_SYMBOLIC_NAME_N             L"\\DosDevices\\LwidiaLwSwitchDevice%u"
#define LWSWITCH_DEVICE_FILE_NAME_N                  "\\\\.\\LwidiaLwSwitchDevice%u"

#ifdef __cplusplus
}
#endif

#endif //_COMMON_TYPES_H_
