/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LWSWITCH_VMKMGMT_H_
#define _LWSWITCH_VMKMGMT_H_

#include <vmkapi.h>

#define LWSWITCH_KERNEL_OPEN   (VMK_MGMT_RESERVED_CALLBACKS + 1)
#define LWSWITCH_KERNEL_CLOSE  (VMK_MGMT_RESERVED_CALLBACKS + 2)
#define LWSWITCH_KERNEL_IOCTL  (VMK_MGMT_RESERVED_CALLBACKS + 3)

/*
 * The total number of callbacks (both user and kernel).
 */

#define NUM_CALLBACKS         3     // open(), close(), ioctl().
#define MGMT_INTERFACE_NAME   "LwswitchVmkMgmt"
#define MGMT_INTERFACE_VENDOR "LWPU"

#ifdef __VMKERNEL__

int lwswitch_ctl_open(vmk_MgmtCookies     *lwvCookie,
                    vmk_MgmtElwelope    *lwvElwelope);

int lwswitch_ctl_close(vmk_MgmtCookies    *lwvCookie,
                     vmk_MgmtElwelope   *lwvElwelope);

int lwswitch_ctl_ioctl(vmk_MgmtCookies    *lwvCookie,
                     vmk_MgmtElwelope   *lwvElwelope,
                     vmk_uint32         *cmd,
                     vmk_MgmtVectorParm *cmdParam);
#else
/*
 * Kernel-run callbacks are defined as NULL for the user-
 * compiled portion of the interface.
 */
#define lwswitch_ctl_open           NULL
#define lwswitch_ctl_close          NULL
#define lwswitch_ctl_ioctl          NULL

#endif /* __VMKERNEL__ */

vmk_MgmtCallbackInfo lwswitchMgmtCallbacks[NUM_CALLBACKS] = {
   {
       .location    = VMK_MGMT_CALLBACK_KERNEL,
       .callback    = lwswitch_ctl_open,
       .synchronous = 1, /* 0 indicates asynchronous */
       .callbackId  = LWSWITCH_KERNEL_OPEN,
   },

   {
       .location    = VMK_MGMT_CALLBACK_KERNEL,
       .callback    = lwswitch_ctl_close,
       .synchronous = 1, /* 0 indicates asynchronous */
       .callbackId  = LWSWITCH_KERNEL_CLOSE,
   },

   {
       .location    = VMK_MGMT_CALLBACK_KERNEL,
       .callback    = lwswitch_ctl_ioctl,
       .synchronous = 1, /* 0 indicates asynchronous */
       .numParms    = 2,
        /*
            vmk_MgmtVectorParm isn't the structure to pass to ioctl, it's
            typically the structure that gets passed in the ioctl call, and
            the total size. The vmkAPI layer then checks that total size is
            a multiple of the structure that's passed in. Passing in
            sizeof(char) inhibits the warning, as it's always a multiple.
        */
       .parmSizes   = { sizeof(vmk_uint32), /* ioctl cmd */
                        sizeof(char)},      /* ioctl cmd parameter  */
       .parmTypes   = { VMK_MGMT_PARMTYPE_IN, /* cmd is an input parameter */
                        VMK_MGMT_PARMTYPE_VECTOR_INOUT}, /* ioctl cmd parameters*/
       .callbackId  = LWSWITCH_KERNEL_IOCTL,
   },
};

vmk_MgmtApiSignature lwswitchMgmtSignature = {
   .version        = VMK_REVISION_FROM_NUMBERS(1,0,0,0),
   .name.string    = MGMT_INTERFACE_NAME,
   .vendor.string  = MGMT_INTERFACE_VENDOR,
   .numCallbacks   = NUM_CALLBACKS,
   .callbacks      = lwswitchMgmtCallbacks,
};

#endif
