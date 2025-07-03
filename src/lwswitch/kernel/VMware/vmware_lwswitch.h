/*******************************************************************************
    Copyright (c) 2022 LWPU Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/
#ifndef VMWARE_LWSWITCH_H
#define VMWARE_LWSWITCH_H

#include "lw-linux.h"

VMK_ReturnStatus lwswitch_pci_device_attach(lw_vmware_state_t *lwv);
void lwswitch_pci_device_detach(lw_vmware_state_t *lwv);
VMK_ReturnStatus lwswitch_register_device(lw_vmware_state_t *lwv, vmk_Device physical);
VMK_ReturnStatus lwswitch_ctl_init(void);
VMK_ReturnStatus lwswitch_init_device(lw_vmware_state_t *lwv);
void lwswitch_device_remove(lw_vmware_state_t *lwv);
void lwswitch_ctl_deinit(void);
VMK_ReturnStatus lwswitch_post_init_device(lw_vmware_state_t *lwv);
LwBool lwswitch_is_device_blacklisted(lw_vmware_state_t *lwv);
void lwswitch_post_init_blacklisted(lw_vmware_state_t *lwv);

#endif // VMWARE_LWSWITCH_H
