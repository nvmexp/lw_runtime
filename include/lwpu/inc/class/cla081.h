/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the writte
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cla081_h_
#define _cla081_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWA081_VGPU_CONFIG                                  (0x0000a081)

/*event values*/
#define LWA081_NOTIFIERS_EVENT_VGPU_GUEST_CREATED               (0)
#define LWA081_NOTIFIERS_EVENT_VGPU_GUEST_INITIALISING          (1)
#define LWA081_NOTIFIERS_EVENT_VGPU_GUEST_DESTROYED             (2)
#define LWA081_NOTIFIERS_EVENT_VGPU_GUEST_LICENSE_ACQUIRED      (3)
#define LWA081_NOTIFIERS_EVENT_VGPU_GUEST_LICENSE_STATE_CHANGED (4)
#define LWA081_NOTIFIERS_MAXCOUNT                               (5)


#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cla081_h
