/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2014 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl503b_h_
#define _cl503b_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define   LW50_P2P                                      (0x0000503b)

#define LW503B_FLAGS_P2P_TYPE            0:0
#define LW503B_FLAGS_P2P_TYPE_GPA        0
#define LW503B_FLAGS_P2P_TYPE_SPA        1

/* LwRmAlloc parameters */
typedef struct {
    LwHandle hSubDevice;                /* subDevice handle of local GPU              */
    LwHandle hPeerSubDevice;            /* subDevice handle of peer GPU               */
    LwU32    subDevicePeerIdMask;       /* Bit mask of peer ID for SubDevice
                                         * A value of 0 defaults to RM selected
                                         * PeerIdMasks must match in loopback         */
    LwU32    peerSubDevicePeerIdMask;   /* Bit mask of peer ID for PeerSubDevice
                                         * A value of 0 defaults to RM selected
                                         * PeerIdMasks must match in loopback         */
    LwU64    mailboxBar1Addr;           /* P2P Mailbox area base offset in BAR1
                                         * Must have the same value across the GPUs   */
    LwU32    mailboxTotalSize;          /* Size of the P2P Mailbox area
                                         * Must have the same value across the GPUs   */
    LwU32    flags;                     /* Flag to indicate types/attib of p2p   */
} LW503B_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl503b_h_ */
