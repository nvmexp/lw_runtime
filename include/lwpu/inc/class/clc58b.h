/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the writte
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _clc58b_h_
#define _clc58b_h_

#ifdef __cplusplus
extern "C" {
#endif

#define TURING_VMMU_A   (0x0000c58b)

/**
 * @brief LwAlloc parameters for TuringVmmuA class
 *
 * This class represents mapping between guest physical and system physical.
 * Will also be used to represent VF specific state for a given guest.
 *
 * gfid [in]
 *       GFID of VF
 **/

typedef struct
{
    LwHandle hHostVgpuDevice;
} TURING_VMMU_A_ALLOCATION_PARAMETERS;


#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif // _clc58b_h

