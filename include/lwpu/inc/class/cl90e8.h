/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2016 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the writte
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl90e8_h_
#define _cl90e8_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW_PHYS_MEM_SUBALLOCATOR  (0x000090e8)

/**
 * @brief LwAlloc parameters for PhysMemSubAlloc class
 *
 * Used to create a new per-process physical memory suballocator
 *
 * hObj [in]
 *       Handle to the memory allocation or to TURING_VMMU_A object from where space to be managed
 *       by suballocator will be allocated/reserved.
 *
 * hHostVgpuDeviceKernel [in]
 *       HostVgpuDeviceKernel points to HostVgpuDeviceApi_KERNEL where this suballocator is applicable
 *
 * offset [in]
 *       Relative offset in the contiguous FB physical memory _allocation_(hMemory)
 *       from which to be suballocated
 *
 * size [in]
 *       Extent of the region the suballocator should manage
 **/

typedef struct
{
    LwHandle hObj;
    LwHandle hHostVgpuDeviceKernel;
    LwU64    offset LW_ALIGN_BYTES(8);
    LwU64    size LW_ALIGN_BYTES(8);
} LW_PHYS_MEM_SUBALLOCATOR_PARAMETERS;


#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif // _cl90e8_h

