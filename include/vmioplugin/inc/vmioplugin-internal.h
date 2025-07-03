/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/* For internal use only. Not to be distributed */
#ifndef _vmioplugin_internal_h_
#define _vmioplugin_internal_h_

#if defined(LWCPU_AARCH64)
#define LW_CPU_DSP(opt) asm volatile("dsb " #opt : : : "memory")
#define LW_CPU_DMB(opt) asm volatile("dmb " #opt : : : "memory")
#define LW_CPU_MEM_BARRIER() LW_CPU_DSP(sy)
#define LW_CPU_MEM_WR_BARRIER() LW_CPU_DSP(st)
#define LW_CPU_MEM_RD_BARRIER() LW_CPU_DSP(ld)
#else
#define LW_CPU_DSP(opt)
#define LW_CPU_DMB(opt)
#define LW_CPU_MEM_BARRIER()
#define LW_CPU_MEM_WR_BARRIER()
#define LW_CPU_MEM_RD_BARRIER()
#endif

/**
 * Plugin handles shared with the environment
 */

typedef struct vmiop_guest_handle_s {
    LwHandle client;
    LwHandle vgpu_guest;
    vmiop_handle_t handle;
} vmiop_guest_handle_t;

/* minimum handle for a plugin event ID */
#define VMIOPD_EVENT_ID_MIN 0x7F000000
/* maximum handle for a plugin event ID */
#define VMIOPD_EVENT_ID_MAX 0x7F0FFF00

/* minimum handle for a lwpu-vgpu-mgr event ID */
#define VGPU_MGR_EVENT_ID_MIN 0x7F0FFF01
/* maximum handle for a lwpu-vgpu-mgr event ID */
#define VGPU_MGR_EVENT_ID_MAX 0x7F0FFFFF

/*
 * Based on Linux kernels UUID library which uses little endian indexing
 * mechanism.  KVM uses similar uuid_le index for maintaining guest uuid.
 * refer `uuid_le_index' from linux kernel.
 */
static LW_INLINE void uuid_u8_to_str(LwU8 *vm_uuid, char *uuid)
{
    LwU32  j;
    LwU32  index[] = {3, 2, 1, 0, 5, 4, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15};
    char  *ptr     = uuid;

    for (j = 0; j < 16; ++j) {
        sprintf(ptr, "%02x", vm_uuid[index[j]]);
        ptr += 2;

        if (3 == j || 5 == j || 7 == j || 9 == j)
            *ptr++ = '-';
    }
}

static LW_INLINE void uuid_str_to_u8(LwU8 *b_uuid, char *uuid_str)
{
    LwU32 j;

    for (j = 0; j < 16; j++) {
        sscanf(uuid_str, "%2hhx",&b_uuid[j]);
        uuid_str += 2;

        if (3 == j || 5 == j || 7 == j || 9 == j)
            uuid_str++;
    }
}

#endif /* _vmioplugin_internal_h_ */
