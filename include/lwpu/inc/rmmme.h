
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _RMMME_H_
#define _RMMME_H_

// 
// defines used between RM and drivers to lock up various MME Shadow Scratch
// ram entries and set the base starting entry above which drivers are free
// to use as they see fit
//

#define RM_MME_FIRMWARE_SHADOW_SCRATCH_0         0
#define RM_MME_FIRMWARE_SHADOW_SCRATCH_1         1
#define RM_MME_FIRMWARE_SHADOW_SCRATCH_2         2
#define RM_MME_FIRMWARE_SHADOW_SCRATCH_3         3

#define RM_MME_QUADRO_TUNE_SHADOW_SCRATCH        4
#define RM_MME_QUADRO_TUNE_2_SHADOW_SCRATCH      5

// RM_MME_VGX_MIGRATION_COUNTER_SHADOW_SCRATCH will replace both RM_MME_FS_SIGNATURE_LO32_SHADOW_SCRATCH and 
// RM_MME_FS_SIGNATURE_HI32_SHADOW_SCRATCH, however will have to separate the change as it is dependent across
// chips_a and dev_a
#define RM_MME_VGX_MIGRATION_COUNTER_SHADOW_SCRATCH  6
#define RM_MME_FS_SIGNATURE_LO32_SHADOW_SCRATCH  6
#define RM_MME_FS_SIGNATURE_HI32_SHADOW_SCRATCH  7

#define RM_MME_FIRST_USABLE_SHADOW_SCRATCH       8

//
// In D3D1x UMD, LWD3D_MMESCRATCH_SHADOWSHADERPROGRAMOFFSET0 is defined as LW9097_SET_MME_SHADOW_SCRATCH(84) in g_mmeProgramDefines.h
// which is used in MME code to write a shadow copy of the program offset 0.
// The value 84 is automatically generated according to wgf2um\tools\autogenerate\mmeScratchRegSpec.xml.
// This shadow copy of program offsets applies to pre-Volta chips only.
// As long as UMD is backward compatible to pre-Volta chips, it won't change.
// Since it is determined by D3D1x UMD SW in the above xml file for pre-Volta chips,
//
#define DX_MME_SHADERPROGRAMOFFSET_SCRATCH(pipeline)  (84 + (pipeline))

#endif  // _RMMME_H_
