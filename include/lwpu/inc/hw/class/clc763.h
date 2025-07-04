/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2022 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

// AUTO GENERATED -- DO NOT EDIT - this file automatically generated by refhdr2class.pl
// Command: ../../../bin/manuals/refhdr2class.pl clc763.h c763 MMU_VIDMEM_ACCESS_BIT_BUFFER --search_str=LW_MMU_VIDMEM_ACCESS_BIT_BUF --input_file=lw_ref_dev_mmu_vidmem_access_bits.h


#ifndef _clc763_h_
#define _clc763_h_

#ifdef __cplusplus
extern "C" {
#endif

#define MMU_VIDMEM_ACCESS_BIT_BUFFER (0xc763)

#define LWC763                                                                           
#define LWC763_ENTRY                                                0x000001ff:0x00000000
#define LWC763_SIZE                           LW_MMU_VIDMEM_ACCESS_BIT_BUFFER_PACKET_SIZE
#define LWC763_ENTRY_ACCESS_BITS(i)                     MW((((i)*512)+255):(((i)*512)+0))
#define LWC763_ENTRY_ACCESS_BITS__SIZE_1 LW_MMU_VIDMEM_ACCESS_BIT_BUFFER_PACKET_SIZE / 64
#define LWC763_ENTRY_VALID(i)                         MW((((i)*512)+511):(((i)*512)+511))
#define LWC763_ENTRY_VALID__SIZE_1       LW_MMU_VIDMEM_ACCESS_BIT_BUFFER_PACKET_SIZE / 64
#define LWC763_ENTRY_GPC_DISABLE(i)                             MW(((i)+3840):((i)+3840))
#define LWC763_ENTRY_GPC_DISABLE__SIZE_1                                               64
#define LWC763_ENTRY_GPC_DISABLE_TRUE                                                 0x1
#define LWC763_ENTRY_GPC_DISABLE_FALSE                                                0x0
#define LWC763_ENTRY_HUB_DISABLE(i)                             MW(((i)+3904):((i)+3904))
#define LWC763_ENTRY_HUB_DISABLE__SIZE_1                                               32
#define LWC763_ENTRY_HUB_DISABLE_TRUE                                                 0x1
#define LWC763_ENTRY_HUB_DISABLE_FALSE                                                0x0
#define LWC763_ENTRY_HSCE_DISABLE(i)                            MW(((i)+3936):((i)+3936))
#define LWC763_ENTRY_HSCE_DISABLE__SIZE_1                                              16
#define LWC763_ENTRY_HSCE_DISABLE_TRUE                                                0x1
#define LWC763_ENTRY_HSCE_DISABLE_FALSE                                               0x0
#define LWC763_ENTRY_LINK_DISABLE(i)                            MW(((i)+3952):((i)+3952))
#define LWC763_ENTRY_LINK_DISABLE__SIZE_1                                              16
#define LWC763_ENTRY_LINK_DISABLE_TRUE                                                0x1
#define LWC763_ENTRY_LINK_DISABLE_FALSE                                               0x0

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clc763_h_ */
