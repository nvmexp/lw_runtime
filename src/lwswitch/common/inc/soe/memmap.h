/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef MEMMAP_H_
#define MEMMAP_H_

#ifdef __LINKER__
#define MAKE_ULL(x)    x
#else
#define MAKE_ULL(x)    (x##ULL)
#endif

//
// Engine memory map
//
// Unrelocated DMESG address (used by bootloader)
#define SOE_DMESG_BUFFER_PA_INIT              MAKE_ULL( 0xFF10000000000000 )
#define SOE_DMESG_BUFFER_VA_BASE              MAKE_ULL( 0x0000000001F00000 )
#define SOE_DMESG_BUFFER_SIZE                 MAKE_ULL( 0x0000000000001000 )

#define engineEMEM_VA_BASE                     MAKE_ULL( 0x0000000001F40000 )
#define engineEMEM_SIZE                        MAKE_ULL( 0x0000000000002000 )

#define engineDTCM_VA_BASE                     MAKE_ULL( 0x0000000001F80000 )
#define engineDTCM_SIZE                        MAKE_ULL( 0x0000000000010000 )

#define engineITCM_VA_BASE                     MAKE_ULL( 0x0000000001FC0000 )
#define engineITCM_SIZE                        MAKE_ULL( 0x0000000000010000 )

#define engineFBGPA_PA_SIZE                    MAKE_ULL( 0x0000800000000000 )

// Unrelocated FB addresses (used by bootloader)
#define engineFBGPA_PA_INIT                    MAKE_ULL( 0xFF00000000000000 )
#define engineFBGPA_VA_BASE                    MAKE_ULL( 0x0000000002000000 )
#define engineFBGPA_VA_SIZE                    MAKE_ULL( 0x000000000E000000 )

// Address space used to map to kernel/tasks - 256 M should be enough for now.
#define engineDYNAMIC_VA_BASE                  MAKE_ULL( 0x0000000010000000 )
#define engineDYNAMIC_VA_SIZE                  MAKE_ULL( 0x0000000010000000 )

// op = GVA, PA = 0x20000000
#define engineGVA_VA_BASE                      MAKE_ULL( 0x00000000B0000000 )
#define engineGVA_SIZE                         MAKE_ULL( 0x0000000010000000 )

#define enginePRIV_VA_BASE                     MAKE_ULL( 0x00000000C0000000 )
#define enginePRIV_SIZE                        MAKE_ULL( 0x0000000100000000 )

// op = SYSGPA, engid = 0xC0, PA = 0x00000000
#define engineSYSGPA_VA_BASE                   MAKE_ULL( 0x00000001C0000000 )
#define engineSYSGPA_SIZE                      MAKE_ULL( 0x0002000000000000 )

#define engineKMEM_PA_BASE                     MAKE_ULL( 0x0000000104000000 )

// Full FB mapping for RM-FW accesses
#define engineFBGPA_FULL_VA_BASE               MAKE_ULL( 0x1000000000000000 )
// Full FB mapping for RM-FW uncached accesses
#define engineFBGPA_FULL_VA_UC_BASE            MAKE_ULL( 0x1000800000000000 )

#endif // MEMMAP_H_
