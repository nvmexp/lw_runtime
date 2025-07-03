 /***************************************************************************\
|*                                                                           *|
|*       Copyright 1993-2022 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     NOTICE TO USER:   The source code  is copyrighted under  U.S. and     *|
|*     international laws.  LWPU, Corp. of Sunnyvale,  California owns     *|
|*     copyrights, patents, and has design patents pending on the design     *|
|*     and  interface  of the LW chips.   Users and  possessors  of this     *|
|*     source code are hereby granted a nonexclusive, royalty-free copy-     *|
|*     right  and design patent license  to use this code  in individual     *|
|*     and commercial software.                                              *|
|*                                                                           *|
|*     Any use of this source code must include,  in the user dolwmenta-     *|
|*     tion and  internal comments to the code,  notices to the end user     *|
|*     as follows:                                                           *|
|*                                                                           *|
|*     Copyright  1993-2016  LWPU,  Corporation.   LWPU  has  design     *|
|*     patents and patents pending in the U.S. and foreign countries.        *|
|*                                                                           *|
|*     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY     *|
|*     OF THIS SOURCE CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITH-     *|
|*     OUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPORATION     *|
|*     DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOURCE CODE, INCLUD-     *|
|*     ING ALL IMPLIED WARRANTIES  OF MERCHANTABILITY  AND FITNESS FOR A     *|
|*     PARTICULAR  PURPOSE.  IN NO EVENT  SHALL LWPU,  CORPORATION  BE     *|
|*     LIABLE FOR ANY SPECIAL,  INDIRECT,  INCIDENTAL,  OR CONSEQUENTIAL     *|
|*     DAMAGES, OR ANY DAMAGES  WHATSOEVER  RESULTING  FROM LOSS OF USE,     *|
|*     DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR     *|
|*     OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION  WITH THE     *|
|*     USE OR PERFORMANCE OF THIS SOURCE CODE.                               *|
|*                                                                           *|
|*     RESTRICTED RIGHTS LEGEND:  Use, duplication, or disclosure by the     *|
|*     Government is subject  to restrictions  as set forth  in subpara-     *|
|*     graph (c) (1) (ii) of the Rights  in Technical Data  and Computer     *|
|*     Software  clause  at DFARS  52.227-7013 and in similar clauses in     *|
|*     the FAR and NASA FAR Supplement.                                      *|
|*                                                                           *|
 \***************************************************************************/

/******************* Operating System Interface Routines *******************\
*                                                                           *
* Module: LWCM.H                                                            *
*   Windows Configuration Manager defines and prototypes.                   *
*                                                                           *
* ***IMPORTANT***  The interfaces defined in this file are *deprecated*     *
* ***IMPORTANT***  in favor of RmControl.                                   *
* ***IMPORTANT***  Try hard to not use this file at all and definitely      *
* ***IMPORTANT***  do not add or modify interfaces here.                    *
*                  Ref: bug 488474: delete CFG and CFG_EX                   *
*                                                                           *
\***************************************************************************/

#ifndef _LWCM_H_
#define _LWCM_H_

#include "lwdeprecated.h"

#if LW_DEPRECATED_COMPAT(RM_CONFIG_GET_SET)

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(XAPIGEN)        /* avoid duplicate generated xapi fns */
#include "lwgputypes.h"
#ifndef _H2INC
#include "rmcd.h"
#endif

#include "lwerror.h"
#endif  /* !XAPIGEN */

//---------------------------------------------------------------------------
//
//  Configuration Defines.
//
//---------------------------------------------------------------------------

#if !(LWOS_IS_UNIX || LWOS_IS_INTEGRITY) && !defined(XAPIGEN)
#pragma pack(1) // assure byte alignment on structures shared among modules
#endif

// ***************************************************************************
// *                                                                         *
// ***IMPORTANT***  Do not add or modify CFG or CFGEX interfaces.            *
// ***IMPORTANT***  The interfaces defined in this file are *deprecated*     *
// *                Ref: bug 488474: delete CFG and CFG_EX                   *
// *                                                                         *
// ***************************************************************************


//
// Index parameters to ConfigGet/Set.  All other values are reserved.
//
// *** IMPORTANT ***
//
// Please update lwDbgRmConfigName(), when adding a new LW_CFG_* value,
//

//
// LW_CFG_ARCHITECTURE - Return the architecture of this device
//
//  0 - LW0
//  1 - LW1, etc
//
// NOTE: new ARCHITECTURE values should be added to
// LW2080_CTRL_CMD_GET_ARCH_INFO only!!!
//
#define LW_CFG_ARCHITECTURE                     2

#define LW_CFG_ARCHITECTURE_LW10                0x10
#define LW_CFG_ARCHITECTURE_LW40                0x40
#define LW_CFG_ARCHITECTURE_LW50                0x50
#define LW_CFG_ARCHITECTURE_G74                 0x60
#define LW_CFG_ARCHITECTURE_G80                 0x80
#define LW_CFG_ARCHITECTURE_G90                 0x90
#define LW_CFG_ARCHITECTURE_GT200               0xA0
#define LW_CFG_ARCHITECTURE_GF100               0xC0
#define LW_CFG_ARCHITECTURE_GF110               0xD0
#define LW_CFG_ARCHITECTURE_GK100               0xE0
#define LW_CFG_ARCHITECTURE_GK110               0xF0
#define LW_CFG_ARCHITECTURE_GK200               0x100
#define LW_CFG_ARCHITECTURE_GM000               0x110
#define LW_CFG_ARCHITECTURE_GP100               0x130
#define LW_CFG_ARCHITECTURE_GV100               0x140
#define LW_CFG_ARCHITECTURE_GV110               0x150
#define LW_CFG_ARCHITECTURE_TU100               0x160

//
// LW_CFG_REVISION - Return the major revision of this device
//
//  0 - RevA
//  1 - RevB, etc
//
#define LW_CFG_REVISION                         3

//
// LW_CFG_BUS_TYPE - Return the bus implementation of this device
//
//  1 - PCI
//  2 - VL
//  4 - AGP
//  8 - PCI Express
//  All other values are reserved
//
#define LW_CFG_BUS_TYPE                         5

//
// BUS type.
//
#define LW_BUS_TYPE_NONE                        0
#define LW_BUS_TYPE_PCI                         1
#define LW_BUS_TYPE_AGP                         4
#define LW_BUS_TYPE_PCI_EXPRESS                 8
#define LW_BUS_TYPE_FPCI                        16
#define LW_BUS_TYPE_AXI                         32

//
// LW_CFG_IMPLEMENTATION - Return the implementation of a chip architecture.
//
// NOTE: new IMPLEMENTATION values should be added to
// LW2080_CTRL_CMD_GET_ARCH_INFO only!!!
//
#define LW_CFG_IMPLEMENTATION                   8

// When ARCHITECTURE is 0x4, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_LW04      0x0
#define LW_CFG_IMPLEMENTATION_LW05      0x1
#define LW_CFG_IMPLEMENTATION_LW0A      0x2
// When ARCHITECTURE is 0x10, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_LW10      0x0
#define LW_CFG_IMPLEMENTATION_LW11      0x1
#define LW_CFG_IMPLEMENTATION_LW15      0x5
#define LW_CFG_IMPLEMENTATION_LW17      0x7
#define LW_CFG_IMPLEMENTATION_LW18      0x8
#define LW_CFG_IMPLEMENTATION_LW1A      0xA
#define LW_CFG_IMPLEMENTATION_LW1F      0xF
// When ARCHITECTURE is 0x20, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_LW20      0x0
#define LW_CFG_IMPLEMENTATION_LW25      0x5
#define LW_CFG_IMPLEMENTATION_LW28      0x8
// When ARCHITECTURE is 0x30, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_LW30      0x0
#define LW_CFG_IMPLEMENTATION_LW31      0x1
#define LW_CFG_IMPLEMENTATION_LW34      0x4
#define LW_CFG_IMPLEMENTATION_LW35      0x5
#define LW_CFG_IMPLEMENTATION_LW36      0x6
// When ARCHITECTURE is 0x40, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_LW40      0x0
#define LW_CFG_IMPLEMENTATION_LW41      0x1
#define LW_CFG_IMPLEMENTATION_LW42      0x2
#define LW_CFG_IMPLEMENTATION_LW43      0x3
#define LW_CFG_IMPLEMENTATION_LW44      0x4
#define LW_CFG_IMPLEMENTATION_LW44A     0xA
#define LW_CFG_IMPLEMENTATION_LW45      0x5
#define LW_CFG_IMPLEMENTATION_LW46      0x6
#define LW_CFG_IMPLEMENTATION_LW47      0x7
#define LW_CFG_IMPLEMENTATION_LW48      0x8
#define LW_CFG_IMPLEMENTATION_LW49      0x9
#define LW_CFG_IMPLEMENTATION_LW4B      0xB
#define LW_CFG_IMPLEMENTATION_LW4C      0xC
#define LW_CFG_IMPLEMENTATION_LW4D      0xD
#define LW_CFG_IMPLEMENTATION_LW4E      0xE
#define LW_CFG_IMPLEMENTATION_LW4F      0xF
// When ARCHITECTURE is 0x50, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_LW50      0x0
// When ARCHITECTURE is 0x60, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_G74       0x0
#define LW_CFG_IMPLEMENTATION_LW67      0x7
// When ARCHITECTURE is 0x80, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_G84       0x4
#define LW_CFG_IMPLEMENTATION_G86       0x6
// When ARCHITECTURE is 0x90, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_G92       0x2
#define LW_CFG_IMPLEMENTATION_G94       0x4
#define LW_CFG_IMPLEMENTATION_G96       0x6
#define LW_CFG_IMPLEMENTATION_G98       0x8
// When ARCHITECTURE is 0xA0, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GT200     0x0
#define LW_CFG_IMPLEMENTATION_GT212     0x2
#define LW_CFG_IMPLEMENTATION_GT214     0x4
#define LW_CFG_IMPLEMENTATION_GT215     0x3
#define LW_CFG_IMPLEMENTATION_GT216     0x5
#define LW_CFG_IMPLEMENTATION_GT218     0x8
#define LW_CFG_IMPLEMENTATION_MCP77     0xA
#define LW_CFG_IMPLEMENTATION_MCP7B     0xB
#define LW_CFG_IMPLEMENTATION_MCP79     0xC
#define LW_CFG_IMPLEMENTATION_GT21A     0xD
#define LW_CFG_IMPLEMENTATION_MCP89     0xF
// When ARCHITECTURE is 0xC0, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GF100      0x0
#define LW_CFG_IMPLEMENTATION_GF100B     0x08
#define LW_CFG_IMPLEMENTATION_GF104      0x04
#define LW_CFG_IMPLEMENTATION_GF104B     0x0E
#define LW_CFG_IMPLEMENTATION_GF106      0x03
#define LW_CFG_IMPLEMENTATION_GF106B     0x0F
#define LW_CFG_IMPLEMENTATION_GF108      0x01
// When ARCHITECTURE is 0xD0, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GF110      0x00
#define LW_CFG_IMPLEMENTATION_GF112      0x02
#define LW_CFG_IMPLEMENTATION_GF116      0x06
#define LW_CFG_IMPLEMENTATION_GF117      0x07
#define LW_CFG_IMPLEMENTATION_GF118      0x08
#define LW_CFG_IMPLEMENTATION_GF119      0x09
// When ARCHITECTURE is 0xE0, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GK100      0x00
#define LW_CFG_IMPLEMENTATION_GK104      0x04
#define LW_CFG_IMPLEMENTATION_GK106      0x06
#define LW_CFG_IMPLEMENTATION_GK106S     0x0B
#define LW_CFG_IMPLEMENTATION_GK107      0x07
#define LW_CFG_IMPLEMENTATION_GK20A      0x0A
// When ARCHITECTURE is 0xF0, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GK110      0x00
#define LW_CFG_IMPLEMENTATION_GK110B     0x01
#define LW_CFG_IMPLEMENTATION_GK110C     0x02
// When ARCHITECTURE is 0x100, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GK208      0x08
#define LW_CFG_IMPLEMENTATION_GK208S     0x06
// When ARCHITECTURE is 0x130, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GP100      0x00
#define LW_CFG_IMPLEMENTATION_GP000      0x01
#define LW_CFG_IMPLEMENTATION_GP102      0x02
#define LW_CFG_IMPLEMENTATION_GP104      0x04
#define LW_CFG_IMPLEMENTATION_GP106      0x06
#define LW_CFG_IMPLEMENTATION_GP107      0x07
#define LW_CFG_IMPLEMENTATION_GP108      0x08
#define LW_CFG_IMPLEMENTATION_GP10B      0x0B
#define LW_CFG_IMPLEMENTATION_GP10D      0x0D
#define LW_CFG_IMPLEMENTATION_GP10E      0x0E
// When ARCHITECTURE is 0x140, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GV100      0x00
#define LW_CFG_IMPLEMENTATION_GV000      0x01
#define LW_CFG_IMPLEMENTATION_GV10B      0x0B
// When ARCHITECTURE is 0x150, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GV11B      0x0B
// When ARCHITECTURE is 0x160, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_TU101      0x00
#define LW_CFG_IMPLEMENTATION_TU102      0x02
#define LW_CFG_IMPLEMENTATION_TU104      0x04
#define LW_CFG_IMPLEMENTATION_TU106      0x06
#define LW_CFG_IMPLEMENTATION_TU116      0x08
#define LW_CFG_IMPLEMENTATION_TU117      0x07
#define LW_CFG_IMPLEMENTATION_TU000      0x01
// When ARCHITECTURE is 0x170, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_GA100      0x00
#define LW_CFG_IMPLEMENTATION_GA101      0x01
#define LW_CFG_IMPLEMENTATION_GA102      0x02
#define LW_CFG_IMPLEMENTATION_GA103      0x03
#define LW_CFG_IMPLEMENTATION_GA104      0x04
#define LW_CFG_IMPLEMENTATION_GA106      0x06
#define LW_CFG_IMPLEMENTATION_GA107      0x07
#define LW_CFG_IMPLEMENTATION_GA000      0x01
// When ARCHITECTURE is 0x1E0, IMPLEMENTATION may be:
#define LW_CFG_IMPLEMENTATION_G000       0x00
#define LW_CFG_IMPLEMENTATION_G000_SOC   0x01

//
// LW_CFG_ADDRESS - Return the physical PCI address (BAR0) of this device
//
#define LW_CFG_ADDRESS                          10

//
// LW_CFG_PCI_ID - Return the PCI Vendor and Device ID assigned to this device
//
//      DDDDVVVV -  VVVV    PCI Vendor ID
//                  DDDD    PCI Device ID
//
// Careful: For a BR02 + GPU combination, this config call will return the GPU ID,
// and not the BR02 ID. If you need the BR02 ID, use the config call
// LW_CFG_PCI_CONFIG_SPACE_PCI_ID
#define LW_CFG_PCI_ID                           13

//
// LW_CFG_PCI_SUB_ID - Return the PCI Subsystem Vendor and Device ID assigned to this device
//
//      DDDDVVVV -  VVVV    PCI Subsystem Vendor ID
//                  DDDD    PCI Subsystem ID
//
#define LW_CFG_PCI_SUB_ID                       14

//
// LW_CFG_PROCESSOR_TYPE - Return the functionality of the system processor
//
// There are 2 pieces of data passed back, the processor indicator is in
// the low 8 bits and its bitmask of functionality is in the upper 24
//
// For a list of the instructions included in LW_CPU_FUNC_MMX_EXT and
// LW_CPU_FUNC_3DNOW_EXT, see the AMD_3DNow_MMX_Extensions PDF file in
// \\syseng-sc-1\Archive\datasheets\chipsets_and_processors\amd.
//
#define LW_CPU_UNKNOWN         0x00000000    // Unknown / generic
// Intel
#define LW_CPU_P5              0x00000001
#define LW_CPU_P55             0x00000002    // P55C - MMX
#define LW_CPU_P6              0x00000003    // PPro
#define LW_CPU_P2              0x00000004    // PentiumII
#define LW_CPU_P2XC            0x00000005    // Xeon & Celeron
#define LW_CPU_CELA            0x00000006    // Celeron-A
#define LW_CPU_P3              0x00000007    // Pentium-III
#define LW_CPU_P3_INTL2        0x00000008    // Pentium-III w/ integrated L2 (fullspeed, on die, 256K)
#define LW_CPU_P4              0x00000009    // Pentium 4
#define LW_CPU_CORE2           0x00000010    // Intel Core2 Duo
#define LW_CPU_CELN_M16H       0x00000011    // Celeron model 16h (65nm)
#define LW_CPU_CORE2_EXTRM     0x00000012    // Intel Core2 Extreme/Intel Xeon model 17h (45nm)
#define LW_CPU_ATOM            0x00000013    // Intel Atom
#define LW_CPU_IA64            0x00000020    // Itanium

// AMD
#define LW_CPU_K5              0x00000030
#define LW_CPU_K6              0x00000031
#define LW_CPU_K62             0x00000032    // K6-2 w/ 3DNow
#define LW_CPU_K63             0x00000033
#define LW_CPU_K7              0x00000034
#define LW_CPU_K8              0x00000035
#define LW_CPU_K10             0x00000036
#define LW_CPU_K11             0x00000037
// K8 Sub types
#define LW_CPU_K8_NON_OPTERON  0x00000000
#define LW_CPU_K8_OPTERON      0x00000001

// IDT/Centaur
#define LW_CPU_C6              0x00000060    // WinChip C6
#define LW_CPU_C62             0x00000061    // WinChip 2 w/ 3DNow
// Cyrix
#define LW_CPU_GX              0x00000070    // MediaGX
#define LW_CPU_M1              0x00000071    // 6x86
#define LW_CPU_M2              0x00000072    // M2
#define LW_CPU_MGX             0x00000073    // MediaGX w/ MMX
// Transmeta
#define LW_CPU_TM_CRUSOE       0x00000080    // Transmeta Crusoe(tm)
// PowerPC
#define LW_CPU_PPC603          0x00000090    // PowerPC 603
#define LW_CPU_PPC604          0x00000091    // PowerPC 604
#define LW_CPU_PPC750          0x00000092    // PowerPC 750

// Function bits
#define LW_CPU_FUNC_MMX                 0x00000100 // supports MMX
#define LW_CPU_FUNC_SSE                 0x00000200 // supports SSE
#define LW_CPU_FUNC_3DNOW               0x00000400 // supports 3DNow
#define LW_CPU_FUNC_SSE2                0x00000800 // supports SSE2
#define LW_CPU_FUNC_SFENCE              0x00001000 // supports SFENCE
#define LW_CPU_FUNC_WRITE_COMBINING     0x00002000 // supports write-combining
#define LW_CPU_FUNC_ALTIVEC             0x00004000 // supports ALTIVEC
#define LW_CPU_FUNC_PUT_NEEDS_IO        0x00008000 // requires OUT inst w/PUT updates
#define LW_CPU_FUNC_NEEDS_WC_WORKAROUND 0x00010000 // old P4 cpus have a write-combining bug
#define LW_CPU_FUNC_3DNOW_EXT           0x00020000 // supports 3DNow Extensions
#define LW_CPU_FUNC_MMX_EXT             0x00040000 // supports MMX Extensions
#define LW_CPU_FUNC_CMOV                0x00080000 // supports CMOV
#define LW_CPU_FUNC_CLFLUSH             0x00100000 // supports CLFLUSH
#define LW_CPU_FUNC_SSE3                0x00400000 // supports SSE3
#define LW_CPU_FUNC_NEEDS_WAR_124888    0x00800000 // exposed to bug 124888
#define LW_CPU_FUNC_HT_CAPABLE          0x01000000 // supports hyper-threading
#define LW_CPU_FUNC_SSE41               0x02000000 // supports SSE4.1 (Penryn)
#define LW_CPU_FUNC_SSE42               0x04000000 // supports SSE4.2 (Nehalem)
#define LW_CPU_FUNC_AVX                 0x08000000 // supports AVX (Sandy Bridge)
#define LW_CPU_FUNC_ERMS                0x10000000 // supports ERMS (Ivy Bridge)

// Feature mask (as opposed to bugs/requirements etc.)
#define LW_CPU_FUNC_FEATURE_MASK (LW_CPU_FUNC_MMX             | \
                                  LW_CPU_FUNC_SSE             | \
                                  LW_CPU_FUNC_3DNOW           | \
                                  LW_CPU_FUNC_SSE2            | \
                                  LW_CPU_FUNC_SFENCE          | \
                                  LW_CPU_FUNC_WRITE_COMBINING | \
                                  LW_CPU_FUNC_ALTIVEC         | \
                                  LW_CPU_FUNC_3DNOW_EXT       | \
                                  LW_CPU_FUNC_MMX_EXT         | \
                                  LW_CPU_FUNC_CMOV            | \
                                  LW_CPU_FUNC_CLFLUSH         | \
                                  LW_CPU_FUNC_SSE3            | \
                                  LW_CPU_FUNC_HT_CAPABLE      | \
                                  LW_CPU_FUNC_SSE41           | \
                                  LW_CPU_FUNC_SSE42           | \
                                  LW_CPU_FUNC_AVX             | \
                                  LW_CPU_FUNC_ERMS)


#define LW_CFG_PROCESSOR_TYPE                   15

//
// LW_CFG_PROCESSOR_SPEED - Return the speed of the processor in MHz
//
#define LW_CFG_PROCESSOR_SPEED                  16

//
// LW_CFG_GRAPHICS_CAPS* - Return the capabilities of the graphics HW
//
#define LW_CFG_GRAPHICS_CAPS                    18
#define LW_CFG_GRAPHICS_CAPS2                   17

#define LW_CFG_GRAPHICS_CAPS_UNKNOWN                                0x00000000
#define LW_CFG_GRAPHICS_CAPS_MAXCLIPS_MASK                          0x0000000F  // bits 3:0
#define LW_CFG_GRAPHICS_CAPS_FF                                     0x00000040  // bit  6
#define LW_CFG_GRAPHICS_CAPS_MAXCLIPS_SHIFT                         0
#define LW_CFG_GRAPHICS_CAPS_AA_LINES                               0x00000100  // bit  8
#define LW_CFG_GRAPHICS_CAPS_AA_POLYS                               0x00000200  // bit  9
#define LW_CFG_GRAPHICS_CAPS_LOGIC_OPS                              0x00000800  // bit 11
#define LW_CFG_GRAPHICS_CAPS_2SIDED_LIGHTING                        0x00002000  // bit 13
#define LW_CFG_GRAPHICS_CAPS_QUADRO_GENERIC                         0x00004000  // bit 14
#define LW_CFG_GRAPHICS_CAPS_UBB                                    0x00008000  // bit 15
#define LW_CFG_GRAPHICS_CAPS_3D_TEXTURES                            0x00020000  // bit 17
#define LW_CFG_GRAPHICS_CAPS_ANISOTROPIC                            0x00040000  // bit 18
#define LW_CFG_GRAPHICS_CAPS_CLIPPED_ALINES                         0x00080000  // bit 19

#define LW_CFG_GRAPHICS_CAPS2_LARGE_NONCOH_UPSTR_WRITE_BUG_114871   0x01000000  // bit 24
#define LW_CFG_GRAPHICS_CAPS2_LARGE_UPSTREAM_WRITE_BUG_115115       0x02000000  // bit 25
#define LW_CFG_GRAPHICS_CAPS2_SUPPORT_RENDER_TO_SYSMEM              0x04000000  // bit 26
#define LW_CFG_GRAPHICS_CAPS2_BLOCKLINEAR                           0x20000000  // bit 29
#define LW_CFG_GRAPHICS_CAPS2_SUPPORT_SCANOUT_FROM_SYSMEM           0x40000000  // bit 30

//
// LW_CFG_INSTANCE_SIZE - Return the size of the instance pool of this device
//
#define LW_CFG_INSTANCE_TOTAL                   21

#define LW_CFG_VIDEO_OUTPUT_FORMAT_AUTOSELECT          0x0

//
// LW_CFG_DAC_INPUT_WIDTH - Return the width (in bits) of the current framebuffer bus
//
#define LW_CFG_DAC_INPUT_WIDTH                  110

// LW_CFG_MOBILE_FLAGS
// This config get is used to return mobile flags early in the bringup of the RM
// to the drivers.
#define LW_CFG_MOBILE_FLAGS                     192
#define LW_CFG_MOBILE_FLAGS_SYSTEM_DESKTOP      0x00000000
#define LW_CFG_MOBILE_FLAGS_SYSTEM_MOBILE       0x00000001

//
// LW_CFG_DAC_MEMORY_CLOCK - Return the current memory clock (in Hz) for this device
//
#define LW_CFG_DAC_MEMORY_CLOCK                 251

//
// LW_CFG_DAC_GRAPHICS_CLOCK - Return the current graphics clock (in Hz) for this device
//
#define LW_CFG_DAC_GRAPHICS_CLOCK               253

// ***************************************************************************
// *                                                                         *
// ***IMPORTANT***  Do not add or modify CFG or CFGEX interfaces.            *
// ***IMPORTANT***  The interfaces defined in this file are *deprecated*     *
// *                Ref: bug 488474: delete CFG and CFG_EX                   *
// *                                                                         *
// ***************************************************************************

//
// LW_CFG_ALLOW_MAXPERF - Call into RM to get "Silent Running" state.
//
#define LW_CFG_ALLOW_MAXPERF                    521

//
// LW_CFG_RESERVE_PERFMON_HW - Call into RM to tell it an application would like to use the HW
// Performance Monitoring capabilities
//
#define LW_CFG_RESERVE_PERFMON_HW               522
// ConfigSet Args
#define LW_CFG_RESERVE_PERFMON_HW_FREE                0x00000000
#define LW_CFG_RESERVE_PERFMON_HW_RESERVE             0x00000001
#define LW_CFG_RESERVE_PERFMON_CLIENT_HANDLES_113632  0x00000002
#define LW_CFG_RESERVE_PERFMON_RM_HANDLES_IDLE_SLOW   0x00000004

//
// LW_CFG_BOARD_ID
//
// Returns the board ID found in the bios.  Caller is responsible for decoding the
// meaning of this value.  Note that BIOS can and will change values without meaning.
// Use at your own risk!
//
#define LW_CFG_BOARD_ID                         540

// Return the number of subdevices found with a multichip configuration
#define LW_CFG_NUMBER_OF_SUBDEVICES             700

//////////////////////////////////////////////////////////////////////////////
//
// The following Config equates are a 16bit subset of the corresponding
// ConfigEX values
//
// Where possible, make sure to use the 32bit versions of these DDK interfaces
// rather than these legacy equates
//

//
//  Determination of STEREO_CAPS_DIN, is through three properties:
//  1. DCB 2.1 Miscellaneous Stereo bit.
//  2. Device and Sub-device ID only as a last resort
//  3. DCB 3.0 Connector Table Entry for Stereo 3-Pin DIN
//  See bug 127004 for more details
#define LW_CFG_STEREO_CAPS                      40001
#define LW_CFG_STEREO_CAPS_DIN                  0:0
#define LW_CFG_STEREO_CAPS_DIN_NOT_FOUND        0x00000000
#define LW_CFG_STEREO_CAPS_DIN_AVAILABLE        0x00000001

////////////////////////////////////////////////////////////////////
//
// The following Config equates are to be used with the ConfigGetEx()
// and ConfigSetEx() functions.  Rather than just taking a DWORD
// data value, they allow the use of a per-function data structure
// to be used as a parameter block.
//
// Be careful to only use these equates with the appropriate
// functions.
//
// *** IMPORTANT ***
//
// Please update lwDbgRmConfigExName(), when adding a new LW_CFGEX_* value,
//

// Security functions are available through both ConfigSet() & ConfigSetEx() and must be the same
#define LW_CFGEX_DEBUGGER_WATCH         41001

typedef struct {
    LwP64          uESP LW_ALIGN_BYTES(8);
    LwU32          abnormalOffset;
    LwU32          range;
    LwS32          timeInMSecs;          // set to 0 to turn off a 'watch'
} LW_CFGEX_DEBUGGER_WATCH_PARAMS;

// Param structure
// FlatPanelMode: !!Please use Version 20 if at all possible!!
//   Version 0 ---- (Bits 8-15 == 0)
//     0 - Scaled
//     1 - Centered
//     2 - Native
//   Version 20 --- (Bits 8-15 == 0x20)
//     Bits 0-3 - FP Scaler
//       0 - Default - Use whatever was there before....
//       1 - Native
//       2 - Scaled
//       3 - Centered
//       4 - 8-Bit Scaled
//       5 - Aspect Scaled
//     Bits 16-17 FP Dither
//       0 - Default - Use whatever was there before....
//       1 - Force enable dithering
//       2 - Force disable dithering
// FlatPanelSizeX:
//     Max horizontal resolution.
// FlatPanelSizeY:
//     Max vertical resolution.
// FlatPanelConnected
//     0 - Not Connected
//     1 - Connected
// FlatPanelNativeSupported
//     0 - Not supported (No monitor scaler)
//     1 - Supported (Monitor has scaler (based on Edids))

#define LW_CFGEX_GET_FLATPANEL_INFO_NOT_CONNECTED   0
#define LW_CFGEX_GET_FLATPANEL_INFO_CONNECTED       1
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALED          0
#define LW_CFGEX_GET_FLATPANEL_INFO_CENTERED        1
#define LW_CFGEX_GET_FLATPANEL_INFO_NATIVE          2

#define LW_CFGEX_GET_FLATPANEL_INFO_SCALED_8BIT            0x1100
#define LW_CFGEX_GET_FLATPANEL_INFO_CENTERED_DITHER        0x1101
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALED_ASPECT_CORRECT  0x1102

// This bit is used to tell us if calling functions are using the new format
// or the old format.
#define LW_CFGEX_GET_FLATPANEL_INFO_VERSION                 15:8
#define LW_CFGEX_GET_FLATPANEL_INFO_VERSION_00            (0x00000000)
#define LW_CFGEX_GET_FLATPANEL_INFO_VERSION_10            (0x00000010)
#define LW_CFGEX_GET_FLATPANEL_INFO_VERSION_11            (0x00000011)
#define LW_CFGEX_GET_FLATPANEL_INFO_VERSION_20            (0x00000020)

// VERSION 20 defines for the pDev->Dac.HalInfo.CrtcInfo[Head].fpMode variable
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALER                   3:0
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALER_DEFAULT         (0x00000000)
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALER_NATIVE          (0x00000001)
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALER_SCALED          (0x00000002)
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALER_CENTERED        (0x00000003)
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALER_8BIT_SCALE      (0x00000004)
#define LW_CFGEX_GET_FLATPANEL_INFO_SCALER_ASPECT_SCALE    (0x00000005)
// _DEFAULT is only used as an input parameter to get this format
// on the return.  We still need to support the old format for the
// returns, until all drivers move over to the new format.
// All other return values will be the last requested FPMode from the
// LWXXX_SET_DAC_FORMAT, LW_CFGEX_SET_FLAT_PANEL_INFO,
// LW_CFGEX_SET_FLAT_PANEL_SCALING, or LW_CFGEX_SET_FLAT_PANEL_INFO_MULTI.
// Note: These values will now be supported in those CFGEX_SET calls listed above as well.

// VERSION 20 defines for the FPDITHER bits (older versions didn't have these bits)
#define LW_CFGEX_GET_FLATPANEL_INFO_FPDITHER                     17:16
#define LW_CFGEX_GET_FLATPANEL_INFO_FPDITHER_DEFAULT        (0x00000000)
#define LW_CFGEX_GET_FLATPANEL_INFO_FPDITHER_ENABLE         (0x00000001)
#define LW_CFGEX_GET_FLATPANEL_INFO_FPDITHER_DISABLE        (0x00000002)
// _DEFAULT means the RM will choose when to enable or disable dithering.
// _ENABLE will turn on Dithering always.
// _DISABLE will turn off Dithering always.

// Quick defines for old versions in version 20 format
#define LW_CFGEX_GET_FLATPANEL_INFO_V20_DEFAULT             (0x00002000)
#define LW_CFGEX_GET_FLATPANEL_INFO_V20_NATIVE              (0x00002001)
#define LW_CFGEX_GET_FLATPANEL_INFO_V20_SCALED              (0x00002002)
#define LW_CFGEX_GET_FLATPANEL_INFO_V20_CENTERED            (0x00002003)
#define LW_CFGEX_GET_FLATPANEL_INFO_V20_SCALED_ASPECT       (0x00002005)
#define LW_CFGEX_GET_FLATPANEL_INFO_V20_CENTERED_DITHER     (0x00012003)

/************** Multihead CFGEX_SET **************/

// Note: Please add all new events to the LW_CFGEX_EVENT_HANDLE call also (or even better, instead)
#define LW_CFGEX_SET_EVENT_HANDLES                          120
// Param Structure
typedef struct {
    LwP64         IconBegin             LW_ALIGN_BYTES(8);
    LwP64         IconEnd               LW_ALIGN_BYTES(8);
    LwP64         ScaleEvent            LW_ALIGN_BYTES(8);
    LwP64         FullScreenDXEvent     LW_ALIGN_BYTES(8);  // Full-Screen DX event
    LwP64         ThermalEvent          LW_ALIGN_BYTES(8);  // Temperature has exceeded limit
    LwP64         PowerConnectorEvent   LW_ALIGN_BYTES(8);  // Auxillary power disconnected
    LwP64         MobileEvent           LW_ALIGN_BYTES(8);  // Mobile specific events (Lid state changed...)
                                                            // must call LW_CFGEX_GET_MOBILE_EVENT_DATA
                                                            // to get specific data
    LwP64         InhibitEvent          LW_ALIGN_BYTES(8);  // inhibit flags have changed
    LwP64         HotPlugEvent          LW_ALIGN_BYTES(8);  // A display device hot plug/unplug has oclwrred

    LwP64         SREvent               LW_ALIGN_BYTES(8);  // A silent running state change has oclwrred
    LwP64         AgpConfigurationEvent LW_ALIGN_BYTES(8);  // Accelerated Graphical Port running state change has oclwrred
    LwP64         ThermalEventHW        LW_ALIGN_BYTES(8);  // Signal on HW controlled thermal related performance decrease
    LwP64         ThermalEventSW        LW_ALIGN_BYTES(8);  // Signal on SW controlled thermal related performance decrease
    LwP64         SmartDimmerEvent      LW_ALIGN_BYTES(8);
    LwP64         InduceDeviceScanEvent LW_ALIGN_BYTES(8);  // Signal to induce device scan
    LwP64         SetBrightnessRequestEvent LW_ALIGN_BYTES(8);  // BIOS->RM would like to change brightness, does LWCPL agree?

    LwP64         CoppEvent             LW_ALIGN_BYTES(8);  // Signal HDCP or other copy-protection-related action required
    LwP64         DockWar922492Event    LW_ALIGN_BYTES(8);  // For bug 922492, use this instead of InduceDeviceScanEvent

    LwU32         EventNotifyType;       // LW01_EVENT_WIN32_EVENT, LW01_EVENT_KERNEL_CALLBACK
} LW_CFGEX_SET_EVENT_HANDLES_PARAMS;

//
// LW_CFGEX_GET_MCMB_ARCH
// Used to determine characteristics of current multichip or multiboard architecture
//
#define LW_CFGEX_GET_MCMB_ARCH                              123
typedef struct {
    LwU32 Flags;
} LW_CFGEX_GET_MCMB_ARCH_PARAMS;
#define MCMB_ARCH_BROADCAST                     0x00000001
#define MCMB_ARCH_UNICAST                       0x00000002
#define MCMB_ARCH_MULTICAST                     0x00000004
#define MCMB_ARCH_AGP_HEAP                      0x00000008
#define MCMB_ARCH_VIDEO_BRIDGE                  0x00000010
#define MCMB_ARCH_MULTI_GPU                     0x00000020

// Deprecated...
#define DISPLAY_TYPE_MONITOR            0
#define DISPLAY_TYPE_FLAT_PANEL         3

/************** RESERVED **************/
#define LW_CFGEX_RESERVED                       150

//
//  New versions of ConfigGetEx for multi-head devices
//

// Updated _PREDAC_COLOR_SATURATION_BOOST which is device-based rather than head-based.
#define LW_CFGEX_DIGITAL_VIBRANCE               269
typedef struct {
    LwU32 DeviceMap;
    LwU16 dvcVersion;
    LwU16 boostValue;
    LwS32 CosHue_x10K;      // For Evo, cos of the hue angle. multiplied by 10000 to avoid floating
    LwS32 SinHue_x10K;      // For Evo, sin of the hue angle. multiplied by 10000 to avoid floating
    LwU16 dirtyHue;         // Takes value 1 if Hue angle is the information (and not the boostValue). 0 means boostValue is the info.
} LW_CFGEX_DIGITAL_VIBRANCE_PARAMS;

// color saturation boosts for DVC 2.0 could be any value from 0 - 63
#define LW_CFGEX_PREDAC_COLOR_SATURATION_BOOST_DVC2_MIN     0x0000
#define LW_CFGEX_PREDAC_COLOR_SATURATION_BOOST_DVC2_MAX     0x003F

#define LW_CFGEX_FLAT_PANEL_BRIGHTNESS      278
//param structure
//   -this structure is used for setting and getting the settings of the PWM brightness output.
//   -when the getex command is called for this structure, the only two valid values on return will
//    be PWMControllerEnable and PercentRelativeBrightness.

typedef struct
{
    // LW_CFGEX_FLATPANEL_BRIGHTNESS_PARAMS_PANEL fields still used on Mac
    LwU32          FlatPanelBrightnessControlFrequency;
    LwU32          MinBrightnessPWM_HighPercentage;
    LwU32          MaxBrightnessPWM_HighPercentage;

    // Obsolete fields
    LwU32          Head;
    LwU32          BaseTransitionSpeed;  // set transition speed to zero for immediate transition
    LwU32          TransitionBoostTable[36];  // array is of 32 elements
    LwU32          RelativeBrightness; // (0-255)
    LwS16          PWMControllerEnable;
    LwU32          BrightnessControlAvailable;
    LwU32          NumAllowedBrightnessLevels;
    LwU32          MaxAllowedBrightness;
    LwU32          MinAllowedBrightness;
#define LW_CFGEX_FLATPANEL_BRIGHTNESS_PARAMS_HEAD_SPECIFIED   BIT(0)

    LwU16          paramsSpecified;
#define LW_CFGEX_FLATPANEL_BRIGHTNESS_PARAMS_NONE_SPECIFIED 0x0
#define LW_CFGEX_FLATPANEL_BRIGHTNESS_PARAMS_PANEL          BIT(0)

// New fields - Old fields will be removed in the near future.
    LwU32       Available;       // GET ONLY
    LwU32       Levels;          // GET ONLY
    LwU32       MaxBrightness;   // GET ONLY
    LwU32       MinBrightness;   // GET ONLY
    LwU32       Brightness;      // GET/SET: 0-255
    LwU32       TransitionRate;  // SET ONLY: zero for immediate transition
    LwU32       BrightnessReq;   // GET ONLY: See Bug 236953
} LW_CFGEX_FLATPANEL_BRIGHTNESS_PARAMS;

/************** Power Mizer Level Get / Set *******************/
// LW_CFGEX_POWER_MIZER - Read or write the runtime power saving level.  The following
// table shows typical clock frequencies for LW11 and LW17 PowerMizer levels.  The
// clock frequencies can be overridden with registry settings.
//
// PowerMizer  Control Panel        LWCLK (MHz)*  MCLK (MHz)*
// Level       Setting              LW11  LW17    LW11  LW17    Swap Interval
// ==========  ================     ====  ====    ====  ====    =============
// 0           -                       0     0       0     0        0
// 1           Max Performance         0     0       0     0        0
// 2           Balanced              102   190     166   190        1
// 3           Max Battery Life      102   102      83   125        2
// 4           -                     102   102      83   125        2
//
// * 0 means use hardware default (full speed).
//
// When used with ConfigGetEx, Default returns the system default setting for
// use when a user requests [Restore to Defaults]. Selected returns the
// value chosen by the user.  Dynamic returns the value lwrerently being used,
// which is the user's value after adjusting for AC.
//
// When used with ConfigSetEx, Default and Dynamic are ignored, and Selected is used
// to set the current system value which is also cached in the system
// registry for preservation across reboots.
#define LW_CFGEX_POWER_MIZER                280

typedef struct {
    LwU32   All;                                // The universe of levels for this power state
    LwU32   Available;                          // The lwrrently available levels
    LwU32   Default;                            // The default level
    LwU32   Selected;                           // The lwrrently selected level
    LwU32   Forced;                             // The level that the driver is locked at.
    LwU32   Dynamic;                            // The level being used at the moment this info is requested
} PowerStateLevelInfo;

// Bitmask values for PowerStateLevelInfo members
#define POWER_MIZER_LEVEL_MAX_BATT    BIT(0)    // Maximize battery PM slider setting.
#define POWER_MIZER_LEVEL_MIN_POWER   BIT(0)    // Minimum power setting. Same as MAX_BATT but more sensible in the AC case.
#define POWER_MIZER_LEVEL_BALANCED    BIT(1)    // Balanced PM slider setting.
#define POWER_MIZER_LEVEL_MAX_PERF    BIT(2)    // Maximize performance PM slider setting.

typedef struct {
    LwU32   Default;                            // Default value for "Restore Defaults" on property page (Battery).
    LwU32   Selected;                           // Value set on property page slider and saved in registry (Battery).
    LwU32   SelectedHard;                       // Hard limit on power mizer for battery source
    LwU32   DefaultAC;                          // Default value for "Restore Defaults" on property page (AC).
    LwU32   SelectedAC;                         // Value set on property page slider and saved in registry (AC).
    LwU32   SelectedHardAC;                     // Hard limit on power mizer for AC source
    LwU32   Dynamic;                            // Value the driver is lwrrently using.  Affected by AC power state.
    PowerStateLevelInfo  AC;                    // Power state level info while on AC
    PowerStateLevelInfo  Battery;               // Power state level info while on battery
} LW_CFGEX_POWER_MIZER_PARAMS;

#define POWER_MIZER_MIN         0               // Minimum valid level.
#define POWER_MIZER_0           0               // Not used by control panel.  Everything at max - no registry override.
#define POWER_MIZER_MAX_PERF    1               // Maximize performance PM slider setting.
#define POWER_MIZER_BALANCED    2               // Balanced PM slider setting.
#define POWER_MIZER_MAX_BATT    3               // Maximize battery PM slider setting.
#define POWER_MIZER_MIN_POWER   3               // Minimum power setting. Same as MAX_BATT but more sensible in the AC case.
#define POWER_MIZER_4           4               // Not used by control panel.
#define POWER_MIZER_MAX         4               // Maximum valid level.

#define POWER_MIZER_DEFAULT  POWER_MIZER_MAX_PERF   // Default to maximize performance.
#define POWER_MIZER_AC       POWER_MIZER_MAX_PERF   // Use default level when the laptop is plugged in.

/************** AC Power State Get / Set *******************/
#define POWER_MIZER_POWER_BATT     0
#define POWER_MIZER_POWER_AC       1

/************** Return RM mappings to device's regs/fb *******/
#define LW_CFGEX_GET_DEVICE_MAPPINGS                        311
//
// This call is used by LWDE debugger to retrieve the RM's
// (linear) mappings to a given device's registers and framebuffer.
//
// Param structure
typedef struct {
    LwP64 lwVirtAddr LW_ALIGN_BYTES(8);               // register virtual address
    LwU64 lwPhysAddr LW_ALIGN_BYTES(8);               // register physical address
    LwP64 fbVirtAddr LW_ALIGN_BYTES(8);               // fb virtual address
    LwU64 fbPhysAddr LW_ALIGN_BYTES(8);               // fb physical address

} LW_CFGEX_GET_DEVICE_MAPPINGS_PARAMS;

#define LW_DISPLAY_DEVICE_MAP_MAX   32  // 8 CRT + 8 TV + 16 DFP = 32

#define LW_CFGEX_CONNECTOR_TYPE_HDMI_A     0x00000061

// LW_CFGEX_GET_HOTKEY_EVENT - Report Fn+x hotkey events on a mobile system.
//  Some events are informational -- the BIOS has already acted on the event.
//  Other events indicate an action that must be taken care of by the driver.
//  Lwrrently, only the display enable/disable events fall into this category.
//
// The RM checks the BIOS for new events every vblank.  Note that more than one
//  event may be added to the queue at a time (e.g., display enable/disable events
//  will often occur in sets).  The caller should continue to poll until the
//  call returns LW_HOTKEY_EVENT_NONE_PENDING.
//
// If the BIOS does not support this feature, the RM will return LW_HOTKEY_EVENT_NOT_SUPPORTED.
//
// Input: a pointer to a LW_CFGEX_GET_HOTKEY_EVENT_PARAMS paramStruct
// Output: the event field will report a number from the event list below
//         the status field, where appropriate, indicates the new state the event is reporting
//

//Queue Commands
#define LW_HOTKEY_EVENT_NOT_SUPPORTED        0  //No BIOS hotkey support.  Further polling not required.
#define LW_HOTKEY_EVENT_NONE_PENDING         1  //No hotkey events lwrrently pending
#define LW_HOTKEY_EVENT_DISPLAY_ENABLE       2  //status bits decode to disable/enable state for
                                                // each display (definitions below) (implemented by driver)
//#define LW_HOTKEY_EVENT_DISPLAY_LCD          2  //status 0/1 = disable/enable (implemented by driver)
//#define LW_HOTKEY_EVENT_DISPLAY_CRT          3  //status 0/1 = disable/enable (implemented by driver)
//#define LW_HOTKEY_EVENT_DISPLAY_TV           4  //status 0/1 = disable/enable (implemented by driver)
//#define LW_HOTKEY_EVENT_DISPLAY_DFP          5  //status 0/1 = disable/enable (implemented by driver)
#define LW_HOTKEY_EVENT_DISPLAY_CENTERING    6  //scaled/centered display - status values define below (implemented by BIOS)
                                                // mirrors settings for LW_PRAMDAC_FP_TG_CONTROL_MODE in lw_ref.h

// Status bit definitions for LW_HOTKEY_EVENT_DISPLAY_ENABLE event
// Lwrrently identical to the BIOS register bit definitions, but
// we decode it explicitly to avoid implicit dependencies.
#define LW_HOTKEY_STATUS_DISPLAY_ENABLE_LCD 0x01
#define LW_HOTKEY_STATUS_DISPLAY_ENABLE_CRT 0x02
#define LW_HOTKEY_STATUS_DISPLAY_ENABLE_TV  0x04
#define LW_HOTKEY_STATUS_DISPLAY_ENABLE_DFP 0x08

//Enumeration of centering/scaling settings used with
// LW_HOTKEY_EVENT_DISPLAY_CENTERING event
#define LW_HOTKEY_STATUS_DISPLAY_SCALED     0x00
#define LW_HOTKEY_STATUS_DISPLAY_CENTERED   0x01
#define LW_HOTKEY_STATUS_DISPLAY_NATIVE     0x02

#define LW_CFGEX_GET_LOGICAL_DEV_EDID                331
// param structure
// NOTE: deprecated and poorly designed interface (length is behind a ptr (why???))
typedef struct {
    LwP64          edidBuffer   LW_ALIGN_BYTES(8); // ptr to LwU8 buffer
    LwP64          bufferLength LW_ALIGN_BYTES(8); // in/out, ptr to U32
    LwU32          displayMap;
} LW_CFGEX_GET_LOGICAL_DEV_EDID_PARAMS;

#define LW_CFGEX_GET_I2C_PORT_ID                    336
typedef struct {
    LwU32 DeviceMap;      // in: set bit(s) corresponding to device for which the I2C port ID is needed
    LwU16 I2c[LW_DISPLAY_DEVICE_MAP_MAX];  // out: I2C port ID's for any/all possible devices
} LW_CFGEX_GET_I2C_PORT_ID_PARAMS;

// This function returns the overdrive data per panel strap to the
// the client.  See bug 135902 for details.
#define LW_CFGEX_GET_OVERDRIVE_DATA              361
typedef struct
{
    LwU32         DisplayMask;  // Input display mask - 1 bit and only 1 bit can be set here.
    LwU8          data[256];    // Output data associated with Input display mask
} LW_CFGEX_GET_OVERDRIVE_DATA_PARAMS;

#define SYSMEMCTRL_VENDORID_LWIDIA          0x10DE

#define FOS_TYPE_NO_TAPS                    0x01

/************** Client Capabilities */
#define LW_CFGEX_SET_CLIENT_CAPABILITIES    440
// Param get structure
typedef struct {
    LwHandle      hClient;      // client handle
    LwU32         capFlags;     // client capabilities
} LW_CFGEX_SET_CLIENT_CAPABILITIES_PARAMS;

// client capability flags
#define LW_CFGEX_SET_CLIENT_CAPABILITIES_DISABLES_SPEEDSTEP    0x00000001  // bit 0
#define LW_CFGEX_SET_CLIENT_CAPABILITIES_RESERVED_1            0x00000002  // bit 1
// bit 2 -- Client will explicitly map PFIFO
#define LW_CFGEX_SET_CLIENT_CAPABILITIES_CLIENT_MAP_FIFO       0x00000004  // bit 2
#define LW_CFGEX_SET_CLIENT_CAPABILITIES_RESERVED_2            0x00000008  // bit 3

// Defs for LW_CFGEX_GET_DESKTOP_POSITION_MULTI current state
#define POS_LWRRENTLY_DEFAULT 0
#define POS_LWRRENTLY_LWSTOM  1


// This query should happen by LWSVC to the RM to get information
// when the MobileEvent is triggered.
// The LwrrentState and DirtyState are both bit flags.  The LwrrentState
// returns the current state of that flag.  The DirtyStates return which
// bits have changed since the last call.
#define LW_MOBILE_EVENT_LID_STATE              0x00000001
#define LW_MOBILE_EVENT_LID_STATE_OPEN         0x00000001
#define LW_MOBILE_EVENT_LID_STATE_SHUT         0x00000000

// LW_CFGEX_GET_CHANNEL_INFO
// Interface for retrieving information about hardware channels.
// Added primarily for the benefit of Thwap, a utility for testing the robust channels system.

#define LW_CFGEX_GET_CHANNEL_INFO                      460

// More fields will probably be added later.
typedef struct {
    LwU32           ChannelMask;    // mask of active channels
} LW_CFGEX_GET_CHANNEL_INFO_PARAMS;

#define LW_CFGEX_GET_ROBUST_CHANNEL_INFO               461
#define LW_CFGEX_SET_ROBUST_CHANNEL_INFO               462

#define LW_ROBUST_CHANNEL_ALLOCFAIL_CLIENT      0x00000001
#define LW_ROBUST_CHANNEL_ALLOCFAIL_DEVICE      0x00000002
#define LW_ROBUST_CHANNEL_ALLOCFAIL_SUBDEVICE   0x00000004
#define LW_ROBUST_CHANNEL_ALLOCFAIL_CHANNEL     0x00000008
#define LW_ROBUST_CHANNEL_ALLOCFAIL_CTXDMA      0x00000010
#define LW_ROBUST_CHANNEL_ALLOCFAIL_EVENT       0x00000020
#define LW_ROBUST_CHANNEL_ALLOCFAIL_MEMORY      0x00000040
#define LW_ROBUST_CHANNEL_ALLOCFAIL_OBJECT      0x00000080
#define LW_ROBUST_CHANNEL_ALLOCFAIL_HEAP        0x00000100

#define LW_ROBUST_CHANNEL_BREAKONERROR_DEFAULT  0x00000000
#define LW_ROBUST_CHANNEL_BREAKONERROR_DISABLE  0x00000001
#define LW_ROBUST_CHANNEL_BREAKONERROR_ENABLE   0x00000002

typedef struct {
    LwU32         Enabled;            // robust channels enabled
    LwU32         Interval;           // interval at which to ilwoke thwapping and stomping
    LwU32         ThwapChannelMask;   // mask of channels to thwap
    LwU32         StompChannelMask;   // (see fifo.c comments)
    LwU32         ThwapRepeatMask;    // mask of channels to repeatedly thwap
    LwU32         StompRepeatMask;    // mask of channels to repeatedly stomp
    LwU32         AllocFailMask;      // mask of allocation types to randomly fail
    LwU32         BreakOnError;       // enable/disable breakpoint on error
} LW_CFGEX_ROBUST_CHANNEL_INFO_PARAMS;

// LW_CFGEX_GET_CHIPSET_INFO
// Interface to get the vendor and chipset name
#define LW_CFGEX_GET_CHIPSET_INFO                        475

// Length of the strings inside the param
#define LW_CFGEX_GET_CHIPSET_INFO_NAME_LENGTH            32
typedef struct {
    LwU16 vendorID;
    LwU16 deviceID;
    char vendorName[LW_CFGEX_GET_CHIPSET_INFO_NAME_LENGTH];
    char chipsetName[LW_CFGEX_GET_CHIPSET_INFO_NAME_LENGTH];
#define LW_CFGEX_GET_CHIPSET_INFO_PARAMS_UNKNOWN_ID                      0xFFFF
} LW_CFGEX_GET_CHIPSET_INFO_PARAMS;

/******* RM Register Tracing Related ***********************************/

//
// By default we return the filtered calls the CPL wants
// so we are compatible with previous releases.  Calling
// LW_CFGEX_DAC_PERF_AGGRESSIVE will get the overclocked
// 3D level.
//
// If LW_CFGEX_DAC_PERF_INTERNAL is set we can return the
// full perf table, including base modes and overclocked
// modes.
//
// There may be as few as one and more than 3 internal modes.
//
#define LW_CFGEX_PERFCTL_L1             0
#define LW_CFGEX_PERFCTL_L2             1
#define LW_CFGEX_PERFCTL_L3             2
#define LW_CFGEX_PERFCTL_L4             3
#define LW_CFGEX_PERFCTL_L5             4
#define LW_CFGEX_PERFCTL_L6             5

// Compatability with user apps
#define LW_CFGEX_DAC_PERF_STANDARD      LW_CFGEX_PERFCTL_L1
#define LW_CFGEX_DAC_PERF_CONSERVATIVE  LW_CFGEX_PERFCTL_L2
#define LW_CFGEX_DAC_PERF_AGGRESSIVE    LW_CFGEX_PERFCTL_L3

#define LW_CFGEX_DAC_PERF_SAVE          BIT(0)      // Save previous perf state
#define LW_CFGEX_DAC_PERF_RESTORE       BIT(1)      // Restore previous perf state
#define LW_CFGEX_DAC_PERF_TEST          BIT(2)      // Force level change to test
#define LW_CFGEX_DAC_PERF_INTERNAL      BIT(3)      // Expose all internal tables

#define LW_CFGEX_FAN_SPEED_LOW      0               // Compatability 2D
#define LW_CFGEX_FAN_SPEED_MED      1               // Compatability 3D
#define LW_CFGEX_FAN_SPEED_HIGH     2               // Compatability 3D++
#define LW_CFGEX_FAN_SPEED_MAX      3               // Compatability Max
#define LW_CFGEX_FAN_SPEED_INTERNAL 0x1000          // Directly use internal
#define LW_CFGEX_FAN_SPEED_L1       0               // Levels 1 to N
#define LW_CFGEX_FAN_SPEED_L2       1
#define LW_CFGEX_FAN_SPEED_L3       2
#define LW_CFGEX_FAN_SPEED_L4       3
#define LW_CFGEX_FAN_SPEED_L5       4
#define LW_CFGEX_FAN_SPEED_L6       5
#define LW_CFGEX_FAN_SPEED_LWRRENT  100             // Current level (HW may be in transitioning)
#define LW_CFGEX_FAN_SPEED_SAMPLED  101             // Sampled from hardware

// Fan policy control
#define LW_CFGEX_FAN_POLICY_CONTROL_DEFAULT                 0x2000  // Select default fan control operation mode. Set only.
#define LW_CFGEX_FAN_POLICY_CONTROL_UNKNOWN                 0x2001  // Unknown mode. Get only.
#define LW_CFGEX_FAN_POLICY_CONTROL_MANUAL                  0x2002  // Manual mode. Get/Set
//efine LW_CFGEX_FAN_POLICY_CONTROL_PERF                    0x2003  // Perf mode. Get/Set
#define LW_CFGEX_FAN_POLICY_CONTROL_TEMPERATURE             0x2004  // Temperature mode (ADT7473). Get/Set
#define LW_CFGEX_FAN_POLICY_CONTROL_TEMPERATURE_SETFAN      0x2005  // SetFan (discrete) Temperature mode. Get/Set
#define LW_CFGEX_FAN_POLICY_CONTROL_TEMPERATURE_SW          0x2006  // Temperature mode (sw agent). Get/Set
#define LW_CFGEX_FAN_POLICY_CONTROL_GET_LWRRENT 0x2FFF  // Get the current control mode in effect. Get only.


//
// LW_CFGEX_PERF_MODE - Call into RM to specify that maximum performance mode is requested.
//
#define LW_CFGEX_PERF_MODE              496

typedef struct {
    LwU32 GlobalRefCount;  // Global reference count.  Output only.  Use for debugging.
    LwU32 Mode;            // requested mode
} LW_CFGEX_PERF_MODE_PARAMS;

#define LW_CFGEX_PERF_MODE_NONE                 0
#define LW_CFGEX_PERF_MODE_3D_BOOST             3     // Client enabled 3D with immediate boost to Performance
                                                      // level
#define LW_CFGEX_PERF_MODE_3D                   4     // Client enabled 3D with boost to Performance level
                                                      // based on HW Performance Monitor (PerfMon)
#define LW_CFGEX_PERF_MODE_BOOST                6     // Immediate boost to Performance level (Client enabled
                                                      // 3D should already exist)
#define LW_CFGEX_PERF_MODE_2D_ALWAYS          100     // Forces PerfLevel L1
#define LW_CFGEX_PERF_MODE_3D_LP_ALWAYS       101     // ... to L2
#define LW_CFGEX_PERF_MODE_3D_ALWAYS          102     // ... to L3

#define LW_CFGEX_PERF_MODE_ALWAYS_CLEAR       200     // Clear forced PerfLevel

// LW_CFGEX_DISPLAY_IMAGE_POSITION -- Control the position of the image on the display device.
// This call will succeed only for devices that are lwrrently positionable (ie they can be positioned and are active)
// This call will also fail when positions requested are not within range for the target device.
#define LW_CFGEX_DISPLAY_IMAGE_POSITION                 513
// Param Structure
typedef struct {
    LwU32 DeviceMap;          // Device map for target device
    LwS32 HorizontalOffset;   // Steps (+/-) on the horizontal axis;
    LwS32 VerticalOffset;     // Steps (+/-) on the vertical axis;
} LW_CFGEX_DISPLAY_IMAGE_POSITION_PARAMS;

#ifndef BIT
#define BIT(b)          (1U<<(b))
#endif

//
// LW_CFGEX_EVENT_HANDLE
// Set/get the event handle associated with the event ordinal. Please use this function rather than
// the older LW_CFGEX_SET_EVENT_HANDLES so that we can safely remove it in the near future.
#define LW_CFGEX_EVENT_HANDLE                           550
// Param Structure
typedef struct {
    LwU32   EventNotifyType;
    LwU32   EventOrdinal;
    LwP64   EventHandle LW_ALIGN_BYTES(8);
} LW_CFGEX_EVENT_HANDLE_PARAMS;

// Event ordinals
#define LW_CFGEX_EVENT_HANDLE_ICON_BEGIN                    0   //
#define LW_CFGEX_EVENT_HANDLE_ICON_END                      1   //
#define LW_CFGEX_EVENT_HANDLE_SCALE_EVENT                   2   //
#define LW_CFGEX_EVENT_HANDLE_FULL_SCREEN_DX_EVENT          3   // Full-Screen DX event
#define LW_CFGEX_EVENT_HANDLE_THERMAL_EVENT                 4   // Temperature has exceeded limit
#define LW_CFGEX_EVENT_HANDLE_POWER_CONNECTOR_EVENT         5   // Auxillary power disconnected

#define LW_CFGEX_EVENT_HANDLE_MOBILE_EVENT                  6   // Mobile specific events (Lid state changed...)
                                                                // must call LW_CFGEX_GET_MOBILE_EVENT_DATA
                                                                // to get specific data
#define LW_CFGEX_EVENT_HANDLE_INHIBIT_EVENT                 7   // inhibit flags have changed
#define LW_CFGEX_EVENT_HANDLE_HOT_PLUG_EVENT                8   // A display device hot plug/unplug has oclwrred
#define LW_CFGEX_EVENT_HANDLE_SR_EVENT                      9   // A silent running state change has oclwrred
#define LW_CFGEX_EVENT_HANDLE_THERMAL_EVENT_HW              10  // Signal on HW controlled thermal related performance decrease
#define LW_CFGEX_EVENT_HANDLE_THERMAL_EVENT_SW              11  // Signal on SW controlled thermal related performance decrease
#define LW_CFGEX_EVENT_HANDLE_AGP_CONFIGURATION_EVENT       12  //
#define LW_CFGEX_EVENT_HANDLE_SMARTDIMMER_EVENT             13  // SmartDimmer event
#define LW_CFGEX_EVENT_HANDLE_EXT_PERF_CONTROL_EVENT        14  // Externally controlled perf decrease oclwrred
#define LW_CFGEX_EVENT_HANDLE_POWER_SUPPLY_CAPACITY_EVENT   15  // Power supply capacity has changed
#define LW_CFGEX_EVENT_HANDLE_SKIP_DSI_SUPERVISOR_2_EVENT   16  // Skip restarting LW50's DSI after 2nd interrupt for RM Client testers
#define LW_CFGEX_EVENT_HANDLE_INDUCE_DEVICE_SCAN_EVENT      17  // Induce device scan. e.g. if devices change due to a change in
#define LW_CFGEX_EVENT_HANDLE_FAN_STALL_EVENT               18  // Fan should be spinning, but is not.
#define LW_CFGEX_EVENT_HANDLE_SET_BRIGHTNESS_REQUEST        19  // BIOS->RM would like to change brightness, does LWCPL agree?
#define LW_CFGEX_EVENT_HANDLE_COPP_EVENT                    20  // An HDCP or Copy-protection related event has oclwred
#define LW_CFGEX_EVENT_HANDLE_SPDIF_AFTER_TOSLINK_EVENT     21  // Activity was detected on SPDIF after having detected and configured TOSLINK
#define LW_CFGEX_EVENT_HANDLE_DOCK_WAR_922492_EVENT         22  // Replaces INDUCE_DEVICE_SCAN for undock->dock transition, see bug 922492

//
// LW_CFGEX_GET_COPP_EVENT_DETAILS
// Pass back the necessary information about what
// action to take and what display to take it on.
//
#define LW_CFGEX_GET_COPP_EVENT_DETAILS   606
typedef struct
{
    LwU32 DisplayId;
    LwU32 Action;
} LW_CFGEX_COPP_EVENT_PARAMS;

#define LW_CFGEX_COPP_EVENT_ILWALID     0x00000000
#define LW_CFGEX_COPP_EVENT_VP_CHECK    0x00000001

#if !defined(XAPIGEN)
#pragma pack()
#endif

// LwRmOsConfigSet/Get parameters
//
// definitions for OS-specific versions of the config get/set calls.
//   When possible, new config calls should be added to the standard
//   config calls instead of here.
//

//
// tell Apple drivers to step aside so mods can play around
// w/ all fb.
// Input param:
//    BIT(31):   1 -->  step aside
//               0 -->  resume normal operation
//

#define LW_OSCFG_APPLE_STEP_ASIDE                0x0004


// Apple on-board mux to select output device
#define LW_OSCFG_APPLE_SYSTEM_OUTPUT_MUX         0x0005
#define LW_OSCFG_APPLE_SYSTEM_OUTPUT_MUX_LCD     1
#define LW_OSCFG_APPLE_SYSTEM_OUTPUT_MUX_TV      2
#define LW_OSCFG_APPLE_SYSTEM_OUTPUT_MUX_VGA     3

////////////////////////////////////////////////////////////////////
//
// LwRmOsConfigSetEx/GetEx parameters
//
//

//
// Get info about the cards in the system
//
typedef struct
{
    LwU32    flags;               // see below
    LwU32    instance;            // resman's ordinal for the card
    LwU32    bus;                 // bus number (PCI, AGP, etc)
    LwU32    slot;                // card slot
    LwU32    vendor_id;           // PCI vendor id
    LwU32    device_id;
    LwU32    interrupt_line;
} LWCARDINFO;

#define LW_CARD_INFO_FLAG_PRESENT       0x0001


#define LW_OSCFGEX_GET_CARD_INFO           100

typedef struct {
    LwU32         NumCards;                // input size of buffer; output # cards
    LWCARDINFO   *pCardInfo;
} LW_OSCFGEX_GET_CARD_INFO_PARAMS;

// Remove unused LW_CFGEX_GET_HEAD_AND_DISPLAY_MASK but retain some enumerants used by clients
#define LW_CFGEX_GET_HEAD_AND_DISPLAY_MASK_FORCE_DETECT 0x80

#define LW_CFGEX_SET_DISPLAYNODE_MASK   104
typedef struct
{
    LwU32             displayMask;  //this should be the same mask that 7c was allocated with
    LwU32             nodeLetter;   //LWDA,Display-X, where X is A-Z, passed into RM
} LW_CFGEX_SET_DISPLAYNODE_MASK_PARAMS;

#define LW_OSCFG_TRIGGER_HOTPLUG_CALLBACK 105

#define LW_OSCFGEX_QUERY_FOR_OVERRIDABLE_DISPLAYS 106
typedef struct
{
    LwU32             displayMask;
    char displayNodeLetter;
} LW_OSCFGEX_QUERY_FOR_OVERRIDABLE_DISPLAYS_PARAMS;

#define LW_ACCEL_STEPASIDE 0
#define LW_ACCEL_RETURN    1

#define LW_CFGEX_GET_GR_TILEINFO    288

typedef struct
{
    LwU32 reqXBytes; // required X alignment in bytes
    LwU32 reqYLines; // required Y alignment
    LwU32 optYLines; // optimal Y alignment
} LW_CFGEX_GET_GR_TILEINFO_PARAMS;

#define LW_OSCFGEX_VARIABLE_FAN_SPEED_CONTROL 289

// this function will only return an error if its called with a bad parameter block. Otherwise, RM is smart enough to ignore TargetValue
// if things get out of hand. No error will be returned.
typedef struct
{
    LwU32 TargetValue;   // this is the portion of the pwm period we should power the fan. We clamp to max period if you go over.
                         // (returns target value on get, sets the target value on set)
    LwU32 MaxPeriod;     // this is the maximum period of the pulse to the fan, max period is returned on get or set
    LwU32 LwrrentRPMS;   // we have no tach, output always zero on get or set
} LW_OSCFGEX_VARIABLE_FAN_SPEED_CONTROL_PARAMS;

// Connection ignoring (prevents any matching display from being detected).
#define LW_OSCFGEX_APPLE_CONNECTION_IGNORE              290

typedef struct
{
    LwU32 head;
    LwU32 ignoreMask;
} LW_OSCFGEX_APPLE_CONNECTION_IGNORE_PARAMS;

#define LW_VARIABLE_FAN_SPEED_CONTROL_STATUS_TOO_HIGH (1<<0)
#define LW_VARIABLE_FAN_SPEED_CONTROL_STATUS_TOO_LOW  (1<<1)
#define LW_VARIABLE_FAN_SPEED_CONTROL_STATUS_NO_FAN   (1<<2)

#define LW_CFGEX_GET_GPU_INFO    293

// bit definitions enumerating LW_CFGEX_GET_GPU_INFO_PARAMS.HostCaps
#define LW_CFGEX_GPUINFO_HOSTCAPS_SEMA_ACQUIRE_BUG_105665        0x00000001
#define LW_CFGEX_GPUINFO_HOSTCAPS_DUP_CMPLT_BUG_126020           0x00000002
#define LW_CFGEX_GPUINFO_HOSTCAPS_ZOOMBLOAT_BUG_134322           0x00000004
#define LW_CFGEX_GPUINFO_HOSTCAPS_SYS_SEMA_DEADLOCK_BUG_148216   0x00000008
#define LW_CFGEX_GPUINFO_HOSTCAPS_SLOWSLI                        0x00000010
#define LW_CFGEX_GPUINFO_HOSTCAPS_SEMA_READ_ONLY_BUG             0x00000020

// bit definitions enumerating LW_CFGEX_GET_GPU_INFO_PARAMS.VidCaps
#define LW_CFGEX_GPUINFO_VIDCAPS_CAN_ACCESS_NON_CONTIG_SYS_MEM   0x00000001
#define LW_CFGEX_GPUINFO_VIDCAPS_PURE_VIDEO_SUPPORTED            0x00000002

// bit definitions enumerating LW_CFGEX_GET_GPU_INFO_PARAMS.GrCaps
#define LW_CFGEX_GPUINFO_GRCAPS_SET_SHADER_PACKER_SUPPORTED      0x00000001
#define LW_CFGEX_GPUINFO_GRCAPS_SET_SHADER_SAMPLE_MASK_SUPPORTED 0x00000004
#define LW_CFGEX_GPUINFO_GRCAPS_AA_FOS_GAMMA_COMP_SUPPORTED      0x00000008
#define LW_CFGEX_GPUINFO_GRCAPS_FP16_TEXTURE_BLENDING_SUPPORTED  0x00000100

typedef struct _LW_CFGEX_GET_GPU_INFO_PARAMS
{
    LwU32   NumVpes;                // deprecate after AURORA Removal
    LwU32   NumShaderPipes;
    LwU32   HostCaps;
    LwU32   VidCaps;
    LwU32   GrCaps;
    LwU32   NumCompressibleBytes;
    LwU32   NumBufferAlignmentBytes;
    LwU32   NumSwizzledAlignmentBytes;
    LwU32   VertexCacheSize;
    LwU32   RegFloorSweep;          // contents of this is chip specific - var is not meant for general use.
    LwU32   ThreadStackScalingFactor;
    LwU32   DramPageStride;
} LW_CFGEX_GET_GPU_INFO_PARAMS;

// ----------------------------------------------------------------------------

#define LW_CFGEX_DISPLAY_NOTIFIER_MISS_RECORD  291

typedef struct _DISPLAY_NOTIFIER_MISS_RECORD_EVENT {
    LwHandle hDispChannel;          // event_log
    LwHandle hLastChannel;          // event_log
    LwU32    lwDispDmaCount;        // event_log
    LwU32    lwDispDmaCachedGet;    // event_log
    LwU32    lwDispDmaCachedPut;    // event_log
} DISPLAY_NOTIFIER_MISS_RECORD_EVENT;

/* XAPIGEN - deprecated interface + ill-defined union   */
/*           4/09 still used on Vista                   */
/*           For now, just always encode the "event" portion of the union */
typedef struct {

    LwU32    bValidLog;                    // flag for valid log

    /* Info to save */
    LwU32    ulChannelId;
    LwHandle hClient;
    LwHandle hDevice;
    LwU32    lwDispDmaLastFree;
    LwP64    pv_lwDispDmaChannel LW_ALIGN_BYTES(8);
    LwP64    lwDispDmaFifo       LW_ALIGN_BYTES(8);

#if defined(XAPIGEN)
    /* XAPI: hack around ill-defined union */
    struct {
        DISPLAY_NOTIFIER_MISS_RECORD_EVENT DisplayRec;
    } u;
#else
    union {
        DISPLAY_NOTIFIER_MISS_RECORD_EVENT DisplayRec;
        LwU32  DumpBuf;
    } u;
#endif

    /* Parsing LW_NOTIFIER struct */

    LwP64          pNotifierInQuestion LW_ALIGN_BYTES(8);
    LwU32          notifierObject;
    LwU32          notifierClass;

} LW_CFGEX_DISPLAY_NOTIFIER_MISS_RECORD_PARAMS;

#define BKSV_KEY_SIZE (5)
#define AKSV_KEY_SIZE (5)

//this one requires no value and returns no value
//performs hotplug callback action to the operating system
//action performed on LwRmOsCfgSet() call

/************** Quiet State Get / Set *******************/

#define LW_CFGEX_QUIET_STATE_NOT_QUIET   0
#define LW_CFGEX_QUIET_STATE_QUIET       1

// Acquire an lwTimer reading
#define LW_CFGEX_GET_TMR                296
typedef struct {
    LwU32   TimeLo;
    LwU32   TimeHi;
} LW_CFGEX_GET_TMR_PARAMS;

//
// LW_CFGEX_GET_IFB_DATA -
// Returns data from the specified fb offset via Indirect
// Indirect Framebuffer Access (if available).
//
#define LW_CFGEX_GET_IFB_DATA_FLAGS_32BIT       BIT(0)
#define LW_CFGEX_GET_IFB_DATA_FLAGS_16BIT       BIT(1)
#define LW_CFGEX_GET_IFB_DATA_FLAGS_8BIT        BIT(2)
// // Limit for IFB data transfer is 2MB.
#define  LW_CFGEX_GET_IFB_DATA_SIZE_LIMIT       (0x200000)

//
// LW_CFGEX_SET_IFB_DATA -
//
#define LW_CFGEX_SET_IFB_DATA_FLAGS_32BIT       BIT(0)
#define LW_CFGEX_SET_IFB_DATA_FLAGS_16BIT       BIT(1)
#define LW_CFGEX_SET_IFB_DATA_FLAGS_8BIT        BIT(2)

#ifdef LW_VERIF_FEATURES

#define LW_CFGEX_SET_CTXSW_LIMITERS 315

typedef struct _limiter_table {
    LwU32 count;                    // Total Number of Elements
    char ** ppStr;
    LwS32 * pLimiters;
} LIMITER_TABLE, * PLIMITER_TABLE;

typedef struct {
    LwU32           totalCount;
    LwHandle        hChannel;
    PLIMITER_TABLE  pLimiterTable;
} LW_CFGEX_CTXSW_LIMITERS_PARAMS, * PLW_CFGEX_CTXSW_LIMITERS_PARAMS;
#endif // LW_VERIF_FEATURES

#ifdef LW_VERIF_FEATURES

// Remove unused LW_CFGEX_SET_GRCTX_INIT_OVERRIDES but retain structs used by clients
typedef struct {
    LwU32 regAddr;   // Register address
    LwU32 andMask;   // Negated and ANDed to original value
    LwU32 orMask;  // ORed to the original value
} LW_CFGEX_SET_GRCTX_INIT_OVERRIDES_PARAMS, *PLW_CFGEX_SET_GRCTX_INIT_OVERRIDES_PARAMS;
#endif // LW_VERIF_FEATURES

//------------------------------------------------------------------------------
// Configuration manager reserved properties.
//
// #define LW_CFGEX_RESERVED 150
//------------------------------------------------------------------------------

typedef struct
{
    LwU32 Property;
    LwP64 In LW_ALIGN_BYTES(8);
    LwP64 Out LW_ALIGN_BYTES(8);
} LW_CFGEX_RESERVED_PROPERTY;


typedef enum
{
    // Register read and write.
     PROPERTY_REG_RD08               = 0x0      // In:[Offset]       Out:[Data]
    ,PROPERTY_REG_RD16               = 0x1      // In:[Offset]       Out:[Data]
    ,PROPERTY_REG_RD32               = 0x2      // In:[Offset]       Out:[Data]
    ,PROPERTY_REG_WR08               = 0x3      // In:[Offset, Data] Out:[]
    ,PROPERTY_REG_WR16               = 0x4      // In:[Offset, Data] Out:[]
    ,PROPERTY_REG_WR32               = 0x5      // In:[Offset, Data] Out:[]

    // Frame buffer read and write.
    ,PROPERTY_FB_RD08                = 0x6      // In:[Offset]       Out:[Data]
    ,PROPERTY_FB_RD16                = 0x7      // In:[Offset]       Out:[Data]
    ,PROPERTY_FB_RD32                = 0x8      // In:[Offset]       Out:[Data]
    ,PROPERTY_FB_WR08                = 0x9      // In:[Offset, Data] Out:[]
    ,PROPERTY_FB_WR16                = 0xA      // In:[Offset, Data] Out:[]
    ,PROPERTY_FB_WR32                = 0xB      // In:[Offset, Data] Out:[]

    // PCI read and write.
    ,PROPERTY_PCI_RD08               = 0xC      // In:[Bus, Device, Function, Offset]       Out:[Data]
    ,PROPERTY_PCI_RD16               = 0xD      // In:[Bus, Device, Function, Offset]       Out:[Data]
    ,PROPERTY_PCI_RD32               = 0xE      // In:[Bus, Device, Function, Offset]       Out:[Data]
    ,PROPERTY_PCI_WR08               = 0xF      // In:[Bus, Device, Function, Offset, Data] Out:[]
    ,PROPERTY_PCI_WR16               = 0x10     // In:[Bus, Device, Function, Offset, Data] Out:[]
    ,PROPERTY_PCI_WR32               = 0x11     // In:[Bus, Device, Function, Offset, Data] Out:[]

    // Set clocks.
    ,PROPERTY_SET_GRAPHICS_CLOCK     = 0x12     // In:[GFreq (Hz), SFreq (Hz), RFreq (Hz)]  Out:[]
    ,PROPERTY_SET_MEMORY_CLOCK       = 0x13     // In:[Frequency (Hz)]       Out:[]
    ,PROPERTY_SET_PIXEL_CLOCK        = 0x14     // In:[Head, Frequency (Hz)] Out:[]

    // CR read and write.
    ,PROPERTY_CR_RD32                = 0x15     // In:[Head, Offset]            Out:[Data]
    ,PROPERTY_CR_WR32                = 0x16     // In:[Head, Offset, Data]      Out:[]

    // TMDS read and write.
    ,PROPERTY_TMDS_RD32              = 0x17     // In:[Link, Offset]            Out:[Data]
    ,PROPERTY_TMDS_WR32              = 0x18     // In:[Link, Offset, Data]      Out:[]

    // SR read and write.
    ,PROPERTY_SR_RD32                = 0x19     // In:[Head, Offset]            Out:[Data]
    ,PROPERTY_SR_WR32                = 0x1A     // In:[Head, Offset, Data]      Out:[]

    // AR read and write.
    ,PROPERTY_AR_RD32                = 0x1B     // In:[Head, Offset]            Out:[Data]
    ,PROPERTY_AR_WR32                = 0x1C     // In:[Head, Offset, Data]      Out:[]

    // Make osCallVideoBIOS call.
    ,PROPERTY_VBIOS_CALL             = 0x1D     // In:[EAX, EBX, ECX, EDX]      Out:[EAX, EBX, ECX, EDX]

    // Instance memory read and write
    ,PROPERTY_INST_RD08              = 0x1E     // In:[Offset]                  Out:[Data]
    ,PROPERTY_INST_WR08              = 0x1F     // In:[Offset, Data]            Out:[]
    ,PROPERTY_INST_RD32              = 0x20     // In:[Offset]                  Out:[Data]
    ,PROPERTY_INST_WR32              = 0x21     // In:[Offset, Data]            Out:[]

    // DLL reset
    ,PROPERTY_EXELWTE_DLL_RESET      = 0x22     // In:[non zero = set GR gating aggressive around DLL reset] Out:[]

    ,PROPERTY_SET_CORE_VOLTAGE       = 0x23     // In:[Voltage in mV]        Out:[Prev Voltage mV]

    // IO port byte access
    ,PROPERTY_IO_RD08                = 0x24     // In:[Offset]       Out:[Data]
    ,PROPERTY_IO_WR08                = 0x25     // In:[Offset, Data] Out:[]

    // More clocks.
    ,PROPERTY_GET_CLOCK              = 0x26     // In:[Clk] Out:[ClkSrc, Frequency (Hz)]
    ,PROPERTY_GET_GRAPHICS_CLOCK_TARGETS = 0x27     // In:[] Out:[GTargetFreq, STargetFreq, RTargetFreq (Hz)]

     // I2C Read/Write
    ,PROPERTY_I2C_RD08               = 0x28     // In:[Port, Addr, Reg]         Out:[Val]
    ,PROPERTY_I2C_WR08               = 0x29     // In:[Port, Addr, Reg, Val]    Out:[]

     // OnDemand VBlank control
    ,PROPERTY_GET_ONDEMAND_VBLANK    = 0x2A     // Out:[Legal values of LW_REG_STR_RM_ONDEMAND_VBLANK]
    ,PROPERTY_SET_ONDEMAND_VBLANK    = 0x2B     // In: [Legal values of LW_REG_STR_RM_ONDEMAND_VBLANK]    Out:[Prev value]

    ,PROPERTY_GET_VPLL_LOCK_DELAY    = 0x2E     // In: []    Out:[Get current VPLL settle delay]
    ,PROPERTY_SET_VPLL_LOCK_DELAY    = 0x2F     // In: [New VPLL settle delay]    Out:[]

    ,PROPERTY_SET_HOT_CLOCK          = 0x30     // In:[Frequency (Hz)]       Out:[]

     // GR read and write.
    ,PROPERTY_GR_RD32                = 0x31     // In:[Head, Offset]            Out:[Data]
    ,PROPERTY_GR_WR32                = 0x32     // In:[Head, Offset, Data]      Out:[]

    // GPIO access for MODS
    ,PROPERTY_GPIO_READ_IN           = 0x33     // In:[Pin]                  Out:[Data]
    ,PROPERTY_GPIO_WRITE_OUT         = 0x34     // In:[Pin, Data]            Out:[]
    ,PROPERTY_GPIO_READ_OUT          = 0x35     // In:[Pin]                  Out:[Data]
    ,PROPERTY_GPIO_GET_DIRECTION     = 0x36     // In:[Pin]                  Out:[Data]
    ,PROPERTY_GPIO_SET_DIRECTION     = 0x37     // In:[Pin, Data]            Out:[]
    ,PROPERTY_GPIO_GET_MODE          = 0x38     // In:[Pin]                  Out:[Data]
    ,PROPERTY_GPIO_SET_MODE          = 0x39     // In:[Pin, Data]            Out:[]
    ,PROPERTY_GPIO_GET_ILWERSION     = 0x3a     // In:[Pin]                  Out:[Data]
    ,PROPERTY_GPIO_SET_ILWERSION     = 0x3b     // In:[Pin, Data]            Out:[]

    // DSI_FORCE_BITS access for MODS
    ,PROPERTY_GET_DSI_FORCE_BITS     = 0x3c     // In: [Head]                Out:[Data]
    ,PROPERTY_SET_DSI_FORCE_BITS     = 0x3d     // In: [Head, Data]          Out:[]

} LW_CFGEX_RESERVED_PROPERTY_PROPERTIES;



// ***************************************************************************
// *                                                                         *
// ***IMPORTANT***  Do not add or modify CFG or CFGEX interfaces.            *
// ***IMPORTANT***  The interfaces defined in this file are *deprecated*     *
// *                Ref: bug 488474: delete CFG and CFG_EX                   *
// *                                                                         *
// ***************************************************************************




//---------------------------------------------------------------------------
//
//  Configuration Manager API.
//
//---------------------------------------------------------------------------
//
//
// DWORD LwConfigVersion(void)
//
//  Returns the revision of the ddk (config) interface built into the resource manager.
//  This is used for version continuity between all the resource manager files,
//  as well as provides the interface version for people using the config interface.
//
//  The format of this is 0xAAAABBCC, where
//   - 0xAAAA is [lwpu internal]
//   -   0xBB is the software release revision
//   -   0xCC is the minor revision
//
//
// DWORD LwConfigGet(DWORD Index, DWORD DeviceHandle)
//
//  Given an Index from LWCM.H and a pointer to a specific device (see SDK), return
//  the current configuration value.  The format of the value is dependent on the
//  index requested.
//
//
// DWORD LwConfigSet(DWORD Index, DWORD NewValue, DWORD DeviceHandle)
//
//  Given an Index from LWCM.H, a pointer to a specific device (see SDK), and a new
//  value, update the current configuration value.  This call returns the original
//  value in that configuration index.
//

/* XAPIGEN - none of this is relevant to xapigen */
#if ! defined(XAPIGEN)

#if !defined(_WIN32)
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || \
    defined(MACOS) || defined(EFI64) || defined(vxworks)
int LwConfigVersion(int);
int LwConfigGet(int, int);
int LwConfigSet(int, int, int);
int LwConfigVga(int, int);
#else
DWORD FAR PASCAL LwConfigVersion(DWORD);
DWORD FAR PASCAL LwConfigGet(void*, DWORD);
DWORD FAR PASCAL LwConfigSet(DWORD, DWORD, DWORD);
DWORD FAR PASCAL LwConfigVga(DWORD, DWORD);
#endif

#elif !defined(WINNT)
int __stdcall LwConfigVersion(void);
int __stdcall LwConfigGet(int, int);
int __stdcall LwConfigSet(int, int, int);
#endif // _WIN32


/////////////////////////////////////////////////////////////////////////////
//
// THE FOLLOWING DEFINES AND ENTRY POINTS ARE LWPU RESERVED
//
//---------------------------------------------------------------------------
//
//  Device Defines.
//
//---------------------------------------------------------------------------

//
// Parameter to DeviceGet.
//
#define LW_DEV_BASE                             1
#define LW_DEV_ALTERNATE                        2
#define LW_DEV_BUFFER_0                         3
#define LW_DEV_BUFFER_1                         4
#define LW_DEV_TIMER                            5
#define LW_DEV_PFB                              6
#define LW_DEV_PGRAPH                           7
#define LW_DEV_PRMCIO                           8
#define LW_DEV_PRMVIO                           9
#define LW_DEV_AGP                              10
#define LW_DEV_GAMMA                            11 /*NUKED!*/
#define LW_DEV_PRAMDAC                          12
#define LW_DEV_PCRTC                            13
#define LW_DEV_MAX                              13

//---------------------------------------------------------------------------
//
//  Device Pointer API.
//
//---------------------------------------------------------------------------
#ifndef WINNT
#ifndef _WIN32
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || \
    defined(MACOS) || defined(EFI64) || defined(vxworks)
int LwIoControl(int, int);
int LwDeviceBaseGet(int, int);
int LwDeviceLimitGet(int, int);
int LwDeviceSelectorGet(int, int);
int LwGetHardwarePointers(int *, int *, int *, int *);
#else
//DWORD FAR PASCAL LwIoControl(DWORD, DWORD);
DWORD FAR PASCAL LwDeviceBaseGet(DWORD, DWORD);
DWORD FAR PASCAL LwDeviceLimitGet(DWORD, DWORD);
WORD  FAR PASCAL LwDeviceSelectorGet(DWORD, DWORD);
DWORD FAR PASCAL LwGetHardwarePointers(DWORD *, DWORD *, DWORD *, DWORD *);
#endif
#else  /* WIN32 */
void __stdcall LwIoControl(int, void *);
int  __stdcall LwDeviceBaseGet(int, int);
int  __stdcall LwDeviceLimitGet(int, int);
int  __stdcall LwDeviceSelectorGet(int, int);
int  __stdcall LwGetHardwarePointers(int *, int *, int *, int *, int);
#endif // _WIN32
#endif // !WINNT
#ifdef __cplusplus
}
#endif // __cplusplus



//******************************************************************************
// adi  Global variable used to detect the OS version under which the driver
// is running; defines for the OS
// I am not quite sure if this is the best place for this thing to reside,
// but I really need to share it with all the modules
// The compiler should initialize the LWOS_Ver to 0, I hope that this will be true
// for all the OS platforms
//******************************************************************************

/*
 The externals below are initialized in lw.c after calling the OS function listed
 below:

 PsGetVersion(pMajorVersion, pMinorVersion, pBuildNumber, pCSDVersion)

             pMajorVersion ?Pointer to variable to store major OS version

                         3 = NT 3.51
                         4 = Win9x/ME/NT4
                         5 = Win2000/WinXP/Win.NET
                         6 = Wilwista

             pMinorVersion ?Pointer to variable to store minor OS version

                         51 = NT 3.51       (Major version 3)
                         0  = 95/NT 4       (Major version 4)
                         10 = Win98         (Major version 4)
                         90 = WinME         (Major version 4)
                         0  = Win2000       (Major version 5)
                         1  = WinXP         (Major version 5)
                         2  = Win.NET       (Major version 5)
                         0  = Wilwista      (Major version 6)

             pBuildNumber ?Pointer to variable to store OS build number

                         950  = Win95
                         1111 = Win98
                         2222 = Win98SE
                         1998 = WinME
                         1381 = NT 4
                         2195 = Win2000
                         2600 = WinXP
                         3790 = Win.NET
*/

#define OS_IS_NT4       0x10
#define OS_IS_W2K       0x100
#define OS_IS_WHISTLER  0x1000

// Some OS version definitions used in RM based on description above
#define OS_MAJ_IS_WIN10         10  // wddm2 based windows version
#define OS_MAJ_IS_VISTA         6   // wddm based windows version
#define OS_MAJ_IS_2K_XP_NET     5
#define OS_MIN_IS_WINXP         1
#define OS_MIN_IS_WINXP_AMD64   2
#define OS_MIN_IS_VISTA         0   // VISTA   is 6.0
#define OS_MIN_IS_WIN7          1   // WIN7    is 6.1
#define OS_MIN_IS_WIN8          2   // WIN8    is 6.2
#define OS_MIN_IS_WIN8_1        3   // WIN8.1  is 6.3
#define OS_MIN_IS_WIN10         4   // WIN10   is 6.4: TO BE REMOVED
#define OS_MIN_IS_WIN10_0       0   // WIN10   is 10.0

// LWOS_ProductType definitions
#define OS_PT_IS_WORKSTATION        1
#define OS_PT_IS_DOMAIN_CONTROLLER  2
#define OS_PT_IS_SERVER             3

// LWOS_SP_Maj definitions
#define OS_SP_MAJ_IS_NONE           0
#define OS_SP_MAJ_IS_SP1            1
#define OS_SP_MAJ_IS_SP2            2
#define OS_SP_MAJ_IS_SP3            3
#define OS_SP_MAJ_IS_UNKNOWN        0xFFFF

// LWOS_SP_Min definitions
#define OS_SP_MIN_IS_NONE           0
#define OS_SP_MIN_IS_UNKNOWN        0xFFFF

#endif // !XAPIGEN

#endif // LW_DEPRECATED_RM_CONFIG_GET_SET

#endif // _LWCM_H_
