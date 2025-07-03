/*
 * Copyright (c) 2001-2013 LWPU CORPORATION.  All rights reserved.
*
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
*/


#ifndef _cle276_h_
#define _cle276_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LWE2_AVP */
#define LWE2_AVP                                        (0x0000E276)

/* event values */
#define LWE276_NOTIFIERS_APP_NOTIFY                                (0)
#define LWE276_NOTIFIERS_DH_COMPLETE                               (1)
#define LWE276_NOTIFIERS_MAXCOUNT                                  (2)

 /*
 * At alloc time clients must pass in a LW_VP_ALLOCATION_PARAMETERS param. 
 * LW_VP_ALLOCATION_PARAMETERS.hMemoryCmdBuffer should specify the Cmd Buffer 
 * Memory use to program LwE276Control.
 * LW_VP_ALLOCATION_PARAMETERS.pControl will be filled in upon alloc with a
 * mapping to the LwE276Control area. 
 *  
 */

typedef volatile struct _cle276_tag0 {
LwU32 Reserved00[5];
LwU32 DmaStart;
LwU32 Reserved01[2];
LwU32 DmaEnd;
LwU32 Reserved02[7];
LwU32 Put;
LwU32 Reserved03[15];
LwU32 Get;
LwU32 Reserved04[9];
LwU32 SyncPtIncrTrapEnable; // Route syncpt increment methods to CPU
LwU32 WatchdogTimeout;      // Channel watchdog timeout value in microseconds (0=disabled)
LwU32 IdleNotifyEnable;     // 1: enable notifications when channel is idle
LwU32 IdleNotifyDelay;      // Minimum idle time in milliseconds before idle notification event
LwU32 IdleClocksEnable;     // 1: disable automatic engine clock gating
LwU32 IramClockGating;      // 1: enable IRAM/AVPUCQ clock gating

LwU32 Idle;
LwU32 OutboxData;
LwU64 AppStartTime;
LwU32 AppIntrEnable;
LwU32 AppInIRam;
LwU32 IRamUCodeAddr;
LwU32 IRamUCodeSize;
LwU32 SchedAppEvents;       // Number of scheduler events during most recent application exelwtion
LwU32 SchedPeakBusyTime;    // Longest continuous non-idle run time (usec) of AVP application thread (within the last ~50-100ms)
LwU64 SchedAvpBusyTime;     // Total AVP run time in this channel (usec)
LwU64 SchedAvpIdleTime;     // Total AVP time in idle thread (usec)
LwU64 SchedChIdleTime;      // Total AVP channel idle time in this channel (usec)
LwU32 SchedEventAppStart;   // Scheduler event counter at app start time
LwU32 SchedEventCounter;    // Current scheduler event counter
LwU64 SchedStatTime;        // Time of stats history update
LwU32 SchedStatHistory[4];  // Moving window history
LwU32 SchedReserved[5];
LwU32 CachedUCodeAddr;      // Last SDRAM address of ucode lwrrently in IRAM
LwU32 IRamUCodeUid0;        // uid of ucode lwrrently in IRAM
LwU32 IRamUCodeUid1;
LwU32 DbgState[32];
LwU32 OsMethodData[16];
LwU32 AppMethodData[128];
} LwE276Control;

#define LWE276_UCODE_MAGIC      0xc0de4a76

typedef struct {
    LwU32 ucode_magic;      // =LWE276_UCODE_MAGIC
    LwU32 entry_point;      // entry-point relative to ucode image
    LwU32 num_relocations;  // number of entries in relocation table (immediately following header)
    LwU32 lwrrent_base;     // current relocation base address (only relocate if different than SetMicrocodeB)
    LwU32 ucode_uid0;       // unique ID of microcode (part1)
    LwU32 ucode_uid1;       // unique ID of microcode (part2)
    LwU32 seg_desc_offset;  // offset of LwE276MicrocodeSegDesc (0=not present)
    LwU32 reserved;         // reserved for AVP OS
} LwE276MicrocodeHeader;

typedef struct {
    LwU32 first_iram_reloc;     // index of 1st iram segment relocation in relocation table
    LwU32 num_iram_relocs;      // number of iram segment relocations in relocation table
    LwU32 iram_segment_offset;  // offset of iram segment
    LwU32 iram_segment_size;    // size of iram segment
    LwU32 iram_align_mask;      // required alignment of iram segment
    LwU32 rel_iram_offset;      // current iram segment address relative to lwrrent_base
    LwU32 first_dynamic_reloc;  // index of 1st dynamic segment relocation in relocation table
    LwU32 num_dynamic_relocs;   // number of dynamic segment relocations in relocation table
    LwU32 dynamic_segment_offset; // offset of dynamic segment
    LwU32 dynamic_segment_size; // size of dynamic segment
    LwU32 dynamic_align_mask;   // required alignment of dynamic segment
    LwU32 rel_dynamic_offset;   // current dynamic segment address relative to lwrrent_base
    LwU32 first_sdram_reloc;    // index of 1st sdram relocation in relocation table
    LwU32 num_sdram_relocs;     // number of sdram relocations in relocation table
    LwU32 reserved[2];
} LwE276MicrocodeSegDesc;

// AVP Host1x methods
#define LWE276_INCR_SYNCPT                              (0x00000000)
#define LWE276_INCR_SYNCPT_INDX                         7:0


// AVP OS methods
#define LWE276_NOP                                      (0x00000080)
#define LWE276_NOP_PARAMETER                            31:0

#define LWE276_SET_APP_TIMEOUT                          (0x00000084)
#define LWE276_SET_APP_TIMEOUT_USEC                     31:0

#define LWE276_SET_MICROCODE_A                          (0x00000085)
#define LWE276_SET_MICROCODE_A_UPPER                    31:0

#define LWE276_SET_MICROCODE_B                          (0x00000086)
#define LWE276_SET_MICROCODE_B_LOWER                    31:0

#define LWE276_SET_MICROCODE_C                          (0x00000087)
#define LWE276_SET_MICROCODE_C_SIZE                     31:0

#define LWE276_EXELWTE                                  (0x00000088)
#define LWE276_EXELWTE_APPID                            7:0
#define LWE276_EXELWTE_NOTIFY                           12:12
#define LWE276_EXELWTE_NOTIFY_DISABLED                  (0x00000000)
#define LWE276_EXELWTE_NOTIFY_ENABLED                   (0x00000001)
#define LWE276_EXELWTE_NOTIFY_ON                        13:13
#define LWE276_EXELWTE_NOTIFY_ON_END                    (0x00000000)
#define LWE276_EXELWTE_NOTIFY_ON_BEGIN                  (0x00000001)
#define LWE276_EXELWTE_AWAKEN                           14:14
#define LWE276_EXELWTE_AWAKEN_DISABLED                  (0x00000000)
#define LWE276_EXELWTE_AWAKEN_ENABLED                   (0x00000001)
#define LWE276_EXELWTE_LOCATION                         15:15
#define LWE276_EXELWTE_LOCATION_SDRAM                   (0x00000000)
#define LWE276_EXELWTE_LOCATION_IRAM                    (0x00000001)
#define LWE276_EXELWTE_RELOAD                           16:16
#define LWE276_EXELWTE_RELOAD_FALSE                     (0x00000000)
#define LWE276_EXELWTE_RELOAD_TRUE                      (0x00000001)

#define LWE276_SEMAPHORE_A                              (0x00000089)
#define LWE276_SEMAPHORE_A_UPPER                        31:0

#define LWE276_SEMAPHORE_B                              (0x0000008A)
#define LWE276_SEMAPHORE_B_LOWER                        31:0

#define LWE276_SEMAPHORE_C                              (0x0000008B)
#define LWE276_SEMAPHORE_C_PAYLOAD                      31:0

#define LWE276_SEMAPHORE_D                              (0x0000008C)
#define LWE276_SEMAPHORE_D_STRUCT_SIZE                  0:0
#define LWE276_SEMAPHORE_D_STRUCT_SIZE_ONE              (0x00000000)
#define LWE276_SEMAPHORE_D_STRUCT_SIZE_FOUR             (0x00000001)
#define LWE276_SEMAPHORE_D_AWAKEN                       8:8
#define LWE276_SEMAPHORE_D_AWAKEN_DISABLED              (0x00000000)
#define LWE276_SEMAPHORE_D_AWAKEN_ENABLED               (0x00000001)

#define LWE276_NOTIFY                                   (0x0000008D)
#define LWE276_NOTIFY_PARAMETER                         31:0

// AVP Application methods
#define LWE276_PARAMETER_METHOD(i)                      (0x000000C0+(i))
#define LWE276_PARAMETER_METHOD__SIZE_1                 (128)
#define LWE276_PARAMETER_METHOD_DATA                    31:0


// Interrupt codes through inbox register (avp->cpu)
#define LWE276_OS_INTERRUPT_NOP                         (0x00000000)
#define LWE276_OS_INTERRUPT_TIMEOUT                     (0x00000001)
#define LWE276_OS_INTERRUPT_SEMAPHORE_AWAKEN            (0x00000002)
#define LWE276_OS_INTERRUPT_EXELWTE_AWAKEN              (0x00000004)
#define LWE276_OS_INTERRUPT_DEBUG_STRING                (0x00000008)
#define LWE276_OS_INTERRUPT_DH_KEYEXCHANGE              (0x00000010)
#define LWE276_OS_INTERRUPT_APP_NOTIFY                  (0x00000020)
#define LWE276_OS_INTERRUPT_VIDEO_IDLE                  (0x00000040)
#define LWE276_OS_INTERRUPT_AUDIO_IDLE                  (0x00000080)
#define LWE276_OS_INTERRUPT_SYNCPT_INCR_TRAP            (0x00002000)
#define LWE276_OS_INTERRUPT_AVP_BREAKPOINT              (0x00800000)
#define LWE276_OS_INTERRUPT_AVP_FATAL_ERROR             (0x01000000)
#define LWE276_OS_SYNCPT_INCR_TRAP_GET_SYNCPT(x)        (((x)>>14)&0x1f)

// Outbox codes for AOS (cpu->avp)
#define LWE276_OS_OUTBOX_NOP                            (0x00000000)    // ping (no-op) -> os hang detection ?
#define LWE276_OS_OUTBOX_SIGNAL_VIDEO                   (0x00000001)    // wake up video channel
#define LWE276_OS_OUTBOX_SIGNAL_AUDIO                   (0x00000002)    // wake up audio channel
#define LWE276_OS_OUTBOX_DISABLE_INTERRUPTS             (0x00000010)    // disable inbox interrupts (potentially stalling avp)
#define LWE276_OS_OUTBOX_ENABLE_INTERRUPTS              (0x00000020)    // enable inbox interrupts (default)
#define LWE276_OS_OUTBOX_DH_KEYEXCHANGE                 (0x00000040)

// ARB semaphore IDs for HW engines arbitration (corresponding bit in ARB_SEMA_SMP)
#define LWE276_OS_ARBSEMA_BSEV                          0
#define LWE276_OS_ARBSEMA_BSEA                          1

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cle276_h_ */

