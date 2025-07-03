/*
 * Copyright (c) 1993-2021, LWPU CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef LWERROR_H
#define LWERROR_H
/******************************************************************************
*
*   File:  lwerror.h
*
*   Description:
*       This file contains the error codes set when the error notifier
*   is signaled.
*
******************************************************************************/

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
//
// NOTE: Please keep the ROBUST_CHANNEL_ERROR_STR{_PUBLIC},
//       SILENT_RUNNING_ERROR_STR_PUBLIC, DISPLAY_ERROR_STR_PUBLIC
//       and FB_ERROR_STR_PUBLIC tables below in sync with the
//       ROBUST_CHANNEL_*, SILENT_RUNNINT_*, DISPLAY_* and FB_*
//       #defines.
//
// NOTE: Enumeration in rc.proto must match, and must remain consistent
//       across branches.  DO NOT DELETE OR RENUMBER ENTRIES.
//
// NOTE: Please update Xid wiki page and the Xid doc are per changes done here.
// Wiki: https://wiki.lwpu.com/engwiki/index.php/Resman/PSG/Projects/Xid_Reporting
// Doc : <branch>/docs/dita/xid-errors/word/DA-06468-001_v012.docx
//
#define ROBUST_CHANNEL_FIFO_ERROR_FIFO_METHOD           (1)
#define ROBUST_CHANNEL_FIFO_ERROR_SW_METHOD             (2)
#define ROBUST_CHANNEL_FIFO_ERROR_UNK_METHOD            (3)
#define ROBUST_CHANNEL_FIFO_ERROR_CHANNEL_BUSY          (4)
#define ROBUST_CHANNEL_FIFO_ERROR_RUNOUT_OVERFLOW       (5)
#define ROBUST_CHANNEL_FIFO_ERROR_PARSE_ERR             (6)
#define ROBUST_CHANNEL_FIFO_ERROR_PTE_ERR               (7)
#define ROBUST_CHANNEL_FIFO_ERROR_IDLE_TIMEOUT          (8)
#define ROBUST_CHANNEL_GR_ERROR_INSTANCE                (9)
#define ROBUST_CHANNEL_GR_ERROR_SINGLE_STEP             (10)
#define ROBUST_CHANNEL_GR_ERROR_MISSING_HW              (11)
#define ROBUST_CHANNEL_GR_ERROR_SW_METHOD               (12)
#define ROBUST_CHANNEL_GR_EXCEPTION                     (13)
#define ROBUST_CHANNEL_GR_ERROR_SW_NOTIFY               (13)
#define ROBUST_CHANNEL_FAKE_ERROR                       (14)
#define ROBUST_CHANNEL_SCANLINE_TIMEOUT                 (15)
#define ROBUST_CHANNEL_VBLANK_CALLBACK_TIMEOUT          (16)
#define ROBUST_CHANNEL_PARAMETER_ERROR                  (17)
#define ROBUST_CHANNEL_BUS_MASTER_TIMEOUT_ERROR         (18)
#define ROBUST_CHANNEL_DISP_MISSED_NOTIFIER             (19)
#define ROBUST_CHANNEL_MPEG_ERROR_SW_METHOD             (20)
#define ROBUST_CHANNEL_ME_ERROR_SW_METHOD               (21)
#define ROBUST_CHANNEL_VP_ERROR_SW_METHOD               (22)
#define ROBUST_CHANNEL_RC_LOGGING_ENABLED               (23)
#define ROBUST_CHANNEL_GR_SEMAPHORE_TIMEOUT             (24)
#define ROBUST_CHANNEL_GR_ILLEGAL_NOTIFY                (25)
#define ROBUST_CHANNEL_FIFO_ERROR_FBISTATE_TIMEOUT      (26)
#define ROBUST_CHANNEL_VP_ERROR                         (27)
#define ROBUST_CHANNEL_VP2_ERROR                        (28)
#define ROBUST_CHANNEL_BSP_ERROR                        (29)
#define ROBUST_CHANNEL_BAD_ADDR_ACCESS                  (30)
#define ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT           (31)
#define ROBUST_CHANNEL_PBDMA_ERROR                      (32)
#define ROBUST_CHANNEL_SEC_ERROR                        (33)
#define ROBUST_CHANNEL_MSVLD_ERROR                      (34)
#define ROBUST_CHANNEL_MSPDEC_ERROR                     (35)
#define ROBUST_CHANNEL_MSPPP_ERROR                      (36)
#define ROBUST_CHANNEL_FECS_ERR_UNIMP_FIRMWARE_METHOD   (37)
#define ROBUST_CHANNEL_FECS_ERR_WATCHDOG_TIMEOUT        (38)
#define ROBUST_CHANNEL_CE0_ERROR                        (39)
#define ROBUST_CHANNEL_CE1_ERROR                        (40)
#define ROBUST_CHANNEL_CE2_ERROR                        (41)
#define ROBUST_CHANNEL_VIC_ERROR                        (42)
#define ROBUST_CHANNEL_RESETCHANNEL_VERIF_ERROR         (43)
#define ROBUST_CHANNEL_GR_FAULT_DURING_CTXSW            (44)
#define ROBUST_CHANNEL_PREEMPTIVE_REMOVAL               (45)
#define ROBUST_CHANNEL_GPU_TIMEOUT_ERROR                (46)
#define ROBUST_CHANNEL_MSENC_ERROR                      (47)
#define ROBUST_CHANNEL_LWENC0_ERROR                     ROBUST_CHANNEL_MSENC_ERROR
#define ROBUST_CHANNEL_GPU_ECC_DBE                      (48)
#define SILENT_RUNNING_CONSTANT_LEVEL_SET_BY_REGISTRY   (49)
#define SILENT_RUNNING_LEVEL_TRANSITION_DUE_TO_RC_ERROR (50)
#define SILENT_RUNNING_STRESS_TEST_FAILURE              (51)
#define SILENT_RUNNING_LEVEL_TRANS_DUE_TO_TEMP_RISE     (52)
#define SILENT_RUNNING_TEMP_REDUCED_CLOCKING            (53)
#define SILENT_RUNNING_PWR_REDUCED_CLOCKING             (54)
#define SILENT_RUNNING_TEMPERATURE_READ_ERROR           (55)
#define DISPLAY_CHANNEL_EXCEPTION                       (56)
#define FB_LINK_TRAINING_FAILURE_ERROR                  (57)
#define FB_MEMORY_ERROR                                 (58)
#define PMU_ERROR                                       (59)
#define ROBUST_CHANNEL_SEC2_ERROR                       (60)
#define PMU_BREAKPOINT                                  (61)
#define PMU_HALT_ERROR                                  (62)
#define INFOROM_PAGE_RETIREMENT_EVENT                   (63)
#define INFOROM_PAGE_RETIREMENT_FAILURE                 (64)
#define INFOROM_DRAM_RETIREMENT_EVENT                   (63)
#define INFOROM_DRAM_RETIREMENT_FAILURE                 (64)
#define ROBUST_CHANNEL_LWENC1_ERROR                     (65)
#define ROBUST_CHANNEL_FECS_ERR_REG_ACCESS_VIOLATION    (66)
#define ROBUST_CHANNEL_FECS_ERR_VERIF_VIOLATION         (67)
#define ROBUST_CHANNEL_LWDEC_ERROR                      (68)
#define ROBUST_CHANNEL_LWDEC0_ERROR                     ROBUST_CHANNEL_LWDEC_ERROR
#define ROBUST_CHANNEL_GR_CLASS_ERROR                   (69)
#define ROBUST_CHANNEL_CE3_ERROR                        (70)
#define ROBUST_CHANNEL_CE4_ERROR                        (71)
#define ROBUST_CHANNEL_CE5_ERROR                        (72)
#define ROBUST_CHANNEL_LWENC2_ERROR                     (73)
#define LWLINK_ERROR                                    (74)
#define ROBUST_CHANNEL_CE6_ERROR                        (75)
#define ROBUST_CHANNEL_CE7_ERROR                        (76)
#define ROBUST_CHANNEL_CE8_ERROR                        (77)
#define VGPU_START_ERROR                                (78)

#endif  // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define ROBUST_CHANNEL_GPU_HAS_FALLEN_OFF_THE_BUS       (79)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define PBDMA_PUSHBUFFER_CRC_MISMATCH                   (80)
#define ROBUST_CHANNEL_VGA_SUBSYSTEM_ERROR              (81)
#define ROBUST_CHANNEL_LWJPG0_ERROR                     (82)
#define ROBUST_CHANNEL_LWDEC1_ERROR                     (83)
#define ROBUST_CHANNEL_LWDEC2_ERROR                     (84)
#define ROBUST_CHANNEL_CE9_ERROR                        (85)
#define ROBUST_CHANNEL_OFA0_ERROR                       (86)
#define LWTELEMETRY_DRIVER_REPORT                       (87)
#define ROBUST_CHANNEL_LWDEC3_ERROR                     (88)
#define ROBUST_CHANNEL_LWDEC4_ERROR                     (89)
#define LTC_ERROR                                       (90)
#define RESERVED_XID                                    (91)
#define EXCESSIVE_SBE_INTERRUPTS                        (92)
#define INFOROM_ERASE_LIMIT_EXCEEDED                    (93)
#define ROBUST_CHANNEL_CONTAINED_ERROR                  (94)
#define ROBUST_CHANNEL_UNCONTAINED_ERROR                (95)
#define ROBUST_CHANNEL_LWDEC5_ERROR                     (96)
#define ROBUST_CHANNEL_LWDEC6_ERROR                     (97)
#define ROBUST_CHANNEL_LWDEC7_ERROR                     (98)
#define ROBUST_CHANNEL_LWJPG1_ERROR                     (99)
#define ROBUST_CHANNEL_LWJPG2_ERROR                     (100)
#define ROBUST_CHANNEL_LWJPG3_ERROR                     (101)
#define ROBUST_CHANNEL_LWJPG4_ERROR                     (102)
#define ROBUST_CHANNEL_LWJPG5_ERROR                     (103)
#define ROBUST_CHANNEL_LWJPG6_ERROR                     (104)
#define ROBUST_CHANNEL_LWJPG7_ERROR                     (105)
#define SMBPBI_TEST_MESSAGE                             (106)
#define SMBPBI_TEST_MESSAGE_SILENT                      (107)
#define DESTINATION_FLA_TRANSLATION_ERROR               (108)
#define ROBUST_CHANNEL_CTXSW_TIMEOUT_ERROR              (109)
#define SEC_FAULT_ERROR                                 (110)
#define BUNDLE_ERROR_EVENT                              (111)
#define DISP_SUPERVISOR_ERROR                           (112)
#define DP_LT_FAILURE                                   (113)
#define HEAD_RG_UNDERFLOW                               (114)
#define CORE_CHANNEL_REGS                               (115)
#define WINDOW_CHANNEL_REGS                             (116)
#define LWRSOR_CHANNEL_REGS                             (117)
#define HEAD_REGS                                       (118)
#define GSP_RPC_TIMEOUT                                 (119)
#define GSP_ERROR                                       (120)
#define C2C_ERROR                                       (121)
#define ROBUST_CHANNEL_LAST_ERROR                       (C2C_ERROR)


// Indexed CE reference
#define ROBUST_CHANNEL_CEn_ERROR(x)                                       \
    (x < 3 ? ROBUST_CHANNEL_CE0_ERROR + (x) :                             \
             ((x < 6) ? (ROBUST_CHANNEL_CE3_ERROR + (x - 3)) :            \
                        ((x < 9) ? (ROBUST_CHANNEL_CE6_ERROR + (x - 6)) : \
                                   ROBUST_CHANNEL_CE9_ERROR)))

#define ROBUST_CHANNEL_IS_CE_ERROR(x)                                      \
    ((x == ROBUST_CHANNEL_CE0_ERROR) || (x == ROBUST_CHANNEL_CE1_ERROR) || \
     (x == ROBUST_CHANNEL_CE2_ERROR) || (x == ROBUST_CHANNEL_CE3_ERROR) || \
     (x == ROBUST_CHANNEL_CE4_ERROR) || (x == ROBUST_CHANNEL_CE5_ERROR) || \
     (x == ROBUST_CHANNEL_CE6_ERROR) || (x == ROBUST_CHANNEL_CE7_ERROR) || \
     (x == ROBUST_CHANNEL_CE8_ERROR) || (x == ROBUST_CHANNEL_CE9_ERROR))

#define ROBUST_CHANNEL_CE_ERROR_IDX(x)                                      \
    (((x >= ROBUST_CHANNEL_CE0_ERROR) && (x <= ROBUST_CHANNEL_CE2_ERROR)) ? \
         (x - ROBUST_CHANNEL_CE0_ERROR) :                                   \
         (((x >= ROBUST_CHANNEL_CE3_ERROR) &&                               \
           (x <= ROBUST_CHANNEL_CE5_ERROR)) ?                               \
              (x - ROBUST_CHANNEL_CE3_ERROR) :                              \
              (((x >= ROBUST_CHANNEL_CE6_ERROR) &&                          \
                (x <= ROBUST_CHANNEL_CE8_ERROR)) ?                          \
                   (x - ROBUST_CHANNEL_CE6_ERROR) :                         \
                   (x - ROBUST_CHANNEL_CE9_ERROR))))

// Indexed LWDEC reference
#define ROBUST_CHANNEL_LWDECn_ERROR(x)                                     \
    ((x == 0) ?                                                            \
         (ROBUST_CHANNEL_LWDEC0_ERROR) :                                   \
         (((x >= 1) && (x <= 2)) ? (ROBUST_CHANNEL_LWDEC1_ERROR + x - 1) : \
                                   (ROBUST_CHANNEL_LWDEC3_ERROR + x - 3)))

#define ROBUST_CHANNEL_IS_LWDEC_ERROR(x)   \
    ((x == ROBUST_CHANNEL_LWDEC0_ERROR) || \
     (x == ROBUST_CHANNEL_LWDEC1_ERROR) || \
     (x == ROBUST_CHANNEL_LWDEC2_ERROR) || \
     (x == ROBUST_CHANNEL_LWDEC3_ERROR) || \
     (x == ROBUST_CHANNEL_LWDEC4_ERROR))

#define ROBUST_CHANNEL_LWDEC_ERROR_IDX(x)             \
    (((x == ROBUST_CHANNEL_LWDEC0_ERROR)) ?           \
         (x - ROBUST_CHANNEL_LWDEC0_ERROR) :          \
         (((x >= ROBUST_CHANNEL_LWDEC1_ERROR) &&      \
           (x <= ROBUST_CHANNEL_LWDEC2_ERROR)) ?      \
              (x - ROBUST_CHANNEL_LWDEC1_ERROR + 1) : \
              (x - ROBUST_CHANNEL_LWDEC3_ERROR + 3)))

// Indexed LWENC reference
#define ROBUST_CHANNEL_LWENCn_ERROR(x)                      \
    ((x == 0) ? (ROBUST_CHANNEL_LWENC0_ERROR) :             \
                ((x == 1) ? (ROBUST_CHANNEL_LWENC1_ERROR) : \
                            (ROBUST_CHANNEL_LWENC2_ERROR)))

#define ROBUST_CHANNEL_IS_LWENC_ERROR(x)   \
    ((x == ROBUST_CHANNEL_LWENC0_ERROR) || \
     (x == ROBUST_CHANNEL_LWENC1_ERROR) || \
     (x == ROBUST_CHANNEL_LWENC2_ERROR))

#define ROBUST_CHANNEL_LWENC_ERROR_IDX(x)             \
    (((x == ROBUST_CHANNEL_LWENC0_ERROR)) ?           \
         (x - ROBUST_CHANNEL_LWENC0_ERROR) :          \
         (((x == ROBUST_CHANNEL_LWENC1_ERROR)) ?      \
              (x - ROBUST_CHANNEL_LWENC1_ERROR + 1) : \
              (x - ROBUST_CHANNEL_LWENC2_ERROR + 2)))

// Error Levels
#define ROBUST_CHANNEL_ERROR_RECOVERY_LEVEL_INFO      (0)
#define ROBUST_CHANNEL_ERROR_RECOVERY_LEVEL_NON_FATAL (1)
#define ROBUST_CHANNEL_ERROR_RECOVERY_LEVEL_FATAL     (2)

//
// NOTE: These tables must be kept in sync with the ROBUST_CHANNEL_*,
//       SILENT_RUNNING_*, DISPLAY_* and FB_* #defines.
//
#define ROBUST_CHANNEL_ERROR_STR            \
       {"Unknown Error",                    \
        "Fifo: Fifo Method Error",          \
        "Fifo: SW Method Error",            \
        "Fifo: Unknown Method Error",       \
        "Fifo: Channel Busy Error",         \
        "Fifo: Runout Overflow Error",      \
        "Fifo: Parse Error",                \
        "Fifo: PTE Error",                  \
        "Fifo: Watchdog Timeout Error",     \
        "GR: Instance Error",               \
        "GR: In Single Step Mode Error",    \
        "GR: Missing HW Error",             \
        "GR: SW Method Error",              \
        "GR: Exception Error",              \
        "Fake Error",                       \
        "Scanline Timeout Error",           \
        "Vblank Callback Timeout Error",    \
        "Parameter Error",                  \
        "Bus Master Timeout Error",         \
        "Disp: Missed Notifier",            \
        "MPEG: SW Method Error",            \
        "ME: SW Method Error",              \
        "VP: SW Method Error",              \
        "RC Logging Enabled",               \
        "GR: Semaphore Timeout",            \
        "GR: Illegal Notify",               \
        "Fifo: FBISTATE Timeout Error",     \
        "VP: Unknown Error",                \
        "VP2: Unknown Error",               \
        "BSP: Unknown Error",               \
        "Bad Address Accessed Error",       \
        "Fifo: MMU Error",                  \
        "PBDMA Error",                      \
        "SEC Error",                        \
        "MSVLD Error",                      \
        "MSPDEC Error",                     \
        "MSPPP Error",                      \
        "FECS Err: Unimpl Firmware Method", \
        "FECS Err: Watchdog Timeout",       \
        "CE0: Unknown Error",               \
        "CE1: Unknown Error",               \
        "CE2: Unknown Error",               \
        "VIC: Unknown Error",               \
        "Reset Channel for Verif",          \
        "GR: Fault During Context Switch",  \
        "OS: Preemptive Channel Removal",   \
        "OS: Os indicates GPU has Timed out",                          \
        "MSENC (LWENC0) Error",                                        \
        "DBE (Double Bit Error) ECC Error",                            \
        "Silent running constant level set by registry",               \
        "Silent running level transition due to RC error",             \
        "Silent running stress test failure",                          \
        "Silent running level transition due to temperature rise",     \
        "Silent running clocks reduced due to temperature rise",       \
        "Silent running clocks reduced due to power limits",           \
        "Silent running temperature read error",                       \
        "Display channel exception",                                   \
        "FB link training failure",                                    \
        "FB memory error",                                             \
        "PMU error",                                                   \
        "SEC2 error",                                                  \
        "PMU Breakpoint (non-fatal)",                                  \
        "PMU Halt Error",                                              \
        "INFOROM Page Retirement Event",                               \
        "INFOROM Page Retirement Failure",                             \
        "LWENC1 Error",                                                \
        "FECS Firmware Register Access Violation",                     \
        "FECS Firmware Verif Method Violation",                        \
        "LWDEC Error",                                                 \
        "GR: Class Error",                                             \
        "CE3: Unknown Error",                                          \
        "CE4: Unknown Error",                                          \
        "CE5: Unknown Error",                                          \
        "LWENC2 Error",                                                \
        "LWLink Error",                                                \
        "CE6: Unknown Error",                                          \
        "CE7: Unknown Error",                                          \
        "CE8: Unknown Error",                                          \
        "vGPU Start Error",                                            \
        "GPU has fallen off the bus",                                  \
        "Pushbuffer CRC Mismatch",                                     \
        "VGA Subsystem Error",                                         \
        "LWJPG0 Error",                                                \
        "LWDEC1 Error",                                                \
        "LWDEC2 Error",                                                \
        "CE9: Unknown Error",                                          \
        "OFA0 Error",                                                  \
        "LwTelemetry Driver Reoprt",                                   \
        "LWDEC3 Error",                                                \
        "LWDEC4 Error",                                                \
        "Level 2 Cache Error",                                         \
        "Reserved Xid",                                                \
        "Excessive SBE interrupts",                                    \
        "INFOROM Erase Limit Exceeded",                                \
        "Contained error",                                             \
        "Uncontained error",                                           \
        "LWDEC5 Error",                                                \
        "LWDEC6 Error",                                                \
        "LWDEC7 Error",                                                \
        "LWJPG1 Error",                                                \
        "LWJPG2 Error",                                                \
        "LWJPG3 Error",                                                \
        "LWJPG4 Error",                                                \
        "LWJPG5 Error",                                                \
        "LWJPG6 Error",                                                \
        "LWJPG7 Error",                                                \
        "SMBPBI test message",                                         \
        "SMBPBI test message (silent)",                                \
        "Destination side FLA translation error",                      \
        "Context Switch Timeout Error",                                \
        "Security Fault Error",                                        \
        "Bundle Error Event",                                          \
        "Display Supervisor Error",                                    \
        "DisplayPort LinkTrain Failure",                               \
        "Head Register Underflow Error",                               \
        "Core Channel Register Dump",                                  \
        "Window Channel Register Dump",                                \
        "Cursor Channel Register Dump",                                \
        "Head Register Dump",                                          \
        "GSP RPC Timeout",                                             \
        "GSP Error",                                                   \
        "C2C Error"}

#endif  // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define ROBUST_CHANNEL_ERROR_STR_PUBLIC_PUBLISHED  \
       {"Unknown Error",                         \
        "DMA Engine Error (FIFO Error 1)",       \
        "DMA Engine Error (FIFO Error 2)",       \
        "DMA Engine Error (FIFO Error 3)",       \
        "DMA Engine Error (FIFO Error 4)",       \
        "DMA Engine Error (FIFO Error 5)",       \
        "DMA Engine Error (FIFO Error 6)",       \
        "DMA Engine Error (FIFO Error 7)",       \
        "DMA Engine Error (FIFO Error 8)",       \
        "Graphics Engine Error (GR Error 1)",    \
        "Graphics Engine Error (GR Error 2)",    \
        "Graphics Engine Error (GR Error 3)",    \
        "Graphics Engine Error (GR Error 4)",    \
        "Graphics Engine Error (GR Exception Error)",\
        "Fake Error",                            \
        "Display Engine Error (CRTC Error 1)",   \
        "Display Engine Error (CRTC Error 2)",   \
        "Display Engine Error (CRTC Error 3)",   \
        "Bus Interface Error (BIF Error)",       \
        "Client Reported Error",                 \
        "Video Engine Error (MPEG Error)",       \
        "Video Engine Error (ME Error)",         \
        "Video Engine Error (VP Error 1)",       \
        "Error Reporting Enabled",               \
        "Graphics Engine Error (GR Error 6)",    \
        "Graphics Engine Error (GR Error 7)",    \
        "DMA Engine Error (FIFO Error 9)",       \
        "Video Engine Error (VP Error 2)",       \
        "Video Engine Error (VP2 Error)",        \
        "Video Engine Error (BSP Error)",        \
        "Access Violation Error (MMU Error 1)",  \
        "Access Violation Error (MMU Error 2)",  \
        "DMA Engine Error (PBDMA Error)",        \
        "Security Engine Error (SEC Error)",     \
        "Video Engine Error (MSVLD Error)",      \
        "Video Engine Error (MSPDEC Error)",     \
        "Video Engine Error (MSPPP Error)",      \
        "Graphics Engine Error (FECS Error 1)",  \
        "Graphics Engine Error (FECS Error 2)",  \
        "DMA Engine Error (CE Error 1)",         \
        "DMA Engine Error (CE Error 2)",         \
        "DMA Engine Error (CE Error 3)",         \
        "Video Engine Error (VIC Error)",        \
        "Verification Error",                    \
        "Access Violation Error (MMU Error 3)",  \
        "Operating System Error (OS Error 1)",   \
        "Operating System Error (OS Error 2)",   \
        "Video Engine Error (MSENC/LWENC0 Error)",\
        "ECC Error (DBE Error)",                 \
        "Power State Locked",                    \
        "Power State Event (RC Error)",          \
        "Power State Event (Stress Test Error)", \
        "Power State Event (Thermal Event 1)",   \
        "Power State Event (Thermal Event 2)",   \
        "Power State Event (Power Event)",       \
        "Power State Event (Thermal Event 3)",   \
        "Display Engine Error (EVO Error)",      \
        "FB Interface Error (FBPA Error 1)",     \
        "FB Interface Error (FBPA Error 2)",     \
        "PMU error",                             \
        "SEC2 error",                            \
        "PMU Breakpoint (non-fatal)",            \
        "PMU Halt Error",                        \
        "INFOROM Page Retirement Event",         \
        "INFOROM Page Retirement Failure",       \
        "Video Engine Error (LWENC1 Error)",     \
        "Graphics Engine Error (FECS Error 3)",  \
        "Graphics Engine Error (FECS Error 4)",  \
        "Video Engine Error (LWDEC0 Error)",     \
        "Graphics Engine Error (GR Class Error)",\
        "DMA Engine Error (CE Error 4)",         \
        "DMA Engine Error (CE Error 5)",         \
        "DMA Engine Error (CE Error 6)",         \
        "Video Engine Error (LWENC2 Error)",     \
        "LWLink Error",                          \
        "DMA Engine Error (CE Error 6)",         \
        "DMA Engine Error (CE Error 7)",         \
        "DMA Engine Error (CE Error 8)",         \
        "vGPU device cannot be started",         \
        "GPU has fallen off the bus",            \
        "DMA Engine Error (Pushbuffer CRC mismatch)",\
        "VGA Subsystem Error",                   \
        "Video JPEG Engine Error (LWJPG Error)", \
        "Video Engine Error (LWDEC1 Error)",     \
        "Video Engine Error (LWDEC2 Error)",     \
        "DMA Engine Error (CE Error 9)",         \
        "Video OFA Engine Error (OFA0 Error)",   \
        "LwTelemetry Driver Reoprt",             \
        "Video Engine Error (LWDEC3 Error)",     \
        "Video Engine Error (LWDEC4 Error)",     \
        "FB Interface Error (FBPA Error 3)",     \
        "Reserved Xid",                          \
        "Excessive SBE interrupts"

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
//
// ROBUST_CHANNEL_ERROR_STR_PUBLIC_PUBLISHED contains strings that we have already published
//   at https://docs.lwpu.com/deploy/xid-errors/index.html
// ROBUST_CHANNEL_ERROR_STR_PUBLIC_NOT_PUBLISHED contains strings that we will publish once
//   the embargo on future chips is lifted OR docs.lwpu.com contains updated XID errors.
//   Move strings from ROBUST_CHANNEL_ERROR_STR_PUBLIC_NOT_PUBLISHED to
//                     ROBUST_CHANNEL_ERROR_STR_PUBLIC_PUBLISHED as embargo is lifted.
//   All new strings should be added to ROBUST_CHANNEL_ERROR_STR_PUBLIC_NOT_PUBLISHED.
//

#define ROBUST_CHANNEL_ERROR_STR_PUBLIC_NOT_PUBLISHED  \
        "INFOROM Erase Limit Exceeded",          \
        "Contained error",                       \
        "Uncontained error",                     \
        "Video Engine Error (LWDEC5 Error)",     \
        "Video Engine Error (LWDEC6 Error)",     \
        "Video Engine Error (LWDEC7 Error)",     \
        "Video JPEG Engine Error (LWJPG1 Error)", \
        "Video JPEG Engine Error (LWJPG2 Error)", \
        "Video JPEG Engine Error (LWJPG3 Error)", \
        "Video JPEG Engine Error (LWJPG4 Error)", \
        "Video JPEG Engine Error (LWJPG5 Error)", \
        "Video JPEG Engine Error (LWJPG6 Error)", \
        "Video JPEG Engine Error (LWJPG7 Error)", \
        "SMBPBI test message",                    \
        "SMBPBI test message (silent)",           \
        "Destination side FLA Translation Error", \
        "Context Switch Error",                   \
        "Security Fault Error",                   \
        "Bundle Error Event",                     \
        "Display Supervisor Error",               \
        "DisplayPort LinkTrain Failure",          \
        "Head Register Underflow Error",          \
        "Core Channel Register Dump",             \
        "Window Channel Register Dump",           \
        "Cursor Channel Register Dump",           \
        "Head Register Dump",                     \
        "GSP RPC Timeout",                        \
        "GSP Error",                              \
        "C2C Error"}

#define ROBUST_CHANNEL_ERROR_STR_PUBLIC             \
        ROBUST_CHANNEL_ERROR_STR_PUBLIC_PUBLISHED,  \
        ROBUST_CHANNEL_ERROR_STR_PUBLIC_NOT_PUBLISHED

#else

#define ROBUST_CHANNEL_ERROR_STR_PUBLIC             \
        ROBUST_CHANNEL_ERROR_STR_PUBLIC_PUBLISHED}

#endif  // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#endif  // LWERROR_H
