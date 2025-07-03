/*
 * SPDX-FileCopyrightText: Copyright (c) 2002-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _cl2080_h_
#define _cl2080_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"
#include "lwlimits.h"

#define  LW20_SUBDEVICE_0                                          (0x00002080)

/* event values */
#define LW2080_NOTIFIERS_SW                                        (0)
#define LW2080_NOTIFIERS_HOTPLUG                                   (1)
#define LW2080_NOTIFIERS_POWER_CONNECTOR                           (2)
#define LW2080_NOTIFIERS_THERMAL_SW                                (3)
#define LW2080_NOTIFIERS_THERMAL_HW                                (4)
#define LW2080_NOTIFIERS_FULL_SCREEN_CHANGE                        (5)
#define LW2080_NOTIFIERS_EVENTBUFFER                               (6)
#define LW2080_NOTIFIERS_DP_IRQ                                    (7)
#define LW2080_NOTIFIERS_GR_DEBUG_INTR                             (8)
#define LW2080_NOTIFIERS_PMU_EVENT                                 (9)
#define LW2080_NOTIFIERS_PMU_COMMAND                               (10)
#define LW2080_NOTIFIERS_TIMER                                     (11)
#define LW2080_NOTIFIERS_GRAPHICS                                  (12)
#define LW2080_NOTIFIERS_PPP                                       (13)
#define LW2080_NOTIFIERS_VLD                                       (14) // also known as BSP
#define LW2080_NOTIFIERS_LWDEC0                                    LW2080_NOTIFIERS_VLD
#define LW2080_NOTIFIERS_LWDEC1                                    (15)
#define LW2080_NOTIFIERS_LWDEC2                                    (16)
#define LW2080_NOTIFIERS_LWDEC3                                    (17)
#define LW2080_NOTIFIERS_LWDEC4                                    (18)
#define LW2080_NOTIFIERS_LWDEC5                                    (19)
#define LW2080_NOTIFIERS_LWDEC6                                    (20)
#define LW2080_NOTIFIERS_LWDEC7                                    (21)
#define LW2080_NOTIFIERS_PDEC                                      (22) // also known as VP
#define LW2080_NOTIFIERS_CE0                                       (23)
#define LW2080_NOTIFIERS_CE1                                       (24)
#define LW2080_NOTIFIERS_CE2                                       (25)
#define LW2080_NOTIFIERS_CE3                                       (26)
#define LW2080_NOTIFIERS_CE4                                       (27)
#define LW2080_NOTIFIERS_CE5                                       (28)
#define LW2080_NOTIFIERS_CE6                                       (29)
#define LW2080_NOTIFIERS_CE7                                       (30)
#define LW2080_NOTIFIERS_CE8                                       (31)
#define LW2080_NOTIFIERS_CE9                                       (32)
#define LW2080_NOTIFIERS_PSTATE_CHANGE                             (33)
#define LW2080_NOTIFIERS_HDCP_STATUS_CHANGE                        (34)
#define LW2080_NOTIFIERS_FIFO_EVENT_MTHD                           (35)
#define LW2080_NOTIFIERS_PRIV_RING_HANG                            (36)
#define LW2080_NOTIFIERS_RC_ERROR                                  (37)
#define LW2080_NOTIFIERS_MSENC                                     (38)
#define LW2080_NOTIFIERS_LWENC0                                    LW2080_NOTIFIERS_MSENC
#define LW2080_NOTIFIERS_LWENC1                                    (39)
#define LW2080_NOTIFIERS_LWENC2                                    (40)
#define LW2080_NOTIFIERS_UNUSED_0                                  (41) // Unused
#define LW2080_NOTIFIERS_ACPI_NOTIFY                               (42)
#define LW2080_NOTIFIERS_COOLER_DIAG_ZONE                          (43)
#define LW2080_NOTIFIERS_THERMAL_DIAG_ZONE                         (44)
#define LW2080_NOTIFIERS_AUDIO_HDCP_REQUEST                        (45)
#define LW2080_NOTIFIERS_WORKLOAD_MODULATION_CHANGE                (46)
#define LW2080_NOTIFIERS_GPIO_0_RISING_INTERRUPT                   (47)
#define LW2080_NOTIFIERS_GPIO_1_RISING_INTERRUPT                   (48)
#define LW2080_NOTIFIERS_GPIO_2_RISING_INTERRUPT                   (49)
#define LW2080_NOTIFIERS_GPIO_3_RISING_INTERRUPT                   (50)
#define LW2080_NOTIFIERS_GPIO_4_RISING_INTERRUPT                   (51)
#define LW2080_NOTIFIERS_GPIO_5_RISING_INTERRUPT                   (52)
#define LW2080_NOTIFIERS_GPIO_6_RISING_INTERRUPT                   (53)
#define LW2080_NOTIFIERS_GPIO_7_RISING_INTERRUPT                   (54)
#define LW2080_NOTIFIERS_GPIO_8_RISING_INTERRUPT                   (55)
#define LW2080_NOTIFIERS_GPIO_9_RISING_INTERRUPT                   (56)
#define LW2080_NOTIFIERS_GPIO_10_RISING_INTERRUPT                  (57)
#define LW2080_NOTIFIERS_GPIO_11_RISING_INTERRUPT                  (58)
#define LW2080_NOTIFIERS_GPIO_12_RISING_INTERRUPT                  (59)
#define LW2080_NOTIFIERS_GPIO_13_RISING_INTERRUPT                  (60)
#define LW2080_NOTIFIERS_GPIO_14_RISING_INTERRUPT                  (61)
#define LW2080_NOTIFIERS_GPIO_15_RISING_INTERRUPT                  (62)
#define LW2080_NOTIFIERS_GPIO_16_RISING_INTERRUPT                  (63)
#define LW2080_NOTIFIERS_GPIO_17_RISING_INTERRUPT                  (64)
#define LW2080_NOTIFIERS_GPIO_18_RISING_INTERRUPT                  (65)
#define LW2080_NOTIFIERS_GPIO_19_RISING_INTERRUPT                  (66)
#define LW2080_NOTIFIERS_GPIO_20_RISING_INTERRUPT                  (67)
#define LW2080_NOTIFIERS_GPIO_21_RISING_INTERRUPT                  (68)
#define LW2080_NOTIFIERS_GPIO_22_RISING_INTERRUPT                  (69)
#define LW2080_NOTIFIERS_GPIO_23_RISING_INTERRUPT                  (70)
#define LW2080_NOTIFIERS_GPIO_24_RISING_INTERRUPT                  (71)
#define LW2080_NOTIFIERS_GPIO_25_RISING_INTERRUPT                  (72)
#define LW2080_NOTIFIERS_GPIO_26_RISING_INTERRUPT                  (73)
#define LW2080_NOTIFIERS_GPIO_27_RISING_INTERRUPT                  (74)
#define LW2080_NOTIFIERS_GPIO_28_RISING_INTERRUPT                  (75)
#define LW2080_NOTIFIERS_GPIO_29_RISING_INTERRUPT                  (76)
#define LW2080_NOTIFIERS_GPIO_30_RISING_INTERRUPT                  (77)
#define LW2080_NOTIFIERS_GPIO_31_RISING_INTERRUPT                  (78)
#define LW2080_NOTIFIERS_GPIO_0_FALLING_INTERRUPT                  (79)
#define LW2080_NOTIFIERS_GPIO_1_FALLING_INTERRUPT                  (80)
#define LW2080_NOTIFIERS_GPIO_2_FALLING_INTERRUPT                  (81)
#define LW2080_NOTIFIERS_GPIO_3_FALLING_INTERRUPT                  (82)
#define LW2080_NOTIFIERS_GPIO_4_FALLING_INTERRUPT                  (83)
#define LW2080_NOTIFIERS_GPIO_5_FALLING_INTERRUPT                  (84)
#define LW2080_NOTIFIERS_GPIO_6_FALLING_INTERRUPT                  (85)
#define LW2080_NOTIFIERS_GPIO_7_FALLING_INTERRUPT                  (86)
#define LW2080_NOTIFIERS_GPIO_8_FALLING_INTERRUPT                  (87)
#define LW2080_NOTIFIERS_GPIO_9_FALLING_INTERRUPT                  (88)
#define LW2080_NOTIFIERS_GPIO_10_FALLING_INTERRUPT                 (89)
#define LW2080_NOTIFIERS_GPIO_11_FALLING_INTERRUPT                 (90)
#define LW2080_NOTIFIERS_GPIO_12_FALLING_INTERRUPT                 (91)
#define LW2080_NOTIFIERS_GPIO_13_FALLING_INTERRUPT                 (92)
#define LW2080_NOTIFIERS_GPIO_14_FALLING_INTERRUPT                 (93)
#define LW2080_NOTIFIERS_GPIO_15_FALLING_INTERRUPT                 (94)
#define LW2080_NOTIFIERS_GPIO_16_FALLING_INTERRUPT                 (95)
#define LW2080_NOTIFIERS_GPIO_17_FALLING_INTERRUPT                 (96)
#define LW2080_NOTIFIERS_GPIO_18_FALLING_INTERRUPT                 (97)
#define LW2080_NOTIFIERS_GPIO_19_FALLING_INTERRUPT                 (98)
#define LW2080_NOTIFIERS_GPIO_20_FALLING_INTERRUPT                 (99)
#define LW2080_NOTIFIERS_GPIO_21_FALLING_INTERRUPT                 (100)
#define LW2080_NOTIFIERS_GPIO_22_FALLING_INTERRUPT                 (101)
#define LW2080_NOTIFIERS_GPIO_23_FALLING_INTERRUPT                 (102)
#define LW2080_NOTIFIERS_GPIO_24_FALLING_INTERRUPT                 (103)
#define LW2080_NOTIFIERS_GPIO_25_FALLING_INTERRUPT                 (104)
#define LW2080_NOTIFIERS_GPIO_26_FALLING_INTERRUPT                 (105)
#define LW2080_NOTIFIERS_GPIO_27_FALLING_INTERRUPT                 (106)
#define LW2080_NOTIFIERS_GPIO_28_FALLING_INTERRUPT                 (107)
#define LW2080_NOTIFIERS_GPIO_29_FALLING_INTERRUPT                 (108)
#define LW2080_NOTIFIERS_GPIO_30_FALLING_INTERRUPT                 (109)
#define LW2080_NOTIFIERS_GPIO_31_FALLING_INTERRUPT                 (110)
#define LW2080_NOTIFIERS_ECC_SBE                                   (111)
#define LW2080_NOTIFIERS_ECC_DBE                                   (112)
#define LW2080_NOTIFIERS_STEREO_EMITTER_DETECTION                  (113)
#define LW2080_NOTIFIERS_GC5_GPU_READY                             (114)
#define LW2080_NOTIFIERS_SEC2                                      (115)
#define LW2080_NOTIFIERS_GC6_REFCOUNT_INC                          (116)
#define LW2080_NOTIFIERS_GC6_REFCOUNT_DEC                          (117)
#define LW2080_NOTIFIERS_POWER_EVENT                               (118)
#define LW2080_NOTIFIERS_CLOCKS_CHANGE                             (119)
#define LW2080_NOTIFIERS_HOTPLUG_PROCESSING_COMPLETE               (120)
#define LW2080_NOTIFIERS_PHYSICAL_PAGE_FAULT                       (121)
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
#define LW2080_NOTIFIERS_LICENSE_CHANGED                           (122)
#else
#define LW2080_NOTIFIERS_RESERVED_122                              (122)
#endif
#define LW2080_NOTIFIERS_LWLINK_ERROR_FATAL                        (123)
#define LW2080_NOTIFIERS_PRIV_REG_ACCESS_FAULT                     (124)
#define LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED            (125)
#define LW2080_NOTIFIERS_LWJPG                                     (126)
#define LW2080_NOTIFIERS_LWJPEG0                                   LW2080_NOTIFIERS_LWJPG
#define LW2080_NOTIFIERS_LWJPEG1                                   (127)
#define LW2080_NOTIFIERS_LWJPEG2                                   (128)
#define LW2080_NOTIFIERS_LWJPEG3                                   (129)
#define LW2080_NOTIFIERS_LWJPEG4                                   (130)
#define LW2080_NOTIFIERS_LWJPEG5                                   (131)
#define LW2080_NOTIFIERS_LWJPEG6                                   (132)
#define LW2080_NOTIFIERS_LWJPEG7                                   (133)
#define LW2080_NOTIFIERS_RUNLIST_AND_ENG_IDLE                      (134)
#define LW2080_NOTIFIERS_RUNLIST_ACQUIRE                           (135)
#define LW2080_NOTIFIERS_RUNLIST_ACQUIRE_AND_ENG_IDLE              (136)
#define LW2080_NOTIFIERS_RUNLIST_IDLE                              (137)
#define LW2080_NOTIFIERS_TSG_PREEMPT_COMPLETE                      (138)
#define LW2080_NOTIFIERS_RUNLIST_PREEMPT_COMPLETE                  (139)
#define LW2080_NOTIFIERS_CTXSW_TIMEOUT                             (140)
#define LW2080_NOTIFIERS_INFOROM_ECC_OBJECT_UPDATED                (141)
#define LW2080_NOTIFIERS_LWTELEMETRY_REPORT_EVENT                  (142)
#define LW2080_NOTIFIERS_DSTATE_XUSB_PPC                           (143)
#define LW2080_NOTIFIERS_FECS_CTX_SWITCH                           (144)
#define LW2080_NOTIFIERS_XUSB_PPC_CONNECTED                        (145)
#define LW2080_NOTIFIERS_GR0                                       LW2080_NOTIFIERS_GRAPHICS
#define LW2080_NOTIFIERS_GR1                                       (146)
#define LW2080_NOTIFIERS_GR2                                       (147)
#define LW2080_NOTIFIERS_GR3                                       (148)
#define LW2080_NOTIFIERS_GR4                                       (149)
#define LW2080_NOTIFIERS_GR5                                       (150)
#define LW2080_NOTIFIERS_GR6                                       (151)
#define LW2080_NOTIFIERS_GR7                                       (152)
#define LW2080_NOTIFIERS_OFA                                       (153)
#define LW2080_NOTIFIERS_DSTATE_HDA                                (154)
#define LW2080_NOTIFIERS_POISON_ERROR_NON_FATAL                    (155)
#define LW2080_NOTIFIERS_POISON_ERROR_FATAL                        (156)
#define LW2080_NOTIFIERS_UCODE_RESET                               (157)
#define LW2080_NOTIFIERS_PLATFORM_POWER_MODE_CHANGE                (158)
#define LW2080_NOTIFIERS_SMC_CONFIG_UPDATE                         (159)
#define LW2080_NOTIFIERS_INFOROM_RRL_OBJECT_UPDATED                (160)
#define LW2080_NOTIFIERS_INFOROM_PBL_OBJECT_UPDATED                (161)
#define LW2080_NOTIFIERS_LPWR_DIFR_PREFETCH_REQUEST                (162)
#define LW2080_NOTIFIERS_SEC_FAULT_ERROR                           (163)
#define LW2080_NOTIFIERS_POSSIBLE_ERROR                            (164)
#define LW2080_NOTIFIERS_MAXCOUNT                                  (165)

// Indexed GR notifier reference
#define LW2080_NOTIFIERS_GR(x)                                     ((x == 0) ? (LW2080_NOTIFIERS_GR0) : (LW2080_NOTIFIERS_GR1 + (x-1)))
#define LW2080_NOTIFIER_TYPE_IS_GR(x)  (((x) == LW2080_NOTIFIERS_GR0) || (((x) >= LW2080_NOTIFIERS_GR1) && ((x) <= LW2080_NOTIFIERS_GR7)))
// Indexed CE notifier reference
#define LW2080_NOTIFIERS_CE(x)                                     (LW2080_NOTIFIERS_CE0 + (x))
#define LW2080_NOTIFIER_TYPE_IS_CE(x)  (((x) >= LW2080_NOTIFIERS_CE0) && ((x) <= LW2080_NOTIFIERS_CE9))
// Indexed MSENC notifier reference
#define LW2080_NOTIFIERS_LWENC(x)                                  (LW2080_NOTIFIERS_LWENC0 + (x))
#define LW2080_NOTIFIER_TYPE_IS_LWENC(x)  (((x) >= LW2080_NOTIFIERS_LWENC0) && ((x) <= LW2080_NOTIFIERS_LWENC2))
// Indexed LWDEC notifier reference
#define LW2080_NOTIFIERS_LWDEC(x)                                  (LW2080_NOTIFIERS_LWDEC0 + (x))
#define LW2080_NOTIFIER_TYPE_IS_LWDEC(x)  (((x) >= LW2080_NOTIFIERS_LWDEC0) && ((x) <= LW2080_NOTIFIERS_LWDEC7))
// Indexed LWJPEG notifier reference
#define LW2080_NOTIFIERS_LWJPEG(x)                                 (LW2080_NOTIFIERS_LWJPEG0 + (x))
#define LW2080_NOTIFIER_TYPE_IS_LWJPEG(x)  (((x) >= LW2080_NOTIFIERS_LWJPEG0) && ((x) <= LW2080_NOTIFIERS_LWJPEG7))

#define LW2080_NOTIFIERS_GPIO_RISING_INTERRUPT(pin)                (LW2080_NOTIFIERS_GPIO_0_RISING_INTERRUPT+(pin))
#define LW2080_NOTIFIERS_GPIO_FALLING_INTERRUPT(pin)               (LW2080_NOTIFIERS_GPIO_0_FALLING_INTERRUPT+(pin))

#define LW2080_SUBDEVICE_NOTIFICATION_STATUS_IN_PROGRESS              (0x8000)
#define LW2080_SUBDEVICE_NOTIFICATION_STATUS_BAD_ARGUMENT             (0x4000)
#define LW2080_SUBDEVICE_NOTIFICATION_STATUS_ERROR_ILWALID_STATE      (0x2000)
#define LW2080_SUBDEVICE_NOTIFICATION_STATUS_ERROR_STATE_IN_USE       (0x1000)
#define LW2080_SUBDEVICE_NOTIFICATION_STATUS_DONE_SUCCESS             (0x0000)

/* exported engine defines */
#define LW2080_ENGINE_TYPE_NULL                       (0x00000000)
#define LW2080_ENGINE_TYPE_GRAPHICS                   (0x00000001)
#define LW2080_ENGINE_TYPE_GR0                        LW2080_ENGINE_TYPE_GRAPHICS
#define LW2080_ENGINE_TYPE_GR1                        (0x00000002)
#define LW2080_ENGINE_TYPE_GR2                        (0x00000003)
#define LW2080_ENGINE_TYPE_GR3                        (0x00000004)
#define LW2080_ENGINE_TYPE_GR4                        (0x00000005)
#define LW2080_ENGINE_TYPE_GR5                        (0x00000006)
#define LW2080_ENGINE_TYPE_GR6                        (0x00000007)
#define LW2080_ENGINE_TYPE_GR7                        (0x00000008)
#define LW2080_ENGINE_TYPE_COPY0                      (0x00000009)
#define LW2080_ENGINE_TYPE_COPY1                      (0x0000000a)
#define LW2080_ENGINE_TYPE_COPY2                      (0x0000000b)
#define LW2080_ENGINE_TYPE_COPY3                      (0x0000000c)
#define LW2080_ENGINE_TYPE_COPY4                      (0x0000000d)
#define LW2080_ENGINE_TYPE_COPY5                      (0x0000000e)
#define LW2080_ENGINE_TYPE_COPY6                      (0x0000000f)
#define LW2080_ENGINE_TYPE_COPY7                      (0x00000010)
#define LW2080_ENGINE_TYPE_COPY8                      (0x00000011)
#define LW2080_ENGINE_TYPE_COPY9                      (0x00000012)
#define LW2080_ENGINE_TYPE_BSP                        (0x00000013)
#define LW2080_ENGINE_TYPE_LWDEC0                     LW2080_ENGINE_TYPE_BSP
#define LW2080_ENGINE_TYPE_LWDEC1                     (0x00000014)
#define LW2080_ENGINE_TYPE_LWDEC2                     (0x00000015)
#define LW2080_ENGINE_TYPE_LWDEC3                     (0x00000016)
#define LW2080_ENGINE_TYPE_LWDEC4                     (0x00000017)
#define LW2080_ENGINE_TYPE_LWDEC5                     (0x00000018)
#define LW2080_ENGINE_TYPE_LWDEC6                     (0x00000019)
#define LW2080_ENGINE_TYPE_LWDEC7                     (0x0000001a)
#define LW2080_ENGINE_TYPE_MSENC                      (0x0000001b)
#define LW2080_ENGINE_TYPE_LWENC0                      LW2080_ENGINE_TYPE_MSENC  /* Mutually exclusive alias */
#define LW2080_ENGINE_TYPE_LWENC1                     (0x0000001c)
#define LW2080_ENGINE_TYPE_LWENC2                     (0x0000001d)
#define LW2080_ENGINE_TYPE_VP                         (0x0000001e)
#define LW2080_ENGINE_TYPE_ME                         (0x0000001f)
#define LW2080_ENGINE_TYPE_PPP                        (0x00000020)
#define LW2080_ENGINE_TYPE_MPEG                       (0x00000021)
#define LW2080_ENGINE_TYPE_SW                         (0x00000022)
#define LW2080_ENGINE_TYPE_CIPHER                     (0x00000023)
#define LW2080_ENGINE_TYPE_TSEC                       LW2080_ENGINE_TYPE_CIPHER
#define LW2080_ENGINE_TYPE_VIC                        (0x00000024)
#define LW2080_ENGINE_TYPE_MP                         (0x00000025)
#define LW2080_ENGINE_TYPE_SEC2                       (0x00000026)
#define LW2080_ENGINE_TYPE_HOST                       (0x00000027)
#define LW2080_ENGINE_TYPE_DPU                        (0x00000028)
#define LW2080_ENGINE_TYPE_PMU                        (0x00000029)
#define LW2080_ENGINE_TYPE_FBFLCN                     (0x0000002a)
#define LW2080_ENGINE_TYPE_LWJPG                      (0x0000002b)
#define LW2080_ENGINE_TYPE_LWJPEG0                     LW2080_ENGINE_TYPE_LWJPG
#define LW2080_ENGINE_TYPE_LWJPEG1                    (0x0000002c)
#define LW2080_ENGINE_TYPE_LWJPEG2                    (0x0000002d)
#define LW2080_ENGINE_TYPE_LWJPEG3                    (0x0000002e)
#define LW2080_ENGINE_TYPE_LWJPEG4                    (0x0000002f)
#define LW2080_ENGINE_TYPE_LWJPEG5                    (0x00000030)
#define LW2080_ENGINE_TYPE_LWJPEG6                    (0x00000031)
#define LW2080_ENGINE_TYPE_LWJPEG7                    (0x00000032)
#define LW2080_ENGINE_TYPE_OFA                        (0x00000033)
#define LW2080_ENGINE_TYPE_LAST                       (0x00000034)
#define LW2080_ENGINE_TYPE_ALLENGINES                 (0xffffffff)

// Indexed copy engines
#define LW2080_ENGINE_TYPE_COPY(i)     (LW2080_ENGINE_TYPE_COPY0+(i))
#define LW2080_ENGINE_TYPE_IS_COPY(i)  (((i) >= LW2080_ENGINE_TYPE_COPY0) && ((i) <= LW2080_ENGINE_TYPE_COPY9))
#define LW2080_ENGINE_TYPE_COPY_IDX(i) ((i) - LW2080_ENGINE_TYPE_COPY0)
#define LW2080_ENGINE_TYPE_LWENC(i)    (LW2080_ENGINE_TYPE_LWENC0+(i))
#define LW2080_ENGINE_TYPE_IS_LWENC(i)  (((i) >= LW2080_ENGINE_TYPE_LWENC0) && ((i) <= LW2080_ENGINE_TYPE_LWENC2))
#define LW2080_ENGINE_TYPE_LWENC_IDX(i) ((i) - LW2080_ENGINE_TYPE_LWENC0)
#define LW2080_ENGINE_TYPE_LWDEC(i)    (LW2080_ENGINE_TYPE_LWDEC0+(i))
#define LW2080_ENGINE_TYPE_IS_LWDEC(i)  (((i) >= LW2080_ENGINE_TYPE_LWDEC0) && ((i) <= LW2080_ENGINE_TYPE_LWDEC7))
#define LW2080_ENGINE_TYPE_LWDEC_IDX(i) ((i) - LW2080_ENGINE_TYPE_LWDEC0)
#define LW2080_ENGINE_TYPE_LWJPEG(i)    (LW2080_ENGINE_TYPE_LWJPEG0+(i))
#define LW2080_ENGINE_TYPE_IS_LWJPEG(i)  (((i) >= LW2080_ENGINE_TYPE_LWJPEG0) && ((i) <= LW2080_ENGINE_TYPE_LWJPEG7))
#define LW2080_ENGINE_TYPE_LWJPEG_IDX(i) ((i) - LW2080_ENGINE_TYPE_LWJPEG0)
#define LW2080_ENGINE_TYPE_GR(i)       (LW2080_ENGINE_TYPE_GR0 + (i))
#define LW2080_ENGINE_TYPE_IS_GR(i)    (((i) >= LW2080_ENGINE_TYPE_GR0) && ((i) <= LW2080_ENGINE_TYPE_GR7))
#define LW2080_ENGINE_TYPE_GR_IDX(i)   ((i) - LW2080_ENGINE_TYPE_GR0)
#define LW2080_ENGINE_TYPE_COPY_SIZE 10
#define LW2080_ENGINE_TYPE_LWENC_SIZE 3
#define LW2080_ENGINE_TYPE_LWDEC_SIZE 8
#define LW2080_ENGINE_TYPE_LWJPEG_SIZE 8
#define LW2080_ENGINE_TYPE_GR_SIZE 8
#define LW2080_ENGINE_TYPE_IS_VALID(i) (((i) > (LW2080_ENGINE_TYPE_NULL)) && ((i) < (LW2080_ENGINE_TYPE_LAST)))

/* exported client defines */
#define LW2080_CLIENT_TYPE_TEX                        (0x00000001)
#define LW2080_CLIENT_TYPE_COLOR                      (0x00000002)
#define LW2080_CLIENT_TYPE_DEPTH                      (0x00000003)
#define LW2080_CLIENT_TYPE_DA                         (0x00000004)
#define LW2080_CLIENT_TYPE_FE                         (0x00000005)
#define LW2080_CLIENT_TYPE_SCC                        (0x00000006)
#define LW2080_CLIENT_TYPE_WID                        (0x00000007)
#define LW2080_CLIENT_TYPE_MSVLD                      (0x00000008)
#define LW2080_CLIENT_TYPE_MSPDEC                     (0x00000009)
#define LW2080_CLIENT_TYPE_MSPPP                      (0x0000000a)
#define LW2080_CLIENT_TYPE_VIC                        (0x0000000b)
#define LW2080_CLIENT_TYPE_ALLCLIENTS                 (0xffffffff)

/* GC5 Gpu Ready event defines */
#define LW2080_GC5_EXIT_COMPLETE                      (0x00000001)
#define LW2080_GC5_ENTRY_ABORTED                      (0x00000002)

/* Platform Power Mode event defines */
#define LW2080_PLATFORM_POWER_MODE_CHANGE_COMPLETION        (0x00000000)
#define LW2080_PLATFORM_POWER_MODE_CHANGE_ACPI_NOTIFICATION (0x00000001)

/* LwNotification[] fields and values */
#define LW2080_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT          (0x4000)
/* pio method data structure */
typedef volatile struct _cl2080_tag0 {
    LwV32 Reserved00[0x7c0];
} Lw2080Typedef, Lw20Subdevice0;
#define  LW2080_TYPEDEF                                          Lw20Subdevice0

/* LwAlloc parameteters */
#define LW2080_MAX_SUBDEVICES                                 LW_MAX_SUBDEVICES
typedef struct {
    LwU32   subDeviceId;
} LW2080_ALLOC_PARAMETERS;

/* HDCP Status change notification information */
typedef struct Lw2080HdcpStatusChangeNotificationRec {
    LwU32 displayId;
    LwU32 hdcpStatusChangeNotif;
} Lw2080HdcpStatusChangeNotification;

/* Pstate change notification information */
typedef struct Lw2080PStateChangeNotificationRec {
    struct {
        LwU32 nanoseconds[2];  /* nanoseconds since Jan. 1, 1970       0-   7*/
    } timeStamp;               /*                                       -0007*/
    LwU32 NewPstate;
} Lw2080PStateChangeNotification;

/* Clocks change notification information */
typedef struct Lw2080ClocksChangeNotificationRec {
    struct {
        LwU32 nanoseconds[2];  /* nanoseconds since Jan. 1, 1970       0-   7*/
    } timeStamp;               /*                                       -0007*/
} Lw2080ClocksChangeNotification;

/* WorkLoad Modulation state change notification information*/
typedef struct Lw2080WorkloadModulationChangeNotificationRec {
    struct {
        LwU32 nanoseconds[2];  /* nanoseconds since Jan. 1, 1970       0-   7*/
    } timeStamp;               /*                                       -0007*/
    LwBool WorkloadModulationEnabled;
} Lw2080WorkloadModulationChangeNotification;

/* Hotplug notification information */
typedef struct {
    LwU32 plugDisplayMask;
    LwU32 unplugDisplayMask;
} Lw2080HotplugNotification;

/* Power state changing notification information */
typedef struct {
    LwBool bSwitchToAC;
    LwBool bGPUCapabilityChanged;
    LwU32  displayMaskAffected;
} Lw2080PowerEventNotification;

/* DP IRQ notification information */
typedef struct Lw2080DpIrqNotificationRec {
    LwU32 displayId;
} Lw2080DpIrqNotification;

/* XUSB/PPC D-State change notification information */
typedef struct Lw2080DstateXusbPpcNotificationRec {
    LwU32 dstateXusb;
    LwU32 dstatePpc;
} Lw2080DstateXusbPpcNotification;

/* XUSB/PPC Connection status notification information */
typedef struct Lw2080XusbPpcConnectStateNotificationRec {
    LwBool bConnected;
} Lw2080XusbPpcConnectStateNotification;

/* ACPI event notification information */
typedef struct Lw2080ACPIEvent {
    LwU32 event;
} Lw2080ACPIEvent;

/* Cooler Zone notification information */
typedef struct _LW2080_COOLER_DIAG_ZONE_NOTIFICATION_REC {
    LwU32 lwrrentZone;
} LW2080_COOLER_DIAG_ZONE_NOTIFICATION_REC;

/* Thermal Zone notification information */
typedef struct _LW2080_THERM_DIAG_ZONE_NOTIFICATION_REC {
    LwU32 lwrrentZone;
} LW2080_THERM_DIAG_ZONE_NOTIFICATION_REC;

/* HDCP ref count change notification information */
typedef struct Lw2080AudioHdcpRequestRec {
    LwU32 displayId;
    LwU32 requestedState;
} Lw2080AudioHdcpRequest;

/* Gpu ready event information */
typedef struct Lw2080GC5GpuReadyParams {
    LwU32 event;
    LwU32 sciIntr0;
    LwU32 sciIntr1;
} Lw2080GC5GpuReadyParams;

/* Priv reg access fault notification information */
typedef struct {
    LwU32 errAddr;
} Lw2080PrivRegAccessFaultNotification;

/* HDA D-State change notification information
 * See @HDACODEC_DSTATE for definitions
 */
typedef struct Lw2080DstateHdaCodecNotificationRec {
    LwU32 dstateHdaCodec;
} Lw2080DstateHdaCodecNotification;

/* 
 * Platform Power Mode event information
 */
typedef struct _LW2080_PLATFORM_POWER_MODE_CHANGE_STATUS {
    LwU8 platformPowerModeIndex;
    LwU8 platformPowerModeMask;
    LwU8 eventReason;
} LW2080_PLATFORM_POWER_MODE_CHANGE_STATUS;

#define LW2080_PLATFORM_POWER_MODE_CHANGE_INFO_INDEX                         7:0            
#define LW2080_PLATFORM_POWER_MODE_CHANGE_INFO_MASK                          15:8            
#define LW2080_PLATFORM_POWER_MODE_CHANGE_INFO_REASON                        23:16

/*
 * ENGINE_INFO_TYPE_LW2080 of the engine for which the QOS interrupt has been raised
 */
typedef struct {
    LwU32 engineType;
} Lw2080QosIntrNotification;

typedef struct {
    LwU64 physAddress  LW_ALIGN_BYTES(8);
} Lw2080EccDbeNotification;

/*
 * LPWR DIFR Prefetch Request - Size of L2 Cache
 */
typedef struct {
    LwU32 l2CacheSize;
} Lw2080LpwrDifrPrefetchNotification;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl2080_h_ */
