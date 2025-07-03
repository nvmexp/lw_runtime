 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: ocarc.h                                                           *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OCARC_H
#define _OCARC_H

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************

// Define the RC error codes (Taken from lwerror.h)
#define RC_FIFO_ERROR_FIFO_METHOD               (1)
#define RC_FIFO_ERROR_SW_METHOD                 (2)
#define RC_FIFO_ERROR_UNK_METHOD                (3)
#define RC_FIFO_ERROR_CHANNEL_BUSY              (4)
#define RC_FIFO_ERROR_RUNOUT_OVERFLOW           (5)
#define RC_FIFO_ERROR_PARSE_ERR                 (6)
#define RC_FIFO_ERROR_PTE_ERR                   (7)
#define RC_FIFO_ERROR_IDLE_TIMEOUT              (8)
#define RC_GR_ERROR_INSTANCE                    (9)
#define RC_GR_ERROR_SINGLE_STEP                 (10)
#define RC_GR_ERROR_MISSING_HW                  (11)
#define RC_GR_ERROR_SW_METHOD                   (12)
#define RC_GR_ERROR_SW_NOTIFY                   (13)
#define RC_FAKE_ERROR                           (14)
#define RC_SCANLINE_TIMEOUT                     (15)
#define RC_VBLANK_CALLBACK_TIMEOUT              (16)
#define RC_PARAMETER_ERROR                      (17)
#define RC_BUS_MASTER_TIMEOUT_ERROR             (18)
#define RC_DISP_MISSED_NOTIFIER                 (19)
#define RC_MPEG_ERROR_SW_METHOD                 (20)
#define RC_ME_ERROR_SW_METHOD                   (21)
#define RC_VP_ERROR_SW_METHOD                   (22)
#define RC_RC_LOGGING_ENABLED                   (23)
#define RC_GR_SEMAPHORE_TIMEOUT                 (24)
#define RC_GR_ILLEGAL_NOTIFY                    (25)
#define RC_FIFO_ERROR_FBISTATE_TIMEOUT          (26)
#define RC_VP_ERROR                             (27)
#define RC_VP2_ERROR                            (28)
#define RC_BSP_ERROR                            (29)
#define RC_BAD_ADDR_ACCESS                      (30)
#define RC_FIFO_ERROR_MMU_ERR_FLT               (31)
#define RC_PBDMA_ERROR                          (32)
#define RC_SEC_ERROR                            (33)
#define RC_MSVLD_ERROR                          (34)
#define RC_MSPDEC_ERROR                         (35)
#define RC_MSPPP_ERROR                          (36)
#define RC_FECS_ERR_UNIMP_FIRMWARE_METHOD       (37)
#define RC_FECS_ERR_WATCHDOG_TIMEOUT            (38)
#define RC_CE0_ERROR                            (39)
#define RC_CE1_ERROR                            (40)
#define RC_CE2_ERROR                            (41)
#define RC_VIC_ERROR                            (42)
#define RC_RESETCHANNEL_VERIF_ERROR             (43)
#define RC_GR_FAULT_DURING_CTXSW                (44)
#define RC_PREEMPTIVE_REMOVAL                   (45)
#define RC_GPU_TIMEOUT_ERROR                    (46)
#define RC_LWENC0_ERROR                         (47)
#define RC_GPU_ECC_DBE                          (48)
#define RC_SR_CONSTANT_LEVEL_SET_BY_REGISTRY    (49)
#define RC_SR_LEVEL_TRANSITION_DUE_TO_RC_ERROR  (50)
#define RC_SR_STRESS_TEST_FAILURE               (51)
#define RC_SR_LEVEL_TRANS_DUE_TO_TEMP_RISE      (52)
#define RC_SR_TEMP_REDUCED_CLOCKING             (53)
#define RC_SR_PWR_REDUCED_CLOCKING              (54)
#define RC_SR_TEMPERATURE_READ_ERROR            (55)
#define RC_DISPLAY_CHANNEL_EXCEPTION            (56)
#define RC_FB_LINK_TRAINING_FAILURE_ERROR       (57)
#define RC_FB_MEMORY_ERROR                      (58)
#define RC_PMU_ERROR                            (59)
#define RC_SEC2_ERROR                           (60)
#define RC_PMU_BREAKPOINT                       (61)
#define RC_PMU_HALT_ERROR                       (62)
#define RC_INFOROM_PAGE_RETIREMENT_EVENT        (63)
#define RC_INFOROM_PAGE_RETIREMENT_FAILURE      (64)
#define RC_LWENC1_ERROR                         (65)
#define RC_FECS_ERR_REG_ACCESS_VIOLATION        (66)
#define RC_FECS_ERR_VERIF_VIOLATION             (67)
#define RC_LWDEC_ERROR                          (68)
#define RC_GR_CLASS_ERROR                       (69)
#define RC_CE3_ERROR                            (70)
#define RC_CE4_ERROR                            (71)
#define RC_CE5_ERROR                            (72)
#define RC_LWENC2_ERROR                         (73)
#define RC_LWLINK_LINK_DISABLED                 (74)
#define RC_CE6_ERROR                            (75)
#define RC_CE7_ERROR                            (76)
#define RC_CE8_ERROR                            (77)
#define RC_VGPU_START_ERROR                     (78)
#define RC_GPU_HAS_FALLEN_OFF_THE_BUS           (79)
#define RC_PBDMA_PUSHBUFFER_CRC_MISMATCH        (80)

// Define the RC level codes (Taken from lwerror.h)
#define RC_LEVEL_INFO                           (0)
#define RC_LEVEL_NON_FATAL                      (1)
#define RC_LEVEL_FATAL                          (2)

// Define the RC engine values (Taken from lwerror.h)
#define RC_GRAPHICS_ENGINE                      0x00000001
#define RC_SEC_ENGINE                           0x00000002
#define RC_COPY0_ENGINE                         0x00000004
#define RC_COPY1_ENGINE                         0x00000008
#define RC_COPY2_ENGINE                         0x00000010
#define RC_MSPDEC_ENGINE                        0x00000020
#define RC_MSPPP_ENGINE                         0x00000040
#define RC_MSVLD_ENGINE                         0x00000080
#define RC_HOST_ENGINE                          0x00000100
#define RC_DISPLAY_ENGINE                       0x00000200
#define RC_CAPTURE_ENGINE                       0x00000400
#define RC_PERF_MON_ENGINE                      0x00000800
#define RC_PMU_ENGINE                           0x00001000
#define RC_LWENC0_ENGINE                        0x00002000
#define RC_SEC2_ENGINE                          0x00004000
#define RC_LWDEC_ENGINE                         0x00008000
#define RC_LWENC1_ENGINE                        0x00010000
#define RC_COPY3_ENGINE                         0x00020000
#define RC_COPY4_ENGINE                         0x00040000
#define RC_COPY5_ENGINE                         0x00080000
#define RC_LWENC2_ENGINE                        0x00100000
#define RC_COPY6_ENGINE                         0x00200000
#define RC_COPY7_ENGINE                         0x00400000
#define RC_COPY8_ENGINE                         0x00800000

//******************************************************************************
//
//  Structures
//
//******************************************************************************
typedef struct _ENGINE_DATA
{
    ULONG               ulEngineId;
    const char         *pEngineString;

} ENGINE_DATA, *PENGINE_DATA;

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  CString                     rcType(ULONG ulRcType);
extern  CString                     rcLevel(ULONG ulRcLevel);
extern  CString                     rcEngine(ULONG ulRcEngine);

extern  ULONG                       ocaRcErrorCount(ULONG64 ulAdapter);

extern  CErrorInfoData**            findRcErrorRecords(ULONG64 ulAdapter);
extern  void                        displayOcaRcErrors(ULONG ulRcErrorCount, CErrorInfoData** pRcArray);

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OCARC_H
