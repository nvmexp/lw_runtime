/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2005 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl7476os_h_
#define _cl7476os_h_

#ifdef __cplusplus
extern "C" {
#endif


/*

    DO NOT EDIT - THIS FILE WAS GENERATED FROM usr_vp2_os.ref

*/



/* Tile modes from class manual usr_vp2_os.ref */
#define LW7476_SURFACE_TILE_MODE_LINEAR_1D                        (0)
#define LW7476_SURFACE_TILE_MODE_DXVA                             (1)
#define LW7476_SURFACE_TILE_MODE_PITCH_LINEAR                     (3)
#define LW7476_SURFACE_TILE_MODE_RESERVED                         (4)
#define LW7476_SURFACE_TILE_MODE_BL_T                             (5)
#define LW7476_SURFACE_TILE_MODE_BL_N                             (6)
#define LW7476_SURFACE_TILE_MODE_16x16_VP2                        (7)


/* Error/interrupt codes and mailbox definitions from class manual usr_vp2_os.ref */
#define LW7476_ERROR_NONE                                         (0x00000000)
#define LW7476_ERROR_EXELWTE_INSUFFICIENT_DATA                    (0x00000001)
#define LW7476_ERROR_SEMAPHORE_INSUFFICIENT_DATA                  (0x00000002)
#define LW7476_ERROR_ILWALID_METHOD                               (0x00000003)
#define LW7476_ERROR_ILWALID_DMA_PAGE                             (0x00000004)
#define LW7476_ERROR_EXELWTE                                      (0x00000005)
#define LW7476_MAILBOX_0_EXELWTE_STATUS                           31:0
#define LW7476_ERROR_MICROCODE_B_ALIGN_256                        (0x00000006)
#define LW7476_ERROR_EXELWTE_OFFSET_B_ALIGN_256                   (0x00000007)
#define LW7476_ERROR_DATA_SEGMENT_SIZE_MULTIPLE_256               (0x00000008)
#define LW7476_ERROR_EXELWTE_INADEQUATE_DATA_SEGMENT_SIZE         (0x00000009)
#define LW7476_ERROR_EXELWTE_ILWALID_MAGIC_KEY                    (0x0000000A)
#define LW7476_ERROR_VECTOR                                       (0x00000010)
#define LW7476_ERROR_MACRO_PLAYER                                 (0x00000020)
#define LW7476_INTERRUPT_EXELWTE_AWAKEN                           (0x00000100)
#define LW7476_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                 (0x00000200)
#define LW7476_MAILBOX_4_APP_ID                                   31:24
#define LW7476_MAILBOX_4_APP_ID_OS                                ((0x00000010))
#define LW7476_MAILBOX_4_APP_STATUS                               23:0
#define LW7476_MAILBOX_4_APP_STATUS_START                         (1)
#define LW7476_MAILBOX_4_APP_STATUS_FINISH                        (2)


#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl7476os_h_ */
