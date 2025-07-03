// AUTO GENERATED -- DO NOT EDIT

/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2015 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _clc371_h_
#define _clc371_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWC371_DISP_SF_USER (0x000C371)

typedef volatile struct _clc371_tag0 {
    LwU32 dispSfUserOffset[0x400];    /* LW_PDISP_SF_USER   0x000D0FFF:0x000D0000 */
} _LwC371DispSfUser, LwC371DispSfUserMap;

#define LWC371_SF_HDMI_INFO_IDX_AVI_INFOFRAME                             0x00000000 /*       */
#define LWC371_SF_HDMI_INFO_IDX_GENERIC_INFOFRAME                         0x00000001 /*       */
#define LWC371_SF_HDMI_INFO_IDX_ACR                                       0x00000002 /*       */
#define LWC371_SF_HDMI_INFO_IDX_GCP                                       0x00000003 /*       */
#define LWC371_SF_HDMI_INFO_IDX_VSI                                       0x00000004 /*       */
#define LWC371_SF_HDMI_INFO_CTRL(i,j)                 (0x00690000-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_CTRL__SIZE_1                                         4 /*       */
#define LWC371_SF_HDMI_INFO_CTRL__SIZE_2                                         5 /*       */
#define LWC371_SF_HDMI_INFO_CTRL_ENABLE                                        0:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_CTRL_ENABLE_NO                              0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_CTRL_ENABLE_YES                             0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_ENABLE_DIS                             0x00000000 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_ENABLE_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_OTHER                                         4:4 /* RWIVF */
#define LWC371_SF_HDMI_INFO_CTRL_OTHER_DIS                              0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_CTRL_OTHER_EN                               0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_SINGLE                                        8:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_CTRL_SINGLE_DIS                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_CTRL_SINGLE_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_CHKSUM_HW                                     9:9 /* RWIVF */
#define LWC371_SF_HDMI_INFO_CTRL_CHKSUM_HW_ENABLE                       0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_CHKSUM_HW_EN                           0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_CHKSUM_HW_DISABLE                      0x00000000 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_CHKSUM_HW_DIS                          0x00000000 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_CHKSUM_HW_INIT                         0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_INFO_CTRL_HBLANK                                      12:12 /* RWIVF */
#define LWC371_SF_HDMI_INFO_CTRL_HBLANK_DIS                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_CTRL_HBLANK_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_VIDEO_FMT                                   16:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_CTRL_VIDEO_FMT_SW_CONTROLLED                0x00000000 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_VIDEO_FMT_HW_CONTROLLED                0x00000001 /* RW--V */
#define LWC371_SF_HDMI_INFO_CTRL_VIDEO_FMT_INIT                         0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_INFO_STATUS(i,j)               (0x00690004-0x00690000+(i)*1024+(j)*64) /* R--4A */
#define LWC371_SF_HDMI_INFO_STATUS__SIZE_1                                       4 /*       */
#define LWC371_SF_HDMI_INFO_STATUS__SIZE_2                                       5 /*       */
#define LWC371_SF_HDMI_INFO_STATUS_SENT                                        0:0 /* R--VF */
#define LWC371_SF_HDMI_INFO_STATUS_SENT_DONE                            0x00000001 /* R---V */
#define LWC371_SF_HDMI_INFO_STATUS_SENT_WAITING                         0x00000000 /* R---V */
#define LWC371_SF_HDMI_INFO_STATUS_SENT_INIT                            0x00000000 /* R-I-V */
#define LWC371_SF_HDMI_INFO_HEADER(i,j)               (0x00690008-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_HEADER__SIZE_1                                       4 /*       */
#define LWC371_SF_HDMI_INFO_HEADER__SIZE_2                                       5 /*       */
#define LWC371_SF_HDMI_INFO_HEADER_HB0                                         7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_HEADER_HB0_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_HEADER_HB1                                        15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_HEADER_HB1_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_HEADER_HB2                                       23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_HEADER_HB2_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW(i,j)         (0x0069000C-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW__SIZE_2                                 5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB0                                   7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB0_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB1                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB3                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK0_LOW_PB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH(i,j)        (0x00690010-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH__SIZE_2                                5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH_PB4                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH_PB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH_PB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH_PB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH_PB6                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK0_HIGH_PB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW(i,j)         (0x00690014-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW__SIZE_2                                 5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB7                                   7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB7_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB8                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB8_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB9                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB9_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB10                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK1_LOW_PB10_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH(i,j)        (0x00690018-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH__SIZE_2                                5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH_PB11                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH_PB11_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH_PB12                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH_PB12_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH_PB13                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK1_HIGH_PB13_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW(i,j)         (0x0069001C-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW__SIZE_2                                 5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB14                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB14_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB15                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB15_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB16                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB16_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB17                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK2_LOW_PB17_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH(i,j)        (0x00690020-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH__SIZE_2                                5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH_PB18                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH_PB18_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH_PB19                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH_PB19_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH_PB20                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK2_HIGH_PB20_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW(i,j)         (0x00690024-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW__SIZE_2                                 5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB21                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB21_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB22                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB22_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB23                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB23_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB24                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK3_LOW_PB24_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH(i,j)        (0x00690028-0x00690000+(i)*1024+(j)*64) /* RWX4A */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH__SIZE_2                                5 /*       */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH_PB25                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH_PB25_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH_PB26                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH_PB26_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH_PB27                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_INFO_SUBPACK3_HIGH_PB27_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL(i)                    (0x00690000-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL__SIZE_1                                   4 /*       */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_ENABLE                                  0:0 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_ENABLE_NO                        0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_ENABLE_YES                       0x00000001 /* RW--V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_ENABLE_DIS                       0x00000000 /* RW--V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_ENABLE_EN                        0x00000001 /* RW--V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_OTHER                                   4:4 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_OTHER_DIS                        0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_OTHER_EN                         0x00000001 /* RW--V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_SINGLE                                  8:8 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_SINGLE_DIS                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_SINGLE_EN                        0x00000001 /* RW--V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_CHKSUM_HW                               9:9 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_CHKSUM_HW_ENABLE                 0x00000001 /* RW--V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_CHKSUM_HW_DISABLE                0x00000000 /* RW--V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_CTRL_CHKSUM_HW_INIT                   0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_STATUS(i)                  (0x00690004-0x00690000+(i)*1024) /* R--4A */
#define LWC371_SF_HDMI_AVI_INFOFRAME_STATUS__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_AVI_INFOFRAME_STATUS_SENT                                  0:0 /* R--VF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_STATUS_SENT_DONE                      0x00000001 /* R---V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_STATUS_SENT_WAITING                   0x00000000 /* R---V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_STATUS_SENT_INIT                      0x00000000 /* R-I-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER(i)                  (0x00690008-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER_HB0                                   7:0 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER_HB0_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER_HB1                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER_HB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER_HB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_HEADER_HB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW(i)            (0x0069000C-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW__SIZE_1                           4 /*       */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB0                             7:0 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB0_INIT                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB1                            15:8 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB1_INIT                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB2                           23:16 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB2_INIT                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB3                           31:24 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_LOW_PB3_INIT                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH(i)           (0x00690010-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH__SIZE_1                          4 /*       */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH_PB4                            7:0 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH_PB4_INIT                0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH_PB5                           15:8 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH_PB5_INIT                0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH_PB6                          23:16 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK0_HIGH_PB6_INIT                0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW(i)            (0x00690014-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW__SIZE_1                           4 /*       */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB7                             7:0 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB7_INIT                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB8                            15:8 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB8_INIT                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB9                           23:16 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB9_INIT                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB10                          31:24 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_LOW_PB10_INIT                0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH(i)           (0x00690018-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH__SIZE_1                          4 /*       */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH_PB11                           7:0 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH_PB11_INIT               0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH_PB12                          15:8 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH_PB12_INIT               0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH_PB13                         23:16 /* RWIVF */
#define LWC371_SF_HDMI_AVI_INFOFRAME_SUBPACK1_HIGH_PB13_INIT               0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_CTRL(i)                          (0x00690040-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_CTRL__SIZE_1                                         4 /*       */
#define LWC371_SF_HDMI_GENERIC_CTRL_ENABLE                                        0:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_CTRL_ENABLE_NO                              0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_CTRL_ENABLE_YES                             0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GENERIC_CTRL_ENABLE_DIS                             0x00000000 /* RW--V */
#define LWC371_SF_HDMI_GENERIC_CTRL_ENABLE_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GENERIC_CTRL_OTHER                                         4:4 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_CTRL_OTHER_DIS                              0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_CTRL_OTHER_EN                               0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GENERIC_CTRL_SINGLE                                        8:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_CTRL_SINGLE_DIS                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_CTRL_SINGLE_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GENERIC_CTRL_HBLANK                                      12:12 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_CTRL_HBLANK_DIS                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_CTRL_HBLANK_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GENERIC_STATUS(i)                        (0x00690044-0x00690000+(i)*1024) /* R--4A */
#define LWC371_SF_HDMI_GENERIC_STATUS__SIZE_1                                       4 /*       */
#define LWC371_SF_HDMI_GENERIC_STATUS_SENT                                        0:0 /* R--VF */
#define LWC371_SF_HDMI_GENERIC_STATUS_SENT_DONE                            0x00000001 /* R---V */
#define LWC371_SF_HDMI_GENERIC_STATUS_SENT_WAITING                         0x00000000 /* R---V */
#define LWC371_SF_HDMI_GENERIC_STATUS_SENT_INIT                            0x00000000 /* R-I-V */
#define LWC371_SF_HDMI_GENERIC_HEADER(i)                        (0x00690048-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_HEADER__SIZE_1                                       4 /*       */
#define LWC371_SF_HDMI_GENERIC_HEADER_HB0                                         7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_HEADER_HB0_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_HEADER_HB1                                        15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_HEADER_HB1_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_HEADER_HB2                                       23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_HEADER_HB2_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW(i)                  (0x0069004C-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB0                                   7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB0_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB1                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB3                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_LOW_PB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH(i)                 (0x00690050-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH_PB4                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH_PB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH_PB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH_PB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH_PB6                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK0_HIGH_PB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW(i)                  (0x00690054-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB7                                   7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB7_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB8                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB8_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB9                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB9_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB10                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_LOW_PB10_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH(i)                 (0x00690058-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH_PB11                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH_PB11_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH_PB12                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH_PB12_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH_PB13                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK1_HIGH_PB13_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW(i)                  (0x0069005C-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB14                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB14_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB15                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB15_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB16                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB16_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB17                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_LOW_PB17_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH(i)                 (0x00690060-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH_PB18                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH_PB18_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH_PB19                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH_PB19_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH_PB20                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK2_HIGH_PB20_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW(i)                  (0x00690064-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB21                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB21_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB22                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB22_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB23                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB23_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB24                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_LOW_PB24_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH(i)                 (0x00690068-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH_PB25                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH_PB25_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH_PB26                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH_PB26_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH_PB27                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_GENERIC_SUBPACK3_HIGH_PB27_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_CTRL(i)                              (0x00690080-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_CTRL__SIZE_1                                             4 /*       */
#define LWC371_SF_HDMI_ACR_CTRL_PACKET_ENABLE                                     0:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_CTRL_PACKET_ENABLE_NO                           0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_CTRL_PACKET_ENABLE_YES                          0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_PACKET_ENABLE_DIS                          0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_PACKET_ENABLE_EN                           0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_ENABLE                                    16:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_ENABLE_NO                            0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_ENABLE_YES                           0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_ENABLE_DIS                           0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_ENABLE_EN                            0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_PRIORITY                                        20:20 /* RWIVF */
#define LWC371_SF_HDMI_ACR_CTRL_PRIORITY_INIT                              0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_CTRL_PRIORITY_HIGH                              0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_PRIORITY_LOW                               0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS                                           27:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_INIT                                 0x00000002 /* RWI-V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_32KHZ                                0x00000003 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_44_1KHZ                              0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_48KHZ                                0x00000002 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_88_2KHZ                              0x00000008 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_96KHZ                                0x0000000A /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_176_4KHZ                             0x0000000C /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_FREQS_192KHZ                               0x0000000E /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_CTS_SOURCE                                      31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_CTRL_CTS_SOURCE_INIT                            0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_ACR_CTRL_CTS_SOURCE_HW                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_CTRL_CTS_SOURCE_SW                              0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_SUBPACK_IDX_32KHZ                               0x00000000 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_IDX_44_1KHZ                             0x00000001 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_IDX_88_2KHZ                             0x00000002 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_IDX_176_4KHZ                            0x00000003 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_IDX_48KHZ                               0x00000004 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_IDX_96KHZ                               0x00000005 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_IDX_192KHZ                              0x00000006 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW(i,j)               (0x00690088-0x00690000+(i)*1024+(j)*8) /* RWX4A */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW__SIZE_1                                      4 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW__SIZE_2                                      7 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW_SB1                                      31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW_SB1_INIT                            0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW_SB2                                      23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW_SB2_INIT                            0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW_SB3                                       15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW_SB3_INIT                            0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_SUBPACK_LOW_CTS                                       31:8 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH(i,j)              (0x0069008c-0x00690000+(i)*1024+(j)*8) /* RWX4A */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH__SIZE_1                                     4 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH__SIZE_2                                     7 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_SB4                                     23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_SB4_INIT                           0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_SB5                                      15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_SB5_INIT                           0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_SB6                                       7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_SB6_INIT                           0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_N                                        23:0 /*       */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_ENABLE                                  31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_ENABLE_NO                          0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_ENABLE_YES                         0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_ENABLE_DIS                         0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_SUBPACK_HIGH_ENABLE_EN                          0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW(i)                  (0x00690088-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW_SB1                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW_SB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW_SB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW_SB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW_SB3                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW_SB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_LOW_CTS                                  31:8 /*       */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH(i)                 (0x0069008c-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_SB6                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_SB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_SB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_SB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_SB4                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_SB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_N                                   23:0 /*       */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_ENABLE                             31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_ENABLE_NO                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_ENABLE_YES                    0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_ENABLE_DIS                    0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_0320_SUBPACK_HIGH_ENABLE_EN                     0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW(i)                  (0x00690090-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW_SB3                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW_SB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW_SB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW_SB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW_SB1                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW_SB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_LOW_CTS                                  31:8 /*       */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH(i)                 (0x00690094-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_SB6                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_SB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_SB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_SB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_SB4                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_SB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_ENABLE                             31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_ENABLE_NO                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_ENABLE_YES                    0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_ENABLE_DIS                    0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_0441_SUBPACK_HIGH_ENABLE_EN                     0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW(i)                  (0x00690098-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW_SB3                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW_SB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW_SB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW_SB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW_SB1                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW_SB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_LOW_CTS                                  31:8 /*       */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH(i)                 (0x0069009C-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_SB6                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_SB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_SB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_SB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_SB4                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_SB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_N                                   23:0 /*       */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_ENABLE                             31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_ENABLE_NO                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_ENABLE_YES                    0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_ENABLE_DIS                    0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_0882_SUBPACK_HIGH_ENABLE_EN                     0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW(i)                  (0x006900A0-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW_SB3                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW_SB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW_SB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW_SB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW_SB1                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW_SB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_LOW_CTS                                  31:8 /*       */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH(i)                 (0x006900A4-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_SB6                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_SB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_SB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_SB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_SB4                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_SB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_N                                   23:0 /*       */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_ENABLE                             31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_ENABLE_NO                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_ENABLE_YES                    0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_ENABLE_DIS                    0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_1764_SUBPACK_HIGH_ENABLE_EN                     0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW(i)                  (0x006900A8-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW_SB3                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW_SB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW_SB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW_SB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW_SB1                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW_SB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_LOW_CTS                                  31:8 /*       */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH(i)                 (0x006900AC-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_SB6                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_SB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_SB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_SB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_SB4                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_SB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_N                                   23:0 /*       */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_ENABLE                             31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_ENABLE_NO                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_ENABLE_YES                    0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_ENABLE_DIS                    0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_0480_SUBPACK_HIGH_ENABLE_EN                     0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW(i)                  (0x006900B0-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW_SB3                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW_SB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW_SB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW_SB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW_SB1                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW_SB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_LOW_CTS                                  31:8 /*       */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH(i)                 (0x006900B4-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_SB6                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_SB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_SB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_SB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_SB4                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_SB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_N                                   23:0 /*       */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_ENABLE                             31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_ENABLE_NO                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_ENABLE_YES                    0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_ENABLE_DIS                    0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_0960_SUBPACK_HIGH_ENABLE_EN                     0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW(i)                  (0x006900B8-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW_SB3                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW_SB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW_SB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW_SB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW_SB1                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW_SB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_LOW_CTS                                  31:8 /*       */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH(i)                 (0x006900BC-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_SB6                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_SB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_SB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_SB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_SB4                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_SB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_N                                   23:0 /*       */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_ENABLE                             31:31 /* RWIVF */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_ENABLE_NO                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_ENABLE_YES                    0x00000001 /* RW--V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_ENABLE_DIS                    0x00000000 /* RW--V */
#define LWC371_SF_HDMI_ACR_1920_SUBPACK_HIGH_ENABLE_EN                     0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GCP_CTRL(i)                              (0x006900C0-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GCP_CTRL__SIZE_1                                             4 /*       */
#define LWC371_SF_HDMI_GCP_CTRL_ENABLE                                            0:0 /* RWIVF */
#define LWC371_SF_HDMI_GCP_CTRL_ENABLE_NO                                  0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GCP_CTRL_ENABLE_YES                                 0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GCP_CTRL_ENABLE_DIS                                 0x00000000 /* RW--V */
#define LWC371_SF_HDMI_GCP_CTRL_ENABLE_EN                                  0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GCP_CTRL_OTHER                                             4:4 /* RWIVF */
#define LWC371_SF_HDMI_GCP_CTRL_OTHER_DIS                                  0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GCP_CTRL_OTHER_EN                                   0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GCP_CTRL_SINGLE                                            8:8 /* RWIVF */
#define LWC371_SF_HDMI_GCP_CTRL_SINGLE_DIS                                 0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GCP_CTRL_SINGLE_EN                                  0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GCP_STATUS(i)                            (0x006900C4-0x00690000+(i)*1024) /* R--4A */
#define LWC371_SF_HDMI_GCP_STATUS__SIZE_1                                           4 /*       */
#define LWC371_SF_HDMI_GCP_STATUS_SENT                                            0:0 /* R--VF */
#define LWC371_SF_HDMI_GCP_STATUS_SENT_DONE                                0x00000001 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_SENT_WAITING                             0x00000000 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_SENT_INIT                                0x00000000 /* R-I-V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_START_PP                                 6:4 /* R--VF */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_START_PP_0                        0x00000004 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_START_PP_1                        0x00000001 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_START_PP_2                        0x00000002 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_START_PP_3                        0x00000003 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_END_PP                                  10:8 /* R--VF */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_END_PP_0                          0x00000004 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_END_PP_1                          0x00000001 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_END_PP_2                          0x00000002 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_ACTIVE_END_PP_3                          0x00000003 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_START_PP                                14:12 /* R--VF */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_START_PP_0                         0x00000004 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_START_PP_1                         0x00000001 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_START_PP_2                         0x00000002 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_START_PP_3                         0x00000003 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_END_PP                                  18:16 /* R--VF */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_END_PP_0                           0x00000004 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_END_PP_1                           0x00000001 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_END_PP_2                           0x00000002 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_VSYNC_END_PP_3                           0x00000003 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_START_PP                                22:20 /* R--VF */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_START_PP_0                         0x00000004 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_START_PP_1                         0x00000001 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_START_PP_2                         0x00000002 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_START_PP_3                         0x00000003 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_END_PP                                  26:24 /* R--VF */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_END_PP_0                           0x00000004 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_END_PP_1                           0x00000001 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_END_PP_2                           0x00000002 /* R---V */
#define LWC371_SF_HDMI_GCP_STATUS_HSYNC_END_PP_3                           0x00000003 /* R---V */
#define LWC371_SF_HDMI_GCP_SUBPACK(i)                           (0x006900CC-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_GCP_SUBPACK__SIZE_1                                          4 /*       */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB0                                            7:0 /* RWIVF */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB0_INIT                                0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB0_SET_AVMUTE                          0x00000001 /* RW--V */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB0_CLR_AVMUTE                          0x00000010 /* RW--V */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB1                                           15:8 /* RWIVF */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB1_INIT                                0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB2                                          23:16 /* RWIVF */
#define LWC371_SF_HDMI_GCP_SUBPACK_SB2_INIT                                0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_CTRL(i)                          (0x00690100-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_CTRL__SIZE_1                                         4 /*       */
#define LWC371_SF_HDMI_VSI_CTRL_ENABLE                                        0:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_CTRL_ENABLE_NO                              0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_CTRL_ENABLE_YES                             0x00000001 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_ENABLE_DIS                             0x00000000 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_ENABLE_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_OTHER                                         4:4 /* RWIVF */
#define LWC371_SF_HDMI_VSI_CTRL_OTHER_DIS                              0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_CTRL_OTHER_EN                               0x00000001 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_SINGLE                                        8:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_CTRL_SINGLE_DIS                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_CTRL_SINGLE_EN                              0x00000001 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_CHKSUM_HW                                     9:9 /* RWIVF */
#define LWC371_SF_HDMI_VSI_CTRL_CHKSUM_HW_ENABLE                       0x00000001 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_CHKSUM_HW_EN                           0x00000001 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_CHKSUM_HW_DISABLE                      0x00000000 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_CHKSUM_HW_DIS                          0x00000000 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_CHKSUM_HW_INIT                         0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_VSI_CTRL_VIDEO_FMT                                   16:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_CTRL_VIDEO_FMT_SW_CONTROLLED                0x00000000 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_VIDEO_FMT_HW_CONTROLLED                0x00000001 /* RW--V */
#define LWC371_SF_HDMI_VSI_CTRL_VIDEO_FMT_INIT                         0x00000001 /* RWI-V */
#define LWC371_SF_HDMI_VSI_STATUS(i)                        (0x00690104-0x00690000+(i)*1024) /* R--4A */
#define LWC371_SF_HDMI_VSI_STATUS__SIZE_1                                       4 /*       */
#define LWC371_SF_HDMI_VSI_STATUS_SENT                                        0:0 /* R--VF */
#define LWC371_SF_HDMI_VSI_STATUS_SENT_DONE                            0x00000001 /* R---V */
#define LWC371_SF_HDMI_VSI_STATUS_SENT_WAITING                         0x00000000 /* R---V */
#define LWC371_SF_HDMI_VSI_STATUS_SENT_INIT                            0x00000000 /* R-I-V */
#define LWC371_SF_HDMI_VSI_HEADER(i)                        (0x00690108-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_HEADER__SIZE_1                                       4 /*       */
#define LWC371_SF_HDMI_VSI_HEADER_HB0                                         7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_HEADER_HB0_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_HEADER_HB1                                        15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_HEADER_HB1_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_HEADER_HB2                                       23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_HEADER_HB2_INIT                             0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW(i)                  (0x0069010C-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB0                                   7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB0_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB1                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB2                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB3                                 31:24 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK0_LOW_PB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH(i)                 (0x00690110-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH_PB4                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH_PB4_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH_PB5                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH_PB5_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH_PB6                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK0_HIGH_PB6_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW(i)                  (0x00690114-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB7                                   7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB7_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB8                                  15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB8_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB9                                 23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB9_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB10                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK1_LOW_PB10_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH(i)                 (0x00690118-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH_PB11                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH_PB11_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH_PB12                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH_PB12_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH_PB13                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK1_HIGH_PB13_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW(i)                  (0x0069011C-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB14                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB14_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB15                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB15_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB16                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB16_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB17                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK2_LOW_PB17_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH(i)                 (0x00690120-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH_PB18                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH_PB18_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH_PB19                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH_PB19_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH_PB20                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK2_HIGH_PB20_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW(i)                  (0x00690124-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW__SIZE_1                                 4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB21                                  7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB21_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB22                                 15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB22_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB23                                23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB23_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB24                                31:24 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK3_LOW_PB24_INIT                      0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH(i)                 (0x00690128-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH__SIZE_1                                4 /*       */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH_PB25                                 7:0 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH_PB25_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH_PB26                                15:8 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH_PB26_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH_PB27                               23:16 /* RWIVF */
#define LWC371_SF_HDMI_VSI_SUBPACK3_HIGH_PB27_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL(i)                  (0x00690300-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL__SIZE_1                                 4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_ENABLE                                0:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_ENABLE_YES                     0x00000001 /* RW--V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_ENABLE_NO                      0x00000000 /* RW--V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_ENABLE_INIT                    0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_IMMEDIATE                             1:1 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_IMMEDIATE_INIT                 0x00000000 /* R-I-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_IMMEDIATE_DONE                 0x00000000 /* R---V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_IMMEDIATE_PENDING              0x00000001 /* R---V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_IMMEDIATE_TRIGGER              0x00000001 /* -W--V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_MSA_STEREO_OVERRIDE                   2:2 /* RWIVF */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_MSA_STEREO_OVERRIDE_YES        0x00000001 /* RW--V */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_MSA_STEREO_OVERRIDE_NO         0x00000000 /* RW--V */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME_CTRL_MSA_STEREO_OVERRIDE_INIT       0x00000000 /* RWI-V */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER(i)                (0x00690304-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER__SIZE_1                               4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB0                                 7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB0_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB1                                15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB1_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB2                               23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB2_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB3                               31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_HEADER_HB3_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0(i)              (0x00690308-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB0                               7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB0_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB1                              15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB1_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB2                             23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB2_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB3                             31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK0_DB3_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1(i)              (0x0069030c-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB4                               7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB4_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB5                              15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB5_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB6                             23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB6_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB7                             31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK1_DB7_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2(i)              (0x00690310-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB8                               7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB8_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB9                              15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB9_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB10                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB10_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB11                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK2_DB11_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3(i)              (0x00690314-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB12                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB12_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB13                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB13_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB14                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB14_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB15                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK3_DB15_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4(i)              (0x00690318-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB16                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB16_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB17                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB17_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB18                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB18_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB19                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK4_DB19_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5(i)              (0x0069031c-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB20                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB20_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB21                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB21_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB22                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB22_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB23                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK5_DB23_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6(i)              (0x00690320-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB24                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB24_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB25                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB25_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB26                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB26_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB27                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK6_DB27_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7(i)              (0x00690324-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB28                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB28_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB29                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB29_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB30                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB30_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB31                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME_SUBPACK7_DB31_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_AUDIO_INFOFRAME_CTRL(i)                     (0x00690330-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_AUDIO_INFOFRAME_CTRL__SIZE_1                                    4 /*       */
#define LWC371_SF_DP_AUDIO_INFOFRAME_CTRL_HEADER_OVERRIDE                          4:4 /* RWIVF */
#define LWC371_SF_DP_AUDIO_INFOFRAME_CTRL_HEADER_OVERRIDE_ENABLE            0x00000001 /* RW--V */
#define LWC371_SF_DP_AUDIO_INFOFRAME_CTRL_HEADER_OVERRIDE_DISABLE           0x00000000 /* RW--V */
#define LWC371_SF_DP_AUDIO_INFOFRAME_CTRL_HEADER_OVERRIDE_INIT              0x00000000 /* RWI-V */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER(i)                  (0x00690334-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER__SIZE_1                                 4 /*       */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB0                                   7:0 /* RWIVF */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB0_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB1                                  15:8 /* RWIVF */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB1_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB2                                 23:16 /* RWIVF */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB2_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB3                                 31:24 /* RWIVF */
#define LWC371_SF_DP_AUDIO_INFOFRAME_HEADER_HB3_INIT                       0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL(i)                  (0x00690340-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL__SIZE_1                                 4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_ENABLE                                0:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_ENABLE_YES                     0x00000001 /* RW--V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_ENABLE_NO                      0x00000000 /* RW--V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_ENABLE_INIT                    0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_IMMEDIATE                             1:1 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_IMMEDIATE_INIT                 0x00000000 /* R-I-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_IMMEDIATE_DONE                 0x00000000 /* R---V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_IMMEDIATE_PENDING              0x00000001 /* R---V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_IMMEDIATE_TRIGGER              0x00000001 /* -W--V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_MSA_STEREO_OVERRIDE                   2:2 /* RWIVF */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_MSA_STEREO_OVERRIDE_YES        0x00000001 /* RW--V */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_MSA_STEREO_OVERRIDE_NO         0x00000000 /* RW--V */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME1_CTRL_MSA_STEREO_OVERRIDE_INIT       0x00000000 /* RWI-V */ 
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER(i)                (0x00690344-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER__SIZE_1                               4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB0                                 7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB0_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB1                                15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB1_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB2                               23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB2_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB3                               31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_HEADER_HB3_INIT                     0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0(i)              (0x00690348-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB0                               7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB0_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB1                              15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB1_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB2                             23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB2_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB3                             31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK0_DB3_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1(i)              (0x0069034c-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB4                               7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB4_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB5                              15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB5_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB6                             23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB6_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB7                             31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK1_DB7_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2(i)              (0x00690350-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB8                               7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB8_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB9                              15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB9_INIT                   0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB10                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB10_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB11                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK2_DB11_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3(i)              (0x00690354-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB12                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB12_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB13                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB13_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB14                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB14_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB15                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK3_DB15_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4(i)              (0x00690358-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB16                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB16_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB17                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB17_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB18                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB18_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB19                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK4_DB19_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5(i)              (0x0069035c-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB20                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB20_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB21                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB21_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB22                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB22_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB23                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK5_DB23_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6(i)              (0x00690360-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB24                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB24_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB25                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB25_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB26                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB26_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB27                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK6_DB27_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7(i)              (0x00690364-0x00690000+(i)*1024) /* RWX4A */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7__SIZE_1                             4 /*       */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB28                              7:0 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB28_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB29                             15:8 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB29_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB30                            23:16 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB30_INIT                  0x00000000 /* RWI-V */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB31                            31:24 /* RWIVF */
#define LWC371_SF_DP_GENERIC_INFOFRAME1_SUBPACK7_DB31_INIT                  0x00000000 /* RWI-V */

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif // _clc371_h_
